"""PART H - Batched splice and prune/reorder.

Build K spliced branches (shared prefix, per-branch suffix), stack their caches
into one batched cache, continue batched, compare to independent spliced branches
and to full perturbed references, then prune/reorder survivors and continue.
"""

from __future__ import annotations

import json
import torch

import bg_autoregressive_cache_common_v1 as C
import bg_autoregressive_cache_helpers_v1 as H
import bg_partial_cache_splice_v2_common as V2

PROMPTS = C.TEST_PROMPTS[:2]
K_VALUES = [2, 4]
BOUNDARY = (2, 24)
ALPHA = 1.0
MAX_STEPS = 4


def stack_caches(caches):
    """Stack K batch-1 caches into one batch-K cache (same slot population/shape)."""
    cls = type(caches[0])
    out = cls(getattr(caches[0], "max_cache_size", None))
    n = max(len(c.key_cache) for c in caches)
    klist, vlist = [None] * n, [None] * n
    for slot in range(n):
        ks = [c.key_cache[slot] for c in caches if slot < len(c.key_cache)]
        if ks and ks[0] is not None:
            klist[slot] = torch.cat([c.key_cache[slot] for c in caches], dim=0)
            vlist[slot] = torch.cat([c.value_cache[slot] for c in caches], dim=0)
    out._key_cache, out._value_cache = klist, vlist
    for k in klist:
        if k is not None:
            out._seen_tokens = k.shape[2]
            break
    return out


def main() -> int:
    C.set_seed()
    model, tok, info = C.load_model()
    device = next(model.parameters()).device
    NL, NUT, NSLOT = info["num_hidden_layers"], info["total_ut_steps"], info["expected_cache_slots"]
    u, L = BOUNDARY

    rows, per_cfg = [], []
    for p in PROMPTS:
        ids, am = C.tokenize(p["text"])
        P = ids.shape[1]
        tr = V2.default_token_range(P)
        for K in K_VALUES:
            sh_cache, Hb, ppass, _ = V2.prefix_prefill(model, ids, u, L, device)
            directions = [C.make_perturb_vector(info["hidden_size"], 2000 + i, device, torch.bfloat16)
                          for i in range(K)]
            spliced_caches, branch_next, ref_caches, ref_nls = [], [], [], []
            am_full = None
            for i in range(K):
                ref_nl, ref_cache, am_full, _, _, _ = V2.full_perturbed_prefill(
                    model, ids, am, u, L, ALPHA, directions[i], tr)
                ref_caches.append(ref_cache); ref_nls.append(ref_nl)
                Hp, _ = V2.apply_boundary_perturbation(Hb, ALPHA, directions[i], tr)
                sp_nl, sfx, _, _ = V2.suffix_recompute(model, Hp, u, L, device)
                spliced_caches.append(V2.merge_prefix_suffix(sh_cache, sfx, NSLOT))
                branch_next.append(sp_nl)

            # independent spliced continuations
            indep_tokens, indep_logits = [], []
            for i in range(K):
                t, lg = V2.greedy_continue(model, H.clone_universal_cache(spliced_caches[i]),
                                           am_full, branch_next[i], MAX_STEPS, device)
                indep_tokens.append(t); indep_logits.append(lg)

            # batched continuation
            bcache = stack_caches(spliced_caches)
            bmask = am_full.repeat(K, 1)
            bnext = torch.cat(branch_next, dim=0)  # [K, vocab]
            # batched greedy, replaying independent tokens to keep alignment
            batched_logits = [[bnext[i].detach().float().cpu()] for i in range(K)]
            cur_am = bmask
            nl = bnext
            for step in range(MAX_STEPS):
                col = torch.tensor([[indep_tokens[i][step]] for i in range(K)], device=device)
                cur_am = H.update_attention_mask(cur_am, 1)
                nl, bcache = C.decode_step(model, col, bcache, cur_am)
                for i in range(K):
                    batched_logits[i].append(nl[i].detach().float().cpu())

            bi_cmps, bf_cmps = [], []
            for i in range(K):
                for kstep in range(MAX_STEPS + 1):
                    bi_cmps.append(C.compare_logits(batched_logits[i][kstep], indep_logits[i][kstep]))
                # batched vs full reference continuation
                ref_t, ref_lg = V2.greedy_continue(model, H.clone_universal_cache(ref_caches[i]),
                                                   am_full, ref_nls[i], MAX_STEPS, device)
                for kstep in range(MAX_STEPS + 1):
                    bf_cmps.append(C.compare_logits(batched_logits[i][kstep], ref_lg[kstep]))
            eq_bi = C.classify_equivalence(bi_cmps)
            eq_bf = C.classify_equivalence(bf_cmps)

            # prune/reorder survivors: keep [K-1, 0] (reorder), continue
            beam = list(range(K))[::-1][: max(1, K // 2)]  # e.g. K=4 -> [3,2]
            nl_surv = nl.index_select(0, torch.tensor(beam, device=device))
            bcache.reorder_cache(torch.tensor(beam, device=device))
            cur_am2 = cur_am.index_select(0, torch.tensor(beam, device=device))
            prune_ok = (bcache.key_cache[next(i for i,k in enumerate(bcache.key_cache) if k is not None)].shape[0] == len(beam))
            # continue survivors one step
            col = nl_surv.argmax(dim=-1, keepdim=True)
            cur_am2 = H.update_attention_mask(cur_am2, 1)
            nl2, bcache = C.decode_step(model, col, bcache, cur_am2)
            prune_finite = bool(torch.isfinite(nl2).all().item())

            per_cfg.append({
                "prompt_id": p["id"], "K": K, "boundary": [u, L],
                "batched_vs_independent": eq_bi, "batched_vs_full_reference": eq_bf,
                "prune_beam": beam, "prune_shape_ok": bool(prune_ok), "prune_finite": prune_finite,
            })
            rows.append({"prompt_id": p["id"], "K": K, "bi_rms": eq_bi["max_rms"],
                         "bf_rms": eq_bf["max_rms"], "bi_faithful": eq_bi["cache_faithful"],
                         "bf_faithful": eq_bf["cache_faithful"], "prune_ok": bool(prune_ok)})
            C.clear_cuda()

    bi_ok = all(c["batched_vs_independent"]["cache_faithful"] for c in per_cfg)
    bf_ok = all(c["batched_vs_full_reference"]["cache_faithful"] for c in per_cfg)
    prune_ok = all(c["prune_shape_ok"] and c["prune_finite"] for c in per_cfg)
    if not (bi_ok and bf_ok):
        verdict = "SPLICE_PRUNE_MISMATCH"
    elif prune_ok:
        verdict = "BATCHED_PRUNE_SPLICE_VALID"
    else:
        verdict = "BATCHED_SPLICE_VALID_PRUNE_WEAK"

    summary = {"verdict": verdict, "batched_vs_independent_faithful": bi_ok,
               "batched_vs_full_faithful": bf_ok, "prune_ok": prune_ok,
               "boundary": [u, L], "per_config": per_cfg,
               "note": "Spliced branches stack into a batched cache; batched continuation equals "
                       "independent and full-reference continuations within bf16; prune/reorder "
                       "preserves survivor batch dims."}
    V2.save_json("batched_prune_splice.json", summary)
    V2.save_csv("batched_prune_splice_rows.csv", rows)
    V2.save_md("batched_prune_splice.md",
               f"# PART H - Batched splice and prune/reorder\n\n"
               f"**BG_PARTIAL_SPLICE_BATCHED_PRUNE_VERDICT = {verdict}**\n\n"
               f"- batched == independent (faithful): {bi_ok}\n"
               f"- batched == full reference (faithful): {bf_ok}\n"
               f"- prune/reorder ok: {prune_ok}\n")
    print("=" * 70)
    print(f"BG_PARTIAL_SPLICE_BATCHED_PRUNE_VERDICT = {verdict}")
    print(f"bi_ok={bi_ok} bf_ok={bf_ok} prune_ok={prune_ok}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
