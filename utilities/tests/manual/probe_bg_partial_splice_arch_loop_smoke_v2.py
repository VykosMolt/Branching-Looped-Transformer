"""PART K - Architecture-looped integration smoke (partial splice).

Tiny smoke: at ONE stage, fork prompt-internal perturbation branches via the v2
partial splice (shared prefix + per-branch suffix), proxy-score, prune/reorder,
continue survivors, and compare to full perturbed references. Not steering, not a
science/reasoning run.
"""

from __future__ import annotations

import json
import torch
import torch.nn.functional as Fnn

import bg_autoregressive_cache_common_v1 as C
import bg_autoregressive_cache_helpers_v1 as H
import bg_partial_cache_splice_v2_common as V2

PROMPTS = C.TEST_PROMPTS[:3]
K = 4
SURVIVORS = 2
BOUNDARY = (2, 24)
ALPHA = 1.0
INIT_STEPS = 1
CONT_STEPS = 2


@torch.no_grad()
def main() -> int:
    C.set_seed()
    model, tok, info = C.load_model()
    device = next(model.parameters()).device
    NL, NUT, NSLOT = info["num_hidden_layers"], info["total_ut_steps"], info["expected_cache_slots"]
    u, L = BOUNDARY

    rows, per_prompt = [], []
    for p in PROMPTS:
        ids, am = C.tokenize(p["text"])
        tr = V2.default_token_range(ids.shape[1])
        sh, Hb, ppass, _ = V2.prefix_prefill(model, ids, u, L, device)
        directions = [C.make_perturb_vector(info["hidden_size"], 7000 + i, device, torch.bfloat16)
                      for i in range(K)]
        # fork branches via splice
        branch_caches, branch_next, refs = [], [], []
        am_full = None
        for i in range(K):
            ref_nl, ref_cache, am_full, _, _, _ = V2.full_perturbed_prefill(
                model, ids, am, u, L, ALPHA, directions[i], tr)
            refs.append((ref_nl, ref_cache))
            Hp, _ = V2.apply_boundary_perturbation(Hb, ALPHA, directions[i], tr)
            sp_nl, sfx, _, _ = V2.suffix_recompute(model, Hp, u, L, device)
            branch_caches.append(V2.merge_prefix_suffix(sh, sfx, NSLOT))
            branch_next.append(sp_nl)

        # decode INIT_STEPS, proxy-score by mean logprob, build batched cache
        states = []
        for i in range(K):
            states.append(H.BranchState(
                branch_id=f"{p['id']}.s{i}", parent_branch_id="root", root_branch_id=p["id"],
                input_ids=ids.clone(), attention_mask=am_full.clone(),
                past_key_values=branch_caches[i], cache_position=torch.tensor([ids.shape[1]], device=device),
                generated_ids=[], lineage_path=[p["id"], f"{p['id']}.s{i}"], score=None))

        # batched continue
        def stack(caches):
            out = type(caches[0])(getattr(caches[0], "max_cache_size", None))
            n = max(len(c.key_cache) for c in caches)
            kl, vl = [None] * n, [None] * n
            for slot in range(n):
                if caches[0].key_cache[slot] is not None:
                    kl[slot] = torch.cat([c.key_cache[slot] for c in caches], dim=0)
                    vl[slot] = torch.cat([c.value_cache[slot] for c in caches], dim=0)
            out._key_cache, out._value_cache = kl, vl
            for k in kl:
                if k is not None:
                    out._seen_tokens = k.shape[2]; break
            return out

        bcache = stack(branch_caches)
        bmask = am_full.repeat(K, 1)
        nl = torch.cat(branch_next, dim=0)
        cum_lp = [0.0] * K
        for _ in range(INIT_STEPS):
            lp = Fnn.log_softmax(nl.float(), dim=-1)
            col = nl.argmax(dim=-1, keepdim=True)
            for i in range(K):
                cum_lp[i] += float(lp[i, int(col[i, 0])].item())
                states[i].generated_ids.append(int(col[i, 0]))
            bmask = H.update_attention_mask(bmask, 1)
            nl, bcache = C.decode_step(model, col, bcache, bmask)

        scores = [cum_lp[i] / max(1, len(states[i].generated_ids)) for i in range(K)]
        survivor_idx = sorted(range(K), key=lambda i: scores[i], reverse=True)[:SURVIVORS]
        beam = torch.tensor(survivor_idx, device=device)
        bcache.reorder_cache(beam)
        nl = nl.index_select(0, beam)
        bmask = bmask.index_select(0, beam)
        states = [states[i] for i in survivor_idx]
        # continue survivors
        for _ in range(CONT_STEPS):
            col = nl.argmax(dim=-1, keepdim=True)
            bmask = H.update_attention_mask(bmask, 1)
            nl, bcache = C.decode_step(model, col, bcache, bmask)
            for r in range(SURVIVORS):
                states[r].generated_ids.append(int(col[r, 0]))

        # validate survivors vs full perturbed reference continuation
        surv_rows = []
        for r in range(SURVIVORS):
            bi = survivor_idx[r]
            ref_nl, ref_cache = refs[bi]
            # full reference continuation for the same total tokens
            ref_tokens, ref_logits = V2.greedy_continue(
                model, H.clone_universal_cache(ref_cache), am_full, ref_nl, INIT_STEPS + CONT_STEPS, device)
            cmp = C.compare_logits(nl[r], ref_logits[INIT_STEPS + CONT_STEPS - 0 - 1]
                                   if False else ref_logits[-1])
            surv_rows.append({"branch_id": states[r].branch_id, "score": scores[bi],
                              "token_match_prefixwise": states[r].generated_ids[:len(ref_tokens)] == ref_tokens[:len(states[r].generated_ids)],
                              **cmp})
            rows.append({"prompt_id": p["id"], "branch_id": states[r].branch_id,
                         "logit_rms": cmp["logit_rms"], "top1": cmp["top1_match"], "score": scores[bi]})
        eq = C.classify_equivalence(surv_rows)
        per_prompt.append({"prompt_id": p["id"], "survivor_idx": survivor_idx, "scores": scores,
                           "survivor_equivalence": eq, "survivors": surv_rows,
                           "scorer": "proxy_mean_token_logprob", "dualanchor": "skipped"})
        C.clear_cuda()

    faithful = all(pp["survivor_equivalence"]["cache_faithful"] for pp in per_prompt)
    verdict = "ARCH_LOOP_SPLICE_SMOKE_VALID" if faithful else "ARCH_LOOP_SPLICE_MISMATCH"

    summary = {"verdict": verdict, "K": K, "survivors": SURVIVORS, "boundary": [u, L],
               "survivors_faithful": faithful, "per_prompt": per_prompt,
               "note": "Fork-via-splice -> proxy-score -> prune/reorder -> carry survivors; survivor "
                       "continuation compared to full perturbed references. DualAnchor scoring skipped."}
    V2.save_json("arch_loop_smoke.json", summary)
    V2.save_csv("arch_loop_smoke_rows.csv", rows)
    V2.save_md("arch_loop_smoke.md",
               f"# PART K - Architecture-looped splice smoke\n\n"
               f"**BG_PARTIAL_SPLICE_ARCH_LOOP_SMOKE_VERDICT = {verdict}**\n\n"
               f"- survivors faithful to full perturbed references: {faithful}\n"
               f"- fork via partial splice (shared prefix), proxy logprob selection, prune/reorder, carry.\n")
    print("=" * 70)
    print(f"BG_PARTIAL_SPLICE_ARCH_LOOP_SMOKE_VERDICT = {verdict}")
    print(f"survivors_faithful={faithful}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
