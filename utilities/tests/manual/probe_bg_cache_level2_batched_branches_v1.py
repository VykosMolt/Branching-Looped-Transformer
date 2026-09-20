"""PART E - Level 2: batched branch cache equivalence.

Validate that decoding K branches batched in a single cache (batch dim K)
equals (a) decoding each branch independently with its own clone, and
(b) full no-cache recomputation of each branch.

To isolate the batched-vs-independent question from model-intrinsic argmax
near-ties, each branch's token sequence is determined once (independent greedy)
and then REPLAYED in the batched path, so all three paths score identical
sequences.
"""

from __future__ import annotations

import json

import torch

import bg_autoregressive_cache_common_v1 as C
import bg_autoregressive_cache_helpers_v1 as H

K_VALUES = [2, 4, 8]
N_STEPS = 4


@torch.no_grad()
def independent_branch_run(model, root_next_logits, root_cache, ids, am_full, forced_first, n_steps):
    """Independent (batch-1) cached decode for one branch, greedy after the
    forced first token. Returns (tokens, per_step_logits[list of [vocab]])."""
    device = ids.device
    cache = H.clone_universal_cache(root_cache)
    next_logits = root_next_logits.clone()
    cur_am = am_full.clone()
    tokens = []
    step_logits = []
    for step in range(n_steps):
        t = forced_first if step == 0 else int(next_logits[0].argmax().item())
        tokens.append(t)
        cur_am = H.update_attention_mask(cur_am, 1)
        next_logits, cache = C.decode_step(
            model, torch.tensor([[t]], device=device), cache, cur_am
        )
        step_logits.append(next_logits[0].detach().float().cpu())
    return tokens, step_logits


@torch.no_grad()
def batched_replay(model, root_cache, ids, am_full, branch_tokens, n_steps):
    """Expand root cache to batch K and replay the given per-branch token
    sequences. Returns per-step batched logits [K, vocab] (cpu float)."""
    device = ids.device
    K = len(branch_tokens)
    bcache = H.expand_universal_cache_for_branches(root_cache, K)
    # batched attention mask [K, prompt_len]
    bmask = am_full.repeat(K, 1)
    step_logits = []
    for step in range(n_steps):
        col = torch.tensor([[branch_tokens[i][step]] for i in range(K)], device=device)  # [K,1]
        bmask = H.update_attention_mask(bmask, 1)
        logits, bcache = C.decode_step(model, col, bcache, bmask)
        step_logits.append(logits.detach().float().cpu())  # [K, vocab]
    return step_logits, bcache


@torch.no_grad()
def full_recompute_branch(model, ids, am, branch_tokens, n_steps):
    """Full no-cache recompute per step for one branch's exact token sequence."""
    device = ids.device
    cur_ids = ids.clone()
    cur_am = am.clone()
    step_logits = []
    for step in range(n_steps):
        cur_ids = torch.cat([cur_ids, torch.tensor([[branch_tokens[step]]], device=device)], dim=1)
        cur_am = H.update_attention_mask(cur_am, 1)
        logits = C.full_recompute_logits(model, cur_ids, cur_am)[:, -1, :]
        step_logits.append(logits[0].detach().float().cpu())
    return step_logits


def main() -> int:
    C.set_seed()
    model, tok, info = C.load_model()

    all_rows = []
    per_config = []
    for p in C.TEST_PROMPTS:
        ids, am = C.tokenize(p["text"])
        root_next_logits, root_cache, _, am_full = C.prefill(model, ids, am)
        for K in K_VALUES:
            first_tokens = root_next_logits[0].topk(K).indices.tolist()
            # 1) independent runs -> token sequences + per-step logits
            branch_tokens = []
            indep_logits = []
            for ft in first_tokens:
                toks, slog = independent_branch_run(
                    model, root_next_logits, root_cache, ids, am_full, ft, N_STEPS
                )
                branch_tokens.append(toks)
                indep_logits.append(slog)
            # 2) batched replay
            batched_logits, bcache = batched_replay(model, root_cache, ids, am_full, branch_tokens, N_STEPS)
            # 3) full recompute per branch
            full_logits = [full_recompute_branch(model, ids, am, bt, N_STEPS) for bt in branch_tokens]

            # batched cache shape / batch-dim correctness
            fs = next(i for i, k in enumerate(bcache.key_cache) if k is not None)
            batch_dim_ok = (bcache.key_cache[fs].shape[0] == K)
            seqlen_ok = (bcache.get_seq_length(0) == int(ids.shape[1]) + N_STEPS)
            slots_ok = (sum(1 for k in bcache.key_cache if k is not None) == 192)
            nan_ok = all(torch.isfinite(batched_logits[s]).all().item() for s in range(N_STEPS))

            bi_rows = []   # batched vs independent
            bf_rows = []   # batched vs full
            for i in range(K):
                for step in range(N_STEPS):
                    cmp_bi = C.compare_logits(batched_logits[step][i], indep_logits[i][step])
                    cmp_bf = C.compare_logits(batched_logits[step][i], full_logits[i][step])
                    bi_rows.append(cmp_bi)
                    bf_rows.append(cmp_bf)
                    all_rows.append({
                        "prompt_id": p["id"], "K": K, "branch": i, "step": step + 1,
                        "bi_rms": cmp_bi["logit_rms"], "bi_max_abs": cmp_bi["logit_max_abs"],
                        "bi_top1": cmp_bi["top1_match"],
                        "bf_rms": cmp_bf["logit_rms"], "bf_max_abs": cmp_bf["logit_max_abs"],
                        "bf_top1": cmp_bf["top1_match"],
                    })
            eq_bi = C.classify_equivalence(bi_rows)
            eq_bf = C.classify_equivalence(bf_rows)
            per_config.append({
                "prompt_id": p["id"], "K": K, "first_tokens": first_tokens,
                "branch_tokens": branch_tokens,
                "batched_vs_independent": eq_bi,
                "batched_vs_full": eq_bf,
                "batch_dim_ok": batch_dim_ok, "seqlen_ok": seqlen_ok,
                "slots_ok": slots_ok, "logits_finite": nan_ok,
                "batched_cache_shape": list(bcache.key_cache[fs].shape),
            })
            C.clear_cuda()

    struct_ok = all(pc["batch_dim_ok"] and pc["seqlen_ok"] and pc["slots_ok"] and pc["logits_finite"]
                    for pc in per_config)
    bi_faithful = all(pc["batched_vs_independent"]["cache_faithful"] for pc in per_config)
    bf_faithful = all(pc["batched_vs_full"]["cache_faithful"] for pc in per_config)
    bi_strict = all(pc["batched_vs_independent"]["strict_equiv"] for pc in per_config)
    max_bi_rms = max(pc["batched_vs_independent"]["max_rms"] for pc in per_config)
    max_bf_rms = max(pc["batched_vs_full"]["max_rms"] for pc in per_config)
    max_bi_abs = max(pc["batched_vs_independent"]["max_abs"] for pc in per_config)
    max_bf_abs = max(pc["batched_vs_full"]["max_abs"] for pc in per_config)

    if not struct_ok:
        verdict = "BATCH_MASK_BUG"
    elif not (bi_faithful and bf_faithful):
        verdict = "BATCH_CACHE_MISMATCH"
    elif bi_strict:
        verdict = "BATCHED_BRANCH_CARRY_VALID"
    else:
        verdict = "BATCHED_NUMERIC_DRIFT_SMALL"

    summary = {
        "verdict": verdict, "K_values": K_VALUES, "n_steps": N_STEPS,
        "struct_ok": struct_ok,
        "batched_vs_independent_faithful": bi_faithful,
        "batched_vs_full_faithful": bf_faithful,
        "batched_vs_independent_strict": bi_strict,
        "max_batched_vs_independent_rms": max_bi_rms,
        "max_batched_vs_independent_max_abs": max_bi_abs,
        "max_batched_vs_full_rms": max_bf_rms,
        "max_batched_vs_full_max_abs": max_bf_abs,
        "per_config": per_config,
        "note": "Branch token sequences fixed via independent greedy then replayed batched "
                "so all three paths score identical sequences. Stress cases with left-padding "
                "and unequal lengths are in Part J (padding/mask stress).",
    }
    C.save_json("level2_batched_branches.json", summary)
    C.save_csv("level2_rows.csv", all_rows)

    md = ["# PART E - Level 2: batched branch cache equivalence\n",
          f"**BG_AUTOREGRESSIVE_CACHE_LEVEL2_VERDICT = {verdict}**\n",
          f"- structural (batch dim / seqlen / slots / finite): {struct_ok}",
          f"- batched==independent faithful: {bi_faithful} (max RMS {max_bi_rms:.4g}, max-abs {max_bi_abs:.4g})",
          f"- batched==full-recompute faithful: {bf_faithful} (max RMS {max_bf_rms:.4g}, max-abs {max_bf_abs:.4g})",
          f"- batched==independent strict (rms<=1e-4): {bi_strict}\n",
          "## Per (prompt, K)\n"]
    for pc in per_config:
        md.append(f"### {pc['prompt_id']} K={pc['K']}")
        md.append(f"- batch dim ok={pc['batch_dim_ok']} seqlen ok={pc['seqlen_ok']} slots ok={pc['slots_ok']} "
                  f"cache shape={pc['batched_cache_shape']}")
        md.append(f"- batched vs independent: {json.dumps(pc['batched_vs_independent'])}")
        md.append(f"- batched vs full: {json.dumps(pc['batched_vs_full'])}\n")
    C.save_md("level2_batched_branches.md", "\n".join(md))

    print("=" * 70)
    print(f"BG_AUTOREGRESSIVE_CACHE_LEVEL2_VERDICT = {verdict}")
    print(f"struct_ok={struct_ok} bi_faithful={bi_faithful} bf_faithful={bf_faithful} "
          f"max_bi_rms={max_bi_rms:.4g} max_bf_rms={max_bf_rms:.4g}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
