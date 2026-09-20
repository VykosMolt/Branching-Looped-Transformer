"""PART G - Multi-branch splice equivalence.

From ONE shared root-prefix prefill, build K perturbation branches (different
seeds and/or layers) via suffix recompute, continue each, and compare each to its
own full perturbed reference. Verify storage independence and no contamination.
This is where compute saving materializes: the prefix is computed once.
"""

from __future__ import annotations

import json
import torch

import bg_autoregressive_cache_common_v1 as C
import bg_autoregressive_cache_helpers_v1 as H
import bg_partial_cache_splice_v2_common as V2

PROMPTS = C.TEST_PROMPTS[:3]
K_VALUES = [2, 4]
BOUNDARY = (2, 24)   # loop 2, layer 24 -> meaningful suffix saving
ALPHA = 1.0
MAX_STEPS = 4


def main() -> int:
    C.set_seed()
    model, tok, info = C.load_model()
    device = next(model.parameters()).device
    NL, NUT, NSLOT = info["num_hidden_layers"], info["total_ut_steps"], info["expected_cache_slots"]
    u, L = BOUNDARY

    rows, per_cfg = [], []
    contamination_any = False
    for p in PROMPTS:
        ids, am = C.tokenize(p["text"])
        P = ids.shape[1]
        tr = V2.default_token_range(P)
        for K in K_VALUES:
            # shared prefix computed ONCE for all K branches
            sh_cache, Hb, ppass, psec = V2.prefix_prefill(model, ids, u, L, device)
            directions = [C.make_perturb_vector(info["hidden_size"], 1000 + i, device, torch.bfloat16)
                          for i in range(K)]
            branch_results = []
            spliced_caches = []
            total_suffix_passes = 0
            for i in range(K):
                # full perturbed reference for this branch
                ref_nl, ref_cache, am_full, fpass, _, _ = V2.full_perturbed_prefill(
                    model, ids, am, u, L, ALPHA, directions[i], tr)
                ref_tokens, ref_logits = V2.greedy_continue(
                    model, H.clone_universal_cache(ref_cache), am_full, ref_nl, MAX_STEPS, device)
                # spliced branch (shares prefix)
                Hp, _ = V2.apply_boundary_perturbation(Hb, ALPHA, directions[i], tr)
                sp_nl, sfx, spass, _ = V2.suffix_recompute(model, Hp, u, L, device)
                total_suffix_passes += spass
                spliced = V2.merge_prefix_suffix(sh_cache, sfx, NSLOT)
                spliced_caches.append(spliced)
                sp_tokens, sp_logits = V2.greedy_continue(
                    model, H.clone_universal_cache(spliced), am_full, sp_nl, MAX_STEPS, device)
                eq = C.classify_equivalence([C.compare_logits(sp_logits[k], ref_logits[k])
                                             for k in range(MAX_STEPS + 1)])
                cc = H.compare_universal_caches(spliced, ref_cache)
                branch_results.append({
                    "branch": i, "token_match": (sp_tokens == ref_tokens),
                    "equivalence": eq, "cache_vs_ref_max_abs": cc["max_abs"],
                    "suffix_passes": spass, "full_passes": fpass,
                })
                rows.append({"prompt_id": p["id"], "K": K, "branch": i, "rms": eq["max_rms"],
                             "max_abs": eq["max_abs"], "top1": eq["all_top1_match"],
                             "token_match": (sp_tokens == ref_tokens), "cache_max_abs": cc["max_abs"]})

            # storage independence + contamination: mutate branch 0 cache, verify branch 1 unchanged
            if K >= 2:
                fs = next(i for i, k in enumerate(spliced_caches[0].key_cache) if k is not None)
                ptrs = [c.key_cache[fs].data_ptr() for c in spliced_caches]
                indep = len(set(ptrs)) == len(ptrs)
                snap = H.clone_universal_cache(spliced_caches[1])
                spliced_caches[0]._key_cache[fs] = spliced_caches[0].key_cache[fs] + 99.0
                cc2 = H.compare_universal_caches(spliced_caches[1], snap)
                contaminated = cc2["max_abs"] > 0.0
                contamination_any = contamination_any or contaminated
            else:
                indep, contaminated = True, False

            baseline = K * branch_results[0]["full_passes"]
            spliced_total = ppass + total_suffix_passes
            per_cfg.append({
                "prompt_id": p["id"], "K": K, "boundary": [u, L],
                "branches": branch_results, "independent_storage": indep,
                "contaminated": contaminated,
                "compute_baseline_passes": baseline,
                "compute_spliced_passes": spliced_total,
                "compute_saved_fraction": round(1 - spliced_total / baseline, 4),
                "all_token_match": all(b["token_match"] for b in branch_results),
                "all_faithful": all(b["equivalence"]["cache_faithful"] for b in branch_results),
            })
            C.clear_cuda()

    all_faithful = all(c["all_faithful"] for c in per_cfg)
    all_indep = all(c["independent_storage"] for c in per_cfg)
    strict = all(b["equivalence"]["strict_equiv"] for c in per_cfg for b in c["branches"])
    if contamination_any or not all_indep:
        verdict = "BRANCH_CONTAMINATION"
    elif not all_faithful:
        verdict = "SPLICE_INVALID"
    elif strict:
        verdict = "MULTI_BRANCH_SPLICE_VALID"
    else:
        verdict = "MULTI_BRANCH_SPLICE_NUMERIC_DRIFT_SMALL"

    summary = {"verdict": verdict, "all_faithful": all_faithful, "independent_storage_all": all_indep,
               "contamination_any": contamination_any, "boundary": [u, L], "alpha": ALPHA,
               "per_config": per_cfg,
               "note": "Each branch's spliced cache matches its own full perturbed reference. The "
                       "shared prefix prefill is computed once across K branches -> compute saved "
                       "fraction reported per config."}
    V2.save_json("multi_branch_splice.json", summary)
    V2.save_csv("multi_branch_splice_rows.csv", rows)
    V2.save_md("multi_branch_splice.md",
               f"# PART G - Multi-branch splice equivalence\n\n"
               f"**BG_PARTIAL_SPLICE_MULTI_BRANCH_VERDICT = {verdict}**\n\n"
               f"- all branches faithful to own reference: {all_faithful}\n"
               f"- independent storage (all): {all_indep}\n"
               f"- contamination: {contamination_any}\n\n"
               + "\n".join(f"- {c['prompt_id']} K={c['K']}: saved {c['compute_saved_fraction']*100:.1f}% "
                           f"({c['compute_spliced_passes']} vs {c['compute_baseline_passes']} passes)"
                           for c in per_cfg) + "\n")
    print("=" * 70)
    print(f"BG_PARTIAL_SPLICE_MULTI_BRANCH_VERDICT = {verdict}")
    print(f"faithful={all_faithful} indep={all_indep} contam={contamination_any}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
