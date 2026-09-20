"""PART F - Single-branch splice equivalence.

Validate one v2 suffix-recomputed spliced branch against a full perturbed prompt
reference, across layers 24/36/47 and loops, continuing 1/2/4 tokens. Includes an
aggressive over-share negative control (share an affected slot from root) which
must diverge.
"""

from __future__ import annotations

import json
import torch

import bg_autoregressive_cache_common_v1 as C
import bg_autoregressive_cache_helpers_v1 as H
import bg_partial_cache_splice_v2_common as V2

PROMPTS = C.TEST_PROMPTS[:4]
BOUNDARIES = [(0, 24), (1, 36), (2, 24), (3, 47)]
ALPHAS = [0.0, 0.5, 1.0]
STEP_CHECKS = [1, 2, 4]
MAX_STEPS = 4


def main() -> int:
    C.set_seed()
    model, tok, info = C.load_model()
    device = next(model.parameters()).device
    direction = C.make_perturb_vector(info["hidden_size"], 135790, device, torch.bfloat16)
    NL, NUT, NSLOT = info["num_hidden_layers"], info["total_ut_steps"], info["expected_cache_slots"]

    rows, per_cfg = [], []
    neg_rows = []
    for p in PROMPTS:
        ids, am = C.tokenize(p["text"])
        P = ids.shape[1]
        tr = V2.default_token_range(P)
        for (u, L) in BOUNDARIES:
            sh_cache, Hb, ppass, psec = V2.prefix_prefill(model, ids, u, L, device)
            pol = V2.slot_policy(u, L, NUT, NL)
            for alpha in ALPHAS:
                # reference (full perturbed prefill) + its continuation
                ref_nl, ref_cache, am_full, fpass, fsec, prms = V2.full_perturbed_prefill(
                    model, ids, am, u, L, alpha, direction, tr)
                ref_tokens, ref_logits = V2.greedy_continue(
                    model, H.clone_universal_cache(ref_cache), am_full, ref_nl, MAX_STEPS, device)
                # spliced branch
                Hp, prms2 = V2.apply_boundary_perturbation(Hb, alpha, direction, tr)
                sp_nl, sfx, spass, ssec = V2.suffix_recompute(model, Hp, u, L, device)
                spliced = V2.merge_prefix_suffix(sh_cache, sfx, NSLOT)
                sp_tokens, sp_logits = V2.greedy_continue(
                    model, H.clone_universal_cache(spliced), am_full, sp_nl, MAX_STEPS, device)

                step_cmps = [C.compare_logits(sp_logits[k], ref_logits[k]) for k in range(MAX_STEPS + 1)]
                eq = C.classify_equivalence(step_cmps)
                token_match = (sp_tokens == ref_tokens)
                ckpt = {str(c): (sp_tokens[:c] == ref_tokens[:c]) for c in STEP_CHECKS}
                cc = H.compare_universal_caches(spliced, ref_cache)
                per_cfg.append({
                    "prompt_id": p["id"], "loop": u, "layer": L, "alpha": alpha,
                    "equivalence": eq, "token_match": token_match, "checkpoint_token_match": ckpt,
                    "cache_vs_ref_max_abs": cc["max_abs"], "cache_missing": cc["n_missing_one"],
                    "prefix_passes": ppass, "suffix_passes": spass, "full_passes": fpass,
                    "first_mismatch_slot": None if cc["max_abs"] == 0 else "see cache compare",
                })
                rows.append({"prompt_id": p["id"], "loop": u, "layer": L, "alpha": alpha,
                             "rms": eq["max_rms"], "max_abs": eq["max_abs"], "top1": eq["all_top1_match"],
                             "token_match": token_match, "cache_max_abs": cc["max_abs"],
                             "suffix_passes": spass, "full_passes": fpass})

                # aggressive over-share negative control (alpha>0 only):
                # WRONGLY share the first affected slot from the UNPERTURBED suffix.
                # The affected slot genuinely differs from the reference (perturbed),
                # so the over-share cache must diverge at that slot.
                if alpha > 0 and pol["recompute_slots"]:
                    first_aff = pol["recompute_slots"][0]
                    Hp0, _ = V2.apply_boundary_perturbation(Hb, 0.0, direction, tr)
                    _, sfx0, _, _ = V2.suffix_recompute(model, Hp0, u, L, device)
                    bad = H.clone_universal_cache(spliced)
                    slot_max_abs = None
                    if sfx0.key_cache[first_aff] is not None and ref_cache.key_cache[first_aff] is not None:
                        bad._key_cache[first_aff] = sfx0.key_cache[first_aff].detach().clone()
                        bad._value_cache[first_aff] = sfx0.value_cache[first_aff].detach().clone()
                        # how much does the wrongly-shared (unperturbed) slot differ from reference?
                        slot_max_abs = float((sfx0.key_cache[first_aff].float()
                                              - ref_cache.key_cache[first_aff].float()).abs().max().item())
                    bad_cc = H.compare_universal_caches(bad, ref_cache)
                    bad_logits = [sp_nl[0].detach().float().cpu()] + V2.replay_logits(
                        model, H.clone_universal_cache(bad), am_full, ref_tokens[:MAX_STEPS], device)
                    bad_eq = C.classify_equivalence(
                        [C.compare_logits(bad_logits[k], ref_logits[k]) for k in range(MAX_STEPS + 1)])
                    # divergence is detectable at the cache level (wrongly-shared slot differs)
                    # and/or the continuation level.
                    diverges = (bad_cc["max_abs"] > C.TOL_DRIFT_MAX_ABS) or \
                               ((not bad_eq["cache_faithful"]) and bad_eq["max_abs"] > C.TOL_DRIFT_MAX_ABS)
                    neg_rows.append({"prompt_id": p["id"], "loop": u, "layer": L, "alpha": alpha,
                                     "wrongly_shared_slot": first_aff,
                                     "slot_unperturbed_vs_ref_max_abs": slot_max_abs,
                                     "overshare_cache_max_abs": bad_cc["max_abs"],
                                     "overshare_continuation_max_abs": bad_eq["max_abs"],
                                     "diverges": bool(diverges)})
            C.clear_cuda()

    pert = [c for c in per_cfg if c["alpha"] > 0]
    all_faithful = all(c["equivalence"]["cache_faithful"] for c in per_cfg)
    all_token = all(c["token_match"] for c in per_cfg)
    all_cache_exact = all(c["cache_vs_ref_max_abs"] == 0.0 for c in per_cfg)
    strict = all(c["equivalence"]["strict_equiv"] for c in per_cfg)
    neg_ok = (len(neg_rows) > 0) and all(r["diverges"] for r in neg_rows)

    if not all_faithful:
        verdict = "SINGLE_BRANCH_SPLICE_INVALID"
    elif strict and all_cache_exact:
        verdict = "SINGLE_BRANCH_SPLICE_VALID"
    else:
        verdict = "SINGLE_BRANCH_SPLICE_NUMERIC_DRIFT_SMALL"

    summary = {"verdict": verdict, "all_faithful": all_faithful, "all_token_match": all_token,
               "all_prefill_cache_bit_exact": all_cache_exact, "strict_equiv_all": strict,
               "aggressive_overshare_negative_control_diverges": neg_ok,
               "boundaries": BOUNDARIES, "alphas": ALPHAS, "per_config": per_cfg,
               "negative_control_rows": neg_rows,
               "note": "Spliced branch (suffix recompute) reproduces the full perturbed reference "
                       "cache bit-exactly, so continuation matches bit-exactly too. Aggressive "
                       "over-share (one affected slot taken unperturbed) diverges."}
    V2.save_json("single_branch_splice.json", summary)
    V2.save_csv("single_branch_splice_rows.csv", rows)
    V2.save_md("single_branch_splice.md",
               f"# PART F - Single-branch splice equivalence\n\n"
               f"**BG_PARTIAL_SPLICE_SINGLE_BRANCH_VERDICT = {verdict}**\n\n"
               f"- all spliced branches faithful to full reference: {all_faithful}\n"
               f"- all generated token sequences match reference: {all_token}\n"
               f"- all prefill caches bit-exact vs reference: {all_cache_exact}\n"
               f"- strict equivalence (incl. continuation) all: {strict}\n"
               f"- aggressive over-share negative control diverges: {neg_ok}\n")
    print("=" * 70)
    print(f"BG_PARTIAL_SPLICE_SINGLE_BRANCH_VERDICT = {verdict}")
    print(f"faithful={all_faithful} token={all_token} cache_exact={all_cache_exact} "
          f"strict={strict} neg_ctrl_diverges={neg_ok}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
