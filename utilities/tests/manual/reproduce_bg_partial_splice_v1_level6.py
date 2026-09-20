"""PART B - Reproduce v1 Level 6 diagnostic (copy-affected-slots splice).

Confirms the v1 diagnostic behavior before implementing the v2 suffix recompute:
- copy-affected-slots splice (root shared + affected copied FROM full perturbed
  reference) matches the reference (diagnostic, no compute saving),
- over-sharing (share affected slots from root) diverges,
- unperturbed root cache diverges for nonzero perturbation,
- zero perturbation matches.
"""

from __future__ import annotations

import json
import torch

import bg_autoregressive_cache_common_v1 as C
import bg_autoregressive_cache_helpers_v1 as H
import bg_partial_cache_splice_v2_common as V2

PROMPTS = C.TEST_PROMPTS[:3]
LAYERS = [24, 36, 47]
PERTURB_LOOP = 0
N_STEPS = 4


def copy_affected_oracle(root_cache, ref_cache, recompute_slots):
    """v1 Option A: clone root, copy affected slots from the full perturbed reference."""
    spliced = H.clone_universal_cache(root_cache)
    for slot in recompute_slots:
        if slot < len(ref_cache.key_cache) and ref_cache.key_cache[slot] is not None:
            spliced._key_cache[slot] = ref_cache.key_cache[slot].detach().clone()
            spliced._value_cache[slot] = ref_cache.value_cache[slot].detach().clone()
    return spliced


def main() -> int:
    C.set_seed()
    model, tok, info = C.load_model()
    device = next(model.parameters()).device
    direction = C.make_perturb_vector(info["hidden_size"], 31337, device, torch.bfloat16)
    NL, NUT = info["num_hidden_layers"], info["total_ut_steps"]

    rows, per_cfg = [], []
    for p in PROMPTS:
        ids, am = C.tokenize(p["text"])
        P = ids.shape[1]
        tr = V2.default_token_range(P)
        for layer in LAYERS:
            pol = V2.slot_policy(PERTURB_LOOP, layer, NUT, NL)
            root_nl, root_cache, am_full, _ = V2.root_prefill_with_boundary(model, ids, am, PERTURB_LOOP, layer)

            # alpha=0.5 reference + oracle copy splice
            ref_nl, ref_cache, am_full_b, _, _, prms = V2.full_perturbed_prefill(
                model, ids, am, PERTURB_LOOP, layer, 2.0, direction, tr)
            ref_tokens, ref_logits = V2.greedy_continue(model, H.clone_universal_cache(ref_cache),
                                                         am_full_b, ref_nl, N_STEPS, device)

            oracle = copy_affected_oracle(root_cache, ref_cache, pol["recompute_slots"])
            oracle_nl_seed = ref_nl  # same prompt-boundary prediction (cache identical to ref)
            oracle_logits = [ref_nl[0].detach().float().cpu()] + V2.replay_logits(
                model, H.clone_universal_cache(oracle), am_full_b, ref_tokens[:N_STEPS - 1], device)
            oracle_eq = C.classify_equivalence(
                [C.compare_logits(oracle_logits[k], ref_logits[k]) for k in range(N_STEPS)])
            oracle_cache_cmp = H.compare_universal_caches(oracle, ref_cache)

            # over-share negative control: share affected slots from ROOT (do not copy)
            overshare = H.clone_universal_cache(root_cache)  # affected slots left as root
            overshare_logits = [root_nl[0].detach().float().cpu()] + V2.replay_logits(
                model, H.clone_universal_cache(overshare), am_full, ref_tokens[:N_STEPS - 1], device)
            overshare_eq = C.classify_equivalence(
                [C.compare_logits(overshare_logits[k], ref_logits[k]) for k in range(N_STEPS)])
            overshare_diverges = (not overshare_eq["cache_faithful"]) and overshare_eq["max_abs"] > C.TOL_DRIFT_MAX_ABS

            # zero perturbation: oracle should match root/reference
            ref0_nl, ref0_cache, am0, _, _, _ = V2.full_perturbed_prefill(
                model, ids, am, PERTURB_LOOP, layer, 0.0, direction, tr)
            zero_cmp = H.compare_universal_caches(ref0_cache, root_cache)
            zero_matches = (zero_cmp["max_abs"] == 0.0)

            per_cfg.append({
                "prompt_id": p["id"], "layer": layer, "n_shared": pol["n_shared"],
                "n_affected": pol["n_recompute"], "perturb_rms": prms,
                "oracle_matches_ref": oracle_eq["cache_faithful"],
                "oracle_cache_max_abs": oracle_cache_cmp["max_abs"],
                "oracle_eq": oracle_eq,
                "overshare_diverges": overshare_diverges, "overshare_max_abs": overshare_eq["max_abs"],
                "zero_perturbation_matches_root": zero_matches,
            })
            rows.append({"prompt_id": p["id"], "layer": layer,
                         "oracle_rms": oracle_eq["max_rms"], "oracle_top1": oracle_eq["all_top1_match"],
                         "overshare_max_abs": overshare_eq["max_abs"], "overshare_diverges": overshare_diverges,
                         "zero_matches": zero_matches})
            C.clear_cuda()

    oracle_ok = all(c["oracle_matches_ref"] for c in per_cfg)
    overshare_ok = all(c["overshare_diverges"] for c in per_cfg)
    zero_ok = all(c["zero_perturbation_matches_root"] for c in per_cfg)
    if oracle_ok and overshare_ok and zero_ok:
        verdict = "REPRODUCED"
    elif oracle_ok:
        verdict = "SMALL_DRIFT"
    else:
        verdict = "FAILED_REPRODUCTION"

    summary = {"verdict": verdict, "oracle_matches_ref_all": oracle_ok,
               "overshare_diverges_all": overshare_ok, "zero_matches_all": zero_ok,
               "per_config": per_cfg,
               "note": "Reproduces v1 Level 6 diagnostic: copy-affected-slots splice equals the "
                       "full perturbed reference (DIAGNOSTIC ONLY, no compute saving); over-share "
                       "diverges; zero perturbation matches root."}
    V2.save_json("reproduce_level6_v1.json", summary)
    V2.save_csv("reproduce_level6_rows.csv", rows)
    V2.save_md("reproduce_level6_v1.md",
               f"# PART B - Reproduce v1 Level 6 diagnostic\n\n"
               f"**BG_PARTIAL_SPLICE_V2_REPRODUCE_V1_VERDICT = {verdict}**\n\n"
               f"- copy-affected oracle matches reference (all): {oracle_ok}\n"
               f"- over-share diverges (all): {overshare_ok}\n"
               f"- zero perturbation matches root (all): {zero_ok}\n\n"
               f"Copy-affected splice is DIAGNOSTIC ONLY (affected slots copied from a full "
               f"perturbed prefill) — it does not save compute. v2 implements the real suffix "
               f"recompute in Part E.\n")
    print("=" * 70)
    print(f"BG_PARTIAL_SPLICE_V2_REPRODUCE_V1_VERDICT = {verdict}")
    print(f"oracle_ok={oracle_ok} overshare_diverges={overshare_ok} zero_ok={zero_ok}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
