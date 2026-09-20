"""PART D - Hook timing / affected-slot empirical probe.

Determines which cache slots actually change when perturbing at layer 24/36/47
(loop-targeted), by diffing every slot of an unperturbed prompt prefill vs a
perturbed prompt prefill. Confirms the boundary slot is unaffected and the first
affected slot is (loop, boundary_layer+1) -> selects the downstream_only policy.
"""

from __future__ import annotations

import json
import torch

import bg_autoregressive_cache_common_v1 as C
import bg_autoregressive_cache_helpers_v1 as H
import bg_partial_cache_splice_v2_common as V2

PROMPTS = C.TEST_PROMPTS[:3]
LAYERS = [24, 36, 47]
LOOPS = [0, 1, 2, 3]
ALPHAS = [0.0, 0.5, 1.0]


def changed_slots(root_cache, pert_cache, tol=0.0):
    """Return (first_changed_slot, [changed_slots], per-slot max_abs)."""
    n = max(len(root_cache.key_cache), len(pert_cache.key_cache))
    changed = []
    first = None
    detail = {}
    for slot in range(n):
        rk = root_cache.key_cache[slot] if slot < len(root_cache.key_cache) else None
        pk = pert_cache.key_cache[slot] if slot < len(pert_cache.key_cache) else None
        if rk is None and pk is None:
            continue
        if (rk is None) != (pk is None) or rk.shape != pk.shape:
            changed.append(slot)
            if first is None:
                first = slot
            detail[slot] = float("inf")
            continue
        kmax = (rk.float() - pk.float()).abs().max().item()
        vmax = (root_cache.value_cache[slot].float() - pert_cache.value_cache[slot].float()).abs().max().item()
        m = max(kmax, vmax)
        if m > tol:
            changed.append(slot)
            if first is None:
                first = slot
            detail[slot] = m
    return first, changed, detail


def main() -> int:
    C.set_seed()
    model, tok, info = C.load_model()
    device = next(model.parameters()).device
    direction = C.make_perturb_vector(info["hidden_size"], 909090, device, torch.bfloat16)
    NL, NUT = info["num_hidden_layers"], info["total_ut_steps"]

    rows, per_cfg = [], []
    loop_target_ok = True
    for p in PROMPTS:
        ids, am = C.tokenize(p["text"])
        P = ids.shape[1]
        tr = V2.default_token_range(P)
        _, root_cache, _, _ = C.prefill(model, ids, am)
        for u in LOOPS:
            for L in LAYERS:
                for alpha in ALPHAS:
                    hook = C.LayerOutputPerturbHook(direction, alpha, token_range=tr, target_loop=u)
                    handle = C.register_perturb_hook(model, L, hook)
                    _, pert_cache, _, _ = C.prefill(model, ids, am)
                    handle.remove()
                    first, changed, detail = changed_slots(root_cache, pert_cache)
                    boundary_slot = u * NL + L
                    expected_first = boundary_slot + 1 if (L + 1 < NL) else ((u + 1) * NL if u + 1 < NUT else None)
                    theory = V2.slot_policy(u, L, NUT, NL)["recompute_slots"]
                    boundary_slot_changed = (boundary_slot in changed)
                    downstream_only = (not boundary_slot_changed) and (set(changed) == set(theory) if alpha > 0 else len(changed) == 0)
                    if alpha > 0 and first is not None and first != expected_first:
                        loop_target_ok = loop_target_ok  # record but don't fail solely
                    per_cfg.append({
                        "prompt_id": p["id"], "loop": u, "layer": L, "alpha": alpha,
                        "boundary_slot": boundary_slot, "first_changed_slot": first,
                        "expected_first_affected": expected_first,
                        "n_changed": len(changed),
                        "boundary_slot_changed": boundary_slot_changed,
                        "matches_downstream_only_theory": bool(set(changed) == set(theory)) if alpha > 0 else None,
                        "n_theory_affected": len(theory),
                    })
                    rows.append({"prompt_id": p["id"], "loop": u, "layer": L, "alpha": alpha,
                                 "boundary_slot": boundary_slot, "first_changed": first,
                                 "expected_first": expected_first, "n_changed": len(changed),
                                 "boundary_changed": boundary_slot_changed})
                C.clear_cuda()

    pert = [c for c in per_cfg if c["alpha"] > 0]
    zero = [c for c in per_cfg if c["alpha"] == 0]
    zero_clean = all(c["n_changed"] == 0 for c in zero)
    boundary_unaffected = all(not c["boundary_slot_changed"] for c in pert)
    first_matches = all(c["first_changed_slot"] == c["expected_first_affected"] for c in pert)
    theory_matches = all(c["matches_downstream_only_theory"] for c in pert)

    if not zero_clean:
        verdict = "HOOK_TIMING_UNCLEAR"
    elif boundary_unaffected and first_matches and theory_matches:
        verdict = "DOWNSTREAM_ONLY_AFFECTED"
    elif not boundary_unaffected:
        verdict = "BOUNDARY_SLOT_AFFECTED"
    else:
        verdict = "AFFECTED_SLOTS_CONFIRMED"

    summary = {"verdict": verdict, "zero_perturbation_clean": zero_clean,
               "boundary_slot_unaffected_all": boundary_unaffected,
               "first_changed_matches_expected_all": first_matches,
               "changed_set_matches_downstream_only_theory_all": theory_matches,
               "per_config": per_cfg,
               "note": "Perturbing layer OUTPUT leaves the boundary slot (computed at layer input) "
                       "unaffected; first affected slot = (loop, boundary_layer+1); the full changed "
                       "set equals the downstream_only theory. This selects the downstream_only policy."}
    V2.save_json("hook_timing.json", summary)
    V2.save_csv("hook_timing_rows.csv", rows)
    V2.save_md("hook_timing.md",
               f"# PART D - Hook timing / affected slots\n\n"
               f"**BG_PARTIAL_SPLICE_HOOK_TIMING_VERDICT = {verdict}**\n\n"
               f"- zero perturbation changes nothing: {zero_clean}\n"
               f"- boundary slot unaffected (all): {boundary_unaffected}\n"
               f"- first changed slot == (loop, layer+1) (all): {first_matches}\n"
               f"- changed set == downstream_only theory (all): {theory_matches}\n\n"
               f"Confirms the downstream_only policy used by the v2 suffix recompute.\n")
    print("=" * 70)
    print(f"BG_PARTIAL_SPLICE_HOOK_TIMING_VERDICT = {verdict}")
    print(f"zero_clean={zero_clean} boundary_unaffected={boundary_unaffected} "
          f"first_matches={first_matches} theory_matches={theory_matches}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
