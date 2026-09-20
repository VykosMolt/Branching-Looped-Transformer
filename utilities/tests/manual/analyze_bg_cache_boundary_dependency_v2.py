"""PART C - Cache dependency and boundary theory audit.

Formalizes, per (boundary_loop, boundary_layer), which cache slots are shareable
under each candidate policy, and the layer-pass accounting. The empirical
confirmation of which slots actually change is in Part D (hook timing).
"""

from __future__ import annotations

import json

import bg_autoregressive_cache_common_v1 as C
import bg_partial_cache_splice_v2_common as V2

LAYERS = [24, 36, 47]
POLICIES = ["conservative", "downstream_only", "aggressive", "loop_boundary"]


def main() -> int:
    C.set_seed()
    _, _, info = C.load_model()
    NL, NUT = info["num_hidden_layers"], info["total_ut_steps"]

    rows, per_boundary = [], []
    for u in range(NUT):
        for L in LAYERS:
            slot = u * NL + L
            acc = V2.layer_passes(u, L, NUT, NL)
            policy_results = {}
            for pol_name in POLICIES:
                pol = V2.slot_policy(u, L, NUT, NL, pol_name)
                # expected correctness: downstream_only & conservative are correct;
                # aggressive shares affected slots (expected to fail); loop_boundary is
                # correct but over-conservative (shares less, recomputes more).
                expected = {
                    "conservative": "CORRECT (shares < boundary_slot; recomputes boundary_slot+)",
                    "downstream_only": "CORRECT (perturb at layer output -> boundary slot unaffected)",
                    "aggressive": "EXPECTED_FAIL (shares affected downstream slot)",
                    "loop_boundary": "CORRECT_BUT_OVERCONSERVATIVE (recomputes whole boundary loop)",
                }[pol_name]
                policy_results[pol_name] = {
                    "n_shared": pol["n_shared"], "n_recompute": pol["n_recompute"],
                    "expected": expected,
                    "shareable_fraction": round(pol["n_shared"] / (NL * NUT), 4),
                }
                rows.append({"boundary_loop": u, "boundary_layer": L, "boundary_slot": slot,
                             "policy": pol_name, "n_shared": pol["n_shared"],
                             "n_recompute": pol["n_recompute"],
                             "shareable_fraction": round(pol["n_shared"] / (NL * NUT), 4),
                             "expected": expected})
            per_boundary.append({
                "boundary_loop": u, "boundary_layer": L, "boundary_slot": slot,
                "layer_pass_accounting": acc,
                "min_prefix_passes": acc["prefix_passes"], "suffix_passes": acc["suffix_passes"],
                "full_passes": acc["full_passes"],
                "policies": policy_results,
            })

    verdict = "BOUNDARY_THEORY_READY"
    summary = {
        "verdict": verdict, "total_ut_steps": NUT, "num_hidden_layers": NL,
        "boundaries": per_boundary,
        "key_assumption": "The v1 hook perturbs the LAYER OUTPUT (after that layer's K/V was "
                          "written), so slot (u, boundary_layer) is UNAFFECTED. The first affected "
                          "slot is (u, boundary_layer+1). This is confirmed empirically in Part D.",
        "compute_note": "Minimal shared-prefix prefill = prefix_passes (loops<u + loop u layers "
                        "0..L). Per-branch suffix = suffix_passes. Amortized over K branches: "
                        "total = prefix + K*suffix vs baseline K*full.",
    }
    V2.save_json("boundary_dependency.json", summary)
    V2.save_csv("boundary_policy_rows.csv", rows)

    md = ["# PART C - Cache boundary dependency theory\n",
          f"**BG_PARTIAL_SPLICE_BOUNDARY_DEPENDENCY_VERDICT = {verdict}**\n",
          f"- total_ut_steps={NUT}, num_hidden_layers={NL}, slots={NL*NUT}\n",
          "Perturbation at layer OUTPUT => boundary slot unaffected; first affected slot = "
          "(u, boundary_layer+1). downstream_only and conservative policies are correct; "
          "aggressive (shares an affected slot) should fail; loop_boundary is correct but "
          "recomputes more than necessary.\n",
          "## Per boundary (downstream_only)\n"]
    for b in per_boundary:
        do = b["policies"]["downstream_only"]
        md.append(f"- loop {b['boundary_loop']} layer {b['boundary_layer']} (slot {b['boundary_slot']}): "
                  f"shared={do['n_shared']} recompute={do['n_recompute']} "
                  f"(share {do['shareable_fraction']*100:.1f}%) | prefix_passes={b['min_prefix_passes']} "
                  f"suffix_passes={b['suffix_passes']} full={b['full_passes']}")
    V2.save_md("boundary_dependency.md", "\n".join(md))

    print("=" * 70)
    print(f"BG_PARTIAL_SPLICE_BOUNDARY_DEPENDENCY_VERDICT = {verdict}")
    print(f"boundaries={len(per_boundary)}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
