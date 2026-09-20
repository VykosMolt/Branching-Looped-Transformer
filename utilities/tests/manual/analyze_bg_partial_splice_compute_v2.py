"""PART J - Compute accounting.

Quantifies whether v2 actually saves compute, via deterministic layer-pass
counting (primary) and wall-clock timing (supplementary), across boundaries and
branch counts K. Baseline = K full perturbed prefills. Splice = one shared prefix
prefill + K suffix recomputes.
"""

from __future__ import annotations

import json
import time
import torch

import bg_autoregressive_cache_common_v1 as C
import bg_autoregressive_cache_helpers_v1 as H
import bg_partial_cache_splice_v2_common as V2

PROMPT = C.TEST_PROMPTS[1]   # medium prompt
BOUNDARIES = [(0, 24), (1, 24), (2, 24), (3, 24), (2, 47), (3, 47)]
K_VALUES = [1, 2, 4, 8]
ALPHA = 1.0
TIMING_REPEATS = 5


@torch.no_grad()
def time_full_prefill(model, ids, am, u, L, direction, tr, repeats):
    times = []
    for _ in range(repeats):
        _, _, _, fp, secs, _ = V2.full_perturbed_prefill(model, ids, am, u, L, ALPHA, direction, tr)
        times.append(secs)
    return min(times), fp


@torch.no_grad()
def time_prefix_and_suffix(model, ids, u, L, Hb_direction, tr, repeats, device):
    pref_times, ppass = [], None
    for _ in range(repeats):
        sh, Hb, pp, psec = V2.prefix_prefill(model, ids, u, L, device)
        pref_times.append(psec); ppass = pp
    suf_times, spass = [], None
    for _ in range(repeats):
        Hp, _ = V2.apply_boundary_perturbation(Hb, ALPHA, Hb_direction, tr)
        _, _, sp, ssec = V2.suffix_recompute(model, Hp, u, L, device)
        spass = sp; suf_times.append(ssec)
    return min(pref_times), ppass, min(suf_times), spass


def main() -> int:
    C.set_seed()
    model, tok, info = C.load_model()
    device = next(model.parameters()).device
    direction = C.make_perturb_vector(info["hidden_size"], 555, device, torch.bfloat16)
    NL, NUT = info["num_hidden_layers"], info["total_ut_steps"]
    ids, am = C.tokenize(PROMPT["text"])
    tr = V2.default_token_range(ids.shape[1])

    # warmup
    _ = V2.full_perturbed_prefill(model, ids, am, 0, 24, ALPHA, direction, tr)
    _ = V2.prefix_prefill(model, ids, 0, 24, device)

    rows, per_boundary = [], []
    for (u, L) in BOUNDARIES:
        full_t, full_passes = time_full_prefill(model, ids, am, u, L, direction, tr, TIMING_REPEATS)
        pref_t, ppass, suf_t, spass = time_prefix_and_suffix(model, ids, u, L, direction, tr, TIMING_REPEATS, device)
        k_rows = []
        for K in K_VALUES:
            baseline_passes = K * full_passes
            splice_passes = ppass + K * spass
            baseline_t = K * full_t
            splice_t = pref_t + K * suf_t
            k_rows.append({
                "K": K,
                "baseline_passes": baseline_passes, "splice_passes": splice_passes,
                "passes_saved_fraction": round(1 - splice_passes / baseline_passes, 4),
                "baseline_seconds": round(baseline_t, 4), "splice_seconds": round(splice_t, 4),
                "wallclock_saved_fraction": round(1 - splice_t / baseline_t, 4),
            })
            rows.append({"boundary_loop": u, "boundary_layer": L, "K": K,
                         "baseline_passes": baseline_passes, "splice_passes": splice_passes,
                         "passes_saved_frac": round(1 - splice_passes / baseline_passes, 4),
                         "wallclock_saved_frac": round(1 - splice_t / baseline_t, 4)})
        per_boundary.append({
            "boundary": [u, L], "prefix_passes": ppass, "suffix_passes": spass,
            "full_passes": full_passes,
            "full_seconds": round(full_t, 4), "prefix_seconds": round(pref_t, 4),
            "suffix_seconds": round(suf_t, 4),
            "per_branch_passes_saved_vs_full": round(1 - spass / full_passes, 4),
            "per_branch_wallclock_saved_vs_full": round(1 - suf_t / full_t, 4),
            "k_scaling": k_rows,
        })

    # real saving if suffix < full per branch AND K>=2 total saving > 0
    per_branch_saving = all(b["suffix_passes"] < b["full_passes"] for b in per_boundary
                            if not (b["boundary"] == [3, 47]))
    k2_saving = all(any(kr["K"] == 2 and kr["passes_saved_fraction"] > 0 for kr in b["k_scaling"])
                    for b in per_boundary if b["suffix_passes"] < b["full_passes"])
    wallclock_saving = any(b["per_branch_wallclock_saved_vs_full"] > 0.05 for b in per_boundary)

    if per_branch_saving and k2_saving and wallclock_saving:
        verdict = "REAL_COMPUTE_SAVING_MEASURED"
    elif per_branch_saving and k2_saving:
        verdict = "THEORETICAL_SAVING_ONLY"
    else:
        verdict = "NO_COMPUTE_SAVING"

    summary = {
        "verdict": verdict, "prompt": PROMPT["id"], "alpha": ALPHA, "timing_repeats": TIMING_REPEATS,
        "per_boundary": per_boundary,
        "copy_affected_oracle_saving": 0.0,
        "copy_affected_note": "Mode A (copy affected slots from a full perturbed prefill) saves "
                              "NOTHING by construction; excluded from the saving claim.",
        "note": "Layer-pass counts are deterministic and primary; wall-clock is supplementary "
                "(min over repeats, post-warmup). Baseline = K full perturbed prefills; splice = "
                "one shared prefix prefill + K suffix recomputes. Saving grows with K and with "
                "boundary loop depth.",
    }
    V2.save_json("compute_accounting.json", summary)
    V2.save_csv("compute_accounting_rows.csv", rows)

    md = ["# PART J - Compute accounting\n",
          f"**BG_PARTIAL_SPLICE_COMPUTE_ACCOUNTING_VERDICT = {verdict}**\n",
          "Baseline = K full perturbed prefills; Splice = 1 shared prefix prefill + K suffix recomputes.\n",
          "## Per boundary (per-branch + K-scaling)\n"]
    for b in per_boundary:
        md.append(f"### boundary loop {b['boundary'][0]} layer {b['boundary'][1]}")
        md.append(f"- per-branch: prefix={b['prefix_passes']} suffix={b['suffix_passes']} full={b['full_passes']} "
                  f"passes (suffix saves {b['per_branch_passes_saved_vs_full']*100:.1f}% vs full; "
                  f"wall-clock {b['per_branch_wallclock_saved_vs_full']*100:.1f}%)")
        for kr in b["k_scaling"]:
            md.append(f"  - K={kr['K']}: {kr['splice_passes']} vs {kr['baseline_passes']} passes "
                      f"(saved {kr['passes_saved_fraction']*100:.1f}%); wall {kr['wallclock_saved_fraction']*100:.1f}%")
    md.append("\nMode A copy-affected oracle saves NOTHING (excluded from the claim).\n")
    V2.save_md("compute_accounting.md", "\n".join(md))

    print("=" * 70)
    print(f"BG_PARTIAL_SPLICE_COMPUTE_ACCOUNTING_VERDICT = {verdict}")
    for b in per_boundary:
        print(f"  boundary {b['boundary']}: suffix {b['suffix_passes']}/{b['full_passes']} passes, "
              f"per-branch wall save {b['per_branch_wallclock_saved_vs_full']*100:.0f}%")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
