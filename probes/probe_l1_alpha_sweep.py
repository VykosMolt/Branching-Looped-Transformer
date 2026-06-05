"""L1<->L4 weighted-mean alpha sweep (zero-shot, frozen published head).

Follow-up to probe_l1_ablation.py. That probe showed mean(L1, L4) centered
acc 0.620 beats every single-loop variant. Open question for the
Experiment-2 arm design: is the fusion gain about *mixing weight* (a
2048-dim weighted mean captures it) or about the head seeing L1 and L4 as
*separate* signals (only a 4096-dim concat/diff arm can capture it)?

This sweeps fusion = alpha * L1 + (1 - alpha) * L4 over a grid, frozen
published head, swap-balanced, fp32, no model forward. Reuses the saved
v10 Thinking states.

  alpha = 0.0  -> pure L4 (== L4-only baseline)
  alpha = 0.5  -> mean(L1, L4) (== the 0.620 result, internal consistency check)
  alpha = 1.0  -> pure L1 (== L1-only)

Reports the full centered_acc curve AND bias_to_signal per alpha (not just
the max), because a flat curve vs a sharp peak have different
architectural implications even at the same headline number, and because
Experiment 2 optimizes against a symmetric-offset penalty so a low-bias
alpha is the better learned-init seed.

Output: artifacts/reports/evaluator/probe_l1_alpha_sweep.json
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator_core.pairwise_evaluator import PairwiseEvaluator

STATES_PT = "artifacts/reports/evaluator/hh_loop_states_200_thinking.pt"
HEAD_CKPT = "artifacts/checkpoints/evaluator/pairwise_epoch2.pt"
OUT_JSON = "artifacts/reports/evaluator/probe_l1_alpha_sweep.json"

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0]
MEAN_REF_CENTERED = 0.620          # mean(L1,L4) from probe_l1_ablation.py
MEANINGFUL_GAIN = 0.005            # peak must beat the mean by >= 0.5 pp to be "meaningful"
NEAR_MAX_BAND = 0.005              # alphas within 0.5 pp of the max are "near-optimal"
FLAT_BAND_LO, FLAT_BAND_HI = 0.3, 0.5  # prior-optimum band for flatness check
DEGEN_SIGN_REV_MAX = 0.05
DEGEN_POS_RATE_MIN = 0.95


def fused(states: List[torch.Tensor], alpha: float) -> List[torch.Tensor]:
    """alpha*L1 + (1-alpha)*L4, replicated into all 4 head input slots."""
    base = alpha * states[0] + (1.0 - alpha) * states[3]
    return [base, base, base, base]


@torch.no_grad()
def score(head, a_states, a_mask, b_states, b_mask, device) -> float:
    a = [s.to(device=device, dtype=torch.float32).unsqueeze(0) for s in a_states]
    b = [s.to(device=device, dtype=torch.float32).unsqueeze(0) for s in b_states]
    am = a_mask.to(device=device, dtype=torch.float32).unsqueeze(0)
    bm = b_mask.to(device=device, dtype=torch.float32).unsqueeze(0)
    v = float(head(a, am, b, bm).view(-1).item())
    return v if math.isfinite(v) else float("nan")


def metrics(s_cr: np.ndarray, s_rc: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(s_cr) & np.isfinite(s_rc)
    s_cr, s_rc = s_cr[valid], s_rc[valid]
    n = int(s_cr.size)
    centered = s_cr - s_rc
    summed = s_cr + s_rc
    std_c = float(np.std(centered))
    strict = np.sign(s_cr) != np.sign(s_rc)
    return {
        "n_valid": n,
        "canonical_acc": float(np.mean(s_cr > 0)),
        "centered_acc": float(np.mean(centered > 0)),
        "bias_to_signal": (abs(float(np.mean(summed))) / std_c) if std_c > 1e-9 else float("inf"),
        "s_cr_mean": float(np.mean(s_cr)),
        "s_cr_std": float(np.std(s_cr)),
        "s_rc_mean": float(np.mean(s_rc)),
        "s_rc_std": float(np.std(s_rc)),
        "pos_rate_cr": float(np.mean(s_cr > 0)),
        "strict_sign_reversal_rate": float(np.mean(strict)),
        "centered_mean": float(np.mean(centered)),
        "summed_mean": float(np.mean(summed)),
    }


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sp = PROJECT_ROOT / STATES_PT
    if not sp.exists():
        print(f"STOP: {STATES_PT} not found.")
        sys.exit(2)
    payload = torch.load(sp, map_location="cpu", weights_only=False)
    packs = payload["packs"]
    if len(packs[0]["chosen_states"]) < 4:
        print("STOP: saved states lack per-token L1..L4.")
        sys.exit(2)
    print(f"Loaded {len(packs)} packs; source meta: {payload.get('meta')}")

    head = PairwiseEvaluator().to(device)
    ckpt = torch.load(PROJECT_ROOT / HEAD_CKPT, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    head.load_state_dict(sd)
    head.eval()
    print(f"Head loaded; device={device}")

    res: Dict[str, Dict[str, float]] = {}
    for alpha in ALPHAS:
        s_cr, s_rc = [], []
        for pk in packs:
            cv = fused(pk["chosen_states"], alpha)
            rv = fused(pk["rejected_states"], alpha)
            s_cr.append(score(head, cv, pk["chosen_mask"], rv, pk["rejected_mask"], device))
            s_rc.append(score(head, rv, pk["rejected_mask"], cv, pk["chosen_mask"], device))
        res[f"{alpha:.2f}"] = metrics(np.asarray(s_cr, dtype=np.float64),
                                      np.asarray(s_rc, dtype=np.float64))

    # ---- analysis ----
    centered = {a: res[a]["centered_acc"] for a in res}
    best_a = max(centered, key=centered.get)
    best_c = centered[best_a]
    a0 = res["0.00"]["centered_acc"]   # L4-only
    a1 = res["1.00"]["centered_acc"]   # L1-only
    a_half = res["0.50"]["centered_acc"]  # mean(L1,L4) consistency check

    near_max = [a for a in res if best_c - centered[a] <= NEAR_MAX_BAND]
    # lowest-bias alpha among near-optimal -> best learned-init seed
    seed_a = min(near_max, key=lambda a: res[a]["bias_to_signal"])

    band = [a for a in res if FLAT_BAND_LO - 1e-9 <= float(a) <= FLAT_BAND_HI + 1e-9]
    band_spread = max(centered[a] for a in band) - min(centered[a] for a in band)

    endpoint_optimal = best_a in ("0.00", "1.00")
    meaningful_peak = (best_c - MEAN_REF_CENTERED) >= MEANINGFUL_GAIN and not endpoint_optimal

    degen = {a: (res[a]["strict_sign_reversal_rate"] < DEGEN_SIGN_REV_MAX
                 and res[a]["pos_rate_cr"] > DEGEN_POS_RATE_MIN) for a in res}
    any_degen = any(degen[a] for a in near_max)

    if endpoint_optimal:
        verdict = ("STOP — endpoint-optimal. Best centered acc is at "
                   f"alpha={best_a} ({'L4-only' if best_a=='0.00' else 'L1-only'}). "
                   "mean(L1,L4)=0.620 may be a sampling fluke; re-examine the v10 "
                   "bipartite architectural implication before training.")
        arm_order = None
    elif any_degen:
        verdict = ("Inconclusive — degeneracy near the optimum (constant-output "
                   "flip behaviour). Inspect before locking the arm design.")
        arm_order = None
    elif meaningful_peak:
        verdict = (f"Weighted-mean wins. Peak centered acc {best_c:.4f} at "
                   f"alpha={best_a} beats mean(L1,L4) {MEAN_REF_CENTERED:.3f} by "
                   f"{(best_c-MEAN_REF_CENTERED)*100:.2f} pp. Fusion gain is largely "
                   "about mixing weight; a 2048-dim parameterization captures most "
                   "of it.")
        arm_order = ["weighted_mean", "concat", "diff", "L4_only_control"]
    else:
        verdict = (f"Flat / no meaningful peak. Best centered acc {best_c:.4f} at "
                   f"alpha={best_a} does not beat mean(L1,L4) {MEAN_REF_CENTERED:.3f} "
                   f"by >= {MEANINGFUL_GAIN*100:.1f} pp. The win is NOT about mixing "
                   "weight; any further gain needs the head to see L1 and L4 as "
                   "separate slots -> concat/diff do structurally different work.")
        arm_order = ["concat", "diff", "weighted_mean", "L4_only_control"]

    # ---- table ----
    print("\n=== L1<->L4 weighted-mean alpha sweep (frozen head, 200 HH pairs) ===")
    print("fusion = alpha*L1 + (1-alpha)*L4   |   alpha=0 -> L4-only, alpha=1 -> L1-only")
    hdr = f"{'alpha':>6} | {'canon':>6} | {'center':>6} | {'bias/sig':>8} | {'pos':>5} | {'flip':>5}"
    print(hdr)
    print("-" * len(hdr))
    for a in res:
        m = res[a]
        mark = ""
        if a == best_a:
            mark += "  <- max centered"
        if a == seed_a and seed_a != best_a:
            mark += "  <- low-bias seed"
        if a == "0.50":
            mark += "  (== mean ref 0.620)"
        print(f"{a:>6} | {m['canonical_acc']:.4f} | {m['centered_acc']:.4f} | "
              f"{m['bias_to_signal']:8.3f} | {m['pos_rate_cr']:.3f} | "
              f"{m['strict_sign_reversal_rate']:.3f}{mark}")

    print(f"\nL4-only (a=0.00) centered={a0:.4f}   "
          f"L1-only (a=1.00) centered={a1:.4f}   "
          f"mean (a=0.50) centered={a_half:.4f}  [ref 0.620 sanity]")
    print(f"Flat-band [{FLAT_BAND_LO},{FLAT_BAND_HI}] centered spread = {band_spread*100:.2f} pp "
          f"({'broad/flat optimum' if band_spread < 0.01 else 'has structure'})")
    print(f"Near-optimal alphas (within {NEAR_MAX_BAND*100:.1f} pp of max): {near_max}")
    print(f"Recommended learned-init seed alpha = {seed_a} "
          f"(lowest bias_to_signal={res[seed_a]['bias_to_signal']:.3f} among near-optimal, "
          f"centered={res[seed_a]['centered_acc']:.4f})")
    print(f"\nVERDICT: {verdict}")
    if arm_order:
        print(f"Tonight's arm priority order: {' > '.join(arm_order)}")

    out = {
        "source_states": STATES_PT,
        "head_checkpoint": HEAD_CKPT,
        "n_pairs": len(packs),
        "protocol": "swap-balanced, zero-shot, fp32, no model forward",
        "fusion": "alpha*L1 + (1-alpha)*L4 replicated into all 4 head slots",
        "bias_to_signal_formula": "abs(mean(s_cr + s_rc)) / std(s_cr - s_rc)",
        "alphas": ALPHAS,
        "metrics_per_alpha": res,
        "analysis": {
            "best_alpha": best_a,
            "best_centered_acc": best_c,
            "L4_only_centered": a0,
            "L1_only_centered": a1,
            "mean_centered": a_half,
            "mean_ref_centered": MEAN_REF_CENTERED,
            "peak_gain_over_mean_pp": (best_c - MEAN_REF_CENTERED) * 100.0,
            "near_optimal_alphas": near_max,
            "flat_band": [FLAT_BAND_LO, FLAT_BAND_HI],
            "flat_band_spread_pp": band_spread * 100.0,
            "recommended_seed_alpha": seed_a,
            "recommended_seed_bias_to_signal": res[seed_a]["bias_to_signal"],
            "endpoint_optimal": endpoint_optimal,
            "meaningful_peak": meaningful_peak,
            "degeneracy_near_optimum": any_degen,
        },
        "verdict": verdict,
        "arm_priority_order": arm_order,
        "wall_seconds": round(time.time() - t0, 2),
    }
    op = PROJECT_ROOT / OUT_JSON
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_JSON}  ({out['wall_seconds']}s)")


if __name__ == "__main__":
    main()
