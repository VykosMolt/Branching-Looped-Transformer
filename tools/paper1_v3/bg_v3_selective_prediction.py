"""V3 selective prediction (abstention) analysis -- zero-GPU, preserved data only.

Question: do hidden-state pre-answer scores buy a better selective-risk profile
than shortcut scores — i.e., at a fixed answer-coverage, higher accuracy on the
answered subset? This converts the readout into a frozen, deployable
reliability gain (calibrated abstention) without touching the model.

Data (read-only, preserved):
  - Horizon pooled cohort: raw_predictions_heldout_v3.json (pooled_4sc and the
    adversarial pooled_5sc arms; scores for shortcut / hidden / combined).
  - GSM8K: preanswer_oof_predictions.csv from the sealed 2026-07-10 hardening
    check (task-grouped out-of-fold scores; shortcut composite and hidden).

Endpoints (fixed before running):
  - Selective accuracy at coverage c in {0.50..1.00, step 0.05}: answer the
    top-c fraction of candidates by score, accuracy over the answered subset.
  - Delta(combined - shortcut) at each c, and Delta AUARC (area under the
    accuracy-coverage curve), with a paired task-clustered bootstrap
    (2000 rounds, seed 20260724; identical draws for both curves).
No claim of improved raw accuracy is made or implied: coverage=1.0 recovers
the base rate by construction. The claim under test is the risk-coverage
trade-off.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = PROJECT_ROOT / "artifacts/reports/selective_prediction_v3_20260726"
SEED = 20260724
COVERAGES = [round(c, 2) for c in np.arange(0.50, 1.0001, 0.05)]
ROUNDS = 2000


def selective_curve(scores: np.ndarray, y: np.ndarray, coverages) -> list[float]:
    order = np.argsort(-scores, kind="stable")
    y_sorted = y[order]
    n = len(y)
    out = []
    for c in coverages:
        k = max(1, int(round(c * n)))
        out.append(float(y_sorted[:k].mean()))
    return out


def auarc(curve: list[float], coverages) -> float:
    return float(np.trapezoid(curve, coverages))


def clustered_delta(scores_a, scores_b, y, tasks, coverages):
    """Paired task-clustered bootstrap for curve deltas and AUARC delta."""
    rng = np.random.default_rng(SEED)
    by_task = defaultdict(list)
    for i, t in enumerate(tasks):
        by_task[t].append(i)
    task_list = sorted(by_task)
    point_a = selective_curve(scores_a, y, coverages)
    point_b = selective_curve(scores_b, y, coverages)
    deltas = np.array(point_a) - np.array(point_b)
    d_auarc = auarc(point_a, coverages) - auarc(point_b, coverages)
    samples = np.zeros((ROUNDS, len(coverages)))
    samples_auarc = np.zeros(ROUNDS)
    for r in range(ROUNDS):
        picked = rng.choice(len(task_list), size=len(task_list), replace=True)
        idx = np.concatenate([by_task[task_list[i]] for i in picked])
        ya, sa, sb = y[idx], scores_a[idx], scores_b[idx]
        if ya.mean() in (0.0, 1.0):
            samples[r] = np.nan
            samples_auarc[r] = np.nan
            continue
        ca = selective_curve(sa, ya, coverages)
        cb = selective_curve(sb, ya, coverages)
        samples[r] = np.array(ca) - np.array(cb)
        samples_auarc[r] = auarc(ca, coverages) - auarc(cb, coverages)
    lo, hi = np.nanpercentile(samples, [2.5, 97.5], axis=0)
    alo, ahi = np.nanpercentile(samples_auarc, [2.5, 97.5])
    return {
        "curve_a": [round(v, 4) for v in point_a],
        "curve_b": [round(v, 4) for v in point_b],
        "delta": [round(float(v), 4) for v in deltas],
        "delta_ci_lo": [round(float(v), 4) for v in lo],
        "delta_ci_hi": [round(float(v), 4) for v in hi],
        "delta_excludes_zero": [bool(l > 0 or h < 0) for l, h in zip(lo, hi)],
        "auarc_a": round(auarc(point_a, coverages), 4),
        "auarc_b": round(auarc(point_b, coverages), 4),
        "delta_auarc": round(d_auarc, 4),
        "delta_auarc_ci": [round(float(alo), 4), round(float(ahi), 4)],
        "delta_auarc_excludes_zero": bool(alo > 0 or ahi < 0),
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = {"seed": SEED, "rounds": ROUNDS, "coverages": COVERAGES,
               "definition": "answer top-c fraction by score; selective accuracy "
                             "over answered subset; paired task-clustered bootstrap"}

    # --- Horizon pooled (both shortcut composites) ---------------------------
    hp = json.load(open(PROJECT_ROOT / "artifacts/reports/horizon_power_v3_20260726/raw_predictions_heldout_v3.json"))
    for arm in ("pooled_4sc", "pooled_5sc", "new_only_5sc"):
        a = hp[arm]
        y = np.array(a["y"], dtype=float)
        tasks = a["tasks"]
        comb = np.array(a["scores"]["combined"], dtype=float)
        sc = np.array(a["scores"]["shortcut"], dtype=float)
        results[f"horizon_{arm}"] = {
            "n": len(y), "n_tasks": len(set(tasks)), "base_rate": round(float(y.mean()), 4),
            "combined_vs_shortcut": clustered_delta(comb, sc, y, tasks, COVERAGES),
        }
        print(f"[selective] horizon {arm}: base {y.mean():.3f} "
              f"dAUARC {results[f'horizon_{arm}']['combined_vs_shortcut']['delta_auarc']:+.4f} "
              f"CI {results[f'horizon_{arm}']['combined_vs_shortcut']['delta_auarc_ci']} "
              f"excl0={results[f'horizon_{arm}']['combined_vs_shortcut']['delta_auarc_excludes_zero']}")

    # --- GSM8K (preserved out-of-fold predictions) ----------------------------
    rows = list(csv.DictReader(open(PROJECT_ROOT / "artifacts/reports/paper_verification/fable_hardening_checks_20260710_175017/preanswer_oof_predictions.csv")))
    y = np.array([float(r["label"]) for r in rows])
    tasks = [r["task_id"] for r in rows]
    hid = np.array([float(r["hidden_all_score"]) for r in rows])
    sc = np.array([float(r["length_logprob_score"]) for r in rows])
    results["gsm8k_oof"] = {
        "n": len(y), "n_tasks": len(set(tasks)), "base_rate": round(float(y.mean()), 4),
        "note": "out-of-fold scores; per-row combined arm not preserved, so the "
                "comparison is hidden-alone vs shortcut composite",
        "hidden_vs_shortcut": clustered_delta(hid, sc, y, tasks, COVERAGES),
    }
    print(f"[selective] gsm8k: base {y.mean():.3f} "
          f"dAUARC {results['gsm8k_oof']['hidden_vs_shortcut']['delta_auarc']:+.4f} "
          f"CI {results['gsm8k_oof']['hidden_vs_shortcut']['delta_auarc_ci']} "
          f"excl0={results['gsm8k_oof']['hidden_vs_shortcut']['delta_auarc_excludes_zero']}")

    (OUT_ROOT / "selective_prediction_results.json").write_text(
        json.dumps(results, indent=2) + "\n")
    print("[selective] written:", OUT_ROOT / "selective_prediction_results.json")


if __name__ == "__main__":
    main()
