"""Paired bootstrap over held-out source pairs for the powered HH probes.

All three backbones use the same pair-selection and split seed, so the eval
source pairs are identical across backbones and the comparison is paired.
Resampling the per-pair difference is strictly tighter than comparing the
marginal per-backbone intervals.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import torch

OUT = Path("artifacts/reports/paper_verification/powered_clean_probe_and_corecontent_20260711")
PAIRS = 40000
SEED = 20260711
DRAWS = 10000
BACKBONES = ["base", "thinking", "rltt"]


def correctness():
    """Per-backbone boolean correctness on the shared held-out pairs."""
    corr, ev_ref = {}, None
    for b in BACKBONES:
        blob = torch.load(OUT / f"hh_{b}_{PAIRS}_pair_disjoint_probe.pt",
                          map_location="cpu", weights_only=False)
        ev, w = blob["eval_positions"], blob["weight"]
        if ev_ref is None:
            ev_ref = ev
        elif not np.array_equal(ev, ev_ref):
            raise RuntimeError(f"{b} eval positions differ; comparison is not paired")

        files = sorted((OUT / f"hh_{b}_{PAIRS}_diffs").glob("diffs_*.pt"))
        diffs = torch.cat([torch.load(p, map_location="cpu", weights_only=False)["diffs"]
                           for p in files]).float()
        if len(diffs) != PAIRS:
            raise RuntimeError(f"{b}: expected {PAIRS} diffs, found {len(diffs)}")
        corr[b] = (diffs[ev] @ w > 0).numpy()
    return corr, ev_ref


def main():
    corr, ev = correctness()
    n = len(ev)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(DRAWS, n))

    result = {
        "n_eval_pairs": n,
        "bootstrap_draws": DRAWS,
        "seed": SEED,
        "paired": True,
        "accuracy": {b: float(corr[b].mean()) for b in BACKBONES},
        "comparisons": {},
    }

    for a, b in itertools.combinations(BACKBONES, 2):
        d = corr[b].astype(np.float64) - corr[a].astype(np.float64)
        boots = d[idx].mean(axis=1)
        lo, hi = np.quantile(boots, [0.025, 0.975])
        # two-sided bootstrap p-value: mass on the far side of zero, doubled
        p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
        result["comparisons"][f"{b}_minus_{a}"] = {
            "delta": float(d.mean()),
            "ci95": [float(lo), float(hi)],
            "p_two_sided": float(min(p, 1.0)),
            "significant_at_95": bool(lo > 0 or hi < 0),
            "n_discordant": int((d != 0).sum()),
        }

    (OUT / "powered_hh_paired_bootstrap.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
