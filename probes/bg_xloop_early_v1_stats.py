"""Cross-loop early-layer v1 — task-clustered bootstrap statistics and verdict labels.

Clusters = task_uid within domain; macro = unweighted mean of domain means; 10,000 draws,
seed 20260720; identical draws reused for every paired comparison (cells share the same
heldout groups by construction).
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bg_xloop_early_v1_common as X

EARLY_MARGIN = 0.03  # predeclared: within 0.03 macro top-1


def load_pred_vectors():
    """Returns {series_name: {domain: {task_uid: [group top1 values]}}} plus chance series."""
    series: dict[str, dict] = {}

    def add(name, rows, field="top1_oracle"):
        d: dict[str, dict] = defaultdict(lambda: defaultdict(list))
        for r in rows:
            d[r["domain"]][r["task_uid"]].append(float(r[field]))
        series[name] = {dom: dict(t) for dom, t in d.items()}

    ref_rows = None
    for (layer, loop) in X.REFIT_CELLS:
        cn = X.cell_name(layer, loop)
        rows = X.read_json(X.PRED_ROOT / f"refit_{cn}.json")
        add(f"refit:{cn}", rows)
        if ref_rows is None:
            ref_rows = rows
    for (layer, u, v) in X.TRANSFERS:
        rows = X.read_json(X.PRED_ROOT / f"transfer_{layer}_L{u}_to_L{v}.json")
        add(f"transfer:{layer}:L{u}->L{v}", rows)
    add("chance", ref_rows, field="chance_top1")
    return series


def bootstrap(series: dict, draws: int, seed: int):
    """Clustered bootstrap. Returns (point, samples) per series, deltas computed by caller."""
    rng = np.random.default_rng(seed)
    domains = sorted(next(iter(series.values())).keys())
    # common task list per domain (identical across series by construction; assert)
    tasks_by_dom = {}
    for dom in domains:
        tasks = sorted(series[next(iter(series))][dom].keys())
        for name, s in series.items():
            assert sorted(s[dom].keys()) == tasks, f"task mismatch in {name}/{dom}"
        tasks_by_dom[dom] = tasks

    # per-series per-domain arrays of (task_sum, task_count)
    sums, counts = {}, {}
    for name, s in series.items():
        sums[name], counts[name] = {}, {}
        for dom in domains:
            sums[name][dom] = np.array([np.sum(s[dom][t]) for t in tasks_by_dom[dom]])
            counts[name][dom] = np.array([len(s[dom][t]) for t in tasks_by_dom[dom]])

    # identical draw indices for all series (paired inference)
    idx = {dom: rng.integers(0, len(tasks_by_dom[dom]), size=(draws, len(tasks_by_dom[dom])))
           for dom in domains}

    samples = {}
    points = {}
    for name in series:
        dom_means = []
        dom_points = []
        for dom in domains:
            ts, tc = sums[name][dom], counts[name][dom]
            drawn_s = ts[idx[dom]].sum(axis=1)
            drawn_c = tc[idx[dom]].sum(axis=1)
            dom_means.append(drawn_s / np.maximum(drawn_c, 1))
            dom_points.append(ts.sum() / tc.sum())
        samples[name] = np.mean(dom_means, axis=0)
        points[name] = float(np.mean(dom_points))
    return points, samples


def ci(v: np.ndarray) -> list[float]:
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def main() -> int:
    t0 = time.time()
    series = load_pred_vectors()
    points, samples = bootstrap(series, X.BOOT_DRAWS, X.BOOT_SEED)
    res = X.read_json(X.RUN_ROOT / "train_eval_results.json")

    out = {"draws": X.BOOT_DRAWS, "seed": X.BOOT_SEED, "cluster": "task_uid within domain",
           "macro": {}, "paired_deltas": {}, "verdict_labels": {}}
    for name in sorted(series):
        out["macro"][name] = {"point": round(points[name], 4), "ci95": ci(samples[name])}

    def paired(a: str, b: str) -> dict:
        d = samples[a] - samples[b]
        return {"point": round(points[a] - points[b], 4), "ci95": ci(d),
                "p_gt_zero": round(float((d > 0).mean()), 4)}

    # cell vs chance, cell vs L4_24 reference, and required reference comparisons
    for (layer, loop) in X.REFIT_CELLS:
        cn = X.cell_name(layer, loop)
        out["paired_deltas"][f"refit:{cn} - chance"] = paired(f"refit:{cn}", "chance")
        if cn != "24_L4":
            out["paired_deltas"][f"refit:{cn} - refit:24_L4"] = paired(f"refit:{cn}", "refit:24_L4")
    for cn in ("8_L4", "16_L4", "16_L3"):
        out["paired_deltas"][f"refit:{cn} - refit:36_L4"] = paired(f"refit:{cn}", "refit:36_L4")
    # best later-loop early-layer cell (layers 8/16, loops 3-4) vs L4_36
    early_cells = [X.cell_name(l, u) for l in (8, 16) for u in (3, 4)]
    best_early = max(early_cells, key=lambda cn: points[f"refit:{cn}"])
    out["best_late_loop_early_layer_cell"] = best_early
    out["paired_deltas"][f"refit:{best_early} - refit:36_L4 (best-early)"] = paired(f"refit:{best_early}", "refit:36_L4")
    # transfers vs target refit and vs chance
    for (layer, u, v) in X.TRANSFERS:
        tn = f"transfer:{layer}:L{u}->L{v}"
        tgt = f"refit:{X.cell_name(layer, v)}"
        out["paired_deltas"][f"{tn} - {tgt}"] = paired(tn, tgt)
        out["paired_deltas"][f"{tn} - chance"] = paired(tn, "chance")
    # loop trend within each layer: L4 - L1 local refit
    for layer in (8, 16, 24):
        out["paired_deltas"][f"refit:{layer}_L4 - refit:{layer}_L1"] = paired(
            f"refit:{X.cell_name(layer, 4)}", f"refit:{X.cell_name(layer, 1)}")

    # ---------------- predeclared labels
    labels = {}
    for (layer, loop) in X.REFIT_CELLS:
        cn = X.cell_name(layer, loop)
        above_chance = out["paired_deltas"][f"refit:{cn} - chance"]["ci95"][0] > 0
        if cn == "24_L4":
            labels[cn] = {"above_chance_ci": above_chance}
            continue
        near_ref = points[f"refit:{cn}"] >= points["refit:24_L4"] - EARLY_MARGIN
        lab = {"above_chance_ci": above_chance, "within_margin_of_24_L4": near_ref}
        if layer in (8, 16) and loop >= 2:
            lab["EARLY_READABLE"] = bool(above_chance and near_ref)
            if lab["EARLY_READABLE"] and loop >= 3:
                lab["EARLY_ACTION_SURFACE_CANDIDATE_READOUT_ONLY"] = True
        labels[cn] = lab
    for (layer, u, v) in X.TRANSFERS:
        tn = f"transfer:{layer}:L{u}->L{v}"
        tr = res["transfers"][f"{layer}:L{u}->L{v}"]
        above_chance = out["paired_deltas"][f"{tn} - chance"]["ci95"][0] > 0
        near_refit = points[tn] >= points[f"refit:{X.cell_name(layer, v)}"] - EARLY_MARGIN
        corr = tr["score_spearman_vs_target_refit"]
        stable = bool(above_chance and near_refit and isinstance(corr, float) and corr > 0)
        refit_ok = out["paired_deltas"][f"refit:{X.cell_name(layer, v)} - chance"]["ci95"][0] > 0
        labels[tn] = {"above_chance_ci": above_chance, "within_margin_of_target_refit": near_refit,
                      "spearman_vs_target_refit": corr,
                      "DIRECTION_STABLE": stable,
                      "ROTATED_BUT_READABLE": bool(refit_ok and not near_refit)}
    out["verdict_labels"] = labels
    out["elapsed_seconds"] = round(time.time() - t0, 1)
    X.write_json(X.RUN_ROOT / "bootstrap_stats.json", out)
    print("bootstrap stats written")
    for name in sorted(out["macro"]):
        m = out["macro"][name]
        print(f"  {name}: {m['point']:.4f} CI{m['ci95']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
