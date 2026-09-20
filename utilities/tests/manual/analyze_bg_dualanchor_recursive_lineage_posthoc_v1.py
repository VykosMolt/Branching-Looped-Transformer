"""Posthoc tables for DualAnchor recursive lineage probe v1."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from bg_hidden_origin_tap_common import PROBE_ROOT, md_table, rel
from run_bg_layer_native_two_tap_readiness_v1 import write_csv, write_json, write_md


ROOT = PROBE_ROOT / "bg_dualanchor_recursive_lineage_probe_v1_2026-05-31"
ROWS_CSV = ROOT / "recursive_lineage_rows.csv"
OUT_JSON = ROOT / "recursive_lineage_posthoc.json"
OUT_MD = ROOT / "recursive_lineage_posthoc.md"
OUT_CSV = ROOT / "recursive_lineage_posthoc_rows.csv"


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def finite_mean(vals: Iterable[Any]) -> float:
    xs = [finite_float(v) for v in vals]
    xs = [x for x in xs if math.isfinite(x)]
    return float(mean(xs)) if xs else float("nan")


def main() -> int:
    rows = list(csv.DictReader(ROWS_CSV.open()))
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    task_rows = []
    for task_id, vals in sorted(by_task.items()):
        clean_rewards = [finite_float(row.get("reward")) for row in vals if int(row.get("perturb_count") or 0) == 0]
        clean_best = max(clean_rewards) if clean_rewards else float("nan")
        by_depth = {}
        for depth in (1, 2, 3):
            rewards = [finite_float(row.get("reward")) for row in vals if int(row.get("perturb_count") or 0) == depth]
            by_depth[depth] = max(rewards) if rewards else float("nan")
        best_pert = max((v for v in by_depth.values() if math.isfinite(v)), default=float("nan"))
        best_depths = [depth for depth, value in by_depth.items() if math.isfinite(value) and value == best_pert]
        task_rows.append(
            {
                "task_id": task_id,
                "domain": vals[0].get("domain"),
                "clean_best": clean_best,
                "depth1_best": by_depth[1],
                "depth2_best": by_depth[2],
                "depth3_best": by_depth[3],
                "best_perturbed": best_pert,
                "best_perturbed_gt_clean": 1.0 if best_pert > clean_best else 0.0,
                "best_perturbed_eq_clean": 1.0 if best_pert == clean_best else 0.0,
                "best_perturbed_lt_clean": 1.0 if best_pert < clean_best else 0.0,
                "best_depths": ",".join(str(x) for x in best_depths),
            }
        )
    depth_rows = []
    for depth in (1, 2, 3):
        depth_rows.append(
            {
                "perturb_count": depth,
                "best_gt_clean_rate": finite_mean(1.0 if row[f"depth{depth}_best"] > row["clean_best"] else 0.0 for row in task_rows),
                "best_eq_clean_rate": finite_mean(1.0 if row[f"depth{depth}_best"] == row["clean_best"] else 0.0 for row in task_rows),
                "best_lt_clean_rate": finite_mean(1.0 if row[f"depth{depth}_best"] < row["clean_best"] else 0.0 for row in task_rows),
                "mean_best_reward": finite_mean(row[f"depth{depth}_best"] for row in task_rows),
            }
        )
    payload = {
        "task_count": len(task_rows),
        "best_perturbed_gt_clean_rate": finite_mean(row["best_perturbed_gt_clean"] for row in task_rows),
        "best_perturbed_eq_clean_rate": finite_mean(row["best_perturbed_eq_clean"] for row in task_rows),
        "best_perturbed_lt_clean_rate": finite_mean(row["best_perturbed_lt_clean"] for row in task_rows),
        "best_depth_counts": dict(Counter(depth for row in task_rows for depth in str(row["best_depths"]).split(",") if depth)),
        "task_rows": task_rows,
        "depth_rows": depth_rows,
    }
    write_json(OUT_JSON, payload)
    write_csv(OUT_CSV, task_rows + depth_rows)
    lines = [
        "# DualAnchor Recursive Lineage Posthoc v1",
        "",
        f"- task_count: `{payload['task_count']}`",
        f"- best perturbed > clean: `{payload['best_perturbed_gt_clean_rate']}`",
        f"- best perturbed = clean: `{payload['best_perturbed_eq_clean_rate']}`",
        f"- best perturbed < clean: `{payload['best_perturbed_lt_clean_rate']}`",
        f"- best depth counts: `{payload['best_depth_counts']}`",
        "",
        "## Task Maxima",
        "",
    ]
    lines.extend(md_table(task_rows, ["task_id", "domain", "clean_best", "depth1_best", "depth2_best", "depth3_best", "best_perturbed", "best_perturbed_gt_clean", "best_perturbed_eq_clean", "best_depths"]))
    lines.extend(["", "## Depth vs Clean", ""])
    lines.extend(md_table(depth_rows, ["perturb_count", "best_gt_clean_rate", "best_eq_clean_rate", "best_lt_clean_rate", "mean_best_reward"]))
    lines.extend(["", "## Files", "", f"- json: `{rel(OUT_JSON)}`", f"- csv: `{rel(OUT_CSV)}`"])
    write_md(OUT_MD, lines)
    print(f"posthoc = {rel(OUT_MD)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
