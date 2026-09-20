"""Analyze Branch Generator v1 geometry and mechanism diagnostics."""
from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any

from bg_branch_generator_v1_common import (
    BASIS_BANK_PT,
    GEOMETRY_JSON,
    GEOMETRY_MD,
    candidate_group_summary,
    config_vector_from_row,
    cosine,
    ensure_bgv1_root,
    group_rows,
    load_generator_rows,
    load_pt,
    md_table,
    primary_safe_generator_row,
    rate,
    rel,
    row_reward,
    tensor_stats,
    write_json,
    write_md,
)


def corr(xs: list[float], ys: list[float]) -> float:
    vals = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(vals) < 2:
        return float("nan")
    mx = sum(x for x, _ in vals) / len(vals)
    my = sum(y for _, y in vals) / len(vals)
    num = sum((x - mx) * (y - my) for x, y in vals)
    dx = math.sqrt(sum((x - mx) ** 2 for x, _ in vals))
    dy = math.sqrt(sum((y - my) ** 2 for _, y in vals))
    return num / max(dx * dy, 1e-12)


def group_geometry_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for gid, vals in group_rows([row for row in rows if primary_safe_generator_row(row)]).items():
        if len(vals) < 2:
            continue
        dists = [float(row.get("branch_hidden_distance_from_clean") or 0.0) for row in vals if row.get("branch_hidden_distance_from_clean") is not None]
        tap_scores = [float(row.get("old_frozen_tap_score", row.get("tap_margin_sum", 0.0)) or 0.0) for row in vals]
        rewards = [row_reward(row) for row in vals]
        out.append(
            {
                "branch_group_id": gid,
                "split": vals[0].get("split"),
                "task_id": vals[0].get("task_id"),
                "generator_method": vals[0].get("generator_method"),
                "branch_point": vals[0].get("branch_point"),
                "alpha_bucket": vals[0].get("alpha_bucket"),
                "delta_family": vals[0].get("primary_delta_family") or vals[0].get("delta_family"),
                "mean_distance_from_clean": sum(dists) / max(len(dists), 1),
                "max_distance_from_clean": max(dists) if dists else 0.0,
                "tap_score_spread": max(tap_scores) - min(tap_scores) if tap_scores else 0.0,
                "reward_spread": max(rewards) - min(rewards) if rewards else 0.0,
                "off_manifold_warnings": sum(1 for row in vals if row.get("off_manifold_warning")),
            }
        )
    return out


def basis_alignment(bank: dict[str, Any]) -> dict[str, Any]:
    directions = list(bank.get("directions") or [])
    refs = defaultdict(list)
    for row in directions:
        if row.get("family") in {"old_tap_aligned", "v1_tap_aligned", "v2_tap_aligned", "v3_tap_aligned", "v4_tap_aligned", "salvage_tap_aligned"}:
            refs[str(row.get("family"))].append(row.get("tensor"))
    out = {}
    for family, items in refs.items():
        vals = []
        for other_family, other_items in refs.items():
            if family >= other_family:
                continue
            cosines = [abs(cosine(a, b)) for a in items for b in other_items if a is not None and b is not None and int(a.numel()) == int(b.numel())]
            finite = [x for x in cosines if isinstance(x, float) and math.isfinite(x)]
            if finite:
                vals.append({"pair": f"{family}__{other_family}", "max_abs_cosine": max(finite), "mean_abs_cosine": sum(finite) / len(finite)})
        out[family] = vals
    return out


def verdict_for(metrics: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "INCONCLUSIVE"
    if float(metrics.get("distance_reward_correlation") or 0.0) > 0.15 and float(metrics.get("off_manifold_rate") or 0.0) < 0.10:
        return "STRUCTURED_DIVERSITY"
    if float(metrics.get("tap_spread_reward_correlation") or 0.0) > 0.10:
        return "OLD_GEOMETRY_CONFIRMED"
    if float(metrics.get("off_manifold_rate") or 0.0) > 0.25:
        return "NOISY_OR_UNSTABLE"
    return "INCONCLUSIVE"


def main() -> int:
    started = time.time()
    ensure_bgv1_root()
    rows = load_generator_rows()
    bank = load_pt(BASIS_BANK_PT, {}) or {}
    geometry_rows = group_geometry_rows(rows)
    xs_dist = [float(row["mean_distance_from_clean"]) for row in geometry_rows]
    xs_tap = [float(row["tap_score_spread"]) for row in geometry_rows]
    ys = [float(row["reward_spread"]) for row in geometry_rows]
    off = sum(int(row["off_manifold_warnings"] > 0) for row in geometry_rows)
    metrics = {
        "group_count": len(geometry_rows),
        "distance_reward_correlation": corr(xs_dist, ys),
        "tap_spread_reward_correlation": corr(xs_tap, ys),
        "off_manifold_rate": off / max(len(geometry_rows), 1),
        "primary_group_metrics": candidate_group_summary({gid: vals for gid, vals in group_rows([row for row in rows if primary_safe_generator_row(row)]).items() if len(vals) >= 2}) if rows else {},
    }
    align = basis_alignment(bank)
    verdict = verdict_for(metrics, geometry_rows)
    payload = {
        "BG_BRANCH_GENERATOR_V1_GEOMETRY_VERDICT": verdict,
        "verdict": verdict,
        "metrics": metrics,
        "group_geometry_rows": geometry_rows,
        "basis_alignment": align,
        "basis_bank": rel(BASIS_BANK_PT),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(GEOMETRY_JSON, payload)
    display = [{**row, "mean_distance_from_clean": rate(row["mean_distance_from_clean"]), "tap_score_spread": rate(row["tap_score_spread"]), "reward_spread": rate(row["reward_spread"])} for row in geometry_rows[:120]]
    lines = [
        "# Branch Generator V1 Geometry",
        "",
        f"BG_BRANCH_GENERATOR_V1_GEOMETRY_VERDICT = {verdict}",
        "",
        f"- metrics: `{metrics}`",
        f"- basis_alignment_keys: `{list(align.keys())}`",
        "",
        "## Group Geometry",
        "",
    ]
    lines.extend(md_table(display, ["split", "task_id", "generator_method", "branch_point", "alpha_bucket", "delta_family", "mean_distance_from_clean", "tap_score_spread", "reward_spread", "off_manifold_warnings"]))
    write_md(GEOMETRY_MD, lines)
    print(f"BG_BRANCH_GENERATOR_V1_GEOMETRY_VERDICT = {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
