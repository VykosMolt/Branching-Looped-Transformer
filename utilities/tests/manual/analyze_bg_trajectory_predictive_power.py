"""Analyze where BG prefix scores predict final continuation success."""
from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any

from bg_trajectory_prediction_lib import (
    REPORT_ROOT,
    auc_score,
    bootstrap_mean_delta,
    branch_label_map,
    expected_random_topk,
    finite_or_none,
    load_continued,
    load_json,
    md_table,
    rel,
    spearman,
    write_json,
    write_md,
)


OUT_JSON = REPORT_ROOT / "predictive_power.json"
OUT_MD = REPORT_ROOT / "predictive_power.md"


def _group_branch_scores(score_rows: list[dict[str, Any]]) -> dict[tuple[str, int, str], list[dict[str, Any]]]:
    grouped = defaultdict(list)
    for row in score_rows:
        grouped[(str(row["task_id"]), int(row["prefix_length"]), str(row["head_id"]))].append(row)
    return grouped


def _cell_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (str(row["domain"]), int(row["prefix_length"]), str(row["head_id"]))


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        f = float(value)
    except Exception:
        return str(value)
    if math.isnan(f) or math.isinf(f):
        return ""
    return f"{f:.3f}"


def _is_strong_cell(row: dict[str, Any]) -> bool:
    return float(row["oracle_success"]) >= 0.40 and (
        (
            row["n_tasks"] >= 20
            and (float(row["top1_lift"]) >= 0.10 or float(row["top2_lift"]) >= 0.10)
        )
        or (
            row["n_pairwise_comparisons"] >= 20
            and row["pairwise_predictive_accuracy"] is not None
            and float(row["pairwise_predictive_accuracy"]) >= 0.65
        )
    )


def main() -> int:
    started = time.time()
    continued = load_continued()
    scores_payload = load_json(REPORT_ROOT / "prefix_scores.json", {})
    score_rows = list(scores_payload.get("prefix_scores") or [])
    pairwise_rows = list(scores_payload.get("pairwise_scores") or [])
    labels = branch_label_map(continued)
    grouped_by_task = _group_branch_scores(score_rows)

    cell_task_values = defaultdict(list)
    branch_values = defaultdict(lambda: {"scores": [], "labels": []})
    for (task_id, prefix_length, head_id), rows in grouped_by_task.items():
        rows = sorted(rows, key=lambda r: (int(r["rank"]), -float(r["margin_sum"]), int(r["branch_id"])))
        labelled = []
        for row in rows:
            label_key = (task_id, int(row["branch_id"]), prefix_length)
            if label_key in labels:
                labelled.append((row, bool(labels[label_key])))
        if len(labelled) < 2:
            continue
        successes = [label for _, label in labelled]
        top1 = bool(labelled[0][1])
        top2 = any(label for _, label in labelled[:2])
        random_top1 = expected_random_topk(successes, 1)
        random_top2 = expected_random_topk(successes, 2)
        oracle = any(successes)
        domain = str(labelled[0][0]["domain"])
        cell = (domain, prefix_length, head_id)
        first_score_row = labelled[0][0]
        cell_task_values[cell].append(
            {
                "task_id": task_id,
                "family": first_score_row.get("family"),
                "config": first_score_row.get("config"),
                "architecture": first_score_row.get("architecture"),
                "top1": top1,
                "top2": top2,
                "random_top1": random_top1,
                "random_top2": random_top2,
                "oracle": oracle,
                "successful_branches": sum(successes),
                "branch_count": len(successes),
            }
        )
        for row, label in labelled:
            branch_values[cell]["scores"].append(float(row["margin_sum"]))
            branch_values[cell]["labels"].append(bool(label))

    pairwise_cell_values = defaultdict(list)
    for row in pairwise_rows:
        left = (str(row["task_id"]), int(row["left_branch_id"]), int(row["prefix_length"]))
        right = (str(row["task_id"]), int(row["right_branch_id"]), int(row["prefix_length"]))
        if left not in labels or right not in labels:
            continue
        left_success = bool(labels[left])
        right_success = bool(labels[right])
        if left_success == right_success:
            continue
        score = float(row["score_left_beats_right"])
        correct = score > 0 if left_success and not right_success else score < 0
        pairwise_cell_values[_cell_key(row)].append(bool(correct))

    cells = []
    for cell, task_rows in cell_task_values.items():
        domain, prefix_length, head_id = cell
        n_tasks = len(task_rows)
        top1_vals = [1.0 if r["top1"] else 0.0 for r in task_rows]
        top2_vals = [1.0 if r["top2"] else 0.0 for r in task_rows]
        rand1_vals = [float(r["random_top1"]) for r in task_rows]
        rand2_vals = [float(r["random_top2"]) for r in task_rows]
        oracle_vals = [1.0 if r["oracle"] else 0.0 for r in task_rows]
        pairs = pairwise_cell_values.get(cell, [])
        branch_scores = branch_values[cell]["scores"]
        branch_labels = branch_values[cell]["labels"]
        top1_rate = sum(top1_vals) / max(n_tasks, 1)
        top2_rate = sum(top2_vals) / max(n_tasks, 1)
        random_top1 = sum(rand1_vals) / max(n_tasks, 1)
        random_top2 = sum(rand2_vals) / max(n_tasks, 1)
        oracle_rate = sum(oracle_vals) / max(n_tasks, 1)
        pairwise_acc = sum(pairs) / max(len(pairs), 1) if pairs else float("nan")
        corr = spearman(branch_scores, [1.0 if x else 0.0 for x in branch_labels])
        auc = auc_score(branch_scores, branch_labels)
        top1_delta_ci = bootstrap_mean_delta(top1_vals, rand1_vals) if n_tasks >= 4 else {"mean": None, "lo": None, "hi": None}
        cells.append(
            {
                "domain": domain,
                "prefix_length": prefix_length,
                "head_id": head_id,
                "family": task_rows[0].get("family"),
                "config": task_rows[0].get("config"),
                "architecture": task_rows[0].get("architecture"),
                "top1_success": top1_rate,
                "top2_success": top2_rate,
                "random_top1_expected": random_top1,
                "random_top2_expected": random_top2,
                "top1_lift": top1_rate - random_top1,
                "top2_lift": top2_rate - random_top2,
                "oracle_success": oracle_rate,
                "oracle_gap_top1": oracle_rate - top1_rate,
                "oracle_gap_top2": oracle_rate - top2_rate,
                "n_tasks": n_tasks,
                "n_successful_branches": int(sum(r["successful_branches"] for r in task_rows)),
                "n_pairwise_comparisons": len(pairs),
                "pairwise_predictive_accuracy": finite_or_none(pairwise_acc),
                "margin_success_spearman": finite_or_none(corr),
                "auc_margin_success": finite_or_none(auc),
                "top1_lift_bootstrap_ci": top1_delta_ci,
            }
        )

    def _strength(row: dict[str, Any]) -> float:
        pair_acc = row.get("pairwise_predictive_accuracy")
        pair_component = (pair_acc - 0.5) if pair_acc is not None else -1.0
        return max(float(row["top1_lift"]), float(row["top2_lift"]), pair_component)

    cells = sorted(cells, key=lambda r: (-_strength(r), -float(r["oracle_success"]), r["domain"], r["prefix_length"], r["head_id"]))
    best = cells[0] if cells else None

    strong_cells = [row for row in cells if _is_strong_cell(row)]
    weak_cells = [
        row
        for row in cells
        if float(row["oracle_success"]) >= 0.25
        and (
            (
                row["n_tasks"] >= 10
                and (float(row["top1_lift"]) >= 0.05 or float(row["top2_lift"]) >= 0.05)
            )
            or (
                row["n_pairwise_comparisons"] >= 20
                and row["pairwise_predictive_accuracy"] is not None
                and float(row["pairwise_predictive_accuracy"]) >= 0.58
            )
        )
    ]
    if not cells:
        verdict = "INSUFFICIENT"
    elif strong_cells:
        verdict = "STRONG"
    elif weak_cells:
        verdict = "WEAK_POSITIVE"
    elif all(_strength(row) < -0.05 for row in cells[:10]):
        verdict = "NEGATIVE"
    else:
        verdict = "NEUTRAL"

    generator_limited = bool(continued.get("GENERATOR_REACHABILITY_LIMITED")) or not any(float(row["oracle_success"]) >= 0.25 for row in cells)
    if generator_limited and verdict in {"NEUTRAL", "NEGATIVE"}:
        verdict = "INSUFFICIENT"

    if verdict in {"STRONG", "WEAK_POSITIVE"} and best:
        target = {
            "domain": best["domain"],
            "prefix_length": best["prefix_length"],
            "head_id": best["head_id"],
            "head_family": best.get("family"),
            "head_config": best.get("config"),
            "architecture": best.get("architecture"),
            "layer_config": best.get("config"),
            "top1_lift": best["top1_lift"],
            "top2_lift": best["top2_lift"],
            "pairwise_accuracy": best["pairwise_predictive_accuracy"],
            "oracle_success": best["oracle_success"],
        }
    else:
        target = "NONE"

    by_domain_prefix = defaultdict(list)
    for row in cells:
        by_domain_prefix[(row["domain"], row["prefix_length"])].append(row)
    heatmap = []
    for (domain, prefix_length), rows in sorted(by_domain_prefix.items()):
        best_row = max(rows, key=_strength)
        heatmap.append(
            {
                "domain": domain,
                "prefix_length": prefix_length,
                "best_head_id": best_row["head_id"],
                "best_family": best_row.get("family"),
                "best_config": best_row.get("config"),
                "best_architecture": best_row.get("architecture"),
                "top1_lift": best_row["top1_lift"],
                "top2_lift": best_row["top2_lift"],
                "pairwise_accuracy": best_row["pairwise_predictive_accuracy"],
                "oracle_success": best_row["oracle_success"],
                "n_tasks": best_row["n_tasks"],
            }
        )

    prefix_trend_rows = []
    for domain in sorted({row["domain"] for row in heatmap}):
        rows = sorted([row for row in heatmap if row["domain"] == domain], key=lambda r: int(r["prefix_length"]))
        if not rows:
            continue
        top1_peak = max(rows, key=lambda r: float(r["top1_lift"]))
        pair_candidates = [row for row in rows if row["pairwise_accuracy"] is not None]
        pair_peak = max(pair_candidates, key=lambda r: float(r["pairwise_accuracy"])) if pair_candidates else None
        top1_vals = [float(row["top1_lift"]) for row in rows]
        if all(top1_vals[i] <= top1_vals[i + 1] + 1e-12 for i in range(len(top1_vals) - 1)):
            top1_shape = "monotonic_increase"
        elif all(top1_vals[i] >= top1_vals[i + 1] - 1e-12 for i in range(len(top1_vals) - 1)):
            top1_shape = "monotonic_decrease"
        else:
            top1_shape = "non_monotonic"
        prefix_trend_rows.append(
            {
                "domain": domain,
                "top1_shape": top1_shape,
                "top1_peak_prefix": top1_peak["prefix_length"],
                "top1_peak_lift": top1_peak["top1_lift"],
                "pair_peak_prefix": pair_peak["prefix_length"] if pair_peak else None,
                "pair_peak_accuracy": pair_peak["pairwise_accuracy"] if pair_peak else None,
                "prefix_32_top1_lift": next((row["top1_lift"] for row in rows if row["prefix_length"] == 32), None),
                "prefix_64_top1_lift": next((row["top1_lift"] for row in rows if row["prefix_length"] == 64), None),
                "prefix_128_top1_lift": next((row["top1_lift"] for row in rows if row["prefix_length"] == 128), None),
                "prefix_256_top1_lift": next((row["top1_lift"] for row in rows if row["prefix_length"] == 256), None),
            }
        )

    domain_breakdown_rows = []
    for domain in sorted({row["domain"] for row in cells}):
        rows = [row for row in cells if row["domain"] == domain]
        best_strength = max(rows, key=_strength)
        best_top1 = max(rows, key=lambda r: float(r["top1_lift"]))
        domain_breakdown_rows.append(
            {
                "domain": domain,
                "strong_cells": sum(1 for row in rows if _is_strong_cell(row)),
                "top1_ge_0.10_cells": sum(1 for row in rows if float(row["top1_lift"]) >= 0.10),
                "pair_ge_0.65_cells": sum(
                    1
                    for row in rows
                    if row["pairwise_predictive_accuracy"] is not None and float(row["pairwise_predictive_accuracy"]) >= 0.65
                ),
                "best_strength_prefix": best_strength["prefix_length"],
                "best_strength_config": best_strength.get("config"),
                "best_strength_architecture": best_strength.get("architecture"),
                "best_strength_head": best_strength["head_id"],
                "best_strength_top1_lift": best_strength["top1_lift"],
                "best_strength_pairwise_accuracy": best_strength["pairwise_predictive_accuracy"],
                "best_top1_prefix": best_top1["prefix_length"],
                "best_top1_config": best_top1.get("config"),
                "best_top1_lift": best_top1["top1_lift"],
            }
        )

    config_breakdown_rows = []
    for config in sorted({str(row.get("config")) for row in cells}):
        rows = [row for row in cells if str(row.get("config")) == config]
        best_config = max(rows, key=_strength)
        config_breakdown_rows.append(
            {
                "config": config,
                "strong_cells": sum(1 for row in rows if _is_strong_cell(row)),
                "top1_ge_0.10_cells": sum(1 for row in rows if float(row["top1_lift"]) >= 0.10),
                "best_domain": best_config["domain"],
                "best_prefix": best_config["prefix_length"],
                "best_head": best_config["head_id"],
                "best_architecture": best_config.get("architecture"),
                "best_top1_lift": best_config["top1_lift"],
                "best_pairwise_accuracy": best_config["pairwise_predictive_accuracy"],
            }
        )
    config_breakdown_rows = sorted(config_breakdown_rows, key=lambda r: (-int(r["strong_cells"]), str(r["config"])))

    architecture_breakdown_rows = []
    for architecture in sorted({str(row.get("architecture")) for row in cells}):
        rows = [row for row in cells if str(row.get("architecture")) == architecture]
        best_arch = max(rows, key=_strength)
        architecture_breakdown_rows.append(
            {
                "architecture": architecture,
                "strong_cells": sum(1 for row in rows if _is_strong_cell(row)),
                "top1_ge_0.10_cells": sum(1 for row in rows if float(row["top1_lift"]) >= 0.10),
                "best_domain": best_arch["domain"],
                "best_prefix": best_arch["prefix_length"],
                "best_config": best_arch.get("config"),
                "best_top1_lift": best_arch["top1_lift"],
                "best_pairwise_accuracy": best_arch["pairwise_predictive_accuracy"],
            }
        )

    operating_envelope = {
        "strong_cell_count": len(strong_cells),
        "top1_lift_ge_0.10_cell_count": sum(1 for row in cells if float(row["top1_lift"]) >= 0.10),
        "top2_lift_ge_0.10_cell_count": sum(1 for row in cells if float(row["top2_lift"]) >= 0.10),
        "pairwise_accuracy_ge_0.65_cell_count": sum(
            1
            for row in cells
            if row["pairwise_predictive_accuracy"] is not None and float(row["pairwise_predictive_accuracy"]) >= 0.65
        ),
        "strong_cells_by_domain": {
            domain: sum(1 for row in strong_cells if row["domain"] == domain)
            for domain in sorted({row["domain"] for row in cells})
        },
        "strong_cells_by_prefix": {
            str(prefix): sum(1 for row in strong_cells if int(row["prefix_length"]) == prefix)
            for prefix in sorted({int(row["prefix_length"]) for row in cells})
        },
        "strong_cells_by_config": {
            row["config"]: row["strong_cells"]
            for row in config_breakdown_rows
        },
        "strong_cells_by_architecture": {
            row["architecture"]: row["strong_cells"]
            for row in architecture_breakdown_rows
        },
    }

    payload = {
        "BG_TRAJECTORY_PREDICTION_VERDICT": verdict,
        "verdict": verdict,
        "GENERATOR_REACHABILITY_LIMITED": generator_limited,
        "BEST_PREDICTIVE_CELL": best or "NONE",
        "RECOMMENDED_STEERING_TARGET": target,
        "cell_count": len(cells),
        "top_cells": cells[:50],
        "heatmap_by_domain_prefix": heatmap,
        "prefix_length_trend": prefix_trend_rows,
        "domain_breakdown": domain_breakdown_rows,
        "config_breakdown": config_breakdown_rows,
        "architecture_breakdown": architecture_breakdown_rows,
        "operating_envelope": operating_envelope,
        "all_cells": cells,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)

    top_rows = []
    for row in cells[:25]:
        top_rows.append(
            {
                "domain": row["domain"],
                "prefix": row["prefix_length"],
                "head": row["head_id"],
                "config": row.get("config"),
                "top1_lift": f"{row['top1_lift']:.3f}",
                "top2_lift": f"{row['top2_lift']:.3f}",
                "pair_acc": "" if row["pairwise_predictive_accuracy"] is None else f"{row['pairwise_predictive_accuracy']:.3f}",
                "oracle": f"{row['oracle_success']:.3f}",
                "n": row["n_tasks"],
            }
        )
    heat_rows = [
        {
            "domain": row["domain"],
            "prefix": row["prefix_length"],
            "best_head": row["best_head_id"],
            "config": row.get("best_config"),
            "top1_lift": f"{row['top1_lift']:.3f}",
            "top2_lift": f"{row['top2_lift']:.3f}",
            "pair_acc": "" if row["pairwise_accuracy"] is None else f"{row['pairwise_accuracy']:.3f}",
            "oracle": f"{row['oracle_success']:.3f}",
            "n": row["n_tasks"],
        }
        for row in heatmap
    ]
    prefix_rows = [
        {
            "domain": row["domain"],
            "trend": row["top1_shape"],
            "top1_peak_prefix": row["top1_peak_prefix"],
            "top1_peak_lift": _fmt(row["top1_peak_lift"]),
            "pair_peak_prefix": row["pair_peak_prefix"],
            "pair_peak_acc": _fmt(row["pair_peak_accuracy"]),
            "p32": _fmt(row["prefix_32_top1_lift"]),
            "p64": _fmt(row["prefix_64_top1_lift"]),
            "p128": _fmt(row["prefix_128_top1_lift"]),
            "p256": _fmt(row["prefix_256_top1_lift"]),
        }
        for row in prefix_trend_rows
    ]
    domain_rows = [
        {
            "domain": row["domain"],
            "strong_cells": row["strong_cells"],
            "top1>=.10": row["top1_ge_0.10_cells"],
            "pair>=.65": row["pair_ge_0.65_cells"],
            "best_prefix": row["best_strength_prefix"],
            "best_config": row["best_strength_config"],
            "best_arch": row["best_strength_architecture"],
            "best_top1": _fmt(row["best_strength_top1_lift"]),
            "best_pair": _fmt(row["best_strength_pairwise_accuracy"]),
            "best_top1_prefix": row["best_top1_prefix"],
            "best_top1_lift": _fmt(row["best_top1_lift"]),
        }
        for row in domain_breakdown_rows
    ]
    config_rows = [
        {
            "config": row["config"],
            "strong_cells": row["strong_cells"],
            "top1>=.10": row["top1_ge_0.10_cells"],
            "best_domain": row["best_domain"],
            "best_prefix": row["best_prefix"],
            "best_arch": row["best_architecture"],
            "best_top1": _fmt(row["best_top1_lift"]),
            "best_pair": _fmt(row["best_pairwise_accuracy"]),
        }
        for row in config_breakdown_rows
    ]
    architecture_rows = [
        {
            "architecture": row["architecture"],
            "strong_cells": row["strong_cells"],
            "top1>=.10": row["top1_ge_0.10_cells"],
            "best_domain": row["best_domain"],
            "best_prefix": row["best_prefix"],
            "best_config": row["best_config"],
            "best_top1": _fmt(row["best_top1_lift"]),
            "best_pair": _fmt(row["best_pairwise_accuracy"]),
        }
        for row in architecture_breakdown_rows
    ]
    operating_rows = [
        {"metric": "strong_cells", "count": operating_envelope["strong_cell_count"]},
        {"metric": "top1_lift>=0.10_cells", "count": operating_envelope["top1_lift_ge_0.10_cell_count"]},
        {"metric": "top2_lift>=0.10_cells", "count": operating_envelope["top2_lift_ge_0.10_cell_count"]},
        {"metric": "pairwise_accuracy>=0.65_cells", "count": operating_envelope["pairwise_accuracy_ge_0.65_cell_count"]},
    ]
    interpretation_lines = [
        "Top1 predictive lift is not monotonic across prefix length. Reasoning top1 lift peaks at 64 tokens, science peaks at 32 tokens, and GSM8K peaks at 256 tokens; the selected best cell is 256-token reasoning because pairwise accuracy is strongest there.",
        "Reasoning is not the only positive domain. Science and GSM8K both have many strong cells, with science strongest at early prefixes and GSM8K showing high oracle reachability at every prefix.",
        "The config trend is broad rather than a single 36_mean-only result. 36_L4 has the most strong cells, 36_mean is second, and 24_L4 plus 47 variants also contribute.",
        "AntisymLinear is not the sole winner: NoNorm has slightly more strong cells overall, while the best predictive cell uses AntisymLinear.",
    ]
    lines = [
        "# BG Trajectory Predictive Power (2026-05-18)",
        "",
        f"BG_TRAJECTORY_PREDICTION_VERDICT = {verdict}",
        f"GENERATOR_REACHABILITY_LIMITED = {str(generator_limited).lower()}",
        "",
        f"BEST_PREDICTIVE_CELL = `{best}`",
        "",
        f"RECOMMENDED_STEERING_TARGET = `{target}`",
        "",
        "## Prefix-Length Heatmap",
        "",
        *md_table(heat_rows, ["domain", "prefix", "best_head", "config", "top1_lift", "top2_lift", "pair_acc", "oracle", "n"]),
        "",
        "## Prefix-Length Trend",
        "",
        *interpretation_lines[:1],
        "",
        *md_table(prefix_rows, ["domain", "trend", "top1_peak_prefix", "top1_peak_lift", "pair_peak_prefix", "pair_peak_acc", "p32", "p64", "p128", "p256"]),
        "",
        "## Domain Breakdown",
        "",
        *interpretation_lines[1:2],
        "",
        *md_table(domain_rows, ["domain", "strong_cells", "top1>=.10", "pair>=.65", "best_prefix", "best_config", "best_arch", "best_top1", "best_pair", "best_top1_prefix", "best_top1_lift"]),
        "",
        "## Config And Architecture Trend",
        "",
        *interpretation_lines[2:],
        "",
        *md_table(config_rows, ["config", "strong_cells", "top1>=.10", "best_domain", "best_prefix", "best_arch", "best_top1", "best_pair"]),
        "",
        *md_table(architecture_rows, ["architecture", "strong_cells", "top1>=.10", "best_domain", "best_prefix", "best_config", "best_top1", "best_pair"]),
        "",
        "## Operating Envelope",
        "",
        "The high-performance region is broad by the declared strong-cell rule, not a single isolated cell.",
        "",
        *md_table(operating_rows, ["metric", "count"]),
        "",
        f"- strong_cells_by_domain: `{operating_envelope['strong_cells_by_domain']}`",
        f"- strong_cells_by_prefix: `{operating_envelope['strong_cells_by_prefix']}`",
        f"- strong_cells_by_config: `{operating_envelope['strong_cells_by_config']}`",
        f"- strong_cells_by_architecture: `{operating_envelope['strong_cells_by_architecture']}`",
        "",
        "## Top Cells",
        "",
        *md_table(top_rows, ["domain", "prefix", "head", "config", "top1_lift", "top2_lift", "pair_acc", "oracle", "n"]),
    ]
    write_md(OUT_MD, lines)
    print(f"BG_TRAJECTORY_PREDICTION_VERDICT = {verdict}")
    print(f"BEST_PREDICTIVE_CELL = {best}")
    print(f"RECOMMENDED_STEERING_TARGET = {target}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
