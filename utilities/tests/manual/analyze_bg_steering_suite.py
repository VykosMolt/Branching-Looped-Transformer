"""Analyze and aggregate the BG steering suite results."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from bg_steering_suite_lib import (
    REPORT_ROOT,
    auc_score,
    bootstrap_delta,
    load_json,
    rel,
    spearman,
    write_json,
    write_md,
)


OUT_JSON = REPORT_ROOT / "analysis.json"
OUT_MD = REPORT_ROOT / "analysis.md"


def _v(name: str, path: str, key: str) -> str:
    data = load_json(REPORT_ROOT / path, {})
    return data.get(key) or data.get("verdict") or "NOT_RUN"


def main() -> int:
    reach = load_json(REPORT_ROOT / "reachability_gate.json", {})
    partial = load_json(REPORT_ROOT / "partial_routing_results.json", {})
    alloc = load_json(REPORT_ROOT / "compute_allocation_results.json", {})
    wrapper = load_json(REPORT_ROOT / "wrapper_matched_results.json", {})
    steer = load_json(REPORT_ROOT / "soft_steering_results.json", {})
    prefix = load_json(REPORT_ROOT / "text_prefix_branch_selection_results.json", {})
    task_suite = load_json(REPORT_ROOT / "task_suite.json", {})
    branch_pool = load_json(REPORT_ROOT / "branch_pools.json", {})

    scores = []
    labels = []
    head_success = defaultdict(list)
    head_selected_success = defaultdict(list)
    devil = []
    for task in partial.get("task_results", []) or []:
        successes = {int(c["branch_id"]): bool(c.get("evaluation", {}).get("success")) for c in task.get("continuations", [])}
        for idx, bid in enumerate(task.get("branch_ids", [])):
            margins = task.get("rankings", {}).get("conservative", {}).get("margin_sum", [])
            if idx < len(margins):
                scores.append(float(margins[idx]))
                labels.append(successes.get(int(bid), False))
        diag = task.get("rankings", {}).get("diagnostic_selected", {})
        for head, bid in diag.items():
            head_selected_success[head].append(successes.get(int(bid), False))
        if task.get("is_devil"):
            parseable = sum(1 for c in task.get("continuations", []) if c.get("evaluation", {}).get("parsed"))
            any_success = any(successes.values())
            selected = task.get("rankings", {}).get("conservative", {}).get("ranking_branch_ids", [None])[0]
            devil.append({"task_id": task["task_id"], "any_success": any_success, "parseable_branches": parseable, "bg_selected": selected})

    partial_metrics = partial.get("metrics") or {}
    alloc_metrics = alloc.get("metrics") or {}
    generator_limited = bool(reach.get("GENERATOR_REACHABILITY_LIMITED")) or partial_metrics.get("oracle_success_rate", 1.0) < 0.10
    deployable_verdicts = [partial.get("BG_PARTIAL_ROUTING_VERDICT"), alloc.get("BG_COMPUTE_ALLOCATION_VERDICT")]
    if "HELPS" in deployable_verdicts:
        overall = "HELPS"
    elif "HURTS" in deployable_verdicts:
        overall = "HURTS"
    elif any(v == "NEUTRAL" for v in deployable_verdicts):
        overall = "NEUTRAL"
    elif generator_limited:
        overall = "INSUFFICIENT"
    else:
        overall = "INSUFFICIENT"
    if "HELPS" in deployable_verdicts and "HURTS" in deployable_verdicts:
        overall = "MIXED"

    head_summary = {
        head: {
            "n": len(vals),
            "selected_success_rate": sum(vals) / max(len(vals), 1),
        }
        for head, vals in head_selected_success.items()
    }
    analysis = {
        "BG_PARTIAL_ROUTING_VERDICT": partial.get("BG_PARTIAL_ROUTING_VERDICT", "NOT_RUN"),
        "BG_COMPUTE_ALLOCATION_VERDICT": alloc.get("BG_COMPUTE_ALLOCATION_VERDICT", "NOT_RUN"),
        "BG_WRAPPER_MATCHED_VERDICT": wrapper.get("BG_WRAPPER_MATCHED_VERDICT", "NOT_RUN"),
        "BG_SOFT_STEERING_VERDICT": steer.get("BG_SOFT_STEERING_VERDICT", "NOT_RUN"),
        "BG_LATENT_BRANCH_SELECTION_VERDICT": prefix.get("BG_LATENT_BRANCH_SELECTION_VERDICT", "NOT_RUN"),
        "OVERALL_BG_STEERING_VERDICT": overall,
        "GENERATOR_REACHABILITY_LIMITED": generator_limited,
        "generator_reachability": reach.get("by_domain", {}),
        "bg_selection_value": {
            "partial_metrics": partial_metrics,
            "compute_allocation_metrics": alloc_metrics,
        },
        "partial_score_predictive_power": {
            "spearman_margin_success": spearman(scores, [1.0 if x else 0.0 for x in labels]),
            "auc_margin_success": auc_score(scores, labels),
            "n_branch_scores": len(scores),
        },
        "head_comparison": head_summary,
        "devil_task_analysis": devil,
        "steering_analysis": steer.get("metrics", {}),
        "wrapper_analysis": {"verdict": wrapper.get("BG_WRAPPER_MATCHED_VERDICT"), "reason": wrapper.get("reason")},
        "warnings": {
            "SMALL_N_WARNING": int(partial_metrics.get("evaluable_tasks", 0)) < 30,
            "GENERATOR_REACHABILITY_LIMITED": generator_limited,
            "COMPUTE_MISMATCH_WARNING": bool(wrapper.get("compute_mismatch_warning")),
            "SOFT_STEERING_UNSTABLE": steer.get("BG_SOFT_STEERING_VERDICT") == "DESTABILIZING",
        },
        "task_count": task_suite.get("task_count"),
        "branch_count": branch_pool.get("branch_count"),
    }
    write_json(OUT_JSON, analysis)
    lines = [
        "# BG Steering Suite Analysis (2026-05-18)",
        "",
        f"BG_PARTIAL_ROUTING_VERDICT = {analysis['BG_PARTIAL_ROUTING_VERDICT']}",
        f"BG_COMPUTE_ALLOCATION_VERDICT = {analysis['BG_COMPUTE_ALLOCATION_VERDICT']}",
        f"BG_WRAPPER_MATCHED_VERDICT = {analysis['BG_WRAPPER_MATCHED_VERDICT']}",
        f"BG_SOFT_STEERING_VERDICT = {analysis['BG_SOFT_STEERING_VERDICT']}",
        f"BG_LATENT_BRANCH_SELECTION_VERDICT = {analysis['BG_LATENT_BRANCH_SELECTION_VERDICT']}",
        f"OVERALL_BG_STEERING_VERDICT = {overall}",
        "",
        "## Generator Reachability",
        "",
        f"`{analysis['generator_reachability']}`",
        "",
        "## BG Selection Value",
        "",
        f"`{analysis['bg_selection_value']}`",
        "",
        "## Predictive Power",
        "",
        f"`{analysis['partial_score_predictive_power']}`",
        "",
        "## Devil Tasks",
        "",
        f"`{devil}`",
        "",
        "## Warnings",
        "",
        f"`{analysis['warnings']}`",
    ]
    write_md(OUT_MD, lines)
    print(f"OVERALL_BG_STEERING_VERDICT = {overall}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
