"""Compare adaptive survival policies with simple top-k for hidden branches."""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from bg_hidden_branch_suite_common import REPORT_ROOT, ensure_report_root, load_json, md_table, rel, write_csv, write_json, write_md


OUTCOMES_JSON = REPORT_ROOT / "hidden_branch_outcomes.json"
OUT_JSON = REPORT_ROOT / "hidden_branch_adaptive_threshold_sweep.json"
OUT_MD = REPORT_ROOT / "hidden_branch_adaptive_threshold_sweep.md"
OUT_CSV = REPORT_ROOT / "hidden_branch_adaptive_threshold_rows.csv"


def _select(policy: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda r: (-float(r.get("tap_margin_sum", 0.0)), int(r["branch_id"])))
    if policy == "clean_only":
        return sorted(rows, key=lambda r: int(r["branch_id"]))[:1]
    if policy == "random_keep_k":
        return sorted(rows, key=lambda r: int(r["branch_id"]))[:2]
    if policy == "top1":
        return ordered[:1]
    if policy == "top2":
        return ordered[:2]
    if policy == "top3":
        return ordered[:3]
    margins = [float(r.get("tap_margin_sum", 0.0)) for r in ordered]
    spread = max(margins) - min(margins) if margins else 0.0
    if policy == "fixed_absolute_threshold":
        keep = [r for r in ordered if float(r.get("tap_margin_sum", 0.0)) >= 0.0]
        return keep[:3] or ordered[:1]
    if policy == "relative_margin_threshold":
        threshold = max(margins) - 0.25 * spread if margins else 0.0
        keep = [r for r in ordered if float(r.get("tap_margin_sum", 0.0)) >= threshold]
        return keep[:3] or ordered[:1]
    if policy == "score_spread_adaptive":
        if spread < 1e-6:
            return ordered[:2]
        threshold = max(margins) - 0.40 * spread
        keep = [r for r in ordered if float(r.get("tap_margin_sum", 0.0)) >= threshold]
        return keep[:3] or ordered[:1]
    if policy == "diversity_bonus_policy":
        keep = ordered[:1]
        for row in ordered[1:]:
            if len(keep) >= 3:
                break
            if abs(int(row["branch_id"]) - int(keep[0]["branch_id"])) >= 2:
                keep.append(row)
        return keep
    if policy == "instability_penalty_policy":
        stable = [r for r in ordered if float(r.get("repetition_rate", 0.0)) < 0.30 and not r.get("empty_output")]
        return (stable or ordered)[:2]
    if policy == "compute_budget_policy":
        return ordered[:2]
    raise ValueError(policy)


def main() -> int:
    ensure_report_root()
    started = time.time()
    payload = load_json(OUTCOMES_JSON, {})
    rows = [r for r in payload.get("rows", []) if r.get("safety_envelope")]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["branch_group_id"]].append(row)
    groups = [vals for vals in grouped.values() if len(vals) >= 2]
    policies = [
        "clean_only",
        "random_keep_k",
        "top1",
        "top2",
        "top3",
        "fixed_absolute_threshold",
        "relative_margin_threshold",
        "score_spread_adaptive",
        "diversity_bonus_policy",
        "instability_penalty_policy",
        "compute_budget_policy",
    ]
    sweep_rows = []
    for vals in groups:
        oracle = any(v.get("correct") for v in vals)
        for policy in policies:
            kept = _select(policy, vals)
            sweep_rows.append(
                {
                    "branch_group_id": vals[0]["branch_group_id"],
                    "task_id": vals[0]["task_id"],
                    "domain": vals[0]["domain"],
                    "branch_point": vals[0]["branch_point"],
                    "alpha": vals[0]["alpha"],
                    "policy": policy,
                    "survivors": len(kept),
                    "compute_cost_proxy": len(kept) / max(len(vals), 1),
                    "oracle_available": oracle,
                    "oracle_retained": any(v.get("correct") for v in kept),
                    "top1_success": bool(kept[0].get("correct")) if kept else False,
                    "top2_success": any(v.get("correct") for v in kept[:2]),
                    "reward_mean": sum(float(v.get("reward", 0.0)) for v in kept) / max(len(kept), 1),
                    "false_prune": oracle and not any(v.get("correct") for v in kept),
                }
            )
    metrics = {}
    for policy in policies:
        vals = [r for r in sweep_rows if r["policy"] == policy]
        metrics[policy] = {
            "groups": len(vals),
            "oracle_retention": sum(1 for r in vals if r["oracle_retained"]) / max(len(vals), 1),
            "top1_success": sum(1 for r in vals if r["top1_success"]) / max(len(vals), 1),
            "top2_success": sum(1 for r in vals if r["top2_success"]) / max(len(vals), 1),
            "reward_mean": sum(float(r["reward_mean"]) for r in vals) / max(len(vals), 1),
            "average_survivors": sum(int(r["survivors"]) for r in vals) / max(len(vals), 1),
            "compute_cost_proxy": sum(float(r["compute_cost_proxy"]) for r in vals) / max(len(vals), 1),
            "false_prune_rate": sum(1 for r in vals if r["false_prune"]) / max(len(vals), 1),
        }
    top2 = metrics.get("top2", {})
    adaptive = metrics.get("score_spread_adaptive", {})
    diversity = sum(1 for vals in groups if len({float(v.get("reward", 0.0)) for v in vals}) > 1)
    if not groups:
        verdict = "INSUFFICIENT"
    elif diversity == 0:
        verdict = "NO_BEHAVIORAL_VARIATION"
    elif adaptive.get("oracle_retention", 0.0) > top2.get("oracle_retention", 0.0) + 0.02 and adaptive.get("compute_cost_proxy", 1.0) <= top2.get("compute_cost_proxy", 1.0) + 0.05:
        verdict = "ADAPTIVE_BEATS_TOPK"
    elif adaptive.get("oracle_retention", 0.0) + 0.02 >= top2.get("oracle_retention", 0.0):
        verdict = "TOPK_SUFFICIENT"
    else:
        verdict = "THRESHOLDS_UNRELIABLE"
    out = {
        "BG_HIDDEN_BRANCH_ADAPTIVE_THRESHOLD_VERDICT": verdict,
        "verdict": verdict,
        "group_count": len(groups),
        "behaviorally_diverse_group_count": diversity,
        "metrics": metrics,
        "rows": sweep_rows,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, out)
    write_csv(OUT_CSV, sweep_rows)
    lines = [
        "# BG Hidden Branch Adaptive Threshold Sweep",
        "",
        f"BG_HIDDEN_BRANCH_ADAPTIVE_THRESHOLD_VERDICT = {verdict}",
        "",
        f"- group_count: `{len(groups)}`",
        f"- behaviorally_diverse_group_count: `{diversity}`",
        "",
        "## Policy Metrics",
        "",
    ]
    lines.extend(md_table([
        {"policy": k, **{kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}}
        for k, v in metrics.items()
    ], ["policy", "groups", "oracle_retention", "top1_success", "top2_success", "reward_mean", "average_survivors", "compute_cost_proxy", "false_prune_rate"]))
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_BRANCH_ADAPTIVE_THRESHOLD_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict != "INSUFFICIENT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
