"""Simulate frozen BG tap selection on hidden-origin branch outcomes."""
from __future__ import annotations

import statistics
import time
from collections import defaultdict
from typing import Any

from bg_hidden_branch_suite_common import REPORT_ROOT, ensure_report_root, load_json, md_table, rel, write_csv, write_json, write_md


OUT_JSON = REPORT_ROOT / "hidden_origin_branch_selection.json"
OUT_MD = REPORT_ROOT / "hidden_origin_branch_selection.md"
OUT_CSV = REPORT_ROOT / "hidden_origin_branch_selection_rows.csv"
OUTCOMES_JSON = REPORT_ROOT / "hidden_branch_outcomes.json"


def _mean(vals: list[float]) -> float:
    return sum(vals) / max(len(vals), 1)


def _policy_rows(group: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows = sorted(group, key=lambda r: int(r["branch_id"]))
    rewards = [float(r.get("reward", 0.0)) for r in rows]
    successes = [bool(r.get("correct")) for r in rows]
    by_tap = sorted(rows, key=lambda r: (-float(r.get("tap_margin_sum", 0.0)), int(r["branch_id"])))
    random_top1 = _mean([1.0 if s else 0.0 for s in successes])
    random_reward = _mean(rewards)
    top2 = by_tap[:2]
    return {
        "random_top1": {
            "success": random_top1,
            "reward": random_reward,
            "oracle_coverage": random_top1,
            "survivors": len(rows),
        },
        "clean_branch_baseline": {
            "success": 1.0 if rows[0].get("correct") else 0.0,
            "reward": float(rows[0].get("reward", 0.0)),
            "oracle_coverage": 1.0 if rows[0].get("correct") else 0.0,
            "survivors": 1,
        },
        "highest_tap_score": {
            "success": 1.0 if by_tap[0].get("correct") else 0.0,
            "reward": float(by_tap[0].get("reward", 0.0)),
            "oracle_coverage": 1.0 if by_tap[0].get("correct") else 0.0,
            "survivors": 1,
        },
        "pairwise_tournament_winner": {
            "success": 1.0 if by_tap[0].get("correct") else 0.0,
            "reward": float(by_tap[0].get("reward", 0.0)),
            "oracle_coverage": 1.0 if by_tap[0].get("correct") else 0.0,
            "survivors": 1,
        },
        "pairwise_tournament_top2": {
            "success": 1.0 if any(r.get("correct") for r in top2) else 0.0,
            "reward": max(float(r.get("reward", 0.0)) for r in top2),
            "oracle_coverage": 1.0 if any(r.get("correct") for r in top2) else 0.0,
            "survivors": len(top2),
        },
        "adaptive_threshold_policy_v0": adaptive_policy(rows),
    }


def adaptive_policy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda r: (-float(r.get("tap_margin_sum", 0.0)), int(r["branch_id"])))
    margins = [float(r.get("tap_margin_sum", 0.0)) for r in ordered]
    spread = max(margins) - min(margins) if margins else 0.0
    if spread < 1e-6:
        keep = ordered[:2]
    else:
        threshold = max(margins) - 0.35 * spread
        keep = [r for r in ordered if float(r.get("tap_margin_sum", 0.0)) >= threshold]
        keep = keep[:3] or ordered[:1]
    return {
        "success": 1.0 if any(r.get("correct") for r in keep) else 0.0,
        "reward": max(float(r.get("reward", 0.0)) for r in keep),
        "oracle_coverage": 1.0 if any(r.get("correct") for r in keep) else 0.0,
        "survivors": len(keep),
    }


def main() -> int:
    ensure_report_root()
    started = time.time()
    payload = load_json(OUTCOMES_JSON, {})
    rows = list(payload.get("rows") or [])
    safe_rows = [r for r in rows if r.get("safety_envelope")]
    grouped = defaultdict(list)
    for row in safe_rows:
        grouped[row["branch_group_id"]].append(row)
    groups = [vals for vals in grouped.values() if len(vals) >= 2]
    sim_rows = []
    for vals in groups:
        policies = _policy_rows(vals)
        oracle = 1.0 if any(v.get("correct") for v in vals) else 0.0
        diversity = len({str(v.get("parsed_answer")) for v in vals}) > 1 or len({float(v.get("reward", 0.0)) for v in vals}) > 1
        for policy, metric in policies.items():
            sim_rows.append(
                {
                    "branch_group_id": vals[0]["branch_group_id"],
                    "task_id": vals[0]["task_id"],
                    "domain": vals[0]["domain"],
                    "branch_point": vals[0]["branch_point"],
                    "alpha": vals[0]["alpha"],
                    "policy": policy,
                    "success": metric["success"],
                    "reward": metric["reward"],
                    "oracle_coverage": metric["oracle_coverage"],
                    "survivors": metric["survivors"],
                    "oracle_available": oracle,
                    "behavioral_diversity": diversity,
                }
            )
    by_policy = defaultdict(list)
    for row in sim_rows:
        by_policy[row["policy"]].append(row)
    metrics = {}
    for policy, vals in sorted(by_policy.items()):
        metrics[policy] = {
            "groups": len(vals),
            "top1_success": _mean([float(v["success"]) for v in vals]),
            "reward_mean": _mean([float(v["reward"]) for v in vals]),
            "top2_oracle_coverage": _mean([float(v["oracle_coverage"]) for v in vals]),
            "average_survivors": _mean([float(v["survivors"]) for v in vals]),
        }
    random_rate = metrics.get("random_top1", {}).get("top1_success", 0.0)
    tap_rate = metrics.get("pairwise_tournament_winner", {}).get("top1_success", 0.0)
    lift = tap_rate - random_rate
    diversity_groups = sum(1 for vals in groups if len({float(v.get("reward", 0.0)) for v in vals}) > 1)
    if not groups:
        verdict = "INSUFFICIENT"
    elif diversity_groups == 0:
        verdict = "NO_BEHAVIORAL_VARIATION"
    elif lift >= 0.05:
        verdict = "FROZEN_TAPS_SELECT_GOOD_HIDDEN_BRANCHES"
    elif lift > 0.0:
        verdict = "WEAK_HIDDEN_BRANCH_SELECTION_SIGNAL"
    else:
        verdict = "NO_HIDDEN_BRANCH_SELECTION_SIGNAL"
    out = {
        "BG_HIDDEN_ORIGIN_BRANCH_SELECTION_VERDICT": verdict,
        "verdict": verdict,
        "group_count": len(groups),
        "row_count": len(sim_rows),
        "behaviorally_diverse_group_count": diversity_groups,
        "tap_selection_vs_random_lift": lift,
        "metrics": metrics,
        "rows": sim_rows,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, out)
    write_csv(OUT_CSV, sim_rows)
    lines = [
        "# BG Hidden-Origin Branch Selection",
        "",
        f"BG_HIDDEN_ORIGIN_BRANCH_SELECTION_VERDICT = {verdict}",
        "",
        f"- group_count: `{len(groups)}`",
        f"- behaviorally_diverse_group_count: `{diversity_groups}`",
        f"- tap_selection_vs_random_lift: `{lift:.4f}`",
        "",
        "## Policy Metrics",
        "",
    ]
    lines.extend(md_table([
        {"policy": k, **{kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}}
        for k, v in metrics.items()
    ], ["policy", "groups", "top1_success", "reward_mean", "top2_oracle_coverage", "average_survivors"]))
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_BRANCH_SELECTION_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict != "INSUFFICIENT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
