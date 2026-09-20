"""Analyze v4 quota-generation diversity drivers."""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from bg_hidden_origin_quota_v4_common import (
    V4_ROOT,
    branch_group_metrics,
    candidate_pair_stats,
    diagnostic_alpha_v4_row,
    ensure_v4_root,
    group_is_behaviorally_diverse_v2,
    group_is_reward_diverse_v2,
    group_rows,
    load_v4_branch_rows,
    md_table,
    normalize_branch_point,
    primary_safe_v4_row,
    rate,
    read_jsonl,
    row_reward,
    sampled_reward,
    stable_v2_row,
    write_json,
    write_md,
    RECIPE_ENGRAMS_JSONL,
)


OUT_JSON = V4_ROOT / "quota_diversity_drivers.json"
OUT_MD = V4_ROOT / "quota_diversity_drivers.md"


def condition_stats(rows: list[dict[str, Any]], field: str, label_source: str = "deterministic") -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for gid, vals in group_rows(rows).items():
        if len(vals) < 2:
            continue
        first = vals[0]
        if field == "branch_point":
            value = normalize_branch_point(first)
        elif field == "delta_family":
            value = str(first.get("primary_delta_family") or first.get("delta_family") or "unknown")
        elif field == "K":
            value = str(first.get("K") or len(vals))
        else:
            value = str(first.get(field) or "unknown")
        bucket = out.setdefault(
            value,
            {
                "rows": 0,
                "groups": 0,
                "behaviorally_diverse_groups": 0,
                "reward_diverse_groups": 0,
                "candidate_pairs": 0,
                "tie_pairs": 0,
                "non_tie_pairs": 0,
                "stable_rows": 0,
                "parse_success_rows": 0,
            },
        )
        bucket["rows"] += len(vals)
        bucket["groups"] += 1
        bucket["stable_rows"] += sum(1 for row in vals if stable_v2_row(row))
        bucket["parse_success_rows"] += sum(1 for row in vals if row.get("parse_success"))
        if group_is_behaviorally_diverse_v2(vals, label_source):
            bucket["behaviorally_diverse_groups"] += 1
        if group_is_reward_diverse_v2(vals, label_source):
            bucket["reward_diverse_groups"] += 1
        pairs = candidate_pair_stats({"g": vals}, label_source)
        for key in ("candidate_pairs", "tie_pairs", "non_tie_pairs"):
            bucket[key] += pairs[key]
    for row in out.values():
        row["tie_rate"] = row["tie_pairs"] / max(row["candidate_pairs"], 1)
        row["stable_rate"] = row["stable_rows"] / max(row["rows"], 1)
        row["parse_rate"] = row["parse_success_rows"] / max(row["rows"], 1)
        row["behaviorally_diverse_groups_per_100_rows"] = 100.0 * row["behaviorally_diverse_groups"] / max(row["rows"], 1)
        row["non_tie_pairs_per_100_rows"] = 100.0 * row["non_tie_pairs"] / max(row["rows"], 1)
    return out


def top_conditions(stats_by_factor: dict[str, dict[str, dict[str, Any]]], limit: int = 20) -> list[dict[str, Any]]:
    rows = []
    for factor, stats in stats_by_factor.items():
        for value, row in stats.items():
            rows.append({"factor": factor, "condition": value, **row})
    rows.sort(key=lambda row: (float(row["behaviorally_diverse_groups_per_100_rows"]), float(row["non_tie_pairs_per_100_rows"]), int(row["behaviorally_diverse_groups"])), reverse=True)
    return rows[:limit]


def answer_questions(primary_stats: dict[str, dict[str, dict[str, Any]]], engrams: list[dict[str, Any]]) -> dict[str, Any]:
    branch = primary_stats.get("branch_point", {})
    alpha = primary_stats.get("alpha_bucket", {})
    delta = primary_stats.get("delta_family", {})
    k_stats = primary_stats.get("K", {})
    recipe_source = primary_stats.get("recipe_source", {})
    random_yield = max(
        delta.get("random_orthogonal", {}).get("behaviorally_diverse_groups_per_100_rows", 0.0),
        delta.get("paired_plus_minus", {}).get("behaviorally_diverse_groups_per_100_rows", 0.0),
    )
    non_random_yield = max([row.get("behaviorally_diverse_groups_per_100_rows", 0.0) for name, row in delta.items() if name not in {"random_orthogonal", "paired_plus_minus", "clean", "random_fallback"}] or [0.0])
    l36 = branch.get("L36", {}).get("behaviorally_diverse_groups_per_100_rows", 0.0)
    l24 = branch.get("L24", {}).get("behaviorally_diverse_groups_per_100_rows", 0.0)
    k8 = k_stats.get("8", {}).get("behaviorally_diverse_groups_per_100_rows", 0.0)
    k6 = k_stats.get("6", {}).get("behaviorally_diverse_groups_per_100_rows", 0.0)
    hs = recipe_source.get("hs_inspired_controller", {}).get("behaviorally_diverse_groups_per_100_rows", 0.0)
    static = recipe_source.get("static_high_yield", {}).get("behaviorally_diverse_groups_per_100_rows", 0.0)
    engram_rows = sum(int(row.get("rows_generated") or 0) for row in engrams)
    engram_behavior = sum(int(row.get("behaviorally_diverse_groups") or 0) for row in engrams)
    return {
        "L36_remains_best": l36 >= l24,
        "L36_yield": l36,
        "L24_yield": l24,
        "alpha_0_005_vs_0_01": {
            "alpha_0_005": alpha.get("alpha_0_005", {}).get("behaviorally_diverse_groups_per_100_rows", 0.0),
            "alpha_0_01": alpha.get("alpha_0_01", {}).get("behaviorally_diverse_groups_per_100_rows", 0.0),
        },
        "non_random_directions_help": non_random_yield > random_yield,
        "random_yield": random_yield,
        "non_random_yield": non_random_yield,
        "K8_helps": k8 >= k6 and k8 > 0.0,
        "K8_yield": k8,
        "K6_yield": k6,
        "hs_controller_yield": hs,
        "static_high_yield_yield": static,
        "hs_controller_helped": hs > static or (engram_rows > 0 and 100.0 * engram_behavior / max(engram_rows, 1) > 0.0),
    }


def verdict(primary_metrics: dict[str, Any], questions: dict[str, Any], engrams: list[dict[str, Any]]) -> str:
    if int(primary_metrics.get("groups") or 0) < 3:
        return "INSUFFICIENT"
    if questions["hs_controller_helped"]:
        return "HS_INSPIRED_CONTROLLER_HELPS"
    if questions["non_random_directions_help"]:
        return "NON_RANDOM_DIRECTIONS_HELP"
    if questions["K8_helps"] and questions["L36_remains_best"]:
        return "L36_K8_HELPS"
    if int(primary_metrics.get("behaviorally_diverse_groups") or 0) > 0:
        return "CLEAR_QUOTA_RECIPE"
    return "NEEDS_BETTER_BRANCH_GENERATOR"


def display(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        for key in ("behaviorally_diverse_groups_per_100_rows", "non_tie_pairs_per_100_rows", "tie_rate", "stable_rate", "parse_rate"):
            if key in item:
                item[key] = rate(item[key])
        out.append(item)
    return out


def main() -> int:
    started = time.time()
    ensure_v4_root()
    rows = load_v4_branch_rows()
    if not rows:
        payload = {"BG_HIDDEN_ORIGIN_QUOTA_DIVERSITY_DRIVER_V4_VERDICT": "INSUFFICIENT", "verdict": "INSUFFICIENT", "blocker": "no v4 branch rows"}
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# Hidden-Origin Quota Diversity Drivers V4", "", "BG_HIDDEN_ORIGIN_QUOTA_DIVERSITY_DRIVER_V4_VERDICT = INSUFFICIENT"])
        print("BG_HIDDEN_ORIGIN_QUOTA_DIVERSITY_DRIVER_V4_VERDICT = INSUFFICIENT", flush=True)
        return 0
    primary_rows = [row for row in rows if primary_safe_v4_row(row)]
    diagnostic_rows = [row for row in rows if diagnostic_alpha_v4_row(row)]
    sampled_rows = [row for row in rows if sampled_reward(row) is not None and stable_v2_row(row)]
    factors = ["split", "task_screening_class", "task_class", "branch_point", "alpha_bucket", "K", "delta_family", "domain", "recipe_source", "recipe_id"]
    primary_stats = {factor: condition_stats(primary_rows, factor) for factor in factors}
    diagnostic_stats = {factor: condition_stats(diagnostic_rows, factor) for factor in factors}
    sampled_stats = {factor: condition_stats(sampled_rows, factor, "sampled_expected") for factor in factors}
    primary_metrics = branch_group_metrics(primary_rows)
    diagnostic_metrics = branch_group_metrics(diagnostic_rows)
    sampled_metrics = branch_group_metrics(sampled_rows, "sampled_expected")
    engrams = read_jsonl(RECIPE_ENGRAMS_JSONL)
    top = top_conditions(primary_stats)
    questions = answer_questions(primary_stats, engrams)
    v = verdict(primary_metrics, questions, engrams)
    payload = {
        "BG_HIDDEN_ORIGIN_QUOTA_DIVERSITY_DRIVER_V4_VERDICT": v,
        "verdict": v,
        "primary_metrics": primary_metrics,
        "diagnostic_alpha_0_02_metrics": diagnostic_metrics,
        "sampled_expected_metrics": sampled_metrics,
        "primary_stats_by_factor": primary_stats,
        "diagnostic_alpha_0_02_stats_by_factor": diagnostic_stats,
        "sampled_expected_stats_by_factor": sampled_stats,
        "top_conditions": top,
        "questions": questions,
        "recipe_engrams": engrams[-300:],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    lines = [
        "# Hidden-Origin Quota Diversity Drivers V4",
        "",
        f"BG_HIDDEN_ORIGIN_QUOTA_DIVERSITY_DRIVER_V4_VERDICT = {v}",
        "",
        f"- primary_metrics: `{primary_metrics}`",
        f"- diagnostic_alpha_0_02_metrics: `{diagnostic_metrics}`",
        f"- sampled_expected_metrics: `{sampled_metrics}`",
        f"- questions: `{questions}`",
        "",
        "## Top Conditions",
        "",
    ]
    lines.extend(md_table(display(top), ["factor", "condition", "rows", "groups", "behaviorally_diverse_groups", "reward_diverse_groups", "behaviorally_diverse_groups_per_100_rows", "non_tie_pairs_per_100_rows", "tie_rate", "stable_rate", "parse_rate"]))
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_QUOTA_DIVERSITY_DRIVER_V4_VERDICT = {v}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
