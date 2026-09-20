"""Analyze which v3 branch-generation choices increased diversity yield."""
from __future__ import annotations

import time
from typing import Any

from bg_hidden_origin_diversity_v3_common import (
    V3_ROOT,
    branch_group_metrics,
    condition_group_stats,
    diagnostic_alpha_v3_row,
    ensure_v3_root,
    group_rows,
    load_v3_branch_rows,
    md_table,
    primary_safe_v3_row,
    rate,
    sampled_reward,
    stable_v2_row,
    write_json,
    write_md,
)


OUT_JSON = V3_ROOT / "diversity_drivers_v3.json"
OUT_MD = V3_ROOT / "diversity_drivers_v3.md"


def top_conditions(stats_by_factor: dict[str, dict[str, dict[str, Any]]], limit: int = 10) -> list[dict[str, Any]]:
    rows = []
    for factor, stats in stats_by_factor.items():
        for value, row in stats.items():
            rows.append(
                {
                    "factor": factor,
                    "condition": value,
                    "rows": row["rows"],
                    "groups": row["groups"],
                    "behaviorally_diverse_groups": row["behaviorally_diverse_groups"],
                    "reward_diverse_groups": row["reward_diverse_groups"],
                    "non_tie_pairs_per_100_rows": row["non_tie_pairs_per_100_rows"],
                    "behaviorally_diverse_groups_per_100_rows": row["behaviorally_diverse_groups_per_100_rows"],
                    "tie_rate": row["tie_rate"],
                    "stable_rate": row["stable_rate"],
                    "parse_rate": row["parse_rate"],
                }
            )
    rows.sort(
        key=lambda row: (
            float(row["behaviorally_diverse_groups_per_100_rows"]),
            float(row["non_tie_pairs_per_100_rows"]),
            int(row["behaviorally_diverse_groups"]),
        ),
        reverse=True,
    )
    return rows[:limit]


def avoid_conditions(stats_by_factor: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for factor, stats in stats_by_factor.items():
        for value, row in stats.items():
            if int(row["groups"]) >= 2 and int(row["behaviorally_diverse_groups"]) == 0:
                rows.append(
                    {
                        "factor": factor,
                        "condition": value,
                        "groups": row["groups"],
                        "rows": row["rows"],
                        "tie_rate": row["tie_rate"],
                        "stable_rate": row["stable_rate"],
                        "parse_rate": row["parse_rate"],
                    }
                )
    rows.sort(key=lambda row: (-float(row["tie_rate"]), -int(row["groups"])))
    return rows[:20]


def max_rate(stats: dict[str, dict[str, Any]], key: str) -> float:
    values = [float(row.get(key, 0.0)) for row in stats.values()]
    return max(values) if values else 0.0


def answer_questions(primary_stats: dict[str, dict[str, dict[str, Any]]], diagnostic_stats: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    alpha = primary_stats.get("alpha_bucket", {})
    alpha_diag = diagnostic_stats.get("alpha_bucket", {})
    delta = primary_stats.get("delta_family", {})
    k_stats = primary_stats.get("K", {})
    cls_stats = primary_stats.get("task_screening_class", {})
    random_yield = max(
        delta.get("random_orthogonal", {}).get("behaviorally_diverse_groups_per_100_rows", 0.0),
        delta.get("paired_plus_minus", {}).get("behaviorally_diverse_groups_per_100_rows", 0.0),
    )
    non_random_yield = max(
        [
            row.get("behaviorally_diverse_groups_per_100_rows", 0.0)
            for name, row in delta.items()
            if name not in {"random_orthogonal", "paired_plus_minus", "clean", "random_fallback"}
        ]
        or [0.0]
    )
    k4 = k_stats.get("4", {}).get("behaviorally_diverse_groups_per_100_rows", 0.0)
    k6 = k_stats.get("6", {}).get("behaviorally_diverse_groups_per_100_rows", 0.0)
    k8 = k_stats.get("8", {}).get("behaviorally_diverse_groups_per_100_rows", 0.0)
    preferred_classes = [
        "baseline_parse_fragile",
        "baseline_wrong_parseable",
        "baseline_correct_low_confidence",
        "perturbation_sensitive",
    ]
    preferred_yield = max([cls_stats.get(name, {}).get("behaviorally_diverse_groups_per_100_rows", 0.0) for name in preferred_classes] or [0.0])
    confident_yield = cls_stats.get("baseline_correct_confident", {}).get("behaviorally_diverse_groups_per_100_rows", 0.0)
    primary_alpha_yield = max(alpha.get("alpha_0_005", {}).get("behaviorally_diverse_groups_per_100_rows", 0.0), alpha.get("alpha_0_01", {}).get("behaviorally_diverse_groups_per_100_rows", 0.0))
    alpha02_yield = alpha_diag.get("alpha_0_02", {}).get("behaviorally_diverse_groups_per_100_rows", 0.0)
    return {
        "l24_vs_l36": {
            "L24": primary_stats.get("branch_point", {}).get("L24", {}),
            "L36": primary_stats.get("branch_point", {}).get("L36", {}),
        },
        "alpha_0_02_helps_safely": alpha02_yield > primary_alpha_yield + 1.0,
        "alpha_0_02_required": alpha02_yield > 0.0 and primary_alpha_yield == 0.0,
        "non_random_directions_help": non_random_yield > random_yield + 1.0,
        "k_expansion_helps": max(k6, k8) > k4 + 1.0,
        "task_screening_continues_to_help": preferred_yield > confident_yield + 1.0,
        "random_yield": random_yield,
        "non_random_yield": non_random_yield,
        "k_yields": {"4": k4, "6": k6, "8": k8},
        "preferred_class_yield": preferred_yield,
        "confident_class_yield": confident_yield,
        "primary_alpha_yield": primary_alpha_yield,
        "alpha02_yield": alpha02_yield,
    }


def driver_verdict(primary_metrics: dict[str, Any], questions: dict[str, Any], top: list[dict[str, Any]]) -> str:
    if primary_metrics["groups"] < 3:
        return "INSUFFICIENT"
    if questions["alpha_0_02_required"]:
        return "ALPHA_0_02_REQUIRED"
    if questions["non_random_directions_help"]:
        return "NON_RANDOM_DIRECTIONS_HELP"
    if questions["k_expansion_helps"]:
        return "K_EXPANSION_HELPS"
    if questions["task_screening_continues_to_help"]:
        return "TASK_SCREENING_ONLY"
    if top and float(top[0]["behaviorally_diverse_groups_per_100_rows"]) > 5.0 and primary_metrics["behaviorally_diverse_groups"] >= 5:
        return "CLEAR_DIVERSITY_RECIPE"
    if primary_metrics["behaviorally_diverse_groups"] > 0:
        return "NO_CLEAR_DRIVER"
    return "INSUFFICIENT"


def display_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        for key in ("non_tie_pairs_per_100_rows", "behaviorally_diverse_groups_per_100_rows", "tie_rate", "stable_rate", "parse_rate"):
            if key in item:
                item[key] = rate(item[key])
        out.append(item)
    return out


def main() -> int:
    started = time.time()
    ensure_v3_root()
    rows = load_v3_branch_rows()
    primary_rows = [row for row in rows if primary_safe_v3_row(row)]
    diagnostic_alpha_rows = [row for row in rows if diagnostic_alpha_v3_row(row)]
    sampled_rows = [row for row in rows if sampled_reward(row) is not None and stable_v2_row(row)]
    if not rows:
        payload = {
            "BG_HIDDEN_ORIGIN_DIVERSITY_DRIVER_V3_VERDICT": "INSUFFICIENT",
            "verdict": "INSUFFICIENT",
            "blocker": "no v3 branch rows",
        }
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# Hidden-Origin Diversity Drivers V3", "", "BG_HIDDEN_ORIGIN_DIVERSITY_DRIVER_V3_VERDICT = INSUFFICIENT"])
        print("BG_HIDDEN_ORIGIN_DIVERSITY_DRIVER_V3_VERDICT = INSUFFICIENT", flush=True)
        return 0

    factors = ["task_screening_class", "branch_point", "alpha_bucket", "K", "delta_family", "domain", "primary_delta_family", "split_guard_role"]
    primary_stats = {factor: condition_group_stats(primary_rows, factor) for factor in factors}
    diagnostic_stats = {factor: condition_group_stats(diagnostic_alpha_rows, factor) for factor in factors}
    sampled_stats = {factor: condition_group_stats(sampled_rows, factor, label_source="sampled_expected") for factor in factors}
    primary_metrics = branch_group_metrics(primary_rows)
    diagnostic_metrics = branch_group_metrics(diagnostic_alpha_rows)
    sampled_metrics = branch_group_metrics(sampled_rows, label_source="sampled_expected")
    top = top_conditions(primary_stats)
    avoid = avoid_conditions(primary_stats)
    questions = answer_questions(primary_stats, diagnostic_stats)
    verdict = driver_verdict(primary_metrics, questions, top)
    recommended_recipe = top[0] if top else None
    payload = {
        "BG_HIDDEN_ORIGIN_DIVERSITY_DRIVER_V3_VERDICT": verdict,
        "verdict": verdict,
        "primary_metrics": primary_metrics,
        "diagnostic_alpha_0_02_metrics": diagnostic_metrics,
        "sampled_expected_metrics": sampled_metrics,
        "primary_stats_by_factor": primary_stats,
        "diagnostic_alpha_0_02_stats_by_factor": diagnostic_stats,
        "sampled_expected_stats_by_factor": sampled_stats,
        "top_10_highest_yield_conditions": top,
        "conditions_to_avoid": avoid,
        "best_stability_diversity_tradeoff": recommended_recipe,
        "questions": questions,
        "recommended_branch_generation_recipe": recommended_recipe,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    lines = [
        "# Hidden-Origin Diversity Drivers V3",
        "",
        f"BG_HIDDEN_ORIGIN_DIVERSITY_DRIVER_V3_VERDICT = {verdict}",
        "",
        f"- primary_metrics: `{primary_metrics}`",
        f"- diagnostic_alpha_0_02_metrics: `{diagnostic_metrics}`",
        f"- sampled_expected_metrics: `{sampled_metrics}`",
        f"- questions: `{questions}`",
        "",
        "## Top 10 Highest-Yield Conditions",
        "",
    ]
    lines.extend(md_table(display_rows(top), ["factor", "condition", "rows", "groups", "behaviorally_diverse_groups", "reward_diverse_groups", "behaviorally_diverse_groups_per_100_rows", "non_tie_pairs_per_100_rows", "tie_rate", "stable_rate", "parse_rate"]))
    lines.extend(["", "## Conditions To Avoid", ""])
    lines.extend(md_table(display_rows(avoid), ["factor", "condition", "groups", "rows", "tie_rate", "stable_rate", "parse_rate"]))
    lines.extend(["", "## Recommended Recipe", "", f"`{recommended_recipe}`"])
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_DIVERSITY_DRIVER_V3_VERDICT = {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

