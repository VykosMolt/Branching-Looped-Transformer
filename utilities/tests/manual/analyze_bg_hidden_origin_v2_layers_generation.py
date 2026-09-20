"""Analyze v2 hidden-origin layer/config and generation diversity drivers."""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from bg_hidden_origin_diversity_v2_common import (
    V2_ROOT,
    alpha_bucket,
    deterministic_reward,
    group_is_behaviorally_diverse_v2,
    group_is_reward_diverse_v2,
    group_rows,
    load_all_v2_branch_rows,
    load_json,
    md_table,
    rate,
    safe_primary_row,
    stable_v2_row,
    write_json,
    write_md,
)


EVAL_JSON = V2_ROOT / "heldout_eval_v2.json"
DATASET_JSON = V2_ROOT / "hidden_origin_tap_dataset_v2.json"
OUT_JSON = V2_ROOT / "layer_generation_analysis_v2.json"
OUT_MD = V2_ROOT / "layer_generation_analysis_v2.md"


def config_family(config: str) -> str:
    if config.startswith("24_"):
        return "early_L24"
    if config.startswith("30_") or config.startswith("36_") or config.startswith("42_"):
        return "mid"
    if config.startswith("47_"):
        return "late_L47"
    if config.startswith("concat"):
        return "concat"
    return "unknown"


def group_stats_by(field: str, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups = [vals for vals in group_rows(rows).values() if len(vals) >= 2]
    out = {}
    values = sorted({str(vals[0].get(field) if field != "alpha_bucket" else alpha_bucket(vals[0])) for vals in groups})
    for value in values:
        subset = [
            vals
            for vals in groups
            if str(vals[0].get(field) if field != "alpha_bucket" else alpha_bucket(vals[0])) == value
        ]
        diverse = [vals for vals in subset if group_is_behaviorally_diverse_v2(vals)]
        reward_diverse = [vals for vals in subset if group_is_reward_diverse_v2(vals)]
        out[value] = {
            "groups": len(subset),
            "behaviorally_diverse_groups": len(diverse),
            "reward_diverse_groups": len(reward_diverse),
            "behaviorally_diverse_rate": len(diverse) / max(len(subset), 1),
            "reward_diverse_rate": len(reward_diverse) / max(len(subset), 1),
        }
    return out


def delta_family_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for family in sorted({str(row.get("delta_family")) for row in rows}):
        vals = [row for row in rows if str(row.get("delta_family")) == family]
        rewards = [deterministic_reward(row) for row in vals]
        out[family] = {
            "branches": len(vals),
            "mean_reward": float(mean(rewards)) if rewards else 0.0,
            "correct_rate": sum(1 for row in vals if bool(row.get("correct"))) / max(len(vals), 1),
            "parse_failure_rate": sum(1 for row in vals if not bool(row.get("parse_success"))) / max(len(vals), 1),
            "oracle_branch_count": 0,
        }
    for vals in group_rows(rows).values():
        if len(vals) < 2:
            continue
        oracle = max(deterministic_reward(row) for row in vals)
        for row in vals:
            if deterministic_reward(row) == oracle:
                fam = str(row.get("delta_family"))
                if fam in out:
                    out[fam]["oracle_branch_count"] += 1
    return out


def layer_verdict(eval_payload: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any] | None, str]:
    behavior = eval_payload.get("behaviorally_diverse_metrics", {}) or {}
    config_rows = []
    for row in behavior.values():
        policy = str(row.get("policy") or "")
        if not policy.startswith("new_hidden_origin_tap_v2_config_") or not policy.endswith("_pairwise_tournament"):
            continue
        config_rows.append(
            {
                "config": row.get("config"),
                "family": config_family(str(row.get("config"))),
                "architecture": row.get("architecture"),
                "groups": row.get("groups"),
                "top1_success": row.get("top1_success"),
                "reward_mean": row.get("reward_mean"),
                "selection_regret": row.get("selection_regret"),
            }
        )
    best = max(config_rows, key=lambda row: (float(row.get("top1_success", 0.0)), float(row.get("reward_mean", 0.0)), -float(row.get("selection_regret", 999.0)))) if config_rows else None
    if eval_payload.get("verdict") in {"INSUFFICIENT", "DATA_LIMITED"} or not config_rows:
        return "INSUFFICIENT", best, "insufficient"
    if best is None:
        return "NO_CLEAR_LAYER", None, "insufficient"
    fam = best["family"]
    if fam == "early_L24":
        return "EARLY_READOUT_SUFFICIENT", best, "L24"
    if fam == "mid":
        return "MID_LAYER_BEST", best, "L36"
    if fam == "late_L47":
        return "LATE_READOUT_REQUIRED", best, "L47"
    if fam == "concat":
        return "CONCAT_REQUIRED", best, "concat"
    return "NO_CLEAR_LAYER", best, "insufficient"


def diversity_source_verdict(class_stats: dict[str, Any], alpha_stats: dict[str, Any], delta_stats: dict[str, Any]) -> str:
    preferred = [
        class_stats.get(name, {}).get("behaviorally_diverse_rate", 0.0)
        for name in ("perturbation_sensitive", "baseline_wrong_parseable", "baseline_parse_fragile", "baseline_correct_low_confidence")
        if name in class_stats
    ]
    confident = class_stats.get("baseline_correct_confident", {}).get("behaviorally_diverse_rate", 0.0)
    if preferred and max(preferred) > confident + 0.05:
        return "TASK_SCREENING_HELPS"
    alpha02 = alpha_stats.get("alpha_0_02", {}).get("behaviorally_diverse_rate", 0.0)
    primary = max(
        alpha_stats.get("alpha_0_005", {}).get("behaviorally_diverse_rate", 0.0),
        alpha_stats.get("alpha_0_01", {}).get("behaviorally_diverse_rate", 0.0),
    )
    if alpha02 > primary + 0.05:
        return "ALPHA_0_02_HELPS"
    non_random = [v.get("mean_reward", 0.0) for k, v in delta_stats.items() if k not in {"random", "clean"}]
    random_mean = delta_stats.get("random", {}).get("mean_reward", 0.0)
    if non_random and max(non_random) > random_mean + 0.05:
        return "DELTA_FAMILY_HELPS"
    if class_stats or alpha_stats:
        return "NO_CLEAR_DIVERSITY_DRIVER"
    return "INSUFFICIENT"


def main() -> int:
    started = time.time()
    rows = [row for row in load_all_v2_branch_rows(include_prior=True) if stable_v2_row(row)]
    primary_rows = [row for row in rows if safe_primary_row(row)]
    eval_payload = load_json(EVAL_JSON, {}) or {}
    dataset_payload = load_json(DATASET_JSON, {}) or {}
    class_stats = group_stats_by("task_screening_class", primary_rows)
    branch_point_stats = group_stats_by("branch_point", primary_rows)
    alpha_stats = group_stats_by("alpha_bucket", rows)
    domain_stats = group_stats_by("domain", primary_rows)
    delta_stats = delta_family_stats(primary_rows)
    layer_v, best_config, recommendation = layer_verdict(eval_payload, primary_rows)
    source_v = diversity_source_verdict(class_stats, alpha_stats, delta_stats)

    payload = {
        "BG_HIDDEN_ORIGIN_DIVERSITY_SOURCE_VERDICT": source_v,
        "BG_HIDDEN_ORIGIN_LAYER_CONFIG_V2_VERDICT": layer_v,
        "diversity_source_verdict": source_v,
        "layer_config_verdict": layer_v,
        "dataset_verdict": dataset_payload.get("verdict"),
        "eval_verdict": eval_payload.get("verdict"),
        "best_behaviorally_diverse_config": best_config,
        "first_phase2_scoring_point_recommendation": recommendation,
        "task_screening_class_stats": class_stats,
        "branch_point_stats": branch_point_stats,
        "alpha_bucket_stats": alpha_stats,
        "domain_stats": domain_stats,
        "delta_family_stats": delta_stats,
        "questions": {
            "task_screening_classes_produced_diversity": source_v == "TASK_SCREENING_HELPS",
            "l24_or_l36_branch_point_better": branch_point_stats,
            "alpha_0_02_improved_diversity": source_v == "ALPHA_0_02_HELPS",
            "non_random_direction_family_helped": source_v == "DELTA_FAMILY_HELPS",
            "sampled_expected_reward_available": bool((dataset_payload.get("variant_meta") or {}).get("sampled_expected_diagnostic", {}).get("candidate_unordered_pairs")),
            "concat_24_36_still_best": bool(best_config and best_config.get("config") == "concat_24_36"),
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)

    class_rows = [
        {
            "class": key,
            "groups": val["groups"],
            "diverse": val["behaviorally_diverse_groups"],
            "diverse_rate": rate(val["behaviorally_diverse_rate"]),
        }
        for key, val in sorted(class_stats.items())
    ]
    alpha_rows = [
        {
            "alpha": key,
            "groups": val["groups"],
            "diverse": val["behaviorally_diverse_groups"],
            "diverse_rate": rate(val["behaviorally_diverse_rate"]),
        }
        for key, val in sorted(alpha_stats.items())
    ]
    delta_rows = [
        {
            "family": key,
            "branches": val["branches"],
            "mean_reward": rate(val["mean_reward"]),
            "correct_rate": rate(val["correct_rate"]),
            "oracle_branches": val["oracle_branch_count"],
        }
        for key, val in sorted(delta_stats.items())
    ]
    lines = [
        "# Hidden-Origin V2 Layer And Generation Analysis",
        "",
        f"BG_HIDDEN_ORIGIN_DIVERSITY_SOURCE_VERDICT = {source_v}",
        f"BG_HIDDEN_ORIGIN_LAYER_CONFIG_V2_VERDICT = {layer_v}",
        "",
        f"- dataset_verdict: `{dataset_payload.get('verdict')}`",
        f"- eval_verdict: `{eval_payload.get('verdict')}`",
        f"- best_behaviorally_diverse_config: `{best_config}`",
        f"- first_phase2_scoring_point_recommendation: `{recommendation}`",
        "",
        "## Screening Class Diversity",
        "",
    ]
    lines.extend(md_table(class_rows, ["class", "groups", "diverse", "diverse_rate"]))
    lines.extend(["", "## Alpha Buckets", ""])
    lines.extend(md_table(alpha_rows, ["alpha", "groups", "diverse", "diverse_rate"]))
    lines.extend(["", "## Delta Families", ""])
    lines.extend(md_table(delta_rows, ["family", "branches", "mean_reward", "correct_rate", "oracle_branches"]))
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_DIVERSITY_SOURCE_VERDICT = {source_v}", flush=True)
    print(f"BG_HIDDEN_ORIGIN_LAYER_CONFIG_V2_VERDICT = {layer_v}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

