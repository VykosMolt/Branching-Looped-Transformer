"""Build Branch Generator v1 audit and target plan."""
from __future__ import annotations

import time
from collections import Counter
from typing import Any

from bg_branch_generator_v1_common import (
    AUDIT_PLAN_JSON,
    AUDIT_PLAN_MD,
    BGV1_ROOT,
    PREFERRED_MINIMUMS_V1,
    PRIMARY_MINIMUMS_V1,
    TASK_PLAN_CSV,
    V4_ROOT,
    balanced_pick,
    build_task_priority_rows,
    ensure_bgv1_root,
    load_json,
    md_table,
    rel,
    write_csv,
    write_json,
    write_md,
)


V4_HELDOUT_PRODUCTIVE = ["OpenBookQA/14", "OpenBookQA/18", "mmlu/high_school_chemistry/1"]
PRIOR_HIGH_YIELD = [
    "sciq/sciq/22",
    "mmlu/high_school_chemistry/10",
    "mmlu/high_school_physics/11",
    "mmlu/anatomy/12",
    "mmlu/anatomy/7",
]


def recipe_rows_for_task(task: dict[str, Any], split: str) -> list[dict[str, Any]]:
    cls = str(task.get("task_class") or "perturbation_sensitive")
    k = 8 if split == "heldout" or task.get("priority_tier") == "high" else 6
    base = [
        ("CEM_best_schedule", "L24", "alpha_0_005", "old_tap_aligned", k),
        ("hs_inspired_controller", "L24", "alpha_0_005", "paired_plus_minus", k),
        ("static_v4_best_recipe", "L24", "alpha_0_005", "v2_tap_aligned", k),
        ("branch_generator_proposer", "L24", "alpha_0_005", "salvage_tap_aligned", k),
        ("CEM_best_schedule", "L36", "alpha_0_005", "old_tap_aligned", k),
        ("random_structured_baseline", "L24", "alpha_0_005", "random_orthogonal", min(k, 6)),
        ("hs_inspired_controller", "L24", "alpha_0_01", "old_tap_aligned", k),
        ("static_v4_best_recipe", "L36", "alpha_0_01", "v3_tap_aligned", k),
        ("ES_lite_best_schedule", "L24", "alpha_0_005", "hidden_origin_empirical_train_only", k),
        ("CEM_best_schedule", "L36", "alpha_0_005", "paired_plus_minus", k),
        ("diagnostic_alpha", "L24", "alpha_0_02", "old_tap_aligned", min(k, 6)),
        ("true_fork_carry_diagnostic", "L47_diagnostic", "alpha_0_02", "random_orthogonal", 4),
    ]
    recipes = []
    for idx, (method, branch_point, alpha_bucket, delta_family, kk) in enumerate(base):
        recipes.append(
            {
                "recipe_id": f"bgv1_recipe::{split}::{cls}::{idx}",
                "recipe_source": "audit_plan",
                "generator_method": method,
                "task_class": cls,
                "branch_point": branch_point,
                "K": int(kk),
                "alpha_bucket": alpha_bucket,
                "delta_family": delta_family,
                "decode_mode": "deterministic",
                "target_loop": 1,
                "diagnostic": alpha_bucket == "alpha_0_02" or branch_point == "L47_diagnostic",
            }
        )
    return recipes


def force_pick_by_id(tasks: list[dict[str, Any]], ids: list[str], used: set[str]) -> list[dict[str, Any]]:
    by_id = {str(row["task_id"]): row for row in tasks}
    out = []
    for tid in ids:
        row = by_id.get(tid)
        if row and tid not in used:
            out.append(row)
            used.add(tid)
    return out


def main() -> int:
    started = time.time()
    ensure_bgv1_root()
    tasks = build_task_priority_rows()
    if not tasks:
        payload = {"blocker": "no reasoning/science MCQ tasks available"}
        write_json(AUDIT_PLAN_JSON, {"BG_BRANCH_GENERATOR_V1_AUDIT_PLAN_VERDICT": "BLOCKED", "verdict": "BLOCKED", **payload})
        write_md(AUDIT_PLAN_MD, ["# Branch Generator v1 Audit Plan", "", "BG_BRANCH_GENERATOR_V1_AUDIT_PLAN_VERDICT = BLOCKED"])
        print("BG_BRANCH_GENERATOR_V1_AUDIT_PLAN_VERDICT = BLOCKED", flush=True)
        return 1

    v4_driver = load_json(V4_ROOT / "quota_diversity_drivers.json", {}) or {}
    v4_generation = load_json(V4_ROOT / "quota_generation_report.json", {}) or {}
    used: set[str] = set()
    heldout = force_pick_by_id(tasks, PRIOR_HIGH_YIELD + V4_HELDOUT_PRODUCTIVE, used)
    heldout.extend(balanced_pick(tasks, PREFERRED_MINIMUMS_V1["heldout"]["task_ids"] - len(heldout), used, prefer_clean=False))
    val = balanced_pick(tasks, PREFERRED_MINIMUMS_V1["val"]["task_ids"], used, prefer_clean=True)
    train = balanced_pick(tasks, PREFERRED_MINIMUMS_V1["train"]["task_ids"], used, prefer_clean=False)

    split_rows = []
    for split, rows in (("train", train), ("val", val), ("heldout", heldout)):
        for idx, row in enumerate(rows):
            compact = {
                key: row.get(key)
                for key in (
                    "task_id",
                    "domain",
                    "source_dataset",
                    "source_subject",
                    "subdomain_bucket",
                    "question",
                    "options",
                    "correct_option",
                    "prompt",
                    "task_class",
                    "task_screening_class",
                    "priority_tier",
                    "priority_score_v4",
                    "prior_behaviorally_diverse_groups",
                    "prior_reward_diverse_groups",
                    "prior_non_tie_pairs",
                    "prior_groups",
                    "prior_any_contamination",
                    "prior_train_val_contamination",
                )
            }
            compact["split"] = split
            compact["split_priority_rank"] = idx + 1
            compact["priority_recipes"] = recipe_rows_for_task(compact, split)
            compact["heldout_likely_non_tie"] = compact["task_id"] in set(PRIOR_HIGH_YIELD + V4_HELDOUT_PRODUCTIVE)
            split_rows.append(compact)

    split_counts = Counter(row["split"] for row in split_rows)
    verdict = "READY"
    if split_counts["heldout"] < PRIMARY_MINIMUMS_V1["heldout"]["task_ids"]:
        verdict = "BLOCKED"
    elif split_counts["train"] < PRIMARY_MINIMUMS_V1["train"]["task_ids"] or split_counts["val"] < PRIMARY_MINIMUMS_V1["val"]["task_ids"]:
        verdict = "PARTIAL"

    task_plan_rows = [
        {
            "split": row["split"],
            "task_id": row["task_id"],
            "domain": row.get("domain"),
            "task_class": row.get("task_class"),
            "priority_score_v4": row.get("priority_score_v4"),
            "prior_behaviorally_diverse_groups": row.get("prior_behaviorally_diverse_groups"),
            "prior_non_tie_pairs": row.get("prior_non_tie_pairs"),
            "prior_train_val_contamination": row.get("prior_train_val_contamination"),
            "heldout_likely_non_tie": row.get("heldout_likely_non_tie"),
        }
        for row in split_rows
    ]
    payload: dict[str, Any] = {
        "BG_BRANCH_GENERATOR_V1_AUDIT_PLAN_VERDICT": verdict,
        "verdict": verdict,
        "output_root": rel(BGV1_ROOT),
        "minimum_quotas": PRIMARY_MINIMUMS_V1,
        "preferred_quotas": PREFERRED_MINIMUMS_V1,
        "split_task_ids": {split: [row["task_id"] for row in split_rows if row["split"] == split] for split in ("train", "val", "heldout")},
        "heldout_task_ids_reserved_before_training": [row["task_id"] for row in split_rows if row["split"] == "heldout"],
        "empirical_direction_excluded_task_ids": [row["task_id"] for row in split_rows if row["split"] in {"val", "heldout"}],
        "proposer_training_excluded_task_ids": [row["task_id"] for row in split_rows if row["split"] in {"val", "heldout"}],
        "tap_training_excluded_task_ids": [row["task_id"] for row in split_rows if row["split"] in {"val", "heldout"}],
        "tasks": split_rows,
        "comparison_arms": [
            "static_v4_best_recipe",
            "hs_inspired_recipe_controller_v2",
            "structured_direction_combinations",
            "low_rank_branch_generator_mlp_if_feasible",
            "CEM_ES_recipe_and_coefficient_search",
            "true_fork_carry_smoke_if_feasible",
        ],
        "targeted_recipe_guidance": {
            "primary_branch_points": ["L24", "L36"],
            "early_branch_hypothesis": "L24/L1 and old_tap_aligned directions get more trajectory length to amplify branch differences",
            "primary_alpha_buckets": ["alpha_0_005", "alpha_0_01"],
            "diagnostic_alpha_buckets": ["alpha_0_02"],
            "default_K": 8,
            "v4_lessons": v4_driver.get("questions", {}),
        },
        "heldout_failure_cases_from_v4": (v4_generation.get("quota_progress_by_split") or {}).get("heldout", {}),
        "baseline_contamination_by_split": {
            split: {
                "prior_train_val_contaminated_task_ids": [row["task_id"] for row in split_rows if row["split"] == split and row.get("prior_train_val_contamination")],
                "prior_any_contaminated_task_ids": [row["task_id"] for row in split_rows if row["split"] == split and row.get("prior_any_contamination")],
            }
            for split in ("train", "val", "heldout")
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(AUDIT_PLAN_JSON, payload)
    write_csv(TASK_PLAN_CSV, task_plan_rows)
    lines = [
        "# Branch Generator v1 Audit Plan",
        "",
        f"BG_BRANCH_GENERATOR_V1_AUDIT_PLAN_VERDICT = {verdict}",
        "",
        f"- train_task_ids: `{split_counts['train']}`",
        f"- val_task_ids: `{split_counts['val']}`",
        f"- heldout_task_ids: `{split_counts['heldout']}`",
        f"- heldout_likely_non_tie: `{[row['task_id'] for row in split_rows if row['split'] == 'heldout' and row.get('heldout_likely_non_tie')]}`",
        "",
        "The plan promotes early L24 branch causation, old-tap-aligned directions, K=8, and alpha 0.005 while keeping L36 as a controlled comparison.",
        "",
        "## Task Plan",
        "",
    ]
    lines.extend(md_table(task_plan_rows, ["split", "task_id", "domain", "task_class", "priority_score_v4", "prior_behaviorally_diverse_groups", "prior_non_tie_pairs", "heldout_likely_non_tie", "prior_train_val_contamination"]))
    write_md(AUDIT_PLAN_MD, lines)
    print(f"BG_BRANCH_GENERATOR_V1_AUDIT_PLAN_VERDICT = {verdict}", flush=True)
    print(f"Wrote {rel(AUDIT_PLAN_JSON)}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
