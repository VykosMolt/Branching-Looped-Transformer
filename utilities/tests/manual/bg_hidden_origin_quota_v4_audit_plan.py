"""Build quota-directed hidden-origin v4 generation plan."""
from __future__ import annotations

import time
from collections import Counter
from typing import Any

from bg_hidden_origin_quota_v4_common import (
    PREFERRED_MINIMUMS,
    PRIMARY_MINIMUMS,
    QUOTA_PLAN_CSV,
    QUOTA_PLAN_JSON,
    QUOTA_PLAN_MD,
    V4_ROOT,
    balanced_pick,
    build_task_priority_rows,
    ensure_v4_root,
    load_json,
    md_table,
    rel,
    write_csv,
    write_json,
    write_md,
)


def recipe_priorities_for_task(task: dict[str, Any], split: str) -> list[dict[str, Any]]:
    cls = str(task.get("task_class") or "wrong_parseable")
    k = 8 if task.get("priority_tier") == "high" or int(task.get("prior_behaviorally_diverse_groups") or 0) > 0 else 6
    primary_families = list(
        dict.fromkeys(
            [
                "hidden_origin_empirical_train_only",
                "v3_tap_aligned",
                "v2_tap_aligned",
                "salvage_tap_aligned",
                "old_tap_aligned",
                "paired_plus_minus",
                "random_orthogonal",
                "empirical_plus_noise",
            ]
        )
    )
    if split == "heldout":
        primary_families = [
            "v3_tap_aligned",
            "v2_tap_aligned",
            "salvage_tap_aligned",
            "old_tap_aligned",
            "hidden_origin_empirical_train_only",
            "paired_plus_minus",
            "random_orthogonal",
        ]
    rows = []
    for branch_point, alpha_bucket, family in (
        ("L36", "alpha_0_01", primary_families[0]),
        ("L36", "alpha_0_005", primary_families[1]),
        ("L24", "alpha_0_01", primary_families[2]),
        ("L24", "alpha_0_005", primary_families[3]),
        ("L36", "alpha_0_01", primary_families[4]),
        ("L24", "alpha_0_01", primary_families[5]),
        ("L36", "alpha_0_02", primary_families[6]),
        ("L47_diagnostic", "alpha_0_02", "random_orthogonal"),
    ):
        diagnostic = alpha_bucket == "alpha_0_02" or branch_point == "L47_diagnostic"
        rows.append(
            {
                "task_class": cls,
                "branch_point": branch_point,
                "K": k,
                "alpha_bucket": alpha_bucket,
                "delta_family": family,
                "decode_mode": "deterministic",
                "diagnostic": diagnostic,
            }
        )
    return rows


def main() -> int:
    started = time.time()
    ensure_v4_root()
    tasks = build_task_priority_rows()
    if not tasks:
        payload = {
            "BG_HIDDEN_ORIGIN_QUOTA_PLAN_V4_VERDICT": "BLOCKED",
            "verdict": "BLOCKED",
            "blocker": "no reasoning/science MCQ task pool available",
        }
        write_json(QUOTA_PLAN_JSON, payload)
        write_md(QUOTA_PLAN_MD, ["# Hidden-Origin Quota Plan V4", "", "BG_HIDDEN_ORIGIN_QUOTA_PLAN_V4_VERDICT = BLOCKED"])
        print("BG_HIDDEN_ORIGIN_QUOTA_PLAN_V4_VERDICT = BLOCKED", flush=True)
        return 1

    used: set[str] = set()
    heldout = balanced_pick(tasks, PREFERRED_MINIMUMS["heldout"]["task_ids"], used, prefer_clean=True)
    val = balanced_pick(tasks, PREFERRED_MINIMUMS["val"]["task_ids"], used, prefer_clean=True)
    train = balanced_pick(tasks, PREFERRED_MINIMUMS["train"]["task_ids"], used, prefer_clean=False)
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
            compact["priority_recipes"] = recipe_priorities_for_task(row, split)
            split_rows.append(compact)

    split_counts = Counter(row["split"] for row in split_rows)
    verdict = "READY"
    if split_counts["heldout"] < PRIMARY_MINIMUMS["heldout"]["task_ids"]:
        verdict = "BLOCKED"
    elif split_counts["train"] < PRIMARY_MINIMUMS["train"]["task_ids"] or split_counts["val"] < PRIMARY_MINIMUMS["val"]["task_ids"]:
        verdict = "PARTIAL"
    high_yield = sorted(
        [
            {
                "task_id": row["task_id"],
                "split": row.get("split"),
                "domain": row.get("domain"),
                "class": row.get("task_class"),
                "priority_score_v4": row.get("priority_score_v4"),
                "prior_behaviorally_diverse_groups": row.get("prior_behaviorally_diverse_groups"),
                "prior_non_tie_pairs": row.get("prior_non_tie_pairs"),
                "prior_train_val_contamination": row.get("prior_train_val_contamination"),
            }
            for row in split_rows
        ],
        key=lambda row: (float(row.get("priority_score_v4") or 0.0), int(row.get("prior_non_tie_pairs") or 0)),
        reverse=True,
    )
    contamination = {
        split: {
            "task_ids": [row["task_id"] for row in split_rows if row["split"] == split],
            "prior_train_val_contaminated_task_ids": [row["task_id"] for row in split_rows if row["split"] == split and row.get("prior_train_val_contamination")],
            "prior_any_contaminated_task_ids": [row["task_id"] for row in split_rows if row["split"] == split and row.get("prior_any_contamination")],
        }
        for split in ("train", "val", "heldout")
    }
    payload: dict[str, Any] = {
        "BG_HIDDEN_ORIGIN_QUOTA_PLAN_V4_VERDICT": verdict,
        "verdict": verdict,
        "output_root": rel(V4_ROOT),
        "minimum_quotas": PRIMARY_MINIMUMS,
        "preferred_quotas": PREFERRED_MINIMUMS,
        "split_task_ids": {split: [row["task_id"] for row in split_rows if row["split"] == split] for split in ("train", "val", "heldout")},
        "heldout_task_ids_reserved_before_direction_bank": [row["task_id"] for row in split_rows if row["split"] == "heldout"],
        "empirical_direction_excluded_task_ids": [row["task_id"] for row in split_rows if row["split"] in {"val", "heldout"}],
        "v4_tap_training_excluded_task_ids": [row["task_id"] for row in split_rows if row["split"] in {"val", "heldout"}],
        "direction_bank_empirical_train_task_ids": [row["task_id"] for row in split_rows if row["split"] == "train"],
        "tasks": split_rows,
        "high_yield_tasks": high_yield[:40],
        "domain_coverage": {split: dict(Counter(row.get("domain") for row in split_rows if row["split"] == split)) for split in ("train", "val", "heldout")},
        "task_class_coverage": {split: dict(Counter(row.get("task_class") for row in split_rows if row["split"] == split)) for split in ("train", "val", "heldout")},
        "baseline_contamination_by_split": contamination,
        "recipe_guidance": {
            "prioritize_non_random_directions": True,
            "primary_branch_point": "L36",
            "comparative_branch_point": "L24",
            "diagnostic_branch_point": "L47_diagnostic",
            "primary_alpha_buckets": ["alpha_0_005", "alpha_0_01"],
            "diagnostic_alpha_buckets": ["alpha_0_02"],
            "default_K": 6,
            "high_priority_K": 8,
            "task_classes": ["perturbation_sensitive", "wrong_parseable", "parse_fragile", "low_confidence_correct", "evaluator_disagreement"],
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(QUOTA_PLAN_JSON, payload)
    write_csv(QUOTA_PLAN_CSV, high_yield)
    lines = [
        "# Hidden-Origin Quota Plan V4",
        "",
        f"BG_HIDDEN_ORIGIN_QUOTA_PLAN_V4_VERDICT = {verdict}",
        "",
        f"- train_task_ids: `{split_counts['train']}`",
        f"- val_task_ids: `{split_counts['val']}`",
        f"- heldout_task_ids: `{split_counts['heldout']}`",
        f"- heldout reserved before direction bank: `{payload['heldout_task_ids_reserved_before_direction_bank']}`",
        "",
        "Heldout task IDs are excluded from empirical direction construction and v4 tap training. Baseline contamination is reported separately for old/v1/v2/v3/salvage selectors.",
        "",
        "## Domain Coverage",
        "",
        f"`{payload['domain_coverage']}`",
        "",
        "## High-Yield Planned Tasks",
        "",
    ]
    lines.extend(md_table(high_yield[:60], ["task_id", "split", "domain", "class", "priority_score_v4", "prior_behaviorally_diverse_groups", "prior_non_tie_pairs", "prior_train_val_contamination"]))
    lines.extend(["", "## Outputs", "", f"- JSON: `{rel(QUOTA_PLAN_JSON)}`", f"- CSV: `{rel(QUOTA_PLAN_CSV)}`"])
    write_md(QUOTA_PLAN_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_QUOTA_PLAN_V4_VERDICT = {verdict}", flush=True)
    print(f"Wrote {rel(QUOTA_PLAN_JSON)}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
