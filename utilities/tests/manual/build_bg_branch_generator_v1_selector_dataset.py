"""Build pairwise selector dataset from Branch Generator v1 outputs."""
from __future__ import annotations

import time
from collections import Counter
from typing import Any

import torch

from bg_branch_generator_v1_common import (
    SELECTOR_DATASET_JSON,
    SELECTOR_DATASET_MD,
    SELECTOR_DATASET_PT,
    build_pairs_from_rows,
    compact_pair,
    dataset_split_counts,
    diagnostic_generator_row,
    ensure_bgv1_root,
    load_generator_rows,
    md_table,
    primary_safe_generator_row,
    rate,
    sampled_reward,
    split_quota_stats_v1,
    stable_v2_row,
    verdict_for_selector_dataset_v1,
    write_json,
    write_md,
)


def high_yield_row(row: dict[str, Any]) -> bool:
    return primary_safe_generator_row(row) and str(row.get("primary_delta_family") or row.get("delta_family")) not in {"random_orthogonal", "paired_plus_minus", "random_fallback"}


def sampled_row(row: dict[str, Any]) -> bool:
    return stable_v2_row(row) and sampled_reward(row) is not None and str(row.get("split")) in {"train", "val", "heldout"}


def true_fork_row(row: dict[str, Any]) -> bool:
    return stable_v2_row(row) and str(row.get("branch_method")) == "true_fork_carry"


def main() -> int:
    started = time.time()
    ensure_bgv1_root()
    rows = load_generator_rows()
    if not rows:
        payload = {"BG_BRANCH_GENERATOR_V1_SELECTOR_DATASET_VERDICT": "BLOCKED", "verdict": "BLOCKED", "blocker": "no Branch Generator v1 rows"}
        write_json(SELECTOR_DATASET_JSON, payload)
        write_md(SELECTOR_DATASET_MD, ["# Branch Generator V1 Selector Dataset", "", "BG_BRANCH_GENERATOR_V1_SELECTOR_DATASET_VERDICT = BLOCKED"])
        print("BG_BRANCH_GENERATOR_V1_SELECTOR_DATASET_VERDICT = BLOCKED", flush=True)
        return 1
    variants = {
        "primary_safe_deterministic": (primary_safe_generator_row, "deterministic"),
        "alpha_0_02_diagnostic": (diagnostic_generator_row, "deterministic"),
        "sampled_expected_diagnostic": (sampled_row, "sampled_expected"),
        "high_yield_generator_subset": (high_yield_row, "deterministic"),
        "true_fork_carry_diagnostic": (true_fork_row, "deterministic"),
    }
    pairs_by_variant: dict[str, list[dict[str, Any]]] = {}
    meta_by_variant: dict[str, Any] = {}
    tie_rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    for name, (row_filter, label_source) in variants.items():
        pairs, meta, ties = build_pairs_from_rows(rows, variant=name, label_source=label_source, row_filter=row_filter)
        pairs_by_variant[name] = pairs
        meta_by_variant[name] = meta
        tie_rows_by_variant[name] = ties
    primary_pairs = pairs_by_variant["primary_safe_deterministic"]
    quota = split_quota_stats_v1(rows)
    split_counts = dataset_split_counts(primary_pairs, [row for row in rows if primary_safe_generator_row(row)])
    verdict = verdict_for_selector_dataset_v1(quota, len(primary_pairs))
    payload: dict[str, Any] = {
        "BG_BRANCH_GENERATOR_V1_SELECTOR_DATASET_VERDICT": verdict,
        "verdict": verdict,
        "pairs": primary_pairs,
        "pairs_by_variant": pairs_by_variant,
        "meta_by_variant": meta_by_variant,
        "tie_rows_by_variant": tie_rows_by_variant,
        "quota_progress_by_split": quota,
        "split_counts": split_counts,
        "tasks_by_split": split_counts["tasks_by_split"],
        "augmented_train_used": False,
        "augmentation_policy": "not_used; generator-v1-only primary-safe headline",
        "primary_pair_count": len(primary_pairs),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(payload, SELECTOR_DATASET_PT)
    compact_payload = {k: v for k, v in payload.items() if k not in {"pairs", "pairs_by_variant", "tie_rows_by_variant"}}
    compact_payload["pairs"] = [compact_pair(pair) for pair in primary_pairs]
    compact_payload["pairs_by_variant"] = {name: [compact_pair(pair) for pair in vals] for name, vals in pairs_by_variant.items()}
    compact_payload["tie_rows_by_variant"] = tie_rows_by_variant
    write_json(SELECTOR_DATASET_JSON, compact_payload)
    rows_md = []
    for split in ("train", "val", "heldout"):
        rows_md.append(
            {
                "split": split,
                "task_ids": len(split_counts["tasks_by_split"].get(split, [])),
                "pairs": split_counts["pairs_by_split"].get(split, 0),
                "groups": split_counts["groups_by_split"].get(split, 0),
                "behaviorally_diverse_groups": split_counts["behaviorally_diverse_groups_by_split"].get(split, 0),
                "reward_diverse_groups": split_counts["reward_diverse_groups_by_split"].get(split, 0),
                "quota_minimum_met": quota.get(split, {}).get("minimum_met"),
                "tie_rate": rate(quota.get(split, {}).get("tie_rate")),
            }
        )
    lines = [
        "# Branch Generator V1 Selector Dataset",
        "",
        f"BG_BRANCH_GENERATOR_V1_SELECTOR_DATASET_VERDICT = {verdict}",
        "",
        f"- primary_pair_count: `{len(primary_pairs)}`",
        f"- augmented_train_used: `False`",
        f"- meta_by_variant: `{meta_by_variant}`",
        "",
        "## Split Counts",
        "",
    ]
    lines.extend(md_table(rows_md, ["split", "task_ids", "pairs", "groups", "behaviorally_diverse_groups", "reward_diverse_groups", "quota_minimum_met", "tie_rate"]))
    lines.extend(["", "## Distributions", "", f"- domain: `{dict(Counter(row.get('domain') for row in rows if primary_safe_generator_row(row)))}`", f"- branch_point: `{dict(Counter(row.get('branch_point') for row in rows if primary_safe_generator_row(row)))}`", f"- alpha: `{dict(Counter(row.get('alpha_bucket') for row in rows if primary_safe_generator_row(row)))}`", f"- delta_family: `{dict(Counter(row.get('primary_delta_family') or row.get('delta_family') for row in rows if primary_safe_generator_row(row)))}`", f"- generator_method: `{dict(Counter(row.get('generator_method') for row in rows if primary_safe_generator_row(row)))}`"])
    write_md(SELECTOR_DATASET_MD, lines)
    print(f"BG_BRANCH_GENERATOR_V1_SELECTOR_DATASET_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
