"""Build clean pairwise selector dataset from v4 quota branches."""
from __future__ import annotations

import time
from collections import Counter
from typing import Any

import torch

from bg_hidden_origin_quota_v4_common import (
    DATASET_V4_JSON,
    DATASET_V4_MD,
    DATASET_V4_PT,
    PRIMARY_MINIMUMS,
    build_pairs_from_rows,
    compact_pair,
    dataset_split_counts,
    diagnostic_alpha_v4_row,
    ensure_v4_root,
    group_rows,
    load_v4_branch_rows,
    md_table,
    primary_safe_v4_row,
    rate,
    sampled_reward,
    split_quota_stats,
    stable_v2_row,
    write_json,
    write_md,
)


def high_yield_row(row: dict[str, Any]) -> bool:
    return primary_safe_v4_row(row) and str(row.get("primary_delta_family") or row.get("delta_family")) not in {"random_orthogonal", "paired_plus_minus", "random_fallback"}


def sampled_row(row: dict[str, Any]) -> bool:
    return stable_v2_row(row) and sampled_reward(row) is not None and str(row.get("split")) in {"train", "val", "heldout"}


def l47_row(row: dict[str, Any]) -> bool:
    return stable_v2_row(row) and str(row.get("branch_point")) == "L47" and str(row.get("split")) in {"train", "val", "heldout"}


def verdict_for(quota: dict[str, Any], pairs: list[dict[str, Any]]) -> str:
    if not pairs:
        return "BLOCKED"
    ready = bool(quota.get("all_minimums_met"))
    heldout = quota.get("heldout", {})
    train = quota.get("train", {})
    val = quota.get("val", {})
    heldout_ready = bool(heldout.get("minimum_met"))
    train_weak = int(train.get("non_tie_pairs") or 0) < PRIMARY_MINIMUMS["train"]["non_tie_pairs"]
    val_weak = int(val.get("non_tie_pairs") or 0) < PRIMARY_MINIMUMS["val"]["non_tie_pairs"]
    if ready:
        return "READY"
    if heldout_ready and not (train_weak and val_weak):
        return "SMALL_BUT_USABLE"
    if heldout_ready:
        return "HELDOUT_READY_TRAIN_WEAK"
    return "STILL_DATA_LIMITED"


def main() -> int:
    started = time.time()
    ensure_v4_root()
    rows = load_v4_branch_rows()
    if not rows:
        payload = {"BG_HIDDEN_ORIGIN_QUOTA_DATASET_V4_VERDICT": "BLOCKED", "verdict": "BLOCKED", "blocker": "no v4 branch rows"}
        write_json(DATASET_V4_JSON, payload)
        write_md(DATASET_V4_MD, ["# Hidden-Origin Quota Dataset V4", "", "BG_HIDDEN_ORIGIN_QUOTA_DATASET_V4_VERDICT = BLOCKED"])
        print("BG_HIDDEN_ORIGIN_QUOTA_DATASET_V4_VERDICT = BLOCKED", flush=True)
        return 1
    variants = {
        "primary_safe_deterministic": (primary_safe_v4_row, "deterministic"),
        "alpha_0_02_diagnostic": (diagnostic_alpha_v4_row, "deterministic"),
        "sampled_expected_diagnostic": (sampled_row, "sampled_expected"),
        "L47_diagnostic": (l47_row, "deterministic"),
        "high_yield_recipe_subset": (high_yield_row, "deterministic"),
    }
    pairs_by_variant: dict[str, list[dict[str, Any]]] = {}
    meta_by_variant: dict[str, Any] = {}
    tie_rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    for name, (row_filter, label_source) in variants.items():
        pairs, meta, tie_rows = build_pairs_from_rows(rows, variant=name, label_source=label_source, row_filter=row_filter)
        pairs_by_variant[name] = pairs
        meta_by_variant[name] = meta
        tie_rows_by_variant[name] = tie_rows
    primary_pairs = pairs_by_variant["primary_safe_deterministic"]
    quota = split_quota_stats(rows)
    split_counts = dataset_split_counts(primary_pairs, [row for row in rows if primary_safe_v4_row(row)])
    verdict = verdict_for(quota, primary_pairs)
    payload: dict[str, Any] = {
        "BG_HIDDEN_ORIGIN_QUOTA_DATASET_V4_VERDICT": verdict,
        "verdict": verdict,
        "pairs": primary_pairs,
        "pairs_by_variant": pairs_by_variant,
        "meta_by_variant": meta_by_variant,
        "tie_rows_by_variant": tie_rows_by_variant,
        "quota_progress_by_split": quota,
        "split_counts": split_counts,
        "tasks_by_split": split_counts["tasks_by_split"],
        "augmented_train_used": False,
        "augmentation_policy": "not_used; v4-only primary-safe headline",
        "primary_pair_count": len(primary_pairs),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(payload, DATASET_V4_PT)
    compact_payload = {k: v for k, v in payload.items() if k not in {"pairs", "pairs_by_variant", "tie_rows_by_variant"}}
    compact_payload["pairs"] = [compact_pair(pair) for pair in primary_pairs]
    compact_payload["pairs_by_variant"] = {name: [compact_pair(pair) for pair in vals] for name, vals in pairs_by_variant.items()}
    compact_payload["tie_rows_by_variant"] = tie_rows_by_variant
    write_json(DATASET_V4_JSON, compact_payload)
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
        "# Hidden-Origin Quota Dataset V4",
        "",
        f"BG_HIDDEN_ORIGIN_QUOTA_DATASET_V4_VERDICT = {verdict}",
        "",
        f"- primary_pair_count: `{len(primary_pairs)}`",
        f"- augmented_train_used: `False`",
        f"- meta_by_variant: `{meta_by_variant}`",
        "",
        "## Split Counts",
        "",
    ]
    lines.extend(md_table(rows_md, ["split", "task_ids", "pairs", "groups", "behaviorally_diverse_groups", "reward_diverse_groups", "quota_minimum_met", "tie_rate"]))
    lines.extend(["", "## Distributions", "", f"- domain: `{dict(Counter(row.get('domain') for row in rows if primary_safe_v4_row(row)))}`", f"- branch_point: `{dict(Counter(row.get('branch_point') for row in rows if primary_safe_v4_row(row)))}`", f"- alpha: `{dict(Counter(row.get('alpha_bucket') for row in rows if primary_safe_v4_row(row)))}`", f"- delta_family: `{dict(Counter(row.get('primary_delta_family') or row.get('delta_family') for row in rows if primary_safe_v4_row(row)))}`", f"- recipe_source: `{dict(Counter(row.get('recipe_source') for row in rows if primary_safe_v4_row(row)))}`"])
    write_md(DATASET_V4_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_QUOTA_DATASET_V4_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
