"""Shared helpers for hidden-origin split-salvage probes.

The salvage probes only read existing hidden-origin branch/tap artifacts,
rebuild task-disjoint pairwise datasets, and train tiny comparator heads.  They
do not generate branches, train Ouro, mutate tokenizer/checkpoint/model files,
update old tap registries, or collapse reward ties into labels.
"""
from __future__ import annotations

import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Iterable, Sequence

import torch

from bg_hidden_origin_diversity_v3_common import (
    CONFIGS,
    DATASET_V3_PT,
    HEADS_V3_PT,
    PRIMARY_DOMAINS,
    V1_ROOT,
    V2_ROOT,
    V3_ROOT,
    alpha_bucket,
    compact_head_id,
    config_dim,
    config_vector_from_row,
    deterministic_reward,
    diagnostic_alpha_v3_row,
    group_is_behaviorally_diverse_v2,
    group_is_reward_diverse_v2,
    group_rows,
    load_all_v3_branch_rows,
    load_head_rows,
    load_json,
    md_table,
    primary_safe_v3_row,
    rate,
    rel,
    row_reward,
    sampled_reward,
    stable_v2_row,
    write_csv,
    write_json,
    write_md,
)
from bg_hidden_origin_tap_common import (
    PROBE_ROOT,
    append_doc_section,
    pairwise_accuracy_from_pairs,
    ranking_from_matrix,
    score_matrix,
    top2_random_reward,
    top2_random_success,
)
from evaluate_bg_hidden_origin_taps import build_head, selected_metric
from train_bg_hidden_origin_taps import ARCHITECTURES, compact_head, flip_diagnostics, pairs_for_config, train_one


SALVAGE_ROOT = PROBE_ROOT / "bg_hidden_origin_split_salvage_2026-05-18"
AUDIT_JSON = SALVAGE_ROOT / "audit.json"
EVAL_MODES_JSON = SALVAGE_ROOT / "eval_modes.json"
SALVAGE_DATASETS_PT = SALVAGE_ROOT / "salvage_datasets.pt"
SALVAGE_HEADS_PT = SALVAGE_ROOT / "salvage_heads.pt"
SALVAGE_EVAL_JSON = SALVAGE_ROOT / "salvage_selector_eval.json"
CV_STABILITY_JSON = SALVAGE_ROOT / "cv_stability.json"
V4_QUOTA_JSON = SALVAGE_ROOT / "v4_quota_need.json"

USEFUL_CONFIGS = (
    "30_L4",
    "concat_24_30_36",
    "concat_24_36_47",
    "concat_24_36",
    "concat_36_47",
    "24_L4",
    "36_L4",
    "47_L4",
)
READINESS_MINIMUMS = {"task_ids": 6, "behaviorally_diverse_groups": 15, "non_tie_pairs": 80}
WEAK_MINIMUMS = {"task_ids": 4, "behaviorally_diverse_groups": 8, "non_tie_pairs": 40}
SEED = 20260518


def ensure_salvage_root() -> None:
    SALVAGE_ROOT.mkdir(parents=True, exist_ok=True)


def load_pt(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return torch.load(path, map_location="cpu", weights_only=False)


def version_for_row(row: dict[str, Any]) -> str:
    gid = str(row.get("branch_group_id") or "")
    if gid.startswith("v3::"):
        return "v3"
    if gid.startswith("v2::"):
        return "v2"
    return "v1"


def primary_rows() -> list[dict[str, Any]]:
    return [row for row in load_all_v3_branch_rows(include_prior=True) if primary_safe_v3_row(row)]


def diagnostic_alpha_rows() -> list[dict[str, Any]]:
    return [row for row in load_all_v3_branch_rows(include_prior=True) if diagnostic_alpha_v3_row(row)]


def sampled_expected_rows() -> list[dict[str, Any]]:
    return [
        row
        for row in load_all_v3_branch_rows(include_prior=True)
        if stable_v2_row(row)
        and str(row.get("domain") or "").lower() in PRIMARY_DOMAINS
        and sampled_reward(row) is not None
    ]


def task_sets_from_dataset(path: Path) -> dict[str, set[str]]:
    payload = load_pt(path, {}) or {}
    tasks = payload.get("tasks_by_split") or {}
    out: dict[str, set[str]] = {name: {str(x) for x in vals} for name, vals in tasks.items()}
    out.setdefault("train", set())
    out.setdefault("val", set())
    out.setdefault("test", set())
    out["train_val"] = set(out["train"]) | set(out["val"])
    out["all"] = set().union(*out.values()) if out else set()
    return out


def prior_seen_sets() -> dict[str, Any]:
    v1 = task_sets_from_dataset(V1_ROOT / "hidden_origin_tap_dataset.pt")
    v2 = task_sets_from_dataset(V2_ROOT / "hidden_origin_tap_dataset_v2.pt")
    v3 = task_sets_from_dataset(DATASET_V3_PT)
    split_guard = load_json(V3_ROOT / "split_guard_v3.json", {}) or {}
    return {
        "v1": {k: sorted(v) for k, v in v1.items()},
        "v2": {k: sorted(v) for k, v in v2.items()},
        "v3": {k: sorted(v) for k, v in v3.items()},
        "v1_v2_train_val": sorted(set(v1["train_val"]) | set(v2["train_val"])),
        "v1_v2_any": sorted(set(v1["all"]) | set(v2["all"])),
        "v3_train_val": sorted(v3["train_val"]),
        "v3_any": sorted(v3["all"]),
        "v3_heldout_candidate_task_ids": sorted(str(x) for x in split_guard.get("v3_heldout_candidate_task_ids", [])),
        "clean_cross_version_heldout_task_ids": sorted(str(x) for x in split_guard.get("clean_cross_version_heldout_task_ids", [])),
        "v3_empirical_direction_train_eligible_task_ids": sorted(
            str(x) for x in split_guard.get("v3_empirical_direction_train_eligible_task_ids", [])
        ),
        "prior_seen_by_v1_v2_task_ids": sorted(str(x) for x in split_guard.get("prior_seen_by_v1_v2_task_ids", [])),
        "split_guard_verdict": split_guard.get("verdict"),
    }


def pair_counts_for_group(rows: Sequence[dict[str, Any]], label_source: str = "deterministic") -> dict[str, int]:
    candidate = 0
    non_tie = 0
    tie = 0
    ordered = list(rows)
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            if label_source == "sampled_expected" and (sampled_reward(ordered[i]) is None or sampled_reward(ordered[j]) is None):
                continue
            candidate += 1
            if row_reward(ordered[i], label_source) == row_reward(ordered[j], label_source):
                tie += 1
            else:
                non_tie += 1
    return {"candidate_pairs": candidate, "non_tie_pairs": non_tie, "tie_pairs": tie}


def support_for_task_ids(rows: Sequence[dict[str, Any]], task_ids: Iterable[str], label_source: str = "deterministic") -> dict[str, Any]:
    task_set = {str(x) for x in task_ids}
    selected = [row for row in rows if str(row.get("task_id")) in task_set and stable_v2_row(row)]
    groups = {gid: vals for gid, vals in group_rows(selected).items() if len(vals) >= 2}
    candidate_pairs = 0
    non_tie_pairs = 0
    tie_pairs = 0
    behavior_groups = 0
    reward_groups = 0
    task_ids_with_behavior: set[str] = set()
    task_ids_with_non_tie: set[str] = set()
    for vals in groups.values():
        tid = str(vals[0].get("task_id"))
        counts = pair_counts_for_group(vals, label_source)
        candidate_pairs += counts["candidate_pairs"]
        non_tie_pairs += counts["non_tie_pairs"]
        tie_pairs += counts["tie_pairs"]
        if counts["non_tie_pairs"] > 0:
            task_ids_with_non_tie.add(tid)
        if group_is_behaviorally_diverse_v2(vals, label_source):
            behavior_groups += 1
            task_ids_with_behavior.add(tid)
        if group_is_reward_diverse_v2(vals, label_source):
            reward_groups += 1
    support_tasks = sorted(task_ids_with_non_tie)
    return {
        "task_ids_all": sorted(task_set & {str(row.get("task_id")) for row in selected}),
        "task_ids_with_behavioral_diversity": sorted(task_ids_with_behavior),
        "task_ids_with_non_tie_pairs": support_tasks,
        "support_task_count": len(support_tasks),
        "row_count": len(selected),
        "group_count": len(groups),
        "behaviorally_diverse_groups": behavior_groups,
        "reward_diverse_groups": reward_groups,
        "candidate_pairs": candidate_pairs,
        "non_tie_pairs": non_tie_pairs,
        "tie_pairs": tie_pairs,
        "tie_rate": tie_pairs / max(candidate_pairs, 1),
        "readiness_support": (
            len(support_tasks) >= READINESS_MINIMUMS["task_ids"]
            and behavior_groups >= READINESS_MINIMUMS["behaviorally_diverse_groups"]
            and non_tie_pairs >= READINESS_MINIMUMS["non_tie_pairs"]
        ),
        "weak_support": (
            len(support_tasks) >= WEAK_MINIMUMS["task_ids"]
            and behavior_groups >= WEAK_MINIMUMS["behaviorally_diverse_groups"]
            and non_tie_pairs >= WEAK_MINIMUMS["non_tie_pairs"]
        ),
    }


def task_distribution(rows: Sequence[dict[str, Any]], label_source: str = "deterministic") -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row.get("task_id"))].append(row)
    out: list[dict[str, Any]] = []
    for task_id, vals in sorted(by_task.items()):
        groups = {gid: g for gid, g in group_rows(vals).items() if len(g) >= 2}
        candidate_pairs = non_tie_pairs = tie_pairs = 0
        behavior_groups = reward_groups = 0
        for g in groups.values():
            counts = pair_counts_for_group(g, label_source)
            candidate_pairs += counts["candidate_pairs"]
            non_tie_pairs += counts["non_tie_pairs"]
            tie_pairs += counts["tie_pairs"]
            behavior_groups += int(group_is_behaviorally_diverse_v2(g, label_source))
            reward_groups += int(group_is_reward_diverse_v2(g, label_source))
        out.append(
            {
                "task_id": task_id,
                "domain": Counter(str(row.get("domain")) for row in vals).most_common(1)[0][0],
                "rows": len(vals),
                "groups": len(groups),
                "behaviorally_diverse_groups": behavior_groups,
                "reward_diverse_groups": reward_groups,
                "candidate_pairs": candidate_pairs,
                "non_tie_pairs": non_tie_pairs,
                "tie_pairs": tie_pairs,
                "tie_rate": tie_pairs / max(candidate_pairs, 1),
                "versions": ",".join(sorted({version_for_row(row) for row in vals})),
                "branch_points": ",".join(sorted({str(row.get("branch_point")) for row in vals})),
                "delta_families": ",".join(sorted({str(row.get("primary_delta_family") or row.get("delta_family") or "unknown") for row in vals})),
                "alpha_buckets": ",".join(sorted({alpha_bucket(row) for row in vals})),
                "task_classes": ",".join(sorted({str(row.get("task_screening_class") or "unknown") for row in vals})),
                "split_guard_roles": ",".join(sorted({str(row.get("split_guard_role") or "prior_or_unknown") for row in vals})),
            }
        )
    return sorted(out, key=lambda row: (-int(row["behaviorally_diverse_groups"]), -int(row["non_tie_pairs"]), row["task_id"]))


def version_summary(rows: Sequence[dict[str, Any]], label_source: str = "deterministic") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for version in ("v1", "v2", "v3", "combined"):
        vals = list(rows) if version == "combined" else [row for row in rows if version_for_row(row) == version]
        groups = {gid: g for gid, g in group_rows(vals).items() if len(g) >= 2}
        candidate_pairs = non_tie_pairs = tie_pairs = 0
        for g in groups.values():
            counts = pair_counts_for_group(g, label_source)
            candidate_pairs += counts["candidate_pairs"]
            non_tie_pairs += counts["non_tie_pairs"]
            tie_pairs += counts["tie_pairs"]
        out[version] = {
            "rows": len(vals),
            "stable_primary_rows": sum(1 for row in vals if stable_v2_row(row)),
            "groups": len(groups),
            "behaviorally_diverse_groups": sum(1 for g in groups.values() if group_is_behaviorally_diverse_v2(g, label_source)),
            "reward_diverse_groups": sum(1 for g in groups.values() if group_is_reward_diverse_v2(g, label_source)),
            "candidate_pairs": candidate_pairs,
            "non_tie_pairs": non_tie_pairs,
            "tie_pairs": tie_pairs,
            "tie_rate": tie_pairs / max(candidate_pairs, 1),
        }
    return out


def choose_val_tasks(train_ids: Sequence[str], stats_by_task: dict[str, dict[str, Any]]) -> set[str]:
    ids = [tid for tid in train_ids if int(stats_by_task.get(tid, {}).get("non_tie_pairs", 0)) > 0]
    if len(ids) <= 2:
        return set(ids[:1])
    target_pairs = max(1, round(sum(int(stats_by_task[tid]["non_tie_pairs"]) for tid in ids) * 0.20))
    selected: set[str] = set()
    running = 0
    for tid in sorted(ids, key=lambda t: (int(stats_by_task[t]["non_tie_pairs"]), t)):
        selected.add(tid)
        running += int(stats_by_task[tid]["non_tie_pairs"])
        if running >= target_pairs and len(selected) >= 1:
            break
    if len(selected) >= len(ids):
        selected = {ids[0]}
    return selected


def make_balanced_folds(task_ids: Sequence[str], stats_by_task: dict[str, dict[str, Any]], k: int) -> list[list[str]]:
    folds: list[list[str]] = [[] for _ in range(k)]
    fold_weight = [0 for _ in range(k)]
    ordered = sorted(task_ids, key=lambda tid: (-int(stats_by_task.get(tid, {}).get("non_tie_pairs", 0)), tid))
    for tid in ordered:
        idx = min(range(k), key=lambda i: (fold_weight[i], len(folds[i]), i))
        folds[idx].append(tid)
        fold_weight[idx] += int(stats_by_task.get(tid, {}).get("non_tie_pairs", 0))
    return [sorted(fold) for fold in folds if fold]


def split_record(
    *,
    mode_name: str,
    fold_id: str,
    mode_type: str,
    heldout_task_ids: Sequence[str],
    train_task_ids: Sequence[str],
    val_task_ids: Sequence[str],
    rows: Sequence[dict[str, Any]],
    contamination: dict[str, Any],
    notes: str = "",
) -> dict[str, Any]:
    heldout_support = support_for_task_ids(rows, heldout_task_ids)
    train_support = support_for_task_ids(rows, train_task_ids)
    val_support = support_for_task_ids(rows, val_task_ids)
    can_support_selector_readiness = bool(heldout_support["readiness_support"] and contamination.get("v3_clean", False))
    return {
        "mode_name": mode_name,
        "fold_id": fold_id,
        "mode_type": mode_type,
        "heldout_task_ids": sorted(str(x) for x in heldout_task_ids),
        "train_task_ids": sorted(str(x) for x in train_task_ids),
        "val_task_ids": sorted(str(x) for x in val_task_ids),
        "heldout_support": heldout_support,
        "train_support": train_support,
        "val_support": val_support,
        "contamination_flags": contamination,
        "can_support_selector_readiness": can_support_selector_readiness,
        "weak_support": bool(heldout_support["weak_support"]),
        "notes": notes,
    }


def define_eval_modes_from_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    dist = task_distribution(rows)
    stats_by_task = {str(row["task_id"]): row for row in dist}
    all_tasks = sorted(stats_by_task)
    signal_tasks = sorted(tid for tid, row in stats_by_task.items() if int(row["non_tie_pairs"]) > 0)
    behavior_tasks = sorted(tid for tid, row in stats_by_task.items() if int(row["behaviorally_diverse_groups"]) > 0)
    seen = prior_seen_sets()
    v1v2_train_val = set(seen["v1_v2_train_val"])
    v3_train_val = set(seen["v3_train_val"])
    v3_heldout = set(seen["v3_heldout_candidate_task_ids"])
    strict_clean = set(seen["clean_cross_version_heldout_task_ids"]) - v1v2_train_val - v3_train_val

    records: list[dict[str, Any]] = []
    strict_heldout = sorted(strict_clean & set(all_tasks))
    strict_train = sorted(set(all_tasks) - set(strict_heldout))
    strict_val = sorted(choose_val_tasks(strict_train, stats_by_task))
    strict_train = sorted(set(strict_train) - set(strict_val))
    records.append(
        split_record(
            mode_name="strict_cross_version_clean",
            fold_id="main",
            mode_type="fixed",
            heldout_task_ids=strict_heldout,
            train_task_ids=strict_train,
            val_task_ids=strict_val,
            rows=rows,
            contamination={
                "old_frozen_bg_clean": True,
                "v1_clean": not bool(set(strict_heldout) & v1v2_train_val),
                "v2_clean": not bool(set(strict_heldout) & v1v2_train_val),
                "v3_clean": not bool(set(strict_heldout) & v3_train_val) and set(strict_heldout).issubset(v3_heldout),
                "v3_empirical_direction_clean": set(strict_heldout).issubset(v3_heldout),
            },
            notes="Cleanest cross-version mode; support uses only heldout tasks with non-tie reward pairs.",
        )
    )

    v3_clean_heldout = sorted(v3_heldout & set(all_tasks))
    v3_clean_train = sorted(set(all_tasks) - set(v3_clean_heldout))
    v3_clean_val = sorted(choose_val_tasks(v3_clean_train, stats_by_task))
    v3_clean_train = sorted(set(v3_clean_train) - set(v3_clean_val))
    records.append(
        split_record(
            mode_name="v3_clean",
            fold_id="main",
            mode_type="fixed",
            heldout_task_ids=v3_clean_heldout,
            train_task_ids=v3_clean_train,
            val_task_ids=v3_clean_val,
            rows=rows,
            contamination={
                "old_frozen_bg_clean": True,
                "v1_clean": not bool(set(v3_clean_heldout) & v1v2_train_val),
                "v2_clean": not bool(set(v3_clean_heldout) & v1v2_train_val),
                "v3_clean": not bool(set(v3_clean_heldout) & v3_train_val) and set(v3_clean_heldout).issubset(v3_heldout),
                "v3_empirical_direction_clean": set(v3_clean_heldout).issubset(v3_heldout),
            },
            notes="Heldout tasks are excluded from v3 tap training and empirical direction construction; v1/v2 contamination is flagged separately.",
        )
    )

    records.append(
        split_record(
            mode_name="old_frozen_tap_clean",
            fold_id="all_signal_tasks",
            mode_type="eval_only",
            heldout_task_ids=signal_tasks,
            train_task_ids=[],
            val_task_ids=[],
            rows=rows,
            contamination={
                "old_frozen_bg_clean": True,
                "v1_clean": False,
                "v2_clean": False,
                "v3_clean": False,
                "v3_empirical_direction_clean": False,
            },
            notes="Old frozen tap is fixed relative to hidden-origin tap training; learned hidden-origin baselines are diagnostic here.",
        )
    )

    k = 5 if len(signal_tasks) >= 5 else max(1, len(signal_tasks))
    folds = make_balanced_folds(signal_tasks, stats_by_task, k)
    for idx, heldout in enumerate(folds):
        train_ids = sorted(set(signal_tasks) - set(heldout))
        val_ids = sorted(choose_val_tasks(train_ids, stats_by_task))
        train_ids = sorted(set(train_ids) - set(val_ids))
        records.append(
            split_record(
                mode_name="grouped_kfold_v3",
                fold_id=f"fold_{idx}",
                mode_type="grouped_kfold",
                heldout_task_ids=heldout,
                train_task_ids=train_ids,
                val_task_ids=val_ids,
                rows=rows,
                contamination={
                    "old_frozen_bg_clean": True,
                    "v1_clean": not bool(set(heldout) & v1v2_train_val),
                    "v2_clean": not bool(set(heldout) & v1v2_train_val),
                    "v3_clean": not bool(set(heldout) & v3_train_val),
                    "v3_empirical_direction_clean": set(heldout).issubset(v3_heldout),
                },
                notes="Grouped CV is task-disjoint for retrained salvage heads; empirical-direction cleanliness is only true for original v3 heldout tasks.",
            )
        )

    for task_id in signal_tasks:
        train_ids = sorted(set(signal_tasks) - {task_id})
        val_ids = sorted(choose_val_tasks(train_ids, stats_by_task))
        train_ids = sorted(set(train_ids) - set(val_ids))
        records.append(
            split_record(
                mode_name="leave_one_task_out_v3",
                fold_id=task_id.replace("/", "__"),
                mode_type="leave_one_task_out",
                heldout_task_ids=[task_id],
                train_task_ids=train_ids,
                val_task_ids=val_ids,
                rows=rows,
                contamination={
                    "old_frozen_bg_clean": True,
                    "v1_clean": task_id not in v1v2_train_val,
                    "v2_clean": task_id not in v1v2_train_val,
                    "v3_clean": task_id not in v3_train_val,
                    "v3_empirical_direction_clean": task_id in v3_heldout,
                },
                notes="LOTO estimates task-level variance; single-task folds cannot independently support readiness.",
            )
        )
    return {
        "records": records,
        "task_distribution": dist,
        "all_task_ids": all_tasks,
        "behaviorally_diverse_task_ids": behavior_tasks,
        "reward_signal_task_ids": signal_tasks,
        "prior_seen_sets": seen,
        "fold_count": {"grouped_kfold_v3": len(folds), "leave_one_task_out_v3": len(signal_tasks)},
    }


def feature_dict(pref: dict[str, Any], rej: dict[str, Any], configs: Sequence[str] = CONFIGS) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    for config in configs:
        left = config_vector_from_row(pref, config)
        right = config_vector_from_row(rej, config)
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            if tuple(left.shape) == (config_dim(config),) and tuple(right.shape) == (config_dim(config),):
                out[config] = {"preferred": left.to(torch.float32), "rejected": right.to(torch.float32)}
    return out


def split_for_task(task_id: str, record: dict[str, Any]) -> str:
    tid = str(task_id)
    if tid in set(record.get("heldout_task_ids") or []):
        return "test"
    if tid in set(record.get("val_task_ids") or []):
        return "val"
    if tid in set(record.get("train_task_ids") or []):
        return "train"
    return "unused"


def build_pairs_for_record(
    rows: Sequence[dict[str, Any]],
    record: dict[str, Any],
    *,
    variant: str,
    label_source: str,
    row_filter: Callable[[dict[str, Any]], bool],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid = [row for row in rows if row_filter(row) and stable_v2_row(row)]
    pairs: list[dict[str, Any]] = []
    groups = {gid: vals for gid, vals in group_rows(valid).items() if len(vals) >= 2}
    candidate_pairs = tie_pairs = missing_features = 0
    for gid, vals in sorted(groups.items()):
        ordered = sorted(vals, key=lambda row: int(row.get("branch_id", -1)))
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a, b = ordered[i], ordered[j]
                if label_source == "sampled_expected" and (sampled_reward(a) is None or sampled_reward(b) is None):
                    continue
                candidate_pairs += 1
                ra = row_reward(a, label_source)
                rb = row_reward(b, label_source)
                if ra == rb:
                    tie_pairs += 1
                    continue
                pref, rej = (a, b) if ra > rb else (b, a)
                feats = feature_dict(pref, rej)
                if not feats:
                    missing_features += 1
                    continue
                split = split_for_task(str(pref.get("task_id")), record)
                pairs.append(
                    {
                        "pair_id": f"{record['mode_name']}::{record['fold_id']}::{variant}::{gid}::{int(a['branch_id'])}-{int(b['branch_id'])}",
                        "mode_name": record["mode_name"],
                        "fold_id": record["fold_id"],
                        "variant": variant,
                        "label_source": label_source,
                        "task_id": pref["task_id"],
                        "branch_group_id": pref["branch_group_id"],
                        "domain": pref.get("domain"),
                        "preferred_branch_id": int(pref["branch_id"]),
                        "rejected_branch_id": int(rej["branch_id"]),
                        "reward_preferred": row_reward(pref, label_source),
                        "reward_rejected": row_reward(rej, label_source),
                        "reward_gap": row_reward(pref, label_source) - row_reward(rej, label_source),
                        "deterministic_reward_preferred": deterministic_reward(pref),
                        "deterministic_reward_rejected": deterministic_reward(rej),
                        "branch_point_preferred": pref.get("branch_point"),
                        "branch_point_rejected": rej.get("branch_point"),
                        "alpha_preferred": float(pref.get("alpha", 0.0)),
                        "alpha_rejected": float(rej.get("alpha", 0.0)),
                        "alpha_bucket_preferred": alpha_bucket(pref),
                        "alpha_bucket_rejected": alpha_bucket(rej),
                        "delta_family_preferred": pref.get("primary_delta_family") or pref.get("delta_family"),
                        "delta_family_rejected": rej.get("primary_delta_family") or rej.get("delta_family"),
                        "task_screening_class": pref.get("task_screening_class"),
                        "old_frozen_tap_score_preferred": float(pref.get("tap_margin_sum", pref.get("old_frozen_tap_score", 0.0))),
                        "old_frozen_tap_score_rejected": float(rej.get("tap_margin_sum", rej.get("old_frozen_tap_score", 0.0))),
                        "v1_tap_score_preferred": pref.get("v1_tap_score"),
                        "v1_tap_score_rejected": rej.get("v1_tap_score"),
                        "v2_tap_score_preferred": pref.get("v2_tap_score"),
                        "v2_tap_score_rejected": rej.get("v2_tap_score"),
                        "available_configs": sorted(feats.keys()),
                        "features": feats,
                        "split": split,
                    }
                )
    split_counts = Counter(pair["split"] for pair in pairs)
    meta = {
        "variant": variant,
        "label_source": label_source,
        "valid_branch_count": len(valid),
        "valid_group_count": len(groups),
        "candidate_pairs": candidate_pairs,
        "tie_pairs_omitted": tie_pairs,
        "non_tie_pairs": len(pairs),
        "tie_rate": tie_pairs / max(candidate_pairs, 1),
        "missing_feature_pairs": missing_features,
        "pairs_by_split": dict(split_counts),
    }
    return pairs, meta


def compact_pair(pair: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in pair.items() if k != "features"}
    out["feature_dims"] = {cfg: int(feats["preferred"].numel()) for cfg, feats in pair.get("features", {}).items()}
    return out


def rows_for_tasks(rows: Sequence[dict[str, Any]], task_ids: Iterable[str], row_filter: Callable[[dict[str, Any]], bool]) -> list[dict[str, Any]]:
    task_set = {str(x) for x in task_ids}
    return [row for row in rows if str(row.get("task_id")) in task_set and row_filter(row) and stable_v2_row(row)]


def groups_for_tasks(rows: Sequence[dict[str, Any]], task_ids: Iterable[str], row_filter: Callable[[dict[str, Any]], bool]) -> list[list[dict[str, Any]]]:
    return [vals for vals in group_rows(rows_for_tasks(rows, task_ids, row_filter)).values() if len(vals) >= 2]


def selector_rows_for_group(
    group: Sequence[dict[str, Any]],
    *,
    policy: str,
    metric: dict[str, Any],
    mode_record: dict[str, Any],
    subset: str,
    config: str = "baseline",
    architecture: str = "baseline",
    head_id: str | None = None,
    clean: bool | None = None,
) -> dict[str, Any]:
    first = group[0]
    return {
        "mode_name": mode_record["mode_name"],
        "fold_id": mode_record["fold_id"],
        "subset": subset,
        "branch_group_id": first.get("branch_group_id"),
        "task_id": first.get("task_id"),
        "domain": first.get("domain"),
        "branch_point": first.get("branch_point"),
        "alpha": first.get("alpha"),
        "alpha_bucket": first.get("alpha_bucket"),
        "delta_family": first.get("primary_delta_family") or first.get("delta_family"),
        "task_screening_class": first.get("task_screening_class"),
        "behaviorally_diverse": group_is_behaviorally_diverse_v2(group),
        "reward_diverse": group_is_reward_diverse_v2(group),
        "policy": policy,
        "config": config,
        "architecture": architecture,
        "head_id": head_id,
        "selector_clean_for_mode": clean,
        "old_frozen_bg_clean": mode_record.get("contamination_flags", {}).get("old_frozen_bg_clean"),
        "v1_clean": mode_record.get("contamination_flags", {}).get("v1_clean"),
        "v2_clean": mode_record.get("contamination_flags", {}).get("v2_clean"),
        "v3_clean": mode_record.get("contamination_flags", {}).get("v3_clean"),
        **metric,
    }


def baseline_selector_rows(groups: Sequence[list[dict[str, Any]]], mode_record: dict[str, Any], subset: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group in groups:
        ordered = sorted(group, key=lambda row: int(row.get("branch_id", -1)))
        rewards = [float(row.get("deterministic_reward", row.get("reward", 0.0))) for row in ordered]
        correct = [1.0 if row.get("deterministic_correct", row.get("correct")) else 0.0 for row in ordered]
        oracle_reward = max(rewards)
        oracle_ids = {int(row["branch_id"]) for row in ordered if float(row.get("deterministic_reward", row.get("reward", 0.0))) == oracle_reward}
        policies = {
            "clean_branch_baseline": selected_metric(ordered, [0]),
            "old_frozen_bg_highest_tap_score": selected_metric(
                ordered,
                [max(range(len(ordered)), key=lambda i: (float(ordered[i].get("tap_margin_sum", ordered[i].get("old_frozen_tap_score", 0.0))), -int(ordered[i]["branch_id"])))],
            ),
            "old_frozen_bg_pairwise_tournament": selected_metric(
                ordered,
                [max(range(len(ordered)), key=lambda i: (float(ordered[i].get("tap_margin_sum", ordered[i].get("old_frozen_tap_score", 0.0))), -int(ordered[i]["branch_id"])))],
            ),
            "simple_top2_branch_id_order": selected_metric(ordered, list(range(min(2, len(ordered))))),
        }
        policies["random_top1"] = {
            "success": float(mean(correct)),
            "reward": float(mean(rewards)),
            "oracle_coverage": len(oracle_ids) / max(len(ordered), 1),
            "oracle_gap": oracle_reward - float(mean(rewards)),
            "selection_regret": oracle_reward - float(mean(rewards)),
            "pruned_oracle_branch_rate": 1.0 - len(oracle_ids) / max(len(ordered), 1),
            "kept_good_branch_rate": len(oracle_ids) / max(len(ordered), 1),
            "selected_branch_ids": [],
            "oracle_branch_ids": sorted(oracle_ids),
            "survivors": 1,
        }
        policies["random_top2"] = {
            "success": top2_random_success(ordered),
            "reward": top2_random_reward(ordered),
            "oracle_coverage": min(1.0, 2.0 * len(oracle_ids) / max(len(ordered), 1)),
            "oracle_gap": oracle_reward - top2_random_reward(ordered),
            "selection_regret": oracle_reward - top2_random_reward(ordered),
            "pruned_oracle_branch_rate": 1.0 - min(1.0, 2.0 * len(oracle_ids) / max(len(ordered), 1)),
            "kept_good_branch_rate": min(1.0, 2.0 * len(oracle_ids) / max(len(ordered), 1)),
            "selected_branch_ids": [],
            "oracle_branch_ids": sorted(oracle_ids),
            "survivors": 2,
        }
        for policy, metric in policies.items():
            clean = True if policy.startswith(("random", "clean", "simple")) else bool(mode_record.get("contamination_flags", {}).get("old_frozen_bg_clean"))
            out.append(selector_rows_for_group(ordered, policy=policy, metric=metric, mode_record=mode_record, subset=subset, clean=clean))
    return out


def tap_selector_rows(
    groups: Sequence[list[dict[str, Any]]],
    mode_record: dict[str, Any],
    subset: str,
    head_row: dict[str, Any] | None,
    device: torch.device,
    label: str,
    clean: bool,
) -> list[dict[str, Any]]:
    if head_row is None:
        return []
    head = build_head(head_row, device)
    config = str(head_row["config"])
    out: list[dict[str, Any]] = []
    for group in groups:
        ordered = sorted(group, key=lambda row: int(row.get("branch_id", -1)))
        vectors = [config_vector_from_row(row, config) for row in ordered]
        if not vectors or not all(isinstance(vec, torch.Tensor) and int(vec.numel()) == config_dim(config) for vec in vectors):
            continue
        mat = score_matrix(head, [vec for vec in vectors if isinstance(vec, torch.Tensor)], device)
        ranking = list(ranking_from_matrix(mat)["ranking"])
        if not ranking:
            continue
        for policy, metric in (
            (f"{label}_highest_score", selected_metric(ordered, [ranking[0]])),
            (f"{label}_pairwise_tournament", selected_metric(ordered, [ranking[0]])),
            (f"{label}_top2", selected_metric(ordered, ranking[:2])),
        ):
            out.append(
                selector_rows_for_group(
                    ordered,
                    policy=policy,
                    metric=metric,
                    mode_record=mode_record,
                    subset=subset,
                    config=config,
                    architecture=str(head_row["architecture"]),
                    head_id=compact_head_id(head_row),
                    clean=clean,
                )
            )
    head.to("cpu")
    return out


def rank_from_head(group: Sequence[dict[str, Any]], head_row: dict[str, Any] | None, device: torch.device) -> list[int]:
    if head_row is None:
        return []
    head = build_head(head_row, device)
    config = str(head_row["config"])
    ordered = sorted(group, key=lambda row: int(row.get("branch_id", -1)))
    vectors = [config_vector_from_row(row, config) for row in ordered]
    if not all(isinstance(vec, torch.Tensor) and int(vec.numel()) == config_dim(config) for vec in vectors):
        head.to("cpu")
        return []
    mat = score_matrix(head, [vec for vec in vectors if isinstance(vec, torch.Tensor)], device)
    ranking = list(ranking_from_matrix(mat)["ranking"])
    head.to("cpu")
    return ranking


def ensemble_selector_rows(
    groups: Sequence[list[dict[str, Any]]],
    mode_record: dict[str, Any],
    subset: str,
    heads: Sequence[tuple[str, dict[str, Any] | None, bool]],
    device: torch.device,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group in groups:
        ordered = sorted(group, key=lambda row: int(row.get("branch_id", -1)))
        n = len(ordered)
        scores = [0.0 for _ in ordered]
        contributors = 0
        old_rank = sorted(range(n), key=lambda i: (float(ordered[i].get("tap_margin_sum", ordered[i].get("old_frozen_tap_score", 0.0))), -int(ordered[i]["branch_id"])), reverse=True)
        for rank, idx in enumerate(old_rank):
            scores[idx] += n - rank
        contributors += 1
        clean = bool(mode_record.get("contamination_flags", {}).get("old_frozen_bg_clean"))
        for _label, head_row, head_clean in heads:
            ranking = rank_from_head(ordered, head_row, device)
            if not ranking:
                continue
            for rank, idx in enumerate(ranking):
                scores[idx] += n - rank
            contributors += 1
            clean = clean and bool(head_clean)
        if contributors <= 1:
            continue
        ranking = sorted(range(n), key=lambda idx: (-scores[idx], idx))
        for policy, metric in (
            ("diagnostic_rank_aggregation_ensemble", selected_metric(ordered, [ranking[0]])),
            ("diagnostic_rank_aggregation_ensemble_top2", selected_metric(ordered, ranking[:2])),
        ):
            out.append(
                selector_rows_for_group(
                    ordered,
                    policy=policy,
                    metric=metric,
                    mode_record=mode_record,
                    subset=subset,
                    config="ensemble",
                    architecture="rank_aggregation",
                    head_id=f"contributors={contributors}",
                    clean=clean,
                )
            )
    return out


def aggregate_selector_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["mode_name"]), str(row["fold_id"]), str(row["subset"]), str(row["policy"]), str(row.get("config", "")))].append(row)
    out: dict[str, Any] = {}
    for (mode, fold, subset, policy, config), vals in sorted(grouped.items()):
        task_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for val in vals:
            task_groups[str(val["task_id"])].append(val)
        task_success = [mean(float(v["success"]) for v in tv) for tv in task_groups.values()]
        task_reward = [mean(float(v["reward"]) for v in tv) for tv in task_groups.values()]
        key = f"{mode}::{fold}::{subset}::{policy}::{config}"
        out[key] = {
            "mode_name": mode,
            "fold_id": fold,
            "subset": subset,
            "policy": policy,
            "config": config,
            "architecture": vals[0].get("architecture"),
            "groups": len(vals),
            "task_count": len(task_groups),
            "selector_clean_for_mode": all(bool(v.get("selector_clean_for_mode")) for v in vals),
            "top1_success": float(mean(float(v["success"]) for v in vals)),
            "task_macro_top1_success": float(mean(task_success)) if task_success else float("nan"),
            "reward_mean": float(mean(float(v["reward"]) for v in vals)),
            "task_macro_reward_mean": float(mean(task_reward)) if task_reward else float("nan"),
            "top2_oracle_coverage": float(mean(float(v["oracle_coverage"]) for v in vals)),
            "oracle_gap": float(mean(float(v["oracle_gap"]) for v in vals)),
            "selection_regret": float(mean(float(v["selection_regret"]) for v in vals)),
            "kept_good_branch_rate": float(mean(float(v["kept_good_branch_rate"]) for v in vals)),
            "pruned_oracle_branch_rate": float(mean(float(v["pruned_oracle_branch_rate"]) for v in vals)),
        }
    return out


def best_metric(metrics: dict[str, Any], *, mode: str | None = None, subset: str = "behaviorally_diverse", clean_only: bool = False) -> dict[str, Any] | None:
    vals = [
        row
        for row in metrics.values()
        if row.get("subset") == subset
        and (mode is None or row.get("mode_name") == mode)
        and (not clean_only or row.get("selector_clean_for_mode"))
    ]
    if not vals:
        return None
    return max(vals, key=lambda row: (float(row["task_macro_top1_success"]), float(row["top1_success"]), float(row["reward_mean"])))


def compact_head_for_json(row: dict[str, Any]) -> dict[str, Any]:
    return compact_head(row) | {
        "mode_name": row.get("mode_name"),
        "fold_id": row.get("fold_id"),
        "head_id": compact_head_id(row),
    }


def finite_mean(vals: Sequence[float]) -> float:
    filtered = [float(v) for v in vals if math.isfinite(float(v))]
    return float(mean(filtered)) if filtered else float("nan")


def finite_pstdev(vals: Sequence[float]) -> float:
    filtered = [float(v) for v in vals if math.isfinite(float(v))]
    return float(pstdev(filtered)) if len(filtered) > 1 else 0.0 if filtered else float("nan")

