"""Shared helpers for hidden-origin quota v4 probes.

The v4 probes are split-quota manual diagnostics.  They do not train Ouro,
change tokenizer/checkpoint/model files, update old tap registries, run wrapper
or local-agent code, execute ARC environment actions, or use tap scores as
labels.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Iterable, Sequence

import torch

from bg_hidden_origin_diversity_v3_common import (
    CONFIGS,
    HIDDEN_DIM,
    MAX_NEW_TOKENS,
    PRIMARY_ALPHA_CAP,
    PRIMARY_DOMAINS,
    PROBE_ROOT,
    PROJECT_ROOT,
    SEED,
    V1_ROOT,
    V2_ROOT,
    V3_ROOT,
    alpha_bucket,
    branch_group_metrics,
    compact_branch_row_v3,
    compact_head_id,
    config_dim,
    config_layer,
    config_vector_from_row,
    cosine,
    deterministic_correct,
    deterministic_reward,
    diagnostic_alpha_v3_row,
    evaluate_mcq,
    finite_float,
    generate_with_hook_v2,
    group_is_behaviorally_diverse_v2,
    group_is_reward_diverse_v2,
    group_rows,
    load_all_v3_branch_rows,
    load_best_v1_head,
    load_best_v2_head,
    load_best_v3_head,
    load_head_rows,
    load_json,
    load_more_candidate_tasks,
    md_table,
    normalize_branch_point,
    normalize_task,
    primary_safe_v3_row,
    rate,
    rel,
    rms_normalize,
    row_reward,
    sampled_reward,
    score_group_with_head,
    selected_v3_tasks,
    stable_v2_row,
    tensor_stats,
    write_csv,
    write_json,
    write_md,
)
from bg_hidden_origin_split_salvage_common import SALVAGE_HEADS_PT, SALVAGE_ROOT, prior_seen_sets
from bg_hidden_origin_tap_common import (
    HEAD_CLASSES,
    branch_key,
    capture_prefix_features,
    direction_from_state_dict,
    pairwise_accuracy_from_pairs,
    ranking_from_matrix,
    score_matrix,
    top2_random_reward,
    top2_random_success,
)
from evaluate_bg_hidden_origin_taps import aggregate, best_metric, build_head, selected_metric
from expand_bg_hidden_origin_branch_dataset import score_old_taps
from train_bg_hidden_origin_taps import ARCHITECTURES, compact_head, flip_diagnostics, pairs_for_config, train_one


V4_ROOT = PROBE_ROOT / "bg_hidden_origin_quota_v4_2026-05-18"
QUOTA_PLAN_JSON = V4_ROOT / "quota_plan.json"
QUOTA_PLAN_MD = V4_ROOT / "quota_plan.md"
QUOTA_PLAN_CSV = V4_ROOT / "quota_plan_tasks.csv"
DIRECTION_BANK_V4_PT = V4_ROOT / "direction_bank_v4.pt"
DIRECTION_BANK_V4_JSON = V4_ROOT / "direction_bank_v4.json"
DIRECTION_BANK_V4_MD = V4_ROOT / "direction_bank_v4.md"
CONTROLLER_JSON = V4_ROOT / "hs_inspired_quota_controller.json"
CONTROLLER_MD = V4_ROOT / "hs_inspired_quota_controller.md"
RECIPE_ENGRAMS_JSONL = V4_ROOT / "recipe_engrams_v4.jsonl"
BRANCHES_PT = V4_ROOT / "quota_hidden_origin_branches.pt"
BRANCHES_PARTIAL_PT = V4_ROOT / "quota_hidden_origin_branches.partial.pt"
BRANCHES_JSON = V4_ROOT / "quota_hidden_origin_branches.json"
BRANCHES_CSV = V4_ROOT / "quota_hidden_origin_branches.csv"
PROGRESS_JSONL = V4_ROOT / "quota_generation_progress.jsonl"
GEN_STATE_JSON = V4_ROOT / "quota_generation_state.json"
GEN_REPORT_MD = V4_ROOT / "quota_generation_report.md"
GEN_REPORT_JSON = V4_ROOT / "quota_generation_report.json"
DATASET_V4_PT = V4_ROOT / "hidden_origin_quota_dataset_v4.pt"
DATASET_V4_JSON = V4_ROOT / "hidden_origin_quota_dataset_v4.json"
DATASET_V4_MD = V4_ROOT / "hidden_origin_quota_dataset_v4.md"
HEADS_V4_PT = V4_ROOT / "hidden_origin_tap_heads_v4.pt"

PRIMARY_MINIMUMS = {
    "train": {"task_ids": 24, "behaviorally_diverse_groups": 60, "non_tie_pairs": 250},
    "val": {"task_ids": 6, "behaviorally_diverse_groups": 15, "non_tie_pairs": 60},
    "heldout": {"task_ids": 8, "behaviorally_diverse_groups": 20, "non_tie_pairs": 120},
}
PREFERRED_MINIMUMS = {
    "train": {"task_ids": 40, "behaviorally_diverse_groups": 100, "non_tie_pairs": 400},
    "val": {"task_ids": 8, "behaviorally_diverse_groups": 25, "non_tie_pairs": 100},
    "heldout": {"task_ids": 10, "behaviorally_diverse_groups": 30, "non_tie_pairs": 180},
}
SPLIT_ROW_CAPS = {"train": 2000, "val": 800, "heldout": 1200}
TOTAL_ROW_CAP = 4000
MAX_SELECTED_TASKS = 240
PRIMARY_BRANCH_POINTS = {"L24", "L36"}
DIAGNOSTIC_BRANCH_POINTS = {"L47"}
PRIMARY_TASK_CLASSES = {
    "perturbation_sensitive",
    "wrong_parseable",
    "baseline_wrong_parseable",
    "parse_fragile",
    "baseline_parse_fragile",
    "low_confidence_correct",
    "baseline_correct_low_confidence",
    "evaluator_disagreement",
}
TASK_CLASS_ALIASES = {
    "baseline_wrong_parseable": "wrong_parseable",
    "baseline_parse_fragile": "parse_fragile",
    "baseline_correct_low_confidence": "low_confidence_correct",
    "baseline_correct_confident": "baseline_correct_confident",
}
DELTA_FAMILIES = (
    "random_orthogonal",
    "paired_plus_minus",
    "old_tap_aligned",
    "v1_tap_aligned",
    "v2_tap_aligned",
    "v3_tap_aligned",
    "salvage_tap_aligned",
    "hidden_origin_empirical_train_only",
    "hidden_origin_whitened_train_only",
    "adapter_proxy",
    "sequence_adapter_proxy",
    "empirical_plus_noise",
)
ALPHA_VALUE = {"alpha_0_005": 0.005, "alpha_0_01": 0.010, "alpha_0_02": 0.020}
V4_COMMAND_SCRIPTS = [
    "bg_hidden_origin_quota_v4_audit_plan.py",
    "build_bg_hidden_origin_direction_bank_v4.py",
    "bg_hidden_origin_hs_inspired_quota_controller_v4.py",
    "generate_bg_hidden_origin_quota_branches_v4.py",
    "analyze_bg_hidden_origin_quota_v4_drivers.py",
    "build_bg_hidden_origin_quota_dataset_v4.py",
    "train_bg_hidden_origin_quota_taps_v4.py",
    "evaluate_bg_hidden_origin_quota_selectors_v4.py",
    "evaluate_hidden_origin_taps_on_old_contexts_v4.py",
    "analyze_bg_hidden_origin_quota_v4_geometry.py",
    "analyze_bg_hidden_origin_quota_v4_experiment.py",
]


def ensure_v4_root() -> None:
    V4_ROOT.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=json_default_compact) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def json_default_compact(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return tensor_stats(value)
    if isinstance(value, Path):
        return rel(value)
    try:
        return float(value)
    except Exception:
        return str(value)


def load_pt(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return default


def mean_or_nan(values: Iterable[Any]) -> float:
    vals = []
    for value in values:
        try:
            x = float(value)
        except Exception:
            continue
        if math.isfinite(x):
            vals.append(x)
    return float(mean(vals)) if vals else float("nan")


def pstdev_or_zero(values: Iterable[Any]) -> float:
    vals = []
    for value in values:
        try:
            x = float(value)
        except Exception:
            continue
        if math.isfinite(x):
            vals.append(x)
    return float(pstdev(vals)) if len(vals) > 1 else 0.0


def normalized_task_class(value: Any) -> str:
    raw = str(value or "unknown")
    return TASK_CLASS_ALIASES.get(raw, raw)


def primary_domain_row(row: dict[str, Any]) -> bool:
    return str(row.get("domain") or "").lower() in PRIMARY_DOMAINS


def same_prefix_hidden_origin_row(row: dict[str, Any]) -> bool:
    method = str(row.get("branch_method") or row.get("method") or "")
    return method in {"", "hook_intervention_per_branch", "hook_hidden_origin_branch"}


def primary_safe_v4_row(row: dict[str, Any]) -> bool:
    return (
        str(row.get("split") or "") in {"train", "val", "heldout"}
        and primary_domain_row(row)
        and same_prefix_hidden_origin_row(row)
        and normalize_branch_point(row) in PRIMARY_BRANCH_POINTS
        and finite_float(row.get("alpha"), 999.0) <= PRIMARY_ALPHA_CAP
        and bool(row.get("safety_envelope", True))
        and str(row.get("label_source") or "deterministic") in {"deterministic", ""}
        and stable_v2_row(row)
    )


def diagnostic_alpha_v4_row(row: dict[str, Any]) -> bool:
    return (
        str(row.get("split") or "") in {"train", "val", "heldout"}
        and primary_domain_row(row)
        and same_prefix_hidden_origin_row(row)
        and abs(finite_float(row.get("alpha"), -1.0) - 0.02) <= 1e-6
        and stable_v2_row(row)
    )


def compact_v4_row(row: dict[str, Any]) -> dict[str, Any]:
    out = compact_branch_row_v3(row)
    for key in (
        "split",
        "recipe_id",
        "recipe_source",
        "task_class",
        "hs_score_at_allocation",
        "old_frozen_tap_score",
        "v1_tap_score",
        "v2_tap_score",
        "v3_tap_score",
        "salvage_tap_score",
    ):
        if key in row:
            out[key] = row.get(key)
    return out


def candidate_pair_stats(groups: dict[str, Sequence[dict[str, Any]]] | Iterable[Sequence[dict[str, Any]]], label_source: str = "deterministic") -> dict[str, Any]:
    vals_iter = groups.values() if isinstance(groups, dict) else groups
    candidate_pairs = tie_pairs = non_tie_pairs = 0
    for vals in vals_iter:
        ordered = list(vals)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                if label_source == "sampled_expected" and (sampled_reward(ordered[i]) is None or sampled_reward(ordered[j]) is None):
                    continue
                candidate_pairs += 1
                if row_reward(ordered[i], label_source) == row_reward(ordered[j], label_source):
                    tie_pairs += 1
                else:
                    non_tie_pairs += 1
    return {
        "candidate_pairs": candidate_pairs,
        "tie_pairs": tie_pairs,
        "non_tie_pairs": non_tie_pairs,
        "tie_rate": tie_pairs / max(candidate_pairs, 1),
    }


def split_quota_stats(rows: Sequence[dict[str, Any]], *, label_source: str = "deterministic", row_filter: Callable[[dict[str, Any]], bool] | None = None) -> dict[str, Any]:
    filt = row_filter or primary_safe_v4_row
    selected = [row for row in rows if filt(row)]
    out: dict[str, Any] = {}
    for split in ("train", "val", "heldout"):
        split_rows = [row for row in selected if str(row.get("split")) == split]
        groups = {gid: vals for gid, vals in group_rows(split_rows).items() if len(vals) >= 2}
        behavior = {gid for gid, vals in groups.items() if group_is_behaviorally_diverse_v2(vals, label_source)}
        reward = {gid for gid, vals in groups.items() if group_is_reward_diverse_v2(vals, label_source)}
        pair_stats = candidate_pair_stats(groups, label_source)
        task_ids_with_non_tie = set()
        for vals in groups.values():
            if candidate_pair_stats({"g": vals}, label_source)["non_tie_pairs"] > 0:
                task_ids_with_non_tie.add(str(vals[0].get("task_id")))
        out[split] = {
            "stable_primary_rows": len(split_rows),
            "groups": len(groups),
            "behaviorally_diverse_groups": len(behavior),
            "reward_diverse_groups": len(reward),
            "non_tie_pairs": pair_stats["non_tie_pairs"],
            "candidate_pairs": pair_stats["candidate_pairs"],
            "tie_pairs": pair_stats["tie_pairs"],
            "tie_rate": pair_stats["tie_rate"],
            "task_ids": len({str(row.get("task_id")) for row in split_rows}),
            "task_ids_with_non_tie_pairs": len(task_ids_with_non_tie),
            "task_ids_with_non_tie_pair_list": sorted(task_ids_with_non_tie),
            "parse_rate": sum(1 for row in split_rows if bool(row.get("parse_success"))) / max(len(split_rows), 1),
            "stability_rate": len(split_rows) / max(len([row for row in rows if str(row.get("split")) == split]), 1),
            "quota_minimums": PRIMARY_MINIMUMS[split],
            "minimum_met": quota_met_for_split(
                {
                    "task_ids_with_non_tie_pairs": len(task_ids_with_non_tie),
                    "task_ids": len({str(row.get("task_id")) for row in split_rows}),
                    "behaviorally_diverse_groups": len(behavior),
                    "non_tie_pairs": pair_stats["non_tie_pairs"],
                },
                split,
            ),
        }
    out["all_minimums_met"] = all(out[split]["minimum_met"] for split in ("train", "val", "heldout"))
    return out


def quota_met_for_split(stats: dict[str, Any], split: str, preferred: bool = False) -> bool:
    minimums = PREFERRED_MINIMUMS if preferred else PRIMARY_MINIMUMS
    task_count = int(stats.get("task_ids_with_non_tie_pairs") or stats.get("task_ids") or 0)
    return (
        task_count >= int(minimums[split]["task_ids"])
        and int(stats.get("behaviorally_diverse_groups") or 0) >= int(minimums[split]["behaviorally_diverse_groups"])
        and int(stats.get("non_tie_pairs") or 0) >= int(minimums[split]["non_tie_pairs"])
    )


def split_deficit_score(stats: dict[str, Any], split: str) -> float:
    current = stats.get(split, {})
    m = PRIMARY_MINIMUMS[split]
    task_count = max(int(current.get("task_ids_with_non_tie_pairs") or 0), int(current.get("task_ids") or 0))
    parts = [
        max(0.0, (m["task_ids"] - task_count) / max(m["task_ids"], 1)),
        max(0.0, (m["behaviorally_diverse_groups"] - int(current.get("behaviorally_diverse_groups") or 0)) / max(m["behaviorally_diverse_groups"], 1)),
        max(0.0, (m["non_tie_pairs"] - int(current.get("non_tie_pairs") or 0)) / max(m["non_tie_pairs"], 1)),
    ]
    return float(sum(parts))


def load_v4_branch_rows() -> list[dict[str, Any]]:
    payload = load_pt(BRANCHES_PT, None)
    if not payload:
        payload = load_pt(BRANCHES_PARTIAL_PT, {})
    rows = []
    for row in list((payload or {}).get("rows") or []):
        norm = dict(row)
        if "deterministic_reward" not in norm:
            norm["deterministic_reward"] = finite_float(norm.get("reward"), 0.0)
        if "reward" not in norm:
            norm["reward"] = norm["deterministic_reward"]
        if "deterministic_correct" not in norm:
            norm["deterministic_correct"] = bool(norm.get("correct"))
        if "correct" not in norm:
            norm["correct"] = bool(norm["deterministic_correct"])
        if "alpha_bucket" not in norm:
            norm["alpha_bucket"] = alpha_bucket(norm)
        if "label_source" not in norm:
            norm["label_source"] = "deterministic"
        rows.append(norm)
    return rows


def task_signal_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for task_id, vals in group_rows(rows, key="task_id").items():
        groups = {gid: g for gid, g in group_rows(vals).items() if len(g) >= 2}
        pair_stats = candidate_pair_stats(groups)
        behavior = sum(1 for g in groups.values() if group_is_behaviorally_diverse_v2(g))
        reward = sum(1 for g in groups.values() if group_is_reward_diverse_v2(g))
        first = vals[0]
        out.append(
            {
                "task_id": str(task_id),
                "domain": first.get("domain"),
                "source_dataset": first.get("source_dataset"),
                "source_subject": first.get("source_subject"),
                "task_screening_class": normalized_task_class(first.get("task_screening_class") or first.get("screening_class")),
                "groups": len(groups),
                "behaviorally_diverse_groups": behavior,
                "reward_diverse_groups": reward,
                "non_tie_pairs": pair_stats["non_tie_pairs"],
                "tie_rate": pair_stats["tie_rate"],
                "rows": len(vals),
            }
        )
    out.sort(key=lambda row: (int(row["behaviorally_diverse_groups"]), int(row["non_tie_pairs"]), -float(row["tie_rate"])), reverse=True)
    return out


def all_candidate_tasks() -> list[dict[str, Any]]:
    seen: set[str] = set()
    tasks = []
    for task in selected_v3_tasks() + load_more_candidate_tasks():
        norm = normalize_task(task, task.get("domain"), len(tasks)) or task
        if not norm:
            continue
        norm.update({k: v for k, v in task.items() if k not in norm})
        if str(norm.get("domain")) not in PRIMARY_DOMAINS:
            continue
        tid = str(norm["task_id"])
        lowered = tid.lower()
        if "devil" in lowered or lowered.startswith("code/") or "/code" in lowered:
            continue
        if tid in seen:
            continue
        seen.add(tid)
        tasks.append(norm)
    return tasks


def build_task_priority_rows() -> list[dict[str, Any]]:
    prior_rows = [row for row in load_all_v3_branch_rows(include_prior=True) if primary_safe_v3_row(row)]
    signal = {row["task_id"]: row for row in task_signal_rows(prior_rows)}
    seen = prior_seen_sets()
    prior_train_val = set(seen.get("v1_v2_train_val", [])) | set(seen.get("v3_train_val", []))
    prior_any = set(seen.get("v1_v2_any", [])) | set(seen.get("v3_any", []))
    high_yield_bonus = {
        "sciq/sciq/22": 5.0,
        "mmlu/high_school_chemistry/10": 4.0,
        "mmlu/high_school_physics/11": 4.0,
        "mmlu/anatomy/12": 3.0,
        "mmlu/anatomy/7": 2.5,
    }
    rows = []
    for task in all_candidate_tasks():
        tid = str(task["task_id"])
        sig = signal.get(tid, {})
        raw_class = normalized_task_class(task.get("screening_class") or sig.get("task_screening_class") or task.get("task_class"))
        if raw_class == "unknown":
            raw_class = "perturbation_sensitive" if sig.get("behaviorally_diverse_groups") else "wrong_parseable"
        cls_bonus = 2.0 if raw_class in PRIMARY_TASK_CLASSES else -0.5
        domain_bonus = 0.7 if str(task.get("domain")) in {"reasoning", "science"} else 0.0
        score = (
            float(task.get("priority_score") or 0.0)
            + 2.5 * int(sig.get("behaviorally_diverse_groups") or 0)
            + 0.06 * int(sig.get("non_tie_pairs") or 0)
            + cls_bonus
            + domain_bonus
            + high_yield_bonus.get(tid, 0.0)
        )
        rows.append(
            {
                **task,
                "task_id": tid,
                "task_class": raw_class,
                "task_screening_class": raw_class,
                "prior_behaviorally_diverse_groups": int(sig.get("behaviorally_diverse_groups") or 0),
                "prior_reward_diverse_groups": int(sig.get("reward_diverse_groups") or 0),
                "prior_non_tie_pairs": int(sig.get("non_tie_pairs") or 0),
                "prior_groups": int(sig.get("groups") or 0),
                "prior_any_contamination": tid in prior_any,
                "prior_train_val_contamination": tid in prior_train_val,
                "priority_score_v4": float(score),
                "priority_tier": "high" if score >= 8 else "medium" if score >= 2 else "low",
            }
        )
    rows.sort(key=lambda row: (-float(row["priority_score_v4"]), str(row.get("domain")), str(row["task_id"])))
    return rows


def balanced_pick(candidates: list[dict[str, Any]], count: int, used: set[str], *, prefer_clean: bool = False) -> list[dict[str, Any]]:
    pools = {"reasoning": [], "science": []}
    for row in candidates:
        tid = str(row["task_id"])
        if tid in used:
            continue
        if prefer_clean and row.get("prior_train_val_contamination"):
            continue
        pools.setdefault(str(row.get("domain")), []).append(row)
    picked: list[dict[str, Any]] = []
    while len(picked) < count and any(pools.values()):
        for domain in ("science", "reasoning"):
            if len(picked) >= count:
                break
            vals = pools.get(domain) or []
            if vals:
                row = vals.pop(0)
                picked.append(row)
                used.add(str(row["task_id"]))
    if len(picked) < count and prefer_clean:
        picked.extend(balanced_pick(candidates, count - len(picked), used, prefer_clean=False))
    return picked[:count]


def load_quota_plan() -> dict[str, Any]:
    return load_json(QUOTA_PLAN_JSON, {}) or {}


def plan_task_rows(plan: dict[str, Any], split: str | None = None) -> list[dict[str, Any]]:
    rows = list(plan.get("tasks") or [])
    if split:
        rows = [row for row in rows if row.get("split") == split]
    return rows


def split_task_ids(plan: dict[str, Any], split: str) -> set[str]:
    return {str(row["task_id"]) for row in plan_task_rows(plan, split)}


def task_by_id_from_plan(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["task_id"]): row for row in plan_task_rows(plan)}


def compatible_branch_points(layer: int | None) -> list[str]:
    if layer == 24:
        return ["L24"]
    if layer == 36:
        return ["L36"]
    if layer == 47:
        return ["L47"]
    return []


def direction_entry(
    *,
    name: str,
    family: str,
    tensor: torch.Tensor,
    source: str,
    target_layer: int | None,
    target_config: str | None = None,
    count: int = 0,
    trained_on_task_ids: Sequence[str] | None = None,
    heldout_task_ids: set[str] | None = None,
    recommended_alpha_bucket: str = "alpha_0_01",
) -> dict[str, Any]:
    vec = tensor.detach().cpu().flatten().to(torch.float32)
    trained = sorted({str(x) for x in (trained_on_task_ids or [])})
    heldout = heldout_task_ids or set()
    leakage_free = not bool(set(trained) & heldout)
    usable = target_layer in {24, 36, 47} and int(vec.numel()) == HIDDEN_DIM and leakage_free
    return {
        "name": name,
        "family": family,
        "source": source,
        "tensor": rms_normalize(vec) if int(vec.numel()) > 0 else vec,
        "direction_dim": int(vec.numel()),
        "target_layer": int(target_layer) if target_layer is not None else None,
        "target_config": target_config,
        "compatible_branch_points": compatible_branch_points(target_layer),
        "perturbation_usable": bool(usable),
        "scoring_only": not bool(usable),
        "count": int(count),
        "recommended_alpha_bucket": recommended_alpha_bucket,
        "trained_on_task_ids": trained,
        "heldout_leakage_free": bool(leakage_free),
        "leakage_intersection_task_ids": sorted(set(trained) & heldout),
        "stats": tensor_stats(vec) if int(vec.numel()) else {},
    }


def compact_direction_entry(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k != "tensor"}


def direction_bank_entries(bank: dict[str, Any], layer: int, family: str) -> list[dict[str, Any]]:
    by_layer = bank.get("directions_by_layer") or {}
    vals = list(by_layer.get(str(layer)) or by_layer.get(layer) or [])
    aliases = {
        "empirical_plus_noise": {"hidden_origin_empirical_train_only", "hidden_origin_whitened_train_only", "v3_tap_aligned", "v2_tap_aligned"},
        "hidden_origin_empirical": {"hidden_origin_empirical_train_only"},
        "hidden_origin_whitened": {"hidden_origin_whitened_train_only"},
    }
    allowed = aliases.get(family, {family})
    out = []
    for row in vals:
        if row.get("family") in allowed and row.get("perturbation_usable") and isinstance(row.get("tensor"), torch.Tensor):
            if int(row["tensor"].numel()) == HIDDEN_DIM:
                out.append(row)
    return out


def orthogonalize(raw: torch.Tensor, basis: list[torch.Tensor]) -> torch.Tensor:
    out = raw.detach().cpu().flatten().to(torch.float32).clone()
    for base in basis:
        unit = rms_normalize(base)
        out = out - torch.dot(out, unit) / torch.dot(unit, unit).clamp(min=1e-8) * unit
    return rms_normalize(out)


def make_family_deltas(layer: int, alpha: float, k: int, family: str, seed: int, bank: dict[str, Any]) -> list[dict[str, Any]]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    entries = [
        {
            "branch_id": 0,
            "delta": torch.zeros(HIDDEN_DIM, dtype=torch.float32),
            "delta_family": "clean",
            "delta_type": "clean_zero",
            "direction_name": "clean_zero",
        }
    ]
    basis: list[torch.Tensor] = []
    if family == "paired_plus_minus":
        while len(entries) < k:
            base = orthogonalize(torch.randn(HIDDEN_DIM, generator=gen), basis)
            basis.append(base)
            for sign, label in ((1.0, "plus"), (-1.0, "minus")):
                if len(entries) >= k:
                    break
                entries.append(
                    {
                        "branch_id": len(entries),
                        "delta": base * float(alpha) * sign,
                        "delta_family": family,
                        "delta_type": f"paired_{label}",
                        "direction_name": f"paired_base_{len(basis) - 1}",
                    }
                )
        return entries[:k]
    if family == "random_orthogonal":
        for idx in range(k - 1):
            vec = orthogonalize(torch.randn(HIDDEN_DIM, generator=gen), basis)
            basis.append(vec)
            entries.append(
                {
                    "branch_id": len(entries),
                    "delta": vec * float(alpha),
                    "delta_family": family,
                    "delta_type": f"random_orthogonal_{idx}",
                    "direction_name": f"random_{idx}",
                }
            )
        return entries[:k]
    candidates = direction_bank_entries(bank, layer, family)
    if not candidates:
        fallback = "random_fallback"
        for idx in range(k - 1):
            vec = orthogonalize(torch.randn(HIDDEN_DIM, generator=gen), basis)
            basis.append(vec)
            entries.append(
                {
                    "branch_id": len(entries),
                    "delta": vec * float(alpha),
                    "delta_family": fallback,
                    "delta_type": f"{family}_fallback_random_{idx}",
                    "direction_name": f"{family}_fallback",
                }
            )
        return entries[:k]
    idx = 0
    while len(entries) < k:
        item = candidates[idx % len(candidates)]
        base = item["tensor"].detach().cpu().flatten().to(torch.float32)
        if family == "empirical_plus_noise":
            raw = base + 0.20 * torch.randn(HIDDEN_DIM, generator=gen)
            vec = orthogonalize(raw, basis)
            basis.append(vec)
            entries.append(
                {
                    "branch_id": len(entries),
                    "delta": vec * float(alpha),
                    "delta_family": family,
                    "delta_type": f"noise_around_{item.get('family')}",
                    "direction_name": str(item.get("name")),
                }
            )
        else:
            vec = orthogonalize(base, basis)
            basis.append(vec)
            for sign, label in ((1.0, "plus"), (-1.0, "minus")):
                if len(entries) >= k:
                    break
                entries.append(
                    {
                        "branch_id": len(entries),
                        "delta": vec * float(alpha) * sign,
                        "delta_family": family,
                        "delta_type": f"{family}_{label}",
                        "direction_name": str(item.get("name") or family),
                    }
                )
        idx += 1
        if idx > len(candidates) * 4 and len(entries) < k:
            vec = orthogonalize(torch.randn(HIDDEN_DIM, generator=gen), basis)
            basis.append(vec)
            entries.append(
                {
                    "branch_id": len(entries),
                    "delta": vec * float(alpha),
                    "delta_family": f"{family}_plus_random",
                    "delta_type": "random_fill",
                    "direction_name": f"{family}_random_fill",
                }
            )
    return entries[:k]


def feature_pair(pref: dict[str, Any], rej: dict[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
    features: dict[str, dict[str, torch.Tensor]] = {}
    for config in CONFIGS:
        left = config_vector_from_row(pref, config)
        right = config_vector_from_row(rej, config)
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            if tuple(left.shape) == (config_dim(config),) and tuple(right.shape) == (config_dim(config),):
                features[config] = {"preferred": left.to(torch.float32), "rejected": right.to(torch.float32)}
    return features


def pair_metadata(
    pref: dict[str, Any],
    rej: dict[str, Any],
    *,
    pair_id: str,
    split: str,
    variant: str,
    label_source: str,
    features: dict[str, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
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
        "correctness_preferred": deterministic_correct(pref),
        "correctness_rejected": deterministic_correct(rej),
        "branch_point_preferred": pref.get("branch_point"),
        "branch_point_rejected": rej.get("branch_point"),
        "alpha_preferred": float(pref.get("alpha", 0.0)),
        "alpha_rejected": float(rej.get("alpha", 0.0)),
        "alpha_bucket_preferred": alpha_bucket(pref),
        "alpha_bucket_rejected": alpha_bucket(rej),
        "delta_family_preferred": pref.get("primary_delta_family") or pref.get("delta_family"),
        "delta_family_rejected": rej.get("primary_delta_family") or rej.get("delta_family"),
        "delta_type_preferred": pref.get("delta_type"),
        "delta_type_rejected": rej.get("delta_type"),
        "task_screening_class": pref.get("task_screening_class"),
        "recipe_id": pref.get("recipe_id"),
        "recipe_source": pref.get("recipe_source"),
        "old_frozen_tap_score_preferred": float(pref.get("old_frozen_tap_score", pref.get("tap_margin_sum", 0.0))),
        "old_frozen_tap_score_rejected": float(rej.get("old_frozen_tap_score", rej.get("tap_margin_sum", 0.0))),
        "v1_tap_score_preferred": pref.get("v1_tap_score"),
        "v1_tap_score_rejected": rej.get("v1_tap_score"),
        "v2_tap_score_preferred": pref.get("v2_tap_score"),
        "v2_tap_score_rejected": rej.get("v2_tap_score"),
        "v3_tap_score_preferred": pref.get("v3_tap_score"),
        "v3_tap_score_rejected": rej.get("v3_tap_score"),
        "salvage_tap_score_preferred": pref.get("salvage_tap_score"),
        "salvage_tap_score_rejected": rej.get("salvage_tap_score"),
        "available_configs": sorted(features.keys()),
        "features": features,
        "split": split,
    }


def compact_pair(pair: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in pair.items() if k != "features"}
    out["feature_dims"] = {cfg: int(pair["features"][cfg]["preferred"].numel()) for cfg in pair.get("features", {})}
    return out


def build_pairs_from_rows(
    rows: Sequence[dict[str, Any]],
    *,
    variant: str,
    label_source: str,
    row_filter: Callable[[dict[str, Any]], bool],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    valid_rows = [row for row in rows if row_filter(row) and stable_v2_row(row)]
    groups = {gid: vals for gid, vals in group_rows(valid_rows).items() if len(vals) >= 2}
    pairs: list[dict[str, Any]] = []
    tie_rows: list[dict[str, Any]] = []
    candidate_pairs = tie_pairs = omitted_missing_features = 0
    for gid, vals in sorted(groups.items()):
        ordered = sorted(vals, key=lambda row: int(row.get("branch_id", -1)))
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a = ordered[i]
                b = ordered[j]
                if label_source == "sampled_expected" and (sampled_reward(a) is None or sampled_reward(b) is None):
                    continue
                candidate_pairs += 1
                ra = row_reward(a, label_source)
                rb = row_reward(b, label_source)
                if ra == rb:
                    tie_pairs += 1
                    tie_rows.append(
                        {
                            "variant": variant,
                            "task_id": a.get("task_id"),
                            "branch_group_id": gid,
                            "branch_i": int(a.get("branch_id", -1)),
                            "branch_j": int(b.get("branch_id", -1)),
                            "reward": ra,
                            "split": a.get("split"),
                        }
                    )
                    continue
                pref, rej = (a, b) if ra > rb else (b, a)
                features = feature_pair(pref, rej)
                if not features:
                    omitted_missing_features += 1
                    continue
                pair_id = f"{variant}::{gid}::pair={int(a['branch_id'])}-{int(b['branch_id'])}"
                pairs.append(pair_metadata(pref, rej, pair_id=pair_id, split=str(pref["split"]), variant=variant, label_source=label_source, features=features))
    meta = {
        "valid_branch_count": len(valid_rows),
        "valid_group_count": len(groups),
        "candidate_unordered_pairs": candidate_pairs,
        "tie_pairs_omitted": tie_pairs,
        "tie_rate": tie_pairs / max(candidate_pairs, 1),
        "omitted_missing_features": omitted_missing_features,
        "behaviorally_diverse_groups": len([gid for gid, vals in groups.items() if group_is_behaviorally_diverse_v2(vals, label_source)]),
        "reward_diverse_groups": len([gid for gid, vals in groups.items() if group_is_reward_diverse_v2(vals, label_source)]),
    }
    return pairs, meta, tie_rows


def dataset_split_counts(pairs: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]], label_source: str = "deterministic") -> dict[str, Any]:
    pairs_by_split = Counter(str(pair["split"]) for pair in pairs)
    tasks_by_split = {name: sorted({str(pair["task_id"]) for pair in pairs if pair["split"] == name}) for name in ("train", "val", "heldout")}
    groups_by_split = {name: sorted({str(pair["branch_group_id"]) for pair in pairs if pair["split"] == name}) for name in ("train", "val", "heldout")}
    valid_groups = {gid: vals for gid, vals in group_rows(rows).items() if len(vals) >= 2}
    behavior_by_split: dict[str, int] = {}
    reward_by_split: dict[str, int] = {}
    for name, task_ids in tasks_by_split.items():
        task_set = set(task_ids)
        behavior_by_split[name] = len([gid for gid, vals in valid_groups.items() if str(vals[0].get("task_id")) in task_set and group_is_behaviorally_diverse_v2(vals, label_source)])
        reward_by_split[name] = len([gid for gid, vals in valid_groups.items() if str(vals[0].get("task_id")) in task_set and group_is_reward_diverse_v2(vals, label_source)])
    return {
        "pairs_by_split": dict(pairs_by_split),
        "tasks_by_split": tasks_by_split,
        "groups_by_split": {k: len(v) for k, v in groups_by_split.items()},
        "behaviorally_diverse_groups_by_split": behavior_by_split,
        "reward_diverse_groups_by_split": reward_by_split,
    }


def best_primary_head(path: Path, variant: str | None = None) -> dict[str, Any] | None:
    rows = load_head_rows(path, variant=variant, only_passing=True)
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("metrics", {}).get("validation_pairwise_accuracy", -1.0)))


def best_salvage_head() -> dict[str, Any] | None:
    return best_primary_head(SALVAGE_HEADS_PT, None)


def best_v4_head() -> dict[str, Any] | None:
    return best_primary_head(HEADS_V4_PT, "v4_only_primary_safe")


def load_selector_heads() -> dict[str, dict[str, Any] | None]:
    return {
        "v1": load_best_v1_head(),
        "v2": load_best_v2_head(),
        "v3": load_best_v3_head(),
        "salvage": best_salvage_head(),
        "v4": best_v4_head(),
    }


def annotate_group_metric(group: Sequence[dict[str, Any]], row: dict[str, Any], subset: str) -> dict[str, Any]:
    first = group[0]
    out = dict(row)
    out.update(
        {
            "branch_group_id": first.get("branch_group_id"),
            "task_id": first.get("task_id"),
            "domain": first.get("domain"),
            "branch_point": first.get("branch_point"),
            "alpha": first.get("alpha"),
            "alpha_bucket": first.get("alpha_bucket"),
            "delta_family": first.get("primary_delta_family") or first.get("delta_family"),
            "task_screening_class": first.get("task_screening_class"),
            "recipe_id": first.get("recipe_id"),
            "recipe_source": first.get("recipe_source"),
            "behaviorally_diverse": group_is_behaviorally_diverse_v2(group),
            "reward_diverse": group_is_reward_diverse_v2(group),
            "subset": subset,
        }
    )
    return out


def baseline_policy_rows(groups: Sequence[list[dict[str, Any]]], subset: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in groups:
        ordered = sorted(group, key=lambda row: int(row["branch_id"]))
        if not ordered:
            continue
        rewards = [float(row.get("deterministic_reward", row.get("reward", 0.0))) for row in ordered]
        correct = [1.0 if row.get("deterministic_correct", row.get("correct")) else 0.0 for row in ordered]
        oracle_reward = max(rewards)
        oracle_ids = {int(row["branch_id"]) for row in ordered if float(row.get("deterministic_reward", row.get("reward", 0.0))) == oracle_reward}
        old_idx = max(range(len(ordered)), key=lambda i: (float(ordered[i].get("old_frozen_tap_score", ordered[i].get("tap_margin_sum", 0.0))), -int(ordered[i]["branch_id"])))
        policies = {
            "clean_branch_baseline": selected_metric(ordered, [0]),
            "old_frozen_bg_highest_tap_score": selected_metric(ordered, [old_idx]),
            "old_frozen_bg_pairwise_tournament": selected_metric(ordered, [old_idx]),
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
            rows.append(annotate_group_metric(ordered, {"policy": policy, "config": "baseline", "architecture": "baseline", **metric}, subset))
    return rows


def tap_policy_rows(groups: Sequence[list[dict[str, Any]]], head_row: dict[str, Any] | None, device: torch.device, label: str, subset: str) -> list[dict[str, Any]]:
    if head_row is None:
        return []
    head = build_head(head_row, device)
    config = str(head_row["config"])
    rows: list[dict[str, Any]] = []
    for group in groups:
        ordered = sorted(group, key=lambda row: int(row["branch_id"]))
        vectors = [config_vector_from_row(row, config) for row in ordered]
        if not vectors or not all(isinstance(vec, torch.Tensor) and int(vec.numel()) == config_dim(config) for vec in vectors):
            continue
        mat = score_matrix(head, [vec for vec in vectors if isinstance(vec, torch.Tensor)], device)
        ranked = ranking_from_matrix(mat)
        ranking = list(ranked["ranking"])
        if not ranking:
            continue
        top1 = selected_metric(ordered, [ranking[0]])
        top2 = selected_metric(ordered, ranking[:2])
        for policy, metric in (
            (f"{label}_highest_score", top1),
            (f"{label}_pairwise_tournament", top1),
            (f"{label}_top2", top2),
        ):
            rows.append(
                annotate_group_metric(
                    ordered,
                    {
                        "policy": policy,
                        "config": config,
                        "architecture": head_row["architecture"],
                        "head_id": compact_head_id(head_row),
                        **metric,
                    },
                    subset,
                )
            )
    head.to("cpu")
    return rows


def ensemble_policy_rows(groups: Sequence[list[dict[str, Any]]], heads: dict[str, dict[str, Any] | None], device: torch.device, subset: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in groups:
        ordered = sorted(group, key=lambda row: int(row["branch_id"]))
        if not ordered:
            continue
        rank_votes = torch.zeros(len(ordered), dtype=torch.float32)
        used = []
        for label, head_row in heads.items():
            if head_row is None:
                continue
            config = str(head_row.get("config") or "")
            vectors = [config_vector_from_row(row, config) for row in ordered]
            if not vectors or not all(isinstance(vec, torch.Tensor) and int(vec.numel()) == config_dim(config) for vec in vectors):
                continue
            head = build_head(head_row, device)
            mat = score_matrix(head, [vec for vec in vectors if isinstance(vec, torch.Tensor)], device)
            head.to("cpu")
            ranking = list(ranking_from_matrix(mat)["ranking"])
            for rank, idx in enumerate(ranking):
                rank_votes[idx] += float(len(ordered) - rank)
            used.append(label)
        if not used:
            continue
        ranking = sorted(range(len(ordered)), key=lambda idx: (-float(rank_votes[idx]), idx))
        top1 = selected_metric(ordered, [ranking[0]])
        top2 = selected_metric(ordered, ranking[:2])
        for policy, metric in (("ensemble_rank_aggregation", top1), ("ensemble_rank_aggregation_top2", top2)):
            rows.append(
                annotate_group_metric(
                    ordered,
                    {
                        "policy": policy,
                        "config": "compatible_rank_votes",
                        "architecture": "rank_aggregation",
                        "head_id": "+".join(used),
                        **metric,
                    },
                    subset,
                )
            )
    return rows


def aggregate_subsets(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for subset in sorted({str(row.get("subset")) for row in rows}):
        vals = [row for row in rows if str(row.get("subset")) == subset]
        out[f"{subset}_metrics"] = aggregate(vals)
        behavior = [row for row in vals if bool(row.get("behaviorally_diverse"))]
        reward = [row for row in vals if bool(row.get("reward_diverse"))]
        out[f"{subset}_behaviorally_diverse_metrics"] = aggregate(behavior)
        out[f"{subset}_reward_diverse_metrics"] = aggregate(reward)
    return out


def breakdown(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    return {value: aggregate([row for row in rows if str(row.get(field)) == value]) for value in sorted({str(row.get(field)) for row in rows})}


def score_group_with_available_heads(group_new: list[dict[str, Any]], device: torch.device | None = None) -> None:
    score_old_taps(group_new)
    for row in group_new:
        if "tap_margin_sum" in row:
            row["old_frozen_tap_score"] = float(row.get("tap_margin_sum", 0.0))
    heads = {
        "v1_tap_score": load_best_v1_head(),
        "v2_tap_score": load_best_v2_head(),
        "v3_tap_score": load_best_v3_head(),
        "salvage_tap_score": best_salvage_head(),
    }
    for key, head in heads.items():
        score_group_with_head(group_new, head, score_key=key, device=device)


def commands_run() -> list[str]:
    return [f"venv/bin/python -m py_compile utilities/tests/manual/{script}" for script in V4_COMMAND_SCRIPTS] + [
        f"venv/bin/python -u utilities/tests/manual/{script}" for script in V4_COMMAND_SCRIPTS
    ]


def append_doc_section(path: Path, section_title: str, lines: Sequence[str]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if section_title in existing:
        return False
    prefix = "" if not existing or existing.endswith("\n") else "\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(prefix + "\n".join(lines) + "\n")
    return True
