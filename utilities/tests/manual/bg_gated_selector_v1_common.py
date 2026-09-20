"""Shared helpers for Gated/Fusion Branch-Content Selector v1.

This manual experiment trains only new standalone fusion/gated selectors over
cached expert scores, metadata, and diagnostics. It does not train Ouro, mutate
existing taps or registries, run wrapper/local-agent code, or use tap scores as
labels.
"""
from __future__ import annotations

import itertools
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from bg_hidden_origin_quota_v4_common import (
    CONFIGS,
    PROBE_ROOT,
    PROJECT_ROOT,
    best_primary_head,
    config_dim,
    md_table,
    rate,
    rel,
    tensor_stats,
    write_csv,
    write_md,
)
from bg_universal_tap_v1_common import (
    BRIDGE_PT,
    HIDDEN_BRANCH_PT,
    OLD_CONTENT_PT,
    UNIVERSAL_DATASET_PT,
    UNIVERSAL_HEADS_PT,
    aggregate_metric_rows,
    best_heads_from_training,
    branch_row_to_candidate,
    deterministic_split,
    domain_bucket,
    group_metric,
    load_branch_candidates,
    load_json,
    pair_counts,
    rank_group_with_head,
    write_json_local,
)
from bg_hidden_origin_tap_common import HEAD_CLASSES
from evaluate_bg_hidden_origin_taps import build_head


GATED_ROOT = PROBE_ROOT / "bg_gated_branch_content_selector_v1_2026-05-18"

INVENTORY_JSON = GATED_ROOT / "inventory.json"
INVENTORY_MD = GATED_ROOT / "inventory.md"
EXPERT_INVENTORY_CSV = GATED_ROOT / "expert_inventory.csv"
DATA_INVENTORY_CSV = GATED_ROOT / "data_inventory.csv"

EXPERT_SCORES_PT = GATED_ROOT / "expert_scores.pt"
EXPERT_SCORES_JSON = GATED_ROOT / "expert_scores.json"
EXPERT_SCORES_MD = GATED_ROOT / "expert_scores.md"
EXPERT_SCORE_ROWS_CSV = GATED_ROOT / "expert_score_rows.csv"

DATASET_PT = GATED_ROOT / "gated_selector_dataset.pt"
DATASET_JSON = GATED_ROOT / "gated_selector_dataset.json"
DATASET_MD = GATED_ROOT / "gated_selector_dataset.md"

HEADS_PT = GATED_ROOT / "gated_branch_content_selector_heads.pt"
TRAINING_JSON = GATED_ROOT / "training_log.json"
TRAINING_MD = GATED_ROOT / "training_report.md"

ABLATION_JSON = GATED_ROOT / "expert_ablation.json"
ABLATION_MD = GATED_ROOT / "expert_ablation.md"
ABLATION_CSV = GATED_ROOT / "expert_ablation_rows.csv"

OLD_CONTEXT_JSON = GATED_ROOT / "old_context_eval.json"
OLD_CONTEXT_MD = GATED_ROOT / "old_context_eval.md"
OLD_CONTEXT_CSV = GATED_ROOT / "old_context_eval_rows.csv"

HIDDEN_BRANCH_JSON = GATED_ROOT / "hidden_branch_eval.json"
HIDDEN_BRANCH_MD = GATED_ROOT / "hidden_branch_eval.md"
HIDDEN_BRANCH_CSV = GATED_ROOT / "hidden_branch_eval_rows.csv"

BRIDGE_JSON = GATED_ROOT / "bridge_eval.json"
BRIDGE_MD = GATED_ROOT / "bridge_eval.md"
BRIDGE_CSV = GATED_ROOT / "bridge_eval_rows.csv"

PRUNING_JSON = GATED_ROOT / "layerwise_pruning_sim.json"
PRUNING_MD = GATED_ROOT / "layerwise_pruning_sim.md"
PRUNING_CSV = GATED_ROOT / "layerwise_pruning_rows.csv"

DOMAIN_JSON = GATED_ROOT / "domain_coverage.json"
DOMAIN_MD = GATED_ROOT / "domain_coverage.md"

CALIBRATION_JSON = GATED_ROOT / "calibration_ood.json"
CALIBRATION_MD = GATED_ROOT / "calibration_ood.md"

GEOMETRY_JSON = GATED_ROOT / "geometry_interpretability.json"
GEOMETRY_MD = GATED_ROOT / "geometry_interpretability.md"

OLD_REPLACEMENT_JSON = GATED_ROOT / "old_tap_replacement_probe.json"
OLD_REPLACEMENT_MD = GATED_ROOT / "old_tap_replacement_probe.md"
OLD_REPLACEMENT_CSV = GATED_ROOT / "old_tap_replacement_probe_rows.csv"

SUMMARY_JSON = GATED_ROOT / "summary.json"
SUMMARY_MD = GATED_ROOT / "summary.md"
ANALYSIS_JSON = GATED_ROOT / "analysis.json"
ANALYSIS_MD = GATED_ROOT / "analysis.md"
DOC_MD = PROJECT_ROOT / "docs/evaluator/bg_gated_branch_content_selector_v1.md"
SIMPLE_DOC_MD = PROJECT_ROOT / "docs/evaluator/gated-branch-content-selector-v1.md"

OLD_CODE_HEAD_REGISTRY_PT = PROBE_ROOT / "bg_head_registry_2026-05-17.pt"
CODE_EXPANDED_FEATURES_PT = PROBE_ROOT / "code_expanded_strict_clean_features_2026-05-17.pt"
CODE_SPECIFIC_FEATURES_PT = PROBE_ROOT / "code_specific_training_features_2026-05-17.pt"
MIXED_HEADS_PT = PROBE_ROOT / "mixed_domain_tiny_heads_2026-05-17.pt"

GATED_SCRIPTS = [
    "bg_gated_selector_inventory_v1.py",
    "build_bg_gated_selector_expert_scores_v1.py",
    "build_bg_gated_selector_dataset_v1.py",
    "train_bg_gated_branch_content_selector_v1.py",
    "analyze_bg_gated_selector_expert_ablation_v1.py",
    "evaluate_bg_gated_selector_old_contexts_v1.py",
    "evaluate_bg_gated_selector_hidden_branches_v1.py",
    "evaluate_bg_gated_selector_bridge_pairs_v1.py",
    "simulate_bg_gated_layerwise_pruning_v1.py",
    "analyze_bg_gated_selector_domain_coverage_v1.py",
    "analyze_bg_gated_selector_calibration_ood_v1.py",
    "analyze_bg_gated_selector_geometry_v1.py",
    "evaluate_bg_gated_selector_as_old_tap_replacement_v1.py",
    "analyze_bg_gated_branch_content_selector_v1.py",
]

EXPERT_NAMES = [
    "old_frozen_bg",
    "old_content_head",
    "old_code_head",
    "mixed_code_reasoning_head",
    "mixed_objective_all_head",
    "hidden_origin_scalar",
    "v4_hidden_origin",
    "generator_v1_selector",
    "hidden_branch_head",
    "bridge_only_head",
    "universal",
    "universal_no_bridge",
]
HEAD_EXPERTS = {
    "old_content_head": "old_content_only",
    "old_code_head": "old_code_head",
    "mixed_code_reasoning_head": "mixed_code_reasoning_head",
    "mixed_objective_all_head": "mixed_objective_all_head",
    "v4_hidden_origin": "v4_hidden_origin",
    "generator_v1_selector": "generator_v1_selector",
    "hidden_branch_head": "hidden_branch_only",
    "bridge_only_head": "bridge_only",
    "universal": "universal_balanced",
    "universal_no_bridge": "universal_no_bridge",
}
PAIR_TYPES = ("old_content", "hidden_branch", "bridge")

OLD_CODE_CONFIG_DIMS = {
    "24_L1": 2048,
    "24_L4": 2048,
    "24_mean": 2048,
    "36_L1": 2048,
    "36_L4": 2048,
    "36_mean": 2048,
    "47_L4": 2048,
    "47_mean": 2048,
    "47_concat_L1_L4": 4096,
    "47_concat_all_loops": 8192,
    "concat_24_36": 4096,
    "concat_36_47": 4096,
    "concat_24_36_47": 6144,
}

CODE_LABEL_VALUES = {
    "correct": 1.0,
    "near_miss": 0.5,
    "wrong_code": 0.0,
    "incorrect": 0.0,
    "runtime_error": -0.2,
    "syntax_error": -0.3,
    "empty": -0.5,
}


def ensure_root() -> None:
    GATED_ROOT.mkdir(parents=True, exist_ok=True)


def load_pt(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return default


def json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return tensor_stats(value)
    if isinstance(value, Path):
        return rel(value)
    try:
        x = float(value)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        x = float(value)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def finite_values(values: Iterable[Any]) -> list[float]:
    out = []
    for value in values:
        x = safe_float(value)
        if math.isfinite(x):
            out.append(x)
    return out


def mean_or_nan(values: Iterable[Any]) -> float:
    vals = finite_values(values)
    return float(mean(vals)) if vals else float("nan")


def std_or_zero(values: Iterable[Any]) -> float:
    vals = finite_values(values)
    return float(pstdev(vals)) if len(vals) > 1 else 0.0


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def config_dim_any(config: str) -> int:
    if config in OLD_CODE_CONFIG_DIMS:
        return int(OLD_CODE_CONFIG_DIMS[config])
    return int(config_dim(config))


def expected_head_dim(head_row: dict[str, Any] | None) -> int:
    if not head_row:
        return 0
    dim = safe_float(head_row.get("dim"))
    if math.isfinite(dim) and dim > 0:
        return int(dim)
    return config_dim_any(str(head_row.get("config")))


def build_any_head(row: dict[str, Any], device: torch.device) -> nn.Module:
    architecture = str(row.get("architecture"))
    if architecture not in HEAD_CLASSES:
        return build_head(row, device)
    head = HEAD_CLASSES[architecture](expected_head_dim(row))
    head.load_state_dict(row["state_dict"])
    head.to(device=device, dtype=torch.float32)
    head.eval()
    return head


def feature_map_from_old_code_pooled(pooled: Any) -> dict[str, torch.Tensor]:
    if not isinstance(pooled, torch.Tensor):
        return {}
    x = pooled.detach().cpu().to(torch.float32)
    if x.ndim != 3 or x.shape[0] < 3 or x.shape[1] < 4 or x.shape[-1] != 2048:
        return {}
    out: dict[str, torch.Tensor] = {
        "24_L1": x[0, 0].clone(),
        "24_L4": x[0, 3].clone(),
        "24_mean": x[0, :4].mean(dim=0).clone(),
        "36_L1": x[1, 0].clone(),
        "36_L4": x[1, 3].clone(),
        "36_mean": x[1, :4].mean(dim=0).clone(),
        "47_L4": x[2, 3].clone(),
        "47_mean": x[2, :4].mean(dim=0).clone(),
    }
    out["47_concat_L1_L4"] = torch.cat([x[2, 0], x[2, 3]], dim=0)
    out["47_concat_all_loops"] = torch.cat([x[2, i] for i in range(4)], dim=0)
    out["concat_24_36"] = torch.cat([out["24_L4"], out["36_L4"]], dim=0)
    out["concat_36_47"] = torch.cat([out["36_L4"], out["47_L4"]], dim=0)
    out["concat_24_36_47"] = torch.cat([out["24_L4"], out["36_L4"], out["47_L4"]], dim=0)
    return {k: v for k, v in out.items() if int(v.numel()) == config_dim_any(k)}


def code_label_value(label: Any) -> float:
    return float(CODE_LABEL_VALUES.get(str(label or "").lower(), float("nan")))


def code_feature_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("candidate_features") or []
    index = {str(row.get("candidate_uid")): row for row in rows if row.get("candidate_uid")}
    for row in rows:
        meta = row.get("candidate_metadata") or {}
        for alias in meta.get("aliases") or []:
            index.setdefault(str(alias), row)
    return index


def code_feature_maps(payload: dict[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
    return {uid: feature_map_from_old_code_pooled(row.get("pooled")) for uid, row in code_feature_index(payload).items()}


def pair_features_from_maps(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    for config in sorted(set(left) & set(right)):
        lvec = left.get(config)
        rvec = right.get(config)
        if isinstance(lvec, torch.Tensor) and isinstance(rvec, torch.Tensor) and int(lvec.numel()) == int(rvec.numel()) == config_dim_any(config):
            out[config] = {"preferred": lvec.detach().cpu().to(torch.float32), "rejected": rvec.detach().cpu().to(torch.float32)}
    return out


def code_validation_task_ids() -> set[str]:
    registry = load_pt(OLD_CODE_HEAD_REGISTRY_PT, {}) or {}
    meta = registry.get("meta") or {}
    code_training = meta.get("code_training") or {}
    validation = code_training.get("validation") or {}
    return {str(task_id) for task_id in validation.get("val_task_ids") or []}


def code_pair_from_uids(
    *,
    task_id: str,
    preferred_uid: str,
    rejected_uid: str,
    preferred_label: str,
    rejected_label: str,
    source_id: str,
    split: str,
    maps: dict[str, dict[str, torch.Tensor]],
    pair_suffix: str,
) -> dict[str, Any] | None:
    left = maps.get(str(preferred_uid))
    right = maps.get(str(rejected_uid))
    if not left or not right:
        return None
    features = pair_features_from_maps(left, right)
    if not features:
        return None
    pref_reward = code_label_value(preferred_label)
    rej_reward = code_label_value(rejected_label)
    if not (math.isfinite(pref_reward) and math.isfinite(rej_reward)) or pref_reward == rej_reward:
        return None
    return {
        "pair_id": f"old_code::{source_id}::{pair_suffix}::{preferred_uid}__gt__{rejected_uid}",
        "pair_type": "old_content",
        "source_id": source_id,
        "variant": "old_code_trained_pool",
        "domain": "coding",
        "task_id": str(task_id),
        "group_id": f"old_code::{source_id}::{task_id}",
        "split": split,
        "left_origin": "old_candidate",
        "right_origin": "old_candidate",
        "origin_pair": "old_candidate+old_candidate",
        "branch_point": "old_context",
        "generator_method": "old_code_candidate_pool",
        "label_source": "code_unit_test_label",
        "preferred_label": preferred_label,
        "rejected_label": rejected_label,
        "reward_preferred": pref_reward,
        "reward_rejected": rej_reward,
        "reward_gap": pref_reward - rej_reward,
        "features": features,
        "available_configs": sorted(features.keys()),
        "contamination_flags": {"old_code_training_source": split in {"train", "val"}, "strict_clean_heldout": split == "heldout"},
    }


def load_old_code_pairs() -> list[dict[str, Any]]:
    payload = load_pt(CODE_EXPANDED_FEATURES_PT, {}) or load_pt(CODE_SPECIFIC_FEATURES_PT, {}) or {}
    if not payload:
        return []
    maps = code_feature_maps(payload)
    val_tasks = code_validation_task_ids()
    pairs: list[dict[str, Any]] = []
    for idx, row in enumerate(payload.get("training_pairs_primary") or []):
        task_id = str(row.get("task_id"))
        split = "val" if task_id in val_tasks else "train"
        pair = code_pair_from_uids(
            task_id=task_id,
            preferred_uid=str(row.get("preferred_uid")),
            rejected_uid=str(row.get("rejected_uid")),
            preferred_label=str(row.get("preferred_label")),
            rejected_label=str(row.get("rejected_label")),
            source_id="old_code_training_pairs",
            split=split,
            maps=maps,
            pair_suffix=str(idx),
        )
        if pair:
            pairs.append(pair)
    for set_name in ("ALL16_primary",):
        for tour in payload.get("eval_sets", {}).get(set_name, []) or []:
            uids = [str(uid) for uid in tour.get("candidate_uids") or []]
            labels = [str(label) for label in tour.get("labels") or []]
            task_id = str(tour.get("task_id"))
            for i in range(len(uids)):
                for j in range(i + 1, len(uids)):
                    li = code_label_value(labels[i] if i < len(labels) else "")
                    lj = code_label_value(labels[j] if j < len(labels) else "")
                    if not (math.isfinite(li) and math.isfinite(lj)) or li == lj:
                        continue
                    if li > lj:
                        pref_uid, rej_uid = uids[i], uids[j]
                        pref_label, rej_label = labels[i], labels[j]
                    else:
                        pref_uid, rej_uid = uids[j], uids[i]
                        pref_label, rej_label = labels[j], labels[i]
                    pair = code_pair_from_uids(
                        task_id=task_id,
                        preferred_uid=pref_uid,
                        rejected_uid=rej_uid,
                        preferred_label=pref_label,
                        rejected_label=rej_label,
                        source_id=f"old_code_strict_clean_{set_name}",
                        split="heldout",
                        maps=maps,
                        pair_suffix=f"{tour.get('tournament_id')}::{i}-{j}",
                    )
                    if pair:
                        pairs.append(pair)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for pair in pairs:
        pair_id = str(pair.get("pair_id"))
        if pair_id not in seen:
            seen.add(pair_id)
            out.append(pair)
    return out


def load_old_code_candidate_groups(split: str = "heldout") -> list[list[dict[str, Any]]]:
    payload = load_pt(CODE_EXPANDED_FEATURES_PT, {}) or {}
    if not payload:
        return []
    index = code_feature_index(payload)
    groups: list[list[dict[str, Any]]] = []
    set_names = ("ALL16_primary",) if split == "heldout" else ()
    for set_name in set_names:
        for tour in payload.get("eval_sets", {}).get(set_name, []) or []:
            candidates = []
            for uid, label in zip(tour.get("candidate_uids") or [], tour.get("labels") or []):
                row = index.get(str(uid))
                features = feature_map_from_old_code_pooled(row.get("pooled")) if row else {}
                reward = code_label_value(label)
                if not features or not math.isfinite(reward):
                    continue
                candidates.append(
                    {
                        "candidate_id": str(uid),
                        "task_id": str(tour.get("task_id")),
                        "group_id": f"old_code_strict_clean::{tour.get('task_id')}",
                        "domain": "coding",
                        "origin": "old_candidate",
                        "branch_point": "old_context",
                        "reward": reward,
                        "correct": str(label) == "correct",
                        "label": str(label),
                        "features_by_config": features,
                    }
                )
            if len(candidates) >= 2 and len({float(c.get("reward", 0.0)) for c in candidates}) >= 2:
                groups.append(candidates)
    return groups


def selection_score(row: dict[str, Any], preferred_domain: str | None = None) -> float:
    metrics = row.get("train_metrics") or row.get("metrics") or {}
    direct_keys = ("balanced_validation_score", "validation_pairwise_accuracy", "val_pair_acc", "heldout_pairwise_accuracy")
    vals = [safe_float(metrics.get(key)) for key in direct_keys]
    if preferred_domain and isinstance(metrics.get("domain_balance"), dict):
        domain_metrics = metrics["domain_balance"].get(preferred_domain)
        if isinstance(domain_metrics, dict):
            vals.insert(0, safe_float(domain_metrics.get("validation_pairwise_accuracy")))
    domain_vals = []
    if isinstance(metrics.get("domain_balance"), dict):
        for domain_metrics in metrics["domain_balance"].values():
            if isinstance(domain_metrics, dict):
                v = safe_float(domain_metrics.get("validation_pairwise_accuracy"))
                if math.isfinite(v):
                    domain_vals.append(v)
    if domain_vals:
        vals.append(float(mean(domain_vals)))
    finite = [v for v in vals if math.isfinite(v)]
    return max(finite) if finite else -1.0


def best_old_code_head() -> dict[str, Any] | None:
    registry = load_pt(OLD_CODE_HEAD_REGISTRY_PT, {}) or {}
    heads = [row for row in registry.get("heads") or [] if str(row.get("head_family")) == "CODE"]
    if not heads:
        return None
    row = max(heads, key=selection_score)
    out = dict(row)
    out["expert_source"] = "bg_head_registry_2026-05-17"
    return out


def best_mixed_head(head_group: str, preferred_domain: str = "CODE") -> dict[str, Any] | None:
    payload = load_pt(MIXED_HEADS_PT, {}) or {}
    heads = [row for row in payload.get("heads") or [] if str(row.get("head_group")) == head_group]
    if not heads:
        return None
    row = max(heads, key=lambda h: selection_score(h, preferred_domain))
    out = dict(row)
    out["expert_source"] = "mixed_domain_tiny_heads_2026-05-17"
    return out


def compact_pair(pair: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in pair.items() if k != "features"}
    out["feature_dims"] = {cfg: int(vals["preferred"].numel()) for cfg, vals in (pair.get("features") or {}).items()}
    return out


def feature_map_from_candidates(left: dict[str, Any], right: dict[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
    left_map = left.get("features_by_config") or {}
    right_map = right.get("features_by_config") or {}
    out: dict[str, dict[str, torch.Tensor]] = {}
    for config in sorted(set(left_map) & set(right_map)):
        lvec = left_map.get(config)
        rvec = right_map.get(config)
        if isinstance(lvec, torch.Tensor) and isinstance(rvec, torch.Tensor):
            if int(lvec.numel()) == int(rvec.numel()) == config_dim_any(config):
                out[config] = {
                    "preferred": lvec.detach().cpu().to(torch.float32),
                    "rejected": rvec.detach().cpu().to(torch.float32),
                }
    return out


def synthetic_pair_from_candidates(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    pair_type: str,
    source_id: str,
    split: str,
) -> dict[str, Any]:
    return {
        "pair_id": f"synthetic::{pair_type}::{left.get('candidate_id')}::{right.get('candidate_id')}",
        "pair_type": pair_type,
        "source_id": source_id,
        "variant": "synthetic_candidate_pair",
        "bridge_type": "candidate_pair" if pair_type == "bridge" else None,
        "domain": domain_bucket(left.get("domain") or right.get("domain")),
        "task_id": str(left.get("task_id") or right.get("task_id")),
        "group_id": str(left.get("group_id") or right.get("group_id")),
        "split": split,
        "left_origin": left.get("origin"),
        "right_origin": right.get("origin"),
        "branch_point": left.get("branch_point") or right.get("branch_point") or "old_context",
        "generator_method": left.get("generator_method") or right.get("generator_method"),
        "features": feature_map_from_candidates(left, right),
        "available_configs": sorted(feature_map_from_candidates(left, right).keys()),
        "old_frozen_tap_score_preferred": left.get("old_frozen_tap_score"),
        "old_frozen_tap_score_rejected": right.get("old_frozen_tap_score"),
        "hidden_origin_tap_score_preferred": left.get("hidden_origin_tap_score"),
        "hidden_origin_tap_score_rejected": right.get("hidden_origin_tap_score"),
    }


def all_source_pairs() -> list[dict[str, Any]]:
    dataset = load_pt(UNIVERSAL_DATASET_PT, {}) or {}
    by_variant = dataset.get("pairs_by_variant") or {}
    pairs: list[dict[str, Any]] = []
    for variant in ("old_content_only", "hidden_branch_only", "bridge_only"):
        for pair in by_variant.get(variant) or []:
            row = dict(pair)
            row["dataset_variant"] = variant
            row.setdefault("pair_type", variant.replace("_only", ""))
            if row["pair_type"] == "old_content":
                row.setdefault("branch_point", "old_context")
            else:
                row.setdefault("branch_point", row.get("branch_point_preferred") or row.get("branch_point_rejected") or row.get("branch_point") or "unknown")
            row.setdefault("domain", domain_bucket(row.get("domain")))
            row.setdefault("split", deterministic_split(str(row.get("task_id")), salt=f"gated::{variant}"))
            pairs.append(row)
    pairs.extend(load_old_code_pairs())
    return pairs


def expert_head_rows() -> dict[str, dict[str, Any] | None]:
    heads = best_heads_from_training()
    out: dict[str, dict[str, Any] | None] = {}
    for expert, label in HEAD_EXPERTS.items():
        out[expert] = heads.get(label)
    out["old_code_head"] = best_old_code_head()
    out["mixed_code_reasoning_head"] = best_mixed_head("MIX_CODE_REASONING", "CODE")
    out["mixed_objective_all_head"] = best_mixed_head("MIX_OBJECTIVE_ALL", "CODE")
    return out


def scalar_margin(pair: dict[str, Any], pref_keys: Sequence[str], rej_keys: Sequence[str]) -> float:
    pref = None
    rej = None
    for key in pref_keys:
        if pair.get(key) is not None:
            pref = pair.get(key)
            break
    for key in rej_keys:
        if pair.get(key) is not None:
            rej = pair.get(key)
            break
    a = safe_float(pref)
    b = safe_float(rej)
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("nan")
    return a - b


def score_head_pair(head: nn.Module, head_row: dict[str, Any], pair: dict[str, Any], device: torch.device) -> float:
    config = str(head_row.get("config"))
    vals = (pair.get("features") or {}).get(config)
    if not vals:
        return float("nan")
    left = vals.get("preferred")
    right = vals.get("rejected")
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return float("nan")
    expected = expected_head_dim(head_row)
    if int(left.numel()) != expected or int(right.numel()) != expected:
        return float("nan")
    with torch.no_grad():
        score = head(left.view(1, -1).to(device), right.view(1, -1).to(device))
    return float(score.detach().cpu().flatten()[0].item())


def score_pair_with_experts(
    pair: dict[str, Any],
    head_rows: dict[str, dict[str, Any] | None],
    head_modules: dict[str, nn.Module],
    device: torch.device,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    scores["old_frozen_bg"] = scalar_margin(
        pair,
        ("old_frozen_tap_score_preferred", "tap_margin_sum_preferred", "old_score_preferred"),
        ("old_frozen_tap_score_rejected", "tap_margin_sum_rejected", "old_score_rejected"),
    )
    scores["hidden_origin_scalar"] = scalar_margin(
        pair,
        (
            "hidden_origin_tap_score_preferred",
            "v4_tap_score_preferred",
            "v3_tap_score_preferred",
            "v2_tap_score_preferred",
            "v1_tap_score_preferred",
            "salvage_tap_score_preferred",
        ),
        (
            "hidden_origin_tap_score_rejected",
            "v4_tap_score_rejected",
            "v3_tap_score_rejected",
            "v2_tap_score_rejected",
            "v1_tap_score_rejected",
            "salvage_tap_score_rejected",
        ),
    )
    for expert in HEAD_EXPERTS:
        head_row = head_rows.get(expert)
        head = head_modules.get(expert)
        scores[expert] = score_head_pair(head, head_row, pair, device) if head_row and head else float("nan")
    return scores


def calibration_from_rows(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for expert in EXPERT_NAMES:
        vals = [row["raw_scores"].get(expert) for row in rows if row.get("split") == "train"]
        finite = finite_values(vals)
        mu = float(mean(finite)) if finite else 0.0
        sd = float(pstdev(finite)) if len(finite) > 1 else 1.0
        if sd < 1e-8:
            sd = 1.0
        out[expert] = {"mean": mu, "std": sd, "train_count": len(finite)}
    return out


def apply_calibration(raw: dict[str, float], cal: dict[str, dict[str, float]]) -> dict[str, float]:
    out = {}
    for expert in EXPERT_NAMES:
        value = safe_float(raw.get(expert))
        if not math.isfinite(value):
            out[expert] = 0.0
            continue
        stats = cal.get(expert, {"mean": 0.0, "std": 1.0})
        out[expert] = (value - float(stats.get("mean", 0.0))) / max(float(stats.get("std", 1.0)), 1e-8)
    return out


def score_diagnostics(zscores: dict[str, float], raw: dict[str, float]) -> dict[str, float]:
    available = [zscores[e] for e in EXPERT_NAMES if math.isfinite(safe_float(raw.get(e)))]
    signs = [1 if x > 0 else -1 if x < 0 else 0 for x in available]
    pos = sum(1 for s in signs if s > 0)
    neg = sum(1 for s in signs if s < 0)
    denom = max(pos + neg, 1)
    old = max(
        zscores.get("old_content_head", zscores.get("old_frozen_bg", 0.0)),
        zscores.get("old_code_head", 0.0),
        zscores.get("mixed_objective_all_head", 0.0),
    )
    branch = zscores.get("v4_hidden_origin", zscores.get("hidden_branch_head", 0.0))
    bridge = zscores.get("bridge_only_head", 0.0)
    uni = zscores.get("universal", 0.0)
    return {
        "score_count": float(len(available)),
        "score_mean": float(mean(available)) if available else 0.0,
        "score_std": float(pstdev(available)) if len(available) > 1 else 0.0,
        "score_abs_mean": float(mean(abs(x) for x in available)) if available else 0.0,
        "score_abs_max": float(max(abs(x) for x in available)) if available else 0.0,
        "expert_disagreement_rate": float(min(pos, neg) / denom),
        "old_branch_gap": float(old - branch),
        "branch_bridge_gap": float(branch - bridge),
        "universal_specialist_gap": float(uni - mean([old, branch, bridge])),
        "abs_old_branch_gap": float(abs(old - branch)),
        "abs_branch_bridge_gap": float(abs(branch - bridge)),
        "abs_universal_specialist_gap": float(abs(uni - mean([old, branch, bridge]))),
    }


def compact_score_row(row: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in row.items() if k not in {"pair", "features"}}
    out["raw_scores"] = {k: rate(v) for k, v in row.get("raw_scores", {}).items()}
    out["calibrated_scores"] = {k: rate(v) for k, v in row.get("calibrated_scores", {}).items()}
    return out


def score_row_csv(row: dict[str, Any]) -> dict[str, Any]:
    out = {
        "pair_id": row.get("pair_id"),
        "split": row.get("split"),
        "pair_type": row.get("pair_type"),
        "domain": row.get("domain"),
        "source_id": row.get("source_id"),
        "task_id": row.get("task_id"),
    }
    for expert in EXPERT_NAMES:
        out[f"raw_{expert}"] = row.get("raw_scores", {}).get(expert)
        out[f"z_{expert}"] = row.get("calibrated_scores", {}).get(expert)
        out[f"missing_{expert}"] = row.get("missing", {}).get(expert)
    for key, value in (row.get("diagnostics") or {}).items():
        out[key] = value
    return out


def run_inventory() -> int:
    ensure_root()
    started = time.time()
    pairs = all_source_pairs()
    head_rows = expert_head_rows()
    data_rows = []
    for name, path in (
        ("universal_tap_dataset", UNIVERSAL_DATASET_PT),
        ("old_content_dataset", OLD_CONTENT_PT),
        ("hidden_branch_dataset", HIDDEN_BRANCH_PT),
        ("bridge_dataset", BRIDGE_PT),
        ("old_code_expanded_strict_clean_features", CODE_EXPANDED_FEATURES_PT),
        ("old_code_head_registry", OLD_CODE_HEAD_REGISTRY_PT),
        ("mixed_domain_tiny_heads", MIXED_HEADS_PT),
    ):
        payload = load_pt(path, {}) or {}
        if name == "old_code_expanded_strict_clean_features":
            source_pairs = load_old_code_pairs()
        else:
            source_pairs = payload.get("pairs")
        if source_pairs is None and name == "universal_tap_dataset":
            source_pairs = pairs
        source_pairs = list(source_pairs or [])
        data_rows.append(
            {
                "source_name": name,
                "path": rel(path),
                "pair_count": len(source_pairs),
                "task_count": len({str(p.get("task_id")) for p in source_pairs}),
                "group_count": len({str(p.get("group_id") or p.get("branch_group_id")) for p in source_pairs}),
                "domains": dict(Counter(str(p.get("domain")) for p in source_pairs)),
                "pair_types": dict(Counter(str(p.get("pair_type")) for p in source_pairs)),
                "splits": dict(Counter(str(p.get("split")) for p in source_pairs)),
                "feature_configs": sorted({cfg for p in source_pairs for cfg in (p.get("features") or {})}),
                "status": "usable" if source_pairs else "missing_or_empty",
            }
        )
    expert_rows = []
    for expert in EXPERT_NAMES:
        row = {
            "expert_name": expert,
            "source_run": "cached_scalar"
            if expert in {"old_frozen_bg", "hidden_origin_scalar"}
            else "old_code_or_mixed_head"
            if expert in {"old_code_head", "mixed_code_reasoning_head", "mixed_objective_all_head"}
            else "universal_or_hidden_origin_head",
            "score_orientation": "positive_prefers_left_or_preferred",
            "score_is_label": False,
            "compatible_data_sources": "computed_if_feature_config_or_cached_scalar_available",
        }
        head = head_rows.get(expert)
        if head:
            row.update(
                {
                    "architecture": head.get("architecture"),
                    "config": head.get("config"),
                    "feature_dim": expected_head_dim(head),
                    "domains_supported": "reasoning/science/old-context where compatible",
                    "pair_types_supported": "old_content, hidden_branch, bridge if config compatible",
                    "heldout_contamination_risk": "reported_per_source; not used as label",
                    "calibration_stats_exist": False,
                }
            )
        else:
            row.update({"architecture": "scalar_or_missing", "config": "score_margin", "feature_dim": 1})
        expert_rows.append(row)
    counts = pair_counts(pairs)
    mandatory = all(counts["pairs_by_type"].get(pt, 0) > 0 for pt in PAIR_TYPES)
    has_heads = any(head_rows.get(expert) for expert in HEAD_EXPERTS)
    if mandatory and has_heads:
        verdict = "READY"
    elif pairs and has_heads:
        verdict = "PARTIAL"
    elif pairs:
        verdict = "DATA_LIMITED"
    else:
        verdict = "BLOCKED"
    payload = {
        "BG_GATED_SELECTOR_INVENTORY_VERDICT": verdict,
        "verdict": verdict,
        "pair_counts": counts,
        "expert_inventory": expert_rows,
        "data_inventory": data_rows,
        "constraints": {
            "no_tap_score_labels": True,
            "tap_scores_as_input_features_only": True,
            "no_ouro_training": True,
            "no_production_routing_change": True,
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(INVENTORY_JSON, payload)
    write_csv(EXPERT_INVENTORY_CSV, expert_rows)
    write_csv(DATA_INVENTORY_CSV, data_rows)
    lines = ["# Gated Selector Inventory", "", f"BG_GATED_SELECTOR_INVENTORY_VERDICT = {verdict}", "", "## Experts", ""]
    lines.extend(md_table(expert_rows, ["expert_name", "architecture", "config", "feature_dim", "score_orientation"]))
    lines.extend(["", "## Data Sources", ""])
    lines.extend(md_table(data_rows, ["source_name", "pair_count", "task_count", "group_count", "status"]))
    write_md(INVENTORY_MD, lines)
    print(f"BG_GATED_SELECTOR_INVENTORY_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


def run_expert_scores() -> int:
    ensure_root()
    started = time.time()
    pairs = all_source_pairs()
    if not pairs:
        payload = {"BG_GATED_SELECTOR_EXPERT_SCORES_VERDICT": "BLOCKED", "verdict": "BLOCKED", "blocker": "missing universal source pairs"}
        write_json(EXPERT_SCORES_JSON, payload)
        write_md(EXPERT_SCORES_MD, ["# Gated Expert Scores", "", "BG_GATED_SELECTOR_EXPERT_SCORES_VERDICT = BLOCKED"])
        print("BG_GATED_SELECTOR_EXPERT_SCORES_VERDICT = BLOCKED", flush=True)
        return 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head_rows = expert_head_rows()
    head_modules: dict[str, nn.Module] = {}
    for expert, row in head_rows.items():
        if row:
            head_modules[expert] = build_any_head(row, device)
    rows = []
    for pair in pairs:
        raw = score_pair_with_experts(pair, head_rows, head_modules, device)
        missing = {expert: 0 if math.isfinite(safe_float(raw.get(expert))) else 1 for expert in EXPERT_NAMES}
        left_origin = str(pair.get("left_origin") or "unknown")
        right_origin = str(pair.get("right_origin") or "unknown")
        branch_pref = str(pair.get("branch_point_preferred") or pair.get("branch_point") or "unknown")
        branch_rej = str(pair.get("branch_point_rejected") or pair.get("branch_point") or "unknown")
        branch_point = branch_pref if branch_pref == branch_rej else "mixed"
        rows.append(
            {
                "pair_id": pair.get("pair_id"),
                "pair_type": str(pair.get("pair_type")),
                "split": str(pair.get("split")),
                "domain": domain_bucket(pair.get("domain")),
                "task_id": str(pair.get("task_id")),
                "group_id": str(pair.get("group_id") or pair.get("branch_group_id")),
                "source_id": str(pair.get("source_id")),
                "bridge_type": str(pair.get("bridge_type") or "none"),
                "left_origin": left_origin,
                "right_origin": right_origin,
                "origin_pair": "+".join(sorted([left_origin, right_origin])),
                "branch_point": branch_point,
                "generator_method": str(pair.get("generator_method") or pair.get("recipe_source") or "unknown"),
                "reward_gap": safe_float(pair.get("reward_gap"), 1.0),
                "raw_scores": raw,
                "missing": missing,
                "pair": pair,
            }
        )
    calibration = calibration_from_rows(rows)
    for row in rows:
        row["calibrated_scores"] = apply_calibration(row["raw_scores"], calibration)
        row["diagnostics"] = score_diagnostics(row["calibrated_scores"], row["raw_scores"])
    coverage = {
        expert: {
            "available": sum(1 for row in rows if row["missing"].get(expert) == 0),
            "coverage": sum(1 for row in rows if row["missing"].get(expert) == 0) / max(len(rows), 1),
            "train_calibration_count": calibration.get(expert, {}).get("train_count", 0),
        }
        for expert in EXPERT_NAMES
    }
    family_coverage = {}
    for pair_type in PAIR_TYPES:
        vals = [row for row in rows if row["pair_type"] == pair_type]
        family_coverage[pair_type] = {
            expert: sum(1 for row in vals if row["missing"].get(expert) == 0) / max(len(vals), 1)
            for expert in EXPERT_NAMES
        }
    ready_families = sum(1 for pt, cov in family_coverage.items() if max(cov.values() or [0.0]) >= 0.5)
    if ready_families == 3:
        verdict = "READY"
    elif ready_families >= 2:
        verdict = "PARTIAL"
    elif rows:
        verdict = "CALIBRATION_WEAK"
    else:
        verdict = "BLOCKED"
    payload = {
        "BG_GATED_SELECTOR_EXPERT_SCORES_VERDICT": verdict,
        "verdict": verdict,
        "expert_names": EXPERT_NAMES,
        "head_rows": {k: {kk: vv for kk, vv in (v or {}).items() if kk != "state_dict"} for k, v in head_rows.items()},
        "calibration": calibration,
        "coverage": coverage,
        "family_coverage": family_coverage,
        "rows": rows,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(payload, EXPERT_SCORES_PT)
    json_payload = {k: v for k, v in payload.items() if k != "rows"}
    json_payload["rows"] = [compact_score_row(row) for row in rows[:200]]
    json_payload["row_count"] = len(rows)
    write_json(EXPERT_SCORES_JSON, json_payload)
    write_csv(EXPERT_SCORE_ROWS_CSV, [score_row_csv(row) for row in rows])
    coverage_rows = [{"expert": expert, **stats} for expert, stats in coverage.items()]
    lines = ["# Gated Expert Scores", "", f"BG_GATED_SELECTOR_EXPERT_SCORES_VERDICT = {verdict}", "", "## Coverage", ""]
    lines.extend(md_table(coverage_rows, ["expert", "available", "coverage", "train_calibration_count"]))
    write_md(EXPERT_SCORES_MD, lines)
    for module in head_modules.values():
        module.to("cpu")
    print(f"BG_GATED_SELECTOR_EXPERT_SCORES_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


def categorical_vocab(rows: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    fields = ["pair_type", "domain", "source_id", "bridge_type", "origin_pair", "branch_point", "generator_method"]
    return {field: sorted({str(row.get(field) or "unknown") for row in rows}) for field in fields}


SIGNED_DIAGNOSTICS = {"score_mean", "old_branch_gap", "branch_bridge_gap", "universal_specialist_gap"}


def build_feature_names(vocab: dict[str, list[str]]) -> tuple[list[str], set[str]]:
    names: list[str] = []
    directional: set[str] = set()
    for expert in EXPERT_NAMES:
        for prefix in ("raw", "z"):
            name = f"{prefix}:{expert}"
            names.append(name)
            directional.add(name)
        names.append(f"missing:{expert}")
    diag_keys = [
        "score_count",
        "score_mean",
        "score_std",
        "score_abs_mean",
        "score_abs_max",
        "expert_disagreement_rate",
        "old_branch_gap",
        "branch_bridge_gap",
        "universal_specialist_gap",
        "abs_old_branch_gap",
        "abs_branch_bridge_gap",
        "abs_universal_specialist_gap",
    ]
    for key in diag_keys:
        name = f"diag:{key}"
        names.append(name)
        if key in SIGNED_DIAGNOSTICS:
            directional.add(name)
    for field, values in vocab.items():
        for value in values:
            names.append(f"cat:{field}={value}")
    return names, directional


def vector_from_score_row(row: dict[str, Any], feature_names: Sequence[str], vocab: dict[str, list[str]]) -> list[float]:
    vals = []
    raw = row.get("raw_scores") or {}
    zscores = row.get("calibrated_scores") or {}
    missing = row.get("missing") or {}
    diag = row.get("diagnostics") or {}
    cats = {field: str(row.get(field) or "unknown") for field in vocab}
    for name in feature_names:
        if name.startswith("raw:"):
            expert = name.split(":", 1)[1]
            x = safe_float(raw.get(expert), 0.0)
            vals.append(x if math.isfinite(x) else 0.0)
        elif name.startswith("z:"):
            expert = name.split(":", 1)[1]
            x = safe_float(zscores.get(expert), 0.0)
            vals.append(x if math.isfinite(x) else 0.0)
        elif name.startswith("missing:"):
            expert = name.split(":", 1)[1]
            vals.append(float(missing.get(expert, 1)))
        elif name.startswith("diag:"):
            key = name.split(":", 1)[1]
            vals.append(float(diag.get(key, 0.0)))
        elif name.startswith("cat:"):
            field_value = name.split(":", 1)[1]
            field, value = field_value.split("=", 1)
            vals.append(1.0 if cats.get(field) == value else 0.0)
        else:
            vals.append(0.0)
    return vals


def dataset_counts(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pairs": len(rows),
        "pairs_by_split": dict(Counter(str(r.get("split")) for r in rows)),
        "pairs_by_type": dict(Counter(str(r.get("pair_type")) for r in rows)),
        "pairs_by_domain": dict(Counter(str(r.get("domain")) for r in rows)),
        "tasks_by_split": {split: len({str(r.get("task_id")) for r in rows if r.get("split") == split}) for split in ("train", "val", "heldout")},
    }


def run_dataset() -> int:
    ensure_root()
    started = time.time()
    score_payload = load_pt(EXPERT_SCORES_PT, {}) or {}
    rows = list(score_payload.get("rows") or [])
    if not rows:
        payload = {"BG_GATED_SELECTOR_DATASET_VERDICT": "BLOCKED", "verdict": "BLOCKED", "blocker": "missing expert score rows"}
        write_json(DATASET_JSON, payload)
        write_md(DATASET_MD, ["# Gated Selector Dataset", "", "BG_GATED_SELECTOR_DATASET_VERDICT = BLOCKED"])
        print("BG_GATED_SELECTOR_DATASET_VERDICT = BLOCKED", flush=True)
        return 1
    vocab = categorical_vocab(rows)
    feature_names, directional_names = build_feature_names(vocab)
    x = torch.tensor([vector_from_score_row(row, feature_names, vocab) for row in rows], dtype=torch.float32)
    y = torch.ones((len(rows),), dtype=torch.float32)
    directional_indices = [i for i, name in enumerate(feature_names) if name in directional_names]
    score_indices = [i for i, name in enumerate(feature_names) if name.startswith("raw:") or name.startswith("z:") or name.startswith("missing:")]
    metadata_indices = [i for i, name in enumerate(feature_names) if name.startswith("cat:") or name.startswith("missing:")]
    diagnostic_indices = [i for i, name in enumerate(feature_names) if name.startswith("diag:")]
    expert_z_indices = {expert: feature_names.index(f"z:{expert}") for expert in EXPERT_NAMES if f"z:{expert}" in feature_names}
    variants = {
        "gated_full": list(range(len(feature_names))),
        "gated_no_bridge": list(range(len(feature_names))),
        "gated_scores_only": score_indices,
        "gated_scores_plus_metadata": sorted(set(score_indices + metadata_indices)),
        "gated_scores_plus_diagnostics": list(range(len(feature_names))),
        "old_new_composite_baseline": [
            expert_z_indices[e]
            for e in ("old_content_head", "old_code_head", "mixed_code_reasoning_head", "v4_hidden_origin", "bridge_only_head", "universal")
            if e in expert_z_indices
        ],
        "universal_baseline": [expert_z_indices[e] for e in ("universal",) if e in expert_z_indices],
    }
    counts = dataset_counts(rows)
    heldout_ok = all(counts["pairs_by_split"].get(split, 0) > 0 for split in ("train", "val", "heldout"))
    family_ok = all(counts["pairs_by_type"].get(pt, 0) > 0 for pt in PAIR_TYPES)
    if heldout_ok and family_ok:
        verdict = "READY"
    elif family_ok:
        verdict = "DATA_LIMITED"
    elif counts["pairs_by_type"].get("bridge", 0) > 0:
        verdict = "BRIDGE_WEAK_BUT_USABLE"
    else:
        verdict = "BLOCKED"
    payload = {
        "BG_GATED_SELECTOR_DATASET_VERDICT": verdict,
        "verdict": verdict,
        "rows": rows,
        "x": x,
        "y": y,
        "feature_names": feature_names,
        "directional_indices": directional_indices,
        "score_indices": score_indices,
        "metadata_indices": metadata_indices,
        "diagnostic_indices": diagnostic_indices,
        "expert_z_indices": expert_z_indices,
        "category_vocab": vocab,
        "variants": variants,
        "counts": counts,
        "calibration": score_payload.get("calibration", {}),
        "expert_names": EXPERT_NAMES,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(payload, DATASET_PT)
    compact = {k: v for k, v in payload.items() if k not in {"rows", "x", "y"}}
    compact["rows"] = [compact_score_row(row) for row in rows[:200]]
    compact["feature_count"] = len(feature_names)
    compact["row_count"] = len(rows)
    write_json(DATASET_JSON, compact)
    variant_rows = [{"variant": k, "feature_count": len(v)} for k, v in variants.items()]
    lines = ["# Gated/Fusion Selector Dataset", "", f"BG_GATED_SELECTOR_DATASET_VERDICT = {verdict}", "", f"- pairs: `{len(rows)}`", f"- counts: `{counts}`", "", "## Variants", ""]
    lines.extend(md_table(variant_rows, ["variant", "feature_count"]))
    write_md(DATASET_MD, lines)
    print(f"BG_GATED_SELECTOR_DATASET_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


class LogisticFusion(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class TinyMLPFusion(nn.Module):
    def __init__(self, dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class GatedExpertSelector(nn.Module):
    def __init__(self, dim: int, expert_indices: Sequence[int], gate_indices: Sequence[int]) -> None:
        super().__init__()
        self.expert_indices = list(expert_indices)
        self.gate_indices = list(gate_indices)
        gate_dim = max(len(self.gate_indices), 1)
        self.gate = nn.Linear(gate_dim, len(self.expert_indices))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expert_scores = x[:, self.expert_indices]
        gate_input = x[:, self.gate_indices] if self.gate_indices else torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        weights = torch.softmax(self.gate(gate_input), dim=-1)
        return (weights * expert_scores).sum(dim=-1)

    @torch.no_grad()
    def gate_weights(self, x: torch.Tensor) -> torch.Tensor:
        gate_input = x[:, self.gate_indices] if self.gate_indices else torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        return torch.softmax(self.gate(gate_input), dim=-1)


def split_indices(rows: Sequence[dict[str, Any]], split: str, *, include_bridge: bool = True) -> list[int]:
    return [i for i, row in enumerate(rows) if row.get("split") == split and (include_bridge or row.get("pair_type") != "bridge")]


def local_directional(global_indices: Sequence[int], feature_indices: Sequence[int]) -> list[int]:
    pos = {idx: i for i, idx in enumerate(feature_indices)}
    return [pos[idx] for idx in global_indices if idx in pos]


def eval_scores(scores: torch.Tensor, rows: Sequence[dict[str, Any]], indices: Sequence[int]) -> dict[str, Any]:
    if not indices:
        return {"pairwise_accuracy": float("nan"), "pair_count": 0, "balanced_accuracy": float("nan"), "by_pair_type": {}, "by_domain": {}}
    s = scores.detach().cpu()
    correct = (s > 0).to(torch.float32)
    by_type = {}
    by_domain = {}
    for pair_type in sorted({str(rows[i].get("pair_type")) for i in indices}):
        loc = [j for j, i in enumerate(indices) if str(rows[i].get("pair_type")) == pair_type]
        by_type[pair_type] = float(correct[loc].mean().item()) if loc else float("nan")
    for domain in sorted({str(rows[i].get("domain")) for i in indices}):
        loc = [j for j, i in enumerate(indices) if str(rows[i].get("domain")) == domain]
        by_domain[domain] = float(correct[loc].mean().item()) if loc else float("nan")
    finite_type = [v for v in by_type.values() if math.isfinite(v)]
    return {
        "pairwise_accuracy": float(correct.mean().item()),
        "pair_count": len(indices),
        "balanced_accuracy": float(mean(finite_type)) if finite_type else float("nan"),
        "by_pair_type": by_type,
        "by_domain": by_domain,
    }


def trainable_eval(model: nn.Module, x: torch.Tensor, rows: Sequence[dict[str, Any]], indices: Sequence[int], device: torch.device) -> dict[str, Any]:
    if not indices:
        return eval_scores(torch.empty(0), rows, indices)
    with torch.no_grad():
        scores = model(x[indices].to(device)).detach().cpu()
    return eval_scores(scores, rows, indices)


def train_model(
    model: nn.Module,
    *,
    x_train: torch.Tensor,
    x_val: torch.Tensor,
    rows: Sequence[dict[str, Any]],
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    directional: Sequence[int],
    seed: int,
    lr: float,
    device: torch.device,
    epochs: int = 100,
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    random.seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    batch_size = min(64, max(1, x_train.shape[0]))
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_score = -1.0
    best_epoch = -1
    stale = 0
    history = []
    x_train_dev = x_train.to(device)
    x_val_dev = x_val.to(device)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(x_train_dev.shape[0], generator=generator, device=device)
        losses = []
        for start in range(0, x_train_dev.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            xb = x_train_dev[idx].clone()
            target = torch.ones((xb.shape[0],), dtype=torch.float32, device=device)
            swap = torch.rand(xb.shape[0], generator=generator, device=device) < 0.5
            if bool(swap.any().item()) and directional:
                swap_idx = swap.nonzero(as_tuple=False).flatten()
                dir_idx = torch.tensor(list(directional), dtype=torch.long, device=device)
                xb[swap_idx[:, None], dir_idx[None, :]] *= -1.0
                target[swap] = -1.0
            scores = model(xb)
            loss = F.softplus(-target * scores).mean() + 1e-4 * scores.pow(2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        model.eval()
        val_scores = model(x_val_dev).detach().cpu()
        val_metrics = eval_scores(val_scores, rows, val_indices)
        score = float(val_metrics.get("balanced_accuracy", float("nan")))
        history.append({"epoch": epoch, "train_loss": mean(losses), **val_metrics})
        if math.isfinite(score) and score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 12:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_scores = model(x_train.to(device)).detach().cpu()
        val_scores = model(x_val.to(device)).detach().cpu()
    train_metrics = eval_scores(train_scores, rows, train_indices)
    val_metrics = eval_scores(val_scores, rows, val_indices)
    metrics = {
        "seed": seed,
        "lr": lr,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "train_pairwise_accuracy": train_metrics["pairwise_accuracy"],
        "validation_pairwise_accuracy": val_metrics["pairwise_accuracy"],
        "balanced_validation_score": val_metrics["balanced_accuracy"],
        "validation_by_pair_type": val_metrics["by_pair_type"],
        "validation_by_domain": val_metrics["by_domain"],
        "train_by_pair_type": train_metrics["by_pair_type"],
        "history": history,
    }
    model.to("cpu")
    return model, metrics


def score_tensor_for_weights(x: torch.Tensor, feature_names: Sequence[str], weights: dict[str, float]) -> torch.Tensor:
    out = torch.zeros((x.shape[0],), dtype=torch.float32)
    for expert, weight in weights.items():
        name = f"z:{expert}"
        if name in feature_names:
            out += float(weight) * x[:, feature_names.index(name)]
    return out


def balanced_metric_for_scores(scores: torch.Tensor, rows: Sequence[dict[str, Any]], indices: Sequence[int]) -> float:
    return float(eval_scores(scores[indices], rows, indices).get("balanced_accuracy", float("nan")))


def candidate_weight_sets() -> list[dict[str, float]]:
    experts = [
        "old_content_head",
        "old_code_head",
        "mixed_code_reasoning_head",
        "mixed_objective_all_head",
        "v4_hidden_origin",
        "hidden_branch_head",
        "bridge_only_head",
        "universal",
        "generator_v1_selector",
    ]
    out: list[dict[str, float]] = [{expert: 1.0} for expert in experts]
    groups = [
        ("old_content_head", "v4_hidden_origin"),
        ("old_content_head", "hidden_branch_head"),
        ("v4_hidden_origin", "bridge_only_head"),
        ("hidden_branch_head", "bridge_only_head"),
        ("old_content_head", "v4_hidden_origin", "bridge_only_head"),
        ("old_content_head", "hidden_branch_head", "bridge_only_head", "universal"),
        tuple(experts),
    ]
    for group in groups:
        out.append({expert: 1.0 / len(group) for expert in group})
    grid_experts = ["old_content_head", "old_code_head", "v4_hidden_origin", "bridge_only_head", "universal"]
    vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    for weights in itertools.product(vals, repeat=len(grid_experts)):
        if abs(sum(weights) - 1.0) <= 1e-6:
            out.append({expert: weight for expert, weight in zip(grid_experts, weights) if weight > 0})
    dedup = []
    seen = set()
    for weights in out:
        key = tuple(sorted(weights.items()))
        if key not in seen:
            seen.add(key)
            dedup.append(weights)
    return dedup


def route_map_from_val(x: torch.Tensor, rows: Sequence[dict[str, Any]], val_indices: Sequence[int], feature_names: Sequence[str]) -> dict[str, str]:
    route: dict[str, str] = {}
    for pair_type in PAIR_TYPES:
        idx = [i for i in val_indices if rows[i].get("pair_type") == pair_type]
        best_expert = "universal"
        best_acc = -1.0
        for expert in EXPERT_NAMES:
            name = f"z:{expert}"
            if name not in feature_names:
                continue
            scores = x[idx, feature_names.index(name)] if idx else torch.empty(0)
            acc = eval_scores(scores, rows, idx).get("pairwise_accuracy", float("nan"))
            if math.isfinite(float(acc)) and float(acc) > best_acc:
                best_acc = float(acc)
                best_expert = expert
        route[pair_type] = best_expert
    return route


def route_scores(x: torch.Tensor, rows: Sequence[dict[str, Any]], feature_names: Sequence[str], route: dict[str, str]) -> torch.Tensor:
    out = torch.zeros((x.shape[0],), dtype=torch.float32)
    for i, row in enumerate(rows):
        expert = route.get(str(row.get("pair_type")), "universal")
        name = f"z:{expert}"
        out[i] = x[i, feature_names.index(name)] if name in feature_names else 0.0
    return out


def threshold_veto_scores(x: torch.Tensor, feature_names: Sequence[str]) -> torch.Tensor:
    def col(expert: str) -> torch.Tensor:
        name = f"z:{expert}"
        return x[:, feature_names.index(name)] if name in feature_names else torch.zeros((x.shape[0],), dtype=torch.float32)

    old = torch.maximum(col("old_content_head"), torch.maximum(col("old_code_head"), col("mixed_objective_all_head")))
    branch = 0.5 * col("v4_hidden_origin") + 0.5 * col("hidden_branch_head")
    bridge = col("bridge_only_head")
    uni = col("universal")
    base = 0.35 * old + 0.35 * branch + 0.20 * bridge + 0.10 * uni
    veto = (branch > 0.5) & (old < -1.0)
    rescue = (bridge > 1.0) & (branch > -0.25)
    out = base.clone()
    out[veto] -= 0.5
    out[rescue] += 0.35
    return out


def run_training() -> int:
    ensure_root()
    started = time.time()
    dataset = load_pt(DATASET_PT, {}) or {}
    if not dataset:
        payload = {"BG_GATED_SELECTOR_TRAINING_VERDICT": "INSUFFICIENT", "verdict": "INSUFFICIENT", "blocker": "missing gated dataset"}
        write_json(TRAINING_JSON, payload)
        write_md(TRAINING_MD, ["# Gated Selector Training", "", "BG_GATED_SELECTOR_TRAINING_VERDICT = INSUFFICIENT"])
        print("BG_GATED_SELECTOR_TRAINING_VERDICT = INSUFFICIENT", flush=True)
        return 1
    rows = list(dataset["rows"])
    x_all: torch.Tensor = dataset["x"].to(torch.float32)
    feature_names = list(dataset["feature_names"])
    train_idx = split_indices(rows, "train")
    val_idx = split_indices(rows, "val")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = dataset["variants"]
    directional_global = dataset["directional_indices"]
    heads: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []

    fixed_candidates = candidate_weight_sets()
    fixed_rows = []
    best_fixed = None
    for weights in fixed_candidates:
        scores = score_tensor_for_weights(x_all, feature_names, weights)
        val_metrics = eval_scores(scores[val_idx], rows, val_idx)
        row = {"model_type": "fixed_weighted_composite", "weights": weights, **val_metrics}
        fixed_rows.append(row)
        if best_fixed is None or float(row.get("balanced_accuracy", -1.0)) > float(best_fixed.get("balanced_accuracy", -1.0)):
            best_fixed = row
    if best_fixed:
        heads.append(
            {
                "model_type": "fixed_weighted_composite",
                "variant": "fixed_weighted_composite",
                "weights": best_fixed["weights"],
                "metrics": {
                    "validation_pairwise_accuracy": best_fixed["pairwise_accuracy"],
                    "balanced_validation_score": best_fixed["balanced_accuracy"],
                    "validation_by_pair_type": best_fixed["by_pair_type"],
                    "validation_by_domain": best_fixed["by_domain"],
                },
                "label_policy": "actual rewards only; expert scores are input features",
            }
        )
        training_rows.append({"model_type": "fixed_weighted_composite", "weights": best_fixed["weights"], "balanced_val": best_fixed["balanced_accuracy"], "global_val": best_fixed["pairwise_accuracy"]})

    route = route_map_from_val(x_all, rows, val_idx, feature_names)
    router_scores = route_scores(x_all, rows, feature_names, route)
    router_metrics = eval_scores(router_scores[val_idx], rows, val_idx)
    heads.append(
        {
            "model_type": "specialist_oracle_router",
            "variant": "pair_type_aware_diagnostic_router",
            "route_map": route,
            "metrics": {
                "validation_pairwise_accuracy": router_metrics["pairwise_accuracy"],
                "balanced_validation_score": router_metrics["balanced_accuracy"],
                "validation_by_pair_type": router_metrics["by_pair_type"],
                "validation_by_domain": router_metrics["by_domain"],
            },
            "diagnostic_only": True,
        }
    )

    threshold_scores = threshold_veto_scores(x_all, feature_names)
    threshold_metrics = eval_scores(threshold_scores[val_idx], rows, val_idx)
    heads.append(
        {
            "model_type": "thresholded_veto_rescue_policy",
            "variant": "thresholded_veto_rescue_policy",
            "metrics": {
                "validation_pairwise_accuracy": threshold_metrics["pairwise_accuracy"],
                "balanced_validation_score": threshold_metrics["balanced_accuracy"],
                "validation_by_pair_type": threshold_metrics["by_pair_type"],
                "validation_by_domain": threshold_metrics["by_domain"],
            },
            "policy": "old score can veto strong branch score; bridge score can rescue borderline branch score",
        }
    )

    seeds = [42, 43, 44]
    lrs = [1e-4, 3e-4, 1e-3]
    trainable_specs = [
        ("logistic_fusion", "gated_scores_only", LogisticFusion),
        ("logistic_fusion", "gated_scores_plus_metadata", LogisticFusion),
        ("logistic_fusion", "gated_scores_plus_diagnostics", LogisticFusion),
        ("tiny_mlp_fusion", "gated_scores_plus_diagnostics", TinyMLPFusion),
    ]
    for model_type, variant, cls in trainable_specs:
        feat_idx = list(variants[variant])
        if len(train_idx) < 8 or len(val_idx) < 4 or not feat_idx:
            continue
        x_train = x_all[train_idx][:, feat_idx]
        x_val = x_all[val_idx][:, feat_idx]
        directional = local_directional(directional_global, feat_idx)
        for seed in seeds:
            for lr in lrs:
                model = cls(len(feat_idx))
                trained, metrics = train_model(
                    model,
                    x_train=x_train,
                    x_val=x_val,
                    rows=rows,
                    train_indices=train_idx,
                    val_indices=val_idx,
                    directional=directional,
                    seed=seed,
                    lr=lr,
                    device=device,
                )
                head = {
                    "model_type": model_type,
                    "variant": variant,
                    "feature_indices": feat_idx,
                    "feature_names": [feature_names[i] for i in feat_idx],
                    "state_dict": {k: v.detach().cpu().clone() for k, v in trained.state_dict().items()},
                    "metrics": metrics,
                    "label_policy": "actual pairwise outcomes only; no tap-score labels",
                }
                heads.append(head)
                training_rows.append({"model_type": model_type, "variant": variant, "seed": seed, "lr": lr, "balanced_val": metrics["balanced_validation_score"], "global_val": metrics["validation_pairwise_accuracy"]})

    expert_indices = [
        dataset["expert_z_indices"][e]
        for e in (
            "old_content_head",
            "old_code_head",
            "mixed_code_reasoning_head",
            "mixed_objective_all_head",
            "v4_hidden_origin",
            "hidden_branch_head",
            "bridge_only_head",
            "universal",
            "generator_v1_selector",
        )
        if e in dataset["expert_z_indices"]
    ]
    gate_indices = [i for i, name in enumerate(feature_names) if name.startswith("missing:") or name.startswith("cat:") or name.startswith("diag:abs") or name in {"diag:score_count", "diag:score_std", "diag:score_abs_mean", "diag:score_abs_max", "diag:expert_disagreement_rate"}]
    if expert_indices:
        feat_idx = list(range(len(feature_names)))
        x_train = x_all[train_idx]
        x_val = x_all[val_idx]
        directional = local_directional(directional_global, feat_idx)
        for seed in seeds:
            for lr in lrs:
                model = GatedExpertSelector(len(feature_names), expert_indices, gate_indices)
                trained, metrics = train_model(
                    model,
                    x_train=x_train,
                    x_val=x_val,
                    rows=rows,
                    train_indices=train_idx,
                    val_indices=val_idx,
                    directional=directional,
                    seed=seed,
                    lr=lr,
                    device=device,
                )
                heads.append(
                    {
                        "model_type": "gated_expert_selector",
                        "variant": "gated_scores_plus_metadata_diagnostics",
                        "feature_indices": feat_idx,
                        "feature_names": feature_names,
                        "expert_indices": expert_indices,
                        "expert_names": [feature_names[i].split(":", 1)[1] for i in expert_indices],
                        "gate_indices": gate_indices,
                        "gate_feature_names": [feature_names[i] for i in gate_indices],
                        "state_dict": {k: v.detach().cpu().clone() for k, v in trained.state_dict().items()},
                        "metrics": metrics,
                        "label_policy": "actual pairwise outcomes only; expert scores are inputs",
                    }
                )
                training_rows.append({"model_type": "gated_expert_selector", "variant": "gated_scores_plus_metadata_diagnostics", "seed": seed, "lr": lr, "balanced_val": metrics["balanced_validation_score"], "global_val": metrics["validation_pairwise_accuracy"]})

    usable = [h for h in heads if math.isfinite(float((h.get("metrics") or {}).get("balanced_validation_score", float("nan"))))]
    headline_usable = [
        h
        for h in usable
        if h.get("model_type") not in {"tiny_mlp_fusion", "specialist_oracle_router"}
    ]
    best = max(headline_usable or usable, key=lambda h: float(h["metrics"]["balanced_validation_score"])) if usable else None
    best_diagnostic = max(usable, key=lambda h: float(h["metrics"]["balanced_validation_score"])) if usable else None
    best_bal = float(best["metrics"]["balanced_validation_score"]) if best else float("nan")
    best_by_type = best.get("metrics", {}).get("validation_by_pair_type", {}) if best else {}
    if not usable:
        verdict = "INSUFFICIENT"
    elif best_bal >= 0.66 and all(float(best_by_type.get(pt, 0.0)) >= 0.58 for pt in PAIR_TYPES):
        verdict = "READY"
    elif best_bal >= 0.58:
        verdict = "WEAK"
    elif best_by_type.get("old_content", 0.0) > 0.60 and best_by_type.get("hidden_branch", 0.0) < 0.53:
        verdict = "OLD_EXPERT_DOMINATES"
    elif best_by_type.get("hidden_branch", 0.0) > 0.60 and best_by_type.get("old_content", 0.0) < 0.53:
        verdict = "BRANCH_EXPERT_DOMINATES"
    else:
        verdict = "NO_LEARNING"
    payload = {
        "BG_GATED_SELECTOR_TRAINING_VERDICT": verdict,
        "verdict": verdict,
        "device": str(device),
        "heads": heads,
        "best_head": compact_model_head(best) if best else None,
        "best_diagnostic_head": compact_model_head(best_diagnostic) if best_diagnostic else None,
        "training_rows": training_rows,
        "fixed_weight_candidates": fixed_rows,
        "route_map": route,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(payload, HEADS_PT)
    json_payload = {k: v for k, v in payload.items() if k != "heads"}
    json_payload["heads"] = [compact_model_head(h) for h in heads]
    write_json(TRAINING_JSON, json_payload)
    summary_rows = sorted([compact_model_head(h) for h in usable], key=lambda r: -float(r.get("balanced_validation_score", -1.0)))[:20]
    lines = ["# Gated/Fusion Selector Training", "", f"BG_GATED_SELECTOR_TRAINING_VERDICT = {verdict}", "", f"- best_headline_model: `{compact_model_head(best)}`", f"- best_diagnostic_model: `{compact_model_head(best_diagnostic)}`", "", "## Best Models", ""]
    lines.extend(md_table(summary_rows, ["model_type", "variant", "balanced_validation_score", "validation_pairwise_accuracy", "old_val", "hidden_val", "bridge_val"]))
    write_md(TRAINING_MD, lines)
    print(f"BG_GATED_SELECTOR_TRAINING_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "INSUFFICIENT" else 1


def compact_model_head(head: dict[str, Any] | None) -> dict[str, Any] | None:
    if not head:
        return None
    metrics = head.get("metrics", {})
    by_type = metrics.get("validation_by_pair_type", {})
    return {
        "model_type": head.get("model_type"),
        "variant": head.get("variant"),
        "balanced_validation_score": metrics.get("balanced_validation_score"),
        "validation_pairwise_accuracy": metrics.get("validation_pairwise_accuracy"),
        "old_val": by_type.get("old_content"),
        "hidden_val": by_type.get("hidden_branch"),
        "bridge_val": by_type.get("bridge"),
        "weights": head.get("weights"),
        "route_map": head.get("route_map"),
        "seed": metrics.get("seed"),
        "lr": metrics.get("lr"),
    }


def best_model(payload: dict[str, Any], *, include_diagnostic: bool = False) -> dict[str, Any] | None:
    heads = list(payload.get("heads") or [])
    if not include_diagnostic:
        heads = [h for h in heads if h.get("model_type") != "specialist_oracle_router" and h.get("model_type") != "tiny_mlp_fusion"]
    valid = [h for h in heads if math.isfinite(float((h.get("metrics") or {}).get("balanced_validation_score", float("nan"))))]
    if not valid:
        return None
    return max(valid, key=lambda h: float(h["metrics"]["balanced_validation_score"]))


def instantiate_model(head: dict[str, Any], dataset: dict[str, Any]) -> nn.Module | None:
    model_type = head.get("model_type")
    if model_type == "logistic_fusion":
        model = LogisticFusion(len(head.get("feature_indices") or []))
    elif model_type == "tiny_mlp_fusion":
        model = TinyMLPFusion(len(head.get("feature_indices") or []))
    elif model_type == "gated_expert_selector":
        model = GatedExpertSelector(len(dataset["feature_names"]), head.get("expert_indices") or [], head.get("gate_indices") or [])
    else:
        return None
    model.load_state_dict(head["state_dict"])
    model.eval()
    return model


def model_scores(head: dict[str, Any], dataset: dict[str, Any], indices: Sequence[int] | None = None) -> torch.Tensor:
    x: torch.Tensor = dataset["x"].to(torch.float32)
    rows = dataset["rows"]
    feature_names = dataset["feature_names"]
    use_idx = list(range(x.shape[0])) if indices is None else list(indices)
    model_type = head.get("model_type")
    if model_type == "fixed_weighted_composite":
        scores = score_tensor_for_weights(x, feature_names, head.get("weights") or {})
    elif model_type == "specialist_oracle_router":
        scores = route_scores(x, rows, feature_names, head.get("route_map") or {})
    elif model_type == "thresholded_veto_rescue_policy":
        scores = threshold_veto_scores(x, feature_names)
    else:
        model = instantiate_model(head, dataset)
        if model is None:
            scores = torch.zeros((x.shape[0],), dtype=torch.float32)
        else:
            with torch.no_grad():
                if model_type == "gated_expert_selector":
                    scores = model(x).detach().cpu()
                else:
                    feat_idx = list(head.get("feature_indices") or [])
                    scores = model(x[:, feat_idx]).detach().cpu()
    return scores[use_idx]


class RuntimeScorer:
    def __init__(self, model_head: dict[str, Any] | None = None) -> None:
        self.dataset = load_pt(DATASET_PT, {}) or {}
        self.score_payload = load_pt(EXPERT_SCORES_PT, {}) or {}
        self.training = load_pt(HEADS_PT, {}) or {}
        self.model_head = model_head or best_model(self.training) or {}
        self.head_rows = expert_head_rows()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.head_modules = {expert: build_any_head(row, self.device) for expert, row in self.head_rows.items() if row}
        self.model = instantiate_model(self.model_head, self.dataset) if self.model_head else None
        if self.model:
            self.model.eval()

    def close(self) -> None:
        for module in self.head_modules.values():
            module.to("cpu")

    def score_pair(self, pair: dict[str, Any], *, model_head: dict[str, Any] | None = None) -> float:
        head = model_head or self.model_head
        raw = score_pair_with_experts(pair, self.head_rows, self.head_modules, self.device)
        zscores = apply_calibration(raw, self.dataset.get("calibration") or self.score_payload.get("calibration") or {})
        row = {
            "pair_type": str(pair.get("pair_type")),
            "split": str(pair.get("split") or "heldout"),
            "domain": domain_bucket(pair.get("domain")),
            "task_id": str(pair.get("task_id")),
            "group_id": str(pair.get("group_id") or pair.get("branch_group_id")),
            "source_id": str(pair.get("source_id") or "synthetic"),
            "bridge_type": str(pair.get("bridge_type") or "none"),
            "left_origin": str(pair.get("left_origin") or "unknown"),
            "right_origin": str(pair.get("right_origin") or "unknown"),
            "origin_pair": "+".join(sorted([str(pair.get("left_origin") or "unknown"), str(pair.get("right_origin") or "unknown")])),
            "branch_point": str(pair.get("branch_point") or pair.get("branch_point_preferred") or "old_context"),
            "generator_method": str(pair.get("generator_method") or "unknown"),
            "raw_scores": raw,
            "calibrated_scores": zscores,
            "missing": {expert: 0 if math.isfinite(safe_float(raw.get(expert))) else 1 for expert in EXPERT_NAMES},
            "diagnostics": score_diagnostics(zscores, raw),
        }
        feature_names = self.dataset["feature_names"]
        vec = torch.tensor(vector_from_score_row(row, feature_names, self.dataset["category_vocab"]), dtype=torch.float32).view(1, -1)
        if head.get("model_type") == "fixed_weighted_composite":
            return float(score_tensor_for_weights(vec, feature_names, head.get("weights") or {})[0].item())
        if head.get("model_type") == "specialist_oracle_router":
            return float(route_scores(vec, [row], feature_names, head.get("route_map") or {})[0].item())
        if head.get("model_type") == "thresholded_veto_rescue_policy":
            return float(threshold_veto_scores(vec, feature_names)[0].item())
        model = self.model if head is self.model_head else instantiate_model(head, self.dataset)
        if model is None:
            return 0.0
        with torch.no_grad():
            if head.get("model_type") == "gated_expert_selector":
                return float(model(vec).detach().cpu().item())
            feat_idx = list(head.get("feature_indices") or [])
            return float(model(vec[:, feat_idx]).detach().cpu().item())

    def rank_candidates(self, candidates: Sequence[dict[str, Any]], *, pair_type: str, source_id: str, split: str, model_head: dict[str, Any] | None = None) -> list[int]:
        n = len(candidates)
        if n == 0:
            return []
        mat = torch.zeros((n, n), dtype=torch.float32)
        for i in range(n):
            for j in range(i + 1, n):
                pair = synthetic_pair_from_candidates(candidates[i], candidates[j], pair_type=pair_type, source_id=source_id, split=split)
                if not pair.get("features"):
                    score = 0.0
                else:
                    score = self.score_pair(pair, model_head=model_head)
                mat[i, j] = score
                mat[j, i] = -score
        net = mat.mean(dim=1)
        return [i for i, _ in sorted(enumerate(net.tolist()), key=lambda item: (-item[1], item[0]))]


def pairwise_eval_rows(indices: Sequence[int], selectors: dict[str, dict[str, Any]], dataset: dict[str, Any]) -> list[dict[str, Any]]:
    rows = dataset["rows"]
    out = [{"selector": "random", "pairwise_accuracy": 0.5, "pair_count": len(indices), "balanced_accuracy": 0.5}]
    feature_names = dataset["feature_names"]
    for expert in EXPERT_NAMES:
        name = f"raw:{expert}"
        if name in feature_names:
            scores = dataset["x"][indices, feature_names.index(name)]
            metrics = eval_scores(scores, rows, indices)
            out.append({"selector": expert, **metrics})
    for label, head in selectors.items():
        scores = model_scores(head, dataset, indices)
        out.append({"selector": label, **eval_scores(scores, rows, indices)})
    return out


def verdict_from_pairwise(best_acc: float, baseline_acc: float, *, ready: str, small: str, large: str) -> str:
    if not math.isfinite(best_acc):
        return "INSUFFICIENT"
    if best_acc + 0.01 >= baseline_acc:
        return ready
    if best_acc >= max(0.50, baseline_acc - 0.06):
        return small
    return large


def load_selector_heads() -> dict[str, dict[str, Any]]:
    payload = load_pt(HEADS_PT, {}) or {}
    best = best_model(payload) or {}
    best_diag = best_model(payload, include_diagnostic=True) or best
    out = {}
    for head in payload.get("heads") or []:
        mt = head.get("model_type")
        if mt == "fixed_weighted_composite":
            out["fixed_composite"] = head
        elif mt == "thresholded_veto_rescue_policy":
            out["veto_rescue_policy"] = head
        elif mt == "specialist_oracle_router":
            out["specialist_router_diagnostic"] = head
    if best:
        out["gated_selector"] = best
    if best_diag and best_diag is not best:
        out["best_diagnostic_fusion"] = best_diag
    return out


def run_expert_ablation() -> int:
    ensure_root()
    started = time.time()
    dataset = load_pt(DATASET_PT, {}) or {}
    training = load_pt(HEADS_PT, {}) or {}
    rows = list(dataset.get("rows") or [])
    val_idx = split_indices(rows, "val")
    best = best_model(training) or {}
    if not dataset or not best:
        payload = {"BG_GATED_SELECTOR_EXPERT_ABLATION_VERDICT": "INCONCLUSIVE", "verdict": "INCONCLUSIVE", "blocker": "missing trained selector"}
        write_json(ABLATION_JSON, payload)
        write_md(ABLATION_MD, ["# Gated Expert Ablation", "", "BG_GATED_SELECTOR_EXPERT_ABLATION_VERDICT = INCONCLUSIVE"])
        print("BG_GATED_SELECTOR_EXPERT_ABLATION_VERDICT = INCONCLUSIVE", flush=True)
        return 0
    base_scores = model_scores(best, dataset)
    base = eval_scores(base_scores[val_idx], rows, val_idx)
    feature_names = dataset["feature_names"]
    ablation_rows = [{"ablation": "all_experts", **base}]
    for expert in EXPERT_NAMES:
        x_mod = dataset["x"].clone()
        for prefix in ("raw", "z"):
            name = f"{prefix}:{expert}"
            if name in feature_names:
                x_mod[:, feature_names.index(name)] = 0.0
        miss = f"missing:{expert}"
        if miss in feature_names:
            x_mod[:, feature_names.index(miss)] = 1.0
        mod_dataset = dict(dataset)
        mod_dataset["x"] = x_mod
        scores = model_scores(best, mod_dataset)
        metrics = eval_scores(scores[val_idx], rows, val_idx)
        metrics["drop_vs_all"] = float(base["balanced_accuracy"]) - float(metrics["balanced_accuracy"])
        ablation_rows.append({"ablation": f"no_{expert}", **metrics})
    for names, label in (
        (["old_content_head", "v4_hidden_origin"], "old_branch_only"),
        (["old_content_head", "bridge_only_head"], "old_bridge_only"),
        (["v4_hidden_origin", "bridge_only_head"], "branch_bridge_only"),
        (["universal"], "universal_only"),
    ):
        weights = {name: 1.0 / len(names) for name in names}
        scores = score_tensor_for_weights(dataset["x"], feature_names, weights)
        metrics = eval_scores(scores[val_idx], rows, val_idx)
        metrics["drop_vs_all"] = float(base["balanced_accuracy"]) - float(metrics["balanced_accuracy"])
        ablation_rows.append({"ablation": label, **metrics})
    drops = {r["ablation"]: safe_float(r.get("drop_vs_all"), 0.0) for r in ablation_rows}
    if drops.get("no_bridge_only_head", 0.0) > 0.03:
        verdict = "BRIDGE_MATTERS"
    elif drops.get("no_old_content_head", 0.0) > 0.03 and drops.get("no_v4_hidden_origin", 0.0) > 0.03:
        verdict = "COMPLEMENTARY_EXPERTS"
    elif max(drops.values() or [0.0]) < 0.01:
        verdict = "FUSION_UNNECESSARY"
    elif drops.get("no_universal", 0.0) <= 0.0:
        verdict = "UNIVERSAL_REDUNDANT"
    else:
        verdict = "INCONCLUSIVE"
    payload = {
        "BG_GATED_SELECTOR_EXPERT_ABLATION_VERDICT": verdict,
        "verdict": verdict,
        "best_model": compact_model_head(best),
        "base_validation": base,
        "ablation_rows": ablation_rows,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(ABLATION_JSON, payload)
    write_csv(ABLATION_CSV, ablation_rows)
    lines = ["# Gated Selector Expert Ablation", "", f"BG_GATED_SELECTOR_EXPERT_ABLATION_VERDICT = {verdict}", "", "## Ablations", ""]
    lines.extend(md_table(ablation_rows, ["ablation", "balanced_accuracy", "pairwise_accuracy", "drop_vs_all"]))
    write_md(ABLATION_MD, lines)
    print(f"BG_GATED_SELECTOR_EXPERT_ABLATION_VERDICT = {verdict}", flush=True)
    return 0


def old_candidate_groups(split: str = "heldout") -> list[list[dict[str, Any]]]:
    payload = load_pt(OLD_CONTENT_PT, {}) or {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in payload.get("candidate_rows") or []:
        if row.get("split") == split:
            grouped[str(row.get("group_id"))].append(row)
    groups = [vals for vals in grouped.values() if len(vals) >= 2 and len({float(v.get("reward", 0.0)) for v in vals}) >= 2]
    groups.extend(load_old_code_candidate_groups(split))
    return groups


def branch_candidate_groups(split: str = "heldout") -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_branch_candidates():
        if row.get("split") == split:
            grouped[str(row.get("group_id"))].append(row)
    return [vals for vals in grouped.values() if len(vals) >= 2 and len({float(v.get("reward", 0.0)) for v in vals}) >= 2]


def random_group_row(base: dict[str, Any], candidates: Sequence[dict[str, Any]], k: int) -> dict[str, Any]:
    rewards = [float(c.get("reward", 0.0)) for c in candidates]
    oracle = max(rewards)
    oracle_count = sum(1 for r in rewards if r == oracle)
    coverage = min(1.0, k * oracle_count / max(len(candidates), 1))
    return {
        **base,
        "policy": f"random_top{k}",
        "top_k": k,
        "top1_success": oracle_count / max(len(candidates), 1) if k == 1 else coverage,
        "topk_oracle_coverage": coverage,
        "oracle_retention": coverage,
        "reward_mean": mean(rewards),
        "regret": oracle - mean(rewards),
        "false_prune_rate": 1.0 - coverage,
    }


def scalar_ranking(candidates: Sequence[dict[str, Any]], key: str) -> list[int]:
    vals = [(i, safe_float(row.get(key), float("nan"))) for i, row in enumerate(candidates)]
    if any(not math.isfinite(v) for _, v in vals):
        return []
    return [i for i, _ in sorted(vals, key=lambda item: (-item[1], item[0]))]


def rank_group_with_any_head(rows: Sequence[dict[str, Any]], head_row: dict[str, Any] | None, device: torch.device) -> list[int]:
    if not head_row:
        return []
    config = str(head_row.get("config"))
    expected = expected_head_dim(head_row)
    vectors = []
    for row in rows:
        fmap = row.get("features_by_config") or {}
        vec = fmap.get(config)
        if not isinstance(vec, torch.Tensor) or int(vec.numel()) != expected:
            return []
        vectors.append(vec.detach().cpu().to(torch.float32))
    if not vectors:
        return []
    head = build_any_head(head_row, device)
    n = len(vectors)
    mat = torch.zeros((n, n), dtype=torch.float32)
    try:
        with torch.no_grad():
            for i in range(n):
                for j in range(i + 1, n):
                    score = head(vectors[i].view(1, -1).to(device), vectors[j].view(1, -1).to(device))
                    value = float(score.detach().cpu().flatten()[0].item())
                    mat[i, j] = value
                    mat[j, i] = -value
    finally:
        head.to("cpu")
    net = mat.mean(dim=1).tolist()
    return [i for i, _ in sorted(enumerate(net), key=lambda item: (-item[1], item[0]))]


def group_eval(
    groups: Sequence[list[dict[str, Any]]],
    *,
    pair_type: str,
    source_id: str,
    split: str,
    selectors: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    scorer = RuntimeScorer()
    heads = best_heads_from_training()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out: list[dict[str, Any]] = []
    try:
        for vals in groups:
            vals = sorted(vals, key=lambda row: str(row.get("candidate_id")))
            base = {
                "group_id": vals[0].get("group_id"),
                "task_id": vals[0].get("task_id"),
                "domain": vals[0].get("domain"),
                "source_id": source_id,
                "group_size": len(vals),
            }
            out.append({**base, "policy": "first_or_clean_baseline", "top_k": 1, **group_metric(vals, [0], 1)})
            out.append(random_group_row(base, vals, 1))
            out.append(random_group_row(base, vals, 2))
            old_rank = scalar_ranking(vals, "old_frozen_tap_score")
            if old_rank:
                out.append({**base, "policy": "old_frozen_bg", "top_k": 1, **group_metric(vals, old_rank, 1)})
                out.append({**base, "policy": "old_frozen_bg_top2", "top_k": 2, **group_metric(vals, old_rank, 2)})
            for label in ("old_code_head", "mixed_code_reasoning_head", "mixed_objective_all_head"):
                ranking = rank_group_with_any_head(vals, scorer.head_rows.get(label), device)
                if ranking:
                    out.append({**base, "policy": label, "top_k": 1, **group_metric(vals, ranking, 1)})
                    for k in (2, 3):
                        out.append({**base, "policy": f"{label}_top{k}", "top_k": k, **group_metric(vals, ranking, k)})
            for label, head_row in (("universal", heads.get("universal_balanced")), ("v4_hidden_origin", heads.get("v4_hidden_origin")), ("hidden_branch_only", heads.get("hidden_branch_only")), ("bridge_only_head", heads.get("bridge_only"))):
                ranking = rank_group_with_head(vals, head_row, device)
                if ranking:
                    out.append({**base, "policy": label, "top_k": 1, **group_metric(vals, ranking, 1)})
                    for k in (2, 3):
                        out.append({**base, "policy": f"{label}_top{k}", "top_k": k, **group_metric(vals, ranking, k)})
            for label, head in selectors.items():
                ranking = scorer.rank_candidates(vals, pair_type=pair_type, source_id=source_id, split=split, model_head=head)
                if ranking:
                    out.append({**base, "policy": label, "top_k": 1, **group_metric(vals, ranking, 1)})
                    for k in (2, 3):
                        out.append({**base, "policy": f"{label}_top{k}", "top_k": k, **group_metric(vals, ranking, k)})
    finally:
        scorer.close()
    return out


def run_old_context_eval() -> int:
    ensure_root()
    started = time.time()
    dataset = load_pt(DATASET_PT, {}) or {}
    selectors = load_selector_heads()
    rows = dataset.get("rows") or []
    idx = [i for i, r in enumerate(rows) if r.get("split") == "heldout" and r.get("pair_type") == "old_content"]
    pairwise_rows = pairwise_eval_rows(idx, selectors, dataset) if dataset else []
    groups = old_candidate_groups("heldout")
    group_rows_eval = group_eval(groups, pair_type="old_content", source_id="old_context", split="heldout", selectors=selectors) if groups else []
    metrics = aggregate_metric_rows(group_rows_eval, ("policy",))
    gated = next((r for r in pairwise_rows if r.get("selector") == "gated_selector"), {})
    old = max(
        [r for r in pairwise_rows if r.get("selector") in {"old_frozen_bg", "old_content_head", "old_code_head", "mixed_code_reasoning_head", "mixed_objective_all_head"}],
        key=lambda r: safe_float(r.get("pairwise_accuracy"), -1.0),
        default={},
    )
    gated_acc = safe_float(gated.get("pairwise_accuracy"))
    old_acc = safe_float(old.get("pairwise_accuracy"), 0.5)
    verdict = verdict_from_pairwise(gated_acc, old_acc, ready="MATCHES_OR_BEATS_OLD_TAPS", small="SMALL_DEGRADATION", large="LARGE_DEGRADATION") if idx else "INSUFFICIENT"
    payload = {
        "BG_GATED_OLD_CONTEXT_EVAL_VERDICT": verdict,
        "verdict": verdict,
        "pairwise_rows": pairwise_rows,
        "group_eval_rows": group_rows_eval,
        "metrics_by_policy": metrics,
        "domain_breakdown": aggregate_metric_rows(group_rows_eval, ("domain", "policy")),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OLD_CONTEXT_JSON, payload)
    write_csv(OLD_CONTEXT_CSV, pairwise_rows + group_rows_eval)
    lines = ["# Gated Old-Context Evaluation", "", f"BG_GATED_OLD_CONTEXT_EVAL_VERDICT = {verdict}", "", "## Pairwise", ""]
    lines.extend(md_table(pairwise_rows, ["selector", "pairwise_accuracy", "pair_count", "balanced_accuracy"]))
    write_md(OLD_CONTEXT_MD, lines)
    print(f"BG_GATED_OLD_CONTEXT_EVAL_VERDICT = {verdict}", flush=True)
    return 0


def run_hidden_branch_eval() -> int:
    ensure_root()
    started = time.time()
    dataset = load_pt(DATASET_PT, {}) or {}
    selectors = load_selector_heads()
    rows = dataset.get("rows") or []
    idx = [i for i, r in enumerate(rows) if r.get("split") == "heldout" and r.get("pair_type") == "hidden_branch"]
    pairwise_rows = pairwise_eval_rows(idx, selectors, dataset) if dataset else []
    groups = branch_candidate_groups("heldout")
    group_rows_eval = group_eval(groups, pair_type="hidden_branch", source_id="hidden_branch", split="heldout", selectors=selectors) if groups else []
    metrics = aggregate_metric_rows(group_rows_eval, ("policy",))
    gated_top2 = max([safe_float((metrics.get(k) or {}).get("topk_oracle_coverage")) for k in metrics if k.startswith("gated_selector_top2")] or [float("nan")])
    branch_best = max([safe_float((metrics.get(k) or {}).get("topk_oracle_coverage")) for k in ("v4_hidden_origin_top2", "hidden_branch_only_top2")] or [float("nan")])
    gated_pair = next((r for r in pairwise_rows if r.get("selector") == "gated_selector"), {})
    best_branch_pair = max([r for r in pairwise_rows if r.get("selector") in {"v4_hidden_origin", "hidden_branch_head"}], key=lambda r: safe_float(r.get("pairwise_accuracy"), -1.0), default={})
    if not idx and not group_rows_eval:
        verdict = "INSUFFICIENT"
    elif math.isfinite(gated_top2) and math.isfinite(branch_best) and gated_top2 + 0.02 >= branch_best:
        verdict = "MATCHES_OR_BEATS_BRANCH_TAPS"
    elif safe_float(gated_pair.get("pairwise_accuracy")) >= safe_float(best_branch_pair.get("pairwise_accuracy"), 1.0) - 0.06:
        verdict = "SMALL_DEGRADATION"
    elif math.isfinite(gated_top2) and gated_top2 >= 0.65:
        verdict = "TOPK_ONLY"
    else:
        verdict = "LARGE_DEGRADATION"
    payload = {
        "BG_GATED_HIDDEN_BRANCH_EVAL_VERDICT": verdict,
        "verdict": verdict,
        "pairwise_rows": pairwise_rows,
        "group_eval_rows": group_rows_eval,
        "metrics_by_policy": metrics,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(HIDDEN_BRANCH_JSON, payload)
    write_csv(HIDDEN_BRANCH_CSV, pairwise_rows + group_rows_eval)
    lines = ["# Gated Hidden-Branch Evaluation", "", f"BG_GATED_HIDDEN_BRANCH_EVAL_VERDICT = {verdict}", "", "## Pairwise", ""]
    lines.extend(md_table(pairwise_rows, ["selector", "pairwise_accuracy", "pair_count", "balanced_accuracy"]))
    write_md(HIDDEN_BRANCH_MD, lines)
    print(f"BG_GATED_HIDDEN_BRANCH_EVAL_VERDICT = {verdict}", flush=True)
    return 0


def run_bridge_eval() -> int:
    ensure_root()
    started = time.time()
    dataset = load_pt(DATASET_PT, {}) or {}
    selectors = load_selector_heads()
    rows = dataset.get("rows") or []
    idx = [i for i, r in enumerate(rows) if r.get("split") == "heldout" and r.get("pair_type") == "bridge"]
    pairwise_rows = pairwise_eval_rows(idx, selectors, dataset) if dataset else []
    gated = next((r for r in pairwise_rows if r.get("selector") == "gated_selector"), {})
    bridge = next((r for r in pairwise_rows if r.get("selector") == "bridge_only_head"), {})
    universal = next((r for r in pairwise_rows if r.get("selector") == "universal"), {})
    gated_acc = safe_float(gated.get("pairwise_accuracy"))
    bridge_acc = safe_float(bridge.get("pairwise_accuracy"), 0.5)
    universal_acc = safe_float(universal.get("pairwise_accuracy"), 0.5)
    if not idx:
        verdict = "INSUFFICIENT"
    elif gated_acc >= bridge_acc - 0.03 and gated_acc > universal_acc + 0.05:
        verdict = "BRIDGE_FIXED"
    elif gated_acc > universal_acc + 0.03:
        verdict = "BRIDGE_IMPROVED_WEAK"
    elif bridge_acc > gated_acc + 0.05:
        verdict = "BRIDGE_ONLY_BEST"
    else:
        verdict = "BRIDGE_STILL_FAILS"
    payload = {
        "BG_GATED_BRIDGE_EVAL_VERDICT": verdict,
        "verdict": verdict,
        "pairwise_rows": pairwise_rows,
        "bridge_type_breakdown": aggregate_pairwise_by(rows, idx, dataset, selectors, "bridge_type"),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(BRIDGE_JSON, payload)
    write_csv(BRIDGE_CSV, pairwise_rows)
    lines = ["# Gated Bridge Evaluation", "", f"BG_GATED_BRIDGE_EVAL_VERDICT = {verdict}", "", "## Pairwise", ""]
    lines.extend(md_table(pairwise_rows, ["selector", "pairwise_accuracy", "pair_count", "balanced_accuracy"]))
    write_md(BRIDGE_MD, lines)
    print(f"BG_GATED_BRIDGE_EVAL_VERDICT = {verdict}", flush=True)
    return 0


def aggregate_pairwise_by(rows: Sequence[dict[str, Any]], indices: Sequence[int], dataset: dict[str, Any], selectors: dict[str, dict[str, Any]], field: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for value in sorted({str(rows[i].get(field)) for i in indices}):
        idx = [i for i in indices if str(rows[i].get(field)) == value]
        out[value] = pairwise_eval_rows(idx, selectors, dataset)
    return out


DEFAULT_POLICY = {
    "L24": {"K_min": 3, "K_max": 4, "dynamic_margin": 0.25},
    "L36": {"K_min": 2, "K_max": 3, "dynamic_margin": 0.18},
    "L47": {"K_min": 2, "K_max": 2, "dynamic_margin": 0.12},
}


def threshold_sweep_policies() -> list[dict[str, dict[str, float]]]:
    policies = [DEFAULT_POLICY]
    for l24_margin in (0.15, 0.25, 0.35):
        for l36_margin in (0.10, 0.18, 0.25):
            for l47_margin in (0.08, 0.12, 0.18):
                for l24_k in ((2, 4), (3, 4), (3, 5)):
                    for l36_k in ((2, 3), (2, 4)):
                        for l47_k in ((1, 2), (2, 2)):
                            policies.append(
                                {
                                    "L24": {"K_min": l24_k[0], "K_max": l24_k[1], "dynamic_margin": l24_margin},
                                    "L36": {"K_min": l36_k[0], "K_max": l36_k[1], "dynamic_margin": l36_margin},
                                    "L47": {"K_min": l47_k[0], "K_max": l47_k[1], "dynamic_margin": l47_margin},
                                }
                            )
    return policies


def survivors_from_scores(scores: Sequence[float], policy: dict[str, float]) -> list[int]:
    order = [i for i, _ in sorted(enumerate(scores), key=lambda item: (-item[1], item[0]))]
    k_min = int(policy["K_min"])
    k_max = int(policy["K_max"])
    margin = float(policy["dynamic_margin"])
    best = scores[order[0]]
    keep = set(order[: min(k_min, len(order))])
    for i, score in enumerate(scores):
        if best - float(score) <= margin:
            keep.add(i)
    if len(keep) > k_max:
        keep = set(order[:k_max])
    return [i for i in order if i in keep]


def net_scores_for_policy(candidates: Sequence[dict[str, Any]], policy_name: str, scorer: RuntimeScorer, selectors: dict[str, dict[str, Any]], split: str) -> list[float]:
    n = len(candidates)
    if policy_name == "random":
        return [0.0 for _ in candidates]
    old_rank = scalar_ranking(candidates, "old_frozen_tap_score")
    if policy_name == "old_frozen_bg" and old_rank:
        return [float(n - old_rank.index(i)) for i in range(n)]
    head = selectors.get(policy_name)
    if not head:
        head = selectors.get("gated_selector")
    mat = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        for j in range(i + 1, n):
            pair = synthetic_pair_from_candidates(candidates[i], candidates[j], pair_type="hidden_branch", source_id="pruning", split=split)
            s = scorer.score_pair(pair, model_head=head) if pair.get("features") else 0.0
            mat[i, j] = s
            mat[j, i] = -s
    return mat.mean(dim=1).tolist()


def pruning_eval_rows(groups: Sequence[list[dict[str, Any]]], selectors: dict[str, dict[str, Any]], split: str, threshold_policy: dict[str, dict[str, float]], policies: Sequence[str]) -> list[dict[str, Any]]:
    scorer = RuntimeScorer()
    rows: list[dict[str, Any]] = []
    try:
        for vals in groups:
            vals = sorted(vals, key=lambda row: str(row.get("candidate_id")))
            rewards = [float(v.get("reward", 0.0)) for v in vals]
            oracle = max(rewards)
            oracle_indices = {i for i, reward in enumerate(rewards) if reward == oracle}
            for layer in ("L24", "L36", "L47"):
                layer_policy = threshold_policy[layer]
                for policy in policies:
                    scores = net_scores_for_policy(vals, policy, scorer, selectors, split)
                    survivors = survivors_from_scores(scores, layer_policy)
                    kept_oracle = bool(set(survivors) & oracle_indices)
                    rows.append(
                        {
                            "group_id": vals[0].get("group_id"),
                            "task_id": vals[0].get("task_id"),
                            "domain": vals[0].get("domain"),
                            "layer": layer,
                            "policy": policy,
                            "survivors": len(survivors),
                            "group_size": len(vals),
                            "oracle_retention": 1.0 if kept_oracle else 0.0,
                            "false_prune_rate": 0.0 if kept_oracle else 1.0,
                            "compute_saved": 1.0 - len(survivors) / max(len(vals), 1),
                            "average_survivors": len(survivors),
                            "best_score": max(scores) if scores else 0.0,
                            "was_oracle_best_pruned": not kept_oracle,
                            "threshold_policy": json.dumps(layer_policy, sort_keys=True),
                        }
                    )
    finally:
        scorer.close()
    return rows


def choose_threshold_policy(selectors: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    val_groups = branch_candidate_groups("val")
    if not val_groups:
        return DEFAULT_POLICY, []
    policies_to_eval = threshold_sweep_policies()
    score_cache = []
    scorer = RuntimeScorer()
    try:
        for vals in val_groups:
            vals = sorted(vals, key=lambda row: str(row.get("candidate_id")))
            rewards = [float(v.get("reward", 0.0)) for v in vals]
            oracle = max(rewards)
            score_cache.append(
                {
                    "scores": net_scores_for_policy(vals, "gated_selector", scorer, selectors, "val"),
                    "oracle_indices": {i for i, reward in enumerate(rewards) if reward == oracle},
                    "group_size": len(vals),
                }
            )
    finally:
        scorer.close()
    rows = []
    best_policy = DEFAULT_POLICY
    best_key = (-1.0, -1.0, 0.0)
    for idx, policy in enumerate(policies_to_eval):
        retained = []
        survivor_counts = []
        for item in score_cache:
            for layer in ("L24", "L36", "L47"):
                survivors_idx = survivors_from_scores(item["scores"], policy[layer])
                kept = bool(set(survivors_idx) & item["oracle_indices"])
                retained.append(1.0 if kept else 0.0)
                survivor_counts.append(float(len(survivors_idx)))
        retention = float(mean(retained)) if retained else 0.0
        false_prune = 1.0 - retention
        survivors = float(mean(survivor_counts)) if survivor_counts else 99.0
        key = (retention, -false_prune, -survivors)
        rows.append({"policy_index": idx, "oracle_retention": retention, "false_prune_rate": false_prune, "average_survivors": survivors, "threshold_policy": policy})
        if key > best_key:
            best_key = key
            best_policy = policy
    return best_policy, rows


def run_layerwise_pruning() -> int:
    ensure_root()
    started = time.time()
    selectors = load_selector_heads()
    best_threshold_policy, sweep_rows = choose_threshold_policy(selectors)
    groups = branch_candidate_groups("heldout")
    policies = ["old_frozen_bg", "fixed_composite", "gated_selector", "veto_rescue_policy", "specialist_router_diagnostic"]
    rows = pruning_eval_rows(groups, selectors, "heldout", best_threshold_policy, policies) if groups else []
    metrics = aggregate_metric_rows(rows, ("policy",))
    gated = metrics.get("gated_selector") or {}
    fixed = metrics.get("fixed_composite") or {}
    gated_ret = safe_float(gated.get("oracle_retention"))
    gated_false = safe_float(gated.get("false_prune_rate"))
    fixed_ret = safe_float(fixed.get("oracle_retention"), 0.0)
    if not rows:
        verdict = "INSUFFICIENT"
    elif gated_ret >= 0.80 and gated_false <= 0.20:
        verdict = "GATED_PRUNING_READY"
    elif gated_ret >= 0.65:
        verdict = "TOPK_SURVIVAL_ONLY"
    elif fixed_ret > gated_ret + 0.05:
        verdict = "OLD_NEW_COMPOSITE_BEST"
    else:
        verdict = "TOO_MANY_FALSE_PRUNES"
    payload = {
        "BG_GATED_LAYERWISE_PRUNING_VERDICT": verdict,
        "verdict": verdict,
        "selected_threshold_policy": best_threshold_policy,
        "validation_sweep_rows": sweep_rows,
        "rows": rows,
        "metrics_by_policy": metrics,
        "waste_algorithm": "pairwise sigmoid tournament net_score with validation-selected top-k/margin survival",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(PRUNING_JSON, payload)
    write_csv(PRUNING_CSV, rows)
    summary_rows = [{"policy": k, **v} for k, v in metrics.items()]
    lines = ["# Gated Layerwise Pruning Simulation", "", f"BG_GATED_LAYERWISE_PRUNING_VERDICT = {verdict}", "", "## Metrics", ""]
    lines.extend(md_table(summary_rows, ["policy", "n", "oracle_retention", "false_prune_rate", "average_survivors", "compute_saved"]))
    write_md(PRUNING_MD, lines)
    print(f"BG_GATED_LAYERWISE_PRUNING_VERDICT = {verdict}", flush=True)
    return 0


def run_domain_coverage() -> int:
    ensure_root()
    started = time.time()
    dataset = load_pt(DATASET_PT, {}) or {}
    rows = dataset.get("rows") or []
    domains = sorted({str(r.get("domain")) for r in rows})
    domain_rows = []
    for domain in domains:
        vals = [r for r in rows if str(r.get("domain")) == domain]
        domain_rows.append(
            {
                "domain": domain,
                "pairs": len(vals),
                "heldout_pairs": sum(1 for r in vals if r.get("split") == "heldout"),
                "pair_types": dict(Counter(str(r.get("pair_type")) for r in vals)),
                "task_count": len({str(r.get("task_id")) for r in vals}),
            }
        )
    coding_pairs = next((r for r in domain_rows if r["domain"] == "coding"), {"pairs": 0, "heldout_pairs": 0})
    math_pairs = next((r for r in domain_rows if r["domain"] == "math_simple_arithmetic"), {"pairs": 0, "heldout_pairs": 0})
    old_eval = load_json(OLD_CONTEXT_JSON, {}) or {}
    hidden_eval = load_json(HIDDEN_BRANCH_JSON, {}) or {}
    bridge_eval = load_json(BRIDGE_JSON, {}) or {}
    if int(coding_pairs.get("pairs", 0)) == 0:
        verdict = "CODING_DATA_MISSING"
        coding_status = "NO_PRIMARY_SIGNAL"
    elif old_eval.get("verdict") in {"MATCHES_OR_BEATS_OLD_TAPS", "SMALL_DEGRADATION"} and hidden_eval.get("verdict") in {"MATCHES_OR_BEATS_BRANCH_TAPS", "TOPK_ONLY", "SMALL_DEGRADATION"}:
        verdict = "MULTIDOMAIN_READY"
        coding_status = "USABLE"
    elif int(math_pairs.get("pairs", 0)) == 0:
        verdict = "MATH_DATA_MISSING"
        coding_status = "NO_PRIMARY_SIGNAL" if int(coding_pairs.get("pairs", 0)) == 0 else "USABLE"
    else:
        verdict = "REASONING_SCIENCE_ONLY"
        coding_status = "NO_PRIMARY_SIGNAL" if int(coding_pairs.get("pairs", 0)) == 0 else "DIAGNOSTIC"
    payload = {
        "BG_GATED_DOMAIN_COVERAGE_VERDICT": verdict,
        "verdict": verdict,
        "CODING_DATA_STATUS": coding_status,
        "domain_rows": domain_rows,
        "old_context_verdict": old_eval.get("verdict"),
        "hidden_branch_verdict": hidden_eval.get("verdict"),
        "bridge_verdict": bridge_eval.get("verdict"),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(DOMAIN_JSON, payload)
    lines = ["# Gated Selector Domain Coverage", "", f"BG_GATED_DOMAIN_COVERAGE_VERDICT = {verdict}", "", f"CODING_DATA_STATUS = {coding_status}", "", "## Domains", ""]
    lines.extend(md_table(domain_rows, ["domain", "pairs", "heldout_pairs", "task_count", "pair_types"]))
    write_md(DOMAIN_MD, lines)
    print(f"BG_GATED_DOMAIN_COVERAGE_VERDICT = {verdict}", flush=True)
    return 0


def calibration_bins(scores: torch.Tensor, rows: Sequence[dict[str, Any]], indices: Sequence[int], bins: int = 10) -> dict[str, Any]:
    if not indices:
        return {"ece": float("nan"), "bins": []}
    probs = torch.sigmoid(scores).detach().cpu().tolist()
    correct = [1.0 if s > 0 else 0.0 for s in scores.detach().cpu().tolist()]
    bin_rows = []
    ece = 0.0
    for b in range(bins):
        lo = b / bins
        hi = (b + 1) / bins
        loc = [i for i, p in enumerate(probs) if (lo <= p < hi or (b == bins - 1 and p <= hi))]
        if not loc:
            continue
        conf = mean(probs[i] for i in loc)
        acc = mean(correct[i] for i in loc)
        weight = len(loc) / max(len(probs), 1)
        ece += weight * abs(acc - conf)
        bin_rows.append({"bin": b, "lo": lo, "hi": hi, "count": len(loc), "confidence": conf, "accuracy": acc})
    return {"ece": ece, "bins": bin_rows}


def run_calibration_ood() -> int:
    ensure_root()
    started = time.time()
    dataset = load_pt(DATASET_PT, {}) or {}
    training = load_pt(HEADS_PT, {}) or {}
    best = best_model(training) or {}
    rows = dataset.get("rows") or []
    heldout_idx = split_indices(rows, "heldout")
    if not dataset or not best or not heldout_idx:
        payload = {"BG_GATED_CALIBRATION_OOD_VERDICT": "INSUFFICIENT", "verdict": "INSUFFICIENT"}
        write_json(CALIBRATION_JSON, payload)
        write_md(CALIBRATION_MD, ["# Gated Calibration/OOD", "", "BG_GATED_CALIBRATION_OOD_VERDICT = INSUFFICIENT"])
        print("BG_GATED_CALIBRATION_OOD_VERDICT = INSUFFICIENT", flush=True)
        return 0
    base_scores = model_scores(best, dataset, heldout_idx)
    base_metrics = eval_scores(base_scores, rows, heldout_idx)
    cal = calibration_bins(base_scores, rows, heldout_idx)
    feature_names = dataset["feature_names"]
    stress_rows = []
    for expert in EXPERT_NAMES:
        x_mod = dataset["x"].clone()
        for prefix in ("raw", "z"):
            name = f"{prefix}:{expert}"
            if name in feature_names:
                x_mod[:, feature_names.index(name)] = 0.0
        miss = f"missing:{expert}"
        if miss in feature_names:
            x_mod[:, feature_names.index(miss)] = 1.0
        mod_dataset = dict(dataset)
        mod_dataset["x"] = x_mod
        scores = model_scores(best, mod_dataset, heldout_idx)
        metrics = eval_scores(scores, rows, heldout_idx)
        stress_rows.append({"stress": f"missing_{expert}", "balanced_accuracy": metrics["balanced_accuracy"], "drop": base_metrics["balanced_accuracy"] - metrics["balanced_accuracy"]})
    max_drop = max([safe_float(r.get("drop"), 0.0) for r in stress_rows] or [0.0])
    ece = safe_float(cal.get("ece"))
    if math.isfinite(ece) and ece <= 0.12 and max_drop <= 0.08:
        verdict = "CALIBRATED_AND_ROBUST"
    elif max_drop > 0.12:
        verdict = "EXPERT_MISSING_FRAGILE"
    elif math.isfinite(ece) and ece > 0.18:
        verdict = "CALIBRATION_WEAK"
    else:
        verdict = "OOD_FRAGILE"
    payload = {
        "BG_GATED_CALIBRATION_OOD_VERDICT": verdict,
        "verdict": verdict,
        "base_metrics": base_metrics,
        "calibration": cal,
        "missing_expert_stress": stress_rows,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(CALIBRATION_JSON, payload)
    lines = ["# Gated Selector Calibration/OOD", "", f"BG_GATED_CALIBRATION_OOD_VERDICT = {verdict}", "", f"- ECE: `{rate(ece)}`", f"- max missing-expert drop: `{rate(max_drop)}`", "", "## Missing Expert Stress", ""]
    lines.extend(md_table(stress_rows, ["stress", "balanced_accuracy", "drop"]))
    write_md(CALIBRATION_MD, lines)
    print(f"BG_GATED_CALIBRATION_OOD_VERDICT = {verdict}", flush=True)
    return 0


def run_geometry() -> int:
    ensure_root()
    started = time.time()
    dataset = load_pt(DATASET_PT, {}) or {}
    training = load_pt(HEADS_PT, {}) or {}
    rows = dataset.get("rows") or []
    x = dataset.get("x")
    feature_names = dataset.get("feature_names") or []
    corr_rows = []
    if isinstance(x, torch.Tensor):
        for a, b in itertools.combinations(EXPERT_NAMES, 2):
            na, nb = f"z:{a}", f"z:{b}"
            if na in feature_names and nb in feature_names:
                va = x[:, feature_names.index(na)].to(torch.float32)
                vb = x[:, feature_names.index(nb)].to(torch.float32)
                if va.numel() > 1 and float(va.std(unbiased=False)) > 1e-8 and float(vb.std(unbiased=False)) > 1e-8:
                    corr = float(torch.corrcoef(torch.stack([va, vb]))[0, 1].item())
                    corr_rows.append({"expert_a": a, "expert_b": b, "correlation": corr})
    best = best_model(training) or {}
    model_type = best.get("model_type")
    gate_rows = []
    if model_type == "gated_expert_selector" and isinstance(x, torch.Tensor):
        model = instantiate_model(best, dataset)
        if isinstance(model, GatedExpertSelector):
            with torch.no_grad():
                weights = model.gate_weights(x).detach().cpu()
            expert_names = best.get("expert_names") or []
            for pair_type in PAIR_TYPES:
                idx = [i for i, row in enumerate(rows) if row.get("pair_type") == pair_type]
                if not idx:
                    continue
                avg = weights[idx].mean(dim=0).tolist()
                for expert, value in zip(expert_names, avg):
                    gate_rows.append({"pair_type": pair_type, "expert": expert, "average_gate_weight": value})
    max_corr = max([abs(safe_float(r.get("correlation"), 0.0)) for r in corr_rows] or [0.0])
    bridge_used = any(r.get("expert") == "bridge_only_head" and safe_float(r.get("average_gate_weight"), 0.0) > 0.10 for r in gate_rows)
    if gate_rows and bridge_used:
        verdict = "BRIDGE_GEOMETRY_USED"
    elif gate_rows:
        verdict = "INTERPRETABLE_MIXTURE"
    elif max_corr >= 0.90:
        verdict = "OLD_GEOMETRY_DOMINATES"
    else:
        verdict = "INCONCLUSIVE"
    payload = {
        "BG_GATED_GEOMETRY_VERDICT": verdict,
        "verdict": verdict,
        "expert_correlations": corr_rows,
        "gate_weight_rows": gate_rows,
        "best_model": compact_model_head(best),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(GEOMETRY_JSON, payload)
    lines = ["# Gated Selector Geometry/Interpretability", "", f"BG_GATED_GEOMETRY_VERDICT = {verdict}", "", "## Gate Weights", ""]
    lines.extend(md_table(gate_rows[:40], ["pair_type", "expert", "average_gate_weight"]))
    write_md(GEOMETRY_MD, lines)
    print(f"BG_GATED_GEOMETRY_VERDICT = {verdict}", flush=True)
    return 0


def run_old_tap_replacement_probe() -> int:
    ensure_root()
    started = time.time()
    old_eval = load_json(OLD_CONTEXT_JSON, {}) or {}
    pairwise_rows = list(old_eval.get("pairwise_rows") or [])
    group_rows_eval = list(old_eval.get("group_eval_rows") or [])
    gated = next((r for r in pairwise_rows if r.get("selector") == "gated_selector"), {})
    old = max(
        [r for r in pairwise_rows if r.get("selector") in {"old_frozen_bg", "old_content_head", "old_code_head", "mixed_code_reasoning_head", "mixed_objective_all_head"}],
        key=lambda r: safe_float(r.get("pairwise_accuracy"), -1.0),
        default={},
    )
    gated_acc = safe_float(gated.get("pairwise_accuracy"))
    old_acc = safe_float(old.get("pairwise_accuracy"), 0.5)
    if not pairwise_rows:
        verdict = "INSUFFICIENT"
    elif gated_acc >= old_acc + 0.03:
        verdict = "SAFE_REPLACEMENT_CANDIDATE"
    elif gated_acc + 0.03 >= old_acc:
        verdict = "PARTIAL_REPLACEMENT_ONLY"
    else:
        verdict = "NOT_A_REPLACEMENT"
    payload = {
        "BG_GATED_AS_OLD_TAP_REPLACEMENT_VERDICT": verdict,
        "verdict": verdict,
        "pairwise_rows": pairwise_rows,
        "group_eval_rows": group_rows_eval,
        "interpretation": "diagnostic only; no production routing changed",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OLD_REPLACEMENT_JSON, payload)
    write_csv(OLD_REPLACEMENT_CSV, pairwise_rows + group_rows_eval)
    lines = ["# Gated Selector as Old Tap Replacement Probe", "", f"BG_GATED_AS_OLD_TAP_REPLACEMENT_VERDICT = {verdict}", "", "This probe is diagnostic only and does not change production routing.", "", "## Pairwise", ""]
    lines.extend(md_table(pairwise_rows, ["selector", "pairwise_accuracy", "pair_count", "balanced_accuracy"]))
    write_md(OLD_REPLACEMENT_MD, lines)
    print(f"BG_GATED_AS_OLD_TAP_REPLACEMENT_VERDICT = {verdict}", flush=True)
    return 0


def stage_payloads() -> dict[str, dict[str, Any]]:
    return {
        "inventory": load_json(INVENTORY_JSON, {}) or {},
        "expert_scores": load_json(EXPERT_SCORES_JSON, {}) or {},
        "dataset": load_json(DATASET_JSON, {}) or {},
        "training": load_json(TRAINING_JSON, {}) or {},
        "ablation": load_json(ABLATION_JSON, {}) or {},
        "old_context": load_json(OLD_CONTEXT_JSON, {}) or {},
        "hidden_branch": load_json(HIDDEN_BRANCH_JSON, {}) or {},
        "bridge": load_json(BRIDGE_JSON, {}) or {},
        "pruning": load_json(PRUNING_JSON, {}) or {},
        "domain": load_json(DOMAIN_JSON, {}) or {},
        "calibration": load_json(CALIBRATION_JSON, {}) or {},
        "geometry": load_json(GEOMETRY_JSON, {}) or {},
        "old_replacement": load_json(OLD_REPLACEMENT_JSON, {}) or {},
    }


def verdict(payload: dict[str, Any], key: str, default: str = "INSUFFICIENT") -> str:
    return str(payload.get(key) or payload.get("verdict") or default)


def gated_status(data: dict[str, dict[str, Any]]) -> str:
    train = verdict(data["training"], "BG_GATED_SELECTOR_TRAINING_VERDICT")
    old = verdict(data["old_context"], "BG_GATED_OLD_CONTEXT_EVAL_VERDICT")
    hidden = verdict(data["hidden_branch"], "BG_GATED_HIDDEN_BRANCH_EVAL_VERDICT")
    bridge = verdict(data["bridge"], "BG_GATED_BRIDGE_EVAL_VERDICT")
    pruning = verdict(data["pruning"], "BG_GATED_LAYERWISE_PRUNING_VERDICT")
    domain = verdict(data["domain"], "BG_GATED_DOMAIN_COVERAGE_VERDICT")
    cal = verdict(data["calibration"], "BG_GATED_CALIBRATION_OOD_VERDICT")
    replacement = verdict(data["old_replacement"], "BG_GATED_AS_OLD_TAP_REPLACEMENT_VERDICT")
    if train == "READY" and old == "MATCHES_OR_BEATS_OLD_TAPS" and hidden == "MATCHES_OR_BEATS_BRANCH_TAPS" and bridge == "BRIDGE_FIXED" and pruning == "GATED_PRUNING_READY" and cal == "CALIBRATED_AND_ROBUST":
        return "GATED_READY"
    if pruning == "OLD_NEW_COMPOSITE_BEST":
        return "OLD_NEW_COMPOSITE_SUFFICIENT"
    if pruning == "TOPK_SURVIVAL_ONLY" and old in {"MATCHES_OR_BEATS_OLD_TAPS", "SMALL_DEGRADATION"}:
        return "TOPK_GATED_WEAK"
    if bridge in {"BRIDGE_STILL_FAILS", "BRIDGE_ONLY_BEST"}:
        return "BRIDGE_STILL_UNSOLVED"
    if pruning == "TOO_MANY_FALSE_PRUNES":
        return "TOO_MANY_FALSE_PRUNES"
    if replacement in {"NOT_A_REPLACEMENT"}:
        return "OLD_TAP_REMAINS_CONTENT_AUTHORITY"
    if domain in {"CODING_DATA_MISSING", "MATH_DATA_MISSING", "DATA_LIMITED"}:
        return "DATA_LIMITED"
    if train in {"READY", "WEAK"}:
        return "FUSION_IMPROVES_BUT_NOT_READY"
    return "NOT_READY"


def top_lines(data: dict[str, dict[str, Any]], status: str) -> list[str]:
    return [
        f"BG_GATED_SELECTOR_INVENTORY_VERDICT = {verdict(data['inventory'], 'BG_GATED_SELECTOR_INVENTORY_VERDICT')}",
        f"BG_GATED_SELECTOR_EXPERT_SCORES_VERDICT = {verdict(data['expert_scores'], 'BG_GATED_SELECTOR_EXPERT_SCORES_VERDICT')}",
        f"BG_GATED_SELECTOR_DATASET_VERDICT = {verdict(data['dataset'], 'BG_GATED_SELECTOR_DATASET_VERDICT')}",
        f"BG_GATED_SELECTOR_TRAINING_VERDICT = {verdict(data['training'], 'BG_GATED_SELECTOR_TRAINING_VERDICT')}",
        f"BG_GATED_SELECTOR_EXPERT_ABLATION_VERDICT = {verdict(data['ablation'], 'BG_GATED_SELECTOR_EXPERT_ABLATION_VERDICT')}",
        f"BG_GATED_OLD_CONTEXT_EVAL_VERDICT = {verdict(data['old_context'], 'BG_GATED_OLD_CONTEXT_EVAL_VERDICT')}",
        f"BG_GATED_HIDDEN_BRANCH_EVAL_VERDICT = {verdict(data['hidden_branch'], 'BG_GATED_HIDDEN_BRANCH_EVAL_VERDICT')}",
        f"BG_GATED_BRIDGE_EVAL_VERDICT = {verdict(data['bridge'], 'BG_GATED_BRIDGE_EVAL_VERDICT')}",
        f"BG_GATED_LAYERWISE_PRUNING_VERDICT = {verdict(data['pruning'], 'BG_GATED_LAYERWISE_PRUNING_VERDICT')}",
        f"BG_GATED_DOMAIN_COVERAGE_VERDICT = {verdict(data['domain'], 'BG_GATED_DOMAIN_COVERAGE_VERDICT')}",
        f"BG_GATED_CALIBRATION_OOD_VERDICT = {verdict(data['calibration'], 'BG_GATED_CALIBRATION_OOD_VERDICT')}",
        f"BG_GATED_GEOMETRY_VERDICT = {verdict(data['geometry'], 'BG_GATED_GEOMETRY_VERDICT')}",
        f"BG_GATED_AS_OLD_TAP_REPLACEMENT_VERDICT = {verdict(data['old_replacement'], 'BG_GATED_AS_OLD_TAP_REPLACEMENT_VERDICT')}",
        f"GATED_BRANCH_CONTENT_SELECTOR_STATUS = {status}",
    ]


def recommendation(status: str) -> str:
    if status == "GATED_READY":
        return "Use the gated selector in a selection-only Phase 2 prototype with L24/L36/L47 top-k survival; do not claim action steering."
    if status == "TOPK_GATED_WEAK":
        return "Use gated selection only for top2/top3 survival. Avoid hard top1 pruning."
    if status == "OLD_NEW_COMPOSITE_SUFFICIENT":
        return "Prefer the simpler old+branch+bridge composite over the learned gate for now; keep top-k survival and do not change production routing."
    if status == "BRIDGE_STILL_UNSOLVED":
        return "Keep a dedicated bridge expert and collect better bridge data before treating fusion as ready."
    if status == "TOO_MANY_FALSE_PRUNES":
        return "Do not use the selector for pruning yet; improve thresholds/top-k survival or branch scoring."
    if status == "OLD_TAP_REMAINS_CONTENT_AUTHORITY":
        return "Keep old taps as the content authority and use gated/branch signals only as branch-survival diagnostics."
    if status == "DATA_LIMITED":
        return "Collect missing coding/math/bridge support before broad readiness claims."
    return "Treat fusion as diagnostic; do not change production routing."


def docs_section(data: dict[str, dict[str, Any]], status: str) -> str:
    return "\n".join(
        [
            "## Gated branch-content selector v1 (2026-05-18)",
            "",
            "Gated/Fusion Branch-Content Selector v1 tested whether old content taps, hidden-origin branch taps, bridge heads, universal heads, and readiness diagnostics can be combined without collapsing all roles into one linear universal tap.",
            "",
            *top_lines(data, status),
            "",
            f"- recommendation: `{recommendation(status)}`",
            "- no Ouro weights, tokenizer files, checkpoints, old taps, tap registries, wrapper/local-agent routing, or production routing were modified.",
            "- expert/tap scores were used only as input features, not as labels.",
            "",
        ]
    )


def append_if_missing(path: Path, text: str, marker: str) -> None:
    if not path.exists():
        return
    current = path.read_text(encoding="utf-8")
    section = text.strip()
    if marker in current:
        start = current.index(marker)
        next_start = current.find("\n## ", start + len(marker))
        if next_start == -1:
            updated = current[:start].rstrip() + "\n\n" + section + "\n"
        else:
            updated = current[:start].rstrip() + "\n\n" + section + "\n" + current[next_start:]
        path.write_text(updated, encoding="utf-8")
        return
    path.write_text(current.rstrip() + "\n\n" + section + "\n", encoding="utf-8")


def update_nav_docs(status: str) -> None:
    evaluator_readme = PROJECT_ROOT / "shared/docs/evaluator/README.md"
    docs_readme = PROJECT_ROOT / "shared/docs/README.md"
    if evaluator_readme.exists():
        text = evaluator_readme.read_text(encoding="utf-8")
        if "`gated-branch-content-selector-v1.md`" not in text:
            text = text.replace(
                "- `universal-branch-content-taps-v1.md` is the latest universal-vs-composite selector decision.",
                "- `universal-branch-content-taps-v1.md` is the universal-vs-composite selector decision.\n- `gated-branch-content-selector-v1.md` is the latest gated/composite selector decision.",
            )
            text = text.replace(
                "- `universal-branch-content-taps-v1.md`\n",
                "- `universal-branch-content-taps-v1.md`\n- `gated-branch-content-selector-v1.md`\n",
            )
            text = text.replace(
                "- Universal branch-content taps are not ready as a single replacement head; current status is `FUSION_NEEDED`.",
                f"- Universal branch-content taps are not ready as a single replacement head; current status is `FUSION_NEEDED`.\n- Gated branch-content selector status is `{status}`.",
            )
            text = text.replace(
                "- `opi/taps/probes/bg_universal_branch_content_taps_v1_2026-05-18/summary.md`\n",
                "- `opi/taps/probes/bg_universal_branch_content_taps_v1_2026-05-18/summary.md`\n- `opi/taps/probes/bg_gated_branch_content_selector_v1_2026-05-18/summary.md`\n",
            )
            evaluator_readme.write_text(text, encoding="utf-8")
    if docs_readme.exists():
        text = docs_readme.read_text(encoding="utf-8")
        if "`evaluator/gated-branch-content-selector-v1.md`" not in text:
            text = text.replace(
                "- `evaluator/universal-branch-content-taps-v1.md`\n",
                "- `evaluator/universal-branch-content-taps-v1.md`\n- `evaluator/gated-branch-content-selector-v1.md`\n",
            )
            docs_readme.write_text(text, encoding="utf-8")


def append_docs(section: str) -> None:
    marker = "## Gated branch-content selector v1 (2026-05-18)"
    targets = [
        PROJECT_ROOT / "docs/evaluator/current_state.md",
        PROJECT_ROOT / "shared/docs/evaluator/current-state.md",
        PROJECT_ROOT / "docs/evaluator/domain_transfer_ledger.md",
        PROJECT_ROOT / "shared/docs/evaluator/domain-transfer-ledger.md",
        PROJECT_ROOT / "docs/evaluator/bg_universal_branch_content_taps_v1.md",
        PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_branch_generator_v1.md",
        PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_quota_v4.md",
        PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_split_salvage.md",
        PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_taps.md",
        PROJECT_ROOT / "docs/evaluator/bg_hidden_state_branch_generation.md",
        PROJECT_ROOT / "docs/evaluator/bg_steering_consolidation_2026-05-18.md",
        PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-18_v8.1_routing_locked.md",
    ]
    for target in targets:
        append_if_missing(target, section, marker)


def run_synthesis() -> int:
    ensure_root()
    started = time.time()
    data = stage_payloads()
    status = gated_status(data)
    rec = recommendation(status)
    payload = {
        "top_lines": top_lines(data, status),
        "stage_payloads": data,
        "GATED_BRANCH_CONTENT_SELECTOR_STATUS": status,
        "recommended_next": rec,
        "files_created": {
            "inventory": rel(INVENTORY_JSON),
            "expert_scores": rel(EXPERT_SCORES_PT),
            "dataset": rel(DATASET_PT),
            "heads": rel(HEADS_PT),
            "ablation": rel(ABLATION_JSON),
            "old_context_eval": rel(OLD_CONTEXT_JSON),
            "hidden_branch_eval": rel(HIDDEN_BRANCH_JSON),
            "bridge_eval": rel(BRIDGE_JSON),
            "layerwise_pruning": rel(PRUNING_JSON),
            "domain_coverage": rel(DOMAIN_JSON),
            "calibration_ood": rel(CALIBRATION_JSON),
            "geometry": rel(GEOMETRY_JSON),
            "old_tap_replacement": rel(OLD_REPLACEMENT_JSON),
            "summary": rel(SUMMARY_JSON),
            "analysis": rel(ANALYSIS_JSON),
            "docs": rel(DOC_MD),
            "simple_docs": rel(SIMPLE_DOC_MD),
        },
        "commands_run": [f"venv/bin/python -u utilities/tests/manual/{script}" for script in GATED_SCRIPTS],
        "blockers": [v.get("blocker") for v in data.values() if isinstance(v, dict) and v.get("blocker")],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(SUMMARY_JSON, payload)
    write_json(ANALYSIS_JSON, payload)
    summary_lines = ["# Gated Branch-Content Selector V1 Summary", "", *top_lines(data, status), "", f"Recommended next: {rec}", "", "## Files Created", ""]
    summary_lines.extend([f"- {name}: `{path}`" for name, path in payload["files_created"].items()])
    write_md(SUMMARY_MD, summary_lines)
    write_md(ANALYSIS_MD, summary_lines + ["", "## Stage Payloads", "", "See `summary.json` / `analysis.json` for compact machine-readable payloads."])
    doc_lines = [
        "# Gated Branch-Content Selector V1",
        "",
        "This experiment tested a gated/composite branch-content selector after the universal linear tap result showed `FUSION_NEEDED`.",
        "",
        *top_lines(data, status),
        "",
        "## Result",
        "",
        rec,
        "",
        "## Architecture",
        "",
        "The selector combines cached old-content, hidden-origin, bridge, universal, and generator selector scores with metadata and readiness diagnostics. Expert scores are inputs only; labels remain actual correctness, reward, verifier, preference, or deterministic branch reward.",
        "",
        "## Safety",
        "",
        "- No Ouro training or checkpoint/tokenizer/model edits.",
        "- No old tap registry update.",
        "- No wrapper/local-agent or actual Hunter-Seeker execution.",
        "- No production routing change or action-steering claim.",
    ]
    write_md(DOC_MD, doc_lines)
    write_md(SIMPLE_DOC_MD, doc_lines)
    section = docs_section(data, status)
    append_docs(section)
    update_nav_docs(status)
    print(f"GATED_BRANCH_CONTENT_SELECTOR_STATUS = {status}", flush=True)
    return 0
