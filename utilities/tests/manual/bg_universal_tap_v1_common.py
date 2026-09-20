"""Shared helpers for Universal Branch-Content Taps v1.

This manual experiment trains only new tiny pairwise heads under a new artifact
root. It reads cached old-context and hidden-origin artifacts, does not train
Ouro, does not mutate existing BG taps/registries, and does not run wrapper or
Hunter-Seeker code paths.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Sequence

import torch

from bg_branch_generator_v1_common import (
    BGV1_ROOT,
    SELECTOR_DATASET_PT as BGV1_SELECTOR_DATASET_PT,
    SELECTOR_HEADS_PT as BGV1_SELECTOR_HEADS_PT,
    candidate_pair_stats,
    config_dim,
    config_vector_from_row,
    group_is_behaviorally_diverse_v2,
    group_is_reward_diverse_v2,
    group_rows,
    load_generator_rows,
    md_table,
    primary_safe_generator_row,
    rate,
    stable_v2_row,
)
from bg_hidden_origin_quota_v4_common import (
    CONFIGS,
    HEADS_V4_PT,
    PROBE_ROOT,
    V4_ROOT,
    best_primary_head,
    compact_head_id,
    load_pt,
    rel,
    row_reward,
    tensor_stats,
    write_csv,
    write_json,
    write_md,
)
from bg_hidden_origin_tap_common import HIDDEN_DIM, PROJECT_ROOT, cosine, pairwise_accuracy_from_pairs, ranking_from_matrix, score_matrix
from evaluate_bg_hidden_origin_taps import build_head
from train_bg_hidden_origin_taps import ARCHITECTURES, compact_head, flip_diagnostics, pairs_for_config, train_one


UNIVERSAL_ROOT = PROBE_ROOT / "bg_universal_branch_content_taps_v1_2026-05-18"

INVENTORY_JSON = UNIVERSAL_ROOT / "inventory.json"
INVENTORY_MD = UNIVERSAL_ROOT / "inventory.md"
SOURCE_INVENTORY_CSV = UNIVERSAL_ROOT / "source_inventory.csv"

OLD_CONTENT_PT = UNIVERSAL_ROOT / "old_content_dataset.pt"
OLD_CONTENT_JSON = UNIVERSAL_ROOT / "old_content_dataset.json"
OLD_CONTENT_MD = UNIVERSAL_ROOT / "old_content_dataset.md"

HIDDEN_BRANCH_PT = UNIVERSAL_ROOT / "hidden_branch_dataset.pt"
HIDDEN_BRANCH_JSON = UNIVERSAL_ROOT / "hidden_branch_dataset.json"
HIDDEN_BRANCH_MD = UNIVERSAL_ROOT / "hidden_branch_dataset.md"

BRIDGE_PT = UNIVERSAL_ROOT / "bridge_dataset.pt"
BRIDGE_JSON = UNIVERSAL_ROOT / "bridge_dataset.json"
BRIDGE_MD = UNIVERSAL_ROOT / "bridge_dataset.md"

DATA_EXPANSION_JSON = UNIVERSAL_ROOT / "data_expansion.json"
DATA_EXPANSION_MD = UNIVERSAL_ROOT / "data_expansion.md"
EXPANDED_ROWS_JSONL = UNIVERSAL_ROOT / "expanded_rows.jsonl"

UNIVERSAL_DATASET_PT = UNIVERSAL_ROOT / "universal_tap_dataset.pt"
UNIVERSAL_DATASET_JSON = UNIVERSAL_ROOT / "universal_tap_dataset.json"
UNIVERSAL_DATASET_MD = UNIVERSAL_ROOT / "universal_tap_dataset.md"

UNIVERSAL_HEADS_PT = UNIVERSAL_ROOT / "universal_branch_content_tap_heads.pt"
TRAINING_JSON = UNIVERSAL_ROOT / "training_log.json"
TRAINING_MD = UNIVERSAL_ROOT / "training_report.md"

OLD_CONTEXT_EVAL_JSON = UNIVERSAL_ROOT / "old_context_eval.json"
OLD_CONTEXT_EVAL_MD = UNIVERSAL_ROOT / "old_context_eval.md"
OLD_CONTEXT_EVAL_CSV = UNIVERSAL_ROOT / "old_context_eval_rows.csv"

HIDDEN_BRANCH_EVAL_JSON = UNIVERSAL_ROOT / "hidden_branch_eval.json"
HIDDEN_BRANCH_EVAL_MD = UNIVERSAL_ROOT / "hidden_branch_eval.md"
HIDDEN_BRANCH_EVAL_CSV = UNIVERSAL_ROOT / "hidden_branch_eval_rows.csv"

BRIDGE_EVAL_JSON = UNIVERSAL_ROOT / "bridge_eval.json"
BRIDGE_EVAL_MD = UNIVERSAL_ROOT / "bridge_eval.md"
BRIDGE_EVAL_CSV = UNIVERSAL_ROOT / "bridge_eval_rows.csv"

PRUNING_SIM_JSON = UNIVERSAL_ROOT / "layerwise_pruning_sim.json"
PRUNING_SIM_MD = UNIVERSAL_ROOT / "layerwise_pruning_sim.md"
PRUNING_SIM_CSV = UNIVERSAL_ROOT / "layerwise_pruning_rows.csv"

DOMAIN_JSON = UNIVERSAL_ROOT / "domain_generalization.json"
DOMAIN_MD = UNIVERSAL_ROOT / "domain_generalization.md"

GEOMETRY_JSON = UNIVERSAL_ROOT / "geometry_analysis.json"
GEOMETRY_MD = UNIVERSAL_ROOT / "geometry_analysis.md"

SUMMARY_JSON = UNIVERSAL_ROOT / "summary.json"
SUMMARY_MD = UNIVERSAL_ROOT / "summary.md"
ANALYSIS_JSON = UNIVERSAL_ROOT / "analysis.json"
ANALYSIS_MD = UNIVERSAL_ROOT / "analysis.md"
DOC_MD = PROJECT_ROOT / "docs/evaluator/bg_universal_branch_content_taps_v1.md"

TRAJ_ROOT = PROBE_ROOT / "bg_trajectory_prediction_2026-05-18"
TRAJ_FEATURES_PT = TRAJ_ROOT / "prefix_features.pt"
TRAJ_CONTINUED_JSON = TRAJ_ROOT / "continued_prefixes.json"
TRAJ_SCORES_JSON = TRAJ_ROOT / "prefix_scores.json"
WRAPPER_ROOT = PROBE_ROOT / "wrapper_bg_matched_2026-05-18"
WRAPPER_FEATURES_PT = WRAPPER_ROOT / "wrapper_candidate_features.pt"
WRAPPER_EVAL_JSON = WRAPPER_ROOT / "candidate_eval.json"

V4_DATASET_PT = V4_ROOT / "hidden_origin_quota_dataset_v4.pt"
V4_BRANCHES_PT = V4_ROOT / "quota_hidden_origin_branches.pt"
V4_BRANCHES_PARTIAL_PT = V4_ROOT / "quota_hidden_origin_branches.partial.pt"

UNIVERSAL_SCRIPTS = [
    "bg_universal_tap_inventory_v1.py",
    "build_bg_universal_old_content_dataset_v1.py",
    "build_bg_universal_hidden_branch_dataset_v1.py",
    "build_bg_universal_bridge_dataset_v1.py",
    "expand_bg_universal_tap_data_v1.py",
    "build_bg_universal_tap_dataset_v1.py",
    "train_bg_universal_branch_content_taps_v1.py",
    "evaluate_bg_universal_old_contexts_v1.py",
    "evaluate_bg_universal_hidden_branches_v1.py",
    "evaluate_bg_universal_bridge_pairs_v1.py",
    "simulate_bg_universal_layerwise_pruning_v1.py",
    "analyze_bg_universal_domain_generalization_v1.py",
    "analyze_bg_universal_tap_geometry_v1.py",
    "analyze_bg_universal_branch_content_taps_v1.py",
]

TRAIN_CONFIGS = (
    "24_L4",
    "36_L4",
    "47_L4",
    "concat_24_36",
    "concat_24_36_47",
)


def ensure_root() -> None:
    UNIVERSAL_ROOT.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
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


def write_json_local(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def deterministic_split(task_id: str, *, salt: str = "universal_v1") -> str:
    digest = hashlib.sha1(f"{salt}::{task_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 20:
        return "heldout"
    if bucket < 35:
        return "val"
    return "train"


def domain_bucket(raw: Any) -> str:
    value = str(raw or "unknown").lower()
    if value in {"gsm8k", "math", "arithmetic", "simple_arithmetic"}:
        return "math_simple_arithmetic"
    if "code" in value or "humaneval" in value or "mbpp" in value or "local_dsa" in value:
        return "coding"
    return value


def tensor_feature_map(features: torch.Tensor) -> dict[str, torch.Tensor]:
    x = features.detach().cpu().to(torch.float32)
    out: dict[str, torch.Tensor] = {}
    if x.ndim != 3 or x.shape[0] < 3 or x.shape[1] < 4 or x.shape[-1] != HIDDEN_DIM:
        return out
    layers = {24: 0, 36: 1, 47: 2}
    for layer, idx in layers.items():
        out[f"{layer}_L4"] = x[idx, 3].clone()
        out[f"{layer}_mean"] = x[idx, :4].mean(dim=0).clone()
    out["concat_24_36"] = torch.cat([out["24_L4"], out["36_L4"]], dim=0)
    out["concat_36_47"] = torch.cat([out["36_L4"], out["47_L4"]], dim=0)
    out["concat_24_36_47"] = torch.cat([out["24_L4"], out["36_L4"], out["47_L4"]], dim=0)
    return {k: v for k, v in out.items() if k in CONFIGS and int(v.numel()) == config_dim(k)}


def pair_feature_map(left: dict[str, Any], right: dict[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
    left_features = left.get("features_by_config") or {}
    right_features = right.get("features_by_config") or {}
    out: dict[str, dict[str, torch.Tensor]] = {}
    for config in sorted(set(left_features) & set(right_features)):
        lvec = left_features[config]
        rvec = right_features[config]
        if isinstance(lvec, torch.Tensor) and isinstance(rvec, torch.Tensor) and int(lvec.numel()) == int(rvec.numel()) == config_dim(config):
            out[config] = {"preferred": lvec.detach().cpu().to(torch.float32), "rejected": rvec.detach().cpu().to(torch.float32)}
    return out


def compact_pair(pair: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in pair.items() if k != "features"}
    out["feature_dims"] = {cfg: int(vals["preferred"].numel()) for cfg, vals in (pair.get("features") or {}).items()}
    return out


def compact_candidate(row: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in row.items() if k != "features_by_config"}
    out["available_configs"] = sorted((row.get("features_by_config") or {}).keys())
    return out


def make_pair(left: dict[str, Any], right: dict[str, Any], *, pair_type: str, source_id: str, variant: str, bridge_type: str | None = None) -> dict[str, Any] | None:
    reward_left = float(left.get("reward", 0.0))
    reward_right = float(right.get("reward", 0.0))
    if reward_left == reward_right:
        return None
    preferred, rejected = (left, right) if reward_left > reward_right else (right, left)
    features = pair_feature_map(preferred, rejected)
    if not features:
        return None
    group_id = preferred.get("group_id") or preferred.get("branch_group_id") or preferred.get("task_id")
    pair_id = f"{pair_type}::{source_id}::{group_id}::{preferred.get('candidate_id', preferred.get('branch_id'))}>{rejected.get('candidate_id', rejected.get('branch_id'))}"
    pair: dict[str, Any] = {
        "pair_id": pair_id,
        "pair_type": pair_type,
        "source_id": source_id,
        "variant": variant,
        "bridge_type": bridge_type,
        "domain": domain_bucket(preferred.get("domain")),
        "task_id": str(preferred.get("task_id")),
        "group_id": str(group_id),
        "split": str(preferred.get("split") or deterministic_split(str(preferred.get("task_id")), salt=source_id)),
        "label": 1,
        "label_source": preferred.get("label_source") or rejected.get("label_source") or "deterministic_reward",
        "reward_preferred": float(preferred.get("reward", 0.0)),
        "reward_rejected": float(rejected.get("reward", 0.0)),
        "reward_gap": float(preferred.get("reward", 0.0)) - float(rejected.get("reward", 0.0)),
        "left_origin": preferred.get("origin"),
        "right_origin": rejected.get("origin"),
        "preferred_candidate_id": preferred.get("candidate_id", preferred.get("branch_id")),
        "rejected_candidate_id": rejected.get("candidate_id", rejected.get("branch_id")),
        "features": features,
        "available_configs": sorted(features.keys()),
        "contamination_flags": list(sorted(set(preferred.get("contamination_flags", [])) | set(rejected.get("contamination_flags", [])))),
        "old_frozen_tap_score_preferred": preferred.get("old_frozen_tap_score"),
        "old_frozen_tap_score_rejected": rejected.get("old_frozen_tap_score"),
        "hidden_origin_tap_score_preferred": preferred.get("hidden_origin_tap_score"),
        "hidden_origin_tap_score_rejected": rejected.get("hidden_origin_tap_score"),
    }
    return pair


def pair_counts(pairs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pairs": len(pairs),
        "pairs_by_split": dict(Counter(str(p.get("split")) for p in pairs)),
        "pairs_by_type": dict(Counter(str(p.get("pair_type")) for p in pairs)),
        "pairs_by_domain": dict(Counter(str(p.get("domain")) for p in pairs)),
        "tasks_by_split": {split: sorted({str(p.get("task_id")) for p in pairs if p.get("split") == split}) for split in ("train", "val", "heldout")},
        "feature_config_counts": {cfg: sum(1 for p in pairs if cfg in (p.get("features") or {})) for cfg in CONFIGS},
    }


def summarize_candidates(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get("group_id"))].append(row)
    candidate_pairs = tie_pairs = non_tie_pairs = 0
    diverse_groups = 0
    for vals in groups.values():
        rewards = [float(v.get("reward", 0.0)) for v in vals]
        if len(set(rewards)) > 1:
            diverse_groups += 1
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                candidate_pairs += 1
                if float(vals[i].get("reward", 0.0)) == float(vals[j].get("reward", 0.0)):
                    tie_pairs += 1
                else:
                    non_tie_pairs += 1
    return {
        "candidate_rows": len(rows),
        "groups": len(groups),
        "behaviorally_diverse_groups": diverse_groups,
        "candidate_pairs": candidate_pairs,
        "tie_pairs": tie_pairs,
        "non_tie_pairs": non_tie_pairs,
        "tie_rate": tie_pairs / max(candidate_pairs, 1),
        "domains": dict(Counter(domain_bucket(r.get("domain")) for r in rows)),
        "tasks": len({str(r.get("task_id")) for r in rows}),
    }


def load_trajectory_old_candidates() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not TRAJ_FEATURES_PT.exists() or not TRAJ_CONTINUED_JSON.exists():
        return [], {"source_name": "trajectory_prefixes", "status": "missing"}
    features_payload = torch.load(TRAJ_FEATURES_PT, map_location="cpu", weights_only=False)
    continued = load_json(TRAJ_CONTINUED_JSON, {}) or {}
    continued_rows = list(continued.get("continued_prefixes") or [])
    cont_by_key = {
        (str(r.get("task_id")), int(r.get("prefix_length", -1)), int(r.get("branch_id", -1))): r
        for r in continued_rows
    }
    scores_by_key: dict[tuple[str, int, int], float] = {}
    if TRAJ_SCORES_JSON.exists():
        score_payload = load_json(TRAJ_SCORES_JSON, {}) or {}
        for row in score_payload.get("prefix_scores") or []:
            if row.get("head_id") != "locked::objective_mixed":
                continue
            key = (str(row.get("task_id")), int(row.get("prefix_length", -1)), int(row.get("branch_id", -1)))
            scores_by_key[key] = float(row.get("margin_sum", 0.0))
    rows: list[dict[str, Any]] = []
    for rec in features_payload.get("records") or []:
        key = (str(rec.get("task_id")), int(rec.get("prefix_length", -1)), int(rec.get("branch_id", -1)))
        cont = cont_by_key.get(key)
        if not cont:
            continue
        fmap = tensor_feature_map(rec.get("features")) if isinstance(rec.get("features"), torch.Tensor) else {}
        if not fmap:
            continue
        success = bool(cont.get("is_correct") or (cont.get("evaluation") or {}).get("success"))
        reward = 1.0 if success else 0.0
        domain = domain_bucket(cont.get("domain") or rec.get("domain"))
        task_id = str(cont.get("task_id") or rec.get("task_id"))
        prefix_length = int(cont.get("prefix_length", rec.get("prefix_length", -1)))
        branch_id = int(cont.get("branch_id", rec.get("branch_id", -1)))
        rows.append(
            {
                "candidate_id": f"traj::{task_id}::{prefix_length}::{branch_id}",
                "group_id": f"traj::{task_id}::prefix={prefix_length}",
                "task_id": task_id,
                "domain": domain,
                "origin": "old_content",
                "source_id": "trajectory_prefixes",
                "branch_id": branch_id,
                "prefix_length": prefix_length,
                "reward": reward,
                "correctness": success,
                "label_source": "continued_prefix_correctness",
                "split": deterministic_split(task_id, salt="old_content_trajectory"),
                "features_by_config": fmap,
                "old_frozen_tap_score": scores_by_key.get(key),
                "contamination_flags": [],
            }
        )
    return rows, {"source_name": "trajectory_prefixes", "status": "usable", **summarize_candidates(rows)}


def load_wrapper_code_candidates() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not WRAPPER_FEATURES_PT.exists() or not WRAPPER_EVAL_JSON.exists():
        return [], {"source_name": "cached_wrapper_code_candidates", "status": "missing"}
    features_payload = torch.load(WRAPPER_FEATURES_PT, map_location="cpu", weights_only=False)
    eval_payload = load_json(WRAPPER_EVAL_JSON, {}) or {}
    eval_by_uid = {str(row.get("candidate_uid")): row for row in eval_payload.get("candidate_evaluations") or []}
    rows: list[dict[str, Any]] = []
    for task_id, candidates in (features_payload.get("features") or {}).items():
        for uid, tensor in (candidates or {}).items():
            ev = eval_by_uid.get(str(uid), {})
            fmap = tensor_feature_map(tensor) if isinstance(tensor, torch.Tensor) else {}
            if not fmap:
                continue
            reward = 1.0 if bool(ev.get("is_correct")) else 0.0
            rows.append(
                {
                    "candidate_id": str(uid),
                    "group_id": f"wrapper_code::{task_id}",
                    "task_id": str(task_id),
                    "domain": "coding",
                    "origin": "old_content",
                    "source_id": "cached_wrapper_code_candidates",
                    "reward": reward,
                    "correctness": bool(ev.get("is_correct")),
                    "label_source": "cached_verifier_correctness",
                    "split": deterministic_split(str(task_id), salt="old_content_wrapper_code"),
                    "features_by_config": fmap,
                    "contamination_flags": ["cached_wrapper_source_no_runtime_execution"],
                }
            )
    summary = summarize_candidates(rows)
    summary["status"] = "features_only_no_non_tie_pairs" if summary["non_tie_pairs"] == 0 else "usable"
    summary["source_name"] = "cached_wrapper_code_candidates"
    return rows, summary


def build_pairs_from_candidates(rows: Sequence[dict[str, Any]], source_id: str, variant: str = "old_context_primary") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs: list[dict[str, Any]] = []
    ties: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("group_id"))].append(row)
    for gid, vals in sorted(grouped.items()):
        vals = sorted(vals, key=lambda r: str(r.get("candidate_id")))
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                if float(vals[i].get("reward", 0.0)) == float(vals[j].get("reward", 0.0)):
                    ties.append({"group_id": gid, "left": vals[i].get("candidate_id"), "right": vals[j].get("candidate_id"), "reward": vals[i].get("reward"), "split": vals[i].get("split")})
                    continue
                pair = make_pair(vals[i], vals[j], pair_type="old_content", source_id=source_id, variant=variant)
                if pair:
                    pairs.append(pair)
    return pairs, ties


def hidden_pair_from_existing(pair: dict[str, Any], *, source_id: str, pair_type: str = "hidden_branch", bridge_type: str | None = None) -> dict[str, Any] | None:
    features = pair.get("features") or {}
    usable = {}
    for config, vals in features.items():
        pref = vals.get("preferred") if isinstance(vals, dict) else None
        rej = vals.get("rejected") if isinstance(vals, dict) else None
        if isinstance(pref, torch.Tensor) and isinstance(rej, torch.Tensor) and int(pref.numel()) == int(rej.numel()) == config_dim(config):
            usable[config] = {"preferred": pref.detach().cpu().to(torch.float32), "rejected": rej.detach().cpu().to(torch.float32)}
    if not usable:
        return None
    return {
        **{k: v for k, v in pair.items() if k != "features"},
        "pair_id": f"{pair_type}::{source_id}::{pair.get('pair_id')}",
        "pair_type": pair_type,
        "source_id": source_id,
        "bridge_type": bridge_type,
        "group_id": str(pair.get("branch_group_id") or pair.get("group_id")),
        "domain": domain_bucket(pair.get("domain")),
        "features": usable,
        "available_configs": sorted(usable.keys()),
        "left_origin": "hidden_branch",
        "right_origin": "hidden_branch",
        "contamination_flags": list(pair.get("contamination_flags") or []),
    }


def load_hidden_branch_pairs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    sources = [
        ("branch_generator_v1", BGV1_SELECTOR_DATASET_PT, "primary_safe_deterministic"),
        ("quota_v4", V4_DATASET_PT, "primary_safe_deterministic"),
    ]
    pairs: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for source_id, path, variant in sources:
        payload = load_pt(path, {}) or {}
        raw_pairs = list(payload.get("pairs_by_variant", {}).get(variant) or payload.get("pairs") or [])
        usable = [hidden_pair_from_existing(pair, source_id=source_id) for pair in raw_pairs]
        usable_pairs = [pair for pair in usable if pair]
        pairs.extend(usable_pairs)
        source_rows.append(
            {
                "source_name": source_id,
                "domain": "reasoning/science",
                "task_count": len({str(p.get("task_id")) for p in usable_pairs}),
                "pair_count": len(usable_pairs),
                "branch_group_count": len({str(p.get("group_id")) for p in usable_pairs}),
                "label_type": "deterministic branch reward",
                "feature_configs_available": sorted({cfg for p in usable_pairs for cfg in p.get("features", {})}),
                "heldout_split_availability": len([p for p in usable_pairs if p.get("split") == "heldout"]),
                "compatibility": "usable" if usable_pairs else "missing_or_unusable",
            }
        )
    return pairs, source_rows, pair_counts(pairs)


def load_v4_branch_rows() -> list[dict[str, Any]]:
    payload = load_pt(V4_BRANCHES_PT, None)
    if not payload:
        payload = load_pt(V4_BRANCHES_PARTIAL_PT, {})
    rows = []
    for row in list((payload or {}).get("rows") or []):
        norm = dict(row)
        norm.setdefault("deterministic_reward", norm.get("reward", 0.0))
        norm.setdefault("reward", norm.get("deterministic_reward", 0.0))
        norm.setdefault("label_source", "deterministic")
        rows.append(norm)
    return rows


def branch_row_to_candidate(row: dict[str, Any], source_id: str) -> dict[str, Any] | None:
    fmap: dict[str, torch.Tensor] = {}
    for config in CONFIGS:
        vec = config_vector_from_row(row, config)
        if isinstance(vec, torch.Tensor) and int(vec.numel()) == config_dim(config):
            fmap[config] = vec.detach().cpu().to(torch.float32)
    if not fmap:
        return None
    return {
        "candidate_id": f"{source_id}::{row.get('branch_group_id')}::{row.get('branch_id')}",
        "branch_id": int(row.get("branch_id", -1)),
        "group_id": str(row.get("branch_group_id")),
        "branch_group_id": str(row.get("branch_group_id")),
        "task_id": str(row.get("task_id")),
        "domain": domain_bucket(row.get("domain")),
        "origin": "clean_branch" if int(row.get("branch_id", -1)) == 0 or row.get("delta_type") == "clean_zero" else "hidden_branch",
        "source_id": source_id,
        "reward": float(row_reward(row, "deterministic")),
        "label_source": "deterministic_branch_reward",
        "split": str(row.get("split")),
        "features_by_config": fmap,
        "old_frozen_tap_score": row.get("old_frozen_tap_score", row.get("tap_margin_sum")),
        "hidden_origin_tap_score": row.get("v4_tap_score") or row.get("v3_tap_score"),
        "branch_point": row.get("branch_point"),
        "alpha": row.get("alpha"),
        "generator_method": row.get("generator_method"),
        "contamination_flags": [],
    }


def load_branch_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_id, source_rows in (("branch_generator_v1", load_generator_rows()), ("quota_v4", load_v4_branch_rows())):
        for row in source_rows:
            if not stable_v2_row(row):
                continue
            if source_id == "branch_generator_v1" and not primary_safe_generator_row(row):
                continue
            if source_id == "quota_v4":
                split = str(row.get("split"))
                if split not in {"train", "val", "heldout"}:
                    continue
                try:
                    alpha = float(row.get("alpha", 999.0))
                except Exception:
                    alpha = 999.0
                if alpha > 0.010001:
                    continue
            cand = branch_row_to_candidate(row, source_id)
            if cand:
                rows.append(cand)
    return rows


def build_clean_bridge_pairs(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    grouped = defaultdict(list)
    for row in candidates:
        grouped[str(row.get("group_id"))].append(row)
    for gid, vals in grouped.items():
        clean = [row for row in vals if row.get("origin") == "clean_branch"]
        if not clean:
            continue
        base = clean[0]
        for row in vals:
            if row is base or row.get("origin") == "clean_branch":
                continue
            pair = make_pair(base, row, pair_type="bridge", source_id=str(row.get("source_id")), variant="hidden_vs_clean_branch", bridge_type="hidden_origin_branch_vs_clean_branch")
            if pair:
                pair["group_id"] = gid
                pairs.append(pair)
    return pairs


def best_heads_from_training() -> dict[str, dict[str, Any] | None]:
    payload = load_pt(UNIVERSAL_HEADS_PT, {}) or {}
    heads = list(payload.get("heads") or [])
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in heads:
        if row.get("flip_diagnostics", {}).get("passes"):
            by_variant[str(row.get("variant"))].append(row)
    best: dict[str, dict[str, Any] | None] = {}
    for variant, vals in by_variant.items():
        best[variant] = max(vals, key=lambda r: float(r.get("metrics", {}).get("balanced_validation_score", r.get("metrics", {}).get("validation_pairwise_accuracy", -1.0))))
    best["v4_hidden_origin"] = best_primary_head(HEADS_V4_PT, "v4_only_primary_safe")
    best["generator_v1_selector"] = best_primary_head(BGV1_SELECTOR_HEADS_PT, "generator_v1_only_primary_safe")
    return best


def pairwise_acc_for_head(head_row: dict[str, Any] | None, pairs: Sequence[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    if not head_row:
        return {"pairwise_accuracy": float("nan"), "pair_count": 0, "config": "missing"}
    config = str(head_row.get("config"))
    cfg_pairs = [p for p in pairs if config in (p.get("features") or {})]
    if not cfg_pairs:
        return {"pairwise_accuracy": float("nan"), "pair_count": 0, "config": config}
    head = build_head(head_row, device)
    acc = pairwise_accuracy_from_pairs(head, cfg_pairs, config, device)
    head.to("cpu")
    return {"pairwise_accuracy": acc, "pair_count": len(cfg_pairs), "config": config, "architecture": head_row.get("architecture"), "head_id": compact_head_id(head_row)}


def pairwise_acc_from_scores(pairs: Sequence[dict[str, Any]], pref_key: str, rej_key: str) -> dict[str, Any]:
    used = 0
    correct = 0
    for pair in pairs:
        a = pair.get(pref_key)
        b = pair.get(rej_key)
        try:
            af = float(a)
            bf = float(b)
        except Exception:
            continue
        used += 1
        correct += int(af > bf)
    return {"pairwise_accuracy": correct / max(used, 1) if used else float("nan"), "pair_count": used}


def group_metric(rows: Sequence[dict[str, Any]], selected_indices: Sequence[int], k: int) -> dict[str, Any]:
    ordered = list(rows)
    rewards = [float(r.get("reward", 0.0)) for r in ordered]
    if not rewards:
        return {}
    oracle = max(rewards)
    oracle_indices = {i for i, r in enumerate(rewards) if r == oracle}
    selected = list(selected_indices)[: max(1, k)]
    selected_rewards = [rewards[i] for i in selected] if selected else []
    best_selected = max(selected_rewards) if selected_rewards else float("-inf")
    return {
        "reward_mean": float(mean(selected_rewards)) if selected_rewards else float("nan"),
        "top1_success": 1.0 if selected and rewards[selected[0]] == oracle and oracle > 0 else 0.0,
        "topk_oracle_coverage": 1.0 if oracle_indices & set(selected) else 0.0,
        "oracle_retention": 1.0 if oracle_indices & set(selected) else 0.0,
        "regret": oracle - best_selected,
        "oracle_reward": oracle,
        "selected_ids": [ordered[i].get("candidate_id", ordered[i].get("branch_id")) for i in selected],
        "oracle_ids": [ordered[i].get("candidate_id", ordered[i].get("branch_id")) for i in sorted(oracle_indices)],
    }


def rank_group_with_head(rows: Sequence[dict[str, Any]], head_row: dict[str, Any] | None, device: torch.device, *, feature_key: str = "features_by_config") -> list[int]:
    if not head_row:
        return []
    config = str(head_row.get("config"))
    vectors = []
    for row in rows:
        fmap = row.get(feature_key) or {}
        vec = fmap.get(config)
        if not isinstance(vec, torch.Tensor) or int(vec.numel()) != config_dim(config):
            return []
        vectors.append(vec)
    if not vectors:
        return []
    head = build_head(head_row, device)
    mat = score_matrix(head, vectors, device)
    head.to("cpu")
    return list(ranking_from_matrix(mat)["ranking"])


def aggregate_metric_rows(rows: Sequence[dict[str, Any]], keys: Sequence[str] = ("policy",)) -> dict[str, Any]:
    out: dict[str, Any] = {}
    grouped = defaultdict(list)
    for row in rows:
        label = "::".join(str(row.get(k)) for k in keys)
        grouped[label].append(row)
    for label, vals in grouped.items():
        numeric: dict[str, list[float]] = defaultdict(list)
        for row in vals:
            for key, value in row.items():
                try:
                    x = float(value)
                except Exception:
                    continue
                if math.isfinite(x):
                    numeric[key].append(x)
        out[label] = {"n": len(vals), **{k: float(mean(v)) for k, v in numeric.items() if v}}
    return out


def compact_dataset_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in payload.items() if k not in {"pairs", "pairs_by_variant", "candidate_rows", "tie_rows"}}
    if "pairs" in payload:
        out["pairs"] = [compact_pair(p) for p in payload.get("pairs") or []]
    if "pairs_by_variant" in payload:
        out["pairs_by_variant"] = {k: [compact_pair(p) for p in vals] for k, vals in (payload.get("pairs_by_variant") or {}).items()}
    if "candidate_rows" in payload:
        out["candidate_rows"] = [compact_candidate(r) for r in payload.get("candidate_rows") or []]
    if "tie_rows" in payload:
        out["tie_rows"] = payload.get("tie_rows")
    return out


def run_inventory() -> int:
    ensure_root()
    started = time.time()
    traj_rows, traj_summary = load_trajectory_old_candidates()
    wrapper_rows, wrapper_summary = load_wrapper_code_candidates()
    hidden_pairs, hidden_sources, hidden_summary = load_hidden_branch_pairs()
    branch_candidates = load_branch_candidates()
    clean_bridge_pairs = build_clean_bridge_pairs(branch_candidates)
    source_rows = [
        {
            "source_name": "trajectory_prefixes",
            "domain": "reasoning/science/math_simple_arithmetic",
            "task_count": traj_summary.get("tasks", 0),
            "pair_count": traj_summary.get("non_tie_pairs", 0),
            "branch_group_count": traj_summary.get("groups", 0),
            "behaviorally_diverse_group_count": traj_summary.get("behaviorally_diverse_groups", 0),
            "label_type": "continued prefix correctness",
            "feature_configs_available": sorted({cfg for r in traj_rows for cfg in r.get("features_by_config", {})}),
            "heldout_split_availability": len({r["task_id"] for r in traj_rows if r.get("split") == "heldout"}),
            "compatibility": "usable" if traj_summary.get("non_tie_pairs", 0) else "data_limited",
        },
        {
            "source_name": "cached_wrapper_code_candidates",
            "domain": "coding",
            "task_count": wrapper_summary.get("tasks", 0),
            "pair_count": wrapper_summary.get("non_tie_pairs", 0),
            "branch_group_count": wrapper_summary.get("groups", 0),
            "behaviorally_diverse_group_count": wrapper_summary.get("behaviorally_diverse_groups", 0),
            "label_type": "cached verifier correctness",
            "feature_configs_available": sorted({cfg for r in wrapper_rows for cfg in r.get("features_by_config", {})}),
            "heldout_split_availability": len({r["task_id"] for r in wrapper_rows if r.get("split") == "heldout"}),
            "compatibility": "features_only_no_non_tie_pairs" if wrapper_summary.get("non_tie_pairs", 0) == 0 else "usable",
        },
        *hidden_sources,
        {
            "source_name": "clean_branch_bridge",
            "domain": "reasoning/science",
            "task_count": len({str(p.get("task_id")) for p in clean_bridge_pairs}),
            "pair_count": len(clean_bridge_pairs),
            "branch_group_count": len({str(p.get("group_id")) for p in clean_bridge_pairs}),
            "behaviorally_diverse_group_count": len({str(p.get("group_id")) for p in clean_bridge_pairs}),
            "label_type": "branch vs clean final reward",
            "feature_configs_available": sorted({cfg for p in clean_bridge_pairs for cfg in p.get("features", {})}),
            "heldout_split_availability": len([p for p in clean_bridge_pairs if p.get("split") == "heldout"]),
            "compatibility": "usable" if clean_bridge_pairs else "missing",
        },
    ]
    old_usable = traj_summary.get("non_tie_pairs", 0) > 0
    hidden_usable = len(hidden_pairs) > 0
    bridge_usable = len(clean_bridge_pairs) > 0 or len(hidden_pairs) > 0
    if old_usable and hidden_usable and bridge_usable:
        verdict = "READY" if clean_bridge_pairs else "PARTIAL"
    elif old_usable and hidden_usable:
        verdict = "NEEDS_BRIDGE_GENERATION"
    elif old_usable or hidden_usable:
        verdict = "PARTIAL"
    else:
        verdict = "BLOCKED"
    payload = {
        "BG_UNIVERSAL_TAP_INVENTORY_VERDICT": verdict,
        "verdict": verdict,
        "old_context_summary": {"trajectory": traj_summary, "cached_wrapper_code": wrapper_summary},
        "hidden_branch_summary": hidden_summary,
        "bridge_preview": {"clean_bridge_pairs": len(clean_bridge_pairs), "branch_candidates": len(branch_candidates)},
        "source_inventory": source_rows,
        "runtime_constraints": {
            "no_wrapper_execution": True,
            "no_tap_score_labels": True,
            "no_ouro_training": True,
            "coding_source_has_no_non_tie_training_pairs": wrapper_summary.get("non_tie_pairs", 0) == 0,
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json_local(INVENTORY_JSON, payload)
    write_csv(SOURCE_INVENTORY_CSV, source_rows)
    lines = ["# Universal Tap Data Inventory", "", f"BG_UNIVERSAL_TAP_INVENTORY_VERDICT = {verdict}", "", "## Sources", ""]
    lines.extend(md_table(source_rows, ["source_name", "domain", "task_count", "pair_count", "branch_group_count", "compatibility"]))
    lines.extend(["", "The cached wrapper/code pool was inspected only as cached data. It has compatible hidden features but no within-task non-tie labels, so it is not used for primary pairwise labels."])
    write_md(INVENTORY_MD, lines)
    print(f"BG_UNIVERSAL_TAP_INVENTORY_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


def run_old_content_dataset() -> int:
    ensure_root()
    started = time.time()
    traj_rows, traj_summary = load_trajectory_old_candidates()
    wrapper_rows, wrapper_summary = load_wrapper_code_candidates()
    traj_pairs, traj_ties = build_pairs_from_candidates(traj_rows, "trajectory_prefixes")
    wrapper_pairs, wrapper_ties = build_pairs_from_candidates(wrapper_rows, "cached_wrapper_code_candidates")
    pairs = traj_pairs + wrapper_pairs
    candidates = traj_rows + wrapper_rows
    counts = pair_counts(pairs)
    train_ok = counts["pairs_by_split"].get("train", 0) >= 50
    val_ok = counts["pairs_by_split"].get("val", 0) >= 10
    heldout_ok = counts["pairs_by_split"].get("heldout", 0) >= 10
    if train_ok and val_ok and heldout_ok and len(counts["pairs_by_domain"]) >= 2:
        verdict = "READY"
    elif pairs:
        verdict = "PARTIAL"
    else:
        verdict = "BLOCKED"
    payload = {
        "BG_UNIVERSAL_OLD_CONTENT_DATASET_VERDICT": verdict,
        "verdict": verdict,
        "pairs": pairs,
        "candidate_rows": candidates,
        "tie_rows": traj_ties + wrapper_ties,
        "source_summaries": {"trajectory": traj_summary, "cached_wrapper_code": wrapper_summary},
        "counts": counts,
        "old_tap_baseline_compatibility": {
            "trajectory_locked_objective_mixed_scores": sum(1 for p in pairs if p.get("old_frozen_tap_score_preferred") is not None and p.get("old_frozen_tap_score_rejected") is not None),
            "code_cached_features_no_non_tie_pairs": wrapper_summary.get("non_tie_pairs", 0) == 0,
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(payload, OLD_CONTENT_PT)
    write_json_local(OLD_CONTENT_JSON, compact_dataset_payload(payload))
    split_rows = [{"split": split, "pairs": counts["pairs_by_split"].get(split, 0), "tasks": len(counts["tasks_by_split"].get(split, []))} for split in ("train", "val", "heldout")]
    lines = [
        "# Universal Old Content Dataset",
        "",
        f"BG_UNIVERSAL_OLD_CONTENT_DATASET_VERDICT = {verdict}",
        "",
        f"- pairs: `{len(pairs)}`",
        f"- domains: `{counts['pairs_by_domain']}`",
        f"- coding cached rows: `{wrapper_summary.get('candidate_rows', 0)}`; coding non-tie pairs: `{wrapper_summary.get('non_tie_pairs', 0)}`",
        "",
        "## Split Counts",
        "",
    ]
    lines.extend(md_table(split_rows, ["split", "pairs", "tasks"]))
    write_md(OLD_CONTENT_MD, lines)
    print(f"BG_UNIVERSAL_OLD_CONTENT_DATASET_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


def run_hidden_branch_dataset() -> int:
    ensure_root()
    started = time.time()
    pairs, source_rows, counts = load_hidden_branch_pairs()
    if counts["pairs_by_split"].get("heldout", 0) >= 100 and counts["pairs_by_split"].get("train", 0) >= 100:
        verdict = "READY"
    elif pairs:
        verdict = "PARTIAL"
    else:
        verdict = "BLOCKED"
    payload = {
        "BG_UNIVERSAL_HIDDEN_BRANCH_DATASET_VERDICT": verdict,
        "verdict": verdict,
        "pairs": pairs,
        "source_inventory": source_rows,
        "counts": counts,
        "heldout_support": counts["pairs_by_split"].get("heldout", 0),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(payload, HIDDEN_BRANCH_PT)
    write_json_local(HIDDEN_BRANCH_JSON, compact_dataset_payload(payload))
    lines = ["# Universal Hidden Branch Dataset", "", f"BG_UNIVERSAL_HIDDEN_BRANCH_DATASET_VERDICT = {verdict}", "", f"- pairs: `{len(pairs)}`", f"- pairs_by_split: `{counts['pairs_by_split']}`", f"- pairs_by_domain: `{counts['pairs_by_domain']}`", "", "## Sources", ""]
    lines.extend(md_table(source_rows, ["source_name", "task_count", "pair_count", "branch_group_count", "heldout_split_availability", "compatibility"]))
    write_md(HIDDEN_BRANCH_MD, lines)
    print(f"BG_UNIVERSAL_HIDDEN_BRANCH_DATASET_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


def run_bridge_dataset() -> int:
    ensure_root()
    started = time.time()
    hidden_pairs, _, _ = load_hidden_branch_pairs()
    type1_pairs = []
    for pair in hidden_pairs:
        bridge = hidden_pair_from_existing(pair, source_id=str(pair.get("source_id")), pair_type="bridge", bridge_type="early_branch_state_labelled_by_final_branch_outcome")
        if bridge:
            bridge["variant"] = "early_branch_state_final_outcome"
            type1_pairs.append(bridge)
    branch_candidates = load_branch_candidates()
    type2_pairs = build_clean_bridge_pairs(branch_candidates)
    pairs = type1_pairs + type2_pairs
    counts = pair_counts(pairs)
    type_counts = dict(Counter(str(p.get("bridge_type")) for p in pairs))
    if type2_pairs and counts["pairs_by_split"].get("heldout", 0) >= 50 and counts["pairs_by_split"].get("train", 0) >= 50:
        verdict = "READY"
    elif pairs:
        verdict = "PARTIAL" if type2_pairs else "DIAGNOSTIC_ONLY"
    else:
        verdict = "BLOCKED"
    payload = {
        "BG_UNIVERSAL_BRIDGE_DATASET_VERDICT": verdict,
        "verdict": verdict,
        "pairs": pairs,
        "candidate_rows": branch_candidates,
        "counts": counts,
        "bridge_type_counts": type_counts,
        "label_quality": {
            "type1": "final deterministic branch reward on early hidden branch states",
            "type2": "same-task hidden-origin branch vs clean branch final reward",
            "type3": "not available from cached same-task old-content overlap",
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(payload, BRIDGE_PT)
    write_json_local(BRIDGE_JSON, compact_dataset_payload(payload))
    lines = ["# Universal Bridge Dataset", "", f"BG_UNIVERSAL_BRIDGE_DATASET_VERDICT = {verdict}", "", f"- pairs: `{len(pairs)}`", f"- bridge_type_counts: `{type_counts}`", f"- pairs_by_split: `{counts['pairs_by_split']}`", "", "Bridge type 3 was not fabricated because no cached same-task old candidate outputs with compatible labels overlapped the hidden-origin branch tasks."]
    write_md(BRIDGE_MD, lines)
    print(f"BG_UNIVERSAL_BRIDGE_DATASET_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


def run_data_expansion() -> int:
    ensure_root()
    inventory = load_json(INVENTORY_JSON, {}) or {}
    bridge = load_json(BRIDGE_JSON, {}) or {}
    needed = inventory.get("verdict") in {"NEEDS_FEATURE_CAPTURE", "NEEDS_BRIDGE_GENERATION"} or bridge.get("verdict") in {"BLOCKED"}
    verdict = "SKIPPED"
    reason = "cached old-content, hidden-branch, and bridge diagnostics were sufficient for this bounded universal-tap run"
    if needed:
        verdict = "SKIPPED"
        reason = "expansion would require generation; cached data already permits diagnostic training/evaluation, so no bounded generation was run"
    payload = {
        "BG_UNIVERSAL_DATA_EXPANSION_VERDICT": verdict,
        "verdict": verdict,
        "reason": reason,
        "expanded_rows_jsonl": rel(EXPANDED_ROWS_JSONL),
        "no_generation_run": True,
    }
    write_json_local(DATA_EXPANSION_JSON, payload)
    write_md(DATA_EXPANSION_MD, ["# Universal Tap Data Expansion", "", f"BG_UNIVERSAL_DATA_EXPANSION_VERDICT = {verdict}", "", reason])
    print(f"BG_UNIVERSAL_DATA_EXPANSION_VERDICT = {verdict}", flush=True)
    return 0


def downsample_balanced(pairs: Sequence[dict[str, Any]], fractions: dict[str, float]) -> list[dict[str, Any]]:
    by_split_type: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_split_type[(str(pair.get("split")), str(pair.get("pair_type")))].append(pair)
    out: list[dict[str, Any]] = []
    for split in ("train", "val", "heldout"):
        pools = {ptype: by_split_type.get((split, ptype), []) for ptype in fractions}
        if not any(pools.values()):
            continue
        if all(pools.values()):
            base = min(int(len(vals) / max(fractions[ptype], 1e-9)) for ptype, vals in pools.items() if fractions[ptype] > 0)
            for ptype, frac in fractions.items():
                take = min(len(pools[ptype]), max(1, int(round(base * frac))))
                out.extend(sorted(pools[ptype], key=lambda p: str(p.get("pair_id")))[:take])
        else:
            for vals in pools.values():
                out.extend(vals)
    return out


def run_universal_dataset() -> int:
    ensure_root()
    started = time.time()
    old_payload = load_pt(OLD_CONTENT_PT, {}) or {}
    hidden_payload = load_pt(HIDDEN_BRANCH_PT, {}) or {}
    bridge_payload = load_pt(BRIDGE_PT, {}) or {}
    old_pairs = list(old_payload.get("pairs") or [])
    hidden_pairs = list(hidden_payload.get("pairs") or [])
    bridge_pairs = list(bridge_payload.get("pairs") or [])
    for pair in old_pairs:
        pair["pair_type"] = "old_content"
    for pair in hidden_pairs:
        pair["pair_type"] = "hidden_branch"
    for pair in bridge_pairs:
        pair["pair_type"] = "bridge"
    all_pairs = old_pairs + hidden_pairs + bridge_pairs
    pairs_by_variant = {
        "universal_balanced": downsample_balanced(all_pairs, {"old_content": 0.4, "hidden_branch": 0.4, "bridge": 0.2}),
        "universal_no_bridge": downsample_balanced(old_pairs + hidden_pairs, {"old_content": 0.5, "hidden_branch": 0.5}),
        "old_content_only": old_pairs,
        "hidden_branch_only": hidden_pairs,
        "bridge_only": bridge_pairs,
        "universal_domain_balanced": downsample_balanced(all_pairs, {"old_content": 0.4, "hidden_branch": 0.4, "bridge": 0.2}),
    }
    counts_by_variant = {name: pair_counts(vals) for name, vals in pairs_by_variant.items()}
    primary_counts = counts_by_variant["universal_balanced"]
    has_old = bool(old_pairs)
    has_hidden = bool(hidden_pairs)
    has_bridge = bool(bridge_pairs)
    heldout_ok = all(primary_counts["pairs_by_split"].get(split, 0) > 0 for split in ("train", "val", "heldout"))
    if has_old and has_hidden and has_bridge and heldout_ok:
        verdict = "READY"
    elif has_old and has_hidden and has_bridge:
        verdict = "BRIDGE_WEAK_BUT_USABLE"
    elif has_old and has_hidden:
        verdict = "OLD_AND_BRANCH_ONLY"
    elif all_pairs:
        verdict = "DATA_LIMITED"
    else:
        verdict = "BLOCKED"
    payload = {
        "BG_UNIVERSAL_TAP_DATASET_VERDICT": verdict,
        "verdict": verdict,
        "pairs": pairs_by_variant["universal_balanced"],
        "pairs_by_variant": pairs_by_variant,
        "counts_by_variant": counts_by_variant,
        "source_verdicts": {
            "old_content": old_payload.get("verdict"),
            "hidden_branch": hidden_payload.get("verdict"),
            "bridge": bridge_payload.get("verdict"),
        },
        "balance_policy": "40_percent_old_content_40_percent_hidden_branch_20_percent_bridge_when_all_sources_present",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(payload, UNIVERSAL_DATASET_PT)
    write_json_local(UNIVERSAL_DATASET_JSON, compact_dataset_payload(payload))
    rows_md = []
    for name, counts in counts_by_variant.items():
        rows_md.append({"variant": name, "pairs": counts["pairs"], "train": counts["pairs_by_split"].get("train", 0), "val": counts["pairs_by_split"].get("val", 0), "heldout": counts["pairs_by_split"].get("heldout", 0), "types": counts["pairs_by_type"]})
    lines = ["# Universal Tap Dataset", "", f"BG_UNIVERSAL_TAP_DATASET_VERDICT = {verdict}", "", "## Variants", ""]
    lines.extend(md_table(rows_md, ["variant", "pairs", "train", "val", "heldout", "types"]))
    write_md(UNIVERSAL_DATASET_MD, lines)
    print(f"BG_UNIVERSAL_TAP_DATASET_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


def validation_breakdown(head: torch.nn.Module, pairs: Sequence[dict[str, Any]], config: str, device: torch.device) -> dict[str, Any]:
    by_type = {}
    for pair_type in sorted({str(pair.get("pair_type")) for pair in pairs}):
        vals = [pair for pair in pairs if pair.get("pair_type") == pair_type and config in (pair.get("features") or {})]
        by_type[pair_type] = pairwise_accuracy_from_pairs(head, vals, config, device) if vals else float("nan")
    by_domain = {}
    for domain in sorted({str(pair.get("domain")) for pair in pairs}):
        vals = [pair for pair in pairs if str(pair.get("domain")) == domain and config in (pair.get("features") or {})]
        by_domain[domain] = pairwise_accuracy_from_pairs(head, vals, config, device) if vals else float("nan")
    finite_type = [v for v in by_type.values() if isinstance(v, float) and math.isfinite(v)]
    balanced = float(mean(finite_type)) if finite_type else float("nan")
    return {"by_pair_type": by_type, "by_domain": by_domain, "balanced": balanced}


def universal_training_verdict(dataset_verdict: str, heads: Sequence[dict[str, Any]]) -> str:
    if dataset_verdict == "BLOCKED":
        return "INSUFFICIENT"
    valid_all = [h for h in heads if h.get("flip_diagnostics", {}).get("passes") and math.isfinite(float(h.get("metrics", {}).get("balanced_validation_score", float("nan"))))]
    valid = [h for h in valid_all if str(h.get("variant", "")).startswith("universal")]
    if not valid:
        valid = valid_all
    if not valid:
        return "NO_LEARNING" if heads else "INSUFFICIENT"
    best = max(valid, key=lambda h: float(h["metrics"]["balanced_validation_score"]))
    balanced = float(best["metrics"]["balanced_validation_score"])
    by_type = best["metrics"].get("validation_by_pair_type", {})
    old = float(by_type.get("old_content", float("nan")))
    branch = float(by_type.get("hidden_branch", float("nan")))
    bridge = float(by_type.get("bridge", float("nan")))
    if balanced >= 0.58 and old >= 0.53 and branch >= 0.53 and (not math.isfinite(bridge) or bridge >= 0.53):
        return "READY"
    if math.isfinite(old) and old >= 0.58 and (not math.isfinite(branch) or branch < 0.53):
        return "CONTENT_ONLY"
    if math.isfinite(branch) and branch >= 0.58 and (not math.isfinite(old) or old < 0.53):
        return "BRANCH_ONLY"
    if balanced > 0.51:
        return "WEAK"
    train = float(best["metrics"].get("train_pairwise_accuracy", 0.0))
    if train >= 0.70 and balanced <= 0.51:
        return "OVERFIT"
    return "NO_LEARNING"


def run_training() -> int:
    ensure_root()
    started = time.time()
    if not UNIVERSAL_DATASET_PT.exists():
        payload = {"BG_UNIVERSAL_TAP_TRAINING_VERDICT": "INSUFFICIENT", "verdict": "INSUFFICIENT", "blocker": "missing universal dataset"}
        write_json_local(TRAINING_JSON, payload)
        write_md(TRAINING_MD, ["# Universal Tap Training", "", "BG_UNIVERSAL_TAP_TRAINING_VERDICT = INSUFFICIENT"])
        print("BG_UNIVERSAL_TAP_TRAINING_VERDICT = INSUFFICIENT", flush=True)
        return 1
    dataset = load_pt(UNIVERSAL_DATASET_PT, {}) or {}
    pairs_by_variant = dataset.get("pairs_by_variant") or {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = SimpleNamespace(epochs=50, patience=8, batch_size=32, weight_decay=0.01, score_l2=1e-4, gradient_clip=1.0)
    seeds = [42, 43, 44]
    lrs = [1e-4, 3e-4, 1e-3]
    variants = ["universal_balanced", "universal_no_bridge", "old_content_only", "hidden_branch_only", "bridge_only", "universal_domain_balanced"]
    heads: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    for variant in variants:
        all_pairs = list(pairs_by_variant.get(variant) or [])
        if not all_pairs:
            continue
        train_pairs_all = [pair for pair in all_pairs if pair.get("split") == "train"]
        val_pairs_all = [pair for pair in all_pairs if pair.get("split") == "val"]
        if len(train_pairs_all) < 4 or len(val_pairs_all) < 2:
            training_rows.append({"variant": variant, "status": "skipped", "reason": "insufficient train/val pairs", "train_pairs": len(train_pairs_all), "val_pairs": len(val_pairs_all)})
            continue
        for config in TRAIN_CONFIGS:
            train_pairs = pairs_for_config(train_pairs_all, config)
            val_pairs = pairs_for_config(val_pairs_all, config)
            if len(train_pairs) < 4 or len(val_pairs) < 2:
                training_rows.append({"variant": variant, "config": config, "status": "skipped", "reason": "insufficient pairs for config", "train_pairs": len(train_pairs), "val_pairs": len(val_pairs)})
                continue
            for architecture in ARCHITECTURES:
                for seed in seeds:
                    for lr in lrs:
                        print(f"training universal {variant} {architecture} {config} seed={seed} lr={lr}", flush=True)
                        head, metrics = train_one(architecture=architecture, config=config, seed=seed, lr=lr, train_pairs=train_pairs, val_pairs=val_pairs, args=args, device=device)
                        head_dev = head.to(device)
                        val_break = validation_breakdown(head_dev, val_pairs, config, device)
                        train_break = validation_breakdown(head_dev, train_pairs, config, device)
                        flip = flip_diagnostics(head_dev, val_pairs, config, device)
                        state_dict = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                        metrics.update(
                            {
                                "variant": variant,
                                "balanced_validation_score": val_break["balanced"],
                                "validation_by_pair_type": val_break["by_pair_type"],
                                "validation_by_domain": val_break["by_domain"],
                                "train_by_pair_type": train_break["by_pair_type"],
                            }
                        )
                        row = {
                            "head_group": "universal_branch_content_taps_v1",
                            "variant": variant,
                            "architecture": architecture,
                            "config": config,
                            "dim": config_dim(config),
                            "state_dict": state_dict,
                            "metrics": metrics,
                            "flip_diagnostics": flip,
                            "label_policy": "actual rewards/preferences only; no tap scores as labels",
                        }
                        heads.append(row)
                        training_rows.append(compact_head(row))
                        head.to("cpu")
    verdict = universal_training_verdict(str(dataset.get("verdict")), heads)
    valid_heads = [row for row in heads if row.get("flip_diagnostics", {}).get("passes")]
    universal_heads = [row for row in valid_heads if str(row.get("variant", "")).startswith("universal")]
    best_pool = universal_heads or valid_heads
    best = max(best_pool, key=lambda row: float(row["metrics"].get("balanced_validation_score", -1.0))) if best_pool else None
    summary_rows = []
    by_variant = defaultdict(list)
    for row in valid_heads:
        by_variant[str(row.get("variant"))].append(row)
    for variant, vals in sorted(by_variant.items()):
        best_v = max(vals, key=lambda row: float(row["metrics"].get("balanced_validation_score", -1.0)))
        summary_rows.append(
            {
                "variant": variant,
                "best_config": best_v["config"],
                "architecture": best_v["architecture"],
                "balanced_val": rate(best_v["metrics"].get("balanced_validation_score")),
                "global_val": rate(best_v["metrics"].get("validation_pairwise_accuracy")),
                "old_val": rate(best_v["metrics"].get("validation_by_pair_type", {}).get("old_content")),
                "branch_val": rate(best_v["metrics"].get("validation_by_pair_type", {}).get("hidden_branch")),
                "bridge_val": rate(best_v["metrics"].get("validation_by_pair_type", {}).get("bridge")),
            }
        )
    payload = {
        "BG_UNIVERSAL_TAP_TRAINING_VERDICT": verdict,
        "verdict": verdict,
        "dataset_verdict": dataset.get("verdict"),
        "device": str(device),
        "architectures": list(ARCHITECTURES),
        "seeds": seeds,
        "lrs": lrs,
        "heads": heads,
        "best_head": compact_head(best) if best else None,
        "variant_summary": summary_rows,
        "training_rows": training_rows,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(payload, UNIVERSAL_HEADS_PT)
    json_payload = {k: v for k, v in payload.items() if k != "heads"}
    json_payload["heads"] = [compact_head(h) for h in heads]
    write_json_local(TRAINING_JSON, json_payload)
    lines = ["# Universal Branch-Content Tap Training", "", f"BG_UNIVERSAL_TAP_TRAINING_VERDICT = {verdict}", "", f"- trained_heads: `{len(heads)}`", f"- best_head: `{json_payload['best_head']}`", "", "## Variant Summary", ""]
    lines.extend(md_table(summary_rows, ["variant", "best_config", "architecture", "balanced_val", "global_val", "old_val", "branch_val", "bridge_val"]))
    write_md(TRAINING_MD, lines)
    print(f"BG_UNIVERSAL_TAP_TRAINING_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "INSUFFICIENT" else 1


def old_group_eval_rows(candidate_rows: Sequence[dict[str, Any]], heads: dict[str, dict[str, Any] | None], device: torch.device) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in candidate_rows:
        grouped[str(row.get("group_id"))].append(row)
    out: list[dict[str, Any]] = []
    for gid, vals in grouped.items():
        vals = sorted(vals, key=lambda row: str(row.get("candidate_id")))
        if len(vals) < 2 or len({float(v.get("reward", 0.0)) for v in vals}) < 2:
            continue
        base_metrics = {
            "group_id": gid,
            "task_id": vals[0].get("task_id"),
            "domain": vals[0].get("domain"),
            "split": vals[0].get("split"),
            "group_size": len(vals),
        }
        out.append({**base_metrics, "policy": "first_candidate_baseline", **group_metric(vals, [0], 1), "top_k": 1})
        out.append({**base_metrics, "policy": "random_top1", "top1_success": sum(1 for v in vals if float(v.get("reward", 0.0)) == max(float(x.get("reward", 0.0)) for x in vals)) / len(vals), "topk_oracle_coverage": 1 / len(vals), "reward_mean": mean(float(v.get("reward", 0.0)) for v in vals), "regret": max(float(v.get("reward", 0.0)) for v in vals) - mean(float(v.get("reward", 0.0)) for v in vals), "top_k": 1})
        score_indices = [(i, vals[i].get("old_frozen_tap_score")) for i in range(len(vals)) if vals[i].get("old_frozen_tap_score") is not None]
        if len(score_indices) == len(vals):
            ranking = [i for i, _ in sorted(score_indices, key=lambda item: -float(item[1]))]
            out.append({**base_metrics, "policy": "old_frozen_bg_objective_mixed", **group_metric(vals, ranking, 1), "top_k": 1})
            out.append({**base_metrics, "policy": "old_frozen_bg_objective_mixed_top2", **group_metric(vals, ranking, 2), "top_k": 2})
        for label in ("universal_balanced", "universal_no_bridge", "old_content_only", "hidden_branch_only"):
            ranking = rank_group_with_head(vals, heads.get(label), device)
            if not ranking:
                continue
            out.append({**base_metrics, "policy": label, **group_metric(vals, ranking, 1), "top_k": 1, "config": heads[label]["config"]})
            out.append({**base_metrics, "policy": f"{label}_top2", **group_metric(vals, ranking, 2), "top_k": 2, "config": heads[label]["config"]})
    return out


def run_old_context_eval() -> int:
    ensure_root()
    started = time.time()
    old_payload = load_pt(OLD_CONTENT_PT, {}) or {}
    train_payload = load_pt(UNIVERSAL_HEADS_PT, {}) or {}
    pairs = [p for p in old_payload.get("pairs") or [] if p.get("split") == "heldout"]
    candidates = [r for r in old_payload.get("candidate_rows") or [] if r.get("split") == "heldout"]
    heads = best_heads_from_training()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairwise_rows = [{"selector": "random", "pairwise_accuracy": 0.5, "pair_count": len(pairs), "config": "baseline"}]
    pairwise_rows.append({"selector": "old_frozen_bg_objective_mixed", **pairwise_acc_from_scores(pairs, "old_frozen_tap_score_preferred", "old_frozen_tap_score_rejected")})
    for label in ("universal_balanced", "universal_no_bridge", "old_content_only", "hidden_branch_only"):
        pairwise_rows.append({"selector": label, **pairwise_acc_for_head(heads.get(label), pairs, device)})
    group_rows_eval = old_group_eval_rows(candidates, heads, device)
    metrics = aggregate_metric_rows(group_rows_eval, ("policy",))
    best_uni = max([r for r in pairwise_rows if str(r["selector"]).startswith("universal")], key=lambda r: float(r.get("pairwise_accuracy", -1.0))) if pairwise_rows else None
    old_row = next((r for r in pairwise_rows if r["selector"] == "old_frozen_bg_objective_mixed"), None)
    uni_acc = float(best_uni.get("pairwise_accuracy", float("nan"))) if best_uni else float("nan")
    old_acc = float(old_row.get("pairwise_accuracy", float("nan"))) if old_row else float("nan")
    if not pairs:
        verdict = "INSUFFICIENT"
    elif math.isfinite(old_acc) and math.isfinite(uni_acc) and uni_acc + 0.02 >= old_acc:
        verdict = "MATCHES_OR_BEATS_OLD_TAPS"
    elif math.isfinite(uni_acc) and uni_acc >= 0.50:
        verdict = "SMALL_DEGRADATION"
    elif math.isfinite(uni_acc):
        verdict = "LARGE_DEGRADATION"
    else:
        verdict = "INCOMPATIBLE"
    payload = {
        "BG_UNIVERSAL_OLD_CONTEXT_EVAL_VERDICT": verdict,
        "verdict": verdict,
        "dataset_verdict": old_payload.get("verdict"),
        "training_verdict": train_payload.get("verdict"),
        "pairwise_rows": pairwise_rows,
        "group_eval_rows": group_rows_eval,
        "metrics_by_policy": metrics,
        "domain_breakdown": aggregate_metric_rows(group_rows_eval, ("domain", "policy")),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json_local(OLD_CONTEXT_EVAL_JSON, payload)
    write_csv(OLD_CONTEXT_EVAL_CSV, group_rows_eval + pairwise_rows)
    lines = ["# Universal Old-Context Evaluation", "", f"BG_UNIVERSAL_OLD_CONTEXT_EVAL_VERDICT = {verdict}", "", "## Pairwise", ""]
    lines.extend(md_table(pairwise_rows, ["selector", "pairwise_accuracy", "pair_count", "config"]))
    write_md(OLD_CONTEXT_EVAL_MD, lines)
    print(f"BG_UNIVERSAL_OLD_CONTEXT_EVAL_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "INSUFFICIENT" else 1


def hidden_group_eval_rows(heads: dict[str, dict[str, Any] | None], device: torch.device) -> list[dict[str, Any]]:
    rows = []
    for source_id, source_rows in (("branch_generator_v1", load_generator_rows()), ("quota_v4", load_v4_branch_rows())):
        selected = []
        for row in source_rows:
            if source_id == "branch_generator_v1":
                ok = primary_safe_generator_row(row)
            else:
                try:
                    ok = stable_v2_row(row) and str(row.get("split")) == "heldout" and float(row.get("alpha", 999.0)) <= 0.010001
                except Exception:
                    ok = False
            if ok and str(row.get("split")) == "heldout":
                norm = dict(row)
                norm["reward"] = float(row_reward(norm, "deterministic"))
                selected.append(norm)
        for gid, vals in group_rows(selected).items():
            vals = sorted(vals, key=lambda row: int(row.get("branch_id", -1)))
            if len(vals) < 2 or len({float(v.get("reward", v.get("deterministic_reward", 0.0))) for v in vals}) < 2:
                continue
            candidate_rows = []
            for row in vals:
                cand = branch_row_to_candidate(row, source_id)
                if cand:
                    candidate_rows.append(cand)
            if len(candidate_rows) < 2:
                continue
            base = {
                "source_id": source_id,
                "group_id": gid,
                "task_id": vals[0].get("task_id"),
                "domain": domain_bucket(vals[0].get("domain")),
                "branch_point": vals[0].get("branch_point"),
                "generator_method": vals[0].get("generator_method", vals[0].get("recipe_source")),
                "group_size": len(candidate_rows),
                "behaviorally_diverse": True,
            }
            rewards = [float(r.get("reward", 0.0)) for r in candidate_rows]
            out_policies = [
                ("clean_branch_baseline", [0]),
                ("random_top1", []),
            ]
            for policy, ranking in out_policies:
                if policy == "random_top1":
                    oracle_count = sum(1 for r in rewards if r == max(rewards))
                    rows.append({**base, "policy": policy, "top_k": 1, "top1_success": oracle_count / len(rewards), "topk_oracle_coverage": 1 / len(rewards), "reward_mean": mean(rewards), "regret": max(rewards) - mean(rewards)})
                else:
                    rows.append({**base, "policy": policy, "top_k": 1, **group_metric(candidate_rows, ranking, 1)})
            score_indices = [(i, vals[i].get("old_frozen_tap_score", vals[i].get("tap_margin_sum"))) for i in range(len(vals))]
            if all(v is not None for _, v in score_indices):
                ranking = [i for i, _ in sorted(score_indices, key=lambda item: -float(item[1]))]
                rows.append({**base, "policy": "old_frozen_bg_pairwise", "top_k": 1, **group_metric(candidate_rows, ranking, 1)})
                rows.append({**base, "policy": "old_frozen_bg_top2", "top_k": 2, **group_metric(candidate_rows, ranking, 2)})
            for label in ("v4_hidden_origin", "generator_v1_selector", "universal_balanced", "universal_no_bridge", "hidden_branch_only", "old_content_only"):
                ranking = rank_group_with_head(candidate_rows, heads.get(label), device)
                if not ranking:
                    continue
                rows.append({**base, "policy": label, "top_k": 1, **group_metric(candidate_rows, ranking, 1), "config": heads[label]["config"]})
                for k in (2, 3):
                    rows.append({**base, "policy": f"{label}_top{k}", "top_k": k, **group_metric(candidate_rows, ranking, k), "config": heads[label]["config"]})
    return rows


def run_hidden_branch_eval() -> int:
    ensure_root()
    started = time.time()
    heads = best_heads_from_training()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = hidden_group_eval_rows(heads, device)
    metrics = aggregate_metric_rows(rows, ("policy",))
    uni = metrics.get("universal_balanced") or {}
    best_branch = max([metrics.get(k, {}) for k in ("v4_hidden_origin", "generator_v1_selector", "hidden_branch_only")], key=lambda r: float(r.get("top1_success", -1.0))) if metrics else {}
    uni_top1 = float(uni.get("top1_success", float("nan")))
    branch_top1 = float(best_branch.get("top1_success", float("nan")))
    uni_top2 = float((metrics.get("universal_balanced_top2") or {}).get("topk_oracle_coverage", float("nan")))
    branch_top2 = max(float((metrics.get(k) or {}).get("topk_oracle_coverage", float("nan"))) for k in ("v4_hidden_origin_top2", "generator_v1_selector_top2", "hidden_branch_only_top2")) if metrics else float("nan")
    if not rows:
        verdict = "INSUFFICIENT"
    elif math.isfinite(uni_top1) and math.isfinite(branch_top1) and uni_top1 + 0.02 >= branch_top1:
        verdict = "MATCHES_OR_BEATS_BRANCH_TAPS"
    elif math.isfinite(uni_top2) and math.isfinite(branch_top2) and uni_top2 + 0.02 >= branch_top2:
        verdict = "TOPK_ONLY"
    elif math.isfinite(uni_top1) and uni_top1 >= 0.50 * max(branch_top1, 1e-6):
        verdict = "SMALL_DEGRADATION"
    else:
        verdict = "LARGE_DEGRADATION"
    payload = {
        "BG_UNIVERSAL_HIDDEN_BRANCH_EVAL_VERDICT": verdict,
        "verdict": verdict,
        "rows": rows,
        "metrics_by_policy": metrics,
        "by_source": aggregate_metric_rows(rows, ("source_id", "policy")),
        "by_branch_point": aggregate_metric_rows(rows, ("branch_point", "policy")),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json_local(HIDDEN_BRANCH_EVAL_JSON, payload)
    write_csv(HIDDEN_BRANCH_EVAL_CSV, rows)
    summary_rows = [{"policy": k, **{m: rate(v) for m, v in val.items() if isinstance(v, (int, float)) and m in {"top1_success", "topk_oracle_coverage", "regret", "n"}}} for k, val in metrics.items()]
    lines = ["# Universal Hidden-Branch Evaluation", "", f"BG_UNIVERSAL_HIDDEN_BRANCH_EVAL_VERDICT = {verdict}", "", "## Metrics", ""]
    lines.extend(md_table(summary_rows, ["policy", "n", "top1_success", "topk_oracle_coverage", "regret"]))
    write_md(HIDDEN_BRANCH_EVAL_MD, lines)
    print(f"BG_UNIVERSAL_HIDDEN_BRANCH_EVAL_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "INSUFFICIENT" else 1


def run_bridge_eval() -> int:
    ensure_root()
    started = time.time()
    payload_bridge = load_pt(BRIDGE_PT, {}) or {}
    pairs = [p for p in payload_bridge.get("pairs") or [] if p.get("split") == "heldout"]
    heads = best_heads_from_training()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [{"selector": "random", "pairwise_accuracy": 0.5, "pair_count": len(pairs), "config": "baseline"}]
    rows.append({"selector": "old_frozen_score_if_available", **pairwise_acc_from_scores(pairs, "old_frozen_tap_score_preferred", "old_frozen_tap_score_rejected")})
    for label in ("universal_balanced", "universal_no_bridge", "old_content_only", "hidden_branch_only", "bridge_only"):
        rows.append({"selector": label, **pairwise_acc_for_head(heads.get(label), pairs, device)})
    uni = next((r for r in rows if r["selector"] == "universal_balanced"), {})
    nobridge = next((r for r in rows if r["selector"] == "universal_no_bridge"), {})
    hidden = next((r for r in rows if r["selector"] == "hidden_branch_only"), {})
    uni_acc = float(uni.get("pairwise_accuracy", float("nan")))
    best_base = max(float((row or {}).get("pairwise_accuracy", float("nan"))) for row in (nobridge, hidden) if math.isfinite(float((row or {}).get("pairwise_accuracy", float("nan"))))) if rows else float("nan")
    if not pairs:
        verdict = "INSUFFICIENT"
    elif math.isfinite(uni_acc) and math.isfinite(best_base) and uni_acc > best_base + 0.02:
        verdict = "BRIDGE_READY"
    elif math.isfinite(uni_acc) and uni_acc > 0.50:
        verdict = "WEAK_BRIDGE"
    elif math.isfinite(uni_acc):
        verdict = "NO_BRIDGE_SIGNAL"
    else:
        verdict = "DATA_LIMITED"
    bridge_breakdown = {bt: pairwise_acc_for_head(heads.get("universal_balanced"), [p for p in pairs if p.get("bridge_type") == bt], device) for bt in sorted({str(p.get("bridge_type")) for p in pairs})}
    payload = {
        "BG_UNIVERSAL_BRIDGE_EVAL_VERDICT": verdict,
        "verdict": verdict,
        "rows": rows,
        "heldout_pairs": len(pairs),
        "bridge_type_breakdown": bridge_breakdown,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json_local(BRIDGE_EVAL_JSON, payload)
    write_csv(BRIDGE_EVAL_CSV, rows)
    lines = ["# Universal Bridge Evaluation", "", f"BG_UNIVERSAL_BRIDGE_EVAL_VERDICT = {verdict}", "", "## Pairwise", ""]
    lines.extend(md_table(rows, ["selector", "pairwise_accuracy", "pair_count", "config"]))
    write_md(BRIDGE_EVAL_MD, lines)
    print(f"BG_UNIVERSAL_BRIDGE_EVAL_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "INSUFFICIENT" else 1


def run_layerwise_pruning_sim() -> int:
    ensure_root()
    started = time.time()
    heads = best_heads_from_training()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_rows = hidden_group_eval_rows(heads, device)
    rows: list[dict[str, Any]] = []
    for row in base_rows:
        if row.get("policy") in {"random_top1"}:
            continue
        for layer in ("L24", "L36", "L47"):
            sim = dict(row)
            sim["tap_layer"] = layer
            sim["policy"] = f"{row.get('policy')}@{layer}"
            sim["false_prune_rate"] = 1.0 - float(row.get("oracle_retention", row.get("topk_oracle_coverage", 0.0)))
            sim["waste_discarded_proxy"] = max(0, int(row.get("group_size", 0)) - int(row.get("top_k", 1)))
            rows.append(sim)
    metrics = aggregate_metric_rows(rows, ("policy",))
    uni2 = [v for k, v in metrics.items() if "universal_balanced_top2" in k]
    uni3 = [v for k, v in metrics.items() if "universal_balanced_top3" in k]
    best_uni_ret = max([float(v.get("oracle_retention", v.get("topk_oracle_coverage", 0.0))) for v in uni2 + uni3] or [float("nan")])
    old_new = [v for k, v in metrics.items() if "v4_hidden_origin" in k or "old_frozen_bg" in k]
    best_base_ret = max([float(v.get("oracle_retention", v.get("topk_oracle_coverage", 0.0))) for v in old_new] or [float("nan")])
    if not rows:
        verdict = "INSUFFICIENT"
    elif math.isfinite(best_uni_ret) and best_uni_ret >= 0.80 and best_uni_ret >= best_base_ret - 0.02:
        verdict = "UNIVERSAL_PRUNING_READY"
    elif math.isfinite(best_uni_ret) and best_uni_ret >= 0.65:
        verdict = "TOPK_SURVIVAL_ONLY"
    elif math.isfinite(best_base_ret) and best_base_ret > best_uni_ret + 0.05:
        verdict = "OLD_NEW_COMPOSITE_BEST"
    else:
        verdict = "TOO_MANY_FALSE_PRUNES"
    payload = {
        "BG_UNIVERSAL_LAYERWISE_PRUNING_VERDICT": verdict,
        "verdict": verdict,
        "rows": rows,
        "metrics_by_policy": metrics,
        "simulation_note": "offline layer-wise pruning proxy; no action steering or Ouro execution",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json_local(PRUNING_SIM_JSON, payload)
    write_csv(PRUNING_SIM_CSV, rows)
    lines = ["# Universal Layer-Wise Pruning Simulation", "", f"BG_UNIVERSAL_LAYERWISE_PRUNING_VERDICT = {verdict}", "", "This is an offline top-k retention simulation only; it does not execute action steering."]
    write_md(PRUNING_SIM_MD, lines)
    print(f"BG_UNIVERSAL_LAYERWISE_PRUNING_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "INSUFFICIENT" else 1


def run_domain_generalization() -> int:
    ensure_root()
    started = time.time()
    old_eval = load_json(OLD_CONTEXT_EVAL_JSON, {}) or {}
    hidden_eval = load_json(HIDDEN_BRANCH_EVAL_JSON, {}) or {}
    bridge_eval = load_json(BRIDGE_EVAL_JSON, {}) or {}
    old_rows = list(old_eval.get("group_eval_rows") or [])
    hidden_rows = list(hidden_eval.get("rows") or [])
    domain_rows = []
    domain_rows.extend(old_rows)
    domain_rows.extend(hidden_rows)
    by_domain_policy = aggregate_metric_rows(domain_rows, ("domain", "policy"))
    domains = sorted({str(row.get("domain")) for row in domain_rows if row.get("domain")})
    has_coding_pairs = "coding" in domains and any(str(row.get("policy")).startswith("universal") for row in domain_rows if row.get("domain") == "coding")
    old_verdict = old_eval.get("verdict")
    hidden_verdict = hidden_eval.get("verdict")
    if old_verdict == "MATCHES_OR_BEATS_OLD_TAPS" and hidden_verdict in {"MATCHES_OR_BEATS_BRANCH_TAPS", "TOPK_ONLY"} and has_coding_pairs:
        verdict = "MULTIDOMAIN_READY"
    elif old_verdict in {"MATCHES_OR_BEATS_OLD_TAPS", "SMALL_DEGRADATION"} and hidden_verdict in {"MATCHES_OR_BEATS_BRANCH_TAPS", "TOPK_ONLY", "SMALL_DEGRADATION"}:
        verdict = "REASONING_SCIENCE_ONLY"
    elif old_verdict in {"LARGE_DEGRADATION"}:
        verdict = "CONTENT_DEGRADES"
    elif hidden_verdict in {"LARGE_DEGRADATION"}:
        verdict = "BRANCH_DEGRADES"
    else:
        verdict = "DATA_LIMITED"
    payload = {
        "BG_UNIVERSAL_DOMAIN_GENERALIZATION_VERDICT": verdict,
        "verdict": verdict,
        "domains_seen": domains,
        "coding_pair_support": has_coding_pairs,
        "old_context_verdict": old_verdict,
        "hidden_branch_verdict": hidden_verdict,
        "bridge_verdict": bridge_eval.get("verdict"),
        "by_domain_policy": by_domain_policy,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json_local(DOMAIN_JSON, payload)
    lines = ["# Universal Domain Generalization", "", f"BG_UNIVERSAL_DOMAIN_GENERALIZATION_VERDICT = {verdict}", "", f"- domains_seen: `{domains}`", f"- coding_pair_support: `{has_coding_pairs}`", "Cached coding features had no non-tie within-task labels, so coding remains coverage-limited."]
    write_md(DOMAIN_MD, lines)
    print(f"BG_UNIVERSAL_DOMAIN_GENERALIZATION_VERDICT = {verdict}", flush=True)
    return 0


def collect_reference_heads() -> list[tuple[str, dict[str, Any]]]:
    refs: list[tuple[str, dict[str, Any]]] = []
    for label, path, variant in (
        ("v4_hidden_origin", HEADS_V4_PT, "v4_only_primary_safe"),
        ("generator_v1_selector", BGV1_SELECTOR_HEADS_PT, "generator_v1_only_primary_safe"),
    ):
        head = best_primary_head(path, variant)
        if head:
            refs.append((label, head))
    return refs


def run_geometry_analysis() -> int:
    ensure_root()
    started = time.time()
    trained = load_pt(UNIVERSAL_HEADS_PT, {}) or {}
    heads = [h for h in trained.get("heads") or [] if h.get("flip_diagnostics", {}).get("passes")]
    refs = collect_reference_heads()
    rows = []
    for head in heads:
        state = head.get("state_dict") or {}
        vec = state.get("linear.weight")
        if not isinstance(vec, torch.Tensor):
            continue
        vec = vec.detach().cpu().flatten().to(torch.float32)
        for label, ref in refs:
            if ref.get("config") != head.get("config"):
                continue
            rvec = (ref.get("state_dict") or {}).get("linear.weight")
            if isinstance(rvec, torch.Tensor):
                rows.append(
                    {
                        "universal_variant": head.get("variant"),
                        "config": head.get("config"),
                        "architecture": head.get("architecture"),
                        "reference": label,
                        "cosine_alignment": cosine(vec, rvec.detach().cpu().flatten().to(torch.float32)),
                        "balanced_validation_score": head.get("metrics", {}).get("balanced_validation_score"),
                    }
                )
    best_alignment = max([abs(float(r.get("cosine_alignment", 0.0))) for r in rows if math.isfinite(float(r.get("cosine_alignment", float("nan"))))] or [float("nan")])
    if not rows:
        verdict = "INCONCLUSIVE"
    elif best_alignment >= 0.95:
        verdict = "OLD_GEOMETRY_CONFIRMED"
    elif best_alignment >= 0.75:
        verdict = "MIXED_SHARED_GEOMETRY"
    else:
        verdict = "BRIDGE_GEOMETRY"
    payload = {
        "BG_UNIVERSAL_TAP_GEOMETRY_VERDICT": verdict,
        "verdict": verdict,
        "alignment_rows": rows,
        "max_abs_reference_alignment": best_alignment,
        "reference_heads": [label for label, _ in refs],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json_local(GEOMETRY_JSON, payload)
    write_md(GEOMETRY_MD, ["# Universal Tap Geometry", "", f"BG_UNIVERSAL_TAP_GEOMETRY_VERDICT = {verdict}", "", f"- max_abs_reference_alignment: `{rate(best_alignment)}`"])
    print(f"BG_UNIVERSAL_TAP_GEOMETRY_VERDICT = {verdict}", flush=True)
    return 0


def verdict(payload: dict[str, Any], key: str, default: str = "INSUFFICIENT") -> str:
    return str(payload.get(key) or payload.get("verdict") or default)


def load_stage_payloads() -> dict[str, dict[str, Any]]:
    return {
        "inventory": load_json(INVENTORY_JSON, {}) or {},
        "old_content": load_json(OLD_CONTENT_JSON, {}) or {},
        "hidden_branch": load_json(HIDDEN_BRANCH_JSON, {}) or {},
        "bridge": load_json(BRIDGE_JSON, {}) or {},
        "expansion": load_json(DATA_EXPANSION_JSON, {}) or {},
        "dataset": load_json(UNIVERSAL_DATASET_JSON, {}) or {},
        "training": load_json(TRAINING_JSON, {}) or {},
        "old_eval": load_json(OLD_CONTEXT_EVAL_JSON, {}) or {},
        "hidden_eval": load_json(HIDDEN_BRANCH_EVAL_JSON, {}) or {},
        "bridge_eval": load_json(BRIDGE_EVAL_JSON, {}) or {},
        "pruning": load_json(PRUNING_SIM_JSON, {}) or {},
        "domain": load_json(DOMAIN_JSON, {}) or {},
        "geometry": load_json(GEOMETRY_JSON, {}) or {},
    }


def universal_status(data: dict[str, dict[str, Any]]) -> str:
    training = verdict(data["training"], "BG_UNIVERSAL_TAP_TRAINING_VERDICT")
    old_eval = verdict(data["old_eval"], "BG_UNIVERSAL_OLD_CONTEXT_EVAL_VERDICT")
    hidden_eval = verdict(data["hidden_eval"], "BG_UNIVERSAL_HIDDEN_BRANCH_EVAL_VERDICT")
    bridge_eval = verdict(data["bridge_eval"], "BG_UNIVERSAL_BRIDGE_EVAL_VERDICT")
    pruning = verdict(data["pruning"], "BG_UNIVERSAL_LAYERWISE_PRUNING_VERDICT")
    domain = verdict(data["domain"], "BG_UNIVERSAL_DOMAIN_GENERALIZATION_VERDICT")
    if training == "READY" and old_eval == "MATCHES_OR_BEATS_OLD_TAPS" and hidden_eval == "MATCHES_OR_BEATS_BRANCH_TAPS" and bridge_eval in {"BRIDGE_READY", "WEAK_BRIDGE"} and pruning == "UNIVERSAL_PRUNING_READY" and domain in {"MULTIDOMAIN_READY", "REASONING_SCIENCE_ONLY"}:
        return "UNIVERSAL_READY"
    if bridge_eval in {"NO_BRIDGE_SIGNAL"}:
        return "FUSION_NEEDED"
    if hidden_eval == "TOPK_ONLY" or pruning == "TOPK_SURVIVAL_ONLY":
        return "TOPK_UNIVERSAL_WEAK"
    if training == "CONTENT_ONLY" or (old_eval in {"MATCHES_OR_BEATS_OLD_TAPS", "SMALL_DEGRADATION"} and hidden_eval == "LARGE_DEGRADATION"):
        return "CONTENT_ONLY"
    if training == "BRANCH_ONLY" or (hidden_eval in {"MATCHES_OR_BEATS_BRANCH_TAPS", "TOPK_ONLY"} and old_eval == "LARGE_DEGRADATION"):
        return "BRANCH_ONLY"
    if pruning in {"OLD_NEW_COMPOSITE_BEST"}:
        return "OLD_NEW_COMPOSITE_NEEDED"
    if old_eval == "LARGE_DEGRADATION" and hidden_eval == "LARGE_DEGRADATION":
        return "NEGATIVE_TRANSFER"
    if any(verdict(data[k], "", "INSUFFICIENT") in {"DATA_LIMITED", "INSUFFICIENT", "INCOMPATIBLE"} for k in ("old_eval", "hidden_eval", "bridge_eval")):
        return "DATA_LIMITED"
    return "NOT_READY"


def top_lines(data: dict[str, dict[str, Any]], status: str) -> list[str]:
    return [
        f"BG_UNIVERSAL_TAP_INVENTORY_VERDICT = {verdict(data['inventory'], 'BG_UNIVERSAL_TAP_INVENTORY_VERDICT')}",
        f"BG_UNIVERSAL_OLD_CONTENT_DATASET_VERDICT = {verdict(data['old_content'], 'BG_UNIVERSAL_OLD_CONTENT_DATASET_VERDICT')}",
        f"BG_UNIVERSAL_HIDDEN_BRANCH_DATASET_VERDICT = {verdict(data['hidden_branch'], 'BG_UNIVERSAL_HIDDEN_BRANCH_DATASET_VERDICT')}",
        f"BG_UNIVERSAL_BRIDGE_DATASET_VERDICT = {verdict(data['bridge'], 'BG_UNIVERSAL_BRIDGE_DATASET_VERDICT')}",
        f"BG_UNIVERSAL_DATA_EXPANSION_VERDICT = {verdict(data['expansion'], 'BG_UNIVERSAL_DATA_EXPANSION_VERDICT')}",
        f"BG_UNIVERSAL_TAP_DATASET_VERDICT = {verdict(data['dataset'], 'BG_UNIVERSAL_TAP_DATASET_VERDICT')}",
        f"BG_UNIVERSAL_TAP_TRAINING_VERDICT = {verdict(data['training'], 'BG_UNIVERSAL_TAP_TRAINING_VERDICT')}",
        f"BG_UNIVERSAL_OLD_CONTEXT_EVAL_VERDICT = {verdict(data['old_eval'], 'BG_UNIVERSAL_OLD_CONTEXT_EVAL_VERDICT')}",
        f"BG_UNIVERSAL_HIDDEN_BRANCH_EVAL_VERDICT = {verdict(data['hidden_eval'], 'BG_UNIVERSAL_HIDDEN_BRANCH_EVAL_VERDICT')}",
        f"BG_UNIVERSAL_BRIDGE_EVAL_VERDICT = {verdict(data['bridge_eval'], 'BG_UNIVERSAL_BRIDGE_EVAL_VERDICT')}",
        f"BG_UNIVERSAL_LAYERWISE_PRUNING_VERDICT = {verdict(data['pruning'], 'BG_UNIVERSAL_LAYERWISE_PRUNING_VERDICT')}",
        f"BG_UNIVERSAL_DOMAIN_GENERALIZATION_VERDICT = {verdict(data['domain'], 'BG_UNIVERSAL_DOMAIN_GENERALIZATION_VERDICT')}",
        f"BG_UNIVERSAL_TAP_GEOMETRY_VERDICT = {verdict(data['geometry'], 'BG_UNIVERSAL_TAP_GEOMETRY_VERDICT')}",
        f"UNIVERSAL_BRANCH_CONTENT_TAP_STATUS = {status}",
    ]


def recommendation(status: str) -> str:
    if status == "UNIVERSAL_READY":
        return "Use the universal tap in a selection-only Phase 2 prototype with L24/L36/L47 top-k candidate survival; do not claim action steering."
    if status == "TOPK_UNIVERSAL_WEAK":
        return "Use universal taps only as top2/top3 survival diagnostics; avoid top1 pruning."
    if status in {"FUSION_NEEDED", "OLD_NEW_COMPOSITE_NEEDED"}:
        return "Build an explicit composite selector rather than forcing a single universal head."
    if status in {"CONTENT_ONLY", "BRANCH_ONLY"}:
        return "Keep old content and hidden-branch roles separated for now."
    if status == "DATA_LIMITED":
        return "Collect missing bridge/domain data, especially non-tie coding/content pairs."
    if status == "NEGATIVE_TRANSFER":
        return "Avoid naive mixed training; use separated heads or gated experts."
    return "Do not treat the universal tap as ready."


def docs_section(data: dict[str, dict[str, Any]], status: str) -> str:
    return "\n".join(
        [
            "## Universal branch-content taps v1 (2026-05-18)",
            "",
            "Universal Branch-Content Taps v1 tested whether one tiny hidden-state pairwise evaluator can cover both old content/candidate selection and same-prefix hidden-origin branch survival. It trained only new standalone tap heads and did not alter Ouro, existing BG taps, registries, wrapper/local-agent routing, or production behavior.",
            "",
            *top_lines(data, status),
            "",
            f"- old_content_counts: `{data['old_content'].get('counts', {})}`",
            f"- hidden_branch_counts: `{data['hidden_branch'].get('counts', {})}`",
            f"- bridge_counts: `{data['bridge'].get('counts', {})}`",
            f"- recommendation: `{recommendation(status)}`",
            "",
            "Readiness requires old-context, hidden-branch, and bridge support. Cached coding features were inspected but had no non-tie within-task labels, so coding remains coverage-limited.",
            "",
        ]
    )


def append_docs(section: str) -> None:
    targets = [
        PROJECT_ROOT / "docs/evaluator/current_state.md",
        PROJECT_ROOT / "docs/evaluator/domain_transfer_ledger.md",
        PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_branch_generator_v1.md",
        PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_quota_v4.md",
        PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_split_salvage.md",
        PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_diversity_v3.md",
        PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_taps.md",
        PROJECT_ROOT / "docs/evaluator/bg_hidden_state_branch_generation.md",
        PROJECT_ROOT / "docs/evaluator/bg_steering_consolidation_2026-05-18.md",
        PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-18_v8.1_routing_locked.md",
    ]
    for path in targets:
        if path.exists():
            with path.open("a", encoding="utf-8") as handle:
                handle.write("\n" + section + "\n")


def run_synthesis() -> int:
    ensure_root()
    started = time.time()
    data = load_stage_payloads()
    status = universal_status(data)
    rec = recommendation(status)
    payload = {
        "top_lines": top_lines(data, status),
        "stage_payloads": data,
        "UNIVERSAL_BRANCH_CONTENT_TAP_STATUS": status,
        "recommended_next": rec,
        "files_created": {
            "inventory": rel(INVENTORY_JSON),
            "old_content_dataset": rel(OLD_CONTENT_PT),
            "hidden_branch_dataset": rel(HIDDEN_BRANCH_PT),
            "bridge_dataset": rel(BRIDGE_PT),
            "universal_dataset": rel(UNIVERSAL_DATASET_PT),
            "universal_heads": rel(UNIVERSAL_HEADS_PT),
            "old_context_eval": rel(OLD_CONTEXT_EVAL_JSON),
            "hidden_branch_eval": rel(HIDDEN_BRANCH_EVAL_JSON),
            "bridge_eval": rel(BRIDGE_EVAL_JSON),
            "layerwise_pruning": rel(PRUNING_SIM_JSON),
            "domain_generalization": rel(DOMAIN_JSON),
            "geometry": rel(GEOMETRY_JSON),
            "summary": rel(SUMMARY_JSON),
            "analysis": rel(ANALYSIS_JSON),
            "docs": rel(DOC_MD),
        },
        "commands_run": [f"venv/bin/python -u utilities/tests/manual/{script}" for script in UNIVERSAL_SCRIPTS],
        "blockers": [v.get("blocker") for v in data.values() if isinstance(v, dict) and v.get("blocker")],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json_local(SUMMARY_JSON, payload)
    write_json_local(ANALYSIS_JSON, payload)
    section = docs_section(data, status)
    summary_lines = ["# Universal Branch-Content Taps V1 Summary", "", *top_lines(data, status), "", f"Recommended next: {rec}", "", "## Files Created", ""]
    summary_lines.extend([f"- {name}: `{path}`" for name, path in payload["files_created"].items()])
    write_md(SUMMARY_MD, summary_lines)
    write_md(ANALYSIS_MD, summary_lines + ["", "## Stage Payloads", "", "See `summary.json` / `analysis.json` for compact machine-readable stage payloads."])
    doc_lines = [
        "# Universal Branch-Content Taps V1",
        "",
        "This experiment tested whether one universal hidden-state pairwise tap family can judge both old content/candidate quality and hidden-origin branch survival.",
        "",
        *top_lines(data, status),
        "",
        "## Motivation",
        "",
        "The old BG taps already judged hidden states, and hidden-origin taps aligned strongly with that old geometry. The universal run tested whether mixed old-content, hidden-branch, and bridge pairs can support a single selection head without replacing production routing.",
        "",
        "## Result",
        "",
        rec,
        "",
        "## Notes",
        "",
        "- No Ouro weights, tokenizer files, checkpoints, old BG taps, or tap registries were modified.",
        "- No wrapper/local-agent or Hunter-Seeker code path was executed.",
        "- Coding cached features were compatible but lacked non-tie within-task labels, so no coding readiness claim is made.",
    ]
    write_md(DOC_MD, doc_lines)
    append_docs(section)
    print(f"UNIVERSAL_BRANCH_CONTENT_TAP_STATUS = {status}", flush=True)
    return 0
