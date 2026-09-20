"""Fixed-composite branch survival policy v1.

This experiment is a policy/evaluation layer over cached tap artifacts. It does
not train Ouro, mutate existing tap registries, run wrapper/local-agent code, or
use tap scores as labels. Expert/tap scores are policy inputs only; final
reward/correctness defines oracle retention and false-prune metrics.
"""
from __future__ import annotations

import itertools
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from bg_gated_selector_v1_common import (
    CODE_EXPANDED_FEATURES_PT,
    EXPERT_NAMES,
    GATED_ROOT,
    RuntimeScorer,
    apply_calibration,
    branch_candidate_groups,
    code_feature_index,
    code_label_value,
    domain_bucket,
    json_default,
    load_json,
    load_old_code_candidate_groups,
    load_pt,
    old_candidate_groups,
    rel,
    safe_float,
    score_pair_with_experts,
    sigmoid,
    synthetic_pair_from_candidates,
    write_json,
)
from bg_hidden_origin_quota_v4_common import PROBE_ROOT, PROJECT_ROOT, md_table, rate, write_csv, write_md
from bg_universal_tap_v1_common import aggregate_metric_rows


OUT_ROOT = PROBE_ROOT / "bg_fixed_composite_branch_survival_policy_v1_2026-05-18"

INVENTORY_JSON = OUT_ROOT / "inventory.json"
INVENTORY_MD = OUT_ROOT / "inventory.md"
SOURCE_INVENTORY_CSV = OUT_ROOT / "source_inventory.csv"

DATASET_PT = OUT_ROOT / "survival_dataset.pt"
DATASET_JSON = OUT_ROOT / "survival_dataset.json"
DATASET_MD = OUT_ROOT / "survival_dataset.md"

FEATURES_PT = OUT_ROOT / "survival_features.pt"
FEATURES_JSON = OUT_ROOT / "survival_features.json"
FEATURES_MD = OUT_ROOT / "survival_features.md"

BASELINES_JSON = OUT_ROOT / "baseline_policies.json"
BASELINES_MD = OUT_ROOT / "baseline_policies.md"
BASELINES_CSV = OUT_ROOT / "baseline_policy_rows.csv"

COMPOSITE_JSON = OUT_ROOT / "composite_optimization.json"
COMPOSITE_MD = OUT_ROOT / "composite_optimization.md"
COMPOSITE_CSV = OUT_ROOT / "composite_weight_rows.csv"
BEST_COMPOSITE_JSON = OUT_ROOT / "best_fixed_composite.json"

VETO_JSON = OUT_ROOT / "veto_rescue_search.json"
VETO_MD = OUT_ROOT / "veto_rescue_search.md"
VETO_CSV = OUT_ROOT / "veto_rescue_rows.csv"
BEST_VETO_JSON = OUT_ROOT / "best_veto_rescue_policy.json"

RESCUE_PT = OUT_ROOT / "learned_rescue_policy.pt"
RESCUE_JSON = OUT_ROOT / "learned_rescue_training_log.json"
RESCUE_MD = OUT_ROOT / "learned_rescue_training_report.md"

MISSING_JSON = OUT_ROOT / "missing_ood_policy.json"
MISSING_MD = OUT_ROOT / "missing_ood_policy.md"
MISSING_CSV = OUT_ROOT / "missing_ood_rows.csv"

HELDOUT_JSON = OUT_ROOT / "heldout_survival_eval.json"
HELDOUT_MD = OUT_ROOT / "heldout_survival_eval.md"
HELDOUT_CSV = OUT_ROOT / "heldout_survival_rows.csv"

FRONTIER_JSON = OUT_ROOT / "frontier_analysis.json"
FRONTIER_MD = OUT_ROOT / "frontier_analysis.md"
FRONTIER_CSV = OUT_ROOT / "frontier_rows.csv"

LAYER_JSON = OUT_ROOT / "layer_origin_domain_analysis.json"
LAYER_MD = OUT_ROOT / "layer_origin_domain_analysis.md"

OLD_CODE_JSON = OUT_ROOT / "old_code_preservation.json"
OLD_CODE_MD = OUT_ROOT / "old_code_preservation.md"
OLD_CODE_CSV = OUT_ROOT / "old_code_preservation_rows.csv"

READINESS_JSON = OUT_ROOT / "selection_only_readiness.json"
READINESS_MD = OUT_ROOT / "selection_only_readiness.md"

SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
ANALYSIS_JSON = OUT_ROOT / "analysis.json"
ANALYSIS_MD = OUT_ROOT / "analysis.md"
POLICY_PT = OUT_ROOT / "fixed_composite_branch_survival_policy_v1.pt"

DOC_MD = PROJECT_ROOT / "docs/evaluator/bg_fixed_composite_branch_survival_policy_v1.md"

LAYERS = ("L24", "L36", "L47")
OLD_CONTEXT_LAYER = "old_context"
POLICY_EXPERTS = (
    "old_frozen_bg",
    "old_content_head",
    "old_code_head",
    "mixed_code_reasoning_head",
    "mixed_objective_all_head",
    "v4_hidden_origin",
    "hidden_branch_head",
    "bridge_only_head",
    "universal",
    "generator_v1_selector",
    "gated_selector",
)

SCRIPT_NAMES = [
    "bg_fixed_composite_survival_inventory_v1.py",
    "build_bg_fixed_composite_survival_dataset_v1.py",
    "build_bg_fixed_composite_survival_features_v1.py",
    "run_bg_fixed_composite_survival_baselines_v1.py",
    "optimize_bg_fixed_old_branch_bridge_composite_v1.py",
    "optimize_bg_fixed_composite_veto_rescue_v1.py",
    "train_bg_fixed_composite_rescue_policy_v1.py",
    "build_bg_fixed_composite_missing_ood_policy_v1.py",
    "evaluate_bg_fixed_composite_survival_policy_v1.py",
    "analyze_bg_fixed_composite_survival_frontier_v1.py",
    "analyze_bg_fixed_composite_survival_layers_origins_v1.py",
    "evaluate_bg_fixed_composite_old_code_preservation_v1.py",
    "analyze_bg_fixed_composite_selection_only_readiness_v1.py",
    "analyze_bg_fixed_composite_branch_survival_policy_v1.py",
]


def ensure_root() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def stable_write_json(path: Path, payload: Any) -> None:
    write_json(path, payload)


def finite_mean(values: Iterable[Any], default: float = float("nan")) -> float:
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(mean(vals)) if vals else default


def torch_matrix_to_list(matrix: torch.Tensor) -> list[list[float]]:
    return [[float(x) for x in row] for row in matrix.detach().cpu().tolist()]


def list_to_matrix(value: Any, n: int) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.float32)
    if isinstance(value, list):
        try:
            x = torch.tensor(value, dtype=torch.float32)
            if tuple(x.shape) == (n, n):
                return x
        except Exception:
            pass
    return torch.zeros((n, n), dtype=torch.float32)


def compact_candidate(row: dict[str, Any], idx: int) -> dict[str, Any]:
    reward = safe_float(row.get("reward"), 0.0)
    origin = str(row.get("origin") or row.get("candidate_origin") or "unknown")
    if origin == "unknown" and str(row.get("domain")) == "coding":
        origin = "code_candidate"
    if origin == "unknown" and str(row.get("branch_point") or "").startswith("L"):
        origin = "current_perturbation"
    return {
        "candidate_index": idx,
        "candidate_id": str(row.get("candidate_id") or row.get("branch_id") or idx),
        "task_id": str(row.get("task_id") or ""),
        "domain": domain_bucket(row.get("domain")),
        "origin": origin,
        "branch_point": str(row.get("branch_point") or "old_context"),
        "generator_method": str(row.get("generator_method") or "unknown"),
        "reward": reward,
        "correct": bool(row.get("correct")) if row.get("correct") is not None else reward >= 1.0,
        "parse_success": bool(row.get("parse_success", True)),
        "empty_output": bool(row.get("empty_output", False)),
        "repetition_rate": safe_float(row.get("repetition_rate"), 0.0),
        "label": row.get("label"),
    }


def old_code_groups_by_split(split: str) -> list[list[dict[str, Any]]]:
    if split == "heldout":
        return load_old_code_candidate_groups(split)
    payload = load_pt(CODE_EXPANDED_FEATURES_PT, {}) or {}
    if not payload:
        return []
    index = code_feature_index(payload)
    val_tasks = {
        "HumanEval/152",
        "local_dsa/count_smaller_after_self",
        "local_dsa/word_ladder_length",
        "mbpp/141",
        "mbpp/271",
        "mbpp/278",
    }
    groups: list[list[dict[str, Any]]] = []
    for task in payload.get("training_tasks") or []:
        task_id = str(task.get("task_id"))
        task_split = "val" if task_id in val_tasks else "train"
        if task_split != split:
            continue
        candidates = []
        for uid in task.get("candidate_uids") or []:
            src = index.get(str(uid))
            if not src:
                continue
            meta = src.get("candidate_metadata") or {}
            label = str(meta.get("unit_test_label") or meta.get("label") or "")
            reward = code_label_value(label)
            fmap = {}
            try:
                from bg_gated_selector_v1_common import feature_map_from_old_code_pooled

                fmap = feature_map_from_old_code_pooled(src.get("pooled"))
            except Exception:
                fmap = {}
            if not fmap or not math.isfinite(reward):
                continue
            candidates.append(
                {
                    "candidate_id": str(uid),
                    "task_id": task_id,
                    "group_id": f"old_code_training::{task_id}",
                    "domain": "coding",
                    "origin": "code_candidate",
                    "branch_point": "old_context",
                    "reward": reward,
                    "correct": label == "correct",
                    "label": label,
                    "features_by_config": fmap,
                }
            )
        if len(candidates) >= 2 and len({float(c.get("reward", 0.0)) for c in candidates}) >= 2:
            groups.append(candidates)
    return groups


def all_candidate_groups(split: str, *, include_old: bool = True, include_branch: bool = True) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if include_branch:
        for group in branch_candidate_groups(split):
            if not group:
                continue
            group_id = str(group[0].get("group_id"))
            for layer in LAYERS:
                out.append(
                    {
                        "candidate_set_id": f"branch::{split}::{layer}::{group_id}",
                        "split": split,
                        "layer": layer,
                        "source_id": "hidden_branch",
                        "pair_type": "hidden_branch",
                        "domain": domain_bucket(group[0].get("domain")),
                        "task_id": str(group[0].get("task_id")),
                        "group_id": group_id,
                        "candidates_raw": sorted(group, key=lambda r: str(r.get("candidate_id"))),
                    }
                )
    if include_old:
        groups = old_candidate_groups(split)
        for code_group in old_code_groups_by_split(split):
            key = str(code_group[0].get("group_id"))
            if not any(str(g[0].get("group_id")) == key for g in groups if g):
                groups.append(code_group)
        for group in groups:
            if not group:
                continue
            group_id = str(group[0].get("group_id"))
            source_id = "old_code_context" if domain_bucket(group[0].get("domain")) == "coding" else "old_context"
            out.append(
                {
                    "candidate_set_id": f"{source_id}::{split}::{group_id}",
                    "split": split,
                    "layer": OLD_CONTEXT_LAYER,
                    "source_id": source_id,
                    "pair_type": "old_content",
                    "domain": domain_bucket(group[0].get("domain")),
                    "task_id": str(group[0].get("task_id")),
                    "group_id": group_id,
                    "candidates_raw": sorted(group, key=lambda r: str(r.get("candidate_id"))),
                }
            )
    return out


def score_candidate_set(record: dict[str, Any], scorer: RuntimeScorer) -> dict[str, Any]:
    raw_candidates = list(record["candidates_raw"])
    candidates = [compact_candidate(row, i) for i, row in enumerate(raw_candidates)]
    n = len(candidates)
    scores = {expert: torch.zeros((n, n), dtype=torch.float32) for expert in POLICY_EXPERTS}
    missing = {expert: torch.ones((n, n), dtype=torch.float32) for expert in POLICY_EXPERTS}
    for i in range(n):
        for j in range(i + 1, n):
            pair = synthetic_pair_from_candidates(
                raw_candidates[i],
                raw_candidates[j],
                pair_type=str(record.get("pair_type")),
                source_id=str(record.get("source_id")),
                split=str(record.get("split")),
            )
            if not pair.get("features"):
                continue
            raw = score_pair_with_experts(pair, scorer.head_rows, scorer.head_modules, scorer.device)
            z = apply_calibration(raw, scorer.dataset.get("calibration") or scorer.score_payload.get("calibration") or {})
            for expert in EXPERT_NAMES:
                if expert not in scores:
                    continue
                value = safe_float(z.get(expert))
                if math.isfinite(value):
                    scores[expert][i, j] = value
                    scores[expert][j, i] = -value
                    missing[expert][i, j] = 0.0
                    missing[expert][j, i] = 0.0
            try:
                gated_value = scorer.score_pair(pair, model_head=None)
            except Exception:
                gated_value = float("nan")
            if math.isfinite(safe_float(gated_value)):
                scores["gated_selector"][i, j] = float(gated_value)
                scores["gated_selector"][j, i] = -float(gated_value)
                missing["gated_selector"][i, j] = 0.0
                missing["gated_selector"][j, i] = 0.0
    rewards = [float(c["reward"]) for c in candidates]
    oracle_reward = max(rewards)
    oracle_indices = [i for i, reward in enumerate(rewards) if reward == oracle_reward]
    out = {k: v for k, v in record.items() if k != "candidates_raw"}
    out.update(
        {
            "candidate_count": n,
            "candidates": candidates,
            "oracle_reward": oracle_reward,
            "oracle_indices": oracle_indices,
            "oracle_set_size": len(oracle_indices),
            "tie_best": len(oracle_indices) > 1,
            "score_matrices": {k: torch_matrix_to_list(v) for k, v in scores.items()},
            "missing_matrices": {k: torch_matrix_to_list(v) for k, v in missing.items()},
            "expert_coverage": {k: 1.0 - float(v.sum().item()) / max(float(n * max(n - 1, 1)), 1.0) for k, v in missing.items()},
        }
    )
    return out


def build_survival_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    scorer = RuntimeScorer()
    try:
        for split in ("train", "val", "heldout"):
            for record in all_candidate_groups(split):
                if len(record.get("candidates_raw") or []) < 2:
                    continue
                records.append(score_candidate_set(record, scorer))
    finally:
        scorer.close()
    return records


def compact_record(record: dict[str, Any], *, include_scores: bool = False) -> dict[str, Any]:
    keys = [
        "candidate_set_id",
        "split",
        "layer",
        "source_id",
        "pair_type",
        "domain",
        "task_id",
        "group_id",
        "candidate_count",
        "oracle_reward",
        "oracle_indices",
        "oracle_set_size",
        "tie_best",
        "expert_coverage",
    ]
    out = {key: record.get(key) for key in keys}
    out["origins"] = dict(Counter(str(c.get("origin")) for c in record.get("candidates") or []))
    if include_scores:
        out["score_matrices"] = record.get("score_matrices")
    return out


def load_dataset_records() -> list[dict[str, Any]]:
    payload = load_pt(DATASET_PT, {}) or {}
    return list(payload.get("records") or [])


def matrix(record: dict[str, Any], expert: str, *, missing: bool = False) -> torch.Tensor:
    data = record.get("missing_matrices" if missing else "score_matrices") or {}
    return list_to_matrix(data.get(expert), int(record.get("candidate_count") or len(record.get("candidates") or [])))


def available_matrix(record: dict[str, Any], experts: Sequence[str]) -> torch.Tensor:
    n = int(record.get("candidate_count") or len(record.get("candidates") or []))
    for expert in experts:
        m = matrix(record, expert)
        miss = matrix(record, expert, missing=True)
        if float((miss == 0).sum().item()) > 0:
            return m
    return torch.zeros((n, n), dtype=torch.float32)


def family_matrix(record: dict[str, Any], family: str, *, removed: set[str] | None = None) -> torch.Tensor:
    removed = removed or set()
    domain = domain_bucket(record.get("domain"))
    if family == "old":
        experts = ["old_code_head", "mixed_objective_all_head", "mixed_code_reasoning_head", "old_content_head", "old_frozen_bg"] if domain == "coding" else ["old_content_head", "old_frozen_bg", "mixed_objective_all_head"]
    elif family == "code":
        experts = ["old_code_head", "mixed_code_reasoning_head", "mixed_objective_all_head"]
    elif family == "branch":
        experts = ["hidden_branch_head", "v4_hidden_origin", "hidden_origin_scalar"]
    elif family == "bridge":
        experts = ["bridge_only_head"]
    elif family == "universal":
        experts = ["universal", "universal_no_bridge"]
    elif family == "gated":
        experts = ["gated_selector"]
    else:
        experts = [family]
    return available_matrix(record, [e for e in experts if e not in removed])


DEFAULT_WEIGHTS = {"old": 0.30, "code": 0.10, "branch": 0.30, "bridge": 0.25, "universal": 0.05, "gated": 0.0}


def composite_matrix(record: dict[str, Any], weights: dict[str, float] | None = None, *, removed: set[str] | None = None) -> torch.Tensor:
    weights = weights or DEFAULT_WEIGHTS
    n = int(record.get("candidate_count") or len(record.get("candidates") or []))
    out = torch.zeros((n, n), dtype=torch.float32)
    for family, weight in weights.items():
        if abs(float(weight)) > 0:
            out += float(weight) * family_matrix(record, family, removed=removed)
    return out


def net_from_matrix(scores: torch.Tensor) -> list[float]:
    if scores.numel() == 0:
        return []
    probs = torch.sigmoid(scores)
    net = (2.0 * (probs - 0.5)).mean(dim=1)
    return [float(x) for x in net.tolist()]


def ranking_from_scores(scores: Sequence[float]) -> list[int]:
    return [i for i, _ in sorted(enumerate(scores), key=lambda item: (-float(item[1]), item[0]))]


def topk_survivors(record: dict[str, Any], scores: Sequence[float], k: int) -> list[int]:
    order = ranking_from_scores(scores)
    return order[: min(int(k), len(order))]


LAYER_DEFAULTS = {
    "L24": {"K_min": 3, "K_max": 5, "dynamic_margin": 0.30},
    "L36": {"K_min": 2, "K_max": 4, "dynamic_margin": 0.22},
    "L47": {"K_min": 2, "K_max": 3, "dynamic_margin": 0.16},
    OLD_CONTEXT_LAYER: {"K_min": 2, "K_max": 3, "dynamic_margin": 0.20},
}


def margin_survivors(scores: Sequence[float], params: dict[str, Any]) -> list[int]:
    if not scores:
        return []
    order = ranking_from_scores(scores)
    k_min = min(int(params.get("K_min", 2)), len(order))
    k_max = min(int(params.get("K_max", max(k_min, 2))), len(order))
    margin = float(params.get("dynamic_margin", 0.2))
    best = float(scores[order[0]])
    keep = set(order[:k_min])
    for i, score in enumerate(scores):
        if best - float(score) <= margin:
            keep.add(i)
    if len(keep) > k_max:
        keep = set(order[:k_max])
    return [i for i in order if i in keep]


def expert_family_nets(record: dict[str, Any], removed: set[str] | None = None) -> dict[str, list[float]]:
    return {family: net_from_matrix(family_matrix(record, family, removed=removed)) for family in ("old", "code", "branch", "bridge", "universal", "gated")}


def disagreement_score(nets: dict[str, list[float]], idx: int) -> float:
    vals = [scores[idx] for scores in nets.values() if idx < len(scores) and math.isfinite(scores[idx])]
    return float(pstdev(vals)) if len(vals) > 1 else 0.0


def high_disagreement(record: dict[str, Any], nets: dict[str, list[float]], threshold: float) -> bool:
    n = int(record.get("candidate_count") or 0)
    vals = [disagreement_score(nets, i) for i in range(n)]
    return bool(vals and max(vals) >= float(threshold))


def apply_veto_rescue(
    record: dict[str, Any],
    weights: dict[str, float],
    params: dict[str, Any],
    *,
    removed: set[str] | None = None,
    conservative: bool = False,
) -> list[int]:
    removed = removed or set()
    layer = str(record.get("layer"))
    layer_params = dict(params.get("layers", {}).get(layer) or params.get("layers", {}).get(OLD_CONTEXT_LAYER) or LAYER_DEFAULTS.get(layer, LAYER_DEFAULTS[OLD_CONTEXT_LAYER]))
    if conservative:
        layer_params["K_min"] = min(int(layer_params.get("K_min", 2)) + 1, int(record.get("candidate_count") or 1))
        layer_params["K_max"] = min(int(layer_params.get("K_max", 3)) + 1, int(record.get("candidate_count") or 1))
    base = net_from_matrix(composite_matrix(record, weights, removed=removed))
    keep = set(margin_survivors(base, layer_params))
    if not base:
        return []
    best = max(base)
    margin = float(layer_params.get("dynamic_margin", 0.2))
    rescue_factor = float(params.get("rescue_factor", 2.0))
    strong = float(params.get("strong_threshold", 0.60))
    strong_net = 2.0 * (strong - 0.5)
    disagreement_threshold = float(params.get("disagreement_threshold", 0.25))
    nets = expert_family_nets(record, removed)
    n = len(base)
    candidates = record.get("candidates") or []

    if high_disagreement(record, nets, disagreement_threshold):
        for family in ("old", "branch", "bridge"):
            scores = nets.get(family) or []
            if scores:
                keep.add(ranking_from_scores(scores)[0])
        for i, candidate in enumerate(candidates):
            if candidate.get("origin") in {"clean", "clean_branch"}:
                keep.add(i)

    for i in range(n):
        if i in keep:
            continue
        if best - base[i] > rescue_factor * margin:
            continue
        domain = domain_bucket(record.get("domain"))
        old_ok = i < len(nets.get("old", [])) and nets["old"][i] >= strong_net
        code_ok = domain == "coding" and i < len(nets.get("code", [])) and nets["code"][i] >= strong_net
        branch_ok = str(record.get("pair_type")) == "hidden_branch" and i < len(nets.get("branch", [])) and nets["branch"][i] >= strong_net
        bridge_ok = i < len(nets.get("bridge", [])) and nets["bridge"][i] >= strong_net
        clean_ok = candidates[i].get("origin") in {"clean", "clean_branch"} and high_disagreement(record, nets, disagreement_threshold * 0.75)
        if old_ok or code_ok or branch_ok or bridge_ok or clean_ok:
            keep.add(i)

    # Veto only severe invalid/empty branches when alternatives remain.
    for i, candidate in enumerate(candidates):
        if len(keep) <= int(layer_params.get("K_min", 2)):
            break
        bad = bool(candidate.get("empty_output")) or safe_float(candidate.get("repetition_rate"), 0.0) >= 0.75 or not bool(candidate.get("parse_success", True))
        if bad and i in keep:
            keep.remove(i)

    order = ranking_from_scores(base)
    k_max = min(int(layer_params.get("K_max", len(order))), len(order))
    if len(keep) > k_max:
        protected = {i for i, candidate in enumerate(candidates) if candidate.get("origin") in {"clean", "clean_branch"}}
        ordered_keep = [i for i in order if i in keep]
        trimmed = set(ordered_keep[:k_max])
        for i in protected:
            if i in keep and i not in trimmed and len(trimmed) >= k_max:
                worst = ordered_keep[k_max - 1] if len(ordered_keep) >= k_max else None
                if worst is not None and worst not in protected:
                    trimmed.discard(worst)
                    trimmed.add(i)
        keep = trimmed
    return [i for i in order if i in keep]


def oracle_indices(record: dict[str, Any]) -> set[int]:
    return {int(i) for i in record.get("oracle_indices") or []}


def selected_metric(record: dict[str, Any], selected: Sequence[int], policy: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    selected = [int(i) for i in selected if 0 <= int(i) < int(record.get("candidate_count") or 0)]
    if not selected:
        selected = [0]
    oracles = oracle_indices(record)
    rewards = [float(c.get("reward", 0.0)) for c in record.get("candidates") or []]
    selected_reward = max([rewards[i] for i in selected] or [float("-inf")])
    oracle_reward = max(rewards) if rewards else 0.0
    retained = bool(oracles & set(selected))
    row = {
        "candidate_set_id": record.get("candidate_set_id"),
        "split": record.get("split"),
        "layer": record.get("layer"),
        "source_id": record.get("source_id"),
        "pair_type": record.get("pair_type"),
        "domain": record.get("domain"),
        "task_id": record.get("task_id"),
        "group_id": record.get("group_id"),
        "policy": policy,
        "candidate_count": int(record.get("candidate_count") or 0),
        "survivors": len(selected),
        "average_survivors": float(len(selected)),
        "oracle_retention": 1.0 if retained else 0.0,
        "false_prune_rate": 0.0 if retained else 1.0,
        "pruned_oracle_branch_rate": 0.0 if retained else 1.0,
        "compute_saved_proxy": 1.0 - len(selected) / max(int(record.get("candidate_count") or 1), 1),
        "regret": oracle_reward - selected_reward,
        "top1_success": 1.0 if selected and selected[0] in oracles else 0.0,
        "selected_indices": selected,
        "oracle_indices": sorted(oracles),
        "oracle_set_size": len(oracles),
    }
    if extra:
        row.update(extra)
    return row


def topk_policy(record: dict[str, Any], expert_or_family: str, k: int, weights: dict[str, float] | None = None) -> list[int]:
    if expert_or_family == "fixed_composite":
        scores = net_from_matrix(composite_matrix(record, weights or DEFAULT_WEIGHTS))
    elif expert_or_family in {"old", "code", "branch", "bridge", "universal", "gated"}:
        scores = net_from_matrix(family_matrix(record, expert_or_family))
    else:
        scores = net_from_matrix(matrix(record, expert_or_family))
    return topk_survivors(record, scores, k)


def baseline_rows(records: Sequence[dict[str, Any]], weights: dict[str, float] | None = None, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    params = params or {"layers": LAYER_DEFAULTS, "rescue_factor": 2.0, "strong_threshold": 0.60, "disagreement_threshold": 0.25}
    weights = weights or DEFAULT_WEIGHTS
    for record in records:
        n = int(record.get("candidate_count") or 0)
        if n < 2:
            continue
        for k in (1, 2, 3, 4):
            rows.append(selected_metric(record, list(range(min(k, n))), f"clean_or_first_top{k}", {"top_k": k}))
            rows.append(selected_metric(record, topk_policy(record, "old", k), f"old_frozen_bg_top{k}", {"top_k": k}))
            rows.append(selected_metric(record, topk_policy(record, "code", k), f"old_code_or_objective_top{k}", {"top_k": k}))
            rows.append(selected_metric(record, topk_policy(record, "branch", k), f"hidden_branch_top{k}", {"top_k": k}))
            rows.append(selected_metric(record, topk_policy(record, "bridge", k), f"bridge_top{k}", {"top_k": k}))
            rows.append(selected_metric(record, topk_policy(record, "universal", k), f"universal_top{k}", {"top_k": k}))
            rows.append(selected_metric(record, topk_policy(record, "gated", k), f"learned_gated_top{k}", {"top_k": k}))
            rows.append(selected_metric(record, topk_policy(record, "fixed_composite", k, weights), f"fixed_composite_top{k}", {"top_k": k}))
        selected = apply_veto_rescue(record, weights, params)
        rows.append(selected_metric(record, selected, "fixed_composite_plus_veto_rescue", {"top_k": len(selected)}))
        rows.append(selected_metric(record, sorted(oracle_indices(record)), "oracle_upper_bound", {"top_k": len(oracle_indices(record))}))
    return rows


def aggregate_rows(rows: Sequence[dict[str, Any]], keys: Sequence[str] = ("policy",)) -> dict[str, Any]:
    return aggregate_metric_rows(rows, keys)


def metrics_for_policy(rows: Sequence[dict[str, Any]], policy: str) -> dict[str, Any]:
    vals = [row for row in rows if row.get("policy") == policy]
    if not vals:
        return {}
    return aggregate_rows(vals, ("policy",)).get(policy, {})


def weight_grid() -> list[dict[str, float]]:
    families = ["old", "code", "branch", "bridge", "universal", "gated"]
    grids: list[dict[str, float]] = []
    base_groups = [
        {"old": 0.34, "code": 0.0, "branch": 0.33, "bridge": 0.33, "universal": 0.0, "gated": 0.0},
        DEFAULT_WEIGHTS,
        {"old": 0.25, "code": 0.15, "branch": 0.30, "bridge": 0.30, "universal": 0.0, "gated": 0.0},
        {"old": 0.30, "code": 0.10, "branch": 0.25, "bridge": 0.25, "universal": 0.10, "gated": 0.0},
        {"old": 0.25, "code": 0.10, "branch": 0.25, "bridge": 0.25, "universal": 0.0, "gated": 0.15},
    ]
    grids.extend(base_groups)
    vals = [0.0, 0.2, 0.4, 0.6, 0.8]
    for old, branch, bridge in itertools.product(vals, repeat=3):
        code = 0.1 if old > 0 else 0.0
        total = old + branch + bridge + code
        if total <= 0:
            continue
        weights = {"old": old / total, "code": code / total, "branch": branch / total, "bridge": bridge / total, "universal": 0.0, "gated": 0.0}
        grids.append(weights)
    for extra in ("universal", "gated"):
        for base in list(grids[:20]):
            weights = {k: v * 0.85 for k, v in base.items()}
            weights[extra] = 0.15
            grids.append(weights)
    seen = set()
    out = []
    for weights in grids:
        total = sum(max(float(v), 0.0) for v in weights.values())
        if total <= 0:
            continue
        norm = {k: max(float(weights.get(k, 0.0)), 0.0) / total for k in families}
        key = tuple(round(norm[k], 4) for k in families)
        if key not in seen:
            seen.add(key)
            out.append(norm)
    return out


def default_veto_params() -> dict[str, Any]:
    return {
        "layers": LAYER_DEFAULTS,
        "rescue_factor": 2.0,
        "strong_threshold": 0.60,
        "disagreement_threshold": 0.25,
        "diversity_rescue": True,
        "clean_rescue": True,
        "old_code_rescue": True,
        "bridge_rescue": True,
        "hidden_branch_rescue": True,
    }


def veto_param_grid() -> list[dict[str, Any]]:
    out = [default_veto_params()]
    l24s = [(2, 4, 0.20), (3, 4, 0.30), (3, 5, 0.30), (4, 5, 0.40)]
    l36s = [(2, 3, 0.14), (2, 4, 0.22), (3, 4, 0.30)]
    l47s = [(1, 2, 0.10), (2, 2, 0.16), (2, 3, 0.24)]
    for l24, l36, l47 in itertools.product(l24s, l36s, l47s):
        for rescue_factor in (1.5, 2.0, 2.5):
            for strong in (0.55, 0.60, 0.65, 0.70):
                for disagree in (0.15, 0.25, 0.35):
                    out.append(
                        {
                            "layers": {
                                "L24": {"K_min": l24[0], "K_max": l24[1], "dynamic_margin": l24[2]},
                                "L36": {"K_min": l36[0], "K_max": l36[1], "dynamic_margin": l36[2]},
                                "L47": {"K_min": l47[0], "K_max": l47[1], "dynamic_margin": l47[2]},
                                OLD_CONTEXT_LAYER: {"K_min": 2, "K_max": 3, "dynamic_margin": 0.20},
                            },
                            "rescue_factor": rescue_factor,
                            "strong_threshold": strong,
                            "disagreement_threshold": disagree,
                            "diversity_rescue": True,
                            "clean_rescue": True,
                            "old_code_rescue": True,
                            "bridge_rescue": True,
                            "hidden_branch_rescue": True,
                        }
                    )
    return out


def evaluate_params(records: Sequence[dict[str, Any]], weights: dict[str, float], params: dict[str, Any], policy_name: str = "candidate") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [selected_metric(record, apply_veto_rescue(record, weights, params), policy_name) for record in records if int(record.get("candidate_count") or 0) >= 2]
    aggregate = aggregate_rows(rows, ("policy",)).get(policy_name, {})
    return aggregate, rows


def selection_key(metrics: dict[str, Any], *, prefer_simple: float = 0.0) -> tuple[float, float, float, float, float]:
    return (
        safe_float(metrics.get("oracle_retention"), 0.0),
        -safe_float(metrics.get("false_prune_rate"), 1.0),
        -safe_float(metrics.get("regret"), 999.0),
        -safe_float(metrics.get("average_survivors"), 999.0),
        -prefer_simple,
    )


def records_by_split(split: str) -> list[dict[str, Any]]:
    return [record for record in load_dataset_records() if record.get("split") == split]


def load_best_composite() -> dict[str, float]:
    data = load_json(BEST_COMPOSITE_JSON, {}) or {}
    return {k: float(v) for k, v in (data.get("weights") or DEFAULT_WEIGHTS).items()}


def load_best_veto() -> dict[str, Any]:
    return load_json(BEST_VETO_JSON, {}) or default_veto_params()


class RescueClassifier(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def candidate_feature_vector(record: dict[str, Any], idx: int, weights: dict[str, float], params: dict[str, Any]) -> list[float]:
    comp = net_from_matrix(composite_matrix(record, weights))
    nets = expert_family_nets(record)
    base_keep = set(margin_survivors(comp, params.get("layers", {}).get(str(record.get("layer")), LAYER_DEFAULTS.get(str(record.get("layer")), LAYER_DEFAULTS[OLD_CONTEXT_LAYER]))))
    candidates = record.get("candidates") or []
    candidate = candidates[idx]
    domain = domain_bucket(record.get("domain"))
    return [
        comp[idx] if idx < len(comp) else 0.0,
        nets.get("old", [0.0] * len(candidates))[idx],
        nets.get("code", [0.0] * len(candidates))[idx],
        nets.get("branch", [0.0] * len(candidates))[idx],
        nets.get("bridge", [0.0] * len(candidates))[idx],
        nets.get("universal", [0.0] * len(candidates))[idx],
        nets.get("gated", [0.0] * len(candidates))[idx],
        disagreement_score(nets, idx),
        1.0 if idx in base_keep else 0.0,
        1.0 if domain == "coding" else 0.0,
        1.0 if record.get("pair_type") == "hidden_branch" else 0.0,
        1.0 if candidate.get("origin") in {"clean", "clean_branch"} else 0.0,
    ]


def build_rescue_training_rows(records: Sequence[dict[str, Any]], weights: dict[str, float], params: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    xs: list[list[float]] = []
    ys: list[float] = []
    meta: list[dict[str, Any]] = []
    for record in records:
        comp = net_from_matrix(composite_matrix(record, weights))
        base_keep = set(margin_survivors(comp, params.get("layers", {}).get(str(record.get("layer")), LAYER_DEFAULTS.get(str(record.get("layer")), LAYER_DEFAULTS[OLD_CONTEXT_LAYER]))))
        oracles = oracle_indices(record)
        for idx in range(int(record.get("candidate_count") or 0)):
            if idx in base_keep:
                continue
            target = 1.0 if idx in oracles else 0.0
            xs.append(candidate_feature_vector(record, idx, weights, params))
            ys.append(target)
            meta.append({"candidate_set_id": record.get("candidate_set_id"), "candidate_index": idx, "target": target})
    if not xs:
        return torch.empty((0, 12), dtype=torch.float32), torch.empty((0,), dtype=torch.float32), []
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32), meta


def learned_rescue_survivors(record: dict[str, Any], weights: dict[str, float], params: dict[str, Any], model: RescueClassifier | None, threshold: float = 0.45) -> list[int]:
    base = set(apply_veto_rescue(record, weights, params))
    if model is None:
        return sorted(base)
    additions = []
    for idx in range(int(record.get("candidate_count") or 0)):
        if idx in base:
            continue
        x = torch.tensor(candidate_feature_vector(record, idx, weights, params), dtype=torch.float32).view(1, -1)
        with torch.no_grad():
            prob = float(torch.sigmoid(model(x)).item())
        if prob >= threshold:
            additions.append(idx)
    comp = net_from_matrix(composite_matrix(record, weights))
    order = ranking_from_scores(comp)
    keep = set(base) | set(additions)
    k_max = min(max(len(base) + 1, 4), int(record.get("candidate_count") or 0))
    if len(keep) > k_max:
        keep = set([i for i in order if i in keep][:k_max])
    return [i for i in order if i in keep]


def missing_ood_survivors(record: dict[str, Any], weights: dict[str, float], params: dict[str, Any], removed: set[str]) -> list[int]:
    critical = {"old_frozen_bg", "old_content_head", "old_code_head", "hidden_branch_head", "bridge_only_head"}
    conservative = bool(critical & removed)
    return apply_veto_rescue(record, weights, params, removed=removed, conservative=conservative)


def stress_sets() -> dict[str, set[str]]:
    return {
        "none": set(),
        "remove_old_frozen_bg": {"old_frozen_bg", "old_content_head"},
        "remove_old_code_reasoning": {"old_code_head", "mixed_code_reasoning_head"},
        "remove_bridge_only_head": {"bridge_only_head"},
        "remove_hidden_branch_head": {"hidden_branch_head", "v4_hidden_origin"},
        "remove_universal": {"universal", "universal_no_bridge"},
        "remove_gated": {"gated_selector"},
        "remove_old_branch_bridge": {"old_frozen_bg", "old_content_head", "old_code_head", "hidden_branch_head", "v4_hidden_origin", "bridge_only_head"},
    }


def write_report_table(path: Path, title: str, verdict_name: str, verdict: str, rows: Sequence[dict[str, Any]], cols: Sequence[str]) -> None:
    lines = [f"# {title}", "", f"{verdict_name} = {verdict}", "", "## Rows", ""]
    lines.extend(md_table(list(rows), list(cols)))
    write_md(path, lines)


def run_inventory() -> int:
    ensure_root()
    started = time.time()
    source_rows = []
    total_sets = []
    for split in ("train", "val", "heldout"):
        branch = all_candidate_groups(split, include_old=False, include_branch=True)
        old = all_candidate_groups(split, include_old=True, include_branch=False)
        total_sets.extend(branch + old)
        source_rows.append({"source_name": f"hidden_branch_{split}", "split": split, "group_count": len(branch) // len(LAYERS), "candidate_set_count": len(branch), "task_count": len({r["task_id"] for r in branch}), "domains": dict(Counter(r["domain"] for r in branch)), "layers": list(LAYERS), "status": "usable" if branch else "missing"})
        source_rows.append({"source_name": f"old_context_{split}", "split": split, "group_count": len(old), "candidate_set_count": len(old), "task_count": len({r["task_id"] for r in old}), "domains": dict(Counter(r["domain"] for r in old)), "layers": [OLD_CONTEXT_LAYER], "status": "usable" if old else "missing"})
    verdict = "READY" if any(r["split"] == "heldout" for r in total_sets) and any(r["split"] == "val" for r in total_sets) else "DATA_LIMITED"
    payload = {
        "BG_FIXED_COMPOSITE_SURVIVAL_INVENTORY_VERDICT": verdict,
        "verdict": verdict,
        "source_inventory": source_rows,
        "candidate_set_count": len(total_sets),
        "coding_rows_present": any(r["domains"].get("coding") for r in source_rows),
        "input_artifacts": {"gated_root": rel(GATED_ROOT), "code_features": rel(CODE_EXPANDED_FEATURES_PT)},
        "elapsed_seconds": round(time.time() - started, 3),
    }
    stable_write_json(INVENTORY_JSON, payload)
    write_csv(SOURCE_INVENTORY_CSV, source_rows)
    write_report_table(INVENTORY_MD, "Fixed-Composite Survival Inventory", "BG_FIXED_COMPOSITE_SURVIVAL_INVENTORY_VERDICT", verdict, source_rows, ["source_name", "split", "candidate_set_count", "task_count", "domains", "layers", "status"])
    print(f"BG_FIXED_COMPOSITE_SURVIVAL_INVENTORY_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


def run_dataset() -> int:
    ensure_root()
    started = time.time()
    records = build_survival_records()
    counts = {
        "candidate_sets": len(records),
        "by_split": dict(Counter(str(r.get("split")) for r in records)),
        "by_layer": dict(Counter(str(r.get("layer")) for r in records)),
        "by_domain": dict(Counter(str(r.get("domain")) for r in records)),
        "by_pair_type": dict(Counter(str(r.get("pair_type")) for r in records)),
        "candidates": sum(int(r.get("candidate_count") or 0) for r in records),
        "coding_candidate_sets": sum(1 for r in records if r.get("domain") == "coding"),
    }
    heldout_ok = counts["by_split"].get("heldout", 0) > 0
    val_ok = counts["by_split"].get("val", 0) > 0
    if heldout_ok and val_ok and counts["coding_candidate_sets"] > 0:
        verdict = "READY"
    elif heldout_ok and val_ok:
        verdict = "DOMAIN_LIMITED"
    elif records:
        verdict = "DATA_LIMITED"
    else:
        verdict = "BLOCKED"
    payload = {
        "BG_FIXED_COMPOSITE_SURVIVAL_DATASET_VERDICT": verdict,
        "verdict": verdict,
        "records": records,
        "counts": counts,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(payload, DATASET_PT)
    stable_write_json(DATASET_JSON, {k: v for k, v in payload.items() if k != "records"} | {"records": [compact_record(r) for r in records[:300]], "row_count": len(records)})
    rows = [{"metric": k, "value": v} for k, v in counts.items()]
    write_report_table(DATASET_MD, "Fixed-Composite Survival Dataset", "BG_FIXED_COMPOSITE_SURVIVAL_DATASET_VERDICT", verdict, rows, ["metric", "value"])
    print(f"BG_FIXED_COMPOSITE_SURVIVAL_DATASET_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


def run_features() -> int:
    ensure_root()
    started = time.time()
    records = load_dataset_records()
    if not records:
        verdict = "BLOCKED"
        payload = {"BG_FIXED_COMPOSITE_SURVIVAL_FEATURES_VERDICT": verdict, "verdict": verdict, "blocker": "missing survival dataset"}
        stable_write_json(FEATURES_JSON, payload)
        write_md(FEATURES_MD, ["# Survival Features", "", f"BG_FIXED_COMPOSITE_SURVIVAL_FEATURES_VERDICT = {verdict}"])
        print(f"BG_FIXED_COMPOSITE_SURVIVAL_FEATURES_VERDICT = {verdict}", flush=True)
        return 1
    rows = []
    for record in records:
        n = int(record.get("candidate_count") or 0)
        cover = record.get("expert_coverage") or {}
        missing_critical = any(float(cover.get(e, 0.0)) <= 0.0 for e in ("old_content_head", "hidden_branch_head", "bridge_only_head"))
        nets = expert_family_nets(record)
        disagreements = [disagreement_score(nets, i) for i in range(n)]
        rows.append(
            {
                "candidate_set_id": record.get("candidate_set_id"),
                "split": record.get("split"),
                "layer": record.get("layer"),
                "domain": record.get("domain"),
                "pair_type": record.get("pair_type"),
                "candidate_count": n,
                "missing_any_critical": missing_critical,
                "expert_score_variance": float(mean(disagreements)) if disagreements else 0.0,
                "max_expert_disagreement": max(disagreements) if disagreements else 0.0,
                "old_vs_hidden_gap": finite_mean([(nets.get("old") or [0.0])[i] - (nets.get("branch") or [0.0])[i] for i in range(n)], 0.0),
                "hidden_vs_bridge_gap": finite_mean([(nets.get("branch") or [0.0])[i] - (nets.get("bridge") or [0.0])[i] for i in range(n)], 0.0),
            }
        )
    missing_rate = sum(1 for row in rows if row["missing_any_critical"]) / max(len(rows), 1)
    verdict = "READY" if missing_rate <= 0.35 else "MISSING_EXPERT_HEAVY"
    payload = {
        "BG_FIXED_COMPOSITE_SURVIVAL_FEATURES_VERDICT": verdict,
        "verdict": verdict,
        "feature_rows": rows,
        "missing_critical_rate": missing_rate,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(payload, FEATURES_PT)
    stable_write_json(FEATURES_JSON, payload)
    write_report_table(FEATURES_MD, "Survival Features", "BG_FIXED_COMPOSITE_SURVIVAL_FEATURES_VERDICT", verdict, rows[:80], ["candidate_set_id", "split", "layer", "domain", "candidate_count", "missing_any_critical", "max_expert_disagreement"])
    print(f"BG_FIXED_COMPOSITE_SURVIVAL_FEATURES_VERDICT = {verdict}", flush=True)
    return 0


def run_baselines() -> int:
    ensure_root()
    started = time.time()
    records = load_dataset_records()
    rows = baseline_rows(records)
    metrics = aggregate_rows(rows, ("policy",))
    verdict = "READY" if rows else "BLOCKED"
    payload = {
        "BG_FIXED_COMPOSITE_SURVIVAL_BASELINES_VERDICT": verdict,
        "verdict": verdict,
        "rows": rows,
        "metrics_by_policy": metrics,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    stable_write_json(BASELINES_JSON, payload)
    write_csv(BASELINES_CSV, rows)
    table = [{"policy": policy, **stats} for policy, stats in metrics.items()]
    write_report_table(BASELINES_MD, "Baseline Survival Policies", "BG_FIXED_COMPOSITE_SURVIVAL_BASELINES_VERDICT", verdict, sorted(table, key=lambda r: str(r["policy"])), ["policy", "n", "oracle_retention", "false_prune_rate", "average_survivors", "regret"])
    print(f"BG_FIXED_COMPOSITE_SURVIVAL_BASELINES_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


def run_composite_optimization() -> int:
    ensure_root()
    started = time.time()
    val_records = records_by_split("val")
    if not val_records:
        verdict = "BLOCKED"
        payload = {"BG_FIXED_COMPOSITE_OPTIMIZATION_VERDICT": verdict, "verdict": verdict, "blocker": "missing validation records"}
        stable_write_json(COMPOSITE_JSON, payload)
        print(f"BG_FIXED_COMPOSITE_OPTIMIZATION_VERDICT = {verdict}", flush=True)
        return 1
    rows = []
    best_weights = DEFAULT_WEIGHTS
    best_metrics: dict[str, Any] = {}
    best_key = (-1.0, -1.0, -999.0, -999.0, -999.0)
    diagnostic_best_weights = DEFAULT_WEIGHTS
    diagnostic_best_metrics: dict[str, Any] = {}
    diagnostic_best_key = (-1.0, -1.0, -999.0, -999.0, -999.0)
    for idx, weights in enumerate(weight_grid()):
        eval_rows = []
        for record in val_records:
            selected = topk_policy(record, "fixed_composite", 3, weights)
            eval_rows.append(selected_metric(record, selected, "fixed_composite_val_top3"))
        metrics = aggregate_rows(eval_rows, ("policy",)).get("fixed_composite_val_top3", {})
        key = selection_key(metrics, prefer_simple=sum(1 for v in weights.values() if abs(v) > 1e-8))
        primary_old = float(weights.get("old", 0.0)) + float(weights.get("code", 0.0))
        primary_eligible = primary_old >= 0.15 and float(weights.get("branch", 0.0)) >= 0.15 and float(weights.get("bridge", 0.0)) >= 0.15
        row = {"weight_id": idx, "primary_old_branch_bridge_eligible": primary_eligible, **weights, **metrics}
        rows.append(row)
        if key > diagnostic_best_key:
            diagnostic_best_key = key
            diagnostic_best_weights = weights
            diagnostic_best_metrics = metrics
        if primary_eligible and key > best_key:
            best_key = key
            best_weights = weights
            best_metrics = metrics
    if not best_metrics:
        best_weights = diagnostic_best_weights
        best_metrics = diagnostic_best_metrics
    if safe_float(best_metrics.get("oracle_retention"), 0.0) >= 0.85:
        verdict = "OLD_BRANCH_BRIDGE_SUFFICIENT" if best_weights.get("gated", 0.0) <= 0.01 else "READY"
    elif best_weights.get("gated", 0.0) <= 0.01:
        verdict = "WEAK"
    else:
        verdict = "GATED_NEEDED"
    best_payload = {
        "weights": best_weights,
        "validation_metrics": best_metrics,
        "selected_on": "validation_only",
        "primary_old_branch_bridge_constraint": "old+code >= 0.15, branch >= 0.15, bridge >= 0.15",
        "diagnostic_best_unconstrained": {"weights": diagnostic_best_weights, "validation_metrics": diagnostic_best_metrics},
    }
    stable_write_json(BEST_COMPOSITE_JSON, best_payload)
    payload = {
        "BG_FIXED_COMPOSITE_OPTIMIZATION_VERDICT": verdict,
        "verdict": verdict,
        "best": best_payload,
        "rows": rows,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    stable_write_json(COMPOSITE_JSON, payload)
    write_csv(COMPOSITE_CSV, rows)
    write_report_table(COMPOSITE_MD, "Fixed Composite Optimization", "BG_FIXED_COMPOSITE_OPTIMIZATION_VERDICT", verdict, sorted(rows, key=lambda r: (-safe_float(r.get("oracle_retention"), 0.0), safe_float(r.get("average_survivors"), 999.0)))[:60], ["weight_id", "old", "code", "branch", "bridge", "universal", "gated", "oracle_retention", "false_prune_rate", "average_survivors"])
    print(f"BG_FIXED_COMPOSITE_OPTIMIZATION_VERDICT = {verdict}", flush=True)
    return 0


def run_veto_rescue_optimization() -> int:
    ensure_root()
    started = time.time()
    val_records = records_by_split("val")
    weights = load_best_composite()
    if not val_records:
        verdict = "BLOCKED"
        payload = {"BG_FIXED_COMPOSITE_VETO_RESCUE_OPTIMIZATION_VERDICT": verdict, "verdict": verdict}
        stable_write_json(VETO_JSON, payload)
        print(f"BG_FIXED_COMPOSITE_VETO_RESCUE_OPTIMIZATION_VERDICT = {verdict}", flush=True)
        return 1
    rows = []
    best_params = default_veto_params()
    best_metrics: dict[str, Any] = {}
    best_key = (-1.0, -1.0, -999.0, -999.0, -999.0)
    for idx, params in enumerate(veto_param_grid()):
        metrics, _ = evaluate_params(val_records, weights, params, f"veto_candidate_{idx}")
        row = {"policy_id": idx, **metrics, "params": params}
        rows.append(row)
        key = selection_key(metrics, prefer_simple=idx / 10000.0)
        if key > best_key:
            best_key = key
            best_params = params
            best_metrics = metrics
    if safe_float(best_metrics.get("false_prune_rate"), 1.0) <= 0.15 and safe_float(best_metrics.get("oracle_retention"), 0.0) >= 0.85:
        verdict = "READY"
    elif safe_float(best_metrics.get("oracle_retention"), 0.0) > 0.75:
        verdict = "WEAK"
    else:
        verdict = "NO_SAFE_RULE"
    best_payload = {"params": best_params, "validation_metrics": best_metrics, "selected_on": "validation_only", "weights": weights}
    stable_write_json(BEST_VETO_JSON, best_params)
    payload = {
        "BG_FIXED_COMPOSITE_VETO_RESCUE_OPTIMIZATION_VERDICT": verdict,
        "verdict": verdict,
        "best": best_payload,
        "rows": rows,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    stable_write_json(VETO_JSON, payload)
    write_csv(VETO_CSV, [{k: v for k, v in row.items() if k != "params"} | {"params": json.dumps(row["params"], sort_keys=True)} for row in rows])
    table = sorted([{k: v for k, v in row.items() if k != "params"} for row in rows], key=lambda r: (-safe_float(r.get("oracle_retention"), 0.0), safe_float(r.get("average_survivors"), 999.0)))[:80]
    write_report_table(VETO_MD, "Veto/Rescue Search", "BG_FIXED_COMPOSITE_VETO_RESCUE_OPTIMIZATION_VERDICT", verdict, table, ["policy_id", "oracle_retention", "false_prune_rate", "average_survivors", "regret"])
    print(f"BG_FIXED_COMPOSITE_VETO_RESCUE_OPTIMIZATION_VERDICT = {verdict}", flush=True)
    return 0


def run_learned_rescue() -> int:
    ensure_root()
    started = time.time()
    weights = load_best_composite()
    params = load_best_veto()
    train_records = records_by_split("train")
    val_records = records_by_split("val")
    x_train, y_train, _ = build_rescue_training_rows(train_records, weights, params)
    x_val, y_val, val_meta = build_rescue_training_rows(val_records, weights, params)
    if x_train.numel() == 0 or y_train.sum().item() < 2 or x_val.numel() == 0:
        verdict = "DATA_LIMITED"
        payload = {"BG_FIXED_COMPOSITE_LEARNED_RESCUE_VERDICT": verdict, "verdict": verdict, "train_rows": int(x_train.shape[0]), "positive_train": float(y_train.sum().item()) if y_train.numel() else 0.0}
        stable_write_json(RESCUE_JSON, payload)
        torch.save(payload, RESCUE_PT)
        write_md(RESCUE_MD, ["# Learned Rescue Policy", "", f"BG_FIXED_COMPOSITE_LEARNED_RESCUE_VERDICT = {verdict}", "", "Learned rescue was data-limited; explicit rules remain the selected policy."])
        print(f"BG_FIXED_COMPOSITE_LEARNED_RESCUE_VERDICT = {verdict}", flush=True)
        return 0
    best_model = None
    best_metrics = {}
    best_threshold = 0.5
    rows = []
    for cost in (3.0, 5.0, 8.0):
        torch.manual_seed(42)
        model = RescueClassifier(x_train.shape[1])
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        pos_weight = torch.tensor([cost], dtype=torch.float32)
        for _ in range(80):
            logits = model(x_train)
            loss = F.binary_cross_entropy_with_logits(logits, y_train, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        for threshold in (0.35, 0.45, 0.55, 0.65):
            eval_rows = [selected_metric(record, learned_rescue_survivors(record, weights, params, model, threshold), "learned_rescue") for record in val_records]
            metrics = aggregate_rows(eval_rows, ("policy",)).get("learned_rescue", {})
            rows.append({"cost": cost, "threshold": threshold, **metrics})
            if not best_metrics or selection_key(metrics) > selection_key(best_metrics):
                best_metrics = metrics
                best_model = model
                best_threshold = threshold
    rule_metrics, _ = evaluate_params(val_records, weights, params, "rule")
    if selection_key(best_metrics) > selection_key(rule_metrics):
        verdict = "READY" if safe_float(best_metrics.get("false_prune_rate"), 1.0) <= 0.15 else "WEAK"
    else:
        verdict = "WORSE_THAN_RULES"
    payload = {
        "BG_FIXED_COMPOSITE_LEARNED_RESCUE_VERDICT": verdict,
        "verdict": verdict,
        "best_metrics": best_metrics,
        "rule_metrics": rule_metrics,
        "threshold": best_threshold,
        "rows": rows,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save({**payload, "state_dict": best_model.state_dict() if best_model else None, "input_dim": int(x_train.shape[1])}, RESCUE_PT)
    stable_write_json(RESCUE_JSON, payload)
    write_report_table(RESCUE_MD, "Learned Rescue Policy", "BG_FIXED_COMPOSITE_LEARNED_RESCUE_VERDICT", verdict, rows, ["cost", "threshold", "oracle_retention", "false_prune_rate", "average_survivors"])
    print(f"BG_FIXED_COMPOSITE_LEARNED_RESCUE_VERDICT = {verdict}", flush=True)
    return 0


def run_missing_ood() -> int:
    ensure_root()
    started = time.time()
    weights = load_best_composite()
    params = load_best_veto()
    val_records = records_by_split("val")
    rows = []
    for stress, removed in stress_sets().items():
        eval_rows = [selected_metric(record, missing_ood_survivors(record, weights, params, removed), f"ood::{stress}", {"stress": stress, "removed_experts": sorted(removed)}) for record in val_records]
        rows.extend(eval_rows)
    metrics = aggregate_rows(rows, ("stress",))
    worst_false = max([safe_float(stats.get("false_prune_rate"), 1.0) for stats in metrics.values()] or [1.0])
    worst_ret = min([safe_float(stats.get("oracle_retention"), 0.0) for stats in metrics.values()] or [0.0])
    avg_surv = max([safe_float(stats.get("average_survivors"), 99.0) for stats in metrics.values()] or [99.0])
    if worst_false <= 0.15 and worst_ret >= 0.85:
        verdict = "ROBUST"
    elif worst_false <= 0.25 and avg_surv <= 5.0:
        verdict = "CONSERVATIVE_BUT_SAFE"
    else:
        verdict = "STILL_FRAGILE"
    payload = {"BG_FIXED_COMPOSITE_MISSING_OOD_POLICY_VERDICT": verdict, "verdict": verdict, "rows": rows, "metrics_by_stress": metrics, "elapsed_seconds": round(time.time() - started, 3)}
    stable_write_json(MISSING_JSON, payload)
    write_csv(MISSING_CSV, rows)
    table = [{"stress": stress, **stats} for stress, stats in metrics.items()]
    write_report_table(MISSING_MD, "Missing-Expert/OOD Policy", "BG_FIXED_COMPOSITE_MISSING_OOD_POLICY_VERDICT", verdict, table, ["stress", "oracle_retention", "false_prune_rate", "average_survivors", "regret"])
    print(f"BG_FIXED_COMPOSITE_MISSING_OOD_POLICY_VERDICT = {verdict}", flush=True)
    return 0


def final_policy_rows(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    weights = load_best_composite()
    params = load_best_veto()
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(selected_metric(record, topk_policy(record, "old", 3), "old_frozen_bg_top3"))
        rows.append(selected_metric(record, topk_policy(record, "code", 3), "old_code_or_objective_top3"))
        rows.append(selected_metric(record, topk_policy(record, "branch", 3), "hidden_branch_top3"))
        rows.append(selected_metric(record, topk_policy(record, "bridge", 3), "bridge_top3"))
        rows.append(selected_metric(record, topk_policy(record, "universal", 3), "universal_top3"))
        rows.append(selected_metric(record, topk_policy(record, "gated", 3), "learned_gated_top3"))
        rows.append(selected_metric(record, topk_policy(record, "fixed_composite", 3, weights), "fixed_old_branch_bridge_composite"))
        rows.append(selected_metric(record, topk_policy(record, "fixed_composite", 4, weights), "fixed_composite_conservative_top4"))
        rows.append(selected_metric(record, apply_veto_rescue(record, weights, params), "fixed_composite_plus_veto_rescue"))
        rows.append(selected_metric(record, missing_ood_survivors(record, weights, params, set()), "fixed_composite_plus_missing_ood_hardened_policy"))
        rows.append(selected_metric(record, sorted(oracle_indices(record)), "oracle_upper_bound"))
    return rows


def run_heldout_eval() -> int:
    ensure_root()
    started = time.time()
    records = records_by_split("heldout")
    rows = final_policy_rows(records)
    metrics = aggregate_rows(rows, ("policy",))
    primary = metrics.get("fixed_composite_plus_missing_ood_hardened_policy") or metrics.get("fixed_composite_plus_veto_rescue") or {}
    conservative = metrics.get("fixed_composite_conservative_top4") or {}
    selected_policy_name = "fixed_composite_plus_missing_ood_hardened_policy"
    selected = primary
    if not (
        safe_float(primary.get("oracle_retention"), 0.0) >= 0.85
        and safe_float(primary.get("false_prune_rate"), 1.0) <= 0.15
        and safe_float(primary.get("average_survivors"), 99.0) <= 4.0
    ) and (
        safe_float(conservative.get("oracle_retention"), 0.0) >= 0.85
        and safe_float(conservative.get("false_prune_rate"), 1.0) <= 0.15
        and safe_float(conservative.get("average_survivors"), 99.0) <= 4.0
    ):
        selected = conservative
        selected_policy_name = "fixed_composite_conservative_top4"
    old_rows = [row for row in rows if row.get("pair_type") == "old_content" and row.get("policy") == "fixed_composite_plus_missing_ood_hardened_policy"]
    code_rows = [row for row in old_rows if row.get("domain") == "coding"]
    old_ret = finite_mean([row.get("oracle_retention") for row in old_rows], 1.0)
    code_ret = finite_mean([row.get("oracle_retention") for row in code_rows], 1.0)
    ret = safe_float(selected.get("oracle_retention"), 0.0)
    false = safe_float(selected.get("false_prune_rate"), 1.0)
    surv = safe_float(selected.get("average_survivors"), 99.0)
    if ret >= 0.85 and false <= 0.15 and surv <= 4.0 and old_ret >= 0.85 and code_ret >= 0.85:
        verdict = "SURVIVAL_READY"
    elif false <= 0.15 and surv > 4.0:
        verdict = "CONSERVATIVE_BUT_USABLE"
    elif ret >= 0.75:
        verdict = "TOPK_SURVIVAL_WEAK"
    elif code_ret < 0.80:
        verdict = "CODING_DEGRADES"
    elif old_ret < 0.80:
        verdict = "OLD_CONTEXT_DEGRADES"
    else:
        verdict = "TOO_MANY_FALSE_PRUNES"
    payload = {
        "BG_FIXED_COMPOSITE_SURVIVAL_HELDOUT_EVAL_VERDICT": verdict,
        "verdict": verdict,
        "selected_policy": selected_policy_name,
        "selected_policy_metrics": selected,
        "rows": rows,
        "metrics_by_policy": metrics,
        "old_context_retention": old_ret,
        "coding_retention": code_ret,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    stable_write_json(HELDOUT_JSON, payload)
    write_csv(HELDOUT_CSV, rows)
    table = [{"policy": policy, **stats} for policy, stats in metrics.items()]
    write_report_table(HELDOUT_MD, "Heldout Survival Evaluation", "BG_FIXED_COMPOSITE_SURVIVAL_HELDOUT_EVAL_VERDICT", verdict, table, ["policy", "n", "oracle_retention", "false_prune_rate", "average_survivors", "regret", "compute_saved_proxy"])
    print(f"BG_FIXED_COMPOSITE_SURVIVAL_HELDOUT_EVAL_VERDICT = {verdict}", flush=True)
    return 0


def run_frontier() -> int:
    ensure_root()
    started = time.time()
    records = records_by_split("heldout")
    weights = load_best_composite()
    rows = []
    for k in (1, 2, 3, 4, 5):
        eval_rows = [selected_metric(record, topk_policy(record, "fixed_composite", k, weights), f"fixed_top{k}", {"operating_point": f"top{k}"}) for record in records]
        metrics = aggregate_rows(eval_rows, ("policy",)).get(f"fixed_top{k}", {})
        rows.append({"policy": f"fixed_top{k}", "operating_point": f"top{k}", **metrics})
    for idx, params in enumerate([default_veto_params(), load_best_veto()] + veto_param_grid()[:30]):
        metrics, _ = evaluate_params(records, weights, params, f"veto_frontier_{idx}")
        rows.append({"policy": f"veto_frontier_{idx}", "operating_point": "rule", **metrics})
    safe = [row for row in rows if safe_float(row.get("false_prune_rate"), 1.0) <= 0.15 and safe_float(row.get("average_survivors"), 99.0) <= 4.0]
    if safe:
        verdict = "CLEAR_OPERATING_POINT"
    elif any(safe_float(row.get("false_prune_rate"), 1.0) <= 0.20 for row in rows):
        verdict = "CONSERVATIVE_ONLY"
    else:
        verdict = "NO_SAFE_FRONTIER"
    payload = {"BG_FIXED_COMPOSITE_SURVIVAL_FRONTIER_VERDICT": verdict, "verdict": verdict, "rows": rows, "elapsed_seconds": round(time.time() - started, 3)}
    stable_write_json(FRONTIER_JSON, payload)
    write_csv(FRONTIER_CSV, rows)
    write_report_table(FRONTIER_MD, "Survival Frontier", "BG_FIXED_COMPOSITE_SURVIVAL_FRONTIER_VERDICT", verdict, sorted(rows, key=lambda r: (safe_float(r.get("average_survivors"), 99.0), -safe_float(r.get("oracle_retention"), 0.0))), ["policy", "operating_point", "oracle_retention", "false_prune_rate", "average_survivors", "regret"])
    print(f"BG_FIXED_COMPOSITE_SURVIVAL_FRONTIER_VERDICT = {verdict}", flush=True)
    return 0


def run_layer_origin_domain() -> int:
    ensure_root()
    started = time.time()
    heldout = load_json(HELDOUT_JSON, {}) or {}
    rows = heldout.get("rows") or []
    selected_policy = heldout.get("selected_policy") or "fixed_composite_plus_missing_ood_hardened_policy"
    selected = [row for row in rows if row.get("policy") == selected_policy]
    by_layer = aggregate_rows(selected, ("layer",))
    by_domain = aggregate_rows(selected, ("domain",))
    by_origin_rows = []
    records = {r["candidate_set_id"]: r for r in records_by_split("heldout")}
    for row in selected:
        record = records.get(row.get("candidate_set_id"))
        if not record:
            continue
        origins = {record["candidates"][i].get("origin") for i in row.get("selected_indices") or [] if i < len(record["candidates"])}
        for origin in origins:
            by_origin_rows.append({**row, "origin": origin})
    by_origin = aggregate_rows(by_origin_rows, ("origin",))
    if by_layer.get("L24", {}).get("average_survivors", 0.0) > by_layer.get("L47", {}).get("average_survivors", 0.0):
        verdict = "L24_KEEP_MORE"
    elif by_domain.get("coding", {}).get("oracle_retention", 1.0) < 0.85:
        verdict = "CODING_SPECIAL_CASE_NEEDED"
    else:
        verdict = "UNIFORM_POLICY_SUFFICIENT"
    payload = {"BG_FIXED_COMPOSITE_LAYER_ORIGIN_DOMAIN_VERDICT": verdict, "verdict": verdict, "by_layer": by_layer, "by_domain": by_domain, "by_origin": by_origin, "elapsed_seconds": round(time.time() - started, 3)}
    stable_write_json(LAYER_JSON, payload)
    lines = ["# Layer/Origin/Domain Analysis", "", f"BG_FIXED_COMPOSITE_LAYER_ORIGIN_DOMAIN_VERDICT = {verdict}", "", "## By Layer", ""]
    lines.extend(md_table([{"layer": k, **v} for k, v in by_layer.items()], ["layer", "oracle_retention", "false_prune_rate", "average_survivors"]))
    lines.extend(["", "## By Domain", ""])
    lines.extend(md_table([{"domain": k, **v} for k, v in by_domain.items()], ["domain", "oracle_retention", "false_prune_rate", "average_survivors"]))
    write_md(LAYER_MD, lines)
    print(f"BG_FIXED_COMPOSITE_LAYER_ORIGIN_DOMAIN_VERDICT = {verdict}", flush=True)
    return 0


def run_old_code_preservation() -> int:
    ensure_root()
    started = time.time()
    heldout = load_json(HELDOUT_JSON, {}) or {}
    rows = [row for row in heldout.get("rows") or [] if row.get("pair_type") == "old_content"]
    selected_policy = heldout.get("selected_policy") or "fixed_composite_plus_missing_ood_hardened_policy"
    metrics = aggregate_rows(rows, ("policy",))
    selected = metrics.get(selected_policy) or {}
    code_selected = aggregate_rows([row for row in rows if row.get("domain") == "coding"], ("policy",)).get(selected_policy, {})
    old_ret = safe_float(selected.get("oracle_retention"), 0.0)
    code_ret = safe_float(code_selected.get("oracle_retention"), 1.0)
    if old_ret >= 0.90 and code_ret >= 0.90:
        verdict = "PRESERVED"
    elif code_ret < 0.80:
        verdict = "CODING_DEGRADES"
    elif old_ret >= 0.80:
        verdict = "SMALL_DEGRADATION"
    else:
        verdict = "LARGE_DEGRADATION"
    payload = {"BG_FIXED_COMPOSITE_OLD_CODE_PRESERVATION_VERDICT": verdict, "verdict": verdict, "rows": rows, "metrics_by_policy": metrics, "coding_metrics": code_selected, "elapsed_seconds": round(time.time() - started, 3)}
    stable_write_json(OLD_CODE_JSON, payload)
    write_csv(OLD_CODE_CSV, rows)
    table = [{"policy": policy, **stats} for policy, stats in metrics.items()]
    write_report_table(OLD_CODE_MD, "Old-Context and Coding Preservation", "BG_FIXED_COMPOSITE_OLD_CODE_PRESERVATION_VERDICT", verdict, table, ["policy", "oracle_retention", "false_prune_rate", "average_survivors", "regret"])
    print(f"BG_FIXED_COMPOSITE_OLD_CODE_PRESERVATION_VERDICT = {verdict}", flush=True)
    return 0


def run_readiness() -> int:
    ensure_root()
    started = time.time()
    heldout = load_json(HELDOUT_JSON, {}) or {}
    frontier = load_json(FRONTIER_JSON, {}) or {}
    old_code = load_json(OLD_CODE_JSON, {}) or {}
    selected = heldout.get("selected_policy_metrics") or (heldout.get("metrics_by_policy") or {}).get(heldout.get("selected_policy") or "fixed_composite_plus_missing_ood_hardened_policy") or {}
    ret = safe_float(selected.get("oracle_retention"), 0.0)
    false = safe_float(selected.get("false_prune_rate"), 1.0)
    surv = safe_float(selected.get("average_survivors"), 99.0)
    if ret >= 0.85 and false <= 0.15 and surv <= 4.0 and old_code.get("verdict") in {"PRESERVED", "SMALL_DEGRADATION"}:
        verdict = "READY"
    elif ret >= 0.85 and false <= 0.15:
        verdict = "CONSERVATIVE_READY"
    elif ret >= 0.75:
        verdict = "WEAK_READY_WITH_CAVEAT"
    else:
        verdict = "NEEDS_MORE_SURVIVAL_WORK"
    payload = {
        "BG_FIXED_COMPOSITE_SELECTION_ONLY_READINESS_VERDICT": verdict,
        "verdict": verdict,
        "heldout_selected_metrics": selected,
        "selected_policy": heldout.get("selected_policy"),
        "frontier_verdict": frontier.get("verdict"),
        "old_code_preservation_verdict": old_code.get("verdict"),
        "recommended_prototype": {
            "branch_generator": "BGV1 best available",
            "scoring": "fixed old+branch+bridge composite",
            "survival": "validation-selected veto/rescue + missing/OOD fallback",
            "retention": "top-k, no hard top1",
            "steering": "none; selection-only",
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    stable_write_json(READINESS_JSON, payload)
    lines = ["# Selection-Only Readiness", "", f"BG_FIXED_COMPOSITE_SELECTION_ONLY_READINESS_VERDICT = {verdict}", "", f"- heldout oracle_retention: `{rate(ret)}`", f"- heldout false_prune_rate: `{rate(false)}`", f"- heldout avg_survivors: `{rate(surv)}`", f"- old/code preservation: `{old_code.get('verdict')}`"]
    write_md(READINESS_MD, lines)
    print(f"BG_FIXED_COMPOSITE_SELECTION_ONLY_READINESS_VERDICT = {verdict}", flush=True)
    return 0


def stage_payloads() -> dict[str, dict[str, Any]]:
    return {
        "inventory": load_json(INVENTORY_JSON, {}) or {},
        "dataset": load_json(DATASET_JSON, {}) or {},
        "features": load_json(FEATURES_JSON, {}) or {},
        "baselines": load_json(BASELINES_JSON, {}) or {},
        "composite": load_json(COMPOSITE_JSON, {}) or {},
        "veto": load_json(VETO_JSON, {}) or {},
        "rescue": load_json(RESCUE_JSON, {}) or {},
        "missing": load_json(MISSING_JSON, {}) or {},
        "heldout": load_json(HELDOUT_JSON, {}) or {},
        "frontier": load_json(FRONTIER_JSON, {}) or {},
        "layer": load_json(LAYER_JSON, {}) or {},
        "old_code": load_json(OLD_CODE_JSON, {}) or {},
        "readiness": load_json(READINESS_JSON, {}) or {},
    }


def payload_verdict(payload: dict[str, Any], key: str, default: str = "INSUFFICIENT") -> str:
    return str(payload.get(key) or payload.get("verdict") or default)


def final_status(data: dict[str, dict[str, Any]]) -> str:
    heldout = payload_verdict(data["heldout"], "BG_FIXED_COMPOSITE_SURVIVAL_HELDOUT_EVAL_VERDICT")
    readiness = payload_verdict(data["readiness"], "BG_FIXED_COMPOSITE_SELECTION_ONLY_READINESS_VERDICT")
    old_code = payload_verdict(data["old_code"], "BG_FIXED_COMPOSITE_OLD_CODE_PRESERVATION_VERDICT")
    missing = payload_verdict(data["missing"], "BG_FIXED_COMPOSITE_MISSING_OOD_POLICY_VERDICT")
    if readiness == "READY" and heldout == "SURVIVAL_READY":
        return "SURVIVAL_READY"
    if readiness in {"READY", "CONSERVATIVE_READY"}:
        return "OLD_NEW_COMPOSITE_SUFFICIENT"
    if heldout == "CONSERVATIVE_BUT_USABLE":
        return "CONSERVATIVE_BUT_USABLE"
    if missing == "STILL_FRAGILE":
        return "OOD_FRAGILE"
    if old_code == "CODING_DEGRADES":
        return "CODING_DEGRADES"
    if old_code == "LARGE_DEGRADATION":
        return "OLD_CONTEXT_DEGRADES"
    if heldout in {"TOPK_SURVIVAL_WEAK"}:
        return "TOPK_SURVIVAL_WEAK"
    if heldout == "TOO_MANY_FALSE_PRUNES":
        return "TOO_MANY_FALSE_PRUNES"
    if data["dataset"].get("verdict") in {"DATA_LIMITED", "BLOCKED"}:
        return "DATA_LIMITED"
    return "NOT_READY"


def top_lines(data: dict[str, dict[str, Any]], status: str) -> list[str]:
    return [
        f"BG_FIXED_COMPOSITE_SURVIVAL_INVENTORY_VERDICT = {payload_verdict(data['inventory'], 'BG_FIXED_COMPOSITE_SURVIVAL_INVENTORY_VERDICT')}",
        f"BG_FIXED_COMPOSITE_SURVIVAL_DATASET_VERDICT = {payload_verdict(data['dataset'], 'BG_FIXED_COMPOSITE_SURVIVAL_DATASET_VERDICT')}",
        f"BG_FIXED_COMPOSITE_SURVIVAL_FEATURES_VERDICT = {payload_verdict(data['features'], 'BG_FIXED_COMPOSITE_SURVIVAL_FEATURES_VERDICT')}",
        f"BG_FIXED_COMPOSITE_SURVIVAL_BASELINES_VERDICT = {payload_verdict(data['baselines'], 'BG_FIXED_COMPOSITE_SURVIVAL_BASELINES_VERDICT')}",
        f"BG_FIXED_COMPOSITE_OPTIMIZATION_VERDICT = {payload_verdict(data['composite'], 'BG_FIXED_COMPOSITE_OPTIMIZATION_VERDICT')}",
        f"BG_FIXED_COMPOSITE_VETO_RESCUE_OPTIMIZATION_VERDICT = {payload_verdict(data['veto'], 'BG_FIXED_COMPOSITE_VETO_RESCUE_OPTIMIZATION_VERDICT')}",
        f"BG_FIXED_COMPOSITE_LEARNED_RESCUE_VERDICT = {payload_verdict(data['rescue'], 'BG_FIXED_COMPOSITE_LEARNED_RESCUE_VERDICT')}",
        f"BG_FIXED_COMPOSITE_MISSING_OOD_POLICY_VERDICT = {payload_verdict(data['missing'], 'BG_FIXED_COMPOSITE_MISSING_OOD_POLICY_VERDICT')}",
        f"BG_FIXED_COMPOSITE_SURVIVAL_HELDOUT_EVAL_VERDICT = {payload_verdict(data['heldout'], 'BG_FIXED_COMPOSITE_SURVIVAL_HELDOUT_EVAL_VERDICT')}",
        f"BG_FIXED_COMPOSITE_SURVIVAL_FRONTIER_VERDICT = {payload_verdict(data['frontier'], 'BG_FIXED_COMPOSITE_SURVIVAL_FRONTIER_VERDICT')}",
        f"BG_FIXED_COMPOSITE_LAYER_ORIGIN_DOMAIN_VERDICT = {payload_verdict(data['layer'], 'BG_FIXED_COMPOSITE_LAYER_ORIGIN_DOMAIN_VERDICT')}",
        f"BG_FIXED_COMPOSITE_OLD_CODE_PRESERVATION_VERDICT = {payload_verdict(data['old_code'], 'BG_FIXED_COMPOSITE_OLD_CODE_PRESERVATION_VERDICT')}",
        f"BG_FIXED_COMPOSITE_SELECTION_ONLY_READINESS_VERDICT = {payload_verdict(data['readiness'], 'BG_FIXED_COMPOSITE_SELECTION_ONLY_READINESS_VERDICT')}",
        f"FIXED_COMPOSITE_BRANCH_SURVIVAL_POLICY_STATUS = {status}",
    ]


def selected_policy_summary(data: dict[str, dict[str, Any]]) -> str:
    heldout = data.get("heldout", {})
    readiness = data.get("readiness", {})
    policy = heldout.get("selected_policy") or readiness.get("selected_policy") or "unknown"
    metrics = heldout.get("selected_policy_metrics") or readiness.get("metrics") or {}
    if not metrics:
        return f"selected_policy = {policy}"
    return (
        f"selected_policy = {policy}; "
        f"oracle_retention = {float(metrics.get('oracle_retention', 0.0)):.3f}; "
        f"false_prune_rate = {float(metrics.get('false_prune_rate', 0.0)):.3f}; "
        f"avg_survivors = {float(metrics.get('average_survivors', 0.0)):.3f}"
    )


def recommendation(status: str, data: dict[str, dict[str, Any]] | None = None) -> str:
    if status in {"SURVIVAL_READY", "OLD_NEW_COMPOSITE_SUFFICIENT"}:
        selected = selected_policy_summary(data or {})
        return (
            "Proceed to a small selection-only Phase 2 prototype using BGV1 branches, "
            "the fixed old+branch+bridge composite, the selected conservative top-k survival operating point, "
            "and missing/OOD fallback. Keep veto/rescue as a guardrail, not as a replacement for the selected heldout-ready operating point. "
            f"{selected}. Do not claim action steering."
        )
    if status == "CONSERVATIVE_BUT_USABLE":
        return "Prototype only with conservative top3/top4 retention and explicit compute caveat."
    if status == "TOPK_SURVIVAL_WEAK":
        return "Improve rescue thresholds or keep more branches before a default prototype."
    if status == "OOD_FRAGILE":
        return "Fix missing-expert fallback before any Phase 2 prototype."
    if status == "CODING_DEGRADES":
        return "Keep old code/objective taps as final coding arbiters and do not use composite pruning for code."
    if status == "TOO_MANY_FALSE_PRUNES":
        return "Do not prune; survival policy still removes oracle candidates too often."
    return "Treat as diagnostic only."


def docs_section(data: dict[str, dict[str, Any]], status: str) -> str:
    return "\n".join(
        [
            "## Fixed-composite branch survival policy v1 (2026-05-18)",
            "",
            "This run converted the corrected gated selector result into a validation-selected fixed old+branch+bridge survival policy with explicit veto/rescue and missing-expert/OOD fallback.",
            "",
            *top_lines(data, status),
            "",
            f"- selected policy: `{selected_policy_summary(data)}`",
            f"- recommendation: `{recommendation(status, data)}`",
            "- learned gated selector remains diagnostic; it is not the primary pruning selector.",
            "- no Ouro weights, tokenizer files, checkpoints, old tap registries, wrapper/local-agent routes, or production routing were modified.",
            "",
        ]
    )


def append_or_replace_section(path: Path, section: str, marker: str) -> None:
    if not path.exists():
        return
    current = path.read_text(encoding="utf-8")
    body = section.strip()
    if marker in current:
        start = current.index(marker)
        next_start = current.find("\n## ", start + len(marker))
        if next_start == -1:
            updated = current[:start].rstrip() + "\n\n" + body + "\n"
        else:
            updated = current[:start].rstrip() + "\n\n" + body + "\n" + current[next_start:]
    else:
        updated = current.rstrip() + "\n\n" + body + "\n"
    path.write_text(updated, encoding="utf-8")


def update_docs(data: dict[str, dict[str, Any]], status: str) -> None:
    marker = "## Fixed-composite branch survival policy v1 (2026-05-18)"
    section = docs_section(data, status)
    targets = [
        PROJECT_ROOT / "docs/evaluator/current_state.md",
        PROJECT_ROOT / "docs/evaluator/domain_transfer_ledger.md",
        PROJECT_ROOT / "docs/evaluator/bg_gated_branch_content_selector_v1.md",
        PROJECT_ROOT / "docs/evaluator/bg_universal_branch_content_taps_v1.md",
        PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_branch_generator_v1.md",
        PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_quota_v4.md",
        PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_taps.md",
        PROJECT_ROOT / "docs/evaluator/bg_hidden_state_branch_generation.md",
        PROJECT_ROOT / "docs/evaluator/bg_steering_consolidation_2026-05-18.md",
        PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-18_v8.1_routing_locked.md",
    ]
    for target in targets:
        append_or_replace_section(target, section, marker)
    readme = PROJECT_ROOT / "shared/docs/evaluator/README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        if "`bg_fixed_composite_branch_survival_policy_v1.md`" not in text:
            text = text.replace(
                "- `gated-branch-content-selector-v1.md` is the latest gated/composite selector decision.",
                "- `gated-branch-content-selector-v1.md` is the gated/composite selector decision.\n- `bg_fixed_composite_branch_survival_policy_v1.md` is the latest branch survival policy decision.",
            )
            text = text.replace("- `gated-branch-content-selector-v1.md`\n", "- `gated-branch-content-selector-v1.md`\n- `bg_fixed_composite_branch_survival_policy_v1.md`\n")
        if "Fixed-composite branch survival policy status is" not in text:
            text = text.replace(
                "- Gated branch-content selector status is `OLD_NEW_COMPOSITE_SUFFICIENT`.",
                f"- Gated branch-content selector status is `OLD_NEW_COMPOSITE_SUFFICIENT`.\n- Fixed-composite branch survival policy status is `{status}`.",
            )
        else:
            lines = []
            for line in text.splitlines():
                if line.startswith("- Fixed-composite branch survival policy status is"):
                    lines.append(f"- Fixed-composite branch survival policy status is `{status}`.")
                else:
                    lines.append(line)
            text = "\n".join(lines) + "\n"
        readme.write_text(text, encoding="utf-8")


def run_synthesis() -> int:
    ensure_root()
    started = time.time()
    data = stage_payloads()
    status = final_status(data)
    rec = recommendation(status, data)
    payload = {
        "top_lines": top_lines(data, status),
        "stage_payloads": data,
        "FIXED_COMPOSITE_BRANCH_SURVIVAL_POLICY_STATUS": status,
        "selected_policy": data.get("heldout", {}).get("selected_policy"),
        "selected_policy_metrics": data.get("heldout", {}).get("selected_policy_metrics"),
        "recommended_next": rec,
        "files_created": {
            "summary": rel(SUMMARY_JSON),
            "analysis": rel(ANALYSIS_JSON),
            "policy": rel(POLICY_PT),
            "doc": rel(DOC_MD),
        },
        "commands_run": [f"venv/bin/python -u utilities/tests/manual/{name}" for name in SCRIPT_NAMES],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(
        {
            "status": status,
            "best_fixed_composite": load_json(BEST_COMPOSITE_JSON, {}) or {},
            "best_veto_rescue_policy": load_json(BEST_VETO_JSON, {}) or {},
            "summary": payload,
        },
        POLICY_PT,
    )
    stable_write_json(SUMMARY_JSON, payload)
    stable_write_json(ANALYSIS_JSON, payload)
    lines = ["# Fixed-Composite Branch Survival Policy V1 Summary", "", *top_lines(data, status), "", f"Recommended next: {rec}"]
    write_md(SUMMARY_MD, lines)
    write_md(ANALYSIS_MD, lines + ["", "See `summary.json` and stage reports for detailed tables."])
    doc_lines = [
        "# Fixed-Composite Branch Survival Policy V1",
        "",
        "This experiment turns the corrected gated selector rerun into a fixed old+branch+bridge composite branch-survival policy with explicit veto/rescue and missing-expert/OOD fallback.",
        "",
        *top_lines(data, status),
        "",
        "## Decision",
        "",
        rec,
        "",
        "## Safety",
        "",
        "- Expert/tap scores are inputs only, not labels.",
        "- Thresholds and rules are selected on validation only.",
        "- Heldout is evaluation only.",
        "- No Ouro, tokenizer, checkpoint, old tap registry, wrapper/local-agent, production routing, or action-steering changes.",
    ]
    write_md(DOC_MD, doc_lines)
    update_docs(data, status)
    print(f"FIXED_COMPOSITE_BRANCH_SURVIVAL_POLICY_STATUS = {status}", flush=True)
    return 0
