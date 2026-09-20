"""Final arbiter among fixed-composite top4 survivors v1.

This experiment trains only small standalone final-arbiter models over cached
top4 survivor sets. It does not train Ouro, mutate checkpoints/tokenizers/tap
registries, run wrapper/local-agent code, import Hunter-Seeker modules, execute
ARC loops, or apply action steering.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from bg_branch_generator_v1_common import load_generator_rows
from bg_fixed_composite_survival_v1_common import (
    BEST_COMPOSITE_JSON,
    BEST_VETO_JSON,
    HELDOUT_JSON,
    POLICY_PT,
    composite_matrix,
    family_matrix,
    load_best_composite,
    load_dataset_records,
    matrix,
    net_from_matrix,
    oracle_indices,
    safe_float,
    topk_policy,
)
from bg_hidden_origin_quota_v4_common import PROBE_ROOT, PROJECT_ROOT, md_table, rel, write_csv, write_md
from bg_universal_tap_v1_common import load_json, load_v4_branch_rows


OUT_ROOT = PROBE_ROOT / "bg_final_arbiter_top4_survivors_v1_2026-05-18"
SEL_ROOT = PROBE_ROOT / "bg_selection_only_phase2_prototype_v1_2026-05-18"

INVENTORY_JSON = OUT_ROOT / "inventory.json"
INVENTORY_MD = OUT_ROOT / "inventory.md"
SURVIVOR_INVENTORY_CSV = OUT_ROOT / "survivor_inventory.csv"
DATASET_PT = OUT_ROOT / "final_arbiter_dataset.pt"
DATASET_JSON = OUT_ROOT / "final_arbiter_dataset.json"
DATASET_MD = OUT_ROOT / "final_arbiter_dataset.md"
FEATURES_PT = OUT_ROOT / "final_arbiter_features.pt"
FEATURES_JSON = OUT_ROOT / "final_arbiter_features.json"
FEATURES_MD = OUT_ROOT / "final_arbiter_features.md"
BASELINES_JSON = OUT_ROOT / "baseline_arbiters.json"
BASELINES_MD = OUT_ROOT / "baseline_arbiters.md"
BASELINES_CSV = OUT_ROOT / "baseline_arbiter_rows.csv"
MODEL_PT = OUT_ROOT / "final_arbiter_top4_v1.pt"
TRAINING_JSON = OUT_ROOT / "training_log.json"
TRAINING_MD = OUT_ROOT / "training_report.md"
HELDOUT_JSON_OUT = OUT_ROOT / "heldout_eval.json"
HELDOUT_MD = OUT_ROOT / "heldout_eval.md"
HELDOUT_CSV = OUT_ROOT / "heldout_eval_rows.csv"
DOMAIN_JSON = OUT_ROOT / "domain_analysis.json"
DOMAIN_MD = OUT_ROOT / "domain_analysis.md"
EXPERT_ABLATION_JSON = OUT_ROOT / "expert_ablation.json"
EXPERT_ABLATION_MD = OUT_ROOT / "expert_ablation.md"
EXPERT_ABLATION_CSV = OUT_ROOT / "expert_ablation_rows.csv"
CALIBRATION_JSON = OUT_ROOT / "calibration_ood.json"
CALIBRATION_MD = OUT_ROOT / "calibration_ood.md"
FAILURE_JSON = OUT_ROOT / "failure_analysis.json"
FAILURE_MD = OUT_ROOT / "failure_analysis.md"
FAILURE_CSV = OUT_ROOT / "failure_cases.csv"
READINESS_JSON = OUT_ROOT / "selection_only_readiness_update.json"
READINESS_MD = OUT_ROOT / "selection_only_readiness_update.md"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
ANALYSIS_JSON = OUT_ROOT / "analysis.json"
ANALYSIS_MD = OUT_ROOT / "analysis.md"
DOC_MD = PROJECT_ROOT / "docs/evaluator/bg_final_arbiter_top4_survivors_v1.md"

SELECTION_OUTPUTS_PT = SEL_ROOT / "selection_only_policy_outputs.pt"
SELECTION_OUTPUTS_JSON = SEL_ROOT / "selection_only_policy_outputs.json"
LIVE_SELECTED_CSV = SEL_ROOT / "live_selected_branches.csv"
SELECTION_FINAL_ARBITER_JSON = SEL_ROOT / "final_arbiter_analysis.json"
SELECTION_BASELINE_JSON = SEL_ROOT / "baseline_comparison.json"
SELECTION_DOMAIN_JSON = SEL_ROOT / "domain_coding_analysis.json"

POLICY_NAME = "fixed_composite_conservative_top4"
SEED = 20260518

DIRECT_EXPERTS = [
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
]
FAMILY_EXPERTS = ["fixed_composite", "old", "code", "branch", "bridge", "universal_family", "gated_family"]
SCORE_KEYS = [
    "fixed_composite_score",
    "old_score",
    "code_score",
    "branch_score",
    "bridge_score",
    "universal_score",
    "gated_score",
    "old_frozen_bg_score",
    "old_objective_mixed_score",
    "old_code_reasoning_score",
    "v4_hidden_origin_score",
    "bridge_only_score",
    "hidden_branch_score",
    "generator_v1_selector_score",
]
DOMAINS = ["reasoning", "science", "math_simple_arithmetic", "coding"]
LAYERS = ["L24", "L36", "L47", "old_context"]
ORIGINS = ["clean_branch", "hidden_branch", "code_candidate", "old_content", "unknown"]

SCRIPT_NAMES = [
    "bg_final_arbiter_inventory_v1.py",
    "build_bg_final_arbiter_dataset_v1.py",
    "build_bg_final_arbiter_features_v1.py",
    "run_bg_final_arbiter_baselines_v1.py",
    "train_bg_final_arbiter_top4_v1.py",
    "evaluate_bg_final_arbiter_top4_v1.py",
    "analyze_bg_final_arbiter_domains_v1.py",
    "analyze_bg_final_arbiter_expert_ablation_v1.py",
    "analyze_bg_final_arbiter_calibration_ood_v1.py",
    "analyze_bg_final_arbiter_failures_v1.py",
    "analyze_bg_final_arbiter_selection_only_readiness_v1.py",
    "analyze_bg_final_arbiter_top4_survivors_v1.py",
]

_FEATURE_PAYLOAD_CACHE: dict[str, Any] | None = None

DOC_TARGETS = [
    PROJECT_ROOT / "docs/evaluator/current_state.md",
    PROJECT_ROOT / "docs/evaluator/domain_transfer_ledger.md",
    PROJECT_ROOT / "docs/evaluator/bg_selection_only_phase2_prototype_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_fixed_composite_branch_survival_policy_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_gated_branch_content_selector_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_universal_branch_content_taps_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_branch_generator_v1.md",
    PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_quota_v4.md",
    PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_taps.md",
    PROJECT_ROOT / "docs/evaluator/bg_hidden_state_branch_generation.md",
    PROJECT_ROOT / "docs/evaluator/bg_steering_consolidation_2026-05-18.md",
    PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-18_v8.1_routing_locked.md",
]


def ensure_root() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
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


def finite_mean(values: Iterable[Any], default: float = float("nan")) -> float:
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(mean(vals)) if vals else default


def stable_hash(value: str) -> int:
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:12], 16)


def task_split_assignments(records: Sequence[dict[str, Any]]) -> dict[str, str]:
    by_task: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_task[str(record.get("task_id"))].add(str(record.get("split")))
    out = {}
    for task_id, splits in by_task.items():
        if "heldout" in splits:
            out[task_id] = "heldout"
        elif "val" in splits:
            out[task_id] = "val"
        else:
            out[task_id] = "train"
    return out


def score_values(record: dict[str, Any], key: str, weights: dict[str, float]) -> list[float]:
    if key == "fixed_composite":
        return net_from_matrix(composite_matrix(record, weights))
    family = {
        "old": "old",
        "code": "code",
        "branch": "branch",
        "bridge": "bridge",
        "universal_family": "universal",
        "gated_family": "gated",
    }.get(key)
    if family:
        return net_from_matrix(family_matrix(record, family))
    return net_from_matrix(matrix(record, key))


def rank_map(scores: dict[str, float]) -> dict[str, int]:
    ordered = sorted(scores, key=lambda cid: (-safe_float(scores[cid], 0.0), cid))
    return {cid: rank + 1 for rank, cid in enumerate(ordered)}


def clean_candidate_id(candidates: Sequence[dict[str, Any]]) -> str | None:
    for candidate in candidates:
        if candidate.get("branch_origin") in {"clean_branch", "clean"}:
            return str(candidate.get("candidate_id"))
    return str(candidates[0].get("candidate_id")) if candidates else None


def raw_branch_output_index() -> dict[str, dict[str, Any]]:
    rows = []
    for source_id, source_rows in (("branch_generator_v1", load_generator_rows()), ("quota_v4", load_v4_branch_rows())):
        for row in source_rows:
            norm = dict(row)
            norm["source_id_for_candidate"] = source_id
            rows.append(norm)
    out = {}
    for row in rows:
        gid = str(row.get("branch_group_id"))
        bid = str(row.get("branch_id"))
        key = f"{row.get('source_id_for_candidate')}::{gid}::{bid}"
        out[key] = row
    return out


def candidate_raw(candidate_id: str, raw_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if candidate_id in raw_index:
        return raw_index[candidate_id]
    parts = candidate_id.split("::")
    if len(parts) >= 3:
        key = f"{parts[0]}::{'::'.join(parts[1:-1])}::{parts[-1]}"
        return raw_index.get(key, {})
    return {}


def available_score_keys(candidate: dict[str, Any]) -> list[str]:
    return [key for key in SCORE_KEYS if math.isfinite(safe_float(candidate.get(key), float("nan")))]


def make_survivor_sets(records: Sequence[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    records = list(records or load_dataset_records())
    split_for_task = task_split_assignments(records)
    weights = load_best_composite()
    raw_index = raw_branch_output_index()
    sets = []
    for record in records:
        n = int(record.get("candidate_count") or len(record.get("candidates") or []))
        if n < 2:
            continue
        selected = topk_policy(record, "fixed_composite", 4, weights)
        if not selected:
            continue
        all_score_arrays: dict[str, list[float]] = {}
        for family in FAMILY_EXPERTS:
            all_score_arrays[family] = score_values(record, family, weights)
        for expert in DIRECT_EXPERTS:
            all_score_arrays[expert] = score_values(record, expert, weights)
        raw_candidates = record.get("candidates") or []
        selected_candidates: list[dict[str, Any]] = []
        for idx in selected:
            compact = raw_candidates[idx]
            cid = str(compact.get("candidate_id"))
            raw = candidate_raw(cid, raw_index)
            reward = safe_float(compact.get("reward"), 0.0)
            correctness = 1.0 if bool(compact.get("correct", reward > 0.0)) else 0.0
            cand = {
                "candidate_index": int(idx),
                "candidate_id": cid,
                "task_id": str(record.get("task_id")),
                "domain": str(record.get("domain")),
                "branch_origin": str(compact.get("origin") or "unknown"),
                "branch_point": str(compact.get("branch_point") or record.get("layer") or "unknown"),
                "layer": str(record.get("layer")),
                "generator_method": str(raw.get("generator_method") or compact.get("generator_method") or "unknown"),
                "alpha": raw.get("alpha"),
                "delta_family": raw.get("delta_family"),
                "output_text": raw.get("output_text", ""),
                "output_length": int(raw.get("output_length") or len(str(raw.get("output_text", "")).split())),
                "parsed_answer": raw.get("parsed_answer"),
                "final_reward": reward,
                "correctness": correctness,
                "coding_label": compact.get("label") if str(record.get("domain")) == "coding" else None,
                "parse_success": bool(compact.get("parse_success", True)),
                "repetition_rate": safe_float(compact.get("repetition_rate"), 0.0),
                "empty_output": bool(compact.get("empty_output", False)),
                "hit_max_tokens": bool(raw.get("hit_max_tokens", False)),
                "parse_failure_reason": raw.get("parse_failure_reason", ""),
                "stability_flags": {
                    "parse_success": bool(compact.get("parse_success", True)),
                    "empty_output": bool(compact.get("empty_output", False)),
                    "high_repetition": safe_float(compact.get("repetition_rate"), 0.0) >= 0.75,
                    "hit_max_tokens": bool(raw.get("hit_max_tokens", False)),
                    "nan_inf": bool(raw.get("nan_inf", False)),
                    "off_manifold_warning": bool(raw.get("off_manifold_warning", False)),
                },
                "hidden_distance_from_clean": raw.get("branch_hidden_distance_from_clean"),
                "logit_kl_from_clean": raw.get("branch_logit_kl_from_clean"),
                "fixed_composite_score": all_score_arrays["fixed_composite"][idx] if idx < len(all_score_arrays["fixed_composite"]) else 0.0,
                "old_score": all_score_arrays["old"][idx] if idx < len(all_score_arrays["old"]) else 0.0,
                "code_score": all_score_arrays["code"][idx] if idx < len(all_score_arrays["code"]) else 0.0,
                "branch_score": all_score_arrays["branch"][idx] if idx < len(all_score_arrays["branch"]) else 0.0,
                "bridge_score": all_score_arrays["bridge"][idx] if idx < len(all_score_arrays["bridge"]) else 0.0,
                "universal_score": all_score_arrays["universal_family"][idx] if idx < len(all_score_arrays["universal_family"]) else 0.0,
                "gated_score": all_score_arrays["gated_family"][idx] if idx < len(all_score_arrays["gated_family"]) else 0.0,
                "old_frozen_bg_score": all_score_arrays["old_frozen_bg"][idx] if idx < len(all_score_arrays["old_frozen_bg"]) else 0.0,
                "old_objective_mixed_score": all_score_arrays["mixed_objective_all_head"][idx] if idx < len(all_score_arrays["mixed_objective_all_head"]) else 0.0,
                "old_code_reasoning_score": all_score_arrays["mixed_code_reasoning_head"][idx] if idx < len(all_score_arrays["mixed_code_reasoning_head"]) else 0.0,
                "v4_hidden_origin_score": all_score_arrays["v4_hidden_origin"][idx] if idx < len(all_score_arrays["v4_hidden_origin"]) else 0.0,
                "bridge_only_score": all_score_arrays["bridge_only_head"][idx] if idx < len(all_score_arrays["bridge_only_head"]) else 0.0,
                "hidden_branch_score": all_score_arrays["hidden_branch_head"][idx] if idx < len(all_score_arrays["hidden_branch_head"]) else 0.0,
                "generator_v1_selector_score": all_score_arrays["generator_v1_selector"][idx] if idx < len(all_score_arrays["generator_v1_selector"]) else 0.0,
            }
            selected_candidates.append(cand)
        if not selected_candidates:
            continue
        rewards = [safe_float(c["final_reward"], 0.0) for c in selected_candidates]
        best_reward = max(rewards)
        best_ids = [str(c["candidate_id"]) for c in selected_candidates if safe_float(c["final_reward"], 0.0) == best_reward]
        for score_key in SCORE_KEYS:
            ranks = rank_map({str(c["candidate_id"]): safe_float(c.get(score_key), 0.0) for c in selected_candidates})
            for c in selected_candidates:
                c[f"rank_{score_key.removesuffix('_score')}"] = ranks[str(c["candidate_id"])]
        majority_ranks = {}
        for c in selected_candidates:
            ranks = [int(c.get(f"rank_{key.removesuffix('_score')}", len(selected_candidates))) for key in SCORE_KEYS if key in available_score_keys(c)]
            majority_ranks[str(c["candidate_id"])] = float(sum(ranks) / max(len(ranks), 1))
        majority_order = sorted(majority_ranks, key=lambda cid: (majority_ranks[cid], cid))
        for rank, cid in enumerate(majority_order, 1):
            for c in selected_candidates:
                if str(c["candidate_id"]) == cid:
                    c["rank_majority"] = rank
                    c["majority_rank_score"] = -float(rank)
        current_fixed = min(selected_candidates, key=lambda c: int(c.get("rank_fixed_composite", 999)))
        current_majority = min(selected_candidates, key=lambda c: int(c.get("rank_majority", 999)))
        task_id = str(record.get("task_id"))
        split = split_for_task.get(task_id, str(record.get("split")))
        set_id = str(record.get("candidate_set_id"))
        sets.append(
            {
                "survivor_set_id": set_id,
                "candidate_set_id": set_id,
                "task_id": task_id,
                "domain": str(record.get("domain")),
                "original_split": str(record.get("split")),
                "split": split,
                "source_status": "reused_heldout" if split == "heldout" else "reused_diagnostic",
                "layer": str(record.get("layer")),
                "source_id": str(record.get("source_id")),
                "pair_type": str(record.get("pair_type")),
                "group_id": str(record.get("group_id")),
                "survivor_count": len(selected_candidates),
                "oracle_in_top4": bool(best_ids),
                "oracle_count": len(best_ids),
                "tie_count": len(best_ids),
                "best_reward": best_reward,
                "best_candidate_ids": best_ids,
                "current_fixed_choice": str(current_fixed.get("candidate_id")),
                "current_majority_choice": str(current_majority.get("candidate_id")),
                "current_arbiter_failed_despite_oracle_in_top4": str(current_majority.get("candidate_id")) not in set(best_ids),
                "branch_origins": dict(Counter(str(c.get("branch_origin")) for c in selected_candidates)),
                "available_expert_scores": sorted({key for c in selected_candidates for key in available_score_keys(c)}),
                "missing_expert_mask": {
                    key: not any(math.isfinite(safe_float(c.get(key), float("nan"))) for c in selected_candidates)
                    for key in SCORE_KEYS
                },
                "candidates": selected_candidates,
            }
        )
    return sets


def compact_set(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "survivor_set_id": row.get("survivor_set_id"),
        "task_id": row.get("task_id"),
        "domain": row.get("domain"),
        "split": row.get("split"),
        "original_split": row.get("original_split"),
        "source_status": row.get("source_status"),
        "survivor_count": row.get("survivor_count"),
        "oracle_in_top4": row.get("oracle_in_top4"),
        "oracle_count": row.get("oracle_count"),
        "tie_count": row.get("tie_count"),
        "best_reward": row.get("best_reward"),
        "current_arbiter_failed_despite_oracle_in_top4": row.get("current_arbiter_failed_despite_oracle_in_top4"),
        "branch_origins": row.get("branch_origins"),
        "available_expert_scores": row.get("available_expert_scores"),
    }


def aggregate_counts(sets: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "survivor_sets": len(sets),
        "tasks": len({str(s.get("task_id")) for s in sets}),
        "domains": dict(Counter(str(s.get("domain")) for s in sets)),
        "splits": dict(Counter(str(s.get("split")) for s in sets)),
        "sets_by_domain_split": dict(Counter(f"{s.get('domain')}::{s.get('split')}" for s in sets)),
        "survivor_candidates": sum(len(s.get("candidates") or []) for s in sets),
        "oracle_in_top4_rate": finite_mean([1.0 if s.get("oracle_in_top4") else 0.0 for s in sets], 0.0),
        "current_arbiter_failure_rate": finite_mean([1.0 if s.get("current_arbiter_failed_despite_oracle_in_top4") else 0.0 for s in sets], 0.0),
        "tie_distribution": dict(Counter(int(s.get("tie_count") or 0) for s in sets)),
        "coding_sets": sum(1 for s in sets if s.get("domain") == "coding"),
        "science_sets": sum(1 for s in sets if s.get("domain") == "science"),
    }


def run_inventory() -> int:
    ensure_root()
    started = time.time()
    artifacts = [
        SELECTION_OUTPUTS_PT,
        SELECTION_OUTPUTS_JSON,
        LIVE_SELECTED_CSV,
        SELECTION_FINAL_ARBITER_JSON,
        SELECTION_BASELINE_JSON,
        SELECTION_DOMAIN_JSON,
        POLICY_PT,
        BEST_COMPOSITE_JSON,
        BEST_VETO_JSON,
        HELDOUT_JSON,
    ]
    artifact_rows = [{"path": rel(path), "exists": path.exists(), "size": path.stat().st_size if path.exists() else 0} for path in artifacts]
    sets = make_survivor_sets()
    counts = aggregate_counts(sets)
    coverage = Counter()
    missing = Counter()
    for s in sets:
        for key in s.get("available_expert_scores") or []:
            coverage[key] += 1
        for key, is_missing in (s.get("missing_expert_mask") or {}).items():
            if is_missing:
                missing[key] += 1
    split_ok = counts["splits"].get("train", 0) > 0 and counts["splits"].get("val", 0) > 0 and counts["splits"].get("heldout", 0) > 0
    domain_ok = counts["domains"].get("coding", 0) > 0 and counts["domains"].get("science", 0) > 0
    missing_required = [row["path"] for row in artifact_rows if not row["exists"]]
    if not sets:
        verdict = "BLOCKED"
    elif missing_required:
        verdict = "PARTIAL"
    elif split_ok and domain_ok and counts["survivor_sets"] >= 150:
        verdict = "READY"
    elif split_ok:
        verdict = "PARTIAL"
    else:
        verdict = "DATA_LIMITED"
    survivor_rows = [compact_set(s) for s in sets]
    payload = {
        "BG_FINAL_ARBITER_INVENTORY_VERDICT": verdict,
        "verdict": verdict,
        "counts": counts,
        "artifact_rows": artifact_rows,
        "expert_score_coverage": dict(coverage),
        "missing_expert_coverage": dict(missing),
        "survivor_sets_sample": survivor_rows[:200],
        "known_limitations": [
            "Uses cached top4 survivor sets, not fresh branch generation.",
            "Task-disjoint split assignment moves tasks with any heldout records entirely to heldout.",
            "No true branch-batch fork/carry or compute-saving claim is made.",
        ],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(INVENTORY_JSON, payload)
    write_csv(SURVIVOR_INVENTORY_CSV, survivor_rows)
    lines = ["# Final Arbiter Inventory", "", f"BG_FINAL_ARBITER_INVENTORY_VERDICT = {verdict}", "", f"- counts: `{counts}`", "", "## Artifacts", ""]
    lines.extend(md_table(artifact_rows, ["path", "exists", "size"]))
    lines.extend(["", "## Survivor Sets", ""])
    lines.extend(md_table(survivor_rows[:80], ["task_id", "domain", "split", "survivor_count", "oracle_count", "tie_count", "current_arbiter_failed_despite_oracle_in_top4"]))
    write_md(INVENTORY_MD, lines)
    print(f"BG_FINAL_ARBITER_INVENTORY_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


def build_pair_rows(sets: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = []
    for s in sets:
        candidates = s.get("candidates") or []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                ri = safe_float(candidates[i].get("final_reward"), 0.0)
                rj = safe_float(candidates[j].get("final_reward"), 0.0)
                if ri == rj:
                    continue
                pref, rej = (candidates[i], candidates[j]) if ri > rj else (candidates[j], candidates[i])
                pairs.append(
                    {
                        "survivor_set_id": s.get("survivor_set_id"),
                        "task_id": s.get("task_id"),
                        "domain": s.get("domain"),
                        "split": s.get("split"),
                        "preferred_candidate_id": pref.get("candidate_id"),
                        "rejected_candidate_id": rej.get("candidate_id"),
                        "reward_gap": safe_float(pref.get("final_reward"), 0.0) - safe_float(rej.get("final_reward"), 0.0),
                    }
                )
    return pairs


def run_dataset() -> int:
    ensure_root()
    started = time.time()
    if not INVENTORY_JSON.exists():
        run_inventory()
    sets = make_survivor_sets()
    pairs = build_pair_rows(sets)
    counts = aggregate_counts(sets)
    counts.update(
        {
            "pairwise_pairs": len(pairs),
            "pairwise_by_split": dict(Counter(str(p.get("split")) for p in pairs)),
            "pairwise_by_domain": dict(Counter(str(p.get("domain")) for p in pairs)),
            "candidates_by_split": dict(Counter(str(s.get("split")) for s in sets for _ in (s.get("candidates") or []))),
            "listwise_sets_by_split": dict(Counter(str(s.get("split")) for s in sets)),
        }
    )
    if counts["survivor_sets"] >= 150 and counts["splits"].get("train", 0) and counts["splits"].get("val", 0) and counts["splits"].get("heldout", 0):
        verdict = "READY"
    elif counts["survivor_sets"] >= 60:
        verdict = "SMALL_BUT_USABLE"
    elif counts["survivor_sets"]:
        verdict = "DATA_LIMITED"
    else:
        verdict = "BLOCKED"
    payload = {
        "BG_FINAL_ARBITER_DATASET_VERDICT": verdict,
        "verdict": verdict,
        "survivor_sets": sets,
        "pairwise_pairs": pairs,
        "counts": counts,
        "label_policy": "final reward/correctness only; tap scores are input features, not labels; exact reward ties omitted from pairwise training",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(payload, DATASET_PT)
    write_json(DATASET_JSON, {k: v for k, v in payload.items() if k not in {"survivor_sets"}} | {"survivor_sets_sample": [compact_set(s) for s in sets[:200]]})
    lines = ["# Final Arbiter Dataset", "", f"BG_FINAL_ARBITER_DATASET_VERDICT = {verdict}", "", f"- counts: `{counts}`", "", "## Split/Domain Counts", ""]
    split_domain_rows = [{"key": k, "sets": v} for k, v in counts.get("sets_by_domain_split", {}).items()]
    lines.extend(md_table(split_domain_rows, ["key", "sets"]))
    write_md(DATASET_MD, lines)
    print(f"BG_FINAL_ARBITER_DATASET_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


def load_dataset_payload() -> dict[str, Any]:
    if not DATASET_PT.exists():
        run_dataset()
    return torch.load(DATASET_PT, map_location="cpu", weights_only=False)


def one_hot(value: str, choices: Sequence[str]) -> list[float]:
    return [1.0 if value == choice else 0.0 for choice in choices]


def feature_dict_for_candidate(candidate: dict[str, Any], survivor_set: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in SCORE_KEYS:
        out[key] = safe_float(candidate.get(key), 0.0)
    k = max(int(survivor_set.get("survivor_count") or 1), 1)
    for key in SCORE_KEYS:
        rank = safe_float(candidate.get(f"rank_{key.removesuffix('_score')}"), float(k))
        out[f"rank_{key.removesuffix('_score')}"] = rank / max(k, 1)
        out[f"is_top1_{key.removesuffix('_score')}"] = 1.0 if int(rank) == 1 else 0.0
        out[f"is_top2_{key.removesuffix('_score')}"] = 1.0 if int(rank) <= 2 else 0.0
        scores = [safe_float(c.get(key), 0.0) for c in survivor_set.get("candidates") or []]
        out[f"margin_to_best_{key.removesuffix('_score')}"] = safe_float(candidate.get(key), 0.0) - max(scores or [0.0])
        out[f"score_gap_to_mean_{key.removesuffix('_score')}"] = safe_float(candidate.get(key), 0.0) - finite_mean(scores, 0.0)
    clean_id = clean_candidate_id(survivor_set.get("candidates") or [])
    clean = next((c for c in survivor_set.get("candidates") or [] if str(c.get("candidate_id")) == str(clean_id)), None)
    if clean:
        for key in SCORE_KEYS:
            out[f"score_gap_to_clean_{key.removesuffix('_score')}"] = safe_float(candidate.get(key), 0.0) - safe_float(clean.get(key), 0.0)
    expert_scores = [safe_float(candidate.get(key), 0.0) for key in SCORE_KEYS]
    expert_ranks = [safe_float(candidate.get(f"rank_{key.removesuffix('_score')}"), 0.0) for key in SCORE_KEYS]
    out["expert_score_mean"] = finite_mean(expert_scores, 0.0)
    out["expert_score_variance"] = finite_mean([(x - out["expert_score_mean"]) ** 2 for x in expert_scores], 0.0)
    out["rank_mean"] = finite_mean(expert_ranks, 0.0)
    out["rank_variance"] = finite_mean([(x - out["rank_mean"]) ** 2 for x in expert_ranks], 0.0)
    out["experts_top1_count"] = sum(1.0 for key in SCORE_KEYS if int(safe_float(candidate.get(f"rank_{key.removesuffix('_score')}"), 99)) == 1)
    out["experts_top2_count"] = sum(1.0 for key in SCORE_KEYS if int(safe_float(candidate.get(f"rank_{key.removesuffix('_score')}"), 99)) <= 2)
    out["old_vs_bridge_disagreement"] = abs(out["old_score"] - out["bridge_score"])
    out["old_vs_hidden_disagreement"] = abs(out["old_score"] - out["branch_score"])
    out["bridge_vs_hidden_disagreement"] = abs(out["bridge_score"] - out["branch_score"])
    out["gated_vs_fixed_disagreement"] = abs(out["gated_score"] - out["fixed_composite_score"])
    out["code_vs_fixed_disagreement"] = abs(out["code_score"] - out["fixed_composite_score"])
    for i, value in enumerate(one_hot(str(survivor_set.get("domain")), DOMAINS)):
        out[f"domain_{DOMAINS[i]}"] = value
    for i, value in enumerate(one_hot(str(survivor_set.get("layer")), LAYERS)):
        out[f"layer_{LAYERS[i]}"] = value
    origin = str(candidate.get("branch_origin") or "unknown")
    if origin not in ORIGINS:
        origin = "unknown"
    for i, value in enumerate(one_hot(origin, ORIGINS)):
        out[f"origin_{ORIGINS[i]}"] = value
    out["clean_branch_flag"] = 1.0 if candidate.get("branch_origin") == "clean_branch" else 0.0
    out["code_candidate_flag"] = 1.0 if candidate.get("branch_origin") == "code_candidate" else 0.0
    out["old_candidate_flag"] = 1.0 if candidate.get("branch_origin") in {"old_content", "code_candidate"} else 0.0
    out["current_perturbation_flag"] = 1.0 if candidate.get("branch_origin") == "hidden_branch" else 0.0
    out["parse_success"] = 1.0 if candidate.get("parse_success", True) else 0.0
    out["repetition_rate"] = safe_float(candidate.get("repetition_rate"), 0.0)
    out["empty_output"] = 1.0 if candidate.get("empty_output") else 0.0
    out["hit_max_tokens"] = 1.0 if candidate.get("hit_max_tokens") else 0.0
    out["output_length_log"] = math.log1p(max(safe_float(candidate.get("output_length"), 0.0), 0.0))
    out["hidden_distance_from_clean"] = safe_float(candidate.get("hidden_distance_from_clean"), 0.0)
    out["logit_kl_from_clean"] = safe_float(candidate.get("logit_kl_from_clean"), 0.0)
    out["majority_rank_score"] = safe_float(candidate.get("majority_rank_score"), 0.0)
    return out


def build_feature_payload() -> dict[str, Any]:
    dataset = load_dataset_payload()
    sets = dataset.get("survivor_sets") or []
    feature_dicts = []
    candidate_rows = []
    for set_idx, survivor_set in enumerate(sets):
        for cand_idx, candidate in enumerate(survivor_set.get("candidates") or []):
            fmap = feature_dict_for_candidate(candidate, survivor_set)
            feature_dicts.append(fmap)
            candidate_rows.append(
                {
                    "set_idx": set_idx,
                    "candidate_local_idx": cand_idx,
                    "survivor_set_id": survivor_set.get("survivor_set_id"),
                    "task_id": survivor_set.get("task_id"),
                    "domain": survivor_set.get("domain"),
                    "split": survivor_set.get("split"),
                    "candidate_id": candidate.get("candidate_id"),
                    "reward": safe_float(candidate.get("final_reward"), 0.0),
                    "correctness": safe_float(candidate.get("correctness"), 0.0),
                    "is_best": 1.0 if str(candidate.get("candidate_id")) in set(survivor_set.get("best_candidate_ids") or []) else 0.0,
                    "parse_success": 1.0 if candidate.get("parse_success", True) else 0.0,
                    "stable": 1.0 if candidate.get("parse_success", True) and not candidate.get("empty_output") and safe_float(candidate.get("repetition_rate"), 0.0) < 0.75 else 0.0,
                }
            )
    feature_names = sorted({key for fmap in feature_dicts for key in fmap})
    x = torch.tensor([[safe_float(fmap.get(name), 0.0) for name in feature_names] for fmap in feature_dicts], dtype=torch.float32)
    train_indices = [i for i, row in enumerate(candidate_rows) if row["split"] == "train"]
    if train_indices:
        train_x = x[train_indices]
        mean_vec = train_x.mean(dim=0)
        std_vec = train_x.std(dim=0).clamp_min(1e-6)
    else:
        mean_vec = torch.zeros(x.shape[1], dtype=torch.float32)
        std_vec = torch.ones(x.shape[1], dtype=torch.float32)
    x_norm = (x - mean_vec) / std_vec
    offset = 0
    feature_sets = []
    for set_idx, survivor_set in enumerate(sets):
        n = len(survivor_set.get("candidates") or [])
        feature_sets.append(
            {
                **{k: v for k, v in survivor_set.items() if k != "candidates"},
                "candidate_rows": candidate_rows[offset : offset + n],
                "candidate_feature_indices": list(range(offset, offset + n)),
                "rewards": [candidate_rows[offset + j]["reward"] for j in range(n)],
                "best_mask": [candidate_rows[offset + j]["is_best"] for j in range(n)],
            }
        )
        offset += n
    return {
        "feature_names": feature_names,
        "x_raw": x,
        "x": x_norm,
        "calibration": {"mean": mean_vec, "std": std_vec, "fit_split": "train"},
        "candidate_rows": candidate_rows,
        "sets": feature_sets,
        "raw_sets": sets,
    }


def run_features() -> int:
    ensure_root()
    started = time.time()
    if not DATASET_PT.exists():
        run_dataset()
    payload = build_feature_payload()
    feature_names = payload["feature_names"]
    candidate_rows = payload["candidate_rows"]
    missing_heavy = False
    required = ["fixed_composite_score", "old_score", "branch_score", "bridge_score"]
    for key in required:
        idx = feature_names.index(key) if key in feature_names else -1
        if idx < 0:
            missing_heavy = True
    if not candidate_rows:
        verdict = "BLOCKED"
    elif missing_heavy:
        verdict = "MISSING_EXPERT_HEAVY"
    else:
        verdict = "READY"
    out = {
        "BG_FINAL_ARBITER_FEATURES_VERDICT": verdict,
        "verdict": verdict,
        **payload,
        "counts": {
            "feature_count": len(feature_names),
            "candidate_rows": len(candidate_rows),
            "sets": len(payload["sets"]),
            "by_split": dict(Counter(str(row.get("split")) for row in candidate_rows)),
            "by_domain": dict(Counter(str(row.get("domain")) for row in candidate_rows)),
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(out, FEATURES_PT)
    write_json(FEATURES_JSON, {k: v for k, v in out.items() if k not in {"x", "x_raw", "calibration", "raw_sets"}} | {"feature_names": feature_names})
    lines = ["# Final Arbiter Features", "", f"BG_FINAL_ARBITER_FEATURES_VERDICT = {verdict}", "", f"- feature count: `{len(feature_names)}`", f"- candidates: `{len(candidate_rows)}`", "", "## Feature Names", ""]
    lines.extend(f"- `{name}`" for name in feature_names)
    write_md(FEATURES_MD, lines)
    print(f"BG_FINAL_ARBITER_FEATURES_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


def load_features_payload() -> dict[str, Any]:
    global _FEATURE_PAYLOAD_CACHE
    if _FEATURE_PAYLOAD_CACHE is not None:
        return _FEATURE_PAYLOAD_CACHE
    if not FEATURES_PT.exists():
        run_features()
    _FEATURE_PAYLOAD_CACHE = torch.load(FEATURES_PT, map_location="cpu", weights_only=False)
    return _FEATURE_PAYLOAD_CACHE


def choose_by_score(s: dict[str, Any], score_key: str) -> int:
    rows = s.get("candidate_rows") or []
    if not rows:
        return 0
    raw = load_features_payload().get("raw_sets", [])[int(rows[0]["set_idx"])]["candidates"]
    return sorted(range(len(rows)), key=lambda i: (-safe_float(raw[i].get(score_key), 0.0), i))[0]


def deterministic_random_choice(s: dict[str, Any]) -> int:
    n = len(s.get("candidate_rows") or [])
    if n <= 1:
        return 0
    return stable_hash(str(s.get("survivor_set_id"))) % n


def majority_choice(s: dict[str, Any], *, keys: Sequence[str] = SCORE_KEYS) -> int:
    payload = load_features_payload()
    raw = payload.get("raw_sets", [])[int((s.get("candidate_rows") or [{"set_idx": 0}])[0]["set_idx"])]["candidates"]
    ranks = []
    for i, candidate in enumerate(raw):
        vals = [safe_float(candidate.get(f"rank_{key.removesuffix('_score')}"), 99.0) for key in keys]
        ranks.append((finite_mean(vals, 99.0), i))
    return sorted(ranks)[0][1] if ranks else 0


def score_normalized_choice(s: dict[str, Any], keys: Sequence[str] = SCORE_KEYS) -> int:
    payload = load_features_payload()
    raw = payload.get("raw_sets", [])[int((s.get("candidate_rows") or [{"set_idx": 0}])[0]["set_idx"])]["candidates"]
    if not raw:
        return 0
    totals = [0.0 for _ in raw]
    used = 0
    for key in keys:
        vals = [safe_float(c.get(key), 0.0) for c in raw]
        mu = finite_mean(vals, 0.0)
        var = finite_mean([(v - mu) ** 2 for v in vals], 0.0)
        sd = math.sqrt(var) if var > 1e-9 else 1.0
        for i, v in enumerate(vals):
            totals[i] += (v - mu) / sd
        used += 1
    return sorted(range(len(raw)), key=lambda i: (-totals[i] / max(used, 1), i))[0]


def domain_rule_choice(s: dict[str, Any]) -> int:
    domain = str(s.get("domain"))
    if domain == "coding":
        return score_normalized_choice(s, ["old_code_reasoning_score", "old_objective_mixed_score", "code_score", "fixed_composite_score"])
    if domain == "math_simple_arithmetic":
        return score_normalized_choice(s, ["old_score", "old_objective_mixed_score", "fixed_composite_score"])
    if str(s.get("pair_type")) == "hidden_branch":
        return score_normalized_choice(s, ["bridge_score", "hidden_branch_score", "branch_score", "fixed_composite_score"])
    return majority_choice(s)


def oracle_choice(s: dict[str, Any]) -> int:
    rewards = [safe_float(row.get("reward"), 0.0) for row in s.get("candidate_rows") or []]
    return sorted(range(len(rewards)), key=lambda i: (-rewards[i], i))[0] if rewards else 0


def baseline_choice(s: dict[str, Any], policy: str) -> int:
    rows = s.get("candidate_rows") or []
    if not rows:
        return 0
    if policy == "random_survivor":
        return deterministic_random_choice(s)
    if policy == "first_survivor" or policy == "fixed_composite_top1":
        return 0
    if policy == "clean_branch_if_present":
        raw = load_features_payload().get("raw_sets", [])[int(rows[0]["set_idx"])]["candidates"]
        for i, c in enumerate(raw):
            if c.get("branch_origin") == "clean_branch":
                return i
        return 0
    score_policy = {
        "old_frozen_bg_top1": "old_frozen_bg_score",
        "old_objective_mixed_top1": "old_objective_mixed_score",
        "old_code_reasoning_top1": "old_code_reasoning_score",
        "v4_hidden_origin_top1": "v4_hidden_origin_score",
        "bridge_top1": "bridge_score",
        "bridge_only_top1": "bridge_only_score",
        "hidden_branch_top1": "hidden_branch_score",
        "universal_top1": "universal_score",
        "learned_gated_top1": "gated_score",
    }
    if policy in score_policy:
        return choose_by_score(s, score_policy[policy])
    if policy in {"majority_rank_aggregation", "borda_rank_sum_aggregation"}:
        return majority_choice(s)
    if policy == "score_normalized_ensemble":
        return score_normalized_choice(s)
    if policy == "domain_rule_baseline":
        return domain_rule_choice(s)
    if policy in {"verifier_oracle_if_available", "oracle_best_survivor"}:
        return oracle_choice(s)
    return 0


def metrics_for_choices(sets: Sequence[dict[str, Any]], choices: dict[str, int], policy: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for s in sets:
        cand_rows = s.get("candidate_rows") or []
        if not cand_rows:
            continue
        idx = max(0, min(int(choices.get(str(s.get("survivor_set_id")), 0)), len(cand_rows) - 1))
        rewards = [safe_float(row.get("reward"), 0.0) for row in cand_rows]
        best = max(rewards)
        selected = cand_rows[idx]
        reward = safe_float(selected.get("reward"), 0.0)
        rows.append(
            {
                "policy": policy,
                "survivor_set_id": s.get("survivor_set_id"),
                "task_id": s.get("task_id"),
                "domain": s.get("domain"),
                "split": s.get("split"),
                "selected_candidate_id": selected.get("candidate_id"),
                "selected_index": idx,
                "final_selected_reward": reward,
                "final_selected_correctness": safe_float(selected.get("correctness"), 0.0),
                "top1_success": 1.0 if reward == best else 0.0,
                "oracle_selected_rate": 1.0 if reward == best else 0.0,
                "regret": best - reward,
                "best_survivor_reward": best,
                "parse_success_rate": safe_float(selected.get("parse_success"), 1.0),
                "stable_rate": safe_float(selected.get("stable"), 1.0),
            }
        )
    metrics = aggregate_rows(rows)
    return metrics.get(policy, {"n": 0}), rows


def aggregate_rows(rows: Sequence[dict[str, Any]], keys: Sequence[str] = ("policy",)) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped["::".join(str(row.get(k)) for k in keys)].append(row)
    out = {}
    for key, vals in grouped.items():
        nums: dict[str, list[float]] = defaultdict(list)
        for row in vals:
            for name, value in row.items():
                x = safe_float(value)
                if math.isfinite(x):
                    nums[name].append(x)
        task_macro = []
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in vals:
            by_task[str(row.get("task_id"))].append(row)
        for task_rows in by_task.values():
            task_macro.append(finite_mean([r.get("final_selected_reward") for r in task_rows], 0.0))
        out[key] = {
            "n": len(vals),
            **{name: finite_mean(values, 0.0) for name, values in nums.items()},
            "task_macro_final_reward": finite_mean(task_macro, 0.0),
        }
    return out


def baseline_policies() -> list[str]:
    return [
        "random_survivor",
        "first_survivor",
        "clean_branch_if_present",
        "fixed_composite_top1",
        "old_frozen_bg_top1",
        "old_objective_mixed_top1",
        "old_code_reasoning_top1",
        "v4_hidden_origin_top1",
        "bridge_top1",
        "bridge_only_top1",
        "hidden_branch_top1",
        "universal_top1",
        "learned_gated_top1",
        "majority_rank_aggregation",
        "borda_rank_sum_aggregation",
        "score_normalized_ensemble",
        "domain_rule_baseline",
        "verifier_oracle_if_available",
        "oracle_best_survivor",
    ]


def evaluate_baseline_rows(split: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = load_features_payload()
    sets = [s for s in payload["sets"] if split is None or s.get("split") == split]
    all_rows = []
    for policy in baseline_policies():
        choices = {str(s.get("survivor_set_id")): baseline_choice(s, policy) for s in sets}
        _, rows = metrics_for_choices(sets, choices, policy)
        all_rows.extend(rows)
    return all_rows, aggregate_rows(all_rows)


def run_baselines() -> int:
    ensure_root()
    started = time.time()
    if not FEATURES_PT.exists():
        run_features()
    rows, metrics = evaluate_baseline_rows()
    heldout_rows, heldout_metrics = evaluate_baseline_rows("heldout")
    verdict = "READY" if rows else "BLOCKED"
    payload = {
        "BG_FINAL_ARBITER_BASELINES_VERDICT": verdict,
        "verdict": verdict,
        "rows": rows,
        "metrics_by_policy": metrics,
        "heldout_metrics_by_policy": heldout_metrics,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(BASELINES_JSON, payload)
    write_csv(BASELINES_CSV, rows)
    table = [{"policy": k, **v} for k, v in sorted(heldout_metrics.items())]
    lines = ["# Baseline Final Arbiters", "", f"BG_FINAL_ARBITER_BASELINES_VERDICT = {verdict}", "", "## Heldout Metrics", ""]
    lines.extend(md_table(table, ["policy", "n", "final_selected_reward", "task_macro_final_reward", "top1_success", "regret"]))
    write_md(BASELINES_MD, lines)
    print(f"BG_FINAL_ARBITER_BASELINES_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


class LinearScorer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class TinyMLPScorer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = min(32, max(8, dim // 4))
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DomainGateScorer(nn.Module):
    def __init__(self, feature_names: Sequence[str]) -> None:
        super().__init__()
        self.feature_names = list(feature_names)
        self.expert_indices = [self.feature_names.index(k) for k in SCORE_KEYS if k in self.feature_names]
        self.domain_indices = [self.feature_names.index(f"domain_{d}") for d in DOMAINS if f"domain_{d}" in self.feature_names]
        self.weights = nn.Parameter(torch.zeros((len(self.domain_indices), len(self.expert_indices))))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        experts = x[:, self.expert_indices]
        domains = x[:, self.domain_indices]
        w = domains @ self.weights
        return (experts * w).sum(dim=-1) + self.bias


def split_sets(split: str) -> list[dict[str, Any]]:
    return [s for s in load_features_payload()["sets"] if s.get("split") == split]


def set_loss(scores: torch.Tensor, rewards: Sequence[float]) -> torch.Tensor:
    r = torch.tensor([safe_float(v, 0.0) for v in rewards], dtype=torch.float32)
    best = float(r.max().item()) if r.numel() else 0.0
    target = (r == best).to(torch.float32)
    target = target / target.sum().clamp_min(1.0)
    return -(target * F.log_softmax(scores, dim=0)).sum()


def pairwise_loss(scores: torch.Tensor, rewards: Sequence[float]) -> torch.Tensor:
    losses = []
    for i in range(len(rewards)):
        for j in range(i + 1, len(rewards)):
            ri, rj = safe_float(rewards[i], 0.0), safe_float(rewards[j], 0.0)
            if ri == rj:
                continue
            sign = 1.0 if ri > rj else -1.0
            losses.append(F.softplus(-sign * (scores[i] - scores[j])) * abs(ri - rj))
    return torch.stack(losses).mean() if losses else scores.sum() * 0.0


def evaluate_model(model: nn.Module, sets: Sequence[dict[str, Any]], x: torch.Tensor, policy: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    choices = {}
    with torch.no_grad():
        for s in sets:
            indices = s.get("candidate_feature_indices") or []
            scores = model(x[indices]) if indices else torch.empty(0)
            choice = int(torch.argmax(scores).item()) if scores.numel() else 0
            choices[str(s.get("survivor_set_id"))] = choice
    return metrics_for_choices(sets, choices, policy)


def train_one_model(family: str, seed: int, lr: float, epochs: int = 100) -> dict[str, Any]:
    payload = load_features_payload()
    x = payload["x"]
    feature_names = payload["feature_names"]
    train_sets = split_sets("train")
    val_sets = split_sets("val")
    torch.manual_seed(seed)
    if family in {"pairwise_logistic", "listwise_softmax", "ranknet_linear", "residual_linear"}:
        model: nn.Module = LinearScorer(x.shape[1])
    elif family == "domain_gated":
        model = DomainGateScorer(feature_names)
    elif family == "tiny_mlp":
        model = TinyMLPScorer(x.shape[1])
    else:
        model = LinearScorer(x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best_state = None
    best_val = -999.0
    best_epoch = 0
    train_log = []
    patience = 20
    stale = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        count = 0
        order = sorted(range(len(train_sets)), key=lambda i: stable_hash(f"{seed}:{epoch}:{train_sets[i].get('survivor_set_id')}"))
        for set_idx in order:
            s = train_sets[set_idx]
            indices = s.get("candidate_feature_indices") or []
            if len(indices) < 2:
                continue
            scores = model(x[indices])
            rewards = s.get("rewards") or []
            loss = pairwise_loss(scores, rewards) if family in {"pairwise_logistic", "ranknet_linear"} else set_loss(scores, rewards)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.detach().item())
            count += 1
        train_metrics, _ = evaluate_model(model, train_sets, x, f"{family}_train")
        val_metrics, _ = evaluate_model(model, val_sets, x, f"{family}_val")
        val_score = safe_float(val_metrics.get("task_macro_final_reward"), -999.0)
        train_log.append({"epoch": epoch, "loss": total_loss / max(count, 1), "train": train_metrics, "val": val_metrics})
        if val_score > best_val + 1e-9:
            best_val = val_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    val_metrics, val_rows = evaluate_model(model, val_sets, x, family)
    train_metrics, _ = evaluate_model(model, train_sets, x, family)
    return {
        "family": family,
        "seed": seed,
        "lr": lr,
        "best_epoch": best_epoch,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "val_rows": val_rows,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_class": type(model).__name__,
        "train_log": train_log,
    }


def run_training() -> int:
    ensure_root()
    started = time.time()
    if not FEATURES_PT.exists():
        run_features()
    if not BASELINES_JSON.exists():
        run_baselines()
    payload = load_features_payload()
    families = ["pairwise_logistic", "listwise_softmax", "ranknet_linear", "domain_gated", "residual_linear", "tiny_mlp"]
    rows = []
    runs = []
    for family in families:
        for lr in (1e-4, 3e-4, 1e-3):
            for seed in (SEED, SEED + 1, SEED + 2):
                run = train_one_model(family, seed, lr)
                runs.append(run)
                rows.append(
                    {
                        "family": family,
                        "seed": seed,
                        "lr": lr,
                        "best_epoch": run["best_epoch"],
                        **{f"val_{k}": v for k, v in run["val_metrics"].items() if isinstance(v, (int, float))},
                        **{f"train_{k}": v for k, v in run["train_metrics"].items() if isinstance(v, (int, float))},
                    }
                )
    best = max(runs, key=lambda r: safe_float(r["val_metrics"].get("task_macro_final_reward"), -999.0)) if runs else None
    baseline_payload = load_json(BASELINES_JSON, {}) or {}
    val_baselines = aggregate_rows([r for r in baseline_payload.get("rows") or [] if r.get("split") == "val"])
    majority_val = safe_float(val_baselines.get("majority_rank_aggregation", {}).get("task_macro_final_reward"), -999.0)
    fixed_val = safe_float(val_baselines.get("fixed_composite_top1", {}).get("task_macro_final_reward"), -999.0)
    best_val = safe_float((best or {}).get("val_metrics", {}).get("task_macro_final_reward"), -999.0)
    if not best:
        verdict = "INSUFFICIENT"
    elif best_val > max(majority_val, fixed_val) + 1e-6:
        verdict = "READY"
    elif best_val > fixed_val + 1e-6 or best_val > majority_val + 1e-6:
        verdict = "WEAK"
    elif best_val < safe_float((best or {}).get("train_metrics", {}).get("task_macro_final_reward"), 0.0) - 0.20:
        verdict = "OVERFIT"
    else:
        verdict = "NO_LEARNING"
    artifact = {
        "BG_FINAL_ARBITER_TRAINING_VERDICT": verdict,
        "verdict": verdict,
        "selected_model": {k: v for k, v in (best or {}).items() if k not in {"train_log", "val_rows"}},
        "all_runs": [{k: v for k, v in r.items() if k not in {"state_dict", "train_log", "val_rows"}} for r in runs],
        "training_rows": rows,
        "feature_names": payload["feature_names"],
        "calibration": payload["calibration"],
        "baseline_validation": {"majority_rank_aggregation": majority_val, "fixed_composite_top1": fixed_val},
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(artifact, MODEL_PT)
    write_json(TRAINING_JSON, {k: v for k, v in artifact.items() if k not in {"calibration"}})
    lines = ["# Final Arbiter Training", "", f"BG_FINAL_ARBITER_TRAINING_VERDICT = {verdict}", "", f"- selected family: `{(best or {}).get('family')}`", f"- selected val task macro reward: `{best_val:.4f}`", f"- majority val task macro reward: `{majority_val:.4f}`", f"- fixed val task macro reward: `{fixed_val:.4f}`", "", "## Runs", ""]
    lines.extend(md_table(rows, ["family", "seed", "lr", "best_epoch", "val_task_macro_final_reward", "train_task_macro_final_reward", "val_regret"]))
    write_md(TRAINING_MD, lines)
    print(f"BG_FINAL_ARBITER_TRAINING_VERDICT = {verdict}", flush=True)
    return 0


def load_model_artifact() -> dict[str, Any]:
    if not MODEL_PT.exists():
        run_training()
    return torch.load(MODEL_PT, map_location="cpu", weights_only=False)


def instantiate_selected_model(artifact: dict[str, Any]) -> nn.Module | None:
    selected = artifact.get("selected_model") or {}
    payload = load_features_payload()
    dim = len(payload["feature_names"])
    cls = selected.get("model_class")
    if cls == "DomainGateScorer":
        model: nn.Module = DomainGateScorer(payload["feature_names"])
    elif cls == "TinyMLPScorer":
        model = TinyMLPScorer(dim)
    elif cls == "LinearScorer":
        model = LinearScorer(dim)
    else:
        return None
    state = selected.get("state_dict")
    if state:
        model.load_state_dict(state)
    model.eval()
    return model


def evaluate_trained_on_split(split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = load_features_payload()
    artifact = load_model_artifact()
    sets = [s for s in payload["sets"] if s.get("split") == split]
    rows = []
    metrics = {}
    baseline_rows, baseline_metrics = evaluate_baseline_rows(split)
    rows.extend(baseline_rows)
    metrics.update(baseline_metrics)
    model = instantiate_selected_model(artifact)
    if model is not None:
        policy = f"trained_{(artifact.get('selected_model') or {}).get('family', 'arbiter')}"
        m, r = evaluate_model(model, sets, payload["x"], policy)
        rows.extend(r)
        metrics[policy] = m
    return rows, metrics


def run_heldout_eval() -> int:
    ensure_root()
    started = time.time()
    if not MODEL_PT.exists():
        run_training()
    rows, metrics = evaluate_trained_on_split("heldout")
    artifact = load_model_artifact()
    trained_policy = f"trained_{(artifact.get('selected_model') or {}).get('family', 'arbiter')}"
    trained = metrics.get(trained_policy, {})
    majority = metrics.get("majority_rank_aggregation", {})
    fixed = metrics.get("fixed_composite_top1", {})
    oracle = metrics.get("oracle_best_survivor", {})
    trained_task = safe_float(trained.get("task_macro_final_reward"), -999.0)
    majority_task = safe_float(majority.get("task_macro_final_reward"), -999.0)
    fixed_task = safe_float(fixed.get("task_macro_final_reward"), -999.0)
    oracle_task = safe_float(oracle.get("task_macro_final_reward"), -999.0)
    gap = max(oracle_task - majority_task, 1e-9)
    closure = (trained_task - majority_task) / gap
    by_domain = aggregate_rows(rows, ("policy", "domain"))
    coding_ok = True
    math_ok = True
    for domain, flag_name in (("coding", "coding_ok"), ("math_simple_arithmetic", "math_ok")):
        trained_domain = by_domain.get(f"{trained_policy}::{domain}", {})
        fixed_domain = by_domain.get(f"fixed_composite_top1::{domain}", {})
        ok = safe_float(trained_domain.get("task_macro_final_reward"), 0.0) >= safe_float(fixed_domain.get("task_macro_final_reward"), 0.0) - 1e-6
        if flag_name == "coding_ok":
            coding_ok = ok
        else:
            math_ok = ok
    if trained_task >= 0.75 and trained_task > majority_task + 1e-9 and trained_task > fixed_task + 1e-9 and coding_ok and math_ok:
        verdict = "FINAL_ARBITER_READY"
    elif trained_task > majority_task + 1e-9 or trained_task > fixed_task + 1e-9:
        verdict = "FINAL_ARBITER_WEAK"
    elif not coding_ok:
        verdict = "CODING_DEGRADES"
    elif rows:
        verdict = "NO_IMPROVEMENT"
    else:
        verdict = "INSUFFICIENT"
    payload = {
        "BG_FINAL_ARBITER_HELDOUT_EVAL_VERDICT": verdict,
        "verdict": verdict,
        "trained_policy": trained_policy,
        "rows": rows,
        "metrics_by_policy": metrics,
        "metrics_by_policy_domain": by_domain,
        "gap_closure_from_majority_to_oracle": closure,
        "success_checks": {
            "task_macro_final_reward_ge_0_75": trained_task >= 0.75,
            "improves_majority": trained_task > majority_task + 1e-9,
            "improves_fixed": trained_task > fixed_task + 1e-9,
            "coding_not_degraded": coding_ok,
            "math_not_degraded": math_ok,
            "gap_closure_ge_35pct": closure >= 0.35,
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(HELDOUT_JSON_OUT, payload)
    write_csv(HELDOUT_CSV, rows)
    lines = ["# Heldout Final Arbiter Evaluation", "", f"BG_FINAL_ARBITER_HELDOUT_EVAL_VERDICT = {verdict}", "", f"- trained policy: `{trained_policy}`", f"- gap closure: `{closure:.3f}`", "", "## Metrics", ""]
    table = [{"policy": key, **value} for key, value in sorted(metrics.items())]
    lines.extend(md_table(table, ["policy", "n", "final_selected_reward", "task_macro_final_reward", "top1_success", "regret"]))
    write_md(HELDOUT_MD, lines)
    print(f"BG_FINAL_ARBITER_HELDOUT_EVAL_VERDICT = {verdict}", flush=True)
    return 0


def run_domain_analysis() -> int:
    ensure_root()
    started = time.time()
    if not HELDOUT_JSON_OUT.exists():
        run_heldout_eval()
    heldout = load_json(HELDOUT_JSON_OUT, {}) or {}
    trained_policy = heldout.get("trained_policy", "trained_arbiter")
    rows = heldout.get("rows") or []
    by_domain = aggregate_rows(rows, ("policy", "domain"))
    domain_rows = []
    for domain in DOMAINS:
        trained = by_domain.get(f"{trained_policy}::{domain}", {})
        fixed = by_domain.get(f"fixed_composite_top1::{domain}", {})
        majority = by_domain.get(f"majority_rank_aggregation::{domain}", {})
        oracle = by_domain.get(f"oracle_best_survivor::{domain}", {})
        domain_rows.append(
            {
                "domain": domain,
                "trained_task_macro_reward": trained.get("task_macro_final_reward"),
                "fixed_task_macro_reward": fixed.get("task_macro_final_reward"),
                "majority_task_macro_reward": majority.get("task_macro_final_reward"),
                "oracle_task_macro_reward": oracle.get("task_macro_final_reward"),
                "trained_regret": trained.get("regret"),
                "trained_top1_success": trained.get("top1_success"),
                "survivor_sets": trained.get("n"),
            }
        )
    coding = next((r for r in domain_rows if r["domain"] == "coding"), {})
    science = next((r for r in domain_rows if r["domain"] == "science"), {})
    if safe_float(coding.get("trained_task_macro_reward"), 0.0) < safe_float(coding.get("fixed_task_macro_reward"), 0.0):
        verdict = "CODING_DEGRADES"
    elif safe_float(science.get("trained_task_macro_reward"), 0.0) < safe_float(science.get("majority_task_macro_reward"), 0.0):
        verdict = "SCIENCE_REMAINS_WEAK"
    elif all(safe_float(r.get("trained_task_macro_reward"), -1.0) >= safe_float(r.get("fixed_task_macro_reward"), -2.0) for r in domain_rows if r.get("survivor_sets")):
        verdict = "MULTIDOMAIN_READY"
    elif safe_float(coding.get("trained_task_macro_reward"), 0.0) >= safe_float(coding.get("fixed_task_macro_reward"), 0.0):
        verdict = "CODING_PRESERVED"
    else:
        verdict = "DOMAIN_SPECIALIZATION_NEEDED"
    payload = {
        "BG_FINAL_ARBITER_DOMAIN_VERDICT": verdict,
        "verdict": verdict,
        "domain_rows": domain_rows,
        "trained_policy": trained_policy,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(DOMAIN_JSON, payload)
    lines = ["# Final Arbiter Domain Analysis", "", f"BG_FINAL_ARBITER_DOMAIN_VERDICT = {verdict}", "", "## Domains", ""]
    lines.extend(md_table(domain_rows, ["domain", "survivor_sets", "trained_task_macro_reward", "fixed_task_macro_reward", "majority_task_macro_reward", "oracle_task_macro_reward", "trained_top1_success"]))
    write_md(DOMAIN_MD, lines)
    print(f"BG_FINAL_ARBITER_DOMAIN_VERDICT = {verdict}", flush=True)
    return 0


def feature_groups(feature_names: Sequence[str]) -> dict[str, set[int]]:
    groups = {
        "fixed_composite": {"fixed_composite_score", "rank_fixed_composite", "is_top1_fixed_composite", "is_top2_fixed_composite"},
        "old_content": {name for name in feature_names if "old" in name},
        "old_code_objective": {name for name in feature_names if "code" in name or "objective" in name},
        "hidden_branch": {name for name in feature_names if "branch" in name or "hidden" in name},
        "bridge": {name for name in feature_names if "bridge" in name},
        "universal": {name for name in feature_names if "universal" in name},
        "learned_gated": {name for name in feature_names if "gated" in name},
        "rank_features": {name for name in feature_names if name.startswith("rank_") or name.startswith("is_top")},
        "metadata": {name for name in feature_names if name.startswith("domain_") or name.startswith("layer_") or name.startswith("origin_") or name.endswith("_flag")},
        "stability": {"parse_success", "repetition_rate", "empty_output", "hit_max_tokens", "output_length_log"},
        "domain_features": {name for name in feature_names if name.startswith("domain_")},
        "scores_only": {name for name in feature_names if name.endswith("_score")},
        "ranks_only": {name for name in feature_names if name.startswith("rank_") or name.startswith("is_top")},
        "metadata_only": {name for name in feature_names if name.startswith("domain_") or name.startswith("layer_") or name.startswith("origin_") or name.endswith("_flag")},
    }
    return {key: {feature_names.index(name) for name in vals if name in feature_names} for key, vals in groups.items()}


def evaluate_model_with_mask(mask_mode: str, remove_group: str | None = None, keep_group: str | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = load_features_payload()
    artifact = load_model_artifact()
    model = instantiate_selected_model(artifact)
    if model is None:
        return {}, []
    x = payload["x"].clone()
    groups = feature_groups(payload["feature_names"])
    if mask_mode == "remove" and remove_group:
        for idx in groups.get(remove_group, set()):
            x[:, idx] = 0.0
    if mask_mode == "keep" and keep_group:
        keep = groups.get(keep_group, set())
        for idx in range(x.shape[1]):
            if idx not in keep:
                x[:, idx] = 0.0
    sets = [s for s in payload["sets"] if s.get("split") == "heldout"]
    policy = f"selected_model_{mask_mode}_{remove_group or keep_group or 'all'}"
    return evaluate_model(model, sets, x, policy)


def run_expert_ablation() -> int:
    ensure_root()
    started = time.time()
    if not HELDOUT_JSON_OUT.exists():
        run_heldout_eval()
    rows = []
    base_metrics, base_rows = evaluate_model_with_mask("all")
    rows.append({"ablation": "all_features", **base_metrics})
    rows.extend({**r, "ablation": "all_features"} for r in base_rows)
    ablation_summary = [{"ablation": "all_features", **base_metrics}]
    for group in ["fixed_composite", "old_content", "old_code_objective", "hidden_branch", "bridge", "universal", "learned_gated", "rank_features", "metadata", "stability", "domain_features"]:
        metrics, detail_rows = evaluate_model_with_mask("remove", remove_group=group)
        ablation_summary.append({"ablation": f"no_{group}", **metrics})
        rows.extend({**r, "ablation": f"no_{group}"} for r in detail_rows)
    for group in ["scores_only", "ranks_only", "metadata_only"]:
        metrics, detail_rows = evaluate_model_with_mask("keep", keep_group=group)
        ablation_summary.append({"ablation": group, **metrics})
        rows.extend({**r, "ablation": group} for r in detail_rows)
    base_reward = safe_float(base_metrics.get("task_macro_final_reward"), 0.0)
    drops = {r["ablation"]: base_reward - safe_float(r.get("task_macro_final_reward"), 0.0) for r in ablation_summary if r["ablation"] != "all_features"}
    largest = max(drops, key=drops.get) if drops else ""
    if largest.startswith("no_old"):
        verdict = "OLD_CONTENT_DOMINATES"
    elif largest == "no_bridge":
        verdict = "BRIDGE_MATTERS"
    elif largest == "no_hidden_branch":
        verdict = "HIDDEN_BRANCH_MATTERS"
    elif largest == "no_domain_features":
        verdict = "DOMAIN_GATE_NEEDED"
    elif safe_float(next((r for r in ablation_summary if r["ablation"] == "ranks_only"), {}).get("task_macro_final_reward"), 0.0) >= base_reward - 0.01:
        verdict = "SIMPLE_RANK_AGGREGATION_SUFFICIENT"
    elif drops:
        verdict = "COMPLEMENTARY_EXPERTS"
    else:
        verdict = "INCONCLUSIVE"
    payload = {
        "BG_FINAL_ARBITER_EXPERT_ABLATION_VERDICT": verdict,
        "verdict": verdict,
        "ablation_summary": ablation_summary,
        "largest_drop": largest,
        "rows": rows,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(EXPERT_ABLATION_JSON, payload)
    write_csv(EXPERT_ABLATION_CSV, rows)
    lines = ["# Expert Ablation", "", f"BG_FINAL_ARBITER_EXPERT_ABLATION_VERDICT = {verdict}", "", f"- largest drop: `{largest}`", "", "## Ablations", ""]
    lines.extend(md_table(ablation_summary, ["ablation", "n", "task_macro_final_reward", "final_selected_reward", "top1_success", "regret"]))
    write_md(EXPERT_ABLATION_MD, lines)
    print(f"BG_FINAL_ARBITER_EXPERT_ABLATION_VERDICT = {verdict}", flush=True)
    return 0


def run_calibration_ood() -> int:
    ensure_root()
    started = time.time()
    if not HELDOUT_JSON_OUT.exists():
        run_heldout_eval()
    stress_groups = {
        "remove_old_content_score": "old_content",
        "remove_old_code_objective_score": "old_code_objective",
        "remove_bridge_score": "bridge",
        "remove_hidden_branch_score": "hidden_branch",
        "remove_universal_score": "universal",
        "remove_learned_gated_score": "learned_gated",
        "remove_multiple_critical_experts": "fixed_composite",
    }
    nominal, _ = evaluate_model_with_mask("all")
    rows = [{"stress": "nominal", **nominal}]
    for stress, group in stress_groups.items():
        metrics, _ = evaluate_model_with_mask("remove", remove_group=group)
        rows.append({"stress": stress, **metrics})
    # Domain stress is evaluation slices, not retraining.
    heldout = load_json(HELDOUT_JSON_OUT, {}) or {}
    trained_policy = heldout.get("trained_policy")
    for domain in DOMAINS:
        vals = [r for r in heldout.get("rows") or [] if r.get("policy") == trained_policy and r.get("domain") == domain]
        metric = aggregate_rows(vals).get(trained_policy, {})
        rows.append({"stress": f"{domain}_only", **metric})
    nominal_reward = safe_float(nominal.get("task_macro_final_reward"), 0.0)
    worst_degradation = max([nominal_reward - safe_float(row.get("task_macro_final_reward"), nominal_reward) for row in rows[1:]] or [0.0])
    if worst_degradation <= 0.05:
        verdict = "ROBUST"
    elif worst_degradation <= 0.12:
        verdict = "CONSERVATIVE_BUT_SAFE"
    elif any(row["stress"].startswith("remove") and nominal_reward - safe_float(row.get("task_macro_final_reward"), nominal_reward) > 0.12 for row in rows):
        verdict = "MISSING_EXPERT_FRAGILE"
    else:
        verdict = "CALIBRATION_WEAK"
    payload = {
        "BG_FINAL_ARBITER_CALIBRATION_OOD_VERDICT": verdict,
        "verdict": verdict,
        "stress_rows": rows,
        "worst_degradation": worst_degradation,
        "fallback_policy": "If critical expert missing, use conservative majority/rank aggregation; preserve old/content/code when available.",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(CALIBRATION_JSON, payload)
    lines = ["# Calibration/OOD", "", f"BG_FINAL_ARBITER_CALIBRATION_OOD_VERDICT = {verdict}", "", f"- worst degradation: `{worst_degradation:.4f}`", "", "## Stress", ""]
    lines.extend(md_table(rows, ["stress", "n", "task_macro_final_reward", "final_selected_reward", "top1_success", "regret"]))
    write_md(CALIBRATION_MD, lines)
    print(f"BG_FINAL_ARBITER_CALIBRATION_OOD_VERDICT = {verdict}", flush=True)
    return 0


def run_failure_analysis() -> int:
    ensure_root()
    started = time.time()
    if not HELDOUT_JSON_OUT.exists():
        run_heldout_eval()
    heldout = load_json(HELDOUT_JSON_OUT, {}) or {}
    trained_policy = heldout.get("trained_policy")
    rows = [r for r in heldout.get("rows") or [] if r.get("policy") == trained_policy]
    failure_rows = []
    summary = Counter()
    raw_sets = {s.get("survivor_set_id"): s for s in load_features_payload().get("raw_sets", [])}
    for row in rows:
        if safe_float(row.get("top1_success"), 0.0) >= 1.0:
            continue
        s = raw_sets.get(row.get("survivor_set_id"), {})
        candidates = s.get("candidates") or []
        selected_id = str(row.get("selected_candidate_id"))
        selected = next((c for c in candidates if str(c.get("candidate_id")) == selected_id), {})
        root = "expert_misrank"
        if s.get("domain") == "science":
            root = "science_weak"
        if any(c.get("branch_origin") == "clean_branch" and safe_float(c.get("final_reward"), 0.0) == safe_float(s.get("best_reward"), 0.0) for c in candidates):
            root = "clean_branch_should_win"
        if int(s.get("tie_count") or 0) > 1:
            root = "tie_ambiguity"
        if any(s.get("missing_expert_mask", {}).values()):
            root = "missing_expert"
        summary[root] += 1
        failure_rows.append(
            {
                "task_id": row.get("task_id"),
                "domain": row.get("domain"),
                "survivor_set_id": row.get("survivor_set_id"),
                "selected_candidate_id": selected_id,
                "oracle_candidate_ids": s.get("best_candidate_ids"),
                "selected_reward": row.get("final_selected_reward"),
                "oracle_reward": s.get("best_reward"),
                "expert_ranks_selected": {k: selected.get(f"rank_{k.removesuffix('_score')}") for k in SCORE_KEYS},
                "expert_scores_selected": {k: selected.get(k) for k in SCORE_KEYS},
                "root_cause_tag": root,
            }
        )
    if summary.get("science_weak", 0) >= max(1, sum(summary.values()) // 3):
        verdict = "SCIENCE_BLOCKER"
    elif summary.get("missing_expert", 0):
        verdict = "EXPERT_SIGNAL_BLOCKER"
    elif failure_rows:
        verdict = "FAILURES_UNDERSTOOD"
    else:
        verdict = "FAILURES_UNDERSTOOD"
    payload = {
        "BG_FINAL_ARBITER_FAILURE_ANALYSIS_VERDICT": verdict,
        "verdict": verdict,
        "failure_summary": dict(summary),
        "failure_cases": failure_rows,
        "top_20_failure_cases": failure_rows[:20],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(FAILURE_JSON, payload)
    write_csv(FAILURE_CSV, failure_rows)
    lines = ["# Final Arbiter Failure Analysis", "", f"BG_FINAL_ARBITER_FAILURE_ANALYSIS_VERDICT = {verdict}", "", f"- summary: `{dict(summary)}`", "", "## Top Failures", ""]
    lines.extend(md_table(failure_rows[:20], ["task_id", "domain", "root_cause_tag", "selected_reward", "oracle_reward", "selected_candidate_id", "oracle_candidate_ids"]))
    write_md(FAILURE_MD, lines)
    print(f"BG_FINAL_ARBITER_FAILURE_ANALYSIS_VERDICT = {verdict}", flush=True)
    return 0


def run_selection_readiness() -> int:
    ensure_root()
    started = time.time()
    if not FAILURE_JSON.exists():
        run_failure_analysis()
    heldout = load_json(HELDOUT_JSON_OUT, {}) or {}
    domain = load_json(DOMAIN_JSON, {}) or {}
    ood = load_json(CALIBRATION_JSON, {}) or {}
    failure = load_json(FAILURE_JSON, {}) or {}
    checks = heldout.get("success_checks") or {}
    if heldout.get("verdict") == "FINAL_ARBITER_READY" and ood.get("verdict") in {"ROBUST", "CONSERVATIVE_BUT_SAFE"}:
        verdict = "READY_FOR_STEERING_COMPARISON"
        blocker = ""
    elif checks.get("improves_majority") or checks.get("improves_fixed"):
        verdict = "FINAL_ARBITER_WEAK_BUT_IMPROVED"
        blocker = "final arbiter improved but did not meet all readiness criteria"
    elif domain.get("verdict") in {"SCIENCE_REMAINS_WEAK", "REASONING_SCIENCE_WEAK"}:
        verdict = "NEEDS_DOMAIN_SPECIALIZATION"
        blocker = "science/domain behavior remains weak"
    elif heldout.get("verdict") in {"NO_IMPROVEMENT", "FINAL_ARBITER_WEAK"}:
        verdict = "NEEDS_MORE_FINAL_ARBITER_WORK"
        blocker = "heldout final arbiter does not clear readiness"
    else:
        verdict = "NOT_READY"
        blocker = "insufficient final arbiter improvement"
    payload = {
        "BG_FINAL_ARBITER_SELECTION_ONLY_READINESS_VERDICT": verdict,
        "verdict": verdict,
        "checks": checks,
        "blocker": blocker,
        "no_steering_tested": True,
        "next_prompt": (
            "Phase 2b selection-only locked baseline vs selection + trained steering corridor"
            if verdict == "READY_FOR_STEERING_COMPARISON"
            else ""
        ),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(READINESS_JSON, payload)
    lines = ["# Selection-Only Readiness Update", "", f"BG_FINAL_ARBITER_SELECTION_ONLY_READINESS_VERDICT = {verdict}", "", f"- blocker: `{blocker or 'none'}`", "- no steering was tested."]
    write_md(READINESS_MD, lines)
    print(f"BG_FINAL_ARBITER_SELECTION_ONLY_READINESS_VERDICT = {verdict}", flush=True)
    return 0


def load_stage_payloads() -> dict[str, dict[str, Any]]:
    return {
        "inventory": load_json(INVENTORY_JSON, {}) or {},
        "dataset": load_json(DATASET_JSON, {}) or {},
        "features": load_json(FEATURES_JSON, {}) or {},
        "baselines": load_json(BASELINES_JSON, {}) or {},
        "training": load_json(TRAINING_JSON, {}) or {},
        "heldout": load_json(HELDOUT_JSON_OUT, {}) or {},
        "domain": load_json(DOMAIN_JSON, {}) or {},
        "ablation": load_json(EXPERT_ABLATION_JSON, {}) or {},
        "ood": load_json(CALIBRATION_JSON, {}) or {},
        "failures": load_json(FAILURE_JSON, {}) or {},
        "readiness": load_json(READINESS_JSON, {}) or {},
    }


def payload_verdict(payload: dict[str, Any], key: str, default: str = "INSUFFICIENT") -> str:
    return str(payload.get(key) or payload.get("verdict") or default)


def final_status(data: dict[str, dict[str, Any]]) -> tuple[str, str]:
    heldout = payload_verdict(data["heldout"], "BG_FINAL_ARBITER_HELDOUT_EVAL_VERDICT")
    readiness = payload_verdict(data["readiness"], "BG_FINAL_ARBITER_SELECTION_ONLY_READINESS_VERDICT")
    domain = payload_verdict(data["domain"], "BG_FINAL_ARBITER_DOMAIN_VERDICT")
    if heldout == "FINAL_ARBITER_READY":
        arbiter_status = "FINAL_ARBITER_READY"
    elif heldout == "FINAL_ARBITER_WEAK":
        arbiter_status = "FINAL_ARBITER_WEAK_BUT_USEFUL"
    elif domain == "SCIENCE_REMAINS_WEAK":
        arbiter_status = "SCIENCE_LIMITED"
    elif domain == "CODING_DEGRADES":
        arbiter_status = "CODING_LIMITED"
    elif domain == "DOMAIN_SPECIALIZATION_NEEDED":
        arbiter_status = "DOMAIN_SPECIALIZATION_NEEDED"
    elif heldout == "NO_IMPROVEMENT":
        arbiter_status = "NO_IMPROVEMENT"
    elif heldout == "DATA_LIMITED":
        arbiter_status = "DATA_LIMITED"
    else:
        arbiter_status = "NOT_READY"
    if readiness == "READY_FOR_STEERING_COMPARISON":
        phase_status = "READY_FOR_PHASE2B_STEERING_COMPARISON"
    elif readiness in {"FINAL_ARBITER_WEAK_BUT_IMPROVED", "NEEDS_MORE_FINAL_ARBITER_WORK"}:
        phase_status = "NEEDS_MORE_FINAL_ARBITER_WORK"
    elif arbiter_status in {"NO_IMPROVEMENT", "NOT_READY"}:
        phase_status = "SURVIVAL_READY_FINAL_ARBITER_WEAK"
    else:
        phase_status = "NOT_READY"
    return arbiter_status, phase_status


def top_lines(data: dict[str, dict[str, Any]], arbiter_status: str, phase_status: str) -> list[str]:
    return [
        f"BG_FINAL_ARBITER_INVENTORY_VERDICT = {payload_verdict(data['inventory'], 'BG_FINAL_ARBITER_INVENTORY_VERDICT')}",
        f"BG_FINAL_ARBITER_DATASET_VERDICT = {payload_verdict(data['dataset'], 'BG_FINAL_ARBITER_DATASET_VERDICT')}",
        f"BG_FINAL_ARBITER_FEATURES_VERDICT = {payload_verdict(data['features'], 'BG_FINAL_ARBITER_FEATURES_VERDICT')}",
        f"BG_FINAL_ARBITER_BASELINES_VERDICT = {payload_verdict(data['baselines'], 'BG_FINAL_ARBITER_BASELINES_VERDICT')}",
        f"BG_FINAL_ARBITER_TRAINING_VERDICT = {payload_verdict(data['training'], 'BG_FINAL_ARBITER_TRAINING_VERDICT')}",
        f"BG_FINAL_ARBITER_HELDOUT_EVAL_VERDICT = {payload_verdict(data['heldout'], 'BG_FINAL_ARBITER_HELDOUT_EVAL_VERDICT')}",
        f"BG_FINAL_ARBITER_DOMAIN_VERDICT = {payload_verdict(data['domain'], 'BG_FINAL_ARBITER_DOMAIN_VERDICT')}",
        f"BG_FINAL_ARBITER_EXPERT_ABLATION_VERDICT = {payload_verdict(data['ablation'], 'BG_FINAL_ARBITER_EXPERT_ABLATION_VERDICT')}",
        f"BG_FINAL_ARBITER_CALIBRATION_OOD_VERDICT = {payload_verdict(data['ood'], 'BG_FINAL_ARBITER_CALIBRATION_OOD_VERDICT')}",
        f"BG_FINAL_ARBITER_FAILURE_ANALYSIS_VERDICT = {payload_verdict(data['failures'], 'BG_FINAL_ARBITER_FAILURE_ANALYSIS_VERDICT')}",
        f"BG_FINAL_ARBITER_SELECTION_ONLY_READINESS_VERDICT = {payload_verdict(data['readiness'], 'BG_FINAL_ARBITER_SELECTION_ONLY_READINESS_VERDICT')}",
        f"FINAL_ARBITER_TOP4_STATUS = {arbiter_status}",
        f"SELECTION_ONLY_PHASE2A_STATUS_AFTER_FINAL_ARBITER = {phase_status}",
    ]


def recommendation(phase_status: str, arbiter_status: str) -> str:
    if phase_status == "READY_FOR_PHASE2B_STEERING_COMPARISON":
        return "Proceed to Phase 2b with selection-only locked baseline vs selection plus trained steering corridor; no other changes."
    if arbiter_status == "FINAL_ARBITER_WEAK_BUT_USEFUL":
        return "Run a small improved-arbiter v1.1 or proceed only with explicit weak-baseline caveat."
    if arbiter_status == "SCIENCE_LIMITED":
        return "Collect or train science-specific final arbiter data."
    if arbiter_status == "DOMAIN_SPECIALIZATION_NEEDED":
        return "Train a domain-gated final arbiter."
    if arbiter_status == "NO_IMPROVEMENT":
        return "Return to expert signals and bridge data; current trained arbiter did not improve heldout final selection."
    return "Continue final-arbiter work before any Phase 2b steering comparison."


def docs_lines(data: dict[str, dict[str, Any]], arbiter_status: str, phase_status: str) -> list[str]:
    rec = recommendation(phase_status, arbiter_status)
    heldout = data.get("heldout", {})
    training = data.get("training", {})
    return [
        "# Final Arbiter Among Top4 Survivors V1",
        "",
        "This experiment addresses the Phase 2a blocker where fixed-composite top4 survival retained good branches, but final selection among survivors was weak. It trains only small standalone final-arbiter models over cached survivor sets.",
        "",
        "## Verdicts",
        "",
        *top_lines(data, arbiter_status, phase_status),
        "",
        "## Selection-Only Prototype Context",
        "",
        "The prior selection-only run ended at `SURVIVAL_READY_FINAL_ARBITER_WEAK`; top4 survival was strong, but the final arbiter lagged the best-survivor upper bound.",
        "",
        "## Dataset",
        "",
        f"- dataset counts: `{(data.get('dataset') or {}).get('counts')}`",
        "- labels are final reward/correctness/verifier results; tap scores are input features only.",
        "- splits are task-disjoint; heldout is not used for model selection.",
        "",
        "## Training",
        "",
        f"- training verdict: `{payload_verdict(training, 'BG_FINAL_ARBITER_TRAINING_VERDICT')}`",
        f"- selected model: `{((training.get('selected_model') or {}).get('family'))}`",
        "",
        "## Heldout Results",
        "",
        f"- heldout verdict: `{payload_verdict(heldout, 'BG_FINAL_ARBITER_HELDOUT_EVAL_VERDICT')}`",
        f"- trained policy: `{heldout.get('trained_policy')}`",
        f"- success checks: `{heldout.get('success_checks')}`",
        "",
        "## Domain, Ablation, OOD, Failures",
        "",
        f"- domain verdict: `{payload_verdict(data['domain'], 'BG_FINAL_ARBITER_DOMAIN_VERDICT')}`",
        f"- expert ablation verdict: `{payload_verdict(data['ablation'], 'BG_FINAL_ARBITER_EXPERT_ABLATION_VERDICT')}`",
        f"- calibration/OOD verdict: `{payload_verdict(data['ood'], 'BG_FINAL_ARBITER_CALIBRATION_OOD_VERDICT')}`",
        f"- failure verdict: `{payload_verdict(data['failures'], 'BG_FINAL_ARBITER_FAILURE_ANALYSIS_VERDICT')}`",
        "",
        "## Recommendation",
        "",
        rec,
        "",
        "No action steering was tested. No production routing changed. No true fork/carry or compute-saving claim is made.",
        "",
    ]


def append_once(path: Path, section: Sequence[str], marker: str) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    path.write_text(text.rstrip() + "\n\n" + "\n".join(section).strip() + "\n", encoding="utf-8")


def update_docs(data: dict[str, dict[str, Any]], arbiter_status: str, phase_status: str) -> None:
    marker = "## Final arbiter among top4 survivors v1 (2026-05-18)"
    section = [
        marker,
        "",
        f"FINAL_ARBITER_TOP4_STATUS = {arbiter_status}",
        f"SELECTION_ONLY_PHASE2A_STATUS_AFTER_FINAL_ARBITER = {phase_status}",
        "",
        f"- heldout eval: `{payload_verdict(data['heldout'], 'BG_FINAL_ARBITER_HELDOUT_EVAL_VERDICT')}`",
        f"- selected model: `{((data.get('training') or {}).get('selected_model') or {}).get('family')}`",
        f"- readiness: `{payload_verdict(data['readiness'], 'BG_FINAL_ARBITER_SELECTION_ONLY_READINESS_VERDICT')}`",
        f"- recommendation: {recommendation(phase_status, arbiter_status)}",
        "- no action steering was tested.",
        "",
    ]
    write_md(DOC_MD, docs_lines(data, arbiter_status, phase_status))
    for target in DOC_TARGETS:
        append_once(target, section, marker)


def run_synthesis() -> int:
    ensure_root()
    started = time.time()
    required = [
        (INVENTORY_JSON, run_inventory),
        (DATASET_PT, run_dataset),
        (FEATURES_PT, run_features),
        (BASELINES_JSON, run_baselines),
        (MODEL_PT, run_training),
        (HELDOUT_JSON_OUT, run_heldout_eval),
        (DOMAIN_JSON, run_domain_analysis),
        (EXPERT_ABLATION_JSON, run_expert_ablation),
        (CALIBRATION_JSON, run_calibration_ood),
        (FAILURE_JSON, run_failure_analysis),
        (READINESS_JSON, run_selection_readiness),
    ]
    for path, fn in required:
        if not path.exists():
            fn()
    data = load_stage_payloads()
    arbiter_status, phase_status = final_status(data)
    rec = recommendation(phase_status, arbiter_status)
    payload = {
        "top_lines": top_lines(data, arbiter_status, phase_status),
        "stage_payloads": data,
        "FINAL_ARBITER_TOP4_STATUS": arbiter_status,
        "SELECTION_ONLY_PHASE2A_STATUS_AFTER_FINAL_ARBITER": phase_status,
        "recommended_next": rec,
        "files_created": {
            "output_root": rel(OUT_ROOT),
            "model": rel(MODEL_PT),
            "summary": rel(SUMMARY_JSON),
            "analysis": rel(ANALYSIS_JSON),
            "doc": rel(DOC_MD),
        },
        "commands_run": [f"venv/bin/python -u utilities/tests/manual/{name}" for name in SCRIPT_NAMES],
        "blockers": [] if phase_status == "READY_FOR_PHASE2B_STEERING_COMPARISON" else [rec],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(SUMMARY_JSON, payload)
    write_json(ANALYSIS_JSON, payload)
    lines = ["# Final Arbiter Top4 Survivors V1 Summary", "", *payload["top_lines"], "", "## Recommendation", "", rec, "", "## Files Created", ""]
    lines.extend(f"- `{path}`" for path in payload["files_created"].values())
    lines.extend(["", "## Commands Run", ""])
    lines.extend(f"- `{cmd}`" for cmd in payload["commands_run"])
    if payload["blockers"]:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {b}" for b in payload["blockers"])
    write_md(SUMMARY_MD, lines)
    write_md(ANALYSIS_MD, docs_lines(data, arbiter_status, phase_status))
    update_docs(data, arbiter_status, phase_status)
    print(f"FINAL_ARBITER_TOP4_STATUS = {arbiter_status}", flush=True)
    print(f"SELECTION_ONLY_PHASE2A_STATUS_AFTER_FINAL_ARBITER = {phase_status}", flush=True)
    return 0
