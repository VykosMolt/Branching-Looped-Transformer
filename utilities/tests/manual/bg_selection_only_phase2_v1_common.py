"""Selection-only Phase 2 prototype v1 helpers.

This experiment is an evaluation layer over cached completed branch outcomes.
It does not train Ouro, mutate checkpoints/tokenizers/tap registries, execute
wrapper/local-agent code, import Hunter-Seeker modules, run ARC loops, or apply
action steering.
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

from bg_branch_generator_v1_common import (
    BGV1_ROOT,
    BRANCH_GENERATOR_PT,
    BRANCHES_PT,
    BEST_SCHEDULE_JSON,
    BASIS_BANK_PT,
    HIDDEN_ORIGIN_TAP_HEADS_GENERATOR_V1_PT,
    load_generator_rows,
)
from bg_fixed_composite_survival_v1_common import (
    BEST_COMPOSITE_JSON,
    BEST_VETO_JSON,
    DEFAULT_WEIGHTS,
    HELDOUT_JSON,
    LAYER_DEFAULTS,
    MISSING_JSON,
    POLICY_PT,
    READINESS_JSON,
    composite_matrix,
    family_matrix,
    load_best_composite,
    load_best_veto,
    load_dataset_records,
    matrix,
    missing_ood_survivors,
    net_from_matrix,
    oracle_indices,
    safe_float,
    selected_metric,
    topk_policy,
)
from bg_gated_selector_v1_common import GATED_ROOT, HEADS_PT as GATED_HEADS_PT
from bg_hidden_origin_quota_v4_common import (
    HEADS_V4_PT,
    PROBE_ROOT,
    PROJECT_ROOT,
    V4_ROOT,
    md_table,
    rate,
    rel,
    write_csv,
    write_md,
)
from bg_universal_tap_v1_common import (
    BRIDGE_PT,
    HIDDEN_BRANCH_PT,
    OLD_CONTENT_PT,
    UNIVERSAL_HEADS_PT,
    domain_bucket,
    load_json,
    load_v4_branch_rows,
)


OUT_ROOT = PROBE_ROOT / "bg_selection_only_phase2_prototype_v1_2026-05-18"

INVENTORY_JSON = OUT_ROOT / "inventory.json"
INVENTORY_MD = OUT_ROOT / "inventory.md"
TASK_SUITE_JSON = OUT_ROOT / "task_suite.json"
TASK_SUITE_MD = OUT_ROOT / "task_suite.md"
TASK_SUITE_CSV = OUT_ROOT / "task_suite.csv"
POLICY_OUTPUTS_PT = OUT_ROOT / "selection_only_policy_outputs.pt"
POLICY_OUTPUTS_JSON = OUT_ROOT / "selection_only_policy_outputs.json"
POLICY_ROWS_CSV = OUT_ROOT / "selection_only_policy_rows.csv"
POLICY_REPORT_MD = OUT_ROOT / "policy_runner_report.md"
POLICY_REPORT_JSON = OUT_ROOT / "policy_runner_report.json"
POLICY_PROGRESS_JSON = OUT_ROOT / "policy_runner_progress.json"
POLICY_PROGRESS_JSONL = OUT_ROOT / "policy_runner_progress.jsonl"
CACHED_MD = OUT_ROOT / "cached_reproduction.md"
CACHED_JSON = OUT_ROOT / "cached_reproduction.json"
CACHED_CSV = OUT_ROOT / "cached_reproduction_rows.csv"
LIVE_MD = OUT_ROOT / "live_prototype.md"
LIVE_JSON = OUT_ROOT / "live_prototype.json"
LIVE_CSV = OUT_ROOT / "live_prototype_rows.csv"
LIVE_SELECTED_CSV = OUT_ROOT / "live_selected_branches.csv"
ARBITER_MD = OUT_ROOT / "final_arbiter_analysis.md"
ARBITER_JSON = OUT_ROOT / "final_arbiter_analysis.json"
ARBITER_CSV = OUT_ROOT / "final_arbiter_rows.csv"
BASELINE_MD = OUT_ROOT / "baseline_comparison.md"
BASELINE_JSON = OUT_ROOT / "baseline_comparison.json"
BASELINE_CSV = OUT_ROOT / "baseline_rows.csv"
DOMAIN_MD = OUT_ROOT / "domain_coding_analysis.md"
DOMAIN_JSON = OUT_ROOT / "domain_coding_analysis.json"
FAILURE_MD = OUT_ROOT / "failure_analysis.md"
FAILURE_JSON = OUT_ROOT / "failure_analysis.json"
FAILURE_CSV = OUT_ROOT / "failure_cases.csv"
STEERING_MD = OUT_ROOT / "steering_readiness.md"
STEERING_JSON = OUT_ROOT / "steering_readiness.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
SUMMARY_JSON = OUT_ROOT / "summary.json"
ANALYSIS_MD = OUT_ROOT / "analysis.md"
ANALYSIS_JSON = OUT_ROOT / "analysis.json"
DOC_MD = PROJECT_ROOT / "docs/evaluator/bg_selection_only_phase2_prototype_v1.md"

TASK_SOURCE_PATHS = [
    PROBE_ROOT / "bg_trajectory_prediction_2026-05-18/task_suite.json",
    PROBE_ROOT / "bg_steering_suite_2026-05-18/task_suite.json",
    PROBE_ROOT / "bg_stage2_steering_2026-05-18/task_suite.json",
    PROBE_ROOT / "bg_hidden_state_branch_generation_2026-05-18/task_subset.json",
    PROBE_ROOT / "code_strict_clean_screening_taskpool_2026-05-17.json",
    PROBE_ROOT / "code_branch_taskset_v2_2026-05-16.json",
    PROBE_ROOT / "code_branch_taskset_v2_near_miss10_2026-05-17.json",
]

OLD_HEADS_PT = PROBE_ROOT / "bg_hidden_origin_taps_2026-05-18/hidden_origin_tap_heads.pt"

DOC_TARGETS = [
    PROJECT_ROOT / "docs/evaluator/current_state.md",
    PROJECT_ROOT / "docs/evaluator/domain_transfer_ledger.md",
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

SCRIPT_NAMES = [
    "bg_selection_only_phase2_inventory_v1.py",
    "build_bg_selection_only_task_suite_v1.py",
    "run_bg_selection_only_policy_v1.py",
    "reproduce_bg_selection_only_cached_v1.py",
    "run_bg_selection_only_live_prototype_v1.py",
    "analyze_bg_selection_only_final_arbiter_v1.py",
    "analyze_bg_selection_only_baselines_v1.py",
    "analyze_bg_selection_only_domain_coding_v1.py",
    "analyze_bg_selection_only_failures_v1.py",
    "analyze_bg_selection_only_to_steering_readiness_v1.py",
    "analyze_bg_selection_only_phase2_prototype_v1.py",
]

POLICY_NAME = "fixed_composite_conservative_top4"
EXPECTED_HELDOUT = {
    "oracle_retention": 0.930635838150289,
    "false_prune_rate": 0.06936416184971098,
    "average_survivors": 3.8728323699421967,
}


def ensure_out_root() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "mean": float(value.detach().float().mean().item()) if value.numel() else 0.0,
        }
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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=json_default) + "\n")


def finite_mean(values: Iterable[Any], default: float = float("nan")) -> float:
    vals = []
    for value in values:
        x = safe_float(value)
        if math.isfinite(x):
            vals.append(x)
    return float(mean(vals)) if vals else default


def stable_hash(value: str) -> int:
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:12], 16)


def split_source_status(split: str) -> str:
    if split == "heldout":
        return "reused_heldout"
    return "reused_diagnostic"


def normalize_domain(value: Any) -> str:
    d = domain_bucket(value)
    if d == "gsm8k":
        return "math_simple_arithmetic"
    return d


def aggregate_numeric(rows: Sequence[dict[str, Any]], keys: Sequence[str] = ("policy",)) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        label = "::".join(str(row.get(k)) for k in keys)
        grouped[label].append(row)
    out: dict[str, Any] = {}
    for label, vals in grouped.items():
        nums: dict[str, list[float]] = defaultdict(list)
        for row in vals:
            for key, value in row.items():
                x = safe_float(value)
                if math.isfinite(x):
                    nums[key].append(x)
        out[label] = {"n": len(vals), **{key: float(mean(v)) for key, v in nums.items() if v}}
    return out


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "n",
        "oracle_retention",
        "false_prune_rate",
        "average_survivors",
        "top4_oracle_coverage",
        "best_selected_reward",
        "mean_selected_reward",
        "final_selected_reward",
        "final_selected_correctness",
        "clean_reward",
        "regret",
        "parse_success_rate",
        "stable_rate",
        "task_macro_reward",
    ]
    return {key: metrics.get(key) for key in keys if key in metrics}


def candidate_rewards(record: dict[str, Any]) -> list[float]:
    return [safe_float(c.get("reward"), 0.0) for c in record.get("candidates") or []]


def candidate_correct(record: dict[str, Any], idx: int) -> float:
    candidates = record.get("candidates") or []
    if not (0 <= idx < len(candidates)):
        return 0.0
    if candidates[idx].get("correct") is not None:
        return 1.0 if bool(candidates[idx].get("correct")) else 0.0
    return 1.0 if safe_float(candidates[idx].get("reward"), 0.0) > 0.0 else 0.0


def clean_index(record: dict[str, Any]) -> int:
    for idx, candidate in enumerate(record.get("candidates") or []):
        if candidate.get("origin") in {"clean", "clean_branch"}:
            return idx
    return 0


def deterministic_random_indices(record: dict[str, Any], k: int) -> list[int]:
    n = int(record.get("candidate_count") or len(record.get("candidates") or []))
    order = list(range(n))
    seed = stable_hash(str(record.get("candidate_set_id")))
    order.sort(key=lambda idx: stable_hash(f"{seed}:{idx}"))
    return order[: min(k, n)]


def score_values(record: dict[str, Any], family: str, weights: dict[str, float] | None = None) -> list[float]:
    if family == "fixed_composite":
        return net_from_matrix(composite_matrix(record, weights or load_best_composite()))
    if family in {"old", "code", "branch", "bridge", "universal", "gated"}:
        return net_from_matrix(family_matrix(record, family))
    return net_from_matrix(matrix(record, family))


def top1_among(indices: Sequence[int], scores: Sequence[float]) -> int | None:
    usable = [int(i) for i in indices if 0 <= int(i) < len(scores)]
    if not usable:
        return None
    return sorted(usable, key=lambda i: (-safe_float(scores[i], 0.0), i))[0]


def majority_rank_top1(record: dict[str, Any], indices: Sequence[int], weights: dict[str, float]) -> int | None:
    families = ["fixed_composite", "old", "branch", "bridge", "universal", "gated"]
    rank_sum = {int(i): 0.0 for i in indices}
    for family in families:
        scores = score_values(record, family, weights)
        order = sorted([int(i) for i in indices if 0 <= int(i) < len(scores)], key=lambda i: (-safe_float(scores[i], 0.0), i))
        for rank, idx in enumerate(order):
            rank_sum[idx] += rank
    if not rank_sum:
        return None
    return sorted(rank_sum, key=lambda i: (rank_sum[i], i))[0]


def selection_for_policy(record: dict[str, Any], policy: str, weights: dict[str, float], params: dict[str, Any]) -> list[int]:
    if policy == "clean_only":
        return [clean_index(record)]
    if policy == "oracle":
        return sorted(oracle_indices(record))
    if policy == "random_top1":
        return deterministic_random_indices(record, 1)
    if policy == "random_top2":
        return deterministic_random_indices(record, 2)
    if policy == "random_top4":
        return deterministic_random_indices(record, 4)
    if policy == "fixed_composite_top1":
        return topk_policy(record, "fixed_composite", 1, weights)
    if policy == "fixed_composite_top2":
        return topk_policy(record, "fixed_composite", 2, weights)
    if policy in {"fixed_composite_top3", "previous_veto_rescue", "fixed_composite_top4_without_rescue", "fixed_composite_top4_without_missing_ood_fallback"}:
        if policy == "fixed_composite_top3":
            return topk_policy(record, "fixed_composite", 3, weights)
        if policy == "previous_veto_rescue":
            # Diagnostic only; the selected Phase 2a operating point remains fixed top4.
            from bg_fixed_composite_survival_v1_common import apply_veto_rescue

            return apply_veto_rescue(record, weights, params)
        return topk_policy(record, "fixed_composite", 4, weights)
    if policy == POLICY_NAME:
        return topk_policy(record, "fixed_composite", 4, weights)
    families = {
        "old_frozen_bg_top1": ("old", 1),
        "old_frozen_bg_top4": ("old", 4),
        "old_code_objective_top1": ("code", 1),
        "old_code_objective_top4": ("code", 4),
        "hidden_origin_v4_top1": ("v4_hidden_origin", 1),
        "hidden_origin_v4_top4": ("v4_hidden_origin", 4),
        "learned_gated_top1": ("gated", 1),
        "learned_gated_top4": ("gated", 4),
        "universal_top1": ("universal", 1),
        "universal_top4": ("universal", 4),
        "bridge_top1": ("bridge", 1),
        "bridge_top4": ("bridge", 4),
    }
    if policy in families:
        family, k = families[policy]
        return topk_policy(record, family, k, weights if family == "fixed_composite" else None)
    return topk_policy(record, "fixed_composite", 4, weights)


def evaluation_row(record: dict[str, Any], policy: str, selected: Sequence[int]) -> dict[str, Any]:
    selected = [int(i) for i in selected if 0 <= int(i) < int(record.get("candidate_count") or 0)]
    if not selected:
        selected = [clean_index(record)]
    rewards = candidate_rewards(record)
    selected_rewards = [rewards[i] for i in selected if 0 <= i < len(rewards)]
    best_selected = max(selected_rewards) if selected_rewards else float("-inf")
    oracle_reward = max(rewards) if rewards else 0.0
    oracles = oracle_indices(record)
    retained = bool(oracles & set(selected))
    first = selected[0]
    parse_vals = [1.0 if bool((record.get("candidates") or [])[i].get("parse_success", True)) else 0.0 for i in selected if i < len(record.get("candidates") or [])]
    stable_vals = [
        1.0
        if (
            bool((record.get("candidates") or [])[i].get("parse_success", True))
            and not bool((record.get("candidates") or [])[i].get("empty_output", False))
            and safe_float((record.get("candidates") or [])[i].get("repetition_rate"), 0.0) < 0.75
        )
        else 0.0
        for i in selected
        if i < len(record.get("candidates") or [])
    ]
    clean = clean_index(record)
    return {
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
        "top4_oracle_coverage": 1.0 if retained else 0.0,
        "best_selected_reward": best_selected,
        "mean_selected_reward": finite_mean(selected_rewards, 0.0),
        "final_selected_reward": rewards[first] if first < len(rewards) else 0.0,
        "final_selected_correctness": candidate_correct(record, first),
        "top1_success": 1.0 if first in oracles else 0.0,
        "clean_reward": rewards[clean] if clean < len(rewards) else 0.0,
        "clean_correctness": candidate_correct(record, clean),
        "regret": oracle_reward - best_selected,
        "final_regret": oracle_reward - (rewards[first] if first < len(rewards) else 0.0),
        "oracle_reward": oracle_reward,
        "selected_indices": selected,
        "oracle_indices": sorted(oracles),
        "selected_ids": [(record.get("candidates") or [])[i].get("candidate_id") for i in selected if i < len(record.get("candidates") or [])],
        "parse_success_rate": finite_mean(parse_vals, 1.0),
        "stable_rate": finite_mean(stable_vals, 1.0),
        "compute_saved_proxy": 1.0 - len(selected) / max(int(record.get("candidate_count") or 1), 1),
    }


def add_task_macro(policy_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    micro = aggregate_numeric(policy_rows, ("policy",))
    task_rows = []
    for policy, by_task in group_nested(policy_rows, "policy", "task_id").items():
        for task_id, vals in by_task.items():
            task_rows.append(
                {
                    "policy": policy,
                    "task_id": task_id,
                    "task_macro_reward": finite_mean([row.get("final_selected_reward") for row in vals], 0.0),
                    "task_macro_best_selected_reward": finite_mean([row.get("best_selected_reward") for row in vals], 0.0),
                    "task_macro_oracle_retention": finite_mean([row.get("oracle_retention") for row in vals], 0.0),
                    "task_macro_false_prune_rate": finite_mean([row.get("false_prune_rate") for row in vals], 0.0),
                }
            )
    task_metrics = aggregate_numeric(task_rows, ("policy",))
    out = {}
    for policy, vals in micro.items():
        merged = dict(vals)
        merged.update({k: v for k, v in task_metrics.get(policy, {}).items() if k.startswith("task_macro")})
        out[policy] = merged
    return out


def group_nested(rows: Sequence[dict[str, Any]], outer_key: str, inner_key: str) -> dict[str, dict[str, list[dict[str, Any]]]]:
    out: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        out[str(row.get(outer_key))][str(row.get(inner_key))].append(row)
    return out


def load_policy_payload() -> dict[str, Any]:
    payload = torch.load(POLICY_PT, map_location="cpu", weights_only=False) if POLICY_PT.exists() else {}
    return payload if isinstance(payload, dict) else {}


def artifact_status(path: Path) -> dict[str, Any]:
    out = {"path": rel(path), "exists": path.exists(), "loadable": False, "kind": path.suffix.lstrip(".")}
    if not path.exists():
        return out
    try:
        if path.suffix == ".json":
            value = load_json(path, None)
            out["loadable"] = value is not None
            if isinstance(value, dict):
                out["keys"] = sorted(value.keys())[:20]
        elif path.suffix == ".pt":
            value = torch.load(path, map_location="cpu", weights_only=False)
            out["loadable"] = value is not None
            if isinstance(value, dict):
                out["keys"] = sorted(value.keys())[:20]
        else:
            out["loadable"] = True
    except Exception as exc:
        out["error"] = str(exc)
    return out


def feature_config_inventory(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    configs = Counter()
    score_experts = Counter()
    for record in records:
        for expert, cov in (record.get("expert_coverage") or {}).items():
            if safe_float(cov, 0.0) > 0.0:
                score_experts[expert] += 1
        for candidate in record.get("candidates") or []:
            for key in candidate.keys():
                if key.endswith("_score"):
                    configs[key] += 1
    return {
        "expert_score_coverage_sets": dict(score_experts),
        "candidate_score_fields": dict(configs),
        "feature_layers": ["L24", "L36", "L47"],
        "optional_diagnostic_layers": ["L30", "L42"],
    }


def run_inventory() -> int:
    ensure_out_root()
    started = time.time()
    records = load_dataset_records()
    required = {
        "fixed_composite_branch_survival_policy_v1.pt": POLICY_PT,
        "best_fixed_composite.json": BEST_COMPOSITE_JSON,
        "best_veto_rescue_policy.json": BEST_VETO_JSON,
        "heldout_survival_eval.json": HELDOUT_JSON,
        "selection_only_readiness.json": READINESS_JSON,
        "BGV1 branch_generator_v1.pt": BRANCH_GENERATOR_PT,
        "BGV1 branches.pt": BRANCHES_PT,
        "BGV1 basis_bank_v1.pt": BASIS_BANK_PT,
        "BGV1 best_branch_generator_schedule.json": BEST_SCHEDULE_JSON,
        "v4 hidden-origin tap heads": HEADS_V4_PT,
        "old frozen BG/objective/code taps": OLD_HEADS_PT,
        "bridge-only dataset/head source": BRIDGE_PT,
        "universal head": UNIVERSAL_HEADS_PT,
        "learned gated selector": GATED_HEADS_PT,
        "hidden branch head dataset": HIDDEN_BRANCH_PT,
        "old content head dataset": OLD_CONTENT_PT,
        "missing/OOD policy": MISSING_JSON,
        "hidden_origin_tap_heads_generator_v1.pt": HIDDEN_ORIGIN_TAP_HEADS_GENERATOR_V1_PT,
    }
    artifact_rows = [{"artifact": name, **artifact_status(path)} for name, path in required.items()]
    missing = [row["artifact"] for row in artifact_rows if not row["exists"]]
    unloadable = [row["artifact"] for row in artifact_rows if row["exists"] and not row["loadable"]]
    readiness = load_json(READINESS_JSON, {}) or {}
    heldout = load_json(HELDOUT_JSON, {}) or {}
    best_composite = load_json(BEST_COMPOSITE_JSON, {}) or {}
    best_veto = load_json(BEST_VETO_JSON, {}) or {}
    schedule = load_json(BEST_SCHEDULE_JSON, {}) or {}
    domains = dict(Counter(str(r.get("domain")) for r in records))
    splits = dict(Counter(str(r.get("split")) for r in records))
    task_pools = {
        "survival_dataset_tasks": len({str(r.get("task_id")) for r in records}),
        "heldout_tasks": len({str(r.get("task_id")) for r in records if r.get("split") == "heldout"}),
        "task_source_files": [rel(path) for path in TASK_SOURCE_PATHS if path.exists()],
    }
    verdict = "READY"
    if not POLICY_PT.exists():
        verdict = "POLICY_MISSING"
    elif missing:
        verdict = "PARTIAL"
    elif unloadable:
        verdict = "FEATURE_MISMATCH"
    if readiness.get("verdict") not in {"READY", "CONSERVATIVE_READY"}:
        verdict = "BLOCKED" if verdict == "READY" else verdict
    payload = {
        "BG_SELECTION_ONLY_INVENTORY_VERDICT": verdict,
        "verdict": verdict,
        "policy_loaded": POLICY_PT.exists() and artifact_status(POLICY_PT).get("loadable", False),
        "selected_operating_point": POLICY_NAME,
        "composite_weights": best_composite.get("weights") or DEFAULT_WEIGHTS,
        "top4_policy_parameters": {"k": 4, "primary_policy": POLICY_NAME, "no_hard_top1_top2_top3": True},
        "veto_rescue_parameters": best_veto,
        "missing_ood_fallback_parameters": load_json(MISSING_JSON, {}) or {},
        "available_branch_generator_methods": {
            "best_schedule": schedule,
            "available_methods": ["hs-inspired controller pattern", "hook_intervention_per_branch"],
            "true_fork_carry": "diagnostic_only_not_claimed",
        },
        "available_feature_configs": feature_config_inventory(records),
        "available_domains": domains,
        "available_splits": splits,
        "available_task_pools": task_pools,
        "old_context_coding_compatibility": {
            "old_context_retention": heldout.get("old_context_retention"),
            "coding_retention": heldout.get("coding_retention"),
            "old_code_preservation": (load_json(PROBE_ROOT / "bg_fixed_composite_branch_survival_policy_v1_2026-05-18/old_code_preservation.json", {}) or {}).get("verdict"),
        },
        "known_limitations": [
            "This Phase 2a implementation uses cached completed branch outcomes when available.",
            "No true branch-batch fork/carry is claimed.",
            "No action steering or trained steering corridor is tested.",
            "Compute savings are not claimed because counterfactual mode continues all candidates in the source artifacts.",
        ],
        "artifacts": artifact_rows,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(INVENTORY_JSON, payload)
    lines = [
        "# Selection-Only Phase 2 Prototype Inventory",
        "",
        f"BG_SELECTION_ONLY_INVENTORY_VERDICT = {verdict}",
        "",
        f"- policy loaded: `{payload['policy_loaded']}`",
        f"- selected operating point: `{POLICY_NAME}`",
        f"- composite weights: `{payload['composite_weights']}`",
        f"- domains: `{domains}`",
        f"- task pools: `{task_pools}`",
        "",
        "## Artifacts",
        "",
    ]
    lines.extend(md_table(artifact_rows, ["artifact", "exists", "loadable", "path"]))
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in payload["known_limitations"])
    write_md(INVENTORY_MD, lines)
    print(f"BG_SELECTION_ONLY_INVENTORY_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


def collect_task_dicts(value: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if value.get("task_id") and (value.get("prompt") or value.get("question") or value.get("source")):
            out.append(value)
        for child in value.values():
            out.extend(collect_task_dicts(child))
    elif isinstance(value, list):
        for child in value:
            out.extend(collect_task_dicts(child))
    return out


def load_task_catalog() -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for path in TASK_SOURCE_PATHS:
        if not path.exists():
            continue
        payload = load_json(path, None)
        for task in collect_task_dicts(payload):
            tid = str(task.get("task_id"))
            if not tid or tid in catalog:
                continue
            catalog[tid] = {
                "task_id": tid,
                "domain": normalize_domain(task.get("domain") or task.get("source")),
                "source_dataset": task.get("source_dataset") or task.get("source") or "",
                "prompt": task.get("prompt") or task.get("question") or "",
                "question": task.get("question") or "",
                "options": task.get("options") or {},
                "answer": task.get("answer_key") or task.get("gold_answer") or "",
                "parser_verifier_type": task.get("evaluator_type") or ("python_tests" if task.get("tests") else "cached_reward_label"),
            }
    return catalog


def record_task_summary(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[str(record.get("task_id"))].append(record)
    out = {}
    for task_id, vals in by_task.items():
        first = vals[0]
        rewards = []
        for record in vals:
            rewards.extend(candidate_rewards(record))
        non_tie_sets = 0
        for record in vals:
            r = candidate_rewards(record)
            if len(set(r)) > 1:
                non_tie_sets += 1
        splits = Counter(str(v.get("split")) for v in vals)
        split = "heldout" if splits.get("heldout") else ("val" if splits.get("val") else sorted(splits)[0])
        split_vals = [v for v in vals if v.get("split") == split]
        out[task_id] = {
            "task_id": task_id,
            "domain": str(first.get("domain")),
            "split": split,
            "source_dataset": "",
            "candidate_set_ids": [str(v.get("candidate_set_id")) for v in split_vals],
            "branch_group_ids": sorted({str(v.get("group_id")) for v in split_vals}),
            "candidate_set_count": len(split_vals),
            "non_tie_candidate_sets": non_tie_sets,
            "clean_not_trivially_always_correct": any(candidate_rewards(v)[clean_index(v)] < max(candidate_rewards(v) or [0.0]) for v in split_vals),
            "parse_rate": finite_mean(
                [
                    1.0 if bool(c.get("parse_success", True)) else 0.0
                    for v in split_vals
                    for c in (v.get("candidates") or [])
                ],
                1.0,
            ),
            "stable_candidate_rows": sum(
                1
                for v in split_vals
                for c in (v.get("candidates") or [])
                if bool(c.get("parse_success", True)) and not bool(c.get("empty_output", False)) and safe_float(c.get("repetition_rate"), 0.0) < 0.75
            ),
            "candidate_rows": sum(len(v.get("candidates") or []) for v in split_vals),
        }
    return out


def choose_tasks(task_summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    quotas = [
        ("reasoning", 12),
        ("science", 12),
        ("math_simple_arithmetic", 8),
        ("coding", 8),
    ]
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def score(row: dict[str, Any]) -> tuple[int, int, int, float, str]:
        split_rank = {"heldout": 3, "val": 2, "train": 1}.get(str(row.get("split")), 0)
        return (
            split_rank,
            int(row.get("non_tie_candidate_sets") or 0),
            int(row.get("candidate_set_count") or 0),
            safe_float(row.get("parse_rate"), 0.0),
            str(row.get("task_id")),
        )

    for domain, quota in quotas:
        rows = [row for row in task_summaries.values() if normalize_domain(row.get("domain")) == domain]
        rows = sorted(rows, key=score, reverse=True)
        for row in rows[:quota]:
            if row["task_id"] not in selected_ids:
                selected.append(row)
                selected_ids.add(row["task_id"])
    if len(selected) < 48:
        remaining = [row for row in task_summaries.values() if row["task_id"] not in selected_ids]
        for row in sorted(remaining, key=score, reverse=True):
            selected.append(row)
            selected_ids.add(row["task_id"])
            if len(selected) >= 48:
                break
    return selected[:48]


def run_task_suite() -> int:
    ensure_out_root()
    started = time.time()
    if not INVENTORY_JSON.exists():
        run_inventory()
    records = load_dataset_records()
    catalog = load_task_catalog()
    summaries = record_task_summary(records)
    selected = choose_tasks(summaries)
    task_rows = []
    for idx, row in enumerate(selected):
        cat = catalog.get(str(row.get("task_id")), {})
        domain = normalize_domain(row.get("domain") or cat.get("domain"))
        source_status = split_source_status(str(row.get("split")))
        task_rows.append(
            {
                "task_id": row.get("task_id"),
                "domain": domain,
                "source_dataset": cat.get("source_dataset") or row.get("source_dataset") or "cached_survival_dataset",
                "prompt": cat.get("prompt") or "Prompt text not present in cached source artifact; branch outcomes and labels are cached.",
                "options": cat.get("options") or {},
                "answer": cat.get("answer") or "",
                "parser_verifier_type": cat.get("parser_verifier_type") or ("cached_python_tests" if domain == "coding" else "cached_reward_label"),
                "clean_baseline_status": "available_as_clean_or_first_candidate",
                "prior_hidden_origin_availability": "available_cached" if row.get("candidate_set_count") else "unknown",
                "split_source_status": source_status,
                "split": row.get("split"),
                "candidate_set_ids": row.get("candidate_set_ids") or [],
                "branch_group_ids": row.get("branch_group_ids") or [],
                "candidate_set_count": row.get("candidate_set_count"),
                "non_tie_candidate_sets": row.get("non_tie_candidate_sets"),
                "parse_rate": row.get("parse_rate"),
                "suite_index": idx,
            }
        )
    domains = Counter(row["domain"] for row in task_rows)
    statuses = Counter(row["split_source_status"] for row in task_rows)
    if len(task_rows) >= 48 and domains.get("reasoning", 0) >= 12 and domains.get("science", 0) >= 12 and len(domains) >= 2:
        verdict = "REUSED_ONLY" if not statuses.get("fresh") else "READY"
    elif len(task_rows) >= 24 and len(domains) >= 2:
        verdict = "PARTIAL"
    elif task_rows:
        verdict = "REUSED_ONLY"
    else:
        verdict = "BLOCKED"
    payload = {
        "BG_SELECTION_ONLY_TASK_SUITE_VERDICT": verdict,
        "verdict": verdict,
        "tasks": task_rows,
        "counts": {
            "tasks": len(task_rows),
            "domains": dict(domains),
            "split_source_status": dict(statuses),
            "candidate_sets": sum(len(row.get("candidate_set_ids") or []) for row in task_rows),
        },
        "selection_notes": [
            "Fresh live task generation was not required because completed cached BGV1/v4/survival candidate sets were available.",
            "Every task is marked reused_heldout or reused_diagnostic; no threshold was selected on heldout in this stage.",
            "Coding rows are cached verifier-label candidate pools, not headline-only code/devil claims.",
        ],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(TASK_SUITE_JSON, payload)
    write_csv(TASK_SUITE_CSV, [{k: v for k, v in row.items() if k not in {"prompt", "options", "candidate_set_ids", "branch_group_ids"}} | {"candidate_sets": len(row.get("candidate_set_ids") or []), "branch_groups": len(row.get("branch_group_ids") or [])} for row in task_rows])
    lines = ["# Selection-Only Phase 2 Task Suite", "", f"BG_SELECTION_ONLY_TASK_SUITE_VERDICT = {verdict}", "", f"- tasks: `{len(task_rows)}`", f"- domains: `{dict(domains)}`", f"- split/source: `{dict(statuses)}`", "", "## Tasks", ""]
    lines.extend(md_table(task_rows, ["suite_index", "task_id", "domain", "source_dataset", "split", "split_source_status", "candidate_set_count", "non_tie_candidate_sets", "parser_verifier_type"]))
    write_md(TASK_SUITE_MD, lines)
    print(f"BG_SELECTION_ONLY_TASK_SUITE_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


def load_task_suite() -> dict[str, Any]:
    if not TASK_SUITE_JSON.exists():
        run_task_suite()
    return load_json(TASK_SUITE_JSON, {}) or {}


def selected_suite_records() -> list[dict[str, Any]]:
    suite = load_task_suite()
    ids = {cid for task in suite.get("tasks") or [] for cid in (task.get("candidate_set_ids") or [])}
    records = [record for record in load_dataset_records() if str(record.get("candidate_set_id")) in ids]
    order = {cid: idx for idx, cid in enumerate(sorted(ids))}
    return sorted(records, key=lambda r: order.get(str(r.get("candidate_set_id")), 10**9))


def raw_branch_index() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for source_id, rows in (("branch_generator_v1", load_generator_rows()), ("quota_v4", load_v4_branch_rows())):
        for row in rows:
            gid = str(row.get("branch_group_id"))
            bid = str(row.get("branch_id"))
            out[f"{source_id}::{gid}::{bid}"] = row
    return out


def candidate_raw(candidate: dict[str, Any], raw_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cid = str(candidate.get("candidate_id"))
    raw = raw_index.get(cid, {})
    if raw:
        return raw
    parts = cid.split("::")
    if len(parts) >= 3:
        key = "::".join([parts[0], "::".join(parts[1:-1]), parts[-1]])
        return raw_index.get(key, {})
    return {}


def missing_flags(record: dict[str, Any]) -> dict[str, Any]:
    coverage = record.get("expert_coverage") or {}
    critical = ["old_frozen_bg", "old_content_head", "old_code_head", "hidden_branch_head", "bridge_only_head", "universal"]
    return {f"missing_{name}": safe_float(coverage.get(name), 0.0) <= 0.0 for name in critical}


def run_policy_runner() -> int:
    ensure_out_root()
    started = time.time()
    if not TASK_SUITE_JSON.exists():
        run_task_suite()
    force = os.environ.get("FORCE_RERUN") == "1"
    if force and POLICY_PROGRESS_JSON.exists():
        POLICY_PROGRESS_JSON.unlink()
    records = selected_suite_records()
    weights = load_best_composite()
    params = load_best_veto()
    raw_index = raw_branch_index()
    output_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    completed_tasks: set[str] = set()
    completed_groups: set[str] = set()
    for record in records:
        selected = topk_policy(record, "fixed_composite", 4, weights)
        fixed_scores = score_values(record, "fixed_composite", weights)
        order = sorted(range(len(fixed_scores)), key=lambda i: (-safe_float(fixed_scores[i], 0.0), i))
        rank_map = {idx: rank + 1 for rank, idx in enumerate(order)}
        family_scores = {family: score_values(record, family, weights) for family in ["old", "code", "branch", "bridge", "universal", "gated"]}
        v4_scores = score_values(record, "v4_hidden_origin", weights)
        guardrail_selected = missing_ood_survivors(record, weights, params, set())
        veto_selected = selection_for_policy(record, "previous_veto_rescue", weights, params)
        flags = missing_flags(record)
        selected_set = set(selected)
        oracles = oracle_indices(record)
        for idx, candidate in enumerate(record.get("candidates") or []):
            raw = candidate_raw(candidate, raw_index)
            row = {
                "task_id": record.get("task_id"),
                "domain": record.get("domain"),
                "split": record.get("split"),
                "candidate_set_id": record.get("candidate_set_id"),
                "branch_group_id": record.get("group_id"),
                "branch_id": candidate.get("candidate_id"),
                "candidate_index": idx,
                "layer": record.get("layer"),
                "branch_origin": candidate.get("origin"),
                "generator_method": raw.get("generator_method") or candidate.get("generator_method"),
                "branch_method": raw.get("branch_method"),
                "alpha": raw.get("alpha"),
                "delta_family": raw.get("delta_family"),
                "effective_rms": raw.get("effective_delta_rms"),
                "selected_by_policy": idx in selected_set,
                "selected_rank": rank_map.get(idx) if idx in selected_set else None,
                "fixed_composite_score": fixed_scores[idx] if idx < len(fixed_scores) else None,
                "old_score": family_scores["old"][idx] if idx < len(family_scores["old"]) else None,
                "code_score": family_scores["code"][idx] if idx < len(family_scores["code"]) else None,
                "branch_score": family_scores["branch"][idx] if idx < len(family_scores["branch"]) else None,
                "bridge_score": family_scores["bridge"][idx] if idx < len(family_scores["bridge"]) else None,
                "universal_score": family_scores["universal"][idx] if idx < len(family_scores["universal"]) else None,
                "gated_score": family_scores["gated"][idx] if idx < len(family_scores["gated"]) else None,
                "v4_hidden_origin_score": v4_scores[idx] if idx < len(v4_scores) else None,
                "ood_missing_flags": flags,
                "veto_rescue_selected": idx in set(veto_selected),
                "missing_ood_guardrail_selected": idx in set(guardrail_selected),
                "output_text": raw.get("output_text", ""),
                "parsed_answer": raw.get("parsed_answer"),
                "correctness": candidate_correct(record, idx),
                "reward": safe_float(candidate.get("reward"), 0.0),
                "parse_success": bool(candidate.get("parse_success", True)),
                "repetition_rate": safe_float(candidate.get("repetition_rate"), 0.0),
                "empty_output": bool(candidate.get("empty_output", False)),
                "hit_max_tokens": bool(raw.get("hit_max_tokens", False)),
                "stability_flags": {
                    "parse_success": bool(candidate.get("parse_success", True)),
                    "empty_output": bool(candidate.get("empty_output", False)),
                    "high_repetition": safe_float(candidate.get("repetition_rate"), 0.0) >= 0.75,
                    "off_manifold_warning": bool(raw.get("off_manifold_warning", False)),
                    "nan_inf": bool(raw.get("nan_inf", False)),
                },
                "oracle_best": idx in oracles,
                "false_prune_contribution": 1.0 if idx in oracles and idx not in selected_set else 0.0,
            }
            output_rows.append(row)
        group_metric = evaluation_row(record, POLICY_NAME, selected)
        group_metric.update(
            {
                "veto_rescue_indices": veto_selected,
                "missing_ood_guardrail_indices": guardrail_selected,
                "selected_mode": "cached_counterfactual_full_generation",
                "lineage_mode": "HOOK_LAYERWISE_APPROX",
                "true_fork_carry_claimed": False,
                "compute_savings_claimed": False,
            }
        )
        group_rows.append(group_metric)
        completed_tasks.add(str(record.get("task_id")))
        completed_groups.add(str(record.get("group_id")))
        progress = {
            "completed_task_ids": sorted(completed_tasks),
            "completed_branch_group_ids": sorted(completed_groups),
            "last_candidate_set_id": record.get("candidate_set_id"),
            "rows_written": len(output_rows),
            "updated_at_seconds": round(time.time(), 3),
        }
        write_json(POLICY_PROGRESS_JSON, progress)
        append_jsonl(POLICY_PROGRESS_JSONL, {"event": "candidate_set_complete", **progress})
    metrics = add_task_macro(group_rows)
    verdict = "HOOK_LAYERWISE_APPROX" if records else "BLOCKED"
    payload = {
        "BG_SELECTION_ONLY_POLICY_RUNNER_VERDICT": verdict,
        "verdict": verdict,
        "mode": "counterfactual_full_generation_from_cached_completed_candidates",
        "budgeted_selected_generation": "SKIPPED",
        "cached_candidate_replay": "RUN",
        "lineage_mode": "HOOK_LAYERWISE_APPROX",
        "policy": POLICY_NAME,
        "weights": weights,
        "veto_rescue_parameters": params,
        "records": records,
        "group_rows": group_rows,
        "rows": output_rows,
        "metrics_by_policy": metrics,
        "completed_task_ids": sorted(completed_tasks),
        "completed_branch_group_ids": sorted(completed_groups),
        "notes": [
            "All generated branches were already continued to outcome in cached artifacts.",
            "This run applies fixed-composite top4 offline and does not claim compute savings.",
            "Layerwise carry is approximated by layer-specific hook candidate sets; true fork/carry is not claimed.",
        ],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(payload, POLICY_OUTPUTS_PT)
    write_json(POLICY_OUTPUTS_JSON, {k: v for k, v in payload.items() if k != "records"})
    write_csv(POLICY_ROWS_CSV, [{k: v for k, v in row.items() if k not in {"output_text", "ood_missing_flags", "stability_flags"}} for row in output_rows])
    lines = [
        "# Selection-Only Policy Runner",
        "",
        f"BG_SELECTION_ONLY_POLICY_RUNNER_VERDICT = {verdict}",
        "",
        "- primary mode: `counterfactual_full_generation` over cached completed candidates",
        "- budgeted selected-only mode: `SKIPPED`",
        "- action steering: `not tested`",
        "- true fork/carry claim: `false`",
        "",
        "## Metrics",
        "",
    ]
    lines.extend(md_table([{"policy": key, **compact_metrics(value)} for key, value in metrics.items()], ["policy", "n", "oracle_retention", "false_prune_rate", "average_survivors", "best_selected_reward", "mean_selected_reward", "final_selected_reward", "task_macro_reward"]))
    write_md(POLICY_REPORT_MD, lines)
    write_json(POLICY_REPORT_JSON, {k: v for k, v in payload.items() if k not in {"records", "rows"}})
    print(f"BG_SELECTION_ONLY_POLICY_RUNNER_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


def policy_rows_for_records(records: Sequence[dict[str, Any]], policies: Sequence[str]) -> list[dict[str, Any]]:
    weights = load_best_composite()
    params = load_best_veto()
    rows = []
    for record in records:
        for policy in policies:
            rows.append(evaluation_row(record, policy, selection_for_policy(record, policy, weights, params)))
    return rows


def run_cached_reproduction() -> int:
    ensure_out_root()
    started = time.time()
    records = [r for r in load_dataset_records() if r.get("split") == "heldout"]
    policies = [
        POLICY_NAME,
        "fixed_composite_top3",
        "old_frozen_bg_top4",
        "hidden_origin_v4_top4",
        "learned_gated_top4",
        "random_top4",
        "clean_only",
        "previous_veto_rescue",
        "oracle",
    ]
    rows = policy_rows_for_records(records, policies)
    metrics = add_task_macro(rows)
    selected = metrics.get(POLICY_NAME, {})
    drift = {key: safe_float(selected.get(key), 999.0) - expected for key, expected in EXPECTED_HELDOUT.items()}
    code_rows = [row for row in rows if row.get("domain") == "coding" and row.get("policy") == POLICY_NAME]
    code_ret = finite_mean([row.get("oracle_retention") for row in code_rows], 1.0)
    code_false = finite_mean([row.get("false_prune_rate") for row in code_rows], 0.0)
    max_abs = max([abs(v) for v in drift.values()] or [999.0])
    if max_abs <= 0.005 and code_ret >= 0.99 and code_false <= 0.01:
        verdict = "REPRODUCED"
    elif max_abs <= 0.025:
        verdict = "SMALL_DRIFT"
    elif rows:
        verdict = "FAILED_REPRODUCTION"
    else:
        verdict = "INSUFFICIENT"
    payload = {
        "BG_SELECTION_ONLY_CACHED_REPRODUCTION_VERDICT": verdict,
        "verdict": verdict,
        "expected": EXPECTED_HELDOUT,
        "selected_metrics": selected,
        "drift": drift,
        "coding_preservation": {"oracle_retention": code_ret, "false_prune_rate": code_false, "n": len(code_rows)},
        "rows": rows,
        "metrics_by_policy": metrics,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(CACHED_JSON, payload)
    write_csv(CACHED_CSV, rows)
    lines = ["# Cached Reproduction", "", f"BG_SELECTION_ONLY_CACHED_REPRODUCTION_VERDICT = {verdict}", "", "## Metrics", ""]
    lines.extend(md_table([{"policy": key, **compact_metrics(value)} for key, value in metrics.items()], ["policy", "n", "oracle_retention", "false_prune_rate", "average_survivors", "best_selected_reward", "final_selected_reward", "task_macro_reward"]))
    lines.extend(["", f"- coding retention: `{code_ret:.3f}`", f"- coding false-prune: `{code_false:.3f}`"])
    write_md(CACHED_MD, lines)
    print(f"BG_SELECTION_ONLY_CACHED_REPRODUCTION_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "FAILED_REPRODUCTION" else 1


def load_policy_outputs() -> dict[str, Any]:
    if not POLICY_OUTPUTS_PT.exists():
        run_policy_runner()
    return torch.load(POLICY_OUTPUTS_PT, map_location="cpu", weights_only=False)


def run_live_prototype() -> int:
    ensure_out_root()
    started = time.time()
    if not POLICY_OUTPUTS_PT.exists():
        run_policy_runner()
    payload = load_policy_outputs()
    records = payload.get("records") or []
    policies = [
        "clean_only",
        "random_top4",
        "old_frozen_bg_top4",
        "old_code_objective_top4",
        "hidden_origin_v4_top4",
        "learned_gated_top4",
        "fixed_composite_top3",
        "previous_veto_rescue",
        POLICY_NAME,
        "oracle",
    ]
    rows = policy_rows_for_records(records, policies)
    metrics = add_task_macro(rows)
    selected = metrics.get(POLICY_NAME, {})
    clean = metrics.get("clean_only", {})
    random = metrics.get("random_top4", {})
    selected_reward = safe_float(selected.get("task_macro_best_selected_reward"), safe_float(selected.get("best_selected_reward"), 0.0))
    clean_reward = safe_float(clean.get("task_macro_reward"), safe_float(clean.get("final_selected_reward"), 0.0))
    random_reward = safe_float(random.get("task_macro_best_selected_reward"), safe_float(random.get("best_selected_reward"), 0.0))
    survival_ok = safe_float(selected.get("oracle_retention"), 0.0) >= 0.85 and safe_float(selected.get("false_prune_rate"), 1.0) <= 0.15
    beats_minimum = selected_reward > clean_reward and selected_reward >= random_reward
    final_top1 = safe_float(selected.get("task_macro_reward"), safe_float(selected.get("final_selected_reward"), 0.0))
    final_weak = final_top1 + 1e-9 < selected_reward
    stable = safe_float(selected.get("parse_success_rate"), 1.0) >= 0.80 and safe_float(selected.get("stable_rate"), 1.0) >= 0.80
    non_oracle_policies = [p for p in policies if p not in {POLICY_NAME, "oracle"}]
    best_non_oracle = max([safe_float(metrics.get(p, {}).get("task_macro_best_selected_reward"), safe_float(metrics.get(p, {}).get("best_selected_reward"), -999.0)) for p in non_oracle_policies] or [-999.0])
    if survival_ok and beats_minimum and selected_reward > best_non_oracle and not final_weak and stable:
        verdict = "SELECTION_ONLY_POSITIVE"
    elif survival_ok and beats_minimum and stable:
        verdict = "SURVIVAL_POSITIVE_FINAL_SELECTION_WEAK" if final_weak else "SELECTION_ONLY_POSITIVE"
    elif not survival_ok:
        verdict = "SURVIVAL_WEAK"
    elif not stable:
        verdict = "UNSTABLE"
    elif not beats_minimum:
        verdict = "NO_IMPROVEMENT"
    else:
        verdict = "DATA_LIMITED"
    selected_branch_rows = [row for row in payload.get("rows") or [] if row.get("selected_by_policy")]
    out = {
        "BG_SELECTION_ONLY_LIVE_PROTOTYPE_VERDICT": verdict,
        "verdict": verdict,
        "mode": "cached_counterfactual_full_generation_no_fresh_model_run",
        "lineage_mode": payload.get("lineage_mode"),
        "selection_policy": POLICY_NAME,
        "rows": rows,
        "selected_branch_rows": selected_branch_rows,
        "metrics_by_policy": metrics,
        "primary_success_checks": {
            "survival_ok": survival_ok,
            "beats_clean_random_on_best_selected_reward": beats_minimum,
            "final_selection_weak": final_weak,
            "stable": stable,
        },
        "notes": [
            "This stage used cached completed branches; it is counterfactual selection-only, not fresh live generation.",
            "Top4 survival is evaluated separately from final top1 arbiter quality.",
            "No action steering was tested.",
        ],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(LIVE_JSON, out)
    write_csv(LIVE_CSV, rows)
    write_csv(LIVE_SELECTED_CSV, [{k: v for k, v in row.items() if k not in {"output_text", "ood_missing_flags", "stability_flags"}} for row in selected_branch_rows])
    lines = ["# Live/Counterfactual Selection-Only Prototype", "", f"BG_SELECTION_ONLY_LIVE_PROTOTYPE_VERDICT = {verdict}", "", "## Metrics", ""]
    lines.extend(md_table([{"policy": key, **compact_metrics(value)} for key, value in metrics.items()], ["policy", "n", "oracle_retention", "false_prune_rate", "average_survivors", "best_selected_reward", "mean_selected_reward", "final_selected_reward", "task_macro_reward"]))
    lines.extend(["", "## Mode Notes", ""])
    lines.extend(f"- {item}" for item in out["notes"])
    write_md(LIVE_MD, lines)
    print(f"BG_SELECTION_ONLY_LIVE_PROTOTYPE_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


def final_arbiter_rows(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    weights = load_best_composite()
    params = load_best_veto()
    rows = []
    for record in records:
        survivors = selection_for_policy(record, POLICY_NAME, weights, params)
        if not survivors:
            continue
        families = {
            "fixed_composite_top1_among_survivors": "fixed_composite",
            "old_content_top1_among_survivors": "old",
            "old_code_objective_top1_among_survivors": "code",
            "bridge_top1_among_survivors": "bridge",
            "v4_hidden_origin_top1_among_survivors": "v4_hidden_origin",
            "learned_gated_top1_among_survivors": "gated",
            "universal_top1_among_survivors": "universal",
        }
        for policy, family in families.items():
            scores = score_values(record, family, weights)
            idx = top1_among(survivors, scores)
            rows.append(evaluation_row(record, policy, [idx] if idx is not None else survivors[:1]))
        majority = majority_rank_top1(record, survivors, weights)
        rows.append(evaluation_row(record, "majority_rank_aggregation_among_survivors", [majority] if majority is not None else survivors[:1]))
        rows.append(evaluation_row(record, "verifier_or_test_if_available_diagnostic", sorted(oracle_indices(record))))
        rows.append(evaluation_row(record, "oracle_upper_bound_diagnostic", sorted(oracle_indices(record))))
    return rows


def run_final_arbiter() -> int:
    ensure_out_root()
    started = time.time()
    if not LIVE_JSON.exists():
        run_live_prototype()
    records = load_policy_outputs().get("records") or []
    rows = final_arbiter_rows(records)
    metrics = add_task_macro(rows)
    non_oracle = {k: v for k, v in metrics.items() if "oracle" not in k and "verifier" not in k}
    best_policy = max(non_oracle, key=lambda k: safe_float(non_oracle[k].get("task_macro_reward"), safe_float(non_oracle[k].get("final_selected_reward"), -999.0))) if non_oracle else ""
    best_reward = safe_float(non_oracle.get(best_policy, {}).get("task_macro_reward"), safe_float(non_oracle.get(best_policy, {}).get("final_selected_reward"), 0.0))
    survival = (load_json(LIVE_JSON, {}) or {}).get("metrics_by_policy", {}).get(POLICY_NAME, {})
    survival_reward = safe_float(survival.get("task_macro_best_selected_reward"), safe_float(survival.get("best_selected_reward"), 0.0))
    if not rows:
        verdict = "INSUFFICIENT"
    elif best_reward + 1e-9 < survival_reward:
        verdict = "FINAL_SELECTION_WEAK"
    elif best_policy.startswith("old"):
        verdict = "OLD_CONTENT_BEST"
    elif best_policy.startswith("fixed"):
        verdict = "FIXED_COMPOSITE_BEST"
    elif best_policy.startswith("bridge"):
        verdict = "BRIDGE_BEST"
    else:
        verdict = "FINAL_ARBITER_READY"
    out = {
        "BG_SELECTION_ONLY_FINAL_ARBITER_VERDICT": verdict,
        "verdict": verdict,
        "best_policy": best_policy,
        "rows": rows,
        "metrics_by_policy": metrics,
        "survival_best_selected_reward": survival_reward,
        "cases_survival_succeeded_final_failed": [
            row
            for row in rows
            if row.get("policy") == best_policy and safe_float(row.get("oracle_retention"), 0.0) >= 1.0 and safe_float(row.get("top1_success"), 0.0) < 1.0
        ][:50],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(ARBITER_JSON, out)
    write_csv(ARBITER_CSV, rows)
    lines = ["# Final Arbiter Analysis", "", f"BG_SELECTION_ONLY_FINAL_ARBITER_VERDICT = {verdict}", "", f"- best non-oracle arbiter: `{best_policy}`", "", "## Metrics", ""]
    lines.extend(md_table([{"policy": key, **compact_metrics(value)} for key, value in metrics.items()], ["policy", "n", "final_selected_reward", "final_selected_correctness", "top1_success", "task_macro_reward", "final_regret"]))
    write_md(ARBITER_MD, lines)
    print(f"BG_SELECTION_ONLY_FINAL_ARBITER_VERDICT = {verdict}", flush=True)
    return 0


def run_baseline_comparison() -> int:
    ensure_out_root()
    started = time.time()
    if not POLICY_OUTPUTS_PT.exists():
        run_policy_runner()
    records = load_policy_outputs().get("records") or []
    policies = [
        "clean_only",
        "random_top1",
        "random_top2",
        "random_top4",
        "old_frozen_bg_top1",
        "old_frozen_bg_top4",
        "old_code_objective_top1",
        "old_code_objective_top4",
        "hidden_origin_v4_top1",
        "hidden_origin_v4_top4",
        "learned_gated_top1",
        "learned_gated_top4",
        "universal_top1",
        "universal_top4",
        "bridge_top1",
        "bridge_top4",
        "fixed_composite_top3",
        POLICY_NAME,
        "fixed_composite_top4_without_rescue",
        "fixed_composite_top4_without_missing_ood_fallback",
        "previous_veto_rescue",
        "oracle",
    ]
    rows = policy_rows_for_records(records, policies)
    metrics = add_task_macro(rows)
    selected_reward = safe_float(metrics.get(POLICY_NAME, {}).get("task_macro_best_selected_reward"), safe_float(metrics.get(POLICY_NAME, {}).get("best_selected_reward"), -999.0))
    non_oracle = {p: m for p, m in metrics.items() if p not in {POLICY_NAME, "oracle"}}
    best_other = max(non_oracle, key=lambda p: safe_float(non_oracle[p].get("task_macro_best_selected_reward"), safe_float(non_oracle[p].get("best_selected_reward"), -999.0))) if non_oracle else ""
    best_other_reward = safe_float(non_oracle.get(best_other, {}).get("task_macro_best_selected_reward"), safe_float(non_oracle.get(best_other, {}).get("best_selected_reward"), -999.0))
    if selected_reward > best_other_reward + 1e-9:
        verdict = "FIXED_COMPOSITE_TOP4_WINS"
    elif abs(selected_reward - best_other_reward) <= 1e-9:
        verdict = "NO_CLEAR_WINNER"
    elif best_other.startswith("old") or best_other.startswith("hidden_origin"):
        verdict = "OLD_OR_V4_BEST"
    elif best_other.startswith("random"):
        verdict = "RANDOM_COMPETITIVE"
    elif best_other == "clean_only":
        verdict = "CLEAN_BEST"
    elif records:
        verdict = "FIXED_COMPOSITE_TOP4_WEAK"
    else:
        verdict = "DATA_LIMITED"
    out = {
        "BG_SELECTION_ONLY_BASELINE_COMPARISON_VERDICT": verdict,
        "verdict": verdict,
        "best_non_oracle_baseline": best_other,
        "rows": rows,
        "metrics_by_policy": metrics,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(BASELINE_JSON, out)
    write_csv(BASELINE_CSV, rows)
    lines = ["# Baseline Comparison", "", f"BG_SELECTION_ONLY_BASELINE_COMPARISON_VERDICT = {verdict}", "", f"- best non-oracle baseline: `{best_other}`", "", "## Metrics", ""]
    lines.extend(md_table([{"policy": key, **compact_metrics(value)} for key, value in metrics.items()], ["policy", "n", "oracle_retention", "false_prune_rate", "average_survivors", "best_selected_reward", "final_selected_reward", "task_macro_reward"]))
    write_md(BASELINE_MD, lines)
    print(f"BG_SELECTION_ONLY_BASELINE_COMPARISON_VERDICT = {verdict}", flush=True)
    return 0


def run_domain_coding() -> int:
    ensure_out_root()
    started = time.time()
    if not BASELINE_JSON.exists():
        run_baseline_comparison()
    baseline = load_json(BASELINE_JSON, {}) or {}
    rows = [row for row in baseline.get("rows") or [] if row.get("policy") in {POLICY_NAME, "clean_only"}]
    by_domain = aggregate_numeric(rows, ("policy", "domain"))
    fixed_domains = {key.split("::", 1)[1]: vals for key, vals in by_domain.items() if key.startswith(f"{POLICY_NAME}::")}
    clean_domains = {key.split("::", 1)[1]: vals for key, vals in by_domain.items() if key.startswith("clean_only::")}
    domain_rows = []
    for domain, fixed in sorted(fixed_domains.items()):
        clean = clean_domains.get(domain, {})
        domain_rows.append(
            {
                "domain": domain,
                "fixed_oracle_retention": fixed.get("oracle_retention"),
                "fixed_false_prune_rate": fixed.get("false_prune_rate"),
                "fixed_best_selected_reward": fixed.get("best_selected_reward"),
                "fixed_final_selected_reward": fixed.get("final_selected_reward"),
                "clean_final_selected_reward": clean.get("final_selected_reward"),
                "clean_vs_selected_improvement": safe_float(fixed.get("best_selected_reward"), 0.0) - safe_float(clean.get("final_selected_reward"), 0.0),
                "parse_success_rate": fixed.get("parse_success_rate"),
                "stable_rate": fixed.get("stable_rate"),
            }
        )
    coding = next((row for row in domain_rows if row["domain"] == "coding"), None)
    coding_status = "NOT_TESTED"
    if coding:
        if safe_float(coding.get("fixed_oracle_retention"), 0.0) >= 0.95 and safe_float(coding.get("fixed_false_prune_rate"), 1.0) <= 0.05:
            coding_status = "PRESERVED"
        else:
            coding_status = "WEAK"
    positive_domains = sum(1 for row in domain_rows if safe_float(row.get("clean_vs_selected_improvement"), 0.0) >= -1e-9 and safe_float(row.get("fixed_oracle_retention"), 0.0) >= 0.85)
    if coding_status == "PRESERVED" and positive_domains >= 3:
        verdict = "MULTIDOMAIN_POSITIVE"
    elif coding_status == "PRESERVED":
        verdict = "CODING_PRESERVED"
    elif set(fixed_domains).issubset({"reasoning", "science"}):
        verdict = "REASONING_SCIENCE_ONLY"
    elif coding_status == "WEAK":
        verdict = "CODING_WEAK"
    elif positive_domains:
        verdict = "DOMAIN_MIXED"
    else:
        verdict = "DATA_LIMITED"
    out = {
        "BG_SELECTION_ONLY_DOMAIN_CODING_VERDICT": verdict,
        "verdict": verdict,
        "CODING_STATUS": "NOT_TESTED" if coding is None else coding_status,
        "domain_rows": domain_rows,
        "metrics_by_policy_domain": by_domain,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(DOMAIN_JSON, out)
    lines = ["# Domain and Coding Analysis", "", f"BG_SELECTION_ONLY_DOMAIN_CODING_VERDICT = {verdict}", "", f"CODING_STATUS = {out['CODING_STATUS']}", "", "## Domains", ""]
    lines.extend(md_table(domain_rows, ["domain", "fixed_oracle_retention", "fixed_false_prune_rate", "fixed_best_selected_reward", "fixed_final_selected_reward", "clean_final_selected_reward", "clean_vs_selected_improvement", "parse_success_rate"]))
    write_md(DOMAIN_MD, lines)
    print(f"BG_SELECTION_ONLY_DOMAIN_CODING_VERDICT = {verdict}", flush=True)
    return 0


def run_failures() -> int:
    ensure_out_root()
    started = time.time()
    if not POLICY_OUTPUTS_PT.exists():
        run_policy_runner()
    records = load_policy_outputs().get("records") or []
    weights = load_best_composite()
    params = load_best_veto()
    failures = []
    summary = Counter()
    for record in records:
        selected = selection_for_policy(record, POLICY_NAME, weights, params)
        selected_set = set(selected)
        oracles = oracle_indices(record)
        fixed_top1 = selected[0] if selected else clean_index(record)
        rewards = candidate_rewards(record)
        clean = clean_index(record)
        tags = []
        if not (oracles & selected_set):
            tags.append("top4_missed_oracle")
        if oracles & selected_set and fixed_top1 not in oracles:
            tags.append("survival_succeeded_final_arbiter_failed")
        if rewards and rewards[clean] >= max(rewards) and any(i != clean and rewards[i] < rewards[clean] for i in range(len(rewards))):
            tags.append("clean_branch_beat_generated")
        if any(not bool(c.get("parse_success", True)) for c in record.get("candidates") or []):
            tags.append("parse_failures_present")
        if any(safe_float(c.get("repetition_rate"), 0.0) >= 0.75 for c in record.get("candidates") or []):
            tags.append("repetition_failures_present")
        flags = missing_flags(record)
        if any(flags.values()):
            tags.append("missing_expert_fallback_relevant")
        for tag in tags:
            summary[tag] += 1
        if tags:
            root = "final_arbiter" if "survival_succeeded_final_arbiter_failed" in tags else ("survival_policy" if "top4_missed_oracle" in tags else "branch_generator")
            failures.append(
                {
                    "candidate_set_id": record.get("candidate_set_id"),
                    "task_id": record.get("task_id"),
                    "domain": record.get("domain"),
                    "split": record.get("split"),
                    "layer": record.get("layer"),
                    "branch_group_id": record.get("group_id"),
                    "tags": tags,
                    "root_cause_tag": root,
                    "suggested_fix": {
                        "final_arbiter": "train/evaluate a stronger arbiter among top4 survivors",
                        "survival_policy": "revisit survival rescue or keep more branches for affected regimes",
                        "branch_generator": "improve branch generator stability and useful alternative rate",
                    }.get(root, "audit parser/verifier and OOD guard"),
                    "selected_indices": selected,
                    "oracle_indices": sorted(oracles),
                    "fixed_top1": fixed_top1,
                    "oracle_reward": max(rewards) if rewards else 0.0,
                    "fixed_top1_reward": rewards[fixed_top1] if fixed_top1 < len(rewards) else 0.0,
                    "clean_reward": rewards[clean] if clean < len(rewards) else 0.0,
                    "candidate_count": len(rewards),
                }
            )
    false_prune = summary.get("top4_missed_oracle", 0)
    final_fail = summary.get("survival_succeeded_final_arbiter_failed", 0)
    if false_prune == 0 and final_fail == 0:
        verdict = "FAILURES_UNDERSTOOD"
    elif final_fail >= false_prune:
        verdict = "FINAL_ARBITER_BLOCKER"
    elif false_prune:
        verdict = "SURVIVAL_POLICY_BLOCKER"
    elif summary.get("parse_failures_present", 0):
        verdict = "PARSER_VERIFIER_BLOCKER"
    elif records:
        verdict = "FAILURES_UNDERSTOOD"
    else:
        verdict = "INCONCLUSIVE"
    out = {
        "BG_SELECTION_ONLY_FAILURE_ANALYSIS_VERDICT": verdict,
        "verdict": verdict,
        "failure_summary": dict(summary),
        "failure_cases": failures[:200],
        "top_20_failure_cases": failures[:20],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(FAILURE_JSON, out)
    write_csv(FAILURE_CSV, failures)
    lines = ["# Failure Analysis", "", f"BG_SELECTION_ONLY_FAILURE_ANALYSIS_VERDICT = {verdict}", "", f"- failure summary: `{dict(summary)}`", "", "## Top Failure Cases", ""]
    lines.extend(md_table(failures[:20], ["task_id", "domain", "layer", "tags", "root_cause_tag", "suggested_fix", "selected_indices", "oracle_indices", "fixed_top1_reward", "oracle_reward"]))
    write_md(FAILURE_MD, lines)
    print(f"BG_SELECTION_ONLY_FAILURE_ANALYSIS_VERDICT = {verdict}", flush=True)
    return 0


def run_steering_readiness() -> int:
    ensure_out_root()
    started = time.time()
    if not FAILURE_JSON.exists():
        run_failures()
    live = load_json(LIVE_JSON, {}) or {}
    arbiter = load_json(ARBITER_JSON, {}) or {}
    domain = load_json(DOMAIN_JSON, {}) or {}
    failure = load_json(FAILURE_JSON, {}) or {}
    selected = (live.get("metrics_by_policy") or {}).get(POLICY_NAME, {})
    checks = {
        "selection_positive_vs_clean_random": bool((live.get("primary_success_checks") or {}).get("beats_clean_random_on_best_selected_reward")),
        "survival_retains_good_branches": safe_float(selected.get("oracle_retention"), 0.0) >= 0.85 and safe_float(selected.get("false_prune_rate"), 1.0) <= 0.15,
        "final_arbiter_not_broken": arbiter.get("verdict") not in {"FINAL_SELECTION_WEAK", "INSUFFICIENT", ""},
        "old_context_coding_not_degraded": domain.get("CODING_STATUS") in {"PRESERVED", "NOT_TESTED"},
        "stability_acceptable": safe_float(selected.get("parse_success_rate"), 1.0) >= 0.80,
        "failure_modes_understood": failure.get("verdict") in {"FAILURES_UNDERSTOOD", "FINAL_ARBITER_BLOCKER"},
        "no_action_steering_claim": True,
    }
    if all(checks.values()):
        verdict = "READY_FOR_SELECTION_PLUS_STEERING_TEST"
        blocker = ""
    elif not checks["final_arbiter_not_broken"]:
        verdict = "NEEDS_FINAL_ARBITER_FIRST"
        blocker = "final arbiter among top4 survivors is weaker than survival upper bound"
    elif not checks["selection_positive_vs_clean_random"]:
        verdict = "NOT_READY"
        blocker = "selection-only did not beat clean/random on the primary proxy"
    elif not checks["survival_retains_good_branches"]:
        verdict = "NEEDS_SURVIVAL_POLICY_FIRST"
        blocker = "top4 survival missed oracle too often"
    elif not checks["stability_acceptable"]:
        verdict = "NEEDS_BRANCH_GENERATOR_FIRST"
        blocker = "branch generation stability/parse rate is weak"
    elif not checks["old_context_coding_not_degraded"]:
        verdict = "NOT_READY"
        blocker = "old-context or coding behavior degraded"
    else:
        verdict = "NEEDS_MORE_SELECTION_ONLY_DATA"
        blocker = "cached-only data limits the steering comparison decision"
    out = {
        "BG_SELECTION_ONLY_TO_STEERING_READINESS_VERDICT": verdict,
        "verdict": verdict,
        "checks": checks,
        "blocker": blocker,
        "selection_only_baseline_locked": verdict == "READY_FOR_SELECTION_PLUS_STEERING_TEST",
        "next_prompt": "selection-only vs selection + trained steering corridor" if verdict == "READY_FOR_SELECTION_PLUS_STEERING_TEST" else "",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(STEERING_JSON, out)
    lines = ["# Steering Readiness", "", f"BG_SELECTION_ONLY_TO_STEERING_READINESS_VERDICT = {verdict}", "", f"- blocker: `{blocker or 'none'}`", "", "## Checks", ""]
    lines.extend(md_table([{"check": key, "passed": value} for key, value in checks.items()], ["check", "passed"]))
    write_md(STEERING_MD, lines)
    print(f"BG_SELECTION_ONLY_TO_STEERING_READINESS_VERDICT = {verdict}", flush=True)
    return 0


def load_stage_payloads() -> dict[str, dict[str, Any]]:
    return {
        "inventory": load_json(INVENTORY_JSON, {}) or {},
        "task_suite": load_json(TASK_SUITE_JSON, {}) or {},
        "policy_runner": load_json(POLICY_REPORT_JSON, {}) or {},
        "cached_reproduction": load_json(CACHED_JSON, {}) or {},
        "live_prototype": load_json(LIVE_JSON, {}) or {},
        "final_arbiter": load_json(ARBITER_JSON, {}) or {},
        "baseline_comparison": load_json(BASELINE_JSON, {}) or {},
        "domain_coding": load_json(DOMAIN_JSON, {}) or {},
        "failure_analysis": load_json(FAILURE_JSON, {}) or {},
        "steering_readiness": load_json(STEERING_JSON, {}) or {},
    }


def verdict(payload: dict[str, Any], key: str, default: str = "INSUFFICIENT") -> str:
    return str(payload.get(key) or payload.get("verdict") or default)


def final_status(data: dict[str, dict[str, Any]]) -> str:
    live = verdict(data["live_prototype"], "BG_SELECTION_ONLY_LIVE_PROTOTYPE_VERDICT")
    arb = verdict(data["final_arbiter"], "BG_SELECTION_ONLY_FINAL_ARBITER_VERDICT")
    steering = verdict(data["steering_readiness"], "BG_SELECTION_ONLY_TO_STEERING_READINESS_VERDICT")
    domain = verdict(data["domain_coding"], "BG_SELECTION_ONLY_DOMAIN_CODING_VERDICT")
    failure = verdict(data["failure_analysis"], "BG_SELECTION_ONLY_FAILURE_ANALYSIS_VERDICT")
    if steering == "READY_FOR_SELECTION_PLUS_STEERING_TEST":
        return "SELECTION_ONLY_READY"
    if live == "SURVIVAL_POSITIVE_FINAL_SELECTION_WEAK" or arb == "FINAL_SELECTION_WEAK" or failure == "FINAL_ARBITER_BLOCKER":
        return "SURVIVAL_READY_FINAL_ARBITER_WEAK"
    if live == "SURVIVAL_WEAK":
        return "SURVIVAL_WEAK"
    if live == "UNSTABLE":
        return "BRANCH_GENERATOR_WEAK"
    if domain in {"REASONING_SCIENCE_ONLY", "DOMAIN_MIXED", "DATA_LIMITED"}:
        return "DOMAIN_LIMITED"
    if live in {"DATA_LIMITED", "BLOCKED"}:
        return "INSUFFICIENT"
    return "NOT_READY"


def top_lines(data: dict[str, dict[str, Any]], status: str) -> list[str]:
    return [
        f"BG_SELECTION_ONLY_INVENTORY_VERDICT = {verdict(data['inventory'], 'BG_SELECTION_ONLY_INVENTORY_VERDICT')}",
        f"BG_SELECTION_ONLY_TASK_SUITE_VERDICT = {verdict(data['task_suite'], 'BG_SELECTION_ONLY_TASK_SUITE_VERDICT')}",
        f"BG_SELECTION_ONLY_POLICY_RUNNER_VERDICT = {verdict(data['policy_runner'], 'BG_SELECTION_ONLY_POLICY_RUNNER_VERDICT')}",
        f"BG_SELECTION_ONLY_CACHED_REPRODUCTION_VERDICT = {verdict(data['cached_reproduction'], 'BG_SELECTION_ONLY_CACHED_REPRODUCTION_VERDICT')}",
        f"BG_SELECTION_ONLY_LIVE_PROTOTYPE_VERDICT = {verdict(data['live_prototype'], 'BG_SELECTION_ONLY_LIVE_PROTOTYPE_VERDICT')}",
        f"BG_SELECTION_ONLY_FINAL_ARBITER_VERDICT = {verdict(data['final_arbiter'], 'BG_SELECTION_ONLY_FINAL_ARBITER_VERDICT')}",
        f"BG_SELECTION_ONLY_BASELINE_COMPARISON_VERDICT = {verdict(data['baseline_comparison'], 'BG_SELECTION_ONLY_BASELINE_COMPARISON_VERDICT')}",
        f"BG_SELECTION_ONLY_DOMAIN_CODING_VERDICT = {verdict(data['domain_coding'], 'BG_SELECTION_ONLY_DOMAIN_CODING_VERDICT')}",
        f"BG_SELECTION_ONLY_FAILURE_ANALYSIS_VERDICT = {verdict(data['failure_analysis'], 'BG_SELECTION_ONLY_FAILURE_ANALYSIS_VERDICT')}",
        f"BG_SELECTION_ONLY_TO_STEERING_READINESS_VERDICT = {verdict(data['steering_readiness'], 'BG_SELECTION_ONLY_TO_STEERING_READINESS_VERDICT')}",
        f"SELECTION_ONLY_PHASE2_PROTOTYPE_STATUS = {status}",
    ]


def recommendation_for_status(status: str) -> str:
    if status == "SELECTION_ONLY_READY":
        return "Use this result as the locked baseline for a Phase 2b selection-only vs selection plus trained steering corridor test."
    if status == "SURVIVAL_READY_FINAL_ARBITER_WEAK":
        return "Train or evaluate a stronger final arbiter among top4 survivors before steering."
    if status == "SURVIVAL_WEAK":
        return "Return to the survival policy before Phase 2b."
    if status == "BRANCH_GENERATOR_WEAK":
        return "Improve branch generation stability and useful alternative rate before Phase 2b."
    if status == "DOMAIN_LIMITED":
        return "Collect more domain-specific selection-only data before Phase 2b."
    return "Do not proceed to steering comparison from this prototype."


def docs_lines(data: dict[str, dict[str, Any]], status: str) -> list[str]:
    live = data.get("live_prototype", {})
    cached = data.get("cached_reproduction", {})
    arb = data.get("final_arbiter", {})
    domain = data.get("domain_coding", {})
    failure = data.get("failure_analysis", {})
    rec = recommendation_for_status(status)
    return [
        "# Selection-Only Phase 2 Prototype V1",
        "",
        "This Phase 2a prototype tested branch generation plus fixed-composite top4 branch survival plus final selection. It did not test action steering, train Ouro, modify checkpoints, update tokenizer files, update existing taps, or change production routing.",
        "",
        "## Verdicts",
        "",
        *top_lines(data, status),
        "",
        "## Fixed-Composite Context",
        "",
        "The prototype used the selected fixed_composite_conservative_top4 operating point from the fixed-composite branch survival policy v1 artifacts.",
        "",
        "## BGV1 Branch Generator Context",
        "",
        "Completed cached BGV1/v4 hook-intervention branch groups were used in counterfactual replay mode. Layerwise lineage is HOOK_LAYERWISE_APPROX; true fork/carry is not claimed.",
        "",
        "## Task Suite",
        "",
        f"- task-suite verdict: `{verdict(data['task_suite'], 'BG_SELECTION_ONLY_TASK_SUITE_VERDICT')}`",
        f"- counts: `{(data.get('task_suite') or {}).get('counts')}`",
        "",
        "## Cached Reproduction",
        "",
        f"- reproduction verdict: `{verdict(cached, 'BG_SELECTION_ONLY_CACHED_REPRODUCTION_VERDICT')}`",
        f"- selected metrics: `{compact_metrics((cached.get('metrics_by_policy') or {}).get(POLICY_NAME, {}))}`",
        "",
        "## Selection-Only Results",
        "",
        f"- live/counterfactual verdict: `{verdict(live, 'BG_SELECTION_ONLY_LIVE_PROTOTYPE_VERDICT')}`",
        f"- selected metrics: `{compact_metrics((live.get('metrics_by_policy') or {}).get(POLICY_NAME, {}))}`",
        "",
        "## Final Arbiter",
        "",
        f"- arbiter verdict: `{verdict(arb, 'BG_SELECTION_ONLY_FINAL_ARBITER_VERDICT')}`",
        f"- best policy: `{arb.get('best_policy')}`",
        "",
        "## Baselines And Domain/Coding",
        "",
        f"- baseline verdict: `{verdict(data['baseline_comparison'], 'BG_SELECTION_ONLY_BASELINE_COMPARISON_VERDICT')}`",
        f"- domain/coding verdict: `{verdict(domain, 'BG_SELECTION_ONLY_DOMAIN_CODING_VERDICT')}`",
        f"- coding status: `{domain.get('CODING_STATUS')}`",
        "",
        "## Failure Analysis",
        "",
        f"- failure verdict: `{verdict(failure, 'BG_SELECTION_ONLY_FAILURE_ANALYSIS_VERDICT')}`",
        f"- failure summary: `{failure.get('failure_summary')}`",
        "",
        "## Steering Readiness",
        "",
        f"- steering-readiness verdict: `{verdict(data['steering_readiness'], 'BG_SELECTION_ONLY_TO_STEERING_READINESS_VERDICT')}`",
        f"- recommendation: {rec}",
        "",
        "## Explicit Non-Claims",
        "",
        "- No action steering was tested.",
        "- No steering-vector intervention is a tested condition.",
        "- No production routing change was made.",
        "- No compute savings are claimed.",
        "- No true branch-batch fork/carry is claimed.",
        "",
    ]


def append_once(path: Path, section_lines: Sequence[str], marker: str) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    body = "\n".join(section_lines).strip() + "\n"
    path.write_text(text.rstrip() + "\n\n" + body, encoding="utf-8")


def update_docs(data: dict[str, dict[str, Any]], status: str) -> None:
    marker = "## Selection-only Phase 2 prototype v1 (2026-05-18)"
    status_line = f"SELECTION_ONLY_PHASE2_PROTOTYPE_STATUS = {status}"
    section = [
        marker,
        "",
        status_line,
        "",
        f"- cached reproduction: `{verdict(data['cached_reproduction'], 'BG_SELECTION_ONLY_CACHED_REPRODUCTION_VERDICT')}`",
        f"- live/counterfactual prototype: `{verdict(data['live_prototype'], 'BG_SELECTION_ONLY_LIVE_PROTOTYPE_VERDICT')}`",
        f"- final arbiter: `{verdict(data['final_arbiter'], 'BG_SELECTION_ONLY_FINAL_ARBITER_VERDICT')}`",
        f"- steering readiness: `{verdict(data['steering_readiness'], 'BG_SELECTION_ONLY_TO_STEERING_READINESS_VERDICT')}`",
        f"- recommendation: {recommendation_for_status(status)}",
        "- no action steering was tested; no production routing changed.",
        "",
    ]
    write_md(DOC_MD, docs_lines(data, status))
    for target in DOC_TARGETS:
        append_once(target, section, marker)


def run_synthesis() -> int:
    ensure_out_root()
    started = time.time()
    required = [
        (INVENTORY_JSON, run_inventory),
        (TASK_SUITE_JSON, run_task_suite),
        (POLICY_REPORT_JSON, run_policy_runner),
        (CACHED_JSON, run_cached_reproduction),
        (LIVE_JSON, run_live_prototype),
        (ARBITER_JSON, run_final_arbiter),
        (BASELINE_JSON, run_baseline_comparison),
        (DOMAIN_JSON, run_domain_coding),
        (FAILURE_JSON, run_failures),
        (STEERING_JSON, run_steering_readiness),
    ]
    for path, fn in required:
        if not path.exists():
            fn()
    data = load_stage_payloads()
    status = final_status(data)
    rec = recommendation_for_status(status)
    payload = {
        "top_lines": top_lines(data, status),
        "stage_payloads": data,
        "SELECTION_ONLY_PHASE2_PROTOTYPE_STATUS": status,
        "recommended_next": rec,
        "files_created": {
            "output_root": rel(OUT_ROOT),
            "prototype_artifact": rel(POLICY_OUTPUTS_PT),
            "summary": rel(SUMMARY_JSON),
            "analysis": rel(ANALYSIS_JSON),
            "doc": rel(DOC_MD),
        },
        "commands_run": [f"venv/bin/python -u utilities/tests/manual/{name}" for name in SCRIPT_NAMES],
        "blockers": [
            rec
        ] if status != "SELECTION_ONLY_READY" else [],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(SUMMARY_JSON, payload)
    write_json(ANALYSIS_JSON, payload)
    lines = [
        "# Selection-Only Phase 2 Prototype V1 Summary",
        "",
        *top_lines(data, status),
        "",
        "## Decision",
        "",
        rec,
        "",
        "## Files Created",
        "",
    ]
    lines.extend(f"- `{path}`" for path in payload["files_created"].values())
    lines.extend(["", "## Commands Run", ""])
    lines.extend(f"- `{cmd}`" for cmd in payload["commands_run"])
    if payload["blockers"]:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {item}" for item in payload["blockers"])
    write_md(SUMMARY_MD, lines)
    write_md(ANALYSIS_MD, docs_lines(data, status))
    update_docs(data, status)
    print(f"SELECTION_ONLY_PHASE2_PROTOTYPE_STATUS = {status}", flush=True)
    return 0
