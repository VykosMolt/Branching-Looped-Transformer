"""Shared helpers for hidden-origin Branch Generator v1 diagnostics.

This module is local to the manual v1 branch-generator experiment. It does not
train Ouro, update checkpoints/tokenizers/registries, run wrapper/local-agent
code, or import the actual Hunter-Seeker implementation.
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Sequence

import torch

from bg_hidden_origin_quota_v4_common import *  # noqa: F403
from bg_hidden_origin_quota_v4_common import (
    ALPHA_VALUE,
    CONFIGS,
    HIDDEN_DIM,
    PRIMARY_ALPHA_CAP,
    PRIMARY_DOMAINS,
    PROBE_ROOT,
    V1_ROOT,
    V2_ROOT,
    V3_ROOT,
    candidate_pair_stats,
    compact_direction_entry,
    compact_v4_row,
    config_dim,
    config_layer,
    config_vector_from_row,
    deterministic_reward,
    direction_entry,
    finite_float,
    group_is_behaviorally_diverse_v2,
    group_is_reward_diverse_v2,
    group_rows,
    load_json,
    load_pt,
    md_table,
    normalize_branch_point,
    primary_domain_row,
    rate,
    rel,
    row_reward,
    same_prefix_hidden_origin_row,
    sampled_reward,
    stable_v2_row,
    tensor_stats,
    write_csv,
    write_json,
    write_md,
)


BGV1_ROOT = PROBE_ROOT / "bg_hidden_origin_branch_generator_v1_2026-05-18"

AUDIT_PLAN_JSON = BGV1_ROOT / "audit_plan.json"
AUDIT_PLAN_MD = BGV1_ROOT / "audit_plan.md"
TASK_PLAN_CSV = BGV1_ROOT / "task_plan.csv"

TRUE_FORK_JSON = BGV1_ROOT / "true_fork_carry_probe.json"
TRUE_FORK_MD = BGV1_ROOT / "true_fork_carry_probe.md"
TRUE_FORK_CSV = BGV1_ROOT / "true_fork_carry_rows.csv"

RICH_SCHEMA_JSON = BGV1_ROOT / "rich_outcome_schema.json"
RICH_SCHEMA_MD = BGV1_ROOT / "rich_outcome_schema.md"

BASIS_BANK_PT = BGV1_ROOT / "basis_bank_v1.pt"
BASIS_BANK_JSON = BGV1_ROOT / "basis_bank_v1.json"
BASIS_BANK_MD = BGV1_ROOT / "basis_bank_v1.md"

PROPOSER_PT = BGV1_ROOT / "branch_generator_proposer_v1.pt"
PROPOSER_JSON = BGV1_ROOT / "proposer_training_log.json"
PROPOSER_MD = BGV1_ROOT / "proposer_training_report.md"

BLACKBOX_JSON = BGV1_ROOT / "blackbox_search.json"
BLACKBOX_MD = BGV1_ROOT / "blackbox_search_report.md"
RECIPE_ENGRAMS_JSONL = BGV1_ROOT / "recipe_engrams_v1.jsonl"
BEST_SCHEDULE_JSON = BGV1_ROOT / "best_branch_generator_schedule.json"

BRANCHES_PT = BGV1_ROOT / "branch_generator_v1_branches.pt"
BRANCHES_PARTIAL_PT = BGV1_ROOT / "branch_generator_v1_branches.partial.pt"
BRANCHES_JSON = BGV1_ROOT / "branch_generator_v1_branches.json"
BRANCHES_CSV = BGV1_ROOT / "branch_generator_v1_branches.csv"
PROGRESS_JSONL = BGV1_ROOT / "branch_generation_progress.jsonl"
GEN_STATE_JSON = BGV1_ROOT / "branch_generation_state.json"
GEN_REPORT_JSON = BGV1_ROOT / "branch_generation_report.json"
GEN_REPORT_MD = BGV1_ROOT / "branch_generation_report.md"

SELECTOR_DATASET_PT = BGV1_ROOT / "selector_dataset.pt"
SELECTOR_DATASET_JSON = BGV1_ROOT / "selector_dataset.json"
SELECTOR_DATASET_MD = BGV1_ROOT / "selector_dataset.md"
SELECTOR_HEADS_PT = BGV1_ROOT / "selector_heads.pt"
SELECTOR_TRAINING_JSON = BGV1_ROOT / "selector_training_log.json"
SELECTOR_TRAINING_MD = BGV1_ROOT / "selector_training_report.md"
SELECTOR_EVAL_JSON = BGV1_ROOT / "selector_eval.json"
SELECTOR_EVAL_MD = BGV1_ROOT / "selector_eval.md"
SELECTOR_EVAL_CSV = BGV1_ROOT / "selector_eval_rows.csv"
OLD_CONTEXT_JSON = BGV1_ROOT / "old_context_replay.json"
OLD_CONTEXT_MD = BGV1_ROOT / "old_context_replay.md"
OLD_CONTEXT_CSV = BGV1_ROOT / "old_context_replay_rows.csv"
GEOMETRY_JSON = BGV1_ROOT / "geometry_analysis.json"
GEOMETRY_MD = BGV1_ROOT / "geometry_analysis.md"
SUMMARY_JSON = BGV1_ROOT / "summary.json"
SUMMARY_MD = BGV1_ROOT / "summary.md"
ANALYSIS_JSON = BGV1_ROOT / "analysis.json"
ANALYSIS_MD = BGV1_ROOT / "analysis.md"
DOC_MD = Path("docs/evaluator/bg_hidden_origin_branch_generator_v1.md")

BRANCH_GENERATOR_PT = BGV1_ROOT / "branch_generator_v1.pt"
HIDDEN_ORIGIN_TAP_HEADS_GENERATOR_V1_PT = BGV1_ROOT / "hidden_origin_tap_heads_generator_v1.pt"

PRIMARY_MINIMUMS_V1 = {
    "train": {"task_ids": 24, "behaviorally_diverse_groups": 60, "non_tie_pairs": 250},
    "val": {"task_ids": 6, "behaviorally_diverse_groups": 15, "non_tie_pairs": 60},
    "heldout": {"task_ids": 8, "behaviorally_diverse_groups": 20, "non_tie_pairs": 120},
}
PREFERRED_MINIMUMS_V1 = {
    "train": {"task_ids": 40, "behaviorally_diverse_groups": 100, "non_tie_pairs": 400},
    "val": {"task_ids": 8, "behaviorally_diverse_groups": 25, "non_tie_pairs": 100},
    "heldout": {"task_ids": 10, "behaviorally_diverse_groups": 30, "non_tie_pairs": 180},
}
SPLIT_ROW_CAPS_V1 = {"train": 2400, "val": 1000, "heldout": 1600}
TOTAL_ROW_CAP_V1 = 5000

BGV1_COMMAND_SCRIPTS = [
    "bg_hidden_origin_branch_generator_v1_audit_plan.py",
    "bg_hidden_origin_true_fork_carry_probe_v1.py",
    "bg_hidden_origin_rich_outcome_schema_v1.py",
    "build_bg_branch_generator_v1_basis_bank.py",
    "train_bg_branch_generator_proposer_v1.py",
    "run_bg_branch_generator_blackbox_search_v1.py",
    "generate_bg_branch_generator_v1_quota_branches.py",
    "analyze_bg_branch_generator_v1_diversity.py",
    "build_bg_branch_generator_v1_selector_dataset.py",
    "train_bg_branch_generator_v1_selectors.py",
    "evaluate_bg_branch_generator_v1_selectors.py",
    "evaluate_bg_branch_generator_v1_old_context_replay.py",
    "analyze_bg_branch_generator_v1_geometry.py",
    "analyze_bg_branch_generator_v1_experiment.py",
]


def ensure_bgv1_root() -> None:
    BGV1_ROOT.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=json_default_compact) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def json_default_compact(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return tensor_stats(value)
    if isinstance(value, Path):
        return rel(value)
    try:
        if math.isfinite(float(value)):
            return float(value)
    except Exception:
        pass
    return str(value)


def load_audit_plan() -> dict[str, Any]:
    return load_json(AUDIT_PLAN_JSON, {}) or {}


def load_basis_bank() -> dict[str, Any]:
    return load_pt(BASIS_BANK_PT, {"directions": [], "directions_by_layer": {}}) or {"directions": [], "directions_by_layer": {}}


def load_best_schedule() -> dict[str, Any]:
    return load_json(BEST_SCHEDULE_JSON, {}) or {}


def primary_safe_generator_row(row: dict[str, Any]) -> bool:
    method = str(row.get("branch_method") or row.get("method") or "")
    same_prefix = method in {"", "hook_intervention_per_branch", "hook_hidden_origin_branch", "true_fork_carry"}
    return (
        str(row.get("split") or "") in {"train", "val", "heldout"}
        and primary_domain_row(row)
        and same_prefix
        and normalize_branch_point(row) in {"L24", "L36"}
        and finite_float(row.get("alpha"), 999.0) <= PRIMARY_ALPHA_CAP
        and bool(row.get("safety_envelope", True))
        and str(row.get("label_source") or "deterministic") in {"deterministic", ""}
        and stable_v2_row(row)
    )


def diagnostic_generator_row(row: dict[str, Any]) -> bool:
    return (
        str(row.get("split") or "") in {"train", "val", "heldout"}
        and primary_domain_row(row)
        and stable_v2_row(row)
        and (abs(finite_float(row.get("alpha"), -1.0) - 0.02) <= 1e-6 or normalize_branch_point(row) == "L47")
    )


def compact_generator_row(row: dict[str, Any]) -> dict[str, Any]:
    out = compact_v4_row(row)
    for key in (
        "generator_method",
        "branch_generator_method",
        "low_rank_coeff_summary",
        "v4_tap_score",
        "option_logit_margin",
        "option_entropy",
        "branch_distance_from_clean",
        "branch_logit_kl_from_clean",
        "off_manifold_warning",
    ):
        if key in row:
            out[key] = row.get(key)
    return out


def split_quota_stats_v1(rows: Sequence[dict[str, Any]], *, label_source: str = "deterministic", row_filter: Callable[[dict[str, Any]], bool] | None = None) -> dict[str, Any]:
    selected = [row for row in rows if (row_filter or primary_safe_generator_row)(row)]
    out: dict[str, Any] = {}
    for split in ("train", "val", "heldout"):
        split_rows = [row for row in selected if str(row.get("split")) == split]
        groups = {gid: vals for gid, vals in group_rows(split_rows).items() if len(vals) >= 2}
        behavior = {gid for gid, vals in groups.items() if group_is_behaviorally_diverse_v2(vals, label_source)}
        reward = {gid for gid, vals in groups.items() if group_is_reward_diverse_v2(vals, label_source)}
        pair_stats = candidate_pair_stats(groups, label_source)
        task_ids_with_non_tie: set[str] = set()
        for vals in groups.values():
            if candidate_pair_stats({"g": vals}, label_source)["non_tie_pairs"] > 0:
                task_ids_with_non_tie.add(str(vals[0].get("task_id")))
        split_all_rows = [row for row in rows if str(row.get("split")) == split]
        stats = {
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
            "stability_rate": len(split_rows) / max(len(split_all_rows), 1),
            "behaviorally_diverse_groups_per_100_rows": 100.0 * len(behavior) / max(len(split_rows), 1),
            "non_tie_pairs_per_100_rows": 100.0 * pair_stats["non_tie_pairs"] / max(len(split_rows), 1),
            "quota_minimums": PRIMARY_MINIMUMS_V1[split],
        }
        stats["minimum_met"] = quota_met_for_split_v1(stats, split)
        out[split] = stats
    out["all_minimums_met"] = all(out[split]["minimum_met"] for split in ("train", "val", "heldout"))
    return out


def quota_met_for_split_v1(stats: dict[str, Any], split: str, preferred: bool = False) -> bool:
    minimums = PREFERRED_MINIMUMS_V1 if preferred else PRIMARY_MINIMUMS_V1
    task_count = int(stats.get("task_ids_with_non_tie_pairs") or stats.get("task_ids") or 0)
    return (
        task_count >= int(minimums[split]["task_ids"])
        and int(stats.get("behaviorally_diverse_groups") or 0) >= int(minimums[split]["behaviorally_diverse_groups"])
        and int(stats.get("non_tie_pairs") or 0) >= int(minimums[split]["non_tie_pairs"])
    )


def split_deficit_score_v1(stats: dict[str, Any], split: str) -> float:
    current = stats.get(split, {})
    m = PRIMARY_MINIMUMS_V1[split]
    task_count = max(int(current.get("task_ids_with_non_tie_pairs") or 0), int(current.get("task_ids") or 0))
    return float(
        max(0.0, (m["task_ids"] - task_count) / max(m["task_ids"], 1))
        + max(0.0, (m["behaviorally_diverse_groups"] - int(current.get("behaviorally_diverse_groups") or 0)) / max(m["behaviorally_diverse_groups"], 1))
        + max(0.0, (m["non_tie_pairs"] - int(current.get("non_tie_pairs") or 0)) / max(m["non_tie_pairs"], 1))
    )


def load_generator_rows() -> list[dict[str, Any]]:
    payload = load_pt(BRANCHES_PT, None)
    if not payload:
        payload = load_pt(BRANCHES_PARTIAL_PT, {})
    rows: list[dict[str, Any]] = []
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
        if "label_source" not in norm:
            norm["label_source"] = "deterministic"
        rows.append(norm)
    return rows


def candidate_group_summary(groups: dict[str, Sequence[dict[str, Any]]]) -> dict[str, Any]:
    pair_stats = candidate_pair_stats(groups)
    behavior = sum(1 for vals in groups.values() if group_is_behaviorally_diverse_v2(vals))
    reward = sum(1 for vals in groups.values() if group_is_reward_diverse_v2(vals))
    rows = sum(len(vals) for vals in groups.values())
    stable_rows = sum(1 for vals in groups.values() for row in vals if stable_v2_row(row))
    return {
        "rows": rows,
        "stable_rows": stable_rows,
        "groups": len(groups),
        "behaviorally_diverse_groups": behavior,
        "reward_diverse_groups": reward,
        "non_tie_pairs": pair_stats["non_tie_pairs"],
        "candidate_pairs": pair_stats["candidate_pairs"],
        "tie_pairs": pair_stats["tie_pairs"],
        "tie_rate": pair_stats["tie_rate"],
        "stable_rate": stable_rows / max(rows, 1),
        "parse_rate": sum(1 for vals in groups.values() for row in vals if row.get("parse_success")) / max(rows, 1),
        "behaviorally_diverse_groups_per_100_rows": 100.0 * behavior / max(rows, 1),
        "non_tie_pairs_per_100_rows": 100.0 * pair_stats["non_tie_pairs"] / max(rows, 1),
    }


def stats_by_factor(rows: Sequence[dict[str, Any]], factor: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    selected = [row for row in rows if primary_safe_generator_row(row)]
    for value in sorted({str(row.get(factor)) for row in selected}):
        groups = {gid: vals for gid, vals in group_rows([row for row in selected if str(row.get(factor)) == value]).items() if len(vals) >= 2}
        out[value] = candidate_group_summary(groups)
    return out


def verdict_for_generation_v1(rows: list[dict[str, Any]], errors: list[dict[str, Any]]) -> str:
    stats = split_quota_stats_v1(rows)
    if not rows and errors:
        return "BLOCKED"
    if stats.get("all_minimums_met"):
        return "QUOTAS_MET"
    heldout_met = bool(stats.get("heldout", {}).get("minimum_met"))
    train_val_met = bool(stats.get("train", {}).get("minimum_met")) and bool(stats.get("val", {}).get("minimum_met"))
    if heldout_met and not train_val_met:
        return "HELDOUT_QUOTA_MET_ONLY"
    if train_val_met and not heldout_met:
        return "TRAIN_VAL_QUOTA_MET_ONLY"
    heldout = stats.get("heldout", {})
    if int(heldout.get("behaviorally_diverse_groups") or 0) > 8 or float(heldout.get("behaviorally_diverse_groups_per_100_rows") or 0.0) >= 3.0:
        return "DIVERSITY_IMPROVED"
    stable_rates = [float(stats.get(split, {}).get("stability_rate") or 0.0) for split in ("train", "val", "heldout")]
    if rows and min(stable_rates or [1.0]) < 0.50:
        return "UNSTABLE"
    return "LOW_DIVERSITY"


def verdict_for_selector_dataset_v1(stats: dict[str, Any], primary_pairs: int) -> str:
    heldout = stats.get("heldout", {})
    train = stats.get("train", {})
    val = stats.get("val", {})
    if primary_pairs <= 0:
        return "BLOCKED"
    heldout_ready = (
        int(heldout.get("task_ids_with_non_tie_pairs") or 0) >= 8
        and int(heldout.get("behaviorally_diverse_groups") or 0) >= 20
        and int(heldout.get("non_tie_pairs") or 0) >= 120
    )
    train_val_ready = int(train.get("non_tie_pairs") or 0) >= 250 and int(val.get("non_tie_pairs") or 0) >= 60
    if heldout_ready and train_val_ready:
        return "READY"
    if heldout_ready:
        return "HELDOUT_READY_TRAIN_WEAK"
    if int(heldout.get("behaviorally_diverse_groups") or 0) >= 16 and int(heldout.get("non_tie_pairs") or 0) >= 90:
        return "SMALL_BUT_USABLE"
    return "STILL_DATA_LIMITED"


def group_reward_variance(vals: Sequence[dict[str, Any]]) -> float:
    rewards = [row_reward(row) for row in vals]
    if not rewards:
        return 0.0
    avg = mean(rewards)
    return float(sum((x - avg) ** 2 for x in rewards) / max(len(rewards), 1))


def write_stage_verdict(path_json: Path, path_md: Path, verdict_key: str, verdict: str, payload: dict[str, Any], title: str) -> None:
    ensure_bgv1_root()
    write_json(path_json, {verdict_key: verdict, "verdict": verdict, **payload})
    lines = [f"# {title}", "", f"{verdict_key} = {verdict}", ""]
    for key, value in payload.items():
        if key in {"rows", "records"}:
            continue
        lines.append(f"- {key}: `{value}`")
    write_md(path_md, lines)
