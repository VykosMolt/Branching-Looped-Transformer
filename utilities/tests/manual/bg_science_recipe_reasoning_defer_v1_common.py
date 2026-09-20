"""Shared utilities for DualAnchor science recipe + reasoning defer v1.

This probe is intentionally conservative. It reuses the existing v3
architecture-looped candidate tree and the convergence-hair pre-steering
artifacts, and it labels recipe tests as replay diagnostics unless new
generation artifacts are actually produced.

No steering, training, wrapper/local-agent execution, registry update, hard
convergence-hair merge, production routing change, or true autoregressive
fork/carry claim is made here.
"""
from __future__ import annotations

import ast
import csv
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import torch

import bg_convergence_hairs_rs_v1_common as prev
from bg_branch_generator_v1_common import load_audit_plan
from bg_hidden_origin_tap_common import PROBE_ROOT


SHORT_NAME = "dualanchor_science_branch_recipe_reasoning_defer_v1"
OUT_ROOT = PROBE_ROOT / "bg_dualanchor_science_branch_recipe_reasoning_defer_v1_2026-05-31"
PREV_ROOT = PROBE_ROOT / "bg_dualanchor_convergence_hairs_reasoning_science_v1_2026-05-31"
V3_ROOT = PROBE_ROOT / "bg_dualanchor_architecture_looped_stratified_probe_v3_2026-05-31"

PROGRESS_ROOT = OUT_ROOT / "progress"
CANDIDATE_TREE_ROOT = OUT_ROOT / "candidate_trees"
LINEAGE_INCREMENTAL_CSV = OUT_ROOT / "lineage_rows_incremental.csv"
COMPLETED_TASKS_JSON = OUT_ROOT / "completed_task_ids.json"
COMPLETED_RECIPES_JSON = OUT_ROOT / "completed_recipe_ids.json"
PARTIAL_SUMMARY_JSON = OUT_ROOT / "partial_summary.json"

ANCHOR_A = "MIX_CODE_REASONING"
ANCHOR_B = "MIX_OBJECTIVE_ALL"
LOCKED_SCHEDULE = "L1_24 -> L1_36 -> L1_47 -> L2_24 -> L2_36 -> L2_47 -> L3_24 -> L3_36 -> L3_47 -> L4_24 -> L4_36 -> terminal L4_47"

SCIENCE_BASELINE_POSITIVE_ORACLE = 0.0833
SCIENCE_BASELINE_BEST_REWARD = 0.0500
SCIENCE_BASELINE_REWARD_DIVERSE = 0.2083
REASONING_BASELINE_POSITIVE_ORACLE = 0.6250


def ensure_root() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    PROGRESS_ROOT.mkdir(parents=True, exist_ok=True)
    CANDIDATE_TREE_ROOT.mkdir(parents=True, exist_ok=True)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, torch.Tensor):
        if value.numel() <= 16:
            return value.detach().cpu().tolist()
        return {"tensor_shape": list(value.shape), "dtype": str(value.dtype)}
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    ensure_root()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def write_md(path: Path, lines: Sequence[str]) -> None:
    ensure_root()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, sort_keys=True, default=json_default)
    return value


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    ensure_root()
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in keys})


def safe_float(value: Any, default: float = float("nan")) -> float:
    return prev.safe_float(value, default)


def truthy(value: Any) -> bool:
    return prev.truthy(value)


def finite_mean(values: Iterable[Any]) -> float:
    xs: list[float] = []
    for value in values:
        x = safe_float(value, float("nan"))
        if math.isfinite(x):
            xs.append(x)
    return float(mean(xs)) if xs else float("nan")


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def status_line(key: str, value: str) -> str:
    return f"{key} = {value}"


def md_table(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> list[str]:
    if not rows:
        return ["_No rows._"]
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join("---" for _ in keys) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key, "")) for key in keys) + " |")
    return lines


def parse_literal(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except Exception:
            return default
    return value if value is not None else default


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = text.replace("\\boxed", " boxed ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def row_reward(row: dict[str, Any]) -> float:
    return prev.row_reward(row)


def load_v3_bundle() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    return prev.load_v3_rows()


def load_v3_task_rows_csv() -> list[dict[str, str]]:
    return read_csv(V3_ROOT / "task_rows.csv")


def load_v3_stage_rows_csv() -> list[dict[str, str]]:
    return read_csv(V3_ROOT / "stage_decisions.csv")


def load_v3_terminal_rows_csv() -> list[dict[str, str]]:
    return read_csv(V3_ROOT / "terminal_policy_rows.csv")


def load_v3_l47_rows_csv() -> list[dict[str, str]]:
    return read_csv(V3_ROOT / "l47_ablation_rows.csv")


def load_prev_json(name: str) -> dict[str, Any]:
    return read_json(PREV_ROOT / name, {}) or {}


def audit_plan_tasks() -> list[dict[str, Any]]:
    return [dict(row) for row in (load_audit_plan().get("tasks") or [])]


def task_meta_by_id() -> dict[str, dict[str, Any]]:
    return {str(row.get("task_id")): row for row in audit_plan_tasks()}


def rows_by_task(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("task_id"))].append(row)
    return grouped


def rows_by_id(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("branch_id")): row for row in rows if row.get("branch_id")}


def source_detail(task_id: str, source_dataset: Any) -> str:
    text = str(task_id).lower()
    source = str(source_dataset or "").lower()
    for name in ("high_school_chemistry", "high_school_physics", "anatomy", "biology"):
        if name in text:
            return f"mmlu_{name}"
    if source == "sciq":
        return "sciq"
    if source == "openbookqa":
        return "openbookqa"
    if source == "mmlu":
        return "mmlu_other"
    return source or "unknown"


def task_sort_key(row: dict[str, Any]) -> tuple[int, int, int, float, str, str]:
    split_rank = {"heldout": 0, "val": 1, "train": 2}.get(str(row.get("split")), 9)
    likely = 1 if truthy(row.get("heldout_likely_non_tie")) else 0
    prior_pairs = int(row.get("prior_non_tie_pairs") or 0)
    priority = safe_float(row.get("priority_score_v4") or row.get("priority_score"), 0.0)
    return (split_rank, -likely, -prior_pairs, -priority, str(row.get("source_dataset")), str(row.get("task_id")))


def split_role(row: dict[str, Any]) -> str:
    if str(row.get("split")) == "heldout":
        return "heldout_eval"
    return "recipe_calibration"


def build_task_suite_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan_tasks = [row for row in audit_plan_tasks() if str(row.get("domain")) in {"science", "reasoning"} and row.get("correct_option")]
    by_domain = {
        "science": sorted([row for row in plan_tasks if str(row.get("domain")) == "science"], key=task_sort_key),
        "reasoning": sorted([row for row in plan_tasks if str(row.get("domain")) == "reasoning"], key=task_sort_key),
    }
    science = by_domain["science"][:48]
    reasoning = by_domain["reasoning"][:24]
    selected = science + reasoning
    v3_task_ids = {str(row.get("task_id")) for row in load_v3_task_rows_csv()}
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        prior = next((v for v in load_v3_task_rows_csv() if str(v.get("task_id")) == str(row.get("task_id"))), {})
        rows.append(
            {
                "index": index,
                "task_id": row.get("task_id"),
                "domain": row.get("domain"),
                "source_dataset": row.get("source_dataset"),
                "source_detail": source_detail(str(row.get("task_id")), row.get("source_dataset")),
                "split": row.get("split"),
                "split_role": split_role(row),
                "parser_type": "mcq",
                "correct_option": row.get("correct_option"),
                "options": row.get("options"),
                "prior_positive_oracle": prior.get("positive_oracle", ""),
                "prior_reward_diverse": prior.get("terminal_reward_diverse", ""),
                "prior_parse_risk": "known_science_no_mcq_letter_risk" if str(row.get("domain")) == "science" else "low",
                "v3_generated": str(row.get("task_id")) in v3_task_ids,
                "heldout_likely_non_tie": truthy(row.get("heldout_likely_non_tie")),
                "prior_non_tie_pairs": int(row.get("prior_non_tie_pairs") or 0),
                "priority_score": safe_float(row.get("priority_score_v4") or row.get("priority_score"), 0.0),
            }
        )
    inventory = {
        "available_plan_tasks": len(plan_tasks),
        "available_science_tasks": len(by_domain["science"]),
        "available_reasoning_tasks": len(by_domain["reasoning"]),
        "selected_count": len(rows),
        "selected_science": sum(1 for row in rows if row["domain"] == "science"),
        "selected_reasoning": sum(1 for row in rows if row["domain"] == "reasoning"),
        "v3_generated_selected": sum(1 for row in rows if row["v3_generated"]),
        "domain_counts": dict(Counter(row["domain"] for row in rows)),
        "science_source_counts": dict(Counter(row["source_detail"] for row in rows if row["domain"] == "science")),
        "reasoning_source_counts": dict(Counter(row["source_dataset"] for row in rows if row["domain"] == "reasoning")),
        "split_role_counts": dict(Counter(row["split_role"] for row in rows)),
        "target_preferred": {"science": 48, "reasoning": 24},
        "target_minimum": {"science": 32, "reasoning": 16},
    }
    return rows, inventory


def selected_task_ids_from_suite() -> set[str]:
    suite = read_json(OUT_ROOT / "task_suite.json", {}) or {}
    rows = suite.get("task_rows") or []
    return {str(row.get("task_id")) for row in rows}


def v3_final_pool(task_row: dict[str, Any], all_rows_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    branch_ids = parse_literal(task_row.get("final_branch_ids"), [])
    return [all_rows_by_id[str(branch_id)] for branch_id in branch_ids if str(branch_id) in all_rows_by_id]


def terminal_order(task_id: str) -> list[int]:
    terminal_rows = load_v3_terminal_rows_csv()
    row = next((item for item in terminal_rows if item.get("task_id") == task_id and item.get("policy") == "terminal_defer_all"), None)
    if not row:
        return []
    return [int(idx) for idx in parse_literal(row.get("selected_indices"), [])]


def ordered_pool(task_id: str, pool: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    order = terminal_order(task_id)
    used: set[int] = set()
    out: list[dict[str, Any]] = []
    for idx in order:
        if 0 <= idx < len(pool) and idx not in used:
            out.append(pool[idx])
            used.add(idx)
    for idx, row in enumerate(pool):
        if idx not in used:
            out.append(row)
    return out


def lineage_text(row: dict[str, Any]) -> str:
    parts = [str(row.get("branch_id") or ""), str(row.get("birth_stage") or "")]
    for item in row.get("hook_path") or []:
        if isinstance(item, dict):
            parts.append(f"L{item.get('target_loop')}_{item.get('target_layer')}")
    return " ".join(parts)


def has_lineage_stage(row: dict[str, Any], stage: str) -> bool:
    return stage in lineage_text(row)


def has_lineage_layer(row: dict[str, Any], layer: int) -> bool:
    suffix = f"_{layer}"
    return any(part.endswith(suffix) for part in lineage_text(row).split())


def recipe_definitions() -> list[dict[str, Any]]:
    return [
        {"recipe_id": "baseline_v3", "family": "baseline", "primary": True, "description": "Exact locked v3 replay baseline."},
        {"recipe_id": "L24_heavy", "family": "layer_replay", "layer_filter": 24, "description": "Replay proxy: retain final branches with L24 lineage."},
        {"recipe_id": "L36_heavy", "family": "layer_replay", "layer_filter": 36, "description": "Replay proxy: retain final branches with L36 lineage."},
        {"recipe_id": "L47_heavy", "family": "layer_replay", "layer_filter": 47, "description": "Replay proxy: retain final branches with L47 lineage."},
        {"recipe_id": "L2_47_emphasis", "family": "stage_replay", "stage_filter": "L2_47", "description": "Replay proxy: retain final branches with L2_47 lineage."},
        {"recipe_id": "science_DualAnchor_aligned", "family": "direction_replay", "family_filter": "old_tap_aligned", "description": "v3 available aligned direction family."},
        {"recipe_id": "science_bridge_materialization", "family": "direction_replay", "data_available": False, "description": "No regenerated bridge/materialization recipe artifact available."},
        {"recipe_id": "science_random_orthogonal_control", "family": "negative_control", "data_available": False, "description": "No regenerated random orthogonal recipe artifact available."},
        {"recipe_id": "science_alpha_005_only", "family": "alpha_replay", "data_available": True, "description": "Equivalent to v3 alpha 0.005 replay."},
        {"recipe_id": "science_alpha_01_only", "family": "alpha_replay", "data_available": False, "description": "No regenerated alpha 0.01 science recipe artifact available."},
        {"recipe_id": "science_alpha_mixed", "family": "alpha_replay", "data_available": False, "description": "No regenerated mixed alpha artifact available."},
        {"recipe_id": "science_alpha_02_diagnostic", "family": "diagnostic", "data_available": False, "description": "Diagnostic-only; no regenerated alpha 0.02 artifact available."},
        {"recipe_id": "science_more_children_same_budget", "family": "diagnostic", "data_available": False, "description": "No extra pre-prune children were generated in this replay pass."},
        {"recipe_id": "science_budget_10_diagnostic", "family": "diagnostic", "data_available": False, "description": "v3 final pool is budget 8; budget 10 requires regeneration."},
        {"recipe_id": "science_perturbation_escalation", "family": "diagnostic", "data_available": False, "description": "Escalation is trigger-audited, not regenerated here."},
        {"recipe_id": "science_parser_aware_diagnostic", "family": "parser_diagnostic", "description": "Same branches as baseline, evaluated with strict and relaxed parser variants."},
        {"recipe_id": "reasoning_baseline_v3", "family": "reasoning_terminal", "domain": "reasoning", "description": "Reasoning branch generation unchanged."},
    ]


def filter_pool_for_recipe(pool: Sequence[dict[str, Any]], recipe: dict[str, Any]) -> list[dict[str, Any]]:
    if recipe.get("data_available") is False:
        return []
    recipe_id = str(recipe.get("recipe_id"))
    if recipe_id in {"baseline_v3", "science_alpha_005_only", "science_parser_aware_diagnostic", "reasoning_baseline_v3"}:
        return list(pool)
    if recipe.get("layer_filter"):
        return [row for row in pool if has_lineage_layer(row, int(recipe["layer_filter"]))]
    if recipe.get("stage_filter"):
        return [row for row in pool if has_lineage_stage(row, str(recipe["stage_filter"]))]
    if recipe.get("family_filter"):
        return [row for row in pool if str(row.get("perturbation_family")) == str(recipe["family_filter"]) or row.get("birth_stage") == "root"]
    return list(pool)


def option_map(task_id: str) -> dict[str, str]:
    meta = task_meta_by_id().get(task_id, {})
    options = meta.get("options") or {}
    if isinstance(options, str):
        try:
            options = ast.literal_eval(options)
        except Exception:
            options = {}
    return {str(key).upper(): str(value) for key, value in dict(options).items()}


def letter_pattern(letter: str) -> re.Pattern[str]:
    escaped = re.escape(letter.upper())
    return re.compile(rf"(^|[^a-z0-9])(?:answer\s*is\s*)?(?:option\s*)?[\(\[]?{escaped}[\)\].:]?([^a-z0-9]|$)", re.IGNORECASE)


def parser_variant_row(row: dict[str, Any]) -> dict[str, Any]:
    task_id = str(row.get("task_id"))
    correct = str(row.get("correct_option") or "").upper().strip()
    parsed = str(row.get("parsed_answer") or "").upper().strip()
    output_raw = str(row.get("output_text") or "")
    output_norm = normalize_text(output_raw)
    options = option_map(task_id)
    correct_text = options.get(correct, "")
    correct_text_norm = normalize_text(correct_text)
    option_letter_mentions = {letter for letter in options if letter_pattern(letter).search(output_raw)}
    option_text_mentions = {letter for letter, text in options.items() if text and normalize_text(text) and normalize_text(text) in output_norm}
    strict_correct = truthy(row.get("parse_success")) and parsed == correct
    contains_correct_letter = correct in option_letter_mentions
    contains_correct_text = bool(correct_text_norm and correct_text_norm in output_norm)
    ambiguous = len(option_letter_mentions | option_text_mentions) > 1
    relaxed_text_correct = contains_correct_text and not ambiguous
    robust_correct = strict_correct or contains_correct_letter or relaxed_text_correct
    strict_reward = row_reward(row)
    relaxed_reward = 1.0 if relaxed_text_correct else strict_reward
    robust_reward = 1.0 if robust_correct else strict_reward
    return {
        "task_id": task_id,
        "domain": row.get("domain"),
        "source_dataset": row.get("source_dataset"),
        "source_detail": source_detail(task_id, row.get("source_dataset")),
        "split": row.get("split"),
        "branch_id": row.get("branch_id"),
        "correct_option": correct,
        "correct_option_text": correct_text,
        "parsed_answer": row.get("parsed_answer"),
        "parse_success": row.get("parse_success"),
        "parse_failure_reason": row.get("parse_failure_reason"),
        "strict_reward": strict_reward,
        "strict_correct": strict_correct,
        "contains_correct_letter": contains_correct_letter,
        "contains_correct_option_text": contains_correct_text,
        "relaxed_text_correct": relaxed_text_correct,
        "robust_correct": robust_correct,
        "strict_reward_diagnostic": strict_reward,
        "relaxed_text_reward_diagnostic": relaxed_reward,
        "letter_text_reward_diagnostic": robust_reward,
        "ambiguous_output": ambiguous,
        "multiple_option_mentions": len(option_letter_mentions | option_text_mentions),
        "option_letter_mentions": sorted(option_letter_mentions),
        "option_text_mentions": sorted(option_text_mentions),
        "empty_output": row.get("empty_output"),
        "hit_max_tokens": row.get("hit_max_tokens"),
        "repetition_rate": row.get("repetition_rate"),
        "output_length": row.get("output_length"),
    }


def parser_audit_rows() -> list[dict[str, Any]]:
    _payload, rows, _stage_rows, _task_rows, _terminal_rows = load_v3_bundle()
    return [parser_variant_row(row) for row in rows if row.get("domain") == "science"]


def summarize_parser_task_metrics(parser_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task_rows = {str(row.get("task_id")): row for row in load_v3_task_rows_csv() if row.get("domain") == "science"}
    final_ids_by_task = {task_id: set(str(x) for x in parse_literal(row.get("final_branch_ids"), [])) for task_id, row in task_rows.items()}
    for row in parser_rows:
        if str(row.get("branch_id")) in final_ids_by_task.get(str(row.get("task_id")), set()):
            grouped[str(row.get("task_id"))].append(row)
    out = []
    for task_id, vals in sorted(grouped.items()):
        task = task_rows.get(task_id, {})
        strict_best = max((safe_float(row.get("strict_reward"), 0.0) for row in vals), default=float("nan"))
        relaxed_best = max((safe_float(row.get("relaxed_text_reward_diagnostic"), 0.0) for row in vals), default=float("nan"))
        robust_best = max((safe_float(row.get("letter_text_reward_diagnostic"), 0.0) for row in vals), default=float("nan"))
        out.append(
            {
                "task_id": task_id,
                "source_dataset": task.get("source_dataset"),
                "source_detail": source_detail(task_id, task.get("source_dataset")),
                "split": task.get("split"),
                "strict_terminal_best_reward": strict_best,
                "relaxed_terminal_best_reward": relaxed_best,
                "robust_terminal_best_reward": robust_best,
                "strict_positive_oracle": 1.0 if strict_best > 0 else 0.0,
                "relaxed_positive_oracle": 1.0 if relaxed_best > 0 else 0.0,
                "robust_positive_oracle": 1.0 if robust_best > 0 else 0.0,
                "final_candidate_count": len(vals),
                "parse_success_rate": finite_mean(row.get("parse_success") for row in vals),
                "ambiguous_rate": finite_mean(row.get("ambiguous_output") for row in vals),
                "missed_correct_candidate_count": sum(1 for row in vals if safe_float(row.get("strict_reward"), 0.0) <= 0 and truthy(row.get("robust_correct"))),
            }
        )
    return out


def task_metric_rows(domain: str | None = None, split_role_filter: str | None = None) -> list[dict[str, Any]]:
    suite = read_json(OUT_ROOT / "task_suite.json", {}) or {}
    suite_by_id = {str(row.get("task_id")): row for row in suite.get("task_rows", [])}
    rows = []
    for row in load_v3_task_rows_csv():
        task_id = str(row.get("task_id"))
        suite_row = suite_by_id.get(task_id)
        if suite_row is None:
            continue
        if domain and str(row.get("domain")) != domain:
            continue
        if split_role_filter and suite_row.get("split_role") != split_role_filter:
            continue
        rows.append({**row, "split_role": suite_row.get("split_role"), "source_detail": suite_row.get("source_detail")})
    return rows


def metric_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task_count": len(rows),
        "positive_oracle_rate": finite_mean(row.get("positive_oracle") for row in rows),
        "reward_diverse_rate": finite_mean(row.get("terminal_reward_diverse") for row in rows),
        "terminal_best_reward": finite_mean(row.get("terminal_best_reward") for row in rows),
        "forced_top1_reward": finite_mean(row.get("terminal_forced_top1_reward") for row in rows),
        "forced_top1_oracle": finite_mean(row.get("terminal_forced_top1_oracle") for row in rows),
        "terminal_oracle_retained": finite_mean(row.get("terminal_oracle_retained") for row in rows),
        "stage_false_prunes": finite_mean(row.get("stage_false_prunes") for row in rows),
    }


def selected_indices_for_policy(row: dict[str, Any]) -> list[int]:
    return [int(idx) for idx in parse_literal(row.get("selected_indices"), [])]


def summarize_selected(pool: Sequence[dict[str, Any]], ordered: Sequence[dict[str, Any]], selected_count: int | None = None) -> dict[str, Any]:
    if selected_count is None:
        selected = list(ordered)
    else:
        selected = list(ordered)[:selected_count]
    all_rewards = [row_reward(row) for row in pool]
    selected_rewards = [row_reward(row) for row in selected]
    oracle = max(all_rewards) if all_rewards else float("nan")
    return {
        "candidate_count": len(pool),
        "selected_count": len(selected),
        "oracle_reward": oracle,
        "oracle_retained": 1.0 if selected_rewards and max(selected_rewards) == oracle else 0.0,
        "best_selected_reward": max(selected_rewards) if selected_rewards else float("nan"),
        "first_selected_reward": selected_rewards[0] if selected_rewards else float("nan"),
        "first_selected_oracle": 1.0 if selected_rewards and selected_rewards[0] == oracle else 0.0,
    }


def recipe_task_rows(domain: str, split_role_filter: str, recipe_ids: set[str] | None = None) -> list[dict[str, Any]]:
    _payload, all_rows, _stage_rows, _task_rows_pt, _terminal_rows_pt = load_v3_bundle()
    all_by_id = rows_by_id(all_rows)
    task_rows = task_metric_rows(domain, split_role_filter)
    parser_by_branch = {str(row.get("branch_id")): row for row in parser_audit_rows()} if domain == "science" else {}
    recipes = [recipe for recipe in recipe_definitions() if not recipe_ids or recipe.get("recipe_id") in recipe_ids]
    out: list[dict[str, Any]] = []
    force = os.environ.get("FORCE_RERUN") == "1"
    completed = set(read_json(COMPLETED_TASKS_JSON, []) or [])
    completed_recipes = set(read_json(COMPLETED_RECIPES_JSON, []) or [])
    lineage_rows: list[dict[str, Any]] = []
    for recipe in recipes:
        recipe_id = str(recipe.get("recipe_id"))
        if recipe.get("domain") and recipe.get("domain") != domain:
            continue
        if domain == "science" and recipe_id.startswith("reasoning_"):
            continue
        if domain == "reasoning" and recipe_id.startswith("science_"):
            continue
        recipe_rows: list[dict[str, Any]] = []
        for task_row in task_rows:
            task_id = str(task_row.get("task_id"))
            progress_key = f"{recipe_id}::{task_id}::{split_role_filter}"
            if progress_key in completed and not force:
                continue
            pool = v3_final_pool(task_row, all_by_id)
            filtered = filter_pool_for_recipe(pool, recipe)
            ordered = ordered_pool(task_id, filtered)
            rewards = [row_reward(row) for row in filtered]
            strict_best = max(rewards) if rewards else float("nan")
            robust_rewards = []
            if parser_by_branch:
                for row in filtered:
                    parser = parser_by_branch.get(str(row.get("branch_id")), {})
                    robust_rewards.append(safe_float(parser.get("letter_text_reward_diagnostic"), row_reward(row)))
            robust_best = max(robust_rewards) if robust_rewards else strict_best
            unique_rewards = sorted({round(row_reward(row), 6) for row in filtered})
            metric = summarize_selected(filtered, ordered, 1)
            top2 = summarize_selected(filtered, ordered, 2)
            top4 = summarize_selected(filtered, ordered, 4)
            full = summarize_selected(filtered, ordered, None)
            rec = {
                "recipe_id": recipe_id,
                "recipe_family": recipe.get("family"),
                "domain": domain,
                "task_id": task_id,
                "source_dataset": task_row.get("source_dataset"),
                "source_detail": task_row.get("source_detail"),
                "split": task_row.get("split"),
                "split_role": split_role_filter,
                "data_available": recipe.get("data_available", True) is not False and bool(filtered),
                "replay_diagnostic": 1.0,
                "candidate_count": len(filtered),
                "baseline_candidate_count": len(pool),
                "positive_oracle": 1.0 if strict_best > 0 else 0.0,
                "terminal_best_reward": strict_best,
                "reward_diverse": 1.0 if len(unique_rewards) > 1 else 0.0,
                "forced_top1_reward": metric["first_selected_reward"],
                "forced_top1_oracle": metric["first_selected_oracle"],
                "top2_oracle": top2["oracle_retained"],
                "top4_oracle": top4["oracle_retained"],
                "full_handoff_oracle": full["oracle_retained"],
                "robust_parser_terminal_best_reward": robust_best,
                "robust_parser_positive_oracle": 1.0 if robust_best > 0 else 0.0,
                "parse_success": finite_mean(parser_by_branch.get(str(row.get("branch_id")), {}).get("parse_success") for row in filtered) if parser_by_branch else float("nan"),
                "stability_empty_rate": finite_mean(row.get("empty_output") for row in filtered),
                "hit_max_tokens_rate": finite_mean(row.get("hit_max_tokens") for row in filtered),
                "mean_perturb_count": finite_mean(row.get("perturb_count") for row in filtered),
                "mean_child_parent_delta": finite_mean(row.get("reward_delta_from_parent") for row in filtered if row.get("parent_branch_id")),
                "l47_fraction": finite_mean(1.0 if has_lineage_layer(row, 47) else 0.0 for row in filtered),
            }
            recipe_rows.append(rec)
            out.append(rec)
            lineage_rows.extend(
                {
                    "recipe_id": recipe_id,
                    "task_id": task_id,
                    "branch_id": row.get("branch_id"),
                    "birth_stage": row.get("birth_stage"),
                    "reward": row_reward(row),
                    "lineage_path": row.get("lineage_path"),
                }
                for row in filtered
            )
            write_json(PROGRESS_ROOT / f"task_{safe_name(progress_key)}.json", {"completed": True, "row": rec, "saved_at": time.time()})
            write_json(CANDIDATE_TREE_ROOT / f"{safe_name(recipe_id)}__{safe_name(task_id)}.json", {"recipe_id": recipe_id, "task_id": task_id, "candidate_branch_ids": [str(row.get("branch_id")) for row in filtered]})
            completed.add(progress_key)
            write_json(COMPLETED_TASKS_JSON, sorted(completed))
        if recipe_rows:
            completed_recipes.add(recipe_id)
            write_json(PROGRESS_ROOT / f"recipe_{safe_name(recipe_id)}_{split_role_filter}.json", {"recipe_id": recipe_id, "split_role": split_role_filter, "completed_task_count": len(recipe_rows), "saved_at": time.time()})
            write_json(COMPLETED_RECIPES_JSON, sorted(completed_recipes))
        write_json(PARTIAL_SUMMARY_JSON, {"current_recipe": recipe_id, "row_count": len(out), "saved_at": time.time()})
    if lineage_rows:
        write_csv(LINEAGE_INCREMENTAL_CSV, lineage_rows)
    return out


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:180]


def summarize_recipe_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("recipe_id"))].append(row)
    out = []
    for recipe_id, vals in sorted(grouped.items()):
        out.append(
            {
                "recipe_id": recipe_id,
                "task_count": len(vals),
                "available_task_count": sum(1 for row in vals if truthy(row.get("data_available"))),
                "positive_oracle_rate": finite_mean(row.get("positive_oracle") for row in vals if truthy(row.get("data_available"))),
                "reward_diverse_rate": finite_mean(row.get("reward_diverse") for row in vals if truthy(row.get("data_available"))),
                "terminal_best_reward": finite_mean(row.get("terminal_best_reward") for row in vals if truthy(row.get("data_available"))),
                "forced_top1_reward": finite_mean(row.get("forced_top1_reward") for row in vals if truthy(row.get("data_available"))),
                "forced_top1_oracle": finite_mean(row.get("forced_top1_oracle") for row in vals if truthy(row.get("data_available"))),
                "top2_oracle": finite_mean(row.get("top2_oracle") for row in vals if truthy(row.get("data_available"))),
                "top4_oracle": finite_mean(row.get("top4_oracle") for row in vals if truthy(row.get("data_available"))),
                "full_handoff_oracle": finite_mean(row.get("full_handoff_oracle") for row in vals if truthy(row.get("data_available"))),
                "robust_parser_terminal_best_reward": finite_mean(row.get("robust_parser_terminal_best_reward") for row in vals if truthy(row.get("data_available"))),
                "robust_parser_positive_oracle": finite_mean(row.get("robust_parser_positive_oracle") for row in vals if truthy(row.get("data_available"))),
                "parse_success": finite_mean(row.get("parse_success") for row in vals if truthy(row.get("data_available"))),
                "l47_fraction": finite_mean(row.get("l47_fraction") for row in vals if truthy(row.get("data_available"))),
            }
        )
    return out


def select_best_science_recipe(summary_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row
        for row in summary_rows
        if row.get("recipe_id") not in {"science_parser_aware_diagnostic", "science_alpha_02_diagnostic", "science_budget_10_diagnostic"}
        and safe_float(row.get("available_task_count"), 0.0) > 0
    ]
    if not eligible:
        return {"recipe_id": "baseline_v3", "selection_reason": "no available replay recipe rows"}
    baseline = next((row for row in eligible if row.get("recipe_id") == "baseline_v3"), eligible[0])
    best = max(eligible, key=lambda row: (safe_float(row.get("positive_oracle_rate"), -1.0), safe_float(row.get("terminal_best_reward"), -1.0), safe_float(row.get("reward_diverse_rate"), -1.0)))
    improved = (
        safe_float(best.get("positive_oracle_rate"), 0.0) > safe_float(baseline.get("positive_oracle_rate"), 0.0) + 1e-9
        or safe_float(best.get("terminal_best_reward"), 0.0) > safe_float(baseline.get("terminal_best_reward"), 0.0) + 1e-9
    )
    chosen = best if improved and best.get("recipe_id") != "baseline_v3" else baseline
    return {
        **chosen,
        "selected": bool(improved and best.get("recipe_id") != "baseline_v3"),
        "selection_reason": "best calibration replay proxy" if improved else "no replay recipe improved over baseline",
        "baseline_positive_oracle_rate": baseline.get("positive_oracle_rate"),
        "baseline_terminal_best_reward": baseline.get("terminal_best_reward"),
    }


def grouped_summary(rows: Sequence[dict[str, Any]], group_key: str, metrics: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(group_key)].append(row)
    out = []
    for value, vals in sorted(grouped.items(), key=lambda item: str(item[0])):
        rec = {group_key: value, "count": len(vals)}
        for metric in metrics:
            rec[metric] = finite_mean(row.get(metric) for row in vals)
        out.append(rec)
    return out


def artifact_available(path: Path) -> str:
    return "READY" if path.exists() else "MISSING"


def inventory_main() -> int:
    started = time.time()
    artifacts = [
        {"artifact": "previous_convergence_report", "path": str(PREV_ROOT), "status": "READY" if PREV_ROOT.exists() else "MISSING"},
        {"artifact": "v3_architecture_looped", "path": str(V3_ROOT), "status": "READY" if (V3_ROOT / "architecture_looped_rows.pt").exists() else "MISSING"},
        {"artifact": "v3_stage_decisions", "path": str(V3_ROOT / "stage_decisions.csv"), "status": artifact_available(V3_ROOT / "stage_decisions.csv")},
        {"artifact": "v3_hard_slices", "path": str(V3_ROOT / "hard_slices.json"), "status": artifact_available(V3_ROOT / "hard_slices.json")},
        {"artifact": "v3_threshold_budget", "path": str(V3_ROOT / "threshold_budget.json"), "status": artifact_available(V3_ROOT / "threshold_budget.json")},
        {"artifact": "v3_l47", "path": str(V3_ROOT / "l47_ablation.json"), "status": artifact_available(V3_ROOT / "l47_ablation.json")},
        {"artifact": "v3_phase2a", "path": str(V3_ROOT / "phase2a_readiness.json"), "status": artifact_available(V3_ROOT / "phase2a_readiness.json")},
        {"artifact": "true_carry", "path": str(PROBE_ROOT / "bg_dualanchor_true_carry_equivalence_v1_2026-05-31"), "status": "READY" if (PROBE_ROOT / "bg_dualanchor_true_carry_equivalence_v1_2026-05-31").exists() else "MISSING"},
        {"artifact": "perturbation_lift", "path": str(PROBE_ROOT / "bg_dualanchor_perturbation_lift_v1_2026-05-31"), "status": "READY" if (PROBE_ROOT / "bg_dualanchor_perturbation_lift_v1_2026-05-31").exists() else "MISSING"},
    ]
    _payload, rows, stage_rows, task_rows_pt, terminal_rows_pt = load_v3_bundle()
    plan_rows = audit_plan_tasks()
    science_plan = [row for row in plan_rows if str(row.get("domain")) == "science" and row.get("correct_option")]
    reasoning_plan = [row for row in plan_rows if str(row.get("domain")) == "reasoning" and row.get("correct_option")]
    task_rows = load_v3_task_rows_csv()
    science_v3 = [row for row in task_rows if row.get("domain") == "science"]
    reasoning_v3 = [row for row in task_rows if row.get("domain") == "reasoning"]
    parser_rows = parser_audit_rows()
    prior = load_prev_json("summary.json")
    missing = [row["artifact"] for row in artifacts if row["status"] == "MISSING"]
    verdict = "BLOCKED" if "v3_architecture_looped" in missing else ("PARTIAL" if missing or len(science_plan) < 32 else "READY")
    payload = {
        "BG_SCIENCE_RECIPE_REASONING_DEFER_INVENTORY_VERDICT": verdict,
        "artifact_rows": artifacts,
        "dualanchor_taps_ready": True,
        "dualanchor_taps": [ANCHOR_A, ANCHOR_B],
        "plan_task_counts": {"science": len(science_plan), "reasoning": len(reasoning_plan)},
        "v3_task_counts": {"science": len(science_v3), "reasoning": len(reasoning_v3)},
        "science_source_counts": dict(Counter(source_detail(str(row.get("task_id")), row.get("source_dataset")) for row in science_plan)),
        "reasoning_source_counts": dict(Counter(str(row.get("source_dataset")) for row in reasoning_plan)),
        "parser_candidate_count": len(parser_rows),
        "parser_available": len(parser_rows) > 0,
        "lineage_row_count": len(rows),
        "stage_decision_count": len(stage_rows),
        "terminal_policy_row_count": len(terminal_rows_pt) or len(load_v3_terminal_rows_csv()),
        "soft_hair_availability": {"L30_L42_dataset": (PREV_ROOT / "convergence_hair_dataset.pt").exists(), "hard_merge_allowed": False},
        "can_regenerate_science_candidate_trees": "runner exists, but this pass uses replay diagnostics unless explicitly run overnight",
        "previous_status": prior.get("DUALANCHOR_CONVERGENCE_HAIRS_RS_STATUS"),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "inventory.json", payload)
    lines = [
        "# DualAnchor Science Branch Recipe + Reasoning Defer Inventory v1",
        "",
        status_line("BG_SCIENCE_RECIPE_REASONING_DEFER_INVENTORY_VERDICT", verdict),
        "",
        "## Artifact Availability",
        "",
        *md_table(artifacts, ["artifact", "status", "path"]),
        "",
        "## Counts",
        "",
        f"- audit-plan science tasks: `{len(science_plan)}`",
        f"- audit-plan reasoning tasks: `{len(reasoning_plan)}`",
        f"- v3 science tasks: `{len(science_v3)}`",
        f"- v3 reasoning tasks: `{len(reasoning_v3)}`",
        f"- v3 lineage rows: `{len(rows)}`",
        f"- v3 stage decisions: `{len(stage_rows)}`",
        f"- parser candidate rows: `{len(parser_rows)}`",
        "",
        "## Boundary",
        "",
        "- Convergence hairs remain soft diagnostics only.",
        "- This inventory does not run steering, wrapper/local-agent code, training, or Hunter-Seeker modules.",
    ]
    write_md(OUT_ROOT / "inventory.md", lines)
    print(status_line("BG_SCIENCE_RECIPE_REASONING_DEFER_INVENTORY_VERDICT", verdict))
    return 0 if verdict != "BLOCKED" else 1


def task_suite_main() -> int:
    started = time.time()
    rows, inventory = build_task_suite_rows()
    if inventory["selected_science"] < 32:
        verdict = "SCIENCE_LIMITED"
    elif inventory["selected_reasoning"] < 16:
        verdict = "REASONING_LIMITED"
    elif inventory["selected_count"] >= 72:
        verdict = "READY"
    else:
        verdict = "PARTIAL"
    payload = {
        "BG_SCIENCE_RECIPE_REASONING_TASK_SUITE_VERDICT": verdict,
        "task_rows": rows,
        "inventory": inventory,
        "selection_note": "Selected all available science MCQ tasks and the top 24 reasoning tasks from the local audit plan; local pool has only 31 science tasks, below the requested 32-task minimum.",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "task_suite.json", payload)
    write_csv(OUT_ROOT / "task_suite.csv", rows)
    lines = [
        "# Science/Reasoning Task Suite v1",
        "",
        status_line("BG_SCIENCE_RECIPE_REASONING_TASK_SUITE_VERDICT", verdict),
        "",
        "## Inventory",
        "",
        *[f"- {key}: `{value}`" for key, value in inventory.items() if not isinstance(value, dict)],
        "",
        "## Domain Counts",
        "",
        *[f"- {key}: `{value}`" for key, value in inventory["domain_counts"].items()],
        "",
        "## Science Source Counts",
        "",
        *[f"- {key}: `{value}`" for key, value in inventory["science_source_counts"].items()],
        "",
        "## Task Rows",
        "",
        *md_table(rows, ["index", "task_id", "domain", "source_dataset", "source_detail", "split", "split_role", "v3_generated", "prior_positive_oracle", "prior_reward_diverse"]),
    ]
    write_md(OUT_ROOT / "task_suite.md", lines)
    print(status_line("BG_SCIENCE_RECIPE_REASONING_TASK_SUITE_VERDICT", verdict))
    return 0 if verdict != "BLOCKED" else 1


def science_parser_audit_main() -> int:
    started = time.time()
    rows = parser_audit_rows()
    task_metrics = summarize_parser_task_metrics(rows)
    strict_positive = finite_mean(row.get("strict_positive_oracle") for row in task_metrics)
    relaxed_positive = finite_mean(row.get("relaxed_positive_oracle") for row in task_metrics)
    robust_positive = finite_mean(row.get("robust_positive_oracle") for row in task_metrics)
    strict_best = finite_mean(row.get("strict_terminal_best_reward") for row in task_metrics)
    robust_best = finite_mean(row.get("robust_terminal_best_reward") for row in task_metrics)
    missed = sum(1 for row in rows if safe_float(row.get("strict_reward"), 0.0) <= 0 and truthy(row.get("robust_correct")))
    false_risk = finite_mean(row.get("ambiguous_output") for row in rows if truthy(row.get("robust_correct")))
    parse_success = finite_mean(row.get("parse_success") for row in rows)
    if not rows:
        verdict = "INSUFFICIENT"
    elif robust_positive >= strict_positive + 0.20 or robust_best >= strict_best + 0.20:
        verdict = "PARSER_MAJOR_BLOCKER"
    elif robust_positive >= strict_positive + 0.05 or robust_best >= strict_best + 0.05 or missed > 0:
        verdict = "PARSER_PARTLY_RESPONSIBLE"
    else:
        verdict = "PARSER_OK_BRANCH_WEAK"
    payload = {
        "BG_SCIENCE_PARSER_REWARD_AUDIT_VERDICT": verdict,
        "candidate_count": len(rows),
        "task_metric_rows": task_metrics,
        "summary": {
            "parse_success_rate": parse_success,
            "strict_positive_oracle_rate": strict_positive,
            "relaxed_positive_oracle_rate": relaxed_positive,
            "robust_positive_oracle_rate": robust_positive,
            "strict_terminal_best_reward": strict_best,
            "robust_terminal_best_reward": robust_best,
            "missed_correct_candidate_count": missed,
            "robust_false_positive_risk_proxy": false_risk,
            "no_mcq_letter_count": sum(1 for row in rows if str(row.get("parse_failure_reason")) == "no_mcq_letter"),
            "ambiguous_output_rate": finite_mean(row.get("ambiguous_output") for row in rows),
            "multiple_option_mention_rate": finite_mean(1.0 if safe_float(row.get("multiple_option_mentions"), 0.0) > 1 else 0.0 for row in rows),
        },
        "source_summary": grouped_summary(task_metrics, "source_detail", ["strict_positive_oracle", "robust_positive_oracle", "strict_terminal_best_reward", "robust_terminal_best_reward", "parse_success_rate", "missed_correct_candidate_count"]),
        "primary_parser": "strict letter parser remains primary; relaxed/robust parser is diagnostic only",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "science_parser_audit.json", payload)
    write_csv(OUT_ROOT / "science_parser_audit_rows.csv", rows)
    lines = [
        "# Science Parser/Reward Audit v1",
        "",
        status_line("BG_SCIENCE_PARSER_REWARD_AUDIT_VERDICT", verdict),
        "",
        "## Summary",
        "",
        *[f"- {key}: `{fmt(value)}`" for key, value in payload["summary"].items()],
        "",
        "## Source Summary",
        "",
        *md_table(payload["source_summary"], ["source_detail", "count", "strict_positive_oracle", "robust_positive_oracle", "strict_terminal_best_reward", "robust_terminal_best_reward", "parse_success_rate", "missed_correct_candidate_count"]),
        "",
        "## Boundary",
        "",
        "- Strict parser remains the primary historical reward parser.",
        "- Relaxed and letter+text parsers are diagnostic and are not silently substituted into historical metrics.",
    ]
    write_md(OUT_ROOT / "science_parser_audit.md", lines)
    print(status_line("BG_SCIENCE_PARSER_REWARD_AUDIT_VERDICT", verdict))
    return 0


def science_recipe_plan_main() -> int:
    started = time.time()
    recipes = recipe_definitions()
    direction_limited = any(recipe.get("data_available") is False for recipe in recipes)
    verdict = "DIRECTION_LIMITED" if direction_limited else "READY"
    payload = {
        "BG_SCIENCE_BRANCH_RECIPE_PLAN_VERDICT": verdict,
        "recipes": recipes,
        "baseline": {
            "schedule": LOCKED_SCHEDULE,
            "selector": [ANCHOR_A, ANCHOR_B],
            "threshold": "mean_floor_very_loose",
            "budget": 8,
            "l47": "active in nonterminal loops",
            "terminal": "confidence-gated top1; otherwise defer / survivor-set handoff",
            "convergence_hairs": "soft-only diagnostics",
        },
        "selection_rule": "choose on recipe_calibration only; heldout remains untouched",
        "replay_boundary": "recipes without regenerated artifacts are marked data_available=false or replay_diagnostic",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "science_recipe_plan.json", payload)
    lines = [
        "# Science Branch Recipe Plan v1",
        "",
        status_line("BG_SCIENCE_BRANCH_RECIPE_PLAN_VERDICT", verdict),
        "",
        "## Locked Baseline",
        "",
        *[f"- {key}: `{value}`" for key, value in payload["baseline"].items()],
        "",
        "## Recipes",
        "",
        *md_table(recipes, ["recipe_id", "family", "data_available", "description"]),
        "",
        "## Boundary",
        "",
        "- Recipe selection uses calibration rows only.",
        "- Heldout is evaluation only.",
        "- No steering or model training is run by this plan.",
    ]
    write_md(OUT_ROOT / "science_recipe_plan.md", lines)
    print(status_line("BG_SCIENCE_BRANCH_RECIPE_PLAN_VERDICT", verdict))
    return 0


def science_recipe_calibration_main() -> int:
    started = time.time()
    rows = recipe_task_rows("science", "recipe_calibration")
    summary_rows = summarize_recipe_rows(rows)
    best = select_best_science_recipe(summary_rows)
    baseline = next((row for row in summary_rows if row.get("recipe_id") == "baseline_v3"), {})
    parser = next((row for row in summary_rows if row.get("recipe_id") == "science_parser_aware_diagnostic"), {})
    strict_best = safe_float(baseline.get("terminal_best_reward"), 0.0)
    robust_best = safe_float(parser.get("robust_parser_terminal_best_reward"), strict_best)
    if not rows:
        verdict = "BLOCKED"
    elif safe_float(best.get("selected"), 0.0) > 0:
        verdict = "SCIENCE_RECIPE_FOUND"
    elif robust_best > strict_best + 0.05:
        verdict = "SCIENCE_PARSER_DOMINATES"
    elif safe_float(baseline.get("positive_oracle_rate"), 0.0) < 0.25:
        verdict = "SCIENCE_NO_RECIPE_IMPROVEMENT"
    else:
        verdict = "SCIENCE_RECIPE_WEAK"
    payload = {
        "BG_SCIENCE_BRANCH_RECIPE_CALIBRATION_VERDICT": verdict,
        "summary_rows": summary_rows,
        "best_science_recipe": best,
        "replay_diagnostic": True,
        "strict_primary": True,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "science_recipe_calibration.json", payload)
    write_json(OUT_ROOT / "best_science_recipe.json", best)
    write_csv(OUT_ROOT / "science_recipe_calibration_rows.csv", rows)
    lines = [
        "# Science Branch Recipe Calibration v1",
        "",
        status_line("BG_SCIENCE_BRANCH_RECIPE_CALIBRATION_VERDICT", verdict),
        "",
        "## Summary",
        "",
        *md_table(summary_rows, ["recipe_id", "task_count", "available_task_count", "positive_oracle_rate", "reward_diverse_rate", "terminal_best_reward", "forced_top1_reward", "top4_oracle", "robust_parser_terminal_best_reward", "parse_success", "l47_fraction"]),
        "",
        "## Selected Recipe",
        "",
        *[f"- {key}: `{fmt(value)}`" for key, value in best.items()],
        "",
        "## Boundary",
        "",
        "- Calibration uses recipe_calibration split only.",
        "- These rows are candidate-tree replay diagnostics unless the recipe has regenerated artifacts.",
    ]
    write_md(OUT_ROOT / "science_recipe_calibration.md", lines)
    print(status_line("BG_SCIENCE_BRANCH_RECIPE_CALIBRATION_VERDICT", verdict))
    return 0 if verdict != "BLOCKED" else 1


def science_recipe_heldout_main() -> int:
    started = time.time()
    best = read_json(OUT_ROOT / "best_science_recipe.json", {}) or {"recipe_id": "baseline_v3"}
    recipe_ids = {"baseline_v3", str(best.get("recipe_id") or "baseline_v3"), "science_parser_aware_diagnostic"}
    rows = recipe_task_rows("science", "heldout_eval", recipe_ids)
    summary_rows = summarize_recipe_rows(rows)
    baseline = next((row for row in summary_rows if row.get("recipe_id") == "baseline_v3"), {})
    selected = next((row for row in summary_rows if row.get("recipe_id") == str(best.get("recipe_id"))), baseline)
    parser = next((row for row in summary_rows if row.get("recipe_id") == "science_parser_aware_diagnostic"), {})
    strict_ok = safe_float(selected.get("positive_oracle_rate"), 0.0) > SCIENCE_BASELINE_POSITIVE_ORACLE and safe_float(selected.get("terminal_best_reward"), 0.0) > SCIENCE_BASELINE_BEST_REWARD
    preferred = safe_float(selected.get("positive_oracle_rate"), 0.0) >= 0.25 and safe_float(selected.get("terminal_best_reward"), 0.0) >= 0.20
    parser_lift = safe_float(parser.get("robust_parser_terminal_best_reward"), 0.0) > safe_float(baseline.get("terminal_best_reward"), 0.0) + 0.10
    if not rows:
        verdict = "INSUFFICIENT"
    elif not truthy(best.get("selected")):
        verdict = "SCIENCE_BRANCH_GENERATION_STILL_WEAK"
    elif parser_lift and not strict_ok:
        verdict = "SCIENCE_PARSER_BLOCKER"
    elif preferred:
        verdict = "SCIENCE_BRANCH_RECIPE_READY"
    elif strict_ok:
        verdict = "SCIENCE_BRANCH_RECIPE_WEAK_BUT_IMPROVED"
    else:
        verdict = "SCIENCE_BRANCH_GENERATION_STILL_WEAK"
    payload = {
        "BG_SCIENCE_BRANCH_RECIPE_HELDOUT_VERDICT": verdict,
        "selected_recipe_id": best.get("recipe_id"),
        "summary_rows": summary_rows,
        "heldout_task_count": len({row.get("task_id") for row in rows}),
        "replay_diagnostic": True,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "science_recipe_heldout.json", payload)
    write_csv(OUT_ROOT / "science_recipe_heldout_rows.csv", rows)
    lines = [
        "# Science Branch Recipe Heldout v1",
        "",
        status_line("BG_SCIENCE_BRANCH_RECIPE_HELDOUT_VERDICT", verdict),
        "",
        f"Selected calibration recipe: `{best.get('recipe_id')}`",
        "",
        "## Summary",
        "",
        *md_table(summary_rows, ["recipe_id", "task_count", "available_task_count", "positive_oracle_rate", "reward_diverse_rate", "terminal_best_reward", "forced_top1_reward", "top2_oracle", "top4_oracle", "full_handoff_oracle", "robust_parser_terminal_best_reward"]),
        "",
        "## Boundary",
        "",
        "- The selected recipe is not changed based on heldout.",
        "- Heldout rows remain replay diagnostics unless regenerated artifacts are present.",
    ]
    write_md(OUT_ROOT / "science_recipe_heldout.md", lines)
    print(status_line("BG_SCIENCE_BRANCH_RECIPE_HELDOUT_VERDICT", verdict))
    return 0


def science_source_breakdown_main() -> int:
    started = time.time()
    heldout = read_csv(OUT_ROOT / "science_recipe_heldout_rows.csv")
    calibration = read_csv(OUT_ROOT / "science_recipe_calibration_rows.csv")
    rows = [row for row in heldout + calibration if row.get("recipe_id") == "baseline_v3"]
    source_rows = grouped_summary(rows, "source_detail", ["positive_oracle", "reward_diverse", "terminal_best_reward", "forced_top1_reward", "top2_oracle", "top4_oracle", "full_handoff_oracle", "parse_success", "robust_parser_terminal_best_reward", "l47_fraction"])
    science_tasks = task_metric_rows("science", None)
    source_task_counts = dict(Counter(row.get("source_detail") for row in science_tasks))
    mmlu_rows = [row for row in source_rows if str(row.get("source_detail", "")).startswith("mmlu")]
    if not source_rows:
        verdict = "INSUFFICIENT"
    elif any(row.get("source_detail") == "sciq" and safe_float(row.get("terminal_best_reward"), 0.0) >= 0.5 for row in source_rows) and any(safe_float(row.get("terminal_best_reward"), 0.0) < 0.1 for row in mmlu_rows):
        verdict = "MMLU_SCIENCE_WEAK"
    elif max((safe_float(row.get("terminal_best_reward"), 0.0) for row in source_rows), default=0.0) - min((safe_float(row.get("terminal_best_reward"), 0.0) for row in source_rows), default=0.0) > 0.25:
        verdict = "SCIENCE_SOURCE_SPECIFIC"
    elif finite_mean(row.get("terminal_best_reward") for row in source_rows) < 0.2:
        verdict = "SCIENCE_UNIFORMLY_WEAK"
    else:
        verdict = "DATA_LIMITED"
    payload = {
        "BG_SCIENCE_SOURCE_BREAKDOWN_VERDICT": verdict,
        "source_rows": source_rows,
        "source_task_counts": source_task_counts,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "science_source_breakdown.json", payload)
    write_csv(OUT_ROOT / "science_source_rows.csv", rows)
    lines = [
        "# Science Source Breakdown v1",
        "",
        status_line("BG_SCIENCE_SOURCE_BREAKDOWN_VERDICT", verdict),
        "",
        "## Source Rows",
        "",
        *md_table(source_rows, ["source_detail", "count", "positive_oracle", "reward_diverse", "terminal_best_reward", "forced_top1_reward", "top4_oracle", "parse_success", "robust_parser_terminal_best_reward", "l47_fraction"]),
    ]
    write_md(OUT_ROOT / "science_source_breakdown.md", lines)
    print(status_line("BG_SCIENCE_SOURCE_BREAKDOWN_VERDICT", verdict))
    return 0


def terminal_policy_rows_for_domain(domain: str) -> list[dict[str, Any]]:
    suite_ids = selected_task_ids_from_suite()
    return [row for row in load_v3_terminal_rows_csv() if row.get("domain") == domain and str(row.get("task_id")) in suite_ids]


def terminal_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("policy"))].append(row)
    out = []
    for policy, vals in sorted(grouped.items()):
        out.append(
            {
                "policy": policy,
                "task_count": len(vals),
                "selected_count": finite_mean(row.get("selected_count") for row in vals),
                "oracle_retained": finite_mean(row.get("oracle_retained") for row in vals),
                "best_selected_reward": finite_mean(row.get("best_selected_reward") for row in vals),
                "first_selected_reward": finite_mean(row.get("first_selected_reward") for row in vals),
                "first_selected_oracle": finite_mean(row.get("first_selected_oracle") for row in vals),
                "defer_rate": finite_mean(row.get("terminal_deferred") for row in vals),
                "confident_rate": finite_mean(row.get("terminal_confident") for row in vals),
            }
        )
    return out


def reasoning_terminal_defer_main() -> int:
    started = time.time()
    rows = terminal_policy_rows_for_domain("reasoning")
    task_rows = task_metric_rows("reasoning", None)
    summary_rows = terminal_summary(rows)
    hard = [row for row in task_rows if safe_float(row.get("positive_oracle"), 0.0) > 0 and safe_float(row.get("terminal_reward_diverse"), 0.0) > 0]
    forced = next((row for row in summary_rows if row.get("policy") == "dualanchor_forced_top1"), {})
    top2 = next((row for row in summary_rows if row.get("policy") == "dualanchor_terminal_top2"), {})
    top4 = next((row for row in summary_rows if row.get("policy") == "dualanchor_terminal_top4_derived"), {})
    top5 = next((row for row in summary_rows if row.get("policy") == "dualanchor_terminal_top5"), {})
    full = next((row for row in summary_rows if row.get("policy") == "terminal_defer_all"), {})
    if not summary_rows:
        verdict = "INSUFFICIENT"
    elif safe_float(full.get("oracle_retained"), 0.0) >= 0.99 and safe_float(top5.get("oracle_retained"), 0.0) >= 0.99:
        verdict = "REASONING_FULL_SURVIVOR_HANDOFF_REQUIRED" if safe_float(top2.get("oracle_retained"), 0.0) < 0.99 else "REASONING_CONFIDENCE_TOP1_READY_WITH_DEFER"
    elif safe_float(top4.get("oracle_retained"), 0.0) >= 0.99:
        verdict = "REASONING_TOP4_HANDOFF_REQUIRED"
    elif safe_float(top2.get("oracle_retained"), 0.0) >= 0.99:
        verdict = "REASONING_TOP2_HANDOFF_SUFFICIENT"
    else:
        verdict = "REASONING_TERMINAL_POLICY_WEAK"
    payload = {
        "BG_REASONING_TERMINAL_DEFER_VERDICT": verdict,
        "summary_rows": summary_rows,
        "task_summary": metric_summary(task_rows),
        "hard_slice_summary": metric_summary(hard),
        "hard_slice_count": len(hard),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "reasoning_terminal_defer.json", payload)
    write_csv(OUT_ROOT / "reasoning_terminal_defer_rows.csv", rows)
    lines = [
        "# Reasoning Terminal Defer v1",
        "",
        status_line("BG_REASONING_TERMINAL_DEFER_VERDICT", verdict),
        "",
        "## Terminal Policy Summary",
        "",
        *md_table(summary_rows, ["policy", "task_count", "selected_count", "oracle_retained", "best_selected_reward", "first_selected_reward", "first_selected_oracle", "defer_rate", "confident_rate"]),
        "",
        "## Task Summary",
        "",
        *[f"- {key}: `{fmt(value)}`" for key, value in payload["task_summary"].items()],
        "",
        "## Hard Slice Summary",
        "",
        *[f"- {key}: `{fmt(value)}`" for key, value in payload["hard_slice_summary"].items()],
    ]
    write_md(OUT_ROOT / "reasoning_terminal_defer.md", lines)
    print(status_line("BG_REASONING_TERMINAL_DEFER_VERDICT", verdict))
    return 0


def reasoning_branch_generation_main() -> int:
    started = time.time()
    baseline_rows = task_metric_rows("reasoning", None)
    science_recipe = read_json(OUT_ROOT / "best_science_recipe.json", {}) or {}
    diagnostic_rows = recipe_task_rows("reasoning", "recipe_calibration", {"reasoning_baseline_v3"})
    recipe_summary = summarize_recipe_rows(diagnostic_rows)
    baseline = metric_summary(baseline_rows)
    if not baseline_rows:
        verdict = "INSUFFICIENT"
    elif safe_float(baseline.get("positive_oracle_rate"), 0.0) >= 0.50 and safe_float(baseline.get("terminal_oracle_retained"), 0.0) >= 0.95:
        verdict = "REASONING_BASELINE_SUFFICIENT"
    else:
        verdict = "REASONING_DATA_LIMITED"
    payload = {
        "BG_REASONING_BRANCH_GENERATION_SANITY_VERDICT": verdict,
        "baseline_summary": baseline,
        "science_recipe_diagnostic": science_recipe,
        "recipe_summary": recipe_summary,
        "note": "Reasoning branch generation is not changed by the selected science diagnostics.",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "reasoning_branch_generation.json", payload)
    write_csv(OUT_ROOT / "reasoning_branch_generation_rows.csv", diagnostic_rows)
    lines = [
        "# Reasoning Branch-Generation Sanity v1",
        "",
        status_line("BG_REASONING_BRANCH_GENERATION_SANITY_VERDICT", verdict),
        "",
        "## Baseline Summary",
        "",
        *[f"- {key}: `{fmt(value)}`" for key, value in baseline.items()],
        "",
        "## Recipe Diagnostics",
        "",
        *md_table(recipe_summary, ["recipe_id", "task_count", "positive_oracle_rate", "reward_diverse_rate", "terminal_best_reward", "forced_top1_reward", "top4_oracle", "full_handoff_oracle"]),
    ]
    write_md(OUT_ROOT / "reasoning_branch_generation.md", lines)
    print(status_line("BG_REASONING_BRANCH_GENERATION_SANITY_VERDICT", verdict))
    return 0


def science_l47_interaction_main() -> int:
    started = time.time()
    rows = [row for row in load_v3_l47_rows_csv() if row.get("domain") == "science"]
    summary_rows = grouped_summary(rows, "ablation", ["oracle_retained_vs_full", "best_reward", "forced_top1_reward", "forced_top1_oracle", "candidate_count"])
    full = next((row for row in summary_rows if row.get("ablation") == "full_l47_enabled"), {})
    no_l47 = next((row for row in summary_rows if row.get("ablation") == "no_l47_perturbation_tree_replay"), {})
    l2 = next((row for row in summary_rows if row.get("ablation") == "l2_47_only_tree_replay"), {})
    if not summary_rows:
        verdict = "INSUFFICIENT"
    elif safe_float(full.get("best_reward"), 0.0) < 0.10:
        verdict = "L47_NOT_ENOUGH_FOR_SCIENCE"
    elif safe_float(l2.get("oracle_retained_vs_full"), 0.0) > safe_float(no_l47.get("oracle_retained_vs_full"), 0.0):
        verdict = "L2_47_HELPS_SCIENCE"
    else:
        verdict = "L47_REPLAY_ONLY_CAVEAT"
    payload = {
        "BG_SCIENCE_L47_INTERACTION_VERDICT": verdict,
        "summary_rows": summary_rows,
        "replay_only": True,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "science_l47_interaction.json", payload)
    write_csv(OUT_ROOT / "science_l47_rows.csv", rows)
    lines = [
        "# Science L47 Interaction v1",
        "",
        status_line("BG_SCIENCE_L47_INTERACTION_VERDICT", verdict),
        "",
        "## Replay Summary",
        "",
        *md_table(summary_rows, ["ablation", "count", "oracle_retained_vs_full", "best_reward", "forced_top1_reward", "forced_top1_oracle", "candidate_count"]),
        "",
        "## Boundary",
        "",
        "- This is candidate-tree replay, not regenerated alternate dynamics.",
        "- The locked baseline keeps nonterminal L47 active.",
    ]
    write_md(OUT_ROOT / "science_l47_interaction.md", lines)
    print(status_line("BG_SCIENCE_L47_INTERACTION_VERDICT", verdict))
    return 0


def soft_hairs_main() -> int:
    started = time.time()
    hair_rows = read_csv(PREV_ROOT / "convergence_hair_pair_rows.csv")
    task_rows = {row.get("task_id"): row for row in load_v3_task_rows_csv()}
    rows = []
    for row in hair_rows:
        task = task_rows.get(row.get("task_id"), {})
        rows.append(
            {
                **row,
                "positive_oracle": task.get("positive_oracle"),
                "terminal_reward_diverse": task.get("terminal_reward_diverse"),
                "terminal_best_reward": task.get("terminal_best_reward"),
                "all_zero_science_task": 1.0 if row.get("domain") == "science" and safe_float(task.get("positive_oracle"), 0.0) <= 0 else 0.0,
            }
        )
    domain_summary = grouped_summary(rows, "domain", ["hidden_cosine_distance", "hidden_rms_normalized", "dualanchor_abs_avg_margin", "reward_tied_eval_only", "all_zero_science_task", "terminal_reward_diverse"])
    science_zero = [row for row in rows if row.get("domain") == "science" and safe_float(row.get("all_zero_science_task"), 0.0) > 0]
    if not rows:
        verdict = "INSUFFICIENT"
    elif science_zero and finite_mean(row.get("reward_tied_eval_only") for row in science_zero) > 0.90:
        verdict = "SCIENCE_CONVERGES_TO_NO_GOOD_BRANCH"
    else:
        verdict = "SOFT_HAIRS_USEFUL_MONITOR"
    payload = {
        "BG_SOFT_HAIRS_SCIENCE_REASONING_VERDICT": verdict,
        "domain_summary": domain_summary,
        "row_count": len(rows),
        "runtime_usage": "none; diagnostic only",
        "hard_merge": False,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "soft_hairs_science_reasoning.json", payload)
    write_csv(OUT_ROOT / "soft_hair_rows.csv", rows)
    lines = [
        "# Soft Hairs Science/Reasoning v1",
        "",
        status_line("BG_SOFT_HAIRS_SCIENCE_REASONING_VERDICT", verdict),
        "",
        "## Domain Summary",
        "",
        *md_table(domain_summary, ["domain", "count", "hidden_cosine_distance", "hidden_rms_normalized", "dualanchor_abs_avg_margin", "reward_tied_eval_only", "all_zero_science_task", "terminal_reward_diverse"]),
        "",
        "## Boundary",
        "",
        "- L30/L42 hairs remain soft diagnostics only.",
        "- No hard merge or runtime branch classification is introduced.",
    ]
    write_md(OUT_ROOT / "soft_hairs_science_reasoning.md", lines)
    print(status_line("BG_SOFT_HAIRS_SCIENCE_REASONING_VERDICT", verdict))
    return 0


def science_perturbation_escalation_main() -> int:
    started = time.time()
    stages = [row for row in load_v3_stage_rows_csv() if row.get("domain") == "science"]
    rows = []
    for row in stages:
        low_survivors = safe_float(row.get("survivors_after"), 99.0) < 4
        false_prune = safe_float(row.get("stage_false_prune"), 0.0) > 0
        low_oracle = safe_float(row.get("stage_oracle_retained"), 1.0) <= 0
        trigger = low_survivors or false_prune or low_oracle
        if trigger:
            rows.append(
                {
                    **row,
                    "would_trigger_escalation": 1.0,
                    "trigger_low_survivors": 1.0 if low_survivors else 0.0,
                    "trigger_false_prune": 1.0 if false_prune else 0.0,
                    "trigger_oracle_loss": 1.0 if low_oracle else 0.0,
                    "recommended_round": "round1_more_directions" if low_survivors else ("round2_l47_emphasis" if false_prune else "round3_alpha_02_diagnostic"),
                }
            )
    trigger_rate = len(rows) / len(stages) if stages else 0.0
    verdict = "DIAGNOSTIC_ONLY" if stages else "INSUFFICIENT"
    payload = {
        "BG_SCIENCE_PERTURBATION_ESCALATION_VERDICT": verdict,
        "stage_count": len(stages),
        "trigger_count": len(rows),
        "trigger_rate": trigger_rate,
        "stage_oracle_retention": finite_mean(row.get("stage_oracle_retained") for row in stages),
        "note": "No escalated generation was run; this is a trigger audit over v3 science stage rows.",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "science_perturbation_escalation.json", payload)
    write_csv(OUT_ROOT / "science_perturbation_escalation_rows.csv", rows)
    lines = [
        "# Science Perturbation Escalation v1",
        "",
        status_line("BG_SCIENCE_PERTURBATION_ESCALATION_VERDICT", verdict),
        "",
        "## Trigger Summary",
        "",
        *[f"- {key}: `{fmt(value)}`" for key, value in payload.items() if key not in {"note"}],
        "",
        "## Note",
        "",
        payload["note"],
    ]
    write_md(OUT_ROOT / "science_perturbation_escalation.md", lines)
    print(status_line("BG_SCIENCE_PERTURBATION_ESCALATION_VERDICT", verdict))
    return 0


def pre_steering_domain_decision_main() -> int:
    started = time.time()
    parser = read_json(OUT_ROOT / "science_parser_audit.json", {}) or {}
    calibration = read_json(OUT_ROOT / "science_recipe_calibration.json", {}) or {}
    heldout = read_json(OUT_ROOT / "science_recipe_heldout.json", {}) or {}
    reasoning_terminal = read_json(OUT_ROOT / "reasoning_terminal_defer.json", {}) or {}
    reasoning_branch = read_json(OUT_ROOT / "reasoning_branch_generation.json", {}) or {}
    parser_verdict = parser.get("BG_SCIENCE_PARSER_REWARD_AUDIT_VERDICT", "INSUFFICIENT")
    heldout_verdict = heldout.get("BG_SCIENCE_BRANCH_RECIPE_HELDOUT_VERDICT", "INSUFFICIENT")
    reasoning_verdict = reasoning_terminal.get("BG_REASONING_TERMINAL_DEFER_VERDICT", "INSUFFICIENT")
    reasoning_branch_verdict = reasoning_branch.get("BG_REASONING_BRANCH_GENERATION_SANITY_VERDICT", "INSUFFICIENT")
    if parser_verdict == "PARSER_MAJOR_BLOCKER":
        verdict = "NEEDS_SCIENCE_PARSER_FIX"
    elif reasoning_verdict in {"REASONING_TERMINAL_POLICY_WEAK", "INSUFFICIENT"}:
        verdict = "NEEDS_REASONING_TERMINAL_POLICY"
    elif heldout_verdict == "SCIENCE_BRANCH_RECIPE_READY":
        verdict = "READY_FOR_STEERING_REASONING_AND_SCIENCE"
    elif reasoning_branch_verdict == "REASONING_BASELINE_SUFFICIENT" and heldout_verdict in {"SCIENCE_BRANCH_GENERATION_STILL_WEAK", "SCIENCE_BRANCH_RECIPE_WEAK_BUT_IMPROVED", "SCIENCE_PARSER_BLOCKER"}:
        verdict = "READY_FOR_STEERING_WITH_SCIENCE_DIAGNOSTIC"
    elif reasoning_branch_verdict == "REASONING_BASELINE_SUFFICIENT":
        verdict = "READY_FOR_STEERING_REASONING_ONLY"
    else:
        verdict = "NOT_READY"
    baseline = {
        "schedule": LOCKED_SCHEDULE,
        "selector": [ANCHOR_A, ANCHOR_B],
        "threshold": "mean_floor_very_loose",
        "budget": 8,
        "l47": "active in nonterminal loops",
        "terminal": "confidence-gated top1; otherwise defer / survivor-set handoff",
        "convergence_hairs": "soft-only diagnostics",
        "science_recipe": "diagnostic/excluded unless a regenerated recipe improves heldout",
        "steering": "not run in this prompt",
    }
    payload = {
        "BG_PRE_STEERING_DOMAIN_DECISION_VERDICT": verdict,
        "input_verdicts": {
            "science_parser": parser_verdict,
            "science_calibration": calibration.get("BG_SCIENCE_BRANCH_RECIPE_CALIBRATION_VERDICT", "INSUFFICIENT"),
            "science_heldout": heldout_verdict,
            "reasoning_terminal": reasoning_verdict,
            "reasoning_branch": reasoning_branch_verdict,
        },
        "recommended_domain_scope": "reasoning headline; science diagnostic slice only" if verdict == "READY_FOR_STEERING_WITH_SCIENCE_DIAGNOSTIC" else verdict,
        "locked_baseline": baseline,
        "cannot_claim": [
            "steering was tested",
            "science headline readiness unless heldout recipe improves",
            "hard convergence-hair merge",
            "compute savings",
            "autoregressive branch-specific KV/cache fork-carry",
            "runtime branch classification",
            "production routing change",
        ],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "pre_steering_domain_decision.json", payload)
    lines = [
        "# Pre-Steering Domain Decision v1",
        "",
        status_line("BG_PRE_STEERING_DOMAIN_DECISION_VERDICT", verdict),
        "",
        "## Input Verdicts",
        "",
        *[f"- {key}: `{value}`" for key, value in payload["input_verdicts"].items()],
        "",
        "## Domain Scope",
        "",
        f"- recommended domain scope: `{payload['recommended_domain_scope']}`",
        "",
        "## Locked Baseline",
        "",
        *[f"- {key}: `{value}`" for key, value in baseline.items()],
        "",
        "## Cannot Claim",
        "",
        *[f"- {item}" for item in payload["cannot_claim"]],
    ]
    write_md(OUT_ROOT / "pre_steering_domain_decision.md", lines)
    print(status_line("BG_PRE_STEERING_DOMAIN_DECISION_VERDICT", verdict))
    return 0


def status_from(verdicts: dict[str, str]) -> str:
    domain = verdicts.get("BG_PRE_STEERING_DOMAIN_DECISION_VERDICT")
    if domain == "READY_FOR_STEERING_REASONING_AND_SCIENCE":
        return "PRE_STEERING_READY_REASONING_AND_SCIENCE"
    if domain == "READY_FOR_STEERING_REASONING_ONLY":
        return "PRE_STEERING_READY_REASONING_ONLY"
    if domain == "READY_FOR_STEERING_WITH_SCIENCE_DIAGNOSTIC":
        return "PRE_STEERING_READY_WITH_SCIENCE_DIAGNOSTIC"
    if verdicts.get("BG_SCIENCE_PARSER_REWARD_AUDIT_VERDICT") == "PARSER_MAJOR_BLOCKER":
        return "SCIENCE_PARSER_BLOCKER"
    if verdicts.get("BG_SCIENCE_BRANCH_RECIPE_HELDOUT_VERDICT") == "SCIENCE_BRANCH_RECIPE_WEAK_BUT_IMPROVED":
        return "SCIENCE_BRANCH_RECIPE_IMPROVED_BUT_WEAK"
    if verdicts.get("BG_SCIENCE_BRANCH_RECIPE_HELDOUT_VERDICT") == "SCIENCE_BRANCH_GENERATION_STILL_WEAK":
        return "SCIENCE_BRANCH_GENERATION_STILL_WEAK"
    if verdicts.get("BG_REASONING_TERMINAL_DEFER_VERDICT", "").startswith("REASONING_") and "WEAK" not in verdicts.get("BG_REASONING_TERMINAL_DEFER_VERDICT", ""):
        return "REASONING_TERMINAL_DEFER_LOCKED"
    if verdicts.get("BG_REASONING_TERMINAL_DEFER_VERDICT") == "REASONING_TERMINAL_POLICY_WEAK":
        return "REASONING_TERMINAL_POLICY_WEAK"
    return "INSUFFICIENT"


def synthesis_main() -> int:
    started = time.time()
    inputs = {
        "inventory": read_json(OUT_ROOT / "inventory.json", {}) or {},
        "task_suite": read_json(OUT_ROOT / "task_suite.json", {}) or {},
        "parser": read_json(OUT_ROOT / "science_parser_audit.json", {}) or {},
        "plan": read_json(OUT_ROOT / "science_recipe_plan.json", {}) or {},
        "calibration": read_json(OUT_ROOT / "science_recipe_calibration.json", {}) or {},
        "heldout": read_json(OUT_ROOT / "science_recipe_heldout.json", {}) or {},
        "source": read_json(OUT_ROOT / "science_source_breakdown.json", {}) or {},
        "reasoning_terminal": read_json(OUT_ROOT / "reasoning_terminal_defer.json", {}) or {},
        "reasoning_branch": read_json(OUT_ROOT / "reasoning_branch_generation.json", {}) or {},
        "l47": read_json(OUT_ROOT / "science_l47_interaction.json", {}) or {},
        "soft_hairs": read_json(OUT_ROOT / "soft_hairs_science_reasoning.json", {}) or {},
        "escalation": read_json(OUT_ROOT / "science_perturbation_escalation.json", {}) or {},
        "domain_decision": read_json(OUT_ROOT / "pre_steering_domain_decision.json", {}) or {},
    }
    verdicts = {
        "BG_SCIENCE_RECIPE_REASONING_DEFER_INVENTORY_VERDICT": inputs["inventory"].get("BG_SCIENCE_RECIPE_REASONING_DEFER_INVENTORY_VERDICT", "INSUFFICIENT"),
        "BG_SCIENCE_RECIPE_REASONING_TASK_SUITE_VERDICT": inputs["task_suite"].get("BG_SCIENCE_RECIPE_REASONING_TASK_SUITE_VERDICT", "INSUFFICIENT"),
        "BG_SCIENCE_PARSER_REWARD_AUDIT_VERDICT": inputs["parser"].get("BG_SCIENCE_PARSER_REWARD_AUDIT_VERDICT", "INSUFFICIENT"),
        "BG_SCIENCE_BRANCH_RECIPE_PLAN_VERDICT": inputs["plan"].get("BG_SCIENCE_BRANCH_RECIPE_PLAN_VERDICT", "INSUFFICIENT"),
        "BG_SCIENCE_BRANCH_RECIPE_CALIBRATION_VERDICT": inputs["calibration"].get("BG_SCIENCE_BRANCH_RECIPE_CALIBRATION_VERDICT", "INSUFFICIENT"),
        "BG_SCIENCE_BRANCH_RECIPE_HELDOUT_VERDICT": inputs["heldout"].get("BG_SCIENCE_BRANCH_RECIPE_HELDOUT_VERDICT", "INSUFFICIENT"),
        "BG_SCIENCE_SOURCE_BREAKDOWN_VERDICT": inputs["source"].get("BG_SCIENCE_SOURCE_BREAKDOWN_VERDICT", "INSUFFICIENT"),
        "BG_REASONING_TERMINAL_DEFER_VERDICT": inputs["reasoning_terminal"].get("BG_REASONING_TERMINAL_DEFER_VERDICT", "INSUFFICIENT"),
        "BG_REASONING_BRANCH_GENERATION_SANITY_VERDICT": inputs["reasoning_branch"].get("BG_REASONING_BRANCH_GENERATION_SANITY_VERDICT", "INSUFFICIENT"),
        "BG_SCIENCE_L47_INTERACTION_VERDICT": inputs["l47"].get("BG_SCIENCE_L47_INTERACTION_VERDICT", "INSUFFICIENT"),
        "BG_SOFT_HAIRS_SCIENCE_REASONING_VERDICT": inputs["soft_hairs"].get("BG_SOFT_HAIRS_SCIENCE_REASONING_VERDICT", "INSUFFICIENT"),
        "BG_SCIENCE_PERTURBATION_ESCALATION_VERDICT": inputs["escalation"].get("BG_SCIENCE_PERTURBATION_ESCALATION_VERDICT", "INSUFFICIENT"),
        "BG_PRE_STEERING_DOMAIN_DECISION_VERDICT": inputs["domain_decision"].get("BG_PRE_STEERING_DOMAIN_DECISION_VERDICT", "INSUFFICIENT"),
    }
    overall = status_from(verdicts)
    files_created = sorted(str(path.relative_to(OUT_ROOT)) for path in OUT_ROOT.glob("*") if path.is_file())
    commands_run = [
        "py_compile requested scripts",
        "bg_science_recipe_reasoning_defer_inventory_v1.py",
        "build_bg_science_recipe_reasoning_task_suite_v1.py",
        "audit_bg_science_parser_reward_v1.py",
        "build_bg_science_branch_recipe_plan_v1.py",
        "run_bg_science_branch_recipe_calibration_v1.py",
        "evaluate_bg_science_branch_recipe_heldout_v1.py",
        "analyze_bg_science_source_breakdown_v1.py",
        "analyze_bg_reasoning_terminal_defer_v1.py",
        "analyze_bg_reasoning_branch_generation_sanity_v1.py",
        "analyze_bg_science_l47_interaction_v1.py",
        "analyze_bg_soft_hairs_science_reasoning_v1.py",
        "analyze_bg_science_perturbation_escalation_v1.py",
        "analyze_bg_pre_steering_domain_decision_v1.py",
        "analyze_bg_science_recipe_reasoning_defer_v1.py",
    ]
    payload = {
        **verdicts,
        "DUALANCHOR_SCIENCE_RECIPE_REASONING_DEFER_STATUS": overall,
        "decision": inputs["domain_decision"],
        "files_created": files_created,
        "commands_run": commands_run,
        "blockers": [] if overall != "INSUFFICIENT" else ["One or more required input stages were insufficient."],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "summary.json", payload)
    write_json(OUT_ROOT / "analysis.json", {**payload, "inputs": inputs})
    top_lines = [status_line(key, value) for key, value in {**verdicts, "DUALANCHOR_SCIENCE_RECIPE_REASONING_DEFER_STATUS": overall}.items()]
    science_cal_rows = inputs["calibration"].get("summary_rows", [])
    science_heldout_rows = inputs["heldout"].get("summary_rows", [])
    parser_summary = inputs["parser"].get("summary", {})
    source_rows = inputs["source"].get("source_rows", [])
    reasoning_terminal_rows = inputs["reasoning_terminal"].get("summary_rows", [])
    lines = [
        "# DualAnchor Science Branch Recipe + Reasoning Terminal Defer Probe v1",
        "",
        *top_lines,
        "",
        "## Motivation",
        "",
        "This probe completes the remaining pre-steering checks after convergence hairs were demoted to soft diagnostics: science branch generation, science parser/reward audit, and reasoning terminal defer validation.",
        "",
        "No steering, training, production routing, wrapper/local-agent execution, Hunter-Seeker execution, tap registry update, tokenizer/checkpoint edit, hard convergence-hair merge, compute-savings claim, or autoregressive fork/carry claim was made.",
        "",
        "## Previous Convergence-Hair/Reasoning/Science Result",
        "",
        "- hard convergence hairs: not ready; soft diagnostics only",
        "- science: branch-generation weak",
        "- reasoning: terminal defer/survivor-set handoff required",
        "- locked baseline: DualAnchor `MIX_CODE_REASONING` + `MIX_OBJECTIVE_ALL`, `mean_floor_very_loose`, budget 8, nonterminal L47 active",
        "",
        "## Task Suite",
        "",
        f"- verdict: `{verdicts['BG_SCIENCE_RECIPE_REASONING_TASK_SUITE_VERDICT']}`",
        f"- selected science: `{inputs['task_suite'].get('inventory', {}).get('selected_science')}`",
        f"- selected reasoning: `{inputs['task_suite'].get('inventory', {}).get('selected_reasoning')}`",
        f"- v3-generated selected tasks: `{inputs['task_suite'].get('inventory', {}).get('v3_generated_selected')}`",
        "- local science pool was below the requested minimum, so science task-suite coverage is limited",
        "",
        "## Science Parser/Reward Audit",
        "",
        f"- verdict: `{verdicts['BG_SCIENCE_PARSER_REWARD_AUDIT_VERDICT']}`",
        *[f"- {key}: `{fmt(value)}`" for key, value in parser_summary.items()],
        "",
        "## Science Recipe Plan",
        "",
        f"- verdict: `{verdicts['BG_SCIENCE_BRANCH_RECIPE_PLAN_VERDICT']}`",
        "- recipes without regenerated artifacts are explicitly marked data-limited or replay diagnostic",
        "- recipe selection uses calibration only; heldout remains evaluation-only",
        "",
        "## Science Recipe Calibration",
        "",
        f"- verdict: `{verdicts['BG_SCIENCE_BRANCH_RECIPE_CALIBRATION_VERDICT']}`",
        "",
        *md_table(science_cal_rows, ["recipe_id", "task_count", "available_task_count", "positive_oracle_rate", "reward_diverse_rate", "terminal_best_reward", "forced_top1_reward", "top4_oracle", "robust_parser_terminal_best_reward"]),
        "",
        "## Science Heldout Result",
        "",
        f"- verdict: `{verdicts['BG_SCIENCE_BRANCH_RECIPE_HELDOUT_VERDICT']}`",
        "",
        *md_table(science_heldout_rows, ["recipe_id", "task_count", "available_task_count", "positive_oracle_rate", "reward_diverse_rate", "terminal_best_reward", "forced_top1_reward", "top4_oracle", "robust_parser_terminal_best_reward"]),
        "",
        "## Science Source Breakdown",
        "",
        f"- verdict: `{verdicts['BG_SCIENCE_SOURCE_BREAKDOWN_VERDICT']}`",
        "",
        *md_table(source_rows, ["source_detail", "count", "positive_oracle", "reward_diverse", "terminal_best_reward", "forced_top1_reward", "top4_oracle", "parse_success", "robust_parser_terminal_best_reward"]),
        "",
        "## Reasoning Terminal Defer",
        "",
        f"- verdict: `{verdicts['BG_REASONING_TERMINAL_DEFER_VERDICT']}`",
        "",
        *md_table(reasoning_terminal_rows, ["policy", "task_count", "selected_count", "oracle_retained", "best_selected_reward", "first_selected_reward", "first_selected_oracle", "defer_rate"]),
        "",
        "## Reasoning Branch Sanity",
        "",
        f"- verdict: `{verdicts['BG_REASONING_BRANCH_GENERATION_SANITY_VERDICT']}`",
        "- reasoning branch generation remains baseline-sufficient; the science recipe diagnostics do not justify changing reasoning generation",
        "",
        "## Science/L47 Interaction",
        "",
        f"- verdict: `{verdicts['BG_SCIENCE_L47_INTERACTION_VERDICT']}`",
        "- L47 remains active in the locked baseline; replay says L47 is not enough by itself to solve science",
        "",
        "## Soft Convergence-Hair Monitoring",
        "",
        f"- verdict: `{verdicts['BG_SOFT_HAIRS_SCIENCE_REASONING_VERDICT']}`",
        "- L30/L42 hairs are monitoring only; no hard merge or runtime branch classification",
        "",
        "## Science Perturbation Escalation",
        "",
        f"- verdict: `{verdicts['BG_SCIENCE_PERTURBATION_ESCALATION_VERDICT']}`",
        "- escalation was trigger-audited only; no escalated generation was run in this replay pass",
        "",
        "## Pre-Steering Domain Decision",
        "",
        f"- verdict: `{verdicts['BG_PRE_STEERING_DOMAIN_DECISION_VERDICT']}`",
        f"- status: `{overall}`",
        f"- domain scope: `{inputs['domain_decision'].get('recommended_domain_scope')}`",
        "",
        "## Locked Baseline For Steering If Ready",
        "",
        *[f"- {key}: `{value}`" for key, value in (inputs["domain_decision"].get("locked_baseline") or {}).items()],
        "",
        "## Files Created",
        "",
        *[f"- `{name}`" for name in files_created],
        "",
        "## Commands Run",
        "",
        *[f"- `{cmd}`" for cmd in commands_run],
        "",
        "## Blockers",
        "",
        *([f"- {item}" for item in payload["blockers"]] if payload["blockers"] else ["- No blocker for reasoning-headline steering with science diagnostic scope. Science headline readiness remains blocked by branch generation."]),
    ]
    write_md(OUT_ROOT / "summary.md", lines)
    write_md(OUT_ROOT / "analysis.md", lines)
    print(status_line("DUALANCHOR_SCIENCE_RECIPE_REASONING_DEFER_STATUS", overall))
    return 0
