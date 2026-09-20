"""Shared helpers for DualAnchor science + reasoning repair v2.

This repair pass is pre-steering only. It may run regenerated cumulative-hook
branch recipe experiments, but it does not train Ouro, modify checkpoints or
tokenizers, update tap registries, run wrapper/local-agent code, import
Hunter-Seeker modules, apply steering, or claim compute savings / true
autoregressive fork-carry.
"""
from __future__ import annotations

import ast
import csv
import json
import math
import os
import re
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import torch

from bg_branch_generator_v1_common import ALPHA_VALUE, BASIS_BANK_PT, load_audit_plan, load_pt, make_family_deltas
from bg_hidden_origin_tap_common import PROBE_ROOT
from bg_science_mcq_parser_v2 import PARSER_IDS, option_map as parser_option_map, parse_with
from run_bg_dualanchor_architecture_looped_stratified_probe_v2 import (
    NONTERMINAL_STAGES,
    terminal_policy_rows,
    task_summary as v2_task_summary,
)
from run_bg_dualanchor_recursive_lineage_probe_v1 import (
    compact_row,
    evaluate_branch,
    finite_mean as runner_finite_mean,
    make_hook_entry,
    threshold_survive,
    terminal_select,
)
from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor

import run_bg_dualanchor_all_loop_audit_v1 as base


SHORT_NAME = "dualanchor_science_reasoning_repair_v2"
OUT_ROOT = PROBE_ROOT / "bg_dualanchor_science_reasoning_repair_v2_2026-06-01"
V1_ROOT = PROBE_ROOT / "bg_dualanchor_science_branch_recipe_reasoning_defer_v1_2026-05-31"
HAIRS_ROOT = PROBE_ROOT / "bg_dualanchor_convergence_hairs_reasoning_science_v1_2026-05-31"
V3_ROOT = PROBE_ROOT / "bg_dualanchor_architecture_looped_stratified_probe_v3_2026-05-31"

PROGRESS_ROOT = OUT_ROOT / "progress"
CANDIDATE_TREE_ROOT = OUT_ROOT / "candidate_trees"
GENERATED_PT = OUT_ROOT / "science_recipe_v2_generated.pt"
GENERATED_ROWS_CSV = OUT_ROOT / "science_recipe_v2_generated_rows.csv"
GENERATED_TASK_ROWS_CSV = OUT_ROOT / "science_recipe_v2_generated_task_rows.csv"
GENERATED_TERMINAL_ROWS_CSV = OUT_ROOT / "science_recipe_v2_generated_terminal_rows.csv"
GENERATED_STAGE_ROWS_CSV = OUT_ROOT / "science_recipe_v2_generated_stage_rows.csv"
LINEAGE_INCREMENTAL_CSV = OUT_ROOT / "lineage_rows_incremental.csv"
COMPLETED_TASKS_JSON = OUT_ROOT / "completed_task_ids.json"
COMPLETED_RECIPES_JSON = OUT_ROOT / "completed_recipe_ids.json"
COMPLETED_SOURCES_JSON = OUT_ROOT / "completed_source_ids.json"
PARTIAL_SUMMARY_JSON = OUT_ROOT / "partial_summary.json"

SELECTED_POLICY = "dualanchor_adaptive_branch_anchor_light_AntisymLinear"
ANCHOR_A = "MIX_CODE_REASONING"
ANCHOR_B = "MIX_OBJECTIVE_ALL"
LOCKED_SCHEDULE = "L1_24 -> L1_36 -> L1_47 -> L2_24 -> L2_36 -> L2_47 -> L3_24 -> L3_36 -> L3_47 -> L4_24 -> L4_36 -> terminal L4_47"

SCIENCE_BASELINE_POSITIVE_ORACLE = 0.0833
SCIENCE_BASELINE_TERMINAL_BEST = 0.0500


def ensure_root() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    PROGRESS_ROOT.mkdir(parents=True, exist_ok=True)
    CANDIDATE_TREE_ROOT.mkdir(parents=True, exist_ok=True)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() <= 16:
            return value.detach().cpu().tolist()
        return {"tensor_shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, set):
        return sorted(value)
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
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def finite_mean(values: Iterable[Any]) -> float:
    xs = []
    for value in values:
        x = safe_float(value)
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


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:180]


def row_reward(row: dict[str, Any]) -> float:
    for key in ("deterministic_reward", "reward", "final_reward", "correctness", "oracle_reward"):
        x = safe_float(row.get(key), float("nan"))
        if math.isfinite(x):
            return x
    if truthy(row.get("deterministic_correct")) or truthy(row.get("correct")):
        return 1.0
    return 0.0


def source_detail(task_id: str, source_dataset: Any) -> str:
    text = str(task_id).lower()
    for name in ("high_school_chemistry", "high_school_physics", "anatomy", "biology"):
        if name in text:
            return f"mmlu_{name}"
    source = str(source_dataset or "").lower()
    if source == "sciq":
        return "sciq"
    if source == "openbookqa":
        return "openbookqa"
    if source == "mmlu":
        return "mmlu_other"
    return source or "unknown"


def audit_plan_tasks() -> list[dict[str, Any]]:
    return [dict(row) for row in (load_audit_plan().get("tasks") or []) if row.get("correct_option")]


def task_by_id() -> dict[str, dict[str, Any]]:
    return {str(row.get("task_id")): row for row in audit_plan_tasks()}


def task_sort_key(row: dict[str, Any]) -> tuple[int, int, int, float, str]:
    split_rank = {"heldout": 0, "val": 1, "train": 2}.get(str(row.get("split")), 9)
    likely = 1 if truthy(row.get("heldout_likely_non_tie")) else 0
    prior = int(row.get("prior_non_tie_pairs") or 0)
    priority = safe_float(row.get("priority_score_v4") or row.get("priority_score"), 0.0)
    return (split_rank, -likely, -prior, -priority, str(row.get("task_id")))


def v3_task_rows() -> list[dict[str, str]]:
    return read_csv(V3_ROOT / "task_rows.csv")


def v3_terminal_rows() -> list[dict[str, str]]:
    return read_csv(V3_ROOT / "terminal_policy_rows.csv")


def v3_stage_rows() -> list[dict[str, str]]:
    return read_csv(V3_ROOT / "stage_decisions.csv")


def load_v3_pt() -> dict[str, Any]:
    path = V3_ROOT / "architecture_looped_rows.pt"
    if not path.exists():
        return {}
    return torch.load(path, map_location="cpu", weights_only=False)


def load_generated() -> dict[str, Any]:
    if GENERATED_PT.exists():
        return torch.load(GENERATED_PT, map_location="cpu", weights_only=False)
    return {"rows": [], "stage_rows": [], "task_rows": [], "terminal_rows": [], "errors": []}


def save_generated(payload: dict[str, Any]) -> None:
    ensure_root()
    torch.save(payload, GENERATED_PT)
    write_csv(GENERATED_ROWS_CSV, [compact_row(row) for row in payload.get("rows", [])])
    write_csv(GENERATED_STAGE_ROWS_CSV, payload.get("stage_rows", []))
    write_csv(GENERATED_TASK_ROWS_CSV, payload.get("task_rows", []))
    write_csv(GENERATED_TERMINAL_ROWS_CSV, payload.get("terminal_rows", []))


def source_split_role(task: dict[str, Any], domain: str) -> str:
    split = str(task.get("split"))
    if domain == "science":
        if split == "heldout":
            return "science_recipe_heldout"
        if split == "val":
            return "parser_calibration"
        return "science_recipe_calibration"
    if split == "heldout":
        return "reasoning_terminal_heldout"
    return "reasoning_terminal_calibration"


def build_task_suite_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tasks = audit_plan_tasks()
    science = sorted([row for row in tasks if str(row.get("domain")) == "science"], key=task_sort_key)
    reasoning = sorted([row for row in tasks if str(row.get("domain")) == "reasoning"], key=task_sort_key)[:32]
    selected = science + reasoning
    v3_ids = {str(row.get("task_id")) for row in v3_task_rows()}
    rows = []
    for idx, task in enumerate(selected):
        domain = str(task.get("domain"))
        source = source_detail(str(task.get("task_id")), task.get("source_dataset"))
        rows.append(
            {
                "index": idx,
                "task_id": task.get("task_id"),
                "domain": domain,
                "source_dataset": task.get("source_dataset"),
                "source_detail": source,
                "split": task.get("split"),
                "repair_split": source_split_role(task, domain),
                "correct_option": task.get("correct_option"),
                "options": task.get("options"),
                "parser_type": "mcq",
                "v3_generated": str(task.get("task_id")) in v3_ids,
                "heldout_likely_non_tie": truthy(task.get("heldout_likely_non_tie")),
                "prior_non_tie_pairs": int(task.get("prior_non_tie_pairs") or 0),
                "priority_score": safe_float(task.get("priority_score_v4") or task.get("priority_score"), 0.0),
            }
        )
    inventory = {
        "selected_count": len(rows),
        "science_count": sum(1 for row in rows if row["domain"] == "science"),
        "reasoning_count": sum(1 for row in rows if row["domain"] == "reasoning"),
        "science_source_counts": dict(Counter(row["source_detail"] for row in rows if row["domain"] == "science")),
        "reasoning_source_counts": dict(Counter(row["source_dataset"] for row in rows if row["domain"] == "reasoning")),
        "repair_split_counts": dict(Counter(row["repair_split"] for row in rows)),
        "missing_requested_science_sources": ["science_openbookqa", "mmlu_biology"],
        "v3_generated_selected": sum(1 for row in rows if row["v3_generated"]),
    }
    return rows, inventory


def suite_rows() -> list[dict[str, Any]]:
    data = read_json(OUT_ROOT / "task_suite.json", {}) or {}
    return list(data.get("task_rows") or [])


def suite_by_id() -> dict[str, dict[str, Any]]:
    return {str(row.get("task_id")): row for row in suite_rows()}


def option_map_for_task(task_id: str) -> dict[str, str]:
    meta = task_by_id().get(task_id) or suite_by_id().get(task_id) or {}
    options = meta.get("options") or {}
    if isinstance(options, str):
        options = parse_literal(options, {})
    return parser_option_map(options)


def parser_eval_row(candidate: dict[str, Any], parser_id: str) -> dict[str, Any]:
    task_id = str(candidate.get("task_id"))
    suite = suite_by_id().get(task_id, {})
    source = suite.get("source_detail") or source_detail(task_id, candidate.get("source_dataset"))
    correct = str(candidate.get("correct_option") or task_by_id().get(task_id, {}).get("correct_option") or "").upper().strip()
    result = parse_with(
        parser_id,
        str(candidate.get("output_text") or ""),
        option_map_for_task(task_id),
        candidate.get("parsed_answer"),
        truthy(candidate.get("parse_success")),
        source=source,
    )
    parsed = result.parsed_option
    return {
        "task_id": task_id,
        "branch_id": candidate.get("branch_id"),
        "domain": candidate.get("domain"),
        "source_dataset": candidate.get("source_dataset"),
        "source_detail": source,
        "split": candidate.get("split"),
        "repair_split": suite.get("repair_split"),
        "parser_id": parser_id,
        "correct_option": correct,
        "parsed_option": parsed,
        "parse_success": result.parse_success,
        "parse_failure_reason": result.parse_failure_reason,
        "ambiguity_flag": result.ambiguity_flag,
        "confidence": result.confidence,
        "span_used": result.span_used,
        "risk_flags": result.risk_flags,
        "parser_correct": bool(result.parse_success and parsed == correct),
        "parser_wrong": bool(result.parse_success and parsed and parsed != correct),
        "strict_parse_success": truthy(candidate.get("parse_success")),
        "strict_parsed_answer": candidate.get("parsed_answer"),
        "strict_reward": row_reward(candidate),
        "parser_reward": 1.0 if result.parse_success and parsed == correct else (-0.2 if result.parse_success else 0.0),
        "strict_abstained": not truthy(candidate.get("parse_success")),
        "false_positive_risk_flag": bool(result.parse_success and parsed == correct and result.ambiguity_flag),
        "wrong_option_parse_from_abstain": bool((not truthy(candidate.get("parse_success"))) and result.parse_success and parsed != correct),
    }


def all_science_candidate_rows(include_generated: bool = True) -> list[dict[str, Any]]:
    rows = []
    v3 = load_v3_pt()
    rows.extend([row for row in v3.get("rows", []) if row.get("domain") == "science"])
    if include_generated and GENERATED_PT.exists():
        gen = load_generated()
        rows.extend([row for row in gen.get("rows", []) if row.get("domain") == "science"])
    return rows


def parser_validation_rows(parser_ids: Sequence[str] = PARSER_IDS, include_generated: bool = True) -> list[dict[str, Any]]:
    candidates = all_science_candidate_rows(include_generated=include_generated)
    suite_ids = set(suite_by_id())
    if suite_ids:
        candidates = [row for row in candidates if str(row.get("task_id")) in suite_ids]
    rows = []
    for candidate in candidates:
        for parser_id in parser_ids:
            rows.append(parser_eval_row(candidate, parser_id))
    return rows


def parser_summary(rows: Sequence[dict[str, Any]], split_filter: str | None = None) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if split_filter and row.get("repair_split") != split_filter:
            continue
        grouped[str(row.get("parser_id"))].append(row)
    out = []
    for parser_id, vals in sorted(grouped.items()):
        task_best: dict[str, list[float]] = defaultdict(list)
        for row in vals:
            task_best[str(row.get("task_id"))].append(safe_float(row.get("parser_reward"), 0.0))
        best_rewards = [max(v) for v in task_best.values() if v]
        out.append(
            {
                "parser_id": parser_id,
                "candidate_count": len(vals),
                "task_count": len(task_best),
                "parse_success_rate": finite_mean(row.get("parse_success") for row in vals),
                "parser_correct_rate": finite_mean(row.get("parser_correct") for row in vals),
                "parser_wrong_rate": finite_mean(row.get("parser_wrong") for row in vals),
                "ambiguity_rate": finite_mean(row.get("ambiguity_flag") for row in vals),
                "false_positive_risk_rate": finite_mean(row.get("false_positive_risk_flag") for row in vals),
                "wrong_option_parse_from_abstain_rate": finite_mean(row.get("wrong_option_parse_from_abstain") for row in vals),
                "positive_oracle_rate": finite_mean(1.0 if reward > 0 else 0.0 for reward in best_rewards),
                "terminal_best_reward": finite_mean(best_rewards),
                "missed_correct_recovered_count": sum(1 for row in vals if row.get("parser_correct") and safe_float(row.get("strict_reward"), 0.0) <= 0),
            }
        )
    return out


def select_parser(summary_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    strict = next((row for row in summary_rows if row.get("parser_id") == "strict_letter_current"), {})
    candidates = []
    for row in summary_rows:
        if row.get("parser_id") in {"option_text_fuzzy_parser"}:
            continue
        if safe_float(row.get("false_positive_risk_rate"), 1.0) > 0.05:
            continue
        if safe_float(row.get("ambiguity_rate"), 1.0) > 0.25:
            continue
        if safe_float(row.get("positive_oracle_rate"), 0.0) + 1e-9 < safe_float(strict.get("positive_oracle_rate"), 0.0):
            continue
        candidates.append(row)
    if not candidates:
        return {"parser_id": "strict_letter_current", "selected": False, "selection_reason": "no candidate met ambiguity/false-positive gates"}
    best = max(candidates, key=lambda row: (safe_float(row.get("positive_oracle_rate"), 0.0), safe_float(row.get("terminal_best_reward"), 0.0), safe_float(row.get("parse_success_rate"), 0.0)))
    selected = best.get("parser_id") != "strict_letter_current" and (
        safe_float(best.get("positive_oracle_rate"), 0.0) > safe_float(strict.get("positive_oracle_rate"), 0.0) + 0.02
        or safe_float(best.get("parse_success_rate"), 0.0) > safe_float(strict.get("parse_success_rate"), 0.0) + 0.05
    )
    return {**best, "selected": bool(selected), "selection_reason": "calibration parser gate"}


def selected_parser_id() -> str:
    data = read_json(OUT_ROOT / "selected_science_parser.json", {}) or {}
    return str(data.get("parser_id") or "strict_letter_current")


def recipe_configs() -> dict[str, dict[str, Any]]:
    return {
        "baseline_v3_regenerated": {"recipe_id": "baseline_v3_regenerated", "family": "old_tap_aligned", "alpha": 0.005, "default_children": 1},
        "L24_heavy": {"recipe_id": "L24_heavy", "family": "old_tap_aligned", "alpha": 0.005, "default_children": 1, "layer_children": {24: 2}},
        "L36_heavy": {"recipe_id": "L36_heavy", "family": "old_tap_aligned", "alpha": 0.005, "default_children": 1, "layer_children": {36: 2}},
        "L47_heavy": {"recipe_id": "L47_heavy", "family": "old_tap_aligned", "alpha": 0.005, "default_children": 1, "layer_children": {47: 2}},
        "L2_47_emphasis": {"recipe_id": "L2_47_emphasis", "family": "old_tap_aligned", "alpha": 0.005, "default_children": 1, "stage_children": {"L2_47": 2}},
        "DualAnchor_aligned": {"recipe_id": "DualAnchor_aligned", "family": "v4_tap_aligned", "alpha": 0.005, "default_children": 1},
        "bridge_materialization": {"recipe_id": "bridge_materialization", "family": "high_yield_recipe_direction", "alpha": 0.005, "default_children": 1},
        "old_content_residual": {"recipe_id": "old_content_residual", "family": "old_tap_aligned", "alpha": 0.005, "default_children": 1},
        "random_orthogonal_control": {"recipe_id": "random_orthogonal_control", "family": "random_orthogonal", "alpha": 0.005, "default_children": 1, "control": True},
        "alpha_01_only": {"recipe_id": "alpha_01_only", "family": "old_tap_aligned", "alpha": 0.01, "default_children": 1},
        "alpha_02_diagnostic": {"recipe_id": "alpha_02_diagnostic", "family": "old_tap_aligned", "alpha": 0.02, "default_children": 1, "diagnostic": True},
        "more_children_same_budget": {"recipe_id": "more_children_same_budget", "family": "old_tap_aligned", "alpha": 0.005, "default_children": 2},
    }


def source_recipe_ids(source: str) -> list[str]:
    mapping = {
        "sciq": ["baseline_v3_regenerated", "alpha_01_only", "random_orthogonal_control"],
        "mmlu_high_school_physics": ["baseline_v3_regenerated", "L47_heavy", "L2_47_emphasis", "bridge_materialization", "random_orthogonal_control"],
        "mmlu_high_school_chemistry": ["baseline_v3_regenerated", "L24_heavy", "L36_heavy", "DualAnchor_aligned", "bridge_materialization", "random_orthogonal_control"],
        "mmlu_anatomy": ["baseline_v3_regenerated", "L24_heavy", "bridge_materialization", "old_content_residual", "random_orthogonal_control"],
        "mmlu_biology": ["baseline_v3_regenerated", "L24_heavy", "bridge_materialization", "old_content_residual", "random_orthogonal_control"],
    }
    return mapping.get(source, ["baseline_v3_regenerated", "L24_heavy", "random_orthogonal_control"])


def child_count_for_recipe(recipe: dict[str, Any], loop: int, layer: int) -> int:
    stage = f"L{loop}_{layer}"
    if stage in recipe.get("stage_children", {}):
        return int(recipe["stage_children"][stage])
    if int(layer) in recipe.get("layer_children", {}):
        return int(recipe["layer_children"][int(layer)])
    return int(recipe.get("default_children", 1))


def repair_split_tasks(repair_split: str, *, max_per_source: int | None = None) -> list[dict[str, Any]]:
    rows = [row for row in suite_rows() if row.get("repair_split") == repair_split and row.get("domain") == "science"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("source_detail"))].append(row)
    out = []
    for source, vals in sorted(grouped.items()):
        vals = sorted(vals, key=lambda row: str(row.get("task_id")))
        out.extend(vals[:max_per_source] if max_per_source else vals)
    return out


def task_to_prompt_task(row: dict[str, Any]) -> dict[str, Any]:
    meta = task_by_id().get(str(row.get("task_id")), {})
    out = dict(meta)
    out.update({k: row.get(k) for k in ("task_id", "domain", "source_dataset", "split", "correct_option") if row.get(k) is not None})
    if "prompt" not in out or not out.get("prompt"):
        question = out.get("question") or ""
        options = out.get("options") or {}
        if isinstance(options, str):
            options = parse_literal(options, {})
        option_lines = "\n".join(f"{key}. {value}" for key, value in dict(options).items())
        out["prompt"] = f"Question: {question}\nOptions:\n{option_lines}\nThink briefly if needed.\nFINAL ANSWER:"
    return out


def new_child_id(parent_id: str, loop: int, layer: int, mutation_index: int, recipe_id: str) -> str:
    return f"{parent_id}/L{loop}_{layer}_{safe_name(recipe_id)}_p{int(mutation_index)}"


def generated_key(recipe_id: str, task_id: str) -> str:
    return f"{recipe_id}::{task_id}"


def apply_selected_parser_metrics(row: dict[str, Any], parser_id: str) -> dict[str, Any]:
    parsed = parser_eval_row(row, parser_id)
    return {
        "selected_parser_id": parser_id,
        "selected_parser_parse_success": parsed["parse_success"],
        "selected_parser_parsed_option": parsed["parsed_option"],
        "selected_parser_reward": parsed["parser_reward"],
        "selected_parser_correct": parsed["parser_correct"],
        "selected_parser_ambiguity": parsed["ambiguity_flag"],
        "selected_parser_risk_flags": parsed["risk_flags"],
    }


def run_recipe_generation(repair_split: str, *, mode: str) -> dict[str, Any]:
    """Run actual architecture-looped generation for science recipe tasks.

    The default caps keep the run bounded while still regenerating branches. Set
    REPAIR_V2_CAL_TASKS_PER_SOURCE / REPAIR_V2_HELDOUT_TASKS_PER_SOURCE higher
    or FORCE_RERUN=1 to broaden/resume.
    """
    ensure_root()
    force = os.environ.get("FORCE_RERUN") == "1"
    if repair_split == "science_recipe_calibration":
        max_per_source = int(os.environ.get("REPAIR_V2_CAL_TASKS_PER_SOURCE", "4"))
    else:
        max_per_source = int(os.environ.get("REPAIR_V2_HELDOUT_TASKS_PER_SOURCE", "99"))
    tasks = repair_split_tasks(repair_split, max_per_source=max_per_source)
    recipes = recipe_configs()
    payload = load_generated()
    rows: list[dict[str, Any]] = list(payload.get("rows") or [])
    stage_rows: list[dict[str, Any]] = list(payload.get("stage_rows") or [])
    task_rows: list[dict[str, Any]] = list(payload.get("task_rows") or [])
    terminal_rows: list[dict[str, Any]] = list(payload.get("terminal_rows") or [])
    errors: list[dict[str, Any]] = list(payload.get("errors") or [])
    completed = set(read_json(COMPLETED_TASKS_JSON, []) or [])
    completed_recipes = set(read_json(COMPLETED_RECIPES_JSON, []) or [])
    completed_sources = set(read_json(COMPLETED_SOURCES_JSON, []) or [])
    policies, _policy_rows = base.load_constrained_policies()
    taps = policies[SELECTED_POLICY]
    bank = load_pt(BASIS_BANK_PT, {"directions_by_layer": {}, "directions": []}) or {"directions_by_layer": {}, "directions": []}
    parser_id = selected_parser_id()
    started = time.time()
    extractor: BGTransformerFeatureExtractor | None = None
    try:
        extractor = BGTransformerFeatureExtractor(device="cuda" if torch.cuda.is_available() else "cpu", dtype="auto", force_all_loops=True)
        model = extractor.model
        tokenizer = extractor.tokenizer
        device = extractor.device
        for task_index, suite_row in enumerate(tasks):
            task_id = str(suite_row.get("task_id"))
            source = str(suite_row.get("source_detail"))
            recipe_ids = source_recipe_ids(source)
            if mode == "heldout":
                best = read_json(OUT_ROOT / "best_science_recipe_v2.json", {}) or {}
                selected_by_source = (best.get("selected_by_source") or {}).get(source) or best.get("shared_recipe_id") or "baseline_v3_regenerated"
                recipe_ids = sorted({"baseline_v3_regenerated", str(selected_by_source)})
            for recipe_id in recipe_ids:
                recipe = recipes[recipe_id]
                key = generated_key(recipe_id, task_id)
                if key in completed and not force:
                    continue
                task_stage_rows: list[dict[str, Any]] = []
                final_pool: list[dict[str, Any]] = []
                branch_group_id = f"repair_v2::{repair_split}::{safe_name(recipe_id)}::{safe_name(task_id)}"
                task = task_to_prompt_task(suite_row)
                try:
                    root = evaluate_branch(
                        model=model,
                        tokenizer=tokenizer,
                        task=task,
                        branch_group_id=branch_group_id,
                        branch_id=f"{branch_group_id}/root",
                        parent=None,
                        hooks=[],
                        birth_layer=None,
                        birth_loop=None,
                        birth_stage="root",
                        perturbation_family="clean",
                        mutation_index=0,
                        device=device,
                        max_new_tokens=int(os.environ.get("REPAIR_V2_MAX_NEW_TOKENS", "32")),
                    )
                    root.update({"repair_recipe_id": recipe_id, "repair_split": repair_split, "source_detail": source, **apply_selected_parser_metrics(root, parser_id)})
                    survivors = [root]
                    rows.append(root)
                    for stage_index, (loop, layer) in enumerate(NONTERMINAL_STAGES):
                        expanded = list(survivors)
                        child_count = child_count_for_recipe(recipe, loop, layer)
                        for parent_index, parent in enumerate(survivors):
                            seed = 20260601 + 100000 * (hash(recipe_id) % 997) + 1009 * task_index + 101 * stage_index + 17 * parent_index
                            entries = make_family_deltas(layer, float(recipe.get("alpha", 0.005)), child_count + 1, str(recipe.get("family", "old_tap_aligned")), seed, bank)
                            for entry in entries:
                                if int(entry["branch_id"]) == 0:
                                    continue
                                hook = make_hook_entry(entry, layer, loop, float(recipe.get("alpha", 0.005)), seed)
                                hooks = list(parent.get("hooks") or []) + [hook]
                                child = evaluate_branch(
                                    model=model,
                                    tokenizer=tokenizer,
                                    task=task,
                                    branch_group_id=branch_group_id,
                                    branch_id=new_child_id(str(parent["branch_id"]), loop, layer, int(entry["branch_id"]), recipe_id),
                                    parent=parent,
                                    hooks=hooks,
                                    birth_layer=layer,
                                    birth_loop=loop,
                                    birth_stage=f"L{loop}_{layer}",
                                    perturbation_family=str(entry.get("delta_family")),
                                    mutation_index=int(entry["branch_id"]),
                                    device=device,
                                    max_new_tokens=int(os.environ.get("REPAIR_V2_MAX_NEW_TOKENS", "32")),
                                )
                                child.update({"repair_recipe_id": recipe_id, "repair_split": repair_split, "source_detail": source, **apply_selected_parser_metrics(child, parser_id)})
                                expanded.append(child)
                                rows.append(child)
                        survivors, stage = threshold_survive(expanded, taps, layer, loop)
                        stage.update(
                            {
                                "task_id": task_id,
                                "domain": task.get("domain"),
                                "split": task.get("split"),
                                "repair_split": repair_split,
                                "source_dataset": task.get("source_dataset"),
                                "source_detail": source,
                                "recipe_id": recipe_id,
                                "stage_name": f"L{loop}_{layer}",
                                "stage_index": stage_index,
                            }
                        )
                        stage_rows.append(stage)
                        task_stage_rows.append(stage)
                        write_json(PROGRESS_ROOT / f"stage_{safe_name(key)}_{stage_index:02d}.json", {"stage": stage, "saved_at": time.time()})
                    terminal = terminal_select(survivors, taps)
                    final_pool = list(survivors)
                    trow = v2_task_summary(task, final_pool, terminal, task_stage_rows)
                    selected_rewards = [safe_float(row.get("selected_parser_reward"), row_reward(row)) for row in final_pool]
                    trow.update(
                        {
                            "repair_split": repair_split,
                            "source_detail": source,
                            "recipe_id": recipe_id,
                            "selected_parser_id": parser_id,
                            "selected_parser_terminal_best_reward": max(selected_rewards) if selected_rewards else float("nan"),
                            "selected_parser_positive_oracle": 1.0 if selected_rewards and max(selected_rewards) > 0 else 0.0,
                            "selected_parser_parse_success": finite_mean(row.get("selected_parser_parse_success") for row in final_pool),
                            "replay_only": 0.0,
                        }
                    )
                    task_rows.append(trow)
                    for term_row in terminal_policy_rows(task, final_pool, taps):
                        term_row.update({"recipe_id": recipe_id, "repair_split": repair_split, "source_detail": source})
                        terminal_rows.append(term_row)
                    write_json(
                        CANDIDATE_TREE_ROOT / f"{safe_name(key)}.json",
                        {
                            "task_id": task_id,
                            "recipe_id": recipe_id,
                            "repair_split": repair_split,
                            "source_detail": source,
                            "final_branch_ids": [str(row.get("branch_id")) for row in final_pool],
                            "candidate_branch_ids": [str(row.get("branch_id")) for row in rows if row.get("repair_recipe_id") == recipe_id and row.get("task_id") == task_id],
                        },
                    )
                    completed.add(key)
                    completed_recipes.add(recipe_id)
                    completed_sources.add(source)
                    write_json(COMPLETED_TASKS_JSON, sorted(completed))
                    write_json(COMPLETED_RECIPES_JSON, sorted(completed_recipes))
                    write_json(COMPLETED_SOURCES_JSON, sorted(completed_sources))
                    write_json(PROGRESS_ROOT / f"task_{safe_name(key)}.json", {"completed": True, "task_row": trow, "saved_at": time.time()})
                    save_generated({"rows": rows, "stage_rows": stage_rows, "task_rows": task_rows, "terminal_rows": terminal_rows, "errors": errors})
                except Exception as exc:
                    errors.append({"task_id": task_id, "recipe_id": recipe_id, "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-6000:]})
                    write_json(PROGRESS_ROOT / f"error_{safe_name(key)}.json", errors[-1])
                    save_generated({"rows": rows, "stage_rows": stage_rows, "task_rows": task_rows, "terminal_rows": terminal_rows, "errors": errors})
        write_csv(LINEAGE_INCREMENTAL_CSV, [compact_row(row) for row in rows])
    finally:
        if extractor is not None:
            extractor.cleanup()
    out = {"rows": rows, "stage_rows": stage_rows, "task_rows": task_rows, "terminal_rows": terminal_rows, "errors": errors, "elapsed_seconds": round(time.time() - started, 3)}
    save_generated(out)
    write_json(PARTIAL_SUMMARY_JSON, {"generated_rows": len(rows), "generated_task_rows": len(task_rows), "errors": len(errors), "saved_at": time.time()})
    return out


def generated_task_rows(repair_split: str | None = None) -> list[dict[str, Any]]:
    rows = list(load_generated().get("task_rows") or [])
    if repair_split:
        rows = [row for row in rows if row.get("repair_split") == repair_split]
    return rows


def generated_terminal_rows(domain: str | None = None) -> list[dict[str, Any]]:
    rows = list(load_generated().get("terminal_rows") or [])
    if domain:
        rows = [row for row in rows if row.get("domain") == domain]
    return rows


def summarize_generated_by_recipe(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("recipe_id"))].append(row)
    out = []
    for recipe_id, vals in sorted(grouped.items()):
        out.append(
            {
                "recipe_id": recipe_id,
                "task_count": len(vals),
                "source_count": len({row.get("source_detail") for row in vals}),
                "positive_oracle_rate": finite_mean(row.get("positive_oracle") for row in vals),
                "selected_parser_positive_oracle_rate": finite_mean(row.get("selected_parser_positive_oracle") for row in vals),
                "reward_diverse_rate": finite_mean(row.get("terminal_reward_diverse") for row in vals),
                "terminal_best_reward": finite_mean(row.get("terminal_best_reward") for row in vals),
                "selected_parser_terminal_best_reward": finite_mean(row.get("selected_parser_terminal_best_reward") for row in vals),
                "forced_top1_reward": finite_mean(row.get("terminal_forced_top1_reward") for row in vals),
                "forced_top1_oracle": finite_mean(row.get("terminal_forced_top1_oracle") for row in vals),
                "terminal_oracle_retained": finite_mean(row.get("terminal_oracle_retained") for row in vals),
                "selected_parser_parse_success": finite_mean(row.get("selected_parser_parse_success") for row in vals),
                "stage_false_prunes": finite_mean(row.get("stage_false_prunes") for row in vals),
            }
        )
    return out


def summarize_generated_by_source(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("source_detail"))].append(row)
    out = []
    for source, vals in sorted(grouped.items()):
        best_recipe = max(vals, key=lambda row: (safe_float(row.get("selected_parser_terminal_best_reward"), -9), safe_float(row.get("terminal_best_reward"), -9))).get("recipe_id") if vals else ""
        out.append(
            {
                "source_detail": source,
                "task_recipe_count": len(vals),
                "unique_task_count": len({row.get("task_id") for row in vals}),
                "best_recipe": best_recipe,
                "positive_oracle_rate": finite_mean(row.get("positive_oracle") for row in vals),
                "selected_parser_positive_oracle_rate": finite_mean(row.get("selected_parser_positive_oracle") for row in vals),
                "reward_diverse_rate": finite_mean(row.get("terminal_reward_diverse") for row in vals),
                "terminal_best_reward": finite_mean(row.get("terminal_best_reward") for row in vals),
                "selected_parser_terminal_best_reward": finite_mean(row.get("selected_parser_terminal_best_reward") for row in vals),
                "forced_top1_reward": finite_mean(row.get("terminal_forced_top1_reward") for row in vals),
                "terminal_oracle_retained": finite_mean(row.get("terminal_oracle_retained") for row in vals),
                "parse_success": finite_mean(row.get("selected_parser_parse_success") for row in vals),
            }
        )
    return out


def select_best_recipes(calibration_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_recipe = summarize_generated_by_recipe(calibration_rows)
    baseline = next((row for row in by_recipe if row.get("recipe_id") == "baseline_v3_regenerated"), {})
    eligible = [row for row in by_recipe if row.get("recipe_id") != "random_orthogonal_control"]
    if not eligible:
        return {"shared_recipe_id": "baseline_v3_regenerated", "selected": False, "selected_by_source": {}}
    best_shared = max(eligible, key=lambda row: (safe_float(row.get("selected_parser_positive_oracle_rate"), 0.0), safe_float(row.get("selected_parser_terminal_best_reward"), 0.0), safe_float(row.get("reward_diverse_rate"), 0.0)))
    improved = (
        safe_float(best_shared.get("selected_parser_positive_oracle_rate"), 0.0) > safe_float(baseline.get("selected_parser_positive_oracle_rate"), 0.0) + 0.05
        or safe_float(best_shared.get("selected_parser_terminal_best_reward"), 0.0) > safe_float(baseline.get("selected_parser_terminal_best_reward"), 0.0) + 0.05
    )
    selected_by_source = {}
    for source in sorted({row.get("source_detail") for row in calibration_rows}):
        vals = [row for row in calibration_rows if row.get("source_detail") == source]
        grouped = summarize_generated_by_recipe(vals)
        source_baseline = next((row for row in grouped if row.get("recipe_id") == "baseline_v3_regenerated"), {})
        if not grouped:
            continue
        best = max(grouped, key=lambda row: (safe_float(row.get("selected_parser_positive_oracle_rate"), 0.0), safe_float(row.get("selected_parser_terminal_best_reward"), 0.0)))
        source_improved = safe_float(best.get("selected_parser_terminal_best_reward"), 0.0) > safe_float(source_baseline.get("selected_parser_terminal_best_reward"), 0.0) + 0.05
        selected_by_source[str(source)] = str(best.get("recipe_id") if source_improved else "baseline_v3_regenerated")
    return {
        "shared_recipe_id": str(best_shared.get("recipe_id") if improved else "baseline_v3_regenerated"),
        "selected": bool(improved),
        "best_shared": best_shared,
        "baseline": baseline,
        "selected_by_source": selected_by_source,
        "selection_rule": "calibration only; heldout untouched",
    }


def grouped_terminal_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
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


def v3_reasoning_task_rows() -> list[dict[str, Any]]:
    suite_ids = {str(row.get("task_id")) for row in suite_rows() if row.get("domain") == "reasoning"}
    rows = [row for row in v3_task_rows() if row.get("domain") == "reasoning"]
    return [row for row in rows if not suite_ids or str(row.get("task_id")) in suite_ids]


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


def load_stage_json(path: Path, key: str) -> str:
    return str((read_json(path, {}) or {}).get(key, "INSUFFICIENT"))


def inventory_main() -> int:
    started = time.time()
    tasks = audit_plan_tasks()
    science = [row for row in tasks if str(row.get("domain")) == "science"]
    reasoning = [row for row in tasks if str(row.get("domain")) == "reasoning"]
    artifacts = [
        {"artifact": "science_reasoning_defer_v1", "path": str(V1_ROOT), "status": "READY" if V1_ROOT.exists() else "MISSING"},
        {"artifact": "convergence_hairs_rs_v1", "path": str(HAIRS_ROOT), "status": "READY" if HAIRS_ROOT.exists() else "MISSING"},
        {"artifact": "v3_architecture_rows", "path": str(V3_ROOT / "architecture_looped_rows.pt"), "status": "READY" if (V3_ROOT / "architecture_looped_rows.pt").exists() else "MISSING"},
        {"artifact": "v3_stage_decisions", "path": str(V3_ROOT / "stage_decisions.csv"), "status": "READY" if (V3_ROOT / "stage_decisions.csv").exists() else "MISSING"},
        {"artifact": "v3_hard_slices", "path": str(V3_ROOT / "hard_slices.json"), "status": "READY" if (V3_ROOT / "hard_slices.json").exists() else "MISSING"},
        {"artifact": "v3_threshold_budget", "path": str(V3_ROOT / "threshold_budget.json"), "status": "READY" if (V3_ROOT / "threshold_budget.json").exists() else "MISSING"},
        {"artifact": "v3_l47", "path": str(V3_ROOT / "l47_ablation.json"), "status": "READY" if (V3_ROOT / "l47_ablation.json").exists() else "MISSING"},
        {"artifact": "true_carry", "path": str(PROBE_ROOT / "bg_dualanchor_true_carry_equivalence_v1_2026-05-31"), "status": "READY" if (PROBE_ROOT / "bg_dualanchor_true_carry_equivalence_v1_2026-05-31").exists() else "MISSING"},
        {"artifact": "perturb_lift", "path": str(PROBE_ROOT / "bg_dualanchor_perturbation_lift_v1_2026-05-31"), "status": "READY" if (PROBE_ROOT / "bg_dualanchor_perturbation_lift_v1_2026-05-31").exists() else "MISSING"},
        {"artifact": "basis_bank", "path": str(BASIS_BANK_PT), "status": "READY" if BASIS_BANK_PT.exists() else "MISSING"},
    ]
    missing = [row["artifact"] for row in artifacts if row["status"] == "MISSING"]
    verdict = "BLOCKED" if "v3_architecture_rows" in missing or "basis_bank" in missing else ("PARTIAL" if missing or len(science) < 48 else "READY")
    source_counts = dict(Counter(source_detail(str(row.get("task_id")), row.get("source_dataset")) for row in science))
    payload = {
        "BG_SCIENCE_REASONING_REPAIR_V2_INVENTORY_VERDICT": verdict,
        "artifact_rows": artifacts,
        "science_task_count": len(science),
        "reasoning_task_count": len(reasoning),
        "science_source_counts": source_counts,
        "missing_requested_sources": [src for src in ("openbookqa_science", "mmlu_biology") if src not in source_counts],
        "dualanchor_taps": [ANCHOR_A, ANCHOR_B],
        "soft_hairs_available": (HAIRS_ROOT / "convergence_hair_dataset.pt").exists(),
        "can_regenerate": True,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "inventory.json", payload)
    write_md(
        OUT_ROOT / "inventory.md",
        [
            "# DualAnchor Science + Reasoning Repair v2 Inventory",
            "",
            status_line("BG_SCIENCE_REASONING_REPAIR_V2_INVENTORY_VERDICT", verdict),
            "",
            "## Artifacts",
            "",
            *md_table(artifacts, ["artifact", "status", "path"]),
            "",
            "## Task Pools",
            "",
            f"- science tasks: `{len(science)}`",
            f"- reasoning tasks: `{len(reasoning)}`",
            f"- science source counts: `{source_counts}`",
            "- local science OpenBookQA and MMLU biology pools are not present in the current audit plan",
            "",
            "## Boundary",
            "",
            "- no steering/training/routing change",
            "- convergence hairs remain soft-only",
        ],
    )
    print(status_line("BG_SCIENCE_REASONING_REPAIR_V2_INVENTORY_VERDICT", verdict))
    return 0 if verdict != "BLOCKED" else 1


def task_suite_main() -> int:
    started = time.time()
    rows, inv = build_task_suite_rows()
    if inv["science_count"] < 48:
        verdict = "SCIENCE_LIMITED"
    elif inv["reasoning_count"] < 24:
        verdict = "REASONING_LIMITED"
    elif any(src in inv["missing_requested_science_sources"] for src in ("science_openbookqa", "mmlu_biology")):
        verdict = "MMLU_SCIENCE_LIMITED"
    else:
        verdict = "READY"
    payload = {
        "BG_SCIENCE_REASONING_REPAIR_V2_TASK_SUITE_VERDICT": verdict,
        "task_rows": rows,
        "inventory": inv,
        "selection_note": "All locally available science tasks are included; requested science OpenBookQA/Biology pools are unavailable.",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "task_suite.json", payload)
    write_csv(OUT_ROOT / "task_suite.csv", rows)
    write_md(
        OUT_ROOT / "task_suite.md",
        [
            "# Science/Reasoning Repair v2 Task Suite",
            "",
            status_line("BG_SCIENCE_REASONING_REPAIR_V2_TASK_SUITE_VERDICT", verdict),
            "",
            "## Inventory",
            "",
            *[f"- {k}: `{v}`" for k, v in inv.items() if not isinstance(v, dict)],
            "",
            "## Science Sources",
            "",
            *[f"- {k}: `{v}`" for k, v in inv["science_source_counts"].items()],
            "",
            "## Tasks",
            "",
            *md_table(rows, ["index", "task_id", "domain", "source_detail", "split", "repair_split", "v3_generated"]),
        ],
    )
    print(status_line("BG_SCIENCE_REASONING_REPAIR_V2_TASK_SUITE_VERDICT", verdict))
    return 0


def parser_candidates_main() -> int:
    started = time.time()
    candidates = [
        {"parser_id": parser_id, "description": desc}
        for parser_id, desc in {
            "strict_letter_current": "existing strict parser result from candidate rows",
            "normalized_letter_parser": "normalizes boxed/lowercase/punctuated letters and final spans",
            "option_text_exact_parser": "parses exact unique option text only",
            "option_text_fuzzy_parser": "high-threshold fuzzy text parser, diagnostic",
            "rationale_then_final_parser": "uses final-answer/rationale markers before parsing",
            "MCQ_robust_parser": "strict, final-span, exact text, conservative fuzzy with ambiguity rejection",
            "source_specific_parser": "SciQ normalized-letter; MMLU robust parser",
        }.items()
    ]
    verdict = "READY"
    payload = {
        "BG_SCIENCE_PARSER_PATCH_CANDIDATES_VERDICT": verdict,
        "parser_candidates": candidates,
        "ambiguity_rules": [
            "explicit final-answer span preferred",
            "multiple option letters without final marker rejected",
            "multiple exact option texts rejected",
            "fuzzy multi-match rejected",
        ],
        "experimental_module": "shared/utilities/tests/manual/bg_science_mcq_parser_v2.py",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "parser_candidates.json", payload)
    write_md(
        OUT_ROOT / "parser_candidates.md",
        [
            "# Science Parser Patch Candidates v2",
            "",
            status_line("BG_SCIENCE_PARSER_PATCH_CANDIDATES_VERDICT", verdict),
            "",
            "## Candidates",
            "",
            *md_table(candidates, ["parser_id", "description"]),
            "",
            "## Boundary",
            "",
            "- These are experimental parser candidates.",
            "- Strict parser is not overwritten.",
        ],
    )
    print(status_line("BG_SCIENCE_PARSER_PATCH_CANDIDATES_VERDICT", verdict))
    return 0


def parser_validation_main() -> int:
    started = time.time()
    rows = parser_validation_rows(include_generated=False)
    calibration = parser_summary(rows, "parser_calibration")
    if not calibration:
        calibration = parser_summary(rows, None)
    selected = select_parser(calibration)
    heldout_summary = parser_summary(rows, "science_recipe_heldout")
    if not calibration:
        verdict = "BLOCKED"
    elif selected.get("parser_id") == "strict_letter_current":
        verdict = "STRICT_PARSER_REMAINS_PRIMARY"
    elif selected.get("parser_id") == "source_specific_parser":
        verdict = "SOURCE_SPECIFIC_PARSER_READY"
    elif selected.get("selected"):
        verdict = "PARSER_PATCH_READY"
    else:
        verdict = "PARSER_PATCH_DIAGNOSTIC_ONLY"
    payload = {
        "BG_SCIENCE_PARSER_VALIDATION_VERDICT": verdict,
        "calibration_summary": calibration,
        "heldout_summary": heldout_summary,
        "selected_parser": selected,
        "strict_primary_replaced": bool(selected.get("selected")),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "parser_validation.json", payload)
    write_json(OUT_ROOT / "selected_science_parser.json", selected)
    write_csv(OUT_ROOT / "parser_validation_rows.csv", rows)
    write_md(
        OUT_ROOT / "parser_validation.md",
        [
            "# Science Parser Patch Validation v2",
            "",
            status_line("BG_SCIENCE_PARSER_VALIDATION_VERDICT", verdict),
            "",
            "## Calibration Summary",
            "",
            *md_table(calibration, ["parser_id", "candidate_count", "task_count", "parse_success_rate", "positive_oracle_rate", "terminal_best_reward", "ambiguity_rate", "false_positive_risk_rate", "missed_correct_recovered_count"]),
            "",
            "## Selected Parser",
            "",
            *[f"- {k}: `{fmt(v)}`" for k, v in selected.items()],
            "",
            "## Heldout Summary",
            "",
            *md_table(heldout_summary, ["parser_id", "candidate_count", "task_count", "parse_success_rate", "positive_oracle_rate", "terminal_best_reward", "ambiguity_rate", "false_positive_risk_rate"]),
        ],
    )
    print(status_line("BG_SCIENCE_PARSER_VALIDATION_VERDICT", verdict))
    return 0 if verdict != "BLOCKED" else 1


def recipe_plan_main() -> int:
    started = time.time()
    recipes = list(recipe_configs().values())
    verdict = "SOURCE_SPECIFIC_READY"
    payload = {
        "BG_SCIENCE_BRANCH_RECIPE_V2_PLAN_VERDICT": verdict,
        "recipes": recipes,
        "source_recipe_ids": {src: source_recipe_ids(src) for src in ("sciq", "mmlu_high_school_physics", "mmlu_high_school_chemistry", "mmlu_anatomy")},
        "baseline": {
            "schedule": LOCKED_SCHEDULE,
            "selector": [ANCHOR_A, ANCHOR_B],
            "threshold": "mean_floor_very_loose",
            "budget": 8,
            "l47": "active in nonterminal loops",
            "terminal": "confidence-gated top1; otherwise survivor handoff",
            "hairs": "soft-only",
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "science_recipe_v2_plan.json", payload)
    write_md(
        OUT_ROOT / "science_recipe_v2_plan.md",
        [
            "# Science Branch Recipe v2 Plan",
            "",
            status_line("BG_SCIENCE_BRANCH_RECIPE_V2_PLAN_VERDICT", verdict),
            "",
            "## Recipes",
            "",
            *md_table(recipes, ["recipe_id", "family", "alpha", "default_children", "diagnostic", "control"]),
            "",
            "## Source Recipes",
            "",
            *[f"- {k}: `{v}`" for k, v in payload["source_recipe_ids"].items()],
        ],
    )
    print(status_line("BG_SCIENCE_BRANCH_RECIPE_V2_PLAN_VERDICT", verdict))
    return 0


def calibration_main() -> int:
    started = time.time()
    generated = run_recipe_generation("science_recipe_calibration", mode="calibration")
    rows = [row for row in generated.get("task_rows", []) if row.get("repair_split") == "science_recipe_calibration"]
    summary = summarize_generated_by_recipe(rows)
    best = select_best_recipes(rows)
    baseline = next((row for row in summary if row.get("recipe_id") == "baseline_v3_regenerated"), {})
    best_row = best.get("best_shared", {})
    if generated.get("errors") and not rows:
        verdict = "BLOCKED"
    elif not rows:
        verdict = "DATA_LIMITED"
    elif best.get("selected"):
        verdict = "SCIENCE_RECIPE_FOUND"
    elif safe_float(baseline.get("selected_parser_terminal_best_reward"), 0.0) > SCIENCE_BASELINE_TERMINAL_BEST:
        verdict = "SCIENCE_RECIPE_WEAK_BUT_IMPROVED"
    else:
        verdict = "SCIENCE_NO_RECIPE_IMPROVEMENT"
    payload = {
        "BG_SCIENCE_RECIPE_V2_CALIBRATION_VERDICT": verdict,
        "summary_rows": summary,
        "best_science_recipe_v2": best,
        "generated_task_rows": len(rows),
        "errors": generated.get("errors", []),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "science_recipe_v2_calibration.json", payload)
    write_json(OUT_ROOT / "best_science_recipe_v2.json", best)
    write_csv(OUT_ROOT / "science_recipe_v2_calibration_rows.csv", rows)
    write_md(
        OUT_ROOT / "science_recipe_v2_calibration.md",
        [
            "# Regenerated Science Recipe v2 Calibration",
            "",
            status_line("BG_SCIENCE_RECIPE_V2_CALIBRATION_VERDICT", verdict),
            "",
            "## Summary",
            "",
            *md_table(summary, ["recipe_id", "task_count", "source_count", "positive_oracle_rate", "selected_parser_positive_oracle_rate", "reward_diverse_rate", "terminal_best_reward", "selected_parser_terminal_best_reward", "forced_top1_reward", "terminal_oracle_retained"]),
            "",
            "## Selected Recipes",
            "",
            *[f"- {k}: `{fmt(v)}`" for k, v in best.items() if k != "selected_by_source"],
            f"- selected_by_source: `{best.get('selected_by_source')}`",
        ],
    )
    print(status_line("BG_SCIENCE_RECIPE_V2_CALIBRATION_VERDICT", verdict))
    return 0 if verdict != "BLOCKED" else 1


def heldout_main() -> int:
    started = time.time()
    generated = run_recipe_generation("science_recipe_heldout", mode="heldout")
    rows = [row for row in generated.get("task_rows", []) if row.get("repair_split") == "science_recipe_heldout"]
    summary = summarize_generated_by_recipe(rows)
    baseline = next((row for row in summary if row.get("recipe_id") == "baseline_v3_regenerated"), {})
    selected = [row for row in summary if row.get("recipe_id") != "baseline_v3_regenerated"]
    best = max(selected or [baseline], key=lambda row: safe_float(row.get("selected_parser_terminal_best_reward"), -9))
    if not rows:
        verdict = "INSUFFICIENT"
    elif safe_float(best.get("selected_parser_positive_oracle_rate"), 0.0) >= 0.25 and safe_float(best.get("selected_parser_terminal_best_reward"), 0.0) >= 0.20:
        verdict = "SCIENCE_SOURCE_SPECIFIC_READY"
    elif safe_float(best.get("selected_parser_terminal_best_reward"), 0.0) > SCIENCE_BASELINE_TERMINAL_BEST:
        verdict = "SCIENCE_RECIPE_IMPROVED_BUT_WEAK"
    else:
        verdict = "SCIENCE_BRANCH_GENERATION_STILL_WEAK"
    payload = {
        "BG_SCIENCE_RECIPE_V2_HELDOUT_VERDICT": verdict,
        "summary_rows": summary,
        "source_rows": summarize_generated_by_source(rows),
        "generated_task_rows": len(rows),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "science_recipe_v2_heldout.json", payload)
    write_csv(OUT_ROOT / "science_recipe_v2_heldout_rows.csv", rows)
    write_md(
        OUT_ROOT / "science_recipe_v2_heldout.md",
        [
            "# Science Recipe v2 Heldout",
            "",
            status_line("BG_SCIENCE_RECIPE_V2_HELDOUT_VERDICT", verdict),
            "",
            "## Recipe Summary",
            "",
            *md_table(summary, ["recipe_id", "task_count", "source_count", "positive_oracle_rate", "selected_parser_positive_oracle_rate", "terminal_best_reward", "selected_parser_terminal_best_reward", "reward_diverse_rate", "terminal_oracle_retained"]),
            "",
            "## Source Summary",
            "",
            *md_table(payload["source_rows"], ["source_detail", "unique_task_count", "best_recipe", "positive_oracle_rate", "selected_parser_positive_oracle_rate", "terminal_best_reward", "selected_parser_terminal_best_reward", "parse_success"]),
        ],
    )
    print(status_line("BG_SCIENCE_RECIPE_V2_HELDOUT_VERDICT", verdict))
    return 0


def source_specific_main() -> int:
    started = time.time()
    rows = generated_task_rows("science_recipe_heldout") or generated_task_rows("science_recipe_calibration")
    source_rows = summarize_generated_by_source(rows)
    for row in source_rows:
        if row["unique_task_count"] < 2:
            row["source_readiness"] = "DATA_LIMITED"
        elif safe_float(row.get("selected_parser_positive_oracle_rate"), 0.0) >= 0.50:
            row["source_readiness"] = "HEADLINE_READY"
        elif safe_float(row.get("selected_parser_terminal_best_reward"), 0.0) >= 0.20:
            row["source_readiness"] = "DIAGNOSTIC_READY"
        else:
            row["source_readiness"] = "BRANCH_GENERATION_BLOCKED"
    readiness = Counter(row.get("source_readiness") for row in source_rows)
    if not source_rows:
        verdict = "INSUFFICIENT"
    elif readiness.get("HEADLINE_READY", 0) and readiness.get("BRANCH_GENERATION_BLOCKED", 0):
        verdict = "SCIENCE_SOURCES_PARTIALLY_READY"
    elif any(row.get("source_detail") in {"mmlu_high_school_chemistry", "mmlu_anatomy"} and row.get("source_readiness") == "BRANCH_GENERATION_BLOCKED" for row in source_rows):
        verdict = "MMLU_CHEM_ANATOMY_BLOCKED"
    else:
        verdict = "DATA_LIMITED"
    payload = {
        "BG_SCIENCE_V2_SOURCE_SPECIFIC_VERDICT": verdict,
        "source_rows": source_rows,
        "readiness_counts": dict(readiness),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "science_source_specific.json", payload)
    write_csv(OUT_ROOT / "science_source_specific_rows.csv", source_rows)
    write_md(
        OUT_ROOT / "science_source_specific.md",
        [
            "# Science v2 Source-Specific Diagnosis",
            "",
            status_line("BG_SCIENCE_V2_SOURCE_SPECIFIC_VERDICT", verdict),
            "",
            *md_table(source_rows, ["source_detail", "unique_task_count", "best_recipe", "selected_parser_positive_oracle_rate", "selected_parser_terminal_best_reward", "parse_success", "source_readiness"]),
        ],
    )
    print(status_line("BG_SCIENCE_V2_SOURCE_SPECIFIC_VERDICT", verdict))
    return 0


def parser_recommendation_main() -> int:
    started = time.time()
    validation = read_json(OUT_ROOT / "parser_validation.json", {}) or {}
    selected = validation.get("selected_parser", {})
    parser_id = str(selected.get("parser_id") or "strict_letter_current")
    if parser_id == "strict_letter_current":
        verdict = "KEEP_STRICT_PRIMARY"
    elif parser_id == "normalized_letter_parser":
        verdict = "ADOPT_NORMALIZED_LETTER"
    elif parser_id == "source_specific_parser":
        verdict = "ADOPT_SOURCE_SPECIFIC"
    elif parser_id == "MCQ_robust_parser":
        verdict = "ADOPT_MCQ_ROBUST_WITH_AMBIGUITY_REJECTION"
    else:
        verdict = "ROBUST_DIAGNOSTIC_ONLY"
    payload = {
        "BG_SCIENCE_PARSER_PATCH_RECOMMENDATION_VERDICT": verdict,
        "selected_parser": selected,
        "experimental_module": "shared/utilities/tests/manual/bg_science_mcq_parser_v2.py",
        "production_parser_changed": False,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "parser_patch_recommendation.json", payload)
    write_md(
        OUT_ROOT / "parser_patch_recommendation.md",
        [
            "# Science Parser Patch Recommendation v2",
            "",
            status_line("BG_SCIENCE_PARSER_PATCH_RECOMMENDATION_VERDICT", verdict),
            "",
            *[f"- {k}: `{fmt(v)}`" for k, v in selected.items()],
            "",
            "Production parser changed: `False`.",
        ],
    )
    print(status_line("BG_SCIENCE_PARSER_PATCH_RECOMMENDATION_VERDICT", verdict))
    return 0


def reasoning_terminal_handoff_main() -> int:
    started = time.time()
    suite_ids = {str(row.get("task_id")) for row in suite_rows() if row.get("domain") == "reasoning"}
    rows = [row for row in v3_terminal_rows() if row.get("domain") == "reasoning" and (not suite_ids or str(row.get("task_id")) in suite_ids)]
    summary = grouped_terminal_summary(rows)
    top2 = next((row for row in summary if row.get("policy") == "dualanchor_terminal_top2"), {})
    top5 = next((row for row in summary if row.get("policy") == "dualanchor_terminal_top5"), {})
    full = next((row for row in summary if row.get("policy") == "terminal_defer_all"), {})
    gated = next((row for row in summary if row.get("policy") == "dualanchor_confidence_gated"), {})
    if not summary:
        verdict = "INSUFFICIENT"
    elif safe_float(top5.get("oracle_retained"), 0.0) >= 0.99 and safe_float(gated.get("oracle_retained"), 0.0) >= 0.99:
        verdict = "REASONING_HANDOFF_LOCKED"
    elif safe_float(full.get("oracle_retained"), 0.0) >= 0.99:
        verdict = "REASONING_CONFIDENCE_TOP1_READY_WITH_FULL_HANDOFF"
    elif safe_float(top2.get("oracle_retained"), 0.0) >= 0.99:
        verdict = "REASONING_CONFIDENCE_TOP1_READY_WITH_TOP5"
    else:
        verdict = "REASONING_TERMINAL_POLICY_WEAK"
    payload = {
        "BG_REASONING_TERMINAL_HANDOFF_V2_VERDICT": verdict,
        "summary_rows": summary,
        "locked_policy": "confidence-gated top1; otherwise top5/full survivor-set handoff",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "reasoning_terminal_handoff_v2.json", payload)
    write_csv(OUT_ROOT / "reasoning_terminal_handoff_rows.csv", rows)
    write_md(
        OUT_ROOT / "reasoning_terminal_handoff_v2.md",
        [
            "# Reasoning Terminal Handoff v2",
            "",
            status_line("BG_REASONING_TERMINAL_HANDOFF_V2_VERDICT", verdict),
            "",
            *md_table(summary, ["policy", "task_count", "selected_count", "oracle_retained", "best_selected_reward", "first_selected_reward", "first_selected_oracle", "defer_rate", "confident_rate"]),
            "",
            f"Locked policy: `{payload['locked_policy']}`",
        ],
    )
    print(status_line("BG_REASONING_TERMINAL_HANDOFF_V2_VERDICT", verdict))
    return 0


def reasoning_hard_slice_main() -> int:
    started = time.time()
    rows = v3_reasoning_task_rows()
    hard = [row for row in rows if safe_float(row.get("positive_oracle"), 0.0) > 0 and safe_float(row.get("terminal_reward_diverse"), 0.0) > 0]
    summary = metric_summary(rows)
    hard_summary = metric_summary(hard)
    if not rows:
        verdict = "INSUFFICIENT"
    elif safe_float(summary.get("positive_oracle_rate"), 0.0) >= 0.50 and safe_float(summary.get("terminal_oracle_retained"), 0.0) >= 0.95:
        verdict = "REASONING_READY_WITH_HANDOFF"
    elif safe_float(summary.get("positive_oracle_rate"), 0.0) < 0.25:
        verdict = "REASONING_BRANCH_GENERATION_WEAK"
    else:
        verdict = "REASONING_TERMINAL_STILL_WEAK"
    payload = {
        "BG_REASONING_HARD_SLICE_V2_VERDICT": verdict,
        "summary": summary,
        "hard_slice_summary": hard_summary,
        "hard_slice_count": len(hard),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "reasoning_hard_slice_v2.json", payload)
    write_csv(OUT_ROOT / "reasoning_hard_slice_rows.csv", rows)
    write_md(
        OUT_ROOT / "reasoning_hard_slice_v2.md",
        [
            "# Reasoning Hard-Slice v2",
            "",
            status_line("BG_REASONING_HARD_SLICE_V2_VERDICT", verdict),
            "",
            "## Summary",
            "",
            *[f"- {k}: `{fmt(v)}`" for k, v in summary.items()],
            "",
            "## Positive + Reward-Diverse Hard Slice",
            "",
            *[f"- {k}: `{fmt(v)}`" for k, v in hard_summary.items()],
        ],
    )
    print(status_line("BG_REASONING_HARD_SLICE_V2_VERDICT", verdict))
    return 0


def l47_layer_ablation_main() -> int:
    started = time.time()
    rows = generated_task_rows("science_recipe_calibration") + generated_task_rows("science_recipe_heldout")
    layer_rows = []
    for row in rows:
        rid = str(row.get("recipe_id"))
        mode = "other"
        if rid in {"baseline_v3_regenerated", "L47_heavy", "L2_47_emphasis", "L24_heavy", "L36_heavy", "more_children_same_budget"}:
            mode = rid
        layer_rows.append({**row, "mode": mode})
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in layer_rows:
        grouped[str(row.get("mode"))].append(row)
    summary = []
    for mode, vals in sorted(grouped.items()):
        summary.append(
            {
                "mode": mode,
                "task_count": len(vals),
                "positive_oracle_rate": finite_mean(row.get("selected_parser_positive_oracle") for row in vals),
                "terminal_best_reward": finite_mean(row.get("selected_parser_terminal_best_reward") for row in vals),
                "reward_diverse_rate": finite_mean(row.get("terminal_reward_diverse") for row in vals),
                "terminal_oracle_retained": finite_mean(row.get("terminal_oracle_retained") for row in vals),
            }
        )
    best = max(summary, key=lambda row: safe_float(row.get("terminal_best_reward"), -9), default={})
    verdict_map = {
        "L47_heavy": "L47_HEAVY_HELPS",
        "L2_47_emphasis": "L2_47_HELPS",
        "L24_heavy": "L24_HEAVY_HELPS",
        "L36_heavy": "L36_HEAVY_HELPS",
    }
    if not summary:
        verdict = "INSUFFICIENT"
    else:
        verdict = verdict_map.get(str(best.get("mode")), "L47_NOT_ENOUGH")
        if safe_float(best.get("terminal_best_reward"), 0.0) <= SCIENCE_BASELINE_TERMINAL_BEST:
            verdict = "L47_NOT_ENOUGH"
    payload = {
        "BG_SCIENCE_L47_LAYER_ABLATION_V2_VERDICT": verdict,
        "summary_rows": summary,
        "regenerated": True,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "science_l47_layer_ablation_v2.json", payload)
    write_csv(OUT_ROOT / "science_l47_layer_rows.csv", layer_rows)
    write_md(
        OUT_ROOT / "science_l47_layer_ablation_v2.md",
        [
            "# Science L47/Layer Ablation v2",
            "",
            status_line("BG_SCIENCE_L47_LAYER_ABLATION_V2_VERDICT", verdict),
            "",
            *md_table(summary, ["mode", "task_count", "positive_oracle_rate", "terminal_best_reward", "reward_diverse_rate", "terminal_oracle_retained"]),
        ],
    )
    print(status_line("BG_SCIENCE_L47_LAYER_ABLATION_V2_VERDICT", verdict))
    return 0


def perturbation_escalation_main() -> int:
    started = time.time()
    rows = generated_task_rows("science_recipe_calibration")
    baseline = [row for row in rows if row.get("recipe_id") == "baseline_v3_regenerated"]
    escalated = [row for row in rows if row.get("recipe_id") in {"more_children_same_budget", "alpha_02_diagnostic", "L47_heavy", "L2_47_emphasis"}]
    baseline_by_task = {str(row.get("task_id")): row for row in baseline}
    eval_rows = []
    helped = 0
    for row in escalated:
        base = baseline_by_task.get(str(row.get("task_id")), {})
        delta = safe_float(row.get("selected_parser_terminal_best_reward"), 0.0) - safe_float(base.get("selected_parser_terminal_best_reward"), 0.0)
        rec = {**row, "baseline_selected_parser_terminal_best_reward": base.get("selected_parser_terminal_best_reward"), "terminal_best_delta": delta, "created_positive_where_baseline_not": 1.0 if delta > 0 and safe_float(base.get("selected_parser_positive_oracle"), 0.0) <= 0 and safe_float(row.get("selected_parser_positive_oracle"), 0.0) > 0 else 0.0}
        helped += 1 if delta > 0 else 0
        eval_rows.append(rec)
    if not eval_rows:
        verdict = "DATA_LIMITED"
    elif helped > 0:
        verdict = "ESCALATION_HELPS_SPECIFIC_SOURCES"
    else:
        verdict = "ESCALATION_NO_HELP"
    payload = {
        "BG_SCIENCE_PERTURBATION_ESCALATION_V2_VERDICT": verdict,
        "row_count": len(eval_rows),
        "helped_count": helped,
        "created_positive_count": sum(1 for row in eval_rows if safe_float(row.get("created_positive_where_baseline_not"), 0.0) > 0),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "science_perturbation_escalation_v2.json", payload)
    write_csv(OUT_ROOT / "science_perturbation_escalation_rows.csv", eval_rows)
    write_md(
        OUT_ROOT / "science_perturbation_escalation_v2.md",
        [
            "# Science Perturbation Escalation v2",
            "",
            status_line("BG_SCIENCE_PERTURBATION_ESCALATION_V2_VERDICT", verdict),
            "",
            *[f"- {k}: `{fmt(v)}`" for k, v in payload.items() if k != "elapsed_seconds"],
        ],
    )
    print(status_line("BG_SCIENCE_PERTURBATION_ESCALATION_V2_VERDICT", verdict))
    return 0


def soft_hairs_main() -> int:
    started = time.time()
    hair_rows = read_csv(HAIRS_ROOT / "convergence_hair_pair_rows.csv")
    task_rows = {row.get("task_id"): row for row in v3_task_rows()}
    rows = []
    for row in hair_rows:
        task = task_rows.get(row.get("task_id"), {})
        source = source_detail(str(row.get("task_id")), row.get("source_dataset"))
        no_good = row.get("domain") == "science" and safe_float(task.get("positive_oracle"), 0.0) <= 0
        rows.append({**row, "source_detail": source, "no_good_science": 1.0 if no_good else 0.0, "positive_oracle": task.get("positive_oracle"), "terminal_reward_diverse": task.get("terminal_reward_diverse")})
    source_summary = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("source_detail"))].append(row)
    for source, vals in sorted(grouped.items()):
        source_summary.append(
            {
                "source_detail": source,
                "pair_count": len(vals),
                "hidden_rms_normalized": finite_mean(row.get("hidden_rms_normalized") for row in vals),
                "dualanchor_abs_avg_margin": finite_mean(row.get("dualanchor_abs_avg_margin") for row in vals),
                "reward_tied_rate": finite_mean(row.get("reward_tied_eval_only") for row in vals),
                "no_good_science_rate": finite_mean(row.get("no_good_science") for row in vals),
            }
        )
    verdict = "SCIENCE_NO_GOOD_WARNING_USEFUL" if any(safe_float(row.get("no_good_science_rate"), 0.0) > 0.5 for row in source_summary) else "SOFT_HAIRS_USEFUL_MONITOR"
    payload = {
        "BG_SOFT_HAIRS_V2_VERDICT": verdict,
        "source_summary": source_summary,
        "row_count": len(rows),
        "runtime_usage": "diagnostic only",
        "hard_merge": False,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "soft_hairs_v2.json", payload)
    write_csv(OUT_ROOT / "soft_hairs_v2_rows.csv", rows)
    write_md(
        OUT_ROOT / "soft_hairs_v2.md",
        [
            "# Soft Convergence-Hair Monitoring v2",
            "",
            status_line("BG_SOFT_HAIRS_V2_VERDICT", verdict),
            "",
            *md_table(source_summary, ["source_detail", "pair_count", "hidden_rms_normalized", "dualanchor_abs_avg_margin", "reward_tied_rate", "no_good_science_rate"]),
            "",
            "L30/L42 hairs remain soft diagnostics only.",
        ],
    )
    print(status_line("BG_SOFT_HAIRS_V2_VERDICT", verdict))
    return 0


def integrated_repair_main() -> int:
    started = time.time()
    science_heldout = read_json(OUT_ROOT / "science_recipe_v2_heldout.json", {}) or {}
    source = read_json(OUT_ROOT / "science_source_specific.json", {}) or {}
    reasoning = read_json(OUT_ROOT / "reasoning_terminal_handoff_v2.json", {}) or {}
    parser = read_json(OUT_ROOT / "parser_validation.json", {}) or {}
    science_verdict = science_heldout.get("BG_SCIENCE_RECIPE_V2_HELDOUT_VERDICT", "INSUFFICIENT")
    reasoning_verdict = reasoning.get("BG_REASONING_TERMINAL_HANDOFF_V2_VERDICT", "INSUFFICIENT")
    if reasoning_verdict not in {"REASONING_HANDOFF_LOCKED", "REASONING_CONFIDENCE_TOP1_READY_WITH_FULL_HANDOFF", "REASONING_CONFIDENCE_TOP1_READY_WITH_TOP5"}:
        verdict = "REASONING_STILL_BLOCKED"
    elif science_verdict in {"SCIENCE_HEADLINE_READY", "SCIENCE_SOURCE_SPECIFIC_READY"}:
        verdict = "REASONING_REPAIRED_SCIENCE_PARTIAL"
    elif parser.get("BG_SCIENCE_PARSER_VALIDATION_VERDICT") in {"PARSER_PATCH_READY", "SOURCE_SPECIFIC_PARSER_READY"}:
        verdict = "SCIENCE_PARSER_REPAIRED_BRANCH_WEAK"
    else:
        verdict = "SCIENCE_STILL_BLOCKED"
    rows = []
    for row in science_heldout.get("summary_rows", []):
        rows.append({"domain": "science", **row})
    for row in reasoning.get("summary_rows", []):
        rows.append({"domain": "reasoning", **row})
    payload = {
        "BG_SCIENCE_REASONING_INTEGRATED_REPAIR_V2_VERDICT": verdict,
        "science_heldout_verdict": science_verdict,
        "reasoning_handoff_verdict": reasoning_verdict,
        "source_verdict": source.get("BG_SCIENCE_V2_SOURCE_SPECIFIC_VERDICT"),
        "parser_verdict": parser.get("BG_SCIENCE_PARSER_VALIDATION_VERDICT"),
        "rows": rows,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "integrated_repair_eval.json", payload)
    write_csv(OUT_ROOT / "integrated_repair_rows.csv", rows)
    write_md(
        OUT_ROOT / "integrated_repair_eval.md",
        [
            "# Integrated Science + Reasoning Repair v2",
            "",
            status_line("BG_SCIENCE_REASONING_INTEGRATED_REPAIR_V2_VERDICT", verdict),
            "",
            *[f"- {k}: `{fmt(v)}`" for k, v in payload.items() if k not in {"rows", "elapsed_seconds"}],
        ],
    )
    print(status_line("BG_SCIENCE_REASONING_INTEGRATED_REPAIR_V2_VERDICT", verdict))
    return 0


def pre_steering_main() -> int:
    started = time.time()
    parser = read_json(OUT_ROOT / "parser_validation.json", {}) or {}
    science = read_json(OUT_ROOT / "science_recipe_v2_heldout.json", {}) or {}
    source = read_json(OUT_ROOT / "science_source_specific.json", {}) or {}
    reasoning = read_json(OUT_ROOT / "reasoning_terminal_handoff_v2.json", {}) or {}
    integrated = read_json(OUT_ROOT / "integrated_repair_eval.json", {}) or {}
    science_verdict = science.get("BG_SCIENCE_RECIPE_V2_HELDOUT_VERDICT", "INSUFFICIENT")
    reasoning_verdict = reasoning.get("BG_REASONING_TERMINAL_HANDOFF_V2_VERDICT", "INSUFFICIENT")
    if reasoning_verdict not in {"REASONING_HANDOFF_LOCKED", "REASONING_CONFIDENCE_TOP1_READY_WITH_FULL_HANDOFF", "REASONING_CONFIDENCE_TOP1_READY_WITH_TOP5"}:
        verdict = "NEEDS_REASONING_TERMINAL_MORE"
    elif science_verdict == "SCIENCE_HEADLINE_READY":
        verdict = "READY_FOR_STEERING_REASONING_AND_SCIENCE"
    elif science_verdict == "SCIENCE_SOURCE_SPECIFIC_READY":
        verdict = "READY_FOR_STEERING_REASONING_PLUS_SCIQ_OPENBOOK"
    elif science_verdict in {"SCIENCE_RECIPE_IMPROVED_BUT_WEAK", "SCIENCE_BRANCH_GENERATION_STILL_WEAK"}:
        verdict = "READY_FOR_STEERING_WITH_SCIENCE_DIAGNOSTIC"
    elif parser.get("BG_SCIENCE_PARSER_VALIDATION_VERDICT") in {"PARSER_FALSE_POSITIVE_RISK_HIGH", "STRICT_PARSER_REMAINS_PRIMARY"}:
        verdict = "NEEDS_SCIENCE_PARSER_MORE"
    else:
        verdict = "NEEDS_SCIENCE_BRANCH_RECIPE_MORE"
    baseline = {
        "schedule": LOCKED_SCHEDULE,
        "selector": f"{ANCHOR_A} + {ANCHOR_B}",
        "threshold": "mean_floor_very_loose",
        "budget": 8,
        "l47": "active in nonterminal loops",
        "terminal": "confidence-gated top1; otherwise top5/full survivor-set handoff",
        "convergence_hairs": "soft-only monitoring",
        "parser": selected_parser_id(),
        "science_scope": "headline" if verdict == "READY_FOR_STEERING_REASONING_AND_SCIENCE" else "diagnostic/excluded from headline",
        "reasoning_scope": "headline" if verdict.startswith("READY_FOR_STEERING") else "not ready",
        "steering": "not run in this prompt",
    }
    payload = {
        "BG_PRE_STEERING_READINESS_V2_VERDICT": verdict,
        "input_verdicts": {
            "parser": parser.get("BG_SCIENCE_PARSER_VALIDATION_VERDICT"),
            "science_heldout": science_verdict,
            "source": source.get("BG_SCIENCE_V2_SOURCE_SPECIFIC_VERDICT"),
            "reasoning": reasoning_verdict,
            "integrated": integrated.get("BG_SCIENCE_REASONING_INTEGRATED_REPAIR_V2_VERDICT"),
        },
        "locked_baseline": baseline,
        "cannot_claim": [
            "steering was tested",
            "hard convergence-hair merge",
            "compute savings",
            "autoregressive branch-specific KV/cache fork-carry",
            "production routing change",
            "science headline readiness unless verdict says ready",
        ],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "pre_steering_readiness_v2.json", payload)
    write_md(
        OUT_ROOT / "pre_steering_readiness_v2.md",
        [
            "# Pre-Steering Readiness v2",
            "",
            status_line("BG_PRE_STEERING_READINESS_V2_VERDICT", verdict),
            "",
            "## Input Verdicts",
            "",
            *[f"- {k}: `{v}`" for k, v in payload["input_verdicts"].items()],
            "",
            "## Locked Baseline",
            "",
            *[f"- {k}: `{v}`" for k, v in baseline.items()],
        ],
    )
    print(status_line("BG_PRE_STEERING_READINESS_V2_VERDICT", verdict))
    return 0


def status_from(verdict: str, integrated: str) -> str:
    if verdict == "READY_FOR_STEERING_REASONING_AND_SCIENCE":
        return "SCIENCE_AND_REASONING_READY"
    if verdict == "READY_FOR_STEERING_REASONING_PLUS_SCIQ_OPENBOOK":
        return "REASONING_READY_SCIENCE_PARTIAL"
    if verdict in {"READY_FOR_STEERING_REASONING_ONLY", "READY_FOR_STEERING_WITH_SCIENCE_DIAGNOSTIC"}:
        return "REASONING_READY_SCIENCE_DIAGNOSTIC"
    if integrated == "SCIENCE_PARSER_REPAIRED_BRANCH_WEAK":
        return "SCIENCE_PARSER_REPAIRED_BRANCH_WEAK"
    if integrated == "SCIENCE_STILL_BLOCKED":
        return "SCIENCE_STILL_BLOCKED"
    if integrated == "REASONING_STILL_BLOCKED":
        return "REASONING_STILL_BLOCKED"
    if verdict.startswith("NEEDS_SCIENCE"):
        return "SCIENCE_STILL_BLOCKED"
    return "INSUFFICIENT"


def synthesis_main() -> int:
    started = time.time()
    inputs = {
        "inventory": read_json(OUT_ROOT / "inventory.json", {}) or {},
        "task_suite": read_json(OUT_ROOT / "task_suite.json", {}) or {},
        "parser_candidates": read_json(OUT_ROOT / "parser_candidates.json", {}) or {},
        "parser_validation": read_json(OUT_ROOT / "parser_validation.json", {}) or {},
        "recipe_plan": read_json(OUT_ROOT / "science_recipe_v2_plan.json", {}) or {},
        "calibration": read_json(OUT_ROOT / "science_recipe_v2_calibration.json", {}) or {},
        "heldout": read_json(OUT_ROOT / "science_recipe_v2_heldout.json", {}) or {},
        "source": read_json(OUT_ROOT / "science_source_specific.json", {}) or {},
        "parser_recommendation": read_json(OUT_ROOT / "parser_patch_recommendation.json", {}) or {},
        "reasoning_handoff": read_json(OUT_ROOT / "reasoning_terminal_handoff_v2.json", {}) or {},
        "reasoning_hard": read_json(OUT_ROOT / "reasoning_hard_slice_v2.json", {}) or {},
        "l47": read_json(OUT_ROOT / "science_l47_layer_ablation_v2.json", {}) or {},
        "escalation": read_json(OUT_ROOT / "science_perturbation_escalation_v2.json", {}) or {},
        "soft_hairs": read_json(OUT_ROOT / "soft_hairs_v2.json", {}) or {},
        "integrated": read_json(OUT_ROOT / "integrated_repair_eval.json", {}) or {},
        "readiness": read_json(OUT_ROOT / "pre_steering_readiness_v2.json", {}) or {},
    }
    verdicts = {
        "BG_SCIENCE_REASONING_REPAIR_V2_INVENTORY_VERDICT": inputs["inventory"].get("BG_SCIENCE_REASONING_REPAIR_V2_INVENTORY_VERDICT", "INSUFFICIENT"),
        "BG_SCIENCE_REASONING_REPAIR_V2_TASK_SUITE_VERDICT": inputs["task_suite"].get("BG_SCIENCE_REASONING_REPAIR_V2_TASK_SUITE_VERDICT", "INSUFFICIENT"),
        "BG_SCIENCE_PARSER_PATCH_CANDIDATES_VERDICT": inputs["parser_candidates"].get("BG_SCIENCE_PARSER_PATCH_CANDIDATES_VERDICT", "INSUFFICIENT"),
        "BG_SCIENCE_PARSER_VALIDATION_VERDICT": inputs["parser_validation"].get("BG_SCIENCE_PARSER_VALIDATION_VERDICT", "INSUFFICIENT"),
        "BG_SCIENCE_BRANCH_RECIPE_V2_PLAN_VERDICT": inputs["recipe_plan"].get("BG_SCIENCE_BRANCH_RECIPE_V2_PLAN_VERDICT", "INSUFFICIENT"),
        "BG_SCIENCE_RECIPE_V2_CALIBRATION_VERDICT": inputs["calibration"].get("BG_SCIENCE_RECIPE_V2_CALIBRATION_VERDICT", "INSUFFICIENT"),
        "BG_SCIENCE_RECIPE_V2_HELDOUT_VERDICT": inputs["heldout"].get("BG_SCIENCE_RECIPE_V2_HELDOUT_VERDICT", "INSUFFICIENT"),
        "BG_SCIENCE_V2_SOURCE_SPECIFIC_VERDICT": inputs["source"].get("BG_SCIENCE_V2_SOURCE_SPECIFIC_VERDICT", "INSUFFICIENT"),
        "BG_SCIENCE_PARSER_PATCH_RECOMMENDATION_VERDICT": inputs["parser_recommendation"].get("BG_SCIENCE_PARSER_PATCH_RECOMMENDATION_VERDICT", "INSUFFICIENT"),
        "BG_REASONING_TERMINAL_HANDOFF_V2_VERDICT": inputs["reasoning_handoff"].get("BG_REASONING_TERMINAL_HANDOFF_V2_VERDICT", "INSUFFICIENT"),
        "BG_REASONING_HARD_SLICE_V2_VERDICT": inputs["reasoning_hard"].get("BG_REASONING_HARD_SLICE_V2_VERDICT", "INSUFFICIENT"),
        "BG_SCIENCE_L47_LAYER_ABLATION_V2_VERDICT": inputs["l47"].get("BG_SCIENCE_L47_LAYER_ABLATION_V2_VERDICT", "INSUFFICIENT"),
        "BG_SCIENCE_PERTURBATION_ESCALATION_V2_VERDICT": inputs["escalation"].get("BG_SCIENCE_PERTURBATION_ESCALATION_V2_VERDICT", "INSUFFICIENT"),
        "BG_SOFT_HAIRS_V2_VERDICT": inputs["soft_hairs"].get("BG_SOFT_HAIRS_V2_VERDICT", "INSUFFICIENT"),
        "BG_SCIENCE_REASONING_INTEGRATED_REPAIR_V2_VERDICT": inputs["integrated"].get("BG_SCIENCE_REASONING_INTEGRATED_REPAIR_V2_VERDICT", "INSUFFICIENT"),
        "BG_PRE_STEERING_READINESS_V2_VERDICT": inputs["readiness"].get("BG_PRE_STEERING_READINESS_V2_VERDICT", "INSUFFICIENT"),
    }
    overall = status_from(verdicts["BG_PRE_STEERING_READINESS_V2_VERDICT"], verdicts["BG_SCIENCE_REASONING_INTEGRATED_REPAIR_V2_VERDICT"])
    files_created = sorted(str(path.relative_to(OUT_ROOT)) for path in OUT_ROOT.glob("*") if path.is_file())
    commands = [
        "py_compile requested scripts",
        "bg_science_reasoning_repair_v2_inventory.py",
        "build_bg_science_reasoning_repair_v2_task_suite.py",
        "build_bg_science_parser_patch_candidates_v2.py",
        "validate_bg_science_parser_patch_v2.py",
        "build_bg_science_branch_recipe_v2_plan.py",
        "run_bg_science_recipe_v2_calibration.py",
        "evaluate_bg_science_recipe_v2_heldout.py",
        "analyze_bg_science_v2_source_specific.py",
        "analyze_bg_science_parser_patch_recommendation_v2.py",
        "analyze_bg_reasoning_terminal_handoff_v2.py",
        "analyze_bg_reasoning_hard_slice_v2.py",
        "run_bg_science_l47_layer_ablation_v2.py",
        "run_bg_science_perturbation_escalation_v2.py",
        "analyze_bg_soft_hairs_science_reasoning_v2.py",
        "evaluate_bg_science_reasoning_integrated_repair_v2.py",
        "analyze_bg_pre_steering_readiness_v2.py",
        "analyze_bg_science_reasoning_repair_v2.py",
    ]
    payload = {
        **verdicts,
        "DUALANCHOR_SCIENCE_REASONING_REPAIR_V2_STATUS": overall,
        "files_created": files_created,
        "commands_run": commands,
        "decision": inputs["readiness"],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "summary.json", payload)
    write_json(OUT_ROOT / "analysis.json", {**payload, "inputs": inputs})
    top_lines = [status_line(k, v) for k, v in {**verdicts, "DUALANCHOR_SCIENCE_REASONING_REPAIR_V2_STATUS": overall}.items()]
    lines = [
        "# DualAnchor Science + Reasoning Repair v2",
        "",
        *top_lines,
        "",
        "## Motivation",
        "",
        "This run patches/evaluates science MCQ parsers, runs regenerated source-specific science branch recipes, locks reasoning terminal handoff, keeps convergence hairs soft-only, and decides whether steering can start.",
        "",
        "## Prior Science/Reasoning Potholes",
        "",
        "- science parser/reward was partly responsible",
        "- prior recipe work was replay-only",
        "- MMLU chemistry/anatomy were blocked",
        "- reasoning needed terminal survivor-set handoff",
        "",
        "## Task Suite",
        "",
        f"- verdict: `{verdicts['BG_SCIENCE_REASONING_REPAIR_V2_TASK_SUITE_VERDICT']}`",
        f"- inventory: `{inputs['task_suite'].get('inventory')}`",
        "",
        "## Science Parser Candidates",
        "",
        f"- verdict: `{verdicts['BG_SCIENCE_PARSER_PATCH_CANDIDATES_VERDICT']}`",
        "- experimental parser module: `utilities/tests/manual/bg_science_mcq_parser_v2.py`",
        "",
        "## Parser Validation",
        "",
        f"- verdict: `{verdicts['BG_SCIENCE_PARSER_VALIDATION_VERDICT']}`",
        f"- selected parser: `{inputs['parser_validation'].get('selected_parser')}`",
        "",
        "## Science Branch Recipe Plan",
        "",
        f"- verdict: `{verdicts['BG_SCIENCE_BRANCH_RECIPE_V2_PLAN_VERDICT']}`",
        "",
        "## Science Recipe Calibration",
        "",
        f"- verdict: `{verdicts['BG_SCIENCE_RECIPE_V2_CALIBRATION_VERDICT']}`",
        *md_table(inputs["calibration"].get("summary_rows", []), ["recipe_id", "task_count", "source_count", "selected_parser_positive_oracle_rate", "selected_parser_terminal_best_reward", "reward_diverse_rate", "terminal_oracle_retained"]),
        "",
        "## Science Heldout",
        "",
        f"- verdict: `{verdicts['BG_SCIENCE_RECIPE_V2_HELDOUT_VERDICT']}`",
        *md_table(inputs["heldout"].get("summary_rows", []), ["recipe_id", "task_count", "source_count", "selected_parser_positive_oracle_rate", "selected_parser_terminal_best_reward", "reward_diverse_rate", "terminal_oracle_retained"]),
        "",
        "## Science Source-Specific Diagnosis",
        "",
        f"- verdict: `{verdicts['BG_SCIENCE_V2_SOURCE_SPECIFIC_VERDICT']}`",
        *md_table(inputs["source"].get("source_rows", []), ["source_detail", "unique_task_count", "best_recipe", "selected_parser_positive_oracle_rate", "selected_parser_terminal_best_reward", "source_readiness"]),
        "",
        "## Parser Patch Recommendation",
        "",
        f"- verdict: `{verdicts['BG_SCIENCE_PARSER_PATCH_RECOMMENDATION_VERDICT']}`",
        "",
        "## Reasoning Terminal Handoff",
        "",
        f"- verdict: `{verdicts['BG_REASONING_TERMINAL_HANDOFF_V2_VERDICT']}`",
        *md_table(inputs["reasoning_handoff"].get("summary_rows", []), ["policy", "task_count", "oracle_retained", "best_selected_reward", "first_selected_oracle", "defer_rate"]),
        "",
        "## Reasoning Hard-Slice",
        "",
        f"- verdict: `{verdicts['BG_REASONING_HARD_SLICE_V2_VERDICT']}`",
        "",
        "## Science L47/Layer Ablation",
        "",
        f"- verdict: `{verdicts['BG_SCIENCE_L47_LAYER_ABLATION_V2_VERDICT']}`",
        "",
        "## Science Perturbation Escalation",
        "",
        f"- verdict: `{verdicts['BG_SCIENCE_PERTURBATION_ESCALATION_V2_VERDICT']}`",
        "",
        "## Soft Convergence-Hair Monitoring",
        "",
        f"- verdict: `{verdicts['BG_SOFT_HAIRS_V2_VERDICT']}`",
        "- L30/L42 hairs remain diagnostic-only; no hard merge.",
        "",
        "## Integrated Repair Evaluation",
        "",
        f"- verdict: `{verdicts['BG_SCIENCE_REASONING_INTEGRATED_REPAIR_V2_VERDICT']}`",
        "",
        "## Pre-Steering Readiness",
        "",
        f"- verdict: `{verdicts['BG_PRE_STEERING_READINESS_V2_VERDICT']}`",
        f"- status: `{overall}`",
        "",
        "## Locked Baseline If Ready",
        "",
        *[f"- {k}: `{v}`" for k, v in (inputs["readiness"].get("locked_baseline") or {}).items()],
        "",
        "## Files Created",
        "",
        *[f"- `{name}`" for name in files_created],
        "",
        "## Commands Run",
        "",
        *[f"- `{cmd}`" for cmd in commands],
        "",
        "## Blockers",
        "",
        "- Science headline readiness depends on the heldout verdict above; do not include science headline if status remains diagnostic/blocked.",
    ]
    write_md(OUT_ROOT / "summary.md", lines)
    write_md(OUT_ROOT / "analysis.md", lines)
    print(status_line("DUALANCHOR_SCIENCE_REASONING_REPAIR_V2_STATUS", overall))
    return 0
