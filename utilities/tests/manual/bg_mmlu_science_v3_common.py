"""Shared helpers for MMLU science branch generation + parser repair v3.

This run is pre-steering only. It may run regenerated cumulative-hook branch
recipe experiments, but it does not train Ouro, modify checkpoints or
tokenizers, update tap registries, run wrapper/local-agent code, import
Hunter-Seeker modules, apply steering, or claim compute savings / true
autoregressive fork-carry.
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import torch

import bg_science_reasoning_repair_v2_common as v2
from bg_branch_generator_v1_common import BASIS_BANK_PT, load_audit_plan
from bg_hidden_origin_tap_common import PROBE_ROOT
from bg_mmlu_science_parser_v3 import PARSER_IDS, option_map as parser_option_map, parse_with
from run_bg_dualanchor_recursive_lineage_probe_v1 import compact_row, evaluate_branch
from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor


SHORT_NAME = "mmlu_science_branch_parser_repair_v3"
OUT_ROOT = PROBE_ROOT / "bg_mmlu_science_branch_parser_repair_v3_2026-06-01"
V2_ROOT = PROBE_ROOT / "bg_dualanchor_science_reasoning_repair_v2_2026-06-01"
V1_ROOT = PROBE_ROOT / "bg_dualanchor_science_branch_recipe_reasoning_defer_v1_2026-05-31"
HAIRS_ROOT = PROBE_ROOT / "bg_dualanchor_convergence_hairs_reasoning_science_v1_2026-05-31"
V3_ARCH_ROOT = PROBE_ROOT / "bg_dualanchor_architecture_looped_stratified_probe_v3_2026-05-31"

PROGRESS_ROOT = OUT_ROOT / "progress"
CANDIDATE_TREE_ROOT = OUT_ROOT / "candidate_trees"
GENERATED_PT = OUT_ROOT / "recipe_v3_generated.pt"
GENERATED_ROWS_CSV = OUT_ROOT / "recipe_v3_generated_rows.csv"
GENERATED_TASK_ROWS_CSV = OUT_ROOT / "recipe_v3_generated_task_rows.csv"
GENERATED_TERMINAL_ROWS_CSV = OUT_ROOT / "recipe_v3_generated_terminal_rows.csv"
GENERATED_STAGE_ROWS_CSV = OUT_ROOT / "recipe_v3_generated_stage_rows.csv"
LINEAGE_INCREMENTAL_CSV = OUT_ROOT / "lineage_rows_incremental.csv"
COMPLETED_TASKS_JSON = OUT_ROOT / "completed_task_ids.json"
COMPLETED_RECIPES_JSON = OUT_ROOT / "completed_recipe_ids.json"
COMPLETED_SOURCES_JSON = OUT_ROOT / "completed_source_ids.json"
PARTIAL_SUMMARY_JSON = OUT_ROOT / "partial_summary.json"

PRIMARY_SOURCES = {
    "mmlu_high_school_chemistry",
    "mmlu_anatomy",
    "mmlu_high_school_physics",
    "mmlu_biology",
}
CONTROL_SOURCES = {"sciq", "openbookqa"}
ANCHOR_A = "MIX_CODE_REASONING"
ANCHOR_B = "MIX_OBJECTIVE_ALL"
LOCKED_SCHEDULE = "L1_24 -> L1_36 -> L1_47 -> L2_24 -> L2_36 -> L2_47 -> L3_24 -> L3_36 -> L3_47 -> L4_24 -> L4_36 -> terminal L4_47"


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
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in keys})


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


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
    xs: list[float] = []
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


def audit_plan_tasks() -> list[dict[str, Any]]:
    return [dict(row) for row in (load_audit_plan().get("tasks") or []) if row.get("correct_option")]


def source_detail(task_id: str, source_dataset: Any) -> str:
    return v2.source_detail(task_id, source_dataset)


def task_by_id() -> dict[str, dict[str, Any]]:
    return {str(row.get("task_id")): row for row in audit_plan_tasks()}


def source_rank(source: str) -> int:
    order = {
        "mmlu_high_school_chemistry": 0,
        "mmlu_anatomy": 1,
        "mmlu_high_school_physics": 2,
        "mmlu_biology": 3,
        "sciq": 4,
        "openbookqa": 5,
    }
    return order.get(source, 99)


def task_sort_key(row: dict[str, Any]) -> tuple[int, int, int, float, str]:
    split_rank = {"heldout": 0, "val": 1, "train": 2}.get(str(row.get("split")), 9)
    likely = 1 if truthy(row.get("heldout_likely_non_tie")) else 0
    prior = int(row.get("prior_non_tie_pairs") or 0)
    priority = safe_float(row.get("priority_score_v4") or row.get("priority_score"), 0.0)
    return (split_rank, -likely, -prior, -priority, str(row.get("task_id")))


def selected_science_tasks() -> list[dict[str, Any]]:
    tasks = []
    for row in audit_plan_tasks():
        source = source_detail(str(row.get("task_id")), row.get("source_dataset"))
        domain = str(row.get("domain"))
        if domain == "science" and (source in PRIMARY_SOURCES or source in CONTROL_SOURCES):
            tasks.append(dict(row))
    return sorted(tasks, key=lambda row: (source_rank(source_detail(str(row.get("task_id")), row.get("source_dataset"))), task_sort_key(row)))


def build_task_suite_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    science = selected_science_tasks()
    heldout_seen: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for index, task in enumerate(science):
        source = source_detail(str(task.get("task_id")), task.get("source_dataset"))
        split = str(task.get("split"))
        if split == "val":
            parser_split = "parser_calibration"
            recipe_split = "diagnostic"
            repair_split = "parser_calibration"
        elif split == "heldout":
            parser_split = "parser_heldout"
            recipe_split = "recipe_heldout" if heldout_seen[source] == 0 else "source_heldout"
            repair_split = recipe_split
            heldout_seen[source] += 1
        else:
            parser_split = "diagnostic"
            recipe_split = "recipe_calibration"
            repair_split = "recipe_calibration"
        rows.append(
            {
                "index": index,
                "task_id": task.get("task_id"),
                "domain": "science",
                "source_dataset": task.get("source_dataset"),
                "source_detail": source,
                "source_role": "primary_mmlu" if source in PRIMARY_SOURCES else "control",
                "split": split,
                "parser_split": parser_split,
                "recipe_split": recipe_split,
                "repair_split": repair_split,
                "correct_option": task.get("correct_option"),
                "options": task.get("options"),
                "question": task.get("question"),
                "parser_type": "mcq",
                "prior_status": prior_source_status(source),
                "heldout_likely_non_tie": truthy(task.get("heldout_likely_non_tie")),
                "prior_non_tie_pairs": int(task.get("prior_non_tie_pairs") or 0),
                "priority_score": safe_float(task.get("priority_score_v4") or task.get("priority_score"), 0.0),
            }
        )
    reasoning = sorted([row for row in audit_plan_tasks() if str(row.get("domain")) == "reasoning"], key=task_sort_key)[:24]
    for task in reasoning:
        rows.append(
            {
                "index": len(rows),
                "task_id": task.get("task_id"),
                "domain": "reasoning",
                "source_dataset": task.get("source_dataset"),
                "source_detail": source_detail(str(task.get("task_id")), task.get("source_dataset")),
                "source_role": "reasoning_guardrail",
                "split": task.get("split"),
                "parser_split": "diagnostic",
                "recipe_split": "reasoning_guardrail",
                "repair_split": "reasoning_guardrail",
                "correct_option": task.get("correct_option"),
                "options": task.get("options"),
                "question": task.get("question"),
                "parser_type": "mcq",
            }
        )
    source_counts = Counter(row["source_detail"] for row in rows if row["domain"] == "science")
    inv = {
        "selected_count": len(rows),
        "science_count": sum(1 for row in rows if row["domain"] == "science"),
        "reasoning_guardrail_count": sum(1 for row in rows if row["domain"] == "reasoning"),
        "science_source_counts": dict(source_counts),
        "repair_split_counts": dict(Counter(row["repair_split"] for row in rows)),
        "parser_split_counts": dict(Counter(row["parser_split"] for row in rows)),
        "recipe_split_counts": dict(Counter(row["recipe_split"] for row in rows)),
        "missing_requested_sources": [
            source
            for source in ("mmlu_biology", "openbookqa")
            if source_counts.get(source, 0) == 0
        ],
        "preferred_target_met": sum(source_counts.values()) >= 96,
        "minimum_target_met": (
            sum(source_counts.values()) >= 48
            and source_counts.get("mmlu_high_school_chemistry", 0) >= 12
            and source_counts.get("mmlu_anatomy", 0) >= 12
            and source_counts.get("mmlu_high_school_physics", 0) >= 8
            and sum(source_counts.get(src, 0) for src in CONTROL_SOURCES) >= 8
        ),
    }
    return rows, inv


def prior_source_status(source: str) -> str:
    mapping = {
        "mmlu_high_school_chemistry": "BRANCH_GENERATION_BLOCKED",
        "mmlu_anatomy": "BRANCH_GENERATION_BLOCKED",
        "mmlu_high_school_physics": "DATA_LIMITED",
        "sciq": "DATA_LIMITED",
    }
    return mapping.get(source, "UNKNOWN_OR_MISSING")


def suite_rows() -> list[dict[str, Any]]:
    data = read_json(OUT_ROOT / "task_suite.json", {}) or {}
    return list(data.get("task_rows") or [])


def suite_by_id() -> dict[str, dict[str, Any]]:
    return {str(row.get("task_id")): row for row in suite_rows()}


def option_map_for_task(task_id: str) -> dict[str, str]:
    meta = task_by_id().get(str(task_id)) or suite_by_id().get(str(task_id)) or {}
    options = meta.get("options") or {}
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except Exception:
            options = {}
    return parser_option_map(options)


def row_reward(row: dict[str, Any]) -> float:
    return v2.row_reward(row)


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
        "parser_split": suite.get("parser_split"),
        "recipe_split": suite.get("recipe_split"),
        "repair_split": suite.get("repair_split"),
        "parser_id": parser_id,
        "correct_option": correct,
        "parsed_option": parsed,
        "parse_success": result.parse_success,
        "parse_failure_reason": result.parse_failure_reason,
        "ambiguity_flag": result.ambiguity_flag,
        "confidence": result.confidence,
        "span_used": result.span_used,
        "matched_option_text": result.matched_option_text,
        "risk_flags": result.risk_flags,
        "parser_correct": bool(result.parse_success and parsed == correct),
        "parser_wrong": bool(result.parse_success and parsed and parsed != correct),
        "strict_parse_success": truthy(candidate.get("parse_success")),
        "strict_parsed_answer": candidate.get("parsed_answer"),
        "strict_reward": row_reward(candidate),
        "parser_reward": 1.0 if result.parse_success and parsed == correct else (-0.2 if result.parse_success else 0.0),
        "false_positive_risk_flag": bool(result.parse_success and parsed == correct and result.ambiguity_flag),
        "wrong_option_parse_from_abstain": bool((not truthy(candidate.get("parse_success"))) and result.parse_success and parsed != correct),
    }


def all_science_candidate_rows(include_generated: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    v3 = v2.load_v3_pt()
    rows.extend([row for row in v3.get("rows", []) if row.get("domain") == "science"])
    if include_generated and GENERATED_PT.exists():
        rows.extend([row for row in load_generated().get("rows", []) if row.get("domain") == "science"])
    suite_ids = {str(row.get("task_id")) for row in suite_rows() if row.get("domain") == "science"}
    if suite_ids:
        rows = [row for row in rows if str(row.get("task_id")) in suite_ids]
    return rows


def parser_validation_rows(parser_ids: Sequence[str] = PARSER_IDS, include_generated: bool = True) -> list[dict[str, Any]]:
    rows = []
    for candidate in all_science_candidate_rows(include_generated=include_generated):
        for parser_id in parser_ids:
            rows.append(parser_eval_row(candidate, parser_id))
    return rows


def parser_summary(rows: Sequence[dict[str, Any]], split_key: str = "parser_split", split_value: str | None = None) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if split_value and row.get(split_key) != split_value:
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


def select_parser(summary_rows: Sequence[dict[str, Any]], adversarial_rows: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    strict = next((row for row in summary_rows if row.get("parser_id") == "strict_letter_current"), {})
    adv_summary = adversarial_parser_summary(adversarial_rows or [])
    candidates = []
    for row in summary_rows:
        parser_id = str(row.get("parser_id"))
        adv = adv_summary.get(parser_id, {})
        if parser_id == "conservative_fuzzy_option_parser":
            continue
        if safe_float(row.get("false_positive_risk_rate"), 1.0) > 0.05:
            continue
        if safe_float(adv.get("false_positive_rate"), 1.0) > 0.03:
            continue
        if safe_float(row.get("ambiguity_rate"), 1.0) > 0.40:
            continue
        if safe_float(row.get("positive_oracle_rate"), 0.0) + 1e-9 < safe_float(strict.get("positive_oracle_rate"), 0.0):
            continue
        candidates.append({**row, "adversarial_false_positive_rate": adv.get("false_positive_rate"), "adversarial_ambiguity_rejection_rate": adv.get("ambiguity_rejection_rate")})
    if not candidates:
        return {"parser_id": "strict_letter_current", "selected": False, "selection_reason": "no parser met calibration/adversarial gates"}
    best = max(candidates, key=lambda row: (safe_float(row.get("positive_oracle_rate"), 0.0), safe_float(row.get("terminal_best_reward"), 0.0), safe_float(row.get("parse_success_rate"), 0.0)))
    selected = best.get("parser_id") != "strict_letter_current" and (
        safe_float(best.get("positive_oracle_rate"), 0.0) > safe_float(strict.get("positive_oracle_rate"), 0.0) + 0.02
        or safe_float(best.get("parse_success_rate"), 0.0) > safe_float(strict.get("parse_success_rate"), 0.0) + 0.05
    )
    return {**best, "selected": bool(selected), "selection_reason": "parser calibration plus adversarial gate"}


def selected_parser_id() -> str:
    data = read_json(OUT_ROOT / "selected_parser_v3.json", {}) or read_json(OUT_ROOT / "selected_science_parser.json", {}) or {}
    return str(data.get("parser_id") or "strict_letter_current")


def prompt_formats() -> dict[str, str]:
    return {
        "baseline_current_format": "baseline",
        "explicit_letter_options": "Question:\n{question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\nAnswer with only the letter.",
        "final_answer_marker": "Question:\n{question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\nReason briefly, then end with:\nFinal answer: <letter>",
        "no_rationale_letter_only": "Question:\n{question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\nAnswer only with A, B, C, or D.",
        "option_text_allowed": "Question:\n{question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\nEnd with:\nFinal answer: <letter> - <option text>",
        "mmlu_native_format": "{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:",
    }


def render_prompt(task: dict[str, Any], format_id: str) -> str:
    options = task.get("options") or {}
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except Exception:
            options = {}
    if format_id == "baseline_current_format":
        return str(task.get("prompt") or v2.task_to_prompt_task(task).get("prompt") or "")
    template = prompt_formats()[format_id]
    vals = {letter: dict(options).get(letter, "") for letter in ("A", "B", "C", "D", "E")}
    vals["question"] = task.get("question") or ""
    return template.format(**vals)


def selected_prompt_format() -> str:
    data = read_json(OUT_ROOT / "selected_prompt_format.json", {}) or {}
    return str(data.get("format_id") or "baseline_current_format")


def task_to_prompt_task(row: dict[str, Any]) -> dict[str, Any]:
    meta = task_by_id().get(str(row.get("task_id")), {})
    out = dict(meta)
    out.update({k: row.get(k) for k in ("task_id", "domain", "source_dataset", "split", "correct_option") if row.get(k) is not None})
    out["prompt"] = render_prompt(out, selected_prompt_format())
    return out


def recipe_configs() -> dict[str, dict[str, Any]]:
    return {
        "baseline_v3_regenerated": {"recipe_id": "baseline_v3_regenerated", "family": "old_tap_aligned", "alpha": 0.005, "default_children": 1},
        "L2_47_emphasis": {"recipe_id": "L2_47_emphasis", "family": "old_tap_aligned", "alpha": 0.005, "default_children": 1, "stage_children": {"L2_47": 2}},
        "L47_heavy": {"recipe_id": "L47_heavy", "family": "old_tap_aligned", "alpha": 0.005, "default_children": 1, "layer_children": {47: 2}},
        "L24_heavy": {"recipe_id": "L24_heavy", "family": "old_tap_aligned", "alpha": 0.005, "default_children": 1, "layer_children": {24: 2}},
        "L36_heavy": {"recipe_id": "L36_heavy", "family": "old_tap_aligned", "alpha": 0.005, "default_children": 1, "layer_children": {36: 2}},
        "L24_L47_combo": {"recipe_id": "L24_L47_combo", "family": "old_tap_aligned", "alpha": 0.005, "default_children": 1, "layer_children": {24: 2, 47: 2}},
        "L36_L47_combo": {"recipe_id": "L36_L47_combo", "family": "old_tap_aligned", "alpha": 0.005, "default_children": 1, "layer_children": {36: 2, 47: 2}},
        "DualAnchor_aligned": {"recipe_id": "DualAnchor_aligned", "family": "v4_tap_aligned", "alpha": 0.005, "default_children": 1},
        "bridge_materialization": {"recipe_id": "bridge_materialization", "family": "high_yield_recipe_direction", "alpha": 0.005, "default_children": 1},
        "old_content_residual": {"recipe_id": "old_content_residual", "family": "old_tap_aligned", "alpha": 0.005, "default_children": 1},
        "random_orthogonal_control": {"recipe_id": "random_orthogonal_control", "family": "random_orthogonal", "alpha": 0.005, "default_children": 1, "control": True},
        "more_children_same_budget": {"recipe_id": "more_children_same_budget", "family": "old_tap_aligned", "alpha": 0.005, "default_children": 2, "breadth_diagnostic": True},
        "alpha_005_only": {"recipe_id": "alpha_005_only", "family": "old_tap_aligned", "alpha": 0.005, "default_children": 1},
        "alpha_01_only": {"recipe_id": "alpha_01_only", "family": "old_tap_aligned", "alpha": 0.01, "default_children": 1},
        "alpha_mixed_005_01": {"recipe_id": "alpha_mixed_005_01", "family": "old_tap_aligned", "alpha": 0.01, "default_children": 1, "stage_children": {"L1_24": 2}},
        "alpha_02_diagnostic": {"recipe_id": "alpha_02_diagnostic", "family": "old_tap_aligned", "alpha": 0.02, "default_children": 1, "diagnostic": True},
        "perturbation_escalation_diagnostic": {"recipe_id": "perturbation_escalation_diagnostic", "family": "old_tap_aligned", "alpha": 0.01, "default_children": 2, "diagnostic": True},
    }


def source_recipe_ids(source: str) -> list[str]:
    mapping = {
        "mmlu_high_school_chemistry": ["baseline_v3_regenerated", "L24_heavy", "L36_heavy", "L2_47_emphasis", "bridge_materialization", "DualAnchor_aligned", "more_children_same_budget", "random_orthogonal_control"],
        "mmlu_anatomy": ["baseline_v3_regenerated", "L24_heavy", "bridge_materialization", "old_content_residual", "more_children_same_budget", "L24_L47_combo", "random_orthogonal_control"],
        "mmlu_high_school_physics": ["baseline_v3_regenerated", "L2_47_emphasis", "L47_heavy", "L36_L47_combo", "DualAnchor_aligned", "random_orthogonal_control"],
        "mmlu_biology": ["baseline_v3_regenerated", "L24_heavy", "bridge_materialization", "old_content_residual", "random_orthogonal_control"],
        "sciq": ["baseline_v3_regenerated", "alpha_005_only", "alpha_01_only", "random_orthogonal_control"],
        "openbookqa": ["baseline_v3_regenerated", "alpha_005_only", "random_orthogonal_control"],
    }
    return mapping.get(source, ["baseline_v3_regenerated", "L24_heavy", "random_orthogonal_control"])


def repair_split_tasks(repair_split: str, *, max_per_source: int | None = None) -> list[dict[str, Any]]:
    rows = [row for row in suite_rows() if row.get("repair_split") == repair_split and row.get("domain") == "science"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("source_detail"))].append(row)
    out: list[dict[str, Any]] = []
    for source, vals in sorted(grouped.items(), key=lambda item: source_rank(item[0])):
        vals = sorted(vals, key=lambda row: str(row.get("task_id")))
        out.extend(vals[:max_per_source] if max_per_source else vals)
    return out


def configure_v2_engine() -> None:
    ensure_root()
    v2.OUT_ROOT = OUT_ROOT
    v2.PROGRESS_ROOT = PROGRESS_ROOT
    v2.CANDIDATE_TREE_ROOT = CANDIDATE_TREE_ROOT
    v2.GENERATED_PT = GENERATED_PT
    v2.GENERATED_ROWS_CSV = GENERATED_ROWS_CSV
    v2.GENERATED_TASK_ROWS_CSV = GENERATED_TASK_ROWS_CSV
    v2.GENERATED_TERMINAL_ROWS_CSV = GENERATED_TERMINAL_ROWS_CSV
    v2.GENERATED_STAGE_ROWS_CSV = GENERATED_STAGE_ROWS_CSV
    v2.LINEAGE_INCREMENTAL_CSV = LINEAGE_INCREMENTAL_CSV
    v2.COMPLETED_TASKS_JSON = COMPLETED_TASKS_JSON
    v2.COMPLETED_RECIPES_JSON = COMPLETED_RECIPES_JSON
    v2.COMPLETED_SOURCES_JSON = COMPLETED_SOURCES_JSON
    v2.PARTIAL_SUMMARY_JSON = PARTIAL_SUMMARY_JSON
    v2.suite_rows = suite_rows
    v2.suite_by_id = suite_by_id
    v2.task_by_id = task_by_id
    v2.option_map_for_task = option_map_for_task
    v2.parser_eval_row = parser_eval_row
    v2.apply_selected_parser_metrics = apply_selected_parser_metrics
    v2.selected_parser_id = selected_parser_id
    v2.task_to_prompt_task = task_to_prompt_task
    v2.recipe_configs = recipe_configs
    v2.source_recipe_ids = source_recipe_ids
    v2.repair_split_tasks = repair_split_tasks


def load_generated() -> dict[str, Any]:
    configure_v2_engine()
    return v2.load_generated()


def save_generated(payload: dict[str, Any]) -> None:
    configure_v2_engine()
    v2.save_generated(payload)


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
    configure_v2_engine()
    if "REPAIR_V2_MAX_NEW_TOKENS" not in os.environ:
        os.environ["REPAIR_V2_MAX_NEW_TOKENS"] = os.environ.get("REPAIR_V3_MAX_NEW_TOKENS", "40")
    if "REPAIR_V2_HELDOUT_TASKS_PER_SOURCE" not in os.environ:
        os.environ["REPAIR_V2_HELDOUT_TASKS_PER_SOURCE"] = os.environ.get("REPAIR_V3_TASKS_PER_SOURCE", "99")
    return v2.run_recipe_generation(repair_split, mode=mode)


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


def summarize_by_recipe(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return v2.summarize_generated_by_recipe(rows)


def summarize_by_source(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return v2.summarize_generated_by_source(rows)


def select_best_recipes(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    best = v2.select_best_recipes(rows)
    best["selection_run"] = "mmlu_science_v3_calibration_only"
    return best


def metric_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task_count": len(rows),
        "positive_oracle_rate": finite_mean(row.get("positive_oracle") for row in rows),
        "selected_parser_positive_oracle_rate": finite_mean(row.get("selected_parser_positive_oracle") for row in rows),
        "reward_diverse_rate": finite_mean(row.get("terminal_reward_diverse") for row in rows),
        "terminal_best_reward": finite_mean(row.get("terminal_best_reward") for row in rows),
        "selected_parser_terminal_best_reward": finite_mean(row.get("selected_parser_terminal_best_reward") for row in rows),
        "forced_top1_reward": finite_mean(row.get("terminal_forced_top1_reward") for row in rows),
        "terminal_oracle_retained": finite_mean(row.get("terminal_oracle_retained") for row in rows),
        "parse_success": finite_mean(row.get("selected_parser_parse_success") for row in rows),
    }


def inventory_main() -> int:
    started = time.time()
    rows = selected_science_tasks()
    source_counts = Counter(source_detail(str(row.get("task_id")), row.get("source_dataset")) for row in rows)
    artifacts = [
        {"artifact": "v2_science_reasoning_repair", "path": str(V2_ROOT), "status": "READY" if V2_ROOT.exists() else "MISSING"},
        {"artifact": "v1_science_recipe_reasoning_defer", "path": str(V1_ROOT), "status": "READY" if V1_ROOT.exists() else "MISSING"},
        {"artifact": "convergence_hairs_v1", "path": str(HAIRS_ROOT), "status": "READY" if HAIRS_ROOT.exists() else "MISSING"},
        {"artifact": "dualanchor_architecture_v3", "path": str(V3_ARCH_ROOT / "architecture_looped_rows.pt"), "status": "READY" if (V3_ARCH_ROOT / "architecture_looped_rows.pt").exists() else "MISSING"},
        {"artifact": "basis_bank", "path": str(BASIS_BANK_PT), "status": "READY" if BASIS_BANK_PT.exists() else "MISSING"},
    ]
    missing = [row["artifact"] for row in artifacts if row["status"] == "MISSING"]
    if "dualanchor_architecture_v3" in missing or "basis_bank" in missing:
        verdict = "BLOCKED"
    elif len(rows) < 48:
        verdict = "MMLU_TASKS_LIMITED"
    elif missing:
        verdict = "PARTIAL"
    else:
        verdict = "READY"
    payload = {
        "BG_MMLU_SCIENCE_V3_INVENTORY_VERDICT": verdict,
        "artifact_rows": artifacts,
        "source_task_counts": dict(source_counts),
        "science_task_count": len(rows),
        "missing_requested_sources": [src for src in ("mmlu_biology", "openbookqa") if source_counts.get(src, 0) == 0],
        "prior_blockers": {
            "mmlu_high_school_chemistry": "BRANCH_GENERATION_BLOCKED",
            "mmlu_anatomy": "BRANCH_GENERATION_BLOCKED",
            "mmlu_high_school_physics": "DATA_LIMITED",
            "sciq": "DATA_LIMITED",
        },
        "can_regenerate": verdict != "BLOCKED",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "inventory.json", payload)
    write_md(
        OUT_ROOT / "inventory.md",
        [
            "# MMLU Science Branch/Parser Repair v3 Inventory",
            "",
            status_line("BG_MMLU_SCIENCE_V3_INVENTORY_VERDICT", verdict),
            "",
            "## Artifacts",
            "",
            *md_table(artifacts, ["artifact", "status", "path"]),
            "",
            "## Source Task Counts",
            "",
            *[f"- {k}: `{v}`" for k, v in sorted(source_counts.items(), key=lambda item: source_rank(item[0]))],
            "",
            f"- missing requested sources: `{payload['missing_requested_sources']}`",
            "- no steering/training/routing change",
        ],
    )
    print(status_line("BG_MMLU_SCIENCE_V3_INVENTORY_VERDICT", verdict))
    return 0 if verdict != "BLOCKED" else 1


def task_suite_main() -> int:
    started = time.time()
    rows, inv = build_task_suite_rows()
    source_counts = inv["science_source_counts"]
    if inv["minimum_target_met"]:
        verdict = "READY"
    elif source_counts.get("mmlu_high_school_chemistry", 0) < 12 or source_counts.get("mmlu_anatomy", 0) < 12:
        verdict = "CHEM_ANATOMY_LIMITED"
    elif source_counts.get("mmlu_high_school_physics", 0) < 8:
        verdict = "PHYSICS_LIMITED"
    elif sum(source_counts.get(src, 0) for src in CONTROL_SOURCES) < 8:
        verdict = "CONTROLS_LIMITED"
    else:
        verdict = "PARTIAL"
    payload = {
        "BG_MMLU_SCIENCE_V3_TASK_SUITE_VERDICT": verdict,
        "task_rows": rows,
        "inventory": inv,
        "selection_note": "Local pool is smaller than requested; all available MMLU science tasks are retained with heldout separated from calibration.",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "task_suite.json", payload)
    write_csv(OUT_ROOT / "task_suite.csv", rows)
    write_md(
        OUT_ROOT / "task_suite.md",
        [
            "# MMLU Science v3 Task Suite",
            "",
            status_line("BG_MMLU_SCIENCE_V3_TASK_SUITE_VERDICT", verdict),
            "",
            "## Inventory",
            "",
            *[f"- {k}: `{v}`" for k, v in inv.items() if not isinstance(v, dict)],
            "",
            "## Science Sources",
            "",
            *[f"- {k}: `{v}`" for k, v in sorted(source_counts.items(), key=lambda item: source_rank(item[0]))],
            "",
            "## Tasks",
            "",
            *md_table(rows, ["index", "task_id", "domain", "source_detail", "source_role", "split", "parser_split", "recipe_split", "repair_split", "prior_status"]),
        ],
    )
    print(status_line("BG_MMLU_SCIENCE_V3_TASK_SUITE_VERDICT", verdict))
    return 0


def prompt_format_audit_main() -> int:
    started = time.time()
    ensure_root()
    rows = read_json(OUT_ROOT / "prompt_format_audit.json", {}) or {}
    existing_rows = list(rows.get("rows") or [])
    completed = {f"{row.get('task_id')}::{row.get('format_id')}" for row in existing_rows}
    tasks = [row for row in suite_rows() if row.get("parser_split") == "parser_calibration" and row.get("domain") == "science"]
    if not tasks:
        tasks = [row for row in suite_rows() if row.get("domain") == "science"][:4]
    force = os.environ.get("FORCE_RERUN") == "1"
    format_ids = list(prompt_formats())
    out_rows = existing_rows if not force else []
    if force:
        completed = set()
    extractor: BGTransformerFeatureExtractor | None = None
    if tasks:
        extractor = BGTransformerFeatureExtractor(device="cuda" if torch.cuda.is_available() else "cpu", dtype="auto", force_all_loops=True)
    try:
        for suite_row in tasks:
            meta = task_by_id().get(str(suite_row.get("task_id")), {})
            task = dict(meta)
            task.update(suite_row)
            for format_id in format_ids:
                key = f"{suite_row.get('task_id')}::{format_id}"
                if key in completed:
                    continue
                prompt = render_prompt(task, format_id)
                eval_task = dict(task)
                eval_task["prompt"] = prompt
                branch = evaluate_branch(
                    model=extractor.model,
                    tokenizer=extractor.tokenizer,
                    task=eval_task,
                    branch_group_id=f"prompt_format_v3::{format_id}::{suite_row.get('task_id')}",
                    branch_id=f"prompt_format_v3::{format_id}::{suite_row.get('task_id')}::root",
                    parent=None,
                    hooks=[],
                    birth_layer=None,
                    birth_loop=None,
                    birth_stage="root",
                    perturbation_family="clean",
                    mutation_index=0,
                    device=extractor.device,
                    max_new_tokens=int(os.environ.get("REPAIR_V3_PROMPT_MAX_NEW_TOKENS", "40")),
                )
                for parser_id in PARSER_IDS:
                    parsed = parser_eval_row(branch, parser_id)
                    out_rows.append(
                        {
                            **compact_row(branch),
                            "format_id": format_id,
                            "prompt_length": len(prompt),
                            "parser_id": parser_id,
                            "parser_parse_success": parsed["parse_success"],
                            "parser_correct": parsed["parser_correct"],
                            "parser_wrong": parsed["parser_wrong"],
                            "parser_ambiguity": parsed["ambiguity_flag"],
                            "parser_reward": parsed["parser_reward"],
                            "source_detail": suite_row.get("source_detail"),
                        }
                    )
                write_json(PROGRESS_ROOT / f"prompt_format_{v2.safe_name(key)}.json", {"completed": True, "saved_at": time.time()})
                write_json(OUT_ROOT / "prompt_format_audit.json", {"rows": out_rows, "partial": True})
    finally:
        if extractor is not None:
            extractor.cleanup()
    summary = prompt_format_summary(out_rows)
    selected = select_prompt_format(summary)
    if not summary:
        verdict = "INSUFFICIENT"
    elif selected.get("format_id") == "baseline_current_format":
        verdict = "FORMAT_NOT_MAIN_ISSUE"
    elif "final_answer" in str(selected.get("format_id")):
        verdict = "FINAL_ANSWER_MARKER_HELPS"
    elif selected.get("format_id") == "no_rationale_letter_only":
        verdict = "LETTER_ONLY_HELPS_PARSE_ONLY"
    else:
        verdict = "FORMAT_FIX_HELPS"
    payload = {
        "BG_MMLU_SCIENCE_PROMPT_FORMAT_VERDICT": verdict,
        "rows": out_rows,
        "summary_rows": summary,
        "selected_prompt_format": selected,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "prompt_format_audit.json", payload)
    write_json(OUT_ROOT / "selected_prompt_format.json", selected)
    write_csv(OUT_ROOT / "prompt_format_rows.csv", out_rows)
    write_md(
        OUT_ROOT / "prompt_format_audit.md",
        [
            "# MMLU Science Prompt Format Audit v3",
            "",
            status_line("BG_MMLU_SCIENCE_PROMPT_FORMAT_VERDICT", verdict),
            "",
            "## Summary",
            "",
            *md_table(summary, ["format_id", "parser_id", "candidate_count", "parse_success_rate", "correct_rate", "wrong_rate", "ambiguity_rate", "positive_oracle_rate", "terminal_best_reward"]),
            "",
            "## Selected Prompt Format",
            "",
            *[f"- {k}: `{fmt(v)}`" for k, v in selected.items()],
        ],
    )
    print(status_line("BG_MMLU_SCIENCE_PROMPT_FORMAT_VERDICT", verdict))
    return 0


def prompt_format_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("format_id")), str(row.get("parser_id")))].append(row)
    out = []
    for (format_id, parser_id), vals in sorted(grouped.items()):
        by_task: dict[str, list[float]] = defaultdict(list)
        for row in vals:
            by_task[str(row.get("task_id"))].append(safe_float(row.get("parser_reward"), 0.0))
        best = [max(v) for v in by_task.values() if v]
        out.append(
            {
                "format_id": format_id,
                "parser_id": parser_id,
                "candidate_count": len(vals),
                "task_count": len(by_task),
                "parse_success_rate": finite_mean(row.get("parser_parse_success") for row in vals),
                "correct_rate": finite_mean(row.get("parser_correct") for row in vals),
                "wrong_rate": finite_mean(row.get("parser_wrong") for row in vals),
                "ambiguity_rate": finite_mean(row.get("parser_ambiguity") for row in vals),
                "positive_oracle_rate": finite_mean(1.0 if reward > 0 else 0.0 for reward in best),
                "terminal_best_reward": finite_mean(best),
            }
        )
    return out


def select_prompt_format(summary: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not summary:
        return {"format_id": "baseline_current_format", "selected": False, "reason": "no prompt audit rows"}
    robust_ids = {"strict_letter_current", "final_answer_span_letter_parser", "MCQ_robust_ambiguity_rejecting_parser", "source_specific_mmlu_parser"}
    candidates = [row for row in summary if row.get("parser_id") in robust_ids and safe_float(row.get("wrong_rate"), 1.0) <= 0.35]
    if not candidates:
        candidates = list(summary)
    best = max(candidates, key=lambda row: (safe_float(row.get("positive_oracle_rate"), 0.0), safe_float(row.get("terminal_best_reward"), 0.0), safe_float(row.get("parse_success_rate"), 0.0), -safe_float(row.get("ambiguity_rate"), 1.0)))
    return {**best, "selected": str(best.get("format_id")) != "baseline_current_format", "reason": "parser_calibration prompt-format gate"}


def parser_build_main() -> int:
    started = time.time()
    candidates = [
        {"parser_id": parser_id, "description": desc}
        for parser_id, desc in {
            "strict_letter_current": "existing strict parser result from candidate rows",
            "normalized_letter_parser": "normalizes boxed/lowercase/punctuated letters",
            "final_answer_span_letter_parser": "requires final-answer span and parses a unique letter",
            "exact_option_text_parser": "unique exact option text parser",
            "final_answer_span_text_parser": "exact option text inside final-answer span",
            "conservative_fuzzy_option_parser": "high-threshold fuzzy parser, diagnostic",
            "source_specific_mmlu_parser": "MMLU robust parser, controls normalized letter",
            "MCQ_robust_ambiguity_rejecting_parser": "final span, exact text, conservative fallback, ambiguity rejection",
        }.items()
    ]
    verdict = "READY"
    payload = {
        "BG_MMLU_SCIENCE_PARSER_V3_BUILD_VERDICT": verdict,
        "parser_candidates": candidates,
        "experimental_module": "shared/utilities/tests/manual/bg_mmlu_science_parser_v3.py",
        "ambiguity_rejection": [
            "multiple options without final marker rejected",
            "fuzzy tie rejected",
            "negated option ambiguity rejected",
            "contradictory final/no-final outputs flagged",
        ],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "parser_v3_candidates.json", payload)
    write_md(
        OUT_ROOT / "parser_v3_candidates.md",
        [
            "# MMLU Science Parser v3 Candidates",
            "",
            status_line("BG_MMLU_SCIENCE_PARSER_V3_BUILD_VERDICT", verdict),
            "",
            *md_table(candidates, ["parser_id", "description"]),
            "",
            "Strict parser remains available side-by-side.",
        ],
    )
    print(status_line("BG_MMLU_SCIENCE_PARSER_V3_BUILD_VERDICT", verdict))
    return 0


def adversarial_cases() -> list[dict[str, Any]]:
    options = {"A": "alpha option", "B": "beta option", "C": "gamma option", "D": "delta option"}
    return [
        {"case_id": "mentions_all_final_c", "output_text": "A and B are tempting. C fits best. Final answer: C", "options": options, "correct_option": "C", "expected": "C"},
        {"case_id": "not_a_not_b_final_c", "output_text": "It is not A and not B. Final answer: C", "options": options, "correct_option": "C", "expected": "C"},
        {"case_id": "wrong_text_correct_letter", "output_text": "The beta option is discussed, but Final answer: C", "options": options, "correct_option": "C", "expected": "C"},
        {"case_id": "correct_text_wrong_letter", "output_text": "The gamma option seems right. Final answer: B", "options": options, "correct_option": "C", "expected": "B"},
        {"case_id": "no_final", "output_text": "The best option may be gamma option.", "options": options, "correct_option": "C", "expected": "C"},
        {"case_id": "multiple_final", "output_text": "Final answer: B or C", "options": options, "correct_option": "C", "expected": None},
        {"case_id": "text_without_letter", "output_text": "Final answer: gamma option", "options": options, "correct_option": "C", "expected": "C"},
        {"case_id": "lowercase", "output_text": "final answer: c.", "options": options, "correct_option": "C", "expected": "C"},
        {"case_id": "cannot_determine", "output_text": "I cannot determine the answer.", "options": options, "correct_option": "C", "expected": None},
        {"case_id": "invalid_letter", "output_text": "Final answer: F", "options": options, "correct_option": "C", "expected": None},
    ]


def parser_adversarial_validation_main() -> int:
    started = time.time()
    synthetic_rows = []
    for case in adversarial_cases():
        for parser_id in PARSER_IDS:
            result = parse_with(parser_id, case["output_text"], case["options"], None, False, source="mmlu_high_school_chemistry")
            parsed = result.parsed_option
            expected = case.get("expected")
            false_positive = bool(parsed and expected is None)
            synthetic_rows.append(
                {
                    "row_type": "synthetic",
                    "case_id": case["case_id"],
                    "parser_id": parser_id,
                    "expected": expected,
                    "parsed_option": parsed,
                    "parse_success": result.parse_success,
                    "parse_failure_reason": result.parse_failure_reason,
                    "ambiguity_flag": result.ambiguity_flag,
                    "risk_flags": result.risk_flags,
                    "correct_parse": bool(expected is not None and parsed == expected),
                    "false_positive": false_positive,
                    "false_negative": bool(expected is not None and not result.parse_success),
                    "ambiguity_rejected": bool(expected is None and not result.parse_success),
                }
            )
    real_rows = parser_validation_rows(include_generated=False)
    calibration = parser_summary(real_rows, "parser_split", "parser_calibration") or parser_summary(real_rows)
    selected = select_parser(calibration, synthetic_rows)
    verdict = parser_adversarial_verdict(selected, synthetic_rows, calibration)
    payload = {
        "BG_MMLU_SCIENCE_PARSER_ADVERSARIAL_VERDICT": verdict,
        "synthetic_summary": list(adversarial_parser_summary(synthetic_rows).values()),
        "calibration_summary": calibration,
        "selected_parser": selected,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "parser_adversarial_validation.json", payload)
    write_json(OUT_ROOT / "selected_parser_v3.json", selected)
    write_json(OUT_ROOT / "selected_science_parser.json", selected)
    write_csv(OUT_ROOT / "parser_adversarial_rows.csv", synthetic_rows + real_rows)
    write_md(
        OUT_ROOT / "parser_adversarial_validation.md",
        [
            "# MMLU Science Parser Adversarial Validation v3",
            "",
            status_line("BG_MMLU_SCIENCE_PARSER_ADVERSARIAL_VERDICT", verdict),
            "",
            "## Synthetic Summary",
            "",
            *md_table(payload["synthetic_summary"], ["parser_id", "case_count", "correct_parse_rate", "false_positive_rate", "false_negative_rate", "ambiguity_rejection_rate"]),
            "",
            "## Calibration Summary",
            "",
            *md_table(calibration, ["parser_id", "candidate_count", "task_count", "parse_success_rate", "positive_oracle_rate", "terminal_best_reward", "ambiguity_rate", "false_positive_risk_rate"]),
            "",
            "## Selected Parser",
            "",
            *[f"- {k}: `{fmt(v)}`" for k, v in selected.items()],
        ],
    )
    print(status_line("BG_MMLU_SCIENCE_PARSER_ADVERSARIAL_VERDICT", verdict))
    return 0 if verdict != "BLOCKED" else 1


def adversarial_parser_summary(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("row_type") != "synthetic":
            continue
        grouped[str(row.get("parser_id"))].append(row)
    out = {}
    for parser_id, vals in sorted(grouped.items()):
        out[parser_id] = {
            "parser_id": parser_id,
            "case_count": len(vals),
            "correct_parse_rate": finite_mean(row.get("correct_parse") for row in vals),
            "false_positive_rate": finite_mean(row.get("false_positive") for row in vals),
            "false_negative_rate": finite_mean(row.get("false_negative") for row in vals),
            "ambiguity_rejection_rate": finite_mean(row.get("ambiguity_rejected") for row in vals if row.get("expected") is None),
        }
    return out


def parser_adversarial_verdict(selected: dict[str, Any], synthetic_rows: Sequence[dict[str, Any]], calibration: Sequence[dict[str, Any]]) -> str:
    if not calibration:
        return "DATA_LIMITED"
    summary = adversarial_parser_summary(synthetic_rows)
    parser_id = str(selected.get("parser_id") or "strict_letter_current")
    adv = summary.get(parser_id, {})
    if safe_float(adv.get("false_positive_rate"), 1.0) > 0.03:
        return "FALSE_POSITIVE_RISK_HIGH"
    if parser_id == "strict_letter_current":
        return "STRICT_REMAINS_PRIMARY"
    if parser_id == "source_specific_mmlu_parser":
        return "SOURCE_SPECIFIC_PARSER_READY"
    if bool(selected.get("selected")):
        return "PARSER_V3_READY"
    return "ROBUST_PARSER_STILL_DIAGNOSTIC"


def recipe_plan_main() -> int:
    started = time.time()
    recipes = list(recipe_configs().values())
    verdict = "SOURCE_SPECIFIC_READY" if suite_rows() else "BLOCKED"
    payload = {
        "BG_MMLU_SCIENCE_RECIPE_V3_PLAN_VERDICT": verdict,
        "recipes": recipes,
        "source_recipe_ids": {source: source_recipe_ids(source) for source in ("mmlu_high_school_chemistry", "mmlu_anatomy", "mmlu_high_school_physics", "mmlu_biology", "sciq", "openbookqa")},
        "baseline": {
            "schedule": LOCKED_SCHEDULE,
            "selector": [ANCHOR_A, ANCHOR_B],
            "threshold": "mean_floor_very_loose",
            "budget": 8,
            "l47": "active in nonterminal loops",
            "terminal": "confidence-gated top1; otherwise survivor handoff",
            "convergence_hairs": "soft-only",
        },
        "budget_note": "true budget10/12 policies are not available in the existing guarded engine; breadth diagnostics use more children before budget8 pruning.",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "science_recipe_v3_plan.json", payload)
    write_md(
        OUT_ROOT / "science_recipe_v3_plan.md",
        [
            "# MMLU Science Recipe v3 Plan",
            "",
            status_line("BG_MMLU_SCIENCE_RECIPE_V3_PLAN_VERDICT", verdict),
            "",
            "## Recipes",
            "",
            *md_table(recipes, ["recipe_id", "family", "alpha", "default_children", "diagnostic", "control", "breadth_diagnostic"]),
            "",
            "## Source Recipes",
            "",
            *[f"- {k}: `{v}`" for k, v in payload["source_recipe_ids"].items()],
            "",
            f"- budget note: `{payload['budget_note']}`",
        ],
    )
    print(status_line("BG_MMLU_SCIENCE_RECIPE_V3_PLAN_VERDICT", verdict))
    return 0 if verdict != "BLOCKED" else 1


def calibration_main() -> int:
    started = time.time()
    generated = run_recipe_generation("recipe_calibration", mode="calibration")
    rows = [row for row in generated.get("task_rows", []) if row.get("repair_split") == "recipe_calibration"]
    summary = summarize_by_recipe(rows)
    source_summary = summarize_by_source(rows)
    best = select_best_recipes(rows)
    if generated.get("errors") and not rows:
        verdict = "BLOCKED"
    elif not rows:
        verdict = "DATA_LIMITED"
    elif best.get("selected"):
        verdict = "SOURCE_SPECIFIC_RECIPE_FOUND"
    elif any(safe_float(row.get("selected_parser_positive_oracle_rate"), 0.0) >= 0.20 for row in summary):
        verdict = "RECIPE_WEAK_BUT_IMPROVED"
    else:
        verdict = "NO_RECIPE_IMPROVEMENT"
    payload = {
        "BG_MMLU_SCIENCE_RECIPE_V3_CALIBRATION_VERDICT": verdict,
        "summary_rows": summary,
        "source_rows": source_summary,
        "best_mmlu_science_recipe_v3": best,
        "generated_task_rows": len(rows),
        "errors": generated.get("errors", []),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "recipe_v3_calibration.json", payload)
    write_json(OUT_ROOT / "best_mmlu_science_recipe_v3.json", best)
    write_json(OUT_ROOT / "best_science_recipe_v2.json", best)
    write_csv(OUT_ROOT / "recipe_v3_calibration_rows.csv", rows)
    write_md(
        OUT_ROOT / "recipe_v3_calibration.md",
        [
            "# Regenerated MMLU Science Recipe v3 Calibration",
            "",
            status_line("BG_MMLU_SCIENCE_RECIPE_V3_CALIBRATION_VERDICT", verdict),
            "",
            "## Recipe Summary",
            "",
            *md_table(summary, ["recipe_id", "task_count", "source_count", "selected_parser_positive_oracle_rate", "selected_parser_terminal_best_reward", "reward_diverse_rate", "terminal_oracle_retained"]),
            "",
            "## Source Summary",
            "",
            *md_table(source_summary, ["source_detail", "unique_task_count", "best_recipe", "selected_parser_positive_oracle_rate", "selected_parser_terminal_best_reward", "parse_success"]),
            "",
            "## Selected Recipes",
            "",
            *[f"- {k}: `{fmt(v)}`" for k, v in best.items() if k != "selected_by_source"],
            f"- selected_by_source: `{best.get('selected_by_source')}`",
        ],
    )
    print(status_line("BG_MMLU_SCIENCE_RECIPE_V3_CALIBRATION_VERDICT", verdict))
    return 0 if verdict != "BLOCKED" else 1


def heldout_main() -> int:
    started = time.time()
    generated_a = run_recipe_generation("recipe_heldout", mode="heldout")
    generated_b = run_recipe_generation("source_heldout", mode="heldout")
    rows = [row for row in load_generated().get("task_rows", []) if row.get("repair_split") in {"recipe_heldout", "source_heldout"}]
    summary = summarize_by_recipe(rows)
    source_rows = summarize_by_source(rows)
    verdict = heldout_verdict(source_rows, rows)
    payload = {
        "BG_MMLU_SCIENCE_RECIPE_V3_HELDOUT_VERDICT": verdict,
        "summary_rows": summary,
        "source_rows": source_rows,
        "generated_task_rows": len(rows),
        "errors": list(generated_a.get("errors", [])) + list(generated_b.get("errors", [])),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "recipe_v3_heldout.json", payload)
    write_csv(OUT_ROOT / "recipe_v3_heldout_rows.csv", rows)
    write_md(
        OUT_ROOT / "recipe_v3_heldout.md",
        [
            "# MMLU Science Recipe v3 Heldout",
            "",
            status_line("BG_MMLU_SCIENCE_RECIPE_V3_HELDOUT_VERDICT", verdict),
            "",
            "## Recipe Summary",
            "",
            *md_table(summary, ["recipe_id", "task_count", "source_count", "selected_parser_positive_oracle_rate", "selected_parser_terminal_best_reward", "reward_diverse_rate", "terminal_oracle_retained"]),
            "",
            "## Source Summary",
            "",
            *md_table(source_rows, ["source_detail", "unique_task_count", "best_recipe", "positive_oracle_rate", "selected_parser_positive_oracle_rate", "terminal_best_reward", "selected_parser_terminal_best_reward", "parse_success"]),
        ],
    )
    print(status_line("BG_MMLU_SCIENCE_RECIPE_V3_HELDOUT_VERDICT", verdict))
    return 0


def heldout_verdict(source_rows: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return "INSUFFICIENT"
    mmlu_rows = [row for row in rows if str(row.get("source_detail")) in PRIMARY_SOURCES]
    mmlu_positive = finite_mean(row.get("selected_parser_positive_oracle") for row in mmlu_rows)
    mmlu_terminal = finite_mean(row.get("selected_parser_terminal_best_reward") for row in mmlu_rows)
    repaired_sources = [
        row for row in source_rows
        if row.get("source_detail") in {"mmlu_high_school_chemistry", "mmlu_anatomy"}
        and safe_float(row.get("selected_parser_positive_oracle_rate"), 0.0) > 0
    ]
    if mmlu_positive >= 0.20 and mmlu_terminal >= 0.15 and len(repaired_sources) >= 1:
        return "MMLU_SCIENCE_HEADLINE_READY"
    if len(repaired_sources) >= 2:
        return "CHEM_ANATOMY_IMPROVED"
    if repaired_sources:
        return "SCIENCE_RECIPE_IMPROVED_BUT_WEAK"
    physics = next((row for row in source_rows if row.get("source_detail") == "mmlu_high_school_physics"), {})
    if safe_float(physics.get("selected_parser_positive_oracle_rate"), 0.0) >= 0.25:
        return "PHYSICS_READY"
    if any(safe_float(row.get("parse_success"), 0.0) < 0.2 for row in source_rows):
        return "PARSER_BLOCKER"
    return "SCIENCE_STILL_BLOCKED"


def source_failures_main() -> int:
    started = time.time()
    rows = generated_task_rows("recipe_heldout") + generated_task_rows("source_heldout")
    if not rows:
        rows = generated_task_rows("recipe_calibration")
    source_rows = []
    failure_cases = []
    for source in sorted({str(row.get("source_detail")) for row in rows}, key=source_rank):
        vals = [row for row in rows if str(row.get("source_detail")) == source]
        summary = metric_summary(vals)
        categories = Counter()
        for row in vals:
            if safe_float(row.get("selected_parser_parse_success"), 0.0) <= 0:
                categories["parser_missed_or_failed"] += 1
            if safe_float(row.get("selected_parser_positive_oracle"), 0.0) <= 0:
                categories["no_positive_branch"] += 1
            elif safe_float(row.get("terminal_forced_top1_reward"), 0.0) <= 0:
                categories["positive_branch_exists_but_terminal_missed"] += 1
            failure_cases.append({**row, "failure_category": most_likely_failure(row)})
        source_rows.append(
            {
                "source_detail": source,
                "task_recipe_count": len(vals),
                "unique_task_count": len({row.get("task_id") for row in vals}),
                **summary,
                "failure_counts": dict(categories),
                "source_status": source_status_from_summary(source, summary),
            }
        )
    verdict = source_failure_verdict(source_rows)
    payload = {
        "BG_MMLU_SCIENCE_SOURCE_FAILURE_VERDICT": verdict,
        "source_rows": source_rows,
        "failure_case_count": len(failure_cases),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "source_failure_analysis.json", payload)
    write_csv(OUT_ROOT / "source_failure_cases.csv", failure_cases)
    write_md(
        OUT_ROOT / "source_failure_analysis.md",
        [
            "# MMLU Science v3 Source Failure Analysis",
            "",
            status_line("BG_MMLU_SCIENCE_SOURCE_FAILURE_VERDICT", verdict),
            "",
            *md_table(source_rows, ["source_detail", "unique_task_count", "selected_parser_positive_oracle_rate", "selected_parser_terminal_best_reward", "parse_success", "source_status"]),
        ],
    )
    print(status_line("BG_MMLU_SCIENCE_SOURCE_FAILURE_VERDICT", verdict))
    return 0


def most_likely_failure(row: dict[str, Any]) -> str:
    if safe_float(row.get("selected_parser_parse_success"), 0.0) <= 0:
        return "parser_missed_correct_or_failed"
    if safe_float(row.get("selected_parser_positive_oracle"), 0.0) <= 0:
        return "no_positive_branch"
    if safe_float(row.get("terminal_forced_top1_reward"), 0.0) <= 0:
        return "positive_branch_exists_but_terminal_missed"
    return "not_failed"


def source_status_from_summary(source: str, summary: dict[str, Any]) -> str:
    if safe_float(summary.get("selected_parser_positive_oracle_rate"), 0.0) > 0:
        if source == "mmlu_high_school_chemistry":
            return "CHEMISTRY_REPAIRED"
        if source == "mmlu_anatomy":
            return "ANATOMY_REPAIRED"
        if source == "mmlu_high_school_physics":
            return "PHYSICS_REPAIRED"
        return "SOURCE_PARTIAL"
    if safe_float(summary.get("parse_success"), 0.0) < 0.2:
        return "PARSER_REMAINS_BLOCKER"
    if source in {"mmlu_high_school_chemistry", "mmlu_anatomy"}:
        return "BRANCH_GENERATION_BLOCKED"
    return "DATA_LIMITED"


def source_failure_verdict(source_rows: Sequence[dict[str, Any]]) -> str:
    statuses = {row.get("source_status") for row in source_rows}
    if "CHEMISTRY_REPAIRED" in statuses:
        return "CHEMISTRY_REPAIRED"
    if "ANATOMY_REPAIRED" in statuses:
        return "ANATOMY_REPAIRED"
    if "PHYSICS_REPAIRED" in statuses:
        return "PHYSICS_REPAIRED"
    if any(row.get("source_detail") in {"mmlu_high_school_chemistry", "mmlu_anatomy"} and row.get("source_status") == "BRANCH_GENERATION_BLOCKED" for row in source_rows):
        return "CHEM_ANATOMY_STILL_BRANCH_BLOCKED"
    if "PARSER_REMAINS_BLOCKER" in statuses:
        return "PARSER_REMAINS_BLOCKER"
    return "DATA_LIMITED"


def l47_ablation_main() -> int:
    started = time.time()
    rows = generated_task_rows("recipe_calibration") + generated_task_rows("recipe_heldout") + generated_task_rows("source_heldout")
    modes = {"baseline_v3_regenerated", "L2_47_emphasis", "L47_heavy", "L24_L47_combo", "L36_L47_combo", "L24_heavy", "L36_heavy"}
    ablation_rows = [{**row, "mode": row.get("recipe_id") if row.get("recipe_id") in modes else "other"} for row in rows]
    summary = []
    for mode in sorted({row["mode"] for row in ablation_rows}):
        vals = [row for row in ablation_rows if row["mode"] == mode]
        summary.append({"mode": mode, **metric_summary(vals)})
    verdict = l47_verdict(summary)
    payload = {
        "BG_MMLU_SCIENCE_L47_V3_ABLATION_VERDICT": verdict,
        "summary_rows": summary,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "l47_v3_ablation.json", payload)
    write_csv(OUT_ROOT / "l47_v3_ablation_rows.csv", ablation_rows)
    write_md(
        OUT_ROOT / "l47_v3_ablation.md",
        [
            "# MMLU Science L47/L2_47 Ablation v3",
            "",
            status_line("BG_MMLU_SCIENCE_L47_V3_ABLATION_VERDICT", verdict),
            "",
            *md_table(summary, ["mode", "task_count", "selected_parser_positive_oracle_rate", "selected_parser_terminal_best_reward", "reward_diverse_rate", "terminal_oracle_retained"]),
        ],
    )
    print(status_line("BG_MMLU_SCIENCE_L47_V3_ABLATION_VERDICT", verdict))
    return 0


def l47_verdict(summary: Sequence[dict[str, Any]]) -> str:
    if not summary:
        return "INSUFFICIENT"
    best = max(summary, key=lambda row: (safe_float(row.get("selected_parser_terminal_best_reward"), -9), safe_float(row.get("selected_parser_positive_oracle_rate"), 0.0)))
    mode = str(best.get("mode"))
    if safe_float(best.get("selected_parser_positive_oracle_rate"), 0.0) <= 0 and safe_float(best.get("selected_parser_terminal_best_reward"), 0.0) <= 0:
        return "L47_NECESSARY_BUT_NOT_SUFFICIENT"
    if mode == "L2_47_emphasis":
        return "L2_47_CONFIRMED_FOR_MMLU"
    if mode == "L47_heavy":
        return "L47_HEAVY_CONFIRMED_FOR_MMLU"
    if mode in {"L24_L47_combo", "L36_L47_combo"}:
        return "SOURCE_SPECIFIC_L47"
    return "L47_SIGNAL_NOT_REPRODUCED"


def budget_breadth_main() -> int:
    started = time.time()
    rows = generated_task_rows("recipe_calibration") + generated_task_rows("recipe_heldout") + generated_task_rows("source_heldout")
    budget_rows = []
    for row in rows:
        rid = str(row.get("recipe_id"))
        if rid in {"baseline_v3_regenerated", "more_children_same_budget", "perturbation_escalation_diagnostic"}:
            mode = rid
        elif rid in {"L2_47_emphasis", "L47_heavy"}:
            mode = "layer_breadth_proxy"
        else:
            mode = "other"
        budget_rows.append({**row, "budget_breadth_mode": mode, "true_budget": 8})
    summary = []
    for mode in sorted({row["budget_breadth_mode"] for row in budget_rows}):
        vals = [row for row in budget_rows if row["budget_breadth_mode"] == mode]
        summary.append({"mode": mode, **metric_summary(vals)})
    verdict = budget_verdict(summary)
    payload = {
        "BG_MMLU_SCIENCE_BUDGET_BREADTH_VERDICT": verdict,
        "summary_rows": summary,
        "true_budget10_12_executed": False,
        "budget_note": "existing guarded engine exposes budget6/8 only; v3 tests breadth before locked budget8 pruning.",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "budget_breadth_v3.json", payload)
    write_csv(OUT_ROOT / "budget_breadth_rows.csv", budget_rows)
    write_md(
        OUT_ROOT / "budget_breadth_v3.md",
        [
            "# MMLU Science Budget/Breadth v3",
            "",
            status_line("BG_MMLU_SCIENCE_BUDGET_BREADTH_VERDICT", verdict),
            "",
            f"- true budget10/12 executed: `{payload['true_budget10_12_executed']}`",
            f"- note: `{payload['budget_note']}`",
            "",
            *md_table(summary, ["mode", "task_count", "selected_parser_positive_oracle_rate", "selected_parser_terminal_best_reward", "reward_diverse_rate", "terminal_oracle_retained"]),
        ],
    )
    print(status_line("BG_MMLU_SCIENCE_BUDGET_BREADTH_VERDICT", verdict))
    return 0


def budget_verdict(summary: Sequence[dict[str, Any]]) -> str:
    if not summary:
        return "INSUFFICIENT"
    baseline = next((row for row in summary if row.get("mode") == "baseline_v3_regenerated"), {})
    breadth = next((row for row in summary if row.get("mode") == "more_children_same_budget"), {})
    if safe_float(breadth.get("selected_parser_terminal_best_reward"), -9) > safe_float(baseline.get("selected_parser_terminal_best_reward"), -9) + 0.05:
        return "MORE_PREPRUNE_HELPS"
    if baseline:
        return "BUDGET8_SUFFICIENT"
    return "DATA_LIMITED"


def soft_hair_main() -> int:
    started = time.time()
    hair_rows = read_csv(HAIRS_ROOT / "convergence_hair_pair_rows.csv")
    task_map = {row.get("task_id"): row for row in generated_task_rows("recipe_heldout") + generated_task_rows("source_heldout") + generated_task_rows("recipe_calibration")}
    rows = []
    for row in hair_rows:
        task_id = row.get("task_id")
        generated = task_map.get(task_id, {})
        source = source_detail(str(task_id), row.get("source_dataset"))
        if source not in PRIMARY_SOURCES and source not in CONTROL_SOURCES:
            continue
        no_good = safe_float(generated.get("selected_parser_positive_oracle"), safe_float(generated.get("positive_oracle"), 0.0)) <= 0
        rows.append({**row, "source_detail": source, "generated_positive_oracle": generated.get("selected_parser_positive_oracle"), "no_good_branch": 1.0 if no_good else 0.0})
    summary = []
    for source in sorted({row.get("source_detail") for row in rows}, key=source_rank):
        vals = [row for row in rows if row.get("source_detail") == source]
        summary.append(
            {
                "source_detail": source,
                "pair_count": len(vals),
                "hidden_rms_normalized": finite_mean(row.get("hidden_rms_normalized") for row in vals),
                "dualanchor_abs_avg_margin": finite_mean(row.get("dualanchor_abs_avg_margin") for row in vals),
                "reward_tied_rate": finite_mean(row.get("reward_tied_eval_only") for row in vals),
                "no_good_branch_rate": finite_mean(row.get("no_good_branch") for row in vals),
            }
        )
    if any(row.get("source_detail") in {"mmlu_high_school_chemistry", "mmlu_anatomy"} and safe_float(row.get("no_good_branch_rate"), 0.0) > 0.5 for row in summary):
        verdict = "CHEM_ANATOMY_NO_GOOD_CONFIRMED"
    elif summary:
        verdict = "NO_GOOD_MONITOR_USEFUL"
    else:
        verdict = "DATA_LIMITED"
    payload = {
        "BG_MMLU_SCIENCE_SOFT_HAIR_NO_GOOD_VERDICT": verdict,
        "summary_rows": summary,
        "hard_merge": False,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "soft_hair_no_good_v3.json", payload)
    write_csv(OUT_ROOT / "soft_hair_no_good_rows.csv", rows)
    write_md(
        OUT_ROOT / "soft_hair_no_good_v3.md",
        [
            "# MMLU Science Soft-Hair No-Good Monitor v3",
            "",
            status_line("BG_MMLU_SCIENCE_SOFT_HAIR_NO_GOOD_VERDICT", verdict),
            "",
            *md_table(summary, ["source_detail", "pair_count", "hidden_rms_normalized", "dualanchor_abs_avg_margin", "reward_tied_rate", "no_good_branch_rate"]),
            "",
            "L30/L42 convergence hairs remain soft diagnostics only.",
        ],
    )
    print(status_line("BG_MMLU_SCIENCE_SOFT_HAIR_NO_GOOD_VERDICT", verdict))
    return 0


def reasoning_guardrail_main() -> int:
    started = time.time()
    suite_ids = {str(row.get("task_id")) for row in suite_rows() if row.get("domain") == "reasoning"}
    task_rows = [row for row in v2.v3_task_rows() if row.get("domain") == "reasoning" and (not suite_ids or str(row.get("task_id")) in suite_ids)]
    terminal_rows = [row for row in v2.v3_terminal_rows() if row.get("domain") == "reasoning" and (not suite_ids or str(row.get("task_id")) in suite_ids)]
    summary = v2.metric_summary(task_rows)
    terminal_summary = v2.grouped_terminal_summary(terminal_rows)
    top5 = next((row for row in terminal_summary if row.get("policy") == "dualanchor_terminal_top5"), {})
    gated = next((row for row in terminal_summary if row.get("policy") == "dualanchor_confidence_gated"), {})
    if safe_float(top5.get("oracle_retained"), 0.0) >= 0.99 and safe_float(gated.get("oracle_retained"), 0.0) >= 0.99:
        verdict = "REASONING_BASELINE_STILL_READY"
    elif not task_rows:
        verdict = "DATA_LIMITED"
    else:
        verdict = "REASONING_REGRESSED"
    payload = {
        "BG_REASONING_GUARDRAIL_V3_VERDICT": verdict,
        "task_summary": summary,
        "terminal_summary": terminal_summary,
        "locked_policy": "confidence-gated top1 else top5/full survivor handoff",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "reasoning_guardrail.json", payload)
    write_md(
        OUT_ROOT / "reasoning_guardrail.md",
        [
            "# Reasoning Guardrail v3",
            "",
            status_line("BG_REASONING_GUARDRAIL_V3_VERDICT", verdict),
            "",
            "## Task Summary",
            "",
            *[f"- {k}: `{fmt(v)}`" for k, v in summary.items()],
            "",
            "## Terminal Policies",
            "",
            *md_table(terminal_summary, ["policy", "task_count", "oracle_retained", "best_selected_reward", "first_selected_oracle", "defer_rate"]),
        ],
    )
    print(status_line("BG_REASONING_GUARDRAIL_V3_VERDICT", verdict))
    return 0


def readiness_main() -> int:
    started = time.time()
    parser = read_json(OUT_ROOT / "parser_adversarial_validation.json", {}) or {}
    heldout = read_json(OUT_ROOT / "recipe_v3_heldout.json", {}) or {}
    failures = read_json(OUT_ROOT / "source_failure_analysis.json", {}) or {}
    guardrail = read_json(OUT_ROOT / "reasoning_guardrail.json", {}) or {}
    verdict = science_readiness_verdict(parser, heldout, failures, guardrail)
    payload = {
        "BG_MMLU_SCIENCE_V3_READINESS_VERDICT": verdict,
        "input_verdicts": {
            "parser": parser.get("BG_MMLU_SCIENCE_PARSER_ADVERSARIAL_VERDICT"),
            "heldout": heldout.get("BG_MMLU_SCIENCE_RECIPE_V3_HELDOUT_VERDICT"),
            "source_failure": failures.get("BG_MMLU_SCIENCE_SOURCE_FAILURE_VERDICT"),
            "reasoning_guardrail": guardrail.get("BG_REASONING_GUARDRAIL_V3_VERDICT"),
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "science_v3_readiness.json", payload)
    write_md(
        OUT_ROOT / "science_v3_readiness.md",
        [
            "# MMLU Science v3 Readiness",
            "",
            status_line("BG_MMLU_SCIENCE_V3_READINESS_VERDICT", verdict),
            "",
            *[f"- {k}: `{v}`" for k, v in payload["input_verdicts"].items()],
        ],
    )
    print(status_line("BG_MMLU_SCIENCE_V3_READINESS_VERDICT", verdict))
    return 0


def science_readiness_verdict(parser: dict[str, Any], heldout: dict[str, Any], failures: dict[str, Any], guardrail: dict[str, Any]) -> str:
    if guardrail.get("BG_REASONING_GUARDRAIL_V3_VERDICT") == "REASONING_REGRESSED":
        return "NOT_READY"
    parser_verdict = parser.get("BG_MMLU_SCIENCE_PARSER_ADVERSARIAL_VERDICT")
    heldout_verdict_value = heldout.get("BG_MMLU_SCIENCE_RECIPE_V3_HELDOUT_VERDICT")
    failure_verdict = failures.get("BG_MMLU_SCIENCE_SOURCE_FAILURE_VERDICT")
    if parser_verdict in {"FALSE_POSITIVE_RISK_HIGH", "BLOCKED"}:
        return "SCIENCE_PARSER_BLOCKED"
    if heldout_verdict_value == "MMLU_SCIENCE_HEADLINE_READY":
        return "SCIENCE_HEADLINE_READY"
    if heldout_verdict_value in {"CHEM_ANATOMY_IMPROVED", "PHYSICS_READY", "SCIENCE_RECIPE_IMPROVED_BUT_WEAK"}:
        return "SCIENCE_PARTIAL_HEADLINE_READY"
    if failure_verdict == "CHEM_ANATOMY_STILL_BRANCH_BLOCKED":
        return "SCIENCE_BRANCH_GENERATION_BLOCKED"
    if failure_verdict == "PARSER_REMAINS_BLOCKER":
        return "SCIENCE_PARSER_BLOCKED"
    return "SCIENCE_DIAGNOSTIC_ONLY"


def pre_steering_domain_decision_main() -> int:
    started = time.time()
    readiness = read_json(OUT_ROOT / "science_v3_readiness.json", {}) or {}
    guardrail = read_json(OUT_ROOT / "reasoning_guardrail.json", {}) or {}
    science = readiness.get("BG_MMLU_SCIENCE_V3_READINESS_VERDICT")
    reasoning = guardrail.get("BG_REASONING_GUARDRAIL_V3_VERDICT")
    if reasoning != "REASONING_BASELINE_STILL_READY":
        verdict = "DO_NOT_START_STEERING"
    elif science == "SCIENCE_HEADLINE_READY":
        verdict = "READY_FOR_STEERING_REASONING_AND_SCIENCE"
    elif science == "SCIENCE_PARTIAL_HEADLINE_READY":
        verdict = "READY_FOR_STEERING_REASONING_PLUS_PARTIAL_SCIENCE"
    elif science in {"SCIENCE_DIAGNOSTIC_ONLY", "SCIENCE_BRANCH_GENERATION_BLOCKED", "SCIENCE_PARSER_BLOCKED"}:
        verdict = "READY_FOR_STEERING_WITH_SCIENCE_DIAGNOSTIC"
    else:
        verdict = "INSUFFICIENT"
    payload = {
        "BG_PRE_STEERING_DOMAIN_DECISION_V3_VERDICT": verdict,
        "science_readiness": science,
        "reasoning_guardrail": reasoning,
        "locked_baseline": {
            "schedule": LOCKED_SCHEDULE,
            "selector": f"{ANCHOR_A} + {ANCHOR_B}",
            "threshold": "mean_floor_very_loose",
            "budget": 8,
            "l47": "active in nonterminal loops",
            "terminal": "confidence-gated top1; otherwise top5/full survivor-set handoff",
            "convergence_hairs": "soft-only diagnostics",
            "parser": selected_parser_id(),
            "steering": "not run in this prompt",
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "pre_steering_domain_decision_v3.json", payload)
    write_md(
        OUT_ROOT / "pre_steering_domain_decision_v3.md",
        [
            "# Pre-Steering Domain Decision v3",
            "",
            status_line("BG_PRE_STEERING_DOMAIN_DECISION_V3_VERDICT", verdict),
            "",
            f"- science_readiness: `{science}`",
            f"- reasoning_guardrail: `{reasoning}`",
            "",
            "Steering was not run in this prompt.",
        ],
    )
    print(status_line("BG_PRE_STEERING_DOMAIN_DECISION_V3_VERDICT", verdict))
    return 0


def status_from(readiness: str, source_failure: str, parser: str) -> str:
    if readiness == "SCIENCE_HEADLINE_READY":
        return "SCIENCE_REPAIRED"
    if readiness == "SCIENCE_PARTIAL_HEADLINE_READY":
        if source_failure == "PHYSICS_REPAIRED":
            return "PHYSICS_READY_CHEM_ANATOMY_BLOCKED"
        return "SCIENCE_PARTIALLY_REPAIRED"
    if parser in {"PARSER_V3_READY", "SOURCE_SPECIFIC_PARSER_READY"} and source_failure == "CHEM_ANATOMY_STILL_BRANCH_BLOCKED":
        return "PARSER_REPAIRED_BRANCH_BLOCKED"
    if readiness == "SCIENCE_PARSER_BLOCKED":
        return "PARSER_BLOCKED"
    if source_failure == "CHEM_ANATOMY_STILL_BRANCH_BLOCKED":
        return "CHEM_ANATOMY_STILL_BLOCKED"
    if readiness == "SCIENCE_BRANCH_GENERATION_BLOCKED":
        return "BRANCH_GENERATION_BLOCKED"
    if readiness == "SCIENCE_DIAGNOSTIC_ONLY":
        return "SCIENCE_DIAGNOSTIC_ONLY"
    return "INSUFFICIENT"


def synthesis_main() -> int:
    started = time.time()
    inputs = {
        "inventory": read_json(OUT_ROOT / "inventory.json", {}) or {},
        "task_suite": read_json(OUT_ROOT / "task_suite.json", {}) or {},
        "prompt": read_json(OUT_ROOT / "prompt_format_audit.json", {}) or {},
        "parser_build": read_json(OUT_ROOT / "parser_v3_candidates.json", {}) or {},
        "parser": read_json(OUT_ROOT / "parser_adversarial_validation.json", {}) or {},
        "recipe_plan": read_json(OUT_ROOT / "science_recipe_v3_plan.json", {}) or {},
        "calibration": read_json(OUT_ROOT / "recipe_v3_calibration.json", {}) or {},
        "heldout": read_json(OUT_ROOT / "recipe_v3_heldout.json", {}) or {},
        "source_failure": read_json(OUT_ROOT / "source_failure_analysis.json", {}) or {},
        "l47": read_json(OUT_ROOT / "l47_v3_ablation.json", {}) or {},
        "budget": read_json(OUT_ROOT / "budget_breadth_v3.json", {}) or {},
        "soft_hair": read_json(OUT_ROOT / "soft_hair_no_good_v3.json", {}) or {},
        "reasoning": read_json(OUT_ROOT / "reasoning_guardrail.json", {}) or {},
        "readiness": read_json(OUT_ROOT / "science_v3_readiness.json", {}) or {},
        "domain_decision": read_json(OUT_ROOT / "pre_steering_domain_decision_v3.json", {}) or {},
    }
    verdicts = {
        "BG_MMLU_SCIENCE_V3_INVENTORY_VERDICT": inputs["inventory"].get("BG_MMLU_SCIENCE_V3_INVENTORY_VERDICT", "INSUFFICIENT"),
        "BG_MMLU_SCIENCE_V3_TASK_SUITE_VERDICT": inputs["task_suite"].get("BG_MMLU_SCIENCE_V3_TASK_SUITE_VERDICT", "INSUFFICIENT"),
        "BG_MMLU_SCIENCE_PROMPT_FORMAT_VERDICT": inputs["prompt"].get("BG_MMLU_SCIENCE_PROMPT_FORMAT_VERDICT", "INSUFFICIENT"),
        "BG_MMLU_SCIENCE_PARSER_V3_BUILD_VERDICT": inputs["parser_build"].get("BG_MMLU_SCIENCE_PARSER_V3_BUILD_VERDICT", "INSUFFICIENT"),
        "BG_MMLU_SCIENCE_PARSER_ADVERSARIAL_VERDICT": inputs["parser"].get("BG_MMLU_SCIENCE_PARSER_ADVERSARIAL_VERDICT", "INSUFFICIENT"),
        "BG_MMLU_SCIENCE_RECIPE_V3_PLAN_VERDICT": inputs["recipe_plan"].get("BG_MMLU_SCIENCE_RECIPE_V3_PLAN_VERDICT", "INSUFFICIENT"),
        "BG_MMLU_SCIENCE_RECIPE_V3_CALIBRATION_VERDICT": inputs["calibration"].get("BG_MMLU_SCIENCE_RECIPE_V3_CALIBRATION_VERDICT", "INSUFFICIENT"),
        "BG_MMLU_SCIENCE_RECIPE_V3_HELDOUT_VERDICT": inputs["heldout"].get("BG_MMLU_SCIENCE_RECIPE_V3_HELDOUT_VERDICT", "INSUFFICIENT"),
        "BG_MMLU_SCIENCE_SOURCE_FAILURE_VERDICT": inputs["source_failure"].get("BG_MMLU_SCIENCE_SOURCE_FAILURE_VERDICT", "INSUFFICIENT"),
        "BG_MMLU_SCIENCE_L47_V3_ABLATION_VERDICT": inputs["l47"].get("BG_MMLU_SCIENCE_L47_V3_ABLATION_VERDICT", "INSUFFICIENT"),
        "BG_MMLU_SCIENCE_BUDGET_BREADTH_VERDICT": inputs["budget"].get("BG_MMLU_SCIENCE_BUDGET_BREADTH_VERDICT", "INSUFFICIENT"),
        "BG_MMLU_SCIENCE_SOFT_HAIR_NO_GOOD_VERDICT": inputs["soft_hair"].get("BG_MMLU_SCIENCE_SOFT_HAIR_NO_GOOD_VERDICT", "INSUFFICIENT"),
        "BG_REASONING_GUARDRAIL_V3_VERDICT": inputs["reasoning"].get("BG_REASONING_GUARDRAIL_V3_VERDICT", "INSUFFICIENT"),
        "BG_MMLU_SCIENCE_V3_READINESS_VERDICT": inputs["readiness"].get("BG_MMLU_SCIENCE_V3_READINESS_VERDICT", "INSUFFICIENT"),
        "BG_PRE_STEERING_DOMAIN_DECISION_V3_VERDICT": inputs["domain_decision"].get("BG_PRE_STEERING_DOMAIN_DECISION_V3_VERDICT", "INSUFFICIENT"),
    }
    overall = status_from(verdicts["BG_MMLU_SCIENCE_V3_READINESS_VERDICT"], verdicts["BG_MMLU_SCIENCE_SOURCE_FAILURE_VERDICT"], verdicts["BG_MMLU_SCIENCE_PARSER_ADVERSARIAL_VERDICT"])
    files_created = sorted(str(path.relative_to(OUT_ROOT)) for path in OUT_ROOT.glob("*") if path.is_file())
    payload = {
        **verdicts,
        "MMLU_SCIENCE_BRANCH_PARSER_REPAIR_V3_STATUS": overall,
        "files_created": files_created,
        "commands_run": V3_COMMANDS,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    lines = [
        "# MMLU Science Branch Generation + Parser Repair v3",
        "",
        *[status_line(k, v) for k, v in verdicts.items()],
        status_line("MMLU_SCIENCE_BRANCH_PARSER_REPAIR_V3_STATUS", overall),
        "",
        "## Motivation",
        "",
        "This run focuses on MMLU science branch generation and parser repair before any steering/model-training work begins.",
        "",
        "## Previous v2 Result",
        "",
        "`DUALANCHOR_SCIENCE_REASONING_REPAIR_V2_STATUS = REASONING_READY_SCIENCE_DIAGNOSTIC`.",
        "",
        "## Heldout Evaluation",
        "",
        *md_table(inputs["heldout"].get("source_rows", []), ["source_detail", "unique_task_count", "best_recipe", "selected_parser_positive_oracle_rate", "selected_parser_terminal_best_reward", "parse_success"]),
        "",
        "## Science Readiness",
        "",
        f"- readiness: `{verdicts['BG_MMLU_SCIENCE_V3_READINESS_VERDICT']}`",
        f"- pre-steering decision: `{verdicts['BG_PRE_STEERING_DOMAIN_DECISION_V3_VERDICT']}`",
        "",
        "## Locked Baseline If Ready",
        "",
        f"- schedule: `{LOCKED_SCHEDULE}`",
        f"- selector: `{ANCHOR_A} + {ANCHOR_B}`",
        "- threshold: `mean_floor_very_loose`",
        "- budget: `8`",
        "- L47: `active in nonterminal loops`",
        "- terminal: `confidence-gated top1; otherwise top5/full survivor-set handoff`",
        "- convergence hairs: `soft-only diagnostics`",
        "- steering: `not run in this prompt`",
        "",
        "## Files Created",
        "",
        *[f"- `{name}`" for name in files_created],
        "",
        "## Commands Run",
        "",
        *[f"- `{cmd}`" for cmd in V3_COMMANDS],
        "",
        "## Blockers",
        "",
        "- Local science task pool is smaller than requested; no MMLU biology or science-domain OpenBookQA was available in the audit plan.",
    ]
    write_json(OUT_ROOT / "summary.json", payload)
    write_json(OUT_ROOT / "analysis.json", {**payload, "inputs": inputs})
    write_md(OUT_ROOT / "summary.md", lines)
    write_md(OUT_ROOT / "analysis.md", lines)
    print(status_line("MMLU_SCIENCE_BRANCH_PARSER_REPAIR_V3_STATUS", overall))
    return 0


V3_COMMANDS = [
    "bg_mmlu_science_v3_inventory.py",
    "build_bg_mmlu_science_v3_task_suite.py",
    "audit_bg_mmlu_science_prompt_format_v3.py",
    "build_bg_mmlu_science_parser_v3.py",
    "validate_bg_mmlu_science_parser_adversarial_v3.py",
    "build_bg_mmlu_science_recipe_v3_plan.py",
    "run_bg_mmlu_science_recipe_v3_calibration.py",
    "evaluate_bg_mmlu_science_recipe_v3_heldout.py",
    "analyze_bg_mmlu_science_v3_source_failures.py",
    "run_bg_mmlu_science_l47_v3_ablation.py",
    "analyze_bg_mmlu_science_budget_breadth_v3.py",
    "analyze_bg_mmlu_science_soft_hair_no_good_v3.py",
    "check_bg_reasoning_guardrail_v3.py",
    "analyze_bg_mmlu_science_v3_readiness.py",
    "analyze_bg_pre_steering_domain_decision_v3.py",
    "analyze_bg_mmlu_science_branch_parser_repair_v3.py",
]

