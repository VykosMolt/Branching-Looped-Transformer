"""Aggregate Branch Generator v1 experiment outputs and update docs."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from bg_branch_generator_v1_common import (
    ANALYSIS_JSON,
    ANALYSIS_MD,
    AUDIT_PLAN_JSON,
    BASIS_BANK_JSON,
    BEST_SCHEDULE_JSON,
    BLACKBOX_JSON,
    BGV1_COMMAND_SCRIPTS,
    BGV1_ROOT,
    DOC_MD,
    GEN_REPORT_JSON,
    GEOMETRY_JSON,
    OLD_CONTEXT_JSON,
    PROPOSER_JSON,
    RICH_SCHEMA_JSON,
    SELECTOR_DATASET_JSON,
    SELECTOR_EVAL_JSON,
    SELECTOR_TRAINING_JSON,
    SUMMARY_JSON,
    SUMMARY_MD,
    TRUE_FORK_JSON,
    ensure_bgv1_root,
    load_json,
    rel,
    write_json,
    write_md,
)


DIVERSITY_JSON = BGV1_ROOT / "diversity_analysis.json"


def verdict(payload: dict[str, Any], key: str, default: str = "INSUFFICIENT") -> str:
    return str(payload.get(key) or payload.get("verdict") or default)


def load_all() -> dict[str, dict[str, Any]]:
    return {
        "audit": load_json(AUDIT_PLAN_JSON, {}) or {},
        "true_fork": load_json(TRUE_FORK_JSON, {}) or {},
        "schema": load_json(RICH_SCHEMA_JSON, {}) or {},
        "basis": load_json(BASIS_BANK_JSON, {}) or {},
        "proposer": load_json(PROPOSER_JSON, {}) or {},
        "blackbox": load_json(BLACKBOX_JSON, {}) or {},
        "generation": load_json(GEN_REPORT_JSON, {}) or {},
        "diversity": load_json(DIVERSITY_JSON, {}) or {},
        "dataset": load_json(SELECTOR_DATASET_JSON, {}) or {},
        "training": load_json(SELECTOR_TRAINING_JSON, {}) or {},
        "eval": load_json(SELECTOR_EVAL_JSON, {}) or {},
        "old_context": load_json(OLD_CONTEXT_JSON, {}) or {},
        "geometry": load_json(GEOMETRY_JSON, {}) or {},
    }


def status_for(data: dict[str, dict[str, Any]]) -> str:
    generation = verdict(data["generation"], "BG_BRANCH_GENERATOR_V1_GENERATION_VERDICT")
    diversity = verdict(data["diversity"], "BG_BRANCH_GENERATOR_V1_DIVERSITY_VERDICT")
    dataset = verdict(data["dataset"], "BG_BRANCH_GENERATOR_V1_SELECTOR_DATASET_VERDICT")
    evaluation = verdict(data["eval"], "BG_BRANCH_GENERATOR_V1_SELECTOR_EVAL_VERDICT")
    true_fork = verdict(data["true_fork"], "BG_TRUE_FORK_CARRY_PROBE_V1_VERDICT")
    if generation == "QUOTAS_MET" and evaluation in {"SELECTOR_READY", "OLD_TAPS_BEST", "ENSEMBLE_BEST"}:
        return "READY_FOR_SELECTION_ONLY_PROTOTYPE"
    if diversity in {"STRONG_IMPROVEMENT", "WEAK_IMPROVEMENT"} and dataset in {"SMALL_BUT_USABLE", "READY", "HELDOUT_READY_TRAIN_WEAK"}:
        return "WEAK_BUT_USABLE"
    if generation in {"LOW_DIVERSITY", "DIVERSITY_IMPROVED", "PARTIAL"} and true_fork in {"HOOK_FALLBACK_ONLY", "STATE_HANDLING_BLOCKED"}:
        return "NEEDS_STRONGER_GENERATOR"
    if true_fork in {"STATE_HANDLING_BLOCKED", "HOOK_FALLBACK_ONLY"} and diversity in {"NO_IMPROVEMENT", "NEEDS_TRUE_FORK_CARRY"}:
        return "NEEDS_TRUE_FORK_CARRY"
    if dataset == "STILL_DATA_LIMITED":
        return "NEEDS_STRONGER_GENERATOR"
    if evaluation == "NO_SELECTOR_SIGNAL":
        return "NOT_READY"
    return "INSUFFICIENT"


def best_selector(data: dict[str, dict[str, Any]]) -> str:
    row = data["eval"]
    value = row.get("HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE_AFTER_GENERATOR_V1_EVAL") or row.get("best_selector")
    allowed = {
        "old_frozen_bg",
        "v1_hidden_origin_tap",
        "v2_hidden_origin_tap",
        "v3_hidden_origin_tap",
        "v4_hidden_origin_tap",
        "salvage_retrained_head",
        "generator_v1_selector",
        "ensemble",
        "random",
    }
    return str(value) if value in allowed else "insufficient"


def recommended_next(status: str) -> str:
    if status == "READY_FOR_SELECTION_ONLY_PROTOTYPE":
        return "Design a small hidden-origin selection-only Phase 2 prototype with Branch Generator v1, the best available selector, and top-k selection. Do not claim action steering."
    if status == "WEAK_BUT_USABLE":
        return "Either run a small selection-only prototype with caveat or run targeted generator v1.1 if one recipe clearly remains."
    if status == "NEEDS_TRUE_FORK_CARRY":
        return "Prioritize custom Ouro branch-forward/cache implementation before another tap expansion."
    if status == "NEEDS_RICHER_OUTCOME_SIGNAL":
        return "Build auxiliary outcome/margin probes while keeping final deterministic reward as readiness label."
    if status == "NEEDS_STRONGER_GENERATOR":
        return "Stop tap-expansion loops and design a stronger branch generator: deeper low-rank generator, true fork/carry, or branch-maintenance/write module."
    return "Do not proceed to Phase 2 steering."


def top_lines(data: dict[str, dict[str, Any]], status: str, selector: str) -> list[str]:
    return [
        f"BG_BRANCH_GENERATOR_V1_AUDIT_PLAN_VERDICT = {verdict(data['audit'], 'BG_BRANCH_GENERATOR_V1_AUDIT_PLAN_VERDICT')}",
        f"BG_TRUE_FORK_CARRY_PROBE_V1_VERDICT = {verdict(data['true_fork'], 'BG_TRUE_FORK_CARRY_PROBE_V1_VERDICT')}",
        f"BG_RICH_OUTCOME_SCHEMA_V1_VERDICT = {verdict(data['schema'], 'BG_RICH_OUTCOME_SCHEMA_V1_VERDICT')}",
        f"BG_BRANCH_GENERATOR_BASIS_BANK_V1_VERDICT = {verdict(data['basis'], 'BG_BRANCH_GENERATOR_BASIS_BANK_V1_VERDICT')}",
        f"BG_BRANCH_GENERATOR_PROPOSER_TRAINING_V1_VERDICT = {verdict(data['proposer'], 'BG_BRANCH_GENERATOR_PROPOSER_TRAINING_V1_VERDICT')}",
        f"BG_BRANCH_GENERATOR_BLACKBOX_SEARCH_V1_VERDICT = {verdict(data['blackbox'], 'BG_BRANCH_GENERATOR_BLACKBOX_SEARCH_V1_VERDICT')}",
        f"BG_BRANCH_GENERATOR_V1_GENERATION_VERDICT = {verdict(data['generation'], 'BG_BRANCH_GENERATOR_V1_GENERATION_VERDICT')}",
        f"BG_BRANCH_GENERATOR_V1_DIVERSITY_VERDICT = {verdict(data['diversity'], 'BG_BRANCH_GENERATOR_V1_DIVERSITY_VERDICT')}",
        f"BG_BRANCH_GENERATOR_V1_BEST_METHOD = {data['diversity'].get('BG_BRANCH_GENERATOR_V1_BEST_METHOD', 'insufficient')}",
        f"BG_BRANCH_GENERATOR_V1_SELECTOR_DATASET_VERDICT = {verdict(data['dataset'], 'BG_BRANCH_GENERATOR_V1_SELECTOR_DATASET_VERDICT')}",
        f"BG_BRANCH_GENERATOR_V1_SELECTOR_TRAINING_VERDICT = {verdict(data['training'], 'BG_BRANCH_GENERATOR_V1_SELECTOR_TRAINING_VERDICT')}",
        f"BG_BRANCH_GENERATOR_V1_SELECTOR_EVAL_VERDICT = {verdict(data['eval'], 'BG_BRANCH_GENERATOR_V1_SELECTOR_EVAL_VERDICT')}",
        f"BG_BRANCH_GENERATOR_V1_OLD_CONTEXT_REPLAY_VERDICT = {verdict(data['old_context'], 'BG_BRANCH_GENERATOR_V1_OLD_CONTEXT_REPLAY_VERDICT')}",
        f"BG_BRANCH_GENERATOR_V1_GEOMETRY_VERDICT = {verdict(data['geometry'], 'BG_BRANCH_GENERATOR_V1_GEOMETRY_VERDICT')}",
        f"HIDDEN_ORIGIN_BRANCH_GENERATOR_STATUS_V1 = {status}",
        f"HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE_AFTER_GENERATOR_V1 = {selector}",
    ]


def docs_section(data: dict[str, dict[str, Any]], status: str, selector: str) -> str:
    gen = data["generation"].get("quota_progress_by_split", {})
    diversity = data["diversity"].get("questions", {})
    return "\n".join(
        [
            "## Hidden-origin Branch Generator v1 (2026-05-18)",
            "",
            "Branch Generator v1 was run because v4 confirmed selector geometry but remained heldout-diversity limited. The v1 run tested early L24/L1-style hook perturbations, high-yield non-random directions, a lightweight recipe/CEM schedule, true fork/carry feasibility, and richer outcome diagnostics without training Ouro or changing production routing.",
            "",
            *top_lines(data, status, selector),
            "",
            f"- quota_progress_by_split: `{gen}`",
            f"- diversity_questions: `{diversity}`",
            f"- recommended_next: `{recommended_next(status)}`",
            "",
            "Selector readiness, if claimed, uses only primary-safe deterministic alpha <= 0.01 heldout rows. Diagnostic alpha 0.02, sampled labels, L47 branches, old-context replay, and auxiliary diagnostics are not readiness support.",
            "",
        ]
    )


def append_docs(section: str) -> None:
    targets = [
        Path("docs/evaluator/current_state.md"),
        Path("docs/evaluator/domain_transfer_ledger.md"),
        Path("docs/evaluator/bg_hidden_origin_quota_v4.md"),
        Path("docs/evaluator/bg_hidden_origin_split_salvage.md"),
        Path("docs/evaluator/bg_hidden_origin_diversity_v3.md"),
        Path("docs/evaluator/bg_hidden_origin_diversity_v2.md"),
        Path("docs/evaluator/bg_hidden_origin_taps.md"),
        Path("docs/evaluator/bg_hidden_state_branch_generation.md"),
        Path("docs/evaluator/bg_steering_consolidation_2026-05-18.md"),
        Path("docs/evaluator/post_v10_synthesis_2026-05-18_v8.1_routing_locked.md"),
    ]
    for path in targets:
        if not path.exists():
            continue
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n" + section + "\n")


def main() -> int:
    started = time.time()
    ensure_bgv1_root()
    data = load_all()
    status = status_for(data)
    selector = best_selector(data)
    rec = recommended_next(status)
    payload = {
        "top_lines": top_lines(data, status, selector),
        "stage_payloads": data,
        "HIDDEN_ORIGIN_BRANCH_GENERATOR_STATUS_V1": status,
        "HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE_AFTER_GENERATOR_V1": selector,
        "recommended_next": rec,
        "files_created": {name: rel(path) for name, path in {
            "audit_plan": AUDIT_PLAN_JSON,
            "true_fork_carry_probe": TRUE_FORK_JSON,
            "rich_outcome_schema": RICH_SCHEMA_JSON,
            "basis_bank": BASIS_BANK_JSON,
            "proposer_training": PROPOSER_JSON,
            "blackbox_search": BLACKBOX_JSON,
            "best_schedule": BEST_SCHEDULE_JSON,
            "generation_report": GEN_REPORT_JSON,
            "diversity_analysis": DIVERSITY_JSON,
            "selector_dataset": SELECTOR_DATASET_JSON,
            "selector_training": SELECTOR_TRAINING_JSON,
            "selector_eval": SELECTOR_EVAL_JSON,
            "old_context_replay": OLD_CONTEXT_JSON,
            "geometry": GEOMETRY_JSON,
            "summary": SUMMARY_JSON,
            "analysis": ANALYSIS_JSON,
            "docs": DOC_MD,
        }.items()},
        "commands_run": [f"venv/bin/python -u utilities/tests/manual/{script}" for script in BGV1_COMMAND_SCRIPTS],
        "blockers": [value.get("blocker") for value in data.values() if isinstance(value, dict) and value.get("blocker")],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(SUMMARY_JSON, payload)
    write_json(ANALYSIS_JSON, payload)
    section = docs_section(data, status, selector)
    summary_lines = [
        "# Branch Generator V1 Summary",
        "",
        *top_lines(data, status, selector),
        "",
        f"Recommended next: {rec}",
        "",
        "## Motivation and V4 Result",
        "",
        "V4 confirmed old readout geometry but did not meet heldout branch diversity. Branch Generator v1 therefore focused on creating stronger same-prefix hidden-origin behavioral diversity rather than more tap-only expansion.",
        "",
        "## Stage Notes",
        "",
        section,
        "## Files Created",
        "",
    ]
    summary_lines.extend(f"- {name}: `{path}`" for name, path in payload["files_created"].items())
    summary_lines.extend(["", "## Commands Run", ""])
    summary_lines.extend(f"- `{cmd}`" for cmd in payload["commands_run"])
    if payload["blockers"]:
        summary_lines.extend(["", "## Blockers", ""])
        summary_lines.extend(f"- `{item}`" for item in payload["blockers"])
    write_md(SUMMARY_MD, summary_lines)
    write_md(ANALYSIS_MD, summary_lines)
    write_md(
        DOC_MD,
        [
            "# Hidden-Origin Branch Generator V1",
            "",
            "This document records the hidden-origin Branch Generator v1 experiment.",
            "",
            section,
            "## Phase 2 Implication",
            "",
            rec,
            "",
            "No Ouro weights, tokenizers, checkpoints, production tap registries, wrapper/local-agent code, ARC environment action loops, or production routing were modified.",
        ],
    )
    append_docs(section)
    print(f"HIDDEN_ORIGIN_BRANCH_GENERATOR_STATUS_V1 = {status}", flush=True)
    print(f"HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE_AFTER_GENERATOR_V1 = {selector}", flush=True)
    print(f"Wrote {rel(SUMMARY_MD)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
