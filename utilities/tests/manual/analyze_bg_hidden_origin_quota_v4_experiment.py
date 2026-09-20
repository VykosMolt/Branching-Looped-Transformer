"""Synthesize hidden-origin quota v4 experiment and update docs."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from bg_hidden_origin_quota_v4_common import (
    CONTROLLER_JSON,
    DATASET_V4_JSON,
    DIRECTION_BANK_V4_JSON,
    GEN_REPORT_JSON,
    PROJECT_ROOT,
    QUOTA_PLAN_JSON,
    V4_ROOT,
    append_doc_section,
    commands_run,
    ensure_v4_root,
    load_json,
    md_table,
    rel,
    write_json,
    write_md,
)


OUT_SUMMARY_MD = V4_ROOT / "summary.md"
OUT_SUMMARY_JSON = V4_ROOT / "summary.json"
OUT_ANALYSIS_MD = V4_ROOT / "analysis.md"
OUT_ANALYSIS_JSON = V4_ROOT / "analysis.json"
DOC_MD = PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_quota_v4.md"
SECTION_TITLE = "Hidden-origin branch quota v4 and old-context replay (2026-05-18)"


def verdict(payload: dict[str, Any], key: str, default: str = "INSUFFICIENT") -> str:
    return str(payload.get(key) or payload.get("verdict") or default)


def best_selector(eval_payload: dict[str, Any]) -> str:
    return str(eval_payload.get("HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE_V4_EVAL") or "insufficient")


def phase_status(eval_v: str, generation_v: str, dataset_v: str, best: str) -> str:
    if eval_v in {"SELECTOR_READY", "OLD_TAPS_BEST", "ENSEMBLE_BEST"}:
        return "READY"
    if eval_v == "WEAK_SELECTOR":
        return "WEAK"
    if generation_v in {"PARTIAL", "LOW_DIVERSITY", "HELDOUT_QUOTA_MET_ONLY", "TRAIN_VAL_QUOTA_MET_ONLY"} or dataset_v == "STILL_DATA_LIMITED":
        return "STILL_DATA_LIMITED"
    if generation_v == "LOW_DIVERSITY":
        return "NEEDS_BETTER_BRANCH_GENERATOR"
    if eval_v == "NO_SELECTOR_SIGNAL":
        return "NOT_READY"
    return "INSUFFICIENT"


def recommendation(phase: str) -> str:
    if phase == "READY":
        return "Design a small hidden-origin selection-only Phase 2 prototype with hook-hidden-origin branch generation, best available selector, and top-k selection. Do not claim steering."
    if phase == "WEAK":
        return "Run a small selection-only prototype with caveats only if the heldout support is adequate; otherwise do one more targeted quota generation pass."
    if phase == "STILL_DATA_LIMITED":
        return "Inspect quota failures and continue only if a clear recipe remains under the primary-safe constraints."
    if phase == "NOT_READY":
        return "Do not proceed to Phase 2 steering; improve evaluator/features."
    if phase == "NEEDS_BETTER_BRANCH_GENERATOR":
        return "Stop tap expansion and focus on stronger branch generation, true fork-carry, or a different perturbation mechanism."
    return "Resolve blockers before further selector claims."


def main() -> int:
    started = time.time()
    ensure_v4_root()
    plan = load_json(QUOTA_PLAN_JSON, {}) or {}
    bank = load_json(DIRECTION_BANK_V4_JSON, {}) or {}
    controller = load_json(CONTROLLER_JSON, {}) or {}
    generation = load_json(GEN_REPORT_JSON, {}) or {}
    drivers = load_json(V4_ROOT / "quota_diversity_drivers.json", {}) or {}
    dataset = load_json(DATASET_V4_JSON, {}) or {}
    training = load_json(V4_ROOT / "training_log_v4.json", {}) or {}
    eval_payload = load_json(V4_ROOT / "heldout_eval_v4.json", {}) or {}
    old_context = load_json(V4_ROOT / "old_context_replay_v4.json", {}) or {}
    geometry = load_json(V4_ROOT / "geometry_analysis_v4.json", {}) or {}

    plan_v = verdict(plan, "BG_HIDDEN_ORIGIN_QUOTA_PLAN_V4_VERDICT")
    bank_v = verdict(bank, "BG_HIDDEN_ORIGIN_DIRECTION_BANK_V4_VERDICT")
    controller_v = verdict(controller, "BG_HS_INSPIRED_QUOTA_CONTROLLER_V4_VERDICT")
    generation_v = verdict(generation, "BG_HIDDEN_ORIGIN_QUOTA_GENERATION_V4_VERDICT")
    drivers_v = verdict(drivers, "BG_HIDDEN_ORIGIN_QUOTA_DIVERSITY_DRIVER_V4_VERDICT")
    dataset_v = verdict(dataset, "BG_HIDDEN_ORIGIN_QUOTA_DATASET_V4_VERDICT")
    training_v = verdict(training, "BG_HIDDEN_ORIGIN_TAP_TRAINING_V4_VERDICT")
    eval_v = verdict(eval_payload, "BG_HIDDEN_ORIGIN_SELECTOR_EVAL_V4_VERDICT")
    old_context_v = verdict(old_context, "BG_HIDDEN_ORIGIN_OLD_CONTEXT_REPLAY_V4_VERDICT")
    geometry_v = verdict(geometry, "BG_HIDDEN_ORIGIN_TAP_GEOMETRY_V4_VERDICT")
    best = best_selector(eval_payload)
    phase = phase_status(eval_v, generation_v, dataset_v, best)
    top_lines = {
        "BG_HIDDEN_ORIGIN_QUOTA_PLAN_V4_VERDICT": plan_v,
        "BG_HIDDEN_ORIGIN_DIRECTION_BANK_V4_VERDICT": bank_v,
        "BG_HS_INSPIRED_QUOTA_CONTROLLER_V4_VERDICT": controller_v,
        "BG_HIDDEN_ORIGIN_QUOTA_GENERATION_V4_VERDICT": generation_v,
        "BG_HIDDEN_ORIGIN_QUOTA_DIVERSITY_DRIVER_V4_VERDICT": drivers_v,
        "BG_HIDDEN_ORIGIN_QUOTA_DATASET_V4_VERDICT": dataset_v,
        "BG_HIDDEN_ORIGIN_TAP_TRAINING_V4_VERDICT": training_v,
        "BG_HIDDEN_ORIGIN_SELECTOR_EVAL_V4_VERDICT": eval_v,
        "BG_HIDDEN_ORIGIN_OLD_CONTEXT_REPLAY_V4_VERDICT": old_context_v,
        "BG_HIDDEN_ORIGIN_TAP_GEOMETRY_V4_VERDICT": geometry_v,
        "HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE_V4": best,
        "PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_V4": phase,
    }
    files_created = [
        QUOTA_PLAN_JSON,
        V4_ROOT / "quota_plan.md",
        V4_ROOT / "quota_plan_tasks.csv",
        DIRECTION_BANK_V4_JSON,
        V4_ROOT / "direction_bank_v4.md",
        V4_ROOT / "direction_bank_v4.pt",
        CONTROLLER_JSON,
        V4_ROOT / "hs_inspired_quota_controller.md",
        V4_ROOT / "recipe_engrams_v4.jsonl",
        V4_ROOT / "quota_hidden_origin_branches.pt",
        V4_ROOT / "quota_hidden_origin_branches.json",
        V4_ROOT / "quota_hidden_origin_branches.csv",
        GEN_REPORT_JSON,
        V4_ROOT / "quota_generation_report.md",
        V4_ROOT / "quota_diversity_drivers.json",
        V4_ROOT / "quota_diversity_drivers.md",
        V4_ROOT / "hidden_origin_quota_dataset_v4.pt",
        DATASET_V4_JSON,
        V4_ROOT / "hidden_origin_quota_dataset_v4.md",
        V4_ROOT / "hidden_origin_tap_heads_v4.pt",
        V4_ROOT / "training_log_v4.json",
        V4_ROOT / "training_report_v4.md",
        V4_ROOT / "heldout_eval_v4.json",
        V4_ROOT / "heldout_eval_v4.md",
        V4_ROOT / "heldout_eval_v4_rows.csv",
        V4_ROOT / "old_context_replay_v4.json",
        V4_ROOT / "old_context_replay_v4.md",
        V4_ROOT / "old_context_replay_v4_rows.csv",
        V4_ROOT / "geometry_analysis_v4.json",
        V4_ROOT / "geometry_analysis_v4.md",
        OUT_SUMMARY_MD,
        OUT_SUMMARY_JSON,
        OUT_ANALYSIS_MD,
        OUT_ANALYSIS_JSON,
        DOC_MD,
    ]
    blockers = []
    quota = generation.get("quota_progress_by_split") or dataset.get("quota_progress_by_split") or {}
    if not quota.get("all_minimums_met"):
        blockers.append("primary split quotas are not all met")
    if eval_v == "STILL_DATA_LIMITED":
        blockers.append("selector readiness cannot be claimed without enough primary-safe heldout support")
    if old_context_v in {"INCOMPATIBLE", "INSUFFICIENT"}:
        blockers.append("old-context replay has limited compatible coverage")
    payload = {
        **top_lines,
        "quota_progress_by_split": quota,
        "direction_bank_family_status": bank.get("family_status"),
        "controller_mode": controller.get("mode"),
        "diversity_driver_questions": drivers.get("questions"),
        "heldout_eval_support": {
            "heldout_task_ids": len(eval_payload.get("heldout_task_ids") or []),
            "heldout_pair_count": eval_payload.get("heldout_pair_count"),
            "behaviorally_diverse_heldout_groups": eval_payload.get("behaviorally_diverse_heldout_groups"),
            "readiness_support_met": eval_payload.get("readiness_support_met"),
        },
        "old_context_replay": {
            "verdict": old_context_v,
            "record_count": old_context.get("record_count"),
            "diagnostic_only": True,
            "production_routing_changed": False,
        },
        "geometry": {
            "max_abs_old_tap_alignment": geometry.get("max_abs_old_tap_alignment"),
            "v1_v2_v3_v4_alignment": geometry.get("v1_v2_v3_v4_alignment"),
            "v4_vs_salvage_alignment": geometry.get("v4_vs_salvage_alignment"),
            "old_context_replay_geometry_compatibility": geometry.get("old_context_replay_geometry_compatibility"),
        },
        "recommended_next": recommendation(phase),
        "files_created": [rel(path) for path in files_created],
        "commands_run": commands_run(),
        "blockers": blockers,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_SUMMARY_JSON, payload)
    write_json(OUT_ANALYSIS_JSON, payload)
    lines = ["# Hidden-Origin Quota V4 Summary", ""]
    lines.extend(f"{key} = {value}" for key, value in top_lines.items())
    lines.extend(
        [
            "",
            "## Motivation and Salvage Result",
            "",
            "V4 was needed because salvage found real hidden-origin ranking signal but inadequate strict heldout balance. V4 reserves train/val/heldout tasks before generation and generates directly into each split.",
            "",
            "## Quota Plan",
            "",
            f"- plan verdict: `{plan_v}`",
            f"- split task IDs: `{plan.get('split_task_ids')}`",
            "",
            "## Direction Bank",
            "",
            f"- direction bank verdict: `{bank_v}`",
            f"- ready non-random families: `{bank.get('ready_non_random_heldout_clean_families')}`",
            "",
            "## Hunter-Seeker-Inspired Recipe Controller",
            "",
            f"- controller verdict: `{controller_v}`",
            f"- recipe count: `{controller.get('recipe_count')}`",
            "",
            "## Quota Generation",
            "",
            f"- generation verdict: `{generation_v}`",
            f"- quota progress: `{quota}`",
            "",
            "## Diversity Drivers",
            "",
            f"- diversity driver verdict: `{drivers_v}`",
            f"- questions: `{drivers.get('questions')}`",
            "",
            "## Dataset",
            "",
            f"- dataset verdict: `{dataset_v}`",
            f"- primary pairs: `{dataset.get('primary_pair_count')}`",
            "",
            "## Training",
            "",
            f"- training verdict: `{training_v}`",
            f"- best head: `{training.get('best_head')}`",
            "",
            "## Heldout Selector Eval",
            "",
            f"- selector eval verdict: `{eval_v}`",
            f"- best selector available: `{best}`",
            f"- heldout support: `{payload['heldout_eval_support']}`",
            "",
            "## Old-Context Replay",
            "",
            f"- old-context replay verdict: `{old_context_v}`",
            "- diagnostic only; production routing unchanged",
            "",
            "## Geometry",
            "",
            f"- geometry verdict: `{geometry_v}`",
            f"- geometry summary: `{payload['geometry']}`",
            "",
            "## Phase 2 Readiness",
            "",
            f"PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_V4 = `{phase}`",
            "",
            "## Recommended Next Prototype",
            "",
            recommendation(phase),
            "",
            "## Files Created",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in payload["files_created"])
    lines.extend(["", "## Commands Run", ""])
    lines.extend(f"- `{cmd}`" for cmd in payload["commands_run"])
    lines.extend(["", "## Blockers", ""])
    lines.extend(f"- {item}" for item in blockers or ["none"])
    write_md(OUT_SUMMARY_MD, lines)
    write_md(OUT_ANALYSIS_MD, lines)

    doc_lines = [
        "# Hidden-Origin Branch Quota V4",
        "",
        *[f"{key} = {value}" for key, value in top_lines.items()],
        "",
        "## Why V4 Was Needed",
        "",
        "The split-salvage result showed positive selector signal but inadequate strict heldout support. V4 generates directly into pre-reserved train, val, and heldout splits.",
        "",
        "## Quota-Directed Generation",
        "",
        f"Generation verdict: `{generation_v}`. Quota progress: `{quota}`.",
        "",
        "## Recipe Controller",
        "",
        f"The local Hunter-Seeker-inspired controller verdict is `{controller_v}`. It is a recipe allocator only and does not use the actual architecture or action/runtime code paths.",
        "",
        "## Heldout Selector Result",
        "",
        f"Selector eval verdict: `{eval_v}`. Best available selector: `{best}`. Readiness support: `{payload['heldout_eval_support']}`.",
        "",
        "## Old-Context Replay",
        "",
        f"Old-context replay verdict: `{old_context_v}`. This is diagnostic only and does not replace old BG routing.",
        "",
        "## Geometry",
        "",
        f"Geometry verdict: `{geometry_v}` with summary `{payload['geometry']}`.",
        "",
        "## Phase 2 Implication",
        "",
        f"`PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_V4 = {phase}`. {recommendation(phase)}",
    ]
    write_md(DOC_MD, doc_lines)
    append_lines = [
        f"## {SECTION_TITLE}",
        "",
        f"- `BG_HIDDEN_ORIGIN_QUOTA_GENERATION_V4_VERDICT = {generation_v}`",
        f"- `BG_HIDDEN_ORIGIN_SELECTOR_EVAL_V4_VERDICT = {eval_v}`",
        f"- `BG_HIDDEN_ORIGIN_OLD_CONTEXT_REPLAY_V4_VERDICT = {old_context_v}`",
        f"- `BG_HIDDEN_ORIGIN_TAP_GEOMETRY_V4_VERDICT = {geometry_v}`",
        f"- `HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE_V4 = {best}`",
        f"- `PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_V4 = {phase}`",
        "",
        "V4 reserves train/val/heldout task IDs before generation and keeps old-context replay diagnostic-only. Alpha 0.02, sampled labels, and L47 remain excluded from primary readiness claims.",
    ]
    append_targets = [
        PROJECT_ROOT / "docs/evaluator/current_state.md",
        PROJECT_ROOT / "docs/evaluator/domain_transfer_ledger.md",
        PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_split_salvage.md",
        PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_diversity_v3.md",
        PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_diversity_v2.md",
        PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_taps.md",
        PROJECT_ROOT / "docs/evaluator/bg_hidden_state_branch_generation.md",
        PROJECT_ROOT / "docs/evaluator/bg_steering_consolidation_2026-05-18.md",
        PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-18_v8.1_routing_locked.md",
    ]
    appended = [rel(path) for path in append_targets if path.exists() and append_doc_section(path, SECTION_TITLE, append_lines)]
    payload["docs_appended"] = appended
    write_json(OUT_SUMMARY_JSON, payload)
    write_json(OUT_ANALYSIS_JSON, payload)
    print(f"PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_V4 = {phase}", flush=True)
    print(f"Wrote {rel(OUT_SUMMARY_MD)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
