"""Aggregate hidden-origin diversity v2 experiment outputs and update docs."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from bg_hidden_origin_diversity_v2_common import (
    PROJECT_ROOT,
    V2_ROOT,
    append_doc_section,
    load_json,
    md_table,
    rate,
    rel,
    v2_commands_run,
    write_json,
    write_md,
)


SUMMARY_JSON = V2_ROOT / "summary.json"
SUMMARY_MD = V2_ROOT / "summary.md"
ANALYSIS_JSON = V2_ROOT / "analysis.json"
ANALYSIS_MD = V2_ROOT / "analysis.md"
DOC_PATH = PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_diversity_v2.md"
SECTION_TITLE = "Hidden-origin branch diversity v2 and tap reevaluation (2026-05-18)"


def verdict(filename: str, key: str) -> str:
    payload = load_json(V2_ROOT / filename, {}) or {}
    return str(payload.get(key) or payload.get("verdict") or "NOT_RUN")


def phase2_status(summary: dict[str, str], metrics: dict[str, Any]) -> str:
    values = set(summary.values())
    if "BLOCKED" in values or "INSUFFICIENT" in {
        summary["BG_HIDDEN_ORIGIN_DIVERSITY_AUDIT_VERDICT"],
        summary["BG_HIDDEN_ORIGIN_TASK_SCREENING_VERDICT"],
        summary["BG_HIDDEN_ORIGIN_DIRECTION_BANK_VERDICT"],
    }:
        return "INSUFFICIENT"
    generation = summary["BG_HIDDEN_ORIGIN_DIVERSITY_V2_GENERATION_VERDICT"]
    dataset = summary["BG_HIDDEN_ORIGIN_TAP_DATASET_V2_VERDICT"]
    training = summary["BG_HIDDEN_ORIGIN_TAP_TRAINING_V2_VERDICT"]
    eval_v = summary["BG_HIDDEN_ORIGIN_TAP_EVAL_V2_VERDICT"]
    geometry = summary["BG_HIDDEN_ORIGIN_TAP_GEOMETRY_V2_VERDICT"]
    tie_rate = metrics.get("dataset", {}).get("primary_tie_rate")
    if generation == "LOW_DIVERSITY" or (tie_rate is not None and float(tie_rate) > 0.90 and dataset == "DATA_LIMITED"):
        return "NEEDS_BETTER_BRANCH_GENERATOR"
    if eval_v == "DATA_LIMITED" or dataset == "DATA_LIMITED":
        return "STILL_DATA_LIMITED"
    if eval_v == "SELECTOR_READY" and training in {"READY", "WEAK"} and geometry in {"OLD_GEOMETRY_CONFIRMED", "NEW_STABLE_GEOMETRY", "MIXED_GEOMETRY"}:
        return "READY"
    if eval_v == "WEAK_SELECTOR":
        return "WEAK"
    if eval_v in {"NO_SELECTOR_SIGNAL", "OVERFIT"}:
        return "NOT_READY"
    return "STILL_DATA_LIMITED"


def compact_metrics() -> dict[str, Any]:
    audit = load_json(V2_ROOT / "prior_audit.json", {}) or {}
    screening = load_json(V2_ROOT / "task_screening.json", {}) or {}
    bank = load_json(V2_ROOT / "direction_bank.json", {}) or {}
    generation = load_json(V2_ROOT / "diverse_generation_report.json", {}) or {}
    dataset = load_json(V2_ROOT / "hidden_origin_tap_dataset_v2.json", {}) or {}
    training = load_json(V2_ROOT / "training_log_v2.json", {}) or {}
    eval_payload = load_json(V2_ROOT / "heldout_eval_v2.json", {}) or {}
    layers = load_json(V2_ROOT / "layer_generation_analysis_v2.json", {}) or {}
    geometry = load_json(V2_ROOT / "geometry_analysis_v2.json", {}) or {}
    primary_meta = (dataset.get("variant_meta") or {}).get("primary_safe_deterministic", {})
    return {
        "audit": {
            "prior_rows": audit.get("prior_total_rows"),
            "prior_behaviorally_diverse_groups": audit.get("prior_behaviorally_diverse_groups"),
            "prior_tie_rate": audit.get("prior_tie_rate"),
        },
        "screening": {
            "screened_count": screening.get("screened_count"),
            "selected_count": screening.get("selected_count"),
            "selected_preferred_count": screening.get("selected_preferred_count"),
            "selected_class_counts": screening.get("selected_class_counts"),
        },
        "direction_bank": {
            "family_counts": bank.get("family_counts"),
            "layers": bank.get("layers"),
        },
        "generation": {
            "stats": generation.get("stats"),
            "counts_by_screening_class": generation.get("counts_by_screening_class"),
            "counts_by_branch_point": generation.get("counts_by_branch_point"),
            "counts_by_alpha_bucket": generation.get("counts_by_alpha_bucket"),
            "counts_by_delta_family": generation.get("counts_by_delta_family"),
        },
        "dataset": {
            "primary_pairs": len(dataset.get("pairs") or []),
            "pairs_by_split": dataset.get("pairs_by_split"),
            "tasks_by_split": {k: len(v) for k, v in (dataset.get("tasks_by_split") or {}).items()},
            "behaviorally_diverse_groups_by_split": dataset.get("behaviorally_diverse_groups_by_split"),
            "primary_tie_rate": primary_meta.get("tie_rate"),
            "variant_meta": dataset.get("variant_meta"),
        },
        "training": {
            "best_head": training.get("best_head"),
            "trained_heads": len(training.get("heads") or []),
            "anti_degeneracy": training.get("anti_degeneracy"),
        },
        "heldout": {
            "heldout_task_ids": eval_payload.get("heldout_task_ids"),
            "heldout_group_count": eval_payload.get("heldout_group_count"),
            "heldout_pair_count": eval_payload.get("heldout_pair_count"),
            "behaviorally_diverse_heldout_groups": eval_payload.get("behaviorally_diverse_heldout_groups"),
            "best_head": eval_payload.get("best_head"),
            "previous_v1_head": eval_payload.get("previous_v1_head"),
            "old_frozen_pairwise_accuracy": eval_payload.get("old_frozen_pairwise_accuracy"),
        },
        "layer_generation": {
            "diversity_source_verdict": layers.get("BG_HIDDEN_ORIGIN_DIVERSITY_SOURCE_VERDICT"),
            "layer_config_verdict": layers.get("BG_HIDDEN_ORIGIN_LAYER_CONFIG_V2_VERDICT"),
            "best_behaviorally_diverse_config": layers.get("best_behaviorally_diverse_config"),
            "first_phase2_scoring_point_recommendation": layers.get("first_phase2_scoring_point_recommendation"),
        },
        "geometry": {
            "max_abs_old_tap_alignment": geometry.get("max_abs_old_tap_alignment"),
            "mean_old_tap_alignment": geometry.get("mean_old_tap_alignment"),
            "v1_v2_alignment": geometry.get("v1_v2_alignment"),
            "mean_seed_config_stability": geometry.get("mean_seed_config_stability"),
        },
    }


def recommended_next(status: str) -> str:
    if status == "READY":
        return "Design a minimal Phase 2 selection-only prototype with hook-hidden-origin branch generation, hidden-origin tap selection, top-k retention, and no L30/L42 gates yet."
    if status == "WEAK":
        return "Either expand once more or proceed only to a small selection-only prototype with the caveat locked in."
    if status == "STILL_DATA_LIMITED":
        return "Continue targeted data expansion with stronger diversity drivers and enforce at least four heldout task IDs before selector claims."
    if status == "NOT_READY":
        return "Improve feature capture or evaluator design before any steering work."
    if status == "NEEDS_BETTER_BRANCH_GENERATOR":
        return "Focus on branch generation strength and diversity; tap training is not the current bottleneck."
    return "Resolve blocked stages before interpreting v2 selection."


def write_docs(summary: dict[str, str], metrics: dict[str, Any], status: str) -> list[str]:
    created = []
    eval_payload = load_json(V2_ROOT / "heldout_eval_v2.json", {}) or {}
    best_behavior = None
    for row in (eval_payload.get("behaviorally_diverse_metrics") or {}).values():
        if str(row.get("policy", "")).startswith("new_hidden_origin_tap_v2_pairwise_tournament"):
            if best_behavior is None or float(row.get("top1_success", 0.0)) > float(best_behavior.get("top1_success", 0.0)):
                best_behavior = row
    doc_lines = [
        "# Hidden-Origin Branch Diversity V2",
        "",
        "V2 was run because the first hidden-origin tap experiment trained technically but remained data-limited: the prior expansion had a very high tie rate and too few task-disjoint behaviorally diverse heldout groups.",
        "",
        "## Verdicts",
        "",
        *[f"- {key} = `{value}`" for key, value in summary.items()],
        "",
        "## Task Selection",
        "",
        "Task screening favored clean-wrong, parse-fragile, low-confidence, and perturbation-sensitive reasoning/science MCQ tasks. Confident clean-correct tasks were deprioritized.",
        "",
        f"- selected_count: `{metrics['screening'].get('selected_count')}`",
        f"- selected_preferred_count: `{metrics['screening'].get('selected_preferred_count')}`",
        f"- selected_class_counts: `{metrics['screening'].get('selected_class_counts')}`",
        "",
        "## Direction Bank And Generation",
        "",
        "The v2 generator used clean, random paired, orthogonal random, old-tap-aligned, hidden-origin empirical, and available proxy directions as perturbation candidates. Alpha `0.02` is reported as a separate diagnostic bucket and is not mixed into the primary safe-alpha headline.",
        "",
        f"- direction_families: `{metrics['direction_bank'].get('family_counts')}`",
        f"- generation_stats: `{metrics['generation'].get('stats')}`",
        "",
        "## Dataset V2",
        "",
        "Primary labels are deterministic downstream rewards within the same branch group, using only stable rows with `alpha <= 0.01`. Reward ties are omitted, not assigned arbitrary labels. Alpha `0.02` and sampled expected reward are diagnostic variants.",
        "",
        f"- primary_pairs: `{metrics['dataset'].get('primary_pairs')}`",
        f"- pairs_by_split: `{metrics['dataset'].get('pairs_by_split')}`",
        f"- behaviorally_diverse_groups_by_split: `{metrics['dataset'].get('behaviorally_diverse_groups_by_split')}`",
        f"- primary_tie_rate: `{rate(metrics['dataset'].get('primary_tie_rate'))}`",
        "",
        "## Training And Evaluation",
        "",
        "The headline heads remain exact antisymmetric `AntisymLinear` and `AntisymLinearNoNorm` trained with pairwise logsigmoid ranking, random left/right swaps, target sign flips, and flip diagnostics.",
        "",
        f"- best_v2_head: `{metrics['training'].get('best_head')}`",
        f"- heldout_summary: `{metrics['heldout']}`",
        f"- best_behaviorally_diverse_new_policy: `{best_behavior}`",
        "",
        "## Layer And Geometry",
        "",
        f"- diversity_source_verdict: `{summary['BG_HIDDEN_ORIGIN_DIVERSITY_SOURCE_VERDICT']}`",
        f"- layer_config_verdict: `{summary['BG_HIDDEN_ORIGIN_LAYER_CONFIG_V2_VERDICT']}`",
        f"- geometry_verdict: `{summary['BG_HIDDEN_ORIGIN_TAP_GEOMETRY_V2_VERDICT']}`",
        f"- geometry: `{metrics['geometry']}`",
        "",
        "## Phase 2 Implication",
        "",
        f"PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_V2 = `{status}`",
        "",
        recommended_next(status),
    ]
    write_md(DOC_PATH, doc_lines)
    created.append(rel(DOC_PATH))

    append_lines = [
        f"## {SECTION_TITLE}",
        "",
        f"- PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_V2 = `{status}`",
        f"- generation_verdict = `{summary['BG_HIDDEN_ORIGIN_DIVERSITY_V2_GENERATION_VERDICT']}`",
        f"- dataset_verdict = `{summary['BG_HIDDEN_ORIGIN_TAP_DATASET_V2_VERDICT']}`",
        f"- training_verdict = `{summary['BG_HIDDEN_ORIGIN_TAP_TRAINING_V2_VERDICT']}`",
        f"- eval_verdict = `{summary['BG_HIDDEN_ORIGIN_TAP_EVAL_V2_VERDICT']}`",
        f"- layer_config_verdict = `{summary['BG_HIDDEN_ORIGIN_LAYER_CONFIG_V2_VERDICT']}`",
        f"- geometry_verdict = `{summary['BG_HIDDEN_ORIGIN_TAP_GEOMETRY_V2_VERDICT']}`",
        f"- report: `{rel(V2_ROOT / 'summary.md')}`",
        "",
        recommended_next(status),
        "",
    ]
    for doc in (
        PROJECT_ROOT / "docs/evaluator/current_state.md",
        PROJECT_ROOT / "docs/evaluator/domain_transfer_ledger.md",
        PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_taps.md",
        PROJECT_ROOT / "docs/evaluator/bg_hidden_state_branch_generation.md",
        PROJECT_ROOT / "docs/evaluator/bg_steering_consolidation_2026-05-18.md",
        PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-18_v8.1_routing_locked.md",
    ):
        if append_doc_section(doc, SECTION_TITLE, append_lines):
            created.append(rel(doc))
    return created


def main() -> int:
    started = time.time()
    summary = {
        "BG_HIDDEN_ORIGIN_DIVERSITY_AUDIT_VERDICT": verdict("prior_audit.json", "BG_HIDDEN_ORIGIN_DIVERSITY_AUDIT_VERDICT"),
        "BG_HIDDEN_ORIGIN_TASK_SCREENING_VERDICT": verdict("task_screening.json", "BG_HIDDEN_ORIGIN_TASK_SCREENING_VERDICT"),
        "BG_HIDDEN_ORIGIN_DIRECTION_BANK_VERDICT": verdict("direction_bank.json", "BG_HIDDEN_ORIGIN_DIRECTION_BANK_VERDICT"),
        "BG_HIDDEN_ORIGIN_DIVERSITY_V2_GENERATION_VERDICT": verdict("diverse_generation_report.json", "BG_HIDDEN_ORIGIN_DIVERSITY_V2_GENERATION_VERDICT"),
        "BG_HIDDEN_ORIGIN_TAP_DATASET_V2_VERDICT": verdict("hidden_origin_tap_dataset_v2.json", "BG_HIDDEN_ORIGIN_TAP_DATASET_V2_VERDICT"),
        "BG_HIDDEN_ORIGIN_TAP_TRAINING_V2_VERDICT": verdict("training_log_v2.json", "BG_HIDDEN_ORIGIN_TAP_TRAINING_V2_VERDICT"),
        "BG_HIDDEN_ORIGIN_TAP_EVAL_V2_VERDICT": verdict("heldout_eval_v2.json", "BG_HIDDEN_ORIGIN_TAP_EVAL_V2_VERDICT"),
        "BG_HIDDEN_ORIGIN_DIVERSITY_SOURCE_VERDICT": verdict("layer_generation_analysis_v2.json", "BG_HIDDEN_ORIGIN_DIVERSITY_SOURCE_VERDICT"),
        "BG_HIDDEN_ORIGIN_LAYER_CONFIG_V2_VERDICT": verdict("layer_generation_analysis_v2.json", "BG_HIDDEN_ORIGIN_LAYER_CONFIG_V2_VERDICT"),
        "BG_HIDDEN_ORIGIN_TAP_GEOMETRY_V2_VERDICT": verdict("geometry_analysis_v2.json", "BG_HIDDEN_ORIGIN_TAP_GEOMETRY_V2_VERDICT"),
    }
    metrics = compact_metrics()
    status = phase2_status(summary, metrics)
    summary["PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_V2"] = status
    docs_created = write_docs(summary, metrics, status)
    files_created = [
        rel(V2_ROOT / name)
        for name in (
            "prior_audit.md",
            "prior_audit.json",
            "task_screening.md",
            "task_screening.json",
            "task_screening_rows.csv",
            "direction_bank.pt",
            "direction_bank.json",
            "direction_bank.md",
            "diverse_hidden_origin_branches.pt",
            "diverse_hidden_origin_branches.json",
            "diverse_hidden_origin_branches.csv",
            "diverse_generation_report.md",
            "diverse_generation_report.json",
            "hidden_origin_tap_dataset_v2.pt",
            "hidden_origin_tap_dataset_v2.json",
            "hidden_origin_tap_dataset_v2.md",
            "hidden_origin_tap_heads_v2.pt",
            "training_log_v2.json",
            "training_report_v2.md",
            "heldout_eval_v2.md",
            "heldout_eval_v2.json",
            "heldout_eval_v2_rows.csv",
            "layer_generation_analysis_v2.md",
            "layer_generation_analysis_v2.json",
            "geometry_analysis_v2.md",
            "geometry_analysis_v2.json",
            "summary.md",
            "summary.json",
            "analysis.md",
            "analysis.json",
        )
        if (V2_ROOT / name).exists() or name in {"summary.md", "summary.json", "analysis.md", "analysis.json"}
    ]
    blockers = []
    if status in {"STILL_DATA_LIMITED", "INSUFFICIENT"}:
        blockers.append("heldout behaviorally diverse groups, heldout task IDs, or heldout pairs remain below target")
    if status == "NEEDS_BETTER_BRANCH_GENERATOR":
        blockers.append("behavioral diversity or non-tie label yield remains too low for robust selector evaluation")
    if status == "NOT_READY":
        blockers.append("v2 tap did not beat required heldout behaviorally diverse baselines")
    payload = {
        **summary,
        "metrics": metrics,
        "recommended_next": recommended_next(status),
        "files_created": files_created + docs_created,
        "commands_run": v2_commands_run(),
        "blockers": blockers,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(SUMMARY_JSON, payload)
    write_json(ANALYSIS_JSON, payload)

    lines = [
        "# Hidden-Origin Branch Diversity V2 Summary",
        "",
        *[f"{key} = {value}" for key, value in summary.items()],
        "",
        "## Motivation And Prior Bottleneck",
        "",
        "The first hidden-origin tap run trained technically but was data-limited: too many reward ties and too few task-disjoint behaviorally diverse heldout groups.",
        "",
        "## Task Screening",
        "",
        f"- selected_count: `{metrics['screening'].get('selected_count')}`",
        f"- selected_preferred_count: `{metrics['screening'].get('selected_preferred_count')}`",
        f"- selected_class_counts: `{metrics['screening'].get('selected_class_counts')}`",
        "",
        "## Direction Bank",
        "",
        f"- family_counts: `{metrics['direction_bank'].get('family_counts')}`",
        f"- layers: `{metrics['direction_bank'].get('layers')}`",
        "",
        "## Behaviorally Targeted Generation",
        "",
        f"- generation_stats: `{metrics['generation'].get('stats')}`",
        "",
        "## Dataset V2",
        "",
        f"- primary_pairs: `{metrics['dataset'].get('primary_pairs')}`",
        f"- pairs_by_split: `{metrics['dataset'].get('pairs_by_split')}`",
        f"- behaviorally_diverse_groups_by_split: `{metrics['dataset'].get('behaviorally_diverse_groups_by_split')}`",
        f"- primary_tie_rate: `{rate(metrics['dataset'].get('primary_tie_rate'))}`",
        "",
        "## Training V2",
        "",
        f"- best_head: `{metrics['training'].get('best_head')}`",
        f"- trained_heads: `{metrics['training'].get('trained_heads')}`",
        "",
        "## Heldout Eval V2",
        "",
        f"- heldout: `{metrics['heldout']}`",
        "",
        "## Layer And Generation Analysis",
        "",
        f"- layer_generation: `{metrics['layer_generation']}`",
        "",
        "## Geometry",
        "",
        f"- geometry: `{metrics['geometry']}`",
        "",
        "## Phase 2 Readiness",
        "",
        f"PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_V2 = `{status}`",
        "",
        recommended_next(status),
        "",
        "## Files Created",
        "",
        *[f"- `{path}`" for path in files_created + docs_created],
        "",
        "## Commands Run",
        "",
        *[f"- `{cmd}`" for cmd in v2_commands_run()],
        "",
        "## Blockers",
        "",
        *([f"- {item}" for item in blockers] if blockers else ["- None."]),
    ]
    write_md(SUMMARY_MD, lines)
    write_md(ANALYSIS_MD, lines)
    print(f"PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_V2 = {status}", flush=True)
    print(f"Wrote {rel(SUMMARY_MD)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

