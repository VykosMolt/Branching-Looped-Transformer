"""Aggregate hidden-origin diversity v3 experiment outputs and update docs."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from bg_hidden_origin_diversity_v3_common import (
    PROJECT_ROOT,
    V3_ROOT,
    append_doc_section,
    load_json,
    rate,
    rel,
    v3_commands_run,
    write_json,
    write_md,
)


SUMMARY_JSON = V3_ROOT / "summary.json"
SUMMARY_MD = V3_ROOT / "summary.md"
ANALYSIS_JSON = V3_ROOT / "analysis.json"
ANALYSIS_MD = V3_ROOT / "analysis.md"
DOC_PATH = PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_diversity_v3.md"
SECTION_TITLE = "Hidden-origin branch diversity v3 and selector reevaluation (2026-05-18)"


def verdict(filename: str, key: str) -> str:
    payload = load_json(V3_ROOT / filename, {}) or {}
    return str(payload.get(key) or payload.get("verdict") or "NOT_RUN")


def compact_metrics() -> dict[str, Any]:
    audit = load_json(V3_ROOT / "v3_audit.json", {}) or {}
    task_selection = load_json(V3_ROOT / "task_selection_v3.json", {}) or {}
    split = load_json(V3_ROOT / "split_guard_v3.json", {}) or {}
    bank = load_json(V3_ROOT / "direction_bank_v3.json", {}) or {}
    generation = load_json(V3_ROOT / "diversity_ablation_report_v3.json", {}) or {}
    drivers = load_json(V3_ROOT / "diversity_drivers_v3.json", {}) or {}
    dataset = load_json(V3_ROOT / "hidden_origin_tap_dataset_v3.json", {}) or {}
    training = load_json(V3_ROOT / "training_log_v3.json", {}) or {}
    eval_payload = load_json(V3_ROOT / "heldout_eval_v3.json", {}) or {}
    geometry = load_json(V3_ROOT / "geometry_analysis_v3.json", {}) or {}
    primary_meta = (dataset.get("variant_meta") or {}).get("primary_safe_deterministic", {})
    return {
        "audit": {
            "prior_rows": audit.get("prior_rows"),
            "primary_metrics": audit.get("primary_metrics"),
            "top_task_classes": audit.get("top_diversity_yielding_task_classes"),
        },
        "task_selection": {
            "selected_count": task_selection.get("selected_count"),
            "selected_high_medium_count": task_selection.get("selected_high_medium_count"),
            "selected_domain_counts": task_selection.get("selected_domain_counts"),
            "selected_screening_class_counts": task_selection.get("selected_screening_class_counts"),
        },
        "split_guard": {
            "counts": split.get("counts"),
            "baseline_overlap": split.get("v1_v2_baseline_may_have_seen_v3_heldout_task_ids"),
            "clean_cross_version_heldout_task_ids": split.get("clean_cross_version_heldout_task_ids"),
        },
        "direction_bank": {
            "family_counts": bank.get("family_counts"),
            "family_status": bank.get("family_status"),
            "layers": bank.get("layers"),
        },
        "generation": {
            "stats": generation.get("stats"),
            "counts_by_primary_delta_family": generation.get("counts_by_primary_delta_family"),
            "counts_by_branch_point": generation.get("counts_by_branch_point"),
            "counts_by_alpha_bucket": generation.get("counts_by_alpha_bucket"),
            "checkpointing": generation.get("checkpointing"),
        },
        "drivers": {
            "primary_metrics": drivers.get("primary_metrics"),
            "questions": drivers.get("questions"),
            "recommended_branch_generation_recipe": drivers.get("recommended_branch_generation_recipe"),
        },
        "dataset": {
            "primary_pairs": len(dataset.get("pairs") or []),
            "pairs_by_split": dataset.get("pairs_by_split"),
            "tasks_by_split": {k: len(v) for k, v in (dataset.get("tasks_by_split") or {}).items()},
            "behaviorally_diverse_groups_by_split": dataset.get("behaviorally_diverse_groups_by_split"),
            "primary_tie_rate": primary_meta.get("tie_rate"),
            "variant_meta": dataset.get("variant_meta"),
            "baseline_leakage_warning_task_ids": dataset.get("baseline_leakage_warning_task_ids"),
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
            "best_v3_head": eval_payload.get("best_v3_head"),
            "previous_v1_head": eval_payload.get("previous_v1_head"),
            "previous_v2_head": eval_payload.get("previous_v2_head"),
            "old_frozen_pairwise_accuracy": eval_payload.get("old_frozen_pairwise_accuracy"),
        },
        "geometry": {
            "max_abs_old_tap_alignment": geometry.get("max_abs_old_tap_alignment"),
            "mean_old_tap_alignment": geometry.get("mean_old_tap_alignment"),
            "v1_v2_v3_alignment": geometry.get("v1_v2_v3_alignment"),
            "mean_seed_config_stability": geometry.get("mean_seed_config_stability"),
        },
    }


def best_available_selector(eval_payload: dict[str, Any]) -> str:
    behavior = eval_payload.get("behaviorally_diverse_metrics") or {}
    candidates = []
    mapping = {
        "old_frozen_bg_pairwise_tournament": "old_frozen_bg",
        "previous_hidden_origin_tap_v1_pairwise_tournament": "v1_hidden_origin_tap",
        "previous_hidden_origin_tap_v2_pairwise_tournament": "v2_hidden_origin_tap",
        "new_hidden_origin_tap_v3_pairwise_tournament": "v3_hidden_origin_tap",
        "random_top1": "random",
    }
    for row in behavior.values():
        policy = str(row.get("policy"))
        if policy in mapping:
            candidates.append((mapping[policy], float(row.get("top1_success", 0.0)), float(row.get("reward_mean", 0.0)), float(row.get("selection_regret", 999.0))))
    if not candidates:
        return "insufficient"
    candidates.sort(key=lambda item: (item[1], item[2], -item[3]), reverse=True)
    return candidates[0][0]


def phase2_status(summary: dict[str, str], metrics: dict[str, Any], best_selector: str) -> str:
    generation = summary["BG_HIDDEN_ORIGIN_DIVERSITY_ABLATION_V3_VERDICT"]
    dataset = summary["BG_HIDDEN_ORIGIN_TAP_DATASET_V3_VERDICT"]
    training = summary["BG_HIDDEN_ORIGIN_TAP_TRAINING_V3_VERDICT"]
    eval_v = summary["BG_HIDDEN_ORIGIN_TAP_EVAL_V3_VERDICT"]
    geometry = summary["BG_HIDDEN_ORIGIN_TAP_GEOMETRY_V3_VERDICT"]
    heldout = metrics["heldout"]
    heldout_tasks = len(heldout.get("heldout_task_ids") or [])
    heldout_pairs = int(heldout.get("heldout_pair_count") or 0)
    heldout_behavior = int(heldout.get("behaviorally_diverse_heldout_groups") or 0)
    primary_tie_rate = metrics["dataset"].get("primary_tie_rate")
    if "BLOCKED" in summary.values() or "NOT_RUN" in summary.values():
        if generation == "LOW_DIVERSITY":
            return "NEEDS_BETTER_BRANCH_GENERATOR"
        return "INSUFFICIENT"
    if generation == "LOW_DIVERSITY" or (primary_tie_rate is not None and float(primary_tie_rate) > 0.90 and dataset in {"STILL_DATA_LIMITED", "BLOCKED"}):
        return "NEEDS_BETTER_BRANCH_GENERATOR"
    if dataset == "STILL_DATA_LIMITED" or eval_v == "DATA_LIMITED" or heldout_tasks < 6 or heldout_pairs < 80 or heldout_behavior < 15:
        return "STILL_DATA_LIMITED"
    if eval_v == "SELECTOR_READY" and training in {"READY", "WEAK"} and dataset in {"READY", "SMALL_BUT_USABLE"}:
        return "READY"
    if best_selector == "old_frozen_bg" and dataset in {"READY", "SMALL_BUT_USABLE"} and heldout_tasks >= 6 and heldout_pairs >= 80 and heldout_behavior >= 15:
        return "READY"
    if eval_v == "WEAK_SELECTOR":
        return "WEAK"
    if eval_v in {"NO_SELECTOR_SIGNAL", "OVERFIT"}:
        return "NOT_READY"
    if geometry == "INCONCLUSIVE":
        return "INSUFFICIENT"
    return "WEAK"


def recommended_next(status: str, driver_verdict: str) -> str:
    if status == "READY":
        return "Design a minimal Phase 2 selection-only prototype: hook-hidden-origin branch generation, best available hidden-origin selector, top-k selection, no L30/L42 gates yet, and a locked selection-only baseline."
    if status == "WEAK":
        return "Run only a small selection-only prototype with the caveat locked in, or expand once more if the diversity recipe is clear."
    if status == "STILL_DATA_LIMITED":
        if driver_verdict in {"CLEAR_DIVERSITY_RECIPE", "NON_RANDOM_DIRECTIONS_HELP", "K_EXPANSION_HELPS", "TASK_SCREENING_ONLY"}:
            return "Continue targeted data expansion using the v3 recipe before making selector-readiness claims."
        return "Do not expand blindly; first improve the branch generator or task-screening recipe."
    if status == "NOT_READY":
        return "Improve evaluator/features before steering; the heldout selector signal is not sufficient."
    if status == "NEEDS_BETTER_BRANCH_GENERATOR":
        return "Stop expanding this same approach and focus on stronger branch generation, true fork-carry, or a different perturbation mechanism."
    return "Resolve blocked stages before interpreting v3."


def write_docs(summary: dict[str, str], metrics: dict[str, Any], status: str, best_selector: str, recommendation: str) -> list[str]:
    created = []
    doc_lines = [
        "# Hidden-Origin Branch Diversity V3",
        "",
        "V3 was needed because v1 was data-limited and v2 produced only a weak selector: the evaluator machinery worked, but hidden-origin branch generation still produced too many downstream reward ties.",
        "",
        "## Verdicts",
        "",
        *[f"- {key} = `{value}`" for key, value in summary.items()],
        f"- HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE = `{best_selector}`",
        f"- PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_V3 = `{status}`",
        "",
        "## Task Selection",
        "",
        "Task selection prioritized reasoning/science MCQ tasks with parse fragility, wrong-but-parseable clean answers, low-confidence clean correctness, perturbation sensitivity, prior reward variance, and old/v1/v2 disagreement.",
        "",
        f"- selection: `{metrics['task_selection']}`",
        "",
        "## Split And Leakage Guard",
        "",
        "The v3 heldout set was reserved before empirical hidden-origin direction construction. Heldout task IDs are excluded from v3 empirical mean-diff/whitened directions and v3 tap training.",
        "",
        f"- split_guard: `{metrics['split_guard']}`",
        "",
        "## Direction Bank",
        "",
        "The bank records perturbation compatibility explicitly. Concat directions are scoring/geometry-only unless a validated projection exists, and L47 remains diagnostic.",
        "",
        f"- direction_bank: `{metrics['direction_bank']}`",
        "",
        "## Diversity Ablation",
        "",
        "Primary diversity uses stable deterministic alpha `<= 0.01` L24/L36 reasoning/science same-prefix hidden-origin rows. Alpha `0.02`, sampled expected reward, and L47 are diagnostic only.",
        "",
        f"- generation: `{metrics['generation']}`",
        "",
        "## Diversity Drivers",
        "",
        f"- drivers: `{metrics['drivers']}`",
        "",
        "## Dataset V3",
        "",
        "Primary pairwise labels omit ties rather than assigning arbitrary labels.",
        "",
        f"- dataset: `{metrics['dataset']}`",
        "",
        "## Training And Evaluation",
        "",
        "The headline v3 taps are old-style antisymmetric tiny heads trained with same-group Bradley-Terry/logsigmoid ranking, random left/right swaps, target sign flips, and flip diagnostics.",
        "",
        f"- training: `{metrics['training']}`",
        f"- heldout: `{metrics['heldout']}`",
        "",
        "## Geometry V3",
        "",
        f"- geometry: `{metrics['geometry']}`",
        "",
        "## Phase 2 Implication",
        "",
        recommendation,
    ]
    write_md(DOC_PATH, doc_lines)
    created.append(rel(DOC_PATH))
    append_lines = [
        f"## {SECTION_TITLE}",
        "",
        f"- PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_V3 = `{status}`",
        f"- HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE = `{best_selector}`",
        f"- diversity_ablation_verdict = `{summary['BG_HIDDEN_ORIGIN_DIVERSITY_ABLATION_V3_VERDICT']}`",
        f"- driver_verdict = `{summary['BG_HIDDEN_ORIGIN_DIVERSITY_DRIVER_V3_VERDICT']}`",
        f"- dataset_verdict = `{summary['BG_HIDDEN_ORIGIN_TAP_DATASET_V3_VERDICT']}`",
        f"- training_verdict = `{summary['BG_HIDDEN_ORIGIN_TAP_TRAINING_V3_VERDICT']}`",
        f"- eval_verdict = `{summary['BG_HIDDEN_ORIGIN_TAP_EVAL_V3_VERDICT']}`",
        f"- geometry_verdict = `{summary['BG_HIDDEN_ORIGIN_TAP_GEOMETRY_V3_VERDICT']}`",
        f"- report: `{rel(V3_ROOT / 'summary.md')}`",
        "",
        recommendation,
        "",
    ]
    for doc in (
        PROJECT_ROOT / "docs/evaluator/current_state.md",
        PROJECT_ROOT / "docs/evaluator/domain_transfer_ledger.md",
        PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_diversity_v2.md",
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
    V3_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {
        "BG_HIDDEN_ORIGIN_DIVERSITY_V3_AUDIT_VERDICT": verdict("v3_audit.json", "BG_HIDDEN_ORIGIN_DIVERSITY_V3_AUDIT_VERDICT"),
        "BG_HIDDEN_ORIGIN_TASK_SELECTION_V3_VERDICT": verdict("task_selection_v3.json", "BG_HIDDEN_ORIGIN_TASK_SELECTION_V3_VERDICT"),
        "BG_HIDDEN_ORIGIN_SPLIT_GUARD_V3_VERDICT": verdict("split_guard_v3.json", "BG_HIDDEN_ORIGIN_SPLIT_GUARD_V3_VERDICT"),
        "BG_HIDDEN_ORIGIN_DIRECTION_BANK_V3_VERDICT": verdict("direction_bank_v3.json", "BG_HIDDEN_ORIGIN_DIRECTION_BANK_V3_VERDICT"),
        "BG_HIDDEN_ORIGIN_DIVERSITY_ABLATION_V3_VERDICT": verdict("diversity_ablation_report_v3.json", "BG_HIDDEN_ORIGIN_DIVERSITY_ABLATION_V3_VERDICT"),
        "BG_HIDDEN_ORIGIN_DIVERSITY_DRIVER_V3_VERDICT": verdict("diversity_drivers_v3.json", "BG_HIDDEN_ORIGIN_DIVERSITY_DRIVER_V3_VERDICT"),
        "BG_HIDDEN_ORIGIN_TAP_DATASET_V3_VERDICT": verdict("hidden_origin_tap_dataset_v3.json", "BG_HIDDEN_ORIGIN_TAP_DATASET_V3_VERDICT"),
        "BG_HIDDEN_ORIGIN_TAP_TRAINING_V3_VERDICT": verdict("training_log_v3.json", "BG_HIDDEN_ORIGIN_TAP_TRAINING_V3_VERDICT"),
        "BG_HIDDEN_ORIGIN_TAP_EVAL_V3_VERDICT": verdict("heldout_eval_v3.json", "BG_HIDDEN_ORIGIN_TAP_EVAL_V3_VERDICT"),
        "BG_HIDDEN_ORIGIN_TAP_GEOMETRY_V3_VERDICT": verdict("geometry_analysis_v3.json", "BG_HIDDEN_ORIGIN_TAP_GEOMETRY_V3_VERDICT"),
    }
    metrics = compact_metrics()
    eval_payload = load_json(V3_ROOT / "heldout_eval_v3.json", {}) or {}
    best_selector = best_available_selector(eval_payload)
    status = phase2_status(summary, metrics, best_selector)
    summary["HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE"] = best_selector
    summary["PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_V3"] = status
    recommendation = recommended_next(status, summary["BG_HIDDEN_ORIGIN_DIVERSITY_DRIVER_V3_VERDICT"])
    docs_created = write_docs(summary, metrics, status, best_selector, recommendation)
    files_created = [
        rel(V3_ROOT / name)
        for name in (
            "v3_audit.md",
            "v3_audit.json",
            "task_selection_v3.md",
            "task_selection_v3.json",
            "task_selection_v3_rows.csv",
            "split_guard_v3.md",
            "split_guard_v3.json",
            "direction_bank_v3.pt",
            "direction_bank_v3.json",
            "direction_bank_v3.md",
            "diversity_ablation_v3.pt",
            "diversity_ablation_v3.json",
            "diversity_ablation_v3.csv",
            "diversity_ablation_progress.jsonl",
            "diversity_ablation_report_v3.md",
            "diversity_ablation_report_v3.json",
            "diversity_drivers_v3.md",
            "diversity_drivers_v3.json",
            "hidden_origin_tap_dataset_v3.pt",
            "hidden_origin_tap_dataset_v3.json",
            "hidden_origin_tap_dataset_v3.md",
            "hidden_origin_tap_heads_v3.pt",
            "training_log_v3.json",
            "training_report_v3.md",
            "heldout_eval_v3.md",
            "heldout_eval_v3.json",
            "heldout_eval_v3_rows.csv",
            "geometry_analysis_v3.md",
            "geometry_analysis_v3.json",
            "summary.md",
            "summary.json",
            "analysis.md",
            "analysis.json",
        )
        if (V3_ROOT / name).exists() or name in {"summary.md", "summary.json", "analysis.md", "analysis.json"}
    ]
    blockers = []
    if status in {"STILL_DATA_LIMITED", "INSUFFICIENT"}:
        blockers.append("heldout behaviorally diverse groups, task IDs, or non-tie pairs remain below minimum")
    if status == "NEEDS_BETTER_BRANCH_GENERATOR":
        blockers.append("tie rate or behavioral diversity remains too poor for meaningful selector readiness")
    if status == "NOT_READY":
        blockers.append("heldout selector signal did not beat required baselines")
    payload = {
        **summary,
        "metrics": metrics,
        "recommended_next": recommendation,
        "files_created": files_created + docs_created,
        "commands_run": v3_commands_run(),
        "blockers": blockers,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(SUMMARY_JSON, payload)
    write_json(ANALYSIS_JSON, payload)
    lines = [
        "# Hidden-Origin Branch Diversity V3 Summary",
        "",
        *[f"{key} = {value}" for key, value in summary.items()],
        "",
        "## Motivation And V2 Bottleneck",
        "",
        "V2 confirmed that hidden-origin tap training/evaluation works technically, but branch generation still produced too many reward ties for selector-ready claims.",
        "",
        "## V3 Task Selection",
        "",
        f"- task_selection: `{metrics['task_selection']}`",
        "",
        "## Split/Leakage Guard",
        "",
        f"- split_guard: `{metrics['split_guard']}`",
        "",
        "## Direction Bank",
        "",
        f"- direction_bank: `{metrics['direction_bank']}`",
        "",
        "## Diversity Ablation",
        "",
        f"- generation: `{metrics['generation']}`",
        "",
        "## Diversity Drivers",
        "",
        f"- drivers: `{metrics['drivers']}`",
        "",
        "## Dataset V3",
        "",
        f"- dataset: `{metrics['dataset']}`",
        "",
        "## Training V3",
        "",
        f"- training: `{metrics['training']}`",
        "",
        "## Heldout Eval V3",
        "",
        f"- heldout: `{metrics['heldout']}`",
        "",
        "## Selector Best Available",
        "",
        f"HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE = `{best_selector}`",
        "",
        "## Geometry V3",
        "",
        f"- geometry: `{metrics['geometry']}`",
        "",
        "## Phase 2 Readiness",
        "",
        f"PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_V3 = `{status}`",
        "",
        "## Recommended Next Prototype",
        "",
        recommendation,
        "",
        "## Files Created",
        "",
        *[f"- `{path}`" for path in files_created + docs_created],
        "",
        "## Commands Run",
        "",
        *[f"- `{cmd}`" for cmd in v3_commands_run()],
        "",
        "## Blockers",
        "",
        *([f"- {item}" for item in blockers] if blockers else ["- None."]),
    ]
    write_md(SUMMARY_MD, lines)
    write_md(ANALYSIS_MD, lines)
    print(f"PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_V3 = {status}", flush=True)
    print(f"HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE = {best_selector}", flush=True)
    print(f"Wrote {rel(SUMMARY_MD)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

