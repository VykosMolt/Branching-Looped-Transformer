"""Aggregate hidden-origin tap experiment outputs and update docs."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from bg_hidden_origin_tap_common import (
    OUT_ROOT,
    PROJECT_ROOT,
    append_doc_section,
    commands_run,
    load_json,
    md_table,
    rate,
    rel,
    write_json,
    write_md,
)


SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
ANALYSIS_JSON = OUT_ROOT / "analysis.json"
ANALYSIS_MD = OUT_ROOT / "analysis.md"
DOC_PATH = PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_taps.md"
SECTION_TITLE = "Hidden-origin branch taps (2026-05-18)"


def verdict(payload_name: str, key: str) -> str:
    payload = load_json(OUT_ROOT / payload_name, {}) or {}
    return str(payload.get(key) or payload.get("verdict") or "NOT_RUN")


def phase2_status(summary: dict[str, str]) -> str:
    inv = summary["BG_HIDDEN_ORIGIN_TAP_INVENTORY_VERDICT"]
    data = summary["BG_HIDDEN_ORIGIN_TAP_DATASET_VERDICT"]
    train = summary["BG_HIDDEN_ORIGIN_TAP_TRAINING_VERDICT"]
    eval_v = summary["BG_HIDDEN_ORIGIN_TAP_EVAL_VERDICT"]
    geom = summary["BG_HIDDEN_ORIGIN_TAP_GEOMETRY_VERDICT"]
    if "BLOCKED" in {inv, data} or train == "INSUFFICIENT":
        return "INSUFFICIENT"
    if eval_v == "INSUFFICIENT" or data in {"TOO_SMALL", "SMALL_BUT_USABLE"} and eval_v != "SELECTOR_READY":
        return "DATA_LIMITED"
    if eval_v == "SELECTOR_READY" and train in {"READY", "WEAK"} and geom in {"NEW_STABLE_GEOMETRY", "ALIGNS_WITH_OLD_TAPS"}:
        return "READY"
    if eval_v == "WEAK_SELECTOR" or train == "WEAK":
        return "WEAK"
    if eval_v in {"NO_SELECTOR_SIGNAL", "OVERFIT"}:
        return "NOT_READY"
    return "DATA_LIMITED"


def compact_metrics() -> dict[str, Any]:
    inventory = load_json(OUT_ROOT / "inventory.json", {}) or {}
    dataset = load_json(OUT_ROOT / "hidden_origin_tap_dataset.json", {}) or {}
    training = load_json(OUT_ROOT / "training_log.json", {}) or {}
    eval_payload = load_json(OUT_ROOT / "heldout_eval.json", {}) or {}
    layers = load_json(OUT_ROOT / "layer_config_analysis.json", {}) or {}
    geometry = load_json(OUT_ROOT / "geometry_analysis.json", {}) or {}
    return {
        "inventory_counts": {
            "rows": inventory.get("total_rows"),
            "groups": inventory.get("total_branch_groups"),
            "stable_safe_groups": inventory.get("stable_safe_branch_groups"),
            "behaviorally_diverse_groups": inventory.get("behaviorally_diverse_group_count"),
            "reward_diverse_groups": inventory.get("reward_diverse_group_count"),
            "tasks": inventory.get("tasks"),
        },
        "dataset_counts": {
            "valid_branches": dataset.get("valid_branch_count"),
            "pairs_by_split": dataset.get("pairs_by_split"),
            "tasks_by_split": {k: len(v) for k, v in (dataset.get("tasks_by_split") or {}).items()},
            "tie_rate": (dataset.get("pair_meta") or {}).get("tie_rate"),
        },
        "training": {
            "best_head": training.get("best_head"),
            "trained_heads": len(training.get("heads") or []),
            "anti_degeneracy": training.get("anti_degeneracy"),
        },
        "heldout": {
            "heldout_group_count": eval_payload.get("heldout_group_count"),
            "heldout_pair_count": eval_payload.get("heldout_pair_count"),
            "behaviorally_diverse_heldout_groups": eval_payload.get("behaviorally_diverse_heldout_groups"),
            "best_head": eval_payload.get("best_head"),
            "old_frozen_pairwise_accuracy": eval_payload.get("old_frozen_pairwise_accuracy"),
        },
        "layer": {
            "best_behaviorally_diverse_config": layers.get("best_behaviorally_diverse_config"),
            "first_phase2_scoring_point_recommendation": layers.get("first_phase2_scoring_point_recommendation"),
        },
        "geometry": {
            "max_abs_old_tap_alignment": geometry.get("max_abs_old_tap_alignment"),
            "mean_seed_config_stability": geometry.get("mean_seed_config_stability"),
            "empirical_geometry_vs_heldout_pairwise_correlation": geometry.get("empirical_geometry_vs_heldout_pairwise_correlation"),
        },
    }


def next_step(status: str) -> str:
    if status == "READY":
        return "Design minimal Phase 2 prototype with hook-hidden-origin branches, the new selector, top-k selection, and no L30/L42 gates yet."
    if status == "WEAK":
        return "Expand hidden-origin outcome data before training steering."
    if status == "NOT_READY":
        return "Improve branch generation or feature capture before Phase 2 training."
    if status == "DATA_LIMITED":
        return "Generate more hidden-origin branch outcome groups."
    return "Resolve blockers before interpreting hidden-origin tap selection."


def write_docs(summary: dict[str, str], metrics: dict[str, Any], status: str) -> list[str]:
    created = []
    eval_payload = load_json(OUT_ROOT / "heldout_eval.json", {}) or {}
    best_behavior = None
    for row in (eval_payload.get("behaviorally_diverse_metrics") or {}).values():
        if str(row.get("policy", "")).startswith("new_hidden_origin_tap_pairwise_tournament"):
            if best_behavior is None or float(row.get("top1_success", 0.0)) > float(best_behavior.get("top1_success", 0.0)):
                best_behavior = row
    doc_lines = [
        "# Hidden-Origin Branch Taps",
        "",
        "Hidden-origin taps are separate from the frozen canonical BG taps. The old BG heads read hidden states, but their training states came from normal candidate/text/code/option trajectories. This experiment trains tiny heads on same-prefix hidden-state perturbation branches and labels pairs by downstream branch outcomes.",
        "",
        "## Verdicts",
        "",
        *[f"- {key} = `{value}`" for key, value in summary.items()],
        "",
        "## Dataset And Labels",
        "",
        "Rows are filtered to safe alpha branches (`alpha <= 0.01`) and stable outputs. Pair labels are built only within the same hidden-origin branch group: the preferred branch has the higher downstream reward. Reward ties are omitted from primary training rather than assigned arbitrary labels.",
        "",
        f"- pairs_by_split: `{metrics['dataset_counts'].get('pairs_by_split')}`",
        f"- tie_rate: `{rate(metrics['dataset_counts'].get('tie_rate'))}`",
        "",
        "## Training",
        "",
        "The headline architectures are exact antisymmetric tiny heads: `AntisymLinear` and `AntisymLinearNoNorm`. Training uses a Bradley-Terry/logsigmoid-style pairwise objective with 50 percent random left/right swaps and target sign flips. The report includes flip diagnostics and rejects constant-score solutions.",
        "",
        f"- best_head: `{metrics['training'].get('best_head')}`",
        "",
        "## Heldout Selection",
        "",
        "Heldout evaluation uses task IDs excluded from training and compares random, clean branch, old frozen BG tap margins, and the new hidden-origin tap policies. The behaviorally diverse subset is the load-bearing subset.",
        "",
        f"- best_behaviorally_diverse_new_policy: `{best_behavior}`",
        f"- old_frozen_pairwise_accuracy: `{rate(metrics['heldout'].get('old_frozen_pairwise_accuracy'))}`",
        "",
        "## Layer And Geometry",
        "",
        f"- layer_config_verdict: `{summary['BG_HIDDEN_ORIGIN_LAYER_CONFIG_VERDICT']}`",
        f"- first_phase2_scoring_point_recommendation: `{metrics['layer'].get('first_phase2_scoring_point_recommendation')}`",
        f"- geometry_verdict: `{summary['BG_HIDDEN_ORIGIN_TAP_GEOMETRY_VERDICT']}`",
        "",
        "## Phase 2 Implication",
        "",
        f"PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS = `{status}`",
        "",
        next_step(status),
    ]
    write_md(DOC_PATH, doc_lines)
    created.append(rel(DOC_PATH))

    append_lines = [
        f"## {SECTION_TITLE}",
        "",
        f"- PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS = `{status}`",
        f"- tap_eval_verdict = `{summary['BG_HIDDEN_ORIGIN_TAP_EVAL_VERDICT']}`",
        f"- tap_training_verdict = `{summary['BG_HIDDEN_ORIGIN_TAP_TRAINING_VERDICT']}`",
        f"- layer_config_verdict = `{summary['BG_HIDDEN_ORIGIN_LAYER_CONFIG_VERDICT']}`",
        f"- geometry_verdict = `{summary['BG_HIDDEN_ORIGIN_TAP_GEOMETRY_VERDICT']}`",
        f"- report: `{rel(OUT_ROOT / 'summary.md')}`",
        "",
        next_step(status),
        "",
    ]
    for doc in (
        PROJECT_ROOT / "docs/evaluator/current_state.md",
        PROJECT_ROOT / "docs/evaluator/domain_transfer_ledger.md",
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
        "BG_HIDDEN_ORIGIN_TAP_INVENTORY_VERDICT": verdict("inventory.json", "BG_HIDDEN_ORIGIN_TAP_INVENTORY_VERDICT"),
        "BG_HIDDEN_ORIGIN_DATA_EXPANSION_VERDICT": verdict("expansion_report.json", "BG_HIDDEN_ORIGIN_DATA_EXPANSION_VERDICT"),
        "BG_HIDDEN_ORIGIN_TAP_DATASET_VERDICT": verdict("hidden_origin_tap_dataset.json", "BG_HIDDEN_ORIGIN_TAP_DATASET_VERDICT"),
        "BG_HIDDEN_ORIGIN_TAP_TRAINING_VERDICT": verdict("training_log.json", "BG_HIDDEN_ORIGIN_TAP_TRAINING_VERDICT"),
        "BG_HIDDEN_ORIGIN_TAP_EVAL_VERDICT": verdict("heldout_eval.json", "BG_HIDDEN_ORIGIN_TAP_EVAL_VERDICT"),
        "BG_HIDDEN_ORIGIN_LAYER_CONFIG_VERDICT": verdict("layer_config_analysis.json", "BG_HIDDEN_ORIGIN_LAYER_CONFIG_VERDICT"),
        "BG_HIDDEN_ORIGIN_TAP_GEOMETRY_VERDICT": verdict("geometry_analysis.json", "BG_HIDDEN_ORIGIN_TAP_GEOMETRY_VERDICT"),
    }
    status = phase2_status(summary)
    summary["PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS"] = status
    metrics = compact_metrics()
    docs_created = write_docs(summary, metrics, status)
    files_created = [
        rel(OUT_ROOT / name)
        for name in (
            "inventory.md",
            "inventory.json",
            "expanded_hidden_origin_branches.pt",
            "expanded_hidden_origin_branches.json",
            "expanded_hidden_origin_branches.csv",
            "expansion_report.md",
            "expansion_report.json",
            "hidden_origin_tap_dataset.pt",
            "hidden_origin_tap_dataset.json",
            "hidden_origin_tap_dataset.md",
            "hidden_origin_tap_heads.pt",
            "training_log.json",
            "training_report.md",
            "heldout_eval.md",
            "heldout_eval.json",
            "heldout_eval_rows.csv",
            "layer_config_analysis.md",
            "layer_config_analysis.json",
            "geometry_analysis.md",
            "geometry_analysis.json",
            "summary.md",
            "summary.json",
            "analysis.md",
            "analysis.json",
        )
        if (OUT_ROOT / name).exists() or name in {"summary.md", "summary.json", "analysis.md", "analysis.json"}
    ]
    blockers = []
    if status in {"DATA_LIMITED", "INSUFFICIENT"}:
        blockers.append("heldout behaviorally diverse groups or task-disjoint pairs are still limited")
    if summary["BG_HIDDEN_ORIGIN_TAP_EVAL_VERDICT"] in {"NO_SELECTOR_SIGNAL", "OVERFIT"}:
        blockers.append("new tap did not beat the required heldout behaviorally diverse baselines")

    payload = {
        **summary,
        "metrics": metrics,
        "recommended_next": next_step(status),
        "files_created": files_created + docs_created,
        "commands_run": commands_run(),
        "blockers": blockers,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(SUMMARY_JSON, payload)
    write_json(ANALYSIS_JSON, payload)

    lines = [
        "# Hidden-Origin Branch Taps Summary",
        "",
        *[f"{key} = {value}" for key, value in summary.items()],
        "",
        "## Motivation",
        "",
        "Existing frozen BG taps read hidden states, but they were trained on normal candidate trajectories. This run trains tiny heads on same-prefix hidden-origin branch states and labels branch pairs by downstream rewards.",
        "",
        "## Dataset",
        "",
        f"- inventory_counts: `{metrics['inventory_counts']}`",
        f"- dataset_counts: `{metrics['dataset_counts']}`",
        "",
        "## Training Procedure",
        "",
        "Exact antisymmetric `AntisymLinear` and `AntisymLinearNoNorm` heads were trained with same-group pairwise ranking, random swap/sign-flip augmentation, score L2, gradient clipping, and validation selection.",
        "",
        "## Anti-Degeneracy And Flip Tests",
        "",
        f"- anti_degeneracy: `{metrics['training'].get('anti_degeneracy')}`",
        f"- best_head: `{metrics['training'].get('best_head')}`",
        "",
        "## Heldout Selection",
        "",
        f"- heldout: `{metrics['heldout']}`",
        "",
        "## Layer/Config Analysis",
        "",
        f"- layer: `{metrics['layer']}`",
        "",
        "## Geometry Analysis",
        "",
        f"- geometry: `{metrics['geometry']}`",
        "",
        "## Phase 2 Readiness",
        "",
        f"- recommended_next: `{next_step(status)}`",
        "",
        "## Files Created",
        "",
        *[f"- `{path}`" for path in payload["files_created"]],
        "",
        "## Commands Run",
        "",
        *[f"- `{cmd}`" for cmd in commands_run()],
        "",
        "## Blockers",
        "",
        *(f"- {item}" for item in blockers),
    ]
    write_md(SUMMARY_MD, lines)
    write_md(ANALYSIS_MD, lines)
    print(f"PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS = {status}", flush=True)
    print(f"Wrote {rel(SUMMARY_JSON)}", flush=True)
    print(f"Wrote {rel(SUMMARY_MD)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
