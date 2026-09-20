#!/usr/bin/env python3
"""Aggregate the final frozen-backbone BG sequence-adapter experiment."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from bg_sequence_adapter_common import OUT_ROOT, QUICK_OUT_ROOT, append_once, finite, load_json, rel, write_json, write_md


ANALYSIS_JSON = OUT_ROOT / "analysis.json"
ANALYSIS_MD = OUT_ROOT / "analysis.md"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
DOC_PATH = Path("docs/evaluator/bg_sequence_level_adapter.md")
DOC_APPEND_TARGETS = [
    Path("docs/evaluator/current_state.md"),
    Path("docs/evaluator/domain_transfer_ledger.md"),
    Path("docs/evaluator/bg_causal_intervention_adapter.md"),
    Path("docs/evaluator/bg_preconsolidation_control_probes.md"),
    Path("docs/evaluator/post_v10_synthesis_2026-05-18_v8.1_routing_locked.md"),
]
SECTION_TITLE = "BG sequence-level adapter / final frozen-backbone steering test (2026-05-18)"


def verdict(payload: dict[str, Any], key: str, default: str = "INSUFFICIENT") -> str:
    return str(payload.get(key) or default)


def main() -> int:
    started = time.time()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    quick = load_json(QUICK_OUT_ROOT / "summary.json", {})
    preflight = load_json(OUT_ROOT / "preflight.json", {})
    dataset = load_json(OUT_ROOT / "sequence_adapter_dataset.json", {})
    impl = load_json(OUT_ROOT / "implementation_tests.json", {})
    baseline = load_json(OUT_ROOT / "baseline_eval.json", {})
    opt = load_json(OUT_ROOT / "optimizer_sanity.json", {})
    training = load_json(OUT_ROOT / "sequence_training_log.json", {})
    heldout = load_json(OUT_ROOT / "heldout_free_generation_eval.json", {})
    diag = load_json(OUT_ROOT / "diagnostics.json", {})

    parser_v = verdict(quick, "BG_SEQUENCE_PARSER_AUDIT_VERDICT", "BLOCKED")
    reward_v = verdict(quick, "BG_SEQUENCE_REWARD_DISTRIBUTION_VERDICT", "BLOCKED")
    micro_v = verdict(quick, "BG_SEQUENCE_OPTIMIZER_SANITY_MICRO_VERDICT", "BLOCKED")
    throughput_v = verdict(quick, "BG_SEQUENCE_GPU_THROUGHPUT_VERDICT", "BLOCKED")
    readiness_v = verdict(quick, "OVERNIGHT_SEQUENCE_ADAPTER_READINESS", "BLOCKED")
    preflight_v = verdict(preflight, "BG_SEQUENCE_ADAPTER_PREFLIGHT_VERDICT", "BLOCKED")
    dataset_v = verdict(dataset, "BG_SEQUENCE_ADAPTER_DATASET_VERDICT", "BLOCKED")
    impl_v = verdict(impl, "BG_SEQUENCE_ADAPTER_IMPLEMENTATION_VERDICT", "BLOCKED")
    baseline_v = verdict(baseline, "BG_SEQUENCE_BASELINE_EVAL_VERDICT", "BLOCKED")
    opt_v = verdict(opt, "BG_SEQUENCE_OPTIMIZER_SANITY_VERDICT", "BLOCKED")
    training_v = verdict(training, "BG_SEQUENCE_ADAPTER_TRAINING_VERDICT", "BLOCKED")
    heldout_v = verdict(heldout, "BG_SEQUENCE_ADAPTER_HELDOUT_VERDICT", "INSUFFICIENT")
    tf_v = verdict(diag, "BG_SEQUENCE_ADAPTER_TEACHER_FORCED_DIAG_VERDICT", "INSUFFICIENT")
    bg_v = verdict(diag, "BG_SEQUENCE_ADAPTER_BG_SCORE_DIAG_VERDICT", "INSUFFICIENT")
    geometry_v = verdict(diag, "BG_SEQUENCE_ADAPTER_GEOMETRY_VERDICT", "INCONCLUSIVE")

    if opt_v in {"OPTIMIZER_CANNOT_LEARN_TRIVIAL_TARGET", "OPTIMIZER_WEAK_ON_TRIVIAL_TARGET", "BLOCKED"}:
        learning_v = "OPTIMIZER_LIMITED"
    elif training_v == "SEQUENCE_REWARD_IMPROVES":
        learning_v = "LEARNS_SEQUENCE_REWARD"
    elif training_v == "WEAK_REWARD_IMPROVEMENT":
        learning_v = "WEAK_SEQUENCE_LEARNING"
    elif training_v == "NO_REWARD_IMPROVEMENT":
        learning_v = "NO_SEQUENCE_LEARNING"
    else:
        learning_v = "INSUFFICIENT"

    adapter_success = finite(heldout.get("mean_success_over_samples"), 0.0)
    non_adapter_success = finite(heldout.get("best_non_adapter_success_over_samples"), 0.0)
    if heldout_v == "INSUFFICIENT":
        vs_random_v = "INSUFFICIENT"
    elif adapter_success > non_adapter_success:
        vs_random_v = "BEATS_RANDOM"
    elif adapter_success < non_adapter_success:
        vs_random_v = "WORSE_THAN_RANDOM"
    else:
        vs_random_v = "MATCHES_RANDOM"

    if heldout_v == "DESTABILIZING" or training_v == "DESTABILIZING":
        stability_v = "DESTABILIZING"
    elif heldout_v == "INSUFFICIENT":
        stability_v = "INSUFFICIENT"
    elif finite((heldout.get("sampled_aggregate") or {}).get("method=trained_sequence_adapter|alpha=0.01", {}).get("parse_rate"), 1.0) < 0.75:
        stability_v = "STABLE_BUT_KL_HIGH"
    else:
        stability_v = "STABLE"

    if heldout_v == "ADAPTER_SPECIFIC_FREE_GEN_LIFT":
        transfer_v = "TRANSFERS_TO_HELDOUT_FREE_GEN"
    elif heldout_v == "WEAK_ADAPTER_SPECIFIC_LIFT":
        transfer_v = "WEAK_TRANSFER"
    elif heldout_v == "NO_ADAPTER_SPECIFIC_TRANSFER":
        transfer_v = "NO_TRANSFER"
    else:
        transfer_v = "INSUFFICIENT"

    if stability_v == "DESTABILIZING":
        overall_v = "DESTABILIZING"
    elif heldout_v == "ADAPTER_SPECIFIC_FREE_GEN_LIFT" and vs_random_v == "BEATS_RANDOM" and stability_v.startswith("STABLE"):
        overall_v = "INFERENCE_TIME_WRITE_PATH_FOUND"
    elif heldout_v == "WEAK_ADAPTER_SPECIFIC_LIFT" and vs_random_v == "BEATS_RANDOM":
        overall_v = "WEAK_INFERENCE_TIME_SIGNAL"
    elif opt_v in {"OPTIMIZER_CANNOT_LEARN_TRIVIAL_TARGET", "OPTIMIZER_WEAK_ON_TRIVIAL_TARGET", "BLOCKED"}:
        overall_v = "OPTIMIZER_LIMITED"
    elif heldout_v == "NO_ADAPTER_SPECIFIC_TRANSFER":
        overall_v = "NO_FROZEN_BACKBONE_WRITE_PATH"
    else:
        overall_v = "INSUFFICIENT"

    stopping_rule_applies = overall_v == "NO_FROZEN_BACKBONE_WRITE_PATH" and opt_v == "OPTIMIZER_CAN_LEARN_TRIVIAL_TARGET"
    if overall_v == "NO_FROZEN_BACKBONE_WRITE_PATH":
        steering_status = "CLOSED_UNDER_TESTED_METHODS" if stopping_rule_applies else "OPTIMIZER_LIMITED"
    elif overall_v == "OPTIMIZER_LIMITED":
        steering_status = "OPTIMIZER_LIMITED"
    elif overall_v in {"INFERENCE_TIME_WRITE_PATH_FOUND", "WEAK_INFERENCE_TIME_SIGNAL"}:
        steering_status = "STILL_PLAUSIBLE"
    else:
        steering_status = "INSUFFICIENT"
    if stopping_rule_applies:
        scope = "safe_alpha_leq_0_02_under_tested_optimizers"
        recommended_next = "consolidate_phase1_phase1_5_and_design_phase2_training_time_integration"
    elif overall_v == "OPTIMIZER_LIMITED" or opt_v == "OPTIMIZER_WEAK_ON_TRIVIAL_TARGET":
        scope = "optimizer_limited"
        recommended_next = "improve_black_box_optimizer_or_switch_to_ES_before_architectural_closure"
    elif overall_v == "INFERENCE_TIME_WRITE_PATH_FOUND":
        scope = "safe_alpha_leq_0_02_under_tested_optimizers"
        recommended_next = "expand_sequence_adapter_training_and_prepare_phase2_adapter_path"
    elif overall_v == "WEAK_INFERENCE_TIME_SIGNAL":
        scope = "insufficient"
        recommended_next = "expand_once_or_move_to_phase2_with_caveat"
    elif overall_v == "DESTABILIZING":
        scope = "safe_alpha_leq_0_02_under_tested_optimizers"
        recommended_next = "abandon_frozen_backbone_inference_time_steering_and_move_to_phase2"
    else:
        scope = "insufficient"
        recommended_next = "rerun_failed_component_only_or_consolidate_with_caveat"

    top = {
        "BG_SEQUENCE_PARSER_AUDIT_VERDICT": parser_v,
        "BG_SEQUENCE_REWARD_DISTRIBUTION_VERDICT": reward_v,
        "BG_SEQUENCE_OPTIMIZER_SANITY_MICRO_VERDICT": micro_v,
        "BG_SEQUENCE_GPU_THROUGHPUT_VERDICT": throughput_v,
        "OVERNIGHT_SEQUENCE_ADAPTER_READINESS": readiness_v,
        "BG_SEQUENCE_ADAPTER_PREFLIGHT_VERDICT": preflight_v,
        "BG_SEQUENCE_ADAPTER_DATASET_VERDICT": dataset_v,
        "BG_SEQUENCE_ADAPTER_IMPLEMENTATION_VERDICT": impl_v,
        "BG_SEQUENCE_BASELINE_EVAL_VERDICT": baseline_v,
        "BG_SEQUENCE_OPTIMIZER_SANITY_VERDICT": opt_v,
        "BG_SEQUENCE_ADAPTER_TRAINING_VERDICT": training_v,
        "BG_SEQUENCE_ADAPTER_HELDOUT_VERDICT": heldout_v,
        "BG_SEQUENCE_ADAPTER_TEACHER_FORCED_DIAG_VERDICT": tf_v,
        "BG_SEQUENCE_ADAPTER_BG_SCORE_DIAG_VERDICT": bg_v,
        "BG_SEQUENCE_ADAPTER_GEOMETRY_VERDICT": geometry_v,
        "BG_SEQUENCE_ADAPTER_LEARNING_VERDICT": learning_v,
        "BG_SEQUENCE_ADAPTER_VS_RANDOM_VERDICT": vs_random_v,
        "BG_SEQUENCE_ADAPTER_STABILITY_VERDICT": stability_v,
        "BG_SEQUENCE_ADAPTER_TRANSFER_VERDICT": transfer_v,
        "BG_SEQUENCE_LEVEL_ADAPTER_VERDICT": overall_v,
        "FROZEN_BACKBONE_INFERENCE_STEERING_STATUS": steering_status,
        "STOPPING_RULE_APPLIES": stopping_rule_applies,
        "STOPPING_RULE_SCOPE": scope,
        "RECOMMENDED_NEXT": recommended_next,
    }
    analysis = {
        **top,
        "required_analyses": {
            "quick_preflight_passed": readiness_v in {"READY", "READY_WITH_WARNINGS"},
            "full_optimizer_sanity_passed": opt_v == "OPTIMIZER_CAN_LEARN_TRIVIAL_TARGET",
            "sequence_level_reward_improved_on_validation": training_v in {"SEQUENCE_REWARD_IMPROVES", "WEAK_REWARD_IMPROVEMENT"},
            "sequence_level_reward_transferred_to_heldout": transfer_v in {"TRANSFERS_TO_HELDOUT_FREE_GEN", "WEAK_TRANSFER"},
            "adapter_beat_random_static": vs_random_v == "BEATS_RANDOM",
            "adapter_improved_more_than_one_task_sample": int(heldout.get("adapter_only_moved_task_count") or 0) >= 2,
            "effect_adapter_specific": heldout_v in {"ADAPTER_SPECIFIC_FREE_GEN_LIFT", "WEAK_ADAPTER_SPECIFIC_LIFT"},
            "stable": stability_v.startswith("STABLE"),
            "new_geometry": geometry_v == "NEW_WRITE_GEOMETRY",
            "keeps_frozen_backbone_steering_alive": overall_v in {"INFERENCE_TIME_WRITE_PATH_FOUND", "WEAK_INFERENCE_TIME_SIGNAL"},
        },
        "key_metrics": {
            "best_val_reward": training.get("best_val_reward"),
            "baseline_val_reward": training.get("baseline_val_reward"),
            "heldout_adapter_success": adapter_success,
            "heldout_best_non_adapter_success": non_adapter_success,
            "heldout_adapter_only_moved_task_count": heldout.get("adapter_only_moved_task_count"),
            "heldout_moved_task_count": heldout.get("moved_task_count"),
            "teacher_forced_margin_lift_mean": diag.get("teacher_forced_margin_lift_mean"),
        },
        "report_paths": {
            "quick_summary": rel(QUICK_OUT_ROOT / "summary.md"),
            "preflight": rel(OUT_ROOT / "preflight.md"),
            "dataset": rel(OUT_ROOT / "sequence_adapter_dataset.md"),
            "implementation": rel(OUT_ROOT / "implementation_tests.md"),
            "baseline_eval": rel(OUT_ROOT / "baseline_eval.md"),
            "optimizer_sanity": rel(OUT_ROOT / "optimizer_sanity.md"),
            "training": rel(OUT_ROOT / "sequence_training_report.md"),
            "heldout": rel(OUT_ROOT / "heldout_free_generation_eval.md"),
            "diagnostics": rel(OUT_ROOT / "diagnostics.md"),
            "analysis": rel(ANALYSIS_MD),
            "summary": rel(SUMMARY_MD),
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(ANALYSIS_JSON, analysis)

    def top_lines(include_status: bool = False) -> list[str]:
        lines = [f"{key} = {value}" for key, value in top.items()]
        return lines if include_status else lines[:-4]

    analysis_lines = [
        "# BG Sequence-Level Adapter Analysis",
        "",
        *top_lines(include_status=False),
        "",
        "## Stopping Rule Decision",
        "",
        f"- stopping rule applies: `{stopping_rule_applies}`",
        f"- stopping rule scope: `{scope}`",
        f"- recommended next: `{recommended_next}`",
        "",
        "## Key Metrics",
        "",
        f"- best validation reward: `{training.get('best_val_reward')}`",
        f"- baseline validation reward: `{training.get('baseline_val_reward')}`",
        f"- heldout adapter success: `{adapter_success:.3f}`",
        f"- heldout best non-adapter success: `{non_adapter_success:.3f}`",
        f"- adapter-only moved tasks: `{heldout.get('adapter_only_moved_task_count')}`",
        f"- teacher-forced margin lift: `{diag.get('teacher_forced_margin_lift_mean')}`",
    ]
    write_md(ANALYSIS_MD, analysis_lines)

    summary_lines = [
        "# BG Sequence-Level Adapter Summary",
        "",
        *[f"{key} = {value}" for key, value in top.items()],
        "",
        "## Motivation and Stopping Rule",
        "",
        "This was the final serious frozen-backbone inference-time BG steering test at safe intervention magnitude.",
        "",
        "## Quick Preflight Gate",
        "",
        f"- readiness: `{readiness_v}`",
        "",
        "## Dataset",
        "",
        f"- dataset verdict: `{dataset_v}`",
        f"- task counts: `{dataset.get('task_counts')}`",
        "",
        "## Adapter Architecture",
        "",
        "- `LowRankDeltaAdapter(rank=32)` through a layer-36 `multi_loop_decayed` hook.",
        "",
        "## Optimizer Sanity Check",
        "",
        f"- full sanity verdict: `{opt_v}`",
        "",
        "## Optimizer and Reward",
        "",
        "- optimizer: `REINFORCE_score_function`",
        "- reward: `+1 correct`, `0 wrong parseable`, negative parse/empty/repetition/error penalties.",
        "",
        "## Baselines",
        "",
        f"- baseline verdict: `{baseline_v}`",
        "",
        "## Training Results",
        "",
        f"- training verdict: `{training_v}`",
        f"- best validation reward: `{training.get('best_val_reward')}`",
        "",
        "## Heldout Sampled Free-Generation Eval",
        "",
        f"- heldout verdict: `{heldout_v}`",
        f"- sampled n: `{heldout.get('n_samples')}`",
        f"- adapter success: `{adapter_success:.3f}`",
        f"- best non-adapter success: `{non_adapter_success:.3f}`",
        "",
        "## Adapter-Specificity Analysis",
        "",
        f"- vs random/static: `{vs_random_v}`",
        f"- adapter-only moved tasks: `{heldout.get('adapter_only_moved_task_count')}`",
        "",
        "## Diagnostics",
        "",
        f"- teacher-forced: `{tf_v}`",
        f"- BG score: `{bg_v}`",
        f"- geometry: `{geometry_v}`",
        "",
        "## Phase 2 Implication",
        "",
        f"- recommended next: `{recommended_next}`",
        "",
        "## Docs Updated",
        "",
        f"- `{rel(DOC_PATH)}`",
        *[f"- `{rel(path)}`" for path in DOC_APPEND_TARGETS],
        "",
        "## Files Modified / Created",
        "",
        *[f"- `{path}`" for path in analysis["report_paths"].values()],
        "",
        "## Commands Run",
        "",
        "- see prompt run sequence; no git commands were run by these scripts.",
        "",
        "## Blockers",
        "",
        "- none recorded" if overall_v not in {"INSUFFICIENT", "OPTIMIZER_LIMITED"} else f"- overall verdict: `{overall_v}`",
    ]
    write_json(SUMMARY_JSON, {**analysis, "summary_top_lines": top})
    write_md(SUMMARY_MD, summary_lines)

    doc_lines = [
        "# BG Sequence-Level Adapter",
        "",
        "## Why This Was the Final Frozen-Backbone Test",
        "",
        "The prior BG steering-control stack showed that Ouro is BG-readable and mechanically writable, but static and teacher-forced inference-time write paths did not transfer to free generation. This run used actual generated-output reward rather than answer-token teacher forcing.",
        "",
        "## Result",
        "",
        *[f"- {key}: `{value}`" for key, value in top.items()],
        "",
        "## Optimizer",
        "",
        "- optimizer: `REINFORCE_score_function`",
        "- sampled tokens were treated with a score-function estimator; the run did not backpropagate through sampling.",
        "- reward: correct parsed MCQ answer `+1`, parseable wrong `0`, parse failure/empty/repetition/error penalties.",
        "",
        "## Data and Evaluation",
        "",
        f"- dataset task counts: `{dataset.get('task_counts')}`",
        f"- heldout sampled n: `{heldout.get('n_samples')}`",
        f"- heldout adapter success: `{adapter_success:.3f}`",
        f"- heldout best non-adapter success: `{non_adapter_success:.3f}`",
        "",
        "## Stopping Rule",
        "",
        f"- applies: `{stopping_rule_applies}`",
        f"- scope: `{scope}`",
        f"- Phase 2 implication: `{recommended_next}`",
        "",
        "## Reports",
        "",
        *[f"- `{path}`" for path in analysis["report_paths"].values()],
    ]
    write_md(DOC_PATH, doc_lines)

    append_lines = [
        *[f"- {key}: `{value}`" for key, value in top.items()],
        "- STOPPING_RULE_SCOPE: `" + scope + "`",
        "- report paths:",
        *[f"  - `{path}`" for path in analysis["report_paths"].values()],
    ]
    for path in DOC_APPEND_TARGETS:
        append_once(path, SECTION_TITLE, append_lines)

    print(f"BG_SEQUENCE_LEVEL_ADAPTER_VERDICT = {overall_v}")
    print(f"FROZEN_BACKBONE_INFERENCE_STEERING_STATUS = {steering_status}")
    print(f"Wrote {rel(ANALYSIS_JSON)}")
    print(f"Wrote {rel(SUMMARY_JSON)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
