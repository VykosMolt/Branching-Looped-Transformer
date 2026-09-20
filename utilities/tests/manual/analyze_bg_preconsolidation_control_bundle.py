"""Consolidate BG pre-consolidation control probes and update docs."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from bg_preconsolidation_common import (
    EMPIRICAL_ROOT,
    OUT_ROOT,
    PROJECT_ROOT,
    STAGE2_LAYERHOOK_ROOT,
    append_once,
    finite,
    load_json,
    rel,
    write_json,
    write_md,
)


OUT_JSON = OUT_ROOT / "final_analysis.json"
OUT_MD = OUT_ROOT / "final_analysis.md"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
DOC_MAIN = PROJECT_ROOT / "docs/evaluator/bg_preconsolidation_control_probes.md"
APPEND_DOCS = [
    PROJECT_ROOT / "docs/evaluator/current_state.md",
    PROJECT_ROOT / "docs/evaluator/domain_transfer_ledger.md",
    PROJECT_ROOT / "docs/evaluator/bg_stage2_layerhook_followup.md",
    PROJECT_ROOT / "docs/evaluator/bg_empirical_steering_direction.md",
    PROJECT_ROOT / "docs/evaluator/bg_trajectory_prediction_sweep.md",
    PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-18_v8.1_routing_locked.md",
]


def get(payload: dict[str, Any], key: str, default: Any = "INSUFFICIENT") -> Any:
    value = payload.get(key)
    return default if value in {None, ""} else value


def steering_verdict(rms: dict[str, Any], gradient: dict[str, Any], empirical: dict[str, Any]) -> str:
    rms_v = get(rms, "BG_RMS_STEERING_VERDICT")
    rms_s = get(rms, "BG_RMS_STABILITY_VERDICT")
    grad_v = get(gradient, "BG_CAUSAL_GRADIENT_VERDICT", "SKIPPED")
    emp_v = get(empirical, "BG_EMPIRICAL_DIRECTION_CAUSAL_VERDICT")
    if rms_v == "RMS_SIGNED_CAUSAL" and rms_s in {"STABLE", "STABLE_BUT_NOISY"}:
        return "STATIC_STEERING_WORKS"
    if grad_v == "LOCAL_CAUSAL_DIRECTION_EXISTS":
        return "GRADIENT_ONLY"
    if rms_v == "RMS_DESTABILIZING" or grad_v == "GRADIENT_DESTABILIZING":
        return "DESTABILIZING"
    if rms_v == "RMS_UNSIGNED_ONLY" or emp_v == "EMPIRICAL_UNSIGNED_ONLY":
        return "UNSIGNED_ONLY"
    if rms_v in {"INSUFFICIENT", "BLOCKED"} and grad_v in {"INSUFFICIENT", "SKIPPED"}:
        return "INSUFFICIENT"
    return "NO_RELIABLE_INFERENCE_STEERING"


def normalized_rms(rms: dict[str, Any], row_audit: dict[str, Any]) -> dict[str, Any]:
    """Apply the row-level RMS reinterpretation for downstream synthesis.

    The first-pass RMS analyzer intentionally used a hard conservative rule:
    any safety row made the whole RMS probe DESTABILIZING.  The row-level
    audit distinguishes broad instability from a single output-quality outlier.
    """
    out = dict(rms or {})
    if row_audit.get("RMS_DESTABILIZING_REINTERPRETATION") == "single_output_quality_outlier_not_broad_destabilization":
        out["BG_RMS_STABILITY_VERDICT_RAW"] = out.get("BG_RMS_STABILITY_VERDICT")
        out["BG_RMS_STEERING_VERDICT_RAW"] = out.get("BG_RMS_STEERING_VERDICT")
        out["BG_RMS_STABILITY_VERDICT"] = row_audit.get("BG_RMS_STABILITY_VERDICT_ROW_LEVEL", "STABLE_BUT_NOISY")
        out["BG_RMS_STEERING_VERDICT"] = row_audit.get("BG_RMS_STEERING_VERDICT_ROW_LEVEL", "RMS_UNSIGNED_ONLY")
        out["RMS_ROW_LEVEL_REINTERPRETATION"] = row_audit.get("RMS_DESTABILIZING_REINTERPRETATION")
        out["RMS_ROW_LEVEL_SAFETY_FAILURE_COUNT"] = row_audit.get("safety_failure_count")
    return out


def branch_verdict(text: dict[str, Any]) -> str:
    value = get(text, "BG_TEXT_PREFIX_EXPANSION_VERDICT")
    if value == "HELPS":
        return "DEPLOYABLE"
    if value == "WEAK_POSITIVE":
        return "PROMISING"
    if value in {"NEUTRAL", "HURTS"}:
        return "NEUTRAL"
    return "INSUFFICIENT"


def phase2_verdict(inference: str, branch: str) -> str:
    if inference == "STATIC_STEERING_WORKS":
        return "INFERENCE_STEERING_POSSIBLE"
    if inference == "GRADIENT_ONLY":
        return "ADAPTER_REQUIRED"
    if inference in {"UNSIGNED_ONLY", "NO_RELIABLE_INFERENCE_STEERING", "DESTABILIZING"}:
        return "TRAINING_REQUIRED"
    if branch in {"DEPLOYABLE", "PROMISING"}:
        return "TRAINING_REQUIRED"
    return "INSUFFICIENT"


def recommended_next(inference: str, branch: str) -> str:
    if inference == "STATIC_STEERING_WORKS":
        return "expand_static_steering_and_prepare_v8.2_inference_controller"
    if inference == "GRADIENT_ONLY":
        return "train_causal_controller_adapter_to_approximate_local_gradients"
    if inference in {"UNSIGNED_ONLY", "NO_RELIABLE_INFERENCE_STEERING"}:
        if branch == "DEPLOYABLE":
            return "preserve_text_prefix_BG_branch_allocation_and_consolidate_phase1_readout_selector_plus_phase2_training_regime"
        return "consolidate_phase1_readout_selector_and_design_phase2_training_regime"
    if inference == "DESTABILIZING":
        return "consolidate_with_caveat_and_avoid_current_inference_time_steering_path"
    return "rerun_failed_probe_only_or_consolidate_with_caveat"


def interpretation(inference: str, branch: str, phase2: str) -> str:
    if inference == "STATIC_STEERING_WORKS":
        return "RMS-calibrated hidden-state steering produced a stable signed control signal, so inference-time steering remains an architecture candidate."
    if inference == "GRADIENT_ONLY":
        return "Static vectors still failed, but a local causal gradient exists; this points toward an adapter/controller rather than raw readout-vector steering."
    if branch == "DEPLOYABLE":
        return "Hidden-state steering remains unreliable under tested static directions, while text-prefix BG branch allocation is the deployable Phase 1.5 control path."
    if inference == "UNSIGNED_ONLY":
        return "The model can be nudged in BG-readable state space, but tested directions do not provide reliable signed control; Phase 2 training is required."
    if phase2 == "TRAINING_REQUIRED":
        return "The remaining inference-time steering probes did not produce reliable control, so BG should be consolidated as a readout/selector and Phase 2 should train propagation."
    return "The bundle was incomplete, so architecture consolidation should keep the caveats attached to the missing probes."


def write_reports(analysis: dict[str, Any]) -> None:
    summary = {
        "BG_PRECONSOLIDATION_PREFLIGHT_VERDICT": analysis["BG_PRECONSOLIDATION_PREFLIGHT_VERDICT"],
        "BG_RMS_STEERING_VERDICT": analysis["BG_RMS_STEERING_VERDICT"],
        "BG_RMS_VS_L2_VERDICT": analysis["BG_RMS_VS_L2_VERDICT"],
        "BG_RMS_STABILITY_VERDICT": analysis["BG_RMS_STABILITY_VERDICT"],
        "BG_PROPAGATION_VERDICT": analysis["BG_PROPAGATION_VERDICT"],
        "BG_PROPAGATION_DECAY_PROFILE": analysis["BG_PROPAGATION_DECAY_PROFILE"],
        "BG_LOGIT_EFFECT_VERDICT": analysis["BG_LOGIT_EFFECT_VERDICT"],
        "BG_TEXT_PREFIX_EXPANSION_VERDICT": analysis["BG_TEXT_PREFIX_EXPANSION_VERDICT"],
        "BG_CAUSAL_GRADIENT_VERDICT": analysis["BG_CAUSAL_GRADIENT_VERDICT"],
        "BG_INFERENCE_TIME_STEERING_VERDICT": analysis["BG_INFERENCE_TIME_STEERING_VERDICT"],
        "BG_BRANCH_ALLOCATION_VERDICT": analysis["BG_BRANCH_ALLOCATION_VERDICT"],
        "BG_PHASE2_REQUIREMENT_VERDICT": analysis["BG_PHASE2_REQUIREMENT_VERDICT"],
        "RECOMMENDED_NEXT": analysis["RECOMMENDED_NEXT"],
        "interpretation": analysis["interpretation"],
        "report_paths": analysis["report_paths"],
    }
    write_json(OUT_JSON, analysis)
    write_json(SUMMARY_JSON, summary)

    summary_lines = ["# BG Pre-Consolidation Control Probe Summary", ""]
    for key in [
        "BG_PRECONSOLIDATION_PREFLIGHT_VERDICT",
        "BG_RMS_STEERING_VERDICT",
        "BG_RMS_VS_L2_VERDICT",
        "BG_RMS_STABILITY_VERDICT",
        "BG_PROPAGATION_VERDICT",
        "BG_PROPAGATION_DECAY_PROFILE",
        "BG_LOGIT_EFFECT_VERDICT",
        "BG_TEXT_PREFIX_EXPANSION_VERDICT",
        "BG_CAUSAL_GRADIENT_VERDICT",
        "BG_INFERENCE_TIME_STEERING_VERDICT",
        "BG_BRANCH_ALLOCATION_VERDICT",
        "BG_PHASE2_REQUIREMENT_VERDICT",
        "RECOMMENDED_NEXT",
    ]:
        summary_lines.append(f"{key} = {summary.get(key)}")
    summary_lines.extend(
        [
            "",
            "## 1. Motivation",
            "",
            "This bundle tested the remaining inference-time control hypotheses before architecture consolidation: RMS-calibrated steering, propagation/decay, deployable text-prefix branch allocation, and a local causal-gradient diagnostic.",
            "",
            "## 2. RMS-Calibrated Steering",
            "",
            f"- verdict: `{analysis['BG_RMS_STEERING_VERDICT']}`",
            f"- RMS vs L2: `{analysis['BG_RMS_VS_L2_VERDICT']}`",
            f"- stability: `{analysis['BG_RMS_STABILITY_VERDICT']}`",
            f"- raw first-pass RMS verdict: `{analysis.get('BG_RMS_STEERING_VERDICT_RAW')}` / `{analysis.get('BG_RMS_STABILITY_VERDICT_RAW')}`",
            f"- row-level reinterpretation: `{analysis.get('RMS_ROW_LEVEL_REINTERPRETATION')}` with `{analysis.get('RMS_ROW_LEVEL_SAFETY_FAILURE_COUNT')}` safety row",
            f"- best RMS cell: `{analysis.get('rms_best_key')}`",
            f"- RMS gain over L2: `{analysis.get('RMS_GAIN_OVER_L2')}`",
            "",
            "## 3. Propagation/Decay Map",
            "",
            f"- propagation: `{analysis['BG_PROPAGATION_VERDICT']}`",
            f"- decay profile: `{analysis['BG_PROPAGATION_DECAY_PROFILE']}`",
            f"- logit effect: `{analysis['BG_LOGIT_EFFECT_VERDICT']}`",
            "",
            "## 4. Text-Prefix Branch Selection Expansion",
            "",
            f"- verdict: `{analysis['BG_TEXT_PREFIX_EXPANSION_VERDICT']}`",
            f"- branch allocation verdict: `{analysis['BG_BRANCH_ALLOCATION_VERDICT']}`",
            f"- BG top1 lift: `{analysis.get('text_bg_top1_lift')}`",
            f"- BG top2 lift: `{analysis.get('text_bg_top2_lift')}`",
            "",
            "## 5. Causal-Gradient Probe",
            "",
            f"- verdict: `{analysis['BG_CAUSAL_GRADIENT_VERDICT']}`",
            f"- positive z mean: `{analysis.get('gradient_positive_z_mean')}`",
            f"- negative z mean: `{analysis.get('gradient_negative_z_mean')}`",
            f"- random z mean: `{analysis.get('gradient_random_z_mean')}`",
            "",
            "## 6. Cross-Probe Interpretation",
            "",
            analysis["interpretation"],
            "",
            "## 7. Architecture Consolidation Implications",
            "",
            f"- inference-time steering: `{analysis['BG_INFERENCE_TIME_STEERING_VERDICT']}`",
            f"- branch allocation: `{analysis['BG_BRANCH_ALLOCATION_VERDICT']}`",
            f"- Phase 2 requirement: `{analysis['BG_PHASE2_REQUIREMENT_VERDICT']}`",
            "",
            "## 8. Docs Updated",
            "",
        ]
    )
    for path in analysis["docs_updated"]:
        summary_lines.append(f"- `{path}`")
    summary_lines.extend(
        [
            "",
            "## 9. Files Modified / Created",
            "",
        ]
    )
    for path in analysis["files_created"]:
        summary_lines.append(f"- `{path}`")
    summary_lines.extend(
        [
            "",
            "## 10. Commands Run",
            "",
        ]
    )
    for command in analysis["commands_run"]:
        summary_lines.append(f"- `{command}`")
    summary_lines.extend(
        [
            "",
            "## 11. Blockers",
            "",
            analysis.get("blockers") or "No hard blockers recorded by the final analyzer.",
        ]
    )
    write_md(SUMMARY_MD, summary_lines)

    doc_lines = [
        "# BG Pre-Consolidation Control Probes",
        "",
        "## Why These Probes Were Run",
        "",
        "The prior steering work validated BG as a readout and selection surface, but raw L2-normalized readout directions and empirical static directions did not yield reliable signed hidden-state control. This bundle tested the remaining bounded explanations before architecture consolidation.",
        "",
        "## RMS-Normalization Correction",
        "",
        "Earlier layer-hook steering used unit-L2 directions. In 2048 dimensions, that makes the direction RMS roughly 1/sqrt(2048), so alpha 0.02 was only a tiny hidden-RMS perturbation. The RMS-calibrated probe used RMS-unit directions so alpha maps directly to an approximate hidden-RMS fraction.",
        "",
        "## Propagation / Decay Findings",
        "",
        f"`{analysis['BG_PROPAGATION_VERDICT']}` with decay profile `{analysis['BG_PROPAGATION_DECAY_PROFILE']}` and logit verdict `{analysis['BG_LOGIT_EFFECT_VERDICT']}`.",
        "",
        "## Text-Prefix Branch Expansion",
        "",
        f"`{analysis['BG_TEXT_PREFIX_EXPANSION_VERDICT']}` over cached non-code text-prefix branch pools. This path remains ordinary text-prefix branch allocation, not hidden-state steering.",
        "",
        "## Causal-Gradient Result",
        "",
        f"`{analysis['BG_CAUSAL_GRADIENT_VERDICT']}`.",
        "",
        "## Final Architecture Implications",
        "",
        f"- BG inference-time steering: `{analysis['BG_INFERENCE_TIME_STEERING_VERDICT']}`",
        f"- BG branch allocation: `{analysis['BG_BRANCH_ALLOCATION_VERDICT']}`",
        f"- Phase 2 requirement: `{analysis['BG_PHASE2_REQUIREMENT_VERDICT']}`",
        "",
        analysis["interpretation"],
        "",
        "## Report Paths",
        "",
    ]
    for key, path in analysis["report_paths"].items():
        doc_lines.append(f"- {key}: `{path}`")
    write_md(DOC_MAIN, doc_lines)

    append_lines = [
        f"- BG_RMS_STEERING_VERDICT = `{analysis['BG_RMS_STEERING_VERDICT']}`",
        f"- BG_RMS_VS_L2_VERDICT = `{analysis['BG_RMS_VS_L2_VERDICT']}`",
        f"- BG_PROPAGATION_VERDICT = `{analysis['BG_PROPAGATION_VERDICT']}`",
        f"- BG_PROPAGATION_DECAY_PROFILE = `{analysis['BG_PROPAGATION_DECAY_PROFILE']}`",
        f"- BG_TEXT_PREFIX_EXPANSION_VERDICT = `{analysis['BG_TEXT_PREFIX_EXPANSION_VERDICT']}`",
        f"- BG_CAUSAL_GRADIENT_VERDICT = `{analysis['BG_CAUSAL_GRADIENT_VERDICT']}`",
        f"- BG_INFERENCE_TIME_STEERING_VERDICT = `{analysis['BG_INFERENCE_TIME_STEERING_VERDICT']}`",
        f"- BG_BRANCH_ALLOCATION_VERDICT = `{analysis['BG_BRANCH_ALLOCATION_VERDICT']}`",
        f"- BG_PHASE2_REQUIREMENT_VERDICT = `{analysis['BG_PHASE2_REQUIREMENT_VERDICT']}`",
        f"- interpretation: {analysis['interpretation']}",
        f"- reports: `{analysis['report_paths']['summary']}`, `{analysis['report_paths']['final_analysis']}`, `{analysis['report_paths']['docs']}`",
    ]
    for path in APPEND_DOCS:
        append_once(path, "BG pre-consolidation control probes (2026-05-18)", append_lines)


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    preflight = load_json(OUT_ROOT / "preflight.json", {})
    rms = load_json(OUT_ROOT / "rms_steering_analysis.json", {})
    rms_row = load_json(OUT_ROOT / "rms_row_level_analysis.json", {})
    rms = normalized_rms(rms, rms_row)
    prop = load_json(OUT_ROOT / "propagation_decay_analysis.json", {})
    text = load_json(OUT_ROOT / "text_prefix_expansion_analysis.json", {})
    gradient = load_json(OUT_ROOT / "causal_gradient_probe.json", {})
    layerhook = load_json(STAGE2_LAYERHOOK_ROOT / "summary.json", {})
    empirical = load_json(EMPIRICAL_ROOT / "summary.json", load_json(EMPIRICAL_ROOT / "analysis.json", {}))

    inference = steering_verdict(rms, gradient, empirical)
    branch = branch_verdict(text)
    phase2 = phase2_verdict(inference, branch)
    rec = recommended_next(inference, branch)
    interp = interpretation(inference, branch, phase2)
    text_metrics = text.get("metrics") or {}
    grad_summary = gradient.get("summary") or {}

    report_paths = {
        "preflight": rel(OUT_ROOT / "preflight.md"),
        "rms_steering": rel(OUT_ROOT / "rms_steering_analysis.md"),
        "rms_row_level": rel(OUT_ROOT / "rms_row_level_analysis.md"),
        "propagation_decay": rel(OUT_ROOT / "propagation_decay_analysis.md"),
        "text_prefix_expansion": rel(OUT_ROOT / "text_prefix_expansion_analysis.md"),
        "causal_gradient": rel(OUT_ROOT / "causal_gradient_probe.md"),
        "final_analysis": rel(OUT_MD),
        "summary": rel(SUMMARY_MD),
        "docs": rel(DOC_MAIN),
        "prior_layerhook": rel(STAGE2_LAYERHOOK_ROOT / "summary.md"),
        "prior_empirical": rel(EMPIRICAL_ROOT / "summary.md"),
    }
    files_created = [
        rel(OUT_ROOT / "preflight.json"),
        rel(OUT_ROOT / "rms_steering_traces.json"),
        rel(OUT_ROOT / "rms_steering_analysis.json"),
        rel(OUT_ROOT / "rms_row_level_analysis.json"),
        rel(OUT_ROOT / "propagation_decay_traces.json"),
        rel(OUT_ROOT / "propagation_decay_analysis.json"),
        rel(OUT_ROOT / "text_prefix_expansion_results.json"),
        rel(OUT_ROOT / "text_prefix_expansion_analysis.json"),
        rel(OUT_ROOT / "causal_gradient_probe.json"),
        rel(OUT_JSON),
        rel(SUMMARY_JSON),
        rel(DOC_MAIN),
    ]
    commands_run = [
        "venv/bin/python -m py_compile utilities/tests/manual/bg_preconsolidation_preflight.py",
        "venv/bin/python -m py_compile utilities/tests/manual/run_bg_rms_calibrated_steering_probe.py",
        "venv/bin/python -m py_compile utilities/tests/manual/analyze_bg_rms_calibrated_steering.py",
        "venv/bin/python -m py_compile utilities/tests/manual/run_bg_steering_propagation_decay_map.py",
        "venv/bin/python -m py_compile utilities/tests/manual/run_bg_text_prefix_expansion.py",
        "venv/bin/python -m py_compile utilities/tests/manual/run_bg_local_causal_gradient_probe.py",
        "venv/bin/python -m py_compile utilities/tests/manual/analyze_bg_preconsolidation_control_bundle.py",
        "venv/bin/python -u utilities/tests/manual/bg_preconsolidation_preflight.py",
        "venv/bin/python -u utilities/tests/manual/run_bg_rms_calibrated_steering_probe.py",
        "venv/bin/python -u utilities/tests/manual/analyze_bg_rms_calibrated_steering.py",
        "venv/bin/python -u utilities/tests/manual/run_bg_steering_propagation_decay_map.py",
        "venv/bin/python -u utilities/tests/manual/run_bg_text_prefix_expansion.py",
        "venv/bin/python -u utilities/tests/manual/run_bg_local_causal_gradient_probe.py",
        "venv/bin/python -u utilities/tests/manual/analyze_bg_preconsolidation_control_bundle.py",
    ]

    analysis = {
        "BG_PRECONSOLIDATION_PREFLIGHT_VERDICT": get(preflight, "BG_PRECONSOLIDATION_PREFLIGHT_VERDICT"),
        "BG_RMS_STEERING_VERDICT": get(rms, "BG_RMS_STEERING_VERDICT"),
        "BG_RMS_VS_L2_VERDICT": get(rms, "BG_RMS_VS_L2_VERDICT"),
        "BG_RMS_STABILITY_VERDICT": get(rms, "BG_RMS_STABILITY_VERDICT"),
        "BG_PROPAGATION_VERDICT": get(prop, "BG_PROPAGATION_VERDICT"),
        "BG_PROPAGATION_DECAY_PROFILE": get(prop, "BG_PROPAGATION_DECAY_PROFILE"),
        "BG_LOGIT_EFFECT_VERDICT": get(prop, "BG_LOGIT_EFFECT_VERDICT"),
        "BG_TEXT_PREFIX_EXPANSION_VERDICT": get(text, "BG_TEXT_PREFIX_EXPANSION_VERDICT"),
        "BG_CAUSAL_GRADIENT_VERDICT": get(gradient, "BG_CAUSAL_GRADIENT_VERDICT", "SKIPPED"),
        "BG_INFERENCE_TIME_STEERING_VERDICT": inference,
        "BG_BRANCH_ALLOCATION_VERDICT": branch,
        "BG_PHASE2_REQUIREMENT_VERDICT": phase2,
        "RECOMMENDED_NEXT": rec,
        "interpretation": interp,
        "rms_best_key": rms.get("best_rms_key"),
        "RMS_GAIN_OVER_L2": rms.get("RMS_GAIN_OVER_L2"),
        "BG_RMS_STEERING_VERDICT_RAW": rms.get("BG_RMS_STEERING_VERDICT_RAW"),
        "BG_RMS_STABILITY_VERDICT_RAW": rms.get("BG_RMS_STABILITY_VERDICT_RAW"),
        "RMS_ROW_LEVEL_REINTERPRETATION": rms.get("RMS_ROW_LEVEL_REINTERPRETATION"),
        "RMS_ROW_LEVEL_SAFETY_FAILURE_COUNT": rms.get("RMS_ROW_LEVEL_SAFETY_FAILURE_COUNT"),
        "text_bg_top1_lift": text_metrics.get("bg_top1_lift"),
        "text_bg_top2_lift": text_metrics.get("bg_top2_lift"),
        "text_pairwise_branch_ranking_accuracy": text_metrics.get("pairwise_branch_ranking_accuracy"),
        "gradient_positive_z_mean": grad_summary.get("positive_z_mean"),
        "gradient_negative_z_mean": grad_summary.get("negative_z_mean"),
        "gradient_random_z_mean": grad_summary.get("random_z_mean"),
        "prior_layerhook_verdict": layerhook.get("BG_LAYERHOOK_FOLLOWUP_VERDICT"),
        "prior_layerhook_signed_causal": layerhook.get("BG_LAYERHOOK_SIGNED_CAUSAL_VERDICT"),
        "prior_empirical_verdict": empirical.get("BG_EMPIRICAL_STEERING_VERDICT"),
        "prior_empirical_causal": empirical.get("BG_EMPIRICAL_DIRECTION_CAUSAL_VERDICT"),
        "report_paths": report_paths,
        "docs_updated": [rel(path) for path in [DOC_MAIN, *APPEND_DOCS]],
        "files_created": files_created,
        "commands_run": commands_run,
        "elapsed_seconds": round(time.time() - started, 3),
        "blockers": "",
    }
    blockers = []
    for name, payload in [("preflight", preflight), ("rms", rms), ("propagation", prop), ("text_prefix", text)]:
        if not payload:
            blockers.append(f"{name} output missing")
    if blockers:
        analysis["blockers"] = "; ".join(blockers)
    write_reports(analysis)
    print(f"BG_INFERENCE_TIME_STEERING_VERDICT = {inference}")
    print(f"BG_BRANCH_ALLOCATION_VERDICT = {branch}")
    print(f"BG_PHASE2_REQUIREMENT_VERDICT = {phase2}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(SUMMARY_MD)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
