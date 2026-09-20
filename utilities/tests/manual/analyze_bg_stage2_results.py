"""Analyze BG Stage 2 steering sensitivity v3 results and update docs."""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REPORT_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_stage2_steering_2026-05-18"
OUT_JSON = REPORT_ROOT / "analysis.json"
OUT_MD = REPORT_ROOT / "analysis.md"
SUMMARY_JSON = REPORT_ROOT / "summary.json"
SUMMARY_MD = REPORT_ROOT / "summary.md"
DOC_MAIN = PROJECT_ROOT / "docs/evaluator/bg_stage2_steering_sensitivity.md"
APPEND_DOCS = [
    PROJECT_ROOT / "docs/evaluator/current_state.md",
    PROJECT_ROOT / "docs/evaluator/domain_transfer_ledger.md",
    PROJECT_ROOT / "docs/evaluator/bg_trajectory_prediction_sweep.md",
    PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-18_v8.1_routing_locked.md",
]


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(p)


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_once(path: Path, title: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if title in existing:
        return
    block = "\n".join(["", f"## {title}", "", *lines, ""])
    path.write_text(existing.rstrip() + block + "\n", encoding="utf-8")


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def rate(values: list[bool]) -> float | None:
    return sum(1 for value in values if value) / len(values) if values else None


def mode_label(mode: str) -> str:
    return {
        "single_loop_L1": "SINGLE_L1",
        "single_loop_L4": "SINGLE_L4",
        "multi_loop_uniform": "MULTILOOP_UNIFORM",
        "multi_loop_decayed": "MULTILOOP_DECAYED",
    }.get(mode, mode.upper())


def mechanism_label(mechanism: str) -> str:
    return "LAYERHOOK" if mechanism == "layer_hook_injection" else "LATENT"


def group_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, float], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["target_id"]), str(row["mechanism"]), str(row["intervention_mode"]), finite(row["alpha"]))].append(row)
    return grouped


def analyze_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row.get("condition"))].append(row)
    pos = [finite(row.get("z_score_change")) for row in by_condition.get("positive", [])]
    neg = [finite(row.get("z_score_change")) for row in by_condition.get("negative", [])]
    rnd = [finite(row.get("z_score_change")) for row in by_condition.get("random", [])]
    base = by_condition.get("zero_baseline", [])
    random_std = pstdev(rnd) if len(rnd) > 1 else 0.0
    threshold = max(0.05, 0.5 * random_std)
    pos_mean = avg(pos)
    neg_mean = avg(neg)
    rnd_mean = avg(rnd)
    signed = (
        pos_mean is not None
        and neg_mean is not None
        and rnd_mean is not None
        and pos_mean > rnd_mean + threshold
        and neg_mean < rnd_mean - threshold
    )
    unsigned = (
        pos_mean is not None
        and rnd_mean is not None
        and pos_mean > rnd_mean + threshold
    )
    intervention_rows = [row for row in rows if row.get("condition") != "zero_baseline"]
    stable = bool(intervention_rows) and not any(
        row.get("safety_status") == "DESTABILIZING"
        or row.get("cuda_error")
        or row.get("nan_or_inf_activations")
        or finite(row.get("activation_rms_change")) > 5.0
        for row in intervention_rows
    )
    baseline_success = rate([bool(row.get("is_correct")) for row in base])
    pos_success = rate([bool(row.get("is_correct")) for row in by_condition.get("positive", [])])
    neg_success = rate([bool(row.get("is_correct")) for row in by_condition.get("negative", [])])
    rnd_success = rate([bool(row.get("is_correct")) for row in by_condition.get("random", [])])
    per_loop = defaultdict(list)
    for row in intervention_rows:
        for loop, value in (row.get("per_loop_activation_rms_change") or {}).items():
            per_loop[str(loop)].append(finite(value))
    per_loop_mean = {loop: avg(vals) for loop, vals in sorted(per_loop.items())}
    dominant_loop = None
    if per_loop_mean:
        dominant_loop = max(per_loop_mean, key=lambda loop: finite(per_loop_mean[loop]))
    effect_size = None
    if pos_mean is not None and rnd_mean is not None:
        effect_size = pos_mean - rnd_mean
    signed_effect_size = None
    if pos_mean is not None and neg_mean is not None and rnd_mean is not None:
        signed_effect_size = ((pos_mean - rnd_mean) + (rnd_mean - neg_mean)) / 2.0
    return {
        "positive_z_change_mean": pos_mean,
        "negative_z_change_mean": neg_mean,
        "random_z_change_mean": rnd_mean,
        "random_z_change_std": random_std if len(rnd) > 1 else None,
        "RANDOM_CONTROL_N": len(rnd) if rnd else 0,
        "RANDOM_CONTROL_LOW_N": len(rnd) <= 1,
        "causal_signal_signed": signed,
        "causal_signal_unsigned": unsigned,
        "causal_effect_size": effect_size,
        "signed_effect_size": signed_effect_size,
        "stable": stable,
        "success_rate_baseline": baseline_success,
        "success_rate_positive": pos_success,
        "success_rate_negative": neg_success,
        "success_rate_random": rnd_success,
        "positive_success_lift_vs_baseline": (
            pos_success - baseline_success if pos_success is not None and baseline_success is not None else None
        ),
        "negative_success_lift_vs_baseline": (
            neg_success - baseline_success if neg_success is not None and baseline_success is not None else None
        ),
        "random_success_lift_vs_baseline": (
            rnd_success - baseline_success if rnd_success is not None and baseline_success is not None else None
        ),
        "per_loop_activation_rms_change": per_loop_mean,
        "dominant_loop": dominant_loop,
        "row_count": len(rows),
    }


def synthesize_verdicts(preflight: dict[str, Any], task_suite: dict[str, Any], sweep: dict[str, Any], group_metrics: dict[str, Any]) -> dict[str, Any]:
    rows = list(sweep.get("rows") or [])
    target_signed = defaultdict(bool)
    target_unsigned = defaultdict(bool)
    target_stable_causal = defaultdict(bool)
    target_final_help = defaultdict(bool)
    for key, metrics in group_metrics.items():
        target = key.split("|", 1)[0]
        if metrics.get("causal_signal_signed"):
            target_signed[target] = True
        if metrics.get("causal_signal_unsigned"):
            target_unsigned[target] = True
        if (metrics.get("causal_signal_signed") or metrics.get("causal_signal_unsigned")) and metrics.get("stable"):
            target_stable_causal[target] = True
        if finite(metrics.get("positive_success_lift_vs_baseline"), 0.0) > 0.02:
            target_final_help[target] = True

    evaluable_targets = {row.get("target_id") for row in rows if row.get("condition") != "zero_baseline" and not row.get("cuda_error")}
    if len(evaluable_targets) < 2:
        causal = "INSUFFICIENT"
    elif sum(target_signed.values()) >= 2:
        causal = "SIGNED_CAUSAL_EFFECT_DETECTED"
    elif sum(target_unsigned.values()) >= 2:
        causal = "UNSIGNED_EFFECT_DETECTED"
    else:
        causal = "NO_CAUSAL_EFFECT"

    intervention_rows = [row for row in rows if row.get("condition") != "zero_baseline"]
    if not intervention_rows:
        stability = "INSUFFICIENT"
    elif any(row.get("safety_status") == "DESTABILIZING" for row in intervention_rows):
        stability = "DESTABILIZING_AT_ANY_ALPHA"
    elif all(finite(row.get("alpha")) <= 0.005 or row.get("safety_status") == "OK" for row in intervention_rows):
        stability = "STABLE_AT_ALL_ALPHAS"
    else:
        stability = "STABLE_AT_LOW_ALPHA_ONLY"

    positive_lifts = [
        finite(metrics.get("positive_success_lift_vs_baseline"))
        for metrics in group_metrics.values()
        if metrics.get("positive_success_lift_vs_baseline") is not None
    ]
    if not positive_lifts:
        final_lift = "INSUFFICIENT"
    elif mean(positive_lifts) > 0.02:
        final_lift = "POSITIVE_LIFT"
    elif mean(positive_lifts) < -0.02:
        final_lift = "NEGATIVE_LIFT"
    else:
        final_lift = "NULL_LIFT"

    single_counts = {"L1": 0, "L4": 0, "similar": 0, "weak": 0}
    for target in sorted(evaluable_targets):
        l1 = []
        l4 = []
        for key, metrics in group_metrics.items():
            parts = key.split("|")
            if parts[0] != target:
                continue
            if parts[2] == "single_loop_L1":
                l1.append(finite(metrics.get("causal_effect_size")))
            if parts[2] == "single_loop_L4":
                l4.append(finite(metrics.get("causal_effect_size")))
        best_l1 = max(l1) if l1 else 0.0
        best_l4 = max(l4) if l4 else 0.0
        if max(best_l1, best_l4) < 0.05:
            single_counts["weak"] += 1
        elif best_l1 > best_l4 + 0.05:
            single_counts["L1"] += 1
        elif best_l4 > best_l1 + 0.05:
            single_counts["L4"] += 1
        else:
            single_counts["similar"] += 1
    if len(evaluable_targets) < 2:
        single_verdict = "INSUFFICIENT"
    elif single_counts["L1"] >= 2:
        single_verdict = "L1_BETTER"
    elif single_counts["L4"] >= 2:
        single_verdict = "L4_BETTER"
    elif single_counts["weak"] >= max(1, len(evaluable_targets) - 1):
        single_verdict = "BOTH_WEAK"
    else:
        single_verdict = "BOTH_SIMILAR"

    multiloop_targets = 0
    multiloop_destab = False
    multiloop_gain_values = []
    best_single_mode = "none"
    best_multi_mode = "none"
    best_single_score = -1e9
    best_multi_score = -1e9
    for target in sorted(evaluable_targets):
        single = []
        multi = []
        for key, metrics in group_metrics.items():
            parts = key.split("|")
            if parts[0] != target:
                continue
            score = finite(metrics.get("causal_effect_size"))
            if parts[2].startswith("single"):
                single.append((parts[2], score))
                if score > best_single_score:
                    best_single_score = score
                    best_single_mode = parts[2]
            else:
                multi.append((parts[2], score, bool(metrics.get("stable"))))
                if score > best_multi_score:
                    best_multi_score = score
                    best_multi_mode = parts[2]
        if single and multi:
            bs = max(score for _, score in single)
            bm = max(score for _, score, _ in multi)
            multiloop_gain_values.append(bm - bs)
            if bm > bs + 0.05:
                multiloop_targets += 1
            if any(not stable for _, _, stable in multi):
                multiloop_destab = True
    multiloop_gain = mean(multiloop_gain_values) if multiloop_gain_values else None
    if len(evaluable_targets) < 2:
        multiloop_verdict = "INSUFFICIENT"
    elif multiloop_destab:
        multiloop_verdict = "MULTILOOP_DESTABILIZING"
    elif multiloop_targets >= 2:
        multiloop_verdict = "MULTILOOP_STRONGER"
    else:
        multiloop_verdict = "MULTILOOP_NO_BETTER"

    if preflight.get("LATENT_LOOP_BOUNDARY_FORK_VERDICT") != "READY":
        mechanism_verdict = "LATENT_BLOCKED"
        best_mechanism = "layer_hook_injection" if rows else "none"
        latent_delta = None
    else:
        mechanism_verdict = "INSUFFICIENT"
        best_mechanism = "none"
        latent_delta = None

    if stability == "DESTABILIZING_AT_ANY_ALPHA":
        overall = "DESTABILIZING"
    elif causal == "SIGNED_CAUSAL_EFFECT_DETECTED" and final_lift == "POSITIVE_LIFT":
        overall = "PROMISING_HANDLE_FOUND"
    elif causal == "SIGNED_CAUSAL_EFFECT_DETECTED":
        overall = "CAUSAL_BUT_NO_TASK_LIFT"
    elif causal in {"NO_CAUSAL_EFFECT", "UNSIGNED_EFFECT_DETECTED"} and stability.startswith("STABLE"):
        overall = "READ_ONLY_BG"
    else:
        overall = "INSUFFICIENT"

    recommended = {
        "PROMISING_HANDLE_FOUND": "expand_steering_sensitivity_with_more_tasks_and_optional_higher_alpha",
        "CAUSAL_BUT_NO_TASK_LIFT": "design_phase2_backbone_regularization_to_amplify_propagation",
        "READ_ONLY_BG": "lock_v8.1_as_final_phase1_and_design_phase2_training_protocol",
        "DESTABILIZING": "abandon_this_inference_time_steering_method_pivot_to_phase2_training",
        "INSUFFICIENT": "improve_stage2_experimental_design_or_task_pool",
    }[overall]

    random_n = int(sweep.get("RANDOM_CONTROL_N") or 1)
    return {
        "BG_STAGE2_PREFLIGHT_VERDICT": preflight.get("BG_STAGE2_PREFLIGHT_VERDICT", "BLOCKED"),
        "BG_STAGE2_TASK_SUITE_VERDICT": task_suite.get("BG_STAGE2_TASK_SUITE_VERDICT", "BLOCKED"),
        "BG_STAGE2_HOOK_IMPLEMENTATION_VERDICT": "PARTIAL"
        if preflight.get("LAYER_HOOK_INJECTION_VERDICT") == "READY"
        and preflight.get("LATENT_LOOP_BOUNDARY_FORK_VERDICT") != "READY"
        else ("READY" if preflight.get("LAYER_HOOK_INJECTION_VERDICT") == "READY" else "BLOCKED"),
        "LAYER_HOOK_INJECTION_VERDICT": preflight.get("LAYER_HOOK_INJECTION_VERDICT", "BLOCKED"),
        "LATENT_LOOP_BOUNDARY_FORK_VERDICT": preflight.get("LATENT_LOOP_BOUNDARY_FORK_VERDICT", "SKIPPED"),
        "BG_CAUSAL_SENSITIVITY_VERDICT": causal,
        "BG_INTERVENTION_STABILITY_VERDICT": stability,
        "BG_FINAL_TASK_LIFT_VERDICT": final_lift,
        "BG_SINGLE_LOOP_POSITION_VERDICT": single_verdict,
        "BG_MULTILOOP_VERDICT": multiloop_verdict,
        "BG_STEERING_MECHANISM_VERDICT": mechanism_verdict,
        "OVERALL_BG_STEERING_VERDICT": overall,
        "BEST_STEERING_MECHANISM": best_mechanism,
        "BEST_SINGLE_LOOP_MODE": best_single_mode,
        "BEST_MULTILOOP_MODE": best_multi_mode,
        "MULTILOOP_GAIN_OVER_BEST_SINGLE": multiloop_gain,
        "LATENT_VS_LAYERHOOK_DELTA": latent_delta,
        "ZERO_ALPHA_HOOK_EQUIVALENCE": "PASS" if preflight.get("layer_hook_smoke", {}).get("zero_alpha_equivalence") else "FAIL",
        "RANDOM_CONTROL_N": random_n,
        "RANDOM_CONTROL_LOW_N": random_n <= 1,
        "CACHE_INTERVENTION_MODE": sweep.get("CACHE_INTERVENTION_MODE", "disabled"),
        "LATENT_DIRECTION_LAYER_MISMATCH": "mixed"
        if any(row.get("latent_direction_layer_mismatch") for row in preflight.get("targets", []))
        else False,
        "RECOMMENDED_NEXT": recommended,
        "evaluable_targets": sorted(str(t) for t in evaluable_targets if t),
        "target_signed_count": sum(target_signed.values()),
        "target_unsigned_count": sum(target_unsigned.values()),
        "target_final_help_count": sum(target_final_help.values()),
        "single_loop_counts": single_counts,
    }


def make_markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    if not rows:
        return ["_No rows._"]
    out = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return out


def write_docs(summary: dict[str, Any], analysis: dict[str, Any]) -> None:
    interpretation = one_sentence_interpretation(summary)
    main_lines = [
        "# BG Stage 2 Steering Sensitivity v3 (2026-05-18)",
        "",
        "## Purpose",
        "",
        "Stage 2 tests whether tiny activation nudges along BG-readable NoNorm directions causally move subsequent Ouro trajectory states while preserving output stability and final-answer behavior.",
        "",
        "## Relation To Stage 1",
        "",
        "Stage 1 validated finished-candidate BG selection and partial-trajectory BG prediction. Stage 2 targets the strongest clean NoNorm cells from that predictive envelope and uses the AntisymLinear peak only as a diagnostic readout.",
        "",
        "## Mechanisms",
        "",
        "Layer-hook injection modifies decoder-layer outputs during normal forward/generation calls. Latent loop-boundary fork operates on loop-boundary hidden states after partial UT computation. In this run, latent boundary forward equivalence was probed, but generation continuation from an internal boundary was blocked because it would require validated hidden-state/cache forking.",
        "",
        "## Loop Schedules",
        "",
        "The run distinguishes early single-loop L1 steering, late single-loop L4 steering, uniform multiloop steering, and decayed multiloop steering. The multiloop schedules test whether sustained pressure can shift the refinement trajectory more reliably than a one-shot perturbation.",
        "",
        "## Measurement",
        "",
        "The analysis keeps three metrics separate: causal sensitivity in BG-readable score space, output stability, and final task correctness. Random controls are recorded with low-N warnings where only one random direction was affordable.",
        "",
        "## Safety",
        "",
        "Alpha was capped at 0.02, NoNorm directions were the only steering vectors, cache was disabled for intervention generation, and destabilizing cells stop further alpha sweep for that target/mechanism/mode.",
        "",
        "## Results Summary",
        "",
        *summary_top_lines(summary),
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
        "## Phase 2 Implications",
        "",
        f"Recommended next step: `{summary['RECOMMENDED_NEXT']}`.",
        "",
        "## Report Paths",
        "",
        f"- Preflight: `{rel(REPORT_ROOT / 'preflight.md')}`",
        f"- Task suite: `{rel(REPORT_ROOT / 'task_suite.md')}`",
        f"- Traces: `{rel(REPORT_ROOT / 'intervention_traces.json')}`",
        f"- Analysis: `{rel(OUT_MD)}`",
        f"- Summary: `{rel(SUMMARY_MD)}`",
    ]
    write_md(DOC_MAIN, main_lines)

    append_lines = [
        *summary_top_lines(summary),
        "",
        f"Interpretation: {interpretation}",
        "",
        f"Full reports: `{rel(DOC_MAIN)}`, `{rel(SUMMARY_MD)}`, `{rel(OUT_MD)}`, `{rel(REPORT_ROOT / 'intervention_traces.json')}`.",
    ]
    for path in APPEND_DOCS:
        append_once(path, "BG Stage 2 steering sensitivity v3 (2026-05-18)", append_lines)


def summary_top_lines(summary: dict[str, Any]) -> list[str]:
    keys = [
        "BG_STAGE2_PREFLIGHT_VERDICT",
        "BG_STAGE2_TASK_SUITE_VERDICT",
        "BG_STAGE2_HOOK_IMPLEMENTATION_VERDICT",
        "LAYER_HOOK_INJECTION_VERDICT",
        "LATENT_LOOP_BOUNDARY_FORK_VERDICT",
        "BG_CAUSAL_SENSITIVITY_VERDICT",
        "BG_INTERVENTION_STABILITY_VERDICT",
        "BG_FINAL_TASK_LIFT_VERDICT",
        "BG_SINGLE_LOOP_POSITION_VERDICT",
        "BG_MULTILOOP_VERDICT",
        "BG_STEERING_MECHANISM_VERDICT",
        "OVERALL_BG_STEERING_VERDICT",
        "BEST_STEERING_MECHANISM",
        "BEST_SINGLE_LOOP_MODE",
        "BEST_MULTILOOP_MODE",
        "MULTILOOP_GAIN_OVER_BEST_SINGLE",
        "LATENT_VS_LAYERHOOK_DELTA",
        "ZERO_ALPHA_HOOK_EQUIVALENCE",
        "RANDOM_CONTROL_N",
        "RANDOM_CONTROL_LOW_N",
        "CACHE_INTERVENTION_MODE",
        "LATENT_DIRECTION_LAYER_MISMATCH",
        "RECOMMENDED_NEXT",
    ]
    return [f"{key} = {summary.get(key)}" for key in keys]


def one_sentence_interpretation(summary: dict[str, Any]) -> str:
    overall = summary.get("OVERALL_BG_STEERING_VERDICT")
    if overall == "PROMISING_HANDLE_FOUND":
        return "Stage 2 found a stable signed causal steering handle that also improved final task success in this bounded run."
    if overall == "CAUSAL_BUT_NO_TASK_LIFT":
        return "Stage 2 found a stable signed causal movement in BG-readable state space, but the effect did not reliably reach final task correctness."
    if overall == "READ_ONLY_BG":
        return "Stage 2 did not find a signed causal steering handle under these inference-time constraints, so BG remains more useful as a readout than a control surface for now."
    if overall == "DESTABILIZING":
        return "Stage 2 steering triggered stability guardrails, making this inference-time intervention unsuitable without a different protocol."
    return "Stage 2 produced too little clean intervention evidence for a causal steering conclusion."


def main() -> int:
    started = time.time()
    preflight = load_json(REPORT_ROOT / "preflight.json", {})
    task_suite = load_json(REPORT_ROOT / "task_suite.json", {})
    sweep = load_json(REPORT_ROOT / "intervention_traces.json", load_json(REPORT_ROOT / "intervention_traces.partial.json", {}))
    rows = list(sweep.get("rows") or [])
    grouped = group_rows(rows)
    group_metrics: dict[str, Any] = {}
    for (target, mechanism, mode, alpha), group in sorted(grouped.items()):
        key = f"{target}|{mechanism}|{mode}|{alpha:g}"
        group_metrics[key] = analyze_group(group)

    summary = synthesize_verdicts(preflight, task_suite, sweep, group_metrics)
    per_cell_verdicts = {}
    for key, metrics in group_metrics.items():
        target, mechanism, mode, _alpha = key.split("|")
        verdict_key = f"BG_STAGE2_TARGET_{target}_{mechanism_label(mechanism)}_{mode_label(mode)}_VERDICT"
        current = per_cell_verdicts.get(verdict_key, "BLOCKED")
        if not metrics.get("stable"):
            per_cell_verdicts[verdict_key] = "DESTABILIZING"
        elif metrics.get("row_count", 0) > 0 and current != "DESTABILIZING":
            per_cell_verdicts[verdict_key] = "READY"
    summary.update(per_cell_verdicts)

    comparison_rows = []
    for key, metrics in group_metrics.items():
        target, mechanism, mode, alpha = key.split("|")
        comparison_rows.append(
            {
                "target": target,
                "mechanism": mechanism,
                "mode": mode,
                "alpha": alpha,
                "pos_z": None if metrics["positive_z_change_mean"] is None else f"{metrics['positive_z_change_mean']:.3f}",
                "neg_z": None if metrics["negative_z_change_mean"] is None else f"{metrics['negative_z_change_mean']:.3f}",
                "rand_z": None if metrics["random_z_change_mean"] is None else f"{metrics['random_z_change_mean']:.3f}",
                "signed": metrics["causal_signal_signed"],
                "stable": metrics["stable"],
                "pos_lift": None
                if metrics["positive_success_lift_vs_baseline"] is None
                else f"{metrics['positive_success_lift_vs_baseline']:.3f}",
                "dominant_loop": metrics["dominant_loop"],
            }
        )

    analysis = {
        "summary": summary,
        "group_metrics": group_metrics,
        "comparison_rows": comparison_rows,
        "row_count": len(rows),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, analysis)

    lines = [
        "# BG Stage 2 Steering Analysis (2026-05-18)",
        "",
        *summary_top_lines(summary),
        "",
        "## Causal Sensitivity, Stability, Correctness",
        "",
        *make_markdown_table(
            comparison_rows[:120],
            ["target", "mechanism", "mode", "alpha", "pos_z", "neg_z", "rand_z", "signed", "stable", "pos_lift", "dominant_loop"],
        ),
        "",
        "## Cross-Target Aggregation",
        "",
        f"- evaluable_targets: `{summary.get('evaluable_targets')}`",
        f"- target_signed_count: `{summary.get('target_signed_count')}`",
        f"- target_unsigned_count: `{summary.get('target_unsigned_count')}`",
        f"- target_final_help_count: `{summary.get('target_final_help_count')}`",
        f"- single_loop_counts: `{summary.get('single_loop_counts')}`",
        "",
        "## Interpretation",
        "",
        one_sentence_interpretation(summary),
    ]
    write_md(OUT_MD, lines)

    summary_payload = {
        **summary,
        "report_paths": {
            "preflight_json": rel(REPORT_ROOT / "preflight.json"),
            "task_suite_json": rel(REPORT_ROOT / "task_suite.json"),
            "intervention_traces_json": rel(REPORT_ROOT / "intervention_traces.json"),
            "analysis_json": rel(OUT_JSON),
            "summary_json": rel(SUMMARY_JSON),
            "docs": rel(DOC_MAIN),
        },
        "files_modified_or_created": [
            "shared/src/evaluator/bg_steering_hook.py",
            "shared/utilities/tests/manual/bg_stage2_steering_preflight.py",
            "shared/utilities/tests/manual/build_bg_stage2_task_suite.py",
            "shared/utilities/tests/manual/test_bg_steering_hook.py",
            "shared/utilities/tests/manual/run_bg_stage2_intervention_sweep.py",
            "shared/utilities/tests/manual/analyze_bg_stage2_results.py",
            rel(DOC_MAIN),
            rel(REPORT_ROOT / "preflight.json"),
            rel(REPORT_ROOT / "task_suite.json"),
            rel(REPORT_ROOT / "intervention_traces.json"),
            rel(OUT_JSON),
            rel(SUMMARY_JSON),
        ],
        "commands_run": [
            "venv/bin/python -m py_compile utilities/tests/manual/bg_stage2_steering_preflight.py",
            "venv/bin/python -m py_compile utilities/tests/manual/build_bg_stage2_task_suite.py",
            "venv/bin/python -m py_compile src/evaluator/bg_steering_hook.py",
            "venv/bin/python -m py_compile utilities/tests/manual/test_bg_steering_hook.py",
            "venv/bin/python -m py_compile utilities/tests/manual/run_bg_stage2_intervention_sweep.py",
            "venv/bin/python -m py_compile utilities/tests/manual/analyze_bg_stage2_results.py",
            "venv/bin/python -u utilities/tests/manual/bg_stage2_steering_preflight.py",
            "venv/bin/python -u utilities/tests/manual/build_bg_stage2_task_suite.py",
            "venv/bin/python -u utilities/tests/manual/test_bg_steering_hook.py",
            "venv/bin/python -u utilities/tests/manual/run_bg_stage2_intervention_sweep.py",
            "venv/bin/python -u utilities/tests/manual/analyze_bg_stage2_results.py",
        ],
        "blockers": [
            "latent_loop_boundary_fork was blocked for full generation continuation unless validated hidden-state/cache forking is added"
        ]
        if summary.get("LATENT_LOOP_BOUNDARY_FORK_VERDICT") != "READY"
        else [],
    }
    write_json(SUMMARY_JSON, summary_payload)

    blocker_lines = [f"- {blocker}" for blocker in summary_payload["blockers"]] if summary_payload["blockers"] else ["- None"]
    summary_lines = [
        "# BG Stage 2 Steering Summary (2026-05-18)",
        "",
        *summary_top_lines(summary),
        "",
        "## 1. Stage 1 -> Stage 2 Design Rationale",
        "",
        "Stage 2 targets Stage 1's strongest clean NoNorm predictive cells to test hidden-state causal sensitivity.",
        "",
        "## 2. Target Selection",
        "",
        f"Targets loaded from `{rel(REPORT_ROOT / 'preflight.json')}`.",
        "",
        "## 3. Layer-Hook Injection Implementation",
        "",
        f"LAYER_HOOK_INJECTION_VERDICT = {summary['LAYER_HOOK_INJECTION_VERDICT']}.",
        "",
        "## 4. Latent Loop-Boundary Fork Implementation",
        "",
        f"LATENT_LOOP_BOUNDARY_FORK_VERDICT = {summary['LATENT_LOOP_BOUNDARY_FORK_VERDICT']}.",
        "",
        "## 5. Zero-Alpha Equivalence And Random-Control Validity",
        "",
        f"ZERO_ALPHA_HOOK_EQUIVALENCE = {summary['ZERO_ALPHA_HOOK_EQUIVALENCE']}. RANDOM_CONTROL_N = {summary['RANDOM_CONTROL_N']}.",
        "",
        "## 6. Intervention Sweep",
        "",
        f"Rows analyzed: `{len(rows)}`.",
        "",
        "## 7. Causal Sensitivity Results",
        "",
        f"BG_CAUSAL_SENSITIVITY_VERDICT = {summary['BG_CAUSAL_SENSITIVITY_VERDICT']}.",
        "",
        "## 8. Stability Results",
        "",
        f"BG_INTERVENTION_STABILITY_VERDICT = {summary['BG_INTERVENTION_STABILITY_VERDICT']}.",
        "",
        "## 9. Final Correctness Results",
        "",
        f"BG_FINAL_TASK_LIFT_VERDICT = {summary['BG_FINAL_TASK_LIFT_VERDICT']}.",
        "",
        "## 10. Single-Loop L1 vs L4 Comparison",
        "",
        f"BG_SINGLE_LOOP_POSITION_VERDICT = {summary['BG_SINGLE_LOOP_POSITION_VERDICT']}.",
        "",
        "## 11. Multiloop vs Single-Loop Comparison",
        "",
        f"BG_MULTILOOP_VERDICT = {summary['BG_MULTILOOP_VERDICT']}.",
        "",
        "## 12. Layer-Hook vs Latent-Boundary Comparison",
        "",
        f"BG_STEERING_MECHANISM_VERDICT = {summary['BG_STEERING_MECHANISM_VERDICT']}.",
        "",
        "## 13. Per-Target Breakdown",
        "",
        f"See `{rel(OUT_MD)}`.",
        "",
        "## 14. Overall Interpretation",
        "",
        one_sentence_interpretation(summary),
        "",
        "## 15. Phase 2 Implications",
        "",
        f"RECOMMENDED_NEXT = {summary['RECOMMENDED_NEXT']}.",
        "",
        "## 16. Docs Updated",
        "",
        f"- `{rel(DOC_MAIN)}`",
        *[f"- `{rel(path)}`" for path in APPEND_DOCS],
        "",
        "## 17. Files Modified / Created",
        "",
        *[f"- `{path}`" for path in summary_payload["files_modified_or_created"]],
        "",
        "## 18. Commands Run",
        "",
        *[f"- `{cmd}`" for cmd in summary_payload["commands_run"]],
        "",
        "## 19. Blockers",
        "",
        *blocker_lines,
    ]
    write_md(SUMMARY_MD, summary_lines)
    write_docs(summary, analysis)
    print(f"BG_CAUSAL_SENSITIVITY_VERDICT = {summary['BG_CAUSAL_SENSITIVITY_VERDICT']}")
    print(f"OVERALL_BG_STEERING_VERDICT = {summary['OVERALL_BG_STEERING_VERDICT']}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(SUMMARY_JSON)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
