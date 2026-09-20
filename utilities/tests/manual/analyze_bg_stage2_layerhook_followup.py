"""Analyze narrowed BG Stage 2 layer-hook follow-up results and update docs."""
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


OUT_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_stage2_layerhook_followup_2026-05-18"
PREV_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_stage2_steering_2026-05-18"
OUT_JSON = OUT_ROOT / "analysis.json"
OUT_MD = OUT_ROOT / "analysis.md"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
DOC_MAIN = PROJECT_ROOT / "docs/evaluator/bg_stage2_layerhook_followup.md"
APPEND_DOCS = [
    PROJECT_ROOT / "docs/evaluator/current_state.md",
    PROJECT_ROOT / "docs/evaluator/domain_transfer_ledger.md",
    PROJECT_ROOT / "docs/evaluator/bg_trajectory_prediction_sweep.md",
    PROJECT_ROOT / "docs/evaluator/bg_stage2_steering_sensitivity.md",
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
    path.write_text(existing.rstrip() + "\n\n## " + title + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


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


def normalize_row(row: dict[str, Any], source: str) -> dict[str, Any]:
    out = dict(row)
    out["analysis_source"] = source
    out["mode"] = out.get("mode") or out.get("intervention_mode")
    out["target_head_score_baseline"] = out.get("target_head_score_baseline", out.get("baseline_score"))
    out["target_head_score_post"] = out.get("target_head_score_post", out.get("post_intervention_score"))
    out["random_control_n"] = out.get("random_control_n", out.get("random_control_n_for_target", 1))
    out["task_subset_index"] = out.get("task_subset_index", out.get("task_suite_index"))
    out["parse_failed"] = bool(out.get("parse_failed", finite(out.get("parse_rate"), 0.0) < 1.0))
    return out


def compatible_previous_rows(preflight: dict[str, Any]) -> list[dict[str, Any]]:
    target = preflight.get("target") or {}
    prev = load_json(PREV_ROOT / "intervention_traces.partial.json", {})
    rows = []
    for row in prev.get("rows") or []:
        if row.get("target_id") != "T1":
            continue
        if row.get("mechanism") != "layer_hook_injection":
            continue
        if row.get("cache_intervention_mode") != "disabled":
            continue
        if row.get("condition") != "zero_baseline" and row.get("direction_source_head_id") != target.get("head_id"):
            continue
        rows.append(normalize_row(row, "previous_partial"))
    return rows


def summarize_mode_alpha(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cond: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cond[str(row.get("condition"))].append(row)
    pos = [finite(row.get("z_score_change")) for row in by_cond.get("positive", [])]
    neg = [finite(row.get("z_score_change")) for row in by_cond.get("negative", [])]
    rnd = [finite(row.get("z_score_change")) for row in by_cond.get("random", [])]
    base = by_cond.get("zero_baseline", [])
    pos_mean = avg(pos)
    neg_mean = avg(neg)
    rnd_mean = avg(rnd)
    rnd_std = pstdev(rnd) if len(rnd) > 1 else None
    threshold = 0.0 if rnd_std is None else 0.5 * rnd_std
    signed = (
        pos_mean is not None
        and neg_mean is not None
        and rnd_mean is not None
        and pos_mean > rnd_mean
        and neg_mean < rnd_mean
    )
    strong_signed = (
        pos_mean is not None
        and neg_mean is not None
        and rnd_mean is not None
        and rnd_std is not None
        and pos_mean >= rnd_mean + threshold
        and neg_mean <= rnd_mean - threshold
    )
    unsigned = pos_mean is not None and rnd_mean is not None and pos_mean > rnd_mean
    baseline_success = rate([bool(row.get("is_correct")) for row in base])
    pos_success = rate([bool(row.get("is_correct")) for row in by_cond.get("positive", [])])
    neg_success = rate([bool(row.get("is_correct")) for row in by_cond.get("negative", [])])
    rnd_success = rate([bool(row.get("is_correct")) for row in by_cond.get("random", [])])
    intervention = [row for row in rows if row.get("condition") != "zero_baseline"]
    stable = bool(intervention) and not any(
        row.get("safety_status") == "DESTABILIZING"
        or row.get("cuda_error")
        or row.get("nan_or_inf_activations")
        for row in intervention
    )
    rms_vals = [finite(row.get("activation_rms_change")) for row in intervention]
    return {
        "row_count": len(rows),
        "positive_z_mean": pos_mean,
        "negative_z_mean": neg_mean,
        "random_z_mean": rnd_mean,
        "random_z_std": rnd_std,
        "signed_causal_signature": signed,
        "strong_signed_causal_signature": strong_signed,
        "unsigned_effect": unsigned,
        "effect_size_pos_minus_random": None if pos_mean is None or rnd_mean is None else pos_mean - rnd_mean,
        "effect_size_random_minus_neg": None if neg_mean is None or rnd_mean is None else rnd_mean - neg_mean,
        "stable": stable,
        "rms_change_mean": avg(rms_vals),
        "parse_failed_rate": rate([bool(row.get("parse_failed")) for row in intervention]),
        "empty_output_rate": rate([bool(row.get("empty_output")) for row in intervention]),
        "hit_max_tokens_rate": rate([bool(row.get("hit_max_tokens")) for row in intervention]),
        "repetition_rate_mean": avg([finite(row.get("repetition_rate")) for row in intervention]),
        "output_length_mean": avg([finite(row.get("output_length")) for row in intervention]),
        "success_rate_baseline": baseline_success,
        "success_rate_positive": pos_success,
        "success_rate_negative": neg_success,
        "success_rate_random": rnd_success,
        "positive_success_lift_vs_baseline": (
            pos_success - baseline_success if pos_success is not None and baseline_success is not None else None
        ),
    }


def analyze_rows(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("mode")), finite(row.get("alpha")))].append(row)
    cells = {
        f"{mode}|{alpha:g}": summarize_mode_alpha(group)
        for (mode, alpha), group in sorted(grouped.items())
    }
    intervention_rows = [row for row in rows if row.get("condition") != "zero_baseline"]
    mechanical_ready = (
        bool(intervention_rows)
        and all(str(row.get("hook_loop_index_source")) == "current_ut" for row in intervention_rows)
        and all(str(row.get("cache_intervention_mode")) == "disabled" for row in intervention_rows)
        and not any(row.get("cuda_error") or row.get("nan_or_inf_activations") for row in intervention_rows)
    )
    signed_cells = [key for key, metrics in cells.items() if metrics.get("signed_causal_signature")]
    strong_signed_cells = [key for key, metrics in cells.items() if metrics.get("strong_signed_causal_signature")]
    unsigned_cells = [key for key, metrics in cells.items() if metrics.get("unsigned_effect")]
    stable = bool(intervention_rows) and not any(row.get("safety_status") == "DESTABILIZING" for row in intervention_rows)
    return {
        "label": label,
        "row_count": len(rows),
        "intervention_row_count": len(intervention_rows),
        "task_count": len({row.get("task_subset_index") for row in rows}),
        "mechanical_ready": mechanical_ready,
        "stable": stable,
        "cells": cells,
        "signed_cells": signed_cells,
        "strong_signed_cells": strong_signed_cells,
        "unsigned_cells": unsigned_cells,
        "cuda_error_count": sum(1 for row in intervention_rows if row.get("cuda_error")),
        "nan_or_inf_count": sum(1 for row in intervention_rows if row.get("nan_or_inf_activations")),
        "rms_change_by_alpha": {
            str(alpha): avg([finite(row.get("activation_rms_change")) for row in intervention_rows if finite(row.get("alpha")) == alpha])
            for alpha in sorted({finite(row.get("alpha")) for row in intervention_rows})
        },
    }


def best_mode_delta(cells: dict[str, Any], modes: list[str]) -> tuple[str, float]:
    best_mode = "none"
    best_score = -1e9
    for key, metrics in cells.items():
        mode, _alpha = key.split("|")
        if mode not in modes:
            continue
        score = finite(metrics.get("effect_size_pos_minus_random")) + finite(metrics.get("effect_size_random_minus_neg"))
        if score > best_score:
            best_score = score
            best_mode = mode
    return best_mode, best_score


def verdicts(audit: dict[str, Any], preflight: dict[str, Any], tasks: dict[str, Any], sweep: dict[str, Any], follow: dict[str, Any]) -> dict[str, Any]:
    cells = follow["cells"]
    mechanical = "READY" if follow["mechanical_ready"] else ("PARTIAL" if follow["intervention_row_count"] else "BLOCKED")
    if follow["intervention_row_count"] < 12:
        causal = "INSUFFICIENT"
    elif follow["strong_signed_cells"]:
        causal = "SIGNED_CAUSAL_EFFECT"
    elif follow["unsigned_cells"]:
        causal = "UNSIGNED_EFFECT"
    else:
        causal = "NO_RELIABLE_EFFECT"
    if not follow["intervention_row_count"]:
        stability = "INSUFFICIENT"
    elif not follow["stable"]:
        stability = "DESTABILIZING"
    else:
        rms_max = max((finite(v) for v in follow.get("rms_change_by_alpha", {}).values() if v is not None), default=0.0)
        stability = "STABLE_BUT_TINY" if rms_max < 0.001 else "STABLE"
    single_l1, score_l1 = best_mode_delta(cells, ["single_loop_L1"])
    single_l4, score_l4 = best_mode_delta(cells, ["single_loop_L4"])
    if follow["intervention_row_count"] < 12:
        single_verdict = "INSUFFICIENT"
    elif max(score_l1, score_l4) <= 0.05:
        single_verdict = "BOTH_WEAK"
    elif score_l1 > score_l4 + 0.05:
        single_verdict = "L1_BETTER"
    elif score_l4 > score_l1 + 0.05:
        single_verdict = "L4_BETTER"
    else:
        single_verdict = "BOTH_SIMILAR"
    best_single, best_single_score = best_mode_delta(cells, ["single_loop_L1", "single_loop_L4"])
    best_multi, best_multi_score = best_mode_delta(cells, ["multi_loop_uniform", "multi_loop_decayed"])
    multiloop_gain = None if best_multi == "none" or best_single == "none" else best_multi_score - best_single_score
    if follow["intervention_row_count"] < 12:
        multi_verdict = "INSUFFICIENT"
    elif not follow["stable"]:
        multi_verdict = "MULTILOOP_DESTABILIZING"
    elif multiloop_gain is not None and multiloop_gain > 0.05:
        multi_verdict = "MULTILOOP_STRONGER"
    else:
        multi_verdict = "MULTILOOP_NO_BETTER"
    lifts = [
        finite(metrics.get("positive_success_lift_vs_baseline"))
        for metrics in cells.values()
        if metrics.get("positive_success_lift_vs_baseline") is not None
    ]
    if not lifts:
        final_lift = "INSUFFICIENT"
    elif mean(lifts) > 0.02:
        final_lift = "POSITIVE_LIFT"
    elif mean(lifts) < -0.02:
        final_lift = "NEGATIVE_LIFT"
    else:
        final_lift = "NULL_LIFT"
    if stability == "DESTABILIZING":
        overall = "DESTABILIZING"
    elif causal == "SIGNED_CAUSAL_EFFECT" and final_lift == "POSITIVE_LIFT":
        overall = "PROMISING_HANDLE_FOUND"
    elif causal == "SIGNED_CAUSAL_EFFECT":
        overall = "CAUSAL_BUT_NO_TASK_LIFT"
    elif mechanical == "READY" and stability in {"STABLE", "STABLE_BUT_TINY"} and causal == "NO_RELIABLE_EFFECT":
        overall = "MECHANICAL_ONLY"
    elif causal in {"NO_RELIABLE_EFFECT", "UNSIGNED_EFFECT"}:
        overall = "READ_ONLY_BG_FOR_NOW"
    else:
        overall = "INSUFFICIENT"
    recommended = {
        "PROMISING_HANDLE_FOUND": "expand_layerhook_steering_to_more_targets_and_consider_alpha_sweep_extension",
        "CAUSAL_BUT_NO_TASK_LIFT": "design_phase2_regularization_to_amplify_causal_handle",
        "MECHANICAL_ONLY": "keep_BG_as_readout_selector_and_revisit_steering_with_empirical_success_direction_or_training",
        "READ_ONLY_BG_FOR_NOW": "keep_BG_as_readout_selector_and_revisit_steering_with_empirical_success_direction_or_training",
        "DESTABILIZING": "abandon_current_inference_time_layerhook_steering_method",
        "INSUFFICIENT": "rerun_smaller_or_adjust_runtime_budget",
    }[overall]
    return {
        "BG_STAGE2_PARTIAL_TRACE_AUDIT_VERDICT": audit.get("BG_STAGE2_PARTIAL_TRACE_AUDIT_VERDICT"),
        "BG_LAYERHOOK_FOLLOWUP_PREFLIGHT_VERDICT": preflight.get("BG_LAYERHOOK_FOLLOWUP_PREFLIGHT_VERDICT"),
        "BG_LAYERHOOK_FOLLOWUP_TASKS_VERDICT": tasks.get("BG_LAYERHOOK_FOLLOWUP_TASKS_VERDICT"),
        "BG_LAYERHOOK_FOLLOWUP_SWEEP_VERDICT": sweep.get("BG_LAYERHOOK_FOLLOWUP_SWEEP_VERDICT"),
        "BG_LAYERHOOK_MECHANICAL_VERDICT": mechanical,
        "BG_LAYERHOOK_SIGNED_CAUSAL_VERDICT": causal,
        "BG_SINGLE_LOOP_POSITION_VERDICT": single_verdict,
        "BG_MULTILOOP_VERDICT": multi_verdict,
        "BG_LAYERHOOK_STABILITY_VERDICT": stability,
        "BG_FINAL_TASK_LIFT_VERDICT": final_lift,
        "BG_LAYERHOOK_FOLLOWUP_VERDICT": overall,
        "BEST_SINGLE_LOOP_MODE": best_single,
        "BEST_MULTILOOP_MODE": best_multi,
        "MULTILOOP_GAIN_OVER_BEST_SINGLE": multiloop_gain,
        "RECOMMENDED_NEXT": recommended,
    }


def top_lines(summary: dict[str, Any]) -> list[str]:
    keys = [
        "BG_STAGE2_PARTIAL_TRACE_AUDIT_VERDICT",
        "BG_LAYERHOOK_FOLLOWUP_PREFLIGHT_VERDICT",
        "BG_LAYERHOOK_FOLLOWUP_TASKS_VERDICT",
        "BG_LAYERHOOK_FOLLOWUP_SWEEP_VERDICT",
        "BG_LAYERHOOK_MECHANICAL_VERDICT",
        "BG_LAYERHOOK_SIGNED_CAUSAL_VERDICT",
        "BG_SINGLE_LOOP_POSITION_VERDICT",
        "BG_MULTILOOP_VERDICT",
        "BG_LAYERHOOK_STABILITY_VERDICT",
        "BG_FINAL_TASK_LIFT_VERDICT",
        "BG_LAYERHOOK_FOLLOWUP_VERDICT",
        "BEST_SINGLE_LOOP_MODE",
        "BEST_MULTILOOP_MODE",
        "MULTILOOP_GAIN_OVER_BEST_SINGLE",
        "RECOMMENDED_NEXT",
    ]
    return [f"{key} = {summary.get(key)}" for key in keys]


def interpretation(summary: dict[str, Any]) -> str:
    overall = summary.get("BG_LAYERHOOK_FOLLOWUP_VERDICT")
    if overall == "PROMISING_HANDLE_FOUND":
        return "The narrowed layer-hook test found a stable signed BG-readable causal handle with positive final-task lift."
    if overall == "CAUSAL_BUT_NO_TASK_LIFT":
        return "The narrowed layer-hook test found stable signed BG-readable movement, but it did not translate into final-task lift."
    if overall == "MECHANICAL_ONLY":
        return "The layer-hook mechanism is valid and stable, but this follow-up did not show reliable signed causal steering."
    if overall == "READ_ONLY_BG_FOR_NOW":
        return "BG remains more reliable as a readout selector than as an inference-time steering vector under this protocol."
    if overall == "DESTABILIZING":
        return "The narrowed layer-hook protocol destabilized generation and should not be extended as-is."
    return "The narrowed follow-up produced too little usable data for a firm layer-hook steering verdict."


def table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    if not rows:
        return ["_No rows._"]
    out = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return out


def write_docs(summary: dict[str, Any]) -> None:
    interp = interpretation(summary)
    main = [
        "# BG Stage 2 Layer-Hook Follow-Up (2026-05-18)",
        "",
        "## Why This Follow-Up Was Needed",
        "",
        "The broad Stage 2 v3 sweep validated layer-hook mechanics but was too large for a clean completion under no-cache generation. This narrowed follow-up isolates T1 reasoning@64 and asks whether the already validated layer-hook surface produces reliable signed BG-readable movement.",
        "",
        "## Latent Boundary Fork Status",
        "",
        "The latent boundary fork remains blocked for full generation continuation because the local model does not expose a validated API for resuming autoregressive generation from a copied post-loop hidden boundary without cache/state forking.",
        "",
        "## Layer-Hook Validity",
        "",
        "Layer-hook injection is transformer-native, uses decoder-layer forward hooks, identifies loop position through `current_ut`, runs with `use_cache=False`, and preserves zero-alpha equivalence.",
        "",
        "## Target And Metrics",
        "",
        "The follow-up targets T1 reasoning@64 with the best NoNorm Stage 1 cell, keeps AntisymLinear as diagnostic readout only, and separates causal sensitivity, stability, and final correctness.",
        "",
        "## Results",
        "",
        *top_lines(summary),
        "",
        "## Interpretation",
        "",
        interp,
        "",
        "## Recommended Next Step",
        "",
        f"`{summary['RECOMMENDED_NEXT']}`",
    ]
    write_md(DOC_MAIN, main)
    append_lines = [
        *top_lines(summary),
        "",
        f"Interpretation: {interp}",
        "",
        f"Full reports: `{rel(DOC_MAIN)}`, `{rel(SUMMARY_MD)}`, `{rel(OUT_MD)}`.",
    ]
    for path in APPEND_DOCS:
        append_once(path, "BG Stage 2 layer-hook follow-up (2026-05-18)", append_lines)


def main() -> int:
    started = time.time()
    audit = load_json(OUT_ROOT / "partial_trace_audit.json", {})
    preflight = load_json(OUT_ROOT / "preflight.json", {})
    tasks = load_json(OUT_ROOT / "task_subset.json", {})
    sweep = load_json(OUT_ROOT / "layerhook_followup_traces.json", load_json(OUT_ROOT / "layerhook_followup_traces.partial.json", {}))
    follow_rows = [normalize_row(row, "followup") for row in sweep.get("rows") or []]
    previous_rows = compatible_previous_rows(preflight)
    combined_rows = previous_rows + follow_rows

    follow_analysis = analyze_rows(follow_rows, "followup_only")
    previous_analysis = analyze_rows(previous_rows, "previous_partial_only") if previous_rows else {"label": "previous_partial_only", "row_count": 0}
    combined_analysis = analyze_rows(combined_rows, "combined") if combined_rows else {"label": "combined", "row_count": 0}
    summary = verdicts(audit, preflight, tasks, sweep, follow_analysis)

    cell_rows = []
    for key, metrics in sorted(follow_analysis["cells"].items()):
        mode, alpha = key.split("|")
        cell_rows.append(
            {
                "mode": mode,
                "alpha": alpha,
                "pos_z": "" if metrics["positive_z_mean"] is None else f"{metrics['positive_z_mean']:.4f}",
                "neg_z": "" if metrics["negative_z_mean"] is None else f"{metrics['negative_z_mean']:.4f}",
                "rand_z": "" if metrics["random_z_mean"] is None else f"{metrics['random_z_mean']:.4f}",
                "rand_std": "" if metrics["random_z_std"] is None else f"{metrics['random_z_std']:.4f}",
                "signed": metrics["signed_causal_signature"],
                "strong_signed": metrics["strong_signed_causal_signature"],
                "stable": metrics["stable"],
                "pos_lift": "" if metrics["positive_success_lift_vs_baseline"] is None else f"{metrics['positive_success_lift_vs_baseline']:.4f}",
            }
        )

    analysis_payload = {
        "summary": summary,
        "followup_only": follow_analysis,
        "previous_partial_only": previous_analysis,
        "combined": combined_analysis,
        "partial_trace_merge": {
            "compatible": bool(previous_rows),
            "previous_row_count": len(previous_rows),
            "followup_row_count": len(follow_rows),
            "combined_row_count": len(combined_rows),
            "merge_rule": "T1 layer_hook_injection rows with same NoNorm direction head and disabled cache",
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, analysis_payload)

    lines = [
        "# BG Stage 2 Layer-Hook Follow-Up Analysis",
        "",
        *top_lines(summary),
        "",
        "## Mechanical Validation",
        "",
        f"- mechanical_ready: `{follow_analysis.get('mechanical_ready')}`",
        f"- cuda_error_count: `{follow_analysis.get('cuda_error_count')}`",
        f"- nan_or_inf_count: `{follow_analysis.get('nan_or_inf_count')}`",
        f"- rms_change_by_alpha: `{follow_analysis.get('rms_change_by_alpha')}`",
        "",
        "## Signed Causal Sensitivity",
        "",
        *table(cell_rows, ["mode", "alpha", "pos_z", "neg_z", "rand_z", "rand_std", "signed", "strong_signed", "stable", "pos_lift"]),
        "",
        "## Partial Trace Merge",
        "",
        f"- previous_row_count: `{len(previous_rows)}`",
        f"- followup_row_count: `{len(follow_rows)}`",
        f"- combined_row_count: `{len(combined_rows)}`",
        "",
        "## Interpretation",
        "",
        interpretation(summary),
    ]
    write_md(OUT_MD, lines)

    summary_payload = {
        **summary,
        "report_paths": {
            "partial_trace_audit": rel(OUT_ROOT / "partial_trace_audit.md"),
            "preflight": rel(OUT_ROOT / "preflight.md"),
            "task_subset": rel(OUT_ROOT / "task_subset.md"),
            "traces": rel(OUT_ROOT / "layerhook_followup_traces.json"),
            "analysis": rel(OUT_MD),
            "summary": rel(SUMMARY_MD),
            "docs": rel(DOC_MAIN),
        },
        "commands_run": [
            "venv/bin/python -m py_compile utilities/tests/manual/analyze_bg_stage2_partial_trace.py",
            "venv/bin/python -m py_compile utilities/tests/manual/bg_stage2_layerhook_followup_preflight.py",
            "venv/bin/python -m py_compile utilities/tests/manual/build_bg_stage2_layerhook_followup_tasks.py",
            "venv/bin/python -m py_compile utilities/tests/manual/run_bg_stage2_layerhook_followup.py",
            "venv/bin/python -m py_compile utilities/tests/manual/analyze_bg_stage2_layerhook_followup.py",
            "venv/bin/python -u utilities/tests/manual/analyze_bg_stage2_partial_trace.py",
            "venv/bin/python -u utilities/tests/manual/bg_stage2_layerhook_followup_preflight.py",
            "venv/bin/python -u utilities/tests/manual/build_bg_stage2_layerhook_followup_tasks.py",
            "venv/bin/python -u utilities/tests/manual/run_bg_stage2_layerhook_followup.py",
            "venv/bin/python -u utilities/tests/manual/analyze_bg_stage2_layerhook_followup.py",
        ],
        "files_modified_or_created": [
            "shared/utilities/tests/manual/analyze_bg_stage2_partial_trace.py",
            "shared/utilities/tests/manual/bg_stage2_layerhook_followup_preflight.py",
            "shared/utilities/tests/manual/build_bg_stage2_layerhook_followup_tasks.py",
            "shared/utilities/tests/manual/run_bg_stage2_layerhook_followup.py",
            "shared/utilities/tests/manual/analyze_bg_stage2_layerhook_followup.py",
            rel(DOC_MAIN),
            rel(SUMMARY_JSON),
            rel(SUMMARY_MD),
        ],
        "blockers": [],
    }
    write_json(SUMMARY_JSON, summary_payload)
    summary_lines = [
        "# BG Stage 2 Layer-Hook Follow-Up Summary",
        "",
        *top_lines(summary),
        "",
        "## 1. Previous Partial Trace Audit",
        "",
        f"See `{rel(OUT_ROOT / 'partial_trace_audit.md')}`.",
        "",
        "## 2. Preflight",
        "",
        f"See `{rel(OUT_ROOT / 'preflight.md')}`.",
        "",
        "## 3. Task Subset",
        "",
        f"See `{rel(OUT_ROOT / 'task_subset.md')}`.",
        "",
        "## 4. Layer-Hook Sweep",
        "",
        f"See `{rel(OUT_ROOT / 'layerhook_followup_traces.md')}`.",
        "",
        "## 5. Mechanical Validation",
        "",
        f"BG_LAYERHOOK_MECHANICAL_VERDICT = {summary['BG_LAYERHOOK_MECHANICAL_VERDICT']}.",
        "",
        "## 6. Signed Causal Analysis",
        "",
        f"BG_LAYERHOOK_SIGNED_CAUSAL_VERDICT = {summary['BG_LAYERHOOK_SIGNED_CAUSAL_VERDICT']}.",
        "",
        "## 7. L1 vs L4",
        "",
        f"BG_SINGLE_LOOP_POSITION_VERDICT = {summary['BG_SINGLE_LOOP_POSITION_VERDICT']}.",
        "",
        "## 8. Multiloop vs Single-Loop",
        "",
        f"BG_MULTILOOP_VERDICT = {summary['BG_MULTILOOP_VERDICT']}.",
        "",
        "## 9. Stability",
        "",
        f"BG_LAYERHOOK_STABILITY_VERDICT = {summary['BG_LAYERHOOK_STABILITY_VERDICT']}.",
        "",
        "## 10. Final Correctness",
        "",
        f"BG_FINAL_TASK_LIFT_VERDICT = {summary['BG_FINAL_TASK_LIFT_VERDICT']}.",
        "",
        "## 11. Interpretation",
        "",
        interpretation(summary),
        "",
        "## 12. Docs Updated",
        "",
        f"- `{rel(DOC_MAIN)}`",
        *[f"- `{rel(path)}`" for path in APPEND_DOCS],
        "",
        "## 13. Files Modified / Created",
        "",
        *[f"- `{path}`" for path in summary_payload["files_modified_or_created"]],
        "",
        "## 14. Commands Run",
        "",
        *[f"- `{cmd}`" for cmd in summary_payload["commands_run"]],
        "",
        "## 15. Blockers",
        "",
        "- None",
    ]
    write_md(SUMMARY_MD, summary_lines)
    write_docs(summary)
    print(f"BG_LAYERHOOK_SIGNED_CAUSAL_VERDICT = {summary['BG_LAYERHOOK_SIGNED_CAUSAL_VERDICT']}")
    print(f"BG_LAYERHOOK_FOLLOWUP_VERDICT = {summary['BG_LAYERHOOK_FOLLOWUP_VERDICT']}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(SUMMARY_JSON)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
