"""Analyze BG empirical steering direction results and update documentation."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from bg_empirical_steering_common import (
    OUT_ROOT,
    PROJECT_ROOT,
    append_once,
    avg,
    finite,
    load_json,
    rate,
    rel,
    summarize_cell,
    write_json,
    write_md,
)


OUT_JSON = OUT_ROOT / "analysis.json"
OUT_MD = OUT_ROOT / "analysis.md"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
DOC_MAIN = PROJECT_ROOT / "docs/evaluator/bg_empirical_steering_direction.md"
APPEND_DOCS = [
    PROJECT_ROOT / "docs/evaluator/current_state.md",
    PROJECT_ROOT / "docs/evaluator/domain_transfer_ledger.md",
    PROJECT_ROOT / "docs/evaluator/bg_stage2_layerhook_followup.md",
    PROJECT_ROOT / "docs/evaluator/bg_trajectory_prediction_sweep.md",
    PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-18_v8.1_routing_locked.md",
]


def group_cells(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("direction_name")), str(row.get("mode")), finite(row.get("alpha")))].append(row)
    return {
        f"{direction}|{mode}|{alpha:g}": summarize_cell(group)
        for (direction, mode, alpha), group in sorted(grouped.items())
    }


def best_score(cells: dict[str, Any], direction_filter: set[str] | None = None, mode_filter: set[str] | None = None) -> tuple[str, float]:
    best_key = "none"
    best = -1e9
    for key, metrics in cells.items():
        direction, mode, _alpha = key.split("|")
        if direction_filter is not None and direction not in direction_filter:
            continue
        if mode_filter is not None and mode not in mode_filter:
            continue
        score = finite(metrics.get("signed_score"))
        if score > best:
            best = score
            best_key = key
    return best_key, best


def analyze(rows: list[dict[str, Any]], sweep: dict[str, Any], directions: dict[str, Any]) -> dict[str, Any]:
    intervention = [row for row in rows if row.get("condition") != "zero_baseline"]
    cells = group_cells(rows)
    direction_names = sorted({str(row.get("direction_name")) for row in rows})
    empirical_names = {name for name in direction_names if name != "RAW_NONORM_READOUT"}
    raw_key, raw_score = best_score(cells, {"RAW_NONORM_READOUT"})
    empirical_key, empirical_score = best_score(cells, empirical_names)
    decayed_key, decayed_score = best_score(cells, None, {"multi_loop_decayed"})
    l1_key, l1_score = best_score(cells, None, {"single_loop_L1"})
    strong_empirical = [
        key
        for key, value in cells.items()
        if key.split("|")[0] in empirical_names and value.get("strong_signed_causal_signature")
    ]
    unsigned_empirical = [
        key
        for key, value in cells.items()
        if key.split("|")[0] in empirical_names and value.get("unsigned_effect")
    ]
    strong_raw = [
        key
        for key, value in cells.items()
        if key.split("|")[0] == "RAW_NONORM_READOUT" and value.get("strong_signed_causal_signature")
    ]
    signed_cells = [key for key, value in cells.items() if value.get("signed_causal_signature")]
    stability_ok = bool(intervention) and not any(
        row.get("safety_status") == "DESTABILIZING" or row.get("cuda_error") or row.get("nan_or_inf_activations")
        for row in intervention
    )
    if len(intervention) < 12:
        causal_verdict = "INSUFFICIENT"
    elif strong_empirical and not strong_raw:
        causal_verdict = "EMPIRICAL_SIGNED_CAUSAL"
    elif unsigned_empirical:
        causal_verdict = "EMPIRICAL_UNSIGNED_ONLY"
    else:
        causal_verdict = "NO_EMPIRICAL_CAUSAL_EFFECT"
    gain = empirical_score - raw_score
    if len(intervention) < 12:
        vs_raw = "INSUFFICIENT"
    elif gain > 0.10:
        vs_raw = "EMPIRICAL_BEATS_RAW"
    elif raw_score > empirical_score + 0.10:
        vs_raw = "RAW_BETTER"
    else:
        vs_raw = "RAW_MATCHES_EMPIRICAL"

    geometry = "INCONCLUSIVE"
    target = next((row for row in directions.get("targets", []) if row.get("target_id") == "T1"), {})
    direction_meta = {row.get("direction_name"): row for row in target.get("directions", [])}
    best_emp_name = empirical_key.split("|")[0] if empirical_key != "none" else ""
    best_cos = direction_meta.get(best_emp_name, {}).get("cosine_to_raw_nonorm")
    if best_cos is not None and abs(float(best_cos)) < 0.20 and vs_raw == "EMPIRICAL_BEATS_RAW":
        geometry = "RAW_READOUT_NOT_PRODUCTION_DIRECTION"
    elif best_cos is not None and abs(float(best_cos)) >= 0.50:
        geometry = "RAW_AND_EMPIRICAL_ALIGNED"

    if not intervention:
        stability = "INSUFFICIENT"
    elif not stability_ok:
        stability = "DESTABILIZING"
    else:
        stability = "STABLE"
    base = [row for row in rows if row.get("condition") == "zero_baseline"]
    base_success = rate([bool(row.get("is_correct")) for row in base])
    pos_success = rate([bool(row.get("is_correct")) for row in intervention if row.get("condition") == "positive"])
    random_success = rate([bool(row.get("is_correct")) for row in intervention if row.get("condition") == "random"])
    if base_success is None or pos_success is None:
        final_lift = "INSUFFICIENT"
    elif pos_success > base_success + 0.05:
        final_lift = "POSITIVE_LIFT"
    elif pos_success < base_success - 0.05:
        final_lift = "NEGATIVE_LIFT"
    else:
        final_lift = "NULL_LIFT"

    tiny = load_json(OUT_ROOT / "tiny_adapter_diagnostic.json", {})
    tiny_verdict = tiny.get("BG_TINY_STEERING_ADAPTER_VERDICT", "SKIPPED")
    if causal_verdict == "EMPIRICAL_SIGNED_CAUSAL" and stability == "STABLE" and final_lift in {"POSITIVE_LIFT", "NULL_LIFT"}:
        overall = "PROMISING_EMPIRICAL_HANDLE" if final_lift == "POSITIVE_LIFT" else "CAUSAL_BUT_NO_TASK_LIFT"
    elif stability == "DESTABILIZING":
        overall = "DESTABILIZING"
    elif len(intervention) < 12:
        overall = "INSUFFICIENT"
    else:
        overall = "READOUT_ONLY_UNDER_TESTED_DIRECTIONS"
    if overall == "PROMISING_EMPIRICAL_HANDLE":
        rec = "expand_empirical_direction_steering_across_targets"
    elif overall == "CAUSAL_BUT_NO_TASK_LIFT":
        rec = "design_phase2_regularization_to_amplify_propagation"
    elif overall == "DESTABILIZING":
        rec = "abandon_current_inference_time_steering_directions"
    elif overall == "INSUFFICIENT":
        rec = "improve_direction_contrast_dataset_or_reduce_scope"
    else:
        rec = "move_to_phase2_training_or_adapter_based_steering_protocol"

    return {
        "cells": cells,
        "direction_names": direction_names,
        "row_count": len(rows),
        "intervention_row_count": len(intervention),
        "task_count": len({row.get("task_subset_index") for row in rows}),
        "cuda_error_count": sum(1 for row in intervention if row.get("cuda_error")),
        "nan_or_inf_count": sum(1 for row in intervention if row.get("nan_or_inf_activations")),
        "safety_ok": stability_ok,
        "rms_change_by_alpha": {
            str(alpha): avg([finite(row.get("activation_rms_change")) for row in intervention if finite(row.get("alpha")) == alpha])
            for alpha in sorted({finite(row.get("alpha")) for row in intervention})
        },
        "signed_cells": signed_cells,
        "strong_empirical_cells": strong_empirical,
        "unsigned_empirical_cells": unsigned_empirical,
        "raw_best_key": raw_key,
        "raw_best_signed_score": raw_score,
        "empirical_best_key": empirical_key,
        "empirical_best_signed_score": empirical_score,
        "EMPIRICAL_GAIN_OVER_RAW": gain,
        "MODE_COVERAGE": sweep.get("MODE_COVERAGE", {}),
        "MULTILOOP_DECAYED_VS_L1_DELTA": decayed_score - l1_score,
        "best_multiloop_key": decayed_key,
        "best_l1_key": l1_key,
        "direction_geometry": {
            "best_empirical_direction": best_emp_name,
            "best_empirical_cosine_to_raw": best_cos,
            "cosines": target.get("direction_cosines", {}),
        },
        "baseline_success_rate": base_success,
        "positive_success_rate": pos_success,
        "random_success_rate": random_success,
        "BG_EMPIRICAL_DIRECTION_CAUSAL_VERDICT": causal_verdict,
        "BG_EMPIRICAL_VS_RAW_VERDICT": vs_raw,
        "BG_STEERING_DIRECTION_GEOMETRY_VERDICT": geometry,
        "BG_EMPIRICAL_STEERING_STABILITY_VERDICT": stability,
        "BG_EMPIRICAL_FINAL_LIFT_VERDICT": final_lift,
        "BG_TINY_STEERING_ADAPTER_VERDICT": tiny_verdict,
        "BG_EMPIRICAL_STEERING_VERDICT": overall,
        "RECOMMENDED_NEXT": rec,
    }


def write_reports(analysis: dict[str, Any], preflight: dict[str, Any], directions: dict[str, Any], tasks: dict[str, Any], sweep: dict[str, Any]) -> None:
    summary = {
        "BG_EMPIRICAL_STEERING_PREFLIGHT_VERDICT": preflight.get("BG_EMPIRICAL_STEERING_PREFLIGHT_VERDICT", "BLOCKED"),
        "BG_EMPIRICAL_DIRECTION_BUILD_VERDICT": directions.get("BG_EMPIRICAL_DIRECTION_BUILD_VERDICT", "BLOCKED"),
        "BG_EMPIRICAL_STEERING_TASKS_VERDICT": tasks.get("BG_EMPIRICAL_STEERING_TASKS_VERDICT", "BLOCKED"),
        "BG_EMPIRICAL_STEERING_SWEEP_VERDICT": sweep.get("BG_EMPIRICAL_STEERING_SWEEP_VERDICT", "BLOCKED"),
        "BG_EMPIRICAL_DIRECTION_CAUSAL_VERDICT": analysis["BG_EMPIRICAL_DIRECTION_CAUSAL_VERDICT"],
        "BG_EMPIRICAL_VS_RAW_VERDICT": analysis["BG_EMPIRICAL_VS_RAW_VERDICT"],
        "BG_STEERING_DIRECTION_GEOMETRY_VERDICT": analysis["BG_STEERING_DIRECTION_GEOMETRY_VERDICT"],
        "BG_EMPIRICAL_STEERING_STABILITY_VERDICT": analysis["BG_EMPIRICAL_STEERING_STABILITY_VERDICT"],
        "BG_EMPIRICAL_FINAL_LIFT_VERDICT": analysis["BG_EMPIRICAL_FINAL_LIFT_VERDICT"],
        "BG_TINY_STEERING_ADAPTER_VERDICT": analysis["BG_TINY_STEERING_ADAPTER_VERDICT"],
        "BG_EMPIRICAL_STEERING_VERDICT": analysis["BG_EMPIRICAL_STEERING_VERDICT"],
        "MODE_COVERAGE": analysis["MODE_COVERAGE"],
        "MULTILOOP_DECAYED_VS_L1_DELTA": analysis["MULTILOOP_DECAYED_VS_L1_DELTA"],
        "EMPIRICAL_GAIN_OVER_RAW": analysis["EMPIRICAL_GAIN_OVER_RAW"],
        "RECOMMENDED_NEXT": analysis["RECOMMENDED_NEXT"],
        "report_paths": {
            "preflight": rel(OUT_ROOT / "preflight.md"),
            "directions": rel(OUT_ROOT / "directions.md"),
            "task_subset": rel(OUT_ROOT / "task_subset.md"),
            "traces": rel(OUT_ROOT / "empirical_steering_traces.json"),
            "analysis": rel(OUT_MD),
            "summary": rel(SUMMARY_MD),
            "docs": rel(DOC_MAIN),
            "tiny_adapter": rel(OUT_ROOT / "tiny_adapter_diagnostic.md"),
        },
    }
    write_json(OUT_JSON, analysis)
    write_json(SUMMARY_JSON, summary)

    top = [
        "# BG Empirical Steering Direction Summary",
        "",
    ]
    for key in [
        "BG_EMPIRICAL_STEERING_PREFLIGHT_VERDICT",
        "BG_EMPIRICAL_DIRECTION_BUILD_VERDICT",
        "BG_EMPIRICAL_STEERING_TASKS_VERDICT",
        "BG_EMPIRICAL_STEERING_SWEEP_VERDICT",
        "BG_EMPIRICAL_DIRECTION_CAUSAL_VERDICT",
        "BG_EMPIRICAL_VS_RAW_VERDICT",
        "BG_STEERING_DIRECTION_GEOMETRY_VERDICT",
        "BG_EMPIRICAL_STEERING_STABILITY_VERDICT",
        "BG_EMPIRICAL_FINAL_LIFT_VERDICT",
        "BG_TINY_STEERING_ADAPTER_VERDICT",
        "BG_EMPIRICAL_STEERING_VERDICT",
        "MULTILOOP_DECAYED_VS_L1_DELTA",
        "RECOMMENDED_NEXT",
    ]:
        top.append(f"{key} = {summary.get(key)}")
    top.extend(
        [
            "",
            "## 1. Motivation",
            "",
            "The raw NoNorm readout vector produced mostly unsigned movement in the prior layer-hook follow-up. This probe tested whether success-derived frozen-feature directions improve signed steering.",
            "",
            "## 2. Direction Construction",
            "",
            f"Directions tested: `{', '.join(analysis['direction_names'])}`.",
            "",
            "## 3. Steering Sweep",
            "",
            f"Rows: `{analysis['row_count']}`; intervention rows: `{analysis['intervention_row_count']}`; tasks: `{analysis['task_count']}`.",
            "",
            "## 4. Causal Sensitivity",
            "",
            f"Strong empirical cells: `{analysis['strong_empirical_cells']}`.",
            f"Unsigned empirical cells: `{analysis['unsigned_empirical_cells']}`.",
            f"Empirical gain over raw: `{analysis['EMPIRICAL_GAIN_OVER_RAW']}`.",
            "",
            "## 5. Stability",
            "",
            f"CUDA errors: `{analysis['cuda_error_count']}`; NaN/Inf rows: `{analysis['nan_or_inf_count']}`; safety_ok: `{analysis['safety_ok']}`.",
            "",
            "## 6. Mode Comparison",
            "",
            f"MULTILOOP_DECAYED_VS_L1_DELTA = `{analysis['MULTILOOP_DECAYED_VS_L1_DELTA']}`.",
            "",
            "## 7. Interpretation",
            "",
            "Empirical directions are evaluated as steering directions, not as branch selectors. Final correctness is secondary to signed BG-readable movement.",
            "",
            "## 8. Files Modified / Created",
            "",
            "- `utilities/tests/manual/bg_empirical_steering_preflight.py`",
            "- `utilities/tests/manual/build_empirical_steering_directions.py`",
            "- `utilities/tests/manual/build_empirical_steering_task_subset.py`",
            "- `utilities/tests/manual/run_empirical_direction_layerhook_probe.py`",
            "- `utilities/tests/manual/analyze_empirical_steering_results.py`",
            "- `utilities/tests/manual/train_tiny_steering_adapter_diagnostic.py`",
            "- `docs/evaluator/bg_empirical_steering_direction.md`",
            f"- `{rel(SUMMARY_JSON)}`",
            f"- `{rel(OUT_JSON)}`",
            "",
            "## 9. Blockers",
            "",
            "- None if verdict is READY/PARTIAL; see JSON for stage-specific blockers.",
        ]
    )
    write_md(SUMMARY_MD, top)

    cell_lines = [
        "# BG Empirical Steering Direction Analysis",
        "",
    ]
    for key in [
        "BG_EMPIRICAL_DIRECTION_CAUSAL_VERDICT",
        "BG_EMPIRICAL_VS_RAW_VERDICT",
        "BG_STEERING_DIRECTION_GEOMETRY_VERDICT",
        "BG_EMPIRICAL_STEERING_STABILITY_VERDICT",
        "BG_EMPIRICAL_FINAL_LIFT_VERDICT",
        "BG_TINY_STEERING_ADAPTER_VERDICT",
        "BG_EMPIRICAL_STEERING_VERDICT",
    ]:
        cell_lines.append(f"{key} = {analysis[key]}")
    cell_lines.extend(
        [
            "",
            "## Direction Cells",
            "",
            "| direction | mode | alpha | pos_z | neg_z | rand_z | rand_std | signed | strong | stable |",
            "|---|---|---:|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for key, value in sorted(analysis["cells"].items()):
        direction, mode, alpha = key.split("|")
        cell_lines.append(
            f"| `{direction}` | `{mode}` | {alpha} | {finite(value.get('positive_z_mean')):.4f} | "
            f"{finite(value.get('negative_z_mean')):.4f} | {finite(value.get('random_z_mean')):.4f} | "
            f"{finite(value.get('random_z_std')):.4f} | `{value.get('signed_causal_signature')}` | "
            f"`{value.get('strong_signed_causal_signature')}` | `{value.get('stable')}` |"
        )
    cell_lines.extend(
        [
            "",
            "## Geometry",
            "",
            f"- best_empirical_direction: `{analysis['direction_geometry'].get('best_empirical_direction')}`",
            f"- best_empirical_cosine_to_raw: `{analysis['direction_geometry'].get('best_empirical_cosine_to_raw')}`",
        ]
    )
    write_md(OUT_MD, cell_lines)

    doc_lines = [
        "# BG Empirical Steering Direction Probe",
        "",
        "## Purpose",
        "",
        "The prior layer-hook follow-up showed that the hook mechanism is stable, but raw BG NoNorm readout vectors produce mostly unsigned score movement. This probe tests whether success-derived frozen-feature directions are better steering axes.",
        "",
        "## Direction Construction",
        "",
        "Directions were built from Stage 1 frozen prefix features using raw NoNorm, empirical mean difference, diagonal-whitened mean difference, and a logistic success probe. Ouro weights and BG heads were not trained.",
        "",
        "## Layer-Hook Setup",
        "",
        "The probe uses layer_hook_injection only, with use_cache=False, current_ut loop identity, position -1, alpha <= 0.02, and the prior best modes: multi_loop_decayed as primary and single_loop_L1 as comparison.",
        "",
        "## Results",
        "",
    ]
    for key in [
        "BG_EMPIRICAL_DIRECTION_CAUSAL_VERDICT",
        "BG_EMPIRICAL_VS_RAW_VERDICT",
        "BG_STEERING_DIRECTION_GEOMETRY_VERDICT",
        "BG_EMPIRICAL_STEERING_STABILITY_VERDICT",
        "BG_EMPIRICAL_FINAL_LIFT_VERDICT",
        "BG_TINY_STEERING_ADAPTER_VERDICT",
        "BG_EMPIRICAL_STEERING_VERDICT",
    ]:
        doc_lines.append(f"- {key} = `{analysis[key]}`")
    doc_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If empirical directions show only unsigned movement, the likely issue is not the layer-hook mechanism but the absence of a calibrated production-space control direction.",
            "",
            "## Reports",
            "",
            f"- summary: `{rel(SUMMARY_MD)}`",
            f"- analysis: `{rel(OUT_MD)}`",
            f"- traces: `{rel(OUT_ROOT / 'empirical_steering_traces.json')}`",
        ]
    )
    write_md(DOC_MAIN, doc_lines)

    append_lines = [
        f"BG_EMPIRICAL_STEERING_PREFLIGHT_VERDICT = {summary['BG_EMPIRICAL_STEERING_PREFLIGHT_VERDICT']}",
        f"BG_EMPIRICAL_DIRECTION_BUILD_VERDICT = {summary['BG_EMPIRICAL_DIRECTION_BUILD_VERDICT']}",
        f"BG_EMPIRICAL_STEERING_TASKS_VERDICT = {summary['BG_EMPIRICAL_STEERING_TASKS_VERDICT']}",
        f"BG_EMPIRICAL_STEERING_SWEEP_VERDICT = {summary['BG_EMPIRICAL_STEERING_SWEEP_VERDICT']}",
        f"BG_EMPIRICAL_DIRECTION_CAUSAL_VERDICT = {summary['BG_EMPIRICAL_DIRECTION_CAUSAL_VERDICT']}",
        f"BG_EMPIRICAL_VS_RAW_VERDICT = {summary['BG_EMPIRICAL_VS_RAW_VERDICT']}",
        f"BG_STEERING_DIRECTION_GEOMETRY_VERDICT = {summary['BG_STEERING_DIRECTION_GEOMETRY_VERDICT']}",
        f"BG_EMPIRICAL_STEERING_STABILITY_VERDICT = {summary['BG_EMPIRICAL_STEERING_STABILITY_VERDICT']}",
        f"BG_EMPIRICAL_FINAL_LIFT_VERDICT = {summary['BG_EMPIRICAL_FINAL_LIFT_VERDICT']}",
        f"BG_TINY_STEERING_ADAPTER_VERDICT = {summary['BG_TINY_STEERING_ADAPTER_VERDICT']}",
        f"BG_EMPIRICAL_STEERING_VERDICT = {summary['BG_EMPIRICAL_STEERING_VERDICT']}",
        f"MODE_COVERAGE = {json.dumps(summary['MODE_COVERAGE'], sort_keys=True)}",
        f"MULTILOOP_DECAYED_VS_L1_DELTA = {summary['MULTILOOP_DECAYED_VS_L1_DELTA']}",
        "Interpretation: empirical directions test whether BG is readout-only or whether calibrated success-space directions can become causal handles.",
        f"Full reports: `{rel(SUMMARY_MD)}`, `{rel(OUT_MD)}`, `{rel(DOC_MAIN)}`.",
    ]
    for path in APPEND_DOCS:
        append_once(path, "BG empirical steering direction probe (2026-05-18)", append_lines)


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    preflight = load_json(OUT_ROOT / "preflight.json", {})
    directions = load_json(OUT_ROOT / "directions.json", {})
    tasks = load_json(OUT_ROOT / "task_subset.json", {})
    sweep = load_json(OUT_ROOT / "empirical_steering_traces.json", {})
    rows = list(sweep.get("rows") or [])
    analysis = analyze(rows, sweep, directions)
    analysis["elapsed_seconds"] = round(time.time() - started, 3)
    write_reports(analysis, preflight, directions, tasks, sweep)
    print(f"BG_EMPIRICAL_DIRECTION_CAUSAL_VERDICT = {analysis['BG_EMPIRICAL_DIRECTION_CAUSAL_VERDICT']}")
    print(f"BG_EMPIRICAL_STEERING_VERDICT = {analysis['BG_EMPIRICAL_STEERING_VERDICT']}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(SUMMARY_JSON)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
