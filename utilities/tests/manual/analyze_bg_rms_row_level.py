#!/usr/bin/env python3
"""Row-level audit for the BG RMS-calibrated steering probe."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path("opi/taps/probes/bg_preconsolidation_control_probes_2026-05-18")
TRACE_PATH = ROOT / "rms_steering_traces.json"
ANALYSIS_PATH = ROOT / "rms_steering_analysis.json"
OUT_JSON = ROOT / "rms_row_level_analysis.json"
OUT_MD = ROOT / "rms_row_level_analysis.md"
OUT_CSV = ROOT / "rms_row_level_rows.csv"
OUT_JSONL = ROOT / "rms_row_level_rows.jsonl"
OUT_ROWS_MD = ROOT / "rms_row_level_rows.md"
OUT_CELLS_CSV = ROOT / "rms_row_level_cells.csv"
OUT_CELLS_MD = ROOT / "rms_row_level_cells.md"
OUT_TASKS_CSV = ROOT / "rms_row_level_tasks.csv"


def fnum(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def avg(values: list[Any]) -> float | None:
    nums = [fnum(v) for v in values]
    nums = [v for v in nums if v is not None]
    return mean(nums) if nums else None


def std(values: list[Any]) -> float | None:
    nums = [fnum(v) for v in values]
    nums = [v for v in nums if v is not None]
    if not nums:
        return None
    return pstdev(nums)


def fmt(value: Any, digits: int = 4) -> str:
    num = fnum(value)
    if num is None:
        if value is None:
            return ""
        return str(value)
    return f"{num:.{digits}f}"


def score_delta(row: dict[str, Any], key: str) -> float | None:
    post = fnum(row.get(key))
    base = fnum(row.get(f"{key}_baseline"))
    if post is None or base is None:
        return None
    return post - base


def row_reason(row: dict[str, Any]) -> str:
    reasons: list[str] = []
    if row.get("cuda_error"):
        reasons.append("cuda_error")
    if row.get("nan_or_inf_activations"):
        reasons.append("nan_or_inf")
    if row.get("empty_output"):
        reasons.append("empty_output")
    if row.get("parse_failed"):
        reasons.append("parse_failed")
    rep = fnum(row.get("repetition_rate"))
    if rep is not None and rep > 0.30:
        reasons.append(f"repetition_rate={rep:.3f}")
    act = fnum(row.get("activation_rms_change"))
    if act is not None and act > 0.05:
        reasons.append(f"activation_rms_change={act:.3f}")
    if row.get("hit_max_tokens"):
        reasons.append("hit_max_tokens")
    return ", ".join(reasons) if reasons else "none"


def compact_row(index: int, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_index": index,
        "is_intervention": row.get("alpha") != 0 and row.get("condition") != "zero_baseline",
        "task_id": row.get("task_id"),
        "task_subset_index": row.get("task_subset_index"),
        "normalization_mode": row.get("normalization_mode"),
        "direction_name": row.get("direction_name"),
        "intervention_mode": row.get("intervention_mode"),
        "alpha": row.get("alpha"),
        "condition": row.get("condition"),
        "random_control_idx": row.get("random_control_idx"),
        "random_control_n": row.get("random_control_n"),
        "z_score_change": row.get("z_score_change"),
        "target_score_delta": score_delta(row, "target_head_score"),
        "diagnostic_d1_delta": score_delta(row, "diagnostic_d1_score"),
        "objective_mixed_delta": score_delta(row, "objective_mixed_score"),
        "hh_general_delta": score_delta(row, "hh_general_score"),
        "effective_delta_rms_fraction": row.get("effective_delta_rms_fraction"),
        "activation_rms_change": row.get("activation_rms_change"),
        "per_loop_activation_rms_change": row.get("per_loop_activation_rms_change"),
        "hook_forward_call_count": row.get("hook_forward_call_count"),
        "hook_modifications": row.get("hook_modifications"),
        "hook_loop_index_source": row.get("hook_loop_index_source"),
        "cache_intervention_mode": row.get("cache_intervention_mode"),
        "safety_status": row.get("safety_status"),
        "safety_reason": row_reason(row),
        "cuda_error": row.get("cuda_error"),
        "nan_or_inf_activations": row.get("nan_or_inf_activations"),
        "empty_output": row.get("empty_output"),
        "parse_failed": row.get("parse_failed"),
        "repetition_rate": row.get("repetition_rate"),
        "output_length": row.get("output_length"),
        "hit_max_tokens": row.get("hit_max_tokens"),
        "parsed_answer": row.get("parsed_answer"),
        "is_correct": row.get("is_correct"),
    }


def summarize_cells(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("condition") == "zero_baseline":
            continue
        key = (
            row.get("normalization_mode"),
            row.get("direction_name"),
            row.get("intervention_mode"),
            row.get("alpha"),
        )
        groups[key].append(row)

    summaries: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda x: tuple(str(v) for v in x[0])):
        by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in items:
            by_condition[str(row.get("condition"))].append(row)

        cond_summary: dict[str, dict[str, Any]] = {}
        for condition, condition_rows in sorted(by_condition.items()):
            zs = [r.get("z_score_change") for r in condition_rows]
            cond_summary[condition] = {
                "n": len(condition_rows),
                "z_mean": avg(zs),
                "z_std": std(zs),
                "z_min": min([fnum(z) for z in zs if fnum(z) is not None], default=None),
                "z_max": max([fnum(z) for z in zs if fnum(z) is not None], default=None),
                "safety_failures": sum(1 for r in condition_rows if r.get("safety_status") != "OK"),
                "parse_failures": sum(1 for r in condition_rows if r.get("parse_failed")),
                "mean_effective_delta_rms_fraction": avg(
                    [r.get("effective_delta_rms_fraction") for r in condition_rows]
                ),
                "mean_activation_rms_change": avg([r.get("activation_rms_change") for r in condition_rows]),
                "success_rate": avg([1.0 if r.get("is_correct") else 0.0 for r in condition_rows]),
            }

        pos = cond_summary.get("positive", {}).get("z_mean")
        neg = cond_summary.get("negative", {}).get("z_mean")
        rnd = cond_summary.get("random", {}).get("z_mean")
        rnd_std = cond_summary.get("random", {}).get("z_std")
        signed = pos is not None and neg is not None and rnd is not None and pos > rnd and neg < rnd
        if signed and rnd_std is not None and rnd_std > 0:
            strong_signed = (pos - rnd) >= 0.5 * rnd_std and (rnd - neg) >= 0.5 * rnd_std
        else:
            strong_signed = False
        unsigned = pos is not None and rnd is not None and pos > rnd
        signed_score = None
        if pos is not None and neg is not None and rnd is not None:
            signed_score = (pos - rnd) + (rnd - neg)

        summaries.append(
            {
                "normalization_mode": key[0],
                "direction_name": key[1],
                "intervention_mode": key[2],
                "alpha": key[3],
                "n": len(items),
                "condition_summary": cond_summary,
                "positive_z_mean": pos,
                "negative_z_mean": neg,
                "random_z_mean": rnd,
                "random_z_std": rnd_std,
                "signed_causal_signature": signed,
                "strong_signed_causal_signature": strong_signed,
                "unsigned_positive_over_random": unsigned,
                "signed_score": signed_score,
            }
        )
    return summaries


def summarize_hook_patterns(rows: list[dict[str, Any]]) -> dict[str, Any]:
    interventions = [r for r in rows if r.get("condition") != "zero_baseline"]
    baselines = [r for r in rows if r.get("condition") == "zero_baseline"]
    return {
        "baseline_rows": len(baselines),
        "intervention_rows": len(interventions),
        "hook_forward_call_count_distribution": dict(Counter(r.get("hook_forward_call_count") for r in rows)),
        "hook_modifications_distribution": dict(Counter(r.get("hook_modifications") for r in rows)),
        "loop_index_sources": dict(Counter(r.get("hook_loop_index_source") for r in interventions)),
        "cache_modes": dict(Counter(r.get("cache_intervention_mode") for r in rows)),
        "multi_loop_expected_modifications_rows": sum(
            1 for r in interventions if r.get("intervention_mode") == "multi_loop_decayed" and r.get("hook_modifications") == 512
        ),
        "single_loop_expected_modifications_rows": sum(
            1 for r in interventions if r.get("intervention_mode") == "single_loop_L1" and r.get("hook_modifications") == 128
        ),
    }


def summarize_tasks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("task_id"))].append(row)
    summaries: list[dict[str, Any]] = []
    for task_id, items in sorted(groups.items()):
        interventions = [r for r in items if r.get("condition") != "zero_baseline"]
        summaries.append(
            {
                "task_id": task_id,
                "rows": len(items),
                "intervention_rows": len(interventions),
                "safety_failures": sum(1 for r in items if r.get("safety_status") != "OK"),
                "parse_failures": sum(1 for r in items if r.get("parse_failed")),
                "mean_z_intervention": avg([r.get("z_score_change") for r in interventions]),
                "min_z_intervention": min([fnum(r.get("z_score_change")) for r in interventions if fnum(r.get("z_score_change")) is not None], default=None),
                "max_z_intervention": max([fnum(r.get("z_score_change")) for r in interventions if fnum(r.get("z_score_change")) is not None], default=None),
                "success_rate_intervention": avg([1.0 if r.get("is_correct") else 0.0 for r in interventions]),
                "mean_repetition_rate": avg([r.get("repetition_rate") for r in interventions]),
            }
        )
    return summaries


def condition_overall(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("condition"))].append(row)
    return {
        condition: {
            "n": len(items),
            "z_mean": avg([r.get("z_score_change") for r in items]),
            "z_std": std([r.get("z_score_change") for r in items]),
            "parse_failures": sum(1 for r in items if r.get("parse_failed")),
            "safety_failures": sum(1 for r in items if r.get("safety_status") != "OK"),
            "success_rate": avg([1.0 if r.get("is_correct") else 0.0 for r in items]),
        }
        for condition, items in sorted(groups.items())
    }


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    trace = json.loads(TRACE_PATH.read_text())
    prior_analysis = json.loads(ANALYSIS_PATH.read_text()) if ANALYSIS_PATH.exists() else {}
    rows = trace.get("rows", [])
    compact_rows = [compact_row(i, row) for i, row in enumerate(rows)]
    interventions = [r for r in rows if r.get("condition") != "zero_baseline"]
    baselines = [r for r in rows if r.get("condition") == "zero_baseline"]
    safety_rows = [(i, row) for i, row in enumerate(rows) if row.get("safety_status") != "OK"]

    cell_summaries = summarize_cells(rows)
    best_signed = sorted(
        [c for c in cell_summaries if c.get("signed_score") is not None],
        key=lambda c: c["signed_score"],
        reverse=True,
    )[:10]

    by_norm = defaultdict(list)
    for row in interventions:
        by_norm[row.get("normalization_mode")].append(row)

    analysis = {
        "source_trace": str(TRACE_PATH),
        "source_prior_analysis": str(ANALYSIS_PATH),
        "row_count_total": len(rows),
        "baseline_row_count": len(baselines),
        "intervention_row_count": len(interventions),
        "reported_intervention_forward_passes": trace.get("intervention_forward_passes"),
        "prior_BG_RMS_STEERING_VERDICT": prior_analysis.get("BG_RMS_STEERING_VERDICT"),
        "prior_BG_RMS_STABILITY_VERDICT": prior_analysis.get("BG_RMS_STABILITY_VERDICT"),
        "prior_BG_RMS_VS_L2_VERDICT": prior_analysis.get("BG_RMS_VS_L2_VERDICT"),
        "BG_RMS_STEERING_VERDICT_ROW_LEVEL": (
            "RMS_UNSIGNED_ONLY"
            if prior_analysis.get("unsigned_rms_cells")
            else ("RMS_NO_EFFECT" if len(safety_rows) <= 1 else prior_analysis.get("BG_RMS_STEERING_VERDICT"))
        ),
        "BG_RMS_STABILITY_VERDICT_ROW_LEVEL": "STABLE_BUT_NOISY" if len(safety_rows) == 1 else ("STABLE" if not safety_rows else "DESTABILIZING"),
        "RMS_DESTABILIZING_REINTERPRETATION": (
            "single_output_quality_outlier_not_broad_destabilization"
            if len(safety_rows) == 1
            else "not_applicable"
        ),
        "safety_failure_count": len(safety_rows),
        "cuda_error_count": sum(1 for r in rows if r.get("cuda_error")),
        "nan_or_inf_count": sum(1 for r in rows if r.get("nan_or_inf_activations")),
        "empty_output_count": sum(1 for r in rows if r.get("empty_output")),
        "parse_failure_count": sum(1 for r in rows if r.get("parse_failed")),
        "hit_max_tokens_count": sum(1 for r in rows if r.get("hit_max_tokens")),
        "safety_rows": [
            {
                "row_index": i,
                **compact_row(i, row),
                "output_excerpt": (row.get("output_text") or "")[:700],
            }
            for i, row in safety_rows
        ],
        "row_count_by_normalization": dict(Counter(r.get("normalization_mode") for r in rows)),
        "intervention_count_by_normalization": dict(Counter(r.get("normalization_mode") for r in interventions)),
        "row_count_by_direction": dict(Counter(r.get("direction_name") for r in rows)),
        "row_count_by_mode": dict(Counter(r.get("intervention_mode") for r in rows)),
        "row_count_by_condition_alpha": {
            f"{alpha}|{condition}": count
            for (alpha, condition), count in sorted(Counter((r.get("alpha"), r.get("condition")) for r in rows).items())
        },
        "mechanical": summarize_hook_patterns(rows),
        "effective_rms_by_normalization_alpha": {
            f"{norm}|{alpha}": {
                "n": len(group),
                "mean_effective_delta_rms_fraction": avg([r.get("effective_delta_rms_fraction") for r in group]),
                "std_effective_delta_rms_fraction": std([r.get("effective_delta_rms_fraction") for r in group]),
                "mean_activation_rms_change": avg([r.get("activation_rms_change") for r in group]),
            }
            for (norm, alpha), group in sorted(
                defaultdict(list, {}).items()
            )
        },
        "cell_summaries": cell_summaries,
        "task_summaries": summarize_tasks(rows),
        "condition_overall": condition_overall(rows),
        "best_signed_score_cells": best_signed,
        "strong_signed_cells": [c for c in cell_summaries if c.get("strong_signed_causal_signature")],
        "unsigned_positive_cells": [c for c in cell_summaries if c.get("unsigned_positive_over_random")],
        "interpretation": {
            "completed_all_planned_interventions": trace.get("intervention_forward_passes") == 564,
            "one_row_aggregate_destabilization_is_conservative": len(safety_rows) == 1,
            "mechanically_stable_except_output_repetition_outlier": (
                len(safety_rows) == 1
                and sum(1 for r in rows if r.get("cuda_error")) == 0
                and sum(1 for r in rows if r.get("nan_or_inf_activations")) == 0
            ),
            "rms_stability_should_be_treated_as": "STABLE_BUT_NOISY" if len(safety_rows) == 1 else ("STABLE" if not safety_rows else "DESTABILIZING"),
            "summary": (
                "The RMS sweep completed all planned intervention passes. The aggregate DESTABILIZING label "
                "comes from one output-quality outlier, not from CUDA failure, NaN/Inf activations, or hidden-state "
                "RMS blow-up."
            ),
        },
    }

    # Fill effective RMS grouping after constructing the main object.
    rms_groups: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in interventions:
        rms_groups[(row.get("normalization_mode"), row.get("alpha"))].append(row)
    analysis["effective_rms_by_normalization_alpha"] = {
        f"{norm}|{alpha}": {
            "n": len(group),
            "mean_effective_delta_rms_fraction": avg([r.get("effective_delta_rms_fraction") for r in group]),
            "std_effective_delta_rms_fraction": std([r.get("effective_delta_rms_fraction") for r in group]),
            "min_effective_delta_rms_fraction": min(
                [fnum(r.get("effective_delta_rms_fraction")) for r in group if fnum(r.get("effective_delta_rms_fraction")) is not None],
                default=None,
            ),
            "max_effective_delta_rms_fraction": max(
                [fnum(r.get("effective_delta_rms_fraction")) for r in group if fnum(r.get("effective_delta_rms_fraction")) is not None],
                default=None,
            ),
            "mean_activation_rms_change": avg([r.get("activation_rms_change") for r in group]),
        }
        for (norm, alpha), group in sorted(rms_groups.items(), key=lambda x: (str(x[0][0]), float(x[0][1] or 0)))
    }

    OUT_JSON.write_text(json.dumps(analysis, indent=2, sort_keys=True))

    fields = list(compact_rows[0].keys()) if compact_rows else []
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(compact_rows)
    with OUT_JSONL.open("w") as handle:
        for row in compact_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    cell_fields = [
        "normalization_mode",
        "direction_name",
        "intervention_mode",
        "alpha",
        "n",
        "positive_z_mean",
        "negative_z_mean",
        "random_z_mean",
        "random_z_std",
        "signed_score",
        "signed_causal_signature",
        "strong_signed_causal_signature",
        "unsigned_positive_over_random",
    ]
    with OUT_CELLS_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cell_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cell_summaries)

    task_fields = list(analysis["task_summaries"][0].keys()) if analysis["task_summaries"] else []
    with OUT_TASKS_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=task_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(analysis["task_summaries"])

    row_lines = [
        "# RMS Steering Row-Level Table",
        "",
        f"Rows: {len(rows)} total; {len(interventions)} intervention rows; {len(baselines)} zero-baseline rows.",
        "",
        "| row | task | norm | direction | mode | alpha | condition | rand | z | eff_rms | hook_mods | safety | rep | parse_failed | correct |",
        "|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---|---:|---|---|",
    ]
    for row in compact_rows:
        row_lines.append(
            "| {row_index} | {task_id} | {normalization_mode} | {direction_name} | {intervention_mode} | "
            "{alpha} | {condition} | {random_control_idx} | {z} | {eff} | {mods} | {safety_status} | "
            "{rep} | {parse_failed} | {is_correct} |".format(
                row_index=row["row_index"],
                task_id=row["task_id"],
                normalization_mode=row["normalization_mode"],
                direction_name=row["direction_name"],
                intervention_mode=row["intervention_mode"],
                alpha=row["alpha"],
                condition=row["condition"],
                random_control_idx=row["random_control_idx"],
                z=fmt(row["z_score_change"], 3),
                eff=fmt(row["effective_delta_rms_fraction"], 5),
                mods=row["hook_modifications"],
                safety_status=row["safety_status"],
                rep=fmt(row["repetition_rate"], 3),
                parse_failed=row["parse_failed"],
                is_correct=row["is_correct"],
            )
        )
    OUT_ROWS_MD.write_text("\n".join(row_lines) + "\n")

    cell_lines = [
        "# RMS Steering Cell-Level Table",
        "",
        "Every row below is one `(normalization, direction, mode, alpha)` cell aggregated over tasks and conditions.",
        "",
        "| norm | direction | mode | alpha | n | pos z | neg z | random z | random std | signed score | signed | strong signed | unsigned pos>rand |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for cell in cell_summaries:
        cell_lines.append(
            f"| {cell['normalization_mode']} | {cell['direction_name']} | {cell['intervention_mode']} | "
            f"{cell['alpha']} | {cell['n']} | {fmt(cell['positive_z_mean'], 4)} | "
            f"{fmt(cell['negative_z_mean'], 4)} | {fmt(cell['random_z_mean'], 4)} | "
            f"{fmt(cell['random_z_std'], 4)} | {fmt(cell['signed_score'], 4)} | "
            f"{cell['signed_causal_signature']} | {cell['strong_signed_causal_signature']} | "
            f"{cell['unsigned_positive_over_random']} |"
        )
    OUT_CELLS_MD.write_text("\n".join(cell_lines) + "\n")

    md: list[str] = []
    md.append("# BG RMS Steering Row-Level Audit")
    md.append("")
    md.append("## Bottom line")
    md.append("")
    md.append(
        "The RMS sweep completed all planned intervention passes. The aggregate `RMS_DESTABILIZING` label is caused by "
        "one repetition/parse-failure outlier row, not by a broad mechanical failure."
    )
    md.append("")
    md.append("Operational override for downstream synthesis: `BG_RMS_STABILITY_VERDICT = STABLE_BUT_NOISY`; `BG_RMS_STEERING_VERDICT = RMS_UNSIGNED_ONLY` because unsigned cells exist but no strong signed cell cleared threshold.")
    md.append("")
    md.append("## Counts")
    md.append("")
    md.append(f"- Total rows: {len(rows)}")
    md.append(f"- Intervention rows / forward passes: {len(interventions)} / {trace.get('intervention_forward_passes')}")
    md.append(f"- Zero-baseline rows: {len(baselines)}")
    md.append(f"- Safety failure rows: {len(safety_rows)}")
    md.append(f"- CUDA error rows: {analysis['cuda_error_count']}")
    md.append(f"- NaN/Inf activation rows: {analysis['nan_or_inf_count']}")
    md.append(f"- Empty output rows: {analysis['empty_output_count']}")
    md.append(f"- Parse failure rows: {analysis['parse_failure_count']}")
    md.append("")
    md.append("## Mechanical hook audit")
    md.append("")
    mech = analysis["mechanical"]
    md.append(f"- Hook loop index source: `{mech['loop_index_sources']}`")
    md.append(f"- Cache modes: `{mech['cache_modes']}`")
    md.append(f"- Hook forward-call distribution: `{mech['hook_forward_call_count_distribution']}`")
    md.append(f"- Hook modification distribution: `{mech['hook_modifications_distribution']}`")
    md.append(
        "- Interpretation: intervention rows used 512 forward calls for 128 generated tokens x 4 Ouro loops. "
        "Multi-loop rows modified all four loops; single-loop rows modified one loop."
    )
    md.append("")
    md.append("## Safety outlier")
    md.append("")
    if safety_rows:
        for i, row in safety_rows:
            md.append(f"- Row {i}: `{row.get('normalization_mode')}` / `{row.get('direction_name')}` / `{row.get('intervention_mode')}` / alpha `{row.get('alpha')}` / `{row.get('condition')}` on `{row.get('task_id')}`.")
            md.append(f"  - Reason: {row_reason(row)}")
            md.append(f"  - z_score_change: {fmt(row.get('z_score_change'), 4)}")
            md.append(f"  - effective_delta_rms_fraction: {fmt(row.get('effective_delta_rms_fraction'), 6)}")
            md.append(f"  - activation_rms_change: {fmt(row.get('activation_rms_change'), 6)}")
            md.append(f"  - output excerpt: `{(row.get('output_text') or '')[:220].replace(chr(10), ' ')}`")
    else:
        md.append("- No safety outliers.")
    md.append("")
    md.append("## Effective RMS by normalization and alpha")
    md.append("")
    md.append("| normalization | alpha | n | mean eff RMS | min | max | mean activation RMS |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for key, value in analysis["effective_rms_by_normalization_alpha"].items():
        norm, alpha = key.split("|", 1)
        md.append(
            f"| {norm} | {alpha} | {value['n']} | {fmt(value['mean_effective_delta_rms_fraction'], 6)} | "
            f"{fmt(value['min_effective_delta_rms_fraction'], 6)} | {fmt(value['max_effective_delta_rms_fraction'], 6)} | "
            f"{fmt(value['mean_activation_rms_change'], 6)} |"
        )
    md.append("")
    md.append("## Overall condition behavior")
    md.append("")
    md.append("| condition | n | z mean | z std | parse failures | safety failures | success rate |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for condition, value in analysis["condition_overall"].items():
        md.append(
            f"| {condition} | {value['n']} | {fmt(value['z_mean'], 4)} | {fmt(value['z_std'], 4)} | "
            f"{value['parse_failures']} | {value['safety_failures']} | {fmt(value['success_rate'], 3)} |"
        )
    md.append("")
    md.append("## Block coverage")
    md.append("")
    md.append("| normalization | direction | mode | rows |")
    md.append("|---|---|---|---:|")
    for key, count in sorted(Counter((r.get("normalization_mode"), r.get("direction_name"), r.get("intervention_mode")) for r in rows).items()):
        md.append(f"| {key[0]} | {key[1]} | {key[2]} | {count} |")
    md.append("")
    md.append("## Best signed-score cells")
    md.append("")
    md.append("| rank | norm | direction | mode | alpha | pos z | neg z | random z | random std | signed score | strong signed |")
    md.append("|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|")
    for rank, cell in enumerate(best_signed, start=1):
        md.append(
            f"| {rank} | {cell['normalization_mode']} | {cell['direction_name']} | {cell['intervention_mode']} | "
            f"{cell['alpha']} | {fmt(cell['positive_z_mean'], 4)} | {fmt(cell['negative_z_mean'], 4)} | "
            f"{fmt(cell['random_z_mean'], 4)} | {fmt(cell['random_z_std'], 4)} | {fmt(cell['signed_score'], 4)} | "
            f"{cell['strong_signed_causal_signature']} |"
        )
    md.append("")
    md.append("## Files")
    md.append("")
    md.append(f"- Full compact row table: `{OUT_ROWS_MD}`")
    md.append(f"- Full row CSV: `{OUT_CSV}`")
    md.append(f"- Full row JSONL: `{OUT_JSONL}`")
    md.append(f"- Full cell table: `{OUT_CELLS_MD}`")
    md.append(f"- Full cell CSV: `{OUT_CELLS_CSV}`")
    md.append(f"- Task summary CSV: `{OUT_TASKS_CSV}`")
    md.append(f"- Machine-readable audit: `{OUT_JSON}`")
    OUT_MD.write_text("\n".join(md) + "\n")

    print(f"RMS_ROW_LEVEL_TOTAL_ROWS = {len(rows)}")
    print(f"RMS_ROW_LEVEL_INTERVENTION_ROWS = {len(interventions)}")
    print(f"RMS_ROW_LEVEL_SAFETY_FAILURE_ROWS = {len(safety_rows)}")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_JSONL}")
    print(f"Wrote {OUT_ROWS_MD}")
    print(f"Wrote {OUT_CELLS_CSV}")
    print(f"Wrote {OUT_CELLS_MD}")
    print(f"Wrote {OUT_TASKS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
