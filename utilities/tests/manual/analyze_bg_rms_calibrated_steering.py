"""Analyze RMS-calibrated BG steering probe results."""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from bg_preconsolidation_common import OUT_ROOT, finite, rel, summarize_cell, write_json, write_md


TRACE_JSON = OUT_ROOT / "rms_steering_traces.json"
TRACE_PARTIAL = OUT_ROOT / "rms_steering_traces.partial.json"
OUT_JSON = OUT_ROOT / "rms_steering_analysis.json"
OUT_MD = OUT_ROOT / "rms_steering_analysis.md"


def load_json(path, default=None):
    import json

    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def best_score(cells: dict[str, Any], norm_filter: str | None = None) -> tuple[str, float]:
    best_key = "none"
    best = -1e9
    for key, value in cells.items():
        norm = key.split("|")[0]
        if norm_filter is not None and norm != norm_filter:
            continue
        score = finite(value.get("signed_score"))
        if score > best:
            best_key = key
            best = score
    return best_key, best


def main() -> int:
    started = time.time()
    traces = load_json(TRACE_JSON, load_json(TRACE_PARTIAL, {}))
    rows = list(traces.get("rows") or [])
    grouped: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("normalization_mode")),
                str(row.get("direction_name")),
                str(row.get("intervention_mode") or row.get("mode")),
                finite(row.get("alpha")),
            )
        ].append(row)
    cells = {f"{norm}|{direction}|{mode}|{alpha:g}": summarize_cell(group) for (norm, direction, mode, alpha), group in sorted(grouped.items())}
    intervention = [row for row in rows if row.get("condition") != "zero_baseline"]
    rms_cells = {k: v for k, v in cells.items() if k.startswith("RMS_NORMALIZED|")}
    strong_rms = [k for k, v in rms_cells.items() if v.get("strong_signed_causal_signature") and v.get("stable")]
    unsigned_rms = [k for k, v in rms_cells.items() if v.get("unsigned_effect") and v.get("stable")]
    safety_failures = [row for row in intervention if row.get("safety_status") == "DESTABILIZING"]
    cuda_errors = sum(1 for row in intervention if row.get("cuda_error"))
    nan_rows = sum(1 for row in intervention if row.get("nan_or_inf_activations"))
    if not intervention:
        steering_verdict = "INSUFFICIENT"
    elif safety_failures or cuda_errors or nan_rows:
        steering_verdict = "RMS_DESTABILIZING"
    elif strong_rms:
        steering_verdict = "RMS_SIGNED_CAUSAL"
    elif unsigned_rms:
        steering_verdict = "RMS_UNSIGNED_ONLY"
    else:
        steering_verdict = "RMS_NO_EFFECT"

    rms_key, rms_score = best_score(cells, "RMS_NORMALIZED")
    l2_key, l2_score = best_score(cells, "L2_NORMALIZED_CONTROL")
    if rms_key == "none" or l2_key == "none":
        vs_l2 = "INSUFFICIENT"
    elif rms_score > l2_score + 0.10:
        vs_l2 = "RMS_BEATS_L2"
    elif l2_score > rms_score + 0.10:
        vs_l2 = "RMS_WORSE_THAN_L2"
    else:
        vs_l2 = "RMS_MATCHES_L2"

    if not intervention:
        stability = "INSUFFICIENT"
    elif cuda_errors or nan_rows or safety_failures:
        stability = "DESTABILIZING"
    elif any((v.get("parse_failed_rate") or 0.0) > 0.20 or (v.get("repetition_rate_mean") or 0.0) > 0.10 for v in rms_cells.values()):
        stability = "STABLE_BUT_NOISY"
    else:
        stability = "STABLE"

    analysis = {
        "BG_RMS_STEERING_VERDICT": steering_verdict,
        "BG_RMS_VS_L2_VERDICT": vs_l2,
        "BG_RMS_STABILITY_VERDICT": stability,
        "RMS_GAIN_OVER_L2": rms_score - l2_score if rms_key != "none" and l2_key != "none" else None,
        "best_rms_key": rms_key,
        "best_l2_key": l2_key,
        "best_rms_signed_score": rms_score,
        "best_l2_signed_score": l2_score,
        "strong_rms_cells": strong_rms,
        "unsigned_rms_cells": unsigned_rms,
        "safety_failure_count": len(safety_failures),
        "cuda_error_count": cuda_errors,
        "nan_or_inf_count": nan_rows,
        "row_count": len(rows),
        "intervention_forward_passes": traces.get("intervention_forward_passes"),
        "cells": cells,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, analysis)
    lines = [
        "# BG RMS-Calibrated Steering Analysis",
        "",
        f"BG_RMS_STEERING_VERDICT = {steering_verdict}",
        f"BG_RMS_VS_L2_VERDICT = {vs_l2}",
        f"BG_RMS_STABILITY_VERDICT = {stability}",
        "",
        f"- row_count: `{len(rows)}`",
        f"- intervention_forward_passes: `{traces.get('intervention_forward_passes')}`",
        f"- best_rms_key: `{rms_key}`",
        f"- best_l2_key: `{l2_key}`",
        f"- RMS_GAIN_OVER_L2: `{analysis['RMS_GAIN_OVER_L2']}`",
        f"- safety_failure_count: `{len(safety_failures)}`",
        "",
        "| cell | pos_z | neg_z | rand_z | rand_std | strong | unsigned | stable | eff_rms |",
        "|---|---:|---:|---:|---:|---|---|---|---:|",
    ]
    for key, value in sorted(cells.items()):
        lines.append(
            f"| `{key}` | {finite(value.get('positive_z_mean')):.4f} | {finite(value.get('negative_z_mean')):.4f} | "
            f"{finite(value.get('random_z_mean')):.4f} | {finite(value.get('random_z_std')):.4f} | "
            f"`{value.get('strong_signed_causal_signature')}` | `{value.get('unsigned_effect')}` | "
            f"`{value.get('stable')}` | {finite(value.get('effective_delta_rms_fraction_mean')):.5f} |"
        )
    write_md(OUT_MD, lines)
    print(f"BG_RMS_STEERING_VERDICT = {steering_verdict}")
    print(f"BG_RMS_VS_L2_VERDICT = {vs_l2}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
