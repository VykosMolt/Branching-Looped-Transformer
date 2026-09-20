"""Audit the interrupted BG Stage 2 v3 partial layer-hook trace."""
from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SRC_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_stage2_steering_2026-05-18"
OUT_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_stage2_layerhook_followup_2026-05-18"
TRACE_PATH = SRC_ROOT / "intervention_traces.partial.json"
PREFLIGHT_PATH = SRC_ROOT / "preflight.json"
OUT_JSON = OUT_ROOT / "partial_trace_audit.json"
OUT_MD = OUT_ROOT / "partial_trace_audit.md"


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


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def main() -> int:
    started = time.time()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    trace = load_json(TRACE_PATH)
    preflight = load_json(PREFLIGHT_PATH, {})
    if not isinstance(trace, dict):
        payload = {
            "BG_STAGE2_PARTIAL_TRACE_AUDIT_VERDICT": "BLOCKED",
            "blocker": f"partial trace unreadable or missing: {rel(TRACE_PATH)}",
            "elapsed_seconds": round(time.time() - started, 3),
        }
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# BG Stage 2 Partial Trace Audit", "", "BG_STAGE2_PARTIAL_TRACE_AUDIT_VERDICT = BLOCKED"])
        print("BG_STAGE2_PARTIAL_TRACE_AUDIT_VERDICT = BLOCKED")
        return 1

    rows = list(trace.get("rows") or [])
    intervention_rows = [row for row in rows if row.get("condition") != "zero_baseline"]
    safety_failures = [
        row
        for row in intervention_rows
        if row.get("safety_status") not in {None, "OK"}
        or row.get("cuda_error")
        or row.get("nan_or_inf_activations")
    ]
    hook_rows = [row for row in intervention_rows if row.get("mechanism") == "layer_hook_injection"]
    hook_sources = Counter(str(row.get("hook_loop_index_source")) for row in hook_rows)
    hook_count_consistency = []
    for row in hook_rows:
        out_len = int(row.get("output_length") or 0)
        loops = 4
        expected_forward_calls = out_len * loops
        mode = str(row.get("intervention_mode"))
        expected_mods = out_len * (4 if mode.startswith("multi_loop") else 1)
        hook_count_consistency.append(
            {
                "task_id": row.get("task_id"),
                "mode": mode,
                "alpha": row.get("alpha"),
                "condition": row.get("condition"),
                "output_length": out_len,
                "hook_forward_call_count": row.get("hook_forward_call_count"),
                "expected_forward_call_count": expected_forward_calls,
                "forward_count_matches": int(row.get("hook_forward_call_count") or -1) == expected_forward_calls,
                "hook_modifications": row.get("hook_modifications"),
                "expected_modifications": expected_mods,
                "modification_count_matches": int(row.get("hook_modifications") or -1) == expected_mods,
            }
        )
    forward_match_rate = (
        sum(1 for row in hook_count_consistency if row["forward_count_matches"]) / max(len(hook_count_consistency), 1)
    )
    mod_match_rate = (
        sum(1 for row in hook_count_consistency if row["modification_count_matches"]) / max(len(hook_count_consistency), 1)
    )

    rms_by_alpha: dict[str, list[float]] = defaultdict(list)
    z_by_cell: dict[tuple[str, float, str], list[float]] = defaultdict(list)
    for row in hook_rows:
        alpha = str(row.get("alpha"))
        rms_by_alpha[alpha].append(finite(row.get("activation_rms_change")))
        z_by_cell[(str(row.get("intervention_mode")), finite(row.get("alpha")), str(row.get("condition")))].append(
            finite(row.get("z_score_change"))
        )
    rms_summary = {alpha: mean(vals) for alpha, vals in sorted(rms_by_alpha.items()) if vals}
    z_summary = [
        {
            "mode": mode,
            "alpha": alpha,
            "condition": condition,
            "n": len(vals),
            "z_score_change_mean": mean(vals),
            "z_score_change_values": vals,
        }
        for (mode, alpha, condition), vals in sorted(z_by_cell.items())
    ]

    has_required_fields = bool(rows) and all(
        key in rows[0]
        for key in (
            "hook_forward_call_count",
            "hook_modifications",
            "hook_loop_index_source",
            "activation_rms_change",
            "z_score_change",
        )
    )
    mechanics_ready = (
        bool(hook_rows)
        and hook_sources.get("current_ut", 0) >= max(1, len(hook_rows) // 2)
        and forward_match_rate >= 0.90
        and mod_match_rate >= 0.90
        and not safety_failures
    )
    if mechanics_ready:
        verdict = "READY"
    elif trace and rows:
        verdict = "PARTIAL"
    else:
        verdict = "BLOCKED"

    payload = {
        "BG_STAGE2_PARTIAL_TRACE_AUDIT_VERDICT": verdict,
        "verdict": verdict,
        "trace_path": rel(TRACE_PATH),
        "preflight_path": rel(PREFLIGHT_PATH),
        "source_preflight_verdict": preflight.get("BG_STAGE2_PREFLIGHT_VERDICT"),
        "source_layer_hook_verdict": preflight.get("LAYER_HOOK_INJECTION_VERDICT"),
        "row_count": len(rows),
        "intervention_forward_passes": trace.get("intervention_forward_passes"),
        "targets_present": sorted({str(row.get("target_id")) for row in rows}),
        "modes_present": sorted({str(row.get("intervention_mode")) for row in rows}),
        "alphas_present": sorted({finite(row.get("alpha")) for row in rows}),
        "conditions_present": sorted({str(row.get("condition")) for row in rows}),
        "safety_failure_count": len(safety_failures),
        "nan_or_inf_count": sum(1 for row in rows if row.get("nan_or_inf_activations")),
        "cuda_error_count": sum(1 for row in rows if row.get("cuda_error")),
        "hook_loop_index_sources": dict(hook_sources),
        "hook_forward_count_match_rate": forward_match_rate,
        "hook_modification_count_match_rate": mod_match_rate,
        "hook_count_consistency_examples": hook_count_consistency[:20],
        "rms_change_by_alpha": rms_summary,
        "z_score_change_by_mode_alpha_condition": z_summary,
        "has_required_followup_fields": has_required_fields,
        "usable_for_followup_aggregation": verdict in {"READY", "PARTIAL"} and has_required_fields,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)

    lines = [
        "# BG Stage 2 Partial Trace Audit",
        "",
        f"BG_STAGE2_PARTIAL_TRACE_AUDIT_VERDICT = {verdict}",
        "",
        f"- row_count: `{len(rows)}`",
        f"- intervention_forward_passes: `{trace.get('intervention_forward_passes')}`",
        f"- targets_present: `{payload['targets_present']}`",
        f"- modes_present: `{payload['modes_present']}`",
        f"- alphas_present: `{payload['alphas_present']}`",
        f"- conditions_present: `{payload['conditions_present']}`",
        f"- safety_failure_count: `{len(safety_failures)}`",
        f"- nan_or_inf_count: `{payload['nan_or_inf_count']}`",
        f"- cuda_error_count: `{payload['cuda_error_count']}`",
        f"- hook_loop_index_sources: `{dict(hook_sources)}`",
        f"- hook_forward_count_match_rate: `{forward_match_rate:.3f}`",
        f"- hook_modification_count_match_rate: `{mod_match_rate:.3f}`",
        f"- usable_for_followup_aggregation: `{payload['usable_for_followup_aggregation']}`",
        "",
        "## RMS Change By Alpha",
        "",
        *[f"- alpha `{alpha}`: `{value:.8f}`" for alpha, value in rms_summary.items()],
        "",
        "## Preliminary Z Score Change",
        "",
        "| Mode | Alpha | Condition | n | mean z |",
        "|---|---:|---|---:|---:|",
    ]
    for row in z_summary:
        lines.append(
            f"| `{row['mode']}` | {row['alpha']} | `{row['condition']}` | {row['n']} | {row['z_score_change_mean']:.4f} |"
        )
    write_md(OUT_MD, lines)
    print(f"BG_STAGE2_PARTIAL_TRACE_AUDIT_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict in {"READY", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
