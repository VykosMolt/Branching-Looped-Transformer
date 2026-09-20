"""Probe true fork/carry feasibility for Branch Generator v1.

This is a bounded mechanical inspection/smoke stage. It does not train Ouro,
run ARC actions, or use wrapper/local-agent code. If true latent resume is not
validated, later stages must continue with hook_intervention_per_branch.
"""
from __future__ import annotations

import inspect
import time
from typing import Any

from bg_branch_generator_v1_common import (
    TRUE_FORK_CSV,
    TRUE_FORK_JSON,
    TRUE_FORK_MD,
    ensure_bgv1_root,
    rel,
    write_csv,
    write_json,
    write_md,
)


def inspect_true_fork_substrate() -> dict[str, Any]:
    out: dict[str, Any] = {
        "true_fork_class_present": False,
        "placeholder_detected": False,
        "exact_layer_resume_api_detected": False,
        "loop_boundary_resume_api_detected": False,
        "cache_branching_api_detected": False,
        "use_cache_false_hook_path_available": True,
        "notes": [],
    }
    try:
        import src.evaluator.bg_hidden_branching as branching

        cls = getattr(branching, "TrueLatentForkCarry", None)
        out["true_fork_class_present"] = cls is not None
        source = inspect.getsource(cls) if cls is not None else ""
        lowered = source.lower()
        out["placeholder_detected"] = "notimplementederror" in lowered or "not implemented" in lowered
        out["exact_layer_resume_api_detected"] = "resume" in lowered and "target_layer" in lowered and not out["placeholder_detected"]
        out["loop_boundary_resume_api_detected"] = "loop_boundary" in lowered and "resume" in lowered and not out["placeholder_detected"]
        out["cache_branching_api_detected"] = "past_key_values" in lowered and "branch" in lowered and not out["placeholder_detected"]
        out["source_excerpt"] = source[:1400]
        if out["placeholder_detected"]:
            out["notes"].append("TrueLatentForkCarry is present but still placeholder/NotImplemented.")
    except Exception as exc:
        out["notes"].append(f"inspection_error: {type(exc).__name__}: {exc}")
    return out


def verdict_for(info: dict[str, Any]) -> str:
    if info.get("exact_layer_resume_api_detected") and info.get("cache_branching_api_detected"):
        return "TRUE_FORK_CARRY_READY"
    if info.get("loop_boundary_resume_api_detected"):
        return "LOOP_BOUNDARY_FORK_PARTIAL"
    if info.get("true_fork_class_present") or info.get("use_cache_false_hook_path_available"):
        return "HOOK_FALLBACK_ONLY"
    if info.get("notes"):
        return "STATE_HANDLING_BLOCKED"
    return "BLOCKED"


def main() -> int:
    started = time.time()
    ensure_bgv1_root()
    info = inspect_true_fork_substrate()
    verdict = verdict_for(info)
    rows = [
        {
            "mode": "no_resume_inspection",
            "state_shape_correctness": "not_applicable",
            "branch_lineage_correctness": "not_applicable",
            "hidden_distinctness_over_depth": "not_measured",
            "logit_kl_spread": "not_measured",
            "output_diversity": "not_measured",
            "cache_status": "not_validated",
            "result": verdict,
        }
    ]
    payload = {
        "BG_TRUE_FORK_CARRY_PROBE_V1_VERDICT": verdict,
        "verdict": verdict,
        "inspection": info,
        "rows": rows,
        "mechanical_smoke_run": False,
        "true_fork_carry_usable_for_generation": verdict == "TRUE_FORK_CARRY_READY",
        "fallback_branch_method": "hook_intervention_per_branch",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(TRUE_FORK_JSON, payload)
    write_csv(TRUE_FORK_CSV, rows)
    lines = [
        "# True Fork/Carry Probe V1",
        "",
        f"BG_TRUE_FORK_CARRY_PROBE_V1_VERDICT = {verdict}",
        "",
        f"- true_fork_class_present: `{info.get('true_fork_class_present')}`",
        f"- placeholder_detected: `{info.get('placeholder_detected')}`",
        f"- exact_layer_resume_api_detected: `{info.get('exact_layer_resume_api_detected')}`",
        f"- loop_boundary_resume_api_detected: `{info.get('loop_boundary_resume_api_detected')}`",
        f"- cache_branching_api_detected: `{info.get('cache_branching_api_detected')}`",
        f"- fallback_branch_method: `hook_intervention_per_branch`",
        "",
        "No selector or branch-generator readiness claim uses this diagnostic. Later stages may use true fork/carry only if the verdict is TRUE_FORK_CARRY_READY.",
        "",
        f"JSON: `{rel(TRUE_FORK_JSON)}`",
    ]
    write_md(TRUE_FORK_MD, lines)
    print(f"BG_TRUE_FORK_CARRY_PROBE_V1_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
