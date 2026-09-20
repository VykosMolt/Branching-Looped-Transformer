from __future__ import annotations

from bg_dualanchor_arch_loop_v3_common import OUT_ROOT, RUN_JSON, read_json, report_lines, safe_float, write_json, write_md


def main() -> int:
    run = read_json(RUN_JSON, {}) or {}
    terminal = read_json(OUT_ROOT / "terminal_confidence.json", {}) or {}
    l47 = read_json(OUT_ROOT / "l47_ablation.json", {}) or {}
    threshold = read_json(OUT_ROOT / "threshold_budget.json", {}) or {}
    lineage = read_json(OUT_ROOT / "lineage_recovery.json", {}) or {}
    hard = read_json(OUT_ROOT / "hard_slices.json", {}) or {}
    task_summary = run.get("task_summary") or {}
    survival_ok = safe_float(run.get("stage_oracle_retention"), 0.0) >= 0.95 and safe_float(task_summary.get("terminal_oracle_retained"), 0.0) >= 0.95
    terminal_verdict = terminal.get("BG_DUALANCHOR_TERMINAL_CONFIDENCE_V3_VERDICT")
    hard_verdict = hard.get("BG_DUALANCHOR_HARD_SLICE_V3_VERDICT")
    if survival_ok and terminal_verdict in {"TERMINAL_TOP1_READY", "TERMINAL_CONFIDENCE_GATED_READY"}:
        verdict = "SURVIVAL_READY_TERMINAL_CONFIDENCE_GATED"
    elif survival_ok and terminal_verdict in {"TERMINAL_DEFER_REQUIRED", "TERMINAL_WEAK_ON_HARD_SLICE"}:
        verdict = "READY_WITH_TERMINAL_DEFER"
    elif survival_ok:
        verdict = "NEEDS_TERMINAL_WORK"
    else:
        verdict = "NOT_READY"
    payload = {
        "BG_DUALANCHOR_PHASE2A_READINESS_V3_VERDICT": verdict,
        "locked_baseline_candidate": {
            "branch_schedule": "L1_24 -> L1_36 -> L1_47 -> L2_24 -> L2_36 -> L2_47 -> L3_24 -> L3_36 -> L3_47 -> L4_24 -> L4_36 -> terminal L4_47",
            "scoring": "DualAnchor MIX_CODE_REASONING + MIX_OBJECTIVE_ALL",
            "threshold": "mean_floor_very_loose",
            "budget": 8,
            "terminal": "confidence-gated top1, otherwise defer/keep terminal survivors",
            "l47": "active in nonterminal loops",
            "lineage_logging": True,
        },
        "inputs": {
            "run_status": run.get("status"),
            "stage_oracle_retention": run.get("stage_oracle_retention"),
            "terminal_confidence": terminal_verdict,
            "l47": l47.get("BG_DUALANCHOR_L47_ABLATION_V3_VERDICT"),
            "threshold": threshold.get("BG_DUALANCHOR_THRESHOLD_BUDGET_V3_VERDICT"),
            "lineage": lineage.get("BG_DUALANCHOR_LINEAGE_RECOVERY_V3_VERDICT"),
            "hard_slices": hard_verdict,
        },
        "caveats": ["No steering tested.", "No autoregressive fork/carry claim.", "No compute-savings claim."],
    }
    write_json(OUT_ROOT / "phase2a_readiness.json", payload)
    lines = report_lines("DualAnchor Phase 2a Readiness v3", "BG_DUALANCHOR_PHASE2A_READINESS_V3_VERDICT", verdict, [("Inputs", [f"- {k}: `{v}`" for k, v in payload["inputs"].items()]), ("Locked Baseline Candidate", [f"- {k}: `{v}`" for k, v in payload["locked_baseline_candidate"].items()]), ("Caveats", [f"- {x}" for x in payload["caveats"]])])
    write_md(OUT_ROOT / "phase2a_readiness.md", lines)
    print(f"BG_DUALANCHOR_PHASE2A_READINESS_V3_VERDICT = {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

