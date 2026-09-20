from __future__ import annotations

from bg_dualanchor_arch_loop_v3_common import OUT_ROOT, RUN_JSON, read_json, report_lines, safe_float, write_json, write_md


def main() -> int:
    inventory = read_json(OUT_ROOT / "inventory.json", {}) or {}
    suite = read_json(OUT_ROOT / "task_suite.json", {}) or {}
    run = read_json(RUN_JSON, {}) or {}
    reproduction = read_json(OUT_ROOT / "cached_reproduction.json", {}) or {}
    terminal = read_json(OUT_ROOT / "terminal_confidence.json", {}) or {}
    l47 = read_json(OUT_ROOT / "l47_ablation.json", {}) or {}
    threshold = read_json(OUT_ROOT / "threshold_budget.json", {}) or {}
    lineage = read_json(OUT_ROOT / "lineage_recovery.json", {}) or {}
    escalation = read_json(OUT_ROOT / "perturbation_escalation.json", {}) or {}
    hard = read_json(OUT_ROOT / "hard_slices.json", {}) or {}
    carry = read_json(OUT_ROOT / "prompt_carry_reference.json", {}) or {}
    readiness = read_json(OUT_ROOT / "phase2a_readiness.json", {}) or {}
    top = {
        "BG_DUALANCHOR_ARCH_LOOP_V3_INVENTORY_VERDICT": inventory.get("BG_DUALANCHOR_ARCH_LOOP_V3_INVENTORY_VERDICT", "INSUFFICIENT"),
        "BG_DUALANCHOR_ARCH_LOOP_V3_TASK_SUITE_VERDICT": suite.get("BG_DUALANCHOR_ARCH_LOOP_V3_TASK_SUITE_VERDICT", "INSUFFICIENT"),
        "BG_DUALANCHOR_ARCH_LOOP_V3_RUN_VERDICT": run.get("BG_DUALANCHOR_ARCH_LOOP_V3_RUN_VERDICT", "INSUFFICIENT"),
        "BG_DUALANCHOR_ARCH_LOOP_V3_REPRODUCTION_VERDICT": reproduction.get("BG_DUALANCHOR_ARCH_LOOP_V3_REPRODUCTION_VERDICT", "INSUFFICIENT"),
        "BG_DUALANCHOR_TERMINAL_CONFIDENCE_V3_VERDICT": terminal.get("BG_DUALANCHOR_TERMINAL_CONFIDENCE_V3_VERDICT", "INSUFFICIENT"),
        "BG_DUALANCHOR_L47_ABLATION_V3_VERDICT": l47.get("BG_DUALANCHOR_L47_ABLATION_V3_VERDICT", "INSUFFICIENT"),
        "BG_DUALANCHOR_THRESHOLD_BUDGET_V3_VERDICT": threshold.get("BG_DUALANCHOR_THRESHOLD_BUDGET_V3_VERDICT", "INSUFFICIENT"),
        "BG_DUALANCHOR_LINEAGE_RECOVERY_V3_VERDICT": lineage.get("BG_DUALANCHOR_LINEAGE_RECOVERY_V3_VERDICT", "INSUFFICIENT"),
        "BG_DUALANCHOR_PERTURBATION_ESCALATION_V3_VERDICT": escalation.get("BG_DUALANCHOR_PERTURBATION_ESCALATION_V3_VERDICT", "INSUFFICIENT"),
        "BG_DUALANCHOR_HARD_SLICE_V3_VERDICT": hard.get("BG_DUALANCHOR_HARD_SLICE_V3_VERDICT", "INSUFFICIENT"),
        "BG_DUALANCHOR_PROMPT_CARRY_REFERENCE_V3_VERDICT": carry.get("BG_DUALANCHOR_PROMPT_CARRY_REFERENCE_V3_VERDICT", "INSUFFICIENT"),
        "BG_DUALANCHOR_PHASE2A_READINESS_V3_VERDICT": readiness.get("BG_DUALANCHOR_PHASE2A_READINESS_V3_VERDICT", "INSUFFICIENT"),
    }
    terminal_status = top["BG_DUALANCHOR_TERMINAL_CONFIDENCE_V3_VERDICT"]
    hard_status = top["BG_DUALANCHOR_HARD_SLICE_V3_VERDICT"]
    if top["BG_DUALANCHOR_PHASE2A_READINESS_V3_VERDICT"] == "SURVIVAL_READY_TERMINAL_CONFIDENCE_GATED":
        status = "ARCHITECTURE_LOOPED_SURVIVAL_READY_TERMINAL_CONFIDENCE_GATED"
    elif top["BG_DUALANCHOR_PHASE2A_READINESS_V3_VERDICT"] == "READY_WITH_TERMINAL_DEFER":
        status = "ARCHITECTURE_LOOPED_SURVIVAL_READY_TERMINAL_DEFER_REQUIRED"
    elif hard_status in {"TERMINAL_WEAK_ON_REWARD_DIVERSE", "TIE_HEAVY_INFLATED"}:
        status = "ARCHITECTURE_LOOPED_SURVIVAL_READY_TERMINAL_WEAK_ON_HARD_SLICE"
    elif safe_float(run.get("stage_oracle_retention"), 0.0) < 0.95:
        status = "ARCHITECTURE_LOOPED_SURVIVAL_WEAK"
    else:
        status = "NEEDS_MORE_SCALE"
    top["DUALANCHOR_ARCHITECTURE_LOOPED_V3_STATUS"] = status
    payload = {
        **top,
        "run_headline": {
            "tasks": (run.get("task_summary") or {}).get("count"),
            "stage_oracle_retention": run.get("stage_oracle_retention"),
            "terminal_oracle_retained": (run.get("task_summary") or {}).get("terminal_oracle_retained"),
            "terminal_forced_top1_oracle": (run.get("task_summary") or {}).get("terminal_forced_top1_oracle"),
            "terminal_reward_diverse": (run.get("task_summary") or {}).get("terminal_reward_diverse"),
            "positive_oracle": (run.get("task_summary") or {}).get("positive_oracle"),
        },
        "files_created": [str(p.relative_to(OUT_ROOT.parent)) for p in sorted(OUT_ROOT.glob("*")) if p.is_file()],
        "commands_run": [
            "py_compile for v3 scripts",
            "inventory",
            "task suite",
            "run_bg_dualanchor_arch_loop_v3.py",
            "reproduction",
            "terminal confidence",
            "L47 ablation",
            "threshold/budget",
            "lineage/recovery",
            "perturbation escalation trigger audit",
            "hard slices",
            "prompt carry reference",
            "phase2a readiness",
            "synthesis",
        ],
        "blockers": [],
    }
    write_json(OUT_ROOT / "summary.json", payload)
    write_json(OUT_ROOT / "analysis.json", payload)
    lines = report_lines(
        "DualAnchor Architecture-Looped Stratified Probe v3",
        "DUALANCHOR_ARCHITECTURE_LOOPED_V3_STATUS",
        status,
        [
            ("Top Lines", [f"- {k} = `{v}`" for k, v in top.items()]),
            ("Run Headline", [f"- {k}: `{v}`" for k, v in payload["run_headline"].items()]),
            ("Decision", ["- Lock Phase 2a as DualAnchor architecture-looped selection with active nonterminal L47 and terminal confidence gate if readiness is confidence-gated or defer-ready.", "- No steering, autoregressive fork/carry, or compute savings are claimed."]),
            ("Files Created", [f"- `{name}`" for name in payload["files_created"]]),
        ],
    )
    write_md(OUT_ROOT / "summary.md", lines)
    write_md(OUT_ROOT / "analysis.md", lines)
    print(f"DUALANCHOR_ARCHITECTURE_LOOPED_V3_STATUS = {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

