"""Audit BG/tap experiment backlog against current docs and artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from code_branch_pilot_lib import PROJECT_ROOT, REPORT_DIR, repo_path, write_json


OUTPUT_JSON = REPORT_DIR / "bg_experiment_backlog_audit_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "bg_experiment_backlog_audit_2026-05-17.md"


def exists(path: str) -> bool:
    return (PROJECT_ROOT / path).exists()


def item(name: str, status: str, reason: str, paths: list[str], should_run: bool) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "reason": reason,
        "artifact_paths": [path for path in paths if exists(path)],
        "missing_paths": [path for path in paths if not exists(path)],
        "should_still_run": should_run,
    }


def main() -> None:
    docs = sorted((PROJECT_ROOT / "shared/docs" / "evaluator").glob("*.md"))
    reports_md = sorted(REPORT_DIR.glob("*.md"))
    reports_json = sorted(REPORT_DIR.glob("*.json"))
    backlog = [
        item("HH locus / centered bias decomposition", "DONE", "Covered by HH locus, L1/alpha, and centered-input diagnostics in evaluator notes.", [
            "opi/taps/probes/probe_converged_tap_inputs.md",
            "docs/evaluator/evaluator_domain_transfer_notes.md",
        ], False),
        item("RLTT / Thinking layer geometry", "DONE", "Layer 24/36/47 RLTT vs Thinking geometry and old-head compatibility were recorded.", [
            "docs/evaluator/evaluator_domain_transfer_notes.md",
            "rpe/evaluator/hh_layer_states_200_rltt.pt",
            "rpe/evaluator/hh_layer_states_200_thinking.pt",
        ], False),
        item("Math layer geometry", "DONE", "Math text geometry was measured and summarized.", [
            "opi/taps/probes/math_layer_geometry_rltt.md",
            "docs/evaluator/math_bg_gate_pilot_2026-05-15.md",
        ], False),
        item("Math validity / truncation probe", "DONE", "Old mixed math pilot was marked truncation-confounded.", [
            "opi/taps/probes/math_data_validity_2026-05-16.md",
            "docs/evaluator/post_v10_synthesis_2026-05-16_v5_clean_gsm8k_code_next.md",
        ], False),
        item("Clean GSM8K micro", "DONE", "Clean n=5 micro showed preliminary HH-trained transfer.", [
            "opi/taps/probes/clean_gsm8k_extreme_transfer_2026-05-16.json",
        ], False),
        item("Expanded clean GSM8K + GRU control", "DONE", "Expanded clean GSM8K linear transfer was GOOD and GRU control was weak.", [
            "opi/taps/probes/clean_gsm8k_expanded_transfer_gru_2026-05-16_summary.md",
        ], False),
        item("Code v1 all-correct collapse", "OBSOLETE", "Superseded by patched v2/v2-mini candidate-stage harvesting and strict label taxonomy.", [
            "opi/taps/probes/code_branch_pilot_2026-05-16_summary.md",
        ], False),
        item("Code v2 pre-fix dirty diagnostic", "OBSOLETE", "Historical pre-fix diagnostic; wrapper/prose and label taxonomy were fixed.", [
            "opi/taps/probes/code_branch_pilot_v2_2026-05-16_summary.md",
        ], False),
        item("Harness/wrapper fixes", "DONE", "Wrapper/prose rejection, MBPP signature fixes, and safety taxonomy were validated.", [
            "opi/taps/probes/code_branch_v2_harness_agent_fixes_2026-05-16.md",
            "opi/taps/probes/code_branch_pilot_v2_mini_patched_2026-05-16_summary.md",
        ], False),
        item("Patched code v2-mini", "DONE", "Runnable diagnostic code transfer produced a GOOD result.", [
            "opi/taps/probes/code_branch_pilot_v2_mini_patched_2026-05-16_summary.md",
        ], False),
        item("Strict-clean task screening", "DONE", "Original screening found six strict-clean tasks; expansion found ten more.", [
            "opi/taps/probes/code_strict_clean_screening_2026-05-17_summary.md",
            "opi/taps/probes/code_strict_clean_screening_expansion_2026-05-17.md",
        ], False),
        item("Strict-clean code transfer with HH-trained taps", "DONE", "HH-trained transfer on OLD6 strict-clean code was WEAK.", [
            "opi/taps/probes/code_strict_clean_transfer_2026-05-17_summary.md",
        ], False),
        item("Code-specific tiny-head control", "DONE", "Code-specific tiny heads were GOOD on OLD6 strict-clean.", [
            "opi/taps/probes/code_specific_tiny_head_control_2026-05-17.md",
        ], False),
        item("Expanded strict-clean HH-vs-code projection comparison", "DONE", "ALL16 comparison found CODE_SPECIFIC_ADVANTAGE with both families GOOD.", [
            "opi/taps/probes/expanded_strict_clean_code_projection_comparison_2026-05-17_summary.md",
        ], False),
        item("Code-trained taps on HH inverse transfer", "DONE", "Single held-out split inverse transfer returned GOOD, but all-200 diagnostic was weaker.", [
            "opi/taps/probes/code_trained_taps_on_hh_2026-05-17.md",
        ], False),
        item("Random 20-pair HH split inverse-transfer check", "DONE", "Ten random 20-pair HH splits measured stability of code-vs-HH heads.", [
            "opi/taps/probes/code_trained_vs_hh_trained_random20_hh_splits_2026-05-17.md",
        ], False),
        item("Fixed-config inverse transfer audit", "ACTIVE", "Best-of-config selection is now suspect; fixed-config cross-domain audit is this run.", [
            "opi/taps/probes/bg_fixed_config_cross_domain_audit_2026-05-17.md",
        ], True),
        item("Reasoning / logic branch dataset pilot", "ACTIVE", "Small multiple-choice branch pilot is requested in this run.", [
            "opi/taps/probes/reasoning_branch_pilot_2026-05-17.md",
        ], True),
        item("Controller-policy simulator", "DEFERRED", "Useful after fixed-config/domain matrix clarifies head disagreement patterns.", [], True),
        item("Full HH capture / full split", "DEFERRED", "Not needed for this artifact audit; useful for later paper-scale HH readout.", [
            "rpe/evaluator/hh_layer_states_200_rltt.pt",
        ], True),
        item("MATH gate-scale", "BLOCKED", "Deferred by local budget, generation verbosity, and source-yield constraints; no MATH generation in this run.", [
            "docs/evaluator/math_bg_gate_pilot_2026-05-15.md",
        ], False),
        item("GRU temporal controls", "OBSOLETE", "GRU is no longer default; keep as escalation/control only.", [
            "opi/taps/probes/clean_gsm8k_expanded_gru_control_2026-05-16.md",
        ], False),
    ]
    blocked_required = [row for row in backlog if row["status"] == "BLOCKED" and row["should_still_run"]]
    verdict = "BLOCKED" if blocked_required else "READY"
    payload = {
        "bg_backlog_audit_verdict": verdict,
        "summary": {
            "BG_BACKLOG_AUDIT_VERDICT": verdict,
            "docs_scanned": [repo_path(path) for path in docs],
            "probe_md_count": len(reports_md),
            "probe_json_count": len(reports_json),
            "status_counts": {status: sum(1 for row in backlog if row["status"] == status) for status in sorted({row["status"] for row in backlog})},
            "should_still_run": [row["name"] for row in backlog if row["should_still_run"]],
        },
        "backlog": backlog,
        "outputs": {"json": repo_path(OUTPUT_JSON), "md": repo_path(OUTPUT_MD)},
    }
    write_json(OUTPUT_JSON, payload)
    lines = ["# BG Experiment Backlog Audit", "", f"BG_BACKLOG_AUDIT_VERDICT = {verdict}", ""]
    lines.append("| # | experiment | status | should run | reason | artifacts |")
    lines.append("| ---: | --- | --- | --- | --- | --- |")
    for idx, row in enumerate(backlog, 1):
        artifacts = ", ".join(f"`{path}`" for path in row["artifact_paths"]) or "none"
        lines.append(f"| {idx} | {row['name']} | `{row['status']}` | `{row['should_still_run']}` | {row['reason']} | {artifacts} |")
    lines.extend(["", "## Status Counts", "", f"`{payload['summary']['status_counts']}`", ""])
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"BG_BACKLOG_AUDIT_VERDICT = {verdict}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    if verdict == "BLOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
