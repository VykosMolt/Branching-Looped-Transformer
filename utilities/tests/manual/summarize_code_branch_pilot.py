"""Write the stopped code-branch pilot summary and append doc notes."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from code_branch_pilot_lib import PROJECT_ROOT, REPORT_DIR, repo_path, write_json


SUMMARY_JSON = REPORT_DIR / "code_branch_pilot_2026-05-16_summary.json"
SUMMARY_MD = REPORT_DIR / "code_branch_pilot_2026-05-16_summary.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def recommended_next(verdicts: dict[str, str]) -> str:
    if verdicts["CODE_INTERFACE_VERDICT"] == "BLOCKED":
        return "fix_local_agent_invocation_before_code_pilot"
    if verdicts["CODE_TASKSET_VERDICT"] == "BLOCKED":
        return "install_or_restore_humaneval_mbpp_taskset"
    if verdicts["CODE_GENERATION_VERDICT"] == "WRAPPER_BLOCKED":
        return "inspect_local_agent_wrapper_or_use_direct_route_only"
    if verdicts["CODE_TOURNAMENT_VERDICT"] == "TOO_FEW_TOURNAMENTS":
        return "tune_code_generation_modes_or_increase_tasks"
    if verdicts["CODE_TRANSFER_VERDICT"] == "GOOD":
        return "expand_code_branch_pilot_to_30_tournaments"
    if verdicts["CODE_TRANSFER_VERDICT"] == "WEAK":
        return "expand_code_pilot_or_add_code_specific_training_control"
    if verdicts["CODE_TRANSFER_VERDICT"] == "POOR":
        return "investigate_domain_mismatch_or_train_code_specific_taps"
    return "tune_code_generation_modes_or_increase_tasks"


def append_docs(summary: dict[str, Any]) -> list[str]:
    docs = [
        PROJECT_ROOT / "docs/evaluator/evaluator_domain_transfer_notes.md",
        PROJECT_ROOT / "docs/evaluator/math_bg_gate_pilot_2026-05-15.md",
        PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-15_v4.md",
        PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-16_v5_clean_gsm8k_code_next.md",
    ]
    title = "## Code branch pilot (2026-05-16)"
    text = "\n".join([
        "",
        title,
        "",
        f"- CODE_INTERFACE_VERDICT: `{summary['CODE_INTERFACE_VERDICT']}`",
        f"- CODE_TASKSET_VERDICT: `{summary['CODE_TASKSET_VERDICT']}`",
        f"- CODE_GENERATION_VERDICT: `{summary['CODE_GENERATION_VERDICT']}`",
        f"- CODE_TOURNAMENT_VERDICT: `{summary['CODE_TOURNAMENT_VERDICT']}`",
        f"- CODE_TRANSFER_VERDICT: `{summary['CODE_TRANSFER_VERDICT']}`",
        f"- tasks: `{summary['tasks']}`",
        f"- candidates: `{summary['candidates']}`",
        f"- strict_clean tournaments: `{summary['strict_clean_tournaments']}`",
        f"- diagnostic_mixed tournaments: `{summary['diagnostic_mixed_tournaments']}`",
        f"- random_top1_baseline: `{summary['random_top1_baseline']}`",
        f"- best AntisymLinear row: `NOT_RUN`",
        f"- best NoNorm row: `NOT_RUN`",
        "- winner: `NOT_RUN`",
        "- full report: `opi/taps/probes/code_branch_pilot_2026-05-16_summary.md`",
        "- interpretation: The local-agent wrapper generated mostly correct code on the selected local/MBPP task mix, leaving too few objective mixed unit-test tournaments for a relational tap-transfer evaluation.",
        "",
    ])
    appended = []
    for path in docs:
        if not path.exists():
            continue
        current = path.read_text(encoding="utf-8")
        if title in current:
            continue
        path.write_text(current.rstrip() + "\n" + text, encoding="utf-8")
        appended.append(repo_path(path))
    return appended


def write_md(summary: dict[str, Any]) -> None:
    lines = [
        "# Code Branch Pilot Summary",
        "",
        f"CODE_INTERFACE_VERDICT = {summary['CODE_INTERFACE_VERDICT']}",
        f"CODE_TASKSET_VERDICT = {summary['CODE_TASKSET_VERDICT']}",
        f"CODE_GENERATION_VERDICT = {summary['CODE_GENERATION_VERDICT']}",
        f"CODE_TOURNAMENT_VERDICT = {summary['CODE_TOURNAMENT_VERDICT']}",
        f"CODE_TRANSFER_VERDICT = {summary['CODE_TRANSFER_VERDICT']}",
        f"RECOMMENDED_NEXT = {summary['RECOMMENDED_NEXT']}",
        "",
        "## Interface Inspection Summary",
        "",
        f"- interface report: `{summary['interface_report']}`",
        "",
        "## Taskset Summary",
        "",
        f"- tasks: `{summary['tasks']}`",
        f"- task sources: `{summary['task_sources']}`",
        f"- devil tasks excluded: `{summary['devil_excluded']}`",
        "",
        "## Candidate Generation Summary",
        "",
        f"- candidates: `{summary['candidates']}`",
        f"- generation stage breakdown: `{summary['stage_breakdown']}`",
        f"- duplicate signature note: `{summary['duplicate_signature_note']}`",
        "",
        "## Manual Inspection",
        "",
        "- The four retained local DSA tasks were mostly solved by every direct candidate.",
        "- The two known devil-grade local tasks were excluded before generation.",
        "- Several first-tool candidates include generated self-asserts; those are preserved as first-tool-code artifacts and evaluated objectively.",
        "- Mixed outcomes came mostly from first-tool self-assert failures or MBPP tasks with ambiguous/incorrect sampled direct code.",
        "",
        "## Unit-Test Label Summary",
        "",
        f"- label_counts: `{summary['label_counts']}`",
        f"- label_by_stage: `{summary['label_by_stage']}`",
        "",
        "## Tournament Construction Summary",
        "",
        f"- strict_clean tournaments: `{summary['strict_clean_tournaments']}`",
        f"- diagnostic_mixed tournaments: `{summary['diagnostic_mixed_tournaments']}`",
        f"- primary_eval_set: `{summary['primary_eval_set']}`",
        f"- tournament tasks: `{summary['tournament_tasks']}`",
        "",
        "## Feature Capture Summary",
        "",
        "Feature capture was not run because `CODE_TOURNAMENT_VERDICT = TOO_FEW_TOURNAMENTS`.",
        "",
        "## HH-Trained AntisymLinear / NoNorm Transfer Table",
        "",
        "Transfer was not run. The pilot produced fewer than 5 usable tournaments.",
        "",
        "## Random Baseline and Correct-Candidate Distribution",
        "",
        f"- random_top1_baseline: `{summary['random_top1_baseline']}`",
        f"- correct_candidate_count_distribution: `{summary['correct_candidate_count_distribution']}`",
        "",
        "## Relation To Expanded Clean GSM8K Result",
        "",
        "The expanded clean GSM8K result remains the active positive transfer result: `EXPANDED_LINEAR_TRANSFER_VERDICT = GOOD`; `GRU_CONTROL_VERDICT = GRU_WEAK`.",
        "",
        "## Markdown Docs Updated",
        "",
    ]
    lines.extend(f"- `{path}`" for path in summary["docs_updated"])
    lines.extend([
        "",
        "## Files Modified / Created",
        "",
    ])
    lines.extend(f"- `{path}`" for path in summary["files_created"])
    lines.extend([
        "",
        "## Commands Run",
        "",
        "```bash",
    ])
    lines.extend(summary["commands_run"])
    lines.extend([
        "```",
        "",
        "## Blockers",
        "",
        summary["blockers"],
        "",
    ])
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    interface = load(REPORT_DIR / "code_branch_interface_inspection_2026-05-16.json")
    taskset = load(REPORT_DIR / "code_branch_taskset_2026-05-16.json")
    gen = load(REPORT_DIR / "code_branch_candidates_2026-05-16.json")
    tourn = load(REPORT_DIR / "code_branch_tournaments_2026-05-16.json")

    verdicts = {
        "CODE_INTERFACE_VERDICT": interface.get("code_interface_verdict", "BLOCKED"),
        "CODE_TASKSET_VERDICT": taskset.get("code_taskset_verdict", "BLOCKED"),
        "CODE_GENERATION_VERDICT": gen.get("code_generation_verdict", "WRAPPER_BLOCKED"),
        "CODE_TOURNAMENT_VERDICT": tourn.get("code_tournament_verdict", "TOO_FEW_TOURNAMENTS"),
        "CODE_TRANSFER_VERDICT": "NOT_RUN",
    }
    labels = Counter(row["unit_test_label"] for row in tourn.get("candidate_evaluations", []))
    correct_dist = Counter(
        sum(1 for c in t["diagnostic_candidates"] if c.get("is_correct"))
        for t in tourn.get("tournaments", [])
    )
    sig_counts = {}
    for cand in gen.get("candidates", []):
        sig_counts.setdefault(cand["task_id"], set()).add(cand.get("code_signature", ""))
    duplicate_note = {
        task_id: {"candidates": sum(1 for c in gen.get("candidates", []) if c["task_id"] == task_id), "unique_signatures": len(sigs)}
        for task_id, sigs in sorted(sig_counts.items())
    }
    summary = {
        **verdicts,
        "RECOMMENDED_NEXT": recommended_next(verdicts),
        "interface_report": "opi/taps/probes/code_branch_interface_inspection_2026-05-16.md",
        "tasks": len(taskset.get("tasks", [])),
        "task_sources": dict(Counter(task["source"] for task in taskset.get("tasks", []))),
        "devil_excluded": taskset.get("devil_excluded", []),
        "candidates": len(gen.get("candidates", [])),
        "stage_breakdown": gen.get("summary", {}).get("stage_breakdown", {}),
        "duplicate_signature_note": duplicate_note,
        "label_counts": dict(labels),
        "label_by_stage": tourn.get("summary", {}).get("label_by_stage", {}),
        "strict_clean_tournaments": tourn.get("summary", {}).get("strict_clean_tournaments", 0),
        "diagnostic_mixed_tournaments": tourn.get("summary", {}).get("diagnostic_mixed_tournaments", 0),
        "primary_eval_set": tourn.get("primary_eval_set"),
        "random_top1_baseline": tourn.get("summary", {}).get("random_top1_baseline"),
        "correct_candidate_count_distribution": dict(correct_dist),
        "tournament_tasks": [
            {
                "task_id": t["task_id"],
                "source": t["source"],
                "strict_clean": t["strict_clean"],
                "diagnostic_mixed": t["diagnostic_mixed"],
                "labels": dict(Counter(c["unit_test_label"] for c in t["diagnostic_candidates"])),
            }
            for t in tourn.get("tournaments", [])
        ],
        "files_created": [
            "shared/utilities/tests/manual/code_branch_pilot_lib.py",
            "shared/utilities/tests/manual/inspect_code_branch_interface.py",
            "shared/utilities/tests/manual/build_code_branch_taskset.py",
            "shared/utilities/tests/manual/generate_code_branch_candidates_local_agent.py",
            "shared/utilities/tests/manual/evaluate_code_branch_candidates.py",
            "shared/utilities/tests/manual/capture_code_branch_tap_features.py",
            "shared/utilities/tests/manual/evaluate_hh_transfer_on_code_branches.py",
            "shared/utilities/tests/manual/summarize_code_branch_pilot.py",
            "opi/taps/probes/code_branch_interface_inspection_2026-05-16.json",
            "opi/taps/probes/code_branch_interface_inspection_2026-05-16.md",
            "opi/taps/probes/code_branch_taskset_2026-05-16.json",
            "opi/taps/probes/code_branch_taskset_2026-05-16.md",
            "opi/taps/probes/code_branch_candidates_2026-05-16.json",
            "opi/taps/probes/code_branch_candidates_2026-05-16.md",
            "opi/taps/probes/code_branch_candidates_2026-05-16.log",
            "opi/taps/probes/code_branch_tournaments_2026-05-16.json",
            "opi/taps/probes/code_branch_tournaments_2026-05-16.md",
            "opi/taps/probes/code_branch_pilot_2026-05-16_summary.json",
            "opi/taps/probes/code_branch_pilot_2026-05-16_summary.md",
        ],
        "commands_run": [
            "venv/bin/python -m py_compile utilities/tests/manual/code_branch_pilot_lib.py utilities/tests/manual/inspect_code_branch_interface.py utilities/tests/manual/build_code_branch_taskset.py",
            "venv/bin/python -m py_compile utilities/tests/manual/generate_code_branch_candidates_local_agent.py utilities/tests/manual/evaluate_code_branch_candidates.py utilities/tests/manual/capture_code_branch_tap_features.py utilities/tests/manual/evaluate_hh_transfer_on_code_branches.py",
            "venv/bin/python -u utilities/tests/manual/inspect_code_branch_interface.py",
            "venv/bin/python -u utilities/tests/manual/build_code_branch_taskset.py --target-tasks 10 --min-benchmark-tasks 6 --output opi/taps/probes/code_branch_taskset_2026-05-16.json",
            "venv/bin/python -u -c \"...run_module('utilities.tests.manual.generate_code_branch_candidates_local_agent')...\"",
            "venv/bin/python -u utilities/tests/manual/evaluate_code_branch_candidates.py --candidates opi/taps/probes/code_branch_candidates_2026-05-16.json --taskset opi/taps/probes/code_branch_taskset_2026-05-16.json --output opi/taps/probes/code_branch_tournaments_2026-05-16.json",
            "venv/bin/python -m py_compile utilities/tests/manual/summarize_code_branch_pilot.py",
            "venv/bin/python -u utilities/tests/manual/summarize_code_branch_pilot.py",
        ],
        "blockers": "The selected local/MBPP task mix produced 32/40 correct candidates and only 4 diagnostic mixed tournaments, below the minimum of 5 usable tournaments for feature capture and HH-trained tap transfer.",
    }
    summary["docs_updated"] = append_docs(summary)
    write_json(SUMMARY_JSON, summary)
    write_md(summary)
    for key in (
        "CODE_INTERFACE_VERDICT",
        "CODE_TASKSET_VERDICT",
        "CODE_GENERATION_VERDICT",
        "CODE_TOURNAMENT_VERDICT",
        "CODE_TRANSFER_VERDICT",
        "RECOMMENDED_NEXT",
    ):
        print(f"{key} = {summary[key]}")
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
