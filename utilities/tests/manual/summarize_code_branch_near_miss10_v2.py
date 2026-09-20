"""Summarize the independent 10-task code near-miss enrichment run."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from code_branch_pilot_lib import REPORT_DIR, output_path, repo_path, write_json


DEFAULT_TASKSET = REPORT_DIR / "code_branch_taskset_v2_near_miss10_2026-05-17.json"
DEFAULT_CANDIDATES = REPORT_DIR / "code_branch_candidates_v2_near_miss10_2026-05-17.json"
DEFAULT_TOURNAMENTS = REPORT_DIR / "code_branch_tournaments_v2_near_miss10_2026-05-17.json"
DEFAULT_OUTPUT = REPORT_DIR / "code_branch_near_miss_enrichment10_2026-05-17_summary.json"
DEFAULT_MD = REPORT_DIR / "code_branch_near_miss_enrichment10_2026-05-17_summary.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taskset", default=str(DEFAULT_TASKSET))
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--tournaments", default=str(DEFAULT_TOURNAMENTS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()
    if not args.output_md:
        args.output_md = str(Path(args.output).with_suffix(".md"))
    return args


def load(path: str | Path) -> dict[str, Any]:
    return json.loads(output_path(path).read_text(encoding="utf-8"))


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(row.get("unit_test_label", "unknown") for row in rows))


def task_rows(tournament: dict[str, Any]) -> dict[str, Any]:
    labels = label_counts(tournament.get("diagnostic_candidates", []))
    return {
        "task_id": tournament["task_id"],
        "source": tournament["source"],
        "difficulty": tournament.get("difficulty", "unknown"),
        "strict_clean": bool(tournament.get("strict_clean")),
        "diagnostic_runnable": bool(tournament.get("diagnostic_runnable")),
        "label_counts": labels,
        "has_correct": labels.get("correct", 0) > 0,
        "has_near_miss_partial_pass": labels.get("near_miss", 0) > 0,
    }


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Code Branch Near-Miss Enrichment 10 Summary",
        "",
        f"CODE_V2_NEARMISS10_TASKSET_VERDICT = {payload['CODE_V2_NEARMISS10_TASKSET_VERDICT']}",
        f"CODE_V2_NEARMISS10_GENERATION_VERDICT = {payload['CODE_V2_NEARMISS10_GENERATION_VERDICT']}",
        f"CODE_V2_NEARMISS10_TOURNAMENT_VERDICT = {payload['CODE_V2_NEARMISS10_TOURNAMENT_VERDICT']}",
        f"CODE_V2_NEARMISS10_SUCCESS = {payload['CODE_V2_NEARMISS10_SUCCESS']}",
        "",
        "## Target",
        "",
        "- objective: `correct vs near_miss_partial_pass`",
        "- success_criterion: `strict_clean >= 5`",
        "- transfer: `NOT_RUN`",
        "",
        "## Summary",
        "",
        f"- tasks: `{payload['tasks']}`",
        f"- unique_candidates: `{payload['unique_candidates']}`",
        f"- duplicate_rate: `{payload['duplicate_rate']}`",
        f"- label_counts: `{payload['label_counts']}`",
        f"- strict_clean: `{payload['strict_clean_tournaments']}`",
        f"- diagnostic_runnable: `{payload['diagnostic_runnable_tournaments']}`",
        f"- diagnostic_mixed: `{payload['diagnostic_mixed_tournaments']}`",
        f"- near_miss_tasks: `{payload['near_miss_tasks']}`",
        f"- correct_and_near_miss_tasks: `{payload['correct_and_near_miss_tasks']}`",
        f"- malformed_candidates: `{payload['label_counts'].get('malformed', 0)}`",
        f"- safety_rejected_candidates: `{payload['label_counts'].get('safety_rejected', 0)}`",
        f"- raw_wrapper_status_rejected_count: `{payload['raw_wrapper_status_rejected_count']}`",
        f"- admitted_wrapper_artifact_count: `{payload['admitted_wrapper_artifact_count']}`",
        "",
        "## Tasks",
        "",
        "| task_id | source | difficulty | strict_clean | diagnostic_runnable | labels |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in payload["task_outcomes"]:
        lines.append(
            f"| `{row['task_id']}` | `{row['source']}` | `{row['difficulty']}` | "
            f"{row['strict_clean']} | {row['diagnostic_runnable']} | `{row['label_counts']}` |"
        )
    lines.extend([
        "",
        "## Files",
        "",
    ])
    lines.extend(f"- `{path}`" for path in payload["files"])
    lines.extend(["", "## Commands Run", "", "```bash"])
    lines.extend(payload["commands_run"])
    lines.extend(["```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    taskset = load(args.taskset)
    candidates = load(args.candidates)
    tournaments = load(args.tournaments)
    rows = tournaments.get("candidate_evaluations", [])
    summary = tournaments.get("summary", {})
    task_outcomes = [task_rows(t) for t in tournaments.get("tournaments", [])]
    raw_wrapper_errors = [
        row for row in candidates.get("generation_errors", [])
        if "max steps reached" in str(row).lower() or "wrapper" in str(row).lower() or "prose" in str(row).lower()
    ]
    admitted_wrapper = [
        row for row in rows
        if "max steps reached" in str(row.get("final_code", "")).lower()
        or "max steps reached" in str(row.get("raw_final_code", "")).lower()
    ]
    strict = int(summary.get("strict_clean_tournaments", 0))
    success = "PASS" if strict >= 5 else "MISS"
    output = output_path(args.output)
    output_md = output_path(args.output_md)
    payload = {
        "CODE_V2_NEARMISS10_TASKSET_VERDICT": taskset.get("code_v2_nearmiss10_taskset_verdict", "UNKNOWN"),
        "CODE_V2_NEARMISS10_GENERATION_VERDICT": candidates.get("code_v2_generation_verdict", "UNKNOWN"),
        "CODE_V2_NEARMISS10_TOURNAMENT_VERDICT": tournaments.get("code_v2_tournament_verdict", "UNKNOWN"),
        "CODE_V2_NEARMISS10_SUCCESS": success,
        "tasks": len(taskset.get("tasks", [])),
        "source_mix": taskset.get("source_mix", {}),
        "difficulty_mix": taskset.get("difficulty_mix", {}),
        "unique_candidates": candidates.get("summary", {}).get("unique_candidates", summary.get("unique_candidates")),
        "duplicate_rate": candidates.get("summary", {}).get("duplicate_rate", summary.get("duplicate_rate")),
        "stage_breakdown": candidates.get("summary", {}).get("stage_breakdown", {}),
        "label_counts": label_counts(rows),
        "strict_clean_tournaments": strict,
        "diagnostic_runnable_tournaments": int(summary.get("diagnostic_runnable_tournaments", 0)),
        "diagnostic_mixed_tournaments": int(summary.get("diagnostic_mixed_tournaments", 0)),
        "near_miss_tasks": sum(1 for row in task_outcomes if row["has_near_miss_partial_pass"]),
        "correct_and_near_miss_tasks": sum(
            1 for row in task_outcomes if row["has_correct"] and row["has_near_miss_partial_pass"]
        ),
        "raw_wrapper_status_rejected_count": len(raw_wrapper_errors),
        "admitted_wrapper_artifact_count": len(admitted_wrapper),
        "task_outcomes": task_outcomes,
        "files": [
            repo_path(output_path(args.taskset)),
            repo_path(Path(str(output_path(args.taskset))).with_suffix(".md")),
            repo_path(output_path(args.candidates)),
            repo_path(Path(str(output_path(args.candidates))).with_suffix(".md")),
            repo_path(Path(str(output_path(args.candidates))).with_suffix(".log")),
            repo_path(output_path(args.tournaments)),
            repo_path(Path(str(output_path(args.tournaments))).with_suffix(".md")),
            repo_path(output),
            repo_path(output_md),
        ],
        "commands_run": [
            "venv/bin/python -m py_compile utilities/tests/manual/build_code_branch_near_miss_taskset_v2.py utilities/tests/manual/generate_code_branch_candidates_v2.py utilities/tests/manual/evaluate_code_branch_candidates_v2.py utilities/tests/manual/summarize_code_branch_near_miss10_v2.py",
            "venv/bin/python -u utilities/tests/manual/build_code_branch_near_miss_taskset_v2.py",
            "venv/bin/python -u utilities/tests/manual/generate_code_branch_candidates_v2.py --taskset opi/taps/probes/code_branch_taskset_v2_near_miss10_2026-05-17.json --output opi/taps/probes/code_branch_candidates_v2_near_miss10_2026-05-17.json --log-file opi/taps/probes/code_branch_candidates_v2_near_miss10_2026-05-17.log --partial-output opi/taps/probes/code_branch_candidates_v2_near_miss10_2026-05-17.partial.json --max-tasks 10 --min-tasks 10 --max-candidates-per-task 6 --hard-cap-total-candidates 70 --target-usable-tournaments 10 --prefer-prefinal --limit-repaired-final-per-task 1 --device cuda --no-resume",
            "venv/bin/python -u utilities/tests/manual/evaluate_code_branch_candidates_v2.py --candidates opi/taps/probes/code_branch_candidates_v2_near_miss10_2026-05-17.json --taskset opi/taps/probes/code_branch_taskset_v2_near_miss10_2026-05-17.json --output opi/taps/probes/code_branch_tournaments_v2_near_miss10_2026-05-17.json --no-resume",
            "venv/bin/python -u utilities/tests/manual/summarize_code_branch_near_miss10_v2.py",
        ],
    }
    write_json(output, payload)
    write_md(output_md, payload)
    print(f"CODE_V2_NEARMISS10_SUCCESS = {success}")
    print(f"strict_clean = {strict}")
    print(f"Wrote {output}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
