"""Evaluate strict-clean screening candidates and classify tasks."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from code_branch_pilot_lib import PROJECT_ROOT, REPORT_DIR, repo_path, snippet, write_json
from evaluate_code_branch_candidates_v2 import (
    candidate_code_for_unit_tests,
    eval_candidate,
    label_from_eval,
    legacy_label_from_eval,
)


TASKPOOL_JSON = REPORT_DIR / "code_strict_clean_screening_taskpool_2026-05-17.json"
CANDIDATES_JSON = REPORT_DIR / "code_strict_clean_screening_candidates_2026-05-17.json"
OUTPUT_JSON = REPORT_DIR / "code_strict_clean_screening_results_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "code_strict_clean_screening_results_2026-05-17.md"
SUMMARY_JSON = REPORT_DIR / "code_strict_clean_screening_2026-05-17_summary.json"
SUMMARY_MD = REPORT_DIR / "code_strict_clean_screening_2026-05-17_summary.md"

DOCS_TO_APPEND = [
    PROJECT_ROOT / "docs/evaluator/evaluator_domain_transfer_notes.md",
    PROJECT_ROOT / "docs/evaluator/math_bg_gate_pilot_2026-05-15.md",
    PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-15_v4.md",
    PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-16_v5_clean_gsm8k_code_next.md",
    PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-17_v7_actual_state_and_next.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taskpool", default=str(TASKPOOL_JSON))
    parser.add_argument("--candidates", default=str(CANDIDATES_JSON))
    parser.add_argument("--output", default=str(OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    parser.add_argument("--summary-output", default=str(SUMMARY_JSON))
    parser.add_argument("--summary-md", default=str(SUMMARY_MD))
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tests_for(task: dict[str, Any]) -> list[str]:
    tests = list(task.get("tests") or [])
    if not tests:
        tests = list(task.get("public_tests", [])) + list(task.get("hidden_tests", []))
    return [str(test).strip() for test in tests if str(test).strip()]


def is_primary_candidate(row: dict[str, Any]) -> bool:
    return not bool(row.get("duplicate_of"))


def classify_task(rows: list[dict[str, Any]]) -> str:
    labels = Counter(str(row.get("unit_test_label", "unknown")) for row in rows if is_primary_candidate(row))
    usable = labels.get("correct", 0) + labels.get("near_miss", 0) + labels.get("wrong_code", 0) + labels.get("runtime_error", 0)
    if labels.get("correct", 0) and labels.get("near_miss", 0):
        return "strict_clean_ready"
    if usable and labels.get("correct", 0) == usable:
        return "all_correct"
    if labels.get("correct", 0):
        return "anchor_only"
    if labels.get("near_miss", 0):
        return "near_miss_only"
    if labels.get("wrong_code", 0) or labels.get("runtime_error", 0):
        return "all_wrong"
    return "malformed_only"


def task_has_strict_clean(rows: list[dict[str, Any]]) -> bool:
    primary = [row for row in rows if is_primary_candidate(row)]
    return any(row.get("unit_test_label") == "correct" for row in primary) and any(
        row.get("unit_test_label") == "near_miss" for row in primary
    )


def evaluate_candidate(candidate: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    code, recovered = candidate_code_for_unit_tests(candidate, task)
    result = eval_candidate(
        code,
        tests_for(task),
        str(task.get("function_name", "")),
        float(task.get("timeout_seconds", 5.0)),
    )
    label = label_from_eval(result, code, str(task.get("function_name", "")))
    return {
        **candidate,
        **result,
        "unit_test_label": label,
        "legacy_unit_test_label": legacy_label_from_eval(result),
        "is_correct": label == "correct",
        "is_runnable": bool(result.get("safety_ok") and result.get("syntax_ok") and result.get("import_ok") and result.get("runtime_ok")),
        "unit_test_code_recovered": recovered,
        "final_code_for_unit_tests": code,
    }


def task_breakdowns(task_rows: dict[str, list[dict[str, Any]]], tasks: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_task: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for task_id, task in tasks.items():
        cand_rows = task_rows.get(task_id, [])
        primary = [row for row in cand_rows if is_primary_candidate(row)]
        labels = Counter(str(row.get("unit_test_label", "unknown")) for row in primary)
        classification = classify_task(cand_rows)
        row = {
            "task_id": task_id,
            "source": task.get("source", "unknown"),
            "difficulty": task.get("difficulty", "unknown"),
            "function_name": task.get("function_name", ""),
            "prior_outcome": task.get("prior_outcome", "unknown"),
            "number_of_tests": task.get("number_of_tests", len(tests_for(task))),
            "n_candidates": len(cand_rows),
            "n_primary_candidates": len(primary),
            "label_counts": dict(labels),
            "classification": classification,
            "strict_clean_ready": classification == "strict_clean_ready",
            "has_correct": labels.get("correct", 0) > 0,
            "has_near_miss": labels.get("near_miss", 0) > 0,
            "has_wrong_code": labels.get("wrong_code", 0) > 0,
        }
        by_task[task_id] = row
        if cand_rows:
            rows.append(row)
    return by_task, rows


def rate(numerator: int, denominator: int) -> float:
    return float(numerator) / max(int(denominator), 1)


def role_success(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for role in sorted({str(row.get("screening_role", "unknown")) for row in rows}):
        role_rows = [row for row in rows if row.get("screening_role") == role and is_primary_candidate(row)]
        out[role] = {
            "n": len(role_rows),
            "label_counts": dict(Counter(row.get("unit_test_label", "unknown") for row in role_rows)),
            "correct_rate": rate(sum(1 for row in role_rows if row.get("unit_test_label") == "correct"), len(role_rows)),
            "near_miss_rate": rate(sum(1 for row in role_rows if row.get("unit_test_label") == "near_miss"), len(role_rows)),
        }
    return out


def verdict_for(strict_count: int) -> str:
    if strict_count >= 10:
        return "GREEN"
    if strict_count >= 5:
        return "YELLOW"
    return "RED"


def recommended_next(verdict: str) -> str:
    if verdict == "GREEN":
        return "run_transfer_on_screened_strict_clean_tasks"
    if verdict == "YELLOW":
        return "run_small_transfer_or_screen_more_tasks"
    return "improve_task_source_or_test_granularity_before_more_generation"


def write_results_md(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# Strict-Clean Code Screening Results",
        "",
        f"STRICT_CLEAN_SCREENING_VERDICT = {payload['strict_clean_screening_verdict']}",
        "",
        f"- tasks_screened: `{s['tasks_screened']}`",
        f"- strict_clean_ready: `{s['strict_clean_ready']}`",
        f"- anchor_only: `{s['anchor_only']}`",
        f"- near_miss_only: `{s['near_miss_only']}`",
        f"- all_correct: `{s['all_correct']}`",
        f"- all_wrong: `{s['all_wrong']}`",
        f"- malformed_only: `{s['malformed_only']}`",
        f"- label_totals: `{s['label_totals']}`",
        f"- per_source: `{s['per_source_classification']}`",
        f"- per_difficulty: `{s['per_difficulty_classification']}`",
        f"- role_success: `{s['role_success']}`",
        "",
        "## Strict-Clean-Ready Tasks",
        "",
        "| task_id | source | difficulty | function | tests | labels | prior |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for row in payload["strict_clean_ready_tasks"][:20]:
        lines.append(
            f"| `{row['task_id']}` | `{row['source']}` | `{row['difficulty']}` | `{row['function_name']}` | "
            f"{row['number_of_tests']} | `{row['label_counts']}` | `{row['prior_outcome']}` |"
        )
    lines.extend(["", "## Task Classification", "", "| task_id | classification | labels |", "| --- | --- | --- |"])
    for row in payload["task_rows"]:
        lines.append(f"| `{row['task_id']}` | `{row['classification']}` | `{row['label_counts']}` |")
    lines.extend(["", "## Avoid In Future", ""])
    for row in payload["avoid_in_future"][:30]:
        lines.append(
            f"- `{row['task_id']}` `{row['classification']}` labels=`{row['label_counts']}` "
            f"reason=`{row['avoid_reason']}`"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(summary: dict[str, Any]) -> list[str]:
    title = "## Strict-clean code task screening (2026-05-17)"
    interpretation = summary["one_sentence_interpretation"]
    text = "\n".join([
        "",
        title,
        "",
        f"- SCREENING_TASKPOOL_VERDICT: `{summary['SCREENING_TASKPOOL_VERDICT']}`",
        f"- SCREENING_GENERATION_VERDICT: `{summary['SCREENING_GENERATION_VERDICT']}`",
        f"- STRICT_CLEAN_SCREENING_VERDICT: `{summary['STRICT_CLEAN_SCREENING_VERDICT']}`",
        f"- tasks screened: `{summary['tasks_screened']}`",
        f"- strict_clean_ready: `{summary['strict_clean_ready']}`",
        f"- anchor_only / near_miss_only / all_correct / all_wrong: `{summary['anchor_only']} / {summary['near_miss_only']} / {summary['all_correct']} / {summary['all_wrong']}`",
        f"- label totals: `{summary['label_totals']}`",
        f"- per-source summary: `{summary['per_source_classification']}`",
        f"- per-difficulty summary: `{summary['per_difficulty_classification']}`",
        f"- within-task pairing bottleneck: `{summary['within_task_pairing_bottleneck']}`",
        "- full report: `opi/taps/probes/code_strict_clean_screening_2026-05-17_summary.md`",
        f"- interpretation: {interpretation}",
        "",
    ])
    appended: list[str] = []
    for path in DOCS_TO_APPEND:
        if not path.exists():
            continue
        current = path.read_text(encoding="utf-8")
        if title in current:
            continue
        path.write_text(current.rstrip() + "\n" + text, encoding="utf-8")
        appended.append(repo_path(path))
    return appended


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Strict-Clean Code Screening Summary",
        "",
        f"SCREENING_TASKPOOL_VERDICT = {summary['SCREENING_TASKPOOL_VERDICT']}",
        f"SCREENING_GENERATION_VERDICT = {summary['SCREENING_GENERATION_VERDICT']}",
        f"STRICT_CLEAN_SCREENING_VERDICT = {summary['STRICT_CLEAN_SCREENING_VERDICT']}",
        f"RECOMMENDED_NEXT = {summary['RECOMMENDED_NEXT']}",
        "",
        "## 1. Task Pool Construction",
        "",
        f"- task pool verdict: `{summary['SCREENING_TASKPOOL_VERDICT']}`",
        f"- tasks in pool: `{summary['tasks_in_pool']}`",
        f"- tasks screened: `{summary['tasks_screened']}`",
        f"- source mix: `{summary['taskpool_source_mix']}`",
        f"- difficulty mix: `{summary['taskpool_difficulty_mix']}`",
        "",
        "## 2. Generation Summary",
        "",
        f"- generation verdict: `{summary['SCREENING_GENERATION_VERDICT']}`",
        f"- total screening candidates: `{summary['total_screening_candidates']}`",
        f"- primary unique candidates: `{summary['primary_unique_candidates']}`",
        f"- duplicate candidates: `{summary['duplicate_candidates']}`",
        f"- generation errors: `{summary['generation_errors']}`",
        "",
        "## 3. Candidate Label Summary",
        "",
        f"- label totals: `{summary['label_totals']}`",
        f"- role success: `{summary['role_success']}`",
        "",
        "## 4. Task Classification Summary",
        "",
        f"- strict_clean_ready: `{summary['strict_clean_ready']}`",
        f"- anchor_only: `{summary['anchor_only']}`",
        f"- near_miss_only: `{summary['near_miss_only']}`",
        f"- all_correct: `{summary['all_correct']}`",
        f"- all_wrong: `{summary['all_wrong']}`",
        f"- malformed_only: `{summary['malformed_only']}`",
        "",
        "## 5. Strict-Clean-Ready Task List",
        "",
    ]
    if summary["strict_clean_ready_tasks"]:
        for row in summary["strict_clean_ready_tasks"]:
            lines.append(f"- `{row['task_id']}` `{row['source']}` `{row['difficulty']}` labels=`{row['label_counts']}`")
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## 6. What Failed",
        "",
        summary["what_failed"],
        "",
        "## 7. Recommended Next",
        "",
        f"`{summary['RECOMMENDED_NEXT']}`",
        "",
        "## 8. Docs Updated",
        "",
    ])
    for doc in summary["docs_updated"]:
        lines.append(f"- `{doc}`")
    if not summary["docs_updated"]:
        lines.append("- none appended; section already present or doc missing")
    lines.extend([
        "",
        "## 9. Files Modified / Created",
        "",
    ])
    for file_path in summary["files_modified_or_created"]:
        lines.append(f"- `{file_path}`")
    lines.extend([
        "",
        "## 10. Commands Run",
        "",
        "```bash",
        "venv/bin/python -m py_compile utilities/tests/manual/build_code_strict_clean_screening_taskpool.py",
        "venv/bin/python -m py_compile utilities/tests/manual/generate_code_strict_clean_screening_candidates.py",
        "venv/bin/python -m py_compile utilities/tests/manual/evaluate_code_strict_clean_screening.py",
        "venv/bin/python -u utilities/tests/manual/build_code_strict_clean_screening_taskpool.py",
        "# The shell-redirection form was attempted first, but it hid CUDA in this environment.",
        "# The successful generation run used the script's internal --log-file tee instead.",
        "venv/bin/python -u utilities/tests/manual/generate_code_strict_clean_screening_candidates.py --max-tasks 60 --min-tasks 30 --target-strict-clean-ready 10 --candidates-per-task 3 --chunk-size 5 --device cuda",
        "venv/bin/python -u utilities/tests/manual/evaluate_code_strict_clean_screening.py",
        "```",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _main(args: argparse.Namespace) -> None:
    taskpool = load_json(args.taskpool)
    candidate_payload = load_json(args.candidates)
    tasks = {str(task["task_id"]): task for task in taskpool.get("tasks", [])}
    candidate_rows = list(candidate_payload.get("candidates", []))
    evaluated: list[dict[str, Any]] = []

    for index, candidate in enumerate(candidate_rows, start=1):
        task_id = str(candidate.get("task_id", ""))
        task = tasks.get(task_id)
        if not task:
            continue
        row = evaluate_candidate(candidate, task)
        evaluated.append(row)
        if index % 25 == 0:
            print(f"[screen-eval] evaluated {index}/{len(candidate_rows)}", flush=True)

    task_rows_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated:
        task_rows_by_id[str(row.get("task_id", ""))].append(row)
    by_task, task_rows = task_breakdowns(task_rows_by_id, tasks)
    screened_task_rows = [row for row in task_rows if row["n_candidates"] > 0]
    classification_counts = Counter(row["classification"] for row in screened_task_rows)
    primary_rows = [row for row in evaluated if is_primary_candidate(row)]
    label_totals = dict(Counter(row.get("unit_test_label", "unknown") for row in primary_rows))
    strict_rows = [row for row in screened_task_rows if row["classification"] == "strict_clean_ready"]
    avoid_rows = []
    for row in screened_task_rows:
        if row["classification"] in {"all_correct", "all_wrong", "malformed_only"}:
            avoid_reason = {
                "all_correct": "screening produced no contrast side",
                "all_wrong": "strong anchor failed to produce a correct branch",
                "malformed_only": "no usable code under screening modes",
            }[row["classification"]]
            avoid_rows.append({**row, "avoid_reason": avoid_reason})

    per_source = {
        source: dict(Counter(row["classification"] for row in screened_task_rows if row["source"] == source))
        for source in sorted({row["source"] for row in screened_task_rows})
    }
    per_difficulty = {
        diff: dict(Counter(row["classification"] for row in screened_task_rows if row["difficulty"] == diff))
        for diff in sorted({row["difficulty"] for row in screened_task_rows})
    }
    verdict = verdict_for(len(strict_rows))
    generation_summary = candidate_payload.get("summary", {})
    taskpool_summary = taskpool.get("summary", {})
    near_or_anchor_only = classification_counts.get("anchor_only", 0) + classification_counts.get("near_miss_only", 0)
    bottleneck = "CONFIRMED" if len(strict_rows) < 10 and near_or_anchor_only > 0 else "NOT_CONFIRMED_BY_SCREENING"
    if verdict == "GREEN":
        bottleneck = "SCREENING_FOUND_READY_TASKS"
    one_sentence = (
        "Screening found enough same-task correct-vs-near-miss pairs for a transfer-only follow-up."
        if verdict == "GREEN"
        else "Screening still shows the main constraint is finding same-task correct-vs-near-miss pairs cheaply."
    )
    what_failed = (
        "The screen did not reach ten strict-clean-ready tasks; one-sided and collapse classifications remain the useful diagnostic."
        if verdict != "GREEN"
        else "No screening blocker: enough tasks naturally produced both a correct anchor and a near-miss branch."
    )
    summary = {
        "SCREENING_TASKPOOL_VERDICT": taskpool.get("screening_taskpool_verdict", "UNKNOWN"),
        "SCREENING_GENERATION_VERDICT": candidate_payload.get("screening_generation_verdict", "UNKNOWN"),
        "STRICT_CLEAN_SCREENING_VERDICT": verdict,
        "RECOMMENDED_NEXT": recommended_next(verdict),
        "tasks_in_pool": len(taskpool.get("tasks", [])),
        "tasks_screened": len(screened_task_rows),
        "taskpool_source_mix": taskpool_summary.get("source_mix", {}),
        "taskpool_difficulty_mix": taskpool_summary.get("difficulty_mix", {}),
        "total_screening_candidates": len(candidate_rows),
        "primary_unique_candidates": len(primary_rows),
        "duplicate_candidates": sum(1 for row in evaluated if row.get("duplicate_of")),
        "generation_errors": len(candidate_payload.get("generation_errors", [])),
        "strict_clean_ready": len(strict_rows),
        "anchor_only": classification_counts.get("anchor_only", 0),
        "near_miss_only": classification_counts.get("near_miss_only", 0),
        "all_correct": classification_counts.get("all_correct", 0),
        "all_wrong": classification_counts.get("all_wrong", 0),
        "malformed_only": classification_counts.get("malformed_only", 0),
        "label_totals": label_totals,
        "per_source_classification": per_source,
        "per_difficulty_classification": per_difficulty,
        "role_success": role_success(evaluated),
        "strong_anchor_correct_rate": role_success(evaluated).get("strong_anchor", {}).get("correct_rate", 0.0),
        "near_miss_probe_near_miss_rate": mean([
            role_success(evaluated).get(role, {}).get("near_miss_rate", 0.0)
            for role in ("near_miss_probe_1", "near_miss_probe_2")
            if role in role_success(evaluated)
        ]) if any(role in role_success(evaluated) for role in ("near_miss_probe_1", "near_miss_probe_2")) else 0.0,
        "strict_clean_ready_tasks": strict_rows[:10],
        "within_task_pairing_bottleneck": bottleneck,
        "one_sentence_interpretation": one_sentence,
        "what_failed": what_failed,
        "full_results_path": repo_path(Path(args.output)),
        "summary_path": repo_path(Path(args.summary_md)),
        "generation_runtime_note": (
            "The requested shell-redirection form was attempted, but torch.cuda.is_available() was false in that process. "
            "The successful bounded generation run used the same arguments without shell redirection and wrote the log through the script's internal tee."
        ),
    }

    results_payload = {
        "strict_clean_screening_verdict": verdict,
        "inputs": {"taskpool": repo_path(Path(args.taskpool)), "candidates": repo_path(Path(args.candidates))},
        "summary": {
            "tasks_screened": summary["tasks_screened"],
            "strict_clean_ready": summary["strict_clean_ready"],
            "anchor_only": summary["anchor_only"],
            "near_miss_only": summary["near_miss_only"],
            "all_correct": summary["all_correct"],
            "all_wrong": summary["all_wrong"],
            "malformed_only": summary["malformed_only"],
            "label_totals": summary["label_totals"],
            "per_source_classification": per_source,
            "per_difficulty_classification": per_difficulty,
            "role_success": summary["role_success"],
        },
        "candidate_evaluations": evaluated,
        "by_task": by_task,
        "task_rows": screened_task_rows,
        "strict_clean_ready_tasks": strict_rows[:10],
        "avoid_in_future": avoid_rows,
        "outputs": {"json": repo_path(Path(args.output)), "md": repo_path(Path(args.output_md))},
    }
    write_json(Path(args.output), results_payload)
    write_results_md(Path(args.output_md), results_payload)

    docs_updated = append_docs(summary)
    summary["docs_updated"] = docs_updated
    summary["files_modified_or_created"] = [
        "shared/utilities/tests/manual/build_code_strict_clean_screening_taskpool.py",
        "shared/utilities/tests/manual/generate_code_strict_clean_screening_candidates.py",
        "shared/utilities/tests/manual/evaluate_code_strict_clean_screening.py",
        repo_path(Path(args.taskpool)),
        "opi/taps/probes/code_strict_clean_screening_taskpool_2026-05-17.md",
        repo_path(Path(args.candidates)),
        "opi/taps/probes/code_strict_clean_screening_candidates_2026-05-17.md",
        "opi/taps/probes/code_strict_clean_screening_candidates_2026-05-17.log",
        repo_path(Path(args.output)),
        repo_path(Path(args.output_md)),
        repo_path(Path(args.summary_output)),
        repo_path(Path(args.summary_md)),
        *docs_updated,
    ]
    write_json(Path(args.summary_output), summary)
    write_summary_md(Path(args.summary_md), summary)
    print(f"STRICT_CLEAN_SCREENING_VERDICT = {verdict}")
    print(f"strict_clean_ready = {len(strict_rows)}")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.summary_output}")
    print(f"Wrote {args.summary_md}")


def main() -> None:
    args = parse_args()
    _main(args)


if __name__ == "__main__":
    main()
