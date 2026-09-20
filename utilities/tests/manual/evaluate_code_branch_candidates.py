"""Unit-test code branch candidates and construct branch-selection tournaments."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import textwrap
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from code_branch_pilot_lib import REPORT_DIR, load_json, repo_path, snippet, write_json

LOCAL_AGENT_DIR = Path(__file__).resolve().parents[4] / "src" / "local_agent"
if str(LOCAL_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_AGENT_DIR))

try:
    import ouro_agent_improved as _agent
except Exception:
    _agent = None


DEFAULT_CANDIDATES = REPORT_DIR / "code_branch_candidates_2026-05-16.json"
DEFAULT_TASKSET = REPORT_DIR / "code_branch_taskset_2026-05-16.json"
DEFAULT_OUTPUT = REPORT_DIR / "code_branch_tournaments_2026-05-16.json"
DEFAULT_MD = REPORT_DIR / "code_branch_tournaments_2026-05-16.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--taskset", default=str(DEFAULT_TASKSET))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--output-md", default=str(DEFAULT_MD))
    return parser.parse_args()


def eval_candidate(code: str, tests: list[str], function_name: str, timeout: float) -> dict[str, Any]:
    harness = {
        "code": code,
        "tests": tests,
        "function_name": function_name,
    }
    script = textwrap.dedent(
        """
        import json, sys, traceback, contextlib, io

        payload = json.loads(sys.stdin.read())
        code = payload["code"]
        tests = payload["tests"]
        fn_name = payload["function_name"]
        result = {
            "syntax_ok": False,
            "import_ok": False,
            "runtime_ok": False,
            "tests_total": len(tests),
            "tests_passed": 0,
            "tests_failed": len(tests),
            "error_type": "",
            "error_message_short": "",
        }
        scope = {"__name__": "__candidate_eval__"}
        try:
            compiled = compile(code, "<candidate>", "exec")
            result["syntax_ok"] = True
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compiled, scope, scope)
            result["import_ok"] = True
            if fn_name and not callable(scope.get(fn_name)):
                raise AssertionError(f"function not found: {fn_name}")
            for test in tests:
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        exec(test, scope, scope)
                    result["tests_passed"] += 1
                except Exception:
                    pass
            result["tests_failed"] = len(tests) - result["tests_passed"]
            result["runtime_ok"] = True
        except Exception as exc:
            result["error_type"] = type(exc).__name__
            result["error_message_short"] = str(exc)[:300]
        print("JSON_RESULT:" + json.dumps(result))
        """
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            input=json.dumps(harness),
            text=True,
            capture_output=True,
            timeout=max(0.5, float(timeout)),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "syntax_ok": False,
            "import_ok": False,
            "runtime_ok": False,
            "tests_total": len(tests),
            "tests_passed": 0,
            "tests_failed": len(tests),
            "pass_rate": 0.0,
            "error_type": "TimeoutExpired",
            "error_message_short": "candidate timed out",
            "execution_seconds": float(timeout),
        }
    marker = "JSON_RESULT:"
    result = None
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(marker):
            result = json.loads(line[len(marker):])
            break
    if result is None:
        result = {
            "syntax_ok": False,
            "import_ok": False,
            "runtime_ok": False,
            "tests_total": len(tests),
            "tests_passed": 0,
            "tests_failed": len(tests),
            "error_type": "HarnessParseError",
            "error_message_short": (proc.stderr or proc.stdout)[-300:],
        }
    result["pass_rate"] = float(result["tests_passed"]) / max(int(result["tests_total"]), 1)
    result["execution_seconds"] = 0.0
    return result


def label_from_eval(row: dict[str, Any]) -> str:
    if row["tests_total"] > 0 and row["tests_passed"] == row["tests_total"]:
        return "correct"
    if row.get("syntax_ok") and row.get("import_ok") and row.get("runtime_ok") and row.get("tests_passed", 0) > 0:
        return "near_miss"
    return "nonsense"


def candidate_code_for_unit_tests(candidate: dict[str, Any], task: dict[str, Any]) -> tuple[str, bool]:
    code = candidate.get("final_code") or candidate.get("sanitized_code") or ""
    if not code or _agent is None or not hasattr(_agent, "final_python_code_for_answer"):
        return code, False
    try:
        recovered = _agent.final_python_code_for_answer(task.get("prompt", ""), code)
    except Exception:
        return code, False
    if recovered and recovered != code:
        return recovered, True
    return code, False


def write_md(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# Code Branch Tournaments",
        "",
        f"CODE_TOURNAMENT_VERDICT = {payload['code_tournament_verdict']}",
        "",
        f"- primary_eval_set: `{payload['primary_eval_set']}`",
        f"- tasks_attempted: `{s['tasks_attempted']}`",
        f"- candidates_evaluated: `{s['candidates_evaluated']}`",
        f"- correct_candidates: `{s['correct_candidates']}`",
        f"- near_miss_candidates: `{s['near_miss_candidates']}`",
        f"- nonsense_candidates: `{s['nonsense_candidates']}`",
        f"- strict_clean_tournaments: `{s['strict_clean_tournaments']}`",
        f"- diagnostic_mixed_tournaments: `{s['diagnostic_mixed_tournaments']}`",
        f"- random_top1_baseline: `{s['random_top1_baseline']}`",
        f"- near_miss_fraction: `{s['near_miss_fraction']}`",
        f"- stage_breakdown: `{s['stage_breakdown']}`",
        "",
        "## Tournaments",
        "",
        "| task_id | source | strict_clean | diagnostic_mixed | candidates | labels |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for t in payload["tournaments"]:
        labels = Counter(c["unit_test_label"] for c in t["diagnostic_candidates"])
        lines.append(
            f"| `{t['task_id']}` | `{t['source']}` | {t['strict_clean']} | {t['diagnostic_mixed']} | "
            f"{len(t['diagnostic_candidates'])} | `{dict(labels)}` |"
        )
    lines.extend(["", "## Candidate Examples", ""])
    for row in payload["candidate_evaluations"][:16]:
        lines.append(
            f"- `{row['task_id']}` {row['mode']} {row['candidate_stage']} "
            f"label=`{row['unit_test_label']}` pass={row['tests_passed']}/{row['tests_total']} "
            f"err=`{row['error_type']}` code={snippet(row['final_code'], 100)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    taskset = load_json(args.taskset)
    candidate_payload = load_json(args.candidates)
    tasks = {task["task_id"]: task for task in taskset["tasks"]}
    rows: list[dict[str, Any]] = []
    for cand in candidate_payload.get("candidates", []):
        task = tasks.get(cand["task_id"])
        if not task:
            continue
        tests = list(task.get("tests") or task.get("public_tests", []) + task.get("hidden_tests", []))
        eval_code, recovered_for_eval = candidate_code_for_unit_tests(cand, task)
        eval_result = eval_candidate(
            eval_code,
            tests,
            task.get("function_name", ""),
            float(task.get("timeout_seconds", 3.0)),
        )
        label = label_from_eval(eval_result)
        row = {
            **cand,
            "raw_final_code": cand.get("final_code", ""),
            "final_code": eval_code,
            "candidate_code_recovered_by_evaluator": recovered_for_eval,
            **eval_result,
            "unit_test_label": label,
            "is_correct": label == "correct",
            "is_runnable": bool(eval_result.get("syntax_ok") and eval_result.get("import_ok") and eval_result.get("runtime_ok")),
        }
        rows.append(row)

    tournaments = []
    for task_id, task in tasks.items():
        cand_rows = [row for row in rows if row["task_id"] == task_id]
        if len(cand_rows) < 2:
            continue
        correct = [row for row in cand_rows if row["unit_test_label"] == "correct"]
        incorrect = [row for row in cand_rows if row["unit_test_label"] != "correct"]
        runnable = [row for row in cand_rows if row["is_runnable"]]
        strict_candidates = [row for row in cand_rows if row["unit_test_label"] in {"correct", "near_miss"}]
        strict_clean = bool(correct and any(row["unit_test_label"] == "near_miss" for row in strict_candidates) and len(runnable) >= 2)
        diagnostic_mixed = bool(correct and incorrect)
        if not diagnostic_mixed:
            continue
        tournaments.append({
            "tournament_id": len(tournaments),
            "task_id": task_id,
            "source": task["source"],
            "prompt": task["prompt"],
            "function_name": task["function_name"],
            "strict_clean": strict_clean,
            "diagnostic_mixed": diagnostic_mixed,
            "strict_candidates": strict_candidates,
            "diagnostic_candidates": cand_rows,
        })

    strict_count = sum(1 for t in tournaments if t["strict_clean"])
    diag_count = sum(1 for t in tournaments if t["diagnostic_mixed"])
    if strict_count >= 5:
        verdict = "CLEAN"
        primary = "strict_clean"
        primary_tournaments = [t for t in tournaments if t["strict_clean"]]
    elif diag_count >= 5:
        verdict = "DIAGNOSTIC_ONLY"
        primary = "diagnostic_mixed"
        primary_tournaments = [t for t in tournaments if t["diagnostic_mixed"]]
    else:
        verdict = "TOO_FEW_TOURNAMENTS"
        primary = "none"
        primary_tournaments = []

    if primary_tournaments:
        baseline = mean(
            sum(1 for c in (t["strict_candidates"] if primary == "strict_clean" else t["diagnostic_candidates"]) if c["is_correct"])
            / max(len(t["strict_candidates"] if primary == "strict_clean" else t["diagnostic_candidates"]), 1)
            for t in primary_tournaments
        )
    else:
        baseline = float("nan")
    incorrect = [row for row in rows if row["unit_test_label"] != "correct"]
    near = sum(1 for row in incorrect if row["unit_test_label"] == "near_miss")
    summary = {
        "tasks_attempted": len({row["task_id"] for row in rows}),
        "candidates_evaluated": len(rows),
        "correct_candidates": sum(1 for row in rows if row["unit_test_label"] == "correct"),
        "near_miss_candidates": sum(1 for row in rows if row["unit_test_label"] == "near_miss"),
        "nonsense_candidates": sum(1 for row in rows if row["unit_test_label"] == "nonsense"),
        "strict_clean_tournaments": strict_count,
        "diagnostic_mixed_tournaments": diag_count,
        "random_top1_baseline": baseline,
        "pass_rate_distribution": dict(Counter(str(row["pass_rate"]) for row in rows)),
        "near_miss_fraction": near / max(len(incorrect), 1),
        "stage_breakdown": dict(Counter(row["candidate_stage"] for row in rows)),
        "label_by_stage": {
            stage: dict(Counter(row["unit_test_label"] for row in rows if row["candidate_stage"] == stage))
            for stage in sorted({row["candidate_stage"] for row in rows})
        },
    }
    payload = {
        "code_tournament_verdict": verdict,
        "primary_eval_set": primary,
        "taskset": repo_path(Path(args.taskset)),
        "candidates_json": repo_path(Path(args.candidates)),
        "summary": summary,
        "candidate_evaluations": rows,
        "tournaments": tournaments,
        "recommended_next_if_stopped": (
            "increase task count, adjust generation modes, capture first-tool code, reduce repair strength if all branches become correct, or increase repair if all branches are nonsense"
            if verdict == "TOO_FEW_TOURNAMENTS"
            else ""
        ),
    }
    out_json = Path(args.output)
    out_md = Path(args.output_md)
    write_json(out_json, payload)
    write_md(out_md, payload)
    print(f"CODE_TOURNAMENT_VERDICT = {verdict}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
