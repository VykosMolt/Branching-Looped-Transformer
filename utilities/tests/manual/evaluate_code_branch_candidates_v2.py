"""Unit-test v2 code candidates and construct branch-selection tournaments."""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from code_branch_pilot_lib import PROJECT_ROOT, REPORT_DIR, load_json, output_path, repo_path, snippet, write_json

LOCAL_AGENT_DIR = PROJECT_ROOT / "shared/src" / "local_agent"
if str(LOCAL_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_AGENT_DIR))

try:
    import ouro_agent_improved as _agent
except Exception:
    _agent = None


DEFAULT_CANDIDATES = REPORT_DIR / "code_branch_candidates_v2_2026-05-16.json"
DEFAULT_TASKSET = REPORT_DIR / "code_branch_taskset_v2_2026-05-16.json"
DEFAULT_OUTPUT = REPORT_DIR / "code_branch_tournaments_v2_2026-05-16.json"
DEFAULT_MD = REPORT_DIR / "code_branch_tournaments_v2_2026-05-16.md"
DEFAULT_PARTIAL = REPORT_DIR / "code_branch_tournaments_v2_2026-05-16.partial.json"
SUMMARY_JSON = REPORT_DIR / "code_branch_pilot_v2_2026-05-16_summary.json"
SUMMARY_MD = REPORT_DIR / "code_branch_pilot_v2_2026-05-16_summary.md"


SAFE_IMPORT_ROOTS = {
    "bisect",
    "collections",
    "copy",
    "dataclasses",
    "functools",
    "heapq",
    "itertools",
    "math",
    "operator",
    "random",
    "re",
    "statistics",
    "string",
    "typing",
}
DANGEROUS_IMPORT_ROOTS = {
    "ctypes",
    "multiprocessing",
    "os",
    "pathlib",
    "requests",
    "shutil",
    "signal",
    "socket",
    "subprocess",
    "sys",
    "urllib",
}
DANGEROUS_CALL_NAMES = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "input",
    "open",
}
DANGEROUS_ATTR_ROOTS = set(DANGEROUS_IMPORT_ROOTS)
SAFE_SYS_IMPORT_NAMES = {"setrecursionlimit"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--taskset", default=str(DEFAULT_TASKSET))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--output-md", default="")
    parser.add_argument("--partial-output", default="")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--default-timeout", type=float, default=5.0)
    args = parser.parse_args()
    output = Path(args.output)
    if not args.output_md:
        args.output_md = str(output.with_suffix(".md"))
    if not args.partial_output:
        args.partial_output = str(output.with_name(output.stem + ".partial.json"))
    return args


def safety_check(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code or "")
    except SyntaxError as exc:
        return True, f"syntax_deferred:{exc}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root == "sys":
                    continue
                if root in DANGEROUS_IMPORT_ROOTS or root not in SAFE_IMPORT_ROOTS:
                    return False, f"unsafe import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root == "sys" and all(alias.name in SAFE_SYS_IMPORT_NAMES for alias in node.names):
                continue
            if root in DANGEROUS_IMPORT_ROOTS or root not in SAFE_IMPORT_ROOTS:
                return False, f"unsafe import: {node.module}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in DANGEROUS_CALL_NAMES:
                return False, f"unsafe call: {node.func.id}"
            if isinstance(node.func, ast.Attribute):
                root = node.func.value
                while isinstance(root, ast.Attribute):
                    root = root.value
                if (
                    isinstance(root, ast.Name)
                    and root.id == "sys"
                    and node.func.attr in SAFE_SYS_IMPORT_NAMES
                ):
                    continue
                if isinstance(root, ast.Name) and root.id in DANGEROUS_ATTR_ROOTS:
                    return False, f"unsafe attribute call: {root.id}.{node.func.attr}"
        elif isinstance(node, ast.Attribute):
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id == "sys" and node.attr not in SAFE_SYS_IMPORT_NAMES:
                return False, f"unsafe sys attribute: sys.{node.attr}"
    return True, ""


def eval_candidate(code: str, tests: list[str], function_name: str, timeout: float) -> dict[str, Any]:
    safety_ok, safety_reason = safety_check(code)
    if not safety_ok:
        return {
            "syntax_ok": False,
            "import_ok": False,
            "runtime_ok": False,
            "safety_ok": False,
            "tests_total": len(tests),
            "tests_passed": 0,
            "tests_failed": len(tests),
            "pass_rate": 0.0,
            "error_type": "SafetyRejected",
            "error_message_short": safety_reason[:300],
            "execution_seconds": 0.0,
            "stdout_short": "",
            "stderr_short": "",
        }
    harness = {"code": code, "tests": tests, "function_name": function_name}
    script = textwrap.dedent(
        """
        import contextlib, io, json, sys, time
        payload=json.loads(sys.stdin.read())
        code=payload["code"]; tests=payload["tests"]; fn_name=payload["function_name"]
        stdout=io.StringIO(); stderr=io.StringIO(); started=time.perf_counter()
        result={
            "syntax_ok": False, "import_ok": False, "runtime_ok": False, "safety_ok": True,
            "tests_total": len(tests), "tests_passed": 0, "tests_failed": len(tests),
            "error_type": "", "error_message_short": "", "stdout_short": "", "stderr_short": "",
        }
        scope={"__name__": "__candidate_eval__"}
        try:
            compiled=compile(code, "<candidate>", "exec")
            result["syntax_ok"]=True
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compiled, scope, scope)
            result["import_ok"]=True
            if fn_name and not callable(scope.get(fn_name)):
                raise AssertionError(f"function not found: {fn_name}")
            for test in tests:
                try:
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        exec(test, scope, scope)
                    result["tests_passed"] += 1
                except Exception as exc:
                    if not result["error_type"]:
                        result["error_type"] = type(exc).__name__
                        result["error_message_short"] = str(exc)[:300]
            result["tests_failed"] = len(tests) - result["tests_passed"]
            result["runtime_ok"] = True
        except Exception as exc:
            result["error_type"] = type(exc).__name__
            result["error_message_short"] = str(exc)[:300]
        result["execution_seconds"] = time.perf_counter() - started
        result["stdout_short"] = stdout.getvalue()[-300:]
        result["stderr_short"] = stderr.getvalue()[-300:]
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
            "safety_ok": True,
            "tests_total": len(tests),
            "tests_passed": 0,
            "tests_failed": len(tests),
            "pass_rate": 0.0,
            "error_type": "TimeoutExpired",
            "error_message_short": "candidate timed out",
            "execution_seconds": float(timeout),
            "stdout_short": "",
            "stderr_short": "",
        }
    result = None
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("JSON_RESULT:"):
            result = json.loads(line[len("JSON_RESULT:"):])
            break
    if result is None:
        result = {
            "syntax_ok": False,
            "import_ok": False,
            "runtime_ok": False,
            "safety_ok": True,
            "tests_total": len(tests),
            "tests_passed": 0,
            "tests_failed": len(tests),
            "error_type": "HarnessParseError",
            "error_message_short": (proc.stderr or proc.stdout)[-300:],
            "execution_seconds": 0.0,
            "stdout_short": proc.stdout[-300:],
            "stderr_short": proc.stderr[-300:],
        }
    result["pass_rate"] = float(result["tests_passed"]) / max(int(result["tests_total"]), 1)
    result.setdefault("safety_ok", True)
    return result


_NON_CODE_PREFIXES = (
    "[Max steps reached",
    "Okay,",
    "The function",
    "The task",
    "To solve",
    "We need",
    "First,",
)


def has_expected_function_shape(code: str, function_name: str) -> bool:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return False
    functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    if function_name:
        return function_name in functions
    return bool(functions)


def looks_like_non_code_payload(code: str) -> bool:
    stripped = (code or "").strip()
    if not stripped:
        return True
    if stripped.startswith(_NON_CODE_PREFIXES) and not re.search(r"^\s*(def|class|from|import)\b", stripped, flags=re.MULTILINE):
        return True
    return False


def label_from_eval(row: dict[str, Any], code: str, function_name: str) -> str:
    if row.get("safety_ok") is False:
        return "safety_rejected"
    if row.get("tests_total", 0) > 0 and row.get("tests_passed") == row.get("tests_total"):
        return "correct"
    if row.get("safety_ok") and row.get("syntax_ok") and row.get("import_ok") and row.get("runtime_ok") and row.get("tests_passed", 0) > 0:
        return "near_miss"
    if (
        row.get("safety_ok")
        and row.get("syntax_ok")
        and row.get("import_ok")
        and row.get("runtime_ok")
        and row.get("tests_passed", 0) == 0
        and has_expected_function_shape(code, function_name)
    ):
        return "wrong_code"
    if row.get("safety_ok") and row.get("syntax_ok") and row.get("import_ok") and has_expected_function_shape(code, function_name):
        return "runtime_error"
    return "malformed"


def legacy_label_from_eval(row: dict[str, Any]) -> str:
    if row.get("safety_ok") is False:
        return "nonsense"
    if row.get("tests_total", 0) > 0 and row.get("tests_passed") == row.get("tests_total"):
        return "correct"
    if row.get("safety_ok") and row.get("syntax_ok") and row.get("import_ok") and row.get("runtime_ok") and row.get("tests_passed", 0) > 0:
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


def candidate_pool(tournament: dict[str, Any], primary: str) -> list[dict[str, Any]]:
    if primary == "strict_clean":
        return list(tournament["strict_candidates"])
    if primary == "diagnostic_runnable":
        return list(tournament["diagnostic_runnable_candidates"])
    if primary == "diagnostic_mixed":
        return list(tournament.get("diagnostic_mixed_primary_candidates", tournament["diagnostic_candidates"]))
    return []


def write_eval_checkpoint(
    path: str,
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    completed_candidate_keys: list[str],
    completed_task_ids: list[str],
    tournament_payload: dict[str, Any] | None = None,
) -> None:
    payload = {
        "checkpoint_type": "code_branch_candidates_v2_eval_partial",
        "complete": False,
        "output": str(args.output),
        "completed_candidate_keys": completed_candidate_keys,
        "completed_task_ids": completed_task_ids,
        "candidate_evaluations": rows,
        "summary": {
            "candidates_evaluated": len(rows),
            "label_counts": dict(Counter(row.get("unit_test_label") for row in rows)),
            "legacy_label_counts": dict(Counter(row.get("legacy_unit_test_label") for row in rows)),
            "stage_breakdown": dict(Counter(row.get("candidate_stage", "unknown") for row in rows)),
        },
    }
    if tournament_payload:
        payload.update({
            "code_v2_tournament_verdict": tournament_payload.get("code_v2_tournament_verdict"),
            "primary_eval_set": tournament_payload.get("primary_eval_set"),
            "summary": tournament_payload.get("summary", payload["summary"]),
            "tournaments": tournament_payload.get("tournaments", []),
            "primary_tournament_ids": tournament_payload.get("primary_tournament_ids", []),
        })
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(checkpoint_path)


def load_eval_checkpoint(path: str, resume: bool) -> tuple[list[dict[str, Any]], set[str], list[str], list[str]]:
    if not resume or not Path(path).exists():
        return [], set(), [], []
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return [], set(), [], []
    rows = list(payload.get("candidate_evaluations", []))
    keys = list(payload.get("completed_candidate_keys", []))
    task_ids = list(payload.get("completed_task_ids", []))
    return rows, set(keys), keys, task_ids


def candidate_eval_key(candidate: dict[str, Any]) -> str:
    return "|".join(str(candidate.get(k, "")) for k in ("task_id", "candidate_index", "ast_hash", "normalized_code_hash", "mode"))


def random_baseline(tournaments: list[dict[str, Any]], primary: str) -> float:
    if not tournaments:
        return float("nan")
    return mean(
        sum(1 for cand in candidate_pool(t, primary) if cand["is_correct"]) / max(len(candidate_pool(t, primary)), 1)
        for t in tournaments
    )


def source_difficulty_breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("source", "difficulty", "candidate_stage"):
        out[key] = {
            value: dict(Counter(row["unit_test_label"] for row in rows if row.get(key, "unknown") == value))
            for value in sorted({str(row.get(key, "unknown")) for row in rows})
        }
    return out


def build_tournament_payload(
    *,
    rows: list[dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
    candidate_payload: dict[str, Any],
) -> dict[str, Any]:
    tournaments = []
    for task_id, task in tasks.items():
        cand_rows = [row for row in rows if row["task_id"] == task_id]
        if len(cand_rows) < 2:
            continue
        correct = [row for row in cand_rows if row["unit_test_label"] == "correct"]
        incorrect = [row for row in cand_rows if row["unit_test_label"] != "correct"]
        near = [row for row in cand_rows if row["unit_test_label"] == "near_miss"]
        wrong_for_task = [row for row in cand_rows if row["unit_test_label"] == "wrong_code"]
        runnable = [row for row in cand_rows if row["is_runnable"]]
        strict_candidates = [row for row in cand_rows if row["unit_test_label"] in {"correct", "near_miss"}]
        diagnostic_runnable_candidates = [
            row for row in cand_rows if row["unit_test_label"] in {"correct", "near_miss", "wrong_code"}
        ]
        diagnostic_mixed_primary_candidates = [
            row for row in cand_rows if row["unit_test_label"] not in {"malformed", "safety_rejected"}
        ]
        strict_clean = bool(correct and near and len(runnable) >= 2)
        diagnostic_runnable = bool(correct and (near or wrong_for_task) and len(diagnostic_runnable_candidates) >= 2)
        diagnostic_mixed = bool(correct and incorrect)
        too_easy = bool(len(cand_rows) >= 2 and len(correct) == len(cand_rows))
        too_hard = bool(not correct)
        all_nonsense = bool(cand_rows and all(row["legacy_unit_test_label"] == "nonsense" for row in cand_rows))
        all_malformed = bool(cand_rows and all(row["unit_test_label"] == "malformed" for row in cand_rows))
        tournaments.append({
            "tournament_id": len(tournaments),
            "task_id": task_id,
            "source": task["source"],
            "difficulty": task.get("difficulty", "unknown"),
            "prompt": task["prompt"],
            "function_name": task["function_name"],
            "strict_clean": strict_clean,
            "diagnostic_runnable": diagnostic_runnable,
            "diagnostic_mixed": diagnostic_mixed,
            "too_easy": too_easy,
            "too_hard": too_hard,
            "all_nonsense": all_nonsense,
            "all_malformed": all_malformed,
            "strict_candidates": strict_candidates,
            "diagnostic_runnable_candidates": diagnostic_runnable_candidates,
            "diagnostic_mixed_primary_candidates": diagnostic_mixed_primary_candidates,
            "diagnostic_candidates": cand_rows,
        })

    strict = [t for t in tournaments if t["strict_clean"]]
    diagnostic_runnable_tournaments = [t for t in tournaments if t["diagnostic_runnable"]]
    diagnostic = [t for t in tournaments if t["diagnostic_mixed"]]
    if len(strict) >= 5:
        verdict = "CLEAN"
        primary = "strict_clean"
        primary_tournaments = strict
    elif len(diagnostic_runnable_tournaments) >= 8:
        verdict = "RUNNABLE_DIAGNOSTIC"
        primary = "diagnostic_runnable"
        primary_tournaments = diagnostic_runnable_tournaments
    elif len(diagnostic) >= 10:
        verdict = "BROAD_DIAGNOSTIC"
        primary = "diagnostic_mixed"
        primary_tournaments = diagnostic
    else:
        correct_rate = sum(1 for row in rows if row["unit_test_label"] == "correct") / max(len(rows), 1)
        if correct_rate > 0.75:
            verdict = "TOO_EASY"
        elif correct_rate < 0.10:
            verdict = "TOO_HARD"
        else:
            verdict = "TOO_FEW_TOURNAMENTS"
        primary = "none"
        primary_tournaments = []

    incorrect_rows = [row for row in rows if row["unit_test_label"] != "correct"]
    near_rows = [row for row in incorrect_rows if row["unit_test_label"] == "near_miss"]
    wrong_rows = [row for row in rows if row["unit_test_label"] == "wrong_code"]
    runtime_error_rows = [row for row in rows if row["unit_test_label"] == "runtime_error"]
    malformed_rows = [row for row in rows if row["unit_test_label"] == "malformed"]
    safety_rows = [row for row in rows if row["unit_test_label"] == "safety_rejected"]
    legacy_nonsense_rows = [row for row in rows if row["legacy_unit_test_label"] == "nonsense"]
    stage_labels = {
        stage: dict(Counter(row["unit_test_label"] for row in rows if row.get("candidate_stage") == stage))
        for stage in sorted({row.get("candidate_stage", "unknown") for row in rows})
    }
    duplicate_rate = candidate_payload.get("summary", {}).get("duplicate_rate", 0.0)
    breakdown = source_difficulty_breakdown(rows) if rows else {"source": {}, "difficulty": {}, "candidate_stage": {}}
    summary = {
        "tasks_attempted": len({row["task_id"] for row in rows}),
        "candidates_generated": candidate_payload.get("summary", {}).get("total_generation_attempts_recorded"),
        "unique_candidates": len(candidate_payload.get("candidates", [])),
        "duplicate_rate": duplicate_rate,
        "candidates_evaluated": len(rows),
        "correct_candidates": sum(1 for row in rows if row["unit_test_label"] == "correct"),
        "near_miss_candidates": len(near_rows),
        "wrong_code_candidates": len(wrong_rows),
        "runtime_error_candidates": len(runtime_error_rows),
        "malformed_candidates": len(malformed_rows),
        "safety_rejected_candidates": len(safety_rows),
        "legacy_nonsense_candidates": len(legacy_nonsense_rows),
        "nonsense_candidates": len(legacy_nonsense_rows),
        "strict_clean_tournaments": len(strict),
        "diagnostic_runnable_tournaments": len(diagnostic_runnable_tournaments),
        "diagnostic_mixed_tournaments": len(diagnostic),
        "too_easy_tasks": sum(1 for t in tournaments if t["too_easy"]),
        "too_hard_tasks": sum(1 for t in tournaments if t["too_hard"]),
        "all_nonsense_tasks": sum(1 for t in tournaments if t["all_nonsense"]),
        "all_malformed_tasks": sum(1 for t in tournaments if t["all_malformed"]),
        "random_top1_baseline": random_baseline(primary_tournaments, primary) if primary_tournaments else float("nan"),
        "random_top1_baselines": {
            "strict_clean": random_baseline(strict, "strict_clean") if strict else float("nan"),
            "diagnostic_runnable": random_baseline(diagnostic_runnable_tournaments, "diagnostic_runnable") if diagnostic_runnable_tournaments else float("nan"),
            "diagnostic_mixed": random_baseline(diagnostic, "diagnostic_mixed") if diagnostic else float("nan"),
        },
        "pass_rate_distribution": dict(Counter(str(row["pass_rate"]) for row in rows)),
        "near_miss_fraction": len(near_rows) / max(len(incorrect_rows), 1),
        "code_like_wrong_fraction": (len(near_rows) + len(wrong_rows) + len(runtime_error_rows)) / max(len(incorrect_rows), 1),
        "source_breakdown": breakdown["source"],
        "difficulty_breakdown": breakdown["difficulty"],
        "candidate_stage_breakdown": breakdown["candidate_stage"],
        "stage_breakdown": dict(Counter(row.get("candidate_stage", "unknown") for row in rows)),
        "label_by_stage": stage_labels,
        "correct_candidate_count_distribution": dict(Counter(sum(1 for c in t["diagnostic_candidates"] if c["is_correct"]) for t in diagnostic)),
        "mbpp_232_outcome": next((dict(Counter(c["unit_test_label"] for c in t["diagnostic_candidates"])) for t in tournaments if t["task_id"] == "mbpp/232"), {}),
        "mbpp_306_outcome": next((dict(Counter(c["unit_test_label"] for c in t["diagnostic_candidates"])) for t in tournaments if t["task_id"] == "mbpp/306"), {}),
        "malformed_or_prose_candidates": len(malformed_rows),
    }
    return {
        "code_v2_tournament_verdict": verdict,
        "primary_eval_set": primary,
        "summary": summary,
        "tournaments": tournaments,
        "primary_tournament_ids": [t["tournament_id"] for t in primary_tournaments],
        "strict": strict,
        "diagnostic_runnable": diagnostic_runnable_tournaments,
        "diagnostic": diagnostic,
        "primary_tournaments": primary_tournaments,
    }


def recommended_next(verdicts: dict[str, str]) -> str:
    if verdicts["CODE_V2_INTERFACE_VERDICT"] == "BLOCKED":
        return "fix_local_agent_invocation_before_code_pilot"
    if verdicts["CODE_V2_TASKSET_VERDICT"] == "BLOCKED":
        return "restore_or_add_harder_code_taskset"
    if verdicts["CODE_V2_GENERATION_VERDICT"] == "WRAPPER_BLOCKED":
        return "inspect_local_agent_wrapper_or_use_direct_route_only"
    if verdicts["CODE_V2_GENERATION_VERDICT"] == "TOO_DUPLICATE":
        return "reduce_final_outputs_and_harvest_pre_repair_candidates"
    if verdicts["CODE_V2_TOURNAMENT_VERDICT"] == "TOO_EASY":
        return "add_harder_tasks_and_disable_final_repair_candidates"
    if verdicts["CODE_V2_TOURNAMENT_VERDICT"] == "TOO_HARD":
        return "add_medium_tasks_or_allow_repaired_final_candidates"
    if verdicts["CODE_V2_TOURNAMENT_VERDICT"] == "TOO_FEW_TOURNAMENTS":
        return "increase_tasks_and_candidate_modes"
    return "expand_code_pilot_or_add_code_specific_training_control"


def append_docs(summary: dict[str, Any]) -> list[str]:
    docs = [
        PROJECT_ROOT / "docs/evaluator/evaluator_domain_transfer_notes.md",
        PROJECT_ROOT / "docs/evaluator/math_bg_gate_pilot_2026-05-15.md",
        PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-15_v4.md",
        PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-16_v5_clean_gsm8k_code_next.md",
    ]
    title = "## Code branch pilot v2 (2026-05-16)"
    text = "\n".join([
        "",
        title,
        "",
        f"- CODE_V2_INTERFACE_VERDICT: `{summary['CODE_V2_INTERFACE_VERDICT']}`",
        f"- CODE_V2_TASKSET_VERDICT: `{summary['CODE_V2_TASKSET_VERDICT']}`",
        f"- CODE_V2_GENERATION_VERDICT: `{summary['CODE_V2_GENERATION_VERDICT']}`",
        f"- CODE_V2_TOURNAMENT_VERDICT: `{summary['CODE_V2_TOURNAMENT_VERDICT']}`",
        f"- CODE_V2_TRANSFER_VERDICT: `{summary['CODE_V2_TRANSFER_VERDICT']}`",
        f"- tasks: `{summary['tasks']}`",
        f"- candidates: `{summary['candidates']}`",
        f"- duplicate rate: `{summary['duplicate_rate']}`",
        f"- unit-test label counts: `{summary['label_counts']}`",
        f"- strict_clean tournaments: `{summary['strict_clean_tournaments']}`",
        f"- diagnostic_mixed tournaments: `{summary['diagnostic_mixed_tournaments']}`",
        f"- random_top1_baseline: `{summary['random_top1_baseline']}`",
        f"- best AntisymLinear row: `{summary.get('best_antisymlinear', 'NOT_RUN')}`",
        f"- best NoNorm row: `{summary.get('best_nonorm', 'NOT_RUN')}`",
        f"- candidate-stage harvesting fixed v1 distribution: `{summary['stage_harvesting_interpretation']}`",
        "- full report: `opi/taps/probes/code_branch_pilot_v2_2026-05-16_summary.md`",
        f"- interpretation: {summary['interpretation']}",
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


def write_summary(summary: dict[str, Any]) -> None:
    lines = [
        "# Code Branch Pilot v2 Summary",
        "",
        f"CODE_V2_INTERFACE_VERDICT = {summary['CODE_V2_INTERFACE_VERDICT']}",
        f"CODE_V2_TASKSET_VERDICT = {summary['CODE_V2_TASKSET_VERDICT']}",
        f"CODE_V2_GENERATION_VERDICT = {summary['CODE_V2_GENERATION_VERDICT']}",
        f"CODE_V2_TOURNAMENT_VERDICT = {summary['CODE_V2_TOURNAMENT_VERDICT']}",
        f"CODE_V2_TRANSFER_VERDICT = {summary['CODE_V2_TRANSFER_VERDICT']}",
        f"RECOMMENDED_NEXT = {summary['RECOMMENDED_NEXT']}",
        "",
        "## Interface v2 Summary",
        "",
        f"- interface report: `{summary['interface_report']}`",
        "",
        "## Taskset v2 Summary",
        "",
        f"- tasks: `{summary['tasks']}`",
        f"- source_mix: `{summary['source_mix']}`",
        f"- difficulty_mix: `{summary['difficulty_mix']}`",
        "",
        "## Candidate Generation Diversity Summary",
        "",
        f"- candidates: `{summary['candidates']}`",
        f"- duplicate_rate: `{summary['duplicate_rate']}`",
        f"- stage_breakdown: `{summary['stage_breakdown']}`",
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
        f"- random_top1_baseline: `{summary['random_top1_baseline']}`",
        "",
        "## Feature Capture Summary",
        "",
        "Feature capture was not run because the v2 tournament verdict did not meet the required threshold.",
        "",
        "## HH-Trained AntisymLinear / NoNorm Transfer Table",
        "",
        "Transfer was not run.",
        "",
        "## Relation To v1 Code Pilot",
        "",
        f"- v1 relation: `{summary['v1_relation']}`",
        "",
        "## Relation To Expanded Clean GSM8K Result",
        "",
        "Prior clean GSM8K expanded result: `EXPANDED_LINEAR_TRANSFER_VERDICT = GOOD`; `GRU_CONTROL_VERDICT = GRU_WEAK`.",
        "",
        "## Markdown Docs Updated",
        "",
    ]
    lines.extend(f"- `{path}`" for path in summary.get("docs_updated", []))
    lines.extend(["", "## Files Modified / Created", ""])
    lines.extend(f"- `{path}`" for path in summary.get("files_created", []))
    lines.extend(["", "## Commands Run", "", "```bash"])
    lines.extend(summary.get("commands_run", []))
    lines.extend(["```", "", "## Blockers", "", summary.get("blockers") or "None.", ""])
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")
    write_json(SUMMARY_JSON, summary)


def write_md(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# Code Branch Tournaments v2",
        "",
        f"CODE_V2_TOURNAMENT_VERDICT = {payload['code_v2_tournament_verdict']}",
        "",
        f"- primary_eval_set: `{payload['primary_eval_set']}`",
        f"- tasks_attempted: `{s['tasks_attempted']}`",
        f"- candidates_evaluated: `{s['candidates_evaluated']}`",
        f"- duplicate_rate: `{s['duplicate_rate']}`",
        f"- correct_candidates: `{s['correct_candidates']}`",
        f"- near_miss_candidates: `{s['near_miss_candidates']}`",
        f"- wrong_code_candidates: `{s.get('wrong_code_candidates', 0)}`",
        f"- runtime_error_candidates: `{s.get('runtime_error_candidates', 0)}`",
        f"- malformed_candidates: `{s.get('malformed_candidates', 0)}`",
        f"- safety_rejected_candidates: `{s.get('safety_rejected_candidates', 0)}`",
        f"- legacy_nonsense_candidates: `{s.get('legacy_nonsense_candidates', s['nonsense_candidates'])}`",
        f"- strict_clean_tournaments: `{s['strict_clean_tournaments']}`",
        f"- diagnostic_runnable_tournaments: `{s.get('diagnostic_runnable_tournaments', 0)}`",
        f"- diagnostic_mixed_tournaments: `{s['diagnostic_mixed_tournaments']}`",
        f"- too_easy_tasks: `{s['too_easy_tasks']}`",
        f"- too_hard_tasks: `{s['too_hard_tasks']}`",
        f"- random_top1_baseline: `{s['random_top1_baseline']}`",
        f"- random_top1_baselines: `{s.get('random_top1_baselines', {})}`",
        f"- mbpp_232_outcome: `{s.get('mbpp_232_outcome', {})}`",
        f"- mbpp_306_outcome: `{s.get('mbpp_306_outcome', {})}`",
        f"- near_miss_fraction: `{s['near_miss_fraction']}`",
        f"- code_like_wrong_fraction: `{s.get('code_like_wrong_fraction', 0.0)}`",
        "",
        "## Tournaments",
        "",
        "| task_id | source | difficulty | strict_clean | diagnostic_runnable | diagnostic_mixed | too_easy | too_hard | labels |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for t in payload["tournaments"]:
        labels = Counter(c["unit_test_label"] for c in t["diagnostic_candidates"])
        lines.append(
            f"| `{t['task_id']}` | `{t['source']}` | `{t['difficulty']}` | {t['strict_clean']} | "
            f"{t.get('diagnostic_runnable', False)} | {t['diagnostic_mixed']} | {t['too_easy']} | {t['too_hard']} | `{dict(labels)}` |"
        )
    lines.extend(["", "## Candidate Examples", ""])
    for row in payload["candidate_evaluations"][:24]:
        lines.append(
            f"- `{row['task_id']}` `{row['mode']}` `{row['candidate_stage']}` "
            f"label=`{row['unit_test_label']}` pass={row['tests_passed']}/{row['tests_total']} "
            f"legacy=`{row.get('legacy_unit_test_label', row['unit_test_label'])}` safety={row['safety_ok']} "
            f"err=`{row['error_type']}` code={snippet(row['final_code'], 120)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    taskset = load_json(args.taskset)
    candidate_payload = load_json(args.candidates)
    tasks = {task["task_id"]: task for task in taskset["tasks"]}
    rows, completed_keys, completed_key_order, completed_task_ids = load_eval_checkpoint(args.partial_output, bool(args.resume))
    completed_task_set = set(completed_task_ids)
    if rows:
        print(f"[code-v2-eval] resuming from {args.partial_output}: {len(rows)} candidates evaluated", flush=True)
    candidates_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cand in candidate_payload.get("candidates", []):
        candidates_by_task[str(cand.get("task_id", ""))].append(cand)
    for task_id, task_candidates in candidates_by_task.items():
        if task_id in completed_task_set:
            continue
        task = tasks.get(task_id)
        if not task:
            continue
        tests = list(task.get("tests") or list(task.get("public_tests", [])) + list(task.get("hidden_tests", [])))
        for cand in task_candidates:
            key = candidate_eval_key(cand)
            if key in completed_keys:
                continue
            eval_code, recovered = candidate_code_for_unit_tests(cand, task)
            eval_result = eval_candidate(
                eval_code,
                tests,
                task.get("function_name", ""),
                float(task.get("timeout_seconds") or args.default_timeout),
            )
            label = label_from_eval(eval_result, eval_code, task.get("function_name", ""))
            legacy_label = legacy_label_from_eval(eval_result)
            row = {
                **cand,
                "raw_final_code": cand.get("final_code", ""),
                "final_code": eval_code,
                "candidate_code_recovered_by_evaluator": recovered,
                **eval_result,
                "unit_test_label": label,
                "legacy_unit_test_label": legacy_label,
                "is_malformed": label == "malformed",
                "is_code_like_wrong": label in {"wrong_code", "runtime_error", "near_miss"},
                "is_correct": label == "correct",
                "is_runnable": bool(eval_result.get("safety_ok") and eval_result.get("syntax_ok") and eval_result.get("import_ok") and eval_result.get("runtime_ok")),
            }
            rows.append(row)
            completed_keys.add(key)
            completed_key_order.append(key)
        completed_task_ids.append(task_id)
        completed_task_set.add(task_id)
        interim = build_tournament_payload(rows=rows, tasks=tasks, candidate_payload=candidate_payload)
        write_eval_checkpoint(
            args.partial_output,
            args=args,
            rows=rows,
            completed_candidate_keys=completed_key_order,
            completed_task_ids=completed_task_ids,
            tournament_payload=interim,
        )
    state = build_tournament_payload(rows=rows, tasks=tasks, candidate_payload=candidate_payload)
    verdict = state["code_v2_tournament_verdict"]
    primary = state["primary_eval_set"]
    tournaments = state["tournaments"]
    strict = state["strict"]
    diagnostic = state["diagnostic"]
    primary_tournaments = state["primary_tournaments"]
    summary = state["summary"]
    payload = {
        "code_v2_tournament_verdict": verdict,
        "primary_eval_set": primary,
        "taskset": repo_path(Path(args.taskset)),
        "candidates_json": repo_path(Path(args.candidates)),
        "summary": summary,
        "candidate_evaluations": rows,
        "tournaments": tournaments,
        "primary_tournament_ids": state["primary_tournament_ids"],
        "recommended_next_if_stopped": (
            "add harder tasks and disable final repaired candidates"
            if verdict == "TOO_EASY"
            else "add medium tasks or allow repaired final candidates"
            if verdict == "TOO_HARD"
            else "increase tasks or candidate modes"
            if verdict == "TOO_FEW_TOURNAMENTS"
            else ""
        ),
    }
    write_eval_checkpoint(
        args.partial_output,
        args=args,
        rows=rows,
        completed_candidate_keys=completed_key_order,
        completed_task_ids=completed_task_ids,
        tournament_payload=state,
    )
    partial_payload = json.loads(Path(args.partial_output).read_text(encoding="utf-8"))
    partial_payload["complete"] = True
    partial_payload["code_v2_tournament_verdict"] = verdict
    partial_payload["primary_eval_set"] = primary
    partial_payload["primary_tournament_ids"] = state["primary_tournament_ids"]
    Path(args.partial_output).write_text(json.dumps(partial_payload, indent=2, default=str) + "\n", encoding="utf-8")
    out_json = output_path(args.output)
    out_md = output_path(args.output_md)
    write_json(out_json, payload)
    write_md(out_md, payload)
    print(f"CODE_V2_TOURNAMENT_VERDICT = {verdict}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")

    if (
        verdict not in {"CLEAN", "RUNNABLE_DIAGNOSTIC", "BROAD_DIAGNOSTIC"}
        and "mini_patched" not in str(out_json)
        and "near_miss10" not in str(out_json)
    ):
        interface = load_json(REPORT_DIR / "code_branch_v2_interface_inspection_2026-05-16.json")
        verdicts = {
            "CODE_V2_INTERFACE_VERDICT": interface.get("code_v2_interface_verdict", "BLOCKED"),
            "CODE_V2_TASKSET_VERDICT": taskset.get("code_v2_taskset_verdict", "BLOCKED"),
            "CODE_V2_GENERATION_VERDICT": candidate_payload.get("code_v2_generation_verdict", "WRAPPER_BLOCKED"),
            "CODE_V2_TOURNAMENT_VERDICT": verdict,
            "CODE_V2_TRANSFER_VERDICT": "NOT_RUN",
        }
        final_summary = {
            **verdicts,
            "RECOMMENDED_NEXT": recommended_next(verdicts),
            "interface_report": "opi/taps/probes/code_branch_v2_interface_inspection_2026-05-16.md",
            "tasks": len(taskset.get("tasks", [])),
            "source_mix": taskset.get("source_mix", {}),
            "difficulty_mix": taskset.get("difficulty_mix", {}),
            "candidates": len(rows),
            "duplicate_rate": duplicate_rate,
            "stage_breakdown": summary["stage_breakdown"],
            "label_counts": {
                "correct": summary["correct_candidates"],
                "near_miss": summary["near_miss_candidates"],
                "wrong_code": summary["wrong_code_candidates"],
                "runtime_error": summary["runtime_error_candidates"],
                "malformed": summary["malformed_candidates"],
                "safety_rejected": summary["safety_rejected_candidates"],
                "legacy_nonsense": summary["legacy_nonsense_candidates"],
            },
            "label_by_stage": summary["label_by_stage"],
            "strict_clean_tournaments": len(strict),
            "diagnostic_runnable_tournaments": summary.get("diagnostic_runnable_tournaments", 0),
            "diagnostic_mixed_tournaments": len(diagnostic),
            "primary_eval_set": primary,
            "random_top1_baseline": summary["random_top1_baseline"],
            "best_antisymlinear": "NOT_RUN",
            "best_nonorm": "NOT_RUN",
            "stage_harvesting_interpretation": "yes" if len(diagnostic) > 4 or len(strict) > 1 else "not_enough",
            "interpretation": "The v2 candidate distribution still did not produce enough usable objective mixed code tournaments for tap transfer.",
            "v1_relation": "v2 increased task count and stage harvesting, but stopped before feature capture due to tournament verdict",
            "docs_updated": [],
            "files_created": [
                "shared/utilities/tests/manual/inspect_code_branch_v2_requirements.py",
                "shared/utilities/tests/manual/build_code_branch_taskset_v2.py",
                "shared/utilities/tests/manual/generate_code_branch_candidates_v2.py",
                "shared/utilities/tests/manual/evaluate_code_branch_candidates_v2.py",
                "shared/utilities/tests/manual/capture_code_branch_tap_features_v2.py",
                "shared/utilities/tests/manual/evaluate_hh_transfer_on_code_branches_v2.py",
                repo_path(out_json),
                repo_path(out_md),
                "opi/taps/probes/code_branch_pilot_v2_2026-05-16_summary.json",
                "opi/taps/probes/code_branch_pilot_v2_2026-05-16_summary.md",
            ],
            "commands_run": [
                "venv/bin/python -u utilities/tests/manual/evaluate_code_branch_candidates_v2.py --candidates opi/taps/probes/code_branch_candidates_v2_2026-05-16.json --taskset opi/taps/probes/code_branch_taskset_v2_2026-05-16.json --output opi/taps/probes/code_branch_tournaments_v2_2026-05-16.json",
            ],
            "blockers": payload["recommended_next_if_stopped"],
        }
        final_summary["docs_updated"] = append_docs(final_summary)
        write_summary(final_summary)
        print(f"Wrote {SUMMARY_JSON}")
        print(f"Wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
