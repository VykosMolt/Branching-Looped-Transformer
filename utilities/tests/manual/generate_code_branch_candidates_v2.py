"""Generate diverse v2 code-branch candidates with pre-final stage harvesting."""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == THIS_DIR:
    sys.path.pop(0)
import torch
CUDA_AVAILABLE_AT_IMPORT = torch.cuda.is_available()
sys.path.insert(0, str(THIS_DIR))

try:
    from utilities.tests.manual.code_branch_pilot_lib import (
        REPORT_DIR,
        configure_local_agent_env,
        import_agent_modules,
        load_json,
        repo_path,
        sanitize_python_candidate,
        snippet,
        write_json,
    )
except ModuleNotFoundError:
    from code_branch_pilot_lib import (
        REPORT_DIR,
        configure_local_agent_env,
        import_agent_modules,
        load_json,
        repo_path,
        sanitize_python_candidate,
        snippet,
        write_json,
    )


DEFAULT_TASKSET = REPORT_DIR / "code_branch_taskset_v2_2026-05-16.json"
DEFAULT_OUTPUT = REPORT_DIR / "code_branch_candidates_v2_2026-05-16.json"
DEFAULT_MD = REPORT_DIR / "code_branch_candidates_v2_2026-05-16.md"
DEFAULT_LOG = REPORT_DIR / "code_branch_candidates_v2_2026-05-16.log"
DEFAULT_PARTIAL = REPORT_DIR / "code_branch_candidates_v2_2026-05-16.partial.json"
MINI_OUTPUT = REPORT_DIR / "code_branch_candidates_v2_mini_patched_2026-05-16.json"


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taskset", default=str(DEFAULT_TASKSET))
    parser.add_argument("--output", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--log-file", default="")
    parser.add_argument("--partial-output", default="")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-tasks", type=int, default=40)
    parser.add_argument("--min-tasks", type=int, default=25)
    parser.add_argument("--max-candidates-per-task", type=int, default=6)
    parser.add_argument("--hard-cap-total-candidates", type=int, default=300)
    parser.add_argument("--target-usable-tournaments", type=int, default=25)
    parser.add_argument("--prefer-prefinal", action="store_true")
    parser.add_argument("--limit-repaired-final-per-task", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--direct-max-tokens", type=int, default=512)
    parser.add_argument("--direct-short-tokens", type=int, default=256)
    parser.add_argument("--first-action-max-tokens", type=int, default=768)
    parser.add_argument("--repair-max-tokens", type=int, default=512)
    args = parser.parse_args()
    taskset_name = Path(args.taskset).name
    default_output = MINI_OUTPUT if "mini_patched" in taskset_name else DEFAULT_OUTPUT
    if not args.output:
        args.output = str(default_output)
    output_path = Path(args.output)
    if not args.output_md:
        args.output_md = str(output_path.with_suffix(".md"))
    if not args.log_file:
        args.log_file = str(output_path.with_suffix(".log"))
    if not args.partial_output:
        args.partial_output = str(output_path.with_name(output_path.stem + ".partial.json"))
    return args


def setup_log_tee(path: str) -> Any:
    if not path or os.environ.get("CODE_BRANCH_V2_LOG_TEE_ACTIVE") == "1":
        return contextlib.nullcontext()
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")

    @contextlib.contextmanager
    def _ctx():
        old_stdout, old_stderr = sys.stdout, sys.stderr
        os.environ["CODE_BRANCH_V2_LOG_TEE_ACTIVE"] = "1"
        sys.stdout = Tee(old_stdout, log)  # type: ignore[assignment]
        sys.stderr = Tee(old_stderr, log)  # type: ignore[assignment]
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            log.close()

    return _ctx()


def sha(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="replace")).hexdigest()


def normalize_code(code: str) -> str:
    return "\n".join(line.rstrip() for line in (code or "").strip().splitlines() if line.strip())


def ast_signature(code: str) -> tuple[str, str]:
    try:
        tree = ast.parse(code or "")
        normalized = ast.unparse(tree)
        return sha(normalized), normalized
    except Exception:
        return "", normalize_code(code)


def hash_fields(raw_code: str, final_code: str) -> dict[str, str]:
    ast_hash, ast_normalized = ast_signature(final_code)
    normalized = ast_normalized if ast_hash else normalize_code(final_code)
    return {
        "raw_code_hash": sha(raw_code or "")[:16] if raw_code else "",
        "normalized_code_hash": sha(normalized)[:16] if normalized else "",
        "ast_hash": ast_hash[:16] if ast_hash else "",
    }


_CODE_SHAPE_RE = re.compile(r"^\s*(def|class|from|import)\s+", re.MULTILINE)
_NON_CODE_PREFIXES = (
    "[Max steps reached",
    "Okay,",
    "The function",
    "The task",
    "To solve",
    "We need",
    "First,",
)


def code_shape_error(code: str) -> str:
    stripped = (code or "").strip()
    if not stripped:
        return "no_final_code"
    if stripped.startswith(_NON_CODE_PREFIXES):
        return "non_code_wrapper_or_prose"
    if not _CODE_SHAPE_RE.search(stripped):
        return "no_python_code_shape"
    return ""


def task_prompt(task: dict[str, Any]) -> str:
    return (
        f"{task['prompt']}\n\n"
        f"Return a complete Python implementation of `{task['function_name']}` in one code block. "
        "Do not use file I/O. Do not read stdin. Do not include unit-test results."
    )


def merge_prefill(raw: str, prefill: str) -> str:
    raw = raw or ""
    return raw if raw.startswith(prefill) else prefill + raw


def finalize_code(agent: Any, task: dict[str, Any], code: str) -> str:
    if not code:
        return ""
    if hasattr(agent, "final_python_code_for_answer"):
        try:
            return agent.final_python_code_for_answer(task_prompt(task), code)
        except Exception:
            return code
    return code


def direct_code_candidate(
    model_mgr: Any,
    modules: dict[str, Any],
    task: dict[str, Any],
    *,
    mode: str,
    stage: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    config = modules["config"]
    prompts = modules["prompts"]
    policies = modules["policies"]
    agent = modules["agent"]
    system = (
        "You are a strong reasoning and coding assistant.\n\n"
        + prompts.OURO_SOLVER_POSTURE
        + "\nYou are running inside a local RLTT wrapper."
        + "\nFor coding tasks, answer code-first. Avoid long approach narration before the code."
    )
    budget_note = "Keep the implementation compact." if stage == "direct_short_budget" else "Return complete code."
    user = (
        "Start immediately with `FINAL ANSWER:` and complete code in a fenced Python code block.\n"
        f"{budget_note}\n"
        "When the final deliverable is complete, write `<END_FINAL>` on its own line.\n\n"
        f"User task:\n{task_prompt(task)}"
    )
    prefill = "FINAL ANSWER:\n```python\n"
    started = time.perf_counter()
    raw = model_mgr.chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": prefill},
        ],
        max_tokens=max(64, min(int(max_tokens), int(config.DIRECT_MAX_TOKENS))),
        temperature=float(temperature),
        runtime_overrides={
            "total_ut_steps": max(1, min(int(config.CODE_FIRST_PASS_UT_STEPS), 2)),
            "stop_strings": ["<END_FINAL>", "```"],
        },
    )
    raw_model_text = merge_prefill(raw, prefill)
    if "<END_FINAL>" in raw_model_text:
        raw_model_text = raw_model_text.split("<END_FINAL>", 1)[0].rstrip()
    code = sanitize_python_candidate(raw_model_text, policies=policies, agent=agent)
    final_code = finalize_code(agent, task, code)
    return {
        "mode": mode,
        "candidate_stage": stage,
        "temperature": float(temperature),
        "route": "local_agent_direct_prompt",
        "raw_model_text": raw_model_text,
        "raw_tool_input": "",
        "sanitized_code": code,
        "final_code": final_code,
        "wallclock_seconds": round(time.perf_counter() - started, 3),
        "token_budget": int(max_tokens),
        "parser_sanitizer_path": "ouro_policies.sanitize_tool_input/extract_python_code/final_python_code_for_answer",
        "generation_error": "" if code else "no_code_extracted",
    }


def action_code_candidate(
    model_mgr: Any,
    modules: dict[str, Any],
    task: dict[str, Any],
    *,
    mode: str,
    stage: str,
    temperature: float,
    max_tokens: int,
    extra_directive: str = "",
) -> dict[str, Any]:
    agent = modules["agent"]
    policies = modules["policies"]
    prompts = modules["prompts"]
    config = modules["config"]
    directive = (
        "Return exactly one next action using:\n"
        "[Action]: python\n"
        "[Input]:\n"
        "<complete Python source code to execute>\n\n"
        "The Python source should define the requested function. "
        "Include a few meaningful asserts only if they help local checking. "
        "Do not write FINAL ANSWER."
    )
    if extra_directive:
        directive += "\n\n" + extra_directive
    user = f"Task:\n{task_prompt(task)}\n\n{directive}"
    prefill = agent.code_action_prefill_for_state(None)
    started = time.perf_counter()
    raw = model_mgr.chat(
        [
            {"role": "system", "content": prompts.SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": prefill},
        ],
        max_tokens=max(64, min(int(max_tokens), int(config.HARD_CODE_FIRST_ACTION_TOKENS))),
        temperature=float(temperature),
        runtime_overrides={
            "total_ut_steps": max(1, min(int(config.CODE_FIRST_PASS_UT_STEPS), 2)),
            "stop_strings": ["\n[Observation]", "\n[System]", "\n[Verifier]", "\nFINAL ANSWER:"],
        },
    )
    raw_model_text = merge_prefill(raw, prefill)
    action, tool_input = agent.parse_action(raw_model_text)
    raw_tool_input = tool_input or ""
    code = policies.sanitize_tool_input(action, raw_tool_input) if action == "python" else ""
    if not code:
        code = sanitize_python_candidate(raw_model_text, policies=policies, agent=agent)
    final_code = finalize_code(agent, task, code)
    return {
        "mode": mode,
        "candidate_stage": stage,
        "temperature": float(temperature),
        "route": "local_agent_action_prefill",
        "raw_model_text": raw_model_text,
        "raw_tool_input": raw_tool_input,
        "sanitized_code": code,
        "final_code": final_code,
        "wallclock_seconds": round(time.perf_counter() - started, 3),
        "token_budget": int(max_tokens),
        "parser_sanitizer_path": "parse_action + ouro_policies.sanitize_tool_input + final_python_code_for_answer",
        "generation_error": "" if code else f"no_python_action_extracted:{action}",
    }


def public_eval(code: str, task: dict[str, Any]) -> dict[str, Any]:
    tests = list(task.get("public_tests", []))
    if not tests:
        tests = list(task.get("hidden_tests", []))[:1]
    script = textwrap.dedent(
        """
        import contextlib, io, json, sys
        payload=json.loads(sys.stdin.read())
        code=payload["code"]; tests=payload["tests"]; fn_name=payload["function_name"]
        result={"ok": False, "tests_total": len(tests), "tests_passed": 0, "error": ""}
        scope={"__name__": "__candidate_public_eval__"}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(code, "<candidate>", "exec"), scope, scope)
            if fn_name and not callable(scope.get(fn_name)):
                raise AssertionError(f"function not found: {fn_name}")
            for test in tests:
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        exec(test, scope, scope)
                    result["tests_passed"] += 1
                except Exception as exc:
                    if not result["error"]:
                        result["error"] = f"{type(exc).__name__}: {exc}"[:500]
            result["ok"] = result["tests_passed"] == len(tests)
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"[:500]
        print(json.dumps(result))
        """
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            input=json.dumps({"code": code, "tests": tests, "function_name": task.get("function_name", "")}),
            text=True,
            capture_output=True,
            timeout=max(1.0, min(float(task.get("timeout_seconds", 5.0)), 5.0)),
            check=False,
        )
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {"ok": False, "tests_total": len(tests), "tests_passed": 0, "error": f"{type(exc).__name__}: {exc}"}


def repair_or_failed_candidate(
    model_mgr: Any,
    modules: dict[str, Any],
    task: dict[str, Any],
    *,
    first_tokens: int,
    repair_tokens: int,
) -> dict[str, Any]:
    agent = modules["agent"]
    policies = modules["policies"]
    prompts = modules["prompts"]
    config = modules["config"]
    first = action_code_candidate(
        model_mgr,
        modules,
        task,
        mode="first_failed_probe",
        stage="first_failed_or_first_repair_code",
        temperature=0.35,
        max_tokens=min(int(first_tokens), 512),
        extra_directive="Prefer a compact first attempt. Do not spend tokens on explanations.",
    )
    first_eval = public_eval(first.get("final_code", ""), task) if first.get("final_code") else {"ok": False, "error": "no first code"}
    if first_eval.get("ok"):
        feedback = (
            "The first attempt passed the public examples. Produce a different compact implementation that still solves the task, "
            "using a different internal route or edge-case handling where possible."
        )
    else:
        feedback = (
            "The first attempt failed public checking. Repair it using the failure below.\n\n"
            f"Failure: {first_eval.get('error') or 'public tests failed'}\n"
            f"Public tests passed: {first_eval.get('tests_passed')}/{first_eval.get('tests_total')}"
        )
    repair_prompt = (
        f"Task:\n{task_prompt(task)}\n\n"
        "Previous candidate code:\n```python\n"
        + str(first.get("final_code", "")).strip()
        + "\n```\n\n"
        + feedback
        + "\n\nReturn exactly one next action using:\n[Action]: python\n[Input]:\n<repaired or alternate Python source code>"
    )
    prefill = agent.code_action_prefill_for_state(None)
    started = time.perf_counter()
    raw = model_mgr.chat(
        [
            {"role": "system", "content": prompts.SYSTEM_PROMPT},
            {"role": "user", "content": repair_prompt},
            {"role": "assistant", "content": prefill},
        ],
        max_tokens=max(64, min(int(repair_tokens), int(config.HARD_CODE_FIRST_ACTION_TOKENS))),
        temperature=0.35,
        runtime_overrides={
            "total_ut_steps": 1,
            "stop_strings": ["\n[Observation]", "\n[System]", "\n[Verifier]", "\nFINAL ANSWER:"],
        },
    )
    raw_model_text = merge_prefill(raw, prefill)
    action, tool_input = agent.parse_action(raw_model_text)
    raw_tool_input = tool_input or ""
    code = policies.sanitize_tool_input(action, raw_tool_input) if action == "python" else ""
    if not code:
        code = sanitize_python_candidate(raw_model_text, policies=policies, agent=agent)
    final_code = finalize_code(agent, task, code)
    return {
        "mode": "first_failed_or_first_repair_code",
        "candidate_stage": "first_failed_or_first_repair_code",
        "temperature": 0.35,
        "route": "local_agent_action_prefill_public_test_repair",
        "raw_model_text": raw_model_text,
        "raw_tool_input": raw_tool_input,
        "sanitized_code": code,
        "final_code": final_code,
        "first_failed_tool_input": first.get("sanitized_code", ""),
        "first_failed_final_code": first.get("final_code", ""),
        "first_public_eval": first_eval,
        "wallclock_seconds": round(first.get("wallclock_seconds", 0.0) + (time.perf_counter() - started), 3),
        "token_budget": int(repair_tokens),
        "parser_sanitizer_path": "action_prefill_public_test_repair + sanitize_tool_input + final_python_code_for_answer",
        "generation_error": "" if code else f"no_repair_python_action_extracted:{action}",
    }


def hard_code_final_candidate(
    model_mgr: Any,
    modules: dict[str, Any],
    project: Any,
    history: list[str],
    classifier: Any,
    self_model: Any,
    task: dict[str, Any],
) -> dict[str, Any]:
    agent = modules["agent"]
    policies = modules["policies"]
    started = time.perf_counter()
    answer = agent.run_task_mode(
        model_mgr,
        project,
        history,
        task_prompt(task),
        mode="agent",
        classifier=classifier,
        self_model=self_model,
        model="rltt",
        ut_steps=2,
        max_tokens=768,
        react_steps=3,
        external_evaluator=None,
    )
    code = sanitize_python_candidate(answer, policies=policies, agent=agent)
    final_code = finalize_code(agent, task, code)
    return {
        "mode": "repaired_final",
        "candidate_stage": "repaired_final",
        "temperature": 0.12,
        "route": "ouro_agent_improved.run_task_mode(agent)",
        "raw_model_text": answer,
        "raw_tool_input": "",
        "sanitized_code": code,
        "final_code": final_code,
        "wallclock_seconds": round(time.perf_counter() - started, 3),
        "token_budget": 768,
        "parser_sanitizer_path": "run_task_mode final + sanitizer + final_python_code_for_answer",
        "generation_error": "" if code else "no_code_extracted",
    }


def add_candidate(
    candidates: list[dict[str, Any]],
    duplicates: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    seen: dict[str, int],
    cand: dict[str, Any],
    task: dict[str, Any],
    candidate_index: int,
) -> bool:
    raw_code = cand.get("sanitized_code") or cand.get("raw_tool_input") or cand.get("raw_model_text") or ""
    final_code = cand.get("final_code") or ""
    cand.update(hash_fields(raw_code, final_code))
    cand.update({
        "task_id": task["task_id"],
        "source": task["source"],
        "difficulty": task.get("difficulty", "unknown"),
        "function_name": task["function_name"],
        "candidate_index": candidate_index,
    })
    shape_error = code_shape_error(final_code)
    if shape_error:
        errors.append({
            "task_id": task["task_id"],
            "mode": cand.get("mode"),
            "candidate_stage": cand.get("candidate_stage"),
            "error": cand.get("generation_error") or shape_error,
            "raw_snippet": snippet(cand.get("raw_model_text", ""), 300),
            "final_code_snippet": snippet(final_code, 300),
        })
        return False
    key = cand.get("ast_hash") or cand.get("normalized_code_hash") or cand.get("raw_code_hash")
    if key and key in seen:
        dup = dict(cand)
        dup["duplicate_of"] = seen[key]
        duplicates.append(dup)
        return False
    if key:
        seen[key] = candidate_index
    cand["duplicate_of"] = None
    candidates.append(cand)
    return True


def load_generation_checkpoint(path: str, resume: bool) -> dict[str, Any]:
    if not resume:
        return {}
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return {}
    try:
        return json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_generation_checkpoint(
    path: str,
    *,
    args: argparse.Namespace,
    selected_tasks: list[dict[str, Any]],
    completed_task_ids: list[str],
    candidates: list[dict[str, Any]],
    duplicates: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> None:
    task_ids_with_candidates = {c["task_id"] for c in candidates}
    tasks_with_2plus = len([tid for tid in task_ids_with_candidates if sum(1 for c in candidates if c["task_id"] == tid) >= 2])
    total_attempts = len(candidates) + len(duplicates) + len(errors)
    payload = {
        "checkpoint_type": "code_branch_candidates_v2_generation_partial",
        "complete": False,
        "output": str(args.output),
        "completed_task_ids": completed_task_ids,
        "remaining_task_ids": [task["task_id"] for task in selected_tasks if task["task_id"] not in set(completed_task_ids)],
        "summary": {
            "tasks_seen": len(completed_task_ids),
            "total_generation_attempts_recorded": total_attempts,
            "unique_candidates": len(candidates),
            "duplicate_candidates": len(duplicates),
            "generation_errors": len(errors),
            "duplicate_rate": len(duplicates) / max(len(candidates) + len(duplicates), 1),
            "tasks_with_2plus_unique_candidates": tasks_with_2plus,
            "stage_breakdown": dict(Counter(c["candidate_stage"] for c in candidates)),
            "mode_breakdown": dict(Counter(c["mode"] for c in candidates)),
            "duplicate_by_mode": dict(Counter(c["mode"] for c in duplicates)),
        },
        "candidates": candidates,
        "duplicate_candidates": duplicates,
        "generation_errors": errors,
    }
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(checkpoint_path)


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Code Branch Candidates v2",
        "",
        f"CODE_V2_GENERATION_VERDICT = {payload['code_v2_generation_verdict']}",
        "",
        f"- tasks_seen: `{payload['summary']['tasks_seen']}`",
        f"- unique_candidates: `{payload['summary']['unique_candidates']}`",
        f"- duplicate_candidates: `{payload['summary']['duplicate_candidates']}`",
        f"- duplicate_rate: `{payload['summary']['duplicate_rate']}`",
        f"- tasks_with_2plus_unique_candidates: `{payload['summary']['tasks_with_2plus_unique_candidates']}`",
        f"- stage_breakdown: `{payload['summary']['stage_breakdown']}`",
        f"- mode_breakdown: `{payload['summary']['mode_breakdown']}`",
        "",
        "## Task Candidate Counts",
        "",
        "| task_id | source | difficulty | unique | duplicates | errors |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for task_id, row in payload["by_task"].items():
        lines.append(
            f"| `{task_id}` | `{row['source']}` | `{row['difficulty']}` | {row['unique_candidates']} | "
            f"{row['duplicate_candidates']} | {row['errors']} |"
        )
    lines.extend(["", "## Candidate Preview", ""])
    for cand in payload["candidates"][:20]:
        lines.append(
            f"- `{cand['task_id']}` `{cand['mode']}` `{cand['candidate_stage']}` "
            f"hash=`{cand.get('ast_hash') or cand.get('normalized_code_hash')}` code={snippet(cand['final_code'], 120)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    with setup_log_tee(args.log_file):
        _main(args)


def _main(args: argparse.Namespace) -> None:
    configure_local_agent_env()
    if args.device == "cuda" and not CUDA_AVAILABLE_AT_IMPORT:
        print("[code-v2-gen] warning: preflight torch.cuda.is_available() is false; continuing so the Ouro backend can perform its own CUDA check", flush=True)
    taskset = load_json(args.taskset)
    if taskset.get("code_v2_taskset_verdict") == "BLOCKED":
        raise SystemExit("taskset verdict BLOCKED")
    modules = import_agent_modules()
    agent = modules["agent"]
    model_mgr, _, classifier, project, self_model, history = agent.create_agent_runtime()
    selected_tasks = taskset["tasks"][: max(0, int(args.max_tasks))]
    if len(selected_tasks) < int(args.min_tasks):
        print(f"[code-v2-gen] warning: only {len(selected_tasks)} tasks selected, below min {args.min_tasks}", flush=True)
    checkpoint = load_generation_checkpoint(args.partial_output, bool(args.resume))
    completed_task_ids: list[str] = list(checkpoint.get("completed_task_ids", []))
    completed_set = set(completed_task_ids)
    candidates: list[dict[str, Any]] = list(checkpoint.get("candidates", []))
    duplicates: list[dict[str, Any]] = list(checkpoint.get("duplicate_candidates", []))
    errors: list[dict[str, Any]] = list(checkpoint.get("generation_errors", []))
    if checkpoint:
        print(
            f"[code-v2-gen] resuming from {args.partial_output}: "
            f"{len(completed_task_ids)} tasks, {len(candidates)} candidates",
            flush=True,
        )
    per_task_seen: dict[str, dict[str, int]] = {}
    modes = [
        ("direct_short_budget", lambda task: direct_code_candidate(model_mgr, modules, task, mode="direct_short_budget", stage="direct_short_budget", temperature=0.7, max_tokens=args.direct_short_tokens)),
        ("first_tool_code", lambda task: action_code_candidate(model_mgr, modules, task, mode="first_tool_code", stage="first_tool_code", temperature=0.2, max_tokens=args.first_action_max_tokens)),
        ("first_failed_or_first_repair_code", lambda task: repair_or_failed_candidate(model_mgr, modules, task, first_tokens=args.first_action_max_tokens, repair_tokens=args.repair_max_tokens)),
        ("direct_sampled_high", lambda task: direct_code_candidate(model_mgr, modules, task, mode="direct_sampled_high", stage="direct_final", temperature=0.9, max_tokens=args.direct_max_tokens)),
        ("direct_sampled_medium", lambda task: direct_code_candidate(model_mgr, modules, task, mode="direct_sampled_medium", stage="direct_final", temperature=0.5, max_tokens=args.direct_max_tokens)),
        ("repaired_final", lambda task: hard_code_final_candidate(model_mgr, modules, project, history, classifier, self_model, task)),
    ]
    if not args.prefer_prefinal:
        modes = [
            ("direct_sampled_medium", lambda task: direct_code_candidate(model_mgr, modules, task, mode="direct_sampled_medium", stage="direct_final", temperature=0.5, max_tokens=args.direct_max_tokens)),
            ("direct_sampled_high", lambda task: direct_code_candidate(model_mgr, modules, task, mode="direct_sampled_high", stage="direct_final", temperature=0.9, max_tokens=args.direct_max_tokens)),
            ("direct_short_budget", lambda task: direct_code_candidate(model_mgr, modules, task, mode="direct_short_budget", stage="direct_short_budget", temperature=0.7, max_tokens=args.direct_short_tokens)),
            ("first_tool_code", lambda task: action_code_candidate(model_mgr, modules, task, mode="first_tool_code", stage="first_tool_code", temperature=0.2, max_tokens=args.first_action_max_tokens)),
            ("first_failed_or_first_repair_code", lambda task: repair_or_failed_candidate(model_mgr, modules, task, first_tokens=args.first_action_max_tokens, repair_tokens=args.repair_max_tokens)),
            ("repaired_final", lambda task: hard_code_final_candidate(model_mgr, modules, project, history, classifier, self_model, task)),
        ]
    try:
        for task in selected_tasks:
            if task["task_id"] in completed_set:
                continue
            print(f"[code-v2-gen] task {task['task_id']} difficulty={task.get('difficulty')}", flush=True)
            seen: dict[str, int] = {}
            for idx, existing in enumerate(candidates):
                if existing.get("task_id") != task["task_id"]:
                    continue
                key = existing.get("ast_hash") or existing.get("normalized_code_hash") or existing.get("raw_code_hash")
                if key:
                    seen[key] = idx
            per_task_seen[task["task_id"]] = seen
            kept_for_task = sum(1 for c in candidates if c.get("task_id") == task["task_id"])
            for mode_name, fn in modes:
                if kept_for_task >= int(args.max_candidates_per_task):
                    break
                if len(candidates) >= int(args.hard_cap_total_candidates):
                    break
                repaired_count = sum(
                    1
                    for c in candidates
                    if c.get("task_id") == task["task_id"] and c.get("candidate_stage") == "repaired_final"
                )
                if mode_name == "repaired_final" and repaired_count >= int(args.limit_repaired_final_per_task):
                    break
                try:
                    cand = fn(task)
                    added = add_candidate(candidates, duplicates, errors, seen, cand, task, kept_for_task)
                    if added:
                        kept_for_task += 1
                except Exception as exc:
                    errors.append({"task_id": task["task_id"], "mode": mode_name, "error": f"{type(exc).__name__}: {exc}"})
                    print(f"[code-v2-gen] error {task['task_id']} {mode_name}: {type(exc).__name__}: {exc}", flush=True)
                if len(candidates) >= int(args.hard_cap_total_candidates):
                    break
            completed_task_ids.append(task["task_id"])
            completed_set.add(task["task_id"])
            write_generation_checkpoint(
                args.partial_output,
                args=args,
                selected_tasks=selected_tasks,
                completed_task_ids=completed_task_ids,
                candidates=candidates,
                duplicates=duplicates,
                errors=errors,
            )
            if len(candidates) >= int(args.hard_cap_total_candidates):
                break
    finally:
        if hasattr(model_mgr, "unload"):
            with contextlib.suppress(Exception):
                model_mgr.unload()

    task_ids_with_candidates = {c["task_id"] for c in candidates}
    tasks_with_2plus = len([tid for tid in task_ids_with_candidates if sum(1 for c in candidates if c["task_id"] == tid) >= 2])
    total_attempts = len(candidates) + len(duplicates) + len(errors)
    duplicate_rate = len(duplicates) / max(len(candidates) + len(duplicates), 1)
    target_ready = max(2, min(int(args.target_usable_tournaments), int(args.min_tasks)))
    if not candidates:
        verdict = "WRAPPER_BLOCKED"
    elif tasks_with_2plus >= target_ready:
        verdict = "READY"
    elif tasks_with_2plus >= max(2, int(args.min_tasks)) and duplicate_rate > 0.35:
        verdict = "TOO_DUPLICATE"
    elif tasks_with_2plus >= max(2, int(args.min_tasks)):
        verdict = "TOO_FEW_CANDIDATES"
    elif len(errors) > len(candidates) + len(duplicates):
        verdict = "WRAPPER_BLOCKED"
    else:
        verdict = "TOO_DUPLICATE" if duplicate_rate > 0.35 else "TOO_FEW_CANDIDATES"

    by_task: dict[str, dict[str, Any]] = {}
    for task in selected_tasks:
        tid = task["task_id"]
        by_task[tid] = {
            "source": task["source"],
            "difficulty": task.get("difficulty", "unknown"),
            "unique_candidates": sum(1 for c in candidates if c["task_id"] == tid),
            "duplicate_candidates": sum(1 for c in duplicates if c["task_id"] == tid),
            "errors": sum(1 for e in errors if e["task_id"] == tid),
        }
    summary = {
        "tasks_seen": len(selected_tasks),
        "total_generation_attempts_recorded": total_attempts,
        "unique_candidates": len(candidates),
        "duplicate_candidates": len(duplicates),
        "generation_errors": len(errors),
        "duplicate_rate": duplicate_rate,
        "tasks_with_2plus_unique_candidates": tasks_with_2plus,
        "stage_breakdown": dict(Counter(c["candidate_stage"] for c in candidates)),
        "mode_breakdown": dict(Counter(c["mode"] for c in candidates)),
        "duplicate_by_mode": dict(Counter(c["mode"] for c in duplicates)),
    }
    payload = {
        "code_v2_generation_verdict": verdict,
        "taskset": repo_path(Path(args.taskset)),
        "summary": summary,
        "by_task": by_task,
        "candidates": candidates,
        "duplicate_candidates": duplicates,
        "generation_errors": errors,
        "partial_output": repo_path(Path(args.partial_output)),
        "completed_task_ids": completed_task_ids,
        "notes": "Candidates are unit-test labeled only in evaluate_code_branch_candidates_v2.py; generation uses public tests only to produce one repair-stage branch.",
    }
    out_json = Path(args.output)
    out_md = Path(args.output_md)
    write_json(out_json, payload)
    write_generation_checkpoint(
        args.partial_output,
        args=args,
        selected_tasks=selected_tasks,
        completed_task_ids=completed_task_ids,
        candidates=candidates,
        duplicates=duplicates,
        errors=errors,
    )
    partial_payload = json.loads(Path(args.partial_output).read_text(encoding="utf-8"))
    partial_payload["complete"] = True
    Path(args.partial_output).write_text(json.dumps(partial_payload, indent=2, default=str) + "\n", encoding="utf-8")
    write_md(out_md, payload)
    print(f"CODE_V2_GENERATION_VERDICT = {verdict}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
