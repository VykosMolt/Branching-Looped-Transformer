"""Generate code branch candidates with the local-agent/Ouro-RLTT wrapper."""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
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
        code_signature,
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
        code_signature,
        import_agent_modules,
        load_json,
        repo_path,
        sanitize_python_candidate,
        snippet,
        write_json,
    )


DEFAULT_TASKSET = REPORT_DIR / "code_branch_taskset_2026-05-16.json"
DEFAULT_OUTPUT = REPORT_DIR / "code_branch_candidates_2026-05-16.json"
DEFAULT_MD = REPORT_DIR / "code_branch_candidates_2026-05-16.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taskset", default=str(DEFAULT_TASKSET))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--output-md", default=str(DEFAULT_MD))
    parser.add_argument("--max-tasks", type=int, default=10)
    parser.add_argument("--max-candidates-per-task", type=int, default=4)
    parser.add_argument("--max-total-candidates", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--direct-max-tokens", type=int, default=512)
    parser.add_argument("--first-action-max-tokens", type=int, default=768)
    return parser.parse_args()


def task_prompt(task: dict[str, Any]) -> str:
    return (
        f"{task['prompt']}\n\n"
        f"Return a complete Python implementation of `{task['function_name']}` in one code block. "
        "Do not use file I/O. Do not read stdin. Do not include unit-test results."
    )


def merge_prefill(raw: str, prefill: str) -> str:
    raw = raw or ""
    if raw.startswith(prefill):
        return raw
    return prefill + raw


def direct_code_candidate(
    model_mgr: Any,
    modules: dict[str, Any],
    task: dict[str, Any],
    *,
    mode: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    config = modules["config"]
    prompts = modules["prompts"]
    policies = modules["policies"]
    agent = modules["agent"]
    system = (
        "You are a strong reasoning, math, and coding assistant.\n\n"
        + prompts.OURO_SOLVER_POSTURE
        + "\nYou are running inside a local RLTT wrapper."
        + "\nFor coding tasks, answer code-first. Avoid long approach narration before the code."
        + "\nStart with `FINAL ANSWER:` and include complete code plus requested sample results."
    )
    user = (
        "Start immediately with `FINAL ANSWER:` and complete code in a fenced code block.\n"
        "After the code, include no extra prose unless the task explicitly requested it.\n\n"
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
    solution_code = agent.final_python_code_for_answer(task_prompt(task), code) if code and hasattr(agent, "final_python_code_for_answer") else code
    return {
        "mode": mode,
        "candidate_stage": "direct_final",
        "temperature": float(temperature),
        "route": "local_agent_direct_prompt",
        "raw_model_text": raw_model_text,
        "raw_tool_input": "",
        "sanitized_code": code,
        "final_code": solution_code,
        "wallclock_seconds": round(time.perf_counter() - started, 3),
        "token_budget": int(max_tokens),
        "parser_sanitizer_path": "ouro_policies.sanitize_tool_input/extract_python_code/final_python_code_for_answer",
        "generation_error": "" if code else "no_code_extracted",
    }


def first_tool_candidate(
    model_mgr: Any,
    modules: dict[str, Any],
    task: dict[str, Any],
    *,
    max_tokens: int,
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
        "The Python source should define the requested function and include a few meaningful asserts. "
        "Do not write FINAL ANSWER."
    )
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
        temperature=0.2,
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
    solution_code = agent.final_python_code_for_answer(task_prompt(task), code) if code and hasattr(agent, "final_python_code_for_answer") else code
    return {
        "mode": "hard_code_first_action",
        "candidate_stage": "first_tool_code",
        "temperature": 0.2,
        "route": "local_agent_action_prefill",
        "raw_model_text": raw_model_text,
        "raw_tool_input": raw_tool_input,
        "sanitized_code": code,
        "final_code": solution_code,
        "wallclock_seconds": round(time.perf_counter() - started, 3),
        "token_budget": int(max_tokens),
        "parser_sanitizer_path": "parse_action + ouro_policies.sanitize_tool_input + final_python_code_for_answer",
        "generation_error": "" if code else f"no_python_action_extracted:{action}",
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
    solution_code = agent.final_python_code_for_answer(task_prompt(task), code) if code and hasattr(agent, "final_python_code_for_answer") else code
    return {
        "mode": "hard_code_final",
        "candidate_stage": "repaired_final",
        "temperature": 0.12,
        "route": "ouro_agent_improved.run_task_mode(agent)",
        "raw_model_text": answer,
        "raw_tool_input": "",
        "sanitized_code": code,
        "final_code": solution_code,
        "wallclock_seconds": round(time.perf_counter() - started, 3),
        "token_budget": 768,
        "parser_sanitizer_path": "run_task_mode final + sanitizer + final_python_code_for_answer",
        "generation_error": "" if code else "no_code_extracted",
    }


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Code Branch Candidates",
        "",
        f"CODE_GENERATION_VERDICT = {payload['code_generation_verdict']}",
        "",
        f"- tasks_seen: `{payload['summary']['tasks_seen']}`",
        f"- candidates_generated: `{payload['summary']['candidates_generated']}`",
        f"- tasks_with_2plus_candidates: `{payload['summary']['tasks_with_2plus_candidates']}`",
        f"- stage_breakdown: `{payload['summary']['stage_breakdown']}`",
        f"- mode_breakdown: `{payload['summary']['mode_breakdown']}`",
        "",
        "## Task Candidate Counts",
        "",
        "| task_id | source | candidates | errors |",
        "| --- | --- | ---: | ---: |",
    ]
    by_task = payload["by_task"]
    for task_id, row in by_task.items():
        lines.append(
            f"| `{task_id}` | `{row['source']}` | {row['candidates']} | {row['errors']} |"
        )
    lines.extend(["", "## Candidate Preview", ""])
    for cand in payload["candidates"][:12]:
        lines.append(
            f"- `{cand['task_id']}` {cand['mode']} {cand['candidate_stage']} "
            f"error=`{cand['generation_error']}` code={snippet(cand['final_code'], 120)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_local_agent_env()
    if args.device == "cuda" and not CUDA_AVAILABLE_AT_IMPORT:
        print("[code-gen] warning: preflight torch.cuda.is_available() is false; continuing so the Ouro backend can perform its own CUDA check", flush=True)
    taskset = load_json(args.taskset)
    if taskset.get("code_taskset_verdict") == "BLOCKED":
        raise SystemExit("taskset verdict BLOCKED")
    modules = import_agent_modules()
    agent = modules["agent"]
    model_mgr, _, classifier, project, self_model, history = agent.create_agent_runtime()
    candidates: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    selected_tasks = taskset["tasks"][: max(0, int(args.max_tasks))]
    modes = [
        ("direct_deterministic", lambda task: direct_code_candidate(
            model_mgr, modules, task, mode="direct_deterministic", temperature=0.0, max_tokens=args.direct_max_tokens)),
        ("direct_sampled_low", lambda task: direct_code_candidate(
            model_mgr, modules, task, mode="direct_sampled_low", temperature=0.3, max_tokens=args.direct_max_tokens)),
        ("direct_sampled_high", lambda task: direct_code_candidate(
            model_mgr, modules, task, mode="direct_sampled_high", temperature=0.7, max_tokens=args.direct_max_tokens)),
        ("hard_code_first_action", lambda task: first_tool_candidate(
            model_mgr, modules, task, max_tokens=args.first_action_max_tokens)),
        ("hard_code_final", lambda task: hard_code_final_candidate(
            model_mgr, modules, project, history, classifier, self_model, task)),
    ]
    try:
        for task in selected_tasks:
            per_task = 0
            print(f"[code-gen] task {task['task_id']}", flush=True)
            for mode_name, fn in modes:
                if per_task >= int(args.max_candidates_per_task):
                    break
                if len(candidates) >= int(args.max_total_candidates):
                    break
                try:
                    cand = fn(task)
                    cand.update({
                        "task_id": task["task_id"],
                        "source": task["source"],
                        "function_name": task["function_name"],
                        "candidate_index": per_task,
                        "code_signature": code_signature(cand.get("final_code", ""))[:16] if cand.get("final_code") else "",
                    })
                    if cand.get("final_code"):
                        candidates.append(cand)
                        per_task += 1
                    else:
                        errors.append({"task_id": task["task_id"], "mode": mode_name, "error": cand.get("generation_error", "")})
                except Exception as exc:
                    errors.append({"task_id": task["task_id"], "mode": mode_name, "error": f"{type(exc).__name__}: {exc}"})
                    print(f"[code-gen] error {task['task_id']} {mode_name}: {type(exc).__name__}: {exc}", flush=True)
                if len(candidates) >= int(args.max_total_candidates):
                    break
    finally:
        if hasattr(model_mgr, "unload"):
            with contextlib.suppress(Exception):
                model_mgr.unload()

    tasks_with_2plus = len({tid for tid in {c["task_id"] for c in candidates} if sum(1 for c in candidates if c["task_id"] == tid) >= 2})
    if not candidates:
        verdict = "WRAPPER_BLOCKED"
    elif len(candidates) < len(selected_tasks) * 2:
        verdict = "TOO_FEW_CANDIDATES"
    else:
        verdict = "READY"
    if verdict == "TOO_FEW_CANDIDATES" and tasks_with_2plus < 3:
        verdict = "WRAPPER_BLOCKED"

    by_task: dict[str, dict[str, Any]] = {}
    for task in selected_tasks:
        tid = task["task_id"]
        by_task[tid] = {
            "source": task["source"],
            "candidates": sum(1 for c in candidates if c["task_id"] == tid),
            "errors": sum(1 for e in errors if e["task_id"] == tid),
        }
    summary = {
        "tasks_seen": len(selected_tasks),
        "candidates_generated": len(candidates),
        "tasks_with_2plus_candidates": tasks_with_2plus,
        "stage_breakdown": dict(Counter(c["candidate_stage"] for c in candidates)),
        "mode_breakdown": dict(Counter(c["mode"] for c in candidates)),
    }
    payload = {
        "code_generation_verdict": verdict,
        "taskset": repo_path(Path(args.taskset)),
        "summary": summary,
        "by_task": by_task,
        "candidates": candidates,
        "generation_errors": errors,
        "notes": "Candidate labels are not computed here; unit tests are applied only in the evaluation script.",
    }
    out_json = Path(args.output)
    out_md = Path(args.output_md)
    write_json(out_json, payload)
    write_md(out_md, payload)
    print(f"CODE_GENERATION_VERDICT = {verdict}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
