"""Generate minimal candidates for strict-clean code task screening."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import torch

from code_branch_pilot_lib import REPORT_DIR, configure_local_agent_env, import_agent_modules, repo_path, snippet, write_json
from evaluate_code_branch_candidates_v2 import eval_candidate, label_from_eval, legacy_label_from_eval
from generate_code_branch_candidates_v2 import (
    action_code_candidate,
    code_shape_error,
    direct_code_candidate,
    hash_fields,
)


TASKPOOL_JSON = REPORT_DIR / "code_strict_clean_screening_taskpool_2026-05-17.json"
OUTPUT_JSON = REPORT_DIR / "code_strict_clean_screening_candidates_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "code_strict_clean_screening_candidates_2026-05-17.md"
OUTPUT_LOG = REPORT_DIR / "code_strict_clean_screening_candidates_2026-05-17.log"

CUDA_AVAILABLE_AT_IMPORT = torch.cuda.is_available()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taskpool", default=str(TASKPOOL_JSON))
    parser.add_argument("--output", default=str(OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    parser.add_argument("--log-file", default=str(OUTPUT_LOG))
    parser.add_argument("--max-tasks", type=int, default=60)
    parser.add_argument("--min-tasks", type=int, default=30)
    parser.add_argument("--target-strict-clean-ready", type=int, default=10)
    parser.add_argument("--candidates-per-task", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--direct-max-tokens", type=int, default=512)
    parser.add_argument("--direct-short-tokens", type=int, default=192)
    parser.add_argument("--first-action-max-tokens", type=int, default=640)
    return parser.parse_args()


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


@contextlib.contextmanager
def tee_log(path: str):
    if not path or os.environ.get("CODE_STRICT_SCREEN_LOG_TEE_ACTIVE") == "1":
        yield
        return
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    os.environ["CODE_STRICT_SCREEN_LOG_TEE_ACTIVE"] = "1"
    sys.stdout = Tee(old_stdout, log)  # type: ignore[assignment]
    sys.stderr = Tee(old_stderr, log)  # type: ignore[assignment]
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log.close()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tests_for(task: dict[str, Any]) -> list[str]:
    tests = list(task.get("tests") or [])
    if not tests:
        tests = list(task.get("public_tests", [])) + list(task.get("hidden_tests", []))
    return [str(test).strip() for test in tests if str(test).strip()]


def candidate_key(candidate: dict[str, Any]) -> str:
    return str(candidate.get("ast_hash") or candidate.get("normalized_code_hash") or candidate.get("raw_code_hash") or "")


def make_role_fn(
    role: str,
    model_mgr: Any,
    modules: dict[str, Any],
    args: argparse.Namespace,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    if role == "strong_anchor":
        return lambda task: direct_code_candidate(
            model_mgr,
            modules,
            task,
            mode="direct_deterministic_final",
            stage="direct_final",
            temperature=0.05,
            max_tokens=args.direct_max_tokens,
        )
    if role == "near_miss_probe_1":
        return lambda task: action_code_candidate(
            model_mgr,
            modules,
            task,
            mode="first_tool_code",
            stage="first_tool_code",
            temperature=0.7,
            max_tokens=args.first_action_max_tokens,
            extra_directive=(
                "Prefer a compact first attempt. Do not spend tokens on explanations. "
                "Do not over-repair edge cases before returning the Python action."
            ),
        )
    if role == "near_miss_probe_2":
        return lambda task: direct_code_candidate(
            model_mgr,
            modules,
            task,
            mode="direct_sampled_high",
            stage="direct_final",
            temperature=1.05,
            max_tokens=args.direct_max_tokens,
        )
    raise KeyError(role)


def provisional_eval(candidate: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    code = candidate.get("final_code", "") or ""
    result = eval_candidate(
        code,
        tests_for(task),
        str(task.get("function_name", "")),
        float(task.get("timeout_seconds", 5.0)),
    )
    label = label_from_eval(result, code, str(task.get("function_name", "")))
    return {
        **result,
        "unit_test_label": label,
        "legacy_unit_test_label": legacy_label_from_eval(result),
        "is_correct": label == "correct",
        "is_runnable": bool(result.get("safety_ok") and result.get("syntax_ok") and result.get("import_ok") and result.get("runtime_ok")),
    }


def strict_clean_ready_count(rows: list[dict[str, Any]]) -> int:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("duplicate_of"):
            continue
        by_task[str(row.get("task_id", ""))].append(row)
    total = 0
    for task_rows in by_task.values():
        if any(row.get("unit_test_label") == "correct" for row in task_rows) and any(
            row.get("unit_test_label") == "near_miss" for row in task_rows
        ):
            total += 1
    return total


def classify_task(rows: list[dict[str, Any]]) -> str:
    primary = [row for row in rows if not row.get("duplicate_of")]
    labels = Counter(str(row.get("unit_test_label", "unknown")) for row in primary)
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


def admit_candidate(
    *,
    candidate: dict[str, Any],
    task: dict[str, Any],
    role: str,
    seen: dict[str, str],
    candidate_index: int,
) -> dict[str, Any]:
    raw_code = candidate.get("sanitized_code") or candidate.get("raw_tool_input") or candidate.get("raw_model_text") or ""
    final_code = candidate.get("final_code") or ""
    candidate.update(hash_fields(raw_code, final_code))
    key = candidate_key(candidate)
    uid_tail = key or str(int(time.time() * 1000))
    uid = f"screening::{task['task_id']}::{role}::{candidate_index}::{uid_tail}"
    shape_error = code_shape_error(final_code)
    duplicate_of = seen.get(key) if key else None
    if key and not duplicate_of and not shape_error:
        seen[key] = uid
    candidate.update({
        "task_id": task["task_id"],
        "source": task.get("source", "unknown"),
        "difficulty": task.get("difficulty", "unknown"),
        "function_name": task.get("function_name", ""),
        "candidate_uid": uid,
        "candidate_index": candidate_index,
        "screening_role": role,
        "generation_source": "strict_clean_screening",
        "duplicate_of": duplicate_of,
        "admitted_primary": not bool(duplicate_of),
        "admission_error": shape_error,
    })
    if shape_error and not candidate.get("generation_error"):
        candidate["generation_error"] = shape_error
    return candidate


def write_partial(
    path: Path,
    *,
    args: argparse.Namespace,
    taskpool_path: Path,
    selected_tasks: list[dict[str, Any]],
    completed_task_ids: list[str],
    candidates: list[dict[str, Any]],
    generation_errors: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    complete: bool,
    verdict: str = "IN_PROGRESS",
    stop_reason: str = "",
) -> None:
    by_task: dict[str, dict[str, Any]] = {}
    for task in selected_tasks:
        task_id = str(task["task_id"])
        task_rows = [row for row in eval_rows if row.get("task_id") == task_id and not row.get("duplicate_of")]
        by_task[task_id] = {
            "source": task.get("source", "unknown"),
            "difficulty": task.get("difficulty", "unknown"),
            "prior_outcome": task.get("prior_outcome", "unknown"),
            "completed": task_id in set(completed_task_ids),
            "candidates": sum(1 for cand in candidates if cand.get("task_id") == task_id),
            "primary_unique_candidates": len(task_rows),
            "label_counts": dict(Counter(row.get("unit_test_label", "unknown") for row in task_rows)),
            "classification": classify_task(task_rows) if task_rows else "malformed_only",
        }
    summary = {
        "tasks_selected": len(selected_tasks),
        "tasks_completed": len(completed_task_ids),
        "total_screening_candidates": len(candidates),
        "primary_unique_candidates": sum(1 for cand in candidates if not cand.get("duplicate_of")),
        "duplicate_candidates": sum(1 for cand in candidates if cand.get("duplicate_of")),
        "generation_errors": len(generation_errors),
        "strict_clean_ready_provisional": strict_clean_ready_count(eval_rows),
        "label_counts_provisional": dict(Counter(row.get("unit_test_label", "unknown") for row in eval_rows if not row.get("duplicate_of"))),
        "role_breakdown": dict(Counter(cand.get("screening_role", "unknown") for cand in candidates)),
    }
    payload = {
        "screening_generation_verdict": verdict,
        "complete": complete,
        "stop_reason": stop_reason,
        "taskpool": repo_path(taskpool_path),
        "summary": summary,
        "by_task": by_task,
        "candidates": candidates,
        "generation_errors": generation_errors,
        "provisional_candidate_evaluations": eval_rows,
        "completed_task_ids": completed_task_ids,
        "outputs": {"json": repo_path(path), "md": repo_path(Path(args.output_md)), "log": repo_path(OUTPUT_LOG)},
    }
    write_json(path, payload)


def write_md(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# Strict-Clean Code Screening Candidates",
        "",
        f"SCREENING_GENERATION_VERDICT = {payload['screening_generation_verdict']}",
        "",
        f"- stop_reason: `{payload.get('stop_reason', '')}`",
        f"- tasks_completed: `{summary['tasks_completed']}`",
        f"- total_screening_candidates: `{summary['total_screening_candidates']}`",
        f"- primary_unique_candidates: `{summary['primary_unique_candidates']}`",
        f"- duplicate_candidates: `{summary['duplicate_candidates']}`",
        f"- generation_errors: `{summary['generation_errors']}`",
        f"- strict_clean_ready_provisional: `{summary['strict_clean_ready_provisional']}`",
        f"- label_counts_provisional: `{summary['label_counts_provisional']}`",
        f"- role_breakdown: `{summary['role_breakdown']}`",
        "",
        "## Task Status",
        "",
        "| task_id | source | difficulty | prior | candidates | primary | labels | classification |",
        "| --- | --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for task_id, row in payload["by_task"].items():
        if not row["completed"]:
            continue
        lines.append(
            f"| `{task_id}` | `{row['source']}` | `{row['difficulty']}` | `{row['prior_outcome']}` | "
            f"{row['candidates']} | {row['primary_unique_candidates']} | `{row['label_counts']}` | `{row['classification']}` |"
        )
    lines.extend(["", "## Candidate Preview", ""])
    for cand in payload["candidates"][:30]:
        lines.append(
            f"- `{cand['task_id']}` `{cand.get('screening_role')}` `{cand.get('mode')}` "
            f"dup=`{cand.get('duplicate_of') or ''}` code={snippet(cand.get('final_code', ''), 120)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _main(args: argparse.Namespace) -> None:
    configure_local_agent_env()
    if args.device == "cuda" and not CUDA_AVAILABLE_AT_IMPORT:
        print("[screen-gen] warning: torch.cuda.is_available() false at import; continuing for backend CUDA check", flush=True)
    taskpool_path = Path(args.taskpool)
    taskpool = load_json(taskpool_path)
    if taskpool.get("screening_taskpool_verdict") == "BLOCKED":
        raise SystemExit("SCREENING_TASKPOOL_VERDICT=BLOCKED")
    selected_tasks = list(taskpool.get("tasks", []))[: max(0, int(args.max_tasks))]
    if len(selected_tasks) < int(args.min_tasks):
        raise SystemExit(f"task pool has {len(selected_tasks)} tasks, below min {args.min_tasks}")

    modules = import_agent_modules()
    agent = modules["agent"]
    model_mgr, _, _, _, _, _ = agent.create_agent_runtime()
    candidates: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    generation_errors: list[dict[str, Any]] = []
    completed_task_ids: list[str] = []
    seen_by_task: dict[str, dict[str, str]] = defaultdict(dict)
    roles = ["strong_anchor", "near_miss_probe_1", "near_miss_probe_2"][: max(1, min(int(args.candidates_per_task), 3))]
    stop_reason = ""
    verdict = "COMPLETED"

    try:
        for task_index, task in enumerate(selected_tasks, start=1):
            task_id = str(task["task_id"])
            print(f"[screen-gen] task {task_index}/{len(selected_tasks)} {task_id}", flush=True)
            for role in roles:
                if len(candidates) >= 180:
                    stop_reason = "hard_cap_total_candidates"
                    break
                current_rows = [row for row in eval_rows if row.get("task_id") == task_id and not row.get("duplicate_of")]
                if any(row.get("unit_test_label") == "correct" for row in current_rows) and any(
                    row.get("unit_test_label") == "near_miss" for row in current_rows
                ):
                    break
                try:
                    fn = make_role_fn(role, model_mgr, modules, args)
                    raw_candidate = fn(task)
                    candidate = admit_candidate(
                        candidate=raw_candidate,
                        task=task,
                        role=role,
                        seen=seen_by_task[task_id],
                        candidate_index=sum(1 for cand in candidates if cand.get("task_id") == task_id),
                    )
                    eval_row = provisional_eval(candidate, task)
                    eval_row.update({
                        "task_id": task_id,
                        "source": task.get("source", "unknown"),
                        "difficulty": task.get("difficulty", "unknown"),
                        "function_name": task.get("function_name", ""),
                        "candidate_uid": candidate["candidate_uid"],
                        "candidate_index": candidate["candidate_index"],
                        "screening_role": role,
                        "mode": candidate.get("mode", ""),
                        "candidate_stage": candidate.get("candidate_stage", ""),
                        "duplicate_of": candidate.get("duplicate_of"),
                    })
                    candidate["provisional_unit_test_label"] = eval_row["unit_test_label"]
                    candidate["provisional_tests_passed"] = eval_row.get("tests_passed")
                    candidate["provisional_tests_total"] = eval_row.get("tests_total")
                    candidates.append(candidate)
                    eval_rows.append(eval_row)
                except Exception as exc:
                    generation_errors.append({
                        "task_id": task_id,
                        "screening_role": role,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    print(f"[screen-gen] error {task_id} {role}: {type(exc).__name__}: {exc}", flush=True)
            completed_task_ids.append(task_id)
            strict_ready = strict_clean_ready_count(eval_rows)
            write_partial(
                Path(args.output),
                args=args,
                taskpool_path=taskpool_path,
                selected_tasks=selected_tasks,
                completed_task_ids=completed_task_ids,
                candidates=candidates,
                generation_errors=generation_errors,
                eval_rows=eval_rows,
                complete=False,
                verdict="IN_PROGRESS",
                stop_reason=stop_reason,
            )
            if strict_ready >= int(args.target_strict_clean_ready):
                verdict = "EARLY_STOP_READY"
                stop_reason = "target_strict_clean_ready_reached"
                break
            if len(completed_task_ids) >= 30 and strict_ready == 0:
                verdict = "COMPLETED"
                stop_reason = "zero_strict_clean_ready_after_30_tasks"
                break
            if task_index % max(1, int(args.chunk_size)) == 0:
                print(
                    f"[screen-gen] chunk complete tasks={len(completed_task_ids)} strict_clean_ready={strict_ready}",
                    flush=True,
                )
            if stop_reason == "hard_cap_total_candidates":
                verdict = "COMPLETED"
                break
    finally:
        if hasattr(model_mgr, "unload"):
            with contextlib.suppress(Exception):
                model_mgr.unload()

    if not candidates and generation_errors:
        verdict = "WRAPPER_BLOCKED"
    elif len(completed_task_ids) < int(args.min_tasks) and verdict != "EARLY_STOP_READY":
        verdict = "TOO_FEW_CANDIDATES"
    elif verdict != "EARLY_STOP_READY":
        verdict = "COMPLETED"
    if not stop_reason:
        stop_reason = "completed_task_cap_or_pool"

    write_partial(
        Path(args.output),
        args=args,
        taskpool_path=taskpool_path,
        selected_tasks=selected_tasks,
        completed_task_ids=completed_task_ids,
        candidates=candidates,
        generation_errors=generation_errors,
        eval_rows=eval_rows,
        complete=True,
        verdict=verdict,
        stop_reason=stop_reason,
    )
    payload = load_json(args.output)
    write_md(Path(args.output_md), payload)
    print(f"SCREENING_GENERATION_VERDICT = {verdict}")
    print(f"strict_clean_ready_provisional = {payload['summary']['strict_clean_ready_provisional']}")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output_md}")
    if verdict == "WRAPPER_BLOCKED":
        raise SystemExit(1)


def main() -> None:
    args = parse_args()
    with tee_log(args.log_file):
        _main(args)


if __name__ == "__main__":
    main()
