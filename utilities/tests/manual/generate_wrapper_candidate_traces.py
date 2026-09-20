"""Generate local-agent wrapper candidate traces with opt-in export enabled."""

from __future__ import annotations

import os
import time
from collections import Counter
from typing import Any

from wrapper_bg_matched_lib import (
    OUT_DIR,
    PROJECT_ROOT,
    TRACE_DIR,
    candidate_code,
    load_task_suite,
    repo_path,
    stage_counts,
    task_slug,
    trace_candidates,
    write_json,
    write_text,
)


OUT_JSON = OUT_DIR / "candidate_traces.json"
OUT_MD = OUT_DIR / "candidate_traces.md"
PARTIAL_JSON = OUT_DIR / "candidate_traces.partial.json"
MAX_TOTAL_SECONDS = 90 * 60
NON_DEVIL_CAP = 180.0
DEVIL_CAP = 300.0


def code_like_count(trace: dict[str, Any]) -> int:
    count = 0
    for artifact in trace_candidates(trace):
        code, _ = candidate_code(artifact)
        if code:
            count += 1
    return count


def write_checkpoint(rows: list[dict[str, Any]], errors: list[dict[str, Any]], completed: list[str]) -> None:
    write_json(
        PARTIAL_JSON,
        {
            "complete": False,
            "candidate_traces": rows,
            "errors": errors,
            "completed_task_ids": completed,
        },
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("LOCAL_AGENT_OURO_MODEL_ID", str((PROJECT_ROOT / "shared/models/ouro_rltt_local").resolve()))
    os.environ.setdefault("LOCAL_AGENT_ORACLE_LOOKUPS_ENABLED", "0")
    os.environ.setdefault("LOCAL_AGENT_CODE_REFERENCE_FASTPATH", "0")
    os.environ.setdefault("LOCAL_AGENT_FAST_ASSISTED_SOLVER_ENABLED", "0")
    os.environ.setdefault("LOCAL_AGENT_SEARCH_PROVIDER", "disabled")
    os.environ.setdefault("LOCAL_AGENT_EXTERNAL_EXPERTS_ENABLED", "0")
    os.environ.setdefault("LOCAL_AGENT_EXTERNAL_EXPERT_AUTO_ENABLED", "0")
    os.environ.setdefault("LOCAL_AGENT_HARD_CODE_TASK_WALLCLOCK_SEC", str(int(NON_DEVIL_CAP)))
    os.environ.setdefault("LOCAL_AGENT_AGENT_TASK_WALLCLOCK_SEC", str(int(NON_DEVIL_CAP)))

    from utilities.tests.manual.code_branch_pilot_lib import configure_local_agent_env, import_agent_modules
    from candidate_capture import run_agent_with_candidates

    configure_local_agent_env()
    modules = import_agent_modules()
    agent = modules["agent"]
    model_mgr, _, classifier, project, self_model, history = agent.create_agent_runtime()
    tasks = load_task_suite()
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    completed: list[str] = []
    started_all = time.perf_counter()

    try:
        for task in tasks:
            if time.perf_counter() - started_all > MAX_TOTAL_SECONDS:
                errors.append({"task_id": task["task_id"], "error": "total_generation_wall_time_cap_exceeded"})
                break
            started = time.perf_counter()
            task_id = task["task_id"]
            print(f"[wrapper-trace] task {task_id} difficulty={task.get('difficulty')}", flush=True)
            try:
                profile = agent.build_profile_for_mode(
                    task["prompt"],
                    "agent",
                    classifier,
                    ut_steps=2,
                    max_tokens=768 if not task.get("is_devil") else 1024,
                    react_steps=3 if not task.get("is_devil") else 4,
                )
                profile = agent.strengthen_hard_code_tool_profile(task["prompt"], profile)
                trace = run_agent_with_candidates(
                    task["prompt"],
                    task_id=task_id,
                    model_mgr=model_mgr,
                    project=project,
                    history=list(history),
                    task_profile=profile,
                    self_model=self_model,
                    external_evaluator=None,
                )
                trace_data = {
                    "trace_id": trace.trace_id,
                    "task_id": trace.task_id,
                    "prompt_hash": trace.prompt_hash,
                    "route": trace.route,
                    "selected_candidate_uid": trace.selected_candidate_uid,
                    "final_answer": trace.final_answer,
                    "metadata": trace.metadata,
                    "candidates": [artifact.__dict__ for artifact in trace.candidates],
                }
                trace_path = TRACE_DIR / f"{task_slug(task_id)}.json"
                write_json(trace_path, trace_data)
                elapsed = time.perf_counter() - started
                candidates = trace_candidates(trace_data)
                row = {
                    "task_id": task_id,
                    "trace_id": trace.trace_id,
                    "trace_path": repo_path(trace_path),
                    "candidate_count": len(candidates),
                    "code_like_candidate_count": code_like_count(trace_data),
                    "stages_present": sorted(stage_counts(candidates)),
                    "stage_counts": stage_counts(candidates),
                    "selected_candidate_uid": trace.selected_candidate_uid,
                    "final_answer_present": bool(trace.final_answer),
                    "tool_stages_present": any(str(c.get("stage", "")).endswith("tool_code") or "tool" in str(c.get("stage", "")) for c in candidates),
                    "rejected_stages_count": sum(1 for c in candidates if str(c.get("stage", "")).startswith("rejected")),
                    "elapsed_seconds": round(elapsed, 3),
                    "wall_time_cap_exceeded": elapsed > (DEVIL_CAP if task.get("is_devil") else NON_DEVIL_CAP),
                    "error": "",
                }
                rows.append(row)
                completed.append(task_id)
            except Exception as exc:  # noqa: BLE001 - report and continue
                elapsed = time.perf_counter() - started
                err = {
                    "task_id": task_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": round(elapsed, 3),
                }
                print(f"[wrapper-trace] error {task_id}: {err['error']}", flush=True)
                errors.append(err)
            write_checkpoint(rows, errors, completed)
    finally:
        if hasattr(model_mgr, "unload"):
            try:
                model_mgr.unload()
            except Exception:
                pass

    useful = [row for row in rows if row["code_like_candidate_count"] >= 2]
    if len(useful) >= 8:
        verdict = "READY"
    elif len(useful) >= 4:
        verdict = "PARTIAL"
    else:
        verdict = "BLOCKED"
    payload = {
        "WRAPPER_TRACE_GENERATION_VERDICT": verdict,
        "candidate_traces": rows,
        "errors": errors,
        "summary": {
            "tasks_attempted": len(rows) + len(errors),
            "traces_written": len(rows),
            "useful_traces_ge2_code_like": len(useful),
            "total_candidates": sum(row["candidate_count"] for row in rows),
            "total_code_like_candidates": sum(row["code_like_candidate_count"] for row in rows),
            "stage_counts": dict(Counter(stage for row in rows for stage in row["stages_present"])),
            "elapsed_seconds": round(time.perf_counter() - started_all, 3),
        },
    }
    write_json(OUT_JSON, payload)
    lines = [
        "# Wrapper Candidate Traces",
        "",
        f"WRAPPER_TRACE_GENERATION_VERDICT = {verdict}",
        "",
        f"- traces written: `{len(rows)}`",
        f"- useful traces (>=2 code-like): `{len(useful)}`",
        f"- total candidates: `{payload['summary']['total_candidates']}`",
        f"- total code-like candidates: `{payload['summary']['total_code_like_candidates']}`",
        "",
        "| task_id | candidates | code-like | stages | selected | elapsed | error |",
        "| --- | ---: | ---: | --- | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['task_id']}` | {row['candidate_count']} | {row['code_like_candidate_count']} | "
            f"`{','.join(row['stages_present'])}` | `{row.get('selected_candidate_uid') or ''}` | "
            f"{row['elapsed_seconds']} | `{row.get('error','')}` |"
        )
    for err in errors:
        lines.append(f"| `{err['task_id']}` | 0 | 0 |  |  | {err.get('elapsed_seconds', 0)} | `{err['error']}` |")
    write_text(OUT_MD, "\n".join(lines) + "\n")
    print(f"WRAPPER_TRACE_GENERATION_VERDICT = {verdict}")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    if verdict == "BLOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
