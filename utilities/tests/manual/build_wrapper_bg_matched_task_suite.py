"""Build a bounded task suite for wrapper-vs-BG matched comparison."""

from __future__ import annotations

from collections import Counter
from typing import Any

from wrapper_bg_matched_lib import (
    OUT_DIR,
    TASKSET_SOURCE,
    load_json,
    markdown_table,
    normalize_task,
    repo_path,
    snippet,
    write_json,
    write_text,
)


OUT_JSON = OUT_DIR / "task_suite.json"
OUT_MD = OUT_DIR / "task_suite.md"
TARGET_TASKS = 20


def select_tasks(source_tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks = [normalize_task(task) for task in source_tasks if task.get("prompt") and task.get("function_name") and task.get("tests")]
    by_id = {task["task_id"]: task for task in tasks}
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    for task_id in ("local_dsa/offline_dynamic_connectivity", "local_dsa/minimum_xor_paths"):
        task = by_id.get(task_id)
        if task is not None:
            selected.append(task)
            seen.add(task_id)

    hard = [task for task in tasks if task["task_id"] not in seen and task.get("difficulty") == "hard"]
    local = [task for task in tasks if task["task_id"] not in seen and task.get("source") == "local_dsa"]
    humaneval = [task for task in tasks if task["task_id"] not in seen and task.get("source") == "humaneval"]
    mbpp = [task for task in tasks if task["task_id"] not in seen and task.get("source") == "mbpp"]
    for bucket, cap in ((hard, 5), (local, 3), (humaneval, 6), (mbpp, 10)):
        for task in bucket:
            if len(selected) >= TARGET_TASKS:
                break
            if task["task_id"] in seen:
                continue
            selected.append(task)
            seen.add(task["task_id"])

    for task in tasks:
        if len(selected) >= TARGET_TASKS:
            break
        if task["task_id"] not in seen:
            selected.append(task)
            seen.add(task["task_id"])
    return selected[:TARGET_TASKS]


def verdict_for(tasks: list[dict[str, Any]]) -> str:
    evaluable = [task for task in tasks if task.get("tests")]
    non_devil = [task for task in evaluable if not task.get("is_devil")]
    hard_or_devil = [task for task in evaluable if task.get("difficulty") in {"hard", "devil"}]
    if len(evaluable) >= 12 and len(non_devil) >= 8 and hard_or_devil:
        return "READY"
    if len(evaluable) >= 6:
        return "PARTIAL"
    return "BLOCKED"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_payload = load_json(TASKSET_SOURCE)
    tasks = select_tasks(source_payload.get("tasks", []))
    verdict = verdict_for(tasks)
    payload = {
        "WRAPPER_BG_TASK_SUITE_VERDICT": verdict,
        "source_taskset": repo_path(TASKSET_SOURCE),
        "target_tasks": TARGET_TASKS,
        "tasks": tasks,
        "summary": {
            "task_count": len(tasks),
            "evaluable_count": sum(1 for task in tasks if task.get("tests")),
            "non_devil_count": sum(1 for task in tasks if not task.get("is_devil")),
            "devil_count": sum(1 for task in tasks if task.get("is_devil")),
            "source_mix": dict(Counter(task.get("source", "unknown") for task in tasks)),
            "difficulty_mix": dict(Counter(task.get("difficulty", "unknown") for task in tasks)),
        },
    }
    write_json(OUT_JSON, payload)
    rows = [
        [
            f"`{task['task_id']}`",
            f"`{task.get('source')}`",
            f"`{task.get('difficulty')}`",
            f"`{task.get('function_name')}`",
            len(task.get("tests") or []),
            "yes" if task.get("is_devil") else "no",
            snippet(task.get("prompt", ""), 110),
        ]
        for task in tasks
    ]
    lines = [
        "# Wrapper-Matched BG Task Suite",
        "",
        f"WRAPPER_BG_TASK_SUITE_VERDICT = {verdict}",
        "",
        f"- source taskset: `{repo_path(TASKSET_SOURCE)}`",
        f"- tasks: `{len(tasks)}`",
        f"- source mix: `{payload['summary']['source_mix']}`",
        f"- difficulty mix: `{payload['summary']['difficulty_mix']}`",
        "",
        *markdown_table(rows, ["task_id", "source", "difficulty", "function", "tests", "devil", "prompt"]),
    ]
    write_text(OUT_MD, "\n".join(lines) + "\n")
    print(f"WRAPPER_BG_TASK_SUITE_VERDICT = {verdict}")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    if verdict == "BLOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
