"""Build a 10-task code-branch near-miss enrichment taskset.

This taskset deliberately avoids HumanEval-style monolithic checks and favors
MBPP/local tasks with multiple assertions so partial-pass candidates can be
distinguished from zero-pass wrong code.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from build_code_branch_taskset import LOCAL_TASKS
from build_code_branch_taskset_v2 import finalize_task, load_mbpp_tasks
from code_branch_pilot_lib import REPORT_DIR, repo_path, snippet, write_json


DEFAULT_OUTPUT = REPORT_DIR / "code_branch_taskset_v2_near_miss10_2026-05-17.json"
DEFAULT_MD = REPORT_DIR / "code_branch_taskset_v2_near_miss10_2026-05-17.md"

LOCAL_INCLUDE_IDS = [
    "local_dsa/count_smaller_after_self",
    "local_dsa/word_ladder_length",
]

MBPP_INCLUDE_IDS = [
    "mbpp/108",
    "mbpp/223",
    "mbpp/229",
    "mbpp/239",
    "mbpp/245",
    "mbpp/251",
    "mbpp/255",
    "mbpp/265",
]

PRIOR_NEAR_MISS_SIGNAL = {
    "local_dsa/count_smaller_after_self": "prior patched v2-mini produced correct+near_miss",
    "mbpp/108": "prior patched v2-mini produced near_miss candidates but lacked a correct anchor",
    "mbpp/223": "prior patched v2-mini produced correct+near_miss",
    "mbpp/229": "prior patched v2-mini produced near_miss candidates but lacked a correct anchor",
    "mbpp/239": "prior patched v2-mini produced wrong_code+near_miss candidates",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()
    if not args.output_md:
        args.output_md = str(Path(args.output).with_suffix(".md"))
    return args


def local_tasks_by_id() -> dict[str, dict[str, Any]]:
    tasks = [finalize_task({**task, "tests_visibility": "local_public_and_hidden"}) for task in LOCAL_TASKS]
    return {task["task_id"]: task for task in tasks}


def task_reason(task: dict[str, Any]) -> str:
    parts = [
        "granular_tests",
        "mbpp_or_local",
        f"{len(task.get('tests', []))}_assertion_groups",
    ]
    if task["task_id"] in PRIOR_NEAR_MISS_SIGNAL:
        parts.append(PRIOR_NEAR_MISS_SIGNAL[task["task_id"]])
    if task.get("difficulty") in {"hard", "devil"}:
        parts.append("medium_hard_surface")
    return "; ".join(parts)


def verdict_for(tasks: list[dict[str, Any]]) -> str:
    if len(tasks) != 10:
        return "BLOCKED"
    if any(task.get("source") == "humaneval" for task in tasks):
        return "BLOCKED"
    if any(len(task.get("tests", [])) < 3 for task in tasks):
        return "BLOCKED"
    return "READY"


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Code Branch Near-Miss Enrichment Taskset v2",
        "",
        f"CODE_V2_NEARMISS10_TASKSET_VERDICT = {payload['code_v2_nearmiss10_taskset_verdict']}",
        "",
        f"- tasks: `{len(payload['tasks'])}`",
        f"- source_mix: `{payload['source_mix']}`",
        f"- difficulty_mix: `{payload['difficulty_mix']}`",
        f"- target: `correct vs near_miss_partial_pass`",
        f"- avoided_sources: `{payload['avoided_sources']}`",
        "",
        "| task_id | source | difficulty | function | tests | reason | prompt |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for task in payload["tasks"]:
        lines.append(
            f"| `{task['task_id']}` | `{task['source']}` | `{task['difficulty']}` | "
            f"`{task['function_name']}` | {len(task.get('tests', []))} | "
            f"{task['selection_reason']} | {snippet(task['prompt'], 120)} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    local_by_id = local_tasks_by_id()
    local = [local_by_id[task_id] for task_id in LOCAL_INCLUDE_IDS if task_id in local_by_id]
    existing_names = {task["function_name"] for task in local}
    mbpp, mbpp_error = load_mbpp_tasks(20, existing_names, set(MBPP_INCLUDE_IDS))
    mbpp_by_id = {task["task_id"]: task for task in mbpp}
    tasks = local + [mbpp_by_id[task_id] for task_id in MBPP_INCLUDE_IDS if task_id in mbpp_by_id]
    tasks = tasks[:10]
    for task in tasks:
        task["selection_reason"] = task_reason(task)
        task["near_miss_enrichment_role"] = (
            "prior_near_miss_seed" if task["task_id"] in PRIOR_NEAR_MISS_SIGNAL else "similar_granular_task"
        )
    verdict = verdict_for(tasks)
    payload = {
        "code_v2_nearmiss10_taskset_verdict": verdict,
        "tasks": tasks,
        "target_tasks": 10,
        "source_mix": dict(Counter(task["source"] for task in tasks)),
        "difficulty_mix": dict(Counter(task.get("difficulty", "unknown") for task in tasks)),
        "dataset_status": {"mbpp": "loaded" if mbpp else f"unavailable: {mbpp_error}"},
        "avoided_sources": ["humaneval_monolithic_checks"],
        "candidate_strategy": [
            "one repaired_final positive anchor per task",
            "direct_short_budget",
            "first_tool_code",
            "first_failed_or_first_repair_code",
            "direct_sampled_high",
            "direct_sampled_medium",
        ],
        "success_criterion": "strict_clean >= 5",
        "outputs": {"json": repo_path(Path(args.output)), "md": repo_path(Path(args.output_md))},
    }
    out_json = Path(args.output)
    out_md = Path(args.output_md)
    write_json(out_json, payload)
    write_md(out_md, payload)
    print(f"CODE_V2_NEARMISS10_TASKSET_VERDICT = {verdict}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    if verdict == "BLOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
