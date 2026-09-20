"""Build a tiny code-branch taskset from non-devil local tasks plus MBPP."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from code_branch_pilot_lib import (
    REPORT_DIR,
    function_arity_from_tests,
    function_name_from_tests,
    function_signature_hint,
    repo_path,
    snippet,
    write_json,
)


DEFAULT_OUTPUT = REPORT_DIR / "code_branch_taskset_2026-05-16.json"
DEFAULT_MD = REPORT_DIR / "code_branch_taskset_2026-05-16.md"

DEVIL_EXCLUDED = {"offline_dynamic_connectivity", "minimum_xor_paths"}


LOCAL_TASKS: list[dict[str, Any]] = [
    {
        "task_id": "local_dsa/subarray_sum_count",
        "source": "local_dsa",
        "difficulty_label": "normal",
        "function_name": "subarray_sum_count",
        "prompt": """Implement a Python function:

```python
def subarray_sum_count(nums, k):
    ...
```

Return the number of contiguous subarrays whose sum equals `k`. `nums` may contain
negative values, zeroes, and duplicates. Provide complete code only.""",
        "public_tests": [
            "assert subarray_sum_count([1, 1, 1], 2) == 2",
            "assert subarray_sum_count([1, -1, 0], 0) == 3",
        ],
        "hidden_tests": [
            "assert subarray_sum_count([3, 4, 7, 2, -3, 1, 4, 2], 7) == 4",
            "assert subarray_sum_count([], 0) == 0",
            "assert subarray_sum_count([0, 0, 0], 0) == 6",
        ],
        "timeout_seconds": 3.0,
    },
    {
        "task_id": "local_dsa/longest_increasing_subsequence",
        "source": "local_dsa",
        "difficulty_label": "normal",
        "function_name": "longest_increasing_subsequence",
        "prompt": """Implement a Python function:

```python
def longest_increasing_subsequence(nums):
    ...
```

Return the length of the longest strictly increasing subsequence in `nums`.
The input may contain duplicates and negative values. Provide complete code only.""",
        "public_tests": [
            "assert longest_increasing_subsequence([10, 9, 2, 5, 3, 7, 101, 18]) == 4",
            "assert longest_increasing_subsequence([7, 7, 7, 7]) == 1",
        ],
        "hidden_tests": [
            "assert longest_increasing_subsequence([0, 1, 0, 3, 2, 3]) == 4",
            "assert longest_increasing_subsequence([]) == 0",
            "assert longest_increasing_subsequence([-4, -3, -2, -1]) == 4",
        ],
        "timeout_seconds": 3.0,
    },
    {
        "task_id": "local_dsa/word_ladder_length",
        "source": "local_dsa",
        "difficulty_label": "hard_not_devil",
        "function_name": "word_ladder_length",
        "prompt": """Implement a Python function:

```python
def word_ladder_length(begin_word, end_word, word_list):
    ...
```

Each step may change exactly one character, and every intermediate word must be in
`word_list`. Return the number of words in the shortest transformation sequence
from `begin_word` to `end_word`, including both endpoints. Return `0` if no such
sequence exists. Provide complete code only.""",
        "public_tests": [
            "assert word_ladder_length('hit', 'cog', ['hot', 'dot', 'dog', 'lot', 'log', 'cog']) == 5",
            "assert word_ladder_length('hit', 'cog', ['hot', 'dot', 'dog', 'lot', 'log']) == 0",
        ],
        "hidden_tests": [
            "assert word_ladder_length('a', 'c', ['a', 'b', 'c']) == 2",
            "assert word_ladder_length('same', 'same', ['same']) == 1",
        ],
        "timeout_seconds": 3.0,
    },
    {
        "task_id": "local_dsa/count_smaller_after_self",
        "source": "local_dsa",
        "difficulty_label": "hard_not_devil",
        "function_name": "count_smaller_after_self",
        "prompt": """Implement a Python function:

```python
def count_smaller_after_self(nums):
    ...
```

For each index `i`, return how many elements to the right of `i` are strictly
smaller than `nums[i]`. The input may contain duplicates and negative values.
Provide complete code only.""",
        "public_tests": [
            "assert count_smaller_after_self([5, 2, 6, 1]) == [2, 1, 1, 0]",
            "assert count_smaller_after_self([-1, -1]) == [0, 0]",
        ],
        "hidden_tests": [
            "assert count_smaller_after_self([3, 2, 2, 6, 1]) == [3, 1, 1, 1, 0]",
            "assert count_smaller_after_self([]) == []",
            "assert count_smaller_after_self([1, 2, 3, 4]) == [0, 0, 0, 0]",
        ],
        "timeout_seconds": 3.0,
    },
]


def normalise_task(task: dict[str, Any]) -> dict[str, Any]:
    public_tests = [str(x).strip() for x in task.get("public_tests", []) if str(x).strip()]
    hidden_tests = [str(x).strip() for x in task.get("hidden_tests", []) if str(x).strip()]
    all_tests = public_tests + hidden_tests
    task["tests"] = all_tests
    task["starter_code"] = task.get("starter_code", f"def {task['function_name']}(...):\n    ...")
    return task


def load_mbpp_tasks(target_count: int) -> tuple[list[dict[str, Any]], str]:
    try:
        from datasets import load_dataset
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"

    tasks: list[dict[str, Any]] = []
    seen_names = {task["function_name"] for task in LOCAL_TASKS}
    for row in ds:
        tests = [str(x).strip() for x in row.get("test_list", []) if str(x).strip()]
        if len(tests) < 3:
            continue
        fn = function_name_from_tests(tests)
        if not fn or fn in seen_names:
            continue
        arity = function_arity_from_tests(tests, fn)
        signature = function_signature_hint(fn, arity)
        prompt = str(row.get("prompt") or "").strip()
        if not prompt or len(prompt) > 420:
            continue
        if re.search(r"\b(input|stdin|print)\b", prompt, flags=re.IGNORECASE):
            continue
        public_tests = tests[:1]
        hidden_tests = tests[1:]
        full_prompt = (
            f"{prompt}\n\n"
            f"Implement a Python function named `{fn}`. Use this exact callable signature shape: "
            f"`{signature}`. Provide complete code only.\n\n"
            "Public example:\n"
            f"```python\n{public_tests[0]}\n```"
        )
        tasks.append(normalise_task({
            "task_id": f"mbpp/{row.get('task_id')}",
            "source": "mbpp",
            "difficulty_label": "benchmark",
            "function_name": fn,
            "signature_hint": signature,
            "arity_hint": arity,
            "prompt": full_prompt,
            "starter_code": f"{signature}\n    ...",
            "public_tests": public_tests,
            "hidden_tests": hidden_tests,
            "timeout_seconds": 3.0,
            "mbpp_source_file": row.get("source_file", ""),
        }))
        seen_names.add(fn)
        if len(tasks) >= target_count:
            break
    return tasks, ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-tasks", type=int, default=10)
    parser.add_argument("--min-benchmark-tasks", type=int, default=6)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--output-md", default=str(DEFAULT_MD))
    return parser.parse_args()


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Code Branch Taskset",
        "",
        f"CODE_TASKSET_VERDICT = {payload['code_taskset_verdict']}",
        "",
        f"- tasks: `{len(payload['tasks'])}`",
        f"- benchmark_tasks: `{payload['benchmark_tasks']}`",
        f"- handwritten_smoke_tasks: `{payload['handwritten_smoke_tasks']}`",
        f"- devil_excluded: `{payload['devil_excluded']}`",
        f"- dataset_status: `{payload['dataset_status']}`",
        "",
        "## Tasks",
        "",
        "| task_id | source | function | public | hidden | prompt |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for task in payload["tasks"]:
        lines.append(
            f"| `{task['task_id']}` | `{task['source']}` | `{task['function_name']}` | "
            f"{len(task.get('public_tests', []))} | {len(task.get('hidden_tests', []))} | "
            f"{snippet(task['prompt'], 120)} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    local = [normalise_task(dict(task)) for task in LOCAL_TASKS]
    needed = max(0, int(args.target_tasks) - len(local))
    mbpp, dataset_error = load_mbpp_tasks(needed)
    tasks = (local + mbpp)[: int(args.target_tasks)]
    benchmark_tasks = sum(1 for task in tasks if task["source"] != "handwritten_smoke")
    handwritten = sum(1 for task in tasks if task["source"] == "handwritten_smoke")
    has_new_dataset = any(task["source"] == "mbpp" for task in tasks)
    if benchmark_tasks >= int(args.min_benchmark_tasks) and has_new_dataset:
        verdict = "READY"
    elif tasks and not has_new_dataset:
        verdict = "BLOCKED"
    else:
        verdict = "BLOCKED"
    payload = {
        "code_taskset_verdict": verdict,
        "tasks": tasks,
        "benchmark_tasks": benchmark_tasks,
        "handwritten_smoke_tasks": handwritten,
        "devil_excluded": sorted(DEVIL_EXCLUDED),
        "dataset_status": "mbpp_loaded" if has_new_dataset else f"mbpp_unavailable: {dataset_error}",
        "notes": (
            "The two local devil-grade tasks are excluded by user direction. "
            "The other four local DSA tasks are retained and combined with MBPP sanitized tasks."
        ),
        "outputs": {"json": repo_path(Path(args.output)), "md": repo_path(Path(args.output_md))},
    }
    out_json = Path(args.output)
    out_md = Path(args.output_md)
    write_json(out_json, payload)
    write_md(out_md, payload)
    print(f"CODE_TASKSET_VERDICT = {verdict}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
