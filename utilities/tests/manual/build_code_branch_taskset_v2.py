"""Build an expanded, harder code-branch taskset for pilot v2."""
from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from code_branch_pilot_lib import (
    PROJECT_ROOT,
    REPORT_DIR,
    function_arity_from_tests,
    function_name_from_tests,
    function_signature_hint,
    repo_path,
    snippet,
    write_json,
)
from build_code_branch_taskset import LOCAL_TASKS, normalise_task


DEFAULT_OUTPUT = REPORT_DIR / "code_branch_taskset_v2_mini_patched_2026-05-16.json"
DEFAULT_MD = REPORT_DIR / "code_branch_taskset_v2_mini_patched_2026-05-16.md"

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "artifacts" / "hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "artifacts" / "hf_cache" / "datasets"))


DEVIL_TASKS: list[dict[str, Any]] = [
    {
        "task_id": "local_dsa/offline_dynamic_connectivity",
        "source": "local_dsa",
        "difficulty": "devil",
        "function_name": "offline_dynamic_connectivity",
        "prompt": """
Implement a Python function:

```python
def offline_dynamic_connectivity(n, events):
    ...
```

`n` is the number of 0-indexed vertices. `events` is a list of tuples:
`("add", u, v)`, `("remove", u, v)`, or `("query", u, v)`.
An add activates an undirected edge, a remove deactivates it, and a query asks
whether `u` and `v` are connected at that instant. Adds are not duplicated while
active. Removes only appear for currently active edges. Return the list of
booleans for the queries, in order.

The intended asymptotic solution is offline dynamic connectivity with a segment
tree over time and rollback DSU. Provide complete code only.
""".strip(),
        "public_tests": [
            "events=[('add',0,1),('query',0,1),('query',0,2),('add',1,2),('query',0,2),('remove',0,1),('query',0,2)]\nassert offline_dynamic_connectivity(3, events) == [True, False, True, False]",
            "assert offline_dynamic_connectivity(2, [('query',0,1),('add',0,1),('query',0,1),('remove',0,1),('query',0,1)]) == [False, True, False]",
        ],
        "hidden_tests": [
            """
def _naive_dc(n, events):
    active=set(); out=[]
    for kind,u,v in events:
        e=(u,v) if u<=v else (v,u)
        if kind=='add':
            active.add(e)
        elif kind=='remove':
            active.discard(e)
        else:
            graph=[[] for _ in range(n)]
            for a,b in active:
                graph[a].append(b); graph[b].append(a)
            stack=[u]; seen={u}
            while stack:
                cur=stack.pop()
                for nxt in graph[cur]:
                    if nxt not in seen:
                        seen.add(nxt); stack.append(nxt)
            out.append(v in seen)
    return out
events=[('add',0,1),('add',2,3),('query',0,3),('add',1,2),('query',0,3),('remove',1,2),('query',0,3),('add',3,4),('query',2,4)]
assert offline_dynamic_connectivity(5, events) == _naive_dc(5, events)
""".strip(),
            """
import random
rng=random.Random(123)
for _case in range(8):
    n=6
    active=set(); events=[]
    for _ in range(35):
        possible=[(a,b) for a in range(n) for b in range(a+1,n)]
        inactive=[e for e in possible if e not in active]
        roll=rng.random()
        if roll<0.38 and inactive:
            e=rng.choice(inactive); active.add(e); events.append(('add',e[0],e[1]))
        elif roll<0.62 and active:
            e=rng.choice(tuple(active)); active.remove(e); events.append(('remove',e[0],e[1]))
        else:
            events.append(('query',rng.randrange(n),rng.randrange(n)))
    assert offline_dynamic_connectivity(n, events) == _naive_dc(n, events)
""".strip(),
        ],
        "timeout_seconds": 8.0,
        "tests_visibility": "local_public_and_hidden",
    },
    {
        "task_id": "local_dsa/minimum_xor_paths",
        "source": "local_dsa",
        "difficulty": "devil",
        "function_name": "minimum_xor_paths",
        "prompt": """
Implement a Python function:

```python
def minimum_xor_paths(n, edges, queries):
    ...
```

`edges` contains undirected weighted edges `(u, v, w)`. For each query `(s, t)`,
return the minimum possible XOR value of any walk from `s` to `t`; because walks
may revisit vertices, cycle XORs may be used to reduce the path XOR. Return `-1`
when `s` and `t` are disconnected. Vertices are 0-indexed.

The intended solution is DFS/DSU over components plus a linear XOR basis for
cycle values. Provide complete code only.
""".strip(),
        "public_tests": [
            "edges=[(0,1,7),(1,2,3),(0,2,6),(2,3,4)]\nassert minimum_xor_paths(5, edges, [(0,2),(0,3),(3,1),(0,4)]) == [0,4,7,-1]",
            "assert minimum_xor_paths(3, [(0,1,5)], [(0,1),(1,2)]) == [5,-1]",
        ],
        "hidden_tests": [
            """
def _brute_min_xor(n, edges, queries):
    graph=[[] for _ in range(n)]
    max_xor=64
    for u,v,w in edges:
        graph[u].append((v,w)); graph[v].append((u,w))
        max_xor=max(max_xor, 1 << max(1, int(w).bit_length()))
    out=[]
    for src,dst in queries:
        seen={(src,0)}; queue=[(src,0)]; best=None; head=0
        while head < len(queue):
            node,value=queue[head]; head += 1
            if node == dst:
                best = value if best is None else min(best, value)
            for nxt,weight in graph[node]:
                state=(nxt, value ^ weight)
                if state not in seen and state[1] < max_xor:
                    seen.add(state); queue.append(state)
        out.append(-1 if best is None else best)
    return out
edges=[(0,1,2),(1,2,4),(2,0,7),(2,3,8),(3,4,3),(1,4,10)]
queries=[(0,2),(0,4),(3,0),(4,4)]
assert minimum_xor_paths(5, edges, queries) == _brute_min_xor(5, edges, queries)
""".strip(),
            """
import random
rng=random.Random(456)
for _case in range(8):
    n=rng.randint(2,6)
    edges=[]
    for u in range(n):
        for v in range(u+1,n):
            if rng.random() < 0.45:
                edges.append((u,v,rng.randrange(1,16)))
    queries=[(rng.randrange(n), rng.randrange(n)) for _ in range(10)]
    assert minimum_xor_paths(n, edges, queries) == _brute_min_xor(n, edges, queries)
""".strip(),
        ],
        "timeout_seconds": 8.0,
        "tests_visibility": "local_public_and_hidden",
    },
]


HANDWRITTEN_SMOKE: list[dict[str, Any]] = [
    {
        "task_id": "handwritten_smoke/rotate_left",
        "source": "handwritten_smoke",
        "difficulty": "easy",
        "function_name": "rotate_left",
        "prompt": "Implement `rotate_left(nums, k)` returning a new list rotated left by k positions.",
        "public_tests": ["assert rotate_left([1,2,3,4], 1) == [2,3,4,1]"],
        "hidden_tests": ["assert rotate_left([1,2,3], 4) == [2,3,1]", "assert rotate_left([], 5) == []"],
        "timeout_seconds": 3.0,
        "tests_visibility": "handwritten_smoke",
    }
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-tasks", type=int, default=40)
    parser.add_argument("--min-tasks", type=int, default=25)
    parser.add_argument("--include-devil", action="store_true")
    parser.add_argument("--include-specific", nargs="*", default=[])
    parser.add_argument("--max-easy-fraction", type=float, default=0.40)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()
    if not args.output_md:
        args.output_md = str(Path(args.output).with_suffix(".md"))
    return args


def task_difficulty(task: dict[str, Any]) -> str:
    if task.get("difficulty") in {"easy", "medium", "hard", "devil", "unknown"}:
        return str(task["difficulty"])
    label = str(task.get("difficulty_label", "")).lower()
    prompt = str(task.get("prompt", "")).lower()
    name = str(task.get("function_name", "")).lower()
    if label in {"devil", "devil_grade", "devil-grade"}:
        return "devil"
    if "hard" in label or any(term in prompt + " " + name for term in ("graph", "shortest", "subsequence", "dynamic", "xor", "connectivity", "ladder")):
        return "hard"
    if task.get("source") == "humaneval":
        return "medium"
    if task.get("source") == "mbpp":
        easy_terms = ("string", "tuple", "list", "remove", "count", "sum", "area", "volume", "valid", "sort")
        if len(prompt) < 170 and any(term in prompt for term in easy_terms):
            return "easy"
        return "medium"
    return "medium"


def finalize_task(task: dict[str, Any]) -> dict[str, Any]:
    out = normalise_task(dict(task))
    out["difficulty"] = task_difficulty(out)
    out.setdefault("tests_visibility", "public_and_hidden")
    out.setdefault("timeout_seconds", 5.0)
    return out


def mbpp_task_note(task_id: str) -> str:
    notes = {
        "mbpp/237": (
            "Important: tuples that contain the same values in different orders are treated as the same key; "
            "canonicalize each tuple by sorting it before counting."
        ),
    }
    return notes.get(task_id, "")


def load_humaneval_tasks(limit: int) -> tuple[list[dict[str, Any]], str]:
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/openai_humaneval", split="test")
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"
    tasks: list[dict[str, Any]] = []
    skip_names = {task["function_name"] for task in LOCAL_TASKS} | {task["function_name"] for task in DEVIL_TASKS}
    rows = sorted(list(ds), key=lambda row: (len(str(row.get("prompt") or "")), str(row.get("task_id"))), reverse=True)
    for row in rows:
        fn = str(row.get("entry_point") or "").strip()
        prompt = str(row.get("prompt") or "").strip()
        test = str(row.get("test") or "").strip()
        if not fn or not prompt or not test or fn in skip_names:
            continue
        if len(prompt) > 2200:
            continue
        hidden = test + f"\ncheck({fn})"
        tasks.append(finalize_task({
            "task_id": str(row.get("task_id")),
            "source": "humaneval",
            "difficulty": "medium",
            "function_name": fn,
            "prompt": prompt + "\n\nComplete the function above. Provide complete Python code only.",
            "starter_code": prompt,
            "public_tests": [],
            "hidden_tests": [hidden],
            "timeout_seconds": 5.0,
            "tests_visibility": "official_hidden",
        }))
        skip_names.add(fn)
        if len(tasks) >= limit:
            break
    return tasks, ""


def load_mbpp_tasks(limit: int, existing_names: set[str], include_task_ids: set[str] | None = None) -> tuple[list[dict[str, Any]], str]:
    include_task_ids = include_task_ids or set()
    try:
        from datasets import load_dataset
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"
    tasks: list[dict[str, Any]] = []
    rows = []
    for row in ds:
        tests = [str(x).strip() for x in row.get("test_list", []) if str(x).strip()]
        fn = function_name_from_tests(tests)
        prompt = str(row.get("prompt") or "").strip()
        if not fn or fn in existing_names or len(tests) < 3 or not prompt:
            continue
        if re.search(r"\b(input|stdin|print)\b", prompt, flags=re.IGNORECASE):
            continue
        source = str(row.get("source_file") or "")
        task_id = f"mbpp/{row.get('task_id')}"
        arity = function_arity_from_tests(tests, fn)
        signature = function_signature_hint(fn, arity)
        note = mbpp_task_note(task_id)
        note_text = f"\n\n{note}" if note else ""
        task = finalize_task({
            "task_id": task_id,
            "source": "mbpp",
            "function_name": fn,
            "prompt": (
                f"{prompt}\n\nImplement a Python function named `{fn}`. Use this exact callable signature shape: "
                f"`{signature}`. Provide complete code only.{note_text}\n\n"
                f"Public example:\n```python\n{tests[0]}\n```"
            ),
            "signature_hint": signature,
            "arity_hint": arity,
            "starter_code": f"{signature}\n    ...",
            "public_tests": tests[:1],
            "hidden_tests": tests[1:],
            "timeout_seconds": 4.0,
            "tests_visibility": "official_public_and_hidden",
            "mbpp_source_file": source,
        })
        rows.append(task)
    rows.sort(key=lambda task: (task["difficulty"] == "easy", -len(task["prompt"]), task["task_id"]))
    included = [task for task in rows if task["task_id"] in include_task_ids]
    remaining = [task for task in rows if task["task_id"] not in include_task_ids]
    for task in included + remaining:
        if task["function_name"] in existing_names:
            continue
        tasks.append(task)
        existing_names.add(task["function_name"])
        if len(tasks) >= limit and include_task_ids.issubset({item["task_id"] for item in tasks}):
            break
    return tasks, ""


def select_tasks(pool: list[dict[str, Any]], target: int, max_easy_fraction: float, include_task_ids: set[str] | None = None) -> list[dict[str, Any]]:
    include_task_ids = include_task_ids or set()
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    easy_cap = int(target * max_easy_fraction)
    order = {"devil": 0, "hard": 1, "medium": 2, "unknown": 3, "easy": 4}
    by_id = {task["task_id"]: task for task in pool}
    for task_id in sorted(include_task_ids):
        task = by_id.get(task_id)
        if task is not None and task_id not in seen:
            selected.append(task)
            seen.add(task_id)

    # Keep the mini set mixed: 2-4 local DSA, 1-2 devil tasks, some HumanEval, and the required MBPP patches.
    source_caps = {"local_dsa": 4, "humaneval": max(2, target // 3), "mbpp": max(4, target // 2)}
    devil_cap = 2
    for task in sorted(pool, key=lambda t: (order.get(t.get("difficulty", "unknown"), 9), t["source"], t["task_id"])):
        if len(selected) >= target:
            break
        if task["task_id"] in seen:
            continue
        if task.get("difficulty") == "easy" and sum(1 for item in selected if item.get("difficulty") == "easy") >= easy_cap:
            continue
        if task.get("difficulty") == "devil" and sum(1 for item in selected if item.get("difficulty") == "devil") >= devil_cap:
            continue
        source = str(task.get("source", ""))
        if source in source_caps and sum(1 for item in selected if item.get("source") == source) >= source_caps[source]:
            continue
        selected.append(task)
        seen.add(task["task_id"])
    if len(selected) < target:
        for task in pool:
            if len(selected) >= target:
                break
            if task["task_id"] not in seen:
                selected.append(task)
                seen.add(task["task_id"])
    return selected


def verdict_for(tasks: list[dict[str, Any]], min_tasks: int) -> str:
    if len(tasks) < min_tasks:
        return "BLOCKED"
    if all(task["source"] == "handwritten_smoke" for task in tasks):
        return "SMOKE_ONLY"
    medium_hard = sum(1 for task in tasks if task["difficulty"] in {"medium", "hard", "devil"})
    hard_devil = sum(1 for task in tasks if task["difficulty"] in {"hard", "devil"})
    easy = sum(1 for task in tasks if task["difficulty"] == "easy")
    if len(tasks) >= min_tasks and (hard_devil >= 3 or medium_hard >= min_tasks):
        return "TOO_EASY" if easy / max(len(tasks), 1) > 0.40 else "READY"
    return "TOO_EASY"


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Code Branch Taskset v2",
        "",
        f"CODE_V2_TASKSET_VERDICT = {payload['code_v2_taskset_verdict']}",
        "",
        f"- tasks: `{len(payload['tasks'])}`",
        f"- source_mix: `{payload['source_mix']}`",
        f"- difficulty_mix: `{payload['difficulty_mix']}`",
        f"- dataset_status: `{payload['dataset_status']}`",
        f"- tests_visibility_mix: `{payload['tests_visibility_mix']}`",
        "",
        "## Tasks",
        "",
        "| task_id | source | difficulty | function | public | hidden | visibility | prompt |",
        "| --- | --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for task in payload["tasks"]:
        lines.append(
            f"| `{task['task_id']}` | `{task['source']}` | `{task['difficulty']}` | `{task['function_name']}` | "
            f"{len(task.get('public_tests', []))} | {len(task.get('hidden_tests', []))} | `{task.get('tests_visibility', '')}` | "
            f"{snippet(task['prompt'], 120)} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    include_specific = {str(x) for x in args.include_specific}
    local = [finalize_task({**task, "difficulty": task_difficulty(task), "tests_visibility": "local_public_and_hidden"}) for task in LOCAL_TASKS]
    if args.include_devil:
        local.extend(finalize_task(task) for task in DEVIL_TASKS)
    human, human_error = load_humaneval_tasks(max(2, int(args.target_tasks) // 3))
    existing = {task["function_name"] for task in local + human}
    mbpp, mbpp_error = load_mbpp_tasks(max(int(args.target_tasks), len(include_specific) + 6), existing, include_specific)
    pool = local + human + mbpp
    if len(pool) < int(args.min_tasks):
        pool.extend(finalize_task(task) for task in HANDWRITTEN_SMOKE)
    tasks = select_tasks(pool, int(args.target_tasks), float(args.max_easy_fraction), include_specific)
    verdict = verdict_for(tasks, int(args.min_tasks))
    source_mix = dict(Counter(task["source"] for task in tasks))
    difficulty_mix = dict(Counter(task["difficulty"] for task in tasks))
    payload = {
        "code_v2_taskset_verdict": verdict,
        "tasks": tasks,
        "target_tasks": int(args.target_tasks),
        "min_tasks": int(args.min_tasks),
        "source_mix": source_mix,
        "difficulty_mix": difficulty_mix,
        "tests_visibility_mix": dict(Counter(task.get("tests_visibility", "unknown") for task in tasks)),
        "dataset_status": {
            "humaneval": "loaded" if human else f"unavailable: {human_error}",
            "mbpp": "loaded" if mbpp else f"unavailable: {mbpp_error}",
            "local_dsa": len(local),
        },
        "mix_targets": {
            "hard_devil": sum(1 for task in tasks if task["difficulty"] in {"hard", "devil"}),
            "medium_hard": sum(1 for task in tasks if task["difficulty"] in {"medium", "hard", "devil"}),
            "easy_fraction": sum(1 for task in tasks if task["difficulty"] == "easy") / max(len(tasks), 1),
            "included_specific": sorted(task_id for task_id in include_specific if task_id in {task["task_id"] for task in tasks}),
            "missing_specific": sorted(task_id for task_id in include_specific if task_id not in {task["task_id"] for task in tasks}),
        },
        "outputs": {"json": repo_path(Path(args.output)), "md": repo_path(Path(args.output_md))},
    }
    out_json = Path(args.output)
    out_md = Path(args.output_md)
    write_json(out_json, payload)
    write_md(out_md, payload)
    print(f"CODE_V2_TASKSET_VERDICT = {verdict}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
