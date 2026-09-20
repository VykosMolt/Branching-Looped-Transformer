"""Generate shared initial partial branch pools for the BG steering suite."""
from __future__ import annotations

import time
import traceback
from collections import Counter, defaultdict

from bg_steering_suite_lib import (
    REPORT_ROOT,
    SEED,
    MAX_TOTAL_BRANCHES,
    OuroTextGenerator,
    load_task_suite,
    partial_budget,
    rel,
    task_generation_prompt,
    wall_time_exceeded,
    write_json,
    write_md,
)


OUT_JSON = REPORT_ROOT / "branch_pools.json"
OUT_PARTIAL = REPORT_ROOT / "branch_pools.partial.json"
OUT_MD = REPORT_ROOT / "branch_pools.md"


def main() -> int:
    started = time.time()
    tasks = load_task_suite()
    existing = {}
    branches = []
    old = {}
    if OUT_PARTIAL.exists():
        import json

        try:
            old = json.loads(OUT_PARTIAL.read_text(encoding="utf-8"))
        except Exception:
            old = {}
    for row in old.get("branches") or []:
        key = (row.get("task_id"), int(row.get("branch_id", -1)))
        existing[key] = row
        branches.append(row)

    warnings: list[str] = []
    generator: OuroTextGenerator | None = None
    try:
        generator = OuroTextGenerator(device="cuda")
        for task in tasks:
            task_id = str(task["task_id"])
            task_rows = [row for row in branches if row.get("task_id") == task_id]
            if len(task_rows) >= 4:
                continue
            prompt = task_generation_prompt(task)
            for branch_id in range(4):
                if len(branches) >= MAX_TOTAL_BRANCHES or wall_time_exceeded():
                    warnings.append("hard cap reached during branch generation")
                    break
                key = (task_id, branch_id)
                if key in existing:
                    continue
                budget = partial_budget(task)
                try:
                    gen = generator.generate(
                        prompt,
                        max_new_tokens=budget,
                        temperature=0.7,
                        top_p=0.95,
                        seed=SEED + int(task.get("suite_index", 0)) * 31 + branch_id,
                    )
                    row = {
                        "task_id": task_id,
                        "domain": task["domain"],
                        "branch_id": branch_id,
                        "initial_partial_text": gen["text"],
                        "raw_text": gen["raw_text"],
                        "token_count": gen["token_count"],
                        "hit_max_tokens": gen["hit_max_tokens"],
                        "generation_error": gen["generation_error"],
                        "generation_seconds": gen["seconds"],
                        "generation_params": {
                            "max_new_tokens": budget,
                            "temperature": 0.7,
                            "top_p": 0.95,
                            "seed": SEED + int(task.get("suite_index", 0)) * 31 + branch_id,
                        },
                    }
                except Exception as exc:
                    row = {
                        "task_id": task_id,
                        "domain": task["domain"],
                        "branch_id": branch_id,
                        "initial_partial_text": "",
                        "raw_text": "",
                        "token_count": 0,
                        "hit_max_tokens": False,
                        "generation_error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:],
                        "generation_params": {"max_new_tokens": budget, "temperature": 0.7, "top_p": 0.95},
                    }
                branches.append(row)
                payload = {
                    "complete": False,
                    "branch_count": len(branches),
                    "task_count": len(tasks),
                    "branches": branches,
                    "warnings": warnings,
                }
                write_json(OUT_PARTIAL, payload)
            if len(branches) >= MAX_TOTAL_BRANCHES or wall_time_exceeded():
                break
    finally:
        if generator is not None:
            generator.cleanup()

    by_task: dict[str, int] = defaultdict(int)
    for row in branches:
        if row.get("initial_partial_text") and not row.get("generation_error"):
            by_task[str(row["task_id"])] += 1
    tasks_3 = sum(1 for task in tasks if by_task[str(task["task_id"])] >= 3)
    tasks_2 = sum(1 for task in tasks if by_task[str(task["task_id"])] >= 2)
    if tasks and tasks_3 / len(tasks) >= 0.75:
        verdict = "READY"
    elif tasks and tasks_2 / len(tasks) >= 0.50:
        verdict = "PARTIAL"
    else:
        verdict = "BLOCKED"
    payload = {
        "BG_BRANCH_POOL_VERDICT": verdict,
        "verdict": verdict,
        "complete": True,
        "task_count": len(tasks),
        "branch_count": len(branches),
        "usable_branch_count": sum(1 for row in branches if row.get("initial_partial_text") and not row.get("generation_error")),
        "tasks_with_at_least_3_branches": tasks_3,
        "tasks_with_at_least_2_branches": tasks_2,
        "counts_by_domain": dict(Counter(row.get("domain") for row in branches)),
        "branches": branches,
        "warnings": warnings,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_PARTIAL, payload)
    write_json(OUT_JSON, payload)
    lines = [
        "# BG Shared Branch Pools (2026-05-18)",
        "",
        f"BG_BRANCH_POOL_VERDICT = {verdict}",
        "",
        f"- task_count: `{len(tasks)}`",
        f"- branch_count: `{len(branches)}`",
        f"- usable_branch_count: `{payload['usable_branch_count']}`",
        f"- tasks_with_at_least_3_branches: `{tasks_3}`",
        f"- counts_by_domain: `{payload['counts_by_domain']}`",
    ]
    if warnings:
        lines.extend(["", "## Warnings", *[f"- {w}" for w in warnings]])
    write_md(OUT_MD, lines)
    print(f"BG_BRANCH_POOL_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict in {"READY", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
