"""Generate shared full trajectories and token-prefix checkpoints."""
from __future__ import annotations

import time
import traceback
from collections import Counter, defaultdict

from bg_trajectory_prediction_lib import (
    BRANCHES_PER_TASK,
    MAX_NEW_TOKENS,
    PREFIX_LENGTHS,
    REPORT_ROOT,
    SEED,
    OuroTextGenerator,
    load_json,
    load_task_suite,
    make_prefixes,
    rel,
    trajectory_generation_prompt,
    write_json,
    write_md,
)


OUT_JSON = REPORT_ROOT / "partials.json"
OUT_PARTIAL = REPORT_ROOT / "partials.partial.json"
OUT_MD = REPORT_ROOT / "partials.md"


def main() -> int:
    started = time.time()
    tasks = load_task_suite()
    existing_payload = load_json(OUT_PARTIAL, {})
    branches = list(existing_payload.get("branches") or [])
    existing = {(str(row.get("task_id")), int(row.get("branch_id", -1))) for row in branches}
    warnings: list[str] = []
    generator: OuroTextGenerator | None = None

    try:
        generator = OuroTextGenerator(device="cuda")
        for task in tasks:
            task_id = str(task["task_id"])
            task_count = sum(1 for row in branches if str(row.get("task_id")) == task_id and row.get("full_generated_text"))
            if task_count >= BRANCHES_PER_TASK:
                continue
            prompt = trajectory_generation_prompt(task)
            for branch_id in range(BRANCHES_PER_TASK):
                key = (task_id, branch_id)
                if key in existing:
                    continue
                seed = SEED + int(task.get("suite_index", 0)) * 1009 + branch_id
                try:
                    gen = generator.generate(
                        prompt,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=0.7,
                        top_p=0.95,
                        seed=seed,
                    )
                    prefixes, prefix_counts, checkpoint_missing = make_prefixes(generator.tokenizer, gen["text"])
                    row = {
                        "task_id": task_id,
                        "domain": task["domain"],
                        "branch_id": branch_id,
                        "full_generated_text": gen["text"],
                        "raw_text": gen["raw_text"],
                        "prefixes": prefixes,
                        "prefix_token_counts": prefix_counts,
                        "checkpoint_missing": checkpoint_missing,
                        "token_count": gen["token_count"],
                        "hit_max_tokens": gen["hit_max_tokens"],
                        "generation_error": gen["generation_error"],
                        "generation_seconds": gen["seconds"],
                        "generation_params": {
                            "max_new_tokens": MAX_NEW_TOKENS,
                            "temperature": 0.7,
                            "top_p": 0.95,
                            "seed": seed,
                        },
                    }
                except Exception as exc:
                    row = {
                        "task_id": task_id,
                        "domain": task["domain"],
                        "branch_id": branch_id,
                        "full_generated_text": "",
                        "raw_text": "",
                        "prefixes": {},
                        "prefix_token_counts": {},
                        "checkpoint_missing": {str(p): True for p in PREFIX_LENGTHS},
                        "token_count": 0,
                        "hit_max_tokens": False,
                        "generation_error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:],
                        "generation_params": {
                            "max_new_tokens": MAX_NEW_TOKENS,
                            "temperature": 0.7,
                            "top_p": 0.95,
                            "seed": seed,
                        },
                    }
                branches.append(row)
                existing.add(key)
                write_json(
                    OUT_PARTIAL,
                    {
                        "complete": False,
                        "task_count": len(tasks),
                        "branch_count": len(branches),
                        "branches": branches,
                        "warnings": warnings,
                    },
                )
    finally:
        if generator is not None:
            generator.cleanup()

    by_task = defaultdict(list)
    usable_prefix_counts = []
    for row in branches:
        if row.get("full_generated_text") and not row.get("generation_error"):
            by_task[str(row["task_id"])].append(row)
            usable_prefix_counts.append(sum(1 for p in PREFIX_LENGTHS if row.get("prefixes", {}).get(f"prefix_{p}")))
    tasks_with_4 = sum(1 for task in tasks if len(by_task[str(task["task_id"])]) >= 4)
    tasks_with_2 = sum(1 for task in tasks if len(by_task[str(task["task_id"])]) >= 2)
    tasks_with_3_prefixes = sum(
        1
        for task in tasks
        if len(by_task[str(task["task_id"])]) >= 4
        and all(sum(1 for p in PREFIX_LENGTHS if row.get("prefixes", {}).get(f"prefix_{p}")) >= 3 for row in by_task[str(task["task_id"])])
    )
    if tasks and tasks_with_4 / len(tasks) >= 0.75 and tasks_with_3_prefixes / len(tasks) >= 0.75:
        verdict = "READY"
    elif tasks and tasks_with_2 / len(tasks) >= 0.50:
        verdict = "PARTIAL"
    else:
        verdict = "BLOCKED"
    payload = {
        "BG_TRAJECTORY_PARTIALS_VERDICT": verdict,
        "verdict": verdict,
        "complete": True,
        "task_count": len(tasks),
        "branch_count": len(branches),
        "usable_branch_count": sum(len(v) for v in by_task.values()),
        "tasks_with_4_branches": tasks_with_4,
        "tasks_with_2_branches": tasks_with_2,
        "tasks_with_4_branches_and_3_prefixes": tasks_with_3_prefixes,
        "counts_by_domain": dict(Counter(row.get("domain") for row in branches)),
        "prefix_lengths": list(PREFIX_LENGTHS),
        "branches": branches,
        "warnings": warnings,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_PARTIAL, payload)
    write_json(OUT_JSON, payload)
    lines = [
        "# BG Trajectory Partials (2026-05-18)",
        "",
        f"BG_TRAJECTORY_PARTIALS_VERDICT = {verdict}",
        "",
        f"- task_count: `{len(tasks)}`",
        f"- branch_count: `{len(branches)}`",
        f"- usable_branch_count: `{payload['usable_branch_count']}`",
        f"- tasks_with_4_branches: `{tasks_with_4}`",
        f"- tasks_with_4_branches_and_3_prefixes: `{tasks_with_3_prefixes}`",
        f"- prefix_lengths: `{list(PREFIX_LENGTHS)}`",
    ]
    if warnings:
        lines.extend(["", "## Warnings", "", *[f"- {w}" for w in warnings]])
    write_md(OUT_MD, lines)
    print(f"BG_TRAJECTORY_PARTIALS_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict in {"READY", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
