"""Build the bounded BG steering task suite."""
from __future__ import annotations

import time
from collections import Counter

from bg_steering_suite_lib import REPORT_ROOT, build_task_suite_rows, md_table, rel, write_json, write_md


OUT_JSON = REPORT_ROOT / "task_suite.json"
OUT_MD = REPORT_ROOT / "task_suite.md"


def main() -> int:
    started = time.time()
    tasks = build_task_suite_rows()
    counts = Counter(task["domain"] for task in tasks)
    code_count = counts.get("code", 0)
    non_code_objective = counts.get("reasoning", 0) + counts.get("science", 0) + counts.get("gsm8k", 0)
    if len(tasks) >= 40 and code_count >= 10 and non_code_objective >= 10:
        verdict = "READY"
    elif len(tasks) >= 20:
        verdict = "PARTIAL"
    else:
        verdict = "BLOCKED"
    payload = {
        "BG_STEERING_TASK_SUITE_VERDICT": verdict,
        "verdict": verdict,
        "task_count": len(tasks),
        "counts_by_domain": dict(counts),
        "target_task_count": 60,
        "hard_cap_task_count": 80,
        "tasks": tasks,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    table_rows = [
        {
            "idx": task["suite_index"],
            "task_id": task["task_id"],
            "domain": task["domain"],
            "source": task.get("source_dataset", task.get("source", "")),
            "difficulty": task.get("difficulty", ""),
            "devil": task.get("is_devil", False),
        }
        for task in tasks
    ]
    lines = [
        "# BG Steering Task Suite (2026-05-18)",
        "",
        f"BG_STEERING_TASK_SUITE_VERDICT = {verdict}",
        "",
        f"- task_count: `{len(tasks)}`",
        f"- counts_by_domain: `{dict(counts)}`",
        "",
        "## Tasks",
        "",
        *md_table(table_rows, ["idx", "task_id", "domain", "source", "difficulty", "devil"]),
    ]
    write_md(OUT_MD, lines)
    print(f"BG_STEERING_TASK_SUITE_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict in {"READY", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
