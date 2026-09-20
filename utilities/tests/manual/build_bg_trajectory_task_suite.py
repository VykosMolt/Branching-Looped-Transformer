"""Build the Stage 1 BG trajectory prediction task suite."""
from __future__ import annotations

import time
from collections import Counter

from bg_trajectory_prediction_lib import REPORT_ROOT, build_trajectory_task_rows, md_table, rel, write_json, write_md


OUT_JSON = REPORT_ROOT / "task_suite.json"
OUT_MD = REPORT_ROOT / "task_suite.md"


def main() -> int:
    started = time.time()
    tasks = build_trajectory_task_rows()
    counts = Counter(task["domain"] for task in tasks)
    if len(tasks) >= 45 and sum(1 for count in counts.values() if count >= 10) >= 2:
        verdict = "READY"
    elif len(tasks) >= 20:
        verdict = "PARTIAL"
    else:
        verdict = "BLOCKED"
    payload = {
        "BG_TRAJECTORY_TASK_SUITE_VERDICT": verdict,
        "verdict": verdict,
        "task_count": len(tasks),
        "counts_by_domain": dict(counts),
        "target_counts": {"reasoning": 20, "science": 20, "gsm8k": 20},
        "minimum_total_tasks": 30,
        "hard_cap_tasks": 75,
        "tasks": tasks,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    rows = [
        {
            "idx": task["suite_index"],
            "task_id": task["task_id"],
            "domain": task["domain"],
            "source": task.get("source_dataset", ""),
            "difficulty": task.get("difficulty", ""),
            "evaluator": task.get("evaluator_type", ""),
        }
        for task in tasks
    ]
    lines = [
        "# BG Trajectory Prediction Task Suite (2026-05-18)",
        "",
        f"BG_TRAJECTORY_TASK_SUITE_VERDICT = {verdict}",
        "",
        f"- task_count: `{len(tasks)}`",
        f"- counts_by_domain: `{dict(counts)}`",
        "",
        "## Tasks",
        "",
        *md_table(rows, ["idx", "task_id", "domain", "source", "difficulty", "evaluator"]),
    ]
    write_md(OUT_MD, lines)
    print(f"BG_TRAJECTORY_TASK_SUITE_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict in {"READY", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
