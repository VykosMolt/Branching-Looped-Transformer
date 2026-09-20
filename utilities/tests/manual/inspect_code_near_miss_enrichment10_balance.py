"""Inspect the completed near-miss enrichment10 run for balancing targets."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from code_branch_pilot_lib import REPORT_DIR, repo_path, write_json


TASKSET_JSON = REPORT_DIR / "code_branch_taskset_v2_near_miss10_2026-05-17.json"
CANDIDATES_JSON = REPORT_DIR / "code_branch_candidates_v2_near_miss10_2026-05-17.json"
TOURNAMENTS_JSON = REPORT_DIR / "code_branch_tournaments_v2_near_miss10_2026-05-17.json"
OUTPUT_JSON = REPORT_DIR / "code_branch_near_miss_balance_inspection_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "code_branch_near_miss_balance_inspection_2026-05-17.md"

LABELS = ("correct", "near_miss", "wrong_code", "runtime_error", "malformed", "safety_rejected")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def bucket_for(counts: Counter[str]) -> tuple[str, str]:
    has_correct = counts["correct"] > 0
    has_near = counts["near_miss"] > 0
    has_wrong = counts["wrong_code"] > 0
    has_runtime = counts["runtime_error"] > 0
    if has_correct and has_near:
        return "already_strict_clean", "already_has_correct_and_near_miss"
    if has_near and not has_correct:
        return "needs_correct_anchor", "has_near_miss_missing_correct"
    if has_correct and not has_near and has_wrong:
        return "needs_near_miss", "correct_plus_wrong_code_no_near_miss"
    if counts["correct"] >= 2 and not has_near and not has_wrong:
        return "all_correct_collapse", "multiple_correct_no_incorrect_contrast"
    if has_correct and not has_near:
        return "needs_near_miss", "has_correct_missing_near_miss"
    if not has_correct and not has_near and (has_wrong or has_runtime):
        return "all_wrong_collapse", "wrong_or_runtime_only"
    if not has_correct and not has_near:
        return "needs_correct_and_near_miss", "no_correct_or_near_miss"
    return "needs_correct_and_near_miss", "fallback"


def priority_for(row: dict[str, Any]) -> int:
    bucket = row["bucket"]
    sub = row["subbucket"]
    if bucket == "needs_correct_anchor":
        return 0
    if bucket == "needs_near_miss" and sub != "correct_plus_wrong_code_no_near_miss":
        return 1
    if bucket == "needs_near_miss" and sub == "correct_plus_wrong_code_no_near_miss":
        return 2
    if bucket == "all_wrong_collapse":
        return 3
    if bucket == "all_correct_collapse":
        return 4
    if bucket == "needs_correct_and_near_miss":
        return 5
    return 9


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Code Branch Near-Miss Balance Inspection",
        "",
        f"BALANCE_INSPECTION_VERDICT = {payload['balance_inspection_verdict']}",
        "",
        f"- tasks: `{payload['summary']['tasks']}`",
        f"- existing_strict_clean: `{payload['summary']['existing_strict_clean']}`",
        f"- plausible_target_tasks: `{payload['summary']['plausible_target_tasks']}`",
        f"- bucket_counts: `{payload['summary']['bucket_counts']}`",
        "",
        "## Tasks",
        "",
        "| task_id | source | difficulty | candidates | correct | near_miss | wrong_code | runtime | malformed | safety | strict | diagnostic | bucket | subbucket |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in payload["tasks"]:
        lines.append(
            f"| `{row['task_id']}` | `{row['source']}` | `{row['difficulty']}` | {row['n_candidates']} | "
            f"{row['n_correct']} | {row['n_near_miss']} | {row['n_wrong_code']} | {row['n_runtime_error']} | "
            f"{row['n_malformed']} | {row['n_safety_rejected']} | {row['strict_clean']} | "
            f"{row['diagnostic_runnable']} | `{row['bucket']}` | `{row['subbucket']}` |"
        )
    lines.extend(["", "## Ranked Targets", ""])
    for row in payload["ranked_targets"]:
        lines.append(
            f"- `{row['task_id']}` priority={row['priority']} bucket=`{row['bucket']}` "
            f"labels={row['label_counts']}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    taskset = load(TASKSET_JSON)
    candidates = load(CANDIDATES_JSON)
    tournaments = load(TOURNAMENTS_JSON)
    rows = list(tournaments.get("candidate_evaluations", []))
    candidates_by_task = Counter(str(c.get("task_id", "")) for c in candidates.get("candidates", []))
    rows_by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_task.setdefault(str(row.get("task_id", "")), []).append(row)

    task_rows: list[dict[str, Any]] = []
    for task in taskset.get("tasks", []):
        task_id = str(task["task_id"])
        eval_rows = rows_by_task.get(task_id, [])
        counts = Counter(str(row.get("unit_test_label", "malformed")) for row in eval_rows)
        for label in LABELS:
            counts.setdefault(label, 0)
        has_correct = counts["correct"] > 0
        has_near = counts["near_miss"] > 0
        has_wrong = counts["wrong_code"] > 0
        runnable = sum(1 for row in eval_rows if row.get("is_runnable"))
        bucket, subbucket = bucket_for(counts)
        strict_clean = bool(has_correct and has_near and runnable >= 2)
        diagnostic_runnable = bool(has_correct and (has_near or has_wrong))
        task_rows.append({
            "task_id": task_id,
            "source": task.get("source", "unknown"),
            "difficulty": task.get("difficulty", task.get("difficulty_label", "unknown")),
            "n_candidates": int(candidates_by_task.get(task_id, len(eval_rows))),
            "n_evaluated": len(eval_rows),
            "n_correct": counts["correct"],
            "n_near_miss": counts["near_miss"],
            "n_wrong_code": counts["wrong_code"],
            "n_runtime_error": counts["runtime_error"],
            "n_malformed": counts["malformed"],
            "n_safety_rejected": counts["safety_rejected"],
            "has_correct": has_correct,
            "has_near_miss": has_near,
            "has_wrong_code": has_wrong,
            "strict_clean": strict_clean,
            "diagnostic_runnable": diagnostic_runnable,
            "bucket": bucket,
            "subbucket": subbucket,
            "label_counts": {label: counts[label] for label in LABELS if counts[label]},
        })

    ranked = [
        {**row, "priority": priority_for(row)}
        for row in task_rows
        if row["bucket"] != "already_strict_clean"
    ]
    ranked.sort(key=lambda row: (row["priority"], -row["n_near_miss"], -row["n_correct"], row["task_id"]))
    plausible = [row for row in ranked if row["priority"] <= 4]
    verdict = "READY" if len(plausible) >= 3 else "BLOCKED"
    payload = {
        "balance_inspection_verdict": verdict,
        "inputs": {
            "taskset": repo_path(TASKSET_JSON),
            "candidates": repo_path(CANDIDATES_JSON),
            "tournaments": repo_path(TOURNAMENTS_JSON),
        },
        "summary": {
            "tasks": len(task_rows),
            "existing_strict_clean": sum(1 for row in task_rows if row["strict_clean"]),
            "existing_diagnostic_runnable": sum(1 for row in task_rows if row["diagnostic_runnable"]),
            "plausible_target_tasks": len(plausible),
            "bucket_counts": dict(Counter(row["bucket"] for row in task_rows)),
        },
        "tasks": task_rows,
        "ranked_targets": ranked,
        "outputs": {"json": repo_path(OUTPUT_JSON), "md": repo_path(OUTPUT_MD)},
    }
    write_json(OUTPUT_JSON, payload)
    write_md(OUTPUT_MD, payload)
    print(f"BALANCE_INSPECTION_VERDICT = {verdict}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    if verdict == "BLOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
