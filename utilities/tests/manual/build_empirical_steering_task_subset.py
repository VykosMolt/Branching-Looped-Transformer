"""Build held-out task subset for empirical BG steering directions."""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import torch

from bg_empirical_steering_common import (
    OUT_ROOT,
    STAGE1_ROOT,
    expected_answer,
    load_continuation_index,
    load_json,
    load_partials_branch_index,
    load_stage1_tasks,
    rel,
    write_json,
    write_md,
)


OUT_JSON = OUT_ROOT / "task_subset.json"
OUT_MD = OUT_ROOT / "task_subset.md"
DIRECTIONS_PT = OUT_ROOT / "directions.pt"


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if not DIRECTIONS_PT.exists():
        payload = {"BG_EMPIRICAL_STEERING_TASKS_VERDICT": "BLOCKED", "blocker": "directions.pt missing", "tasks": []}
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# BG Empirical Steering Task Subset", "", "BG_EMPIRICAL_STEERING_TASKS_VERDICT = BLOCKED"])
        print("BG_EMPIRICAL_STEERING_TASKS_VERDICT = BLOCKED")
        return 1
    directions = torch.load(DIRECTIONS_PT, map_location="cpu", weights_only=False)
    primary = next((row for row in directions.get("targets", []) if row.get("target_id") == "T1"), None)
    if not primary:
        payload = {"BG_EMPIRICAL_STEERING_TASKS_VERDICT": "BLOCKED", "blocker": "primary direction target missing", "tasks": []}
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# BG Empirical Steering Task Subset", "", "BG_EMPIRICAL_STEERING_TASKS_VERDICT = BLOCKED"])
        print("BG_EMPIRICAL_STEERING_TASKS_VERDICT = BLOCKED")
        return 1

    stage1_tasks = load_stage1_tasks()
    branch_index = load_partials_branch_index()
    continued = load_continuation_index()
    train_task_ids = set(primary.get("train_task_ids") or [])
    heldout_task_ids = list(primary.get("heldout_task_ids") or [])
    domain = str(primary["domain"])
    prefix_length = int(primary["prefix_length"])

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (task_id, _branch_id, pfx), row in continued.items():
        if int(pfx) != prefix_length or str(row.get("domain")) != domain:
            continue
        if not row.get("evaluable", True):
            continue
        grouped[str(task_id)].append(row)

    candidates = []
    for task_id in heldout_task_ids + sorted(set(grouped) - set(heldout_task_ids)):
        rows = grouped.get(task_id) or []
        task = stage1_tasks.get(task_id)
        if not task or not expected_answer(task):
            continue
        successes = [row for row in rows if bool(row.get("is_correct") or row.get("evaluation", {}).get("success"))]
        if not successes:
            continue
        successes.sort(key=lambda row: int(row.get("branch_id", 0)))
        chosen = successes[0]
        branch_id = int(chosen.get("branch_id", 0))
        branch = branch_index.get((task_id, branch_id), {})
        prefix_text = str(chosen.get("prefix_text") or branch.get("prefixes", {}).get(f"prefix_{prefix_length}") or "")
        if not prefix_text.strip():
            continue
        used_in_fit = task_id in train_task_ids
        candidates.append(
            {
                "task_id": task_id,
                "domain": domain,
                "target_id": "T1",
                "prefix_length": prefix_length,
                "prompt": str(task.get("prompt") or task.get("question") or ""),
                "prefix_text": prefix_text,
                "expected_answer": expected_answer(task),
                "evaluator_info": {
                    "evaluator_type": task.get("evaluator_type"),
                    "expected_answer_format": task.get("expected_answer_format"),
                    "options": task.get("options"),
                },
                "branch_id": branch_id,
                "stage1_oracle_successful_branches": len(successes),
                "stage1_branch_count": len(rows),
                "stage1_continuation_text": str(chosen.get("continuation_text") or ""),
                "stage1_final_text": str(chosen.get("final_text") or ""),
                "stage1_is_correct": bool(chosen.get("is_correct") or chosen.get("evaluation", {}).get("success")),
                "used_in_direction_fitting": used_in_fit,
                "task": task,
            }
        )
    candidates.sort(key=lambda row: (row["used_in_direction_fitting"], str(row["task_id"])))
    selected = candidates[:8]
    for idx, row in enumerate(selected):
        row["suite_index"] = idx
    overlap = any(row["used_in_direction_fitting"] for row in selected)
    if len(selected) >= 6:
        verdict = "READY"
    elif len(selected) >= 3:
        verdict = "PARTIAL"
    else:
        verdict = "BLOCKED"
    payload = {
        "BG_EMPIRICAL_STEERING_TASKS_VERDICT": verdict,
        "verdict": verdict,
        "task_count": len(selected),
        "candidate_count": len(candidates),
        "target_task_count": 8,
        "minimum_task_count": 6,
        "TASK_OVERLAP_WARNING": overlap,
        "selection_rule": "held-out T1 reasoning@64 tasks with oracle-success branch and available prefix_text",
        "primary_target": {
            "target_id": "T1",
            "domain": domain,
            "prefix_length": prefix_length,
            "heldout_task_ids": heldout_task_ids,
            "train_task_count": len(train_task_ids),
        },
        "secondary_targets_skipped": "runtime budget preserved for primary target direction comparison",
        "tasks": selected,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    lines = [
        "# BG Empirical Steering Task Subset",
        "",
        f"BG_EMPIRICAL_STEERING_TASKS_VERDICT = {verdict}",
        "",
        f"- task_count: `{len(selected)}`",
        f"- candidate_count: `{len(candidates)}`",
        f"- TASK_OVERLAP_WARNING: `{overlap}`",
        "",
        "| idx | task_id | branch | expected | used in fit | oracle branches |",
        "|---:|---|---:|---|---|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['suite_index']} | `{row['task_id']}` | {row['branch_id']} | `{row['expected_answer']}` | "
            f"`{row['used_in_direction_fitting']}` | {row['stage1_oracle_successful_branches']} |"
        )
    write_md(OUT_MD, lines)
    print(f"BG_EMPIRICAL_STEERING_TASKS_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict in {"READY", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
