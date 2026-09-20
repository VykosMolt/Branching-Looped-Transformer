"""Build BG Stage 2 steering task suite from Stage 1 trajectory data."""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REPORT_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_stage2_steering_2026-05-18"
STAGE1_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_trajectory_prediction_2026-05-18"
OUT_JSON = REPORT_ROOT / "task_suite.json"
OUT_MD = REPORT_ROOT / "task_suite.md"

TARGETS = [
    {"target_id": "T1", "domain": "reasoning", "prefix_length": 64, "quota": 8},
    {"target_id": "T2", "domain": "reasoning", "prefix_length": 256, "quota": 8},
    {"target_id": "T3", "domain": "science", "prefix_length": 32, "quota": 7},
    {"target_id": "T4", "domain": "gsm8k", "prefix_length": 256, "quota": 7},
]


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(p)


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def task_expected_answer(task: dict[str, Any]) -> str:
    return str(task.get("answer_key") or task.get("gold_answer") or task.get("answer") or "")


def main() -> int:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    stage1_tasks_payload = load_json(STAGE1_ROOT / "task_suite.json", {})
    continued_payload = load_json(STAGE1_ROOT / "continued_prefixes.json", {})
    partials_payload = load_json(STAGE1_ROOT / "partials.json", {})
    stage1_tasks = list(stage1_tasks_payload.get("tasks") or [])
    by_task = {str(task["task_id"]): task for task in stage1_tasks}
    branch_rows = list(partials_payload.get("branches") or [])
    by_branch = {(str(row["task_id"]), int(row["branch_id"])): row for row in branch_rows}
    continued = [
        row
        for row in list(continued_payload.get("continued_prefixes") or [])
        if row.get("evaluable", True)
    ]
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in continued:
        grouped[(str(row.get("domain")), int(row.get("prefix_length", -1)), str(row.get("task_id")))].append(row)

    selected_rows: list[dict[str, Any]] = []
    target_counts: dict[str, int] = {}
    skipped_targets: dict[str, str] = {}
    for target in TARGETS:
        candidates = []
        for (domain, prefix_length, task_id), rows in grouped.items():
            if domain != target["domain"] or prefix_length != int(target["prefix_length"]):
                continue
            successes = [row for row in rows if bool(row.get("is_correct") or row.get("evaluation", {}).get("success"))]
            if not successes:
                continue
            successes.sort(key=lambda row: (not bool(row.get("is_correct")), int(row.get("branch_id", 0))))
            chosen = successes[0]
            candidates.append((task_id, chosen, len(successes), len(rows)))
        candidates.sort(key=lambda item: (str(item[0]), int(item[1].get("branch_id", 0))))
        if len(candidates) < 6:
            skipped_targets[str(target["target_id"])] = f"only {len(candidates)} oracle-success tasks at prefix {target['prefix_length']}"
        for task_id, chosen, success_count, branch_count in candidates[: int(target["quota"])]:
            task = dict(by_task.get(task_id) or {})
            if not task:
                continue
            branch = by_branch.get((task_id, int(chosen["branch_id"])), {})
            row = {
                "suite_index": len(selected_rows),
                "task_id": task_id,
                "domain": target["domain"],
                "target_assignment": target["target_id"],
                "prefix_length": int(target["prefix_length"]),
                "partial_trajectory_branch": int(chosen["branch_id"]),
                "prefix_text": str(chosen.get("prefix_text") or branch.get("prefixes", {}).get(f"prefix_{target['prefix_length']}") or ""),
                "stage1_continuation_text": str(chosen.get("continuation_text") or ""),
                "stage1_final_text": str(chosen.get("final_text") or ""),
                "stage1_is_correct": bool(chosen.get("is_correct") or chosen.get("evaluation", {}).get("success")),
                "stage1_parsed_answer": chosen.get("parsed_answer") or chosen.get("evaluation", {}).get("parsed_answer"),
                "oracle_successful_branches_at_prefix": int(success_count),
                "stage1_branch_count_at_prefix": int(branch_count),
                "expected_answer": task_expected_answer(task),
                "evaluator_info": {
                    "evaluator_type": task.get("evaluator_type"),
                    "expected_answer_format": task.get("expected_answer_format"),
                    "options": task.get("options"),
                },
                "task": task,
            }
            selected_rows.append(row)
        target_counts[str(target["target_id"])] = sum(1 for row in selected_rows if row["target_assignment"] == target["target_id"])

    if all(count >= 6 for count in target_counts.values()) and len(target_counts) == 4:
        verdict = "READY"
    elif sum(1 for count in target_counts.values() if count >= 6) >= 2:
        verdict = "PARTIAL"
    else:
        verdict = "BLOCKED"

    payload = {
        "BG_STAGE2_TASK_SUITE_VERDICT": verdict,
        "verdict": verdict,
        "task_count": len(selected_rows),
        "unique_task_count": len({row["task_id"] for row in selected_rows}),
        "target_counts": target_counts,
        "counts_by_domain": dict(Counter(row["domain"] for row in selected_rows)),
        "hard_cap_tasks": 30,
        "selection_rule": "Stage 1 tasks with at least one oracle-success branch at the target prefix",
        "skipped_targets": skipped_targets,
        "tasks": selected_rows[:30],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)

    lines = [
        "# BG Stage 2 Steering Task Suite (2026-05-18)",
        "",
        f"BG_STAGE2_TASK_SUITE_VERDICT = {verdict}",
        "",
        f"- task_count: `{payload['task_count']}`",
        f"- unique_task_count: `{payload['unique_task_count']}`",
        f"- target_counts: `{target_counts}`",
        f"- counts_by_domain: `{payload['counts_by_domain']}`",
        "",
        "## Tasks",
        "",
        "| idx | target | domain | prefix | task_id | branch | expected | oracle branches |",
        "|---:|---|---|---:|---|---:|---|---:|",
    ]
    for row in payload["tasks"]:
        lines.append(
            f"| {row['suite_index']} | {row['target_assignment']} | {row['domain']} | {row['prefix_length']} | "
            f"`{row['task_id']}` | {row['partial_trajectory_branch']} | `{row['expected_answer']}` | "
            f"{row['oracle_successful_branches_at_prefix']} |"
        )
    if skipped_targets:
        lines.extend(["", "## Skipped Or Thin Targets", ""])
        for target_id, reason in skipped_targets.items():
            lines.append(f"- {target_id}: {reason}")
    write_md(OUT_MD, lines)
    print(f"BG_STAGE2_TASK_SUITE_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict in {"READY", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
