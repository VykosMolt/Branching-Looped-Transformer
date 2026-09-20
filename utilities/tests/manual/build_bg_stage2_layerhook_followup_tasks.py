"""Build a small T1 reasoning@64 task subset for layer-hook follow-up."""
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


OUT_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_stage2_layerhook_followup_2026-05-18"
STAGE1_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_trajectory_prediction_2026-05-18"
STAGE2_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_stage2_steering_2026-05-18"
OUT_JSON = OUT_ROOT / "task_subset.json"
OUT_MD = OUT_ROOT / "task_subset.md"


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


def expected_answer(task: dict[str, Any]) -> str:
    return str(task.get("answer_key") or task.get("gold_answer") or task.get("answer") or "")


def main() -> int:
    started = time.time()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    task_suite = load_json(STAGE1_ROOT / "task_suite.json", {})
    continued = load_json(STAGE1_ROOT / "continued_prefixes.json", {})
    partials = load_json(STAGE1_ROOT / "partials.json", {})
    prior = load_json(STAGE2_ROOT / "intervention_traces.partial.json", {})

    stage1_tasks = {str(task["task_id"]): task for task in task_suite.get("tasks") or []}
    branches = {
        (str(row.get("task_id")), int(row.get("branch_id", -1))): row
        for row in partials.get("branches") or []
    }
    prior_counts = Counter(
        str(row.get("task_id"))
        for row in prior.get("rows") or []
        if row.get("target_id") == "T1" and row.get("mechanism") == "layer_hook_injection"
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in continued.get("continued_prefixes") or []:
        if str(row.get("domain")) != "reasoning" or int(row.get("prefix_length", -1)) != 64:
            continue
        if not row.get("evaluable", True):
            continue
        grouped[str(row.get("task_id"))].append(row)

    candidates = []
    for task_id, rows in grouped.items():
        task = stage1_tasks.get(task_id)
        if not task or not expected_answer(task):
            continue
        successes = [row for row in rows if bool(row.get("is_correct") or row.get("evaluation", {}).get("success"))]
        if not successes:
            continue
        successes.sort(key=lambda row: int(row.get("branch_id", 0)))
        chosen = successes[0]
        branch_id = int(chosen.get("branch_id", 0))
        branch = branches.get((task_id, branch_id), {})
        prefix_text = str(chosen.get("prefix_text") or branch.get("prefixes", {}).get("prefix_64") or "")
        if not prefix_text.strip():
            continue
        candidates.append(
            {
                "task_id": task_id,
                "domain": "reasoning",
                "prefix_length": 64,
                "prefix_text": prefix_text,
                "prompt": str(task.get("prompt") or task.get("question") or ""),
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
                "existing_partial_rows": int(prior_counts.get(task_id, 0)),
                "existing_partial_covered": bool(prior_counts.get(task_id, 0)),
                "task": task,
            }
        )
    candidates.sort(key=lambda row: (int(row["existing_partial_rows"]), str(row["task_id"])))
    selected = candidates[:8]
    for idx, row in enumerate(selected):
        row["suite_index"] = idx

    if len(selected) >= 6:
        verdict = "READY"
    elif len(selected) >= 3:
        verdict = "PARTIAL"
    else:
        verdict = "BLOCKED"
    payload = {
        "BG_LAYERHOOK_FOLLOWUP_TASKS_VERDICT": verdict,
        "verdict": verdict,
        "task_count": len(selected),
        "candidate_count": len(candidates),
        "target_task_count": 8,
        "minimum_task_count": 6,
        "hard_cap_task_count": 10,
        "selection_rule": "T1 reasoning@64 tasks with at least one Stage 1 oracle-success branch and available 64-token prefix",
        "tasks": selected,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)

    lines = [
        "# BG Stage 2 Layer-Hook Follow-Up Task Subset",
        "",
        f"BG_LAYERHOOK_FOLLOWUP_TASKS_VERDICT = {verdict}",
        "",
        f"- task_count: `{len(selected)}`",
        f"- candidate_count: `{len(candidates)}`",
        "",
        "| idx | task_id | branch | expected | oracle branches | existing partial rows |",
        "|---:|---|---:|---|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['suite_index']} | `{row['task_id']}` | {row['branch_id']} | `{row['expected_answer']}` | "
            f"{row['stage1_oracle_successful_branches']} | {row['existing_partial_rows']} |"
        )
    write_md(OUT_MD, lines)
    print(f"BG_LAYERHOOK_FOLLOWUP_TASKS_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict in {"READY", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
