"""Evaluate old+new near-miss balancing candidates and rebuild tournaments."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from code_branch_pilot_lib import REPORT_DIR, repo_path, write_json
from evaluate_code_branch_candidates_v2 import (
    candidate_code_for_unit_tests,
    eval_candidate,
    label_from_eval,
    legacy_label_from_eval,
)


TASKSET_JSON = REPORT_DIR / "code_branch_taskset_v2_near_miss10_2026-05-17.json"
OLD_TOURNAMENTS_JSON = REPORT_DIR / "code_branch_tournaments_v2_near_miss10_2026-05-17.json"
INSPECTION_JSON = REPORT_DIR / "code_branch_near_miss_balance_inspection_2026-05-17.json"
BALANCING_CANDIDATES_JSON = REPORT_DIR / "code_branch_near_miss_balancing_candidates_2026-05-17.json"
OUTPUT_JSON = REPORT_DIR / "code_branch_near_miss_balanced_tournaments_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "code_branch_near_miss_balanced_tournaments_2026-05-17.md"

LABELS = ("correct", "near_miss", "wrong_code", "runtime_error", "malformed", "safety_rejected")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def tests_for(task: dict[str, Any]) -> list[str]:
    return list(task.get("tests") or list(task.get("public_tests", [])) + list(task.get("hidden_tests", [])))


def random_baseline(tournaments: list[dict[str, Any]], candidate_key: str) -> float:
    if not tournaments:
        return float("nan")
    return mean(
        sum(1 for cand in t[candidate_key] if cand["is_correct"]) / max(len(t[candidate_key]), 1)
        for t in tournaments
    )


def build_tournaments(rows: list[dict[str, Any]], tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_task[str(row.get("task_id", ""))].append(row)
    tournaments: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task["task_id"])
        cand_rows = rows_by_task.get(task_id, [])
        correct = [row for row in cand_rows if row["unit_test_label"] == "correct"]
        near = [row for row in cand_rows if row["unit_test_label"] == "near_miss"]
        wrong = [row for row in cand_rows if row["unit_test_label"] == "wrong_code"]
        incorrect = [row for row in cand_rows if row["unit_test_label"] != "correct"]
        runnable = [row for row in cand_rows if row.get("is_runnable")]
        strict_candidates = [row for row in cand_rows if row["unit_test_label"] in {"correct", "near_miss"}]
        diagnostic_runnable_candidates = [
            row for row in cand_rows if row["unit_test_label"] in {"correct", "near_miss", "wrong_code"}
        ]
        diagnostic_mixed_primary_candidates = [
            row for row in cand_rows if row["unit_test_label"] not in {"malformed", "safety_rejected"}
        ]
        strict_clean = bool(correct and near and len(runnable) >= 2)
        diagnostic_runnable = bool(correct and (near or wrong) and len(diagnostic_runnable_candidates) >= 2)
        diagnostic_mixed = bool(correct and incorrect)
        tournaments.append({
            "tournament_id": len(tournaments),
            "task_id": task_id,
            "source": task.get("source", "unknown"),
            "difficulty": task.get("difficulty", task.get("difficulty_label", "unknown")),
            "prompt": task["prompt"],
            "function_name": task["function_name"],
            "strict_clean": strict_clean,
            "diagnostic_runnable": diagnostic_runnable,
            "diagnostic_mixed": diagnostic_mixed,
            "too_easy": bool(len(cand_rows) >= 2 and len(correct) == len(cand_rows)),
            "too_hard": bool(cand_rows and not correct),
            "strict_candidates": strict_candidates,
            "diagnostic_runnable_candidates": diagnostic_runnable_candidates,
            "diagnostic_mixed_primary_candidates": diagnostic_mixed_primary_candidates,
            "diagnostic_candidates": cand_rows,
            "label_counts": dict(Counter(row["unit_test_label"] for row in cand_rows)),
        })
    return tournaments


def verdict_for(strict: int, diagnostic: int) -> str:
    if strict >= 5:
        return "GREEN"
    if strict in {3, 4} or (diagnostic >= 8 and strict >= 3):
        return "YELLOW"
    return "RED"


def primary_for(verdict: str) -> str:
    if verdict == "GREEN":
        return "strict_clean"
    if verdict == "YELLOW":
        return "strict_plus_diagnostic_runnable"
    return "none"


def compact_counts(summary: dict[str, Any]) -> dict[str, int]:
    return {
        "correct": int(summary.get("correct_candidates", 0)),
        "near_miss": int(summary.get("near_miss_candidates", 0)),
        "wrong_code": int(summary.get("wrong_code_candidates", 0)),
        "runtime_error": int(summary.get("runtime_error_candidates", 0)),
        "malformed": int(summary.get("malformed_candidates", 0)),
        "safety_rejected": int(summary.get("safety_rejected_candidates", 0)),
    }


def converted_tasks(
    before: dict[str, dict[str, Any]],
    tournaments: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out = []
    for t in tournaments:
        old = before.get(t["task_id"], {})
        if old.get("strict_clean") or not t["strict_clean"]:
            continue
        added = [
            row for row in rows
            if row.get("task_id") == t["task_id"] and row.get("generation_source") == "balancing_pass"
        ]
        out.append({
            "task_id": t["task_id"],
            "old_bucket": old.get("bucket", "unknown"),
            "new_bucket": "already_strict_clean",
            "helpful_modes": sorted({str(row.get("mode")) for row in added if row.get("unit_test_label") in {"correct", "near_miss"}}),
            "correct_added": any(row.get("unit_test_label") == "correct" for row in added),
            "near_miss_added": any(row.get("unit_test_label") == "near_miss" for row in added),
            "strict_clean_now": True,
        })
    return out


def write_md(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# Code Branch Near-Miss Balanced Tournaments",
        "",
        f"BALANCED_TOURNAMENT_VERDICT = {payload['balanced_tournament_verdict']}",
        "",
        f"- primary_eval_set: `{payload['primary_eval_set']}`",
        f"- before_strict_clean: `{s['before']['strict_clean']}`",
        f"- after_strict_clean: `{s['after']['strict_clean']}`",
        f"- before_label_counts: `{s['before']['label_counts']}`",
        f"- after_label_counts: `{s['after']['label_counts']}`",
        f"- diagnostic_runnable: `{s['after']['diagnostic_runnable']}`",
        f"- diagnostic_mixed: `{s['after']['diagnostic_mixed']}`",
        f"- random_top1_baselines: `{s['after']['random_top1_baselines']}`",
        "",
        "## Converted Tasks",
        "",
    ]
    if payload["converted_tasks"]:
        for row in payload["converted_tasks"]:
            lines.append(
                f"- `{row['task_id']}` old=`{row['old_bucket']}` modes=`{row['helpful_modes']}` "
                f"correct_added={row['correct_added']} near_miss_added={row['near_miss_added']}"
            )
    else:
        lines.append("- None")
    lines.extend([
        "",
        "## Tournaments",
        "",
        "| task_id | strict | diagnostic | labels | new labels |",
        "| --- | ---: | ---: | --- | --- |",
    ])
    for t in payload["tournaments"]:
        new_labels = Counter(
            row["unit_test_label"]
            for row in t["diagnostic_candidates"]
            if row.get("generation_source") == "balancing_pass"
        )
        lines.append(
            f"| `{t['task_id']}` | {t['strict_clean']} | {t['diagnostic_runnable']} | "
            f"`{t['label_counts']}` | `{dict(new_labels)}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    taskset = load(TASKSET_JSON)
    old_tournaments = load(OLD_TOURNAMENTS_JSON)
    inspection = load(INSPECTION_JSON)
    candidate_payload = load(BALANCING_CANDIDATES_JSON)
    tasks = list(taskset.get("tasks", []))
    task_by_id = {str(task["task_id"]): task for task in tasks}
    rows = []
    for cand in candidate_payload.get("candidates", []):
        task = task_by_id.get(str(cand.get("task_id", "")))
        if not task:
            continue
        eval_code, recovered = candidate_code_for_unit_tests(cand, task)
        eval_result = eval_candidate(
            eval_code,
            tests_for(task),
            str(task.get("function_name", "")),
            float(task.get("timeout_seconds", 5.0)),
        )
        label = label_from_eval(eval_result, eval_code, str(task.get("function_name", "")))
        row = {
            **cand,
            "raw_final_code": cand.get("final_code", ""),
            "final_code": eval_code,
            "candidate_code_recovered_by_evaluator": recovered,
            **eval_result,
            "unit_test_label": label,
            "legacy_unit_test_label": legacy_label_from_eval(eval_result),
            "is_malformed": label == "malformed",
            "is_code_like_wrong": label in {"wrong_code", "runtime_error", "near_miss"},
            "is_correct": label == "correct",
            "is_runnable": bool(eval_result.get("safety_ok") and eval_result.get("syntax_ok") and eval_result.get("import_ok") and eval_result.get("runtime_ok")),
        }
        rows.append(row)

    tournaments = build_tournaments(rows, tasks)
    strict = [t for t in tournaments if t["strict_clean"]]
    diagnostic_runnable = [t for t in tournaments if t["diagnostic_runnable"]]
    diagnostic_mixed = [t for t in tournaments if t["diagnostic_mixed"]]
    verdict = verdict_for(len(strict), len(diagnostic_runnable))
    primary = primary_for(verdict)
    old_summary = old_tournaments.get("summary", {})
    after_counts = Counter(row["unit_test_label"] for row in rows)
    for label in LABELS:
        after_counts.setdefault(label, 0)
    before_by_task = {row["task_id"]: row for row in inspection.get("tasks", [])}
    summary = {
        "before": {
            "strict_clean": int(old_summary.get("strict_clean_tournaments", 2)),
            "diagnostic_runnable": int(old_summary.get("diagnostic_runnable_tournaments", 3)),
            "diagnostic_mixed": int(old_summary.get("diagnostic_mixed_tournaments", 3)),
            "label_counts": compact_counts(old_summary),
        },
        "after": {
            "strict_clean": len(strict),
            "diagnostic_runnable": len(diagnostic_runnable),
            "diagnostic_mixed": len(diagnostic_mixed),
            "label_counts": {label: after_counts[label] for label in LABELS},
            "tasks": len(tasks),
            "candidates_evaluated": len(rows),
            "new_candidates_evaluated": sum(1 for row in rows if row.get("generation_source") == "balancing_pass"),
            "random_top1_baselines": {
                "strict_clean": random_baseline(strict, "strict_candidates") if strict else float("nan"),
                "diagnostic_runnable": random_baseline(diagnostic_runnable, "diagnostic_runnable_candidates") if diagnostic_runnable else float("nan"),
                "diagnostic_mixed": random_baseline(diagnostic_mixed, "diagnostic_mixed_primary_candidates") if diagnostic_mixed else float("nan"),
            },
        },
        "label_by_generation_source": {
            source: dict(Counter(row["unit_test_label"] for row in rows if row.get("generation_source", "unknown") == source))
            for source in sorted({str(row.get("generation_source", "unknown")) for row in rows})
        },
        "label_by_mode_new": {
            mode: dict(Counter(row["unit_test_label"] for row in rows if row.get("generation_source") == "balancing_pass" and row.get("mode") == mode))
            for mode in sorted({str(row.get("mode", "unknown")) for row in rows if row.get("generation_source") == "balancing_pass"})
        },
    }
    payload = {
        "balanced_tournament_verdict": verdict,
        "BALANCED_TOURNAMENT_VERDICT": verdict,
        "code_v2_tournament_verdict": "CLEAN" if verdict == "GREEN" else "RUNNABLE_DIAGNOSTIC" if verdict == "YELLOW" else "TOO_FEW_TOURNAMENTS",
        "primary_eval_set": primary,
        "taskset": repo_path(TASKSET_JSON),
        "balancing_candidates_json": repo_path(BALANCING_CANDIDATES_JSON),
        "summary": summary,
        "candidate_evaluations": rows,
        "tournaments": tournaments,
        "primary_tournament_ids": [
            t["tournament_id"]
            for t in tournaments
            if (primary == "strict_clean" and t["strict_clean"])
            or (primary == "strict_plus_diagnostic_runnable" and (t["strict_clean"] or t["diagnostic_runnable"]))
        ],
        "strict": strict,
        "diagnostic_runnable": diagnostic_runnable,
        "diagnostic": diagnostic_mixed,
        "converted_tasks": converted_tasks(before_by_task, tournaments, rows),
        "recommended_next_if_stopped": "" if verdict in {"GREEN", "YELLOW"} else "task_design_or_test_granularity_is_bottleneck",
    }
    write_json(OUTPUT_JSON, payload)
    write_md(OUTPUT_MD, payload)
    print(f"BALANCED_TOURNAMENT_VERDICT = {verdict}")
    print(f"strict_clean = {len(strict)}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
