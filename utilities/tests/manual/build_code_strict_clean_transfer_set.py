"""Build strict-clean transfer tournaments from screened code candidates."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from code_branch_pilot_lib import REPORT_DIR, repo_path, snippet, write_json


TASKPOOL_JSON = REPORT_DIR / "code_strict_clean_screening_taskpool_2026-05-17.json"
CANDIDATES_JSON = REPORT_DIR / "code_strict_clean_screening_candidates_2026-05-17.json"
RESULTS_JSON = REPORT_DIR / "code_strict_clean_screening_results_2026-05-17.json"
OUTPUT_JSON = REPORT_DIR / "code_strict_clean_transfer_set_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "code_strict_clean_transfer_set_2026-05-17.md"

STRICT_TASK_IDS = {"mbpp/100", "mbpp/129", "mbpp/283", "mbpp/291", "mbpp/391", "mbpp/392"}
PRIMARY_LABELS = {"correct", "near_miss"}
SECONDARY_LABELS = {"correct", "near_miss", "wrong_code"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taskpool", default=str(TASKPOOL_JSON))
    parser.add_argument("--candidates", default=str(CANDIDATES_JSON))
    parser.add_argument("--results", default=str(RESULTS_JSON))
    parser.add_argument("--output", default=str(OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dedup_key(row: dict[str, Any]) -> str:
    return str(row.get("candidate_uid") or row.get("ast_hash") or row.get("normalized_code_hash") or row.get("raw_code_hash") or "")


def dedup_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = dedup_key(row)
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(row)
    return out


def candidate_for_output(row: dict[str, Any]) -> dict[str, Any]:
    code = row.get("final_code_for_unit_tests") or row.get("final_code") or row.get("sanitized_code") or ""
    return {
        "candidate_uid": row.get("candidate_uid", ""),
        "task_id": row.get("task_id", ""),
        "source": row.get("source", "unknown"),
        "difficulty": row.get("difficulty", "unknown"),
        "function_name": row.get("function_name", ""),
        "label": row.get("unit_test_label", ""),
        "unit_test_label": row.get("unit_test_label", ""),
        "is_correct": row.get("unit_test_label") == "correct",
        "is_runnable": bool(row.get("is_runnable")),
        "tests_total": row.get("tests_total"),
        "tests_passed": row.get("tests_passed"),
        "pass_rate": row.get("pass_rate"),
        "screening_role": row.get("screening_role", ""),
        "mode": row.get("mode", ""),
        "candidate_stage": row.get("candidate_stage", ""),
        "route": row.get("route", ""),
        "temperature": row.get("temperature"),
        "raw_code_hash": row.get("raw_code_hash", ""),
        "normalized_code_hash": row.get("normalized_code_hash", ""),
        "ast_hash": row.get("ast_hash", ""),
        "duplicate_of": row.get("duplicate_of"),
        "final_code": code,
    }


def random_baseline(tournaments: list[dict[str, Any]], key: str) -> float:
    if not tournaments:
        return float("nan")
    return float(mean(
        sum(1 for cand in t[key] if cand["is_correct"]) / max(len(t[key]), 1)
        for t in tournaments
    ))


def write_md(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# Strict-Clean Code Transfer Set",
        "",
        f"STRICT_CLEAN_TRANSFER_SET_VERDICT = {payload['strict_clean_transfer_set_verdict']}",
        "",
        f"- tasks: `{s['n_tasks']}`",
        f"- primary_candidates: `{s['n_candidates_primary']}`",
        f"- correct: `{s['n_correct']}`",
        f"- near_miss: `{s['n_near_miss']}`",
        f"- wrong_code_secondary: `{s['n_wrong_code_secondary']}`",
        f"- random_top1_baseline_primary: `{s['random_top1_baseline_primary']}`",
        f"- random_top1_baseline_secondary: `{s['random_top1_baseline_secondary']}`",
        "",
        "## Tournaments",
        "",
        "| task_id | primary labels | secondary labels | primary n | secondary n | prompt |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for tournament in payload["tournaments"]:
        lines.append(
            f"| `{tournament['task_id']}` | `{tournament['primary_label_counts']}` | "
            f"`{tournament['secondary_label_counts']}` | {len(tournament['strict_clean_primary_candidates'])} | "
            f"{len(tournament['strict_clean_plus_wrong_code_candidates'])} | {snippet(tournament['prompt'], 100)} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    try:
        taskpool = load_json(args.taskpool)
        candidates_payload = load_json(args.candidates)
        results_payload = load_json(args.results)
    except Exception as exc:
        payload = {
            "strict_clean_transfer_set_verdict": "BLOCKED",
            "error": f"{type(exc).__name__}: {exc}",
            "inputs": {"taskpool": args.taskpool, "candidates": args.candidates, "results": args.results},
        }
        write_json(Path(args.output), payload)
        raise SystemExit("STRICT_CLEAN_TRANSFER_SET_VERDICT=BLOCKED")

    tasks = {str(task["task_id"]): task for task in taskpool.get("tasks", [])}
    eval_rows = [
        row
        for row in results_payload.get("candidate_evaluations", [])
        if str(row.get("task_id", "")) in STRICT_TASK_IDS and not row.get("duplicate_of")
    ]
    tournaments: list[dict[str, Any]] = []
    for task_id in sorted(STRICT_TASK_IDS):
        task = tasks.get(task_id)
        task_rows = dedup_rows([row for row in eval_rows if row.get("task_id") == task_id])
        primary_rows = [
            row for row in task_rows
            if row.get("unit_test_label") in PRIMARY_LABELS and row.get("is_runnable")
        ]
        secondary_rows = [
            row for row in task_rows
            if row.get("unit_test_label") in SECONDARY_LABELS and row.get("is_runnable")
        ]
        primary = [candidate_for_output(row) for row in primary_rows]
        secondary = [candidate_for_output(row) for row in secondary_rows]
        if task and any(c["label"] == "correct" for c in primary) and any(c["label"] == "near_miss" for c in primary):
            tournaments.append({
                "tournament_id": len(tournaments),
                "task_id": task_id,
                "source": task.get("source", "unknown"),
                "difficulty": task.get("difficulty", "unknown"),
                "function_name": task.get("function_name", ""),
                "signature": task.get("signature", task.get("signature_hint", "")),
                "prompt": task.get("prompt", ""),
                "tests_visibility": task.get("tests_visibility", ""),
                "number_of_tests": task.get("number_of_tests", len(task.get("tests", []))),
                "strict_clean_primary_candidates": primary,
                "strict_clean_plus_wrong_code_candidates": secondary,
                "primary_label_counts": dict(Counter(c["label"] for c in primary)),
                "secondary_label_counts": dict(Counter(c["label"] for c in secondary)),
                "strict_clean": True,
            })

    n_tasks = len(tournaments)
    primary_candidates = [cand for t in tournaments for cand in t["strict_clean_primary_candidates"]]
    secondary_candidates = [cand for t in tournaments for cand in t["strict_clean_plus_wrong_code_candidates"]]
    n_correct = sum(1 for cand in primary_candidates if cand["label"] == "correct")
    n_near = sum(1 for cand in primary_candidates if cand["label"] == "near_miss")
    n_wrong = sum(1 for cand in secondary_candidates if cand["label"] == "wrong_code")
    if n_tasks >= 5:
        verdict = "READY"
    elif n_tasks >= 1:
        verdict = "TOO_SMALL"
    else:
        verdict = "BLOCKED"
    summary = {
        "n_tasks": n_tasks,
        "n_candidates_primary": len(primary_candidates),
        "n_candidates_secondary": len(secondary_candidates),
        "n_correct": n_correct,
        "n_near_miss": n_near,
        "n_wrong_code_secondary": n_wrong,
        "random_top1_baseline_primary": random_baseline(tournaments, "strict_clean_primary_candidates"),
        "random_top1_baseline_secondary": random_baseline(tournaments, "strict_clean_plus_wrong_code_candidates"),
        "selected_task_ids": sorted(STRICT_TASK_IDS),
        "loaded_task_ids": [t["task_id"] for t in tournaments],
        "candidate_source_counts": dict(Counter(c.get("mode", "unknown") for c in primary_candidates)),
    }
    payload = {
        "strict_clean_transfer_set_verdict": verdict,
        "primary_eval_set": "strict_clean_primary",
        "secondary_eval_set": "strict_clean_plus_wrong_code",
        "inputs": {
            "taskpool": repo_path(Path(args.taskpool)),
            "candidates": repo_path(Path(args.candidates)),
            "results": repo_path(Path(args.results)),
        },
        "screening_generation_verdict": candidates_payload.get("screening_generation_verdict"),
        "strict_clean_screening_verdict": results_payload.get("strict_clean_screening_verdict"),
        "summary": summary,
        "tournaments": tournaments,
        "notes": (
            "Primary set contains only correct and near_miss candidates. wrong_code candidates from the same tasks "
            "are retained only for the secondary diagnostic set."
        ),
        "outputs": {"json": repo_path(Path(args.output)), "md": repo_path(Path(args.output_md))},
    }
    write_json(Path(args.output), payload)
    write_md(Path(args.output_md), payload)
    print(f"STRICT_CLEAN_TRANSFER_SET_VERDICT = {verdict}")
    print(f"tasks = {n_tasks}")
    print(f"primary_candidates = {len(primary_candidates)}")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output_md}")
    if verdict == "BLOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
