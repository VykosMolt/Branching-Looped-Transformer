"""Build a code task pool for strict-clean-ready screening."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_code_branch_taskset import LOCAL_TASKS
from build_code_branch_taskset_v2 import finalize_task, load_humaneval_tasks, load_mbpp_tasks
from code_branch_pilot_lib import REPORT_DIR, repo_path, snippet, write_json


OUTPUT_JSON = REPORT_DIR / "code_strict_clean_screening_taskpool_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "code_strict_clean_screening_taskpool_2026-05-17.md"

PRIOR_INSPECTION = REPORT_DIR / "code_branch_near_miss_balance_inspection_2026-05-17.json"
PRIOR_BALANCED = REPORT_DIR / "code_branch_near_miss_balanced_tournaments_2026-05-17.json"
PRIOR_MINI = REPORT_DIR / "code_branch_tournaments_v2_mini_patched_2026-05-16.json"

LOW_PRIORITY_PRIOR_OUTCOMES = {"all_correct", "all_wrong", "dirty"}
EXHAUSTED_BUCKETS = {
    "needs_correct_anchor",
    "needs_near_miss",
    "all_correct_collapse",
    "all_wrong_collapse",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-tasks", type=int, default=60)
    parser.add_argument("--min-tasks", type=int, default=30)
    parser.add_argument("--output", default=str(OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    return parser.parse_args()


def load_json_if_present(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def outcome_from_labels(labels: Counter[str]) -> str:
    has_correct = labels.get("correct", 0) > 0
    has_near = labels.get("near_miss", 0) > 0
    has_wrong = labels.get("wrong_code", 0) > 0 or labels.get("runtime_error", 0) > 0
    usable = labels.get("correct", 0) + labels.get("near_miss", 0) + labels.get("wrong_code", 0) + labels.get("runtime_error", 0)
    if has_correct and has_near:
        return "strict_clean"
    if usable and labels.get("correct", 0) == usable:
        return "all_correct"
    if has_correct:
        return "anchor_only"
    if has_near:
        return "near_miss_only"
    if has_wrong:
        return "all_wrong"
    return "unknown"


def load_prior_outcomes() -> dict[str, dict[str, Any]]:
    prior: dict[str, dict[str, Any]] = {}

    inspection = load_json_if_present(PRIOR_INSPECTION)
    bucket_to_outcome = {
        "already_strict_clean": "strict_clean",
        "needs_correct_anchor": "near_miss_only",
        "needs_near_miss": "anchor_only",
        "all_correct_collapse": "all_correct",
        "all_wrong_collapse": "all_wrong",
    }
    for row in inspection.get("tasks", []):
        task_id = str(row.get("task_id", ""))
        if not task_id:
            continue
        bucket = str(row.get("bucket", "unknown"))
        prior[task_id] = {
            "prior_outcome": bucket_to_outcome.get(bucket, "unknown"),
            "prior_bucket": bucket,
            "prior_exhausted": bucket in EXHAUSTED_BUCKETS,
            "prior_label_counts": row.get("label_counts", {}),
            "prior_source": repo_path(PRIOR_INSPECTION),
        }

    for path, source_name in ((PRIOR_BALANCED, "balanced"), (PRIOR_MINI, "patched_mini")):
        payload = load_json_if_present(path)
        for tournament in payload.get("tournaments", []):
            task_id = str(tournament.get("task_id", ""))
            if not task_id or task_id in prior:
                continue
            rows = tournament.get("diagnostic_candidates") or tournament.get("diagnostic_runnable_candidates") or []
            labels = Counter(str(row.get("unit_test_label", "unknown")) for row in rows)
            if not labels:
                continue
            prior[task_id] = {
                "prior_outcome": outcome_from_labels(labels),
                "prior_bucket": source_name,
                "prior_exhausted": False,
                "prior_label_counts": dict(labels),
                "prior_source": repo_path(path),
            }
    return prior


def tests_for(task: dict[str, Any]) -> list[str]:
    tests = list(task.get("tests") or [])
    if not tests:
        tests = list(task.get("public_tests", [])) + list(task.get("hidden_tests", []))
    return [str(test).strip() for test in tests if str(test).strip()]


def signature_for(task: dict[str, Any]) -> str:
    if task.get("signature_hint"):
        return str(task["signature_hint"])
    starter = str(task.get("starter_code", "")).strip()
    for line in starter.splitlines():
        line = line.strip()
        if line.startswith("def "):
            return line.rstrip(":")
    fn = str(task.get("function_name", "candidate"))
    return f"def {fn}(...)"


def add_screening_metadata(task: dict[str, Any], prior: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = finalize_task(dict(task))
    tests = tests_for(out)
    task_id = str(out["task_id"])
    prior_row = prior.get(task_id, {})
    out["tests"] = tests
    out["number_of_tests"] = len(tests)
    out["signature"] = signature_for(out)
    out["prior_outcome"] = prior_row.get("prior_outcome", "unknown")
    out["prior_bucket"] = prior_row.get("prior_bucket", "")
    out["prior_exhausted"] = bool(prior_row.get("prior_exhausted", False))
    out["prior_label_counts"] = prior_row.get("prior_label_counts", {})
    out["prior_source"] = prior_row.get("prior_source", "")
    out.setdefault("tests_visibility", "public_and_hidden")
    return out


def priority(task: dict[str, Any]) -> tuple[int, int, int, str]:
    prior_outcome = str(task.get("prior_outcome", "unknown"))
    exhausted = bool(task.get("prior_exhausted", False))
    source = str(task.get("source", "unknown"))
    difficulty = str(task.get("difficulty", "unknown"))
    n_tests = int(task.get("number_of_tests", 0))
    granular_bonus = 0 if n_tests >= 3 else 1
    difficulty_order = {"hard": 0, "medium": 1, "unknown": 2, "easy": 3, "devil": 4}

    if prior_outcome in {"anchor_only", "near_miss_only"} and not exhausted:
        band = 0
    elif prior_outcome == "unknown" and n_tests >= 3 and source in {"mbpp", "local_dsa"}:
        band = 1
    elif source == "mbpp" and difficulty in {"medium", "hard"}:
        band = 2
    elif source == "local_dsa" and prior_outcome not in LOW_PRIORITY_PRIOR_OUTCOMES:
        band = 3
    elif source == "humaneval":
        band = 5
    else:
        band = 8

    if exhausted:
        band = max(band, 7)
    if prior_outcome in LOW_PRIORITY_PRIOR_OUTCOMES:
        band = max(band, 8)
    if difficulty == "devil":
        band = max(band, 9)
    return (band, granular_bonus, difficulty_order.get(difficulty, 5), str(task.get("task_id", "")))


def select_tasks(tasks: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_functions: set[str] = set()
    for task in sorted(tasks, key=priority):
        task_id = str(task.get("task_id", ""))
        fn = str(task.get("function_name", ""))
        if not task_id or task_id in seen_ids or (fn and fn in seen_functions):
            continue
        selected.append(task)
        seen_ids.add(task_id)
        if fn:
            seen_functions.add(fn)
        if len(selected) >= target:
            break
    return selected


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Strict-Clean Code Screening Task Pool",
        "",
        f"SCREENING_TASKPOOL_VERDICT = {payload['screening_taskpool_verdict']}",
        "",
        f"- tasks: `{len(payload['tasks'])}`",
        f"- source_mix: `{payload['summary']['source_mix']}`",
        f"- difficulty_mix: `{payload['summary']['difficulty_mix']}`",
        f"- prior_outcome_mix: `{payload['summary']['prior_outcome_mix']}`",
        f"- tests_visibility_mix: `{payload['summary']['tests_visibility_mix']}`",
        f"- dataset_status: `{payload['dataset_status']}`",
        "",
        "## Tasks",
        "",
        "| task_id | source | difficulty | function | tests | visibility | prior_outcome | prior_exhausted | priority | prompt |",
        "| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for task in payload["tasks"]:
        p = priority(task)
        lines.append(
            f"| `{task['task_id']}` | `{task['source']}` | `{task.get('difficulty', 'unknown')}` | "
            f"`{task.get('function_name', '')}` | {task.get('number_of_tests', 0)} | "
            f"`{task.get('tests_visibility', '')}` | `{task.get('prior_outcome', 'unknown')}` | "
            f"{bool(task.get('prior_exhausted', False))} | {p[0]} | {snippet(task.get('prompt', ''), 110)} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    prior = load_prior_outcomes()

    local = [add_screening_metadata({**task, "tests_visibility": "local_public_and_hidden"}, prior) for task in LOCAL_TASKS]
    existing_names = {task["function_name"] for task in local}
    mbpp, mbpp_error = load_mbpp_tasks(max(int(args.target_tasks) * 2, 90), existing_names, set())
    mbpp = [add_screening_metadata(task, prior) for task in mbpp]

    pool = local + mbpp
    human: list[dict[str, Any]] = []
    human_error = ""
    if len(pool) < int(args.target_tasks):
        human, human_error = load_humaneval_tasks(int(args.target_tasks) - len(pool))
        human = [add_screening_metadata(task, prior) for task in human]
        pool.extend(human)

    selected = select_tasks(pool, int(args.target_tasks))
    if len(selected) < int(args.min_tasks):
        verdict = "BLOCKED"
    elif len(selected) < int(args.target_tasks):
        verdict = "TOO_SMALL"
    else:
        verdict = "READY"

    summary = {
        "tasks": len(selected),
        "target_tasks": int(args.target_tasks),
        "min_tasks": int(args.min_tasks),
        "source_mix": dict(Counter(task["source"] for task in selected)),
        "difficulty_mix": dict(Counter(task.get("difficulty", "unknown") for task in selected)),
        "prior_outcome_mix": dict(Counter(task.get("prior_outcome", "unknown") for task in selected)),
        "tests_visibility_mix": dict(Counter(task.get("tests_visibility", "unknown") for task in selected)),
        "granular_tests_tasks": sum(1 for task in selected if int(task.get("number_of_tests", 0)) >= 3),
        "low_priority_prior_tasks": sum(
            1
            for task in selected
            if task.get("prior_outcome") in LOW_PRIORITY_PRIOR_OUTCOMES or task.get("prior_exhausted")
        ),
    }
    payload = {
        "screening_taskpool_verdict": verdict,
        "tasks": selected,
        "summary": summary,
        "dataset_status": {
            "local_dsa": len(local),
            "mbpp": "loaded" if mbpp else f"unavailable: {mbpp_error}",
            "humaneval": "not_needed" if not human and len(pool) >= int(args.target_tasks) else ("loaded" if human else f"unavailable: {human_error}"),
        },
        "prior_artifacts": {
            "inspection": repo_path(PRIOR_INSPECTION),
            "balanced": repo_path(PRIOR_BALANCED),
            "patched_mini": repo_path(PRIOR_MINI),
        },
        "selection_notes": (
            "The pool prioritizes granular unknown MBPP/local tasks and deprioritizes tasks already exhausted "
            "by the near-miss balancing pass or known all-correct/all-wrong collapse."
        ),
        "outputs": {"json": repo_path(Path(args.output)), "md": repo_path(Path(args.output_md))},
    }
    out_json = Path(args.output)
    out_md = Path(args.output_md)
    write_json(out_json, payload)
    write_md(out_md, payload)
    print(f"SCREENING_TASKPOOL_VERDICT = {verdict}")
    print(f"tasks = {len(selected)}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    if verdict == "BLOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
