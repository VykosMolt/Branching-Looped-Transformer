"""Build a small official-option MCQ task set for reasoning trace generation."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == THIS_DIR:
    sys.path.pop(0)
sys.path.insert(0, str(THIS_DIR))

from code_branch_pilot_lib import PROJECT_ROOT, REPORT_DIR, repo_path, write_json

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "artifacts" / "hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_ROOT / "artifacts" / "hf_cache" / "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "artifacts" / "hf_cache" / "datasets"))

from build_reasoning_natural_distractor_set import (  # noqa: E402
    clean_text,
    load_arc,
    load_commonsense,
    load_openbook,
)


OUTPUT_JSON = REPORT_DIR / "reasoning_trace_task_set_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "reasoning_trace_task_set_2026-05-17.md"
NATURAL_DISTRACTOR_JSON = REPORT_DIR / "reasoning_natural_distractor_set_2026-05-17.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-total", type=int, default=30)
    parser.add_argument("--min-total", type=int, default=20)
    parser.add_argument("--max-total", type=int, default=40)
    parser.add_argument("--target-arc", type=int, default=15)
    parser.add_argument("--target-secondary", type=int, default=15)
    parser.add_argument("--max-feature-chars", type=int, default=3000)
    parser.add_argument("--output", default=str(OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_options(options: Any) -> dict[str, str]:
    if isinstance(options, dict):
        return {str(k).strip().upper(): clean_text(v) for k, v in options.items() if clean_text(v)}
    out: dict[str, str] = {}
    for row in options or []:
        if isinstance(row, dict):
            letter = str(row.get("letter") or row.get("label") or "").strip().upper()
            text = clean_text(row.get("text") or row.get("option_text"))
            if letter and text:
                out[letter] = text
    return out


def feature_text(question: str, trace: str) -> str:
    return f"Question:\n{question}\n\nCandidate reasoning trace:\n{trace}"


def valid_task(task: dict[str, Any], max_feature_chars: int) -> bool:
    question = clean_text(task.get("question"))
    options = normalize_options(task.get("options"))
    answer = str(task.get("answer_key") or task.get("answer") or "").strip().upper()
    if not question or answer not in options or len(options) < 3:
        return False
    if any(not text for text in options.values()):
        return False
    sample_trace = "Short rationale.\nFINAL ANSWER: A"
    return len(feature_text(question, sample_trace)) <= max_feature_chars


def task_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    options = normalize_options(row.get("options"))
    answer = str(row.get("answer_key") or row.get("answer") or "").strip().upper()
    question = clean_text(row.get("question"))
    if not question or answer not in options:
        return None
    return {
        "task_id": str(row.get("task_id")),
        "dataset": str(row.get("dataset", "unknown")),
        "question": question,
        "options": options,
        "answer_key": answer,
        "n_options": len(options),
        "source_artifact": repo_path(NATURAL_DISTRACTOR_JSON),
    }


def tasks_from_existing(args: argparse.Namespace) -> list[dict[str, Any]]:
    payload = load_json(NATURAL_DISTRACTOR_JSON)
    tasks_by_id: dict[str, dict[str, Any]] = {}
    for row in payload.get("tasks", []) or []:
        task = task_from_row(row)
        if task and valid_task(task, int(args.max_feature_chars)):
            tasks_by_id[task["task_id"]] = task
    if not tasks_by_id:
        for tour in payload.get("tournaments", []) or []:
            task = task_from_row(tour)
            if task and valid_task(task, int(args.max_feature_chars)):
                tasks_by_id[task["task_id"]] = task
    tasks = list(tasks_by_id.values())
    return select_balanced(tasks, int(args.target_arc), int(args.target_secondary), int(args.max_total), int(args.target_total))


def tasks_from_datasets(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    tasks: list[dict[str, Any]] = []
    try:
        arc, err = load_arc(int(args.target_arc), int(args.max_feature_chars))
        if err:
            errors.append(f"load_arc: {err}")
        tasks.extend([{**row, "n_options": len(row["options"]), "source_artifact": "huggingface:ai2_arc/ARC-Challenge"} for row in arc])
    except Exception as exc:
        errors.append(f"load_arc: {type(exc).__name__}: {exc}")
    remaining = max(int(args.target_secondary), int(args.target_total) - len(tasks))
    for loader, source in ((load_openbook, "huggingface:openbookqa/main"), (load_commonsense, "huggingface:commonsense_qa")):
        if len(tasks) >= int(args.target_total):
            break
        try:
            rows, err = loader(remaining, int(args.max_feature_chars))
            if err:
                errors.append(f"{loader.__name__}: {err}")
            for row in rows:
                task = {**row, "n_options": len(row["options"]), "source_artifact": source}
                if valid_task(task, int(args.max_feature_chars)):
                    tasks.append(task)
        except Exception as exc:
            errors.append(f"{loader.__name__}: {type(exc).__name__}: {exc}")
    tasks = select_balanced(tasks, int(args.target_arc), int(args.target_secondary), int(args.max_total), int(args.target_total))
    return tasks, errors


def select_balanced(tasks: list[dict[str, Any]], target_arc: int, target_secondary: int, max_total: int, target_total: int) -> list[dict[str, Any]]:
    arc = [task for task in tasks if "arc" in task["dataset"].lower()]
    secondary = [task for task in tasks if task not in arc]
    selected = arc[:target_arc] + secondary[:target_secondary]
    seen = {task["task_id"] for task in selected}
    for task in tasks:
        if len(selected) >= min(max_total, target_total):
            break
        if task["task_id"] not in seen:
            selected.append(task)
            seen.add(task["task_id"])
    return selected[:max_total]


def summarize(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = []
    for task in tasks:
        for letter, option in task["options"].items():
            trace = f"A concise rationale defending {letter}. {option}\nFINAL ANSWER: {letter}"
            lengths.append(len(feature_text(task["question"], trace)))
    return {
        "n_questions": len(tasks),
        "n_candidates_if_all_options_used": sum(len(task["options"]) for task in tasks),
        "random_top1_baseline": float(mean(1.0 / len(task["options"]) for task in tasks)) if tasks else 0.0,
        "dataset_breakdown": dict(Counter(task["dataset"] for task in tasks)),
        "option_count_breakdown": {str(k): v for k, v in sorted(Counter(len(task["options"]) for task in tasks).items())},
        "prompt_length_stats": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": float(mean(lengths)) if lengths else 0.0,
        },
    }


def verdict(n_questions: int) -> str:
    if n_questions >= 20:
        return "READY"
    if n_questions >= 10:
        return "PARTIAL"
    return "BLOCKED"


def write_md(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# Reasoning Trace Task Set",
        "",
        f"REASONING_TRACE_TASK_SET_VERDICT = {payload['reasoning_trace_task_set_verdict']}",
        "",
        f"- n_questions: `{s['n_questions']}`",
        f"- n_candidates_if_all_options_used: `{s['n_candidates_if_all_options_used']}`",
        f"- random_top1_baseline: `{s['random_top1_baseline']}`",
        f"- dataset_breakdown: `{s['dataset_breakdown']}`",
        f"- option_count_breakdown: `{s['option_count_breakdown']}`",
        f"- prompt_length_stats: `{s['prompt_length_stats']}`",
        f"- dataset_load_errors: `{payload.get('dataset_load_errors', [])}`",
        "",
        "## Tasks",
        "",
    ]
    for task in payload.get("tasks", [])[:80]:
        lines.append(f"- `{task['task_id']}` dataset=`{task['dataset']}` answer=`{task['answer_key']}` options=`{task['n_options']}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_json = Path(args.output)
    out_md = Path(args.output_md)
    tasks = tasks_from_existing(args)
    errors: list[str] = []
    if len(tasks) < int(args.min_total):
        tasks, errors = tasks_from_datasets(args)
    v = verdict(len(tasks))
    payload = {
        "reasoning_trace_task_set_verdict": v,
        "summary": {"REASONING_TRACE_TASK_SET_VERDICT": v, **summarize(tasks)},
        "dataset_load_errors": errors,
        "tasks": tasks,
        "outputs": {"json": repo_path(out_json), "md": repo_path(out_md)},
    }
    write_json(out_json, payload)
    write_md(out_md, payload)
    print(f"REASONING_TRACE_TASK_SET_VERDICT = {v}")
    print(f"n_questions = {len(tasks)}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    if v == "BLOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
