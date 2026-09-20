"""Build natural MCQ-distractor reasoning tournaments from official options."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from code_branch_pilot_lib import PROJECT_ROOT, REPORT_DIR, repo_path, write_json

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "artifacts" / "hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_ROOT / "artifacts" / "hf_cache" / "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "artifacts" / "hf_cache" / "datasets"))

from datasets import load_dataset  # noqa: E402


OUTPUT_JSON = REPORT_DIR / "reasoning_natural_distractor_set_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "reasoning_natural_distractor_set_2026-05-17.md"
LETTERS = tuple("ABCDE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-arc", type=int, default=30)
    parser.add_argument("--target-secondary", type=int, default=30)
    parser.add_argument("--max-total", type=int, default=80)
    parser.add_argument("--output", default=str(OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    parser.add_argument("--max-feature-chars", type=int, default=3000)
    return parser.parse_args()


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def normalize_options(raw_labels: list[Any], raw_texts: list[Any], answer_key: Any) -> tuple[dict[str, str], str] | None:
    options: dict[str, str] = {}
    original_to_new: dict[str, str] = {}
    answer = str(answer_key or "").strip().upper()
    for idx, raw_text in enumerate(raw_texts):
        if idx >= len(LETTERS):
            break
        new_label = LETTERS[idx]
        original = str(raw_labels[idx] if idx < len(raw_labels) else new_label).strip().upper()
        text = clean_text(raw_text)
        if not text:
            return None
        options[new_label] = text
        original_to_new[original] = new_label
        original_to_new[new_label] = new_label
    if len(options) < 3:
        return None
    normalized_answer = original_to_new.get(answer, answer if answer in options else "")
    if normalized_answer not in options:
        return None
    return options, normalized_answer


def candidate_feature_text(question: str, letter: str, option_text: str) -> str:
    return f"Question:\n{question}\n\nCandidate answer:\n{letter}. {option_text}"


def valid_task(question: str, options: dict[str, str], answer: str, max_feature_chars: int) -> bool:
    if not question.strip() or answer not in options:
        return False
    for letter, text in options.items():
        if not text.strip():
            return False
        if len(candidate_feature_text(question, letter, text)) > max_feature_chars:
            return False
    return sum(1 for label in options if label == answer) == 1


def load_arc(target: int, max_feature_chars: int) -> tuple[list[dict[str, Any]], str | None]:
    rows = []
    ds = load_dataset("ai2_arc", "ARC-Challenge", split="validation", cache_dir=os.environ["HF_DATASETS_CACHE"])
    for idx, row in enumerate(ds):
        choices = row["choices"]
        norm = normalize_options(list(choices["label"]), list(choices["text"]), row.get("answerKey"))
        if not norm:
            continue
        options, answer = norm
        question = clean_text(row.get("question"))
        if not valid_task(question, options, answer, max_feature_chars):
            continue
        rows.append({
            "dataset": "ai2_arc_challenge",
            "task_id": f"ARC-Challenge/{idx}",
            "dataset_index": idx,
            "question": question,
            "options": options,
            "answer_key": answer,
        })
        if len(rows) >= target:
            break
    return rows, None


def load_openbook(target: int, max_feature_chars: int) -> tuple[list[dict[str, Any]], str | None]:
    rows = []
    ds = load_dataset("openbookqa", "main", split="validation", cache_dir=os.environ["HF_DATASETS_CACHE"])
    for idx, row in enumerate(ds):
        choices = row["choices"]
        norm = normalize_options(list(choices["label"]), list(choices["text"]), row.get("answerKey"))
        if not norm:
            continue
        options, answer = norm
        question = clean_text(row.get("question_stem"))
        if not valid_task(question, options, answer, max_feature_chars):
            continue
        rows.append({
            "dataset": "openbookqa",
            "task_id": f"OpenBookQA/{idx}",
            "dataset_index": idx,
            "question": question,
            "options": options,
            "answer_key": answer,
        })
        if len(rows) >= target:
            break
    return rows, None


def load_commonsense(target: int, max_feature_chars: int) -> tuple[list[dict[str, Any]], str | None]:
    rows = []
    ds = load_dataset("commonsense_qa", split="validation", cache_dir=os.environ["HF_DATASETS_CACHE"])
    for idx, row in enumerate(ds):
        choices = row["choices"]
        norm = normalize_options(list(choices["label"]), list(choices["text"]), row.get("answerKey"))
        if not norm:
            continue
        options, answer = norm
        question = clean_text(row.get("question"))
        if not valid_task(question, options, answer, max_feature_chars):
            continue
        rows.append({
            "dataset": "commonsense_qa",
            "task_id": f"CommonsenseQA/{idx}",
            "dataset_index": idx,
            "question": question,
            "options": options,
            "answer_key": answer,
        })
        if len(rows) >= target:
            break
    return rows, None


def build_candidates(tasks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    tournaments: list[dict[str, Any]] = []
    for task in tasks:
        candidate_uids = []
        labels = []
        for letter, option_text in task["options"].items():
            uid = f"reasoning_distractor::{task['dataset']}::{task['task_id']}::{letter}"
            is_correct = letter == task["answer_key"]
            candidates.append({
                "candidate_uid": uid,
                "task_id": task["task_id"],
                "dataset": task["dataset"],
                "question": task["question"],
                "option_letter": letter,
                "option_text": option_text,
                "is_correct": is_correct,
                "answer_key": task["answer_key"],
                "n_options": len(task["options"]),
            })
            candidate_uids.append(uid)
            labels.append("correct" if is_correct else "incorrect")
        tournaments.append({
            "task_id": task["task_id"],
            "dataset": task["dataset"],
            "question": task["question"],
            "answer_key": task["answer_key"],
            "n_options": len(task["options"]),
            "candidate_uids": candidate_uids,
            "labels": labels,
        })
    return candidates, tournaments


def summarize(tasks: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    dataset_counts = Counter(task["dataset"] for task in tasks)
    option_counts = Counter(int(task["options"].__len__()) for task in tasks)
    lengths = [len(candidate_feature_text(row["question"], row["option_letter"], row["option_text"])) for row in candidates]
    return {
        "n_questions": len(tasks),
        "n_candidates": len(candidates),
        "random_top1_baseline": float(mean(1.0 / len(task["options"]) for task in tasks)) if tasks else 0.0,
        "dataset_breakdown": dict(dataset_counts),
        "option_count_breakdown": {str(k): v for k, v in sorted(option_counts.items())},
        "prompt_length_stats": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": float(mean(lengths)) if lengths else 0.0,
        },
    }


def verdict(n_questions: int) -> str:
    if n_questions >= 30:
        return "READY"
    if n_questions >= 15:
        return "PARTIAL"
    return "BLOCKED"


def write_md(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# Reasoning Natural Distractor Set",
        "",
        f"REASONING_DISTRACTOR_SET_VERDICT = {payload['reasoning_distractor_set_verdict']}",
        "",
        f"- n_questions: `{s['n_questions']}`",
        f"- n_candidates: `{s['n_candidates']}`",
        f"- random_top1_baseline: `{s['random_top1_baseline']}`",
        f"- dataset_breakdown: `{s['dataset_breakdown']}`",
        f"- option_count_breakdown: `{s['option_count_breakdown']}`",
        f"- prompt_length_stats: `{s['prompt_length_stats']}`",
        f"- dataset_load_errors: `{payload['dataset_load_errors']}`",
        "",
        "## Tournaments",
        "",
    ]
    for task in payload["tasks"][:80]:
        lines.append(f"- `{task['task_id']}` dataset=`{task['dataset']}` options=`{len(task['options'])}` answer=`{task['answer_key']}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    errors: list[str] = []
    tasks: list[dict[str, Any]] = []
    try:
        arc, _ = load_arc(int(args.target_arc), int(args.max_feature_chars))
        tasks.extend(arc)
    except Exception as exc:
        errors.append(f"ai2_arc: {type(exc).__name__}: {exc}")
    secondary_target = min(int(args.target_secondary), max(0, int(args.max_total) - len(tasks)))
    secondary_added = 0
    secondary_needed = secondary_target
    if secondary_needed:
        try:
            openbook, _ = load_openbook(secondary_needed, int(args.max_feature_chars))
            tasks.extend(openbook)
            secondary_added += len(openbook)
            secondary_needed = max(0, secondary_target - secondary_added)
        except Exception as exc:
            errors.append(f"openbookqa: {type(exc).__name__}: {exc}")
    if secondary_needed:
        try:
            commonsense, _ = load_commonsense(secondary_needed, int(args.max_feature_chars))
            tasks.extend(commonsense)
        except Exception as exc:
            errors.append(f"commonsense_qa: {type(exc).__name__}: {exc}")
    tasks = tasks[: int(args.max_total)]
    candidates, tournaments = build_candidates(tasks)
    v = verdict(len(tasks))
    payload = {
        "reasoning_distractor_set_verdict": v,
        "summary": {"REASONING_DISTRACTOR_SET_VERDICT": v, **summarize(tasks, candidates)},
        "dataset_load_errors": errors,
        "tasks": tasks,
        "candidates": candidates,
        "tournaments": tournaments,
        "outputs": {"json": repo_path(args.output), "md": repo_path(args.output_md)},
    }
    write_json(Path(args.output), payload)
    write_md(Path(args.output_md), payload)
    print(f"REASONING_DISTRACTOR_SET_VERDICT = {v}")
    print(f"n_questions = {len(tasks)}")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output_md}")
    if v == "BLOCKED":
        raise SystemExit("REASONING_DISTRACTOR_SET_VERDICT=BLOCKED")


if __name__ == "__main__":
    main()
