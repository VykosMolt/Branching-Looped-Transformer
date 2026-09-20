"""Build science-domain natural MCQ-distractor tournaments from official options."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

THIS_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == THIS_DIR:
    sys.path.pop(0)
sys.path.insert(0, str(THIS_DIR))

from code_branch_pilot_lib import PROJECT_ROOT, REPORT_DIR, repo_path, write_json

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "artifacts" / "hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_ROOT / "artifacts" / "hf_cache" / "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "artifacts" / "hf_cache" / "datasets"))

from datasets import load_dataset  # noqa: E402


OUTPUT_JSON = REPORT_DIR / "science_natural_distractor_set_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "science_natural_distractor_set_2026-05-17.md"
LETTERS = tuple("ABCDE")
MAJOR_BUCKETS = ("biology", "chemistry", "medicine", "general_science")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-total", type=int, default=120)
    parser.add_argument("--target-per-bucket", type=int, default=20)
    parser.add_argument("--min-total", type=int, default=30)
    parser.add_argument("--max-feature-chars", type=int, default=3000)
    parser.add_argument("--output", default=str(OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def normalized_question_key(question: str) -> str:
    compact = re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()
    return re.sub(r"\s+", " ", compact)


def feature_text(question: str, letter: str, option_text: str) -> str:
    return f"Science question:\n{question}\n\nCandidate answer:\n{letter}. {option_text}"


def normalize_answer(answer: Any, labels: list[str], source_dataset: str) -> str:
    if answer is None:
        return ""
    if isinstance(answer, bool):
        return ""
    if isinstance(answer, int):
        if source_dataset.lower().endswith("medmcqa") and 1 <= answer <= len(labels):
            return labels[answer - 1]
        if 0 <= answer < len(labels):
            return labels[answer]
        if 1 <= answer <= len(labels):
            return labels[answer - 1]
    text = str(answer).strip()
    upper = text.upper()
    if upper in labels:
        return upper
    if text.isdigit():
        n = int(text)
        if source_dataset.lower().endswith("medmcqa") and 1 <= n <= len(labels):
            return labels[n - 1]
        if 0 <= n < len(labels):
            return labels[n]
        if 1 <= n <= len(labels):
            return labels[n - 1]
    return ""


def normalize_labeled_options(raw_labels: Iterable[Any], raw_texts: Iterable[Any], answer_key: Any, source_dataset: str) -> tuple[dict[str, str], str] | None:
    raw_labels_list = list(raw_labels)
    raw_texts_list = list(raw_texts)
    options: dict[str, str] = {}
    original_to_new: dict[str, str] = {}
    for idx, raw_text in enumerate(raw_texts_list):
        if idx >= len(LETTERS):
            break
        new_label = LETTERS[idx]
        original = str(raw_labels_list[idx] if idx < len(raw_labels_list) else new_label).strip().upper()
        text = clean_text(raw_text)
        if not text:
            return None
        options[new_label] = text
        original_to_new[original] = new_label
        original_to_new[new_label] = new_label
        original_to_new[str(idx)] = new_label
        original_to_new[str(idx + 1)] = new_label
    if len(options) < 3:
        return None
    answer = normalize_answer(answer_key, list(options), source_dataset)
    answer = original_to_new.get(str(answer_key).strip().upper(), answer)
    if answer not in options:
        return None
    return options, answer


def deterministic_sciq_options(row: dict[str, Any], source_dataset: str) -> tuple[dict[str, str], str] | None:
    correct = clean_text(row.get("correct_answer"))
    distractors = [clean_text(row.get(f"distractor{i}")) for i in (1, 2, 3)]
    if not correct or any(not x for x in distractors):
        return None
    items = [(correct, True)] + [(x, False) for x in distractors]
    seed = int(hashlib.sha256(clean_text(row.get("question")).encode("utf-8", errors="replace")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    rng.shuffle(items)
    options: dict[str, str] = {}
    answer = ""
    for idx, (text, is_correct) in enumerate(items):
        label = LETTERS[idx]
        options[label] = text
        if is_correct:
            answer = label
    return options, answer if answer in options else ""


def options_from_row(row: dict[str, Any], source_dataset: str) -> tuple[dict[str, str], str] | None:
    if {"correct_answer", "distractor1", "distractor2", "distractor3"}.issubset(row.keys()):
        return deterministic_sciq_options(row, source_dataset)
    if all(key in row for key in ("opa", "opb", "opc", "opd")):
        texts = [row.get("opa"), row.get("opb"), row.get("opc"), row.get("opd")]
        return normalize_labeled_options(list("ABCD"), texts, row.get("cop", row.get("answer")), source_dataset)
    choices = row.get("choices")
    if isinstance(choices, dict):
        labels = list(choices.get("label", []))
        texts = list(choices.get("text", []))
        return normalize_labeled_options(labels or list("ABCDE")[: len(texts)], texts, row.get("answerKey", row.get("answer")), source_dataset)
    if isinstance(choices, list):
        texts = [item.get("text", item) if isinstance(item, dict) else item for item in choices]
        labels = [item.get("label", LETTERS[idx]) if isinstance(item, dict) else LETTERS[idx] for idx, item in enumerate(choices[: len(LETTERS)])]
        return normalize_labeled_options(labels, texts, row.get("answer", row.get("answerKey")), source_dataset)
    options = row.get("options")
    if isinstance(options, list):
        texts = [item.get("text", item) if isinstance(item, dict) else item for item in options]
        labels = [item.get("label", LETTERS[idx]) if isinstance(item, dict) else LETTERS[idx] for idx, item in enumerate(options[: len(LETTERS)])]
        return normalize_labeled_options(labels, texts, row.get("answer", row.get("answer_idx", row.get("answer_index"))), source_dataset)
    if isinstance(options, dict):
        labels = list(options)
        texts = [options[label] for label in labels]
        return normalize_labeled_options(labels, texts, row.get("answer", row.get("answerKey", row.get("answer_idx"))), source_dataset)
    return None


def question_from_row(row: dict[str, Any]) -> str:
    for key in ("question", "question_stem", "stem", "input"):
        text = clean_text(row.get(key))
        if text:
            return text
    return ""


def valid_task(question: str, options: dict[str, str], answer: str, max_feature_chars: int) -> bool:
    if not question or answer not in options or len(options) < 3:
        return False
    if len([letter for letter in options if letter == answer]) != 1:
        return False
    for letter, text in options.items():
        if not text:
            return False
        if len(feature_text(question, letter, text)) > max_feature_chars:
            return False
    return True


def make_task(
    *,
    row: dict[str, Any],
    idx: int,
    source_dataset: str,
    source_subject: str,
    subdomain_bucket: str,
    max_feature_chars: int,
) -> dict[str, Any] | None:
    question = question_from_row(row)
    norm = options_from_row(row, source_dataset)
    if not norm:
        return None
    options, answer = norm
    if not valid_task(question, options, answer, max_feature_chars):
        return None
    task_id = f"{source_dataset}/{source_subject}/{idx}".replace(" ", "_")
    return {
        "task_id": task_id,
        "source_dataset": source_dataset,
        "source_subject": source_subject,
        "subdomain_bucket": subdomain_bucket,
        "question": question,
        "options": options,
        "answer_key": answer,
        "n_options": len(options),
        "dataset_index": idx,
    }


def load_split(dataset_name: str, config: str | None, splits: tuple[str, ...]) -> tuple[Any | None, str | None]:
    errors = []
    for split in splits:
        try:
            if config:
                return load_dataset(dataset_name, config, split=split, cache_dir=os.environ["HF_DATASETS_CACHE"]), None
            return load_dataset(dataset_name, split=split, cache_dir=os.environ["HF_DATASETS_CACHE"]), None
        except Exception as exc:
            errors.append(f"{split}: {type(exc).__name__}: {exc}")
    return None, "; ".join(errors)


def collect_from_dataset(
    *,
    dataset_name: str,
    config: str | None,
    source_dataset: str,
    source_subject: str,
    subdomain_bucket: str,
    target: int,
    max_feature_chars: int,
    splits: tuple[str, ...] = ("validation", "test", "train"),
) -> tuple[list[dict[str, Any]], str | None]:
    ds, err = load_split(dataset_name, config, splits)
    if ds is None:
        return [], err
    rows = []
    for idx, row in enumerate(ds):
        task = make_task(
            row=dict(row),
            idx=idx,
            source_dataset=source_dataset,
            source_subject=source_subject,
            subdomain_bucket=subdomain_bucket,
            max_feature_chars=max_feature_chars,
        )
        if task:
            rows.append(task)
        if len(rows) >= target:
            break
    return rows, None


def collect_all(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    per_source_target = max(int(args.target_per_bucket) * 2, 40)
    specs = [
        ("allenai/sciq", None, "sciq", "sciq", "general_science", int(args.target_per_bucket) * 4, ("validation", "test", "train")),
        ("cais/mmlu", "high_school_biology", "mmlu", "high_school_biology", "biology", per_source_target, ("test", "validation", "dev")),
        ("cais/mmlu", "college_biology", "mmlu", "college_biology", "biology", per_source_target, ("test", "validation", "dev")),
        ("cais/mmlu", "high_school_chemistry", "mmlu", "high_school_chemistry", "chemistry", per_source_target, ("test", "validation", "dev")),
        ("cais/mmlu", "college_chemistry", "mmlu", "college_chemistry", "chemistry", per_source_target, ("test", "validation", "dev")),
        ("cais/mmlu", "anatomy", "mmlu", "anatomy", "medicine", per_source_target, ("test", "validation", "dev")),
        ("cais/mmlu", "college_medicine", "mmlu", "college_medicine", "medicine", per_source_target, ("test", "validation", "dev")),
        ("cais/mmlu", "professional_medicine", "mmlu", "professional_medicine", "medicine", per_source_target, ("test", "validation", "dev")),
        ("cais/mmlu", "medical_genetics", "mmlu", "medical_genetics", "medicine", per_source_target, ("test", "validation", "dev")),
        ("cais/mmlu", "high_school_physics", "mmlu", "high_school_physics", "other_science", int(args.target_per_bucket), ("test", "validation", "dev")),
        ("openlifescienceai/medmcqa", None, "medmcqa", "medmcqa", "medicine", per_source_target, ("validation", "train", "test")),
        ("openlifescienceai/mmlu_clinical_knowledge", None, "openlifescienceai_mmlu", "clinical_knowledge", "medicine", per_source_target, ("test", "validation", "train")),
        ("openlifescienceai/mmlu_professional_medicine", None, "openlifescienceai_mmlu", "professional_medicine", "medicine", per_source_target, ("test", "validation", "train")),
        ("openlifescienceai/mmlu_college_medicine", None, "openlifescienceai_mmlu", "college_medicine", "medicine", per_source_target, ("test", "validation", "train")),
        ("openlifescienceai/mmlu_anatomy", None, "openlifescienceai_mmlu", "anatomy", "medicine", per_source_target, ("test", "validation", "train")),
        ("openlifescienceai/mmlu_medical_genetics", None, "openlifescienceai_mmlu", "medical_genetics", "medicine", per_source_target, ("test", "validation", "train")),
    ]
    tasks: list[dict[str, Any]] = []
    errors: list[str] = []
    for dataset_name, config, source_dataset, source_subject, bucket, target, splits in specs:
        rows, err = collect_from_dataset(
            dataset_name=dataset_name,
            config=config,
            source_dataset=source_dataset,
            source_subject=source_subject,
            subdomain_bucket=bucket,
            target=target,
            max_feature_chars=int(args.max_feature_chars),
            splits=splits,
        )
        if err:
            errors.append(f"{dataset_name}/{config or 'default'}: {err}")
        tasks.extend(rows)
    return tasks, errors


def dedupe(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for task in tasks:
        key = normalized_question_key(task["question"])
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(task)
    return out


def select_tasks(tasks: list[dict[str, Any]], target_total: int, target_per_bucket: int) -> list[dict[str, Any]]:
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_bucket[task["subdomain_bucket"]].append(task)
    selected: list[dict[str, Any]] = []
    seen = set()
    for bucket in ("biology", "chemistry", "medicine", "general_science", "other_science"):
        for task in by_bucket.get(bucket, [])[:target_per_bucket]:
            if task["task_id"] not in seen and len(selected) < target_total:
                selected.append(task)
                seen.add(task["task_id"])
    bucket_order = ("biology", "chemistry", "medicine", "general_science", "other_science")
    cursors = {bucket: target_per_bucket for bucket in bucket_order}
    while len(selected) < target_total:
        added = False
        for bucket in bucket_order:
            rows = by_bucket.get(bucket, [])
            idx = cursors[bucket]
            cursors[bucket] += 1
            if idx < len(rows):
                task = rows[idx]
                if task["task_id"] not in seen:
                    selected.append(task)
                    seen.add(task["task_id"])
                    added = True
                    if len(selected) >= target_total:
                        break
        if not added:
            break
    return selected


def build_candidates(tasks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    tournaments: list[dict[str, Any]] = []
    for task in tasks:
        candidate_uids = []
        labels = []
        for letter, option_text in task["options"].items():
            uid = f"science_distractor::{task['source_dataset']}::{task['source_subject']}::{task['dataset_index']}::{letter}"
            is_correct = letter == task["answer_key"]
            candidates.append({
                "candidate_uid": uid,
                "task_id": task["task_id"],
                "source_dataset": task["source_dataset"],
                "source_subject": task["source_subject"],
                "subdomain_bucket": task["subdomain_bucket"],
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
            "source_dataset": task["source_dataset"],
            "source_subject": task["source_subject"],
            "subdomain_bucket": task["subdomain_bucket"],
            "question": task["question"],
            "answer_key": task["answer_key"],
            "n_options": len(task["options"]),
            "candidate_uids": candidate_uids,
            "labels": labels,
        })
    return candidates, tournaments


def summarize(tasks: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [len(feature_text(row["question"], row["option_letter"], row["option_text"])) for row in candidates]
    return {
        "n_questions": len(tasks),
        "n_candidates": len(candidates),
        "random_top1_baseline": float(mean(1.0 / len(task["options"]) for task in tasks)) if tasks else 0.0,
        "dataset_breakdown": dict(Counter(task["source_dataset"] for task in tasks)),
        "subject_breakdown": dict(Counter(f"{task['source_dataset']}::{task['source_subject']}" for task in tasks)),
        "subdomain_breakdown": dict(Counter(task["subdomain_bucket"] for task in tasks)),
        "option_count_breakdown": {str(k): v for k, v in sorted(Counter(len(task["options"]) for task in tasks).items())},
        "prompt_length_stats": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": float(mean(lengths)) if lengths else 0.0,
        },
    }


def verdict(summary: dict[str, Any]) -> str:
    n = int(summary["n_questions"])
    buckets = summary["subdomain_breakdown"]
    ge15 = sum(1 for count in buckets.values() if int(count) >= 15)
    ge10 = sum(1 for count in buckets.values() if int(count) >= 10)
    if n >= 60 and ge15 >= 3:
        return "READY"
    if n >= 30 and ge10 >= 2:
        return "PARTIAL"
    return "BLOCKED"


def write_md(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# Science Natural Distractor Set",
        "",
        f"SCIENCE_DISTRACTOR_SET_VERDICT = {payload['science_distractor_set_verdict']}",
        "",
        f"- n_questions: `{s['n_questions']}`",
        f"- n_candidates: `{s['n_candidates']}`",
        f"- random_top1_baseline: `{s['random_top1_baseline']}`",
        f"- dataset_breakdown: `{s['dataset_breakdown']}`",
        f"- subdomain_breakdown: `{s['subdomain_breakdown']}`",
        f"- option_count_breakdown: `{s['option_count_breakdown']}`",
        f"- prompt_length_stats: `{s['prompt_length_stats']}`",
        f"- dataset_load_errors: `{payload.get('dataset_load_errors', [])}`",
        "",
        "## Tasks",
        "",
    ]
    for task in payload.get("tasks", [])[:160]:
        lines.append(
            f"- `{task['task_id']}` bucket=`{task['subdomain_bucket']}` "
            f"dataset=`{task['source_dataset']}` subject=`{task['source_subject']}` answer=`{task['answer_key']}`"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    all_tasks, errors = collect_all(args)
    tasks = select_tasks(dedupe(all_tasks), int(args.target_total), int(args.target_per_bucket))
    candidates, tournaments = build_candidates(tasks)
    summary = summarize(tasks, candidates)
    v = verdict(summary)
    payload = {
        "science_distractor_set_verdict": v,
        "summary": {"SCIENCE_DISTRACTOR_SET_VERDICT": v, **summary},
        "dataset_load_errors": errors,
        "tasks": tasks,
        "candidates": candidates,
        "tournaments": tournaments,
        "outputs": {"json": repo_path(args.output), "md": repo_path(args.output_md)},
    }
    write_json(Path(args.output), payload)
    write_md(Path(args.output_md), payload)
    print(f"SCIENCE_DISTRACTOR_SET_VERDICT = {v}")
    print(f"n_questions = {summary['n_questions']}")
    print(f"subdomain_breakdown = {summary['subdomain_breakdown']}")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output_md}")
    if v == "BLOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
