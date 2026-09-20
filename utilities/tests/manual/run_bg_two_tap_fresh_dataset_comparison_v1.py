"""Fresh same-content comparison for old BG taps vs new layer-native two taps.

This runner builds fresh domain candidate tournaments from public datasets,
using non-42 reseeds for previously sampled-but-not-exhausted sources and
additional new datasets for the same domains. It captures local Ouro-RLTT
hidden-state features at layers 24/36/47 and evaluates old frozen BG taps and
the new copied/trained two-tap family on exactly the same candidate pairs.

It does not train Ouro, modify checkpoints/tokenizers, update tap registries,
run wrapper/local-agent or Hunter-Seeker code, apply steering, or change
production routing.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
PROBE_ROOT = PROJECT_ROOT / "opi/taps/probes"
OUT_ROOT = PROBE_ROOT / "bg_two_tap_fresh_dataset_comparison_v1_2026-05-30"
DATASET_JSON = OUT_ROOT / "fresh_candidate_dataset.json"
DATASET_MD = OUT_ROOT / "fresh_candidate_dataset.md"
FEATURES_PT = OUT_ROOT / "fresh_candidate_features.pt"
PAIR_ROWS_CSV = OUT_ROOT / "fresh_comparison_pair_rows.csv"
SUMMARY_JSON = OUT_ROOT / "fresh_comparison_summary.json"
SUMMARY_MD = OUT_ROOT / "fresh_comparison_summary.md"
ARTIFACT_PT = OUT_ROOT / "two_tap_fresh_dataset_comparison_v1.pt"
DOC_MD = PROJECT_ROOT / "docs/evaluator/bg_two_tap_fresh_dataset_comparison_v1.md"

OLD_REASONING_JSON = PROBE_ROOT / "reasoning_natural_distractor_set_2026-05-17.json"
OLD_SCIENCE_JSON = PROBE_ROOT / "science_natural_distractor_set_2026-05-17.json"
OLD_GSM8K_PT = PROBE_ROOT / "clean_gsm8k_expanded_tap_features_2026-05-16.pt"
CONSTRAINED_TWO_TAP_PT = (
    PROBE_ROOT
    / "bg_layer_native_two_tap_constrained_train_v1_2026-05-30"
    / "layer_native_two_tap_constrained_train_v1.pt"
)
TARGETED_REHOST_PT = (
    PROBE_ROOT
    / "bg_layer_native_two_tap_targeted_rehost_diagnostic_v1_2026-05-30"
    / "targeted_rehost_diagnostic_v1.pt"
)

MODEL_PATH = PROJECT_ROOT / "shared/models/ouro_rltt_local"
LAYER_CONFIGS = ("24_L4", "36_L4", "47_L4")
LETTERS = tuple("ABCDE")

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "shared/hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_ROOT / "shared/hf_cache/hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "shared/hf_cache/datasets"))

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from math_bg_probe_lib import (  # noqa: E402
    answers_equal,
    capture_pooled_taps,
    config_vector,
    extract_gold_answer,
    perturb_answer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--target-per-domain", type=int, default=24)
    parser.add_argument("--max-feature-chars", type=int, default=3000)
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--force-capture", action="store_true")
    parser.add_argument("--include-targeted-rehost-diagnostic", action="store_true")
    return parser.parse_args()


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(p)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_md(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, torch.Tensor):
        return {"tensor_shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (set, tuple)):
        return list(value)
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def normalize_options(raw_labels: Iterable[Any], raw_texts: Iterable[Any], answer_key: Any) -> tuple[dict[str, str], str] | None:
    labels = [str(x).strip().upper() for x in raw_labels]
    texts = [clean_text(x) for x in raw_texts]
    options: dict[str, str] = {}
    original_to_new: dict[str, str] = {}
    for idx, text in enumerate(texts[: len(LETTERS)]):
        if not text:
            return None
        new = LETTERS[idx]
        old = labels[idx] if idx < len(labels) and labels[idx] else new
        options[new] = text
        original_to_new[old] = new
        original_to_new[new] = new
        original_to_new[str(idx)] = new
        original_to_new[str(idx + 1)] = new
    if len(options) < 3:
        return None
    raw_answer = str(answer_key if answer_key is not None else "").strip().upper()
    if raw_answer.isdigit():
        n = int(raw_answer)
        raw_answer = original_to_new.get(str(n), raw_answer)
    answer = original_to_new.get(raw_answer, raw_answer)
    if answer not in options:
        return None
    return options, answer


def mcq_feature_text(domain: str, question: str, letter: str, option_text: str) -> str:
    prefix = "Science question" if domain == "science" else "Question"
    return f"{prefix}:\n{question}\n\nCandidate answer:\n{letter}. {option_text}"


def valid_feature_text(text: str, max_chars: int) -> bool:
    return bool(text.strip()) and len(text) <= max_chars


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def previous_reasoning_ids() -> set[str]:
    payload = load_json_if_exists(OLD_REASONING_JSON)
    return {str(row.get("task_id")) for row in payload.get("tasks") or [] if row.get("task_id")}


def previous_science_ids() -> set[str]:
    payload = load_json_if_exists(OLD_SCIENCE_JSON)
    return {str(row.get("task_id")) for row in payload.get("tasks") or [] if row.get("task_id")}


def previous_gsm8k_indices() -> set[int]:
    if not OLD_GSM8K_PT.exists():
        return set()
    try:
        payload = torch.load(OLD_GSM8K_PT, map_location="cpu", weights_only=False)
    except Exception:
        return set()
    out = set()
    for row in payload.get("records") or []:
        if row.get("source") == "gsm8k":
            try:
                out.add(int(row.get("dataset_index")))
            except Exception:
                pass
    return out


def make_mcq_task(
    *,
    domain: str,
    source_dataset: str,
    source_status: str,
    task_id: str,
    dataset_index: int,
    question: str,
    options: dict[str, str],
    answer_key: str,
    max_feature_chars: int,
    source_subject: str = "",
) -> dict[str, Any] | None:
    if not question or answer_key not in options:
        return None
    for letter, text in options.items():
        if not valid_feature_text(mcq_feature_text(domain, question, letter, text), max_feature_chars):
            return None
    return {
        "task_id": task_id,
        "domain": domain,
        "source_dataset": source_dataset,
        "source_subject": source_subject,
        "source_status": source_status,
        "dataset_index": dataset_index,
        "question": question,
        "options": options,
        "answer_key": answer_key,
        "task_type": "mcq",
    }


def task_candidates(task: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if task["task_type"] == "mcq":
        for letter, option_text in task["options"].items():
            rows.append(
                {
                    "candidate_uid": f"{task['domain']}::{task['source_dataset']}::{task['task_id']}::{letter}",
                    "task_id": task["task_id"],
                    "domain": task["domain"],
                    "source_dataset": task["source_dataset"],
                    "source_subject": task.get("source_subject", ""),
                    "source_status": task["source_status"],
                    "task_type": "mcq",
                    "candidate_label": letter,
                    "is_correct": letter == task["answer_key"],
                    "feature_text": mcq_feature_text(task["domain"], task["question"], letter, option_text),
                }
            )
    elif task["task_type"] == "math":
        for idx, cand in enumerate(task["candidate_texts"]):
            rows.append(
                {
                    "candidate_uid": f"math::{task['source_dataset']}::{task['task_id']}::{idx}",
                    "task_id": task["task_id"],
                    "domain": "math_simple_arithmetic",
                    "source_dataset": task["source_dataset"],
                    "source_subject": task.get("source_subject", ""),
                    "source_status": task["source_status"],
                    "task_type": "math",
                    "candidate_label": str(idx),
                    "is_correct": bool(task["labels"][idx]),
                    "feature_text": cand,
                }
            )
    elif task["task_type"] == "coding":
        for idx, cand in enumerate(task["candidate_texts"]):
            rows.append(
                {
                    "candidate_uid": f"coding::{task['source_dataset']}::{task['task_id']}::{idx}",
                    "task_id": task["task_id"],
                    "domain": "coding",
                    "source_dataset": task["source_dataset"],
                    "source_subject": task.get("source_subject", ""),
                    "source_status": task["source_status"],
                    "task_type": "coding",
                    "candidate_label": str(idx),
                    "is_correct": bool(task["labels"][idx]),
                    "feature_text": cand,
                }
            )
    return rows


def collect_reasoning(seed: int, target: int, max_feature_chars: int) -> tuple[list[dict[str, Any]], list[str]]:
    rng = random.Random(seed)
    prev = previous_reasoning_ids()
    tasks: list[dict[str, Any]] = []
    errors: list[str] = []
    existing_target = max(1, target // 2)
    new_target = target - existing_target

    existing_specs = [
        ("ai2_arc", "ARC-Challenge", "validation", "ai2_arc_challenge", "ARC-Challenge"),
        ("openbookqa", "main", "validation", "openbookqa", "OpenBookQA"),
        ("commonsense_qa", None, "validation", "commonsense_qa", "CommonsenseQA"),
    ]
    for dataset_name, config, split, source_dataset, id_prefix in existing_specs:
        try:
            ds = load_dataset(dataset_name, config, split=split, cache_dir=os.environ["HF_DATASETS_CACHE"]) if config else load_dataset(dataset_name, split=split, cache_dir=os.environ["HF_DATASETS_CACHE"])
            indices = list(range(len(ds)))
            rng.shuffle(indices)
            for idx in indices:
                row = ds[idx]
                if source_dataset == "ai2_arc":
                    choices = row["choices"]
                    norm = normalize_options(choices.get("label", []), choices.get("text", []), row.get("answerKey"))
                    question = clean_text(row.get("question"))
                elif source_dataset == "openbookqa":
                    choices = row["choices"]
                    norm = normalize_options(choices.get("label", []), choices.get("text", []), row.get("answerKey"))
                    question = clean_text(row.get("question_stem"))
                else:
                    choices = row["choices"]
                    norm = normalize_options(choices.get("label", []), choices.get("text", []), row.get("answerKey"))
                    question = clean_text(row.get("question"))
                if not norm:
                    continue
                task_id = f"{id_prefix}/{idx}"
                if task_id in prev:
                    continue
                task = make_mcq_task(
                    domain="reasoning",
                    source_dataset=source_dataset,
                    source_status="reseed_incomplete_existing",
                    task_id=task_id,
                    dataset_index=idx,
                    question=question,
                    options=norm[0],
                    answer_key=norm[1],
                    max_feature_chars=max_feature_chars,
                )
                if task:
                    tasks.append(task)
                if len([t for t in tasks if t["source_status"] == "reseed_incomplete_existing"]) >= existing_target:
                    break
        except Exception as exc:
            errors.append(f"{dataset_name}/{config or 'default'}: {type(exc).__name__}: {exc}")
        if len([t for t in tasks if t["source_status"] == "reseed_incomplete_existing"]) >= existing_target:
            break

    try:
        ds = load_dataset("Rowan/hellaswag", split="validation", cache_dir=os.environ["HF_DATASETS_CACHE"])
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        for idx in indices:
            row = ds[idx]
            endings = row.get("endings") or []
            label = row.get("label")
            try:
                answer_idx = int(label)
            except Exception:
                continue
            if not (0 <= answer_idx < len(endings) <= len(LETTERS)):
                continue
            question = clean_text(f"{row.get('ctx', '')} {row.get('activity_label', '')}")
            options = {LETTERS[i]: clean_text(text) for i, text in enumerate(endings)}
            task = make_mcq_task(
                domain="reasoning",
                source_dataset="hellaswag",
                source_status="new_dataset",
                task_id=f"HellaSwag/{idx}",
                dataset_index=idx,
                question=question,
                options=options,
                answer_key=LETTERS[answer_idx],
                max_feature_chars=max_feature_chars,
                source_subject=str(row.get("activity_label") or ""),
            )
            if task:
                tasks.append(task)
            if len([t for t in tasks if t["source_status"] == "new_dataset"]) >= new_target:
                break
    except Exception as exc:
        errors.append(f"Rowan/hellaswag: {type(exc).__name__}: {exc}")

    return tasks[:target], errors


def options_from_science_row(row: dict[str, Any], source_dataset: str) -> tuple[dict[str, str], str] | None:
    if {"correct_answer", "distractor1", "distractor2", "distractor3"}.issubset(row):
        items = [(clean_text(row["correct_answer"]), True)] + [
            (clean_text(row.get(f"distractor{i}")), False) for i in (1, 2, 3)
        ]
        if any(not text for text, _ in items):
            return None
        # Stable per-question shuffle; source labels should not leak answer position.
        rng = random.Random(abs(hash(clean_text(row.get("question")))) % (2**32))
        rng.shuffle(items)
        options: dict[str, str] = {}
        answer = ""
        for idx, (text, correct) in enumerate(items):
            letter = LETTERS[idx]
            options[letter] = text
            if correct:
                answer = letter
        return options, answer
    choices = row.get("choices")
    if isinstance(choices, dict):
        return normalize_options(choices.get("label", []), choices.get("text", []), row.get("answerKey", row.get("answer")))
    if isinstance(choices, list):
        texts = [item.get("text", item) if isinstance(item, dict) else item for item in choices]
        labels = [item.get("label", LETTERS[i]) if isinstance(item, dict) else LETTERS[i] for i, item in enumerate(choices[: len(LETTERS)])]
        return normalize_options(labels, texts, row.get("answer", row.get("answerKey")))
    if isinstance(row.get("answer"), int) and isinstance(row.get("choices"), list):
        return normalize_options(list(LETTERS), row["choices"], row["answer"])
    return None


def collect_science(seed: int, target: int, max_feature_chars: int) -> tuple[list[dict[str, Any]], list[str]]:
    rng = random.Random(seed + 17)
    prev = previous_science_ids()
    tasks: list[dict[str, Any]] = []
    errors: list[str] = []
    existing_target = max(1, target // 2)
    new_target = target - existing_target

    existing_specs = [
        ("allenai/sciq", None, "validation", "sciq", "sciq", "general_science"),
        ("cais/mmlu", "high_school_biology", "test", "mmlu", "high_school_biology", "biology"),
        ("cais/mmlu", "high_school_chemistry", "test", "mmlu", "high_school_chemistry", "chemistry"),
        ("cais/mmlu", "college_medicine", "test", "mmlu", "college_medicine", "medicine"),
    ]
    for dataset_name, config, split, source_dataset, subject, bucket in existing_specs:
        try:
            ds = load_dataset(dataset_name, config, split=split, cache_dir=os.environ["HF_DATASETS_CACHE"]) if config else load_dataset(dataset_name, split=split, cache_dir=os.environ["HF_DATASETS_CACHE"])
            indices = list(range(len(ds)))
            rng.shuffle(indices)
            for idx in indices:
                row = dict(ds[idx])
                norm = options_from_science_row(row, source_dataset)
                question = clean_text(row.get("question", row.get("input", row.get("stem", ""))))
                if not norm or not question:
                    continue
                task_id = f"{source_dataset}/{subject}/{idx}".replace(" ", "_")
                if task_id in prev:
                    continue
                task = make_mcq_task(
                    domain="science",
                    source_dataset=source_dataset,
                    source_subject=subject or bucket,
                    source_status="reseed_incomplete_existing",
                    task_id=task_id,
                    dataset_index=idx,
                    question=question,
                    options=norm[0],
                    answer_key=norm[1],
                    max_feature_chars=max_feature_chars,
                )
                if task:
                    tasks.append(task)
                if len([t for t in tasks if t["source_status"] == "reseed_incomplete_existing"]) >= existing_target:
                    break
        except Exception as exc:
            errors.append(f"{dataset_name}/{config or 'default'}: {type(exc).__name__}: {exc}")
        if len([t for t in tasks if t["source_status"] == "reseed_incomplete_existing"]) >= existing_target:
            break

    try:
        ds = load_dataset("allenai/qasc", split="validation", cache_dir=os.environ["HF_DATASETS_CACHE"])
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        for idx in indices:
            row = dict(ds[idx])
            norm = options_from_science_row(row, "qasc")
            question = clean_text(row.get("question"))
            if not norm or not question:
                continue
            task = make_mcq_task(
                domain="science",
                source_dataset="qasc",
                source_subject="qasc",
                source_status="new_dataset",
                task_id=f"QASC/{idx}",
                dataset_index=idx,
                question=question,
                options=norm[0],
                answer_key=norm[1],
                max_feature_chars=max_feature_chars,
            )
            if task:
                tasks.append(task)
            if len([t for t in tasks if t["source_status"] == "new_dataset"]) >= new_target:
                break
    except Exception as exc:
        errors.append(f"allenai/qasc: {type(exc).__name__}: {exc}")

    return tasks[:target], errors


def math_candidate_text(question: str, answer: str) -> str:
    return f"Problem: {question}\n\nFinal answer: {answer}"


def collect_math(seed: int, target: int, max_feature_chars: int) -> tuple[list[dict[str, Any]], list[str]]:
    rng = random.Random(seed + 31)
    tasks: list[dict[str, Any]] = []
    errors: list[str] = []
    prev_gsm = previous_gsm8k_indices()
    existing_target = max(1, target // 2)
    new_target = target - existing_target

    try:
        ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=os.environ["HF_DATASETS_CACHE"])
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        for idx in indices:
            if idx in prev_gsm:
                continue
            row = ds[idx]
            gold = extract_gold_answer("gsm8k", row["answer"])
            if not gold:
                continue
            wrongs: list[str] = []
            for _ in range(8):
                wrong = perturb_answer(gold, rng)
                if wrong and not answers_equal(wrong, gold) and wrong not in wrongs:
                    wrongs.append(wrong)
                if len(wrongs) >= 3:
                    break
            if len(wrongs) < 2:
                continue
            texts = [math_candidate_text(row["question"], gold)] + [math_candidate_text(row["question"], w) for w in wrongs[:3]]
            if any(not valid_feature_text(text, max_feature_chars) for text in texts):
                continue
            tasks.append(
                {
                    "task_id": f"GSM8K/{idx}",
                    "domain": "math_simple_arithmetic",
                    "source_dataset": "gsm8k",
                    "source_subject": "gsm8k",
                    "source_status": "reseed_incomplete_existing",
                    "dataset_index": idx,
                    "task_type": "math",
                    "question": row["question"],
                    "gold_answer": gold,
                    "candidate_texts": texts,
                    "labels": [True] + [False] * len(wrongs[:3]),
                }
            )
            if len([t for t in tasks if t["source_status"] == "reseed_incomplete_existing"]) >= existing_target:
                break
    except Exception as exc:
        errors.append(f"openai/gsm8k: {type(exc).__name__}: {exc}")

    try:
        ds = load_dataset("ChilleD/SVAMP", split="test", cache_dir=os.environ["HF_DATASETS_CACHE"])
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        for idx in indices:
            row = dict(ds[idx])
            question = clean_text(f"{row.get('Body', '')} {row.get('Question', row.get('question', ''))}")
            gold = clean_text(row.get("Answer", row.get("answer")))
            if not question or not gold:
                continue
            wrongs = []
            for _ in range(8):
                wrong = perturb_answer(gold, rng)
                if wrong and not answers_equal(wrong, gold) and wrong not in wrongs:
                    wrongs.append(wrong)
                if len(wrongs) >= 3:
                    break
            if len(wrongs) < 2:
                continue
            texts = [math_candidate_text(question, gold)] + [math_candidate_text(question, w) for w in wrongs[:3]]
            if any(not valid_feature_text(text, max_feature_chars) for text in texts):
                continue
            tasks.append(
                {
                    "task_id": f"SVAMP/{idx}",
                    "domain": "math_simple_arithmetic",
                    "source_dataset": "svamp",
                    "source_subject": "svamp",
                    "source_status": "new_dataset",
                    "dataset_index": idx,
                    "task_type": "math",
                    "question": question,
                    "gold_answer": gold,
                    "candidate_texts": texts,
                    "labels": [True] + [False] * len(wrongs[:3]),
                }
            )
            if len([t for t in tasks if t["source_status"] == "new_dataset"]) >= new_target:
                break
    except Exception as exc:
        errors.append(f"ChilleD/SVAMP: {type(exc).__name__}: {exc}")

    return tasks[:target], errors


def corrupt_code(code: str, rng: random.Random) -> str:
    replacements = [
        ("==", "!="),
        ("!=", "=="),
        (">=", "<"),
        ("<=", ">"),
        ("+", "-"),
        ("-", "+"),
        ("return True", "return False"),
        ("return False", "return True"),
    ]
    candidates = []
    for old, new in replacements:
        if old in code:
            candidates.append(code.replace(old, new, 1))
    if candidates:
        return rng.choice(candidates)
    if "return " in code:
        return re.sub(r"return\s+([^\n]+)", "return None", code, count=1)
    return code + "\n# corrupted negative candidate\n"


def collect_coding(seed: int, target: int, max_feature_chars: int) -> tuple[list[dict[str, Any]], list[str]]:
    rng = random.Random(seed + 47)
    tasks: list[dict[str, Any]] = []
    errors: list[str] = []
    per_source = max(1, target // 2)

    try:
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test", cache_dir=os.environ["HF_DATASETS_CACHE"])
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        for idx in indices:
            row = dict(ds[idx])
            prompt = clean_text(row.get("prompt", row.get("text", "")))
            code = str(row.get("code") or "")
            if not prompt or not code:
                continue
            wrong1 = corrupt_code(code, rng)
            wrong2 = corrupt_code(code[::-1][::-1], rng)
            texts = [
                f"Programming task:\n{prompt}\n\nCandidate solution:\n{code}",
                f"Programming task:\n{prompt}\n\nCandidate solution:\n{wrong1}",
                f"Programming task:\n{prompt}\n\nCandidate solution:\n{wrong2}",
            ]
            if any(not valid_feature_text(text, max_feature_chars) for text in texts):
                continue
            tasks.append(
                {
                    "task_id": f"MBPP/{row.get('task_id', idx)}",
                    "domain": "coding",
                    "source_dataset": "mbpp_sanitized",
                    "source_subject": "python",
                    "source_status": "new_dataset",
                    "dataset_index": idx,
                    "task_type": "coding",
                    "prompt": prompt,
                    "candidate_texts": texts,
                    "labels": [True, False, False],
                    "label_source": "reference_vs_synthetic_corruption",
                }
            )
            if len([t for t in tasks if t["source_dataset"] == "mbpp_sanitized"]) >= per_source:
                break
    except Exception as exc:
        errors.append(f"google-research-datasets/mbpp: {type(exc).__name__}: {exc}")

    try:
        ds = load_dataset("openai/openai_humaneval", split="test", cache_dir=os.environ["HF_DATASETS_CACHE"])
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        for idx in indices:
            row = dict(ds[idx])
            prompt = str(row.get("prompt") or "")
            canonical = str(row.get("canonical_solution") or "")
            if not prompt or not canonical:
                continue
            correct = prompt + canonical
            wrong1 = prompt + corrupt_code(canonical, rng)
            wrong2 = prompt + corrupt_code(canonical[::-1][::-1], rng)
            texts = [
                f"Programming task and candidate solution:\n{correct}",
                f"Programming task and candidate solution:\n{wrong1}",
                f"Programming task and candidate solution:\n{wrong2}",
            ]
            if any(not valid_feature_text(text, max_feature_chars) for text in texts):
                continue
            tasks.append(
                {
                    "task_id": f"HumanEval/{row.get('task_id', idx)}",
                    "domain": "coding",
                    "source_dataset": "humaneval",
                    "source_subject": "python",
                    "source_status": "new_dataset",
                    "dataset_index": idx,
                    "task_type": "coding",
                    "prompt": prompt,
                    "candidate_texts": texts,
                    "labels": [True, False, False],
                    "label_source": "reference_vs_synthetic_corruption",
                }
            )
            if len([t for t in tasks if t["source_dataset"] == "humaneval"]) >= target - per_source:
                break
    except Exception as exc:
        errors.append(f"openai/openai_humaneval: {type(exc).__name__}: {exc}")

    return tasks[:target], errors


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, Any]] = []
    errors: list[str] = []
    for domain, fn in (
        ("reasoning", collect_reasoning),
        ("science", collect_science),
        ("math_simple_arithmetic", collect_math),
        ("coding", collect_coding),
    ):
        rows, errs = fn(int(args.seed), int(args.target_per_domain), int(args.max_feature_chars))
        errors.extend([f"{domain}: {err}" for err in errs])
        tasks.extend(rows)

    candidates: list[dict[str, Any]] = []
    eval_sets: list[dict[str, Any]] = []
    for task in tasks:
        rows = task_candidates(task)
        if not rows:
            continue
        candidate_uids = [r["candidate_uid"] for r in rows]
        labels = ["correct" if r["is_correct"] else "incorrect" for r in rows]
        candidates.extend(rows)
        eval_sets.append(
            {
                "task_id": task["task_id"],
                "domain": task["domain"],
                "source_dataset": task["source_dataset"],
                "source_status": task["source_status"],
                "task_type": task["task_type"],
                "candidate_uids": candidate_uids,
                "labels": labels,
            }
        )

    summary = {
        "seed": int(args.seed),
        "target_per_domain": int(args.target_per_domain),
        "task_count": len(eval_sets),
        "candidate_count": len(candidates),
        "domain_counts": dict(Counter(t["domain"] for t in eval_sets)),
        "source_status_counts": dict(Counter(t["source_status"] for t in eval_sets)),
        "source_dataset_counts": dict(Counter(t["source_dataset"] for t in eval_sets)),
        "dataset_errors": errors,
        "existing_reuse_policy": "Only sampled-again sources whose prior local artifacts were subset samples; previous task IDs were excluded where available. Seed 42 was not used.",
    }
    verdict = "READY" if len(eval_sets) >= 40 and len(summary["domain_counts"]) >= 3 else "PARTIAL" if eval_sets else "BLOCKED"
    payload = {
        "BG_TWO_TAP_FRESH_DATASET_BUILD_VERDICT": verdict,
        "summary": summary,
        "tasks": tasks,
        "candidates": candidates,
        "eval_sets": eval_sets,
    }
    write_json(DATASET_JSON, payload)
    write_dataset_md(DATASET_MD, payload)
    return payload


def write_dataset_md(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# Two-Tap Fresh Candidate Dataset",
        "",
        f"BG_TWO_TAP_FRESH_DATASET_BUILD_VERDICT = {payload['BG_TWO_TAP_FRESH_DATASET_BUILD_VERDICT']}",
        "",
        f"- seed: `{s['seed']}`",
        f"- task_count: `{s['task_count']}`",
        f"- candidate_count: `{s['candidate_count']}`",
        f"- domain_counts: `{s['domain_counts']}`",
        f"- source_status_counts: `{s['source_status_counts']}`",
        f"- source_dataset_counts: `{s['source_dataset_counts']}`",
        f"- existing_reuse_policy: `{s['existing_reuse_policy']}`",
        f"- dataset_errors: `{s['dataset_errors']}`",
        "",
        "## Tasks",
        "",
    ]
    for row in payload.get("eval_sets", [])[:160]:
        lines.append(
            f"- `{row['task_id']}` domain=`{row['domain']}` source=`{row['source_dataset']}` "
            f"status=`{row['source_status']}` candidates=`{len(row['candidate_uids'])}`"
        )
    write_md(path, lines)


def capture_features(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    if FEATURES_PT.exists() and not args.force_capture:
        return torch.load(FEATURES_PT, map_location="cpu", weights_only=False)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    candidates = list(payload.get("candidates") or [])
    texts = [row["feature_text"] for row in candidates]
    start = time.time()
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    pooled = capture_pooled_taps(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        max_length=int(args.max_length),
        device=device,
        report_every=int(args.report_every),
    ).cpu()
    elapsed = time.time() - start
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    features = []
    for idx, row in enumerate(candidates):
        clean = {k: v for k, v in row.items() if k != "feature_text"}
        features.append({**clean, "feature_text": row["feature_text"], "pooled": pooled[idx].to(torch.float32).contiguous()})
    out = {
        "meta": {
            "BG_TWO_TAP_FRESH_FEATURE_CAPTURE_VERDICT": "READY",
            "elapsed_seconds": elapsed,
            "candidate_count": len(features),
            "tap_layers": [24, 36, 47],
            "feature_configs": list(LAYER_CONFIGS),
            "model_path": rel(args.model_path),
        },
        "candidate_features": features,
        "eval_sets": payload.get("eval_sets", []),
        "dataset_summary": payload.get("summary", {}),
    }
    torch.save(out, FEATURES_PT)
    return out


def tensor_weight(candidate: dict[str, Any]) -> torch.Tensor | None:
    weight = candidate.get("weight")
    if isinstance(weight, torch.Tensor):
        return weight.detach().cpu().to(torch.float32).flatten()
    state = candidate.get("state_dict") or {}
    if isinstance(state, dict):
        value = state.get("linear.weight") or state.get("weight") or state.get("score.weight")
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().to(torch.float32).flatten()
    return None


def load_old_bg_taps() -> list[dict[str, Any]]:
    # This import intentionally happens after dataset acquisition because the
    # hidden-origin common module sets offline env vars for old cached probes.
    from run_bg_two_tap_full_readiness_v1 import source_candidates

    out = []
    for row in source_candidates():
        if row.get("candidate_family") != "source_old_content":
            continue
        if row.get("source_run") not in {"old_registry", "old_mixed_domain"}:
            continue
        if str(row.get("target_config")) not in LAYER_CONFIGS:
            continue
        weight = tensor_weight(row)
        if not isinstance(weight, torch.Tensor):
            continue
        out.append({**row, "eval_family": "old_bg_tap"})
    return out


def load_new_two_taps(include_targeted: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = torch.load(CONSTRAINED_TWO_TAP_PT, map_location="cpu", weights_only=False)
    candidates = []
    for row in payload.get("candidates") or []:
        if str(row.get("target_config")) not in LAYER_CONFIGS:
            continue
        family = str(row.get("candidate_family") or "")
        if family == "layer_native_two_tap_targeted_rehost":
            continue
        # Primary means exactly the two requested tap anchors. Some diagnostic
        # transplant candidates rehost other old-content heads, such as
        # MIX_CODE_SCIENCE, under a two-tap identity. Those are useful probes,
        # but they are not evidence that only coding_reasoning and
        # mixed_objective_all are sufficient.
        name = str(row.get("candidate_name") or "")
        old_source = str(row.get("old_source") or "")
        if any(token in name for token in ("old_content_full", "old_content_residual")):
            continue
        if old_source and not any(anchor in old_source for anchor in ("MIX_CODE_REASONING", "MIX_OBJECTIVE_ALL")):
            continue
        weight = tensor_weight(row)
        if not isinstance(weight, torch.Tensor):
            continue
        candidates.append({**row, "eval_family": "new_two_tap_primary"})
    bundles = list(payload.get("bundles") or [])
    if include_targeted and TARGETED_REHOST_PT.exists():
        diag = torch.load(TARGETED_REHOST_PT, map_location="cpu", weights_only=False)
        for row in diag.get("candidates") or []:
            if str(row.get("target_config")) not in LAYER_CONFIGS:
                continue
            weight = tensor_weight(row)
            if isinstance(weight, torch.Tensor):
                candidates.append({**row, "eval_family": "targeted_rehost_diagnostic"})
    return candidates, bundles


def feature_map(row: dict[str, Any]) -> dict[str, torch.Tensor]:
    pooled = row.get("pooled")
    if not isinstance(pooled, torch.Tensor):
        return {}
    out = {}
    for config in LAYER_CONFIGS:
        try:
            out[config] = config_vector(pooled, config).detach().cpu().to(torch.float32).flatten()
        except Exception:
            pass
    return out


def build_pairs(feature_payload: dict[str, Any]) -> list[dict[str, Any]]:
    by_uid = {str(row.get("candidate_uid")): row for row in feature_payload.get("candidate_features") or []}
    maps = {uid: feature_map(row) for uid, row in by_uid.items()}
    pairs: list[dict[str, Any]] = []
    for task in feature_payload.get("eval_sets") or []:
        uids = [str(x) for x in task.get("candidate_uids") or []]
        labels = [str(x).lower() for x in task.get("labels") or []]
        if len(uids) != len(labels):
            continue
        correct = [idx for idx, label in enumerate(labels) if label == "correct"]
        wrong = [idx for idx, label in enumerate(labels) if label != "correct"]
        for ci in correct:
            for wi in wrong:
                c_uid = uids[ci]
                w_uid = uids[wi]
                feats = {}
                for config in LAYER_CONFIGS:
                    cv = maps.get(c_uid, {}).get(config)
                    wv = maps.get(w_uid, {}).get(config)
                    if isinstance(cv, torch.Tensor) and isinstance(wv, torch.Tensor):
                        feats[config] = cv - wv
                if feats:
                    pairs.append(
                        {
                            "pair_id": f"{task.get('task_id')}::{ci}>{wi}",
                            "task_id": task.get("task_id"),
                            "domain": task.get("domain"),
                            "source_dataset": task.get("source_dataset"),
                            "source_status": task.get("source_status"),
                            "task_type": task.get("task_type"),
                            "preferred_uid": c_uid,
                            "rejected_uid": w_uid,
                            "features": feats,
                        }
                    )
    return pairs


def score_vector(weight: torch.Tensor, arch: str, diffs: torch.Tensor) -> torch.Tensor:
    x = diffs.to(torch.float32)
    if arch == "AntisymLinear":
        x = F.layer_norm(x, (x.shape[-1],))
    w = weight.detach().cpu().to(torch.float32).flatten()
    return x @ w


def accuracy_from_scores(scores: Sequence[float]) -> float:
    if not scores:
        return float("nan")
    correct = sum(1.0 if s > 0 else 0.5 if s == 0 else 0.0 for s in scores)
    return correct / len(scores)


def finite_mean(vals: Iterable[Any]) -> float:
    xs = []
    for val in vals:
        try:
            x = float(val)
        except Exception:
            continue
        if math.isfinite(x):
            xs.append(x)
    return float(mean(xs)) if xs else float("nan")


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def evaluate_single_candidates(candidates: Sequence[dict[str, Any]], pairs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pairs_by_config: dict[str, list[tuple[int, torch.Tensor]]] = defaultdict(list)
    for idx, pair in enumerate(pairs):
        for config, diff in (pair.get("features") or {}).items():
            if config in LAYER_CONFIGS and isinstance(diff, torch.Tensor):
                pairs_by_config[config].append((idx, diff))
    for cand in candidates:
        config = str(cand.get("target_config"))
        arch = str(cand.get("architecture"))
        weight = tensor_weight(cand)
        if not isinstance(weight, torch.Tensor) or config not in pairs_by_config:
            continue
        indexed = pairs_by_config[config]
        if not indexed:
            continue
        diffs = torch.stack([d for _i, d in indexed], dim=0)
        scores_t = score_vector(weight, arch, diffs)
        scores = [float(x) for x in scores_t.tolist()]
        rows.append(
            {
                "candidate_name": cand.get("candidate_name"),
                "candidate_family": cand.get("candidate_family"),
                "eval_family": cand.get("eval_family"),
                "source_run": cand.get("source_run"),
                "source_family": cand.get("source_family"),
                "tap_role": cand.get("tap_role"),
                "recipe": cand.get("recipe"),
                "target_config": config,
                "architecture": arch,
                "pair_count": len(scores),
                "pairwise_accuracy": accuracy_from_scores(scores),
                "mean_margin": finite_mean(scores),
                "scores_by_pair": {int(idx): scores[pos] for pos, (idx, _d) in enumerate(indexed)},
            }
        )
    return rows


def normalize_scores(scores: Sequence[float]) -> list[float]:
    xs = [float(x) for x in scores]
    mu = finite_mean(xs)
    var = finite_mean((x - mu) ** 2 for x in xs)
    sd = math.sqrt(var) if math.isfinite(var) and var > 1e-12 else 1.0
    return [(x - mu) / sd for x in xs]


def evaluate_bundles(candidates: Sequence[dict[str, Any]], bundles: Sequence[dict[str, Any]], pairs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {str(c.get("candidate_name")): c for c in candidates}
    member_rows = evaluate_single_candidates(candidates, pairs)
    scores_by_name = {str(row.get("candidate_name")): row.get("scores_by_pair") or {} for row in member_rows}
    rows = []
    for bundle in bundles:
        members = [str(x) for x in bundle.get("members") or []]
        present = [name for name in members if name in by_name and name in scores_by_name]
        if not present:
            continue
        pair_ids = sorted(set.intersection(*(set(scores_by_name[name]) for name in present)))
        if not pair_ids:
            continue
        components = []
        for name in present:
            raw = [safe_float(scores_by_name[name][idx], 0.0) for idx in pair_ids]
            components.append(normalize_scores(raw))
        scores = [finite_mean(component[i] for component in components) for i in range(len(pair_ids))]
        rows.append(
            {
                "candidate_name": bundle.get("bundle_name"),
                "candidate_family": bundle.get("bundle_family", "layer_native_bundle"),
                "eval_family": "new_two_tap_primary",
                "source_run": "",
                "source_family": "layer_native_bundle",
                "tap_role": "bundle",
                "recipe": bundle.get("recipe"),
                "target_config": "24_L4+36_L4+47_L4",
                "architecture": bundle.get("architecture"),
                "pair_count": len(scores),
                "pairwise_accuracy": accuracy_from_scores(scores),
                "mean_margin": finite_mean(scores),
                "scores_by_pair": {int(idx): scores[pos] for pos, idx in enumerate(pair_ids)},
            }
        )
    return rows


def summarize_rows(rows: Sequence[dict[str, Any]], pairs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pair_meta = {idx: pair for idx, pair in enumerate(pairs)}
    summary_rows = []
    slices: dict[str, set[int]] = {
        "all": set(range(len(pairs))),
    }
    for key in ("domain", "source_status", "source_dataset"):
        vals = sorted({str(p.get(key)) for p in pairs})
        for val in vals:
            slices[f"{key}::{val}"] = {idx for idx, pair in pair_meta.items() if str(pair.get(key)) == val}

    for slice_name, idxs in slices.items():
        if not idxs:
            continue
        scored = []
        for row in rows:
            score_map = row.get("scores_by_pair") or {}
            vals = [safe_float(score_map[idx]) for idx in idxs if idx in score_map]
            if not vals:
                continue
            scored.append({**{k: v for k, v in row.items() if k != "scores_by_pair"}, "slice_name": slice_name, "slice_pair_count": len(vals), "slice_accuracy": accuracy_from_scores(vals), "slice_mean_margin": finite_mean(vals)})
        old = [r for r in scored if r.get("eval_family") == "old_bg_tap"]
        new_primary = [r for r in scored if r.get("eval_family") == "new_two_tap_primary"]
        targeted = [r for r in scored if r.get("eval_family") == "targeted_rehost_diagnostic"]
        best_old = max(old, key=lambda r: safe_float(r.get("slice_accuracy"), -1.0), default={})
        best_new = max(new_primary, key=lambda r: safe_float(r.get("slice_accuracy"), -1.0), default={})
        best_targeted = max(targeted, key=lambda r: safe_float(r.get("slice_accuracy"), -1.0), default={})
        old_acc = safe_float(best_old.get("slice_accuracy"), float("nan"))
        new_acc = safe_float(best_new.get("slice_accuracy"), float("nan"))
        summary_rows.append(
            {
                "slice_name": slice_name,
                "pair_count": len(idxs),
                "best_old_bg": best_old.get("candidate_name"),
                "best_old_bg_family": best_old.get("candidate_family"),
                "best_old_bg_accuracy": old_acc,
                "best_new_two_tap": best_new.get("candidate_name"),
                "best_new_two_tap_family": best_new.get("candidate_family"),
                "best_new_two_tap_accuracy": new_acc,
                "delta_new_minus_old": new_acc - old_acc if math.isfinite(new_acc) and math.isfinite(old_acc) else float("nan"),
                "new_matches_or_exceeds_old": bool(math.isfinite(new_acc) and math.isfinite(old_acc) and new_acc + 1e-12 >= old_acc),
                "best_targeted_rehost_diagnostic": best_targeted.get("candidate_name"),
                "best_targeted_rehost_diagnostic_accuracy": best_targeted.get("slice_accuracy"),
            }
        )
    domain_rows = [row for row in summary_rows if str(row.get("slice_name", "")).startswith("domain::")]
    failures = [row for row in domain_rows if not row.get("new_matches_or_exceeds_old")]
    all_row = next((row for row in summary_rows if row.get("slice_name") == "all"), {})
    all_ok = bool(all_row.get("new_matches_or_exceeds_old"))
    if domain_rows and not failures and all_ok:
        verdict = "TWO_TAP_MATCHES_OR_BEATS_OLD_BG_ON_FRESH"
    elif domain_rows and len(failures) < len(domain_rows):
        verdict = "MIXED"
    elif domain_rows:
        verdict = "OLD_BG_BEST"
    else:
        verdict = "DATA_LIMITED"
    return {
        "BG_TWO_TAP_FRESH_DATA_COMPARISON_VERDICT": verdict,
        "summary_rows": summary_rows,
        "domain_failures": failures,
        "overall": all_row,
    }


def strip_scores(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in row.items() if k != "scores_by_pair"} for row in rows]


def write_summary_md(path: Path, payload: dict[str, Any], dataset: dict[str, Any]) -> None:
    rows = payload["summary_rows"]
    overall = payload.get("overall") or {}
    lines = [
        "# Two-Tap Fresh Dataset Comparison v1",
        "",
        f"BG_TWO_TAP_FRESH_DATA_COMPARISON_VERDICT = {payload['BG_TWO_TAP_FRESH_DATA_COMPARISON_VERDICT']}",
        "",
        "## Scope",
        "",
        "- Existing datasets were reused only where prior artifacts were subset samples; previous task IDs were excluded where available.",
        "- New datasets were added for the same domains.",
        "- Seed 42 was not used.",
        "- Old BG taps and new two-tap candidates were scored on identical candidate pairs.",
        "- Primary new two-tap candidates were restricted to MIX_CODE_REASONING and MIX_OBJECTIVE_ALL anchors; old-content transplants from other old taps were excluded.",
        "- Targeted rehost candidates are diagnostic only and are not part of primary readiness.",
        "",
        "## Dataset",
        "",
        f"- task_count: `{dataset['summary']['task_count']}`",
        f"- candidate_count: `{dataset['summary']['candidate_count']}`",
        f"- domain_counts: `{dataset['summary']['domain_counts']}`",
        f"- source_status_counts: `{dataset['summary']['source_status_counts']}`",
        f"- source_dataset_counts: `{dataset['summary']['source_dataset_counts']}`",
        f"- dataset_errors: `{dataset['summary']['dataset_errors']}`",
        "",
        "## Overall",
        "",
        f"- pair_count: `{overall.get('pair_count')}`",
        f"- best_old_bg_accuracy: `{overall.get('best_old_bg_accuracy')}`",
        f"- best_new_two_tap_accuracy: `{overall.get('best_new_two_tap_accuracy')}`",
        f"- delta_new_minus_old: `{overall.get('delta_new_minus_old')}`",
        f"- best_old_bg: `{overall.get('best_old_bg')}`",
        f"- best_new_two_tap: `{overall.get('best_new_two_tap')}`",
        "",
        "## Domain Slices",
        "",
        "| slice | pairs | old bg acc | new two-tap acc | delta | pass |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        if not str(row.get("slice_name")).startswith("domain::"):
            continue
        lines.append(
            "| {slice_name} | {pair_count} | {old:.4f} | {new:.4f} | {delta:.4f} | {ok} |".format(
                slice_name=row.get("slice_name"),
                pair_count=row.get("pair_count"),
                old=safe_float(row.get("best_old_bg_accuracy"), float("nan")),
                new=safe_float(row.get("best_new_two_tap_accuracy"), float("nan")),
                delta=safe_float(row.get("delta_new_minus_old"), float("nan")),
                ok=row.get("new_matches_or_exceeds_old"),
            )
        )
    lines.extend(["", "## Source Status Slices", "", "| slice | pairs | old bg acc | new two-tap acc | delta | pass |", "| --- | ---: | ---: | ---: | ---: | --- |"])
    for row in rows:
        if not str(row.get("slice_name")).startswith("source_status::"):
            continue
        lines.append(
            "| {slice_name} | {pair_count} | {old:.4f} | {new:.4f} | {delta:.4f} | {ok} |".format(
                slice_name=row.get("slice_name"),
                pair_count=row.get("pair_count"),
                old=safe_float(row.get("best_old_bg_accuracy"), float("nan")),
                new=safe_float(row.get("best_new_two_tap_accuracy"), float("nan")),
                delta=safe_float(row.get("delta_new_minus_old"), float("nan")),
                ok=row.get("new_matches_or_exceeds_old"),
            )
        )
    lines.extend(["", "## Files", "", f"- dataset: `{rel(DATASET_JSON)}`", f"- features: `{rel(FEATURES_PT)}`", f"- rows: `{rel(PAIR_ROWS_CSV)}`", f"- summary: `{rel(SUMMARY_JSON)}`", f"- artifact: `{rel(ARTIFACT_PT)}`", ""])
    write_md(path, lines)


def write_doc(summary: dict[str, Any], dataset: dict[str, Any]) -> None:
    overall = summary.get("overall") or {}
    lines = [
        "# BG Two-Tap Fresh Dataset Comparison v1",
        "",
        "This probe compares the old frozen BG taps and the new layer-native two-tap candidates on the same fresh domain candidate content.",
        "",
        f"BG_TWO_TAP_FRESH_DATA_COMPARISON_VERDICT = {summary['BG_TWO_TAP_FRESH_DATA_COMPARISON_VERDICT']}",
        "",
        "No Ouro weights, tokenizer files, checkpoints, tap registries, production routing, wrapper/local-agent code, Hunter-Seeker modules, or steering modules were modified or executed.",
        "",
        "## Dataset Policy",
        "",
        "Previously used public sources were sampled again only because the prior local artifacts were subset samples rather than full-dataset passes. The runner used a non-42 seed and excluded prior task IDs where the earlier artifact exposed them. New public datasets were also added for the same domains.",
        "",
        "Primary new two-tap candidates were restricted to the `MIX_CODE_REASONING` and `MIX_OBJECTIVE_ALL` anchors. Diagnostic old-content transplants from other old taps were excluded from the primary readiness comparison.",
        "",
        "## Result",
        "",
        f"- tasks: `{dataset['summary']['task_count']}`",
        f"- candidates: `{dataset['summary']['candidate_count']}`",
        f"- domains: `{dataset['summary']['domain_counts']}`",
        f"- best old BG overall accuracy: `{overall.get('best_old_bg_accuracy')}`",
        f"- best new two-tap overall accuracy: `{overall.get('best_new_two_tap_accuracy')}`",
        f"- delta: `{overall.get('delta_new_minus_old')}`",
        f"- best old BG: `{overall.get('best_old_bg')}`",
        f"- best new two-tap: `{overall.get('best_new_two_tap')}`",
        "",
        "Targeted rehost candidates remain diagnostic and are not counted in primary readiness.",
        "",
        "## Artifacts",
        "",
        f"- `{rel(DATASET_JSON)}`",
        f"- `{rel(FEATURES_PT)}`",
        f"- `{rel(SUMMARY_MD)}`",
        f"- `{rel(PAIR_ROWS_CSV)}`",
        f"- `{rel(ARTIFACT_PT)}`",
        "",
    ]
    write_md(DOC_MD, lines)


def main() -> None:
    args = parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(args)
    if dataset.get("BG_TWO_TAP_FRESH_DATASET_BUILD_VERDICT") == "BLOCKED":
        raise SystemExit("BG_TWO_TAP_FRESH_DATASET_BUILD_VERDICT=BLOCKED")
    features = capture_features(args, dataset)
    pairs = build_pairs(features)
    old_taps = load_old_bg_taps()
    new_taps, bundles = load_new_two_taps(bool(args.include_targeted_rehost_diagnostic))
    old_rows = evaluate_single_candidates(old_taps, pairs)
    new_rows = evaluate_single_candidates(new_taps, pairs)
    bundle_rows = evaluate_bundles([r for r in new_taps if r.get("eval_family") == "new_two_tap_primary"], bundles, pairs)
    all_rows = old_rows + new_rows + bundle_rows
    summary = summarize_rows(all_rows, pairs)
    write_csv(PAIR_ROWS_CSV, strip_scores(all_rows))
    write_json(SUMMARY_JSON, {**summary, "candidate_row_count": len(all_rows), "pair_count": len(pairs), "dataset_summary": dataset.get("summary", {})})
    write_summary_md(SUMMARY_MD, summary, dataset)
    write_doc(summary, dataset)
    torch.save(
        {
            "summary": summary,
            "dataset_summary": dataset.get("summary", {}),
            "pairs": [{k: v for k, v in p.items() if k != "features"} for p in pairs],
            "row_summary": strip_scores(all_rows),
            "paths": {
                "dataset_json": rel(DATASET_JSON),
                "features_pt": rel(FEATURES_PT),
                "summary_json": rel(SUMMARY_JSON),
                "summary_md": rel(SUMMARY_MD),
                "pair_rows_csv": rel(PAIR_ROWS_CSV),
            },
        },
        ARTIFACT_PT,
    )
    print(f"BG_TWO_TAP_FRESH_DATASET_BUILD_VERDICT = {dataset['BG_TWO_TAP_FRESH_DATASET_BUILD_VERDICT']}", flush=True)
    print(f"BG_TWO_TAP_FRESH_FEATURE_CAPTURE_VERDICT = {features['meta']['BG_TWO_TAP_FRESH_FEATURE_CAPTURE_VERDICT']}", flush=True)
    print(f"BG_TWO_TAP_FRESH_DATA_COMPARISON_VERDICT = {summary['BG_TWO_TAP_FRESH_DATA_COMPARISON_VERDICT']}", flush=True)
    print(f"Wrote {SUMMARY_MD}", flush=True)


if __name__ == "__main__":
    main()
