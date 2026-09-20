"""Shared helpers for the read-only BG trajectory prediction sweep."""
from __future__ import annotations

import itertools
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROBE_ROOT = PROJECT_ROOT / "opi/taps/probes"
REPORT_ROOT = PROBE_ROOT / "bg_trajectory_prediction_2026-05-18"
MODEL_PATH = PROJECT_ROOT / "shared/models/ouro_rltt_local"
SEED = 20260518
PREFIX_LENGTHS = (32, 64, 128, 256)
BRANCHES_PER_TASK = 4
MAX_NEW_TOKENS = 256

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "artifacts" / "hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "artifacts" / "hf_cache" / "datasets"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from bg_steering_suite_lib import (  # noqa: E402
    OuroTextGenerator,
    auc_score,
    evaluate_output,
    gsm_prompt,
    load_gsm8k_tasks,
    load_reasoning_tasks,
    load_science_tasks,
    mcq_prompt,
    md_table,
    spearman,
)


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(p)


def out_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_json(path: str | Path, default: Any = None) -> Any:
    p = out_path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    p = out_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(p)


def write_md(path: str | Path, lines: Sequence[str]) -> None:
    p = out_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_once(path: str | Path, title: str, lines: Sequence[str]) -> None:
    p = out_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = p.read_text(encoding="utf-8") if p.exists() else ""
    if title in existing:
        return
    block = "\n".join(["", f"## {title}", "", *lines, ""])
    p.write_text(existing.rstrip() + block + "\n", encoding="utf-8")


def ensure_report_root() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)


def task_prompt(task: dict[str, Any]) -> str:
    return str(task.get("prompt") or task.get("question") or "").strip()


def trajectory_generation_prompt(task: dict[str, Any]) -> str:
    domain = str(task["domain"])
    if domain in {"reasoning", "science"}:
        return mcq_prompt(task["question"], task["options"])
    if domain == "gsm8k":
        return gsm_prompt(task["question"])
    return task_prompt(task)


def continuation_prompt(task: dict[str, Any], prefix_text: str) -> str:
    base = trajectory_generation_prompt(task)
    if task["domain"] in {"reasoning", "science"}:
        return (
            f"{base}\n\n"
            "Partial attempt so far:\n"
            f"{str(prefix_text).strip()}\n\n"
            "Continue from the partial attempt and end with the required FINAL ANSWER: <letter> format."
        )
    if task["domain"] == "gsm8k":
        return (
            f"{base}\n\n"
            "Partial solution so far:\n"
            f"{str(prefix_text).strip()}\n\n"
            "Continue from the partial attempt and end with the required FINAL ANSWER: <number> format."
        )
    return f"{base}\n\nPartial attempt so far:\n{str(prefix_text).strip()}"


def continuation_budget(task: dict[str, Any]) -> int:
    if task["domain"] in {"reasoning", "science"}:
        return 192
    if task["domain"] == "gsm8k":
        return 256
    return 192


def domain_hint_for_task(task: dict[str, Any]) -> str:
    if task["domain"] == "reasoning":
        return "reasoning"
    if task["domain"] == "science":
        return "science"
    if task["domain"] == "gsm8k":
        return "gsm8k"
    return "objective"


def normalize_reasoning_task(item: dict[str, Any]) -> dict[str, Any] | None:
    answer = str(item.get("answer") or item.get("answer_key") or "").upper()
    options = item.get("options") or {}
    question = str(item.get("question") or "").strip()
    task_id = str(item.get("task_id") or "")
    if not task_id or not question or not answer or not options:
        return None
    return {
        "task_id": task_id,
        "domain": "reasoning",
        "source_dataset": item.get("dataset") or item.get("source_dataset") or "reasoning",
        "prompt": mcq_prompt(question, options),
        "question": question,
        "options": options,
        "answer_key": answer,
        "gold_answer": answer,
        "evaluator_type": "mcq_letter",
        "expected_answer_format": "FINAL ANSWER: <letter>",
        "difficulty": item.get("difficulty") or "mixed",
        "is_devil": False,
    }


def normalize_science_task(item: dict[str, Any]) -> dict[str, Any] | None:
    answer = str(item.get("answer_key") or item.get("answer") or "").upper()
    options = item.get("options") or {}
    question = str(item.get("question") or "").strip()
    task_id = str(item.get("task_id") or "")
    if not task_id or not question or not answer or not options:
        return None
    return {
        "task_id": task_id,
        "domain": "science",
        "source_dataset": item.get("source_dataset") or "science",
        "source_subject": item.get("source_subject"),
        "subdomain_bucket": item.get("subdomain_bucket"),
        "prompt": mcq_prompt(question, options),
        "question": question,
        "options": options,
        "answer_key": answer,
        "gold_answer": answer,
        "evaluator_type": "mcq_letter",
        "expected_answer_format": "FINAL ANSWER: <letter>",
        "difficulty": item.get("difficulty") or "mixed",
        "is_devil": False,
    }


def normalize_gsm_task(item: dict[str, Any], fallback_index: int) -> dict[str, Any] | None:
    question = str(item.get("question") or "").strip()
    answer = str(item.get("gold_answer") or item.get("answer_key") or "").strip()
    dataset_index = item.get("dataset_index", item.get("problem_id", fallback_index))
    task_id = f"gsm8k/{dataset_index}"
    if not question or not answer:
        return None
    return {
        "task_id": task_id,
        "domain": "gsm8k",
        "source_dataset": "gsm8k",
        "prompt": gsm_prompt(question),
        "question": question,
        "answer_key": answer,
        "gold_answer": answer,
        "evaluator_type": "numeric_exact",
        "expected_answer_format": "FINAL ANSWER: <number>",
        "difficulty": item.get("difficulty") or "clean",
        "is_devil": False,
    }


def build_trajectory_task_rows(reasoning_n: int = 20, science_n: int = 20, gsm_n: int = 20) -> list[dict[str, Any]]:
    reasoning = [normalize_reasoning_task(row) for row in load_reasoning_tasks()]
    science = [normalize_science_task(row) for row in load_science_tasks()]
    gsm = [normalize_gsm_task(row, idx) for idx, row in enumerate(load_gsm8k_tasks())]
    rows = [r for r in reasoning if r is not None][:reasoning_n]
    rows += [r for r in science if r is not None][:science_n]
    rows += [r for r in gsm if r is not None][:gsm_n]
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        tid = str(row["task_id"])
        if tid in seen:
            continue
        seen.add(tid)
        row["suite_index"] = len(out)
        out.append(row)
    return out[:75]


def load_task_suite() -> list[dict[str, Any]]:
    payload = load_json(REPORT_ROOT / "task_suite.json", {})
    return list(payload.get("tasks") or [])


def task_by_id(tasks: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(task["task_id"]): task for task in tasks}


def load_partials() -> dict[str, Any]:
    return load_json(REPORT_ROOT / "partials.json", load_json(REPORT_ROOT / "partials.partial.json", {}))


def load_continued() -> dict[str, Any]:
    return load_json(REPORT_ROOT / "continued_prefixes.json", load_json(REPORT_ROOT / "continued_prefixes.partial.json", {}))


def prefix_key(task_id: str, branch_id: int, prefix_length: int) -> str:
    safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "__", str(task_id))
    return f"{safe_task}::b{int(branch_id)}::p{int(prefix_length)}"


def tokenize_completion(tokenizer: Any, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def detokenize_completion(tokenizer: Any, token_ids: Sequence[int]) -> str:
    if not token_ids:
        return ""
    return str(tokenizer.decode(list(token_ids), skip_special_tokens=True)).strip()


def make_prefixes(tokenizer: Any, generated_text: str) -> tuple[dict[str, str], dict[str, int], dict[str, bool]]:
    token_ids = tokenize_completion(tokenizer, generated_text)
    prefixes: dict[str, str] = {}
    token_counts: dict[str, int] = {}
    missing: dict[str, bool] = {}
    for checkpoint in PREFIX_LENGTHS:
        used = token_ids[: min(checkpoint, len(token_ids))]
        prefixes[f"prefix_{checkpoint}"] = detokenize_completion(tokenizer, used)
        token_counts[str(checkpoint)] = len(used)
        missing[str(checkpoint)] = len(token_ids) < checkpoint
    return prefixes, token_counts, missing


def iter_prefix_rows(partials_payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for branch in partials_payload.get("branches") or []:
        prefixes = branch.get("prefixes") or {}
        token_counts = branch.get("prefix_token_counts") or {}
        missing = branch.get("checkpoint_missing") or {}
        for prefix_length in PREFIX_LENGTHS:
            text = prefixes.get(f"prefix_{prefix_length}") or ""
            if not text.strip():
                continue
            yield {
                "task_id": str(branch["task_id"]),
                "domain": branch.get("domain"),
                "branch_id": int(branch["branch_id"]),
                "prefix_length": int(prefix_length),
                "prefix_text": text,
                "prefix_token_count": int(token_counts.get(str(prefix_length), 0)),
                "checkpoint_missing": bool(missing.get(str(prefix_length))),
            }


def expected_random_topk(successes: Sequence[bool], k: int) -> float:
    n = len(successes)
    if n == 0:
        return 0.0
    k = min(k, n)
    combos = list(itertools.combinations(range(n), k))
    return sum(any(successes[i] for i in combo) for combo in combos) / max(len(combos), 1)


def branch_label_map(continued_payload: dict[str, Any]) -> dict[tuple[str, int, int], bool]:
    out = {}
    for row in continued_payload.get("continued_prefixes") or []:
        out[(str(row["task_id"]), int(row["branch_id"]), int(row["prefix_length"]))] = bool(row.get("is_correct"))
    return out


def evaluable_key_set(continued_payload: dict[str, Any]) -> set[tuple[str, int, int]]:
    out = set()
    for row in continued_payload.get("continued_prefixes") or []:
        if row.get("evaluable", True) and not row.get("continuation_error"):
            out.add((str(row["task_id"]), int(row["branch_id"]), int(row["prefix_length"])))
    return out


def bootstrap_mean_delta(a: list[float], b: list[float], rounds: int = 500, seed: int = SEED) -> dict[str, float]:
    if not a or len(a) != len(b):
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = random.Random(seed)
    vals = []
    n = len(a)
    for _ in range(rounds):
        idxs = [rng.randrange(n) for _ in range(n)]
        vals.append(sum(a[i] - b[i] for i in idxs) / n)
    vals.sort()
    return {"mean": sum(vals) / len(vals), "lo": vals[int(0.025 * rounds)], "hi": vals[int(0.975 * rounds)]}


def finite_or_none(value: float) -> float | None:
    if value is None or math.isnan(float(value)) or math.isinf(float(value)):
        return None
    return float(value)


def summarize_bool(values: Iterable[bool]) -> dict[str, Any]:
    vals = [bool(x) for x in values]
    return {"n": len(vals), "successes": sum(vals), "rate": sum(vals) / max(len(vals), 1)}


def write_simple_table(path: Path, title: str, verdict_line: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    lines = [f"# {title}", "", verdict_line, "", *md_table(rows, columns)]
    write_md(path, lines)
