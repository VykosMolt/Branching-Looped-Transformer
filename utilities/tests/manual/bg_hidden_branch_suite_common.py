"""Shared helpers for the BG same-prefix hidden-branch suite."""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROBE_ROOT = PROJECT_ROOT / "opi/taps/probes"
REPORT_ROOT = PROBE_ROOT / "bg_hidden_state_branch_generation_2026-05-18"
MODEL_PATH = PROJECT_ROOT / "shared/models/ouro_rltt_local"
SEED = 20260518
STARTED_AT = time.time()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "artifacts" / "hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "artifacts" / "hf_cache" / "datasets"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(p)


def out_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def ensure_report_root() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)


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


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    p = out_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return out


def options_text(options: dict[str, str]) -> str:
    return "\n".join(f"{letter}. {text}" for letter, text in sorted(options.items()))


def hidden_branch_prompt(question: str, options: dict[str, str]) -> str:
    return (
        f"Question: {str(question).strip()}\n"
        "Options:\n"
        f"{options_text(options)}\n"
        "Think briefly if needed.\n"
        "FINAL ANSWER:"
    )


def parse_mcq_answer(text: str, options: dict[str, str]) -> str | None:
    raw = str(text or "")
    letters = sorted(str(k).upper() for k in options)
    allowed = "".join(re.escape(x) for x in letters)
    patterns = [
        rf"FINAL ANSWER\s*:\s*<?([{allowed}])>?",
        rf"\banswer\s*(?:is|:)\s*<?([{allowed}])>?",
        rf"^\s*<?([{allowed}])>?\s*$",
        rf"\b([{allowed}])\s*[\).]\s",
    ]
    for pattern in patterns:
        found = re.findall(pattern, raw, flags=re.IGNORECASE | re.MULTILINE)
        if found:
            return str(found[-1]).upper()
    return None


def evaluate_mcq(task: dict[str, Any], text: str) -> dict[str, Any]:
    parsed = parse_mcq_answer(text, dict(task.get("options") or {}))
    gold = str(task.get("correct_option") or task.get("answer_key") or task.get("gold_answer") or "").upper()
    parse_success = parsed is not None
    correct = parse_success and parsed == gold
    if correct:
        reward = 1.0
    elif parse_success:
        reward = 0.0
    elif not str(text or "").strip():
        reward = -0.5
    else:
        reward = -0.2
    words = str(text or "").split()
    repeats = sum(1 for a, b in zip(words, words[1:]) if a == b)
    repetition_rate = repeats / max(len(words) - 1, 1)
    if repetition_rate > 0.35 and reward > -0.3:
        reward = -0.3
    return {
        "parsed_answer": parsed,
        "correct": bool(correct),
        "reward": float(reward),
        "parse_success": bool(parse_success),
        "parse_failure_reason": "" if parse_success else "no_mcq_letter",
        "repetition_rate": float(repetition_rate),
        "empty_output": not bool(str(text or "").strip()),
    }


def load_task_subset() -> list[dict[str, Any]]:
    payload = load_json(REPORT_ROOT / "task_subset.json", {})
    return list(payload.get("tasks") or [])


def group_rows(rows: Iterable[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(k) for k in keys)].append(row)
    return grouped


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if out == out and abs(out) != float("inf") else default


def tensor_to_json_stats(t: torch.Tensor) -> dict[str, Any]:
    x = t.detach().to(device="cpu", dtype=torch.float32)
    return {
        "shape": list(x.shape),
        "rms": float(x.pow(2).mean().sqrt().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()) if x.numel() > 1 else 0.0,
        "finite": bool(torch.isfinite(x).all().item()),
    }


def now_elapsed() -> float:
    return time.time() - STARTED_AT
