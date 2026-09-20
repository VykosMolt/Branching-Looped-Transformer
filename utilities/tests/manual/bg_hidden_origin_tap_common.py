"""Shared helpers for hidden-origin branch tap probes.

This module is intentionally local to manual probes.  It reads frozen branch
artifacts, trains only tiny comparator heads, and never mutates Ouro weights,
tokenizer files, old BG tap registries, or checkpoints.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Iterator, Sequence

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROBE_ROOT = PROJECT_ROOT / "opi/taps/probes"
OLD_ROOT = PROBE_ROOT / "bg_hidden_state_branch_generation_2026-05-18"
OUT_ROOT = PROBE_ROOT / "bg_hidden_origin_taps_2026-05-18"
MODEL_PATH = PROJECT_ROOT / "shared/models/ouro_rltt_local"
SEED = 20260518
NUM_LOOPS = 4
HIDDEN_DIM = 2048
BASE_LAYERS = (24, 36, 47)
EXTRA_LAYERS = (30, 42)
STARTED_AT = time.time()

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "shared/hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "shared/hf_cache/datasets"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CONFIGS = (
    "24_L4",
    "24_mean",
    "36_L4",
    "36_mean",
    "47_L4",
    "47_mean",
    "concat_24_36",
    "concat_36_47",
    "concat_24_36_47",
    "30_L4",
    "42_L4",
    "concat_24_30_36",
    "concat_36_42_47",
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


def ensure_out_root() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def load_json(path: str | Path, default: Any = None) -> Any:
    p = out_path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    p = out_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
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
            writer.writerow({key: row.get(key, "") for key in keys})


def json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return tensor_stats(value)
    if isinstance(value, Path):
        return rel(value)
    try:
        return float(value)
    except Exception:
        return str(value)


def md_table(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> list[str]:
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return out


def rate(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return "NA"
    if math.isnan(x) or math.isinf(x):
        return "NA"
    return f"{x:.3f}"


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def tensor_stats(t: torch.Tensor) -> dict[str, Any]:
    x = t.detach().to(device="cpu", dtype=torch.float32)
    return {
        "shape": list(x.shape),
        "mean": float(x.mean().item()) if x.numel() else 0.0,
        "std": float(x.std(unbiased=False).item()) if x.numel() > 1 else 0.0,
        "rms": float(x.pow(2).mean().sqrt().item()) if x.numel() else 0.0,
        "finite": bool(torch.isfinite(x).all().item()),
    }


def branch_key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row.get("branch_group_id")), int(row.get("branch_id", -1)))


def group_rows(rows: Iterable[dict[str, Any]], key: str = "branch_group_id") -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key))].append(row)
    return grouped


def is_safe_alpha(row: dict[str, Any]) -> bool:
    return finite_float(row.get("alpha"), 999.0) <= 0.0100001 and bool(row.get("safety_envelope", True))


def is_diagnostic_high_alpha(row: dict[str, Any]) -> bool:
    return finite_float(row.get("alpha"), 0.0) > 0.0200001


def severe_repetition(row: dict[str, Any]) -> bool:
    return finite_float(row.get("repetition_rate"), 0.0) > 0.35


def nonempty_output(row: dict[str, Any]) -> bool:
    return bool(str(row.get("output_text") or "").strip()) and int(row.get("output_length") or 0) > 0


def stable_row(row: dict[str, Any]) -> bool:
    if str(row.get("cuda_error") or "").strip():
        return False
    if bool(row.get("nan_inf")):
        return False
    if not nonempty_output(row):
        return False
    if severe_repetition(row):
        return False
    feats = row.get("features")
    if isinstance(feats, torch.Tensor) and not bool(torch.isfinite(feats).all().item()):
        return False
    for value in (row.get("pooled_vectors") or {}).values():
        if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all().item()):
            return False
    return True


def group_is_behaviorally_diverse(rows: Sequence[dict[str, Any]]) -> bool:
    rewards = {finite_float(row.get("reward"), 0.0) for row in rows}
    correct = {bool(row.get("correct")) for row in rows}
    return len(rewards) > 1 or len(correct) > 1


def group_is_reward_diverse(rows: Sequence[dict[str, Any]]) -> bool:
    return len({finite_float(row.get("reward"), 0.0) for row in rows}) > 1


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
    correct = bool(parse_success and parsed == gold)
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
        "correct": correct,
        "reward": float(reward),
        "parse_success": bool(parse_success),
        "parse_failure_reason": "" if parse_success else "no_mcq_letter",
        "repetition_rate": float(repetition_rate),
        "empty_output": not bool(str(text or "").strip()),
    }


def normalize_task(row: dict[str, Any], domain: str | None = None, index: int = 0) -> dict[str, Any] | None:
    inferred_domain = str(domain or row.get("domain") or "").strip().lower()
    if inferred_domain not in {"reasoning", "science"}:
        return None
    question = str(row.get("question") or "").strip()
    options = row.get("options") or {}
    answer = str(row.get("correct_option") or row.get("answer_key") or row.get("answer") or row.get("gold_answer") or "").upper()
    task_id = str(row.get("task_id") or f"{inferred_domain}/{index}")
    if not question or not isinstance(options, dict) or len(options) != 4:
        return None
    clean_options = {str(k).upper(): str(v).strip() for k, v in sorted(options.items())}
    if answer not in clean_options:
        return None
    return {
        "task_id": task_id,
        "domain": inferred_domain,
        "source_dataset": row.get("source_dataset") or row.get("dataset") or inferred_domain,
        "source_subject": row.get("source_subject"),
        "subdomain_bucket": row.get("subdomain_bucket"),
        "question": question,
        "options": clean_options,
        "correct_option": answer,
        "expected_answer_text": clean_options.get(answer, ""),
        "parser_type": "mcq_letter",
        "prompt": hidden_branch_prompt(question, clean_options),
    }


def load_candidate_tasks() -> list[dict[str, Any]]:
    sources = [
        (OLD_ROOT / "task_subset.json", None),
        (PROBE_ROOT / "bg_trajectory_prediction_2026-05-18/task_suite.json", None),
        (PROBE_ROOT / "reasoning_branch_pilot_2026-05-17.json", "reasoning"),
        (PROBE_ROOT / "science_natural_distractor_set_2026-05-17.json", "science"),
    ]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path, forced_domain in sources:
        payload = load_json(path, {}) or {}
        for idx, item in enumerate(payload.get("tasks") or []):
            task = normalize_task(item, forced_domain, idx)
            if not task or task["task_id"] in seen:
                continue
            seen.add(task["task_id"])
            rows.append(task)
    return rows


def load_existing_branch_rows() -> list[dict[str, Any]]:
    outcomes = load_json(OLD_ROOT / "hidden_branch_outcomes.json", {}) or {}
    outcome_rows = list(outcomes.get("rows") or [])
    by_key = {branch_key(row): dict(row) for row in outcome_rows}
    pt_path = OLD_ROOT / "hidden_branch_persistence.pt"
    if pt_path.exists():
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        for rec in list(payload.get("records") or []):
            key = branch_key(rec)
            if key not in by_key:
                continue
            merged = dict(rec)
            merged.update(by_key[key])
            by_key[key] = merged
    return list(by_key.values())


def load_expanded_branch_rows() -> list[dict[str, Any]]:
    pt_path = OUT_ROOT / "expanded_hidden_origin_branches.pt"
    if not pt_path.exists():
        return []
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    rows = list(payload.get("rows") or payload.get("records") or [])
    return [dict(row) for row in rows if row.get("branch_group_id")]


def load_all_branch_rows() -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in load_existing_branch_rows() + load_expanded_branch_rows():
        if row.get("branch_group_id") is None:
            continue
        by_key[branch_key(row)] = row
    return list(by_key.values())


def pooled_lookup(row: dict[str, Any], layer: int, loop: int) -> torch.Tensor | None:
    key = f"L{int(layer)}_L{int(loop)}"
    pooled = row.get("pooled_vectors") or {}
    value = pooled.get(key)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.float32)
    features = row.get("features")
    if isinstance(features, torch.Tensor) and int(layer) in BASE_LAYERS:
        pos = BASE_LAYERS.index(int(layer))
        loop_idx = int(loop) - 1
        if 0 <= loop_idx < features.shape[1]:
            return features[pos, loop_idx].detach().cpu().to(torch.float32)
    return None


def config_vector_from_row(row: dict[str, Any], config: str) -> torch.Tensor | None:
    def layer_l4(layer: int) -> torch.Tensor | None:
        return pooled_lookup(row, layer, 4)

    def layer_mean(layer: int) -> torch.Tensor | None:
        vals = [pooled_lookup(row, layer, loop) for loop in range(1, NUM_LOOPS + 1)]
        vals = [v for v in vals if isinstance(v, torch.Tensor)]
        return torch.stack(vals, dim=0).mean(dim=0) if len(vals) == NUM_LOOPS else None

    if config == "24_L4":
        return layer_l4(24)
    if config == "24_mean":
        return layer_mean(24)
    if config == "36_L4":
        return layer_l4(36)
    if config == "36_mean":
        return layer_mean(36)
    if config == "47_L4":
        return layer_l4(47)
    if config == "47_mean":
        return layer_mean(47)
    if config == "30_L4":
        return layer_l4(30)
    if config == "42_L4":
        return layer_l4(42)
    concat_map = {
        "concat_24_36": (24, 36),
        "concat_36_47": (36, 47),
        "concat_24_36_47": (24, 36, 47),
        "concat_24_30_36": (24, 30, 36),
        "concat_36_42_47": (36, 42, 47),
    }
    if config in concat_map:
        vals = [layer_l4(layer) for layer in concat_map[config]]
        if all(isinstance(v, torch.Tensor) for v in vals):
            return torch.cat([v for v in vals if isinstance(v, torch.Tensor)], dim=-1)
        return None
    return None


def config_dim(config: str) -> int:
    if config in {"concat_24_36", "concat_36_47"}:
        return HIDDEN_DIM * 2
    if config in {"concat_24_36_47", "concat_24_30_36", "concat_36_42_47"}:
        return HIDDEN_DIM * 3
    if config in CONFIGS:
        return HIDDEN_DIM
    raise ValueError(f"unsupported config: {config}")


def available_configs_for_rows(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    coverage: dict[str, int] = {}
    for config in CONFIGS:
        count = 0
        for row in rows:
            vec = config_vector_from_row(row, config)
            if isinstance(vec, torch.Tensor) and tuple(vec.shape) == (config_dim(config),):
                count += 1
        coverage[config] = count
    return coverage


class AntisymLinear(nn.Module):
    """LayerNorm(no affine)(left - right) -> Linear(no bias)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.norm = nn.LayerNorm(self.dim, elementwise_affine=False)
        self.linear = nn.Linear(self.dim, 1, bias=False)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(left - right)).squeeze(-1)


class AntisymLinearNoNorm(nn.Module):
    """Linear(left - right)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.linear = nn.Linear(self.dim, 1, bias=False)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.linear(left - right).squeeze(-1)


HEAD_CLASSES = {
    "AntisymLinear": AntisymLinear,
    "AntisymLinearNoNorm": AntisymLinearNoNorm,
}


def direction_from_state_dict(state_dict: dict[str, torch.Tensor]) -> torch.Tensor | None:
    value = state_dict.get("linear.weight")
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.float32).flatten()
    return None


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    af = a.detach().cpu().to(torch.float32).flatten()
    bf = b.detach().cpu().to(torch.float32).flatten()
    if af.numel() != bf.numel():
        return float("nan")
    denom = torch.linalg.vector_norm(af) * torch.linalg.vector_norm(bf)
    if float(denom.item()) <= eps:
        return float("nan")
    return float(torch.dot(af, bf).div(denom.clamp(min=eps)).item())


@torch.no_grad()
def score_pair(head: nn.Module, left: torch.Tensor, right: torch.Tensor, device: torch.device) -> float:
    head.eval()
    out = head(left.to(device=device, dtype=torch.float32).unsqueeze(0), right.to(device=device, dtype=torch.float32).unsqueeze(0))
    return float(out.detach().cpu().flatten()[0].item())


@torch.no_grad()
def score_matrix(head: nn.Module, vectors: Sequence[torch.Tensor], device: torch.device) -> torch.Tensor:
    if not vectors:
        return torch.zeros((0, 0), dtype=torch.float32)
    feats = torch.stack([v.to(torch.float32) for v in vectors], dim=0).to(device)
    k = int(feats.shape[0])
    left = feats[:, None, :].expand(k, k, feats.shape[-1]).reshape(k * k, feats.shape[-1])
    right = feats[None, :, :].expand(k, k, feats.shape[-1]).reshape(k * k, feats.shape[-1])
    mat = head(left, right).reshape(k, k).detach().cpu().to(torch.float32)
    mat.fill_diagonal_(0.0)
    return mat


def ranking_from_matrix(mat: torch.Tensor) -> dict[str, Any]:
    n = int(mat.shape[0])
    if n == 0:
        return {"ranking": [], "wins": [], "margin_sum": []}
    mask = ~torch.eye(n, dtype=torch.bool)
    wins = ((mat > 0) & mask).sum(dim=1).to(torch.int64)
    margins = (mat * mask.to(mat.dtype)).sum(dim=1)
    ranking = sorted(range(n), key=lambda idx: (-int(wins[idx].item()), -float(margins[idx].item()), idx))
    return {"ranking": ranking, "wins": wins.tolist(), "margin_sum": margins.tolist()}


def pairwise_accuracy_from_pairs(
    head: nn.Module,
    pairs: Sequence[dict[str, Any]],
    config: str,
    device: torch.device,
) -> float:
    ok = 0
    total = 0
    for pair in pairs:
        feats = pair.get("features", {}).get(config)
        if not feats:
            continue
        score = score_pair(head, feats["preferred"], feats["rejected"], device)
        ok += int(score > 0)
        total += 1
    return ok / max(total, 1) if total else float("nan")


def split_pairs(pairs: Sequence[dict[str, Any]], seed: int = 42) -> tuple[dict[str, set[str]], dict[str, Any]]:
    task_ids = sorted({str(pair["task_id"]) for pair in pairs})
    if not task_ids:
        return {"train": set(), "val": set(), "test": set()}, {"blocked": True, "reason": "no tasks"}
    by_task = defaultdict(int)
    for pair in pairs:
        by_task[str(pair["task_id"])] += 1
    n = len(task_ids)
    if n == 1:
        return {"train": set(task_ids), "val": set(), "test": set()}, {"weak_split": True, "reason": "one task only"}
    train_n = max(1, round(0.60 * n))
    val_n = max(1, round(0.20 * n)) if n >= 3 else 0
    if train_n + val_n >= n:
        train_n = max(1, n - 2)
        val_n = 1 if n >= 3 else 0
    test_n = n - train_n - val_n
    if test_n <= 0:
        test_n = 1
        train_n = max(1, train_n - 1)

    rng = random.Random(seed)
    best: tuple[float, list[str]] | None = None
    for _ in range(2000):
        shuffled = task_ids[:]
        rng.shuffle(shuffled)
        train = set(shuffled[:train_n])
        val = set(shuffled[train_n : train_n + val_n])
        test = set(shuffled[train_n + val_n :])
        counts = {
            "train": sum(by_task[t] for t in train),
            "val": sum(by_task[t] for t in val),
            "test": sum(by_task[t] for t in test),
        }
        score = min(counts.values()) * 1000 + counts["test"] * 10 + counts["val"]
        if best is None or score > best[0]:
            best = (score, shuffled)
    assert best is not None
    shuffled = best[1]
    split = {
        "train": set(shuffled[:train_n]),
        "val": set(shuffled[train_n : train_n + val_n]),
        "test": set(shuffled[train_n + val_n :]),
    }
    meta = {
        "task_count": n,
        "target_train_tasks": train_n,
        "target_val_tasks": val_n,
        "target_test_tasks": test_n,
        "pair_counts_by_task": dict(by_task),
        "split_task_ids": {name: sorted(vals) for name, vals in split.items()},
    }
    return split, meta


def split_name_for_task(task_id: str, split: dict[str, set[str]]) -> str:
    for name, task_ids in split.items():
        if task_id in task_ids:
            return name
    return "unused"


def mean_or_nan(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(mean(vals)) if vals else float("nan")


def pstdev_or_nan(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(pstdev(vals)) if len(vals) > 1 else 0.0 if vals else float("nan")


def branch_group_subset(rows: Sequence[dict[str, Any]], subset: str) -> list[list[dict[str, Any]]]:
    groups = list(group_rows(rows).values())
    if subset == "all":
        return [g for g in groups if len(g) >= 2]
    if subset == "behaviorally_diverse":
        return [g for g in groups if len(g) >= 2 and group_is_behaviorally_diverse(g)]
    if subset == "reward_diverse":
        return [g for g in groups if len(g) >= 2 and group_is_reward_diverse(g)]
    raise ValueError(f"unknown subset {subset}")


def top2_random_success(rows: Sequence[dict[str, Any]]) -> float:
    k = len(rows)
    good = sum(1 for row in rows if bool(row.get("correct")))
    if k <= 0:
        return 0.0
    if k == 1:
        return 1.0 if good else 0.0
    bad = k - good
    total = k * (k - 1) / 2
    bad_pairs = bad * (bad - 1) / 2
    return 1.0 - bad_pairs / total


def top2_random_reward(rows: Sequence[dict[str, Any]]) -> float:
    vals = [finite_float(row.get("reward"), 0.0) for row in rows]
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    rewards = []
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            rewards.append(max(vals[i], vals[j]))
    return float(mean(rewards)) if rewards else float(mean(vals))


def append_doc_section(path: Path, section_title: str, lines: Sequence[str]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if section_title in existing:
        return False
    prefix = "" if not existing or existing.endswith("\n") else "\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(prefix + "\n".join(lines) + "\n")
    return True


def commands_run() -> list[str]:
    return [
        "venv/bin/python -m py_compile utilities/tests/manual/bg_hidden_origin_tap_inventory.py",
        "venv/bin/python -m py_compile utilities/tests/manual/expand_bg_hidden_origin_branch_dataset.py",
        "venv/bin/python -m py_compile utilities/tests/manual/build_bg_hidden_origin_tap_dataset.py",
        "venv/bin/python -m py_compile utilities/tests/manual/train_bg_hidden_origin_taps.py",
        "venv/bin/python -m py_compile utilities/tests/manual/evaluate_bg_hidden_origin_taps.py",
        "venv/bin/python -m py_compile utilities/tests/manual/analyze_bg_hidden_origin_tap_layers.py",
        "venv/bin/python -m py_compile utilities/tests/manual/analyze_bg_hidden_origin_tap_geometry.py",
        "venv/bin/python -m py_compile utilities/tests/manual/analyze_bg_hidden_origin_tap_experiment.py",
        "venv/bin/python -u utilities/tests/manual/bg_hidden_origin_tap_inventory.py",
        "venv/bin/python -u utilities/tests/manual/expand_bg_hidden_origin_branch_dataset.py",
        "venv/bin/python -u utilities/tests/manual/build_bg_hidden_origin_tap_dataset.py",
        "venv/bin/python -u utilities/tests/manual/train_bg_hidden_origin_taps.py",
        "venv/bin/python -u utilities/tests/manual/evaluate_bg_hidden_origin_taps.py",
        "venv/bin/python -u utilities/tests/manual/analyze_bg_hidden_origin_tap_layers.py",
        "venv/bin/python -u utilities/tests/manual/analyze_bg_hidden_origin_tap_geometry.py",
        "venv/bin/python -u utilities/tests/manual/analyze_bg_hidden_origin_tap_experiment.py",
    ]


class HiddenCapture:
    def __init__(self) -> None:
        self.layer_states: dict[int, list[torch.Tensor]] = {24: [], 30: [], 36: [], 42: []}
        self.boundary: list[torch.Tensor] = []

    def make_layer_hook(self, layer: int):
        def _hook(_module: Any, _args: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            self.layer_states[layer].append(tensor.detach())

        return _hook

    def boundary_hook(self, _module: Any, _args: Any, output: Any) -> None:
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            self.boundary = []
            return
        states = output[1]
        self.boundary = [h.detach() for h in states] if states is not None else []


@contextmanager
def capture_hooks(model: Any, capture: HiddenCapture) -> Iterator[None]:
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if inner is None or layers is None:
        raise RuntimeError("model does not expose model.layers")
    handles = [inner.register_forward_hook(capture.boundary_hook)]
    for layer in (24, 30, 36, 42):
        idx = layer - 1
        if idx < len(layers):
            handles.append(layers[idx].register_forward_hook(capture.make_layer_hook(layer)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def masked_mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    h = hidden.squeeze(0).detach().to(device="cpu", dtype=torch.float32)
    m = attention_mask.squeeze(0).detach().to(device="cpu", dtype=torch.float32).unsqueeze(-1)
    return (h * m).sum(dim=0) / m.sum(dim=0).clamp(min=1.0)


def make_bg_features_from_pooled(pooled: dict[str, torch.Tensor]) -> torch.Tensor:
    by_layer = []
    for layer in BASE_LAYERS:
        loops = []
        for loop in range(1, NUM_LOOPS + 1):
            key = f"L{layer}_L{loop}"
            if key not in pooled:
                raise RuntimeError(f"missing capture key {key}")
            loops.append(pooled[key])
        by_layer.append(torch.stack(loops, dim=0))
    return torch.stack(by_layer, dim=0).to(torch.float32)


def capture_prefix_features(
    model: Any,
    tokenizer: Any,
    prompt: str,
    delta: torch.Tensor,
    spec: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    from src.evaluator.bg_hidden_branching import HiddenDeltaLayerHook, delta_rms

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536, padding=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    if "attention_mask" not in enc:
        enc["attention_mask"] = torch.ones_like(enc["input_ids"], device=device)
    capture = HiddenCapture()
    hook = HiddenDeltaLayerHook(
        model,
        target_layer=int(spec["target_layer"]),
        target_loops=[int(spec["target_loop"])],
        delta=delta,
        position=-1,
        max_rms_fraction=max(delta_rms(delta), 0.02),
    )
    try:
        hook.apply()
        with torch.inference_mode():
            with capture_hooks(model, capture):
                model(**enc, use_cache=False, logits_to_keep=1)
    finally:
        hook.remove()
    pooled: dict[str, torch.Tensor] = {}
    last: dict[str, torch.Tensor] = {}
    for layer in (24, 30, 36, 42):
        for loop_idx, state in enumerate(capture.layer_states.get(layer, [])[:NUM_LOOPS], start=1):
            key = f"L{layer}_L{loop_idx}"
            pooled[key] = masked_mean_pool(state, enc["attention_mask"])
            last[key] = state[0, -1, :].detach().cpu().to(torch.float32)
    for loop_idx, state in enumerate(capture.boundary[:NUM_LOOPS], start=1):
        key = f"L47_L{loop_idx}"
        pooled[key] = masked_mean_pool(state, enc["attention_mask"])
        last[key] = state[0, -1, :].detach().cpu().to(torch.float32)
    return {
        "features": make_bg_features_from_pooled(pooled),
        "pooled_vectors": pooled,
        "last_token_vectors": last,
        "hook_diagnostics": hook.diagnostics(),
        "nan_inf": not all(torch.isfinite(v).all().item() for v in pooled.values()),
    }


def generate_with_hook(
    model: Any,
    tokenizer: Any,
    prompt: str,
    delta: torch.Tensor,
    spec: dict[str, Any],
    device: torch.device,
    max_new_tokens: int = 128,
) -> dict[str, Any]:
    from src.evaluator.bg_hidden_branching import HiddenDeltaLayerHook, delta_rms

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536, padding=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    prompt_len = int(enc["input_ids"].shape[1])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    hook = HiddenDeltaLayerHook(
        model,
        target_layer=int(spec["target_layer"]),
        target_loops=[int(spec["target_loop"])],
        delta=delta,
        position=-1,
        max_rms_fraction=max(delta_rms(delta), 0.02),
    )
    started = time.time()
    try:
        hook.apply()
        with torch.inference_mode():
            generated = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )
    finally:
        hook.remove()
    new_ids = generated[0, prompt_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return {
        "output_text": text,
        "token_count": int(new_ids.numel()),
        "hit_max_tokens": int(new_ids.numel()) >= int(max_new_tokens),
        "generation_seconds": round(time.time() - started, 3),
        "hook_diagnostics": hook.diagnostics(),
    }
