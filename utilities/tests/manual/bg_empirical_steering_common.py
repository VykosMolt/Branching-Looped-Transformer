"""Shared helpers for the BG empirical steering-direction probe.

The helpers in this file operate on frozen Stage 1/Stage 2 artifacts.  They do
not modify Ouro weights, tokenizer files, checkpoints, or BG heads.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "artifacts" / "hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "artifacts" / "hf_cache" / "datasets"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

OUT_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_empirical_steering_direction_2026-05-18"
STAGE1_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_trajectory_prediction_2026-05-18"
STAGE2_LAYERHOOK_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_stage2_layerhook_followup_2026-05-18"
MODEL_PATH = PROJECT_ROOT / "shared/models/ouro_rltt_local"
SEED = 20260518

TARGET_REQUESTS = [
    {"target_id": "T1", "domain": "reasoning", "prefix_length": 64, "primary": True},
    {"target_id": "T3", "domain": "science", "prefix_length": 32, "primary": False},
    {"target_id": "T4", "domain": "gsm8k", "prefix_length": 256, "primary": False},
    {"target_id": "T2", "domain": "reasoning", "prefix_length": 256, "primary": False},
]

STAGE1_REQUIRED = [
    "predictive_power.json",
    "prefix_features.pt",
    "prefix_scores.json",
    "continued_prefixes.json",
    "task_suite.json",
]

STAGE2_REQUIRED = [
    "summary.md",
    "analysis.md",
    "layerhook_followup_traces.json",
]


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(p)


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_once(path: Path, title: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    marker = f"## {title}"
    if marker in existing:
        return
    path.write_text(existing.rstrip() + "\n\n" + marker + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def rate(values: list[bool]) -> float | None:
    return sum(1 for value in values if value) / len(values) if values else None


def unit(vec: torch.Tensor) -> torch.Tensor:
    v = vec.detach().flatten().to(dtype=torch.float32, device="cpu")
    return v / torch.linalg.vector_norm(v).clamp(min=1e-12)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = unit(a)
    bb = unit(b)
    return float(torch.dot(aa, bb).item())


def random_direction(dim: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    vec = torch.randn(int(dim), generator=gen, dtype=torch.float32)
    return unit(vec)


def strength(row: dict[str, Any]) -> float:
    pair = row.get("pairwise_predictive_accuracy")
    pair_component = float(pair) - 0.5 if pair is not None else -1.0
    return max(float(row.get("top1_lift", 0.0)), float(row.get("top2_lift", 0.0)), pair_component)


def layer_from_config(config: str) -> int:
    return int(str(config).split("_", 1)[0])


def load_predictive_cells() -> list[dict[str, Any]]:
    payload = load_json(STAGE1_ROOT / "predictive_power.json", {})
    return list(payload.get("all_cells") or [])


def select_best_nonorm_cell(domain: str, prefix_length: int) -> dict[str, Any] | None:
    from src.evaluator.bg_controller import config_dim

    rows = [
        dict(row)
        for row in load_predictive_cells()
        if str(row.get("domain")) == str(domain)
        and int(row.get("prefix_length", -1)) == int(prefix_length)
        and str(row.get("architecture")) == "AntisymLinearNoNorm"
        and "concat" not in str(row.get("config", ""))
        and config_dim(str(row.get("config"))) == 2048
    ]
    rows.sort(key=lambda row: (-strength(row), -float(row.get("oracle_success", 0.0)), str(row.get("head_id"))))
    return rows[0] if rows else None


def find_diagnostic_d1() -> dict[str, Any] | None:
    for row in load_predictive_cells():
        if (
            str(row.get("domain")) == "reasoning"
            and int(row.get("prefix_length", -1)) == 256
            and str(row.get("family")) == "MIX_CODE_REASONING"
            and str(row.get("config")) == "36_mean"
            and str(row.get("architecture")) == "AntisymLinear"
        ):
            return dict(row)
    return None


def load_continuation_index() -> dict[tuple[str, int, int], dict[str, Any]]:
    payload = load_json(STAGE1_ROOT / "continued_prefixes.json", {})
    out: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in payload.get("continued_prefixes") or []:
        out[(str(row.get("task_id")), int(row.get("branch_id", -1)), int(row.get("prefix_length", -1)))] = row
    return out


def load_stage1_tasks() -> dict[str, dict[str, Any]]:
    payload = load_json(STAGE1_ROOT / "task_suite.json", {})
    return {str(row.get("task_id")): row for row in payload.get("tasks") or []}


def load_partials_branch_index() -> dict[tuple[str, int], dict[str, Any]]:
    payload = load_json(STAGE1_ROOT / "partials.json", {})
    return {
        (str(row.get("task_id")), int(row.get("branch_id", -1))): row
        for row in payload.get("branches") or []
    }


def expected_answer(task: dict[str, Any]) -> str:
    return str(task.get("answer_key") or task.get("gold_answer") or task.get("answer") or "")


def success_label(row: dict[str, Any]) -> bool:
    return bool(row.get("is_correct") or row.get("evaluation", {}).get("success"))


def load_feature_records() -> list[dict[str, Any]]:
    path = STAGE1_ROOT / "prefix_features.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return list(payload.get("records") or [])


def examples_for_target(domain: str, prefix_length: int, config: str) -> list[dict[str, Any]]:
    from src.evaluator.bg_controller import config_vector

    continuation = load_continuation_index()
    examples = []
    for record in load_feature_records():
        if str(record.get("domain")) != str(domain) or int(record.get("prefix_length", -1)) != int(prefix_length):
            continue
        key = (str(record.get("task_id")), int(record.get("branch_id", -1)), int(prefix_length))
        cont = continuation.get(key)
        if not cont or not cont.get("evaluable", True):
            continue
        vec = config_vector(record["features"], config).detach().flatten().to(dtype=torch.float32, device="cpu")
        examples.append(
            {
                "task_id": str(record.get("task_id")),
                "branch_id": int(record.get("branch_id", -1)),
                "domain": str(domain),
                "prefix_length": int(prefix_length),
                "config": str(config),
                "x": vec,
                "success": success_label(cont),
                "continued_row": cont,
            }
        )
    return examples


def contrast_counts(domain: str, prefix_length: int, config: str) -> dict[str, Any]:
    examples = examples_for_target(domain, prefix_length, config)
    success = sum(1 for row in examples if row["success"])
    failure = sum(1 for row in examples if not row["success"])
    return {
        "domain": domain,
        "prefix_length": prefix_length,
        "config": config,
        "example_count": len(examples),
        "successful_prefixes": success,
        "failed_prefixes": failure,
        "task_count": len({row["task_id"] for row in examples}),
        "meets_minimum": success >= 10 and failure >= 10,
    }


def heldout_split(examples: list[dict[str, Any]], heldout_task_count: int = 8) -> tuple[set[str], set[str]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in examples:
        by_task[str(row["task_id"])].append(row)
    candidates = []
    for task_id, rows in by_task.items():
        has_success = any(row["success"] for row in rows)
        has_failure = any(not row["success"] for row in rows)
        oracle_success = has_success
        candidates.append((not oracle_success, not has_failure, task_id))
    candidates.sort()
    task_ids = [task_id for _no_success, _no_failure, task_id in candidates]
    heldout: set[str] = set()
    for task_id in reversed(task_ids):
        if len(heldout) >= heldout_task_count:
            break
        heldout.add(task_id)
    train = set(task_ids) - heldout
    # Keep enough training contrast by moving heldout tasks back if needed.
    while heldout and not _split_has_contrast(examples, train):
        task_id = sorted(heldout)[0]
        heldout.remove(task_id)
        train.add(task_id)
    return train, heldout


def _split_has_contrast(examples: list[dict[str, Any]], train_tasks: set[str]) -> bool:
    rows = [row for row in examples if row["task_id"] in train_tasks]
    return sum(1 for row in rows if row["success"]) >= 10 and sum(1 for row in rows if not row["success"]) >= 10


def stack_xy(examples: list[dict[str, Any]], task_ids: set[str] | None = None) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    rows = [row for row in examples if task_ids is None or row["task_id"] in task_ids]
    x = torch.stack([row["x"] for row in rows], dim=0).to(dtype=torch.float32)
    y = torch.tensor([1.0 if row["success"] else 0.0 for row in rows], dtype=torch.float32)
    return x, y, rows


def auc_score(y_true: torch.Tensor, scores: torch.Tensor) -> float | None:
    y = y_true.detach().cpu().flatten()
    s = scores.detach().cpu().flatten()
    pos = s[y > 0.5]
    neg = s[y <= 0.5]
    if int(pos.numel()) == 0 or int(neg.numel()) == 0:
        return None
    wins = 0.0
    total = 0
    for p in pos:
        wins += float((p > neg).sum().item()) + 0.5 * float((p == neg).sum().item())
        total += int(neg.numel())
    return wins / max(total, 1)


def pairwise_accuracy(y_true: torch.Tensor, scores: torch.Tensor) -> float | None:
    return auc_score(y_true, scores)


def train_logistic_direction(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_heldout: torch.Tensor,
    y_heldout: torch.Tensor,
    *,
    seed: int,
    epochs: int = 250,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    mu = x_train.mean(dim=0)
    sigma = x_train.std(dim=0, unbiased=False).clamp(min=1e-4)
    zx = (x_train - mu) / sigma
    zh = (x_heldout - mu) / sigma
    linear = nn.Linear(int(x_train.shape[1]), 1, bias=True)
    opt = torch.optim.AdamW(linear.parameters(), lr=0.03, weight_decay=0.03)
    best = {"loss": float("inf"), "state": None}
    for _epoch in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        logits = linear(zx).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y_train)
        loss.backward()
        opt.step()
        value = float(loss.detach().cpu().item())
        if value < best["loss"]:
            best = {"loss": value, "state": {k: v.detach().clone() for k, v in linear.state_dict().items()}}
    if best["state"] is not None:
        linear.load_state_dict(best["state"])
    with torch.no_grad():
        train_scores = linear(zx).squeeze(-1)
        heldout_scores = linear(zh).squeeze(-1)
        # Convert the standardized-space classifier weight back to original feature coordinates.
        raw_weight = linear.weight.detach().flatten().cpu() / sigma
    return {
        "direction": unit(raw_weight),
        "norm_before": float(torch.linalg.vector_norm(raw_weight).item()),
        "train_auc": auc_score(y_train, train_scores),
        "heldout_auc": auc_score(y_heldout, heldout_scores),
        "heldout_pairwise_accuracy": pairwise_accuracy(y_heldout, heldout_scores),
        "train_loss": float(best["loss"]),
    }


def summarize_cell(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cond: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cond[str(row.get("condition"))].append(row)
    pos = [finite(row.get("z_score_change")) for row in by_cond.get("positive", [])]
    neg = [finite(row.get("z_score_change")) for row in by_cond.get("negative", [])]
    rnd = [finite(row.get("z_score_change")) for row in by_cond.get("random", [])]
    base = by_cond.get("zero_baseline", [])
    pos_mean = avg(pos)
    neg_mean = avg(neg)
    rnd_mean = avg(rnd)
    rnd_std = pstdev(rnd) if len(rnd) > 1 else None
    threshold = 0.0 if rnd_std is None else 0.5 * rnd_std
    signed = (
        pos_mean is not None
        and neg_mean is not None
        and rnd_mean is not None
        and pos_mean > rnd_mean
        and neg_mean < rnd_mean
    )
    strong = (
        pos_mean is not None
        and neg_mean is not None
        and rnd_mean is not None
        and rnd_std is not None
        and pos_mean >= rnd_mean + threshold
        and neg_mean <= rnd_mean - threshold
    )
    unsigned = pos_mean is not None and rnd_mean is not None and pos_mean > rnd_mean
    intervention = [row for row in rows if row.get("condition") != "zero_baseline"]
    stable = bool(intervention) and not any(
        row.get("safety_status") == "DESTABILIZING"
        or row.get("cuda_error")
        or row.get("nan_or_inf_activations")
        for row in intervention
    )
    base_success = rate([bool(row.get("is_correct")) for row in base])
    pos_success = rate([bool(row.get("is_correct")) for row in by_cond.get("positive", [])])
    return {
        "row_count": len(rows),
        "positive_z_mean": pos_mean,
        "negative_z_mean": neg_mean,
        "random_z_mean": rnd_mean,
        "random_z_std": rnd_std,
        "signed_causal_signature": signed,
        "strong_signed_causal_signature": strong,
        "unsigned_effect": unsigned,
        "signed_score": finite((pos_mean - rnd_mean) if pos_mean is not None and rnd_mean is not None else 0.0)
        + finite((rnd_mean - neg_mean) if neg_mean is not None and rnd_mean is not None else 0.0),
        "stable": stable,
        "rms_change_mean": avg([finite(row.get("activation_rms_change")) for row in intervention]),
        "parse_failed_rate": rate([bool(row.get("parse_failed")) for row in intervention]),
        "empty_output_rate": rate([bool(row.get("empty_output")) for row in intervention]),
        "hit_max_tokens_rate": rate([bool(row.get("hit_max_tokens")) for row in intervention]),
        "repetition_rate_mean": avg([finite(row.get("repetition_rate")) for row in intervention]),
        "output_length_mean": avg([finite(row.get("output_length")) for row in intervention]),
        "success_rate_baseline": base_success,
        "success_rate_positive": pos_success,
        "positive_success_lift_vs_baseline": (
            pos_success - base_success if pos_success is not None and base_success is not None else None
        ),
    }
