"""Shared helpers for the BG pre-consolidation control probe bundle."""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "artifacts" / "hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "artifacts" / "hf_cache" / "datasets"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

OUT_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_preconsolidation_control_probes_2026-05-18"
STAGE1_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_trajectory_prediction_2026-05-18"
STAGE2_LAYERHOOK_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_stage2_layerhook_followup_2026-05-18"
EMPIRICAL_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_empirical_steering_direction_2026-05-18"
STEERING_SUITE_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_steering_suite_2026-05-18"
MODEL_PATH = PROJECT_ROOT / "shared/models/ouro_rltt_local"
SEED = 20260518
POST_INTERVENTION_TOKENS = 32

TARGET_REQUESTS = [
    {"target_id": "T1", "domain": "reasoning", "prefix_length": 64, "quota": 6, "primary": True},
    {"target_id": "T3", "domain": "science", "prefix_length": 32, "quota": 4, "primary": False},
    {"target_id": "T4", "domain": "gsm8k", "prefix_length": 256, "quota": 4, "primary": False},
]


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(p)


def load_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(p)


def write_md(path: str | Path, lines: Iterable[str]) -> None:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(str(line) for line in lines) + "\n", encoding="utf-8")


def append_once(path: str | Path, title: str, lines: list[str]) -> None:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = p.read_text(encoding="utf-8") if p.exists() else ""
    marker = f"## {title}"
    if marker in existing:
        return
    p.write_text(existing.rstrip() + "\n\n" + marker + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def avg(values: Iterable[float]) -> float | None:
    vals = []
    for value in values:
        try:
            numeric = float(value)
        except Exception:
            continue
        if math.isfinite(numeric):
            vals.append(numeric)
    return sum(vals) / len(vals) if vals else None


def rate(values: Iterable[bool]) -> float | None:
    vals = [bool(v) for v in values]
    return sum(1 for value in vals if value) / len(vals) if vals else None


def unit(vec: torch.Tensor) -> torch.Tensor:
    v = vec.detach().flatten().to(dtype=torch.float32, device="cpu")
    return v / torch.linalg.vector_norm(v).clamp(min=1e-12)


def rms(vec: torch.Tensor) -> torch.Tensor:
    return vec.detach().flatten().to(dtype=torch.float32, device="cpu").pow(2).mean().sqrt().clamp(min=1e-12)


def rms_unit(vec: torch.Tensor) -> torch.Tensor:
    v = vec.detach().flatten().to(dtype=torch.float32, device="cpu")
    return v / rms(v)


def random_l2_direction(dim: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    return unit(torch.randn(int(dim), generator=gen, dtype=torch.float32))


def random_rms_direction(dim: int, seed: int) -> torch.Tensor:
    return rms_unit(random_l2_direction(dim, seed))


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(unit(a), unit(b)).item())


def load_stage1_tasks() -> dict[str, dict[str, Any]]:
    payload = load_json(STAGE1_ROOT / "task_suite.json", {})
    return {str(row.get("task_id")): row for row in payload.get("tasks") or []}


def load_continued_index() -> dict[tuple[str, int, int], dict[str, Any]]:
    payload = load_json(STAGE1_ROOT / "continued_prefixes.json", {})
    return {
        (str(row.get("task_id")), int(row.get("branch_id", -1)), int(row.get("prefix_length", -1))): row
        for row in payload.get("continued_prefixes") or []
    }


def load_partials_index() -> dict[tuple[str, int], dict[str, Any]]:
    payload = load_json(STAGE1_ROOT / "partials.json", {})
    return {
        (str(row.get("task_id")), int(row.get("branch_id", -1))): row
        for row in payload.get("branches") or []
    }


def expected_answer(task: dict[str, Any]) -> str:
    return str(task.get("answer_key") or task.get("gold_answer") or task.get("answer") or "")


def success_label(row: dict[str, Any]) -> bool:
    return bool(row.get("is_correct") or row.get("evaluation", {}).get("success"))


def select_targets() -> list[dict[str, Any]]:
    from bg_empirical_steering_common import select_best_nonorm_cell

    targets: list[dict[str, Any]] = []
    for req in TARGET_REQUESTS:
        cell = select_best_nonorm_cell(str(req["domain"]), int(req["prefix_length"]))
        if not cell:
            continue
        targets.append(
            {
                **req,
                "head_id": cell["head_id"],
                "family": cell.get("family"),
                "config": cell["config"],
                "architecture": cell["architecture"],
                "layer": int(str(cell["config"]).split("_", 1)[0]),
                "top1_lift": cell.get("top1_lift"),
                "pairwise_accuracy": cell.get("pairwise_predictive_accuracy"),
                "oracle_success": cell.get("oracle_success"),
            }
        )
    return targets


def load_direction_payload() -> dict[str, Any]:
    path = EMPIRICAL_ROOT / "directions.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    return {}


def target_directions(target: dict[str, Any], controller: Any) -> list[dict[str, Any]]:
    from bg_stage2_steering_preflight import head_weight_vector, load_head_row

    directions: list[dict[str, Any]] = []
    payload = load_direction_payload()
    existing = next(
        (
            row
            for row in payload.get("targets", [])
            if str(row.get("domain")) == str(target["domain"])
            and int(row.get("prefix_length", -1)) == int(target["prefix_length"])
        ),
        None,
    )
    if existing:
        for row in existing.get("directions", []):
            if row.get("direction_name") in {
                "RAW_NONORM_READOUT",
                "EMPIRICAL_MEAN_DIFF",
                "EMPIRICAL_WHITENED_DIFF",
                "LOGISTIC_SUCCESS_PROBE",
            }:
                vec = unit(row["vector"])
                directions.append(
                    {
                        **{k: v for k, v in row.items() if k != "vector"},
                        "vector": vec,
                        "direction_dim": int(vec.numel()),
                        "source_target_id": existing.get("target_id"),
                    }
                )
    if not any(row.get("direction_name") == "RAW_NONORM_READOUT" for row in directions):
        head = load_head_row(str(target["head_id"]), controller)
        vec = unit(head_weight_vector(head))
        directions.insert(
            0,
            {
                "direction_name": "RAW_NONORM_READOUT",
                "direction_source": str(target["head_id"]),
                "vector": vec,
                "direction_dim": int(vec.numel()),
                "direction_norm": float(torch.linalg.vector_norm(vec).item()),
                "source_target_id": target["target_id"],
            },
        )
    return [row for row in directions if int(row.get("direction_dim", row["vector"].numel())) == 2048]


def task_subset_for_target(target: dict[str, Any], max_tasks: int | None = None) -> list[dict[str, Any]]:
    tasks = load_stage1_tasks()
    partials = load_partials_index()
    continued = load_continued_index()
    domain = str(target["domain"])
    prefix_length = int(target["prefix_length"])
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (task_id, branch_id, pfx), row in continued.items():
        if int(pfx) == prefix_length and str(row.get("domain")) == domain and row.get("evaluable", True):
            by_task[task_id].append(row)
    rows: list[dict[str, Any]] = []
    for task_id, cont_rows in sorted(by_task.items()):
        task = tasks.get(task_id)
        if not task or not expected_answer(task):
            continue
        successes = [row for row in cont_rows if success_label(row)]
        if not successes:
            continue
        chosen = sorted(successes, key=lambda row: int(row.get("branch_id", 0)))[0]
        branch_id = int(chosen.get("branch_id", 0))
        branch = partials.get((task_id, branch_id), {})
        prefix_text = str(chosen.get("prefix_text") or branch.get("prefixes", {}).get(f"prefix_{prefix_length}") or "")
        if not prefix_text.strip():
            continue
        rows.append(
            {
                "suite_index": len(rows),
                "task_id": task_id,
                "target_id": target["target_id"],
                "domain": domain,
                "prefix_length": prefix_length,
                "branch_id": branch_id,
                "prompt": str(task.get("prompt") or task.get("question") or ""),
                "prefix_text": prefix_text,
                "expected_answer": expected_answer(task),
                "stage1_continuation_text": str(chosen.get("continuation_text") or ""),
                "stage1_is_correct": success_label(chosen),
                "stage1_oracle_successful_branches": len(successes),
                "task": task,
            }
        )
        if max_tasks is not None and len(rows) >= int(max_tasks):
            break
    return rows


def condition_plan(alphas: list[float] | None = None, extra_random_at_001: bool = True) -> list[dict[str, Any]]:
    alpha_values = alphas or [0.005, 0.01, 0.02]
    rows = [{"alpha": 0.0, "condition": "zero_baseline", "random_control_idx": 0, "random_control_n": 0}]
    for alpha in alpha_values:
        random_n = 3 if extra_random_at_001 and abs(float(alpha) - 0.01) < 1e-9 else 1
        rows.append({"alpha": float(alpha), "condition": "positive", "random_control_idx": 0, "random_control_n": random_n})
        rows.append({"alpha": float(alpha), "condition": "negative", "random_control_idx": 0, "random_control_n": random_n})
        for idx in range(random_n):
            rows.append({"alpha": float(alpha), "condition": "random", "random_control_idx": idx, "random_control_n": random_n})
    return rows


def generation_seed(*parts: int) -> int:
    out = SEED
    for idx, part in enumerate(parts):
        out += int(part) * (10 ** max(0, 6 - idx))
    return int(out)


def summarize_cell(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cond: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cond[str(row.get("condition"))].append(row)
    pos = [finite(row.get("z_score_change")) for row in by_cond.get("positive", [])]
    neg = [finite(row.get("z_score_change")) for row in by_cond.get("negative", [])]
    rnd = [finite(row.get("z_score_change")) for row in by_cond.get("random", [])]
    pos_mean = avg(pos)
    neg_mean = avg(neg)
    rnd_mean = avg(rnd)
    rnd_std = pstdev(rnd) if len(rnd) > 1 else None
    threshold = 0.0 if rnd_std is None else 0.5 * rnd_std
    signed = pos_mean is not None and neg_mean is not None and rnd_mean is not None and pos_mean > rnd_mean and neg_mean < rnd_mean
    strong = (
        signed
        and rnd_std is not None
        and pos_mean is not None
        and neg_mean is not None
        and rnd_mean is not None
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
        "rms_change_mean": avg(finite(row.get("activation_rms_change")) for row in intervention),
        "effective_delta_rms_fraction_mean": avg(finite(row.get("effective_delta_rms_fraction")) for row in intervention),
        "parse_failed_rate": rate(bool(row.get("parse_failed")) for row in intervention),
        "empty_output_rate": rate(bool(row.get("empty_output")) for row in intervention),
        "hit_max_tokens_rate": rate(bool(row.get("hit_max_tokens")) for row in intervention),
        "repetition_rate_mean": avg(finite(row.get("repetition_rate")) for row in intervention),
        "output_length_mean": avg(finite(row.get("output_length")) for row in intervention),
        "success_rate_positive": rate(bool(row.get("is_correct")) for row in by_cond.get("positive", [])),
    }
