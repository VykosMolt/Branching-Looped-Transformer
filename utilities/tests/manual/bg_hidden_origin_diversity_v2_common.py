"""Shared helpers for hidden-origin branch diversity v2 probes.

This module is intentionally limited to manual probe scripts.  It loads frozen
Ouro for inference only, writes only v2 report artifacts, and never mutates
model weights, tokenizer files, checkpoints, old BG tap registries, or git
state.
"""
from __future__ import annotations

import json
import math
import random
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Iterator, Sequence

import torch

from bg_hidden_origin_tap_common import (
    CONFIGS,
    HIDDEN_DIM,
    MODEL_PATH,
    OLD_ROOT,
    OUT_ROOT as V1_ROOT,
    PROBE_ROOT,
    PROJECT_ROOT,
    append_doc_section,
    available_configs_for_rows,
    branch_key,
    config_dim,
    config_vector_from_row,
    cosine,
    evaluate_mcq,
    finite_float,
    group_rows,
    hidden_branch_prompt,
    is_safe_alpha,
    load_all_branch_rows,
    load_candidate_tasks,
    load_json,
    md_table,
    normalize_task,
    parse_mcq_answer,
    rate,
    rel,
    stable_row,
    tensor_stats,
    write_csv,
    write_json,
    write_md,
)
from src.evaluator.bg_hidden_branching import delta_rms, rms_normalize


V2_ROOT = PROBE_ROOT / "bg_hidden_origin_diversity_v2_2026-05-18"
SEED = 20260518
MAX_NEW_TOKENS = 128
PRIMARY_ALPHA_CAP = 0.0100001
DIAGNOSTIC_ALPHA = 0.02
TASK_SCREENING_JSON = V2_ROOT / "task_screening.json"
DIRECTION_BANK_PT = V2_ROOT / "direction_bank.pt"
DIVERSE_BRANCHES_PT = V2_ROOT / "diverse_hidden_origin_branches.pt"
DIVERSE_BRANCHES_PARTIAL_PT = V2_ROOT / "diverse_hidden_origin_branches.partial.pt"
DATASET_V2_PT = V2_ROOT / "hidden_origin_tap_dataset_v2.pt"
HEADS_V2_PT = V2_ROOT / "hidden_origin_tap_heads_v2.pt"


def ensure_v2_root() -> None:
    V2_ROOT.mkdir(parents=True, exist_ok=True)


def alpha_bucket(row_or_alpha: dict[str, Any] | float | int | str) -> str:
    value = finite_float(row_or_alpha.get("alpha") if isinstance(row_or_alpha, dict) else row_or_alpha, float("nan"))
    if not math.isfinite(value):
        return "alpha_unknown"
    if abs(value - 0.005) <= 1e-6:
        return "alpha_0_005"
    if abs(value - 0.010) <= 1e-6:
        return "alpha_0_01"
    if abs(value - 0.020) <= 1e-6:
        return "alpha_0_02"
    return f"alpha_{value:g}"


def deterministic_reward(row: dict[str, Any]) -> float:
    return finite_float(row.get("deterministic_reward", row.get("reward")), 0.0)


def deterministic_correct(row: dict[str, Any]) -> bool:
    return bool(row.get("deterministic_correct", row.get("correct", False)))


def sampled_reward(row: dict[str, Any]) -> float | None:
    value = row.get("sampled_expected_reward")
    if value is None:
        return None
    out = finite_float(value, float("nan"))
    return out if math.isfinite(out) else None


def row_reward(row: dict[str, Any], label_source: str = "deterministic") -> float:
    if label_source == "sampled_expected":
        value = sampled_reward(row)
        if value is not None:
            return value
    return deterministic_reward(row)


def group_is_behaviorally_diverse_v2(rows: Sequence[dict[str, Any]], label_source: str = "deterministic") -> bool:
    rewards = {row_reward(row, label_source) for row in rows}
    correct = {deterministic_correct(row) for row in rows}
    answers = {str(row.get("parsed_answer")) for row in rows}
    return len(rewards) > 1 or len(correct) > 1 or len(answers) > 1


def group_is_reward_diverse_v2(rows: Sequence[dict[str, Any]], label_source: str = "deterministic") -> bool:
    return len({row_reward(row, label_source) for row in rows}) > 1


def safe_primary_row(row: dict[str, Any]) -> bool:
    return finite_float(row.get("alpha"), 999.0) <= PRIMARY_ALPHA_CAP and bool(row.get("safety_envelope", True))


def diagnostic_alpha_row(row: dict[str, Any]) -> bool:
    return abs(finite_float(row.get("alpha"), -1.0) - DIAGNOSTIC_ALPHA) <= 1e-6


def stable_v2_row(row: dict[str, Any]) -> bool:
    normalized = dict(row)
    if "reward" not in normalized and "deterministic_reward" in normalized:
        normalized["reward"] = normalized["deterministic_reward"]
    if "correct" not in normalized and "deterministic_correct" in normalized:
        normalized["correct"] = normalized["deterministic_correct"]
    return stable_row(normalized)


def compact_branch_row(row: dict[str, Any]) -> dict[str, Any]:
    skip = {"features", "pooled_vectors", "last_token_vectors", "delta"}
    out = {k: v for k, v in row.items() if k not in skip}
    if isinstance(row.get("features"), torch.Tensor):
        out["features_shape"] = list(row["features"].shape)
    pooled = row.get("pooled_vectors") or {}
    out["pooled_vector_keys"] = sorted(pooled.keys())
    if isinstance(row.get("delta"), torch.Tensor):
        out["delta_stats"] = tensor_stats(row["delta"])
    return out


def load_v2_diverse_rows() -> list[dict[str, Any]]:
    path = DIVERSE_BRANCHES_PT if DIVERSE_BRANCHES_PT.exists() else DIVERSE_BRANCHES_PARTIAL_PT
    if not path.exists():
        return []
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return [dict(row) for row in list(payload.get("rows") or payload.get("records") or []) if row.get("branch_group_id")]


def load_all_v2_branch_rows(include_prior: bool = True) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    rows = []
    if include_prior:
        rows.extend(load_all_branch_rows())
    rows.extend(load_v2_diverse_rows())
    for row in rows:
        if row.get("branch_group_id") is None:
            continue
        normalized = dict(row)
        if "deterministic_reward" not in normalized:
            normalized["deterministic_reward"] = finite_float(normalized.get("reward"), 0.0)
        if "reward" not in normalized:
            normalized["reward"] = normalized["deterministic_reward"]
        if "deterministic_correct" not in normalized:
            normalized["deterministic_correct"] = bool(normalized.get("correct"))
        if "correct" not in normalized:
            normalized["correct"] = bool(normalized["deterministic_correct"])
        if "alpha_bucket" not in normalized:
            normalized["alpha_bucket"] = alpha_bucket(normalized)
        by_key[branch_key(normalized)] = normalized
    return list(by_key.values())


def load_more_candidate_tasks() -> list[dict[str, Any]]:
    """Load reasoning/science MCQ tasks from existing local probe suites."""
    sources = [
        (OLD_ROOT / "task_subset.json", None),
        (PROBE_ROOT / "bg_trajectory_prediction_2026-05-18/task_suite.json", None),
        (PROBE_ROOT / "bg_stage2_layerhook_followup_2026-05-18/task_subset.json", None),
        (PROBE_ROOT / "bg_stage2_steering_2026-05-18/task_suite.json", None),
        (PROBE_ROOT / "bg_steering_suite_2026-05-18/task_suite.json", None),
        (PROBE_ROOT / "reasoning_branch_pilot_2026-05-17.json", "reasoning"),
        (PROBE_ROOT / "science_natural_distractor_set_2026-05-17.json", "science"),
    ]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for task in load_candidate_tasks():
        if task["task_id"] not in seen:
            rows.append(task)
            seen.add(task["task_id"])
    for path, forced_domain in sources:
        payload = load_json(path, {}) or {}
        items = payload.get("tasks") or payload.get("rows") or []
        for idx, item in enumerate(items):
            task = normalize_task(item, forced_domain, idx)
            if not task or task["task_id"] in seen:
                continue
            rows.append(task)
            seen.add(task["task_id"])
    return rows


def balanced_tasks(tasks: Sequence[dict[str, Any]], limit: int, seed: int = SEED) -> list[dict[str, Any]]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_domain[str(task.get("domain"))].append(dict(task))
    rng = random.Random(seed)
    for vals in by_domain.values():
        rng.shuffle(vals)
    selected: list[dict[str, Any]] = []
    per_domain = max(1, int(limit) // max(len(by_domain), 1))
    for domain in ("reasoning", "science"):
        selected.extend(by_domain.get(domain, [])[:per_domain])
    if len(selected) < limit:
        used = {task["task_id"] for task in selected}
        leftovers = [task for task in tasks if task["task_id"] not in used]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: max(0, limit - len(selected))])
    return selected[:limit]


def classify_screening_row(row: dict[str, Any]) -> str:
    if row.get("unstable"):
        return "baseline_empty_or_unstable"
    if row.get("perturbation_sensitive"):
        return "perturbation_sensitive"
    if not row.get("clean_parse_success"):
        return "baseline_parse_fragile"
    if not row.get("clean_correct"):
        return "baseline_wrong_parseable"
    margin = finite_float(row.get("answer_margin"), float("nan"))
    if math.isfinite(margin) and margin < 1.0:
        return "baseline_correct_low_confidence"
    return "baseline_correct_confident"


def selected_screening_classes() -> set[str]:
    return {
        "baseline_correct_low_confidence",
        "baseline_wrong_parseable",
        "baseline_parse_fragile",
        "perturbation_sensitive",
    }


def clean_generation(model: Any, tokenizer: Any, prompt: str, device: torch.device, max_new_tokens: int = MAX_NEW_TOKENS) -> dict[str, Any]:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536, padding=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    prompt_len = int(enc["input_ids"].shape[1])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    started = time.time()
    with torch.inference_mode():
        generated = model.generate(
            **enc,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
    new_ids = generated[0, prompt_len:]
    return {
        "output_text": tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
        "token_count": int(new_ids.numel()),
        "hit_max_tokens": int(new_ids.numel()) >= int(max_new_tokens),
        "generation_seconds": round(time.time() - started, 3),
    }


def answer_logit_margin(model: Any, tokenizer: Any, task: dict[str, Any], device: torch.device) -> dict[str, Any]:
    letters = sorted(str(k).upper() for k in task.get("options", {}))
    if len(letters) < 2:
        return {"answer_margin": float("nan"), "answer_logits": {}}
    enc = tokenizer(task["prompt"], return_tensors="pt", truncation=True, max_length=1536, padding=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        logits = model(**enc, use_cache=False).logits[0, -1].detach().float().cpu()
    scores: dict[str, float] = {}
    for letter in letters:
        candidates = []
        for text in (letter, " " + letter):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 1:
                candidates.append(float(logits[int(ids[0])].item()))
        scores[letter] = max(candidates) if candidates else float("nan")
    finite = sorted((v, k) for k, v in scores.items() if math.isfinite(v))
    if len(finite) < 2:
        margin = float("nan")
    else:
        margin = float(finite[-1][0] - finite[-2][0])
    return {"answer_margin": margin, "answer_logits": scores}


def generate_with_hook_v2(
    model: Any,
    tokenizer: Any,
    prompt: str,
    delta: torch.Tensor,
    spec: dict[str, Any],
    device: torch.device,
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> dict[str, Any]:
    from src.evaluator.bg_hidden_branching import HiddenDeltaLayerHook

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
        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(do_sample),
            "pad_token_id": pad_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": False,
        }
        if do_sample:
            gen_kwargs.update({"temperature": float(temperature), "top_p": float(top_p)})
        with torch.inference_mode():
            generated = model.generate(**enc, **gen_kwargs)
    finally:
        hook.remove()
    new_ids = generated[0, prompt_len:]
    return {
        "output_text": tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
        "token_count": int(new_ids.numel()),
        "hit_max_tokens": int(new_ids.numel()) >= int(max_new_tokens),
        "generation_seconds": round(time.time() - started, 3),
        "hook_diagnostics": hook.diagnostics(),
        "do_sample": bool(do_sample),
    }


def orthogonalize(candidate: torch.Tensor, basis: Iterable[torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    out = candidate.detach().flatten().to(torch.float32).clone()
    for base in basis:
        unit = rms_normalize(base.flatten(), eps=eps)
        denom = torch.dot(unit, unit).clamp(min=eps)
        out = out - torch.dot(out, unit) / denom * unit
    return rms_normalize(out, eps=eps)


def load_direction_bank_payload() -> dict[str, Any]:
    if not DIRECTION_BANK_PT.exists():
        return {"directions_by_layer": {}, "directions": []}
    return torch.load(DIRECTION_BANK_PT, map_location="cpu", weights_only=False)


def direction_entries_for_layer(bank: dict[str, Any], layer: int) -> list[dict[str, Any]]:
    by_layer = bank.get("directions_by_layer") or {}
    items = by_layer.get(str(int(layer))) or by_layer.get(int(layer)) or []
    return [item for item in items if isinstance(item.get("tensor"), torch.Tensor) and int(item["tensor"].numel()) == HIDDEN_DIM]


def make_diverse_branch_deltas(
    *,
    layer: int,
    alpha: float,
    k: int,
    seed: int,
    direction_bank: dict[str, Any],
) -> list[dict[str, Any]]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    a = float(alpha)
    entries: list[dict[str, Any]] = []
    basis: list[torch.Tensor] = []

    clean = torch.zeros(HIDDEN_DIM, dtype=torch.float32)
    entries.append({"branch_id": 0, "delta": clean, "delta_family": "clean", "delta_type": "clean_zero", "direction_name": "clean_zero"})

    base = rms_normalize(torch.randn(HIDDEN_DIM, generator=gen, dtype=torch.float32))
    basis.append(base)
    if len(entries) < k:
        entries.append({"branch_id": len(entries), "delta": base * a, "delta_family": "random", "delta_type": "random_plus", "direction_name": "random_base"})
    if len(entries) < k:
        entries.append({"branch_id": len(entries), "delta": -base * a, "delta_family": "random", "delta_type": "random_minus", "direction_name": "random_base"})
    if len(entries) < k:
        second = orthogonalize(torch.randn(HIDDEN_DIM, generator=gen, dtype=torch.float32), basis)
        basis.append(second)
        entries.append({"branch_id": len(entries), "delta": second * a, "delta_family": "random", "delta_type": "random_orthogonal", "direction_name": "random_second"})

    preferred_families = [
        "hidden_origin_empirical",
        "hidden_origin_whitened",
        "v1_hidden_origin_tap",
        "old_tap_aligned",
        "adapter_proxy",
    ]
    bank_entries = direction_entries_for_layer(direction_bank, int(layer))
    bank_entries = sorted(bank_entries, key=lambda item: (preferred_families.index(item.get("family")) if item.get("family") in preferred_families else 99, str(item.get("name"))))
    for item in bank_entries:
        if len(entries) >= k:
            break
        direction = orthogonalize(item["tensor"], basis)
        basis.append(direction)
        entries.append(
            {
                "branch_id": len(entries),
                "delta": direction * a,
                "delta_family": str(item.get("family") or "bank"),
                "delta_type": str(item.get("name") or item.get("family") or "bank_direction"),
                "direction_name": str(item.get("name") or item.get("family") or "bank_direction"),
            }
        )

    empirical = next((item for item in bank_entries if str(item.get("family")).startswith("hidden_origin")), None)
    if empirical is not None and len(entries) < k:
        raw = empirical["tensor"].flatten().to(torch.float32) + 0.20 * torch.randn(HIDDEN_DIM, generator=gen, dtype=torch.float32)
        direction = orthogonalize(raw, basis)
        basis.append(direction)
        entries.append(
            {
                "branch_id": len(entries),
                "delta": direction * a,
                "delta_family": "branch_specific_noise",
                "delta_type": "noise_around_hidden_origin_empirical",
                "direction_name": "noise_around_hidden_origin_empirical",
            }
        )

    while len(entries) < k:
        direction = orthogonalize(torch.randn(HIDDEN_DIM, generator=gen, dtype=torch.float32), basis)
        basis.append(direction)
        entries.append({"branch_id": len(entries), "delta": direction * a, "delta_family": "random", "delta_type": "random_extra", "direction_name": "random_extra"})

    return entries[:k]


def split_task_ids_for_pairs(pairs: Sequence[dict[str, Any]], seed: int = SEED) -> tuple[dict[str, set[str]], dict[str, Any]]:
    by_task_pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_task_pairs[str(pair["task_id"])].append(pair)
    task_ids = sorted(by_task_pairs)
    if not task_ids:
        return {"train": set(), "val": set(), "test": set()}, {"blocked": True, "reason": "no tasks"}
    if len(task_ids) < 3:
        train = set(task_ids[:1])
        val = set(task_ids[1:2])
        test = set(task_ids[2:])
        return {"train": train, "val": val, "test": test}, {"weak_split": True, "task_count": len(task_ids)}

    n = len(task_ids)
    test_n = min(max(4, round(0.20 * n)), max(1, n - 2))
    val_n = min(max(1, round(0.20 * n)), max(1, n - test_n - 1))
    train_n = max(1, n - val_n - test_n)
    rng = random.Random(seed)
    best: tuple[float, list[str]] | None = None
    for _ in range(8000):
        shuffled = task_ids[:]
        rng.shuffle(shuffled)
        train = set(shuffled[:train_n])
        val = set(shuffled[train_n : train_n + val_n])
        test = set(shuffled[train_n + val_n :])
        counts = {
            "train_pairs": sum(len(by_task_pairs[t]) for t in train),
            "val_pairs": sum(len(by_task_pairs[t]) for t in val),
            "test_pairs": sum(len(by_task_pairs[t]) for t in test),
            "test_behavior_groups": len({p["branch_group_id"] for t in test for p in by_task_pairs[t]}),
            "test_domains": len({str(p.get("domain")) for t in test for p in by_task_pairs[t]}),
        }
        score = (
            min(counts["train_pairs"], counts["test_pairs"]) * 10000
            + counts["test_pairs"] * 200
            + counts["test_behavior_groups"] * 50
            + counts["val_pairs"] * 10
            + counts["test_domains"]
        )
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
        "pair_counts_by_task": {task: len(vals) for task, vals in by_task_pairs.items()},
        "split_task_ids": {name: sorted(vals) for name, vals in split.items()},
    }
    return split, meta


def v2_commands_run() -> list[str]:
    scripts = [
        "bg_hidden_origin_diversity_v2_audit.py",
        "bg_hidden_origin_task_screening_v2.py",
        "build_bg_hidden_origin_direction_bank_v2.py",
        "generate_bg_hidden_origin_diverse_branches_v2.py",
        "build_bg_hidden_origin_tap_dataset_v2.py",
        "train_bg_hidden_origin_taps_v2.py",
        "evaluate_bg_hidden_origin_taps_v2.py",
        "analyze_bg_hidden_origin_v2_layers_generation.py",
        "analyze_bg_hidden_origin_tap_geometry_v2.py",
        "analyze_bg_hidden_origin_diversity_v2_experiment.py",
    ]
    return [f"venv/bin/python -m py_compile utilities/tests/manual/{script}" for script in scripts] + [
        f"venv/bin/python -u utilities/tests/manual/{script}" for script in scripts
    ]

