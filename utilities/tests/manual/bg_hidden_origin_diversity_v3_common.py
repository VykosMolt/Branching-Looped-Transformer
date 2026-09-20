"""Shared helpers for hidden-origin branch diversity v3 probes.

The v3 manual probes are intentionally constrained to frozen local Ouro
inference, branch outcome generation, tiny tap heads, and reports.  They do not
train Ouro, mutate tokenizer/checkpoint/model files, update tap registries, use
wrapper/local-agent code paths, or collapse reward ties into labels.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Sequence

import torch

from bg_hidden_origin_diversity_v2_common import (
    DIAGNOSTIC_ALPHA,
    HIDDEN_DIM,
    PRIMARY_ALPHA_CAP,
    PROBE_ROOT,
    PROJECT_ROOT,
    SEED,
    V1_ROOT,
    V2_ROOT,
    alpha_bucket,
    compact_branch_row,
    config_dim,
    config_vector_from_row,
    cosine,
    deterministic_correct,
    deterministic_reward,
    diagnostic_alpha_row,
    ensure_v2_root,
    evaluate_mcq,
    finite_float,
    generate_with_hook_v2,
    group_is_behaviorally_diverse_v2,
    group_is_reward_diverse_v2,
    group_rows,
    load_all_v2_branch_rows,
    load_json,
    load_more_candidate_tasks,
    md_table,
    normalize_task,
    rate,
    rel,
    rms_normalize,
    row_reward,
    safe_primary_row,
    sampled_reward,
    stable_v2_row,
    tensor_stats,
    write_csv,
    write_json,
    write_md,
)
from bg_hidden_origin_tap_common import (
    CONFIGS,
    HEAD_CLASSES,
    append_doc_section,
    available_configs_for_rows,
    branch_key,
    direction_from_state_dict,
    score_matrix,
)
from src.evaluator.bg_hidden_branching import delta_rms


V3_ROOT = PROBE_ROOT / "bg_hidden_origin_diversity_v3_2026-05-18"
TASK_SELECTION_JSON = V3_ROOT / "task_selection_v3.json"
SPLIT_GUARD_JSON = V3_ROOT / "split_guard_v3.json"
DIRECTION_BANK_V3_PT = V3_ROOT / "direction_bank_v3.pt"
DIVERSITY_ABLATION_PT = V3_ROOT / "diversity_ablation_v3.pt"
DIVERSITY_ABLATION_PARTIAL_PT = V3_ROOT / "diversity_ablation_v3.partial.pt"
DATASET_V3_PT = V3_ROOT / "hidden_origin_tap_dataset_v3.pt"
HEADS_V3_PT = V3_ROOT / "hidden_origin_tap_heads_v3.pt"
MAX_NEW_TOKENS = 128
PRIMARY_DOMAINS = {"reasoning", "science"}
PRIMARY_BRANCH_POINTS = {"L24", "L36", "L24_L1", "L36_L1"}
DIAGNOSTIC_BRANCH_POINTS = {"L47", "L47_L1"}
SELECTOR_MINIMUMS = {
    "new_v3_behaviorally_diverse_groups": 20,
    "combined_behaviorally_diverse_groups": 60,
    "heldout_task_ids": 6,
    "heldout_behaviorally_diverse_groups": 15,
    "heldout_non_tie_pairs": 80,
}
SELECTOR_PREFERRED = {
    "new_v3_behaviorally_diverse_groups": 40,
    "combined_behaviorally_diverse_groups": 80,
    "heldout_task_ids": 8,
    "heldout_behaviorally_diverse_groups": 20,
    "heldout_non_tie_pairs": 120,
}


def ensure_v3_root() -> None:
    ensure_v2_root()
    V3_ROOT.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=_json_default_compact) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _json_default_compact(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return tensor_stats(value)
    if isinstance(value, Path):
        return rel(value)
    try:
        return float(value)
    except Exception:
        return str(value)


def write_json_atomic(path: Path, payload: Any) -> None:
    write_json(path, payload)


def primary_domain_row(row: dict[str, Any]) -> bool:
    return str(row.get("domain") or "").lower() in PRIMARY_DOMAINS


def same_prefix_hidden_origin_row(row: dict[str, Any]) -> bool:
    method = str(row.get("branch_method") or row.get("method") or "")
    if not method:
        return True
    return method in {"hook_intervention_per_branch", "hook_hidden_origin_branch"}


def primary_safe_v3_row(row: dict[str, Any]) -> bool:
    return (
        primary_domain_row(row)
        and same_prefix_hidden_origin_row(row)
        and normalize_branch_point(row) in {"L24", "L36"}
        and safe_primary_row(row)
        and stable_v2_row(row)
    )


def primary_safe_deterministic_v3_row(row: dict[str, Any]) -> bool:
    return primary_safe_v3_row(row) and str(row.get("label_source") or "deterministic") in {"deterministic", ""}


def diagnostic_alpha_v3_row(row: dict[str, Any]) -> bool:
    return (
        primary_domain_row(row)
        and same_prefix_hidden_origin_row(row)
        and diagnostic_alpha_row(row)
        and stable_v2_row(row)
    )


def normalized_branch_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if "deterministic_reward" not in out:
        out["deterministic_reward"] = finite_float(out.get("reward"), 0.0)
    if "reward" not in out:
        out["reward"] = out["deterministic_reward"]
    if "deterministic_correct" not in out:
        out["deterministic_correct"] = bool(out.get("correct"))
    if "correct" not in out:
        out["correct"] = bool(out["deterministic_correct"])
    if "alpha_bucket" not in out:
        out["alpha_bucket"] = alpha_bucket(out)
    if "label_source" not in out:
        out["label_source"] = "deterministic"
    if "safety_envelope" not in out:
        out["safety_envelope"] = finite_float(out.get("alpha"), 999.0) <= PRIMARY_ALPHA_CAP
    if not out.get("branch_point") and out.get("target_layer"):
        out["branch_point"] = f"L{int(out['target_layer'])}"
    return out


def load_v3_branch_rows() -> list[dict[str, Any]]:
    path = DIVERSITY_ABLATION_PT if DIVERSITY_ABLATION_PT.exists() else DIVERSITY_ABLATION_PARTIAL_PT
    if not path.exists():
        return []
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return []
    return [
        normalized_branch_row(row)
        for row in list(payload.get("rows") or payload.get("records") or [])
        if row.get("branch_group_id") is not None
    ]


def load_all_v3_branch_rows(include_prior: bool = True) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    rows = []
    if include_prior:
        rows.extend(load_all_v2_branch_rows(include_prior=True))
    rows.extend(load_v3_branch_rows())
    for row in rows:
        if row.get("branch_group_id") is None:
            continue
        norm = normalized_branch_row(row)
        by_key[branch_key(norm)] = norm
    return list(by_key.values())


def compact_branch_row_v3(row: dict[str, Any]) -> dict[str, Any]:
    out = compact_branch_row(row)
    for key in ("v1_tap_score", "v2_tap_score", "old_frozen_tap_score"):
        if key in row:
            out[key] = row.get(key)
    pooled = row.get("pooled_vectors") or {}
    out["available_configs"] = [
        cfg
        for cfg in CONFIGS
        if isinstance(config_vector_from_row(row, cfg), torch.Tensor)
        and int(config_vector_from_row(row, cfg).numel()) == config_dim(cfg)  # type: ignore[union-attr]
    ]
    out["pooled_vector_keys"] = sorted(pooled.keys())
    return out


def candidate_pair_stats(groups: dict[str, list[dict[str, Any]]], label_source: str = "deterministic") -> dict[str, Any]:
    candidate_pairs = 0
    tie_pairs = 0
    non_tie_pairs = 0
    for vals in groups.values():
        ordered = list(vals)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                if label_source == "sampled_expected" and (sampled_reward(ordered[i]) is None or sampled_reward(ordered[j]) is None):
                    continue
                candidate_pairs += 1
                if row_reward(ordered[i], label_source) == row_reward(ordered[j], label_source):
                    tie_pairs += 1
                else:
                    non_tie_pairs += 1
    return {
        "candidate_pairs": candidate_pairs,
        "tie_pairs": tie_pairs,
        "non_tie_pairs": non_tie_pairs,
        "tie_rate": tie_pairs / max(candidate_pairs, 1),
    }


def branch_group_metrics(rows: Sequence[dict[str, Any]], label_source: str = "deterministic") -> dict[str, Any]:
    stable = [row for row in rows if stable_v2_row(row)]
    groups = {gid: vals for gid, vals in group_rows(stable).items() if len(vals) >= 2}
    diverse = [gid for gid, vals in groups.items() if group_is_behaviorally_diverse_v2(vals, label_source)]
    reward_diverse = [gid for gid, vals in groups.items() if group_is_reward_diverse_v2(vals, label_source)]
    pair_stats = candidate_pair_stats(groups, label_source)
    return {
        "rows": len(rows),
        "stable_rows": len(stable),
        "stable_rate": len(stable) / max(len(rows), 1),
        "parse_rate": sum(1 for row in stable if bool(row.get("parse_success"))) / max(len(stable), 1),
        "groups": len(groups),
        "behaviorally_diverse_groups": len(diverse),
        "reward_diverse_groups": len(reward_diverse),
        "behaviorally_diverse_groups_per_100_rows": 100.0 * len(diverse) / max(len(rows), 1),
        "non_tie_pairs_per_100_rows": 100.0 * pair_stats["non_tie_pairs"] / max(len(rows), 1),
        "candidate_pair_stats": pair_stats,
        "task_ids_with_reward_diversity": sorted({str(vals[0].get("task_id")) for vals in groups.values() if group_is_reward_diverse_v2(vals, label_source)}),
    }


def condition_group_stats(
    rows: Sequence[dict[str, Any]],
    field: str,
    *,
    label_source: str = "deterministic",
    group_field: str | None = None,
) -> dict[str, dict[str, Any]]:
    groups = [vals for vals in group_rows(rows).values() if len(vals) >= 2]
    out: dict[str, dict[str, Any]] = {}
    for vals in groups:
        first = vals[0]
        value = str(first.get(group_field or field) if field != "alpha_bucket" else alpha_bucket(first))
        if field == "K":
            value = str(first.get("K") or len(vals))
        if field == "delta_family":
            value = str(first.get("primary_delta_family") or first.get("delta_family") or "unknown")
        if field == "branch_point":
            value = normalize_branch_point(first)
        bucket = out.setdefault(
            value,
            {
                "rows": 0,
                "groups": 0,
                "behaviorally_diverse_groups": 0,
                "reward_diverse_groups": 0,
                "candidate_pairs": 0,
                "tie_pairs": 0,
                "non_tie_pairs": 0,
                "parse_success_rows": 0,
                "stable_rows": 0,
                "reward_values": [],
            },
        )
        bucket["rows"] += len(vals)
        bucket["groups"] += 1
        bucket["stable_rows"] += sum(1 for row in vals if stable_v2_row(row))
        bucket["parse_success_rows"] += sum(1 for row in vals if bool(row.get("parse_success")))
        if group_is_behaviorally_diverse_v2(vals, label_source):
            bucket["behaviorally_diverse_groups"] += 1
        if group_is_reward_diverse_v2(vals, label_source):
            bucket["reward_diverse_groups"] += 1
        stats = candidate_pair_stats({"group": vals}, label_source)
        bucket["candidate_pairs"] += stats["candidate_pairs"]
        bucket["tie_pairs"] += stats["tie_pairs"]
        bucket["non_tie_pairs"] += stats["non_tie_pairs"]
        bucket["reward_values"].extend(row_reward(row, label_source) for row in vals)
    for val in out.values():
        val["tie_rate"] = val["tie_pairs"] / max(val["candidate_pairs"], 1)
        val["stable_rate"] = val["stable_rows"] / max(val["rows"], 1)
        val["parse_rate"] = val["parse_success_rows"] / max(val["rows"], 1)
        val["behaviorally_diverse_rate"] = val["behaviorally_diverse_groups"] / max(val["groups"], 1)
        val["reward_diverse_rate"] = val["reward_diverse_groups"] / max(val["groups"], 1)
        val["behaviorally_diverse_groups_per_100_rows"] = 100.0 * val["behaviorally_diverse_groups"] / max(val["rows"], 1)
        val["non_tie_pairs_per_100_rows"] = 100.0 * val["non_tie_pairs"] / max(val["rows"], 1)
        rewards = [float(x) for x in val.pop("reward_values") if math.isfinite(float(x))]
        val["reward_mean"] = float(mean(rewards)) if rewards else float("nan")
    return out


def normalize_branch_point(row: dict[str, Any] | str | int) -> str:
    if isinstance(row, dict):
        value = str(row.get("branch_point") or row.get("target_layer") or "")
    else:
        value = str(row)
    if value in {"24", "L24_L1"}:
        return "L24"
    if value in {"36", "L36_L1"}:
        return "L36"
    if value in {"47", "L47_L1"}:
        return "L47"
    if value.startswith("L24"):
        return "L24"
    if value.startswith("L36"):
        return "L36"
    if value.startswith("L47"):
        return "L47"
    return value or "unknown"


def load_task_selection() -> dict[str, Any]:
    return load_json(TASK_SELECTION_JSON, {}) or {}


def selected_v3_tasks() -> list[dict[str, Any]]:
    payload = load_task_selection()
    rows = list(payload.get("selected_tasks") or payload.get("selected_rows") or [])
    tasks = []
    for idx, row in enumerate(rows):
        task = normalize_task(row, row.get("domain"), idx)
        if not task:
            continue
        task.update({k: v for k, v in row.items() if k not in task})
        if "prompt" not in task or not task["prompt"]:
            task["prompt"] = row.get("prompt")
        tasks.append(task)
    return tasks


def load_split_guard() -> dict[str, Any]:
    return load_json(SPLIT_GUARD_JSON, {}) or {}


def split_role_for_task(task_id: str, split_payload: dict[str, Any] | None = None) -> str:
    split = split_payload or load_split_guard()
    tid = str(task_id)
    for key, role in (
        ("v3_train_candidate_task_ids", "train_candidate"),
        ("v3_val_candidate_task_ids", "val_candidate"),
        ("v3_heldout_candidate_task_ids", "heldout_candidate"),
    ):
        if tid in {str(x) for x in split.get(key, [])}:
            return role
    return "unassigned"


def config_layer(config: str) -> int | None:
    if config.startswith("24_"):
        return 24
    if config.startswith("30_"):
        return 30
    if config.startswith("36_"):
        return 36
    if config.startswith("42_"):
        return 42
    if config.startswith("47_"):
        return 47
    return None


def load_head_rows(path: Path, *, variant: str | None = None, only_passing: bool = True) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return []
    rows = []
    for row in list(payload.get("heads") or []):
        if variant is not None and row.get("variant") != variant:
            continue
        if only_passing and not row.get("flip_diagnostics", {}).get("passes"):
            continue
        rows.append(row)
    return rows


def compact_head_id(row: dict[str, Any]) -> str:
    metrics = row.get("metrics", {})
    variant = row.get("variant", "primary")
    return f"{variant}::{row.get('config')}::{row.get('architecture')}::seed={metrics.get('seed')}::lr={metrics.get('lr')}"


def best_head_by_validation(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [
        row
        for row in rows
        if row.get("flip_diagnostics", {}).get("passes")
        and math.isfinite(float(row.get("metrics", {}).get("validation_pairwise_accuracy", float("nan"))))
    ]
    if not valid:
        return None
    return max(
        valid,
        key=lambda row: (
            float(row.get("metrics", {}).get("validation_pairwise_accuracy", -1.0)),
            float(row.get("metrics", {}).get("train_pairwise_accuracy", -1.0)),
        ),
    )


def load_best_v1_head() -> dict[str, Any] | None:
    return best_head_by_validation(load_head_rows(V1_ROOT / "hidden_origin_tap_heads.pt"))


def load_best_v2_head() -> dict[str, Any] | None:
    return best_head_by_validation(load_head_rows(V2_ROOT / "hidden_origin_tap_heads_v2.pt", variant="primary_safe_deterministic"))


def load_best_v3_head() -> dict[str, Any] | None:
    return best_head_by_validation(load_head_rows(HEADS_V3_PT, variant="primary_safe_deterministic"))


def score_group_with_head(
    group: Sequence[dict[str, Any]],
    head_row: dict[str, Any] | None,
    *,
    score_key: str,
    device: torch.device | None = None,
) -> None:
    if head_row is None:
        return
    config = str(head_row.get("config") or "")
    vectors = [config_vector_from_row(row, config) for row in group]
    if not vectors or not all(isinstance(vec, torch.Tensor) and int(vec.numel()) == config_dim(config) for vec in vectors):
        return
    dev = device or torch.device("cpu")
    head = HEAD_CLASSES[str(head_row["architecture"])](config_dim(config))
    head.load_state_dict(head_row["state_dict"])
    head.to(device=dev, dtype=torch.float32)
    head.eval()
    mat = score_matrix(head, [vec for vec in vectors if isinstance(vec, torch.Tensor)], dev)
    margins = mat.sum(dim=1).detach().cpu().tolist()
    wins = ((mat > 0) & ~torch.eye(mat.shape[0], dtype=torch.bool)).sum(dim=1).detach().cpu().tolist()
    for row, margin, win_count in zip(group, margins, wins):
        row[score_key] = float(margin)
        row[f"{score_key}_wins"] = int(win_count)
        row[f"{score_key}_head"] = compact_head_id(head_row)
    head.to("cpu")


def v3_commands_run() -> list[str]:
    scripts = [
        "bg_hidden_origin_diversity_v3_audit.py",
        "bg_hidden_origin_task_selection_v3.py",
        "bg_hidden_origin_v3_split_guard.py",
        "build_bg_hidden_origin_direction_bank_v3.py",
        "generate_bg_hidden_origin_diversity_ablation_v3.py",
        "analyze_bg_hidden_origin_diversity_drivers_v3.py",
        "build_bg_hidden_origin_tap_dataset_v3.py",
        "train_bg_hidden_origin_taps_v3.py",
        "evaluate_bg_hidden_origin_taps_v3.py",
        "analyze_bg_hidden_origin_tap_geometry_v3.py",
        "analyze_bg_hidden_origin_diversity_v3_experiment.py",
    ]
    return [f"venv/bin/python -m py_compile utilities/tests/manual/{script}" for script in scripts] + [
        f"venv/bin/python -u utilities/tests/manual/{script}" for script in scripts
    ]


def verdict_for_data_targets(
    *,
    new_v3_behaviorally_diverse_groups: int,
    combined_behaviorally_diverse_groups: int,
    heldout_task_ids: int,
    heldout_behaviorally_diverse_groups: int,
    heldout_non_tie_pairs: int,
    stable_rate: float,
    tie_rate: float,
    errors: int = 0,
    rows: int = 0,
) -> str:
    if rows == 0 and errors:
        return "BLOCKED"
    if rows and stable_rate < 0.50:
        return "UNSTABLE"
    preferred = (
        new_v3_behaviorally_diverse_groups >= SELECTOR_PREFERRED["new_v3_behaviorally_diverse_groups"]
        and combined_behaviorally_diverse_groups >= SELECTOR_PREFERRED["combined_behaviorally_diverse_groups"]
        and heldout_task_ids >= SELECTOR_PREFERRED["heldout_task_ids"]
        and heldout_behaviorally_diverse_groups >= SELECTOR_PREFERRED["heldout_behaviorally_diverse_groups"]
        and heldout_non_tie_pairs >= SELECTOR_PREFERRED["heldout_non_tie_pairs"]
    )
    minimum = (
        new_v3_behaviorally_diverse_groups >= SELECTOR_MINIMUMS["new_v3_behaviorally_diverse_groups"]
        and combined_behaviorally_diverse_groups >= SELECTOR_MINIMUMS["combined_behaviorally_diverse_groups"]
        and heldout_task_ids >= SELECTOR_MINIMUMS["heldout_task_ids"]
        and heldout_behaviorally_diverse_groups >= SELECTOR_MINIMUMS["heldout_behaviorally_diverse_groups"]
        and heldout_non_tie_pairs >= SELECTOR_MINIMUMS["heldout_non_tie_pairs"]
    )
    if preferred or minimum:
        return "DIVERSITY_TARGET_MET"
    if rows and (new_v3_behaviorally_diverse_groups > 0 or tie_rate < 0.932):
        return "DIVERSITY_IMPROVED"
    if rows:
        return "LOW_DIVERSITY"
    return "BLOCKED"


def markdown_key_values(title: str, payload: dict[str, Any]) -> list[str]:
    lines = [f"## {title}", ""]
    for key, value in payload.items():
        lines.append(f"- {key}: `{value}`")
    return lines


def safe_mean(values: Iterable[Any]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(mean(vals)) if vals else float("nan")
