"""Shared utilities for DualAnchor convergence hairs + RS hard-slice v1.

This probe is replay-first. It reads the existing DualAnchor v3 candidate tree
artifact, uses the cached L30/L42 pooled hidden states as convergence-hair
signals, and evaluates representative-merge safety counterfactually.

No steering, training, wrapper/local-agent execution, registry update, or true
autoregressive fork/carry claim is made here.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import torch

from bg_hidden_origin_tap_common import PROBE_ROOT
from bg_merged_tap_v1_common import safe_float, score_diff
from run_bg_layer_native_two_tap_readiness_v1 import ANCHOR_GROUPS, tensor_weight

import run_bg_dualanchor_all_loop_audit_v1 as base
import run_bg_dualanchor_all_loop_guarded_policy_v1 as guard


SHORT_NAME = "dualanchor_convergence_hairs_reasoning_science_v1"
OUT_ROOT = PROBE_ROOT / "bg_dualanchor_convergence_hairs_reasoning_science_v1_2026-05-31"
V3_ROOT = PROBE_ROOT / "bg_dualanchor_architecture_looped_stratified_probe_v3_2026-05-31"
V2_ROOT = PROBE_ROOT / "bg_dualanchor_architecture_looped_stratified_probe_v2_2026-05-31"
LINEAGE_ROOT = PROBE_ROOT / "bg_dualanchor_architecture_looped_lineage_probe_v1_2026-05-31"
TRUE_CARRY_ROOT = PROBE_ROOT / "bg_dualanchor_true_carry_equivalence_v1_2026-05-31"
PERTURB_LIFT_ROOT = PROBE_ROOT / "bg_dualanchor_perturbation_lift_v1_2026-05-31"
FIXED_ROOT = PROBE_ROOT / "bg_fixed_composite_branch_survival_policy_v1_2026-05-18"
GATED_ROOT = PROBE_ROOT / "bg_gated_branch_content_selector_v1_2026-05-18"
SELECTION_ROOT = PROBE_ROOT / "bg_selection_only_phase2_prototype_v1_2026-05-18"
HIDDEN_GENERATOR_ROOT = PROBE_ROOT / "bg_hidden_origin_branch_generator_v1_2026-05-18"
HIDDEN_QUOTA_ROOT = PROBE_ROOT / "bg_hidden_origin_quota_v4_2026-05-18"
HIDDEN_TAPS_ROOT = PROBE_ROOT / "bg_hidden_origin_taps_2026-05-18"
HIDDEN_STATE_ROOT = PROBE_ROOT / "bg_hidden_state_branch_generation_2026-05-18"

V3_PT = V3_ROOT / "architecture_looped_rows.pt"
V3_TASK_ROWS = V3_ROOT / "task_rows.csv"
V3_STAGE_ROWS = V3_ROOT / "stage_decisions.csv"
V3_TERMINAL_ROWS = V3_ROOT / "terminal_policy_rows.csv"
V3_CONFIDENCE_ROWS = V3_ROOT / "terminal_confidence_rows.csv"
V3_HARD_ROWS = V3_ROOT / "hard_slices_rows.csv"

DATASET_PT = OUT_ROOT / "convergence_hair_dataset.pt"
DATASET_JSON = OUT_ROOT / "convergence_hair_dataset.json"
DATASET_MD = OUT_ROOT / "convergence_hair_dataset.md"
HAIR_CANDIDATES_CSV = OUT_ROOT / "convergence_hair_candidate_rows.csv"
HAIR_PAIRS_CSV = OUT_ROOT / "convergence_hair_pair_rows.csv"
POLICIES_JSON = OUT_ROOT / "convergence_hair_policies.json"
REPLAY_JSON = OUT_ROOT / "convergence_hair_replay_eval.json"
REPLAY_ROWS_CSV = OUT_ROOT / "convergence_hair_replay_rows.csv"
TIE_JSON = OUT_ROOT / "tie_decomposition_diagnostic.json"
TERMINAL_JSON = OUT_ROOT / "terminal_defer_policy.json"

PRIMARY_DOMAINS = {"reasoning", "science"}
DUALANCHOR_POLICY = "dualanchor_adaptive_branch_anchor_light_AntisymLinear"
ANCHOR_A = "MIX_CODE_REASONING"
ANCHOR_B = "MIX_OBJECTIVE_ALL"
HAIR_SPECS = [
    {"hair_stage": f"L{loop}_30", "loop": loop, "hair_layer": 30, "previous_stage": f"L{loop}_24", "next_stage": f"L{loop}_36", "next_layer": 36}
    for loop in (1, 2, 3, 4)
] + [
    {"hair_stage": f"L{loop}_42", "loop": loop, "hair_layer": 42, "previous_stage": f"L{loop}_36", "next_stage": f"L{loop}_47", "next_layer": 47}
    for loop in (1, 2, 3)
] + [
    {"hair_stage": "L4_42", "loop": 4, "hair_layer": 42, "previous_stage": "L4_36", "next_stage": "terminal_L4_47", "next_layer": 47}
]


def ensure_root() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_root()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def write_md(path: Path, lines: Sequence[str]) -> None:
    ensure_root()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    ensure_root()
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in keys})


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() <= 16:
            return value.detach().cpu().tolist()
        return {"tensor_shape": list(value.shape), "dtype": str(value.dtype)}
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return str(value)


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, sort_keys=True, default=json_default)
    if isinstance(value, torch.Tensor):
        return json.dumps(json_default(value), sort_keys=True)
    return value


def md_table(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> list[str]:
    if not rows:
        return ["_No rows._"]
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join("---" for _ in keys) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |")
    return lines


def finite_mean(values: Iterable[Any]) -> float:
    xs: list[float] = []
    for value in values:
        x = safe_float(value, float("nan"))
        if math.isfinite(x):
            xs.append(x)
    return float(mean(xs)) if xs else float("nan")


def truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def parse_literal(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except Exception:
            return default
    return value if value is not None else default


def row_reward(row: dict[str, Any]) -> float:
    for key in ("deterministic_reward", "reward", "final_reward", "correctness", "oracle_reward"):
        x = safe_float(row.get(key), float("nan"))
        if math.isfinite(x):
            return x
    if truthy(row.get("deterministic_correct")) or truthy(row.get("correct")):
        return 1.0
    return 0.0


def stable_risk(row: dict[str, Any]) -> float:
    risk = 0.0
    if not truthy(row.get("parse_success", True)):
        risk += 1.0
    if truthy(row.get("empty_output")):
        risk += 1.0
    if truthy(row.get("nan_inf")):
        risk += 1.0
    if safe_float(row.get("repetition_rate"), 0.0) > 0.35:
        risk += 0.5
    return risk


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def load_v3_pt() -> dict[str, Any]:
    if not V3_PT.exists():
        return {}
    return torch.load(V3_PT, map_location="cpu", weights_only=False)


def load_v3_rows() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    payload = load_v3_pt()
    return (
        payload,
        list(payload.get("rows") or []),
        list(payload.get("stage_rows") or []),
        list(payload.get("task_rows") or []),
        list(payload.get("terminal_rows") or []),
    )


def load_dualanchor_taps() -> dict[str, dict[str, dict[str, Any]]]:
    policies, _policy_rows = base.load_constrained_policies()
    return policies.get(DUALANCHOR_POLICY, {})


def artifact_status() -> list[dict[str, Any]]:
    targets = [
        ("v3_root", V3_ROOT, ["summary.json", "architecture_looped_rows.pt", "task_rows.csv", "stage_decisions.csv", "hard_slices.json", "threshold_budget.json", "l47_ablation.json", "phase2a_readiness.json"]),
        ("v2_root", V2_ROOT, ["summary.json"]),
        ("lineage_root", LINEAGE_ROOT, ["summary.json"]),
        ("true_carry_root", TRUE_CARRY_ROOT, ["summary.json", "true_carry_equivalence.json"]),
        ("perturb_lift_root", PERTURB_LIFT_ROOT, ["summary.json"]),
        ("fixed_composite_root", FIXED_ROOT, ["summary.json"]),
        ("gated_root", GATED_ROOT, ["summary.json"]),
        ("selection_only_root", SELECTION_ROOT, ["summary.json"]),
        ("hidden_generator_root", HIDDEN_GENERATOR_ROOT, ["summary.json"]),
        ("hidden_quota_root", HIDDEN_QUOTA_ROOT, ["summary.json"]),
        ("hidden_taps_root", HIDDEN_TAPS_ROOT, ["summary.json"]),
        ("hidden_state_root", HIDDEN_STATE_ROOT, ["summary.json"]),
    ]
    rows = []
    for name, root, expected in targets:
        missing = [item for item in expected if not (root / item).exists()]
        rows.append({"artifact": name, "path": str(root), "exists": root.exists(), "missing_expected": missing, "status": "READY" if root.exists() and not missing else ("PARTIAL" if root.exists() else "MISSING")})
    return rows


def hair_vector(row: dict[str, Any], hair_layer: int, loop: int, kind: str = "pooled") -> torch.Tensor | None:
    source = row.get("pooled_vectors") if kind == "pooled" else row.get("last_token_vectors")
    if not isinstance(source, dict):
        return None
    for key in (f"L{hair_layer}_L{loop}", f"{hair_layer}_L{loop}"):
        value = source.get(key)
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().to(torch.float32).flatten()
    return None


def tensor_cosine_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    denom = float(torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right))
    if denom <= 1e-12:
        return 0.0 if float(torch.linalg.vector_norm(left - right)) <= 1e-12 else 1.0
    cos = float(torch.dot(left, right) / denom)
    return float(max(0.0, min(2.0, 1.0 - cos)))


def tensor_rms(value: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(value.to(torch.float32) ** 2)).item())


def hidden_distances(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    diff = left - right
    rms = tensor_rms(diff)
    denom = (tensor_rms(left) + tensor_rms(right)) / 2.0
    return {
        "hidden_cosine_distance": tensor_cosine_distance(left, right),
        "hidden_rms_distance": rms,
        "hidden_rms_normalized": rms / max(denom, 1e-12),
    }


def descendant_of(branch_id: str, ancestor_id: str) -> bool:
    return branch_id == ancestor_id or branch_id.startswith(ancestor_id + "/")


def child_birth_stage(row: dict[str, Any]) -> str:
    return str(row.get("birth_stage") or "")


def task_final_ids(task_row: dict[str, Any]) -> list[str]:
    ids = parse_literal(task_row.get("final_branch_ids"), [])
    return [str(item) for item in ids or []]


def build_indexes(rows: Sequence[dict[str, Any]], task_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows_by_id = {str(row.get("branch_id")): row for row in rows if row.get("branch_id")}
    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    children_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    parents_by_birth: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        task_id = str(row.get("task_id"))
        rows_by_task[task_id].append(row)
        parent = row.get("parent_branch_id")
        if parent:
            parent_id = str(parent)
            children_by_parent[parent_id].append(row)
            parents_by_birth[(task_id, child_birth_stage(row))].add(parent_id)
    task_by_id = {str(row.get("task_id")): row for row in task_rows}
    final_ids_by_task = {task_id: task_final_ids(row) for task_id, row in task_by_id.items()}
    return {
        "rows_by_id": rows_by_id,
        "rows_by_task": rows_by_task,
        "children_by_parent": children_by_parent,
        "parents_by_birth": parents_by_birth,
        "task_by_id": task_by_id,
        "final_ids_by_task": final_ids_by_task,
    }


def available_ids_for_hair(task_id: str, spec: dict[str, Any], indexes: dict[str, Any]) -> list[str]:
    if spec["hair_stage"] == "L4_42":
        return list(indexes["final_ids_by_task"].get(task_id, []))
    return sorted(indexes["parents_by_birth"].get((task_id, spec["next_stage"]), set()))


def next_stage_score_details(candidates: Sequence[dict[str, Any]], spec: dict[str, Any], taps: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    layer = int(spec["next_layer"])
    loop = int(spec["loop"])
    config = f"{layer}_L{loop}"
    return guard.stage_score_details(candidates, taps, layer, config)


def pair_role_margins(details: dict[str, Any], i: int, j: int) -> dict[str, float]:
    out: dict[str, float] = {}
    for role in ANCHOR_GROUPS:
        vals = (details.get("norm_by_role") or {}).get(role)
        if vals is None or i >= len(vals) or j >= len(vals):
            out[f"{role}_margin"] = float("nan")
        else:
            out[f"{role}_margin"] = safe_float(vals[i], 0.0) - safe_float(vals[j], 0.0)
    combo = details.get("combined") or []
    out["dualanchor_avg_margin"] = safe_float(combo[i], 0.0) - safe_float(combo[j], 0.0) if i < len(combo) and j < len(combo) else float("nan")
    return out


def ranks_from_order(order: Sequence[int], n: int) -> list[int]:
    ranks = [n] * n
    for rank, idx in enumerate(order):
        if 0 <= int(idx) < n:
            ranks[int(idx)] = rank
    return ranks


def pair_key(task_id: str, hair_stage: str, left_id: str, right_id: str) -> str:
    a, b = sorted([left_id, right_id])
    return f"{task_id}::{hair_stage}::{a}::{b}"


def build_hair_dataset() -> dict[str, Any]:
    _payload, rows, stage_rows, task_rows, terminal_rows = load_v3_rows()
    indexes = build_indexes(rows, task_rows)
    taps = load_dualanchor_taps()
    candidate_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    task_ids = sorted(indexes["task_by_id"])
    for task_id in task_ids:
        task_meta = indexes["task_by_id"][task_id]
        if str(task_meta.get("domain")) not in PRIMARY_DOMAINS:
            continue
        for spec in HAIR_SPECS:
            ids = [branch_id for branch_id in available_ids_for_hair(task_id, spec, indexes) if branch_id in indexes["rows_by_id"]]
            candidates = [indexes["rows_by_id"][branch_id] for branch_id in ids]
            if not candidates:
                continue
            details = next_stage_score_details(candidates, spec, taps)
            order = [int(x) for x in details.get("order") or list(range(len(candidates)))]
            ranks = ranks_from_order(order, len(candidates))
            combined = details.get("combined") or [float("nan")] * len(candidates)
            top_by_role = details.get("top1_by_role") or {}
            for idx, row in enumerate(candidates):
                vec = hair_vector(row, int(spec["hair_layer"]), int(spec["loop"]), "pooled")
                last = hair_vector(row, int(spec["hair_layer"]), int(spec["loop"]), "last")
                candidate_rows.append(
                    {
                        "task_id": task_id,
                        "domain": row.get("domain"),
                        "source_dataset": row.get("source_dataset"),
                        "split": row.get("split"),
                        "hair_stage": spec["hair_stage"],
                        "loop": spec["loop"],
                        "hair_layer": spec["hair_layer"],
                        "previous_stage": spec["previous_stage"],
                        "next_stage": spec["next_stage"],
                        "candidate_id": row.get("branch_id"),
                        "parent_branch_id": row.get("parent_branch_id"),
                        "root_branch_id": row.get("root_branch_id"),
                        "generation_depth": row.get("generation_depth"),
                        "perturb_count": row.get("perturb_count"),
                        "birth_layer": row.get("birth_layer"),
                        "birth_loop": row.get("birth_loop"),
                        "birth_stage": row.get("birth_stage"),
                        "lineage_path": row.get("lineage_path"),
                        "hidden_available": isinstance(vec, torch.Tensor),
                        "last_token_available": isinstance(last, torch.Tensor),
                        "logits_available": False,
                        "dualanchor_combined_score": combined[idx] if idx < len(combined) else float("nan"),
                        "dualanchor_rank": ranks[idx],
                        "dualanchor_anchor_top1": [role for role, top_idx in top_by_role.items() if int(top_idx) == idx],
                        "stability_risk": stable_risk(row),
                        "final_reward_eval_only": row_reward(row),
                        "terminal_oracle_eval_only": 1.0 if row_reward(row) == safe_float(task_meta.get("oracle_reward"), float("nan")) else 0.0,
                        "parsed_answer_diagnostic_only": row.get("parsed_answer"),
                        "output_text_diagnostic_only": row.get("output_text"),
                    }
                )
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    left = candidates[i]
                    right = candidates[j]
                    lvec = hair_vector(left, int(spec["hair_layer"]), int(spec["loop"]), "pooled")
                    rvec = hair_vector(right, int(spec["hair_layer"]), int(spec["loop"]), "pooled")
                    last_l = hair_vector(left, int(spec["hair_layer"]), int(spec["loop"]), "last")
                    last_r = hair_vector(right, int(spec["hair_layer"]), int(spec["loop"]), "last")
                    hidden = hidden_distances(lvec, rvec) if isinstance(lvec, torch.Tensor) and isinstance(rvec, torch.Tensor) else {}
                    last_hidden = hidden_distances(last_l, last_r) if isinstance(last_l, torch.Tensor) and isinstance(last_r, torch.Tensor) else {}
                    margins = pair_role_margins(details, i, j)
                    reward_delta = row_reward(left) - row_reward(right)
                    pair_rows.append(
                        {
                            "pair_id": pair_key(task_id, str(spec["hair_stage"]), str(left.get("branch_id")), str(right.get("branch_id"))),
                            "task_id": task_id,
                            "domain": left.get("domain"),
                            "source_dataset": left.get("source_dataset"),
                            "split": left.get("split"),
                            "hair_stage": spec["hair_stage"],
                            "loop": spec["loop"],
                            "hair_layer": spec["hair_layer"],
                            "previous_stage": spec["previous_stage"],
                            "next_stage": spec["next_stage"],
                            "left_candidate_id": left.get("branch_id"),
                            "right_candidate_id": right.get("branch_id"),
                            "left_parent_branch_id": left.get("parent_branch_id"),
                            "right_parent_branch_id": right.get("parent_branch_id"),
                            "left_root_branch_id": left.get("root_branch_id"),
                            "right_root_branch_id": right.get("root_branch_id"),
                            "same_parent": 1.0 if left.get("parent_branch_id") and left.get("parent_branch_id") == right.get("parent_branch_id") else 0.0,
                            "same_root": 1.0 if left.get("root_branch_id") == right.get("root_branch_id") else 0.0,
                            "rank_difference": abs(ranks[i] - ranks[j]),
                            "logit_kl_available": False,
                            "topk_overlap_available": False,
                            "logit_kl": float("nan"),
                            "topk_token_overlap": float("nan"),
                            "hidden_cosine_distance": hidden.get("hidden_cosine_distance", float("nan")),
                            "hidden_rms_distance": hidden.get("hidden_rms_distance", float("nan")),
                            "hidden_rms_normalized": hidden.get("hidden_rms_normalized", float("nan")),
                            "last_token_cosine_distance": last_hidden.get("hidden_cosine_distance", float("nan")),
                            "last_token_rms_normalized": last_hidden.get("hidden_rms_normalized", float("nan")),
                            **margins,
                            "dualanchor_abs_avg_margin": abs(safe_float(margins.get("dualanchor_avg_margin"), 0.0)),
                            "anchor_abs_margin_max": max(
                                abs(safe_float(margins.get(f"{ANCHOR_A}_margin"), 0.0)),
                                abs(safe_float(margins.get(f"{ANCHOR_B}_margin"), 0.0)),
                            ),
                            "anchor_margin_product": safe_float(margins.get(f"{ANCHOR_A}_margin"), 0.0) * safe_float(margins.get(f"{ANCHOR_B}_margin"), 0.0),
                            "anchor_disagreement_pair": 1.0 if safe_float(margins.get(f"{ANCHOR_A}_margin"), 0.0) * safe_float(margins.get(f"{ANCHOR_B}_margin"), 0.0) < 0 else 0.0,
                            "left_final_reward_eval_only": row_reward(left),
                            "right_final_reward_eval_only": row_reward(right),
                            "reward_delta_eval_only": reward_delta,
                            "reward_tied_eval_only": 1.0 if abs(reward_delta) <= 1e-9 else 0.0,
                            "parsed_answer_tied_diagnostic_only": 1.0 if normalize_text(left.get("parsed_answer")) == normalize_text(right.get("parsed_answer")) else 0.0,
                            "output_text_tied_diagnostic_only": 1.0 if normalize_text(left.get("output_text")) == normalize_text(right.get("output_text")) else 0.0,
                        }
                    )
    summary = {
        "candidate_count": len(candidate_rows),
        "pair_count": len(pair_rows),
        "task_count": len({row["task_id"] for row in candidate_rows}),
        "domain_counts": dict(Counter(row.get("domain") for row in candidate_rows)),
        "hair_counts": dict(Counter(row.get("hair_stage") for row in candidate_rows)),
        "pair_hair_counts": dict(Counter(row.get("hair_stage") for row in pair_rows)),
        "l30_candidate_count": sum(1 for row in candidate_rows if int(row.get("hair_layer") or 0) == 30),
        "l42_candidate_count": sum(1 for row in candidate_rows if int(row.get("hair_layer") or 0) == 42),
        "hidden_available_rate": finite_mean(row.get("hidden_available") for row in candidate_rows),
        "logits_available": False,
        "dualanchor_margin_kind": "adjacent_next_stage_proxy",
        "classification_runtime_use": "forbidden; diagnostics only",
    }
    return {
        "summary": summary,
        "candidate_rows": candidate_rows,
        "pair_rows": pair_rows,
        "stage_rows": stage_rows,
        "task_rows": task_rows,
        "terminal_rows": terminal_rows,
    }


def quantile(values: Sequence[float], q: float) -> float:
    xs = sorted(x for x in values if math.isfinite(x))
    if not xs:
        return float("nan")
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def load_dataset() -> dict[str, Any]:
    if DATASET_PT.exists():
        return torch.load(DATASET_PT, map_location="cpu", weights_only=False)
    return build_hair_dataset()


def calibration_pairs(pair_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [row for row in pair_rows if str(row.get("split")) != "heldout"]
    return selected if selected else list(pair_rows)


def define_policies(pair_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    cal = calibration_pairs(pair_rows)
    rms_vals = [safe_float(row.get("hidden_rms_normalized"), float("nan")) for row in cal]
    quantiles = {
        "hidden_rms_q05": quantile(rms_vals, 0.05),
        "hidden_rms_q10": quantile(rms_vals, 0.10),
        "hidden_rms_q20": quantile(rms_vals, 0.20),
    }
    policies = [
        {"name": "no_hair_baseline", "family": 0, "hard_merge": False, "description": "Existing v3 behavior."},
        {"name": "soft_cluster_diagnostic", "family": 1, "hard_merge": False, "hidden_cosine_max": 0.02, "hidden_rms_norm_max": quantiles["hidden_rms_q20"], "dualanchor_abs_margin_max": 0.20, "description": "Cluster only; no merge."},
        {"name": "representative_merge", "family": 2, "hard_merge": True, "hidden_cosine_max": 0.02, "hidden_rms_norm_max": quantiles["hidden_rms_q20"], "dualanchor_abs_margin_max": 0.20, "anchor_disagreement_allowed": False, "lineage_condition": "same_root_or_any"},
        {"name": "conservative_representative_merge", "family": 3, "hard_merge": True, "hidden_cosine_max": 0.005, "hidden_rms_norm_max": quantiles["hidden_rms_q05"], "dualanchor_abs_margin_max": 0.05, "anchor_disagreement_allowed": False, "lineage_condition": "same_root_or_any"},
        {"name": "L30_only_representative_merge", "family": 4, "hard_merge": True, "hair_layers": [30], "hidden_cosine_max": 0.02, "hidden_rms_norm_max": quantiles["hidden_rms_q20"], "dualanchor_abs_margin_max": 0.20, "anchor_disagreement_allowed": False, "lineage_condition": "same_root_or_any"},
        {"name": "L42_only_representative_merge", "family": 5, "hard_merge": True, "hair_layers": [42], "hidden_cosine_max": 0.02, "hidden_rms_norm_max": quantiles["hidden_rms_q20"], "dualanchor_abs_margin_max": 0.20, "anchor_disagreement_allowed": False, "lineage_condition": "same_root_or_any"},
        {"name": "L30_L42_representative_merge", "family": 6, "hard_merge": True, "hair_layers": [30, 42], "hidden_cosine_max": 0.02, "hidden_rms_norm_max": quantiles["hidden_rms_q20"], "dualanchor_abs_margin_max": 0.20, "anchor_disagreement_allowed": False, "lineage_condition": "same_root_or_any"},
        {"name": "dualanchor_indifference_merge", "family": 7, "hard_merge": True, "dualanchor_abs_margin_max": 0.10, "anchor_abs_margin_max": 0.10, "anchor_disagreement_allowed": False, "lineage_condition": "same_root_or_any"},
        {"name": "hidden_logit_convergence_merge", "family": 8, "hard_merge": True, "hidden_cosine_max": 0.01, "hidden_rms_norm_max": quantiles["hidden_rms_q10"], "logit_required": False, "logit_kl_max": None, "dualanchor_abs_margin_max": 0.15, "anchor_disagreement_allowed": False, "lineage_condition": "same_root_or_any"},
        {"name": "same_parent_convergence_merge", "family": 9, "hard_merge": True, "hidden_cosine_max": 0.02, "hidden_rms_norm_max": quantiles["hidden_rms_q20"], "dualanchor_abs_margin_max": 0.20, "anchor_disagreement_allowed": False, "lineage_condition": "same_parent"},
        {"name": "reward_tie_merge_diagnostic_only", "family": "negative_control", "hard_merge": True, "diagnostic_only": True, "reward_tie_required": True, "description": "Not architecture: uses final reward ties to expose unsafe tie merging."},
        {"name": "random_merge_control_diagnostic_only", "family": "negative_control", "hard_merge": True, "diagnostic_only": True, "random_control": True, "description": "Not architecture: deterministic pseudo-random pair merge control."},
    ]
    return {
        "BG_CONVERGENCE_HAIR_POLICY_DEF_VERDICT": "READY" if pair_rows else "BLOCKED",
        "threshold_source": "calibration_non_heldout_only" if any(str(row.get("split")) != "heldout" for row in pair_rows) else "all_rows_no_non_heldout_available",
        "calibration_pair_count": len(cal),
        "heldout_pair_count": sum(1 for row in pair_rows if str(row.get("split")) == "heldout"),
        "quantiles": quantiles,
        "policies": policies,
        "runtime_classification_policy": "forbidden",
    }


def policy_allows_pair(policy: dict[str, Any], pair: dict[str, Any]) -> bool:
    if policy.get("name") == "no_hair_baseline":
        return False
    layers = policy.get("hair_layers")
    if layers and int(pair.get("hair_layer") or 0) not in {int(x) for x in layers}:
        return False
    if policy.get("reward_tie_required") and safe_float(pair.get("reward_tied_eval_only"), 0.0) <= 0:
        return False
    if policy.get("random_control"):
        digest = hashlib.sha1(str(pair.get("pair_id")).encode("utf-8")).hexdigest()
        return int(digest[:4], 16) % 7 == 0
    if policy.get("lineage_condition") == "same_parent" and safe_float(pair.get("same_parent"), 0.0) <= 0:
        return False
    if policy.get("lineage_condition") == "same_root" and safe_float(pair.get("same_root"), 0.0) <= 0:
        return False
    if not policy.get("anchor_disagreement_allowed", True) and safe_float(pair.get("anchor_disagreement_pair"), 0.0) > 0:
        return False
    for key, field in (
        ("hidden_cosine_max", "hidden_cosine_distance"),
        ("hidden_rms_norm_max", "hidden_rms_normalized"),
        ("dualanchor_abs_margin_max", "dualanchor_abs_avg_margin"),
        ("anchor_abs_margin_max", "anchor_abs_margin_max"),
    ):
        limit = policy.get(key)
        if limit is not None and math.isfinite(safe_float(limit, float("nan"))):
            if safe_float(pair.get(field), float("inf")) > float(limit):
                return False
    if policy.get("logit_required") and not truthy(pair.get("logit_kl_available")):
        return False
    return True


class UnionFind:
    def __init__(self, items: Sequence[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        rl = self.find(left)
        rr = self.find(right)
        if rl != rr:
            self.parent[rr] = rl

    def clusters(self) -> list[list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for item in self.parent:
            grouped[self.find(item)].append(item)
        return [sorted(vals) for vals in grouped.values()]


def cluster_for_policy(policy: dict[str, Any], ids: Sequence[str], pair_rows: Sequence[dict[str, Any]]) -> list[list[str]]:
    uf = UnionFind([str(item) for item in ids])
    idset = set(str(item) for item in ids)
    for pair in pair_rows:
        left = str(pair.get("left_candidate_id"))
        right = str(pair.get("right_candidate_id"))
        if left in idset and right in idset and policy_allows_pair(policy, pair):
            uf.union(left, right)
    return uf.clusters()


def choose_representative(cluster: Sequence[str], rows_by_id: dict[str, dict[str, Any]], scores: dict[str, float], top_roles: dict[str, int] | None = None) -> str:
    def score_tuple(branch_id: str) -> tuple[float, float, float, str]:
        row = rows_by_id[branch_id]
        clean_root = 1.0 if str(row.get("birth_stage")) == "root" or int(row.get("perturb_count") or 0) == 0 else 0.0
        return (-stable_risk(row), clean_root, safe_float(scores.get(branch_id), 0.0), branch_id)

    return max(cluster, key=score_tuple)


def max_subtree_reward(branch_id: str, rows_by_task: Sequence[dict[str, Any]]) -> float:
    vals = [row_reward(row) for row in rows_by_task if descendant_of(str(row.get("branch_id")), branch_id)]
    return max(vals) if vals else float("nan")


def max_terminal_subtree_reward(branch_id: str, final_ids: Sequence[str], rows_by_id: dict[str, dict[str, Any]]) -> float:
    vals = [row_reward(rows_by_id[item]) for item in final_ids if item in rows_by_id and descendant_of(item, branch_id)]
    return max(vals) if vals else float("nan")


def removed_by_any(branch_id: str, removed_ancestors: Sequence[str]) -> bool:
    return any(descendant_of(branch_id, ancestor) for ancestor in removed_ancestors)


def summarize_policy_task(
    policy: dict[str, Any],
    task_id: str,
    indexes: dict[str, Any],
    pairs_by_task_hair: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    rows_by_id = indexes["rows_by_id"]
    task_meta = indexes["task_by_id"][task_id]
    rows_by_task = indexes["rows_by_task"].get(task_id, [])
    final_ids = list(indexes["final_ids_by_task"].get(task_id, []))
    original_final_rewards = [row_reward(rows_by_id[item]) for item in final_ids if item in rows_by_id]
    original_oracle = max(original_final_rewards) if original_final_rewards else float("nan")
    removed: list[str] = []
    merge_events: list[dict[str, Any]] = []
    total_available = 0
    total_after = 0
    l30_before = l30_after = l42_before = l42_after = 0
    for spec in HAIR_SPECS:
        available = [
            branch_id
            for branch_id in available_ids_for_hair(task_id, spec, indexes)
            if branch_id in rows_by_id and not removed_by_any(branch_id, removed)
        ]
        if not available:
            continue
        pair_rows = pairs_by_task_hair.get((task_id, str(spec["hair_stage"])), [])
        total_available += len(available)
        if int(spec["hair_layer"]) == 30:
            l30_before += len(available)
        else:
            l42_before += len(available)
        if not policy.get("hard_merge"):
            clusters = cluster_for_policy(policy, available, pair_rows)
            total_after += len(available)
            if int(spec["hair_layer"]) == 30:
                l30_after += len(available)
            else:
                l42_after += len(available)
            if policy.get("name") == "soft_cluster_diagnostic":
                for cluster in clusters:
                    if len(cluster) > 1:
                        merge_events.append({"hair_stage": spec["hair_stage"], "soft_only": True, "cluster_size": len(cluster), "removed_count": 0, "false_merge": 0.0})
            continue
        clusters = cluster_for_policy(policy, available, pair_rows)
        retained_here = 0
        for cluster in clusters:
            if len(cluster) <= 1:
                retained_here += 1
                continue
            pair_scores: dict[str, list[float]] = defaultdict(list)
            for pair in pair_rows:
                left = str(pair.get("left_candidate_id"))
                right = str(pair.get("right_candidate_id"))
                if left in cluster and right in cluster:
                    margin = safe_float(pair.get("dualanchor_avg_margin"), 0.0)
                    pair_scores[left].append(margin)
                    pair_scores[right].append(-margin)
            scores = {branch_id: finite_mean(pair_scores.get(branch_id, [0.0])) for branch_id in cluster}
            rep = choose_representative(cluster, rows_by_id, scores)
            retained_here += 1
            rep_subtree = max_subtree_reward(rep, rows_by_task)
            rep_terminal = max_terminal_subtree_reward(rep, final_ids, rows_by_id)
            removed_cluster = [branch_id for branch_id in cluster if branch_id != rep]
            removed.extend(removed_cluster)
            removed_best = max((max_subtree_reward(branch_id, rows_by_task) for branch_id in removed_cluster), default=float("nan"))
            removed_terminal_best = max((max_terminal_subtree_reward(branch_id, final_ids, rows_by_id) for branch_id in removed_cluster), default=float("nan"))
            false_merge = 0.0
            if math.isfinite(removed_terminal_best) and math.isfinite(rep_terminal) and removed_terminal_best > rep_terminal:
                false_merge = 1.0
            elif math.isfinite(removed_best) and math.isfinite(rep_subtree) and removed_best > rep_subtree:
                false_merge = 1.0
            merge_events.append(
                {
                    "task_id": task_id,
                    "domain": task_meta.get("domain"),
                    "split": task_meta.get("split"),
                    "hair_stage": spec["hair_stage"],
                    "hair_layer": spec["hair_layer"],
                    "cluster_size": len(cluster),
                    "representative_id": rep,
                    "removed_count": len(removed_cluster),
                    "representative_subtree_reward": rep_subtree,
                    "removed_best_subtree_reward": removed_best,
                    "representative_terminal_subtree_reward": rep_terminal,
                    "removed_best_terminal_subtree_reward": removed_terminal_best,
                    "false_merge": false_merge,
                    "diagnostic_only": bool(policy.get("diagnostic_only")),
                }
            )
        total_after += retained_here
        if int(spec["hair_layer"]) == 30:
            l30_after += retained_here
        else:
            l42_after += retained_here
    filtered_final = [branch_id for branch_id in final_ids if not removed_by_any(branch_id, removed)]
    filtered_rewards = [row_reward(rows_by_id[item]) for item in filtered_final if item in rows_by_id]
    best_reward = max(filtered_rewards) if filtered_rewards else float("nan")
    oracle_retained = 1.0 if math.isfinite(best_reward) and math.isfinite(original_oracle) and best_reward >= original_oracle - 1e-9 else 0.0
    false_merge_events = [event for event in merge_events if safe_float(event.get("false_merge"), 0.0) > 0]
    return {
        "policy": policy.get("name"),
        "task_id": task_id,
        "domain": task_meta.get("domain"),
        "split": task_meta.get("split"),
        "source_dataset": task_meta.get("source_dataset"),
        "positive_oracle": task_meta.get("positive_oracle"),
        "terminal_reward_diverse": task_meta.get("terminal_reward_diverse"),
        "original_terminal_count": len(final_ids),
        "final_candidate_count": len(filtered_final),
        "avg_hair_survivors_before": total_available / max(len(HAIR_SPECS), 1),
        "avg_hair_survivors_after": total_after / max(len(HAIR_SPECS), 1),
        "hair_survivor_reduction": 1.0 - (total_after / total_available) if total_available else 0.0,
        "l30_survivors_before": l30_before,
        "l30_survivors_after": l30_after,
        "l42_survivors_before": l42_before,
        "l42_survivors_after": l42_after,
        "terminal_oracle_retained": oracle_retained,
        "stage_oracle_retained_after_hair": oracle_retained,
        "terminal_best_reward": best_reward,
        "original_terminal_best_reward": original_oracle,
        "false_merge_count": len(false_merge_events),
        "merge_event_count": sum(1 for event in merge_events if event.get("removed_count", 0)),
        "removed_ancestor_count": len(set(removed)),
        "terminal_forced_top1_effect": float("nan"),
        "terminal_confidence_gated_effect": float("nan"),
        "diagnostic_only": bool(policy.get("diagnostic_only")),
    }


def replay_policies(dataset: dict[str, Any], policies_payload: dict[str, Any]) -> dict[str, Any]:
    _payload, rows, _stage_rows, task_rows, _terminal_rows = load_v3_rows()
    indexes = build_indexes(rows, task_rows)
    pair_rows = dataset.get("pair_rows") or []
    pairs_by_task_hair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for pair in pair_rows:
        pairs_by_task_hair[(str(pair.get("task_id")), str(pair.get("hair_stage")))].append(pair)
    rows_out: list[dict[str, Any]] = []
    for policy in policies_payload.get("policies") or []:
        for task_id in sorted(indexes["task_by_id"]):
            task = indexes["task_by_id"][task_id]
            if str(task.get("domain")) in PRIMARY_DOMAINS:
                rows_out.append(summarize_policy_task(policy, task_id, indexes, pairs_by_task_hair))
    summary_rows = grouped_policy_summary(rows_out)
    verdict = replay_verdict(summary_rows)
    return {
        "BG_CONVERGENCE_HAIR_REPLAY_EVAL_VERDICT": verdict,
        "summary_rows": summary_rows,
        "rows": rows_out,
        "interpretation": "Replay-based merge evaluation is a safety/redundancy estimate, not a compute-savings claim.",
    }


def grouped_policy_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("policy"))].append(row)
    out: list[dict[str, Any]] = []
    for policy, vals in sorted(grouped.items()):
        non_diag = [row for row in vals if not truthy(row.get("diagnostic_only"))]
        false_merge_rate = finite_mean(1.0 if safe_float(row.get("false_merge_count"), 0.0) > 0 else 0.0 for row in vals)
        out.append(
            {
                "policy": policy,
                "task_count": len(vals),
                "diagnostic_only": finite_mean(row.get("diagnostic_only") for row in vals),
                "terminal_oracle_retained": finite_mean(row.get("terminal_oracle_retained") for row in vals),
                "hard_slice_terminal_oracle_retained": finite_mean(row.get("terminal_oracle_retained") for row in vals if safe_float(row.get("positive_oracle"), 0.0) > 0 and safe_float(row.get("terminal_reward_diverse"), 0.0) > 0),
                "false_merge_rate": false_merge_rate,
                "false_merge_count": sum(int(safe_float(row.get("false_merge_count"), 0.0)) for row in vals),
                "survivor_reduction": finite_mean(row.get("hair_survivor_reduction") for row in vals),
                "avg_final_candidates": finite_mean(row.get("final_candidate_count") for row in vals),
                "reasoning_false_merge_rate": finite_mean(1.0 if safe_float(row.get("false_merge_count"), 0.0) > 0 else 0.0 for row in vals if row.get("domain") == "reasoning"),
                "science_false_merge_rate": finite_mean(1.0 if safe_float(row.get("false_merge_count"), 0.0) > 0 else 0.0 for row in vals if row.get("domain") == "science"),
                "positive_oracle_false_merge_rate": finite_mean(1.0 if safe_float(row.get("false_merge_count"), 0.0) > 0 else 0.0 for row in vals if safe_float(row.get("positive_oracle"), 0.0) > 0),
                "reward_diverse_false_merge_rate": finite_mean(1.0 if safe_float(row.get("false_merge_count"), 0.0) > 0 else 0.0 for row in vals if safe_float(row.get("terminal_reward_diverse"), 0.0) > 0),
            }
        )
    return out


def replay_verdict(summary_rows: Sequence[dict[str, Any]]) -> str:
    architecture_rows = [row for row in summary_rows if not truthy(row.get("diagnostic_only")) and row.get("policy") not in {"no_hair_baseline", "soft_cluster_diagnostic"}]
    safe = [
        row
        for row in architecture_rows
        if safe_float(row.get("false_merge_rate"), 1.0) <= 0.05
        and safe_float(row.get("terminal_oracle_retained"), 0.0) >= 0.98
        and safe_float(row.get("survivor_reduction"), 0.0) >= 0.10
    ]
    acceptable = [
        row
        for row in architecture_rows
        if safe_float(row.get("false_merge_rate"), 1.0) <= 0.10
        and safe_float(row.get("terminal_oracle_retained"), 0.0) >= 0.98
        and safe_float(row.get("survivor_reduction"), 0.0) >= 0.10
    ]
    if safe:
        return "HARD_MERGE_SAFE"
    if acceptable:
        return "L30_HAIR_SAFE" if all("L30" in str(row.get("policy")) for row in acceptable) else "SOFT_CLUSTER_ONLY"
    if any(safe_float(row.get("false_merge_rate"), 0.0) > 0.10 for row in architecture_rows):
        return "FALSE_MERGE_TOO_HIGH"
    soft = next((row for row in summary_rows if row.get("policy") == "soft_cluster_diagnostic"), None)
    if soft and safe_float(soft.get("survivor_reduction"), 0.0) > 0:
        return "SOFT_CLUSTER_ONLY"
    return "DATA_LIMITED" if not architecture_rows else "INSUFFICIENT"


def slice_rows(rows: Sequence[dict[str, Any]], predicate: Any) -> list[dict[str, Any]]:
    return [row for row in rows if predicate(row)]


def task_slice_summary(task_rows: Sequence[dict[str, Any]], name: str, predicate: Any) -> dict[str, Any]:
    vals = slice_rows(task_rows, predicate)
    return {
        "slice": name,
        "count": len(vals),
        "terminal_oracle_retained": finite_mean(row.get("terminal_oracle_retained") for row in vals),
        "terminal_forced_top1_oracle": finite_mean(row.get("terminal_forced_top1_oracle") for row in vals),
        "terminal_forced_top1_reward": finite_mean(row.get("terminal_forced_top1_reward") for row in vals),
        "terminal_best_reward": finite_mean(row.get("terminal_best_reward") for row in vals),
        "terminal_confident": finite_mean(row.get("terminal_confident") for row in vals),
        "terminal_deferred": finite_mean(row.get("terminal_deferred") for row in vals),
        "positive_oracle": finite_mean(row.get("positive_oracle") for row in vals),
        "terminal_reward_diverse": finite_mean(row.get("terminal_reward_diverse") for row in vals),
        "stage_false_prunes": finite_mean(row.get("stage_false_prunes") for row in vals),
        "final_candidate_count": finite_mean(row.get("final_candidate_count") for row in vals),
    }


def terminal_policy_summary(rows: Sequence[dict[str, Any]], domain: str | None = None) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if domain is None or row.get("domain") == domain:
            grouped[str(row.get("policy"))].append(row)
    out = []
    for policy, vals in sorted(grouped.items()):
        out.append(
            {
                "policy": policy,
                "count": len(vals),
                "selected_count": finite_mean(row.get("selected_count") for row in vals),
                "oracle_retained": finite_mean(row.get("oracle_retained") for row in vals),
                "best_selected_reward": finite_mean(row.get("best_selected_reward") for row in vals),
                "first_selected_reward": finite_mean(row.get("first_selected_reward") for row in vals),
                "first_selected_oracle": finite_mean(row.get("first_selected_oracle") for row in vals),
                "defer_rate": finite_mean(row.get("terminal_deferred") for row in vals),
                "confident_rate": finite_mean(row.get("terminal_confident") for row in vals),
            }
        )
    return out


def load_replay_rows() -> list[dict[str, str]]:
    return read_csv(REPLAY_ROWS_CSV)


def load_terminal_rows_csv() -> list[dict[str, str]]:
    return read_csv(V3_TERMINAL_ROWS)


def load_task_rows_csv() -> list[dict[str, str]]:
    return read_csv(V3_TASK_ROWS)


def status_line(key: str, value: str) -> str:
    return f"{key} = {value}"

