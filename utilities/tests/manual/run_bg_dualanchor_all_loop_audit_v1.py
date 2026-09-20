"""DualAnchor all-loop branch/content audit v1.

This probe fixes the L4-only limitation in the earlier cached loop policy
scripts. It evaluates DualAnchor and old-anchor taps as layer-local pairwise
evaluators at transformer layers 24, 36, and 47 over loop states L1..L4.

It does not run generation, true fork/carry, action steering, Ouro training,
registry updates, wrapper/local-agent code, Hunter-Seeker code, or production
routing changes.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import torch

from bg_hidden_origin_tap_common import PROBE_ROOT, md_table, rel
from bg_merged_tap_v1_common import BRIDGE_PT, json_default, safe_float, score_diff
from run_bg_layer_native_two_tap_readiness_v1 import (
    ANCHOR_GROUPS,
    candidate_name,
    group_reward,
    tensor_weight,
    write_csv,
    write_json,
    write_md,
)
from run_bg_two_tap_full_readiness_v1 import (
    SURVIVOR_FEATURES_PT,
    branch_pair_datasets,
    old_domain_pair_datasets,
)

import run_bg_dualanchor_looped_branch_prune_sim_v1 as l4_base


OUT_ROOT = PROBE_ROOT / "bg_dualanchor_all_loop_audit_v1_2026-05-31"
REPORT_JSON = OUT_ROOT / "all_loop_audit.json"
REPORT_MD = OUT_ROOT / "all_loop_audit.md"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
ARTIFACT_PT = OUT_ROOT / "dualanchor_all_loop_audit_v1.pt"
PAIR_ROWS_CSV = OUT_ROOT / "all_loop_pair_rows.csv"
GROUP_ROWS_CSV = OUT_ROOT / "all_loop_group_rows.csv"
STAGE_ROWS_CSV = OUT_ROOT / "all_loop_stage_rows.csv"
CONFIG_COVERAGE_CSV = OUT_ROOT / "config_coverage.csv"
POLICY_ROWS_CSV = OUT_ROOT / "policy_rows.csv"

BGV1_BRANCHES_PT = PROBE_ROOT / "bg_hidden_origin_branch_generator_v1_2026-05-18/branch_generator_v1_branches.pt"
V4_BRANCHES_PT = PROBE_ROOT / "bg_hidden_origin_quota_v4_2026-05-18/quota_hidden_origin_branches.pt"

TAP_LAYERS = (24, 36, 47)
LAYER_TO_POS = {24: 0, 36: 1, 47: 2}
LOOPS = (1, 2, 3, 4)
HIDDEN_DIM = 2048

LOOP_CONFIGS = (
    "24_L1",
    "24_L2",
    "24_L3",
    "24_L4",
    "24_mean",
    "36_L1",
    "36_L2",
    "36_L3",
    "36_L4",
    "36_mean",
    "47_L1",
    "47_L2",
    "47_L3",
    "47_L4",
    "47_mean",
    "47_concat_L1_L4",
    "47_concat_all_loops",
)

LAYER_BY_CONFIG = {
    "24_L1": 24,
    "24_L2": 24,
    "24_L3": 24,
    "24_L4": 24,
    "24_mean": 24,
    "36_L1": 36,
    "36_L2": 36,
    "36_L3": 36,
    "36_L4": 36,
    "36_mean": 36,
    "47_L1": 47,
    "47_L2": 47,
    "47_L3": 47,
    "47_L4": 47,
    "47_mean": 47,
    "47_concat_L1_L4": 47,
    "47_concat_all_loops": 47,
}

PRIMARY_THRESHOLD_POLICIES = {
    "mean_floor_very_loose": {
        "kind": "mean_floor",
        "offset_by_loop": [-2.0, -1.75, -1.50, -1.25],
        "offset_by_layer": {24: -0.20, 36: -0.05, 47: 0.00},
    },
    "mean_floor_loose": {
        "kind": "mean_floor",
        "offset_by_loop": [-1.25, -1.00, -0.75, -0.50],
        "offset_by_layer": {24: -0.15, 36: -0.05, 47: 0.00},
    },
    "maxband_loose": {
        "kind": "max_band",
        "band_by_loop": [2.50, 2.20, 1.90, 1.60],
        "band_by_layer": {24: 1.30, 36: 1.10, 47: 1.00},
    },
}


def ensure_root() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def finite_mean(vals: Iterable[Any]) -> float:
    xs: list[float] = []
    for value in vals:
        try:
            x = float(value)
        except Exception:
            continue
        if math.isfinite(x):
            xs.append(x)
    return float(mean(xs)) if xs else float("nan")


def normalize_scores(vals: Sequence[float]) -> list[float]:
    xs = [safe_float(v, 0.0) for v in vals]
    if not xs:
        return []
    mu = mean(xs)
    sd = math.sqrt(mean((x - mu) ** 2 for x in xs)) if len(xs) > 1 else 0.0
    if sd < 1e-8:
        return [0.0 for _ in xs]
    return [(x - mu) / sd for x in xs]


def vector_dim(config: str) -> int:
    if config == "47_concat_L1_L4":
        return HIDDEN_DIM * 2
    if config == "47_concat_all_loops":
        return HIDDEN_DIM * 4
    return HIDDEN_DIM


def config_to_layer_loop(config: str) -> tuple[int, int | None, str]:
    if config.endswith("_mean"):
        return int(config.split("_")[0]), None, "mean"
    if config == "47_concat_L1_L4":
        return 47, None, "concat_L1_L4"
    if config == "47_concat_all_loops":
        return 47, None, "concat_all_loops"
    layer_s, loop_s = config.split("_L")
    return int(layer_s), int(loop_s), "loop"


def raw_loop_vector(row: dict[str, Any], layer: int, loop: int) -> torch.Tensor | None:
    features = row.get("features")
    if isinstance(features, torch.Tensor) and tuple(features.shape[-3:]) == (3, 4, HIDDEN_DIM):
        value = features[LAYER_TO_POS[layer], loop - 1]
        return value.detach().cpu().to(torch.float32).flatten()
    pooled = row.get("pooled_vectors")
    if isinstance(pooled, dict):
        for key in (f"L{layer}_L{loop}", f"{layer}_L{loop}"):
            value = pooled.get(key)
            if isinstance(value, torch.Tensor) and int(value.numel()) == HIDDEN_DIM:
                return value.detach().cpu().to(torch.float32).flatten()
    fmap = row.get("features_by_config")
    if isinstance(fmap, dict):
        key = f"{layer}_L{loop}"
        value = fmap.get(key)
        if isinstance(value, torch.Tensor) and int(value.numel()) == HIDDEN_DIM:
            return value.detach().cpu().to(torch.float32).flatten()
    return None


def config_vector(row: dict[str, Any], config: str) -> torch.Tensor | None:
    fmap = row.get("features_by_config")
    if isinstance(fmap, dict):
        value = fmap.get(config)
        if isinstance(value, torch.Tensor) and int(value.numel()) == vector_dim(config):
            return value.detach().cpu().to(torch.float32).flatten()
    layer, loop, kind = config_to_layer_loop(config)
    if kind == "loop" and loop is not None:
        return raw_loop_vector(row, layer, loop)
    if kind == "mean":
        vals = [raw_loop_vector(row, layer, i) for i in LOOPS]
        if all(isinstance(v, torch.Tensor) for v in vals):
            return torch.stack([v for v in vals if isinstance(v, torch.Tensor)], dim=0).mean(dim=0)
    if kind == "concat_L1_L4":
        v1 = raw_loop_vector(row, 47, 1)
        v4 = raw_loop_vector(row, 47, 4)
        if isinstance(v1, torch.Tensor) and isinstance(v4, torch.Tensor):
            return torch.cat([v1, v4], dim=0)
    if kind == "concat_all_loops":
        vals = [raw_loop_vector(row, 47, i) for i in LOOPS]
        if all(isinstance(v, torch.Tensor) for v in vals):
            return torch.cat([v for v in vals if isinstance(v, torch.Tensor)], dim=0)
    return None


def has_full_loop_tensor(row: dict[str, Any]) -> bool:
    return all(isinstance(raw_loop_vector(row, layer, loop), torch.Tensor) for layer in TAP_LAYERS for loop in LOOPS)


def row_reward(row: dict[str, Any]) -> float:
    for key in ("deterministic_reward", "reward", "final_reward", "correctness"):
        try:
            x = float(row.get(key))
        except Exception:
            continue
        if math.isfinite(x):
            return x
    if row.get("deterministic_correct") or row.get("correct"):
        return 1.0
    return 0.0


def normalize_branch_row(row: dict[str, Any], source: str) -> dict[str, Any] | None:
    group_id = row.get("branch_group_id") or row.get("group_id") or row.get("survivor_set_id")
    if group_id is None:
        return None
    if not has_full_loop_tensor(row):
        return None
    out = dict(row)
    out["group_id"] = str(group_id)
    out["source_dataset"] = source
    out["reward"] = row_reward(out)
    out.setdefault("domain", row.get("domain") or "unknown")
    out.setdefault("split", row.get("split") or "all")
    return out


def load_full_loop_branch_groups() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    datasets: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for name, path in (("branch_generator_v1", BGV1_BRANCHES_PT), ("hidden_origin_quota_v4", V4_BRANCHES_PT)):
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for raw in payload.get("rows") or []:
            row = normalize_branch_row(dict(raw), name)
            if row is not None:
                grouped[row["group_id"]].append(row)
        groups = [
            vals
            for vals in grouped.values()
            if len(vals) >= 2 and max(row_reward(v) for v in vals) > min(row_reward(v) for v in vals)
        ]
        datasets.append({"dataset_name": name, "groups": groups, "full_loop": True, "readiness_eligible": True})
        inventory.append(
            {
                "dataset_name": name,
                "group_count": len(groups),
                "candidate_count": sum(len(g) for g in groups),
                "full_loop": True,
                "domains": dict(Counter(str(g[0].get("domain")) for g in groups)),
            }
        )
    return datasets, inventory


def load_partial_branch_groups() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    datasets: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    if BRIDGE_PT.exists():
        payload = torch.load(BRIDGE_PT, map_location="cpu", weights_only=False)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in payload.get("candidate_rows") or []:
            if str(row.get("split") or "") != "heldout":
                continue
            group_id = row.get("group_id") or row.get("branch_group_id")
            if group_id is None:
                continue
            grouped[str(group_id)].append(dict(row, reward=row_reward(row), source_dataset="universal_bridge_candidate_groups"))
        groups = [
            vals
            for vals in grouped.values()
            if len(vals) >= 2 and max(row_reward(v) for v in vals) > min(row_reward(v) for v in vals)
        ]
        datasets.append({"dataset_name": "universal_bridge_candidate_groups", "groups": groups, "full_loop": False, "readiness_eligible": True})
        inventory.append(
            {
                "dataset_name": "universal_bridge_candidate_groups",
                "group_count": len(groups),
                "candidate_count": sum(len(g) for g in groups),
                "full_loop": False,
                "domains": dict(Counter(str(g[0].get("domain")) for g in groups)),
            }
        )
    if SURVIVOR_FEATURES_PT.exists():
        payload = torch.load(SURVIVOR_FEATURES_PT, map_location="cpu", weights_only=False)
        groups = []
        for survivor_set in payload.get("survivor_sets") or []:
            if str(survivor_set.get("split") or survivor_set.get("v1_split") or "") != "fresh_holdout":
                continue
            vals = []
            for cand in survivor_set.get("candidates") or []:
                row = dict(cand)
                row["group_id"] = str(survivor_set.get("survivor_set_id") or survivor_set.get("group_id"))
                row["source_dataset"] = "top4_survivor_hidden_feature_groups"
                row["reward"] = row_reward(row)
                vals.append(row)
            if len(vals) >= 2 and max(row_reward(v) for v in vals) > min(row_reward(v) for v in vals):
                groups.append(vals)
        datasets.append({"dataset_name": "top4_survivor_hidden_feature_groups", "groups": groups, "full_loop": False, "readiness_eligible": True})
        inventory.append(
            {
                "dataset_name": "top4_survivor_hidden_feature_groups",
                "group_count": len(groups),
                "candidate_count": sum(len(g) for g in groups),
                "full_loop": False,
                "domains": dict(Counter(str(g[0].get("domain")) for g in groups)),
            }
        )
    return datasets, inventory


def load_constrained_policies() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates, _refs = l4_base.load_constrained()
    policies, policy_rows = l4_base.policy_inventory(candidates)
    keep = {
        "dualanchor_old_only_AntisymLinear",
        "dualanchor_old_only_AntisymLinearNoNorm",
        "dualanchor_adaptive_balanced_rescue_AntisymLinear",
        "dualanchor_adaptive_branch_anchor_light_AntisymLinear",
        "dualanchor_adaptive_branch_from_val_AntisymLinear",
        "dualanchor_adaptive_balanced_rescue_AntisymLinearNoNorm",
        "dualanchor_adaptive_branch_anchor_light_AntisymLinearNoNorm",
        "dualanchor_adaptive_branch_from_val_AntisymLinearNoNorm",
    }
    selected = {name: taps for name, taps in policies.items() if name in keep}
    rows = [row for row in policy_rows if row.get("policy") in selected]
    return selected, rows


def tap_stage_scores(group: Sequence[dict[str, Any]], taps_by_layer: dict[str, dict[str, dict[str, Any]]], layer: int, config: str) -> list[float]:
    layer_key = f"{layer}_L4"
    role_scores: list[list[float]] = []
    for role in ANCHOR_GROUPS:
        tap = (taps_by_layer.get(layer_key) or {}).get(role)
        weight = tensor_weight(tap or {})
        if not isinstance(weight, torch.Tensor):
            continue
        arch = str((tap or {}).get("architecture") or "AntisymLinear")
        scores: list[float] = []
        for i, left in enumerate(group):
            lvec = config_vector(left, config)
            if not isinstance(lvec, torch.Tensor) or int(lvec.numel()) != int(weight.numel()):
                return []
            vals = []
            for j, right in enumerate(group):
                if i == j:
                    continue
                rvec = config_vector(right, config)
                if not isinstance(rvec, torch.Tensor) or int(rvec.numel()) != int(weight.numel()):
                    return []
                vals.append(score_diff(weight, arch, lvec - rvec))
            scores.append(finite_mean(vals))
        role_scores.append(normalize_scores(scores))
    if not role_scores:
        return []
    n = min(len(s) for s in role_scores)
    return [finite_mean(scores[i] for scores in role_scores) for i in range(n)]


def tap_pair_score(left: dict[str, Any], right: dict[str, Any], taps_by_layer: dict[str, dict[str, dict[str, Any]]], config: str) -> float:
    layer = LAYER_BY_CONFIG[config]
    if config == "47_concat_L1_L4":
        vals = [
            tap_pair_score(left, right, taps_by_layer, "47_L1"),
            tap_pair_score(left, right, taps_by_layer, "47_L4"),
        ]
        return finite_mean(vals)
    if config == "47_concat_all_loops":
        return finite_mean(tap_pair_score(left, right, taps_by_layer, f"47_L{i}") for i in LOOPS)
    layer_key = f"{layer}_L4"
    vals = []
    for role in ANCHOR_GROUPS:
        tap = (taps_by_layer.get(layer_key) or {}).get(role)
        weight = tensor_weight(tap or {})
        if not isinstance(weight, torch.Tensor):
            continue
        lvec = config_vector(left, config)
        rvec = config_vector(right, config)
        if not isinstance(lvec, torch.Tensor) or not isinstance(rvec, torch.Tensor):
            continue
        if int(lvec.numel()) != int(weight.numel()) or int(rvec.numel()) != int(weight.numel()):
            continue
        vals.append(score_diff(weight, str(tap.get("architecture") or "AntisymLinear"), lvec - rvec))
    return finite_mean(vals)


def pair_accuracy_for_group_rows(group: Sequence[dict[str, Any]], taps_by_layer: dict[str, dict[str, dict[str, Any]]], config: str) -> tuple[float, int]:
    correct = 0.0
    total = 0
    rewards = [row_reward(row) for row in group]
    for i, left in enumerate(group):
        for j, right in enumerate(group):
            if i >= j or rewards[i] == rewards[j]:
                continue
            if rewards[i] > rewards[j]:
                score = tap_pair_score(left, right, taps_by_layer, config)
            else:
                score = tap_pair_score(right, left, taps_by_layer, config)
            if not math.isfinite(score):
                continue
            correct += 1.0 if score > 0 else 0.5 if score == 0 else 0.0
            total += 1
    return (correct / total if total else float("nan"), total)


def group_metric(group: Sequence[dict[str, Any]], order: Sequence[int], k: int) -> dict[str, Any]:
    rewards = [row_reward(row) for row in group]
    if not rewards or not order:
        return {}
    oracle = max(rewards)
    oracle_idx = {i for i, reward in enumerate(rewards) if reward == oracle}
    selected = list(order[: min(k, len(order))])
    kept = bool(set(selected) & oracle_idx)
    best_selected = max(rewards[i] for i in selected)
    return {
        "oracle_retention": 1.0 if kept else 0.0,
        "false_prune_rate": 0.0 if kept else 1.0,
        "avg_survivors": float(len(selected)),
        "best_selected_reward": best_selected,
        "oracle_reward": oracle,
        "regret": oracle - best_selected,
        "top1_reward": rewards[selected[0]] if selected else float("nan"),
        "top1_success": 1.0 if selected and rewards[selected[0]] == oracle else 0.0,
    }


def order_from_scores(scores: Sequence[float]) -> list[int]:
    return [idx for idx, _score in sorted(enumerate(scores), key=lambda item: (-safe_float(item[1], -1e9), item[0]))]


def score_stats(scores: Sequence[float]) -> tuple[float, float, float, float]:
    vals = [safe_float(v, 0.0) for v in scores]
    if not vals:
        return 0.0, 0.0, 0.0, 0.0
    mu = mean(vals)
    sd = math.sqrt(mean((v - mu) ** 2 for v in vals)) if len(vals) > 1 else 0.0
    return mu, sd, max(vals), min(vals)


def threshold_value(scores: Sequence[float], loop_idx: int, layer: int, spec: dict[str, Any]) -> float:
    mu, sd, max_score, _min_score = score_stats(scores)
    if spec["kind"] == "mean_floor":
        offset = float(spec["offset_by_loop"][loop_idx]) + float(spec["offset_by_layer"].get(layer, 0.0))
        return mu + offset * max(sd, 1e-6)
    if spec["kind"] == "max_band":
        band = float(spec["band_by_loop"][loop_idx]) * float(spec["band_by_layer"].get(layer, 1.0))
        return max_score - band * max(sd, 1e-6)
    raise ValueError(f"unknown threshold kind: {spec['kind']}")


def threshold_keep(scores: Sequence[float], threshold: float) -> tuple[list[int], bool]:
    vals = [safe_float(v, -1e9) for v in scores]
    keep = [i for i, v in enumerate(vals) if v >= threshold]
    if keep:
        return keep, False
    if not vals:
        return [], True
    return [max(range(len(vals)), key=lambda i: (vals[i], -i))], True


def simulate_threshold_loop(
    group: Sequence[dict[str, Any]],
    taps_by_layer: dict[str, dict[str, dict[str, Any]]],
    content_taps_by_layer: dict[str, dict[str, dict[str, Any]]],
    threshold_spec: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    survivors = list(range(len(group)))
    rewards = [row_reward(row) for row in group]
    oracle = max(rewards)
    oracle_idx = {idx for idx, reward in enumerate(rewards) if reward == oracle}
    stage_rows: list[dict[str, Any]] = []
    fallback_count = 0
    first_oracle_loss: dict[str, Any] | None = None

    for loop_idx, loop in enumerate(LOOPS):
        for layer in TAP_LAYERS:
            if len(survivors) <= 1:
                break
            if loop == 4 and layer == 47:
                continue
            config = f"{layer}_L{loop}"
            sub_group = [group[i] for i in survivors]
            scores = tap_stage_scores(sub_group, taps_by_layer, layer, config)
            if not scores:
                continue
            threshold = threshold_value(scores, loop_idx, layer, threshold_spec)
            local_keep, used_fallback = threshold_keep(scores, threshold)
            before = list(survivors)
            survivors = [survivors[i] for i in local_keep]
            fallback_count += 1 if used_fallback else 0
            retained_oracle = bool(set(survivors) & oracle_idx)
            if not retained_oracle and first_oracle_loss is None:
                first_oracle_loss = {"loop": loop, "layer": layer}
            mu, sd, max_score, min_score = score_stats(scores)
            stage_rows.append(
                {
                    "loop": loop,
                    "layer": layer,
                    "config": config,
                    "survivors_before": len(before),
                    "survivors_after": len(survivors),
                    "threshold": threshold,
                    "score_mean": mu,
                    "score_std": sd,
                    "score_max": max_score,
                    "score_min": min_score,
                    "fallback_used": 1.0 if used_fallback else 0.0,
                    "oracle_retained_after_stage": 1.0 if retained_oracle else 0.0,
                }
            )

    pre_terminal = list(survivors)
    pre_retained = bool(set(pre_terminal) & oracle_idx)
    pre_best = max(rewards[i] for i in pre_terminal) if pre_terminal else float("nan")
    terminal_scores = tap_stage_scores([group[i] for i in pre_terminal], content_taps_by_layer, 47, "47_L4")
    if terminal_scores:
        local_best = max(range(len(terminal_scores)), key=lambda i: (safe_float(terminal_scores[i], -1e9), -i))
        selected_idx = pre_terminal[local_best]
    else:
        selected_idx = pre_terminal[0] if pre_terminal else 0
    selected_reward = rewards[selected_idx] if selected_idx < len(rewards) else float("nan")
    return (
        {
            "group_size": len(group),
            "pre_terminal_survivor_count": len(pre_terminal),
            "pre_terminal_oracle_retained": 1.0 if pre_retained else 0.0,
            "pre_terminal_false_prune": 0.0 if pre_retained else 1.0,
            "pre_terminal_best_reward": pre_best,
            "terminal_selected_reward": selected_reward,
            "terminal_oracle_selected": 1.0 if selected_idx in oracle_idx else 0.0,
            "terminal_regret": oracle - selected_reward,
            "best_survivor_regret": oracle - pre_best,
            "threshold_fallback_count": fallback_count,
            "first_oracle_loss_loop": None if first_oracle_loss is None else first_oracle_loss["loop"],
            "first_oracle_loss_layer": None if first_oracle_loss is None else first_oracle_loss["layer"],
            "terminal_selected_index": selected_idx,
            "pre_terminal_selected_indices": pre_terminal,
        },
        stage_rows,
    )


def summarize_rows(rows: Sequence[dict[str, Any]], prefix: str = "") -> dict[str, Any]:
    if not rows:
        return {"group_count": 0}
    keys = [
        "oracle_retention",
        "false_prune_rate",
        "avg_survivors",
        "best_selected_reward",
        "top1_reward",
        "top1_success",
        "pre_terminal_oracle_retained",
        "pre_terminal_false_prune",
        "pre_terminal_survivor_count",
        "pre_terminal_best_reward",
        "terminal_selected_reward",
        "terminal_oracle_selected",
        "terminal_regret",
        "best_survivor_regret",
    ]
    out = {"group_count": len(rows)}
    for key in keys:
        vals = [row.get(key) for row in rows if key in row]
        if vals:
            out[prefix + key] = finite_mean(vals)
    return out


def evaluate_group_configs(datasets: Sequence[dict[str, Any]], policies: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy_name, taps in policies.items():
        for dataset in datasets:
            for config in LOOP_CONFIGS:
                group_metrics = []
                pair_accs = []
                pair_counts = []
                for group in dataset["groups"]:
                    layer = LAYER_BY_CONFIG[config]
                    scores = tap_stage_scores(group, taps, layer, config)
                    if scores:
                        group_metrics.append(group_metric(group, order_from_scores(scores), k=min(4, len(group))))
                    acc, count = pair_accuracy_for_group_rows(group, taps, config)
                    if math.isfinite(acc):
                        pair_accs.append(acc)
                        pair_counts.append(count)
                if not group_metrics and not pair_accs:
                    continue
                sm = summarize_rows(group_metrics)
                rows.append(
                    {
                        "dataset_name": dataset["dataset_name"],
                        "full_loop": bool(dataset.get("full_loop")),
                        "policy": policy_name,
                        "config": config,
                        "group_count": sm.get("group_count", 0),
                        "pair_count": int(sum(pair_counts)),
                        "pairwise_accuracy": finite_mean(pair_accs),
                        "oracle_retention": sm.get("oracle_retention"),
                        "false_prune_rate": sm.get("false_prune_rate"),
                        "avg_survivors": sm.get("avg_survivors"),
                        "best_selected_reward": sm.get("best_selected_reward"),
                        "top1_reward": sm.get("top1_reward"),
                        "top1_success": sm.get("top1_success"),
                    }
                )
    return rows


def pair_feature_vector(pair: dict[str, Any], side: str, config: str) -> torch.Tensor | None:
    vals = (pair.get("features") or {}).get(config)
    if isinstance(vals, dict):
        value = vals.get(side)
        if isinstance(value, torch.Tensor) and int(value.numel()) == vector_dim(config):
            return value.detach().cpu().to(torch.float32).flatten()
    return None


def evaluate_cached_pair_datasets(policies: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pair_datasets = []
    for dataset in old_domain_pair_datasets():
        dataset = dict(dataset)
        dataset["dataset_kind"] = "content_domain"
        pair_datasets.append(dataset)
    for dataset in branch_pair_datasets():
        dataset = dict(dataset)
        dataset["dataset_kind"] = "branch_pair"
        pair_datasets.append(dataset)
    for dataset in pair_datasets:
        pairs = list(dataset.get("pairs") or [])
        if not pairs:
            continue
        for policy_name, taps in policies.items():
            for config in LOOP_CONFIGS:
                if config in {"24_L2", "24_L3", "36_L2", "36_L3", "47_L1", "47_L2", "47_L3"}:
                    # Cached pair datasets generally store config-level features,
                    # not raw loop tensors. Full-loop branch rows are audited
                    # separately above.
                    continue
                correct = 0.0
                total = 0
                margins = []
                for pair in pairs:
                    pref = pair_feature_vector(pair, "preferred", config)
                    rej = pair_feature_vector(pair, "rejected", config)
                    if not isinstance(pref, torch.Tensor) or not isinstance(rej, torch.Tensor):
                        continue
                    fake_left = {"features_by_config": {config: pref}}
                    fake_right = {"features_by_config": {config: rej}}
                    score = tap_pair_score(fake_left, fake_right, taps, config)
                    if not math.isfinite(score):
                        continue
                    correct += 1.0 if score > 0 else 0.5 if score == 0 else 0.0
                    total += 1
                    margins.append(score)
                if total <= 0:
                    continue
                rows.append(
                    {
                        "dataset_name": dataset.get("dataset_name"),
                        "dataset_kind": dataset.get("dataset_kind"),
                        "policy": policy_name,
                        "config": config,
                        "pair_count": total,
                        "pairwise_accuracy": correct / total,
                        "mean_margin": finite_mean(margins),
                        "readiness_eligible": bool(dataset.get("readiness_eligible")),
                    }
                )
    return rows


def evaluate_threshold_policies(full_loop_datasets: Sequence[dict[str, Any]], policies: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    old_content_policies = {name: taps for name, taps in policies.items() if "old_only" in name}
    content_fallback = next(iter(old_content_policies.values())) if old_content_policies else next(iter(policies.values()))
    group_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for policy_name, taps in policies.items():
        for threshold_name, threshold_spec in PRIMARY_THRESHOLD_POLICIES.items():
            all_rows = []
            for dataset in full_loop_datasets:
                dataset_rows = []
                for group in dataset["groups"]:
                    row, stages = simulate_threshold_loop(group, taps, content_fallback, threshold_spec)
                    row.update(
                        {
                            "dataset_name": dataset["dataset_name"],
                            "group_id": group[0].get("group_id"),
                            "domain": group[0].get("domain"),
                            "policy": policy_name,
                            "threshold_policy": threshold_name,
                            "terminal_policy": "old_anchor_content_47_L4",
                        }
                    )
                    dataset_rows.append(row)
                    group_rows.append(row)
                    for stage in stages:
                        stage_rows.append(
                            {
                                **stage,
                                "dataset_name": dataset["dataset_name"],
                                "group_id": group[0].get("group_id"),
                                "domain": group[0].get("domain"),
                                "policy": policy_name,
                                "threshold_policy": threshold_name,
                            }
                        )
                sm = summarize_rows(dataset_rows)
                sm.update({"dataset_name": dataset["dataset_name"], "policy": policy_name, "threshold_policy": threshold_name})
                summary_rows.append(sm)
                all_rows.extend(dataset_rows)
            sm = summarize_rows(all_rows)
            sm.update({"dataset_name": "ALL_FULL_LOOP", "policy": policy_name, "threshold_policy": threshold_name})
            summary_rows.append(sm)
    return group_rows, stage_rows, summary_rows


def best_threshold_summary(summary_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = [r for r in summary_rows if r.get("dataset_name") == "ALL_FULL_LOOP" and int(r.get("group_count") or 0) > 0]
    if not rows:
        return {}
    return sorted(
        rows,
        key=lambda r: (
            -safe_float(r.get("pre_terminal_oracle_retained"), 0.0),
            safe_float(r.get("pre_terminal_false_prune"), 1.0),
            -safe_float(r.get("pre_terminal_best_reward"), 0.0),
            -safe_float(r.get("terminal_selected_reward"), 0.0),
            safe_float(r.get("pre_terminal_survivor_count"), 99.0),
        ),
    )[0]


def verdict_for(best: dict[str, Any], coverage: Sequence[dict[str, Any]]) -> str:
    if not best:
        return "DATA_LIMITED"
    full_loop_groups = sum(int(row.get("group_count") or 0) for row in coverage if row.get("full_loop"))
    if full_loop_groups <= 0:
        return "DATA_LIMITED"
    retention = safe_float(best.get("pre_terminal_oracle_retained"), 0.0)
    false_prune = safe_float(best.get("pre_terminal_false_prune"), 1.0)
    terminal = safe_float(best.get("terminal_selected_reward"), 0.0)
    if retention >= 0.93 and false_prune <= 0.07 and terminal >= 0.75:
        return "ALL_LOOP_DUALANCHOR_READY"
    if retention >= 0.90 and false_prune <= 0.10:
        return "ALL_LOOP_SURVIVAL_READY_TERMINAL_WEAK"
    if retention >= 0.85 and false_prune <= 0.15:
        return "ALL_LOOP_SURVIVAL_PARTIAL"
    return "ALL_LOOP_SURVIVAL_WEAK"


def main() -> int:
    ensure_root()
    policies, policy_rows = load_constrained_policies()
    full_loop_datasets, full_inventory = load_full_loop_branch_groups()
    partial_datasets, partial_inventory = load_partial_branch_groups()
    group_config_rows = evaluate_group_configs(full_loop_datasets + partial_datasets, policies)
    cached_pair_rows = evaluate_cached_pair_datasets(policies)
    group_rows, stage_rows, threshold_summary_rows = evaluate_threshold_policies(full_loop_datasets, policies)
    best = best_threshold_summary(threshold_summary_rows)
    coverage = full_inventory + partial_inventory
    verdict = verdict_for(best, coverage)

    pair_rows = group_config_rows + cached_pair_rows
    write_csv(PAIR_ROWS_CSV, pair_rows)
    write_csv(GROUP_ROWS_CSV, group_rows)
    write_csv(STAGE_ROWS_CSV, stage_rows)
    write_csv(POLICY_ROWS_CSV, policy_rows)
    write_csv(CONFIG_COVERAGE_CSV, coverage)
    payload = {
        "BG_DUALANCHOR_ALL_LOOP_AUDIT_VERDICT": verdict,
        "status": verdict,
        "mode": "CACHED_ALL_LOOP_DUALANCHOR_AUDIT",
        "true_fork_carry_claimed": False,
        "action_steering_claimed": False,
        "uses_loop4_only": False,
        "taps_present_each_loop": True,
        "layers": list(TAP_LAYERS),
        "loops": list(LOOPS),
        "loop_configs": list(LOOP_CONFIGS),
        "terminal_selection": {
            "explicit_top1_only_at_final_47": True,
            "terminal_policy": "old_anchor_content_47_L4",
            "note": "Non-terminal 47 participates as branch/content thresholding at loops 1-3; final loop 4 layer 47 uses content/output winner selection.",
        },
        "policy_count": len(policies),
        "threshold_policies": PRIMARY_THRESHOLD_POLICIES,
        "coverage": coverage,
        "best_threshold_summary": best,
        "threshold_summary_rows": threshold_summary_rows,
        "anti_leakage": {
            "cached_features_only": True,
            "no_true_fork_carry": True,
            "no_action_steering": True,
            "no_ouro_training": True,
            "no_registry_update": True,
            "no_production_routing_change": True,
        },
    }
    write_json(REPORT_JSON, payload)
    write_json(SUMMARY_JSON, payload)
    torch.save(
        {
            "summary": payload,
            "pair_rows": pair_rows,
            "group_rows": group_rows,
            "stage_rows": stage_rows,
            "threshold_summary_rows": threshold_summary_rows,
            "policy_rows": policy_rows,
        },
        ARTIFACT_PT,
    )

    overall_threshold = [r for r in threshold_summary_rows if r.get("dataset_name") == "ALL_FULL_LOOP"]
    overall_threshold = sorted(
        overall_threshold,
        key=lambda r: (
            -safe_float(r.get("pre_terminal_oracle_retained"), 0.0),
            safe_float(r.get("pre_terminal_false_prune"), 1.0),
            -safe_float(r.get("pre_terminal_best_reward"), 0.0),
            -safe_float(r.get("terminal_selected_reward"), 0.0),
        ),
    )[:20]
    selected_policy = best.get("policy")
    selected_threshold = best.get("threshold_policy")
    selected_stage = [
        r
        for r in stage_rows
        if r.get("policy") == selected_policy and r.get("threshold_policy") == selected_threshold
    ]
    by_stage: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in selected_stage:
        by_stage[(int(row["loop"]), int(row["layer"]))].append(row)
    stage_summary = []
    for (loop, layer), vals in sorted(by_stage.items()):
        stage_summary.append(
            {
                "loop": loop,
                "layer": layer,
                "stage_count": len(vals),
                "oracle_retention_after_stage": finite_mean(v.get("oracle_retained_after_stage") for v in vals),
                "avg_survivors_before": finite_mean(v.get("survivors_before") for v in vals),
                "avg_survivors_after": finite_mean(v.get("survivors_after") for v in vals),
                "fallback_rate": finite_mean(v.get("fallback_used") for v in vals),
            }
        )
    best_config_rows = sorted(
        [r for r in group_config_rows if r.get("dataset_name") in {"branch_generator_v1", "hidden_origin_quota_v4"}],
        key=lambda r: (
            str(r.get("dataset_name")),
            -safe_float(r.get("oracle_retention"), 0.0),
            -safe_float(r.get("pairwise_accuracy"), 0.0),
        ),
    )[:30]
    lines = [
        "# DualAnchor All-Loop Audit v1",
        "",
        f"BG_DUALANCHOR_ALL_LOOP_AUDIT_VERDICT = {verdict}",
        "",
        "This is the corrected audit for the transformer-loop interpretation. It applies layer-local taps at layers 24, 36, and 47 over loop states L1-L4 instead of using only `*_L4` cached configs.",
        "",
        "## Scope",
        "",
        "- layers: `24`, `36`, `47`",
        "- loops: `L1`, `L2`, `L3`, `L4`",
        "- audited configs: `24/36/47 L1-L4`, layer means, `47_concat_L1_L4`, `47_concat_all_loops`",
        "- terminal top1: final `47_L4` content/output scoring only",
        "- no true fork/carry, no generation, no steering, no routing change",
        "",
        "## Data Coverage",
        "",
    ]
    lines.extend(md_table(coverage, ["dataset_name", "group_count", "candidate_count", "full_loop", "domains"]))
    lines.extend(
        [
            "",
            "## Best Threshold Policy",
            "",
            f"- policy: `{best.get('policy')}`",
            f"- threshold policy: `{best.get('threshold_policy')}`",
            f"- pre-terminal oracle retention: `{best.get('pre_terminal_oracle_retained')}`",
            f"- pre-terminal false prune: `{best.get('pre_terminal_false_prune')}`",
            f"- avg pre-terminal survivors: `{best.get('pre_terminal_survivor_count')}`",
            f"- pre-terminal best reward: `{best.get('pre_terminal_best_reward')}`",
            f"- terminal selected reward: `{best.get('terminal_selected_reward')}`",
            f"- terminal oracle selected: `{best.get('terminal_oracle_selected')}`",
            "",
            "## Overall Threshold Policies",
            "",
        ]
    )
    lines.extend(
        md_table(
            overall_threshold,
            [
                "policy",
                "threshold_policy",
                "group_count",
                "pre_terminal_oracle_retained",
                "pre_terminal_false_prune",
                "pre_terminal_survivor_count",
                "pre_terminal_best_reward",
                "terminal_selected_reward",
                "terminal_oracle_selected",
            ],
        )
    )
    lines.extend(["", "## Selected Stage Breakdown", ""])
    lines.extend(md_table(stage_summary, ["loop", "layer", "stage_count", "oracle_retention_after_stage", "avg_survivors_before", "avg_survivors_after", "fallback_rate"]))
    lines.extend(["", "## Branch Config Highlights", ""])
    lines.extend(md_table(best_config_rows, ["dataset_name", "policy", "config", "pair_count", "pairwise_accuracy", "oracle_retention", "false_prune_rate", "avg_survivors", "best_selected_reward", "top1_reward"]))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The earlier L4-only loop reports should remain diagnostic only.",
            "- This run tests whether the same layer-local taps can score current loop states, which is the intended transformer placement.",
            "- Layer 24 is expected to be branch/viability heavy because perturbations are newest there; the report therefore emphasizes survival before terminal selection.",
            "- Final selection remains separated: only the final layer-47 step performs explicit top1 content/output selection.",
            "",
            "## Files",
            "",
            f"- artifact: `{rel(ARTIFACT_PT)}`",
            f"- report json: `{rel(REPORT_JSON)}`",
            f"- pair/config rows: `{rel(PAIR_ROWS_CSV)}`",
            f"- threshold group rows: `{rel(GROUP_ROWS_CSV)}`",
            f"- threshold stage rows: `{rel(STAGE_ROWS_CSV)}`",
        ]
    )
    write_md(REPORT_MD, lines)
    write_md(SUMMARY_MD, lines)
    print(f"BG_DUALANCHOR_ALL_LOOP_AUDIT_VERDICT = {verdict}", flush=True)
    print(f"selected_policy = {best.get('policy')} threshold = {best.get('threshold_policy')}", flush=True)
    print(
        "pre_terminal_retention = "
        f"{best.get('pre_terminal_oracle_retained')} false_prune = {best.get('pre_terminal_false_prune')} "
        f"terminal_reward = {best.get('terminal_selected_reward')}",
        flush=True,
    )
    print(f"report = {rel(REPORT_MD)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
