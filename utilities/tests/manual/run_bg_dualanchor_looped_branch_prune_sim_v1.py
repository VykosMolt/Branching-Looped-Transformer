"""DualAnchor looped branch-prune simulator v1.

This is a cached approximation of the intended recursive branch architecture:
24 -> 36 -> 47 repeated for up to four loop passes. Each active layer scores
the current survivor set with DualAnchor layer-local taps and prunes by a
dynamic score-band threshold. Layer 47 participates in every loop and is also
the terminal chooser when the loop budget is exhausted.

This script does not run true fork/carry, generation, action steering, Ouro
training, registry updates, wrapper/local-agent code, Hunter-Seeker modules, or
production routing changes.
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
import torch.nn.functional as F

from bg_hidden_origin_tap_common import PROJECT_ROOT, PROBE_ROOT, config_dim, md_table, rel
from bg_merged_tap_v1_common import BRIDGE_PT, json_default, safe_float, score_diff
from run_bg_layer_native_two_tap_readiness_v1 import (
    ANCHOR_GROUPS,
    LAYER_CONFIGS,
    candidate_feature,
    candidate_name,
    group_reward,
    tensor_weight,
    write_csv,
    write_json,
    write_md,
)


OUT_ROOT = PROBE_ROOT / "bg_dualanchor_looped_branch_prune_sim_v1_2026-05-31"
REPORT_JSON = OUT_ROOT / "looped_branch_prune_sim.json"
REPORT_MD = OUT_ROOT / "looped_branch_prune_sim.md"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
GROUP_ROWS_CSV = OUT_ROOT / "looped_group_rows.csv"
STAGE_ROWS_CSV = OUT_ROOT / "looped_stage_rows.csv"
POLICY_ROWS_CSV = OUT_ROOT / "loop_policy_rows.csv"
GROUP_INVENTORY_CSV = OUT_ROOT / "group_inventory.csv"
ARTIFACT_PT = OUT_ROOT / "dualanchor_looped_branch_prune_sim_v1.pt"

CONSTRAINED_PT = (
    PROBE_ROOT
    / "bg_layer_native_two_tap_constrained_train_v1_2026-05-30"
    / "layer_native_two_tap_constrained_train_v1.pt"
)
SURVIVOR_FEATURES_PT = (
    PROBE_ROOT
    / "bg_merged_weight_branch_content_taps_v1_2026-05-18"
    / "top4_survivor_hidden_features.pt"
)
BGV1_BRANCHES_PT = PROBE_ROOT / "bg_hidden_origin_branch_generator_v1_2026-05-18/branch_generator_v1_branches.pt"
V4_BRANCHES_PT = PROBE_ROOT / "bg_hidden_origin_quota_v4_2026-05-18/quota_hidden_origin_branches.pt"

LAYER_SEQUENCE = ("24_L4", "36_L4", "47_L4")
MAX_LOOPS = 4
DEFAULT_ARCH = "AntisymLinear"
POLICY_RECIPES = (
    "old_only",
    "adaptive_balanced_rescue",
    "adaptive_branch_anchor_light",
    "adaptive_branch_from_val",
)
THRESHOLD_SCHEDULES = {
    "dynamic_conservative": {
        "band_by_loop": [1.00, 0.80, 0.60, 0.45],
        "band_by_layer": {"24_L4": 1.10, "36_L4": 1.00, "47_L4": 0.90},
        "min_keep_by_loop": [4, 3, 2, 1],
        "max_keep_by_loop": [8, 8, 6, 4],
        "terminal_keep": 1,
    },
    "dynamic_moderate": {
        "band_by_loop": [0.75, 0.55, 0.35, 0.20],
        "band_by_layer": {"24_L4": 1.05, "36_L4": 1.00, "47_L4": 0.85},
        "min_keep_by_loop": [3, 2, 1, 1],
        "max_keep_by_loop": [6, 5, 4, 2],
        "terminal_keep": 1,
    },
    "dynamic_fast": {
        "band_by_loop": [0.50, 0.35, 0.20, 0.00],
        "band_by_layer": {"24_L4": 1.00, "36_L4": 0.90, "47_L4": 0.75},
        "min_keep_by_loop": [2, 1, 1, 1],
        "max_keep_by_loop": [4, 3, 2, 1],
        "terminal_keep": 1,
    },
}


def ensure_root() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def runtime_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clean_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k not in {"weight", "state_dict", "members"}}


def load_constrained() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = torch.load(CONSTRAINED_PT, map_location="cpu", weights_only=False)
    return [dict(c) for c in payload.get("candidates") or []], [dict(r) for r in payload.get("references") or []]


def score_metric(candidate: dict[str, Any], recipe: str) -> float:
    if recipe == "old_only":
        # Prefer clean old anchors when evaluating old-only.
        if candidate.get("recipe") == "old_only":
            return 1.0
        return -1.0
    if recipe == "adaptive_balanced_rescue":
        return safe_float(candidate.get("balanced_val_acc"), -1.0)
    if recipe in {"adaptive_branch_anchor_light", "adaptive_branch_from_val"}:
        return safe_float(candidate.get("branch_val_acc"), -1.0)
    return safe_float(candidate.get("balanced_val_acc"), -1.0)


def select_layer_taps(candidates: Sequence[dict[str, Any]], recipe: str, arch: str = DEFAULT_ARCH) -> dict[str, dict[str, dict[str, Any]]]:
    selected: dict[str, dict[str, dict[str, Any]]] = {layer: {} for layer in LAYER_SEQUENCE}
    for layer in LAYER_SEQUENCE:
        for role in ANCHOR_GROUPS:
            rows = [
                c
                for c in candidates
                if c.get("tap_role") == role
                and c.get("target_config") == layer
                and c.get("architecture") == arch
                and (c.get("recipe") == recipe or recipe == "old_only")
                and isinstance(tensor_weight(c), torch.Tensor)
            ]
            if not rows and arch != "AntisymLinearNoNorm":
                rows = [
                    c
                    for c in candidates
                    if c.get("tap_role") == role
                    and c.get("target_config") == layer
                    and c.get("architecture") == "AntisymLinearNoNorm"
                    and (c.get("recipe") == recipe or recipe == "old_only")
                    and isinstance(tensor_weight(c), torch.Tensor)
                ]
            if rows:
                selected[layer][role] = max(rows, key=lambda c: (score_metric(c, recipe), candidate_name(c)))
    return selected


def policy_inventory(candidates: Sequence[dict[str, Any]]) -> tuple[dict[str, dict[str, dict[str, dict[str, Any]]]], list[dict[str, Any]]]:
    policies: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    rows: list[dict[str, Any]] = []
    for recipe in POLICY_RECIPES:
        for arch in ("AntisymLinear", "AntisymLinearNoNorm"):
            name = f"dualanchor_{recipe}_{arch}"
            taps = select_layer_taps(candidates, recipe, arch)
            complete = all(role in taps[layer] for layer in LAYER_SEQUENCE for role in ANCHOR_GROUPS)
            if not complete:
                continue
            policies[name] = taps
            for layer in LAYER_SEQUENCE:
                for role in ANCHOR_GROUPS:
                    tap = taps[layer][role]
                    rows.append(
                        {
                            "policy": name,
                            "recipe": recipe,
                            "architecture": tap.get("architecture"),
                            "layer": layer,
                            "tap_role": role,
                            "candidate_name": candidate_name(tap),
                            "candidate_family": tap.get("candidate_family"),
                            "score_metric": score_metric(tap, recipe),
                            "old_val_acc": tap.get("old_val_acc"),
                            "branch_val_acc": tap.get("branch_val_acc"),
                            "balanced_val_acc": tap.get("balanced_val_acc"),
                        }
                    )
    return policies, rows


def feature_from_raw_row(row: dict[str, Any], config: str) -> torch.Tensor | None:
    fmap = row.get("features_by_config")
    if isinstance(fmap, dict) and isinstance(fmap.get(config), torch.Tensor):
        value = fmap[config]
        if int(value.numel()) == config_dim(config):
            return value.detach().cpu().to(torch.float32).flatten()
    pooled = row.get("pooled_vectors") or {}
    if isinstance(pooled, dict) and isinstance(pooled.get(config), torch.Tensor):
        return pooled[config].detach().cpu().to(torch.float32).flatten()
    features = row.get("features")
    if isinstance(features, torch.Tensor) and tuple(features.shape[:2]) == (3, 4):
        idx = {"24_L4": (0, 3), "36_L4": (1, 3), "47_L4": (2, 3)}.get(config)
        if idx is not None:
            return features[idx[0], idx[1]].detach().cpu().to(torch.float32).flatten()
    return None


def normalize_group_row(row: dict[str, Any], source: str) -> dict[str, Any] | None:
    fmap = {}
    for config in LAYER_SEQUENCE:
        vec = feature_from_raw_row(row, config)
        if isinstance(vec, torch.Tensor) and int(vec.numel()) == config_dim(config):
            fmap[config] = vec
    if len(fmap) < len(LAYER_SEQUENCE):
        return None
    reward = row.get("reward")
    if reward is None:
        reward = row.get("deterministic_reward")
    if reward is None:
        reward = 1.0 if row.get("correct") or row.get("correctness") else 0.0
    group_id = row.get("group_id") or row.get("branch_group_id") or row.get("survivor_set_id")
    if group_id is None:
        return None
    return {
        **row,
        "features_by_config": fmap,
        "reward": safe_float(reward, 0.0),
        "group_id": str(group_id),
        "branch_group_id": row.get("branch_group_id") or group_id,
        "domain": row.get("domain") or "unknown",
        "split": row.get("split") or row.get("v1_split") or row.get("original_split") or "all",
        "source_dataset": source,
    }


def groups_from_rows(rows: Sequence[dict[str, Any]], source: str, *, split: str | None = None) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if split and str(row.get("split") or row.get("v1_split") or row.get("original_split") or "all") != split:
            continue
        norm = normalize_group_row(row, source)
        if norm is None:
            continue
        grouped[norm["group_id"]].append(norm)
    return [vals for vals in grouped.values() if len(vals) >= 2 and max(group_reward(v) for v in vals) > min(group_reward(v) for v in vals)]


def load_group_sources() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    datasets: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []

    if BRIDGE_PT.exists():
        payload = torch.load(BRIDGE_PT, map_location="cpu", weights_only=False)
        groups = groups_from_rows(payload.get("candidate_rows") or [], "universal_bridge_candidate_groups", split="heldout")
        datasets.append({"dataset_name": "universal_bridge_candidate_groups", "groups": groups, "readiness_eligible": True})

    if SURVIVOR_FEATURES_PT.exists():
        payload = torch.load(SURVIVOR_FEATURES_PT, map_location="cpu", weights_only=False)
        rows = []
        for survivor_set in payload.get("survivor_sets") or []:
            if str(survivor_set.get("split") or survivor_set.get("v1_split") or "") != "fresh_holdout":
                continue
            for cand in survivor_set.get("candidates") or []:
                row = dict(cand)
                row["group_id"] = survivor_set.get("survivor_set_id")
                row["domain"] = survivor_set.get("domain")
                row["split"] = survivor_set.get("split") or survivor_set.get("v1_split")
                rows.append(row)
        groups = groups_from_rows(rows, "top4_survivor_hidden_feature_groups")
        datasets.append({"dataset_name": "top4_survivor_hidden_feature_groups", "groups": groups, "readiness_eligible": True})

    for path, name in ((BGV1_BRANCHES_PT, "bgv1_raw_branch_groups"), (V4_BRANCHES_PT, "v4_raw_branch_groups")):
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        groups = groups_from_rows(payload.get("rows") or [], name, split="heldout")
        datasets.append({"dataset_name": name, "groups": groups, "readiness_eligible": True})

    for dataset in datasets:
        groups = dataset["groups"]
        sizes = [len(g) for g in groups]
        rewards = [max(group_reward(r) for r in g) - min(group_reward(r) for r in g) for g in groups]
        inventory.append(
            {
                "dataset_name": dataset["dataset_name"],
                "group_count": len(groups),
                "candidate_count": sum(sizes),
                "mean_group_size": mean(sizes) if sizes else float("nan"),
                "reward_diverse_groups": sum(1 for r in rewards if r > 0),
                "domains": dict(Counter(str(g[0].get("domain")) for g in groups)),
            }
        )
    return datasets, inventory


def normalize_scores(vals: Sequence[float]) -> list[float]:
    xs = [safe_float(v, 0.0) for v in vals]
    if not xs:
        return []
    mu = mean(xs)
    var = mean((x - mu) ** 2 for x in xs)
    sd = math.sqrt(var)
    if sd < 1e-8:
        return [0.0 for _ in xs]
    return [(x - mu) / sd for x in xs]


def tap_group_scores(group: Sequence[dict[str, Any]], tap: dict[str, Any], config: str, device: torch.device) -> list[float]:
    weight = tensor_weight(tap)
    if not isinstance(weight, torch.Tensor):
        return []
    arch = str(tap.get("architecture") or "")
    vecs = []
    for row in group:
        vec = feature_from_raw_row(row, config)
        if not isinstance(vec, torch.Tensor):
            return []
        vecs.append(vec)
    x = torch.stack([v.to(torch.float32).flatten() for v in vecs], dim=0).to(device)
    w = weight.detach().cpu().to(torch.float32).flatten().to(device)
    scores = []
    with torch.no_grad():
        for i in range(x.shape[0]):
            diff = x[i].unsqueeze(0) - x
            mask = torch.ones(x.shape[0], dtype=torch.bool, device=device)
            mask[i] = False
            d = diff[mask]
            if arch == "AntisymLinear":
                d = F.layer_norm(d, (d.shape[1],))
            s = d @ w
            scores.append(float(s.mean().detach().cpu().item()) if int(s.numel()) else 0.0)
    return scores


def layer_scores(group: Sequence[dict[str, Any]], taps: dict[str, dict[str, dict[str, Any]]], config: str, device: torch.device) -> list[float]:
    components = []
    for role in ANCHOR_GROUPS:
        tap = taps.get(config, {}).get(role)
        if tap is None:
            continue
        scores = tap_group_scores(group, tap, config, device)
        if scores:
            components.append(normalize_scores(scores))
    if not components:
        return []
    return [mean(scores[i] for scores in components) for i in range(len(group))]


def prune_indices(scores: Sequence[float], loop_idx: int, layer: str, schedule: dict[str, Any], terminal: bool = False) -> list[int]:
    n = len(scores)
    if n <= 1:
        return list(range(n))
    ordered = [idx for idx, _ in sorted(enumerate(scores), key=lambda item: (-safe_float(item[1], -1e9), item[0]))]
    if terminal:
        return ordered[: int(schedule.get("terminal_keep", 1))]
    vals = [safe_float(s, 0.0) for s in scores]
    max_score = vals[ordered[0]]
    sd = math.sqrt(mean((v - mean(vals)) ** 2 for v in vals)) if len(vals) > 1 else 0.0
    band = float(schedule["band_by_loop"][loop_idx]) * float(schedule["band_by_layer"].get(layer, 1.0)) * max(sd, 1e-6)
    keep = [idx for idx in ordered if vals[idx] >= max_score - band]
    min_keep = min(int(schedule["min_keep_by_loop"][loop_idx]), n)
    max_keep = min(int(schedule["max_keep_by_loop"][loop_idx]), n)
    if len(keep) < min_keep:
        keep = ordered[:min_keep]
    if len(keep) > max_keep:
        keep = ordered[:max_keep]
    return keep


def simulate_group(
    group: Sequence[dict[str, Any]],
    taps: dict[str, dict[str, dict[str, Any]]],
    schedule: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    survivors = list(range(len(group)))
    rewards = [group_reward(row) for row in group]
    oracle_reward = max(rewards)
    oracle_indices = {i for i, r in enumerate(rewards) if r == oracle_reward}
    stage_rows: list[dict[str, Any]] = []
    first_oracle_loss: dict[str, Any] | None = None
    for loop_idx in range(MAX_LOOPS):
        for layer in LAYER_SEQUENCE:
            if len(survivors) <= 1:
                break
            sub_group = [group[i] for i in survivors]
            scores = layer_scores(sub_group, taps, layer, device)
            if not scores:
                continue
            local_keep = prune_indices(scores, loop_idx, layer, schedule, terminal=False)
            before = list(survivors)
            survivors = [survivors[i] for i in local_keep]
            kept_oracle = bool(set(survivors) & oracle_indices)
            if not kept_oracle and first_oracle_loss is None:
                first_oracle_loss = {"loop": loop_idx + 1, "layer": layer}
            stage_rows.append(
                {
                    "loop": loop_idx + 1,
                    "layer": layer,
                    "survivors_before": len(before),
                    "survivors_after": len(survivors),
                    "oracle_retained_after_stage": 1.0 if kept_oracle else 0.0,
                    "score_mean": mean(scores) if scores else float("nan"),
                    "score_std": math.sqrt(mean((s - mean(scores)) ** 2 for s in scores)) if len(scores) > 1 else 0.0,
                    "selected_indices": list(survivors),
                }
            )
        if len(survivors) <= 1:
            break
    if len(survivors) > 1:
        sub_group = [group[i] for i in survivors]
        scores = layer_scores(sub_group, taps, "47_L4", device)
        if scores:
            local_keep = prune_indices(scores, MAX_LOOPS - 1, "47_L4", schedule, terminal=True)
            survivors = [survivors[i] for i in local_keep]
    selected = survivors[0] if survivors else 0
    best_survivor_reward = max(rewards[i] for i in survivors) if survivors else float("nan")
    row = {
        "group_size": len(group),
        "final_survivor_count": len(survivors),
        "oracle_reward": oracle_reward,
        "oracle_retained_final": 1.0 if set(survivors) & oracle_indices else 0.0,
        "false_prune_rate": 0.0 if set(survivors) & oracle_indices else 1.0,
        "best_survivor_reward": best_survivor_reward,
        "final_selected_reward": rewards[selected],
        "top1_success": 1.0 if selected in oracle_indices and oracle_reward > 0 else 0.0,
        "regret": oracle_reward - rewards[selected],
        "best_survivor_regret": oracle_reward - best_survivor_reward,
        "stages_run": len(stage_rows),
        "first_oracle_loss_loop": None if first_oracle_loss is None else first_oracle_loss["loop"],
        "first_oracle_loss_layer": None if first_oracle_loss is None else first_oracle_loss["layer"],
        "selected_index": selected,
        "selected_indices": list(survivors),
    }
    return row, stage_rows


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"group_count": 0}
    out = {
        "group_count": len(rows),
        "oracle_retention": mean(safe_float(r.get("oracle_retained_final"), 0.0) for r in rows),
        "false_prune_rate": mean(safe_float(r.get("false_prune_rate"), 1.0) for r in rows),
        "avg_final_survivors": mean(safe_float(r.get("final_survivor_count"), 0.0) for r in rows),
        "final_selected_reward": mean(safe_float(r.get("final_selected_reward"), 0.0) for r in rows),
        "best_survivor_reward": mean(safe_float(r.get("best_survivor_reward"), 0.0) for r in rows),
        "top1_success": mean(safe_float(r.get("top1_success"), 0.0) for r in rows),
        "regret": mean(safe_float(r.get("regret"), 0.0) for r in rows),
        "best_survivor_regret": mean(safe_float(r.get("best_survivor_regret"), 0.0) for r in rows),
        "avg_stages_run": mean(safe_float(r.get("stages_run"), 0.0) for r in rows),
    }
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row.get("domain") or "unknown")].append(row)
    out["domain_oracle_retention"] = {d: mean(safe_float(r.get("oracle_retained_final"), 0.0) for r in vals) for d, vals in sorted(by_domain.items())}
    out["domain_final_reward"] = {d: mean(safe_float(r.get("final_selected_reward"), 0.0) for r in vals) for d, vals in sorted(by_domain.items())}
    return out


def main() -> int:
    ensure_root()
    device = runtime_device()
    print(f"DualAnchor looped branch-prune sim device = {device}", flush=True)
    candidates, _refs = load_constrained()
    policies, policy_rows = policy_inventory(candidates)
    datasets, inventory = load_group_sources()
    print(f"policies={len(policies)} datasets={len(datasets)} groups={sum(len(d['groups']) for d in datasets)}", flush=True)

    group_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    summary_by_policy: dict[str, Any] = {}
    for policy_name, taps in policies.items():
        for schedule_name, schedule in THRESHOLD_SCHEDULES.items():
            all_rows = []
            for dataset in datasets:
                dataset_rows = []
                for group in dataset["groups"]:
                    row, stages = simulate_group(group, taps, schedule, device)
                    row.update(
                        {
                            "dataset_name": dataset["dataset_name"],
                            "group_id": group[0].get("group_id"),
                            "domain": group[0].get("domain"),
                            "split": group[0].get("split"),
                            "policy": policy_name,
                            "schedule": schedule_name,
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
                                "schedule": schedule_name,
                            }
                        )
                key = f"{policy_name}::{schedule_name}::{dataset['dataset_name']}"
                summary_by_policy[key] = summarize(dataset_rows)
                summary_by_policy[key]["policy"] = policy_name
                summary_by_policy[key]["schedule"] = schedule_name
                summary_by_policy[key]["dataset_name"] = dataset["dataset_name"]
                all_rows.extend(dataset_rows)
            key = f"{policy_name}::{schedule_name}::ALL"
            summary_by_policy[key] = summarize(all_rows)
            summary_by_policy[key]["policy"] = policy_name
            summary_by_policy[key]["schedule"] = schedule_name
            summary_by_policy[key]["dataset_name"] = "ALL"

    all_summaries = list(summary_by_policy.values())
    eligible = [s for s in all_summaries if s.get("dataset_name") == "ALL" and int(s.get("group_count") or 0) > 0]
    best = max(
        eligible,
        key=lambda s: (
            safe_float(s.get("oracle_retention"), -1.0),
            -safe_float(s.get("false_prune_rate"), 1.0),
            safe_float(s.get("final_selected_reward"), -1.0),
            -safe_float(s.get("avg_final_survivors"), 99.0),
        ),
        default={},
    )
    if best and safe_float(best.get("oracle_retention"), 0.0) >= 0.95 and safe_float(best.get("false_prune_rate"), 1.0) <= 0.05:
        verdict = "DUALANCHOR_LOOP_SURVIVAL_READY"
    elif best and safe_float(best.get("oracle_retention"), 0.0) >= 0.90:
        verdict = "DUALANCHOR_LOOP_SURVIVAL_PARTIAL"
    elif best:
        verdict = "DUALANCHOR_LOOP_SURVIVAL_WEAK"
    else:
        verdict = "DATA_LIMITED"

    payload = {
        "BG_DUALANCHOR_LOOPED_BRANCH_PRUNE_SIM_VERDICT": verdict,
        "status": verdict,
        "mode": "CACHED_LOOP_APPROX",
        "true_fork_carry_claimed": False,
        "action_steering_claimed": False,
        "runtime_device": str(device),
        "max_loops": MAX_LOOPS,
        "layer_sequence": list(LAYER_SEQUENCE),
        "policy_count": len(policies),
        "threshold_schedules": THRESHOLD_SCHEDULES,
        "group_inventory": inventory,
        "selected_policy_summary": best,
        "summary_by_policy": summary_by_policy,
        "anti_leakage": {
            "cached_candidate_groups_only": True,
            "no_true_fork_carry": True,
            "no_action_steering": True,
            "no_ouro_training": True,
            "no_production_routing_change": True,
        },
    }
    torch.save(
        {
            "summary": payload,
            "policy_rows": policy_rows,
            "group_rows": group_rows,
            "stage_rows": stage_rows,
        },
        ARTIFACT_PT,
    )
    write_json(REPORT_JSON, payload)
    write_json(SUMMARY_JSON, payload)
    write_csv(POLICY_ROWS_CSV, policy_rows)
    write_csv(GROUP_ROWS_CSV, group_rows)
    write_csv(STAGE_ROWS_CSV, stage_rows)
    write_csv(GROUP_INVENTORY_CSV, inventory)

    table_rows = sorted(
        [s for s in all_summaries if s.get("dataset_name") == "ALL"],
        key=lambda s: (
            -safe_float(s.get("oracle_retention"), -1.0),
            safe_float(s.get("false_prune_rate"), 1.0),
            -safe_float(s.get("final_selected_reward"), -1.0),
        ),
    )
    dataset_rows = sorted(
        [s for s in all_summaries if s.get("dataset_name") != "ALL"],
        key=lambda s: (str(s.get("dataset_name")), str(s.get("policy")), str(s.get("schedule"))),
    )
    lines = [
        "# DualAnchor Looped Branch-Prune Simulator v1",
        "",
        f"BG_DUALANCHOR_LOOPED_BRANCH_PRUNE_SIM_VERDICT = {verdict}",
        "",
        "Mode: `CACHED_LOOP_APPROX`. This simulates 24 -> 36 -> 47 pruning for up to four loop passes over cached candidate groups. It does not claim true fork/carry or action steering.",
        "",
        f"- runtime device: `{device}`",
        f"- policies: `{len(policies)}`",
        f"- datasets: `{len(datasets)}`",
        f"- groups: `{sum(len(d['groups']) for d in datasets)}`",
        f"- selected policy: `{best.get('policy')}`",
        f"- selected schedule: `{best.get('schedule')}`",
        f"- selected oracle retention: `{best.get('oracle_retention')}`",
        f"- selected false prune: `{best.get('false_prune_rate')}`",
        f"- selected final reward: `{best.get('final_selected_reward')}`",
        "",
        "## Group Inventory",
        "",
    ]
    lines.extend(md_table(inventory, ["dataset_name", "group_count", "candidate_count", "mean_group_size", "reward_diverse_groups", "domains"]))
    lines.extend(["", "## Overall Policies", ""])
    lines.extend(md_table(table_rows, ["policy", "schedule", "group_count", "oracle_retention", "false_prune_rate", "avg_final_survivors", "final_selected_reward", "best_survivor_reward", "top1_success", "regret", "avg_stages_run"]))
    lines.extend(["", "## Dataset Breakdown", ""])
    lines.extend(md_table(dataset_rows, ["dataset_name", "policy", "schedule", "group_count", "oracle_retention", "false_prune_rate", "avg_final_survivors", "final_selected_reward", "best_survivor_reward", "top1_success", "regret"]))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Layer 47 participates in every loop pass and is also used for terminal selection.",
            "- Dynamic pruning is score-band based with loop/layer-dependent thresholds and max/min survivor bounds.",
            "- This is an approximation over already-materialized candidate sets; it cannot prove compute savings or true recurrent fork/carry.",
            "",
            "## Files",
            "",
            f"- artifact: `{rel(ARTIFACT_PT)}`",
            f"- report json: `{rel(REPORT_JSON)}`",
            f"- group rows: `{rel(GROUP_ROWS_CSV)}`",
            f"- stage rows: `{rel(STAGE_ROWS_CSV)}`",
            f"- policy rows: `{rel(POLICY_ROWS_CSV)}`",
        ]
    )
    write_md(REPORT_MD, lines)
    write_md(SUMMARY_MD, lines)
    print(f"BG_DUALANCHOR_LOOPED_BRANCH_PRUNE_SIM_VERDICT = {verdict}", flush=True)
    print(f"selected_policy = {best.get('policy')} schedule = {best.get('schedule')}", flush=True)
    print(f"oracle_retention = {best.get('oracle_retention')} false_prune = {best.get('false_prune_rate')}", flush=True)
    print(f"report = {rel(REPORT_MD)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
