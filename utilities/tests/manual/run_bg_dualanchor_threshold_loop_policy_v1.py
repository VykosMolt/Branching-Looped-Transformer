"""DualAnchor dynamic-threshold loop policy probe v1.

This is a cached approximation of the intended in-transformer loop:
24_L4 -> 36_L4 -> 47_L4, repeated up to four loop passes. Every non-terminal
layer application is a dynamic threshold filter: all candidates above the
threshold continue downward, and candidates below it are killed. The only
explicit top1 choice is the terminal 47_L4 decision after the loop budget, and
that choice is still made by pairwise tap response among the surviving
perturbations.

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
from typing import Any, Sequence

import torch

from bg_hidden_origin_tap_common import PROBE_ROOT, md_table, rel
from bg_merged_tap_v1_common import json_default, safe_float
from run_bg_layer_native_two_tap_readiness_v1 import write_csv, write_json, write_md

import run_bg_dualanchor_looped_branch_prune_sim_v1 as base


OUT_ROOT = PROBE_ROOT / "bg_dualanchor_threshold_loop_policy_v1_2026-05-31"
REPORT_JSON = OUT_ROOT / "threshold_loop_policy.json"
REPORT_MD = OUT_ROOT / "threshold_loop_policy.md"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
GROUP_ROWS_CSV = OUT_ROOT / "threshold_group_rows.csv"
STAGE_ROWS_CSV = OUT_ROOT / "threshold_stage_rows.csv"
POLICY_ROWS_CSV = OUT_ROOT / "threshold_policy_rows.csv"
ARTIFACT_PT = OUT_ROOT / "dualanchor_threshold_loop_policy_v1.pt"

TERMINAL_LAYER = "47_L4"

THRESHOLD_POLICIES: dict[str, dict[str, Any]] = {
    "maxband_very_loose": {
        "kind": "max_band",
        "band_by_loop": [3.20, 2.80, 2.40, 2.00],
        "band_by_layer": {"24_L4": 1.35, "36_L4": 1.10, "47_L4": 1.00},
    },
    "maxband_conservative": {
        "kind": "max_band",
        "band_by_loop": [1.80, 1.55, 1.30, 1.10],
        "band_by_layer": {"24_L4": 1.25, "36_L4": 1.05, "47_L4": 0.95},
    },
    "maxband_balanced": {
        "kind": "max_band",
        "band_by_loop": [1.45, 1.20, 0.95, 0.75],
        "band_by_layer": {"24_L4": 1.20, "36_L4": 1.00, "47_L4": 0.90},
    },
    "maxband_tight": {
        "kind": "max_band",
        "band_by_loop": [1.15, 0.90, 0.70, 0.50],
        "band_by_layer": {"24_L4": 1.10, "36_L4": 1.00, "47_L4": 0.85},
    },
    "mean_floor_loose": {
        "kind": "mean_floor",
        "offset_by_loop": [-0.75, -0.55, -0.35, -0.20],
        "offset_by_layer": {"24_L4": -0.10, "36_L4": 0.00, "47_L4": 0.10},
    },
    "mean_floor_very_loose": {
        "kind": "mean_floor",
        "offset_by_loop": [-2.00, -1.75, -1.50, -1.25],
        "offset_by_layer": {"24_L4": -0.20, "36_L4": -0.05, "47_L4": 0.00},
    },
    "mean_floor_loose_carry": {
        "kind": "mean_floor",
        "offset_by_loop": [-1.50, -1.25, -1.00, -0.75],
        "offset_by_layer": {"24_L4": -0.15, "36_L4": -0.05, "47_L4": 0.00},
    },
    "mean_floor_balanced": {
        "kind": "mean_floor",
        "offset_by_loop": [-0.35, -0.15, 0.00, 0.15],
        "offset_by_layer": {"24_L4": -0.05, "36_L4": 0.00, "47_L4": 0.10},
    },
    "dispersion_very_loose": {
        "kind": "dispersion_adaptive",
        "wide_band": [3.00, 2.60, 2.20, 1.80],
        "narrow_offset": [-1.80, -1.50, -1.20, -0.90],
        "low_dispersion_std": 0.35,
        "band_by_layer": {"24_L4": 1.35, "36_L4": 1.10, "47_L4": 1.00},
    },
    "dispersion_adaptive": {
        "kind": "dispersion_adaptive",
        "wide_band": [1.70, 1.45, 1.20, 1.00],
        "narrow_offset": [-0.60, -0.40, -0.20, 0.00],
        "low_dispersion_std": 0.35,
        "band_by_layer": {"24_L4": 1.25, "36_L4": 1.00, "47_L4": 0.90},
    },
}


def ensure_root() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def score_stats(scores: Sequence[float]) -> tuple[float, float, float, float]:
    vals = [safe_float(s, 0.0) for s in scores]
    mu = mean(vals) if vals else 0.0
    sd = math.sqrt(mean((v - mu) ** 2 for v in vals)) if len(vals) > 1 else 0.0
    return mu, sd, max(vals) if vals else 0.0, min(vals) if vals else 0.0


def threshold_for(scores: Sequence[float], loop_idx: int, layer: str, spec: dict[str, Any]) -> float:
    mu, sd, max_score, _min_score = score_stats(scores)
    kind = str(spec.get("kind"))
    if kind == "max_band":
        band = float(spec["band_by_loop"][loop_idx]) * float(spec["band_by_layer"].get(layer, 1.0))
        return max_score - band * max(sd, 1e-6)
    if kind == "mean_floor":
        offset = float(spec["offset_by_loop"][loop_idx]) + float(spec["offset_by_layer"].get(layer, 0.0))
        return mu + offset * max(sd, 1e-6)
    if kind == "dispersion_adaptive":
        if sd < float(spec.get("low_dispersion_std", 0.35)):
            offset = float(spec["narrow_offset"][loop_idx])
            return mu + offset * max(sd, 1e-6)
        band = float(spec["wide_band"][loop_idx]) * float(spec["band_by_layer"].get(layer, 1.0))
        return max_score - band * max(sd, 1e-6)
    raise ValueError(f"unknown threshold policy kind: {kind}")


def threshold_keep_indices(scores: Sequence[float], threshold: float) -> tuple[list[int], bool]:
    vals = [safe_float(s, -1e9) for s in scores]
    keep = [idx for idx, value in enumerate(vals) if value >= threshold]
    if keep:
        return keep, False
    if not vals:
        return [], True
    # Safety fallback only: the threshold policy should never intentionally make
    # an empty survivor set. This is reported separately.
    best = max(range(len(vals)), key=lambda idx: (vals[idx], -idx))
    return [best], True


def terminal_pairwise_top1(
    group: Sequence[dict[str, Any]],
    survivors: Sequence[int],
    taps: dict[str, dict[str, dict[str, Any]]],
    device: torch.device,
) -> tuple[int, list[float]]:
    if len(survivors) <= 1:
        return (survivors[0] if survivors else 0), []
    sub_group = [group[i] for i in survivors]
    scores = base.layer_scores(sub_group, taps, TERMINAL_LAYER, device)
    if not scores:
        return survivors[0], []
    local_best = max(range(len(scores)), key=lambda idx: (safe_float(scores[idx], -1e9), -idx))
    return survivors[local_best], scores


def simulate_group_threshold(
    group: Sequence[dict[str, Any]],
    taps: dict[str, dict[str, dict[str, Any]]],
    threshold_spec: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    survivors = list(range(len(group)))
    rewards = [base.group_reward(row) for row in group]
    oracle_reward = max(rewards)
    oracle_indices = {idx for idx, value in enumerate(rewards) if value == oracle_reward}
    stage_rows: list[dict[str, Any]] = []
    first_oracle_loss: dict[str, Any] | None = None
    fallback_count = 0
    natural_singleton_loop = None
    natural_singleton_layer = None

    for loop_idx in range(base.MAX_LOOPS):
        for layer in base.LAYER_SEQUENCE:
            if len(survivors) <= 1:
                if natural_singleton_loop is None:
                    natural_singleton_loop = loop_idx + 1
                    natural_singleton_layer = layer
                break
            sub_group = [group[i] for i in survivors]
            scores = base.layer_scores(sub_group, taps, layer, device)
            if not scores:
                continue
            threshold = threshold_for(scores, loop_idx, layer, threshold_spec)
            local_keep, used_fallback = threshold_keep_indices(scores, threshold)
            fallback_count += 1 if used_fallback else 0
            before = list(survivors)
            survivors = [survivors[i] for i in local_keep]
            kept_oracle = bool(set(survivors) & oracle_indices)
            if not kept_oracle and first_oracle_loss is None:
                first_oracle_loss = {"loop": loop_idx + 1, "layer": layer}
            mu, sd, max_score, min_score = score_stats(scores)
            stage_rows.append(
                {
                    "loop": loop_idx + 1,
                    "layer": layer,
                    "survivors_before": len(before),
                    "survivors_after": len(survivors),
                    "threshold": threshold,
                    "score_mean": mu,
                    "score_std": sd,
                    "score_max": max_score,
                    "score_min": min_score,
                    "threshold_fallback_used": 1.0 if used_fallback else 0.0,
                    "oracle_retained_after_stage": 1.0 if kept_oracle else 0.0,
                    "selected_indices": list(survivors),
                }
            )

    pre_terminal_survivors = list(survivors)
    pre_terminal_oracle_retained = bool(set(pre_terminal_survivors) & oracle_indices)
    pre_terminal_best_reward = max(rewards[i] for i in pre_terminal_survivors) if pre_terminal_survivors else float("nan")
    selected_idx, terminal_scores = terminal_pairwise_top1(group, pre_terminal_survivors, taps, device)
    selected_reward = rewards[selected_idx] if selected_idx < len(rewards) else float("nan")
    terminal_oracle_selected = selected_idx in oracle_indices
    row = {
        "group_size": len(group),
        "pre_terminal_survivor_count": len(pre_terminal_survivors),
        "terminal_final_survivor_count": 1 if pre_terminal_survivors else 0,
        "oracle_reward": oracle_reward,
        "pre_terminal_oracle_retained": 1.0 if pre_terminal_oracle_retained else 0.0,
        "pre_terminal_false_prune": 0.0 if pre_terminal_oracle_retained else 1.0,
        "pre_terminal_best_reward": pre_terminal_best_reward,
        "terminal_selected_reward": selected_reward,
        "terminal_oracle_selected": 1.0 if terminal_oracle_selected else 0.0,
        "terminal_regret": oracle_reward - selected_reward,
        "best_survivor_regret": oracle_reward - pre_terminal_best_reward,
        "stages_run": len(stage_rows),
        "threshold_fallback_count": fallback_count,
        "natural_singleton_before_terminal": 1.0 if len(pre_terminal_survivors) == 1 else 0.0,
        "natural_singleton_loop": natural_singleton_loop,
        "natural_singleton_layer": natural_singleton_layer,
        "first_oracle_loss_loop": None if first_oracle_loss is None else first_oracle_loss["loop"],
        "first_oracle_loss_layer": None if first_oracle_loss is None else first_oracle_loss["layer"],
        "terminal_selected_index": selected_idx,
        "pre_terminal_selected_indices": list(pre_terminal_survivors),
        "terminal_scores": terminal_scores,
    }
    return row, stage_rows


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"group_count": 0}
    out = {
        "group_count": len(rows),
        "pre_terminal_oracle_retention": mean(safe_float(r.get("pre_terminal_oracle_retained"), 0.0) for r in rows),
        "pre_terminal_false_prune_rate": mean(safe_float(r.get("pre_terminal_false_prune"), 1.0) for r in rows),
        "avg_pre_terminal_survivors": mean(safe_float(r.get("pre_terminal_survivor_count"), 0.0) for r in rows),
        "pre_terminal_best_reward": mean(safe_float(r.get("pre_terminal_best_reward"), 0.0) for r in rows),
        "terminal_selected_reward": mean(safe_float(r.get("terminal_selected_reward"), 0.0) for r in rows),
        "terminal_oracle_selected_rate": mean(safe_float(r.get("terminal_oracle_selected"), 0.0) for r in rows),
        "terminal_regret": mean(safe_float(r.get("terminal_regret"), 0.0) for r in rows),
        "best_survivor_regret": mean(safe_float(r.get("best_survivor_regret"), 0.0) for r in rows),
        "avg_stages_run": mean(safe_float(r.get("stages_run"), 0.0) for r in rows),
        "threshold_fallback_rate": mean(1.0 if safe_float(r.get("threshold_fallback_count"), 0.0) > 0 else 0.0 for r in rows),
        "natural_singleton_before_terminal_rate": mean(safe_float(r.get("natural_singleton_before_terminal"), 0.0) for r in rows),
    }
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row.get("domain") or "unknown")].append(row)
    out["domain_pre_terminal_retention"] = {
        d: mean(safe_float(r.get("pre_terminal_oracle_retained"), 0.0) for r in vals)
        for d, vals in sorted(by_domain.items())
    }
    out["domain_terminal_reward"] = {
        d: mean(safe_float(r.get("terminal_selected_reward"), 0.0) for r in vals)
        for d, vals in sorted(by_domain.items())
    }
    return out


def verdict_for(best: dict[str, Any]) -> str:
    retention = safe_float(best.get("pre_terminal_oracle_retention"), 0.0)
    false_prune = safe_float(best.get("pre_terminal_false_prune_rate"), 1.0)
    terminal_reward = safe_float(best.get("terminal_selected_reward"), 0.0)
    if retention >= 0.93 and false_prune <= 0.07 and terminal_reward >= 0.75:
        return "THRESHOLD_LOOP_READY"
    if retention >= 0.90 and false_prune <= 0.10:
        return "THRESHOLD_SURVIVAL_READY_TERMINAL_SELECTION_WEAK"
    if retention >= 0.85 and false_prune <= 0.15:
        return "THRESHOLD_SURVIVAL_PARTIAL"
    return "THRESHOLD_SURVIVAL_WEAK"


def main() -> int:
    ensure_root()
    device = base.runtime_device()
    print(f"DualAnchor threshold loop policy device = {device}", flush=True)
    candidates, _refs = base.load_constrained()
    policies, policy_rows = base.policy_inventory(candidates)
    datasets, inventory = base.load_group_sources()
    print(
        f"policies={len(policies)} threshold_policies={len(THRESHOLD_POLICIES)} "
        f"datasets={len(datasets)} groups={sum(len(d['groups']) for d in datasets)}",
        flush=True,
    )

    group_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for policy_name, taps in policies.items():
        for threshold_name, threshold_spec in THRESHOLD_POLICIES.items():
            all_rows: list[dict[str, Any]] = []
            for dataset in datasets:
                dataset_rows: list[dict[str, Any]] = []
                for group in dataset["groups"]:
                    row, stages = simulate_group_threshold(group, taps, threshold_spec, device)
                    row.update(
                        {
                            "dataset_name": dataset["dataset_name"],
                            "group_id": group[0].get("group_id"),
                            "domain": group[0].get("domain"),
                            "split": group[0].get("split"),
                            "policy": policy_name,
                            "threshold_policy": threshold_name,
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
                sm = summarize(dataset_rows)
                sm.update({"dataset_name": dataset["dataset_name"], "policy": policy_name, "threshold_policy": threshold_name})
                summary_rows.append(sm)
                all_rows.extend(dataset_rows)
            sm = summarize(all_rows)
            sm.update({"dataset_name": "ALL", "policy": policy_name, "threshold_policy": threshold_name})
            summary_rows.append(sm)

    overall_rows = [r for r in summary_rows if r.get("dataset_name") == "ALL" and int(r.get("group_count") or 0) > 0]
    overall_rows.sort(
        key=lambda r: (
            -safe_float(r.get("pre_terminal_oracle_retention"), 0.0),
            safe_float(r.get("pre_terminal_false_prune_rate"), 1.0),
            -safe_float(r.get("pre_terminal_best_reward"), 0.0),
            -safe_float(r.get("terminal_selected_reward"), 0.0),
            safe_float(r.get("avg_pre_terminal_survivors"), 99.0),
        )
    )
    best = overall_rows[0] if overall_rows else {}
    verdict = verdict_for(best) if best else "DATA_LIMITED"

    dataset_rows = [r for r in summary_rows if r.get("dataset_name") != "ALL"]
    selected_dataset_rows = [
        r
        for r in dataset_rows
        if r.get("policy") == best.get("policy") and r.get("threshold_policy") == best.get("threshold_policy")
    ]
    stage_selected = [
        r
        for r in stage_rows
        if r.get("policy") == best.get("policy") and r.get("threshold_policy") == best.get("threshold_policy")
    ]
    stage_summary = []
    by_stage: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in stage_selected:
        by_stage[(int(row["loop"]), str(row["layer"]))].append(row)
    for (loop, layer), vals in sorted(by_stage.items()):
        stage_summary.append(
            {
                "loop": loop,
                "layer": layer,
                "stage_count": len(vals),
                "oracle_retention_after_stage": mean(safe_float(v.get("oracle_retained_after_stage"), 0.0) for v in vals),
                "avg_survivors_before": mean(safe_float(v.get("survivors_before"), 0.0) for v in vals),
                "avg_survivors_after": mean(safe_float(v.get("survivors_after"), 0.0) for v in vals),
                "avg_threshold": mean(safe_float(v.get("threshold"), 0.0) for v in vals),
                "fallback_rate": mean(safe_float(v.get("threshold_fallback_used"), 0.0) for v in vals),
            }
        )

    payload = {
        "BG_DUALANCHOR_THRESHOLD_LOOP_POLICY_VERDICT": verdict,
        "status": verdict,
        "mode": "CACHED_THRESHOLD_LOOP_APPROX",
        "true_fork_carry_claimed": False,
        "action_steering_claimed": False,
        "explicit_top1_only_at_terminal_47_L4": True,
        "runtime_device": str(device),
        "max_loops": base.MAX_LOOPS,
        "layer_sequence": list(base.LAYER_SEQUENCE),
        "terminal_layer": TERMINAL_LAYER,
        "policy_count": len(policies),
        "threshold_policies": THRESHOLD_POLICIES,
        "group_inventory": inventory,
        "selected_policy_summary": best,
        "selected_dataset_breakdown": selected_dataset_rows,
        "selected_stage_breakdown": stage_summary,
        "summary_rows": summary_rows,
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
            "summary_rows": summary_rows,
        },
        ARTIFACT_PT,
    )
    write_json(REPORT_JSON, payload)
    write_json(SUMMARY_JSON, payload)
    write_csv(POLICY_ROWS_CSV, policy_rows)
    write_csv(GROUP_ROWS_CSV, group_rows)
    write_csv(STAGE_ROWS_CSV, stage_rows)

    row_keys = [
        "dataset_name",
        "policy",
        "threshold_policy",
        "group_count",
        "pre_terminal_oracle_retention",
        "pre_terminal_false_prune_rate",
        "avg_pre_terminal_survivors",
        "pre_terminal_best_reward",
        "terminal_selected_reward",
        "terminal_oracle_selected_rate",
        "terminal_regret",
        "best_survivor_regret",
        "natural_singleton_before_terminal_rate",
        "threshold_fallback_rate",
    ]
    overall_table = overall_rows[:20]
    dataset_table = sorted(selected_dataset_rows, key=lambda r: str(r.get("dataset_name")))
    lines = [
        "# DualAnchor Dynamic-Threshold Loop Policy v1",
        "",
        f"BG_DUALANCHOR_THRESHOLD_LOOP_POLICY_VERDICT = {verdict}",
        "",
        "Mode: `CACHED_THRESHOLD_LOOP_APPROX`. Non-terminal 24_L4/36_L4/47_L4 steps are dynamic threshold filters. The only explicit top1 is terminal 47_L4 pairwise winner selection.",
        "",
        f"- runtime device: `{device}`",
        f"- policies: `{len(policies)}`",
        f"- threshold policies: `{len(THRESHOLD_POLICIES)}`",
        f"- datasets: `{len(datasets)}`",
        f"- groups: `{sum(len(d['groups']) for d in datasets)}`",
        f"- selected policy: `{best.get('policy')}`",
        f"- selected threshold policy: `{best.get('threshold_policy')}`",
        f"- pre-terminal oracle retention: `{best.get('pre_terminal_oracle_retention')}`",
        f"- pre-terminal false prune: `{best.get('pre_terminal_false_prune_rate')}`",
        f"- avg pre-terminal survivors: `{best.get('avg_pre_terminal_survivors')}`",
        f"- pre-terminal best reward: `{best.get('pre_terminal_best_reward')}`",
        f"- terminal selected reward: `{best.get('terminal_selected_reward')}`",
        "",
        "## Group Inventory",
        "",
    ]
    lines.extend(md_table(inventory, ["dataset_name", "group_count", "candidate_count", "mean_group_size", "reward_diverse_groups", "domains"]))
    lines.extend(["", "## Overall Policies", ""])
    lines.extend(md_table(overall_table, row_keys))
    lines.extend(["", "## Selected Dataset Breakdown", ""])
    lines.extend(md_table(dataset_table, row_keys))
    lines.extend(["", "## Selected Stage Breakdown", ""])
    lines.extend(md_table(stage_summary, ["loop", "layer", "stage_count", "oracle_retention_after_stage", "avg_survivors_before", "avg_survivors_after", "avg_threshold", "fallback_rate"]))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This separates threshold survival from terminal pairwise winner selection.",
            "- Non-terminal layers do not use a fixed top-k or top1 rule; survivor count is set by the dynamic threshold and score distribution.",
            "- If the threshold would produce no survivors, a safety fallback keeps the measured best candidate and reports the fallback; selected runs had this tracked explicitly.",
            "- This remains a cached approximation over already-materialized candidate groups; it cannot prove true recurrent fork/carry or compute savings.",
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
    print(f"BG_DUALANCHOR_THRESHOLD_LOOP_POLICY_VERDICT = {verdict}", flush=True)
    print(f"selected_policy = {best.get('policy')} threshold = {best.get('threshold_policy')}", flush=True)
    print(
        "pre_terminal_retention = "
        f"{best.get('pre_terminal_oracle_retention')} false_prune = {best.get('pre_terminal_false_prune_rate')} "
        f"terminal_reward = {best.get('terminal_selected_reward')}",
        flush=True,
    )
    print(f"report = {rel(REPORT_MD)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
