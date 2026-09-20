"""DualAnchor all-loop guarded threshold policy v1.

Follow-up to ``run_bg_dualanchor_all_loop_audit_v1.py``. This keeps the same
all-loop interpretation but patches the threshold policy with:

- anchor disagreement detection,
- top candidate rescue from each DualAnchor tap,
- a post-rescue hard budget cap,
- confidence-gated terminal layer-47 top1.

This is still cached evaluator work only. It does not run true fork/carry,
generation, action steering, Ouro training, registry updates, wrapper/local
agent code, Hunter-Seeker modules, or production routing changes.
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
from bg_merged_tap_v1_common import json_default, safe_float, score_diff
from run_bg_layer_native_two_tap_readiness_v1 import ANCHOR_GROUPS, tensor_weight, write_csv, write_json, write_md

import run_bg_dualanchor_all_loop_audit_v1 as base


OUT_ROOT = PROBE_ROOT / "bg_dualanchor_all_loop_guarded_policy_v1_2026-05-31"
REPORT_JSON = OUT_ROOT / "guarded_policy.json"
REPORT_MD = OUT_ROOT / "guarded_policy.md"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
ARTIFACT_PT = OUT_ROOT / "dualanchor_all_loop_guarded_policy_v1.pt"
GROUP_ROWS_CSV = OUT_ROOT / "guarded_group_rows.csv"
STAGE_ROWS_CSV = OUT_ROOT / "guarded_stage_rows.csv"
POLICY_ROWS_CSV = OUT_ROOT / "policy_rows.csv"

GUARD_SPECS: dict[str, dict[str, Any]] = {
    "core_budget8_keep2_conf080": {
        "rescue_each_anchor_top1": True,
        "hard_budget": 8,
        "terminal_conf_margin": 0.80,
        "terminal_corr_min": 0.10,
        "terminal_max_repetition": 0.35,
        "terminal_defer_keep": 2,
    },
    "core_budget8_keep2_conf120": {
        "rescue_each_anchor_top1": True,
        "hard_budget": 8,
        "terminal_conf_margin": 1.20,
        "terminal_corr_min": 0.20,
        "terminal_max_repetition": 0.35,
        "terminal_defer_keep": 2,
    },
    "core_budget8_keep3_conf120": {
        "rescue_each_anchor_top1": True,
        "hard_budget": 8,
        "terminal_conf_margin": 1.20,
        "terminal_corr_min": 0.20,
        "terminal_max_repetition": 0.35,
        "terminal_defer_keep": 3,
    },
    "core_budget8_keep4_conf120": {
        "rescue_each_anchor_top1": True,
        "hard_budget": 8,
        "terminal_conf_margin": 1.20,
        "terminal_corr_min": 0.20,
        "terminal_max_repetition": 0.35,
        "terminal_defer_keep": 4,
    },
    "core_budget8_keep5_conf120": {
        "rescue_each_anchor_top1": True,
        "hard_budget": 8,
        "terminal_conf_margin": 1.20,
        "terminal_corr_min": 0.20,
        "terminal_max_repetition": 0.35,
        "terminal_defer_keep": 5,
    },
    "core_budget6_keep3_conf120": {
        "rescue_each_anchor_top1": True,
        "hard_budget": 6,
        "terminal_conf_margin": 1.20,
        "terminal_corr_min": 0.20,
        "terminal_max_repetition": 0.35,
        "terminal_defer_keep": 3,
    },
    "core_budget6_keep4_conf120": {
        "rescue_each_anchor_top1": True,
        "hard_budget": 6,
        "terminal_conf_margin": 1.20,
        "terminal_corr_min": 0.20,
        "terminal_max_repetition": 0.35,
        "terminal_defer_keep": 4,
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


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    return float(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy))


def ranks_from_scores(scores: Sequence[float]) -> list[int]:
    order = [idx for idx, _ in sorted(enumerate(scores), key=lambda item: (-safe_float(item[1], -1e9), item[0]))]
    ranks = [0] * len(order)
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    return ranks


def normalized(vals: Sequence[float]) -> list[float]:
    xs = [safe_float(v, 0.0) for v in vals]
    if not xs:
        return []
    mu = mean(xs)
    sd = math.sqrt(mean((x - mu) ** 2 for x in xs)) if len(xs) > 1 else 0.0
    if sd < 1e-8:
        return [0.0 for _ in xs]
    return [(x - mu) / sd for x in xs]


def clean_local_index(group: Sequence[dict[str, Any]]) -> int | None:
    for idx, row in enumerate(group):
        if str(row.get("delta_family")) == "clean" or str(row.get("direction_name")) == "clean_zero" or str(row.get("branch_id")) == "0":
            return idx
    return None


def flag_true(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def terminal_candidate_stable(row: dict[str, Any], guard_spec: dict[str, Any]) -> tuple[bool, float, float]:
    parse_value = row.get("parse_success")
    parse_ok = True if parse_value is None else flag_true(parse_value)
    empty_output = flag_true(row.get("empty_output"))
    repetition = safe_float(row.get("repetition_rate"), 0.0)
    max_repetition = float(guard_spec.get("terminal_max_repetition", 1.0))
    stable = parse_ok and not empty_output and repetition <= max_repetition
    return stable, 1.0 if parse_ok else 0.0, repetition


def stage_score_details(
    group: Sequence[dict[str, Any]],
    taps_by_layer: dict[str, dict[str, dict[str, Any]]],
    layer: int,
    config: str,
) -> dict[str, Any]:
    layer_key = f"{layer}_L4"
    raw_by_role: dict[str, list[float]] = {}
    norm_by_role: dict[str, list[float]] = {}
    for role in ANCHOR_GROUPS:
        tap = (taps_by_layer.get(layer_key) or {}).get(role)
        weight = tensor_weight(tap or {})
        if not isinstance(weight, torch.Tensor):
            continue
        arch = str((tap or {}).get("architecture") or "AntisymLinear")
        scores: list[float] = []
        for i, left in enumerate(group):
            lvec = base.config_vector(left, config)
            if not isinstance(lvec, torch.Tensor) or int(lvec.numel()) != int(weight.numel()):
                return {}
            vals = []
            for j, right in enumerate(group):
                if i == j:
                    continue
                rvec = base.config_vector(right, config)
                if not isinstance(rvec, torch.Tensor) or int(rvec.numel()) != int(weight.numel()):
                    return {}
                vals.append(score_diff(weight, arch, lvec - rvec))
            scores.append(finite_mean(vals))
        raw_by_role[role] = scores
        norm_by_role[role] = normalized(scores)
    if not norm_by_role:
        return {}
    n = min(len(v) for v in norm_by_role.values())
    combined = [finite_mean(scores[i] for scores in norm_by_role.values()) for i in range(n)]
    role_names = [role for role in ANCHOR_GROUPS if role in norm_by_role]
    corr = float("nan")
    top1_by_role: dict[str, int] = {}
    rank_disagreement = 0.0
    if len(role_names) >= 2:
        a = norm_by_role[role_names[0]][:n]
        b = norm_by_role[role_names[1]][:n]
        corr = pearson(a, b)
        top1_by_role = {
            role_names[0]: max(range(n), key=lambda i: (safe_float(a[i], -1e9), -i)),
            role_names[1]: max(range(n), key=lambda i: (safe_float(b[i], -1e9), -i)),
        }
        ra = ranks_from_scores(a)
        rb = ranks_from_scores(b)
        denom = max(n - 1, 1)
        rank_disagreement = mean(abs(ra[i] - rb[i]) / denom for i in range(n))
    else:
        only = norm_by_role[role_names[0]][:n]
        top1_by_role = {role_names[0]: max(range(n), key=lambda i: (safe_float(only[i], -1e9), -i))}
    order = [idx for idx, _ in sorted(enumerate(combined), key=lambda item: (-safe_float(item[1], -1e9), item[0]))]
    return {
        "raw_by_role": raw_by_role,
        "norm_by_role": norm_by_role,
        "combined": combined,
        "order": order,
        "corr": corr,
        "top1_by_role": top1_by_role,
        "top1_disagreement": len(set(top1_by_role.values())) > 1,
        "rank_disagreement": rank_disagreement,
    }


def guarded_keep_indices(
    group: Sequence[dict[str, Any]],
    details: dict[str, Any],
    loop_idx: int,
    layer: int,
    threshold_spec: dict[str, Any],
    guard_spec: dict[str, Any],
) -> tuple[list[int], dict[str, Any]]:
    scores = details.get("combined") or []
    threshold = base.threshold_value(scores, loop_idx, layer, threshold_spec)
    keep = {i for i, score in enumerate(scores) if safe_float(score, -1e9) >= threshold}
    if not keep and scores:
        keep.add(max(range(len(scores)), key=lambda i: (safe_float(scores[i], -1e9), -i)))
    corr = safe_float(details.get("corr"), 0.0)
    top1_disagreement = bool(details.get("top1_disagreement"))
    rank_disagreement = safe_float(details.get("rank_disagreement"), 0.0)
    disagreement = top1_disagreement
    protected: set[int] = set()
    if disagreement and guard_spec.get("rescue_each_anchor_top1"):
        protected.update(int(i) for i in (details.get("top1_by_role") or {}).values())
    keep.update(protected)
    pre_budget_count = len(keep)
    hard_budget = int(guard_spec.get("hard_budget") or 0)
    budget_trimmed = False
    if hard_budget > 0 and len(keep) > hard_budget:
        ordered = list(details.get("order") or range(len(scores)))
        new_keep = set(protected)
        for idx in ordered:
            if len(new_keep) >= hard_budget:
                break
            if idx in keep:
                new_keep.add(idx)
        keep = new_keep
        budget_trimmed = True
    return sorted(keep), {
        "threshold": threshold,
        "corr": corr,
        "top1_disagreement": 1.0 if top1_disagreement else 0.0,
        "rank_disagreement": rank_disagreement,
        "disagreement_guard_active": 1.0 if disagreement else 0.0,
        "anchor_rescue_count": len(set((details.get("top1_by_role") or {}).values())),
        "pre_budget_count": pre_budget_count,
        "budget_trimmed": 1.0 if budget_trimmed else 0.0,
    }


def terminal_gate(
    group: Sequence[dict[str, Any]],
    content_taps: dict[str, dict[str, dict[str, Any]]],
    guard_spec: dict[str, Any],
) -> dict[str, Any]:
    details = stage_score_details(group, content_taps, 47, "47_L4")
    if not details:
        selected = [0] if group else []
        return {
            "terminal_indices": selected,
            "forced_top1_index": selected[0] if selected else 0,
            "terminal_confident": 0.0,
            "terminal_deferred": 1.0,
            "terminal_stable_candidate": 0.0,
        }
    order = list(details["order"])
    forced_top1 = order[0]
    top2 = order[1] if len(order) > 1 else order[0]
    scores = list(details["combined"])
    margin = safe_float(scores[forced_top1], 0.0) - safe_float(scores[top2], 0.0) if len(order) > 1 else 999.0
    stable, parse_success, repetition = terminal_candidate_stable(group[forced_top1], guard_spec)
    confident = (
        not bool(details.get("top1_disagreement"))
        and safe_float(details.get("corr"), 0.0) >= float(guard_spec["terminal_corr_min"])
        and margin >= float(guard_spec["terminal_conf_margin"])
        and stable
    )
    if confident:
        terminal_indices = [forced_top1]
    else:
        terminal_indices = order[: max(1, int(guard_spec.get("terminal_defer_keep") or 2))]
    return {
        "terminal_indices": terminal_indices,
        "forced_top1_index": forced_top1,
        "terminal_confident": 1.0 if confident else 0.0,
        "terminal_deferred": 0.0 if confident else 1.0,
        "terminal_margin": margin,
        "terminal_corr": details.get("corr"),
        "terminal_top1_disagreement": 1.0 if details.get("top1_disagreement") else 0.0,
        "terminal_rank_disagreement": details.get("rank_disagreement"),
        "terminal_stable_candidate": 1.0 if stable else 0.0,
        "terminal_parse_success": parse_success,
        "terminal_repetition_rate": repetition,
    }


def simulate_group(
    group: Sequence[dict[str, Any]],
    taps: dict[str, dict[str, dict[str, Any]]],
    content_taps: dict[str, dict[str, dict[str, Any]]],
    threshold_spec: dict[str, Any],
    guard_spec: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    survivors = list(range(len(group)))
    rewards = [base.row_reward(row) for row in group]
    oracle = max(rewards)
    oracle_idx = {idx for idx, reward in enumerate(rewards) if reward == oracle}
    stage_rows: list[dict[str, Any]] = []
    clean_idx_global = clean_local_index(group)
    first_oracle_loss = None
    for loop_idx, loop in enumerate(base.LOOPS):
        for layer in base.TAP_LAYERS:
            if len(survivors) <= 1:
                break
            if loop == 4 and layer == 47:
                continue
            sub_group = [group[i] for i in survivors]
            details = stage_score_details(sub_group, taps, layer, f"{layer}_L{loop}")
            if not details:
                continue
            local_keep, guard_info = guarded_keep_indices(sub_group, details, loop_idx, layer, threshold_spec, guard_spec)
            before = list(survivors)
            survivors = [survivors[i] for i in local_keep]
            retained = bool(set(survivors) & oracle_idx)
            if not retained and first_oracle_loss is None:
                first_oracle_loss = {"loop": loop, "layer": layer}
            mu, sd, max_score, min_score = base.score_stats(details.get("combined") or [])
            stage_rows.append(
                {
                    "loop": loop,
                    "layer": layer,
                    "config": f"{layer}_L{loop}",
                    "survivors_before": len(before),
                    "survivors_after": len(survivors),
                    "score_mean": mu,
                    "score_std": sd,
                    "score_max": max_score,
                    "score_min": min_score,
                    "oracle_retained_after_stage": 1.0 if retained else 0.0,
                    **guard_info,
                }
            )
    pre_terminal = list(survivors)
    pre_retained = bool(set(pre_terminal) & oracle_idx)
    pre_best = max(rewards[i] for i in pre_terminal) if pre_terminal else float("nan")
    terminal = terminal_gate([group[i] for i in pre_terminal], content_taps, guard_spec)
    terminal_global = [pre_terminal[i] for i in terminal["terminal_indices"] if i < len(pre_terminal)]
    forced_global = pre_terminal[int(terminal["forced_top1_index"])] if pre_terminal else 0
    terminal_best = max(rewards[i] for i in terminal_global) if terminal_global else float("nan")
    terminal_kept_oracle = bool(set(terminal_global) & oracle_idx)
    return (
        {
            "group_size": len(group),
            "clean_start_index": clean_idx_global,
            "clean_start_survived_preterminal": 1.0 if clean_idx_global in pre_terminal else 0.0,
            "clean_start_pruned_preterminal": 0.0 if clean_idx_global in pre_terminal else 1.0,
            "pre_terminal_survivor_count": len(pre_terminal),
            "pre_terminal_oracle_retained": 1.0 if pre_retained else 0.0,
            "pre_terminal_false_prune": 0.0 if pre_retained else 1.0,
            "pre_terminal_best_reward": pre_best,
            "terminal_survivor_count": len(terminal_global),
            "terminal_oracle_retained": 1.0 if terminal_kept_oracle else 0.0,
            "terminal_best_reward": terminal_best,
            "terminal_forced_top1_reward": rewards[forced_global] if forced_global < len(rewards) else float("nan"),
            "terminal_forced_top1_oracle": 1.0 if forced_global in oracle_idx else 0.0,
            "terminal_confident": terminal.get("terminal_confident"),
            "terminal_deferred": terminal.get("terminal_deferred"),
            "terminal_margin": terminal.get("terminal_margin"),
            "terminal_corr": terminal.get("terminal_corr"),
            "terminal_top1_disagreement": terminal.get("terminal_top1_disagreement"),
            "terminal_rank_disagreement": terminal.get("terminal_rank_disagreement"),
            "terminal_stable_candidate": terminal.get("terminal_stable_candidate"),
            "terminal_parse_success": terminal.get("terminal_parse_success"),
            "terminal_repetition_rate": terminal.get("terminal_repetition_rate"),
            "terminal_indices": terminal_global,
            "forced_terminal_index": forced_global,
            "pre_terminal_indices": pre_terminal,
            "first_oracle_loss_loop": None if first_oracle_loss is None else first_oracle_loss["loop"],
            "first_oracle_loss_layer": None if first_oracle_loss is None else first_oracle_loss["layer"],
        },
        stage_rows,
    )


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"group_count": 0}
    keys = [
        "clean_start_pruned_preterminal",
        "pre_terminal_survivor_count",
        "pre_terminal_oracle_retained",
        "pre_terminal_false_prune",
        "pre_terminal_best_reward",
        "terminal_survivor_count",
        "terminal_oracle_retained",
        "terminal_best_reward",
        "terminal_forced_top1_reward",
        "terminal_forced_top1_oracle",
        "terminal_confident",
        "terminal_deferred",
        "terminal_margin",
        "terminal_corr",
        "terminal_top1_disagreement",
        "terminal_stable_candidate",
        "terminal_parse_success",
        "terminal_repetition_rate",
    ]
    out = {"group_count": len(rows)}
    for key in keys:
        out[key] = finite_mean(row.get(key) for row in rows)
    return out


def verdict_for(best: dict[str, Any]) -> str:
    retention = safe_float(best.get("pre_terminal_oracle_retained"), 0.0)
    false_prune = safe_float(best.get("pre_terminal_false_prune"), 1.0)
    terminal_retention = safe_float(best.get("terminal_oracle_retained"), 0.0)
    avg_terminal = safe_float(best.get("terminal_survivor_count"), 99.0)
    if retention >= 0.93 and false_prune <= 0.07 and terminal_retention >= 0.90 and avg_terminal <= 2.5:
        return "GUARDED_SURVIVAL_READY_TERMINAL_DEFER"
    if retention >= 0.90 and false_prune <= 0.10:
        return "GUARDED_SURVIVAL_READY_TERMINAL_WEAK"
    if retention >= 0.85 and false_prune <= 0.15:
        return "GUARDED_SURVIVAL_PARTIAL"
    return "GUARDED_SURVIVAL_WEAK"


def main() -> int:
    ensure_root()
    policies, policy_rows = base.load_constrained_policies()
    full_loop_datasets, coverage = base.load_full_loop_branch_groups()
    content_policy = next(v for k, v in policies.items() if "old_only" in k)
    group_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for policy_name, taps in policies.items():
        for threshold_name, threshold_spec in base.PRIMARY_THRESHOLD_POLICIES.items():
            if threshold_name != "mean_floor_very_loose":
                continue
            for guard_name, guard_spec in GUARD_SPECS.items():
                all_rows = []
                for dataset in full_loop_datasets:
                    dataset_rows = []
                    for group in dataset["groups"]:
                        row, stages = simulate_group(group, taps, content_policy, threshold_spec, guard_spec)
                        row.update(
                            {
                                "dataset_name": dataset["dataset_name"],
                                "group_id": group[0].get("group_id"),
                                "domain": group[0].get("domain"),
                                "policy": policy_name,
                                "threshold_policy": threshold_name,
                                "guard_policy": guard_name,
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
                                    "guard_policy": guard_name,
                                }
                            )
                    sm = summarize(dataset_rows)
                    sm.update({"dataset_name": dataset["dataset_name"], "policy": policy_name, "threshold_policy": threshold_name, "guard_policy": guard_name})
                    summary_rows.append(sm)
                    all_rows.extend(dataset_rows)
                sm = summarize(all_rows)
                sm.update({"dataset_name": "ALL_FULL_LOOP", "policy": policy_name, "threshold_policy": threshold_name, "guard_policy": guard_name})
                summary_rows.append(sm)
    overall = [r for r in summary_rows if r.get("dataset_name") == "ALL_FULL_LOOP"]
    overall.sort(
        key=lambda r: (
            -safe_float(r.get("pre_terminal_oracle_retained"), 0.0),
            safe_float(r.get("pre_terminal_false_prune"), 1.0),
            -safe_float(r.get("terminal_oracle_retained"), 0.0),
            -safe_float(r.get("terminal_best_reward"), 0.0),
            safe_float(r.get("terminal_survivor_count"), 99.0),
        )
    )
    best = overall[0] if overall else {}
    verdict = verdict_for(best) if best else "DATA_LIMITED"
    selected_group_rows = [
        row
        for row in group_rows
        if row.get("policy") == best.get("policy")
        and row.get("threshold_policy") == best.get("threshold_policy")
        and row.get("guard_policy") == best.get("guard_policy")
    ]
    confidence_breakdown = {
        "all": summarize(selected_group_rows),
        "confident_top1_allowed": summarize([row for row in selected_group_rows if safe_float(row.get("terminal_confident"), 0.0) > 0.5]),
        "deferred_terminal_sets": summarize([row for row in selected_group_rows if safe_float(row.get("terminal_deferred"), 0.0) > 0.5]),
    }
    payload = {
        "BG_DUALANCHOR_ALL_LOOP_GUARDED_POLICY_VERDICT": verdict,
        "status": verdict,
        "mode": "CACHED_ALL_LOOP_GUARDED_THRESHOLD_POLICY",
        "true_fork_carry_claimed": False,
        "action_steering_claimed": False,
        "uses_loop4_only": False,
        "guard_specs": GUARD_SPECS,
        "coverage": coverage,
        "best_summary": best,
        "confidence_breakdown": confidence_breakdown,
        "summary_rows": summary_rows,
    }
    write_json(REPORT_JSON, payload)
    write_json(SUMMARY_JSON, payload)
    write_csv(GROUP_ROWS_CSV, group_rows)
    write_csv(STAGE_ROWS_CSV, stage_rows)
    write_csv(POLICY_ROWS_CSV, policy_rows)
    torch.save({"summary": payload, "group_rows": group_rows, "stage_rows": stage_rows, "summary_rows": summary_rows}, ARTIFACT_PT)

    selected_stage = [
        row
        for row in stage_rows
        if row.get("policy") == best.get("policy")
        and row.get("threshold_policy") == best.get("threshold_policy")
        and row.get("guard_policy") == best.get("guard_policy")
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
                "disagreement_guard_rate": finite_mean(v.get("disagreement_guard_active") for v in vals),
                "anchor_rescue_count": finite_mean(v.get("anchor_rescue_count") for v in vals),
                "budget_trim_rate": finite_mean(v.get("budget_trimmed") for v in vals),
            }
        )
    lines = [
        "# DualAnchor All-Loop Guarded Threshold Policy v1",
        "",
        f"BG_DUALANCHOR_ALL_LOOP_GUARDED_POLICY_VERDICT = {verdict}",
        "",
        "This patch adds DualAnchor top1-disagreement rescue, a hard budget cap, and confidence-gated terminal 47 selection. It does not add a default clean-branch rescue.",
        "",
        "## Best Policy",
        "",
        f"- policy: `{best.get('policy')}`",
        f"- threshold policy: `{best.get('threshold_policy')}`",
        f"- guard policy: `{best.get('guard_policy')}`",
        f"- pre-terminal oracle retention: `{best.get('pre_terminal_oracle_retained')}`",
        f"- pre-terminal false prune: `{best.get('pre_terminal_false_prune')}`",
        f"- avg pre-terminal survivors: `{best.get('pre_terminal_survivor_count')}`",
        f"- clean pruned pre-terminal: `{best.get('clean_start_pruned_preterminal')}`",
        f"- terminal survivor count: `{best.get('terminal_survivor_count')}`",
        f"- terminal oracle retained: `{best.get('terminal_oracle_retained')}`",
        f"- terminal best reward: `{best.get('terminal_best_reward')}`",
        f"- forced terminal top1 reward: `{best.get('terminal_forced_top1_reward')}`",
        f"- forced terminal top1 oracle: `{best.get('terminal_forced_top1_oracle')}`",
        f"- terminal confident rate: `{best.get('terminal_confident')}`",
        f"- terminal deferred rate: `{best.get('terminal_deferred')}`",
        f"- terminal stable-candidate rate: `{best.get('terminal_stable_candidate')}`",
        "",
        "## Overall Policies",
        "",
    ]
    lines.extend(
        md_table(
            overall[:20],
            [
                "policy",
                "guard_policy",
                "group_count",
                "pre_terminal_oracle_retained",
                "pre_terminal_false_prune",
                "pre_terminal_survivor_count",
                "terminal_survivor_count",
                "terminal_oracle_retained",
                "terminal_best_reward",
                "terminal_forced_top1_reward",
                "terminal_forced_top1_oracle",
                "terminal_confident",
                "terminal_deferred",
                "terminal_stable_candidate",
            ],
        )
    )
    confidence_rows = [{"split": name, **values} for name, values in confidence_breakdown.items()]
    lines.extend(["", "## Terminal Confidence Breakdown", ""])
    lines.extend(
        md_table(
            confidence_rows,
            [
                "split",
                "group_count",
                "terminal_survivor_count",
                "terminal_oracle_retained",
                "terminal_best_reward",
                "terminal_forced_top1_reward",
                "terminal_forced_top1_oracle",
                "terminal_confident",
                "terminal_deferred",
            ],
        )
    )
    lines.extend(["", "## Selected Stage Breakdown", ""])
    lines.extend(md_table(stage_summary, ["loop", "layer", "stage_count", "oracle_retention_after_stage", "avg_survivors_before", "avg_survivors_after", "disagreement_guard_rate", "anchor_rescue_count", "budget_trim_rate"]))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The guard improves the policy shape: it no longer relies on averaged DualAnchor scores alone.",
            "- Terminal low-confidence cases are retained as top2/deferred instead of pretending 47_L4 top1 is reliable.",
            "- This still does not prove true fork/carry or production compute savings because it is cached evaluator replay.",
            "",
            "## Files",
            "",
            f"- artifact: `{rel(ARTIFACT_PT)}`",
            f"- report json: `{rel(REPORT_JSON)}`",
            f"- group rows: `{rel(GROUP_ROWS_CSV)}`",
            f"- stage rows: `{rel(STAGE_ROWS_CSV)}`",
        ]
    )
    write_md(REPORT_MD, lines)
    write_md(SUMMARY_MD, lines)
    print(f"BG_DUALANCHOR_ALL_LOOP_GUARDED_POLICY_VERDICT = {verdict}", flush=True)
    print(f"selected_policy = {best.get('policy')} guard = {best.get('guard_policy')}", flush=True)
    print(
        f"pre_terminal_retention = {best.get('pre_terminal_oracle_retained')} "
        f"false_prune = {best.get('pre_terminal_false_prune')} "
        f"terminal_oracle_retained = {best.get('terminal_oracle_retained')} "
        f"terminal_survivors = {best.get('terminal_survivor_count')}",
        flush=True,
    )
    print(f"report = {rel(REPORT_MD)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
