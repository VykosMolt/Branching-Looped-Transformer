"""DualAnchor architecture-looped stratified probe v2.

This extends the v1 architecture-looped lineage probe with:

- stratified task selection across reasoning/science and split/source metadata,
- explicit terminal reward-diversity / positive-oracle accounting,
- terminal policy comparison on the final survivor pool,
- L47 candidate-tree ablations,
- lineage-depth and local false-prune recovery analysis.

It keeps the same mechanics as v1: cumulative hook replay over the intended
loop/layer schedule. It does not claim true latent fork/carry, action steering,
or compute savings.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import torch

from bg_branch_generator_v1_common import ALPHA_VALUE, BASIS_BANK_PT, load_audit_plan, load_pt, make_family_deltas, md_table, rel, row_reward, stable_v2_row, write_csv, write_json, write_md
from bg_hidden_origin_tap_common import PROBE_ROOT
from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor

import run_bg_dualanchor_all_loop_audit_v1 as base
import run_bg_dualanchor_all_loop_guarded_policy_v1 as guard
from run_bg_dualanchor_recursive_lineage_probe_v1 import (
    compact_row,
    evaluate_branch,
    finite_mean,
    make_hook_entry,
    safe_float,
    terminal_select,
    threshold_survive,
)


OUT_ROOT = PROBE_ROOT / "bg_dualanchor_architecture_looped_stratified_probe_v2_2026-05-31"
REPORT_JSON = OUT_ROOT / "architecture_looped_stratified_probe.json"
REPORT_MD = OUT_ROOT / "architecture_looped_stratified_probe.md"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
ARTIFACT_PT = OUT_ROOT / "dualanchor_architecture_looped_stratified_probe_v2.pt"
PARTIAL_PT = OUT_ROOT / "dualanchor_architecture_looped_stratified_probe_v2.partial.pt"
STATE_JSON = OUT_ROOT / "architecture_looped_stratified_state.json"
ROWS_CSV = OUT_ROOT / "architecture_looped_stratified_rows.csv"
STAGE_ROWS_CSV = OUT_ROOT / "architecture_looped_stratified_stage_rows.csv"
TASK_ROWS_CSV = OUT_ROOT / "architecture_looped_stratified_task_rows.csv"
TERMINAL_POLICY_CSV = OUT_ROOT / "terminal_policy_rows.csv"
L47_ABLATION_CSV = OUT_ROOT / "l47_ablation_rows.csv"
RECOVERY_CSV = OUT_ROOT / "false_prune_recovery_rows.csv"

SELECTED_POLICY = "dualanchor_adaptive_branch_anchor_light_AntisymLinear"
THRESHOLD_POLICY = "mean_floor_very_loose"
GUARD_POLICY = "core_budget8_keep5_conf120"
NONTERMINAL_STAGES = tuple((loop, layer) for loop in (1, 2, 3, 4) for layer in (24, 36, 47) if not (loop == 4 and layer == 47))
TERMINAL_STAGE = (4, 47)
PRIMARY_DOMAINS = ("reasoning", "science")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tasks", type=int, default=24)
    parser.add_argument("--children-per-parent", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--alpha-bucket", default="alpha_0_005")
    parser.add_argument("--delta-family", default="old_tap_aligned")
    parser.add_argument("--split-mode", choices=("all", "heldout_only", "heldout_val"), default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_root() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def load_bank() -> dict[str, Any]:
    return load_pt(BASIS_BANK_PT, {"directions_by_layer": {}, "directions": []}) or {"directions_by_layer": {}, "directions": []}


def truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def task_sort_key(row: dict[str, Any]) -> tuple[int, int, float, str, str]:
    split_rank = {"heldout": 0, "val": 1, "train": 2}.get(str(row.get("split")), 9)
    likely = 1 if truthy(row.get("heldout_likely_non_tie")) else 0
    prior_pairs = int(row.get("prior_non_tie_pairs") or 0)
    priority = float(row.get("priority_score_v4") or row.get("priority_score") or 0.0)
    return (split_rank, -likely, -prior_pairs, -priority, str(row.get("source_dataset")), str(row.get("task_id")))


def select_stratified_tasks(max_tasks: int, split_mode: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan = load_audit_plan()
    tasks = [
        dict(task)
        for task in plan.get("tasks", [])
        if str(task.get("domain")) in PRIMARY_DOMAINS and task.get("correct_option")
    ]
    if split_mode == "heldout_only":
        allowed = {"heldout"}
    elif split_mode == "heldout_val":
        allowed = {"heldout", "val"}
    else:
        allowed = {"heldout", "val", "train"}
    tasks = [task for task in tasks if str(task.get("split")) in allowed]

    by_domain = {domain: sorted([task for task in tasks if str(task.get("domain")) == domain], key=task_sort_key) for domain in PRIMARY_DOMAINS}
    target = max(1, int(max_tasks))
    per_domain_target = {domain: target // len(PRIMARY_DOMAINS) for domain in PRIMARY_DOMAINS}
    for domain in PRIMARY_DOMAINS[: target % len(PRIMARY_DOMAINS)]:
        per_domain_target[domain] += 1

    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for domain in PRIMARY_DOMAINS:
        for task in by_domain[domain][: per_domain_target[domain]]:
            selected.append(task)
            used.add(str(task.get("task_id")))
    if len(selected) < target:
        leftovers = sorted([task for task in tasks if str(task.get("task_id")) not in used], key=task_sort_key)
        for task in leftovers[: target - len(selected)]:
            selected.append(task)
            used.add(str(task.get("task_id")))

    inventory = {
        "requested_max_tasks": target,
        "selected_count": len(selected),
        "split_mode": split_mode,
        "domain_counts": dict(Counter(str(task.get("domain")) for task in selected)),
        "split_counts": dict(Counter(str(task.get("split")) for task in selected)),
        "source_counts": dict(Counter(str(task.get("source_dataset")) for task in selected)),
        "likely_non_tie_count": sum(1 for task in selected if truthy(task.get("heldout_likely_non_tie"))),
        "prior_non_tie_positive_count": sum(1 for task in selected if int(task.get("prior_non_tie_pairs") or 0) > 0),
        "selected_task_ids": [str(task.get("task_id")) for task in selected],
    }
    return selected, inventory


def new_child_id(parent_id: str, loop: int, layer: int, mutation_index: int) -> str:
    return f"{parent_id}/L{loop}_{layer}_p{int(mutation_index)}"


def save_partial(rows: list[dict[str, Any]], stage_rows: list[dict[str, Any]], task_rows: list[dict[str, Any]], errors: list[dict[str, Any]], completed: set[str], task_inventory: dict[str, Any]) -> None:
    payload = {
        "rows": rows,
        "stage_rows": stage_rows,
        "task_rows": task_rows,
        "errors": errors,
        "completed_task_ids": sorted(completed),
        "task_inventory": task_inventory,
        "saved_at": time.time(),
    }
    torch.save(payload, PARTIAL_PT)
    write_json(
        STATE_JSON,
        {
            "completed_task_ids": sorted(completed),
            "row_count": len(rows),
            "stage_rows": len(stage_rows),
            "task_rows": len(task_rows),
            "errors": len(errors),
            "task_inventory": task_inventory,
            "saved_at": time.time(),
        },
    )


def summarize_by(rows: Sequence[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(key)].append(row)
    out = []
    for value, vals in sorted(grouped.items(), key=lambda item: str(item[0])):
        child_vals = [row for row in vals if row.get("parent_branch_id")]
        out.append(
            {
                key: value,
                "candidate_count": len(vals),
                "mean_reward": finite_mean(row_reward(row) for row in vals),
                "parse_rate": finite_mean(1.0 if row.get("parse_success") else 0.0 for row in vals),
                "stable_rate": finite_mean(1.0 if stable_v2_row(row) else 0.0 for row in vals),
                "mean_parent_delta": finite_mean(row.get("reward_delta_from_parent") for row in child_vals),
                "child_improved_parent_rate": finite_mean(1.0 if safe_float(row.get("reward_delta_from_parent"), 0.0) > 0 else 0.0 for row in child_vals),
                "child_tied_parent_rate": finite_mean(1.0 if safe_float(row.get("reward_delta_from_parent"), 999.0) == 0 else 0.0 for row in child_vals),
                "child_worse_parent_rate": finite_mean(1.0 if safe_float(row.get("reward_delta_from_parent"), 0.0) < 0 else 0.0 for row in child_vals),
            }
        )
    return out


def lineage_targets(row: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for item in row.get("hook_path") or []:
        try:
            layer = int(item.get("target_layer"))
            loop = int(item.get("target_loop"))
        except Exception:
            continue
        out.add(f"L{loop}_{layer}")
    return out


def candidate_scores(candidates: Sequence[dict[str, Any]], taps: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    details = guard.stage_score_details(candidates, taps, 47, "47_L4")
    if details:
        return details
    order = list(range(len(candidates)))
    return {
        "combined": [0.0 for _ in candidates],
        "order": order,
        "top1_by_role": {},
        "corr": float("nan"),
        "top1_disagreement": False,
        "rank_disagreement": float("nan"),
    }


def policy_metrics(task: dict[str, Any], candidates: Sequence[dict[str, Any]], selected_indices: Sequence[int], policy_name: str) -> dict[str, Any]:
    rewards = [row_reward(row) for row in candidates]
    oracle = max(rewards) if rewards else float("nan")
    oracle_idx = {idx for idx, value in enumerate(rewards) if value == oracle}
    selected = [int(i) for i in selected_indices if 0 <= int(i) < len(candidates)]
    selected_rewards = [rewards[i] for i in selected]
    return {
        "task_id": task.get("task_id"),
        "domain": task.get("domain"),
        "policy": policy_name,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "oracle_reward": oracle,
        "oracle_retained": 1.0 if set(selected) & oracle_idx else 0.0,
        "best_selected_reward": max(selected_rewards) if selected_rewards else float("nan"),
        "first_selected_reward": selected_rewards[0] if selected_rewards else float("nan"),
        "first_selected_oracle": 1.0 if selected and selected[0] in oracle_idx else 0.0,
        "selected_indices": selected,
    }


def terminal_policy_rows(task: dict[str, Any], final_pool: Sequence[dict[str, Any]], taps: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    details = candidate_scores(final_pool, taps)
    order = [int(i) for i in details.get("order") or list(range(len(final_pool)))]
    terminal = terminal_select(final_pool, taps)
    rows = [
        policy_metrics(task, final_pool, order[:1], "dualanchor_forced_top1"),
        policy_metrics(task, final_pool, terminal.get("terminal_indices") or [], "dualanchor_confidence_gated"),
        policy_metrics(task, final_pool, order[:2], "dualanchor_terminal_top2"),
        policy_metrics(task, final_pool, order[:5], "dualanchor_terminal_top5"),
        policy_metrics(task, final_pool, order, "terminal_defer_all"),
    ]
    role_top = sorted({int(i) for i in (details.get("top1_by_role") or {}).values() if 0 <= int(i) < len(final_pool)})
    if role_top:
        rows.append(policy_metrics(task, final_pool, role_top, "dualanchor_each_anchor_top1_rescue"))
    for row in rows:
        row.update(
            {
                "terminal_confident": terminal.get("terminal_confident"),
                "terminal_deferred": terminal.get("terminal_deferred"),
                "terminal_corr": terminal.get("terminal_corr"),
                "terminal_margin": terminal.get("terminal_margin"),
                "terminal_top1_disagreement": terminal.get("terminal_top1_disagreement"),
                "terminal_rank_disagreement": terminal.get("terminal_rank_disagreement"),
            }
        )
    return rows


def l47_ablation_rows(task: dict[str, Any], final_pool: Sequence[dict[str, Any]], taps: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    variants = {
        "full_l47_enabled": lambda row: True,
        "nonterminal_l47_disabled_tree_replay": lambda row: not any(x in lineage_targets(row) for x in {"L1_47", "L2_47", "L3_47"}),
        "terminal_only_l47_tree_replay": lambda row: not any(x in lineage_targets(row) for x in {"L1_47", "L2_47", "L3_47"}),
        "l2_47_only_tree_replay": lambda row: not any(x in lineage_targets(row) for x in {"L1_47", "L3_47"}),
        "no_l47_perturbation_tree_replay": lambda row: not any(target.endswith("_47") for target in lineage_targets(row)),
    }
    original_oracle = max((row_reward(row) for row in final_pool), default=float("nan"))
    rows = []
    for name, keep_fn in variants.items():
        subset = [row for row in final_pool if keep_fn(row)]
        if not subset:
            rows.append({"task_id": task.get("task_id"), "domain": task.get("domain"), "ablation": name, "candidate_count": 0, "oracle_retained_vs_full": 0.0, "best_reward": float("nan"), "forced_top1_reward": float("nan"), "forced_top1_oracle": 0.0})
            continue
        details = candidate_scores(subset, taps)
        order = [int(i) for i in details.get("order") or list(range(len(subset)))]
        rewards = [row_reward(row) for row in subset]
        best = max(rewards)
        forced = rewards[order[0]] if order else float("nan")
        rows.append(
            {
                "task_id": task.get("task_id"),
                "domain": task.get("domain"),
                "ablation": name,
                "candidate_count": len(subset),
                "oracle_retained_vs_full": 1.0 if best == original_oracle else 0.0,
                "best_reward": best,
                "forced_top1_reward": forced,
                "forced_top1_oracle": 1.0 if forced == best else 0.0,
                "tree_replay_only": 1.0 if name != "full_l47_enabled" else 0.0,
            }
        )
    return rows


def task_summary(task: dict[str, Any], final_pool: Sequence[dict[str, Any]], terminal: dict[str, Any], task_stage_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rewards = [row_reward(row) for row in final_pool]
    oracle = max(rewards) if rewards else float("nan")
    oracle_rows = [row for row in final_pool if row_reward(row) == oracle]
    unique_rewards = sorted({round(row_reward(row), 6) for row in final_pool})
    forced_idx = int(terminal.get("forced_top1_index", 0)) if final_pool else 0
    forced = final_pool[forced_idx] if forced_idx < len(final_pool) else {}
    false_prunes = [row for row in task_stage_rows if safe_float(row.get("stage_false_prune"), 0.0) > 0]
    return {
        "task_id": task.get("task_id"),
        "domain": task.get("domain"),
        "split": task.get("split"),
        "source_dataset": task.get("source_dataset"),
        "heldout_likely_non_tie": truthy(task.get("heldout_likely_non_tie")),
        "prior_non_tie_pairs": int(task.get("prior_non_tie_pairs") or 0),
        "final_candidate_count": len(final_pool),
        "final_perturbed_fraction": finite_mean(1.0 if int(row.get("perturb_count") or 0) > 0 else 0.0 for row in final_pool),
        "oracle_reward": oracle,
        "positive_oracle": 1.0 if oracle > 0 else 0.0,
        "terminal_reward_diverse": 1.0 if len(unique_rewards) > 1 else 0.0,
        "unique_terminal_rewards": unique_rewards,
        "oracle_perturb_counts": dict(Counter(int(row.get("perturb_count") or 0) for row in oracle_rows)),
        "oracle_birth_stages": dict(Counter(str(row.get("birth_stage")) for row in oracle_rows)),
        "final_branch_ids": [str(row.get("branch_id")) for row in final_pool],
        "terminal_survivor_count": terminal.get("terminal_survivor_count"),
        "terminal_oracle_retained": terminal.get("terminal_oracle_retained"),
        "terminal_best_reward": terminal.get("terminal_best_reward"),
        "terminal_forced_top1_reward": terminal.get("terminal_forced_top1_reward"),
        "terminal_forced_top1_oracle": terminal.get("terminal_forced_top1_oracle"),
        "terminal_confident": terminal.get("terminal_confident"),
        "terminal_deferred": terminal.get("terminal_deferred"),
        "forced_top1_perturb_count": forced.get("perturb_count"),
        "forced_top1_birth_stage": forced.get("birth_stage"),
        "stage_false_prunes": len(false_prunes),
        "false_prune_stages": [str(row.get("stage_name")) for row in false_prunes],
        "local_false_prune_recovered": 1.0 if false_prunes and safe_float(terminal.get("terminal_oracle_retained"), 0.0) > 0 else 0.0,
        "local_false_prune_terminal_failure": 1.0 if false_prunes and safe_float(terminal.get("terminal_oracle_retained"), 0.0) <= 0 else 0.0,
    }


def summarize_numeric(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> dict[str, Any]:
    out = {"count": len(rows)}
    for key in keys:
        out[key] = finite_mean(row.get(key) for row in rows)
    return out


def grouped_summary(rows: Sequence[dict[str, Any]], group_key: str, keys: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(group_key)].append(row)
    out = []
    for value, vals in sorted(grouped.items(), key=lambda item: str(item[0])):
        row = {group_key: value, **summarize_numeric(vals, keys)}
        out.append(row)
    return out


def recovery_rows(task_rows: Sequence[dict[str, Any]], stage_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    task_by_id = {str(row.get("task_id")): row for row in task_rows}
    out = []
    for stage in stage_rows:
        if safe_float(stage.get("stage_false_prune"), 0.0) <= 0:
            continue
        task = task_by_id.get(str(stage.get("task_id")), {})
        out.append(
            {
                "task_id": stage.get("task_id"),
                "domain": stage.get("domain"),
                "stage_name": stage.get("stage_name"),
                "survivors_before": stage.get("survivors_before"),
                "survivors_after": stage.get("survivors_after"),
                "terminal_oracle_retained": task.get("terminal_oracle_retained"),
                "terminal_forced_top1_oracle": task.get("terminal_forced_top1_oracle"),
                "terminal_best_reward": task.get("terminal_best_reward"),
                "terminal_forced_top1_reward": task.get("terminal_forced_top1_reward"),
                "recovered": 1.0 if safe_float(task.get("terminal_oracle_retained"), 0.0) > 0 else 0.0,
            }
        )
    return out


def verdict_for(task_rows: Sequence[dict[str, Any]], stage_retention: float, terminal_policy_summary: Sequence[dict[str, Any]]) -> str:
    forced = next((row for row in terminal_policy_summary if row.get("policy") == "dualanchor_forced_top1"), {})
    gated = next((row for row in terminal_policy_summary if row.get("policy") == "dualanchor_confidence_gated"), {})
    positive_tasks = [row for row in task_rows if safe_float(row.get("positive_oracle"), 0.0) > 0]
    positive_count = len(positive_tasks)
    forced_oracle = safe_float(forced.get("first_selected_oracle"), 0.0)
    gated_retention = safe_float(gated.get("oracle_retained"), 0.0)
    if len(task_rows) < 16 or positive_count < 8:
        size_suffix = "_DATA_LIMITED"
    else:
        size_suffix = ""
    if stage_retention < 0.90:
        return "ARCHITECTURE_LOOPED_SURVIVAL_WEAK" + size_suffix
    if forced_oracle >= 0.85 and positive_count >= 8:
        return "ARCHITECTURE_LOOPED_SURVIVAL_READY_TERMINAL_READY" + size_suffix
    if gated_retention >= 0.95 and forced_oracle >= 0.70:
        return "ARCHITECTURE_LOOPED_SURVIVAL_READY_TERMINAL_CONFIDENCE_ONLY" + size_suffix
    return "ARCHITECTURE_LOOPED_SURVIVAL_READY_TERMINAL_DEFER_REQUIRED" + size_suffix


def main() -> int:
    args = parse_args()
    ensure_root()
    started = time.time()
    selected_tasks, task_inventory = select_stratified_tasks(int(args.max_tasks), str(args.split_mode))
    partial = torch.load(PARTIAL_PT, map_location="cpu", weights_only=False) if PARTIAL_PT.exists() and not args.force else {}
    all_rows: list[dict[str, Any]] = list(partial.get("rows") or [])
    stage_rows: list[dict[str, Any]] = list(partial.get("stage_rows") or [])
    task_rows: list[dict[str, Any]] = list(partial.get("task_rows") or [])
    errors: list[dict[str, Any]] = list(partial.get("errors") or [])
    completed: set[str] = set(str(x) for x in partial.get("completed_task_ids", []))

    policies, _policy_rows = base.load_constrained_policies()
    taps = policies[SELECTED_POLICY]
    bank = load_bank()
    alpha = float(ALPHA_VALUE.get(str(args.alpha_bucket), 0.005))
    extractor: BGTransformerFeatureExtractor | None = None
    final_pool_by_task: dict[str, list[dict[str, Any]]] = {}
    try:
        extractor = BGTransformerFeatureExtractor(device=str(args.device), dtype="auto", force_all_loops=True)
        model = extractor.model
        tokenizer = extractor.tokenizer
        device = extractor.device
        for task_index, task in enumerate(selected_tasks):
            task_id = str(task["task_id"])
            if task_id in completed and not args.force:
                continue
            branch_group_id = f"architecture_looped_stratified_dualanchor::{task.get('split')}::{task_id.replace('/', '__')}"
            task_stage_rows: list[dict[str, Any]] = []
            try:
                root = evaluate_branch(
                    model=model,
                    tokenizer=tokenizer,
                    task=task,
                    branch_group_id=branch_group_id,
                    branch_id=f"{branch_group_id}/root",
                    parent=None,
                    hooks=[],
                    birth_layer=None,
                    birth_loop=None,
                    birth_stage="root",
                    perturbation_family="clean",
                    mutation_index=0,
                    device=device,
                    max_new_tokens=int(args.max_new_tokens),
                )
                survivors: list[dict[str, Any]] = [root]
                all_rows.append(root)
                for stage_index, (loop, layer) in enumerate(NONTERMINAL_STAGES):
                    expanded = list(survivors)
                    for parent_index, parent in enumerate(survivors):
                        seed = 20260531 + 1009 * task_index + 101 * stage_index + 17 * parent_index
                        entries = make_family_deltas(layer, alpha, int(args.children_per_parent) + 1, str(args.delta_family), seed, bank)
                        for entry in entries:
                            if int(entry["branch_id"]) == 0:
                                continue
                            hook = make_hook_entry(entry, layer, loop, alpha, seed)
                            hooks = list(parent.get("hooks") or []) + [hook]
                            child = evaluate_branch(
                                model=model,
                                tokenizer=tokenizer,
                                task=task,
                                branch_group_id=branch_group_id,
                                branch_id=new_child_id(str(parent["branch_id"]), loop, layer, int(entry["branch_id"])),
                                parent=parent,
                                hooks=hooks,
                                birth_layer=layer,
                                birth_loop=loop,
                                birth_stage=f"L{loop}_{layer}",
                                perturbation_family=str(entry.get("delta_family")),
                                mutation_index=int(entry["branch_id"]),
                                device=device,
                                max_new_tokens=int(args.max_new_tokens),
                            )
                            expanded.append(child)
                            all_rows.append(child)
                    survivors, stage = threshold_survive(expanded, taps, layer, loop)
                    stage.update({"task_id": task_id, "domain": task.get("domain"), "split": task.get("split"), "source_dataset": task.get("source_dataset"), "stage_name": f"L{loop}_{layer}", "stage_index": stage_index})
                    stage_rows.append(stage)
                    task_stage_rows.append(stage)
                    save_partial(all_rows, stage_rows, task_rows, errors, completed, task_inventory)
                terminal = terminal_select(survivors, taps)
                final_pool_by_task[task_id] = list(survivors)
                task_row = task_summary(task, survivors, terminal, task_stage_rows)
                task_rows.append(task_row)
                completed.add(task_id)
                save_partial(all_rows, stage_rows, task_rows, errors, completed, task_inventory)
            except Exception as exc:
                errors.append({"task_id": task_id, "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-6000:]})
                save_partial(all_rows, stage_rows, task_rows, errors, completed, task_inventory)
    finally:
        if extractor is not None:
            extractor.cleanup()

    if not final_pool_by_task:
        # Resume-only path: reconstruct final pools from task row branch ids.
        rows_by_id = {str(row.get("branch_id")): row for row in all_rows}
        for task_row in task_rows:
            ids = task_row.get("final_branch_ids") or []
            if isinstance(ids, str):
                try:
                    ids = ast.literal_eval(ids)
                except Exception:
                    ids = []
            final_pool_by_task[str(task_row.get("task_id"))] = [rows_by_id[str(branch_id)] for branch_id in ids if str(branch_id) in rows_by_id]

    tasks_by_id = {str(task.get("task_id")): task for task in selected_tasks}
    terminal_rows: list[dict[str, Any]] = []
    l47_rows: list[dict[str, Any]] = []
    for task_id, final_pool in final_pool_by_task.items():
        if not final_pool:
            continue
        task = tasks_by_id.get(task_id, {"task_id": task_id, "domain": final_pool[0].get("domain")})
        terminal_rows.extend(terminal_policy_rows(task, final_pool, taps))
        l47_rows.extend(l47_ablation_rows(task, final_pool, taps))

    write_csv(ROWS_CSV, [compact_row(row) for row in all_rows])
    write_csv(STAGE_ROWS_CSV, stage_rows)
    write_csv(TASK_ROWS_CSV, task_rows)
    write_csv(TERMINAL_POLICY_CSV, terminal_rows)
    write_csv(L47_ABLATION_CSV, l47_rows)
    rec_rows = recovery_rows(task_rows, stage_rows)
    write_csv(RECOVERY_CSV, rec_rows)

    depth_summary = summarize_by(all_rows, "perturb_count")
    birth_stage_summary = summarize_by([row for row in all_rows if row.get("birth_stage") != "root"], "birth_stage")
    stage_summary = summarize_by(stage_rows, "stage_name")
    task_summary_metrics = summarize_numeric(
        task_rows,
        [
            "terminal_oracle_retained",
            "terminal_best_reward",
            "terminal_forced_top1_reward",
            "terminal_forced_top1_oracle",
            "terminal_confident",
            "terminal_deferred",
            "final_candidate_count",
            "final_perturbed_fraction",
            "positive_oracle",
            "terminal_reward_diverse",
            "local_false_prune_recovered",
            "local_false_prune_terminal_failure",
        ],
    )
    child_rows = [row for row in all_rows if row.get("parent_branch_id")]
    child_summary = {
        "child_count": len(child_rows),
        "child_improved_parent_rate": finite_mean(1.0 if safe_float(row.get("reward_delta_from_parent"), 0.0) > 0 else 0.0 for row in child_rows),
        "child_tied_parent_rate": finite_mean(1.0 if safe_float(row.get("reward_delta_from_parent"), 999.0) == 0 else 0.0 for row in child_rows),
        "child_worse_parent_rate": finite_mean(1.0 if safe_float(row.get("reward_delta_from_parent"), 0.0) < 0 else 0.0 for row in child_rows),
        "mean_child_parent_delta": finite_mean(row.get("reward_delta_from_parent") for row in child_rows),
    }
    stage_oracle_retention = finite_mean(row.get("stage_oracle_retained") for row in stage_rows)
    terminal_policy_summary = grouped_summary(terminal_rows, "policy", ["oracle_retained", "best_selected_reward", "first_selected_reward", "first_selected_oracle", "selected_count"])
    l47_ablation_summary = grouped_summary(l47_rows, "ablation", ["oracle_retained_vs_full", "best_reward", "forced_top1_reward", "forced_top1_oracle", "candidate_count"])
    false_prune_summary = {
        "false_prune_count": len(rec_rows),
        "recovered_count": sum(1 for row in rec_rows if safe_float(row.get("recovered"), 0.0) > 0),
        "recovery_rate": finite_mean(row.get("recovered") for row in rec_rows),
        "terminal_failure_after_local_false_prune": sum(1 for row in rec_rows if safe_float(row.get("recovered"), 0.0) <= 0),
        "false_prune_by_stage": dict(Counter(str(row.get("stage_name")) for row in rec_rows)),
    }
    verdict = verdict_for(task_rows, stage_oracle_retention, terminal_policy_summary)
    payload = {
        "BG_DUALANCHOR_ARCHITECTURE_LOOPED_STRATIFIED_PROBE_VERDICT": verdict,
        "status": verdict,
        "mode": "CUMULATIVE_HOOK_ALL_STAGE_LOOP_APPROX_STRATIFIED",
        "true_fork_carry_claimed": False,
        "action_steering_claimed": False,
        "policy": SELECTED_POLICY,
        "threshold_policy": THRESHOLD_POLICY,
        "guard_policy": GUARD_POLICY,
        "args": vars(args),
        "task_inventory": task_inventory,
        "nonterminal_stages": [f"L{loop}_{layer}" for loop, layer in NONTERMINAL_STAGES],
        "terminal_stage": "L4_47",
        "task_summary": task_summary_metrics,
        "child_summary": child_summary,
        "stage_oracle_retention": stage_oracle_retention,
        "terminal_policy_summary": terminal_policy_summary,
        "l47_ablation_summary": l47_ablation_summary,
        "false_prune_summary": false_prune_summary,
        "depth_summary": depth_summary,
        "birth_stage_summary": birth_stage_summary,
        "stage_summary": stage_summary,
        "task_rows": task_rows,
        "errors": errors,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(REPORT_JSON, payload)
    write_json(SUMMARY_JSON, payload)
    torch.save({"summary": payload, "rows": all_rows, "stage_rows": stage_rows, "task_rows": task_rows, "terminal_rows": terminal_rows, "l47_rows": l47_rows, "recovery_rows": rec_rows, "errors": errors}, ARTIFACT_PT)

    lines = [
        "# DualAnchor Architecture-Looped Stratified Probe v2",
        "",
        f"BG_DUALANCHOR_ARCHITECTURE_LOOPED_STRATIFIED_PROBE_VERDICT = {verdict}",
        "",
        "Mode: `CUMULATIVE_HOOK_ALL_STAGE_LOOP_APPROX_STRATIFIED`. Taps/perturbations run at every nonterminal 24/36/47 stage across loops; only `L4_47` is terminal. This is not true latent fork/carry.",
        "",
        "## Task Selection",
        "",
        f"- selected_count: `{task_inventory['selected_count']}`",
        f"- domain_counts: `{task_inventory['domain_counts']}`",
        f"- split_counts: `{task_inventory['split_counts']}`",
        f"- source_counts: `{task_inventory['source_counts']}`",
        f"- likely_non_tie_count: `{task_inventory['likely_non_tie_count']}`",
        f"- prior_non_tie_positive_count: `{task_inventory['prior_non_tie_positive_count']}`",
        "",
        "## Headline",
        "",
        f"- tasks completed: `{len(task_rows)}`",
        f"- generated/evaluated rows: `{len(all_rows)}`",
        f"- nonterminal stage decisions: `{len(stage_rows)}`",
        f"- stage oracle retention: `{stage_oracle_retention}`",
        f"- terminal oracle retained: `{task_summary_metrics.get('terminal_oracle_retained')}`",
        f"- terminal forced top1 oracle: `{task_summary_metrics.get('terminal_forced_top1_oracle')}`",
        f"- terminal forced top1 reward: `{task_summary_metrics.get('terminal_forced_top1_reward')}`",
        f"- terminal confident rate: `{task_summary_metrics.get('terminal_confident')}`",
        f"- terminal reward-diverse rate: `{task_summary_metrics.get('terminal_reward_diverse')}`",
        f"- positive-oracle rate: `{task_summary_metrics.get('positive_oracle')}`",
        f"- child improved parent rate: `{child_summary['child_improved_parent_rate']}`",
        f"- child tied parent rate: `{child_summary['child_tied_parent_rate']}`",
        f"- child worse parent rate: `{child_summary['child_worse_parent_rate']}`",
        f"- mean child-parent reward delta: `{child_summary['mean_child_parent_delta']}`",
        "",
        "## Terminal Policy Summary",
        "",
    ]
    lines.extend(md_table(terminal_policy_summary, ["policy", "count", "oracle_retained", "best_selected_reward", "first_selected_reward", "first_selected_oracle", "selected_count"]))
    lines.extend(["", "## L47 Candidate-Tree Ablation Summary", ""])
    lines.extend(md_table(l47_ablation_summary, ["ablation", "count", "oracle_retained_vs_full", "best_reward", "forced_top1_reward", "forced_top1_oracle", "candidate_count"]))
    lines.extend(["", "## False-Prune Recovery", ""])
    lines.append(f"- false_prune_count: `{false_prune_summary['false_prune_count']}`")
    lines.append(f"- recovered_count: `{false_prune_summary['recovered_count']}`")
    lines.append(f"- recovery_rate: `{false_prune_summary['recovery_rate']}`")
    lines.append(f"- false_prune_by_stage: `{false_prune_summary['false_prune_by_stage']}`")
    lines.extend(["", "## Task Rows", ""])
    lines.extend(md_table(task_rows, ["task_id", "domain", "split", "source_dataset", "final_candidate_count", "oracle_reward", "positive_oracle", "terminal_reward_diverse", "terminal_oracle_retained", "terminal_best_reward", "terminal_forced_top1_reward", "terminal_forced_top1_oracle", "terminal_confident", "terminal_deferred", "stage_false_prunes", "false_prune_stages"]))
    lines.extend(["", "## By Perturb Count", ""])
    lines.extend(md_table(depth_summary, ["perturb_count", "candidate_count", "mean_reward", "parse_rate", "stable_rate", "mean_parent_delta", "child_improved_parent_rate", "child_tied_parent_rate", "child_worse_parent_rate"]))
    lines.extend(["", "## By Birth Stage", ""])
    lines.extend(md_table(birth_stage_summary, ["birth_stage", "candidate_count", "mean_reward", "parse_rate", "stable_rate", "mean_parent_delta", "child_improved_parent_rate", "child_tied_parent_rate", "child_worse_parent_rate"]))
    lines.extend(["", "## Stage Summary", ""])
    lines.extend(md_table(stage_summary, ["stage_name", "candidate_count", "mean_reward", "parse_rate", "stable_rate"]))
    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend([f"- `{err.get('task_id')}`: {str(err.get('error'))[:500]}" for err in errors[:20]])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This scales the architecture-shaped cumulative-hook probe and separates survival from terminal collapse.",
            "- L47 ablations are candidate-tree replays over generated lineages, not regenerated alternate dynamics.",
            "- A high confidence-gated/defer retention score means terminal top1 should remain gated rather than unconditional.",
            "",
            "## Files",
            "",
            f"- artifact: `{rel(ARTIFACT_PT)}`",
            f"- rows: `{rel(ROWS_CSV)}`",
            f"- stage rows: `{rel(STAGE_ROWS_CSV)}`",
            f"- task rows: `{rel(TASK_ROWS_CSV)}`",
            f"- terminal policy rows: `{rel(TERMINAL_POLICY_CSV)}`",
            f"- L47 ablation rows: `{rel(L47_ABLATION_CSV)}`",
            f"- recovery rows: `{rel(RECOVERY_CSV)}`",
            f"- report json: `{rel(REPORT_JSON)}`",
        ]
    )
    write_md(REPORT_MD, lines)
    write_md(SUMMARY_MD, lines)
    print(f"BG_DUALANCHOR_ARCHITECTURE_LOOPED_STRATIFIED_PROBE_VERDICT = {verdict}", flush=True)
    print(f"tasks_completed = {len(task_rows)} rows = {len(all_rows)}", flush=True)
    print(f"stage_oracle_retention = {stage_oracle_retention}", flush=True)
    print(f"terminal_forced_top1_oracle = {task_summary_metrics.get('terminal_forced_top1_oracle')}", flush=True)
    print(f"terminal_reward_diverse_rate = {task_summary_metrics.get('terminal_reward_diverse')}", flush=True)
    print(f"report = {rel(REPORT_MD)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
