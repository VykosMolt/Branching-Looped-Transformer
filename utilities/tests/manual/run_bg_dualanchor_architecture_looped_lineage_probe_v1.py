"""DualAnchor architecture-looped lineage probe v1.

This probe follows the intended loop architecture more closely than the first
recursive lineage probe:

- taps operate at layers 24, 36, and 47 in every loop,
- every nonterminal stage can spawn perturbation children,
- survivors continue downward through the current loop or back to layer 24 in
  the next loop after layer 47,
- only L4_47 is terminal.

Implementation note: this is a cumulative-hook lineage approximation. A child
branch is generated from the original prompt with all parent perturbation hooks
plus the new stage hook. It is not true latent fork/carry and does not claim
compute savings beyond this bounded probe.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import torch

from bg_branch_generator_v1_common import ALPHA_VALUE, BASIS_BANK_PT, load_pt, make_family_deltas, md_table, rel, row_reward, stable_v2_row, write_csv, write_json, write_md
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
    select_tasks,
    terminal_select,
    threshold_survive,
)


OUT_ROOT = PROBE_ROOT / "bg_dualanchor_architecture_looped_lineage_probe_v1_2026-05-31"
REPORT_JSON = OUT_ROOT / "architecture_looped_lineage_probe.json"
REPORT_MD = OUT_ROOT / "architecture_looped_lineage_probe.md"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
ARTIFACT_PT = OUT_ROOT / "dualanchor_architecture_looped_lineage_probe_v1.pt"
PARTIAL_PT = OUT_ROOT / "dualanchor_architecture_looped_lineage_probe_v1.partial.pt"
STATE_JSON = OUT_ROOT / "architecture_looped_lineage_state.json"
ROWS_CSV = OUT_ROOT / "architecture_looped_lineage_rows.csv"
STAGE_ROWS_CSV = OUT_ROOT / "architecture_looped_stage_rows.csv"
TASK_ROWS_CSV = OUT_ROOT / "architecture_looped_task_rows.csv"

SELECTED_POLICY = "dualanchor_adaptive_branch_anchor_light_AntisymLinear"
THRESHOLD_POLICY = "mean_floor_very_loose"
GUARD_POLICY = "core_budget8_keep5_conf120"
NONTERMINAL_STAGES = tuple((loop, layer) for loop in (1, 2, 3, 4) for layer in (24, 36, 47) if not (loop == 4 and layer == 47))
TERMINAL_STAGE = (4, 47)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tasks", type=int, default=2)
    parser.add_argument("--children-per-parent", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--alpha-bucket", default="alpha_0_005")
    parser.add_argument("--delta-family", default="old_tap_aligned")
    parser.add_argument("--task-split", default="heldout")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_root() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def load_bank() -> dict[str, Any]:
    return load_pt(BASIS_BANK_PT, {"directions_by_layer": {}, "directions": []}) or {"directions_by_layer": {}, "directions": []}


def new_child_id(parent_id: str, loop: int, layer: int, mutation_index: int) -> str:
    return f"{parent_id}/L{loop}_{layer}_p{int(mutation_index)}"


def save_partial(rows: list[dict[str, Any]], stage_rows: list[dict[str, Any]], task_rows: list[dict[str, Any]], errors: list[dict[str, Any]], completed: set[str]) -> None:
    payload = {
        "rows": rows,
        "stage_rows": stage_rows,
        "task_rows": task_rows,
        "errors": errors,
        "completed_task_ids": sorted(completed),
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


def task_summary(task: dict[str, Any], final_pool: Sequence[dict[str, Any]], terminal: dict[str, Any], task_stage_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rewards = [row_reward(row) for row in final_pool]
    oracle = max(rewards) if rewards else float("nan")
    oracle_rows = [row for row in final_pool if row_reward(row) == oracle]
    forced_idx = int(terminal.get("forced_top1_index", 0)) if final_pool else 0
    forced = final_pool[forced_idx] if forced_idx < len(final_pool) else {}
    return {
        "task_id": task.get("task_id"),
        "domain": task.get("domain"),
        "final_candidate_count": len(final_pool),
        "final_perturbed_fraction": finite_mean(1.0 if int(row.get("perturb_count") or 0) > 0 else 0.0 for row in final_pool),
        "oracle_reward": oracle,
        "oracle_perturb_counts": dict(Counter(int(row.get("perturb_count") or 0) for row in oracle_rows)),
        "oracle_birth_stages": dict(Counter(str(row.get("birth_stage")) for row in oracle_rows)),
        "terminal_survivor_count": terminal.get("terminal_survivor_count"),
        "terminal_oracle_retained": terminal.get("terminal_oracle_retained"),
        "terminal_best_reward": terminal.get("terminal_best_reward"),
        "terminal_forced_top1_reward": terminal.get("terminal_forced_top1_reward"),
        "terminal_forced_top1_oracle": terminal.get("terminal_forced_top1_oracle"),
        "terminal_confident": terminal.get("terminal_confident"),
        "terminal_deferred": terminal.get("terminal_deferred"),
        "forced_top1_perturb_count": forced.get("perturb_count"),
        "forced_top1_birth_stage": forced.get("birth_stage"),
        "stage_false_prunes": sum(1 for row in task_stage_rows if safe_float(row.get("stage_false_prune"), 0.0) > 0),
    }


def main() -> int:
    args = parse_args()
    ensure_root()
    started = time.time()
    partial = torch.load(PARTIAL_PT, map_location="cpu", weights_only=False) if PARTIAL_PT.exists() and not args.force else {}
    all_rows: list[dict[str, Any]] = list(partial.get("rows") or [])
    stage_rows: list[dict[str, Any]] = list(partial.get("stage_rows") or [])
    task_rows: list[dict[str, Any]] = list(partial.get("task_rows") or [])
    errors: list[dict[str, Any]] = list(partial.get("errors") or [])
    completed: set[str] = set(str(x) for x in partial.get("completed_task_ids", []))

    policies, _policy_rows = base.load_constrained_policies()
    taps = policies[SELECTED_POLICY]
    bank = load_bank()
    tasks = select_tasks(int(args.max_tasks), str(args.task_split))
    alpha = float(ALPHA_VALUE.get(str(args.alpha_bucket), 0.005))
    extractor: BGTransformerFeatureExtractor | None = None
    try:
        extractor = BGTransformerFeatureExtractor(device=str(args.device), dtype="auto", force_all_loops=True)
        model = extractor.model
        tokenizer = extractor.tokenizer
        device = extractor.device
        for task_index, task in enumerate(tasks):
            task_id = str(task["task_id"])
            if task_id in completed and not args.force:
                continue
            branch_group_id = f"architecture_looped_dualanchor::{task.get('split')}::{task_id.replace('/', '__')}"
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
                    stage.update({"task_id": task_id, "domain": task.get("domain"), "stage_name": f"L{loop}_{layer}", "stage_index": stage_index})
                    stage_rows.append(stage)
                    task_stage_rows.append(stage)
                    save_partial(all_rows, stage_rows, task_rows, errors, completed)
                terminal = terminal_select(survivors, taps)
                task_row = task_summary(task, survivors, terminal, task_stage_rows)
                task_rows.append(task_row)
                completed.add(task_id)
                save_partial(all_rows, stage_rows, task_rows, errors, completed)
            except Exception as exc:
                errors.append({"task_id": task_id, "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-6000:]})
                save_partial(all_rows, stage_rows, task_rows, errors, completed)
    finally:
        if extractor is not None:
            extractor.cleanup()

    write_csv(ROWS_CSV, [compact_row(row) for row in all_rows])
    write_csv(STAGE_ROWS_CSV, stage_rows)
    write_csv(TASK_ROWS_CSV, task_rows)
    depth_summary = summarize_by(all_rows, "perturb_count")
    birth_stage_summary = summarize_by([row for row in all_rows if row.get("birth_stage") != "root"], "birth_stage")
    stage_summary = summarize_by(stage_rows, "stage_name")
    final_task_summary = {
        "task_count": len(task_rows),
        "terminal_oracle_retained": finite_mean(row.get("terminal_oracle_retained") for row in task_rows),
        "terminal_best_reward": finite_mean(row.get("terminal_best_reward") for row in task_rows),
        "terminal_forced_top1_reward": finite_mean(row.get("terminal_forced_top1_reward") for row in task_rows),
        "terminal_forced_top1_oracle": finite_mean(row.get("terminal_forced_top1_oracle") for row in task_rows),
        "terminal_confident": finite_mean(row.get("terminal_confident") for row in task_rows),
        "terminal_deferred": finite_mean(row.get("terminal_deferred") for row in task_rows),
        "final_candidate_count": finite_mean(row.get("final_candidate_count") for row in task_rows),
        "final_perturbed_fraction": finite_mean(row.get("final_perturbed_fraction") for row in task_rows),
    }
    child_rows = [row for row in all_rows if row.get("parent_branch_id")]
    child_summary = {
        "child_count": len(child_rows),
        "child_improved_parent_rate": finite_mean(1.0 if safe_float(row.get("reward_delta_from_parent"), 0.0) > 0 else 0.0 for row in child_rows),
        "child_tied_parent_rate": finite_mean(1.0 if safe_float(row.get("reward_delta_from_parent"), 999.0) == 0 else 0.0 for row in child_rows),
        "child_worse_parent_rate": finite_mean(1.0 if safe_float(row.get("reward_delta_from_parent"), 0.0) < 0 else 0.0 for row in child_rows),
        "mean_child_parent_delta": finite_mean(row.get("reward_delta_from_parent") for row in child_rows),
    }
    stage_oracle_retention = finite_mean(row.get("stage_oracle_retained") for row in stage_rows)
    verdict = "ARCHITECTURE_LOOPED_DATA_LIMITED"
    if len(task_rows) >= 2 and stage_oracle_retention >= 0.95 and final_task_summary["terminal_oracle_retained"] >= 0.80:
        verdict = "ARCHITECTURE_LOOPED_SURVIVAL_READY_TERMINAL_WEAK"
    if len(task_rows) >= 2 and final_task_summary["terminal_forced_top1_oracle"] >= 0.85:
        verdict = "ARCHITECTURE_LOOPED_TERMINAL_READY_SMALL_N"
    payload = {
        "BG_DUALANCHOR_ARCHITECTURE_LOOPED_LINEAGE_PROBE_VERDICT": verdict,
        "status": verdict,
        "mode": "CUMULATIVE_HOOK_ALL_STAGE_LOOP_APPROX",
        "true_fork_carry_claimed": False,
        "action_steering_claimed": False,
        "policy": SELECTED_POLICY,
        "threshold_policy": THRESHOLD_POLICY,
        "guard_policy": GUARD_POLICY,
        "args": vars(args),
        "nonterminal_stages": [f"L{loop}_{layer}" for loop, layer in NONTERMINAL_STAGES],
        "terminal_stage": "L4_47",
        "task_summary": final_task_summary,
        "child_summary": child_summary,
        "stage_oracle_retention": stage_oracle_retention,
        "depth_summary": depth_summary,
        "birth_stage_summary": birth_stage_summary,
        "stage_summary": stage_summary,
        "task_rows": task_rows,
        "errors": errors,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(REPORT_JSON, payload)
    write_json(SUMMARY_JSON, payload)
    torch.save({"summary": payload, "rows": all_rows, "stage_rows": stage_rows, "task_rows": task_rows, "errors": errors}, ARTIFACT_PT)

    lines = [
        "# DualAnchor Architecture-Looped Lineage Probe v1",
        "",
        f"BG_DUALANCHOR_ARCHITECTURE_LOOPED_LINEAGE_PROBE_VERDICT = {verdict}",
        "",
        "Mode: `CUMULATIVE_HOOK_ALL_STAGE_LOOP_APPROX`. Taps/perturbations run at every nonterminal 24/36/47 stage across loops; only `L4_47` is terminal. This is not true latent fork/carry.",
        "",
        "## Headline",
        "",
        f"- tasks completed: `{len(task_rows)}`",
        f"- generated/evaluated rows: `{len(all_rows)}`",
        f"- nonterminal stages per task: `{len(NONTERMINAL_STAGES)}`",
        f"- stage oracle retention: `{stage_oracle_retention}`",
        f"- terminal oracle retained: `{final_task_summary['terminal_oracle_retained']}`",
        f"- terminal forced top1 oracle: `{final_task_summary['terminal_forced_top1_oracle']}`",
        f"- terminal forced top1 reward: `{final_task_summary['terminal_forced_top1_reward']}`",
        f"- terminal confident rate: `{final_task_summary['terminal_confident']}`",
        f"- avg final candidates: `{final_task_summary['final_candidate_count']}`",
        f"- child improved parent rate: `{child_summary['child_improved_parent_rate']}`",
        f"- child tied parent rate: `{child_summary['child_tied_parent_rate']}`",
        f"- child worse parent rate: `{child_summary['child_worse_parent_rate']}`",
        f"- mean child-parent reward delta: `{child_summary['mean_child_parent_delta']}`",
        "",
        "## Task Rows",
        "",
    ]
    lines.extend(md_table(task_rows, ["task_id", "domain", "final_candidate_count", "final_perturbed_fraction", "oracle_reward", "terminal_oracle_retained", "terminal_best_reward", "terminal_forced_top1_reward", "terminal_forced_top1_oracle", "terminal_confident", "terminal_deferred", "forced_top1_perturb_count", "forced_top1_birth_stage", "stage_false_prunes"]))
    lines.extend(["", "## By Perturb Count", ""])
    lines.extend(md_table(depth_summary, ["perturb_count", "candidate_count", "mean_reward", "parse_rate", "stable_rate", "mean_parent_delta", "child_improved_parent_rate", "child_tied_parent_rate", "child_worse_parent_rate"]))
    lines.extend(["", "## By Birth Stage", ""])
    lines.extend(md_table(birth_stage_summary, ["birth_stage", "candidate_count", "mean_reward", "parse_rate", "stable_rate", "mean_parent_delta", "child_improved_parent_rate", "child_tied_parent_rate", "child_worse_parent_rate"]))
    lines.extend(["", "## Stage Rows", ""])
    lines.extend(md_table(stage_rows, ["task_id", "domain", "stage_name", "survivors_before", "survivors_after", "stage_oracle_retained", "stage_false_prune", "disagreement_guard_active", "budget_trimmed"]))
    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend([f"- `{err.get('task_id')}`: {str(err.get('error'))[:500]}" for err in errors[:20]])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is the closest current probe to the intended architecture without true fork/carry: all nonterminal layers in all loops can branch and prune.",
            "- `L4_47` is terminal only; earlier L47 stages are normal branch/evaluator stages and feed the next loop.",
            "- The report separates survival quality from forced terminal top1.",
            "",
            "## Files",
            "",
            f"- artifact: `{rel(ARTIFACT_PT)}`",
            f"- rows: `{rel(ROWS_CSV)}`",
            f"- stage rows: `{rel(STAGE_ROWS_CSV)}`",
            f"- task rows: `{rel(TASK_ROWS_CSV)}`",
            f"- report json: `{rel(REPORT_JSON)}`",
        ]
    )
    write_md(REPORT_MD, lines)
    write_md(SUMMARY_MD, lines)
    print(f"BG_DUALANCHOR_ARCHITECTURE_LOOPED_LINEAGE_PROBE_VERDICT = {verdict}", flush=True)
    print(f"tasks_completed = {len(task_rows)} rows = {len(all_rows)}", flush=True)
    print(f"stage_oracle_retention = {stage_oracle_retention}", flush=True)
    print(f"terminal_oracle_retained = {final_task_summary['terminal_oracle_retained']}", flush=True)
    print(f"terminal_forced_top1_oracle = {final_task_summary['terminal_forced_top1_oracle']}", flush=True)
    print(f"report = {rel(REPORT_MD)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
