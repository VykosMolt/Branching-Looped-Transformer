"""DualAnchor recursive lineage branch probe v1.

This is a bounded live/cached-hybrid probe for:

    recursive perturb -> DualAnchor pairwise score -> threshold survival -> lineage analysis

It uses cumulative hook interventions to represent lineage inheritance: a child
branch is generated from the original prompt with all parent perturbation hooks
plus a new hook at the child birth layer. This is not true latent fork/carry;
the local true fork/carry interface remains generation-unready.

No Ouro weights, tokenizer files, checkpoints, tap registries, production
routing, action steering, wrapper/local-agent code, Hunter-Seeker modules, or
ARC/MATH loops are modified or executed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import torch

from bg_branch_generator_v1_common import (
    ALPHA_VALUE,
    BASIS_BANK_PT,
    HIDDEN_DIM,
    candidate_pair_stats,
    compact_direction_entry,
    deterministic_reward,
    evaluate_mcq,
    load_audit_plan,
    load_pt,
    make_family_deltas,
    md_table,
    rel,
    row_reward,
    stable_v2_row,
    write_csv,
    write_json,
    write_md,
)
from bg_hidden_origin_tap_common import (
    PROBE_ROOT,
    HiddenCapture,
    capture_hooks,
    make_bg_features_from_pooled,
    masked_mean_pool,
)
from src.evaluator.bg_hidden_branching import HiddenDeltaLayerHook, delta_rms
from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor

import run_bg_dualanchor_all_loop_audit_v1 as base
import run_bg_dualanchor_all_loop_guarded_policy_v1 as guard


OUT_ROOT = PROBE_ROOT / "bg_dualanchor_recursive_lineage_probe_v1_2026-05-31"
REPORT_JSON = OUT_ROOT / "recursive_lineage_probe.json"
REPORT_MD = OUT_ROOT / "recursive_lineage_probe.md"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
ARTIFACT_PT = OUT_ROOT / "dualanchor_recursive_lineage_probe_v1.pt"
ROWS_CSV = OUT_ROOT / "recursive_lineage_rows.csv"
STAGE_ROWS_CSV = OUT_ROOT / "recursive_lineage_stage_rows.csv"
TASK_ROWS_CSV = OUT_ROOT / "recursive_lineage_task_rows.csv"
STATE_JSON = OUT_ROOT / "recursive_lineage_state.json"
PARTIAL_PT = OUT_ROOT / "dualanchor_recursive_lineage_probe_v1.partial.pt"

SELECTED_POLICY = "dualanchor_adaptive_branch_anchor_light_AntisymLinear"
THRESHOLD_POLICY = "mean_floor_very_loose"
GUARD_POLICY = "core_budget8_keep5_conf120"
STAGES = (24, 36, 47)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tasks", type=int, default=4)
    parser.add_argument("--root-k", type=int, default=4)
    parser.add_argument("--child-k", type=int, default=3)
    parser.add_argument("--l47-k", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--alpha-bucket", default="alpha_0_005")
    parser.add_argument("--delta-family", default="old_tap_aligned")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--task-split", default="heldout")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


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


def load_bank() -> dict[str, Any]:
    return load_pt(BASIS_BANK_PT, {"directions_by_layer": {}, "directions": []}) or {"directions_by_layer": {}, "directions": []}


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    skip = {"features", "pooled_vectors", "last_token_vectors", "hooks", "delta"}
    out = {k: v for k, v in row.items() if k not in skip}
    if isinstance(row.get("features"), torch.Tensor):
        out["features_shape"] = list(row["features"].shape)
    if isinstance(row.get("hooks"), list):
        out["hook_count"] = len(row["hooks"])
    return out


def task_sort_key(row: dict[str, Any]) -> tuple[int, str, str]:
    likely = 1 if row.get("heldout_likely_non_tie") else 0
    prior_pairs = int(row.get("prior_non_tie_pairs") or 0)
    priority = float(row.get("priority_score_v4") or row.get("priority_score") or 0.0)
    return (-likely, -prior_pairs, -priority, str(row.get("task_class")), str(row.get("task_id")))


def select_tasks(max_tasks: int, split: str) -> list[dict[str, Any]]:
    plan = load_audit_plan()
    tasks = [
        dict(task)
        for task in plan.get("tasks", [])
        if str(task.get("split")) == split and str(task.get("domain")) in {"reasoning", "science"} and task.get("correct_option")
    ]
    if not tasks and split != "heldout":
        tasks = [
            dict(task)
            for task in plan.get("tasks", [])
            if str(task.get("domain")) in {"reasoning", "science"} and task.get("correct_option")
        ]
    reasoning = sorted([t for t in tasks if str(t.get("domain")) == "reasoning"], key=task_sort_key)
    science = sorted([t for t in tasks if str(t.get("domain")) == "science"], key=task_sort_key)
    selected: list[dict[str, Any]] = []
    while len(selected) < max_tasks and (reasoning or science):
        if reasoning and len(selected) < max_tasks:
            selected.append(reasoning.pop(0))
        if science and len(selected) < max_tasks:
            selected.append(science.pop(0))
        if not reasoning and science:
            while len(selected) < max_tasks and science:
                selected.append(science.pop(0))
        if not science and reasoning:
            while len(selected) < max_tasks and reasoning:
                selected.append(reasoning.pop(0))
    return selected


def apply_hooks(model: Any, hooks: Sequence[dict[str, Any]]) -> list[HiddenDeltaLayerHook]:
    active: list[HiddenDeltaLayerHook] = []
    for item in hooks:
        hook = HiddenDeltaLayerHook(
            model,
            target_layer=int(item["target_layer"]),
            target_loops=[int(item["target_loop"])],
            delta=item["delta"],
            position=-1,
            max_rms_fraction=max(delta_rms(item["delta"]), 0.02),
        )
        hook.apply()
        active.append(hook)
    return active


def remove_hooks(active: Sequence[HiddenDeltaLayerHook]) -> None:
    for hook in reversed(list(active)):
        hook.remove()


def encode_prompt(tokenizer: Any, prompt: str, device: torch.device) -> dict[str, torch.Tensor]:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536, padding=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    if "attention_mask" not in enc:
        enc["attention_mask"] = torch.ones_like(enc["input_ids"], device=device)
    return enc


def capture_features_with_hooks(model: Any, tokenizer: Any, prompt: str, hooks: Sequence[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    enc = encode_prompt(tokenizer, prompt, device)
    capture = HiddenCapture()
    active: list[HiddenDeltaLayerHook] = []
    try:
        active = apply_hooks(model, hooks)
        with torch.inference_mode():
            with capture_hooks(model, capture):
                model(**enc, use_cache=False, logits_to_keep=1)
    finally:
        remove_hooks(active)
    pooled: dict[str, torch.Tensor] = {}
    last: dict[str, torch.Tensor] = {}
    for layer in (24, 30, 36, 42):
        for loop_idx, state in enumerate(capture.layer_states.get(layer, [])[:4], start=1):
            key = f"L{layer}_L{loop_idx}"
            pooled[key] = masked_mean_pool(state, enc["attention_mask"])
            last[key] = state[0, -1, :].detach().cpu().to(torch.float32)
    for loop_idx, state in enumerate(capture.boundary[:4], start=1):
        key = f"L47_L{loop_idx}"
        pooled[key] = masked_mean_pool(state, enc["attention_mask"])
        last[key] = state[0, -1, :].detach().cpu().to(torch.float32)
    hook_diags = [hook.diagnostics() for hook in active]
    return {
        "features": make_bg_features_from_pooled(pooled),
        "pooled_vectors": pooled,
        "last_token_vectors": last,
        "hook_diagnostics": hook_diags,
        "nan_inf": not all(torch.isfinite(v).all().item() for v in pooled.values()),
    }


def generate_with_hooks(
    model: Any,
    tokenizer: Any,
    prompt: str,
    hooks: Sequence[dict[str, Any]],
    device: torch.device,
    max_new_tokens: int,
) -> dict[str, Any]:
    enc = encode_prompt(tokenizer, prompt, device)
    prompt_len = int(enc["input_ids"].shape[1])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    active: list[HiddenDeltaLayerHook] = []
    started = time.time()
    try:
        active = apply_hooks(model, hooks)
        with torch.inference_mode():
            generated = model.generate(
                **enc,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )
    finally:
        hook_diags = [hook.diagnostics() for hook in active]
        remove_hooks(active)
    new_ids = generated[0, prompt_len:]
    return {
        "output_text": tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
        "token_count": int(new_ids.numel()),
        "hit_max_tokens": int(new_ids.numel()) >= int(max_new_tokens),
        "generation_seconds": round(time.time() - started, 3),
        "hook_diagnostics": hook_diags,
    }


def make_hook_entry(delta_entry: dict[str, Any], layer: int, loop: int, alpha: float, seed: int) -> dict[str, Any]:
    return {
        "target_layer": int(layer),
        "target_loop": int(loop),
        "delta": delta_entry["delta"].detach().cpu().to(torch.float32),
        "delta_family": delta_entry.get("delta_family"),
        "delta_type": delta_entry.get("delta_type"),
        "direction_name": delta_entry.get("direction_name"),
        "alpha": float(alpha),
        "seed": int(seed),
        "mutation_index": int(delta_entry["branch_id"]),
        "effective_delta_rms": float(delta_rms(delta_entry["delta"])),
    }


def hook_path_json(hooks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for item in hooks:
        out.append({k: v for k, v in item.items() if k != "delta"})
    return out


def new_child_id(parent_id: str, layer: int, mutation_index: int) -> str:
    return f"{parent_id}/L{layer}_p{int(mutation_index)}"


def evaluate_branch(
    *,
    model: Any,
    tokenizer: Any,
    task: dict[str, Any],
    branch_group_id: str,
    branch_id: str,
    parent: dict[str, Any] | None,
    hooks: Sequence[dict[str, Any]],
    birth_layer: int | None,
    birth_loop: int | None,
    birth_stage: str,
    perturbation_family: str,
    mutation_index: int,
    device: torch.device,
    max_new_tokens: int,
) -> dict[str, Any]:
    features = capture_features_with_hooks(model, tokenizer, task["prompt"], hooks, device)
    gen = generate_with_hooks(model, tokenizer, task["prompt"], hooks, device, max_new_tokens)
    score = evaluate_mcq(task, gen["output_text"])
    parent_reward = row_reward(parent) if parent is not None else float("nan")
    row = {
        "task_id": task["task_id"],
        "domain": task["domain"],
        "source_dataset": task.get("source_dataset"),
        "split": task.get("split"),
        "correct_option": task.get("correct_option"),
        "branch_group_id": branch_group_id,
        "branch_id": branch_id,
        "parent_branch_id": None if parent is None else parent.get("branch_id"),
        "root_branch_id": f"{branch_group_id}/root",
        "generation_depth": len(hooks),
        "perturb_count": len(hooks),
        "birth_layer": birth_layer,
        "birth_loop": birth_loop,
        "birth_stage": birth_stage,
        "perturbation_family": perturbation_family,
        "mutation_index": int(mutation_index),
        "lineage_path": [f"L{h['target_layer']}_L{h['target_loop']}:{h['delta_family']}:{h['direction_name']}:alpha={h['alpha']}:seed={h['seed']}:mutation={h['mutation_index']}" for h in hooks],
        "hooks": list(hooks),
        "hook_path": hook_path_json(hooks),
        "output_text": gen["output_text"],
        "parsed_answer": score["parsed_answer"],
        "correct": bool(score["correct"]),
        "deterministic_correct": bool(score["correct"]),
        "reward": float(score["reward"]),
        "deterministic_reward": float(score["reward"]),
        "parent_reward": parent_reward,
        "reward_delta_from_parent": float(score["reward"]) - parent_reward if math.isfinite(parent_reward) else float("nan"),
        "parse_success": bool(score["parse_success"]),
        "parse_failure_reason": score["parse_failure_reason"],
        "repetition_rate": float(score["repetition_rate"]),
        "empty_output": bool(score["empty_output"]),
        "hit_max_tokens": bool(gen["hit_max_tokens"]),
        "output_length": int(gen["token_count"]),
        "generation_seconds": float(gen["generation_seconds"]),
        "nan_inf": bool(features["nan_inf"]),
        "features": features["features"],
        "pooled_vectors": features["pooled_vectors"],
        "last_token_vectors": features["last_token_vectors"],
        "hook_diagnostics": features["hook_diagnostics"],
        "generation_hook_diagnostics": gen["hook_diagnostics"],
    }
    return row


def threshold_survive(
    candidates: Sequence[dict[str, Any]],
    taps: dict[str, dict[str, dict[str, Any]]],
    layer: int,
    loop: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    threshold_spec = base.PRIMARY_THRESHOLD_POLICIES[THRESHOLD_POLICY]
    guard_spec = guard.GUARD_SPECS[GUARD_POLICY]
    details = guard.stage_score_details(candidates, taps, layer, f"{layer}_L{loop}")
    if not details:
        return list(candidates), {"stage_blocked": 1.0, "survivors_before": len(candidates), "survivors_after": len(candidates)}
    local_keep, guard_info = guard.guarded_keep_indices(candidates, details, loop - 1, layer, threshold_spec, guard_spec)
    survivors = [candidates[i] for i in local_keep]
    rewards = [row_reward(row) for row in candidates]
    oracle = max(rewards) if rewards else float("nan")
    oracle_ids = {idx for idx, value in enumerate(rewards) if value == oracle}
    retained = bool(set(local_keep) & oracle_ids)
    return survivors, {
        "stage_blocked": 0.0,
        "layer": layer,
        "loop": loop,
        "config": f"{layer}_L{loop}",
        "survivors_before": len(candidates),
        "survivors_after": len(survivors),
        "stage_oracle_reward": oracle,
        "stage_oracle_retained": 1.0 if retained else 0.0,
        "stage_false_prune": 0.0 if retained else 1.0,
        **guard_info,
    }


def terminal_select(candidates: Sequence[dict[str, Any]], taps: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    terminal = guard.terminal_gate(candidates, taps, guard.GUARD_SPECS[GUARD_POLICY])
    rewards = [row_reward(row) for row in candidates]
    oracle = max(rewards) if rewards else float("nan")
    oracle_idx = {idx for idx, value in enumerate(rewards) if value == oracle}
    terminal_indices = [int(i) for i in terminal.get("terminal_indices") or [] if int(i) < len(candidates)]
    forced_idx = int(terminal.get("forced_top1_index", 0)) if candidates else 0
    return {
        **terminal,
        "terminal_indices": terminal_indices,
        "forced_top1_index": forced_idx,
        "terminal_oracle_reward": oracle,
        "terminal_oracle_retained": 1.0 if set(terminal_indices) & oracle_idx else 0.0,
        "terminal_best_reward": max((rewards[i] for i in terminal_indices), default=float("nan")),
        "terminal_forced_top1_reward": rewards[forced_idx] if forced_idx < len(rewards) else float("nan"),
        "terminal_forced_top1_oracle": 1.0 if forced_idx in oracle_idx else 0.0,
        "terminal_survivor_count": len(terminal_indices),
    }


def save_partial(all_rows: list[dict[str, Any]], stage_rows: list[dict[str, Any]], task_rows: list[dict[str, Any]], errors: list[dict[str, Any]], completed: set[str]) -> None:
    payload = {
        "rows": all_rows,
        "stage_rows": stage_rows,
        "task_rows": task_rows,
        "errors": errors,
        "completed_task_ids": sorted(completed),
        "saved_at": time.time(),
    }
    torch.save(payload, PARTIAL_PT)
    write_json(STATE_JSON, {"completed_task_ids": sorted(completed), "row_count": len(all_rows), "stage_rows": len(stage_rows), "task_rows": len(task_rows), "errors": len(errors), "saved_at": time.time()})


def summarize(rows: Sequence[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row.get(key)].append(row)
    out = []
    for value, vals in sorted(groups.items(), key=lambda item: str(item[0])):
        out.append(
            {
                key: value,
                "candidate_count": len(vals),
                "mean_reward": finite_mean(row_reward(row) for row in vals),
                "parse_rate": finite_mean(1.0 if row.get("parse_success") else 0.0 for row in vals),
                "stable_rate": finite_mean(1.0 if stable_v2_row(row) else 0.0 for row in vals),
                "mean_parent_delta": finite_mean(row.get("reward_delta_from_parent") for row in vals),
                "child_improved_parent_rate": finite_mean(1.0 if safe_float(row.get("reward_delta_from_parent"), 0.0) > 0 else 0.0 for row in vals if row.get("parent_branch_id")),
                "child_tied_parent_rate": finite_mean(1.0 if safe_float(row.get("reward_delta_from_parent"), 999.0) == 0 else 0.0 for row in vals if row.get("parent_branch_id")),
                "child_worse_parent_rate": finite_mean(1.0 if safe_float(row.get("reward_delta_from_parent"), 0.0) < 0 else 0.0 for row in vals if row.get("parent_branch_id")),
            }
        )
    return out


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        x = float(value)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def task_summary(task_id: str, domain: str, final_pool: Sequence[dict[str, Any]], terminal: dict[str, Any], stage_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rewards = [row_reward(row) for row in final_pool]
    oracle = max(rewards) if rewards else float("nan")
    oracle_rows = [row for row in final_pool if row_reward(row) == oracle]
    pert_frac = finite_mean(1.0 if int(row.get("perturb_count") or 0) > 0 else 0.0 for row in final_pool)
    forced_idx = int(terminal.get("forced_top1_index", 0)) if final_pool else 0
    forced = final_pool[forced_idx] if forced_idx < len(final_pool) else {}
    return {
        "task_id": task_id,
        "domain": domain,
        "final_candidate_count": len(final_pool),
        "final_perturbed_fraction": pert_frac,
        "oracle_reward": oracle,
        "oracle_perturb_counts": dict(Counter(int(row.get("perturb_count") or 0) for row in oracle_rows)),
        "oracle_birth_layers": dict(Counter(str(row.get("birth_layer")) for row in oracle_rows)),
        "terminal_survivor_count": terminal.get("terminal_survivor_count"),
        "terminal_oracle_retained": terminal.get("terminal_oracle_retained"),
        "terminal_best_reward": terminal.get("terminal_best_reward"),
        "terminal_forced_top1_reward": terminal.get("terminal_forced_top1_reward"),
        "terminal_forced_top1_oracle": terminal.get("terminal_forced_top1_oracle"),
        "terminal_confident": terminal.get("terminal_confident"),
        "terminal_deferred": terminal.get("terminal_deferred"),
        "forced_top1_perturb_count": forced.get("perturb_count"),
        "forced_top1_birth_layer": forced.get("birth_layer"),
        "stage_false_prunes": sum(1 for row in stage_rows if safe_float(row.get("stage_false_prune"), 0.0) > 0),
    }


def main() -> int:
    args = parse_args()
    ensure_root()
    started = time.time()
    force = bool(args.force)
    partial = torch.load(PARTIAL_PT, map_location="cpu", weights_only=False) if PARTIAL_PT.exists() and not force else {}
    all_rows: list[dict[str, Any]] = list(partial.get("rows") or [])
    stage_rows: list[dict[str, Any]] = list(partial.get("stage_rows") or [])
    task_rows: list[dict[str, Any]] = list(partial.get("task_rows") or [])
    errors: list[dict[str, Any]] = list(partial.get("errors") or [])
    completed: set[str] = set(str(x) for x in partial.get("completed_task_ids", []))

    policies, _policy_rows = base.load_constrained_policies()
    taps = policies[SELECTED_POLICY]
    bank = load_bank()
    tasks = select_tasks(int(args.max_tasks), str(args.task_split))
    device_name = str(args.device)
    extractor: BGTransformerFeatureExtractor | None = None
    try:
        extractor = BGTransformerFeatureExtractor(device=device_name, dtype="auto", force_all_loops=True)
        model = extractor.model
        tokenizer = extractor.tokenizer
        device = extractor.device
        for task_index, task in enumerate(tasks):
            task_id = str(task["task_id"])
            if task_id in completed and not force:
                continue
            task_stage_rows: list[dict[str, Any]] = []
            branch_group_id = f"recursive_dualanchor::{task.get('split')}::{task_id.replace('/', '__')}"
            try:
                alpha = float(ALPHA_VALUE.get(str(args.alpha_bucket), 0.005))
                root_seed = 20260531 + 1009 * task_index
                root_entries = make_family_deltas(24, alpha, int(args.root_k), str(args.delta_family), root_seed, bank)
                root_candidates: list[dict[str, Any]] = []
                for entry in root_entries:
                    hooks = [] if int(entry["branch_id"]) == 0 else [make_hook_entry(entry, 24, 1, alpha, root_seed)]
                    branch_id = f"{branch_group_id}/root" if not hooks else new_child_id(f"{branch_group_id}/root", 24, int(entry["branch_id"]))
                    row = evaluate_branch(
                        model=model,
                        tokenizer=tokenizer,
                        task=task,
                        branch_group_id=branch_group_id,
                        branch_id=branch_id,
                        parent=None,
                        hooks=hooks,
                        birth_layer=None if not hooks else 24,
                        birth_loop=None if not hooks else 1,
                        birth_stage="root" if not hooks else "L24_L1",
                        perturbation_family="clean" if not hooks else str(entry.get("delta_family")),
                        mutation_index=int(entry["branch_id"]),
                        device=device,
                        max_new_tokens=int(args.max_new_tokens),
                    )
                    root_candidates.append(row)
                    all_rows.append(row)
                survivors, stage = threshold_survive(root_candidates, taps, 24, 1)
                stage.update({"task_id": task_id, "domain": task.get("domain"), "stage_name": "L24_birth"})
                stage_rows.append(stage)
                task_stage_rows.append(stage)

                for layer, k, stage_name in ((36, int(args.child_k), "L36_child"), (47, int(args.l47_k), "L47_child")):
                    expanded: list[dict[str, Any]] = list(survivors)
                    for parent_index, parent in enumerate(survivors):
                        seed = 20260531 + 1009 * task_index + 97 * layer + 13 * parent_index
                        entries = make_family_deltas(layer, alpha, int(k), str(args.delta_family), seed, bank)
                        for entry in entries:
                            if int(entry["branch_id"]) == 0:
                                continue
                            hook = make_hook_entry(entry, layer, 1, alpha, seed)
                            hooks = list(parent.get("hooks") or []) + [hook]
                            child = evaluate_branch(
                                model=model,
                                tokenizer=tokenizer,
                                task=task,
                                branch_group_id=branch_group_id,
                                branch_id=new_child_id(str(parent["branch_id"]), layer, int(entry["branch_id"])),
                                parent=parent,
                                hooks=hooks,
                                birth_layer=layer,
                                birth_loop=1,
                                birth_stage=f"L{layer}_L1",
                                perturbation_family=str(entry.get("delta_family")),
                                mutation_index=int(entry["branch_id"]),
                                device=device,
                                max_new_tokens=int(args.max_new_tokens),
                            )
                            expanded.append(child)
                            all_rows.append(child)
                    if layer == 47:
                        survivors = expanded
                    else:
                        survivors, stage = threshold_survive(expanded, taps, layer, 1)
                        stage.update({"task_id": task_id, "domain": task.get("domain"), "stage_name": stage_name})
                        stage_rows.append(stage)
                        task_stage_rows.append(stage)

                terminal = terminal_select(survivors, taps)
                task_row = task_summary(task_id, str(task.get("domain")), survivors, terminal, task_stage_rows)
                task_rows.append(task_row)
                completed.add(task_id)
                save_partial(all_rows, stage_rows, task_rows, errors, completed)
            except Exception as exc:
                errors.append({"task_id": task_id, "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-6000:]})
                save_partial(all_rows, stage_rows, task_rows, errors, completed)
    finally:
        if extractor is not None:
            extractor.cleanup()

    row_compact = [compact_row(row) for row in all_rows]
    write_csv(ROWS_CSV, row_compact)
    write_csv(STAGE_ROWS_CSV, stage_rows)
    write_csv(TASK_ROWS_CSV, task_rows)

    depth_summary = summarize(all_rows, "perturb_count")
    birth_summary = summarize([row for row in all_rows if row.get("birth_layer") is not None], "birth_layer")
    stage_summary = summarize(stage_rows, "stage_name")
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
    verdict = "RECURSIVE_LINEAGE_DATA_LIMITED"
    if len(task_rows) >= 2 and final_task_summary["terminal_oracle_retained"] >= 0.80:
        verdict = "RECURSIVE_LINEAGE_SURVIVAL_USEFUL_TERMINAL_WEAK"
    if len(task_rows) >= 2 and child_summary["child_improved_parent_rate"] > child_summary["child_worse_parent_rate"]:
        verdict = "RECURSIVE_PERTURB_CHILDREN_USEFUL"
    payload = {
        "BG_DUALANCHOR_RECURSIVE_LINEAGE_PROBE_VERDICT": verdict,
        "status": verdict,
        "mode": "CUMULATIVE_HOOK_LINEAGE_APPROX",
        "true_fork_carry_claimed": False,
        "action_steering_claimed": False,
        "policy": SELECTED_POLICY,
        "threshold_policy": THRESHOLD_POLICY,
        "guard_policy": GUARD_POLICY,
        "args": vars(args),
        "task_summary": final_task_summary,
        "child_summary": child_summary,
        "depth_summary": depth_summary,
        "birth_summary": birth_summary,
        "stage_summary": stage_summary,
        "task_rows": task_rows,
        "errors": errors,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(REPORT_JSON, payload)
    write_json(SUMMARY_JSON, payload)
    torch.save({"summary": payload, "rows": all_rows, "stage_rows": stage_rows, "task_rows": task_rows, "errors": errors}, ARTIFACT_PT)

    lines = [
        "# DualAnchor Recursive Lineage Probe v1",
        "",
        f"BG_DUALANCHOR_RECURSIVE_LINEAGE_PROBE_VERDICT = {verdict}",
        "",
        "Mode: `CUMULATIVE_HOOK_LINEAGE_APPROX`. This is not true latent fork/carry; child rows inherit parent perturbation hooks and add one new hook in a fresh generation pass.",
        "",
        "## Headline",
        "",
        f"- tasks completed: `{len(task_rows)}`",
        f"- generated/evaluated rows: `{len(all_rows)}`",
        f"- terminal oracle retained: `{final_task_summary['terminal_oracle_retained']}`",
        f"- terminal best reward: `{final_task_summary['terminal_best_reward']}`",
        f"- forced terminal top1 reward: `{final_task_summary['terminal_forced_top1_reward']}`",
        f"- forced terminal top1 oracle: `{final_task_summary['terminal_forced_top1_oracle']}`",
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
    lines.extend(md_table(task_rows, ["task_id", "domain", "final_candidate_count", "final_perturbed_fraction", "oracle_reward", "terminal_oracle_retained", "terminal_best_reward", "terminal_forced_top1_reward", "terminal_forced_top1_oracle", "terminal_confident", "terminal_deferred", "forced_top1_perturb_count", "forced_top1_birth_layer"]))
    lines.extend(["", "## By Perturb Count", ""])
    lines.extend(md_table(depth_summary, ["perturb_count", "candidate_count", "mean_reward", "parse_rate", "stable_rate", "mean_parent_delta", "child_improved_parent_rate", "child_tied_parent_rate", "child_worse_parent_rate"]))
    lines.extend(["", "## By Birth Layer", ""])
    lines.extend(md_table(birth_summary, ["birth_layer", "candidate_count", "mean_reward", "parse_rate", "stable_rate", "mean_parent_delta", "child_improved_parent_rate", "child_tied_parent_rate", "child_worse_parent_rate"]))
    lines.extend(["", "## Stage Summary", ""])
    lines.extend(md_table(stage_rows, ["task_id", "domain", "stage_name", "layer", "loop", "survivors_before", "survivors_after", "stage_oracle_retained", "stage_false_prune", "disagreement_guard_active", "budget_trimmed"]))
    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend([f"- `{err.get('task_id')}`: {str(err.get('error'))[:500]}" for err in errors[:20]])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This directly tests whether selected perturbation lineages can produce useful children under cumulative hooks.",
            "- It still does not prove true branch-batch carry or compute savings.",
            "- The terminal L47 top1 remains reported separately from terminal-set retention.",
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
    print(f"BG_DUALANCHOR_RECURSIVE_LINEAGE_PROBE_VERDICT = {verdict}", flush=True)
    print(f"tasks_completed = {len(task_rows)} rows = {len(all_rows)}", flush=True)
    print(f"terminal_oracle_retained = {final_task_summary['terminal_oracle_retained']}", flush=True)
    print(f"child_improved_parent_rate = {child_summary['child_improved_parent_rate']}", flush=True)
    print(f"report = {rel(REPORT_MD)}", flush=True)
    return 0 if errors == [] else 0


if __name__ == "__main__":
    raise SystemExit(main())
