"""Generate quota-directed hidden-origin branch groups v4."""
from __future__ import annotations

import argparse
import os
import time
import traceback
from collections import Counter, defaultdict
from typing import Any

import torch

from bg_hidden_origin_quota_v4_common import (
    ALPHA_VALUE,
    BRANCHES_CSV,
    BRANCHES_JSON,
    BRANCHES_PARTIAL_PT,
    BRANCHES_PT,
    CONTROLLER_JSON,
    GEN_REPORT_JSON,
    GEN_REPORT_MD,
    GEN_STATE_JSON,
    MAX_NEW_TOKENS,
    PROGRESS_JSONL,
    RECIPE_ENGRAMS_JSONL,
    SPLIT_ROW_CAPS,
    TOTAL_ROW_CAP,
    append_jsonl,
    branch_group_metrics,
    candidate_pair_stats,
    compact_v4_row,
    deterministic_reward,
    ensure_v4_root,
    evaluate_mcq,
    finite_float,
    generate_with_hook_v2,
    group_is_behaviorally_diverse_v2,
    group_is_reward_diverse_v2,
    group_rows,
    load_json,
    load_pt,
    load_quota_plan,
    make_family_deltas,
    md_table,
    primary_safe_v4_row,
    read_jsonl,
    rel,
    row_reward,
    score_group_with_head,
    score_old_taps,
    split_deficit_score,
    split_quota_stats,
    stable_v2_row,
    write_csv,
    write_json,
    write_md,
)
from bg_hidden_origin_quota_v4_common import DIRECTION_BANK_V4_PT, load_best_v1_head, load_best_v2_head, load_best_v3_head, best_salvage_head
from bg_hidden_origin_tap_common import capture_prefix_features
from src.evaluator.bg_hidden_branching import delta_rms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-branch-rows", type=int, default=TOTAL_ROW_CAP)
    parser.add_argument("--max-selected-tasks-used", type=int, default=240)
    parser.add_argument("--max-heldout-branch-rows", type=int, default=SPLIT_ROW_CAPS["heldout"])
    parser.add_argument("--max-val-branch-rows", type=int, default=SPLIT_ROW_CAPS["val"])
    parser.add_argument("--max-train-branch-rows", type=int, default=SPLIT_ROW_CAPS["train"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--sample-budget", type=int, default=12)
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def safe_group_id(task_id: str) -> str:
    return str(task_id).replace(" ", "_")


def load_bank() -> dict[str, Any]:
    return load_pt(DIRECTION_BANK_V4_PT, {"directions_by_layer": {}, "directions": []}) or {"directions_by_layer": {}, "directions": []}


def load_partial() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = load_pt(BRANCHES_PARTIAL_PT, {}) or {}
    return list(payload.get("rows") or []), list(payload.get("errors") or [])


def load_completed(rows: list[dict[str, Any]]) -> set[str]:
    completed = set()
    state = load_json(GEN_STATE_JSON, {}) or {}
    completed.update(str(x) for x in state.get("completed_branch_group_ids", []))
    for row in read_jsonl(PROGRESS_JSONL):
        if row.get("status") == "completed" and row.get("branch_group_id"):
            completed.add(str(row["branch_group_id"]))
    for gid, vals in group_rows(rows).items():
        if len([row for row in vals if stable_v2_row(row)]) >= 2:
            completed.add(str(gid))
    return completed


def save_partial(rows: list[dict[str, Any]], errors: list[dict[str, Any]], completed: set[str], complete: bool = False) -> None:
    compact = [compact_v4_row(row) for row in rows]
    quota = split_quota_stats(rows)
    state = {
        "complete": bool(complete),
        "completed_branch_group_ids": sorted(completed),
        "completed_task_ids": sorted({str(row.get("task_id")) for row in rows if str(row.get("branch_group_id")) in completed}),
        "quota_progress": quota,
        "saved_at": time.time(),
    }
    torch.save({"complete": bool(complete), "rows": rows, "errors": errors, **state}, BRANCHES_PARTIAL_PT)
    write_json(GEN_STATE_JSON, {**state, "errors": len(errors)})
    write_json(BRANCHES_JSON, {"complete": bool(complete), "rows": compact, "errors": errors, **state})
    write_csv(BRANCHES_CSV, compact)


def reconcile_completed_progress(rows: list[dict[str, Any]], completed: set[str]) -> None:
    """Backfill progress/engram rows if an interrupt landed after state save."""
    progress_ids = {str(row.get("branch_group_id")) for row in read_jsonl(PROGRESS_JSONL) if row.get("branch_group_id")}
    quota = split_quota_stats(rows)
    for gid, vals in sorted(group_rows(rows).items()):
        if str(gid) not in completed or str(gid) in progress_ids or len(vals) < 2:
            continue
        first = vals[0]
        pair_stats = candidate_pair_stats({"g": vals})
        stable_rows = [row for row in vals if stable_v2_row(row)]
        engram = {
            "recipe_id": first.get("recipe_id"),
            "split": first.get("split"),
            "task_class": first.get("task_class") or first.get("task_screening_class"),
            "branch_point": first.get("branch_point"),
            "K": first.get("K"),
            "alpha_bucket": first.get("alpha_bucket"),
            "delta_family": first.get("primary_delta_family") or first.get("delta_family"),
            "decode_mode": "deterministic",
            "rows_generated": len(vals),
            "behaviorally_diverse_groups": int(group_is_behaviorally_diverse_v2(vals)),
            "reward_diverse_groups": int(group_is_reward_diverse_v2(vals)),
            "non_tie_pairs": pair_stats["non_tie_pairs"],
            "tie_rate": pair_stats["tie_rate"],
            "parse_rate": sum(1 for row in stable_rows if row.get("parse_success")) / max(len(stable_rows), 1),
            "stability_rate": len(stable_rows) / max(len(vals), 1),
            "reward_variance": float(torch.tensor([row_reward(row) for row in vals], dtype=torch.float32).var(unbiased=False).item()),
            "compute_cost": sum(float(row.get("generation_seconds") or 0.0) for row in vals),
            "quota_progress": quota.get(str(first.get("split")), {}),
            "leakage_flags": ["recovered_after_interrupt"],
            "errors": 0,
            "saved_at": time.time(),
        }
        append_jsonl(RECIPE_ENGRAMS_JSONL, engram)
        append_jsonl(
            PROGRESS_JSONL,
            {
                "status": "completed",
                "recovered_after_interrupt": True,
                "branch_group_id": str(gid),
                "task_id": first.get("task_id"),
                "split": first.get("split"),
                "branch_point": first.get("branch_point"),
                "alpha_bucket": first.get("alpha_bucket"),
                "delta_family": first.get("primary_delta_family") or first.get("delta_family"),
                "K": first.get("K"),
                "rows_added": len(vals),
                "errors_added": 0,
                "behaviorally_diverse": group_is_behaviorally_diverse_v2(vals),
                "non_tie_pairs": pair_stats["non_tie_pairs"],
                "quota_progress": quota,
                "saved_at": time.time(),
            },
        )


def controller_recipes() -> list[dict[str, Any]]:
    payload = load_json(CONTROLLER_JSON, {}) or {}
    recipes = list(payload.get("recipes") or [])
    if recipes:
        return recipes
    plan = load_quota_plan()
    out = []
    for task in plan.get("tasks", []):
        for rec in task.get("priority_recipes") or []:
            out.append({"split": task.get("split"), **rec, "recipe_id": f"static::{task.get('split')}::{len(out)}", "current_score": 0.0, "risk_flags": []})
    return out


def branch_point_to_layer(bp: str) -> tuple[str, int]:
    if bp == "L47_diagnostic":
        return "L47", 47
    if str(bp).startswith("L24"):
        return "L24", 24
    if str(bp).startswith("L36"):
        return "L36", 36
    if str(bp).startswith("L47"):
        return "L47", 47
    return "L36", 36


def recipe_is_diagnostic(rec: dict[str, Any]) -> bool:
    bp, _ = branch_point_to_layer(str(rec.get("branch_point")))
    return bp == "L47" or str(rec.get("alpha_bucket")) == "alpha_0_02"


def lineage_fields(group_id: str, delta_entry: dict[str, Any], spec: dict[str, Any], seed: int) -> dict[str, Any]:
    branch_id = int(delta_entry["branch_id"])
    is_clean = branch_id == 0 or str(delta_entry.get("delta_family")) == "clean"
    birth_stage = f"{spec['branch_point']}_L{int(spec['target_loop'])}"
    path_item = (
        f"{birth_stage}:{delta_entry.get('delta_family')}:{delta_entry.get('direction_name')}:"
        f"alpha={float(spec['alpha']):.6g}:seed={int(seed)}:mutation={branch_id}"
    )
    return {
        "parent_branch_id": None if is_clean else 0,
        "root_branch_id": 0,
        "lineage_branch_id": f"{group_id}/b{branch_id}",
        "root_branch_lineage_id": f"{group_id}/b0",
        "parent_branch_lineage_id": None if is_clean else f"{group_id}/b0",
        "generation_depth": 0 if is_clean else 1,
        "perturb_count": 0 if is_clean else 1,
        "birth_layer": None if is_clean else int(spec["target_layer"]),
        "birth_loop": None if is_clean else int(spec["target_loop"]),
        "birth_stage": "root" if is_clean else birth_stage,
        "perturbation_family": "clean" if is_clean else delta_entry.get("delta_family"),
        "perturbation_alpha": 0.0 if is_clean else float(spec["alpha"]),
        "perturbation_rms": float(delta_rms(delta_entry["delta"])),
        "perturbation_seed": int(seed),
        "mutation_index": branch_id,
        "lineage_path": [] if is_clean else [path_item],
    }


def build_queues(plan: dict[str, Any], recipes: list[dict[str, Any]], max_rows: int) -> dict[str, list[dict[str, Any]]]:
    tasks_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in plan.get("tasks", []):
        tasks_by_split[str(task.get("split"))].append(task)
    for vals in tasks_by_split.values():
        vals.sort(key=lambda row: (-float(row.get("priority_score_v4") or 0.0), str(row.get("task_id"))))
    recipes_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in recipes:
        if "direction_family_not_usable" in rec.get("risk_flags", []):
            continue
        recipes_by_split[str(rec.get("split"))].append(rec)
    for split, vals in recipes_by_split.items():
        vals.sort(key=lambda row: (-float(row.get("current_score", row.get("initial_score", 0.0)) or 0.0), str(row.get("recipe_id"))))
    queues: dict[str, list[dict[str, Any]]] = {split: [] for split in ("train", "val", "heldout")}
    for split in ("heldout", "val", "train"):
        tasks = tasks_by_split.get(split, [])
        recs = recipes_by_split.get(split, [])
        if not recs:
            continue
        per_task: list[tuple[int, dict[str, Any], list[dict[str, Any]]]] = []
        for task_idx, task in enumerate(tasks):
            matching = [r for r in recs if str(r.get("task_class")) == str(task.get("task_class"))] or recs
            primary_matching = [r for r in matching if not recipe_is_diagnostic(r)]
            diagnostic_matching = [r for r in matching if recipe_is_diagnostic(r)]
            matching = primary_matching[:8] + diagnostic_matching[:2]
            if matching:
                per_task.append((task_idx, task, matching))
        rows_budget = 0
        l47_count = 0
        round_idx = 0
        while rows_budget < max_rows and round_idx < 20:
            progressed = False
            for recipe_rank in range(8):
                for task_idx, task, matching in per_task:
                    if recipe_rank >= len(matching):
                        continue
                    rec = matching[recipe_rank]
                    bp, layer = branch_point_to_layer(str(rec.get("branch_point")))
                    if bp == "L47":
                        if l47_count >= 24:
                            continue
                        l47_count += 1
                    k = int(rec.get("K") or 6)
                    if rows_budget + k > max_rows:
                        continue
                    alpha_bucket = str(rec.get("alpha_bucket") or "alpha_0_01")
                    group_id = (
                        f"v4::{split}::{safe_group_id(task['task_id'])}::bp={bp}::alpha={alpha_bucket}::"
                        f"family={rec.get('delta_family')}::k={k}::round={round_idx}::recipe={rec.get('recipe_id')}"
                    )
                    queues[split].append(
                        {
                            "split": split,
                            "task": task,
                            "task_index": task_idx,
                            "recipe": rec,
                            "recipe_id": rec.get("recipe_id"),
                            "recipe_source": rec.get("recipe_source") or "hs_inspired_controller",
                            "branch_group_id": group_id,
                            "branch_point": bp,
                            "target_layer": layer,
                            "target_loop": 1,
                            "alpha_bucket": alpha_bucket,
                            "alpha": ALPHA_VALUE.get(alpha_bucket, 0.01),
                            "safety_envelope": alpha_bucket != "alpha_0_02",
                            "K": k,
                            "delta_family": rec.get("delta_family") or "random_orthogonal",
                            "decode_mode": rec.get("decode_mode") or "deterministic",
                            "round_idx": round_idx,
                            "diagnostic": alpha_bucket == "alpha_0_02" or bp == "L47",
                        }
                    )
                    rows_budget += k
                    progressed = True
                    if rows_budget >= max_rows:
                        break
                if rows_budget >= max_rows:
                    break
            if not progressed:
                break
            round_idx += 1
    return queues


def should_sample(task: dict[str, Any], group: list[dict[str, Any]], spec: dict[str, Any]) -> bool:
    if spec.get("branch_point") == "L47" or spec.get("alpha_bucket") == "alpha_0_02":
        return False
    if str(task.get("priority_tier")) == "high":
        return True
    if group_is_behaviorally_diverse_v2(group):
        return True
    return str(task.get("task_class")) in {"perturbation_sensitive", "parse_fragile", "wrong_parseable"}


def add_sampled_rewards(model: Any, tokenizer: Any, task: dict[str, Any], group: list[dict[str, Any]], spec: dict[str, Any], device: torch.device, max_new_tokens: int, remaining_budget: int) -> int:
    used = 0
    if remaining_budget <= 0:
        return used
    targets = sorted(group, key=lambda row: (int(row.get("branch_id", 999)) != 0, int(row.get("branch_id", 999))))[:2]
    branch_spec = {
        "target_layer": int(spec["target_layer"]),
        "target_loop": int(spec["target_loop"]),
        "branch_point": spec["branch_point"],
        "alpha": float(spec["alpha"]),
        "safety_envelope": bool(spec["safety_envelope"]),
    }
    for row in targets:
        if used + 2 > remaining_budget:
            break
        samples = []
        rewards = []
        for sample_id in range(2):
            gen = generate_with_hook_v2(
                model,
                tokenizer,
                task["prompt"],
                row["delta"],
                branch_spec,
                device,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
            )
            score = evaluate_mcq(task, gen["output_text"])
            rewards.append(float(score["reward"]))
            samples.append(
                {
                    "sample_id": sample_id,
                    "output_text": gen["output_text"],
                    "parsed_answer": score["parsed_answer"],
                    "correct": bool(score["correct"]),
                    "reward": float(score["reward"]),
                    "parse_success": bool(score["parse_success"]),
                    "hit_max_tokens": bool(gen["hit_max_tokens"]),
                    "output_length": int(gen["token_count"]),
                }
            )
            used += 1
        if rewards:
            row["sampled_outputs"] = samples
            row["sampled_expected_reward"] = float(sum(rewards) / len(rewards))
    return used


def group_engram(spec: dict[str, Any], group_new: list[dict[str, Any]], errors: list[dict[str, Any]], quota_progress: dict[str, Any]) -> dict[str, Any]:
    groups = {"group": group_new} if len(group_new) >= 2 else {}
    pair_stats = candidate_pair_stats(groups)
    rows_generated = len(group_new)
    stable_rows = [row for row in group_new if stable_v2_row(row)]
    return {
        "recipe_id": spec.get("recipe_id"),
        "split": spec.get("split"),
        "task_class": spec["task"].get("task_class"),
        "branch_point": spec.get("branch_point"),
        "K": spec.get("K"),
        "alpha_bucket": spec.get("alpha_bucket"),
        "delta_family": spec.get("delta_family"),
        "decode_mode": spec.get("decode_mode"),
        "rows_generated": rows_generated,
        "behaviorally_diverse_groups": int(len(group_new) >= 2 and group_is_behaviorally_diverse_v2(group_new)),
        "reward_diverse_groups": int(len(group_new) >= 2 and group_is_reward_diverse_v2(group_new)),
        "non_tie_pairs": pair_stats["non_tie_pairs"],
        "tie_rate": pair_stats["tie_rate"],
        "parse_rate": sum(1 for row in stable_rows if row.get("parse_success")) / max(len(stable_rows), 1),
        "stability_rate": len(stable_rows) / max(rows_generated, 1),
        "reward_variance": float(torch.tensor([row_reward(row) for row in group_new], dtype=torch.float32).var(unbiased=False).item()) if group_new else 0.0,
        "compute_cost": sum(float(row.get("generation_seconds") or 0.0) for row in group_new),
        "quota_progress": quota_progress.get(str(spec.get("split")), {}),
        "leakage_flags": list((spec.get("recipe") or {}).get("risk_flags") or []),
        "errors": len(errors),
        "saved_at": time.time(),
    }


def update_controller_state(engram: dict[str, Any]) -> None:
    payload = load_json(CONTROLLER_JSON, {}) or {}
    history = list(payload.get("allocation_history") or [])
    history.append(engram)
    payload["allocation_history"] = history[-500:]
    recipes = list(payload.get("recipes") or [])
    for rec in recipes:
        if rec.get("recipe_id") != engram.get("recipe_id"):
            continue
        rows = max(int(engram.get("rows_generated") or 0), 1)
        yield_score = 2.0 * float(engram.get("behaviorally_diverse_groups") or 0) + 0.05 * float(engram.get("non_tie_pairs") or 0)
        yield_score += float(engram.get("parse_rate") or 0.0) + float(engram.get("stability_rate") or 0.0)
        yield_score -= 0.2 * len(engram.get("leakage_flags") or [])
        rec["current_score"] = 0.70 * float(rec.get("current_score", rec.get("initial_score", 0.0)) or 0.0) + 0.30 * (100.0 * yield_score / rows)
        rec["updated_from_engrams"] = int(rec.get("updated_from_engrams") or 0) + 1
    payload["recipes"] = recipes
    payload["last_updated_by_generation"] = time.time()
    write_json(CONTROLLER_JSON, payload)


def verdict_for_generation(rows: list[dict[str, Any]], errors: list[dict[str, Any]]) -> str:
    stats = split_quota_stats(rows)
    if not rows and errors:
        return "BLOCKED"
    if stats.get("all_minimums_met"):
        return "QUOTAS_MET"
    heldout_met = bool(stats.get("heldout", {}).get("minimum_met"))
    train_val_met = bool(stats.get("train", {}).get("minimum_met")) and bool(stats.get("val", {}).get("minimum_met"))
    if heldout_met and not train_val_met:
        return "HELDOUT_QUOTA_MET_ONLY"
    if train_val_met and not heldout_met:
        return "TRAIN_VAL_QUOTA_MET_ONLY"
    stable_rates = [float(stats.get(split, {}).get("stability_rate") or 0.0) for split in ("train", "val", "heldout")]
    if rows and min(stable_rates or [1.0]) < 0.50:
        return "UNSTABLE"
    if rows and sum(int(stats.get(split, {}).get("behaviorally_diverse_groups") or 0) for split in ("train", "val", "heldout")) == 0:
        return "LOW_DIVERSITY"
    return "PARTIAL"


def write_report(verdict: str, rows: list[dict[str, Any]], errors: list[dict[str, Any]], elapsed: float, selected_task_count: int, sampled_used: int) -> None:
    stats = split_quota_stats(rows)
    payload = {
        "BG_HIDDEN_ORIGIN_QUOTA_GENERATION_V4_VERDICT": verdict,
        "verdict": verdict,
        "quota_progress_by_split": stats,
        "row_count": len(rows),
        "errors": errors,
        "selected_task_count": selected_task_count,
        "completed_branch_group_ids": sorted({str(row.get("branch_group_id")) for row in rows}),
        "completed_task_ids": sorted({str(row.get("task_id")) for row in rows}),
        "sampled_outputs": sampled_used,
        "counts_by_split": dict(Counter(str(row.get("split")) for row in rows)),
        "counts_by_branch_point": dict(Counter(str(row.get("branch_point")) for row in rows)),
        "counts_by_alpha_bucket": dict(Counter(str(row.get("alpha_bucket")) for row in rows)),
        "counts_by_delta_family": dict(Counter(str(row.get("primary_delta_family") or row.get("delta_family")) for row in rows)),
        "checkpointing": {
            "progress_jsonl": rel(PROGRESS_JSONL),
            "state_json": rel(GEN_STATE_JSON),
            "partial_pt": rel(BRANCHES_PARTIAL_PT),
            "resumable": True,
            "skip_completed_branch_groups_unless_FORCE_RERUN": True,
        },
        "elapsed_seconds": round(elapsed, 3),
    }
    write_json(GEN_REPORT_JSON, payload)
    display = []
    for gid, vals in sorted(group_rows([row for row in rows if primary_safe_v4_row(row)]).items())[:220]:
        if len(vals) < 2:
            continue
        display.append(
            {
                "branch_group_id": gid,
                "split": vals[0].get("split"),
                "task_id": vals[0].get("task_id"),
                "branch_point": vals[0].get("branch_point"),
                "alpha": vals[0].get("alpha_bucket"),
                "family": vals[0].get("primary_delta_family") or vals[0].get("delta_family"),
                "K": vals[0].get("K"),
                "rewards": sorted({deterministic_reward(row) for row in vals}),
                "answers": sorted({str(row.get("parsed_answer")) for row in vals}),
                "diverse": group_is_behaviorally_diverse_v2(vals),
                "non_tie_pairs": candidate_pair_stats({"g": vals})["non_tie_pairs"],
            }
        )
    lines = [
        "# Hidden-Origin Quota Generation V4",
        "",
        f"BG_HIDDEN_ORIGIN_QUOTA_GENERATION_V4_VERDICT = {verdict}",
        "",
        f"- row_count: `{len(rows)}`",
        f"- sampled_outputs: `{sampled_used}`",
        f"- errors: `{len(errors)}`",
        f"- quota_progress_by_split: `{stats}`",
        "",
        "Alpha 0.02, sampled expected reward, and L47 branches are diagnostic only and not counted toward primary selector readiness.",
        "",
        "## Primary-Safe Groups",
        "",
    ]
    lines.extend(md_table(display, ["branch_group_id", "split", "task_id", "branch_point", "alpha", "family", "K", "rewards", "answers", "diverse", "non_tie_pairs"]))
    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- `{e.get('task_id')}` `{e.get('branch_group_id')}` b{e.get('branch_id')}: {str(e.get('error'))[:240]}" for e in errors[:80])
    write_md(GEN_REPORT_MD, lines)


def main() -> int:
    args = parse_args()
    ensure_v4_root()
    started = time.time()
    rows, errors = load_partial()
    completed = load_completed(rows)
    reconcile_completed_progress(rows, completed)
    if args.finalize_only:
        verdict = verdict_for_generation(rows, errors)
        save_partial(rows, errors, completed, complete=True)
        torch.save({"rows": rows, "errors": errors, "verdict": verdict, "complete": True}, BRANCHES_PT)
        write_report(verdict, rows, errors, time.time() - started, len({str(row.get("task_id")) for row in rows}), sum(len(row.get("sampled_outputs") or []) for row in rows))
        print(f"BG_HIDDEN_ORIGIN_QUOTA_GENERATION_V4_VERDICT = {verdict}", flush=True)
        return 0 if verdict != "BLOCKED" else 1
    plan = load_quota_plan()
    bank = load_bank()
    if plan.get("verdict") == "BLOCKED" or not bank.get("directions_by_layer"):
        verdict = "BLOCKED"
        errors.append({"error": "missing usable quota plan or direction bank"})
        save_partial(rows, errors, completed, complete=False)
        write_report(verdict, rows, errors, time.time() - started, 0, 0)
        print("BG_HIDDEN_ORIGIN_QUOTA_GENERATION_V4_VERDICT = BLOCKED", flush=True)
        return 1
    caps = {"train": int(args.max_train_branch_rows), "val": int(args.max_val_branch_rows), "heldout": int(args.max_heldout_branch_rows)}
    queues = build_queues(plan, controller_recipes(), int(args.max_branch_rows))
    selected_task_count = len({str(task.get("task_id")) for task in plan.get("tasks", [])})
    force = os.environ.get("FORCE_RERUN") == "1"
    sampled_used = sum(len(row.get("sampled_outputs") or []) for row in rows)
    v1_head = load_best_v1_head()
    v2_head = load_best_v2_head()
    v3_head = load_best_v3_head()
    salvage_head = best_salvage_head()
    from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor

    extractor = None
    try:
        extractor = BGTransformerFeatureExtractor(device=args.device, dtype="auto", force_all_loops=True)
        model = extractor.model
        tokenizer = extractor.tokenizer
        device = extractor.device
        while len(rows) < int(args.max_branch_rows):
            quota = split_quota_stats(rows)
            if quota.get("all_minimums_met"):
                break
            split_counts = Counter(str(row.get("split")) for row in rows)
            available_splits = [s for s in ("heldout", "val", "train") if queues.get(s) and split_counts[s] < caps[s]]
            if not available_splits:
                break
            split = max(available_splits, key=lambda s: (split_deficit_score(quota, s), s == "heldout", s == "val"))
            spec = queues[split].pop(0)
            if spec.get("diagnostic") and not bool(quota.get(split, {}).get("minimum_met")) and any(not bool(item.get("diagnostic")) for item in queues.get(split, [])):
                queues[split].append(spec)
                continue
            group_id = str(spec["branch_group_id"])
            if group_id in completed and not force:
                continue
            task = spec["task"]
            seed = (
                20260518
                + 7919 * int(spec["task_index"])
                + 101 * int(spec["target_layer"])
                + int(float(spec["alpha"]) * 100000)
                + 17 * int(spec["round_idx"])
                + len(completed)
            )
            delta_entries = make_family_deltas(
                int(spec["target_layer"]),
                float(spec["alpha"]),
                int(spec["K"]),
                str(spec["delta_family"]),
                seed,
                bank,
            )
            branch_spec = {
                "target_layer": int(spec["target_layer"]),
                "target_loop": int(spec["target_loop"]),
                "branch_point": spec["branch_point"],
                "alpha": float(spec["alpha"]),
                "safety_envelope": bool(spec["safety_envelope"]),
            }
            group_new: list[dict[str, Any]] = []
            group_errors: list[dict[str, Any]] = []
            for delta_entry in delta_entries:
                if len(rows) + len(group_new) >= int(args.max_branch_rows):
                    break
                try:
                    prefix = capture_prefix_features(model, tokenizer, task["prompt"], delta_entry["delta"], branch_spec, device)
                    gen = generate_with_hook_v2(
                        model,
                        tokenizer,
                        task["prompt"],
                        delta_entry["delta"],
                        branch_spec,
                        device,
                        max_new_tokens=int(args.max_new_tokens),
                        do_sample=False,
                    )
                    score = evaluate_mcq(task, gen["output_text"])
                    row = {
                        "split": split,
                        "task_id": task["task_id"],
                        "domain": task["domain"],
                        "source_dataset": task.get("source_dataset"),
                        "source_subject": task.get("source_subject"),
                        "subdomain_bucket": task.get("subdomain_bucket"),
                        "task_screening_class": task.get("task_screening_class") or task.get("task_class"),
                        "task_class": task.get("task_class"),
                        "priority_score": task.get("priority_score_v4"),
                        "priority_tier": task.get("priority_tier"),
                        "recipe_id": spec.get("recipe_id"),
                        "recipe_source": spec.get("recipe_source"),
                        "branch_group_id": group_id,
                        "branch_id": int(delta_entry["branch_id"]),
                        **lineage_fields(group_id, delta_entry, spec, seed),
                        "branch_method": "hook_intervention_per_branch",
                        "branch_point": spec["branch_point"],
                        "target_layer": int(spec["target_layer"]),
                        "target_loop": int(spec["target_loop"]),
                        "alpha": float(spec["alpha"]),
                        "alpha_bucket": spec["alpha_bucket"],
                        "safety_envelope": bool(spec["safety_envelope"]),
                        "diagnostic_alpha": bool(spec["alpha_bucket"] == "alpha_0_02"),
                        "diagnostic_l47": bool(spec["branch_point"] == "L47"),
                        "K": int(spec["K"]),
                        "primary_delta_family": spec["delta_family"],
                        "delta_family": delta_entry["delta_family"],
                        "delta_type": delta_entry["delta_type"],
                        "direction_name": delta_entry["direction_name"],
                        "effective_delta_rms": float(delta_rms(delta_entry["delta"])),
                        "delta": delta_entry["delta"].detach().cpu().to(torch.float32),
                        "features": prefix["features"],
                        "pooled_vectors": prefix["pooled_vectors"],
                        "last_token_vectors": prefix["last_token_vectors"],
                        "output_text": gen["output_text"],
                        "parsed_answer": score["parsed_answer"],
                        "correct": bool(score["correct"]),
                        "deterministic_correct": bool(score["correct"]),
                        "reward": float(score["reward"]),
                        "deterministic_reward": float(score["reward"]),
                        "sampled_expected_reward": None,
                        "label_source": "deterministic",
                        "parse_success": bool(score["parse_success"]),
                        "parse_failure_reason": score["parse_failure_reason"],
                        "repetition_rate": float(score["repetition_rate"]),
                        "empty_output": bool(score["empty_output"]),
                        "hit_max_tokens": bool(gen["hit_max_tokens"]),
                        "output_length": int(gen["token_count"]),
                        "generation_seconds": float(gen["generation_seconds"]),
                        "hook_modifications": int(gen["hook_diagnostics"].get("modifications", 0)),
                        "prefix_hook_modifications": int(prefix["hook_diagnostics"].get("modifications", 0)),
                        "cuda_error": "",
                        "nan_inf": bool(prefix["nan_inf"]),
                    }
                    group_new.append(row)
                except Exception as exc:
                    err = {
                        "task_id": task.get("task_id"),
                        "branch_group_id": group_id,
                        "branch_id": int(delta_entry.get("branch_id", -1)),
                        "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:],
                    }
                    errors.append(err)
                    group_errors.append(err)
            if group_new:
                score_old_taps(group_new)
                for row in group_new:
                    row["old_frozen_tap_score"] = float(row.get("tap_margin_sum", 0.0))
                score_group_with_head(group_new, v1_head, score_key="v1_tap_score", device=device)
                score_group_with_head(group_new, v2_head, score_key="v2_tap_score", device=device)
                score_group_with_head(group_new, v3_head, score_key="v3_tap_score", device=device)
                score_group_with_head(group_new, salvage_head, score_key="salvage_tap_score", device=device)
                rows.extend(group_new)
                completed.add(group_id)
                save_partial(rows, errors, completed, complete=False)
                if should_sample(task, group_new, spec) and sampled_used < int(args.sample_budget):
                    try:
                        sampled_used += add_sampled_rewards(model, tokenizer, task, group_new, spec, device, int(args.max_new_tokens), int(args.sample_budget) - sampled_used)
                    except Exception as exc:
                        err = {
                            "task_id": task.get("task_id"),
                            "branch_group_id": group_id,
                            "branch_id": "sampled_group",
                            "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:],
                        }
                        errors.append(err)
                        group_errors.append(err)
                save_partial(rows, errors, completed, complete=False)
            else:
                completed.add(group_id)
            quota = split_quota_stats(rows)
            engram = group_engram(spec, group_new, group_errors, quota)
            append_jsonl(RECIPE_ENGRAMS_JSONL, engram)
            update_controller_state(engram)
            append_jsonl(
                PROGRESS_JSONL,
                {
                    "status": "completed",
                    "branch_group_id": group_id,
                    "task_id": task.get("task_id"),
                    "split": split,
                    "branch_point": spec["branch_point"],
                    "alpha_bucket": spec["alpha_bucket"],
                    "delta_family": spec["delta_family"],
                    "K": spec["K"],
                    "rows_added": len(group_new),
                    "errors_added": len(group_errors),
                    "behaviorally_diverse": group_is_behaviorally_diverse_v2(group_new) if len(group_new) >= 2 else False,
                    "non_tie_pairs": candidate_pair_stats({"g": group_new})["non_tie_pairs"] if len(group_new) >= 2 else 0,
                    "quota_progress": quota,
                    "saved_at": time.time(),
                },
            )
            save_partial(rows, errors, completed, complete=False)
    except KeyboardInterrupt:
        verdict = verdict_for_generation(rows, errors)
        save_partial(rows, errors, completed, complete=False)
        write_report(verdict, rows, errors, time.time() - started, selected_task_count, sampled_used)
        raise
    finally:
        if extractor is not None:
            extractor.cleanup()
    verdict = verdict_for_generation(rows, errors)
    payload = {
        "BG_HIDDEN_ORIGIN_QUOTA_GENERATION_V4_VERDICT": verdict,
        "verdict": verdict,
        "rows": rows,
        "errors": errors,
        "completed_branch_group_ids": sorted(completed),
        "completed_task_ids": sorted({str(row.get("task_id")) for row in rows}),
        "sampled_outputs": sampled_used,
        "max_branch_rows": int(args.max_branch_rows),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(payload, BRANCHES_PT)
    save_partial(rows, errors, completed, complete=True)
    write_report(verdict, rows, errors, time.time() - started, selected_task_count, sampled_used)
    print(f"BG_HIDDEN_ORIGIN_QUOTA_GENERATION_V4_VERDICT = {verdict}", flush=True)
    print(f"Wrote {rel(BRANCHES_PT)}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
