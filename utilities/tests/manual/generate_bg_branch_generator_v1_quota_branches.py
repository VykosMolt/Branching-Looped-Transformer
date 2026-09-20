"""Generate quota-directed hidden-origin branches with Branch Generator v1."""
from __future__ import annotations

import argparse
import os
import time
import traceback
from collections import Counter, defaultdict
from typing import Any

import torch

from bg_branch_generator_v1_common import (
    ALPHA_VALUE,
    BASIS_BANK_PT,
    BEST_SCHEDULE_JSON,
    BRANCHES_CSV,
    BRANCHES_JSON,
    BRANCHES_PARTIAL_PT,
    BRANCHES_PT,
    GEN_REPORT_JSON,
    GEN_REPORT_MD,
    GEN_STATE_JSON,
    PROGRESS_JSONL,
    RECIPE_ENGRAMS_JSONL,
    SPLIT_ROW_CAPS_V1,
    TOTAL_ROW_CAP_V1,
    append_jsonl,
    best_salvage_head,
    best_v4_head,
    candidate_pair_stats,
    compact_generator_row,
    config_vector_from_row,
    deterministic_reward,
    ensure_bgv1_root,
    evaluate_mcq,
    generate_with_hook_v2,
    group_is_behaviorally_diverse_v2,
    group_is_reward_diverse_v2,
    group_rows,
    load_audit_plan,
    load_best_v1_head,
    load_best_v2_head,
    load_best_v3_head,
    load_json,
    load_pt,
    make_family_deltas,
    md_table,
    primary_safe_generator_row,
    read_jsonl,
    rel,
    row_reward,
    score_group_with_head,
    score_old_taps,
    split_deficit_score_v1,
    split_quota_stats_v1,
    stable_v2_row,
    verdict_for_generation_v1,
    write_csv,
    write_json,
    write_md,
)
from bg_hidden_origin_tap_common import capture_prefix_features
from src.evaluator.bg_hidden_branching import delta_rms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-branch-rows", type=int, default=TOTAL_ROW_CAP_V1)
    parser.add_argument("--max-selected-tasks-used", type=int, default=260)
    parser.add_argument("--max-heldout-branch-rows", type=int, default=SPLIT_ROW_CAPS_V1["heldout"])
    parser.add_argument("--max-val-branch-rows", type=int, default=SPLIT_ROW_CAPS_V1["val"])
    parser.add_argument("--max-train-branch-rows", type=int, default=SPLIT_ROW_CAPS_V1["train"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--sample-budget", type=int, default=16)
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def safe_group_id(task_id: str) -> str:
    return str(task_id).replace(" ", "_").replace("/", "__")


def branch_point_to_layer(bp: str) -> tuple[str, int]:
    if bp == "L47_diagnostic" or str(bp).startswith("L47"):
        return "L47", 47
    if str(bp).startswith("L24"):
        return "L24", 24
    if str(bp).startswith("L36"):
        return "L36", 36
    return "L24", 24


def recipe_is_diagnostic(rec: dict[str, Any]) -> bool:
    bp, _ = branch_point_to_layer(str(rec.get("branch_point")))
    return bp == "L47" or str(rec.get("alpha_bucket")) == "alpha_0_02" or bool(rec.get("diagnostic"))


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


def load_bank() -> dict[str, Any]:
    return load_pt(BASIS_BANK_PT, {"directions_by_layer": {}, "directions": []}) or {"directions_by_layer": {}, "directions": []}


def load_partial() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = load_pt(BRANCHES_PARTIAL_PT, {}) or {}
    return list(payload.get("rows") or []), list(payload.get("errors") or [])


def load_completed(rows: list[dict[str, Any]]) -> set[str]:
    completed: set[str] = set()
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
    compact = [compact_generator_row(row) for row in rows]
    quota = split_quota_stats_v1(rows)
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


def schedule_recipes(plan: dict[str, Any]) -> list[dict[str, Any]]:
    schedule = load_json(BEST_SCHEDULE_JSON, {}) or {}
    recipes = list(schedule.get("recipes") or [])
    if recipes:
        return recipes
    out = []
    for task in plan.get("tasks", []):
        for rec in task.get("priority_recipes") or []:
            out.append({"split": task.get("split"), "task_id": task.get("task_id"), **rec, "current_score": 0.0})
    return out


def build_queues(plan: dict[str, Any], recipes: list[dict[str, Any]], max_rows: int) -> dict[str, list[dict[str, Any]]]:
    tasks_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in plan.get("tasks", []):
        tasks_by_split[str(task.get("split"))].append(task)
    for vals in tasks_by_split.values():
        vals.sort(key=lambda row: (-float(row.get("priority_score_v4") or 0.0), str(row.get("task_id"))))
    recipes_by_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in recipes:
        recipes_by_task[(str(rec.get("split")), str(rec.get("task_id")))].append(rec)
    for vals in recipes_by_task.values():
        vals.sort(key=lambda row: (recipe_is_diagnostic(row), -float(row.get("current_score", row.get("initial_score", 0.0)) or 0.0), str(row.get("recipe_id"))))
    queues: dict[str, list[dict[str, Any]]] = {split: [] for split in ("train", "val", "heldout")}
    per_split_budget = {
        "train": min(max_rows, SPLIT_ROW_CAPS_V1["train"]),
        "val": min(max_rows, SPLIT_ROW_CAPS_V1["val"]),
        "heldout": min(max_rows, SPLIT_ROW_CAPS_V1["heldout"]),
    }
    for split in ("heldout", "val", "train"):
        rows_budget = 0
        l47_count = 0
        for round_idx in range(24):
            progressed = False
            for recipe_rank in range(10):
                for task_idx, task in enumerate(tasks_by_split.get(split, [])):
                    matching = recipes_by_task.get((split, str(task.get("task_id"))), []) or list(task.get("priority_recipes") or [])
                    if recipe_rank >= len(matching):
                        continue
                    rec = matching[recipe_rank]
                    bp, layer = branch_point_to_layer(str(rec.get("branch_point")))
                    if bp == "L47":
                        if l47_count >= 24:
                            continue
                        l47_count += 1
                    k = int(rec.get("K") or 8)
                    if rows_budget + k > per_split_budget[split]:
                        continue
                    alpha_bucket = str(rec.get("alpha_bucket") or "alpha_0_005")
                    generator_method = str(rec.get("generator_method") or rec.get("search_method") or rec.get("recipe_source") or "static_v4_best_recipe")
                    group_id = (
                        f"bgv1::{split}::{safe_group_id(str(task['task_id']))}::method={generator_method}::bp={bp}::"
                        f"alpha={alpha_bucket}::family={rec.get('delta_family')}::k={k}::round={round_idx}::rank={recipe_rank}"
                    )
                    queues[split].append(
                        {
                            "split": split,
                            "task": task,
                            "task_index": task_idx,
                            "recipe": rec,
                            "recipe_id": rec.get("recipe_id") or group_id,
                            "recipe_source": rec.get("recipe_source") or "branch_generator_v1_schedule",
                            "generator_method": generator_method,
                            "branch_group_id": group_id,
                            "branch_point": bp,
                            "target_layer": layer,
                            "target_loop": int(rec.get("target_loop") or 1),
                            "alpha_bucket": alpha_bucket,
                            "alpha": ALPHA_VALUE.get(alpha_bucket, 0.005),
                            "safety_envelope": alpha_bucket != "alpha_0_02",
                            "K": k,
                            "delta_family": rec.get("delta_family") or "old_tap_aligned",
                            "decode_mode": rec.get("decode_mode") or "deterministic",
                            "round_idx": round_idx,
                            "diagnostic": alpha_bucket == "alpha_0_02" or bp == "L47",
                        }
                    )
                    rows_budget += k
                    progressed = True
                    if rows_budget >= per_split_budget[split]:
                        break
                if rows_budget >= per_split_budget[split]:
                    break
            if rows_budget >= per_split_budget[split] or not progressed:
                break
    return queues


def should_sample(task: dict[str, Any], group: list[dict[str, Any]], spec: dict[str, Any]) -> bool:
    if spec.get("branch_point") == "L47" or spec.get("alpha_bucket") == "alpha_0_02":
        return False
    return str(task.get("priority_tier")) == "high" or group_is_behaviorally_diverse_v2(group) or str(task.get("task_class")) in {"perturbation_sensitive", "parse_fragile", "wrong_parseable"}


def add_sampled_rewards(model: Any, tokenizer: Any, task: dict[str, Any], group: list[dict[str, Any]], spec: dict[str, Any], device: torch.device, max_new_tokens: int, remaining_budget: int) -> int:
    used = 0
    if remaining_budget <= 0:
        return 0
    branch_spec = {"target_layer": int(spec["target_layer"]), "target_loop": int(spec["target_loop"]), "branch_point": spec["branch_point"], "alpha": float(spec["alpha"]), "safety_envelope": bool(spec["safety_envelope"])}
    targets = sorted(group, key=lambda row: (int(row.get("branch_id", 999)) != 0, int(row.get("branch_id", 999))))[:2]
    for row in targets:
        if used + 2 > remaining_budget:
            break
        rewards = []
        samples = []
        for sample_id in range(2):
            gen = generate_with_hook_v2(model, tokenizer, task["prompt"], row["delta"], branch_spec, device, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.95)
            score = evaluate_mcq(task, gen["output_text"])
            rewards.append(float(score["reward"]))
            samples.append({"sample_id": sample_id, "output_text": gen["output_text"], "parsed_answer": score["parsed_answer"], "correct": bool(score["correct"]), "reward": float(score["reward"]), "parse_success": bool(score["parse_success"]), "hit_max_tokens": bool(gen["hit_max_tokens"]), "output_length": int(gen["token_count"])})
            used += 1
        if rewards:
            row["sampled_outputs"] = samples
            row["sampled_expected_reward"] = float(sum(rewards) / len(rewards))
    return used


def add_rich_group_diagnostics(group: list[dict[str, Any]]) -> None:
    clean = next((row for row in group if int(row.get("branch_id", -1)) == 0), None)
    for row in group:
        cfg = "24_L4" if str(row.get("branch_point")) == "L24" else "36_L4" if str(row.get("branch_point")) == "L36" else "47_L4"
        vec = config_vector_from_row(row, cfg)
        clean_vec = config_vector_from_row(clean, cfg) if clean else None
        if isinstance(vec, torch.Tensor):
            row["branch_feature_norm_rms"] = float(torch.sqrt(torch.mean(vec.detach().cpu().float() ** 2)).item())
        else:
            row["branch_feature_norm_rms"] = None
        if isinstance(vec, torch.Tensor) and isinstance(clean_vec, torch.Tensor) and vec.numel() == clean_vec.numel():
            row["branch_hidden_distance_from_clean"] = float(torch.sqrt(torch.mean((vec.detach().cpu().float() - clean_vec.detach().cpu().float()) ** 2)).item())
        else:
            row["branch_hidden_distance_from_clean"] = None
        row["branch_logit_kl_from_clean"] = None
        row["option_logit_margin"] = None
        row["option_entropy"] = None
        row["off_manifold_warning"] = bool(row.get("nan_inf")) or float(row.get("effective_delta_rms") or 0.0) > 0.05


def group_engram(spec: dict[str, Any], group_new: list[dict[str, Any]], errors: list[dict[str, Any]], quota_progress: dict[str, Any]) -> dict[str, Any]:
    groups = {"group": group_new} if len(group_new) >= 2 else {}
    pair_stats = candidate_pair_stats(groups)
    stable_rows = [row for row in group_new if stable_v2_row(row)]
    rewards = torch.tensor([row_reward(row) for row in group_new], dtype=torch.float32) if group_new else torch.zeros(1)
    return {
        "recipe_id": spec.get("recipe_id"),
        "split": spec.get("split"),
        "task_class": spec["task"].get("task_class"),
        "branch_point": spec.get("branch_point"),
        "K": spec.get("K"),
        "alpha_bucket": spec.get("alpha_bucket"),
        "delta_family": spec.get("delta_family"),
        "generator_method": spec.get("generator_method"),
        "coefficient_summary": "basis_direction_or_random_family",
        "decode_mode": spec.get("decode_mode"),
        "rows_generated": len(group_new),
        "behaviorally_diverse_groups": int(len(group_new) >= 2 and group_is_behaviorally_diverse_v2(group_new)),
        "reward_diverse_groups": int(len(group_new) >= 2 and group_is_reward_diverse_v2(group_new)),
        "non_tie_pairs": pair_stats["non_tie_pairs"],
        "tie_rate": pair_stats["tie_rate"],
        "parse_rate": sum(1 for row in stable_rows if row.get("parse_success")) / max(len(stable_rows), 1),
        "stability_rate": len(stable_rows) / max(len(group_new), 1),
        "reward_variance": float(rewards.var(unbiased=False).item()) if group_new else 0.0,
        "compute_cost": sum(float(row.get("generation_seconds") or 0.0) for row in group_new),
        "quota_progress": quota_progress.get(str(spec.get("split")), {}),
        "leakage_flags": list((spec.get("recipe") or {}).get("risk_flags") or []),
        "errors": len(errors),
        "saved_at": time.time(),
    }


def write_report(verdict: str, rows: list[dict[str, Any]], errors: list[dict[str, Any]], elapsed: float, sampled_used: int) -> None:
    stats = split_quota_stats_v1(rows)
    payload = {
        "BG_BRANCH_GENERATOR_V1_GENERATION_VERDICT": verdict,
        "verdict": verdict,
        "quota_progress_by_split": stats,
        "row_count": len(rows),
        "errors": errors,
        "sampled_outputs": sampled_used,
        "counts_by_split": dict(Counter(str(row.get("split")) for row in rows)),
        "counts_by_generator_method": dict(Counter(str(row.get("generator_method")) for row in rows)),
        "counts_by_branch_point": dict(Counter(str(row.get("branch_point")) for row in rows)),
        "counts_by_alpha_bucket": dict(Counter(str(row.get("alpha_bucket")) for row in rows)),
        "counts_by_delta_family": dict(Counter(str(row.get("primary_delta_family") or row.get("delta_family")) for row in rows)),
        "checkpointing": {"progress_jsonl": rel(PROGRESS_JSONL), "state_json": rel(GEN_STATE_JSON), "partial_pt": rel(BRANCHES_PARTIAL_PT), "resumable": True, "skip_completed_branch_groups_unless_FORCE_RERUN": True},
        "elapsed_seconds": round(elapsed, 3),
    }
    write_json(GEN_REPORT_JSON, payload)
    display = []
    for gid, vals in sorted(group_rows([row for row in rows if primary_safe_generator_row(row)]).items())[:260]:
        if len(vals) < 2:
            continue
        display.append(
            {
                "branch_group_id": gid,
                "split": vals[0].get("split"),
                "task_id": vals[0].get("task_id"),
                "method": vals[0].get("generator_method"),
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
        "# Branch Generator V1 Quota Generation",
        "",
        f"BG_BRANCH_GENERATOR_V1_GENERATION_VERDICT = {verdict}",
        "",
        f"- row_count: `{len(rows)}`",
        f"- sampled_outputs: `{sampled_used}`",
        f"- errors: `{len(errors)}`",
        f"- quota_progress_by_split: `{stats}`",
        "",
        "Readiness remains limited to primary-safe deterministic alpha <= 0.01 reasoning/science same-prefix rows.",
        "",
        "## Primary-Safe Groups",
        "",
    ]
    lines.extend(md_table(display, ["branch_group_id", "split", "task_id", "method", "branch_point", "alpha", "family", "K", "rewards", "answers", "diverse", "non_tie_pairs"]))
    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- `{e.get('task_id')}` `{e.get('branch_group_id')}` b{e.get('branch_id')}: {str(e.get('error'))[:240]}" for e in errors[:120])
    write_md(GEN_REPORT_MD, lines)


def main() -> int:
    args = parse_args()
    ensure_bgv1_root()
    started = time.time()
    rows, errors = load_partial()
    completed = load_completed(rows)
    if args.finalize_only:
        verdict = verdict_for_generation_v1(rows, errors)
        save_partial(rows, errors, completed, complete=True)
        torch.save({"rows": rows, "errors": errors, "verdict": verdict, "complete": True}, BRANCHES_PT)
        write_report(verdict, rows, errors, time.time() - started, sum(len(row.get("sampled_outputs") or []) for row in rows))
        print(f"BG_BRANCH_GENERATOR_V1_GENERATION_VERDICT = {verdict}", flush=True)
        return 0 if verdict != "BLOCKED" else 1
    plan = load_audit_plan()
    bank = load_bank()
    if plan.get("verdict") == "BLOCKED" or not bank.get("directions_by_layer"):
        verdict = "BLOCKED"
        errors.append({"error": "missing usable audit plan or basis bank"})
        save_partial(rows, errors, completed, complete=False)
        write_report(verdict, rows, errors, time.time() - started, 0)
        print("BG_BRANCH_GENERATOR_V1_GENERATION_VERDICT = BLOCKED", flush=True)
        return 1
    caps = {"train": int(args.max_train_branch_rows), "val": int(args.max_val_branch_rows), "heldout": int(args.max_heldout_branch_rows)}
    queues = build_queues(plan, schedule_recipes(plan), int(args.max_branch_rows))
    force = os.environ.get("FORCE_RERUN") == "1"
    sampled_used = sum(len(row.get("sampled_outputs") or []) for row in rows)
    v1_head = load_best_v1_head()
    v2_head = load_best_v2_head()
    v3_head = load_best_v3_head()
    v4_head = best_v4_head()
    salvage_head = best_salvage_head()
    from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor

    extractor = None
    try:
        extractor = BGTransformerFeatureExtractor(device=args.device, dtype="auto", force_all_loops=True)
        model = extractor.model
        tokenizer = extractor.tokenizer
        device = extractor.device
        while len(rows) < int(args.max_branch_rows):
            quota = split_quota_stats_v1(rows)
            if quota.get("all_minimums_met"):
                break
            split_counts = Counter(str(row.get("split")) for row in rows)
            available_splits = [s for s in ("heldout", "val", "train") if queues.get(s) and split_counts[s] < caps[s]]
            if not available_splits:
                break
            split = max(available_splits, key=lambda s: (split_deficit_score_v1(quota, s), s == "heldout", s == "val"))
            spec = queues[split].pop(0)
            if spec.get("diagnostic") and not bool(quota.get(split, {}).get("minimum_met")) and any(not bool(item.get("diagnostic")) for item in queues.get(split, [])):
                queues[split].append(spec)
                continue
            group_id = str(spec["branch_group_id"])
            if group_id in completed and not force:
                continue
            task = spec["task"]
            seed = 20260518 + 911 * int(spec["task_index"]) + 101 * int(spec["target_layer"]) + int(float(spec["alpha"]) * 100000) + 19 * int(spec["round_idx"]) + len(completed)
            delta_entries = make_family_deltas(int(spec["target_layer"]), float(spec["alpha"]), int(spec["K"]), str(spec["delta_family"]), seed, bank)
            branch_spec = {"target_layer": int(spec["target_layer"]), "target_loop": int(spec["target_loop"]), "branch_point": spec["branch_point"], "alpha": float(spec["alpha"]), "safety_envelope": bool(spec["safety_envelope"])}
            group_new: list[dict[str, Any]] = []
            group_errors: list[dict[str, Any]] = []
            for delta_entry in delta_entries:
                if len(rows) + len(group_new) >= int(args.max_branch_rows):
                    break
                try:
                    prefix = capture_prefix_features(model, tokenizer, task["prompt"], delta_entry["delta"], branch_spec, device)
                    gen = generate_with_hook_v2(model, tokenizer, task["prompt"], delta_entry["delta"], branch_spec, device, max_new_tokens=int(args.max_new_tokens), do_sample=False)
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
                        "generator_method": spec.get("generator_method"),
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
                        "low_rank_coeff_summary": "basis_family_delta",
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
                    err = {"task_id": task.get("task_id"), "branch_group_id": group_id, "branch_id": int(delta_entry.get("branch_id", -1)), "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:]}
                    errors.append(err)
                    group_errors.append(err)
            if group_new:
                score_old_taps(group_new)
                for row in group_new:
                    row["old_frozen_tap_score"] = float(row.get("tap_margin_sum", 0.0))
                score_group_with_head(group_new, v1_head, score_key="v1_tap_score", device=device)
                score_group_with_head(group_new, v2_head, score_key="v2_tap_score", device=device)
                score_group_with_head(group_new, v3_head, score_key="v3_tap_score", device=device)
                score_group_with_head(group_new, v4_head, score_key="v4_tap_score", device=device)
                score_group_with_head(group_new, salvage_head, score_key="salvage_tap_score", device=device)
                add_rich_group_diagnostics(group_new)
                rows.extend(group_new)
                completed.add(group_id)
                save_partial(rows, errors, completed, complete=False)
                if should_sample(task, group_new, spec) and sampled_used < int(args.sample_budget):
                    try:
                        sampled_used += add_sampled_rewards(model, tokenizer, task, group_new, spec, device, int(args.max_new_tokens), int(args.sample_budget) - sampled_used)
                    except Exception as exc:
                        err = {"task_id": task.get("task_id"), "branch_group_id": group_id, "branch_id": "sampled_group", "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:]}
                        errors.append(err)
                        group_errors.append(err)
                save_partial(rows, errors, completed, complete=False)
            else:
                completed.add(group_id)
            quota = split_quota_stats_v1(rows)
            engram = group_engram(spec, group_new, group_errors, quota)
            append_jsonl(RECIPE_ENGRAMS_JSONL, engram)
            append_jsonl(PROGRESS_JSONL, {"status": "completed", "branch_group_id": group_id, "task_id": task.get("task_id"), "split": split, "generator_method": spec.get("generator_method"), "branch_point": spec["branch_point"], "alpha_bucket": spec["alpha_bucket"], "delta_family": spec["delta_family"], "K": spec["K"], "rows_added": len(group_new), "errors_added": len(group_errors), "behaviorally_diverse": group_is_behaviorally_diverse_v2(group_new) if len(group_new) >= 2 else False, "non_tie_pairs": candidate_pair_stats({"g": group_new})["non_tie_pairs"] if len(group_new) >= 2 else 0, "quota_progress": quota, "saved_at": time.time()})
            save_partial(rows, errors, completed, complete=False)
    except KeyboardInterrupt:
        verdict = verdict_for_generation_v1(rows, errors)
        save_partial(rows, errors, completed, complete=False)
        write_report(verdict, rows, errors, time.time() - started, sampled_used)
        raise
    finally:
        if extractor is not None:
            extractor.cleanup()
    verdict = verdict_for_generation_v1(rows, errors)
    payload = {"rows": rows, "errors": errors, "verdict": verdict, "complete": True, "completed_branch_group_ids": sorted(completed), "completed_task_ids": sorted({str(row.get("task_id")) for row in rows}), "sampled_outputs": sampled_used, "max_branch_rows": int(args.max_branch_rows), "elapsed_seconds": round(time.time() - started, 3)}
    torch.save(payload, BRANCHES_PT)
    save_partial(rows, errors, completed, complete=True)
    write_report(verdict, rows, errors, time.time() - started, sampled_used)
    print(f"BG_BRANCH_GENERATOR_V1_GENERATION_VERDICT = {verdict}", flush=True)
    print(f"Wrote {rel(BRANCHES_PT)}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
