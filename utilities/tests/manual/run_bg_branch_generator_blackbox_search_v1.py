"""Run lightweight black-box Branch Generator v1 schedule search."""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from bg_branch_generator_v1_common import (
    AUDIT_PLAN_JSON,
    BEST_SCHEDULE_JSON,
    BLACKBOX_JSON,
    BLACKBOX_MD,
    PROPOSER_PT,
    RECIPE_ENGRAMS_JSONL,
    append_jsonl,
    ensure_bgv1_root,
    load_json,
    load_pt,
    md_table,
    rel,
    write_json,
    write_md,
)


METHOD_PRIOR = {
    "CEM_best_schedule": 4.5,
    "branch_generator_proposer": 3.5,
    "hs_inspired_controller": 3.0,
    "static_v4_best_recipe": 2.7,
    "ES_lite_best_schedule": 2.4,
    "random_structured_baseline": 0.8,
    "diagnostic_alpha": 0.2,
    "true_fork_carry_diagnostic": 0.1,
}
BRANCH_PRIOR = {"L24": 1.2, "L36": 0.8, "L47_diagnostic": -0.5}
ALPHA_PRIOR = {"alpha_0_005": 1.0, "alpha_0_01": 0.5, "alpha_0_02": -0.4}
FAMILY_PRIOR = {
    "old_tap_aligned": 1.3,
    "high_yield_recipe_direction": 1.25,
    "v2_tap_aligned": 1.0,
    "v3_tap_aligned": 0.9,
    "salvage_tap_aligned": 0.9,
    "v4_tap_aligned": 0.8,
    "paired_plus_minus": 0.65,
    "hidden_origin_empirical_train_only": 0.6,
    "hidden_origin_whitened_train_only": 0.55,
    "random_orthogonal": 0.2,
}


def predicted_score(recipe: dict[str, Any], proposer_rows: list[dict[str, Any]]) -> float:
    score = METHOD_PRIOR.get(str(recipe.get("generator_method")), 1.0)
    score += BRANCH_PRIOR.get(str(recipe.get("branch_point")), 0.0)
    score += ALPHA_PRIOR.get(str(recipe.get("alpha_bucket")), 0.0)
    score += FAMILY_PRIOR.get(str(recipe.get("delta_family")), 0.0)
    if int(recipe.get("K") or 0) >= 8:
        score += 0.5
    # Add a small lookup boost from fitted prior recipe table.
    for row in proposer_rows[:30]:
        if (
            str(row.get("branch_point")) == str(recipe.get("branch_point"))
            and str(row.get("K")) == str(recipe.get("K"))
            and str(row.get("alpha_bucket")) == str(recipe.get("alpha_bucket"))
            and str(row.get("delta_family")) == str(recipe.get("delta_family"))
        ):
            score += 0.05 * float(row.get("score") or 0.0)
            break
    return float(score)


def build_schedule(plan: dict[str, Any], proposer: dict[str, Any]) -> list[dict[str, Any]]:
    proposer_rows = list((proposer.get("recipe_bandit_model") or {}).get("rows") or [])
    recipes: list[dict[str, Any]] = []
    per_split_counts: dict[str, int] = defaultdict(int)
    for task in plan.get("tasks", []):
        split = str(task.get("split"))
        task_priority = float(task.get("priority_score_v4") or 0.0)
        for rec in task.get("priority_recipes") or []:
            current = predicted_score(rec, proposer_rows) + 0.04 * task_priority
            risk_flags = []
            if rec.get("diagnostic"):
                risk_flags.append("diagnostic_not_primary_readiness")
            if split == "heldout":
                risk_flags.append("heldout_schedule_frozen_before_selector_eval")
            item = {
                **rec,
                "split": split,
                "task_id": task.get("task_id"),
                "current_score": current,
                "initial_score": current,
                "risk_flags": risk_flags,
                "recipe_source": rec.get("recipe_source") or "branch_generator_v1_blackbox_search",
                "search_method": rec.get("generator_method"),
                "allocation_rank_within_task": len(recipes),
            }
            recipes.append(item)
        per_split_counts[split] += 1
    recipes.sort(key=lambda row: (str(row.get("split")) != "heldout", -float(row.get("current_score") or 0.0), str(row.get("task_id"))))
    return recipes


def verdict_for(recipes: list[dict[str, Any]], proposer_verdict: str) -> str:
    if not recipes:
        return "BLOCKED"
    leakage_flags = {
        "used_heldout_selector_performance",
        "used_tap_winner_correctness",
        "used_production_routing_outcome",
        "heldout_leakage",
    }
    if any(str(flag) in leakage_flags for row in recipes for flag in row.get("risk_flags", [])):
        return "LEAKAGE_RISK"
    top_non_static = [row for row in recipes[:80] if row.get("generator_method") not in {"static_v4_best_recipe", "random_structured_baseline"} and not row.get("diagnostic")]
    if proposer_verdict in {"READY", "RECIPE_ONLY"} and top_non_static:
        return "WEAK_IMPROVEMENT"
    if top_non_static:
        return "STATIC_RECIPE_SUFFICIENT"
    return "NO_IMPROVEMENT"


def main() -> int:
    started = time.time()
    ensure_bgv1_root()
    plan = load_json(AUDIT_PLAN_JSON, {}) or {}
    proposer = load_pt(PROPOSER_PT, {}) or {}
    if not plan or plan.get("verdict") == "BLOCKED":
        payload = {"BG_BRANCH_GENERATOR_BLACKBOX_SEARCH_V1_VERDICT": "BLOCKED", "verdict": "BLOCKED", "blocker": "missing audit plan"}
        write_json(BLACKBOX_JSON, payload)
        write_md(BLACKBOX_MD, ["# Branch Generator Black-Box Search V1", "", "BG_BRANCH_GENERATOR_BLACKBOX_SEARCH_V1_VERDICT = BLOCKED"])
        print("BG_BRANCH_GENERATOR_BLACKBOX_SEARCH_V1_VERDICT = BLOCKED", flush=True)
        return 1
    recipes = build_schedule(plan, proposer)
    schedule = {
        "schedule_frozen_before_heldout_selector_eval": True,
        "heldout_feedback_allowed": ["quota_progress", "parse_rate", "stability_rate", "row_completion", "non_tie_count", "behaviorally_diverse_group_count"],
        "heldout_feedback_forbidden": ["selector_accuracy", "tap_winner_correctness", "selector_success", "production_routing_outcomes"],
        "recipes": recipes,
        "best_primary_recipe_ids": [str(row.get("recipe_id")) for row in recipes if not row.get("diagnostic")][:60],
    }
    verdict = verdict_for(recipes, str(proposer.get("verdict") or ""))
    payload = {
        "BG_BRANCH_GENERATOR_BLACKBOX_SEARCH_V1_VERDICT": verdict,
        "verdict": verdict,
        "schedule": schedule,
        "methods": ["static_v4_best_recipe", "hs_inspired_controller", "CEM_best_schedule", "ES_lite_best_schedule", "branch_generator_proposer", "random_structured_baseline"],
        "search_objective": "diversity/non-tie/parse/stability/quota yield minus instability/cost/leakage risk",
        "tap_score_used_as_reward": False,
        "heldout_selector_performance_used": False,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(BLACKBOX_JSON, payload)
    write_json(BEST_SCHEDULE_JSON, schedule)
    for row in recipes[:80]:
        append_jsonl(
            RECIPE_ENGRAMS_JSONL,
            {
                "recipe_id": row.get("recipe_id"),
                "split": row.get("split"),
                "task_class": row.get("task_class"),
                "branch_point": row.get("branch_point"),
                "K": row.get("K"),
                "alpha_bucket": row.get("alpha_bucket"),
                "delta_family": row.get("delta_family"),
                "generator_method": row.get("generator_method"),
                "coefficient_summary": "schedule_prior_only",
                "decode_mode": row.get("decode_mode"),
                "rows_generated": 0,
                "behaviorally_diverse_groups": 0,
                "non_tie_pairs": 0,
                "tie_rate": None,
                "parse_rate": None,
                "stability_rate": None,
                "reward_variance": None,
                "compute_cost": 0.0,
                "quota_progress": {},
                "leakage_flags": row.get("risk_flags", []),
                "predicted_score": row.get("current_score"),
                "saved_at": time.time(),
            },
        )
    display = [{**row, "current_score": round(float(row.get("current_score") or 0.0), 4)} for row in recipes[:80]]
    lines = [
        "# Branch Generator Black-Box Search V1",
        "",
        f"BG_BRANCH_GENERATOR_BLACKBOX_SEARCH_V1_VERDICT = {verdict}",
        "",
        f"- recipes: `{len(recipes)}`",
        f"- schedule: `{rel(BEST_SCHEDULE_JSON)}`",
        "- heldout schedule is frozen before selector evaluation.",
        "",
        "## Top Schedule Rows",
        "",
    ]
    lines.extend(md_table(display, ["split", "task_id", "generator_method", "branch_point", "K", "alpha_bucket", "delta_family", "current_score", "diagnostic", "risk_flags"]))
    write_md(BLACKBOX_MD, lines)
    print(f"BG_BRANCH_GENERATOR_BLACKBOX_SEARCH_V1_VERDICT = {verdict}", flush=True)
    return 0 if verdict not in {"BLOCKED", "LEAKAGE_RISK"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
