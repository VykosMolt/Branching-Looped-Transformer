"""Initialize lightweight quota recipe controller for hidden-origin v4.

This is a local recipe allocator only.  It proposes branch-generation recipes,
tracks per-recipe outcome records, and suppresses high-risk recipes.  It does
not use or execute any external Hunter-Seeker architecture.
"""
from __future__ import annotations

import hashlib
import time
from collections import Counter
from typing import Any

from bg_hidden_origin_quota_v4_common import (
    CONTROLLER_JSON,
    CONTROLLER_MD,
    DELTA_FAMILIES,
    DIRECTION_BANK_V4_JSON,
    PRIMARY_MINIMUMS,
    RECIPE_ENGRAMS_JSONL,
    V4_ROOT,
    ensure_v4_root,
    load_json,
    load_quota_plan,
    md_table,
    read_jsonl,
    rel,
    write_json,
    write_md,
)


def recipe_id_for(row: dict[str, Any]) -> str:
    key = "|".join(str(row.get(k)) for k in ("split", "task_class", "branch_point", "K", "alpha_bucket", "delta_family", "decode_mode"))
    return "v4_recipe_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def initial_score(recipe: dict[str, Any], family_status: dict[str, Any]) -> float:
    score = 0.0
    if recipe["branch_point"] == "L36":
        score += 3.0
    if recipe["K"] == 8:
        score += 2.0
    if recipe["alpha_bucket"] in {"alpha_0_005", "alpha_0_01"}:
        score += 2.0
    if recipe["task_class"] in {"perturbation_sensitive", "wrong_parseable", "parse_fragile", "low_confidence_correct"}:
        score += 1.5
    fam = recipe["delta_family"]
    status = family_status.get(fam, {})
    if fam not in {"random_orthogonal", "paired_plus_minus"} and int(status.get("perturbation_usable_entries") or 0) > 0:
        score += 2.0
    if recipe["alpha_bucket"] == "alpha_0_02":
        score -= 2.5
    if recipe["branch_point"] == "L47_diagnostic":
        score -= 3.0
    return score


def build_recipes(plan: dict[str, Any], family_status: dict[str, Any]) -> list[dict[str, Any]]:
    recipes: dict[str, dict[str, Any]] = {}
    for task in plan.get("tasks", []):
        split = str(task.get("split"))
        for rec in task.get("priority_recipes") or []:
            row = {
                "split": split,
                "task_class": rec.get("task_class") or task.get("task_class") or "wrong_parseable",
                "branch_point": rec.get("branch_point"),
                "K": int(rec.get("K") or 6),
                "alpha_bucket": rec.get("alpha_bucket"),
                "delta_family": rec.get("delta_family"),
                "decode_mode": rec.get("decode_mode") or "deterministic",
                "recipe_source": "hs_inspired_controller",
                "risk_flags": [],
            }
            if row["branch_point"] == "L47_diagnostic" or row["alpha_bucket"] == "alpha_0_02":
                row["risk_flags"].append("diagnostic_not_primary_readiness")
            if row["delta_family"] not in {"random_orthogonal", "paired_plus_minus"}:
                status = family_status.get(row["delta_family"], {})
                if int(status.get("perturbation_usable_entries") or 0) <= 0:
                    row["risk_flags"].append("direction_family_not_usable")
            row["recipe_id"] = recipe_id_for(row)
            row["initial_score"] = initial_score(row, family_status)
            row["current_score"] = row["initial_score"]
            recipes[row["recipe_id"]] = row
    return sorted(recipes.values(), key=lambda row: (row["split"], -float(row["current_score"]), row["recipe_id"]))


def summarize_engrams(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_recipe: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_recipe.setdefault(str(row.get("recipe_id")), []).append(row)
    summary = {}
    for rid, vals in by_recipe.items():
        rows_generated = sum(int(v.get("rows_generated") or 0) for v in vals)
        behavior = sum(int(v.get("behaviorally_diverse_groups") or 0) for v in vals)
        non_tie = sum(int(v.get("non_tie_pairs") or 0) for v in vals)
        parse_rates = [float(v.get("parse_rate") or 0.0) for v in vals]
        stable_rates = [float(v.get("stability_rate") or 0.0) for v in vals]
        summary[rid] = {
            "records": len(vals),
            "rows_generated": rows_generated,
            "behaviorally_diverse_groups": behavior,
            "non_tie_pairs": non_tie,
            "behaviorally_diverse_groups_per_100_rows": 100.0 * behavior / max(rows_generated, 1),
            "non_tie_pairs_per_100_rows": 100.0 * non_tie / max(rows_generated, 1),
            "parse_rate": sum(parse_rates) / max(len(parse_rates), 1),
            "stability_rate": sum(stable_rates) / max(len(stable_rates), 1),
        }
    return summary


def verdict_from_state(engram_summary: dict[str, Any]) -> str:
    if not engram_summary:
        return "INSUFFICIENT"
    values = list(engram_summary.values())
    best_controller = max((v for v in values), key=lambda v: (float(v["behaviorally_diverse_groups_per_100_rows"]), float(v["non_tie_pairs_per_100_rows"])))
    if float(best_controller["stability_rate"]) < 0.5:
        return "UNSTABLE"
    if float(best_controller["behaviorally_diverse_groups_per_100_rows"]) >= 5.0 or float(best_controller["non_tie_pairs_per_100_rows"]) >= 20.0:
        return "IMPROVES_DIVERSITY_YIELD"
    if float(best_controller["behaviorally_diverse_groups_per_100_rows"]) > 0.0:
        return "WEAK_IMPROVEMENT"
    return "NO_IMPROVEMENT"


def main() -> int:
    started = time.time()
    ensure_v4_root()
    plan = load_quota_plan()
    bank = load_json(DIRECTION_BANK_V4_JSON, {}) or {}
    if plan.get("verdict") == "BLOCKED" or bank.get("verdict") == "BLOCKED" or not plan:
        verdict = "INSUFFICIENT" if plan else "LEAKAGE_RISK"
        payload = {
            "BG_HS_INSPIRED_QUOTA_CONTROLLER_V4_VERDICT": verdict,
            "verdict": verdict,
            "blocker": "missing usable quota plan or direction bank",
        }
        write_json(CONTROLLER_JSON, payload)
        write_md(CONTROLLER_MD, ["# HS-Inspired Quota Controller V4", "", f"BG_HS_INSPIRED_QUOTA_CONTROLLER_V4_VERDICT = {verdict}"])
        print(f"BG_HS_INSPIRED_QUOTA_CONTROLLER_V4_VERDICT = {verdict}", flush=True)
        return 0
    family_status = bank.get("family_status") or {}
    recipes = build_recipes(plan, family_status)
    engrams = read_jsonl(RECIPE_ENGRAMS_JSONL)
    engram_summary = summarize_engrams(engrams)
    verdict = verdict_from_state(engram_summary)
    payload: dict[str, Any] = {
        "BG_HS_INSPIRED_QUOTA_CONTROLLER_V4_VERDICT": verdict,
        "verdict": verdict,
        "mode": "pre_generation_initialization" if not engrams else "post_generation_state",
        "quota_minimums": PRIMARY_MINIMUMS,
        "candidate_proposal": "local_recipe_grid_from_quota_plan",
        "scoring": "yield_parse_stability_quota_cost_risk_score",
        "memory": "local_recipe_engrams_v4_jsonl",
        "risk_arbitration": {
            "suppress_unusable_direction_families": True,
            "alpha_0_02_diagnostic_only": True,
            "L47_diagnostic_only": True,
            "leakage_flags_exclude_readiness": True,
            "heldout_selector_performance_feedback_forbidden": True,
        },
        "recipes": recipes,
        "recipe_count": len(recipes),
        "recipes_by_split": dict(Counter(row["split"] for row in recipes)),
        "recipe_engrams_path": rel(RECIPE_ENGRAMS_JSONL),
        "recipe_engrams": engram_summary,
        "heldout_feedback_policy": {
            "allowed": ["quota_progress", "parse_rate", "stability_rate", "row_completion", "non_tie_pairs", "behaviorally_diverse_groups"],
            "forbidden": ["selector_accuracy", "old_v1_v2_v3_salvage_v4_selector_success", "tap_winner_correctness", "final_selector_eval_score"],
            "heldout_schedule_frozen_before_selector_eval": True,
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(CONTROLLER_JSON, payload)
    if not RECIPE_ENGRAMS_JSONL.exists():
        RECIPE_ENGRAMS_JSONL.write_text("", encoding="utf-8")
    rows = [
        {
            "recipe_id": row["recipe_id"],
            "split": row["split"],
            "task_class": row["task_class"],
            "branch_point": row["branch_point"],
            "K": row["K"],
            "alpha_bucket": row["alpha_bucket"],
            "delta_family": row["delta_family"],
            "decode_mode": row["decode_mode"],
            "score": row["current_score"],
            "risk_flags": row["risk_flags"],
        }
        for row in recipes[:120]
    ]
    lines = [
        "# HS-Inspired Quota Controller V4",
        "",
        f"BG_HS_INSPIRED_QUOTA_CONTROLLER_V4_VERDICT = {verdict}",
        "",
        "This controller is a local hidden-origin recipe allocator only. It does not execute external agent architecture, ARC runtime, wrapper/local-agent code, or environment actions.",
        "",
        f"- recipe_count: `{len(recipes)}`",
        f"- recipes_by_split: `{payload['recipes_by_split']}`",
        f"- engram_records: `{len(engrams)}`",
        "",
        "## Top Recipes",
        "",
    ]
    lines.extend(md_table(rows, ["recipe_id", "split", "task_class", "branch_point", "K", "alpha_bucket", "delta_family", "decode_mode", "score", "risk_flags"]))
    lines.extend(["", "## Engram Summary", "", f"`{engram_summary}`"])
    write_md(CONTROLLER_MD, lines)
    print(f"BG_HS_INSPIRED_QUOTA_CONTROLLER_V4_VERDICT = {verdict}", flush=True)
    print(f"Wrote {rel(CONTROLLER_JSON)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
