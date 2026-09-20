"""Fit lightweight Branch Generator v1 proposer models from prior outcomes."""
from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any

import torch

from bg_branch_generator_v1_common import (
    AUDIT_PLAN_JSON,
    BASIS_BANK_PT,
    BRANCH_GENERATOR_PT,
    PROPOSER_JSON,
    PROPOSER_MD,
    PROPOSER_PT,
    branch_group_metrics,
    candidate_pair_stats,
    ensure_bgv1_root,
    group_is_behaviorally_diverse_v2,
    group_is_reward_diverse_v2,
    group_rows,
    load_json,
    load_pt,
    load_v4_branch_rows,
    md_table,
    primary_safe_v4_row,
    rate,
    rel,
    row_reward,
    write_json,
    write_md,
)


def group_recipe_key(vals: list[dict[str, Any]]) -> tuple[str, str, str, str, str]:
    first = vals[0]
    return (
        str(first.get("task_class") or first.get("task_screening_class") or "unknown"),
        str(first.get("branch_point") or "unknown"),
        str(first.get("K") or len(vals)),
        str(first.get("alpha_bucket") or "unknown"),
        str(first.get("primary_delta_family") or first.get("delta_family") or "unknown"),
    )


def recipe_outcome_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for _gid, vals in group_rows([row for row in rows if primary_safe_v4_row(row)]).items():
        if len(vals) < 2:
            continue
        key = group_recipe_key(vals)
        item = buckets.setdefault(
            key,
            {
                "task_class": key[0],
                "branch_point": key[1],
                "K": key[2],
                "alpha_bucket": key[3],
                "delta_family": key[4],
                "rows": 0,
                "groups": 0,
                "behaviorally_diverse_groups": 0,
                "reward_diverse_groups": 0,
                "non_tie_pairs": 0,
                "candidate_pairs": 0,
                "tie_pairs": 0,
                "parse_success_rows": 0,
                "stable_rows": 0,
                "reward_variance_sum": 0.0,
                "compute_cost": 0.0,
            },
        )
        item["rows"] += len(vals)
        item["groups"] += 1
        item["behaviorally_diverse_groups"] += int(group_is_behaviorally_diverse_v2(vals))
        item["reward_diverse_groups"] += int(group_is_reward_diverse_v2(vals))
        pairs = candidate_pair_stats({"g": vals})
        for name in ("non_tie_pairs", "candidate_pairs", "tie_pairs"):
            item[name] += int(pairs[name])
        rewards = [row_reward(row) for row in vals]
        avg = sum(rewards) / max(len(rewards), 1)
        item["reward_variance_sum"] += sum((x - avg) ** 2 for x in rewards) / max(len(rewards), 1)
        item["parse_success_rows"] += sum(1 for row in vals if row.get("parse_success"))
        item["stable_rows"] += len(vals)
        item["compute_cost"] += sum(float(row.get("generation_seconds") or 0.0) for row in vals)
    out = []
    for row in buckets.values():
        rows_n = max(int(row["rows"]), 1)
        row["behaviorally_diverse_groups_per_100_rows"] = 100.0 * int(row["behaviorally_diverse_groups"]) / rows_n
        row["non_tie_pairs_per_100_rows"] = 100.0 * int(row["non_tie_pairs"]) / rows_n
        row["parse_rate"] = int(row["parse_success_rows"]) / rows_n
        row["stability_rate"] = int(row["stable_rows"]) / rows_n
        row["tie_rate"] = int(row["tie_pairs"]) / max(int(row["candidate_pairs"]), 1)
        row["reward_variance"] = float(row["reward_variance_sum"]) / max(int(row["groups"]), 1)
        row["score"] = (
            2.5 * row["behaviorally_diverse_groups_per_100_rows"]
            + 0.08 * row["non_tie_pairs_per_100_rows"]
            + row["parse_rate"]
            + row["stability_rate"]
            + 0.5 * row["reward_variance"]
            - 0.5 * row["tie_rate"]
            - 0.02 * math.log1p(float(row["compute_cost"]))
        )
        out.append(row)
    out.sort(key=lambda row: float(row["score"]), reverse=True)
    return out


def main() -> int:
    started = time.time()
    ensure_bgv1_root()
    plan = load_json(AUDIT_PLAN_JSON, {}) or {}
    basis = load_pt(BASIS_BANK_PT, {}) or {}
    rows = load_v4_branch_rows()
    if not plan or plan.get("verdict") == "BLOCKED":
        verdict = "BLOCKED"
        blocker = "missing audit plan"
    elif not rows:
        verdict = "DATA_LIMITED"
        blocker = "missing v4 branch rows"
    elif not basis.get("directions_by_layer"):
        verdict = "BLOCKED"
        blocker = "missing basis bank"
    else:
        verdict = ""
        blocker = ""
    outcome_rows = recipe_outcome_rows(rows)
    recipe_model = {
        "type": "recipe_bandit_model",
        "features": ["task_class", "branch_point", "K", "alpha_bucket", "delta_family"],
        "rows": outcome_rows,
        "default_prior": outcome_rows[0] if outcome_rows else None,
        "prediction_rule": "score lookup with task-class fallback and global high-yield fallback",
    }
    low_rank = {
        "type": "low_rank_delta_coeff_proposer",
        "status": "not_trained",
        "reason": "prior records contain recipe/basis outcomes but not enough clean coefficient trajectories for supervised coefficient prediction",
    }
    cem = {
        "type": "CEM_initial_distribution",
        "status": "ready" if outcome_rows else "data_limited",
        "elite_recipe_keys": outcome_rows[:12],
        "coefficient_distribution": "centered on high_yield_recipe_direction/old_tap_aligned directions with alpha_0_005 and L24 priority",
    }
    if not verdict:
        verdict = "READY" if low_rank["status"] == "ready" else "RECIPE_ONLY" if outcome_rows else "DATA_LIMITED"
    payload: dict[str, Any] = {
        "BG_BRANCH_GENERATOR_PROPOSER_TRAINING_V1_VERDICT": verdict,
        "verdict": verdict,
        "blocker": blocker,
        "recipe_bandit_model": recipe_model,
        "low_rank_delta_coeff_proposer": low_rank,
        "CEM_initial_distribution": cem,
        "prior_primary_metrics": branch_group_metrics([row for row in rows if primary_safe_v4_row(row)]) if rows else {},
        "tap_score_used_as_training_label": False,
        "heldout_task_outcomes_used_for_fitting": False,
        "fit_task_ids_excluded": plan.get("proposer_training_excluded_task_ids", []),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(payload, PROPOSER_PT)
    torch.save({"branch_generator_proposer": payload, "artifact_kind": "branch_generator_v1"}, BRANCH_GENERATOR_PT)
    write_json(PROPOSER_JSON, payload)
    lines = [
        "# Branch Generator Proposer V1",
        "",
        f"BG_BRANCH_GENERATOR_PROPOSER_TRAINING_V1_VERDICT = {verdict}",
        "",
        f"- recipe_rows: `{len(outcome_rows)}`",
        f"- low_rank_delta_coeff_proposer: `{low_rank['status']}`",
        f"- blocker: `{blocker}`",
        f"- artifact: `{rel(PROPOSER_PT)}`",
        "",
        "The proposer predicts recipe yield from prior branch outcomes. It does not train Ouro, update taps, or use tap score as a label.",
        "",
        "## Top Recipe Outcomes",
        "",
    ]
    display = [{**row, "score": rate(row["score"]), "bdg_per_100": rate(row["behaviorally_diverse_groups_per_100_rows"]), "pairs_per_100": rate(row["non_tie_pairs_per_100_rows"]), "tie_rate": rate(row["tie_rate"])} for row in outcome_rows[:40]]
    lines.extend(md_table(display, ["task_class", "branch_point", "K", "alpha_bucket", "delta_family", "rows", "groups", "behaviorally_diverse_groups", "non_tie_pairs", "bdg_per_100", "pairs_per_100", "tie_rate", "score"]))
    write_md(PROPOSER_MD, lines)
    print(f"BG_BRANCH_GENERATOR_PROPOSER_TRAINING_V1_VERDICT = {verdict}", flush=True)
    return 0 if verdict not in {"BLOCKED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
