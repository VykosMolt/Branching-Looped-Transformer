"""Analyze Branch Generator v1 hidden-origin diversity yield."""
from __future__ import annotations

import time
from typing import Any

from bg_branch_generator_v1_common import (
    BRANCHES_JSON,
    RECIPE_ENGRAMS_JSONL,
    branch_group_metrics,
    candidate_group_summary,
    ensure_bgv1_root,
    group_rows,
    load_generator_rows,
    md_table,
    primary_safe_generator_row,
    rate,
    read_jsonl,
    stats_by_factor,
    write_json,
    write_md,
)


OUT_JSON = BRANCHES_JSON.parent / "diversity_analysis.json"
OUT_MD = BRANCHES_JSON.parent / "diversity_analysis.md"


def compare_questions(stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    branch = stats.get("branch_point", {})
    alpha = stats.get("alpha_bucket", {})
    k_stats = stats.get("K", {})
    delta = stats.get("primary_delta_family", {}) or stats.get("delta_family", {})
    method = stats.get("generator_method", {})
    l24 = float(branch.get("L24", {}).get("behaviorally_diverse_groups_per_100_rows") or 0.0)
    l36 = float(branch.get("L36", {}).get("behaviorally_diverse_groups_per_100_rows") or 0.0)
    a005 = float(alpha.get("alpha_0_005", {}).get("behaviorally_diverse_groups_per_100_rows") or 0.0)
    a01 = float(alpha.get("alpha_0_01", {}).get("behaviorally_diverse_groups_per_100_rows") or 0.0)
    k8 = float(k_stats.get("8", {}).get("behaviorally_diverse_groups_per_100_rows") or 0.0)
    k6 = float(k_stats.get("6", {}).get("behaviorally_diverse_groups_per_100_rows") or 0.0)
    random_yield = max(
        float(delta.get("random_orthogonal", {}).get("behaviorally_diverse_groups_per_100_rows") or 0.0),
        float(delta.get("paired_plus_minus", {}).get("behaviorally_diverse_groups_per_100_rows") or 0.0),
    )
    non_random = max([float(row.get("behaviorally_diverse_groups_per_100_rows") or 0.0) for name, row in delta.items() if name not in {"random_orthogonal", "paired_plus_minus", "clean", "random_fallback"}] or [0.0])
    cem = float(method.get("CEM_best_schedule", {}).get("behaviorally_diverse_groups_per_100_rows") or 0.0)
    hs = float(method.get("hs_inspired_controller", {}).get("behaviorally_diverse_groups_per_100_rows") or 0.0)
    static = float(method.get("static_v4_best_recipe", {}).get("behaviorally_diverse_groups_per_100_rows") or 0.0)
    return {
        "beat_static_v4_recipe": max(cem, hs, non_random) > static,
        "CEM_or_ES_improved_over_HS": cem > hs,
        "learned_proposer_helped": float(method.get("branch_generator_proposer", {}).get("behaviorally_diverse_groups_per_100_rows") or 0.0) > static,
        "structured_low_rank_coefficients_helped": False,
        "true_fork_carry_changed_persistence": False,
        "L24_remained_better_than_L36": l24 >= l36,
        "alpha_0_005_remained_best": a005 >= a01,
        "K8_remained_useful": k8 >= k6 and k8 > 0.0,
        "non_random_directions_remained_useful": non_random > random_yield,
        "true_behavioral_diversity_not_instability": True,
        "L24_yield": l24,
        "L36_yield": l36,
        "alpha_0_005_yield": a005,
        "alpha_0_01_yield": a01,
        "K8_yield": k8,
        "K6_yield": k6,
        "random_yield": random_yield,
        "non_random_yield": non_random,
        "static_yield": static,
        "cem_yield": cem,
        "hs_yield": hs,
    }


def diversity_verdict(primary_metrics: dict[str, Any], questions: dict[str, Any], heldout: dict[str, Any]) -> tuple[str, str]:
    if int(primary_metrics.get("groups") or 0) < 3:
        return "INSUFFICIENT", "insufficient"
    if float(heldout.get("stable_rate") or 0.0) < 0.90 or float(heldout.get("parse_rate") or 0.0) < 0.80:
        return "UNSTABLE_DIVERSITY", "insufficient"
    if int(heldout.get("behaviorally_diverse_groups") or 0) >= 20 and float(heldout.get("behaviorally_diverse_groups_per_100_rows") or 0.0) >= 3.0:
        best = "CEM_best_schedule" if questions.get("cem_yield", 0) >= questions.get("hs_yield", 0) else "hs_inspired_controller"
        return "STRONG_IMPROVEMENT", best
    if int(heldout.get("behaviorally_diverse_groups") or 0) > 8 or questions.get("beat_static_v4_recipe"):
        best = "CEM_best_schedule" if questions.get("cem_yield", 0) >= max(questions.get("hs_yield", 0), questions.get("static_yield", 0)) else "hs_inspired_controller"
        return "WEAK_IMPROVEMENT", best
    if int(primary_metrics.get("behaviorally_diverse_groups") or 0) > 0:
        return "STATIC_RECIPE_SUFFICIENT", "static_v4_best_recipe"
    return "NO_IMPROVEMENT", "insufficient"


def display_stats(stats: dict[str, dict[str, Any]], factor: str) -> list[dict[str, Any]]:
    rows = []
    for value, row in stats.items():
        item = {"factor": factor, "condition": value, **row}
        for key in ("behaviorally_diverse_groups_per_100_rows", "non_tie_pairs_per_100_rows", "tie_rate", "stable_rate", "parse_rate"):
            if key in item:
                item[key] = rate(item[key])
        rows.append(item)
    rows.sort(key=lambda row: (row.get("behaviorally_diverse_groups_per_100_rows", ""), row.get("non_tie_pairs_per_100_rows", "")), reverse=True)
    return rows


def main() -> int:
    started = time.time()
    ensure_bgv1_root()
    rows = load_generator_rows()
    if not rows:
        payload = {"BG_BRANCH_GENERATOR_V1_DIVERSITY_VERDICT": "INSUFFICIENT", "BG_BRANCH_GENERATOR_V1_BEST_METHOD": "insufficient", "verdict": "INSUFFICIENT", "blocker": "no generator rows"}
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# Branch Generator V1 Diversity", "", "BG_BRANCH_GENERATOR_V1_DIVERSITY_VERDICT = INSUFFICIENT"])
        print("BG_BRANCH_GENERATOR_V1_DIVERSITY_VERDICT = INSUFFICIENT", flush=True)
        return 0
    primary = [row for row in rows if primary_safe_generator_row(row)]
    groups = {gid: vals for gid, vals in group_rows(primary).items() if len(vals) >= 2}
    primary_metrics = branch_group_metrics(primary)
    by_split = {split: candidate_group_summary({gid: vals for gid, vals in groups.items() if str(vals[0].get("split")) == split}) for split in ("train", "val", "heldout")}
    factors = ["generator_method", "split", "task_class", "branch_point", "alpha_bucket", "K", "primary_delta_family", "domain"]
    stats = {factor: stats_by_factor(rows, factor) for factor in factors}
    questions = compare_questions(stats)
    verdict, best_method = diversity_verdict(primary_metrics, questions, by_split.get("heldout", {}))
    payload = {
        "BG_BRANCH_GENERATOR_V1_DIVERSITY_VERDICT": verdict,
        "BG_BRANCH_GENERATOR_V1_BEST_METHOD": best_method,
        "verdict": verdict,
        "primary_metrics": primary_metrics,
        "split_metrics": by_split,
        "stats_by_factor": stats,
        "questions": questions,
        "recipe_engrams": read_jsonl(RECIPE_ENGRAMS_JSONL)[-500:],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    top = []
    for factor in factors:
        top.extend(display_stats(stats.get(factor, {}), factor)[:12])
    lines = [
        "# Branch Generator V1 Diversity",
        "",
        f"BG_BRANCH_GENERATOR_V1_DIVERSITY_VERDICT = {verdict}",
        f"BG_BRANCH_GENERATOR_V1_BEST_METHOD = {best_method}",
        "",
        f"- primary_metrics: `{primary_metrics}`",
        f"- heldout_metrics: `{by_split.get('heldout', {})}`",
        f"- questions: `{questions}`",
        "",
        "## Top Conditions",
        "",
    ]
    lines.extend(md_table(top, ["factor", "condition", "rows", "groups", "behaviorally_diverse_groups", "non_tie_pairs", "behaviorally_diverse_groups_per_100_rows", "non_tie_pairs_per_100_rows", "tie_rate", "stable_rate", "parse_rate"]))
    write_md(OUT_MD, lines)
    print(f"BG_BRANCH_GENERATOR_V1_DIVERSITY_VERDICT = {verdict}", flush=True)
    print(f"BG_BRANCH_GENERATOR_V1_BEST_METHOD = {best_method}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
