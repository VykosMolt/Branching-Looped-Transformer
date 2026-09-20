"""Audit v1/v2 hidden-origin branch data for v3 diversity-yield planning."""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from typing import Any

from bg_hidden_origin_diversity_v3_common import (
    V2_ROOT,
    V3_ROOT,
    alpha_bucket,
    branch_group_metrics,
    candidate_pair_stats,
    compact_branch_row_v3,
    condition_group_stats,
    deterministic_reward,
    ensure_v3_root,
    group_is_behaviorally_diverse_v2,
    group_is_reward_diverse_v2,
    group_rows,
    load_all_v3_branch_rows,
    load_best_v1_head,
    load_best_v2_head,
    load_json,
    md_table,
    primary_safe_v3_row,
    rate,
    rel,
    row_reward,
    same_prefix_hidden_origin_row,
    score_group_with_head,
    stable_v2_row,
    write_json,
    write_md,
)


OUT_JSON = V3_ROOT / "v3_audit.json"
OUT_MD = V3_ROOT / "v3_audit.md"


def screening_map() -> dict[str, dict[str, Any]]:
    payload = load_json(V2_ROOT / "task_screening.json", {}) or {}
    rows = list(payload.get("rows") or []) + list(payload.get("selected_rows") or [])
    out = {}
    for row in rows:
        out[str(row.get("task_id"))] = row
    return out


def enrich_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_task = screening_map()
    out = []
    for row in rows:
        item = dict(row)
        screen = by_task.get(str(item.get("task_id")), {})
        if not item.get("task_screening_class"):
            item["task_screening_class"] = screen.get("screening_class") or "unknown"
        item["baseline_correctness_class"] = (
            "baseline_correct" if bool(screen.get("clean_correct")) else "baseline_wrong"
        ) if screen else "unknown"
        item["parse_fragile_class"] = (
            "parse_fragile" if not bool(screen.get("clean_parse_success", True)) else "parseable"
        ) if screen else "unknown"
        if "answer_margin" not in item and "answer_margin" in screen:
            item["answer_margin"] = screen.get("answer_margin")
        out.append(item)
    return out


def task_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for task_id, vals in sorted(group_rows(rows, key="task_id").items()):
        groups = {gid: g for gid, g in group_rows(vals).items() if len(g) >= 2}
        diverse = [gid for gid, g in groups.items() if group_is_behaviorally_diverse_v2(g)]
        reward_diverse = [gid for gid, g in groups.items() if group_is_reward_diverse_v2(g)]
        pair_stats = candidate_pair_stats(groups)
        stable = [row for row in vals if stable_v2_row(row)]
        screen_class = vals[0].get("task_screening_class")
        out.append(
            {
                "task_id": task_id,
                "domain": vals[0].get("domain"),
                "screening_class": screen_class,
                "rows": len(vals),
                "stable_rows": len(stable),
                "groups": len(groups),
                "behaviorally_diverse_groups": len(diverse),
                "reward_diverse_groups": len(reward_diverse),
                "non_tie_pairs": pair_stats["non_tie_pairs"],
                "tie_rate": pair_stats["tie_rate"],
                "parse_rate": sum(1 for row in stable if row.get("parse_success")) / max(len(stable), 1),
                "priority_signal": len(diverse) * 5 + pair_stats["non_tie_pairs"] - pair_stats["tie_rate"],
            }
        )
    return out


def disagreement_rows(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    v1_head = load_best_v1_head()
    v2_head = load_best_v2_head()
    for vals in groups.values():
        ordered = sorted(vals, key=lambda row: int(row.get("branch_id", -1)))
        score_group_with_head(ordered, v1_head, score_key="v1_tap_score")
        score_group_with_head(ordered, v2_head, score_key="v2_tap_score")
        old_best = max(ordered, key=lambda row: (float(row.get("tap_margin_sum", row.get("old_frozen_tap_score", 0.0))), -int(row.get("branch_id", -1))))
        candidates = {"old": int(old_best.get("branch_id", -1))}
        if any("v1_tap_score" in row for row in ordered):
            candidates["v1"] = int(max(ordered, key=lambda row: (float(row.get("v1_tap_score", -1e9)), -int(row.get("branch_id", -1)))).get("branch_id", -1))
        if any("v2_tap_score" in row for row in ordered):
            candidates["v2"] = int(max(ordered, key=lambda row: (float(row.get("v2_tap_score", -1e9)), -int(row.get("branch_id", -1)))).get("branch_id", -1))
        rows.append(
            {
                "branch_group_id": ordered[0].get("branch_group_id"),
                "task_id": ordered[0].get("task_id"),
                "domain": ordered[0].get("domain"),
                "task_screening_class": ordered[0].get("task_screening_class"),
                "branch_point": ordered[0].get("branch_point"),
                "alpha_bucket": alpha_bucket(ordered[0]),
                "delta_family": ordered[0].get("primary_delta_family") or ordered[0].get("delta_family"),
                "behaviorally_diverse": group_is_behaviorally_diverse_v2(ordered),
                "reward_diverse": group_is_reward_diverse_v2(ordered),
                "reward_values": sorted({deterministic_reward(row) for row in ordered}),
                "best_branch_by_selector": candidates,
                "old_v1_v2_disagree": len(set(candidates.values())) > 1,
            }
        )
    return rows


def top_conditions(stats: dict[str, dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    rows = []
    for value, row in stats.items():
        rows.append(
            {
                "condition": value,
                "rows": row["rows"],
                "groups": row["groups"],
                "diverse_groups": row["behaviorally_diverse_groups"],
                "reward_diverse_groups": row["reward_diverse_groups"],
                "non_tie_pairs_per_100_rows": rate(row["non_tie_pairs_per_100_rows"]),
                "tie_rate": rate(row["tie_rate"]),
                "stable_rate": rate(row["stable_rate"]),
                "parse_rate": rate(row["parse_rate"]),
            }
        )
    rows.sort(key=lambda row: (float(row["non_tie_pairs_per_100_rows"]) if row["non_tie_pairs_per_100_rows"] != "NA" else -1, row["diverse_groups"]), reverse=True)
    return rows[:limit]


def main() -> int:
    started = time.time()
    ensure_v3_root()
    rows = enrich_rows(load_all_v3_branch_rows(include_prior=True))
    prior_rows = [row for row in rows if not str(row.get("branch_group_id", "")).startswith("v3::")]
    primary_rows = [row for row in prior_rows if primary_safe_v3_row(row)]
    if not prior_rows:
        payload = {
            "BG_HIDDEN_ORIGIN_DIVERSITY_V3_AUDIT_VERDICT": "BLOCKED",
            "verdict": "BLOCKED",
            "blocker": "no v1/v2 hidden-origin rows found",
        }
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# Hidden-Origin Diversity V3 Audit", "", "BG_HIDDEN_ORIGIN_DIVERSITY_V3_AUDIT_VERDICT = BLOCKED"])
        print("BG_HIDDEN_ORIGIN_DIVERSITY_V3_AUDIT_VERDICT = BLOCKED", flush=True)
        return 1

    groups = {gid: vals for gid, vals in group_rows(primary_rows).items() if len(vals) >= 2}
    scored_disagreement = disagreement_rows(groups)
    disagreement_group_ids = {row["branch_group_id"] for row in scored_disagreement if row["old_v1_v2_disagree"]}
    for vals in groups.values():
        disagree = str(vals[0].get("branch_group_id")) in disagreement_group_ids
        for row in vals:
            row["old_v1_v2_disagreement"] = disagree

    stats = {
        "overall_primary": branch_group_metrics(primary_rows),
        "by_task_id": condition_group_stats(primary_rows, "task_id"),
        "by_domain": condition_group_stats(primary_rows, "domain"),
        "by_task_screening_class": condition_group_stats(primary_rows, "task_screening_class"),
        "by_branch_point": condition_group_stats(primary_rows, "branch_point"),
        "by_alpha_bucket": condition_group_stats(prior_rows, "alpha_bucket"),
        "by_delta_family": condition_group_stats(primary_rows, "delta_family"),
        "by_K": condition_group_stats(primary_rows, "K"),
        "by_label_source": condition_group_stats(prior_rows, "label_source"),
        "by_parse_fragile_class": condition_group_stats(primary_rows, "parse_fragile_class"),
        "by_baseline_correctness_class": condition_group_stats(primary_rows, "baseline_correctness_class"),
        "by_old_v1_v2_disagreement": condition_group_stats(primary_rows, "old_v1_v2_disagreement"),
    }
    task_audit = task_rows(primary_rows)
    tasks_prioritize = [
        row
        for row in sorted(task_audit, key=lambda item: (item["behaviorally_diverse_groups"], item["non_tie_pairs"], item["priority_signal"]), reverse=True)
        if row["behaviorally_diverse_groups"] or row["non_tie_pairs"]
    ][:30]
    tasks_avoid = [
        row
        for row in sorted(task_audit, key=lambda item: (item["non_tie_pairs"], -item["tie_rate"], item["parse_rate"]))
        if row["groups"] and row["non_tie_pairs"] == 0
    ][:30]
    alpha02 = stats["by_alpha_bucket"].get("alpha_0_02", {})
    sampled = stats["by_label_source"].get("sampled_expected", {})
    verdict = "READY" if stats["overall_primary"]["groups"] and stats["overall_primary"]["behaviorally_diverse_groups"] else "PARTIAL"
    payload = {
        "BG_HIDDEN_ORIGIN_DIVERSITY_V3_AUDIT_VERDICT": verdict,
        "verdict": verdict,
        "prior_rows": len(prior_rows),
        "primary_rows": len(primary_rows),
        "primary_same_prefix_rows": sum(1 for row in primary_rows if same_prefix_hidden_origin_row(row)),
        "primary_metrics": stats["overall_primary"],
        "condition_stats": stats,
        "top_diversity_yielding_task_classes": top_conditions(stats["by_task_screening_class"]),
        "top_diversity_yielding_delta_families": top_conditions(stats["by_delta_family"]),
        "top_diversity_yielding_branch_points": top_conditions(stats["by_branch_point"]),
        "alpha_0_02_contribution": alpha02,
        "sampled_expected_reward_contribution": sampled,
        "old_v1_v2_disagreement_groups": len(disagreement_group_ids),
        "selector_disagreement_rows": scored_disagreement,
        "tasks_to_prioritize": tasks_prioritize,
        "tasks_to_avoid": tasks_avoid,
        "task_rows": task_audit,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)

    lines = [
        "# Hidden-Origin Diversity V3 Audit",
        "",
        f"BG_HIDDEN_ORIGIN_DIVERSITY_V3_AUDIT_VERDICT = {verdict}",
        "",
        f"- prior_rows: `{len(prior_rows)}`",
        f"- primary_rows: `{len(primary_rows)}`",
        f"- primary_metrics: `{payload['primary_metrics']}`",
        f"- old_v1_v2_disagreement_groups: `{len(disagreement_group_ids)}`",
        "",
        "## Top Task Classes",
        "",
    ]
    lines.extend(md_table(payload["top_diversity_yielding_task_classes"], ["condition", "rows", "groups", "diverse_groups", "reward_diverse_groups", "non_tie_pairs_per_100_rows", "tie_rate", "stable_rate", "parse_rate"]))
    lines.extend(["", "## Top Delta Families", ""])
    lines.extend(md_table(payload["top_diversity_yielding_delta_families"], ["condition", "rows", "groups", "diverse_groups", "reward_diverse_groups", "non_tie_pairs_per_100_rows", "tie_rate", "stable_rate", "parse_rate"]))
    lines.extend(["", "## Top Branch Points", ""])
    lines.extend(md_table(payload["top_diversity_yielding_branch_points"], ["condition", "rows", "groups", "diverse_groups", "reward_diverse_groups", "non_tie_pairs_per_100_rows", "tie_rate", "stable_rate", "parse_rate"]))
    lines.extend(["", "## Alpha 0.02 And Sampled Diagnostics", ""])
    lines.append(f"- alpha_0_02_contribution: `{alpha02}`")
    lines.append(f"- sampled_expected_reward_contribution: `{sampled}`")
    lines.extend(["", "## Tasks To Prioritize", ""])
    lines.extend(md_table(tasks_prioritize[:20], ["task_id", "domain", "screening_class", "groups", "behaviorally_diverse_groups", "non_tie_pairs", "tie_rate", "parse_rate"]))
    lines.extend(["", "## Tasks To Avoid", ""])
    lines.extend(md_table(tasks_avoid[:20], ["task_id", "domain", "screening_class", "groups", "behaviorally_diverse_groups", "non_tie_pairs", "tie_rate", "parse_rate"]))
    lines.extend(["", "## Outputs", "", f"- JSON: `{rel(OUT_JSON)}`"])
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_DIVERSITY_V3_AUDIT_VERDICT = {verdict}", flush=True)
    print(f"Wrote {rel(OUT_JSON)}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

