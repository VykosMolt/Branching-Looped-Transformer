"""Audit v3 hidden-origin split bottlenecks without generating new branches."""
from __future__ import annotations

import time
from collections import Counter

from bg_hidden_origin_split_salvage_common import (
    AUDIT_JSON,
    SALVAGE_ROOT,
    ensure_salvage_root,
    load_json,
    md_table,
    primary_rows,
    prior_seen_sets,
    rel,
    support_for_task_ids,
    task_distribution,
    version_summary,
    write_csv,
    write_json,
    write_md,
)
from bg_hidden_origin_diversity_v3_common import V3_ROOT


OUT_MD = SALVAGE_ROOT / "audit.md"
OUT_CSV = SALVAGE_ROOT / "task_distribution.csv"


def main() -> int:
    started = time.time()
    ensure_salvage_root()
    rows = primary_rows()
    if not rows:
        payload = {"BG_HIDDEN_ORIGIN_SPLIT_AUDIT_VERDICT": "BLOCKED", "verdict": "BLOCKED", "blocker": "no primary rows"}
        write_json(AUDIT_JSON, payload)
        write_md(OUT_MD, ["# Hidden-Origin Split Salvage Audit", "", "BG_HIDDEN_ORIGIN_SPLIT_AUDIT_VERDICT = BLOCKED"])
        print("BG_HIDDEN_ORIGIN_SPLIT_AUDIT_VERDICT = BLOCKED", flush=True)
        return 1

    split_guard = load_json(V3_ROOT / "split_guard_v3.json", {}) or {}
    v3_dataset = load_json(V3_ROOT / "hidden_origin_tap_dataset_v3.json", {}) or {}
    dist = task_distribution(rows)
    by_task = {str(row["task_id"]): row for row in dist}
    seen = prior_seen_sets()
    clean_cross_version = set(seen["clean_cross_version_heldout_task_ids"])
    v3_heldout = set(seen["v3_heldout_candidate_task_ids"])
    strict_clean_tasks_with_rows = sorted(clean_cross_version & set(by_task))
    strict_clean_tasks_with_behavior = sorted(t for t in strict_clean_tasks_with_rows if int(by_task[t]["behaviorally_diverse_groups"]) > 0)
    strict_clean_tasks_with_non_tie = sorted(t for t in strict_clean_tasks_with_rows if int(by_task[t]["non_tie_pairs"]) > 0)

    top_behavior = sorted(dist, key=lambda row: (-int(row["behaviorally_diverse_groups"]), -int(row["non_tie_pairs"]), row["task_id"]))[:20]
    top_pairs = sorted(dist, key=lambda row: (-int(row["non_tie_pairs"]), -int(row["behaviorally_diverse_groups"]), row["task_id"]))[:20]
    total_behavior = sum(int(row["behaviorally_diverse_groups"]) for row in dist)
    top3_behavior = sum(int(row["behaviorally_diverse_groups"]) for row in top_behavior[:3])
    concentration = top3_behavior / max(total_behavior, 1)
    support_strict = support_for_task_ids(rows, strict_clean_tasks_with_rows)
    support_v3 = support_for_task_ids(rows, sorted(v3_heldout & set(by_task)))
    support_all_signal = support_for_task_ids(rows, [row["task_id"] for row in dist if int(row["non_tie_pairs"]) > 0])

    verdict = "READY"
    if len(dist) < 1:
        verdict = "BLOCKED"
    elif not strict_clean_tasks_with_rows:
        verdict = "PARTIAL"

    payload = {
        "BG_HIDDEN_ORIGIN_SPLIT_AUDIT_VERDICT": verdict,
        "verdict": verdict,
        "version_summary": version_summary(rows),
        "task_distribution": dist,
        "diverse_group_concentration": {
            "total_behaviorally_diverse_groups": total_behavior,
            "top3_behaviorally_diverse_groups": top3_behavior,
            "top3_fraction": concentration,
            "tasks_with_behavioral_diversity": sum(1 for row in dist if int(row["behaviorally_diverse_groups"]) > 0),
            "tasks_with_non_tie_pairs": sum(1 for row in dist if int(row["non_tie_pairs"]) > 0),
        },
        "strict_split_bottleneck": {
            "strict_clean_tasks_with_primary_rows": strict_clean_tasks_with_rows,
            "strict_clean_tasks_with_behavioral_diversity": strict_clean_tasks_with_behavior,
            "strict_clean_tasks_with_non_tie_pairs": strict_clean_tasks_with_non_tie,
            "support": support_strict,
            "v3_dataset_test_tasks": (v3_dataset.get("tasks_by_split") or {}).get("test", []),
            "v3_dataset_behaviorally_diverse_groups_by_split": v3_dataset.get("behaviorally_diverse_groups_by_split"),
            "v3_dataset_pairs_by_split": v3_dataset.get("pairs_by_split"),
        },
        "v3_clean_candidate_support": support_v3,
        "all_reward_signal_support": support_all_signal,
        "prior_seen_sets": seen,
        "split_guard_counts": split_guard.get("counts"),
        "domain_counts": dict(Counter(str(row.get("domain")) for row in rows)),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(AUDIT_JSON, payload)
    write_csv(OUT_CSV, dist)

    lines = [
        "# Hidden-Origin Split Salvage Audit",
        "",
        f"BG_HIDDEN_ORIGIN_SPLIT_AUDIT_VERDICT = {verdict}",
        "",
        "This audit uses only existing stable primary-safe deterministic rows. Behavior-only diversity and non-tie reward-pair support are reported separately.",
        "",
        "## Version Summary",
        "",
    ]
    lines.extend(
        md_table(
            [
                {"version": version, **stats}
                for version, stats in payload["version_summary"].items()
            ],
            ["version", "rows", "groups", "behaviorally_diverse_groups", "reward_diverse_groups", "non_tie_pairs", "tie_rate"],
        )
    )
    lines.extend(
        [
            "",
            "## Strict Heldout Bottleneck",
            "",
            f"- strict_clean_tasks_with_primary_rows: `{len(strict_clean_tasks_with_rows)}`",
            f"- strict_clean_tasks_with_behavioral_diversity: `{len(strict_clean_tasks_with_behavior)}`",
            f"- strict_clean_tasks_with_non_tie_pairs: `{len(strict_clean_tasks_with_non_tie)}`",
            f"- strict_clean_non_tie_pairs: `{support_strict['non_tie_pairs']}`",
            f"- strict_clean_behaviorally_diverse_groups: `{support_strict['behaviorally_diverse_groups']}`",
            "",
            "Strict v3 heldout collapsed because most clean heldout tasks with behavior-level answer variation still had tied deterministic rewards, leaving too few non-tie pair labels.",
            "",
            "## Top Behaviorally Diverse Tasks",
            "",
        ]
    )
    lines.extend(md_table(top_behavior, ["task_id", "domain", "behaviorally_diverse_groups", "reward_diverse_groups", "non_tie_pairs", "tie_rate", "versions"]))
    lines.extend(["", "## Top Non-Tie Pair Tasks", ""])
    lines.extend(md_table(top_pairs, ["task_id", "domain", "behaviorally_diverse_groups", "reward_diverse_groups", "non_tie_pairs", "tie_rate", "versions"]))
    lines.extend(
        [
            "",
            "## Clean Task Sets",
            "",
            f"- v3_clean_candidate_tasks_with_rows: `{len(sorted(v3_heldout & set(by_task)))}`",
            f"- all_reward_signal_tasks: `{support_all_signal['support_task_count']}`",
            f"- prior_v1_v2_train_val_tasks: `{len(seen['v1_v2_train_val'])}`",
            f"- prior_v3_train_val_tasks: `{len(seen['v3_train_val'])}`",
            "",
            f"Wrote `{rel(OUT_CSV)}`.",
        ]
    )
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_SPLIT_AUDIT_VERDICT = {verdict}", flush=True)
    print(f"Wrote {rel(AUDIT_JSON)}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

