"""Audit prior hidden-origin data before v2 diversity expansion."""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from typing import Any

from bg_hidden_origin_diversity_v2_common import (
    V1_ROOT,
    V2_ROOT,
    alpha_bucket,
    deterministic_correct,
    deterministic_reward,
    ensure_v2_root,
    group_is_behaviorally_diverse_v2,
    group_is_reward_diverse_v2,
    group_rows,
    load_all_branch_rows,
    load_json,
    md_table,
    rate,
    safe_primary_row,
    stable_v2_row,
    write_json,
    write_md,
)


OUT_JSON = V2_ROOT / "prior_audit.json"
OUT_MD = V2_ROOT / "prior_audit.md"


def task_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for task_id, vals in sorted(group_rows(rows, key="task_id").items()):
        stable = [row for row in vals if stable_v2_row(row)]
        rewards = {deterministic_reward(row) for row in stable}
        correct = [deterministic_correct(row) for row in stable]
        parsed = {str(row.get("parsed_answer")) for row in stable}
        clean_rows = [row for row in stable if int(row.get("branch_id", -1)) == 0]
        clean = clean_rows[0] if clean_rows else {}
        out.append(
            {
                "task_id": task_id,
                "domain": stable[0].get("domain") if stable else vals[0].get("domain"),
                "rows": len(vals),
                "stable_rows": len(stable),
                "groups": len({row.get("branch_group_id") for row in vals}),
                "all_branches_correct": bool(stable) and all(correct),
                "all_branches_wrong_or_parsefail": bool(stable) and not any(correct),
                "parse_fragile": any(not row.get("parse_success", False) for row in stable),
                "clean_wrong": bool(clean) and not deterministic_correct(clean) and bool(clean.get("parse_success")),
                "clean_parse_failure": bool(clean) and not bool(clean.get("parse_success")),
                "perturbation_changed_answer": len(parsed) > 1,
                "perturbation_changed_reward": len(rewards) > 1,
            }
        )
    return out


def main() -> int:
    started = time.time()
    ensure_v2_root()
    rows = load_all_branch_rows()
    if not rows:
        payload = {
            "BG_HIDDEN_ORIGIN_DIVERSITY_AUDIT_VERDICT": "BLOCKED",
            "verdict": "BLOCKED",
            "blocker": "no prior hidden-origin branch rows found",
        }
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# Hidden-Origin Diversity V2 Prior Audit", "", "BG_HIDDEN_ORIGIN_DIVERSITY_AUDIT_VERDICT = BLOCKED"])
        print("BG_HIDDEN_ORIGIN_DIVERSITY_AUDIT_VERDICT = BLOCKED", flush=True)
        return 1

    stable_safe = [row for row in rows if safe_primary_row(row) and stable_v2_row(row)]
    groups = {gid: vals for gid, vals in group_rows(stable_safe).items() if len(vals) >= 2}
    behavior_groups = [gid for gid, vals in groups.items() if group_is_behaviorally_diverse_v2(vals)]
    reward_groups = [gid for gid, vals in groups.items() if group_is_reward_diverse_v2(vals)]
    task_rows = task_audit(stable_safe)
    candidate_pairs = 0
    tie_pairs = 0
    for vals in groups.values():
        ordered = list(vals)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                candidate_pairs += 1
                if deterministic_reward(ordered[i]) == deterministic_reward(ordered[j]):
                    tie_pairs += 1
    prior_dataset = load_json(V1_ROOT / "hidden_origin_tap_dataset.json", {}) or {}
    prior_eval = load_json(V1_ROOT / "heldout_eval.json", {}) or {}
    payload = {
        "BG_HIDDEN_ORIGIN_DIVERSITY_AUDIT_VERDICT": "READY",
        "verdict": "READY",
        "prior_total_rows": len(rows),
        "prior_stable_safe_rows": len(stable_safe),
        "prior_task_ids": sorted({str(row.get("task_id")) for row in rows}),
        "prior_domains": dict(Counter(str(row.get("domain")) for row in rows)),
        "prior_branch_groups": len(groups),
        "prior_behaviorally_diverse_groups": len(behavior_groups),
        "prior_reward_diverse_groups": len(reward_groups),
        "prior_branch_points": dict(Counter(str(row.get("branch_point")) for row in rows)),
        "prior_delta_families": dict(Counter(str(row.get("delta_family") or row.get("delta_type")) for row in rows)),
        "prior_alphas": dict(Counter(alpha_bucket(row) for row in rows)),
        "prior_deterministic_decode": all(not bool(row.get("do_sample")) for row in rows),
        "prior_sampled_decode_rows": sum(1 for row in rows if bool(row.get("do_sample"))),
        "prior_tie_rate": tie_pairs / max(candidate_pairs, 1),
        "prior_pair_counts": {
            "candidate_pairs": candidate_pairs,
            "tie_pairs": tie_pairs,
            "non_tie_pairs": candidate_pairs - tie_pairs,
        },
        "prior_heldout_task_split": (prior_dataset.get("tasks_by_split") or {}),
        "prior_eval_heldout_task_ids": prior_eval.get("heldout_task_ids"),
        "tasks_all_branches_correct": [row["task_id"] for row in task_rows if row["all_branches_correct"]],
        "tasks_all_branches_wrong": [row["task_id"] for row in task_rows if row["all_branches_wrong_or_parsefail"]],
        "tasks_parse_fragile": [row["task_id"] for row in task_rows if row["parse_fragile"]],
        "tasks_clean_wrong": [row["task_id"] for row in task_rows if row["clean_wrong"]],
        "tasks_clean_parse_failure": [row["task_id"] for row in task_rows if row["clean_parse_failure"]],
        "tasks_perturbation_changed_answer": [row["task_id"] for row in task_rows if row["perturbation_changed_answer"]],
        "tasks_perturbation_changed_reward": [row["task_id"] for row in task_rows if row["perturbation_changed_reward"]],
        "task_rows": task_rows,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)

    summary_rows = [
        {"metric": "rows", "value": len(rows)},
        {"metric": "stable_safe_rows", "value": len(stable_safe)},
        {"metric": "stable_safe_groups", "value": len(groups)},
        {"metric": "behaviorally_diverse_groups", "value": len(behavior_groups)},
        {"metric": "reward_diverse_groups", "value": len(reward_groups)},
        {"metric": "tie_rate", "value": rate(payload["prior_tie_rate"])},
        {"metric": "tasks_changed_reward", "value": len(payload["tasks_perturbation_changed_reward"])},
        {"metric": "tasks_changed_answer", "value": len(payload["tasks_perturbation_changed_answer"])},
    ]
    task_display = [
        {
            "task_id": row["task_id"],
            "domain": row["domain"],
            "groups": row["groups"],
            "all_correct": row["all_branches_correct"],
            "all_wrong": row["all_branches_wrong_or_parsefail"],
            "parse_fragile": row["parse_fragile"],
            "changed_answer": row["perturbation_changed_answer"],
            "changed_reward": row["perturbation_changed_reward"],
        }
        for row in task_rows[:80]
    ]
    lines = [
        "# Hidden-Origin Diversity V2 Prior Audit",
        "",
        "BG_HIDDEN_ORIGIN_DIVERSITY_AUDIT_VERDICT = READY",
        "",
        "## Summary",
        "",
    ]
    lines.extend(md_table(summary_rows, ["metric", "value"]))
    lines.extend(["", "## Prior Task Rows", ""])
    lines.extend(md_table(task_display, ["task_id", "domain", "groups", "all_correct", "all_wrong", "parse_fragile", "changed_answer", "changed_reward"]))
    write_md(OUT_MD, lines)
    print("BG_HIDDEN_ORIGIN_DIVERSITY_AUDIT_VERDICT = READY", flush=True)
    print(f"Wrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

