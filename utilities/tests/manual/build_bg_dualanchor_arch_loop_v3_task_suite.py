from __future__ import annotations

import argparse
from collections import Counter

from bg_dualanchor_arch_loop_v3_common import OUT_ROOT, ensure_root, report_lines, write_csv, write_json, write_md
import run_bg_dualanchor_architecture_looped_stratified_probe_v2 as v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tasks", type=int, default=48)
    parser.add_argument("--split-mode", choices=("all", "heldout_only", "heldout_val"), default="all")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_root()
    tasks, inventory = v2.select_stratified_tasks(args.max_tasks, args.split_mode)
    rows = []
    for idx, task in enumerate(tasks):
        split = str(task.get("split"))
        if split == "heldout":
            role = "heldout"
        elif split == "val":
            role = "calibration"
        else:
            role = "diagnostic"
        rows.append(
            {
                "index": idx,
                "task_id": task.get("task_id"),
                "domain": task.get("domain"),
                "source_dataset": task.get("source_dataset"),
                "split": split,
                "split_role": role,
                "heldout_likely_non_tie": task.get("heldout_likely_non_tie"),
                "prior_non_tie_pairs": task.get("prior_non_tie_pairs"),
                "priority_score": task.get("priority_score_v4") or task.get("priority_score"),
                "parser": "mcq",
            }
        )
    verdict = "READY" if len(rows) >= 32 and len(Counter(r["domain"] for r in rows)) >= 2 else "PARTIAL"
    payload = {
        "BG_DUALANCHOR_ARCH_LOOP_V3_TASK_SUITE_VERDICT": verdict,
        "inventory": inventory,
        "split_role_counts": dict(Counter(row["split_role"] for row in rows)),
        "tasks": rows,
        "target_hard_slice_quotas": {
            "positive_oracle_tasks": 16,
            "reward_diverse_tasks": 16,
            "positive_and_reward_diverse_tasks": 10,
            "measured_after_run": True,
        },
    }
    write_json(OUT_ROOT / "task_suite.json", payload)
    write_csv(OUT_ROOT / "task_suite.csv", rows)
    lines = report_lines(
        "DualAnchor Architecture Loop v3 Task Suite",
        "BG_DUALANCHOR_ARCH_LOOP_V3_TASK_SUITE_VERDICT",
        verdict,
        [
            ("Inventory", [f"- selected_count: `{inventory.get('selected_count')}`", f"- domain_counts: `{inventory.get('domain_counts')}`", f"- split_counts: `{inventory.get('split_counts')}`", f"- source_counts: `{inventory.get('source_counts')}`"]),
            ("Tasks", [f"- `{row['task_id']}`: {row['domain']} / {row['source_dataset']} / {row['split_role']}" for row in rows]),
        ],
    )
    write_md(OUT_ROOT / "task_suite.md", lines)
    print(f"BG_DUALANCHOR_ARCH_LOOP_V3_TASK_SUITE_VERDICT = {verdict}")
    print(f"selected_count = {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

