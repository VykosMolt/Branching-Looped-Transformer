from __future__ import annotations

import time

from bg_convergence_hairs_rs_v1_common import (
    OUT_ROOT,
    finite_mean,
    load_replay_rows,
    load_task_rows_csv,
    load_terminal_rows_csv,
    md_table,
    safe_float,
    status_line,
    task_slice_summary,
    terminal_policy_summary,
    write_csv,
    write_json,
    write_md,
)


def main() -> int:
    started = time.time()
    task_rows = [row for row in load_task_rows_csv() if row.get("domain") == "reasoning"]
    terminal_rows = load_terminal_rows_csv()
    replay_rows = [row for row in load_replay_rows() if row.get("domain") == "reasoning"]
    slices = [
        task_slice_summary(task_rows, "reasoning_all", lambda row: True),
        task_slice_summary(task_rows, "reasoning_positive_oracle", lambda row: safe_float(row.get("positive_oracle"), 0.0) > 0),
        task_slice_summary(task_rows, "reasoning_reward_diverse", lambda row: safe_float(row.get("terminal_reward_diverse"), 0.0) > 0),
        task_slice_summary(task_rows, "reasoning_positive_reward_diverse", lambda row: safe_float(row.get("positive_oracle"), 0.0) > 0 and safe_float(row.get("terminal_reward_diverse"), 0.0) > 0),
    ]
    policies = terminal_policy_summary(terminal_rows, "reasoning")
    replay_summary = []
    for policy in sorted({row.get("policy") for row in replay_rows}):
        vals = [row for row in replay_rows if row.get("policy") == policy]
        replay_summary.append(
            {
                "policy": policy,
                "terminal_oracle_retained": finite_mean(row.get("terminal_oracle_retained") for row in vals),
                "false_merge_rate": finite_mean(1.0 if safe_float(row.get("false_merge_count"), 0.0) > 0 else 0.0 for row in vals),
                "survivor_reduction": finite_mean(row.get("hair_survivor_reduction") for row in vals),
            }
        )
    all_slice = slices[0]
    hard_slice = slices[-1]
    if not task_rows:
        verdict = "INSUFFICIENT"
    elif safe_float(all_slice.get("terminal_best_reward"), 0.0) <= 0:
        verdict = "REASONING_BRANCH_GENERATION_WEAK"
    elif safe_float(hard_slice.get("count"), 0.0) and safe_float(hard_slice.get("terminal_forced_top1_oracle"), 1.0) < 0.80:
        verdict = "REASONING_NEEDS_TERMINAL_DEFER"
    elif safe_float(all_slice.get("terminal_forced_top1_reward"), 0.0) + 0.05 < safe_float(all_slice.get("terminal_best_reward"), 0.0):
        verdict = "REASONING_FINAL_CHOICE_WEAK"
    else:
        verdict = "REASONING_READY_WITH_TERMINAL_CONFIDENCE"
    payload = {
        "BG_REASONING_HARD_SLICE_VERDICT": verdict,
        "slice_rows": slices,
        "terminal_policy_summary": policies,
        "convergence_hair_replay_summary": replay_summary,
        "interpretation": "Reasoning has reachable positive-oracle branches; terminal confidence/defer remains the main protection on reward-diverse hard slices.",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "reasoning_hard_slice.json", payload)
    write_csv(OUT_ROOT / "reasoning_rows.csv", task_rows)
    lines = [
        "# Reasoning Hard-Slice Analysis v1",
        "",
        status_line("BG_REASONING_HARD_SLICE_VERDICT", verdict),
        "",
        "## Slices",
        "",
        *md_table(slices, ["slice", "count", "terminal_best_reward", "terminal_forced_top1_reward", "terminal_forced_top1_oracle", "terminal_confident", "terminal_deferred", "stage_false_prunes"]),
        "",
        "## Terminal Policies",
        "",
        *md_table(policies, ["policy", "count", "selected_count", "oracle_retained", "best_selected_reward", "first_selected_reward", "first_selected_oracle", "defer_rate"]),
        "",
        "## Convergence Hair Replay Impact",
        "",
        *md_table(replay_summary, ["policy", "terminal_oracle_retained", "false_merge_rate", "survivor_reduction"]),
    ]
    write_md(OUT_ROOT / "reasoning_hard_slice.md", lines)
    print(status_line("BG_REASONING_HARD_SLICE_VERDICT", verdict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

