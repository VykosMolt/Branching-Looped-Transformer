from __future__ import annotations

import time
import re
from collections import Counter

from bg_convergence_hairs_rs_v1_common import (
    OUT_ROOT,
    finite_mean,
    load_replay_rows,
    load_task_rows_csv,
    load_terminal_rows_csv,
    load_v3_rows,
    md_table,
    normalize_text,
    safe_float,
    status_line,
    task_slice_summary,
    terminal_policy_summary,
    truthy,
    write_csv,
    write_json,
    write_md,
)


def main() -> int:
    started = time.time()
    _payload, all_rows, _stage_rows, _task_rows, _terminal_rows = load_v3_rows()
    task_rows = [row for row in load_task_rows_csv() if row.get("domain") == "science"]
    terminal_rows = load_terminal_rows_csv()
    replay_rows = [row for row in load_replay_rows() if row.get("domain") == "science"]
    science_candidates = [row for row in all_rows if row.get("domain") == "science"]
    parser_rows = []
    for row in science_candidates:
        output = normalize_text(row.get("output_text"))
        correct_option = normalize_text(row.get("correct_option"))
        parsed = normalize_text(row.get("parsed_answer"))
        option_pattern = re.compile(rf"(^|[^a-z0-9])(?:option\s*)?{re.escape(correct_option)}([^a-z0-9]|$)") if correct_option else None
        contains_correct_option = bool(option_pattern and option_pattern.search(output))
        parser_rows.append(
            {
                "task_id": row.get("task_id"),
                "source_dataset": row.get("source_dataset"),
                "split": row.get("split"),
                "branch_id": row.get("branch_id"),
                "correct_option": row.get("correct_option"),
                "parsed_answer": row.get("parsed_answer"),
                "reward": row.get("reward"),
                "parse_success": row.get("parse_success"),
                "parse_failure_reason": row.get("parse_failure_reason"),
                "empty_output": row.get("empty_output"),
                "hit_max_tokens": row.get("hit_max_tokens"),
                "repetition_rate": row.get("repetition_rate"),
                "output_contains_correct_option": contains_correct_option,
                "parser_missed_possible_correct_option": bool(contains_correct_option and parsed != correct_option),
            }
        )
    slices = [
        task_slice_summary(task_rows, "science_all", lambda row: True),
        task_slice_summary(task_rows, "science_positive_oracle", lambda row: safe_float(row.get("positive_oracle"), 0.0) > 0),
        task_slice_summary(task_rows, "science_reward_diverse", lambda row: safe_float(row.get("terminal_reward_diverse"), 0.0) > 0),
        task_slice_summary(task_rows, "science_positive_reward_diverse", lambda row: safe_float(row.get("positive_oracle"), 0.0) > 0 and safe_float(row.get("terminal_reward_diverse"), 0.0) > 0),
    ]
    source_summary = []
    for source in sorted({row.get("source_dataset") for row in task_rows}):
        vals = [row for row in task_rows if row.get("source_dataset") == source]
        source_summary.append(
            {
                "source_dataset": source,
                "count": len(vals),
                "positive_oracle_rate": finite_mean(row.get("positive_oracle") for row in vals),
                "reward_diverse_rate": finite_mean(row.get("terminal_reward_diverse") for row in vals),
                "terminal_best_reward": finite_mean(row.get("terminal_best_reward") for row in vals),
                "forced_top1_reward": finite_mean(row.get("terminal_forced_top1_reward") for row in vals),
            }
        )
    policies = terminal_policy_summary(terminal_rows, "science")
    parser_summary = {
        "candidate_count": len(parser_rows),
        "parse_success_rate": finite_mean(row.get("parse_success") for row in parser_rows),
        "empty_output_rate": finite_mean(row.get("empty_output") for row in parser_rows),
        "hit_max_tokens_rate": finite_mean(row.get("hit_max_tokens") for row in parser_rows),
        "parser_missed_possible_correct_option_count": sum(1 for row in parser_rows if row["parser_missed_possible_correct_option"]),
        "parse_failure_reasons": dict(Counter(str(row.get("parse_failure_reason")) for row in parser_rows if not truthy(row.get("parse_success")))),
    }
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
    if not task_rows:
        verdict = "INSUFFICIENT"
    elif safe_float(all_slice.get("positive_oracle"), 0.0) < 0.20:
        verdict = "SCIENCE_BRANCH_GENERATION_WEAK"
    elif parser_summary["parser_missed_possible_correct_option_count"] > 0:
        verdict = "SCIENCE_REWARD_PARSER_WEAK"
    elif safe_float(all_slice.get("terminal_forced_top1_reward"), 0.0) + 0.02 < safe_float(all_slice.get("terminal_best_reward"), 0.0):
        verdict = "SCIENCE_FINAL_CHOICE_WEAK"
    else:
        verdict = "SCIENCE_READY_WITH_TERMINAL_DEFER"
    payload = {
        "BG_SCIENCE_HARD_SLICE_VERDICT": verdict,
        "slice_rows": slices,
        "source_summary": source_summary,
        "terminal_policy_summary": policies,
        "parser_summary": parser_summary,
        "convergence_hair_replay_summary": replay_summary,
        "interpretation": "Science is primarily diagnosed by positive-oracle availability before terminal-choice conclusions.",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "science_hard_slice.json", payload)
    write_csv(OUT_ROOT / "science_rows.csv", task_rows)
    write_csv(OUT_ROOT / "science_parser_audit.csv", parser_rows)
    lines = [
        "# Science Hard-Slice and Parser Audit v1",
        "",
        status_line("BG_SCIENCE_HARD_SLICE_VERDICT", verdict),
        "",
        "## Slices",
        "",
        *md_table(slices, ["slice", "count", "positive_oracle", "terminal_reward_diverse", "terminal_best_reward", "terminal_forced_top1_reward", "terminal_forced_top1_oracle", "stage_false_prunes"]),
        "",
        "## Source Breakdown",
        "",
        *md_table(source_summary, ["source_dataset", "count", "positive_oracle_rate", "reward_diverse_rate", "terminal_best_reward", "forced_top1_reward"]),
        "",
        "## Parser Summary",
        "",
        f"- parse success rate: `{parser_summary['parse_success_rate']}`",
        f"- parser missed possible correct option count: `{parser_summary['parser_missed_possible_correct_option_count']}`",
        f"- parse failure reasons: `{parser_summary['parse_failure_reasons']}`",
        "",
        "## Terminal Policies",
        "",
        *md_table(policies, ["policy", "count", "selected_count", "oracle_retained", "best_selected_reward", "first_selected_reward", "first_selected_oracle", "defer_rate"]),
        "",
        "## Convergence Hair Replay Impact",
        "",
        *md_table(replay_summary, ["policy", "terminal_oracle_retained", "false_merge_rate", "survivor_reduction"]),
    ]
    write_md(OUT_ROOT / "science_hard_slice.md", lines)
    print(status_line("BG_SCIENCE_HARD_SLICE_VERDICT", verdict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
