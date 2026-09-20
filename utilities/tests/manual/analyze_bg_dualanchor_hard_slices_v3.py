from __future__ import annotations

from bg_dualanchor_arch_loop_v3_common import OUT_ROOT, load_task_rows, md_table, report_lines, safe_float, summarize_slices, write_csv, write_json, write_md


def main() -> int:
    tasks = load_task_rows()
    rows = summarize_slices(tasks)
    hard = next((row for row in rows if row["slice"] == "positive_and_reward_diverse"), {})
    reward_diverse = next((row for row in rows if row["slice"] == "reward_diverse"), {})
    if safe_float(hard.get("count"), 0.0) >= 10 and safe_float(hard.get("terminal_forced_top1_oracle"), 0.0) >= 0.85:
        verdict = "HARD_SLICES_READY"
    elif safe_float(reward_diverse.get("terminal_oracle_retained"), 0.0) >= 0.95 and safe_float(reward_diverse.get("terminal_forced_top1_oracle"), 0.0) < 0.85:
        verdict = "TERMINAL_CONFIDENCE_ONLY"
    elif safe_float(hard.get("count"), 0.0) < 10:
        verdict = "DATA_LIMITED"
    else:
        verdict = "TERMINAL_WEAK_ON_REWARD_DIVERSE"
    payload = {"BG_DUALANCHOR_HARD_SLICE_V3_VERDICT": verdict, "slice_rows": rows}
    write_json(OUT_ROOT / "hard_slices.json", payload)
    write_csv(OUT_ROOT / "hard_slices_rows.csv", rows)
    lines = report_lines("DualAnchor Hard Slices v3", "BG_DUALANCHOR_HARD_SLICE_V3_VERDICT", verdict, [("Slices", md_table(rows, ["slice", "count", "terminal_oracle_retained", "terminal_forced_top1_oracle", "terminal_forced_top1_reward", "terminal_best_reward", "terminal_confident", "terminal_deferred", "positive_oracle", "terminal_reward_diverse"]))])
    write_md(OUT_ROOT / "hard_slices.md", lines)
    print(f"BG_DUALANCHOR_HARD_SLICE_V3_VERDICT = {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

