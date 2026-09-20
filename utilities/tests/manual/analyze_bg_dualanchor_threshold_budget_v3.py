from __future__ import annotations

from bg_dualanchor_arch_loop_v3_common import GUARDED_ROOT, OUT_ROOT, RUN_JSON, grouped_mean, load_stage_rows, md_table, read_json, report_lines, safe_float, write_csv, write_json, write_md


def main() -> int:
    run = read_json(RUN_JSON, {}) or {}
    stage_rows = load_stage_rows()
    primary = {
        "policy": "v3_live_mean_floor_very_loose_budget8",
        "stage_oracle_retention": run.get("stage_oracle_retention"),
        "terminal_oracle_retained": (run.get("task_summary") or {}).get("terminal_oracle_retained"),
        "terminal_forced_top1_oracle": (run.get("task_summary") or {}).get("terminal_forced_top1_oracle"),
        "avg_stage_survivors_after": safe_float(grouped_mean(stage_rows, "stage_name", ["survivors_after"])[0].get("survivors_after"), 0.0) if stage_rows else float("nan"),
    }
    guarded = read_json(GUARDED_ROOT / "guarded_policy.json", {}) or {}
    rows = [primary]
    for row in guarded.get("summary_rows", []):
        if row.get("dataset_name") != "ALL_FULL_LOOP":
            continue
        rows.append(
            {
                "policy": f"{row.get('threshold_policy')}::{row.get('guard_policy')}::{row.get('policy')}",
                "stage_oracle_retention": row.get("pre_terminal_oracle_retained"),
                "terminal_oracle_retained": row.get("terminal_oracle_retained"),
                "terminal_forced_top1_oracle": row.get("terminal_forced_top1_oracle"),
                "avg_stage_survivors_after": row.get("pre_terminal_survivor_count"),
                "source": "cached_all_loop_guarded_policy_v1",
            }
        )
    verdict = "MEAN_FLOOR_VERY_LOOSE_CONFIRMED" if safe_float(primary.get("stage_oracle_retention"), 0.0) >= 0.95 else "THRESHOLD_UNSTABLE"
    payload = {
        "BG_DUALANCHOR_THRESHOLD_BUDGET_V3_VERDICT": verdict,
        "primary_live_policy": primary,
        "comparison_rows": rows,
        "caveat": "Live v3 generated only the primary threshold; broader threshold/budget comparison uses cached all-loop guarded-policy diagnostics where available.",
    }
    write_json(OUT_ROOT / "threshold_budget.json", payload)
    write_csv(OUT_ROOT / "threshold_budget_rows.csv", rows)
    lines = report_lines("DualAnchor Threshold/Budget v3", "BG_DUALANCHOR_THRESHOLD_BUDGET_V3_VERDICT", verdict, [("Rows", md_table(rows[:30], ["policy", "stage_oracle_retention", "terminal_oracle_retained", "terminal_forced_top1_oracle", "avg_stage_survivors_after", "source"])), ("Caveat", [payload["caveat"]])])
    write_md(OUT_ROOT / "threshold_budget.md", lines)
    print(f"BG_DUALANCHOR_THRESHOLD_BUDGET_V3_VERDICT = {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

