from __future__ import annotations

from bg_dualanchor_arch_loop_v3_common import OUT_ROOT, RUN_JSON, V2_ROOT, finite_mean, read_json, report_lines, safe_float, write_json, write_md


def main() -> int:
    v2 = read_json(V2_ROOT / "architecture_looped_stratified_probe.json", {}) or {}
    v3 = read_json(RUN_JSON, {}) or {}
    if not v2 or not v3:
        verdict = "INSUFFICIENT"
        drift = {}
    else:
        metrics = [
            "stage_oracle_retention",
        ]
        drift = {key: safe_float(v3.get(key)) - safe_float(v2.get(key)) for key in metrics}
        v2_tasks = {row.get("task_id"): row for row in v2.get("task_rows", [])}
        v3_tasks = {row.get("task_id"): row for row in v3.get("task_rows", [])}
        overlap = sorted(set(v2_tasks) & set(v3_tasks))
        overlap_drift = {
            "overlap_task_count": len(overlap),
            "terminal_oracle_retained_drift": finite_mean(safe_float(v3_tasks[t].get("terminal_oracle_retained")) - safe_float(v2_tasks[t].get("terminal_oracle_retained")) for t in overlap),
            "terminal_forced_top1_oracle_drift": finite_mean(safe_float(v3_tasks[t].get("terminal_forced_top1_oracle")) - safe_float(v2_tasks[t].get("terminal_forced_top1_oracle")) for t in overlap),
            "terminal_best_reward_drift": finite_mean(safe_float(v3_tasks[t].get("terminal_best_reward")) - safe_float(v2_tasks[t].get("terminal_best_reward")) for t in overlap),
        }
        drift.update(overlap_drift)
        verdict = "REPRODUCED" if len(overlap) >= 20 and abs(drift["terminal_oracle_retained_drift"]) <= 0.05 else "SMALL_DRIFT"
    payload = {"BG_DUALANCHOR_ARCH_LOOP_V3_REPRODUCTION_VERDICT": verdict, "drift": drift}
    write_json(OUT_ROOT / "cached_reproduction.json", payload)
    lines = report_lines("DualAnchor v2/v3 Reproduction", "BG_DUALANCHOR_ARCH_LOOP_V3_REPRODUCTION_VERDICT", verdict, [("Drift", [f"- {k}: `{v}`" for k, v in drift.items()])])
    write_md(OUT_ROOT / "cached_reproduction.md", lines)
    print(f"BG_DUALANCHOR_ARCH_LOOP_V3_REPRODUCTION_VERDICT = {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

