from __future__ import annotations

from collections import Counter

from bg_dualanchor_arch_loop_v3_common import OUT_ROOT, finite_mean, grouped_mean, load_lineage_rows, load_task_rows, md_table, read_csv, report_lines, safe_float, write_csv, write_json, write_md


def main() -> int:
    rows = load_lineage_rows()
    tasks = load_task_rows()
    recovery_rows = read_csv(OUT_ROOT / "false_prune_recovery_rows.csv")
    child_rows = [row for row in rows if row.get("parent_branch_id")]
    depth_summary = grouped_mean(rows, "perturb_count", ["reward", "parse_success", "reward_delta_from_parent"])
    birth_summary = grouped_mean([row for row in rows if row.get("birth_stage") != "root"], "birth_stage", ["reward", "parse_success", "reward_delta_from_parent"])
    final_perturbed_fraction = finite_mean(row.get("final_perturbed_fraction") for row in tasks)
    forced_perturbed = finite_mean(1.0 if safe_float(row.get("forced_top1_perturb_count"), 0.0) > 0 else 0.0 for row in tasks)
    lift = forced_perturbed - final_perturbed_fraction
    child_summary = {
        "child_count": len(child_rows),
        "child_improved_parent_rate": finite_mean(1.0 if safe_float(row.get("reward_delta_from_parent"), 0.0) > 0 else 0.0 for row in child_rows),
        "child_tied_parent_rate": finite_mean(1.0 if safe_float(row.get("reward_delta_from_parent"), 999.0) == 0 else 0.0 for row in child_rows),
        "child_worse_parent_rate": finite_mean(1.0 if safe_float(row.get("reward_delta_from_parent"), 0.0) < 0 else 0.0 for row in child_rows),
        "mean_child_parent_delta": finite_mean(row.get("reward_delta_from_parent") for row in child_rows),
        "perturbed_candidate_fraction_final": final_perturbed_fraction,
        "observed_forced_top1_perturbed_rate": forced_perturbed,
        "lift_over_count_share": lift,
    }
    verdict = "RECOVERY_CONFIRMED" if recovery_rows and finite_mean(row.get("recovered") for row in recovery_rows) >= 0.95 else "PERTURBATION_MOSTLY_TIES"
    if safe_float(child_summary["child_improved_parent_rate"], 0.0) > safe_float(child_summary["child_worse_parent_rate"], 1.0):
        verdict = "RECURSIVE_PERTURBATION_USEFUL"
    payload = {
        "BG_DUALANCHOR_LINEAGE_RECOVERY_V3_VERDICT": verdict,
        "child_summary": child_summary,
        "depth_summary": depth_summary,
        "birth_stage_summary": birth_summary,
        "false_prune_recovery_rows": recovery_rows,
        "false_prune_by_stage": dict(Counter(row.get("stage_name") for row in recovery_rows)),
    }
    write_json(OUT_ROOT / "lineage_recovery.json", payload)
    write_csv(OUT_ROOT / "lineage_recovery_rows.csv", recovery_rows)
    lines = report_lines("DualAnchor Lineage/Recovery v3", "BG_DUALANCHOR_LINEAGE_RECOVERY_V3_VERDICT", verdict, [("Child Summary", [f"- {k}: `{v}`" for k, v in child_summary.items()]), ("By Perturb Count", md_table(depth_summary, ["perturb_count", "count", "reward", "parse_success", "reward_delta_from_parent"])), ("By Birth Stage", md_table(birth_summary, ["birth_stage", "count", "reward", "parse_success", "reward_delta_from_parent"])), ("False-Prune Recovery", md_table(recovery_rows, ["task_id", "domain", "stage_name", "terminal_oracle_retained", "recovered"]))])
    write_md(OUT_ROOT / "lineage_recovery.md", lines)
    print(f"BG_DUALANCHOR_LINEAGE_RECOVERY_V3_VERDICT = {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

