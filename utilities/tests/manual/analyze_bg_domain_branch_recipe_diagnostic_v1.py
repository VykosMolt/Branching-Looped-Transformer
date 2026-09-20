from __future__ import annotations

import time
from collections import defaultdict

from bg_convergence_hairs_rs_v1_common import OUT_ROOT, finite_mean, load_task_rows_csv, load_v3_rows, md_table, read_csv, safe_float, status_line, write_csv, write_json, write_md


PERTURB_ROWS = OUT_ROOT.parent / "bg_dualanchor_architecture_looped_stratified_probe_v3_2026-05-31" / "perturbation_escalation_rows.csv"


def main() -> int:
    started = time.time()
    _payload, rows, _stage_rows, _task_rows, _terminal_rows = load_v3_rows()
    task_rows = [row for row in load_task_rows_csv() if row.get("domain") in {"reasoning", "science"}]
    perturb_rows = read_csv(PERTURB_ROWS)
    domain_summary = []
    for domain in ("reasoning", "science"):
        vals = [row for row in task_rows if row.get("domain") == domain]
        domain_summary.append(
            {
                "domain": domain,
                "task_count": len(vals),
                "positive_oracle_rate": finite_mean(row.get("positive_oracle") for row in vals),
                "reward_diverse_rate": finite_mean(row.get("terminal_reward_diverse") for row in vals),
                "terminal_best_reward": finite_mean(row.get("terminal_best_reward") for row in vals),
                "forced_top1_reward": finite_mean(row.get("terminal_forced_top1_reward") for row in vals),
                "stage_false_prunes": finite_mean(row.get("stage_false_prunes") for row in vals),
            }
        )
    birth_summary = []
    for domain in ("reasoning", "science"):
        domain_rows = [row for row in rows if row.get("domain") == domain]
        by_stage = defaultdict(list)
        for row in domain_rows:
            by_stage[str(row.get("birth_stage"))].append(row)
        for stage, vals in sorted(by_stage.items()):
            birth_summary.append(
                {
                    "domain": domain,
                    "birth_stage": stage,
                    "count": len(vals),
                    "mean_reward": finite_mean(row.get("reward") for row in vals),
                    "positive_rate": finite_mean(1.0 if safe_float(row.get("reward"), 0.0) > 0 else 0.0 for row in vals),
                    "child_parent_delta": finite_mean(row.get("reward_delta_from_parent") for row in vals if row.get("parent_branch_id")),
                }
            )
    trigger_summary = []
    for domain in ("reasoning", "science"):
        vals = [row for row in perturb_rows if row.get("domain") == domain]
        trigger_summary.append(
            {
                "domain": domain,
                "stage_rows": len(vals),
                "trigger_rate": finite_mean(row.get("would_trigger_escalation") for row in vals),
                "false_prune_trigger_rate": finite_mean(row.get("trigger_false_prune") for row in vals),
                "low_survivor_trigger_rate": finite_mean(row.get("trigger_low_survivors") for row in vals),
            }
        )
    science = next(row for row in domain_summary if row["domain"] == "science")
    reasoning = next(row for row in domain_summary if row["domain"] == "reasoning")
    if safe_float(science.get("positive_oracle_rate"), 0.0) + 0.20 < safe_float(reasoning.get("positive_oracle_rate"), 0.0):
        verdict = "SCIENCE_NEEDS_DIFFERENT_RECIPE"
    elif safe_float(reasoning.get("positive_oracle_rate"), 0.0) < 0.30:
        verdict = "REASONING_NEEDS_DIFFERENT_RECIPE"
    elif any(row.get("domain") == "science" and safe_float(row.get("trigger_rate"), 0.0) > 0.15 for row in trigger_summary):
        verdict = "SCIENCE_NEEDS_DIFFERENT_RECIPE"
    else:
        verdict = "SHARED_RECIPE_SUFFICIENT"
    payload = {
        "BG_DOMAIN_BRANCH_RECIPE_DIAGNOSTIC_VERDICT": verdict,
        "domain_summary": domain_summary,
        "birth_stage_summary": birth_summary,
        "perturbation_trigger_summary": trigger_summary,
        "note": "Recipe comparisons are diagnostic from v3 and trigger-audit artifacts; no steering or new generation was run.",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "domain_branch_recipe.json", payload)
    write_csv(OUT_ROOT / "domain_branch_recipe_rows.csv", birth_summary)
    lines = [
        "# Domain Branch Recipe Diagnostic v1",
        "",
        status_line("BG_DOMAIN_BRANCH_RECIPE_DIAGNOSTIC_VERDICT", verdict),
        "",
        "This is diagnostic only. No steering, model training, or new branch-generation recipe was executed.",
        "",
        "## Domain Summary",
        "",
        *md_table(domain_summary, ["domain", "task_count", "positive_oracle_rate", "reward_diverse_rate", "terminal_best_reward", "forced_top1_reward", "stage_false_prunes"]),
        "",
        "## Perturbation Trigger Summary",
        "",
        *md_table(trigger_summary, ["domain", "stage_rows", "trigger_rate", "false_prune_trigger_rate", "low_survivor_trigger_rate"]),
        "",
        "## Birth Stage Summary",
        "",
        *md_table(birth_summary[:40], ["domain", "birth_stage", "count", "mean_reward", "positive_rate", "child_parent_delta"]),
    ]
    write_md(OUT_ROOT / "domain_branch_recipe.md", lines)
    print(status_line("BG_DOMAIN_BRANCH_RECIPE_DIAGNOSTIC_VERDICT", verdict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

