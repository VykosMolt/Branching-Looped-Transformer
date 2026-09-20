from __future__ import annotations

import ast
import time
from collections import defaultdict

from bg_convergence_hairs_rs_v1_common import (
    OUT_ROOT,
    finite_mean,
    load_task_rows_csv,
    load_terminal_rows_csv,
    load_v3_rows,
    md_table,
    row_reward,
    safe_float,
    status_line,
    task_final_ids,
    terminal_policy_summary,
    write_csv,
    write_json,
    write_md,
)


def parse_indices(value: str) -> list[int]:
    try:
        return [int(x) for x in ast.literal_eval(value)]
    except Exception:
        return []


def selected_metrics(task_id: str, policy: str, indices: list[int], final_ids: list[str], rows_by_id: dict[str, dict]) -> dict[str, object]:
    rewards = [row_reward(rows_by_id[item]) for item in final_ids if item in rows_by_id]
    oracle = max(rewards) if rewards else float("nan")
    selected = [idx for idx in indices if 0 <= idx < len(rewards)]
    selected_rewards = [rewards[idx] for idx in selected]
    return {
        "task_id": task_id,
        "policy": policy,
        "selected_count": len(selected),
        "oracle_retained": 1.0 if any(abs(rewards[idx] - oracle) <= 1e-9 for idx in selected) else 0.0,
        "best_selected_reward": max(selected_rewards) if selected_rewards else float("nan"),
        "first_selected_reward": selected_rewards[0] if selected_rewards else float("nan"),
        "first_selected_oracle": 1.0 if selected and abs(rewards[selected[0]] - oracle) <= 1e-9 else 0.0,
    }


def main() -> int:
    started = time.time()
    _payload, rows, _stage_rows, task_rows_pt, _terminal_rows_pt = load_v3_rows()
    rows_by_id = {str(row.get("branch_id")): row for row in rows if row.get("branch_id")}
    task_rows = load_task_rows_csv()
    terminal_rows = load_terminal_rows_csv()
    existing_summary = terminal_policy_summary(terminal_rows)
    top_rows = [row for row in terminal_rows if row.get("policy") == "dualanchor_terminal_top5"]
    task_by_id = {str(row.get("task_id")): row for row in task_rows_pt}
    derived_rows = []
    for row in top_rows:
        task_id = str(row.get("task_id"))
        final_ids = task_final_ids(task_by_id.get(task_id, {}))
        indices = parse_indices(str(row.get("selected_indices")))
        meta = next((t for t in task_rows if t.get("task_id") == task_id), {})
        for policy, subset in (
            ("dualanchor_terminal_top4_derived", indices[:4]),
            ("dualanchor_terminal_full_budget8", list(range(len(final_ids)))),
        ):
            rec = selected_metrics(task_id, policy, subset, final_ids, rows_by_id)
            rec.update({"domain": meta.get("domain"), "split": meta.get("split"), "positive_oracle": meta.get("positive_oracle"), "terminal_reward_diverse": meta.get("terminal_reward_diverse"), "terminal_deferred": row.get("terminal_deferred"), "terminal_confident": row.get("terminal_confident")})
            derived_rows.append(rec)
    all_policy_rows = list(terminal_rows) + derived_rows
    domain_summary = {}
    for domain in ("all", "reasoning", "science"):
        selected = all_policy_rows if domain == "all" else [row for row in all_policy_rows if row.get("domain") == domain]
        domain_summary[domain] = terminal_policy_summary(selected)
    defer_rows = [row for row in task_rows if safe_float(row.get("terminal_deferred"), 0.0) > 0]
    hard_defer_rows = [row for row in defer_rows if safe_float(row.get("positive_oracle"), 0.0) > 0 and safe_float(row.get("terminal_reward_diverse"), 0.0) > 0]
    top2 = next((row for row in domain_summary["all"] if row.get("policy") == "dualanchor_terminal_top2"), {})
    top4 = next((row for row in domain_summary["all"] if row.get("policy") == "dualanchor_terminal_top4_derived"), {})
    full = next((row for row in domain_summary["all"] if row.get("policy") == "dualanchor_terminal_full_budget8"), {})
    gated = next((row for row in domain_summary["all"] if row.get("policy") == "dualanchor_confidence_gated"), {})
    if safe_float(gated.get("oracle_retained"), 0.0) >= 0.98:
        verdict = "CONFIDENCE_TOP1_READY_WITH_DEFER"
    elif safe_float(top2.get("oracle_retained"), 0.0) >= 0.98:
        verdict = "TOP2_HANDOFF_SUFFICIENT"
    elif safe_float(top4.get("oracle_retained"), 0.0) >= 0.98:
        verdict = "TOP4_HANDOFF_REQUIRED"
    elif safe_float(full.get("oracle_retained"), 0.0) >= 0.98:
        verdict = "FULL_SURVIVOR_SET_REQUIRED"
    else:
        verdict = "TERMINAL_POLICY_WEAK"
    payload = {
        "BG_TERMINAL_DEFER_POLICY_VERDICT": verdict,
        "domain_summary": domain_summary,
        "defer_rate": len(defer_rows) / max(len(task_rows), 1),
        "hard_slice_defer_count": len(hard_defer_rows),
        "derived_policy_note": "top4 and full-budget rows are derived from the terminal top5/final survivor order in v3 artifacts.",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "terminal_defer_policy.json", payload)
    write_csv(OUT_ROOT / "terminal_defer_rows.csv", all_policy_rows)
    lines = [
        "# Terminal Defer Policy v1",
        "",
        status_line("BG_TERMINAL_DEFER_POLICY_VERDICT", verdict),
        "",
        "## All Domains",
        "",
        *md_table(domain_summary["all"], ["policy", "count", "selected_count", "oracle_retained", "best_selected_reward", "first_selected_reward", "first_selected_oracle", "defer_rate"]),
        "",
        "## Reasoning",
        "",
        *md_table(domain_summary["reasoning"], ["policy", "count", "selected_count", "oracle_retained", "best_selected_reward", "first_selected_reward", "first_selected_oracle", "defer_rate"]),
        "",
        "## Science",
        "",
        *md_table(domain_summary["science"], ["policy", "count", "selected_count", "oracle_retained", "best_selected_reward", "first_selected_reward", "first_selected_oracle", "defer_rate"]),
        "",
        "## Defer Behavior",
        "",
        f"- defer rate: `{payload['defer_rate']}`",
        f"- hard-slice defer count: `{len(hard_defer_rows)}`",
        "",
        "Terminal confidence remains a gate; this analysis does not remove it.",
    ]
    write_md(OUT_ROOT / "terminal_defer_policy.md", lines)
    print(status_line("BG_TERMINAL_DEFER_POLICY_VERDICT", verdict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

