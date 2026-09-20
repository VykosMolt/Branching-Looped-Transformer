"""Analyze grouped CV and leave-one-task-out selector stability."""
from __future__ import annotations

import time
from collections import defaultdict
from statistics import mean

from bg_hidden_origin_split_salvage_common import (
    CV_STABILITY_JSON,
    SALVAGE_EVAL_JSON,
    SALVAGE_ROOT,
    ensure_salvage_root,
    finite_mean,
    finite_pstdev,
    load_json,
    md_table,
    rate,
    rel,
    write_json,
    write_md,
)


OUT_MD = SALVAGE_ROOT / "cv_stability.md"


def verdict_for(policy_rows: list[dict[str, object]], task_count: int) -> str:
    if task_count < 4:
        return "DATA_LIMITED"
    if not policy_rows:
        return "NEGATIVE"
    best = max(policy_rows, key=lambda row: (float(row["mean_lift_vs_random"]), int(row["tasks_positive"])))
    lift = float(best["mean_lift_vs_random"])
    frac = float(best["tasks_positive"]) / max(int(best["task_count"]), 1)
    if lift > 0.05 and frac >= 0.60:
        return "STABLE_POSITIVE"
    if lift > 0.0 and frac >= 0.40:
        return "WEAK_POSITIVE"
    if lift > 0.0:
        return "UNSTABLE"
    return "NEGATIVE"


def main() -> int:
    started = time.time()
    ensure_salvage_root()
    eval_payload = load_json(SALVAGE_EVAL_JSON, {}) or {}
    rows = list(eval_payload.get("rows") or [])
    if not rows:
        payload = {"BG_HIDDEN_ORIGIN_CV_STABILITY_VERDICT": "DATA_LIMITED", "verdict": "DATA_LIMITED", "blocker": "missing eval rows"}
        write_json(CV_STABILITY_JSON, payload)
        write_md(OUT_MD, ["# Hidden-Origin Salvage CV Stability", "", "BG_HIDDEN_ORIGIN_CV_STABILITY_VERDICT = DATA_LIMITED"])
        print("BG_HIDDEN_ORIGIN_CV_STABILITY_VERDICT = DATA_LIMITED", flush=True)
        return 0

    cv_rows = [
        row for row in rows
        if row.get("subset") == "behaviorally_diverse"
        and row.get("mode_name") in {"grouped_kfold_v3", "leave_one_task_out_v3"}
    ]
    by_mode_task_policy: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in cv_rows:
        by_mode_task_policy[(str(row["mode_name"]), str(row["task_id"]), str(row["policy"]))].append(row)

    random_by_mode_task = {}
    for (mode, task, policy), vals in by_mode_task_policy.items():
        if policy == "random_top1":
            random_by_mode_task[(mode, task)] = mean(float(v["success"]) for v in vals)

    policy_summary = []
    grouped_policy_task: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (mode, task, policy), vals in by_mode_task_policy.items():
        if policy == "random_top1":
            continue
        success = mean(float(v["success"]) for v in vals)
        rand = random_by_mode_task.get((mode, task))
        if rand is None:
            continue
        grouped_policy_task[(mode, policy)].append(success - rand)

    for (mode, policy), lifts in sorted(grouped_policy_task.items()):
        policy_summary.append(
            {
                "mode": mode,
                "policy": policy,
                "task_count": len(lifts),
                "mean_lift_vs_random": finite_mean(lifts),
                "stdev_lift": finite_pstdev(lifts),
                "tasks_positive": sum(1 for x in lifts if x > 0),
                "tasks_negative": sum(1 for x in lifts if x < 0),
                "tasks_tied": sum(1 for x in lifts if x == 0),
            }
        )

    all_tasks = sorted({str(row["task_id"]) for row in cv_rows if row.get("policy") == "random_top1"})
    verdict = verdict_for(policy_summary, len(all_tasks))
    best = max(policy_summary, key=lambda row: (float(row["mean_lift_vs_random"]), int(row["tasks_positive"]))) if policy_summary else None
    task_rows = []
    for (mode, task), random_success in sorted(random_by_mode_task.items()):
        candidates = []
        for (m, t, policy), vals in by_mode_task_policy.items():
            if m == mode and t == task and policy != "random_top1":
                candidates.append((policy, mean(float(v["success"]) for v in vals)))
        if not candidates:
            continue
        best_policy, best_success = max(candidates, key=lambda x: x[1])
        task_rows.append(
            {
                "mode": mode,
                "task_id": task,
                "random_top1": random_success,
                "best_policy": best_policy,
                "best_success": best_success,
                "best_lift": best_success - random_success,
            }
        )

    payload = {
        "BG_HIDDEN_ORIGIN_CV_STABILITY_VERDICT": verdict,
        "verdict": verdict,
        "task_count": len(all_tasks),
        "best_policy_summary": best,
        "policy_summary": policy_summary,
        "task_rows": task_rows,
        "one_task_dominates": bool(task_rows and max(abs(float(row["best_lift"])) for row in task_rows) > max(0.20, sum(abs(float(row["best_lift"])) for row in task_rows) * 0.40)),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(CV_STABILITY_JSON, payload)
    lines = [
        "# Hidden-Origin Salvage CV Stability",
        "",
        f"BG_HIDDEN_ORIGIN_CV_STABILITY_VERDICT = {verdict}",
        "",
        f"- task_count: `{len(all_tasks)}`",
        f"- best_policy_summary: `{best}`",
        f"- one_task_dominates: `{payload['one_task_dominates']}`",
        "",
        "## Policy Lift By Task Fold",
        "",
    ]
    lines.extend(
        md_table(
            [
                {
                    "mode": row["mode"],
                    "policy": row["policy"],
                    "tasks": row["task_count"],
                    "mean_lift": rate(row["mean_lift_vs_random"]),
                    "stdev": rate(row["stdev_lift"]),
                    "positive": row["tasks_positive"],
                    "negative": row["tasks_negative"],
                }
                for row in sorted(policy_summary, key=lambda r: float(r["mean_lift_vs_random"]), reverse=True)[:80]
            ],
            ["mode", "policy", "tasks", "mean_lift", "stdev", "positive", "negative"],
        )
    )
    lines.extend(["", "## Per-Task Best", ""])
    lines.extend(
        md_table(
            [
                {
                    "mode": row["mode"],
                    "task_id": row["task_id"],
                    "random": rate(row["random_top1"]),
                    "best_policy": row["best_policy"],
                    "best_success": rate(row["best_success"]),
                    "lift": rate(row["best_lift"]),
                }
                for row in sorted(task_rows, key=lambda r: (r["mode"], r["task_id"]))[:120]
            ],
            ["mode", "task_id", "random", "best_policy", "best_success", "lift"],
        )
    )
    lines.extend(["", f"Wrote `{rel(CV_STABILITY_JSON)}`."])
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_CV_STABILITY_VERDICT = {verdict}", flush=True)
    print(f"Wrote {rel(CV_STABILITY_JSON)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

