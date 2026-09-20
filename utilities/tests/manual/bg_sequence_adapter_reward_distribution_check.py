#!/usr/bin/env python3
"""Quick reward-variance check for sequence-level BG adapter optimization."""
from __future__ import annotations

import os
import time
import traceback
from collections import defaultdict
from typing import Any

from bg_sequence_adapter_common import (
    PRIMARY_MODE,
    QUICK_OUT_ROOT,
    aggregate_rows,
    avg,
    load_empirical_direction,
    load_model_tokenizer,
    load_stage1_mcq_tasks,
    load_teacher_adapter,
    random_rms_direction,
    rel,
    run_generation_condition,
    select_balanced_tasks,
    set_seed,
    task_reward_variance,
    write_json,
    write_md,
)


OUT_JSON = QUICK_OUT_ROOT / "reward_distribution.json"
OUT_MD = QUICK_OUT_ROOT / "reward_distribution.md"
TASK_CAP = min(20, max(2, int(os.environ.get("BG_SEQUENCE_REWARD_DIST_TASKS", "4"))))
MAX_NEW_TOKENS = min(96, int(os.environ.get("BG_SEQUENCE_REWARD_DIST_TOKENS", "96")))
ALPHA = min(0.02, float(os.environ.get("BG_SEQUENCE_REWARD_DIST_ALPHA", "0.01")))


def main() -> int:
    started = time.time()
    set_seed()
    QUICK_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    methods_run: list[str] = []
    try:
        tasks = select_balanced_tasks(load_stage1_mcq_tasks(), TASK_CAP)
        if not tasks:
            raise RuntimeError("no clean Stage 1 reasoning/science MCQ tasks found")
        model, tokenizer, device = load_model_tokenizer()
        raw_direction = load_empirical_direction("RAW_NONORM_READOUT")
        teacher_adapter = load_teacher_adapter(device)
        methods_run = ["no_intervention_baseline", "random_same_rms"]
        if raw_direction is not None:
            methods_run.append("raw_nonorm_static")
        if teacher_adapter is not None:
            methods_run.append("teacher_forced_adapter_checkpoint")
        for task_idx, task in enumerate(tasks):
            rows.append(
                run_generation_condition(
                    model,
                    tokenizer,
                    device,
                    task,
                    method="no_intervention_baseline",
                    seed=20260518 + task_idx,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
            )
            rows.append(
                run_generation_condition(
                    model,
                    tokenizer,
                    device,
                    task,
                    method="random_same_rms",
                    seed=20261518 + task_idx,
                    alpha=ALPHA,
                    direction=random_rms_direction(20261518 + task_idx),
                    max_new_tokens=MAX_NEW_TOKENS,
                )
            )
            if raw_direction is not None:
                rows.append(
                    run_generation_condition(
                        model,
                        tokenizer,
                        device,
                        task,
                        method="raw_nonorm_static",
                        seed=20262518 + task_idx,
                        alpha=ALPHA,
                        direction=raw_direction,
                        max_new_tokens=MAX_NEW_TOKENS,
                    )
                )
            if teacher_adapter is not None:
                rows.append(
                    run_generation_condition(
                        model,
                        tokenizer,
                        device,
                        task,
                        method="teacher_forced_adapter_checkpoint",
                        seed=20263518 + task_idx,
                        alpha=ALPHA,
                        adapter=teacher_adapter,
                        max_new_tokens=MAX_NEW_TOKENS,
                    )
                )
    except Exception as exc:
        payload = {
            "BG_SEQUENCE_REWARD_DISTRIBUTION_VERDICT": "BLOCKED",
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "traceback": traceback.format_exc()[-4000:],
            "rows": rows,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# BG Sequence Adapter Reward Distribution", "", "BG_SEQUENCE_REWARD_DISTRIBUTION_VERDICT = BLOCKED", "", f"- error: `{type(exc).__name__}: {str(exc)[:500]}`"])
        print("BG_SEQUENCE_REWARD_DISTRIBUTION_VERDICT = BLOCKED")
        return 1

    variance = task_reward_variance(rows)
    nonzero_rate = float(variance["nonzero_reward_variance_rate"])
    if not rows:
        verdict = "BLOCKED"
    elif nonzero_rate >= 0.25:
        verdict = "REWARD_SIGNAL_USABLE"
    elif nonzero_rate > 0.0:
        verdict = "REWARD_SIGNAL_WEAK"
    else:
        verdict = "REWARD_SIGNAL_FLAT"
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    baseline_by_task = {
        task_id: next((row for row in vals if row["method"] == "no_intervention_baseline"), None)
        for task_id, vals in by_task.items()
    }
    baseline_correct = sum(1 for row in baseline_by_task.values() if row and row.get("correct"))
    baseline_wrong = sum(1 for row in baseline_by_task.values() if row and row.get("parse_success") and not row.get("correct"))
    payload = {
        "BG_SEQUENCE_REWARD_DISTRIBUTION_VERDICT": verdict,
        "task_count": len(by_task),
        "generation_count": len(rows),
        "methods_run": methods_run,
        "alpha": ALPHA,
        "mode": PRIMARY_MODE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "aggregate": aggregate_rows(rows, by=("method", "alpha")),
        **variance,
        "random_static_moved_tasks": variance["moved_task_ids"],
        "baseline_correct_saturation": baseline_correct,
        "baseline_wrong_saturation": baseline_wrong,
        "phase0_task_count_reduced_for_generation_cap": TASK_CAP < 12,
        "rows": rows,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    lines = [
        "# BG Sequence Adapter Reward Distribution",
        "",
        f"BG_SEQUENCE_REWARD_DISTRIBUTION_VERDICT = {verdict}",
        "",
        f"- tasks: `{len(by_task)}`",
        f"- generations: `{len(rows)}`",
        f"- methods: `{methods_run}`",
        f"- tasks with nonzero reward variance: `{variance['tasks_with_nonzero_reward_variance']}`",
        f"- nonzero reward variance rate: `{nonzero_rate:.3f}`",
        f"- baseline correct saturation: `{baseline_correct}`",
        f"- baseline wrong saturation: `{baseline_wrong}`",
        f"- reduced for Phase 0 generation cap: `{TASK_CAP < 12}`",
        "",
        "## Aggregate",
        "",
        "| condition | n | reward mean | reward std | success | parse | delta RMS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in payload["aggregate"].items():
        lines.append(
            f"| `{key}` | {row['n']} | {row['reward_mean'] or 0.0:.3f} | {row['reward_std']:.3f} | "
            f"{row['success_rate'] or 0.0:.3f} | {row['parse_rate'] or 0.0:.3f} | {row['activation_rms_change'] or 0.0:.6f} |"
        )
    write_md(OUT_MD, lines)
    print(f"BG_SEQUENCE_REWARD_DISTRIBUTION_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict not in {"BLOCKED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
