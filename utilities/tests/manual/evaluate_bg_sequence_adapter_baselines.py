#!/usr/bin/env python3
"""Baseline free-generation evaluation before sequence-level adapter training."""
from __future__ import annotations

import os
import time
import traceback
from typing import Any

from bg_sequence_adapter_common import (
    OUT_ROOT,
    aggregate_rows,
    avg,
    load_empirical_direction,
    load_model_tokenizer,
    load_sequence_dataset,
    load_teacher_adapter,
    random_rms_direction,
    rel,
    rows_for_split,
    run_generation_condition,
    set_seed,
    write_json,
    write_md,
)


OUT_JSON = OUT_ROOT / "baseline_eval.json"
OUT_MD = OUT_ROOT / "baseline_eval.md"
MAX_NEW_TOKENS = min(96, int(os.environ.get("BG_SEQUENCE_BASELINE_TOKENS", "96")))
ALPHA = min(0.02, float(os.environ.get("BG_SEQUENCE_BASELINE_ALPHA", "0.01")))
SAMPLED_SAMPLES = max(0, min(2, int(os.environ.get("BG_SEQUENCE_BASELINE_SAMPLED_SAMPLES", "0"))))


def main() -> int:
    started = time.time()
    set_seed()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    try:
        dataset = load_sequence_dataset()
        tasks = []
        for split in ["train", "val", "heldout"]:
            tasks.extend(rows_for_split(dataset, split))
        if not tasks:
            raise RuntimeError("sequence_adapter_dataset.json missing or empty")
        model, tokenizer, device = load_model_tokenizer()
        raw_direction = load_empirical_direction("RAW_NONORM_READOUT")
        teacher_adapter = load_teacher_adapter(device)
        for task_idx, task in enumerate(tasks):
            split = task.get("split")
            conditions: list[tuple[str, dict[str, Any]]] = [
                ("no_intervention_baseline", {"alpha": 0.0}),
                ("random_same_rms", {"alpha": ALPHA, "direction": random_rms_direction(20261518 + task_idx)}),
            ]
            if raw_direction is not None:
                conditions.append(("raw_nonorm_static", {"alpha": ALPHA, "direction": raw_direction}))
            if teacher_adapter is not None:
                conditions.append(("teacher_forced_adapter_checkpoint", {"alpha": ALPHA, "adapter": teacher_adapter}))
            for method, kwargs in conditions:
                row = run_generation_condition(
                    model,
                    tokenizer,
                    device,
                    task,
                    method=method,
                    seed=20260518 + task_idx * 100 + len(rows),
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    **kwargs,
                )
                row["split"] = split
                rows.append(row)
                for sample_idx in range(SAMPLED_SAMPLES):
                    srow = run_generation_condition(
                        model,
                        tokenizer,
                        device,
                        task,
                        method=method,
                        seed=20270518 + task_idx * 100 + sample_idx * 10 + len(rows),
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True,
                        sample_index=sample_idx + 1,
                        **kwargs,
                    )
                    srow["split"] = split
                    rows.append(srow)
    except Exception as exc:
        payload = {
            "BG_SEQUENCE_BASELINE_EVAL_VERDICT": "BLOCKED",
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "traceback": traceback.format_exc()[-4000:],
            "rows": rows,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# BG Sequence Adapter Baseline Evaluation", "", "BG_SEQUENCE_BASELINE_EVAL_VERDICT = BLOCKED", "", f"- error: `{type(exc).__name__}: {str(exc)[:500]}`"])
        print("BG_SEQUENCE_BASELINE_EVAL_VERDICT = BLOCKED")
        return 1

    aggregate = aggregate_rows(rows, by=("split", "method", "alpha", "decode"))
    parse_rate = avg(row["parse_success"] for row in rows) or 0.0
    if not rows:
        verdict = "BLOCKED"
    elif parse_rate >= 0.75:
        verdict = "READY"
    else:
        verdict = "PARTIAL"
    payload = {
        "BG_SEQUENCE_BASELINE_EVAL_VERDICT": verdict,
        "max_new_tokens": MAX_NEW_TOKENS,
        "sampled_samples_per_task": SAMPLED_SAMPLES,
        "use_cache": False,
        "alpha": ALPHA,
        "rows": rows,
        "aggregate": aggregate,
        "parse_rate": parse_rate,
        "success_rate": avg(row["correct"] for row in rows),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    lines = [
        "# BG Sequence Adapter Baseline Evaluation",
        "",
        f"BG_SEQUENCE_BASELINE_EVAL_VERDICT = {verdict}",
        "",
        f"- rows: `{len(rows)}`",
        f"- parse rate: `{parse_rate:.3f}`",
        f"- sampled samples per task: `{SAMPLED_SAMPLES}`",
        f"- use_cache: `false`",
        "",
        "## Aggregate",
        "",
        "| condition | n | reward | success | parse | repetition | empty | hit max | delta RMS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in aggregate.items():
        lines.append(
            f"| `{key}` | {row['n']} | {row['reward_mean'] or 0.0:.3f} | {row['success_rate'] or 0.0:.3f} | "
            f"{row['parse_rate'] or 0.0:.3f} | {row['repetition_rate'] or 0.0:.3f} | {row['empty_output_rate'] or 0.0:.3f} | "
            f"{row['hit_max_tokens_rate'] or 0.0:.3f} | {row['activation_rms_change'] or 0.0:.6f} |"
        )
    write_md(OUT_MD, lines)
    print(f"BG_SEQUENCE_BASELINE_EVAL_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
