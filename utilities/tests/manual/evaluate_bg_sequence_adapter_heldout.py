#!/usr/bin/env python3
"""Heldout sampled free-generation evaluation for the sequence adapter."""
from __future__ import annotations

import os
import time
import traceback
from collections import defaultdict
from typing import Any

from bg_sequence_adapter_common import (
    OUT_ROOT,
    aggregate_rows,
    avg,
    load_empirical_direction,
    load_json,
    load_model_tokenizer,
    load_sequence_adapter,
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


OUT_JSON = OUT_ROOT / "heldout_free_generation_eval.json"
OUT_MD = OUT_ROOT / "heldout_free_generation_eval.md"
MAX_NEW_TOKENS = min(128, int(os.environ.get("BG_SEQUENCE_HELDOUT_TOKENS", "128")))
N_SAMPLES = max(4, int(os.environ.get("BG_SEQUENCE_HELDOUT_SAMPLES", "4")))
RUN_DETERMINISTIC = os.environ.get("BG_SEQUENCE_HELDOUT_DETERMINISTIC", "1") == "1"


def per_task_rates(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("decode") != "sampled":
            continue
        groups[(str(row["task_id"]), str(row["method"]))].append(row)
    out: dict[str, dict[str, Any]] = {}
    for (task_id, method), vals in groups.items():
        out.setdefault(task_id, {})[method] = {
            "n": len(vals),
            "success_rate": avg(v.get("correct") for v in vals),
            "reward_mean": avg(v.get("reward") for v in vals),
            "parse_rate": avg(v.get("parse_success") for v in vals),
        }
    return out


def main() -> int:
    started = time.time()
    set_seed()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    try:
        dataset = load_sequence_dataset()
        heldout = rows_for_split(dataset, "heldout")
        if len(heldout) < 8:
            raise RuntimeError(f"insufficient heldout tasks: {len(heldout)}")
        training = load_json(OUT_ROOT / "sequence_training_log.json", {})
        best_alpha = float(training.get("best_val_alpha") or 0.01)
        if best_alpha not in {0.005, 0.01, 0.02}:
            best_alpha = 0.01
        model, tokenizer, device = load_model_tokenizer()
        sequence_adapter = load_sequence_adapter(device)
        if sequence_adapter is None:
            raise RuntimeError("best_sequence_adapter.pt missing")
        teacher_adapter = load_teacher_adapter(device)
        raw_direction = load_empirical_direction("RAW_NONORM_READOUT")
        for task_idx, task in enumerate(heldout):
            methods: list[tuple[str, dict[str, Any]]] = [
                ("no_intervention_baseline", {"alpha": 0.0}),
                ("random_same_rms", {"alpha": best_alpha, "direction": random_rms_direction(20261518 + task_idx)}),
                ("trained_sequence_adapter", {"alpha": best_alpha, "adapter": sequence_adapter}),
            ]
            if raw_direction is not None:
                methods.insert(2, ("raw_nonorm_static", {"alpha": best_alpha, "direction": raw_direction}))
            if teacher_adapter is not None:
                methods.insert(-1, ("teacher_forced_adapter_checkpoint", {"alpha": best_alpha, "adapter": teacher_adapter}))
            for method, kwargs in methods:
                for sample_idx in range(N_SAMPLES):
                    row = run_generation_condition(
                        model,
                        tokenizer,
                        device,
                        task,
                        method=method,
                        seed=20270518 + task_idx * 1000 + sample_idx * 10 + len(rows),
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True,
                        sample_index=sample_idx + 1,
                        **kwargs,
                    )
                    row["split"] = "heldout"
                    rows.append(row)
                if RUN_DETERMINISTIC:
                    drow = run_generation_condition(
                        model,
                        tokenizer,
                        device,
                        task,
                        method=method,
                        seed=20280518 + task_idx * 100 + len(rows),
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        sample_index=0,
                        **kwargs,
                    )
                    drow["split"] = "heldout"
                    rows.append(drow)
            write_json(OUT_JSON.with_suffix(".partial.json"), {"BG_SEQUENCE_ADAPTER_HELDOUT_VERDICT": "PARTIAL", "rows": rows})
    except Exception as exc:
        payload = {
            "BG_SEQUENCE_ADAPTER_HELDOUT_VERDICT": "INSUFFICIENT",
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "traceback": traceback.format_exc()[-4000:],
            "rows": rows,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# BG Sequence Adapter Heldout Free-Generation Evaluation", "", "BG_SEQUENCE_ADAPTER_HELDOUT_VERDICT = INSUFFICIENT", "", f"- error: `{type(exc).__name__}: {str(exc)[:500]}`"])
        print("BG_SEQUENCE_ADAPTER_HELDOUT_VERDICT = INSUFFICIENT")
        return 1

    aggregate = aggregate_rows(rows, by=("method", "alpha", "decode"))
    sampled = [row for row in rows if row.get("decode") == "sampled"]
    sampled_agg = aggregate_rows(sampled, by=("method", "alpha"))
    adapter_best = max((row for row in sampled_agg.values() if row.get("method") == "trained_sequence_adapter"), key=lambda r: r.get("success_rate") or -1e9, default=None)
    baseline_best = max((row for row in sampled_agg.values() if row.get("method") == "no_intervention_baseline"), key=lambda r: r.get("success_rate") or -1e9, default=None)
    non_adapter_best = max(
        (row for row in sampled_agg.values() if row.get("method") != "trained_sequence_adapter"),
        key=lambda r: r.get("success_rate") or -1e9,
        default=None,
    )
    adapter_success = float((adapter_best or {}).get("success_rate") or 0.0)
    baseline_success = float((baseline_best or {}).get("success_rate") or 0.0)
    non_adapter_success = float((non_adapter_best or {}).get("success_rate") or 0.0)
    task_rates = per_task_rates(rows)
    moved = 0
    adapter_only = 0
    random_static_moved = 0
    invariant = 0
    for task_id, by_method in task_rates.items():
        vals = [float(v.get("success_rate") or 0.0) for v in by_method.values()]
        if len(set(vals)) <= 1:
            invariant += 1
            continue
        moved += 1
        base = float((by_method.get("no_intervention_baseline") or {}).get("success_rate") or 0.0)
        adapter_rate = float((by_method.get("trained_sequence_adapter") or {}).get("success_rate") or 0.0)
        non_adapter = max(
            [float(v.get("success_rate") or 0.0) for method, v in by_method.items() if method not in {"trained_sequence_adapter", "no_intervention_baseline"}] or [base]
        )
        if adapter_rate > max(base, non_adapter):
            adapter_only += 1
        if non_adapter > base:
            random_static_moved += 1
    stability_bad = any(row.get("cuda_error") or row.get("nan_or_inf_activations") for row in rows) or (
        adapter_best is not None and float(adapter_best.get("parse_rate") or 1.0) < 0.5
    )
    if stability_bad:
        verdict = "DESTABILIZING"
    elif adapter_success > max(baseline_success, non_adapter_success) and adapter_only >= 2:
        verdict = "ADAPTER_SPECIFIC_FREE_GEN_LIFT"
    elif adapter_success > max(baseline_success, non_adapter_success) and adapter_only >= 1:
        verdict = "WEAK_ADAPTER_SPECIFIC_LIFT"
    else:
        verdict = "NO_ADAPTER_SPECIFIC_TRANSFER"
    payload = {
        "BG_SEQUENCE_ADAPTER_HELDOUT_VERDICT": verdict,
        "heldout_tasks": len({row["task_id"] for row in rows}),
        "n_samples": N_SAMPLES,
        "max_new_tokens": MAX_NEW_TOKENS,
        "best_alpha": float((adapter_best or {}).get("alpha") or 0.01),
        "rows": rows,
        "aggregate": aggregate,
        "sampled_aggregate": sampled_agg,
        "per_task_success_rates": task_rates,
        "mean_success_over_samples": adapter_success,
        "baseline_success_over_samples": baseline_success,
        "best_non_adapter_success_over_samples": non_adapter_success,
        "adapter_specific_lift_over_baseline": adapter_success - baseline_success,
        "adapter_specific_lift_over_random_static_teacher": adapter_success - non_adapter_success,
        "moved_task_count": moved,
        "invariant_task_count": invariant,
        "adapter_only_moved_task_count": adapter_only,
        "random_static_moved_task_count": random_static_moved,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    lines = [
        "# BG Sequence Adapter Heldout Free-Generation Evaluation",
        "",
        f"BG_SEQUENCE_ADAPTER_HELDOUT_VERDICT = {verdict}",
        "",
        f"- heldout tasks: `{payload['heldout_tasks']}`",
        f"- sampled n: `{N_SAMPLES}`",
        f"- adapter sampled success: `{adapter_success:.3f}`",
        f"- baseline sampled success: `{baseline_success:.3f}`",
        f"- best non-adapter sampled success: `{non_adapter_success:.3f}`",
        f"- adapter-only moved tasks: `{adapter_only}`",
        f"- random/static moved tasks: `{random_static_moved}`",
        f"- invariant tasks: `{invariant}`",
        "",
        "## Sampled Aggregate",
        "",
        "| condition | n | reward | success | parse | repetition | empty | hit max | delta RMS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in sampled_agg.items():
        lines.append(
            f"| `{key}` | {row['n']} | {row['reward_mean'] or 0.0:.3f} | {row['success_rate'] or 0.0:.3f} | "
            f"{row['parse_rate'] or 0.0:.3f} | {row['repetition_rate'] or 0.0:.3f} | {row['empty_output_rate'] or 0.0:.3f} | "
            f"{row['hit_max_tokens_rate'] or 0.0:.3f} | {row['activation_rms_change'] or 0.0:.6f} |"
        )
    write_md(OUT_MD, lines)
    print(f"BG_SEQUENCE_ADAPTER_HELDOUT_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
