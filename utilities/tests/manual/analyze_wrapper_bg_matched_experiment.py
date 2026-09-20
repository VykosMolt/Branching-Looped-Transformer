"""Analyze wrapper-matched BG candidate-selection results."""

from __future__ import annotations

import random
from collections import Counter
from statistics import mean
from typing import Any

from wrapper_bg_matched_lib import OUT_DIR, label_is_success, load_json, write_json, write_text


TASK_SUITE_JSON = OUT_DIR / "task_suite.json"
TRACES_JSON = OUT_DIR / "candidate_traces.json"
EVAL_JSON = OUT_DIR / "candidate_eval.json"
FEATURE_MD = OUT_DIR / "wrapper_candidate_features.md"
SELECTION_JSON = OUT_DIR / "bg_selection.json"
OUT_JSON = OUT_DIR / "analysis.json"
OUT_MD = OUT_DIR / "analysis.md"


def bootstrap_delta(rows: list[dict[str, Any]], a_key: str, b_key: str, n_boot: int = 1000) -> dict[str, float]:
    if len(rows) < 2:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = random.Random(20260518)
    vals = []
    for _ in range(n_boot):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        a = mean(1.0 if label_is_success(row[a_key]) else 0.0 for row in sample)
        b = mean(1.0 if label_is_success(row[b_key]) else 0.0 for row in sample)
        vals.append(a - b)
    vals.sort()
    return {"mean": mean(vals), "ci_low": vals[int(0.025 * len(vals))], "ci_high": vals[int(0.975 * len(vals))]}


def main() -> None:
    task_suite = load_json(TASK_SUITE_JSON)
    traces = load_json(TRACES_JSON)
    eval_payload = load_json(EVAL_JSON)
    selection = load_json(SELECTION_JSON)
    by_task = selection.get("by_task", {})
    eval_by_task = eval_payload.get("by_task", {})
    rows = list(by_task.values())
    non_devil = [row for row in rows if not row.get("is_devil")]
    matched = [row for row in non_devil if not row.get("wrapper_final_not_in_trace") and row.get("feature_candidate_count", 0) >= 1]
    oracle_rows = [row for row in non_devil if row.get("oracle_reachable")]
    error_cases = {
        "wrapper_correct_bg_wrong": [
            row["task_id"] for row in matched
            if label_is_success(row["wrapper_final_label"]) and not label_is_success(row["bg_conservative_label"])
        ],
        "bg_correct_wrapper_wrong": [
            row["task_id"] for row in matched
            if label_is_success(row["bg_conservative_label"]) and not label_is_success(row["wrapper_final_label"])
        ],
        "oracle_existed_neither_selected": [
            row["task_id"] for row in matched
            if row.get("oracle_reachable")
            and not label_is_success(row["wrapper_final_label"])
            and not label_is_success(row["bg_conservative_label"])
        ],
        "no_viable_candidate": [row["task_id"] for row in non_devil if not row.get("oracle_reachable")],
    }
    devil = [row for row in rows if row.get("is_devil")]
    devil_analysis = {
        row["task_id"]: {
            "oracle_reachable": row.get("oracle_reachable"),
            "wrapper_label": row.get("wrapper_final_label"),
            "bg_label": row.get("bg_conservative_label"),
            "candidate_count": row.get("candidate_count"),
            "feature_candidate_count": row.get("feature_candidate_count"),
        }
        for row in devil
    }
    matched_delta = bootstrap_delta(matched, "bg_conservative_label", "wrapper_final_label") if len(matched) >= 8 else {}
    small_n = len(matched) < 20
    compute_mismatch = any(row.get("wrapper_final_not_in_trace") for row in rows)
    reachability_rate = eval_payload.get("summary", {}).get("non_devil_oracle_success_rate", 0.0)
    if selection.get("WRAPPER_MATCHED_BG_VERDICT") in {"HELPS", "NEUTRAL", "HURTS"} and len(matched) >= 8:
        experiment_verdict = "READY"
    elif rows and len(matched) >= 4:
        experiment_verdict = "PARTIAL"
    elif rows:
        experiment_verdict = "INSUFFICIENT"
    else:
        experiment_verdict = "BLOCKED"

    payload = {
        "WRAPPER_MATCHED_EXPERIMENT_VERDICT": experiment_verdict,
        "WRAPPER_MATCHED_BG_VERDICT": selection.get("WRAPPER_MATCHED_BG_VERDICT"),
        "BG_VS_RANDOM_VERDICT": selection.get("BG_VS_RANDOM_VERDICT"),
        "BG_VS_STAGE_HEURISTIC_VERDICT": selection.get("BG_VS_STAGE_HEURISTIC_VERDICT"),
        "WRAPPER_ORACLE_GAP_VERDICT": selection.get("WRAPPER_ORACLE_GAP_VERDICT"),
        "candidate_generator_quality": {
            "oracle_reachability_rate": reachability_rate,
            "label_counts": eval_payload.get("summary", {}).get("label_counts", {}),
            "stage_distribution": traces.get("summary", {}).get("stage_counts", {}),
            "malformed_count": eval_payload.get("summary", {}).get("label_counts", {}).get("malformed", 0),
            "runtime_error_count": eval_payload.get("summary", {}).get("label_counts", {}).get("runtime_error", 0),
        },
        "wrapper_final_quality": {
            "pass_rate": selection.get("summary", {}).get("wrapper_final_pass_rate"),
            "selected_stage_counts": selection.get("summary", {}).get("selected_stage_counts", {}).get("wrapper", {}),
            "oracle_gap_verdict": selection.get("WRAPPER_ORACLE_GAP_VERDICT"),
        },
        "bg_quality": {
            "conservative_pass_rate": selection.get("summary", {}).get("bg_conservative_matched_pass_rate"),
            "code_backup_label_counts": selection.get("summary", {}).get("label_counts_by_policy", {}).get("code_backup", {}),
            "selected_stage_counts": selection.get("summary", {}).get("selected_stage_counts", {}).get("bg_conservative", {}),
            "malformed_runtime_selection_count": sum(
                1 for row in rows if row.get("bg_conservative_label") in {"malformed", "runtime_error", "not_code", "safety_rejected"}
            ),
        },
        "matched_comparisons": selection.get("summary", {}),
        "bootstrap_bg_minus_wrapper": matched_delta,
        "head_comparison": {
            "objective_mixed_conservative_labels": dict(Counter(row.get("bg_conservative_label") for row in rows)),
            "code_backup_labels": dict(Counter(row.get("bg_code_backup_label") for row in rows)),
            "head_disagreement_tasks": [
                row["task_id"] for row in rows if row.get("bg_conservative_uid") != row.get("bg_code_backup_uid")
            ],
        },
        "error_cases": error_cases,
        "devil_task_analysis": devil_analysis,
        "warnings": {
            "SMALL_N_WARNING": small_n,
            "COMPUTE_MISMATCH_WARNING": compute_mismatch,
            "GENERATOR_REACHABILITY_LIMITED": reachability_rate < 0.25,
        },
        "inputs": {
            "task_suite": str(TASK_SUITE_JSON),
            "traces": str(TRACES_JSON),
            "candidate_eval": str(EVAL_JSON),
            "feature_report": str(FEATURE_MD),
            "selection": str(SELECTION_JSON),
        },
    }
    write_json(OUT_JSON, payload)
    lines = [
        "# Wrapper-Matched BG Experiment Analysis",
        "",
        f"WRAPPER_MATCHED_EXPERIMENT_VERDICT = {experiment_verdict}",
        f"WRAPPER_MATCHED_BG_VERDICT = {selection.get('WRAPPER_MATCHED_BG_VERDICT')}",
        f"BG_VS_RANDOM_VERDICT = {selection.get('BG_VS_RANDOM_VERDICT')}",
        f"BG_VS_STAGE_HEURISTIC_VERDICT = {selection.get('BG_VS_STAGE_HEURISTIC_VERDICT')}",
        f"WRAPPER_ORACLE_GAP_VERDICT = {selection.get('WRAPPER_ORACLE_GAP_VERDICT')}",
        "",
        "## Candidate Generator Quality",
        "",
        f"- oracle reachability rate: `{reachability_rate:.3f}`",
        f"- label counts: `{payload['candidate_generator_quality']['label_counts']}`",
        f"- stage distribution: `{payload['candidate_generator_quality']['stage_distribution']}`",
        "",
        "## Matched Comparison",
        "",
        f"- matched tasks: `{selection.get('summary', {}).get('matched_evaluable_tasks')}`",
        f"- wrapper pass: `{selection.get('summary', {}).get('wrapper_final_pass_rate')}`",
        f"- BG conservative pass: `{selection.get('summary', {}).get('bg_conservative_matched_pass_rate')}`",
        f"- random expected pass: `{selection.get('summary', {}).get('random_expected_pass_rate')}`",
        f"- stage heuristic pass: `{selection.get('summary', {}).get('stage_heuristic_pass_rate')}`",
        "",
        "## Error Cases",
        "",
        f"- wrapper correct, BG wrong: `{error_cases['wrapper_correct_bg_wrong']}`",
        f"- BG correct, wrapper wrong: `{error_cases['bg_correct_wrapper_wrong']}`",
        f"- oracle existed, neither selected: `{error_cases['oracle_existed_neither_selected']}`",
        f"- no viable candidate: `{error_cases['no_viable_candidate']}`",
        "",
        "## Devil Tasks",
    ]
    for task_id, row in devil_analysis.items():
        lines.append(f"- `{task_id}`: oracle={row['oracle_reachable']} wrapper={row['wrapper_label']} BG={row['bg_label']}")
    lines.extend(["", "## Warnings", f"- `{payload['warnings']}`"])
    write_text(OUT_MD, "\n".join(lines) + "\n")
    print(f"WRAPPER_MATCHED_EXPERIMENT_VERDICT = {experiment_verdict}")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    if experiment_verdict == "BLOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
