"""Run BG selection over wrapper-exported candidate traces."""

from __future__ import annotations

import math
import random
from collections import Counter
from statistics import mean
from typing import Any

import torch

from wrapper_bg_matched_lib import (
    OUT_DIR,
    label_is_near_or_success,
    label_is_success,
    label_rank,
    load_json,
    load_task_suite,
    verdict_from_delta,
    write_json,
    write_text,
)


EVAL_JSON = OUT_DIR / "candidate_eval.json"
FEATURE_PT = OUT_DIR / "wrapper_candidate_features.pt"
OUT_JSON = OUT_DIR / "bg_selection.json"
OUT_MD = OUT_DIR / "bg_selection.md"

STAGE_ORDER = [
    "final_grounded_code",
    "repaired_final",
    "tool_verified_code",
    "direct_final",
    "first_repair_code",
    "first_tool_code",
    "first_failed_tool_code",
    "direct_short_budget",
    "sampled_direct",
]


def metric_label(row: dict[str, Any] | None) -> str:
    return "missing" if row is None else str(row.get("label", "missing"))


def row_for_uid(rows: list[dict[str, Any]], uid: str | None) -> dict[str, Any] | None:
    if not uid:
        return None
    for row in rows:
        if row.get("candidate_uid") == uid:
            return row
    return None


def select_stage_heuristic(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return sorted(
        rows,
        key=lambda row: (
            STAGE_ORDER.index(row.get("stage")) if row.get("stage") in STAGE_ORDER else len(STAGE_ORDER),
            row.get("candidate_uid", ""),
        ),
    )[0]


def select_oracle(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return sorted(rows, key=lambda row: (label_rank(row.get("label", "missing")), row.get("candidate_uid", "")))[0]


def safe_rate(values: list[float]) -> float:
    return float("nan") if not values else float(mean(values))


def tensor_details(details: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in details.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().cpu().tolist()
        elif key == "head_results":
            out[key] = {
                name: {
                    sub_key: (sub_value.detach().cpu().tolist() if isinstance(sub_value, torch.Tensor) else sub_value)
                    for sub_key, sub_value in row.items()
                }
                for name, row in value.items()
            }
        else:
            out[key] = value
    return out


def selected_from_bg(controller: Any, feature_rows: list[dict[str, Any]], features: dict[str, torch.Tensor], mode: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not feature_rows:
        return None, {"error": "no_feature_rows"}
    uids = [row["candidate_uid"] for row in feature_rows]
    stacked = torch.stack([features[uid].cpu() for uid in uids], dim=0)
    if len(uids) == 1:
        return feature_rows[0], {"selected_index": 0, "ranking": [0], "mode": mode, "note": "single_feature_candidate"}
    details = controller.select_best(stacked, domain_hint="code", mode=mode, return_details=True)
    if mode == "diagnostic_all":
        serial = tensor_details(details)
        return None, serial
    selected_index = int(details.get("selected_index", -1))
    selected = feature_rows[selected_index] if 0 <= selected_index < len(feature_rows) else None
    return selected, tensor_details(details)


def main() -> None:
    tasks = {task["task_id"]: task for task in load_task_suite()}
    eval_payload = load_json(EVAL_JSON)
    feature_payload = torch.load(FEATURE_PT, map_location="cpu")
    feature_nested: dict[str, dict[str, torch.Tensor]] = feature_payload.get("features", {})
    from src.evaluator.bg_controller import BGController

    controller = BGController.from_artifacts(device="cpu")
    rows_by_task: dict[str, list[dict[str, Any]]] = {}
    for row in eval_payload.get("candidate_evaluations", []):
        rows_by_task.setdefault(row["task_id"], []).append(row)

    by_task: dict[str, Any] = {}
    random_seed = 20260518
    rng = random.Random(random_seed)

    for task_id, rows in rows_by_task.items():
        features = feature_nested.get(task_id, {})
        feature_rows = [row for row in rows if row.get("candidate_uid") in features]
        wrapper_uid = eval_payload.get("by_task", {}).get(task_id, {}).get("wrapper_selected_candidate_uid")
        wrapper_row = row_for_uid(rows, wrapper_uid)
        stage_row = select_stage_heuristic(feature_rows or rows)
        oracle_row = select_oracle(rows)
        random_rows = feature_rows or [row for row in rows if row.get("candidate_code")]
        random_expected_pass = sum(1 for row in random_rows if label_is_success(row.get("label", ""))) / max(len(random_rows), 1)
        random_expected_near = sum(1 for row in random_rows if label_is_near_or_success(row.get("label", ""))) / max(len(random_rows), 1)
        random_sample = rng.choice(random_rows) if random_rows else None

        bg_row, bg_details = selected_from_bg(controller, feature_rows, features, "conservative")
        code_row, code_details = selected_from_bg(controller, feature_rows, features, "code_backup")
        vote_row, vote_details = selected_from_bg(controller, feature_rows, features, "experimental_vote")
        _, diag_details = selected_from_bg(controller, feature_rows, features, "diagnostic_all")
        diagnostic_head_choices = {}
        for head, selected_idx in diag_details.get("selected_by_head", {}).items():
            if 0 <= int(selected_idx) < len(feature_rows):
                diagnostic_head_choices[head] = feature_rows[int(selected_idx)]["candidate_uid"]

        by_task[task_id] = {
            "task_id": task_id,
            "source": tasks.get(task_id, {}).get("source"),
            "difficulty": tasks.get(task_id, {}).get("difficulty"),
            "is_devil": bool(tasks.get(task_id, {}).get("is_devil")),
            "candidate_count": len(rows),
            "feature_candidate_count": len(feature_rows),
            "wrapper_final_candidate_uid": wrapper_uid,
            "wrapper_final_not_in_trace": wrapper_row is None,
            "wrapper_final_label": metric_label(wrapper_row),
            "random_expected_pass": random_expected_pass,
            "random_expected_near_or_correct": random_expected_near,
            "random_sample_uid": random_sample.get("candidate_uid") if random_sample else None,
            "random_sample_label": metric_label(random_sample),
            "stage_heuristic_uid": stage_row.get("candidate_uid") if stage_row else None,
            "stage_heuristic_label": metric_label(stage_row),
            "bg_conservative_uid": bg_row.get("candidate_uid") if bg_row else None,
            "bg_conservative_label": metric_label(bg_row),
            "bg_code_backup_uid": code_row.get("candidate_uid") if code_row else None,
            "bg_code_backup_label": metric_label(code_row),
            "bg_experimental_vote_uid": vote_row.get("candidate_uid") if vote_row else None,
            "bg_experimental_vote_label": metric_label(vote_row),
            "oracle_candidate_uid": oracle_row.get("candidate_uid") if oracle_row else None,
            "oracle_label": metric_label(oracle_row),
            "oracle_reachable": bool(oracle_row and oracle_row.get("label") == "correct"),
            "diagnostic_head_choices": diagnostic_head_choices,
            "bg_details": bg_details,
            "code_backup_details": code_details,
            "experimental_vote_details": vote_details,
            "diagnostic_all_details": diag_details,
            "selected_stage_distribution": {
                "wrapper": wrapper_row.get("stage") if wrapper_row else "missing",
                "bg_conservative": bg_row.get("stage") if bg_row else "missing",
                "code_backup": code_row.get("stage") if code_row else "missing",
                "stage_heuristic": stage_row.get("stage") if stage_row else "missing",
            },
        }

    matched = [
        row for row in by_task.values()
        if not row["is_devil"] and not row["wrapper_final_not_in_trace"] and row["feature_candidate_count"] >= 1
    ]
    bg_tasks = [row for row in by_task.values() if row["feature_candidate_count"] >= 1]
    random_tasks = [row for row in bg_tasks if not row["is_devil"]]
    stage_tasks = [row for row in bg_tasks if not row["is_devil"]]
    wrapper_pass = safe_rate([1.0 if label_is_success(row["wrapper_final_label"]) else 0.0 for row in matched])
    bg_pass = safe_rate([1.0 if label_is_success(row["bg_conservative_label"]) else 0.0 for row in matched])
    bg_all_pass = safe_rate([1.0 if label_is_success(row["bg_conservative_label"]) else 0.0 for row in random_tasks])
    random_pass = safe_rate([row["random_expected_pass"] for row in random_tasks])
    stage_pass = safe_rate([1.0 if label_is_success(row["stage_heuristic_label"]) else 0.0 for row in stage_tasks])
    oracle_pass = safe_rate([1.0 if row["oracle_reachable"] else 0.0 for row in random_tasks])
    bg_vs_wrapper_delta = bg_pass - wrapper_pass if not math.isnan(bg_pass) and not math.isnan(wrapper_pass) else float("nan")
    bg_vs_random_delta = bg_all_pass - random_pass if not math.isnan(bg_all_pass) and not math.isnan(random_pass) else float("nan")
    bg_vs_stage_delta = bg_all_pass - stage_pass if not math.isnan(bg_all_pass) and not math.isnan(stage_pass) else float("nan")

    matched_verdict = verdict_from_delta(bg_vs_wrapper_delta, len(matched), helps_threshold=0.05, min_n=8)
    random_verdict = verdict_from_delta(bg_vs_random_delta, len(random_tasks), helps_threshold=0.10, min_n=8)
    stage_verdict = verdict_from_delta(bg_vs_stage_delta, len(stage_tasks), helps_threshold=0.05, min_n=8)
    oracle_gap = oracle_pass - wrapper_pass if not math.isnan(oracle_pass) and not math.isnan(wrapper_pass) else float("nan")
    if len(matched) < 8 or math.isnan(oracle_gap):
        oracle_gap_verdict = "INSUFFICIENT"
    elif oracle_gap <= 0.05:
        oracle_gap_verdict = "SMALL"
    elif oracle_gap <= 0.20:
        oracle_gap_verdict = "MODERATE"
    else:
        oracle_gap_verdict = "LARGE"
    if len(matched) < 8 or oracle_pass < 0.10:
        matched_verdict = "INSUFFICIENT"

    summary = {
        "matched_evaluable_tasks": len(matched),
        "bg_evaluable_non_devil_tasks": len(random_tasks),
        "wrapper_final_pass_rate": wrapper_pass,
        "bg_conservative_matched_pass_rate": bg_pass,
        "bg_conservative_all_non_devil_pass_rate": bg_all_pass,
        "random_expected_pass_rate": random_pass,
        "stage_heuristic_pass_rate": stage_pass,
        "oracle_reachability_rate": oracle_pass,
        "bg_lift_vs_wrapper": bg_vs_wrapper_delta,
        "bg_lift_vs_random": bg_vs_random_delta,
        "bg_lift_vs_stage_heuristic": bg_vs_stage_delta,
        "selected_stage_counts": {
            key: dict(Counter(row["selected_stage_distribution"][key] for row in by_task.values()))
            for key in ("wrapper", "bg_conservative", "code_backup", "stage_heuristic")
        },
        "label_counts_by_policy": {
            "wrapper": dict(Counter(row["wrapper_final_label"] for row in by_task.values())),
            "bg_conservative": dict(Counter(row["bg_conservative_label"] for row in by_task.values())),
            "code_backup": dict(Counter(row["bg_code_backup_label"] for row in by_task.values())),
            "stage_heuristic": dict(Counter(row["stage_heuristic_label"] for row in by_task.values())),
        },
    }
    payload = {
        "WRAPPER_MATCHED_BG_VERDICT": matched_verdict,
        "BG_VS_RANDOM_VERDICT": random_verdict,
        "BG_VS_STAGE_HEURISTIC_VERDICT": stage_verdict,
        "WRAPPER_ORACLE_GAP_VERDICT": oracle_gap_verdict,
        "random_seed": random_seed,
        "summary": summary,
        "by_task": by_task,
    }
    write_json(OUT_JSON, payload)
    lines = [
        "# Wrapper-Matched BG Selection",
        "",
        f"WRAPPER_MATCHED_BG_VERDICT = {matched_verdict}",
        f"BG_VS_RANDOM_VERDICT = {random_verdict}",
        f"BG_VS_STAGE_HEURISTIC_VERDICT = {stage_verdict}",
        f"WRAPPER_ORACLE_GAP_VERDICT = {oracle_gap_verdict}",
        "",
        f"- matched evaluable tasks: `{len(matched)}`",
        f"- wrapper final pass rate: `{wrapper_pass:.3f}`",
        f"- BG conservative matched pass rate: `{bg_pass:.3f}`",
        f"- BG conservative non-devil pass rate: `{bg_all_pass:.3f}`",
        f"- random expected pass rate: `{random_pass:.3f}`",
        f"- stage heuristic pass rate: `{stage_pass:.3f}`",
        f"- oracle reachability rate: `{oracle_pass:.3f}`",
        "",
        "| task_id | wrapper | BG | random_exp | stage | oracle | feature_n |",
        "| --- | --- | --- | ---: | --- | --- | ---: |",
    ]
    for task_id, row in by_task.items():
        lines.append(
            f"| `{task_id}` | `{row['wrapper_final_label']}` | `{row['bg_conservative_label']}` | "
            f"{row['random_expected_pass']:.3f} | `{row['stage_heuristic_label']}` | `{row['oracle_label']}` | "
            f"{row['feature_candidate_count']} |"
        )
    write_text(OUT_MD, "\n".join(lines) + "\n")
    print(f"WRAPPER_MATCHED_BG_VERDICT = {matched_verdict}")
    print(f"BG_VS_RANDOM_VERDICT = {random_verdict}")
    print(f"BG_VS_STAGE_HEURISTIC_VERDICT = {stage_verdict}")
    print(f"WRAPPER_ORACLE_GAP_VERDICT = {oracle_gap_verdict}")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
