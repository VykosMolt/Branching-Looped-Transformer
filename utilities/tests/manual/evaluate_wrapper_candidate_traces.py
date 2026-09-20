"""Post-hoc unit-test evaluation for exported wrapper candidate traces."""

from __future__ import annotations

from collections import Counter
from typing import Any

from evaluate_code_branch_candidates_v2 import eval_candidate, label_from_eval
from wrapper_bg_matched_lib import (
    OUT_DIR,
    candidate_code,
    code_parse_status,
    label_rank,
    load_json,
    load_task_suite,
    repo_path,
    task_slug,
    trace_candidates,
    write_json,
    write_text,
)


TRACES_JSON = OUT_DIR / "candidate_traces.json"
OUT_JSON = OUT_DIR / "candidate_eval.json"
OUT_MD = OUT_DIR / "candidate_eval.md"


def main() -> None:
    tasks = {task["task_id"]: task for task in load_task_suite()}
    trace_payload = load_json(TRACES_JSON)
    rows: list[dict[str, Any]] = []
    by_task: dict[str, Any] = {}

    for trace_row in trace_payload.get("candidate_traces", []):
        task_id = trace_row["task_id"]
        task = tasks.get(task_id)
        if task is None:
            continue
        trace_data = load_json(trace_row["trace_path"])
        tests = list(task.get("tests") or [])
        function_name = str(task.get("function_name") or "")
        evaluations = []
        for artifact in trace_candidates(trace_data):
            code, code_source = candidate_code(artifact)
            parse = code_parse_status(code, function_name)
            if not code:
                eval_row = {
                    "syntax_ok": False,
                    "import_ok": False,
                    "runtime_ok": False,
                    "safety_ok": True,
                    "tests_total": len(tests),
                    "tests_passed": 0,
                    "tests_failed": len(tests),
                    "pass_rate": 0.0,
                    "error_type": "NotCode",
                    "error_message_short": "no usable code-like candidate text",
                    "execution_seconds": 0.0,
                    "stdout_short": "",
                    "stderr_short": "",
                }
                label = "not_code"
            else:
                eval_row = eval_candidate(code, tests, function_name, float(task.get("timeout_seconds") or 5.0))
                label = label_from_eval(eval_row, code, function_name)
            row = {
                "task_id": task_id,
                "candidate_uid": artifact.get("candidate_uid"),
                "trace_id": trace_data.get("trace_id"),
                "stage": artifact.get("stage", "unknown"),
                "code_source": code_source,
                "code_sha": __import__("hashlib").sha256(code.encode("utf-8", errors="replace")).hexdigest()[:16] if code else "",
                "candidate_code": code,
                "parse_status": parse,
                "label": label,
                "is_correct": label == "correct",
                "is_near_miss_or_correct": label in {"correct", "near_miss"},
                "eval": eval_row,
            }
            rows.append(row)
            evaluations.append(row)

        label_counts = Counter(row["label"] for row in evaluations)
        correct = [row for row in evaluations if row["label"] == "correct"]
        near = [row for row in evaluations if row["label"] == "near_miss"]
        selected_uid = trace_data.get("selected_candidate_uid")
        wrapper_final_label = "missing"
        if selected_uid:
            for row in evaluations:
                if row["candidate_uid"] == selected_uid:
                    wrapper_final_label = row["label"]
                    break
        best = sorted(evaluations, key=lambda row: (label_rank(row["label"]), row.get("stage", ""), row.get("candidate_uid", "")))
        by_task[task_id] = {
            "task_id": task_id,
            "trace_id": trace_data.get("trace_id"),
            "oracle_candidate_exists": bool(correct),
            "n_correct": len(correct),
            "n_near_miss": len(near),
            "n_wrong_code": label_counts.get("wrong_code", 0),
            "n_runtime_error": label_counts.get("runtime_error", 0),
            "n_malformed": label_counts.get("malformed", 0) + label_counts.get("not_code", 0),
            "n_safety_rejected": label_counts.get("safety_rejected", 0),
            "label_counts": dict(label_counts),
            "wrapper_selected_candidate_uid": selected_uid,
            "wrapper_final_label": wrapper_final_label,
            "best_candidate_uid": best[0]["candidate_uid"] if best else None,
            "best_candidate_label": best[0]["label"] if best else "missing",
            "code_like_candidate_count": sum(1 for row in evaluations if row["candidate_code"]),
        }

    non_devil_task_ids = [task_id for task_id in by_task if not tasks.get(task_id, {}).get("is_devil")]
    oracle_rate = (
        sum(1 for task_id in non_devil_task_ids if by_task[task_id]["oracle_candidate_exists"]) / max(len(non_devil_task_ids), 1)
    )
    if oracle_rate >= 0.25:
        reachability = "REACHABLE"
    elif oracle_rate >= 0.10:
        reachability = "PARTIAL"
    else:
        reachability = "LOW_REACHABILITY"
    verdict = "READY" if rows and by_task else "BLOCKED"
    if verdict == "READY" and len(by_task) < 6:
        verdict = "PARTIAL"
    payload = {
        "WRAPPER_CANDIDATE_EVAL_VERDICT": verdict,
        "WRAPPER_GENERATOR_REACHABILITY_VERDICT": reachability,
        "candidate_evaluations": rows,
        "by_task": by_task,
        "summary": {
            "tasks_evaluated": len(by_task),
            "candidates_evaluated": len(rows),
            "label_counts": dict(Counter(row["label"] for row in rows)),
            "non_devil_oracle_success_rate": oracle_rate,
            "oracle_reachable_tasks": sum(1 for row in by_task.values() if row["oracle_candidate_exists"]),
        },
    }
    write_json(OUT_JSON, payload)
    lines = [
        "# Wrapper Candidate Post-Hoc Evaluation",
        "",
        f"WRAPPER_CANDIDATE_EVAL_VERDICT = {verdict}",
        f"WRAPPER_GENERATOR_REACHABILITY_VERDICT = {reachability}",
        "",
        f"- tasks evaluated: `{len(by_task)}`",
        f"- candidates evaluated: `{len(rows)}`",
        f"- label counts: `{payload['summary']['label_counts']}`",
        f"- non-devil oracle success rate: `{oracle_rate:.3f}`",
        "",
        "| task_id | oracle | wrapper_label | labels | best |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for task_id, row in by_task.items():
        lines.append(
            f"| `{task_id}` | {row['oracle_candidate_exists']} | `{row['wrapper_final_label']}` | "
            f"`{row['label_counts']}` | `{row['best_candidate_label']}` |"
        )
    write_text(OUT_MD, "\n".join(lines) + "\n")
    print(f"WRAPPER_CANDIDATE_EVAL_VERDICT = {verdict}")
    print(f"WRAPPER_GENERATOR_REACHABILITY_VERDICT = {reachability}")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    if verdict == "BLOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
