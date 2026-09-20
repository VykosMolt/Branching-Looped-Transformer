"""Build a bounded reasoning/science MCQ task subset for hidden-origin branches."""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any

from bg_hidden_branch_suite_common import (
    PROBE_ROOT,
    REPORT_ROOT,
    ensure_report_root,
    hidden_branch_prompt,
    load_json,
    md_table,
    rel,
    write_json,
    write_md,
)


OUT_JSON = REPORT_ROOT / "task_subset.json"
OUT_MD = REPORT_ROOT / "task_subset.md"
TARGET_TASKS = 16
MIN_READY = 12
MIN_PARTIAL = 8


def _normalize_task(row: dict[str, Any], domain: str, index: int) -> dict[str, Any] | None:
    question = str(row.get("question") or "").strip()
    options = row.get("options") or {}
    answer = str(row.get("answer_key") or row.get("answer") or row.get("gold_answer") or "").strip().upper()
    task_id = str(row.get("task_id") or f"{domain}/{index}")
    if not question or not isinstance(options, dict) or len(options) != 4 or answer not in {str(k).upper() for k in options}:
        return None
    clean_options = {str(k).upper(): str(v).strip() for k, v in sorted(options.items())}
    return {
        "task_id": task_id,
        "domain": domain,
        "source_dataset": row.get("source_dataset") or row.get("dataset") or domain,
        "source_subject": row.get("source_subject"),
        "subdomain_bucket": row.get("subdomain_bucket"),
        "question": question,
        "options": clean_options,
        "correct_option": answer,
        "expected_answer_text": clean_options.get(answer, ""),
        "parser_type": "mcq_letter",
        "prompt": hidden_branch_prompt(question, clean_options),
        "target_branch_points": ["L24_primary", "L36_secondary", "L47_optional"],
        "split": "analysis_only_no_training",
    }


def _load_trajectory_seed_tasks() -> list[dict[str, Any]]:
    payload = load_json(PROBE_ROOT / "bg_trajectory_prediction_2026-05-18/task_suite.json", {})
    rows = []
    for idx, item in enumerate(payload.get("tasks") or []):
        domain = str(item.get("domain") or "")
        if domain not in {"reasoning", "science"}:
            continue
        norm = _normalize_task(item, domain, idx)
        if norm:
            norm["preferred_source"] = "stage1_trajectory_prediction"
            rows.append(norm)
    return rows


def _load_fallback_tasks() -> list[dict[str, Any]]:
    rows = []
    reasoning = load_json(PROBE_ROOT / "reasoning_branch_pilot_2026-05-17.json", {})
    for idx, item in enumerate(reasoning.get("tasks") or []):
        norm = _normalize_task(item, "reasoning", idx)
        if norm:
            norm["preferred_source"] = "reasoning_branch_pilot"
            rows.append(norm)
    science = load_json(PROBE_ROOT / "science_natural_distractor_set_2026-05-17.json", {})
    for idx, item in enumerate(science.get("tasks") or []):
        norm = _normalize_task(item, "science", idx)
        if norm:
            norm["preferred_source"] = "science_natural_distractor"
            rows.append(norm)
    return rows


def _balanced_select(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    by_domain = {"reasoning": [], "science": []}
    for row in rows:
        if row["task_id"] in seen:
            continue
        seen.add(row["task_id"])
        if row["domain"] in by_domain:
            by_domain[row["domain"]].append(row)
    selected: list[dict[str, Any]] = []
    target_each = TARGET_TASKS // 2
    selected.extend(by_domain["reasoning"][:target_each])
    selected.extend(by_domain["science"][:target_each])
    if len(selected) < TARGET_TASKS:
        for domain in ("reasoning", "science"):
            for row in by_domain[domain][target_each:]:
                if len(selected) >= TARGET_TASKS:
                    break
                selected.append(row)
    for idx, row in enumerate(selected):
        row["subset_index"] = idx
    return selected[:TARGET_TASKS]


def main() -> int:
    ensure_report_root()
    started = time.time()
    candidates = _load_trajectory_seed_tasks() + _load_fallback_tasks()
    tasks = _balanced_select(candidates)
    if len(tasks) >= MIN_READY:
        verdict = "READY"
    elif len(tasks) >= MIN_PARTIAL:
        verdict = "PARTIAL"
    else:
        verdict = "BLOCKED"
    counts = Counter(row["domain"] for row in tasks)
    payload = {
        "BG_HIDDEN_BRANCH_TASK_SUBSET_VERDICT": verdict,
        "verdict": verdict,
        "task_count": len(tasks),
        "target_task_count": TARGET_TASKS,
        "counts_by_domain": dict(counts),
        "tasks": tasks,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    lines = [
        "# BG Hidden Branch Task Subset",
        "",
        f"BG_HIDDEN_BRANCH_TASK_SUBSET_VERDICT = {verdict}",
        "",
        f"- task_count: `{len(tasks)}`",
        f"- counts_by_domain: `{dict(counts)}`",
        f"- split: `analysis_only_no_training`",
        "",
        "## Tasks",
        "",
    ]
    lines.extend(md_table([
        {
            "idx": row["subset_index"],
            "task_id": row["task_id"],
            "domain": row["domain"],
            "source": row["source_dataset"],
            "answer": row["correct_option"],
        }
        for row in tasks
    ], ["idx", "task_id", "domain", "source", "answer"]))
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_BRANCH_TASK_SUBSET_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict in {"READY", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
