#!/usr/bin/env python3
"""Quick parser audit for sequence-level BG adapter rewards."""
from __future__ import annotations

import os
import time
import traceback
from typing import Any

from bg_sequence_adapter_common import (
    QUICK_OUT_ROOT,
    avg,
    load_model_tokenizer,
    load_stage1_mcq_tasks,
    markdown_table,
    rel,
    run_generation_condition,
    select_balanced_tasks,
    set_seed,
    write_json,
    write_md,
)


OUT_JSON = QUICK_OUT_ROOT / "parser_audit.json"
OUT_MD = QUICK_OUT_ROOT / "parser_audit.md"
TASK_CAP = min(12, max(4, int(os.environ.get("BG_SEQUENCE_PARSER_AUDIT_TASKS", "8"))))
RUN_SAMPLED = os.environ.get("BG_SEQUENCE_PARSER_AUDIT_SAMPLED", "0") == "1"
MAX_NEW_TOKENS = min(96, int(os.environ.get("BG_SEQUENCE_PARSER_AUDIT_TOKENS", "96")))


def main() -> int:
    started = time.time()
    set_seed()
    QUICK_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    try:
        tasks = select_balanced_tasks(load_stage1_mcq_tasks(), TASK_CAP)
        if not tasks:
            raise RuntimeError("no clean Stage 1 reasoning/science MCQ tasks found")
        model, tokenizer, device = load_model_tokenizer()
        for idx, task in enumerate(tasks):
            rows.append(
                run_generation_condition(
                    model,
                    tokenizer,
                    device,
                    task,
                    method="no_intervention_baseline",
                    seed=20260518 + idx,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )
            )
            if RUN_SAMPLED:
                rows.append(
                    run_generation_condition(
                        model,
                        tokenizer,
                        device,
                        task,
                        method="no_intervention_baseline",
                        seed=20260618 + idx,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True,
                        sample_index=1,
                    )
                )
    except Exception as exc:
        payload = {
            "BG_SEQUENCE_PARSER_AUDIT_VERDICT": "BLOCKED",
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "traceback": traceback.format_exc()[-4000:],
            "rows": rows,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# BG Sequence Adapter Parser Audit", "", "BG_SEQUENCE_PARSER_AUDIT_VERDICT = BLOCKED", "", f"- error: `{type(exc).__name__}: {str(exc)[:500]}`"])
        print("BG_SEQUENCE_PARSER_AUDIT_VERDICT = BLOCKED")
        return 1

    parse_rate = avg(row["parse_success"] for row in rows) or 0.0
    if parse_rate >= 0.75:
        verdict = "READY"
    elif parse_rate >= 0.50:
        verdict = "PARTIAL"
    else:
        verdict = "BLOCKED"
    payload = {
        "BG_SEQUENCE_PARSER_AUDIT_VERDICT": verdict,
        "task_count": len({row["task_id"] for row in rows}),
        "generation_count": len(rows),
        "sampled_decode_ran": RUN_SAMPLED,
        "max_new_tokens": MAX_NEW_TOKENS,
        "parse_rate": parse_rate,
        "success_rate": avg(row["correct"] for row in rows),
        "rows": rows,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    lines = [
        "# BG Sequence Adapter Parser Audit",
        "",
        f"BG_SEQUENCE_PARSER_AUDIT_VERDICT = {verdict}",
        "",
        f"- tasks: `{payload['task_count']}`",
        f"- generations: `{len(rows)}`",
        f"- sampled decode ran: `{RUN_SAMPLED}`",
        f"- parse rate: `{parse_rate:.3f}`",
        f"- success rate: `{(payload['success_rate'] or 0.0):.3f}`",
        "",
        "## Rows",
        "",
    ]
    lines.extend(
        markdown_table(
            rows,
            ["task_id", "domain", "gold_answer", "parsed_answer", "correct", "parse_success", "parse_failure_reason", "output_length", "hit_max_tokens"],
            max_rows=30,
        )
    )
    write_md(OUT_MD, lines)
    print(f"BG_SEQUENCE_PARSER_AUDIT_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
