"""Continue every generated prefix checkpoint and evaluate final answers."""
from __future__ import annotations

import time
import traceback
from collections import Counter, defaultdict

from bg_trajectory_prediction_lib import (
    REPORT_ROOT,
    SEED,
    OuroTextGenerator,
    continuation_budget,
    continuation_prompt,
    evaluate_output,
    iter_prefix_rows,
    load_json,
    load_partials,
    load_task_suite,
    rel,
    task_by_id,
    write_json,
    write_md,
)


OUT_JSON = REPORT_ROOT / "continued_prefixes.json"
OUT_PARTIAL = REPORT_ROOT / "continued_prefixes.partial.json"
OUT_MD = REPORT_ROOT / "continued_prefixes.md"


def main() -> int:
    started = time.time()
    tasks = load_task_suite()
    by_task = task_by_id(tasks)
    partials = load_partials()
    old = load_json(OUT_PARTIAL, {})
    rows = list(old.get("continued_prefixes") or [])
    done = {
        (str(row.get("task_id")), int(row.get("branch_id", -1)), int(row.get("prefix_length", -1)))
        for row in rows
    }
    prefixes = list(iter_prefix_rows(partials))
    generator: OuroTextGenerator | None = None
    try:
        generator = OuroTextGenerator(device="cuda")
        for prefix in prefixes:
            key = (str(prefix["task_id"]), int(prefix["branch_id"]), int(prefix["prefix_length"]))
            if key in done:
                continue
            task = by_task.get(str(prefix["task_id"]))
            if task is None:
                continue
            prompt = continuation_prompt(task, str(prefix["prefix_text"]))
            seed = SEED + int(task.get("suite_index", 0)) * 2003 + int(prefix["branch_id"]) * 17 + int(prefix["prefix_length"])
            try:
                gen = generator.generate(
                    prompt,
                    max_new_tokens=continuation_budget(task),
                    temperature=0.7,
                    top_p=0.95,
                    seed=seed,
                )
                final_text = (str(prefix["prefix_text"]).strip() + "\n" + gen["text"]).strip()
                evaluation = evaluate_output(task, final_text)
                row = {
                    **prefix,
                    "continuation_text": gen["text"],
                    "final_text": final_text,
                    "parsed_answer": evaluation.get("parsed_answer"),
                    "is_correct": bool(evaluation.get("success")),
                    "parse_failed": not bool(evaluation.get("parsed")),
                    "evaluable": bool(evaluation.get("evaluable", True)),
                    "hit_max_tokens": gen["hit_max_tokens"],
                    "continuation_error": gen["generation_error"],
                    "token_count": gen["token_count"],
                    "continuation_seconds": gen["seconds"],
                    "generation_params": {
                        "max_new_tokens": continuation_budget(task),
                        "temperature": 0.7,
                        "top_p": 0.95,
                        "seed": seed,
                    },
                    "evaluation": evaluation,
                }
            except Exception as exc:
                row = {
                    **prefix,
                    "continuation_text": "",
                    "final_text": str(prefix["prefix_text"]),
                    "parsed_answer": None,
                    "is_correct": False,
                    "parse_failed": True,
                    "evaluable": False,
                    "hit_max_tokens": False,
                    "continuation_error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:],
                    "token_count": 0,
                    "evaluation": {"evaluable": False, "success": False, "parsed": False},
                }
            rows.append(row)
            done.add(key)
            write_json(OUT_PARTIAL, {"complete": False, "continued_prefixes": rows})
    finally:
        if generator is not None:
            generator.cleanup()

    usable_prefixes = len(prefixes)
    evaluable_rows = [row for row in rows if row.get("evaluable") and not row.get("continuation_error")]
    eval_rate = len(evaluable_rows) / max(usable_prefixes, 1)
    if eval_rate >= 0.75 and rows:
        verdict = "READY"
    elif eval_rate >= 0.50:
        verdict = "PARTIAL"
    else:
        verdict = "BLOCKED"

    task_oracle = defaultdict(bool)
    task_domain = {}
    prefix_task_oracle = defaultdict(lambda: defaultdict(bool))
    prefix_task_seen = defaultdict(set)
    for row in rows:
        task_id = str(row["task_id"])
        prefix_length = int(row["prefix_length"])
        task_domain[task_id] = row.get("domain")
        task_oracle[task_id] = task_oracle[task_id] or bool(row.get("is_correct"))
        prefix_task_oracle[prefix_length][task_id] = prefix_task_oracle[prefix_length][task_id] or bool(row.get("is_correct"))
        prefix_task_seen[prefix_length].add(task_id)

    oracle_by_domain_prefix = {}
    for prefix_length, task_ids in prefix_task_seen.items():
        for domain in sorted({task_domain.get(tid) for tid in task_ids}):
            ids = [tid for tid in task_ids if task_domain.get(tid) == domain]
            successes = sum(1 for tid in ids if prefix_task_oracle[prefix_length][tid])
            oracle_by_domain_prefix[f"{domain}::{prefix_length}"] = {
                "domain": domain,
                "prefix_length": prefix_length,
                "task_count": len(ids),
                "oracle_successes": successes,
                "oracle_success_rate": successes / max(len(ids), 1),
            }
    generator_limited = all(row["oracle_success_rate"] < 0.25 for row in oracle_by_domain_prefix.values()) if oracle_by_domain_prefix else True

    payload = {
        "BG_TRAJECTORY_CONTINUATION_VERDICT": verdict,
        "verdict": verdict,
        "GENERATOR_REACHABILITY_LIMITED": generator_limited,
        "usable_prefix_count": usable_prefixes,
        "continued_count": len(rows),
        "evaluable_count": len(evaluable_rows),
        "evaluable_rate": eval_rate,
        "oracle_success_by_domain_prefix": oracle_by_domain_prefix,
        "counts_by_domain": dict(Counter(row.get("domain") for row in rows)),
        "continued_prefixes": rows,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_PARTIAL, payload)
    write_json(OUT_JSON, payload)
    table = [
        {
            "domain": row["domain"],
            "prefix": row["prefix_length"],
            "tasks": row["task_count"],
            "oracle_success_rate": f"{row['oracle_success_rate']:.3f}",
        }
        for row in oracle_by_domain_prefix.values()
    ]
    lines = [
        "# BG Trajectory Prefix Continuations (2026-05-18)",
        "",
        f"BG_TRAJECTORY_CONTINUATION_VERDICT = {verdict}",
        f"GENERATOR_REACHABILITY_LIMITED = {str(generator_limited).lower()}",
        "",
        f"- usable_prefix_count: `{usable_prefixes}`",
        f"- continued_count: `{len(rows)}`",
        f"- evaluable_count: `{len(evaluable_rows)}`",
        f"- evaluable_rate: `{eval_rate:.3f}`",
        "",
        "## Oracle Success By Domain And Prefix",
        "",
        "| Domain | Prefix | Tasks | Oracle success rate |",
        "|---|---:|---:|---:|",
    ]
    for row in table:
        lines.append(f"| `{row['domain']}` | {row['prefix']} | {row['tasks']} | {row['oracle_success_rate']} |")
    write_md(OUT_MD, lines)
    print(f"BG_TRAJECTORY_CONTINUATION_VERDICT = {verdict}")
    print(f"GENERATOR_REACHABILITY_LIMITED = {str(generator_limited).lower()}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict in {"READY", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
