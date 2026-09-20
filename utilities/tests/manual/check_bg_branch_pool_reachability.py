"""Early generator reachability gate for BG branch pools."""
from __future__ import annotations

import time
import traceback
from collections import Counter, defaultdict

from bg_steering_suite_lib import (
    REPORT_ROOT,
    SEED,
    OuroTextGenerator,
    branch_rows_for_task,
    continuation_budget,
    continuation_prompt,
    evaluate_output,
    load_branch_pools,
    load_task_suite,
    rel,
    task_by_id,
    write_json,
    write_md,
)


OUT_JSON = REPORT_ROOT / "reachability_gate.json"
OUT_MD = REPORT_ROOT / "reachability_gate.md"


def _diagnostic_subset(tasks: list[dict]) -> list[dict]:
    out = []
    for domain in ("code", "reasoning", "science", "gsm8k"):
        rows = [t for t in tasks if t["domain"] == domain and not t.get("is_devil")]
        out.extend(rows[:3])
    out.extend([t for t in tasks if t.get("is_devil")][:2])
    return out


def main() -> int:
    started = time.time()
    tasks = load_task_suite()
    branches_payload = load_branch_pools()
    subset = _diagnostic_subset(tasks)
    rows = []
    generator: OuroTextGenerator | None = None
    try:
        generator = OuroTextGenerator(device="cuda")
        for task in subset:
            task_id = str(task["task_id"])
            for branch in branch_rows_for_task(branches_payload, task_id):
                prompt = continuation_prompt(task, branch.get("initial_partial_text", ""))
                try:
                    gen = generator.generate(
                        prompt,
                        max_new_tokens=continuation_budget(task),
                        temperature=0.7,
                        top_p=0.95,
                        seed=SEED + int(task.get("suite_index", 0)) * 101 + int(branch.get("branch_id", 0)),
                    )
                    final_text = (branch.get("initial_partial_text", "") + "\n" + gen["text"]).strip()
                    evaluation = evaluate_output(task, final_text)
                    row = {
                        "task_id": task_id,
                        "domain": task["domain"],
                        "is_devil": bool(task.get("is_devil")),
                        "branch_id": branch.get("branch_id"),
                        "final_text": final_text,
                        "continuation": gen,
                        "evaluation": evaluation,
                    }
                except Exception as exc:
                    row = {
                        "task_id": task_id,
                        "domain": task["domain"],
                        "is_devil": bool(task.get("is_devil")),
                        "branch_id": branch.get("branch_id"),
                        "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:],
                        "evaluation": {"evaluable": False, "success": False},
                    }
                rows.append(row)
    except Exception as exc:
        payload = {
            "BG_REACHABILITY_GATE_VERDICT": "BLOCKED",
            "verdict": "BLOCKED",
            "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            "rows": rows,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# BG Reachability Gate", "", "BG_REACHABILITY_GATE_VERDICT = BLOCKED", "", "```text", payload["error"], "```"])
        print("BG_REACHABILITY_GATE_VERDICT = BLOCKED")
        return 1
    finally:
        if generator is not None:
            generator.cleanup()

    task_success: dict[str, bool] = defaultdict(bool)
    task_domain: dict[str, str] = {}
    task_devil: dict[str, bool] = {}
    for row in rows:
        task_id = str(row["task_id"])
        task_domain[task_id] = row["domain"]
        task_devil[task_id] = bool(row.get("is_devil"))
        task_success[task_id] = task_success[task_id] or bool(row.get("evaluation", {}).get("success"))
    by_domain = {}
    for domain in sorted(set(task_domain.values())):
        ids = [tid for tid, d in task_domain.items() if d == domain and not task_devil[tid]]
        successes = sum(1 for tid in ids if task_success[tid])
        rate = successes / max(len(ids), 1)
        threshold = 0.25 if domain == "code" else 0.40
        by_domain[domain] = {
            "task_count": len(ids),
            "oracle_successes": successes,
            "oracle_success_rate": rate,
            "threshold": threshold,
            "meets_threshold": rate >= threshold and len(ids) > 0,
        }
    devil_ids = [tid for tid in task_domain if task_devil[tid]]
    devil_successes = sum(1 for tid in devil_ids if task_success[tid])
    ready_domains = [d for d, row in by_domain.items() if row["meets_threshold"]]
    if len(ready_domains) >= 2:
        verdict = "READY"
    elif len(ready_domains) == 1:
        verdict = "PARTIAL"
    elif rows:
        verdict = "GENERATOR_REACHABILITY_LIMITED"
    else:
        verdict = "BLOCKED"
    payload = {
        "BG_REACHABILITY_GATE_VERDICT": verdict,
        "verdict": verdict,
        "GENERATOR_REACHABILITY_LIMITED": verdict == "GENERATOR_REACHABILITY_LIMITED",
        "by_domain": by_domain,
        "ready_domains": ready_domains,
        "devil": {
            "task_count": len(devil_ids),
            "oracle_successes": devil_successes,
            "oracle_success_rate": devil_successes / max(len(devil_ids), 1),
        },
        "rows": rows,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    lines = [
        "# BG Branch Pool Reachability Gate (2026-05-18)",
        "",
        f"BG_REACHABILITY_GATE_VERDICT = {verdict}",
        f"GENERATOR_REACHABILITY_LIMITED = {str(verdict == 'GENERATOR_REACHABILITY_LIMITED').lower()}",
        "",
        "## Domain Oracle Rates",
        "",
        "| Domain | Tasks | Oracle successes | Rate | Threshold | Meets |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for domain, row in by_domain.items():
        lines.append(
            f"| `{domain}` | {row['task_count']} | {row['oracle_successes']} | {row['oracle_success_rate']:.3f} | {row['threshold']:.2f} | {row['meets_threshold']} |"
        )
    lines.extend(["", f"- devil diagnostic: `{payload['devil']}`"])
    write_md(OUT_MD, lines)
    print(f"BG_REACHABILITY_GATE_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict in {"READY", "PARTIAL", "GENERATOR_REACHABILITY_LIMITED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
