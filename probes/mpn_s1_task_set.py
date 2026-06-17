"""M+N S1.1 — lock a fixed, domain-balanced task set and smoke the REAL verifier path.

Selects a deterministic balanced subset (logic/math/reasoning/coding, N/domain) from the canary,
and for each task confirms the actual `label_branch` path returns pass on the task's known-good
answer (coding: gold code fenced; others: 'FINAL ANSWER: <gold>', with an MCQ letter fallback).
This is the ground-truth substrate for S1.2+; any task whose verifier won't accept its own gold is
flagged out now. Writes the locked task-set manifest to the S1 root. Alignment is excluded (no
objective verifier for generated branches).

Run: venv/bin/python utilities/tests/manual/mpn_s1_task_set.py
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import branch_training_v1_common as B1  # noqa: E402

OUT = B1.PROJECT_ROOT / "artifacts/reports/probes/mpn_s1_baseline_2026-06-13"
CANARY = B1.DATA_ROOT / "canary/offline_branch_generator_canary_v2.jsonl"
DOMAINS = ("logic", "math", "reasoning", "coding")
N_PER = int(os.environ.get("S1_N_PER_DOMAIN", "8"))


def _gold_completion(task: dict) -> tuple[str, str]:
    """Build a known-good completion + the label-path used, per domain."""
    if task.get("label_type") == "unit_tests":
        return f"```python\n{task.get('gold_answer', '')}\n```", "unit_tests"
    gold = task.get("gold_answer")
    if gold is None and task.get("answer_key") is not None:
        gold = chr(65 + int(task["answer_key"]))
    return f"FINAL ANSWER: {gold}", "final_answer"


def _mcq_letter_fallback(task: dict) -> str | None:
    if task.get("answer_key") is not None:
        return f"FINAL ANSWER: {chr(65 + int(task['answer_key']))}"
    return None


def main() -> int:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(CANARY)]
    by_dom: dict[str, list[dict]] = {d: [] for d in DOMAINS}
    for r in rows:
        if r.get("domain") in by_dom:
            by_dom[r["domain"]].append(r)
    selected, smoke = [], {"pass": 0, "fail": 0, "by_domain": {}}
    for dom in DOMAINS:
        cand = sorted(by_dom[dom], key=lambda r: r["canary_id"])[:N_PER]
        dom_pass = 0
        for task in cand:
            comp, path = _gold_completion(task)
            label, reward, final, parse_ok = B1.label_branch(task, comp)
            verified = reward > 0
            if not verified:  # MCQ letter fallback
                fb = _mcq_letter_fallback(task)
                if fb:
                    label, reward, final, parse_ok = B1.label_branch(task, fb)
                    verified = reward > 0
            dom_pass += int(verified)
            smoke["pass" if verified else "fail"] += 1
            selected.append({"canary_id": task["canary_id"], "domain": dom,
                             "label_type": task.get("label_type"), "category": task.get("category"),
                             "source": task.get("source"), "K": task.get("K"),
                             "verifier_smoke_pass": verified, "label_path": path})
        smoke["by_domain"][dom] = {"n": len(cand), "verifier_pass": dom_pass}

    verdict = "S1_TASK_SET_READY" if smoke["fail"] == 0 and all(
        smoke["by_domain"][d]["n"] == N_PER for d in DOMAINS) else "S1_TASK_SET_VERIFIER_GAP"
    payload = {"verdict": verdict, "n_per_domain": N_PER, "n_total": len(selected),
               "verifier_smoke": smoke, "tasks": selected,
               "alignment_excluded": "no objective verifier for generated branches",
               "elapsed_seconds": round(time.time() - started, 1)}
    (OUT / "s1_1_task_set.json").write_text(json.dumps(payload, indent=2))
    print(f"selected {len(selected)} tasks across {DOMAINS} (N/domain={N_PER})")
    print(f"verifier smoke: {smoke['pass']} pass / {smoke['fail']} fail")
    for d in DOMAINS:
        print(f"  {d}: {smoke['by_domain'][d]}")
    print(f"\n{verdict}")
    return 0 if verdict == "S1_TASK_SET_READY" else 4


if __name__ == "__main__":
    raise SystemExit(main())
