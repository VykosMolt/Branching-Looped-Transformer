"""M+N S1.2-scale (MCQ cheap path) — outcome-distinctness of K injected branches.

For the MCQ domains (reasoning/logic), a branch's outcome reads from a SINGLE forward's
option-letter distribution (no generation, no fragile cache path). For each task we make K
branches at L1_24 (same HiddenDeltaLayerHook injection, varied basis directions, safe alpha),
read each branch's argmax option vs the answer key, and measure:
  - outcome-distinctness: # distinct predicted letters among the K branches (>1 = genuinely
    distinct branches; 1 = near-neighbor ties),
  - positive-oracle: any branch correct,
  - base correctness.
This is the first real "ties vs distinct" measurement on live branches. Uses an MCQ prompt that
ends at 'Answer:' so the immediate next-token option-lean is a valid outcome proxy.

NOTE (surfaced for the build): math/coding outcome-labeling cannot use this single-forward path —
it needs one-time-inject-at-prefill + KV-carry decode to a verified answer. That is the next,
separate S1.2 build (the generation fork-carry, previously hook-fallback-only).

Run on GPU: venv/bin/python utilities/tests/manual/mpn_s1_2_mcq_distinct.py
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

import torch  # noqa: E402
import branch_training_v2_common as V  # noqa: E402
from src.evaluator.bg_hidden_branching import make_branch_deltas, HiddenDeltaLayerHook  # noqa: E402

OUT = V.PROJECT_ROOT / "artifacts/reports/probes/mpn_s1_baseline_2026-06-13"
TASK_SET = OUT / "s1_1_task_set.json"
CANARY = V.DATA_ROOT / "canary/offline_branch_generator_canary_v2.jsonl"
MCQ_DOMAINS = ("reasoning", "logic")
TARGET_LAYER, TARGET_LOOP, ALPHA, MAX_RMS_FRAC = 24, 1, 0.01, 0.02
K = int(os.environ.get("S1_K", "4"))


def _mcq_prompt(task) -> tuple[str, list[str]]:
    opts = task.get("options") or []
    letters = [chr(65 + j) for j in range(len(opts))]
    body = task.get("task_prompt", "").strip()
    lines = "\n".join(f"{L}. {o}" for L, o in zip(letters, opts))
    return f"{body}\nOptions:\n{lines}\nAnswer:", letters


def _letter_ids(tok, letters):
    ids = {}
    for L in letters:
        s = set()
        for t in (" " + L, L, " " + L + ".", L + "."):
            e = tok(t, add_special_tokens=False)["input_ids"]
            if e:
                s.add(int(e[0]))
        ids[L] = s
    return ids


def _pred_letter(tok, model, enc, letters, lid):
    with torch.inference_mode():
        out = model(**enc, use_cache=False)
    lg = (out.logits if hasattr(out, "logits") else out[0])[0, -1].float()
    probs = torch.softmax(lg, dim=-1)
    lean = {L: float(sum(probs[i].item() for i in lid[L] if 0 <= i < probs.numel())) for L in letters}
    return max(lean, key=lean.get), lean


def main() -> int:
    started = time.time()
    sel = json.loads(TASK_SET.read_text())
    want_ids = {t["canary_id"] for t in sel["tasks"] if t["domain"] in MCQ_DOMAINS}
    tasks = [json.loads(l) for l in open(CANARY) if json.loads(l)["canary_id"] in want_ids]
    tasks = [t for t in tasks if (t.get("options") and t.get("answer_key") is not None)]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(V.BASE_MODEL), trust_remote_code=True, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(V.BASE_MODEL), torch_dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True, low_cpu_mem_usage=True).to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    hidden = model.config.hidden_size

    rows = []
    for ti, task in enumerate(tasks):
        prompt, letters = _mcq_prompt(task)
        gold = chr(65 + int(task["answer_key"])) if int(task["answer_key"]) < len(letters) else None
        enc = {k: v.to("cuda") for k, v in tok(prompt, return_tensors="pt", truncation=True, max_length=1536).items()}
        lid = _letter_ids(tok, letters)
        base_pred, _ = _pred_letter(tok, model, enc, letters, lid)
        deltas = make_branch_deltas(hidden, k=K + 1, alpha=ALPHA, seed=20260613 + ti * 17, include_clean=True)
        branch_preds = []
        for d in deltas[1:K + 1]:
            h = HiddenDeltaLayerHook(model, target_layer=TARGET_LAYER, target_loops=[TARGET_LOOP],
                                     delta=d, position=-1, max_rms_fraction=MAX_RMS_FRAC)
            try:
                h.apply(); p, _ = _pred_letter(tok, model, enc, letters, lid)
            finally:
                h.remove()
            branch_preds.append(p)
        all_preds = [base_pred] + branch_preds
        distinct = len(set(all_preds))
        any_correct = gold in all_preds if gold else None
        rows.append({"task": task["canary_id"], "domain": task["domain"], "gold": gold,
                     "base_pred": base_pred, "branch_preds": branch_preds,
                     "distinct_outcomes": distinct, "base_correct": base_pred == gold,
                     "positive_oracle": any_correct})

    n = len(rows)
    distinct_rate = sum(1 for r in rows if r["distinct_outcomes"] > 1) / n if n else 0.0
    mean_distinct = sum(r["distinct_outcomes"] for r in rows) / n if n else 0.0
    base_acc = sum(1 for r in rows if r["base_correct"]) / n if n else 0.0
    oracle = sum(1 for r in rows if r["positive_oracle"]) / n if n else 0.0
    payload = {"verdict": "S1_2_MCQ_DISTINCTNESS_MEASURED", "K": K, "alpha": ALPHA,
               "n_tasks": n, "branch_distinct_rate": round(distinct_rate, 3),
               "mean_distinct_outcomes": round(mean_distinct, 3), "max_possible": K + 1,
               "base_accuracy": round(base_acc, 3), "positive_oracle_rate": round(oracle, 3),
               "by_domain": {d: sum(1 for r in rows if r["domain"] == d) for d in MCQ_DOMAINS},
               "rows": rows, "elapsed_seconds": round(time.time() - started, 1)}
    (OUT / "s1_2_mcq_distinct.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2))
    print(f"\ninterpretation: mean_distinct≈1 over {K+1} branches ⇒ near-neighbor ties (expected pre-training)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
