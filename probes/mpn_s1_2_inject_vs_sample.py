"""M+N S1.2 step two — injection-branch vs sampled-branch diversity gap (validated harness).

Step 1 showed plain temperature sampling already yields diverse, solution-reaching branches
(oracle@4: math 1.0). This isolates whether hidden-state INJECTION adds anything beyond sampling
on the frozen model:
  - Arm A (sampling):  K plain samples, temp 0.8, NO hook  -> diversity from sampling.
  - Arm B (injection): K branches, GREEDY, each a different curated L24 direction (alpha 0.02),
                       sustained HiddenDeltaLayerHook -> diversity ONLY from the injection.
Same validated answering harness (build_prompt+GEN_SYSTEM+DOMAIN_MAXTOK+_B1_make_stop), BT parser
(parse_final/extract_code, the step-1 winner), external verifier only. Per task & arm: distinct
verified finals + oracle@K. The A-minus-B gap = the diversity the frozen injection mechanism
CANNOT reach but sampling can = the deficit S3 (backbone training) must close.

Run on GPU (backgrounded): venv/bin/python utilities/tests/manual/mpn_s1_2_inject_vs_sample.py
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
import branch_training_v1_common as B1  # noqa: E402
import branch_training_v2_common as V  # noqa: E402
from mpn_s1_2_base_validate import gen_raw, verify  # noqa: E402  (validated sampled harness + verifier)
from mpn_s1_2_basisbank import curated_l24_directions  # noqa: E402
from src.evaluator.bg_hidden_branching import HiddenDeltaLayerHook  # noqa: E402

OUT = V.PROJECT_ROOT / "artifacts/reports/probes/mpn_s1_baseline_2026-06-13"
CANARY = V.DATA_ROOT / "canary/offline_branch_generator_canary_v2.jsonl"
DOMAINS = ("math", "reasoning", "logic", "coding")
N_PER = int(os.environ.get("S1_N_PER", "3"))
K = int(os.environ.get("S1_K", "4"))
ALPHA = float(os.environ.get("S1_ALPHA", "0.02"))


def gen_one_hooked(model, tok, task, delta):
    """One GREEDY generation with a sustained injection hook (raw text)."""
    from transformers import StoppingCriteriaList
    d = task["domain"]; mnt = V.DOMAIN_MAXTOK.get(d, 320)
    prompt = V.build_prompt(tok, task, "direct")
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1536).to("cuda")
    plen = enc["input_ids"].shape[1]
    h = HiddenDeltaLayerHook(model, target_layer=24, target_loops=[1], delta=delta,
                             position=-1, max_rms_fraction=min(ALPHA, 0.10))
    try:
        h.apply()
        stp = StoppingCriteriaList([V._B1_make_stop(tok, plen)])
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=mnt, do_sample=False, num_return_sequences=1,
                                 pad_token_id=tok.pad_token_id, stopping_criteria=stp)
    finally:
        h.remove()
    return tok.decode(out[0][plen:], skip_special_tokens=True)


def parse(task, text):
    if task.get("label_type") == "unit_tests":
        return B1.extract_code(text)
    return B1.parse_final(text) or ""


def arm_metrics(task, texts, is_code):
    finals, correct = [], []
    for t in texts:
        p = parse(task, t)
        finals.append((p or "")[:80])
        correct.append(verify(task, p, is_code))
    return {"distinct_finals": len(set(finals)), "n_correct": int(sum(correct)),
            "oracle": bool(any(correct)), "n": len(texts)}


def main() -> int:
    started = time.time()
    sel = json.loads((OUT / "s1_1_task_set.json").read_text())
    want = {}
    for d in DOMAINS:
        for cid in [t["canary_id"] for t in sel["tasks"] if t["domain"] == d][:N_PER]:
            want[cid] = d
    tasks = [json.loads(l) for l in open(CANARY) if json.loads(l)["canary_id"] in want]
    dirs = [v for _n, v in curated_l24_directions(K)]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(V.BASE_MODEL), trust_remote_code=True, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(V.BASE_MODEL), torch_dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True, low_cpu_mem_usage=True).to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)

    rows = []
    for task in tasks:
        d = task["domain"]; is_code = task.get("label_type") == "unit_tests"
        samp = arm_metrics(task, gen_raw(model, tok, task, K), is_code)           # arm A: sampling
        inj_texts = [gen_one_hooked(model, tok, task, dirs[j % len(dirs)] * ALPHA) for j in range(K)]
        inj = arm_metrics(task, inj_texts, is_code)                               # arm B: injection
        rows.append({"task": task["canary_id"], "domain": d, "sampling": samp, "injection": inj})
        print(f"  {d:9} {task['canary_id']}: SAMP(distinct={samp['distinct_finals']},orc={samp['oracle']}) "
              f"INJ(distinct={inj['distinct_finals']},orc={inj['oracle']})", flush=True)

    def agg(arm):
        n = len(rows)
        return {"mean_distinct": round(sum(r[arm]["distinct_finals"] for r in rows) / n, 3),
                "oracle_at_K": round(sum(1 for r in rows if r[arm]["oracle"]) / n, 3),
                "by_domain": {dm: {"mean_distinct": round(sum(r[arm]["distinct_finals"] for r in rows if r["domain"] == dm)
                                   / max(1, sum(1 for r in rows if r["domain"] == dm)), 2),
                                   "oracle": round(sum(1 for r in rows if r["domain"] == dm and r[arm]["oracle"])
                                   / max(1, sum(1 for r in rows if r["domain"] == dm)), 2)} for dm in DOMAINS}}
    sA, sB = agg("sampling"), agg("injection")
    payload = {"verdict": "S1_2_INJECT_VS_SAMPLE", "N_PER": N_PER, "K": K, "alpha": ALPHA, "n_tasks": len(rows),
               "sampling": sA, "injection": sB,
               "oracle_gap_sampling_minus_injection": round(sA["oracle_at_K"] - sB["oracle_at_K"], 3),
               "distinct_gap": round(sA["mean_distinct"] - sB["mean_distinct"], 3),
               "rows": rows, "elapsed_seconds": round(time.time() - started, 1)}
    (OUT / "s1_2_inject_vs_sample.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2))
    print(f"\nGAP oracle(sampling-injection)={payload['oracle_gap_sampling_minus_injection']} "
          f"distinct={payload['distinct_gap']} — the deficit S3 must close. S1_2_INJECT_VS_SAMPLE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
