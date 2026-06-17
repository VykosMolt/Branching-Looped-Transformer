"""M+N S1.2 step two — FAITHFUL generation readout of branch outcome-distinctness.

The MCQ hard-argmax readout (saturated) said 0 distinctness. This is the real test: generate
each branch to a committed answer (FINAL ANSWER / fenced code) under injection and label with the
actual external verifier. Measures whether injected branches produce DISTINCT FINAL ANSWERS (and
distinct CORRECT outcomes) — the deployment-relevant readout, less saturated than the argmax.

Injection = sustained HiddenDeltaLayerHook during generation (fires at the target loop each step);
this is the EASIER bar than a clean one-time-fork — if even sustained injection gives 0 distinct
answers, contractivity under generation is strong; if it gives distinct answers, branches are
achievable. Curated L24 high-yield directions (reused from the basis-bank module).

Bounded: N_PER tasks each of reasoning/math/coding, K injected branches. Run on GPU.
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
import mpn_s1_2_basisbank as BB  # noqa: E402  (curated_l24_directions)
from src.evaluator.bg_hidden_branching import HiddenDeltaLayerHook  # noqa: E402

OUT = V.PROJECT_ROOT / "artifacts/reports/probes/mpn_s1_baseline_2026-06-13"
DOMAINS = ("reasoning", "math", "coding")
N_PER = int(os.environ.get("S1_N_PER", "2"))
K = int(os.environ.get("S1_K", "4"))
ALPHA = float(os.environ.get("S1_ALPHA", "0.02"))


def _gen(model, tok, prompt, domain):
    from transformers import StoppingCriteriaList
    enc = {k: v.to("cuda") for k, v in tok(prompt, return_tensors="pt", truncation=True, max_length=1536).items()}
    plen = enc["input_ids"].shape[1]
    mnt = V.DOMAIN_MAXTOK.get(domain, 320)
    stp = StoppingCriteriaList([V._B1_make_stop(tok, plen)])
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=mnt, do_sample=False, num_return_sequences=1,
                             pad_token_id=tok.pad_token_id, stopping_criteria=stp)
    return V.trim_generation(tok.decode(out[0][plen:], skip_special_tokens=True))


def main() -> int:
    started = time.time()
    sel = json.loads((OUT / "s1_1_task_set.json").read_text())
    want = {}
    for d in DOMAINS:
        ids = [t["canary_id"] for t in sel["tasks"] if t["domain"] == d][:N_PER]
        for i in ids:
            want[i] = d
    tasks = [json.loads(l) for l in open(BB.MCQ.CANARY) if json.loads(l)["canary_id"] in want]
    dirs = BB.curated_l24_directions(K)

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
        dom = task["domain"]
        prompt = B1.build_prompt(tok, task, "direct")
        base_txt = _gen(model, tok, prompt, dom)
        bl, br, bfin, _ = B1.label_branch(task, base_txt)
        finals, labels = [str(bfin)[:48]], [br > 0]
        for _name, vec in dirs:
            h = HiddenDeltaLayerHook(model, target_layer=24, target_loops=[1], delta=vec * ALPHA,
                                     position=-1, max_rms_fraction=min(ALPHA, 0.10))
            try:
                h.apply(); txt = _gen(model, tok, prompt, dom)
            finally:
                h.remove()
            _l, r, fin, _ok = B1.label_branch(task, txt)
            finals.append(str(fin)[:48]); labels.append(r > 0)
        distinct_finals = len(set(finals))
        rows.append({"task": task["canary_id"], "domain": dom, "base_correct": labels[0],
                     "distinct_finals": distinct_finals, "n_correct": int(sum(labels)),
                     "positive_oracle": bool(any(labels)), "finals": finals})
        print(f"  {dom:9} {task['canary_id']}: distinct_finals={distinct_finals}/{K+1} "
              f"base_correct={labels[0]} oracle={any(labels)}", flush=True)

    n = len(rows)
    payload = {"verdict": "S1_2_GEN_LABEL_MEASURED", "alpha": ALPHA, "K": K, "n_tasks": n,
               "mean_distinct_finals": round(sum(r["distinct_finals"] for r in rows) / n, 3) if n else 0,
               "distinct_rate": round(sum(1 for r in rows if r["distinct_finals"] > 1) / n, 3) if n else 0,
               "base_accuracy": round(sum(1 for r in rows if r["base_correct"]) / n, 3) if n else 0,
               "positive_oracle_rate": round(sum(1 for r in rows if r["positive_oracle"]) / n, 3) if n else 0,
               "by_domain": {d: [r["distinct_finals"] for r in rows if r["domain"] == d] for d in DOMAINS},
               "rows": rows, "elapsed_seconds": round(time.time() - started, 1)}
    (OUT / "s1_2_gen_label.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2))
    print("\nS1_2_GEN_LABEL_MEASURED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
