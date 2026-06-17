"""M+N S1.2 — multi-locus compounding injection vs single-locus vs sampling.

Step 2 injected at ONE locus (L24/L1) and found oracle parity with sampling. But the architecture
perturbs at 24, 36, AND 47 across ALL loops (compounding). This tests whether compounding
multi-locus perturbation diverges more / reaches more than single-L24/L1 — vs the sampling bound.
Three arms (K branches each), greedy (diversity only from the perturbation), validated harness +
BT parser + external verifier:
  - SAMP        : K plain samples (temp 0.8)            [reachability bound]
  - SINGLE      : K branches, inject @L24/loop1 only    [what step 2 used]
  - MULTI       : K branches, inject @24+36+47 across loops 1-4 (3 hooks, curated per-layer dirs)
This is SUSTAINED multi-locus (hooks fire each step) — NOT yet the one-time-per-locus+KV-carry loop
(that's the S1.4 build); it's the fast test of whether compounding loci matter at all.

Run on GPU (backgrounded): venv/bin/python utilities/tests/manual/mpn_s1_2_multilocus.py
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
from mpn_s1_2_base_validate import gen_raw, verify  # noqa: E402
from src.evaluator.bg_hidden_branching import HiddenDeltaLayerHook, rms_normalize  # noqa: E402


def parse(task, text):
    if task.get("label_type") == "unit_tests":
        return B1.extract_code(text)
    return B1.parse_final(text) or ""

OUT = V.PROJECT_ROOT / "artifacts/reports/probes/mpn_s1_baseline_2026-06-13"
CANARY = V.DATA_ROOT / "canary/offline_branch_generator_canary_v2.jsonl"
BANK = V.PROJECT_ROOT / "artifacts/reports/probes/bg_hidden_origin_branch_generator_v1_2026-05-18/basis_bank_v1.pt"
DOMAINS = ("math", "reasoning", "logic", "coding")
N_PER = int(os.environ.get("S1_N_PER", "3"))
K = int(os.environ.get("S1_K", "4"))
ALPHA = float(os.environ.get("S1_ALPHA", "0.02"))
LAYERS = (24, 36, 47)


def curated_dirs_by_layer(k):
    """k curated (non-random, high-yield/tap-aligned) directions per layer 24/36/47."""
    bank = torch.load(BANK, map_location="cpu", weights_only=False)
    pri = ["high_yield_recipe_direction", "v4_tap_aligned", "v2_tap_aligned", "v1_tap_aligned",
           "salvage_tap_aligned", "old_tap_aligned"]
    out = {L: [] for L in LAYERS}
    for L in LAYERS:
        byf = {}
        for d in bank["directions"]:
            tl = d.get("target_layer") if isinstance(d, dict) else None
            if tl is None or int(tl) != L or not d.get("perturbation_usable") or d.get("family") == "random_orthogonal":
                continue
            t = d.get("tensor")
            if isinstance(t, torch.Tensor) and t.numel() == 2048:
                byf.setdefault(d["family"], []).append(rms_normalize(t.float().flatten()))
        for fam in pri:
            for v in byf.get(fam, []):
                out[L].append(v)
                if len(out[L]) >= k:
                    break
            if len(out[L]) >= k:
                break
    return out


def gen_hooked(model, tok, task, hooks):
    from transformers import StoppingCriteriaList
    d = task["domain"]; mnt = V.DOMAIN_MAXTOK.get(d, 320)
    enc = tok(V.build_prompt(tok, task, "direct"), return_tensors="pt", truncation=True, max_length=1536).to("cuda")
    plen = enc["input_ids"].shape[1]
    for h in hooks:
        h.apply()
    try:
        stp = StoppingCriteriaList([V._B1_make_stop(tok, plen)])
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=mnt, do_sample=False, num_return_sequences=1,
                                 pad_token_id=tok.pad_token_id, stopping_criteria=stp)
    finally:
        for h in hooks:
            h.remove()
    return tok.decode(out[0][plen:], skip_special_tokens=True)


def metrics(task, texts, is_code):
    finals, correct = [], []
    for t in texts:
        p = parse(task, t); finals.append((p or "")[:80]); correct.append(verify(task, p, is_code))
    return {"distinct": len(set(finals)), "oracle": bool(any(correct)), "n_correct": int(sum(correct))}


def main() -> int:
    started = time.time()
    sel = json.loads((OUT / "s1_1_task_set.json").read_text())
    want = {}
    for d in DOMAINS:
        for cid in [t["canary_id"] for t in sel["tasks"] if t["domain"] == d][:N_PER]:
            want[cid] = d
    tasks = [json.loads(l) for l in open(CANARY) if json.loads(l)["canary_id"] in want]
    dirs = curated_dirs_by_layer(K)

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
        samp = metrics(task, gen_raw(model, tok, task, K), is_code)
        single_t, multi_t = [], []
        for j in range(K):
            h1 = [HiddenDeltaLayerHook(model, target_layer=24, target_loops=[1],
                                       delta=dirs[24][j % len(dirs[24])] * ALPHA, position=-1, max_rms_fraction=min(ALPHA, 0.10))]
            single_t.append(gen_hooked(model, tok, task, h1))
            hm = [HiddenDeltaLayerHook(model, target_layer=L, target_loops=([1, 2, 3, 4] if L != 47 else [1, 2, 3]),
                                       delta=dirs[L][j % len(dirs[L])] * ALPHA, position=-1, max_rms_fraction=min(ALPHA, 0.10))
                  for L in LAYERS if dirs[L]]
            multi_t.append(gen_hooked(model, tok, task, hm))
        rec = {"task": task["canary_id"], "domain": d, "samp": samp,
               "single": metrics(task, single_t, is_code), "multi": metrics(task, multi_t, is_code)}
        rows.append(rec)
        print(f"  {d:9} {task['canary_id']}: SAMP(d={samp['distinct']},o={samp['oracle']}) "
              f"SINGLE(d={rec['single']['distinct']},o={rec['single']['oracle']}) "
              f"MULTI(d={rec['multi']['distinct']},o={rec['multi']['oracle']})", flush=True)

    def agg(arm):
        n = len(rows)
        return {"mean_distinct": round(sum(r[arm]["distinct"] for r in rows) / n, 3),
                "oracle_at_K": round(sum(1 for r in rows if r[arm]["oracle"]) / n, 3)}
    payload = {"verdict": "S1_2_MULTILOCUS", "N_PER": N_PER, "K": K, "alpha": ALPHA, "layers": list(LAYERS),
               "n_tasks": len(rows), "sampling": agg("samp"), "single_locus": agg("single"), "multi_locus": agg("multi"),
               "rows": rows, "elapsed_seconds": round(time.time() - started, 1)}
    (OUT / "s1_2_multilocus.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2))
    print("\nS1_2_MULTILOCUS — does multi-locus compounding beat single-locus / reach sampling?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
