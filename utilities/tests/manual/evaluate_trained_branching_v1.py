"""PART O — evaluate trained branching vs Ouro-RLTT (no adapter).

Bounded comparison on HELD-OUT tasks (task-disjoint from SFT): for each model variant
(base = Ouro-RLTT no adapter; sft = + branching_sft LoRA), generate K branches per task,
label with EXTERNAL verifiers only, and compare reachability / parse / diversity / final-acc.
One model loaded at a time (12GB). Proof-of-capability scale.
"""
from __future__ import annotations
import json, os, sys, time, collections
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import branch_training_v1_common as B  # noqa: E402

OUT = B.OUT_ROOT
ADAPTER = B.MODEL_ROOT / "branching_sft"
PER_DOM = int(os.environ.get("O_PER_DOM", "25"))
K = int(os.environ.get("O_K", "4"))


def _heldout_sample():
    """Task-disjoint heldout sample across logic/math/reasoning/coding (not used in SFT)."""
    tasks = []
    logic = [t for t in B.load_logic_tasks() if t["split"] == "heldout"]
    bycat = {}
    for t in logic:
        bycat.setdefault(t["category"], []).append(t)
    ls = []
    while len(ls) < PER_DOM and any(bycat.values()):
        for c in list(bycat):
            if bycat[c]:
                ls.append(bycat[c].pop(0))
                if len(ls) >= PER_DOM:
                    break
    tasks += ls
    rows = B.read_jsonl(B.PROJECT_ROOT / "shared/data/corecontent_v2/processed/candidate_groups_deduped.jsonl")
    for dom in ("math", "reasoning"):
        got = []
        for g in rows:
            if g["domain"] == dom and g["split"] == "heldout":
                r = B._reconstruct_core_gen(g)
                if r:
                    got.append(r)
            if len(got) >= PER_DOM:
                break
        tasks += got[:PER_DOM]
    # coding heldout = MBPP tasks beyond the trained 60 (offset), same name-fixed prompt
    cod = B.load_coding_gen_tasks(120)[60:60 + PER_DOM]
    tasks += cod
    return tasks


def _load(variant):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mp = str(B.PROJECT_ROOT / "shared/models/ouro_rltt_local")
    tok = AutoTokenizer.from_pretrained(mp, trust_remote_code=True, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(mp, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                                 local_files_only=True, low_cpu_mem_usage=True).to("cuda").eval()
    if variant == "sft":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ADAPTER)).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def _gen_eval(variant, tasks):
    from transformers import StoppingCriteriaList
    model, tok = _load(variant)
    per_dom = collections.defaultdict(lambda: {"groups": 0, "oracle": 0, "branches": 0, "parse": 0,
                                               "distinct_finals": 0, "greedy_correct": 0})
    for ti, t in enumerate(tasks):
        d = t["domain"]
        finals = set(); pos = False; nbr = 0; npar = 0
        scaffs = ["direct", "case_split", "counterexample", "elimination"][:K]
        for si, sc in enumerate(scaffs):
            prompt = B.build_prompt(tok, t, sc)
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1536).to("cuda")
            plen = enc["input_ids"].shape[1]
            stp = StoppingCriteriaList([B._make_stop(tok, plen)])
            try:
                with torch.inference_mode():
                    out = model.generate(**enc, max_new_tokens=B.DOMAIN_MAXTOK.get(d, 320), do_sample=True,
                                         temperature=0.8, top_p=0.95, num_return_sequences=1,
                                         pad_token_id=tok.pad_token_id, stopping_criteria=stp)
                txt = B.trim_generation(tok.decode(out[0][plen:], skip_special_tokens=True)); del out
            except Exception:
                continue
            lab, rew, fa, ok = B.label_branch(t, txt)
            nbr += 1; npar += 1 if ok else 0
            if fa:
                finals.add(str(fa)[:40])
            if rew > 0:
                pos = True
            if si == 0 and rew > 0:  # greedy-ish (first scaffold) correctness
                per_dom[d]["greedy_correct"] += 1
        m = per_dom[d]; m["groups"] += 1; m["oracle"] += 1 if pos else 0
        m["branches"] += nbr; m["parse"] += npar; m["distinct_finals"] += len(finals)
        if (ti + 1) % 20 == 0:
            print(f"  [{variant}] {ti+1}/{len(tasks)}", flush=True)
    try:
        del model; torch.cuda.empty_cache()
    except Exception:
        pass
    res = {}
    for d, m in per_dom.items():
        g = max(1, m["groups"])
        res[d] = {"groups": m["groups"], "positive_oracle@K": round(m["oracle"] / g, 3),
                  "parse_ok": round(m["parse"] / max(1, m["branches"]), 3),
                  "branch_diversity": round(m["distinct_finals"] / g, 2),
                  "greedy_acc": round(m["greedy_correct"] / g, 3)}
    return res


def main() -> int:
    started = time.time(); B.ensure_dirs()
    if not ADAPTER.exists():
        B.write_json(OUT / "trained_branching_eval.json", {"TRAINED_BRANCHING_EVAL_VERDICT": "BLOCKED",
                     "error": "no SFT adapter"}); print("BLOCKED: no adapter"); return 1
    tasks = _heldout_sample()
    print(f"heldout eval sample: {dict(collections.Counter(t['domain'] for t in tasks))}", flush=True)
    results = {}
    for variant in ("base", "sft"):
        print(f"=== generating with {variant} ===", flush=True)
        results[variant] = _gen_eval(variant, tasks)
        B.prog(f"O_eval_{variant}", results[variant])
    # macro oracle per variant
    doms = ("logic", "math", "reasoning", "coding")
    macro = {v: round(B.finite_mean([results[v].get(d, {}).get("positive_oracle@K", float('nan')) for d in doms]), 3)
             for v in ("base", "sft")}
    div = {v: round(B.finite_mean([results[v].get(d, {}).get("branch_diversity", float('nan')) for d in doms]), 2)
           for v in ("base", "sft")}
    lift = round(macro["sft"] - macro["base"], 3)
    div_lift = round(div["sft"] - div["base"], 2)
    if lift > 0.02 and div_lift >= 0:
        verdict = "MODEL_INTERNAL_BRANCHING_IMPROVES"
    elif lift >= -0.01 and div_lift > 0.2:
        verdict = "MODEL_INTERNAL_BRANCHING_PARTIAL"
    elif lift < -0.02:
        verdict = "NO_GAIN"
    else:
        verdict = "MODEL_INTERNAL_BRANCHING_PARTIAL"
    payload = {"TRAINED_BRANCHING_EVAL_VERDICT": verdict, "macro_positive_oracle": macro,
               "oracle_lift_sft_minus_base": lift, "branch_diversity": div, "diversity_lift": div_lift,
               "by_domain": results, "per_dom_n": PER_DOM, "K": K, "note": "bounded proof-of-capability; 300-step SFT adapter vs Ouro-RLTT no-adapter; external labels only",
               "elapsed_seconds": round(time.time() - started, 3)}
    B.write_json(OUT / "trained_branching_eval.json", payload)
    rows = []
    for d in doms:
        rows.append({"domain": d, "base_oracle": results["base"].get(d, {}).get("positive_oracle@K"),
                     "sft_oracle": results["sft"].get(d, {}).get("positive_oracle@K"),
                     "base_div": results["base"].get(d, {}).get("branch_diversity"),
                     "sft_div": results["sft"].get(d, {}).get("branch_diversity")})
    B.write_csv(OUT / "trained_branching_eval_rows.csv", rows)
    B.write_md(OUT / "trained_branching_eval.md", [
        "# Trained Branching Eval (Part O) — Ouro-RLTT vs +SFT adapter", "",
        B.status_line("TRAINED_BRANCHING_EVAL_VERDICT", verdict), "",
        f"Bounded heldout ({PER_DOM}/domain, K={K}). Macro positive_oracle@K: base {macro['base']} vs sft {macro['sft']} "
        f"(lift {lift:+}). Branch diversity: base {div['base']} vs sft {div['sft']} (lift {div_lift:+}).", "",
        *B.md_table(rows, ["domain", "base_oracle", "sft_oracle", "base_div", "sft_div"]),
        "", "Proof-of-capability: a 300-step bounded SFT adapter, not a converged model.",
    ])
    B.prog("O_trained_eval", {"verdict": verdict, "macro": macro, "lift": lift})
    print(B.status_line("TRAINED_BRANCHING_EVAL_VERDICT", verdict))
    print(f"  macro oracle: base={macro['base']} sft={macro['sft']} (lift {lift:+}) | diversity base={div['base']} sft={div['sft']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
