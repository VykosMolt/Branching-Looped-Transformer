"""PART D — baseline canary evaluation (resumable, STOP-pausable).

Establishes fixed pre-training baselines on the Part-C canary. One generating model per
invocation via VARIANT=base|prev_sft (resumable per-variant shard manifest); alignment is a
preference-accuracy sentinel (logprob, no generation). VARIANT=aggregate combines whatever is
complete + derived baselines (oracle/random/first-final) and writes the report with Wilson CIs.

Generation is batched (num_return_sequences=K) for short-budget domains, single-seq for math
(4-loop KV OOM guard). Derived baselines need no extra generation. External DualAnchor+CoreContent
terminal numbers are referenced from v1 Experiment-1 (feature extraction deferred to Part L).
"""
from __future__ import annotations
import json
import math
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

VARIANT = os.environ.get("VARIANT", "base")
SCAFFOLDS = ["direct", "case_split", "counterexample", "elimination", "alt_strategy", "verify", "decompose", "reframe"]
RESULT_DIR = V.OUT_ROOT / "canary_gen"
LOGP_BATCH = 8


def _canary():
    return [json.loads(l) for l in open(V.CANARY_DIR / "offline_branch_generator_canary_v2.jsonl")]


def _load(variant):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(V.BASE_MODEL), trust_remote_code=True, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(V.BASE_MODEL), torch_dtype=torch.bfloat16, trust_remote_code=True,
                                                 local_files_only=True, low_cpu_mem_usage=True).to("cuda").eval()
    if variant == "prev_sft":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(V.PREV_SFT_ADAPTER)).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def _pref_logprob(model, tok, prompt, full_text):
    """length-normalized logprob of the response (the part of full_text AFTER the shared prompt).

    `full_text` already contains `prompt` as a prefix (canary stores full chosen/rejected),
    so we score only the divergent suffix conditioned on the prompt — no double-prepend.
    """
    ids = tok(full_text, return_tensors="pt", truncation=True, max_length=2048).input_ids.to("cuda")
    plen = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).input_ids.shape[1] if prompt else 1
    plen = max(1, min(plen, ids.shape[1] - 1))
    with torch.inference_mode():
        logits = model(ids).logits
    logp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    tgt = ids[0, 1:]
    sel = logp[plen - 1:, :].gather(1, tgt[plen - 1:].unsqueeze(1)).squeeze(1)
    return float(sel.mean()) if sel.numel() else -1e9


def _gen_branches(model, tok, task, K):
    from transformers import StoppingCriteriaList
    d = task["domain"]
    mnt = V.DOMAIN_MAXTOK.get(d, 320)
    prompt = V.build_prompt(tok, task, "direct")
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1536).to("cuda")
    plen = enc["input_ids"].shape[1]
    texts = []
    chunk = 1 if mnt >= 1000 else (2 if mnt >= 600 else 4)  # KV-safe per-call batch (4-loop cache)
    remaining = K
    guard = 0
    while remaining > 0 and guard < K + 6:
        guard += 1
        nb = min(chunk, remaining)
        try:
            stp = StoppingCriteriaList([V._B1_make_stop(tok, plen)])
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=mnt, do_sample=True, temperature=0.8, top_p=0.95,
                                     num_return_sequences=nb, pad_token_id=tok.pad_token_id, stopping_criteria=stp)
            for i in range(out.shape[0]):
                texts.append(V.trim_generation(tok.decode(out[i][plen:], skip_special_tokens=True)))
            del out
            remaining -= nb
            torch.cuda.empty_cache()
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            torch.cuda.empty_cache()
            if "out of memory" in str(e).lower():
                if chunk > 1:
                    chunk = max(1, chunk // 2)  # back off batch size and retry
                    continue
                remaining -= 1  # single-seq still OOMs: skip this branch
                continue
            raise
    return texts


def _score_pool(task, texts):
    finals, labels, parses, recs = [], [], 0, []
    for t in texts:
        lab, rew, fa, ok = V.label_branch(task, t)  # labeled once
        labels.append(rew > 0)
        parses += 1 if ok else 0
        if fa:
            finals.append(str(fa)[:48])
        recs.append({"text": t[:2000], "final": str(fa)[:48] if fa else None, "label": "pass" if rew > 0 else "fail"})
    K = max(1, len(texts))
    distinct = len(set(finals))
    pos_distinct = len(set(f for f, l in zip(finals, labels) if l))
    dup = (len(finals) - distinct) / max(1, len(finals)) if finals else 0.0
    return {"canary_id": task["canary_id"], "domain": task["domain"], "slice": task.get("slice", []),
            "K": len(texts), "n_pos": int(sum(labels)), "pos_oracle": bool(any(labels)),
            "parse_ok": parses / K, "distinct_finals": distinct, "pos_distinct": pos_distinct,
            "dup_rate": round(dup, 3), "first_correct": bool(labels[0]) if labels else False,
            "all_wrong": not any(labels), "branches": recs}


def generate_variant(variant):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULT_DIR / f"{variant}.jsonl"
    done = set()
    if out_path.exists():
        for l in open(out_path):
            try:
                done.add(json.loads(l)["canary_id"])
            except Exception:
                pass
    tasks = [c for c in _canary() if c["domain"] != "alignment" and c["canary_id"] not in done]
    align = [c for c in _canary() if c["domain"] == "alignment" and c["canary_id"] not in done]
    print(f"[{variant}] pending gen {len(tasks)} + alignment {len(align)} (done {len(done)})", flush=True)
    model, tok = _load(variant)
    f = open(out_path, "a")
    t0 = time.time()
    n = 0
    for task in tasks:
        if V.stop_requested(f"canary_{variant}"):
            print(f"[{variant}] STOP requested at {n}", flush=True)
            break
        texts = _gen_branches(model, tok, task, task.get("K", 4))
        rec = _score_pool(task, texts)
        f.write(json.dumps(rec, default=V.v2.json_default) + "\n")
        f.flush()
        n += 1
        if n % 10 == 0:
            rate = (time.time() - t0) / n
            print(f"[{variant}] {n}/{len(tasks)} gen | {rate:.0f}s/task | ETA {rate*(len(tasks)-n)/3600:.1f}h", flush=True)
    # alignment preference sentinel (logprob)
    for i, task in enumerate(align):
        if V.stop_requested(f"canary_{variant}"):
            break
        lc = _pref_logprob(model, tok, task.get("task_prompt", ""), task["pref"]["chosen"])
        lr = _pref_logprob(model, tok, task.get("task_prompt", ""), task["pref"]["rejected"])
        f.write(json.dumps({"canary_id": task["canary_id"], "domain": "alignment", "slice": ["alignment_pref"],
                            "pref_correct": bool(lc > lr), "margin": round(lc - lr, 4)}, default=V.v2.json_default) + "\n")
        f.flush()
        if (i + 1) % 25 == 0:
            print(f"[{variant}] alignment {i+1}/{len(align)}", flush=True)
    f.close()
    print(f"[{variant}] DONE wrote {out_path}", flush=True)


def _agg_variant(variant):
    p = RESULT_DIR / f"{variant}.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in open(p)]
    import collections
    byd = collections.defaultdict(list)
    for r in rows:
        byd[r["domain"]].append(r)
    out = {}
    for d, rs in byd.items():
        if d == "alignment":
            k = sum(1 for r in rs if r.get("pref_correct"))
            n = len(rs)
            out["alignment"] = {"n": n, "pref_acc": round(k / n, 4) if n else None, "ci": V.wilson_ci(k, n)}
            continue
        n = len(rs)
        po = sum(1 for r in rs if r["pos_oracle"])
        out[d] = {"n": n, "positive_oracle@K": round(po / n, 4) if n else None, "oracle_ci": V.wilson_ci(po, n),
                  "parse_ok": round(sum(r["parse_ok"] for r in rs) / n, 4) if n else None,
                  "first_correct": round(sum(1 for r in rs if r["first_correct"]) / n, 4) if n else None,
                  "branch_diversity": round(sum(r["distinct_finals"] for r in rs) / n, 3) if n else None,
                  "verifier_positive_diversity": round(sum(r["pos_distinct"] for r in rs) / n, 3) if n else None,
                  "dup_rate": round(sum(r["dup_rate"] for r in rs) / n, 3) if n else None,
                  "all_wrong_rate": round(sum(1 for r in rs if r["all_wrong"]) / n, 4) if n else None,
                  "small_n": n < 100}
    return out


def aggregate():
    base = _agg_variant("base")
    prev = _agg_variant("prev_sft")
    doms = [d for d in V.CORE_DOMAINS]
    rows = []
    for d in doms:
        b = (base or {}).get(d, {})
        s = (prev or {}).get(d, {})
        bm = b.get("positive_oracle@K") if d != "alignment" else b.get("pref_acc")
        sm = s.get("positive_oracle@K") if d != "alignment" else s.get("pref_acc")
        delta = (sm - bm) if (bm is not None and sm is not None) else None
        rows.append({"domain": d, "n_base": b.get("n"), "base": bm, "prev_sft": sm, "delta": delta,
                     "flag": V.flag(delta) if delta is not None else "NA",
                     "base_parse": b.get("parse_ok"), "base_div": b.get("branch_diversity"),
                     "small_n": b.get("small_n")})
    have_base = base is not None and all((base.get(d, {}).get("n") or 0) > 0 for d in doms)
    enough = base is not None and all((base.get(d, {}).get("n") or 0) >= 100 for d in doms if d != "alignment")
    if not have_base:
        verdict = "BLOCKED"
    elif not enough:
        verdict = "BASELINES_READY"  # partial; small_n flagged per-domain
    elif prev is not None:
        verdict = "PREVIOUS_SFT_BEHAVIOR_CONFIRMED"
    else:
        verdict = "EXTERNAL_BASELINE_CONFIRMED"
    payload = {"CANARY_BASELINE_VERDICT": verdict, "base": base, "prev_sft": prev,
               "external_stack_reference": {"note": "base+DualAnchor+CoreContent_v2 terminal from v1 Experiment-1",
                                            "corecontent_v2_within_survivors": 0.658, "dualanchor_forced_top1": 0.379},
               "rows": rows}
    V.write_json(V.OUT_ROOT / "canary_baselines.json", payload)
    V.write_csv(V.OUT_ROOT / "canary_baseline_rows.csv", rows)
    V.write_md(V.OUT_ROOT / "canary_baselines.md", [
        "# Canary Baselines (Part D)", "", V.status_line("CANARY_BASELINE_VERDICT", verdict),
        "Pre-training fixed baselines on the 610-group canary. positive_oracle@K (alignment = preference accuracy). "
        "Deltas are prev-SFT − base; labeled FLAG_* (not MEASURED_*) until paired CIs exclude zero.", "",
        *V.md_table(rows, ["domain", "n_base", "base", "prev_sft", "delta", "flag", "base_parse", "base_div", "small_n"]),
        "", "External stack reference (from v1 Experiment-1): CoreContent_v2 within DualAnchor survivors 0.658 vs "
        "DualAnchor forced-top1 0.379; feature extraction on the canary deferred to Part L.",
        "", "Generation: batched (num_return_sequences=K) for logic/reasoning/coding, single-seq for math (KV guard); "
        "resumable per-variant; STOP-pausable.",
    ])
    V.set_stage("D_canary_baselines", verdict, {"base_done": base is not None, "prev_sft_done": prev is not None})
    V.prog("D_canary_baselines", {"verdict": verdict, "rows": rows})
    print(V.status_line("CANARY_BASELINE_VERDICT", verdict))
    for r in rows:
        print(f"  {r['domain']:10} n={r['n_base']} base={r['base']} prev_sft={r['prev_sft']} Δ={r['delta']} {r['flag']}")


def main() -> int:
    V.ensure_dirs()
    if VARIANT == "aggregate":
        aggregate()
    else:
        generate_variant(VARIANT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
