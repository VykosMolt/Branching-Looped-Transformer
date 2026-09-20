"""Powered within-domain strict-preanswer recapture for proto-introspection.

For a SINGLE domain at a time, generate K external-verified samples per task and
capture, per sample, a strict pre-answer hidden state plus a logprob/entropy
baseline -- so we can test (within-domain, no domain confound) whether hidden
states predict success beyond length/logprob shortcuts.

Strict pre-answer cut = text BEFORE min(FINAL-ANSWER marker, first occurrence of
the gold answer value). For GSM8K this prevents the answer number leaking via the
reasoning; for MCQ the model front-loads the letter so the cut is usually empty
and prompt_only carries the pre-answer load.

Frozen Ouro only. No training, no checkpoint changes. Labels are EXTERNAL verifier
results only (gold answer / numeric-exact / mcq-letter); tap scores are never labels.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MANUAL = PROJECT_ROOT / "shared/utilities/tests/manual"
if str(MANUAL) not in sys.path:
    sys.path.insert(0, str(MANUAL))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "shared/hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "shared/hf_cache/datasets"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

OUT_DIR = PROJECT_ROOT / "artifacts/reports/proto_introspection"
FINAL_MARKER = re.compile(r"FINAL\s*ANSWE", re.IGNORECASE)

from bg_steering_suite_lib import OuroTextGenerator, evaluate_output, gsm_prompt, mcq_prompt  # noqa: E402


def load_domain_tasks(domain: str, n: int) -> list[dict]:
    from datasets import load_dataset
    rows: list[dict] = []
    if domain == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        for i, ex in enumerate(ds):
            ans = ex["answer"].split("####")[-1].strip().replace(",", "")
            if not ans:
                continue
            rows.append({"task_id": f"gsm8k/{i}", "domain": "gsm8k", "question": ex["question"],
                         "answer_key": ans, "gold_answer": ans, "prompt": gsm_prompt(ex["question"])})
            if len(rows) >= n:
                break
    elif domain == "reasoning":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        for ex in ds:
            labels = ex["choices"]["label"]
            if labels != ["A", "B", "C", "D"]:
                continue  # keep clean 4-option A-D
            if str(ex["answerKey"]).upper() not in {"A", "B", "C", "D"}:
                continue
            opts = {l: t for l, t in zip(ex["choices"]["label"], ex["choices"]["text"])}
            rows.append({"task_id": f"ARC/{ex['id']}", "domain": "reasoning", "question": ex["question"],
                         "options": opts, "answer_key": str(ex["answerKey"]).upper(),
                         "prompt": mcq_prompt(ex["question"], opts)})
            if len(rows) >= n:
                break
    else:
        raise ValueError(domain)
    return rows


def strict_preanswer(tok, gen_ids: list[int], gold: str, domain: str) -> tuple[int, bool, str]:
    """Return (n_preanswer_tokens, leaked_flag, reason) by decoding incrementally
    and cutting at the first FINAL-ANSWER marker OR first standalone gold value."""
    gold_pat = None
    if domain == "gsm8k" and gold.strip():
        try:
            gold_pat = re.compile(rf"(?<![\d.]){re.escape(gold.strip())}(?![\d.])")
        except re.error:
            gold_pat = None
    text = ""
    cut = len(gen_ids)
    reason = "no_marker_no_gold"
    for t in range(len(gen_ids)):
        piece = tok.decode([gen_ids[t]], skip_special_tokens=True)
        new = text + piece
        m = FINAL_MARKER.search(new)
        g = gold_pat.search(new) if gold_pat is not None else None
        hit = None
        if m and g:
            hit = ("marker" if m.start() <= g.start() else "gold")
        elif m:
            hit = "marker"
        elif g:
            hit = "gold"
        if hit:
            cut = t  # exclude this token; everything strictly before is pre-answer
            reason = f"cut_at_{hit}"
            return cut, False, reason
        text = new
    return cut, False, reason  # whole generation is pre-answer (no answer emitted)


def logprob_entropy(model, tok, prompt: str, n_pre: int, gen_ids: list[int], device) -> dict:
    """Teacher-forced logprob + entropy over the first n_pre generated tokens."""
    if n_pre <= 0:
        return {"mean_logprob": float("nan"), "mean_entropy": float("nan"), "last_entropy": float("nan"), "n": 0}
    p_ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
    pre_ids = gen_ids[:n_pre]
    full = torch.tensor([p_ids + pre_ids], device=device)
    P = len(p_ids)
    with torch.inference_mode():
        logits = model(full, use_cache=False).logits[0].float()  # [T, V]
    lps, ents = [], []
    for j in range(P, P + n_pre):
        dist = torch.log_softmax(logits[j - 1], dim=-1)
        tokid = full[0, j].item()
        lps.append(dist[tokid].item())
        p = dist.exp()
        ents.append(float(-(p * dist).sum().item()))
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"mean_logprob": sum(lps) / len(lps), "mean_entropy": sum(ents) / len(ents),
            "last_entropy": ents[-1], "n": n_pre}


def batched_generate_ids(gen, prompt: str, k: int, max_new: int, seed: int):
    tok = gen.tokenizer
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1536).to(gen.device)
    P = int(enc["input_ids"].shape[1])
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    torch.manual_seed(seed)
    if gen.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        out = gen.model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.7,
                                 top_p=0.95, num_return_sequences=k, pad_token_id=pad,
                                 eos_token_id=tok.eos_token_id, use_cache=True)
    samples = []
    for i in range(out.shape[0]):
        gen_ids = out[i, P:].tolist()
        # strip trailing pads
        while gen_ids and gen_ids[-1] == pad:
            gen_ids.pop()
        samples.append({"gen_ids": gen_ids, "text": tok.decode(gen_ids, skip_special_tokens=True).strip()})
    if gen.device.type == "cuda":
        torch.cuda.empty_cache()
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", type=str, default="gsm8k,reasoning")
    ap.add_argument("--n", type=int, default=180)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--budget-seconds", type=float, default=6300.0)
    ap.add_argument("--seed", type=int, default=20260617)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n, args.samples, args.budget_seconds = 3, 2, 600.0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    per_domain_n = {"gsm8k": args.n, "reasoning": max(args.n - 40, 120) if not args.smoke else args.n}

    t0 = time.time()
    gen = OuroTextGenerator(device="cuda", dtype="auto")
    model, tok, device = gen.model, gen.tokenizer, gen.device
    print(f"[wd-recap] model loaded {time.time()-t0:.1f}s")

    all_records: dict[str, list] = {}
    meta: dict[str, dict] = {}
    for domain in domains:
        if time.time() - t0 > args.budget_seconds:
            print(f"[wd-recap] budget hit before {domain}")
            break
        n = per_domain_n.get(domain, args.n)
        tasks = load_domain_tasks(domain, n)
        max_new = 224 if domain == "gsm8k" else 192
        print(f"[wd-recap] domain={domain} tasks={len(tasks)} max_new={max_new}")
        recs, errors = [], []
        dom_t0 = time.time()
        for ti, task in enumerate(tasks):
            if time.time() - t0 > args.budget_seconds:
                print(f"[wd-recap] budget hit during {domain} at {ti}")
                break
            try:
                prompt = task["prompt"]
                samples = batched_generate_ids(gen, prompt, args.samples, max_new, args.seed + ti)
                # prompt_only feature (once per task)
                po_feat = gen.extractor.encode_text_to_pooled_features(prompt)
                sample_rows = []
                for si, s in enumerate(samples):
                    ev = evaluate_output(task, s["text"])
                    correct = bool(ev.get("success"))
                    n_pre, leaked, reason = strict_preanswer(tok, s["gen_ids"], task.get("gold_answer", ""), domain)
                    pre_ids = s["gen_ids"][:n_pre]
                    pre_text = tok.decode(pre_ids, skip_special_tokens=True).strip() if pre_ids else ""
                    row = {"sample": si, "correct": correct, "parsed": bool(ev.get("parsed")),
                           "n_pre_tok": int(n_pre), "preanswer_reason": reason, "gen_tok": len(s["gen_ids"]),
                           "has_preanswer": n_pre > 0}
                    if n_pre > 0:
                        row["preanswer_feat"] = gen.extractor.encode_text_to_pooled_features(f"{prompt}\n{pre_text}")
                        row.update({f"lp_{kk}": vv for kk, vv in
                                    logprob_entropy(model, tok, prompt, n_pre, s["gen_ids"], device).items()})
                    sample_rows.append(row)
                recs.append({"task_id": task["task_id"], "domain": domain, "prompt_tok": int(tok(prompt, return_tensors="pt").input_ids.shape[1]),
                             "question_chars": len(task["question"]), "prompt_only_feat": po_feat,
                             "n_correct": sum(r["correct"] for r in sample_rows), "n_samples": len(sample_rows),
                             "samples": sample_rows})
                if (ti + 1) % 20 == 0:
                    el = time.time() - dom_t0
                    acc = sum(rr["n_correct"] for rr in recs) / max(sum(rr["n_samples"] for rr in recs), 1)
                    print(f"[wd-recap] {domain} {ti+1}/{len(tasks)} sampleAcc={acc:.3f} {el/ (ti+1):.1f}s/task")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{task['task_id']}: {type(exc).__name__}: {exc}")
                print(f"[wd-recap] ERR {task['task_id']}: {exc}")
                continue
        all_records[domain] = recs
        meta[domain] = {"n_tasks": len(recs), "n_requested": len(tasks), "errors": errors,
                        "elapsed_s": round(time.time() - dom_t0, 1),
                        "sample_acc": round(sum(rr["n_correct"] for rr in recs) / max(sum(rr["n_samples"] for rr in recs), 1), 4),
                        "preanswer_coverage": round(sum(1 for rr in recs for s in rr["samples"] if s["has_preanswer"]) /
                                                    max(sum(rr["n_samples"] for rr in recs), 1), 4)}
        print(f"[wd-recap] {domain} done: {meta[domain]}")

    gen.cleanup()
    payload = {"tag": "within_domain_preanswer", "model": "shared/models/ouro_rltt_local",
               "tap_layers": [24, 36, 47], "num_loops": 4, "feature_shape": [3, 4, 2048],
               "samples_per_task": args.samples, "seed": args.seed, "temperature": 0.7, "top_p": 0.95,
               "elapsed_seconds": round(time.time() - t0, 1), "meta": meta, "records": all_records}
    suffix = "_smoke" if args.smoke else ""
    pt = OUT_DIR / f"within_domain_recapture{suffix}.pt"
    torch.save(payload, pt)
    light = {"tag": payload["tag"], "meta": meta, "elapsed_seconds": payload["elapsed_seconds"],
             "samples_per_task": args.samples}
    (OUT_DIR / f"within_domain_recapture{suffix}_index.json").write_text(json.dumps(light, indent=2, default=str) + "\n")
    print(f"[wd-recap] wrote {pt} ({pt.stat().st_size/1e6:.1f} MB); meta={json.dumps(meta, default=str)}")


if __name__ == "__main__":
    main()
