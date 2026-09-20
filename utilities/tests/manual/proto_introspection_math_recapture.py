"""Hardened MATH (Level 1-2, numeric-answer) strict-preanswer recapture.

Second powered domain for the proto-introspection within-domain specificity audit.
Chosen because MATH L1-2 is hard enough to force REAL pre-answer reasoning (unlike
SVAMP/ARC, which the model front-loads) yet bounded enough to terminate.

Hardened against the failure modes found in pre-flight:
  * verifier parses LAST \\boxed{...} (balanced braces) FIRST, then FINAL ANSWER,
    then a bare number -- the GSM8K parser missed \\boxed and mislabeled ~60%.
  * gold restricted to NUMERIC \\boxed answers (Fraction compare) -> trustworthy labels.
  * TRUNCATED samples (no answer marker AND hit token budget) are DROPPED, not labeled.
  * strict pre-answer cut = text before min(first \\boxed, FINAL marker, gold-value
    occurrence) -> the answer cannot leak.

Frozen Ouro only. No training, no checkpoint changes. Labels are EXTERNAL (gold
\\boxed answer / numeric-exact); tap scores are never labels. Records match the
schema consumed by proto_introspection_within_domain_analysis.analyze_domain.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

import torch

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
BOXED_MARKER = re.compile(r"\\boxed")
SUBJECTS = ["algebra", "prealgebra", "number_theory", "counting_and_probability",
            "geometry", "intermediate_algebra", "precalculus"]

from bg_steering_suite_lib import OuroTextGenerator  # noqa: E402


# ----------------------------- answer parsing -----------------------------
def extract_boxed(s: str) -> str | None:
    """Content of the LAST \\boxed{...} with balanced braces, or None."""
    idxs = [m.start() for m in BOXED_MARKER.finditer(s)]
    if not idxs:
        return None
    i = idxs[-1] + len("\\boxed")
    while i < len(s) and s[i] != "{":
        i += 1
    if i >= len(s):
        return None
    depth = 0
    out: list[str] = []
    for j in range(i, len(s)):
        c = s[j]
        if c == "{":
            depth += 1
            if depth == 1:
                continue
        if c == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(c)
    return None


def as_number(x: str | None) -> Fraction | None:
    if x is None:
        return None
    s = x.strip().replace(",", "").replace("\\!", "").replace("\\,", "").replace(" ", "")
    s = s.replace("\\$", "").replace("$", "").replace("\\%", "").replace("%", "")
    s = s.strip(".")
    if not s:
        return None
    m = re.fullmatch(r"\\d?frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m:
        try:
            return Fraction(int(m.group(1)), int(m.group(2)))
        except Exception:
            return None
    m = re.fullmatch(r"(-?\d+)/(-?\d+)", s)
    if m:
        try:
            return Fraction(int(m.group(1)), int(m.group(2)))
        except Exception:
            return None
    try:
        return Fraction(s)
    except Exception:
        pass
    try:
        return Fraction(float(s)).limit_denominator(10**6)
    except Exception:
        return None


def parse_pred(text: str) -> tuple[Fraction | None, bool, str]:
    """Return (pred_number, has_explicit_answer, how). boxed-first, then FINAL ANSWER."""
    b = extract_boxed(text)
    nb = as_number(b)
    if b is not None and nb is not None:
        return nb, True, "boxed"
    m = re.findall(r"FINAL\s*ANSWER\s*:?\s*\$?\\?b?o?x?e?d?\{?([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)", text, flags=re.IGNORECASE)
    if m:
        return as_number(m[-1]), True, "final_marker"
    # explicit but non-numeric boxed -> has answer but unparseable as number
    if b is not None:
        return None, True, "boxed_nonnumeric"
    return None, False, "none"


def verify(text: str, gold: Fraction) -> dict:
    pred, has_ans, how = parse_pred(text)
    success = (pred is not None) and (gold is not None) and (pred == gold)
    return {"success": bool(success), "has_answer": has_ans, "parsed": pred is not None,
            "parsed_answer": str(pred) if pred is not None else None, "how": how}


# ----------------------------- data -----------------------------
def math_prompt(problem: str) -> str:
    return (
        "Solve the following math problem. Think step by step and show your reasoning "
        "first. Only AFTER your reasoning, give the final answer on its own line as "
        "\\boxed{your answer}.\n\n"
        f"Problem:\n{problem}"
    )


def load_math_tasks(n: int, seed: int, levels: set[str], subjects: list[str]) -> list[dict]:
    from datasets import load_dataset
    pool: list[dict] = []
    for cfg in subjects:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", cfg)
        except Exception:
            continue
        sp = "test" if "test" in ds else list(ds.keys())[0]
        for k, ex in enumerate(ds[sp]):
            if ex.get("level") not in levels:
                continue
            gb = extract_boxed(ex["solution"])
            gnum = as_number(gb)
            if gnum is None:
                continue
            pool.append({"task_id": f"math/{cfg}/{sp}/{k}", "domain": "math", "subject": cfg,
                         "level": ex["level"], "question": ex["problem"], "gold_boxed": gb,
                         "gold_num": gnum, "prompt": math_prompt(ex["problem"])})
    random.Random(seed).shuffle(pool)
    return pool[:n]


# ----------------------------- pre-answer cut -----------------------------
def strict_preanswer(tok, gen_ids: list[int], gold_boxed: str) -> tuple[int, str]:
    """n_preanswer_tokens = before min(first \\boxed, FINAL marker, standalone gold value)."""
    gold_pat = None
    gs = (gold_boxed or "").strip()
    if re.fullmatch(r"-?\d+", gs):
        try:
            gold_pat = re.compile(rf"(?<![\d.]){re.escape(gs)}(?![\d.])")
        except re.error:
            gold_pat = None
    text = ""
    for t in range(len(gen_ids)):
        new = text + tok.decode([gen_ids[t]], skip_special_tokens=True)
        mb = BOXED_MARKER.search(new)
        mf = FINAL_MARKER.search(new)
        mg = gold_pat.search(new) if gold_pat is not None else None
        hits = [(m.start(), name) for m, name in ((mb, "boxed"), (mf, "marker"), (mg, "gold")) if m]
        if hits:
            hits.sort()
            return t, f"cut_at_{hits[0][1]}"
        text = new
    return len(gen_ids), "no_answer"


# ----------------------------- logprob baseline -----------------------------
def logprob_entropy(model, tok, prompt: str, n_pre: int, gen_ids: list[int], device) -> dict:
    if n_pre <= 0:
        return {"mean_logprob": float("nan"), "mean_entropy": float("nan"), "last_entropy": float("nan"), "n": 0}
    p_ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
    pre_ids = gen_ids[:n_pre]
    full = torch.tensor([p_ids + pre_ids], device=device)
    P = len(p_ids)
    with torch.inference_mode():
        logits = model(full, use_cache=False).logits[0].float()
    lps, ents = [], []
    for j in range(P, P + n_pre):
        dist = torch.log_softmax(logits[j - 1], dim=-1)
        lps.append(dist[full[0, j].item()].item())
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
        g = out[i, P:].tolist()
        hit_budget = len(g) >= max_new
        while g and g[-1] == pad:
            g.pop()
        samples.append({"gen_ids": g, "text": tok.decode(g, skip_special_tokens=True).strip(),
                        "hit_budget": hit_budget})
    if gen.device.type == "cuda":
        torch.cuda.empty_cache()
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=220)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--budget-seconds", type=float, default=8400.0)
    ap.add_argument("--seed", type=int, default=20260617)
    ap.add_argument("--levels", type=str, default="Level 2,Level 3")
    ap.add_argument("--subjects", type=str,
                    default="algebra,prealgebra,number_theory,counting_and_probability,geometry")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dump", action="store_true", help="print full generations + parse details")
    args = ap.parse_args()
    if args.smoke:
        args.n, args.samples, args.budget_seconds = 10, 2, 1500.0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    levels = {l.strip() for l in args.levels.split(",") if l.strip()}
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    t0 = time.time()
    tasks = load_math_tasks(args.n, args.seed, levels, subjects)
    print(f"[math-recap] loaded {len(tasks)} numeric tasks; levels={sorted(levels)} subjects={subjects} "
          f"max_new={args.max_new} samples={args.samples}")
    gen = OuroTextGenerator(device="cuda", dtype="auto")
    model, tok, device = gen.model, gen.tokenizer, gen.device
    print(f"[math-recap] model loaded {time.time()-t0:.1f}s")

    recs, errors, dropped = [], [], 0
    dom_t0 = time.time()
    for ti, task in enumerate(tasks):
        if time.time() - t0 > args.budget_seconds:
            print(f"[math-recap] budget hit at {ti}")
            break
        try:
            prompt = task["prompt"]
            samples = batched_generate_ids(gen, prompt, args.samples, args.max_new, args.seed + ti)
            po_feat = gen.extractor.encode_text_to_pooled_features(prompt)
            sample_rows = []
            for si, s in enumerate(samples):
                ev = verify(s["text"], task["gold_num"])
                truncated = s["hit_budget"] and not ev["has_answer"]
                # keep ONLY samples with a trustworthy numeric prediction (drop truncated,
                # no-answer, and non-numeric \boxed -> label cannot be trusted)
                if truncated or not ev["has_answer"] or not ev["parsed"]:
                    dropped += 1
                    if args.dump:
                        print(f"  DROP task={task['task_id']} trunc={truncated} parsed={ev['parsed']} how={ev['how']} gen_tok={len(s['gen_ids'])}")
                    continue
                n_pre, reason = strict_preanswer(tok, s["gen_ids"], task["gold_boxed"])
                pre_ids = s["gen_ids"][:n_pre]
                pre_text = tok.decode(pre_ids, skip_special_tokens=True).strip() if pre_ids else ""
                row = {"sample": si, "correct": bool(ev["success"]), "parsed": bool(ev["parsed"]),
                       "n_pre_tok": int(n_pre), "preanswer_reason": reason, "gen_tok": len(s["gen_ids"]),
                       "has_preanswer": n_pre > 0, "how": ev["how"], "hit_budget": s["hit_budget"]}
                if n_pre > 0:
                    row["preanswer_feat"] = gen.extractor.encode_text_to_pooled_features(f"{prompt}\n{pre_text}")
                    row.update({f"lp_{kk}": vv for kk, vv in
                                logprob_entropy(model, tok, prompt, n_pre, s["gen_ids"], device).items()})
                sample_rows.append(row)
                if args.dump:
                    print(f"  task={task['task_id']} gold={task['gold_boxed']} pred={ev['parsed_answer']} "
                          f"correct={ev['success']} how={ev['how']} n_pre={n_pre}({reason}) gen_tok={len(s['gen_ids'])}")
                    print("    TEXT:", repr(s["text"][:500]))
            if not sample_rows:
                continue
            recs.append({"task_id": task["task_id"], "domain": "math", "subject": task["subject"],
                         "level": task["level"],
                         "prompt_tok": int(tok(prompt, return_tensors="pt").input_ids.shape[1]),
                         "question_chars": len(task["question"]), "prompt_only_feat": po_feat,
                         "n_correct": sum(r["correct"] for r in sample_rows), "n_samples": len(sample_rows),
                         "samples": sample_rows})
            if (ti + 1) % 20 == 0:
                el = time.time() - dom_t0
                nkept = sum(rr["n_samples"] for rr in recs)
                acc = sum(rr["n_correct"] for rr in recs) / max(nkept, 1)
                pre = [s["n_pre_tok"] for rr in recs for s in rr["samples"] if s["has_preanswer"]]
                med = sorted(pre)[len(pre) // 2] if pre else 0
                print(f"[math-recap] {ti+1}/{len(tasks)} tasks_kept={len(recs)} kept_samples={nkept} "
                      f"dropped={dropped} sampleAcc={acc:.3f} medPre={med} {el/(ti+1):.1f}s/task")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{task['task_id']}: {type(exc).__name__}: {exc}")
            print(f"[math-recap] ERR {task['task_id']}: {exc}")
            continue

    gen.cleanup()
    nkept = sum(rr["n_samples"] for rr in recs)
    pre = [s["n_pre_tok"] for rr in recs for s in rr["samples"] if s["has_preanswer"]]
    pre_sorted = sorted(pre)
    meta = {"n_tasks_kept": len(recs), "n_requested": len(tasks), "kept_samples": nkept,
            "dropped_samples": dropped, "errors": errors, "elapsed_s": round(time.time() - dom_t0, 1),
            "sample_acc": round(sum(rr["n_correct"] for rr in recs) / max(nkept, 1), 4),
            "preanswer_median_tok": (pre_sorted[len(pre_sorted) // 2] if pre_sorted else 0),
            "preanswer_min_tok": (pre_sorted[0] if pre_sorted else 0),
            "preanswer_p10_tok": (pre_sorted[len(pre_sorted) // 10] if pre_sorted else 0),
            "frac_pre_tok_ge_8": round(sum(1 for x in pre if x >= 8) / max(len(pre), 1), 3)}
    meta["levels"] = sorted(levels)
    meta["subjects"] = subjects
    payload = {"tag": "math_preanswer", "model": "shared/models/ouro_rltt_local",
               "tap_layers": [24, 36, 47], "num_loops": 4, "feature_shape": [3, 4, 2048],
               "samples_per_task": args.samples, "max_new": args.max_new, "seed": args.seed,
               "levels": sorted(levels), "subjects": subjects,
               "temperature": 0.7, "top_p": 0.95, "elapsed_seconds": round(time.time() - t0, 1),
               "meta": {"math": meta}, "records": {"math": recs}}
    suffix = "_smoke" if args.smoke else ""
    pt = OUT_DIR / f"math_recapture{suffix}.pt"
    torch.save(payload, pt)
    (OUT_DIR / f"math_recapture{suffix}_index.json").write_text(
        json.dumps({"tag": payload["tag"], "meta": payload["meta"], "max_new": args.max_new,
                    "elapsed_seconds": payload["elapsed_seconds"], "samples_per_task": args.samples},
                   indent=2, default=str) + "\n")
    print(f"[math-recap] wrote {pt} ({pt.stat().st_size/1e6:.1f} MB)")
    print(f"[math-recap] META: {json.dumps(meta, default=str)}")


if __name__ == "__main__":
    main()
