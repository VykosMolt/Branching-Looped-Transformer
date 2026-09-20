"""Control A: strict pre-answer / no-leakage recapture for proto-introspection.

Read-only w.r.t. model weights. Captures pooled BG hidden-state features at
several capture cuts that range from strictly pre-answer (prompt-only, zero
generated tokens) to fully leaked (prompt + full answer), and labels each task
by an EXTERNAL verifier over K sampled continuations.

Cuts per task:
  prompt_only : features over the QUESTION ONLY (0 generated tokens -> no leak possible)
  gen16       : prompt + first 16 generated tokens of sample 0 (verified pre-answer)
  gen32       : prompt + first 32 generated tokens of sample 0 (verified pre-answer)
  preanswer   : prompt + reasoning up to just before the FINAL ANSWER marker (sample 0)
  full        : prompt + full sample 0 (leaked comparison only)

Labels (external verifier, gold answer key only -- never tap/evaluator scores):
  per-sample success over K samples -> mean_correct, maj_correct, any_correct (oracle)
  sample0_correct (for the generated-cut trajectory framing)

Outputs a feature .pt and an index .json under the proto_introspection report dir.
Nothing here trains or modifies the Ouro checkpoint.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MANUAL = PROJECT_ROOT / "shared/utilities/tests/manual"
if str(MANUAL) not in sys.path:
    sys.path.insert(0, str(MANUAL))

OUT_DIR = PROJECT_ROOT / "artifacts/reports/proto_introspection"
FINAL_MARKER = re.compile(r"FINAL\s*ANSWE", re.IGNORECASE)

from bg_steering_suite_lib import (  # noqa: E402
    OuroTextGenerator,
    evaluate_output,
    load_gsm8k_tasks,
    load_reasoning_tasks,
    load_science_tasks,
    mcq_prompt,
    gsm_prompt,
)


def gen_prompt(task: dict) -> str:
    dom = task["domain"]
    if dom in {"reasoning", "science"}:
        return mcq_prompt(task["question"], task["options"])
    if dom == "gsm8k":
        return gsm_prompt(task["question"])
    return str(task.get("prompt") or task.get("question") or "")


def norm_reasoning(item: dict) -> dict | None:
    ans = str(item.get("answer") or item.get("answer_key") or "").upper()
    opts = item.get("options") or {}
    q = str(item.get("question") or "").strip()
    tid = str(item.get("task_id") or "")
    if not (tid and q and ans and opts):
        return None
    return {"task_id": tid, "domain": "reasoning", "question": q, "options": opts,
            "answer_key": ans, "gold_answer": ans}


def norm_science(item: dict) -> dict | None:
    ans = str(item.get("answer_key") or item.get("answer") or "").upper()
    opts = item.get("options") or {}
    q = str(item.get("question") or "").strip()
    tid = str(item.get("task_id") or "")
    if not (tid and q and ans and opts):
        return None
    return {"task_id": tid, "domain": "science", "question": q, "options": opts,
            "answer_key": ans, "gold_answer": ans}


def norm_gsm(item: dict, idx: int) -> dict | None:
    q = str(item.get("question") or "").strip()
    ans = str(item.get("gold_answer") or item.get("answer_key") or "").strip()
    di = item.get("dataset_index", item.get("problem_id", idx))
    if not (q and ans):
        return None
    return {"task_id": f"gsm8k/{di}", "domain": "gsm8k", "question": q,
            "answer_key": ans, "gold_answer": ans}


def load_tasks(n_reasoning: int, n_science: int, n_gsm: int) -> list[dict]:
    rows: list[dict] = []
    rs = [norm_reasoning(r) for r in load_reasoning_tasks()]
    rows += [r for r in rs if r][:n_reasoning]
    sc = [norm_science(r) for r in load_science_tasks()]
    rows += [r for r in sc if r][:n_science]
    gs = [norm_gsm(r, i) for i, r in enumerate(load_gsm8k_tasks())]
    rows += [r for r in gs if r][:n_gsm]
    return rows


def batched_samples(gen: OuroTextGenerator, prompt: str, k: int, max_new: int,
                    seed: int, temperature: float = 0.7, top_p: float = 0.95) -> list[str]:
    """Generate k samples for a prompt in one batched call (num_return_sequences=k)."""
    tok = gen.tokenizer
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1536, padding=False)
    enc = {kk: vv.to(gen.device) for kk, vv in enc.items()}
    plen = int(enc["input_ids"].shape[1])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    torch.manual_seed(seed)
    if gen.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        out = gen.model.generate(
            **enc, max_new_tokens=max_new, do_sample=True, temperature=temperature,
            top_p=top_p, num_return_sequences=k, pad_token_id=pad_id,
            eos_token_id=tok.eos_token_id, use_cache=True,
        )
    texts = []
    for i in range(out.shape[0]):
        new_ids = out[i, plen:]
        texts.append(tok.decode(new_ids, skip_special_tokens=True).strip())
    if gen.device.type == "cuda":
        torch.cuda.empty_cache()
    return texts


def first_n_tokens_text(tok, text: str, n: int) -> str:
    ids = tok.encode(text, add_special_tokens=False)
    return tok.decode(ids[:n], skip_special_tokens=True).strip()


def preanswer_prefix(text: str) -> str:
    m = FINAL_MARKER.search(text)
    return text[: m.start()].strip() if m else text.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reasoning", type=int, default=40)
    ap.add_argument("--science", type=int, default=40)
    ap.add_argument("--gsm", type=int, default=40)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--budget-seconds", type=float, default=4200.0)
    ap.add_argument("--seed", type=int, default=20260617)
    ap.add_argument("--tag", type=str, default="preanswer_recapture")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.reasoning, args.science, args.gsm, args.samples = 2, 1, 1, 2
        args.budget_seconds = 600.0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(args.reasoning, args.science, args.gsm)
    print(f"[recapture] loaded {len(tasks)} tasks; samples/task={args.samples}; budget={args.budget_seconds}s")

    t0 = time.time()
    gen = OuroTextGenerator(device="cuda", dtype="auto")
    extractor = gen.extractor
    tok = gen.tokenizer
    print(f"[recapture] model loaded in {time.time()-t0:.1f}s")

    records: list[dict] = []
    errors: list[str] = []
    per_task_timing: list[float] = []
    budget_hit = False

    for ti, task in enumerate(tasks):
        if time.time() - t0 > args.budget_seconds:
            budget_hit = True
            print(f"[recapture] budget hit after {ti} tasks")
            break
        tt = time.time()
        dom = task["domain"]
        prompt = gen_prompt(task)
        max_new = 192 if dom in {"reasoning", "science"} else 224
        try:
            # ---- labels: K external-verified samples ----
            samples = batched_samples(gen, prompt, args.samples, max_new, args.seed + ti)
            evals = [evaluate_output(task, s) for s in samples]
            corrects = [bool(e.get("success")) for e in evals]
            parsed_flags = [bool(e.get("parsed")) for e in evals]
            mean_correct = sum(corrects) / max(len(corrects), 1)
            maj_correct = mean_correct >= 0.5
            any_correct = any(corrects)
            s0 = samples[0]
            s0_correct = corrects[0]

            # ---- capture cuts ----
            cuts: dict[str, str] = {"prompt_only": prompt}
            g16 = first_n_tokens_text(tok, s0, 16)
            g32 = first_n_tokens_text(tok, s0, 32)
            pre = preanswer_prefix(s0)
            if g16:
                cuts["gen16"] = f"{prompt}\n{g16}"
            if g32:
                cuts["gen32"] = f"{prompt}\n{g32}"
            if pre:
                cuts["preanswer"] = f"{prompt}\n{pre}"
            cuts["full"] = f"{prompt}\n{s0}"

            cut_feats: dict[str, torch.Tensor] = {}
            cut_leak: dict[str, bool] = {}
            for name, text in cuts.items():
                feats = extractor.encode_text_to_pooled_features(text)  # [3,4,2048] cpu fp32
                cut_feats[name] = feats
                # leak flag: did the FINAL ANSWER marker already appear in this cut's generated part?
                gen_part = text[len(prompt):]
                cut_leak[name] = bool(FINAL_MARKER.search(gen_part)) if name != "prompt_only" else False

            records.append({
                "task_id": task["task_id"], "domain": dom,
                "mean_correct": mean_correct, "maj_correct": bool(maj_correct),
                "any_correct": bool(any_correct), "n_samples": len(samples),
                "n_correct": sum(corrects), "parse_rate": sum(parsed_flags) / max(len(parsed_flags), 1),
                "s0_correct": bool(s0_correct),
                "question_chars": len(task["question"]),
                "prompt_chars": len(prompt),
                "prompt_tok": int(tok(prompt, return_tensors="pt")["input_ids"].shape[1]),
                "s0_chars": len(s0), "s0_tok": len(tok.encode(s0, add_special_tokens=False)),
                "cut_features": {k: v for k, v in cut_feats.items()},
                "cut_leak": cut_leak,
                "cuts_present": list(cut_feats.keys()),
            })
            per_task_timing.append(time.time() - tt)
            if (ti + 1) % 10 == 0 or args.smoke:
                avg = sum(per_task_timing) / len(per_task_timing)
                print(f"[recapture] {ti+1}/{len(tasks)} dom={dom} mean_corr={mean_correct:.2f} "
                      f"cuts={list(cut_feats.keys())} leak={cut_leak} avg={avg:.1f}s/task")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{task['task_id']}: {type(exc).__name__}: {exc}")
            print(f"[recapture] ERROR {task['task_id']}: {exc}")
            continue

    gen.cleanup()

    payload = {
        "tag": args.tag,
        "model": "shared/models/ouro_rltt_local",
        "tap_layers": [24, 36, 47], "num_loops": 4, "force_all_loops": True,
        "feature_shape": [3, 4, 2048],
        "samples_per_task": args.samples, "seed": args.seed,
        "max_new_tokens": {"reasoning": 192, "science": 192, "gsm8k": 224},
        "temperature": 0.7, "top_p": 0.95,
        "n_tasks_requested": len(tasks), "n_tasks_captured": len(records),
        "budget_seconds": args.budget_seconds, "budget_hit": budget_hit,
        "elapsed_seconds": round(time.time() - t0, 1),
        "errors": errors,
        "counts_by_domain": {d: sum(1 for r in records if r["domain"] == d)
                             for d in {r["domain"] for r in records}},
        "records": records,
    }
    suffix = "_smoke" if args.smoke else ""
    pt_path = OUT_DIR / f"preanswer_recapture{suffix}.pt"
    idx_path = OUT_DIR / f"preanswer_recapture{suffix}_index.json"
    torch.save(payload, pt_path)
    light = {k: v for k, v in payload.items() if k != "records"}
    light["records_summary"] = [
        {kk: r[kk] for kk in ("task_id", "domain", "mean_correct", "maj_correct", "any_correct",
                              "n_correct", "n_samples", "s0_correct", "cuts_present", "cut_leak",
                              "question_chars", "prompt_tok", "s0_tok")}
        for r in records
    ]
    idx_path.write_text(json.dumps(light, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"[recapture] wrote {pt_path} ({pt_path.stat().st_size/1e6:.1f} MB) and {idx_path}")
    print(f"[recapture] captured {len(records)} tasks in {payload['elapsed_seconds']}s; errors={len(errors)}")


if __name__ == "__main__":
    main()
