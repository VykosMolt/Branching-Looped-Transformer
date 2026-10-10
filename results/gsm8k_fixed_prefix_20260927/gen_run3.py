"""Run 3 generation: GSM8K, Ouro-RLTT, fresh problems, with exact token IDs saved. See PLAN.md.

Mirrors the published GSM8K generator (proto_introspection_within_domain_recapture.py): same
prompt, model, sampling (k=4, T=0.7, top-p 0.95), labels, strict pre-answer cut and features, and
shortcut features, applied to the first 224 generated tokens, which was that run's whole budget.
Generation continues to at most 448 tokens so that longer solutions still get labels. A batch stops
only after step 224 and once every candidate has written a FINAL ANSWER line with a number (plus 8
tokens), so the first 224 tokens are exactly what the published protocol saw, and nothing read or
labelled depends on the text after the stop.

Adds what the published run lacked:
  - prompt and generated token IDs, the sampling log-probability of every token, and its
    teacher-forced log-probability and entropy;
  - pooled features at fixed prefixes of N generated tokens, from one causal forward pass;
  - the label-free own-answer cut: the published cut with the candidate's own answer in place of
    the reference answer.
Writes resumable shards of 100 problems into this directory.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from pathlib import Path

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (ROOT, ROOT / "shared", ROOT / "shared/utilities/tests/manual", HERE):
    sys.path.insert(0, str(p))

from answer_detector_gsm8k import first_commitment, value  # noqa: E402
from bg_steering_suite_lib import evaluate_output, gsm_prompt  # noqa: E402
from bg_v2_overnight_horizon_generate import _sid, split_for  # noqa: E402
from proto_introspection_within_domain_recapture import logprob_entropy, strict_preanswer  # noqa: E402
from src.evaluator.bg_transformer_features import (  # noqa: E402
    BGTransformerFeatureExtractor, _LoopLayerCapture, _masked_mean_pool, TAP_LAYERS,
    LAYER_TO_CAPTURE_INDEX, NUM_LOOPS,
)

SEED = 20260927
K, MAX_NEW, OLD_BUDGET = 4, 448, 224
N_TASKS, SHARD, OLD_N = 600, 100, 170   # the published run used test problems 0..169
PREFIX_GRID = (0, 16, 32, 48, 64, 96, 128, 160, 192, 256)
CHECK_EVERY = 25
DATA = ROOT / "elastic_reasoner/data/gsm8k/main/test-00000-of-00001.parquet"
DATA_SHA256 = "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"
COMMIT_LINE = re.compile(r"FINAL\s*ANSWER\s*[:\-]?\s*[:\*\(\[]*\s*[-+]?\$?\s?\d", re.I)


def select_tasks() -> list[dict]:
    """GSM8K test problems 170..1318 in hash order. The first N_TASKS are the cohort; the ones
    after it are used only for smoke tests."""
    import pyarrow.parquet as pq
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == DATA_SHA256, "GSM8K test file changed"
    rows = pq.read_table(DATA).to_pylist()
    assert len(rows) == 1319
    tasks = []
    for i in range(OLD_N, len(rows)):
        sol, ans = rows[i]["answer"].split("####")
        tasks.append({"task_id": f"gsm8k/{i}", "dataset_index": i, "question": rows[i]["question"],
                      "gold": ans.strip().replace(",", ""),
                      "ref_steps": sum(1 for line in sol.splitlines() if line.strip())})
    tasks.sort(key=lambda t: hashlib.sha256(t["task_id"].encode()).hexdigest())
    return tasks


def canon(s: str | None) -> str | None:
    """An answer in the format of GSM8K reference answers ('$1,250.00' -> '1250')."""
    v = value(s)
    if v is None:
        return None
    return str(v.numerator) if v.denominator == 1 else format(float(v), "g")


class CommitStop(StoppingCriteria):
    """Stop the batch once past `min_steps` and every candidate has written a FINAL ANSWER line
    with a number, plus `extra` tokens."""

    def __init__(self, tok, plen: int, k: int, min_steps: int, extra: int = 8):
        self.tok, self.plen, self.min_steps, self.extra = tok, plen, min_steps, extra
        self.step_at: list[int | None] = [None] * k

    def __call__(self, input_ids, scores, **kwargs) -> bool:  # noqa: ANN001
        cur = input_ids.shape[1] - self.plen
        done = cur >= self.min_steps
        for i in range(input_ids.shape[0]):
            if self.step_at[i] is None and COMMIT_LINE.search(
                    self.tok.decode(input_ids[i, self.plen:], skip_special_tokens=True)):
                self.step_at[i] = cur
            if self.step_at[i] is None or cur - self.step_at[i] < self.extra:
                done = False
        return done


def forward_all(extractor, ids: list[int], plen: int, cuts):
    """One causal forward over prompt + generation. Returns pooled [len(cuts), 3, 4, 2048] features
    for the prompt plus the first n generated tokens, and the teacher-forced log-probability and
    entropy of every generated token (as in the published `logprob_entropy`, which also passes no
    attention mask; a mask of ones selects another kernel and moves these by up to 0.006)."""
    x = torch.tensor([ids], device=extractor.device)
    cap = _LoopLayerCapture()
    with torch.inference_mode(), extractor._capture_hooks(cap):
        out = extractor.model(input_ids=x, use_cache=False)  # no mask, as in the published logprob_entropy
    extractor._validate_capture(cap)
    tf_lp, tf_ent = [], []
    if len(ids) > plen:
        logp = torch.log_softmax(out.logits[0, plen - 1:-1].float(), dim=-1)  # row j predicts ids[plen + j]
        tgt = torch.tensor(ids[plen:], device=logp.device)
        tf_lp = logp.gather(1, tgt[:, None])[:, 0].cpu().tolist()
        tf_ent = (-(logp.exp() * logp).sum(-1)).cpu().tolist()
        del logp
    del out
    states = {47: cap.boundary}
    for layer, idx in LAYER_TO_CAPTURE_INDEX.items():
        states[layer] = cap.inter[idx]
    feats = torch.zeros(len(cuts), len(TAP_LAYERS), NUM_LOOPS, states[47][0].shape[-1])
    for ci, n in enumerate(cuts):
        m = torch.zeros(1, len(ids))
        m[0, : plen + min(n, len(ids) - plen)] = 1.0
        for li, layer in enumerate(TAP_LAYERS):
            for u in range(NUM_LOOPS):
                feats[ci, li, u] = _masked_mean_pool(states[layer][u], m)
    # the state at the last prompt token, before any generated token (for the prompt-only control)
    pend = torch.stack([torch.stack([states[layer][u][0, plen - 1].float().cpu() for u in range(NUM_LOOPS)])
                        for layer in TAP_LAYERS])
    del cap
    return feats, tf_lp, tf_ent, pend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks", type=int, default=N_TASKS)
    ap.add_argument("--shard", type=int, default=SHARD)
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="3 problems from outside the cohort, all checks, into smoke/")
    ap.add_argument("--short", action="store_true", help="with --smoke: 48/32-token budgets, for a CPU run")
    a = ap.parse_args()
    max_new, old_budget, check_every = MAX_NEW, OLD_BUDGET, CHECK_EVERY
    tasks_all = select_tasks()
    cohort = tasks_all[: a.n_tasks]
    if a.smoke:
        cohort, a.shard, a.out, check_every = tasks_all[N_TASKS: N_TASKS + 3], 3, HERE / "smoke", 1
        if a.short:
            max_new, old_budget = 48, 32
    a.out.mkdir(parents=True, exist_ok=True)

    extractor = BGTransformerFeatureExtractor(device=a.device, dtype="auto", force_all_loops=True)
    model, tok = extractor.model, extractor.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    for s0 in range(0, len(cohort), a.shard):
        path = a.out / f"run3_shard_{s0:03d}.pt"
        if path.exists():
            print(f"[run3] {path.name} exists, skipping", flush=True)
            continue
        tasks = cohort[s0: s0 + a.shard]
        t0, records, errors, checks, tf_checks = time.time(), [], [], [], []
        for ti, task in enumerate(tasks):
            try:
                prompt = gsm_prompt(task["question"])
                enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1536, padding=False)
                enc = {k: v.to(extractor.device) for k, v in enc.items()}
                prompt_ids = enc["input_ids"][0].tolist()
                plen = len(prompt_ids)
                seed_i = _sid(SEED, task["task_id"])
                torch.manual_seed(seed_i)
                torch.cuda.manual_seed_all(seed_i)
                stopper = StoppingCriteriaList([CommitStop(tok, plen, K, min_steps=old_budget)])
                with torch.inference_mode():
                    gen = model.generate(
                        **enc, max_new_tokens=max_new, do_sample=True, temperature=0.7, top_p=0.95,
                        num_return_sequences=K, pad_token_id=pad_id, eos_token_id=tok.eos_token_id,
                        use_cache=True, output_scores=True, return_dict_in_generate=True,
                        stopping_criteria=stopper,
                    )
                seqs = gen.sequences.tolist()
                scores = [sc.detach().to("cpu", dtype=torch.float32) for sc in gen.scores]
                del gen
                if extractor.device.type == "cuda":
                    torch.cuda.empty_cache()
                gold = task["gold"]
                gold_task = {"domain": "gsm8k", "answer_key": gold, "gold_answer": gold}

                for ci in range(K):
                    gen_ids = list(seqs[ci][plen:])
                    while gen_ids and gen_ids[-1] == pad_id:
                        gen_ids.pop()
                    logprobs = [torch.log_softmax(scores[st][ci], dim=-1)[gen_ids[st]].item()
                                for st in range(min(len(gen_ids), len(scores)))]
                    text = tok.decode(gen_ids, skip_special_tokens=True).strip()
                    own = first_commitment(text)
                    # published protocol, on the first 224 generated tokens
                    g224 = gen_ids[:old_budget]
                    text224 = tok.decode(g224, skip_special_tokens=True).strip()
                    ev = evaluate_output(gold_task, text224)
                    own224 = canon(ev.get("parsed_answer"))
                    old_cut, _, old_reason = strict_preanswer(tok, g224, gold, "gsm8k")
                    own_cut, _, own_reason = strict_preanswer(tok, g224, own224 or "", "gsm8k")

                    def retok(n: int):  # published construction: prompt + "\n" + stripped text before the cut
                        if n <= 0:
                            return None
                        pre = tok.decode(g224[:n], skip_special_tokens=True).strip()
                        return extractor.encode_text_to_pooled_features(f"{prompt}\n{pre}")
                    old_feats = retok(old_cut)
                    own_feats = old_feats if own_cut == old_cut else retok(own_cut)
                    prefix_feats, tf_lp, tf_ent, prompt_end = forward_all(extractor, prompt_ids + gen_ids, plen, PREFIX_GRID)
                    rec = {
                        **task, "split": split_for(task["task_id"]), "candidate_idx": ci, "seed": seed_i,
                        "prompt_tok": plen, "question_chars": len(task["question"]),
                        "prompt_ids": prompt_ids, "gen_ids": gen_ids, "n_gen_tok": len(gen_ids),
                        "hit_max_tokens": len(gen_ids) >= max_new, "generated_text": text,
                        "decode_roundtrip_ok": tok.decode(gen_ids, skip_special_tokens=True).strip() == text,
                        "gen_logprobs": logprobs, "tf_logprobs": tf_lp, "tf_entropies": tf_ent,
                        # primary labels: the first committed answer in the whole generation
                        "own_answer": own, "parseable": value(own) is not None,
                        "success": value(own) is not None and value(own) == value(gold),
                        # published protocol (first 224 tokens)
                        "text224": text224, "parsed224": ev.get("parsed_answer"), "success224": bool(ev.get("success")),
                        "old_cut": int(old_cut), "old_reason": old_reason, "old_feats": old_feats,
                        "own224": own224, "own_cut": int(own_cut), "own_reason": own_reason, "own_feats": own_feats,
                        # fixed prefixes
                        "prefix_grid": list(PREFIX_GRID), "prefix_features": prefix_feats,
                        "prefix_texts": [tok.decode(gen_ids[:n], skip_special_tokens=True) for n in PREFIX_GRID],
                        "prompt_end_state": prompt_end,
                    }
                    if ci == 0 and ti % check_every == 0:
                        # Checked on passes of 128+ generated tokens: below ~128 total tokens the GPU uses a
                        # different attention kernel for a short separate pass, which changes bf16 rounding
                        # (up to 0.03 at N=32 on the smoke run; see PLAN.md). The N=32 gap is recorded only.
                        n = min(128, len(gen_ids))
                        sep = forward_all(extractor, prompt_ids + gen_ids[:n], plen + n, [0])[0][0]
                        diff = float((sep - prefix_feats[PREFIX_GRID.index(128)]).abs().max())
                        rec["equivalence_check_n128_maxabs"] = diff
                        checks.append(diff)
                        n32 = min(32, len(gen_ids))
                        sep32 = forward_all(extractor, prompt_ids + gen_ids[:n32], plen + n32, [0])[0][0]
                        rec["short_pass_n32_maxabs"] = float((sep32 - prefix_feats[PREFIX_GRID.index(32)]).abs().max())
                        m = min(max(old_cut, 128), len(gen_ids))
                        ref = logprob_entropy(model, tok, prompt, m, gen_ids, extractor.device)
                        mine = (sum(tf_lp[:m]) / m, sum(tf_ent[:m]) / m, tf_ent[m - 1])
                        td = max(abs(x - y) for x, y in zip(mine, (ref["mean_logprob"], ref["mean_entropy"], ref["last_entropy"])))
                        rec["tf_check_maxabs"] = td
                        tf_checks.append(td)
                    records.append(rec)
                del scores, seqs
                if extractor.device.type == "cuda":
                    torch.cuda.empty_cache()
                if (ti + 1) % 10 == 0 or ti == 0:
                    print(f"[run3] shard {s0}: {ti + 1}/{len(tasks)} tasks, {time.time() - t0:.0f}s", flush=True)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{task['task_id']}: {type(exc).__name__}: {exc}")
                print(f"[run3] ERROR {task['task_id']}: {exc}", flush=True)
                if extractor.device.type == "cuda":
                    torch.cuda.empty_cache()
        torch.save({"shard_start": s0, "n_tasks": len(tasks), "seed": SEED, "k": K, "max_new_tokens": max_new,
                    "old_budget": old_budget, "temperature": 0.7, "top_p": 0.95, "prefix_grid": list(PREFIX_GRID),
                    "model": "shared/models/ouro_rltt_local", "smoke": a.smoke,
                    "elapsed_seconds": round(time.time() - t0, 1), "equivalence_checks_maxabs": checks,
                    "tf_checks_maxabs": tf_checks, "errors": errors, "records": records}, path)
        print(f"[run3] wrote {path.name}: {len(records)} records, {len(errors)} errors, equivalence max "
              f"{max(checks) if checks else None}, tf max {max(tf_checks) if tf_checks else None}", flush=True)
    extractor.cleanup()


if __name__ == "__main__":
    main()
