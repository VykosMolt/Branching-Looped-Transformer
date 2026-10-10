"""Run 1 generation: Horizon Logic, Ouro-RLTT, fresh slice, with exact token IDs saved.

Mirrors the published generator (bg_v2_overnight_horizon_generate.py, used for v2 and v3):
same task pool and hash ordering, prompt, sampling (k=4, T=0.7, top-p 0.95, max 448 new
tokens, FinalAnswerStop), seed derivation, labels and original-protocol features. Adds what
the July 29 audit found missing:

  - prompt and generated token IDs, and the log-probability of every sampled token;
  - pooled features at fixed generated-token prefixes N in {0,16,32,48,64,96,128}, computed
    from the exact IDs in one causal forward pass over prompt + generation (states at a
    position depend only on earlier tokens, so pooling the first plen+N positions equals a
    separate forward on the truncated IDs); a subsample checks this against separate passes.

Writes resumable shards of 100 tasks into this directory. See PLAN.md.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import torch
from transformers import StoppingCriteriaList

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (ROOT, ROOT / "shared", ROOT / "shared/utilities/tests/manual"):
    sys.path.insert(0, str(p))

from bg_steering_suite_lib import mcq_prompt, parse_mcq_answer  # noqa: E402
from bg_v2_overnight_horizon_generate import (  # noqa: E402
    _sid, split_for, load_task_pool, options_dict, preanswer_prefix, FinalAnswerStop, token_cut_index,
)
from src.evaluator.bg_transformer_features import (  # noqa: E402
    BGTransformerFeatureExtractor, _LoopLayerCapture, _masked_mean_pool, TAP_LAYERS,
    LAYER_TO_CAPTURE_INDEX, NUM_LOOPS,
)

SEED = 20260724            # published seed rule, applied to new tasks
K, MAX_NEW = 4, 448
OFFSET, N_TASKS, SHARD = 2400, 600, 100
PREFIX_GRID = (0, 16, 32, 48, 64, 96, 128)
CHECK_EVERY = 25           # separate-forward equivalence check on one candidate per 25 tasks


def multi_cut_features(extractor, ids: list[int], plen: int, cuts) -> torch.Tensor:
    """Pooled [len(cuts), 3, 4, 2048] features for prompt + first N generated tokens, one forward."""
    x = torch.tensor([ids], device=extractor.device)
    mask = torch.ones_like(x)
    cap = _LoopLayerCapture()
    with torch.inference_mode(), extractor._capture_hooks(cap):
        extractor.model(input_ids=x, attention_mask=mask, use_cache=False)
    extractor._validate_capture(cap)
    states = {47: cap.boundary}
    for layer, idx in LAYER_TO_CAPTURE_INDEX.items():
        states[layer] = cap.inter[idx]
    out = torch.zeros(len(cuts), len(TAP_LAYERS), NUM_LOOPS, states[47][0].shape[-1])
    for ci, n in enumerate(cuts):
        m = torch.zeros(1, len(ids))
        m[0, : plen + min(n, len(ids) - plen)] = 1.0
        for li, layer in enumerate(TAP_LAYERS):
            for u in range(NUM_LOOPS):
                out[ci, li, u] = _masked_mean_pool(states[layer][u], m)
    del cap
    return out


def separate_forward(extractor, ids: list[int]) -> torch.Tensor:
    return multi_cut_features(extractor, ids, len(ids), [0])[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", type=int, default=OFFSET)
    ap.add_argument("--n-tasks", type=int, default=N_TASKS)
    ap.add_argument("--shard", type=int, default=SHARD)
    ap.add_argument("--out", type=Path, default=HERE)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    pool = load_task_pool()
    pool.sort(key=lambda d: hashlib.sha256(d["task_uid"].encode()).hexdigest())
    extractor = BGTransformerFeatureExtractor(device="cuda", dtype="auto", force_all_loops=True)
    model, tok = extractor.model, extractor.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    for s0 in range(a.offset, a.offset + a.n_tasks, a.shard):
        path = a.out / f"run1_shard_off{s0}.pt"
        if path.exists():
            print(f"[run1] {path.name} exists, skipping", flush=True)
            continue
        tasks = pool[s0: min(s0 + a.shard, a.offset + a.n_tasks)]
        t0, records, errors, checks = time.time(), [], [], []
        for ti, task in enumerate(tasks):
            try:
                opts = options_dict(task["options"])
                gold_letter = chr(65 + int(task["answer_key"]))
                prompt = mcq_prompt(task["task_prompt"], opts)
                enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1536, padding=False)
                enc = {k: v.to(extractor.device) for k, v in enc.items()}
                prompt_ids = enc["input_ids"][0].tolist()
                plen = len(prompt_ids)
                seed_i = _sid(SEED, task["task_uid"])
                torch.manual_seed(seed_i)
                torch.cuda.manual_seed_all(seed_i)
                stopper = StoppingCriteriaList([FinalAnswerStop(tok, plen, K)])
                with torch.inference_mode():
                    gen = model.generate(
                        **enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=0.7, top_p=0.95,
                        num_return_sequences=K, pad_token_id=pad_id, eos_token_id=tok.eos_token_id,
                        use_cache=True, output_scores=True, return_dict_in_generate=True,
                        stopping_criteria=stopper,
                    )
                seqs = gen.sequences.tolist()
                scores = [sc.detach().to("cpu", dtype=torch.float32) for sc in gen.scores]
                del gen
                torch.cuda.empty_cache()
                prompt_feats = extractor.encode_text_to_pooled_features(prompt)
                split = split_for(task["task_uid"])

                for ci in range(K):
                    raw = seqs[ci][plen:]
                    gen_ids = list(raw)
                    while gen_ids and gen_ids[-1] == pad_id:
                        gen_ids.pop()
                    logprobs = [torch.log_softmax(scores[st][ci], dim=-1)[gen_ids[st]].item()
                                for st in range(min(len(gen_ids), len(scores)))]
                    text = tok.decode(gen_ids, skip_special_tokens=True).strip()
                    pre_text, found = preanswer_prefix(text)
                    cut_k = token_cut_index(tok, gen_ids, len(pre_text)) if found else len(gen_ids)
                    cut_k = max(1, min(cut_k, len(gen_ids)))
                    lp_pre = logprobs[:cut_k]
                    parsed_raw = parse_mcq_answer(text, opts)
                    parsed = parsed_raw if found else None
                    preanswer_feats = extractor.encode_text_to_pooled_features(
                        f"{prompt}\n{pre_text}" if pre_text else prompt)
                    prefix_feats = multi_cut_features(extractor, prompt_ids + gen_ids, plen, PREFIX_GRID)
                    rec = {
                        "task_uid": task["task_uid"], "split": split, "proof_depth": task["proof_depth"],
                        "category": task["category"], "candidate_idx": ci, "gold_answer": task["gold_answer"],
                        "gold_letter": gold_letter, "options": opts, "task_prompt": task["task_prompt"],
                        "generated_text": text, "n_gen_tok": len(gen_ids), "hit_max_tokens": len(gen_ids) >= MAX_NEW,
                        "found_final_marker": bool(found), "parsed_answer": parsed,
                        "success": bool(found and parsed == gold_letter), "malformed": bool(not found or parsed is None),
                        "n_pre_tok": int(cut_k),
                        "mean_logprob_pre": sum(lp_pre) / len(lp_pre) if lp_pre else None,
                        "min_logprob_pre": min(lp_pre) if lp_pre else None,
                        "prompt_tok": plen, "prompt_only_features": prompt_feats,
                        "preanswer_features": preanswer_feats, "seed": seed_i,
                        # new in this run
                        "prompt_ids": prompt_ids, "gen_ids": gen_ids, "gen_ids_raw": raw,
                        "gen_logprobs": logprobs, "prefix_grid": list(PREFIX_GRID),
                        "prefix_features": prefix_feats,
                        "prefix_texts": [tok.decode(gen_ids[:n], skip_special_tokens=True) for n in PREFIX_GRID],
                        "decode_roundtrip_ok": tok.decode(gen_ids, skip_special_tokens=True).strip() == text,
                    }
                    if ci == 0 and ti % CHECK_EVERY == 0 and len(gen_ids) >= 32:
                        sep = separate_forward(extractor, prompt_ids + gen_ids[:32])
                        diff = float((sep - prefix_feats[PREFIX_GRID.index(32)]).abs().max())
                        rec["equivalence_check_n32_maxabs"] = diff
                        checks.append(diff)
                    records.append(rec)
                del scores, seqs
                torch.cuda.empty_cache()
                if (ti + 1) % 10 == 0 or ti == 0:
                    print(f"[run1] shard {s0}: {ti + 1}/{len(tasks)} tasks, {time.time() - t0:.0f}s", flush=True)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{task['task_uid']}: {type(exc).__name__}: {exc}")
                print(f"[run1] ERROR {task['task_uid']}: {exc}", flush=True)
                torch.cuda.empty_cache()
        torch.save({"offset": s0, "n_tasks": len(tasks), "seed": SEED, "k": K, "max_new_tokens": MAX_NEW,
                    "temperature": 0.7, "top_p": 0.95, "prefix_grid": list(PREFIX_GRID),
                    "model": "shared/models/ouro_rltt_local", "elapsed_seconds": round(time.time() - t0, 1),
                    "equivalence_checks_maxabs": checks, "errors": errors, "records": records}, path)
        print(f"[run1] wrote {path.name}: {len(records)} records, {len(errors)} errors, "
              f"equivalence max {max(checks) if checks else None}", flush=True)
    extractor.cleanup()


if __name__ == "__main__":
    main()
