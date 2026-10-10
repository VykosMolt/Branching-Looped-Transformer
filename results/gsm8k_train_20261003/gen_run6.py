"""Run 6 generation: GSM8K training-split problems, Ouro-RLTT, first-commitment labels. See PLAN.md.

Same prompt, sampling and features as Run 3 (gen_run3.py, imported unchanged; its hash is in this run's SEAL), with:
  - problems: the first 2,000 GSM8K training-split problems in hash order (Run 3 used the test split);
  - stop: once every candidate has written "FINAL ANSWER: <number>" plus 8 tokens (Run 3 additionally waited for
    step 224 to replicate a published protocol, which this run does not need);
  - no published-protocol cuts; label = the candidate's first committed answer (as Run 3's primary);
  - --part/--nparts to split shards across parallel processes (every problem keeps its own seed).
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
R3 = ROOT / "opi/preanswer/gsm8k_fixed_prefix_20260927"
for p in (ROOT, ROOT / "shared", ROOT / "shared/utilities/tests/manual", R3, HERE):
    sys.path.insert(0, str(p))

from answer_detector_gsm8k import first_commitment, value  # noqa: E402
from bg_steering_suite_lib import gsm_prompt  # noqa: E402
from bg_v2_overnight_horizon_generate import _sid, split_for  # noqa: E402
from gen_run3 import CommitStop, forward_all  # noqa: E402
from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor  # noqa: E402

SEED = 20261003
K, MAX_NEW, N_TASKS, SHARD = 4, 448, 2000, 100
PREFIX_GRID = (0, 16, 32, 48, 64, 96, 128, 160, 192, 256)
CHECK_EVERY = 25
REVISION = "740312add88f781978c0658806c59bc2815b9866"
TRAIN_SHA256 = "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d"


def select_tasks() -> list[dict]:
    """All 7,473 GSM8K training problems in hash order of 'gsm8k-train/{index}'; the first N_TASKS are the cohort,
    the last ones are used only for smoke tests."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("openai/gsm8k", "main/train-00000-of-00001.parquet", repo_type="dataset", revision=REVISION)
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == TRAIN_SHA256, "GSM8K training file changed"
    rows = pq.read_table(path).to_pylist()
    assert len(rows) == 7473
    tasks = []
    for i, r in enumerate(rows):
        sol, ans = r["answer"].split("####")
        tasks.append({"task_id": f"gsm8k-train/{i}", "dataset_index": i, "question": r["question"],
                      "gold": ans.strip().replace(",", ""), "ref_steps": sum(1 for line in sol.splitlines() if line.strip())})
    tasks.sort(key=lambda t: hashlib.sha256(t["task_id"].encode()).hexdigest())
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--smoke", action="store_true", help="3 problems from the end of the hash order, all checks, into smoke/")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    a = ap.parse_args()
    tasks_all = select_tasks()
    cohort, shard, check_every = tasks_all[:N_TASKS], SHARD, CHECK_EVERY
    if a.smoke:
        cohort, shard, check_every, a.out = tasks_all[-3:], 3, 1, HERE / "smoke"
    a.out.mkdir(parents=True, exist_ok=True)
    extractor = BGTransformerFeatureExtractor(device="cuda", dtype="auto", force_all_loops=True)
    model, tok = extractor.model, extractor.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    for s0 in range(0, len(cohort), shard):
        if (s0 // shard) % a.nparts != a.part:
            continue
        path = a.out / f"run6_shard_{s0:04d}.pt"
        if path.exists():
            print(f"[run6] {path.name} exists, skipping", flush=True)
            continue
        t0, records, errors, checks = time.time(), [], [], []
        for ti, task in enumerate(cohort[s0:s0 + shard]):
            try:
                prompt = gsm_prompt(task["question"])
                enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1536, padding=False)
                enc = {k: v.to(extractor.device) for k, v in enc.items()}
                prompt_ids = enc["input_ids"][0].tolist()
                plen = len(prompt_ids)
                seed_i = _sid(SEED, task["task_id"])
                torch.manual_seed(seed_i)
                torch.cuda.manual_seed_all(seed_i)
                with torch.inference_mode():
                    gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=0.7, top_p=0.95,
                                         num_return_sequences=K, pad_token_id=pad_id, eos_token_id=tok.eos_token_id,
                                         use_cache=True, output_scores=True, return_dict_in_generate=True,
                                         stopping_criteria=StoppingCriteriaList([CommitStop(tok, plen, K, min_steps=0)]))
                seqs = gen.sequences.tolist()
                scores = [sc.detach().to("cpu", dtype=torch.float32) for sc in gen.scores]
                del gen
                torch.cuda.empty_cache()
                for ci in range(K):
                    gen_ids = list(seqs[ci][plen:])
                    while gen_ids and gen_ids[-1] == pad_id:
                        gen_ids.pop()
                    logprobs = [torch.log_softmax(scores[st][ci], dim=-1)[gen_ids[st]].item() for st in range(min(len(gen_ids), len(scores)))]
                    text = tok.decode(gen_ids, skip_special_tokens=True).strip()
                    own = first_commitment(text)
                    prefix_feats, _, _, prompt_end = forward_all(extractor, prompt_ids + gen_ids, plen, PREFIX_GRID)
                    rec = {**task, "split": split_for(task["task_id"]), "candidate_idx": ci, "seed": seed_i,
                           "prompt_tok": plen, "question_chars": len(task["question"]), "prompt_ids": prompt_ids,
                           "gen_ids": gen_ids, "n_gen_tok": len(gen_ids), "hit_max_tokens": len(gen_ids) >= MAX_NEW,
                           "generated_text": text, "gen_logprobs": logprobs,
                           "decode_roundtrip_ok": tok.decode(gen_ids, skip_special_tokens=True).strip() == text,
                           "own_answer": own, "parseable": value(own) is not None,
                           "success": value(own) is not None and value(own) == value(task["gold"]),
                           "prefix_grid": list(PREFIX_GRID), "prefix_features": prefix_feats, "prompt_end_state": prompt_end,
                           "prefix_texts": [tok.decode(gen_ids[:n], skip_special_tokens=True) for n in PREFIX_GRID]}
                    if ci == 0 and ti % check_every == 0 and len(gen_ids) > 0:
                        # same-length check (amended 2026-10-04, before any cohort data): the prefix's pooled features
                        # must not change when the tokens after it are replaced; same total length, so the same kernels
                        n = min(128, len(gen_ids))
                        ids = prompt_ids + gen_ids
                        real = forward_all(extractor, ids, plen, [n])[0][0]
                        other = forward_all(extractor, ids[:plen + n] + [13] * (len(ids) - plen - n), plen, [n])[0][0]
                        diff = float((real - other).abs().max())
                        rec["equivalence_check_n128_maxabs"] = diff
                        checks.append(diff)
                    records.append(rec)
                del scores, seqs
                torch.cuda.empty_cache()
                if (ti + 1) % 10 == 0 or ti == 0:
                    print(f"[run6] shard {s0}: {ti + 1}/{len(cohort[s0:s0 + shard])} tasks, {time.time() - t0:.0f}s", flush=True)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{task['task_id']}: {type(exc).__name__}: {exc}")
                print(f"[run6] ERROR {task['task_id']}: {exc}", flush=True)
                torch.cuda.empty_cache()
        torch.save({"shard_start": s0, "seed": SEED, "k": K, "max_new_tokens": MAX_NEW, "prefix_grid": list(PREFIX_GRID),
                    "model": "shared/models/ouro_rltt_local", "smoke": a.smoke, "elapsed_seconds": round(time.time() - t0, 1),
                    "equivalence_checks_maxabs": checks, "errors": errors, "records": records}, path)
        print(f"[run6] wrote {path.name}: {len(records)} records, {len(errors)} errors, equivalence max "
              f"{max(checks) if checks else None}", flush=True)
    extractor.cleanup()


if __name__ == "__main__":
    main()
