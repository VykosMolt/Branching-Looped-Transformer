"""Run 7 generation: every GSM8K problem no earlier run used, Ouro-RLTT, first-commitment labels. See PLAN.md.

A copy of Run 6's gen_run6.py with only the problem list, the paths and the run seed changed:
  - problems: GSM8K training split minus Run 6's cohort (first 2,000 in hash order) and Run 6's 3 smoke problems
    (last 3), plus GSM8K test problems outside the original 170 (indices 0-169), Run 3's 600-problem cohort and
    Run 3's 3 smoke problems; read from local, SHA-256-pinned parquet files (no network); asserted disjoint from
    every earlier run by normalized question text; all of them in hash order of task_id;
  - seed _sid(20261010, task_id) (Run 6's rule with this run's date); shards run7_shard_*.pt;
  - smoke: 3 problems from Run 6's cohort (so that every unused problem stays in the cohort).
Unchanged from Run 6 (gen_run3.py imported unchanged; its hash is in this run's SEAL): prompt, sampling, stop rule,
labels, features, prefix grid and the same-length equivalence check (amended 2026-10-04).
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

SEED = 20261010
K, MAX_NEW, SHARD = 4, 448, 100
PREFIX_GRID = (0, 16, 32, 48, 64, 96, 128, 160, 192, 256)
CHECK_EVERY = 25
DATA = ROOT / "elastic_reasoner/data/gsm8k/main"
TRAIN_SHA256 = "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d"  # Run 6's pin (openai/gsm8k 740312ad)
TEST_SHA256 = "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"   # Run 3's pin
N_COHORT = 6016


def H(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def norm_q(q: str) -> str:
    """Normalized question text for the overlap check: NFKC, case-folded, every non-alphanumeric run -> one space."""
    import re
    import unicodedata
    return re.sub(r"[^0-9a-z]+", " ", unicodedata.normalize("NFKC", q).casefold()).strip()


def _split(name: str, prefix: str, sha: str) -> list[dict]:
    """One GSM8K split from the local, SHA-256-pinned parquet (no network), as task dicts in hash order of task_id."""
    import pyarrow.parquet as pq
    path = DATA / f"{name}-00000-of-00001.parquet"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha, f"GSM8K {name} file changed"
    tasks = []
    for i, r in enumerate(pq.read_table(path).to_pylist()):
        sol, ans = r["answer"].split("####")
        tasks.append({"task_id": f"{prefix}/{i}", "dataset_split": name, "dataset_index": i, "question": r["question"],
                      "gold": ans.strip().replace(",", ""), "ref_steps": sum(1 for line in sol.splitlines() if line.strip())})
    return sorted(tasks, key=lambda t: H(t["task_id"]))


def earlier_runs() -> dict[str, list[dict]]:
    """Every GSM8K problem an earlier run generated on, by the earlier runs' own selection rules."""
    train, test = _split("train", "gsm8k-train", TRAIN_SHA256), _split("test", "gsm8k", TEST_SHA256)
    assert len(train) == 7473 and len(test) == 1319
    test_170 = sorted(test, key=lambda t: t["dataset_index"])[:170]  # original experiments: test 0..169 (June 17)
    run3_pool = [t for t in test if t["dataset_index"] >= 170]       # gen_run3.select_tasks: 170..1318, hash order
    return {"original_170": test_170, "run3_cohort": run3_pool[:600], "run3_smoke": run3_pool[600:603],
            "run6_cohort": train[:2000], "run6_smoke": train[-3:], "_train": train, "_test": test}


def select_tasks() -> list[dict]:
    """The cohort: every GSM8K problem (train + test) used by no earlier run, in hash order of task_id. Asserts zero
    overlap with every earlier run by task_id and by normalized question text."""
    runs = earlier_runs()
    used = [t for k, v in runs.items() if not k.startswith("_") for t in v]
    used_ids, used_q = {t["task_id"] for t in used}, {norm_q(t["question"]) for t in used}
    out = sorted([t for t in runs["_train"] + runs["_test"] if t["task_id"] not in used_ids], key=lambda t: H(t["task_id"]))
    assert not used_q & {norm_q(t["question"]) for t in out}, "overlap with an earlier run by question text"
    assert len({norm_q(t["question"]) for t in out}) == len(out), "duplicate question text inside the cohort"
    assert len(out) == N_COHORT, len(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--smoke", action="store_true", help="3 problems from Run 6's cohort, all checks, into smoke/")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    a = ap.parse_args()
    cohort, shard, check_every = select_tasks(), SHARD, CHECK_EVERY
    if a.smoke:
        cohort, shard, check_every, a.out = earlier_runs()["run6_cohort"][:3], 3, 1, HERE / "smoke"
    a.out.mkdir(parents=True, exist_ok=True)
    extractor = BGTransformerFeatureExtractor(device="cuda", dtype="auto", force_all_loops=True)
    model, tok = extractor.model, extractor.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    for s0 in range(0, len(cohort), shard):
        if (s0 // shard) % a.nparts != a.part:
            continue
        path = a.out / f"run7_shard_{s0:04d}.pt"
        if path.exists():
            print(f"[run7] {path.name} exists, skipping", flush=True)
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
                    print(f"[run7] shard {s0}: {ti + 1}/{len(cohort[s0:s0 + shard])} tasks, {time.time() - t0:.0f}s", flush=True)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{task['task_id']}: {type(exc).__name__}: {exc}")
                print(f"[run7] ERROR {task['task_id']}: {exc}", flush=True)
                torch.cuda.empty_cache()
        torch.save({"shard_start": s0, "seed": SEED, "k": K, "max_new_tokens": MAX_NEW, "prefix_grid": list(PREFIX_GRID),
                    "model": "shared/models/ouro_rltt_local", "smoke": a.smoke, "elapsed_seconds": round(time.time() - t0, 1),
                    "equivalence_checks_maxabs": checks, "errors": errors, "records": records}, path)
        print(f"[run7] wrote {path.name}: {len(records)} records, {len(errors)} errors, equivalence max "
              f"{max(checks) if checks else None}", flush=True)
    extractor.cleanup()


if __name__ == "__main__":
    main()
