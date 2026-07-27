"""Frozen conversion #4 -- all-well-formed terminal-selection pool: generation.

Sealed plan: artifacts/reports/wellformed_terminal_v4_20260727/EXPERIMENT_PLAN.md.
Minimal delta from bg_v2_overnight_horizon_generate.py: per task, sample rounds of 4
candidates (identical sampling params) until 4 WELL-FORMED candidates are kept or
MAX_ROUNDS rounds are spent. Well-formed = committed FINAL ANSWER marker + parseable
letter -- a form property; correctness is never consulted for keep/drop decisions.
Features: full_features only (terminal selection scores completed candidates).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from transformers import StoppingCriteriaList

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANUAL = PROJECT_ROOT / "utilities/tests/manual"
for p in (str(PROJECT_ROOT), str(MANUAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

from bg_steering_suite_lib import mcq_prompt, parse_mcq_answer  # noqa: E402
from bg_v2_overnight_horizon_generate import (  # noqa: E402
    _sid, split_for, load_task_pool, options_dict, preanswer_prefix,
    FinalAnswerStop, token_cut_index,
)
import hashlib  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "artifacts/reports/wellformed_terminal_v4_20260727"
SEED = 20260727
TASK_OFFSET, N_TASKS = 860, 220
K_WF = 4            # well-formed candidates to keep per task
ROUND_K = 4         # candidates sampled per round
MAX_ROUNDS = 4      # hard cap: 16 draws per task
MAX_NEW = 320
MIN_GROUP = 2       # tasks ending with <2 well-formed are dropped (counted)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-offset", type=int, default=TASK_OFFSET)
    ap.add_argument("--n-tasks", type=int, default=N_TASKS)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()
    offset, n_tasks = args.task_offset, args.n_tasks
    assert offset == 860 or offset >= 1350, \
        "sealed: primary slice is [860,1080); extensions must start at >= 1350"
    out_name = f"wellformed_pool{('_' + args.tag) if args.tag else ''}"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pool = load_task_pool()
    pool.sort(key=lambda d: hashlib.sha256(d["task_uid"].encode()).hexdigest())
    tasks = pool[offset: offset + n_tasks]
    print(f"[wf-gen] {len(tasks)} tasks, slice [{offset},{offset + n_tasks})")

    t0 = time.time()
    from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor

    extractor = BGTransformerFeatureExtractor(device="cuda", dtype="auto", force_all_loops=True)
    model = extractor.model
    tok = extractor.tokenizer
    print(f"[wf-gen] model loaded in {time.time() - t0:.1f}s")

    records, errors = [], []
    n_draws_total = 0
    dropped_tasks = []

    for ti, task in enumerate(tasks):
        try:
            opts = options_dict(task["options"])
            gold_letter = chr(65 + int(task["answer_key"]))
            prompt = mcq_prompt(task["task_prompt"], opts)
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1536, padding=False)
            enc = {kk: vv.to(extractor.device) for kk, vv in enc.items()}
            plen = int(enc["input_ids"].shape[1])
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            split = split_for(task["task_uid"])

            kept = []
            for rnd in range(MAX_ROUNDS):
                if len(kept) >= K_WF:
                    break
                seed_r = _sid(SEED, task["task_uid"], rnd)
                torch.manual_seed(seed_r)
                if extractor.device.type == "cuda":
                    torch.cuda.manual_seed_all(seed_r)
                stopper = StoppingCriteriaList([FinalAnswerStop(tok, plen, ROUND_K)])
                with torch.inference_mode():
                    out = model.generate(
                        **enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=0.7, top_p=0.95,
                        num_return_sequences=ROUND_K, pad_token_id=pad_id, eos_token_id=tok.eos_token_id,
                        use_cache=True, stopping_criteria=stopper,
                    )
                seqs = out.sequences.tolist() if hasattr(out, "sequences") else out.tolist()
                del out
                if extractor.device.type == "cuda":
                    torch.cuda.empty_cache()
                n_draws_total += ROUND_K

                for ci in range(ROUND_K):
                    if len(kept) >= K_WF:
                        break
                    new_ids = seqs[ci][plen:]
                    while new_ids and new_ids[-1] == pad_id:
                        new_ids.pop()
                    text = tok.decode(new_ids, skip_special_tokens=True).strip()
                    hit_max = len(new_ids) >= MAX_NEW
                    pre_text, found_marker = preanswer_prefix(text)
                    parsed_raw = parse_mcq_answer(text, opts)
                    parsed = parsed_raw if found_marker else None
                    well_formed = bool(found_marker and parsed is not None)
                    if not well_formed:
                        continue
                    cut_k = token_cut_index(tok, new_ids, len(pre_text))
                    cut_k = max(1, min(cut_k, len(new_ids)))
                    kept.append({
                        "task_uid": task["task_uid"], "split": split,
                        "proof_depth": task["proof_depth"], "category": task["category"],
                        "candidate_idx": len(kept), "draw_round": rnd, "draw_idx": ci,
                        "gold_letter": gold_letter, "options": opts,
                        "generated_text": text, "n_gen_tok": len(new_ids),
                        "hit_max_tokens": bool(hit_max), "found_final_marker": True,
                        "parsed_answer": parsed, "success": bool(parsed == gold_letter),
                        "malformed": False, "n_pre_tok": int(cut_k), "seed": seed_r,
                    })

            if len(kept) < MIN_GROUP:
                dropped_tasks.append(task["task_uid"])
                continue
            # full_features for kept candidates only (form filter already applied)
            for r in kept:
                full_text = f"{prompt}\n{r['generated_text']}".strip()
                r["full_features"] = extractor.encode_text_to_pooled_features(full_text)
            records.extend(kept)

            if (ti + 1) % 10 == 0 or ti == 0:
                print(f"[wf-gen] {ti + 1}/{len(tasks)} tasks, kept={len(records)} "
                      f"draws={n_draws_total} dropped={len(dropped_tasks)} "
                      f"elapsed={time.time() - t0:.0f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{task['task_uid']}: {type(exc).__name__}: {exc}")
            print(f"[wf-gen] ERROR {task['task_uid']}: {exc}", flush=True)
            if extractor.device.type == "cuda":
                torch.cuda.empty_cache()

    extractor.cleanup()
    payload = {
        "seed": SEED, "task_offset": offset, "n_tasks": n_tasks,
        "k_wf": K_WF, "round_k": ROUND_K, "max_rounds": MAX_ROUNDS, "max_new": MAX_NEW,
        "model": "models/ouro_rltt_local", "feature_shape": [3, 4, 2048],
        "n_records": len(records), "n_draws_total": n_draws_total,
        "dropped_tasks": dropped_tasks, "errors": errors,
        "elapsed_seconds": round(time.time() - t0, 1), "records": records,
    }
    out_path = OUT_ROOT / f"{out_name}.pt"
    torch.save(payload, out_path)
    light = {k: v for k, v in payload.items() if k != "records"}
    light["n_success"] = sum(1 for r in records if r["success"])
    group_sizes = {}
    for r in records:
        group_sizes[r["task_uid"]] = group_sizes.get(r["task_uid"], 0) + 1
    light["group_size_hist"] = {str(s): sum(1 for v in group_sizes.values() if v == s)
                                for s in sorted(set(group_sizes.values()))}
    (OUT_ROOT / f"{out_name}_index.json").write_text(
        json.dumps(light, indent=2, default=str) + "\n")
    print(f"[wf-gen] wrote {out_path} -- {len(records)} records over {len(group_sizes)} tasks, "
          f"{n_draws_total} draws, {len(dropped_tasks)} dropped, {len(errors)} errors")


if __name__ == "__main__":
    main()
