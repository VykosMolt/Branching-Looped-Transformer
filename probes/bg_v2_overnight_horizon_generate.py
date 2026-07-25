"""Paper 1 v2 overnight programme -- Part I generation.

Horizon Logic strict pre-answer + terminal-selection shared candidate pool.

Reuses, unmodified:
  - src.evaluator.bg_transformer_features.BGTransformerFeatureExtractor
    (canonical pooled [3,4,2048] features at layers {24,36,47} x loops {1..4})
  - utilities/tests/manual/bg_steering_suite_lib.mcq_prompt / parse_mcq_answer
  - the strict pre-answer cut convention from proto_introspection_preanswer_recapture.py
    (cut = text before the first `FINAL\\s*ANSWE` marker)

Task source: data/branch_training_logic_expansion_v1/processed/logic_tasks.jsonl,
category == synthetic_propositional (deterministic truth-table verifier, proof_depth
1-4 used as the reasoning-horizon control knob). depth==1 excluded a priori (pilot
checked for front-loading risk at low depth).

Writes one resumable .pt payload with: per-task split (train/val/heldout, task-level,
deterministic, prospective), per-candidate generated text, parsed answer, correctness
label (external verifier only), pre-answer cut text, canonical pooled hidden features
for prompt_only and preanswer cuts, and pre-cut shortcut statistics (token count, mean
logprob, min logprob, hit_max_tokens).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MANUAL = PROJECT_ROOT / "utilities/tests/manual"
if str(MANUAL) not in sys.path:
    sys.path.insert(0, str(MANUAL))

from bg_steering_suite_lib import mcq_prompt, parse_mcq_answer  # noqa: E402

LOGIC_TASKS_PATH = PROJECT_ROOT / "data/branch_training_logic_expansion_v1/processed/logic_tasks.jsonl"
OUT_ROOT = PROJECT_ROOT / "artifacts/reports/paper1_v2_overnight_20260724/horizon_logic"
FINAL_MARKER = re.compile(r"FINAL\s*ANSWE", re.IGNORECASE)

CATEGORY = "synthetic_propositional"
MIN_DEPTH = 2
MAX_DEPTH = 4


def _sid(seed_base: int, *parts) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed_base}\x1f{parts}".encode()).digest()[:8], "big") % (2**31)


def split_for(task_uid: str) -> str:
    h = int(hashlib.sha256(task_uid.encode()).hexdigest()[:8], 16) % 100
    if h < 50:
        return "train"
    if h < 70:
        return "val"
    return "heldout"


def load_task_pool(max_pool: int = 4000) -> list[dict]:
    rows = []
    with open(LOGIC_TASKS_PATH) as f:
        for line in f:
            d = json.loads(line)
            if d.get("category") != CATEGORY:
                continue
            depth = d.get("proof_depth")
            if depth is None or not (MIN_DEPTH <= depth <= MAX_DEPTH):
                continue
            rows.append(d)
            if len(rows) >= max_pool:
                break
    return rows


def options_dict(options: list[str]) -> dict[str, str]:
    letters = [chr(65 + i) for i in range(len(options))]
    return dict(zip(letters, options))


def preanswer_prefix(text: str) -> tuple[str, bool]:
    m = FINAL_MARKER.search(text)
    if m is None:
        return text.strip(), False
    return text[: m.start()].strip(), True


class FinalAnswerStop(StoppingCriteria):
    """Stop generation once every sequence in the batch has committed a FINAL ANSWER
    line plus a small trailing margin (to capture the letter/number), or hit max budget.
    Purely a wall-clock optimization -- does not change what counts as pre-answer text,
    since the strict cut is still computed from the marker position in the final text."""

    def __init__(self, tok, prompt_len: int, batch_size: int, extra_tokens: int = 8):
        self.tok = tok
        self.prompt_len = prompt_len
        self.batch_size = batch_size
        self.extra_tokens = extra_tokens
        self.marker_step: list[int | None] = [None] * batch_size

    def __call__(self, input_ids, scores, **kwargs) -> bool:  # noqa: ANN001
        cur_len = input_ids.shape[1] - self.prompt_len
        done = True
        for i in range(input_ids.shape[0]):
            if self.marker_step[i] is None:
                text = self.tok.decode(input_ids[i, self.prompt_len:], skip_special_tokens=True)
                if FINAL_MARKER.search(text):
                    self.marker_step[i] = cur_len
            if self.marker_step[i] is None or (cur_len - self.marker_step[i]) < self.extra_tokens:
                done = False
        return done


def token_cut_index(tok, ids: list[int], char_cut: int) -> int:
    """Smallest k such that decode(ids[:k]) has length >= char_cut (bisection)."""
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        dec_len = len(tok.decode(ids[:mid], skip_special_tokens=True))
        if dec_len >= char_cut:
            hi = mid
        else:
            lo = mid + 1
    return lo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks", type=int, default=24)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--seed", type=int, default=20260724)
    ap.add_argument("--tag", type=str, default="pilot")
    ap.add_argument("--task-offset", type=int, default=0)
    ap.add_argument("--budget-seconds", type=float, default=3600.0)
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pool = load_task_pool()
    print(f"[horizon] task pool (category={CATEGORY}, depth {MIN_DEPTH}-{MAX_DEPTH}): {len(pool)}")

    # Deterministic ordering independent of file order: sort by task_uid hash.
    pool.sort(key=lambda d: hashlib.sha256(d["task_uid"].encode()).hexdigest())
    tasks = pool[args.task_offset: args.task_offset + args.n_tasks]
    print(f"[horizon] selected {len(tasks)} tasks (offset={args.task_offset})")
    depth_hist = {}
    for t in tasks:
        depth_hist[t["proof_depth"]] = depth_hist.get(t["proof_depth"], 0) + 1
    print(f"[horizon] depth histogram: {depth_hist}")

    t0 = time.time()
    from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor

    extractor = BGTransformerFeatureExtractor(device="cuda", dtype="auto", force_all_loops=True)
    model = extractor.model
    tok = extractor.tokenizer
    print(f"[horizon] model loaded in {time.time()-t0:.1f}s")

    records = []
    errors = []
    budget_hit = False

    for ti, task in enumerate(tasks):
        if time.time() - t0 > args.budget_seconds:
            budget_hit = True
            print(f"[horizon] budget hit after {ti} tasks")
            break
        tt = time.time()
        try:
            opts = options_dict(task["options"])
            gold_letter = chr(65 + int(task["answer_key"]))
            prompt = mcq_prompt(task["task_prompt"], opts)

            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1536, padding=False)
            enc = {kk: vv.to(extractor.device) for kk, vv in enc.items()}
            plen = int(enc["input_ids"].shape[1])
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            seed_i = _sid(args.seed, task["task_uid"])
            torch.manual_seed(seed_i)
            if extractor.device.type == "cuda":
                torch.cuda.manual_seed_all(seed_i)
            stopper = StoppingCriteriaList([FinalAnswerStop(tok, plen, args.k)])
            with torch.inference_mode():
                out = model.generate(
                    **enc, max_new_tokens=args.max_new, do_sample=True, temperature=0.7, top_p=0.95,
                    num_return_sequences=args.k, pad_token_id=pad_id, eos_token_id=tok.eos_token_id,
                    use_cache=True, output_scores=True, return_dict_in_generate=True,
                    stopping_criteria=stopper,
                )

            seqs = out.sequences.tolist()  # move off GPU immediately; out.scores tensors stay for logprobs
            n_steps = len(out.scores)
            # Pull each step's logits to CPU float once (small: [k, vocab]) instead of keeping
            # the whole `out` (incl. GPU sequences/attentions) referenced for the rest of the task.
            scores_cpu = [s.detach().to("cpu", dtype=torch.float32) for s in out.scores]
            del out
            if extractor.device.type == "cuda":
                torch.cuda.empty_cache()
            split = split_for(task["task_uid"])

            for ci in range(args.k):
                new_ids_full = seqs[ci][plen:]
                # trim trailing pad/eos
                while new_ids_full and new_ids_full[-1] == pad_id:
                    new_ids_full.pop()
                text = tok.decode(new_ids_full, skip_special_tokens=True).strip()
                hit_max = len(new_ids_full) >= args.max_new

                pre_text, found_marker = preanswer_prefix(text)
                char_cut = len(pre_text)
                cut_k = token_cut_index(tok, new_ids_full, char_cut) if found_marker else len(new_ids_full)
                cut_k = max(1, min(cut_k, len(new_ids_full)))

                # per-token logprob of the sampled token, for steps < cut_k
                logprobs = []
                for step in range(min(cut_k, n_steps)):
                    logits_step = scores_cpu[step][ci]
                    tok_id = new_ids_full[step] if step < len(new_ids_full) else None
                    if tok_id is None:
                        continue
                    logp = torch.log_softmax(logits_step, dim=-1)[tok_id].item()
                    logprobs.append(logp)

                parsed_raw = parse_mcq_answer(text, opts)
                # Require an explicit committed FINAL ANSWER line; a bare letter picked
                # up by a fallback regex without the marker is not a genuine commitment.
                parsed = parsed_raw if found_marker else None
                success = bool(found_marker and parsed == gold_letter)
                malformed = not found_marker or parsed is None

                prompt_feats = extractor.encode_text_to_pooled_features(prompt)
                preanswer_text = f"{prompt}\n{pre_text}" if pre_text else prompt
                preanswer_feats = extractor.encode_text_to_pooled_features(preanswer_text)

                records.append({
                    "task_uid": task["task_uid"], "split": split, "proof_depth": task["proof_depth"],
                    "category": task["category"], "candidate_idx": ci,
                    "gold_answer": task["gold_answer"], "gold_letter": gold_letter,
                    "options": opts, "generated_text": text, "n_gen_tok": len(new_ids_full),
                    "hit_max_tokens": bool(hit_max), "found_final_marker": bool(found_marker),
                    "parsed_answer": parsed, "success": bool(success), "malformed": bool(malformed),
                    "n_pre_tok": int(cut_k), "mean_logprob_pre": (sum(logprobs) / len(logprobs)) if logprobs else None,
                    "min_logprob_pre": (min(logprobs)) if logprobs else None,
                    "prompt_tok": plen,
                    "prompt_only_features": prompt_feats,
                    "preanswer_features": preanswer_feats,
                    "seed": seed_i,
                })
            del scores_cpu, seqs
            if extractor.device.type == "cuda":
                torch.cuda.empty_cache()
            if (ti + 1) % 5 == 0 or ti == 0:
                elapsed = time.time() - t0
                print(f"[horizon] {ti+1}/{len(tasks)} tasks done, elapsed={elapsed:.1f}s, "
                      f"last_task_seconds={time.time()-tt:.1f}s")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{task['task_uid']}: {type(exc).__name__}: {exc}")
            print(f"[horizon] ERROR {task['task_uid']}: {exc}")
            continue

    extractor.cleanup()

    payload = {
        "tag": args.tag, "category": CATEGORY, "depth_range": [MIN_DEPTH, MAX_DEPTH],
        "model": "models/ouro_rltt_local", "tap_layers": [24, 36, 47], "num_loops": 4,
        "feature_shape": [3, 4, 2048], "k_per_task": args.k, "max_new_tokens": args.max_new,
        "temperature": 0.7, "top_p": 0.95, "seed": args.seed, "task_offset": args.task_offset,
        "n_tasks_requested": len(tasks), "n_tasks_completed": len(tasks) - (1 if budget_hit else 0),
        "n_candidates": len(records), "budget_seconds": args.budget_seconds, "budget_hit": budget_hit,
        "elapsed_seconds": round(time.time() - t0, 1), "errors": errors, "records": records,
    }
    out_path = OUT_ROOT / f"horizon_generation_{args.tag}_off{args.task_offset}.pt"
    torch.save(payload, out_path)
    light = {k: v for k, v in payload.items() if k != "records"}
    light["n_success"] = sum(1 for r in records if r["success"])
    light["n_malformed"] = sum(1 for r in records if r["malformed"])
    light["n_hit_max"] = sum(1 for r in records if r["hit_max_tokens"])
    light["mean_n_pre_tok"] = (sum(r["n_pre_tok"] for r in records) / len(records)) if records else None
    light["records_summary"] = [
        {kk: r[kk] for kk in ("task_uid", "split", "proof_depth", "candidate_idx", "success",
                              "malformed", "n_pre_tok", "n_gen_tok", "hit_max_tokens",
                              "found_final_marker", "parsed_answer", "gold_letter")}
        for r in records
    ]
    idx_path = OUT_ROOT / f"horizon_generation_{args.tag}_off{args.task_offset}_index.json"
    idx_path.write_text(json.dumps(light, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"[horizon] wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB) and {idx_path}")
    print(f"[horizon] n_candidates={len(records)} n_success={light['n_success']} "
          f"n_malformed={light['n_malformed']} n_hit_max={light['n_hit_max']} "
          f"mean_n_pre_tok={light['mean_n_pre_tok']} errors={len(errors)} "
          f"elapsed={payload['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
