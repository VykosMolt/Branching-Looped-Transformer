"""Run 2: the matched-budget prefix-prune tournament, rerun on a pool that can differentiate policies.

The sealed tournament (opi/branching/tournament_v4_20260727) returned POOL_CANNOT_DIFFERENTIATE:
97.8% of its depth 2-4 tasks had a correct answer among six samples, above its 0.95 feasibility
ceiling. This rerun keeps that design exactly (bg_v4_tournament.py: B=6 as two seeded rounds of
three, max 320 new tokens, P=80, S=2, the frozen DualAnchor scorer, the same policies, controls,
bootstrap and verdict labels) and changes only the task set: the first 180, in the same hash order,
of the depth-4 Horizon tasks that no earlier run used (outside pool[0:3000); depth 4 had
any-correct-of-6 = 0.882 in the sealed run). It also saves prompt and generated token IDs.
See PLAN.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from transformers import StoppingCriteriaList

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (ROOT, ROOT / "shared", ROOT / "shared/utilities/tests/manual"):
    sys.path.insert(0, str(p))

import bg_v4_tournament as T  # noqa: E402
from bg_steering_suite_lib import mcq_prompt, parse_mcq_answer  # noqa: E402
from bg_v2_overnight_horizon_generate import (  # noqa: E402
    _sid, split_for, load_task_pool, options_dict, preanswer_prefix, FinalAnswerStop,
)

TASK_FILE = ROOT / "shared/data/branch_training_logic_expansion_v1/processed/logic_tasks.jsonl"
N_TASKS, DEPTH, USED_POOL_PREFIX = 180, 4, 3000


def select_tasks() -> list[dict]:
    capped = sorted(load_task_pool(), key=lambda d: hashlib.sha256(d["task_uid"].encode()).hexdigest())
    used = {t["task_uid"] for t in capped[:USED_POOL_PREFIX]}
    rows = []
    for line in open(TASK_FILE):
        d = json.loads(line)
        if d.get("category") == "synthetic_propositional" and d.get("proof_depth") == DEPTH and d["task_uid"] not in used:
            rows.append(d)
    rows.sort(key=lambda d: hashlib.sha256(d["task_uid"].encode()).hexdigest())
    assert len({r["task_uid"] for r in rows}) == len(rows)
    return rows[:N_TASKS]


def stage_generate(out_root: Path, n_tasks: int | None = None) -> None:
    """bg_v4_tournament.stage_generate, with the task set above and token IDs kept."""
    from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor
    tasks = select_tasks()[: n_tasks or N_TASKS]
    print(f"[tour2-gen] {len(tasks)} depth-{DEPTH} tasks", flush=True)
    t0 = time.time()
    extractor = BGTransformerFeatureExtractor(device="cuda", dtype="auto", force_all_loops=True)
    model, tok = extractor.model, extractor.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    records, errors = [], []
    for ti, task in enumerate(tasks):
        try:
            opts = options_dict(task["options"])
            gold_letter = chr(65 + int(task["answer_key"]))
            prompt = mcq_prompt(task["task_prompt"], opts)
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1536, padding=False)
            enc = {kk: vv.to(extractor.device) for kk, vv in enc.items()}
            plen = int(enc["input_ids"].shape[1])
            seqs = []
            for rnd in range(T.ROUNDS_PER_TASK):
                seed_r = _sid(T.SEED, task["task_uid"], rnd)
                torch.manual_seed(seed_r)
                torch.cuda.manual_seed_all(seed_r)
                stopper = StoppingCriteriaList([FinalAnswerStop(tok, plen, T.ROUND_B)])
                with torch.inference_mode():
                    out = model.generate(
                        **enc, max_new_tokens=T.MAX_NEW, do_sample=True, temperature=0.7, top_p=0.95,
                        num_return_sequences=T.ROUND_B, pad_token_id=pad_id, eos_token_id=tok.eos_token_id,
                        use_cache=True, stopping_criteria=stopper,
                    )
                seqs.extend(out.sequences.tolist() if hasattr(out, "sequences") else out.tolist())
                del out
                torch.cuda.empty_cache()
            for ci in range(T.B):
                new_ids = seqs[ci][plen:]
                while new_ids and new_ids[-1] == pad_id:
                    new_ids.pop()
                text = tok.decode(new_ids, skip_special_tokens=True).strip()
                pre_text, found_marker = preanswer_prefix(text)
                parsed_raw = parse_mcq_answer(text, opts)
                parsed = parsed_raw if found_marker else None
                prefix_ids = new_ids[:T.P]
                prefix_text = tok.decode(prefix_ids, skip_special_tokens=True).strip()
                full_feats = extractor.encode_text_to_pooled_features(f"{prompt}\n{text}".strip())
                prefix_feats = extractor.encode_text_to_pooled_features(f"{prompt}\n{prefix_text}".strip())
                records.append({
                    "task_uid": task["task_uid"], "split": split_for(task["task_uid"]),
                    "proof_depth": task["proof_depth"], "draw_idx": ci,
                    "gold_letter": gold_letter, "generated_text": text,
                    "n_gen_tok": len(new_ids), "hit_max_tokens": len(new_ids) >= T.MAX_NEW,
                    "found_final_marker": bool(found_marker), "parsed_answer": parsed,
                    "success": bool(found_marker and parsed == gold_letter),
                    "malformed": bool(not found_marker or parsed is None),
                    "full_features": full_feats, "prefix_features": prefix_feats,
                    "prefix_tok": len(prefix_ids), "seed": _sid(T.SEED, task["task_uid"], ci // T.ROUND_B),
                    "prompt_ids": enc["input_ids"][0].tolist(), "gen_ids": new_ids, "task_prompt": task["task_prompt"],
                })
            if (ti + 1) % 10 == 0 or ti == 0:
                print(f"[tour2-gen] {ti + 1}/{len(tasks)} tasks elapsed={time.time() - t0:.0f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{task['task_uid']}: {type(exc).__name__}: {exc}")
            print(f"[tour2-gen] ERROR {task['task_uid']}: {exc}", flush=True)
            torch.cuda.empty_cache()
    extractor.cleanup()
    torch.save({"seed": T.SEED, "depth": DEPTH, "n_tasks": len(tasks), "B": T.B, "P": T.P,
                "max_new": T.MAX_NEW, "model": "shared/models/ouro_rltt_local",
                "task_uids": [t["task_uid"] for t in tasks], "n_records": len(records), "errors": errors,
                "elapsed_seconds": round(time.time() - t0, 1), "records": records},
               out_root / "tournament_table.pt")
    print(f"[tour2-gen] wrote table: {len(records)} records, {len(errors)} errors", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["generate", "analyze"], required=True)
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--n-tasks", type=int, default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    T.OUT_ROOT = a.out          # stage_analyze reads and writes here
    if a.stage == "generate":
        stage_generate(a.out, a.n_tasks)
    else:
        T.stage_analyze()
