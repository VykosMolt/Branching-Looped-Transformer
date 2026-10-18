"""Run 8 generation: Run 4's 1,164 Horizon questions on a second checkpoint, Ouro-2.6B-Thinking. See PLAN.md.

A copy of Run 4's gen_run4.py with only the model and the paths changed:
  - model: ByteDance/Ouro-2.6B-Thinking at revision f1edd81e7ac41355db670500ceaf204e0f73af68 (public), downloaded into
    shared/models/ouro_thinking_local if missing; its modeling_ouro.py is replaced by modeling_ouro_shim.py (the
    upstream file plus a property/setter for UniversalTransformerCache.key_cache/value_cache, needed by
    transformers 4.54.1; the same shim the July Thinking runs used). Every file is SHA-256-checked before loading;
  - shards run8_shard_*.pt in this directory; the shard records the model and the checked hashes.
Unchanged from Run 4: tasks (select_tasks), prompt, sampling (k=4, T=0.7, top-p 0.95, 448 new tokens,
FinalAnswerStop), seed _sid(20260724, task_uid), split_for, labels (strict_answer), features, prefix grid, the
final-answer-line cut and the same-length equivalence check (amended 2026-10-04).
Writes resumable shards of 100 tasks into this directory.
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
from transformers import StoppingCriteriaList

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (ROOT, ROOT / "shared", ROOT / "shared/utilities/tests/manual", HERE):
    sys.path.insert(0, str(p))

from bg_steering_suite_lib import mcq_prompt, parse_mcq_answer  # noqa: E402
from bg_v2_overnight_horizon_generate import (  # noqa: E402
    LOGIC_TASKS_PATH, _sid, split_for, load_task_pool, options_dict, preanswer_prefix, FinalAnswerStop, token_cut_index,
)
from src.evaluator.bg_transformer_features import (  # noqa: E402
    BGTransformerFeatureExtractor, _LoopLayerCapture, _masked_mean_pool, TAP_LAYERS, LAYER_TO_CAPTURE_INDEX, NUM_LOOPS,
)

SEED = 20260724
K, MAX_NEW, SHARD = 4, 448, 100
PREFIX_GRID = (0, 16, 32, 48, 64, 96, 128)
CHECK_EVERY = 25
TASKS_SHA256 = "d5503195c4eb6ac3e45b5baef0761690b2b6c120f8df2a5d80ada3f7224e881d"
MODEL = ROOT / "shared/models/ouro_thinking_local"
REPO, REVISION = "ByteDance/Ouro-2.6B-Thinking", "f1edd81e7ac41355db670500ceaf204e0f73af68"
UPSTREAM_MODELING_SHA256 = "263c097a2ed6870ce549a65ef6d6bb9e8e167423f87d2c494e0988b9a4f63f4b"
# model.safetensors: the LFS SHA-256 from the HF API at this revision (5,336,011,242 bytes); the small files: SHA-256
# of the files downloaded at this revision on 2026-10-10; modeling_ouro.py: the shim (modeling_ouro_shim.py)
MODEL_FILES = {
    "model.safetensors": "c506a79247dc51fc0400d789365c3d43932f718abce9810f3606ace47d0a3080",
    "config.json": "ad4b51dffae60bebeeac5e1d5055e78d01fc01b1a29cd89cd2e50edb848f645a",
    "configuration_ouro.py": "ba60253335cecc5bd9cb3e5f8cfbd214aa7e865fc97a98208976fb2517811de6",
    "modeling_ouro.py": "84f4a55cdde6fa2f152be619dcf6edb5f08f9f1bf168dc176c3989376d3a8374",
    "tokenizer.json": "fcb808fe5e7642f5299be28aea07fc7f6d4f4364c3ac5e408e15a772cbc8fa8d",
    "tokenizer_config.json": "0d304de54d3f8e0c0573d678c0f1d80cc26e9d23dcb328405f0ec99c4293d04f",
    "special_tokens_map.json": "688ac94df221b18a4d2d4579dfa3c4f0fd5ca3e394da46cd0a317fb5d57d4685",
    "vocab.json": "7b9de3f47796abf8d00ab96be299fea0dc9afdf1827f34e7e0b9fb44593efe5c",
    "merges.txt": "0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510",
}


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def ensure_model() -> dict:
    """Download the pinned snapshot if any file is missing (needs network), install the shim, then assert the
    SHA-256 of every file. Generation never starts on unchecked weights."""
    import shutil
    if not all((MODEL / f).exists() for f in MODEL_FILES):
        from huggingface_hub import snapshot_download
        snapshot_download(REPO, revision=REVISION, local_dir=MODEL, allow_patterns=list(MODEL_FILES))
        assert sha_file(MODEL / "modeling_ouro.py") == UPSTREAM_MODELING_SHA256, "upstream modeling_ouro.py changed"
        shutil.copyfile(HERE / "modeling_ouro_shim.py", MODEL / "modeling_ouro.py")
    got = {f: sha_file(MODEL / f) for f in MODEL_FILES}
    bad = [f for f in MODEL_FILES if got[f] != MODEL_FILES[f]]
    assert not bad, f"model files differ from the pinned SHA-256: {bad}"
    return got


def H(u: str) -> str:
    return hashlib.sha256(u.encode()).hexdigest()


def select_tasks() -> list[dict]:
    """Depth-3 and depth-4 tasks used by no earlier run: not in the hash-ordered capped pool's [0, 3000) (v2/v3
    cohorts, Thinking runs, Run 1) and not among Run 2's 180 depth-4 tasks. Hash order."""
    assert hashlib.sha256(Path(LOGIC_TASKS_PATH).read_bytes()).hexdigest() == TASKS_SHA256
    rows = [json.loads(l) for l in open(LOGIC_TASKS_PATH)]
    pool = [d for d in rows if d.get("category") == "synthetic_propositional" and d.get("proof_depth") is not None
            and 2 <= d["proof_depth"] <= 4]
    capped = sorted(load_task_pool(), key=lambda d: H(d["task_uid"]))
    assert [d["task_uid"] for d in capped] == sorted([d["task_uid"] for d in pool[:4000]], key=H)
    used = {t["task_uid"] for t in capped[:3000]}
    run2 = sorted([d for d in pool if d["proof_depth"] == 4 and d["task_uid"] not in used], key=lambda d: H(d["task_uid"]))[:180]
    used |= {d["task_uid"] for d in run2}
    out = sorted([d for d in pool if d["proof_depth"] in (3, 4) and d["task_uid"] not in used], key=lambda d: H(d["task_uid"]))
    assert len(out) == 1164 and len({d["task_uid"] for d in out}) == len(out)
    return out


def strict_answer(text: str, options: dict) -> str | None:
    """First commitment: the first 'FINAL ANSWER' (any case, optionally in **) followed directly by a capital letter (optionally after ** ( [ < or quotes)
    or a boxed letter, whichever comes first; failing that, the first 'FINAL ANSWER' followed by an option word."""
    letters = "".join(sorted(options))
    lead = r"(?i:final\s*answer)[\s\*]*[:\-]?[\s\*\(\[<\"']*"  # marker in any case; the letter must be a capital
    cands = [m for m in (re.search(lead + rf"([{letters}])\b", text), re.search(rf"\\boxed\{{\s*([{letters}])\s*\}}", text)) if m]
    if cands:
        return min(cands, key=lambda m: m.start()).group(1).upper()
    words = {v.strip().rstrip(".").lower(): k for k, v in options.items()}
    m = re.search(lead + r"(" + "|".join(map(re.escape, words)) + r")\b", text, flags=re.I)
    return words[m.group(1).lower()] if m else None


def forward_all(extractor, ids: list[int], plen: int, cuts):
    """One causal forward (no attention mask). Pooled [len(cuts), 3, 4, 2048] features for the prompt plus the first
    n generated tokens, and the [3, 4, 2048] state at the last prompt token."""
    x = torch.tensor([ids], device=extractor.device)
    cap = _LoopLayerCapture()
    with torch.inference_mode(), extractor._capture_hooks(cap):
        extractor.model(input_ids=x, use_cache=False)
    extractor._validate_capture(cap)
    states = {47: cap.boundary, **{layer: cap.inter[idx] for layer, idx in LAYER_TO_CAPTURE_INDEX.items()}}
    feats = torch.zeros(len(cuts), len(TAP_LAYERS), NUM_LOOPS, states[47][0].shape[-1])
    for ci, n in enumerate(cuts):
        m = torch.zeros(1, len(ids))
        m[0, : plen + min(n, len(ids) - plen)] = 1.0
        for li, layer in enumerate(TAP_LAYERS):
            for u in range(NUM_LOOPS):
                feats[ci, li, u] = _masked_mean_pool(states[layer][u], m)
    pend = torch.stack([torch.stack([states[layer][u][0, plen - 1].float().cpu() for u in range(NUM_LOOPS)])
                        for layer in TAP_LAYERS])
    del cap
    return feats, pend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--smoke", action="store_true", help="2 depth-2 tasks outside every cohort, all checks, into smoke/")
    ap.add_argument("--part", type=int, default=0, help="with --nparts: process only shards whose index %% nparts == part")
    ap.add_argument("--nparts", type=int, default=1)
    a = ap.parse_args()
    tasks, shard, check_every = select_tasks(), SHARD, CHECK_EVERY
    if a.smoke:
        rows = [json.loads(l) for l in open(LOGIC_TASKS_PATH)]
        pool = [d for d in rows if d.get("category") == "synthetic_propositional" and d.get("proof_depth") == 2]
        tasks, shard, check_every, a.out = sorted(pool[4000:], key=lambda d: H(d["task_uid"]))[-2:], 2, 1, HERE / "smoke"
    a.out.mkdir(parents=True, exist_ok=True)
    model_sha = ensure_model()
    extractor = BGTransformerFeatureExtractor(model_path=MODEL, device="cuda", dtype="auto", force_all_loops=True)
    model, tok = extractor.model, extractor.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    for s0 in range(0, len(tasks), shard):
        if (s0 // shard) % a.nparts != a.part:  # parallel processes split the shards; every task keeps its own seed
            continue
        path = a.out / f"run8_shard_{s0:04d}.pt"
        if path.exists():
            print(f"[run8] {path.name} exists, skipping", flush=True)
            continue
        t0, records, errors, checks = time.time(), [], [], []
        for ti, task in enumerate(tasks[s0:s0 + shard]):
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
                with torch.inference_mode():
                    gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=0.7, top_p=0.95,
                                         num_return_sequences=K, pad_token_id=pad_id, eos_token_id=tok.eos_token_id,
                                         use_cache=True, output_scores=True, return_dict_in_generate=True,
                                         stopping_criteria=StoppingCriteriaList([FinalAnswerStop(tok, plen, K)]))
                seqs = gen.sequences.tolist()
                scores = [sc.detach().to("cpu", dtype=torch.float32) for sc in gen.scores]
                del gen
                torch.cuda.empty_cache()
                sym = task["symbolic"]
                for ci in range(K):
                    gen_ids = list(seqs[ci][plen:])
                    while gen_ids and gen_ids[-1] == pad_id:
                        gen_ids.pop()
                    logprobs = [torch.log_softmax(scores[st][ci], dim=-1)[gen_ids[st]].item() for st in range(min(len(gen_ids), len(scores)))]
                    text = tok.decode(gen_ids, skip_special_tokens=True).strip()
                    pre_text, found = preanswer_prefix(text)
                    cut_k = max(1, min(token_cut_index(tok, gen_ids, len(pre_text)) if found else len(gen_ids), len(gen_ids)))
                    lp_pre = logprobs[:cut_k]
                    parsed_pub = parse_mcq_answer(text, opts) if found else None
                    strict = strict_answer(text, opts)
                    prefix_feats, prompt_end = forward_all(extractor, prompt_ids + gen_ids, plen, PREFIX_GRID)
                    rec = {
                        "task_uid": task["task_uid"], "split": split_for(task["task_uid"]), "proof_depth": task["proof_depth"],
                        "n_vars": task["verifier"]["n_vars"], "n_facts": len(sym["facts"]), "n_rules": len(sym["rules"]),
                        "gold_answer": task["gold_answer"], "gold_letter": gold_letter, "options": opts,
                        "task_prompt": task["task_prompt"], "candidate_idx": ci, "seed": seed_i,
                        "generated_text": text, "n_gen_tok": len(gen_ids), "hit_max_tokens": len(gen_ids) >= MAX_NEW,
                        "prompt_tok": plen, "prompt_ids": prompt_ids, "gen_ids": gen_ids, "gen_logprobs": logprobs,
                        # labels: strict first commitment (primary) and the published parser (comparison)
                        "strict_answer": strict, "labelled": strict is not None, "success": strict == gold_letter,
                        "parsed_published": parsed_pub, "malformed_published": bool(not found or parsed_pub is None),
                        "success_published": bool(found and parsed_pub == gold_letter),
                        # final-answer-line cut, as published (v2/v3)
                        "found_final_marker": bool(found), "n_pre_tok": int(cut_k),
                        "mean_logprob_pre": sum(lp_pre) / len(lp_pre) if lp_pre else None,
                        "min_logprob_pre": min(lp_pre) if lp_pre else None,
                        "preanswer_features": extractor.encode_text_to_pooled_features(f"{prompt}\n{pre_text}" if pre_text else prompt),
                        # fixed prefixes and the end-of-prompt state
                        "prefix_grid": list(PREFIX_GRID), "prefix_features": prefix_feats, "prompt_end_state": prompt_end,
                        "prefix_texts": [tok.decode(gen_ids[:n], skip_special_tokens=True) for n in PREFIX_GRID],
                        "decode_roundtrip_ok": tok.decode(gen_ids, skip_special_tokens=True).strip() == text,
                    }
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
                    print(f"[run8] shard {s0}: {ti + 1}/{len(tasks[s0:s0 + shard])} tasks, {time.time() - t0:.0f}s", flush=True)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{task['task_uid']}: {type(exc).__name__}: {exc}")
                print(f"[run8] ERROR {task['task_uid']}: {exc}", flush=True)
                torch.cuda.empty_cache()
        torch.save({"shard_start": s0, "seed": SEED, "k": K, "max_new_tokens": MAX_NEW, "prefix_grid": list(PREFIX_GRID),
                    "model": f"{REPO}@{REVISION}", "model_sha256": model_sha, "smoke": a.smoke, "elapsed_seconds": round(time.time() - t0, 1),
                    "equivalence_checks_maxabs": checks, "errors": errors, "records": records}, path)
        print(f"[run8] wrote {path.name}: {len(records)} records, {len(errors)} errors, equivalence max "
              f"{max(checks) if checks else None}", flush=True)
    extractor.cleanup()


if __name__ == "__main__":
    main()
