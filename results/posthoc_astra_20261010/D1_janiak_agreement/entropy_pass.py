"""Per-answer token entropy for D1 (PLAN.md section 4). NEEDS THE GPU; written 2026-10-10 and NOT run yet.

One bf16 forward pass of Ouro-RLTT (all 4 loops, no early exit, eager attention, logits forced from the last loop)
over prompt_ids + gen_ids of every answer in Runs 4, 6 and 1 (the exact stored token ids; no re-tokenization). Per
answer it saves the mean next-token entropy at temperature 1 over the whole generation and over the first N*
generated tokens (the distributions the first N* tokens were sampled from, i.e. logits at positions
plen-1 .. plen+N*-2, so the window entropy uses no token past the cut).

Diagnostic (not a gate): `lp_check_mad`, the mean |difference| between the stored gen_logprobs (log-probabilities
under the sampling distribution: temperature 0.7, top-k from the model's generation config, top-p 0.95, as HF
generate applied them) and the same quantity recomputed from this pass. Small values (bf16 kernel differences
between cached decoding and a full pass) confirm the token alignment; an off-by-one would give values of order 1+.

Resumable: one part file per shard in entropy/parts/; existing parts are skipped. When every shard of every
requested run has a part, they are merged into entropy/entropy_ouro_rltt.npz, which analysis.py picks up.
Batches are right-padded without an attention mask (causal, so padding after a sequence cannot reach it; this is
how the run code ran its feature pass).

usage: python -B entropy_pass.py [--runs run4 run6 run1] [--tokens-per-batch 4096] [--limit-shards N]
"""
import argparse
import glob
import hashlib
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from extract import RUNS  # noqa: E402  (shard globs and N*)

MODEL = "/home/moloch/ouro_project/shared/models/ouro_rltt_local"
WEIGHTS_SHA256 = {  # as cloud/setup_pod.sh checks them
    "model-00001.safetensors": "9f49e55b0d0ea368f2942cfd3fa256836ea7fa9002c6f860ca2d85ac1de252a1",
    "model-00002.safetensors": "97a0d531a2c624dd98e60c0ebc46a7cc0413642f8cba6df7dac8df01a432df19",
    "model-00003.safetensors": "0bf3d0c111c876ccc7f7ebaeb7cabc8dfccffde30acb4cd780a4c364d3679aad",
}
NUM_LOOPS, TEMP, TOP_P = 4, 0.7, 0.95
OUT = os.path.join(HERE, "entropy")
PARTS = os.path.join(OUT, "parts")


def load_model():
    for f, h in WEIGHTS_SHA256.items():
        with open(os.path.join(MODEL, f), "rb") as fh:
            got = hashlib.file_digest(fh, "sha256").hexdigest()
        assert got == h, f"{f}: weights changed"
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                                 local_files_only=True, attn_implementation="eager").to("cuda").eval()
    model.config.total_ut_steps, model.config.early_exit_threshold = NUM_LOOPS, 1.0  # as BGTransformerFeatureExtractor
    if hasattr(model, "model") and hasattr(model.model, "total_ut_steps"):
        model.model.total_ut_steps = NUM_LOOPS
    assert model.config._attn_implementation == "eager"
    return model


def sampling_warpers(model):
    from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
    top_k = getattr(model.generation_config, "top_k", None)
    ws = [TemperatureLogitsWarper(TEMP)] + ([TopKLogitsWarper(top_k)] if top_k else []) + [TopPLogitsWarper(TOP_P)]
    return LogitsProcessorList(ws), top_k


@torch.inference_mode()
def score_batch(model, warp, items, pad_id):
    """items: list of (prompt_ids, gen_ids, gen_logprobs, n_star). Returns [(ent_full, ent_pre, n_pre, lp_check_mad)]."""
    L = max(len(p) + len(g) for p, g, _, _ in items)
    x = torch.full((len(items), L), pad_id, dtype=torch.long)
    for b, (p, g, _, _) in enumerate(items):
        x[b, :len(p) + len(g)] = torch.tensor(p + g)
    x = x.to(next(model.parameters()).device)
    logits = model(input_ids=x, use_cache=False, exit_at_step=NUM_LOOPS - 1).logits  # [B, L, V], last loop
    out = []
    for b, (p, g, lp, n_star) in enumerate(items):
        if not g:
            out.append((np.nan, np.nan, 0, np.nan))
            continue
        z = logits[b, len(p) - 1: len(p) - 1 + len(g)].float()  # row j predicts generated token j
        logp = torch.log_softmax(z, dim=-1)
        ent = -(logp.exp() * logp).sum(-1)  # temperature 1
        n_pre = min(n_star, len(g))
        tgt = torch.tensor(g, device=z.device)
        warped = torch.log_softmax(warp(tgt[None], z), dim=-1).gather(1, tgt[:, None]).squeeze(1)
        stored = torch.tensor(lp[:len(g)], device=z.device)
        ok = torch.isfinite(warped)
        mad = float((warped[ok] - stored[ok]).abs().mean()) if ok.any() else np.inf
        out.append((float(ent.mean()), float(ent[:n_pre].mean()), n_pre, mad))
        del z, logp, ent, warped
    del logits
    return out


def run_shard(model, warp, pad_id, run, path, n_star, budget):
    d = torch.load(path, map_location="cpu", weights_only=False)
    recs = [(r.get("task_uid", r.get("task_id")), int(r["candidate_idx"]), list(r["prompt_ids"]), list(r["gen_ids"]),
             list(r["gen_logprobs"])) for r in d["records"]]
    del d
    order = sorted(range(len(recs)), key=lambda i: len(recs[i][2]) + len(recs[i][3]))
    res, batch, ntok = [None] * len(recs), [], 0
    for i in order + [None]:
        L = None if i is None else len(recs[i][2]) + len(recs[i][3])
        if batch and (i is None or L * (len(batch) + 1) > budget):
            scored = score_batch(model, warp, [(recs[j][2], recs[j][3], recs[j][4], n_star) for j in batch], pad_id)
            for j, s in zip(batch, scored):
                res[j] = s
            batch = []
        if i is not None:
            batch.append(i)
            ntok += L
    os.makedirs(PARTS, exist_ok=True)
    tmp = os.path.join(PARTS, f"{run}__{os.path.basename(path)}.tmp.npz")
    np.savez(tmp, run=np.array([run] * len(recs)), qid=np.array([r[0] for r in recs]), cand=np.array([r[1] for r in recs]),
             ent_full=np.array([s[0] for s in res]), ent_pre=np.array([s[1] for s in res]),
             n_pre=np.array([s[2] for s in res]), lp_check_mad=np.array([s[3] for s in res]))
    os.replace(tmp, os.path.join(PARTS, f"{run}__{os.path.basename(path)}.npz"))
    return len(recs), ntok, float(np.nanmedian([s[3] for s in res]))


def merge(runs):
    want = [os.path.join(PARTS, f"{run}__{os.path.basename(p)}.npz") for run in runs for p in sorted(glob.glob(RUNS[run][0]))]
    if not all(os.path.exists(w) for w in want):
        print(f"not merging: {sum(map(os.path.exists, want))}/{len(want)} parts present")
        return
    zs = [np.load(w) for w in want]
    keys = ("run", "qid", "cand", "ent_full", "ent_pre", "n_pre", "lp_check_mad")
    merged = {k: np.concatenate([z[k] for z in zs]) for k in keys}
    np.savez(os.path.join(OUT, "entropy_ouro_rltt.npz"), **merged)
    print(f"merged {len(want)} parts, {len(merged['qid'])} answers -> entropy/entropy_ouro_rltt.npz; "
          f"lp_check_mad median {np.nanmedian(merged['lp_check_mad']):.4f}, 99th pct {np.nanpercentile(merged['lp_check_mad'], 99):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["run4", "run6", "run1"])
    ap.add_argument("--tokens-per-batch", type=int, default=4096, help="padded tokens per forward; lower it on OOM")
    ap.add_argument("--limit-shards", type=int, default=None, help="smoke test: only the first N shards per run")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "entropy_pass.py needs the GPU"
    model = load_model()
    warp, top_k = sampling_warpers(model)
    pad_id = model.config.pad_token_id if model.config.pad_token_id is not None else model.config.eos_token_id
    print(f"model loaded (eager attention, {NUM_LOOPS} loops, last-loop logits); sampling warpers T={TEMP} top_k={top_k} top_p={TOP_P}", flush=True)
    t0, done_tok = time.time(), 0
    for run in a.runs:
        pattern, n_star, _ = RUNS[run]
        for path in sorted(glob.glob(pattern))[: a.limit_shards]:
            if os.path.exists(os.path.join(PARTS, f"{run}__{os.path.basename(path)}.npz")):
                print(f"[{run}] {os.path.basename(path)} done, skipping", flush=True)
                continue
            n, ntok, mad = run_shard(model, warp, pad_id, run, path, n_star, a.tokens_per_batch)
            done_tok += ntok
            el = time.time() - t0
            print(f"[{run}] {os.path.basename(path)}: {n} answers, {ntok} tokens, lp_check_mad median {mad:.4f}; "
                  f"{done_tok / el:.0f} tok/s overall, {el / 60:.1f} min elapsed", flush=True)
    if a.limit_shards is None:
        merge(a.runs)


if __name__ == "__main__":
    main()
