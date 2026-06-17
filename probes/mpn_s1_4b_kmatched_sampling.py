"""M+N S1.4b — K-matched plain-sampling baseline (mandated confound check for the S1.4a screen).

S1.4a found "new correct on base-missed" ONLY in decode=sample cells (all 9 greedy/fork cells = 0.0),
~identical across α/token_range — i.e. the apparent reachability looks like plain SAMPLING diversity,
not the injection/fork. Per protocol, before any carry/fork claim we run the K-matched plain-sampling
baseline: NO fork, N plain samples per task (N = 3 loci × K = 12, matching the fork arm's branch budget),
same decode (temp 0.7 / top_p 0.95), MNT=96. We also decode base GREEDY at MNT=160 AND MNT=96 to remove
the screen's MNT-truncation confound (math was base-correct at 160 but truncated at 96).

DECISION:
  * If plain-sampling oracle / new_correct ≈ the screen's SAMPLE-fork cells (and fork-greedy was 0.0),
    then SAMPLING ALONE explains the screen win -> the deterministic FORK adds nothing beyond stochastic
    decoding -> frozen FORK branching is CLOSED for local purposes -> S3/backbone training is the lever.
  * If plain-sampling is clearly BELOW the sample-fork cells, the fork contributes reachability beyond
    sampling -> escalate to the chained loop at the winning regime.

Verifier LABELS only. No fork, no model mutation. N NOT scaled (same 4 tasks).

Run on GPU (backgrounded): venv/bin/python utilities/tests/manual/mpn_s1_4b_kmatched_sampling.py
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import branch_training_v1_common as B1  # noqa: E402
import branch_training_v2_common as V  # noqa: E402
import bg_autoregressive_cache_common_v1 as C  # noqa: E402
import bg_partial_cache_splice_v2_common as S  # noqa: E402
import bg_corecontent_v2_models as M  # noqa: E402
from mpn_s1_4_reference_loop import make_extractor_reusing  # noqa: E402
from mpn_s1_5_divergence_ablation import parse, verify, score_text, tap_scores  # noqa: E402

OUT = V.PROJECT_ROOT / "artifacts/reports/probes/mpn_s1_baseline_2026-06-13"
CANARY = V.DATA_ROOT / "canary/offline_branch_generator_canary_v2.jsonl"
DOMAINS = ("math", "reasoning", "logic", "coding")
N_PER = 1
N_SAMPLES = int(os.environ.get("S1_N_SAMPLES", "12"))   # = 3 loop-1 loci × K=4 (matches fork arm budget)
MNT = int(os.environ.get("S1_MNT", "96"))               # match screen sample cells
MNT_LONG = int(os.environ.get("S1_MNT_LONG", "160"))    # true base (ref-loop terminal length)
TEMP = float(os.environ.get("S1_TEMP", "0.7"))
TOP_P = float(os.environ.get("S1_TOP_P", "0.95"))


def greedy_text(model, tok, task, device, mnt):
    enc = tok(V.build_prompt(tok, task, "direct"), return_tensors="pt", truncation=True, max_length=1536).to(device)
    nl, cache, _cp, amf = C.prefill(model, enc["input_ids"], enc["attention_mask"])
    toks, _ = S.greedy_continue(model, cache, amf, nl, mnt, device)
    return tok.decode(toks, skip_special_tokens=True)


def sample_texts(model, tok, task, device, n, mnt, batch=None):
    """Plain sampling in small BATCHES — the ~12GB GPU OOMs on a 12-way parallel generate of the 4-loop UT
    model (12 simultaneous KV caches), so draw `batch` sequences at a time."""
    batch = batch or int(os.environ.get("S1_BATCH", "4"))
    enc = tok(V.build_prompt(tok, task, "direct"), return_tensors="pt", truncation=True, max_length=1536).to(device)
    plen = enc["input_ids"].shape[1]
    outs = []
    remaining = n
    with torch.inference_mode():
        while remaining > 0:
            b = min(batch, remaining)
            gen = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                 max_new_tokens=mnt, do_sample=True, temperature=TEMP, top_p=TOP_P,
                                 num_return_sequences=b, pad_token_id=tok.pad_token_id)
            for row in gen:
                outs.append(tok.decode(row[plen:], skip_special_tokens=True))
            remaining -= b
            C.clear_cuda()
    return outs


def screen_reference():
    """Pull the S1.4a sample-cell and greedy-cell aggregates for direct comparison."""
    p = OUT / "s1_4a_fork_param_screen.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    sample = [c for c in d["cells"] if c["decode"] == "sample"]
    greedy = [c for c in d["cells"] if c["decode"] == "greedy"]
    def mean(cells, k):
        vals = [c[k] for c in cells if c[k] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None
    return {"sample_oracle_mean": mean(sample, "oracle_over_branches"),
            "sample_new_correct_mean": mean(sample, "new_correct_rate_on_base_missed"),
            "greedy_oracle_mean": mean(greedy, "oracle_over_branches"),
            "greedy_new_correct_mean": mean(greedy, "new_correct_rate_on_base_missed")}


def main() -> int:
    started = time.time()
    sel = json.loads((OUT / "s1_1_task_set.json").read_text())
    want = {}
    for d in DOMAINS:
        for cid in [t["canary_id"] for t in sel["tasks"] if t["domain"] == d][:N_PER]:
            want[cid] = d
    tasks = [json.loads(l) for l in open(CANARY) if json.loads(l)["canary_id"] in want]

    model, tok, info = C.load_model(attn_implementation="eager")
    device = next(model.parameters()).device
    ext = make_extractor_reusing(model, tok, device)
    da_policy = M.baseline_policies(include_diag=True).get("DualAnchor")
    cc_policy = M.crafted_v2_policies().get("CoreContent_v2_blockwise")

    rows = []
    for task in tasks:
        tid = task["canary_id"]
        b160 = verify(task, greedy_text(model, tok, task, device, MNT_LONG))[0]
        b96 = verify(task, greedy_text(model, tok, task, device, MNT))[0]
        C.clear_cuda()
        texts = sample_texts(model, tok, task, device, N_SAMPLES, MNT)
        C.clear_cuda()
        finals, correct = [], []
        for t in texts:
            c, fin = verify(task, t)
            finals.append(fin or ""); correct.append(bool(c))
        da, cc = tap_scores(ext, task, texts, da_policy, cc_policy)
        sel_i = max(range(len(texts)), key=lambda i: (cc[i] if cc[i] == cc[i] else -1e9))
        rows.append({"task": tid, "domain": task["domain"],
                     "base_correct_160": bool(b160), "base_correct_96": bool(b96),
                     "n_samples": len(texts), "oracle": bool(any(correct)), "n_correct": int(sum(correct)),
                     "selected_correct": bool(correct[sel_i]), "unique_finals": len(set(finals)),
                     "parse_invalid": int(sum(1 for f in finals if not f.strip()))})
        print(f"  {task['domain']:9} {tid}: base160={int(b160)} base96={int(b96)} "
              f"oracle@{len(texts)}={int(rows[-1]['oracle'])} n_correct={rows[-1]['n_correct']} "
              f"sel={int(rows[-1]['selected_correct'])} uniq={rows[-1]['unique_finals']}", flush=True)

    n = len(rows)
    bm160 = [r for r in rows if not r["base_correct_160"]]
    bm96 = [r for r in rows if not r["base_correct_96"]]
    agg = {
        "base_acc_160": round(sum(r["base_correct_160"] for r in rows) / n, 3),
        "base_acc_96": round(sum(r["base_correct_96"] for r in rows) / n, 3),
        "sampling_oracle": round(sum(r["oracle"] for r in rows) / n, 3),
        "sampling_new_correct_vs_base160": (round(sum(r["oracle"] for r in bm160) / len(bm160), 3) if bm160 else None),
        "sampling_new_correct_vs_base96": (round(sum(r["oracle"] for r in bm96) / len(bm96), 3) if bm96 else None),
        "sampling_selected_acc": round(sum(r["selected_correct"] for r in rows) / n, 3),
        "mean_unique_finals": round(sum(r["unique_finals"] for r in rows) / n, 3),
        "parse_invalid_rate": round(sum(r["parse_invalid"] for r in rows) / sum(r["n_samples"] for r in rows), 3),
    }
    screen = screen_reference()
    # Decision: does the deterministic fork add anything OVER plain sampling?
    fork_vs_sampling = None
    if screen and screen["sample_oracle_mean"] is not None:
        fork_vs_sampling = round(screen["sample_oracle_mean"] - agg["sampling_oracle"], 3)
    decision = {
        "screen_sample_cells": screen,
        "plain_sampling_oracle": agg["sampling_oracle"],
        "sample_fork_minus_plain_sampling_oracle": fork_vs_sampling,
        "greedy_fork_new_correct": (screen["greedy_new_correct_mean"] if screen else None),
        "verdict": None, "note": None,
    }
    # The fork only "adds" if sample-fork oracle exceeds plain-sampling oracle by a margin (positive diff).
    # A NEGATIVE/near-zero diff (plain sampling >= fork) means the fork adds nothing — the stronger the
    # negative, the clearer. So: CLOSED iff greedy fork is null AND the fork does not exceed sampling.
    TOL = 0.13
    fork_adds = (fork_vs_sampling is not None and fork_vs_sampling > TOL)
    closed = bool(screen and (screen["greedy_new_correct_mean"] in (0.0, None)) and not fork_adds)
    decision["fork_adds_over_sampling"] = fork_adds
    decision["verdict"] = ("FROZEN_FORK_CLOSED__SAMPLING_EXPLAINS_SCREEN__S3_IS_LEVER" if closed
                           else "FORK_MAY_ADD_OVER_SAMPLING__ESCALATE_CHAINED_AT_WINNER")
    decision["note"] = ("greedy/deterministic fork = 0 new-correct AND plain-sampling oracle >= sample-fork "
                        "oracle (fork does NOT exceed sampling) -> the fork adds nothing beyond stochastic "
                        "decoding; the S1.4a 'win' is sampling. Frozen FORK branching closed for local "
                        "purposes; move to S3/backbone." if closed
                        else "sample-fork oracle exceeds plain-sampling oracle by > TOL -> fork may add "
                        "reachability beyond sampling; escalate to the chained loop at the winning regime.")
    payload = {"verdict": "S1_4B_KMATCHED_SAMPLING", "config": {"N_PER": N_PER, "N_SAMPLES": N_SAMPLES,
               "mnt": MNT, "mnt_long": MNT_LONG, "temp": TEMP, "top_p": TOP_P, "score_input_mode": "prompt_plus_answer"},
               "n_tasks": n, "aggregate": agg, "decision": decision, "rows": rows,
               "elapsed_seconds": round(time.time() - started, 1)}
    (OUT / "s1_4b_kmatched_sampling.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2))
    print(f"\n{decision['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
