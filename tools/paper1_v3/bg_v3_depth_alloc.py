"""V3 tap-gated loop allocation (SEALED PROTOCOL) -- frozen conversion #1.

Question: can a frozen hidden-state readout allocate per-input recurrent depth
better than (a) fixed depth and (b) Ouro's native early-exit gate, at matched
average-loop budgets? This drives the one actuator none of the paper's
interventions touched: the loop-count dial. It is also RLTT's stated future
work ("integrate adaptive halting"), done frozen.

Design (per model: thinking26, rltt):
  STAGE capture   One d=4 prefill per task over the PROMPT ONLY: pooled states
                  [5 layers x 4 loops x 2048] + the native exit pdf per token
                  (return_exit_pdf). No generation.
  STAGE generate  Fixed-depth generation table: d in {1,2,3,4} x k=2 samples,
                  max_new=448, T=0.7, per-(task,depth) seeds, verifier labels.
  STAGE analyze   Policies SELECT rows of the table, so every policy is
                  evaluated on identical generations at exact budgets:
                    - fixed-d (4 points)
                    - native-gate: sequence depth = per-token expected exit
                      step from the exit pdf, thresholded (sweep)
                    - tap: ridge allocator on loop-1 prompt features (trained
                      on train+val tasks only, labels = minimal sufficient
                      depth from the table), thresholded (sweep)
                    - random-histogram control: same depth histogram as the
                      tap policy, assignment shuffled (100 draws)
                  Metric: task success (any of k correct) vs average depth.

SEALED ENDPOINTS:
  Primary: at the native gate's own achieved budgets, paired task-clustered
  delta (tap - native) in success rate, 2000 rounds, seed 20260726.
  Gate: the tap policy must also beat the random-histogram control at matched
  histogram; if it does not, allocation is not signal-driven and the verdict
  is ALLOCATION_NOT_SIGNAL_DRIVEN regardless of the native-gate comparison.
  Verdicts: TAP_BEATS_NATIVE_GATE / TAP_MATCHES_NATIVE_GATE /
  TAP_WORSE_THAN_NATIVE_GATE / ALLOCATION_NOT_SIGNAL_DRIVEN /
  DEPTH_INSENSITIVE_POOL (fixed-depth curve flat: d1 within 0.02 of d4).
Task slice: offsets [680, 860) of the sealed hash-ordered pool (disjoint from
all prior runs); sealed split_for(task_uid) gives ~train/val/heldout.
Heldout is opened once, by the analyze stage, after policies are frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANUAL = PROJECT_ROOT / "utilities/tests/manual"
for p in (str(PROJECT_ROOT), str(MANUAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
from bg_steering_suite_lib import mcq_prompt, parse_mcq_answer  # noqa: E402
from bg_v2_overnight_horizon_generate import (  # noqa: E402
    load_task_pool, split_for, _sid, FINAL_MARKER, options_dict,
)

OUT_ROOT = PROJECT_ROOT / "artifacts/reports/depth_alloc_v3_20260726"
SEED = 20260726
TASK_OFFSET, N_TASKS = 680, 180
DEPTHS = (1, 2, 3, 4)
K = 2
MAX_NEW = 448
LAYERS = (8, 16, 24, 36, 47)
CAPTURE_IDX = {8: 7, 16: 15, 24: 23, 36: 35}

MODELS = {
    "thinking26": PROJECT_ROOT / "models/ouro_thinking_local",
    "rltt": PROJECT_ROOT / "models/ouro_rltt_local",
}


def tasks_slice():
    pool = load_task_pool()
    pool.sort(key=lambda d: hashlib.sha256(d["task_uid"].encode()).hexdigest())
    return pool[TASK_OFFSET: TASK_OFFSET + N_TASKS]


def load_model(key: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = MODELS[key]
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda").eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, tok


def set_depth(model, d: int) -> None:
    model.config.total_ut_steps = d
    model.config.early_exit_threshold = 1.0
    model.model.total_ut_steps = d
    if hasattr(model.model, "config"):
        model.model.config.total_ut_steps = d
        model.model.config.early_exit_threshold = 1.0


def stage_capture(key: str) -> None:
    out = OUT_ROOT / key
    out.mkdir(parents=True, exist_ok=True)
    fpath = out / "prompt_capture.pt"
    if fpath.exists():
        print("capture exists; skipping")
        return
    model, tok = load_model(key)
    set_depth(model, 4)
    recs = []
    inter = {idx: [] for idx in CAPTURE_IDX.values()}
    boundary = []

    def mk_hook(idx):
        def _h(_m, _i, o):
            inter[idx].append((o[0] if isinstance(o, (tuple, list)) else o).detach())
        return _h

    gates = []

    def bh(_m, _i, o):
        if isinstance(o, (tuple, list)) and len(o) >= 2:
            boundary.extend(list(o[1]))
        if isinstance(o, (tuple, list)) and len(o) >= 3 and o[2]:
            gates.extend([g.detach() for g in o[2]])

    handles = [model.model.register_forward_hook(bh)]
    for idx in CAPTURE_IDX.values():
        handles.append(model.model.layers[idx].register_forward_hook(mk_hook(idx)))

    t0 = time.time()
    for ti, task in enumerate(tasks_slice()):
        prompt = mcq_prompt(task["task_prompt"], options_dict(task["options"]))
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1536).to("cuda")
        for v in inter.values():
            v.clear()
        boundary.clear()
        gates.clear()
        with torch.inference_mode():
            model(**enc, use_cache=False)
        mask = enc["attention_mask"]
        pooled = []
        for layer in LAYERS:
            states = boundary if layer == 47 else inter[CAPTURE_IDX[layer]]
            assert len(states) == 4, f"layer {layer}: {len(states)} loop states"
            pooled.append(torch.stack(
                [((s.float() * mask.unsqueeze(-1)).sum(1) / mask.sum(1).unsqueeze(-1)).squeeze(0).cpu()
                 for s in states], 0))
        feats = torch.stack(pooled, 0)          # [5,4,2048]
        # Native exit pdf from the gate logits, exactly the modeling-code
        # survival chain: p_i = sigmoid(g_i) * prod_{j<i}(1 - sigmoid(g_j)),
        # last step takes the remaining mass. Mean-pooled over prompt tokens.
        ep = None
        if len(gates) == 4:
            lam = [torch.sigmoid(g.float().squeeze(-1)).squeeze(0) for g in gates]
            remaining = torch.ones_like(lam[0])
            pdf = []
            for i in range(4):
                if i < 3:
                    pdf.append(lam[i] * remaining)
                    remaining = remaining * (1.0 - lam[i])
                else:
                    pdf.append(remaining)
            m = mask.squeeze(0).float().cpu()
            ep = torch.stack([(pi.cpu() * m).sum() / m.sum() for pi in pdf])  # [4]
        recs.append({"task_uid": task["task_uid"], "split": split_for(task["task_uid"]),
                     "gold": task["gold_answer"], "features": feats.to(torch.float16),
                     "exit_pdf": ep, "prompt_tokens": int(mask.sum())})
        if ti % 40 == 0:
            print(f"capture {ti}/{N_TASKS} elapsed={time.time()-t0:.0f}s", flush=True)
    for h in handles:
        h.remove()
    torch.save({"records": recs, "exit_pdf_available": recs[0]["exit_pdf"] is not None},
               fpath)
    print(f"capture done: {len(recs)} tasks, exit_pdf={'YES' if recs[0]['exit_pdf'] is not None else 'NO'}")


def stage_generate(key: str, depth: int) -> None:
    out = OUT_ROOT / key
    out.mkdir(parents=True, exist_ok=True)
    fpath = out / f"gen_d{depth}.pt"
    if fpath.exists():
        print(f"gen_d{depth} exists; skipping")
        return
    model, tok = load_model(key)
    set_depth(model, depth)
    from transformers import StoppingCriteria, StoppingCriteriaList

    class Stop(StoppingCriteria):
        def __init__(self):
            self.hits = 0
        def __call__(self, ids, scores, **kw):
            self.hits += 1
            return False

    recs = []
    t0 = time.time()
    for ti, task in enumerate(tasks_slice()):
        opts = options_dict(task["options"])
        gold_letter = chr(65 + int(task["answer_key"]))
        prompt = mcq_prompt(task["task_prompt"], opts)
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1536).to("cuda")
        seed_i = _sid(SEED, task["task_uid"], depth)
        torch.manual_seed(seed_i)
        with torch.inference_mode():
            gen = model.generate(**enc, do_sample=True, temperature=0.7, top_p=0.95,
                                 num_return_sequences=K, max_new_tokens=MAX_NEW,
                                 use_cache=True, pad_token_id=tok.pad_token_id)
        cands = []
        for k_i in range(K):
            text = tok.decode(gen[k_i][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            found = bool(FINAL_MARKER.search(text))
            parsed = parse_mcq_answer(text, opts)
            cands.append({"success": bool(found and parsed is not None and parsed == gold_letter),
                          "malformed": (not found) or parsed is None,
                          "n_tokens": int((gen[k_i] != tok.pad_token_id).sum())})
        recs.append({"task_uid": task["task_uid"], "split": split_for(task["task_uid"]),
                     "depth": depth, "candidates": cands,
                     "any_success": any(c["success"] for c in cands)})
        del gen
        torch.cuda.empty_cache()
        if ti % 25 == 0:
            print(f"gen d{depth} {ti}/{N_TASKS} elapsed={time.time()-t0:.0f}s", flush=True)
    torch.save({"records": recs, "depth": depth, "k": K, "seed": SEED}, fpath)
    n_s = sum(r["any_success"] for r in recs)
    print(f"gen d{depth} done: any-success {n_s}/{len(recs)} in {time.time()-t0:.0f}s")


def stage_analyze(key: str) -> None:
    import numpy as np
    out = OUT_ROOT / key
    cap = torch.load(out / "prompt_capture.pt", map_location="cpu", weights_only=False)
    caps = {r["task_uid"]: r for r in cap["records"]}
    table = {}
    for d in DEPTHS:
        payload = torch.load(out / f"gen_d{d}.pt", map_location="cpu", weights_only=False)
        for r in payload["records"]:
            table[(r["task_uid"], d)] = r
    uids = [r["task_uid"] for r in cap["records"]]
    split = {u: caps[u]["split"] for u in uids}
    trainval = [u for u in uids if split[u] in ("train", "val")]
    heldout = [u for u in uids if split[u] == "heldout"]
    succ = {(u, d): table[(u, d)]["any_success"] for u in uids for d in DEPTHS}

    # fixed-depth curves (heldout)
    fixed = {d: float(np.mean([succ[(u, d)] for u in heldout])) for d in DEPTHS}
    print("fixed-depth heldout success:", fixed)
    depth_insensitive = (fixed[4] - fixed[1]) < 0.02

    # ----- tap allocator: ridge on loop-1 features (layers 8+16+24 concat) ----
    def feat(u):
        f = caps[u]["features"].float()          # [5,4,2048]
        return torch.cat([f[i, 0] for i in range(3)]).numpy()   # loop-1, layers 8/16/24

    def minimal_depth(u):
        for d in DEPTHS:
            if succ[(u, d)]:
                return d
        return 4

    Xtr = np.stack([feat(u) for u in trainval])
    ytr = np.array([minimal_depth(u) for u in trainval], dtype=float)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_n = (Xtr - mu) / sd
    lam = 10.0
    w = np.linalg.solve(Xtr_n.T @ Xtr_n + lam * np.eye(Xtr_n.shape[1]), Xtr_n.T @ (ytr - ytr.mean()))
    def tap_score(u):
        return float(((feat(u) - mu) / sd) @ w)   # higher = needs more depth

    # ----- native gate score: expected exit step over prompt tokens ----------
    have_gate = cap.get("exit_pdf_available", False)
    def gate_score(u):
        ep = caps[u]["exit_pdf"]
        if ep is None:
            return None
        e = ep.float()                            # [4] mean pdf over prompt tokens
        steps = torch.arange(1, e.shape[-1] + 1).float()
        return float((e * steps).sum())

    # ----- policies over budget sweep -----------------------------------------
    rng = np.random.default_rng(SEED)
    def policy_curve(score_fn, name):
        scores = {u: score_fn(u) for u in heldout}
        if any(v is None for v in scores.values()):
            return None
        vals = sorted(scores.values())
        pts = []
        for q in np.linspace(0.05, 0.95, 19):
            lo_t = np.quantile(vals, max(0.0, q - 0.30))
            hi_t = np.quantile(vals, min(1.0, q + 0.30))
            # map score to depth via three thresholds at quantiles around q
            t1, t2, t3 = (np.quantile(vals, min(1.0, max(0.0, q + off)))
                          for off in (-0.25, 0.0, 0.25))
            alloc = {u: (1 if scores[u] <= t1 else 2 if scores[u] <= t2
                         else 3 if scores[u] <= t3 else 4) for u in heldout}
            acc = float(np.mean([succ[(u, alloc[u])] for u in heldout]))
            avg_d = float(np.mean(list(alloc.values())))
            pts.append({"avg_depth": avg_d, "success": acc,
                        "hist": {d: sum(1 for a in alloc.values() if a == d) for d in DEPTHS},
                        "alloc": alloc})
        return pts

    tap_curve = policy_curve(tap_score, "tap")
    gate_curve = policy_curve(gate_score, "gate") if have_gate else None

    # random-histogram control for each tap point (100 draws)
    def random_control(pts):
        out_pts = []
        for p in pts:
            depths_list = [d for d, c in p["hist"].items() for _ in range(c)]
            accs = []
            for _ in range(100):
                perm = rng.permutation(len(heldout))
                alloc = {heldout[i]: depths_list[perm[i]] for i in range(len(heldout))}
                accs.append(np.mean([succ[(u, alloc[u])] for u in heldout]))
            out_pts.append({"avg_depth": p["avg_depth"],
                            "random_mean": float(np.mean(accs)),
                            "random_p95": float(np.percentile(accs, 95))})
        return out_pts

    rand_curve = random_control(tap_curve)

    # paired clustered bootstrap: tap vs gate at nearest-budget points
    def paired(tapP, gateP):
        alloc_a, alloc_b = tapP["alloc"], gateP["alloc"]
        ya = np.array([float(succ[(u, alloc_a[u])]) for u in heldout])
        yb = np.array([float(succ[(u, alloc_b[u])]) for u in heldout])
        deltas = []
        for _ in range(2000):
            idx = rng.choice(len(heldout), len(heldout), replace=True)
            deltas.append(float(np.mean(ya[idx] - yb[idx])))
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        return {"tap_budget": tapP["avg_depth"], "gate_budget": gateP["avg_depth"],
                "tap_success": tapP["success"], "gate_success": gateP["success"],
                "delta": round(float(np.mean(ya - yb)), 4),
                "ci95": [round(float(lo), 4), round(float(hi), 4)],
                "excludes_zero": bool(lo > 0 or hi < 0)}

    comparisons = []
    if gate_curve:
        for gp in gate_curve[::3]:
            tp = min(tap_curve, key=lambda p: abs(p["avg_depth"] - gp["avg_depth"]))
            if abs(tp["avg_depth"] - gp["avg_depth"]) <= 0.25:
                comparisons.append(paired(tp, gp))

    # verdict per sealed labels
    tap_beats_random = any(
        p["success"] > r["random_p95"] + 1e-9
        for p, r in zip(tap_curve, rand_curve))
    if depth_insensitive:
        verdict = "DEPTH_INSENSITIVE_POOL"
    elif not tap_beats_random:
        verdict = "ALLOCATION_NOT_SIGNAL_DRIVEN"
    elif comparisons and any(c["excludes_zero"] and c["delta"] > 0 for c in comparisons):
        verdict = "TAP_BEATS_NATIVE_GATE"
    elif comparisons and any(c["excludes_zero"] and c["delta"] < 0 for c in comparisons):
        verdict = "TAP_WORSE_THAN_NATIVE_GATE"
    elif comparisons:
        verdict = "TAP_MATCHES_NATIVE_GATE"
    else:
        verdict = "NATIVE_GATE_UNAVAILABLE_TAP_VS_FIXED_ONLY"

    results = {
        "model": key, "seed": SEED, "n_tasks": len(uids),
        "n_heldout_tasks": len(heldout), "k": K,
        "fixed_depth_heldout_success": fixed,
        "tap_curve": [{k2: v for k2, v in p.items() if k2 != "alloc"} for p in tap_curve],
        "gate_curve": ([{k2: v for k2, v in p.items() if k2 != "alloc"} for p in gate_curve]
                       if gate_curve else None),
        "random_histogram_control": rand_curve,
        "tap_vs_gate_matched_budget": comparisons,
        "exit_pdf_available": have_gate,
        "verdict": verdict,
    }
    (out / "depth_alloc_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"[{key}] VERDICT: {verdict}")
    for c in comparisons:
        print(f"  budget~{c['gate_budget']:.2f}: tap {c['tap_success']:.3f} vs "
              f"gate {c['gate_success']:.3f} · Δ {c['delta']:+.3f} {c['ci95']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--stage", required=True,
                    choices=["capture", "generate", "analyze"])
    ap.add_argument("--depth", type=int, default=None)
    args = ap.parse_args()
    if args.stage == "capture":
        stage_capture(args.model)
    elif args.stage == "generate":
        assert args.depth in DEPTHS
        stage_generate(args.model, args.depth)
    else:
        stage_analyze(args.model)


if __name__ == "__main__":
    main()
