"""Cross-loop early-layer v1 — S3B2 secondary readout (generated-branch correctness).

Small-n by construction: 16 tasks x 10 generated branches, labels = saved verifier
correctness (no generation run here). Two modes per (layer, loop) cell:

A. zero-train transfer: the primary CoreContent tap for that cell scores branches
   (tests whether the early-layer quality direction transfers to generated branches);
B. leave-one-task-out local refit: same antisym pairwise head, fixed hyperparameters
   (lr 1e-3, seed 0, 60 epochs), trained on within-task differing-label pairs
   (tests whether the information exists at that locus in this distribution).

Reported: pooled AUROC (per-domain and overall excluding no-positive domains), within-task
pairwise accuracy, sel@oracle, task-clustered bootstrap CIs. Explicitly exploratory.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bg_xloop_early_v1_common as X

S3B1_TEXTS = X.PROJECT_ROOT / "artifacts/reports/probes/mpn_s3b_2026-06-17/s3b1_pool_texts.json"
S3B2_MAXLEN = {"coding": 1024, "math": 768, "logic": 640, "reasoning": 640}
FEATS_PT = X.RUN_ROOT / "s3b2_features.pt"


def load_items():
    data = json.loads(S3B1_TEXTS.read_text())
    items = []
    for task_id, rec in sorted(data.items()):
        for bi, b in enumerate(rec["branches"]):
            items.append({"task_id": task_id, "domain": rec["domain"], "branch_idx": bi,
                          "text": b["text"], "correct": bool(b["correct"]),
                          "provenance": b.get("provenance")})
    return items


def extract() -> None:
    if FEATS_PT.exists():
        print("s3b2 features exist; reusing", flush=True)
        return
    from bg_xloop_early_v1_extract import ExtractorX
    items = load_items()
    ext = ExtractorX(device="cuda", dtype="auto", force_all_loops=True)
    t0 = time.time()
    try:
        for it in items:
            f = ext.encode_multi_layer(it["text"], max_length=S3B2_MAXLEN.get(it["domain"], 768))
            it["features"] = f.to(torch.float16)
    finally:
        try:
            ext.cleanup()
        except Exception:
            pass
    torch.save({"items": items, "layout": "(5,4,2048) layers[8,16,24,36,47] x loops[1..4]"}, FEATS_PT)
    print(f"s3b2 extraction: {len(items)} branches in {time.time() - t0:.0f}s", flush=True)


def auroc(scores, labels) -> float:
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def cell_metrics(items, w: torch.Tensor, layer: int, loop: int):
    by_task = defaultdict(list)
    for it in items:
        s = float(torch.dot(w.flatten().to(torch.float32), X.cell_vector(it, layer, loop)))
        by_task[it["task_id"]].append((s, it["correct"], it["domain"]))
    task_rows = []
    all_s, all_l = [], []
    for tid, rows in sorted(by_task.items()):
        ss = [r[0] for r in rows]
        ll = [r[1] for r in rows]
        dom = rows[0][2]
        all_s.extend(ss)
        all_l.extend(ll)
        pw_c = pw_t = 0
        for i in range(len(ss)):
            for j in range(len(ss)):
                if ll[i] and not ll[j]:
                    pw_t += 1
                    pw_c += 1.0 if ss[i] > ss[j] else 0.5 if ss[i] == ss[j] else 0.0
        top = max(range(len(ss)), key=lambda i: ss[i])
        task_rows.append({"task_id": tid, "domain": dom, "n": len(ss), "n_pos": sum(ll),
                          "auroc": auroc(ss, ll),
                          "pairwise": (pw_c / pw_t) if pw_t else float("nan"),
                          "sel_oracle": (1.0 if ll[top] else 0.0) if any(ll) else float("nan")})
    overall = {"auroc_pooled": auroc(all_s, all_l),
               "pairwise_task_mean": X.finite_mean(r["pairwise"] for r in task_rows),
               "sel_oracle_task_mean": X.finite_mean(r["sel_oracle"] for r in task_rows)}
    return overall, task_rows


def loto_refit(items, layer: int, loop: int) -> tuple[dict, list]:
    tasks = sorted({it["task_id"] for it in items})
    scored = []
    for held in tasks:
        train = [it for it in items if it["task_id"] != held]
        test = [it for it in items if it["task_id"] == held]
        diffs = []
        by_task = defaultdict(list)
        for it in train:
            by_task[it["task_id"]].append(it)
        for tid, rows in by_task.items():
            feats = [X.cell_vector(it, layer, loop) for it in rows]
            for i, a in enumerate(rows):
                for j, b in enumerate(rows):
                    if a["correct"] and not b["correct"]:
                        diffs.append(feats[i] - feats[j])
        if len(diffs) < 8:
            continue
        Xd = torch.stack(diffs, 0)
        torch.manual_seed(0)
        w = torch.zeros(1, Xd.shape[1], requires_grad=True)
        opt = torch.optim.Adam([w], lr=1e-3, weight_decay=0.01)
        for _ in range(60):
            opt.zero_grad()
            loss = torch.nn.functional.softplus(-(Xd @ w.t()).squeeze(1)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([w], 1.0)
            opt.step()
        wd = w.detach()
        for it in test:
            scored.append(dict(it, score=float(torch.dot(wd.flatten(), X.cell_vector(it, layer, loop)))))
    by_task = defaultdict(list)
    for it in scored:
        by_task[it["task_id"]].append((it["score"], it["correct"], it["domain"]))
    task_rows = []
    all_s = [it["score"] for it in scored]
    all_l = [it["correct"] for it in scored]
    for tid, rows in sorted(by_task.items()):
        ss, ll = [r[0] for r in rows], [r[1] for r in rows]
        pw_c = pw_t = 0
        for i in range(len(ss)):
            for j in range(len(ss)):
                if ll[i] and not ll[j]:
                    pw_t += 1
                    pw_c += 1.0 if ss[i] > ss[j] else 0.5 if ss[i] == ss[j] else 0.0
        top = max(range(len(ss)), key=lambda i: ss[i])
        task_rows.append({"task_id": tid, "domain": rows[0][2], "n": len(ss), "n_pos": sum(ll),
                          "auroc": auroc(ss, ll),
                          "pairwise": (pw_c / pw_t) if pw_t else float("nan"),
                          "sel_oracle": (1.0 if ll[top] else 0.0) if any(ll) else float("nan")})
    overall = {"auroc_pooled": auroc(all_s, all_l),
               "pairwise_task_mean": X.finite_mean(r["pairwise"] for r in task_rows),
               "sel_oracle_task_mean": X.finite_mean(r["sel_oracle"] for r in task_rows)}
    return overall, task_rows


def task_boot_ci(task_rows, field, draws=2000, seed=X.BOOT_SEED):
    vals = np.array([r[field] for r in task_rows if np.isfinite(r[field])])
    if len(vals) < 3:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(draws, len(vals)))
    means = vals[idx].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def main() -> int:
    t0 = time.time()
    extract()
    payload = torch.load(FEATS_PT, map_location="cpu", weights_only=False)
    items = payload["items"]
    for it in items:
        it["features"] = it["features"].to(torch.float32)
    taps = torch.load(X.RUN_ROOT / "xloop_taps.pt", map_location="cpu", weights_only=False)

    out = {"n_items": len(items), "n_tasks": len({it['task_id'] for it in items}),
           "labels": "saved verifier correctness (no generation run)",
           "note": "SMALL-N EXPLORATORY SECONDARY; 16 tasks, 160 branches; coding has 0 positives",
           "transfer": {}, "loto_refit": {}}
    for (layer, loop) in X.REFIT_CELLS:
        cn = X.cell_name(layer, loop)
        ov, rows = cell_metrics(items, taps[cn]["weight"], layer, loop)
        ov["pairwise_ci95_task_boot"] = task_boot_ci(rows, "pairwise")
        out["transfer"][cn] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ov.items()}
        X.write_json(X.PRED_ROOT / f"s3b2_transfer_{cn}.json", rows)
    for (layer, loop) in X.REFIT_CELLS:
        cn = X.cell_name(layer, loop)
        ov, rows = loto_refit(items, layer, loop)
        ov["pairwise_ci95_task_boot"] = task_boot_ci(rows, "pairwise")
        out["loto_refit"][cn] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ov.items()}
        X.write_json(X.PRED_ROOT / f"s3b2_loto_{cn}.json", rows)
        print(f"  s3b2 {cn}: transfer_auroc={out['transfer'][cn]['auroc_pooled']} "
              f"loto_auroc={ov['auroc_pooled']:.4f}", flush=True)
    out["elapsed_seconds"] = round(time.time() - t0, 1)
    X.write_json(X.RUN_ROOT / "s3b2_secondary.json", out)
    print("s3b2 secondary written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
