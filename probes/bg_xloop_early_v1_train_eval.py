"""Cross-loop early-layer v1 — local-refit matrix, frozen-transfer matrix, controls.

Consumes this run's (5,4,2048) feature shards. Training is the v2 `train_pairwise`
procedure verbatim (same grid, selection on val macro top-1 only). Heldout is touched
exactly once per selected tap for final metrics + raw per-group predictions.
"""
from __future__ import annotations

import hashlib
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bg_xloop_early_v1_common as X

MAX_PAIRS_PER_GROUP = 24
EPOCHS = 60


# ----------------------------------------------------------------- training (v2 verbatim)
def pairwise_diffs(groups, layer: int, loop: int) -> list[torch.Tensor]:
    diffs = []
    for g in groups:
        cands = g["candidates"]
        if len(cands) < 2:
            continue
        feats = torch.stack([X.cell_vector(c, layer, loop) for c in cands], 0)
        rewards = [float(c["reward"]) for c in cands]
        pairs = [(i, j) for i in range(len(rewards)) for j in range(len(rewards)) if rewards[i] > rewards[j]]
        for (i, j) in pairs[:MAX_PAIRS_PER_GROUP]:
            diffs.append(feats[i] - feats[j])
    return diffs


_DIFF_CACHE: dict[tuple, dict[str, torch.Tensor]] = {}


def diffs_for_cell(train_by: dict, layer: int, loop: int) -> dict[str, torch.Tensor]:
    key = (id(train_by), layer, loop)
    if key not in _DIFF_CACHE:
        per = {d: pairwise_diffs(train_by.get(d, []), layer, loop) for d in X.CORE_DOMAINS}
        _DIFF_CACHE[key] = {d: torch.stack(v, 0) for d, v in per.items() if v}
        if len(_DIFF_CACHE) > 40:  # two matrices' worth; avoid unbounded growth
            _DIFF_CACHE.pop(next(iter(_DIFF_CACHE)))
    return _DIFF_CACHE[key]


def train_pairwise(train_by: dict, layer: int, loop: int, tset: str, lr: float, seed: int) -> torch.Tensor | None:
    torch.manual_seed(seed)
    cached = diffs_for_cell(train_by, layer, loop)
    per = {d: cached[d] for d in X.TRAINING_SETS[tset] if d in cached}
    if not per:
        return None
    balanced = tset == "all_core_balanced"
    if balanced and len(per) > 1:
        m = min(x.shape[0] for x in per.values())
        gen = torch.Generator().manual_seed(seed)
        Xd = torch.cat([x[torch.randperm(x.shape[0], generator=gen)[:m]] for x in per.values()], 0)
    else:
        Xd = torch.cat(list(per.values()), 0)
    if Xd.shape[0] < 8:
        return None
    w = torch.zeros(1, Xd.shape[1], requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr, weight_decay=0.01)
    for _ in range(EPOCHS):
        opt.zero_grad()
        s = (Xd @ w.t()).squeeze(1)
        loss = torch.nn.functional.softplus(-s).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([w], 1.0)
        opt.step()
    return w.detach().clone()


# NOTE (v2 fidelity): in the v2 code `balanced=True` for every grid entry, so
# "all_core" and "all_core_balanced" are duplicates there. We keep both grid entries but
# make `all_core` unbalanced — a superset in spirit; selection still uses val only.
# Recorded here so the deviation is explicit.


def eval_cell(groups_by_dom: dict, weight: torch.Tensor, layer: int, loop: int):
    """Returns (macro_top1, per_domain dict, per-group rows with raw scores)."""
    per_dom = {}
    rows = []
    for d, gs in groups_by_dom.items():
        ms = []
        for g in gs:
            scores = X.score_group_cell(g, weight, layer, loop)
            m = X.group_selection_metrics(g, scores)
            if not m:
                continue
            ms.append(m)
            rows.append({"domain": d, "group_uid": g["group_uid"], "task_uid": g["task_uid"],
                         "n_candidates": len(g["candidates"]),
                         "candidate_uids": [c["candidate_uid"] for c in g["candidates"]],
                         "rewards": [float(c["reward"]) for c in g["candidates"]],
                         "scores": [round(s, 6) for s in scores],
                         "top1_oracle": m["top1_oracle"], "pairwise_accuracy": m["pairwise_accuracy"],
                         "pairwise_pairs": m["pairwise_pairs"], "regret": m["regret"],
                         "chance_top1": X.chance_top1(g)})
        per_dom[d] = {"top1": X.finite_mean(m["top1_oracle"] for m in ms),
                      "pairwise": X.finite_mean(m["pairwise_accuracy"] for m in ms),
                      "groups": len(ms)}
    macro = X.finite_mean(per_dom[d]["top1"] for d in groups_by_dom)
    return macro, per_dom, rows


def weight_hash(w: torch.Tensor) -> str:
    return hashlib.sha256(w.to(torch.float32).numpy().tobytes()).hexdigest()[:16]


def fit_cell_grid(train_by: dict, val_by: dict, layer: int, loop: int, log_prefix: str = ""):
    """Full v2 grid; select by val macro top-1. Never sees heldout groups (asserted by caller)."""
    best = None
    grid_rows = []
    for tset in X.TRAINING_SETS:
        for lr in X.GRID_LRS:
            for seed in X.GRID_SEEDS:
                w = train_pairwise(train_by, layer, loop, tset, lr, seed)
                if w is None:
                    continue
                val_macro, _, _ = eval_cell(val_by, w, layer, loop)
                grid_rows.append({"tset": tset, "lr": lr, "seed": seed, "val_macro": round(val_macro, 4)})
                if best is None or val_macro > best["val_macro"]:
                    best = {"weight": w, "tset": tset, "lr": lr, "seed": seed, "val_macro": val_macro}
    assert best is not None, f"no trainable grid point at {X.cell_name(layer, loop)}"
    print(f"  {log_prefix}{X.cell_name(layer, loop)}: best val={best['val_macro']:.4f} "
          f"({best['tset']}, lr={best['lr']}, seed={best['seed']})", flush=True)
    return best, grid_rows


def main() -> int:
    t0 = time.time()
    X.ensure_roots()
    print("loading fresh feature groups ...", flush=True)
    train_by = {d: [] for d in X.CORE_DOMAINS}
    val_by = {d: [] for d in X.CORE_DOMAINS}
    held_by = {d: [] for d in X.CORE_DOMAINS}
    for g in X.load_fresh_groups():
        {"train": train_by, "val": val_by, "heldout": held_by}[g["split"]][g["domain"]].append(g)
    counts = {s: {d: len(by[d]) for d in X.CORE_DOMAINS} for s, by in
              (("train", train_by), ("val", val_by), ("heldout", held_by))}
    print(f"group counts: {counts}", flush=True)

    # alignment control: heldout identity fixed across all cells (structural: same tensors)
    held_uids = sorted(g["group_uid"] for d in X.CORE_DOMAINS for g in held_by[d])
    held_tasks = {g["task_uid"] for d in X.CORE_DOMAINS for g in held_by[d]}
    train_tasks = {g["task_uid"] for d in X.CORE_DOMAINS for g in train_by[d]}
    val_tasks = {g["task_uid"] for d in X.CORE_DOMAINS for g in val_by[d]}
    assert not (held_tasks & train_tasks) and not (held_tasks & val_tasks) and not (train_tasks & val_tasks), \
        "task crossing between splits"

    # matched-random chance
    chance_by_dom = {d: X.finite_mean(X.chance_top1(g) for g in held_by[d]) for d in X.CORE_DOMAINS}
    chance_macro = X.finite_mean(chance_by_dom.values())

    results = {"counts": counts, "heldout_group_uids_sha": hashlib_sha(held_uids),
               "chance": {"macro": chance_macro, "by_domain": chance_by_dom},
               "cells": {}, "transfers": {}, "shuffle_controls": {}}

    # ---------------- local-refit matrix
    taps = {}
    for (layer, loop) in X.REFIT_CELLS:
        best, grid_rows = fit_cell_grid(train_by, val_by, layer, loop, "refit ")
        macro, per_dom, rows = eval_cell(held_by, best["weight"], layer, loop)
        cn = X.cell_name(layer, loop)
        taps[cn] = best
        results["cells"][cn] = {
            "heldout_macro_top1": round(macro, 4),
            "heldout_by_domain": {d: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in per_dom[d].items()}
                                  for d in per_dom},
            "heldout_macro_pairwise": round(X.finite_mean(per_dom[d]["pairwise"] for d in per_dom), 4),
            "val_macro": round(best["val_macro"], 4),
            "selected": {k: best[k] for k in ("tset", "lr", "seed")},
            "weight_sha": weight_hash(best["weight"]),
            "grid": grid_rows,
        }
        X.write_json(X.PRED_ROOT / f"refit_{cn}.json", rows)
        print(f"    heldout macro top1 = {macro:.4f}", flush=True)

    torch.save({cn: {"weight": taps[cn]["weight"], "arch": "AntisymLinearNoNorm",
                     "cell": cn, "selected": {k: taps[cn][k] for k in ("tset", "lr", "seed")}}
                for cn in taps}, X.RUN_ROOT / "xloop_taps.pt")

    # ---------------- frozen-transfer matrix
    for (layer, u, v) in X.TRANSFERS:
        src, tgt = X.cell_name(layer, u), X.cell_name(layer, v)
        w = taps[src]["weight"]
        h_before = weight_hash(w)
        # source self-reproduction (hard fail)
        macro_src, _, _ = eval_cell(held_by, w, layer, u)
        stored = results["cells"][src]["heldout_macro_top1"]
        assert abs(macro_src - stored) < 5e-5, f"source self-repro failed {src}: {macro_src} vs {stored}"
        macro_t, per_dom_t, rows_t = eval_cell(held_by, w, layer, v)
        assert weight_hash(w) == h_before, "frozen head mutated during transplant"
        # agreement vs target-local refit on identical candidates
        refit_rows = {r["group_uid"]: r for r in X.read_json(X.PRED_ROOT / f"refit_{tgt}.json")}
        pear, spear, sign_agree, group_rank = score_agreement(rows_t, refit_rows)
        key = f"{layer}:L{u}->L{v}"
        results["transfers"][key] = {
            "target_macro_top1": round(macro_t, 4),
            "target_by_domain": {d: round(per_dom_t[d]["top1"], 4) for d in per_dom_t},
            "target_macro_pairwise": round(X.finite_mean(per_dom_t[d]["pairwise"] for d in per_dom_t), 4),
            "delta_vs_source_self": round(macro_t - stored, 4),
            "delta_vs_target_refit": round(macro_t - results["cells"][tgt]["heldout_macro_top1"], 4),
            "score_pearson_vs_target_refit": pear,
            "score_spearman_vs_target_refit": spear,
            "pairwise_sign_agreement": sign_agree,
            "mean_group_rank_corr": group_rank,
            "weight_sha": h_before,
        }
        X.write_json(X.PRED_ROOT / f"transfer_{layer}_L{u}_to_L{v}.json", rows_t)
        print(f"  transfer {key}: tgt={macro_t:.4f} (refit {results['cells'][tgt]['heldout_macro_top1']:.4f})", flush=True)

    # ---------------- label-shuffle control
    rng = torch.Generator().manual_seed(X.BOOT_SEED)
    shuf_train, shuf_val = shuffle_labels(train_by, rng), shuffle_labels(val_by, rng)
    for (layer, loop) in X.SHUFFLE_CELLS:
        best, _ = fit_cell_grid(shuf_train, shuf_val, layer, loop, "shuffle ")
        macro, per_dom, _ = eval_cell(held_by, best["weight"], layer, loop)
        results["shuffle_controls"][X.cell_name(layer, loop)] = {
            "heldout_macro_top1": round(macro, 4),
            "by_domain": {d: round(per_dom[d]["top1"], 4) for d in per_dom},
            "chance_macro": round(chance_macro, 4),
        }
        print(f"  shuffle {X.cell_name(layer, loop)}: heldout={macro:.4f} (chance {chance_macro:.4f})", flush=True)

    results["elapsed_seconds"] = round(time.time() - t0, 1)
    X.write_json(X.RUN_ROOT / "train_eval_results.json", results)
    print(f"train/eval done in {results['elapsed_seconds']}s", flush=True)
    return 0


def hashlib_sha(items) -> str:
    h = hashlib.sha256()
    for it in items:
        h.update(str(it).encode())
    return h.hexdigest()[:16]


def shuffle_labels(by_dom: dict, rng: torch.Generator) -> dict:
    out = {}
    for d, gs in by_dom.items():
        ngs = []
        for g in gs:
            perm = torch.randperm(len(g["candidates"]), generator=rng).tolist()
            rewards = [float(g["candidates"][i]["reward"]) for i in perm]
            ncands = [dict(c, reward=r) for c, r in zip(g["candidates"], rewards)]
            ngs.append(dict(g, candidates=ncands))
        out[d] = ngs
    return out


def score_agreement(t_rows: list, refit_rows: dict):
    """Correlations between transplant and target-refit scores on identical heldout candidates."""
    all_t, all_r = [], []
    signs = []
    group_corrs = []
    for r in t_rows:
        ref = refit_rows.get(r["group_uid"])
        if ref is None or ref["candidate_uids"] != r["candidate_uids"]:
            continue
        ts, rs = r["scores"], ref["scores"]
        all_t.extend(ts)
        all_r.extend(rs)
        n = len(ts)
        for i in range(n):
            for j in range(i + 1, n):
                dt, dr = ts[i] - ts[j], rs[i] - rs[j]
                if dt != 0 and dr != 0:
                    signs.append(1.0 if (dt > 0) == (dr > 0) else 0.0)
        if n >= 3:
            group_corrs.append(spearman(ts, rs))
    return (round(pearson(all_t, all_r), 4), round(spearman(all_t, all_r), 4),
            round(X.finite_mean(signs), 4), round(X.finite_mean(group_corrs), 4))


def pearson(a, b) -> float:
    if len(a) < 3:
        return float("nan")
    ta, tb = torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
    ta, tb = ta - ta.mean(), tb - tb.mean()
    den = ta.norm() * tb.norm()
    return float((ta @ tb / den).item()) if den > 0 else float("nan")


def rankdata(v) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    ranks = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a, b) -> float:
    if len(a) < 3:
        return float("nan")
    return pearson(rankdata(a), rankdata(b))


if __name__ == "__main__":
    raise SystemExit(main())
