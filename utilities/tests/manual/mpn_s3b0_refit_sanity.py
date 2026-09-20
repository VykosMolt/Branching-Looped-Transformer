"""M+N S3B-0 — zero-GPU refit sanity for the branch-correctness selector.

Purpose (sanity ONLY, NOT a Wall-B result): confirm the pairwise antisymmetric relational-preference
training path runs and does not underperform the frozen CoreContent selector IN-DISTRIBUTION (on the
existing corecontent_v2 pre-extracted features). The real Wall-B test is S3B-1 (transfer to generated
loop branch pools). DO NOT claim Wall B solved here.

What it does (no GPU; features already extracted under shared/data/corecontent_v2/features):
  - train a pairwise antisymmetric channel per config (24_L4/36_L4/47_L4) via M.train_pairwise (small lr
    sweep, pick best by val) -> S3B0_pairwise_blockwise; listwise variant -> S3B0_listwise_blockwise (ablation).
  - eval on the HELDOUT split vs frozen baselines: CoreContent_v2_blockwise, DualAnchor,
    mixedhead_MIX_HH_OBJECTIVE, MIX_OBJECTIVE_ALL_only, RANDOM, ORACLE.
  - report the full selection panel (core metric = selected_correct_when_oracle_present), macro over core
    domains + per-domain breakdown.

Gate: trained pairwise selector macro and selected_correct_when_oracle_present should be in the ballpark of
(not materially below) frozen CoreContent in-distribution -> training path is sane -> proceed to S3B-1.

Run (no GPU needed): venv/bin/python utilities/tests/manual/mpn_s3b0_refit_sanity.py
"""
from __future__ import annotations
import json
import math
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import bg_corecontent_v2_models as M  # noqa: E402

OUT = M.v2.PROJECT_ROOT / "opi/taps/probes/mpn_s3b_2026-06-17"
CORE = list(M.CORE_DOMAINS)
LR_GRID = [0.01, 0.03, 0.1, 0.3, 1.0]
SEEDS = [0, 1, 2]
TSET = "all_core_balanced"


def pool_metrics(group, scores):
    cands = group["candidates"]
    rewards = [float(c["reward"]) for c in cands]
    correct = [r > 0 for r in rewards]
    finite = [(i, s) for i, s in enumerate(scores) if isinstance(s, float) and math.isfinite(s)]
    if not finite:
        return None
    order = [i for i, _ in sorted(finite, key=lambda x: x[1], reverse=True)]
    top = order[0]
    oracle_present = any(correct)
    return {
        "oracle_present": oracle_present,
        "selected_correct": bool(correct[top]),
        "regret": max(rewards) - rewards[top],
        "top2_ret": any(correct[i] for i in order[:2]),
        "top4_ret": any(correct[i] for i in order[:4]),
        "n": len(cands),
    }


def eval_policy(policy, heldout_by):
    """Return macro (over core domains) panel + per-domain selected_correct_when_oracle_present."""
    per_dom = {}
    for d in CORE:
        rows = []
        for g in heldout_by.get(d, []):
            scores = M.score_group(g, policy)
            m = pool_metrics(g, scores)
            if m:
                rows.append(m)
        if not rows:
            continue
        op = [r for r in rows if r["oracle_present"]]
        nop = max(1, len(op))
        per_dom[d] = {
            "n_pools": len(rows),
            "oracle_over_pool": round(sum(r["oracle_present"] for r in rows) / len(rows), 4),
            "selected_acc": round(sum(r["selected_correct"] for r in rows) / len(rows), 4),
            "selected_correct_when_oracle_present": round(sum(r["selected_correct"] for r in op) / nop, 4),
            "oracle_conversion_rate": round(sum(r["selected_correct"] for r in op) / nop, 4),
            "top2_oracle_retention": round(sum(r["top2_ret"] for r in op) / nop, 4),
            "top4_oracle_retention": round(sum(r["top4_ret"] for r in op) / nop, 4),
            "selector_regret": round(sum(r["regret"] for r in rows) / len(rows), 4),
        }
    keys = ["oracle_over_pool", "selected_acc", "selected_correct_when_oracle_present",
            "oracle_conversion_rate", "top2_oracle_retention", "top4_oracle_retention", "selector_regret"]
    macro = {k: round(sum(per_dom[d][k] for d in per_dom) / max(1, len(per_dom)), 4) for k in keys}
    return {"macro": macro,
            "by_domain_core_metric": {d: per_dom[d]["selected_correct_when_oracle_present"] for d in per_dom},
            "per_domain": per_dom}


def train_blockwise(trainer, train_by, val_by):
    """Train one antisymmetric channel per config; lr-sweep, pick best by val_core_macro -> channel list."""
    channels = []
    detail = {}
    for cfg in M.TRAIN_CONFIGS:
        best = None
        for lr in LR_GRID:
            for seed in SEEDS:
                r = trainer(train_by, val_by, cfg, TSET, lr, seed)
                if r and (best is None or r["val_core_macro"] > best["val_core_macro"]):
                    best = r
        if best:
            channels.append((best["weight"], best["arch"], cfg))
            detail[cfg] = {"lr": best["lr"], "seed": best["seed"],
                           "val_core_macro": best["val_core_macro"], "pairs": best.get("pairs")}
    return channels, detail


def main() -> int:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    train_by = M.groups_by_domain("train", CORE)
    val_by = M.groups_by_domain("val", CORE)
    heldout_by = M.groups_by_domain("heldout", CORE)
    n_train = {d: len(train_by.get(d, [])) for d in CORE}
    n_held = {d: len(heldout_by.get(d, [])) for d in CORE}

    pair_ch, pair_detail = train_blockwise(M.train_pairwise, train_by, val_by)
    list_ch, list_detail = train_blockwise(M.train_listwise, train_by, val_by)
    # persist trained selector for S3B-1 reuse
    torch.save({"pairwise_blockwise_channels": pair_ch, "listwise_blockwise_channels": list_ch,
                "pairwise_detail": pair_detail, "listwise_detail": list_detail},
               OUT / "s3b0_trained_selector.pt")

    base = M.baseline_policies(include_diag=True)
    crafted = M.crafted_v2_policies()
    policies = {
        "S3B0_pairwise_blockwise": pair_ch,
        "S3B0_listwise_blockwise": list_ch,
        "CoreContent_v2_blockwise": crafted.get("CoreContent_v2_blockwise"),
        "DualAnchor": base.get("DualAnchor"),
        "mixedhead_MIX_HH_OBJECTIVE": base.get("mixedhead_MIX_HH_OBJECTIVE"),
        "MIX_OBJECTIVE_ALL_only": base.get("MIX_OBJECTIVE_ALL_only"),
        "RANDOM": "RANDOM",
        "ORACLE": "ORACLE",
    }
    results = {}
    for name, pol in policies.items():
        if pol is None:
            continue
        results[name] = eval_policy(pol, heldout_by)
        m = results[name]["macro"]
        print(f"  {name:28} sel@oracle={m['selected_correct_when_oracle_present']:.4f} "
              f"sel_acc={m['selected_acc']:.4f} top4_ret={m['top4_oracle_retention']:.4f} "
              f"regret={m['selector_regret']:.4f}", flush=True)

    core = "selected_correct_when_oracle_present"
    trained = results["S3B0_pairwise_blockwise"]["macro"][core]
    frozen = results.get("CoreContent_v2_blockwise", {}).get("macro", {}).get(core)
    delta = (round(trained - frozen, 4) if frozen is not None else None)
    sane = (frozen is None) or (delta >= -0.02)   # not materially below frozen in-distribution
    payload = {
        "verdict": "S3B0_REFIT_SANE" if sane else "S3B0_REFIT_UNDERPERFORMS_FROZEN",
        "scope": "ZERO-GPU in-distribution sanity ONLY; NOT a Wall-B result; does not claim Wall B solved",
        "core_metric": core,
        "trained_pairwise_minus_frozen_corecontent": delta,
        "config": {"tset": TSET, "lr_grid": LR_GRID, "configs": list(M.TRAIN_CONFIGS),
                   "n_train_pools": n_train, "n_heldout_pools": n_held},
        "pairwise_detail": pair_detail, "listwise_detail": list_detail,
        "results": results, "elapsed_seconds": round(time.time() - started, 1),
    }
    (OUT / "s3b0_refit_sanity.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"verdict": payload["verdict"], "core_metric": core,
                      "trained_pairwise": trained, "frozen_corecontent": frozen, "delta": delta,
                      "random": results.get("RANDOM", {}).get("macro", {}).get(core),
                      "oracle": results.get("ORACLE", {}).get("macro", {}).get(core)}, indent=2))
    print(f"\n{payload['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
