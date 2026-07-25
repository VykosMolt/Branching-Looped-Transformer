#!/usr/bin/env python3
"""Refit/evaluate Ouro CoreContent-v2 with whole task IDs assigned to one split."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "utilities/tests/manual"))
import bg_corecontent_v2_features as feat  # noqa: E402
import bg_corecontent_v2_models as models  # noqa: E402

SEED = 20260711


def assigned_split(task_id: str) -> str:
    u = int(hashlib.sha256(f"{SEED}:{task_id}".encode()).hexdigest()[:16], 16) / 2**64
    return "train" if u < .8 else ("val" if u < .9 else "heldout")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    groups = feat.load_feature_groups()
    stored = defaultdict(set)
    for g in groups: stored[g["task_id"]].add(g["split"])
    for g in groups: g["split"] = assigned_split(g["task_id"])
    effective = defaultdict(set)
    for g in groups: effective[g["task_id"]].add(g["split"])
    by = defaultdict(list)
    for g in groups: by[(g["domain"], g["split"])].append(g)
    train_by = {d: by[(d, "train")] for d in models.CORE_DOMAINS}
    val_by = {d: by[(d, "val")] for d in models.CORE_DOMAINS}
    held_by = {d: by[(d, "heldout")] for d in models.CORE_DOMAINS}

    results=[]; best={}
    for config in models.TRAIN_CONFIGS:
        for tset in models.TRAINING_SETS:
            for lr in (3e-4, 1e-3):
                for seed in (0, 1, 2):
                    rec=models.train_pairwise(train_by,val_by,config,tset,lr,seed)
                    if rec is None: continue
                    results.append({k:rec[k] for k in ("config","training_set","lr","seed","pairs","val_core_macro")})
                    if config not in best or rec["val_core_macro"] > best[config]["val_core_macro"]: best[config]=rec
    channels=[(best[c]["weight"],"AntisymLinearNoNorm",c) for c in models.TRAIN_CONFIGS]
    val_dom=models.eval_channels_macro(channels,val_by)
    held_dom=models.eval_channels_macro(channels,held_by)
    def counts(split): return {d:len(by[(d,split)]) for d in models.CORE_DOMAINS}
    audit={
        "seed":SEED,"rule":"sha256(seed:task_id), 80/10/10 thresholds",
        "records":len(groups),"unique_tasks":len(effective),
        "stored_task_ids_crossing_splits":sum(len(v)>1 for v in stored.values()),
        "effective_task_ids_crossing_splits":sum(len(v)>1 for v in effective.values()),
        "record_counts":dict(Counter(g["split"] for g in groups)),
        "task_counts":dict(Counter(next(iter(v)) for v in effective.values())),
    }
    payload={
        "status":"COMPLETE","split_audit":audit,"configs":list(models.TRAIN_CONFIGS),
        "architecture":"AntisymLinearNoNorm","scoring":"three-channel blockwise z-score average",
        "best_per_config":{c:{k:v for k,v in best[c].items() if k!="weight"} for c in best},
        "validation":{"groups":counts("val"),"top1_by_domain":val_dom,"macro_top1":models.macro_core(val_dom)},
        "heldout":{"groups":counts("heldout"),"top1_by_domain":held_dom,"macro_top1":models.macro_core(held_dom)},
        "historical_stored_split_macro_top1":0.6691,
        "elapsed_seconds":time.time()-started,
    }
    torch.save({"channels":channels,"best_per_config":best,"split_audit":audit},out/"ouro_corecontent_task_disjoint_policy.pt")
    (out/"ouro_corecontent_task_disjoint_results.json").write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload,indent=2),flush=True)


if __name__ == "__main__": main()
