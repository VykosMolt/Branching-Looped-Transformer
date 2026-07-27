"""V3 within-family cross-loop replication driver.

Re-runs the sealed cross_loop_early_layer_taps_20260720 protocol on other Ouro
checkpoints, over the IDENTICAL 2,150-group subset (copied byte-for-byte from
the sealed run), with the identical tap class, 36-point grid, val-only
selection, seed and bootstrap. Adds one new stage: frozen transfer of the
sealed RLTT taps onto each sibling checkpoint's features (2.6B family only —
dimensions match; the 1.4B gets fresh taps at proportionally mapped layers).

Stages (run per model, in order):
    prep        copy subset + integrity docs into the model's run root
    extract     frozen-forward feature capture [L,4,H] fp16 (resumable)
    traineval   local-refit grid + within-model transfers (sealed procedure)
    crossmodel  score sealed RLTT taps on this model's heldout features
    stats       task-clustered bootstrap (sealed seed/draws)

Usage:
    venv/bin/python -u bg_v3_family_xloop.py --model base26 --stage extract
Models: base26 | thinking26 | ouro14b
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANUAL = PROJECT_ROOT / "utilities/tests/manual"
for p in (str(PROJECT_ROOT), str(MANUAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

import bg_xloop_early_v1_common as X  # noqa: E402

V3_ROOT = PROJECT_ROOT / "artifacts/reports/family_xloop_v3_20260726"
SEALED_ROOT = PROJECT_ROOT / "artifacts/reports/cross_loop_early_layer_taps_20260720"
HF_HUB = PROJECT_ROOT / "artifacts/hf_cache/hub"

MODEL_SPECS = {
    "base26": ("models--ByteDance--Ouro-2.6B", "1ed04250da1a9936042725d302e81c8fa2ab5abd"),
    "thinking26": ("models--ByteDance--Ouro-2.6B-Thinking", "f1edd81e7ac41355db670500ceaf204e0f73af68"),
    "ouro14b": ("models--ByteDance--Ouro-1.4B", None),
}


def snapshot_dir(key: str) -> Path:
    cache_name, rev = MODEL_SPECS[key]
    snaps = HF_HUB / cache_name / "snapshots"
    if rev and (snaps / rev).exists():
        return snaps / rev
    subdirs = sorted(d for d in snaps.iterdir() if d.is_dir())
    assert subdirs, f"no snapshot downloaded for {key}"
    return subdirs[-1]


def layer_plan(key: str) -> dict:
    """Layer capture plan. 2.6B family shares RLTT geometry; 1.4B is mapped
    proportionally from the sealed {8,16,24,36,47}/48 scheme."""
    cfg = json.loads((snapshot_dir(key) / "config.json").read_text())
    n_layers = cfg["num_hidden_layers"]
    hidden = cfg["hidden_size"]
    if n_layers == 48:
        layers = (8, 16, 24, 36, 47)
    else:
        frac = [8 / 48, 16 / 48, 24 / 48, 36 / 48]
        mapped = sorted({max(1, round(f * n_layers)) for f in frac})
        layers = tuple(mapped) + (n_layers - 1,)
    capture_idx = {l: l - 1 for l in layers[:-1]}
    return {"n_layers": n_layers, "hidden": hidden, "layers": layers,
            "capture_idx": capture_idx, "boundary": layers[-1],
            "total_ut_steps": cfg.get("total_ut_steps", 4)}


def patch_common(key: str, plan: dict) -> None:
    root = V3_ROOT / key
    X.RUN_ROOT = root
    X.FEATURE_ROOT = root / "features"
    X.PRED_ROOT = root / "predictions"
    X.LAYERS = plan["layers"]
    X.LAYER_POS = {l: i for i, l in enumerate(plan["layers"])}
    X.CAPTURE_IDX = plan["capture_idx"]
    X.HIDDEN = plan["hidden"]
    early = [l for l in plan["layers"] if l / plan["n_layers"] <= 0.55]
    X.REFIT_CELLS = [(l, u) for l in early for u in X.LOOPS] + \
                    [(l, 4) for l in plan["layers"] if l not in early]
    X.TRANSFERS = [(l, u, v) for l in early for u in X.LOOPS for v in X.LOOPS if v > u]
    X.SHUFFLE_CELLS = [(early[0], 1), (early[0], 4)]


def stage_prep(key: str, plan: dict) -> None:
    root = V3_ROOT / key
    (root / "features").mkdir(parents=True, exist_ok=True)
    (root / "predictions").mkdir(parents=True, exist_ok=True)
    for fname in ("subset_groups.jsonl", "excluded_crossing_tasks.json"):
        src, dst = SEALED_ROOT / fname, root / fname
        if not dst.exists():
            shutil.copy2(src, dst)
        assert X.sha256_file(src) == X.sha256_file(dst), f"{fname} diverged from sealed subset"
    (root / "model_plan.json").write_text(json.dumps(
        {"model_key": key, "snapshot": str(snapshot_dir(key)), **{
            k: (list(v) if isinstance(v, tuple) else v) for k, v in plan.items()}},
        indent=2, default=str) + "\n")
    print(f"[prep:{key}] subset copied+verified; plan: layers={plan['layers']} "
          f"hidden={plan['hidden']} n_layers={plan['n_layers']}")


def load_subset(key: str) -> list[dict]:
    rows = []
    with open(V3_ROOT / key / "subset_groups.jsonl") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def stage_extract(key: str, plan: dict) -> None:
    import bg_xloop_early_v1_extract as EX

    model_path = snapshot_dir(key)
    boundary = plan["boundary"]

    class ExtractorFam(EX.ExtractorX):
        def __init__(self, **kw):
            kw.setdefault("model_path", model_path)
            super().__init__(**kw)

        def encode_multi_layer(self, text: str, max_length: int) -> torch.Tensor:
            # same as sealed ExtractorX.encode_multi_layer, with the boundary
            # layer parameterized instead of the literal 47
            if not str(text).strip():
                raise ValueError("text must be non-empty")
            enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                                 max_length=max_length, padding=False)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            if "attention_mask" not in enc:
                enc["attention_mask"] = torch.ones_like(enc["input_ids"], device=self.device)
            cap = EX._CaptureX()
            cap.inter = {idx: [] for idx in X.CAPTURE_IDX.values()}
            inner = getattr(self.model, "model", None)
            layers = getattr(inner, "layers", None)
            if inner is None or layers is None:
                raise RuntimeError("model does not expose model.model.layers")
            handles = [inner.register_forward_hook(cap.boundary_hook)]
            for idx in X.CAPTURE_IDX.values():
                handles.append(layers[idx].register_forward_hook(cap.make_inter_hook(idx)))
            try:
                with torch.inference_mode():
                    self.model(**enc, use_cache=False)
            finally:
                for h in handles:
                    h.remove()
            if len(cap.boundary) != len(X.LOOPS):
                raise RuntimeError(f"expected {len(X.LOOPS)} boundary states, got {len(cap.boundary)}")
            for layer, idx in X.CAPTURE_IDX.items():
                if len(cap.inter[idx]) != len(X.LOOPS):
                    raise RuntimeError(f"layer {layer}: got {len(cap.inter[idx])} loop states")
            per_layer = []
            for layer in X.LAYERS:
                states = cap.boundary if layer == boundary else cap.inter[X.CAPTURE_IDX[layer]]
                pooled = [EX._masked_mean_pool(states[i], enc["attention_mask"])
                          for i in range(len(X.LOOPS))]
                per_layer.append(torch.stack(pooled, 0))
            feats = torch.stack(per_layer, 0).to(device="cpu", dtype=torch.float32)
            expected = (len(X.LAYERS), len(X.LOOPS), X.HIDDEN)
            if tuple(feats.shape) != expected:
                raise ValueError(f"feature shape {tuple(feats.shape)} != {expected}")
            return feats

    EX.ExtractorX = ExtractorFam
    EX.extract(load_subset(key))


def stage_traineval(key: str) -> None:
    import bg_xloop_early_v1_train_eval as TE
    TE.main()


def stage_crossmodel(key: str, plan: dict) -> None:
    """Frozen transfer of the sealed RLTT taps onto this checkpoint."""
    if plan["hidden"] != 2048 or plan["n_layers"] != 48:
        print(f"[crossmodel:{key}] skipped: geometry differs from RLTT (fresh taps only)")
        return
    sealed = torch.load(SEALED_ROOT / "xloop_taps.pt", map_location="cpu", weights_only=False)
    held_by = {d: [] for d in X.CORE_DOMAINS}
    for g in X.load_fresh_groups():
        if g["split"] == "heldout":
            held_by[g["domain"]].append(g)
    import bg_xloop_early_v1_train_eval as TE
    refits = json.loads((X.RUN_ROOT / "train_eval_results.json").read_text())["cells"]
    out = {}
    for cn, tap in sealed.items():
        layer, loop = cn.split("_L")
        layer, loop = int(layer), int(loop)
        macro, per_dom, rows = TE.eval_cell(held_by, tap["weight"], layer, loop)
        local = refits.get(cn, {}).get("heldout_macro_top1")
        refit_rows = {r["group_uid"]: r for r in
                      X.read_json(X.PRED_ROOT / f"refit_{cn}.json")}
        agree = TE.score_agreement(rows, refit_rows)
        out[cn] = {"rltt_tap_macro_top1": round(macro, 4),
                   "local_refit_macro_top1": local,
                   "delta_vs_local_refit": (round(macro - local, 4)
                                            if local is not None else None),
                   "agreement_vs_local_refit": agree,
                   "weight_sha": TE.weight_hash(tap["weight"])}
        print(f"[crossmodel:{key}] {cn}: rltt_tap={macro:.4f} local={local} ")
    X.write_json(X.RUN_ROOT / "crossmodel_rltt_transfer.json", out)


def generic_stats(ref_cells: list[str], out_name: str = "bootstrap_stats_v3.json") -> None:
    """Sealed clustered-bootstrap core with configurable reference cells, for
    checkpoints whose layer numbering differs from the sealed 48-layer scheme."""
    import numpy as np
    import bg_xloop_early_v1_stats as ST

    series = ST.load_pred_vectors()
    points, samples = ST.bootstrap(series, X.BOOT_DRAWS, X.BOOT_SEED)

    def paired(a: str, b: str) -> dict:
        d = samples[a] - samples[b]
        lo, hi = np.percentile(d, [2.5, 97.5])
        return {"point": round(points[a] - points[b], 4),
                "ci95": [round(float(lo), 4), round(float(hi), 4)],
                "excludes_zero": bool(lo > 0 or hi < 0)}

    out = {"points": {k: round(float(v), 4) for k, v in points.items()},
           "paired_deltas": {}, "reference_cells": ref_cells}
    refs = [f"refit:{c}" for c in ref_cells]
    for name in series:
        if not name.startswith("refit:"):
            continue
        out["paired_deltas"][f"{name} - chance"] = paired(name, "chance")
        for r in refs:
            if name != r:
                out["paired_deltas"][f"{name} - {r}"] = paired(name, r)
    for (layer, u, v) in X.TRANSFERS:
        tn = f"transfer:{layer}:L{u}->L{v}"
        tgt = f"refit:{X.cell_name(layer, v)}"
        if tn in series and tgt in series:
            out["paired_deltas"][f"{tn} - {tgt}"] = paired(tn, tgt)
    X.write_json(X.RUN_ROOT / out_name, out)
    print(f"[stats] wrote {out_name}: {len(out['paired_deltas'])} paired deltas")


def stage_stats(key: str, plan: dict) -> None:
    if plan["layers"] == (8, 16, 24, 36, 47):
        import bg_xloop_early_v1_stats as ST
        ST.main()
    else:
        mid = plan["layers"][-3]   # mapped 24-equivalent
        generic_stats([X.cell_name(mid, 4), X.cell_name(plan["boundary"], 4)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    ap.add_argument("--stage", required=True,
                    choices=["prep", "extract", "traineval", "crossmodel", "stats", "all"])
    args = ap.parse_args()
    plan = layer_plan(args.model)
    patch_common(args.model, plan)
    stages = ["prep", "extract", "traineval", "crossmodel", "stats"] \
        if args.stage == "all" else [args.stage]
    for s in stages:
        t0 = time.time()
        {"prep": lambda: stage_prep(args.model, plan),
         "extract": lambda: stage_extract(args.model, plan),
         "traineval": lambda: stage_traineval(args.model),
         "crossmodel": lambda: stage_crossmodel(args.model, plan),
         "stats": lambda: stage_stats(args.model, plan)}[s]()
        print(f"[{s}:{args.model}] done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
