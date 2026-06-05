"""PROBE 2 — Per-layer Thinking-vs-RLTT comparison at layers 24/36/47.

v10 showed boundary (L47) states are ~identical between Thinking and
RLTT (the final norm equalises them). Intermediate layers (24, 36) might
still differ — the v9 layer recommendations were measured under Thinking
and are about to be baked into the BG architecture on RLTT. This probe
measures cross-backbone agreement per (layer, loop).

Owns the shared capture pass (spec execution order):
  A: load RLTT, capture L24/36/47 × 4 loops for the 200 pairs, save, free.
  B: same for Thinking, save, free.
  C: Probe 1 analysis is a separate script reading the RLTT .pt.
  D: Probe 2 analysis here, from both saved .pt files.

Outputs (artifacts/reports/evaluator/ — repo convention; the spec's
runs/ wording is honoured by the path note in layer_state_capture):
  hh_layer_states_200_rltt.pt, hh_layer_states_200_thinking.pt,
  probe_cross_backbone_layers.json

STOP semantics: if the final verdict is not "v9 layer recommendations
transfer to RLTT", that contradicts the locked §4 spec and must be
reported, not worked around.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from layer_state_capture import (DEFAULT_HEAD_PATH, DEFAULT_OUTPUT_DIR,
                                  DEFAULT_RLTT_PATH, DEFAULT_THINKING_PATH,
                                  DEFAULT_TOKENIZER_PATH, NUM_LOOPS, TAP_LAYERS,
                                  PROJECT_ROOT, ExamplePack, capture_backbone,
                                  load_packs, save_packs, select_examples)

from evaluator_core.pairwise_evaluator import PairwiseEvaluator


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(torch.float32).flatten()
    b = b.to(torch.float32).flatten()
    na, nb = a.norm(), b.norm()
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float((a @ b) / (na * nb))


def _stats(vals: List[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in vals if math.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std()),
            "n": int(arr.size)}


def _cell_verdict(mean_cos: float) -> str:
    if not math.isfinite(mean_cos):
        return "nan"
    if mean_cos > 0.99:
        return "identical"
    if mean_cos >= 0.95:
        return "near-identical"
    return "diverged"


def cross_backbone(packs_t: List[ExamplePack],
                   packs_r: List[ExamplePack]) -> Dict:
    assert len(packs_t) == len(packs_r), "pack count mismatch"
    n = len(packs_t)
    table: Dict[str, Dict[str, Dict]] = {}
    for ly in TAP_LAYERS:
        table[f"L{ly}"] = {}
        for li in range(NUM_LOOPS):
            vals = []
            for k in range(n):
                ct = packs_t[k].chosen.pooled[ly][li]
                cr = packs_r[k].chosen.pooled[ly][li]
                rt = packs_t[k].rejected.pooled[ly][li]
                rr = packs_r[k].rejected.pooled[ly][li]
                vals.append(_cos(ct, cr))
                vals.append(_cos(rt, rr))
            st = _stats(vals)
            st["verdict"] = _cell_verdict(st["mean"])
            table[f"L{ly}"][f"loop{li+1}"] = st
    return table


def head_scores(head: PairwiseEvaluator, packs: List[ExamplePack],
                device: torch.device) -> List[float]:
    head.eval()
    out: List[float] = []
    with torch.no_grad():
        for p in packs:
            cs = [t.to(device, torch.float32).unsqueeze(0)
                  for t in p.chosen.l47_tokens]
            rs = [t.to(device, torch.float32).unsqueeze(0)
                  for t in p.rejected.l47_tokens]
            cm = p.chosen.mask.to(device, torch.float32).unsqueeze(0)
            rm = p.rejected.mask.to(device, torch.float32).unsqueeze(0)
            v = float(head(cs, cm, rs, rm).view(-1).item())
            out.append(v if math.isfinite(v) else float("nan"))
    return out


def head_agreement(head_path: str, packs_t: List[ExamplePack],
                   packs_r: List[ExamplePack], device: torch.device) -> Dict:
    head = PairwiseEvaluator().to(device)
    ck = torch.load(head_path, map_location=device, weights_only=False)
    sd = ck["model_state_dict"] if isinstance(ck, dict) and \
        "model_state_dict" in ck else ck
    head.load_state_dict(sd)
    head.eval()

    s_t = np.asarray(head_scores(head, packs_t, device), dtype=np.float64)
    s_r = np.asarray(head_scores(head, packs_r, device), dtype=np.float64)
    valid = np.isfinite(s_t) & np.isfinite(s_r)
    s_t, s_r = s_t[valid], s_r[valid]
    if s_t.size >= 2 and np.std(s_t) > 1e-12 and np.std(s_r) > 1e-12:
        pearson = float(np.corrcoef(s_t, s_r)[0, 1])
    else:
        pearson = float("nan")
    return {
        "n_valid": int(s_t.size),
        "pearson_score_T_vs_R": pearson,
        "mean_abs_delta_score": float(np.abs(s_t - s_r).mean()),
        "decision_agreement_rate": float(
            (np.sign(s_t) == np.sign(s_r)).mean()),
        "canonical_acc_thinking": float((s_t > 0).mean()),
        "canonical_acc_rltt": float((s_r > 0).mean()),
        "score_T_mean": float(s_t.mean()), "score_T_std": float(s_t.std()),
        "score_R_mean": float(s_r.mean()), "score_R_std": float(s_r.std()),
    }


def v9_verdict(table: Dict) -> Dict:
    bad = []
    for ly in (24, 36):
        for li in range(NUM_LOOPS):
            cell = table[f"L{ly}"][f"loop{li+1}"]
            if cell["verdict"] == "diverged":
                bad.append(f"L{ly}/loop{li+1} (cos={cell['mean']:.4f})")
    if not bad:
        text = ("v9 layer recommendations transfer to RLTT. BG Phase 1 can "
                "use layers 24/36/47 as planned.")
        contradicts = False
    else:
        text = ("v9 recommendations do NOT cleanly transfer. RLTT "
                "intermediate layers differ from Thinking at: "
                + ", ".join(bad)
                + ". Phase 1 must re-do the layer sweep on RLTT "
                "activations or re-justify the layer choice.")
        contradicts = True
    return {"diverged_cells": bad, "v9_validation_verdict": text,
            "contradicts_locked_spec": contradicts}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--thinking-path", default=DEFAULT_THINKING_PATH)
    ap.add_argument("--rltt-path", default=DEFAULT_RLTT_PATH)
    ap.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    ap.add_argument("--head-checkpoint", default=DEFAULT_HEAD_PATH)
    ap.add_argument("--max-examples", type=int, default=200)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-capture", action="store_true",
                    help="reuse existing .pt files in --output-dir")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    rltt_pt = out_dir / "hh_layer_states_200_rltt.pt"
    thinking_pt = out_dir / "hh_layer_states_200_thinking.pt"
    device = torch.device(args.device)

    examples = None

    def _examples():
        nonlocal examples
        if examples is None:
            examples = select_examples(args.seed, args.max_examples)
        return examples

    # Step A: RLTT
    if args.skip_capture and rltt_pt.exists():
        print(f"[A] reusing {rltt_pt}")
        packs_r, meta_r = load_packs(rltt_pt)
    else:
        packs_r, meta_r = capture_backbone(
            args.rltt_path, args.tokenizer_path, _examples(),
            args.max_length, device, label="RLTT")
        save_packs(rltt_pt, packs_r, meta_r)

    # Step B: Thinking
    if args.skip_capture and thinking_pt.exists():
        print(f"[B] reusing {thinking_pt}")
        packs_t, meta_t = load_packs(thinking_pt)
    else:
        packs_t, meta_t = capture_backbone(
            args.thinking_path, args.tokenizer_path, _examples(),
            args.max_length, device, label="Thinking")
        save_packs(thinking_pt, packs_t, meta_t)

    # Step D: analysis
    table = cross_backbone(packs_t, packs_r)
    head = head_agreement(args.head_checkpoint, packs_t, packs_r, device)
    verdict = v9_verdict(table)

    print("\n=== Cross-backbone mean cos(h_T, h_R) (3 layers × 4 loops) ===")
    print(f"{'':>6}" + "".join(f"{'loop'+str(i+1):>12}"
                               for i in range(NUM_LOOPS)))
    for ly in TAP_LAYERS:
        row = f"L{ly:>4}"
        for li in range(NUM_LOOPS):
            row += f"{table[f'L{ly}'][f'loop{li+1}']['mean']:>12.4f}"
        print(row)
    for ly in TAP_LAYERS:
        for li in range(NUM_LOOPS):
            c = table[f"L{ly}"][f"loop{li+1}"]
            print(f"  L{ly}/loop{li+1}: {c['mean']:+.4f} "
                  f"(std {c['std']:.4f}) -> {c['verdict']}")

    print("\n=== Published head sanity (L47 boundary, vs v10) ===")
    print(f"  Pearson(score_T, score_R)  : {head['pearson_score_T_vs_R']:.4f}"
          f"   (v10: 0.9906)")
    print(f"  decision agreement rate    : "
          f"{head['decision_agreement_rate']:.4f}   (v10: 0.9950)")
    print(f"  canonical_acc Thinking     : "
          f"{head['canonical_acc_thinking']:.4f}   (v10: 0.9500)")
    print(f"  canonical_acc RLTT         : "
          f"{head['canonical_acc_rltt']:.4f}   (v10: 0.9450)")

    print(f"\nv9-VALIDATION VERDICT:\n  {verdict['v9_validation_verdict']}")
    if verdict["contradicts_locked_spec"]:
        print("  ** This CONTRADICTS the locked §4 spec. STOP: report, do "
              "not work around, do not start Experiment 2 capture. **")

    result = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "capture_meta": {"rltt": meta_r, "thinking": meta_t},
        "cross_backbone_table": table,
        "head_agreement": head,
        "verdict": verdict,
    }
    out_json = out_dir / "probe_cross_backbone_layers.json"
    out_json.write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
