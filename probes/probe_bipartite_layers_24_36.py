"""PROBE 1 — Layer 24/36/47 bipartite loop geometry on Ouro-RLTT.

v10 established L1<->L4 separation only at the boundary (layer 47). The
Ouro-RLTT-BG architecture (synthesis §4) places taps at layers 24, 36, 47,
each reading (L1, L4) of its own layer. This probe measures whether
layers 24 and 36 also have the bipartite structure, or whether their
loop states cluster — in which case the (L1, L4) concat parameterization
at those taps is wrong.

Reads the RLTT layer states captured by layer_state_capture (shared with
Probe 2 — capture once). RLTT only; the cross-backbone question is Probe 2.

Outputs:
  artifacts/reports/evaluator/probe_bipartite_layers_24_36.json
  (the spec's runs/ wording; artifacts/reports/evaluator/ is the repo
   convention established by v10.)

STOP semantics: if the final verdict is anything other than "all three
layers bipartite -> Phase 1 architecture unchanged", that contradicts the
locked §4 spec and must be reported, not worked around.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from layer_state_capture import (DEFAULT_OUTPUT_DIR, DEFAULT_RLTT_PATH,
                                  DEFAULT_TOKENIZER_PATH, NUM_LOOPS, TAP_LAYERS,
                                  ExamplePack, capture_backbone, load_packs,
                                  resolve_local, save_packs, select_examples)

LOOP_PAIRS = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]


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


def analyse_layer(packs: List[ExamplePack], layer: int) -> Dict:
    # pooled[layer]: [NUM_LOOPS, H] per side per example
    pc = [p.chosen.pooled[layer] for p in packs]
    pr = [p.rejected.pooled[layer] for p in packs]
    n = len(packs)

    # (a) cross-loop pair cosines, n=400 (chosen + rejected pooled).
    pair_cos: Dict[str, Dict[str, float]] = {}
    for i, j in LOOP_PAIRS:
        vals = []
        for k in range(n):
            vals.append(_cos(pc[k][i - 1], pc[k][j - 1]))
            vals.append(_cos(pr[k][i - 1], pr[k][j - 1]))
        pair_cos[f"L{i}_L{j}"] = _stats(vals)
    off_diag_mean = float(np.mean([v["mean"] for v in pair_cos.values()]))
    off_diag_min = float(np.min([v["mean"] for v in pair_cos.values()]))

    # (b) per-loop L2 norm, n=400.
    norm_per_loop: Dict[str, Dict[str, float]] = {}
    for li in range(NUM_LOOPS):
        vals = []
        for k in range(n):
            vals.append(float(pc[k][li].norm()))
            vals.append(float(pr[k][li].norm()))
        norm_per_loop[f"L{li+1}"] = _stats(vals)

    # (c) diff-vector cross-loop cosines over the 200 example pairs.
    diffs = [[pc[k][li] - pr[k][li] for li in range(NUM_LOOPS)]
             for k in range(n)]
    diff_cos: Dict[str, Dict[str, float]] = {}
    for i, j in LOOP_PAIRS:
        diff_cos[f"d{i}_d{j}"] = _stats(
            [_cos(diffs[k][i - 1], diffs[k][j - 1]) for k in range(n)])

    l1l4 = pair_cos["L1_L4"]["mean"]
    l2l4 = pair_cos["L2_L4"]["mean"]
    if off_diag_mean < 0.40:
        verdict = "fully distributed"
    elif off_diag_min > 0.90:
        verdict = "fully converged"
    elif l1l4 < 0.85 and l2l4 > 0.90:
        verdict = "bipartite"
    else:
        verdict = "intermediate / unclear"

    return {
        "layer": layer,
        "pair_cos": pair_cos,
        "norm_per_loop": norm_per_loop,
        "diff_vector_cross_loop_cos": diff_cos,
        "summary": {
            "L1_L4_cos": l1l4,
            "L2_L4_cos": l2l4,
            "mean_off_diag_cos": off_diag_mean,
            "min_off_diag_cos": off_diag_min,
            "verdict": verdict,
        },
    }


def bg_verdict(per_layer: Dict[int, Dict]) -> Dict:
    v = {ly: per_layer[ly]["summary"]["verdict"] for ly in TAP_LAYERS}
    v24, v36, v47 = v[24], v[36], v[47]

    if v24 == "bipartite" and v36 == "bipartite" and v47 == "bipartite":
        text = ("BG taps at all three layers use (L1, L4) concat. "
                "Phase 1 architecture unchanged.")
        contradicts = False
    elif v24 == "fully converged" or v36 == "fully converged":
        text = ("BG tap at converged layer should use L4 only, not concat. "
                "Phase 1 architecture differs from Experiment 2 for that "
                "layer.")
        contradicts = True
    elif len({v24, v36, v47}) == 3:
        text = ("BG taps require per-layer parameterization. Phase 1 "
                "architecture needs revision — see results before "
                "proceeding.")
        contradicts = True
    else:
        text = ("Mixed / unclear regimes across layers — manual review "
                "required before Phase 1; do not assume (L1, L4) concat.")
        contradicts = True

    # Layer 47 not bipartite on RLTT is itself a W2-style finding that
    # retroactively complicates the Experiment 2 spec (spec note).
    l47_flag = (v47 != "bipartite")
    return {
        "per_layer_verdict": v,
        "bg_architecture_verdict": text,
        "contradicts_locked_spec": contradicts,
        "layer47_not_bipartite_on_rltt": l47_flag,
    }


def _v10_l47_sanity(out_dir: Path, per_layer: Dict[int, Dict]) -> Dict:
    """Informational: our L47 uses v10's boundary convention on the same
    200 pairs / same model, so L1_L4 etc. should match v10's RLTT
    intrinsic numbers within bf16 nondeterminism."""
    v10 = out_dir / "probe_loop_geometry_hh.json"
    if not v10.exists():
        return {"available": False}
    j = json.loads(v10.read_text())
    ref = j.get("intrinsic_rltt", {}).get("pair_cos_pooled", {})
    ours = per_layer[47]["pair_cos"]
    cmp = {}
    for key in ("L1_L4", "L2_L4", "L3_L4", "L1_L2"):
        if key in ref and key in ours:
            cmp[key] = {
                "v10_rltt": ref[key]["mean"],
                "probe1_l47": ours[key]["mean"],
                "abs_delta": abs(ref[key]["mean"] - ours[key]["mean"]),
            }
    return {"available": True, "comparison": cmp}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--rltt-path", default=DEFAULT_RLTT_PATH)
    ap.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    ap.add_argument("--max-examples", type=int, default=200)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(resolve_local(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    rltt_pt = out_dir / "hh_layer_states_200_rltt.pt"

    if rltt_pt.exists():
        print(f"[Probe 1] reusing {rltt_pt}")
        packs, meta = load_packs(rltt_pt)
    else:
        print(f"[Probe 1] {rltt_pt} missing — capturing RLTT now "
              f"(run probe_cross_backbone_layers.py to also get Thinking)")
        examples = select_examples(args.seed, args.max_examples)
        packs, meta = capture_backbone(
            args.rltt_path, args.tokenizer_path, examples,
            args.max_length, torch.device(args.device), label="RLTT")
        save_packs(rltt_pt, packs, meta)

    per_layer = {ly: analyse_layer(packs, ly) for ly in TAP_LAYERS}
    verdict = bg_verdict(per_layer)
    sanity = _v10_l47_sanity(out_dir, per_layer)

    for ly in TAP_LAYERS:
        s = per_layer[ly]["summary"]
        print(f"\n=== Layer {ly} (RLTT) ===")
        for k, st in per_layer[ly]["pair_cos"].items():
            print(f"  {k:>6}  mean={st['mean']:+.4f}  std={st['std']:.4f}")
        print(f"  L1_L4={s['L1_L4_cos']:+.4f}  L2_L4={s['L2_L4_cos']:+.4f}"
              f"  mean_off_diag={s['mean_off_diag_cos']:+.4f}"
              f"  min_off_diag={s['min_off_diag_cos']:+.4f}")
        print(f"  per-loop norm: " + "  ".join(
            f"{k}={v['mean']:.2f}" for k, v
            in per_layer[ly]["norm_per_loop"].items()))
        print(f"  VERDICT: {s['verdict']}")

    if sanity.get("available"):
        print("\n[L47 vs v10 RLTT intrinsic — sanity, informational]")
        for k, c in sanity["comparison"].items():
            print(f"  {k}: v10={c['v10_rltt']:+.4f} "
                  f"probe1={c['probe1_l47']:+.4f} "
                  f"|Δ|={c['abs_delta']:.4f}")

    print(f"\nBG-ARCHITECTURE VERDICT:\n  {verdict['bg_architecture_verdict']}")
    if verdict["layer47_not_bipartite_on_rltt"]:
        print("  ** WARNING: layer 47 is NOT bipartite on RLTT — W2-style "
              "finding, complicates the Experiment 2 spec itself. **")
    if verdict["contradicts_locked_spec"]:
        print("  ** This CONTRADICTS the locked §4 spec. STOP: report, do "
              "not work around, do not start Experiment 2 capture. **")

    result = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "capture_meta": meta,
        "per_layer": per_layer,
        "verdict": verdict,
        "l47_v10_sanity": sanity,
    }
    out_json = out_dir / "probe_bipartite_layers_24_36.json"
    out_json.write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
