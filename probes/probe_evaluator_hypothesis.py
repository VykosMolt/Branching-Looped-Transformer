"""Discriminate "loop convergence" vs "proto introspection" hypotheses for the
pairwise evaluator (arxiv:2604.09870 architecture, pairwise_epoch2.pt checkpoint).

The evaluator was trained on per-UT-loop hidden states (4 states per response,
captured at the end of each loop after `self.norm`). Two competing theories for
*why* it discriminates chosen vs rejected:

  1. Loop convergence: the signal is in how the 4 per-loop states settle.
     Requires multiple loops to manifest. Mid-loop (within a single UT loop)
     hidden states should score poorly.
  2. Proto introspection: the signal is a learned geometric correlate of
     "this is right" baked into hidden-state representations. Present at
     every layer, every loop. Mid-loop states should score well.

This probe captures hidden states at four depths within each UT loop
(layers 12, 24, 36, 47 of the 48-layer decoder block), packages them as
the 4-state input the evaluator expects, and compares scores against the
canonical loop-boundary capture.

Note on layer 47 capture: it is the state *just before* `self.norm` is
applied at the end of `_run_single_ut_loop`. So it should be highly but
not perfectly correlated with the canonical loop-boundary state.

Outputs:
- accuracy at each capture depth
- Pearson + Spearman correlation of each mid-layer's per-example score
  series against the canonical loop-boundary score series
- mean and std of the score distribution per depth

Interpretation:
- Strong correlation (> 0.7) + high mid-layer accuracy → proto introspection.
  The signal is present mid-loop. Phase 4 in-loop integration is viable.
- Weak correlation (< 0.4) or accuracy near 50% → loop convergence.
  Phase 4 cannot fire mid-loop usefully; must batch loops before gating.
- Climbing correlation across depths → both, hybrid signal.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator_core.pairwise_evaluator import PairwiseEvaluator, validate_hook_output


DEFAULT_MODEL_PATH = "models/ouro_rltt_local"
DEFAULT_CHECKPOINT_PATH = "artifacts/checkpoints/evaluator/pairwise_epoch2.pt"
# Dense sweep: catch the curve shape around the layer 24 / layer 36 transition
# we saw in the first run, plus quarter-loop landmarks.
PROBE_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 47)
NUM_UT_LOOPS = 4
NUM_DECODER_LAYERS = 48


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument("--dataset", default="Anthropic/hh-rlhf")
    p.add_argument("--split", default="test")
    p.add_argument("--max-examples", type=int, default=300)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--early-exit-threshold", type=float, default=1.0,
                   help="Set to 1.0 so all 4 UT loops always execute (no adaptive exit).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-json", default="artifacts/reports/evaluator/probe_evaluator_hypothesis_v2.json")
    p.add_argument("--probe-layers", default=",".join(str(L) for L in PROBE_LAYERS),
                   help="Comma-separated 0-indexed decoder layer indices to probe.")
    p.add_argument("--skip-per-loop-ablation", action="store_true",
                   help="Disable Probe 1b: per-loop ablation (feed only one loop's state into all 4 evaluator slots).")
    p.add_argument("--skip-flip", action="store_true",
                   help="Disable antisymmetry flip tests: scoring (rejected, chosen) at each depth.")
    p.add_argument("--bootstrap-iters", type=int, default=2000,
                   help="Paired bootstrap iterations for accuracy-delta CIs vs boundary.")
    p.add_argument("--bootstrap-seed", type=int, default=0)
    return p.parse_args()


class MultiHookCapture:
    """Captures the canonical model-level loop-boundary states AND per-layer
    intermediate states for a configured set of decoder-layer indices.

    The decoder layer at index L is called once per UT loop (4 times per
    forward pass). The hook fires 4 times in temporal order, so we just
    append each captured tensor to a per-layer list.

    Boundary hook captures `output[1]` (per-loop hidden states post-norm),
    matching evaluate_pairwise_rltt.py's LoopStateCapture.
    """

    def __init__(self, probe_layer_indices: List[int]) -> None:
        self.probe_layer_indices = list(probe_layer_indices)
        self.boundary_states: List[torch.Tensor] = []
        self.layer_states: Dict[int, List[torch.Tensor]] = {L: [] for L in probe_layer_indices}
        self.boundary_validated = False

    def boundary_hook(self, module, inputs, output) -> None:
        hidden_states = output[1]
        if not self.boundary_validated:
            validate_hook_output(hidden_states)
            self.boundary_validated = True
        self.boundary_states = [h.detach() for h in hidden_states]

    def make_layer_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            # OuroDecoderLayer.forward returns the hidden_states tensor directly.
            tensor = output if isinstance(output, torch.Tensor) else output[0]
            self.layer_states[layer_idx].append(tensor.detach())
        return hook

    def clear(self) -> None:
        self.boundary_states = []
        for L in self.probe_layer_indices:
            self.layer_states[L] = []


def attach_hooks(model: torch.nn.Module, capture: MultiHookCapture) -> List:
    handles = []
    handles.append(model.model.register_forward_hook(capture.boundary_hook))
    for L in capture.probe_layer_indices:
        handles.append(model.model.layers[L].register_forward_hook(capture.make_layer_hook(L)))
    return handles


@torch.no_grad()
def run_forward(model, tokens) -> None:
    model(**tokens, use_cache=False)


def to_fp32_on(states: List[torch.Tensor], device: torch.device) -> List[torch.Tensor]:
    return [h.to(device=device, dtype=torch.float32) for h in states]


def evaluator_score(
    evaluator: PairwiseEvaluator,
    states_c: List[torch.Tensor],
    mask_c: torch.Tensor,
    states_r: List[torch.Tensor],
    mask_r: torch.Tensor,
) -> float:
    score = evaluator(states_c, mask_c, states_r, mask_r)
    return float(score.view(-1).item())


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return 0.0
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return pearson(ra.astype(np.float64), rb.astype(np.float64))


def main() -> None:
    args = parse_args()
    probe_layers = [int(x) for x in args.probe_layers.split(",") if x.strip()]
    for L in probe_layers:
        assert 0 <= L < NUM_DECODER_LAYERS, f"layer {L} out of range"

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_MODULES_CACHE", str(Path.cwd() / ".hf_modules_cache"))

    device = torch.device(args.device)
    model_path = str(Path(args.model_path))
    checkpoint_path = str(Path(args.checkpoint))

    print(f"Model: {model_path}")
    print(f"Evaluator: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Probe layers (0-indexed, out of {NUM_DECODER_LAYERS}): {probe_layers}")
    print(f"Examples: {args.max_examples}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model (base, no wrapper)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": str(device)},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "config"):
        model.config.early_exit_threshold = args.early_exit_threshold

    print("Loading evaluator...")
    evaluator = PairwiseEvaluator().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    evaluator.load_state_dict(state_dict)
    evaluator.eval()

    capture = MultiHookCapture(probe_layers)
    handles = attach_hooks(model, capture)

    print("Loading HH-RLHF...")
    ds = load_dataset(args.dataset, split=args.split)
    if args.max_examples > 0:
        ds = ds.select(range(min(args.max_examples, len(ds))))
    print(f"Examples to evaluate: {len(ds)}")

    # Per-depth score lists. "boundary" = canonical loop-boundary capture.
    # "only_loop_N" = Probe 1b: feed loop N's boundary state into ALL 4 slots,
    # i.e. (h_N, h_N, h_N, h_N) instead of (h_1, h_2, h_3, h_4). Tells us how
    # much each individual loop's state alone discriminates -- separates
    # "depth within loop" (layerL probes) from "loop index" (only_loop_N probes).
    series: Dict[str, List[float]] = {"boundary": []}
    for L in probe_layers:
        series[f"layer{L}"] = []
    if not args.skip_per_loop_ablation:
        for n in range(1, NUM_UT_LOOPS + 1):
            series[f"only_loop_{n}"] = []

    # Flipped: score (rejected, chosen) at the SAME depth. Diff-norm-on-symmetric
    # architecture predicts s_flipped == -s_normal at every depth. Antisymmetry
    # breakdown at mid-loop = contamination by absolute (non-relational) features.
    flipped: Dict[str, List[float]] = {}
    if not args.skip_flip:
        for k in series:
            flipped[k] = []

    correct: Dict[str, int] = {k: 0 for k in series}

    for idx in range(len(ds)):
        chosen_text = ds[idx]["chosen"]
        rejected_text = ds[idx]["rejected"]

        tokens_c = tokenizer(chosen_text, return_tensors="pt",
                             truncation=True, max_length=args.max_length).to(device)
        tokens_r = tokenizer(rejected_text, return_tensors="pt",
                             truncation=True, max_length=args.max_length).to(device)

        # Forward chosen, snapshot all hooks.
        capture.clear()
        run_forward(model, tokens_c)
        c_boundary = to_fp32_on(capture.boundary_states, device)
        c_layer: Dict[int, List[torch.Tensor]] = {
            L: to_fp32_on(capture.layer_states[L], device) for L in probe_layers
        }
        c_mask = tokens_c["attention_mask"]

        # Sanity: each layer hook should fire exactly NUM_UT_LOOPS times.
        for L in probe_layers:
            if len(c_layer[L]) != NUM_UT_LOOPS:
                print(f"WARN: layer {L} fired {len(c_layer[L])} times on chosen "
                      f"(expected {NUM_UT_LOOPS}). Skipping example {idx}.")
                continue

        # Forward rejected.
        capture.clear()
        run_forward(model, tokens_r)
        r_boundary = to_fp32_on(capture.boundary_states, device)
        r_layer: Dict[int, List[torch.Tensor]] = {
            L: to_fp32_on(capture.layer_states[L], device) for L in probe_layers
        }
        r_mask = tokens_r["attention_mask"]

        # Boundary score.
        s = evaluator_score(evaluator, c_boundary, c_mask, r_boundary, r_mask)
        series["boundary"].append(s)
        if s > 0:
            correct["boundary"] += 1

        # Per-probe-layer scores.
        for L in probe_layers:
            s = evaluator_score(evaluator, c_layer[L], c_mask, r_layer[L], r_mask)
            series[f"layer{L}"].append(s)
            if s > 0:
                correct[f"layer{L}"] += 1

        # Probe 1b: per-loop ablation.
        if not args.skip_per_loop_ablation:
            for n in range(1, NUM_UT_LOOPS + 1):
                c_replicated = [c_boundary[n - 1]] * NUM_UT_LOOPS
                r_replicated = [r_boundary[n - 1]] * NUM_UT_LOOPS
                s = evaluator_score(evaluator, c_replicated, c_mask, r_replicated, r_mask)
                series[f"only_loop_{n}"].append(s)
                if s > 0:
                    correct[f"only_loop_{n}"] += 1

        # Flip tests: same inputs, swap chosen<->rejected. Predicts s_flipped == -s_normal.
        if not args.skip_flip:
            # boundary flipped
            s = evaluator_score(evaluator, r_boundary, r_mask, c_boundary, c_mask)
            flipped["boundary"].append(s)
            for L in probe_layers:
                s = evaluator_score(evaluator, r_layer[L], r_mask, c_layer[L], c_mask)
                flipped[f"layer{L}"].append(s)
            if not args.skip_per_loop_ablation:
                for n in range(1, NUM_UT_LOOPS + 1):
                    c_replicated = [c_boundary[n - 1]] * NUM_UT_LOOPS
                    r_replicated = [r_boundary[n - 1]] * NUM_UT_LOOPS
                    s = evaluator_score(evaluator, r_replicated, r_mask, c_replicated, c_mask)
                    flipped[f"only_loop_{n}"].append(s)

        if (idx + 1) % 25 == 0 or idx == len(ds) - 1:
            # Compact progress: just boundary + first/last layer + first/last loop.
            n_so_far = idx + 1
            line = f"[{n_so_far}/{len(ds)}] boundary={correct['boundary']/n_so_far:.3f}"
            for L in (probe_layers[0], probe_layers[len(probe_layers)//2], probe_layers[-1]):
                line += f"  L{L}={correct[f'layer{L}']/n_so_far:.3f}"
            if not args.skip_per_loop_ablation:
                for n in (1, NUM_UT_LOOPS):
                    line += f"  loop{n}={correct[f'only_loop_{n}']/n_so_far:.3f}"
            print(line)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for h in handles:
        h.remove()

    # Aggregate.
    boundary_arr = np.array(series["boundary"], dtype=np.float64)
    boundary_correct = boundary_arr > 0
    n = boundary_arr.size

    summary: Dict[str, Dict[str, float]] = {}
    for k in series:
        arr = np.array(series[k], dtype=np.float64)
        depth_correct = arr > 0
        d = {
            "accuracy": float(np.mean(depth_correct)) if n else 0.0,
            "mean_score": float(np.mean(arr)) if n else 0.0,
            "std_score": float(np.std(arr)) if n else 0.0,
            "min_score": float(np.min(arr)) if n else 0.0,
            "max_score": float(np.max(arr)) if n else 0.0,
            "pearson_with_boundary": pearson(arr, boundary_arr),
            "spearman_with_boundary": spearman(arr, boundary_arr),
            "sign_agreement_with_boundary": float(np.mean(np.sign(arr) == np.sign(boundary_arr))),
        }
        # McNemar: paired counts (boundary correct & depth wrong) vs (boundary wrong & depth correct).
        n_b1_d0 = int(np.sum(boundary_correct & ~depth_correct))  # boundary right, depth wrong
        n_b0_d1 = int(np.sum(~boundary_correct & depth_correct))  # depth right, boundary wrong
        # Continuity-corrected McNemar chi-square (1 dof). Two-sided p via chi2 survival.
        if (n_b1_d0 + n_b0_d1) > 0:
            mc_chi2 = (abs(n_b1_d0 - n_b0_d1) - 1) ** 2 / (n_b1_d0 + n_b0_d1)
            mc_chi2 = max(0.0, mc_chi2)
            # Approx p-value from chi2(1) survival; avoid scipy dep.
            # P(chi2 >= x) ≈ erfc(sqrt(x/2)). Good to ~1e-3 for our needs.
            import math
            mc_p = math.erfc(math.sqrt(mc_chi2 / 2.0))
        else:
            mc_chi2 = 0.0
            mc_p = 1.0
        d["mcnemar_n_boundary_only_correct"] = n_b1_d0
        d["mcnemar_n_depth_only_correct"] = n_b0_d1
        d["mcnemar_chi2_cc"] = float(mc_chi2)
        d["mcnemar_p_two_sided"] = float(mc_p)
        d["accuracy_delta_vs_boundary"] = d["accuracy"] - summary.get("boundary", {}).get("accuracy", d["accuracy"]) \
            if "boundary" in summary else 0.0
        summary[k] = d

    # Recompute accuracy_delta for boundary self-comparison (= 0).
    for k in summary:
        summary[k]["accuracy_delta_vs_boundary"] = summary[k]["accuracy"] - summary["boundary"]["accuracy"]

    # Paired bootstrap CIs on accuracy_delta_vs_boundary for each depth.
    rng = np.random.default_rng(args.bootstrap_seed)
    if n > 0 and args.bootstrap_iters > 0:
        idx_matrix = rng.integers(0, n, size=(args.bootstrap_iters, n))
        # Pre-compute correctness arrays per series.
        correct_arr: Dict[str, np.ndarray] = {
            k: (np.array(series[k], dtype=np.float64) > 0).astype(np.float64) for k in series
        }
        b_correct = correct_arr["boundary"]
        for k in series:
            if k == "boundary":
                summary[k]["bootstrap_ci_95"] = [0.0, 0.0]
                continue
            d_correct = correct_arr[k]
            deltas = d_correct[idx_matrix].mean(axis=1) - b_correct[idx_matrix].mean(axis=1)
            lo = float(np.percentile(deltas, 2.5))
            hi = float(np.percentile(deltas, 97.5))
            summary[k]["bootstrap_ci_95"] = [lo, hi]

    # Antisymmetry per depth: corr(s_normal, -s_flipped). Perfect antisymmetry -> 1.0.
    if flipped:
        for k in series:
            s_norm = np.array(series[k], dtype=np.float64)
            s_flip = np.array(flipped[k], dtype=np.float64)
            anti_pearson = pearson(s_norm, -s_flip)
            mean_abs_residual = float(np.mean(np.abs(s_norm + s_flip)))
            # Sign-flip rate: fraction of examples where sign(s_norm) != sign(s_flip).
            sign_flip_rate = float(np.mean(np.sign(s_norm) != np.sign(s_flip)))
            summary[k]["antisymmetry_pearson"] = anti_pearson
            summary[k]["antisymmetry_mean_abs_residual"] = mean_abs_residual
            summary[k]["antisymmetry_sign_flip_rate"] = sign_flip_rate

    result = {
        "model_path": model_path,
        "checkpoint": checkpoint_path,
        "dataset": args.dataset,
        "split": args.split,
        "max_length": args.max_length,
        "early_exit_threshold": args.early_exit_threshold,
        "probe_layers": probe_layers,
        "num_ut_loops": NUM_UT_LOOPS,
        "num_decoder_layers": NUM_DECODER_LAYERS,
        "examples_evaluated": n,
        "evaluator_zero_shot": True,
        "evaluator_zero_shot_note": (
            "The same pairwise_epoch2.pt head (trained on loop-boundary states) "
            "is applied unchanged at every probe depth and to every per-loop "
            "ablation. No per-depth retraining. This is therefore a zero-shot "
            "transfer test of the boundary-trained relational geometry to "
            "mid-loop and single-loop hidden states."
        ),
        "per_depth": summary,
        "raw_scores": {k: [float(x) for x in v] for k, v in series.items()},
    }
    if flipped:
        result["raw_scores_flipped"] = {k: [float(x) for x in v] for k, v in flipped.items()}

    print("\n=== SUMMARY ===")
    print(f"Examples: {n}")
    # Group: boundary first, layers in order, per-loop ablation last.
    ordered_keys = ["boundary"]
    ordered_keys += [f"layer{L}" for L in probe_layers]
    if not args.skip_per_loop_ablation:
        ordered_keys += [f"only_loop_{i}" for i in range(1, NUM_UT_LOOPS + 1)]
    for k in ordered_keys:
        d = summary[k]
        ci = d.get("bootstrap_ci_95", [0.0, 0.0])
        anti = d.get("antisymmetry_pearson", float("nan"))
        print(f"  {k:>14} | acc={d['accuracy']:.3f} (Δ={d['accuracy_delta_vs_boundary']:+.3f}"
              f" CI[{ci[0]:+.3f},{ci[1]:+.3f}]) | "
              f"r_v_bnd={d['pearson_with_boundary']:+.3f} | "
              f"sign_agr={d['sign_agreement_with_boundary']:.3f} | "
              f"McNemar p={d['mcnemar_p_two_sided']:.1e} | "
              f"antisym_r={anti:+.3f}")

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")

    print("\n=== INTERPRETATION HINTS ===")
    print("Strong introspection signal:")
    print("  * mid-layer accuracy >= 0.85 at multiple depths")
    print("  * pearson_with_boundary >= 0.7 at depths >= layer 24")
    print("Strong convergence signal:")
    print("  * mid-layer accuracy near 0.5 (chance) below layer 36")
    print("  * pearson_with_boundary < 0.4 at mid layers")
    print("Hybrid:")
    print("  * accuracy climbs across depths, never reaches boundary level")
    print("  * correlation climbs across depths")


if __name__ == "__main__":
    main()
