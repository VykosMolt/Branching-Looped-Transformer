"""Comprehensive zero-shot ablation table for the pairwise evaluator.

Extends probe_evaluator_hypothesis.py v3 with:

  - Norm-applied variants:   model.model.norm() applied to mid-layer captures
                             before scoring. Tests whether layer 47 -> boundary
                             gap (+6.7pp) is the OuroRMSNorm or the cross-loop
                             GRU.

  - Pair and pattern configs that loop 2's signal in different positions:
       pair_1_2_alt   = [h1, h2, h1, h2]
       pair_2_3_alt   = [h2, h3, h2, h3]
       pair_2_4_alt   = [h2, h4, h2, h4]
       trunc_to_loop2 = [h1, h2, h2, h2]
       start_from_loop2 = [h2, h3, h4, h4]

  - Mean configurations:
       mean_all_replicated     = [mean(h1..h4)] x 4
       mean_loops_234_replicated = [mean(h2, h3, h4)] x 4

  - Loop-2 stress: apply OuroRMSNorm to h_loop2 a second time, score with that.

  - Masked-loop variants (per GPT item 6):
       mask_zero_only_loop2 = [0, h2, 0, 0]   literal zeros at masked positions
       mask_zero_loop2_front = [h2, 0, 0, 0]
     Caveat: diff-norm at zero positions can produce NaN (LayerNorm on
     all-zero diff). Recorded as nan/inf if so.

All probes zero-shot: same pairwise_epoch2.pt head everywhere.
Statistics: paired bootstrap CIs + McNemar vs boundary + antisymmetry on a
subset of flip-tested configs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

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
PROBE_LAYERS = (4, 12, 24, 36, 47)
NUM_UT_LOOPS = 4
NUM_DECODER_LAYERS = 48
FLIP_SUBSET = (
    "boundary",
    "only_loop_2",
    "layer_24",
    "layer_24_normed",
    "layer_47_normed",
    "mean_all_replicated",
    "loop2_doublenorm",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument("--dataset", default="Anthropic/hh-rlhf")
    p.add_argument("--split", default="test")
    p.add_argument("--max-examples", type=int, default=1000)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--early-exit-threshold", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-json", default="artifacts/reports/evaluator/probe_evaluator_v4_ablations.json")
    p.add_argument("--bootstrap-iters", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=0)
    return p.parse_args()


class MultiHookCapture:
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
            tensor = output if isinstance(output, torch.Tensor) else output[0]
            self.layer_states[layer_idx].append(tensor.detach())
        return hook

    def clear(self) -> None:
        self.boundary_states = []
        for L in self.probe_layer_indices:
            self.layer_states[L] = []


def attach_hooks(model, capture: MultiHookCapture) -> List:
    handles = [model.model.register_forward_hook(capture.boundary_hook)]
    for L in capture.probe_layer_indices:
        handles.append(model.model.layers[L].register_forward_hook(capture.make_layer_hook(L)))
    return handles


def to_fp32(states: List[torch.Tensor], device: torch.device) -> List[torch.Tensor]:
    return [h.to(device=device, dtype=torch.float32) for h in states]


def evaluator_score(
    evaluator: PairwiseEvaluator,
    states_c: List[torch.Tensor],
    mask_c: torch.Tensor,
    states_r: List[torch.Tensor],
    mask_r: torch.Tensor,
) -> float:
    with torch.no_grad():
        s = evaluator(states_c, mask_c, states_r, mask_r)
    return float(s.view(-1).item())


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def build_configurations(
    c_boundary: List[torch.Tensor],
    r_boundary: List[torch.Tensor],
    c_layer: Dict[int, List[torch.Tensor]],
    r_layer: Dict[int, List[torch.Tensor]],
    norm_fn: Callable[[torch.Tensor], torch.Tensor],
    probe_layers: List[int],
) -> Dict[str, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
    """Materialize all configurations for one example (per side).

    Returns dict mapping config_name -> (chosen_states_list, rejected_states_list).
    Each states_list has exactly NUM_UT_LOOPS=4 tensors.
    """
    c_b, r_b = c_boundary, r_boundary
    cfg: Dict[str, Tuple[List[torch.Tensor], List[torch.Tensor]]] = {}

    # Canonical
    cfg["boundary"] = (list(c_b), list(r_b))

    # Per-loop ablation (replicated 4x)
    for n in range(NUM_UT_LOOPS):
        cfg[f"only_loop_{n+1}"] = ([c_b[n]] * 4, [r_b[n]] * 4)

    # Mid-layer raw
    for L in probe_layers:
        cfg[f"layer_{L}"] = (list(c_layer[L]), list(r_layer[L]))

    # Mid-layer with model.model.norm applied
    for L in probe_layers:
        cfg[f"layer_{L}_normed"] = (
            [norm_fn(h) for h in c_layer[L]],
            [norm_fn(h) for h in r_layer[L]],
        )

    # Pair configurations
    cfg["pair_1_2_alt"] = ([c_b[0], c_b[1], c_b[0], c_b[1]], [r_b[0], r_b[1], r_b[0], r_b[1]])
    cfg["pair_2_3_alt"] = ([c_b[1], c_b[2], c_b[1], c_b[2]], [r_b[1], r_b[2], r_b[1], r_b[2]])
    cfg["pair_2_4_alt"] = ([c_b[1], c_b[3], c_b[1], c_b[3]], [r_b[1], r_b[3], r_b[1], r_b[3]])

    # Truncation / extension
    cfg["trunc_to_loop2"]    = ([c_b[0], c_b[1], c_b[1], c_b[1]], [r_b[0], r_b[1], r_b[1], r_b[1]])
    cfg["start_from_loop2"]  = ([c_b[1], c_b[2], c_b[3], c_b[3]], [r_b[1], r_b[2], r_b[3], r_b[3]])
    cfg["loop2_then_zeros"]  = ([c_b[1]] + [torch.zeros_like(c_b[1])] * 3, [r_b[1]] + [torch.zeros_like(r_b[1])] * 3)

    # Mean configurations
    c_mean_all = torch.stack(c_b, dim=0).mean(dim=0)
    r_mean_all = torch.stack(r_b, dim=0).mean(dim=0)
    cfg["mean_all_replicated"] = ([c_mean_all] * 4, [r_mean_all] * 4)

    c_mean_234 = torch.stack(c_b[1:], dim=0).mean(dim=0)
    r_mean_234 = torch.stack(r_b[1:], dim=0).mean(dim=0)
    cfg["mean_loops_234_replicated"] = ([c_mean_234] * 4, [r_mean_234] * 4)

    c_mean_23 = torch.stack(c_b[1:3], dim=0).mean(dim=0)
    r_mean_23 = torch.stack(r_b[1:3], dim=0).mean(dim=0)
    cfg["mean_loops_23_replicated"] = ([c_mean_23] * 4, [r_mean_23] * 4)

    # Loop-2 stress: a second OuroRMSNorm application on boundary loop 2.
    # (Boundary loop 2 already has one self.norm baked into _run_single_ut_loop;
    # this is an extra application on top.)
    cfg["loop2_doublenorm"] = ([norm_fn(c_b[1])] * 4, [norm_fn(r_b[1])] * 4)

    # Masked-loop variants (GPT item 6)
    zero_c = torch.zeros_like(c_b[1])
    zero_r = torch.zeros_like(r_b[1])
    cfg["mask_zero_only_loop2"]  = ([zero_c, c_b[1], zero_c, zero_c], [zero_r, r_b[1], zero_r, zero_r])
    cfg["mask_zero_loop2_front"] = ([c_b[1], zero_c, zero_c, zero_c], [r_b[1], zero_r, zero_r, zero_r])

    return cfg


def main() -> None:
    args = parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_MODULES_CACHE", str(Path.cwd() / ".hf_modules_cache"))

    device = torch.device(args.device)
    model_path = str(Path(args.model_path))
    checkpoint_path = str(Path(args.checkpoint))

    probe_layers = list(PROBE_LAYERS)
    print(f"Model: {model_path}")
    print(f"Evaluator: {checkpoint_path}")
    print(f"Probe layers (out of {NUM_DECODER_LAYERS}): {probe_layers}")
    print(f"Examples: {args.max_examples}, max_length={args.max_length}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model (base, no wrapper)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map={"": str(device)}, torch_dtype=torch.bfloat16,
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "config"):
        model.config.early_exit_threshold = args.early_exit_threshold

    # Norm function: model.model.norm (OuroRMSNorm). Apply to fp32 tensors;
    # cast to bf16 to match weights, run, cast back to fp32.
    @torch.no_grad()
    def norm_fn(x: torch.Tensor) -> torch.Tensor:
        x_bf = x.to(dtype=torch.bfloat16)
        n = model.model.norm(x_bf)
        return n.to(dtype=torch.float32)

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
    n_total = len(ds)
    print(f"Examples to evaluate: {n_total}")

    # Probe each example once to get config names (and validate shapes).
    series: Dict[str, List[float]] = {}
    flipped: Dict[str, List[float]] = {k: [] for k in FLIP_SUBSET}
    correct: Dict[str, int] = {}
    init_done = False

    for idx in range(n_total):
        chosen_text = ds[idx]["chosen"]
        rejected_text = ds[idx]["rejected"]

        tokens_c = tokenizer(chosen_text, return_tensors="pt", truncation=True,
                             max_length=args.max_length).to(device)
        tokens_r = tokenizer(rejected_text, return_tensors="pt", truncation=True,
                             max_length=args.max_length).to(device)

        capture.clear()
        with torch.no_grad():
            model(**tokens_c, use_cache=False)
        c_boundary = to_fp32(capture.boundary_states, device)
        c_layer = {L: to_fp32(capture.layer_states[L], device) for L in probe_layers}
        c_mask = tokens_c["attention_mask"]

        capture.clear()
        with torch.no_grad():
            model(**tokens_r, use_cache=False)
        r_boundary = to_fp32(capture.boundary_states, device)
        r_layer = {L: to_fp32(capture.layer_states[L], device) for L in probe_layers}
        r_mask = tokens_r["attention_mask"]

        # Sanity: each layer hook should fire exactly NUM_UT_LOOPS times.
        if any(len(c_layer[L]) != NUM_UT_LOOPS for L in probe_layers):
            print(f"WARN: layer hook fire count off at example {idx}; skipping.")
            continue

        cfgs = build_configurations(c_boundary, r_boundary, c_layer, r_layer,
                                    norm_fn, probe_layers)

        if not init_done:
            for k in cfgs:
                series[k] = []
                correct[k] = 0
            init_done = True

        for k, (cs, rs) in cfgs.items():
            s = evaluator_score(evaluator, cs, c_mask, rs, r_mask)
            # NaN/inf safety for masked variants.
            if not math.isfinite(s):
                series[k].append(float("nan"))
                continue
            series[k].append(s)
            if s > 0:
                correct[k] += 1

        # Flip subset
        for k in FLIP_SUBSET:
            if k in cfgs:
                cs, rs = cfgs[k]
                s = evaluator_score(evaluator, rs, r_mask, cs, c_mask)
                flipped[k].append(s if math.isfinite(s) else float("nan"))

        if (idx + 1) % 25 == 0 or idx == n_total - 1:
            n_so_far = idx + 1
            line = (f"[{n_so_far}/{n_total}] "
                    f"boundary={correct['boundary']/n_so_far:.3f} "
                    f"only_loop_2={correct['only_loop_2']/n_so_far:.3f} "
                    f"layer_24={correct['layer_24']/n_so_far:.3f} "
                    f"layer_24_normed={correct['layer_24_normed']/n_so_far:.3f} "
                    f"layer_47_normed={correct['layer_47_normed']/n_so_far:.3f} "
                    f"mean_all={correct['mean_all_replicated']/n_so_far:.3f}")
            print(line)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for h in handles:
        h.remove()

    # ---- Aggregate ----
    boundary_arr = np.array(series["boundary"], dtype=np.float64)
    boundary_correct = boundary_arr > 0
    n = boundary_arr.size

    summary: Dict[str, Dict[str, float]] = {}
    for k in series:
        arr = np.array(series[k], dtype=np.float64)
        # NaN handling: treat NaN as "wrong" for accuracy.
        valid = np.isfinite(arr)
        depth_correct = (arr > 0) & valid
        n_valid = int(valid.sum())

        d = {
            "accuracy": float(np.mean(depth_correct)) if n else 0.0,
            "n_valid": n_valid,
            "n_nan": int((~valid).sum()),
        }
        if n_valid > 0:
            valid_arr = arr[valid]
            d["mean_score"] = float(np.mean(valid_arr))
            d["std_score"] = float(np.std(valid_arr))
        else:
            d["mean_score"] = float("nan")
            d["std_score"] = float("nan")

        if n_valid > 0:
            d["pearson_with_boundary"] = pearson(arr * valid, boundary_arr * valid)
            d["sign_agreement_with_boundary"] = float(np.mean(
                np.sign(np.where(valid, arr, 0)) == np.sign(boundary_arr)
            ))
        else:
            d["pearson_with_boundary"] = 0.0
            d["sign_agreement_with_boundary"] = 0.0

        # McNemar vs boundary (treating NaN as wrong).
        n_b1_d0 = int(np.sum(boundary_correct & ~depth_correct))
        n_b0_d1 = int(np.sum(~boundary_correct & depth_correct))
        if (n_b1_d0 + n_b0_d1) > 0:
            chi2 = max(0.0, (abs(n_b1_d0 - n_b0_d1) - 1) ** 2 / (n_b1_d0 + n_b0_d1))
            mc_p = math.erfc(math.sqrt(chi2 / 2.0))
        else:
            chi2, mc_p = 0.0, 1.0
        d["mcnemar_n_boundary_only_correct"] = n_b1_d0
        d["mcnemar_n_depth_only_correct"] = n_b0_d1
        d["mcnemar_chi2_cc"] = float(chi2)
        d["mcnemar_p_two_sided"] = float(mc_p)
        summary[k] = d

    for k in summary:
        summary[k]["accuracy_delta_vs_boundary"] = (
            summary[k]["accuracy"] - summary["boundary"]["accuracy"]
        )

    # Paired bootstrap on accuracy_delta_vs_boundary
    rng = np.random.default_rng(args.bootstrap_seed)
    if n > 0 and args.bootstrap_iters > 0:
        idx_matrix = rng.integers(0, n, size=(args.bootstrap_iters, n))
        correct_arr_d: Dict[str, np.ndarray] = {}
        for k in series:
            arr = np.array(series[k], dtype=np.float64)
            ok = (arr > 0) & np.isfinite(arr)
            correct_arr_d[k] = ok.astype(np.float64)
        b_correct = correct_arr_d["boundary"]
        for k in series:
            if k == "boundary":
                summary[k]["bootstrap_ci_95"] = [0.0, 0.0]
                continue
            d_correct = correct_arr_d[k]
            deltas = d_correct[idx_matrix].mean(axis=1) - b_correct[idx_matrix].mean(axis=1)
            summary[k]["bootstrap_ci_95"] = [
                float(np.percentile(deltas, 2.5)),
                float(np.percentile(deltas, 97.5)),
            ]

    # Antisymmetry on flip subset
    for k in FLIP_SUBSET:
        if k not in series or not flipped[k]:
            continue
        s_norm = np.array(series[k], dtype=np.float64)
        s_flip = np.array(flipped[k], dtype=np.float64)
        valid = np.isfinite(s_norm) & np.isfinite(s_flip)
        if not valid.any():
            continue
        s_norm_v, s_flip_v = s_norm[valid], s_flip[valid]
        summary[k]["antisymmetry_pearson"] = pearson(s_norm_v, -s_flip_v)
        summary[k]["antisymmetry_mean_abs_residual"] = float(np.mean(np.abs(s_norm_v + s_flip_v)))
        summary[k]["antisymmetry_sign_flip_rate"] = float(
            np.mean(np.sign(s_norm_v) != np.sign(s_flip_v))
        )

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
        "flip_subset": list(FLIP_SUBSET),
        "per_config": summary,
        "raw_scores": {k: [float(x) for x in v] for k, v in series.items()},
        "raw_scores_flipped": {k: [float(x) for x in v] for k, v in flipped.items() if v},
    }

    print("\n=== SUMMARY (sorted by accuracy desc) ===")
    print(f"n = {n}")
    sorted_keys = sorted(summary.keys(), key=lambda k: -summary[k]["accuracy"])
    for k in sorted_keys:
        d = summary[k]
        ci = d.get("bootstrap_ci_95", [0.0, 0.0])
        anti = d.get("antisymmetry_pearson", float("nan"))
        nan_note = f" (nan={d['n_nan']})" if d["n_nan"] else ""
        anti_note = f" antisym_r={anti:+.3f}" if not math.isnan(anti) else ""
        print(f"  {k:>28} | acc={d['accuracy']:.3f} (Δ={d['accuracy_delta_vs_boundary']:+.3f}"
              f" CI[{ci[0]:+.3f},{ci[1]:+.3f}]) | "
              f"r_vs_bnd={d['pearson_with_boundary']:+.3f} | "
              f"McN p={d['mcnemar_p_two_sided']:.1e}{anti_note}{nan_note}")

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
