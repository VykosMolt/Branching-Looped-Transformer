"""Redo the v4 masked-loop tests properly.

v4 reported 100% accuracy on three zero-masked variants (mask_zero_only_loop2,
mask_zero_loop2_front, loop2_then_zeros). Inspection of the score distributions
revealed this was a sign-degeneracy artifact:

  - score range: [+1.02, +2.37] -- never crosses zero
  - std: ~0.22 (vs boundary's 1.06)
  - pos_rate: 1.000

Mechanism: nn.Linear(bias=True) after nn.LayerNorm(bias=False) outputs the
projection bias when the diff input is zero. Three zero-diff positions out of
four push the GRU's output to a constant-positive region. HH-RLHF's chosen-
first convention does the rest. Not signal.

This script does two things:
  1. Flip-test the original zero-masked variants. If pos_rate is still ~1.0
     when (chosen, rejected) is swapped, the degeneracy is formally confirmed.
  2. Run properly-masked variants that use a NEUTRAL BASELINE instead of zeros
     at masked positions, so LayerNorm has non-zero diff input. Test whether
     these preserve antisymmetry and what accuracy they produce.

Properly-masked variants:
  - loop2_in_loop1_ctx = [h1, h2, h1, h1]      loop1 as baseline
  - loop2_in_loop4_ctx = [h4, h2, h4, h4]      loop4 as baseline
  - loop2_in_mean_ctx  = [mean_loops, h2, mean_loops, mean_loops]
  - loop2_at_pos_2 = [mean, mean, h2, mean]    loop2 at different position
  - loop2_at_pos_3 = [mean, mean, mean, h2]    loop2 at different position

Controls: boundary, only_loop_2 (for comparison to v4 results).

Output: scores + flipped scores, accuracy, pos_rate, std, antisymmetry r.
The properly-masked variants are the actual scientific question; the
zero-masked re-runs are just to formally close the degeneracy claim.
"""
from __future__ import annotations

import argparse
import json
import math
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
NUM_UT_LOOPS = 4


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
    p.add_argument("--output-json", default="artifacts/reports/evaluator/probe_evaluator_v4_redo_masked.json")
    return p.parse_args()


class BoundaryCapture:
    def __init__(self) -> None:
        self.states: List[torch.Tensor] = []
        self.validated = False

    def hook(self, module, inputs, output) -> None:
        hidden_states = output[1]
        if not self.validated:
            validate_hook_output(hidden_states)
            self.validated = True
        self.states = [h.detach() for h in hidden_states]

    def clear(self) -> None:
        self.states = []


def to_fp32(states: List[torch.Tensor], device: torch.device) -> List[torch.Tensor]:
    return [h.to(device=device, dtype=torch.float32) for h in states]


def score(evaluator, cs, c_mask, rs, r_mask) -> float:
    with torch.no_grad():
        s = evaluator(cs, c_mask, rs, r_mask)
    val = float(s.view(-1).item())
    return val if math.isfinite(val) else float("nan")


def build_configurations(
    c_b: List[torch.Tensor], r_b: List[torch.Tensor]
) -> Dict[str, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
    """Build all configurations to test. c_b and r_b are 4 boundary states each."""
    cfg: Dict[str, Tuple[List[torch.Tensor], List[torch.Tensor]]] = {}

    # Controls (match v4)
    cfg["boundary"] = (list(c_b), list(r_b))
    cfg["only_loop_2"] = ([c_b[1]] * 4, [r_b[1]] * 4)

    # Re-run originals (will be flipped to formally confirm degeneracy).
    zero_c = torch.zeros_like(c_b[1])
    zero_r = torch.zeros_like(r_b[1])
    cfg["mask_zero_only_loop2"] = ([zero_c, c_b[1], zero_c, zero_c],
                                    [zero_r, r_b[1], zero_r, zero_r])
    cfg["mask_zero_loop2_front"] = ([c_b[1], zero_c, zero_c, zero_c],
                                     [r_b[1], zero_r, zero_r, zero_r])

    # Neutral-baseline variants: use a real-loop or mean state instead of zeros.
    cfg["loop2_in_loop1_ctx"] = ([c_b[0], c_b[1], c_b[0], c_b[0]],
                                  [r_b[0], r_b[1], r_b[0], r_b[0]])
    cfg["loop2_in_loop4_ctx"] = ([c_b[3], c_b[1], c_b[3], c_b[3]],
                                  [r_b[3], r_b[1], r_b[3], r_b[3]])

    c_mean = torch.stack(c_b, dim=0).mean(dim=0)
    r_mean = torch.stack(r_b, dim=0).mean(dim=0)
    cfg["loop2_in_mean_ctx"] = ([c_mean, c_b[1], c_mean, c_mean],
                                 [r_mean, r_b[1], r_mean, r_mean])

    # Loop 2 at non-canonical positions (with mean as filler).
    cfg["loop2_at_pos_0_mean_ctx"] = ([c_b[1], c_mean, c_mean, c_mean],
                                       [r_b[1], r_mean, r_mean, r_mean])
    cfg["loop2_at_pos_2_mean_ctx"] = ([c_mean, c_mean, c_b[1], c_mean],
                                       [r_mean, r_mean, r_b[1], r_mean])
    cfg["loop2_at_pos_3_mean_ctx"] = ([c_mean, c_mean, c_mean, c_b[1]],
                                       [r_mean, r_mean, r_mean, r_b[1]])

    # Note: "all_zeros" and "identical_loop2" sanity tests were removed because
    # they require chosen and rejected sides to have the same seq length, but
    # text pairs naturally have different lengths. The flip tests on real
    # configurations cover antisymmetry verification anyway.

    return cfg


def build_norm_configurations(
    c_b: List[torch.Tensor], r_b: List[torch.Tensor], norm_fn
) -> Dict[str, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
    """Norm-based configurations: per-loop doublenorm, triplenorm, etc."""
    cfg: Dict[str, Tuple[List[torch.Tensor], List[torch.Tensor]]] = {}
    c_mean = torch.stack(c_b, dim=0).mean(dim=0)
    r_mean = torch.stack(r_b, dim=0).mean(dim=0)

    # Per-loop doublenorm: tests whether doublenorm benefit is specific to loop 2
    # or universal across loops.
    for n in range(NUM_UT_LOOPS):
        c_dn = norm_fn(c_b[n])
        r_dn = norm_fn(r_b[n])
        cfg[f"only_loop_{n+1}_doublenorm"] = ([c_dn] * 4, [r_dn] * 4)

    # Triple-norm on loop 2: does extra norm keep helping, plateau, or degrade?
    c_tn = norm_fn(norm_fn(c_b[1]))
    r_tn = norm_fn(norm_fn(r_b[1]))
    cfg["loop2_triplenorm"] = ([c_tn] * 4, [r_tn] * 4)

    # Quadruple norm — fixed point check.
    c_qn = norm_fn(c_tn)
    r_qn = norm_fn(r_tn)
    cfg["loop2_quadnorm"] = ([c_qn] * 4, [r_qn] * 4)

    # Position-bias test: put OTHER loops at position 1 (loop 2's natural slot).
    # Each filled with mean context so only the loop at position 1 is informative.
    cfg["loop_1_at_pos_1_mean_ctx"] = ([c_mean, c_b[0], c_mean, c_mean],
                                        [r_mean, r_b[0], r_mean, r_mean])
    cfg["loop_3_at_pos_1_mean_ctx"] = ([c_mean, c_b[2], c_mean, c_mean],
                                        [r_mean, r_b[2], r_mean, r_mean])
    cfg["loop_4_at_pos_1_mean_ctx"] = ([c_mean, c_b[3], c_mean, c_mean],
                                        [r_mean, r_b[3], r_mean, r_mean])
    return cfg


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_MODULES_CACHE", str(Path.cwd() / ".hf_modules_cache"))

    device = torch.device(args.device)
    model_path = str(Path(args.model_path))
    checkpoint_path = str(Path(args.checkpoint))

    print(f"Model: {model_path}")
    print(f"Evaluator: {checkpoint_path}")
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

    capture = BoundaryCapture()
    handle = model.model.register_forward_hook(capture.hook)

    print("Loading HH-RLHF...")
    ds = load_dataset(args.dataset, split=args.split)
    if args.max_examples > 0:
        ds = ds.select(range(min(args.max_examples, len(ds))))
    n_total = len(ds)
    print(f"Examples: {n_total}\n")

    series: Dict[str, List[float]] = {}
    flipped: Dict[str, List[float]] = {}

    for idx in range(n_total):
        tc = tokenizer(ds[idx]["chosen"], return_tensors="pt",
                       truncation=True, max_length=args.max_length).to(device)
        tr = tokenizer(ds[idx]["rejected"], return_tensors="pt",
                       truncation=True, max_length=args.max_length).to(device)

        capture.clear()
        with torch.no_grad():
            model(**tc, use_cache=False)
        c_b = to_fp32(capture.states, device)
        c_mask = tc["attention_mask"]

        capture.clear()
        with torch.no_grad():
            model(**tr, use_cache=False)
        r_b = to_fp32(capture.states, device)
        r_mask = tr["attention_mask"]

        cfgs = build_configurations(c_b, r_b)
        cfgs.update(build_norm_configurations(c_b, r_b, norm_fn))

        if not series:
            for k in cfgs:
                series[k] = []
                flipped[k] = []

        for k, (cs, rs) in cfgs.items():
            # Shape guard: each chosen tensor must match c_mask seq dim, same rejected.
            try:
                for i, c in enumerate(cs):
                    assert c.shape[1] == c_mask.shape[1], \
                        f"config={k} cs[{i}].shape={tuple(c.shape)} c_mask.shape={tuple(c_mask.shape)}"
                for i, r in enumerate(rs):
                    assert r.shape[1] == r_mask.shape[1], \
                        f"config={k} rs[{i}].shape={tuple(r.shape)} r_mask.shape={tuple(r_mask.shape)}"
            except AssertionError as e:
                print(f"SHAPE ERROR ex{idx}: {e}")
                raise
            s = score(evaluator, cs, c_mask, rs, r_mask)
            series[k].append(s)
            # Flip test (rejected, chosen)
            s_flip = score(evaluator, rs, r_mask, cs, c_mask)
            flipped[k].append(s_flip)

        if (idx + 1) % 50 == 0 or idx == n_total - 1:
            n = idx + 1
            line = f"[{n}/{n_total}]"
            for k in ("boundary", "only_loop_2", "mask_zero_only_loop2",
                      "loop2_in_loop1_ctx", "loop2_in_mean_ctx"):
                acc = sum(1 for x in series[k] if x > 0) / n
                line += f"  {k}={acc:.3f}"
            print(line)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    handle.remove()

    # Aggregate.
    n = len(series["boundary"])
    summary: Dict[str, Dict[str, float]] = {}
    for k in series:
        arr = np.array(series[k], dtype=np.float64)
        arr_flip = np.array(flipped[k], dtype=np.float64)
        valid = np.isfinite(arr) & np.isfinite(arr_flip)
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        a, af = arr[valid], arr_flip[valid]
        d = {
            "accuracy": float(np.mean(a > 0)),
            "pos_rate": float(np.mean(a > 0)),
            "flipped_pos_rate": float(np.mean(af > 0)),
            "mean_score": float(np.mean(a)),
            "std_score": float(np.std(a)),
            "min_score": float(np.min(a)),
            "max_score": float(np.max(a)),
            "mean_flipped_score": float(np.mean(af)),
            "std_flipped_score": float(np.std(af)),
            "antisymmetry_pearson": float(np.corrcoef(a, -af)[0, 1]) if np.std(a) > 1e-9 and np.std(af) > 1e-9 else 0.0,
            "antisymmetry_sign_flip_rate": float(np.mean(np.sign(a) != np.sign(af))),
            "antisymmetry_mean_abs_residual": float(np.mean(np.abs(a + af))),
            "n_valid": n_valid,
            "n_nan": int(n - n_valid),
        }
        summary[k] = d

    result = {
        "model_path": model_path,
        "checkpoint": checkpoint_path,
        "dataset": args.dataset,
        "split": args.split,
        "max_length": args.max_length,
        "examples_evaluated": n,
        "per_config": summary,
        "raw_scores": {k: [float(x) for x in v] for k, v in series.items()},
        "raw_scores_flipped": {k: [float(x) for x in v] for k, v in flipped.items()},
    }

    print("\n=== SUMMARY ===")
    print(f"n = {n}")
    print(f"{'config':>32} | {'acc':>5} | {'pos':>5} | {'flip_pos':>8} | {'std':>5} | {'antisym_r':>10} | {'sign_flip_rate':>14}")
    for k in series:
        if k not in summary:
            continue
        d = summary[k]
        print(f"  {k:>30} | {d['accuracy']:.3f} | {d['pos_rate']:.3f} | {d['flipped_pos_rate']:.3f}    | {d['std_score']:.3f} | {d['antisymmetry_pearson']:+.3f}     | {d['antisymmetry_sign_flip_rate']:.3f}")

    print("\n=== DIAGNOSIS ===")
    for k in ("mask_zero_only_loop2", "mask_zero_loop2_front",
              "all_zeros", "identical_loop2"):
        if k not in summary:
            continue
        d = summary[k]
        prediction = "DEGENERATE (constant sign)" if d["pos_rate"] > 0.99 and d["flipped_pos_rate"] > 0.99 else \
                     "ANTISYMMETRIC" if d["antisymmetry_sign_flip_rate"] > 0.9 else \
                     "PARTIAL DEGENERACY"
        print(f"  {k}: {prediction}")
        print(f"    pos_rate={d['pos_rate']:.3f}, flip_pos_rate={d['flipped_pos_rate']:.3f}, "
              f"sign_flip_rate={d['antisymmetry_sign_flip_rate']:.3f}, std={d['std_score']:.3f}")

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
