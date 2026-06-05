"""v5 probe: extend iterated-norm + test per-loop quadnorm.

Two open questions from v4-redo:

  1. Where does iterated norm plateau? loop2_quadnorm hit 97.9 % with std still
     falling. Test pentanorm, hexanorm, heptanorm, octanorm.
  2. Does the loop-2-specialness disappear entirely under quadnorm?
     Test only_loop_{1,3,4}_quadnorm against only_loop_2_quadnorm.

Plus two complementary controls:
  - mean_of_loops_quadnorm: apply 4 norms to mean of loops; replicate.
  - quadnorm_all_loops_natural: apply 4 norms to each loop boundary state
    independently, keep the natural 4-loop sequence (still feeds GRU).

All zero-shot, same frozen pairwise_epoch2.pt head everywhere.

Flip tests on the new norm-extended variants to confirm antisymmetry.
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

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
    p.add_argument("--output-json", default="artifacts/reports/evaluator/probe_evaluator_v5_extended.json")
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


def iter_norm(x: torch.Tensor, norm_fn: Callable[[torch.Tensor], torch.Tensor], n: int) -> torch.Tensor:
    """Apply norm_fn n times to x."""
    for _ in range(n):
        x = norm_fn(x)
    return x


def build_configs(c_b: List[torch.Tensor], r_b: List[torch.Tensor],
                  norm_fn: Callable[[torch.Tensor], torch.Tensor]
                  ) -> Dict[str, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
    cfg: Dict[str, Tuple[List[torch.Tensor], List[torch.Tensor]]] = {}

    # Controls (match v3/v4/v4-redo for cross-pass comparability)
    cfg["boundary"] = (list(c_b), list(r_b))
    cfg["only_loop_2"] = ([c_b[1]] * 4, [r_b[1]] * 4)

    # Iterated norm on loop 2: 2 through 8 applications.
    # n=1 is "only_loop_2"; n=2 was "loop2_doublenorm" in v4-redo; we re-run
    # 2/3/4 here as in-script controls and extend to 5/6/7/8.
    for n in (2, 3, 4, 5, 6, 7, 8):
        c_n = iter_norm(c_b[1], norm_fn, n - 1)  # n applications on top of the implicit one
        r_n = iter_norm(r_b[1], norm_fn, n - 1)
        cfg[f"loop2_norm_x{n}"] = ([c_n] * 4, [r_n] * 4)

    # Per-loop quadnorm: apply 4× extra norm to each loop's boundary state.
    # (Loop 2 quadnorm here should match loop2_norm_x5 above — sanity check.)
    for loop_idx in range(NUM_UT_LOOPS):
        c_q = iter_norm(c_b[loop_idx], norm_fn, 4)
        r_q = iter_norm(r_b[loop_idx], norm_fn, 4)
        cfg[f"only_loop_{loop_idx+1}_quadnorm"] = ([c_q] * 4, [r_q] * 4)

    # Mean-of-loops with quadnorm applied:
    c_mean = torch.stack(c_b, dim=0).mean(dim=0)
    r_mean = torch.stack(r_b, dim=0).mean(dim=0)
    c_mean_q = iter_norm(c_mean, norm_fn, 4)
    r_mean_q = iter_norm(r_mean, norm_fn, 4)
    cfg["mean_loops_quadnorm"] = ([c_mean_q] * 4, [r_mean_q] * 4)

    # Quadnorm-each-loop-then-natural-sequence: gives GRU the same boundary
    # state list but with each entry quadnormed. Tests whether GRU is the
    # limiting factor even with cleaned inputs.
    c_each = [iter_norm(c_b[i], norm_fn, 4) for i in range(NUM_UT_LOOPS)]
    r_each = [iter_norm(r_b[i], norm_fn, 4) for i in range(NUM_UT_LOOPS)]
    cfg["quadnorm_each_natural_seq"] = (c_each, r_each)

    return cfg


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_MODULES_CACHE", str(Path.cwd() / ".hf_modules_cache"))

    device = torch.device(args.device)
    model_path = str(Path(args.model_path))
    checkpoint_path = str(Path(args.checkpoint))

    print(f"Model: {model_path}")
    print(f"Evaluator: {checkpoint_path}")
    print(f"Examples: {args.max_examples}, max_length={args.max_length}\n")

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

        cfgs = build_configs(c_b, r_b, norm_fn)

        if not series:
            for k in cfgs:
                series[k] = []
                flipped[k] = []

        for k, (cs, rs) in cfgs.items():
            s = score(evaluator, cs, c_mask, rs, r_mask)
            series[k].append(s)
            s_flip = score(evaluator, rs, r_mask, cs, c_mask)
            flipped[k].append(s_flip)

        if (idx + 1) % 50 == 0 or idx == n_total - 1:
            n_so_far = idx + 1
            line = f"[{n_so_far}/{n_total}]"
            for k in ("boundary", "only_loop_2", "loop2_norm_x4", "loop2_norm_x6",
                      "loop2_norm_x8", "only_loop_1_quadnorm", "only_loop_3_quadnorm"):
                acc = sum(1 for x in series[k] if x > 0) / n_so_far
                line += f"  {k}={acc:.3f}"
            print(line)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    handle.remove()

    n = len(series["boundary"])
    summary = {}
    for k in series:
        a = np.array(series[k], dtype=np.float64)
        af = np.array(flipped[k], dtype=np.float64)
        valid = np.isfinite(a) & np.isfinite(af)
        if not valid.any():
            continue
        av, afv = a[valid], af[valid]
        summary[k] = {
            "accuracy": float(np.mean(av > 0)),
            "pos_rate": float(np.mean(av > 0)),
            "flipped_pos_rate": float(np.mean(afv > 0)),
            "mean_score": float(np.mean(av)),
            "std_score": float(np.std(av)),
            "min_score": float(np.min(av)),
            "max_score": float(np.max(av)),
            "antisymmetry_pearson": float(np.corrcoef(av, -afv)[0, 1])
                if np.std(av) > 1e-9 and np.std(afv) > 1e-9 else 0.0,
            "antisymmetry_sign_flip_rate": float(np.mean(np.sign(av) != np.sign(afv))),
            "n_valid": int(valid.sum()),
        }

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
    print(f"{'config':>30} | {'acc':>5} | {'std':>5} | {'sign_flip':>9} | {'antisym_r':>9}")
    # Print iterated-norm sweep first
    for n_app in (1, 2, 3, 4, 5, 6, 7, 8):
        k = "only_loop_2" if n_app == 1 else f"loop2_norm_x{n_app}"
        if k in summary:
            d = summary[k]
            print(f"  {k:>28} | {d['accuracy']:.3f} | {d['std_score']:.3f} | {d['antisymmetry_sign_flip_rate']:.3f}     | {d['antisymmetry_pearson']:+.3f}")
    # Per-loop quadnorm
    print(f"  {'---per-loop quadnorm---':>28} |")
    for loop_idx in range(NUM_UT_LOOPS):
        k = f"only_loop_{loop_idx+1}_quadnorm"
        if k in summary:
            d = summary[k]
            print(f"  {k:>28} | {d['accuracy']:.3f} | {d['std_score']:.3f} | {d['antisymmetry_sign_flip_rate']:.3f}     | {d['antisymmetry_pearson']:+.3f}")
    # Misc
    print(f"  {'---misc---':>28} |")
    for k in ("boundary", "mean_loops_quadnorm", "quadnorm_each_natural_seq"):
        if k in summary:
            d = summary[k]
            print(f"  {k:>28} | {d['accuracy']:.3f} | {d['std_score']:.3f} | {d['antisymmetry_sign_flip_rate']:.3f}     | {d['antisymmetry_pearson']:+.3f}")

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
