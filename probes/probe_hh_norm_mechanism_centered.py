"""HH-RLHF centered probe for loop x8, norm mechanism, and K sweep.

This is the shared expensive run for tests 2-4:

  2. Test x8 across all four loop boundary states.
  3. Ablate learned OuroRMSNorm vs pure RMS vs gamma-only power filtering.
  4. Sweep norm count K and rank by centered/debiased pairwise metrics.

Terminology:
  - extra_k=0 means the captured loop boundary state as produced by Ouro.
  - total_x8 means extra_k=7, matching prior loop2_norm_x8 naming.

The decisive metric is centered_acc:
  mean(score(chosen, rejected) - score(rejected, chosen) > 0)

Canonical chosen-first accuracy is still reported, but it is unsafe as a
branch-control metric when the scorer has a symmetric positive offset.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

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
    p.add_argument("--report-every", type=int, default=25)
    p.add_argument("--output-json", default="artifacts/reports/evaluator/probe_hh_norm_mechanism_centered.json")
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


def rms_only(x: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.to(dtype=torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps)


def iter_transform(
    x: torch.Tensor,
    extra_k: int,
    transform: str,
    learned_norm: Callable[[torch.Tensor], torch.Tensor],
    gamma: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    y = x
    for _ in range(extra_k):
        if transform == "learned":
            y = learned_norm(y)
        elif transform == "pure_rms":
            y = rms_only(y, eps)
        elif transform == "gamma_only":
            y = gamma.view(1, 1, -1).to(device=y.device, dtype=torch.float32) * y.to(dtype=torch.float32)
            y = rms_only(y, eps)
        else:
            raise ValueError(f"unknown transform: {transform}")
    return y


def replicated(x: torch.Tensor) -> List[torch.Tensor]:
    return [x] * NUM_UT_LOOPS


def make_config(
    c_b: List[torch.Tensor],
    r_b: List[torch.Tensor],
    name: str,
    transform: str,
    extra_k: int,
    learned_norm: Callable[[torch.Tensor], torch.Tensor],
    gamma: torch.Tensor,
    eps: float,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    if name.startswith("loop_"):
        loop_idx = int(name.split("_")[1]) - 1
        c = iter_transform(c_b[loop_idx], extra_k, transform, learned_norm, gamma, eps)
        r = iter_transform(r_b[loop_idx], extra_k, transform, learned_norm, gamma, eps)
        return replicated(c), replicated(r)

    if name == "mean_then":
        c_mean = torch.stack(c_b, dim=0).mean(dim=0)
        r_mean = torch.stack(r_b, dim=0).mean(dim=0)
        c = iter_transform(c_mean, extra_k, transform, learned_norm, gamma, eps)
        r = iter_transform(r_mean, extra_k, transform, learned_norm, gamma, eps)
        return replicated(c), replicated(r)

    if name == "each_then_mean":
        c_each = [iter_transform(c_b[i], extra_k, transform, learned_norm, gamma, eps) for i in range(NUM_UT_LOOPS)]
        r_each = [iter_transform(r_b[i], extra_k, transform, learned_norm, gamma, eps) for i in range(NUM_UT_LOOPS)]
        c = torch.stack(c_each, dim=0).mean(dim=0)
        r = torch.stack(r_each, dim=0).mean(dim=0)
        return replicated(c), replicated(r)

    if name == "each_natural_seq_GRU":
        c_each = [iter_transform(c_b[i], extra_k, transform, learned_norm, gamma, eps) for i in range(NUM_UT_LOOPS)]
        r_each = [iter_transform(r_b[i], extra_k, transform, learned_norm, gamma, eps) for i in range(NUM_UT_LOOPS)]
        return c_each, r_each

    raise ValueError(f"unknown config family: {name}")


def metric_block(normal: Iterable[float], flipped: Iterable[float]) -> Dict[str, float | int]:
    s = np.array(list(normal), dtype=np.float64)
    sf = np.array(list(flipped), dtype=np.float64)
    valid = np.isfinite(s) & np.isfinite(sf)
    s = s[valid]
    sf = sf[valid]
    if s.size == 0:
        return {"n_valid": 0}

    centered = s - sf
    bias = 0.5 * (s + sf)
    antisym = 0.5 * centered
    strict = np.sign(s) != np.sign(sf)
    corr = 0.0
    if np.std(s) > 1e-9 and np.std(sf) > 1e-9:
        corr = float(np.corrcoef(s, -sf)[0, 1])

    return {
        "n_valid": int(s.size),
        "canonical_acc": float(np.mean(s > 0)),
        "flipped_acc": float(np.mean(sf < 0)),
        "normal_pos_rate": float(np.mean(s > 0)),
        "flipped_pos_rate": float(np.mean(sf > 0)),
        "strict_sign_reversal_rate": float(np.mean(strict)),
        "same_sign_rate": float(1.0 - np.mean(strict)),
        "antisym_corr": corr,
        "centered_acc": float(np.mean(centered > 0)),
        "centered_mean": float(np.mean(centered)),
        "centered_std": float(np.std(centered)),
        "bias_mean": float(np.mean(bias)),
        "bias_std": float(np.std(bias)),
        "antisym_mean": float(np.mean(antisym)),
        "antisym_std": float(np.std(antisym)),
        "bias_to_signal": float(abs(np.mean(bias)) / (abs(np.mean(antisym)) + 1e-8)),
        "mean_sum": float(np.mean(s + sf)),
        "mean_abs_sum": float(np.mean(np.abs(s + sf))),
        "score_mean": float(np.mean(s)),
        "score_std": float(np.std(s)),
        "flipped_mean": float(np.mean(sf)),
        "flipped_std": float(np.std(sf)),
        "score_min": float(np.min(s)),
        "score_max": float(np.max(s)),
    }


def build_plan() -> List[Dict[str, object]]:
    plan: List[Dict[str, object]] = []

    # Test 4: learned K sweep for every loop plus mean/natural aggregations.
    for extra_k in range(0, 9):
        for loop_idx in range(1, NUM_UT_LOOPS + 1):
            plan.append(
                {
                    "config": f"learned_loop_{loop_idx}_extra_k{extra_k}",
                    "family": f"loop_{loop_idx}",
                    "transform": "learned",
                    "extra_k": extra_k,
                    "test": "k_sweep",
                }
            )
        for family, label in (
            ("mean_then", "learned_mean_then"),
            ("each_then_mean", "learned_each_then_mean"),
            ("each_natural_seq_GRU", "learned_each_natural_seq_GRU"),
        ):
            plan.append(
                {
                    "config": f"{label}_extra_k{extra_k}",
                    "family": family,
                    "transform": "learned",
                    "extra_k": extra_k,
                    "test": "k_sweep",
                }
            )

    # Test 3: loop 2 mechanism ablation.
    for transform in ("pure_rms", "gamma_only"):
        for extra_k in range(0, 9):
            plan.append(
                {
                    "config": f"{transform}_loop_2_extra_k{extra_k}",
                    "family": "loop_2",
                    "transform": transform,
                    "extra_k": extra_k,
                    "test": "mechanism",
                }
            )

    return plan


def alias_config_name(config: str) -> str:
    if config.startswith("learned_loop_") and config.endswith("_extra_k7"):
        loop = config.split("_")[2]
        return f"learned_loop_{loop}_total_x8"
    if config == "learned_mean_then_extra_k7":
        return "learned_mean_then_total_x8"
    if config == "learned_each_then_mean_extra_k7":
        return "learned_each_then_mean_total_x8"
    if config == "learned_each_natural_seq_GRU_extra_k7":
        return "learned_each_natural_seq_GRU_total_x8"
    return ""


def print_rows(title: str, keys: List[str], summary: Dict[str, Dict[str, float | int]]) -> None:
    print(f"\n=== {title} ===")
    print(
        f"{'config':>42} | {'canon':>5} | {'center':>6} | {'flip':>5} | "
        f"{'strict':>6} | {'corr':>6} | {'bias/sig':>8} | {'std':>5}"
    )
    for key in keys:
        if key not in summary:
            continue
        d = summary[key]
        print(
            f"  {key:>40} | {d['canonical_acc']:.3f} | {d['centered_acc']:.3f} | "
            f"{d['flipped_acc']:.3f} | {d['strict_sign_reversal_rate']:.3f} | "
            f"{d['antisym_corr']:+.3f} | {d['bias_to_signal']:.2f} | {d['score_std']:.3f}"
        )


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_MODULES_CACHE", str(Path.cwd() / ".hf_modules_cache"))

    device = torch.device(args.device)
    print(f"Model: {args.model_path}")
    print(f"Evaluator: {args.checkpoint}")
    print(f"Dataset: {args.dataset} ({args.split})")
    print(f"Examples: {args.max_examples}, max_length={args.max_length}, device={device}\n")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
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

    eps = float(getattr(model.model.norm, "variance_epsilon", 1e-6))
    gamma = model.model.norm.weight.detach().to(device=device, dtype=torch.float32)

    @torch.no_grad()
    def learned_norm(x: torch.Tensor) -> torch.Tensor:
        return model.model.norm(x.to(dtype=torch.bfloat16)).to(dtype=torch.float32)

    print("Loading evaluator...")
    evaluator = PairwiseEvaluator().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
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
    print(f"Examples loaded: {n_total}\n")

    plan = build_plan()
    series: Dict[str, List[float]] = {str(item["config"]): [] for item in plan}
    flipped: Dict[str, List[float]] = {str(item["config"]): [] for item in plan}

    for idx in range(n_total):
        tc = tokenizer(
            ds[idx]["chosen"],
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        ).to(device)
        tr = tokenizer(
            ds[idx]["rejected"],
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        ).to(device)

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

        for item in plan:
            config = str(item["config"])
            cs, rs = make_config(
                c_b,
                r_b,
                str(item["family"]),
                str(item["transform"]),
                int(item["extra_k"]),
                learned_norm,
                gamma,
                eps,
            )
            series[config].append(score(evaluator, cs, c_mask, rs, r_mask))
            flipped[config].append(score(evaluator, rs, r_mask, cs, c_mask))

        if (idx + 1) % args.report_every == 0 or idx == n_total - 1:
            n_so_far = idx + 1
            preview = []
            for key in (
                "learned_loop_2_extra_k0",
                "learned_loop_2_extra_k3",
                "learned_loop_2_extra_k7",
                "pure_rms_loop_2_extra_k7",
                "gamma_only_loop_2_extra_k7",
            ):
                acc = sum(1 for x in series[key] if x > 0) / n_so_far
                preview.append(f"{key}={acc:.3f}")
            print(f"[{n_so_far}/{n_total}] " + "  ".join(preview))
            if device.type == "cuda":
                torch.cuda.empty_cache()

    handle.remove()

    summary = {k: metric_block(series[k], flipped[k]) for k in series}
    aliases = {}
    for config in list(summary):
        alias = alias_config_name(config)
        if alias:
            aliases[alias] = summary[config]

    result = {
        "model_path": args.model_path,
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "split": args.split,
        "max_length": args.max_length,
        "examples_evaluated": n_total,
        "norm_eps": eps,
        "metrics_definition": {
            "canonical_acc": "mean(score(chosen, rejected) > 0)",
            "flipped_acc": "mean(score(rejected, chosen) < 0)",
            "centered_acc": "mean(score(chosen, rejected) - score(rejected, chosen) > 0)",
            "strict_sign_reversal_rate": "mean(sign(score_normal) != sign(score_flipped))",
            "bias_mean": "mean(0.5 * (score_normal + score_flipped))",
            "antisym_mean": "mean(0.5 * (score_normal - score_flipped))",
        },
        "per_config": summary,
        "aliases": aliases,
        "raw_scores": {k: [float(x) for x in v] for k, v in series.items()},
        "raw_scores_flipped": {k: [float(x) for x in v] for k, v in flipped.items()},
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    x8_keys = [
        "learned_loop_1_extra_k7",
        "learned_loop_2_extra_k7",
        "learned_loop_3_extra_k7",
        "learned_loop_4_extra_k7",
        "learned_mean_then_extra_k7",
        "learned_each_then_mean_extra_k7",
        "learned_each_natural_seq_GRU_extra_k7",
    ]
    print_rows("TEST 2: learned total_x8 across loops/aggregations", x8_keys, summary)

    mech_keys = []
    for extra_k in range(0, 9):
        mech_keys.extend(
            [
                f"learned_loop_2_extra_k{extra_k}",
                f"pure_rms_loop_2_extra_k{extra_k}",
                f"gamma_only_loop_2_extra_k{extra_k}",
            ]
        )
    print_rows("TEST 3: loop 2 mechanism ablation", mech_keys, summary)

    learned_rows = [(k, d) for k, d in summary.items() if k.startswith("learned_")]
    best_centered = sorted(
        learned_rows,
        key=lambda kv: (-float(kv[1]["centered_acc"]), -float(kv[1]["flipped_acc"])),
    )[:15]
    print_rows("TEST 4: top learned K sweep by centered_acc", [k for k, _ in best_centered], summary)

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
