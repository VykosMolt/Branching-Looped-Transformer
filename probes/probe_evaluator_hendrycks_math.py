"""Hendrycks MATH probe — harder math, middle-step perturbation, rigorous flip metrics.

GSM8K-with-final-answer-swap saturated the evaluator at 100% (ceiling effect).
This probe constructs (chosen, rejected) pairs from Hendrycks MATH (Level 4-5
filter for difficulty) by perturbing a numerical value in the MIDDLE of the
solution -- not the final answer. The resulting "rejected" solution looks
plausible at the surface level but has a wrong intermediate step, often
propagating to a wrong final boxed answer. Discrimination is harder.

Configurations cover the v5 question gaps:
  - boundary + per-loop ablation (canonical 1× norm)
  - per-loop iterated norm at x8 (ALL FOUR LOOPS, not just loop 2)
  - iterated-norm sweep on loop 2 (x2..x8) for curve mapping
  - mean variants: mean_replicated, mean_norm_x8, norm_each_then_mean

Metrics (renamed per GPT review for clarity):
  - normal_pos_rate         fraction where score > 0
  - flipped_pos_rate        fraction where flipped_score > 0
  - strict_sign_reversal_rate    fraction where sign(score) != sign(flipped_score)
                                 (= 1.0 for perfectly antisymmetric architecture)
  - same_sign_rate          = 1 - strict_sign_reversal_rate
                            (high value = bias offset dominates over signal)
  - antisym_corr            corr(score_normal, -score_flipped)
                            (= +1.0 for perfectly antisymmetric)
  - mean_sum                mean(score_normal + score_flipped)
                            (= 0 for perfectly antisymmetric; nonzero = bias offset)
  - score_std, score_range  distributional sanity

If accuracy stays at ~100% with these configurations on Hendrycks MATH,
the evaluator generalizes perfectly to math difficulty too. If it drops,
we see headroom on math and can compare the iterated-norm gain pattern.

Seed: 42 (per user request, applies to perturbation choice).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluator_core.pairwise_evaluator import PairwiseEvaluator, validate_hook_output


DEFAULT_MODEL_PATH = "models/ouro_rltt_local"
DEFAULT_CHECKPOINT_PATH = "artifacts/checkpoints/evaluator/pairwise_epoch2.pt"
NUM_UT_LOOPS = 4

# Match any integer or decimal (positive or negative).
NUMBER_RE = re.compile(r"(?<![\\\d.])(-?\d+(?:\.\d+)?)(?![\d.])")
# Last \boxed{...} content — Hendrycks MATH's final-answer marker.
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument("--dataset", default="lighteval/MATH")
    p.add_argument("--dataset-config", default="all")
    p.add_argument("--split", default="test")
    p.add_argument("--max-examples", type=int, default=1000)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--min-level", type=int, default=4,
                   help="Hendrycks MATH difficulty level filter (1=easy, 5=hard).")
    p.add_argument("--early-exit-threshold", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-json", default="artifacts/reports/evaluator/probe_evaluator_hendrycks_math.json")
    p.add_argument("--seed", type=int, default=42)
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


def to_fp32(states, device):
    return [h.to(device=device, dtype=torch.float32) for h in states]


def score(evaluator, cs, c_mask, rs, r_mask) -> float:
    with torch.no_grad():
        s = evaluator(cs, c_mask, rs, r_mask)
    val = float(s.view(-1).item())
    return val if math.isfinite(val) else float("nan")


def iter_norm(x, norm_fn, n):
    for _ in range(n):
        x = norm_fn(x)
    return x


def find_middle_numbers(solution: str) -> List[Tuple[int, int, str]]:
    """Find numeric tokens in the solution that are NOT inside the final \\boxed{...}.
    Returns list of (start, end, value_str)."""
    # Locate boxed regions to exclude.
    boxed_spans = [m.span() for m in BOXED_RE.finditer(solution)]

    def in_boxed(s: int, e: int) -> bool:
        for bs, be in boxed_spans:
            if s >= bs and e <= be:
                return True
        return False

    matches = []
    for m in NUMBER_RE.finditer(solution):
        s, e = m.span(1)
        if in_boxed(s, e):
            continue
        # Skip trivially small numbers like single digits 0-2 which often appear in indices.
        v_str = m.group(1)
        try:
            v = float(v_str)
        except ValueError:
            continue
        if abs(v) < 1.0 and v_str not in ("-1",):
            continue
        matches.append((s, e, v_str))
    return matches


def perturb_middle_number(solution: str, rng: random.Random) -> Optional[str]:
    """Pick a random middle number and perturb it. Return None if no candidates."""
    matches = find_middle_numbers(solution)
    if not matches:
        return None
    start, end, v_str = rng.choice(matches)
    try:
        if "." in v_str:
            v = float(v_str)
            mult = rng.choice([2.0, 0.5, 3.0, -1.0, 1.5, 0.333])
            new_v = v * mult + rng.choice([-1.0, 0.0, 1.0])
            new_s = f"{new_v:.3f}".rstrip("0").rstrip(".") or "0"
        else:
            v = int(v_str)
            sign = 1 if v >= 0 else -1
            mag = max(abs(v), 1)
            shift = rng.choice([-1, 1, -2, 2, -3, 3, -mag, mag])
            new_v = v + shift
            if new_v == v:
                new_v = v + 1
            new_s = str(new_v)
    except (ValueError, OverflowError):
        return None
    if new_s == v_str:
        return None
    return solution[:start] + new_s + solution[end:]


def build_pair(problem: str, solution: str, rng: random.Random) -> Optional[Tuple[str, str]]:
    """Build (chosen, rejected) by perturbing a middle number. Returns None on failure."""
    rejected_sol = perturb_middle_number(solution, rng)
    if rejected_sol is None or rejected_sol == solution:
        return None
    chosen = f"Problem: {problem}\n\nSolution: {solution}"
    rejected = f"Problem: {problem}\n\nSolution: {rejected_sol}"
    return chosen, rejected


def build_configs(c_b, r_b, norm_fn):
    """Configs: boundary, per-loop x1, per-loop x8 (ALL loops), x-sweep on loop 2, mean variants."""
    cfg = {}

    # Canonical and per-loop 1× norm (= 1 norm application baked into _run_single_ut_loop)
    cfg["boundary"] = (list(c_b), list(r_b))
    for n_loop in range(NUM_UT_LOOPS):
        cfg[f"only_loop_{n_loop+1}"] = ([c_b[n_loop]] * 4, [r_b[n_loop]] * 4)

    # Iterated norm sweep on loop 2: extra-norm counts 1..7 (loop2 already has 1 implicit).
    for n_extra in (1, 2, 3, 4, 5, 6, 7):
        c_n = iter_norm(c_b[1], norm_fn, n_extra)
        r_n = iter_norm(r_b[1], norm_fn, n_extra)
        cfg[f"loop2_norm_x{n_extra+1}"] = ([c_n] * 4, [r_n] * 4)

    # All-loops x8 (= 7 extra norms on top of the implicit 1)
    for n_loop in range(NUM_UT_LOOPS):
        c_n = iter_norm(c_b[n_loop], norm_fn, 7)
        r_n = iter_norm(r_b[n_loop], norm_fn, 7)
        cfg[f"only_loop_{n_loop+1}_norm_x8"] = ([c_n] * 4, [r_n] * 4)

    # Mean variants
    c_mean = torch.stack(c_b, dim=0).mean(dim=0)
    r_mean = torch.stack(r_b, dim=0).mean(dim=0)
    cfg["mean_replicated"] = ([c_mean] * 4, [r_mean] * 4)

    c_mean_n = iter_norm(c_mean, norm_fn, 7)
    r_mean_n = iter_norm(r_mean, norm_fn, 7)
    cfg["mean_then_norm_x8_replicated"] = ([c_mean_n] * 4, [r_mean_n] * 4)

    c_each_n = [iter_norm(c_b[i], norm_fn, 7) for i in range(NUM_UT_LOOPS)]
    r_each_n = [iter_norm(r_b[i], norm_fn, 7) for i in range(NUM_UT_LOOPS)]
    c_each_mean = torch.stack(c_each_n, dim=0).mean(dim=0)
    r_each_mean = torch.stack(r_each_n, dim=0).mean(dim=0)
    cfg["norm_x8_each_then_mean_replicated"] = ([c_each_mean] * 4, [r_each_mean] * 4)
    cfg["norm_x8_each_natural_seq_GRU"] = (c_each_n, r_each_n)

    return cfg


def pearson(a, b):
    if a.size == 0 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_MODULES_CACHE", str(Path.cwd() / ".hf_modules_cache"))

    device = torch.device(args.device)
    print(f"Model: {args.model_path}")
    print(f"Evaluator: {args.checkpoint}")
    print(f"Dataset: {args.dataset} ({args.dataset_config}, {args.split})")
    print(f"Examples target: {args.max_examples}, max_length={args.max_length}")
    print(f"Filter: Level >= {args.min_level}, perturbation seed={args.seed}\n")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map={"": str(device)}, torch_dtype=torch.bfloat16,
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "config"):
        model.config.early_exit_threshold = args.early_exit_threshold

    @torch.no_grad()
    def norm_fn(x):
        x_bf = x.to(dtype=torch.bfloat16)
        return model.model.norm(x_bf).to(dtype=torch.float32)

    print("Loading evaluator...")
    evaluator = PairwiseEvaluator().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    evaluator.load_state_dict(state_dict)
    evaluator.eval()

    capture = BoundaryCapture()
    handle = model.model.register_forward_hook(capture.hook)

    print(f"Loading {args.dataset}...")
    try:
        ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
    except Exception as e:
        print(f"load_dataset failed: {e}")
        print("Trying alternate dataset name 'hendrycks/competition_math'...")
        try:
            ds = load_dataset("hendrycks/competition_math", split=args.split)
        except Exception as e2:
            print(f"alternate load_dataset failed: {e2}")
            print("Trying EleutherAI/hendrycks_math subject configs...")
            hendrycks_configs = [
                "algebra",
                "counting_and_probability",
                "geometry",
                "intermediate_algebra",
                "number_theory",
                "prealgebra",
                "precalculus",
            ]
            parts = []
            for cfg in hendrycks_configs:
                part = load_dataset("EleutherAI/hendrycks_math", cfg, split=args.split)
                parts.append(part)
                print(f"Loaded EleutherAI/hendrycks_math/{cfg}")
            ds = concatenate_datasets(parts)

    # Filter to higher-difficulty levels. Level field may be string like "Level 5" or int.
    def get_level(ex) -> int:
        lvl = ex.get("level", ex.get("Level", 5))
        if isinstance(lvl, str):
            m = re.search(r"\d", lvl)
            return int(m.group()) if m else 5
        return int(lvl)

    print(f"Total dataset examples: {len(ds)}")
    ds_filtered_idx = [i for i in range(len(ds)) if get_level(ds[i]) >= args.min_level]
    print(f"Level >= {args.min_level}: {len(ds_filtered_idx)}")

    rng.shuffle(ds_filtered_idx)

    series: Dict[str, List[float]] = {}
    flipped: Dict[str, List[float]] = {}
    used = 0
    skipped = 0

    for ds_idx in ds_filtered_idx:
        if used >= args.max_examples:
            break
        ex = ds[ds_idx]
        problem = ex.get("problem", ex.get("Problem", ""))
        solution = ex.get("solution", ex.get("Solution", ""))
        if not problem or not solution:
            skipped += 1
            continue
        pair = build_pair(problem, solution, rng)
        if pair is None:
            skipped += 1
            continue
        chosen, rejected = pair

        tc = tokenizer(chosen, return_tensors="pt", truncation=True,
                       max_length=args.max_length).to(device)
        tr = tokenizer(rejected, return_tensors="pt", truncation=True,
                       max_length=args.max_length).to(device)

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
        used += 1

        if used % 50 == 0 or used == args.max_examples:
            line = f"[{used}/{args.max_examples}]"
            for k in ("boundary", "only_loop_2", "loop2_norm_x4", "loop2_norm_x8",
                      "only_loop_1_norm_x8", "only_loop_3_norm_x8", "only_loop_4_norm_x8",
                      "mean_then_norm_x8_replicated"):
                if series[k]:
                    acc = sum(1 for x in series[k] if x > 0) / len(series[k])
                    line += f"  {k}={acc:.3f}"
            print(line)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    handle.remove()

    n = used
    summary = {}
    for k in series:
        a = np.array(series[k], dtype=np.float64)
        af = np.array(flipped[k], dtype=np.float64)
        valid = np.isfinite(a) & np.isfinite(af)
        if not valid.any():
            continue
        av, afv = a[valid], af[valid]
        strict_sign_reversal = float(np.mean(np.sign(av) != np.sign(afv)))
        summary[k] = {
            "accuracy": float(np.mean(av > 0)),
            "normal_pos_rate": float(np.mean(av > 0)),
            "flipped_pos_rate": float(np.mean(afv > 0)),
            "strict_sign_reversal_rate": strict_sign_reversal,
            "same_sign_rate": 1.0 - strict_sign_reversal,
            "antisym_corr": pearson(av, -afv),
            "mean_sum": float(np.mean(av + afv)),
            "mean_score": float(np.mean(av)),
            "score_std": float(np.std(av)),
            "score_min": float(np.min(av)),
            "score_max": float(np.max(av)),
            "score_range": float(np.max(av) - np.min(av)),
            "n_valid": int(valid.sum()),
        }

    result = {
        "model_path": args.model_path,
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "min_level": args.min_level,
        "max_length": args.max_length,
        "perturbation_seed": args.seed,
        "examples_evaluated": n,
        "examples_skipped": skipped,
        "metrics_definition": {
            "accuracy": "fraction where score(chosen, rejected) > 0",
            "strict_sign_reversal_rate": "fraction where sign(score) != sign(flipped_score); 1.0 = perfectly antisymmetric",
            "same_sign_rate": "1 - strict_sign_reversal_rate; high = bias offset dominates over relational signal",
            "antisym_corr": "Pearson(score_normal, -score_flipped); +1.0 = perfectly antisymmetric",
            "mean_sum": "mean(score_normal + score_flipped); 0 = perfectly antisymmetric, nonzero = bias offset",
        },
        "per_config": summary,
        "raw_scores": {k: [float(x) for x in v] for k, v in series.items()},
        "raw_scores_flipped": {k: [float(x) for x in v] for k, v in flipped.items()},
    }

    print("\n=== HENDRYCKS MATH SUMMARY (Level >= {}, n = {}) ===".format(args.min_level, n))
    print(f"{'config':>35} | {'acc':>5} | {'std':>5} | {'sign_rev':>8} | {'antisym':>8} | {'mean_sum':>9} | {'pos':>5} | {'flip_pos':>8}")
    ordering = [
        "boundary",
        "only_loop_1", "only_loop_2", "only_loop_3", "only_loop_4",
        "loop2_norm_x2", "loop2_norm_x3", "loop2_norm_x4",
        "loop2_norm_x5", "loop2_norm_x6", "loop2_norm_x7", "loop2_norm_x8",
        "only_loop_1_norm_x8", "only_loop_2_norm_x8",
        "only_loop_3_norm_x8", "only_loop_4_norm_x8",
        "mean_replicated", "mean_then_norm_x8_replicated",
        "norm_x8_each_then_mean_replicated", "norm_x8_each_natural_seq_GRU",
    ]
    for k in ordering:
        if k not in summary:
            continue
        d = summary[k]
        print(f"  {k:>33} | {d['accuracy']:.3f} | {d['score_std']:.3f} | "
              f"{d['strict_sign_reversal_rate']:.3f}    | "
              f"{d['antisym_corr']:+.3f}    | {d['mean_sum']:+.4f} | "
              f"{d['normal_pos_rate']:.3f} | {d['flipped_pos_rate']:.3f}")

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
