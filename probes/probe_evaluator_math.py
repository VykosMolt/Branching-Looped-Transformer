"""Math-distribution probe: same evaluator configurations as v4-redo, but
on GSM8K with synthetic (correct, answer-perturbed) pairs.

Construction: each example is (question, correct_solution). The "chosen" text
is the correct solution; the "rejected" text is the correct solution with
the final numerical answer (after the "#### N" marker in GSM8K format)
replaced with a wrong number. Same length, same vocabulary, same reasoning
structure -- only the final answer differs. If the evaluator picks chosen
over rejected, it is reading "correct answer" signal, not stylistic preference.

The user reported that loop 2 also wins on math/ARC informally. This probe
formally tests whether:
  1. The boundary evaluator's accuracy holds on math distribution.
  2. Loop-2-only and quadnorm configurations win on math the way they win
     on HH-RLHF chat.

If both hold, the locus / norm-iteration findings generalize across task
types (chat AND reasoning). If neither, they are HH-RLHF-specific.
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
GSM8K_ANSWER_RE = re.compile(r"#### (-?\d+(?:\.\d+)?)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument("--dataset", default="openai/gsm8k")
    p.add_argument("--dataset-config", default="main")
    p.add_argument("--split", default="test")
    p.add_argument("--max-examples", type=int, default=1000)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--early-exit-threshold", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-json", default="artifacts/reports/evaluator/probe_evaluator_math.json")
    p.add_argument("--seed", type=int, default=0)
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


def perturb_gsm8k_answer(answer_text: str, rng: random.Random) -> str:
    """Replace the final "#### N" with "#### M" where M != N, same magnitude scale."""
    m = GSM8K_ANSWER_RE.search(answer_text)
    if not m:
        return answer_text  # no marker, return unchanged
    correct = m.group(1)
    try:
        if "." in correct:
            v = float(correct)
            # Perturb by random factor 2-5x in either direction
            mult = rng.choice([2.0, 0.5, 3.0, 0.333, 5.0, 0.2, -1.0])
            wrong_v = v * mult + rng.choice([-1.0, 0.0, 1.0])
            wrong = f"{wrong_v:.2f}".rstrip("0").rstrip(".")
        else:
            v = int(correct)
            sign = 1 if v >= 0 else -1
            mag = max(abs(v), 1)
            # Random perturbation within a meaningful range
            shift = rng.choice([-1, 1, -2, 2, -5, 5, -mag, mag])
            wrong_v = v + shift
            if wrong_v == v:
                wrong_v = v + 1
            wrong = str(wrong_v)
    except (ValueError, OverflowError):
        return answer_text
    if wrong == correct:
        # Fallback: append "+1"
        wrong = correct + "1"
    return answer_text[: m.start(1)] + wrong + answer_text[m.end(1):]


def build_pair(question: str, answer: str, rng: random.Random) -> Tuple[str, str]:
    """Build (chosen, rejected) text pair from a GSM8K (question, answer) example."""
    chosen = f"Question: {question}\n\nAnswer: {answer}"
    rejected_answer = perturb_gsm8k_answer(answer, rng)
    rejected = f"Question: {question}\n\nAnswer: {rejected_answer}"
    return chosen, rejected


def build_configs(c_b, r_b, norm_fn):
    """Subset of v4-redo + v5 configurations focused on key headline tests."""
    cfg = {}
    cfg["boundary"] = (list(c_b), list(r_b))
    cfg["only_loop_1"] = ([c_b[0]] * 4, [r_b[0]] * 4)
    cfg["only_loop_2"] = ([c_b[1]] * 4, [r_b[1]] * 4)
    cfg["only_loop_3"] = ([c_b[2]] * 4, [r_b[2]] * 4)
    cfg["only_loop_4"] = ([c_b[3]] * 4, [r_b[3]] * 4)

    # Iterated norm on loop 2 (the headline sweep from v5)
    for n_app in (2, 3, 4, 5):
        c_n = iter_norm(c_b[1], norm_fn, n_app - 1)
        r_n = iter_norm(r_b[1], norm_fn, n_app - 1)
        cfg[f"loop2_norm_x{n_app}"] = ([c_n] * 4, [r_n] * 4)

    # Per-loop quadnorm
    for loop_idx in range(NUM_UT_LOOPS):
        c_q = iter_norm(c_b[loop_idx], norm_fn, 4)
        r_q = iter_norm(r_b[loop_idx], norm_fn, 4)
        cfg[f"only_loop_{loop_idx+1}_quadnorm"] = ([c_q] * 4, [r_q] * 4)

    # Mean
    c_mean = torch.stack(c_b, dim=0).mean(dim=0)
    r_mean = torch.stack(r_b, dim=0).mean(dim=0)
    cfg["mean_replicated"] = ([c_mean] * 4, [r_mean] * 4)

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
    print(f"Dataset: {args.dataset} (config={args.dataset_config}, split={args.split})")
    print(f"Examples: {args.max_examples}, max_length={args.max_length}\n")

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
    ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
    if args.max_examples > 0:
        ds = ds.select(range(min(args.max_examples, len(ds))))
    n_total = len(ds)
    print(f"Examples available: {n_total}\n")

    series: Dict[str, List[float]] = {}
    flipped: Dict[str, List[float]] = {}
    skipped = 0

    for idx in range(n_total):
        ex = ds[idx]
        question = ex["question"]
        answer = ex["answer"]
        chosen, rejected = build_pair(question, answer, rng)
        if chosen == rejected:
            skipped += 1
            continue

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

        if (idx + 1) % 50 == 0 or idx == n_total - 1:
            n_so_far = idx + 1 - skipped
            line = f"[{n_so_far}/{n_total}]"
            for k in ("boundary", "only_loop_2", "loop2_norm_x4", "loop2_norm_x5",
                      "only_loop_1_quadnorm", "only_loop_3_quadnorm", "only_loop_4_quadnorm"):
                if series[k]:
                    acc = sum(1 for x in series[k] if x > 0) / len(series[k])
                    line += f"  {k}={acc:.3f}"
            print(line)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    handle.remove()

    n = len(series["boundary"]) if "boundary" in series else 0
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
            "antisymmetry_pearson": pearson(av, -afv),
            "antisymmetry_sign_flip_rate": float(np.mean(np.sign(av) != np.sign(afv))),
            "n_valid": int(valid.sum()),
        }

    result = {
        "model_path": args.model_path,
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "max_length": args.max_length,
        "examples_evaluated": n,
        "examples_skipped": skipped,
        "perturbation_seed": args.seed,
        "per_config": summary,
        "raw_scores": {k: [float(x) for x in v] for k, v in series.items()},
        "raw_scores_flipped": {k: [float(x) for x in v] for k, v in flipped.items()},
    }

    print("\n=== MATH SUMMARY ===")
    print(f"n = {n} (skipped {skipped} due to perturbation failure)")
    print(f"{'config':>30} | {'acc':>5} | {'std':>5} | {'sign_flip':>9} | {'antisym_r':>9}")
    ordering = ["boundary", "only_loop_1", "only_loop_2", "only_loop_3", "only_loop_4",
                "loop2_norm_x2", "loop2_norm_x3", "loop2_norm_x4", "loop2_norm_x5",
                "only_loop_1_quadnorm", "only_loop_2_quadnorm",
                "only_loop_3_quadnorm", "only_loop_4_quadnorm",
                "mean_replicated"]
    for k in ordering:
        if k not in summary:
            continue
        d = summary[k]
        print(f"  {k:>28} | {d['accuracy']:.3f} | {d['std_score']:.3f} | {d['antisymmetry_sign_flip_rate']:.3f}     | {d['antisymmetry_pearson']:+.3f}")

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
