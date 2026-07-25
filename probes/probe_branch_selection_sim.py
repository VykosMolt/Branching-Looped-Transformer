"""Branch-selection simulation for the frozen pairwise evaluator.

Builds unordered candidate sets from Hendrycks MATH:
  - one correct solution
  - N-1 wrong branches made by perturbing middle-step numbers

For each evaluator configuration, it compares two selection rules:
  - raw tournament:       argmax_i sum_j score(i, j)
  - debiased tournament:  argmax_i sum_j 0.5 * (score(i, j) - score(j, i))

The debiased rule is the control-relevant one if raw score carries a symmetric
positive offset.
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
from typing import Dict, List, Optional, Tuple

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
NUMBER_RE = re.compile(r"(?<![\\\d.])(-?\d+(?:\.\d+)?)(?![\d.])")
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
HENDRYCKS_CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument("--dataset", default="lighteval/MATH")
    p.add_argument("--dataset-config", default="all")
    p.add_argument("--split", default="test")
    p.add_argument("--max-examples", type=int, default=250)
    p.add_argument("--num-candidates", type=int, default=4)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--min-level", type=int, default=4)
    p.add_argument("--early-exit-threshold", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--report-every", type=int, default=25)
    p.add_argument("--output-json", default="artifacts/reports/evaluator/probe_branch_selection_sim.json")
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


def score(evaluator, left_states, left_mask, right_states, right_mask) -> float:
    with torch.no_grad():
        s = evaluator(left_states, left_mask, right_states, right_mask)
    val = float(s.view(-1).item())
    return val if math.isfinite(val) else float("nan")


def iter_norm(x: torch.Tensor, norm_fn, extra_k: int) -> torch.Tensor:
    y = x
    for _ in range(extra_k):
        y = norm_fn(y)
    return y


def replicated(x: torch.Tensor) -> List[torch.Tensor]:
    return [x] * NUM_UT_LOOPS


def config_states(boundaries: List[torch.Tensor], config: str, norm_fn) -> List[torch.Tensor]:
    if config == "boundary_natural":
        return list(boundaries)
    if config == "only_loop_2":
        return replicated(boundaries[1])
    if config == "loop2_total_x8":
        return replicated(iter_norm(boundaries[1], norm_fn, 7))
    if config == "loop3_extra_k3":
        return replicated(iter_norm(boundaries[2], norm_fn, 3))
    if config == "loop4_extra_k0":
        return replicated(boundaries[3])
    if config == "loop4_extra_k2":
        return replicated(iter_norm(boundaries[3], norm_fn, 2))
    if config == "mean_replicated":
        return replicated(torch.stack(boundaries, dim=0).mean(dim=0))
    if config == "mean_total_x8":
        mean = torch.stack(boundaries, dim=0).mean(dim=0)
        return replicated(iter_norm(mean, norm_fn, 7))
    if config == "natural_seq_total_x8":
        return [iter_norm(boundaries[i], norm_fn, 7) for i in range(NUM_UT_LOOPS)]
    raise ValueError(f"unknown config: {config}")


def find_middle_numbers(solution: str) -> List[Tuple[int, int, str]]:
    boxed_spans = [m.span() for m in BOXED_RE.finditer(solution)]

    def in_boxed(start: int, end: int) -> bool:
        return any(start >= bs and end <= be for bs, be in boxed_spans)

    matches = []
    for m in NUMBER_RE.finditer(solution):
        start, end = m.span(1)
        if in_boxed(start, end):
            continue
        value_str = m.group(1)
        try:
            value = float(value_str)
        except ValueError:
            continue
        if abs(value) < 1.0 and value_str != "-1":
            continue
        matches.append((start, end, value_str))
    return matches


def perturb_middle_number(solution: str, rng: random.Random) -> Optional[str]:
    matches = find_middle_numbers(solution)
    if not matches:
        return None
    start, end, value_str = rng.choice(matches)
    try:
        if "." in value_str:
            value = float(value_str)
            new_value = value * rng.choice([2.0, 0.5, 3.0, -1.0, 1.5, 0.333])
            new_value += rng.choice([-1.0, 0.0, 1.0])
            new_str = f"{new_value:.3f}".rstrip("0").rstrip(".") or "0"
        else:
            value = int(value_str)
            magnitude = max(abs(value), 1)
            new_value = value + rng.choice([-1, 1, -2, 2, -3, 3, -magnitude, magnitude])
            if new_value == value:
                new_value = value + 1
            new_str = str(new_value)
    except (OverflowError, ValueError):
        return None
    if new_str == value_str:
        return None
    return solution[:start] + new_str + solution[end:]


def build_candidate_set(
    problem: str,
    solution: str,
    rng: random.Random,
    num_candidates: int,
) -> Optional[Tuple[List[str], int]]:
    wrong_solutions = []
    seen = {solution}
    for _ in range(80):
        wrong = perturb_middle_number(solution, rng)
        if not wrong or wrong in seen:
            continue
        seen.add(wrong)
        wrong_solutions.append(wrong)
        if len(wrong_solutions) >= num_candidates - 1:
            break
    if len(wrong_solutions) < num_candidates - 1:
        return None

    solutions = [solution] + wrong_solutions
    texts = [f"Problem: {problem}\n\nSolution: {sol}" for sol in solutions]
    order = list(range(num_candidates))
    rng.shuffle(order)
    shuffled = [texts[i] for i in order]
    correct_index = order.index(0)
    return shuffled, correct_index


def load_math_dataset(args: argparse.Namespace):
    errors = {}
    try:
        ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
        print(f"Loaded {args.dataset}/{args.dataset_config}")
        return ds
    except Exception as exc:
        errors[f"{args.dataset}/{args.dataset_config}"] = repr(exc)

    try:
        ds = load_dataset("hendrycks/competition_math", split=args.split)
        print("Loaded hendrycks/competition_math")
        return ds
    except Exception as exc:
        errors["hendrycks/competition_math"] = repr(exc)

    parts = []
    try:
        for cfg in HENDRYCKS_CONFIGS:
            part = load_dataset("EleutherAI/hendrycks_math", cfg, split=args.split)
            parts.append(part)
            print(f"Loaded EleutherAI/hendrycks_math/{cfg}")
        return concatenate_datasets(parts)
    except Exception as exc:
        errors["EleutherAI/hendrycks_math/*"] = repr(exc)

    details = "\n".join(f"{name}: {err}" for name, err in errors.items())
    raise RuntimeError(f"Unable to load MATH dataset:\n{details}")


def get_level(ex) -> int:
    level = ex.get("level", ex.get("Level", 5))
    if isinstance(level, str):
        m = re.search(r"\d", level)
        return int(m.group()) if m else 5
    return int(level)


def update_stats(stats: Dict[str, Dict[str, object]], config: str, raw_scores, debiased_scores, correct_idx: int) -> None:
    raw_winner = int(np.argmax(raw_scores))
    deb_winner = int(np.argmax(debiased_scores))
    row = stats[config]
    row["n"] = int(row["n"]) + 1
    row["raw_correct"] = int(row["raw_correct"]) + int(raw_winner == correct_idx)
    row["debiased_correct"] = int(row["debiased_correct"]) + int(deb_winner == correct_idx)
    row["agreement"] = int(row["agreement"]) + int(raw_winner == deb_winner)
    row["raw_correct_score"].append(float(raw_scores[correct_idx]))
    row["debiased_correct_score"].append(float(debiased_scores[correct_idx]))
    sorted_raw = np.sort(raw_scores)
    sorted_deb = np.sort(debiased_scores)
    row["raw_margin"].append(float(sorted_raw[-1] - sorted_raw[-2]))
    row["debiased_margin"].append(float(sorted_deb[-1] - sorted_deb[-2]))


def summarize_stats(stats: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, float | int]]:
    summary = {}
    for config, row in stats.items():
        n = int(row["n"])
        if n == 0:
            continue
        summary[config] = {
            "n": n,
            "raw_selection_acc": float(int(row["raw_correct"]) / n),
            "debiased_selection_acc": float(int(row["debiased_correct"]) / n),
            "raw_debiased_agreement": float(int(row["agreement"]) / n),
            "raw_margin_mean": float(np.mean(row["raw_margin"])),
            "debiased_margin_mean": float(np.mean(row["debiased_margin"])),
            "raw_correct_score_mean": float(np.mean(row["raw_correct_score"])),
            "debiased_correct_score_mean": float(np.mean(row["debiased_correct_score"])),
        }
    return summary


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_MODULES_CACHE", str(Path.cwd() / ".hf_modules_cache"))

    device = torch.device(args.device)
    print(f"Model: {args.model_path}")
    print(f"Evaluator: {args.checkpoint}")
    print(f"Candidates: {args.num_candidates}, examples target: {args.max_examples}, device={device}\n")

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

    @torch.no_grad()
    def norm_fn(x: torch.Tensor) -> torch.Tensor:
        return model.model.norm(x.to(dtype=torch.bfloat16)).to(dtype=torch.float32)

    print("Loading evaluator...")
    evaluator = PairwiseEvaluator().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    evaluator.load_state_dict(state_dict)
    evaluator.eval()

    capture = BoundaryCapture()
    handle = model.model.register_forward_hook(capture.hook)

    print("Loading MATH dataset...")
    ds = load_math_dataset(args)
    filtered = [i for i in range(len(ds)) if get_level(ds[i]) >= args.min_level]
    rng.shuffle(filtered)
    print(f"Total examples: {len(ds)}, level >= {args.min_level}: {len(filtered)}")

    configs = [
        "boundary_natural",
        "only_loop_2",
        "loop2_total_x8",
        "loop3_extra_k3",
        "loop4_extra_k0",
        "loop4_extra_k2",
        "mean_replicated",
        "mean_total_x8",
        "natural_seq_total_x8",
    ]
    stats: Dict[str, Dict[str, object]] = {
        config: {
            "n": 0,
            "raw_correct": 0,
            "debiased_correct": 0,
            "agreement": 0,
            "raw_margin": [],
            "debiased_margin": [],
            "raw_correct_score": [],
            "debiased_correct_score": [],
        }
        for config in configs
    }
    correct_position_counts = [0 for _ in range(args.num_candidates)]

    used = 0
    skipped = 0
    examples = []

    for ds_idx in filtered:
        if used >= args.max_examples:
            break
        ex = ds[ds_idx]
        problem = ex.get("problem", ex.get("Problem", ""))
        solution = ex.get("solution", ex.get("Solution", ""))
        if not problem or not solution:
            skipped += 1
            continue

        candidate_set = build_candidate_set(problem, solution, rng, args.num_candidates)
        if candidate_set is None:
            skipped += 1
            continue
        candidates, correct_idx = candidate_set
        correct_position_counts[correct_idx] += 1

        states = []
        masks = []
        for text in candidates:
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
            capture.clear()
            with torch.no_grad():
                model(**tokens, use_cache=False)
            states.append(to_fp32(capture.states, device))
            masks.append(tokens["attention_mask"])

        for config in configs:
            config_cache = [config_states(boundary, config, norm_fn) for boundary in states]
            matrix = np.full((args.num_candidates, args.num_candidates), np.nan, dtype=np.float64)
            for i in range(args.num_candidates):
                for j in range(args.num_candidates):
                    if i == j:
                        continue
                    matrix[i, j] = score(evaluator, config_cache[i], masks[i], config_cache[j], masks[j])

            raw_scores = np.nansum(matrix, axis=1)
            debiased_scores = np.zeros(args.num_candidates, dtype=np.float64)
            for i in range(args.num_candidates):
                for j in range(args.num_candidates):
                    if i == j:
                        continue
                    debiased_scores[i] += 0.5 * (matrix[i, j] - matrix[j, i])
            update_stats(stats, config, raw_scores, debiased_scores, correct_idx)

        used += 1
        examples.append({"dataset_index": int(ds_idx), "correct_position": int(correct_idx)})

        if used % args.report_every == 0 or used == args.max_examples:
            summary = summarize_stats(stats)
            line = f"[{used}/{args.max_examples}]"
            for config in ("boundary_natural", "loop2_total_x8", "loop4_extra_k0", "mean_total_x8"):
                row = summary[config]
                line += f"  {config}:raw={row['raw_selection_acc']:.3f},deb={row['debiased_selection_acc']:.3f}"
            print(line)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    handle.remove()

    summary = summarize_stats(stats)
    result = {
        "model_path": args.model_path,
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "min_level": args.min_level,
        "max_length": args.max_length,
        "num_candidates": args.num_candidates,
        "examples_evaluated": used,
        "examples_skipped": skipped,
        "seed": args.seed,
        "correct_position_counts": correct_position_counts,
        "per_config": summary,
        "examples": examples,
    }

    print("\n=== BRANCH-SELECTION SUMMARY ===")
    print(f"{'config':>28} | {'raw':>5} | {'deb':>5} | {'agree':>5} | {'raw_m':>7} | {'deb_m':>7}")
    for config in configs:
        row = summary[config]
        print(
            f"  {config:>26} | {row['raw_selection_acc']:.3f} | "
            f"{row['debiased_selection_acc']:.3f} | {row['raw_debiased_agreement']:.3f} | "
            f"{row['raw_margin_mean']:.3f} | {row['debiased_margin_mean']:.3f}"
        )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
