"""Actual hard-math evaluator runs for selected early and readout layer taps.

This combines two tests on Hendrycks MATH level >= 4:

1. Pairwise readout:
   correct solution vs middle-step-perturbed wrong solution, scored at selected
   intra-loop layer taps and boundary.

2. Branch-selection tournament:
   one correct solution plus N-1 perturbed wrong branches in randomized order,
   selected by pairwise evaluator tournaments at the same taps.

This is the practical follow-up to the HH all-layer sweep: use plausible early
intervention taps (20/24/28/32/36) and best readout taps (40/41/45/46/47) on a
harder non-chat domain.
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
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator_core.pairwise_evaluator import PairwiseEvaluator, validate_hook_output


DEFAULT_MODEL_PATH = "models/ouro_rltt_local"
DEFAULT_CHECKPOINT_PATH = "artifacts/checkpoints/evaluator/pairwise_epoch2.pt"
NUM_UT_LOOPS = 4
NUM_DECODER_LAYERS = 48
DEFAULT_LAYERS = (20, 24, 28, 32, 36, 40, 41, 45, 46, 47)
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
    p.add_argument("--pair-examples", type=int, default=1000)
    p.add_argument("--branch-examples", type=int, default=250)
    p.add_argument("--num-candidates", type=int, default=4)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--min-level", type=int, default=4)
    p.add_argument("--layers", default=",".join(str(x) for x in DEFAULT_LAYERS))
    p.add_argument("--early-exit-threshold", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pair-score-chunk-size", type=int, default=32)
    p.add_argument("--branch-score-chunk-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=44)
    p.add_argument("--report-every", type=int, default=25)
    p.add_argument("--output-json", default="artifacts/reports/evaluator/probe_layer_tap_math_actual.json")
    return p.parse_args()


class SelectedLayerCapture:
    def __init__(self, layers: List[int]) -> None:
        self.layers = list(layers)
        self.boundary_states: List[torch.Tensor] = []
        self.layer_states: Dict[int, List[torch.Tensor]] = {layer: [] for layer in layers}
        self.boundary_validated = False

    def boundary_hook(self, module, inputs, output) -> None:
        hidden_states = output[1]
        if not self.boundary_validated:
            validate_hook_output(hidden_states)
            self.boundary_validated = True
        self.boundary_states = [h.detach() for h in hidden_states]

    def make_layer_hook(self, layer_idx: int):
        def hook(module, inputs, output) -> None:
            tensor = output if isinstance(output, torch.Tensor) else output[0]
            self.layer_states[layer_idx].append(tensor.detach())
        return hook

    def clear(self) -> None:
        self.boundary_states = []
        for layer in self.layers:
            self.layer_states[layer] = []


def attach_hooks(model, capture: SelectedLayerCapture) -> List:
    handles = [model.model.register_forward_hook(capture.boundary_hook)]
    for layer in capture.layers:
        handles.append(model.model.layers[layer].register_forward_hook(capture.make_layer_hook(layer)))
    return handles


def to_fp32(states: List[torch.Tensor], device: torch.device) -> List[torch.Tensor]:
    return [h.to(device=device, dtype=torch.float32) for h in states]


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


def build_pair(problem: str, solution: str, rng: random.Random) -> Optional[Tuple[str, str]]:
    wrong = perturb_middle_number(solution, rng)
    if wrong is None or wrong == solution:
        return None
    return (
        f"Problem: {problem}\n\nSolution: {solution}",
        f"Problem: {problem}\n\nSolution: {wrong}",
    )


def build_candidate_set(
    problem: str,
    solution: str,
    rng: random.Random,
    num_candidates: int,
) -> Optional[Tuple[List[str], int]]:
    wrong_solutions = []
    seen = {solution}
    for _ in range(100):
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
    return [texts[i] for i in order], order.index(0)


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
        match = re.search(r"\d", level)
        return int(match.group()) if match else 5
    return int(level)


def build_specs(layers: List[int]) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, object]]]:
    specs: List[Dict[str, object]] = []

    def add(name: str, **meta: object) -> None:
        item = {"config": name}
        item.update(meta)
        specs.append(item)

    add("boundary_natural", family="boundary_natural", layer=None, loop=None, normed=True)
    for loop_idx in (1, 2, 3, 4):
        add(
            f"boundary_loop_{loop_idx}_replicated",
            family="boundary_loop",
            layer=None,
            loop=loop_idx - 1,
            normed=True,
        )

    for layer in layers:
        add(f"layer_{layer:02d}_natural_raw", family="layer_natural", layer=layer, loop=None, normed=False)
        add(f"layer_{layer:02d}_natural_normed", family="layer_natural", layer=layer, loop=None, normed=True)
        for loop_idx in (2, 3, 4):
            add(
                f"layer_{layer:02d}_loop_{loop_idx}_normed",
                family="layer_loop",
                layer=layer,
                loop=loop_idx - 1,
                normed=True,
            )
    return specs, {str(item["config"]): item for item in specs}


def get_states(spec: Dict[str, object], data: Dict[str, object]) -> List[torch.Tensor]:
    family = str(spec["family"])
    if family == "boundary_natural":
        return list(data["boundary"])
    if family == "boundary_loop":
        loop_idx = int(spec["loop"])
        return [data["boundary"][loop_idx]] * NUM_UT_LOOPS

    layer = int(spec["layer"])
    table = data["layers_normed"] if bool(spec["normed"]) else data["layers_raw"]
    if family == "layer_natural":
        return list(table[layer])
    if family == "layer_loop":
        loop_idx = int(spec["loop"])
        return [table[layer][loop_idx]] * NUM_UT_LOOPS
    raise ValueError(f"unknown family: {family}")


def pad_hidden_batch(tensors: List[torch.Tensor]) -> torch.Tensor:
    max_len = max(t.shape[1] for t in tensors)
    hidden_dim = tensors[0].shape[-1]
    out = tensors[0].new_zeros((len(tensors), max_len, hidden_dim))
    for idx, tensor in enumerate(tensors):
        seq_len = tensor.shape[1]
        out[idx, :seq_len, :] = tensor[0]
    return out


def pad_mask_batch(masks: List[torch.Tensor]) -> torch.Tensor:
    max_len = max(mask.shape[1] for mask in masks)
    out = masks[0].new_zeros((len(masks), max_len))
    for idx, mask in enumerate(masks):
        seq_len = mask.shape[1]
        out[idx, :seq_len] = mask[0]
    return out


@torch.no_grad()
def score_pair_configs(
    evaluator: PairwiseEvaluator,
    specs: List[Dict[str, object]],
    c_data: Dict[str, object],
    c_mask: torch.Tensor,
    r_data: Dict[str, object],
    r_mask: torch.Tensor,
    chunk_size: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    normal: Dict[str, float] = {}
    flipped: Dict[str, float] = {}
    for start in range(0, len(specs), chunk_size):
        chunk = specs[start:start + chunk_size]
        c_slots: List[List[torch.Tensor]] = [[] for _ in range(NUM_UT_LOOPS)]
        r_slots: List[List[torch.Tensor]] = [[] for _ in range(NUM_UT_LOOPS)]
        for spec in chunk:
            cs = get_states(spec, c_data)
            rs = get_states(spec, r_data)
            for slot in range(NUM_UT_LOOPS):
                c_slots[slot].append(cs[slot])
                r_slots[slot].append(rs[slot])
        c_batch = [torch.cat(c_slots[slot], dim=0) for slot in range(NUM_UT_LOOPS)]
        r_batch = [torch.cat(r_slots[slot], dim=0) for slot in range(NUM_UT_LOOPS)]
        c_mask_batch = c_mask.expand(len(chunk), -1).contiguous()
        r_mask_batch = r_mask.expand(len(chunk), -1).contiguous()
        s = evaluator(c_batch, c_mask_batch, r_batch, r_mask_batch).view(-1).detach().float().cpu().numpy()
        sf = evaluator(r_batch, r_mask_batch, c_batch, c_mask_batch).view(-1).detach().float().cpu().numpy()
        for offset, spec in enumerate(chunk):
            normal[str(spec["config"])] = float(s[offset])
            flipped[str(spec["config"])] = float(sf[offset])
    return normal, flipped


@torch.no_grad()
def score_branch_items(
    evaluator: PairwiseEvaluator,
    items: List[Tuple[str, int, int]],
    spec_by_name: Dict[str, Dict[str, object]],
    candidate_data: List[Dict[str, object]],
    candidate_masks: List[torch.Tensor],
    chunk_size: int,
) -> Dict[Tuple[str, int, int], float]:
    scores: Dict[Tuple[str, int, int], float] = {}
    for start in range(0, len(items), chunk_size):
        chunk = items[start:start + chunk_size]
        left_slots: List[List[torch.Tensor]] = [[] for _ in range(NUM_UT_LOOPS)]
        right_slots: List[List[torch.Tensor]] = [[] for _ in range(NUM_UT_LOOPS)]
        left_masks: List[torch.Tensor] = []
        right_masks: List[torch.Tensor] = []
        for config, i, j in chunk:
            spec = spec_by_name[config]
            left = get_states(spec, candidate_data[i])
            right = get_states(spec, candidate_data[j])
            for slot in range(NUM_UT_LOOPS):
                left_slots[slot].append(left[slot])
                right_slots[slot].append(right[slot])
            left_masks.append(candidate_masks[i])
            right_masks.append(candidate_masks[j])
        left_batch = [pad_hidden_batch(left_slots[slot]) for slot in range(NUM_UT_LOOPS)]
        right_batch = [pad_hidden_batch(right_slots[slot]) for slot in range(NUM_UT_LOOPS)]
        left_mask_batch = pad_mask_batch(left_masks)
        right_mask_batch = pad_mask_batch(right_masks)
        values = evaluator(left_batch, left_mask_batch, right_batch, right_mask_batch).view(-1)
        values_np = values.detach().float().cpu().numpy()
        for offset, item in enumerate(chunk):
            scores[item] = float(values_np[offset])
    return scores


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
    corr = 0.0
    if np.std(s) > 1e-9 and np.std(sf) > 1e-9:
        corr = float(np.corrcoef(s, -sf)[0, 1])
    strict = np.sign(s) != np.sign(sf)
    return {
        "n_valid": int(s.size),
        "canonical_acc": float(np.mean(s > 0)),
        "flipped_acc": float(np.mean(sf < 0)),
        "centered_acc": float(np.mean(centered > 0)),
        "normal_pos_rate": float(np.mean(s > 0)),
        "flipped_pos_rate": float(np.mean(sf > 0)),
        "strict_sign_reversal_rate": float(np.mean(strict)),
        "antisym_corr": corr,
        "bias_mean": float(np.mean(bias)),
        "antisym_mean": float(np.mean(antisym)),
        "bias_to_signal": float(abs(np.mean(bias)) / (abs(np.mean(antisym)) + 1e-8)),
        "score_mean": float(np.mean(s)),
        "score_std": float(np.std(s)),
    }


def summarize_branch(stats: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, float | int]]:
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
        }
    return summary


def print_pair_rows(title: str, keys: List[str], rows: Dict[str, Dict[str, float | int]]) -> None:
    print(f"\n=== {title} ===")
    print(f"{'config':>32} | {'canon':>5} | {'center':>6} | {'flip':>5} | {'strict':>6} | {'corr':>6}")
    for key in keys:
        if key not in rows:
            continue
        row = rows[key]
        print(
            f"  {key:>30} | {row['canonical_acc']:.3f} | {row['centered_acc']:.3f} | "
            f"{row['flipped_acc']:.3f} | {row['strict_sign_reversal_rate']:.3f} | "
            f"{row['antisym_corr']:+.3f}"
        )


def print_branch_rows(keys: List[str], rows: Dict[str, Dict[str, float | int]]) -> None:
    print("\n=== BRANCH SELECTION ===")
    print(f"{'config':>32} | {'raw':>5} | {'deb':>5} | {'agree':>5} | {'raw_m':>7} | {'deb_m':>7}")
    for key in keys:
        if key not in rows:
            continue
        row = rows[key]
        print(
            f"  {key:>30} | {row['raw_selection_acc']:.3f} | {row['debiased_selection_acc']:.3f} | "
            f"{row['raw_debiased_agreement']:.3f} | {row['raw_margin_mean']:.3f} | "
            f"{row['debiased_margin_mean']:.3f}"
        )


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    for layer in layers:
        assert 0 <= layer < NUM_DECODER_LAYERS, f"layer {layer} out of range"

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_MODULES_CACHE", str(Path.cwd() / ".hf_modules_cache"))

    device = torch.device(args.device)
    print(f"Model: {args.model_path}")
    print(f"Evaluator: {args.checkpoint}")
    print(f"Layers: {layers}")
    print(f"Pair examples: {args.pair_examples}, branch examples: {args.branch_examples}, device={device}\n")

    specs, spec_by_name = build_specs(layers)
    print(f"Configs: {len(specs)}")

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

    capture = SelectedLayerCapture(layers)
    handles = attach_hooks(model, capture)

    print("Loading MATH dataset...")
    ds = load_math_dataset(args)
    filtered = [i for i in range(len(ds)) if get_level(ds[i]) >= args.min_level]
    rng.shuffle(filtered)
    print(f"Total examples: {len(ds)}, level >= {args.min_level}: {len(filtered)}")

    @torch.no_grad()
    def encode_text(text: str) -> Tuple[Dict[str, object], torch.Tensor]:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
        capture.clear()
        model(**tokens, use_cache=False)
        boundary = to_fp32(capture.boundary_states, device)
        layers_raw = {layer: to_fp32(capture.layer_states[layer], device) for layer in layers}
        for layer in layers:
            if len(layers_raw[layer]) != NUM_UT_LOOPS:
                raise RuntimeError(f"Layer {layer} captured wrong loop count")
        if len(boundary) != NUM_UT_LOOPS:
            raise RuntimeError("Boundary captured wrong loop count")
        layers_normed = {layer: [norm_fn(h) for h in layers_raw[layer]] for layer in layers}
        return {"boundary": boundary, "layers_raw": layers_raw, "layers_normed": layers_normed}, tokens["attention_mask"]

    pair_scores: Dict[str, List[float]] = {str(spec["config"]): [] for spec in specs}
    pair_scores_flipped: Dict[str, List[float]] = {str(spec["config"]): [] for spec in specs}
    pair_used = 0
    pair_skipped = 0

    print("\nRunning pairwise readout...")
    for ds_idx in filtered:
        if pair_used >= args.pair_examples:
            break
        ex = ds[ds_idx]
        problem = ex.get("problem", ex.get("Problem", ""))
        solution = ex.get("solution", ex.get("Solution", ""))
        if not problem or not solution:
            pair_skipped += 1
            continue
        pair = build_pair(problem, solution, rng)
        if pair is None:
            pair_skipped += 1
            continue
        chosen, rejected = pair
        c_data, c_mask = encode_text(chosen)
        r_data, r_mask = encode_text(rejected)
        normal, flipped = score_pair_configs(
            evaluator,
            specs,
            c_data,
            c_mask,
            r_data,
            r_mask,
            args.pair_score_chunk_size,
        )
        for name in pair_scores:
            pair_scores[name].append(normal[name])
            pair_scores_flipped[name].append(flipped[name])
        pair_used += 1
        if pair_used % args.report_every == 0 or pair_used == args.pair_examples:
            preview = []
            for name in ("boundary_natural", "layer_24_natural_normed", "layer_36_natural_normed",
                         "layer_45_natural_normed", "layer_47_natural_normed"):
                acc = sum(1 for x in pair_scores[name] if x > 0) / pair_used
                preview.append(f"{name}={acc:.3f}")
            print(f"[pair {pair_used}/{args.pair_examples}] " + "  ".join(preview))
            if device.type == "cuda":
                torch.cuda.empty_cache()

    pair_summary = {name: metric_block(pair_scores[name], pair_scores_flipped[name]) for name in pair_scores}

    branch_stats: Dict[str, Dict[str, object]] = {
        str(spec["config"]): {
            "n": 0,
            "raw_correct": 0,
            "debiased_correct": 0,
            "agreement": 0,
            "raw_margin": [],
            "debiased_margin": [],
        }
        for spec in specs
    }
    branch_examples = []
    correct_position_counts = [0 for _ in range(args.num_candidates)]
    branch_used = 0
    branch_skipped = 0

    print("\nRunning branch-selection tournaments...")
    # Use a fresh shuffled traversal so branch and pair examples are not identical.
    branch_indices = list(filtered)
    rng.shuffle(branch_indices)
    config_names = [str(spec["config"]) for spec in specs]
    for ds_idx in branch_indices:
        if branch_used >= args.branch_examples:
            break
        ex = ds[ds_idx]
        problem = ex.get("problem", ex.get("Problem", ""))
        solution = ex.get("solution", ex.get("Solution", ""))
        if not problem or not solution:
            branch_skipped += 1
            continue
        candidate_set = build_candidate_set(problem, solution, rng, args.num_candidates)
        if candidate_set is None:
            branch_skipped += 1
            continue
        candidates, correct_idx = candidate_set
        candidate_data = []
        candidate_masks = []
        for text in candidates:
            data, mask = encode_text(text)
            candidate_data.append(data)
            candidate_masks.append(mask)

        items = [
            (config, i, j)
            for config in config_names
            for i in range(args.num_candidates)
            for j in range(args.num_candidates)
            if i != j
        ]
        score_map = score_branch_items(
            evaluator,
            items,
            spec_by_name,
            candidate_data,
            candidate_masks,
            args.branch_score_chunk_size,
        )

        for config in config_names:
            matrix = np.full((args.num_candidates, args.num_candidates), np.nan, dtype=np.float64)
            for i in range(args.num_candidates):
                for j in range(args.num_candidates):
                    if i == j:
                        continue
                    matrix[i, j] = score_map[(config, i, j)]
            raw_scores = np.nansum(matrix, axis=1)
            debiased_scores = np.zeros(args.num_candidates, dtype=np.float64)
            for i in range(args.num_candidates):
                for j in range(args.num_candidates):
                    if i != j:
                        debiased_scores[i] += 0.5 * (matrix[i, j] - matrix[j, i])
            raw_winner = int(np.argmax(raw_scores))
            debiased_winner = int(np.argmax(debiased_scores))
            row = branch_stats[config]
            row["n"] = int(row["n"]) + 1
            row["raw_correct"] = int(row["raw_correct"]) + int(raw_winner == correct_idx)
            row["debiased_correct"] = int(row["debiased_correct"]) + int(debiased_winner == correct_idx)
            row["agreement"] = int(row["agreement"]) + int(raw_winner == debiased_winner)
            row["raw_margin"].append(float(np.sort(raw_scores)[-1] - np.sort(raw_scores)[-2]))
            row["debiased_margin"].append(float(np.sort(debiased_scores)[-1] - np.sort(debiased_scores)[-2]))

        correct_position_counts[correct_idx] += 1
        branch_examples.append({"dataset_index": int(ds_idx), "correct_position": int(correct_idx)})
        branch_used += 1
        if branch_used % args.report_every == 0 or branch_used == args.branch_examples:
            summary = summarize_branch(branch_stats)
            preview = []
            for name in ("boundary_natural", "layer_24_natural_normed", "layer_36_natural_normed",
                         "layer_45_natural_normed", "layer_47_natural_normed"):
                row = summary[name]
                preview.append(f"{name}:raw={row['raw_selection_acc']:.3f},deb={row['debiased_selection_acc']:.3f}")
            print(f"[branch {branch_used}/{args.branch_examples}] " + "  ".join(preview))
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for handle in handles:
        handle.remove()

    branch_summary = summarize_branch(branch_stats)
    result = {
        "model_path": args.model_path,
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "min_level": args.min_level,
        "layers": layers,
        "max_length": args.max_length,
        "pair_examples_evaluated": pair_used,
        "pair_examples_skipped": pair_skipped,
        "branch_examples_evaluated": branch_used,
        "branch_examples_skipped": branch_skipped,
        "num_candidates": args.num_candidates,
        "seed": args.seed,
        "correct_position_counts": correct_position_counts,
        "config_metadata": spec_by_name,
        "pair_per_config": pair_summary,
        "branch_per_config": branch_summary,
        "pair_raw_scores": {k: [float(x) for x in v] for k, v in pair_scores.items()},
        "pair_raw_scores_flipped": {k: [float(x) for x in v] for k, v in pair_scores_flipped.items()},
        "branch_examples": branch_examples,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    selected = [
        "boundary_natural",
        "boundary_loop_4_replicated",
        "layer_20_natural_normed",
        "layer_24_natural_normed",
        "layer_28_natural_normed",
        "layer_32_natural_normed",
        "layer_36_natural_normed",
        "layer_40_natural_normed",
        "layer_41_natural_normed",
        "layer_45_natural_normed",
        "layer_46_natural_normed",
        "layer_47_natural_normed",
    ]
    print_pair_rows("PAIRWISE HARD-MATH READOUT", selected, pair_summary)
    print_branch_rows(selected, branch_summary)

    top_pair = sorted(pair_summary.items(), key=lambda kv: (-float(kv[1]["canonical_acc"]), -float(kv[1]["centered_acc"])))[:12]
    print_pair_rows("TOP PAIRWISE BY CANONICAL", [name for name, _ in top_pair], pair_summary)
    top_branch = sorted(branch_summary.items(), key=lambda kv: (-float(kv[1]["raw_selection_acc"]), -float(kv[1]["debiased_selection_acc"])))[:12]
    print_branch_rows([name for name, _ in top_branch], branch_summary)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
