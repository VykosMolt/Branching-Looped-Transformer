"""Multi-tap evaluator ensemble probe.

This is the evaluator-first follow-up to the layer-tap runs. It tests the
proposed basal-ganglia interface directly:

* raw layer 24 as early gate
* raw layer 36 as mid check
* raw layers 45/46/47 as final selectors
* boundary as fallback/baseline

For every candidate set, the script runs O(N^2) pairwise evaluator tournaments
per tap and compares:

* per-tap raw and debiased winners
* vote over per-tap raw winners
* vote over per-tap debiased winners
* summed raw tournament scores
* summed debiased tournament scores
* held-out calibrated fusion over per-tap debiased scores

Coding examples use less trivial synthetic mutants than the earlier cached-domain
probe: operator, loop-bound, condition, numeric, and return-expression mutations
are preferred before obvious constant-return fallbacks.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
DEFAULT_LAYERS = (20, 24, 36, 40, 45, 46, 47)
RETURN_RE = re.compile(r"(^[ \t]*return\s+)(.+?)([ \t]*(?:#.*)?$)", re.MULTILINE)
NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")
HENDRYCKS_CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def load_helper(module_name: str, rel_path: str):
    path = PROJECT_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helper {rel_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


DOMAIN = load_helper("probe_layer_tap_cached_domains_helper", "probes/probe_layer_tap_cached_domains.py")
MATH = load_helper("probe_layer_tap_math_actual_helper", "probes/probe_layer_tap_math_actual.py")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument("--datasets", default="humaneval,mbpp_sanitized,arc_challenge,arc_easy,bbh_boolean,bbh_multistep,bbh_logical_five,bbh_tracking,strategyqa")
    p.add_argument("--max-examples-per-dataset", type=int, default=80)
    p.add_argument("--include-math", action="store_true")
    p.add_argument("--math-examples", type=int, default=200)
    p.add_argument("--math-min-level", type=int, default=4)
    p.add_argument("--num-candidates", type=int, default=4)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--layers", default=",".join(str(x) for x in DEFAULT_LAYERS))
    p.add_argument("--early-exit-threshold", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--branch-score-chunk-size", type=int, default=16)
    p.add_argument("--calibration-fraction", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=61)
    p.add_argument("--report-every", type=int, default=20)
    p.add_argument("--output-json", default="artifacts/reports/evaluator/probe_multitap_ensemble.json")
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


def build_specs(layers: Sequence[int]) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, object]]]:
    specs: List[Dict[str, object]] = [
        {"config": "boundary_natural", "family": "boundary_natural", "layer": None, "loop": None, "normed": True}
    ]
    for layer in layers:
        specs.append(
            {
                "config": f"layer_{layer:02d}_natural_raw",
                "family": "layer_natural",
                "layer": int(layer),
                "loop": None,
                "normed": False,
            }
        )
    return specs, {str(item["config"]): item for item in specs}


def perturb_number_text(text: str, rng: random.Random) -> str:
    def repl(match: re.Match[str]) -> str:
        value = match.group(0)
        try:
            if "." in value:
                new_value = float(value) + rng.choice([-2.0, -1.0, 1.0, 2.0])
                return f"{new_value:.3f}".rstrip("0").rstrip(".")
            return str(int(value) + rng.choice([-3, -2, -1, 1, 2, 3]))
        except ValueError:
            return value

    return NUMBER_RE.sub(repl, text, count=1)


def replace_return_expr(code: str, transform) -> Optional[str]:
    match = RETURN_RE.search(code)
    if not match:
        return None
    expr = match.group(2).strip()
    replacement = transform(expr)
    if not replacement or replacement == expr:
        return None
    start, end = match.span(2)
    return code[:start] + replacement + code[end:]


def add_unique(items: List[str], value: Optional[str], forbidden: Sequence[str], limit: int) -> None:
    if value is None:
        return
    cleaned = value.strip()
    if not cleaned or cleaned in forbidden or cleaned in items:
        return
    if len(items) < limit:
        items.append(cleaned)


def harder_code_mutants(code: str, rng: random.Random, limit: int) -> List[str]:
    mutants: List[str] = []
    forbidden = (code,)

    replacements = (
        (" + ", " - "),
        (" - ", " + "),
        (" * ", " + "),
        (" // ", " / "),
        (" % ", " // "),
        (" == ", " != "),
        (" != ", " == "),
        (" <= ", " < "),
        (" >= ", " > "),
        (" < ", " <= "),
        (" > ", " >= "),
        (" and ", " or "),
        (" or ", " and "),
        ("range(n)", "range(n - 1)"),
        ("range(1, n)", "range(1, n - 1)"),
        ("range(len(", "range(len("),
        ("reverse=True", "reverse=False"),
    )
    for old, new in replacements:
        if old in code and old != new:
            add_unique(mutants, code.replace(old, new, 1), forbidden, limit)
        if len(mutants) >= limit:
            return mutants

    if NUMBER_RE.search(code):
        add_unique(mutants, perturb_number_text(code, rng), forbidden, limit)

    return_transforms = (
        lambda expr: f"({expr}) + 1",
        lambda expr: f"({expr}) - 1",
        lambda expr: f"not ({expr})",
        lambda expr: f"list(reversed({expr}))",
        lambda expr: f"sorted({expr})",
        lambda expr: f"({expr})[:-1]",
    )
    for transform in return_transforms:
        add_unique(mutants, replace_return_expr(code, transform), forbidden, limit)
        if len(mutants) >= limit:
            return mutants

    # Fallbacks are intentionally last. They keep examples usable when a short
    # reference solution has no obvious perturbation point.
    for replacement in ("None", "0", "False"):
        add_unique(mutants, DOMAIN.replace_first_return(code, replacement), forbidden, limit)
    return mutants[:limit]


def build_harder_code_examples(ds, spec, max_examples: int, rng: random.Random):
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    rows = []
    for idx in indices:
        if len(rows) >= max_examples:
            break
        ex = ds[idx]
        if spec.kind == "humaneval":
            prompt = DOMAIN.normalize_text(ex.get("prompt"))
            solution = DOMAIN.normalize_text(ex.get("canonical_solution"))
            if not prompt or not solution:
                continue
            mutants = harder_code_mutants(solution, rng, limit=8)
            if not mutants:
                continue
            correct = DOMAIN.format_code_candidate(prompt, prompt + solution)
            wrongs = tuple(DOMAIN.format_code_candidate(prompt, prompt + mutant) for mutant in mutants)
        else:
            prompt = DOMAIN.normalize_text(ex.get("prompt"))
            code = DOMAIN.normalize_text(ex.get("code"))
            tests = ex.get("test_list") or []
            if not prompt or not code:
                continue
            mutants = harder_code_mutants(code, rng, limit=8)
            if not mutants:
                continue
            correct = DOMAIN.format_code_candidate(prompt, code, tests)
            wrongs = tuple(DOMAIN.format_code_candidate(prompt, mutant, tests) for mutant in mutants)
        rows.append(DOMAIN.CandidateExample(spec.tag, spec.domain, idx, correct, wrongs))
    return rows


def build_domain_examples(ds, spec, max_examples: int, rng: random.Random):
    if spec.kind in {"humaneval", "mbpp"}:
        return build_harder_code_examples(ds, spec, max_examples, rng)
    return DOMAIN.build_examples(ds, spec, max_examples, rng)


def load_math_dataset():
    errors = {}
    try:
        return load_dataset("lighteval/MATH", "all", split="test")
    except Exception as exc:
        errors["lighteval/MATH/all"] = repr(exc)
    try:
        return load_dataset("hendrycks/competition_math", split="test")
    except Exception as exc:
        errors["hendrycks/competition_math"] = repr(exc)
    parts = []
    try:
        for cfg in HENDRYCKS_CONFIGS:
            parts.append(load_dataset("EleutherAI/hendrycks_math", cfg, split="test"))
        return concatenate_datasets(parts)
    except Exception as exc:
        errors["EleutherAI/hendrycks_math/*"] = repr(exc)
    raise RuntimeError(errors)


def get_states(spec: Dict[str, object], data: Dict[str, object]) -> List[torch.Tensor]:
    if spec["family"] == "boundary_natural":
        return list(data["boundary"])
    layer = int(spec["layer"])
    return list(data["layers_raw"][layer])


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


def row_scores_from_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    raw_scores = np.nansum(matrix, axis=1)
    debiased_scores = np.zeros(matrix.shape[0], dtype=np.float64)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            if i != j:
                debiased_scores[i] += 0.5 * (matrix[i, j] - matrix[j, i])
    return raw_scores, debiased_scores


def zscore(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    if std < 1e-9:
        return values * 0.0
    return (values - float(np.mean(values))) / std


def argmax_with_tiebreak(votes: Sequence[int], tiebreak_scores: np.ndarray) -> int:
    counts = Counter(votes)
    best_count = max(counts.values())
    tied = [idx for idx, count in counts.items() if count == best_count]
    if len(tied) == 1:
        return int(tied[0])
    return int(max(tied, key=lambda idx: float(tiebreak_scores[idx])))


def score_ensembles(record: dict, weights: Optional[Dict[str, float]] = None) -> Dict[str, int]:
    tap_names = list(record["tap_raw_scores"].keys())
    raw_arrays = {name: np.array(record["tap_raw_scores"][name], dtype=np.float64) for name in tap_names}
    deb_arrays = {name: np.array(record["tap_debiased_scores"][name], dtype=np.float64) for name in tap_names}
    raw_sum = np.sum(np.stack([raw_arrays[name] for name in tap_names]), axis=0)
    deb_sum = np.sum(np.stack([deb_arrays[name] for name in tap_names]), axis=0)
    raw_votes = [int(np.argmax(raw_arrays[name])) for name in tap_names]
    deb_votes = [int(np.argmax(deb_arrays[name])) for name in tap_names]

    out = {
        "vote_raw": argmax_with_tiebreak(raw_votes, raw_sum),
        "vote_debiased": argmax_with_tiebreak(deb_votes, deb_sum),
        "sum_raw": int(np.argmax(raw_sum)),
        "sum_debiased": int(np.argmax(deb_sum)),
    }
    if weights:
        calibrated = np.zeros_like(deb_sum)
        for name in tap_names:
            calibrated += float(weights.get(name, 0.0)) * zscore(deb_arrays[name])
        out["calibrated_debiased"] = int(np.argmax(calibrated))

    # Explicit proposed architecture: early gate + mid check + late selector.
    arch_names = [name for name in ("layer_24_natural_raw", "layer_36_natural_raw", "layer_45_natural_raw") if name in deb_arrays]
    if arch_names:
        arch_sum = np.sum(np.stack([zscore(deb_arrays[name]) for name in arch_names]), axis=0)
        out["arch_24_36_45"] = int(np.argmax(arch_sum))
    arch_late = [name for name in ("layer_24_natural_raw", "layer_36_natural_raw", "layer_46_natural_raw") if name in deb_arrays]
    if arch_late:
        arch_sum = np.sum(np.stack([zscore(deb_arrays[name]) for name in arch_late]), axis=0)
        out["arch_24_36_46"] = int(np.argmax(arch_sum))
    return out


def summarize_predictions(records: List[dict], predictions_by_method: Dict[str, List[int]]) -> Dict[str, dict]:
    summary = {}
    for method, preds in predictions_by_method.items():
        correct = [int(pred == rec["correct_idx"]) for pred, rec in zip(preds, records)]
        margins = []
        for pred, rec in zip(preds, records):
            if method.startswith("tap_raw:"):
                name = method.split(":", 1)[1]
                scores = np.array(rec["tap_raw_scores"][name], dtype=np.float64)
            elif method.startswith("tap_debiased:"):
                name = method.split(":", 1)[1]
                scores = np.array(rec["tap_debiased_scores"][name], dtype=np.float64)
            else:
                continue
            sorted_scores = np.sort(scores)
            margins.append(float(sorted_scores[-1] - sorted_scores[-2]) if len(sorted_scores) > 1 else 0.0)
        summary[method] = {
            "n": len(records),
            "accuracy": float(np.mean(correct)) if correct else 0.0,
            "correct": int(sum(correct)),
            "margin_mean": float(np.mean(margins)) if margins else None,
        }
    return summary


def split_records(records: List[dict], calibration_fraction: float) -> Tuple[List[dict], List[dict]]:
    by_dataset: Dict[str, List[dict]] = defaultdict(list)
    for record in records:
        by_dataset[record["dataset"]].append(record)
    cal: List[dict] = []
    eval_records: List[dict] = []
    for rows in by_dataset.values():
        cutoff = max(1, min(len(rows) - 1, int(math.floor(len(rows) * calibration_fraction)))) if len(rows) > 1 else 0
        cal.extend(rows[:cutoff])
        eval_records.extend(rows[cutoff:])
    return cal, eval_records


def train_simple_weights(records: List[dict], tap_names: Sequence[str]) -> Dict[str, float]:
    if not records:
        return {name: 1.0 / len(tap_names) for name in tap_names}
    weights = {}
    for name in tap_names:
        correct = []
        baselines = []
        for rec in records:
            scores = np.array(rec["tap_debiased_scores"][name], dtype=np.float64)
            correct.append(int(np.argmax(scores) == rec["correct_idx"]))
            baselines.append(1.0 / len(scores))
        acc = float(np.mean(correct))
        baseline = float(np.mean(baselines))
        weights[name] = max(0.0, acc - baseline)
    total = sum(weights.values())
    if total <= 1e-9:
        return {name: 1.0 / len(tap_names) for name in tap_names}
    return {name: value / total for name, value in weights.items()}


def make_predictions(records: List[dict], weights: Optional[Dict[str, float]] = None) -> Dict[str, List[int]]:
    predictions: Dict[str, List[int]] = defaultdict(list)
    for rec in records:
        for name, values in rec["tap_raw_scores"].items():
            predictions[f"tap_raw:{name}"].append(int(np.argmax(np.array(values, dtype=np.float64))))
        for name, values in rec["tap_debiased_scores"].items():
            predictions[f"tap_debiased:{name}"].append(int(np.argmax(np.array(values, dtype=np.float64))))
        for method, pred in score_ensembles(rec, weights).items():
            predictions[method].append(pred)
    return dict(predictions)


def build_record(
    evaluator: PairwiseEvaluator,
    specs: List[Dict[str, object]],
    spec_by_name: Dict[str, Dict[str, object]],
    candidate_data: List[Dict[str, object]],
    candidate_masks: List[torch.Tensor],
    correct_idx: int,
    metadata: dict,
    chunk_size: int,
) -> dict:
    n_candidates = len(candidate_data)
    config_names = [str(spec["config"]) for spec in specs]
    items = [
        (config, i, j)
        for config in config_names
        for i in range(n_candidates)
        for j in range(n_candidates)
        if i != j
    ]
    score_map = score_branch_items(evaluator, items, spec_by_name, candidate_data, candidate_masks, chunk_size)
    tap_raw_scores = {}
    tap_debiased_scores = {}
    for config in config_names:
        matrix = np.full((n_candidates, n_candidates), np.nan, dtype=np.float64)
        for i in range(n_candidates):
            for j in range(n_candidates):
                if i != j:
                    matrix[i, j] = score_map[(config, i, j)]
        raw_scores, debiased_scores = row_scores_from_matrix(matrix)
        tap_raw_scores[config] = [float(x) for x in raw_scores]
        tap_debiased_scores[config] = [float(x) for x in debiased_scores]
    return {
        **metadata,
        "correct_idx": int(correct_idx),
        "candidate_count": int(n_candidates),
        "tap_raw_scores": tap_raw_scores,
        "tap_debiased_scores": tap_debiased_scores,
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    for layer in layers:
        assert 0 <= layer < NUM_DECODER_LAYERS, f"layer {layer} out of range"

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    specs, spec_by_name = build_specs(layers)
    tap_names = [str(spec["config"]) for spec in specs]
    selected_dataset_tags = {name.strip() for name in args.datasets.split(",") if name.strip()}
    selected_domain_specs = [spec for spec in DOMAIN.DATASET_SPECS if spec.tag in selected_dataset_tags]

    device = torch.device(args.device)
    print(f"Model: {args.model_path}")
    print(f"Evaluator: {args.checkpoint}")
    print(f"Layers: {layers}")
    print(f"Taps: {tap_names}")
    print(f"Datasets: {[spec.tag for spec in selected_domain_specs]}, include_math={args.include_math}")
    print(f"Device: {device}\n")

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

    print("Loading evaluator...")
    evaluator = PairwiseEvaluator().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    evaluator.load_state_dict(state_dict)
    evaluator.eval()

    capture = SelectedLayerCapture(layers)
    handles = attach_hooks(model, capture)

    @torch.no_grad()
    def encode_text(text: str) -> Tuple[Dict[str, object], torch.Tensor]:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
        capture.clear()
        model(**tokens, use_cache=False)
        boundary = to_fp32(capture.boundary_states, device)
        layers_raw = {layer: to_fp32(capture.layer_states[layer], device) for layer in layers}
        for layer in layers:
            if len(layers_raw[layer]) != NUM_UT_LOOPS:
                raise RuntimeError(f"Layer {layer} captured wrong loop count: {len(layers_raw[layer])}")
        if len(boundary) != NUM_UT_LOOPS:
            raise RuntimeError(f"Boundary captured wrong loop count: {len(boundary)}")
        return {"boundary": boundary, "layers_raw": layers_raw}, tokens["attention_mask"]

    records: List[dict] = []
    dataset_counts = {}
    try:
        for spec in selected_domain_specs:
            print(f"\nLoading {spec.tag}: {spec.path} {spec.config or ''} {spec.split}")
            try:
                ds = DOMAIN.load_dataset_for_spec(spec)
                examples = build_domain_examples(ds, spec, args.max_examples_per_dataset, rng)
            except Exception as exc:
                print(f"  SKIP {exc!r}")
                dataset_counts[spec.tag] = {"error": repr(exc)}
                continue
            print(f"  Built {len(examples)} harder candidate examples")
            used = 0
            for idx, ex in enumerate(examples, start=1):
                texts = [ex.correct] + list(ex.wrongs[: max(1, args.num_candidates - 1)])
                if len(texts) < 2:
                    continue
                order = list(range(len(texts)))
                rng.shuffle(order)
                correct_idx = order.index(0)
                encoded = [encode_text(texts[i]) for i in order]
                record = build_record(
                    evaluator,
                    specs,
                    spec_by_name,
                    [item[0] for item in encoded],
                    [item[1] for item in encoded],
                    correct_idx,
                    {
                        "dataset": spec.tag,
                        "domain": spec.domain,
                        "source_index": int(ex.source_index),
                    },
                    args.branch_score_chunk_size,
                )
                records.append(record)
                used += 1
                if used % args.report_every == 0 or used == len(examples):
                    preds = make_predictions([r for r in records if r["dataset"] == spec.tag])
                    summary = summarize_predictions([r for r in records if r["dataset"] == spec.tag], preds)
                    print(
                        f"  [{spec.tag} {used}/{len(examples)}] "
                        f"sum_deb={summary['sum_debiased']['accuracy']:.3f} "
                        f"vote_deb={summary['vote_debiased']['accuracy']:.3f} "
                        f"arch45={summary.get('arch_24_36_45', {}).get('accuracy', 0.0):.3f}"
                    )
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            dataset_counts[spec.tag] = {"examples": used, "domain": spec.domain, "kind": spec.kind}

        if args.include_math:
            print("\nLoading hard_math")
            ds = load_math_dataset()
            filtered = [i for i in range(len(ds)) if MATH.get_level(ds[i]) >= args.math_min_level]
            rng.shuffle(filtered)
            used = 0
            skipped = 0
            for ds_idx in filtered:
                if used >= args.math_examples:
                    break
                ex = ds[ds_idx]
                problem = ex.get("problem", ex.get("Problem", ""))
                solution = ex.get("solution", ex.get("Solution", ""))
                if not problem or not solution:
                    skipped += 1
                    continue
                candidate_set = MATH.build_candidate_set(problem, solution, rng, args.num_candidates)
                if candidate_set is None:
                    skipped += 1
                    continue
                texts, correct_idx = candidate_set
                encoded = [encode_text(text) for text in texts]
                record = build_record(
                    evaluator,
                    specs,
                    spec_by_name,
                    [item[0] for item in encoded],
                    [item[1] for item in encoded],
                    correct_idx,
                    {"dataset": "hard_math", "domain": "math", "source_index": int(ds_idx)},
                    args.branch_score_chunk_size,
                )
                records.append(record)
                used += 1
                if used % args.report_every == 0 or used == args.math_examples:
                    math_records = [r for r in records if r["dataset"] == "hard_math"]
                    preds = make_predictions(math_records)
                    summary = summarize_predictions(math_records, preds)
                    print(
                        f"  [hard_math {used}/{args.math_examples}] "
                        f"sum_deb={summary['sum_debiased']['accuracy']:.3f} "
                        f"vote_deb={summary['vote_debiased']['accuracy']:.3f} "
                        f"arch45={summary.get('arch_24_36_45', {}).get('accuracy', 0.0):.3f}"
                    )
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            dataset_counts["hard_math"] = {"examples": used, "skipped": skipped, "domain": "math", "kind": "math"}
    finally:
        for handle in handles:
            handle.remove()

    cal_records, eval_records = split_records(records, args.calibration_fraction)
    global_weights = train_simple_weights(cal_records, tap_names)
    by_dataset_weights = {}
    for dataset in sorted({rec["dataset"] for rec in records}):
        ds_cal = [rec for rec in cal_records if rec["dataset"] == dataset]
        by_dataset_weights[dataset] = train_simple_weights(ds_cal, tap_names)

    overall_summary = summarize_predictions(records, make_predictions(records))
    eval_summary = summarize_predictions(eval_records, make_predictions(eval_records))
    eval_global_cal = summarize_predictions(eval_records, make_predictions(eval_records, global_weights))

    dataset_summaries = {}
    for dataset in sorted({rec["dataset"] for rec in records}):
        rows = [rec for rec in records if rec["dataset"] == dataset]
        ds_cal, ds_eval = split_records(rows, args.calibration_fraction)
        ds_weights = train_simple_weights(ds_cal, tap_names)
        dataset_summaries[dataset] = {
            "all": summarize_predictions(rows, make_predictions(rows)),
            "eval": summarize_predictions(ds_eval, make_predictions(ds_eval)),
            "eval_calibrated": summarize_predictions(ds_eval, make_predictions(ds_eval, ds_weights)),
            "weights": ds_weights,
        }

    result = {
        "model_path": args.model_path,
        "checkpoint": args.checkpoint,
        "layers": layers,
        "tap_names": tap_names,
        "max_examples_per_dataset": args.max_examples_per_dataset,
        "include_math": args.include_math,
        "math_examples": args.math_examples,
        "num_candidates": args.num_candidates,
        "calibration_fraction": args.calibration_fraction,
        "seed": args.seed,
        "dataset_counts": dataset_counts,
        "global_weights": global_weights,
        "dataset_weights": by_dataset_weights,
        "overall_summary": overall_summary,
        "eval_summary": eval_summary,
        "eval_global_calibrated_summary": eval_global_cal,
        "dataset_summaries": dataset_summaries,
        "records": records,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    def print_rows(title: str, summary: Dict[str, dict], keys: Sequence[str]) -> None:
        print(f"\n=== {title} ===")
        print(f"{'method':>32} | {'acc':>6} | {'n':>5}")
        for key in keys:
            if key in summary:
                row = summary[key]
                print(f"  {key:>30} | {row['accuracy']:.3f} | {int(row['n']):5d}")

    keys = [
        "tap_debiased:boundary_natural",
        "tap_debiased:layer_24_natural_raw",
        "tap_debiased:layer_36_natural_raw",
        "tap_debiased:layer_45_natural_raw",
        "tap_debiased:layer_46_natural_raw",
        "tap_debiased:layer_47_natural_raw",
        "vote_debiased",
        "sum_raw",
        "sum_debiased",
        "arch_24_36_45",
        "arch_24_36_46",
        "calibrated_debiased",
    ]
    print_rows("overall all examples", overall_summary, keys)
    print_rows("held-out eval split", eval_summary, keys)
    print_rows("held-out global calibrated", eval_global_cal, keys)

    print("\n=== per-dataset held-out calibrated ===")
    print(f"{'dataset':>20} | {'sum_deb':>7} | {'arch45':>7} | {'cal':>7} | {'n':>5}")
    for dataset, row in dataset_summaries.items():
        eval_cal = row["eval_calibrated"]
        eval_plain = row["eval"]
        n = int(eval_plain.get("sum_debiased", {}).get("n", 0))
        print(
            f"  {dataset:>18} | "
            f"{eval_plain.get('sum_debiased', {}).get('accuracy', 0.0):.3f} | "
            f"{eval_plain.get('arch_24_36_45', {}).get('accuracy', 0.0):.3f} | "
            f"{eval_cal.get('calibrated_debiased', {}).get('accuracy', 0.0):.3f} | {n:5d}"
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
