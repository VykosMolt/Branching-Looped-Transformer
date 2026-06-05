"""Evaluator layer-tap runs on cached coding, reasoning, logic, and thinking sets.

This is the non-math companion to probe_layer_tap_math_actual.py. It evaluates
selected early/intervention and late/readout transformer taps on compact cached
benchmarks:

* HumanEval and MBPP code candidates made by mutating reference solutions.
* ARC multiple-choice candidates.
* BBH symbolic/reasoning candidates.
* StrategyQA yes/no candidates.

The probe is intentionally relational: every item becomes one correct candidate
and one or more wrong candidates, then the frozen pairwise evaluator is tested
as both a correct-first pair classifier and an O(N^2) branch selector.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
NUM_DECODER_LAYERS = 48
DEFAULT_LAYERS = (20, 24, 36, 40, 45, 47)

RETURN_RE = re.compile(r"(^[ \t]*return\s+)(.+?)([ \t]*(?:#.*)?$)", re.MULTILINE)
NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")


@dataclass(frozen=True)
class DatasetSpec:
    tag: str
    domain: str
    path: str
    config: Optional[str]
    split: str
    kind: str


@dataclass(frozen=True)
class CandidateExample:
    dataset: str
    domain: str
    source_index: int
    correct: str
    wrongs: Tuple[str, ...]


DATASET_SPECS = (
    DatasetSpec("humaneval", "coding", "openai/openai_humaneval", None, "test", "humaneval"),
    DatasetSpec("mbpp_sanitized", "coding", "google-research-datasets/mbpp", "sanitized", "test", "mbpp"),
    DatasetSpec("arc_challenge", "reasoning", "allenai/ai2_arc", "ARC-Challenge", "validation", "arc"),
    DatasetSpec("arc_easy", "reasoning", "allenai/ai2_arc", "ARC-Easy", "validation", "arc"),
    DatasetSpec("bbh_boolean", "reasoning", "lukaemon/bbh", "boolean_expressions", "test", "bbh"),
    DatasetSpec("bbh_multistep", "reasoning", "lukaemon/bbh", "multistep_arithmetic_two", "test", "bbh"),
    DatasetSpec("bbh_logical_five", "logic", "lukaemon/bbh", "logical_deduction_five_objects", "test", "bbh"),
    DatasetSpec("bbh_tracking", "logic", "lukaemon/bbh", "tracking_shuffled_objects_five_objects", "test", "bbh"),
    DatasetSpec("strategyqa", "thinking", "ChilleD/StrategyQA", None, "test", "strategyqa"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument("--datasets", default=",".join(spec.tag for spec in DATASET_SPECS))
    p.add_argument("--max-examples-per-dataset", type=int, default=120)
    p.add_argument("--num-candidates", type=int, default=4)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--layers", default=",".join(str(x) for x in DEFAULT_LAYERS))
    p.add_argument("--early-exit-threshold", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pair-score-chunk-size", type=int, default=32)
    p.add_argument("--branch-score-chunk-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=52)
    p.add_argument("--report-every", type=int, default=20)
    p.add_argument("--output-json", default="artifacts/reports/evaluator/probe_layer_tap_cached_domains.json")
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


def normalize_text(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return ""
    return str(value).strip()


def add_unique(items: List[str], value: str, forbidden: Sequence[str], limit: int) -> None:
    cleaned = value.strip()
    if not cleaned:
        return
    if cleaned in forbidden or cleaned in items:
        return
    if len(items) < limit:
        items.append(cleaned)


def replace_first_return(code: str, replacement: str) -> Optional[str]:
    match = RETURN_RE.search(code)
    if not match:
        return None
    start, end = match.span(2)
    return code[:start] + replacement + code[end:]


def perturb_number(match: re.Match[str], rng: random.Random) -> str:
    text = match.group(0)
    try:
        if "." in text:
            value = float(text)
            value = value + rng.choice([-2.0, -1.0, 1.0, 2.0])
            return f"{value:.3f}".rstrip("0").rstrip(".")
        value = int(text)
        delta = rng.choice([-3, -2, -1, 1, 2, 3])
        return str(value + delta)
    except ValueError:
        return text


def code_mutants(code: str, rng: random.Random, limit: int) -> List[str]:
    mutants: List[str] = []
    forbidden = (code,)

    for replacement in ("None", "0", "1", "False", "True", "[]", "''"):
        mutated = replace_first_return(code, replacement)
        if mutated is not None:
            add_unique(mutants, mutated, forbidden, limit)

    replacements = (
        (" + ", " - "),
        (" - ", " + "),
        (" * ", " + "),
        (" // ", " / "),
        (" / ", " // "),
        (" == ", " != "),
        (" != ", " == "),
        (" <= ", " < "),
        (" >= ", " > "),
        (" < ", " <= "),
        (" > ", " >= "),
        (" and ", " or "),
        (" or ", " and "),
    )
    for old, new in replacements:
        if old in code:
            add_unique(mutants, code.replace(old, new, 1), forbidden, limit)
        if len(mutants) >= limit:
            return mutants

    number_match = NUMBER_RE.search(code)
    if number_match:
        add_unique(mutants, NUMBER_RE.sub(lambda m: perturb_number(m, rng), code, count=1), forbidden, limit)

    if len(mutants) < limit:
        add_unique(mutants, code + "\n# incorrect fallback mutation\n", forbidden, limit)
    return mutants[:limit]


def format_code_candidate(prompt: str, solution: str, tests: Optional[Sequence[str]] = None) -> str:
    tests_text = ""
    if tests:
        tests_text = "\n\nVisible tests:\n" + "\n".join(str(test) for test in tests[:4])
    return (
        f"Programming task:\n{prompt.strip()}{tests_text}\n\n"
        f"Candidate solution:\n```python\n{solution.rstrip()}\n```"
    )


def build_humaneval_examples(ds, spec: DatasetSpec, max_examples: int, rng: random.Random) -> List[CandidateExample]:
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    rows: List[CandidateExample] = []
    for idx in indices:
        if len(rows) >= max_examples:
            break
        ex = ds[idx]
        prompt = normalize_text(ex.get("prompt"))
        solution = normalize_text(ex.get("canonical_solution"))
        if not prompt or not solution:
            continue
        mutants = code_mutants(solution, rng, limit=8)
        if not mutants:
            continue
        correct = format_code_candidate(prompt, prompt + solution)
        wrongs = tuple(format_code_candidate(prompt, prompt + mutant) for mutant in mutants)
        rows.append(CandidateExample(spec.tag, spec.domain, idx, correct, wrongs))
    return rows


def build_mbpp_examples(ds, spec: DatasetSpec, max_examples: int, rng: random.Random) -> List[CandidateExample]:
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    rows: List[CandidateExample] = []
    for idx in indices:
        if len(rows) >= max_examples:
            break
        ex = ds[idx]
        prompt = normalize_text(ex.get("prompt"))
        code = normalize_text(ex.get("code"))
        tests = ex.get("test_list") or []
        if not prompt or not code:
            continue
        mutants = code_mutants(code, rng, limit=8)
        if not mutants:
            continue
        correct = format_code_candidate(prompt, code, tests)
        wrongs = tuple(format_code_candidate(prompt, mutant, tests) for mutant in mutants)
        rows.append(CandidateExample(spec.tag, spec.domain, idx, correct, wrongs))
    return rows


def arc_choice_rows(ex: dict) -> Tuple[List[str], List[str]]:
    choices = ex.get("choices", {})
    if isinstance(choices, dict):
        labels = [str(x) for x in choices.get("label", [])]
        texts = [str(x) for x in choices.get("text", [])]
    else:
        labels = [str(i + 1) for i in range(len(choices))]
        texts = [str(x) for x in choices]
    return labels, texts


def answer_index(answer_key: object, labels: Sequence[str]) -> Optional[int]:
    key = normalize_text(answer_key)
    if key in labels:
        return labels.index(key)
    if key.isdigit():
        numeric = int(key)
        if 1 <= numeric <= len(labels):
            return numeric - 1
    return None


def format_mc_candidate(question: str, labels: Sequence[str], texts: Sequence[str], idx: int) -> str:
    options = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
    return (
        f"Question:\n{question.strip()}\n\n"
        f"Options:\n{options}\n\n"
        f"Candidate answer: {labels[idx]}. {texts[idx]}"
    )


def build_arc_examples(ds, spec: DatasetSpec, max_examples: int, rng: random.Random) -> List[CandidateExample]:
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    rows: List[CandidateExample] = []
    for idx in indices:
        if len(rows) >= max_examples:
            break
        ex = ds[idx]
        question = normalize_text(ex.get("question"))
        labels, texts = arc_choice_rows(ex)
        correct_idx = answer_index(ex.get("answerKey"), labels)
        if not question or correct_idx is None or correct_idx >= len(texts):
            continue
        correct = format_mc_candidate(question, labels, texts, correct_idx)
        wrongs = tuple(
            format_mc_candidate(question, labels, texts, i)
            for i in range(len(texts))
            if i != correct_idx
        )
        if wrongs:
            rows.append(CandidateExample(spec.tag, spec.domain, idx, correct, wrongs))
    return rows


def perturb_answer_text(answer: str, pool: Sequence[str], rng: random.Random, limit: int) -> List[str]:
    wrongs: List[str] = []
    lower = answer.lower()
    if lower in {"true", "false"}:
        add_unique(wrongs, "False" if lower == "true" else "True", (answer,), limit)
    if lower in {"yes", "no"}:
        add_unique(wrongs, "no" if lower == "yes" else "yes", (answer,), limit)

    number_match = NUMBER_RE.search(answer)
    if number_match:
        add_unique(wrongs, NUMBER_RE.sub(lambda m: perturb_number(m, rng), answer, count=1), (answer,), limit)

    shuffled_pool = list(pool)
    rng.shuffle(shuffled_pool)
    for item in shuffled_pool:
        add_unique(wrongs, normalize_text(item), (answer,), limit)
        if len(wrongs) >= limit:
            break
    return wrongs[:limit]


def format_short_answer_candidate(task: str, answer: str) -> str:
    return f"Task:\n{task.strip()}\n\nCandidate answer: {answer.strip()}"


def build_bbh_examples(ds, spec: DatasetSpec, max_examples: int, rng: random.Random) -> List[CandidateExample]:
    target_pool = sorted({normalize_text(ds[i].get("target")) for i in range(len(ds)) if normalize_text(ds[i].get("target"))})
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    rows: List[CandidateExample] = []
    for idx in indices:
        if len(rows) >= max_examples:
            break
        ex = ds[idx]
        task = normalize_text(ex.get("input"))
        target = normalize_text(ex.get("target"))
        if not task or not target:
            continue
        wrong_answers = perturb_answer_text(target, target_pool, rng, limit=8)
        if not wrong_answers:
            continue
        correct = format_short_answer_candidate(task, target)
        wrongs = tuple(format_short_answer_candidate(task, wrong) for wrong in wrong_answers)
        rows.append(CandidateExample(spec.tag, spec.domain, idx, correct, wrongs))
    return rows


def build_strategyqa_examples(ds, spec: DatasetSpec, max_examples: int, rng: random.Random) -> List[CandidateExample]:
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    rows: List[CandidateExample] = []
    for idx in indices:
        if len(rows) >= max_examples:
            break
        ex = ds[idx]
        question = normalize_text(ex.get("question"))
        answer = normalize_text(ex.get("answer")).lower()
        if not question or answer not in {"true", "false", "yes", "no"}:
            continue
        correct_answer = "yes" if answer in {"true", "yes"} else "no"
        wrong_answer = "no" if correct_answer == "yes" else "yes"
        correct = format_short_answer_candidate(question, correct_answer)
        wrongs = (format_short_answer_candidate(question, wrong_answer),)
        rows.append(CandidateExample(spec.tag, spec.domain, idx, correct, wrongs))
    return rows


def load_dataset_for_spec(spec: DatasetSpec):
    if spec.config is None:
        return load_dataset(spec.path, split=spec.split)
    return load_dataset(spec.path, spec.config, split=spec.split)


def build_examples(ds, spec: DatasetSpec, max_examples: int, rng: random.Random) -> List[CandidateExample]:
    if spec.kind == "humaneval":
        return build_humaneval_examples(ds, spec, max_examples, rng)
    if spec.kind == "mbpp":
        return build_mbpp_examples(ds, spec, max_examples, rng)
    if spec.kind == "arc":
        return build_arc_examples(ds, spec, max_examples, rng)
    if spec.kind == "bbh":
        return build_bbh_examples(ds, spec, max_examples, rng)
    if spec.kind == "strategyqa":
        return build_strategyqa_examples(ds, spec, max_examples, rng)
    raise ValueError(f"unknown dataset kind: {spec.kind}")


def build_specs(layers: List[int]) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, object]]]:
    specs: List[Dict[str, object]] = []

    def add(name: str, **meta: object) -> None:
        item = {"config": name}
        item.update(meta)
        specs.append(item)

    add("boundary_natural", family="boundary_natural", layer=None, loop=None, normed=True)
    add("boundary_loop_4_replicated", family="boundary_loop", layer=None, loop=3, normed=True)
    for layer in layers:
        add(f"layer_{layer:02d}_natural_raw", family="layer_natural", layer=layer, loop=None, normed=False)
        add(f"layer_{layer:02d}_natural_normed", family="layer_natural", layer=layer, loop=None, normed=True)
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
            "avg_candidate_count": float(np.mean(row["candidate_counts"])),
        }
    return summary


def init_branch_stats(specs: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    return {
        str(spec["config"]): {
            "n": 0,
            "raw_correct": 0,
            "debiased_correct": 0,
            "agreement": 0,
            "raw_margin": [],
            "debiased_margin": [],
            "candidate_counts": [],
        }
        for spec in specs
    }


def run_branch_tournament(
    evaluator: PairwiseEvaluator,
    specs: List[Dict[str, object]],
    spec_by_name: Dict[str, Dict[str, object]],
    candidate_data: List[Dict[str, object]],
    candidate_masks: List[torch.Tensor],
    correct_idx: int,
    stats: Dict[str, Dict[str, object]],
    chunk_size: int,
) -> None:
    n_candidates = len(candidate_data)
    config_names = [str(spec["config"]) for spec in specs]
    items = [
        (config, i, j)
        for config in config_names
        for i in range(n_candidates)
        for j in range(n_candidates)
        if i != j
    ]
    score_map = score_branch_items(
        evaluator,
        items,
        spec_by_name,
        candidate_data,
        candidate_masks,
        chunk_size,
    )

    for config in config_names:
        matrix = np.full((n_candidates, n_candidates), np.nan, dtype=np.float64)
        for i in range(n_candidates):
            for j in range(n_candidates):
                if i != j:
                    matrix[i, j] = score_map[(config, i, j)]
        raw_scores = np.nansum(matrix, axis=1)
        debiased_scores = np.zeros(n_candidates, dtype=np.float64)
        for i in range(n_candidates):
            for j in range(n_candidates):
                if i != j:
                    debiased_scores[i] += 0.5 * (matrix[i, j] - matrix[j, i])

        raw_winner = int(np.argmax(raw_scores))
        debiased_winner = int(np.argmax(debiased_scores))
        raw_sorted = np.sort(raw_scores)
        deb_sorted = np.sort(debiased_scores)
        row = stats[config]
        row["n"] = int(row["n"]) + 1
        row["raw_correct"] = int(row["raw_correct"]) + int(raw_winner == correct_idx)
        row["debiased_correct"] = int(row["debiased_correct"]) + int(debiased_winner == correct_idx)
        row["agreement"] = int(row["agreement"]) + int(raw_winner == debiased_winner)
        row["raw_margin"].append(float(raw_sorted[-1] - raw_sorted[-2]))
        row["debiased_margin"].append(float(deb_sorted[-1] - deb_sorted[-2]))
        row["candidate_counts"].append(int(n_candidates))


def print_dataset_summary(dataset: str, pair_summary: Dict[str, Dict[str, float | int]], branch_summary: Dict[str, Dict[str, float | int]]) -> None:
    selected = [
        "boundary_natural",
        "boundary_loop_4_replicated",
        "layer_20_natural_raw",
        "layer_24_natural_raw",
        "layer_36_natural_raw",
        "layer_40_natural_raw",
        "layer_45_natural_raw",
        "layer_47_natural_raw",
    ]
    print(f"\n=== {dataset} selected configs ===")
    print(f"{'config':>30} | {'canon':>5} | {'center':>6} | {'raw':>5} | {'deb':>5}")
    for key in selected:
        if key not in pair_summary or key not in branch_summary:
            continue
        p = pair_summary[key]
        b = branch_summary[key]
        print(
            f"  {key:>28} | {p['canonical_acc']:.3f} | {p['centered_acc']:.3f} | "
            f"{b['raw_selection_acc']:.3f} | {b['debiased_selection_acc']:.3f}"
        )


def aggregate_pair_scores(dataset_results: Dict[str, dict], specs: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float | int]]:
    aggregate = {}
    for spec in specs:
        name = str(spec["config"])
        normal: List[float] = []
        flipped: List[float] = []
        for row in dataset_results.values():
            normal.extend(row["pair_raw_scores"].get(name, []))
            flipped.extend(row["pair_raw_scores_flipped"].get(name, []))
        aggregate[name] = metric_block(normal, flipped)
    return aggregate


def aggregate_branch_stats(dataset_results: Dict[str, dict], specs: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float | int]]:
    merged = init_branch_stats(specs)
    for row in dataset_results.values():
        raw_stats = row["branch_raw_stats"]
        for config, stats in raw_stats.items():
            target = merged[config]
            target["n"] = int(target["n"]) + int(stats["n"])
            target["raw_correct"] = int(target["raw_correct"]) + int(stats["raw_correct"])
            target["debiased_correct"] = int(target["debiased_correct"]) + int(stats["debiased_correct"])
            target["agreement"] = int(target["agreement"]) + int(stats["agreement"])
            target["raw_margin"].extend(stats["raw_margin"])
            target["debiased_margin"].extend(stats["debiased_margin"])
            target["candidate_counts"].extend(stats["candidate_counts"])
    return summarize_branch(merged)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    for layer in layers:
        assert 0 <= layer < NUM_DECODER_LAYERS, f"layer {layer} out of range"

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    wanted = {name.strip() for name in args.datasets.split(",") if name.strip()}
    selected_specs = [spec for spec in DATASET_SPECS if spec.tag in wanted]
    missing = sorted(wanted.difference(spec.tag for spec in DATASET_SPECS))
    if missing:
        raise ValueError(f"unknown dataset tags: {missing}")

    device = torch.device(args.device)
    print(f"Model: {args.model_path}")
    print(f"Evaluator: {args.checkpoint}")
    print(f"Datasets: {[spec.tag for spec in selected_specs]}")
    print(f"Layers: {layers}")
    print(f"Max examples/dataset: {args.max_examples_per_dataset}, candidates: {args.num_candidates}, device={device}\n")

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
        layers_normed = {layer: [norm_fn(h) for h in layers_raw[layer]] for layer in layers}
        return {"boundary": boundary, "layers_raw": layers_raw, "layers_normed": layers_normed}, tokens["attention_mask"]

    dataset_results: Dict[str, dict] = {}

    try:
        for spec in selected_specs:
            print(f"\nLoading {spec.tag}: {spec.path} {spec.config or ''} {spec.split}")
            try:
                ds = load_dataset_for_spec(spec)
            except Exception as exc:
                print(f"  SKIP load failed: {exc!r}")
                dataset_results[spec.tag] = {"spec": asdict(spec), "load_error": repr(exc)}
                continue

            examples = build_examples(ds, spec, args.max_examples_per_dataset, rng)
            print(f"  Built {len(examples)} candidate examples from {len(ds)} rows")
            if not examples:
                dataset_results[spec.tag] = {"spec": asdict(spec), "build_error": "no candidate examples"}
                continue

            pair_scores: Dict[str, List[float]] = {str(item["config"]): [] for item in specs}
            pair_scores_flipped: Dict[str, List[float]] = {str(item["config"]): [] for item in specs}
            branch_stats = init_branch_stats(specs)
            skipped = 0

            for ex_idx, ex in enumerate(examples, start=1):
                texts = [ex.correct] + list(ex.wrongs[: max(1, args.num_candidates - 1)])
                if len(texts) < 2:
                    skipped += 1
                    continue
                encoded = [encode_text(text) for text in texts]
                candidate_data = [item[0] for item in encoded]
                candidate_masks = [item[1] for item in encoded]

                normal, flipped = score_pair_configs(
                    evaluator,
                    specs,
                    candidate_data[0],
                    candidate_masks[0],
                    candidate_data[1],
                    candidate_masks[1],
                    args.pair_score_chunk_size,
                )
                for name in pair_scores:
                    pair_scores[name].append(normal[name])
                    pair_scores_flipped[name].append(flipped[name])

                order = list(range(len(texts)))
                rng.shuffle(order)
                correct_idx = order.index(0)
                run_branch_tournament(
                    evaluator,
                    specs,
                    spec_by_name,
                    [candidate_data[i] for i in order],
                    [candidate_masks[i] for i in order],
                    correct_idx,
                    branch_stats,
                    args.branch_score_chunk_size,
                )

                if ex_idx % args.report_every == 0 or ex_idx == len(examples):
                    bsum = summarize_branch(branch_stats)
                    pair_preview = metric_block(pair_scores["boundary_natural"], pair_scores_flipped["boundary_natural"])
                    layer24 = metric_block(pair_scores["layer_24_natural_raw"], pair_scores_flipped["layer_24_natural_raw"])
                    layer45 = metric_block(pair_scores["layer_45_natural_raw"], pair_scores_flipped["layer_45_natural_raw"])
                    print(
                        f"  [{spec.tag} {ex_idx}/{len(examples)}] "
                        f"boundary canon={pair_preview['canonical_acc']:.3f} branch={bsum['boundary_natural']['raw_selection_acc']:.3f}  "
                        f"l24 center={layer24['centered_acc']:.3f}  l45 branch={bsum['layer_45_natural_raw']['raw_selection_acc']:.3f}"
                    )
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            pair_summary = {
                name: metric_block(pair_scores[name], pair_scores_flipped[name])
                for name in pair_scores
            }
            branch_summary = summarize_branch(branch_stats)
            print_dataset_summary(spec.tag, pair_summary, branch_summary)
            dataset_results[spec.tag] = {
                "spec": asdict(spec),
                "examples_evaluated": len(examples) - skipped,
                "examples_skipped": skipped,
                "pair_per_config": pair_summary,
                "branch_per_config": branch_summary,
                "pair_raw_scores": {k: [float(x) for x in v] for k, v in pair_scores.items()},
                "pair_raw_scores_flipped": {k: [float(x) for x in v] for k, v in pair_scores_flipped.items()},
                "branch_raw_stats": branch_stats,
            }
    finally:
        for handle in handles:
            handle.remove()

    aggregate_pair = aggregate_pair_scores(
        {k: v for k, v in dataset_results.items() if "pair_raw_scores" in v},
        specs,
    )
    aggregate_branch = aggregate_branch_stats(
        {k: v for k, v in dataset_results.items() if "branch_raw_stats" in v},
        specs,
    )

    result = {
        "model_path": args.model_path,
        "checkpoint": args.checkpoint,
        "layers": layers,
        "max_length": args.max_length,
        "max_examples_per_dataset": args.max_examples_per_dataset,
        "num_candidates_requested": args.num_candidates,
        "seed": args.seed,
        "config_metadata": spec_by_name,
        "datasets": dataset_results,
        "aggregate_pair_per_config": aggregate_pair,
        "aggregate_branch_per_config": aggregate_branch,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print_dataset_summary("AGGREGATE", aggregate_pair, aggregate_branch)
    top_branch = sorted(
        aggregate_branch.items(),
        key=lambda kv: (-float(kv[1]["raw_selection_acc"]), -float(kv[1]["debiased_selection_acc"])),
    )[:10]
    print("\n=== aggregate top branch selectors ===")
    print(f"{'config':>30} | {'raw':>5} | {'deb':>5} | {'n':>5} | {'avgN':>5}")
    for name, row in top_branch:
        print(
            f"  {name:>28} | {row['raw_selection_acc']:.3f} | {row['debiased_selection_acc']:.3f} | "
            f"{int(row['n']):5d} | {row['avg_candidate_count']:.2f}"
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
