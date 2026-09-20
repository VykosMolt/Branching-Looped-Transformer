"""Shared utilities for math BG geometry and tap probes.

The scripts in this directory are lightweight manual probes. They keep all
new outputs under opi/taps/probes and intentionally avoid modifying
published evaluator code or checkpoints.
"""
from __future__ import annotations

import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[4]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "artifacts" / "hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "artifacts" / "hf_cache" / "datasets"))

from datasets import concatenate_datasets, load_dataset

if str(PROJECT_ROOT / "shared/src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "shared/src"))

DEFAULT_RLTT_PATH = "shared/models/ouro_rltt_local"
DEFAULT_TOKENIZER_PATH = "shared/models/ouro_rltt_local"
DEFAULT_OUTPUT_DIR = "opi/taps/probes"
TAP_LAYERS = (24, 36, 47)
NUM_LOOPS = 4
HIDDEN_DIM = 2048
LAYER_TO_CAPTURE_INDEX = {24: 23, 36: 35}
LAYER_TO_POS = {layer: idx for idx, layer in enumerate(TAP_LAYERS)}
MATH_CONFIGS = (
    "24_L1",
    "24_L4",
    "24_mean",
    "36_L1",
    "36_L4",
    "36_mean",
    "47_L4",
    "47_mean",
    "47_concat_L1_L4",
    "47_concat_all_loops",
)

NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
FRAC_RE = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
GSM8K_FINAL_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")


@dataclass
class MathProblem:
    source: str
    dataset_index: int
    question: str
    gold_solution: str
    gold_answer: str
    level: Optional[int] = None


class PooledTapCapture:
    """Forward hooks for post-block 24/36 and boundary 47 pooled features."""

    def __init__(self) -> None:
        self.inter: Dict[int, List[torch.Tensor]] = {23: [], 35: []}
        self.boundary: List[torch.Tensor] = []

    def clear(self) -> None:
        self.inter = {23: [], 35: []}
        self.boundary = []

    def make_inter_hook(self, idx: int):
        def _hook(_module, _inp, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            self.inter[idx].append(tensor.detach())
        return _hook

    def boundary_hook(self, _module, _inp, output):
        self.boundary = [h.detach() for h in output[1]]


class AntisymLinearHead(nn.Module):
    """Pairwise linear comparator over normalized feature differences."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.linear = nn.Linear(dim, 1, bias=False)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(left - right)).squeeze(-1)


class AntisymLinearNoNorm(nn.Module):
    """Transitive pairwise linear comparator: score(a,b) = u(a) - u(b)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, 1, bias=False)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.linear(left - right).squeeze(-1)


def resolve_local(path: str | Path) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    candidate = PROJECT_ROOT / p
    if candidate.exists():
        return str(candidate)
    return str(path)


def output_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def clean_latex_answer(text: str) -> str:
    s = str(text).strip()
    s = s.replace("$", "")
    s = s.replace("\\,", "")
    s = s.replace("\\!", "")
    s = s.replace("\\left", "")
    s = s.replace("\\right", "")
    s = s.replace("\\text", "")
    s = s.strip("{}[]() \n\t.:")
    s = FRAC_RE.sub(lambda m: f"({m.group(1)})/({m.group(2)})", s)
    s = s.replace(",", "")
    s = re.sub(r"\s+", "", s)
    return s


def parse_number(text: str) -> Optional[Fraction]:
    s = clean_latex_answer(text)
    if not s:
        return None
    if s.endswith("%"):
        base = parse_number(s[:-1])
        return base / 100 if base is not None else None
    try:
        return Fraction(s)
    except Exception:
        pass
    try:
        return Fraction(float(s)).limit_denominator(1_000_000)
    except Exception:
        return None


def extract_numbers(text: str) -> List[Fraction]:
    numbers: List[Fraction] = []
    for match in NUMBER_RE.findall(str(text)):
        value = parse_number(match)
        if value is not None:
            numbers.append(value)
    return numbers


def answers_equal(pred: str, gold: str) -> bool:
    p_num = parse_number(pred)
    g_num = parse_number(gold)
    if p_num is not None and g_num is not None:
        return p_num == g_num
    return clean_latex_answer(pred).lower() == clean_latex_answer(gold).lower()


def extract_gold_answer(source: str, solution: str) -> Optional[str]:
    if source == "gsm8k":
        match = GSM8K_FINAL_RE.search(solution)
        if match:
            return match.group(1)
    boxed = BOXED_RE.findall(solution)
    if boxed:
        return boxed[-1]
    nums = NUMBER_RE.findall(solution)
    if nums:
        return nums[-1]
    return None


def extract_answer(text: str) -> Optional[str]:
    boxed = BOXED_RE.findall(text)
    if boxed:
        return boxed[-1]
    for pat in (
        r"####\s*([^\n]+)",
        r"final answer\s*(?:is|:)\s*([^\n]+)",
        r"answer\s*(?:is|:)\s*([^\n]+)",
    ):
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        if matches:
            candidate = matches[-1].strip()
            nums = NUMBER_RE.findall(candidate)
            return nums[-1] if nums else candidate
    nums = NUMBER_RE.findall(text)
    return nums[-1] if nums else None


def perturb_answer(answer: str, rng: random.Random) -> Optional[str]:
    value = parse_number(answer)
    if value is None:
        return None
    shifts = [Fraction(1), Fraction(-1), Fraction(2), Fraction(-2)]
    if value != 0:
        shifts.extend([value, -value])
    wrong = value + rng.choice(shifts)
    if wrong == value:
        wrong += 1
    if wrong.denominator == 1:
        return str(wrong.numerator)
    return f"{wrong.numerator}/{wrong.denominator}"


def problem_prompt(problem: MathProblem) -> str:
    return (
        "Solve the math problem. Show the reasoning. End with a line of the "
        "form 'Final answer: <answer>'.\n\n"
        f"Problem: {problem.question}\n\nSolution:"
    )


def candidate_text(problem: MathProblem, completion: str) -> str:
    return f"Problem: {problem.question}\n\nSolution:{completion.strip()}"


def gold_solution_text(problem: MathProblem) -> str:
    if problem.gold_solution.strip():
        return f"Problem: {problem.question}\n\nSolution: {problem.gold_solution}"
    return f"Problem: {problem.question}\n\nFinal answer: {problem.gold_answer}"


def wrong_answer_text(problem: MathProblem, wrong_answer: str) -> str:
    return f"Problem: {problem.question}\n\nFinal answer: {wrong_answer}"


def _load_math_dataset() -> Tuple[object, str]:
    configs = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
    parts = []
    for cfg in configs:
        parts.append(load_dataset("EleutherAI/hendrycks_math", cfg, split="test"))
    return concatenate_datasets(parts), "EleutherAI/hendrycks_math/*"


def _level(ex: Dict[str, object]) -> Optional[int]:
    level = ex.get("level", ex.get("Level"))
    if level is None:
        return None
    if isinstance(level, str):
        match = re.search(r"\d", level)
        return int(match.group()) if match else None
    try:
        return int(level)
    except Exception:
        return None


def load_math_problems(
    source: str,
    max_problems: int,
    seed: int,
    min_math_level: int = 4,
) -> Tuple[List[MathProblem], Dict[str, object]]:
    rng = random.Random(seed)
    problems: List[MathProblem] = []
    meta: Dict[str, object] = {"source_requested": source, "seed": seed}
    if source == "mixed":
        gsm8k_target = max(1, max_problems // 2)
        math_target = max_problems - gsm8k_target
    else:
        gsm8k_target = max_problems
        math_target = max_problems

    if source in ("gsm8k", "mixed"):
        ds = load_dataset("openai/gsm8k", "main", split="test")
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        added = 0
        for idx in indices:
            ex = ds[idx]
            gold = extract_gold_answer("gsm8k", ex["answer"])
            if gold is None:
                continue
            problems.append(MathProblem(
                source="gsm8k",
                dataset_index=idx,
                question=ex["question"],
                gold_solution=ex["answer"],
                gold_answer=gold,
            ))
            added += 1
            if added >= gsm8k_target:
                break
        meta["gsm8k_total"] = len(ds)
        meta["gsm8k_returned"] = added

    if source in ("math", "mixed"):
        ds, ds_name = _load_math_dataset()
        indices = [i for i in range(len(ds)) if (_level(ds[i]) or 5) >= min_math_level]
        rng.shuffle(indices)
        added = 0
        for idx in indices:
            ex = ds[idx]
            solution = ex.get("solution", ex.get("Solution", ""))
            gold = extract_gold_answer("math", solution)
            if gold is None or parse_number(gold) is None:
                continue
            problems.append(MathProblem(
                source="math",
                dataset_index=idx,
                question=ex.get("problem", ex.get("Problem", "")),
                gold_solution=solution,
                gold_answer=gold,
                level=_level(ex),
            ))
            added += 1
            if added >= math_target:
                break
        meta["math_dataset"] = ds_name
        meta["math_total"] = len(ds)
        meta["math_min_level"] = min_math_level
        meta["math_returned"] = added

    rng.shuffle(problems)
    if source == "mixed":
        problems = problems[:max_problems]
    meta["problems_returned"] = len(problems)
    return problems, meta


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    h = hidden.squeeze(0).to(device="cpu", dtype=torch.float32)
    m = mask.squeeze(0).to(device="cpu", dtype=torch.float32).unsqueeze(-1)
    denom = m.sum(dim=0).clamp(min=1.0)
    return (h * m).sum(dim=0) / denom


@torch.no_grad()
def capture_pooled_taps(
    model,
    tokenizer,
    texts: Sequence[str],
    max_length: int,
    device: torch.device,
    report_every: int = 25,
) -> torch.Tensor:
    cap = PooledTapCapture()
    handles = [model.model.register_forward_hook(cap.boundary_hook)]
    for idx in LAYER_TO_CAPTURE_INDEX.values():
        handles.append(model.model.layers[idx].register_forward_hook(
            cap.make_inter_hook(idx)))

    rows: List[torch.Tensor] = []
    start = time.time()
    for i, text in enumerate(texts):
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_length).to(device)
        cap.clear()
        model(**enc, use_cache=False)
        if len(cap.boundary) != NUM_LOOPS:
            raise RuntimeError(f"expected {NUM_LOOPS} boundary states, got {len(cap.boundary)}")
        for idx in LAYER_TO_CAPTURE_INDEX.values():
            if len(cap.inter[idx]) != NUM_LOOPS:
                raise RuntimeError(f"layer hook {idx}: expected {NUM_LOOPS}, got {len(cap.inter[idx])}")

        per_layer: List[torch.Tensor] = []
        for layer in TAP_LAYERS:
            if layer == 47:
                loop_states = [mean_pool(h, enc["attention_mask"]) for h in cap.boundary]
            else:
                hook_idx = LAYER_TO_CAPTURE_INDEX[layer]
                loop_states = [mean_pool(h, enc["attention_mask"]) for h in cap.inter[hook_idx]]
            per_layer.append(torch.stack(loop_states, dim=0))
        rows.append(torch.stack(per_layer, dim=0))

        if report_every > 0 and ((i + 1) % report_every == 0 or i + 1 == len(texts)):
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            print(f"captured {i+1}/{len(texts)} rate={rate:.2f}/s")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for handle in handles:
        handle.remove()
    return torch.stack(rows, dim=0)


def config_vector(pooled: torch.Tensor, config: str) -> torch.Tensor:
    """Return candidate features from pooled [layers=3, loops=4, H]."""
    if config == "24_L1":
        return pooled[LAYER_TO_POS[24], 0]
    if config == "24_L4":
        return pooled[LAYER_TO_POS[24], 3]
    if config == "24_mean":
        return pooled[LAYER_TO_POS[24]].mean(dim=0)
    if config == "36_L1":
        return pooled[LAYER_TO_POS[36], 0]
    if config == "36_L4":
        return pooled[LAYER_TO_POS[36], 3]
    if config == "36_mean":
        return pooled[LAYER_TO_POS[36]].mean(dim=0)
    if config == "47_L4":
        return pooled[LAYER_TO_POS[47], 3]
    if config == "47_mean":
        return pooled[LAYER_TO_POS[47]].mean(dim=0)
    if config == "47_concat_L1_L4":
        return torch.cat([pooled[LAYER_TO_POS[47], 0], pooled[LAYER_TO_POS[47], 3]], dim=-1)
    if config == "47_concat_all_loops":
        return torch.cat([pooled[LAYER_TO_POS[47], i] for i in range(4)], dim=-1)
    raise ValueError(f"unknown config: {config}")


def config_dim(config: str) -> int:
    if config == "47_concat_L1_L4":
        return HIDDEN_DIM * 2
    if config == "47_concat_all_loops":
        return HIDDEN_DIM * 4
    return HIDDEN_DIM


def _score_matrix_list(score_matrices: torch.Tensor | Sequence[torch.Tensor]) -> List[torch.Tensor]:
    if isinstance(score_matrices, torch.Tensor):
        if score_matrices.ndim == 2:
            return [score_matrices.to(torch.float32)]
        if score_matrices.ndim == 3:
            return [score_matrices[i].to(torch.float32) for i in range(score_matrices.shape[0])]
        raise ValueError("score_matrices must be a [K,K] or [N,K,K] tensor")
    return [m.to(torch.float32) for m in score_matrices]


def _label_list(labels: torch.Tensor | Sequence[Sequence[bool] | torch.Tensor]) -> List[torch.Tensor]:
    if isinstance(labels, torch.Tensor):
        if labels.ndim == 1:
            return [labels.to(dtype=torch.bool)]
        if labels.ndim == 2:
            return [labels[i].to(dtype=torch.bool) for i in range(labels.shape[0])]
        raise ValueError("labels must be a [K] or [N,K] tensor")
    out = []
    for row in labels:
        if isinstance(row, torch.Tensor):
            out.append(row.to(dtype=torch.bool))
        else:
            out.append(torch.tensor(list(row), dtype=torch.bool))
    return out


def _paired_scores(
    score_matrices: torch.Tensor | Sequence[torch.Tensor],
    labels: torch.Tensor | Sequence[Sequence[bool] | torch.Tensor],
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[int, int, float]]]:
    matrices = _score_matrix_list(score_matrices)
    label_rows = _label_list(labels)
    if len(matrices) != len(label_rows):
        raise ValueError("score_matrices and labels must have the same tournament count")
    paired: List[Tuple[torch.Tensor, torch.Tensor]] = []
    margins: List[Tuple[int, int, float]] = []
    for mat, lab in zip(matrices, label_rows):
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError("each score matrix must be square")
        if mat.shape[0] != lab.numel():
            raise ValueError("score matrix size must match label count")
        paired.append((mat, lab))
        correct = torch.nonzero(lab, as_tuple=False).flatten().tolist()
        incorrect = torch.nonzero(~lab, as_tuple=False).flatten().tolist()
        for c in correct:
            for r in incorrect:
                margins.append((c, r, float(mat[c, r])))
    return paired, margins


def tournament_top1_accuracy(
    score_matrices: torch.Tensor | Sequence[torch.Tensor],
    labels: torch.Tensor | Sequence[Sequence[bool] | torch.Tensor],
) -> float:
    paired, _ = _paired_scores(score_matrices, labels)
    total = 0
    correct = 0
    for mat, lab in paired:
        totals = mat.sum(dim=1)
        pred = int(torch.argmax(totals).item())
        total += 1
        correct += int(bool(lab[pred]))
    return correct / max(total, 1)


def pairwise_accuracy(
    score_matrices: torch.Tensor | Sequence[torch.Tensor],
    labels: torch.Tensor | Sequence[Sequence[bool] | torch.Tensor],
) -> float:
    _, margins = _paired_scores(score_matrices, labels)
    if not margins:
        return float("nan")
    return sum(1 for _c, _r, margin in margins if margin > 0.0) / len(margins)


def condorcet_winner_rate(
    score_matrices: torch.Tensor | Sequence[torch.Tensor],
    labels: torch.Tensor | Sequence[Sequence[bool] | torch.Tensor],
) -> float:
    paired, _ = _paired_scores(score_matrices, labels)
    total = 0
    winners = 0
    for mat, lab in paired:
        k = mat.shape[0]
        pred = int(torch.argmax(mat.sum(dim=1)).item())
        mask = torch.arange(k, device=mat.device) != pred
        beats_all = bool((mat[pred, mask] > 0).all().item()) if k > 1 else True
        total += 1
        winners += int(bool(lab[pred]) and beats_all)
    return winners / max(total, 1)


def _triplet_has_cycle(mat: torch.Tensor, a: int, b: int, c: int) -> bool:
    return bool(
        (mat[a, b] > 0 and mat[b, c] > 0 and mat[c, a] > 0)
        or (mat[a, c] > 0 and mat[c, b] > 0 and mat[b, a] > 0)
    )


def cycle_rate(
    score_matrices: torch.Tensor | Sequence[torch.Tensor],
    triplets_per_matrix: int = 1000,
    seed: int = 0,
) -> float:
    matrices = _score_matrix_list(score_matrices)
    rng = random.Random(seed)
    sampled = 0
    cyclic = 0
    for mat in matrices:
        k = mat.shape[0]
        if k < 3:
            continue
        for _ in range(max(0, triplets_per_matrix)):
            a, b, c = rng.sample(range(k), 3)
            sampled += 1
            cyclic += int(_triplet_has_cycle(mat, a, b, c))
    return cyclic / max(sampled, 1)


def margin_calibration(
    score_matrices: torch.Tensor | Sequence[torch.Tensor],
    labels: torch.Tensor | Sequence[Sequence[bool] | torch.Tensor],
    n_bins: int = 10,
) -> Dict[str, object]:
    _, margins = _paired_scores(score_matrices, labels)
    if not margins:
        return {"ece": float("nan"), "n_pairs": 0, "bins": []}
    rows = sorted(
        [(abs(margin), margin > 0.0, float(torch.sigmoid(torch.tensor(abs(margin))))) for _c, _r, margin in margins],
        key=lambda x: x[0],
    )
    chunks = [chunk for chunk in np.array_split(np.asarray(rows, dtype=object), min(n_bins, len(rows))) if len(chunk)]
    bins = []
    ece = 0.0
    total = len(rows)
    for idx, chunk in enumerate(chunks):
        abs_margins = np.asarray([float(row[0]) for row in chunk], dtype=np.float64)
        correct = np.asarray([bool(row[1]) for row in chunk], dtype=np.float64)
        conf = np.asarray([float(row[2]) for row in chunk], dtype=np.float64)
        accuracy = float(correct.mean())
        confidence = float(conf.mean())
        contribution = (len(chunk) / total) * abs(accuracy - confidence)
        ece += contribution
        bins.append({
            "bin": idx,
            "count": int(len(chunk)),
            "margin_min": float(abs_margins.min()),
            "margin_max": float(abs_margins.max()),
            "mean_abs_margin": float(abs_margins.mean()),
            "accuracy": accuracy,
            "confidence": confidence,
            "ece_contribution": float(contribution),
        })
    return {"ece": float(ece), "n_pairs": int(total), "bins": bins}


def numeric_near_miss(pred: Fraction, gold: Fraction) -> bool:
    diff = abs(pred - gold)
    if gold == 0:
        return diff <= 1
    if abs(gold) > 1:
        return diff / abs(gold) <= Fraction(1, 10)
    return diff <= 1


def simple_arithmetic_slip(pred: Fraction, gold: Fraction) -> bool:
    if pred == gold:
        return False
    if pred == -gold:
        return True
    if abs(gold) <= 20 and abs(pred - gold) <= 2:
        return True
    if gold == 0:
        return False
    ratio = pred / gold
    return ratio in {Fraction(2), Fraction(1, 2), Fraction(10), Fraction(1, 10)}


def classify_wrong_math_branch(
    prompt: str,
    reference_solution: str,
    gold_answer: str,
    candidate_text: str,
    extracted_answer: Optional[str],
    truncated: bool = False,
) -> Dict[str, object]:
    """Classify an incorrect branch as near-miss or nonsense for diagnostics."""
    if extracted_answer is None:
        return {
            "classification": "nonsense",
            "reason": "unparseable_final_answer",
            "classifier_fallback": True,
            "structurally_math_like": False,
        }
    pred_num = parse_number(extracted_answer)
    gold_num = parse_number(gold_answer)
    if pred_num is None or gold_num is None:
        return {
            "classification": "nonsense",
            "reason": "non_numeric_final_answer",
            "classifier_fallback": True,
            "structurally_math_like": bool(NUMBER_RE.search(candidate_text)),
        }

    math_cues = ("=", "+", "-", "*", "/", "\\frac", "^", "therefore", "so ", "because")
    structurally_math_like = bool(NUMBER_RE.search(candidate_text)) and (
        any(cue in candidate_text.lower() for cue in math_cues)
        or len(extract_numbers(candidate_text)) >= 2
    )
    if truncated and "final answer" not in candidate_text.lower():
        return {
            "classification": "nonsense",
            "reason": "truncated_before_final_answer",
            "classifier_fallback": False,
            "structurally_math_like": structurally_math_like,
        }
    if not structurally_math_like:
        return {
            "classification": "nonsense",
            "reason": "no_recognizable_math_structure",
            "classifier_fallback": False,
            "structurally_math_like": False,
        }

    prompt_numbers = set(extract_numbers(prompt))
    ref_numbers = [
        value for value in extract_numbers(reference_solution)
        if value not in {Fraction(0), Fraction(1), Fraction(-1)} and value not in prompt_numbers
    ]
    candidate_numbers = set(extract_numbers(candidate_text))
    classifier_fallback = len(ref_numbers) == 0
    shared_intermediate = sorted(
        {str(value) for value in ref_numbers if value in candidate_numbers},
        key=str,
    )
    reasons = []
    if numeric_near_miss(pred_num, gold_num):
        reasons.append("numeric_tolerance")
    if shared_intermediate:
        reasons.append(f"shared_intermediate={','.join(shared_intermediate[:5])}")
    if simple_arithmetic_slip(pred_num, gold_num):
        reasons.append("simple_arithmetic_slip")
    if reasons:
        return {
            "classification": "near_miss",
            "reason": ";".join(reasons),
            "classifier_fallback": classifier_fallback,
            "structurally_math_like": structurally_math_like,
            "shared_intermediate_count": len(shared_intermediate),
        }
    return {
        "classification": "nonsense",
        "reason": "wildly_unrelated_or_wrong_setup",
        "classifier_fallback": classifier_fallback,
        "structurally_math_like": structurally_math_like,
        "shared_intermediate_count": len(shared_intermediate),
    }


def tournament_difficulty_classification(attempts: Sequence[Dict[str, object]]) -> Dict[str, object]:
    incorrect = [a for a in attempts if not bool(a.get("is_correct"))]
    if not incorrect:
        return {
            "classification": "trivial",
            "incorrect_branches": 0,
            "near_miss_branches": 0,
            "nonsense_branches": 0,
            "near_miss_fraction": float("nan"),
            "classifier_fallback": False,
        }
    near = sum(1 for a in incorrect if a.get("wrong_classification") == "near_miss")
    nonsense = len(incorrect) - near
    return {
        "classification": "near_miss_dominant" if near >= len(incorrect) / 2 else "nonsense_dominant",
        "incorrect_branches": int(len(incorrect)),
        "near_miss_branches": int(near),
        "nonsense_branches": int(nonsense),
        "near_miss_fraction": near / max(len(incorrect), 1),
        "classifier_fallback": any(bool(a.get("classifier_fallback")) for a in incorrect),
    }


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.to(torch.float32).flatten()
    bf = b.to(torch.float32).flatten()
    denom = af.norm() * bf.norm()
    if denom < 1e-12:
        return float("nan")
    return float((af @ bf) / denom)


def stats(values: Iterable[float]) -> Dict[str, float | int]:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(arr.size)}
