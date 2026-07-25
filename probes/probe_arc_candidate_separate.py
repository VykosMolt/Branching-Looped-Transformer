"""Separate-candidate ARC-shaped choice probes for Ouro/RLTT loop states.

The anti-saturation text probe flattened all candidates into one prompt:

    demo + test input + candidates A/B/C/D -> label A/B/C/D

This script tests a cleaner preference shape:

    demo + test input + one candidate -> binary correct/incorrect

Each problem contributes four prompts sharing one group id. Cross-validation is
grouped by problem, and the main metric is whether the probe assigns the
highest held-out score to the correct candidate.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluator_core.pairwise_evaluator import validate_hook_output


N_LOOPS = 4
DEFAULT_TASKS = ("arc_output_choice", "arc_action_choice", "arc_movable_choice")


@dataclass(frozen=True)
class CandidateExample:
    task: str
    problem_id: int
    prompt: str
    label: int
    candidate_index: int


class LoopStateCapture:
    def __init__(self) -> None:
        self.captured: Dict[str, List[torch.Tensor]] = {}
        self.validated = False

    def hook_fn(self, module, inputs, output) -> None:  # type: ignore[no-untyped-def]
        hidden_states = output[1]
        if not self.validated:
            validate_hook_output(hidden_states)
            self.validated = True
        self.captured["hidden_states_list"] = [h.detach() for h in hidden_states]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Separate-candidate mini-ARC loop-state probe.")
    parser.add_argument("--model-path", default="models/ouro_rltt_local")
    parser.add_argument("--model-label", default="")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--samples-per-task", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--early-exit-threshold", type=float, default=1.0)
    parser.add_argument("--output", default="artifacts/reports/evaluator/probe_arc_candidate_separate.json")
    parser.add_argument("--allow-remote", action="store_true")
    return parser.parse_args()


def grid_to_text(grid: np.ndarray) -> str:
    return "\n".join(" ".join(str(int(v)) for v in row) for row in grid)


def add_rect(grid: np.ndarray, top: int, left: int, height: int, width: int, value: int) -> None:
    grid[top : top + height, left : left + width] = value


def object_grid(rng: np.random.RandomState, size: int = 7) -> Tuple[np.ndarray, Tuple[int, int, int, int], int]:
    grid = np.zeros((size, size), dtype=np.int64)
    height = int(rng.randint(2, 4))
    width = int(rng.randint(2, 4))
    top = int(rng.randint(0, size - height))
    left = int(rng.randint(0, size - width))
    value = int(rng.randint(2, 9))
    add_rect(grid, top, left, height, width, value)
    return grid, (top, left, height, width), value


def transform_grid(grid: np.ndarray, rule: str, value: int) -> np.ndarray:
    out = np.zeros_like(grid)
    positions = np.argwhere(grid == value)
    if positions.size == 0:
        return out
    if rule == "shift_right":
        shifted = positions + np.array([0, 1])
    elif rule == "shift_left":
        shifted = positions + np.array([0, -1])
    elif rule == "shift_down":
        shifted = positions + np.array([1, 0])
    elif rule == "shift_up":
        shifted = positions + np.array([-1, 0])
    elif rule == "mirror_horizontal":
        shifted = positions.copy()
        shifted[:, 1] = grid.shape[1] - 1 - shifted[:, 1]
    elif rule == "mirror_vertical":
        shifted = positions.copy()
        shifted[:, 0] = grid.shape[0] - 1 - shifted[:, 0]
    else:
        raise ValueError(f"unknown rule {rule}")
    valid = (
        (shifted[:, 0] >= 0)
        & (shifted[:, 0] < grid.shape[0])
        & (shifted[:, 1] >= 0)
        & (shifted[:, 1] < grid.shape[1])
    )
    for r, c in shifted[valid]:
        out[int(r), int(c)] = value
    return out


def viable_object_grid_for_rule(
    rng: np.random.RandomState,
    rule: str,
    size: int = 7,
) -> Tuple[np.ndarray, int]:
    for _ in range(100):
        grid, _, value = object_grid(rng, size)
        out = transform_grid(grid, rule, value)
        if out.sum() > 0 and not np.array_equal(out, grid):
            return grid, value
    raise RuntimeError(f"failed to make viable grid for {rule}")


def make_output_choice_problem(rng: np.random.RandomState) -> Tuple[str, List[str], int]:
    rules = ["shift_right", "shift_left", "shift_down", "shift_up", "mirror_horizontal", "mirror_vertical"]
    rule = str(rng.choice(rules))
    demo_in, demo_value = viable_object_grid_for_rule(rng, rule)
    test_in, test_value = viable_object_grid_for_rule(rng, rule)
    demo_out = transform_grid(demo_in, rule, demo_value)
    correct = transform_grid(test_in, rule, test_value)

    wrong_rules = [candidate for candidate in rules if candidate != rule]
    distractors = [
        test_in.copy(),
        transform_grid(test_in, str(rng.choice(wrong_rules)), test_value),
        np.rot90(correct).copy(),
    ]
    candidates = [correct] + distractors
    correct_slot = int(rng.randint(0, 4))
    ordered = candidates[1:]
    ordered.insert(correct_slot, correct)

    stem = (
        "Mini-ARC candidate judgment.\n"
        "A rule maps the demo input grid to the demo output grid. "
        "Judge whether the candidate output correctly applies the same rule to the test input.\n"
        "Demo input:\n"
        f"{grid_to_text(demo_in)}\n"
        "Demo output:\n"
        f"{grid_to_text(demo_out)}\n"
        "Test input:\n"
        f"{grid_to_text(test_in)}\n"
    )
    prompts = [
        f"{stem}Candidate output:\n{grid_to_text(candidate)}\nIs this candidate correct?"
        for candidate in ordered
    ]
    return "arc_output_choice", prompts, correct_slot


def action_grid(rng: np.random.RandomState, size: int = 7) -> Tuple[np.ndarray, List[str], int]:
    grid = np.zeros((size, size), dtype=np.int64)
    row = int(rng.randint(1, size - 1))
    col = int(rng.randint(1, size - 1))
    grid[row, col] = 3
    actions = ["move up", "move down", "move left", "move right"]
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    valid = []
    for idx, (dr, dc) in enumerate(deltas):
        rr = row + dr
        cc = col + dc
        if 0 <= rr < size and 0 <= cc < size:
            valid.append(idx)
    correct = int(rng.choice(valid))
    # Block at least one non-correct action without blocking the correct one.
    for idx, (dr, dc) in enumerate(deltas):
        if idx == correct:
            continue
        rr = row + dr
        cc = col + dc
        if 0 <= rr < size and 0 <= cc < size and rng.rand() < 0.5:
            grid[rr, cc] = 8
    return grid, actions, correct


def make_action_choice_problem(rng: np.random.RandomState) -> Tuple[str, List[str], int]:
    grid, actions, correct = action_grid(rng)
    stem = (
        "Mini-ARC action judgment.\n"
        "The object value 3 may move one cell if the destination is inside the grid and empty.\n"
        "State grid:\n"
        f"{grid_to_text(grid)}\n"
    )
    prompts = [f"{stem}Candidate action: {action}.\nIs this action valid and correct?" for action in actions]
    return "arc_action_choice", prompts, correct


def movable_grid(rng: np.random.RandomState, correct: int, size: int = 7) -> Tuple[np.ndarray, List[int]]:
    grid = np.zeros((size, size), dtype=np.int64)
    labels = [2, 4, 6, 8]
    rows = [1, 2, 4, 5]
    for idx, row in enumerate(rows):
        col = int(rng.randint(1, size - 2))
        grid[row, col] = labels[idx]
        if idx != correct:
            grid[row, col + 1] = 9
    return grid, labels


def make_movable_choice_problem(rng: np.random.RandomState) -> Tuple[str, List[str], int]:
    correct = int(rng.randint(0, 4))
    grid, labels = movable_grid(rng, correct)
    names = ["A", "B", "C", "D"]
    stem = (
        "Mini-ARC object-movement judgment.\n"
        "An object can move right only if the cell immediately to its right is empty.\n"
        "State grid:\n"
        f"{grid_to_text(grid)}\n"
    )
    prompts = [
        f"{stem}Candidate object: {names[idx]} with value {labels[idx]}.\nCan this object move right?"
        for idx in range(4)
    ]
    return "arc_movable_choice", prompts, correct


TASK_GENERATORS: Dict[str, Callable[[np.random.RandomState], Tuple[str, List[str], int]]] = {
    "arc_output_choice": make_output_choice_problem,
    "arc_action_choice": make_action_choice_problem,
    "arc_movable_choice": make_movable_choice_problem,
}


def make_examples(tasks: Sequence[str], samples_per_task: int, seed: int) -> List[CandidateExample]:
    examples: List[CandidateExample] = []
    problem_id = 0
    for task_offset, task in enumerate(tasks):
        rng = np.random.RandomState(seed + task_offset * 1009)
        for _ in range(samples_per_task):
            task_name, prompts, correct = TASK_GENERATORS[task](rng)
            for candidate_index, prompt in enumerate(prompts):
                examples.append(
                    CandidateExample(
                        task=task_name,
                        problem_id=problem_id,
                        prompt=prompt,
                        label=1 if candidate_index == correct else 0,
                        candidate_index=candidate_index,
                    )
                )
            problem_id += 1
    return examples


@torch.no_grad()
def extract_loop_features(
    model,
    tokenizer,
    capture: LoopStateCapture,
    prompts: Sequence[str],
    device: torch.device,
    max_length: int,
) -> Tuple[Dict[str, np.ndarray], List[np.ndarray]]:
    capture.captured.clear()
    tokens = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)
    model(**tokens, use_cache=False)
    states = capture.captured.get("hidden_states_list")
    if states is None or len(states) != N_LOOPS:
        raise RuntimeError(f"expected {N_LOOPS} loop states, got {0 if states is None else len(states)}")
    mask = tokens["attention_mask"].float().to(device)
    pooled = []
    for state in states:
        s = state.to(device=device, dtype=torch.float32)
        pooled.append((s * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True))
    loop_np = [p.cpu().numpy() for p in pooled]
    features = {
        "L1": loop_np[0],
        "L4": loop_np[-1],
        "L4_minus_L1": loop_np[-1] - loop_np[0],
        "all_loops": torch.cat(pooled, dim=-1).cpu().numpy(),
    }
    del tokens, states, pooled
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return features, loop_np


def loop_metrics(loop_features: Sequence[np.ndarray]) -> Dict[str, float]:
    loops = [torch.from_numpy(x).float() for x in loop_features]
    cos_l1_l4 = torch.nn.functional.cosine_similarity(loops[0], loops[-1], dim=-1)
    deltas = [(loops[i + 1] - loops[i]).norm(dim=-1) for i in range(N_LOOPS - 1)]
    return {
        "l1_l4_cosine_mean": float(cos_l1_l4.mean().item()),
        "l1_l4_cosine_std": float(cos_l1_l4.std(unbiased=False).item()),
        "total_refinement_delta_mean": float(sum(d.mean().item() for d in deltas)),
        "l1_l2_delta_mean": float(deltas[0].mean().item()),
        "l2_l3_delta_mean": float(deltas[1].mean().item()),
        "l3_l4_delta_mean": float(deltas[2].mean().item()),
    }


def choose_pca_components(x: np.ndarray, y: np.ndarray) -> int:
    class_count = len(set(int(v) for v in y.tolist()))
    return int(max(1, min(80, x.shape[0] - class_count, x.shape[1])))


def fit_group_probe(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    candidate_indices: np.ndarray,
) -> Dict[str, float]:
    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))
    if n_splits < 2 or len(set(int(v) for v in y.tolist())) < 2:
        return {
            "binary_accuracy": float("nan"),
            "binary_auc": float("nan"),
            "choice_accuracy": float("nan"),
            "n_splits": int(n_splits),
        }
    n_components = choose_pca_components(x, y)
    clf = make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    cv = GroupKFold(n_splits=n_splits)
    scores = cross_val_predict(clf, x, y, groups=groups, cv=cv, method="decision_function")
    binary_pred = (scores >= 0.0).astype(np.int64)
    choice_hits = []
    for group in unique_groups:
        idx = np.where(groups == group)[0]
        winner = idx[int(np.argmax(scores[idx]))]
        choice_hits.append(int(y[winner] == 1))
    try:
        auc = float(roc_auc_score(y, scores))
    except ValueError:
        auc = float("nan")
    return {
        "binary_accuracy": float(accuracy_score(y, binary_pred)),
        "binary_auc": auc,
        "choice_accuracy": float(np.mean(choice_hits)),
        "mean_correct_score": float(scores[y == 1].mean()),
        "mean_incorrect_score": float(scores[y == 0].mean()),
        "score_margin": float(scores[y == 1].mean() - scores[y == 0].mean()),
        "n_splits": int(n_splits),
        "pca_components": int(n_components),
        "groups": int(len(unique_groups)),
        "examples": int(len(y)),
        "candidate_index_histogram": {
            str(int(k)): int(v) for k, v in sorted(Counter(candidate_indices.tolist()).items())
        },
    }


def evaluate_features(
    features: Dict[str, np.ndarray],
    loop_arrays: List[np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    tasks: np.ndarray,
    candidate_indices: np.ndarray,
) -> Dict[str, object]:
    result: Dict[str, object] = {
        "overall": {},
        "by_task": {},
        "loop_metrics": loop_metrics(loop_arrays),
        "label_histogram": {str(int(k)): int(v) for k, v in sorted(Counter(labels.tolist()).items())},
    }
    for feature_name, x in features.items():
        result["overall"][feature_name] = fit_group_probe(x, labels, groups, candidate_indices)  # type: ignore[index]
    for task in sorted(set(tasks.tolist())):
        mask = tasks == task
        task_result: Dict[str, object] = {
            "label_histogram": {str(int(k)): int(v) for k, v in sorted(Counter(labels[mask].tolist()).items())},
            "loop_metrics": loop_metrics([arr[mask] for arr in loop_arrays]),
            "features": {},
        }
        for feature_name, x in features.items():
            task_result["features"][feature_name] = fit_group_probe(  # type: ignore[index]
                x[mask],
                labels[mask],
                groups[mask],
                candidate_indices[mask],
            )
        result["by_task"][str(task)] = task_result  # type: ignore[index]
    return result


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    tasks = [value.strip() for value in args.tasks.split(",") if value.strip()]
    unknown = [task for task in tasks if task not in TASK_GENERATORS]
    if unknown:
        raise ValueError(f"unknown tasks: {unknown}; available: {sorted(TASK_GENERATORS)}")
    device = resolve_device(args.device)
    model_label = args.model_label or args.model_path.replace("/", "_")
    examples = make_examples(tasks, args.samples_per_task, args.seed)
    print(f"Model: {args.model_path}", flush=True)
    print(f"Label: {model_label}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Problems: {len(set(ex.problem_id for ex in examples))}; prompts: {len(examples)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=not args.allow_remote,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map={"": str(device)},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=not args.allow_remote,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.config.early_exit_threshold = args.early_exit_threshold
    capture = LoopStateCapture()
    handle = model.model.register_forward_hook(capture.hook_fn)

    feature_chunks: Dict[str, List[np.ndarray]] = defaultdict(list)
    loop_chunks: List[List[np.ndarray]] = [[] for _ in range(N_LOOPS)]
    try:
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start : start + args.batch_size]
            features, loop_arrays = extract_loop_features(
                model,
                tokenizer,
                capture,
                [ex.prompt for ex in batch],
                device,
                args.max_length,
            )
            for name, arr in features.items():
                feature_chunks[name].append(arr)
            for idx, arr in enumerate(loop_arrays):
                loop_chunks[idx].append(arr)
            if (start // args.batch_size + 1) % 50 == 0:
                print(f"  processed {min(start + args.batch_size, len(examples))}/{len(examples)} prompts", flush=True)
    finally:
        handle.remove()

    features = {name: np.concatenate(chunks, axis=0) for name, chunks in feature_chunks.items()}
    loop_arrays = [np.concatenate(chunks, axis=0) for chunks in loop_chunks]
    labels = np.asarray([ex.label for ex in examples], dtype=np.int64)
    groups = np.asarray([ex.problem_id for ex in examples], dtype=np.int64)
    task_names = np.asarray([ex.task for ex in examples])
    candidate_indices = np.asarray([ex.candidate_index for ex in examples], dtype=np.int64)

    result = {
        "model_path": args.model_path,
        "model_label": model_label,
        "seed": args.seed,
        "tasks": tasks,
        "samples_per_task": args.samples_per_task,
        "prompts": len(examples),
        "problems": int(len(set(int(v) for v in groups.tolist()))),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "early_exit_threshold": args.early_exit_threshold,
        "metrics": evaluate_features(features, loop_arrays, labels, groups, task_names, candidate_indices),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True)[:6000], flush=True)
    print(f"Wrote {output}", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
