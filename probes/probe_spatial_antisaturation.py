"""Anti-saturation spatial probes for Ouro/RLTT loop states.

The earlier spatial probe saturated at L1. This script uses harder synthetic
relational labels and compares linear probes trained on:

    L1, L4, L4 - L1, and all four loops concatenated.

It can be run on the converted RLTT checkpoint and on the base Ouro checkpoint
with the same deterministic examples for a direct comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator_core.pairwise_evaluator import validate_hook_output


N_LOOPS = 4
DEFAULT_TASKS = (
    "components",
    "symmetry",
    "containment",
    "path",
    "occlusion",
    "transform",
    "collision",
    "arc_output_choice",
    "arc_action_choice",
    "arc_movable_choice",
)


@dataclass(frozen=True)
class SpatialExample:
    task: str
    prompt: str
    label: int
    label_name: str


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
    parser = argparse.ArgumentParser(description="Harder anti-saturation spatial loop probes.")
    parser.add_argument("--model-path", default="models/ouro_rltt_local")
    parser.add_argument("--model-label", default="")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--samples-per-task", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--early-exit-threshold", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(device_arg)


def grid_to_text(grid: np.ndarray) -> str:
    return "\n".join(" ".join(str(int(c)) for c in row) for row in grid)


def pair_prompt(title: str, a: np.ndarray, b: np.ndarray, question: str) -> str:
    return (
        f"{title}\n"
        f"Input grid:\n{grid_to_text(a)}\n\n"
        f"Output grid:\n{grid_to_text(b)}\n\n"
        f"Question: {question}"
    )


def candidate_prompt(
    title: str,
    sections: Sequence[Tuple[str, np.ndarray]],
    candidates: Sequence[np.ndarray],
    question: str,
) -> str:
    parts = [title]
    for label, grid in sections:
        parts.append(f"{label}:\n{grid_to_text(grid)}")
    for idx, grid in enumerate(candidates):
        letter = chr(ord("A") + idx)
        parts.append(f"Candidate {letter}:\n{grid_to_text(grid)}")
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)


def single_prompt(title: str, grid: np.ndarray, question: str) -> str:
    return f"{title}\nGrid:\n{grid_to_text(grid)}\n\nQuestion: {question}"


def place_rect(grid: np.ndarray, top: int, left: int, h: int, w: int, value: int) -> None:
    grid[top : top + h, left : left + w] = value


def rects_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int], pad: int = 0) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return not (
        ar + ah + pad <= br
        or br + bh + pad <= ar
        or ac + aw + pad <= bc
        or bc + bw + pad <= ac
    )


def random_nonoverlap_rects(
    rng: np.random.RandomState,
    count: int,
    size: int = 10,
    max_attempts: int = 500,
) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    grid = np.zeros((size, size), dtype=np.int64)
    rects: List[Tuple[int, int, int, int]] = []
    for idx in range(count):
        for _ in range(max_attempts):
            h = int(rng.randint(1, 3))
            w = int(rng.randint(1, 3))
            r = int(rng.randint(0, size - h + 1))
            c = int(rng.randint(0, size - w + 1))
            rect = (r, c, h, w)
            if all(not rects_overlap(rect, other, pad=1) for other in rects):
                rects.append(rect)
                place_rect(grid, r, c, h, w, (idx % 8) + 1)
                break
        else:
            return random_nonoverlap_rects(rng, count, size=size + 2)
    return grid, rects


def make_component_examples(n: int, rng: np.random.RandomState) -> List[SpatialExample]:
    examples = []
    labels = [1, 2, 3, 4]
    for i in range(n):
        label = labels[i % len(labels)]
        grid, _ = random_nonoverlap_rects(rng, label, size=10)
        prompt = single_prompt(
            "Connected-component count task.",
            grid,
            "Classify the number of separate nonzero connected objects.",
        )
        examples.append(SpatialExample("components", prompt, label - 1, str(label)))
    return examples


def make_symmetric_grid(rng: np.random.RandomState, label: int, size: int = 10) -> np.ndarray:
    if label == 0:
        while True:
            grid = rng.randint(0, 4, (size, size))
            if not np.array_equal(grid, np.flipud(grid)) and not np.array_equal(grid, np.fliplr(grid)):
                return grid
    if label == 1:
        top = rng.randint(0, 4, (size // 2, size))
        return np.vstack([top, np.flipud(top)])
    if label == 2:
        left = rng.randint(0, 4, (size, size // 2))
        return np.hstack([left, np.fliplr(left)])
    base = rng.randint(0, 4, (size // 2, size // 2))
    top = np.hstack([base, np.fliplr(base)])
    return np.vstack([top, np.flipud(top)])


def make_symmetry_examples(n: int, rng: np.random.RandomState) -> List[SpatialExample]:
    names = ["none", "horizontal", "vertical", "both"]
    examples = []
    for i in range(n):
        label = i % 4
        grid = make_symmetric_grid(rng, label)
        prompt = single_prompt(
            "Symmetry-axis task.",
            grid,
            "Classify the symmetry axis of the grid.",
        )
        examples.append(SpatialExample("symmetry", prompt, label, names[label]))
    return examples


def make_containment_grid(rng: np.random.RandomState, inside: bool) -> np.ndarray:
    grid = np.zeros((12, 12), dtype=np.int64)
    r = int(rng.randint(1, 4))
    c = int(rng.randint(1, 4))
    h = int(rng.randint(5, 8))
    w = int(rng.randint(5, 8))
    grid[r, c : c + w] = 2
    grid[r + h - 1, c : c + w] = 2
    grid[r : r + h, c] = 2
    grid[r : r + h, c + w - 1] = 2
    if inside:
        rr = int(rng.randint(r + 1, r + h - 1))
        cc = int(rng.randint(c + 1, c + w - 1))
    else:
        outside = []
        for rr in range(12):
            for cc in range(12):
                if not (r <= rr < r + h and c <= cc < c + w):
                    outside.append((rr, cc))
        rr, cc = outside[int(rng.randint(0, len(outside)))]
    grid[rr, cc] = 3
    return grid


def make_containment_examples(n: int, rng: np.random.RandomState) -> List[SpatialExample]:
    examples = []
    for i in range(n):
        label = i % 2
        grid = make_containment_grid(rng, inside=bool(label))
        prompt = single_prompt(
            "Object-containment task. Value 2 is a container boundary; value 3 is an object.",
            grid,
            "Decide whether object 3 is inside the closed boundary.",
        )
        examples.append(SpatialExample("containment", prompt, label, "inside" if label else "outside"))
    return examples


def has_path(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
    queue = [start]
    seen = {start}
    while queue:
        r, c = queue.pop(0)
        if (r, c) == goal:
            return True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                if (nr, nc) not in seen and grid[nr, nc] != 1:
                    seen.add((nr, nc))
                    queue.append((nr, nc))
    return False


def make_path_grid(rng: np.random.RandomState, reachable: bool, size: int = 12) -> np.ndarray:
    for _ in range(500):
        grid = (rng.rand(size, size) < 0.26).astype(np.int64)
        start = (0, int(rng.randint(0, size)))
        goal = (size - 1, int(rng.randint(0, size)))
        grid[start] = 2
        grid[goal] = 3
        ok = has_path(grid, start, goal)
        if ok == reachable:
            return grid
    grid = np.zeros((size, size), dtype=np.int64)
    start = (0, 1)
    goal = (size - 1, size - 2)
    if not reachable:
        grid[size // 2, :] = 1
    grid[start] = 2
    grid[goal] = 3
    return grid


def make_path_examples(n: int, rng: np.random.RandomState) -> List[SpatialExample]:
    examples = []
    for i in range(n):
        label = i % 2
        grid = make_path_grid(rng, reachable=bool(label))
        prompt = single_prompt(
            "Shortest-path existence task. 0 empty, 1 wall, 2 start, 3 goal.",
            grid,
            "Decide whether a four-neighbor path exists from 2 to 3 without crossing walls.",
        )
        examples.append(SpatialExample("path", prompt, label, "reachable" if label else "blocked"))
    return examples


def make_occlusion_grid(rng: np.random.RandomState, continuation: bool, size: int = 11) -> np.ndarray:
    grid = np.zeros((size, size), dtype=np.int64)
    r = int(rng.randint(3, size - 3))
    c = int(rng.randint(3, size - 3))
    grid[r, c] = 9
    if continuation:
        mode = int(rng.randint(0, 4))
        if mode == 0:
            pts = [(r, c - 2), (r, c + 2)]
        elif mode == 1:
            pts = [(r - 2, c), (r + 2, c)]
        elif mode == 2:
            pts = [(r - 2, c - 2), (r + 2, c + 2)]
        else:
            pts = [(r - 2, c + 2), (r + 2, c - 2)]
    else:
        pts = [(r - 2, c - 1), (r + 2, c + 2)]
    for pr, pc in pts:
        grid[pr, pc] = 4
    return grid


def make_occlusion_examples(n: int, rng: np.random.RandomState) -> List[SpatialExample]:
    examples = []
    for i in range(n):
        label = i % 2
        grid = make_occlusion_grid(rng, continuation=bool(label))
        prompt = single_prompt(
            "Occluded-continuation task. Value 9 is an occluder; value 4 marks visible parts of an object.",
            grid,
            "Decide whether the two visible value-4 parts lie on one straight line through the occluder.",
        )
        examples.append(SpatialExample("occlusion", prompt, label, "continuation" if label else "broken"))
    return examples


def base_object(rng: np.random.RandomState) -> np.ndarray:
    obj = np.zeros((4, 4), dtype=np.int64)
    coords = [(0, 1), (1, 1), (2, 1), (2, 2), (3, 0)]
    color = int(rng.randint(2, 8))
    for r, c in coords:
        obj[r, c] = color
    if rng.rand() < 0.5:
        obj = np.rot90(obj)
    return obj.copy()


def paste_obj(grid: np.ndarray, obj: np.ndarray, r: int, c: int) -> None:
    mask = obj != 0
    grid[r : r + obj.shape[0], c : c + obj.shape[1]][mask] = obj[mask]


def make_transform_pair(rng: np.random.RandomState, label: int) -> Tuple[np.ndarray, np.ndarray]:
    inp = np.zeros((10, 10), dtype=np.int64)
    out = np.zeros((10, 10), dtype=np.int64)
    obj = base_object(rng)
    r = int(rng.randint(2, 4))
    c = int(rng.randint(2, 4))
    paste_obj(inp, obj, r, c)
    if label == 0:
        paste_obj(out, obj, r, c + 2)
    elif label == 1:
        paste_obj(out, obj, r + 2, c)
    elif label == 2:
        paste_obj(out, np.fliplr(obj), r, c)
    else:
        recolored = np.where(obj != 0, (obj % 8) + 1, 0)
        paste_obj(out, recolored, r, c)
    return inp, out


def make_transform_examples(n: int, rng: np.random.RandomState) -> List[SpatialExample]:
    names = ["shift_right", "shift_down", "mirror_horizontal", "recolor"]
    examples = []
    for i in range(n):
        label = i % 4
        inp, out = make_transform_pair(rng, label)
        prompt = pair_prompt(
            "Transformation-class task.",
            inp,
            out,
            "Classify the transformation that maps input to output.",
        )
        examples.append(SpatialExample("transform", prompt, label, names[label]))
    return examples


def make_collision_grid(rng: np.random.RandomState, collides: bool, size: int = 10) -> Tuple[np.ndarray, str]:
    dirs = [("up", -1, 0), ("down", 1, 0), ("left", 0, -1), ("right", 0, 1)]
    name, dr, dc = dirs[int(rng.randint(0, len(dirs)))]
    grid = np.zeros((size, size), dtype=np.int64)
    r = int(rng.randint(2, size - 2))
    c = int(rng.randint(2, size - 2))
    grid[r, c] = 2
    target = (r + dr, c + dc)
    if collides:
        grid[target] = 3
    else:
        free = []
        for rr in range(size):
            for cc in range(size):
                if (rr, cc) not in [(r, c), target]:
                    free.append((rr, cc))
        rr, cc = free[int(rng.randint(0, len(free)))]
        grid[rr, cc] = 3
    return grid, name


def make_collision_examples(n: int, rng: np.random.RandomState) -> List[SpatialExample]:
    examples = []
    for i in range(n):
        label = i % 2
        grid, direction = make_collision_grid(rng, collides=bool(label))
        prompt = single_prompt(
            f"Collision-after-action task. Value 2 is the mover. Value 3 is an obstacle. Action: move {direction}.",
            grid,
            "Decide whether the mover would collide with the obstacle after this action.",
        )
        examples.append(SpatialExample("collision", prompt, label, "collides" if label else "clear"))
    return examples


def random_arc_object_grid(rng: np.random.RandomState, size: int = 8) -> np.ndarray:
    """Small asymmetric object grid for mini-ARC candidate tasks."""
    grid = np.zeros((size, size), dtype=np.int64)
    color = int(rng.randint(2, 8))
    base = [(0, 1), (1, 1), (2, 1), (2, 2), (3, 0)]
    if rng.rand() < 0.5:
        base.append((1, 3))
    top = int(rng.randint(1, size - 4))
    left = int(rng.randint(1, size - 4))
    for r, c in base:
        grid[top + r, left + c] = color
    marker = int((color % 8) + 1)
    grid[top, left] = marker
    return grid


def apply_arc_rule(grid: np.ndarray, rule: int) -> np.ndarray:
    """Apply one of four deterministic mini-ARC transformations."""
    if rule == 0:  # shift right
        out = np.zeros_like(grid)
        out[:, 1:] = grid[:, :-1]
        return out
    if rule == 1:  # shift down
        out = np.zeros_like(grid)
        out[1:, :] = grid[:-1, :]
        return out
    if rule == 2:  # horizontal mirror
        return np.fliplr(grid).copy()
    if rule == 3:  # recolor nonzero cells
        return np.where(grid != 0, (grid % 8) + 1, 0).astype(np.int64)
    raise ValueError(f"unknown arc rule: {rule}")


def unique_arc_candidate_outputs(grid: np.ndarray) -> Optional[List[np.ndarray]]:
    outputs = [apply_arc_rule(grid, rule) for rule in range(4)]
    seen = {tuple(out.reshape(-1).tolist()) for out in outputs}
    if len(seen) != len(outputs):
        return None
    return outputs


def make_arc_output_choice_examples(n: int, rng: np.random.RandomState) -> List[SpatialExample]:
    """Infer a rule from one example and choose the correct output among decoys."""
    examples = []
    labels = ["A", "B", "C", "D"]
    for i in range(n):
        rule = i % 4
        correct_position = (i // 4) % 4
        for _ in range(200):
            demo_in = random_arc_object_grid(rng)
            test_in = random_arc_object_grid(rng)
            outputs = unique_arc_candidate_outputs(test_in)
            if outputs is not None:
                break
        else:
            raise RuntimeError("could not generate unique ARC output candidates")
        demo_out = apply_arc_rule(demo_in, rule)
        distractor_rules = [r for r in range(4) if r != rule]
        rng.shuffle(distractor_rules)
        ordered_rules = distractor_rules[:]
        ordered_rules.insert(correct_position, rule)
        candidates = [outputs[r] for r in ordered_rules]
        prompt = candidate_prompt(
            "Mini-ARC output-choice task. Infer the rule from the demonstration pair.",
            [
                ("Demonstration input", demo_in),
                ("Demonstration output", demo_out),
                ("Test input", test_in),
            ],
            candidates,
            "Which candidate output applies the same hidden rule to the test input?",
        )
        examples.append(
            SpatialExample(
                "arc_output_choice",
                prompt,
                correct_position,
                labels[correct_position],
            )
        )
    return examples


def make_action_choice_grid(
    rng: np.random.RandomState,
    correct_action: int,
    size: int = 9,
) -> np.ndarray:
    """Create a grid where exactly one action moves value 2 into empty space."""
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    grid = np.ones((size, size), dtype=np.int64)
    r = int(rng.randint(2, size - 2))
    c = int(rng.randint(2, size - 2))
    grid[r, c] = 2
    dr, dc = dirs[correct_action]
    grid[r + dr, c + dc] = 0
    for idx, (odr, odc) in enumerate(dirs):
        if idx != correct_action:
            grid[r + odr, c + odc] = 1
    # Add irrelevant open cells away from the immediate action neighborhood.
    for _ in range(8):
        rr = int(rng.randint(0, size))
        cc = int(rng.randint(0, size))
        if abs(rr - r) + abs(cc - c) > 2:
            grid[rr, cc] = 0
    return grid


def make_arc_action_choice_examples(n: int, rng: np.random.RandomState) -> List[SpatialExample]:
    names = ["up", "down", "left", "right"]
    examples = []
    for i in range(n):
        label = i % 4
        grid = make_action_choice_grid(rng, label)
        prompt = single_prompt(
            "Mini-ARC action-choice task. Value 2 is the mover, 0 is empty, 1 is blocked.",
            grid,
            "Which action changes the state by moving value 2 into empty space? Options: A up, B down, C left, D right.",
        )
        examples.append(SpatialExample("arc_action_choice", prompt, label, names[label]))
    return examples


def make_movable_choice_grid(
    rng: np.random.RandomState,
    movable_index: int,
    size: int = 11,
) -> Tuple[np.ndarray, List[int]]:
    """Four labeled objects; exactly one has a free rightward cell."""
    grid = np.zeros((size, size), dtype=np.int64)
    labels = [2, 3, 4, 5]
    positions = [(2, 2), (2, 7), (7, 2), (7, 7)]
    for idx, (label, (r, c)) in enumerate(zip(labels, positions)):
        grid[r, c] = label
        grid[r, c + 1] = 0 if idx == movable_index else 1
        grid[r - 1, c] = 1
        grid[r + 1, c] = 1
        grid[r, c - 1] = 1
    # Add mild clutter that does not alter the labeled-object answer.
    for _ in range(10):
        rr = int(rng.randint(0, size))
        cc = int(rng.randint(0, size))
        if grid[rr, cc] == 0 and (rr, cc) not in positions:
            grid[rr, cc] = int(rng.randint(0, 2))
    return grid, labels


def make_arc_movable_choice_examples(n: int, rng: np.random.RandomState) -> List[SpatialExample]:
    options = ["A", "B", "C", "D"]
    examples = []
    for i in range(n):
        label = i % 4
        grid, object_labels = make_movable_choice_grid(rng, label)
        prompt = single_prompt(
            "Mini-ARC movable-object task. Four labeled objects may try to move right.",
            grid,
            (
                "Which object can move right into an empty cell? "
                f"Options: A value {object_labels[0]}, B value {object_labels[1]}, "
                f"C value {object_labels[2]}, D value {object_labels[3]}."
            ),
        )
        examples.append(SpatialExample("arc_movable_choice", prompt, label, options[label]))
    return examples


TASK_GENERATORS: Dict[str, Callable[[int, np.random.RandomState], List[SpatialExample]]] = {
    "components": make_component_examples,
    "symmetry": make_symmetry_examples,
    "containment": make_containment_examples,
    "path": make_path_examples,
    "occlusion": make_occlusion_examples,
    "transform": make_transform_examples,
    "collision": make_collision_examples,
    "arc_output_choice": make_arc_output_choice_examples,
    "arc_action_choice": make_arc_action_choice_examples,
    "arc_movable_choice": make_arc_movable_choice_examples,
}


def make_examples(tasks: Sequence[str], samples_per_task: int, seed: int) -> List[SpatialExample]:
    examples: List[SpatialExample] = []
    for offset, task in enumerate(tasks):
        rng = np.random.RandomState(seed + offset * 1009)
        examples.extend(TASK_GENERATORS[task](samples_per_task, rng))
    return examples


@torch.no_grad()
def extract_loop_features(
    model,
    tokenizer,
    capture: LoopStateCapture,
    prompts: Sequence[str],
    device: torch.device,
    max_length: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
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
        masked = s * mask.unsqueeze(-1)
        pooled.append(masked.sum(dim=1) / mask.sum(dim=1, keepdim=True))
    l1 = pooled[0].cpu().numpy()
    l4 = pooled[-1].cpu().numpy()
    features = {
        "L1": l1,
        "L4": l4,
        "L4_minus_L1": l4 - l1,
        "all_loops": torch.cat(pooled, dim=-1).cpu().numpy(),
    }
    del tokens, pooled
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return l1, features


def fit_probe_scores(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    counts = Counter(int(v) for v in y.tolist())
    min_count = min(counts.values())
    n_splits = min(5, min_count)
    if n_splits < 2:
        return {"accuracy_mean": float("nan"), "accuracy_std": float("nan"), "n_splits": n_splits}
    n_components = min(50, x.shape[0] - len(counts), x.shape[1])
    n_components = max(1, n_components)
    clf = make_pipeline(
        PCA(n_components=n_components),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = cross_val_score(clf, x, y, cv=cv, scoring="accuracy")
    return {
        "accuracy_mean": float(scores.mean()),
        "accuracy_std": float(scores.std()),
        "n_splits": int(n_splits),
        "pca_components": int(n_components),
    }


def loop_metrics(loop_features: List[np.ndarray]) -> Dict[str, float]:
    loops = [torch.from_numpy(x).float() for x in loop_features]
    cos_l1_l4 = torch.nn.functional.cosine_similarity(loops[0], loops[-1], dim=-1)
    deltas = [(loops[i + 1] - loops[i]).norm(dim=-1) for i in range(N_LOOPS - 1)]
    return {
        "l1_l4_cosine_mean": float(cos_l1_l4.mean().item()),
        "l1_l4_cosine_std": float(cos_l1_l4.std().item()),
        "total_refinement_delta_mean": float(sum(d.mean().item() for d in deltas)),
        "l1_l2_delta_mean": float(deltas[0].mean().item()),
        "l2_l3_delta_mean": float(deltas[1].mean().item()),
        "l3_l4_delta_mean": float(deltas[2].mean().item()),
    }


def main() -> None:
    args = parse_args()
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_MODULES_CACHE", str(PROJECT_ROOT / ".hf_modules_cache"))

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in TASK_GENERATORS]
    if unknown:
        raise ValueError(f"unknown tasks: {unknown}; available: {sorted(TASK_GENERATORS)}")

    device = resolve_device(args.device)
    model_path = args.model_path
    model_label = args.model_label or model_path.replace("/", "_")
    examples = make_examples(tasks, args.samples_per_task, args.seed)

    print(f"Model: {model_path}")
    print(f"Label: {model_label}")
    print(f"Device: {device}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Examples: {len(examples)}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": str(device)},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.config.early_exit_threshold = args.early_exit_threshold

    capture = LoopStateCapture()
    handle = model.model.register_forward_hook(capture.hook_fn)

    per_task: Dict[str, Dict[str, object]] = {}
    all_loop_arrays: Dict[str, List[np.ndarray]] = {f"L{i + 1}": [] for i in range(N_LOOPS)}
    all_features_by_name: Dict[str, List[np.ndarray]] = {
        "L1": [],
        "L4": [],
        "L4_minus_L1": [],
        "all_loops": [],
    }
    all_labels = []
    all_task_names = []

    try:
        for task in tasks:
            task_examples = [ex for ex in examples if ex.task == task]
            task_loop_arrays = {f"L{i + 1}": [] for i in range(N_LOOPS)}
            feature_chunks = {name: [] for name in all_features_by_name}
            labels = []
            for start in range(0, len(task_examples), args.batch_size):
                batch = task_examples[start : start + args.batch_size]
                capture.captured.clear()
                prompts = [ex.prompt for ex in batch]
                tokenized = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                ).to(device)
                with torch.no_grad():
                    model(**tokenized, use_cache=False)
                states = capture.captured.get("hidden_states_list")
                if states is None or len(states) != N_LOOPS:
                    raise RuntimeError(
                        f"expected {N_LOOPS} loop states, got {0 if states is None else len(states)}"
                    )
                mask = tokenized["attention_mask"].float().to(device)
                pooled = []
                for state in states:
                    s = state.to(device=device, dtype=torch.float32)
                    pooled.append((s * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True))
                loop_np = [p.cpu().numpy() for p in pooled]
                for idx, arr in enumerate(loop_np):
                    task_loop_arrays[f"L{idx + 1}"].append(arr)
                    all_loop_arrays[f"L{idx + 1}"].append(arr)
                feature_chunks["L1"].append(loop_np[0])
                feature_chunks["L4"].append(loop_np[-1])
                feature_chunks["L4_minus_L1"].append(loop_np[-1] - loop_np[0])
                feature_chunks["all_loops"].append(np.concatenate(loop_np, axis=-1))
                for ex in batch:
                    labels.append(ex.label)
                    all_labels.append(ex.label)
                    all_task_names.append(task)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            task_features = {name: np.concatenate(chunks, axis=0) for name, chunks in feature_chunks.items()}
            task_labels = np.array(labels, dtype=np.int64)
            for name, arr in task_features.items():
                all_features_by_name[name].append(arr)
            probe_scores = {name: fit_probe_scores(arr, task_labels) for name, arr in task_features.items()}
            metrics = loop_metrics([np.concatenate(task_loop_arrays[f"L{i + 1}"], axis=0) for i in range(N_LOOPS)])
            per_task[task] = {
                "examples": int(len(task_labels)),
                "label_counts": {str(k): int(v) for k, v in Counter(task_labels.tolist()).items()},
                "chance": float(1.0 / len(set(task_labels.tolist()))),
                "loop_metrics": metrics,
                "probe_scores": probe_scores,
                "l4_over_l1": float(
                    probe_scores["L4"]["accuracy_mean"] - probe_scores["L1"]["accuracy_mean"]
                ),
                "delta_over_l1": float(
                    probe_scores["L4_minus_L1"]["accuracy_mean"] - probe_scores["L1"]["accuracy_mean"]
                ),
                "all_over_l1": float(
                    probe_scores["all_loops"]["accuracy_mean"] - probe_scores["L1"]["accuracy_mean"]
                ),
            }
            print(
                f"{task:<12} "
                f"L1={probe_scores['L1']['accuracy_mean']:.3f} "
                f"L4={probe_scores['L4']['accuracy_mean']:.3f} "
                f"d={probe_scores['L4_minus_L1']['accuracy_mean']:.3f} "
                f"all={probe_scores['all_loops']['accuracy_mean']:.3f} "
                f"L4-L1={per_task[task]['l4_over_l1']:+.3f}"
            )
    finally:
        handle.remove()

    result = {
        "model_path": model_path,
        "model_label": model_label,
        "device": str(device),
        "seed": args.seed,
        "tasks": tasks,
        "samples_per_task": args.samples_per_task,
        "total_examples": len(examples),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "early_exit_threshold": args.early_exit_threshold,
        "per_task": per_task,
    }

    print("\nSummary")
    for task, row in per_task.items():
        scores = row["probe_scores"]  # type: ignore[index]
        print(
            f"{task:<12} chance={row['chance']:.3f} "
            f"L1={scores['L1']['accuracy_mean']:.3f} "
            f"L4={scores['L4']['accuracy_mean']:.3f} "
            f"L4-L1={row['l4_over_l1']:+.3f} "
            f"delta={scores['L4_minus_L1']['accuracy_mean']:.3f} "
            f"all={scores['all_loops']['accuracy_mean']:.3f}"
        )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
