"""Spatial/grid loop-state probe for a local RLTT/Ouro model.

This ports the archived ``probe_spatial.py`` diagnostic to the converted local
RLTT model. It encodes synthetic grids as text, captures the four loop states,
and checks whether later loops carry spatial-pattern information.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator_core.pairwise_evaluator import validate_hook_output


DEFAULT_MODEL_PATH = "models/ouro_rltt_local"
N_LOOPS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe RLTT loop states on text-encoded spatial grids.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--grid-samples", type=int, default=100)
    parser.add_argument("--comparison-samples", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--early-exit-threshold", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


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


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(device_arg)


def grid_to_text(grid: List[List[int]]) -> str:
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def grid_to_prompt(grid: List[List[int]], mode: str = "raw") -> str:
    if mode == "raw":
        return grid_to_text(grid)
    if mode == "described":
        h, w = len(grid), len(grid[0])
        return f"This is a {h}x{w} grid:\n{grid_to_text(grid)}"
    if mode == "task":
        return f"Input grid:\n{grid_to_text(grid)}\n\nWhat pattern do you see?"
    raise ValueError(f"unknown grid prompt mode: {mode}")


def generate_synthetic_grids(n: int = 100) -> List[Dict[str, object]]:
    grids: List[Dict[str, object]] = []
    rng = np.random.RandomState(42)
    for idx in range(n):
        h = int(rng.randint(3, 10))
        w = int(rng.randint(3, 10))
        pattern = idx % 5
        if pattern == 0:
            grid = rng.randint(0, 10, (h, w)).tolist()
        elif pattern == 1:
            grid = [[row_idx % 3 for _ in range(w)] for row_idx in range(h)]
        elif pattern == 2:
            grid = [[col_idx % 3 for col_idx in range(w)] for _ in range(h)]
        elif pattern == 3:
            color = int(rng.randint(1, 10))
            grid = [[color for _ in range(w)] for _ in range(h)]
        else:
            grid = [[1 if row_idx == col_idx else 0 for col_idx in range(w)] for row_idx in range(h)]
        grids.append(
            {
                "grid": grid,
                "pattern": pattern,
                "h": h,
                "w": w,
                "n_colors": len({c for row in grid for c in row}),
            }
        )
    return grids


def generate_text_samples() -> List[str]:
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The mitochondria is the powerhouse of the cell.",
        "Python is a popular programming language for data science.",
        "Gravity causes objects to fall toward the center of the Earth.",
        "Shakespeare wrote Romeo and Juliet in the 16th century.",
        "Photosynthesis converts sunlight into chemical energy.",
        "The speed of light is approximately 300,000 km per second.",
        "Neural networks are inspired by the structure of the brain.",
        "DNA carries the genetic instructions for life.",
        "The Earth orbits the Sun once every 365.25 days.",
        "Electrons have a negative electrical charge.",
        "The human body contains approximately 37 trillion cells.",
        "Algorithms are step-by-step procedures for solving problems.",
        "The universe is approximately 13.8 billion years old.",
        "Quantum mechanics describes the behavior of subatomic particles.",
        "The Pythagorean theorem relates the sides of a right triangle.",
        "Evolution is driven by natural selection and genetic variation.",
        "Artificial intelligence aims to create systems that can reason.",
    ]


@torch.no_grad()
def extract_loop_states(
    text: str,
    tokenizer,
    model,
    capture: LoopStateCapture,
    device: torch.device,
    max_length: int,
) -> Optional[List[torch.Tensor]]:
    capture.captured.clear()
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)
    model(**tokens, use_cache=False)
    states = capture.captured.get("hidden_states_list")
    if states is None or len(states) != N_LOOPS:
        return None
    mask = tokens["attention_mask"].float().to(device)
    pooled = []
    for state in states:
        s = state.to(device=device, dtype=torch.float32)
        masked = s * mask.unsqueeze(-1)
        pooled.append(masked.sum(dim=1) / mask.sum(dim=1, keepdim=True))
    return pooled


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).item())


def compute_divergence(pooled: List[torch.Tensor]) -> Dict[str, float]:
    sims: Dict[str, float] = {}
    for i in range(len(pooled)):
        for j in range(i + 1, len(pooled)):
            sims[f"L{i + 1}-L{j + 1}"] = cosine_sim(pooled[i], pooled[j])
    return sims


def compute_deltas(pooled: List[torch.Tensor]) -> List[float]:
    return [float((pooled[j + 1] - pooled[j]).norm().item()) for j in range(len(pooled) - 1)]


def compute_norms(pooled: List[torch.Tensor]) -> List[float]:
    return [float(p.norm().item()) for p in pooled]


def classification_scores(features: Dict[str, List[np.ndarray]], y: np.ndarray, n_pca: int) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for key in [f"L{j + 1}" for j in range(N_LOOPS)]:
        x = np.stack(features[key])
        x_pca = PCA(n_components=n_pca).fit_transform(x)
        clf = LogisticRegression(max_iter=1000)
        scores = cross_val_score(clf, x_pca, y, cv=5, scoring="accuracy")
        out[key] = {"mean": float(scores.mean()), "std": float(scores.std())}
    return out


def main() -> None:
    args = parse_args()
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_MODULES_CACHE", str(PROJECT_ROOT / ".hf_modules_cache"))

    device = resolve_device(args.device)
    model_path = str(Path(args.model_path))
    print(f"Transformers model: {model_path}")
    print(f"Device: {device}")

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

    synth_grids = generate_synthetic_grids(args.grid_samples)
    text_samples = generate_text_samples()[: args.comparison_samples]
    probe_grids = synth_grids[: args.comparison_samples]

    print("\nExtracting comparison loop states...")
    grid_pooled_list = []
    for grid_info in probe_grids:
        pooled = extract_loop_states(
            grid_to_prompt(grid_info["grid"], mode="described"),  # type: ignore[arg-type]
            tokenizer,
            model,
            capture,
            device,
            args.max_length,
        )
        if pooled is not None:
            grid_pooled_list.append(pooled)

    text_pooled_list = []
    for text in text_samples:
        pooled = extract_loop_states(text, tokenizer, model, capture, device, args.max_length)
        if pooled is not None:
            text_pooled_list.append(pooled)

    if len(grid_pooled_list) < 5 or len(text_pooled_list) < 5:
        raise RuntimeError("Too few inputs produced all four loop states")

    grid_sims = [compute_divergence(p) for p in grid_pooled_list]
    text_sims = [compute_divergence(p) for p in text_pooled_list]
    grid_norms = [compute_norms(p) for p in grid_pooled_list]
    text_norms = [compute_norms(p) for p in text_pooled_list]
    grid_deltas = [compute_deltas(p) for p in grid_pooled_list]
    text_deltas = [compute_deltas(p) for p in text_pooled_list]

    print("\nProbing synthetic grid properties...")
    features = {f"L{j + 1}": [] for j in range(N_LOOPS)}
    properties = {"pattern": [], "n_colors": [], "h": [], "w": []}
    for idx, grid_info in enumerate(synth_grids):
        pooled = extract_loop_states(
            grid_to_prompt(grid_info["grid"], mode="raw"),  # type: ignore[arg-type]
            tokenizer,
            model,
            capture,
            device,
            args.max_length,
        )
        if pooled is None:
            continue
        for j, p in enumerate(pooled):
            features[f"L{j + 1}"].append(p.squeeze(0).cpu().numpy())
        for key in properties:
            properties[key].append(grid_info[key])
        if (idx + 1) % 25 == 0:
            print(f"  Processed {idx + 1}/{len(synth_grids)} synthetic grids")

    handle.remove()

    n_probed = len(properties["pattern"])
    if n_probed < 20:
        raise RuntimeError("Too few synthetic grids for probing")
    n_pca = min(50, n_probed - 2)
    y_pattern = np.array(properties["pattern"])
    y_n_colors = np.digitize(np.array(properties["n_colors"]), bins=[2, 4, 6])

    pattern_scores = classification_scores(features, y_pattern, n_pca)
    n_color_scores = classification_scores(features, y_n_colors, n_pca)

    divergence = {}
    for key in grid_sims[0]:
        divergence[key] = {
            "grid_mean": float(np.mean([s[key] for s in grid_sims])),
            "text_mean": float(np.mean([s[key] for s in text_sims])),
            "diff": float(np.mean([s[key] for s in grid_sims]) - np.mean([s[key] for s in text_sims])),
        }
    grid_total_delta = float(np.mean([sum(d) for d in grid_deltas]))
    text_total_delta = float(np.mean([sum(d) for d in text_deltas]))
    l1_acc = pattern_scores["L1"]["mean"]
    l4_acc = pattern_scores["L4"]["mean"]

    result = {
        "model_path": model_path,
        "device": str(device),
        "grid_comparison_count": len(grid_pooled_list),
        "text_comparison_count": len(text_pooled_list),
        "synthetic_grid_count": n_probed,
        "divergence": divergence,
        "grid_norm_means": [float(np.mean([n[j] for n in grid_norms])) for j in range(N_LOOPS)],
        "text_norm_means": [float(np.mean([n[j] for n in text_norms])) for j in range(N_LOOPS)],
        "grid_delta_means": [float(np.mean([d[j] for d in grid_deltas])) for j in range(N_LOOPS - 1)],
        "text_delta_means": [float(np.mean([d[j] for d in text_deltas])) for j in range(N_LOOPS - 1)],
        "grid_total_delta": grid_total_delta,
        "text_total_delta": text_total_delta,
        "pattern_scores": pattern_scores,
        "n_color_scores": n_color_scores,
        "pattern_l1_l4_gain": float(l4_acc - l1_acc),
    }

    print("\nLoop divergence")
    print(f"{'Pair':<8} {'Grid':>10} {'Text':>10} {'Diff':>10}")
    for key, row in divergence.items():
        print(f"{key:<8} {row['grid_mean']:>10.6f} {row['text_mean']:>10.6f} {row['diff']:>+10.6f}")

    print("\nTotal refinement")
    print(f"Grid: {grid_total_delta:.2f}")
    print(f"Text: {text_total_delta:.2f}")

    print("\nPattern classification")
    for key, row in pattern_scores.items():
        print(f"{key}: {row['mean']:.4f} +/- {row['std']:.4f}")
    print(f"L1 -> L4 gain: {result['pattern_l1_l4_gain']:+.4f}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
