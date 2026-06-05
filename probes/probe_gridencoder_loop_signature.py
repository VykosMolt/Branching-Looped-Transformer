"""Probe Ouro loop signatures for GridEncoder ``inputs_embeds``.

This diagnostic bypasses text prompts entirely:

    integer grid -> GridEncoder tokens -> Ouro.model(inputs_embeds=...)

It compares base Ouro and a local RLTT checkpoint on the same encoded grids,
then writes compact JSON metrics for CLS, patch-mean, and token-level loop
refinement. The purpose is to test whether grid tokens produce meaningful
loop dynamics in the substrate the agent actually uses.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hunter_seeker_core.grid_encoder import GridEncoder


N_LOOPS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GridEncoder -> Ouro loop-signature diagnostic.")
    parser.add_argument("--encoder-checkpoint", default="artifacts/checkpoints/running/sprint4_encoder_reverted.pt")
    parser.add_argument("--encoder-label", default="")
    parser.add_argument(
        "--model-paths",
        default="ByteDance/Ouro-2.6B-Thinking,models/ouro_rltt_local",
        help="Comma-separated base/RLTT model paths.",
    )
    parser.add_argument("--model-labels", default="base,rltt")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--n-values", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--early-exit-threshold", type=float, default=1.0)
    parser.add_argument("--output", default="artifacts/reports/evaluator/probe_gridencoder_loop_signature.json")
    parser.add_argument("--allow-remote", action="store_true")
    return parser.parse_args()


def add_rectangle(
    grid: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    value: int,
    *,
    hollow: bool = False,
) -> None:
    bottom = min(grid.shape[0], top + height)
    right = min(grid.shape[1], left + width)
    if bottom <= top or right <= left:
        return
    if hollow and bottom - top >= 3 and right - left >= 3:
        grid[top:bottom, left] = value
        grid[top:bottom, right - 1] = value
        grid[top, left:right] = value
        grid[bottom - 1, left:right] = value
    else:
        grid[top:bottom, left:right] = value


def make_object_grid(rng: np.random.RandomState, size: int, n_values: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=np.int64)
    for _ in range(int(rng.randint(3, 8))):
        height = int(rng.randint(3, max(4, size // 5)))
        width = int(rng.randint(3, max(4, size // 5)))
        top = int(rng.randint(1, max(2, size - height - 1)))
        left = int(rng.randint(1, max(2, size - width - 1)))
        value = int(rng.randint(1, n_values))
        add_rectangle(grid, top, left, height, width, value, hollow=bool(rng.rand() < 0.35))
    return grid


def make_path_grid(rng: np.random.RandomState, size: int, n_values: int) -> np.ndarray:
    grid = np.ones((size, size), dtype=np.int64)
    r = int(rng.randint(4, size - 4))
    c = int(rng.randint(4, size - 4))
    grid[r, c] = 0
    for _ in range(size * 2):
        if rng.rand() < 0.5:
            r = int(np.clip(r + rng.choice([-1, 1]), 1, size - 2))
        else:
            c = int(np.clip(c + rng.choice([-1, 1]), 1, size - 2))
        grid[r, c] = 0
    grid[1, 1] = 2
    grid[size - 2, size - 2] = 3
    for _ in range(size // 2):
        rr = int(rng.randint(1, size - 1))
        cc = int(rng.randint(1, size - 1))
        if grid[rr, cc] == 0:
            grid[rr, cc] = int(rng.randint(4, n_values))
    return grid


def make_symmetry_grid(rng: np.random.RandomState, size: int, n_values: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=np.int64)
    half = size // 2
    mask = rng.rand(size, half) < 0.08
    vals = rng.randint(1, n_values, size=(size, half))
    grid[:, :half][mask] = vals[mask]
    grid[:, half:] = np.fliplr(grid[:, : size - half])
    if rng.rand() < 0.5:
        for _ in range(int(rng.randint(3, 8))):
            rr = int(rng.randint(0, size))
            cc = int(rng.randint(half, size))
            grid[rr, cc] = int(rng.randint(0, n_values))
    return grid


def make_collision_grid(rng: np.random.RandomState, size: int, n_values: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=np.int64)
    row = int(rng.randint(8, size - 8))
    left = int(rng.randint(4, size // 3))
    right = int(rng.randint(size // 2, size - 8))
    grid[row - 1 : row + 2, left : left + 3] = 4
    grid[row - 1 : row + 2, right : right + 3] = 7
    grid[row, left + 3 : right] = 1
    for _ in range(int(rng.randint(4, 9))):
        rr = int(rng.randint(1, size - 4))
        cc = int(rng.randint(1, size - 4))
        value = int(rng.randint(2, n_values))
        add_rectangle(grid, rr, cc, 2, 2, value)
    return grid


def make_grids(samples: int, size: int, n_values: int, seed: int) -> Tuple[np.ndarray, List[str]]:
    rng = np.random.RandomState(seed)
    makers = (
        ("objects", make_object_grid),
        ("path", make_path_grid),
        ("symmetry", make_symmetry_grid),
        ("collision", make_collision_grid),
    )
    grids = []
    families = []
    for idx in range(samples):
        family, maker = makers[idx % len(makers)]
        grids.append(maker(rng, size, n_values))
        families.append(family)
    return np.stack(grids, axis=0), families


def load_encoder(path: str, n_values: int, device: torch.device) -> Tuple[GridEncoder, Dict[str, List[str]]]:
    encoder = GridEncoder(n_values=n_values, patch_size=4, d_model=2048).to(device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("encoder", checkpoint)
    result = encoder.load_state_dict(state, strict=False)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    load_report = {
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }
    return encoder, load_report


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


def pool_states(states: Sequence[torch.Tensor], mask: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
    patch_mask = mask[:, 1:]
    return {
        "cls": [s[:, 0].float() for s in states],
        "patch_mean": [masked_mean(s[:, 1:].float(), patch_mask) for s in states],
        "sequence_mean": [masked_mean(s.float(), mask) for s in states],
    }


def summarize_vectors(vectors: Sequence[torch.Tensor]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for idx in range(len(vectors) - 1):
        cos = F.cosine_similarity(vectors[idx], vectors[idx + 1], dim=-1)
        delta = (vectors[idx + 1] - vectors[idx]).norm(dim=-1)
        metrics[f"l{idx + 1}_l{idx + 2}_cosine_mean"] = float(cos.mean().item())
        metrics[f"l{idx + 1}_l{idx + 2}_cosine_std"] = float(cos.std(unbiased=False).item())
        metrics[f"l{idx + 1}_l{idx + 2}_delta_mean"] = float(delta.mean().item())
        metrics[f"l{idx + 1}_l{idx + 2}_delta_std"] = float(delta.std(unbiased=False).item())
    cos_l1_l4 = F.cosine_similarity(vectors[0], vectors[-1], dim=-1)
    metrics["l1_l4_cosine_mean"] = float(cos_l1_l4.mean().item())
    metrics["l1_l4_cosine_std"] = float(cos_l1_l4.std(unbiased=False).item())
    metrics["total_refinement_delta_mean"] = float(
        sum((vectors[idx + 1] - vectors[idx]).norm(dim=-1).mean().item() for idx in range(len(vectors) - 1))
    )
    for idx, vector in enumerate(vectors):
        norm = vector.norm(dim=-1)
        metrics[f"l{idx + 1}_norm_mean"] = float(norm.mean().item())
        metrics[f"l{idx + 1}_norm_std"] = float(norm.std(unbiased=False).item())
    return metrics


def summarize_token_dynamics(states: Sequence[torch.Tensor], mask: torch.Tensor) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    m = mask.to(device=states[0].device, dtype=torch.float32)
    denom = m.sum().clamp_min(1.0)
    for idx in range(len(states) - 1):
        a = states[idx].float()
        b = states[idx + 1].float()
        cos = F.cosine_similarity(a, b, dim=-1)
        delta = (b - a).norm(dim=-1)
        metrics[f"l{idx + 1}_l{idx + 2}_token_cosine_mean"] = float((cos * m).sum().item() / denom.item())
        metrics[f"l{idx + 1}_l{idx + 2}_token_delta_mean"] = float((delta * m).sum().item() / denom.item())
    cos_l1_l4 = F.cosine_similarity(states[0].float(), states[-1].float(), dim=-1)
    metrics["l1_l4_token_cosine_mean"] = float((cos_l1_l4 * m).sum().item() / denom.item())
    metrics["total_token_refinement_delta_mean"] = float(
        sum(
            (((states[idx + 1].float() - states[idx].float()).norm(dim=-1) * m).sum().item() / denom.item())
            for idx in range(len(states) - 1)
        )
    )
    return metrics


@torch.no_grad()
def run_model(
    model_path: str,
    model_label: str,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    families: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    print(f"Loading {model_label}: {model_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
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

    pooled_chunks: Dict[str, List[List[torch.Tensor]]] = {
        "cls": [[] for _ in range(N_LOOPS)],
        "patch_mean": [[] for _ in range(N_LOOPS)],
        "sequence_mean": [[] for _ in range(N_LOOPS)],
    }
    token_metrics_by_family: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    token_metric_chunks: List[Dict[str, float]] = []

    for start in range(0, tokens.shape[0], args.batch_size):
        end = min(tokens.shape[0], start + args.batch_size)
        batch_tokens = tokens[start:end].to(device=device, dtype=torch.bfloat16)
        batch_mask = mask[start:end].to(device=device)
        out = model.model(inputs_embeds=batch_tokens, attention_mask=batch_mask, use_cache=False)
        states = [s.detach() for s in out[1]]
        if len(states) != N_LOOPS:
            raise RuntimeError(f"{model_label}: expected {N_LOOPS} loop states, got {len(states)}")
        pooled = pool_states(states, batch_mask)
        for scope, vectors in pooled.items():
            for idx, vector in enumerate(vectors):
                pooled_chunks[scope][idx].append(vector.cpu())
        token_metrics = summarize_token_dynamics(states, batch_mask)
        token_metric_chunks.append(token_metrics)
        batch_families = families[start:end]
        for family in sorted(set(batch_families)):
            indices = [idx for idx, value in enumerate(batch_families) if value == family]
            if not indices:
                continue
            family_states = [s[indices] for s in states]
            family_mask = batch_mask[indices]
            token_metrics_by_family[family].append(summarize_token_dynamics(family_states, family_mask))
        del out, states, batch_tokens, batch_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    loop_vectors: Dict[str, List[torch.Tensor]] = {
        scope: [torch.cat(chunks, dim=0) for chunks in per_loop]
        for scope, per_loop in pooled_chunks.items()
    }
    summary: Dict[str, object] = {
        "model_path": model_path,
        "model_label": model_label,
        "early_exit_threshold": args.early_exit_threshold,
        "scopes": {scope: summarize_vectors(vectors) for scope, vectors in loop_vectors.items()},
        "token_level": mean_metric_dicts(token_metric_chunks),
        "token_level_by_family": {
            family: mean_metric_dicts(chunks) for family, chunks in sorted(token_metrics_by_family.items())
        },
    }

    summary["_loop_vectors"] = loop_vectors
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def mean_metric_dicts(chunks: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not chunks:
        return {}
    keys = sorted({key for chunk in chunks for key in chunk})
    return {key: float(np.mean([chunk[key] for chunk in chunks if key in chunk])) for key in keys}


def compare_models(model_outputs: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    labels = list(model_outputs)
    if len(labels) < 2:
        return {}
    first, second = labels[0], labels[1]
    a = model_outputs[first]["_loop_vectors"]  # type: ignore[index]
    b = model_outputs[second]["_loop_vectors"]  # type: ignore[index]
    comparisons: Dict[str, object] = {"models": [first, second], "scopes": {}}
    for scope in sorted(a):
        scope_metrics: Dict[str, float] = {}
        for idx in range(N_LOOPS):
            cos = F.cosine_similarity(a[scope][idx].float(), b[scope][idx].float(), dim=-1)
            scope_metrics[f"l{idx + 1}_cross_model_cosine_mean"] = float(cos.mean().item())
            scope_metrics[f"l{idx + 1}_cross_model_cosine_std"] = float(cos.std(unbiased=False).item())
        delta_a = a[scope][-1].float() - a[scope][0].float()
        delta_b = b[scope][-1].float() - b[scope][0].float()
        delta_cos = F.cosine_similarity(delta_a, delta_b, dim=-1)
        scope_metrics["l4_minus_l1_cross_model_cosine_mean"] = float(delta_cos.mean().item())
        scope_metrics["l4_minus_l1_cross_model_cosine_std"] = float(delta_cos.std(unbiased=False).item())
        comparisons["scopes"][scope] = scope_metrics  # type: ignore[index]
    return comparisons


def clean_output(summary: Dict[str, object]) -> Dict[str, object]:
    cleaned = dict(summary)
    cleaned.pop("_loop_vectors", None)
    return cleaned


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This diagnostic is intended to run on CUDA; CUDA is not available.")

    model_paths = [value.strip() for value in args.model_paths.split(",") if value.strip()]
    model_labels = [value.strip() for value in args.model_labels.split(",") if value.strip()]
    if len(model_paths) != len(model_labels):
        raise ValueError("--model-paths and --model-labels must have the same number of entries")

    grids_np, families = make_grids(args.samples, args.grid_size, args.n_values, args.seed)
    grids = torch.from_numpy(grids_np).long().to(device)
    encoder, encoder_load_report = load_encoder(args.encoder_checkpoint, args.n_values, device)
    with torch.no_grad():
        tokens, mask, patch_grid_size = encoder.encode_for_ouro(grids)
    tokens = tokens.detach().cpu()
    mask = mask.detach().cpu()
    del encoder, grids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Encoded {args.samples} grids as tokens {tuple(tokens.shape)}.", flush=True)
    print(f"Patch grid size: {patch_grid_size}", flush=True)
    print(f"Encoder load: {encoder_load_report}", flush=True)

    model_outputs: Dict[str, Dict[str, object]] = {}
    for model_path, model_label in zip(model_paths, model_labels):
        model_outputs[model_label] = run_model(model_path, model_label, tokens, mask, families, args, device)

    comparison = compare_models(model_outputs)
    result = {
        "encoder_checkpoint": args.encoder_checkpoint,
        "encoder_label": args.encoder_label or Path(args.encoder_checkpoint).stem,
        "encoder_load_report": encoder_load_report,
        "samples": args.samples,
        "grid_size": args.grid_size,
        "n_values": args.n_values,
        "seed": args.seed,
        "families": dict(sorted((name, families.count(name)) for name in set(families))),
        "token_shape": list(tokens.shape),
        "patch_grid_size": list(patch_grid_size),
        "models": {label: clean_output(summary) for label, summary in model_outputs.items()},
        "cross_model_comparison": comparison,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True)[:6000], flush=True)
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
