#!/usr/bin/env python3
"""Streaming full-test strict antisymmetry audit for pairwise_epoch2.pt.

Inference only: frozen local RLTT backbone, original evaluator checkpoint,
both candidate orders, no generation and no feature dump. Progress is appended
to CSV and the run resumes by dataset index.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from evaluator_core.pairwise_evaluator import PairwiseEvaluator

MODEL_PATH = ROOT / "models/ouro_rltt_local"
HEAD_PATH = ROOT / "artifacts/checkpoints/evaluator/pairwise_epoch2.pt"
MODEL_SHARD_SHA256 = [
    "9f49e55b0d0ea368f2942cfd3fa256836ea7fa9002c6f860ca2d85ac1de252a1",
    "97a0d531a2c624dd98e60c0ebc46a7cc0413642f8cba6df7dac8df01a432df19",
    "0bf3d0c111c876ccc7f7ebaeb7cabc8dfccffde30acb4cd780a4c364d3679aad",
]
HEAD_SHA256 = "3630c2092eca8db13239f763bc9c212f4b673866e47f811c3095efc57409ec96"


def load_done(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        return {int(r["dataset_index"]) for r in csv.DictReader(f)}


def score_batch(
    head: PairwiseEvaluator,
    chosen: list[torch.Tensor],
    chosen_mask: torch.Tensor,
    rejected: list[torch.Tensor],
    rejected_mask: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        normal = head(chosen, chosen_mask, rejected, rejected_mask).view(-1)
        flipped = head(rejected, rejected_mask, chosen, chosen_mask).view(-1)
    return normal.float().cpu().numpy(), flipped.float().cpu().numpy()


def bootstrap_ci(values: np.ndarray, seed: int, draws: int) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        means[i] = values[rng.integers(0, n, n)].mean()
    return np.quantile(means, [0.025, 0.975]).tolist()


def summarize(rows_path: Path, args: argparse.Namespace, elapsed: float) -> dict[str, Any]:
    with rows_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    normal = np.asarray([float(r["normal_score"]) for r in rows])
    flipped = np.asarray([float(r["flipped_score"]) for r in rows])
    anti = 0.5 * (normal - flipped)
    sym = 0.5 * (normal + flipped)
    canonical = (normal > 0).astype(np.float64)
    swapped_correct = (flipped < 0).astype(np.float64)
    anti_correct = (anti > 0).astype(np.float64)
    sign_flip = ((normal > 0) != (flipped > 0)).astype(np.float64)
    return {
        "audit": "full_hh_strict_antisymmetry",
        "status": "COMPLETE" if len(rows) == args.expected_pairs else "PARTIAL",
        "n_pairs": len(rows),
        "expected_pairs": args.expected_pairs,
        "dataset": "Anthropic/hh-rlhf test",
        "dataset_revision": "cached datasets builder 09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa",
        "backbone": str(MODEL_PATH),
        "backbone_revision": "local checkpoint; shard hashes pinned below",
        "model_shard_sha256": MODEL_SHARD_SHA256,
        "evaluator": str(HEAD_PATH),
        "evaluator_sha256": HEAD_SHA256,
        "tokenizer": str(MODEL_PATH),
        "tokenizer_settings": {"padding_side": "left", "truncation": True},
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "length_bucketed": args.length_bucket,
        "early_exit_threshold": args.early_exit_threshold,
        "model_forward_dtype": "bfloat16",
        "evaluator_dtype": "float32",
        "quantization": "none",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "transformers_version": importlib.metadata.version("transformers"),
        "datasets_version": importlib.metadata.version("datasets"),
        "bootstrap_seed": args.seed,
        "bootstrap_draws": args.bootstrap_draws,
        "metrics": {
            "fixed_order_accuracy": float(canonical.mean()),
            "fixed_order_accuracy_ci95": bootstrap_ci(canonical, args.seed, args.bootstrap_draws),
            "swapped_direction_accuracy": float(swapped_correct.mean()),
            "strict_antisym_accuracy": float(anti_correct.mean()),
            "strict_antisym_accuracy_ci95": bootstrap_ci(anti_correct, args.seed, args.bootstrap_draws),
            "strict_sign_flip_rate": float(sign_flip.mean()),
            "normal_flipped_pearson": float(np.corrcoef(normal, flipped)[0, 1]),
            "normal_mean": float(normal.mean()),
            "flipped_mean": float(flipped.mean()),
            "antisym_mean": float(anti.mean()),
            "antisym_std": float(anti.std()),
            "symmetric_order_mean": float(sym.mean()),
            "symmetric_order_std": float(sym.std()),
            "symmetric_order_abs_mean": float(np.abs(sym).mean()),
            "antisym_abs_mean": float(np.abs(anti).mean()),
            "symmetric_to_antisym_abs_ratio": float(np.abs(sym).mean() / max(np.abs(anti).mean(), 1e-12)),
            "both_orders_prefer_first_rate": float(((normal > 0) & (flipped > 0)).mean()),
            "both_orders_prefer_second_rate": float(((normal < 0) & (flipped < 0)).mean()),
        },
        "elapsed_seconds_this_invocation": elapsed,
        "rows_csv": str(rows_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=768)
    ap.add_argument("--early-exit-threshold", type=float, default=1.0)
    ap.add_argument("--expected-pairs", type=int, default=8552)
    ap.add_argument("--max-pairs", type=int, default=0, help="0 means all expected pairs")
    ap.add_argument("--bootstrap-draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260710)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--length-bucket", action="store_true", help="Sort pending pairs by approximate text length to reduce padding; metrics remain indexed by dataset row.")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "full_hh_antisymmetry_rows.csv"
    result_path = out_dir / "full_hh_antisymmetry.json"
    done = load_done(rows_path)
    ds = load_dataset("Anthropic/hh-rlhf", split="test")
    if len(ds) != args.expected_pairs:
        raise RuntimeError(f"expected {args.expected_pairs} test pairs, found {len(ds)}")
    limit = args.expected_pairs if args.max_pairs <= 0 else min(args.max_pairs, args.expected_pairs)
    pending = [i for i in range(limit) if i not in done]
    if args.length_bucket:
        pending.sort(key=lambda i: max(len(ds[i]["chosen"]), len(ds[i]["rejected"])))
    print(f"full HH audit: done={len(done)} pending={len(pending)} limit={limit}", flush=True)
    if not pending:
        result = summarize(rows_path, args, 0.0)
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        return

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), torch_dtype=torch.bfloat16,
        trust_remote_code=True, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.config.early_exit_threshold = args.early_exit_threshold
    captured: list[torch.Tensor] = []

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        nonlocal captured
        captured = [x.detach() for x in output[1]]

    handle = model.model.register_forward_hook(hook)
    head = PairwiseEvaluator().to(device)
    ckpt = torch.load(HEAD_PATH, map_location=device, weights_only=False)
    head.load_state_dict(ckpt["model_state_dict"])
    head.eval()
    write_header = not rows_path.exists()
    started = time.time()
    processed = 0
    try:
        with rows_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset_index", "normal_score", "flipped_score"])
            if write_header:
                writer.writeheader()
            for start in range(0, len(pending), args.batch_size):
                indices = pending[start:start + args.batch_size]
                chosen_texts = [ds[i]["chosen"] for i in indices]
                rejected_texts = [ds[i]["rejected"] for i in indices]
                tc = tok(chosen_texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length).to(device)
                captured = []
                with torch.no_grad():
                    model(**tc, use_cache=False)
                chosen_states = [x.float() for x in captured]
                chosen_mask = tc["attention_mask"].float()
                tr = tok(rejected_texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length).to(device)
                captured = []
                with torch.no_grad():
                    model(**tr, use_cache=False)
                rejected_states = [x.float() for x in captured]
                rejected_mask = tr["attention_mask"].float()
                normal, flipped = score_batch(head, chosen_states, chosen_mask, rejected_states, rejected_mask)
                for idx, sn, sf in zip(indices, normal, flipped):
                    writer.writerow({"dataset_index": idx, "normal_score": float(sn), "flipped_score": float(sf)})
                f.flush()
                processed += len(indices)
                if processed % 100 == 0 or processed == len(pending):
                    elapsed = time.time() - started
                    rate = processed / elapsed
                    eta = (len(pending) - processed) / max(rate, 1e-9)
                    print(f"processed={processed}/{len(pending)} total_done={len(done)+processed} rate={rate:.2f} pairs/s eta={eta/60:.1f}m", flush=True)
    finally:
        handle.remove()

    result = summarize(rows_path, args, time.time() - started)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
