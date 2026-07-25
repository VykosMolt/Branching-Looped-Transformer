#!/usr/bin/env python3
"""Decompose the fixed-order evaluator across the three Ouro backbones.

Inference only. Reuses saved token-level Thinking/RLTT captures and streams
the same 200 HH-RLHF pairs through the pinned base checkpoint. Scores four
variants: attention/mean pooling crossed with trained LayerNorm on/off.
"""
from __future__ import annotations

import argparse
import csv
import gc
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

BASE_ID = "ByteDance/Ouro-2.6B"
BASE_REVISION = "1ed04250da1a9936042725d302e81c8fa2ab5abd"
THINKING_ID = "ByteDance/Ouro-2.6B-Thinking"
THINKING_REVISION = "f1edd81e7ac41355db670500ceaf204e0f73af68"
RLTT_PATH = ROOT / "models/ouro_rltt_local"
TOKENIZER_PATH = RLTT_PATH
HEAD_PATH = ROOT / "artifacts/checkpoints/evaluator/pairwise_epoch2.pt"
THINKING_STATES = ROOT / "artifacts/reports/evaluator/hh_loop_states_200_thinking.pt"
RLTT_STATES = ROOT / "artifacts/reports/evaluator/hh_loop_states_200_rltt.pt"
HF_CACHE = ROOT / "artifacts/hf_cache"
SELECTION_SEED = 42
BOOTSTRAP_SEED = 20260710
VARIANTS = (
    "attention_with_norm",
    "attention_without_norm",
    "mean_with_norm",
    "mean_without_norm",
)


def load_head(device: torch.device) -> PairwiseEvaluator:
    head = PairwiseEvaluator().to(device)
    ckpt = torch.load(HEAD_PATH, map_location=device, weights_only=False)
    head.load_state_dict(ckpt["model_state_dict"])
    head.eval()
    return head


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


@torch.no_grad()
def score_variant(
    head: PairwiseEvaluator,
    left_states: list[torch.Tensor],
    left_mask: torch.Tensor,
    right_states: list[torch.Tensor],
    right_mask: torch.Tensor,
    pooling: str,
    use_norm: bool,
) -> float:
    if pooling == "attention":
        lp = [head.attention_pool(h, left_mask) for h in left_states]
        rp = [head.attention_pool(h, right_mask) for h in right_states]
    elif pooling == "mean":
        lp = [mean_pool(h, left_mask) for h in left_states]
        rp = [mean_pool(h, right_mask) for h in right_states]
    else:
        raise ValueError(pooling)
    diffs = [l - r for l, r in zip(lp, rp)]
    inputs = [head.input_norm(d) if use_norm else d for d in diffs]
    projected = [head.input_proj(d) for d in inputs]
    seq = torch.stack(projected, dim=0)
    _, final_hidden = head.gru(seq)
    combined = torch.cat([final_hidden[-1], projected[-1]], dim=-1)
    return float(head.scorer(combined).view(-1).item())


@torch.no_grad()
def score_all_variants(
    head: PairwiseEvaluator,
    chosen_states: list[torch.Tensor],
    chosen_mask: torch.Tensor,
    rejected_states: list[torch.Tensor],
    rejected_mask: torch.Tensor,
    device: torch.device,
) -> list[dict[str, Any]]:
    cs = [x.to(device=device, dtype=torch.float32).unsqueeze(0) if x.dim() == 2 else x.to(device=device, dtype=torch.float32) for x in chosen_states]
    rs = [x.to(device=device, dtype=torch.float32).unsqueeze(0) if x.dim() == 2 else x.to(device=device, dtype=torch.float32) for x in rejected_states]
    cm = chosen_mask.to(device=device, dtype=torch.float32)
    rm = rejected_mask.to(device=device, dtype=torch.float32)
    if cm.dim() == 1:
        cm = cm.unsqueeze(0)
    if rm.dim() == 1:
        rm = rm.unsqueeze(0)
    rows = []
    for pooling in ("attention", "mean"):
        for use_norm in (True, False):
            variant = f"{pooling}_{'with' if use_norm else 'without'}_norm"
            normal = score_variant(head, cs, cm, rs, rm, pooling, use_norm)
            flipped = score_variant(head, rs, rm, cs, cm, pooling, use_norm)
            rows.append({
                "variant": variant,
                "normal_score": normal,
                "flipped_score": flipped,
                "antisym_score": 0.5 * (normal - flipped),
                "symmetric_order_score": 0.5 * (normal + flipped),
            })
    return rows


def load_saved_rows(path: Path, backbone: str, head: PairwiseEvaluator, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    rows = []
    started = time.time()
    for i, pack in enumerate(payload["packs"]):
        for row in score_all_variants(
            head,
            pack["chosen_states"], pack["chosen_mask"],
            pack["rejected_states"], pack["rejected_mask"], device,
        ):
            row.update({"backbone": backbone, "pair_index": i})
            rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"{backbone}: {i+1}/200", flush=True)
    return rows, {"source": str(path), "source_meta": payload.get("meta", {}), "elapsed_seconds": time.time() - started}


def select_examples() -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("Anthropic/hh-rlhf", split="test")
    rng = np.random.default_rng(SELECTION_SEED)
    indices = sorted(rng.permutation(len(ds))[:200].tolist())
    return [{"dataset_index": int(i), "chosen": ds[int(i)]["chosen"], "rejected": ds[int(i)]["rejected"]} for i in indices]


def capture_and_score_base(head: PairwiseEvaluator, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    examples = select_examples()
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_ID,
        revision=BASE_REVISION,
        cache_dir=str(HF_CACHE),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.config.early_exit_threshold = 1.0
    captured: list[torch.Tensor] = []

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        nonlocal captured
        captured = [x.detach() for x in output[1]]

    handle = model.model.register_forward_hook(hook)

    def run(text: str) -> tuple[list[torch.Tensor], torch.Tensor]:
        nonlocal captured
        enc = tok(text, return_tensors="pt", truncation=True, max_length=384).to(device)
        captured = []
        with torch.no_grad():
            model(**enc, use_cache=False)
        if len(captured) != 4:
            raise RuntimeError(f"base capture returned {len(captured)} loops")
        # Keep on GPU just long enough to score the pair.
        return [x.squeeze(0) for x in captured], enc["attention_mask"].squeeze(0)

    rows = []
    started = time.time()
    try:
        for i, ex in enumerate(examples):
            cs, cm = run(ex["chosen"])
            # Clone chosen states because the hook list is replaced by rejected run.
            cs = [x.clone() for x in cs]
            cm = cm.clone()
            rs, rm = run(ex["rejected"])
            for row in score_all_variants(head, cs, cm, rs, rm, device):
                row.update({"backbone": "Base Ouro 2.6B", "pair_index": i, "dataset_index": ex["dataset_index"]})
                rows.append(row)
            del cs, rs
            if (i + 1) % 20 == 0:
                print(f"Base Ouro 2.6B: {i+1}/200 elapsed={time.time()-started:.1f}s", flush=True)
    finally:
        handle.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return rows, {
        "source": "streamed token-level capture; states not persisted",
        "model_id": BASE_ID,
        "revision": BASE_REVISION,
        "dataset": "Anthropic/hh-rlhf test",
        "selection_seed": SELECTION_SEED,
        "max_length": 384,
        "early_exit_threshold": 1.0,
        "elapsed_seconds": time.time() - started,
    }


def finite_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def percentile_ci(values: np.ndarray, seed: int = BOOTSTRAP_SEED, draws: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(draws)
    for i in range(draws):
        means[i] = values[rng.integers(0, n, n)].mean()
    return np.quantile(means, [0.025, 0.975]).tolist()


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for backbone in ("Base Ouro 2.6B", "Ouro 2.6B Thinking", "Ouro RLTT local"):
        for variant in VARIANTS:
            rr = [r for r in rows if r["backbone"] == backbone and r["variant"] == variant]
            normal = np.asarray([r["normal_score"] for r in rr], dtype=np.float64)
            flipped = np.asarray([r["flipped_score"] for r in rr], dtype=np.float64)
            anti = np.asarray([r["antisym_score"] for r in rr], dtype=np.float64)
            sym = np.asarray([r["symmetric_order_score"] for r in rr], dtype=np.float64)
            canonical_correct = (normal > 0).astype(np.float64)
            swapped_correct = (flipped < 0).astype(np.float64)
            antisym_correct = (anti > 0).astype(np.float64)
            sign_flip = ((normal > 0) != (flipped > 0)).astype(np.float64)
            out.append({
                "backbone": backbone,
                "variant": variant,
                "n_pairs": len(rr),
                "canonical_accuracy": float(canonical_correct.mean()),
                "canonical_accuracy_ci95": percentile_ci(canonical_correct),
                "swapped_accuracy": float(swapped_correct.mean()),
                "antisym_accuracy": float(antisym_correct.mean()),
                "antisym_accuracy_ci95": percentile_ci(antisym_correct),
                "strict_sign_flip_rate": float(sign_flip.mean()),
                "normal_flipped_pearson": finite_corr(normal, flipped),
                "normal_mean": float(normal.mean()),
                "flipped_mean": float(flipped.mean()),
                "antisym_mean": float(anti.mean()),
                "antisym_abs_mean": float(np.abs(anti).mean()),
                "symmetric_order_mean": float(sym.mean()),
                "symmetric_order_abs_mean": float(np.abs(sym).mean()),
                "symmetric_to_antisym_abs_ratio": float(np.abs(sym).mean() / max(np.abs(anti).mean(), 1e-12)),
                "both_orders_prefer_first_rate": float(((normal > 0) & (flipped > 0)).mean()),
                "both_orders_prefer_second_rate": float(((normal < 0) & (flipped < 0)).mean()),
            })
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    head = load_head(device)

    all_rows = []
    metadata = {}
    rows, meta = load_saved_rows(THINKING_STATES, "Ouro 2.6B Thinking", head, device)
    all_rows.extend(rows); metadata["thinking"] = meta
    rows, meta = load_saved_rows(RLTT_STATES, "Ouro RLTT local", head, device)
    all_rows.extend(rows); metadata["rltt"] = meta
    rows, meta = capture_and_score_base(head, device)
    all_rows.extend(rows); metadata["base"] = meta

    summaries = summarize(all_rows)
    write_csv(out_dir / "evaluator_decomposition_rows.csv", all_rows)
    write_csv(out_dir / "evaluator_decomposition_summary.csv", summaries)
    result = {
        "question": "Where does the historical 24%/95%/95% cross-backbone pattern live: antisymmetric relational score or symmetric/order component?",
        "head_checkpoint": str(HEAD_PATH),
        "head_checkpoint_sha256": "3630c2092eca8db13239f763bc9c212f4b673866e47f811c3095efc57409ec96",
        "variants": {
            "attention_with_norm": "original checkpoint architecture",
            "attention_without_norm": "bypass trained input_norm only",
            "mean_with_norm": "replace learned attention pool by masked mean; retain input_norm",
            "mean_without_norm": "replace learned attention pool by masked mean and bypass input_norm",
            "fixed_components": "input_proj, two-layer GRU, nonlinear scorer, and all checkpoint weights",
        },
        "definitions": {
            "canonical_score": "s(chosen,rejected)",
            "swapped_score": "s(rejected,chosen)",
            "antisymmetric_component": "(canonical-swapped)/2",
            "symmetric_order_component": "(canonical+swapped)/2",
        },
        "metadata": metadata,
        "summaries": summaries,
        "rows_csv": str(out_dir / "evaluator_decomposition_rows.csv"),
        "summary_csv": str(out_dir / "evaluator_decomposition_summary.csv"),
        "environment": {
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "dtype": "backbone bf16; captures/head fp32",
            "quantization": "none",
        },
    }
    (out_dir / "evaluator_decomposition.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
