#!/usr/bin/env python3
"""Resumable, source-pair-disjoint HH pointwise linear probe.

This is the clean counterpart to the historical ``probe_test.py`` independent
classification control.  A source pair is assigned wholly to train or eval
before its chosen/rejected candidates are expanded into classification rows.
The extractor retains individual pooled representations (not only their
difference), allowing both the historical final-boundary feature and the
four-boundary feature used by the powered relational probe to be evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from powered_hh_pair_disjoint_probe import MODEL_SPECS, ROOT, SEED, indices, masked_mean


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(args: argparse.Namespace) -> None:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.output_dir).resolve()
    root = out / f"hh_{args.backbone}_{args.pairs}_pointwise_states"
    root.mkdir(parents=True, exist_ok=True)
    selected = indices(out, args.pairs)
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    existing = sorted(root.glob("states_*.pt"))
    done = sum(
        torch.load(path, map_location="cpu", weights_only=False)["states"].shape[0]
        for path in existing
    )
    print(
        f"backbone={args.backbone} pairs={args.pairs} done={done} pending={args.pairs - done}",
        flush=True,
    )
    if done >= args.pairs:
        return

    model_ref, revision = MODEL_SPECS[args.backbone]
    local = args.backbone == "rltt"
    tokenizer = AutoTokenizer.from_pretrained(
        str(ROOT / "models/ouro_rltt_local"),
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.padding_side = "left"
    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "local_files_only": True,
    }
    if not local:
        kwargs.update(cache_dir=str(ROOT / "artifacts/hf_cache"), revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_ref, **kwargs).to("cuda").eval()
    model.config.early_exit_threshold = 1.0
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output) -> None:
        nonlocal captured
        captured = [hidden.detach() for hidden in output[1]]

    handle = model.model.register_forward_hook(hook)
    buffer: list[torch.Tensor] = []
    buffer_positions: list[int] = []
    shard = len(existing)
    started = time.time()
    try:
        for window_start in range(done, args.pairs, args.sort_window):
            window_end = min(window_start + args.sort_window, args.pairs)
            positions = list(range(window_start, window_end))
            positions.sort(
                key=lambda position: max(
                    len(dataset[selected[position]]["chosen"]),
                    len(dataset[selected[position]]["rejected"]),
                )
            )
            for offset in range(0, len(positions), args.batch_pairs):
                batch_positions = positions[offset : offset + args.batch_pairs]
                texts = []
                for position in batch_positions:
                    dataset_index = selected[position]
                    texts.extend(
                        [dataset[dataset_index]["chosen"], dataset[dataset_index]["rejected"]]
                    )
                encoded = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=384,
                ).to("cuda")
                captured = []
                with torch.inference_mode():
                    model(**encoded, use_cache=False)
                if len(captured) != 4:
                    raise RuntimeError(f"expected four loop boundaries, got {len(captured)}")
                pooled_candidates = torch.stack(
                    [masked_mean(hidden, encoded["attention_mask"]) for hidden in captured],
                    dim=1,
                )
                pooled_pairs = pooled_candidates.reshape(len(batch_positions), 2, 4, -1)
                buffer.extend(pair.half().cpu() for pair in pooled_pairs)
                buffer_positions.extend(batch_positions)
            if len(buffer) >= args.shard_pairs or window_end == args.pairs:
                path = root / f"states_{shard:04d}.pt"
                temporary = path.with_suffix(".tmp")
                array = torch.stack(buffer)
                torch.save(
                    {
                        "states": array,
                        "positions": buffer_positions,
                        "dataset_indices": [selected[position] for position in buffer_positions],
                        "candidate_order": ["chosen", "rejected"],
                        "loop_boundaries": 4,
                        "backbone": args.backbone,
                        "model": model_ref,
                        "revision": revision,
                        "max_length": 384,
                    },
                    temporary,
                )
                temporary.replace(path)
                rate = (window_end - done) / max(time.time() - started, 1e-9)
                print(
                    f"wrote {path.name} total={window_end}/{args.pairs} rate={rate:.2f} pairs/s",
                    flush=True,
                )
                shard += 1
                buffer = []
                buffer_positions = []
    finally:
        handle.remove()
        model.to("cpu")
        del model, tokenizer
        torch.cuda.empty_cache()

    files = sorted(root.glob("states_*.pt"))
    write_json(
        root / "manifest.json",
        {
            "status": "COMPLETE",
            "backbone": args.backbone,
            "pairs": args.pairs,
            "shards": [{"file": path.name, "sha256": sha256(path)} for path in files],
            "model": model_ref,
            "revision": revision,
            "candidate_order": ["chosen", "rejected"],
            "feature_shape_per_pair": [2, 4, 2048],
            "dtype": "bfloat16 forward; float16 saved",
            "quantization": "none",
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "pair_selection_seed": SEED,
        },
    )


def fit_variant(
    states: torch.Tensor,
    train_positions: np.ndarray,
    eval_positions: np.ndarray,
    variant: str,
) -> tuple[dict, dict]:
    if variant == "final_boundary":
        features = states[:, :, -1, :]
    elif variant == "all_boundaries_concat":
        features = states.flatten(2)
    else:
        raise ValueError(variant)

    train = features[train_positions].reshape(-1, features.shape[-1])
    evaluation = features[eval_positions].reshape(-1, features.shape[-1])
    labels_train = torch.tensor([1.0, -1.0]).repeat(len(train_positions))
    labels_eval = torch.tensor([1.0, -1.0]).repeat(len(eval_positions))
    weight = torch.zeros(train.shape[1], requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weight, bias],
        lr=1,
        max_iter=200,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        scores = train @ weight + bias
        loss = F.softplus(-labels_train * scores).mean()
        loss = loss + 0.5 * (weight @ weight) / len(train)
        loss.backward()
        return loss

    started = time.time()
    optimizer.step(closure)
    with torch.no_grad():
        train_scores = train @ weight + bias
        eval_scores = evaluation @ weight + bias
        train_correct = (labels_train * train_scores > 0).numpy()
        eval_correct = (labels_eval * eval_scores > 0).reshape(-1, 2).numpy()
        eval_pair_scores = eval_scores.reshape(-1, 2)
        induced_pairwise = (eval_pair_scores[:, 0] > eval_pair_scores[:, 1]).numpy()

    rng = np.random.default_rng(SEED)
    candidate_bootstrap = []
    induced_bootstrap = []
    for _ in range(10000):
        sample = rng.integers(0, len(eval_positions), len(eval_positions))
        candidate_bootstrap.append(eval_correct[sample].mean())
        induced_bootstrap.append(induced_pairwise[sample].mean())

    result = {
        "variant": variant,
        "feature_dim": int(features.shape[-1]),
        "train_candidate_accuracy": float(train_correct.mean()),
        "eval_candidate_accuracy": float(eval_correct.mean()),
        "eval_chosen_accuracy": float(eval_correct[:, 0].mean()),
        "eval_rejected_accuracy": float(eval_correct[:, 1].mean()),
        "eval_candidate_accuracy_ci95_pair_bootstrap": np.quantile(
            candidate_bootstrap, [0.025, 0.975]
        ).tolist(),
        "eval_induced_pairwise_accuracy": float(induced_pairwise.mean()),
        "eval_induced_pairwise_accuracy_ci95_pair_bootstrap": np.quantile(
            induced_bootstrap, [0.025, 0.975]
        ).tolist(),
        "intercept": float(bias.detach()),
        "weight_norm": float(weight.detach().norm()),
        "bootstrap_draws": 10000,
        "fit_seconds": time.time() - started,
    }
    fitted = {"weight": weight.detach(), "bias": bias.detach(), "result": result}
    del train, evaluation, features
    return result, fitted


def fit(args: argparse.Namespace) -> None:
    out = Path(args.output_dir).resolve()
    root = out / f"hh_{args.backbone}_{args.pairs}_pointwise_states"
    files = sorted(root.glob("states_*.pt"))
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in files]
    states_unsorted = torch.cat([payload["states"] for payload in payloads]).float()
    positions = np.concatenate([np.asarray(payload["positions"]) for payload in payloads])
    order = np.argsort(positions)
    if not np.array_equal(positions[order], np.arange(args.pairs)):
        raise RuntimeError("pointwise state positions are incomplete or duplicated")
    states = states_unsorted[torch.from_numpy(order)]
    if len(states) != args.pairs:
        raise RuntimeError(f"expected {args.pairs} pairs, found {len(states)}")
    permutation = np.random.RandomState(SEED).permutation(args.pairs)
    cut = int(0.8 * args.pairs)
    train_positions = permutation[:cut]
    eval_positions = permutation[cut:]
    variants = {}
    fitted = {}
    for variant in ("final_boundary", "all_boundaries_concat"):
        print(f"fitting {variant}", flush=True)
        variants[variant], fitted[variant] = fit_variant(
            states, train_positions, eval_positions, variant
        )

    model_ref, revision = MODEL_SPECS[args.backbone]
    result = {
        "status": "COMPLETE",
        "backbone": args.backbone,
        "model": model_ref,
        "revision": revision,
        "dataset": "Anthropic/hh-rlhf",
        "dataset_split": "train",
        "n_source_pairs": args.pairs,
        "train_pairs": len(train_positions),
        "eval_pairs": len(eval_positions),
        "train_candidates": 2 * len(train_positions),
        "eval_candidates": 2 * len(eval_positions),
        "pair_selection_seed": SEED,
        "split_seed": SEED,
        "pair_disjoint": True,
        "source_pairs_crossing_splits": 0,
        "candidate_labels": {"chosen": 1, "rejected": 0},
        "feature": "mean-pooled post-final-norm Ouro loop boundary states",
        "max_length": 384,
        "probe": "L-BFGS logistic linear with intercept",
        "normalization": "NoNorm/raw pointwise states",
        "regularization": "0.5 * ||w||^2 / n_train_candidates; intercept unregularized",
        "variants": variants,
    }
    torch.save(
        {
            "variants": fitted,
            "train_positions": train_positions,
            "eval_positions": eval_positions,
            "result": result,
        },
        out / f"hh_{args.backbone}_{args.pairs}_pointwise_pair_disjoint_probe.pt",
    )
    write_json(
        out / f"hh_{args.backbone}_{args.pairs}_pointwise_pair_disjoint_results.json",
        result,
    )
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["extract", "fit"])
    parser.add_argument("--backbone", choices=list(MODEL_SPECS), required=True)
    parser.add_argument("--pairs", type=int, default=40000)
    parser.add_argument("--shard-pairs", type=int, default=500)
    parser.add_argument("--batch-pairs", type=int, default=1)
    parser.add_argument("--sort-window", type=int, default=100)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (extract if args.command == "extract" else fit)(args)


if __name__ == "__main__":
    main()
