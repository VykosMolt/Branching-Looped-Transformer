"""Train tiny pairwise heads on generated math branch tournaments.

This is a focused probe, not the final BG tap implementation. It uses
mean-pooled tap features and an antisymmetric linear pairwise comparator to
test whether layers 24/36 contain trainable branch-selection signal on
Ouro-RLTT-generated math branches.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from math_bg_probe_lib import (
    AntisymLinearHead,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RLTT_PATH,
    DEFAULT_TOKENIZER_PATH,
    MATH_CONFIGS,
    TAP_LAYERS,
    capture_pooled_taps,
    config_dim,
    config_vector,
    output_path,
    resolve_local,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-json", default=f"{DEFAULT_OUTPUT_DIR}/math_branch_tournaments_rltt.json")
    p.add_argument("--features-pt", default=f"{DEFAULT_OUTPUT_DIR}/math_branch_tap_features.pt")
    p.add_argument("--model-path", default=DEFAULT_RLTT_PATH)
    p.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    p.add_argument("--max-length", type=int, default=768)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-frac", type=float, default=0.25)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--report-every", type=int, default=25)
    p.add_argument("--recapture", action="store_true")
    p.add_argument("--output-heads", default=f"{DEFAULT_OUTPUT_DIR}/math_single_state_tap_heads.pt")
    p.add_argument("--output-json", default=f"{DEFAULT_OUTPUT_DIR}/train_single_state_math_taps_24_36.json")
    p.add_argument("--output-md", default=f"{DEFAULT_OUTPUT_DIR}/train_single_state_math_taps_24_36.md")
    return p.parse_args()


def load_tournaments(path: Path) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tournaments = payload.get("tournaments", [])
    if not tournaments:
        raise SystemExit(f"No tournaments found in {path}")
    return tournaments, payload.get("meta", {})


def capture_or_load_features(args: argparse.Namespace, tournaments: List[Dict[str, object]]) -> Dict[str, object]:
    features_path = output_path(args.features_pt)
    if features_path.exists() and not args.recapture:
        print(f"Reusing features: {features_path}")
        return torch.load(features_path, map_location="cpu", weights_only=False)

    texts: List[str] = []
    spans = []
    for t in tournaments:
        start = len(texts)
        for attempt in t["attempts"]:
            texts.append(str(attempt["candidate_text"]))
        spans.append((start, len(texts)))

    device = torch.device(args.device)
    model_path = resolve_local(args.model_path)
    tokenizer_path = resolve_local(args.tokenizer_path)

    print(f"Loading tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading RLTT model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "config"):
        model.config.early_exit_threshold = 1.0

    pooled_flat = capture_pooled_taps(
        model, tokenizer, texts, args.max_length, device, args.report_every)

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    records = []
    for t, (start, end) in zip(tournaments, spans):
        labels = torch.tensor([bool(a["is_correct"]) for a in t["attempts"]], dtype=torch.bool)
        records.append({
            "tournament_id": int(t["tournament_id"]),
            "source": t.get("source"),
            "dataset_index": t.get("dataset_index"),
            "question": t.get("question"),
            "gold_answer": t.get("gold_answer"),
            "labels": labels,
            "candidate_texts": [str(a["candidate_text"]) for a in t["attempts"]],
            "attempt_sources": [str(a.get("source", "")) for a in t["attempts"]],
            "pooled": pooled_flat[start:end].contiguous(),
        })

    payload = {
        "meta": {
            "model_path_resolved": model_path,
            "tokenizer_path_resolved": tokenizer_path,
            "max_length": args.max_length,
            "tap_layers": list(TAP_LAYERS),
            "n_tournaments": len(records),
            "n_candidates": len(texts),
        },
        "records": records,
    }
    features_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, features_path)
    print(f"Wrote {features_path}")
    return payload


def split_indices(n: int, seed: int, eval_frac: float) -> Tuple[List[int], List[int]]:
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    n_eval = max(1, int(round(n * eval_frac))) if n > 1 else 0
    eval_idx = sorted(indices[:n_eval])
    train_idx = sorted(indices[n_eval:]) or sorted(indices)
    return train_idx, eval_idx or train_idx


def config_matrix(records: Sequence[Dict[str, object]], config: str) -> List[torch.Tensor]:
    return [torch.stack([config_vector(row, config) for row in rec["pooled"]], dim=0) for rec in records]


def build_pair_tensors(
    features_by_record: Sequence[torch.Tensor],
    records: Sequence[Dict[str, object]],
    indices: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    left: List[torch.Tensor] = []
    right: List[torch.Tensor] = []
    for idx in indices:
        labels = records[idx]["labels"]
        feats = features_by_record[idx]
        correct = torch.nonzero(labels, as_tuple=False).flatten().tolist()
        incorrect = torch.nonzero(~labels, as_tuple=False).flatten().tolist()
        for c in correct:
            for r in incorrect:
                left.append(feats[c])
                right.append(feats[r])
    if not left:
        raise RuntimeError("No correct-vs-incorrect pairs available.")
    return torch.stack(left, dim=0), torch.stack(right, dim=0)


@torch.no_grad()
def evaluate_head(
    head: AntisymLinearHead,
    features_by_record: Sequence[torch.Tensor],
    records: Sequence[Dict[str, object]],
    indices: Sequence[int],
    device: torch.device,
) -> Dict[str, float | int]:
    pair_total = 0
    pair_correct = 0
    top1_total = 0
    top1_correct = 0
    condorcet_total = 0
    condorcet_correct = 0
    margins = []
    for idx in indices:
        labels = records[idx]["labels"]
        feats = features_by_record[idx].to(device)
        k = feats.shape[0]
        scores = torch.zeros((k, k), device=device)
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                scores[i, j] = head(feats[i:i+1], feats[j:j+1])[0]
        for c in torch.nonzero(labels, as_tuple=False).flatten().tolist():
            for r in torch.nonzero(~labels, as_tuple=False).flatten().tolist():
                pair_total += 1
                pair_correct += int(float(scores[c, r]) > 0.0)
        totals = scores.sum(dim=1)
        order = torch.argsort(totals, descending=True)
        pred = int(order[0])
        top1_total += 1
        top1_correct += int(bool(labels[pred]))
        margin = float(totals[order[0]] - totals[order[1]]) if k > 1 else 0.0
        margins.append(margin)
        beats_all = (scores[pred, torch.arange(k, device=device) != pred] > 0).all().item()
        condorcet_total += 1
        condorcet_correct += int(bool(labels[pred]) and beats_all)
    return {
        "n_tournaments": int(top1_total),
        "n_pairs": int(pair_total),
        "pairwise_acc": pair_correct / max(pair_total, 1),
        "top1_tournament_acc": top1_correct / max(top1_total, 1),
        "condorcet_correct_rate": condorcet_correct / max(condorcet_total, 1),
        "margin_mean": float(np.mean(margins)) if margins else float("nan"),
    }


def train_one_config(
    config: str,
    records: Sequence[Dict[str, object]],
    train_idx: Sequence[int],
    eval_idx: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[AntisymLinearHead, Dict[str, object]]:
    features_by_record = config_matrix(records, config)
    train_left, train_right = build_pair_tensors(features_by_record, records, train_idx)
    train_left = train_left.to(device)
    train_right = train_right.to(device)

    head = AntisymLinearHead(config_dim(config)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n = train_left.shape[0]
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)
    losses = []
    for epoch in range(args.epochs):
        perm = torch.randperm(n, generator=rng, device=device)
        epoch_loss = 0.0
        for start in range(0, n, args.batch_size):
            batch = perm[start:start + args.batch_size]
            logits = head(train_left[batch], train_right[batch])
            loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach()) * len(batch)
        losses.append(epoch_loss / n)

    head.eval()
    train_metrics = evaluate_head(head, features_by_record, records, train_idx, device)
    eval_metrics = evaluate_head(head, features_by_record, records, eval_idx, device)
    metrics = {
        "config": config,
        "dim": config_dim(config),
        "train_pairs": int(n),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "train": train_metrics,
        "eval": eval_metrics,
    }
    return head.to("cpu"), metrics


def write_md(path: Path, result: Dict[str, object]) -> None:
    rows = sorted(result["metrics"], key=lambda r: (-r["eval"]["top1_tournament_acc"], -r["eval"]["pairwise_acc"]))
    lines = [
        "# Trained Single-State Math Tap Probe",
        "",
        "Tiny antisymmetric pairwise heads trained on generated math branch tournaments.",
        "",
        "| Config | eval top1 | eval pairwise | eval condorcet | train top1 | loss last |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['config']}` | {row['eval']['top1_tournament_acc']:.3f} | "
            f"{row['eval']['pairwise_acc']:.3f} | {row['eval']['condorcet_correct_rate']:.3f} | "
            f"{row['train']['top1_tournament_acc']:.3f} | {row['loss_last']:.4f} |"
        )
    lines.extend([
        "",
        "Interpretation: these are relational branch-selection heads over pooled "
        "tap features, not pointwise judges.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    tournaments, tournament_meta = load_tournaments(output_path(args.input_json))
    feature_payload = capture_or_load_features(args, tournaments)
    records = feature_payload["records"]
    train_idx, eval_idx = split_indices(len(records), args.seed, args.eval_frac)
    print(f"records={len(records)} train={len(train_idx)} eval={len(eval_idx)}")

    device = torch.device(args.device)
    heads = {}
    metrics = []
    for config in MATH_CONFIGS:
        print(f"\nTraining {config}")
        head, row = train_one_config(config, records, train_idx, eval_idx, args, device)
        heads[config] = {
            "state_dict": head.state_dict(),
            "dim": config_dim(config),
        }
        metrics.append(row)
        print(
            f"{config}: eval_top1={row['eval']['top1_tournament_acc']:.3f} "
            f"eval_pair={row['eval']['pairwise_acc']:.3f} loss={row['loss_last']:.4f}"
        )

    result = {
        "args": vars(args),
        "input_meta": tournament_meta,
        "features_meta": feature_payload.get("meta", {}),
        "train_indices": train_idx,
        "eval_indices": eval_idx,
        "metrics": metrics,
    }
    out_heads = output_path(args.output_heads)
    out_json = output_path(args.output_json)
    out_md = output_path(args.output_md)
    out_heads.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "meta": {
            "args": vars(args),
            "configs": list(MATH_CONFIGS),
            "head_class": "AntisymLinearHead",
            "train_indices": train_idx,
            "eval_indices": eval_idx,
        },
        "heads": heads,
    }, out_heads)
    out_json.write_text(json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8")
    write_md(out_md, result)
    print(f"\nWrote {out_heads}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
