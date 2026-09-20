"""Train code-specific tiny pairwise heads and evaluate held-out strict-clean code."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from code_branch_pilot_lib import REPORT_DIR, output_path, repo_path, write_json  # noqa: E402
from math_bg_probe_lib import (  # noqa: E402
    AntisymLinearHead,
    AntisymLinearNoNorm,
    MATH_CONFIGS,
    condorcet_winner_rate,
    config_dim,
    config_vector,
    cycle_rate,
    pairwise_accuracy,
    tournament_top1_accuracy,
)


SPLITS_JSON = REPORT_DIR / "code_specific_training_control_splits_2026-05-17.json"
FEATURES_PT = REPORT_DIR / "code_specific_training_features_2026-05-17.pt"
OUTPUT_JSON = REPORT_DIR / "code_specific_tiny_head_control_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "code_specific_tiny_head_control_2026-05-17.md"

HEAD_CLASSES = {
    "AntisymLinear": AntisymLinearHead,
    "AntisymLinearNoNorm": AntisymLinearNoNorm,
}

HH_STRICT_CLEAN_WEAK_BASELINE = {
    "verdict": "WEAK",
    "config": "47_concat_all_loops",
    "architecture": "AntisymLinearNoNorm",
    "top1": 0.500,
    "pairwise": 0.571,
    "cycle": 0.000,
    "random_top1_baseline": 0.527777781089147,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", default=str(SPLITS_JSON))
    parser.add_argument("--features", default=str(FEATURES_PT))
    parser.add_argument("--output", default=str(OUTPUT_JSON))
    parser.add_argument("--output-md", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not args.output_md:
        args.output_md = str(Path(args.output).with_suffix(".md"))
    return args


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rate(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return "NA"
    if math.isnan(x):
        return "NA"
    return f"{x:.3f}"


def row_compact(row: dict[str, Any] | None) -> dict[str, Any] | str:
    if not row:
        return "NA"
    m = row["metrics"]
    return {
        "config": row["config"],
        "architecture": row["architecture"],
        "top1": m["top1_tournament_acc"],
        "pairwise": m["pairwise_acc"],
        "cycle": m["cycle_rate"],
        "margin_mean": m["margin_mean"],
        "margin_std": m["margin_std"],
    }


def best_by_arch(rows: Sequence[dict[str, Any]], architecture: str) -> dict[str, Any] | None:
    arch_rows = [row for row in rows if row.get("architecture") == architecture]
    if not arch_rows:
        return None
    return max(
        arch_rows,
        key=lambda row: (
            row["metrics"]["top1_tournament_acc"],
            row["metrics"]["pairwise_acc"],
            -row["metrics"]["cycle_rate"],
        ),
    )


def best_overall(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            row["metrics"]["top1_tournament_acc"],
            row["metrics"]["pairwise_acc"],
            -row["metrics"]["cycle_rate"],
        ),
    )


def split_pairs_by_task(
    pairs: list[dict[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    task_ids = sorted({str(pair["task_id"]) for pair in pairs})
    rng = random.Random(seed)
    shuffled = task_ids[:]
    rng.shuffle(shuffled)
    if len(task_ids) >= 8:
        val_count = max(1, min(len(task_ids) - 1, round(0.2 * len(task_ids))))
        val_tasks = set(shuffled[:val_count])
        train = [pair for pair in pairs if pair["task_id"] not in val_tasks]
        val = [pair for pair in pairs if pair["task_id"] in val_tasks]
        return train, val, {
            "validation_split": "task",
            "weak_validation": False,
            "train_task_count": len(set(pair["task_id"] for pair in train)),
            "val_task_count": len(val_tasks),
            "val_task_ids": sorted(val_tasks),
        }
    shuffled_pairs = pairs[:]
    rng.shuffle(shuffled_pairs)
    val_count = max(1, min(len(shuffled_pairs) - 1, round(0.2 * len(shuffled_pairs)))) if len(shuffled_pairs) > 1 else 0
    val = shuffled_pairs[:val_count]
    train = shuffled_pairs[val_count:] or shuffled_pairs
    return train, val or train, {
        "validation_split": "pair",
        "weak_validation": True,
        "train_task_count": len(set(pair["task_id"] for pair in train)),
        "val_task_count": len(set(pair["task_id"] for pair in val)),
        "val_task_ids": sorted({pair["task_id"] for pair in val}),
    }


def feature_map(feature_payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for row in feature_payload.get("candidate_features", []) or []:
        out[str(row["candidate_uid"])] = row["pooled"].detach().cpu().to(torch.float32)
    return out


def pair_tensors(
    pairs: Sequence[dict[str, Any]],
    pooled_by_uid: dict[str, torch.Tensor],
    config: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    left = [config_vector(pooled_by_uid[pair["preferred_uid"]], config).to(torch.float32) for pair in pairs]
    right = [config_vector(pooled_by_uid[pair["rejected_uid"]], config).to(torch.float32) for pair in pairs]
    return torch.stack(left, dim=0), torch.stack(right, dim=0)


@torch.no_grad()
def pair_loss(head: torch.nn.Module, left: torch.Tensor, right: torch.Tensor, device: torch.device) -> float:
    logits = head(left.to(device), right.to(device))
    target = torch.ones_like(logits)
    loss = F.binary_cross_entropy_with_logits(logits, target)
    return float(loss.detach().cpu())


@torch.no_grad()
def pair_acc(head: torch.nn.Module, left: torch.Tensor, right: torch.Tensor, device: torch.device) -> float:
    logits = head(left.to(device), right.to(device))
    return float((logits > 0).to(torch.float32).mean().detach().cpu())


def train_head(
    *,
    architecture: str,
    config: str,
    train_pairs: list[dict[str, Any]],
    val_pairs: list[dict[str, Any]],
    pooled_by_uid: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    head = HEAD_CLASSES[architecture](config_dim(config)).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    left_train, right_train = pair_tensors(train_pairs, pooled_by_uid, config)
    left_val, right_val = pair_tensors(val_pairs, pooled_by_uid, config)
    left_train = left_train.to(device)
    right_train = right_train.to(device)
    target = torch.ones(left_train.shape[0], device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    batch_size = max(1, min(int(args.batch_size), int(left_train.shape[0])))
    best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    best_loss = float("inf")
    best_epoch = -1
    stale = 0
    losses: list[float] = []
    val_losses: list[float] = []
    for epoch in range(int(args.epochs)):
        head.train()
        perm = torch.randperm(left_train.shape[0], generator=generator, device=device)
        total = 0.0
        for start in range(0, left_train.shape[0], batch_size):
            batch = perm[start : start + batch_size]
            logits = head(left_train[batch], right_train[batch])
            loss = F.binary_cross_entropy_with_logits(logits, target[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * int(batch.numel())
        train_loss = total / max(int(left_train.shape[0]), 1)
        losses.append(train_loss)
        head.eval()
        val_loss = pair_loss(head, left_val, right_val, device)
        val_losses.append(val_loss)
        if val_loss + 1e-7 < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
        if stale >= int(args.patience):
            break
    head.load_state_dict(best_state)
    head.eval()
    metrics = {
        "architecture": architecture,
        "config": config,
        "dim": config_dim(config),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "batch_size": batch_size,
        "epochs_run": len(losses),
        "best_epoch": best_epoch + 1,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "val_loss_best": best_loss,
        "train_pair_acc": pair_acc(head, left_train.cpu(), right_train.cpu(), device),
        "val_pair_acc": pair_acc(head, left_val, right_val, device),
    }
    return head.to("cpu"), metrics


def records_for_eval_set(
    eval_rows: Sequence[dict[str, Any]],
    pooled_by_uid: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(eval_rows):
        uids = [str(uid) for uid in row.get("candidate_uids", [])]
        records.append({
            "tournament_id": idx,
            "task_id": row["task_id"],
            "source": row.get("source", "unknown"),
            "difficulty": row.get("difficulty", "unknown"),
            "function_name": row.get("function_name", ""),
            "prompt": row.get("prompt", ""),
            "candidate_uids": uids,
            "label_names": list(row.get("labels", [])),
            "labels": torch.tensor([label == "correct" for label in row.get("labels", [])], dtype=torch.bool),
            "pooled": torch.stack([pooled_by_uid[uid] for uid in uids], dim=0),
        })
    return records


def config_features(records: Sequence[dict[str, Any]], config: str) -> list[torch.Tensor]:
    return [
        torch.stack([config_vector(candidate_pooled, config) for candidate_pooled in row["pooled"]], dim=0).to(torch.float32)
        for row in records
    ]


@torch.no_grad()
def score_matrix(head: torch.nn.Module, feats: torch.Tensor, device: torch.device) -> torch.Tensor:
    feats = feats.to(device=device, dtype=torch.float32)
    k = feats.shape[0]
    left = feats[:, None, :].expand(k, k, feats.shape[-1]).reshape(k * k, feats.shape[-1])
    right = feats[None, :, :].expand(k, k, feats.shape[-1]).reshape(k * k, feats.shape[-1])
    mat = head(left, right).reshape(k, k).detach().cpu()
    mat.fill_diagonal_(0.0)
    return mat


def margin_stats(matrices: Sequence[torch.Tensor]) -> tuple[float, float]:
    margins: list[float] = []
    for mat in matrices:
        totals = mat.sum(dim=1)
        if totals.numel() < 2:
            margins.append(0.0)
            continue
        top2 = torch.topk(totals, k=2).values
        margins.append(float(top2[0] - top2[1]))
    return (float(mean(margins)), float(pstdev(margins))) if margins else (float("nan"), float("nan"))


def evaluate_matrices(
    matrices: Sequence[torch.Tensor],
    records: Sequence[dict[str, Any]],
    baseline: float,
) -> dict[str, Any]:
    labels = [row["labels"] for row in records]
    margin_mean, margin_std = margin_stats(matrices)
    top1 = tournament_top1_accuracy(matrices, labels)
    return {
        "n_tournaments": len(records),
        "top1_tournament_acc": top1,
        "top1_over_random_baseline": top1 - baseline,
        "pairwise_acc": pairwise_accuracy(matrices, labels),
        "condorcet_winner_rate": condorcet_winner_rate(matrices, labels),
        "cycle_rate": cycle_rate(matrices, triplets_per_matrix=1000, seed=42),
        "margin_mean": margin_mean,
        "margin_std": margin_std,
    }


def random_top1_baseline(records: Sequence[dict[str, Any]]) -> float:
    return float(mean(float(row["labels"].to(torch.float32).mean()) for row in records)) if records else float("nan")


def per_task_breakdown(
    matrices: Sequence[torch.Tensor],
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mat, record in zip(matrices, records):
        totals = mat.sum(dim=1)
        pred = int(torch.argmax(totals).item())
        top2_margin = 0.0
        if totals.numel() >= 2:
            vals = torch.topk(totals, k=2).values
            top2_margin = float(vals[0] - vals[1])
        label_names = record["label_names"]
        rows.append({
            "task_id": record["task_id"],
            "n_candidates": len(record["candidate_uids"]),
            "label_counts": dict(Counter(label_names)),
            "predicted_index": pred,
            "predicted_candidate_uid": record["candidate_uids"][pred],
            "predicted_label": label_names[pred],
            "predicted_is_correct": bool(record["labels"][pred]),
            "top2_margin": top2_margin,
        })
    return rows


def evaluate_set(
    *,
    set_name: str,
    records: list[dict[str, Any]],
    heads: Sequence[tuple[str, str, torch.nn.Module, dict[str, Any]]],
    baseline: float,
    device: torch.device,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_matrices: list[torch.Tensor] = []
    for config, architecture, head, train_metrics in heads:
        feats_by_record = config_features(records, config)
        head = head.to(device)
        matrices = [score_matrix(head, feats, device) for feats in feats_by_record]
        head = head.to("cpu")
        metrics = evaluate_matrices(matrices, records, baseline)
        row = {
            "eval_set": set_name,
            "config": config,
            "architecture": architecture,
            "train_metrics": train_metrics,
            "metrics": metrics,
        }
        rows.append(row)
        if best is None or (
            metrics["top1_tournament_acc"],
            metrics["pairwise_acc"],
            -metrics["cycle_rate"],
        ) > (
            best["metrics"]["top1_tournament_acc"],
            best["metrics"]["pairwise_acc"],
            -best["metrics"]["cycle_rate"],
        ):
            best = row
            best_matrices = matrices
    return {
        "eval_set": set_name,
        "n_tournaments": len(records),
        "n_candidates": sum(len(row["candidate_uids"]) for row in records),
        "random_top1_baseline": baseline,
        "transfer_table": rows,
        "best_code_trained": best,
        "best_antisymlinear": best_by_arch(rows, "AntisymLinear"),
        "best_nonorm": best_by_arch(rows, "AntisymLinearNoNorm"),
        "per_task_breakdown_best": per_task_breakdown(best_matrices, records) if best_matrices else [],
    }


def verdict_for(best: dict[str, Any] | None, baseline: float) -> str:
    if best is None:
        return "NOT_RUN"
    m = best["metrics"]
    top1 = float(m["top1_tournament_acc"])
    pairwise = float(m["pairwise_acc"])
    cycle = float(m["cycle_rate"])
    if top1 >= baseline + 0.15 and pairwise >= 0.60 and cycle <= 0.05:
        return "GOOD"
    if top1 >= baseline + 0.05 or pairwise >= 0.55:
        return "WEAK"
    return "POOR"


def interpretation_for(verdict: str) -> str:
    if verdict == "GOOD":
        return (
            "States contain a strict-clean code branch signal; HH-trained projection did not transfer strongly enough "
            "to near-miss code, while code-specific tiny taps can read it."
        )
    if verdict == "WEAK":
        return "A signal may exist, but the dataset is too small/noisy or the pairwise projection is underpowered."
    if verdict == "POOR":
        return "Strict-clean near-miss signal is not currently linearly readable from these pooled states, or the eval set is too small/noisy."
    return "The code-specific training control did not run."


def write_md(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# Code-Specific Tiny-Head Control",
        "",
        f"CODE_SPECIFIC_TINY_HEAD_VERDICT = {payload['code_specific_tiny_head_verdict']}",
        "",
        "## Split And Training",
        "",
        f"- split_verdict: `{payload['code_specific_split_verdict']}`",
        f"- feature_verdict: `{payload['code_specific_feature_verdict']}`",
        f"- training_tasks: `{summary['training_task_count']}`",
        f"- primary_training_pairs: `{summary['primary_training_pair_count']}`",
        f"- validation_split: `{summary['validation']['validation_split']}`",
        f"- weak_validation: `{summary['validation']['weak_validation']}`",
        "",
        "## Primary Strict-Clean Eval",
        "",
        f"- heldout_tasks: `{summary['primary_strict_clean_task_count']}`",
        f"- random_top1_baseline: `{summary['random_top1_baseline_primary']}`",
        f"- best AntisymLinear: `{summary['best_code_trained_antisymlinear']}`",
        f"- best NoNorm: `{summary['best_code_trained_nonorm']}`",
        f"- best overall: `{summary['best_code_trained_overall']}`",
        "",
        "HH-trained strict-clean transfer was `WEAK`: "
        "`47_concat_all_loops / AntisymLinearNoNorm`, top1=0.500, pairwise=0.571, cycle=0.000.",
        "",
    ]
    for set_name, result in payload["eval_results"].items():
        lines.extend([
            f"## {set_name}",
            "",
            f"- n_tournaments: `{result['n_tournaments']}`",
            f"- n_candidates: `{result['n_candidates']}`",
            f"- random_top1_baseline: `{result['random_top1_baseline']}`",
            f"- best AntisymLinear: `{row_compact(result.get('best_antisymlinear'))}`",
            f"- best NoNorm: `{row_compact(result.get('best_nonorm'))}`",
            "",
            "| config | architecture | top1 | over_random | pairwise | condorcet | cycle | margin_mean | margin_std |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in result["transfer_table"]:
            m = row["metrics"]
            lines.append(
                f"| `{row['config']}` | `{row['architecture']}` | {rate(m['top1_tournament_acc'])} | "
                f"{rate(m['top1_over_random_baseline'])} | {rate(m['pairwise_acc'])} | "
                f"{rate(m['condorcet_winner_rate'])} | {rate(m['cycle_rate'])} | "
                f"{rate(m['margin_mean'])} | {rate(m['margin_std'])} |"
            )
        lines.extend(["", "### Per-Task Breakdown For Best Row", ""])
        for row in result["per_task_breakdown_best"]:
            lines.append(
                f"- `{row['task_id']}` pred_label=`{row['predicted_label']}` "
                f"correct=`{row['predicted_is_correct']}` margin={row['top2_margin']:.3f}"
            )
        lines.append("")
    lines.extend([
        "## Interpretation",
        "",
        summary["one_sentence_interpretation"],
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    device = torch.device(args.device)

    splits_path = output_path(args.splits)
    features_path = output_path(args.features)
    out_json = output_path(args.output)
    out_md = output_path(args.output_md)
    splits = load_json(splits_path)
    split_verdict = str(splits.get("code_specific_split_verdict", "BLOCKED"))
    if split_verdict not in {"READY", "MISSING_FEATURES"}:
        raise SystemExit(f"CODE_SPECIFIC_TINY_HEAD_VERDICT=NOT_RUN: split verdict {split_verdict}")
    feature_payload = torch.load(features_path, map_location="cpu", weights_only=False)
    feature_meta = feature_payload.get("meta", {})
    feature_verdict = str(feature_meta.get("code_specific_feature_verdict", "BLOCKED"))
    if feature_verdict not in {"READY", "RECAPTURED"}:
        raise SystemExit(f"CODE_SPECIFIC_TINY_HEAD_VERDICT=NOT_RUN: feature verdict {feature_verdict}")

    pooled_by_uid = feature_map(feature_payload)
    pairs = [pair for pair in splits.get("training_pairs_primary", []) if pair["preferred_uid"] in pooled_by_uid and pair["rejected_uid"] in pooled_by_uid]
    if len(pairs) < 1:
        raise SystemExit("CODE_SPECIFIC_TINY_HEAD_VERDICT=NOT_RUN: no train pairs with features")
    heldout = set(splits.get("heldout_strict_clean_task_ids", []))
    leakage = sorted({pair["task_id"] for pair in pairs if pair["task_id"] in heldout})
    if leakage:
        raise SystemExit(f"held-out task leakage in training pairs: {leakage}")
    train_pairs, val_pairs, validation = split_pairs_by_task(pairs, int(args.seed))

    heads: list[tuple[str, str, torch.nn.Module, dict[str, Any]]] = []
    for config in MATH_CONFIGS:
        print(f"training code-specific heads {config}", flush=True)
        for architecture in ("AntisymLinear", "AntisymLinearNoNorm"):
            head, train_metrics = train_head(
                architecture=architecture,
                config=config,
                train_pairs=train_pairs,
                val_pairs=val_pairs,
                pooled_by_uid=pooled_by_uid,
                args=args,
                device=device,
            )
            heads.append((config, architecture, head, train_metrics))

    records_by_set: dict[str, list[dict[str, Any]]] = {}
    for set_name, rows in splits.get("eval_sets", {}).items():
        if not rows:
            continue
        records_by_set[set_name] = records_for_eval_set(rows, pooled_by_uid)
    eval_results: dict[str, Any] = {}
    for set_name, records in records_by_set.items():
        baseline = random_top1_baseline(records)
        eval_results[set_name] = evaluate_set(
            set_name=set_name,
            records=records,
            heads=heads,
            baseline=baseline,
            device=device,
        )

    primary = eval_results.get("primary_strict_clean", {})
    primary_best = primary.get("best_code_trained")
    baseline = float(primary.get("random_top1_baseline", float("nan")))
    verdict = verdict_for(primary_best, baseline)
    summary = {
        "CODE_SPECIFIC_SPLIT_VERDICT": split_verdict,
        "CODE_SPECIFIC_FEATURE_VERDICT": feature_verdict,
        "CODE_SPECIFIC_TINY_HEAD_VERDICT": verdict,
        "training_task_count": splits.get("summary", {}).get("training_task_count"),
        "primary_training_pair_count": len(pairs),
        "heldout_strict_clean_task_count": len(records_by_set.get("primary_strict_clean", [])),
        "primary_strict_clean_task_count": len(records_by_set.get("primary_strict_clean", [])),
        "random_top1_baseline_primary": baseline,
        "best_code_trained_antisymlinear": row_compact(primary.get("best_antisymlinear")),
        "best_code_trained_nonorm": row_compact(primary.get("best_nonorm")),
        "best_code_trained_overall": row_compact(primary_best),
        "validation": validation,
        "hh_trained_strict_clean_baseline": HH_STRICT_CLEAN_WEAK_BASELINE,
        "one_sentence_interpretation": interpretation_for(verdict),
    }
    payload = {
        "code_specific_tiny_head_verdict": verdict,
        "code_specific_split_verdict": split_verdict,
        "code_specific_feature_verdict": feature_verdict,
        "splits": repo_path(splits_path),
        "features": repo_path(features_path),
        "training": {
            "primary_pairs": len(pairs),
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "validation": validation,
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "optimizer": "AdamW",
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "batch_size": min(int(args.batch_size), len(train_pairs)),
            "swap_augmentation": False,
            "lambda_sym": 0.0,
        },
        "eval_results": eval_results,
        "hh_trained_strict_clean_baseline": HH_STRICT_CLEAN_WEAK_BASELINE,
        "summary": summary,
    }
    write_json(out_json, payload)
    write_md(out_md, payload)
    print(f"CODE_SPECIFIC_TINY_HEAD_VERDICT = {verdict}", flush=True)
    print(f"best_code_trained_antisymlinear = {summary['best_code_trained_antisymlinear']}", flush=True)
    print(f"best_code_trained_nonorm = {summary['best_code_trained_nonorm']}", flush=True)
    print(f"Wrote {out_json}", flush=True)
    print(f"Wrote {out_md}", flush=True)


if __name__ == "__main__":
    main()
