"""Evaluate HH-trained exact-antisymmetric linear taps on clean GSM8K tournaments."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from math_bg_probe_lib import (
    AntisymLinearHead,
    AntisymLinearNoNorm,
    MATH_CONFIGS,
    PROJECT_ROOT,
    condorcet_winner_rate,
    config_dim,
    config_vector,
    cycle_rate,
    output_path,
    pairwise_accuracy,
    tournament_top1_accuracy,
)


DEFAULT_FEATURES = PROJECT_ROOT / "opi/taps/probes/clean_gsm8k_extreme_tap_features_2026-05-16.pt"
DEFAULT_HH = PROJECT_ROOT / "rpe/evaluator/hh_layer_states_200_rltt.pt"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "opi/taps/probes/clean_gsm8k_extreme_transfer_2026-05-16.json"
OLD_CONFOUNDED_JSON = PROJECT_ROOT / "opi/taps/probes/hh_to_math_transfer_probe_2026-05-16.json"
MICRO_TRANSFER_JSON = PROJECT_ROOT / "opi/taps/probes/clean_gsm8k_extreme_transfer_2026-05-16.json"


HEAD_CLASSES = {
    "AntisymLinear": AntisymLinearHead,
    "AntisymLinearNoNorm": AntisymLinearNoNorm,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default=str(DEFAULT_FEATURES))
    parser.add_argument("--hh-capture", default=str(DEFAULT_HH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--heldout", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def repo_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def rate(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{value:.3f}"


def snippet(text: object, limit: int = 260) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def split_indices(n: int, heldout: int, seed: int) -> Tuple[List[int], List[int]]:
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    eval_idx = sorted(indices[: min(heldout, n)])
    train_idx = sorted(indices[min(heldout, n) :])
    if not train_idx:
        train_idx = eval_idx
    return train_idx, eval_idx


def hh_config_vector(pooled: Dict[int, torch.Tensor], config: str) -> torch.Tensor:
    if config == "24_L1":
        return pooled[24][0]
    if config == "24_L4":
        return pooled[24][3]
    if config == "24_mean":
        return pooled[24].mean(dim=0)
    if config == "36_L1":
        return pooled[36][0]
    if config == "36_L4":
        return pooled[36][3]
    if config == "36_mean":
        return pooled[36].mean(dim=0)
    if config == "47_L4":
        return pooled[47][3]
    if config == "47_mean":
        return pooled[47].mean(dim=0)
    if config == "47_concat_L1_L4":
        return torch.cat([pooled[47][0], pooled[47][3]], dim=-1)
    if config == "47_concat_all_loops":
        return torch.cat([pooled[47][i] for i in range(4)], dim=-1)
    raise ValueError(f"unknown config: {config}")


def build_hh_features(hh_payload: Dict[str, object], config: str) -> Tuple[torch.Tensor, torch.Tensor]:
    chosen = []
    rejected = []
    for pack in hh_payload["packs"]:
        chosen.append(hh_config_vector(pack["chosen"]["pooled"], config).to(torch.float32))
        rejected.append(hh_config_vector(pack["rejected"]["pooled"], config).to(torch.float32))
    return torch.stack(chosen, dim=0), torch.stack(rejected, dim=0)


@torch.no_grad()
def hh_loss(head: torch.nn.Module, chosen: torch.Tensor, rejected: torch.Tensor, idx: Sequence[int], device: torch.device) -> float:
    logits = head(chosen[list(idx)].to(device), rejected[list(idx)].to(device))
    loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    return float(loss.detach().cpu())


@torch.no_grad()
def hh_acc(head: torch.nn.Module, chosen: torch.Tensor, rejected: torch.Tensor, idx: Sequence[int], device: torch.device) -> float:
    logits = head(chosen[list(idx)].to(device), rejected[list(idx)].to(device))
    return float((logits > 0).to(torch.float32).mean().detach().cpu())


def train_hh_head(
    architecture: str,
    config: str,
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    train_idx: Sequence[int],
    eval_idx: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, object]]:
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    head = HEAD_CLASSES[architecture](config_dim(config)).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    left = chosen[list(train_idx)].to(device)
    right = rejected[list(train_idx)].to(device)
    target = torch.ones(left.shape[0], device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    best_loss = float("inf")
    best_epoch = -1
    stale = 0
    losses: List[float] = []
    eval_losses: List[float] = []
    for epoch in range(args.epochs):
        head.train()
        perm = torch.randperm(left.shape[0], generator=generator, device=device)
        total = 0.0
        for start in range(0, left.shape[0], args.batch_size):
            batch = perm[start : start + args.batch_size]
            logits = head(left[batch], right[batch])
            loss = F.binary_cross_entropy_with_logits(logits, target[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * int(batch.numel())
        train_loss = total / max(left.shape[0], 1)
        losses.append(train_loss)
        head.eval()
        val = hh_loss(head, chosen, rejected, eval_idx, device)
        eval_losses.append(val)
        if val + 1e-7 < best_loss:
            best_loss = val
            best_epoch = epoch
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            break
    head.load_state_dict(best_state)
    head.eval()
    metrics = {
        "architecture": architecture,
        "config": config,
        "dim": config_dim(config),
        "epochs_run": len(losses),
        "best_epoch": best_epoch + 1,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "eval_loss_best": best_loss,
        "hh_train_acc": hh_acc(head, chosen, rejected, train_idx, device),
        "hh_heldout_acc": hh_acc(head, chosen, rejected, eval_idx, device),
    }
    return head.to("cpu"), metrics


def config_features(records: Sequence[Dict[str, object]], config: str) -> List[torch.Tensor]:
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


def margin_stats(matrices: Sequence[torch.Tensor]) -> Tuple[float, float]:
    margins: List[float] = []
    for mat in matrices:
        totals = mat.sum(dim=1)
        if totals.numel() < 2:
            margins.append(0.0)
            continue
        top2 = torch.topk(totals, k=2).values
        margins.append(float(top2[0] - top2[1]))
    return (float(mean(margins)), float(pstdev(margins))) if margins else (float("nan"), float("nan"))


def evaluate_matrices(matrices: Sequence[torch.Tensor], records: Sequence[Dict[str, object]], baseline: float) -> Dict[str, object]:
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


def random_top1_baseline(records: Sequence[Dict[str, object]]) -> float:
    return float(mean(float(row["labels"].to(torch.float32).mean()) for row in records)) if records else float("nan")


def load_generation_payload(feature_payload: Dict[str, object]) -> Tuple[Path, Dict[str, object]]:
    rel = feature_payload.get("meta", {}).get("input_json")
    if not rel:
        raise SystemExit("feature payload lacks meta.input_json")
    path = output_path(rel)
    if not path.exists():
        raise SystemExit(f"missing generation JSON: {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def clean_context(generation: Dict[str, object]) -> Dict[str, object]:
    expanded = generation.get("expanded_clean_gsm8k_verdict")
    micro = generation.get("clean_gsm8k_verdict")
    if expanded is not None:
        return {
            "kind": "expanded",
            "clean_verdict": expanded,
            "allowed": expanded in {"CLEAN_30", "CLEAN_MINIMUM"},
            "min_n": 20,
        }
    return {
        "kind": "micro",
        "clean_verdict": micro,
        "allowed": micro == "CLEAN",
        "min_n": 5,
    }


def linear_verdict(best: Optional[Dict[str, object]], baseline: float, n_tournaments: int, kind: str) -> str:
    if kind == "expanded":
        if best is None or n_tournaments < 20:
            return "NOT_RUN"
        metrics = best["metrics"]
        top1 = float(metrics["top1_tournament_acc"])
        pairwise = float(metrics["pairwise_acc"])
        cycles = float(metrics["cycle_rate"])
        if top1 >= baseline + 0.15 and pairwise >= 0.60 and cycles <= 0.05:
            return "GOOD"
        if top1 >= baseline + 0.05 or pairwise >= 0.55:
            return "WEAK"
        return "POOR"
    if best is None or n_tournaments < 5:
        return "NOT_RUN"
    metrics = best["metrics"]
    top1 = float(metrics["top1_tournament_acc"])
    pairwise = float(metrics["pairwise_acc"])
    cycles = float(metrics["cycle_rate"])
    if top1 >= baseline + 0.20 and pairwise >= 0.60 and cycles <= 0.05:
        return "PRELIM_GOOD"
    if top1 >= baseline + 0.05 or pairwise >= 0.55:
        return "PRELIM_WEAK"
    return "PRELIM_POOR"


def old_confounded_summary() -> Dict[str, object]:
    if not OLD_CONFOUNDED_JSON.exists():
        return {"path": repo_path(OLD_CONFOUNDED_JSON), "exists": False}
    payload = json.loads(OLD_CONFOUNDED_JSON.read_text(encoding="utf-8"))
    best = None
    for row in payload.get("transfer_rows", []):
        if row.get("architecture") != "AntisymLinear":
            continue
        value = row.get("hh_all_top1")
        if value is None:
            continue
        if best is None or value > best.get("hh_all_top1", -1):
            best = row
    return {
        "path": repo_path(OLD_CONFOUNDED_JSON),
        "exists": True,
        "label": "historical_confounded_baseline",
        "transfer_verdict": payload.get("transfer_verdict"),
        "best_antisymlinear": best,
    }


def clean_micro_summary(out_json: Path) -> Dict[str, object]:
    if not MICRO_TRANSFER_JSON.exists() or out_json.resolve() == MICRO_TRANSFER_JSON.resolve():
        return {"path": repo_path(MICRO_TRANSFER_JSON), "exists": MICRO_TRANSFER_JSON.exists()}
    payload = json.loads(MICRO_TRANSFER_JSON.read_text(encoding="utf-8"))
    best = payload.get("best_hh_trained")
    return {
        "path": repo_path(MICRO_TRANSFER_JSON),
        "exists": True,
        "label": "tiny_clean_n5_baseline",
        "clean_gsm8k_verdict": payload.get("clean_gsm8k_verdict"),
        "clean_transfer_verdict": payload.get("clean_transfer_verdict"),
        "random_top1_baseline": payload.get("random_top1_baseline"),
        "best_hh_trained": {
            "config": best.get("config") if best else None,
            "architecture": best.get("architecture") if best else None,
            "metrics": best.get("metrics") if best else None,
        },
    }


def examples_from_generation(generation: Dict[str, object], limit: int = 10) -> List[Dict[str, object]]:
    examples = []
    for tournament in generation.get("tournaments", [])[:limit]:
        examples.append({
            "tournament_id": tournament["tournament_id"],
            "prompt_mode": tournament["prompt_mode"],
            "question": snippet(tournament["question"]),
            "gold_answer": tournament["gold_answer"],
            "branches": [
                {
                    "attempt_index": a["attempt_index"],
                    "temperature": a["temperature"],
                    "is_correct": a["is_correct"],
                    "extracted_answer": a["extracted_answer"],
                    "classification": "correct" if a["is_correct"] else a.get("wrong_classification", "unknown"),
                }
                for a in tournament["attempts"]
            ],
        })
    return examples


def best_by_arch(rows: Sequence[Dict[str, object]], arch: str) -> Optional[Dict[str, object]]:
    candidates = [r for r in rows if r["architecture"] == arch]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            row["metrics"]["top1_tournament_acc"],
            row["metrics"]["pairwise_acc"],
            -row["metrics"]["cycle_rate"],
        ),
    )


def write_md(path: Path, result: Dict[str, object]) -> None:
    kind = result["run_kind"]
    if kind == "expanded":
        title = "# Expanded Clean GSM8K Linear Transfer"
        verdict_lines = [
            f"EXPANDED_CLEAN_GSM8K_VERDICT = {result['expanded_clean_gsm8k_verdict']}",
            f"EXPANDED_LINEAR_TRANSFER_VERDICT = {result['expanded_linear_transfer_verdict']}",
        ]
    else:
        title = "# Clean GSM8K Extreme Transfer"
        verdict_lines = [
            f"CLEAN_GSM8K_VERDICT = {result['clean_gsm8k_verdict']}",
            f"CLEAN_TRANSFER_VERDICT = {result['clean_transfer_verdict']}",
        ]
    lines = [title, "", *verdict_lines, "", "## 1. Generation Summary", ""]
    gen = result["generation_summary"]
    for key in (
        "prompts_processed",
        "attempts_generated",
        "clean_attempts",
        "correct_clean_attempts",
        "incorrect_clean_attempts",
        "clean_tournaments_kept",
        "correct_candidate_count_distribution",
        "random_top1_baseline",
        "near_miss_fraction",
    ):
        if key in gen:
            lines.append(f"- {key}: `{gen.get(key)}`")
    lines.extend(["", "## 2. Clean Tournament Examples", ""])
    for t in result["clean_tournament_examples"]:
        lines.extend([
            f"### Tournament {t['tournament_id']} mode={t['prompt_mode']}",
            "",
            f"Prompt: {t['question']}",
            "",
            f"Gold answer: `{t['gold_answer']}`",
            "",
        ])
        for branch in t["branches"]:
            lines.append(
                f"- branch={branch['attempt_index']} correct={branch['is_correct']} "
                f"answer=`{branch['extracted_answer']}` temp={branch['temperature']} "
                f"class=`{branch['classification']}`"
            )
        lines.append("")
    lines.extend([
        "## 3. Feature Capture Summary",
        "",
        f"- features: `{result['features_path']}`",
        f"- n_tournaments: `{result['feature_summary']['n_tournaments']}`",
        f"- n_candidates: `{result['feature_summary']['n_candidates']}`",
        f"- tap_layers: `{result['feature_summary']['tap_layers']}`",
        "",
        "## 4. HH-Trained AntisymLinear / NoNorm Transfer Table",
        "",
        "| config | architecture | top1 | over_random | pairwise | condorcet | cycle | margin_mean | margin_std |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in result["transfer_table"]:
        metrics = row["metrics"]
        lines.append(
            f"| `{row['config']}` | {row['architecture']} | "
            f"{rate(metrics['top1_tournament_acc'])} | {rate(metrics['top1_over_random_baseline'])} | "
            f"{rate(metrics['pairwise_acc'])} | {rate(metrics['condorcet_winner_rate'])} | "
            f"{rate(metrics['cycle_rate'])} | {rate(metrics['margin_mean'])} | {rate(metrics['margin_std'])} |"
        )
    lines.extend([
        "",
        "## 5. Best-Head Comparison",
        "",
    ])
    for label, row in (
        ("best AntisymLinear", result["best_antisymlinear"]),
        ("best NoNorm", result["best_nonorm"]),
        ("best overall linear/NoNorm", result["best_hh_trained"]),
    ):
        if not row:
            lines.append(f"- {label}: `NA`")
            continue
        m = row["metrics"]
        lines.append(
            f"- {label}: `{row['config']}` / `{row['architecture']}` "
            f"top1={m['top1_tournament_acc']:.3f} pairwise={m['pairwise_acc']:.3f} cycle={m['cycle_rate']:.3f}"
        )
    lines.extend([
        "",
        "## 6. Random Baseline",
        "",
        f"actual random_top1_baseline = `{result['random_top1_baseline']:.3f}`",
        "",
        "## 7. Historical Comparisons",
        "",
        "`hh_to_math_transfer_probe_2026-05-16` is historical/confounded because "
        "`math_data_validity_2026-05-16` marked the old pilot `TRUNCATION_CONFOUNDED`.",
        "",
        f"- old confounded path: `{result['old_confounded_pilot'].get('path')}`",
        f"- old confounded transfer verdict: `{result['old_confounded_pilot'].get('transfer_verdict')}`",
        f"- clean n=5 micro path: `{result['clean_micro_baseline'].get('path')}`",
        f"- clean n=5 micro verdict: `{result['clean_micro_baseline'].get('clean_transfer_verdict')}`",
        "",
        "## 8. Commands Run",
        "",
        "```bash",
        "venv/bin/python -u utilities/tests/manual/evaluate_hh_transfer_on_clean_gsm8k_extreme.py --features opi/taps/probes/clean_gsm8k_expanded_tap_features_2026-05-16.pt --hh-capture rpe/evaluator/hh_layer_states_200_rltt.pt --output opi/taps/probes/clean_gsm8k_expanded_transfer_2026-05-16.json",
        "```",
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
    features_path = output_path(args.features)
    hh_path = output_path(args.hh_capture)
    out_json = output_path(args.output)
    out_md = output_path(args.output_md) if args.output_md else out_json.with_suffix(".md")
    if not features_path.exists():
        raise SystemExit(f"missing features: {features_path}")
    if not hh_path.exists():
        raise SystemExit("HH_CAPTURE_MISSING")

    feature_payload = torch.load(features_path, map_location="cpu", weights_only=False)
    generation_path, generation = load_generation_payload(feature_payload)
    context = clean_context(generation)
    if not context["allowed"]:
        raise SystemExit(f"clean generation verdict is {context['clean_verdict']}; transfer eval blocked")

    hh_payload = torch.load(hh_path, map_location="cpu", weights_only=False)
    records = feature_payload["records"]
    baseline = random_top1_baseline(records)
    train_idx, eval_idx = split_indices(len(hh_payload["packs"]), args.heldout, args.seed)

    transfer_rows: List[Dict[str, object]] = []
    for config in MATH_CONFIGS:
        print(f"training/evaluating {config}", flush=True)
        chosen, rejected = build_hh_features(hh_payload, config)
        feats_by_record = config_features(records, config)
        for architecture in ("AntisymLinear", "AntisymLinearNoNorm"):
            head, train_metrics = train_hh_head(
                architecture, config, chosen, rejected, train_idx, eval_idx, args, device)
            head = head.to(device)
            matrices = [score_matrix(head, feats, device) for feats in feats_by_record]
            metrics = evaluate_matrices(matrices, records, baseline)
            transfer_rows.append({
                "config": config,
                "architecture": architecture,
                "train_metrics": train_metrics,
                "metrics": metrics,
            })

    best = max(
        transfer_rows,
        key=lambda row: (
            row["metrics"]["top1_tournament_acc"],
            row["metrics"]["pairwise_acc"],
            -row["metrics"]["cycle_rate"],
        ),
    ) if transfer_rows else None
    transfer_verdict = linear_verdict(best, baseline, len(records), context["kind"])
    best_antisym = best_by_arch(transfer_rows, "AntisymLinear")
    best_nonorm = best_by_arch(transfer_rows, "AntisymLinearNoNorm")
    correct_dist = Counter(int(row["labels"].sum().item()) for row in records)

    result = {
        "run_kind": context["kind"],
        "features_path": repo_path(features_path),
        "hh_capture": repo_path(hh_path),
        "generation_json": repo_path(generation_path),
        "generation_summary": generation.get("summary", {}),
        "feature_summary": feature_payload.get("meta", {}),
        "random_top1_baseline": baseline,
        "correct_candidate_count_distribution": dict(sorted(correct_dist.items())),
        "best_hh_trained": best,
        "best_antisymlinear": best_antisym,
        "best_nonorm": best_nonorm,
        "transfer_table": transfer_rows,
        "clean_tournament_examples": examples_from_generation(generation),
        "old_confounded_pilot": old_confounded_summary(),
        "clean_micro_baseline": clean_micro_summary(out_json),
        "commands_run": [
            "venv/bin/python -u utilities/tests/manual/evaluate_hh_transfer_on_clean_gsm8k_extreme.py --features opi/taps/probes/clean_gsm8k_expanded_tap_features_2026-05-16.pt --hh-capture rpe/evaluator/hh_layer_states_200_rltt.pt --output opi/taps/probes/clean_gsm8k_expanded_transfer_2026-05-16.json",
        ],
    }
    if context["kind"] == "expanded":
        result["expanded_clean_gsm8k_verdict"] = context["clean_verdict"]
        result["expanded_linear_transfer_verdict"] = transfer_verdict
    else:
        result["clean_gsm8k_verdict"] = context["clean_verdict"]
        result["clean_transfer_verdict"] = transfer_verdict

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    write_md(out_md, result)
    if context["kind"] == "expanded":
        print(f"EXPANDED_CLEAN_GSM8K_VERDICT = {context['clean_verdict']}", flush=True)
        print(f"EXPANDED_LINEAR_TRANSFER_VERDICT = {transfer_verdict}", flush=True)
    else:
        print(f"CLEAN_GSM8K_VERDICT = {context['clean_verdict']}", flush=True)
        print(f"CLEAN_TRANSFER_VERDICT = {transfer_verdict}", flush=True)
    print(f"Wrote {out_json}", flush=True)
    print(f"Wrote {out_md}", flush=True)


if __name__ == "__main__":
    main()
