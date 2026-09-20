"""Post-hoc HH-to-math transfer probe over cached branch features.

This script performs no generation and loads no language model. It inventories
the cached HH captures and math tournament artifacts, then trains tiny fp32
AntisymLinear heads on the 200-example HH capture only if the inventory is
sufficient for a zero-shot transfer readout.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
    condorcet_winner_rate,
    config_dim,
    config_vector,
    cycle_rate,
    pairwise_accuracy,
    tournament_top1_accuracy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REPORT = PROJECT_ROOT / "opi/taps/probes/hh_to_math_transfer_probe_2026-05-16.md"
DEFAULT_JSON = PROJECT_ROOT / "opi/taps/probes/hh_to_math_transfer_probe_2026-05-16.json"
DEFAULT_INVENTORY = Path("/tmp/inventory_listing.txt")
HH_LAYER_RLTT = PROJECT_ROOT / "rpe/evaluator/hh_layer_states_200_rltt.pt"
HH_LAYER_THINKING = PROJECT_ROOT / "rpe/evaluator/hh_layer_states_200_thinking.pt"
MATH_FEATURES = PROJECT_ROOT / "opi/taps/probes/math_branch_tap_features.pt"
MATH_TOURNAMENTS = PROJECT_ROOT / "opi/taps/probes/math_branch_tournaments_rltt.json"
MATH_HEADS = PROJECT_ROOT / "opi/taps/probes/math_single_state_tap_heads.pt"


HEAD_CLASSES = {
    "AntisymLinear": AntisymLinearHead,
    "AntisymLinearNoNorm": AntisymLinearNoNorm,
}
TRANSFER_ORDER = {"TRANSFER_GOOD": 0, "TRANSFER_BORDERLINE": 1, "TRANSFER_POOR": 2}
RECOMMENDED = {
    "TRANSFER_GOOD": "Queue full-split HH capture (overnight, 9-12h) for properly-powered transfer",
    "TRANSFER_POOR": "Investigate domain mismatch; pivot to code as Track B training domain",
    "TRANSFER_BORDERLINE": "Queue full-split HH capture AND build small code-branch eval set",
    "TOURNAMENTS_INSUFFICIENT": "Regenerate GSM8K tournaments (cheaper than MATH) for adequate eval sample",
    "HH_INSUFFICIENT": "Run overnight HH full-split capture before transfer probe",
    "BOTH_INSUFFICIENT": "Run overnight HH full-split capture before transfer probe",
    "UNCLEAR": "Run overnight HH full-split capture before transfer probe",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-listing", default=str(DEFAULT_INVENTORY))
    parser.add_argument("--hh-capture", default=str(HH_LAYER_RLTT))
    parser.add_argument("--math-features", default=str(MATH_FEATURES))
    parser.add_argument("--math-tournaments", default=str(MATH_TOURNAMENTS))
    parser.add_argument("--math-heads", default=str(MATH_HEADS))
    parser.add_argument("--output-md", default=str(DEFAULT_REPORT))
    parser.add_argument("--output-json", default=str(DEFAULT_JSON))
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


def run_git(args: Sequence[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return proc.stdout.strip()
    except Exception:
        return ""


def rate(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{value:.3f}"


def snippet(text: object, limit: int = 220) -> str:
    s = " ".join(str(text).split())
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


def tensor_shape(value: object) -> Optional[List[int]]:
    if torch.is_tensor(value):
        return list(value.shape)
    return None


def inspect_hh_capture(path: Path) -> Dict[str, object]:
    row: Dict[str, object] = {
        "file": repo_path(path),
        "exists": path.exists(),
        "loadable": False,
        "usable_for_transfer": False,
        "top_level_type": None,
        "top_level_keys": [],
        "examples": 0,
        "layers": [],
        "loops": [],
        "first_level_shapes": {},
        "notes": "",
    }
    if not path.exists():
        row["notes"] = "missing"
        return row
    obj = torch.load(path, map_location="cpu", weights_only=False)
    row["loadable"] = True
    row["top_level_type"] = type(obj).__name__
    if isinstance(obj, dict):
        row["top_level_keys"] = list(obj.keys())
        meta = obj.get("meta", {})
        packs = obj.get("packs", [])
        row["examples"] = int(meta.get("n_examples", len(packs)) or len(packs))
        if packs and isinstance(packs[0], dict):
            first = packs[0]
            row["first_level_shapes"] = {}
            if "chosen" in first and "rejected" in first:
                pooled = first["chosen"].get("pooled") if isinstance(first["chosen"], dict) else None
                if isinstance(pooled, dict):
                    layers = sorted(int(k) for k in pooled.keys())
                    loop_counts = []
                    shapes = {}
                    for layer, value in pooled.items():
                        shapes[str(layer)] = tensor_shape(value)
                        if torch.is_tensor(value) and value.ndim >= 2:
                            loop_counts.append(int(value.shape[0]))
                    row["layers"] = layers
                    row["loops"] = sorted(set(loop_counts))
                    row["first_level_shapes"] = shapes
                    row["usable_for_transfer"] = (
                        row["examples"] >= 200
                        and set(layers) >= {24, 36, 47}
                        and 4 in row["loops"]
                    )
                else:
                    row["notes"] = "chosen/rejected present but pooled layer dict missing"
            elif "chosen_states" in first and "rejected_states" in first:
                states = first.get("chosen_states", [])
                row["loops"] = [len(states)] if isinstance(states, list) else []
                row["first_level_shapes"] = {
                    "chosen_states": [tensor_shape(t) for t in states] if isinstance(states, list) else None,
                    "rejected_states": [
                        tensor_shape(t) for t in first.get("rejected_states", [])
                    ] if isinstance(first.get("rejected_states"), list) else None,
                }
                row["notes"] = "per-token loop capture only; lacks 24/36/47 pooled layer dict"
    return row


def inspect_math_feature_pt(path: Path) -> Dict[str, object]:
    row: Dict[str, object] = {
        "file": repo_path(path),
        "exists": path.exists(),
        "loadable": False,
        "n_tournaments": 0,
        "source_mix": {},
        "has_candidate_texts": False,
        "has_labels": False,
        "has_features": False,
        "has_prompt_gold": False,
        "usable": False,
        "notes": "",
    }
    if not path.exists():
        row["notes"] = "missing"
        return row
    payload = torch.load(path, map_location="cpu", weights_only=False)
    row["loadable"] = True
    records = payload.get("records", []) if isinstance(payload, dict) else []
    row["n_tournaments"] = len(records)
    row["source_mix"] = dict(Counter(str(r.get("source", "unknown")) for r in records))
    row["has_candidate_texts"] = bool(records) and all("candidate_texts" in r for r in records)
    row["has_labels"] = bool(records) and all(torch.is_tensor(r.get("labels")) for r in records)
    row["has_features"] = bool(records) and all(torch.is_tensor(r.get("pooled")) for r in records)
    row["has_prompt_gold"] = bool(records) and all("question" in r and "gold_answer" in r for r in records)
    row["usable"] = (
        row["n_tournaments"] >= 30
        and row["has_candidate_texts"]
        and row["has_labels"]
        and row["has_features"]
        and row["has_prompt_gold"]
    )
    shapes = Counter(tuple(r["pooled"].shape) for r in records if torch.is_tensor(r.get("pooled")))
    row["notes"] = f"pooled_shapes={dict(shapes)}"
    return row


def inspect_math_tournament_json(path: Path) -> Dict[str, object]:
    row: Dict[str, object] = {
        "file": repo_path(path),
        "exists": path.exists(),
        "loadable": False,
        "n_tournaments": 0,
        "source_mix": {},
        "has_candidate_texts": False,
        "has_labels": False,
        "has_features": False,
        "has_prompt_gold": False,
        "usable": False,
        "notes": "",
    }
    if not path.exists():
        row["notes"] = "missing"
        return row
    payload = json.loads(path.read_text(encoding="utf-8"))
    row["loadable"] = True
    tournaments = payload.get("tournaments", []) if isinstance(payload, dict) else []
    row["n_tournaments"] = len(tournaments)
    row["source_mix"] = dict(Counter(str(t.get("source", "unknown")) for t in tournaments))
    row["has_prompt_gold"] = bool(tournaments) and all("question" in t and "gold_answer" in t for t in tournaments)
    row["has_candidate_texts"] = bool(tournaments) and all(
        all("candidate_text" in a for a in t.get("attempts", [])) for t in tournaments
    )
    row["has_labels"] = bool(tournaments) and all(
        all("is_correct" in a for a in t.get("attempts", [])) for t in tournaments
    )
    row["has_features"] = False
    row["usable"] = False
    row["notes"] = "candidate texts and verifier labels present; hidden features are stored separately"
    return row


def inventory_verdict(hh_rows: Sequence[Dict[str, object]], tournament_rows: Sequence[Dict[str, object]]) -> str:
    hh_ok = any(bool(row.get("usable_for_transfer")) for row in hh_rows)
    tournament_count = sum(int(row.get("n_tournaments", 0)) for row in tournament_rows if row.get("usable"))
    tourney_ok = tournament_count >= 30
    if hh_ok and tourney_ok:
        return "SUFFICIENT"
    if hh_ok and not tourney_ok:
        return "TOURNAMENTS_INSUFFICIENT"
    if not hh_ok and tourney_ok:
        return "HH_INSUFFICIENT"
    return "BOTH_INSUFFICIENT"


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


def split_indices(n: int, heldout: int, seed: int) -> Tuple[List[int], List[int]]:
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    eval_idx = sorted(indices[: min(heldout, n)])
    train_idx = sorted(indices[min(heldout, n) :])
    if not train_idx:
        train_idx = eval_idx
    return train_idx, eval_idx


def build_hh_features(hh_payload: Dict[str, object], config: str) -> Tuple[torch.Tensor, torch.Tensor]:
    chosen: List[torch.Tensor] = []
    rejected: List[torch.Tensor] = []
    for pack in hh_payload["packs"]:
        chosen.append(hh_config_vector(pack["chosen"]["pooled"], config).to(dtype=torch.float32))
        rejected.append(hh_config_vector(pack["rejected"]["pooled"], config).to(dtype=torch.float32))
    return torch.stack(chosen, dim=0), torch.stack(rejected, dim=0)


@torch.no_grad()
def hh_loss(head: torch.nn.Module, chosen: torch.Tensor, rejected: torch.Tensor, idx: Sequence[int], device: torch.device) -> float:
    if not idx:
        return float("nan")
    left = chosen[list(idx)].to(device)
    right = rejected[list(idx)].to(device)
    logits = head(left, right)
    loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    return float(loss.detach().cpu())


@torch.no_grad()
def hh_acc(head: torch.nn.Module, chosen: torch.Tensor, rejected: torch.Tensor, idx: Sequence[int], device: torch.device) -> float:
    left = chosen[list(idx)].to(device)
    right = rejected[list(idx)].to(device)
    logits = head(left, right)
    return float((logits > 0).to(torch.float32).mean().detach().cpu())


def train_hh_head(
    head_name: str,
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
    head = HEAD_CLASSES[head_name](config_dim(config)).to(device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_left = chosen[list(train_idx)].to(device)
    train_right = rejected[list(train_idx)].to(device)
    n = train_left.shape[0]
    target = torch.ones(n, device=device, dtype=torch.float32)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    losses: List[float] = []
    eval_losses: List[float] = []
    best_loss = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    best_epoch = -1
    stale = 0
    for epoch in range(args.epochs):
        head.train()
        perm = torch.randperm(n, generator=generator, device=device)
        total = 0.0
        for start in range(0, n, args.batch_size):
            batch = perm[start : start + args.batch_size]
            logits = head(train_left[batch], train_right[batch])
            loss = F.binary_cross_entropy_with_logits(logits, target[batch])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * int(batch.numel())
        epoch_loss = total / max(n, 1)
        losses.append(epoch_loss)
        head.eval()
        eval_loss = hh_loss(head, chosen, rejected, eval_idx, device)
        eval_losses.append(eval_loss)
        if eval_loss + 1e-7 < best_loss:
            best_loss = eval_loss
            best_epoch = epoch
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            break
    head.load_state_dict(best_state)
    head.eval()
    loss_decreased = bool(losses and losses[-1] < losses[0])
    if head_name == "AntisymLinear" and not loss_decreased:
        raise RuntimeError(
            f"AntisymLinear failed to converge for {config}: "
            f"loss_first={losses[0]:.6f} loss_last={losses[-1]:.6f}"
        )
    metrics = {
        "config": config,
        "architecture": head_name,
        "dim": config_dim(config),
        "epochs_run": len(losses),
        "best_epoch": best_epoch + 1,
        "loss_first": losses[0] if losses else float("nan"),
        "loss_last": losses[-1] if losses else float("nan"),
        "eval_loss_first": eval_losses[0] if eval_losses else float("nan"),
        "eval_loss_best": best_loss,
        "hh_train_acc": hh_acc(head, chosen, rejected, train_idx, device),
        "hh_heldout_acc": hh_acc(head, chosen, rejected, eval_idx, device),
    }
    return head.to("cpu"), metrics


def math_config_features(records: Sequence[Dict[str, object]], config: str) -> List[torch.Tensor]:
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
        top = torch.topk(totals, k=2).values
        margins.append(float(top[0] - top[1]))
    if not margins:
        return float("nan"), float("nan")
    return float(mean(margins)), float(pstdev(margins))


def evaluate_matrices(
    matrices: Sequence[torch.Tensor],
    records: Sequence[Dict[str, object]],
    indices: Sequence[int],
) -> Dict[str, object]:
    selected_mats = [matrices[i] for i in indices]
    labels = [records[i]["labels"] for i in indices]
    margin_mean, margin_std = margin_stats(selected_mats)
    return {
        "n_tournaments": len(indices),
        "top1_tournament_acc": tournament_top1_accuracy(selected_mats, labels),
        "pairwise_acc": pairwise_accuracy(selected_mats, labels),
        "condorcet_winner_rate": condorcet_winner_rate(selected_mats, labels),
        "cycle_rate": cycle_rate(selected_mats, triplets_per_matrix=1000, seed=42),
        "margin_mean": margin_mean,
        "margin_std": margin_std,
    }


def evaluate_head_on_math(
    head: torch.nn.Module,
    features_by_record: Sequence[torch.Tensor],
    records: Sequence[Dict[str, object]],
    device: torch.device,
) -> Tuple[List[torch.Tensor], Dict[str, object]]:
    head = head.to(device)
    head.eval()
    matrices = [score_matrix(head, feats, device) for feats in features_by_record]
    all_indices = list(range(len(records)))
    combined = evaluate_matrices(matrices, records, all_indices)
    per_source = {}
    sources = sorted(set(str(row.get("source", "unknown")) for row in records))
    for source in sources:
        idx = [i for i, row in enumerate(records) if str(row.get("source", "unknown")) == source]
        per_source[source] = evaluate_matrices(matrices, records, idx)
    return matrices, {"all": combined, "per_source": per_source}


def load_math_trained_heads(path: Path, device: torch.device) -> Tuple[Dict[str, torch.nn.Module], Dict[str, object]]:
    if not path.exists():
        return {}, {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    heads = {}
    for config, row in payload.get("heads", {}).items():
        head = AntisymLinearHead(int(row["dim"]))
        head.load_state_dict(row["state_dict"], strict=True)
        head.to(device)
        head.eval()
        heads[config] = head
    return heads, payload.get("meta", {})


def config_transfer_verdict(delta: Optional[float]) -> str:
    if delta is None or math.isnan(delta):
        return "NO_MATH_BASELINE"
    if delta <= 0.05:
        return "TRANSFER_GOOD"
    if delta >= 0.10:
        return "TRANSFER_POOR"
    return "TRANSFER_BORDERLINE"


def overall_transfer_verdict(rows: Sequence[Dict[str, object]]) -> str:
    comparable = [row for row in rows if row.get("config_verdict") in TRANSFER_ORDER]
    if len(comparable) < 4:
        return "UNCLEAR"
    return max((row["config_verdict"] for row in comparable), key=lambda v: TRANSFER_ORDER[v])


def pick_best_antisym(results: Dict[str, Dict[str, object]]) -> Tuple[str, Dict[str, object]]:
    candidates = [
        (config, row)
        for config, by_arch in results.items()
        for arch, row in by_arch.items()
        if arch == "AntisymLinear"
    ]
    return max(
        candidates,
        key=lambda item: (
            item[1]["math_eval"]["all"]["top1_tournament_acc"],
            item[1]["math_eval"]["all"]["pairwise_acc"],
            item[0],
        ),
    )


def sample_inspection(
    config: str,
    row: Dict[str, object],
    records: Sequence[Dict[str, object]],
    limit_each: int = 5,
) -> Dict[str, List[Dict[str, object]]]:
    matrices = row["matrices"]
    hits: List[Dict[str, object]] = []
    misses: List[Dict[str, object]] = []
    for idx, (mat, record) in enumerate(zip(matrices, records)):
        labels = record["labels"]
        totals = mat.sum(dim=1)
        order = torch.argsort(totals, descending=True)
        pred = int(order[0])
        margin = float(totals[order[0]] - totals[order[1]]) if len(order) > 1 else 0.0
        item = {
            "tournament_id": int(record.get("tournament_id", idx)),
            "source": str(record.get("source", "unknown")),
            "prompt_snippet": snippet(record.get("question", ""), 260),
            "gold_answer": str(record.get("gold_answer", "")),
            "predicted_index": pred,
            "margin": margin,
            "branches": [
                {
                    "index": j,
                    "label": bool(labels[j]),
                    "score_total": float(totals[j]),
                    "snippet": snippet(text, 180),
                }
                for j, text in enumerate(record.get("candidate_texts", []))
            ],
        }
        if bool(labels[pred]):
            if len(hits) < limit_each:
                hits.append(item)
        elif len(misses) < limit_each:
            misses.append(item)
        if len(hits) >= limit_each and len(misses) >= limit_each:
            break
    return {"correct": hits, "missed": misses, "config": config}


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> List[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return lines


def write_report(result: Dict[str, object], path: Path) -> None:
    lines: List[str] = [
        "# HH-to-Math Transfer Probe (2026-05-16)",
        "",
        f"INVENTORY_VERDICT = {result['inventory_verdict']}",
        f"TRANSFER_VERDICT  = {result['transfer_verdict']}",
        f"RECOMMENDED_NEXT  = {result['recommended_next']}",
        "",
        "## Provenance",
        "",
        f"- Timestamp: `{result['timestamp']}`",
        f"- Git commit: `{result['git_commit'] or 'unavailable'}`",
        f"- Device: `{result['device']}`",
        f"- Command: `{result['command']}`",
        f"- Inventory listing: `{result['inventory_listing']}`",
        f"- HH capture used: `{result.get('hh_capture_used', 'not run')}`",
        f"- Math comparison split: `{result.get('math_comparison_note', 'not run')}` "
        f"(n={len(result.get('math_comparison_indices', []))})",
        "",
        "## Matched Inventory Paths",
        "",
    ]
    lines.extend(f"- `{path}`" for path in result["matched_paths"])
    lines.extend(["", "## Inventory Summary", "", "### HH Captures", ""])
    hh_rows = []
    for row in result["hh_captures"]:
        hh_rows.append([
            f"`{row['file']}`",
            ",".join(map(str, row.get("layers", []))) or "NA",
            ",".join(map(str, row.get("loops", []))) or "NA",
            row.get("examples", 0),
            "Y" if row.get("usable_for_transfer") else "N",
            row.get("notes", ""),
        ])
    lines.extend(markdown_table(["file", "layers", "loops", "examples", "usable", "notes"], hh_rows))
    lines.extend(["", "### Tournament Data", ""])
    tournament_rows = []
    for row in result["tournament_data"]:
        tournament_rows.append([
            f"`{row['file']}`",
            row.get("n_tournaments", 0),
            json.dumps(row.get("source_mix", {}), sort_keys=True),
            "Y" if row.get("has_features") else "N",
            "Y" if row.get("has_labels") else "N",
            "Y" if row.get("usable") else "N",
            row.get("notes", ""),
        ])
    lines.extend(markdown_table(
        ["file", "n_tournaments", "source mix", "has_features", "has_labels", "usable", "notes"],
        tournament_rows,
    ))

    if result["transfer_verdict"] == "NOT_RUN":
        lines.extend(["", "## Transfer Probe", "", "Not run because inventory was insufficient."])
    else:
        lines.extend(["", "## Per-Config Transfer Table", ""])
        transfer_rows = []
        for row in result["transfer_rows"]:
            transfer_rows.append([
                f"`{row['config']}`",
                row["architecture"],
                rate(row.get("hh_trained_top1")),
                rate(row.get("math_trained_top1")),
                rate(row.get("delta")),
                row.get("config_verdict", "NA"),
            ])
        lines.extend(markdown_table(
            ["config", "architecture", "hh_trained_top1", "math_trained_top1", "delta", "verdict"],
            transfer_rows,
        ))
        lines.extend(["", "## HH-Trained All-Tournament Metrics", ""])
        metric_rows = []
        for config, by_arch in result["transfer_metrics"].items():
            for arch, row in by_arch.items():
                metrics = row["math_eval"]["all"]
                metric_rows.append([
                    f"`{config}`",
                    arch,
                    rate(metrics["top1_tournament_acc"]),
                    rate(metrics["pairwise_acc"]),
                    rate(metrics["condorcet_winner_rate"]),
                    rate(metrics["cycle_rate"]),
                    rate(metrics["margin_mean"]),
                    rate(metrics["margin_std"]),
                ])
        lines.extend(markdown_table(
            ["config", "architecture", "top1", "pairwise", "condorcet", "cycle", "margin_mean", "margin_std"],
            metric_rows,
        ))
        best = result["best_config"]
        lines.extend(["", f"## Per-Source Breakdown: `{best['config']}` / {best['architecture']}", ""])
        source_rows = []
        for source, metrics in best["per_source"].items():
            source_rows.append([
                source,
                metrics["n_tournaments"],
                rate(metrics["top1_tournament_acc"]),
                rate(metrics["pairwise_acc"]),
                rate(metrics["condorcet_winner_rate"]),
                rate(metrics["cycle_rate"]),
            ])
        lines.extend(markdown_table(["source", "n", "top1", "pairwise", "condorcet", "cycle"], source_rows))
        lines.extend([
            "",
            "## Cycle-Rate Observations",
            "",
            "AntisymLinearNoNorm cycle rate was exactly 0.000 for every config. "
            "No measurement-bug hard stop was triggered.",
            "",
            "## Sample Inspection",
            "",
            f"Best AntisymLinear config by all-tournament top-1/pairwise: `{result['samples']['config']}`.",
            "",
            "### Correct Branch Selected",
            "",
        ])
        for item in result["samples"]["correct"]:
            lines.extend([
                f"- Tournament {item['tournament_id']} ({item['source']}), margin={item['margin']:.4f}, gold=`{item['gold_answer']}`",
                f"  Prompt: {item['prompt_snippet']}",
            ])
            for branch in item["branches"]:
                lines.append(
                    f"  - branch {branch['index']} label={branch['label']} "
                    f"total={branch['score_total']:.4f}: {branch['snippet']}"
                )
        lines.extend(["", "### Missed Correct Branch", ""])
        for item in result["samples"]["missed"]:
            lines.extend([
                f"- Tournament {item['tournament_id']} ({item['source']}), margin={item['margin']:.4f}, gold=`{item['gold_answer']}`",
                f"  Prompt: {item['prompt_snippet']}",
            ])
            for branch in item["branches"]:
                lines.append(
                    f"  - branch {branch['index']} label={branch['label']} "
                    f"total={branch['score_total']:.4f}: {branch['snippet']}"
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def json_sanitize(obj: object) -> object:
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items() if k != "matrices"}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(v) for v in obj]
    return obj


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    listing_path = Path(args.inventory_listing)
    matched_paths = listing_path.read_text(encoding="utf-8").splitlines() if listing_path.exists() else []
    if not matched_paths:
        raise SystemExit(f"Inventory listing is empty or missing: {listing_path}")

    hh_paths = [
        HH_LAYER_RLTT,
        HH_LAYER_THINKING,
        PROJECT_ROOT / "rpe/evaluator/hh_loop_states_200_rltt.pt",
        PROJECT_ROOT / "rpe/evaluator/hh_loop_states_200_thinking.pt",
    ]
    hh_rows = [inspect_hh_capture(path) for path in hh_paths if path.exists()]
    tournament_rows = []
    if Path(args.math_features).exists():
        tournament_rows.append(inspect_math_feature_pt(Path(args.math_features)))
    if Path(args.math_tournaments).exists():
        tournament_rows.append(inspect_math_tournament_json(Path(args.math_tournaments)))
    verdict = inventory_verdict(hh_rows, tournament_rows)

    result: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": run_git(["rev-parse", "HEAD"]),
        "device": str(device),
        "command": " ".join(sys.argv),
        "inventory_listing": str(listing_path),
        "matched_paths": matched_paths,
        "hh_captures": hh_rows,
        "tournament_data": tournament_rows,
        "inventory_verdict": verdict,
        "transfer_verdict": "NOT_RUN",
        "recommended_next": RECOMMENDED.get(verdict, "Run overnight HH full-split capture before transfer probe"),
        "transfer_rows": [],
        "transfer_metrics": {},
    }

    if verdict != "SUFFICIENT":
        write_report(result, Path(args.output_md))
        Path(args.output_json).write_text(json.dumps(json_sanitize(result), indent=2) + "\n", encoding="utf-8")
        print(f"INVENTORY_VERDICT = {verdict}")
        print("TRANSFER_VERDICT  = NOT_RUN")
        print(f"RECOMMENDED_NEXT  = {result['recommended_next']}")
        return

    hh_payload = torch.load(Path(args.hh_capture), map_location="cpu", weights_only=False)
    feature_payload = torch.load(Path(args.math_features), map_location="cpu", weights_only=False)
    records = feature_payload["records"]
    train_idx, hh_eval_idx = split_indices(len(hh_payload["packs"]), args.heldout, args.seed)
    math_heads, math_head_meta = load_math_trained_heads(Path(args.math_heads), device)
    comparison_indices = list(math_head_meta.get("eval_indices", range(len(records))))
    comparison_note = "math_head_eval_split" if "eval_indices" in math_head_meta else "all_tournaments_non_independent"

    transfer_metrics: Dict[str, Dict[str, object]] = defaultdict(dict)
    transfer_rows: List[Dict[str, object]] = []
    math_features_cache: Dict[str, List[torch.Tensor]] = {}

    for config in MATH_CONFIGS:
        print(f"Training HH heads for {config}", flush=True)
        chosen, rejected = build_hh_features(hh_payload, config)
        math_features_cache[config] = math_config_features(records, config)
        for head_name in ("AntisymLinear", "AntisymLinearNoNorm"):
            head, train_metrics = train_hh_head(
                head_name,
                config,
                chosen,
                rejected,
                train_idx,
                hh_eval_idx,
                args,
                device,
            )
            matrices, eval_metrics = evaluate_head_on_math(
                head,
                math_features_cache[config],
                records,
                device,
            )
            if head_name == "AntisymLinearNoNorm":
                nonzero_sources = [
                    ("all", eval_metrics["all"]["cycle_rate"]),
                    *[(src, metrics["cycle_rate"]) for src, metrics in eval_metrics["per_source"].items()],
                ]
                bad = [(src, cyc) for src, cyc in nonzero_sources if abs(float(cyc)) > 0.0]
                if bad:
                    raise RuntimeError(f"AntisymLinearNoNorm nonzero cycle rate for {config}: {bad}")
            transfer_metrics[config][head_name] = {
                "train_metrics": train_metrics,
                "math_eval": eval_metrics,
                "matrices": matrices,
            }
            compare_metrics = evaluate_matrices(matrices, records, comparison_indices)
            math_top1 = None
            delta = None
            config_verdict = "NO_MATH_BASELINE"
            if head_name == "AntisymLinear" and config in math_heads:
                math_mats, _ = evaluate_head_on_math(
                    math_heads[config],
                    math_features_cache[config],
                    records,
                    device,
                )
                math_compare = evaluate_matrices(math_mats, records, comparison_indices)
                math_top1 = float(math_compare["top1_tournament_acc"])
                delta = math_top1 - float(compare_metrics["top1_tournament_acc"])
                config_verdict = config_transfer_verdict(delta)
            transfer_rows.append({
                "config": config,
                "architecture": head_name,
                "hh_trained_top1": float(compare_metrics["top1_tournament_acc"]),
                "math_trained_top1": math_top1,
                "delta": delta,
                "config_verdict": config_verdict,
                "comparison_split": comparison_note,
                "hh_all_top1": float(eval_metrics["all"]["top1_tournament_acc"]),
            })

    transfer_verdict = overall_transfer_verdict(transfer_rows)
    recommended_next = RECOMMENDED.get(transfer_verdict, RECOMMENDED["UNCLEAR"])
    best_config, best_row = pick_best_antisym(transfer_metrics)
    samples = sample_inspection(best_config, best_row, records)

    result.update({
        "transfer_verdict": transfer_verdict,
        "recommended_next": recommended_next,
        "hh_capture_used": repo_path(Path(args.hh_capture)),
        "hh_train_indices": train_idx,
        "hh_heldout_indices": hh_eval_idx,
        "math_comparison_indices": comparison_indices,
        "math_comparison_note": comparison_note,
        "transfer_rows": transfer_rows,
        "transfer_metrics": transfer_metrics,
        "best_config": {
            "config": best_config,
            "architecture": "AntisymLinear",
            "per_source": best_row["math_eval"]["per_source"],
        },
        "samples": samples,
    })
    write_report(result, Path(args.output_md))
    Path(args.output_json).write_text(json.dumps(json_sanitize(result), indent=2) + "\n", encoding="utf-8")
    print(f"INVENTORY_VERDICT = {verdict}")
    print(f"TRANSFER_VERDICT  = {transfer_verdict}")
    print(f"RECOMMENDED_NEXT  = {recommended_next}")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
