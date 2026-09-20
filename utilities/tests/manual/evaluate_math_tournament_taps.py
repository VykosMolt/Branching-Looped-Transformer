"""Evaluate trained math tap heads on generated branch tournaments."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from math_bg_probe_lib import (
    AntisymLinearHead,
    DEFAULT_OUTPUT_DIR,
    MATH_CONFIGS,
    config_dim,
    config_vector,
    output_path,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-pt", default=f"{DEFAULT_OUTPUT_DIR}/math_branch_tap_features.pt")
    p.add_argument("--heads-pt", default=f"{DEFAULT_OUTPUT_DIR}/math_single_state_tap_heads.pt")
    p.add_argument("--split", choices=("eval", "train", "all"), default="eval")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-json", default=f"{DEFAULT_OUTPUT_DIR}/evaluate_math_tournament_taps.json")
    p.add_argument("--output-md", default=f"{DEFAULT_OUTPUT_DIR}/evaluate_math_tournament_taps.md")
    return p.parse_args()


def load_heads(path: Path, device: torch.device) -> Dict[str, AntisymLinearHead]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    heads = {}
    for config, row in payload["heads"].items():
        head = AntisymLinearHead(int(row["dim"]))
        head.load_state_dict(row["state_dict"], strict=True)
        head.to(device)
        head.eval()
        heads[config] = head
    return heads


def select_indices(meta: Dict[str, object], n: int, split: str) -> List[int]:
    if split == "train":
        return list(meta.get("train_indices", range(n)))
    if split == "eval":
        return list(meta.get("eval_indices", range(n)))
    return list(range(n))


def config_features(records: Sequence[Dict[str, object]], config: str) -> List[torch.Tensor]:
    return [torch.stack([config_vector(row, config) for row in rec["pooled"]], dim=0) for rec in records]


@torch.no_grad()
def score_matrix(head: AntisymLinearHead, feats: torch.Tensor, device: torch.device) -> torch.Tensor:
    feats = feats.to(device)
    k = feats.shape[0]
    mat = torch.zeros((k, k), device=device)
    for i in range(k):
        for j in range(k):
            if i != j:
                mat[i, j] = head(feats[i:i+1], feats[j:j+1])[0]
    return mat


def summarize_matrices(
    matrices: Sequence[torch.Tensor],
    records: Sequence[Dict[str, object]],
    indices: Sequence[int],
) -> Dict[str, float | int]:
    pair_total = 0
    pair_correct = 0
    top1_total = 0
    top1_correct = 0
    condorcet_correct = 0
    cycle_count = 0
    margins = []
    for mat, idx in zip(matrices, indices):
        labels = records[idx]["labels"]
        k = mat.shape[0]
        for c in torch.nonzero(labels, as_tuple=False).flatten().tolist():
            for r in torch.nonzero(~labels, as_tuple=False).flatten().tolist():
                pair_total += 1
                pair_correct += int(float(mat[c, r]) > 0.0)
        totals = mat.sum(dim=1)
        order = torch.argsort(totals, descending=True)
        pred = int(order[0])
        top1_total += 1
        top1_correct += int(bool(labels[pred]))
        margins.append(float(totals[order[0]] - totals[order[1]]) if k > 1 else 0.0)
        beats = mat[pred, torch.arange(k, device=mat.device) != pred] > 0
        condorcet_correct += int(bool(labels[pred]) and bool(beats.all().item()))
        has_cycle = False
        if k >= 3:
            for a in range(k):
                for b in range(k):
                    for c in range(k):
                        if a == b or b == c or a == c:
                            continue
                        if mat[a, b] > 0 and mat[b, c] > 0 and mat[c, a] > 0:
                            has_cycle = True
                            break
                    if has_cycle:
                        break
                if has_cycle:
                    break
        cycle_count += int(has_cycle)
    return {
        "n_tournaments": int(top1_total),
        "n_pairs": int(pair_total),
        "pairwise_acc": pair_correct / max(pair_total, 1),
        "top1_tournament_acc": top1_correct / max(top1_total, 1),
        "condorcet_correct_rate": condorcet_correct / max(top1_total, 1),
        "cycle_rate": cycle_count / max(top1_total, 1),
        "margin_mean": float(np.mean(margins)) if margins else float("nan"),
        "margin_std": float(np.std(margins)) if margins else float("nan"),
    }


def write_md(path: Path, result: Dict[str, object]) -> None:
    rows = sorted(
        result["per_config"],
        key=lambda r: (-r["top1_tournament_acc"], -r["pairwise_acc"], r["config"]),
    )
    lines = [
        "# Math Tournament Tap Evaluation",
        "",
        f"Split: `{result['split']}`",
        "",
        "| Config | top1 | pairwise | condorcet | cycle | margin mean | n |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['config']}` | {row['top1_tournament_acc']:.3f} | "
            f"{row['pairwise_acc']:.3f} | {row['condorcet_correct_rate']:.3f} | "
            f"{row['cycle_rate']:.3f} | {row['margin_mean']:.3f} | "
            f"{row['n_tournaments']} |"
        )
    lines.extend([
        "",
        "All scores are pairwise branch-selection scores over generated math branches.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    features = torch.load(output_path(args.features_pt), map_location="cpu", weights_only=False)
    head_payload = torch.load(output_path(args.heads_pt), map_location="cpu", weights_only=False)
    records = features["records"]
    indices = select_indices(head_payload.get("meta", {}), len(records), args.split)
    heads = load_heads(output_path(args.heads_pt), device)

    per_config = []
    matrices_by_config: Dict[str, List[torch.Tensor]] = {}
    for config in MATH_CONFIGS:
        if config not in heads:
            continue
        feats_by_record = config_features(records, config)
        matrices = [score_matrix(heads[config], feats_by_record[idx], device).detach().cpu() for idx in indices]
        matrices_by_config[config] = matrices
        row = summarize_matrices(matrices, records, indices)
        row["config"] = config
        per_config.append(row)
        print(
            f"{config}: top1={row['top1_tournament_acc']:.3f} "
            f"pair={row['pairwise_acc']:.3f} cycle={row['cycle_rate']:.3f}"
        )

    if matrices_by_config:
        ensemble_mats = []
        for pos in range(len(indices)):
            mats = [matrices_by_config[c][pos] for c in matrices_by_config]
            ensemble_mats.append(torch.stack(mats, dim=0).mean(dim=0))
        row = summarize_matrices(ensemble_mats, records, indices)
        row["config"] = "ensemble_mean_logits"
        per_config.append(row)
        print(
            f"ensemble_mean_logits: top1={row['top1_tournament_acc']:.3f} "
            f"pair={row['pairwise_acc']:.3f} cycle={row['cycle_rate']:.3f}"
        )

    result = {
        "split": args.split,
        "indices": indices,
        "features_meta": features.get("meta", {}),
        "heads_meta": head_payload.get("meta", {}),
        "per_config": per_config,
    }
    out_json = output_path(args.output_json)
    out_md = output_path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8")
    write_md(out_md, result)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
