"""Evaluate HH-trained AntisymLinear heads on balanced near-miss code tournaments."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Sequence

THIS_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == THIS_DIR:
    sys.path.pop(0)
import numpy as np
import torch
sys.path.insert(0, str(THIS_DIR))

from code_branch_pilot_lib import PROJECT_ROOT, REPORT_DIR, output_path, repo_path, write_json
from evaluate_hh_transfer_on_clean_gsm8k_extreme import (
    build_hh_features,
    config_features,
    evaluate_matrices,
    random_top1_baseline,
    score_matrix,
    split_indices,
    train_hh_head,
)
from math_bg_probe_lib import MATH_CONFIGS


FEATURES_PT = REPORT_DIR / "code_branch_near_miss_balanced_tap_features_2026-05-17.pt"
HH_CAPTURE = PROJECT_ROOT / "rpe/evaluator/hh_layer_states_200_rltt.pt"
OUTPUT_JSON = REPORT_DIR / "code_branch_near_miss_balanced_transfer_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "code_branch_near_miss_balanced_transfer_2026-05-17.md"
CUDA_AVAILABLE_AT_IMPORT = torch.cuda.is_available()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default=str(FEATURES_PT))
    parser.add_argument("--hh-capture", default=str(HH_CAPTURE))
    parser.add_argument("--output", default=str(OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--heldout", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def rate(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return "NA"
    if math.isnan(x):
        return "NA"
    return f"{x:.3f}"


def best_by_arch(rows: Sequence[dict[str, Any]], arch: str) -> dict[str, Any] | None:
    pool = [row for row in rows if row["architecture"] == arch]
    if not pool:
        return None
    return max(pool, key=lambda row: (
        row["metrics"]["top1_tournament_acc"],
        row["metrics"]["pairwise_acc"],
        -row["metrics"]["cycle_rate"],
    ))


def compact(row: dict[str, Any] | None) -> dict[str, Any] | str:
    if row is None:
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


def transfer_verdict(best: dict[str, Any] | None, baseline: float) -> str:
    if best is None:
        return "NOT_RUN"
    m = best["metrics"]
    top1 = float(m["top1_tournament_acc"])
    pairwise = float(m["pairwise_acc"])
    cycles = float(m["cycle_rate"])
    if top1 >= baseline + 0.15 and pairwise >= 0.60 and cycles <= 0.05:
        return "GOOD"
    if top1 >= baseline + 0.05 or pairwise >= 0.55:
        return "WEAK"
    return "POOR"


def subset_metrics(matrices: list[torch.Tensor], records: list[dict[str, Any]], baseline: float) -> dict[str, Any]:
    out = {}
    for name, pred in {
        "strict_clean": lambda row: row.get("strict_clean"),
        "diagnostic_runnable": lambda row: row.get("diagnostic_runnable"),
    }.items():
        idx = [i for i, row in enumerate(records) if pred(row)]
        if idx:
            out[name] = evaluate_matrices([matrices[i] for i in idx], [records[i] for i in idx], baseline)
    return out


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Code Near-Miss Balanced Transfer",
        "",
        f"BALANCED_TRANSFER_VERDICT = {payload['balanced_transfer_verdict']}",
        "",
        f"- primary_eval_set: `{payload['primary_eval_set']}`",
        f"- n_tournaments: `{payload['feature_summary']['n_tournaments']}`",
        f"- random_top1_baseline: `{payload['random_top1_baseline']:.3f}`",
        f"- best AntisymLinear row: `{payload['best_antisymlinear_compact']}`",
        f"- best NoNorm row: `{payload['best_nonorm_compact']}`",
        "",
        "## Transfer Table",
        "",
        "| config | architecture | top1 | over_random | pairwise | condorcet | cycle | margin_mean | margin_std |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["transfer_table"]:
        m = row["metrics"]
        lines.append(
            f"| `{row['config']}` | {row['architecture']} | {rate(m['top1_tournament_acc'])} | "
            f"{rate(m['top1_over_random_baseline'])} | {rate(m['pairwise_acc'])} | "
            f"{rate(m['condorcet_winner_rate'])} | {rate(m['cycle_rate'])} | "
            f"{rate(m['margin_mean'])} | {rate(m['margin_std'])} |"
        )
    lines.extend([
        "",
        "Small-n interpretation: this is a near-miss-balanced transfer signal check, not proof of general code transfer.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and not CUDA_AVAILABLE_AT_IMPORT:
        raise SystemExit("--device cuda requested but CUDA is not available")
    features_path = output_path(args.features)
    hh_path = output_path(args.hh_capture)
    out_json = output_path(args.output)
    out_md = Path(str(out_json.with_suffix(".md")) if not getattr(args, "output_md", "") else args.output_md)
    if not features_path.exists():
        raise SystemExit(f"missing features: {features_path}")
    if not hh_path.exists():
        raise SystemExit("HH_CAPTURE_MISSING")

    device = torch.device(args.device)
    feature_payload = torch.load(features_path, map_location="cpu", weights_only=False)
    records = list(feature_payload["records"])
    meta = feature_payload.get("meta", {})
    hh_payload = torch.load(hh_path, map_location="cpu", weights_only=False)
    train_idx, eval_idx = split_indices(len(hh_payload["packs"]), args.heldout, args.seed)
    baseline = random_top1_baseline(records)
    rows: list[dict[str, Any]] = []
    best = None
    best_matrices: list[torch.Tensor] = []
    for config in MATH_CONFIGS:
        print(f"training/evaluating {config}", flush=True)
        chosen, rejected = build_hh_features(hh_payload, config)
        feats_by_record = config_features(records, config)
        for architecture in ("AntisymLinear", "AntisymLinearNoNorm"):
            head, train_metrics = train_hh_head(architecture, config, chosen, rejected, train_idx, eval_idx, args, device)
            head = head.to(device)
            matrices = [score_matrix(head, feats, device) for feats in feats_by_record]
            metrics = evaluate_matrices(matrices, records, baseline)
            row = {
                "config": config,
                "architecture": architecture,
                "train_metrics": train_metrics,
                "metrics": metrics,
                "subset_metrics": subset_metrics(matrices, records, baseline),
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
    nonorm_cycle_bug = [
        {"config": row["config"], "cycle_rate": row["metrics"]["cycle_rate"]}
        for row in rows
        if row["architecture"] == "AntisymLinearNoNorm" and float(row["metrics"]["cycle_rate"]) > 1e-9
    ]
    if nonorm_cycle_bug:
        raise SystemExit(f"AntisymLinearNoNorm nonzero cycle rate: {nonorm_cycle_bug}")
    best_antisym = best_by_arch(rows, "AntisymLinear")
    best_nonorm = best_by_arch(rows, "AntisymLinearNoNorm")
    verdict = transfer_verdict(best, baseline)
    result = {
        "balanced_transfer_verdict": verdict,
        "BALANCED_TRANSFER_VERDICT": verdict,
        "features_path": repo_path(features_path),
        "hh_capture": repo_path(hh_path),
        "tournaments_json": meta.get("input_json"),
        "primary_eval_set": meta.get("primary_eval_set"),
        "feature_summary": meta,
        "random_top1_baseline": baseline,
        "transfer_table": rows,
        "best_hh_trained": best,
        "best_antisymlinear": best_antisym,
        "best_nonorm": best_nonorm,
        "best_hh_trained_compact": compact(best),
        "best_antisymlinear_compact": compact(best_antisym),
        "best_nonorm_compact": compact(best_nonorm),
        "best_subset_metrics": subset_metrics(best_matrices, records, baseline) if best_matrices else {},
        "commands_run": [
            "venv/bin/python -u utilities/tests/manual/evaluate_hh_transfer_on_code_near_miss_balanced.py",
        ],
        "small_n_caveat": "near-miss-balanced transfer signal only; do not claim broad code transfer.",
    }
    write_json(out_json, result)
    write_md(out_md, result)
    print(f"BALANCED_TRANSFER_VERDICT = {verdict}", flush=True)
    print(f"Wrote {out_json}", flush=True)
    print(f"Wrote {out_md}", flush=True)


if __name__ == "__main__":
    main()
