"""Evaluate HH-trained and code-trained registry heads on natural MCQ distractors."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import torch

THIS_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == THIS_DIR:
    sys.path.pop(0)
sys.path.insert(0, str(THIS_DIR))

from code_branch_pilot_lib import REPORT_DIR, output_path, repo_path, write_json
from evaluate_bg_fixed_configs_cross_domain import reconstruct_heads, rate
from math_bg_probe_lib import MATH_CONFIGS, config_vector
from train_code_specific_tiny_heads_and_eval import evaluate_matrices, score_matrix


INPUT_JSON = REPORT_DIR / "reasoning_natural_distractor_set_2026-05-17.json"
FEATURES_PT = REPORT_DIR / "reasoning_natural_distractor_features_2026-05-17.pt"
HEADS_PT = REPORT_DIR / "bg_head_registry_2026-05-17.pt"
OUTPUT_JSON = REPORT_DIR / "reasoning_natural_distractor_transfer_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "reasoning_natural_distractor_transfer_2026-05-17.md"

FIXED_ROWS = {
    ("HH", "36_mean", "AntisymLinear"),
    ("HH", "47_concat_L1_L4", "AntisymLinearNoNorm"),
    ("CODE", "24_L4", "AntisymLinear"),
    ("CODE", "36_L4", "AntisymLinear"),
    ("CODE", "36_L4", "AntisymLinearNoNorm"),
    ("CODE", "47_L4", "AntisymLinearNoNorm"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default=str(FEATURES_PT))
    parser.add_argument("--heads", default=str(HEADS_PT))
    parser.add_argument("--input", default=str(INPUT_JSON))
    parser.add_argument("--output", default=str(OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(output_path(path).read_text(encoding="utf-8"))


def pooled_map(feature_payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        str(row["candidate_uid"]): row["pooled"].detach().cpu().to(torch.float32)
        for row in feature_payload.get("candidate_features", []) or []
    }


def records_for_eval(data: dict[str, Any], feature_payload: dict[str, Any]) -> list[dict[str, Any]]:
    by_uid = pooled_map(feature_payload)
    records = []
    for idx, row in enumerate(data.get("tournaments", []) or []):
        uids = [str(uid) for uid in row["candidate_uids"]]
        labels = [label == "correct" for label in row["labels"]]
        missing = [uid for uid in uids if uid not in by_uid]
        if missing:
            raise SystemExit(f"missing features for {row['task_id']}: {missing}")
        records.append({
            "tournament_id": idx,
            "task_id": row["task_id"],
            "source": row.get("dataset", "unknown"),
            "option_count": int(row.get("n_options", len(uids))),
            "candidate_uids": uids,
            "label_names": list(row["labels"]),
            "labels": torch.tensor(labels, dtype=torch.bool),
            "pooled": torch.stack([by_uid[uid] for uid in uids], dim=0),
        })
    return records


def random_top1(records: Sequence[dict[str, Any]]) -> float:
    return float(mean(float(row["labels"].to(torch.float32).mean()) for row in records)) if records else float("nan")


def config_features(records: Sequence[dict[str, Any]], config: str) -> list[torch.Tensor]:
    return [
        torch.stack([config_vector(candidate, config) for candidate in row["pooled"]], dim=0).to(torch.float32)
        for row in records
    ]


@torch.no_grad()
def evaluate_records(records: list[dict[str, Any]], heads: list[dict[str, Any]], device: torch.device, set_name: str) -> list[dict[str, Any]]:
    rows = []
    baseline = random_top1(records)
    for info in heads:
        feats_by_record = config_features(records, info["config"])
        head = info["head"].to(device)
        matrices = [score_matrix(head, feats, device) for feats in feats_by_record]
        head = head.to("cpu")
        rows.append({
            "eval_set": set_name,
            "head_family": info["head_family"],
            "family_architecture": info["family_architecture"],
            "architecture": info["architecture"],
            "config": info["config"],
            "metrics": evaluate_matrices(matrices, records, baseline),
        })
    return rows


def best(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: (
        float(row["metrics"].get("pairwise_acc", float("nan"))),
        float(row["metrics"].get("top1_tournament_acc", float("nan"))),
        -float(row["metrics"].get("cycle_rate", 1.0) or 0.0),
    ))


def compact(row: dict[str, Any] | None) -> dict[str, Any] | str:
    if not row:
        return "NA"
    m = row["metrics"]
    return {
        "family": row["head_family"],
        "config": row["config"],
        "architecture": row["architecture"],
        "top1": m["top1_tournament_acc"],
        "over_random": m["top1_over_random_baseline"],
        "pairwise": m["pairwise_acc"],
        "condorcet": m["condorcet_winner_rate"],
        "cycle": m["cycle_rate"],
        "margin_mean": m["margin_mean"],
        "margin_std": m["margin_std"],
    }


def clears_good(row: dict[str, Any] | None, baseline: float) -> bool:
    if not row:
        return False
    m = row["metrics"]
    return (
        float(m["pairwise_acc"]) >= 0.60
        and float(m["top1_tournament_acc"]) >= baseline + 0.15
        and float(m["cycle_rate"]) <= 0.05
    )


def transfer_verdict(row: dict[str, Any] | None, baseline: float) -> str:
    if not row:
        return "NOT_RUN"
    m = row["metrics"]
    if clears_good(row, baseline):
        return "GOOD"
    if float(m["top1_tournament_acc"]) >= baseline + 0.05 or float(m["pairwise_acc"]) >= 0.55:
        return "WEAK"
    return "POOR"


def specialist_verdict(best_hh: dict[str, Any] | None, best_code: dict[str, Any] | None, best_overall: dict[str, Any] | None, baseline: float, n_records: int) -> str:
    if n_records < 15 or not best_overall or not best_hh or not best_code:
        return "INSUFFICIENT"
    hh = best_hh["metrics"]
    code = best_code["metrics"]
    overall = best_overall["metrics"]
    top_gap = float(code["top1_tournament_acc"]) - float(hh["top1_tournament_acc"])
    pair_gap = float(code["pairwise_acc"]) - float(hh["pairwise_acc"])
    if top_gap >= 0.10 or pair_gap >= 0.10:
        return "SPECIALIST_NEEDED"
    hh_top_gap = float(overall["top1_tournament_acc"]) - float(hh["top1_tournament_acc"])
    hh_pair_gap = float(overall["pairwise_acc"]) - float(hh["pairwise_acc"])
    if hh_top_gap <= 0.05 and hh_pair_gap <= 0.05 and clears_good(best_hh, baseline):
        return "GENERAL_SUFFICIENT"
    if clears_good(best_hh, baseline) and clears_good(best_code, baseline) and abs(top_gap) < 0.10 and abs(pair_gap) < 0.10:
        return "BOTH_GOOD"
    return "INSUFFICIENT"


def subgroup_best(records: list[dict[str, Any]], heads: list[dict[str, Any]], device: torch.device, key: str) -> dict[str, Any]:
    out = {}
    values = sorted({row[key] for row in records})
    for value in values:
        subset = [row for row in records if row[key] == value]
        rows = evaluate_records(subset, heads, device, f"{key}_{value}")
        out[str(value)] = {
            "n_tournaments": len(subset),
            "random_top1_baseline": random_top1(subset),
            "best": compact(best(rows)),
            "best_hh": compact(best([row for row in rows if row["head_family"] == "HH"])),
            "best_code": compact(best([row for row in rows if row["head_family"] == "CODE"])),
        }
    return out


def write_md(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# Reasoning Natural Distractor Transfer",
        "",
        f"REASONING_DISTRACTOR_TRANSFER_VERDICT = {payload['reasoning_distractor_transfer_verdict']}",
        f"REASONING_SPECIALIST_VERDICT = {payload['reasoning_specialist_verdict']}",
        "",
        f"- n_tournaments: `{s['n_tournaments']}`",
        f"- n_candidates: `{s['n_candidates']}`",
        f"- random_top1_baseline: `{s['random_top1_baseline']}`",
        f"- best overall: `{s['best_overall']}`",
        f"- best HH: `{s['best_hh']}`",
        f"- best CODE: `{s['best_code']}`",
        f"- best NoNorm: `{s['best_nonorm']}`",
        f"- best AntisymLinear: `{s['best_antisymlinear']}`",
        "",
        "## Per-Dataset Breakdown",
        "",
        f"`{s['per_dataset']}`",
        "",
        "## Per-Option-Count Breakdown",
        "",
        f"`{s['per_option_count']}`",
        "",
        "## Fixed Config Table",
        "",
        "| family | config | architecture | top1 | over_random | pairwise | condorcet | cycle | margin_mean | margin_std |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["fixed_config_table"]:
        m = row["metrics"]
        lines.append(
            f"| `{row['head_family']}` | `{row['config']}` | `{row['architecture']}` | "
            f"{rate(m['top1_tournament_acc'])} | {rate(m['top1_over_random_baseline'])} | "
            f"{rate(m['pairwise_acc'])} | {rate(m['condorcet_winner_rate'])} | {rate(m['cycle_rate'])} | "
            f"{rate(m['margin_mean'])} | {rate(m['margin_std'])} |"
        )
    lines.extend([
        "",
        "## Per-Config Table",
        "",
        "| family | config | architecture | top1 | over_random | pairwise | condorcet | cycle | margin_mean | margin_std |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in payload["transfer_table"]:
        m = row["metrics"]
        lines.append(
            f"| `{row['head_family']}` | `{row['config']}` | `{row['architecture']}` | "
            f"{rate(m['top1_tournament_acc'])} | {rate(m['top1_over_random_baseline'])} | "
            f"{rate(m['pairwise_acc'])} | {rate(m['condorcet_winner_rate'])} | {rate(m['cycle_rate'])} | "
            f"{rate(m['margin_mean'])} | {rate(m['margin_std'])} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA unavailable")
    device = torch.device(args.device)
    data = load_json(args.input)
    features = torch.load(output_path(args.features), map_location="cpu", weights_only=False)
    registry = torch.load(output_path(args.heads), map_location="cpu", weights_only=False)
    heads = reconstruct_heads(registry)
    records = records_for_eval(data, features)
    baseline = random_top1(records)
    rows = evaluate_records(records, heads, device, "reasoning_natural_distractors")
    best_overall = best(rows)
    best_hh = best([row for row in rows if row["head_family"] == "HH"])
    best_code = best([row for row in rows if row["head_family"] == "CODE"])
    fixed = [
        row for row in rows
        if (row["head_family"], row["config"], row["architecture"]) in FIXED_ROWS
    ]
    payload = {
        "reasoning_distractor_transfer_verdict": transfer_verdict(best_overall, baseline),
        "reasoning_specialist_verdict": specialist_verdict(best_hh, best_code, best_overall, baseline, len(records)),
        "summary": {
            "REASONING_DISTRACTOR_TRANSFER_VERDICT": transfer_verdict(best_overall, baseline),
            "REASONING_SPECIALIST_VERDICT": specialist_verdict(best_hh, best_code, best_overall, baseline, len(records)),
            "n_tournaments": len(records),
            "n_candidates": sum(len(row["candidate_uids"]) for row in records),
            "random_top1_baseline": baseline,
            "best_overall": compact(best_overall),
            "best_hh": compact(best_hh),
            "best_code": compact(best_code),
            "best_nonorm": compact(best([row for row in rows if row["architecture"] == "AntisymLinearNoNorm"])),
            "best_antisymlinear": compact(best([row for row in rows if row["architecture"] == "AntisymLinear"])),
            "per_dataset": subgroup_best(records, heads, device, "source"),
            "per_option_count": subgroup_best(records, heads, device, "option_count"),
        },
        "transfer_table": rows,
        "fixed_config_table": fixed,
        "outputs": {"json": repo_path(args.output), "md": repo_path(args.output_md)},
    }
    write_json(output_path(args.output), payload)
    write_md(output_path(args.output_md), payload)
    print(f"REASONING_DISTRACTOR_TRANSFER_VERDICT = {payload['reasoning_distractor_transfer_verdict']}")
    print(f"REASONING_SPECIALIST_VERDICT = {payload['reasoning_specialist_verdict']}")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
