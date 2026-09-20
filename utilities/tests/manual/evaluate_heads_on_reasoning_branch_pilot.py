"""Evaluate registry heads on the reasoning branch pilot."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

THIS_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == THIS_DIR:
    sys.path.pop(0)
sys.path.insert(0, str(THIS_DIR))

from code_branch_pilot_lib import REPORT_DIR, output_path, repo_path, write_json
from evaluate_bg_fixed_configs_cross_domain import FIXED_CONFIGS, reconstruct_heads, rate
from train_code_specific_tiny_heads_and_eval import evaluate_set, feature_map


FEATURES_PT = REPORT_DIR / "reasoning_branch_tap_features_2026-05-17.pt"
HEADS_PT = REPORT_DIR / "bg_head_registry_2026-05-17.pt"
INPUT_JSON = REPORT_DIR / "reasoning_branch_pilot_2026-05-17.json"
OUTPUT_JSON = REPORT_DIR / "reasoning_branch_transfer_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "reasoning_branch_transfer_2026-05-17.md"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", default=str(FEATURES_PT))
    p.add_argument("--heads", default=str(HEADS_PT))
    p.add_argument("--input", default=str(INPUT_JSON))
    p.add_argument("--output", default=str(OUTPUT_JSON))
    p.add_argument("--output-md", default=str(OUTPUT_MD))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def records_for_reasoning(feature_payload: dict[str, Any]) -> list[dict[str, Any]]:
    pooled = feature_map(feature_payload)
    records = []
    for idx, row in enumerate(feature_payload.get("eval_sets", {}).get("reasoning_primary", []) or []):
        uids = [str(uid) for uid in row["candidate_uids"]]
        labels = [label == "correct" for label in row["labels"]]
        records.append({
            "tournament_id": idx,
            "task_id": row["task_id"],
            "source": row.get("dataset", "unknown"),
            "difficulty": "unknown",
            "function_name": "",
            "prompt": row.get("question", ""),
            "candidate_uids": uids,
            "label_names": list(row["labels"]),
            "labels": torch.tensor(labels, dtype=torch.bool),
            "pooled": torch.stack([pooled[uid] for uid in uids], dim=0),
        })
    return records


def best(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: (row["metrics"]["top1_tournament_acc"], row["metrics"]["pairwise_acc"], -row["metrics"]["cycle_rate"]))


def compact(row: dict[str, Any] | None) -> dict[str, Any] | str:
    if not row:
        return "NA"
    m = row["metrics"]
    return {"family": row.get("head_family"), "config": row["config"], "architecture": row["architecture"], "top1": m["top1_tournament_acc"], "over_random": m["top1_over_random_baseline"], "pairwise": m["pairwise_acc"], "cycle": m["cycle_rate"]}


def verdict(best_row: dict[str, Any] | None, baseline: float) -> str:
    if not best_row:
        return "NOT_RUN"
    m = best_row["metrics"]
    if m["top1_tournament_acc"] >= baseline + 0.15 and m["pairwise_acc"] >= 0.60 and m["cycle_rate"] <= 0.05:
        return "GOOD"
    if m["top1_tournament_acc"] >= baseline + 0.05 or m["pairwise_acc"] >= 0.55:
        return "WEAK"
    return "POOR"


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA unavailable")
    device = torch.device(args.device)
    features = torch.load(output_path(args.features), map_location="cpu", weights_only=False)
    registry = torch.load(output_path(args.heads), map_location="cpu", weights_only=False)
    heads = reconstruct_heads(registry)
    head_tuples = [(h["config"], h["architecture"], h["head"], h.get("train_metrics", {})) for h in heads]
    records = records_for_reasoning(features)
    baseline = sum(float(r["labels"].to(torch.float32).mean()) for r in records) / len(records)
    result = evaluate_set(set_name="reasoning_primary", records=records, heads=head_tuples, baseline=baseline, device=device)
    for row in result["transfer_table"]:
        match = next(h for h in heads if h["config"] == row["config"] and h["architecture"] == row["architecture"] and h["train_metrics"] == row["train_metrics"])
        row["head_family"] = match["head_family"]
        row["family_architecture"] = match["family_architecture"]
    best_overall = best(result["transfer_table"])
    v = verdict(best_overall, baseline)
    by_dataset = {}
    for dataset in sorted({r["source"] for r in records}):
        subset = [r for r in records if r["source"] == dataset]
        b = sum(float(r["labels"].to(torch.float32).mean()) for r in subset) / len(subset)
        rr = evaluate_set(set_name=f"reasoning_{dataset}", records=subset, heads=head_tuples, baseline=b, device=device)
        for row in rr["transfer_table"]:
            match = next(h for h in heads if h["config"] == row["config"] and h["architecture"] == row["architecture"] and h["train_metrics"] == row["train_metrics"])
            row["head_family"] = match["head_family"]
            row["family_architecture"] = match["family_architecture"]
        by_dataset[dataset] = {"baseline": b, "best": compact(best(rr["transfer_table"]))}
    payload = {
        "reasoning_transfer_verdict": v,
        "summary": {
            "REASONING_TRANSFER_VERDICT": v,
            "n_tournaments": len(records),
            "random_top1_baseline": baseline,
            "best_overall": compact(best_overall),
            "best_hh": compact(best([row for row in result["transfer_table"] if row.get("head_family") == "HH"])),
            "best_code": compact(best([row for row in result["transfer_table"] if row.get("head_family") == "CODE"])),
            "per_dataset": by_dataset,
        },
        "eval_results": result,
        "outputs": {"json": repo_path(args.output), "md": repo_path(args.output_md)},
    }
    write_json(output_path(args.output), payload)
    lines = [
        "# Reasoning Branch Transfer",
        "",
        f"REASONING_TRANSFER_VERDICT = {v}",
        "",
        f"- n_tournaments: `{len(records)}`",
        f"- random_top1_baseline: `{baseline}`",
        f"- best overall: `{payload['summary']['best_overall']}`",
        f"- best HH: `{payload['summary']['best_hh']}`",
        f"- best CODE: `{payload['summary']['best_code']}`",
        f"- per_dataset: `{by_dataset}`",
        "",
        "| family | config | architecture | top1 | over_random | pairwise | cycle |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in result["transfer_table"]:
        if row["config"] not in FIXED_CONFIGS:
            continue
        m = row["metrics"]
        lines.append(f"| `{row.get('head_family')}` | `{row['config']}` | `{row['architecture']}` | {rate(m['top1_tournament_acc'])} | {rate(m['top1_over_random_baseline'])} | {rate(m['pairwise_acc'])} | {rate(m['cycle_rate'])} |")
    Path(args.output_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"REASONING_TRANSFER_VERDICT = {v}")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
