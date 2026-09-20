"""Train science-specific tiny heads on a task-disjoint science MCQ split."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

THIS_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == THIS_DIR:
    sys.path.pop(0)
sys.path.insert(0, str(THIS_DIR))

from code_branch_pilot_lib import REPORT_DIR, output_path, repo_path, write_json
from evaluate_bg_fixed_configs_cross_domain import reconstruct_heads, rate
from evaluate_heads_on_science_natural_distractors import (
    best,
    compact,
    evaluate_records,
    random_top1,
    records_for_eval,
    transfer_verdict,
)
from math_bg_probe_lib import MATH_CONFIGS, config_dim
from train_code_specific_tiny_heads_and_eval import HEAD_CLASSES, feature_map, train_head


INPUT_JSON = REPORT_DIR / "science_natural_distractor_set_2026-05-17.json"
FEATURES_PT = REPORT_DIR / "science_natural_distractor_features_2026-05-17.pt"
HEADS_PT = REPORT_DIR / "bg_head_registry_2026-05-17.pt"
OUTPUT_JSON = REPORT_DIR / "science_specific_tiny_head_control_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "science_specific_tiny_head_control_2026-05-17.md"
ARCHITECTURES = ("AntisymLinear", "AntisymLinearNoNorm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(INPUT_JSON))
    parser.add_argument("--features", default=str(FEATURES_PT))
    parser.add_argument("--heads", default=str(HEADS_PT))
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
    return json.loads(output_path(path).read_text(encoding="utf-8"))


def split_records(records: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_bucket[str(row.get("subdomain_bucket", "unknown"))].append(row)
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for bucket, rows in by_bucket.items():
        shuffled = rows[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        if n < 4:
            train.extend(shuffled)
            continue
        train_n = max(1, int(round(0.70 * n)))
        val_n = max(1, int(round(0.15 * n)))
        if train_n + val_n >= n:
            train_n = max(1, n - 2)
            val_n = 1
        train.extend(shuffled[:train_n])
        val.extend(shuffled[train_n : train_n + val_n])
        test.extend(shuffled[train_n + val_n :])
    if not test and len(records) >= 10:
        shuffled = records[:]
        rng.shuffle(shuffled)
        test = shuffled[-max(2, round(0.15 * len(shuffled))) :]
        test_ids = {row["task_id"] for row in test}
        train = [row for row in shuffled if row["task_id"] not in test_ids]
        val = train[-max(1, round(0.15 * len(train))) :]
        val_ids = {row["task_id"] for row in val}
        train = [row for row in train if row["task_id"] not in val_ids]
    meta = {
        "split": "task_disjoint_stratified_by_subdomain_where_possible",
        "train_tasks": len(train),
        "val_tasks": len(val),
        "test_tasks": len(test),
        "train_bucket_counts": dict(defaultdict(int, {b: sum(1 for r in train if r.get("subdomain_bucket") == b) for b in by_bucket})),
        "val_bucket_counts": dict(defaultdict(int, {b: sum(1 for r in val if r.get("subdomain_bucket") == b) for b in by_bucket})),
        "test_bucket_counts": dict(defaultdict(int, {b: sum(1 for r in test if r.get("subdomain_bucket") == b) for b in by_bucket})),
    }
    return train, val, test, meta


def pairs_from_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for row in records:
        correct = [uid for uid, ok in zip(row["candidate_uids"], row["labels"].tolist()) if ok]
        incorrect = [uid for uid, ok in zip(row["candidate_uids"], row["labels"].tolist()) if not ok]
        for c_uid in correct:
            for i_uid in incorrect:
                pairs.append({"task_id": row["task_id"], "preferred_uid": c_uid, "rejected_uid": i_uid})
    return pairs


def science_head_infos(heads: list[tuple[str, str, torch.nn.Module, dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for config, architecture, head, metrics in heads:
        out.append({
            "head_family": "SCIENCE",
            "architecture": architecture,
            "family_architecture": "SCIENCE_NoNorm" if architecture == "AntisymLinearNoNorm" else "SCIENCE_AntisymLinear",
            "config": config,
            "dim": config_dim(config),
            "head": head,
            "train_metrics": metrics,
        })
    return out


def verdict(best_science: dict[str, Any] | None, best_existing: dict[str, Any] | None, baseline: float, n_test: int) -> str:
    if n_test < 8 or not best_science or not best_existing:
        return "INSUFFICIENT"
    sci_pair = float(best_science["metrics"]["pairwise_acc"])
    existing_pair = float(best_existing["metrics"]["pairwise_acc"])
    if sci_pair >= existing_pair + 0.05 and sci_pair >= 0.60:
        return "SPECIALIST_HELPS"
    if existing_pair >= sci_pair - 0.05 and transfer_verdict(best_existing, baseline) == "GOOD":
        return "GENERAL_SUFFICIENT"
    return "INSUFFICIENT"


def write_md(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# Science-Specific Tiny-Head Control",
        "",
        f"SCIENCE_SPECIFIC_HEAD_VERDICT = {payload['science_specific_head_verdict']}",
        "",
        f"- split: `{payload['split']}`",
        f"- best science-specific: `{s.get('best_science')}`",
        f"- best HH: `{s.get('best_hh')}`",
        f"- best code: `{s.get('best_code')}`",
        f"- best existing HH/CODE: `{s.get('best_existing')}`",
        "",
        "## Science-Specific Rows",
        "",
        "| config | architecture | top1 | over_random | pairwise | cycle | margin_mean | margin_std |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload.get("science_rows", []):
        m = row["metrics"]
        lines.append(
            f"| `{row['config']}` | `{row['architecture']}` | {rate(m['top1_tournament_acc'])} | "
            f"{rate(m['top1_over_random_baseline'])} | {rate(m['pairwise_acc'])} | {rate(m['cycle_rate'])} | "
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
    set_verdict = data.get("science_distractor_set_verdict", "BLOCKED")
    if set_verdict != "READY":
        payload = {
            "science_specific_head_verdict": "NOT_RUN",
            "summary": {"SCIENCE_SPECIFIC_HEAD_VERDICT": "NOT_RUN", "reason": f"set_verdict={set_verdict}"},
            "outputs": {"json": repo_path(args.output), "md": repo_path(args.output_md)},
        }
        write_json(output_path(args.output), payload)
        write_md(output_path(args.output_md), payload)
        print("SCIENCE_SPECIFIC_HEAD_VERDICT = NOT_RUN")
        return
    features = torch.load(output_path(args.features), map_location="cpu", weights_only=False)
    registry = torch.load(output_path(args.heads), map_location="cpu", weights_only=False)
    records = records_for_eval(data, features)
    train_records, val_records, test_records, split = split_records(records, int(args.seed))
    train_pairs = pairs_from_records(train_records)
    val_pairs = pairs_from_records(val_records) or train_pairs[:]
    if len(test_records) < 8 or len(train_pairs) < 30:
        payload = {
            "science_specific_head_verdict": "INSUFFICIENT",
            "summary": {"SCIENCE_SPECIFIC_HEAD_VERDICT": "INSUFFICIENT", "train_pairs": len(train_pairs), "test_tasks": len(test_records)},
            "split": split,
            "outputs": {"json": repo_path(args.output), "md": repo_path(args.output_md)},
        }
        write_json(output_path(args.output), payload)
        write_md(output_path(args.output_md), payload)
        print("SCIENCE_SPECIFIC_HEAD_VERDICT = INSUFFICIENT")
        return
    pooled_by_uid = feature_map(features)
    trained: list[tuple[str, str, torch.nn.Module, dict[str, Any]]] = []
    for config in MATH_CONFIGS:
        print(f"training SCIENCE heads {config}", flush=True)
        for architecture in ARCHITECTURES:
            head, metrics = train_head(
                architecture=architecture,
                config=config,
                train_pairs=train_pairs,
                val_pairs=val_pairs,
                pooled_by_uid=pooled_by_uid,
                args=args,
                device=device,
            )
            trained.append((config, architecture, head, metrics))
    science_infos = science_head_infos(trained)
    existing_infos = reconstruct_heads(registry)
    science_rows = evaluate_records(test_records, science_infos, device, "science_heldout_test")
    existing_rows = evaluate_records(test_records, existing_infos, device, "science_heldout_test")
    baseline = random_top1(test_records)
    best_science = best(science_rows)
    best_hh = best([row for row in existing_rows if row["head_family"] == "HH"])
    best_code = best([row for row in existing_rows if row["head_family"] == "CODE"])
    best_existing = best(existing_rows)
    v = verdict(best_science, best_existing, baseline, len(test_records))
    payload = {
        "science_specific_head_verdict": v,
        "summary": {
            "SCIENCE_SPECIFIC_HEAD_VERDICT": v,
            "n_train_tasks": len(train_records),
            "n_val_tasks": len(val_records),
            "n_test_tasks": len(test_records),
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "random_top1_baseline": baseline,
            "best_science": compact(best_science),
            "best_hh": compact(best_hh),
            "best_code": compact(best_code),
            "best_existing": compact(best_existing),
        },
        "split": split,
        "science_rows": science_rows,
        "existing_rows": existing_rows,
        "outputs": {"json": repo_path(args.output), "md": repo_path(args.output_md)},
    }
    write_json(output_path(args.output), payload)
    write_md(output_path(args.output_md), payload)
    print(f"SCIENCE_SPECIFIC_HEAD_VERDICT = {v}")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
