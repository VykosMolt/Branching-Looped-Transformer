"""Retrain v3-style tiny hidden-origin heads under salvage splits/folds."""
from __future__ import annotations

import argparse
import math
import os
import time
from collections import defaultdict

import torch

from bg_hidden_origin_split_salvage_common import (
    SALVAGE_DATASETS_PT,
    SALVAGE_HEADS_PT,
    SALVAGE_ROOT,
    USEFUL_CONFIGS,
    compact_head_for_json,
    ensure_salvage_root,
    md_table,
    rate,
    rel,
    write_json,
    write_md,
)
from bg_hidden_origin_tap_common import config_dim
from train_bg_hidden_origin_taps import ARCHITECTURES, flip_diagnostics, pairs_for_config, train_one
from bg_hidden_origin_diversity_v3_common import direction_from_state_dict


OUT_JSON = SALVAGE_ROOT / "salvage_training_log.json"
OUT_MD = SALVAGE_ROOT / "salvage_training_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--score-l2", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def training_verdict(heads: list[dict[str, object]], trainable_folds: int) -> str:
    if trainable_folds <= 0:
        return "DATA_LIMITED"
    valid = [
        row for row in heads
        if row.get("flip_diagnostics", {}).get("passes")
        and math.isfinite(float(row.get("metrics", {}).get("validation_pairwise_accuracy", float("nan"))))
    ]
    if not valid:
        return "BLOCKED" if heads else "DATA_LIMITED"
    best_val = max(float(row["metrics"]["validation_pairwise_accuracy"]) for row in valid)
    if best_val >= 0.60 and trainable_folds >= 3:
        return "READY"
    if best_val > 0.50:
        return "WEAK"
    return "OVERFIT" if any(float(row["metrics"].get("train_pairwise_accuracy", 0.0)) >= 0.70 for row in valid) else "DATA_LIMITED"


def main() -> int:
    args = parse_args()
    started = time.time()
    ensure_salvage_root()
    if not SALVAGE_DATASETS_PT.exists():
        payload = {"BG_HIDDEN_ORIGIN_SALVAGE_TRAINING_VERDICT": "BLOCKED", "verdict": "BLOCKED", "blocker": "missing salvage datasets"}
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# Hidden-Origin Salvage Training", "", "BG_HIDDEN_ORIGIN_SALVAGE_TRAINING_VERDICT = BLOCKED"])
        print("BG_HIDDEN_ORIGIN_SALVAGE_TRAINING_VERDICT = BLOCKED", flush=True)
        return 1
    if SALVAGE_HEADS_PT.exists() and not os.environ.get("FORCE_RERUN"):
        payload = torch.load(SALVAGE_HEADS_PT, map_location="cpu", weights_only=False)
        heads = list(payload.get("heads") or [])
        json_payload = {k: v for k, v in payload.items() if k != "heads"}
        json_payload["heads"] = [compact_head_for_json(row) for row in heads]
        write_json(OUT_JSON, json_payload)
        verdict = str(payload.get("verdict", "READY"))
        lines = [
            "# Hidden-Origin Salvage Training",
            "",
            f"BG_HIDDEN_ORIGIN_SALVAGE_TRAINING_VERDICT = {verdict}",
            "",
            f"- trainable_folds: `{payload.get('trainable_folds')}`",
            f"- trained_heads: `{payload.get('trained_heads')}`",
            f"- device: `{payload.get('device')}`",
            "",
            "Heads are old-style antisymmetric linear comparators trained only on existing branch-pair labels. No tap score is used as a label.",
            "",
            "## Config Summary",
            "",
        ]
        lines.extend(md_table(list(payload.get("config_summary") or [])[:120], ["mode", "config", "architecture", "heads", "best_val", "best_train", "flip_pass"]))
        best_rows = sorted(
            [compact_head_for_json(row) for row in heads if row.get("flip_diagnostics", {}).get("passes")],
            key=lambda row: float(row["metrics"].get("validation_pairwise_accuracy", -1.0)),
            reverse=True,
        )[:40]
        lines.extend(["", "## Best Passing Heads", ""])
        lines.extend(
            md_table(
                [
                    {
                        "mode": row["mode_name"],
                        "fold": row["fold_id"],
                        "config": row["config"],
                        "architecture": row["architecture"],
                        "val": rate(row["metrics"]["validation_pairwise_accuracy"]),
                        "train": rate(row["metrics"]["train_pairwise_accuracy"]),
                        "flip": row["flip_diagnostics"]["passes"],
                    }
                    for row in best_rows
                ],
                ["mode", "fold", "config", "architecture", "val", "train", "flip"],
            )
        )
        lines.extend(["", f"Wrote `{rel(SALVAGE_HEADS_PT)}`."])
        write_md(OUT_MD, lines)
        print(f"BG_HIDDEN_ORIGIN_SALVAGE_TRAINING_VERDICT = {verdict}", flush=True)
        print(f"Wrote {rel(SALVAGE_HEADS_PT)}", flush=True)
        return 0 if verdict != "BLOCKED" else 1
    dataset = torch.load(SALVAGE_DATASETS_PT, map_location="cpu", weights_only=False)
    modes = dict(dataset.get("modes") or {})
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    seeds = [42, 43, 44]
    lrs = [1e-4, 3e-4, 1e-3]

    heads: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    trainable_folds = 0
    for mode_key, mode_payload in sorted(modes.items()):
        record = mode_payload["record"]
        if record.get("mode_type") == "eval_only":
            training_rows.append({"mode_key": mode_key, "status": "skipped", "reason": "eval-only fixed selector mode"})
            continue
        pairs = list(mode_payload.get("pairs") or [])
        train_all = [pair for pair in pairs if pair.get("split") == "train"]
        val_all = [pair for pair in pairs if pair.get("split") == "val"]
        if len(train_all) < 2 or len(val_all) < 1:
            training_rows.append(
                {
                    "mode_key": mode_key,
                    "mode_name": record.get("mode_name"),
                    "fold_id": record.get("fold_id"),
                    "status": "skipped",
                    "reason": "insufficient train/val pairs",
                    "train_pairs": len(train_all),
                    "val_pairs": len(val_all),
                }
            )
            continue
        trainable_folds += 1
        for config in USEFUL_CONFIGS:
            train_pairs = pairs_for_config(train_all, config)
            val_pairs = pairs_for_config(val_all, config)
            if len(train_pairs) < 2 or len(val_pairs) < 1:
                training_rows.append(
                    {
                        "mode_key": mode_key,
                        "mode_name": record.get("mode_name"),
                        "fold_id": record.get("fold_id"),
                        "config": config,
                        "status": "skipped",
                        "reason": "insufficient feature coverage",
                        "train_pairs": len(train_pairs),
                        "val_pairs": len(val_pairs),
                    }
                )
                continue
            for architecture in ARCHITECTURES:
                for seed in seeds:
                    for lr in lrs:
                        print(f"training salvage {record['mode_name']} {record['fold_id']} {architecture} {config} seed={seed} lr={lr}", flush=True)
                        head, metrics = train_one(
                            architecture=architecture,
                            config=config,
                            seed=seed,
                            lr=lr,
                            train_pairs=train_pairs,
                            val_pairs=val_pairs,
                            args=args,
                            device=device,
                        )
                        metrics["mode_name"] = record["mode_name"]
                        metrics["fold_id"] = record["fold_id"]
                        metrics["mode_type"] = record["mode_type"]
                        state_dict = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                        flip = flip_diagnostics(head.to(device), val_pairs, config, device)
                        row = {
                            "head_group": "hidden_origin_split_salvage_heads",
                            "mode_key": mode_key,
                            "mode_name": record["mode_name"],
                            "fold_id": record["fold_id"],
                            "mode_type": record["mode_type"],
                            "architecture": architecture,
                            "config": config,
                            "dim": config_dim(config),
                            "state_dict": state_dict,
                            "direction": direction_from_state_dict(state_dict),
                            "metrics": metrics,
                            "flip_diagnostics": flip,
                            "contamination_flags": record.get("contamination_flags", {}),
                        }
                        heads.append(row)
                        training_rows.append(compact_head_for_json(row) | {"status": "trained"})
                        head.to("cpu")

    verdict = training_verdict(heads, trainable_folds)
    best_by_mode: dict[str, dict[str, object]] = {}
    for row in heads:
        if not row.get("flip_diagnostics", {}).get("passes"):
            continue
        key = str(row["mode_key"])
        old = best_by_mode.get(key)
        if old is None or float(row["metrics"].get("validation_pairwise_accuracy", -1.0)) > float(old["metrics"].get("validation_pairwise_accuracy", -1.0)):
            best_by_mode[key] = row
    config_summary = []
    grouped = defaultdict(list)
    for row in heads:
        grouped[(row["mode_name"], row["config"], row["architecture"])].append(row)
    for (mode_name, config, architecture), vals in sorted(grouped.items()):
        best = max(vals, key=lambda row: float(row["metrics"].get("validation_pairwise_accuracy", -1.0)))
        config_summary.append(
            {
                "mode": mode_name,
                "config": config,
                "architecture": architecture,
                "heads": len(vals),
                "best_val": best["metrics"]["validation_pairwise_accuracy"],
                "best_train": best["metrics"]["train_pairwise_accuracy"],
                "flip_pass": best["flip_diagnostics"]["passes"],
            }
        )
    payload = {
        "BG_HIDDEN_ORIGIN_SALVAGE_TRAINING_VERDICT": verdict,
        "verdict": verdict,
        "dataset": rel(SALVAGE_DATASETS_PT),
        "device": str(device),
        "architectures": list(ARCHITECTURES),
        "configs": list(USEFUL_CONFIGS),
        "seeds": seeds,
        "lrs": lrs,
        "trainable_folds": trainable_folds,
        "trained_heads": len(heads),
        "heads": heads,
        "best_by_mode": {key: compact_head_for_json(row) for key, row in best_by_mode.items()},
        "training_rows": training_rows,
        "config_summary": config_summary,
        "anti_degeneracy": {
            "random_swap": True,
            "target_sign_flip": True,
            "flip_diagnostics_required": True,
            "tap_score_not_used_as_training_label": True,
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(payload, SALVAGE_HEADS_PT)
    json_payload = {k: v for k, v in payload.items() if k != "heads"}
    json_payload["heads"] = [compact_head_for_json(row) for row in heads]
    write_json(OUT_JSON, json_payload)

    lines = [
        "# Hidden-Origin Salvage Training",
        "",
        f"BG_HIDDEN_ORIGIN_SALVAGE_TRAINING_VERDICT = {verdict}",
        "",
        f"- trainable_folds: `{trainable_folds}`",
        f"- trained_heads: `{len(heads)}`",
        f"- device: `{device}`",
        "",
        "Heads are old-style antisymmetric linear comparators trained only on existing branch-pair labels. No tap score is used as a label.",
        "",
        "## Config Summary",
        "",
    ]
    lines.extend(md_table(config_summary[:120], ["mode", "config", "architecture", "heads", "best_val", "best_train", "flip_pass"]))
    best_rows = sorted(
        [compact_head_for_json(row) for row in heads if row.get("flip_diagnostics", {}).get("passes")],
        key=lambda row: float(row["metrics"].get("validation_pairwise_accuracy", -1.0)),
        reverse=True,
    )[:40]
    lines.extend(["", "## Best Passing Heads", ""])
    lines.extend(
        md_table(
            [
                {
                    "mode": row["mode_name"],
                    "fold": row["fold_id"],
                    "config": row["config"],
                    "architecture": row["architecture"],
                    "val": rate(row["metrics"]["validation_pairwise_accuracy"]),
                    "train": rate(row["metrics"]["train_pairwise_accuracy"]),
                    "flip": row["flip_diagnostics"]["passes"],
                }
                for row in best_rows
            ],
            ["mode", "fold", "config", "architecture", "val", "train", "flip"],
        )
    )
    lines.extend(["", f"Wrote `{rel(SALVAGE_HEADS_PT)}`."])
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_SALVAGE_TRAINING_VERDICT = {verdict}", flush=True)
    print(f"Wrote {rel(SALVAGE_HEADS_PT)}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
