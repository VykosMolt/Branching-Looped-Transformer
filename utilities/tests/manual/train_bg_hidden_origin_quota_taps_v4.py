"""Train tiny hidden-origin selector taps on v4 quota dataset."""
from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from typing import Any, Sequence

import torch

from bg_hidden_origin_quota_v4_common import (
    ARCHITECTURES,
    CONFIGS,
    DATASET_V4_PT,
    HEADS_V4_PT,
    V4_ROOT,
    compact_head,
    config_dim,
    direction_from_state_dict,
    ensure_v4_root,
    flip_diagnostics,
    md_table,
    pairs_for_config,
    rate,
    rel,
    train_one,
    write_json,
    write_md,
)


OUT_JSON = V4_ROOT / "training_log_v4.json"
OUT_MD = V4_ROOT / "training_report_v4.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--score-l2", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def verdict_for(dataset_verdict: str, rows: Sequence[dict[str, Any]], train_pairs: int, val_pairs: int) -> str:
    if train_pairs < 2 or val_pairs < 1:
        return "INSUFFICIENT"
    floor = "DATA_LIMITED" if dataset_verdict in {"STILL_DATA_LIMITED", "HELDOUT_READY_TRAIN_WEAK"} or train_pairs < 60 or val_pairs < 10 else ""
    valid = [
        row
        for row in rows
        if row.get("variant") == "v4_only_primary_safe"
        and row.get("flip_diagnostics", {}).get("passes")
        and math.isfinite(float(row["metrics"].get("validation_pairwise_accuracy", float("nan"))))
    ]
    if not valid:
        return "NO_LEARNING" if rows else "INSUFFICIENT"
    best_val = max(float(row["metrics"]["validation_pairwise_accuracy"]) for row in valid)
    train_best = max(float(row["metrics"]["train_pairwise_accuracy"]) for row in valid if float(row["metrics"]["validation_pairwise_accuracy"]) == best_val)
    if floor and best_val < 0.60:
        return floor
    if best_val >= 0.60:
        return "READY" if not floor else "WEAK"
    if best_val > 0.50:
        return "WEAK"
    if train_best >= 0.70 and best_val <= 0.50:
        return "OVERFIT"
    return "NO_LEARNING"


def main() -> int:
    args = parse_args()
    ensure_v4_root()
    started = time.time()
    if not DATASET_V4_PT.exists():
        payload = {"BG_HIDDEN_ORIGIN_TAP_TRAINING_V4_VERDICT": "INSUFFICIENT", "verdict": "INSUFFICIENT", "blocker": "missing dataset"}
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# Hidden-Origin Tap Training V4", "", "BG_HIDDEN_ORIGIN_TAP_TRAINING_V4_VERDICT = INSUFFICIENT"])
        print("BG_HIDDEN_ORIGIN_TAP_TRAINING_V4_VERDICT = INSUFFICIENT", flush=True)
        return 1
    dataset = torch.load(DATASET_V4_PT, map_location="cpu", weights_only=False)
    pairs_by_variant = dict(dataset.get("pairs_by_variant") or {"primary_safe_deterministic": dataset.get("pairs") or []})
    variant_map = {
        "v4_only_primary_safe": "primary_safe_deterministic",
        "alpha_0_02_diagnostic": "alpha_0_02_diagnostic",
        "sampled_expected_diagnostic": "sampled_expected_diagnostic",
        "high_yield_recipe_subset": "high_yield_recipe_subset",
    }
    primary_pairs = list(pairs_by_variant.get("primary_safe_deterministic") or [])
    train_pairs_all = [pair for pair in primary_pairs if pair.get("split") == "train"]
    val_pairs_all = [pair for pair in primary_pairs if pair.get("split") == "val"]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    seeds = [42, 43, 44, 45, 46] if dataset.get("verdict") == "READY" else [42, 43, 44]
    lrs = [1e-4, 3e-4, 1e-3]
    heads: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    useful_configs = [
        "30_L4",
        "concat_24_30_36",
        "concat_24_36_47",
        "concat_24_36",
        "concat_36_47",
        "24_L4",
        "36_L4",
        "47_L4",
        "36_mean",
        "24_mean",
        "47_mean",
    ]
    for variant, source_variant in variant_map.items():
        variant_pairs = list(pairs_by_variant.get(source_variant) or [])
        train_variant_all = [pair for pair in variant_pairs if pair.get("split") == "train"]
        val_variant_all = [pair for pair in variant_pairs if pair.get("split") == "val"]
        if len(train_variant_all) < 2 or len(val_variant_all) < 1:
            training_rows.append({"variant": variant, "status": "skipped", "reason": "insufficient train/val pairs", "train_pairs": len(train_variant_all), "val_pairs": len(val_variant_all)})
            continue
        for config in [cfg for cfg in useful_configs if cfg in CONFIGS]:
            train_pairs = pairs_for_config(train_variant_all, config)
            val_pairs = pairs_for_config(val_variant_all, config)
            if len(train_pairs) < 2 or len(val_pairs) < 1:
                training_rows.append({"variant": variant, "config": config, "status": "skipped", "reason": "insufficient feature pairs", "train_pairs": len(train_pairs), "val_pairs": len(val_pairs)})
                continue
            for architecture in ARCHITECTURES:
                for seed in seeds:
                    for lr in lrs:
                        print(f"training v4 {variant} {architecture} {config} seed={seed} lr={lr}", flush=True)
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
                        metrics["variant"] = variant
                        diag_pairs = val_pairs if val_pairs else train_pairs
                        flip = flip_diagnostics(head.to(device), diag_pairs, config, device)
                        state_dict = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                        row = {
                            "head_group": "hidden_origin_branch_taps_v4",
                            "variant": variant,
                            "architecture": architecture,
                            "config": config,
                            "dim": config_dim(config),
                            "state_dict": state_dict,
                            "direction": direction_from_state_dict(state_dict),
                            "metrics": metrics,
                            "flip_diagnostics": flip,
                        }
                        heads.append(row)
                        training_rows.append(compact_head(row) | {"variant": variant})
    verdict = verdict_for(str(dataset.get("verdict")), heads, len(train_pairs_all), len(val_pairs_all))
    primary_valid = [row for row in heads if row.get("variant") == "v4_only_primary_safe" and row.get("flip_diagnostics", {}).get("passes")]
    best = None
    if primary_valid:
        best = max(primary_valid, key=lambda row: (float(row["metrics"].get("validation_pairwise_accuracy", -1.0)), float(row["metrics"].get("train_pairwise_accuracy", -1.0))))
    by_variant_config: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in heads:
        by_variant_config[(row["variant"], row["config"])].append(row)
    summary_rows = []
    for (variant, config), items in sorted(by_variant_config.items()):
        best_item = max(items, key=lambda row: float(row["metrics"].get("validation_pairwise_accuracy", -1.0)))
        summary_rows.append(
            {
                "variant": variant,
                "config": config,
                "heads": len(items),
                "best_architecture": best_item["architecture"],
                "best_val_pairwise": best_item["metrics"]["validation_pairwise_accuracy"],
                "best_train_pairwise": best_item["metrics"]["train_pairwise_accuracy"],
                "flip_pass": best_item["flip_diagnostics"]["passes"],
            }
        )
    payload = {
        "BG_HIDDEN_ORIGIN_TAP_TRAINING_V4_VERDICT": verdict,
        "verdict": verdict,
        "dataset": rel(DATASET_V4_PT),
        "dataset_verdict": dataset.get("verdict"),
        "train_pairs": len(train_pairs_all),
        "val_pairs": len(val_pairs_all),
        "device": str(device),
        "architectures": list(ARCHITECTURES),
        "seeds": seeds,
        "lrs": lrs,
        "heads": heads,
        "best_head": (compact_head(best) | {"variant": best.get("variant")}) if best else None,
        "config_summary": summary_rows,
        "training_rows": training_rows,
        "anti_degeneracy": {
            "random_swap": True,
            "target_sign_flip": True,
            "flip_diagnostics_required": True,
            "tap_score_not_used_as_training_label": True,
        },
        "optional_mlp_diagnostic": "not_run",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(payload, HEADS_V4_PT)
    compact_payload = {k: v for k, v in payload.items() if k != "heads"}
    compact_payload["heads"] = [compact_head(row) | {"variant": row.get("variant")} for row in heads]
    write_json(OUT_JSON, compact_payload)
    lines = [
        "# Hidden-Origin Tap Training V4",
        "",
        f"BG_HIDDEN_ORIGIN_TAP_TRAINING_V4_VERDICT = {verdict}",
        "",
        f"- train_pairs: `{len(train_pairs_all)}`",
        f"- val_pairs: `{len(val_pairs_all)}`",
        f"- heads_trained: `{len(heads)}`",
        f"- best_head: `{payload['best_head']}`",
        "",
        "## Config Summary",
        "",
    ]
    lines.extend(md_table(summary_rows, ["variant", "config", "heads", "best_architecture", "best_val_pairwise", "best_train_pairwise", "flip_pass"]))
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_TAP_TRAINING_V4_VERDICT = {verdict}", flush=True)
    return 0 if verdict not in {"INSUFFICIENT"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
