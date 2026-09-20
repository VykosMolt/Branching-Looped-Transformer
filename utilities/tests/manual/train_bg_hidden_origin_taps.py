"""Train tiny pairwise taps for same-prefix hidden-origin branch selection."""
from __future__ import annotations

import argparse
import math
import random
import time
from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from bg_hidden_origin_tap_common import (
    CONFIGS,
    HEAD_CLASSES,
    OUT_ROOT,
    config_dim,
    direction_from_state_dict,
    ensure_out_root,
    md_table,
    pairwise_accuracy_from_pairs,
    rate,
    rel,
    score_pair,
    write_json,
    write_md,
)


DATASET_PT = OUT_ROOT / "hidden_origin_tap_dataset.pt"
OUT_PT = OUT_ROOT / "hidden_origin_tap_heads.pt"
OUT_JSON = OUT_ROOT / "training_log.json"
OUT_MD = OUT_ROOT / "training_report.md"
ARCHITECTURES = ("AntisymLinear", "AntisymLinearNoNorm")


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


def pairs_for_config(pairs: Sequence[dict[str, Any]], config: str) -> list[dict[str, Any]]:
    return [pair for pair in pairs if config in pair.get("features", {})]


def tensors_from_pairs(pairs: Sequence[dict[str, Any]], config: str) -> tuple[torch.Tensor, torch.Tensor]:
    left = [pair["features"][config]["preferred"].to(torch.float32) for pair in pairs]
    right = [pair["features"][config]["rejected"].to(torch.float32) for pair in pairs]
    if not left:
        return torch.empty((0, config_dim(config))), torch.empty((0, config_dim(config)))
    return torch.stack(left, dim=0), torch.stack(right, dim=0)


@torch.no_grad()
def pair_acc(head: torch.nn.Module, left: torch.Tensor, right: torch.Tensor, device: torch.device) -> float:
    if left.numel() == 0:
        return float("nan")
    scores = head(left.to(device), right.to(device))
    return float((scores > 0).to(torch.float32).mean().detach().cpu().item())


@torch.no_grad()
def pair_loss(head: torch.nn.Module, left: torch.Tensor, right: torch.Tensor, device: torch.device) -> float:
    if left.numel() == 0:
        return float("inf")
    scores = head(left.to(device), right.to(device))
    return float(F.softplus(-scores).mean().detach().cpu().item())


@torch.no_grad()
def flip_diagnostics(head: torch.nn.Module, pairs: Sequence[dict[str, Any]], config: str, device: torch.device) -> dict[str, Any]:
    scores = []
    flips = []
    for pair in pairs:
        feats = pair.get("features", {}).get(config)
        if not feats:
            continue
        s = score_pair(head, feats["preferred"], feats["rejected"], device)
        f = score_pair(head, feats["rejected"], feats["preferred"], device)
        scores.append(float(s))
        flips.append(float(f))
    if not scores:
        return {
            "n": 0,
            "antisymmetry_correlation": float("nan"),
            "mean_score_sum": float("nan"),
            "mean_abs_score_sum": float("nan"),
            "strict_sign_flip_rate": float("nan"),
            "score_std": float("nan"),
            "passes": False,
        }
    st = torch.tensor(scores, dtype=torch.float32)
    ft = torch.tensor(flips, dtype=torch.float32)
    target = -ft
    if st.numel() > 1 and float(st.std(unbiased=False).item()) > 1e-8 and float(target.std(unbiased=False).item()) > 1e-8:
        corr = float(torch.corrcoef(torch.stack([st, target]))[0, 1].item())
    else:
        corr = 1.0 if float((st - target).abs().max().item()) < 1e-6 else 0.0
    signs = ((st > 0) & (ft < 0)) | ((st < 0) & (ft > 0)) | ((st == 0) & (ft == 0))
    score_std = float(st.std(unbiased=False).item()) if st.numel() > 1 else 0.0
    mean_abs_sum = float((st + ft).abs().mean().item())
    return {
        "n": len(scores),
        "antisymmetry_correlation": corr,
        "mean_score_sum": float((st + ft).mean().item()),
        "mean_abs_score_sum": mean_abs_sum,
        "strict_sign_flip_rate": float(signs.to(torch.float32).mean().item()),
        "score_mean": float(st.mean().item()),
        "score_std": score_std,
        "passes": bool(corr >= 0.999 and mean_abs_sum <= 1e-5 and score_std > 1e-7),
    }


def train_one(
    *,
    architecture: str,
    config: str,
    seed: int,
    lr: float,
    train_pairs: list[dict[str, Any]],
    val_pairs: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    head = HEAD_CLASSES[architecture](config_dim(config)).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(lr), weight_decay=float(args.weight_decay))
    left_train, right_train = tensors_from_pairs(train_pairs, config)
    left_val, right_val = tensors_from_pairs(val_pairs, config)
    left_train = left_train.to(device)
    right_train = right_train.to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    batch_size = max(1, min(int(args.batch_size), int(left_train.shape[0])))
    best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    best_epoch = -1
    best_val_acc = -1.0
    best_val_loss = float("inf")
    stale = 0
    history = []
    for epoch in range(int(args.epochs)):
        head.train()
        perm = torch.randperm(left_train.shape[0], generator=generator, device=device)
        total = 0.0
        for start in range(0, left_train.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            left = left_train[idx]
            right = right_train[idx]
            target = torch.ones(left.shape[0], device=device, dtype=torch.float32)
            swap = torch.rand(left.shape[0], generator=generator, device=device) < 0.5
            if bool(swap.any().item()):
                left_swapped = left.clone()
                right_swapped = right.clone()
                left_swapped[swap] = right[swap]
                right_swapped[swap] = left[swap]
                left = left_swapped
                right = right_swapped
                target[swap] = -1.0
            scores = head(left, right)
            loss = F.softplus(-target * scores).mean()
            if float(args.score_l2) > 0.0:
                loss = loss + float(args.score_l2) * scores.pow(2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), float(args.gradient_clip))
            optimizer.step()
            total += float(loss.detach().cpu().item()) * int(idx.numel())
        train_loss = total / max(int(left_train.shape[0]), 1)
        head.eval()
        train_acc = pair_acc(head, left_train.detach().cpu(), right_train.detach().cpu(), device)
        val_acc = pair_acc(head, left_val, right_val, device)
        val_loss = pair_loss(head, left_val, right_val, device)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc, "val_loss": val_loss})
        improved = val_acc > best_val_acc + 1e-8 or (abs(val_acc - best_val_acc) <= 1e-8 and val_loss < best_val_loss)
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            stale = 0
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
        "seed": seed,
        "lr": lr,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "train_pairwise_accuracy": pair_acc(head, left_train.detach().cpu(), right_train.detach().cpu(), device),
        "validation_pairwise_accuracy": pair_acc(head, left_val, right_val, device),
        "validation_loss": pair_loss(head, left_val, right_val, device),
        "history": history,
        "swap_protocol": "50_percent_random_left_right_with_target_sign_flip",
        "objective": "directional_pairwise_logsigmoid",
        "score_l2": float(args.score_l2),
        "gradient_clip": float(args.gradient_clip),
    }
    return head.to("cpu"), metrics


def compact_head(row: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in row.items() if k not in {"state_dict", "direction"}}
    return out


def verdict_for(dataset_verdict: str, rows: Sequence[dict[str, Any]]) -> str:
    if dataset_verdict in {"BLOCKED", "TOO_SMALL"}:
        return "INSUFFICIENT"
    valid = [row for row in rows if row.get("flip_diagnostics", {}).get("passes") and math.isfinite(float(row["metrics"].get("validation_pairwise_accuracy", float("nan"))))]
    if not valid:
        return "NO_LEARNING" if rows else "INSUFFICIENT"
    best_val = max(float(row["metrics"]["validation_pairwise_accuracy"]) for row in valid)
    train_best_for_val = max(float(row["metrics"]["train_pairwise_accuracy"]) for row in valid if float(row["metrics"]["validation_pairwise_accuracy"]) == best_val)
    if best_val >= 0.60:
        return "READY"
    if best_val > 0.50:
        return "WEAK"
    if train_best_for_val >= 0.70 and best_val <= 0.50:
        return "OVERFIT"
    return "NO_LEARNING"


def main() -> int:
    args = parse_args()
    ensure_out_root()
    started = time.time()
    if not DATASET_PT.exists():
        payload = {"BG_HIDDEN_ORIGIN_TAP_TRAINING_VERDICT": "INSUFFICIENT", "blocker": "missing dataset pt"}
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# Hidden-Origin Tap Training", "", "BG_HIDDEN_ORIGIN_TAP_TRAINING_VERDICT = INSUFFICIENT"])
        print("BG_HIDDEN_ORIGIN_TAP_TRAINING_VERDICT = INSUFFICIENT", flush=True)
        return 1
    dataset = torch.load(DATASET_PT, map_location="cpu", weights_only=False)
    pairs = list(dataset.get("pairs") or [])
    train_pairs_all = [pair for pair in pairs if pair.get("split") == "train"]
    val_pairs_all = [pair for pair in pairs if pair.get("split") == "val"]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    heads: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    seeds = [42, 43, 44]
    lrs = [1e-4, 3e-4, 1e-3]
    for config in CONFIGS:
        train_pairs = pairs_for_config(train_pairs_all, config)
        val_pairs = pairs_for_config(val_pairs_all, config)
        if len(train_pairs) < 2 or len(val_pairs) < 1:
            training_rows.append(
                {
                    "config": config,
                    "status": "skipped",
                    "reason": "insufficient train/val pairs with features",
                    "train_pairs": len(train_pairs),
                    "val_pairs": len(val_pairs),
                }
            )
            continue
        for architecture in ARCHITECTURES:
            for seed in seeds:
                for lr in lrs:
                    print(f"training {architecture} {config} seed={seed} lr={lr}", flush=True)
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
                    diag_pairs = val_pairs if val_pairs else train_pairs
                    flip = flip_diagnostics(head.to(device), diag_pairs, config, device)
                    state_dict = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                    row = {
                        "head_group": "hidden_origin_branch_taps",
                        "architecture": architecture,
                        "config": config,
                        "dim": config_dim(config),
                        "state_dict": state_dict,
                        "direction": direction_from_state_dict(state_dict),
                        "metrics": metrics,
                        "flip_diagnostics": flip,
                    }
                    heads.append(row)
                    training_rows.append(compact_head(row))

    verdict = verdict_for(str(dataset.get("verdict")), heads)
    best = None
    valid_heads = [row for row in heads if row.get("flip_diagnostics", {}).get("passes")]
    if valid_heads:
        best = max(
            valid_heads,
            key=lambda row: (
                float(row["metrics"].get("validation_pairwise_accuracy", -1.0)),
                float(row["metrics"].get("train_pairwise_accuracy", -1.0)),
                -abs(float(row["flip_diagnostics"].get("mean_abs_score_sum", 999.0))),
            ),
        )
    by_config = defaultdict(list)
    for row in heads:
        by_config[row["config"]].append(row)
    summary_rows = []
    for config, items in sorted(by_config.items()):
        best_item = max(items, key=lambda row: float(row["metrics"].get("validation_pairwise_accuracy", -1.0)))
        summary_rows.append(
            {
                "config": config,
                "heads": len(items),
                "best_architecture": best_item["architecture"],
                "best_val_pairwise": best_item["metrics"]["validation_pairwise_accuracy"],
                "best_train_pairwise": best_item["metrics"]["train_pairwise_accuracy"],
                "flip_pass": best_item["flip_diagnostics"]["passes"],
            }
        )

    payload = {
        "BG_HIDDEN_ORIGIN_TAP_TRAINING_VERDICT": verdict,
        "verdict": verdict,
        "dataset": rel(DATASET_PT),
        "dataset_verdict": dataset.get("verdict"),
        "train_pairs": len(train_pairs_all),
        "val_pairs": len(val_pairs_all),
        "device": str(device),
        "architectures": list(ARCHITECTURES),
        "seeds": seeds,
        "lrs": lrs,
        "heads": heads,
        "best_head": compact_head(best) if best else None,
        "config_summary": summary_rows,
        "training_rows": training_rows,
        "anti_degeneracy": {
            "random_swap": True,
            "target_sign_flip": True,
            "flip_diagnostics_required": True,
            "constant_solution_rejected_by_score_std": True,
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(payload, OUT_PT)
    json_payload = {k: v for k, v in payload.items() if k != "heads"}
    json_payload["heads"] = [compact_head(row) for row in heads]
    write_json(OUT_JSON, json_payload)

    lines = [
        "# Hidden-Origin Tap Training",
        "",
        f"BG_HIDDEN_ORIGIN_TAP_TRAINING_VERDICT = {verdict}",
        "",
        f"- dataset_verdict: `{dataset.get('verdict')}`",
        f"- train_pairs: `{len(train_pairs_all)}`",
        f"- val_pairs: `{len(val_pairs_all)}`",
        f"- trained_heads: `{len(heads)}`",
        f"- best_head: `{json_payload['best_head']}`",
        "",
        "The headline heads are exact antisymmetric `AntisymLinear` and `AntisymLinearNoNorm`; no Ouro weights, tokenizer files, checkpoints, old tap heads, or old registries were modified.",
        "",
        "## Config Summary",
        "",
    ]
    lines.extend(md_table(summary_rows, ["config", "heads", "best_architecture", "best_val_pairwise", "best_train_pairwise", "flip_pass"]))
    lines.extend(["", "## Best Rows", ""])
    best_rows = sorted(
        [row for row in training_rows if row.get("metrics")],
        key=lambda row: float(row["metrics"].get("validation_pairwise_accuracy", -1.0)),
        reverse=True,
    )[:20]
    lines.extend(md_table(
        [
            {
                "config": row["config"],
                "architecture": row["architecture"],
                "seed": row["metrics"]["seed"],
                "lr": row["metrics"]["lr"],
                "train": rate(row["metrics"]["train_pairwise_accuracy"]),
                "val": rate(row["metrics"]["validation_pairwise_accuracy"]),
                "flip_corr": rate(row["flip_diagnostics"]["antisymmetry_correlation"]),
                "score_std": rate(row["flip_diagnostics"]["score_std"]),
            }
            for row in best_rows
        ],
        ["config", "architecture", "seed", "lr", "train", "val", "flip_corr", "score_std"],
    ))
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_TAP_TRAINING_VERDICT = {verdict}", flush=True)
    print(f"Wrote {rel(OUT_PT)}", flush=True)
    print(f"Wrote {rel(OUT_JSON)}", flush=True)
    print(f"Wrote {rel(OUT_MD)}", flush=True)
    return 0 if verdict not in {"INSUFFICIENT"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
