#!/usr/bin/env python3
"""Train a tiny causal BG intervention adapter with teacher-forced logit loss."""
from __future__ import annotations

import math
import os
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from bg_causal_adapter_common import (
    OUT_ROOT,
    PRIMARY_MODE,
    TRAIN_ALPHAS,
    adapter_hook,
    all_frozen,
    avg,
    encode_example,
    finite,
    forward_logits,
    load_adapter_dataset,
    load_model_tokenizer,
    non_answer_kl,
    option_metrics_from_logits,
    rel,
    rows_for_split,
    set_seed,
    write_json,
    write_md,
)
from src.evaluator.bg_causal_adapter import LowRankDeltaAdapter, assert_adapter_budget, parameter_count


OUT_CKPT_DIR = OUT_ROOT / "adapter_checkpoints"
OUT_BEST = OUT_CKPT_DIR / "best_adapter.pt"
OUT_LOG = OUT_ROOT / "training_log.json"
OUT_MD = OUT_ROOT / "training_report.md"

MAX_EPOCHS = min(20, int(os.environ.get("BG_CAUSAL_ADAPTER_EPOCHS", "20")))
PATIENCE = int(os.environ.get("BG_CAUSAL_ADAPTER_PATIENCE", "3"))
TRAIN_CAP = int(os.environ.get("BG_CAUSAL_ADAPTER_TRAIN_CAP", "0"))
VAL_CAP = int(os.environ.get("BG_CAUSAL_ADAPTER_VAL_CAP", "0"))
LR = float(os.environ.get("BG_CAUSAL_ADAPTER_LR", "0.001"))
WEIGHT_DECAY = float(os.environ.get("BG_CAUSAL_ADAPTER_WEIGHT_DECAY", "0.0001"))
LAMBDA_KL = float(os.environ.get("BG_CAUSAL_ADAPTER_LAMBDA_KL", "0.05"))
LAMBDA_DELTA = float(os.environ.get("BG_CAUSAL_ADAPTER_LAMBDA_DELTA", "0.1"))
GRAD_CLIP = float(os.environ.get("BG_CAUSAL_ADAPTER_GRAD_CLIP", "1.0"))


def choose_rows(rows: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    """Keep branch/domain coverage deterministic while staying bounded."""
    if cap <= 0 or len(rows) <= cap:
        return list(rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("domain"))].append(row)
    selected: list[dict[str, Any]] = []
    domains = sorted(grouped)
    while len(selected) < cap:
        progressed = False
        for domain in domains:
            if grouped[domain]:
                selected.append(grouped[domain].pop(0))
                progressed = True
                if len(selected) >= cap:
                    break
        if not progressed:
            break
    return selected


def run_example(
    model: Any,
    tokenizer: Any,
    adapter: torch.nn.Module,
    example: dict[str, Any],
    device: torch.device,
    *,
    alpha: float,
    train: bool,
) -> dict[str, Any]:
    encoded = encode_example(tokenizer, example, device)
    with torch.no_grad():
        baseline_logits = forward_logits(model, encoded["enc"], logits_to_keep=33)
        baseline_metrics = option_metrics_from_logits(baseline_logits, encoded)

    with adapter_hook(
        model,
        adapter,
        alpha=float(alpha),
        intervention_mode=PRIMARY_MODE,
        position=int(encoded["intervention_token_index"]),
    ) as hook:
        if train:
            intervened_logits = forward_logits(model, encoded["enc"], logits_to_keep=33)
        else:
            with torch.no_grad():
                intervened_logits = forward_logits(model, encoded["enc"], logits_to_keep=33)
        intervened_metrics = option_metrics_from_logits(intervened_logits, encoded)
        kl = non_answer_kl(baseline_logits, intervened_logits, window=16)
        if hook.delta_fraction_tensors:
            delta_loss = torch.stack(hook.delta_fraction_tensors).pow(2).mean()
        else:
            delta_loss = torch.zeros((), device=device, dtype=torch.float32)
        loss = intervened_metrics["ce"] + LAMBDA_KL * kl + LAMBDA_DELTA * delta_loss
        diagnostics = hook.diagnostics()

    return {
        "loss": loss,
        "ce": intervened_metrics["ce"],
        "margin": intervened_metrics["margin"],
        "accuracy": float(intervened_metrics["accuracy"]),
        "predicted_option": intervened_metrics["predicted_option"],
        "baseline_margin": baseline_metrics["margin"].detach(),
        "baseline_ce": baseline_metrics["ce"].detach(),
        "baseline_accuracy": float(baseline_metrics["accuracy"]),
        "kl_non_answer": kl,
        "delta_loss": delta_loss,
        "diagnostics": diagnostics,
        "intervention_position_kind": encoded["intervention_position_kind"],
        "intervention_token_index": int(encoded["intervention_token_index"]),
        "answer_logit_token_index": int(encoded["answer_logit_token_index"]),
        "INTERVENTION_POSITION_WARNING": bool(encoded["INTERVENTION_POSITION_WARNING"]),
    }


def eval_rows(
    model: Any,
    tokenizer: Any,
    adapter: torch.nn.Module,
    rows: list[dict[str, Any]],
    device: torch.device,
    *,
    alpha: float,
) -> dict[str, Any]:
    adapter.eval()
    metrics: list[dict[str, float]] = []
    warnings = 0
    position_kinds: set[str] = set()
    for row in rows:
        result = run_example(model, tokenizer, adapter, row, device, alpha=alpha, train=False)
        metrics.append(
            {
                "margin": float(result["margin"].detach().cpu().item()),
                "ce": float(result["ce"].detach().cpu().item()),
                "accuracy": float(result["accuracy"]),
                "baseline_margin": float(result["baseline_margin"].detach().cpu().item()),
                "baseline_ce": float(result["baseline_ce"].detach().cpu().item()),
                "baseline_accuracy": float(result["baseline_accuracy"]),
                "kl_non_answer": float(result["kl_non_answer"].detach().cpu().item()),
                "delta_rms": float(result["diagnostics"].get("activation_rms_change", 0.0)),
            }
        )
        warnings += int(bool(result["INTERVENTION_POSITION_WARNING"]))
        position_kinds.add(str(result["intervention_position_kind"]))
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "alpha": float(alpha),
        "examples": len(metrics),
        "margin": avg(m["margin"] for m in metrics),
        "ce": avg(m["ce"] for m in metrics),
        "accuracy": avg(m["accuracy"] for m in metrics),
        "baseline_margin": avg(m["baseline_margin"] for m in metrics),
        "baseline_ce": avg(m["baseline_ce"] for m in metrics),
        "baseline_accuracy": avg(m["baseline_accuracy"] for m in metrics),
        "margin_lift": (avg(m["margin"] for m in metrics) or 0.0) - (avg(m["baseline_margin"] for m in metrics) or 0.0),
        "kl_non_answer": avg(m["kl_non_answer"] for m in metrics),
        "delta_rms": avg(m["delta_rms"] for m in metrics),
        "INTERVENTION_POSITION_WARNING": warnings > 0,
        "intervention_position_kinds": sorted(position_kinds),
    }


def main() -> int:
    started = time.time()
    set_seed()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_adapter_dataset()
    if dataset.get("BG_CAUSAL_ADAPTER_DATASET_VERDICT") == "BLOCKED" or not dataset.get("examples"):
        payload = {
            "BG_CAUSAL_ADAPTER_TRAINING_VERDICT": "BLOCKED",
            "blocker": "adapter_dataset.json missing or blocked",
            "elapsed_seconds": round(time.time() - started, 3),
        }
        write_json(OUT_LOG, payload)
        write_md(OUT_MD, ["# BG Causal Intervention Adapter Training", "", "BG_CAUSAL_ADAPTER_TRAINING_VERDICT = BLOCKED", "", "- blocker: adapter dataset missing or blocked"])
        print("BG_CAUSAL_ADAPTER_TRAINING_VERDICT = BLOCKED")
        return 1

    train_rows = choose_rows(rows_for_split(dataset, "train"), TRAIN_CAP)
    val_rows = choose_rows(rows_for_split(dataset, "val"), VAL_CAP)
    write_json(
        OUT_LOG,
        {
            "BG_CAUSAL_ADAPTER_TRAINING_VERDICT": "RUNNING",
            "adapter_class": "LowRankDeltaAdapter",
            "adapter_rank": 32,
            "primary_loss": "L_answer_ce",
            "KL_ANSWER_POSITION_MASKED": True,
            "intervention_position_kind": "prefix_last_token",
            "training_rows_used": len(train_rows),
            "validation_rows_used": len(val_rows),
            "train_example_cap": TRAIN_CAP,
            "val_example_cap": VAL_CAP,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "status": "initializing_model",
            "elapsed_seconds": round(time.time() - started, 3),
        },
    )
    if len(train_rows) < 4 or len(val_rows) < 2:
        payload = {
            "BG_CAUSAL_ADAPTER_TRAINING_VERDICT": "BLOCKED",
            "blocker": f"insufficient rows: train={len(train_rows)} val={len(val_rows)}",
            "elapsed_seconds": round(time.time() - started, 3),
        }
        write_json(OUT_LOG, payload)
        write_md(OUT_MD, ["# BG Causal Intervention Adapter Training", "", "BG_CAUSAL_ADAPTER_TRAINING_VERDICT = BLOCKED", "", f"- blocker: `{payload['blocker']}`"])
        print("BG_CAUSAL_ADAPTER_TRAINING_VERDICT = BLOCKED")
        return 1

    try:
        model, tokenizer, device = load_model_tokenizer()
    except Exception as exc:
        payload = {
            "BG_CAUSAL_ADAPTER_TRAINING_VERDICT": "BLOCKED",
            "blocker": "model_initialization_failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "traceback": traceback.format_exc()[-4000:],
            "adapter_class": "LowRankDeltaAdapter",
            "adapter_rank": 32,
            "primary_loss": "L_answer_ce",
            "KL_ANSWER_POSITION_MASKED": True,
            "intervention_position_kind": "prefix_last_token",
            "training_rows_used": len(train_rows),
            "validation_rows_used": len(val_rows),
            "train_example_cap": TRAIN_CAP,
            "val_example_cap": VAL_CAP,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        write_json(OUT_LOG, payload)
        write_md(
            OUT_MD,
            [
                "# BG Causal Intervention Adapter Training",
                "",
                "BG_CAUSAL_ADAPTER_TRAINING_VERDICT = BLOCKED",
                "",
                "- blocker: `model_initialization_failed`",
                f"- error: `{type(exc).__name__}: {str(exc)[:500]}`",
            ],
        )
        print("BG_CAUSAL_ADAPTER_TRAINING_VERDICT = BLOCKED")
        print(f"MODEL_INITIALIZATION_ERROR = {type(exc).__name__}: {str(exc)[:500]}")
        return 1
    adapter = LowRankDeltaAdapter(rank=32).to(device)
    assert_adapter_budget(adapter)
    gpu_name = torch.cuda.get_device_name(device.index or 0)
    if next(adapter.parameters()).device.type != "cuda":
        raise RuntimeError(f"adapter is not on CUDA after .to({device}); got {next(adapter.parameters()).device}")
    write_json(
        OUT_LOG,
        {
            "BG_CAUSAL_ADAPTER_TRAINING_VERDICT": "RUNNING",
            "adapter_class": "LowRankDeltaAdapter",
            "adapter_rank": 32,
            "primary_loss": "L_answer_ce",
            "KL_ANSWER_POSITION_MASKED": True,
            "intervention_position_kind": "prefix_last_token",
            "training_rows_used": len(train_rows),
            "validation_rows_used": len(val_rows),
            "train_example_cap": TRAIN_CAP,
            "val_example_cap": VAL_CAP,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "status": "cuda_model_loaded",
            "device": str(device),
            "gpu_name": gpu_name,
            "model_first_parameter_device": str(next(model.parameters()).device),
            "adapter_first_parameter_device": str(next(adapter.parameters()).device),
            "elapsed_seconds": round(time.time() - started, 3),
        },
    )
    if not all_frozen(model):
        raise RuntimeError("Ouro model parameters are not frozen")
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    baseline_val = eval_rows(model, tokenizer, adapter, val_rows, device, alpha=0.0)
    best_margin = -float("inf")
    best_epoch = -1
    stale_epochs = 0
    epoch_logs: list[dict[str, Any]] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        adapter.train()
        epoch_metrics: list[dict[str, float]] = []
        for idx, row in enumerate(train_rows):
            alpha = TRAIN_ALPHAS[(epoch + idx) % len(TRAIN_ALPHAS)]
            optimizer.zero_grad(set_to_none=True)
            result = run_example(model, tokenizer, adapter, row, device, alpha=alpha, train=True)
            loss = result["loss"]
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_metrics.append(
                {
                    "loss": float(loss.detach().cpu().item()),
                    "ce": float(result["ce"].detach().cpu().item()),
                    "margin": float(result["margin"].detach().cpu().item()),
                    "baseline_margin": float(result["baseline_margin"].detach().cpu().item()),
                    "accuracy": float(result["accuracy"]),
                    "kl_non_answer": float(result["kl_non_answer"].detach().cpu().item()),
                    "delta_rms": float(result["diagnostics"].get("activation_rms_change", 0.0)),
                }
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

        val_by_alpha = [eval_rows(model, tokenizer, adapter, val_rows, device, alpha=alpha) for alpha in TRAIN_ALPHAS]
        best_val_for_epoch = max(val_by_alpha, key=lambda row: finite(row.get("margin"), -1e9))
        epoch_log = {
            "epoch": epoch,
            "train_examples": len(epoch_metrics),
            "train_loss": avg(m["loss"] for m in epoch_metrics),
            "train_margin": avg(m["margin"] for m in epoch_metrics),
            "train_baseline_margin": avg(m["baseline_margin"] for m in epoch_metrics),
            "train_margin_lift": (avg(m["margin"] for m in epoch_metrics) or 0.0)
            - (avg(m["baseline_margin"] for m in epoch_metrics) or 0.0),
            "train_accuracy": avg(m["accuracy"] for m in epoch_metrics),
            "train_kl_non_answer": avg(m["kl_non_answer"] for m in epoch_metrics),
            "train_delta_rms": avg(m["delta_rms"] for m in epoch_metrics),
            "val_by_alpha": val_by_alpha,
            "best_val": best_val_for_epoch,
        }
        epoch_logs.append(epoch_log)
        checkpoint = {
            "adapter_state_dict": adapter.state_dict(),
            "adapter_class": "LowRankDeltaAdapter",
            "adapter_rank": 32,
            "adapter_params": parameter_count(adapter),
            "epoch": epoch,
            "epoch_log": epoch_log,
            "intervention_mode": PRIMARY_MODE,
            "target_layer": 36,
            "KL_ANSWER_POSITION_MASKED": True,
            "intervention_position_kind": "prefix_last_token",
            "train_task_ids": dataset.get("split_task_ids", {}).get("train", []),
            "val_task_ids": dataset.get("split_task_ids", {}).get("val", []),
            "heldout_task_ids": dataset.get("split_task_ids", {}).get("heldout", []),
        }
        torch.save(checkpoint, OUT_CKPT_DIR / f"adapter_epoch_{epoch:02d}.pt")
        current_margin = finite(best_val_for_epoch.get("margin"), -1e9)
        if current_margin > best_margin + 1e-6:
            best_margin = current_margin
            best_epoch = epoch
            stale_epochs = 0
            torch.save(checkpoint, OUT_BEST)
        else:
            stale_epochs += 1
        write_json(
            OUT_LOG,
            {
                "BG_CAUSAL_ADAPTER_TRAINING_VERDICT": "PARTIAL",
                "adapter_class": "LowRankDeltaAdapter",
                "adapter_rank": 32,
                "primary_loss": "L_answer_ce",
                "KL_ANSWER_POSITION_MASKED": True,
                "intervention_position_kind": "prefix_last_token",
                "training_rows_used": len(train_rows),
                "validation_rows_used": len(val_rows),
                "train_example_cap": TRAIN_CAP,
                "val_example_cap": VAL_CAP,
                "max_epochs": MAX_EPOCHS,
                "early_stopping_patience": PATIENCE,
                "device": str(device),
                "gpu_name": gpu_name,
                "model_first_parameter_device": str(next(model.parameters()).device),
                "adapter_first_parameter_device": str(next(adapter.parameters()).device),
                "epoch_logs": epoch_logs,
                "best_epoch": best_epoch,
                "best_val_margin": best_margin,
                "elapsed_seconds": round(time.time() - started, 3),
            },
        )
        if stale_epochs >= PATIENCE:
            break

    best_log = next((row for row in epoch_logs if row["epoch"] == best_epoch), epoch_logs[-1] if epoch_logs else {})
    best_val = best_log.get("best_val", {})
    best_lift = finite(best_val.get("margin_lift"), 0.0)
    if best_epoch > 0 and best_lift > 0.05:
        verdict = "READY"
    elif best_epoch > 0 and best_lift > 0.0:
        verdict = "PARTIAL"
    else:
        verdict = "NO_LEARNING"

    payload = {
        "BG_CAUSAL_ADAPTER_TRAINING_VERDICT": verdict,
        "adapter_class": "LowRankDeltaAdapter",
        "adapter_rank": 32,
        "adapter_params": parameter_count(adapter),
        "primary_loss": "L_answer_ce",
        "loss_weights": {"lambda_kl": LAMBDA_KL, "lambda_delta": LAMBDA_DELTA, "lambda_bg": 0.0},
        "KL_ANSWER_POSITION_MASKED": True,
        "intervention_position_kind": "prefix_last_token",
        "INTERVENTION_POSITION_WARNING": any(
            bool(v.get("INTERVENTION_POSITION_WARNING"))
            for log in epoch_logs
            for v in log.get("val_by_alpha", [])
        ),
        "training_rows_used": len(train_rows),
        "validation_rows_used": len(val_rows),
        "train_example_cap": TRAIN_CAP,
        "val_example_cap": VAL_CAP,
        "max_epochs": MAX_EPOCHS,
        "epochs_completed": len(epoch_logs),
        "early_stopping_patience": PATIENCE,
        "device": str(device),
        "gpu_name": gpu_name,
        "model_first_parameter_device": str(next(model.parameters()).device),
        "adapter_first_parameter_device": str(next(adapter.parameters()).device),
        "best_epoch": best_epoch,
        "baseline_val": baseline_val,
        "best_val": best_val,
        "best_val_margin_lift": best_lift,
        "best_adapter_path": rel(OUT_BEST),
        "epoch_logs": epoch_logs,
        "model_parameters_frozen": all_frozen(model),
        "trainable_parameters": parameter_count(adapter),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_LOG, payload)
    lines = [
        "# BG Causal Intervention Adapter Training",
        "",
        f"BG_CAUSAL_ADAPTER_TRAINING_VERDICT = {verdict}",
        "",
        f"- adapter: `LowRankDeltaAdapter(rank=32)`",
        f"- adapter params: `{parameter_count(adapter)}`",
        f"- training rows used: `{len(train_rows)}`",
        f"- validation rows used: `{len(val_rows)}`",
        f"- epochs completed: `{len(epoch_logs)}`",
        f"- best epoch: `{best_epoch}`",
        f"- best validation margin lift: `{best_lift:.6f}`",
        f"- primary loss: `L_answer_ce`",
        f"- KL answer position masked: `true`",
        f"- intervention position kind: `prefix_last_token`",
        f"- best adapter: `{rel(OUT_BEST)}`",
        "",
        "## Epochs",
        "",
        "| epoch | train loss | train margin lift | best val alpha | val margin | val margin lift | val accuracy | KL | delta RMS |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for log in epoch_logs:
        val = log.get("best_val", {})
        lines.append(
            f"| {log['epoch']} | {finite(log.get('train_loss')):.6f} | {finite(log.get('train_margin_lift')):.6f} | "
            f"{finite(val.get('alpha')):.3f} | {finite(val.get('margin')):.6f} | {finite(val.get('margin_lift')):.6f} | "
            f"{finite(val.get('accuracy')):.3f} | {finite(val.get('kl_non_answer')):.6f} | {finite(val.get('delta_rms')):.6f} |"
        )
    write_md(OUT_MD, lines)
    print(f"BG_CAUSAL_ADAPTER_TRAINING_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_LOG)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
