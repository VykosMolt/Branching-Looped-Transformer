#!/usr/bin/env python3
"""Train a BG intervention adapter with sequence-level MCQ reward."""
from __future__ import annotations

import os
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from bg_sequence_adapter_common import (
    OUT_ROOT,
    PRIMARY_MODE,
    SAFE_ALPHAS,
    aggregate_rows,
    avg,
    generate_completion,
    generation_prompt,
    load_json,
    load_model_tokenizer,
    load_sequence_dataset,
    policy_logprob_terms,
    rel,
    reward_for_output,
    rows_for_split,
    run_generation_condition,
    set_seed,
    write_json,
    write_md,
)
from src.evaluator.bg_sequence_adapter import build_sequence_adapter, parameter_count, sequence_adapter_hook


OUT_CKPT_DIR = OUT_ROOT / "sequence_adapter_checkpoints"
OUT_BEST = OUT_CKPT_DIR / "best_sequence_adapter.pt"
OUT_JSON = OUT_ROOT / "sequence_training_log.json"
OUT_MD = OUT_ROOT / "sequence_training_report.md"

MAX_UPDATES = max(25, int(os.environ.get("BG_SEQUENCE_MAX_UPDATES", "300")))
VAL_EVERY = max(5, int(os.environ.get("BG_SEQUENCE_VAL_EVERY", "25")))
PATIENCE_EVALS = max(2, int(os.environ.get("BG_SEQUENCE_PATIENCE_EVALS", "5")))
MAX_NEW_TOKENS = min(96, int(os.environ.get("BG_SEQUENCE_TRAIN_TOKENS", "96")))
LR = float(os.environ.get("BG_SEQUENCE_LR", "0.001"))
LAMBDA_KL = float(os.environ.get("BG_SEQUENCE_LAMBDA_KL", "0.02"))
LAMBDA_DELTA = float(os.environ.get("BG_SEQUENCE_LAMBDA_DELTA", "0.05"))
LAMBDA_ENTROPY = float(os.environ.get("BG_SEQUENCE_LAMBDA_ENTROPY", "0.001"))
TRAIN_ALPHA = min(0.02, float(os.environ.get("BG_SEQUENCE_TRAIN_ALPHA", "0.01")))
INIT_FROM_CAUSAL = os.environ.get("BG_SEQUENCE_INIT_FROM_CAUSAL", "1") == "1"


def load_baseline_rewards() -> dict[str, float]:
    payload = load_json(OUT_ROOT / "baseline_eval.json", {})
    out: dict[str, float] = {}
    for row in payload.get("rows", []):
        if row.get("method") == "no_intervention_baseline" and row.get("decode") == "deterministic":
            out[str(row.get("task_id"))] = float(row.get("reward", 0.25))
    return out


def evaluate_adapter(model: Any, tokenizer: Any, device: torch.device, adapter: torch.nn.Module, tasks: list[dict[str, Any]], alphas: list[float]) -> dict[str, Any]:
    rows = []
    for alpha in alphas:
        for idx, task in enumerate(tasks):
            row = run_generation_condition(
                model,
                tokenizer,
                device,
                task,
                method="trained_sequence_adapter",
                seed=20280518 + idx * 100 + int(alpha * 1000),
                alpha=alpha,
                adapter=adapter,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
            rows.append(row)
    aggregate = aggregate_rows(rows, by=("method", "alpha", "decode"))
    best_key = None
    best_reward = -1e9
    for key, row in aggregate.items():
        reward = float(row.get("reward_mean") or -1e9)
        if reward > best_reward:
            best_reward = reward
            best_key = key
    best = aggregate.get(best_key or "", {})
    return {"rows": rows, "aggregate": aggregate, "best": best}


def main() -> int:
    started = time.time()
    set_seed()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    update_logs: list[dict[str, Any]] = []
    val_logs: list[dict[str, Any]] = []
    try:
        dataset = load_sequence_dataset()
        train_tasks = rows_for_split(dataset, "train")
        val_tasks = rows_for_split(dataset, "val")
        if len(train_tasks) < 12 or len(val_tasks) < 4:
            raise RuntimeError(f"insufficient split sizes: train={len(train_tasks)} val={len(val_tasks)}")
        model, tokenizer, device = load_model_tokenizer()
        adapter = build_sequence_adapter(kind="low_rank", rank=32, device=device)
        init_source = "fresh"
        causal_ckpt = Path("opi/taps/probes/bg_causal_intervention_adapter_2026-05-18/adapter_checkpoints/best_adapter.pt")
        if INIT_FROM_CAUSAL and causal_ckpt.exists():
            payload = torch.load(causal_ckpt, map_location=device, weights_only=False)
            missing, unexpected = adapter.load_state_dict(payload["adapter_state_dict"], strict=False)
            init_source = f"causal_adapter_checkpoint missing={list(missing)} unexpected={list(unexpected)}"
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=LR, weight_decay=0.0)
        baseline_rewards = load_baseline_rewards()
        moving_baseline = {task["task_id"]: baseline_rewards.get(task["task_id"], 0.25) for task in train_tasks}
        baseline_val_reward = avg(
            row.get("reward")
            for row in load_json(OUT_ROOT / "baseline_eval.json", {}).get("rows", [])
            if row.get("split") == "val" and row.get("method") == "no_intervention_baseline" and row.get("decode") == "deterministic"
        )
        best_val_reward = -1e9
        best_update = 0
        best_alpha = TRAIN_ALPHA
        stale_evals = 0
        error_count = 0
        write_json(
            OUT_JSON,
            {
                "BG_SEQUENCE_ADAPTER_TRAINING_VERDICT": "RUNNING",
                "status": "initialized",
                "adapter_kind": "low_rank",
                "adapter_rank": 32,
                "adapter_params": parameter_count(adapter),
                "init_source": init_source,
                "max_updates": MAX_UPDATES,
                "train_tasks": len(train_tasks),
                "val_tasks": len(val_tasks),
                "elapsed_seconds": round(time.time() - started, 3),
            },
        )
        for update in range(1, MAX_UPDATES + 1):
            task = train_tasks[(update - 1) % len(train_tasks)]
            prompt = generation_prompt(task)
            prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536, padding=False)["input_ids"].to(device)
            position = max(0, int(prompt_ids.shape[1]) - 1)
            try:
                with sequence_adapter_hook(model, adapter, alpha=TRAIN_ALPHA, intervention_mode=PRIMARY_MODE, position=position):
                    gen = generate_completion(
                        model,
                        tokenizer,
                        device,
                        prompt,
                        seed=20260518 + update,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True,
                    )
                reward_info = reward_for_output(task, gen["output_text"], gen["generation_error"])
                reward = float(reward_info["reward"])
                terms = policy_logprob_terms(
                    model,
                    adapter,
                    gen["prompt_input_ids"].to(device),
                    gen["new_ids"].to(device),
                    alpha=TRAIN_ALPHA,
                    mode=PRIMARY_MODE,
                    lambda_kl=LAMBDA_KL,
                    lambda_delta=LAMBDA_DELTA,
                )
                baseline = moving_baseline.get(task["task_id"], 0.25)
                advantage = reward - baseline
                loss = -float(advantage) * terms["sum_logprob"] + LAMBDA_KL * terms["kl"] + LAMBDA_DELTA * terms["delta_loss"] - LAMBDA_ENTROPY * terms["entropy"]
                optimizer.zero_grad(set_to_none=True)
                if torch.isfinite(loss):
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                    optimizer.step()
                else:
                    grad_norm = torch.zeros(())
                moving_baseline[task["task_id"]] = 0.9 * baseline + 0.1 * reward
                log = {
                    "update": update,
                    "task_id": task["task_id"],
                    "domain": task["domain"],
                    "reward": reward,
                    "correct": bool(reward_info["correct"]),
                    "parse_success": bool(reward_info["parse_success"]),
                    "advantage": float(advantage),
                    "baseline": float(baseline),
                    "loss": float(loss.detach().cpu().item()) if torch.isfinite(loss) else None,
                    "sum_logprob": float(terms["sum_logprob"].detach().cpu().item()),
                    "kl": float(terms["kl"].detach().cpu().item()),
                    "entropy": float(terms["entropy"].detach().cpu().item()),
                    "delta_loss": float(terms["delta_loss"].detach().cpu().item()),
                    "grad_norm": float(grad_norm.detach().cpu().item()),
                    "output_length": int(gen["new_ids"].numel()),
                    "generation_error": gen["generation_error"],
                    "activation_rms_change": terms["diagnostics"].get("activation_rms_change", 0.0),
                }
            except Exception as exc:
                error_count += 1
                log = {
                    "update": update,
                    "task_id": task["task_id"],
                    "domain": task["domain"],
                    "reward": -1.0,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                    "traceback": traceback.format_exc()[-2000:],
                }
            update_logs.append(log)
            if update % VAL_EVERY == 0 or update == MAX_UPDATES:
                adapter.eval()
                val = evaluate_adapter(model, tokenizer, device, adapter, val_tasks, SAFE_ALPHAS)
                val_best = val["best"]
                val_reward = float(val_best.get("reward_mean") or -1e9)
                val_log = {
                    "update": update,
                    "val": val,
                    "best_val_reward": val_reward,
                    "best_val_alpha": float(val_best.get("alpha") or TRAIN_ALPHA),
                    "train_recent_reward_mean": avg(row.get("reward") for row in update_logs[-VAL_EVERY:]),
                    "train_recent_success": avg(row.get("correct") for row in update_logs[-VAL_EVERY:] if "correct" in row),
                    "train_recent_parse": avg(row.get("parse_success") for row in update_logs[-VAL_EVERY:] if "parse_success" in row),
                    "error_count": error_count,
                }
                val_logs.append(val_log)
                if val_reward > best_val_reward + 1e-9:
                    best_val_reward = val_reward
                    best_update = update
                    best_alpha = float(val_log["best_val_alpha"])
                    stale_evals = 0
                    torch.save(
                        {
                            "adapter_state_dict": adapter.state_dict(),
                            "adapter_kind": "low_rank",
                            "adapter_rank": 32,
                            "adapter_params": parameter_count(adapter),
                            "update": update,
                            "best_val_reward": best_val_reward,
                            "best_val_alpha": best_alpha,
                            "intervention_mode": PRIMARY_MODE,
                            "target_layer": 36,
                            "train_task_ids": dataset.get("split_task_ids", {}).get("train", []),
                            "val_task_ids": dataset.get("split_task_ids", {}).get("val", []),
                            "heldout_task_ids": dataset.get("split_task_ids", {}).get("heldout", []),
                            "init_source": init_source,
                        },
                        OUT_BEST,
                    )
                else:
                    stale_evals += 1
                write_json(
                    OUT_JSON,
                    {
                        "BG_SEQUENCE_ADAPTER_TRAINING_VERDICT": "PARTIAL",
                        "status": "training",
                        "adapter_kind": "low_rank",
                        "adapter_rank": 32,
                        "adapter_params": parameter_count(adapter),
                        "init_source": init_source,
                        "optimizer": "REINFORCE_score_function",
                        "max_updates": MAX_UPDATES,
                        "updates_completed": update,
                        "best_update": best_update,
                        "best_val_reward": best_val_reward,
                        "best_val_alpha": best_alpha,
                        "baseline_val_reward": baseline_val_reward,
                        "update_logs": update_logs,
                        "val_logs": val_logs,
                        "elapsed_seconds": round(time.time() - started, 3),
                    },
                )
                if stale_evals >= PATIENCE_EVALS or error_count > 5:
                    break
                adapter.train()
            if device.type == "cuda" and update % 5 == 0:
                torch.cuda.empty_cache()
        baseline_val = float(baseline_val_reward if baseline_val_reward is not None else 0.0)
        lift = best_val_reward - baseline_val
        final_parse = avg(row.get("parse_success") for row in update_logs if "parse_success" in row) or 0.0
        if error_count > 5 or final_parse < 0.5:
            verdict = "DESTABILIZING"
        elif best_update <= 0:
            verdict = "NO_REWARD_IMPROVEMENT"
        elif lift > 0.05:
            verdict = "SEQUENCE_REWARD_IMPROVES"
        elif lift > 0:
            verdict = "WEAK_REWARD_IMPROVEMENT"
        else:
            verdict = "NO_REWARD_IMPROVEMENT"
        payload = {
            "BG_SEQUENCE_ADAPTER_TRAINING_VERDICT": verdict,
            "adapter_kind": "low_rank",
            "adapter_rank": 32,
            "adapter_params": parameter_count(adapter),
            "init_source": init_source,
            "optimizer": "REINFORCE_score_function",
            "do_not_backprop_through_sampling": True,
            "max_updates": MAX_UPDATES,
            "updates_completed": len(update_logs),
            "val_every": VAL_EVERY,
            "early_stopping_patience_evals": PATIENCE_EVALS,
            "train_tasks": len(train_tasks),
            "val_tasks": len(val_tasks),
            "train_alpha": TRAIN_ALPHA,
            "best_update": best_update,
            "best_val_reward": best_val_reward,
            "best_val_alpha": best_alpha,
            "baseline_val_reward": baseline_val_reward,
            "best_val_reward_lift_over_baseline": lift,
            "best_adapter_path": rel(OUT_BEST),
            "loss_weights": {"lambda_kl": LAMBDA_KL, "lambda_delta": LAMBDA_DELTA, "lambda_entropy": LAMBDA_ENTROPY},
            "update_logs": update_logs,
            "val_logs": val_logs,
            "error_count": error_count,
            "elapsed_seconds": round(time.time() - started, 3),
        }
    except Exception as exc:
        payload = {
            "BG_SEQUENCE_ADAPTER_TRAINING_VERDICT": "BLOCKED",
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "traceback": traceback.format_exc()[-4000:],
            "update_logs": update_logs,
            "val_logs": val_logs,
            "elapsed_seconds": round(time.time() - started, 3),
        }
    write_json(OUT_JSON, payload)
    verdict = payload["BG_SEQUENCE_ADAPTER_TRAINING_VERDICT"]
    lines = [
        "# BG Sequence-Level Adapter Training",
        "",
        f"BG_SEQUENCE_ADAPTER_TRAINING_VERDICT = {verdict}",
        "",
        f"- optimizer: `{payload.get('optimizer')}`",
        f"- updates completed: `{payload.get('updates_completed')}`",
        f"- best update: `{payload.get('best_update')}`",
        f"- best val reward: `{payload.get('best_val_reward')}`",
        f"- baseline val reward: `{payload.get('baseline_val_reward')}`",
        f"- val reward lift: `{payload.get('best_val_reward_lift_over_baseline')}`",
        f"- best val alpha: `{payload.get('best_val_alpha')}`",
        f"- best adapter: `{payload.get('best_adapter_path')}`",
        "",
        "## Validation Log",
        "",
        "| update | best val reward | best alpha | recent train reward | recent parse | errors |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("val_logs", []):
        lines.append(
            f"| {row['update']} | {row['best_val_reward']:.3f} | {row['best_val_alpha']:.3f} | "
            f"{row.get('train_recent_reward_mean') or 0.0:.3f} | {row.get('train_recent_parse') or 0.0:.3f} | {row.get('error_count', 0)} |"
        )
    if payload.get("error"):
        lines.extend(["", "## Blocker", "", f"- `{payload['error_type']}: {payload['error'][:500]}`"])
    write_md(OUT_MD, lines)
    print(f"BG_SEQUENCE_ADAPTER_TRAINING_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
