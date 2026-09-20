#!/usr/bin/env python3
"""Micro REINFORCE sanity check for sequence-level BG adapter training."""
from __future__ import annotations

import os
import time
import traceback
from typing import Any

import torch

from bg_sequence_adapter_common import (
    PRIMARY_MODE,
    QUICK_OUT_ROOT,
    avg,
    generate_completion,
    generation_prompt,
    load_model_tokenizer,
    load_stage1_mcq_tasks,
    parse_mcq_answer,
    policy_logprob_terms,
    rel,
    select_balanced_tasks,
    set_seed,
    write_json,
    write_md,
)
from src.evaluator.bg_sequence_adapter import build_sequence_adapter, parameter_count, sequence_adapter_hook


OUT_JSON = QUICK_OUT_ROOT / "optimizer_sanity_micro.json"
OUT_MD = QUICK_OUT_ROOT / "optimizer_sanity_micro.md"
TASK_CAP = min(4, max(1, int(os.environ.get("BG_SEQUENCE_SANITY_MICRO_TASKS", "4"))))
MAX_UPDATES = min(20, max(1, int(os.environ.get("BG_SEQUENCE_SANITY_MICRO_UPDATES", "6"))))
MAX_NEW_TOKENS = min(64, int(os.environ.get("BG_SEQUENCE_SANITY_MICRO_TOKENS", "64")))
LR = float(os.environ.get("BG_SEQUENCE_SANITY_MICRO_LR", "0.005"))
TARGET_OPTION = os.environ.get("BG_SEQUENCE_SANITY_TARGET_OPTION", "A").strip().upper()[:1] or "A"


def trivial_reward(task: dict[str, Any], text: str) -> float:
    parsed = parse_mcq_answer(text, task.get("options"))
    return 1.0 if parsed.get("parsed_answer") == TARGET_OPTION else 0.0


def evaluate_trivial(model: Any, tokenizer: Any, device: torch.device, adapter: torch.nn.Module, tasks: list[dict[str, Any]], alpha: float, base_seed: int) -> list[dict[str, Any]]:
    rows = []
    adapter.eval()
    for idx, task in enumerate(tasks):
        prompt = generation_prompt(task)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536, padding=False)
        position = max(0, int(enc["input_ids"].shape[1]) - 1)
        with sequence_adapter_hook(model, adapter, alpha=alpha, intervention_mode=PRIMARY_MODE, position=position, max_alpha=max(0.02, alpha)) as hook:
            gen = generate_completion(
                model,
                tokenizer,
                device,
                prompt,
                seed=base_seed + idx,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                hook_context=None,
            )
            diagnostics = hook.diagnostics()
        parsed = parse_mcq_answer(gen["output_text"], task.get("options"))
        rows.append(
            {
                "task_id": task["task_id"],
                "domain": task["domain"],
                "target_option": TARGET_OPTION,
                "output_text": gen["output_text"],
                "parsed_answer": parsed.get("parsed_answer"),
                "trivial_reward": trivial_reward(task, gen["output_text"]),
                "parse_success": bool(parsed.get("parse_success")),
                "output_length": gen["output_length"],
                "generation_error": gen["generation_error"],
                "activation_rms_change": diagnostics.get("activation_rms_change", 0.0),
            }
        )
    return rows


def train_once(model: Any, tokenizer: Any, device: torch.device, tasks: list[dict[str, Any]], alpha: float) -> dict[str, Any]:
    adapter = build_sequence_adapter(kind="low_rank", rank=32, device=device)
    adapter.train()
    initial_flat = torch.cat([p.detach().flatten().float().cpu() for p in adapter.parameters()])
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=LR, weight_decay=0.0)
    initial_rows = evaluate_trivial(model, tokenizer, device, adapter, tasks, alpha, 20260518)
    baseline = 0.25
    logs: list[dict[str, Any]] = []
    nonzero_grad_updates = 0
    for update in range(1, MAX_UPDATES + 1):
        task = tasks[(update - 1) % len(tasks)]
        prompt = generation_prompt(task)
        prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536, padding=False)["input_ids"].to(device)
        position = max(0, int(prompt_ids.shape[1]) - 1)
        with sequence_adapter_hook(model, adapter, alpha=alpha, intervention_mode=PRIMARY_MODE, position=position, max_alpha=max(0.02, alpha)):
            gen = generate_completion(
                model,
                tokenizer,
                device,
                prompt,
                seed=20261518 + update,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
            )
        reward = trivial_reward(task, gen["output_text"])
        terms = policy_logprob_terms(
            model,
            adapter,
            gen["prompt_input_ids"].to(device),
            gen["new_ids"].to(device),
            alpha=alpha,
            mode=PRIMARY_MODE,
            max_alpha=max(0.02, alpha),
            lambda_kl=0.02,
            lambda_delta=0.05,
        )
        advantage = float(reward - baseline)
        loss = -advantage * terms["sum_logprob"] + 0.02 * terms["kl"] + 0.05 * terms["delta_loss"] - 0.001 * terms["entropy"]
        optimizer.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            nonzero_grad_updates += int(float(grad_norm.detach().cpu().item()) > 0.0)
            optimizer.step()
        else:
            grad_norm = torch.zeros(())
        baseline = 0.8 * baseline + 0.2 * reward
        logs.append(
            {
                "update": update,
                "task_id": task["task_id"],
                "reward": reward,
                "advantage": advantage,
                "loss": float(loss.detach().cpu().item()) if torch.isfinite(loss) else None,
                "sum_logprob": float(terms["sum_logprob"].detach().cpu().item()),
                "kl": float(terms["kl"].detach().cpu().item()),
                "entropy": float(terms["entropy"].detach().cpu().item()),
                "delta_loss": float(terms["delta_loss"].detach().cpu().item()),
                "grad_norm": float(grad_norm.detach().cpu().item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                "output_preview": gen["output_text"][:200],
            }
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    final_rows = evaluate_trivial(model, tokenizer, device, adapter, tasks, alpha, 20262518)
    final_flat = torch.cat([p.detach().flatten().float().cpu() for p in adapter.parameters()])
    param_delta = float(torch.linalg.vector_norm(final_flat - initial_flat).item())
    initial_reward = avg(row["trivial_reward"] for row in initial_rows) or 0.0
    final_reward = avg(row["trivial_reward"] for row in final_rows) or 0.0
    changed_outputs = sum(
        1 for before, after in zip(initial_rows, final_rows) if str(before.get("output_text")) != str(after.get("output_text"))
    )
    return {
        "alpha": alpha,
        "adapter_params": parameter_count(adapter),
        "initial_rows": initial_rows,
        "final_rows": final_rows,
        "update_logs": logs,
        "initial_trivial_reward": initial_reward,
        "final_trivial_reward": final_reward,
        "reward_lift": final_reward - initial_reward,
        "changed_output_count": changed_outputs,
        "adapter_param_delta_l2": param_delta,
        "nonzero_grad_updates": nonzero_grad_updates,
        "adapter_params_changed": param_delta > 0.0,
    }


def main() -> int:
    started = time.time()
    set_seed()
    QUICK_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        tasks = select_balanced_tasks(load_stage1_mcq_tasks(), TASK_CAP)
        if len(tasks) < 1:
            raise RuntimeError("no clean MCQ tasks found")
        model, tokenizer, device = load_model_tokenizer()
        alpha_002 = train_once(model, tokenizer, device, tasks, 0.02)
        relaxed = None
        if alpha_002["reward_lift"] <= 0 and alpha_002["changed_output_count"] == 0 and not alpha_002["adapter_params_changed"]:
            relaxed = train_once(model, tokenizer, device, tasks, 0.05)
        best = relaxed if relaxed and relaxed["reward_lift"] > alpha_002["reward_lift"] else alpha_002
        moved = best["reward_lift"] > 0 or best["changed_output_count"] > 0 or best["adapter_param_delta_l2"] > 0
        if moved and best["reward_lift"] > 0:
            verdict = "OPTIMIZER_MOVES_ADAPTER"
        elif moved:
            verdict = "OPTIMIZER_WEAK"
        else:
            verdict = "OPTIMIZER_NO_MOVEMENT"
        payload = {
            "BG_SEQUENCE_OPTIMIZER_SANITY_MICRO_VERDICT": verdict,
            "SANITY_RELAXED_ALPHA_USED": relaxed is not None,
            "target_option": TARGET_OPTION,
            "task_count": len(tasks),
            "max_updates": MAX_UPDATES,
            "max_new_tokens": MAX_NEW_TOKENS,
            "sanity_alpha_0_02_result": alpha_002,
            "sanity_alpha_relaxed_result": relaxed,
            "selected_result": best,
            "elapsed_seconds": round(time.time() - started, 3),
        }
    except Exception as exc:
        payload = {
            "BG_SEQUENCE_OPTIMIZER_SANITY_MICRO_VERDICT": "BLOCKED",
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "traceback": traceback.format_exc()[-4000:],
            "elapsed_seconds": round(time.time() - started, 3),
        }
        verdict = "BLOCKED"
    write_json(OUT_JSON, payload)
    lines = [
        "# BG Sequence Adapter Optimizer Sanity Micro-Run",
        "",
        f"BG_SEQUENCE_OPTIMIZER_SANITY_MICRO_VERDICT = {payload['BG_SEQUENCE_OPTIMIZER_SANITY_MICRO_VERDICT']}",
        "",
        f"- target option: `{payload.get('target_option', TARGET_OPTION)}`",
        f"- relaxed alpha used: `{payload.get('SANITY_RELAXED_ALPHA_USED', False)}`",
        f"- max updates: `{payload.get('max_updates', MAX_UPDATES)}`",
    ]
    if "selected_result" in payload:
        selected = payload["selected_result"]
        lines.extend(
            [
                f"- selected alpha: `{selected['alpha']}`",
                f"- initial trivial reward: `{selected['initial_trivial_reward']:.3f}`",
                f"- final trivial reward: `{selected['final_trivial_reward']:.3f}`",
                f"- reward lift: `{selected['reward_lift']:.3f}`",
                f"- changed outputs: `{selected['changed_output_count']}`",
                f"- adapter param delta L2: `{selected['adapter_param_delta_l2']:.6f}`",
                f"- nonzero grad updates: `{selected['nonzero_grad_updates']}`",
            ]
        )
    elif "error" in payload:
        lines.append(f"- error: `{payload['error_type']}: {payload['error'][:500]}`")
    write_md(OUT_MD, lines)
    print(f"BG_SEQUENCE_OPTIMIZER_SANITY_MICRO_VERDICT = {payload['BG_SEQUENCE_OPTIMIZER_SANITY_MICRO_VERDICT']}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if payload["BG_SEQUENCE_OPTIMIZER_SANITY_MICRO_VERDICT"] != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
