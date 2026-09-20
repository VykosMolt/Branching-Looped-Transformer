#!/usr/bin/env python3
"""Full-harness optimizer sanity check for BG sequence adapter training."""
from __future__ import annotations

import os
import time
import traceback
from typing import Any

import torch

from bg_sequence_adapter_common import (
    OUT_ROOT,
    PRIMARY_MODE,
    avg,
    generate_completion,
    generation_prompt,
    load_model_tokenizer,
    load_sequence_dataset,
    parse_mcq_answer,
    policy_logprob_terms,
    rel,
    rows_for_split,
    set_seed,
    write_json,
    write_md,
)
from src.evaluator.bg_sequence_adapter import build_sequence_adapter, parameter_count, sequence_adapter_hook


OUT_JSON = OUT_ROOT / "optimizer_sanity.json"
OUT_MD = OUT_ROOT / "optimizer_sanity.md"
MAX_UPDATES = min(100, max(10, int(os.environ.get("BG_SEQUENCE_OPTIMIZER_SANITY_UPDATES", "50"))))
TASK_CAP = min(8, max(4, int(os.environ.get("BG_SEQUENCE_OPTIMIZER_SANITY_TASKS", "6"))))
MAX_NEW_TOKENS = min(64, int(os.environ.get("BG_SEQUENCE_OPTIMIZER_SANITY_TOKENS", "64")))
TARGET_OPTION = os.environ.get("BG_SEQUENCE_SANITY_TARGET_OPTION", "A").strip().upper()[:1] or "A"
LR = float(os.environ.get("BG_SEQUENCE_OPTIMIZER_SANITY_LR", "0.005"))


def trivial_reward(task: dict[str, Any], text: str) -> float:
    parsed = parse_mcq_answer(text, task.get("options"))
    return 1.0 if parsed.get("parsed_answer") == TARGET_OPTION else 0.0


def eval_adapter(model: Any, tokenizer: Any, device: torch.device, adapter: torch.nn.Module, tasks: list[dict[str, Any]], alpha: float, seed_base: int) -> list[dict[str, Any]]:
    adapter.eval()
    rows = []
    for idx, task in enumerate(tasks):
        prompt = generation_prompt(task)
        prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536, padding=False)["input_ids"].to(device)
        position = max(0, int(prompt_ids.shape[1]) - 1)
        with sequence_adapter_hook(model, adapter, alpha=alpha, intervention_mode=PRIMARY_MODE, position=position, max_alpha=max(alpha, 0.02)) as hook:
            gen = generate_completion(model, tokenizer, device, prompt, seed=seed_base + idx, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            diagnostics = hook.diagnostics()
        parsed = parse_mcq_answer(gen["output_text"], task.get("options"))
        rows.append(
            {
                "task_id": task["task_id"],
                "domain": task["domain"],
                "output_text": gen["output_text"],
                "parsed_answer": parsed.get("parsed_answer"),
                "trivial_reward": trivial_reward(task, gen["output_text"]),
                "parse_success": bool(parsed.get("parse_success")),
                "generation_error": gen["generation_error"],
                "activation_rms_change": diagnostics.get("activation_rms_change", 0.0),
            }
        )
    return rows


def run_alpha(model: Any, tokenizer: Any, device: torch.device, tasks: list[dict[str, Any]], alpha: float) -> dict[str, Any]:
    adapter = build_sequence_adapter(kind="low_rank", rank=32, device=device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=LR, weight_decay=0.0)
    initial_params = torch.cat([p.detach().flatten().float().cpu() for p in adapter.parameters()])
    initial = eval_adapter(model, tokenizer, device, adapter, tasks, alpha, 20260518)
    baseline = 0.25
    logs = []
    nonzero_grad = 0
    for update in range(1, MAX_UPDATES + 1):
        adapter.train()
        task = tasks[(update - 1) % len(tasks)]
        prompt = generation_prompt(task)
        prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536, padding=False)["input_ids"].to(device)
        position = max(0, int(prompt_ids.shape[1]) - 1)
        with sequence_adapter_hook(model, adapter, alpha=alpha, intervention_mode=PRIMARY_MODE, position=position, max_alpha=max(alpha, 0.02)):
            gen = generate_completion(model, tokenizer, device, prompt, seed=20261518 + update, max_new_tokens=MAX_NEW_TOKENS, do_sample=True)
        reward = trivial_reward(task, gen["output_text"])
        terms = policy_logprob_terms(model, adapter, gen["prompt_input_ids"].to(device), gen["new_ids"].to(device), alpha=alpha, mode=PRIMARY_MODE, max_alpha=max(alpha, 0.02))
        advantage = reward - baseline
        loss = -float(advantage) * terms["sum_logprob"] + 0.02 * terms["kl"] + 0.05 * terms["delta_loss"] - 0.001 * terms["entropy"]
        optimizer.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            nonzero_grad += int(float(grad_norm.detach().cpu().item()) > 0)
            optimizer.step()
        else:
            grad_norm = torch.zeros(())
        baseline = 0.9 * baseline + 0.1 * reward
        logs.append(
            {
                "update": update,
                "task_id": task["task_id"],
                "reward": reward,
                "advantage": float(advantage),
                "loss": float(loss.detach().cpu().item()) if torch.isfinite(loss) else None,
                "grad_norm": float(grad_norm.detach().cpu().item()),
                "kl": float(terms["kl"].detach().cpu().item()),
                "delta_loss": float(terms["delta_loss"].detach().cpu().item()),
                "entropy": float(terms["entropy"].detach().cpu().item()),
            }
        )
        if update % 10 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()
    final = eval_adapter(model, tokenizer, device, adapter, tasks, alpha, 20262518)
    final_params = torch.cat([p.detach().flatten().float().cpu() for p in adapter.parameters()])
    initial_reward = avg(row["trivial_reward"] for row in initial) or 0.0
    final_reward = avg(row["trivial_reward"] for row in final) or 0.0
    changed = sum(1 for a, b in zip(initial, final) if a.get("output_text") != b.get("output_text"))
    return {
        "alpha": alpha,
        "adapter_params": parameter_count(adapter),
        "initial_rows": initial,
        "final_rows": final,
        "initial_trivial_reward": initial_reward,
        "final_trivial_reward": final_reward,
        "reward_lift": final_reward - initial_reward,
        "changed_output_count": changed,
        "adapter_param_delta_l2": float(torch.linalg.vector_norm(final_params - initial_params).item()),
        "nonzero_grad_updates": nonzero_grad,
        "update_logs": logs,
    }


def main() -> int:
    started = time.time()
    set_seed()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        dataset = load_sequence_dataset()
        tasks = rows_for_split(dataset, "train")[:TASK_CAP]
        if len(tasks) < 4:
            raise RuntimeError(f"insufficient sanity train tasks: {len(tasks)}")
        model, tokenizer, device = load_model_tokenizer()
        alpha_002 = run_alpha(model, tokenizer, device, tasks, 0.02)
        relaxed = None
        if alpha_002["reward_lift"] <= 0 and alpha_002["changed_output_count"] == 0:
            relaxed = run_alpha(model, tokenizer, device, tasks, 0.05)
        best = relaxed if relaxed and relaxed["reward_lift"] > alpha_002["reward_lift"] else alpha_002
        if best["reward_lift"] > 0:
            verdict = "OPTIMIZER_CAN_LEARN_TRIVIAL_TARGET"
        elif best["changed_output_count"] > 0 or best["nonzero_grad_updates"] > 0:
            verdict = "OPTIMIZER_WEAK_ON_TRIVIAL_TARGET"
        else:
            verdict = "OPTIMIZER_CANNOT_LEARN_TRIVIAL_TARGET"
        payload = {
            "BG_SEQUENCE_OPTIMIZER_SANITY_VERDICT": verdict,
            "target_option": TARGET_OPTION,
            "task_count": len(tasks),
            "max_updates": MAX_UPDATES,
            "sanity_alpha_0_02_result": alpha_002,
            "sanity_alpha_relaxed_result": relaxed,
            "selected_result": best,
            "elapsed_seconds": round(time.time() - started, 3),
        }
    except Exception as exc:
        payload = {
            "BG_SEQUENCE_OPTIMIZER_SANITY_VERDICT": "BLOCKED",
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "traceback": traceback.format_exc()[-4000:],
            "elapsed_seconds": round(time.time() - started, 3),
        }
    write_json(OUT_JSON, payload)
    verdict = payload["BG_SEQUENCE_OPTIMIZER_SANITY_VERDICT"]
    lines = [
        "# BG Sequence Adapter Optimizer Sanity",
        "",
        f"BG_SEQUENCE_OPTIMIZER_SANITY_VERDICT = {verdict}",
        "",
        f"- target option: `{payload.get('target_option', TARGET_OPTION)}`",
        f"- max updates: `{payload.get('max_updates', MAX_UPDATES)}`",
    ]
    if payload.get("selected_result"):
        selected = payload["selected_result"]
        lines.extend(
            [
                f"- selected alpha: `{selected['alpha']}`",
                f"- initial trivial reward: `{selected['initial_trivial_reward']:.3f}`",
                f"- final trivial reward: `{selected['final_trivial_reward']:.3f}`",
                f"- reward lift: `{selected['reward_lift']:.3f}`",
                f"- changed outputs: `{selected['changed_output_count']}`",
                f"- nonzero grad updates: `{selected['nonzero_grad_updates']}`",
                f"- adapter param delta L2: `{selected['adapter_param_delta_l2']:.6f}`",
            ]
        )
    elif payload.get("error"):
        lines.append(f"- error: `{payload['error_type']}: {payload['error'][:500]}`")
    write_md(OUT_MD, lines)
    print(f"BG_SEQUENCE_OPTIMIZER_SANITY_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
