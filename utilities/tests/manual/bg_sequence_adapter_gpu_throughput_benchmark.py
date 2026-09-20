#!/usr/bin/env python3
"""GPU throughput benchmark for BG sequence-level adapter runs."""
from __future__ import annotations

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
    policy_logprob_terms,
    rel,
    select_balanced_tasks,
    set_seed,
    write_json,
    write_md,
)
from src.evaluator.bg_sequence_adapter import build_sequence_adapter, sequence_adapter_hook


OUT_JSON = QUICK_OUT_ROOT / "gpu_throughput.json"
OUT_MD = QUICK_OUT_ROOT / "gpu_throughput.md"
MAX_NEW_TOKENS = 96


def main() -> int:
    started = time.time()
    set_seed()
    QUICK_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    update_metric: dict[str, Any] | None = None
    try:
        tasks = select_balanced_tasks(load_stage1_mcq_tasks(), 3)
        model, tokenizer, device = load_model_tokenizer()
        adapter = build_sequence_adapter(kind="low_rank", rank=32, device=device)
        for idx, task in enumerate(tasks):
            prompt = generation_prompt(task)
            gen = generate_completion(
                model,
                tokenizer,
                device,
                prompt,
                seed=20260518 + idx,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
            rows.append({"kind": "baseline_generation", **{k: gen[k] for k in ["output_length", "seconds", "generation_error"]}, "hook_forward_call_count": 0, "hook_modifications": 0})
        for idx, task in enumerate(tasks):
            prompt = generation_prompt(task)
            prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536, padding=False)["input_ids"]
            position = max(0, int(prompt_ids.shape[1]) - 1)
            with sequence_adapter_hook(model, adapter, alpha=0.01, intervention_mode=PRIMARY_MODE, position=position) as hook:
                gen = generate_completion(
                    model,
                    tokenizer,
                    device,
                    prompt,
                    seed=20261518 + idx,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )
                diagnostics = hook.diagnostics()
            rows.append(
                {
                    "kind": "intervention_generation",
                    **{k: gen[k] for k in ["output_length", "seconds", "generation_error"]},
                    "hook_forward_call_count": diagnostics.get("hook_forward_call_count", 0),
                    "hook_modifications": diagnostics.get("hook_modifications", 0),
                }
            )
        if tasks:
            optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.001)
            prompt = generation_prompt(tasks[0])
            prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536, padding=False)["input_ids"].to(device)
            position = max(0, int(prompt_ids.shape[1]) - 1)
            with sequence_adapter_hook(model, adapter, alpha=0.01, intervention_mode=PRIMARY_MODE, position=position):
                gen = generate_completion(model, tokenizer, device, prompt, seed=20262518, max_new_tokens=32, do_sample=True)
            t0 = time.time()
            terms = policy_logprob_terms(model, adapter, gen["prompt_input_ids"].to(device), gen["new_ids"].to(device), alpha=0.01, mode=PRIMARY_MODE)
            loss = -terms["sum_logprob"] + 0.02 * terms["kl"] + 0.05 * terms["delta_loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            update_metric = {
                "sec_update": round(time.time() - t0, 3),
                "loss": float(loss.detach().cpu().item()),
                "grad_norm": float(grad_norm.detach().cpu().item()),
                "generated_tokens_for_update": int(gen["new_ids"].numel()),
            }
    except Exception as exc:
        payload = {
            "BG_SEQUENCE_GPU_THROUGHPUT_VERDICT": "BLOCKED",
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "traceback": traceback.format_exc()[-4000:],
            "rows": rows,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# BG Sequence Adapter GPU Throughput", "", "BG_SEQUENCE_GPU_THROUGHPUT_VERDICT = BLOCKED", "", f"- error: `{type(exc).__name__}: {str(exc)[:500]}`"])
        print("BG_SEQUENCE_GPU_THROUGHPUT_VERDICT = BLOCKED")
        return 1

    baseline_sec = avg(row["seconds"] for row in rows if row["kind"] == "baseline_generation") or 0.0
    intervention_sec = avg(row["seconds"] for row in rows if row["kind"] == "intervention_generation") or 0.0
    sec_update = float((update_metric or {}).get("sec_update") or max(intervention_sec, 1.0))
    heldout12 = 12 * 4 * max(intervention_sec, baseline_sec)
    heldout24 = 24 * 4 * max(intervention_sec, baseline_sec)
    estimates = {
        "100_updates_hours": round((100 * (sec_update + intervention_sec)) / 3600, 3),
        "300_updates_hours": round((300 * (sec_update + intervention_sec)) / 3600, 3),
        "800_updates_hours": round((800 * (sec_update + intervention_sec)) / 3600, 3),
        "heldout_eval_n4_12_tasks_hours": round(heldout12 / 3600, 3),
        "heldout_eval_n4_24_tasks_hours": round(heldout24 / 3600, 3),
    }
    total_300_plus_eval = estimates["300_updates_hours"] + estimates["heldout_eval_n4_12_tasks_hours"]
    if total_300_plus_eval <= 16:
        verdict = "OVERNIGHT_FEASIBLE"
    elif total_300_plus_eval <= 24:
        verdict = "LONG_BUT_FEASIBLE"
    else:
        verdict = "TOO_SLOW_FOR_FULL_SCOPE"
    gpu = {
        "cuda_available": torch.cuda.is_available(),
        "device": str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "vram_allocated_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 3) if torch.cuda.is_available() else 0.0,
        "vram_reserved_gb": round(torch.cuda.memory_reserved(0) / (1024**3), 3) if torch.cuda.is_available() else 0.0,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    payload = {
        "BG_SEQUENCE_GPU_THROUGHPUT_VERDICT": verdict,
        "baseline_sec_per_generation": baseline_sec,
        "intervention_sec_per_generation": intervention_sec,
        "tokens_per_second_baseline": avg(row["output_length"] / max(row["seconds"], 1e-6) for row in rows if row["kind"] == "baseline_generation"),
        "tokens_per_second_intervention": avg(row["output_length"] / max(row["seconds"], 1e-6) for row in rows if row["kind"] == "intervention_generation"),
        "hook_forward_call_count_mean": avg(row["hook_forward_call_count"] for row in rows if row["kind"] == "intervention_generation"),
        "hook_modifications_mean": avg(row["hook_modifications"] for row in rows if row["kind"] == "intervention_generation"),
        "tiny_update": update_metric,
        "runtime_estimates": estimates,
        "gpu": gpu,
        "rows": rows,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    lines = [
        "# BG Sequence Adapter GPU Throughput",
        "",
        f"BG_SEQUENCE_GPU_THROUGHPUT_VERDICT = {verdict}",
        "",
        f"- GPU: `{gpu['gpu_name']}`",
        f"- baseline sec/generation: `{baseline_sec:.3f}`",
        f"- intervention sec/generation: `{intervention_sec:.3f}`",
        f"- sec/update: `{sec_update:.3f}`",
        f"- hook forward calls mean: `{payload['hook_forward_call_count_mean'] or 0.0:.1f}`",
        f"- hook modifications mean: `{payload['hook_modifications_mean'] or 0.0:.1f}`",
        "",
        "## Runtime Estimates",
        "",
    ]
    for key, value in estimates.items():
        lines.append(f"- {key}: `{value}` hours")
    write_md(OUT_MD, lines)
    print(f"BG_SEQUENCE_GPU_THROUGHPUT_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
