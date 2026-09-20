#!/usr/bin/env python3
"""Free-generation transfer evaluation for the causal BG intervention adapter."""
from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any

import torch

from bg_causal_adapter_common import (
    HIDDEN_DIM,
    OUT_ROOT,
    PRIMARY_MODE,
    adapter_hook,
    avg,
    build_teacher_forced_text,
    encode_example,
    finite,
    load_adapter_dataset,
    load_empirical_direction,
    load_json,
    load_model_tokenizer,
    rel,
    rows_for_split,
    set_seed,
    write_json,
    write_md,
)
from bg_steering_suite_lib import evaluate_output
from run_bg_stage2_intervention_sweep import repetition_rate
from src.evaluator.bg_causal_adapter import LowRankDeltaAdapter
from src.evaluator.bg_steering_hook import BGLayerHookSteering, build_intervention_mode


OUT_JSON = OUT_ROOT / "free_generation_eval.json"
OUT_MD = OUT_ROOT / "free_generation_eval.md"
CKPT = OUT_ROOT / "adapter_checkpoints/best_adapter.pt"
TEACHER_JSON = OUT_ROOT / "teacher_forced_eval.json"
HELDOUT_TASK_CAP = int(os.environ.get("BG_CAUSAL_ADAPTER_FREE_GEN_TASK_CAP", "8"))
MAX_NEW_TOKENS = min(128, int(os.environ.get("BG_CAUSAL_ADAPTER_FREE_GEN_TOKENS", "64")))


def rms_direction(vec: torch.Tensor) -> torch.Tensor:
    flat = vec.detach().flatten().to(dtype=torch.float32, device="cpu")
    return flat / flat.pow(2).mean().sqrt().clamp(min=1e-8)


def random_rms_direction(seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    return rms_direction(torch.randn(HIDDEN_DIM, generator=gen))


def unique_task_rows(rows: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        task_id = str(row.get("task_id"))
        if task_id in seen:
            continue
        selected.append(row)
        seen.add(task_id)
        if len(selected) >= cap:
            break
    return selected


def best_teacher_alpha() -> float:
    payload = load_json(TEACHER_JSON, {})
    best = payload.get("best_adapter") or {}
    alpha = finite(best.get("alpha"), 0.01)
    return alpha if alpha in {0.005, 0.01, 0.02} else 0.01


def task_for_eval(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": example["task_id"],
        "domain": example["domain"],
        "prompt": example["prompt"],
        "options": example["options"],
        "answer_key": example["correct_option"],
    }


def generate_text(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompt: str,
    *,
    seed: int,
    hook: Any | None = None,
) -> tuple[str, dict[str, Any]]:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536, padding=False)
    enc = {key: value.to(device) for key, value in enc.items()}
    prompt_len = int(enc["input_ids"].shape[1])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    started = time.time()
    diagnostics: dict[str, Any] = {}
    error = ""
    try:
        if hook is not None:
            hook.apply()
        with torch.no_grad():
            generated = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )
    except Exception as exc:  # record and let caller mark CUDA/runtime failures.
        generated = enc["input_ids"]
        error = str(exc)[:500]
    finally:
        if hook is not None:
            diagnostics = hook.diagnostics()
            hook.remove()
    new_ids = generated[0, prompt_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return text, {
        "token_count": int(new_ids.numel()),
        "hit_max_tokens": int(new_ids.numel()) >= MAX_NEW_TOKENS,
        "seconds": round(time.time() - started, 3),
        "generation_error": error,
        "diagnostics": diagnostics,
    }


class AdapterGenerationHook:
    def __init__(self, model: Any, adapter: torch.nn.Module, alpha: float, mode: str, position: int) -> None:
        self.position = int(position)
        self.inner = adapter_hook(model, adapter, alpha=float(alpha), intervention_mode=mode, position=self.position)
        self.hook = None

    def apply(self) -> None:
        self.hook = self.inner.__enter__()

    def diagnostics(self) -> dict[str, Any]:
        return self.hook.diagnostics() if self.hook is not None else {}

    def remove(self) -> None:
        self.inner.__exit__(None, None, None)


class StaticGenerationHook:
    def __init__(self, model: Any, direction: torch.Tensor, alpha: float, mode: str, position: int) -> None:
        spec = build_intervention_mode(mode, float(alpha))
        self.hook = BGLayerHookSteering(
            model,
            target_layer=36,
            target_loops=spec["target_loops"],
            direction=direction,
            alpha=float(alpha),
            loop_alpha_scales=spec["loop_alpha_scales"],
            max_rms_fraction=0.02,
            direction_normalization="rms",
        )
        self.position = int(position)

    def apply(self) -> None:
        self.hook.apply(position=self.position)

    def diagnostics(self) -> dict[str, Any]:
        raw = self.hook.diagnostics()
        return {
            "hook_forward_call_count": raw.get("forward_call_count", 0),
            "hook_modifications": raw.get("modifications", 0),
            "hook_loop_index_source": raw.get("loop_index_source", ""),
            "activation_rms_change": raw.get("activation_rms_change", 0.0),
            "per_loop_activation_rms_change": raw.get("per_loop_activation_rms_change", {}),
            "nan_or_inf_activations": raw.get("nan_or_inf", False),
        }

    def remove(self) -> None:
        self.hook.remove()


def run_condition(
    model: Any,
    tokenizer: Any,
    adapter: torch.nn.Module,
    device: torch.device,
    example: dict[str, Any],
    *,
    method: str,
    alpha: float,
    direction: torch.Tensor | None,
    seed: int,
) -> dict[str, Any]:
    encoded = encode_example(tokenizer, example, device)
    _, prompt = build_teacher_forced_text(example["prompt"], example["prefix_text"])
    hook = None
    if method == "trained_adapter":
        hook = AdapterGenerationHook(model, adapter, alpha, PRIMARY_MODE, int(encoded["intervention_token_index"]))
    elif method != "baseline":
        if direction is None:
            raise ValueError(f"{method} needs a static direction")
        hook = StaticGenerationHook(model, direction, alpha, PRIMARY_MODE, int(encoded["intervention_token_index"]))
    text, meta = generate_text(model, tokenizer, device, prompt, seed=seed, hook=hook)
    evaluation = evaluate_output(task_for_eval(example), text)
    diagnostics = meta.get("diagnostics") or {}
    return {
        "example_id": example["example_id"],
        "task_id": example["task_id"],
        "domain": example["domain"],
        "method": method,
        "mode": PRIMARY_MODE,
        "alpha": float(alpha),
        "seed": int(seed),
        "output_text": text,
        "parsed_answer": evaluation.get("parsed_answer"),
        "is_correct": bool(evaluation.get("success")),
        "parse_failed": not bool(evaluation.get("parsed")),
        "parse_rate": 1.0 if bool(evaluation.get("parsed")) else 0.0,
        "repetition_rate": repetition_rate(text),
        "empty_output": not bool(text.strip()),
        "hit_max_tokens": bool(meta.get("hit_max_tokens")),
        "output_length": int(meta.get("token_count", 0)),
        "generation_seconds": meta.get("seconds"),
        "generation_error": meta.get("generation_error", ""),
        "cuda_error": "cuda" in str(meta.get("generation_error", "")).lower(),
        "first_answer_logit_measured": False,
        "correct_option_logit_at_first_answer_position": None,
        "BG_score_shift": None,
        "BG_DIAGNOSTICS_COMPUTED": False,
        "cache_intervention_mode": "disabled",
        "intervention_position_kind": encoded["intervention_position_kind"],
        "intervention_token_index": int(encoded["intervention_token_index"]),
        "answer_logit_token_index": int(encoded["answer_logit_token_index"]),
        "hook_forward_call_count": diagnostics.get("hook_forward_call_count", 0),
        "hook_modifications": diagnostics.get("hook_modifications", 0),
        "hook_loop_index_source": diagnostics.get("hook_loop_index_source", diagnostics.get("loop_index_source", "")),
        "activation_rms_change": diagnostics.get("activation_rms_change", 0.0),
        "per_loop_activation_rms_change": diagnostics.get("per_loop_activation_rms_change", {}),
        "nan_or_inf_activations": bool(diagnostics.get("nan_or_inf_activations", diagnostics.get("nan_or_inf", False))),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["method"]), float(row["alpha"]))].append(row)
    out: dict[str, Any] = {}
    for (method, alpha), vals in sorted(groups.items()):
        key = f"{method}@{alpha:.3f}"
        out[key] = {
            "method": method,
            "alpha": alpha,
            "n": len(vals),
            "success_rate": avg(row["is_correct"] for row in vals),
            "parse_rate": avg(1.0 - float(row["parse_failed"]) for row in vals),
            "repetition_rate": avg(row["repetition_rate"] for row in vals),
            "empty_output_rate": avg(row["empty_output"] for row in vals),
            "hit_max_tokens_rate": avg(row["hit_max_tokens"] for row in vals),
            "output_length_mean": avg(row["output_length"] for row in vals),
            "cuda_error_count": sum(1 for row in vals if row.get("cuda_error")),
            "nan_or_inf_count": sum(1 for row in vals if row.get("nan_or_inf_activations")),
            "activation_rms_change": avg(row["activation_rms_change"] for row in vals),
        }
    return out


def main() -> int:
    started = time.time()
    set_seed()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    dataset = load_adapter_dataset()
    examples = unique_task_rows(rows_for_split(dataset, "heldout"), HELDOUT_TASK_CAP)
    if not CKPT.exists() or not examples:
        payload = {
            "BG_CAUSAL_ADAPTER_FREE_GEN_VERDICT": "INSUFFICIENT",
            "FREE_GENERATION_EVAL_COMPLETED": False,
            "blocker": f"checkpoint_exists={CKPT.exists()} heldout_tasks={len(examples)}",
            "rows": [],
            "elapsed_seconds": round(time.time() - started, 3),
        }
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# BG Causal Adapter Free-Generation Evaluation", "", "BG_CAUSAL_ADAPTER_FREE_GEN_VERDICT = INSUFFICIENT", "", f"- blocker: `{payload['blocker']}`"])
        print("BG_CAUSAL_ADAPTER_FREE_GEN_VERDICT = INSUFFICIENT")
        return 1

    model, tokenizer, device = load_model_tokenizer()
    adapter = LowRankDeltaAdapter(rank=32).to(device)
    checkpoint = torch.load(CKPT, map_location=device, weights_only=False)
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    adapter.eval()
    raw_direction = load_empirical_direction("RAW_NONORM_READOUT")
    raw_direction = rms_direction(raw_direction) if raw_direction is not None else None
    alphas = sorted({best_teacher_alpha(), 0.01, 0.02})

    rows: list[dict[str, Any]] = []
    for task_idx, example in enumerate(examples):
        rows.append(
            run_condition(
                model,
                tokenizer,
                adapter,
                device,
                example,
                method="baseline",
                alpha=0.0,
                direction=None,
                seed=20260518 + task_idx,
            )
        )
        for alpha in alphas:
            rows.append(
                run_condition(
                    model,
                    tokenizer,
                    adapter,
                    device,
                    example,
                    method="trained_adapter",
                    alpha=alpha,
                    direction=None,
                    seed=20261518 + task_idx * 10 + int(alpha * 1000),
                )
            )
            if raw_direction is not None:
                rows.append(
                    run_condition(
                        model,
                        tokenizer,
                        adapter,
                        device,
                        example,
                        method="RAW_NONORM_READOUT",
                        alpha=alpha,
                        direction=raw_direction,
                        seed=20262518 + task_idx * 10 + int(alpha * 1000),
                    )
                )
            seed = 20263518 + task_idx * 10 + int(alpha * 1000)
            rows.append(
                run_condition(
                    model,
                    tokenizer,
                    adapter,
                    device,
                    example,
                    method="random_same_rms",
                    alpha=alpha,
                    direction=random_rms_direction(seed),
                    seed=seed,
                )
            )
        write_json(
            OUT_JSON.with_suffix(".partial.json"),
            {
                "BG_CAUSAL_ADAPTER_FREE_GEN_VERDICT": "PARTIAL",
                "FREE_GENERATION_EVAL_COMPLETED": False,
                "rows": rows,
            },
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    agg = aggregate(rows)
    baseline = next((v for v in agg.values() if v["method"] == "baseline"), None)
    adapter_best = max(
        (v for v in agg.values() if v["method"] == "trained_adapter"),
        key=lambda row: finite(row.get("success_rate"), -1e9),
        default=None,
    )
    non_adapter_best = max(
        (v for v in agg.values() if v["method"] in {"RAW_NONORM_READOUT", "random_same_rms"}),
        key=lambda row: finite(row.get("success_rate"), -1e9),
        default=None,
    )
    teacher = load_json(TEACHER_JSON, {})
    teacher_good = teacher.get("BG_CAUSAL_ADAPTER_TEACHER_FORCED_VERDICT") == "ADAPTER_IMPROVES_LOGIT_MARGIN"
    adapter_success = finite(adapter_best.get("success_rate") if adapter_best else None, 0.0)
    baseline_success = finite(baseline.get("success_rate") if baseline else None, 0.0)
    non_adapter_success = finite(non_adapter_best.get("success_rate") if non_adapter_best else None, 0.0)
    stability_bad = any(row.get("cuda_error") or row.get("nan_or_inf_activations") for row in rows) or (
        adapter_best is not None and finite(adapter_best.get("parse_rate"), 1.0) < 0.5
    )
    if stability_bad:
        verdict = "FREE_GEN_DESTABILIZING"
    elif adapter_best is None:
        verdict = "INSUFFICIENT"
    elif adapter_success > max(baseline_success, non_adapter_success) + 1e-9:
        verdict = "FREE_GEN_LIFT"
    elif teacher_good:
        verdict = "TEACHER_FORCED_ONLY"
    else:
        verdict = "NO_FREE_GEN_EFFECT"

    payload = {
        "BG_CAUSAL_ADAPTER_FREE_GEN_VERDICT": verdict,
        "FREE_GENERATION_EVAL_COMPLETED": True,
        "heldout_tasks_used": len(examples),
        "max_new_tokens": MAX_NEW_TOKENS,
        "decode": {"do_sample": False, "use_cache": False},
        "alphas": alphas,
        "rows": rows,
        "aggregate": agg,
        "baseline": baseline,
        "best_adapter": adapter_best,
        "best_static_or_random": non_adapter_best,
        "free_generation_lift_over_baseline": adapter_success - baseline_success,
        "free_generation_lift_over_random_or_static": adapter_success - non_adapter_success,
        "cache_intervention_mode": "disabled",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    lines = [
        "# BG Causal Adapter Free-Generation Evaluation",
        "",
        f"BG_CAUSAL_ADAPTER_FREE_GEN_VERDICT = {verdict}",
        "",
        f"- free generation completed: `true`",
        f"- heldout tasks used: `{len(examples)}`",
        f"- max_new_tokens: `{MAX_NEW_TOKENS}`",
        f"- use_cache: `false`",
        f"- best adapter success: `{adapter_success:.3f}`",
        f"- baseline success: `{baseline_success:.3f}`",
        f"- best static/random success: `{non_adapter_success:.3f}`",
        f"- free-generation lift over baseline: `{adapter_success - baseline_success:.3f}`",
        "",
        "| method | alpha | n | success | parse | repetition | empty | hit max | output len | delta RMS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in agg.items():
        lines.append(
            f"| `{row['method']}` | {row['alpha']:.3f} | {row['n']} | {finite(row.get('success_rate')):.3f} | "
            f"{finite(row.get('parse_rate')):.3f} | {finite(row.get('repetition_rate')):.3f} | "
            f"{finite(row.get('empty_output_rate')):.3f} | {finite(row.get('hit_max_tokens_rate')):.3f} | "
            f"{finite(row.get('output_length_mean')):.1f} | {finite(row.get('activation_rms_change')):.6f} |"
        )
    write_md(OUT_MD, lines)
    print(f"BG_CAUSAL_ADAPTER_FREE_GEN_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
