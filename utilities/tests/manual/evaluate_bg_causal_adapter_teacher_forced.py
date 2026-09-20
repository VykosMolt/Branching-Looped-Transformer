#!/usr/bin/env python3
"""Heldout teacher-forced evaluation for the causal BG intervention adapter."""
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
    TRAIN_ALPHAS,
    adapter_hook,
    avg,
    encode_example,
    finite,
    forward_logits,
    load_adapter_dataset,
    load_empirical_direction,
    load_model_tokenizer,
    non_answer_kl,
    option_metrics_from_logits,
    rel,
    rows_for_split,
    set_seed,
    tensor_rms,
    write_json,
    write_md,
)
from src.evaluator.bg_causal_adapter import LowRankDeltaAdapter
from src.evaluator.bg_steering_hook import BGLayerHookSteering, build_intervention_mode


OUT_JSON = OUT_ROOT / "teacher_forced_eval.json"
OUT_MD = OUT_ROOT / "teacher_forced_eval.md"
CKPT = OUT_ROOT / "adapter_checkpoints/best_adapter.pt"
HELDOUT_CAP = int(os.environ.get("BG_CAUSAL_ADAPTER_TF_HELDOUT_CAP", "32"))
METHODS = ["baseline", "RAW_NONORM_READOUT", "EMPIRICAL_MEAN_DIFF", "trained_adapter", "random_same_rms"]


def choose_rows(rows: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    if cap <= 0 or len(rows) <= cap:
        return list(rows)
    return list(rows[:cap])


def rms_direction(vec: torch.Tensor) -> torch.Tensor:
    flat = vec.detach().flatten().to(dtype=torch.float32, device="cpu")
    return flat / flat.pow(2).mean().sqrt().clamp(min=1e-8)


def random_rms_direction(seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    return rms_direction(torch.randn(HIDDEN_DIM, generator=gen))


def eval_once(
    model: Any,
    tokenizer: Any,
    adapter: torch.nn.Module,
    example: dict[str, Any],
    device: torch.device,
    *,
    method: str,
    alpha: float,
    direction: torch.Tensor | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    encoded = encode_example(tokenizer, example, device)
    with torch.no_grad():
        baseline_logits = forward_logits(model, encoded["enc"], logits_to_keep=33)
        baseline_metrics = option_metrics_from_logits(baseline_logits, encoded)

    diagnostics: dict[str, Any] = {}
    if method == "baseline" or float(alpha) == 0.0:
        logits = baseline_logits
        diagnostics = {
            "hook_forward_call_count": 0,
            "hook_modifications": 0,
            "hook_loop_index_source": "not_applicable",
            "activation_rms_change": 0.0,
            "per_loop_activation_rms_change": {},
            "nan_or_inf_activations": False,
        }
    elif method == "trained_adapter":
        with adapter_hook(
            model,
            adapter,
            alpha=float(alpha),
            intervention_mode=PRIMARY_MODE,
            position=int(encoded["intervention_token_index"]),
        ) as hook:
            with torch.no_grad():
                logits = forward_logits(model, encoded["enc"], logits_to_keep=33)
            diagnostics = hook.diagnostics()
    else:
        if direction is None:
            raise ValueError(f"static method {method} requires direction")
        spec = build_intervention_mode(PRIMARY_MODE, float(alpha))
        hook = BGLayerHookSteering(
            model,
            target_layer=36,
            target_loops=spec["target_loops"],
            direction=direction,
            alpha=float(alpha),
            loop_alpha_scales=spec["loop_alpha_scales"],
            max_rms_fraction=0.02,
            direction_normalization="rms",
        )
        hook.apply(position=int(encoded["intervention_token_index"]))
        try:
            with torch.no_grad():
                logits = forward_logits(model, encoded["enc"], logits_to_keep=33)
        finally:
            hook.remove()
        raw_diag = hook.diagnostics()
        diagnostics = {
            "hook_forward_call_count": raw_diag.get("forward_call_count", 0),
            "hook_modifications": raw_diag.get("modifications", 0),
            "hook_loop_index_source": raw_diag.get("loop_index_source", ""),
            "activation_rms_change": raw_diag.get("activation_rms_change", 0.0),
            "per_loop_activation_rms_change": raw_diag.get("per_loop_activation_rms_change", {}),
            "nan_or_inf_activations": raw_diag.get("nan_or_inf", False),
        }

    metrics = option_metrics_from_logits(logits, encoded)
    kl = non_answer_kl(baseline_logits, logits, window=16)
    return {
        "example_id": example["example_id"],
        "task_id": example["task_id"],
        "domain": example["domain"],
        "method": method,
        "mode": PRIMARY_MODE,
        "alpha": float(alpha),
        "direction_source": method,
        "random_seed": int(seed) if method == "random_same_rms" else None,
        "correct_option": example["correct_option"],
        "predicted_option": metrics["predicted_option"],
        "baseline_predicted_option": baseline_metrics["predicted_option"],
        "correct_option_logit_margin": float(metrics["margin"].detach().cpu().item()),
        "baseline_correct_option_logit_margin": float(baseline_metrics["margin"].detach().cpu().item()),
        "margin_lift_over_baseline": float((metrics["margin"] - baseline_metrics["margin"]).detach().cpu().item()),
        "option_ce": float(metrics["ce"].detach().cpu().item()),
        "baseline_option_ce": float(baseline_metrics["ce"].detach().cpu().item()),
        "option_accuracy": float(metrics["accuracy"]),
        "baseline_option_accuracy": float(baseline_metrics["accuracy"]),
        "KL_ANSWER_POSITION_MASKED": True,
        "kl_non_answer": float(kl.detach().cpu().item()),
        "delta_rms": float(diagnostics.get("activation_rms_change", 0.0)),
        "BG_target_score_shift": None,
        "objective_mixed_score_shift": None,
        "D1_diagnostic_score_shift": None,
        "BG_DIAGNOSTICS_COMPUTED": False,
        "intervention_position_kind": encoded["intervention_position_kind"],
        "intervention_token_index": int(encoded["intervention_token_index"]),
        "answer_logit_token_index": int(encoded["answer_logit_token_index"]),
        "INTERVENTION_POSITION_WARNING": bool(encoded["INTERVENTION_POSITION_WARNING"]),
        **diagnostics,
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
            "margin_mean": avg(row["correct_option_logit_margin"] for row in vals),
            "baseline_margin_mean": avg(row["baseline_correct_option_logit_margin"] for row in vals),
            "margin_lift_mean": avg(row["margin_lift_over_baseline"] for row in vals),
            "ce_mean": avg(row["option_ce"] for row in vals),
            "accuracy_mean": avg(row["option_accuracy"] for row in vals),
            "baseline_accuracy_mean": avg(row["baseline_option_accuracy"] for row in vals),
            "kl_non_answer_mean": avg(row["kl_non_answer"] for row in vals),
            "delta_rms_mean": avg(row["delta_rms"] for row in vals),
            "nan_or_inf_count": sum(1 for row in vals if row.get("nan_or_inf_activations")),
        }
    return out


def main() -> int:
    started = time.time()
    set_seed()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    dataset = load_adapter_dataset()
    heldout = choose_rows(rows_for_split(dataset, "heldout"), HELDOUT_CAP)
    if not CKPT.exists() or not heldout:
        payload = {
            "BG_CAUSAL_ADAPTER_TEACHER_FORCED_VERDICT": "INSUFFICIENT",
            "blocker": f"checkpoint_exists={CKPT.exists()} heldout_rows={len(heldout)}",
            "rows": [],
            "elapsed_seconds": round(time.time() - started, 3),
        }
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# BG Causal Adapter Teacher-Forced Evaluation", "", "BG_CAUSAL_ADAPTER_TEACHER_FORCED_VERDICT = INSUFFICIENT", "", f"- blocker: `{payload['blocker']}`"])
        print("BG_CAUSAL_ADAPTER_TEACHER_FORCED_VERDICT = INSUFFICIENT")
        return 1

    model, tokenizer, device = load_model_tokenizer()
    adapter = LowRankDeltaAdapter(rank=32).to(device)
    checkpoint = torch.load(CKPT, map_location=device, weights_only=False)
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    adapter.eval()
    raw_direction = load_empirical_direction("RAW_NONORM_READOUT")
    mean_direction = load_empirical_direction("EMPIRICAL_MEAN_DIFF")
    directions = {
        "RAW_NONORM_READOUT": rms_direction(raw_direction) if raw_direction is not None else None,
        "EMPIRICAL_MEAN_DIFF": rms_direction(mean_direction) if mean_direction is not None else None,
    }

    rows: list[dict[str, Any]] = []
    for idx, example in enumerate(heldout):
        rows.append(eval_once(model, tokenizer, adapter, example, device, method="baseline", alpha=0.0))
        for alpha in TRAIN_ALPHAS:
            rows.append(eval_once(model, tokenizer, adapter, example, device, method="trained_adapter", alpha=alpha))
            for method in ["RAW_NONORM_READOUT", "EMPIRICAL_MEAN_DIFF"]:
                direction = directions.get(method)
                if direction is None:
                    continue
                rows.append(eval_once(model, tokenizer, adapter, example, device, method=method, alpha=alpha, direction=direction))
            seed = 20260518 + idx * 100 + int(alpha * 10000)
            rows.append(
                eval_once(
                    model,
                    tokenizer,
                    adapter,
                    example,
                    device,
                    method="random_same_rms",
                    alpha=alpha,
                    direction=random_rms_direction(seed),
                    seed=seed,
                )
            )
        write_json(
            OUT_JSON.with_suffix(".partial.json"),
            {
                "BG_CAUSAL_ADAPTER_TEACHER_FORCED_VERDICT": "PARTIAL",
                "rows": rows,
                "heldout_rows_used": len(heldout),
            },
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    agg = aggregate(rows)
    adapter_best = max(
        (v for v in agg.values() if v["method"] == "trained_adapter"),
        key=lambda row: finite(row.get("margin_lift_mean"), -1e9),
        default=None,
    )
    static_best = max(
        (v for v in agg.values() if v["method"] in {"RAW_NONORM_READOUT", "EMPIRICAL_MEAN_DIFF", "random_same_rms"}),
        key=lambda row: finite(row.get("margin_lift_mean"), -1e9),
        default=None,
    )
    if adapter_best is None:
        verdict = "INSUFFICIENT"
    elif finite(adapter_best.get("margin_lift_mean"), 0.0) > max(0.0, finite(static_best.get("margin_lift_mean") if static_best else None, -1e9)) + 1e-4:
        verdict = "ADAPTER_IMPROVES_LOGIT_MARGIN"
    elif finite(adapter_best.get("margin_lift_mean"), 0.0) > 0.0:
        verdict = "ADAPTER_NO_BETTER_THAN_STATIC"
    else:
        verdict = "ADAPTER_NO_EFFECT"

    payload = {
        "BG_CAUSAL_ADAPTER_TEACHER_FORCED_VERDICT": verdict,
        "teacher_forced_interpretation": "necessary_not_sufficient_free_generation_is_load_bearing",
        "heldout_rows_used": len(heldout),
        "rows": rows,
        "aggregate": agg,
        "best_adapter": adapter_best,
        "best_static_or_random": static_best,
        "KL_ANSWER_POSITION_MASKED": True,
        "intervention_position_kind": "prefix_last_token",
        "INTERVENTION_POSITION_WARNING": any(bool(row.get("INTERVENTION_POSITION_WARNING")) for row in rows),
        "BG_DIAGNOSTICS_COMPUTED": False,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    lines = [
        "# BG Causal Adapter Teacher-Forced Evaluation",
        "",
        f"BG_CAUSAL_ADAPTER_TEACHER_FORCED_VERDICT = {verdict}",
        "",
        "- Interpretation: teacher-forced margin lift is necessary but not sufficient; free-generation transfer remains load-bearing.",
        f"- heldout rows used: `{len(heldout)}`",
        f"- best adapter margin lift: `{finite(adapter_best.get('margin_lift_mean') if adapter_best else None):.6f}`",
        f"- best static/random margin lift: `{finite(static_best.get('margin_lift_mean') if static_best else None):.6f}`",
        f"- KL answer position masked: `true`",
        f"- intervention position kind: `prefix_last_token`",
        "",
        "| method | alpha | n | margin | margin lift | accuracy | KL | delta RMS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in agg.items():
        lines.append(
            f"| `{row['method']}` | {row['alpha']:.3f} | {row['n']} | {finite(row.get('margin_mean')):.6f} | "
            f"{finite(row.get('margin_lift_mean')):.6f} | {finite(row.get('accuracy_mean')):.3f} | "
            f"{finite(row.get('kl_non_answer_mean')):.6f} | {finite(row.get('delta_rms_mean')):.6f} |"
        )
    write_md(OUT_MD, lines)
    print(f"BG_CAUSAL_ADAPTER_TEACHER_FORCED_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
