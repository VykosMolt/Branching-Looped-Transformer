"""Tiny frozen-feature steering adapter diagnostic for BG empirical steering."""
from __future__ import annotations

import json
import math
import time
import traceback
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn

from bg_empirical_steering_common import (
    OUT_ROOT,
    SEED,
    auc_score,
    examples_for_target,
    random_direction,
    rel,
    stack_xy,
    summarize_cell,
    unit,
    write_json,
    write_md,
)
from bg_stage2_steering_preflight import load_head_row
from bg_steering_suite_lib import evaluate_output
from bg_trajectory_prediction_lib import continuation_prompt
from run_bg_stage2_intervention_sweep import (
    compute_head_std,
    first_n_tokens_text,
    generate_with_layer_hook,
    load_stage1_baseline_by_key,
    parse_rate,
    raw_head_score,
    repetition_rate,
)
from src.evaluator.bg_controller import BGController
from src.evaluator.bg_steering_hook import BGLayerHookSteering, build_intervention_mode, capture_bg_features_with_model
from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor, format_prompt_candidate


OUT_JSON = OUT_ROOT / "tiny_adapter_diagnostic.json"
OUT_MD = OUT_ROOT / "tiny_adapter_diagnostic.md"
DIRECTIONS_PT = OUT_ROOT / "directions.pt"
TASKS_JSON = OUT_ROOT / "task_subset.json"
ANALYSIS_JSON = OUT_ROOT / "analysis.json"
MAX_TASKS = 3
MAX_NEW_TOKENS = 128
POST_INTERVENTION_TOKENS = 32
MODES = ["multi_loop_decayed", "single_loop_L1"]


class LowRankDirectionAdapter(nn.Module):
    def __init__(self, dim: int = 2048, rank: int = 128) -> None:
        super().__init__()
        self.down = nn.Linear(dim, rank)
        self.up = nn.Linear(rank, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(torch.tanh(self.down(x)))


def load_json(path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def condition_plan() -> list[dict[str, Any]]:
    return [
        {"alpha": 0.0, "condition": "zero_baseline", "random_control_idx": 0, "random_control_n": 0},
        {"alpha": 0.01, "condition": "positive", "random_control_idx": 0, "random_control_n": 1},
        {"alpha": 0.01, "condition": "negative", "random_control_idx": 0, "random_control_n": 1},
        {"alpha": 0.01, "condition": "random", "random_control_idx": 0, "random_control_n": 1},
        {"alpha": 0.02, "condition": "positive", "random_control_idx": 0, "random_control_n": 1},
        {"alpha": 0.02, "condition": "negative", "random_control_idx": 0, "random_control_n": 1},
        {"alpha": 0.02, "condition": "random", "random_control_idx": 0, "random_control_n": 1},
    ]


def train_adapter(x_train: torch.Tensor, y_train: torch.Tensor, x_heldout: torch.Tensor, y_heldout: torch.Tensor) -> dict[str, Any]:
    torch.manual_seed(SEED + 777)
    mu = x_train.mean(dim=0)
    sigma = x_train.std(dim=0, unbiased=False).clamp(min=1e-4)
    zx = (x_train - mu) / sigma
    zh = (x_heldout - mu) / sigma
    model = LowRankDirectionAdapter(dim=int(x_train.shape[1]), rank=128)
    param_count = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    best = {"auc": -1.0, "state": None, "epoch": 0}
    for epoch in range(20):
        opt.zero_grad(set_to_none=True)
        delta = model(zx)
        delta = delta / delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        logits = (zx * delta).sum(dim=-1) / math.sqrt(float(x_train.shape[1]))
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y_train)
        loss.backward()
        opt.step()
        with torch.no_grad():
            hdelta = model(zh)
            hdelta = hdelta / hdelta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            hscores = (zh * hdelta).sum(dim=-1) / math.sqrt(float(x_train.shape[1]))
            auc = auc_score(y_heldout, hscores)
            if auc is not None and auc > best["auc"]:
                best = {"auc": float(auc), "state": {k: v.detach().clone() for k, v in model.state_dict().items()}, "epoch": epoch}
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return {"model": model, "mu": mu, "sigma": sigma, "heldout_auc": best["auc"], "best_epoch": best["epoch"], "param_count": param_count}


def adapter_direction(adapter: dict[str, Any], x: torch.Tensor) -> torch.Tensor:
    z = (x.flatten().to(dtype=torch.float32) - adapter["mu"]) / adapter["sigma"]
    with torch.no_grad():
        delta = adapter["model"](z.unsqueeze(0)).squeeze(0)
    return unit(delta)


def baseline_features(extractor: BGTransformerFeatureExtractor, task: dict[str, Any], prefix_text: str, baseline_continuation: str) -> torch.Tensor:
    continuation_32 = first_n_tokens_text(extractor.tokenizer, baseline_continuation, POST_INTERVENTION_TOKENS)
    candidate = str(prefix_text or "") + continuation_32
    return capture_bg_features_with_model(
        extractor.model,
        extractor.tokenizer,
        format_prompt_candidate(str(task.get("prompt") or task.get("question") or ""), candidate),
        device=extractor.device,
    )


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    analysis = load_json(ANALYSIS_JSON, {})
    if analysis.get("BG_EMPIRICAL_DIRECTION_CAUSAL_VERDICT") == "EMPIRICAL_SIGNED_CAUSAL":
        payload = {"BG_TINY_STEERING_ADAPTER_VERDICT": "SKIPPED", "reason": "empirical signed causal effect already detected"}
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# Tiny Steering Adapter Diagnostic", "", "BG_TINY_STEERING_ADAPTER_VERDICT = SKIPPED"])
        print("BG_TINY_STEERING_ADAPTER_VERDICT = SKIPPED")
        return 0
    if not DIRECTIONS_PT.exists():
        payload = {"BG_TINY_STEERING_ADAPTER_VERDICT": "INSUFFICIENT", "reason": "directions.pt missing"}
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# Tiny Steering Adapter Diagnostic", "", "BG_TINY_STEERING_ADAPTER_VERDICT = INSUFFICIENT"])
        print("BG_TINY_STEERING_ADAPTER_VERDICT = INSUFFICIENT")
        return 0
    directions = torch.load(DIRECTIONS_PT, map_location="cpu", weights_only=False)
    target = next((row for row in directions.get("targets", []) if row.get("target_id") == "T1"), None)
    if not target:
        payload = {"BG_TINY_STEERING_ADAPTER_VERDICT": "INSUFFICIENT", "reason": "target missing"}
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# Tiny Steering Adapter Diagnostic", "", "BG_TINY_STEERING_ADAPTER_VERDICT = INSUFFICIENT"])
        print("BG_TINY_STEERING_ADAPTER_VERDICT = INSUFFICIENT")
        return 0

    examples = examples_for_target(str(target["domain"]), int(target["prefix_length"]), str(target["config"]))
    train_tasks = set(target.get("train_task_ids") or [])
    heldout_tasks = set(target.get("heldout_task_ids") or [])
    x_train, y_train, _ = stack_xy(examples, train_tasks)
    x_heldout, y_heldout, heldout_rows = stack_xy(examples, heldout_tasks)
    if int(y_train.sum().item()) < 10 or int((1.0 - y_train).sum().item()) < 10:
        payload = {"BG_TINY_STEERING_ADAPTER_VERDICT": "INSUFFICIENT", "reason": "not enough training contrast"}
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# Tiny Steering Adapter Diagnostic", "", "BG_TINY_STEERING_ADAPTER_VERDICT = INSUFFICIENT"])
        print("BG_TINY_STEERING_ADAPTER_VERDICT = INSUFFICIENT")
        return 0
    adapter = train_adapter(x_train, y_train, x_heldout, y_heldout)
    feature_by_key = {(row["task_id"], int(row["branch_id"])): row["x"] for row in heldout_rows}

    tasks_payload = load_json(TASKS_JSON, {})
    tasks = list(tasks_payload.get("tasks") or [])[:MAX_TASKS]
    controller = BGController.from_artifacts(device="cpu")
    target_head = load_head_row(str(target["head_id"]), controller)
    objective_head = load_head_row("locked::objective_mixed", controller)
    hh_head = load_head_row("locked::hh_general", controller)
    target_std = compute_head_std(target_head, str(target["domain"]), int(target["prefix_length"]))
    stage1_baselines = load_stage1_baseline_by_key()
    rows: list[dict[str, Any]] = []
    extractor: BGTransformerFeatureExtractor | None = None
    passes = 0
    try:
        extractor = BGTransformerFeatureExtractor(device="cuda" if torch.cuda.is_available() else "cpu", dtype="auto", force_all_loops=True)
        for task_idx, suite_row in enumerate(tasks):
            task = dict(suite_row.get("task") or {})
            prefix_text = str(suite_row.get("prefix_text") or "")
            branch_id = int(suite_row.get("branch_id", 0))
            x = feature_by_key.get((str(suite_row["task_id"]), branch_id))
            if x is None:
                continue
            base_direction = adapter_direction(adapter, x)
            baseline = stage1_baselines.get((str(suite_row["task_id"]), branch_id, int(target["prefix_length"])), {})
            baseline_continuation = str(baseline.get("continuation_text") or suite_row.get("stage1_continuation_text") or "")
            feats = baseline_features(extractor, task, prefix_text, baseline_continuation)
            baseline_eval = baseline.get("evaluation") or evaluate_output(task, baseline_continuation)
            base_scores = {
                "target": raw_head_score(target_head, feats),
                "objective_mixed": raw_head_score(objective_head, feats),
                "hh_general": raw_head_score(hh_head, feats),
            }
            base_stability = {
                "repetition_rate": repetition_rate(baseline_continuation),
                "empty_output": not bool(baseline_continuation.strip()),
                "output_length": int(baseline.get("token_count", 0) or 0),
                "hit_max_tokens": bool(baseline.get("hit_max_tokens", False)),
            }
            for mode_idx, mode in enumerate(MODES):
                for plan in condition_plan():
                    alpha = float(plan["alpha"])
                    condition = str(plan["condition"])
                    random_idx = int(plan["random_control_idx"])
                    base_row = {
                        "task_subset_index": suite_row.get("suite_index"),
                        "task_id": suite_row["task_id"],
                        "target": "T1",
                        "direction_name": "TINY_LINEAR_ADAPTER",
                        "direction_source": "low_rank_frozen_feature_adapter",
                        "direction_heldout_auc": adapter["heldout_auc"],
                        "mode": mode,
                        "alpha": alpha,
                        "condition": condition,
                        "random_control_idx": random_idx,
                        "random_control_n": int(plan["random_control_n"]),
                        "target_layer": int(target["layer"]),
                        "target_head_score_baseline": base_scores["target"],
                        "objective_mixed_score_baseline": base_scores["objective_mixed"],
                        "hh_general_score_baseline": base_scores["hh_general"],
                        "target_baseline_std": target_std,
                        "cache_intervention_mode": "disabled",
                    }
                    if condition == "zero_baseline":
                        rows.append(
                            {
                                **base_row,
                                "direction_norm": 0.0,
                                "target_head_score_post": base_scores["target"],
                                "z_score_change": 0.0,
                                "objective_mixed_score": base_scores["objective_mixed"],
                                "hh_general_score": base_scores["hh_general"],
                                "hook_forward_call_count": 0,
                                "hook_modifications": 0,
                                "hook_loop_index_source": "baseline_reused",
                                "activation_rms_change": 0.0,
                                "per_loop_activation_rms_change": {},
                                "nan_or_inf_activations": False,
                                "cuda_error": False,
                                "safety_status": "OK",
                                "output_text": baseline_continuation,
                                "parsed_answer": baseline_eval.get("parsed_answer"),
                                "is_correct": bool(baseline_eval.get("success")),
                                "parse_failed": not bool(baseline_eval.get("parsed", False)),
                                "repetition_rate": base_stability["repetition_rate"],
                                "output_length": base_stability["output_length"],
                                "hit_max_tokens": base_stability["hit_max_tokens"],
                                "empty_output": base_stability["empty_output"],
                            }
                        )
                        continue
                    if condition == "positive":
                        direction = base_direction
                    elif condition == "negative":
                        direction = -base_direction
                    else:
                        direction = random_direction(int(base_direction.numel()), SEED + 9_000_000 + mode_idx * 10_000 + task_idx * 100 + random_idx)
                    mode_spec = build_intervention_mode(mode, alpha)
                    hook = BGLayerHookSteering(
                        extractor.model,
                        target_layer=int(target["layer"]),
                        target_loops=mode_spec["target_loops"],
                        loop_alpha_scales=mode_spec["loop_alpha_scales"],
                        direction=direction,
                        alpha=alpha,
                    )
                    cuda_error = False
                    error_text = ""
                    try:
                        output_text, gen_meta = generate_with_layer_hook(
                            extractor,
                            continuation_prompt(task, prefix_text),
                            hook,
                            max_new_tokens=MAX_NEW_TOKENS,
                            seed=SEED + 8_000_000 + mode_idx * 10_000 + task_idx * 100 + int(alpha * 10000),
                        )
                        post_candidate = prefix_text + first_n_tokens_text(extractor.tokenizer, output_text, POST_INTERVENTION_TOKENS)
                        post_features = capture_bg_features_with_model(
                            extractor.model,
                            extractor.tokenizer,
                            format_prompt_candidate(str(task.get("prompt") or task.get("question") or ""), post_candidate),
                            device=extractor.device,
                        )
                        eval_result = evaluate_output(task, output_text)
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                        cuda_error = True
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        output_text = ""
                        gen_meta = {"token_count": 0, "hit_max_tokens": False, "seconds": 0.0}
                        post_features = feats
                        eval_result = {"parsed": False, "success": False, "parsed_answer": None}
                        error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:]
                    diag = hook.diagnostics()
                    target_post = raw_head_score(target_head, post_features)
                    objective_post = raw_head_score(objective_head, post_features)
                    hh_post = raw_head_score(hh_head, post_features)
                    row = {
                        **base_row,
                        "direction_norm": float(torch.linalg.vector_norm(direction).item()),
                        "target_head_score_post": target_post,
                        "z_score_change": (target_post - base_scores["target"]) / max(target_std, 1e-8),
                        "objective_mixed_score": objective_post,
                        "hh_general_score": hh_post,
                        "hook_forward_call_count": diag.get("forward_call_count"),
                        "hook_modifications": diag.get("modifications"),
                        "hook_loop_index_source": diag.get("loop_index_source"),
                        "activation_rms_change": diag.get("activation_rms_change", 0.0),
                        "per_loop_activation_rms_change": diag.get("per_loop_activation_rms_change", {}),
                        "nan_or_inf_activations": bool(diag.get("nan_or_inf", False)),
                        "cuda_error": cuda_error,
                        "output_text": output_text,
                        "parsed_answer": eval_result.get("parsed_answer"),
                        "is_correct": bool(eval_result.get("success")),
                        "parse_failed": not bool(eval_result.get("parsed", False)),
                        "repetition_rate": repetition_rate(output_text),
                        "output_length": int(gen_meta.get("token_count", 0)),
                        "hit_max_tokens": bool(gen_meta.get("hit_max_tokens", False)),
                        "empty_output": not bool(str(output_text).strip()),
                        "generation_metadata": gen_meta,
                        "error": error_text,
                    }
                    row["safety_status"] = "DESTABILIZING" if row["cuda_error"] or row["nan_or_inf_activations"] or row["empty_output"] else "OK"
                    rows.append(row)
                    passes += 1
    finally:
        if extractor is not None:
            extractor.cleanup()

    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("mode")), float(row.get("alpha", 0.0)))].append(row)
    cells = {f"{mode}|{alpha:g}": summarize_cell(group) for (mode, alpha), group in sorted(grouped.items())}
    strong = [key for key, value in cells.items() if value.get("strong_signed_causal_signature")]
    if strong:
        verdict = "PROMISING"
    elif rows and not any(row.get("safety_status") == "DESTABILIZING" for row in rows):
        verdict = "NO_BETTER_THAN_STATIC"
    else:
        verdict = "INSUFFICIENT"
    payload = {
        "BG_TINY_STEERING_ADAPTER_VERDICT": verdict,
        "param_count": adapter["param_count"],
        "epochs": 20,
        "heldout_auc": adapter["heldout_auc"],
        "best_epoch": adapter["best_epoch"],
        "rows": rows,
        "row_count": len(rows),
        "intervention_forward_passes": passes,
        "cells": cells,
        "strong_signed_cells": strong,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    lines = [
        "# Tiny Steering Adapter Diagnostic",
        "",
        f"BG_TINY_STEERING_ADAPTER_VERDICT = {verdict}",
        "",
        f"- param_count: `{adapter['param_count']}`",
        f"- heldout_auc: `{adapter['heldout_auc']}`",
        f"- row_count: `{len(rows)}`",
        f"- intervention_forward_passes: `{passes}`",
        "",
        "| mode | alpha | pos_z | neg_z | rand_z | strong |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for key, value in sorted(cells.items()):
        mode, alpha = key.split("|")
        lines.append(
            f"| `{mode}` | {alpha} | {float(value.get('positive_z_mean') or 0):.4f} | "
            f"{float(value.get('negative_z_mean') or 0):.4f} | {float(value.get('random_z_mean') or 0):.4f} | "
            f"`{value.get('strong_signed_causal_signature')}` |"
        )
    write_md(OUT_MD, lines)
    print(f"BG_TINY_STEERING_ADAPTER_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
