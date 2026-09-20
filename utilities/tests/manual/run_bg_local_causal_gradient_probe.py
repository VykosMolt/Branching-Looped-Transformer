"""Local causal-gradient diagnostic for BG layer-hook steering.

This probe does not train Ouro or BG heads. It computes a task-local gradient
with respect to a synthetic delta at the layer-36 intervention site, then tests
that direction through the existing layer-hook mechanism.
"""
from __future__ import annotations

import time
import traceback
from collections import defaultdict
from typing import Any

import torch

from bg_preconsolidation_common import (
    OUT_ROOT,
    POST_INTERVENTION_TOKENS,
    SEED,
    cosine,
    finite,
    random_rms_direction,
    rel,
    rms_unit,
    select_targets,
    target_directions,
    task_subset_for_target,
    unit,
    write_json,
    write_md,
)
from bg_stage2_steering_preflight import head_weight_vector, load_head_row
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


OUT_JSON = OUT_ROOT / "causal_gradient_probe.json"
OUT_MD = OUT_ROOT / "causal_gradient_probe.md"
RMS_ANALYSIS_JSON = OUT_ROOT / "rms_steering_analysis.json"
PROP_ANALYSIS_JSON = OUT_ROOT / "propagation_decay_analysis.json"
MAX_TASKS = 3
ALPHAS = [0.005, 0.01]
MODE = "multi_loop_decayed"


def load_json(path, default=None):
    import json

    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _output_tensor(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, (tuple, list)) else output


def _pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    h = hidden.squeeze(0).float()
    m = attention_mask.squeeze(0).float().unsqueeze(-1)
    return (h * m).sum(dim=0) / m.sum(dim=0).clamp(min=1.0)


def _score_from_layer36(
    loop_states: dict[int, torch.Tensor],
    attention_mask: torch.Tensor,
    head_row: dict[str, Any],
) -> torch.Tensor:
    config = str(head_row.get("config"))
    if not config.startswith("36_"):
        raise ValueError(f"gradient objective requires a layer-36 head, got config={config}")
    pooled = {loop: _pool(tensor, attention_mask) for loop, tensor in loop_states.items()}
    if config == "36_L1":
        vec = pooled[1]
    elif config == "36_L4":
        vec = pooled[4]
    elif config == "36_mean":
        vec = torch.stack([pooled[i] for i in range(1, 5)], dim=0).mean(dim=0)
    else:
        raise ValueError(f"unsupported layer-36 gradient objective config={config}")
    weight = head_weight_vector(head_row).to(device=vec.device, dtype=vec.dtype)
    return torch.dot(vec.flatten(), weight.flatten())


def local_gradient_direction(
    extractor: BGTransformerFeatureExtractor,
    prompt_text: str,
    candidate_text: str,
    head_row: dict[str, Any],
    *,
    base_pos: int,
) -> dict[str, Any]:
    model = extractor.model
    tokenizer = extractor.tokenizer
    device = extractor.device
    enc = tokenizer(
        format_prompt_candidate(prompt_text, candidate_text),
        return_tensors="pt",
        truncation=True,
        max_length=1536,
        padding=False,
    )
    enc = {key: value.to(device) for key, value in enc.items()}
    if "attention_mask" not in enc:
        enc["attention_mask"] = torch.ones_like(enc["input_ids"], device=device)
    seq_len = int(enc["input_ids"].shape[1])
    pos = max(0, min(int(base_pos), seq_len - 1))
    delta = torch.zeros((1, 1, 2048), device=device, dtype=torch.float32, requires_grad=True)
    layer_states: dict[int, torch.Tensor] = {}
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise RuntimeError("model does not expose model.layers")

    def hook(_module: Any, _args: Any, kwargs: dict[str, Any] | None, output: Any) -> Any:
        loop = int(kwargs.get("current_ut", 0)) + 1 if kwargs and "current_ut" in kwargs else len(layer_states) + 1
        tensor = _output_tensor(output)
        changed = tensor.clone()
        if loop == 1:
            changed[:, pos : pos + 1, :] = changed[:, pos : pos + 1, :] + delta.to(device=changed.device, dtype=changed.dtype)
        layer_states[loop] = changed
        if isinstance(output, tuple):
            return (changed,) + output[1:]
        if isinstance(output, list):
            return [changed] + list(output[1:])
        return changed

    try:
        handle = layers[35].register_forward_hook(hook, with_kwargs=True)
    except TypeError:
        handle = layers[35].register_forward_hook(lambda mod, args, out: hook(mod, args, None, out))
    try:
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            model(**enc, use_cache=False, return_dict=True)
            if len(layer_states) < 4:
                raise RuntimeError(f"expected four layer-36 loop states, got {len(layer_states)}")
            score = _score_from_layer36(layer_states, enc["attention_mask"], head_row)
            score.backward()
    finally:
        handle.remove()
    grad = delta.grad.detach().flatten().to(device="cpu", dtype=torch.float32)
    grad_norm = float(torch.linalg.vector_norm(grad).item())
    grad_rms = float(grad.pow(2).mean().sqrt().item())
    if not torch.isfinite(grad).all() or grad_norm <= 0.0 or grad_rms <= 0.0:
        raise RuntimeError("local gradient was zero or non-finite")
    return {
        "direction": rms_unit(grad),
        "gradient_norm": grad_norm,
        "gradient_rms": grad_rms,
        "gradient_objective_score": float(score.detach().cpu().item()),
        "intervention_abs_position": pos,
    }


def signed_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cond: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_cond[str(row.get("condition"))].append(finite(row.get("z_score_change")))
    pos = sum(by_cond["positive"]) / max(len(by_cond["positive"]), 1)
    neg = sum(by_cond["negative"]) / max(len(by_cond["negative"]), 1)
    rnd = sum(by_cond["random"]) / max(len(by_cond["random"]), 1)
    signed = bool(by_cond["positive"] and by_cond["negative"] and by_cond["random"] and pos > rnd and neg < rnd)
    return {
        "positive_z_mean": pos,
        "negative_z_mean": neg,
        "random_z_mean": rnd,
        "signed_causal_signature": signed,
        "signed_score": (pos - rnd) + (rnd - neg),
    }


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rms_analysis = load_json(RMS_ANALYSIS_JSON, {})
    prop_analysis = load_json(PROP_ANALYSIS_JSON, {})
    if rms_analysis.get("BG_RMS_STEERING_VERDICT") == "RMS_SIGNED_CAUSAL":
        payload = {
            "BG_CAUSAL_GRADIENT_VERDICT": "SKIPPED",
            "skipped_reason": "RMS-calibrated static steering already found signed causal effect",
            "rows": [],
            "elapsed_seconds": round(time.time() - started, 3),
        }
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# BG Local Causal-Gradient Probe", "", "BG_CAUSAL_GRADIENT_VERDICT = SKIPPED", "", f"- skipped_reason: `{payload['skipped_reason']}`"])
        print("BG_CAUSAL_GRADIENT_VERDICT = SKIPPED")
        return 0

    controller = BGController.from_artifacts(device="cpu")
    targets = select_targets()
    primary = next(row for row in targets if row["target_id"] == "T1")
    target_head = load_head_row(str(primary["head_id"]), controller)
    if not str(target_head.get("config")).startswith("36_"):
        target_head = load_head_row("locked::objective_mixed", controller)
    target_std = compute_head_std(target_head, str(primary["domain"]), int(primary["prefix_length"]))
    raw_direction = unit(next(row for row in target_directions(primary, controller) if row["direction_name"] == "RAW_NONORM_READOUT")["vector"])
    empirical = next((row for row in target_directions(primary, controller) if row["direction_name"] == "EMPIRICAL_MEAN_DIFF"), None)
    empirical_direction = unit(empirical["vector"]) if empirical else raw_direction
    tasks = task_subset_for_target(primary, MAX_TASKS)
    stage1_baselines = load_stage1_baseline_by_key()
    rows: list[dict[str, Any]] = []
    gradients: list[dict[str, Any]] = []
    extractor: BGTransformerFeatureExtractor | None = None
    try:
        extractor = BGTransformerFeatureExtractor(device="cuda" if torch.cuda.is_available() else "cpu", dtype="auto", force_all_loops=True)
        for task_idx, suite_row in enumerate(tasks):
            task = dict(suite_row.get("task") or {})
            prefix_text = str(suite_row.get("prefix_text") or "")
            prompt_text = str(task.get("prompt") or task.get("question") or "")
            branch_id = int(suite_row.get("branch_id", 0))
            baseline = stage1_baselines.get((str(suite_row["task_id"]), branch_id, int(primary["prefix_length"])), {})
            baseline_cont = str(baseline.get("continuation_text") or suite_row.get("stage1_continuation_text") or "")
            baseline_32 = first_n_tokens_text(extractor.tokenizer, baseline_cont, POST_INTERVENTION_TOKENS)
            prefix_wrapped = format_prompt_candidate(prompt_text, prefix_text)
            full_wrapped = format_prompt_candidate(prompt_text, prefix_text + baseline_32)
            prefix_len = int(extractor.tokenizer(prefix_wrapped, return_tensors="pt", add_special_tokens=True)["input_ids"].shape[1])
            full_len = int(extractor.tokenizer(full_wrapped, return_tensors="pt", add_special_tokens=True)["input_ids"].shape[1])
            base_pos = max(0, min(prefix_len - 1, full_len - 1))
            try:
                grad_info = local_gradient_direction(
                    extractor,
                    prompt_text,
                    prefix_text + baseline_32,
                    target_head,
                    base_pos=base_pos,
                )
            except Exception as exc:
                gradients.append(
                    {
                        "task_id": suite_row["task_id"],
                        "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:],
                    }
                )
                continue
            gradient_direction = grad_info["direction"]
            gradients.append(
                {
                    "task_id": suite_row["task_id"],
                    "gradient_norm": grad_info["gradient_norm"],
                    "gradient_rms": grad_info["gradient_rms"],
                    "gradient_objective_score": grad_info["gradient_objective_score"],
                    "cosine_grad_raw_nonorm": cosine(gradient_direction, raw_direction),
                    "cosine_grad_empirical_mean_diff": cosine(gradient_direction, empirical_direction),
                }
            )
            clean_candidate = prefix_text + baseline_32
            baseline_features = capture_bg_features_with_model(
                extractor.model,
                extractor.tokenizer,
                format_prompt_candidate(prompt_text, clean_candidate),
                device=extractor.device,
            )
            baseline_score = raw_head_score(target_head, baseline_features)
            for alpha in ALPHAS:
                for condition, direction in [
                    ("positive", gradient_direction),
                    ("negative", -gradient_direction),
                    ("random", random_rms_direction(2048, SEED + task_idx * 10_000 + int(alpha * 10000))),
                ]:
                    mode_spec = build_intervention_mode(MODE, alpha)
                    hook = BGLayerHookSteering(
                        extractor.model,
                        target_layer=36,
                        target_loops=mode_spec["target_loops"],
                        loop_alpha_scales=mode_spec["loop_alpha_scales"],
                        direction=direction,
                        alpha=alpha,
                        max_rms_fraction=0.05,
                        direction_normalization="rms",
                    )
                    cuda_error = False
                    error = ""
                    try:
                        output_text, gen_meta = generate_with_layer_hook(
                            extractor,
                            continuation_prompt(task, prefix_text),
                            hook,
                            max_new_tokens=64,
                            seed=SEED + task_idx * 1000 + int(alpha * 10000),
                        )
                        candidate = prefix_text + first_n_tokens_text(extractor.tokenizer, output_text, POST_INTERVENTION_TOKENS)
                        post_features = capture_bg_features_with_model(
                            extractor.model,
                            extractor.tokenizer,
                            format_prompt_candidate(prompt_text, candidate),
                            device=extractor.device,
                        )
                        eval_result = evaluate_output(task, output_text)
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                        cuda_error = True
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        output_text = ""
                        gen_meta = {"token_count": 0, "hit_max_tokens": False}
                        post_features = baseline_features
                        eval_result = {"parsed": False, "success": False, "parsed_answer": None}
                        error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:]
                    diag = hook.diagnostics()
                    post_score = raw_head_score(target_head, post_features)
                    rows.append(
                        {
                            "task_id": suite_row["task_id"],
                            "alpha": alpha,
                            "condition": condition,
                            "mode": MODE,
                            "target_head": target_head.get("head_id"),
                            "target_head_config": target_head.get("config"),
                            "gradient_norm": grad_info["gradient_norm"],
                            "cosine_grad_raw_nonorm": gradients[-1]["cosine_grad_raw_nonorm"],
                            "cosine_grad_empirical_mean_diff": gradients[-1]["cosine_grad_empirical_mean_diff"],
                            "target_head_score_baseline": baseline_score,
                            "target_head_score_post": post_score,
                            "z_score_change": (post_score - baseline_score) / max(target_std, 1e-8),
                            "hook_forward_call_count": diag.get("forward_call_count"),
                            "hook_modifications": diag.get("modifications"),
                            "hook_loop_index_source": diag.get("loop_index_source"),
                            "cache_intervention_mode": "disabled",
                            "activation_rms_change": diag.get("activation_rms_change"),
                            "per_loop_activation_rms_change": diag.get("per_loop_activation_rms_change"),
                            "nan_or_inf_activations": bool(diag.get("nan_or_inf")),
                            "cuda_error": cuda_error,
                            "safety_status": "DESTABILIZING" if cuda_error or diag.get("nan_or_inf") else "OK",
                            "output_text": output_text,
                            "parsed_answer": eval_result.get("parsed_answer"),
                            "is_correct": bool(eval_result.get("success")),
                            "parse_failed": not bool(eval_result.get("parsed", False)),
                            "parse_rate": parse_rate(eval_result),
                            "repetition_rate": repetition_rate(output_text),
                            "output_length": int(gen_meta.get("token_count", 0)),
                            "hit_max_tokens": bool(gen_meta.get("hit_max_tokens", False)),
                            "error": error,
                        }
                    )
            write_json(
                OUT_JSON,
                {
                    "BG_CAUSAL_GRADIENT_VERDICT": "PARTIAL",
                    "rows": rows,
                    "gradient_records": gradients,
                    "elapsed_seconds": round(time.time() - started, 3),
                },
            )
    finally:
        if extractor is not None:
            extractor.cleanup()

    summary = signed_summary(rows)
    if not rows:
        verdict = "INSUFFICIENT"
    elif any(row.get("safety_status") == "DESTABILIZING" for row in rows):
        verdict = "GRADIENT_DESTABILIZING"
    elif summary["signed_causal_signature"]:
        verdict = "LOCAL_CAUSAL_DIRECTION_EXISTS"
    else:
        verdict = "GRADIENT_NO_BETTER_THAN_RANDOM"
    payload = {
        "BG_CAUSAL_GRADIENT_VERDICT": verdict,
        "static_context": {
            "BG_RMS_STEERING_VERDICT": rms_analysis.get("BG_RMS_STEERING_VERDICT"),
            "BG_PROPAGATION_VERDICT": prop_analysis.get("BG_PROPAGATION_VERDICT"),
        },
        "gradient_records": gradients,
        "summary": summary,
        "rows": rows,
        "row_count": len(rows),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    lines = [
        "# BG Local Causal-Gradient Probe",
        "",
        f"BG_CAUSAL_GRADIENT_VERDICT = {verdict}",
        "",
        f"- row_count: `{len(rows)}`",
        f"- positive_z_mean: `{summary.get('positive_z_mean')}`",
        f"- negative_z_mean: `{summary.get('negative_z_mean')}`",
        f"- random_z_mean: `{summary.get('random_z_mean')}`",
        f"- signed_causal_signature: `{summary.get('signed_causal_signature')}`",
        "",
        "| task | grad_norm | cos raw | cos empirical |",
        "|---|---:|---:|---:|",
    ]
    for row in gradients:
        lines.append(
            f"| `{row.get('task_id')}` | {finite(row.get('gradient_norm')):.6f} | "
            f"{finite(row.get('cosine_grad_raw_nonorm')):.4f} | {finite(row.get('cosine_grad_empirical_mean_diff')):.4f} |"
        )
    write_md(OUT_MD, lines)
    print(f"BG_CAUSAL_GRADIENT_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict in {"LOCAL_CAUSAL_DIRECTION_EXISTS", "GRADIENT_NO_BETTER_THAN_RANDOM", "SKIPPED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
