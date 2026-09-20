"""Map propagation and decay of BG layer-hook steering perturbations."""
from __future__ import annotations

import math
import time
import traceback
from contextlib import contextmanager
from typing import Any, Iterator

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
from bg_stage2_steering_preflight import load_head_row
from bg_trajectory_prediction_lib import continuation_prompt
from run_bg_stage2_intervention_sweep import (
    compute_head_std,
    first_n_tokens_text,
    generate_with_layer_hook,
    load_stage1_baseline_by_key,
    raw_head_score,
    repetition_rate,
)
from src.evaluator.bg_controller import BGController
from src.evaluator.bg_steering_hook import BGLayerHookSteering, build_intervention_mode
from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor, format_prompt_candidate


OUT_TRACES = OUT_ROOT / "propagation_decay_traces.json"
OUT_ANALYSIS_JSON = OUT_ROOT / "propagation_decay_analysis.json"
OUT_ANALYSIS_MD = OUT_ROOT / "propagation_decay_analysis.md"
RMS_ANALYSIS_JSON = OUT_ROOT / "rms_steering_analysis.json"
MAX_TASKS = 4
OFFSETS = [0, 1, 8, 32]


def load_json(path, default=None):
    import json

    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _output_tensor(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, (tuple, list)) else output


def _pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    h = hidden.squeeze(0).to(dtype=torch.float32)
    m = attention_mask.squeeze(0).to(dtype=torch.float32).unsqueeze(-1)
    return (h * m).sum(dim=0) / m.sum(dim=0).clamp(min=1.0)


@contextmanager
def capture_selected_states(model: Any, positions: list[int], store: dict[str, Any]) -> Iterator[None]:
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if inner is None or layers is None:
        raise RuntimeError("model does not expose model.layers")
    pos = [int(p) for p in positions]

    def layer_hook(_module: Any, _args: Any, kwargs: dict[str, Any] | None, output: Any) -> None:
        loop = int(kwargs.get("current_ut", 0)) + 1 if kwargs and "current_ut" in kwargs else 1
        tensor = _output_tensor(output)
        values = {}
        for offset_idx, abs_pos in enumerate(pos):
            if 0 <= abs_pos < int(tensor.shape[1]):
                values[str(offset_idx)] = tensor[:, abs_pos, :].detach().cpu()
        store.setdefault("36", {})[str(loop)] = values

    try:
        handle = layers[35].register_forward_hook(layer_hook, with_kwargs=True)
    except TypeError:
        handle = layers[35].register_forward_hook(lambda mod, args, out: layer_hook(mod, args, None, out))
    try:
        yield
    finally:
        handle.remove()


def forward_capture(
    extractor: BGTransformerFeatureExtractor,
    text: str,
    *,
    positions: list[int],
    steering_hook: BGLayerHookSteering | None = None,
    relative_position: int = -1,
) -> dict[str, Any]:
    model = extractor.model
    tokenizer = extractor.tokenizer
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=1536, padding=False)
    enc = {k: v.to(extractor.device) for k, v in enc.items()}
    if "attention_mask" not in enc:
        enc["attention_mask"] = torch.ones_like(enc["input_ids"], device=extractor.device)
    store: dict[str, Any] = {}
    if steering_hook is not None:
        steering_hook.apply(position=relative_position)
    try:
        with torch.inference_mode():
            with capture_selected_states(model, positions, store):
                output = model(**enc, use_cache=False, return_per_loop_hidden_states=True, logits_to_keep=1)
    finally:
        if steering_hook is not None:
            steering_hook.remove()
    boundary = list(getattr(output, "per_loop_hidden_states", None) or [])
    store["47"] = {}
    for loop_idx, tensor in enumerate(boundary, start=1):
        values = {}
        for offset_idx, abs_pos in enumerate(positions):
            if 0 <= abs_pos < int(tensor.shape[1]):
                values[str(offset_idx)] = tensor[:, abs_pos, :].detach().cpu()
        store["47"][str(loop_idx)] = values
    pooled36 = []
    for loop in range(1, 5):
        # Reconstruct pooled layer-36 from selected states only is not valid, so
        # layer-score shifts are computed from offset vectors below.  The full
        # BG feature score for the fixed text is not used as a headline metric.
        pooled36.append(torch.zeros(2048))
    return {
        "states": store,
        "logits": output.logits.detach().cpu(),
        "seq_len": int(enc["input_ids"].shape[1]),
        "positions": positions,
        "pooled36_placeholder": torch.stack(pooled36),
    }


def token_divergence(tokenizer: Any, left: str, right: str) -> int | None:
    a = tokenizer(left, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    b = tokenizer(right, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    n = min(int(a.numel()), int(b.numel()))
    for idx in range(n):
        if int(a[idx]) != int(b[idx]):
            return idx
    if int(a.numel()) != int(b.numel()):
        return n
    return None


def kl_divergence(left_logits: torch.Tensor, right_logits: torch.Tensor) -> float:
    l = left_logits.float().flatten()
    r = right_logits.float().flatten()
    lp = torch.log_softmax(l, dim=-1)
    rp = torch.log_softmax(r, dim=-1)
    p = torch.softmax(l, dim=-1)
    return float((p * (lp - rp)).sum().item())


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    controller = BGController.from_artifacts(device="cpu")
    targets = select_targets()
    primary = next(row for row in targets if row["target_id"] == "T1")
    directions = target_directions(primary, controller)
    by_name = {row["direction_name"]: row for row in directions}
    rms_analysis = load_json(RMS_ANALYSIS_JSON, {})
    best_key = str(rms_analysis.get("best_rms_key") or "")
    selected_names: list[str]
    if rms_analysis.get("BG_RMS_STEERING_VERDICT") == "RMS_SIGNED_CAUSAL" and best_key:
        selected_names = [best_key.split("|")[1]]
    else:
        selected_names = [name for name in ["EMPIRICAL_MEAN_DIFF", "RAW_NONORM_READOUT"] if name in by_name]
    selected_names = selected_names[:2] or [next(iter(by_name))]
    alphas = [0.01]
    if rms_analysis.get("BG_RMS_STABILITY_VERDICT") in {"STABLE", "STABLE_BUT_NOISY"}:
        alphas.append(0.02)
    tasks = task_subset_for_target(primary, MAX_TASKS)
    target_head = load_head_row(str(primary["head_id"]), controller)
    target_std = compute_head_std(target_head, str(primary["domain"]), int(primary["prefix_length"]))
    stage1_baselines = load_stage1_baseline_by_key()

    traces: list[dict[str, Any]] = []
    extractor: BGTransformerFeatureExtractor | None = None
    try:
        extractor = BGTransformerFeatureExtractor(device="cuda" if torch.cuda.is_available() else "cpu", dtype="auto", force_all_loops=True)
        for task_idx, suite_row in enumerate(tasks):
            task = dict(suite_row.get("task") or {})
            prefix_text = str(suite_row.get("prefix_text") or "")
            branch_id = int(suite_row.get("branch_id", 0))
            baseline = stage1_baselines.get((str(suite_row["task_id"]), branch_id, int(primary["prefix_length"])), {})
            baseline_cont = str(baseline.get("continuation_text") or suite_row.get("stage1_continuation_text") or "")
            baseline_32 = first_n_tokens_text(extractor.tokenizer, baseline_cont, POST_INTERVENTION_TOKENS)
            prompt_text = str(task.get("prompt") or task.get("question") or "")
            prefix_wrapped = format_prompt_candidate(prompt_text, prefix_text)
            full_wrapped = format_prompt_candidate(prompt_text, prefix_text + baseline_32)
            prefix_len = int(extractor.tokenizer(prefix_wrapped, return_tensors="pt", add_special_tokens=True)["input_ids"].shape[1])
            full_len = int(extractor.tokenizer(full_wrapped, return_tensors="pt", add_special_tokens=True)["input_ids"].shape[1])
            base_pos = max(0, min(prefix_len - 1, full_len - 1))
            positions = [min(full_len - 1, base_pos + offset) for offset in OFFSETS]
            relative_position = base_pos - full_len
            baseline_capture = forward_capture(extractor, full_wrapped, positions=positions)
            random_ref = random_rms_direction(2048, SEED + 7_000_000 + task_idx)
            for direction_name in selected_names:
                direction = rms_unit(by_name[direction_name]["vector"])
                for alpha in alphas:
                    mode = "multi_loop_decayed"
                    mode_spec = build_intervention_mode(mode, alpha)
                    hook = BGLayerHookSteering(
                        extractor.model,
                        target_layer=int(primary["layer"]),
                        target_loops=mode_spec["target_loops"],
                        loop_alpha_scales=mode_spec["loop_alpha_scales"],
                        direction=direction,
                        alpha=alpha,
                        max_rms_fraction=0.05,
                        direction_normalization="rms",
                    )
                    try:
                        intervened_capture = forward_capture(
                            extractor,
                            full_wrapped,
                            positions=positions,
                            steering_hook=hook,
                            relative_position=relative_position,
                        )
                        diag = hook.diagnostics()
                        for layer in ["36", "47"]:
                            for loop in range(1, 5):
                                base_values = baseline_capture["states"].get(layer, {}).get(str(loop), {})
                                int_values = intervened_capture["states"].get(layer, {}).get(str(loop), {})
                                for offset_idx, offset in enumerate(OFFSETS):
                                    if str(offset_idx) not in base_values or str(offset_idx) not in int_values:
                                        continue
                                    delta = (int_values[str(offset_idx)].float() - base_values[str(offset_idx)].float()).flatten()
                                    delta_rms = float(delta.pow(2).mean().sqrt().item())
                                    traces.append(
                                        {
                                            "target": "T1",
                                            "task_id": suite_row["task_id"],
                                            "variant": "teacher_forced_same_token",
                                            "direction": direction_name,
                                            "alpha": alpha,
                                            "mode": mode,
                                            "capture_layer": int(layer),
                                            "capture_loop": loop,
                                            "token_offset": offset,
                                            "hidden_delta_rms": delta_rms,
                                            "hidden_delta_cosine_to_direction": cosine(delta, direction) if delta_rms > 0 else 0.0,
                                            "hidden_delta_cosine_to_random_control": cosine(delta, random_ref) if delta_rms > 0 else 0.0,
                                            "bg_score_shift": None,
                                            "logit_kl": kl_divergence(baseline_capture["logits"], intervened_capture["logits"]) if layer == "47" and loop == 4 and offset == 32 else None,
                                            "token_divergence_position": None,
                                            "activation_rms_change": diag.get("activation_rms_change"),
                                            "stability_status": "OK" if not diag.get("nan_or_inf") else "DESTABILIZING",
                                        }
                                    )
                    except Exception as exc:
                        traces.append(
                            {
                                "target": "T1",
                                "task_id": suite_row["task_id"],
                                "variant": "teacher_forced_same_token",
                                "direction": direction_name,
                                "alpha": alpha,
                                "mode": mode,
                                "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-2000:],
                                "stability_status": "DESTABILIZING",
                            }
                        )

                    # Free generation check.
                    for condition, signed_direction in [("positive", direction), ("negative", -direction), ("random", random_ref)]:
                        mode_spec = build_intervention_mode(mode, alpha)
                        hook = BGLayerHookSteering(
                            extractor.model,
                            target_layer=int(primary["layer"]),
                            target_loops=mode_spec["target_loops"],
                            loop_alpha_scales=mode_spec["loop_alpha_scales"],
                            direction=signed_direction,
                            alpha=alpha,
                            max_rms_fraction=0.05,
                            direction_normalization="rms",
                        )
                        try:
                            clean_text = baseline_cont
                            out_text, gen_meta = generate_with_layer_hook(
                                extractor,
                                continuation_prompt(task, prefix_text),
                                hook,
                                max_new_tokens=64,
                                seed=SEED + task_idx * 1000 + int(alpha * 10000),
                            )
                            candidate_clean = prefix_text + first_n_tokens_text(extractor.tokenizer, clean_text, POST_INTERVENTION_TOKENS)
                            candidate_int = prefix_text + first_n_tokens_text(extractor.tokenizer, out_text, POST_INTERVENTION_TOKENS)
                            # Score with ordinary capture; this measures end-state effect, not hidden delta.
                            from src.evaluator.bg_steering_hook import capture_bg_features_with_model

                            clean_features = capture_bg_features_with_model(
                                extractor.model,
                                extractor.tokenizer,
                                format_prompt_candidate(prompt_text, candidate_clean),
                                device=extractor.device,
                            )
                            int_features = capture_bg_features_with_model(
                                extractor.model,
                                extractor.tokenizer,
                                format_prompt_candidate(prompt_text, candidate_int),
                                device=extractor.device,
                            )
                            score_shift = (raw_head_score(target_head, int_features) - raw_head_score(target_head, clean_features)) / max(target_std, 1e-8)
                            diag = hook.diagnostics()
                            traces.append(
                                {
                                    "target": "T1",
                                    "task_id": suite_row["task_id"],
                                    "variant": "free_generation",
                                    "condition": condition,
                                    "direction": direction_name,
                                    "alpha": alpha,
                                    "mode": mode,
                                    "capture_layer": None,
                                    "capture_loop": None,
                                    "token_offset": 32,
                                    "hidden_delta_rms": None,
                                    "hidden_delta_cosine_to_direction": None,
                                    "bg_score_shift": score_shift,
                                    "logit_kl": None,
                                    "token_divergence_position": token_divergence(extractor.tokenizer, clean_text, out_text),
                                    "activation_rms_change": diag.get("activation_rms_change"),
                                    "repetition_rate": repetition_rate(out_text),
                                    "output_length": int(gen_meta.get("token_count", 0)),
                                    "stability_status": "OK" if not diag.get("nan_or_inf") else "DESTABILIZING",
                                }
                            )
                        except Exception as exc:
                            traces.append(
                                {
                                    "target": "T1",
                                    "task_id": suite_row["task_id"],
                                    "variant": "free_generation",
                                    "condition": condition,
                                    "direction": direction_name,
                                    "alpha": alpha,
                                    "mode": mode,
                                    "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-2000:],
                                    "stability_status": "DESTABILIZING",
                                }
                            )
            write_json(OUT_TRACES, {"traces": traces, "row_count": len(traces), "elapsed_seconds": round(time.time() - started, 3)})
    finally:
        if extractor is not None:
            extractor.cleanup()

    teacher = [row for row in traces if row.get("variant") == "teacher_forced_same_token" and row.get("hidden_delta_rms") is not None]
    later = [row for row in teacher if row.get("capture_layer") == 47 and row.get("token_offset") in {8, 32}]
    same = [row for row in teacher if row.get("token_offset") == 0 and row.get("capture_layer") == 36]
    survives32 = any(finite(row.get("hidden_delta_rms")) > 1e-5 for row in later if row.get("token_offset") == 32)
    survives8 = any(finite(row.get("hidden_delta_rms")) > 1e-5 for row in later if row.get("token_offset") == 8)
    same_only = any(finite(row.get("hidden_delta_rms")) > 1e-5 for row in same)
    free = [row for row in traces if row.get("variant") == "free_generation"]
    directional_free = [row for row in free if row.get("condition") == "positive" and finite(row.get("bg_score_shift")) > 0]
    if survives32:
        propagation = "PROPAGATES_TO_LATER_STATES"
        decay = "SURVIVES_32_TOKENS"
    elif survives8:
        propagation = "PROPAGATES_TO_LATER_STATES"
        decay = "SURVIVES_8_TOKENS_ONLY"
    elif same_only:
        propagation = "LOCAL_ONLY"
        decay = "SAME_TOKEN_ONLY"
    elif free and any(row.get("token_divergence_position") is not None for row in free):
        propagation = "TOKEN_DIVERGENCE_WITHOUT_BG_ALIGNMENT"
        decay = "NO_MEASURABLE_PROPAGATION"
    else:
        propagation = "INSUFFICIENT" if not traces else "SCRUBBED_BY_LATER_LAYERS"
        decay = "INSUFFICIENT" if not traces else "NO_MEASURABLE_PROPAGATION"
    logit_rows = [finite(row.get("logit_kl")) for row in traces if row.get("logit_kl") is not None]
    if directional_free:
        logit_verdict = "LOGITS_SHIFT_DIRECTIONALLY"
    elif logit_rows and max(logit_rows) > 1e-6:
        logit_verdict = "LOGITS_SHIFT_RANDOMLY"
    else:
        logit_verdict = "NO_LOGIT_EFFECT" if traces else "INSUFFICIENT"
    analysis = {
        "BG_PROPAGATION_VERDICT": propagation,
        "BG_PROPAGATION_DECAY_PROFILE": decay,
        "BG_LOGIT_EFFECT_VERDICT": logit_verdict,
        "selected_directions": selected_names,
        "alphas": alphas,
        "row_count": len(traces),
        "teacher_forced_row_count": len(teacher),
        "free_generation_row_count": len(free),
        "max_hidden_delta_rms": max([finite(row.get("hidden_delta_rms")) for row in teacher] or [0.0]),
        "mean_cosine_to_direction": sum(finite(row.get("hidden_delta_cosine_to_direction")) for row in teacher) / max(len(teacher), 1),
        "max_logit_kl": max(logit_rows or [0.0]),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_TRACES, {"traces": traces, "row_count": len(traces), "analysis": analysis, "elapsed_seconds": round(time.time() - started, 3)})
    write_json(OUT_ANALYSIS_JSON, analysis)
    lines = [
        "# BG Steering Propagation / Decay Map",
        "",
        f"BG_PROPAGATION_VERDICT = {propagation}",
        f"BG_PROPAGATION_DECAY_PROFILE = {decay}",
        f"BG_LOGIT_EFFECT_VERDICT = {logit_verdict}",
        "",
        f"- selected_directions: `{selected_names}`",
        f"- alphas: `{alphas}`",
        f"- row_count: `{len(traces)}`",
        f"- max_hidden_delta_rms: `{analysis['max_hidden_delta_rms']}`",
        f"- mean_cosine_to_direction: `{analysis['mean_cosine_to_direction']}`",
        f"- max_logit_kl: `{analysis['max_logit_kl']}`",
    ]
    write_md(OUT_ANALYSIS_MD, lines)
    print(f"BG_PROPAGATION_VERDICT = {propagation}")
    print(f"Wrote {rel(OUT_ANALYSIS_JSON)}")
    print(f"Wrote {rel(OUT_ANALYSIS_MD)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
