"""Run RMS-calibrated BG layer-hook steering probe."""
from __future__ import annotations

import json
import time
import traceback
from typing import Any

import torch

from bg_preconsolidation_common import (
    OUT_ROOT,
    POST_INTERVENTION_TOKENS,
    SEED,
    condition_plan,
    finite,
    random_l2_direction,
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


OUT_JSON = OUT_ROOT / "rms_steering_traces.json"
OUT_PARTIAL = OUT_ROOT / "rms_steering_traces.partial.json"
OUT_MD = OUT_ROOT / "rms_steering_traces.md"
PREFLIGHT_JSON = OUT_ROOT / "preflight.json"
MAX_WALL_SECONDS = 8 * 60 * 60
MAX_INTERVENTION_PASSES = 800
MAX_TASKS_PRIMARY = 6
MAX_NEW_TOKENS = 128
MODES = ["multi_loop_decayed", "single_loop_L1"]
ALPHAS = [0.005, 0.01, 0.02]
GUARD_VERSION = "rms_calibrated_layerhook_v1"


def load_json(path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def baseline_features(extractor: BGTransformerFeatureExtractor, task: dict[str, Any], prefix_text: str, baseline_continuation: str) -> torch.Tensor:
    continuation_32 = first_n_tokens_text(extractor.tokenizer, baseline_continuation, POST_INTERVENTION_TOKENS)
    candidate = str(prefix_text or "") + continuation_32
    return capture_bg_features_with_model(
        extractor.model,
        extractor.tokenizer,
        format_prompt_candidate(str(task.get("prompt") or task.get("question") or ""), candidate),
        device=extractor.device,
    )


def save_checkpoint(rows: list[dict[str, Any]], passes: int, started: float, verdict: str = "PARTIAL") -> None:
    write_json(
        OUT_PARTIAL,
        {
            "BG_RMS_STEERING_SWEEP_VERDICT": verdict,
            "verdict": verdict,
            "guard_version": GUARD_VERSION,
            "rows": rows,
            "row_count": len(rows),
            "intervention_forward_passes": passes,
            "cache_intervention_mode": "disabled",
            "max_new_tokens": MAX_NEW_TOKENS,
            "elapsed_seconds": round(time.time() - started, 3),
        },
    )


def is_destabilizing(row: dict[str, Any], baseline_stability: dict[str, Any]) -> bool:
    if row.get("cuda_error") or row.get("nan_or_inf_activations") or row.get("empty_output"):
        return True
    base_rep = float(baseline_stability.get("repetition_rate", 0.0))
    rep = float(row.get("repetition_rate", 0.0))
    if rep > base_rep * 1.3 + 0.05 and rep - base_rep > 0.30:
        return True
    if float(row.get("effective_delta_rms_fraction", 0.0)) > 0.021:
        return True
    return False


def jobs_for_target(target: dict[str, Any], directions: list[dict[str, Any]], tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for norm_mode in ["RMS_NORMALIZED", "L2_NORMALIZED_CONTROL"]:
        for direction_idx, direction in enumerate(directions):
            if norm_mode == "L2_NORMALIZED_CONTROL":
                if direction["direction_name"] not in {"RAW_NONORM_READOUT", "EMPIRICAL_WHITENED_DIFF"}:
                    continue
                task_rows = tasks[:3]
                mode_rows = ["multi_loop_decayed"]
                plan_rows = condition_plan([0.01, 0.02], extra_random_at_001=False)
            else:
                task_rows = tasks
                mode_rows = MODES
                plan_rows = condition_plan(ALPHAS, extra_random_at_001=True)
            for mode_idx, mode in enumerate(mode_rows):
                for task_idx, suite_row in enumerate(task_rows):
                    for plan in plan_rows:
                        jobs.append(
                            {
                                "target": target,
                                "direction": direction,
                                "direction_idx": direction_idx,
                                "normalization_mode": norm_mode,
                                "mode": mode,
                                "mode_idx": mode_idx,
                                "suite_row": suite_row,
                                "task_idx": task_idx,
                                "plan": plan,
                            }
                        )
    return jobs


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    preflight = load_json(PREFLIGHT_JSON, {})
    if preflight.get("BG_PRECONSOLIDATION_PREFLIGHT_VERDICT") == "BLOCKED":
        payload = {"BG_RMS_STEERING_SWEEP_VERDICT": "BLOCKED", "blocker": "preflight blocked", "rows": []}
        write_json(OUT_JSON, payload)
        write_json(OUT_PARTIAL, payload)
        print("BG_RMS_STEERING_SWEEP_VERDICT = BLOCKED")
        return 1

    controller = BGController.from_artifacts(device="cpu")
    targets = select_targets()
    primary = next(row for row in targets if row["target_id"] == "T1")
    directions = target_directions(primary, controller)
    directions = [row for row in directions if row["direction_name"] in {"RAW_NONORM_READOUT", "EMPIRICAL_MEAN_DIFF", "EMPIRICAL_WHITENED_DIFF", "LOGISTIC_SUCCESS_PROBE"}]
    tasks = task_subset_for_target(primary, MAX_TASKS_PRIMARY)
    if len(tasks) < 3 or not directions:
        payload = {"BG_RMS_STEERING_SWEEP_VERDICT": "BLOCKED", "blocker": "insufficient tasks or directions", "rows": []}
        write_json(OUT_JSON, payload)
        write_json(OUT_PARTIAL, payload)
        print("BG_RMS_STEERING_SWEEP_VERDICT = BLOCKED")
        return 1

    target_head = load_head_row(str(primary["head_id"]), controller)
    objective_head = load_head_row("locked::objective_mixed", controller)
    hh_head = load_head_row("locked::hh_general", controller)
    d1 = preflight.get("diagnostic_d1")
    d1_head = load_head_row(str(d1["head_id"]), controller) if isinstance(d1, dict) else None
    target_std = compute_head_std(target_head, str(primary["domain"]), int(primary["prefix_length"]))
    stage1_baselines = load_stage1_baseline_by_key()

    previous = load_json(OUT_PARTIAL, {})
    if previous.get("guard_version") != GUARD_VERSION:
        previous = {}
    rows: list[dict[str, Any]] = list(previous.get("rows") or [])
    passes = int(previous.get("intervention_forward_passes") or 0)
    done_keys = {
        (
            row.get("target_id"),
            row.get("task_id"),
            row.get("direction_name"),
            row.get("normalization_mode"),
            row.get("intervention_mode"),
            row.get("alpha"),
            row.get("condition"),
            row.get("random_control_idx"),
        )
        for row in rows
    }

    extractor: BGTransformerFeatureExtractor | None = None
    try:
        extractor = BGTransformerFeatureExtractor(device="cuda" if torch.cuda.is_available() else "cpu", dtype="auto", force_all_loops=True)
        baseline_feature_cache: dict[str, torch.Tensor] = {}
        baseline_scores_cache: dict[str, dict[str, float]] = {}
        baseline_stability_cache: dict[str, dict[str, Any]] = {}
        jobs = jobs_for_target(primary, directions, tasks)
        for job_idx, job in enumerate(jobs):
            if time.time() - started > MAX_WALL_SECONDS or passes >= MAX_INTERVENTION_PASSES:
                break
            suite_row = job["suite_row"]
            plan = job["plan"]
            alpha = float(plan["alpha"])
            condition = str(plan["condition"])
            random_control_idx = int(plan["random_control_idx"])
            direction_meta = job["direction"]
            mode = str(job["mode"])
            norm_mode = str(job["normalization_mode"])
            key = (
                primary["target_id"],
                suite_row["task_id"],
                direction_meta["direction_name"],
                norm_mode,
                mode,
                alpha,
                condition,
                random_control_idx,
            )
            if key in done_keys:
                continue

            task = dict(suite_row.get("task") or {})
            prefix_text = str(suite_row.get("prefix_text") or "")
            branch_id = int(suite_row.get("branch_id", 0))
            baseline = stage1_baselines.get((str(suite_row["task_id"]), branch_id, int(primary["prefix_length"])), {})
            baseline_continuation = str(baseline.get("continuation_text") or suite_row.get("stage1_continuation_text") or "")
            task_cache_key = str(suite_row["task_id"])
            if task_cache_key not in baseline_feature_cache:
                try:
                    feats = baseline_features(extractor, task, prefix_text, baseline_continuation)
                except Exception:
                    feats = capture_bg_features_with_model(
                        extractor.model,
                        extractor.tokenizer,
                        format_prompt_candidate(str(task.get("prompt") or task.get("question") or ""), prefix_text),
                        device=extractor.device,
                    )
                baseline_eval = baseline.get("evaluation") or evaluate_output(task, baseline_continuation)
                baseline_feature_cache[task_cache_key] = feats
                baseline_scores_cache[task_cache_key] = {
                    "target": raw_head_score(target_head, feats),
                    "diagnostic_d1": raw_head_score(d1_head, feats) if d1_head else 0.0,
                    "objective_mixed": raw_head_score(objective_head, feats),
                    "hh_general": raw_head_score(hh_head, feats),
                }
                baseline_stability_cache[task_cache_key] = {
                    "parse_rate": parse_rate(baseline_eval),
                    "repetition_rate": repetition_rate(baseline_continuation),
                    "empty_output": not bool(baseline_continuation.strip()),
                    "output_length": int(baseline.get("token_count", 0) or 0),
                    "hit_max_tokens": bool(baseline.get("hit_max_tokens", False)),
                }
            baseline_scores = baseline_scores_cache[task_cache_key]
            baseline_stability = baseline_stability_cache[task_cache_key]

            base_row = {
                "target": "T1",
                "target_id": primary["target_id"],
                "domain": primary["domain"],
                "prefix_length": int(primary["prefix_length"]),
                "task_id": suite_row["task_id"],
                "task_subset_index": suite_row["suite_index"],
                "branch_id": branch_id,
                "direction_name": direction_meta["direction_name"],
                "direction_source": direction_meta.get("direction_source"),
                "direction_heldout_auc": direction_meta.get("heldout_auc"),
                "normalization_mode": norm_mode,
                "intervention_mode": mode,
                "mode": mode,
                "alpha": alpha,
                "condition": condition,
                "random_control_idx": random_control_idx,
                "random_control_n": int(plan["random_control_n"]),
                "target_layer": int(primary["layer"]),
                "target_head_score_baseline": baseline_scores["target"],
                "diagnostic_d1_score_baseline": baseline_scores["diagnostic_d1"],
                "objective_mixed_score_baseline": baseline_scores["objective_mixed"],
                "hh_general_score_baseline": baseline_scores["hh_general"],
                "target_baseline_std": target_std,
                "cache_intervention_mode": "disabled",
            }
            if condition == "zero_baseline":
                baseline_eval = baseline.get("evaluation") or evaluate_output(task, baseline_continuation)
                rows.append(
                    {
                        **base_row,
                        "direction_norm": 0.0,
                        "effective_delta_rms_fraction": 0.0,
                        "target_head_score_post": baseline_scores["target"],
                        "z_score_change": 0.0,
                        "diagnostic_d1_score": baseline_scores["diagnostic_d1"],
                        "objective_mixed_score": baseline_scores["objective_mixed"],
                        "hh_general_score": baseline_scores["hh_general"],
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
                        "repetition_rate": baseline_stability["repetition_rate"],
                        "output_length": baseline_stability["output_length"],
                        "hit_max_tokens": baseline_stability["hit_max_tokens"],
                        "empty_output": baseline_stability["empty_output"],
                    }
                )
                save_checkpoint(rows, passes, started)
                continue

            base_direction_l2 = unit(direction_meta["vector"])
            if condition == "random":
                seed = SEED + 5_000_000 + job_idx * 101 + random_control_idx
                direction = random_rms_direction(2048, seed) if norm_mode == "RMS_NORMALIZED" else random_l2_direction(2048, seed)
            else:
                sign = 1.0 if condition == "positive" else -1.0
                direction = sign * (rms_unit(base_direction_l2) if norm_mode == "RMS_NORMALIZED" else base_direction_l2)
            hook_direction_norm = "rms" if norm_mode == "RMS_NORMALIZED" else "l2"
            mode_spec = build_intervention_mode(mode, alpha)
            hook = BGLayerHookSteering(
                extractor.model,
                target_layer=int(primary["layer"]),
                target_loops=mode_spec["target_loops"],
                loop_alpha_scales=mode_spec["loop_alpha_scales"],
                direction=direction,
                alpha=alpha,
                max_rms_fraction=0.05,
                direction_normalization=hook_direction_norm,
            )
            cuda_error = False
            error_text = ""
            try:
                output_text, gen_meta = generate_with_layer_hook(
                    extractor,
                    continuation_prompt(task, prefix_text),
                    hook,
                    max_new_tokens=MAX_NEW_TOKENS,
                    seed=SEED + job_idx * 17,
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
                post_features = baseline_feature_cache[task_cache_key]
                eval_result = {"parsed": False, "success": False, "parsed_answer": None}
                error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:]
            diag = hook.diagnostics()
            target_post = raw_head_score(target_head, post_features)
            row = {
                **base_row,
                "direction_norm": float(torch.linalg.vector_norm(direction.float()).item()),
                "direction_rms": float(direction.float().pow(2).mean().sqrt().item()),
                "effective_delta_rms_fraction": finite(diag.get("activation_rms_change")),
                "target_head_score_post": target_post,
                "z_score_change": (target_post - baseline_scores["target"]) / max(target_std, 1e-8),
                "diagnostic_d1_score": raw_head_score(d1_head, post_features) if d1_head else 0.0,
                "objective_mixed_score": raw_head_score(objective_head, post_features),
                "hh_general_score": raw_head_score(hh_head, post_features),
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
            row["safety_status"] = "DESTABILIZING" if is_destabilizing(row, baseline_stability) else "OK"
            rows.append(row)
            passes += 1
            save_checkpoint(rows, passes, started)
    finally:
        if extractor is not None:
            extractor.cleanup()

    intervention_rows = [row for row in rows if row.get("condition") != "zero_baseline"]
    complete = len([row for row in intervention_rows if row.get("normalization_mode") == "RMS_NORMALIZED"]) >= 250
    verdict = "READY" if complete else ("PARTIAL" if intervention_rows else "BLOCKED")
    payload = {
        "BG_RMS_STEERING_SWEEP_VERDICT": verdict,
        "verdict": verdict,
        "guard_version": GUARD_VERSION,
        "rows": rows,
        "row_count": len(rows),
        "intervention_forward_passes": passes,
        "cache_intervention_mode": "disabled",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    save_checkpoint(rows, passes, started, verdict)
    lines = [
        "# BG RMS-Calibrated Steering Traces",
        "",
        f"BG_RMS_STEERING_SWEEP_VERDICT = {verdict}",
        "",
        f"- row_count: `{len(rows)}`",
        f"- intervention_forward_passes: `{passes}`",
        "- cache_intervention_mode: `disabled`",
        "",
        "| norm | direction | mode | alpha | condition | n | mean z | safety failures |",
        "|---|---|---|---:|---|---:|---:|---:|",
    ]
    grouped: dict[tuple[str, str, str, float, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("normalization_mode")),
            str(row.get("direction_name")),
            str(row.get("intervention_mode")),
            finite(row.get("alpha")),
            str(row.get("condition")),
        )
        grouped.setdefault(key, []).append(row)
    for (norm_mode, direction, mode, alpha, condition), group in sorted(grouped.items()):
        z_vals = [finite(row.get("z_score_change")) for row in group]
        lines.append(
            f"| `{norm_mode}` | `{direction}` | `{mode}` | {alpha:g} | `{condition}` | "
            f"{len(group)} | {sum(z_vals) / max(len(z_vals), 1):.4f} | "
            f"{sum(1 for row in group if row.get('safety_status') == 'DESTABILIZING')} |"
        )
    write_md(OUT_MD, lines)
    print(f"BG_RMS_STEERING_SWEEP_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict in {"READY", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
