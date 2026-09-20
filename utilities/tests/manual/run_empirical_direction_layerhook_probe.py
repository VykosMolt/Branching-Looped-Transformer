"""Run layer-hook steering probe with empirical success-derived directions."""
from __future__ import annotations

import json
import os
import time
import traceback
from collections import defaultdict
from typing import Any

import torch

from bg_empirical_steering_common import OUT_ROOT, random_direction, rel, write_json, write_md
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


OUT_JSON = OUT_ROOT / "empirical_steering_traces.json"
OUT_PARTIAL = OUT_ROOT / "empirical_steering_traces.partial.json"
OUT_MD = OUT_ROOT / "empirical_steering_traces.md"
DIRECTIONS_PT = OUT_ROOT / "directions.pt"
TASKS_JSON = OUT_ROOT / "task_subset.json"
PREFLIGHT_JSON = OUT_ROOT / "preflight.json"
SEED = 20260518
MAX_WALL_SECONDS = 4 * 60 * 60
MAX_INTERVENTION_PASSES = 400
MAX_TASKS = int(os.environ.get("BG_EMPIRICAL_STEERING_MAX_TASKS", "6"))
MAX_NEW_TOKENS = 128
POST_INTERVENTION_TOKENS = 32
MODES = ["multi_loop_decayed", "single_loop_L1"]
ALPHAS = [0.01, 0.02]
GUARD_VERSION = "empirical_direction_layerhook_v1"


def load_json(path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def condition_plan() -> list[dict[str, Any]]:
    return [
        {"alpha": 0.0, "condition": "zero_baseline", "random_control_idx": 0, "random_control_n": 0},
        {"alpha": 0.01, "condition": "positive", "random_control_idx": 0, "random_control_n": 3},
        {"alpha": 0.01, "condition": "negative", "random_control_idx": 0, "random_control_n": 3},
        {"alpha": 0.01, "condition": "random", "random_control_idx": 0, "random_control_n": 3},
        {"alpha": 0.01, "condition": "random", "random_control_idx": 1, "random_control_n": 3},
        {"alpha": 0.01, "condition": "random", "random_control_idx": 2, "random_control_n": 3},
        {"alpha": 0.02, "condition": "positive", "random_control_idx": 0, "random_control_n": 1},
        {"alpha": 0.02, "condition": "negative", "random_control_idx": 0, "random_control_n": 1},
        {"alpha": 0.02, "condition": "random", "random_control_idx": 0, "random_control_n": 1},
    ]


def generation_seed(direction_idx: int, mode_idx: int, task_idx: int, alpha: float, condition: str, random_idx: int) -> int:
    condition_offset = {"positive": 1, "negative": 2, "random": 3, "zero_baseline": 0}.get(condition, 9)
    return SEED + direction_idx * 100_000 + mode_idx * 10_000 + task_idx * 100 + int(alpha * 10000) + condition_offset + random_idx


def baseline_features(
    extractor: BGTransformerFeatureExtractor,
    task: dict[str, Any],
    prefix_text: str,
    baseline_continuation: str,
) -> torch.Tensor:
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
            "BG_EMPIRICAL_STEERING_SWEEP_VERDICT": verdict,
            "verdict": verdict,
            "guard_version": GUARD_VERSION,
            "rows": rows,
            "row_count": len(rows),
            "intervention_forward_passes": passes,
            "cache_intervention_mode": "disabled",
            "max_new_tokens": MAX_NEW_TOKENS,
            "RANDOM_CONTROL_N": {"alpha_0.01": 3, "alpha_0.02": 1},
            "elapsed_seconds": round(time.time() - started, 3),
        },
    )


def is_destabilizing(row: dict[str, Any], baseline_stability: dict[str, Any]) -> bool:
    if row.get("cuda_error") or row.get("nan_or_inf_activations"):
        return True
    if row.get("empty_output"):
        return True
    base_rep = float(baseline_stability.get("repetition_rate", 0.0))
    rep = float(row.get("repetition_rate", 0.0))
    if rep > base_rep * 1.3 + 0.05 and rep - base_rep > 0.30:
        return True
    if float(row.get("activation_rms_change", 0.0)) > 5.0:
        return True
    return False


def direction_rows(directions_payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target = next(row for row in directions_payload["targets"] if row["target_id"] == "T1")
    rows = []
    for idx, row in enumerate(target["directions"]):
        if row["direction_name"] in {
            "RAW_NONORM_READOUT",
            "EMPIRICAL_MEAN_DIFF",
            "EMPIRICAL_WHITENED_DIFF",
            "LOGISTIC_SUCCESS_PROBE",
        }:
            rows.append({**row, "direction_idx": idx})
    return target, rows


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    preflight = load_json(PREFLIGHT_JSON, {})
    tasks_payload = load_json(TASKS_JSON, {})
    if preflight.get("BG_EMPIRICAL_STEERING_PREFLIGHT_VERDICT") == "BLOCKED" or tasks_payload.get("BG_EMPIRICAL_STEERING_TASKS_VERDICT") == "BLOCKED":
        payload = {"BG_EMPIRICAL_STEERING_SWEEP_VERDICT": "BLOCKED", "blocker": "preflight or task subset blocked", "rows": []}
        write_json(OUT_JSON, payload)
        write_json(OUT_PARTIAL, payload)
        print("BG_EMPIRICAL_STEERING_SWEEP_VERDICT = BLOCKED")
        return 1
    if not DIRECTIONS_PT.exists():
        payload = {"BG_EMPIRICAL_STEERING_SWEEP_VERDICT": "BLOCKED", "blocker": "directions.pt missing", "rows": []}
        write_json(OUT_JSON, payload)
        write_json(OUT_PARTIAL, payload)
        print("BG_EMPIRICAL_STEERING_SWEEP_VERDICT = BLOCKED")
        return 1

    directions_payload = torch.load(DIRECTIONS_PT, map_location="cpu", weights_only=False)
    target, directions = direction_rows(directions_payload)
    tasks = list(tasks_payload.get("tasks") or [])[:MAX_TASKS]
    controller = BGController.from_artifacts(device="cpu")
    target_head = load_head_row(str(target["head_id"]), controller)
    d1 = preflight.get("diagnostic_d1")
    d1_head = load_head_row(str(d1["head_id"]), controller) if isinstance(d1, dict) else None
    objective_head = load_head_row("locked::objective_mixed", controller)
    hh_head = load_head_row("locked::hh_general", controller)
    target_std = compute_head_std(target_head, str(target["domain"]), int(target["prefix_length"]))
    d1_std = compute_head_std(d1_head, "reasoning", 256) if d1_head else 1.0
    stage1_baselines = load_stage1_baseline_by_key()

    previous = load_json(OUT_PARTIAL, {})
    if previous.get("guard_version") != GUARD_VERSION:
        previous = {}
    rows: list[dict[str, Any]] = list(previous.get("rows") or [])
    passes = int(previous.get("intervention_forward_passes") or 0)
    done_keys = {
        (
            row.get("task_subset_index"),
            row.get("direction_name"),
            row.get("mode"),
            row.get("alpha"),
            row.get("condition"),
            row.get("random_control_idx"),
        )
        for row in rows
    }

    extractor: BGTransformerFeatureExtractor | None = None
    try:
        extractor = BGTransformerFeatureExtractor(device="cuda" if torch.cuda.is_available() else "cpu", dtype="auto", force_all_loops=True)
        baseline_feature_cache: dict[int, torch.Tensor] = {}
        baseline_scores_cache: dict[int, dict[str, float]] = {}
        baseline_stability_cache: dict[int, dict[str, Any]] = {}

        for mode_idx, mode in enumerate(MODES):
            for direction_idx, direction_meta in enumerate(directions):
                direction_base = direction_meta["vector"].detach().flatten().to(dtype=torch.float32, device="cpu")
                for task_idx, suite_row in enumerate(tasks):
                    if time.time() - started > MAX_WALL_SECONDS or passes >= MAX_INTERVENTION_PASSES:
                        break
                    task = dict(suite_row.get("task") or {})
                    prefix_text = str(suite_row.get("prefix_text") or "")
                    branch_id = int(suite_row.get("branch_id", 0))
                    baseline = stage1_baselines.get((str(suite_row["task_id"]), branch_id, int(target["prefix_length"])), {})
                    baseline_continuation = str(baseline.get("continuation_text") or suite_row.get("stage1_continuation_text") or "")
                    if task_idx not in baseline_feature_cache:
                        try:
                            feats = baseline_features(extractor, task, prefix_text, baseline_continuation)
                        except Exception:
                            feats = capture_bg_features_with_model(
                                extractor.model,
                                extractor.tokenizer,
                                format_prompt_candidate(str(task.get("prompt") or task.get("question") or ""), prefix_text),
                                device=extractor.device,
                            )
                        baseline_feature_cache[task_idx] = feats
                        baseline_eval = baseline.get("evaluation") or evaluate_output(task, baseline_continuation)
                        baseline_scores_cache[task_idx] = {
                            "target": raw_head_score(target_head, feats),
                            "diagnostic_d1": raw_head_score(d1_head, feats) if d1_head else 0.0,
                            "objective_mixed": raw_head_score(objective_head, feats),
                            "hh_general": raw_head_score(hh_head, feats),
                        }
                        baseline_stability_cache[task_idx] = {
                            "parse_rate": parse_rate(baseline_eval),
                            "repetition_rate": repetition_rate(baseline_continuation),
                            "empty_output": not bool(baseline_continuation.strip()),
                            "output_length": int(baseline.get("token_count", 0) or 0),
                            "hit_max_tokens": bool(baseline.get("hit_max_tokens", False)),
                        }
                    baseline_scores = baseline_scores_cache[task_idx]
                    baseline_stability = baseline_stability_cache[task_idx]

                    for plan in condition_plan():
                        if time.time() - started > MAX_WALL_SECONDS or passes >= MAX_INTERVENTION_PASSES:
                            break
                        alpha = float(plan["alpha"])
                        condition = str(plan["condition"])
                        random_control_idx = int(plan["random_control_idx"])
                        random_control_n = int(plan["random_control_n"])
                        key = (suite_row.get("suite_index"), direction_meta["direction_name"], mode, alpha, condition, random_control_idx)
                        if key in done_keys:
                            continue
                        base_row = {
                            "task_subset_index": suite_row.get("suite_index"),
                            "task_id": suite_row["task_id"],
                            "target": "T1",
                            "target_id": "T1",
                            "domain": target["domain"],
                            "prefix_length": int(target["prefix_length"]),
                            "branch_id": branch_id,
                            "mechanism": "layer_hook_injection",
                            "direction_name": direction_meta["direction_name"],
                            "direction_source": direction_meta.get("direction_source"),
                            "direction_training_split": {
                                "train_task_ids": target.get("train_task_ids"),
                                "heldout_task_ids": target.get("heldout_task_ids"),
                            },
                            "direction_heldout_auc": direction_meta.get("heldout_auc"),
                            "direction_heldout_pairwise_accuracy": direction_meta.get("heldout_pairwise_accuracy"),
                            "direction_cosine_to_raw_nonorm": direction_meta.get("cosine_to_raw_nonorm"),
                            "mode": mode,
                            "intervention_mode": mode,
                            "alpha": alpha,
                            "condition": condition,
                            "random_control_idx": random_control_idx,
                            "random_control_n": random_control_n,
                            "target_layer": int(target["layer"]),
                            "target_config": target["config"],
                            "target_head_id": target["head_id"],
                            "target_head_score_baseline": baseline_scores["target"],
                            "diagnostic_d1_score_baseline": baseline_scores["diagnostic_d1"],
                            "objective_mixed_score_baseline": baseline_scores["objective_mixed"],
                            "hh_general_score_baseline": baseline_scores["hh_general"],
                            "target_baseline_std": target_std,
                            "cache_intervention_mode": "disabled",
                        }
                        if condition == "zero_baseline":
                            eval_result = baseline.get("evaluation") or evaluate_output(task, baseline_continuation)
                            row = {
                                **base_row,
                                "direction_norm": 0.0,
                                "target_head_score_post": baseline_scores["target"],
                                "z_score_change": 0.0,
                                "diagnostic_d1_score": baseline_scores["diagnostic_d1"],
                                "diagnostic_d1_z_change": 0.0,
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
                                "parsed_answer": eval_result.get("parsed_answer"),
                                "is_correct": bool(eval_result.get("success")),
                                "parse_failed": not bool(eval_result.get("parsed", False)),
                                "repetition_rate": baseline_stability["repetition_rate"],
                                "output_length": baseline_stability["output_length"],
                                "hit_max_tokens": baseline_stability["hit_max_tokens"],
                                "empty_output": baseline_stability["empty_output"],
                            }
                            rows.append(row)
                            done_keys.add(key)
                            continue

                        if condition == "positive":
                            direction = direction_base
                        elif condition == "negative":
                            direction = -direction_base
                        else:
                            seed = 10_000_000 + direction_idx * 1_000_000 + mode_idx * 10_000 + task_idx * 100 + random_control_idx
                            direction = random_direction(int(direction_base.numel()), seed)
                        mode_spec = build_intervention_mode(mode, alpha)
                        hook = BGLayerHookSteering(
                            extractor.model,
                            target_layer=int(target["layer"]),
                            target_loops=mode_spec["target_loops"],
                            loop_alpha_scales=mode_spec["loop_alpha_scales"],
                            direction=direction,
                            alpha=alpha,
                        )
                        prompt = continuation_prompt(task, prefix_text)
                        seed = generation_seed(direction_idx, mode_idx, task_idx, alpha, condition, random_control_idx)
                        cuda_error = False
                        error_text = ""
                        try:
                            output_text, gen_meta = generate_with_layer_hook(
                                extractor,
                                prompt,
                                hook,
                                max_new_tokens=MAX_NEW_TOKENS,
                                seed=seed,
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
                            if "cuda" in str(exc).lower() or "out of memory" in str(exc).lower():
                                cuda_error = True
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            output_text = ""
                            gen_meta = {"token_count": 0, "hit_max_tokens": False, "seconds": 0.0}
                            post_features = baseline_feature_cache[task_idx]
                            eval_result = {"parsed": False, "success": False, "parsed_answer": None}
                            error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:]
                        diag = hook.diagnostics()
                        target_post = raw_head_score(target_head, post_features)
                        d1_post = raw_head_score(d1_head, post_features) if d1_head else 0.0
                        objective_post = raw_head_score(objective_head, post_features)
                        hh_post = raw_head_score(hh_head, post_features)
                        row = {
                            **base_row,
                            "direction_norm": float(torch.linalg.vector_norm(direction).item()),
                            "target_head_score_post": target_post,
                            "z_score_change": (target_post - baseline_scores["target"]) / max(target_std, 1e-8),
                            "diagnostic_d1_score": d1_post,
                            "diagnostic_d1_z_change": (d1_post - baseline_scores["diagnostic_d1"]) / max(d1_std, 1e-8),
                            "objective_mixed_score": objective_post,
                            "objective_mixed_z_change": objective_post - baseline_scores["objective_mixed"],
                            "hh_general_score": hh_post,
                            "hh_general_z_change": hh_post - baseline_scores["hh_general"],
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
                        row["stability_warnings"] = {"parse_failed": bool(row["parse_failed"]), "hit_max_tokens": bool(row["hit_max_tokens"])}
                        row["safety_status"] = "DESTABILIZING" if is_destabilizing(row, baseline_stability) else "OK"
                        rows.append(row)
                        passes += 1
                        done_keys.add(key)
                        save_checkpoint(rows, passes, started)
                save_checkpoint(rows, passes, started)
            print(f"empirical steering mode {mode} complete-ish rows={len(rows)} passes={passes}")
    finally:
        if extractor is not None:
            extractor.cleanup()

    coverage: dict[str, Any] = {}
    for direction_meta in directions:
        name = direction_meta["direction_name"]
        coverage[name] = {}
        for mode in MODES:
            intervention = [
                row
                for row in rows
                if row.get("direction_name") == name and row.get("mode") == mode and row.get("condition") != "zero_baseline"
            ]
            coverage[name][mode] = {
                "intervention_rows": len(intervention),
                "task_count": len({row.get("task_subset_index") for row in intervention}),
                "complete_expected_rows": len(tasks) * 8,
            }
    decayed_complete = all(coverage.get(row["direction_name"], {}).get("multi_loop_decayed", {}).get("intervention_rows", 0) >= len(tasks) * 8 for row in directions[:3])
    empirical_names = {row["direction_name"] for row in directions if row["direction_name"] != "RAW_NONORM_READOUT"}
    empirical_covered = {
        row.get("direction_name")
        for row in rows
        if row.get("direction_name") in empirical_names and row.get("condition") != "zero_baseline"
    }
    if decayed_complete and "RAW_NONORM_READOUT" in coverage and len(empirical_covered) >= 2 and not any(row.get("cuda_error") for row in rows):
        verdict = "READY"
    elif empirical_covered:
        verdict = "PARTIAL"
    else:
        verdict = "BLOCKED"
    payload = {
        "BG_EMPIRICAL_STEERING_SWEEP_VERDICT": verdict,
        "verdict": verdict,
        "guard_version": GUARD_VERSION,
        "rows": rows,
        "row_count": len(rows),
        "intervention_forward_passes": passes,
        "MODE_COVERAGE": coverage,
        "max_tasks": MAX_TASKS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_intervention_forward_passes": MAX_INTERVENTION_PASSES,
        "RANDOM_CONTROL_N": {"alpha_0.01": 3, "alpha_0.02": 1},
        "cache_intervention_mode": "disabled",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_PARTIAL, payload)
    write_json(OUT_JSON, payload)
    lines = [
        "# BG Empirical Direction Layer-Hook Traces",
        "",
        f"BG_EMPIRICAL_STEERING_SWEEP_VERDICT = {verdict}",
        "",
        f"- row_count: `{len(rows)}`",
        f"- intervention_forward_passes: `{passes}`",
        f"- cache_intervention_mode: `disabled`",
        "",
        "| task | direction | mode | alpha | condition | random_idx | z | safety | correct | seconds |",
        "|---:|---|---|---:|---|---:|---:|---|---|---:|",
    ]
    for row in rows:
        seconds = row.get("generation_metadata", {}).get("seconds", "")
        lines.append(
            f"| {row.get('task_subset_index')} | `{row.get('direction_name')}` | `{row.get('mode')}` | {row.get('alpha')} | "
            f"`{row.get('condition')}` | {row.get('random_control_idx')} | {float(row.get('z_score_change') or 0):.4f} | "
            f"`{row.get('safety_status')}` | {row.get('is_correct')} | {seconds} |"
        )
    write_md(OUT_MD, lines)
    print(f"BG_EMPIRICAL_STEERING_SWEEP_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict in {"READY", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
