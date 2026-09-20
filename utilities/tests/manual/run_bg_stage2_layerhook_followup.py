"""Run the narrowed T1 layer-hook-only BG steering follow-up."""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "artifacts" / "hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "artifacts" / "hf_cache" / "datasets"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from bg_stage2_steering_preflight import head_weight_vector, load_head_row, rel  # noqa: E402
from bg_steering_suite_lib import evaluate_output  # noqa: E402
from bg_trajectory_prediction_lib import continuation_prompt  # noqa: E402
from run_bg_stage2_intervention_sweep import (  # noqa: E402
    compute_head_std,
    first_n_tokens_text,
    generate_with_layer_hook,
    load_stage1_baseline_by_key,
    parse_rate,
    random_direction,
    raw_head_score,
    repetition_rate,
)
from src.evaluator.bg_controller import BGController  # noqa: E402
from src.evaluator.bg_steering_hook import BGLayerHookSteering, build_intervention_mode, capture_bg_features_with_model  # noqa: E402
from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor, format_prompt_candidate  # noqa: E402


OUT_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_stage2_layerhook_followup_2026-05-18"
OUT_JSON = OUT_ROOT / "layerhook_followup_traces.json"
OUT_PARTIAL = OUT_ROOT / "layerhook_followup_traces.partial.json"
OUT_MD = OUT_ROOT / "layerhook_followup_traces.md"
SEED = 20260518
MAX_WALL_SECONDS = 3 * 60 * 60
MAX_TASKS = min(8, int(os.environ.get("BG_LAYERHOOK_FOLLOWUP_MAX_TASKS", "6")))
MAX_INTERVENTION_PASSES = 320
MAX_NEW_TOKENS = 128
POST_INTERVENTION_TOKENS = 32
MODES = ["single_loop_L1", "single_loop_L4", "multi_loop_uniform", "multi_loop_decayed"]
ALPHAS = [0.01, 0.02]
GUARD_VERSION = "layerhook_followup_no_parse_hardstop_v2"


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def condition_plan() -> list[dict[str, Any]]:
    rows = [{"alpha": 0.0, "condition": "zero_baseline", "random_control_idx": 0, "random_control_n": 0}]
    rows.extend(
        [
            {"alpha": 0.01, "condition": "positive", "random_control_idx": 0, "random_control_n": 3},
            {"alpha": 0.01, "condition": "negative", "random_control_idx": 0, "random_control_n": 3},
            {"alpha": 0.01, "condition": "random", "random_control_idx": 0, "random_control_n": 3},
            {"alpha": 0.01, "condition": "random", "random_control_idx": 1, "random_control_n": 3},
            {"alpha": 0.01, "condition": "random", "random_control_idx": 2, "random_control_n": 3},
            {"alpha": 0.02, "condition": "positive", "random_control_idx": 0, "random_control_n": 1},
            {"alpha": 0.02, "condition": "negative", "random_control_idx": 0, "random_control_n": 1},
            {"alpha": 0.02, "condition": "random", "random_control_idx": 0, "random_control_n": 1},
        ]
    )
    return rows


def generation_seed(mode_idx: int, task_idx: int, alpha: float, condition: str, random_control_idx: int) -> int:
    condition_offset = {"positive": 1, "negative": 2, "random": 3, "zero_baseline": 0}.get(condition, 9)
    return SEED + mode_idx * 10_000 + task_idx * 100 + int(alpha * 10000) + condition_offset + random_control_idx


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


def save_checkpoint(rows: list[dict[str, Any]], cell_verdicts: dict[str, str], passes: int, started: float, verdict: str = "PARTIAL") -> None:
    write_json(
        OUT_PARTIAL,
        {
            "BG_LAYERHOOK_FOLLOWUP_SWEEP_VERDICT": verdict,
            "verdict": verdict,
            "guard_version": GUARD_VERSION,
            "rows": rows,
            "row_count": len(rows),
            "intervention_forward_passes": passes,
            "cell_verdicts": cell_verdicts,
            "RANDOM_CONTROL_N": {"alpha_0.01": 3, "alpha_0.02": 1},
            "cache_intervention_mode": "disabled",
            "max_new_tokens": MAX_NEW_TOKENS,
            "elapsed_seconds": round(time.time() - started, 3),
        },
    )


def is_followup_destabilizing(row: dict[str, Any], baseline_stability: dict[str, Any]) -> bool:
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


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    preflight = load_json(OUT_ROOT / "preflight.json", {})
    tasks_payload = load_json(OUT_ROOT / "task_subset.json", {})
    if preflight.get("BG_LAYERHOOK_FOLLOWUP_PREFLIGHT_VERDICT") == "BLOCKED":
        payload = {"BG_LAYERHOOK_FOLLOWUP_SWEEP_VERDICT": "BLOCKED", "blocker": "preflight blocked", "rows": []}
        write_json(OUT_PARTIAL, payload)
        write_json(OUT_JSON, payload)
        print("BG_LAYERHOOK_FOLLOWUP_SWEEP_VERDICT = BLOCKED")
        return 1
    if tasks_payload.get("BG_LAYERHOOK_FOLLOWUP_TASKS_VERDICT") == "BLOCKED":
        payload = {"BG_LAYERHOOK_FOLLOWUP_SWEEP_VERDICT": "BLOCKED", "blocker": "task subset blocked", "rows": []}
        write_json(OUT_PARTIAL, payload)
        write_json(OUT_JSON, payload)
        print("BG_LAYERHOOK_FOLLOWUP_SWEEP_VERDICT = BLOCKED")
        return 1

    target = preflight["target"]
    controller = BGController.from_artifacts(device="cpu")
    target_head = load_head_row(str(target["head_id"]), controller)
    target_direction = head_weight_vector(target_head)
    d1 = preflight.get("diagnostic_d1")
    d1_head = load_head_row(str(d1["head_id"]), controller) if isinstance(d1, dict) else None
    objective_head = load_head_row("locked::objective_mixed", controller)
    hh_head = load_head_row("locked::hh_general", controller)
    target_std = compute_head_std(target_head, "reasoning", 64)
    d1_std = compute_head_std(d1_head, "reasoning", 256) if d1_head else 1.0
    stage1_baselines = load_stage1_baseline_by_key()
    tasks = list(tasks_payload.get("tasks") or [])[:MAX_TASKS]

    previous = load_json(OUT_PARTIAL, {})
    if previous.get("guard_version") != GUARD_VERSION:
        previous = {}
    rows: list[dict[str, Any]] = list(previous.get("rows") or [])
    passes = int(previous.get("intervention_forward_passes") or 0)
    cell_verdicts: dict[str, str] = dict(previous.get("cell_verdicts") or {})
    done_keys = {
        (
            row.get("task_subset_index"),
            row.get("mode"),
            row.get("alpha"),
            row.get("condition"),
            row.get("random_control_idx"),
        )
        for row in rows
    }

    extractor: BGTransformerFeatureExtractor | None = None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        extractor = BGTransformerFeatureExtractor(device=device, dtype="auto", force_all_loops=True)
        baseline_feature_cache: dict[int, torch.Tensor] = {}
        baseline_scores_cache: dict[int, dict[str, float]] = {}
        baseline_stability_cache: dict[int, dict[str, Any]] = {}

        for task_idx, suite_row in enumerate(tasks):
            if time.time() - started > MAX_WALL_SECONDS or passes >= MAX_INTERVENTION_PASSES:
                break
            task = dict(suite_row.get("task") or {})
            prefix_text = str(suite_row.get("prefix_text") or "")
            branch_id = int(suite_row.get("branch_id", 0))
            baseline = stage1_baselines.get((str(suite_row["task_id"]), branch_id, 64), {})
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

            for mode_idx, mode in enumerate(MODES):
                if time.time() - started > MAX_WALL_SECONDS or passes >= MAX_INTERVENTION_PASSES:
                    break
                cell_key = f"T1:layer_hook_injection:{mode}"
                if cell_verdicts.get(cell_key) == "DESTABILIZING":
                    continue
                for plan in condition_plan():
                    if time.time() - started > MAX_WALL_SECONDS or passes >= MAX_INTERVENTION_PASSES:
                        break
                    alpha = float(plan["alpha"])
                    condition = str(plan["condition"])
                    random_control_idx = int(plan["random_control_idx"])
                    random_control_n = int(plan["random_control_n"])
                    key = (suite_row.get("suite_index"), mode, alpha, condition, random_control_idx)
                    if key in done_keys:
                        continue
                    base_row = {
                        "task_subset_index": suite_row.get("suite_index"),
                        "task_id": suite_row["task_id"],
                        "domain": "reasoning",
                        "target_id": "T1",
                        "mechanism": "layer_hook_injection",
                        "mode": mode,
                        "intervention_mode": mode,
                        "alpha": alpha,
                        "condition": condition,
                        "random_control_idx": random_control_idx,
                        "random_control_n": random_control_n,
                        "prefix_length": 64,
                        "branch_id": branch_id,
                        "direction_source_head_id": target["head_id"],
                        "direction_source_config": target["config"],
                        "target_layer": target["layer"],
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
                        direction = target_direction
                    elif condition == "negative":
                        direction = -target_direction
                    else:
                        random_seed = 1_000_000 + mode_idx * 10_000 + task_idx * 100 + random_control_idx
                        direction = random_direction(int(target_direction.numel()), random_seed)
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
                    seed = generation_seed(mode_idx, task_idx, alpha, condition, random_control_idx)
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
                        post_candidate = prefix_text + first_n_tokens_text(
                            extractor.tokenizer,
                            output_text,
                            POST_INTERVENTION_TOKENS,
                        )
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
                    row["stability_warnings"] = {
                        "parse_failed": bool(row["parse_failed"]),
                        "hit_max_tokens": bool(row["hit_max_tokens"]),
                    }
                    row["safety_status"] = "DESTABILIZING" if is_followup_destabilizing(row, baseline_stability) else "OK"
                    if row["safety_status"] == "DESTABILIZING":
                        cell_verdicts[cell_key] = "DESTABILIZING"
                    else:
                        cell_verdicts.setdefault(cell_key, "READY")
                    rows.append(row)
                    passes += 1
                    done_keys.add(key)
                    save_checkpoint(rows, cell_verdicts, passes, started)
                    if row["safety_status"] == "DESTABILIZING":
                        break
                save_checkpoint(rows, cell_verdicts, passes, started)
            print(f"layerhook follow-up task {task_idx + 1}/{len(tasks)} rows={len(rows)} passes={passes}")
            save_checkpoint(rows, cell_verdicts, passes, started)
    finally:
        if extractor is not None:
            extractor.cleanup()

    completed_task_modes = defaultdict(set)
    for row in rows:
        if row.get("condition") != "zero_baseline" and not row.get("cuda_error"):
            completed_task_modes[int(row.get("task_subset_index", -1))].add(str(row.get("mode")))
    tasks_all_modes = sum(1 for modes in completed_task_modes.values() if set(MODES).issubset(modes))
    if tasks_all_modes >= 6 and not any(value == "DESTABILIZING" for value in cell_verdicts.values()):
        verdict = "READY"
    elif tasks_all_modes >= 3 or rows:
        verdict = "PARTIAL"
    else:
        verdict = "BLOCKED"
    payload = {
        "BG_LAYERHOOK_FOLLOWUP_SWEEP_VERDICT": verdict,
        "verdict": verdict,
        "guard_version": GUARD_VERSION,
        "rows": rows,
        "row_count": len(rows),
        "intervention_forward_passes": passes,
        "cell_verdicts": cell_verdicts,
        "tasks_completed_all_modes": tasks_all_modes,
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
        "# BG Stage 2 Layer-Hook Follow-Up Traces",
        "",
        f"BG_LAYERHOOK_FOLLOWUP_SWEEP_VERDICT = {verdict}",
        "",
        f"- row_count: `{len(rows)}`",
        f"- intervention_forward_passes: `{passes}`",
        f"- tasks_completed_all_modes: `{tasks_all_modes}`",
        f"- max_new_tokens: `{MAX_NEW_TOKENS}`",
        f"- cache_intervention_mode: `disabled`",
        "",
        "| task | mode | alpha | condition | random_idx | z | safety | correct | seconds |",
        "|---:|---|---:|---|---:|---:|---|---|---:|",
    ]
    for row in rows:
        seconds = row.get("generation_metadata", {}).get("seconds", "")
        lines.append(
            f"| {row.get('task_subset_index')} | `{row.get('mode')}` | {row.get('alpha')} | `{row.get('condition')}` | "
            f"{row.get('random_control_idx')} | {float(row.get('z_score_change') or 0):.4f} | `{row.get('safety_status')}` | "
            f"{row.get('is_correct')} | {seconds} |"
        )
    write_md(OUT_MD, lines)
    print(f"BG_LAYERHOOK_FOLLOWUP_SWEEP_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict in {"READY", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
