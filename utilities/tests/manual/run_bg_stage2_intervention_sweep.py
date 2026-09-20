"""Run BG Stage 2 targeted steering sensitivity sweep."""
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
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "artifacts" / "hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "artifacts" / "hf_cache" / "datasets"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from bg_stage2_steering_preflight import head_weight_vector, load_head_row, rel  # noqa: E402
from bg_trajectory_prediction_lib import continuation_budget, continuation_prompt, domain_hint_for_task  # noqa: E402
from bg_steering_suite_lib import evaluate_output  # noqa: E402
from src.evaluator.bg_controller import BGController, config_vector  # noqa: E402
from src.evaluator.bg_steering_hook import (  # noqa: E402
    BGLayerHookSteering,
    build_intervention_mode,
    capture_bg_features_with_model,
)
from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor, format_prompt_candidate  # noqa: E402


REPORT_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_stage2_steering_2026-05-18"
STAGE1_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_trajectory_prediction_2026-05-18"
OUT_JSON = REPORT_ROOT / "intervention_traces.json"
OUT_PARTIAL = REPORT_ROOT / "intervention_traces.partial.json"
SEED = 20260518
MAX_WALL_SECONDS = 6 * 60 * 60
MAX_TASKS = 30
MAX_TOTAL_INTERVENTION_FORWARD_PASSES = 1500
POST_INTERVENTION_TOKENS = 32
ALPHAS = [0.0, 0.005, 0.01, 0.02]
MODES = ["single_loop_L1", "single_loop_L4", "multi_loop_uniform", "multi_loop_decayed"]


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def first_n_tokens_text(tokenizer: Any, text: str, n: int) -> str:
    ids = tokenizer(str(text or ""), return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    if int(ids.numel()) <= n:
        return str(text or "")
    return tokenizer.decode(ids[:n], skip_special_tokens=True)


def repetition_rate(text: str) -> float:
    words = str(text or "").split()
    repeated = sum(1 for a, b in zip(words, words[1:]) if a == b)
    return repeated / max(len(words) - 1, 1)


def parse_rate(eval_result: dict[str, Any]) -> float:
    return 1.0 if bool(eval_result.get("parsed", eval_result.get("syntax_ok", False))) else 0.0


def load_stage1_baseline_by_key() -> dict[tuple[str, int, int], dict[str, Any]]:
    payload = load_json(STAGE1_ROOT / "continued_prefixes.json", {})
    out = {}
    for row in payload.get("continued_prefixes") or []:
        out[(str(row.get("task_id")), int(row.get("branch_id", -1)), int(row.get("prefix_length", -1)))] = row
    return out


def raw_head_score(head_row: dict[str, Any], features: torch.Tensor) -> float:
    vec = config_vector(features, str(head_row["config"])).to(dtype=torch.float32)
    weight = head_weight_vector(head_row).to(dtype=torch.float32)
    if str(head_row["architecture"]) == "AntisymLinear":
        vec = nn.functional.layer_norm(vec, (int(vec.numel()),), weight=None, bias=None)
    return float(torch.dot(vec.flatten(), weight.flatten()).detach().cpu().item())


def compute_head_std(head_row: dict[str, Any], domain: str, prefix_length: int) -> float:
    feature_path = STAGE1_ROOT / "prefix_features.pt"
    if not feature_path.exists():
        return 1.0
    try:
        payload = torch.load(feature_path, map_location="cpu", weights_only=False)
    except Exception:
        return 1.0
    values = []
    for record in payload.get("records") or []:
        if str(record.get("domain")) == str(domain) and int(record.get("prefix_length", -1)) == int(prefix_length):
            try:
                values.append(raw_head_score(head_row, record["features"]))
            except Exception:
                continue
    if len(values) < 2:
        return 1.0
    tensor = torch.tensor(values, dtype=torch.float32)
    std = float(tensor.std(unbiased=False).item())
    return std if math.isfinite(std) and std > 1e-8 else 1.0


def generation_seed(target_idx: int, mechanism_idx: int, mode_idx: int, task_idx: int, alpha: float, condition: str) -> int:
    condition_offset = {"positive": 1, "negative": 2, "random": 3, "zero_baseline": 0}.get(condition, 9)
    return SEED + target_idx * 1_000_000 + mechanism_idx * 100_000 + mode_idx * 10_000 + task_idx * 100 + int(alpha * 10000) + condition_offset


def random_direction(dim: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    vec = torch.randn(dim, generator=gen, dtype=torch.float32)
    return vec / torch.linalg.vector_norm(vec).clamp(min=1e-12)


def generate_with_layer_hook(
    extractor: BGTransformerFeatureExtractor,
    prompt: str,
    hook: BGLayerHookSteering,
    *,
    max_new_tokens: int,
    seed: int,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> tuple[str, dict[str, Any]]:
    model = extractor.model
    tokenizer = extractor.tokenizer
    device = extractor.device
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536, padding=False)
    enc = {key: value.to(device) for key, value in enc.items()}
    prompt_len = int(enc["input_ids"].shape[1])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    started = time.time()
    hook.apply(position=-1)
    try:
        with torch.inference_mode():
            generated = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )
    finally:
        hook.remove()
    new_ids = generated[0, prompt_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    meta = {
        "token_count": int(new_ids.numel()),
        "hit_max_tokens": int(new_ids.numel()) >= int(max_new_tokens),
        "seconds": round(time.time() - started, 3),
        "generation_seed": int(seed),
    }
    return text, meta


def baseline_post_features(
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


def condition_rows() -> list[tuple[float, str, int]]:
    rows = [(0.0, "zero_baseline", 0)]
    for alpha in ALPHAS[1:]:
        rows.extend([(alpha, "positive", 0), (alpha, "negative", 0), (alpha, "random", 0)])
    return rows


def is_destabilizing(row: dict[str, Any], baseline_stability: dict[str, Any]) -> bool:
    if row.get("cuda_error") or row.get("nan_or_inf_activations"):
        return True
    if row.get("empty_output"):
        return True
    base_rep = float(baseline_stability.get("repetition_rate", 0.0))
    rep = float(row.get("repetition_rate", 0.0))
    if rep > base_rep * 1.3 + 0.05 and rep - base_rep > 0.30:
        return True
    base_parse_fail = 1.0 - float(baseline_stability.get("parse_rate", 0.0))
    parse_fail = 1.0 - float(row.get("parse_rate", 0.0))
    if parse_fail > base_parse_fail * 1.5 + 0.50:
        return True
    if float(row.get("activation_rms_change", 0.0)) > 5.0:
        return True
    return False


def main() -> int:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    preflight = load_json(REPORT_ROOT / "preflight.json", {})
    task_suite = load_json(REPORT_ROOT / "task_suite.json", {})
    if preflight.get("BG_STAGE2_PREFLIGHT_VERDICT") == "BLOCKED":
        payload = {
            "BG_STAGE2_INTERVENTION_SWEEP_VERDICT": "BLOCKED",
            "blocker": "preflight verdict is BLOCKED",
            "rows": [],
            "elapsed_seconds": 0.0,
        }
        write_json(OUT_PARTIAL, payload)
        write_json(OUT_JSON, payload)
        print("BG_STAGE2_INTERVENTION_SWEEP_VERDICT = BLOCKED")
        return 1
    if task_suite.get("BG_STAGE2_TASK_SUITE_VERDICT") == "BLOCKED":
        payload = {
            "BG_STAGE2_INTERVENTION_SWEEP_VERDICT": "BLOCKED",
            "blocker": "task suite verdict is BLOCKED",
            "rows": [],
            "elapsed_seconds": 0.0,
        }
        write_json(OUT_PARTIAL, payload)
        write_json(OUT_JSON, payload)
        print("BG_STAGE2_INTERVENTION_SWEEP_VERDICT = BLOCKED")
        return 1

    controller = BGController.from_artifacts(device="cpu")
    targets = {str(row["target_id"]): row for row in preflight.get("targets", []) if row.get("status") == "ready"}
    d1 = preflight.get("diagnostic_d1")
    d1_head = load_head_row(str(d1["head_id"]), controller) if isinstance(d1, dict) else None
    locked_hh = load_head_row("locked::hh_general", controller)
    locked_obj = load_head_row("locked::objective_mixed", controller)
    target_heads = {target_id: load_head_row(str(row["head_id"]), controller) for target_id, row in targets.items()}
    target_dirs = {target_id: head_weight_vector(target_heads[target_id]) for target_id in target_heads}
    target_stds = {
        target_id: compute_head_std(target_heads[target_id], str(targets[target_id]["domain"]), int(targets[target_id]["prefix_length"]))
        for target_id in target_heads
    }
    d1_std = compute_head_std(d1_head, "reasoning", 256) if d1_head else 1.0
    hh_std = 1.0
    obj_std = 1.0
    stage1_baselines = load_stage1_baseline_by_key()

    mechanisms = ["layer_hook_injection"]
    if preflight.get("LATENT_LOOP_BOUNDARY_FORK_VERDICT") == "READY":
        mechanisms.append("latent_loop_boundary_fork")
    else:
        print("LATENT_LOOP_BOUNDARY_FORK_VERDICT = BLOCKED; sweep will run layer_hook_injection only")

    previous = load_json(OUT_PARTIAL, {})
    rows: list[dict[str, Any]] = list(previous.get("rows") or [])
    done_keys = {
        (
            row.get("task_suite_index"),
            row.get("target_id"),
            row.get("mechanism"),
            row.get("intervention_mode"),
            row.get("alpha"),
            row.get("condition"),
            row.get("random_control_idx"),
        )
        for row in rows
    }
    cell_verdicts: dict[str, str] = dict(previous.get("cell_verdicts") or {})
    intervention_passes = int(previous.get("intervention_forward_passes", 0))

    tasks = list(task_suite.get("tasks") or [])[:MAX_TASKS]
    baseline_feature_cache: dict[int, torch.Tensor] = {}
    baseline_score_cache: dict[int, dict[str, float]] = {}
    baseline_stability_cache: dict[int, dict[str, Any]] = {}

    extractor: BGTransformerFeatureExtractor | None = None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        extractor = BGTransformerFeatureExtractor(device=device, dtype="auto", force_all_loops=True)
        for task_idx, suite_row in enumerate(tasks):
            if time.time() - started > MAX_WALL_SECONDS:
                break
            target_id = str(suite_row["target_assignment"])
            if target_id not in targets:
                continue
            task = dict(suite_row.get("task") or {})
            prefix_text = str(suite_row.get("prefix_text") or "")
            branch_id = int(suite_row.get("partial_trajectory_branch", 0))
            prefix_length = int(suite_row.get("prefix_length", targets[target_id]["prefix_length"]))
            baseline = stage1_baselines.get((str(suite_row["task_id"]), branch_id, prefix_length), {})
            baseline_cont = str(baseline.get("continuation_text") or suite_row.get("stage1_continuation_text") or "")
            if task_idx not in baseline_feature_cache:
                try:
                    features = baseline_post_features(extractor, task, prefix_text, baseline_cont)
                except Exception:
                    features = capture_bg_features_with_model(
                        extractor.model,
                        extractor.tokenizer,
                        format_prompt_candidate(str(task.get("prompt") or task.get("question") or ""), prefix_text),
                        device=extractor.device,
                    )
                baseline_feature_cache[task_idx] = features
                baseline_eval = baseline.get("evaluation") or evaluate_output(task, baseline_cont)
                baseline_stability_cache[task_idx] = {
                    "parse_rate": parse_rate(baseline_eval),
                    "repetition_rate": repetition_rate(baseline_cont),
                    "empty_output": not bool(baseline_cont.strip()),
                    "output_length": int(baseline.get("token_count", 0) or 0),
                    "hit_max_tokens": bool(baseline.get("hit_max_tokens", False)),
                }
                baseline_score_cache[task_idx] = {
                    "target": raw_head_score(target_heads[target_id], features),
                    "diagnostic_d1": raw_head_score(d1_head, features) if d1_head else 0.0,
                    "hh_general": raw_head_score(locked_hh, features),
                    "objective_mixed": raw_head_score(locked_obj, features),
                }
            baseline_scores = baseline_score_cache[task_idx]
            baseline_stability = baseline_stability_cache[task_idx]

            for mechanism_idx, mechanism in enumerate(mechanisms):
                if mechanism != "layer_hook_injection":
                    continue
                for mode_idx, mode in enumerate(MODES):
                    cell_key = f"{target_id}:{mechanism}:{mode}"
                    if cell_verdicts.get(cell_key) == "DESTABILIZING":
                        continue
                    for alpha, condition, random_control_idx in condition_rows():
                        key = (suite_row.get("suite_index"), target_id, mechanism, mode, alpha, condition, random_control_idx)
                        if key in done_keys:
                            continue
                        if intervention_passes >= MAX_TOTAL_INTERVENTION_FORWARD_PASSES:
                            break
                        base_row = {
                            "task_suite_index": suite_row.get("suite_index"),
                            "task_id": suite_row["task_id"],
                            "domain": suite_row["domain"],
                            "target_id": target_id,
                            "mechanism": mechanism,
                            "intervention_mode": mode,
                            "alpha": alpha,
                            "condition": condition,
                            "random_control_idx": random_control_idx,
                            "random_control_n_for_target": 1,
                            "direction_source_config": targets[target_id]["config"],
                            "direction_source_head_id": targets[target_id]["head_id"],
                            "target_layer": targets[target_id]["layer"],
                            "latent_direction_layer_mismatch": None,
                            "cache_intervention_mode": "disabled",
                            "baseline_score": baseline_scores["target"],
                            "baseline_std": target_stds[target_id],
                            "baseline_diagnostic_d1_score": baseline_scores["diagnostic_d1"],
                            "baseline_hh_general_score": baseline_scores["hh_general"],
                            "baseline_objective_mixed_score": baseline_scores["objective_mixed"],
                            "zero_alpha_equivalence_passed": preflight.get("layer_hook_smoke", {}).get("zero_alpha_equivalence"),
                            "prefix_length": prefix_length,
                            "partial_trajectory_branch": branch_id,
                        }
                        if condition == "zero_baseline":
                            eval_result = baseline.get("evaluation") or evaluate_output(task, baseline_cont)
                            row = {
                                **base_row,
                                "direction_norm": 0.0,
                                "post_intervention_score": baseline_scores["target"],
                                "z_score_change": 0.0,
                                "diagnostic_d1_score": baseline_scores["diagnostic_d1"],
                                "diagnostic_d1_z_change": 0.0,
                                "hh_general_score": baseline_scores["hh_general"],
                                "objective_mixed_score": baseline_scores["objective_mixed"],
                                "activation_rms_change": 0.0,
                                "per_loop_activation_rms_change": {},
                                "parse_rate": baseline_stability["parse_rate"],
                                "repetition_rate": baseline_stability["repetition_rate"],
                                "output_length": baseline_stability["output_length"],
                                "hit_max_tokens": baseline_stability["hit_max_tokens"],
                                "empty_output": baseline_stability["empty_output"],
                                "cuda_error": False,
                                "nan_or_inf_activations": False,
                                "final_completion_text": baseline_cont,
                                "parsed_answer": eval_result.get("parsed_answer"),
                                "is_correct": bool(eval_result.get("success")),
                            }
                            rows.append(row)
                            done_keys.add(key)
                            continue

                        if condition == "positive":
                            direction = target_dirs[target_id]
                        elif condition == "negative":
                            direction = -target_dirs[target_id]
                        else:
                            random_seed = (
                                int(target_id[1:]) * 1_000_000
                                + mechanism_idx * 100_000
                                + mode_idx * 10_000
                                + task_idx * 100
                                + random_control_idx
                            )
                            direction = random_direction(int(target_dirs[target_id].numel()), random_seed)
                        mode_spec = build_intervention_mode(mode, alpha)
                        hook = BGLayerHookSteering(
                            extractor.model,
                            target_layer=int(targets[target_id]["layer"]),
                            target_loops=mode_spec["target_loops"],
                            loop_alpha_scales=mode_spec["loop_alpha_scales"],
                            direction=direction,
                            alpha=alpha,
                        )
                        prompt = continuation_prompt(task, prefix_text)
                        budget = min(256, continuation_budget(task))
                        seed = generation_seed(int(target_id[1:]), mechanism_idx, mode_idx, task_idx, alpha, condition)
                        cuda_error = False
                        error_text = ""
                        try:
                            completion, gen_meta = generate_with_layer_hook(
                                extractor,
                                prompt,
                                hook,
                                max_new_tokens=budget,
                                seed=seed,
                            )
                            candidate_post = prefix_text + first_n_tokens_text(
                                extractor.tokenizer, completion, POST_INTERVENTION_TOKENS
                            )
                            post_features = capture_bg_features_with_model(
                                extractor.model,
                                extractor.tokenizer,
                                format_prompt_candidate(str(task.get("prompt") or task.get("question") or ""), candidate_post),
                                device=extractor.device,
                            )
                            eval_result = evaluate_output(task, completion)
                        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                            if "cuda" in str(exc).lower() or "out of memory" in str(exc).lower():
                                cuda_error = True
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            else:
                                cuda_error = False
                            completion = ""
                            gen_meta = {"token_count": 0, "hit_max_tokens": False, "seconds": 0.0}
                            post_features = baseline_feature_cache[task_idx]
                            eval_result = {"parsed": False, "success": False, "parsed_answer": None}
                            error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:]
                            if not cuda_error:
                                print(f"condition error {target_id} {mode} {alpha} {condition}: {str(exc)[:160]}")
                        target_score = raw_head_score(target_heads[target_id], post_features)
                        d1_score = raw_head_score(d1_head, post_features) if d1_head else 0.0
                        hh_score = raw_head_score(locked_hh, post_features)
                        obj_score = raw_head_score(locked_obj, post_features)
                        diag = hook.diagnostics()
                        row = {
                            **base_row,
                            "direction_norm": float(torch.linalg.vector_norm(direction).item()),
                            "post_intervention_score": target_score,
                            "z_score_change": (target_score - baseline_scores["target"]) / max(target_stds[target_id], 1e-8),
                            "diagnostic_d1_score": d1_score,
                            "diagnostic_d1_z_change": (d1_score - baseline_scores["diagnostic_d1"]) / max(d1_std, 1e-8),
                            "hh_general_score": hh_score,
                            "hh_general_z_change": (hh_score - baseline_scores["hh_general"]) / max(hh_std, 1e-8),
                            "objective_mixed_score": obj_score,
                            "objective_mixed_z_change": (obj_score - baseline_scores["objective_mixed"]) / max(obj_std, 1e-8),
                            "activation_rms_change": diag.get("activation_rms_change", 0.0),
                            "per_loop_activation_rms_change": diag.get("per_loop_activation_rms_change", {}),
                            "hook_forward_call_count": diag.get("forward_call_count"),
                            "hook_modifications": diag.get("modifications"),
                            "hook_loop_index_source": diag.get("loop_index_source"),
                            "parse_rate": parse_rate(eval_result),
                            "repetition_rate": repetition_rate(completion),
                            "output_length": int(gen_meta.get("token_count", 0)),
                            "hit_max_tokens": bool(gen_meta.get("hit_max_tokens", False)),
                            "empty_output": not bool(str(completion).strip()),
                            "cuda_error": cuda_error,
                            "nan_or_inf_activations": bool(diag.get("nan_or_inf", False)),
                            "final_completion_text": completion,
                            "parsed_answer": eval_result.get("parsed_answer"),
                            "is_correct": bool(eval_result.get("success")),
                            "generation_metadata": gen_meta,
                            "error": error_text,
                        }
                        if is_destabilizing(row, baseline_stability):
                            row["safety_status"] = "DESTABILIZING"
                            cell_verdicts[cell_key] = "DESTABILIZING"
                        else:
                            row["safety_status"] = "OK"
                            cell_verdicts.setdefault(cell_key, "READY")
                        rows.append(row)
                        done_keys.add(key)
                        intervention_passes += 1
                        write_json(
                            OUT_PARTIAL,
                            {
                                "BG_STAGE2_INTERVENTION_SWEEP_VERDICT": "PARTIAL",
                                "rows": rows,
                                "cell_verdicts": cell_verdicts,
                                "intervention_forward_passes": intervention_passes,
                                "RANDOM_CONTROL_N": 1,
                                "elapsed_seconds": round(time.time() - started, 3),
                            },
                        )
                        if cell_verdicts.get(cell_key) == "DESTABILIZING":
                            break
                    if intervention_passes >= MAX_TOTAL_INTERVENTION_FORWARD_PASSES:
                        break
                if intervention_passes >= MAX_TOTAL_INTERVENTION_FORWARD_PASSES:
                    break
            print(f"stage2 sweep task {task_idx + 1}/{len(tasks)} rows={len(rows)} interventions={intervention_passes}")
    finally:
        if extractor is not None:
            extractor.cleanup()

    evaluable_targets = {row["target_id"] for row in rows if row.get("condition") != "zero_baseline" and not row.get("cuda_error")}
    if len(evaluable_targets) >= 4 and intervention_passes < MAX_TOTAL_INTERVENTION_FORWARD_PASSES:
        verdict = "READY"
    elif len(evaluable_targets) >= 2:
        verdict = "PARTIAL"
    else:
        verdict = "INSUFFICIENT"
    payload = {
        "BG_STAGE2_INTERVENTION_SWEEP_VERDICT": verdict,
        "verdict": verdict,
        "rows": rows,
        "row_count": len(rows),
        "intervention_forward_passes": intervention_passes,
        "max_total_intervention_forward_passes": MAX_TOTAL_INTERVENTION_FORWARD_PASSES,
        "cell_verdicts": cell_verdicts,
        "mechanisms_run": mechanisms,
        "RANDOM_CONTROL_N": 1,
        "RANDOM_CONTROL_LOW_N": True,
        "CACHE_INTERVENTION_MODE": "disabled",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_PARTIAL, payload)
    write_json(OUT_JSON, payload)
    print(f"BG_STAGE2_INTERVENTION_SWEEP_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    return 0 if verdict in {"READY", "PARTIAL", "INSUFFICIENT"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
