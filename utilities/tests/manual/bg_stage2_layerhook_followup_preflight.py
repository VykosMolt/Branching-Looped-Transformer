"""Preflight for the narrowed BG Stage 2 layer-hook follow-up."""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
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

from bg_stage2_steering_preflight import (  # noqa: E402
    find_diagnostic_d1,
    head_weight_vector,
    layer_from_config,
    load_head_row,
    select_nonorm_target,
)
from src.evaluator.bg_controller import BGController  # noqa: E402
from src.evaluator.bg_steering_hook import BGLayerHookSteering, build_intervention_mode  # noqa: E402
from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor  # noqa: E402


OUT_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_stage2_layerhook_followup_2026-05-18"
STAGE1_ROOT = PROJECT_ROOT / "opi/taps/probes/bg_trajectory_prediction_2026-05-18"
MODEL_PATH = PROJECT_ROOT / "shared/models/ouro_rltt_local"
OUT_JSON = OUT_ROOT / "preflight.json"
OUT_MD = OUT_ROOT / "preflight.md"
SEED = 20260518


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(p)


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


def deterministic_generate_ids(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int, hook: BGLayerHookSteering | None = None) -> torch.Tensor:
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256, padding=False)
    enc = {key: value.to(device) for key, value in enc.items()}
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    if hook is not None:
        hook.apply(position=-1)
    try:
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )
    finally:
        if hook is not None:
            hook.remove()
    return out.detach().cpu()


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    payload: dict[str, Any] = {
        "BG_LAYERHOOK_FOLLOWUP_PREFLIGHT_VERDICT": "BLOCKED",
        "errors": [],
        "warnings": [],
    }
    extractor: BGTransformerFeatureExtractor | None = None
    try:
        controller = BGController.from_artifacts(device="cpu")
        controller_heads_loaded = sorted(controller.heads)
        from src.evaluator import bg_steering_hook as hook_module  # noqa: F401

        predictive = load_json(STAGE1_ROOT / "predictive_power.json", {})
        cells = list(predictive.get("all_cells") or [])
        selected = select_nonorm_target(cells, "reasoning", 64)
        d1 = find_diagnostic_d1(cells)
        if selected is None:
            payload.update({"blocker": "missing T1 reasoning@64 NoNorm cell"})
            raise RuntimeError(payload["blocker"])
        target_head = load_head_row(str(selected["head_id"]), controller)
        direction = head_weight_vector(target_head)
        if int(direction.numel()) != 2048:
            raise RuntimeError(f"T1 direction has dim {direction.numel()}, expected 2048")
        direction_norm = float(torch.linalg.vector_norm(direction).item())
        target_layer = layer_from_config(str(selected["config"]))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        extractor = BGTransformerFeatureExtractor(device=device, dtype="auto", force_all_loops=True)
        prompt = "Answer with exactly one token if possible. What is 2 + 2?"
        mode_spec = build_intervention_mode("single_loop_L1", 0.0)
        hook = BGLayerHookSteering(
            extractor.model,
            target_layer=target_layer,
            target_loops=mode_spec["target_loops"],
            loop_alpha_scales=mode_spec["loop_alpha_scales"],
            direction=direction,
            alpha=0.0,
        )
        no_hook = deterministic_generate_ids(extractor.model, extractor.tokenizer, prompt, 8, hook=None)
        zero_hook = deterministic_generate_ids(extractor.model, extractor.tokenizer, prompt, 8, hook=hook)
        zero_alpha_equivalence = bool(torch.equal(no_hook, zero_hook))
        current_ut_available = hook.loop_index_source == "current_ut"
        if not zero_alpha_equivalence:
            verdict = "BLOCKED"
        elif current_ut_available and d1 is not None:
            verdict = "READY"
        else:
            verdict = "PARTIAL"
        payload.update(
            {
                "BG_LAYERHOOK_FOLLOWUP_PREFLIGHT_VERDICT": verdict,
                "controller_load": "OK",
                "transformer_feature_extractor_load": "OK",
                "model_path": rel(MODEL_PATH),
                "model_path_exists": MODEL_PATH.exists(),
                "hook_module_import": "OK",
                "use_cache_for_intervention_generation": False,
                "zero_alpha_hook_equivalence": zero_alpha_equivalence,
                "zero_alpha_no_hook_ids": no_hook[0].tolist(),
                "zero_alpha_hook_ids": zero_hook[0].tolist(),
                "hook_loop_index_source": hook.loop_index_source,
                "current_ut_loop_identity": current_ut_available,
                "target": {
                    "target_id": "T1",
                    "domain": "reasoning",
                    "prefix_length": 64,
                    "head_id": selected["head_id"],
                    "family": selected.get("family"),
                    "config": selected.get("config"),
                    "architecture": selected.get("architecture"),
                    "layer": target_layer,
                    "top1_lift": selected.get("top1_lift"),
                    "pairwise_accuracy": selected.get("pairwise_predictive_accuracy"),
                    "oracle_success": selected.get("oracle_success"),
                    "direction_norm": direction_norm,
                    "direction_dim": int(direction.numel()),
                },
                "diagnostic_d1": d1 or "MISSING",
                "diagnostic_d1_status": "READY" if d1 is not None else "MISSING",
                "controller_heads_loaded": controller_heads_loaded,
                "elapsed_seconds": round(time.time() - started, 3),
            }
        )
    except Exception as exc:
        payload.setdefault("blocker", str(exc))
        payload.update(
            {
                "BG_LAYERHOOK_FOLLOWUP_PREFLIGHT_VERDICT": "BLOCKED",
                "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:],
                "elapsed_seconds": round(time.time() - started, 3),
            }
        )
    finally:
        if extractor is not None:
            extractor.cleanup()

    write_json(OUT_JSON, payload)
    lines = [
        "# BG Stage 2 Layer-Hook Follow-Up Preflight",
        "",
        f"BG_LAYERHOOK_FOLLOWUP_PREFLIGHT_VERDICT = {payload['BG_LAYERHOOK_FOLLOWUP_PREFLIGHT_VERDICT']}",
        "",
        f"- model_path_exists: `{payload.get('model_path_exists')}`",
        f"- hook_module_import: `{payload.get('hook_module_import')}`",
        f"- use_cache_for_intervention_generation: `{payload.get('use_cache_for_intervention_generation')}`",
        f"- zero_alpha_hook_equivalence: `{payload.get('zero_alpha_hook_equivalence')}`",
        f"- hook_loop_index_source: `{payload.get('hook_loop_index_source')}`",
        f"- current_ut_loop_identity: `{payload.get('current_ut_loop_identity')}`",
        f"- diagnostic_d1_status: `{payload.get('diagnostic_d1_status')}`",
        "",
        "## T1 Target",
        "",
        f"`{payload.get('target')}`",
    ]
    if payload.get("error"):
        lines.extend(["", "## Error", "", "```text", payload["error"], "```"])
    write_md(OUT_MD, lines)
    print(f"BG_LAYERHOOK_FOLLOWUP_PREFLIGHT_VERDICT = {payload['BG_LAYERHOOK_FOLLOWUP_PREFLIGHT_VERDICT']}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if payload["BG_LAYERHOOK_FOLLOWUP_PREFLIGHT_VERDICT"] in {"READY", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
