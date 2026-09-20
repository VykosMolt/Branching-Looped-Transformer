"""Inspect live Ouro-RLTT hidden-state capture for BG transformer integration."""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator.bg_transformer_features import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    HIDDEN_DIM,
    NUM_LOOPS,
    TAP_LAYERS,
    BGTransformerFeatureExtractor,
)


OUT_JSON = PROJECT_ROOT / "opi/taps/probes/bg_transformer_capture_inspection_2026-05-18.json"
OUT_MD = PROJECT_ROOT / "opi/taps/probes/bg_transformer_capture_inspection_2026-05-18.md"
PROMPT = "What is 2+2? Answer briefly."


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    return str(value)


def _model_output_structure(obj: Any) -> dict[str, Any]:
    info: dict[str, Any] = {"type": type(obj).__name__}
    for attr in ("logits", "per_loop_hidden_states", "exit_pdf", "hidden_states"):
        value = getattr(obj, attr, None)
        if value is None:
            info[attr] = None
        elif isinstance(value, torch.Tensor):
            info[attr] = {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
        elif isinstance(value, (list, tuple)):
            rows = []
            for item in value[:6]:
                if isinstance(item, torch.Tensor):
                    rows.append({"shape": list(item.shape), "dtype": str(item.dtype), "device": str(item.device)})
                else:
                    rows.append(type(item).__name__)
            info[attr] = {"type": type(value).__name__, "length": len(value), "items": rows}
        else:
            info[attr] = type(value).__name__
    return info


def write_reports(payload: dict[str, Any]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")

    lines = [
        "# BG Transformer Capture Inspection (2026-05-18)",
        "",
        f"BG_TRANSFORMER_CAPTURE_INSPECTION_VERDICT = {payload['verdict']}",
        "",
        "## Summary",
        "",
        f"- Model path: `{payload.get('model_path')}`",
        f"- Model class: `{payload.get('model_class')}`",
        f"- Tokenizer class: `{payload.get('tokenizer_class')}`",
        f"- Device: `{payload.get('device')}`",
        f"- Dtype: `{payload.get('dtype')}`",
        f"- Forced all loops: `{payload.get('force_all_loops')}`",
        f"- Early exit threshold after load: `{payload.get('early_exit_threshold')}`",
        f"- Total UT steps after load: `{payload.get('total_ut_steps')}`",
        f"- Feature shape: `{payload.get('feature_shape')}`",
        "",
        "## Loop And Layer Access",
        "",
    ]
    capture = payload.get("capture", {})
    lines.append(f"- Boundary loop count: `{capture.get('boundary_loop_count')}`")
    lines.append(f"- Layer loop counts: `{capture.get('layer_loop_counts')}`")
    lines.append(f"- Required layers accessible: `{payload.get('required_layers_accessible')}`")
    lines.append("")
    lines.append("## Hidden-State Structure")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(payload.get("model_output_structure", {}), indent=2, sort_keys=True))
    lines.append("```")
    if payload.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in payload["warnings"])
    if payload.get("error"):
        lines.extend(["", "## Error", "", "```text", payload["error"], "```"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload: dict[str, Any] = {
        "verdict": "BLOCKED",
        "prompt": PROMPT,
        "model_path": str(DEFAULT_MODEL_PATH),
        "tokenizer_path": str(DEFAULT_MODEL_PATH),
        "trust_remote_code_needed": True,
        "device": device,
        "warnings": [],
    }
    extractor: BGTransformerFeatureExtractor | None = None
    try:
        extractor = BGTransformerFeatureExtractor(
            model_path=DEFAULT_MODEL_PATH,
            device=device,
            dtype="auto",
            force_all_loops=True,
        )
        features, capture_summary = extractor.inspect_capture_once(PROMPT, max_length=128)

        enc = extractor.tokenizer(PROMPT, return_tensors="pt", truncation=True, max_length=128)
        enc = {k: v.to(extractor.device) for k, v in enc.items()}
        with torch.inference_mode():
            output = extractor.model(
                **enc,
                use_cache=False,
                return_per_loop_hidden_states=True,
                return_exit_pdf=True,
                return_dict=True,
            )

        feature_shape = list(features.shape)
        capture = {
            "boundary_loop_count": capture_summary.boundary_loop_count,
            "layer_loop_counts": capture_summary.layer_loop_counts,
            "shapes_by_layer": capture_summary.shapes_by_layer,
            "dtype_by_layer": capture_summary.dtype_by_layer,
            "device_by_layer": capture_summary.device_by_layer,
        }
        required_layers_accessible = (
            capture_summary.boundary_loop_count == NUM_LOOPS
            and all(capture_summary.layer_loop_counts.get(layer) == NUM_LOOPS for layer in (24, 36))
            and feature_shape == [len(TAP_LAYERS), NUM_LOOPS, HIDDEN_DIM]
        )
        verdict = "READY" if required_layers_accessible else "PARTIAL"
        payload.update(
            {
                "verdict": verdict,
                "model_class": type(extractor.model).__name__,
                "inner_model_class": type(getattr(extractor.model, "model", None)).__name__,
                "tokenizer_class": type(extractor.tokenizer).__name__,
                "dtype": str(extractor.dtype),
                "force_all_loops": extractor.force_all_loops,
                "early_exit_threshold": getattr(extractor.model.config, "early_exit_threshold", None),
                "total_ut_steps": getattr(extractor.model.config, "total_ut_steps", None),
                "inner_total_ut_steps": getattr(getattr(extractor.model, "model", None), "total_ut_steps", None),
                "feature_shape": feature_shape,
                "feature_dtype": str(features.dtype),
                "capture": capture,
                "required_layers_accessible": required_layers_accessible,
                "model_output_structure": _model_output_structure(output),
                "generated_tokens": 0,
                "elapsed_seconds": round(time.time() - start, 3),
            }
        )
    except Exception as exc:
        payload.update(
            {
                "verdict": "BLOCKED",
                "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                "elapsed_seconds": round(time.time() - start, 3),
            }
        )
    finally:
        if extractor is not None:
            extractor.cleanup()
        write_reports(payload)

    print(f"BG_TRANSFORMER_CAPTURE_INSPECTION_VERDICT = {payload['verdict']}")
    print(f"Wrote {OUT_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(PROJECT_ROOT)}")
    return 0 if payload["verdict"] in {"READY", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
