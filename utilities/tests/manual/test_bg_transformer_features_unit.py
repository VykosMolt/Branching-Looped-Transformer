"""Unit and shape tests for read-only BG transformer feature integration."""
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

from src.evaluator.bg_controller import BGController, config_vector  # noqa: E402
from src.evaluator.bg_transformer_features import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    BGTransformerFeatureExtractor,
    format_prompt_candidate,
)


OUT_JSON = PROJECT_ROOT / "opi/taps/probes/bg_transformer_features_unit_2026-05-18.json"
OUT_MD = PROJECT_ROOT / "opi/taps/probes/bg_transformer_features_unit_2026-05-18.md"


def write_reports(payload: dict[str, Any]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    lines = [
        "# BG Transformer Feature Unit Tests (2026-05-18)",
        "",
        f"BG_TRANSFORMER_UNIT_TEST_VERDICT = {payload['verdict']}",
        "",
        "## Results",
        "",
    ]
    for name, row in payload["tests"].items():
        lines.append(f"- `{name}`: `{row['status']}`")
        if row.get("detail"):
            lines.append(f"  - {row['detail']}")
    if payload.get("error"):
        lines.extend(["", "## Error", "", "```text", payload["error"], "```"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_test(results: dict[str, dict[str, Any]], name: str, fn) -> None:
    try:
        detail = fn()
        results[name] = {"status": "PASS", "detail": detail or ""}
    except Exception as exc:
        results[name] = {
            "status": "FAIL",
            "detail": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        }


def main() -> int:
    start = time.time()
    tests: dict[str, dict[str, Any]] = {}

    def import_feature_extractor() -> str:
        assert BGTransformerFeatureExtractor is not None
        return str(BGTransformerFeatureExtractor)

    def import_controller() -> str:
        assert BGController is not None
        controller = BGController.from_artifacts(device="cpu")
        assert {"hh_general", "objective_mixed", "code_specialist_backup"}.issubset(controller.heads)
        return "controller artifacts loaded on CPU"

    def formatting() -> str:
        text = format_prompt_candidate("P", "C")
        assert text == "Prompt:\nP\n\nCandidate:\nC"
        assert "score" not in text.lower()
        assert "selected" not in text.lower()
        return repr(text)

    def mock_shape() -> str:
        pooled = torch.zeros((3, 4, 2048), dtype=torch.float32)
        assert config_vector(pooled, "36_L4").shape == (2048,)
        assert config_vector(pooled, "47_concat_L1_L4").shape == (4096,)
        assert config_vector(pooled, "47_concat_all_loops").shape == (8192,)
        return "mock pooled configs OK"

    _run_test(tests, "feature_extractor_import", import_feature_extractor)
    _run_test(tests, "controller_import_and_artifact_load", import_controller)
    _run_test(tests, "feature_input_formatting", formatting)
    _run_test(tests, "mock_pooled_tensor_shape", mock_shape)

    live_status = "SKIP"
    live_detail = ""
    if os.environ.get("BG_SKIP_LIVE_MODEL") == "1":
        live_detail = "BG_SKIP_LIVE_MODEL=1"
    elif not DEFAULT_MODEL_PATH.exists():
        live_detail = f"model path missing: {DEFAULT_MODEL_PATH}"
    elif not torch.cuda.is_available():
        live_detail = "CUDA unavailable; skipped to avoid slow CPU live encode"
    else:
        extractor: BGTransformerFeatureExtractor | None = None
        try:
            extractor = BGTransformerFeatureExtractor(device="cuda", dtype="auto", force_all_loops=True)
            features = extractor.encode_text_to_pooled_features("What is 2+2? Answer briefly.", max_length=64)
            assert list(features.shape) == [3, 4, 2048]
            assert features.dtype == torch.float32
            live_status = "PASS"
            live_detail = f"shape={list(features.shape)} dtype={features.dtype}"
        except Exception as exc:
            live_status = "FAIL"
            live_detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        finally:
            if extractor is not None:
                extractor.cleanup()
    tests["live_encode_text_to_pooled_features"] = {"status": live_status, "detail": live_detail}

    hard_failures = [name for name, row in tests.items() if row["status"] == "FAIL" and name != "live_encode_text_to_pooled_features"]
    live_failed = tests["live_encode_text_to_pooled_features"]["status"] == "FAIL"
    if hard_failures or live_failed:
        verdict = "FAIL"
    elif tests["live_encode_text_to_pooled_features"]["status"] == "PASS":
        verdict = "PASS"
    else:
        verdict = "PARTIAL"

    payload = {
        "verdict": verdict,
        "BG_TRANSFORMER_UNIT_TEST_VERDICT": verdict,
        "tests": tests,
        "elapsed_seconds": round(time.time() - start, 3),
    }
    write_reports(payload)
    print(f"BG_TRANSFORMER_UNIT_TEST_VERDICT = {verdict}")
    print(f"Wrote {OUT_JSON.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(PROJECT_ROOT)}")
    return 0 if verdict in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
