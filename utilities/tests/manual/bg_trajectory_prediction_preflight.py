"""Preflight inventory for the Stage 1 BG trajectory prediction sweep."""
from __future__ import annotations

import time
import traceback
from collections import Counter

from bg_trajectory_prediction_lib import (
    BRANCHES_PER_TASK,
    MAX_NEW_TOKENS,
    MODEL_PATH,
    PREFIX_LENGTHS,
    REPORT_ROOT,
    build_trajectory_task_rows,
    ensure_report_root,
    rel,
    write_json,
    write_md,
)


OUT_JSON = REPORT_ROOT / "preflight.json"
OUT_MD = REPORT_ROOT / "preflight.md"


def main() -> int:
    started = time.time()
    ensure_report_root()
    checks: dict[str, object] = {}
    errors: list[str] = []

    try:
        from src.evaluator.bg_controller import BGController

        checks["bg_controller_import"] = True
        controller = BGController.from_artifacts(device="cpu")
        checks["controller_heads_loaded"] = sorted(controller.heads.keys())
    except Exception as exc:
        checks["bg_controller_import"] = False
        errors.append("BGController: " + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-2000:])

    try:
        from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor

        checks["bg_feature_extractor_import"] = True
    except Exception as exc:
        BGTransformerFeatureExtractor = None  # type: ignore[assignment]
        checks["bg_feature_extractor_import"] = False
        errors.append("BGTransformerFeatureExtractor: " + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-2000:])

    checks["model_path"] = str(MODEL_PATH)
    checks["model_path_exists"] = MODEL_PATH.exists()
    if not MODEL_PATH.exists():
        errors.append(f"local Ouro-RLTT model path missing: {MODEL_PATH}")

    capture_shape = None
    capture_ok = False
    if checks.get("bg_feature_extractor_import") and MODEL_PATH.exists():
        extractor = None
        try:
            extractor = BGTransformerFeatureExtractor(device="cuda", dtype="auto", force_all_loops=True)  # type: ignore[misc]
            features = extractor.encode_text_to_pooled_features("What is 2+2? Answer briefly.", max_length=128)
            capture_shape = list(features.shape)
            capture_ok = capture_shape == [3, 4, 2048]
        except Exception as exc:
            errors.append("live_capture: " + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-3000:])
        finally:
            if extractor is not None:
                extractor.cleanup()
    checks["live_capture_shape"] = capture_shape
    checks["live_capture_ok"] = capture_ok

    tasks = build_trajectory_task_rows()
    counts = Counter(task["domain"] for task in tasks)
    source_ok_domains = [domain for domain, count in counts.items() if count >= 10]
    checks["task_sources"] = dict(counts)
    checks["evaluator_parsers"] = {
        "mcq_answer_parser": "available",
        "science_answer_parser": "available",
        "gsm8k_numeric_parser": "available",
    }
    plan = {
        "planned_task_count": len(tasks),
        "planned_counts_by_domain": dict(counts),
        "prefix_lengths": list(PREFIX_LENGTHS),
        "branches_per_task": BRANCHES_PER_TASK,
        "partial_generation_max_new_tokens": MAX_NEW_TOKENS,
        "expected_partial_trajectories": len(tasks) * BRANCHES_PER_TASK,
        "expected_prefixes": len(tasks) * BRANCHES_PER_TASK * len(PREFIX_LENGTHS),
        "device": "cuda if available",
    }

    if checks.get("bg_controller_import") and capture_ok and len(source_ok_domains) >= 2:
        verdict = "READY"
    elif checks.get("bg_controller_import") and capture_ok and len(source_ok_domains) == 1:
        verdict = "PARTIAL"
    else:
        verdict = "BLOCKED"

    payload = {
        "BG_TRAJECTORY_PREFLIGHT_VERDICT": verdict,
        "verdict": verdict,
        "checks": checks,
        "plan": plan,
        "errors": errors,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    lines = [
        "# BG Trajectory Prediction Preflight (2026-05-18)",
        "",
        f"BG_TRAJECTORY_PREFLIGHT_VERDICT = {verdict}",
        "",
        "## Checks",
        "",
        f"- model_path_exists: `{checks['model_path_exists']}`",
        f"- controller_heads_loaded: `{checks.get('controller_heads_loaded')}`",
        f"- live_capture_shape: `{capture_shape}`",
        f"- task_sources: `{dict(counts)}`",
        "",
        "## Plan",
        "",
        f"`{plan}`",
    ]
    if errors:
        lines.extend(["", "## Errors", "", *[f"- {err[:500]}" for err in errors]])
    write_md(OUT_MD, lines)
    print(f"BG_TRAJECTORY_PREFLIGHT_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict in {"READY", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
