"""Preflight inventory for the BG steering and routing suite."""
from __future__ import annotations

import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from bg_steering_suite_lib import (
    MODEL_PATH,
    PROJECT_ROOT,
    PROBE_ROOT,
    REPORT_ROOT,
    build_task_suite_rows,
    ensure_report_root,
    load_code_tasks,
    load_gsm8k_tasks,
    load_reasoning_tasks,
    load_science_tasks,
    rel,
    write_json,
    write_md,
)


OUT_JSON = REPORT_ROOT / "preflight.json"
OUT_MD = REPORT_ROOT / "preflight.md"


def _exists(path: str) -> dict[str, Any]:
    p = PROJECT_ROOT / path
    return {"path": path, "exists": p.exists(), "size": p.stat().st_size if p.exists() and p.is_file() else None}


def _artifact(path: str) -> dict[str, Any]:
    p = PROJECT_ROOT / path
    info = _exists(path)
    if p.exists() and p.suffix == ".json":
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            rows = data.get("tasks") or data.get("tournaments") or data.get("candidates") or data.get("all_attempts")
            info["keys"] = list(data)[:20] if isinstance(data, dict) else []
            info["row_count"] = len(rows) if isinstance(rows, list) else None
        except Exception as exc:
            info["error"] = str(exc)
    return info


def main() -> int:
    ensure_report_root()
    started = time.time()
    payload: dict[str, Any] = {
        "BG_STEERING_PREFLIGHT_VERDICT": "BLOCKED",
        "verdict": "BLOCKED",
        "started_at": started,
        "warnings": [],
    }
    try:
        from src.evaluator.bg_controller import BGController
        from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor

        controller = BGController.from_artifacts(device="cpu")
        heads_loaded = sorted(controller.heads)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        capture_shape = None
        capture_error = ""
        try:
            with BGTransformerFeatureExtractor(device=device, dtype="auto", force_all_loops=True) as extractor:
                features = extractor.encode_text_to_pooled_features("What is 2+2? Answer briefly.", max_length=64)
                capture_shape = list(features.shape)
        except Exception as exc:
            capture_error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

        wrapper_files = [
            "src/local_agent/ouro_agent_improved.py",
            "src/local_agent/ouro_direct.py",
            "src/local_agent/ouro_backend.py",
            "src/local_agent/ouro_policies.py",
        ]
        wrapper_inventory = [_exists(path) for path in wrapper_files]
        wrapper_candidate_interface = "MISSING"
        if all(row["exists"] for row in wrapper_inventory):
            wrapper_candidate_interface = "FILES_PRESENT_NO_CLEAN_MULTI_CANDIDATE_INTERFACE_CONFIRMED"

        artifacts = [
            "opi/taps/probes/code_branch_taskset_v2_mini_patched_2026-05-16.json",
            "opi/taps/probes/code_branch_taskset_v2_near_miss10_2026-05-17.json",
            "opi/taps/probes/bg_devil_task_offline_dynamic_connectivity_2026-05-18.json",
            "opi/taps/probes/bg_devil_task_minimum_xor_paths_2026-05-18.json",
            "opi/taps/probes/reasoning_branch_pilot_2026-05-17.json",
            "opi/taps/probes/science_natural_distractor_set_2026-05-17.json",
            "opi/taps/probes/clean_gsm8k_expanded_2026-05-16.json",
        ]
        artifact_inventory = [_artifact(path) for path in artifacts]
        code_tasks = load_code_tasks()
        reasoning_tasks = load_reasoning_tasks()
        science_tasks = load_science_tasks()
        gsm_tasks = load_gsm8k_tasks()
        selected_plan = build_task_suite_rows()
        counts = Counter(row["domain"] for row in selected_plan)
        estimated_candidates = len(selected_plan) * 4
        estimated_partial_tokens = sum(4 * (256 if row["domain"] == "code" and row.get("is_devil") else 192 if row["domain"] == "code" else 96 if row["domain"] in {"reasoning", "science"} else 128) for row in selected_plan)
        ready_sources = bool(code_tasks) and (bool(reasoning_tasks) or bool(science_tasks)) and bool(gsm_tasks)
        capture_ready = capture_shape == [3, 4, 2048]
        if capture_ready and ready_sources:
            verdict = "READY"
        elif capture_ready:
            verdict = "PARTIAL"
        else:
            verdict = "BLOCKED"

        payload.update(
            {
                "BG_STEERING_PREFLIGHT_VERDICT": verdict,
                "verdict": verdict,
                "controller_import": "OK",
                "transformer_feature_import": "OK",
                "model_path": str(MODEL_PATH),
                "model_path_exists": MODEL_PATH.exists(),
                "controller_heads_loaded": heads_loaded,
                "live_capture_shape": capture_shape,
                "live_capture_error": capture_error,
                "wrapper_inventory": wrapper_inventory,
                "wrapper_candidate_interface": wrapper_candidate_interface,
                "artifact_inventory": artifact_inventory,
                "task_source_counts": {
                    "code": len(code_tasks),
                    "reasoning": len(reasoning_tasks),
                    "science": len(science_tasks),
                    "gsm8k": len(gsm_tasks),
                },
                "execution_plan": {
                    "tasks_selected_target": len(selected_plan),
                    "tasks_selected_by_domain": dict(counts),
                    "arms": [
                        "baseline_shared_pool",
                        "BG_finished_branch_selection",
                        "BG_partial_trajectory_routing",
                        "BG_compute_allocation",
                        "wrapper_matched_if_clean_interface",
                        "guarded_soft_hidden_steering_pilot",
                        "text_prefix_branch_selection",
                    ],
                    "estimated_initial_candidate_count": estimated_candidates,
                    "estimated_initial_partial_token_budget": estimated_partial_tokens,
                    "wrapper_arms_available": wrapper_candidate_interface == "CLEAN_MULTI_CANDIDATE_INTERFACE",
                },
                "elapsed_seconds": round(time.time() - started, 3),
            }
        )
    except Exception as exc:
        payload.update(
            {
                "BG_STEERING_PREFLIGHT_VERDICT": "BLOCKED",
                "verdict": "BLOCKED",
                "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                "elapsed_seconds": round(time.time() - started, 3),
            }
        )

    write_json(OUT_JSON, payload)
    lines = [
        "# BG Steering Suite Preflight (2026-05-18)",
        "",
        f"BG_STEERING_PREFLIGHT_VERDICT = {payload['BG_STEERING_PREFLIGHT_VERDICT']}",
        "",
        f"- Controller heads: `{payload.get('controller_heads_loaded')}`",
        f"- Live capture shape: `{payload.get('live_capture_shape')}`",
        f"- Model path exists: `{payload.get('model_path_exists')}`",
        f"- Wrapper candidate interface: `{payload.get('wrapper_candidate_interface')}`",
        f"- Task source counts: `{payload.get('task_source_counts')}`",
        f"- Execution plan: `{payload.get('execution_plan')}`",
        "",
        "## Artifacts",
        "",
    ]
    for row in payload.get("artifact_inventory", []):
        lines.append(f"- `{row['path']}` exists=`{row['exists']}` rows=`{row.get('row_count')}`")
    if payload.get("error"):
        lines.extend(["", "## Error", "", "```text", payload["error"], "```"])
    write_md(OUT_MD, lines)
    print(f"BG_STEERING_PREFLIGHT_VERDICT = {payload['BG_STEERING_PREFLIGHT_VERDICT']}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if payload["BG_STEERING_PREFLIGHT_VERDICT"] in {"READY", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
