"""Preflight for BG empirical steering-direction follow-up."""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from bg_empirical_steering_common import (
    MODEL_PATH,
    OUT_ROOT,
    PROJECT_ROOT,
    STAGE1_REQUIRED,
    STAGE1_ROOT,
    STAGE2_LAYERHOOK_ROOT,
    STAGE2_REQUIRED,
    TARGET_REQUESTS,
    contrast_counts,
    find_diagnostic_d1,
    layer_from_config,
    rel,
    select_best_nonorm_cell,
    write_json,
    write_md,
)
from bg_stage2_steering_preflight import deterministic_generate_ids, head_weight_vector, load_head_row
from src.evaluator.bg_controller import BGController
from src.evaluator.bg_steering_hook import BGLayerHookSteering, build_intervention_mode
from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor


OUT_JSON = OUT_ROOT / "preflight.json"
OUT_MD = OUT_ROOT / "preflight.md"


def artifact_rows(root: Path, names: list[str]) -> list[dict[str, Any]]:
    rows = []
    for name in names:
        path = root / name
        rows.append({"path": rel(path), "exists": path.exists(), "size": path.stat().st_size if path.exists() else None})
    return rows


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    payload: dict[str, Any] = {
        "BG_EMPIRICAL_STEERING_PREFLIGHT_VERDICT": "BLOCKED",
        "errors": [],
        "warnings": [],
    }
    extractor: BGTransformerFeatureExtractor | None = None
    try:
        stage1_artifacts = artifact_rows(STAGE1_ROOT, STAGE1_REQUIRED)
        stage2_artifacts = artifact_rows(STAGE2_LAYERHOOK_ROOT, STAGE2_REQUIRED)
        missing = [row["path"] for row in stage1_artifacts + stage2_artifacts if not row["exists"]]
        payload["stage1_artifacts"] = stage1_artifacts
        payload["stage2_layerhook_artifacts"] = stage2_artifacts
        payload["model_path"] = rel(MODEL_PATH)
        payload["model_path_exists"] = MODEL_PATH.exists()
        if missing or not MODEL_PATH.exists():
            payload["blocker"] = f"missing required artifacts/model: {missing}"
            write_json(OUT_JSON, payload)
            write_md(OUT_MD, ["# BG Empirical Steering Preflight", "", "BG_EMPIRICAL_STEERING_PREFLIGHT_VERDICT = BLOCKED", "", f"- blocker: `{payload['blocker']}`"])
            print("BG_EMPIRICAL_STEERING_PREFLIGHT_VERDICT = BLOCKED")
            return 1

        controller = BGController.from_artifacts(device="cpu")
        payload["controller_load"] = "OK"
        payload["controller_heads_loaded"] = sorted(controller.heads.keys())
        payload["transformer_feature_extractor_import"] = "OK"
        payload["bg_steering_hook_import"] = "OK"

        target_rows = []
        selected_targets = []
        for request in TARGET_REQUESTS:
            cell = select_best_nonorm_cell(str(request["domain"]), int(request["prefix_length"]))
            if not cell:
                target_rows.append({**request, "status": "MISSING_NONORM_CELL"})
                continue
            counts = contrast_counts(str(request["domain"]), int(request["prefix_length"]), str(cell["config"]))
            row = {
                **request,
                "status": "READY" if counts["meets_minimum"] else "INSUFFICIENT_CONTRAST",
                "head_id": cell["head_id"],
                "family": cell.get("family"),
                "config": cell["config"],
                "architecture": cell["architecture"],
                "layer": layer_from_config(str(cell["config"])),
                "top1_lift": cell.get("top1_lift"),
                "pairwise_accuracy": cell.get("pairwise_predictive_accuracy"),
                "oracle_success": cell.get("oracle_success"),
                "contrast": counts,
            }
            target_rows.append(row)
            if counts["meets_minimum"] and (request.get("primary") or len(selected_targets) < 3):
                selected_targets.append(row)
        payload["targets"] = target_rows
        payload["selected_targets"] = selected_targets
        primary_ready = any(row.get("target_id") == "T1" and row.get("status") == "READY" for row in target_rows)
        if not primary_ready:
            payload["blocker"] = "primary target reasoning@64 lacks minimum success/failure contrast"
            write_json(OUT_JSON, payload)
            write_md(OUT_MD, ["# BG Empirical Steering Preflight", "", "BG_EMPIRICAL_STEERING_PREFLIGHT_VERDICT = BLOCKED", "", f"- blocker: `{payload['blocker']}`"])
            print("BG_EMPIRICAL_STEERING_PREFLIGHT_VERDICT = BLOCKED")
            return 1

        primary = next(row for row in selected_targets if row["target_id"] == "T1")
        primary_head = load_head_row(str(primary["head_id"]), controller)
        primary_direction = head_weight_vector(primary_head)
        payload["primary_direction_dim"] = int(primary_direction.numel())
        payload["primary_direction_norm"] = float(torch.linalg.vector_norm(primary_direction).item())

        extractor = BGTransformerFeatureExtractor(device="cuda" if torch.cuda.is_available() else "cpu", dtype="auto", force_all_loops=True)
        mode_spec = build_intervention_mode("single_loop_L1", 0.0)
        hook = BGLayerHookSteering(
            extractor.model,
            target_layer=int(primary["layer"]),
            target_loops=mode_spec["target_loops"],
            loop_alpha_scales=mode_spec["loop_alpha_scales"],
            direction=primary_direction,
            alpha=0.0,
        )
        prompt = "Answer with exactly one letter. Question: Which option is the letter A? Options: A. A B. B C. C D. D\nFINAL ANSWER:"
        no_hook = deterministic_generate_ids(extractor.model, extractor.tokenizer, prompt, max_new_tokens=16, hook=None)
        with_hook = deterministic_generate_ids(extractor.model, extractor.tokenizer, prompt, max_new_tokens=16, hook=hook)
        zero_alpha_ok = bool(torch.equal(no_hook, with_hook))
        diag = hook.diagnostics()
        payload["zero_alpha_hook_equivalence"] = zero_alpha_ok
        payload["zero_alpha_no_hook_ids"] = no_hook.flatten().tolist()
        payload["zero_alpha_hook_ids"] = with_hook.flatten().tolist()
        payload["hook_loop_index_source"] = diag.get("loop_index_source")
        payload["current_ut_loop_identity"] = diag.get("loop_index_source") == "current_ut"
        payload["use_cache_for_intervention_generation"] = False
        payload["diagnostic_d1"] = find_diagnostic_d1()
        payload["diagnostic_d1_status"] = "READY" if payload["diagnostic_d1"] else "MISSING"

        if not zero_alpha_ok:
            verdict = "BLOCKED"
            payload["blocker"] = "zero-alpha layer-hook generation differs from no-hook generation"
        elif payload["current_ut_loop_identity"] and len(selected_targets) >= 1:
            verdict = "READY"
        else:
            verdict = "PARTIAL"
            payload["warnings"].append("layer hook works but current_ut identity or secondary contrast is partial")
        payload["BG_EMPIRICAL_STEERING_PREFLIGHT_VERDICT"] = verdict
        payload["elapsed_seconds"] = round(time.time() - started, 3)
        write_json(OUT_JSON, payload)

        lines = [
            "# BG Empirical Steering Direction Preflight",
            "",
            f"BG_EMPIRICAL_STEERING_PREFLIGHT_VERDICT = {verdict}",
            "",
            f"- zero_alpha_hook_equivalence: `{zero_alpha_ok}`",
            f"- hook_loop_index_source: `{payload.get('hook_loop_index_source')}`",
            f"- use_cache_for_intervention_generation: `False`",
            "",
            "| target | domain | prefix | config | success | failure | status |",
            "|---|---|---:|---|---:|---:|---|",
        ]
        for row in target_rows:
            c = row.get("contrast") or {}
            lines.append(
                f"| `{row.get('target_id')}` | `{row.get('domain')}` | {row.get('prefix_length')} | "
                f"`{row.get('config', '')}` | {c.get('successful_prefixes', '')} | {c.get('failed_prefixes', '')} | `{row.get('status')}` |"
            )
        write_md(OUT_MD, lines)
        print(f"BG_EMPIRICAL_STEERING_PREFLIGHT_VERDICT = {verdict}")
        print(f"Wrote {rel(OUT_JSON)}")
        print(f"Wrote {rel(OUT_MD)}")
        return 0 if verdict in {"READY", "PARTIAL"} else 1
    except Exception as exc:
        payload["errors"].append("".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:])
        payload["elapsed_seconds"] = round(time.time() - started, 3)
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# BG Empirical Steering Preflight", "", "BG_EMPIRICAL_STEERING_PREFLIGHT_VERDICT = BLOCKED", "", "```", payload["errors"][-1], "```"])
        print("BG_EMPIRICAL_STEERING_PREFLIGHT_VERDICT = BLOCKED")
        return 1
    finally:
        if extractor is not None:
            extractor.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
