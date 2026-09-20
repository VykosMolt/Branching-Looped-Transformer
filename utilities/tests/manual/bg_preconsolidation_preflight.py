"""Preflight and prior-result consolidation for BG pre-consolidation probes."""
from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any

import torch

from bg_empirical_steering_common import contrast_counts, find_diagnostic_d1
from bg_preconsolidation_common import (
    EMPIRICAL_ROOT,
    MODEL_PATH,
    OUT_ROOT,
    STAGE1_ROOT,
    STAGE2_LAYERHOOK_ROOT,
    rel,
    select_targets,
    target_directions,
    write_json,
    write_md,
)
from bg_stage2_steering_preflight import deterministic_generate_ids, head_weight_vector, load_head_row
from src.evaluator.bg_controller import BGController
from src.evaluator.bg_steering_hook import BGLayerHookSteering, build_intervention_mode
from src.evaluator.bg_transformer_features import BGTransformerFeatureExtractor


OUT_JSON = OUT_ROOT / "preflight.json"
OUT_MD = OUT_ROOT / "preflight.md"

REQUIRED_STAGE1 = [
    "predictive_power.json",
    "prefix_features.pt",
    "prefix_scores.json",
    "continued_prefixes.json",
    "task_suite.json",
]


def artifact_rows(root: Path, names: list[str]) -> list[dict[str, Any]]:
    rows = []
    for name in names:
        path = root / name
        rows.append({"path": rel(path), "exists": path.exists(), "size": path.stat().st_size if path.exists() else None})
    return rows


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    payload: dict[str, Any] = {"BG_PRECONSOLIDATION_PREFLIGHT_VERDICT": "BLOCKED", "errors": [], "warnings": []}
    extractor: BGTransformerFeatureExtractor | None = None
    try:
        stage1_artifacts = artifact_rows(STAGE1_ROOT, REQUIRED_STAGE1)
        missing = [row["path"] for row in stage1_artifacts if not row["exists"]]
        payload["stage1_artifacts"] = stage1_artifacts
        payload["model_path"] = rel(MODEL_PATH)
        payload["model_path_exists"] = MODEL_PATH.exists()
        payload["stage2_layerhook_summary_exists"] = (STAGE2_LAYERHOOK_ROOT / "summary.md").exists()
        payload["stage2_layerhook_analysis_exists"] = (STAGE2_LAYERHOOK_ROOT / "analysis.md").exists()
        payload["empirical_directions_exists"] = (EMPIRICAL_ROOT / "directions.pt").exists()
        payload["empirical_analysis_exists"] = (EMPIRICAL_ROOT / "analysis.md").exists()
        if missing or not MODEL_PATH.exists():
            payload["blocker"] = f"missing required artifacts/model: {missing}"
            write_json(OUT_JSON, payload)
            write_md(OUT_MD, ["# BG Pre-Consolidation Preflight", "", "BG_PRECONSOLIDATION_PREFLIGHT_VERDICT = BLOCKED", "", f"- blocker: `{payload['blocker']}`"])
            print("BG_PRECONSOLIDATION_PREFLIGHT_VERDICT = BLOCKED")
            return 1

        controller = BGController.from_artifacts(device="cpu")
        payload["controller_load"] = "OK"
        payload["controller_heads_loaded"] = sorted(controller.heads.keys())
        payload["transformer_feature_extractor_import"] = "OK"
        payload["bg_steering_hook_import"] = "OK"

        selected = select_targets()
        target_rows = []
        for target in selected:
            counts = contrast_counts(str(target["domain"]), int(target["prefix_length"]), str(target["config"]))
            dirs = target_directions(target, controller)
            row = {
                **target,
                "contrast": counts,
                "direction_names": [d["direction_name"] for d in dirs],
                "direction_count": len(dirs),
                "all_direction_dims": sorted({int(d["vector"].numel()) for d in dirs}),
                "status": "READY" if counts.get("meets_minimum") and dirs else "PARTIAL",
            }
            target_rows.append(row)
        payload["selected_targets"] = target_rows
        primary = next((row for row in target_rows if row.get("target_id") == "T1"), None)
        if not primary:
            payload["blocker"] = "primary target reasoning@64 missing"
            write_json(OUT_JSON, payload)
            write_md(OUT_MD, ["# BG Pre-Consolidation Preflight", "", "BG_PRECONSOLIDATION_PREFLIGHT_VERDICT = BLOCKED", "", f"- blocker: `{payload['blocker']}`"])
            print("BG_PRECONSOLIDATION_PREFLIGHT_VERDICT = BLOCKED")
            return 1

        primary_head = load_head_row(str(primary["head_id"]), controller)
        primary_direction = head_weight_vector(primary_head)
        payload["primary_direction_dim"] = int(primary_direction.numel())
        payload["primary_direction_norm"] = float(torch.linalg.vector_norm(primary_direction).item())
        payload["primary_direction_rms"] = float(primary_direction.float().pow(2).mean().sqrt().item())

        extractor = BGTransformerFeatureExtractor(device="cuda" if torch.cuda.is_available() else "cpu", dtype="auto", force_all_loops=True)
        mode_spec = build_intervention_mode("single_loop_L1", 0.0)
        hook = BGLayerHookSteering(
            extractor.model,
            target_layer=int(primary["layer"]),
            target_loops=mode_spec["target_loops"],
            loop_alpha_scales=mode_spec["loop_alpha_scales"],
            direction=primary_direction / torch.linalg.vector_norm(primary_direction).clamp(min=1e-12),
            alpha=0.0,
        )
        prompt = "Answer with exactly one letter. Question: Which option is the letter A? Options: A. A B. B C. C D. D\nFINAL ANSWER:"
        no_hook = deterministic_generate_ids(extractor.model, extractor.tokenizer, prompt, max_new_tokens=16, hook=None)
        with_hook = deterministic_generate_ids(extractor.model, extractor.tokenizer, prompt, max_new_tokens=16, hook=hook)
        zero_alpha_ok = bool(torch.equal(no_hook, with_hook))
        diag = hook.diagnostics()
        payload["zero_alpha_hook_equivalence"] = zero_alpha_ok
        payload["hook_loop_index_source"] = diag.get("loop_index_source")
        payload["current_ut_loop_identity"] = diag.get("loop_index_source") == "current_ut"
        payload["use_cache_for_intervention_generation"] = False
        payload["diagnostic_d1"] = find_diagnostic_d1()
        payload["diagnostic_d1_status"] = "READY" if payload["diagnostic_d1"] else "MISSING"

        primary_dirs = set(primary.get("direction_names") or [])
        has_empirical = bool(primary_dirs & {"EMPIRICAL_MEAN_DIFF", "EMPIRICAL_WHITENED_DIFF", "LOGISTIC_SUCCESS_PROBE"})
        if not zero_alpha_ok:
            verdict = "BLOCKED"
            payload["blocker"] = "zero-alpha layer-hook generation differs from no-hook generation"
        elif not primary_dirs:
            verdict = "BLOCKED"
            payload["blocker"] = "no primary steering directions available"
        elif has_empirical and "RAW_NONORM_READOUT" in primary_dirs and payload["current_ut_loop_identity"]:
            verdict = "READY"
        else:
            verdict = "PARTIAL"
            payload["warnings"].append("layer hook works, but empirical directions or current_ut identity are partial")
        payload["BG_PRECONSOLIDATION_PREFLIGHT_VERDICT"] = verdict
        payload["elapsed_seconds"] = round(time.time() - started, 3)
        write_json(OUT_JSON, payload)

        lines = [
            "# BG Pre-Consolidation Control Probe Preflight",
            "",
            f"BG_PRECONSOLIDATION_PREFLIGHT_VERDICT = {verdict}",
            "",
            f"- zero_alpha_hook_equivalence: `{zero_alpha_ok}`",
            f"- hook_loop_index_source: `{payload.get('hook_loop_index_source')}`",
            "- use_cache_for_intervention_generation: `False`",
            f"- empirical_directions_exists: `{payload['empirical_directions_exists']}`",
            "",
            "| target | domain | prefix | config | directions | success | failure | status |",
            "|---|---|---:|---|---|---:|---:|---|",
        ]
        for row in target_rows:
            c = row.get("contrast") or {}
            lines.append(
                f"| `{row.get('target_id')}` | `{row.get('domain')}` | {row.get('prefix_length')} | "
                f"`{row.get('config')}` | `{', '.join(row.get('direction_names') or [])}` | "
                f"{c.get('successful_prefixes', '')} | {c.get('failed_prefixes', '')} | `{row.get('status')}` |"
            )
        write_md(OUT_MD, lines)
        print(f"BG_PRECONSOLIDATION_PREFLIGHT_VERDICT = {verdict}")
        print(f"Wrote {rel(OUT_JSON)}")
        print(f"Wrote {rel(OUT_MD)}")
        return 0 if verdict in {"READY", "PARTIAL"} else 1
    except Exception as exc:
        payload["errors"].append("".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:])
        payload["elapsed_seconds"] = round(time.time() - started, 3)
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# BG Pre-Consolidation Preflight", "", "BG_PRECONSOLIDATION_PREFLIGHT_VERDICT = BLOCKED", "", "```", payload["errors"][-1], "```"])
        print("BG_PRECONSOLIDATION_PREFLIGHT_VERDICT = BLOCKED")
        return 1
    finally:
        if extractor is not None:
            extractor.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
