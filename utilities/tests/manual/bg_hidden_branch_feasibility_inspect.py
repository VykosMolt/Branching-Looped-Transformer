"""Inspect whether local Ouro supports true same-prefix hidden-state forks."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from bg_hidden_branch_suite_common import MODEL_PATH, REPORT_ROOT, ensure_report_root, rel, write_json, write_md


OUT_JSON = REPORT_ROOT / "feasibility.json"
OUT_MD = REPORT_ROOT / "feasibility.md"
TARGET_BRANCH_POINTS = (24, 30, 36, 42, 47)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def main() -> int:
    ensure_report_root()
    started = time.time()
    config_path = MODEL_PATH / "config.json"
    modeling_path = MODEL_PATH / "modeling_ouro.py"
    steering_path = Path(__file__).resolve().parents[4] / "shared/src/evaluator/bg_steering_hook.py"
    config = json.loads(_read(config_path) or "{}")
    modeling = _read(modeling_path)
    steering = _read(steering_path)

    num_layers = int(config.get("num_hidden_layers") or 0)
    hidden_size = int(config.get("hidden_size") or 0)
    total_ut_steps = int(config.get("total_ut_steps") or 0)
    has_run_single_loop = "_run_single_ut_loop" in modeling
    has_inputs_embeds = "inputs_embeds" in modeling
    has_current_ut = "current_ut" in modeling
    has_universal_cache = "UniversalTransformerCache" in modeling
    has_per_loop_hidden = "return_per_loop_hidden_states" in modeling and "per_loop_hidden_states" in modeling
    existing_hook = "BGLayerHookSteering" in steering and "current_ut" in steering
    branch_point_rows = []
    for point in TARGET_BRANCH_POINTS:
        is_boundary = point == 47
        layer_access = (1 <= point <= num_layers) if not is_boundary else True
        branch_point_rows.append(
            {
                "branch_point": f"L{point}",
                "kind": "loop_boundary_tap" if is_boundary else "decoder_layer_post_block",
                "zero_based_layer_index": None if is_boundary else point - 1,
                "can_capture_hidden_at_layer": bool(layer_access and (has_per_loop_hidden or existing_hook)),
                "can_resume_from_hidden_at_layer": False,
                "can_hook_intervene_at_layer": bool((1 <= point <= num_layers) and existing_hook),
                "notes": (
                    "Captured as Ouro loop-boundary/per-loop hidden state; generation resume still lacks branch-aware cache."
                    if is_boundary
                    else "Layer hook/capture accessible; exact resume into later layers plus autoregressive cache is not exposed as a validated API."
                ),
            }
        )

    can_capture = bool(num_layers >= 42 and hidden_size == 2048 and total_ut_steps >= 4 and existing_hook)
    can_resume_layer = False
    can_carry_branch_batch = False
    can_use_cache = bool(has_universal_cache)
    loop_boundary_partial = bool(has_run_single_loop and has_inputs_embeds and has_per_loop_hidden)
    if can_capture and existing_hook:
        verdict = "HOOK_HIDDEN_ORIGIN_READY"
        live_method = "hook_intervention_per_branch"
    elif loop_boundary_partial:
        verdict = "FEATURE_ONLY_PARTIAL"
        live_method = "feature_only"
    else:
        verdict = "BLOCKED"
        live_method = "blocked"

    payload = {
        "BG_HIDDEN_BRANCH_FEASIBILITY_VERDICT": verdict,
        "LIVE_BRANCH_METHOD": live_method,
        "model_path": rel(MODEL_PATH),
        "model_config": {
            "num_hidden_layers": num_layers,
            "hidden_size": hidden_size,
            "total_ut_steps": total_ut_steps,
            "use_cache": bool(config.get("use_cache", False)),
        },
        "inspected_files": [rel(config_path), rel(modeling_path), rel(steering_path)],
        "model_internal_findings": {
            "has_run_single_ut_loop": has_run_single_loop,
            "has_inputs_embeds_path": has_inputs_embeds,
            "has_current_ut": has_current_ut,
            "has_universal_transformer_cache": has_universal_cache,
            "has_per_loop_hidden_states": has_per_loop_hidden,
            "existing_layer_hook_surface": existing_hook,
            "latent_beam_search_present": bool(list(Path(__file__).resolve().parents[4].rglob("latent_beam_search.py"))),
        },
        "branch_points": branch_point_rows,
        "can_capture_hidden_at_layer": can_capture,
        "can_resume_from_hidden_at_layer": can_resume_layer,
        "can_carry_branch_batch": can_carry_branch_batch,
        "can_use_cache": can_use_cache,
        "recommended_cache_mode": "disable_cache_for_hidden_branch_suite",
        "fallback_method_if_true_fork_blocked": "hook_hidden_origin_brancher_with_use_cache_false",
        "true_fork_carry_blocker": (
            "Local Ouro exposes layer hooks, current_ut, per-loop states, and a UniversalTransformerCache, "
            "but no validated generation API resumes from a copied internal layer hidden state with matching "
            "branch-specific KV/cache state. True autoregressive branch-batch carry must not be faked."
        ),
        "loop_boundary_partial_note": (
            "Prompt-only loop-boundary hidden carry can be inspected with inputs_embeds/_run_single_ut_loop patterns, "
            "but this is not sufficient for final-output branch generation."
        ),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    lines = [
        "# BG Hidden Branch Feasibility",
        "",
        f"BG_HIDDEN_BRANCH_FEASIBILITY_VERDICT = {verdict}",
        f"LIVE_BRANCH_METHOD = {live_method}",
        "",
        "## Interpretation",
        "",
        "True autoregressive fork/carry is not generation-ready because copied layer hidden states cannot be resumed with a validated branch-specific cache/state path. The suite therefore uses same-prefix hook-hidden-origin branches and keeps that method label explicit.",
        "",
        "## Branch Points",
        "",
        "| Point | Kind | Capture | Resume | Hook | Notes |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in branch_point_rows:
        lines.append(
            f"| `{row['branch_point']}` | `{row['kind']}` | `{row['can_capture_hidden_at_layer']}` | "
            f"`{row['can_resume_from_hidden_at_layer']}` | `{row['can_hook_intervene_at_layer']}` | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## Cache",
            "",
            f"- can_use_cache: `{can_use_cache}`",
            "- recommended_cache_mode: `disable_cache_for_hidden_branch_suite`",
            "",
            "## Blocker",
            "",
            payload["true_fork_carry_blocker"],
        ]
    )
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_BRANCH_FEASIBILITY_VERDICT = {verdict}")
    print(f"LIVE_BRANCH_METHOD = {live_method}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
