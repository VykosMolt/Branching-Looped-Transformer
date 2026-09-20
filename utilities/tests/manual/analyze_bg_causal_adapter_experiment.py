#!/usr/bin/env python3
"""Analyze and document the causal BG intervention adapter experiment."""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch

from bg_causal_adapter_common import (
    EMPIRICAL_ROOT,
    OUT_ROOT,
    PROJECT_ROOT,
    append_once,
    avg,
    finite,
    load_empirical_direction,
    load_json,
    rel,
    write_json,
    write_md,
)


ANALYSIS_JSON = OUT_ROOT / "analysis.json"
ANALYSIS_MD = OUT_ROOT / "analysis.md"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
DOC_PATH = PROJECT_ROOT / "docs/evaluator/bg_causal_intervention_adapter.md"


def cosine(a: torch.Tensor | None, b: torch.Tensor | None) -> float | None:
    if a is None or b is None:
        return None
    av = a.detach().flatten().float()
    bv = b.detach().flatten().float()
    if av.numel() != bv.numel() or av.numel() == 0:
        return None
    denom = torch.linalg.vector_norm(av) * torch.linalg.vector_norm(bv)
    if float(denom.item()) <= 1e-12:
        return None
    return float(torch.dot(av, bv).item() / denom.item())


def adapter_proxy_direction() -> torch.Tensor | None:
    path = OUT_ROOT / "adapter_checkpoints/best_adapter.pt"
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("adapter_state_dict", {})
    up = state.get("up.weight")
    if isinstance(up, torch.Tensor) and up.ndim == 2:
        vec = up.float().mean(dim=1)
        return vec / torch.linalg.vector_norm(vec).clamp(min=1e-12)
    direction = state.get("direction")
    if isinstance(direction, torch.Tensor):
        return direction.float().flatten() / torch.linalg.vector_norm(direction.float().flatten()).clamp(min=1e-12)
    return None


def aggregate_free_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    intervention = [row for row in rows if row.get("method") != "baseline"]
    return {
        "rows": len(rows),
        "intervention_rows": len(intervention),
        "cuda_errors": sum(1 for row in rows if row.get("cuda_error")),
        "nan_or_inf": sum(1 for row in rows if row.get("nan_or_inf_activations")),
        "parse_rate": avg(1.0 - float(bool(row.get("parse_failed"))) for row in intervention),
        "empty_output_rate": avg(float(bool(row.get("empty_output"))) for row in intervention),
        "hit_max_tokens_rate": avg(float(bool(row.get("hit_max_tokens"))) for row in intervention),
        "repetition_rate": avg(finite(row.get("repetition_rate")) for row in intervention),
    }


def verdicts(
    preflight: dict[str, Any],
    dataset: dict[str, Any],
    implementation: dict[str, Any],
    training: dict[str, Any],
    teacher: dict[str, Any],
    free: dict[str, Any],
    pairwise: dict[str, Any],
) -> dict[str, Any]:
    training_verdict = str(training.get("BG_CAUSAL_ADAPTER_TRAINING_VERDICT", "INSUFFICIENT"))
    teacher_verdict = str(teacher.get("BG_CAUSAL_ADAPTER_TEACHER_FORCED_VERDICT", "INSUFFICIENT"))
    free_verdict = str(free.get("BG_CAUSAL_ADAPTER_FREE_GEN_VERDICT", "INSUFFICIENT"))
    if teacher_verdict == "ADAPTER_IMPROVES_LOGIT_MARGIN" or training_verdict == "READY":
        learning = "LEARNS_LOGIT_CONTROL"
    elif training_verdict == "PARTIAL" or teacher_verdict == "ADAPTER_NO_BETTER_THAN_STATIC":
        learning = "WEAK_LEARNING"
    elif training_verdict == "NO_LEARNING" or teacher_verdict == "ADAPTER_NO_EFFECT":
        learning = "NO_LEARNING"
    else:
        learning = "INSUFFICIENT"

    adapter_best = teacher.get("best_adapter") or {}
    static_best = teacher.get("best_static_or_random") or {}
    adapter_lift = finite(adapter_best.get("margin_lift_mean"), 0.0)
    static_lift = finite(static_best.get("margin_lift_mean"), 0.0)
    if teacher_verdict == "INSUFFICIENT":
        vs_static = "INSUFFICIENT"
    elif adapter_lift > static_lift + 1e-4:
        vs_static = "ADAPTER_BEATS_STATIC"
    elif adapter_lift + 1e-4 < static_lift:
        vs_static = "STATIC_BETTER"
    else:
        vs_static = "STATIC_MATCHES_ADAPTER"

    free_stats = aggregate_free_rows(free.get("rows") or [])
    teacher_kl = max(
        [
            finite(row.get("kl_non_answer_mean"), 0.0)
            for row in (teacher.get("aggregate") or {}).values()
            if isinstance(row, dict)
        ]
        or [0.0]
    )
    if free_verdict == "FREE_GEN_DESTABILIZING" or free_stats["cuda_errors"] or free_stats["nan_or_inf"]:
        stability = "DESTABILIZING"
    elif teacher_kl > 2.0:
        stability = "STABLE_BUT_KL_HIGH"
    elif free_verdict == "INSUFFICIENT":
        stability = "INSUFFICIENT"
    else:
        stability = "STABLE"

    if free_verdict == "FREE_GEN_LIFT":
        transfer = "TRANSFERS_TO_FREE_GENERATION"
    elif free_verdict == "TEACHER_FORCED_ONLY":
        transfer = "TEACHER_FORCED_ONLY"
    elif free_verdict == "NO_FREE_GEN_EFFECT":
        transfer = "NO_TRANSFER"
    else:
        transfer = "INSUFFICIENT"

    if stability == "DESTABILIZING":
        overall = "DESTABILIZING"
    elif teacher_verdict == "ADAPTER_IMPROVES_LOGIT_MARGIN" and free_verdict == "FREE_GEN_LIFT" and vs_static == "ADAPTER_BEATS_STATIC":
        overall = "PROMISING_WRITE_PATH"
    elif teacher_verdict == "ADAPTER_IMPROVES_LOGIT_MARGIN" and free_verdict in {"TEACHER_FORCED_ONLY", "NO_FREE_GEN_EFFECT"}:
        overall = "LOCAL_LOGIT_CONTROL_ONLY"
    elif teacher_verdict in {"ADAPTER_NO_EFFECT", "ADAPTER_NO_BETTER_THAN_STATIC"}:
        overall = "NO_CAUSAL_WRITE_PATH_FOUND"
    else:
        overall = "INSUFFICIENT"

    if teacher_verdict == "ADAPTER_IMPROVES_LOGIT_MARGIN" and free_verdict == "FREE_GEN_LIFT":
        tf_interpretation = "LOAD_BEARING_ONLY_IF_FREE_GEN_TRANSFERS"
    elif teacher_verdict == "ADAPTER_IMPROVES_LOGIT_MARGIN":
        tf_interpretation = "TEACHER_FORCED_SHORTCUT_RISK"
    else:
        tf_interpretation = "INSUFFICIENT"

    if overall == "PROMISING_WRITE_PATH":
        recommended = "expand_adapter_training_and_consider_adapter_as_phase2_write_path"
    elif overall == "LOCAL_LOGIT_CONTROL_ONLY":
        recommended = "train_sequence_level_or_black_box_finetuned_adapter"
    elif overall == "NO_CAUSAL_WRITE_PATH_FOUND":
        recommended = "move_to_backbone_regularization_design"
    elif overall == "DESTABILIZING":
        recommended = "redesign_adapter_objective_or_reduce_intervention_strength"
    else:
        recommended = "improve_dataset_or_reduce_scope"

    return {
        "BG_CAUSAL_ADAPTER_PREFLIGHT_VERDICT": preflight.get("BG_CAUSAL_ADAPTER_PREFLIGHT_VERDICT", "INSUFFICIENT"),
        "BG_CAUSAL_ADAPTER_DATASET_VERDICT": dataset.get("BG_CAUSAL_ADAPTER_DATASET_VERDICT", "INSUFFICIENT"),
        "BG_CAUSAL_ADAPTER_IMPLEMENTATION_VERDICT": implementation.get("BG_CAUSAL_ADAPTER_IMPLEMENTATION_VERDICT", "INSUFFICIENT"),
        "BG_CAUSAL_ADAPTER_TRAINING_VERDICT": training_verdict,
        "BG_CAUSAL_ADAPTER_TEACHER_FORCED_VERDICT": teacher_verdict,
        "BG_CAUSAL_ADAPTER_FREE_GEN_VERDICT": free_verdict,
        "BG_PAIRWISE_CAUSAL_CONTRAST_ADAPTER_VERDICT": pairwise.get("BG_PAIRWISE_CAUSAL_CONTRAST_ADAPTER_VERDICT", "SKIPPED"),
        "BG_CAUSAL_ADAPTER_LEARNING_VERDICT": learning,
        "BG_CAUSAL_ADAPTER_VS_STATIC_VERDICT": vs_static,
        "BG_CAUSAL_ADAPTER_STABILITY_VERDICT": stability,
        "BG_CAUSAL_ADAPTER_GENERATION_TRANSFER_VERDICT": transfer,
        "BG_CAUSAL_ADAPTER_VERDICT": overall,
        "TEACHER_FORCED_RESULT_INTERPRETATION": tf_interpretation,
        "FREE_GENERATION_EVAL_COMPLETED": bool(free.get("FREE_GENERATION_EVAL_COMPLETED", False)),
        "KL_ANSWER_POSITION_MASKED": bool(training.get("KL_ANSWER_POSITION_MASKED", teacher.get("KL_ANSWER_POSITION_MASKED", False))),
        "INTERVENTION_POSITION_KIND": str(training.get("intervention_position_kind") or "unknown"),
        "RECOMMENDED_NEXT": recommended,
    }


def write_docs(v: dict[str, Any], analysis: dict[str, Any]) -> None:
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BG Causal Intervention Adapter",
        "",
        "This experiment follows the static steering failures: raw NoNorm, empirical mean-diff, whitened, logistic, and classifier-style adapter directions did not give reliable signed causal control.",
        "",
        "The adapter is trained as a causal write-path. Ouro and BG heads stay frozen; only a tiny adapter can update. The primary objective is teacher-forced correct-option logit control, not BG-score optimization.",
        "",
        "Teacher-forced margin improvement is necessary but not sufficient because it has direct access to answer logits. Free-generation transfer is the load-bearing result.",
        "",
        "## Methodology",
        "",
        "- adapter variants implemented: `Rank1GatedDirectionAdapter`, `LowRankDeltaAdapter`, `HyperDirectionAdapter`",
        "- trained variant: `LowRankDeltaAdapter(rank=32)`",
        "- primary mode: `multi_loop_decayed`",
        "- intervention layer: `36`",
        "- intervention position: `prefix_last_token`, before the `FINAL ANSWER:` suffix",
        "- KL preservation masks the answer position",
        "- use_cache is disabled for intervention runs",
        "- alpha/effective RMS is capped at `0.02`",
        "",
        "## Verdicts",
        "",
    ]
    for key in [
        "BG_CAUSAL_ADAPTER_PREFLIGHT_VERDICT",
        "BG_CAUSAL_ADAPTER_DATASET_VERDICT",
        "BG_CAUSAL_ADAPTER_IMPLEMENTATION_VERDICT",
        "BG_CAUSAL_ADAPTER_TRAINING_VERDICT",
        "BG_CAUSAL_ADAPTER_TEACHER_FORCED_VERDICT",
        "BG_CAUSAL_ADAPTER_FREE_GEN_VERDICT",
        "BG_PAIRWISE_CAUSAL_CONTRAST_ADAPTER_VERDICT",
        "BG_CAUSAL_ADAPTER_LEARNING_VERDICT",
        "BG_CAUSAL_ADAPTER_VS_STATIC_VERDICT",
        "BG_CAUSAL_ADAPTER_STABILITY_VERDICT",
        "BG_CAUSAL_ADAPTER_GENERATION_TRANSFER_VERDICT",
        "BG_CAUSAL_ADAPTER_VERDICT",
    ]:
        lines.append(f"- {key} = `{v.get(key)}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(analysis.get("one_sentence_interpretation", "")),
            "",
            "## Report Paths",
            "",
            f"- summary: `{rel(SUMMARY_MD)}`",
            f"- analysis: `{rel(ANALYSIS_MD)}`",
            f"- training report: `{rel(OUT_ROOT / 'training_report.md')}`",
            f"- teacher-forced eval: `{rel(OUT_ROOT / 'teacher_forced_eval.md')}`",
            f"- free-generation eval: `{rel(OUT_ROOT / 'free_generation_eval.md')}`",
        ]
    )
    write_md(DOC_PATH, lines)

    section_lines = [
        f"BG_CAUSAL_ADAPTER_PREFLIGHT_VERDICT = {v.get('BG_CAUSAL_ADAPTER_PREFLIGHT_VERDICT')}",
        f"BG_CAUSAL_ADAPTER_DATASET_VERDICT = {v.get('BG_CAUSAL_ADAPTER_DATASET_VERDICT')}",
        f"BG_CAUSAL_ADAPTER_IMPLEMENTATION_VERDICT = {v.get('BG_CAUSAL_ADAPTER_IMPLEMENTATION_VERDICT')}",
        f"BG_CAUSAL_ADAPTER_TRAINING_VERDICT = {v.get('BG_CAUSAL_ADAPTER_TRAINING_VERDICT')}",
        f"BG_CAUSAL_ADAPTER_TEACHER_FORCED_VERDICT = {v.get('BG_CAUSAL_ADAPTER_TEACHER_FORCED_VERDICT')}",
        f"BG_CAUSAL_ADAPTER_FREE_GEN_VERDICT = {v.get('BG_CAUSAL_ADAPTER_FREE_GEN_VERDICT')}",
        f"BG_PAIRWISE_CAUSAL_CONTRAST_ADAPTER_VERDICT = {v.get('BG_PAIRWISE_CAUSAL_CONTRAST_ADAPTER_VERDICT')}",
        f"BG_CAUSAL_ADAPTER_LEARNING_VERDICT = {v.get('BG_CAUSAL_ADAPTER_LEARNING_VERDICT')}",
        f"BG_CAUSAL_ADAPTER_VS_STATIC_VERDICT = {v.get('BG_CAUSAL_ADAPTER_VS_STATIC_VERDICT')}",
        f"BG_CAUSAL_ADAPTER_STABILITY_VERDICT = {v.get('BG_CAUSAL_ADAPTER_STABILITY_VERDICT')}",
        f"BG_CAUSAL_ADAPTER_GENERATION_TRANSFER_VERDICT = {v.get('BG_CAUSAL_ADAPTER_GENERATION_TRANSFER_VERDICT')}",
        f"BG_CAUSAL_ADAPTER_VERDICT = {v.get('BG_CAUSAL_ADAPTER_VERDICT')}",
        f"TEACHER_FORCED_RESULT_INTERPRETATION = {v.get('TEACHER_FORCED_RESULT_INTERPRETATION')}",
        f"FREE_GENERATION_EVAL_COMPLETED = {str(v.get('FREE_GENERATION_EVAL_COMPLETED')).lower()}",
        f"KL_ANSWER_POSITION_MASKED = {str(v.get('KL_ANSWER_POSITION_MASKED')).lower()}",
        f"INTERVENTION_POSITION_KIND = {v.get('INTERVENTION_POSITION_KIND')}",
        "",
        str(analysis.get("one_sentence_interpretation", "")),
        "",
        f"Full reports: `{rel(SUMMARY_MD)}`, `{rel(ANALYSIS_MD)}`, `{rel(DOC_PATH)}`.",
    ]
    for doc in [
        "docs/evaluator/current_state.md",
        "docs/evaluator/domain_transfer_ledger.md",
        "docs/evaluator/bg_empirical_steering_direction.md",
        "docs/evaluator/bg_stage2_layerhook_followup.md",
        "docs/evaluator/bg_preconsolidation_control_probes.md",
        "docs/evaluator/post_v10_synthesis_2026-05-18_v8.1_routing_locked.md",
    ]:
        path = PROJECT_ROOT / doc
        if path.exists():
            append_once(path, "BG causal intervention adapter (2026-05-18)", section_lines)


def main() -> int:
    started = time.time()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    preflight = load_json(OUT_ROOT / "preflight.json", {})
    dataset = load_json(OUT_ROOT / "adapter_dataset.json", {})
    implementation = load_json(OUT_ROOT / "implementation_tests.json", {})
    training = load_json(OUT_ROOT / "training_log.json", {})
    teacher = load_json(OUT_ROOT / "teacher_forced_eval.json", {})
    free = load_json(OUT_ROOT / "free_generation_eval.json", {})
    pairwise = load_json(OUT_ROOT / "pairwise_contrast_adapter.json", {"BG_PAIRWISE_CAUSAL_CONTRAST_ADAPTER_VERDICT": "SKIPPED"})

    v = verdicts(preflight, dataset, implementation, training, teacher, free, pairwise)
    proxy = adapter_proxy_direction()
    raw = load_empirical_direction("RAW_NONORM_READOUT")
    mean = load_empirical_direction("EMPIRICAL_MEAN_DIFF")
    geometry = {
        "adapter_proxy_direction_kind": "LowRankDeltaAdapter_up_weight_mean" if proxy is not None else "unavailable",
        "cosine_adapter_proxy_raw_nonorm": cosine(proxy, raw),
        "cosine_adapter_proxy_empirical_mean_diff": cosine(proxy, mean),
        "cosine_raw_nonorm_empirical_mean_diff": cosine(raw, mean),
        "geometry_note": "LowRank adapter geometry is input-conditioned; cosine values use a static up-weight proxy, not a full hidden-state average delta.",
    }

    free_rows = free.get("rows") or []
    free_stats = aggregate_free_rows(free_rows)
    teacher_best = teacher.get("best_adapter") or {}
    free_best = free.get("best_adapter") or {}
    interpretation = (
        "The causal adapter test separates local teacher-forced logit control from actual trajectory transfer; "
        f"overall verdict is {v['BG_CAUSAL_ADAPTER_VERDICT']}."
    )
    analysis = {
        **v,
        "teacher_forced_best_adapter": teacher_best,
        "free_generation_best_adapter": free_best,
        "training_best_val": training.get("best_val"),
        "free_generation_stats": free_stats,
        "geometry": geometry,
        "one_sentence_interpretation": interpretation,
        "methodology_checks": {
            "KL_ANSWER_POSITION_MASKED": v["KL_ANSWER_POSITION_MASKED"],
            "INTERVENTION_POSITION_KIND": v["INTERVENTION_POSITION_KIND"],
            "INTERVENTION_POSITION_WARNING": bool(training.get("INTERVENTION_POSITION_WARNING", teacher.get("INTERVENTION_POSITION_WARNING", False))),
            "FREE_GENERATION_EVAL_COMPLETED": v["FREE_GENERATION_EVAL_COMPLETED"],
        },
        "report_paths": {
            "preflight": rel(OUT_ROOT / "preflight.md"),
            "dataset": rel(OUT_ROOT / "adapter_dataset.md"),
            "implementation": rel(OUT_ROOT / "implementation_tests.md"),
            "training": rel(OUT_ROOT / "training_report.md"),
            "teacher_forced_eval": rel(OUT_ROOT / "teacher_forced_eval.md"),
            "free_generation_eval": rel(OUT_ROOT / "free_generation_eval.md"),
            "analysis": rel(ANALYSIS_MD),
            "summary": rel(SUMMARY_MD),
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(ANALYSIS_JSON, analysis)

    top_keys = [
        "BG_CAUSAL_ADAPTER_PREFLIGHT_VERDICT",
        "BG_CAUSAL_ADAPTER_DATASET_VERDICT",
        "BG_CAUSAL_ADAPTER_IMPLEMENTATION_VERDICT",
        "BG_CAUSAL_ADAPTER_TRAINING_VERDICT",
        "BG_CAUSAL_ADAPTER_TEACHER_FORCED_VERDICT",
        "BG_CAUSAL_ADAPTER_FREE_GEN_VERDICT",
        "BG_PAIRWISE_CAUSAL_CONTRAST_ADAPTER_VERDICT",
        "BG_CAUSAL_ADAPTER_LEARNING_VERDICT",
        "BG_CAUSAL_ADAPTER_VS_STATIC_VERDICT",
        "BG_CAUSAL_ADAPTER_STABILITY_VERDICT",
        "BG_CAUSAL_ADAPTER_GENERATION_TRANSFER_VERDICT",
        "BG_CAUSAL_ADAPTER_VERDICT",
        "TEACHER_FORCED_RESULT_INTERPRETATION",
        "FREE_GENERATION_EVAL_COMPLETED",
        "KL_ANSWER_POSITION_MASKED",
        "INTERVENTION_POSITION_KIND",
        "RECOMMENDED_NEXT",
    ]
    summary_payload = {key: analysis.get(key) for key in top_keys}
    summary_payload.update(
        {
            "teacher_forced_best_adapter": teacher_best,
            "free_generation_best_adapter": free_best,
            "report_paths": analysis["report_paths"],
        }
    )
    write_json(SUMMARY_JSON, summary_payload)

    analysis_lines = ["# BG Causal Adapter Analysis", ""]
    analysis_lines.extend(f"{key} = {analysis.get(key)}" for key in top_keys)
    analysis_lines.extend(
        [
            "",
            "## Did The Adapter Learn?",
            "",
            f"- training best validation margin lift: `{finite(training.get('best_val_margin_lift')):.6f}`",
            f"- teacher-forced best adapter margin lift: `{finite(teacher_best.get('margin_lift_mean')):.6f}`",
            "",
            "## Teacher-Forced To Free-Generation Transfer",
            "",
            f"- free-generation best adapter success: `{finite(free_best.get('success_rate')):.3f}`",
            f"- free-generation lift over baseline: `{finite(free.get('free_generation_lift_over_baseline')):.3f}`",
            f"- free-generation lift over static/random: `{finite(free.get('free_generation_lift_over_random_or_static')):.3f}`",
            "",
            "## Stability",
            "",
            f"- free-generation CUDA errors: `{free_stats['cuda_errors']}`",
            f"- free-generation NaN/Inf rows: `{free_stats['nan_or_inf']}`",
            f"- intervention parse rate: `{finite(free_stats.get('parse_rate')):.3f}`",
            f"- intervention repetition rate: `{finite(free_stats.get('repetition_rate')):.3f}`",
            "",
            "## Geometry",
            "",
            f"- cosine(adapter proxy, raw NoNorm): `{geometry['cosine_adapter_proxy_raw_nonorm']}`",
            f"- cosine(adapter proxy, empirical mean diff): `{geometry['cosine_adapter_proxy_empirical_mean_diff']}`",
            f"- note: {geometry['geometry_note']}",
            "",
            "## Interpretation",
            "",
            interpretation,
        ]
    )
    write_md(ANALYSIS_MD, analysis_lines)

    summary_lines = ["# BG Causal Intervention Adapter Summary", ""]
    summary_lines.extend(f"{key} = {summary_payload.get(key)}" for key in top_keys)
    summary_lines.extend(
        [
            "",
            "## Motivation",
            "",
            "Static BG/readout directions did not provide reliable signed causal control, so this probe tested a tiny learned write-path adapter with Ouro frozen.",
            "",
            "## Dataset",
            "",
            f"- dataset verdict: `{v['BG_CAUSAL_ADAPTER_DATASET_VERDICT']}`",
            f"- task split counts: `{dataset.get('task_counts')}`",
            "",
            "## Adapter Architecture",
            "",
            "- implemented rank-1 gated, low-rank, and hyper-direction variants; trained low-rank rank-32 adapter.",
            "",
            "## Training Objective",
            "",
            "- primary: teacher-forced correct-option CE/logit margin.",
            "- auxiliary: non-answer KL preservation and delta RMS penalty.",
            "- BG scores are diagnostics only, not the primary loss.",
            "",
            "## KL Masking And Position",
            "",
            f"- KL_ANSWER_POSITION_MASKED = `{v['KL_ANSWER_POSITION_MASKED']}`",
            f"- INTERVENTION_POSITION_KIND = `{v['INTERVENTION_POSITION_KIND']}`",
            "",
            "## Training Results",
            "",
            f"- training verdict: `{v['BG_CAUSAL_ADAPTER_TRAINING_VERDICT']}`",
            f"- best validation margin lift: `{finite(training.get('best_val_margin_lift')):.6f}`",
            "",
            "## Teacher-Forced Eval",
            "",
            f"- verdict: `{v['BG_CAUSAL_ADAPTER_TEACHER_FORCED_VERDICT']}`",
            f"- best adapter margin lift: `{finite(teacher_best.get('margin_lift_mean')):.6f}`",
            "",
            "## Free-Generation Eval",
            "",
            f"- verdict: `{v['BG_CAUSAL_ADAPTER_FREE_GEN_VERDICT']}`",
            f"- completed: `{v['FREE_GENERATION_EVAL_COMPLETED']}`",
            f"- best adapter success: `{finite(free_best.get('success_rate')):.3f}`",
            "",
            "## Optional Pairwise Contrast",
            "",
            f"- verdict: `{v['BG_PAIRWISE_CAUSAL_CONTRAST_ADAPTER_VERDICT']}`",
            "",
            "## Stability",
            "",
            f"- stability verdict: `{v['BG_CAUSAL_ADAPTER_STABILITY_VERDICT']}`",
            "",
            "## Geometry",
            "",
            f"- adapter/raw cosine proxy: `{geometry['cosine_adapter_proxy_raw_nonorm']}`",
            "",
            "## Phase 2 Implication",
            "",
            f"- recommended next: `{v['RECOMMENDED_NEXT']}`",
            "",
            "## Docs Updated",
            "",
            f"- `{rel(DOC_PATH)}`",
            "",
            "## Files Modified / Created",
            "",
            f"- `{rel(OUT_ROOT)}`",
            f"- `{rel(DOC_PATH)}`",
            "",
            "## Commands Run",
            "",
            "- See terminal/session transcript for py_compile and run commands.",
            "",
            "## Blockers",
            "",
            "- None recorded by the analyzer beyond verdict-specific insufficiency markers.",
        ]
    )
    write_md(SUMMARY_MD, summary_lines)
    write_docs(v, analysis)
    print(f"BG_CAUSAL_ADAPTER_VERDICT = {v['BG_CAUSAL_ADAPTER_VERDICT']}")
    print(f"Wrote {rel(ANALYSIS_JSON)}")
    print(f"Wrote {rel(SUMMARY_JSON)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
