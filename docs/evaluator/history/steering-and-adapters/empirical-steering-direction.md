# BG Empirical Steering Direction Probe

## Purpose

The prior layer-hook follow-up showed that the hook mechanism is stable, but raw BG NoNorm readout vectors produce mostly unsigned score movement. This probe tests whether success-derived frozen-feature directions are better steering axes.

## Direction Construction

Directions were built from Stage 1 frozen prefix features using raw NoNorm, empirical mean difference, diagonal-whitened mean difference, and a logistic success probe. Ouro weights and BG heads were not trained.

## Layer-Hook Setup

The probe uses layer_hook_injection only, with use_cache=False, current_ut loop identity, position -1, alpha <= 0.02, and the prior best modes: multi_loop_decayed as primary and single_loop_L1 as comparison.

## Results

- BG_EMPIRICAL_DIRECTION_CAUSAL_VERDICT = `EMPIRICAL_UNSIGNED_ONLY`
- BG_EMPIRICAL_VS_RAW_VERDICT = `EMPIRICAL_BEATS_RAW`
- BG_STEERING_DIRECTION_GEOMETRY_VERDICT = `RAW_READOUT_NOT_PRODUCTION_DIRECTION`
- BG_EMPIRICAL_STEERING_STABILITY_VERDICT = `DESTABILIZING`
- BG_EMPIRICAL_FINAL_LIFT_VERDICT = `NEGATIVE_LIFT`
- BG_TINY_STEERING_ADAPTER_VERDICT = `NO_BETTER_THAN_STATIC`
- BG_EMPIRICAL_STEERING_VERDICT = `DESTABILIZING`

## Interpretation

If empirical directions show only unsigned movement, the likely issue is not the layer-hook mechanism but the absence of a calibrated production-space control direction.

## Reports

- summary: `artifacts/reports/probes/bg_empirical_steering_direction_2026-05-18/summary.md`
- analysis: `artifacts/reports/probes/bg_empirical_steering_direction_2026-05-18/analysis.md`
- traces: `artifacts/reports/probes/bg_empirical_steering_direction_2026-05-18/empirical_steering_traces.json`

## BG pre-consolidation control probes (2026-05-18)

- BG_RMS_STEERING_VERDICT = `RMS_UNSIGNED_ONLY`
- BG_RMS_VS_L2_VERDICT = `RMS_MATCHES_L2`
- BG_PROPAGATION_VERDICT = `PROPAGATES_TO_LATER_STATES`
- BG_PROPAGATION_DECAY_PROFILE = `SURVIVES_32_TOKENS`
- BG_TEXT_PREFIX_EXPANSION_VERDICT = `WEAK_POSITIVE`
- BG_CAUSAL_GRADIENT_VERDICT = `GRADIENT_NO_BETTER_THAN_RANDOM`
- BG_INFERENCE_TIME_STEERING_VERDICT = `UNSIGNED_ONLY`
- BG_BRANCH_ALLOCATION_VERDICT = `PROMISING`
- BG_PHASE2_REQUIREMENT_VERDICT = `TRAINING_REQUIRED`
- interpretation: The model can be nudged in BG-readable state space, but tested directions do not provide reliable signed control; Phase 2 training is required.
- reports: `artifacts/reports/probes/bg_preconsolidation_control_probes_2026-05-18/summary.md`, `artifacts/reports/probes/bg_preconsolidation_control_probes_2026-05-18/final_analysis.json`, `docs/evaluator/preconsolidation-control-probes.md`

## BG causal intervention adapter (2026-05-18)

BG_CAUSAL_ADAPTER_PREFLIGHT_VERDICT = READY
BG_CAUSAL_ADAPTER_DATASET_VERDICT = READY
BG_CAUSAL_ADAPTER_IMPLEMENTATION_VERDICT = READY
BG_CAUSAL_ADAPTER_TRAINING_VERDICT = PARTIAL
BG_CAUSAL_ADAPTER_TEACHER_FORCED_VERDICT = ADAPTER_IMPROVES_LOGIT_MARGIN
BG_CAUSAL_ADAPTER_FREE_GEN_VERDICT = TEACHER_FORCED_ONLY
BG_PAIRWISE_CAUSAL_CONTRAST_ADAPTER_VERDICT = SKIPPED
BG_CAUSAL_ADAPTER_LEARNING_VERDICT = LEARNS_LOGIT_CONTROL
BG_CAUSAL_ADAPTER_VS_STATIC_VERDICT = ADAPTER_BEATS_STATIC
BG_CAUSAL_ADAPTER_STABILITY_VERDICT = STABLE
BG_CAUSAL_ADAPTER_GENERATION_TRANSFER_VERDICT = TEACHER_FORCED_ONLY
BG_CAUSAL_ADAPTER_VERDICT = LOCAL_LOGIT_CONTROL_ONLY
TEACHER_FORCED_RESULT_INTERPRETATION = TEACHER_FORCED_SHORTCUT_RISK
FREE_GENERATION_EVAL_COMPLETED = true
KL_ANSWER_POSITION_MASKED = true
INTERVENTION_POSITION_KIND = prefix_last_token

The causal adapter test separates local teacher-forced logit control from actual trajectory transfer; overall verdict is LOCAL_LOGIT_CONTROL_ONLY.

Full reports: `artifacts/reports/probes/bg_causal_intervention_adapter_2026-05-18/summary.md`, `artifacts/reports/probes/bg_causal_intervention_adapter_2026-05-18/analysis.md`, `docs/evaluator/causal-intervention-adapter.md`.
