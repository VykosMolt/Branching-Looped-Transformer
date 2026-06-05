# BG Stage 2 Layer-Hook Follow-Up (2026-05-18)

## Why This Follow-Up Was Needed

The broad Stage 2 v3 sweep validated layer-hook mechanics but was too large for a clean completion under no-cache generation. This narrowed follow-up isolates T1 reasoning@64 and asks whether the already validated layer-hook surface produces reliable signed BG-readable movement.

## Latent Boundary Fork Status

The latent boundary fork remains blocked for full generation continuation because the local model does not expose a validated API for resuming autoregressive generation from a copied post-loop hidden boundary without cache/state forking.

## Layer-Hook Validity

Layer-hook injection is transformer-native, uses decoder-layer forward hooks, identifies loop position through `current_ut`, runs with `use_cache=False`, and preserves zero-alpha equivalence.

## Target And Metrics

The follow-up targets T1 reasoning@64 with the best NoNorm Stage 1 cell, keeps AntisymLinear as diagnostic readout only, and separates causal sensitivity, stability, and final correctness.

## Results

BG_STAGE2_PARTIAL_TRACE_AUDIT_VERDICT = READY
BG_LAYERHOOK_FOLLOWUP_PREFLIGHT_VERDICT = READY
BG_LAYERHOOK_FOLLOWUP_TASKS_VERDICT = READY
BG_LAYERHOOK_FOLLOWUP_SWEEP_VERDICT = READY
BG_LAYERHOOK_MECHANICAL_VERDICT = READY
BG_LAYERHOOK_SIGNED_CAUSAL_VERDICT = UNSIGNED_EFFECT
BG_SINGLE_LOOP_POSITION_VERDICT = L1_BETTER
BG_MULTILOOP_VERDICT = MULTILOOP_STRONGER
BG_LAYERHOOK_STABILITY_VERDICT = STABLE_BUT_TINY
BG_FINAL_TASK_LIFT_VERDICT = INSUFFICIENT
BG_LAYERHOOK_FOLLOWUP_VERDICT = READ_ONLY_BG_FOR_NOW
BEST_SINGLE_LOOP_MODE = single_loop_L1
BEST_MULTILOOP_MODE = multi_loop_decayed
MULTILOOP_GAIN_OVER_BEST_SINGLE = 0.0706979167497257
RECOMMENDED_NEXT = keep_BG_as_readout_selector_and_revisit_steering_with_empirical_success_direction_or_training

## Interpretation

BG remains more reliable as a readout selector than as an inference-time steering vector under this protocol.

## Recommended Next Step

`keep_BG_as_readout_selector_and_revisit_steering_with_empirical_success_direction_or_training`

## BG empirical steering direction probe (2026-05-18)

BG_EMPIRICAL_STEERING_PREFLIGHT_VERDICT = READY
BG_EMPIRICAL_DIRECTION_BUILD_VERDICT = READY
BG_EMPIRICAL_STEERING_TASKS_VERDICT = READY
BG_EMPIRICAL_STEERING_SWEEP_VERDICT = READY
BG_EMPIRICAL_DIRECTION_CAUSAL_VERDICT = EMPIRICAL_UNSIGNED_ONLY
BG_EMPIRICAL_VS_RAW_VERDICT = EMPIRICAL_BEATS_RAW
BG_STEERING_DIRECTION_GEOMETRY_VERDICT = RAW_READOUT_NOT_PRODUCTION_DIRECTION
BG_EMPIRICAL_STEERING_STABILITY_VERDICT = DESTABILIZING
BG_EMPIRICAL_FINAL_LIFT_VERDICT = NEGATIVE_LIFT
BG_TINY_STEERING_ADAPTER_VERDICT = NO_BETTER_THAN_STATIC
BG_EMPIRICAL_STEERING_VERDICT = DESTABILIZING
MODE_COVERAGE = {"EMPIRICAL_MEAN_DIFF": {"multi_loop_decayed": {"complete_expected_rows": 48, "intervention_rows": 48, "task_count": 6}, "single_loop_L1": {"complete_expected_rows": 48, "intervention_rows": 48, "task_count": 6}}, "EMPIRICAL_WHITENED_DIFF": {"multi_loop_decayed": {"complete_expected_rows": 48, "intervention_rows": 48, "task_count": 6}, "single_loop_L1": {"complete_expected_rows": 48, "intervention_rows": 48, "task_count": 6}}, "LOGISTIC_SUCCESS_PROBE": {"multi_loop_decayed": {"complete_expected_rows": 48, "intervention_rows": 48, "task_count": 6}, "single_loop_L1": {"complete_expected_rows": 48, "intervention_rows": 48, "task_count": 6}}, "RAW_NONORM_READOUT": {"multi_loop_decayed": {"complete_expected_rows": 48, "intervention_rows": 48, "task_count": 6}, "single_loop_L1": {"complete_expected_rows": 48, "intervention_rows": 48, "task_count": 6}}}
MULTILOOP_DECAYED_VS_L1_DELTA = -0.10560436938609094
Interpretation: empirical directions test whether BG is readout-only or whether calibrated success-space directions can become causal handles.
Full reports: `artifacts/reports/probes/bg_empirical_steering_direction_2026-05-18/summary.md`, `artifacts/reports/probes/bg_empirical_steering_direction_2026-05-18/analysis.md`, `docs/evaluator/empirical-steering-direction.md`.

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
