# BG Pre-Consolidation Control Probes

## Why These Probes Were Run

The prior steering work validated BG as a readout and selection surface, but raw L2-normalized readout directions and empirical static directions did not yield reliable signed hidden-state control. This bundle tested the remaining bounded explanations before architecture consolidation.

## RMS-Normalization Correction

Earlier layer-hook steering used unit-L2 directions. In 2048 dimensions, that makes the direction RMS roughly 1/sqrt(2048), so alpha 0.02 was only a tiny hidden-RMS perturbation. The RMS-calibrated probe used RMS-unit directions so alpha maps directly to an approximate hidden-RMS fraction.

## Propagation / Decay Findings

`PROPAGATES_TO_LATER_STATES` with decay profile `SURVIVES_32_TOKENS` and logit verdict `LOGITS_SHIFT_DIRECTIONALLY`.

## Text-Prefix Branch Expansion

`WEAK_POSITIVE` over cached non-code text-prefix branch pools. This path remains ordinary text-prefix branch allocation, not hidden-state steering.

## Causal-Gradient Result

`GRADIENT_NO_BETTER_THAN_RANDOM`.

## Final Architecture Implications

- BG inference-time steering: `UNSIGNED_ONLY`
- BG branch allocation: `PROMISING`
- Phase 2 requirement: `TRAINING_REQUIRED`

The model can be nudged in BG-readable state space, but tested directions do not provide reliable signed control; Phase 2 training is required.

## Report Paths

- preflight: `artifacts/reports/probes/bg_preconsolidation_control_probes_2026-05-18/preflight.md`
- rms_steering: `artifacts/reports/probes/bg_preconsolidation_control_probes_2026-05-18/rms_steering_analysis.md`
- rms_row_level: `artifacts/reports/probes/bg_preconsolidation_control_probes_2026-05-18/rms_row_level_analysis.md`
- propagation_decay: `artifacts/reports/probes/bg_preconsolidation_control_probes_2026-05-18/propagation_decay_analysis.md`
- text_prefix_expansion: `artifacts/reports/probes/bg_preconsolidation_control_probes_2026-05-18/text_prefix_expansion_analysis.md`
- causal_gradient: `artifacts/reports/probes/bg_preconsolidation_control_probes_2026-05-18/causal_gradient_probe.md`
- final_analysis: `artifacts/reports/probes/bg_preconsolidation_control_probes_2026-05-18/final_analysis.json`
- summary: `artifacts/reports/probes/bg_preconsolidation_control_probes_2026-05-18/summary.md`
- docs: `docs/evaluator/preconsolidation-control-probes.md`
- prior_layerhook: `artifacts/reports/probes/bg_stage2_layerhook_followup_2026-05-18/summary.md`
- prior_empirical: `artifacts/reports/probes/bg_empirical_steering_direction_2026-05-18/summary.md`

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

## BG sequence-level adapter / final frozen-backbone steering test (2026-05-18)

- BG_SEQUENCE_PARSER_AUDIT_VERDICT: `READY`
- BG_SEQUENCE_REWARD_DISTRIBUTION_VERDICT: `REWARD_SIGNAL_USABLE`
- BG_SEQUENCE_OPTIMIZER_SANITY_MICRO_VERDICT: `OPTIMIZER_MOVES_ADAPTER`
- BG_SEQUENCE_GPU_THROUGHPUT_VERDICT: `OVERNIGHT_FEASIBLE`
- OVERNIGHT_SEQUENCE_ADAPTER_READINESS: `READY`
- BG_SEQUENCE_ADAPTER_PREFLIGHT_VERDICT: `READY`
- BG_SEQUENCE_ADAPTER_DATASET_VERDICT: `PARTIAL`
- BG_SEQUENCE_ADAPTER_IMPLEMENTATION_VERDICT: `READY`
- BG_SEQUENCE_BASELINE_EVAL_VERDICT: `READY`
- BG_SEQUENCE_OPTIMIZER_SANITY_VERDICT: `OPTIMIZER_CAN_LEARN_TRIVIAL_TARGET`
- BG_SEQUENCE_ADAPTER_TRAINING_VERDICT: `SEQUENCE_REWARD_IMPROVES`
- BG_SEQUENCE_ADAPTER_HELDOUT_VERDICT: `NO_ADAPTER_SPECIFIC_TRANSFER`
- BG_SEQUENCE_ADAPTER_TEACHER_FORCED_DIAG_VERDICT: `NO_LOGIT_MARGIN_EFFECT`
- BG_SEQUENCE_ADAPTER_BG_SCORE_DIAG_VERDICT: `BG_SCORE_MOVES`
- BG_SEQUENCE_ADAPTER_GEOMETRY_VERDICT: `MATCHES_PRIOR_DIRECTIONS`
- BG_SEQUENCE_ADAPTER_LEARNING_VERDICT: `LEARNS_SEQUENCE_REWARD`
- BG_SEQUENCE_ADAPTER_VS_RANDOM_VERDICT: `WORSE_THAN_RANDOM`
- BG_SEQUENCE_ADAPTER_STABILITY_VERDICT: `STABLE`
- BG_SEQUENCE_ADAPTER_TRANSFER_VERDICT: `NO_TRANSFER`
- BG_SEQUENCE_LEVEL_ADAPTER_VERDICT: `NO_FROZEN_BACKBONE_WRITE_PATH`
- FROZEN_BACKBONE_INFERENCE_STEERING_STATUS: `CLOSED_UNDER_TESTED_METHODS`
- STOPPING_RULE_APPLIES: `True`
- STOPPING_RULE_SCOPE: `safe_alpha_leq_0_02_under_tested_optimizers`
- RECOMMENDED_NEXT: `consolidate_phase1_phase1_5_and_design_phase2_training_time_integration`
- STOPPING_RULE_SCOPE: `safe_alpha_leq_0_02_under_tested_optimizers`
- report paths:
  - `artifacts/reports/probes/bg_sequence_adapter_quick_preflight_2026-05-18/summary.md`
  - `artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/preflight.md`
  - `artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/sequence_adapter_dataset.md`
  - `artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/implementation_tests.md`
  - `artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/baseline_eval.md`
  - `artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/optimizer_sanity.md`
  - `artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/sequence_training_report.md`
  - `artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/heldout_free_generation_eval.md`
  - `artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/diagnostics.md`
  - `artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/analysis.md`
  - `artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/summary.md`
