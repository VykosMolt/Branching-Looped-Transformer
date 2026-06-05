# BG Causal Intervention Adapter

This experiment follows the static steering failures: raw NoNorm, empirical mean-diff, whitened, logistic, and classifier-style adapter directions did not give reliable signed causal control.

The adapter is trained as a causal write-path. Ouro and BG heads stay frozen; only a tiny adapter can update. The primary objective is teacher-forced correct-option logit control, not BG-score optimization.

Teacher-forced margin improvement is necessary but not sufficient because it has direct access to answer logits. Free-generation transfer is the load-bearing result.

## Methodology

- adapter variants implemented: `Rank1GatedDirectionAdapter`, `LowRankDeltaAdapter`, `HyperDirectionAdapter`
- trained variant: `LowRankDeltaAdapter(rank=32)`
- primary mode: `multi_loop_decayed`
- intervention layer: `36`
- intervention position: `prefix_last_token`, before the `FINAL ANSWER:` suffix
- KL preservation masks the answer position
- use_cache is disabled for intervention runs
- alpha/effective RMS is capped at `0.02`

## Verdicts

- BG_CAUSAL_ADAPTER_PREFLIGHT_VERDICT = `READY`
- BG_CAUSAL_ADAPTER_DATASET_VERDICT = `READY`
- BG_CAUSAL_ADAPTER_IMPLEMENTATION_VERDICT = `READY`
- BG_CAUSAL_ADAPTER_TRAINING_VERDICT = `PARTIAL`
- BG_CAUSAL_ADAPTER_TEACHER_FORCED_VERDICT = `ADAPTER_IMPROVES_LOGIT_MARGIN`
- BG_CAUSAL_ADAPTER_FREE_GEN_VERDICT = `TEACHER_FORCED_ONLY`
- BG_PAIRWISE_CAUSAL_CONTRAST_ADAPTER_VERDICT = `SKIPPED`
- BG_CAUSAL_ADAPTER_LEARNING_VERDICT = `LEARNS_LOGIT_CONTROL`
- BG_CAUSAL_ADAPTER_VS_STATIC_VERDICT = `ADAPTER_BEATS_STATIC`
- BG_CAUSAL_ADAPTER_STABILITY_VERDICT = `STABLE`
- BG_CAUSAL_ADAPTER_GENERATION_TRANSFER_VERDICT = `TEACHER_FORCED_ONLY`
- BG_CAUSAL_ADAPTER_VERDICT = `LOCAL_LOGIT_CONTROL_ONLY`

## Interpretation

The causal adapter test separates local teacher-forced logit control from actual trajectory transfer; overall verdict is LOCAL_LOGIT_CONTROL_ONLY.

## Report Paths

- summary: `artifacts/reports/probes/bg_causal_intervention_adapter_2026-05-18/summary.md`
- analysis: `artifacts/reports/probes/bg_causal_intervention_adapter_2026-05-18/analysis.md`
- training report: `artifacts/reports/probes/bg_causal_intervention_adapter_2026-05-18/training_report.md`
- teacher-forced eval: `artifacts/reports/probes/bg_causal_intervention_adapter_2026-05-18/teacher_forced_eval.md`
- free-generation eval: `artifacts/reports/probes/bg_causal_intervention_adapter_2026-05-18/free_generation_eval.md`

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
