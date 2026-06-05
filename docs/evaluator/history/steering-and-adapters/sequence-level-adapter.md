# BG Sequence-Level Adapter

## Why This Was the Final Frozen-Backbone Test

The prior BG steering-control stack showed that Ouro is BG-readable and mechanically writable, but static and teacher-forced inference-time write paths did not transfer to free generation. This run used actual generated-output reward rather than answer-token teacher forcing.

## Result

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

## Optimizer

- optimizer: `REINFORCE_score_function`
- sampled tokens were treated with a score-function estimator; the run did not backpropagate through sampling.
- reward: correct parsed MCQ answer `+1`, parseable wrong `0`, parse failure/empty/repetition/error penalties.

## Data and Evaluation

- dataset task counts: `{'heldout': 12, 'train': 20, 'val': 8}`
- heldout sampled n: `4`
- heldout adapter success: `0.479`
- heldout best non-adapter success: `0.521`

## Stopping Rule

- applies: `True`
- scope: `safe_alpha_leq_0_02_under_tested_optimizers`
- Phase 2 implication: `consolidate_phase1_phase1_5_and_design_phase2_training_time_integration`

## Reports

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
