# Stage 2 Steering Sensitivity

## BG Stage 2 Layer-Hook Follow-Up (2026-05-18)

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

Interpretation: BG remains more reliable as a readout selector than as an inference-time steering vector under this protocol.

Full reports: `docs/evaluator/stage2-layerhook-followup.md`, `artifacts/reports/probes/bg_stage2_layerhook_followup_2026-05-18/summary.md`, `artifacts/reports/probes/bg_stage2_layerhook_followup_2026-05-18/analysis.md`.
