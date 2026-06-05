# BG Trajectory Prediction Sweep

## Purpose

This read-only Stage 1 sweep asks where BG trajectory signal is predictive before any steering is attempted.

## Context

It follows the controller, transformer feature capture, best-of-N smoke, steering/routing suite, and wrapper-matched diagnostic. The wrapper is not used here.

## Domains

Headline domains are reasoning MCQ, science MCQ, and GSM8K/simple arithmetic. Code/devil tasks are intentionally excluded from the headline because direct Ouro code reachability was generator-limited.

## Prefix Lengths

Each generated trajectory is sliced at 32, 64, 128, and 256 generated-token checkpoints.

## Configs

The sweep scores locked heads plus compatible existing config-level heads, including 24_L4, 36_L4, 36_mean, 47_L4, 47_concat_L1_L4, and 47_concat_all_loops where artifacts exist.

## Fairness Constraints

All heads score the same generated prefixes. Prefix continuations are generated once and evaluated post-hoc. Answer keys and labels are never included in BG feature text or used during scoring.

## Reachability Logic

Oracle success at a prefix means at least one branch from the same task and prefix length continued to a correct final answer.

## Predictive Summary

`BG_TRAJECTORY_PREDICTION_VERDICT = STRONG`

`BEST_PREDICTIVE_CELL = {'domain': 'reasoning', 'prefix_length': 256, 'head_id': 'mixed::MIX_CODE_REASONING::36_mean::AntisymLinear', 'config': '36_mean', 'architecture': 'AntisymLinear', 'top1_lift': 0.16249999999999998, 'top2_lift': 0.04166666666666663, 'pairwise_accuracy': 0.8536585365853658, 'oracle_success': 0.9, 'n_tasks': 20, 'n_pairwise_comparisons': 41}`

`RECOMMENDED_STEERING_TARGET = {'domain': 'reasoning', 'prefix_length': 256, 'head_id': 'mixed::MIX_CODE_REASONING::36_mean::AntisymLinear', 'head_config': '36_mean', 'architecture': 'AntisymLinear', 'top1_lift': 0.16249999999999998, 'top2_lift': 0.04166666666666663, 'pairwise_accuracy': 0.8536585365853658, 'oracle_success': 0.9}`

## Predictive Heatmap Summary

| domain | prefix | best_head | config | top1_lift | top2_lift | pair_acc | oracle | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | 32 | mixed::MIX_OBJECTIVE_ALL::47_concat_L1_L4::AntisymLinear | 47_concat_L1_L4 | 0.125 | 0.025 | 0.778 | 1.000 | 20 |
| gsm8k | 64 | registry::HH::24_L4::AntisymLinear | 24_L4 | 0.162 | 0.042 | 0.724 | 1.000 | 20 |
| gsm8k | 128 | mixed::MIX_HH_OBJECTIVE::47_L4::AntisymLinearNoNorm | 47_L4 | 0.150 | 0.058 | 0.765 | 1.000 | 20 |
| gsm8k | 256 | mixed::MIX_CODE_REASONING::36_L4::AntisymLinear | 36_L4 | 0.200 | 0.117 | 0.750 | 1.000 | 20 |
| reasoning | 32 | mixed::MIX_REASONING_SCIENCE::36_L4::AntisymLinear | 36_L4 | 0.200 | 0.033 | 0.550 | 0.900 | 20 |
| reasoning | 64 | mixed::MIX_CODE_SCIENCE::36_mean::AntisymLinearNoNorm | 36_mean | 0.262 | 0.117 | 0.638 | 1.000 | 20 |
| reasoning | 128 | mixed::MIX_OBJECTIVE_ALL::47_concat_L1_L4::AntisymLinear | 47_concat_L1_L4 | 0.238 | 0.008 | 0.655 | 0.950 | 20 |
| reasoning | 256 | mixed::MIX_CODE_REASONING::36_mean::AntisymLinear | 36_mean | 0.162 | 0.042 | 0.854 | 0.900 | 20 |
| science | 32 | mixed::MIX_CODE_SCIENCE_MED::47_L4::AntisymLinear | 47_L4 | 0.188 | 0.083 | 0.811 | 0.900 | 20 |
| science | 64 | mixed::MIX_CODE_SCIENCE::36_mean::AntisymLinearNoNorm | 36_mean | 0.125 | -0.008 | 0.750 | 0.900 | 20 |
| science | 128 | mixed::MIX_CODE_SCIENCE::47_concat_all_loops::AntisymLinear | 47_concat_all_loops | 0.025 | -0.017 | 0.647 | 0.950 | 20 |
| science | 256 | locked::objective_mixed | 36_L4 | 0.138 | 0.050 | 0.545 | 0.900 | 20 |

## Caveats

This is read-only evidence. It does not establish that steering along a BG readout direction will improve generation. Stage 2 must use positive, negative, random, and zero controls.

## BG Stage 2 layer-hook follow-up (2026-05-18)

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
