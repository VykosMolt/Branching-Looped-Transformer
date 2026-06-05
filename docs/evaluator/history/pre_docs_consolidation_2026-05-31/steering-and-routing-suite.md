# BG Steering And Partial-Trajectory Routing Suite

This suite tested whether the converged BG architecture improves final task success under compute-matched conditions, compared with the same generator without BG selection. It used local Ouro-RLTT, read-only hidden-state capture, and the conservative BG controller.

## What Was Tested

- Shared initial branch pools over 60 tasks.
- BG partial-trajectory routing over the same branch pool as random and first-branch baselines.
- BG compute allocation, stopped early because the full arm was too expensive for the remaining cap.
- Wrapper-matched branch selection gate.
- Guarded tiny-alpha soft hidden-state steering stability pilot.
- Text-prefix branch selection pilot using ordinary text prefixes only.

## Selection, Routing, Allocation, Steering

Selection ranks completed or partial candidate text branches using BG features.

Routing chooses which partial branch receives continuation compute.

Allocation gives more continuation tokens to BG-preferred branches while preserving total token budget.

Steering temporarily nudges activations during generation. In this suite it was only a stability probe, not an architecture change.

## Fairness Constraints

- Same tasks and branch pools for baseline and BG comparisons.
- Same total candidate counts.
- Labels, tests, answer keys, and verifiers used only after generation.
- Random baselines used the same generated pools as BG.
- Oracle baselines were upper bounds only.
- No model weights, tokenizers, checkpoints, or wrapper files were modified.

## Generator Reachability

Reachability was mixed. The early gate passed because reasoning, science, and GSM8K reached oracle success thresholds, but code and devil tasks did not pass the diagnostic oracle threshold.

Full partial routing had oracle success rate `0.793` over 58 evaluable non-devil tasks. This means enough branches were viable overall to test selection value, but code remained generator-limited.

## Results Summary

- `BG_PARTIAL_ROUTING_VERDICT = NEUTRAL`
- `BG_COMPUTE_ALLOCATION_VERDICT = INSUFFICIENT`
- `BG_WRAPPER_MATCHED_VERDICT = SKIPPED`
- `BG_SOFT_STEERING_VERDICT = STABLE_NO_EFFECT`
- `BG_LATENT_BRANCH_SELECTION_VERDICT = HELPS`
- `OVERALL_BG_STEERING_VERDICT = NEUTRAL`

Main partial-routing metrics:

- BG top1: `0.603`
- random top1 expected: `0.560`
- lift: `+0.043`
- BG top2: `0.741`
- random top2 expected: `0.698`
- lift: `+0.043`
- oracle: `0.793`

The lift was positive but below the predeclared `+0.05` threshold, so the main verdict is neutral.

## What Worked

- The end-to-end read-only branch-pool, feature-capture, and BG-routing path ran on 60 tasks.
- BG selection was positive in aggregate and closed most of the top2 oracle gap.
- Text-prefix selection showed a small-pilot `HELPS` result on 8 tasks.
- Tiny soft steering did not destabilize generation at the tested alphas.

## What Failed Or Was Limited

- Code reachability remained weak in the early gate, including both devil tasks.
- Compute allocation was interrupted after 3 tasks and marked insufficient because the full arm projected too long for the remaining suite cap.
- No clean wrapper multi-candidate interface was available without modifying wrapper internals.
- Soft steering showed no clean directional effect; random control was not worse than positive steering.

## Open Questions

- Whether BG partial routing clears the +0.05 threshold with better candidate generation or calibrated domains.
- Whether wrapper-exposed candidate sets make BG selection useful on hard code.
- Whether compute allocation can be evaluated with a cheaper cached-continuation design.
- Whether text-prefix selection remains helpful beyond the small pilot.

## Local-agent candidate export interface (2026-05-18)

- WRAPPER_CANDIDATE_PATH_INVENTORY_VERDICT = READY
- WRAPPER_CANDIDATE_EXPORT_UNIT_VERDICT = PASS
- WRAPPER_CANDIDATE_EXPORT_SMOKE_VERDICT = SKIPPED
- WRAPPER_CANDIDATE_EXPOSURE_VERDICT = READY
- files modified: `src/local_agent/candidate_export.py`, `src/local_agent/candidate_capture.py`, `src/local_agent/ouro_direct.py`, `src/local_agent/ouro_agent_improved.py`
- API path: `src/local_agent/candidate_capture.py`
- report path: `artifacts/reports/probes/local_agent_candidate_exposure_2026-05-18_summary.md`
- interpretation: the skipped wrapper-matched arm now has the missing infrastructure needed to expose wrapper candidate branches without changing normal wrapper behavior.

## Wrapper-matched BG candidate selection (2026-05-18)

- WRAPPER_TRACE_GENERATION_VERDICT = READY
- WRAPPER_CANDIDATE_EVAL_VERDICT = READY
- WRAPPER_GENERATOR_REACHABILITY_VERDICT = REACHABLE
- WRAPPER_BG_FEATURE_VERDICT = READY
- WRAPPER_MATCHED_BG_VERDICT = NEUTRAL
- BG_VS_RANDOM_VERDICT = NEUTRAL
- BG_VS_STAGE_HEURISTIC_VERDICT = NEUTRAL
- WRAPPER_MATCHED_EXPERIMENT_VERDICT = READY
- devil result: both devil tasks were reachable as code-like wrapper traces but no candidate passed tests.
- report paths: `artifacts/reports/probes/wrapper_bg_matched_2026-05-18/summary.md`, `artifacts/reports/probes/wrapper_bg_matched_2026-05-18/analysis.md`, `docs/evaluator/wrapper-matched-bg-selection.md`
- interpretation: wrapper-matched code reachability is better than direct Ouro, but BG selection was neutral because wrapper final already matched the reachable oracle cases.
## BG trajectory prediction sweep (2026-05-18)

BG_TRAJECTORY_PREFLIGHT_VERDICT = `READY`.
BG_TRAJECTORY_TASK_SUITE_VERDICT = `READY`.
BG_TRAJECTORY_PARTIALS_VERDICT = `READY`.
BG_TRAJECTORY_CONTINUATION_VERDICT = `READY`.
BG_TRAJECTORY_PREFIX_FEATURE_VERDICT = `READY`.
BG_TRAJECTORY_PREFIX_SCORE_VERDICT = `READY`.
BG_TRAJECTORY_PREDICTION_VERDICT = `STRONG`.
BEST_PREDICTIVE_CELL = `{'domain': 'reasoning', 'prefix_length': 256, 'head_id': 'mixed::MIX_CODE_REASONING::36_mean::AntisymLinear', 'config': '36_mean', 'architecture': 'AntisymLinear', 'top1_lift': 0.16249999999999998, 'top2_lift': 0.04166666666666663, 'pairwise_accuracy': 0.8536585365853658, 'oracle_success': 0.9, 'n_tasks': 20, 'n_pairwise_comparisons': 41}`.
RECOMMENDED_STEERING_TARGET = `{'domain': 'reasoning', 'prefix_length': 256, 'head_id': 'mixed::MIX_CODE_REASONING::36_mean::AntisymLinear', 'head_config': '36_mean', 'architecture': 'AntisymLinear', 'top1_lift': 0.16249999999999998, 'top2_lift': 0.04166666666666663, 'pairwise_accuracy': 0.8536585365853658, 'oracle_success': 0.9}`.
GENERATOR_REACHABILITY_LIMITED = `false`.
Interpretation: Run a targeted Stage 2 steering-sensitivity probe at the best predictive cell. Measure state movement in the BG-readable direction, output stability, final correctness, and positive-vs-negative-vs-random controls.
Full reports: `artifacts/reports/probes/bg_trajectory_prediction_2026-05-18/summary.md`, `artifacts/reports/probes/bg_trajectory_prediction_2026-05-18/predictive_power.md`, `artifacts/reports/probes/bg_trajectory_prediction_2026-05-18/stage2_recommendation.md`.

