# Selection-Only Phase 2 Prototype V1

This Phase 2a prototype tested branch generation plus fixed-composite top4 branch survival plus final selection. It did not test action steering, train Ouro, modify checkpoints, update tokenizer files, update existing taps, or change production routing.

## Verdicts

BG_SELECTION_ONLY_INVENTORY_VERDICT = READY
BG_SELECTION_ONLY_TASK_SUITE_VERDICT = REUSED_ONLY
BG_SELECTION_ONLY_POLICY_RUNNER_VERDICT = HOOK_LAYERWISE_APPROX
BG_SELECTION_ONLY_CACHED_REPRODUCTION_VERDICT = REPRODUCED
BG_SELECTION_ONLY_LIVE_PROTOTYPE_VERDICT = SURVIVAL_POSITIVE_FINAL_SELECTION_WEAK
BG_SELECTION_ONLY_FINAL_ARBITER_VERDICT = FINAL_SELECTION_WEAK
BG_SELECTION_ONLY_BASELINE_COMPARISON_VERDICT = NO_CLEAR_WINNER
BG_SELECTION_ONLY_DOMAIN_CODING_VERDICT = MULTIDOMAIN_POSITIVE
BG_SELECTION_ONLY_FAILURE_ANALYSIS_VERDICT = FINAL_ARBITER_BLOCKER
BG_SELECTION_ONLY_TO_STEERING_READINESS_VERDICT = NEEDS_FINAL_ARBITER_FIRST
SELECTION_ONLY_PHASE2_PROTOTYPE_STATUS = SURVIVAL_READY_FINAL_ARBITER_WEAK

## Fixed-Composite Context

The prototype used the selected fixed_composite_conservative_top4 operating point from the fixed-composite branch survival policy v1 artifacts.

## BGV1 Branch Generator Context

Completed cached BGV1/v4 hook-intervention branch groups were used in counterfactual replay mode. Layerwise lineage is HOOK_LAYERWISE_APPROX; true fork/carry is not claimed.

## Task Suite

- task-suite verdict: `REUSED_ONLY`
- counts: `{'candidate_sets': 247, 'domains': {'coding': 16, 'math_simple_arithmetic': 8, 'reasoning': 12, 'science': 12}, 'split_source_status': {'reused_diagnostic': 14, 'reused_heldout': 34}, 'tasks': 48}`

## Cached Reproduction

- reproduction verdict: `REPRODUCED`
- selected metrics: `{'n': 173, 'oracle_retention': 0.930635838150289, 'false_prune_rate': 0.06936416184971098, 'average_survivors': 3.8728323699421967, 'top4_oracle_coverage': 0.930635838150289, 'best_selected_reward': 0.7572254335260116, 'mean_selected_reward': 0.3953275529865125, 'final_selected_reward': 0.4092485549132948, 'final_selected_correctness': 0.41040462427745666, 'clean_reward': 0.25780346820809247, 'regret': 0.06936416184971098, 'parse_success_rate': 1.0, 'stable_rate': 1.0, 'task_macro_reward': 0.6330882352941176}`

## Selection-Only Results

- live/counterfactual verdict: `SURVIVAL_POSITIVE_FINAL_SELECTION_WEAK`
- selected metrics: `{'n': 247, 'oracle_retention': 0.951417004048583, 'false_prune_rate': 0.048582995951417005, 'average_survivors': 3.9109311740890687, 'top4_oracle_coverage': 0.951417004048583, 'best_selected_reward': 0.8299595141700404, 'mean_selected_reward': 0.45016869095816464, 'final_selected_reward': 0.545748987854251, 'final_selected_correctness': 0.5465587044534413, 'clean_reward': 0.4234817813765182, 'regret': 0.048582995951417005, 'parse_success_rate': 1.0, 'stable_rate': 1.0, 'task_macro_reward': 0.6671875}`

## Final Arbiter

- arbiter verdict: `FINAL_SELECTION_WEAK`
- best policy: `majority_rank_aggregation_among_survivors`

## Baselines And Domain/Coding

- baseline verdict: `NO_CLEAR_WINNER`
- domain/coding verdict: `MULTIDOMAIN_POSITIVE`
- coding status: `PRESERVED`

## Failure Analysis

- failure verdict: `FINAL_ARBITER_BLOCKER`
- failure summary: `{'clean_branch_beat_generated': 137, 'missing_expert_fallback_relevant': 8, 'survival_succeeded_final_arbiter_failed': 79, 'top4_missed_oracle': 12}`

## Steering Readiness

- steering-readiness verdict: `NEEDS_FINAL_ARBITER_FIRST`
- recommendation: Train or evaluate a stronger final arbiter among top4 survivors before steering.

## Explicit Non-Claims

- No action steering was tested.
- No steering-vector intervention is a tested condition.
- No production routing change was made.
- No compute savings are claimed.
- No true branch-batch fork/carry is claimed.

## Final arbiter among top4 survivors v1 (2026-05-18)

FINAL_ARBITER_TOP4_STATUS = FINAL_ARBITER_WEAK_BUT_USEFUL
SELECTION_ONLY_PHASE2A_STATUS_AFTER_FINAL_ARBITER = NEEDS_MORE_FINAL_ARBITER_WORK

- heldout eval: `FINAL_ARBITER_WEAK`
- selected model: `listwise_softmax`
- readiness: `FINAL_ARBITER_WEAK_BUT_IMPROVED`
- recommendation: Run a small improved-arbiter v1.1 or proceed only with explicit weak-baseline caveat.
- no action steering was tested.

## Final arbiter among top4 survivors v1.1 (2026-05-18)

FINAL_ARBITER_TOP4_V1_1_STATUS = NO_IMPROVEMENT
SELECTION_ONLY_PHASE2A_STATUS_AFTER_FINAL_ARBITER_V1_1 = NEEDS_DOMAIN_SPECIALIZATION

- split guard: `FRESH_HELDOUT_READY`
- selected model: `tie_aware_rank_listwise`
- heldout eval: `NO_IMPROVEMENT`
- readiness: `NEEDS_REASONING_ARBITER`
- recommendation: Return to expert/bridge signal quality; v1.1 did not improve final selection.
- no action steering was tested.

## Weight-space merged taps proposal (2026-05-18)

Selection-only survival remains strong, but final arbitration is weak. The next proposed experiment is below the policy layer: extract old/universal/hidden/bridge tiny-head weights and test old-preserving branch-validity merged taps as either compact survival selectors or improved final-arbiter features.

- planning doc: `docs/evaluator/bg_weight_space_merged_taps_plan.md`
- no merged-weight run has been executed yet.
- no action steering or production routing change is implied.

## Weight-space merged branch-content taps v1 (2026-05-18)

`MERGED_WEIGHT_BRANCH_CONTENT_TAP_STATUS = FINAL_ARBITER_IMPROVES_ONLY`. The run extracted old/content, hidden-branch, and bridge tap directions, aligned them into shared feature coordinates, built residualized merged candidates, and acquired top4 survivor hidden features from cached raw artifacts for final-arbiter rescoring. No action steering or routing change was tested.

Report: `docs/evaluator/bg_merged_weight_branch_content_taps_v1.md`.

## Merged tap final arbiter integration v1 (2026-05-18)

`MERGED_TAP_FINAL_ARBITER_INTEGRATION_STATUS = MERGED_TOP1_USEFUL`; `SELECTION_ONLY_PHASE2A_STATUS_AFTER_MERGED_TAP_INTEGRATION = USE_MERGED_TAP_TOP1_AS_ARBITER_BUT_NOT_READY`. The selected validation policy was `old_code_reasoning_top1` with fresh-holdout task macro `0.5610`. This is final-arbiter integration only; no action steering or routing change was tested.

Report: `docs/evaluator/bg_merged_tap_final_arbiter_integration_v1.md`.

## Merged tap final arbiter integration v1.1 (2026-05-18)

`MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_STATUS = DOMAIN_FALLBACK_USEFUL_BUT_REASONING_LIMITED`; `SELECTION_ONLY_PHASE2A_STATUS_AFTER_MERGED_TAP_INTEGRATION_V1_1 = NEEDS_REASONING_ARBITER`. Grouped task-disjoint CV best readiness-eligible policy was `math_universal_reasoning_universal_else_merged` with task macro `0.7736`; merged tap top1 was `0.7581`. This is final-arbiter integration only; no action steering or routing change was tested.

Report: `docs/evaluator/bg_merged_tap_final_arbiter_integration_v1_1.md`.

## Two-tap branch selector v1 (2026-05-30)

`BG_TWO_TAP_BRANCH_SELECTOR_STATUS = TWO_TAP_BRANCH_SELECTOR_READY`. Selected `two_tap_equal` using only the transplanted `MIX_CODE_REASONING` and `MIX_OBJECTIVE_ALL` taps. Heldout oracle retention `0.9825`, false prune `0.0175`, avg survivors `4.0000`. No action steering, routing change, Ouro training, or tap-registry update was performed.

Report: `docs/evaluator/bg_two_tap_branch_selector_v1.md`.

## DualAnchor architecture-looped stratified probe v3 (2026-05-31)

Status: `ARCHITECTURE_LOOPED_SURVIVAL_READY_TERMINAL_DEFER_REQUIRED`.

This run scaled the DualAnchor architecture-shaped loop without steering. Taps were active at layers 24, 36, and 47 across loops L1-L4, with only terminal `L4_47` eligible for confidence-gated collapse. It uses cumulative hook approximation at decoder-layer surfaces; it does not claim autoregressive branch-specific KV/cache fork/carry or compute savings.

Headline metrics:

- tasks: `48`
- stage oracle retention: `0.9848484848484849`
- terminal oracle retained: `1.0`
- terminal forced top1 oracle: `0.9166666666666666`
- terminal reward-diverse rate: `0.22916666666666666`
- positive-oracle rate: `0.3541666666666667`

Locked-baseline candidate:

- selector: DualAnchor `MIX_CODE_REASONING` + `MIX_OBJECTIVE_ALL`
- schedule: `L1_24 -> L1_36 -> L1_47 -> L2_24 -> L2_36 -> L2_47 -> L3_24 -> L3_36 -> L3_47 -> L4_24 -> L4_36 -> terminal L4_47`
- threshold: `mean_floor_very_loose`
- budget: `8`
- L47: active in nonterminal loops
- terminal: confidence-gated top1; otherwise defer/keep terminal survivors

Readiness verdict: `READY_WITH_TERMINAL_DEFER`.
No steering was tested.

