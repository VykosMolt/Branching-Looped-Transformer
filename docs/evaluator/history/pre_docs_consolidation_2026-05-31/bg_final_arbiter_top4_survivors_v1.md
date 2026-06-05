# Final Arbiter Among Top4 Survivors V1

This experiment addresses the Phase 2a blocker where fixed-composite top4 survival retained good branches, but final selection among survivors was weak. It trains only small standalone final-arbiter models over cached survivor sets.

## Verdicts

BG_FINAL_ARBITER_INVENTORY_VERDICT = READY
BG_FINAL_ARBITER_DATASET_VERDICT = READY
BG_FINAL_ARBITER_FEATURES_VERDICT = READY
BG_FINAL_ARBITER_BASELINES_VERDICT = READY
BG_FINAL_ARBITER_TRAINING_VERDICT = READY
BG_FINAL_ARBITER_HELDOUT_EVAL_VERDICT = FINAL_ARBITER_WEAK
BG_FINAL_ARBITER_DOMAIN_VERDICT = MULTIDOMAIN_READY
BG_FINAL_ARBITER_EXPERT_ABLATION_VERDICT = BRIDGE_MATTERS
BG_FINAL_ARBITER_CALIBRATION_OOD_VERDICT = CALIBRATION_WEAK
BG_FINAL_ARBITER_FAILURE_ANALYSIS_VERDICT = FAILURES_UNDERSTOOD
BG_FINAL_ARBITER_SELECTION_ONLY_READINESS_VERDICT = FINAL_ARBITER_WEAK_BUT_IMPROVED
FINAL_ARBITER_TOP4_STATUS = FINAL_ARBITER_WEAK_BUT_USEFUL
SELECTION_ONLY_PHASE2A_STATUS_AFTER_FINAL_ARBITER = NEEDS_MORE_FINAL_ARBITER_WORK

## Selection-Only Prototype Context

The prior selection-only run ended at `SURVIVAL_READY_FINAL_ARBITER_WEAK`; top4 survival was strong, but the final arbiter lagged the best-survivor upper bound.

## Dataset

- dataset counts: `{'candidates_by_split': {'heldout': 1102, 'train': 739, 'val': 529}, 'coding_sets': 48, 'current_arbiter_failure_rate': 0.351575456053068, 'domains': {'coding': 48, 'math_simple_arithmetic': 43, 'reasoning': 292, 'science': 220}, 'listwise_sets_by_split': {'heldout': 281, 'train': 189, 'val': 133}, 'oracle_in_top4_rate': 1.0, 'pairwise_by_domain': {'coding': 78, 'math_simple_arithmetic': 143, 'reasoning': 867, 'science': 616}, 'pairwise_by_split': {'heldout': 750, 'train': 561, 'val': 393}, 'pairwise_pairs': 1704, 'science_sets': 220, 'sets_by_domain_split': {'coding::heldout': 16, 'coding::train': 26, 'coding::val': 6, 'math_simple_arithmetic::heldout': 6, 'math_simple_arithmetic::train': 31, 'math_simple_arithmetic::val': 6, 'reasoning::heldout': 72, 'reasoning::train': 106, 'reasoning::val': 114, 'science::heldout': 187, 'science::train': 26, 'science::val': 7}, 'splits': {'heldout': 281, 'train': 189, 'val': 133}, 'survivor_candidates': 2370, 'survivor_sets': 603, 'tasks': 112, 'tie_distribution': {'1': 156, '2': 161, '3': 214, '4': 72}}`
- labels are final reward/correctness/verifier results; tap scores are input features only.
- splits are task-disjoint; heldout is not used for model selection.

## Training

- training verdict: `READY`
- selected model: `listwise_softmax`

## Heldout Results

- heldout verdict: `FINAL_ARBITER_WEAK`
- trained policy: `trained_listwise_softmax`
- success checks: `{'coding_not_degraded': True, 'gap_closure_ge_35pct': False, 'improves_fixed': True, 'improves_majority': True, 'math_not_degraded': True, 'task_macro_final_reward_ge_0_75': False}`

## Domain, Ablation, OOD, Failures

- domain verdict: `MULTIDOMAIN_READY`
- expert ablation verdict: `BRIDGE_MATTERS`
- calibration/OOD verdict: `CALIBRATION_WEAK`
- failure verdict: `FAILURES_UNDERSTOOD`

## Recommendation

Run a small improved-arbiter v1.1 or proceed only with explicit weak-baseline caveat.

No action steering was tested. No production routing changed. No true fork/carry or compute-saving claim is made.

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

Final Arbiter v1/v1.1 did not solve top4 final selection. The proposed next line is to inspect the expert tap directions directly and test old-preserving branch-validity merged weights before adding another learned final arbiter.

- planning doc: `docs/evaluator/bg_weight_space_merged_taps_plan.md`
- no merged-weight run has been executed yet.
- no Phase 2b steering readiness follows from this proposal alone.

## Weight-space merged branch-content taps v1 (2026-05-18)

`MERGED_WEIGHT_BRANCH_CONTENT_TAP_STATUS = FINAL_ARBITER_IMPROVES_ONLY`. The run extracted old/content, hidden-branch, and bridge tap directions, aligned them into shared feature coordinates, built residualized merged candidates, and acquired top4 survivor hidden features from cached raw artifacts for final-arbiter rescoring. No action steering or routing change was tested.

Report: `docs/evaluator/bg_merged_weight_branch_content_taps_v1.md`.
