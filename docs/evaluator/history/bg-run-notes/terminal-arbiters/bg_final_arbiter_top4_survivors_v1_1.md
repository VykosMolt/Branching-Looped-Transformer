<!-- docs-consolidation-source-note -->
> Consolidation note (2026-05-31): this is a source run note. The current consolidated interpretation is in `terminal-selection-and-arbiters.md`. Exact pre-consolidation text is archived under `docs/evaluator/history/pre_docs_consolidation_2026-05-31/`.

# Final Arbiter Among Top4 Survivors V1.1

V1.1 tests the v1 ablation lead that rank-heavy arbitration generalizes better than the full-feature listwise model. It keeps branch generation and fixed-composite top4 survival unchanged.

## Verdicts

BG_FINAL_ARBITER_V1_1_SPLIT_GUARD_VERDICT = FRESH_HELDOUT_READY
BG_FINAL_ARBITER_V1_1_DATASET_VERDICT = READY
BG_FINAL_ARBITER_V1_1_FEATURES_VERDICT = READY
BG_FINAL_ARBITER_V1_1_BASELINES_VERDICT = READY
BG_FINAL_ARBITER_V1_1_TRAINING_VERDICT = READY
BG_FINAL_ARBITER_V1_1_HELDOUT_EVAL_VERDICT = NO_IMPROVEMENT
BG_FINAL_ARBITER_V1_1_DOMAIN_VERDICT = SCIENCE_IMPROVED
BG_FINAL_ARBITER_V1_1_TIE_VERDICT = TIES_ARE_LABEL_NOISE
BG_FINAL_ARBITER_V1_1_ABLATION_VERDICT = RANKS_DOMINATE
BG_FINAL_ARBITER_V1_1_CALIBRATION_OOD_VERDICT = DOMAIN_OOD_FRAGILE
BG_FINAL_ARBITER_V1_1_FAILURE_ANALYSIS_VERDICT = REASONING_BLOCKER_REMAINS
BG_FINAL_ARBITER_V1_1_SELECTION_READINESS_VERDICT = NEEDS_REASONING_ARBITER
FINAL_ARBITER_TOP4_V1_1_STATUS = NO_IMPROVEMENT
SELECTION_ONLY_PHASE2A_STATUS_AFTER_FINAL_ARBITER_V1_1 = NEEDS_DOMAIN_SPECIALIZATION

## Anti-Leakage Split Strategy

The v1 heldout rank-only result is treated as a hypothesis. V1.1 uses a fresh task-disjoint holdout selected from tasks that were not v1 heldout; previous v1 heldout replay is diagnostic only.

## Training and Evaluation

- selected model: `tie_aware_rank_listwise`
- heldout verdict: `NO_IMPROVEMENT`
- success checks: `{'coding_not_degraded': True, 'gap_closure_ge_35pct': False, 'improves_fixed': False, 'improves_majority': True, 'improves_v1_selected_replay': False, 'math_not_degraded': True, 'science_not_worse_than_v1_reference': True, 'task_macro_final_reward_ge_0_75': False}`

## Domain, Ties, Ablation, OOD, Failures

- domain verdict: `SCIENCE_IMPROVED`
- tie verdict: `TIES_ARE_LABEL_NOISE`
- ablation verdict: `RANKS_DOMINATE`
- calibration/OOD verdict: `DOMAIN_OOD_FRAGILE`
- failure verdict: `REASONING_BLOCKER_REMAINS`

## Recommendation

Return to expert/bridge signal quality; v1.1 did not improve final selection.

No action steering was tested. No production routing changed. No true fork/carry or compute-saving claim is made.

## Weight-space merged taps proposal (2026-05-18)

Because v1.1 did not improve final selection, the next proposed line is below the final arbiter: inspect and merge the expert tap directions themselves. The proposed `bg_weight_space_merged_taps_v1` experiment should test whether old-preserving branch-validity residual directions can improve the features supplied to final arbitration or compact the fixed composite without losing old coding/reasoning behavior.

- planning doc: `docs/evaluator/bg_weight_space_merged_taps_plan.md`
- no merged-weight run has been executed yet.
- do not proceed to Phase 2b steering from this plan alone.

## Weight-space merged branch-content taps v1 (2026-05-18)

`MERGED_WEIGHT_BRANCH_CONTENT_TAP_STATUS = FINAL_ARBITER_IMPROVES_ONLY`. The run extracted old/content, hidden-branch, and bridge tap directions, aligned them into shared feature coordinates, built residualized merged candidates, and acquired top4 survivor hidden features from cached raw artifacts for final-arbiter rescoring. No action steering or routing change was tested.

Report: `docs/evaluator/bg_merged_weight_branch_content_taps_v1.md`.

## Merged tap final arbiter integration v1 (2026-05-18)

`MERGED_TAP_FINAL_ARBITER_INTEGRATION_STATUS = MERGED_TOP1_USEFUL`; `SELECTION_ONLY_PHASE2A_STATUS_AFTER_MERGED_TAP_INTEGRATION = USE_MERGED_TAP_TOP1_AS_ARBITER_BUT_NOT_READY`. The selected validation policy was `old_code_reasoning_top1` with fresh-holdout task macro `0.5610`. This is final-arbiter integration only; no action steering or routing change was tested.

Report: `docs/evaluator/bg_merged_tap_final_arbiter_integration_v1.md`.

## Merged tap final arbiter integration v1.1 (2026-05-18)

`MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_STATUS = DOMAIN_FALLBACK_USEFUL_BUT_REASONING_LIMITED`; `SELECTION_ONLY_PHASE2A_STATUS_AFTER_MERGED_TAP_INTEGRATION_V1_1 = NEEDS_REASONING_ARBITER`. Grouped task-disjoint CV best readiness-eligible policy was `math_universal_reasoning_universal_else_merged` with task macro `0.7736`; merged tap top1 was `0.7581`. This is final-arbiter integration only; no action steering or routing change was tested.

Report: `docs/evaluator/bg_merged_tap_final_arbiter_integration_v1_1.md`.
