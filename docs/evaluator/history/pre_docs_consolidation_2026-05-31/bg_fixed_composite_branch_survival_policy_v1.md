# Fixed-Composite Branch Survival Policy V1

This experiment turns the corrected gated selector rerun into a fixed old+branch+bridge composite branch-survival policy with explicit veto/rescue and missing-expert/OOD fallback.

BG_FIXED_COMPOSITE_SURVIVAL_INVENTORY_VERDICT = READY
BG_FIXED_COMPOSITE_SURVIVAL_DATASET_VERDICT = READY
BG_FIXED_COMPOSITE_SURVIVAL_FEATURES_VERDICT = READY
BG_FIXED_COMPOSITE_SURVIVAL_BASELINES_VERDICT = READY
BG_FIXED_COMPOSITE_OPTIMIZATION_VERDICT = OLD_BRANCH_BRIDGE_SUFFICIENT
BG_FIXED_COMPOSITE_VETO_RESCUE_OPTIMIZATION_VERDICT = READY
BG_FIXED_COMPOSITE_LEARNED_RESCUE_VERDICT = WORSE_THAN_RULES
BG_FIXED_COMPOSITE_MISSING_OOD_POLICY_VERDICT = ROBUST
BG_FIXED_COMPOSITE_SURVIVAL_HELDOUT_EVAL_VERDICT = SURVIVAL_READY
BG_FIXED_COMPOSITE_SURVIVAL_FRONTIER_VERDICT = CLEAR_OPERATING_POINT
BG_FIXED_COMPOSITE_LAYER_ORIGIN_DOMAIN_VERDICT = UNIFORM_POLICY_SUFFICIENT
BG_FIXED_COMPOSITE_OLD_CODE_PRESERVATION_VERDICT = PRESERVED
BG_FIXED_COMPOSITE_SELECTION_ONLY_READINESS_VERDICT = READY
FIXED_COMPOSITE_BRANCH_SURVIVAL_POLICY_STATUS = SURVIVAL_READY

## Decision

Proceed to a small selection-only Phase 2 prototype using BGV1 branches, the fixed old+branch+bridge composite, the selected conservative top-k survival operating point, and missing/OOD fallback. Keep veto/rescue as a guardrail, not as a replacement for the selected heldout-ready operating point. selected_policy = fixed_composite_conservative_top4; oracle_retention = 0.931; false_prune_rate = 0.069; avg_survivors = 3.873. Do not claim action steering.

## Safety

- Expert/tap scores are inputs only, not labels.
- Thresholds and rules are selected on validation only.
- Heldout is evaluation only.
- No Ouro, tokenizer, checkpoint, old tap registry, wrapper/local-agent, production routing, or action-steering changes.

## Selection-only Phase 2 prototype v1 (2026-05-18)

SELECTION_ONLY_PHASE2_PROTOTYPE_STATUS = SURVIVAL_READY_FINAL_ARBITER_WEAK

- cached reproduction: `REPRODUCED`
- live/counterfactual prototype: `SURVIVAL_POSITIVE_FINAL_SELECTION_WEAK`
- final arbiter: `FINAL_SELECTION_WEAK`
- steering readiness: `NEEDS_FINAL_ARBITER_FIRST`
- recommendation: Train or evaluate a stronger final arbiter among top4 survivors before steering.
- no action steering was tested; no production routing changed.

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

The fixed score-level old+branch+bridge composite remains the policy to beat. A proposed follow-up is to extract compatible tiny-head weight vectors and test whether old-preserving branch/bridge residual merges can compact the composite without degrading old-context/coding preservation.

- planning doc: `docs/evaluator/bg_weight_space_merged_taps_plan.md`
- promotion bar: match or beat fixed-composite top4 survival and preserve old coding/reasoning behavior.
- status: not yet run; no production routing change.

## Weight-space merged branch-content taps v1 (2026-05-18)

`MERGED_WEIGHT_BRANCH_CONTENT_TAP_STATUS = FINAL_ARBITER_IMPROVES_ONLY`. The run extracted old/content, hidden-branch, and bridge tap directions, aligned them into shared feature coordinates, built residualized merged candidates, and acquired top4 survivor hidden features from cached raw artifacts for final-arbiter rescoring. No action steering or routing change was tested.

Report: `docs/evaluator/bg_merged_weight_branch_content_taps_v1.md`.

## Merged tap final arbiter integration v1 (2026-05-18)

`MERGED_TAP_FINAL_ARBITER_INTEGRATION_STATUS = MERGED_TOP1_USEFUL`; `SELECTION_ONLY_PHASE2A_STATUS_AFTER_MERGED_TAP_INTEGRATION = USE_MERGED_TAP_TOP1_AS_ARBITER_BUT_NOT_READY`. The selected validation policy was `old_code_reasoning_top1` with fresh-holdout task macro `0.5610`. This is final-arbiter integration only; no action steering or routing change was tested.

Report: `docs/evaluator/bg_merged_tap_final_arbiter_integration_v1.md`.

## Merged tap final arbiter integration v1.1 (2026-05-18)

`MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_STATUS = DOMAIN_FALLBACK_USEFUL_BUT_REASONING_LIMITED`; `SELECTION_ONLY_PHASE2A_STATUS_AFTER_MERGED_TAP_INTEGRATION_V1_1 = NEEDS_REASONING_ARBITER`. Grouped task-disjoint CV best readiness-eligible policy was `math_universal_reasoning_universal_else_merged` with task macro `0.7736`; merged tap top1 was `0.7581`. This is final-arbiter integration only; no action steering or routing change was tested.

Report: `docs/evaluator/bg_merged_tap_final_arbiter_integration_v1_1.md`.

## Old-anchored branch-valid taps v1 (2026-05-30)

`BG_OLD_ANCHORED_BRANCH_VALID_TAP_STATUS = OLD_ANCHORED_BRANCH_TAP_USEFUL`. Tested weight-space transplant and constrained fine-tuned copies of old coding/reasoning and mixed-objective anchors. Selected `weight_space_transplant` candidate `transplant::MIX_CODE_REASONING::full_residual::AntisymLinearNoNorm::102` with heldout old/code drops `0.0000` / `0.0000` and branch/bridge gains `0.1594` / `0.2121` vs its matching old anchor. No old taps, registries, Ouro weights, routing, or steering were modified.

Report: `docs/evaluator/bg_old_anchored_branch_valid_taps_v1.md`.

## Two-tap branch selector v1 (2026-05-30)

`BG_TWO_TAP_BRANCH_SELECTOR_STATUS = TWO_TAP_BRANCH_SELECTOR_READY`. Selected `two_tap_equal` using only the transplanted `MIX_CODE_REASONING` and `MIX_OBJECTIVE_ALL` taps. Heldout oracle retention `0.9825`, false prune `0.0175`, avg survivors `4.0000`. No action steering, routing change, Ouro training, or tap-registry update was performed.

Report: `docs/evaluator/bg_two_tap_branch_selector_v1.md`.

## Two-tap full readiness v1 (2026-05-30)

`BG_TWO_TAP_FULL_READINESS_VERDICT = TWO_TAP_PARTIAL_NOT_READY`. Tested the transplanted `MIX_CODE_REASONING` and `MIX_OBJECTIVE_ALL` taps against old-domain references and branch/universal references across cached old-domain and branch datasets. Domain ok `False`; branch ok `False`. No Ouro training, steering, registry update, routing change, or production change was performed.

Report: `docs/evaluator/bg_two_tap_full_readiness_v1.md`.

## Two-tap gap-targeted v2 (2026-05-30)

`BG_TWO_TAP_GAP_TARGETED_V2_STATUS = TWO_TAP_GAP_TARGETED_NOT_READY`. Selected `old_domain_preserve_repair` on validation only. Clean heldout status `TWO_TAP_PARTIAL_NOT_READY`; full replay status `TWO_TAP_PARTIAL_NOT_READY`. Only copied tap vectors were trained; no old taps, registries, Ouro weights, steering, routing, or production behavior were modified.

Report: `docs/evaluator/bg_two_tap_gap_targeted_v2.md`.

## Layer-native two-tap readiness v1 (2026-05-30)

`BG_LAYER_NATIVE_TWO_TAP_READINESS_VERDICT = LAYER_NATIVE_TWO_TAP_PARTIAL_NOT_READY`. Re-tested `MIX_CODE_REASONING` and `MIX_OBJECTIVE_ALL` as native `24_L4`, `36_L4`, and `47_L4` tap bundles instead of concat-only taps. Domain ok `False`; branch ok `False`. No Ouro training, old registry update, steering, wrapper/local-agent execution, or routing change was performed.

Report: `docs/evaluator/bg_layer_native_two_tap_readiness_v1.md`.

## Layer-native two-tap constrained training v1 (2026-05-30)

`BG_LAYER_NATIVE_TWO_TAP_CONSTRAINED_TRAINING_VERDICT = CONSTRAINED_TWO_TAP_NOT_READY`; readiness verdict `LAYER_NATIVE_TWO_TAP_PARTIAL_NOT_READY`. Trained copied `MIX_CODE_REASONING` and `MIX_OBJECTIVE_ALL` layer-local taps at `24_L4`, `36_L4`, and `47_L4` on train-split old-domain plus branch/bridge labels with anchor preservation. Domain ok `False`; branch ok `False`. No Ouro training, old registry update, steering, wrapper/local-agent execution, or routing change was performed.

Report: `docs/evaluator/bg_layer_native_two_tap_constrained_train_v1.md`.

## Layer-native two-tap targeted rehost diagnostic v1 (2026-05-30)

`BG_LAYER_NATIVE_TWO_TAP_TARGETED_REHOST_DIAGNOSTIC_VERDICT = TARGETED_REHOST_DIAGNOSTIC_PASSES`. Diagnostic-only exact source rehost under the two tap identities produced readiness verdict `LAYER_NATIVE_TWO_TAP_READY`. This is not clean readiness because target references came from prior branch failure reports.

Report: `docs/evaluator/bg_layer_native_two_tap_targeted_rehost_diagnostic_v1.md`.

## DualAnchor branch-gap repair v1 (2026-05-30)

`BG_DUALANCHOR_BRANCH_GAP_REPAIR_VERDICT = DUALANCHOR_BRANCH_NOT_READY`; readiness verdict `LAYER_NATIVE_TWO_TAP_PARTIAL_NOT_READY`. Targeted copied `MIX_CODE_REASONING` and `MIX_OBJECTIVE_ALL` layer-native heads at `24_L4`, `36_L4`, and `47_L4` on train-split branch-gap examples with old-domain preservation. Best-candidate domain ok `False`; best-candidate branch ok `False`; selected-bundle branch ok `False`. No Ouro training, old registry update, steering, wrapper/local-agent execution, or routing change was performed.

Report: `docs/evaluator/bg_dualanchor_branch_gap_repair_v1.md`.

## DualAnchor pre-repair fixed-bundle audit v1 (2026-05-30)

`BG_DUALANCHOR_PRE_REPAIR_FIXED_BUNDLE_AUDIT_VERDICT = DUALANCHOR_FIXED_BUNDLE_NOT_READY`. Selected diagnostic fixed bundle `bundle::two_tap_equal::adaptive_balanced_rescue::AntisymLinear::24_36_47`. Ready fixed bundles `0` / `132`; domain-ok fixed bundles `0` / `132`. No training, failed-repair weights, old registry update, steering, or routing change.

Report: `docs/evaluator/bg_dualanchor_pre_repair_fixed_bundle_audit_v1.md`.

## DualAnchor hard-anchor selector v1 (2026-05-30)

`BG_DUALANCHOR_HARD_ANCHOR_SELECTOR_VERDICT = DUALANCHOR_HARD_ANCHOR_NOT_READY`. Selected fixed pre-repair DualAnchor bundle `bundle::two_tap_equal::sparse_old_plus_bridge_30_70_top0p01::AntisymLinear::24_36_47` with validation old-anchor gate `False`. Full replay domain ok `False`; branch ok `False`. No training, failed-repair weights, old registry update, steering, or routing change.

Report: `docs/evaluator/bg_dualanchor_hard_anchor_selector_v1.md`.

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

