# Weight-Space Merged Branch-Content Taps v1

Status: completed cached weight-space merged tap experiment.

MERGED_WEIGHT_BRANCH_CONTENT_TAP_STATUS = FINAL_ARBITER_IMPROVES_ONLY

This run tested whether old/content, hidden-branch, and bridge tap weight directions could be aligned into a shared feature coordinate system and residualized into compact merged pairwise taps.

No Ouro weights, tokenizer files, checkpoints, old tap registries, production routing, wrapper/local-agent code, Hunter-Seeker modules, action steering, or true fork/carry paths were modified or executed.

## Key Result

- selected merged tap: `old_plus_branch_plus_bridge_residual::AntisymLinear::456`
- weight inventory: `READY`
- coordinate alignment: `SHARED_CONCAT_READY`
- validation selection: `WEAK`
- old/code eval: `SMALL_DEGRADATION`
- branch/bridge eval: `BRANCH_AND_BRIDGE_READY`
- survival eval: `MERGED_MATCHES_FIXED_COMPOSITE`
- survivor feature acquisition: `READY`
- final arbiter eval: `FINAL_ARBITER_IMPROVES`
- composite insertion: `MERGED_AS_EXTRA_EXPERT`
- diagnostics: `CLEAN_PAIRWISE_BEHAVIOR`

## Interpretation

The experiment keeps LayerNorm and NoNorm heads separated, uses explicit direct or zero-block lifted alignment, and does not use score distillation for primary readiness. A merged tap can only replace a core scoring component unless fixed-composite veto/rescue and missing/OOD guardrails are also preserved or re-applied.

The top4 survivor hidden-feature gap was filled from cached raw branch, old-content, and code feature artifacts. No fresh generation was required for this acquisition pass.

## Acquired Data

- top4 survivor sets covered: `603 / 603`
- survivor candidates covered: `2370 / 2370`
- complete set rate: `1.0000`
- candidate match rate: `1.0000`
- feature sources:
  - `universal_bridge_candidate_rows`: `1668`
  - `old_content_dataset`: `552`
  - `code_expanded_strict_clean_features`: `150`

## Final Arbiter Rescore

Primary split: `fresh_holdout`

| policy | task macro reward | group micro reward | oracle selected | regret |
| --- | --- | --- | --- | --- |
| merged_tap_top1 | 0.7314 | 0.7569 | 0.7843 | 0.2333 |
| fixed_composite_top1 | 0.5610 | 0.7529 | 0.7745 | 0.2373 |
| majority_rank_aggregation | 0.5353 | 0.7137 | 0.7353 | 0.2765 |
| universal_top1 | 0.6854 | 0.6961 | 0.7353 | 0.2941 |
| oracle_best_survivor | 0.9583 | 0.9902 | 1.0000 | 0.0000 |

Merged tap top1 improved final selection on the fresh holdout split, especially compared with fixed-composite top1 and majority-rank aggregation. This supports using the merged tap as a final-arbiter expert, not as a full fixed-composite replacement.

## Remaining Limits

- validation selection was still `WEAK`
- old-context/code preservation was `SMALL_DEGRADATION`
- the selected tap matched survival in a compatible cached proxy, but a single merged vector still does not include fixed-composite veto/rescue and missing/OOD guardrails

## Files

- summary: `artifacts/reports/probes/bg_merged_weight_branch_content_taps_v1_2026-05-18/summary.md`
- analysis: `artifacts/reports/probes/bg_merged_weight_branch_content_taps_v1_2026-05-18/analysis.md`
- merged tap artifact: `artifacts/reports/probes/bg_merged_weight_branch_content_taps_v1_2026-05-18/merged_weight_branch_content_taps_v1.pt`
- selected tap artifact: `artifacts/reports/probes/bg_merged_weight_branch_content_taps_v1_2026-05-18/selected_merged_taps.pt`

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
