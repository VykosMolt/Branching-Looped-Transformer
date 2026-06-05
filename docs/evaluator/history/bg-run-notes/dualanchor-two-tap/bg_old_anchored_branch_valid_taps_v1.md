<!-- docs-consolidation-source-note -->
> Consolidation note (2026-05-31): this is a source run note. The current consolidated interpretation is in `dualanchor-tap-evolution.md`. Exact pre-consolidation text is archived under `docs/evaluator/history/pre_docs_consolidation_2026-05-31/`.

# Old-Anchored Branch-Valid Taps v1

BG_OLD_ANCHORED_BRANCH_VALID_TAP_STATUS = OLD_ANCHORED_BRANCH_TAP_USEFUL

This run tested both requested routes: direct weight-space transplantation into old coding/reasoning and mixed-objective anchors, and constrained fine-tuning of copied old anchors. It wrote new tap artifacts only and did not overwrite old taps or registries.

## Result

- selected: `transplant::MIX_CODE_REASONING::full_residual::AntisymLinearNoNorm::102`
- selected family: `weight_space_transplant`
- best heldout family: `weight_space_transplant`
- old drop vs matching anchor: `0.0000`
- code drop vs matching anchor: `0.0000`
- hidden-branch gain vs matching anchor: `0.1594`
- bridge gain vs matching anchor: `0.2121`
- survival-retention gain vs matching anchor: `0.0087`

## Interpretation

The selected candidate is validation-selected and heldout-evaluated. It is useful only if branch/bridge/survival gain appears without material old/code degradation. It should be treated as a candidate expert or follow-up anchor, not a production route.

## Files

- report: `artifacts/reports/probes/bg_old_anchored_branch_valid_taps_v1_2026-05-30/old_anchored_branch_valid_taps_v1.md`
- artifact: `artifacts/reports/probes/bg_old_anchored_branch_valid_taps_v1_2026-05-30/old_anchored_branch_valid_taps_v1.pt`
- rows: `artifacts/reports/probes/bg_old_anchored_branch_valid_taps_v1_2026-05-30/old_anchored_branch_valid_tap_rows.csv`

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
