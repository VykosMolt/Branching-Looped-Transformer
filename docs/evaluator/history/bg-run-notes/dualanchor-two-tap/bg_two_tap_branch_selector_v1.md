<!-- docs-consolidation-source-note -->
> Consolidation note (2026-05-31): this is a source run note. The current consolidated interpretation is in `dualanchor-tap-evolution.md`. Exact pre-consolidation text is archived under `docs/evaluator/history/pre_docs_consolidation_2026-05-31/`.

# Two-Tap Branch Selector v1

BG_TWO_TAP_BRANCH_SELECTOR_STATUS = TWO_TAP_BRANCH_SELECTOR_READY

This run tested whether the old-anchored `coding_reasoning` and `mixed_objective_all` transplanted taps can be the only scoring taps for top4 branch selection. It did not use the old+branch+bridge+universal fixed composite as the scoring policy, did not train Ouro, did not run steering, and did not change routing.

## Result

- selected policy: `two_tap_equal`
- heldout oracle retention: `0.9825`
- heldout false prune: `0.0175`
- heldout avg survivors: `4.0000`
- fixed-composite reference retention / false prune: `0.9310` / `0.0690`
- status: `TWO_TAP_BRANCH_SELECTOR_READY`

## Interpretation

The two transplanted old-anchored taps are sufficient for the cached branch-group survival proxy if they meet the selected retention/false-prune thresholds. This is not a production replacement because veto/rescue and missing/OOD guardrails were not reimplemented in the two-tap-only policy.

## Files

- report: `artifacts/reports/probes/bg_two_tap_branch_selector_v1_2026-05-30/two_tap_branch_selector_eval.md`
- artifact: `artifacts/reports/probes/bg_two_tap_branch_selector_v1_2026-05-30/two_tap_branch_selector_v1.pt`
- rows: `artifacts/reports/probes/bg_two_tap_branch_selector_v1_2026-05-30/two_tap_branch_selector_rows.csv`

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
