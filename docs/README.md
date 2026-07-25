# Docs

Documentation is grouped by project area. The master tree and component maps
remain at the repository root as `../PROJECT_TREE_MAP.md` and
`../PROJECT_COMPONENTS.md`.

- `project/`: state indexes and standalone project memos.
- `evaluator/`: pairwise evaluator and domain-transfer notes.
- `local_agent/`: local-agent wrapper state and historical snapshots.
- `hunter_seeker_state/`: chunked Hunter-Seeker/Ouro project state, imported
  historical memos, and preservation manifest.

Project state entry points:

- `project/PROJECT_STATE.md`
- `project/PROJECT_STATE_HUNTER_SEEKER.md`
- `evaluator/README.md`
- `evaluator/current-state.md`
- `evaluator/hidden-origin-branch-generator-v1.md`
- `evaluator/universal-branch-content-taps-v1.md`
- `evaluator/gated-branch-content-selector-v1.md`
- `evaluator/bg_weight_space_merged_taps_plan.md`
- `local_agent/PROJECT_STATE_LOCAL_AGENT.md`

## Weight-space merged branch-content taps v1 (2026-05-18)

Added `docs/evaluator/bg_merged_weight_branch_content_taps_v1.md` for the cached merged-tap weight extraction, residualization, and evaluation run. Status: `FINAL_ARBITER_IMPROVES_ONLY`.

## Merged tap final arbiter integration v1 (2026-05-18)

Added `docs/evaluator/bg_merged_tap_final_arbiter_integration_v1.md`. Status: `MERGED_TOP1_USEFUL`; readiness: `USE_MERGED_TAP_TOP1_AS_ARBITER_BUT_NOT_READY`.

## Merged tap final arbiter integration v1.1 (2026-05-18)

Added `docs/evaluator/bg_merged_tap_final_arbiter_integration_v1_1.md`. Status: `DOMAIN_FALLBACK_USEFUL_BUT_REASONING_LIMITED`; readiness: `NEEDS_REASONING_ARBITER`.

## Old-anchored branch-valid taps v1 (2026-05-30)

Added `docs/evaluator/bg_old_anchored_branch_valid_taps_v1.md`. Status: `OLD_ANCHORED_BRANCH_TAP_USEFUL`; selected family: `weight_space_transplant`.

## Two-tap branch selector v1 (2026-05-30)

Added `docs/evaluator/bg_two_tap_branch_selector_v1.md`. Status: `TWO_TAP_BRANCH_SELECTOR_READY`; selected policy: `two_tap_equal`.

## Two-tap full readiness v1 (2026-05-30)

Added `docs/evaluator/bg_two_tap_full_readiness_v1.md`. Status: `TWO_TAP_PARTIAL_NOT_READY`.

## Two-tap gap-targeted v2 (2026-05-30)

Added `docs/evaluator/bg_two_tap_gap_targeted_v2.md`. Status: `TWO_TAP_GAP_TARGETED_NOT_READY`.

## Layer-native two-tap readiness v1 (2026-05-30)

Added `docs/evaluator/bg_layer_native_two_tap_readiness_v1.md`. Status: `LAYER_NATIVE_TWO_TAP_PARTIAL_NOT_READY`.

## Layer-native two-tap constrained training v1 (2026-05-30)

Added `docs/evaluator/bg_layer_native_two_tap_constrained_train_v1.md`. Status: `CONSTRAINED_TWO_TAP_NOT_READY`; readiness `LAYER_NATIVE_TWO_TAP_PARTIAL_NOT_READY`.

## Layer-native two-tap targeted rehost diagnostic v1 (2026-05-30)

Added `docs/evaluator/bg_layer_native_two_tap_targeted_rehost_diagnostic_v1.md`. Status: `TARGETED_REHOST_DIAGNOSTIC_PASSES`.

## DualAnchor branch-gap repair v1 (2026-05-30)

Added `docs/evaluator/bg_dualanchor_branch_gap_repair_v1.md`. Status: `DUALANCHOR_BRANCH_NOT_READY`; readiness `LAYER_NATIVE_TWO_TAP_PARTIAL_NOT_READY`.

## DualAnchor pre-repair fixed-bundle audit v1 (2026-05-30)

Added `docs/evaluator/bg_dualanchor_pre_repair_fixed_bundle_audit_v1.md`. Status: `DUALANCHOR_FIXED_BUNDLE_NOT_READY`.

## DualAnchor hard-anchor selector v1 (2026-05-30)

Added `docs/evaluator/bg_dualanchor_hard_anchor_selector_v1.md`. Status: `DUALANCHOR_HARD_ANCHOR_NOT_READY`.
