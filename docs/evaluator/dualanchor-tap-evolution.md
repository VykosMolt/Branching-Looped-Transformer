# DualAnchor Tap Evolution

Updated: 2026-05-31

This document collapses the two-tap, old-anchored, layer-native, repair, and DualAnchor probe line into one readable history.

## Current Conclusion

The current selector identity is:

- `MIX_CODE_REASONING`
- `MIX_OBJECTIVE_ALL`

Together these are called DualAnchor.

They passed the architecture-looped survival test when used at layers 24, 36, and 47 across loops L1-L4, but final terminal collapse must remain gated/deferred.

Current status:

`ARCHITECTURE_LOOPED_SURVIVAL_READY_TERMINAL_DEFER_REQUIRED`

## Why These Taps

The old `coding_reasoning` and `mixed_objective_all` directions were the strongest content/action readouts. The later branch work showed that useful branch-validity signal could be transplanted or approximated around those anchors.

The design goal became:

`old coding/reasoning + mixed objective content signal + branch validity = two branch/action taps`

not:

`old content taps + separate branch taps + separate bridge taps forever`

## Evolution

| Step | Status | What It Proved |
| --- | --- | --- |
| old-anchored branch-valid taps v1 | `OLD_ANCHORED_BRANCH_TAP_USEFUL` | Weight-space transplant can add branch/bridge gain without old/code drop. |
| two-tap branch selector v1 | `TWO_TAP_BRANCH_SELECTOR_READY` | The two taps can score cached branch survival groups with high retention. |
| two-tap full readiness v1 | `TWO_TAP_PARTIAL_NOT_READY` | Full old-domain + all branch/universal readiness was not yet cleared. |
| fresh dataset comparison v1 | `TWO_TAP_MATCHES_OR_BEATS_OLD_BG_ON_FRESH` | On fresh same-content domains, new two-tap slightly beat old BG overall. |
| HH-RLHF comparison v1 | `TWO_TAP_MATCHES_OR_BEATS_ALL_OLD_BG_ON_HH_RLHF` | On 512 HH pairs, new two-tap beat old BG modestly, though absolute accuracy stayed modest. |
| layer-native two-tap readiness v1 | `DOMAIN_READY_BRANCH_GAP` | Native 24/36/47 layer-local taps preserved domain behavior but still had branch gaps. |
| constrained layer-native training v1 | `DOMAIN_READY_BRANCH_GAP` | Copied taps could be trained with old-anchor penalties; branch gap remained. |
| targeted rehost diagnostic | `TARGETED_REHOST_DIAGNOSTIC_PASSES` | Mechanical coordinate compatibility exists, but this is diagnostic because targets came from known failure reports. |
| branch-gap repair v1 | `DUALANCHOR_BRANCH_GAP_REDUCED` | Repair reduced gaps but did not produce a clean fixed-bundle readiness pass. |
| fixed-bundle audit / hard-anchor selector | `NOT_READY` | Static bundles were not enough before architecture-looped evaluation. |
| architecture-looped v3 | `READY_WITH_TERMINAL_DEFER` | All-loop repeated branch/prune survival works; terminal forced collapse remains the weak point. |

## Important Metric Anchors

Two-tap branch selector v1:

- heldout oracle retention: `0.9825`
- false prune: `0.0175`
- avg survivors: `4.0000`

Fresh dataset comparison v1:

- tasks: `192`
- candidates: `742`
- pairs: `550`
- best old BG accuracy: `0.8800`
- best new two-tap accuracy: `0.8818`

HH-RLHF comparison v1:

- pairs: `512`
- best old BG accuracy: `0.5977`
- best new two-tap accuracy: `0.6152`

Architecture-looped v3:

- stage oracle retention: `0.9848`
- terminal oracle retained: `1.0000`
- forced terminal top1 oracle: `0.9167`
- reward-diverse forced top1 oracle: `0.6364`

## Current Design Rule

Use DualAnchor as the branch/action/content selector in the architecture loop.

Do not use unconditional final top1. Terminal `L4_47` is only allowed to collapse when confidence is high; otherwise keep/defer the terminal survivors.

Do not revive the old framing where one tap family is only content and another is only branching unless a future ablation forces that split.

## Source Run Notes

- `history/bg-run-notes/dualanchor-two-tap/bg_old_anchored_branch_valid_taps_v1.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_two_tap_branch_selector_v1.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_two_tap_full_readiness_v1.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_two_tap_fresh_dataset_comparison_v1.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_two_tap_hh_rlhf_comparison_v1.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_two_tap_gap_targeted_v2.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_layer_native_two_tap_readiness_v1.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_layer_native_two_tap_constrained_train_v1.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_layer_native_two_tap_targeted_rehost_diagnostic_v1.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_dualanchor_branch_gap_repair_v1.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_dualanchor_pre_repair_fixed_bundle_audit_v1.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_dualanchor_hard_anchor_selector_v1.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_dualanchor_architecture_looped_stratified_probe_v3.md`

