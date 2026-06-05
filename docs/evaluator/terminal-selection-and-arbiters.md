# Terminal Selection And Arbiters

Updated: 2026-05-31

This document collapses the final-arbiter, merged-tap integration, and terminal `L4_47` confidence work.

## Current Conclusion

Terminal collapse remains the weak point.

The current policy is:

- forced terminal top1: diagnostic only,
- confidence-gated terminal top1: allowed when the gate fires,
- otherwise defer or keep terminal survivors.

Current status:

`ARCHITECTURE_LOOPED_SURVIVAL_READY_TERMINAL_DEFER_REQUIRED`

## Why This Matters

Fixed-composite top4 and DualAnchor all-loop survival both showed that good branches usually survive. The hard part is choosing one final output among survivors without overfitting to tie-heavy or weak-signal cases.

## Final-Arbiter History

| Stage | Status | Interpretation |
| --- | --- | --- |
| selection-only prototype v1 | `SURVIVAL_READY_FINAL_ARBITER_WEAK` | Good branches survived; final selection lagged best survivor. |
| final arbiter v1 | `FINAL_ARBITER_WEAK_BUT_USEFUL` | Improved over majority/fixed top1, missed 0.75 target. |
| final arbiter v1.1 | `NO_IMPROVEMENT` | Rank-heavy/tie-aware training did not clear readiness. |
| merged weight taps v1 | `FINAL_ARBITER_IMPROVES_ONLY` | Weight-space merged tap helped final choice but not full selector replacement. |
| merged tap final arbiter integration v1 | `MERGED_TOP1_USEFUL` | Useful signal, still not Phase 2a ready. |
| merged tap final arbiter integration v1.1 | `DOMAIN_FALLBACK_USEFUL_BUT_REASONING_LIMITED` | CV task macro improved, but reasoning remained blocker. |
| DualAnchor architecture-looped v3 | `READY_WITH_TERMINAL_DEFER` | Survival ready; terminal confidence weak on hard slice. |

## Key Numbers

Selection-only prototype v1:

- top4 oracle retention: `0.9514`
- task macro best-selected reward: `0.9453`
- task macro final reward: `0.6672`

Final arbiter v1:

- selected model: `listwise_softmax`
- heldout task macro: `0.6680`
- majority heldout task macro: `0.6251`
- fixed top1 heldout task macro: `0.6159`
- oracle best-survivor task macro: `0.8978`

Final arbiter v1.1:

- selected model: `tie_aware_rank_listwise`
- heldout eval: `NO_IMPROVEMENT`
- readiness: `NEEDS_REASONING_ARBITER`

Merged tap integration v1.1:

- best readiness-eligible grouped-CV policy: `math_universal_reasoning_universal_else_merged`
- grouped-CV task macro: `0.7736`
- merged top1 grouped-CV task macro: `0.7581`
- status: `DOMAIN_FALLBACK_USEFUL_BUT_REASONING_LIMITED`

DualAnchor architecture-looped v3:

- terminal oracle retained: `1.0000`
- forced terminal top1 oracle: `0.9167`
- forced terminal top1 reward: `0.2625`
- best terminal reward: `0.3167`
- reward-diverse forced top1 oracle: `0.6364`
- positive and reward-diverse forced top1 oracle: `0.6667`

## Terminal Confidence

V3 confidence calibration:

`BG_DUALANCHOR_TERMINAL_CONFIDENCE_V3_VERDICT = TERMINAL_WEAK_ON_HARD_SLICE`

Hard-slice analysis:

`BG_DUALANCHOR_HARD_SLICE_V3_VERDICT = TERMINAL_CONFIDENCE_ONLY`

Interpretation:

Aggregate terminal top1 looks strong because many tasks are tie-heavy. On the meaningful reward-diverse and positive+reward-diverse slices, unconditional top1 is not reliable enough. The architecture baseline is therefore defer-ready, not terminal-top1-ready.

## Current Rule

At terminal `L4_47`:

1. Score terminal candidates pairwise with both DualAnchor taps.
2. Evaluate forced top1 only as diagnostic.
3. Collapse only if confidence gate fires.
4. If confidence is weak, keep/defer terminal survivors.

No final arbiter should be described as a steering module.

## Source Run Notes

- `history/bg-run-notes/survival-selection/bg_selection_only_phase2_prototype_v1.md`
- `history/bg-run-notes/terminal-arbiters/bg_final_arbiter_top4_survivors_v1.md`
- `history/bg-run-notes/terminal-arbiters/bg_final_arbiter_top4_survivors_v1_1.md`
- `history/bg-run-notes/weight-merge/bg_merged_weight_branch_content_taps_v1.md`
- `history/bg-run-notes/terminal-arbiters/bg_merged_tap_final_arbiter_integration_v1.md`
- `history/bg-run-notes/terminal-arbiters/bg_merged_tap_final_arbiter_integration_v1_1.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_dualanchor_architecture_looped_stratified_probe_v3.md`

