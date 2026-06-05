<!-- docs-consolidation-source-note -->
> Consolidation note (2026-05-31): this is a source run note. The current consolidated interpretation is in `dualanchor-tap-evolution.md`. Exact pre-consolidation text is archived under `docs/evaluator/history/pre_docs_consolidation_2026-05-31/`.

# Two-Tap Full Readiness v1

BG_TWO_TAP_FULL_READINESS_VERDICT = TWO_TAP_PARTIAL_NOT_READY

This probe tests whether the old-anchored `coding_reasoning` and `mixed_objective_all` transplanted taps can be the only taps for both old-domain scoring and branch scoring.

## Result

- domain ok: `False`
- branch ok: `False`
- status: `TWO_TAP_PARTIAL_NOT_READY`

## Caveat

The fixed-composite survival score-matrix cache cannot be directly rescored by new taps because raw hidden features are not stored there. Raw-feature branch evaluation uses the universal bridge candidate rows and top4 survivor hidden-feature cache.

## Files

- report: `artifacts/reports/probes/bg_two_tap_full_readiness_v1_2026-05-30/two_tap_full_readiness.md`
- rows: `artifacts/reports/probes/bg_two_tap_full_readiness_v1_2026-05-30/two_tap_full_readiness_pair_rows.csv`, `artifacts/reports/probes/bg_two_tap_full_readiness_v1_2026-05-30/two_tap_full_readiness_group_rows.csv`

## Two-tap gap-targeted v2 (2026-05-30)

`BG_TWO_TAP_GAP_TARGETED_V2_STATUS = TWO_TAP_GAP_TARGETED_NOT_READY`. Selected `old_domain_preserve_repair` on validation only. Clean heldout status `TWO_TAP_PARTIAL_NOT_READY`; full replay status `TWO_TAP_PARTIAL_NOT_READY`. Only copied tap vectors were trained; no old taps, registries, Ouro weights, steering, routing, or production behavior were modified.

Report: `docs/evaluator/bg_two_tap_gap_targeted_v2.md`.
