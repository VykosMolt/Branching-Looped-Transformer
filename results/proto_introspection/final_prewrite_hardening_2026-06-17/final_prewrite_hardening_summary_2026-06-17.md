# Final Prewrite Hardening Summary 2026-06-17

FINAL_PREWRITE_VERDICT = BEGIN_DRAFT_WITH_NOTED_OPEN_AUDITS

## Executive Summary

Ran three bounded, read-only/cheap audits. Audit A used existing HH-RLHF normal/flipped scores; no feature extraction was rerun. Audit B found missing actual S1 injection tensors and therefore ran only an inventory plus a clearly labeled injection-basis proxy projection. Audit C trained a lightweight grouped logistic selector on existing generated-branch features and verifier labels only.

Paper can proceed, but with a wording downgrade: HH 95.2% must be fixed-order HH-RLHF pairwise accuracy, not strict antisymmetrized accuracy. S1 orthogonality remains a proxy/inventory result. S3B2 is partial: separability improves, selection wall remains.

## Audit A: Strict Antisymmetrized HH Evaluator

- verdict: `FIXED_ORDER_RESULT_PARTLY_ORDER_ARTIFACTED`
- claim status: `SAFE_AS_FIXED_ORDER_WITH_ANTISYM_CAVEAT`
- examples: 512 saved HH-RLHF same-slice pairs
- fixed_order_acc: 0.9668
- antisym_acc: 0.6367
- strict_sign_flip_rate: 0.1895
- flip_correlation: -0.8989

## Audit B: S1/S3 Orthogonality

- verdict: `UNDERPOWERED_OR_ARTIFACTS_MISSING`
- projection headline: outcome -> curated injection-basis proxy 0.004832, residual 0.995168
- sampling span comparison headline: rank-matched sampling projection 0.280282
- caveat: actual S1 h_injected - h_clean tensors were not found.

## Audit C: S3B2 Generated-Branch Correctness Refit

- verdict: `GENERATED_BRANCH_SELECTOR_PARTIAL`
- selection wall verdict: `SELECTION_WALL_REMAINS_AFTER_REFIT`
- AUROC: 0.7515
- pairwise_accuracy: 0.6835
- sel@oracle: 0.6250

## Paper Impact

| claim | impact | notes |
| --- | --- | --- |
| relational hidden-state readout | weakened | HH 95.2 remains fixed-order only. 512-slice strict antisym_acc is 0.6367, so do not describe the 95.2 number as antisymmetrized. |
| pre-answer process-quality signal | unchanged | No new audit touched the GSM8K pre-answer results. |
| domain-transfer taps | unchanged | S3B2 does not alter prior transfer caveats. |
| survival vs selection wall | strengthened | S3B2 shows generated-branch correctness is learnable to some degree from saved features, but top-1 selection remains unsolved under grouped splits. |
| readout-control boundary | still caveated | Proxy subspace audit is consistent with frozen injection misalignment, but actual S1 injection deltas are missing. |
| S3A motivation | strengthened | S1 frozen closure + S3B2 partial refit + proxy misalignment support training-time integration, with mechanism caveat. |

## Final Paper Readiness

Begin drafting with noted open audits. The pre-answer GSM8K result, S1 frozen-fork closure, S3B corrected transfer failure, and paper-writing package remain usable. The HH evaluator section must be rewritten so 95.2% is fixed-order only, with the strict 512-slice antisymmetrized result reported separately or moved to a caveat/appendix.

Next recommended action: draft now with those caveats; do not spend more pre-writing time on S3A/S3C or large generation.
