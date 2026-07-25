# Antisymmetrized HH Pairwise Audit 2026-06-17

HH_ANTISYM_AUDIT_VERDICT = FIXED_ORDER_RESULT_PARTLY_ORDER_ARTIFACTED
HH_952_CLAIM_STATUS = SAFE_AS_FIXED_ORDER_WITH_ANTISYM_CAVEAT

## Inputs

- checkpoint: `artifacts/checkpoints/evaluator/pairwise_epoch2.pt`
- score artifact: `artifacts/reports/probes/bg_pairwise_epoch2_vs_tiny_taps_hh_same_slice_v1_2026-05-30/pairwise_epoch2_vs_tiny_taps_hh_same_slice_v1.pt`
- split: Anthropic/hh-rlhf test slice; 512-pair saved same-slice artifact, not full 8552 split
- examples scored: 512
- generation/extraction rerun: no

## Metrics

| metric | value |
| --- | ---: |
| fixed_order_acc | 0.9668 |
| fixed_order_acc_ci95 | [0.9512, 0.9805] |
| flipped_direction_acc | 0.1562 |
| strict_sign_flip_rate | 0.1895 |
| flip_correlation | -0.8989 |
| antisym_acc | 0.6367 |
| antisym_acc_ci95 | [0.5957, 0.6777] |
| bias_mean | 1.2421 |
| bias_std | 0.1981 |

## Interpretation

On the existing 512-pair HH-RLHF same-slice artifact, fixed-order accuracy is high but antisymmetrized accuracy is much lower. The 95.2% full-test number must remain fixed-order HH-RLHF pairwise accuracy; it should not be described as strict antisymmetrized accuracy.

Full-test strict audit status: missing full 8,552-pair saved normal/flipped score table or feature pack. Do not call the 95.2% number strict antisymmetrized accuracy.
