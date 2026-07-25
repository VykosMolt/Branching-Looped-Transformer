# S3B2 Generated-Branch Correctness Expanded Audit

No generation was run. Labels are verifier/gold/exact correctness from the saved candidate reward field.

## Inputs

- `artifacts/reports/probes/mpn_s3b_2026-06-17/s3b1_loop_pools.pt`
- `artifacts/reports/probes/mpn_s3b_2026-06-17/s3b1_pool_texts.json`
- `artifacts/reports/probes/mpn_s3b_2026-06-17/s3b1_loop_pool_transfer.json`
- `artifacts/reports/proto_introspection/final_prewrite_hardening_2026-06-17/s3b2_generated_branch_correctness_refit_2026-06-17.json`

## Headline Metrics

| metric | value |
| --- | --- |
| prior L2 logistic AUROC | 0.7515 |
| prior L2 logistic pairwise | 0.6835 |
| prior L2 logistic sel@oracle | 0.6250 |
| expanded hidden ridge AUROC | 0.7755 |
| expanded hidden ridge pairwise | 0.7338 |
| expanded hidden ridge sel@oracle | 0.6250 |

## Split

Leave-one-task-out grouped by `task_id`; leakage check passed: True. Groups: 16; examples: 160; fold train/test examples: [150] / [10].

## Domain Breakdown

| domain | candidates | tasks | positives | class_balance | AUROC | pairwise | sel@oracle | regret |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coding | 40 | 4 | 0 | 0.0000 | NA | NA | NA | NA |
| logic | 40 | 4 | 10 | 0.2500 | 0.5267 | 0.6250 | 1.0000 | 0.0000 |
| math | 40 | 4 | 8 | 0.2000 | 0.9727 | 0.9000 | 0.5000 | 0.5000 |
| reasoning | 40 | 4 | 11 | 0.2750 | 0.5549 | 0.7143 | 0.5000 | 0.5000 |

## Baseline Comparison

ORACLE and RANDOM are controls and are not counted as best real selectors.

| selector | sel@oracle | separability/AUROC | pairwise | top2 | regret | note |
| --- | --- | --- | --- | --- | --- | --- |
| S3B0_pairwise_blockwise | 0.5000 | 0.5077 | NA | 0.7500 | 0.3333 |  |
| S3B0_listwise_blockwise | 0.4167 | 0.4998 | NA | 0.7500 | 0.4167 |  |
| CoreContent_v2_blockwise | 0.4167 | 0.4896 | NA | 0.5833 | 0.4167 |  |
| DualAnchor | 0.4167 | 0.5571 | NA | 0.5000 | 0.4167 |  |
| mixedhead_MIX_HH_OBJECTIVE | 0.6667 | 0.5687 | NA | 0.7500 | 0.2500 |  |
| MIX_OBJECTIVE_ALL_only | 0.5000 | 0.5442 | NA | 0.5000 | 0.3333 |  |
| RANDOM | 0.5833 | 0.4625 | NA | 0.5833 | 0.2500 |  |
| ORACLE | 1.0000 | 1.0000 | NA | 1.0000 | 0.0000 |  |
| S3B2_L2_LOGISTIC_PRIOR_HARDENING | 0.6250 | 0.7515 | 0.6835 | 0.7500 | 0.3750 | registered prior hardening result |
| S3B2_HIDDEN_RIDGE_EXPANDED | 0.6250 | 0.7755 | 0.7338 | 0.7500 | 0.3750 | new grouped ridge expansion scores |

## Ablations

| feature/model | AUROC | pairwise | sel@oracle | selected all | top2 |
| --- | --- | --- | --- | --- | --- |
| hidden_ridge | 0.7755 | 0.7338 | 0.6250 | 0.3125 | 0.7500 |
| length_only | 0.1094 | 0.0647 | 0.2500 | 0.1250 | 0.6250 |
| domain_only | 0.4334 | 0.2590 | 0.2500 | 0.1250 | 0.6250 |
| provenance_only | 0.0754 | 0.1583 | 0.2500 | 0.1250 | 0.3750 |
| domain_length_provenance | 0.4426 | 0.2374 | 0.2500 | 0.1250 | 0.6250 |
| hidden_plus_length | 0.7784 | 0.7266 | 0.6250 | 0.3125 | 0.7500 |
| hidden_plus_domain | 0.7755 | 0.7338 | 0.6250 | 0.3125 | 0.7500 |
| hidden_plus_domain_length_provenance | 0.7797 | 0.7338 | 0.6250 | 0.3125 | 0.7500 |
| hidden_pairwise_ranker_ridge | 0.7915 | 0.7410 | 0.6250 | 0.3125 | 0.8750 |
| tap_scores_only | NOT_RUN | Per-candidate DualAnchor/CoreContent/MIX tap scores are not persisted in s3b1_loop_pools.pt; only aggregate S3B1 comparator metrics are present. |  |  |  |
| tiny_mlp | NOT_RUN | The saved pool has only 16 task groups / 160 candidates; a tiny MLP would be high-variance and less interpretable than the grouped linear controls. |  |  |  |

## Calibration And Abstention

| bin | score_min | score_max | n | empirical_correct_rate | mean_score |
| --- | --- | --- | --- | --- | --- |
| 1 | -1.8383 | -1.1970 | 32 | 0.0312 | -1.4006 |
| 2 | -1.1970 | -0.9454 | 32 | 0.0312 | -1.0855 |
| 3 | -0.9454 | -0.4641 | 32 | 0.1562 | -0.7449 |
| 4 | -0.4641 | 0.1766 | 32 | 0.3750 | -0.1734 |
| 5 | 0.1766 | 1.3674 | 32 | 0.3125 | 0.6486 |

| margin quantile | threshold | coverage | acted | acted acc | oracle acted | sel@oracle acted |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.0024 | 1.0000 | 16 | 0.3125 | 8 | 0.6250 |
| 25 | 0.0665 | 0.7500 | 12 | 0.3333 | 6 | 0.6667 |
| 50 | 0.1193 | 0.5000 | 8 | 0.3750 | 5 | 0.6000 |
| 75 | 0.2571 | 0.2500 | 4 | 0.2500 | 3 | 0.3333 |
| 90 | 0.4603 | 0.1250 | 2 | 0.0000 | 1 | 0.0000 |

## Failure Taxonomy

Oracle-present groups: 8; selector hits on oracle-present groups: 5; selector misses despite oracle present: 3; all-candidates-wrong groups: 8.

| domain | tasks | oracle_present | hits_on_oracle | all_wrong |
| --- | --- | --- | --- | --- |
| coding | 4 | 0 | 0 | 4 |
| logic | 4 | 2 | 2 | 2 |
| math | 4 | 2 | 1 | 2 |
| reasoning | 4 | 4 | 2 | 0 |

## Verdicts

| constant | value |
| --- | --- |
| S3B2_EXPANDED_VERDICT | GENERATED_BRANCH_CORRECTNESS_SIGNAL_CONFIRMED_BUT_SELECTION_UNSOLVED |
| SELECTION_WALL_DIAGNOSIS | PARTIAL_SIGNAL_TERMINAL_SELECTION_REMAINS_BOTTLENECK |
| S3B2_PAPER_CLAIM | SAFE_TO_CLAIM_PARTIAL_GENERATED_BRANCH_CORRECTNESS_READOUT |

## Interpretation

S3B2 has a real generated-branch hidden-state correctness signal: grouped hidden features separate correct from incorrect candidates and metadata-only controls are weak. The signal does not solve terminal selection: forced top-1 remains 5/8 on oracle-present groups, high margin abstention does not reliably improve precision, and the pool has only 16 task groups.
