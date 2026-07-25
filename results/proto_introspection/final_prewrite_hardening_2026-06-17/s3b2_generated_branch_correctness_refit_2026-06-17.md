# S3B2 Generated-Branch Correctness Refit 2026-06-17

S3B2_REFIT_VERDICT = GENERATED_BRANCH_SELECTOR_PARTIAL
SELECTION_WALL_VERDICT = SELECTION_WALL_REMAINS_AFTER_REFIT
S3A_SELECTION_NEED_VERDICT = S3B2_SHOWS_SELECTOR_NEEDS_S3A_OR_MORE_DATA

## Inputs

- features: `artifacts/reports/probes/mpn_s3b_2026-06-17/s3b1_loop_pools.pt`
- labels: verifier/gold/exact correctness only via candidate reward field
- split: leave-one-task-out grouped by task_id; candidates from a held-out task are never in train
- generation rerun: no
- Ouro/base checkpoint training: no

## Metrics

| metric | value |
| --- | ---: |
| n_task_groups | 16 |
| oracle_present_task_groups | 8 |
| binary_auroc | 0.7515 |
| pairwise_accuracy | 0.6835 |
| pairwise_accuracy_ci95 | [0.6043, 0.7626] |
| selected_acc_all_tasks | 0.3125 |
| sel_at_oracle | 0.6250 |
| sel_at_oracle_ci95 | [0.2500, 0.8750] |
| regret | 0.3750 |
| top2_retention | 0.7500 |
| top4_retention | 0.7500 |

## Comparator Reading

Best existing corrected S3B1 real selector: mixedhead_MIX_HH_OBJECTIVE, sel@oracle 0.6667, separability 0.5687. S3B2 reaches sel@oracle 0.625 and pairwise/separability 0.6835 on the small grouped refit.

## Interpretation

The generated-branch refit improves binary separability/pairwise ranking on the small saved generated pool, but forced top-1 selection remains weak and does not cleanly beat the best existing S3B1 selector on sel@oracle. The dataset is only 16 task groups / 8 oracle-present groups, so this sharpens the selection-wall diagnosis rather than solving it.
