# S1/S3 Real Injection Orthogonality Expanded Audit

No generation was run. No checkpoint was modified.

## Exact S1/S3 Tensor Inventory

| item | status |
| --- | --- |
| exact_s1_s3_clean_root_hidden_states | not found in June-17 S1.4 summaries/logs |
| exact_s1_s3_injected_or_carried_hidden_states | not found in June-17 S1.4 summaries/logs |
| exact_s1_s3_h_injected_minus_h_clean_deltas | not found in June-17 S1.4 summaries/logs |
| s1_metadata_found | ['artifacts/reports/probes/mpn_s1_baseline_2026-06-13/s1_4_reference_loop.json', 'artifacts/reports/probes/mpn_s1_baseline_2026-06-13/s1_4a_fork_param_screen.json', 'artifacts/reports/probes/mpn_s1_baseline_2026-06-13/s1_4b_kmatched_sampling.json', 'artifacts/reports/probes/mpn_s1_baseline_2026-06-13/s1_report_2026-06-17.md', 'artifacts/logs/mpn_s1/s1_4_reference_loop_run1.log', 'artifacts/logs/mpn_s1/s1_4a_fork_param_screen.log', 'artifacts/logs/mpn_s1/s1_4b_kmatched_sampling.log'] |
| recovered_adjacent_real_delta_artifact | artifacts/reports/probes/bg_hidden_origin_diversity_v2_2026-05-18/diverse_hidden_origin_branches.pt |
| adjacent_artifact_scope | May hidden-origin hook_intervention_per_branch rows, not exact June-17 S1.4 frozen-fork/carry run. |
| expanded_hidden_origin_delta_file_found_not_primary | artifacts/reports/probes/bg_hidden_origin_taps_2026-05-18/expanded_hidden_origin_branches.pt |
| large_hidden_origin_delta_files_not_loaded | ['artifacts/reports/probes/bg_hidden_origin_quota_v4_2026-05-18/quota_hidden_origin_branches.pt', 'artifacts/reports/probes/bg_hidden_origin_branch_generator_v1_2026-05-18/branch_generator_v1_branches.pt'] |

## Real-Delta Side Audit Scope

The audited real deltas come from the May hidden-origin hook-intervention artifact, not the exact June-17 S1.4 frozen-fork/carry run. This makes the result adjacent engineering evidence, not a strict replacement for the missing S1.4 tensors.

## Metrics

| metric | value |
| --- | --- |
| nonzero real-delta rows audited | 542 |
| positive labels | 149 |
| task groups | 27 |
| real injection rank | 229 |
| real injection effective rank | 82.9919 |
| sampling rank | 144 |
| sampling effective rank | 24.4224 |
| rank match used | 144 |
| proj outcome onto real injection span full rank | 0.001151 |
| residual outside real injection span full rank | 0.998849 |
| proj pairwise outcome onto real injection span full rank | 0.002525 |
| proj outcome onto sampling span rank matched | 0.165930 |

## Random Same-Rank Control

| rank | draws | mean | ci95 | theoretical_mean |
| --- | --- | --- | --- | --- |
| 144 | 25 | 0.006074 | [0.004763, 0.006997] | 0.005859 |

## Classification Sanity

| feature set | AUROC | acc@0 | leakage passed |
| --- | --- | --- | --- |
| full_features | 0.7313 | 0.7011 | True |
| injection_span_projected_rank_matched | 0.6971 | 0.6052 | True |
| injection_residual_rank_matched | 0.7306 | 0.7011 | True |

## Previous Proxy Status

Previous proxy projection fraction was 0.004832 with residual 0.995168.

## Verdicts

| constant | value |
| --- | --- |
| REAL_INJECTION_ORTHOGONALITY_VERDICT | REAL_INJECTION_DELTAS_FOUND_AND_AUDITED |
| EXACT_S1_S3_TENSOR_STATUS | EXACT_JUNE17_S1_S3_INJECTION_DELTAS_NOT_FOUND |
| INJECTION_OUTCOME_ALIGNMENT_VERDICT | OUTCOME_DIRECTION_MOSTLY_OUTSIDE_INJECTION_SPAN |
| S3A_GEOMETRY_CLAIM | SAFE_TO_CLAIM_PROXY_CONSISTENT_WITH_MISALIGNMENT_ONLY |

## Interpretation

The exact June-17 S1.4 clean/injected/carried tensors are still absent. A recovered adjacent hidden-origin intervention artifact does contain real nonzero injection deltas and correctness labels; on that real-delta side audit, the outcome direction is overwhelmingly outside the injection span. This supports the subspace-mismatch story as a caveated appendix/future-audit point, but it should not be used as a strict explanation of the S1.4 frozen-fork null.
