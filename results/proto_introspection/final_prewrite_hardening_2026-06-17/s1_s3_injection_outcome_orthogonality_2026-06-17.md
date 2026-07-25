# S1/S3 Injection-Outcome Orthogonality Audit 2026-06-17

INJECTION_OUTCOME_ALIGNMENT_VERDICT = UNDERPOWERED_OR_ARTIFACTS_MISSING
SAMPLING_SPAN_COMPARISON_VERDICT = UNDERPOWERED_OR_ARTIFACTS_MISSING
S3A_COMPUTE_JUSTIFICATION_VERDICT = TRAINING_JUSTIFIED_BUT_MECHANISM_AMBIGUOUS
READOUT_CONTROL_BOUNDARY_UPDATE = INCONCLUSIVE

## Inventory

| required artifact | status |
| --- | --- |
| clean_root_hidden_states | not found for S1 June-17 runs |
| injected_or_carried_hidden_states | not found for S1 June-17 runs |
| actual_injection_deltas | not persisted; curated injection basis bank exists |
| generated_branch_features | artifacts/reports/probes/mpn_s3b_2026-06-17/s3b1_loop_pools.pt |
| verifier_labels | artifacts/reports/probes/mpn_s3b_2026-06-17/s3b1_loop_pools.pt |
| task_ids_and_domains | artifacts/reports/probes/mpn_s3b_2026-06-17/s3b1_pool_texts.json |

## Proxy Metrics

Actual S1 injection deltas were not persisted. The metrics below project the generated-branch outcome direction onto the saved curated S1 injection-basis proxy.

| metric | value |
| --- | ---: |
| projection_fraction_outcome_onto_injection_proxy | 0.004832 |
| residual_fraction_outcome_outside_injection_proxy | 0.995168 |
| projection_fraction_pairwise_outcome_onto_injection_proxy | 0.001664 |
| projection_fraction_outcome_onto_sampling_span_rank_matched | 0.280282 |
| projection_fraction_pairwise_outcome_onto_sampling_span_rank_matched | 0.962727 |
| random_same_rank_projection_mean | 0.001927 |
| principal_cosine_max_injection_proxy_vs_sampling | 0.223188 |

## Paper Insertion

A limited no-generation audit found that the generated-branch correctness direction is almost entirely outside the saved curated S1 injection-basis proxy (projection fraction 0.0048; residual 0.9952), while rank-matched within-task sampling variation captures substantially more of the outcome direction. Because the actual S1 clean/injected/carried tensors and h_injected - h_clean deltas were not persisted, this is a proxy result rather than a full orthogonality proof. The safe conclusion is that the frozen null remains consistent with subspace mismatch, but the mechanism is not fully measured.

## S3A Compute Pitch

The saved artifacts support the S3A compute pitch as a training-time alignment problem rather than a frozen-control tweak: S1 validates the mechanics and closes the K-matched frozen-fork control, S3B shows generated-branch correctness is not solved by transfer, and the proxy subspace audit suggests the correctness direction is not available through the frozen injection basis. A powered S3A run should therefore learn the write/read geometry jointly instead of relying on static frozen perturbations.
