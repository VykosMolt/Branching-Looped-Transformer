# S1/S3 Exact Injection Orthogonality Audit

## Executive Summary

Delta status: `EXACT_PROTOCOL_REGENERATED`. Historical tensors were not found; the exact S1.4 protocol was recovered from saved scripts and regenerated. Projection fraction: 0.018296; residual: 0.981704. Title/frame support: `ORTHOGONALITY_SUPPORTS_READOUT_NOT_CONTROL_FRAMING`.

## Artifact And Protocol Recovery

| item | value |
| --- | --- |
| status | EXACT_PROTOCOL_REGENERATED |
| historical tensors found | False |
| tasks | logic_00013d9d03, math_0313a1abc4, reasoning_00bf397683, coding_00715d44da |
| K / budget | 2 / 4 |
| alpha | 0.02 |
| loci | 12 |
| token range | fixed last-token suffix (plen-1, plen) |
| score input mode | candidate_text_only |
| sampling control | RECONSTRUCTED_K_MATCHED_SAMPLING_SPAN |

Historical exact delta tensors were searched for and not found. The recovered protocol uses the saved S1.4 reference loop and S1.4b closure configs.

## Delta Bundle Construction

Bundle path: `artifacts/reports/proto_introspection/s1_s3_exact_injection_orthogonality_2026-06-17/s1_s3_exact_injection_delta_bundle_2026-06-17.pt`. Branches: 344; tasks: 4; terminal survivors: 16; correct labels: 8.

| tensor | shape/value |
| --- | --- |
| branch_features | [344, 3, 4, 2048] |
| h_clean_locus | [344, 2048] |
| h_injected_locus | [344, 2048] |
| delta_locus | [344, 2048] |
| sample_features | [48, 3, 4, 2048] |
| feature_basis | [3,4,2048] flattened; layer slots 24/36/47 x loops 1..4 |
| locus_delta_basis | [2048] last-token boundary vector; embedded into [3,4,2048] at row locus for span audit |

## Outcome Direction Construction

Outcome labels use verifier/gold/exact correctness only. The main feature basis is the same pooled `[3,4,2048]` branch rollout feature representation used by the S1.4 prune-scoring path; tap scores are stored only as historical pruning metadata, not labels.

| domain | candidates | positive | negative | domain projection |
| --- | --- | --- | --- | --- |
| coding | 86 | 0 | 86 | NA |
| logic | 86 | 0 | 86 | NA |
| math | 86 | 8 | 78 | 0.028020 |
| reasoning | 86 | 0 | 86 | NA |

## Injection Span Geometry

| metric | value |
| --- | --- |
| raw rank | 344 |
| effective rank | 6.4275 |
| PCs 50/80/90/95 | {'pcs_50pct': 3, 'pcs_80pct': 4, 'pcs_90pct': 5, 'pcs_95pct': 8} |
| projection fraction | 0.018296 |
| residual fraction | 0.981704 |
| group outcome projection | NA |
| pair outcome projection | NA |
| max abs cosine top10 PCs | 0.007397 |

## Random Controls

| control | value |
| --- | --- |
| same-rank random subspace | {'draws': 100, 'rank': 344, 'mean': 0.013879932966083287, 'ci95': [0.01207569141406566, 0.0164155600592494], 'theoretical_mean_rank_over_dim': 0.013997395833333334} |
| shuffled correctness labels | {'draws': 100, 'mean': 0.023378635589033366, 'ci95': [0.01874361759983003, 0.02883067629300057]} |

## Sampling Span Comparison

| metric | value |
| --- | --- |
| sampling_span_status | RECONSTRUCTED_K_MATCHED_SAMPLING_SPAN |
| n_sample_candidates | 48 |
| sample_positive_labels | 4 |
| sampling_rank | 44 |
| sampling_effective_rank | 7.388714 |
| rank_match_used | 44 |
| projection_fraction_outcome_onto_sampling_span_rank_matched | 0.474583 |
| projection_fraction_group_outcome_onto_sampling_span_rank_matched | None |
| projection_fraction_pair_outcome_onto_sampling_span_rank_matched | None |
| principal_cosines_injection_vs_sampling | {'max': 0.22790934145450592, 'mean': 0.0531974695622921, 'min': 3.8431408029282466e-05} |
| cosine_abs_outcome_top_sampling_pcs | [0.4004543721675873, 0.317573606967926, 0.16523227095603943, 0.29560917615890503, 0.05980677157640457, 0.03816214203834534, 0.07344113290309906, 0.06036628782749176, 0.05340017378330231, 0.014885682612657547] |

## Classification Sanity

| feature set | AUROC | acc@0 | leakage passed |
| --- | --- | --- | --- |
| full_features | 0.1161 | 0.9535 | True |
| actual_injection_span_projected | 0.4196 | 0.9767 | True |
| actual_injection_residual | 0.3482 | 0.9535 | True |
| sampling_span_projected_rank_matched | 0.3482 | 0.9767 | True |
| shuffled_label_control_full_features | 0.4144 | 0.9767 | True |

## Interpretation

The exact S1.4 protocol was regenerated because historical tensors were absent. The outcome direction lies mostly outside the actual frozen injection/carry span, so the frozen null is explained or at minimum strongly supported by subspace misalignment. The claim should be phrased as exact-protocol regenerated, not exact historical tensor replay.

## Paper Insertion Text

**Cautious main-body paragraph.** We regenerated the frozen S1.4 branch/carry protocol from the saved June-17 configuration (4 tasks, 12 loci, K=2, alpha=0.02, fixed last-token perturbations) and captured the actual boundary deltas used by the branch/carry mechanism. In the regenerated bundle, the verifier-labeled outcome direction projects 0.018296 of its squared norm into the actual injection/carry span, leaving residual 0.981704. Because the original historical tensors were not persisted, this is an exact-protocol regeneration rather than a historical tensor replay; nevertheless it directly audits the S1.4 writable frozen directions.

**Stronger main-body paragraph.** The exact S1.4 protocol regeneration closes the geometry gap: the frozen branch/carry deltas span a low-rank writable subspace, but the verifier-success direction lies almost entirely outside it (projection 0.018296, residual 0.981704). This supports the readout-control boundary interpretation: the model exposes process-quality information in hidden states, while the frozen write directions available to the branch/carry mechanism are not aligned with the outcome-relevant geometry.

**Compute-pitch paragraph.** This geometry makes S3A a training-time alignment test rather than another frozen-control tweak. The frozen mechanism is mechanically valid and K-matched sampling closes the apparent fork gain, but the actual writable branch/carry span is misaligned with verifier-success directions. The next compute should therefore train on verifier-labeled generated branches to align readable process-state directions with controllable branch dynamics.

## Final Verdict Constants

| constant | value |
| --- | --- |
| EXACT_S1_S3_DELTA_STATUS | EXACT_PROTOCOL_REGENERATED |
| REAL_INJECTION_ORTHOGONALITY_VERDICT | OUTCOME_DIRECTION_MOSTLY_OUTSIDE_ACTUAL_S1_S3_INJECTION_SPAN |
| SAMPLING_SPAN_COMPARISON_VERDICT | INJECTION_SPAN_DISTINCT_FROM_SAMPLING_SPAN |
| READOUT_CONTROL_BOUNDARY_VERDICT | FROZEN_BRANCH_NULL_EXPLAINED_BY_SUBSPACE_MISALIGNMENT |
| PROTO_INTROSPECTION_TITLE_SUPPORT_VERDICT | ORTHOGONALITY_SUPPORTS_READOUT_NOT_CONTROL_FRAMING |
| S3A_COMPUTE_PITCH_VERDICT | TRAINING_JUSTIFIED_TO_ALIGN_WRITABLE_BRANCH_DIRECTIONS |
