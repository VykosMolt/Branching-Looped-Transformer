# S1/S3 Exact Injection Orthogonality Rank/Null Audit

## Scope

Bundle path: `artifacts/reports/proto_introspection/s1_s3_exact_injection_orthogonality_2026-06-17/s1_s3_exact_injection_delta_bundle_2026-06-17.pt`. This audit loaded the saved tensor bundle only; it did not rerun generation, train anything, or modify checkpoints.

## Bundle Inspection

| key | type | shape/len | dtype | keys |
| --- | --- | --- | --- | --- |
| branch_features | tensor | [344, 3, 4, 2048] | float16 |  |
| branch_rows | list | 344 |  | ['alpha', 'base_correct', 'branch_id', 'candidate_group_id', 'cc_score_for_prune', 'correct', 'da_score_for_prune', 'delta_norm', 'direction_index', 'domain', 'is_terminal_locus', 'kept_after_prune', 'label_type', 'layer', 'lineage_depth', 'locus', 'locus_index', 'loop', 'loop_1indexed', 'max_new_tokens', 'parent_branch_id', 'parsed_answer', 'perturb_rms', 'reward', 'row_index', 'score_group_id', 'selected_terminal', 'task_id', 'terminal_survivor', 'text', 'text_snippet', 'token_range', 'token_range_name'] |
| created_at | str | None |  |  |
| delta_locus | tensor | [344, 2048] | float16 |  |
| elapsed_seconds | float | None |  |  |
| h_clean_locus | tensor | [344, 2048] | float16 |  |
| h_injected_locus | tensor | [344, 2048] | float16 |  |
| model_info | dict | 17 |  | ['attn_implementation', 'device', 'dtype', 'early_exit_threshold', 'expected_cache_slots', 'head_dim', 'hidden_size', 'load_seconds', 'model_class', 'model_path', 'num_attention_heads', 'num_hidden_layers', 'num_key_value_heads', 'torch_version', 'total_ut_steps', 'transformers_version', 'vocab_size'] |
| protocol | dict | 18 |  | ['BUDGET', 'K', 'N_PER', 'alpha', 'correctness_labels', 'delta_status', 'domains', 'historical_tensors_found', 'loci', 'mnt_inter', 'mnt_term', 'protocol_recovery', 'prune_mode', 's1_reference_paths', 'sampling_control', 'score_input_mode', 'tasks', 'token_range'] |
| sample_features | tensor | [48, 3, 4, 2048] | float16 |  |
| sample_rows | list | 48 |  | ['branch_id', 'capture_seed', 'correct', 'domain', 'max_new_tokens', 'parsed_answer', 'reward', 'sample_index', 'task_id', 'temperature', 'text', 'text_snippet', 'top_p'] |
| task_summaries | list | 4 |  | ['base_correct', 'base_parsed_answer', 'base_text', 'domain', 'elapsed_seconds', 'n_generated_branches', 'n_terminal_correct', 'n_terminal_survivors', 'oracle_over_terminal_survivors', 'prompt', 'prompt_tokens', 'selected_branch_id', 'selected_correct', 'task_id', 'trace'] |
| tensor_shapes | dict | 7 |  | ['branch_features', 'delta_locus', 'feature_basis', 'h_clean_locus', 'h_injected_locus', 'locus_delta_basis', 'sample_features'] |

Injection/carry deltas are `delta_locus = h_injected_locus - h_clean_locus`, embedded into the flattened `[3,4,2048]` feature basis at the saved row layer/loop slot. No explicit outcome direction tensor is saved; it is recomputed from `branch_features` and `branch_rows[].correct`.

## Main Injection Span Geometry

| metric | value |
| --- | --- |
| ambient dimension D | 24576 |
| raw positive-singular-value rank | 344 |
| torch default matrix_rank | 144 |
| main audit rank, rel 1e-6 | 344 |
| rank abs > 1e-3 | 344 |
| rank abs > 1e-4 | 344 |
| rank abs > 1e-5 | 344 |
| rank rel > 1e-3 max | 246 |
| entropy effective rank | 6.427472 |
| participation ratio | 5.062494 |
| PCs 50/80/90/95/99 | {'pcs_50pct': 3, 'pcs_80pct': 4, 'pcs_90pct': 5, 'pcs_95pct': 8, 'pcs_99pct': 13} |

The main rank/truncation matches the previous audit: uncentered SVD with rank `344` from `s > max(s) * 1e-6`. The centered secondary check gives rank `343` and nearly the same projection.

## Chance Baselines

| rank | k | k/D expected | observed projection | obs/expected | relation |
| --- | --- | --- | --- | --- | --- |
| raw_positive_singular_values | 344 | 0.013997 | 0.018296 | 1.307118 | above chance |
| torch_default_matrix_rank | 144 | 0.005859 | 0.013520 | 2.307440 | above chance |
| legacy_main_audit_rel_1e-6 | 344 | 0.013997 | 0.018296 | 1.307118 | above chance |
| absolute_gt_1e-3 | 344 | 0.013997 | 0.018296 | 1.307118 | above chance |
| absolute_gt_1e-4 | 344 | 0.013997 | 0.018296 | 1.307118 | above chance |
| absolute_gt_1e-5 | 344 | 0.013997 | 0.018296 | 1.307118 | above chance |
| relative_gt_1e-3_of_max | 246 | 0.010010 | 0.018150 | 1.813232 | above chance |

## Empirical Random-Direction Null

| metric | value |
| --- | --- |
| draws | 1000000 |
| null mean | 0.013997 |
| null std | 0.001061 |
| q01/q05/q50/q95/q99 | {'q01': 0.011645217520017175, 'q05': 0.012297469964086889, 'q50': 0.013970607972466622, 'q95': 0.015786847752825243, 'q99': 0.01658235155058102} |
| observed projection | 0.018296 |
| observed percentile | 99.990700 |
| p_left null <= observed | 0.999907 |
| p_right null >= observed | 0.000093 |

## Shuffled-Label Null

| null | draws | mean | std | quantiles | percentile | p_left | p_right |
| --- | --- | --- | --- | --- | --- | --- | --- |
| global_count_preserving | 10000 | 0.022990 | 0.002619 | {'q01': 0.016986495549915385, 'q05': 0.018708181188270168, 'q50': 0.02304504449040062, 'q95': 0.02741061694481296, 'q99': 0.028650791121583453} | 3.440000 | 0.034400 | 0.965600 |
| domain_stratified_count_preserving | 10000 | 0.022702 | 0.000782 | {'q01': 0.020509120573917707, 'q05': 0.021437279165263404, 'q50': 0.022438862105953267, 'q95': 0.02345627903567977, 'q99': 0.02345910449640722} | 0.000000 | 0.000000 | 1.000000 |

## Outcome-Subspace Extension

Candidate-group mixed-label directions: {'mixed': 0, 'all_positive': 4, 'all_negative': 168, 'total': 172}. Score-group mixed-label directions: {'mixed': 0, 'all_positive': 1, 'all_negative': 47, 'total': 48}. No classifier was trained.

| direction | projection | k/D expected | obs/expected | relation |
| --- | --- | --- | --- | --- |
| global_correct_minus_incorrect | 0.018296 | 0.013997 | 1.307118 | above chance |
| domain:math_correct_minus_incorrect | 0.028020 | 0.013997 | 2.001836 | above chance |

Outcome subspace: `{'status': 'computed', 'outcome_subspace_rank': 2, 'construction': 'global correct-minus-incorrect plus any mixed-label domain correct-minus-incorrect directions; no classifier training', 'principal_cosines': [0.17280669510364532, 0.1352633684873581], 'principal_angles_degrees': [80.04895782470703, 82.22615051269531], 'mean_projected_energy_per_outcome_dimension': 0.02407916635274887, 'expected_random_projection_k_over_D': 0.013997395833333334, 'observed_over_expected': 1.7202604426894077, 'chance_relation': 'above chance'}`.

## Geometry Boundary

Same saved [3,4,2048] branch rollout feature basis as the prior S1/S3 projection. This bundle does not contain separate historical pre-answer readout tensors, so claims tying this exact geometry to other readout artifacts remain caveated unless those artifacts used the same extractor.

## Interpretation

The absolute residual is high, but after accounting for the 344-dimensional injection span inside D=24576, the observed projection is above the k/D random-subspace baseline and in the right tail of the random-direction null. The prior orthogonality claim should therefore not be used as load-bearing evidence for subspace misalignment.

## Verdict Constants

| constant | value |
| --- | --- |
| INJECTION_SPAN_RANK_NULL_VERDICT | OBSERVED_PROJECTION_ABOVE_RANDOM_SUBSPACE_CHANCE |
| ORTHOGONALITY_LOAD_BEARING_STATUS | DO_NOT_USE_AS_ORTHOGONALITY_EVIDENCE |
| READOUT_CONTROL_BOUNDARY_UPDATE | SUBSPACE_MISALIGNMENT_NOT_SUPPORTED |
