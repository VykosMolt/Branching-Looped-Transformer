# Hidden-Origin Branch Diversity V3

V3 was needed because v1 was data-limited and v2 produced only a weak selector: the evaluator machinery worked, but hidden-origin branch generation still produced too many downstream reward ties.

## Verdicts

- BG_HIDDEN_ORIGIN_DIVERSITY_V3_AUDIT_VERDICT = `READY`
- BG_HIDDEN_ORIGIN_TASK_SELECTION_V3_VERDICT = `READY`
- BG_HIDDEN_ORIGIN_SPLIT_GUARD_V3_VERDICT = `READY`
- BG_HIDDEN_ORIGIN_DIRECTION_BANK_V3_VERDICT = `READY`
- BG_HIDDEN_ORIGIN_DIVERSITY_ABLATION_V3_VERDICT = `DIVERSITY_IMPROVED`
- BG_HIDDEN_ORIGIN_DIVERSITY_DRIVER_V3_VERDICT = `NON_RANDOM_DIRECTIONS_HELP`
- BG_HIDDEN_ORIGIN_TAP_DATASET_V3_VERDICT = `STILL_DATA_LIMITED`
- BG_HIDDEN_ORIGIN_TAP_TRAINING_V3_VERDICT = `WEAK`
- BG_HIDDEN_ORIGIN_TAP_EVAL_V3_VERDICT = `DATA_LIMITED`
- BG_HIDDEN_ORIGIN_TAP_GEOMETRY_V3_VERDICT = `OLD_GEOMETRY_CONFIRMED`
- HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE = `v3_hidden_origin_tap`
- PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_V3 = `STILL_DATA_LIMITED`
- HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE = `v3_hidden_origin_tap`
- PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_V3 = `STILL_DATA_LIMITED`

## Task Selection

Task selection prioritized reasoning/science MCQ tasks with parse fragility, wrong-but-parseable clean answers, low-confidence clean correctness, perturbation sensitivity, prior reward variance, and old/v1/v2 disagreement.

- selection: `{'selected_count': 128, 'selected_high_medium_count': 64, 'selected_domain_counts': {'reasoning': 40, 'science': 88}, 'selected_screening_class_counts': {'baseline_correct_confident': 12, 'baseline_correct_low_confidence': 5, 'baseline_parse_fragile': 29, 'baseline_wrong_parseable': 20, 'perturbation_sensitive': 2, 'unknown': 47, 'unscreened_candidate': 13}}`

## Split And Leakage Guard

The v3 heldout set was reserved before empirical hidden-origin direction construction. Heldout task IDs are excluded from v3 empirical mean-diff/whitened directions and v3 tap training.

- split_guard: `{'counts': {'baseline_overlap_heldout': 0, 'candidate_tasks': 128, 'clean_cross_version_heldout': 26, 'heldout_candidates': 26, 'train_candidates': 83, 'val_candidates': 19}, 'baseline_overlap': [], 'clean_cross_version_heldout_task_ids': ['ARC-Challenge/12', 'ARC-Challenge/18', 'OpenBookQA/11', 'OpenBookQA/13', 'OpenBookQA/14', 'OpenBookQA/15', 'OpenBookQA/17', 'OpenBookQA/18', 'OpenBookQA/19', 'mmlu/anatomy/1', 'mmlu/anatomy/10', 'mmlu/anatomy/11', 'mmlu/anatomy/13', 'mmlu/anatomy/14', 'mmlu/anatomy/15', 'mmlu/anatomy/2', 'mmlu/anatomy/21', 'mmlu/anatomy/22', 'mmlu/high_school_biology/1', 'mmlu/high_school_biology/10', 'mmlu/high_school_biology/12', 'mmlu/high_school_biology/21', 'mmlu/high_school_biology/22', 'mmlu/high_school_chemistry/1', 'mmlu/high_school_chemistry/12', 'mmlu/high_school_chemistry/22']}`

## Direction Bank

The bank records perturbation compatibility explicitly. Concat directions are scoring/geometry-only unless a validated projection exists, and L47 remains diagnostic.

- direction_bank: `{'family_counts': {'adapter_proxy': 16, 'hidden_origin_empirical': 19, 'hidden_origin_whitened': 19, 'old_tap_aligned': 128, 'random_orthogonal': 12, 'v1_tap_aligned': 342, 'v2_tap_aligned': 234}, 'family_status': {'adapter_proxy': {'entries': 16, 'layers': [24, 36], 'perturbation_usable_entries': 16, 'recommended_alpha_bucket': 'alpha_0_01', 'status': 'ready'}, 'hidden_origin_empirical': {'entries': 19, 'layers': [24, 36, 47], 'perturbation_usable_entries': 12, 'recommended_alpha_bucket': 'alpha_0_01', 'status': 'ready'}, 'hidden_origin_whitened': {'entries': 19, 'layers': [24, 36, 47], 'perturbation_usable_entries': 12, 'recommended_alpha_bucket': 'alpha_0_01', 'status': 'ready'}, 'old_tap_aligned': {'entries': 128, 'layers': [24, 36, 47], 'perturbation_usable_entries': 128, 'recommended_alpha_bucket': 'alpha_0_01', 'status': 'ready'}, 'random_orthogonal': {'entries': 12, 'layers': [24, 36, 47], 'perturbation_usable_entries': 12, 'recommended_alpha_bucket': 'alpha_0_01', 'status': 'ready'}, 'v1_tap_aligned': {'entries': 342, 'layers': [24, 36, 47], 'perturbation_usable_entries': 216, 'recommended_alpha_bucket': 'alpha_0_01', 'status': 'ready'}, 'v2_tap_aligned': {'entries': 234, 'layers': [24, 36, 47], 'perturbation_usable_entries': 108, 'recommended_alpha_bucket': 'alpha_0_01', 'status': 'ready'}}, 'layers': ['24', '36', '47']}`

## Diversity Ablation

Primary diversity uses stable deterministic alpha `<= 0.01` L24/L36 reasoning/science same-prefix hidden-origin rows. Alpha `0.02`, sampled expected reward, and L47 are diagnostic only.

- generation: `{'stats': {'candidate_pair_stats': {'candidate_pairs': 4515, 'non_tie_pairs': 536, 'tie_pairs': 3979, 'tie_rate': 0.8812846068660022}, 'combined_behaviorally_diverse_groups': 71, 'combined_primary_stable_groups': 331, 'combined_reward_diverse_groups': 60, 'combined_tasks': 78, 'errors': 0, 'heldout_behaviorally_diverse_groups': 6, 'heldout_candidate_task_ids': 26, 'heldout_non_tie_pairs': 15, 'heldout_pair_stats': {'candidate_pairs': 1356, 'non_tie_pairs': 15, 'tie_pairs': 1341, 'tie_rate': 0.9889380530973452}, 'new_behaviorally_diverse_groups': 31, 'new_primary_stable_groups': 72, 'new_primary_stable_rows': 576, 'new_reward_diverse_groups': 29, 'new_rows': 784, 'new_stable_rows': 784, 'parse_rate_new': 0.8227040816326531, 'stable_rate_new': 1.0}, 'counts_by_primary_delta_family': {'adapter_proxy': 32, 'empirical_plus_noise': 72, 'hidden_origin_empirical': 80, 'hidden_origin_whitened': 56, 'old_tap_aligned': 64, 'paired_plus_minus': 96, 'random_orthogonal': 168, 'v1_tap_aligned': 144, 'v2_tap_aligned': 72}, 'counts_by_branch_point': {'L24': 304, 'L36': 384, 'L47': 96}, 'counts_by_alpha_bucket': {'alpha_0_005': 296, 'alpha_0_01': 280, 'alpha_0_02': 208}, 'checkpointing': {'partial_pt': 'artifacts/reports/probes/bg_hidden_origin_diversity_v3_2026-05-18/diversity_ablation_v3.partial.pt', 'progress_jsonl': 'artifacts/reports/probes/bg_hidden_origin_diversity_v3_2026-05-18/diversity_ablation_progress.jsonl', 'resumable': True, 'skip_completed_branch_groups_unless_FORCE_RERUN': True, 'state_json': 'artifacts/reports/probes/bg_hidden_origin_diversity_v3_2026-05-18/diversity_ablation_state_v3.json'}}`

## Diversity Drivers

- drivers: `{'primary_metrics': {'behaviorally_diverse_groups': 31, 'behaviorally_diverse_groups_per_100_rows': 5.381944444444445, 'candidate_pair_stats': {'candidate_pairs': 2016, 'non_tie_pairs': 366, 'tie_pairs': 1650, 'tie_rate': 0.8184523809523809}, 'groups': 72, 'non_tie_pairs_per_100_rows': 63.541666666666664, 'parse_rate': 0.8298611111111112, 'reward_diverse_groups': 29, 'rows': 576, 'stable_rate': 1.0, 'stable_rows': 576, 'task_ids_with_reward_diversity': ['OpenBookQA/14', 'mmlu/anatomy/12', 'mmlu/anatomy/7', 'mmlu/high_school_chemistry/10', 'mmlu/high_school_physics/11', 'sciq/sciq/22']}, 'questions': {'alpha02_yield': 4.8076923076923075, 'alpha_0_02_helps_safely': False, 'alpha_0_02_required': False, 'confident_class_yield': 0.0, 'k_expansion_helps': True, 'k_yields': {'4': 0.0, '6': 0.0, '8': 5.381944444444445}, 'l24_vs_l36': {'L24': {'behaviorally_diverse_groups': 13, 'behaviorally_diverse_groups_per_100_rows': 5.078125, 'behaviorally_diverse_rate': 0.40625, 'candidate_pairs': 896, 'groups': 32, 'non_tie_pairs': 122, 'non_tie_pairs_per_100_rows': 47.65625, 'parse_rate': 0.859375, 'parse_success_rows': 220, 'reward_diverse_groups': 11, 'reward_diverse_rate': 0.34375, 'reward_mean': 0.221875, 'rows': 256, 'stable_rate': 1.0, 'stable_rows': 256, 'tie_pairs': 774, 'tie_rate': 0.8638392857142857}, 'L36': {'behaviorally_diverse_groups': 18, 'behaviorally_diverse_groups_per_100_rows': 5.625, 'behaviorally_diverse_rate': 0.45, 'candidate_pairs': 1120, 'groups': 40, 'non_tie_pairs': 244, 'non_tie_pairs_per_100_rows': 76.25, 'parse_rate': 0.80625, 'parse_success_rows': 258, 'reward_diverse_groups': 18, 'reward_diverse_rate': 0.45, 'reward_mean': 0.239375, 'rows': 320, 'stable_rate': 1.0, 'stable_rows': 320, 'tie_pairs': 876, 'tie_rate': 0.7821428571428571}}, 'non_random_directions_help': True, 'non_random_yield': 9.375, 'preferred_class_yield': 9.722222222222221, 'primary_alpha_yield': 5.405405405405405, 'random_yield': 6.944444444444445, 'task_screening_continues_to_help': True}, 'recommended_branch_generation_recipe': {'behaviorally_diverse_groups': 14, 'behaviorally_diverse_groups_per_100_rows': 9.722222222222221, 'condition': 'perturbation_sensitive', 'factor': 'task_screening_class', 'groups': 18, 'non_tie_pairs_per_100_rows': 145.83333333333334, 'parse_rate': 0.625, 'reward_diverse_groups': 14, 'rows': 144, 'stable_rate': 1.0, 'tie_rate': 0.5833333333333334}}`

## Dataset V3

Primary pairwise labels omit ties rather than assigning arbitrary labels.

- dataset: `{'primary_pairs': 536, 'pairs_by_split': {'test': 15, 'train': 134, 'val': 387}, 'tasks_by_split': {'test': 1, 'train': 5, 'unused': 0, 'val': 8}, 'behaviorally_diverse_groups_by_split': {'test': 1, 'train': 21, 'unused': 0, 'val': 43}, 'primary_tie_rate': 0.8812846068660022, 'variant_meta': {'alpha_0_02_diagnostic': {'behaviorally_diverse_groups': 11, 'candidate_unordered_pairs': 799, 'omitted_missing_features': 0, 'reward_diverse_groups': 10, 'tie_pairs_omitted': 674, 'tie_rate': 0.8435544430538173, 'valid_branch_count': 230, 'valid_group_count': 29}, 'high_yield_recipe_subset': {'behaviorally_diverse_groups': 18, 'candidate_unordered_pairs': 728, 'omitted_missing_features': 0, 'reward_diverse_groups': 18, 'tie_pairs_omitted': 469, 'tie_rate': 0.6442307692307693, 'valid_branch_count': 208, 'valid_group_count': 26}, 'primary_safe_deterministic': {'behaviorally_diverse_groups': 71, 'candidate_unordered_pairs': 4515, 'omitted_missing_features': 0, 'reward_diverse_groups': 60, 'tie_pairs_omitted': 3979, 'tie_rate': 0.8812846068660022, 'valid_branch_count': 1814, 'valid_group_count': 331}, 'sampled_expected_diagnostic': {'behaviorally_diverse_groups': 8, 'candidate_unordered_pairs': 76, 'omitted_missing_features': 0, 'reward_diverse_groups': 7, 'tie_pairs_omitted': 25, 'tie_rate': 0.32894736842105265, 'valid_branch_count': 32, 'valid_group_count': 8}}, 'baseline_leakage_warning_task_ids': []}`

## Training And Evaluation

The headline v3 taps are old-style antisymmetric tiny heads trained with same-group Bradley-Terry/logsigmoid ranking, random left/right swaps, target sign flips, and flip diagnostics.

- training: `{'best_head': {'architecture': 'AntisymLinearNoNorm', 'config': '30_L4', 'dim': 2048, 'flip_diagnostics': {'antisymmetry_correlation': 0.9999999403953552, 'mean_abs_score_sum': 0.0, 'mean_score_sum': 0.0, 'n': 387, 'passes': True, 'score_mean': 4.0410432120552287e-05, 'score_std': 8.196941780624911e-05, 'strict_sign_flip_rate': 1.0}, 'head_group': 'hidden_origin_branch_taps_v3', 'metrics': {'architecture': 'AntisymLinearNoNorm', 'best_epoch': 15, 'config': '30_L4', 'dim': 2048, 'epochs_run': 23, 'gradient_clip': 1.0, 'history': [{'epoch': 1, 'train_acc': 0.6716417670249939, 'train_loss': 0.6931455117553028, 'val_acc': 0.5994831919670105, 'val_loss': 0.6931430101394653}, {'epoch': 2, 'train_acc': 0.7611939907073975, 'train_loss': 0.6931420956084977, 'val_acc': 0.6072351336479187, 'val_loss': 0.6931418776512146}, {'epoch': 3, 'train_acc': 0.8507462739944458, 'train_loss': 0.6931390780121532, 'val_acc': 0.6175710558891296, 'val_loss': 0.6931407451629639}, {'epoch': 4, 'train_acc': 0.8805969953536987, 'train_loss': 0.6931361351440202, 'val_acc': 0.6201550364494324, 'val_loss': 0.693139374256134}, {'epoch': 5, 'train_acc': 0.888059675693512, 'train_loss': 0.6931332919135023, 'val_acc': 0.6227390170097351, 'val_loss': 0.6931381225585938}, {'epoch': 6, 'train_acc': 0.888059675693512, 'train_loss': 0.6931304522414705, 'val_acc': 0.6304909586906433, 'val_loss': 0.6931371688842773}, {'epoch': 7, 'train_acc': 0.9029850363731384, 'train_loss': 0.6931275609713882, 'val_acc': 0.6459948420524597, 'val_loss': 0.6931363344192505}, {'epoch': 8, 'train_acc': 0.9179103970527649, 'train_loss': 0.6931247444295171, 'val_acc': 0.6511628031730652, 'val_loss': 0.6931352615356445}, {'epoch': 9, 'train_acc': 0.9253731369972229, 'train_loss': 0.6931219323357539, 'val_acc': 0.6589147448539734, 'val_loss': 0.6931340098381042}, {'epoch': 10, 'train_acc': 0.9402984976768494, 'train_loss': 0.6931189796817836, 'val_acc': 0.6640827059745789, 'val_loss': 0.6931330561637878}, {'epoch': 11, 'train_acc': 0.9402984976768494, 'train_loss': 0.6931162245238005, 'val_acc': 0.6666666865348816, 'val_loss': 0.6931319832801819}, {'epoch': 12, 'train_acc': 0.9328358173370361, 'train_loss': 0.6931134204366314, 'val_acc': 0.6692506670951843, 'val_loss': 0.693130612373352}, {'epoch': 13, 'train_acc': 0.9402984976768494, 'train_loss': 0.6931105389523862, 'val_acc': 0.682170569896698, 'val_loss': 0.6931295394897461}, {'epoch': 14, 'train_acc': 0.9402984976768494, 'train_loss': 0.6931076832671663, 'val_acc': 0.6770026087760925, 'val_loss': 0.6931283473968506}, {'epoch': 15, 'train_acc': 0.9552238583564758, 'train_loss': 0.6931049289988048, 'val_acc': 0.6925064921379089, 'val_loss': 0.693126916885376}, {'epoch': 16, 'train_acc': 0.9552238583564758, 'train_loss': 0.6931020555211537, 'val_acc': 0.6899225115776062, 'val_loss': 0.6931256055831909}, {'epoch': 17, 'train_acc': 0.9402984976768494, 'train_loss': 0.6930992149595004, 'val_acc': 0.6847545504570007, 'val_loss': 0.6931242346763611}, {'epoch': 18, 'train_acc': 0.9402984976768494, 'train_loss': 0.693096440229843, 'val_acc': 0.682170569896698, 'val_loss': 0.693122923374176}, {'epoch': 19, 'train_acc': 0.9402984976768494, 'train_loss': 0.693093472452306, 'val_acc': 0.682170569896698, 'val_loss': 0.6931218504905701}, {'epoch': 20, 'train_acc': 0.9402984976768494, 'train_loss': 0.693090621215194, 'val_acc': 0.682170569896698, 'val_loss': 0.6931205987930298}, {'epoch': 21, 'train_acc': 0.9402984976768494, 'train_loss': 0.6930876569961434, 'val_acc': 0.6795865893363953, 'val_loss': 0.6931191682815552}, {'epoch': 22, 'train_acc': 0.9402984976768494, 'train_loss': 0.6930847630571964, 'val_acc': 0.6770026087760925, 'val_loss': 0.693117618560791}, {'epoch': 23, 'train_acc': 0.9402984976768494, 'train_loss': 0.6930819705351076, 'val_acc': 0.6770026087760925, 'val_loss': 0.6931161880493164}], 'lr': 0.0003, 'objective': 'directional_pairwise_logsigmoid', 'score_l2': 0.0001, 'seed': 42, 'swap_protocol': '50_percent_random_left_right_with_target_sign_flip', 'train_pairs': 134, 'train_pairwise_accuracy': 0.9552238583564758, 'val_pairs': 387, 'validation_loss': 0.693126916885376, 'validation_pairwise_accuracy': 0.6925064921379089, 'variant': 'primary_safe_deterministic'}, 'variant': 'primary_safe_deterministic'}, 'trained_heads': 468, 'anti_degeneracy': {'constant_solution_rejected_by_score_std': True, 'flip_diagnostics_required': True, 'random_swap': True, 'tap_score_not_used_as_training_label': True, 'target_sign_flip': True}}`
- heldout: `{'heldout_task_ids': ['OpenBookQA/14'], 'heldout_group_count': 5, 'heldout_pair_count': 15, 'behaviorally_diverse_heldout_groups': 1, 'best_v3_head': 'primary_safe_deterministic::30_L4::AntisymLinearNoNorm::seed=42::lr=0.0003', 'previous_v1_head': 'primary::concat_24_36::AntisymLinear::seed=44::lr=0.0001', 'previous_v2_head': 'primary_safe_deterministic::concat_24_30_36::AntisymLinear::seed=44::lr=0.0001', 'old_frozen_pairwise_accuracy': 0.26666666666666666}`

## Geometry V3

- geometry: `{'max_abs_old_tap_alignment': 0.9883381724357605, 'mean_old_tap_alignment': 0.12937183947481704, 'v1_v2_v3_alignment': 0.999947726726532, 'mean_seed_config_stability': 0.47207286955763733}`

## Phase 2 Implication

Continue targeted data expansion using the v3 recipe before making selector-readiness claims.
## Hidden-origin branch split salvage and selector reevaluation (2026-05-18)

- `BG_HIDDEN_ORIGIN_SALVAGE_EVAL_VERDICT = WEAK_SELECTOR`
- `BG_HIDDEN_ORIGIN_CV_STABILITY_VERDICT = STABLE_POSITIVE`
- `BG_HIDDEN_ORIGIN_V4_QUOTA_NEED_VERDICT = V4_REQUIRED_HELDOUT_BALANCE`
- `HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE = old_frozen_bg`
- `PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_SALVAGE = WEAK`

Split salvage reused existing v3 branch data only. It reports strict/v3-clean heldout support separately from grouped-CV diagnostics and marks baseline contamination where applicable.
## Hidden-origin branch quota v4 and old-context replay (2026-05-18)

- `BG_HIDDEN_ORIGIN_QUOTA_GENERATION_V4_VERDICT = PARTIAL`
- `BG_HIDDEN_ORIGIN_SELECTOR_EVAL_V4_VERDICT = STILL_DATA_LIMITED`
- `BG_HIDDEN_ORIGIN_OLD_CONTEXT_REPLAY_V4_VERDICT = PARTIAL_MATCH`
- `BG_HIDDEN_ORIGIN_TAP_GEOMETRY_V4_VERDICT = OLD_GEOMETRY_CONFIRMED`
- `HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE_V4 = ensemble`
- `PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_V4 = STILL_DATA_LIMITED`

V4 reserves train/val/heldout task IDs before generation and keeps old-context replay diagnostic-only. Alpha 0.02, sampled labels, and L47 remain excluded from primary readiness claims.

## Hidden-origin Branch Generator v1 (2026-05-18)

Branch Generator v1 was run because v4 confirmed selector geometry but remained heldout-diversity limited. The v1 run tested early L24/L1-style hook perturbations, high-yield non-random directions, a lightweight recipe/CEM schedule, true fork/carry feasibility, and richer outcome diagnostics without training Ouro or changing production routing.

BG_BRANCH_GENERATOR_V1_AUDIT_PLAN_VERDICT = READY
BG_TRUE_FORK_CARRY_PROBE_V1_VERDICT = HOOK_FALLBACK_ONLY
BG_RICH_OUTCOME_SCHEMA_V1_VERDICT = READY
BG_BRANCH_GENERATOR_BASIS_BANK_V1_VERDICT = READY
BG_BRANCH_GENERATOR_PROPOSER_TRAINING_V1_VERDICT = RECIPE_ONLY
BG_BRANCH_GENERATOR_BLACKBOX_SEARCH_V1_VERDICT = WEAK_IMPROVEMENT
BG_BRANCH_GENERATOR_V1_GENERATION_VERDICT = HELDOUT_QUOTA_MET_ONLY
BG_BRANCH_GENERATOR_V1_DIVERSITY_VERDICT = STRONG_IMPROVEMENT
BG_BRANCH_GENERATOR_V1_BEST_METHOD = hs_inspired_controller
BG_BRANCH_GENERATOR_V1_SELECTOR_DATASET_VERDICT = HELDOUT_READY_TRAIN_WEAK
BG_BRANCH_GENERATOR_V1_SELECTOR_TRAINING_VERDICT = WEAK
BG_BRANCH_GENERATOR_V1_SELECTOR_EVAL_VERDICT = WEAK_SELECTOR
BG_BRANCH_GENERATOR_V1_OLD_CONTEXT_REPLAY_VERDICT = PARTIAL_MATCH
BG_BRANCH_GENERATOR_V1_GEOMETRY_VERDICT = OLD_GEOMETRY_CONFIRMED
HIDDEN_ORIGIN_BRANCH_GENERATOR_STATUS_V1 = WEAK_BUT_USABLE
HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE_AFTER_GENERATOR_V1 = v4_hidden_origin_tap

- quota_progress_by_split: `{'all_minimums_met': False, 'heldout': {'behaviorally_diverse_groups': 43, 'behaviorally_diverse_groups_per_100_rows': 6.554878048780488, 'candidate_pairs': 2296, 'groups': 82, 'minimum_met': True, 'non_tie_pairs': 419, 'non_tie_pairs_per_100_rows': 63.8719512195122, 'parse_rate': 0.8262195121951219, 'quota_minimums': {'behaviorally_diverse_groups': 20, 'non_tie_pairs': 120, 'task_ids': 8}, 'reward_diverse_groups': 38, 'stability_rate': 1.0, 'stable_primary_rows': 656, 'task_ids': 10, 'task_ids_with_non_tie_pair_list': ['OpenBookQA/14', 'mmlu/anatomy/12', 'mmlu/anatomy/7', 'mmlu/anatomy/8', 'mmlu/high_school_chemistry/1', 'mmlu/high_school_chemistry/10', 'mmlu/high_school_physics/11', 'sciq/sciq/22'], 'task_ids_with_non_tie_pairs': 8, 'tie_pairs': 1877, 'tie_rate': 0.8175087108013938}, 'train': {'behaviorally_diverse_groups': 35, 'behaviorally_diverse_groups_per_100_rows': 1.4583333333333333, 'candidate_pairs': 8400, 'groups': 300, 'minimum_met': False, 'non_tie_pairs': 473, 'non_tie_pairs_per_100_rows': 19.708333333333332, 'parse_rate': 0.6891666666666667, 'quota_minimums': {'behaviorally_diverse_groups': 60, 'non_tie_pairs': 250, 'task_ids': 24}, 'reward_diverse_groups': 34, 'stability_rate': 1.0, 'stable_primary_rows': 2400, 'task_ids': 40, 'task_ids_with_non_tie_pair_list': ['ARC-Challenge/1', 'ARC-Challenge/17', 'ARC-Challenge/19', 'ARC-Challenge/2', 'OpenBookQA/3'], 'task_ids_with_non_tie_pairs': 5, 'tie_pairs': 7927, 'tie_rate': 0.9436904761904762}, 'val': {'behaviorally_diverse_groups': 1, 'behaviorally_diverse_groups_per_100_rows': 0.1, 'candidate_pairs': 3452, 'groups': 127, 'minimum_met': False, 'non_tie_pairs': 7, 'non_tie_pairs_per_100_rows': 0.7, 'parse_rate': 0.874, 'quota_minimums': {'behaviorally_diverse_groups': 15, 'non_tie_pairs': 60, 'task_ids': 6}, 'reward_diverse_groups': 1, 'stability_rate': 1.0, 'stable_primary_rows': 1000, 'task_ids': 8, 'task_ids_with_non_tie_pair_list': ['mmlu/high_school_chemistry/16'], 'task_ids_with_non_tie_pairs': 1, 'tie_pairs': 3445, 'tie_rate': 0.9979721900347625}}`
- diversity_questions: `{'CEM_or_ES_improved_over_HS': False, 'K6_yield': 0.0, 'K8_remained_useful': True, 'K8_yield': 1.971057884231537, 'L24_remained_better_than_L36': True, 'L24_yield': 1.9863013698630136, 'L36_yield': 1.8485915492957747, 'alpha_0_005_remained_best': False, 'alpha_0_005_yield': 1.8851508120649652, 'alpha_0_01_yield': 2.3026315789473686, 'beat_static_v4_recipe': True, 'cem_yield': 1.957070707070707, 'hs_yield': 2.217741935483871, 'learned_proposer_helped': True, 'non_random_directions_remained_useful': True, 'non_random_yield': 3.75, 'random_yield': 2.1169354838709675, 'static_yield': 1.8333333333333333, 'structured_low_rank_coefficients_helped': False, 'true_behavioral_diversity_not_instability': True, 'true_fork_carry_changed_persistence': False}`
- recommended_next: `Either run a small selection-only prototype with caveat or run targeted generator v1.1 if one recipe clearly remains.`

Selector readiness, if claimed, uses only primary-safe deterministic alpha <= 0.01 heldout rows. Diagnostic alpha 0.02, sampled labels, L47 branches, old-context replay, and auxiliary diagnostics are not readiness support.


## Universal branch-content taps v1 (2026-05-18)

Universal Branch-Content Taps v1 tested whether one tiny hidden-state pairwise evaluator can cover both old content/candidate selection and same-prefix hidden-origin branch survival. It trained only new standalone tap heads and did not alter Ouro, existing BG taps, registries, wrapper/local-agent routing, or production behavior.

BG_UNIVERSAL_TAP_INVENTORY_VERDICT = READY
BG_UNIVERSAL_OLD_CONTENT_DATASET_VERDICT = READY
BG_UNIVERSAL_HIDDEN_BRANCH_DATASET_VERDICT = READY
BG_UNIVERSAL_BRIDGE_DATASET_VERDICT = READY
BG_UNIVERSAL_DATA_EXPANSION_VERDICT = SKIPPED
BG_UNIVERSAL_TAP_DATASET_VERDICT = READY
BG_UNIVERSAL_TAP_TRAINING_VERDICT = READY
BG_UNIVERSAL_OLD_CONTEXT_EVAL_VERDICT = MATCHES_OR_BEATS_OLD_TAPS
BG_UNIVERSAL_HIDDEN_BRANCH_EVAL_VERDICT = SMALL_DEGRADATION
BG_UNIVERSAL_BRIDGE_EVAL_VERDICT = NO_BRIDGE_SIGNAL
BG_UNIVERSAL_LAYERWISE_PRUNING_VERDICT = TOPK_SURVIVAL_ONLY
BG_UNIVERSAL_DOMAIN_GENERALIZATION_VERDICT = REASONING_SCIENCE_ONLY
BG_UNIVERSAL_TAP_GEOMETRY_VERDICT = OLD_GEOMETRY_CONFIRMED
UNIVERSAL_BRANCH_CONTENT_TAP_STATUS = FUSION_NEEDED

- old_content_counts: `{'feature_config_counts': {'24_L4': 462, '24_mean': 462, '30_L4': 0, '36_L4': 462, '36_mean': 462, '42_L4': 0, '47_L4': 462, '47_mean': 462, 'concat_24_30_36': 0, 'concat_24_36': 462, 'concat_24_36_47': 462, 'concat_36_42_47': 0, 'concat_36_47': 462}, 'pairs': 462, 'pairs_by_domain': {'math_simple_arithmetic': 143, 'reasoning': 183, 'science': 136}, 'pairs_by_split': {'heldout': 73, 'train': 295, 'val': 94}, 'pairs_by_type': {'old_content': 462}, 'tasks_by_split': {'heldout': ['ARC-Challenge/0', 'ARC-Challenge/19', 'gsm8k/1', 'gsm8k/12', 'gsm8k/5', 'mmlu/high_school_biology/10', 'mmlu/high_school_biology/12', 'mmlu/high_school_biology/17', 'mmlu/high_school_biology/9'], 'train': ['ARC-Challenge/1', 'ARC-Challenge/11', 'ARC-Challenge/12', 'ARC-Challenge/13', 'ARC-Challenge/14', 'ARC-Challenge/15', 'ARC-Challenge/16', 'ARC-Challenge/17', 'ARC-Challenge/18', 'ARC-Challenge/6', 'ARC-Challenge/8', 'gsm8k/0', 'gsm8k/10', 'gsm8k/11', 'gsm8k/13', 'gsm8k/14', 'gsm8k/16', 'gsm8k/17', 'gsm8k/19', 'gsm8k/2', 'gsm8k/3', 'gsm8k/7', 'gsm8k/8', 'mmlu/high_school_biology/1', 'mmlu/high_school_biology/11', 'mmlu/high_school_biology/15', 'mmlu/high_school_biology/19', 'mmlu/high_school_biology/2', 'mmlu/high_school_biology/3', 'mmlu/high_school_biology/4', 'mmlu/high_school_biology/5', 'mmlu/high_school_biology/7', 'mmlu/high_school_biology/8'], 'val': ['ARC-Challenge/10', 'ARC-Challenge/2', 'ARC-Challenge/3', 'ARC-Challenge/4', 'ARC-Challenge/5', 'ARC-Challenge/7', 'ARC-Challenge/9', 'gsm8k/15', 'gsm8k/9', 'mmlu/high_school_biology/14', 'mmlu/high_school_biology/18']}}`
- hidden_branch_counts: `{'feature_config_counts': {'24_L4': 1753, '24_mean': 1753, '30_L4': 1753, '36_L4': 1753, '36_mean': 1753, '42_L4': 1753, '47_L4': 1753, '47_mean': 1753, 'concat_24_30_36': 1753, 'concat_24_36': 1753, 'concat_24_36_47': 1753, 'concat_36_42_47': 1753, 'concat_36_47': 1753}, 'pairs': 1753, 'pairs_by_domain': {'reasoning': 1074, 'science': 679}, 'pairs_by_split': {'heldout': 483, 'train': 1090, 'val': 180}, 'pairs_by_type': {'hidden_branch': 1753}, 'tasks_by_split': {'heldout': ['OpenBookQA/14', 'OpenBookQA/18', 'mmlu/anatomy/12', 'mmlu/anatomy/7', 'mmlu/anatomy/8', 'mmlu/high_school_chemistry/1', 'mmlu/high_school_chemistry/10', 'mmlu/high_school_physics/11', 'sciq/sciq/22'], 'train': ['ARC-Challenge/1', 'ARC-Challenge/17', 'ARC-Challenge/19', 'ARC-Challenge/2', 'OpenBookQA/3', 'mmlu/anatomy/12', 'mmlu/anatomy/8', 'mmlu/high_school_chemistry/10', 'mmlu/high_school_physics/11', 'sciq/sciq/22'], 'val': ['ARC-Challenge/17', 'mmlu/high_school_chemistry/16']}}`
- bridge_counts: `{'feature_config_counts': {'24_L4': 2142, '24_mean': 2142, '30_L4': 2142, '36_L4': 2142, '36_mean': 2142, '42_L4': 2142, '47_L4': 2142, '47_mean': 2142, 'concat_24_30_36': 2142, 'concat_24_36': 2142, 'concat_24_36_47': 2142, 'concat_36_42_47': 2142, 'concat_36_47': 2142}, 'pairs': 2142, 'pairs_by_domain': {'reasoning': 1316, 'science': 826}, 'pairs_by_split': {'heldout': 580, 'train': 1342, 'val': 220}, 'pairs_by_type': {'bridge': 2142}, 'tasks_by_split': {'heldout': ['OpenBookQA/14', 'OpenBookQA/18', 'mmlu/anatomy/12', 'mmlu/anatomy/7', 'mmlu/anatomy/8', 'mmlu/high_school_chemistry/1', 'mmlu/high_school_chemistry/10', 'mmlu/high_school_physics/11', 'sciq/sciq/22'], 'train': ['ARC-Challenge/1', 'ARC-Challenge/17', 'ARC-Challenge/19', 'ARC-Challenge/2', 'OpenBookQA/3', 'mmlu/anatomy/12', 'mmlu/anatomy/8', 'mmlu/high_school_chemistry/10', 'mmlu/high_school_physics/11', 'sciq/sciq/22'], 'val': ['ARC-Challenge/17', 'mmlu/high_school_chemistry/16']}}`
- recommendation: `Build an explicit composite selector rather than forcing a single universal head.`

Readiness requires old-context, hidden-branch, and bridge support. Cached coding features were inspected but had no non-tie within-task labels, so coding remains coverage-limited.

