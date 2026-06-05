# Hidden-Origin Branch Diversity V2

V2 was run because the first hidden-origin tap experiment trained technically but remained data-limited: the prior expansion had a very high tie rate and too few task-disjoint behaviorally diverse heldout groups.

## Verdicts

- BG_HIDDEN_ORIGIN_DIVERSITY_AUDIT_VERDICT = `READY`
- BG_HIDDEN_ORIGIN_TASK_SCREENING_VERDICT = `READY`
- BG_HIDDEN_ORIGIN_DIRECTION_BANK_VERDICT = `READY`
- BG_HIDDEN_ORIGIN_DIVERSITY_V2_GENERATION_VERDICT = `READY`
- BG_HIDDEN_ORIGIN_TAP_DATASET_V2_VERDICT = `SMALL_BUT_USABLE`
- BG_HIDDEN_ORIGIN_TAP_TRAINING_V2_VERDICT = `READY`
- BG_HIDDEN_ORIGIN_TAP_EVAL_V2_VERDICT = `WEAK_SELECTOR`
- BG_HIDDEN_ORIGIN_DIVERSITY_SOURCE_VERDICT = `TASK_SCREENING_HELPS`
- BG_HIDDEN_ORIGIN_LAYER_CONFIG_V2_VERDICT = `CONCAT_REQUIRED`
- BG_HIDDEN_ORIGIN_TAP_GEOMETRY_V2_VERDICT = `OLD_GEOMETRY_CONFIRMED`
- PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_V2 = `WEAK`

## Task Selection

Task screening favored clean-wrong, parse-fragile, low-confidence, and perturbation-sensitive reasoning/science MCQ tasks. Confident clean-correct tasks were deprioritized.

- selected_count: `96`
- selected_preferred_count: `56`
- selected_class_counts: `{'baseline_correct_confident': 40, 'baseline_correct_low_confidence': 5, 'baseline_parse_fragile': 29, 'baseline_wrong_parseable': 20, 'perturbation_sensitive': 2}`

## Direction Bank And Generation

The v2 generator used clean, random paired, orthogonal random, old-tap-aligned, hidden-origin empirical, and available proxy directions as perturbation candidates. Alpha `0.02` is reported as a separate diagnostic bucket and is not mixed into the primary safe-alpha headline.

- direction_families: `{'adapter_proxy': 8, 'hidden_origin_empirical': 6, 'hidden_origin_whitened': 6, 'old_tap_aligned': 128, 'v1_hidden_origin_tap': 108}`
- generation_stats: `{'candidate_heldout_task_ids': 13, 'candidate_pair_stats': {'candidate_pairs': 2499, 'non_tie_pairs': 170, 'tie_pairs': 2329, 'tie_rate': 0.9319727891156463}, 'combined_behaviorally_diverse_groups': 40, 'combined_primary_stable_groups': 259, 'combined_reward_diverse_groups': 31, 'combined_tasks': 78, 'errors': 0, 'new_behaviorally_diverse_groups': 20, 'new_primary_stable_groups': 107, 'new_primary_stable_rows': 630, 'new_rows': 652, 'new_stable_rows': 652, 'stable_rate_new': 1.0}`

## Dataset V2

Primary labels are deterministic downstream rewards within the same branch group, using only stable rows with `alpha <= 0.01`. Reward ties are omitted, not assigned arbitrary labels. Alpha `0.02` and sampled expected reward are diagnostic variants.

- primary_pairs: `170`
- pairs_by_split: `{'test': 78, 'train': 76, 'val': 16}`
- behaviorally_diverse_groups_by_split: `{'test': 11, 'train': 16, 'unused': 0, 'val': 7}`
- primary_tie_rate: `0.932`

## Training And Evaluation

The headline heads remain exact antisymmetric `AntisymLinear` and `AntisymLinearNoNorm` trained with pairwise logsigmoid ranking, random left/right swaps, target sign flips, and flip diagnostics.

- best_v2_head: `{'architecture': 'AntisymLinear', 'config': 'concat_24_30_36', 'dim': 6144, 'flip_diagnostics': {'antisymmetry_correlation': 0.9999999403953552, 'mean_abs_score_sum': 0.0, 'mean_score_sum': 0.0, 'n': 16, 'passes': True, 'score_mean': -0.001002763630822301, 'score_std': 0.007183428388088942, 'strict_sign_flip_rate': 1.0}, 'head_group': 'hidden_origin_branch_taps_v2', 'metrics': {'architecture': 'AntisymLinear', 'best_epoch': 1, 'config': 'concat_24_30_36', 'dim': 6144, 'epochs_run': 9, 'gradient_clip': 1.0, 'history': [{'epoch': 1, 'train_acc': 0.6710526347160339, 'train_loss': 0.6920909379657946, 'val_acc': 0.6875, 'val_loss': 0.6936551332473755}, {'epoch': 2, 'train_acc': 0.6973684430122375, 'train_loss': 0.6909099911388598, 'val_acc': 0.625, 'val_loss': 0.6943330764770508}, {'epoch': 3, 'train_acc': 0.7763158082962036, 'train_loss': 0.6898334465528789, 'val_acc': 0.375, 'val_loss': 0.6949564814567566}, {'epoch': 4, 'train_acc': 0.8157894611358643, 'train_loss': 0.6887087508251792, 'val_acc': 0.25, 'val_loss': 0.6955516934394836}, {'epoch': 5, 'train_acc': 0.8815789818763733, 'train_loss': 0.6876809847982306, 'val_acc': 0.25, 'val_loss': 0.6961609125137329}, {'epoch': 6, 'train_acc': 0.8947368264198303, 'train_loss': 0.6865983040709245, 'val_acc': 0.25, 'val_loss': 0.6967703700065613}, {'epoch': 7, 'train_acc': 0.9210526347160339, 'train_loss': 0.6855737880656594, 'val_acc': 0.125, 'val_loss': 0.6973947882652283}, {'epoch': 8, 'train_acc': 0.9210526347160339, 'train_loss': 0.684539280439678, 'val_acc': 0.0625, 'val_loss': 0.6980017423629761}, {'epoch': 9, 'train_acc': 0.9210526347160339, 'train_loss': 0.6835113857921801, 'val_acc': 0.0625, 'val_loss': 0.698626697063446}], 'lr': 0.0001, 'objective': 'directional_pairwise_logsigmoid', 'score_l2': 0.0001, 'seed': 44, 'swap_protocol': '50_percent_random_left_right_with_target_sign_flip', 'train_pairs': 76, 'train_pairwise_accuracy': 0.6710526347160339, 'val_pairs': 16, 'validation_loss': 0.6936551332473755, 'validation_pairwise_accuracy': 0.6875, 'variant': 'primary_safe_deterministic'}, 'variant': 'primary_safe_deterministic'}`
- heldout_summary: `{'heldout_task_ids': ['ARC-Challenge/2', 'OpenBookQA/3', 'mmlu/anatomy/12', 'sciq/sciq/22'], 'heldout_group_count': 14, 'heldout_pair_count': 78, 'behaviorally_diverse_heldout_groups': 11, 'best_head': 'concat_24_30_36::AntisymLinear::seed=44::lr=0.0001', 'previous_v1_head': 'concat_24_36::AntisymLinear::seed=44::lr=0.0001', 'old_frozen_pairwise_accuracy': 0.44871794871794873}`
- best_behaviorally_diverse_new_policy: `{'architecture': 'AntisymLinear', 'average_survivors': 1.0, 'config': 'concat_24_30_36', 'groups': 11, 'kept_good_branch_rate': 0.5454545454545454, 'oracle_gap': 0.43636363636363634, 'policy': 'new_hidden_origin_tap_v2_pairwise_tournament', 'pruned_oracle_branch_rate': 0.45454545454545453, 'reward_mean': 0.4727272727272727, 'selection_regret': 0.43636363636363634, 'top1_success': 0.5454545454545454, 'top2_oracle_coverage': 0.5454545454545454}`

## Layer And Geometry

- diversity_source_verdict: `TASK_SCREENING_HELPS`
- layer_config_verdict: `CONCAT_REQUIRED`
- geometry_verdict: `OLD_GEOMETRY_CONFIRMED`
- geometry: `{'max_abs_old_tap_alignment': 0.9883026480674744, 'mean_old_tap_alignment': 0.14519269515240812, 'v1_v2_alignment': 0.9998329877853394, 'mean_seed_config_stability': 0.2639620288193841}`

## Phase 2 Implication

PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_V2 = `WEAK`

Either expand once more or proceed only to a small selection-only prototype with the caveat locked in.
## Hidden-origin branch diversity v3 and selector reevaluation (2026-05-18)

- PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_V3 = `STILL_DATA_LIMITED`
- HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE = `v3_hidden_origin_tap`
- diversity_ablation_verdict = `DIVERSITY_IMPROVED`
- driver_verdict = `NON_RANDOM_DIRECTIONS_HELP`
- dataset_verdict = `STILL_DATA_LIMITED`
- training_verdict = `WEAK`
- eval_verdict = `DATA_LIMITED`
- geometry_verdict = `OLD_GEOMETRY_CONFIRMED`
- report: `artifacts/reports/probes/bg_hidden_origin_diversity_v3_2026-05-18/summary.md`

Continue targeted data expansion using the v3 recipe before making selector-readiness claims.

