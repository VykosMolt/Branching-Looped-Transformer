<!-- docs-consolidation-source-note -->
> Consolidation note (2026-05-31): this is a source run note. The current consolidated interpretation is in `branch-generation-and-survival.md`. Exact pre-consolidation text is archived under `docs/evaluator/history/pre_docs_consolidation_2026-05-31/`.

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

