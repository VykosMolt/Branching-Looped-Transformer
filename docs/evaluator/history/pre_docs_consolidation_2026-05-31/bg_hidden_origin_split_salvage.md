# Hidden-Origin Branch Split Salvage

BG_HIDDEN_ORIGIN_SPLIT_AUDIT_VERDICT = READY
BG_HIDDEN_ORIGIN_EVAL_MODE_VERDICT = WEAK_ONLY
BG_HIDDEN_ORIGIN_SALVAGE_DATASET_VERDICT = READY
BG_HIDDEN_ORIGIN_SALVAGE_TRAINING_VERDICT = READY
BG_HIDDEN_ORIGIN_SALVAGE_EVAL_VERDICT = WEAK_SELECTOR
BG_HIDDEN_ORIGIN_CV_STABILITY_VERDICT = STABLE_POSITIVE
BG_HIDDEN_ORIGIN_V4_QUOTA_NEED_VERDICT = V4_REQUIRED_HELDOUT_BALANCE
HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE = old_frozen_bg
PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_SALVAGE = WEAK

## Why Salvage Was Needed

The v3 strict heldout set had only one task with non-tie reward pairs, despite v3 improving overall diversity. The salvage run tested whether alternate leakage-aware task splits and grouped CV could establish selector readiness without new branch generation.

## Leakage Handling

Strict and v3-clean modes preserve the original v3 empirical-direction heldout guard. Grouped CV and leave-one-task-out are task-disjoint for retrained tiny heads but are marked diagnostic when original empirical direction construction may have seen the heldout task.

## Result

The salvage status is `WEAK` with best available selector `old_frozen_bg` and v4 quota verdict `V4_REQUIRED_HELDOUT_BALANCE`.

## Phase 2 Implication

Do not claim action steering or selector readiness unless the clean heldout support and selector comparisons meet the readiness thresholds. If quota v4 is required, generate by pre-reserved split quotas instead of generate-then-split.
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

## Gated branch-content selector v1 (2026-05-18)

Gated/Fusion Branch-Content Selector v1 tested whether old content taps, hidden-origin branch taps, bridge heads, universal heads, and readiness diagnostics can be combined without collapsing all roles into one linear universal tap.

BG_GATED_SELECTOR_INVENTORY_VERDICT = READY
BG_GATED_SELECTOR_EXPERT_SCORES_VERDICT = READY
BG_GATED_SELECTOR_DATASET_VERDICT = READY
BG_GATED_SELECTOR_TRAINING_VERDICT = READY
BG_GATED_SELECTOR_EXPERT_ABLATION_VERDICT = INCONCLUSIVE
BG_GATED_OLD_CONTEXT_EVAL_VERDICT = MATCHES_OR_BEATS_OLD_TAPS
BG_GATED_HIDDEN_BRANCH_EVAL_VERDICT = SMALL_DEGRADATION
BG_GATED_BRIDGE_EVAL_VERDICT = BRIDGE_FIXED
BG_GATED_LAYERWISE_PRUNING_VERDICT = OLD_NEW_COMPOSITE_BEST
BG_GATED_DOMAIN_COVERAGE_VERDICT = MULTIDOMAIN_READY
BG_GATED_CALIBRATION_OOD_VERDICT = CALIBRATION_WEAK
BG_GATED_GEOMETRY_VERDICT = OLD_GEOMETRY_DOMINATES
BG_GATED_AS_OLD_TAP_REPLACEMENT_VERDICT = SAFE_REPLACEMENT_CANDIDATE
GATED_BRANCH_CONTENT_SELECTOR_STATUS = OLD_NEW_COMPOSITE_SUFFICIENT

- recommendation: `Prefer the simpler old+branch+bridge composite over the learned gate for now; keep top-k survival and do not change production routing.`
- no Ouro weights, tokenizer files, checkpoints, old taps, tap registries, wrapper/local-agent routing, or production routing were modified.
- expert/tap scores were used only as input features, not as labels.
