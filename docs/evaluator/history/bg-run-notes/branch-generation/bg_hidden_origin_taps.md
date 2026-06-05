<!-- docs-consolidation-source-note -->
> Consolidation note (2026-05-31): this is a source run note. The current consolidated interpretation is in `branch-generation-and-survival.md`. Exact pre-consolidation text is archived under `docs/evaluator/history/pre_docs_consolidation_2026-05-31/`.

# Hidden-Origin Branch Taps

Hidden-origin taps are separate from the frozen canonical BG taps. The old BG heads read hidden states, but their training states came from normal candidate/text/code/option trajectories. This experiment trains tiny heads on same-prefix hidden-state perturbation branches and labels pairs by downstream branch outcomes.

## Verdicts

- BG_HIDDEN_ORIGIN_TAP_INVENTORY_VERDICT = `SMALL_BUT_USABLE`
- BG_HIDDEN_ORIGIN_DATA_EXPANSION_VERDICT = `READY`
- BG_HIDDEN_ORIGIN_TAP_DATASET_VERDICT = `SMALL_BUT_USABLE`
- BG_HIDDEN_ORIGIN_TAP_TRAINING_VERDICT = `READY`
- BG_HIDDEN_ORIGIN_TAP_EVAL_VERDICT = `INSUFFICIENT`
- BG_HIDDEN_ORIGIN_LAYER_CONFIG_VERDICT = `INSUFFICIENT`
- BG_HIDDEN_ORIGIN_TAP_GEOMETRY_VERDICT = `ALIGNS_WITH_OLD_TAPS`
- PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS = `DATA_LIMITED`

## Dataset And Labels

Rows are filtered to safe alpha branches (`alpha <= 0.01`) and stable outputs. Pair labels are built only within the same hidden-origin branch group: the preferred branch has the higher downstream reward. Reward ties are omitted from primary training rather than assigned arbitrary labels.

- pairs_by_split: `{'test': 12, 'train': 31, 'val': 12}`
- tie_rate: `0.940`

## Training

The headline architectures are exact antisymmetric tiny heads: `AntisymLinear` and `AntisymLinearNoNorm`. Training uses a Bradley-Terry/logsigmoid-style pairwise objective with 50 percent random left/right swaps and target sign flips. The report includes flip diagnostics and rejects constant-score solutions.

- best_head: `{'architecture': 'AntisymLinear', 'config': 'concat_24_36', 'dim': 4096, 'flip_diagnostics': {'antisymmetry_correlation': 1.0, 'mean_abs_score_sum': 0.0, 'mean_score_sum': 0.0, 'n': 12, 'passes': True, 'score_mean': 0.005578702315688133, 'score_std': 0.0017384453676640987, 'strict_sign_flip_rate': 1.0}, 'head_group': 'hidden_origin_branch_taps', 'metrics': {'architecture': 'AntisymLinear', 'best_epoch': 1, 'config': 'concat_24_36', 'dim': 4096, 'epochs_run': 9, 'gradient_clip': 1.0, 'history': [{'epoch': 1, 'train_acc': 0.8709677457809448, 'train_loss': 0.6920281648635864, 'val_acc': 1.0, 'val_loss': 0.6903620958328247}, {'epoch': 2, 'train_acc': 0.9677419066429138, 'train_loss': 0.6882745623588562, 'val_acc': 0.8333333730697632, 'val_loss': 0.6913203001022339}, {'epoch': 3, 'train_acc': 0.9677419066429138, 'train_loss': 0.6845536828041077, 'val_acc': 0.75, 'val_loss': 0.6922799348831177}, {'epoch': 4, 'train_acc': 1.0, 'train_loss': 0.6808658242225647, 'val_acc': 0.4166666865348816, 'val_loss': 0.6932425498962402}, {'epoch': 5, 'train_acc': 1.0, 'train_loss': 0.6772115230560303, 'val_acc': 0.3333333432674408, 'val_loss': 0.6942092776298523}, {'epoch': 6, 'train_acc': 1.0, 'train_loss': 0.6735910773277283, 'val_acc': 0.1666666716337204, 'val_loss': 0.6951793432235718}, {'epoch': 7, 'train_acc': 1.0, 'train_loss': 0.6700047254562378, 'val_acc': 0.1666666716337204, 'val_loss': 0.69615238904953}, {'epoch': 8, 'train_acc': 1.0, 'train_loss': 0.6664526462554932, 'val_acc': 0.1666666716337204, 'val_loss': 0.6971286535263062}, {'epoch': 9, 'train_acc': 1.0, 'train_loss': 0.6629351377487183, 'val_acc': 0.1666666716337204, 'val_loss': 0.6981080770492554}], 'lr': 0.001, 'objective': 'directional_pairwise_logsigmoid', 'score_l2': 0.0001, 'seed': 44, 'swap_protocol': '50_percent_random_left_right_with_target_sign_flip', 'train_pairs': 31, 'train_pairwise_accuracy': 0.8709677457809448, 'val_pairs': 12, 'validation_loss': 0.6903620958328247, 'validation_pairwise_accuracy': 1.0}}`

## Heldout Selection

Heldout evaluation uses task IDs excluded from training and compares random, clean branch, old frozen BG tap margins, and the new hidden-origin tap policies. The behaviorally diverse subset is the load-bearing subset.

- best_behaviorally_diverse_new_policy: `{'architecture': 'AntisymLinear', 'average_survivors': 1.0, 'config': 'concat_24_36', 'groups': 3, 'kept_good_branch_rate': 1.0, 'oracle_gap': 0.0, 'policy': 'new_hidden_origin_tap_pairwise_tournament', 'pruned_oracle_branch_rate': 0.0, 'reward_mean': 1.0, 'selection_regret': 0.0, 'top1_success': 1.0, 'top2_oracle_coverage': 1.0}`
- old_frozen_pairwise_accuracy: `0.667`

## Layer And Geometry

- layer_config_verdict: `INSUFFICIENT`
- first_phase2_scoring_point_recommendation: `insufficient`
- geometry_verdict: `ALIGNS_WITH_OLD_TAPS`

## Phase 2 Implication

PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS = `DATA_LIMITED`

Generate more hidden-origin branch outcome groups.
## Hidden-origin branch diversity v2 and tap reevaluation (2026-05-18)

- PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_V2 = `WEAK`
- generation_verdict = `READY`
- dataset_verdict = `SMALL_BUT_USABLE`
- training_verdict = `READY`
- eval_verdict = `WEAK_SELECTOR`
- layer_config_verdict = `CONCAT_REQUIRED`
- geometry_verdict = `OLD_GEOMETRY_CONFIRMED`
- report: `artifacts/reports/probes/bg_hidden_origin_diversity_v2_2026-05-18/summary.md`

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

## Fixed-composite branch survival policy v1 (2026-05-18)

This run converted the corrected gated selector result into a validation-selected fixed old+branch+bridge survival policy with explicit veto/rescue and missing-expert/OOD fallback.

BG_FIXED_COMPOSITE_SURVIVAL_INVENTORY_VERDICT = READY
BG_FIXED_COMPOSITE_SURVIVAL_DATASET_VERDICT = READY
BG_FIXED_COMPOSITE_SURVIVAL_FEATURES_VERDICT = READY
BG_FIXED_COMPOSITE_SURVIVAL_BASELINES_VERDICT = READY
BG_FIXED_COMPOSITE_OPTIMIZATION_VERDICT = OLD_BRANCH_BRIDGE_SUFFICIENT
BG_FIXED_COMPOSITE_VETO_RESCUE_OPTIMIZATION_VERDICT = READY
BG_FIXED_COMPOSITE_LEARNED_RESCUE_VERDICT = WORSE_THAN_RULES
BG_FIXED_COMPOSITE_MISSING_OOD_POLICY_VERDICT = ROBUST
BG_FIXED_COMPOSITE_SURVIVAL_HELDOUT_EVAL_VERDICT = SURVIVAL_READY
BG_FIXED_COMPOSITE_SURVIVAL_FRONTIER_VERDICT = CLEAR_OPERATING_POINT
BG_FIXED_COMPOSITE_LAYER_ORIGIN_DOMAIN_VERDICT = UNIFORM_POLICY_SUFFICIENT
BG_FIXED_COMPOSITE_OLD_CODE_PRESERVATION_VERDICT = PRESERVED
BG_FIXED_COMPOSITE_SELECTION_ONLY_READINESS_VERDICT = READY
FIXED_COMPOSITE_BRANCH_SURVIVAL_POLICY_STATUS = SURVIVAL_READY

- selected policy: `selected_policy = fixed_composite_conservative_top4; oracle_retention = 0.931; false_prune_rate = 0.069; avg_survivors = 3.873`
- recommendation: `Proceed to a small selection-only Phase 2 prototype using BGV1 branches, the fixed old+branch+bridge composite, the selected conservative top-k survival operating point, and missing/OOD fallback. Keep veto/rescue as a guardrail, not as a replacement for the selected heldout-ready operating point. selected_policy = fixed_composite_conservative_top4; oracle_retention = 0.931; false_prune_rate = 0.069; avg_survivors = 3.873. Do not claim action steering.`
- learned gated selector remains diagnostic; it is not the primary pruning selector.
- no Ouro weights, tokenizer files, checkpoints, old tap registries, wrapper/local-agent routes, or production routing were modified.

## Selection-only Phase 2 prototype v1 (2026-05-18)

SELECTION_ONLY_PHASE2_PROTOTYPE_STATUS = SURVIVAL_READY_FINAL_ARBITER_WEAK

- cached reproduction: `REPRODUCED`
- live/counterfactual prototype: `SURVIVAL_POSITIVE_FINAL_SELECTION_WEAK`
- final arbiter: `FINAL_SELECTION_WEAK`
- steering readiness: `NEEDS_FINAL_ARBITER_FIRST`
- recommendation: Train or evaluate a stronger final arbiter among top4 survivors before steering.
- no action steering was tested; no production routing changed.

## Final arbiter among top4 survivors v1 (2026-05-18)

FINAL_ARBITER_TOP4_STATUS = FINAL_ARBITER_WEAK_BUT_USEFUL
SELECTION_ONLY_PHASE2A_STATUS_AFTER_FINAL_ARBITER = NEEDS_MORE_FINAL_ARBITER_WORK

- heldout eval: `FINAL_ARBITER_WEAK`
- selected model: `listwise_softmax`
- readiness: `FINAL_ARBITER_WEAK_BUT_IMPROVED`
- recommendation: Run a small improved-arbiter v1.1 or proceed only with explicit weak-baseline caveat.
- no action steering was tested.

## Final arbiter among top4 survivors v1.1 (2026-05-18)

FINAL_ARBITER_TOP4_V1_1_STATUS = NO_IMPROVEMENT
SELECTION_ONLY_PHASE2A_STATUS_AFTER_FINAL_ARBITER_V1_1 = NEEDS_DOMAIN_SPECIALIZATION

- split guard: `FRESH_HELDOUT_READY`
- selected model: `tie_aware_rank_listwise`
- heldout eval: `NO_IMPROVEMENT`
- readiness: `NEEDS_REASONING_ARBITER`
- recommendation: Return to expert/bridge signal quality; v1.1 did not improve final selection.
- no action steering was tested.

## Weight-space merged taps proposal (2026-05-18)

Hidden-origin taps and universal/bridge heads should be treated as candidate directions for an old-preserving merge, not as standalone replacements. The proposed next experiment extracts compatible weight vectors and adds branch-validity residuals to old objective/code anchors.

- planning doc: `docs/evaluator/bg_weight_space_merged_taps_plan.md`
- expected benefit: compact old+branch+bridge readout or better final-arbiter feature.
- status: not yet run.

## Weight-space merged branch-content taps v1 (2026-05-18)

`MERGED_WEIGHT_BRANCH_CONTENT_TAP_STATUS = FINAL_ARBITER_IMPROVES_ONLY`. The run extracted old/content, hidden-branch, and bridge tap directions, aligned them into shared feature coordinates, built residualized merged candidates, and acquired top4 survivor hidden features from cached raw artifacts for final-arbiter rescoring. No action steering or routing change was tested.

Report: `docs/evaluator/bg_merged_weight_branch_content_taps_v1.md`.

## Old-anchored branch-valid taps v1 (2026-05-30)

`BG_OLD_ANCHORED_BRANCH_VALID_TAP_STATUS = OLD_ANCHORED_BRANCH_TAP_USEFUL`. Tested weight-space transplant and constrained fine-tuned copies of old coding/reasoning and mixed-objective anchors. Selected `weight_space_transplant` candidate `transplant::MIX_CODE_REASONING::full_residual::AntisymLinearNoNorm::102` with heldout old/code drops `0.0000` / `0.0000` and branch/bridge gains `0.1594` / `0.2121` vs its matching old anchor. No old taps, registries, Ouro weights, routing, or steering were modified.

Report: `docs/evaluator/bg_old_anchored_branch_valid_taps_v1.md`.

## Two-tap full readiness v1 (2026-05-30)

`BG_TWO_TAP_FULL_READINESS_VERDICT = TWO_TAP_PARTIAL_NOT_READY`. Tested the transplanted `MIX_CODE_REASONING` and `MIX_OBJECTIVE_ALL` taps against old-domain references and branch/universal references across cached old-domain and branch datasets. Domain ok `False`; branch ok `False`. No Ouro training, steering, registry update, routing change, or production change was performed.

Report: `docs/evaluator/bg_two_tap_full_readiness_v1.md`.

## Layer-native two-tap readiness v1 (2026-05-30)

`BG_LAYER_NATIVE_TWO_TAP_READINESS_VERDICT = LAYER_NATIVE_TWO_TAP_PARTIAL_NOT_READY`. Re-tested `MIX_CODE_REASONING` and `MIX_OBJECTIVE_ALL` as native `24_L4`, `36_L4`, and `47_L4` tap bundles instead of concat-only taps. Domain ok `False`; branch ok `False`. No Ouro training, old registry update, steering, wrapper/local-agent execution, or routing change was performed.

Report: `docs/evaluator/bg_layer_native_two_tap_readiness_v1.md`.

## Layer-native two-tap constrained training v1 (2026-05-30)

`BG_LAYER_NATIVE_TWO_TAP_CONSTRAINED_TRAINING_VERDICT = CONSTRAINED_TWO_TAP_NOT_READY`; readiness verdict `LAYER_NATIVE_TWO_TAP_PARTIAL_NOT_READY`. Trained copied `MIX_CODE_REASONING` and `MIX_OBJECTIVE_ALL` layer-local taps at `24_L4`, `36_L4`, and `47_L4` on train-split old-domain plus branch/bridge labels with anchor preservation. Domain ok `False`; branch ok `False`. No Ouro training, old registry update, steering, wrapper/local-agent execution, or routing change was performed.

Report: `docs/evaluator/bg_layer_native_two_tap_constrained_train_v1.md`.

## Layer-native two-tap targeted rehost diagnostic v1 (2026-05-30)

`BG_LAYER_NATIVE_TWO_TAP_TARGETED_REHOST_DIAGNOSTIC_VERDICT = TARGETED_REHOST_DIAGNOSTIC_PASSES`. Diagnostic-only exact source rehost under the two tap identities produced readiness verdict `LAYER_NATIVE_TWO_TAP_READY`. This is not clean readiness because target references came from prior branch failure reports.

Report: `docs/evaluator/bg_layer_native_two_tap_targeted_rehost_diagnostic_v1.md`.

## DualAnchor branch-gap repair v1 (2026-05-30)

`BG_DUALANCHOR_BRANCH_GAP_REPAIR_VERDICT = DUALANCHOR_BRANCH_NOT_READY`; readiness verdict `LAYER_NATIVE_TWO_TAP_PARTIAL_NOT_READY`. Targeted copied `MIX_CODE_REASONING` and `MIX_OBJECTIVE_ALL` layer-native heads at `24_L4`, `36_L4`, and `47_L4` on train-split branch-gap examples with old-domain preservation. Best-candidate domain ok `False`; best-candidate branch ok `False`; selected-bundle branch ok `False`. No Ouro training, old registry update, steering, wrapper/local-agent execution, or routing change was performed.

Report: `docs/evaluator/bg_dualanchor_branch_gap_repair_v1.md`.

## DualAnchor pre-repair fixed-bundle audit v1 (2026-05-30)

`BG_DUALANCHOR_PRE_REPAIR_FIXED_BUNDLE_AUDIT_VERDICT = DUALANCHOR_FIXED_BUNDLE_NOT_READY`. Selected diagnostic fixed bundle `bundle::two_tap_equal::adaptive_balanced_rescue::AntisymLinear::24_36_47`. Ready fixed bundles `0` / `132`; domain-ok fixed bundles `0` / `132`. No training, failed-repair weights, old registry update, steering, or routing change.

Report: `docs/evaluator/bg_dualanchor_pre_repair_fixed_bundle_audit_v1.md`.

## DualAnchor hard-anchor selector v1 (2026-05-30)

`BG_DUALANCHOR_HARD_ANCHOR_SELECTOR_VERDICT = DUALANCHOR_HARD_ANCHOR_NOT_READY`. Selected fixed pre-repair DualAnchor bundle `bundle::two_tap_equal::sparse_old_plus_bridge_30_70_top0p01::AntisymLinear::24_36_47` with validation old-anchor gate `False`. Full replay domain ok `False`; branch ok `False`. No training, failed-repair weights, old registry update, steering, or routing change.

Report: `docs/evaluator/bg_dualanchor_hard_anchor_selector_v1.md`.

## DualAnchor architecture-looped stratified probe v3 (2026-05-31)

Status: `ARCHITECTURE_LOOPED_SURVIVAL_READY_TERMINAL_DEFER_REQUIRED`.

This run scaled the DualAnchor architecture-shaped loop without steering. Taps were active at layers 24, 36, and 47 across loops L1-L4, with only terminal `L4_47` eligible for confidence-gated collapse. It uses cumulative hook approximation at decoder-layer surfaces; it does not claim autoregressive branch-specific KV/cache fork/carry or compute savings.

Headline metrics:

- tasks: `48`
- stage oracle retention: `0.9848484848484849`
- terminal oracle retained: `1.0`
- terminal forced top1 oracle: `0.9166666666666666`
- terminal reward-diverse rate: `0.22916666666666666`
- positive-oracle rate: `0.3541666666666667`

Locked-baseline candidate:

- selector: DualAnchor `MIX_CODE_REASONING` + `MIX_OBJECTIVE_ALL`
- schedule: `L1_24 -> L1_36 -> L1_47 -> L2_24 -> L2_36 -> L2_47 -> L3_24 -> L3_36 -> L3_47 -> L4_24 -> L4_36 -> terminal L4_47`
- threshold: `mean_floor_very_loose`
- budget: `8`
- L47: active in nonterminal loops
- terminal: confidence-gated top1; otherwise defer/keep terminal survivors

Readiness verdict: `READY_WITH_TERMINAL_DEFER`.
No steering was tested.

## DualAnchor convergence hairs and reasoning/science pre-steering probe v1 (2026-05-31)

`DUALANCHOR_CONVERGENCE_HAIRS_RS_STATUS = SCIENCE_BRANCH_GENERATION_WEAK`.
The L30/L42 convergence-hair replay did not clear hard-merge readiness; use hairs as
soft diagnostics only for now. Terminal confidence/defer remains required, especially on
reasoning hard slices, and science needs a different branch-generation recipe before it
can be treated as a headline steering domain. Branch classification remained
diagnostic-only; no steering, routing change, compute-savings claim, or fork/carry claim
was introduced.

Report: `docs/evaluator/bg_dualanchor_convergence_hairs_reasoning_science_v1.md`.

## DualAnchor science branch recipe and reasoning terminal defer v1 (2026-05-31)

`DUALANCHOR_SCIENCE_RECIPE_REASONING_DEFER_STATUS = PRE_STEERING_READY_WITH_SCIENCE_DIAGNOSTIC`.
The science branch recipe replay did not find a calibration improvement over baseline.
Parser/reward is partly responsible, but science headline readiness remains blocked by
MMLU science branch generation. Reasoning branch generation stays baseline-sufficient;
terminal defer/full survivor handoff is required. No steering, training, hard hair merge,
or production routing change was made.

Report: `docs/evaluator/bg_dualanchor_science_branch_recipe_reasoning_defer_v1.md`.
