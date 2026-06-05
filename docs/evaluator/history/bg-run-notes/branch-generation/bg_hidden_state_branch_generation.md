<!-- docs-consolidation-source-note -->
> Consolidation note (2026-05-31): this is a source run note. The current consolidated interpretation is in `branch-generation-and-survival.md`. Exact pre-consolidation text is archived under `docs/evaluator/history/pre_docs_consolidation_2026-05-31/`.

# BG Hidden-State Branch Generation Suite

Date: 2026-05-18

## Corrected distinction

The older evaluator branch artifacts are mostly text/candidate branches represented through BG tap features. They are useful offline selection sanity checks, but they are not latent hidden-state forks.

This suite tests same-prefix hidden-origin branches:

- same prompt and token prefix
- hidden perturbation at an internal Ouro layer/loop
- branch continuation through Ouro
- downstream MCQ outcome scoring

## Feasibility and method

`BG_HIDDEN_BRANCH_FEASIBILITY_VERDICT = HOOK_HIDDEN_ORIGIN_READY`

`LIVE_BRANCH_METHOD = hook_intervention_per_branch`

True autoregressive fork/carry is still blocked. Local Ouro exposes layer hooks, `current_ut`, per-loop hidden states, and a UniversalTransformerCache, but there is no validated generation API for resuming from a copied internal layer hidden state with matching branch-specific cache/state. The suite therefore uses same-prefix hook-hidden-origin branches and keeps that method label explicit.

## Dataset

Artifacts:

- `artifacts/reports/probes/bg_hidden_state_branch_generation_2026-05-18/summary.md`
- `artifacts/reports/probes/bg_hidden_state_branch_generation_2026-05-18/analysis.md`
- `artifacts/reports/probes/bg_hidden_state_branch_generation_2026-05-18/hidden_branch_persistence.pt`
- `artifacts/reports/probes/bg_hidden_state_branch_generation_2026-05-18/hidden_branch_outcomes.json`

Task subset:

- `BG_HIDDEN_BRANCH_TASK_SUBSET_VERDICT = READY`
- 16 reasoning/science MCQ tasks selected.
- Live bounded run used 8 tasks.
- K=4 branches per group.
- Safe alphas: `0.005`, `0.01`.
- Diagnostic alphas: `0.05`, `0.1`, outside the safety envelope and not headline evidence.

## Branch generation and persistence

`BG_HIDDEN_BRANCH_GENERATION_VERDICT = HOOK_HIDDEN_ORIGIN_BRANCHES_GENERATED`

`BG_LATENT_BRANCH_PERSISTENCE_VERDICT = LATENT_BRANCHES_PERSIST_TO_47`

Records:

- branch records: 112
- branch groups: 28
- safe-alpha groups: 24
- diagnostic high-alpha groups: 4
- errors: 0

The persistence verdict is based on geometric hidden-feature distinctness, not tap-score spread. This matters because prior steering work found signed/unsigned ambiguity: `+delta` and `-delta` can be tap-indifferent even when geometrically distinct.

Safe-alpha relative persistence rates:

- L30: 0.667
- L36: 1.000
- L42: 1.000
- L47: 1.000

Caveat: absolute hidden RMS distances remained small. The high-alpha diagnostic did not cleanly separate generation-strength artifact from convergence pressure. Safe-alpha mean L47 retention was larger than diagnostic retention by ratio, while absolute L47 distances were of the same order. Treat this as "measurable persistence exists" rather than "large branch separation exists."

## Outcome diversity

`BG_HIDDEN_BRANCH_OUTCOME_DATASET_VERDICT = READY`

Outcome records:

- rows: 112
- groups: 28
- safe eligible groups: 24
- eligible task count: 8
- behaviorally diverse safe groups: 6
- reward-diverse safe groups: 6
- diversity rate: 0.25

This answers the strongest null: safe same-prefix hidden-origin branches are not always behaviorally neutral. They sometimes produce different parsed answers or rewards.

## Frozen tap selection

`BG_HIDDEN_ORIGIN_BRANCH_SELECTION_VERDICT = NO_HIDDEN_BRANCH_SELECTION_SIGNAL`

Safe hidden-origin branch selection metrics:

| policy | top1 success | reward mean | coverage/survivors |
| --- | ---: | ---: | ---: |
| random_top1 | 0.625 | 0.5708 | 4.0 survivors |
| clean_branch_baseline | 0.500 | 0.4500 | 1.0 survivor |
| pairwise_tournament_winner | 0.583 | 0.5250 | 1.0 survivor |
| pairwise_tournament_top2 | 0.667 | 0.6167 | 2.0 survivors |

The frozen BG taps did not beat random top1 on hidden-origin branches. The measured lift for the tournament winner was `-0.0417`.

Interpretation: the readout heads that work on cached text/candidate branches are not calibrated selectors for same-prefix hidden-origin branches. This is the main Phase 2 bottleneck.

## L30/L42 gate implications

`BG_HIDDEN_BRANCH_L30_L42_GATE_VERDICT = NEEDS_STRONGER_BRANCH_GENERATOR`

The strict gate analyzer found L30/L42 representative merge unsafe:

- L30 collapse rate under strict absolute threshold: 1.000
- L42 collapse rate under strict absolute threshold: 1.000
- mixed outcome cluster rate: 0.25
- oracle retention if merged: 0.667

Do not add L30/L42 convergence merge gates to the first prototype. The relative geometry persists, but absolute separations are tiny and outcome-mixed clusters make merge/prune unsafe.

## Adaptive thresholds

`BG_HIDDEN_BRANCH_ADAPTIVE_THRESHOLD_VERDICT = TOPK_SUFFICIENT`

Adaptive thresholds did not beat simple top-k at matched compute. `top2`, `score_spread_adaptive`, `instability_penalty_policy`, and `compute_budget_policy` all retained oracle branches at 0.667 with roughly two survivors. A diversity-bonus policy reached 0.708 oracle retention but used more compute.

## Cached sanity comparison

`BG_CACHED_BRANCH_SELECTION_SANITY_VERDICT = INSUFFICIENT`

Cached branch selection remains useful as offline sanity only. It does not prove hidden-origin branch viability.

## Phase 2 readiness

`PHASE2_HIDDEN_BRANCH_READINESS = NEEDS_BETTER_BRANCH_EVALUATOR`

The suite establishes:

- same-prefix hidden-origin branches can be generated with the hook fallback
- true fork/carry is still blocked
- geometric branch distinctness is measurable through L47
- some safe-alpha hidden branches produce different downstream outcomes
- frozen taps do not select the good hidden-origin branches better than random
- L30/L42 convergence gates are not justified for the first prototype
- adaptive thresholds do not beat simple top-k on this data

## Recommended minimal prototype

Do not claim action steering from this suite.

The justified next prototype is:

- hidden-origin branch evaluator/calibration dataset from same-prefix outcomes
- hook-hidden-origin branch generator as the branch source
- capture at L24/L30/L36/L42/L47
- frozen BG taps retained as diagnostics, not as the selection policy
- no representative merge at L30/L42
- no hidden-state averaging merge
- no backbone or BG tap training until the evaluator target is made explicit

For true branch-native Phase 2, implement branch-aware Ouro forward/cache so copied hidden state can be carried through generation without relying on repeated hook intervention.
## Hidden-origin branch taps (2026-05-18)

- PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS = `DATA_LIMITED`
- tap_eval_verdict = `INSUFFICIENT`
- tap_training_verdict = `READY`
- layer_config_verdict = `INSUFFICIENT`
- geometry_verdict = `ALIGNS_WITH_OLD_TAPS`
- report: `artifacts/reports/probes/bg_hidden_origin_taps_2026-05-18/summary.md`

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

## Weight-space merged branch-content taps v1 (2026-05-18)

`MERGED_WEIGHT_BRANCH_CONTENT_TAP_STATUS = FINAL_ARBITER_IMPROVES_ONLY`. The run extracted old/content, hidden-branch, and bridge tap directions, aligned them into shared feature coordinates, built residualized merged candidates, and acquired top4 survivor hidden features from cached raw artifacts for final-arbiter rescoring. No action steering or routing change was tested.

Report: `docs/evaluator/bg_merged_weight_branch_content_taps_v1.md`.

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
