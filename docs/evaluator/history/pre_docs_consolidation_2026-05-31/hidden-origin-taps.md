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

## Weight-space merged taps proposal (2026-05-18)

Hidden-origin taps and universal/bridge heads should be treated as candidate directions for an old-preserving merge, not as standalone replacements. The proposed next experiment extracts compatible weight vectors and adds branch-validity residuals to old objective/code anchors.

- planning doc: `docs/evaluator/bg_weight_space_merged_taps_plan.md`
- expected benefit: compact old+branch+bridge readout or better final-arbiter feature.
- status: not yet run.
