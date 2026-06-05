# Gated Branch-Content Selector V1

This experiment tested a gated/composite branch-content selector after the universal linear tap result showed `FUSION_NEEDED`.

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

## Result

Prefer the simpler old+branch+bridge composite over the learned gate for now; keep top-k survival and do not change production routing.

## Architecture

The selector combines cached old-content, hidden-origin, bridge, universal, and generator selector scores with metadata and readiness diagnostics. Expert scores are inputs only; labels remain actual correctness, reward, verifier, preference, or deterministic branch reward.

## Safety

- No Ouro training or checkpoint/tokenizer/model edits.
- No old tap registry update.
- No wrapper/local-agent or actual Hunter-Seeker execution.
- No production routing change or action-steering claim.

## Weight-space merged taps proposal (2026-05-18)

The learned gate remained diagnostic and the fixed composite won. The proposed weight-space follow-up tests a different compression path: merge old objective/code directions with orthogonalized branch and bridge residual directions, then evaluate against the existing score-level composite.

- planning doc: `docs/evaluator/bg_weight_space_merged_taps_plan.md`
- use tap weights as candidate directions only; labels remain external reward/correctness/verifier labels.
- no merged-weight run has been executed yet.
