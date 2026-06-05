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

