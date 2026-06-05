# Hidden-Origin Branch Generator V1

Branch Generator v1 tested stronger same-prefix hidden-origin generation after quota v4 failed heldout diversity. It focused on earlier branch points, richer recipe search, better high-yield directions, and true fork/carry feasibility checks.

## Verdicts

```text
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
```

## Quota Result

| split | stable rows | groups | diverse groups | non-tie pairs | productive task IDs | quota met |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| train | 2400 | 300 | 35 / 60 | 473 / 250 | 5 / 24 | no |
| val | 1000 | 127 | 1 / 15 | 7 / 60 | 1 / 6 | no |
| heldout | 656 | 82 | 43 / 20 | 419 / 120 | 8 / 8 | yes |

BGV1 fixed the previous heldout-diversity blocker, but train and val became weak. The result is usable for diagnostics and selection-only prototypes with caveats, not a clean ready state.

## Generator Findings

- Heldout diversity improved strongly.
- The best generator method was the local HS-inspired controller.
- True fork/carry was still not mechanically ready; the run used hook fallback honestly.
- Non-random directions remained useful.
- K=8 remained useful.
- L24 remained slightly better than L36.
- Stronger diversity did not appear to be just instability.

## Selector Result

The best selector after BGV1 remained the v4 hidden-origin tap:

| selector | top1 | top2 oracle retention |
| --- | ---: | ---: |
| v4 hidden-origin | 0.467 | 0.733 |
| generator-v1 selector | 0.333 | 0.711 |
| old frozen BG | 0.200 | 0.756 |
| clean branch | 0.133 | 0.356 |

## Implication

Branch generation is now weak-but-usable, but the clean train/val/heldout balance is still not strong enough for a broad readiness claim. A small selection-only prototype is defensible only with clear caveats and top-k survival.

Primary report:

- `artifacts/reports/probes/bg_hidden_origin_branch_generator_v1_2026-05-18/summary.md`

