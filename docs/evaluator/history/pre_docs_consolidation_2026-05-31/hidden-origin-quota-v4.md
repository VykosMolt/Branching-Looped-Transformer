# Hidden-Origin Quota V4

Quota v4 tested split-reserved, quota-directed hidden-origin branch generation. It was needed because split salvage found real selector signal, but strict heldout support was too small to certify readiness.

## Verdicts

```text
BG_HIDDEN_ORIGIN_QUOTA_PLAN_V4_VERDICT = READY
BG_HIDDEN_ORIGIN_DIRECTION_BANK_V4_VERDICT = READY
BG_HS_INSPIRED_QUOTA_CONTROLLER_V4_VERDICT = INSUFFICIENT
BG_HIDDEN_ORIGIN_QUOTA_GENERATION_V4_VERDICT = PARTIAL
BG_HIDDEN_ORIGIN_QUOTA_DIVERSITY_DRIVER_V4_VERDICT = HS_INSPIRED_CONTROLLER_HELPS
BG_HIDDEN_ORIGIN_QUOTA_DATASET_V4_VERDICT = STILL_DATA_LIMITED
BG_HIDDEN_ORIGIN_TAP_TRAINING_V4_VERDICT = WEAK
BG_HIDDEN_ORIGIN_SELECTOR_EVAL_V4_VERDICT = STILL_DATA_LIMITED
BG_HIDDEN_ORIGIN_OLD_CONTEXT_REPLAY_V4_VERDICT = PARTIAL_MATCH
BG_HIDDEN_ORIGIN_TAP_GEOMETRY_V4_VERDICT = OLD_GEOMETRY_CONFIRMED
HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE_V4 = ensemble
PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_V4 = STILL_DATA_LIMITED
```

## Quota Result

| split | stable rows | groups | diverse groups | non-tie pairs | productive task IDs | quota met |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| train | 2000 | 250 | 52 / 60 | 617 / 250 | 9 / 24 | no |
| val | 800 | 100 | 12 / 15 | 173 / 60 | 1 / 6 | no |
| heldout | 1176 | 147 | 8 / 20 | 64 / 120 | 3 / 8 | no |

Heldout was the blocker. Only `OpenBookQA/14`, `OpenBookQA/18`, and `mmlu/high_school_chemistry/1` produced heldout non-tie pairs.

## Generation Findings

- The local Hunter-Seeker-inspired recipe controller helped diversity yield, but not enough to make heldout ready.
- K=8 helped.
- Non-random directions helped.
- `alpha=0.005` beat `alpha=0.01` in v4.
- L24 beat L36 in v4, so earlier branch points became a serious candidate for the next generator.

## Selector Result

V4 trained a weak but real hidden-origin head. The best v4 head was:

```text
architecture = AntisymLinearNoNorm
config = concat_24_36_47
train pairwise accuracy = 0.716
val pairwise accuracy = 0.728
```

Readiness was not claimed because heldout support was too small.

## Implication

The selector path was no longer the main suspect after v4. The bottleneck moved to hidden-origin branch generation and heldout behavioral diversity.

Primary report:

- `artifacts/reports/probes/bg_hidden_origin_quota_v4_2026-05-18/summary.md`

