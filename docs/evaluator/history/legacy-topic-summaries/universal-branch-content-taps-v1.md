# Universal Branch-Content Taps V1

Universal Branch-Content Taps v1 tested whether one tiny hidden-state pairwise evaluator can judge both ordinary content/candidate quality and same-prefix hidden-origin branch survival.

The answer is: not yet. The universal head learned useful signal, but bridge evaluation failed, so a composite or gated selector is better than forcing one universal tap.

## Verdicts

```text
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
```

## Dataset

| variant | pairs | train | val | heldout | composition |
| --- | ---: | ---: | ---: | ---: | --- |
| universal balanced | 1154 | 737 | 235 | 182 | 462 old, 462 branch, 230 bridge |
| universal no bridge | 924 | 590 | 188 | 146 | 462 old, 462 branch |
| old content only | 462 | 295 | 94 | 73 | old content |
| hidden branch only | 1753 | 1090 | 180 | 483 | hidden branch |
| bridge only | 2142 | 1342 | 220 | 580 | bridge |

Cached wrapper/code features were inspected, but they had no within-task non-tie labels, so coding remains coverage-limited.

## Best Universal Head

```text
variant = universal_balanced
architecture = AntisymLinear
config = concat_24_36_47
seed = 42
lr = 1e-4
balanced validation = 0.748
old-content validation = 0.649
hidden-branch validation = 0.830
bridge validation = 0.766
flip diagnostics = passed
```

## Heldout Evaluation

Old-context/content selection:

| selector | pairwise | top1 | top2 oracle retention |
| --- | ---: | ---: | ---: |
| universal balanced | 0.726 | 0.818 | 0.909 |
| old frozen objective/mixed | 0.507 | 0.636 | 0.773 |
| random | 0.500 | - | - |

Hidden-origin branch selection:

| selector | top1 | top2 oracle retention |
| --- | ---: | ---: |
| v4 hidden-origin | 0.467 | 0.733 |
| universal balanced | 0.378 | 0.689 |
| generator-v1 selector | 0.333 | 0.711 |
| old frozen BG | 0.200 | 0.756 |

Bridge pairs:

| selector | heldout pairwise |
| --- | ---: |
| hidden-branch only | 0.688 |
| bridge only | 0.667 |
| random | 0.500 |
| universal balanced | 0.460 |
| old frozen score | 0.447 |

The bridge failure is decisive: the mixed universal head did not connect branch viability to final content quality better than specialized heads.

## Geometry

`BG_UNIVERSAL_TAP_GEOMETRY_VERDICT = OLD_GEOMETRY_CONFIRMED`

Maximum alignment with reference hidden-origin/generator selectors was `0.987`. The universal head mostly confirms the old shared readout geometry; it is not a new stable bridge geometry.

## Current Recommendation

Use a composite selector rather than one universal head:

- old/content selector for ordinary content and trajectory contexts
- v4 or best hidden-origin selector for branch survival
- bridge/hidden-only signal as an auxiliary diagnostic
- top2/top3 survival for branch pruning, not top1 collapse

Primary report:

- `artifacts/reports/probes/bg_universal_branch_content_taps_v1_2026-05-18/summary.md`

## Weight-space merged taps proposal (2026-05-18)

Universal taps are useful as source directions, but not as a single replacement head. The follow-up plan is to extract `linear.weight` vectors from `universal_balanced`, `hidden_branch_only`, `bridge_only`, and old objective/code heads, then test old-preserving branch-validity residual merges.

- planning doc: `docs/evaluator/bg_weight_space_merged_taps_plan.md`
- expected role: diagnostic or compact feature, not production selector unless it beats fixed-composite survival and preserves old coding/reasoning behavior.
- no merged-weight run has been executed yet.
