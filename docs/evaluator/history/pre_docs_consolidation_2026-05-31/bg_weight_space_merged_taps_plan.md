# Weight-Space Merged Taps Plan

Status: proposal / next experiment, not yet run.

This note records the proposed follow-up after Universal Branch-Content Taps v1 and Final Arbiter v1.1. The aim is to test whether tiny tap weight vectors can be extracted, analyzed, orthogonalized, and merged into old-style objective heads that preserve `coding_reasoning` / `mixed_objective_all` behavior while adding hidden-branch validity.

## Motivation

Universal Branch-Content Taps v1 showed useful signals but did not justify replacing separated experts with one universal head:

- `universal_balanced` was useful as a general expert.
- `hidden_branch_only` was useful as a branch-survival diagnostic.
- `bridge_only` was useful as a bridge diagnostic.
- The overall status remained `UNIVERSAL_BRANCH_CONTENT_TAP_STATUS = FUSION_NEEDED`.

Fixed-composite survival then succeeded by keeping experts separate:

- old/content score
- branch score
- bridge score

Final Arbiter v1/v1.1 showed that final selection among top4 remains weak. The next clean hypothesis is therefore not "one universal head", but "old-style heads plus explicit branch-validity residual directions".

## Weight Geometry

The useful tap heads are tiny linear or normalized-linear pairwise heads:

```text
AntisymLinearNoNorm:
  score(left, right) = w . (left - right)

AntisymLinear:
  score(left, right) = w . LayerNorm(left - right)
```

The saved heads expose `state_dict["linear.weight"]`, so the learned direction can be extracted as a vector. For example, the best universal head was:

```text
variant = universal_balanced
architecture = AntisymLinear
config = concat_24_36_47
weight shape = (1, 6144)
```

Direct merging is safest when all candidate heads share the same config and architecture. If configs differ, either train/export heads into a common config or lift smaller vectors into a larger concat space with zero-filled missing blocks.

Architecture families must stay separated for primary claims:

- `AntisymLinearNoNorm` heads can be merged directly in raw feature-difference space.
- `AntisymLinear` heads can be merged only as normalized-feature scoring directions.
- Mixed `AntisymLinear` / `AntisymLinearNoNorm` merges are diagnostic unless separately validated.
- Any saved merged tap must include the exact scoring path, feature config, feature ordering, and normalizer/statistics needed to reproduce its scores.

## Candidate Source Directions

Primary source directions:

- old `coding_reasoning`
- old `mixed_objective_all`
- hidden-origin / branch-validity heads
- `universal_balanced`
- `hidden_branch_only`
- `bridge_only`
- v4 hidden-origin tap direction where config-compatible

The old directions should be treated as preservation anchors. Branch and bridge directions should be added as residuals, not allowed to erase the old objective/coding geometry.

Final-arbiter heads are not merge sources unless they expose a compatible linear hidden-state pair direction. Rank/listwise/metadata final arbiters can be downstream baselines or evaluation targets, but they should not be folded into weight-space merged taps.

## Merge Family

Candidate merged taps:

```text
old_preserving_branch_merge
old_preserving_bridge_merge
coding_reasoning_branch_validity
mixed_objective_branch_validity
old_plus_branch_plus_bridge
domain_gated_merged_weights
```

Basic form:

```text
w_merge = normalize(
    a * w_old
  + b * orthogonal_residual(w_branch, against=w_old)
  + c * orthogonal_residual(w_bridge, against=w_old)
)
```

This keeps the old objective/coding direction as the anchor and adds branch-validity geometry only where it contributes independent signal.

Score distillation is diagnostic-only. The primary experiment is weight extraction, residualization, and direct merged-head evaluation. If score distillation is used to fit a shared compact head from expert/composite scores, it must be reported separately and must not support readiness claims.

Tap scores may be input features or diagnostic teacher scores only where explicitly marked. They are not primary labels. Primary labels remain external reward, correctness, verifier/test result, or already established non-tie outcome labels.

## Split Guard

Before selecting merge coefficients or candidate families, the experiment should inventory task IDs and split metadata from the old-context, hidden-branch, bridge, survival, and final-arbiter artifacts.

Selection rules:

- choose residual weights, normalization mode, and candidate family on train/validation only
- do not select models, thresholds, merge coefficients, or source heads on heldout
- previous heldout observations from Universal Taps, Final Arbiter v1, or Final Arbiter v1.1 are hypotheses unless the split is still clean for this experiment
- any replay on previously inspected heldout data is diagnostic unless the split guard marks it readiness-eligible

## Evaluation Rules

This is still selection/evaluator work only:

- no Ouro training
- no checkpoint/tokenizer edits
- no old tap registry overwrite
- no production routing change
- no action steering claim
- no true fork/carry claim

Labels must remain external reward/correctness/verifier labels. Tap scores are never labels.

Primary comparison:

- old `coding_reasoning`
- old `mixed_objective_all`
- hidden-branch-only
- bridge-only
- universal-balanced
- fixed old+branch+bridge score-level composite
- merged-weight candidates
- oracle upper bound

Required checks:

- old coding/reasoning preservation
- branch-survival retention
- false-prune rate
- top4 oracle coverage
- final-selection reward if used as final arbiter input
- domain breakdown, especially reasoning/science/coding
- config/architecture compatibility
- ablation: old-only vs old+branch residual vs old+bridge residual vs full merge

Replacement language must be precise. A merged tap can replace only a core scoring component unless veto/rescue and missing/OOD fallback behavior are also preserved or re-applied. Full fixed-composite policy replacement should not be claimed from a single vector alone.

## Decision Criterion

The merged tap should not be promoted merely because it is compact. It must either:

- match or beat the fixed-composite core scoring component on top4 survival while preserving old coding/reasoning behavior and compatible guardrails, or
- provide a cleaner final-arbiter feature that improves top4 final selection without hurting coding/math.

If it cannot beat the score-level fixed composite, keep it as an interpretability diagnostic and continue using separated expert scores.

## Recommended Next Experiment

Create a bounded manual experiment:

```text
bg_weight_space_merged_taps_v1
```

Suggested stages:

1. Inventory old/universal/hidden/bridge head state dicts and configs.
2. Extract normalized direction vectors and config metadata.
3. Compute cosine alignment, blockwise layer alignment, and old-vs-branch residual energy.
4. Build config-compatible merged candidate vectors.
5. Evaluate merged taps on cached old-context, hidden-branch, bridge, fixed-composite survival, and top4 final-arbiter datasets.
6. Report whether any merged tap is a viable compact replacement or only a diagnostic.

Expected likely outcome: useful interpretability, a possible additional composite/final-arbiter expert, or confirmation that the separated fixed composite remains best. `MERGED_READY` should be treated as the strongest and least likely outcome; `USE_AS_COMPOSITE_EXPERT` or `SCORE_COMPOSITE_STILL_BEST` are both informative outcomes.
