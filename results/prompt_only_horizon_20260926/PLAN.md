# Prompt-only probe on the Horizon replication set: plan

Written 2026-09-26 before any prompt-only result was computed. Not a sealed preregistration in the project's usual sense (no hash seal), but the design below was fixed before running.

## Question

Do Ouro-RLTT's hidden states over the prompt alone predict whether its sampled answers will be correct, beyond prompt length and proof depth? Those states are captured before any token is generated.

## Why

The July 29 audit (`visible_prefix_hidden_increment_20260729T075247Z`) found that the final-marker cut is answer-contaminated in about 90% of tasks. Prompt-only states are answer-free by construction. The features already exist in the v3 shards (`prompt_only_features`, same canonical extractor, layers 24/36/47 × 4 loops), so this needs no generation and no GPU.

## Data

- **Source:** `opi/preanswer/horizon_power_v3_20260726/horizon_logic/horizon_generation_v3_off{170,340,510}.pt` (read-only). This is the new-only replication cohort: 510 tasks, 2,040 answers.
- **Labels:** parseable answers only, as in the published analysis.
- **Split:** the records' deterministic `split` (train + val used for fitting, heldout for evaluation), asserted equal to `split_for(task_uid)`.

## Gate

Before touching prompt features, reproduce the published `new_only_4sc` arm in `auroc_results_v3.json` to 1e-9 with the same code. Abort on mismatch.

## Method

`arm()` from `bg_v2_malformed_sibling_control.py`, unchanged:
- standardize → PCA → L2-logistic
- grid: k_pca ∈ {16, 24}, l2 ∈ {1, 2, 4}
- 5-fold task-grouped CV on train + val
- heldout opened once
- 2,000-round paired task-clustered bootstrap
- seed 20260724

**Features**
- hidden: `prompt_only_features` flattened (24,576)
- baseline: [prompt token count, proof depth]

## Endpoints

- **Primary:** AUROC(hidden + baseline) − AUROC(baseline) on the new-only heldout, with its 95% CI.
- **Also reported:** AUROC(hidden only) and AUROC(baseline).

## Interpretation, fixed in advance

- **CI excludes zero:** hidden states before generation carry correctness information beyond length and depth. This is at the question level only, because all four answers share the prompt.
- **CI includes zero:** no evidence at this sample size. This is not evidence of absence.

## Limits

- All four answers to a question share their prompt features. The probe can therefore only predict which questions are hard, not which sample succeeds.
- The heldout split has already been used by earlier analyses of this cohort, with different features.

No generation, no GPU, no change to existing artifacts.
