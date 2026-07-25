# Cross-loop early-layer tap experiment (2026-07-20)

**Question.** Does process-quality information become readable at earlier physical layers
(8, 16) during later recurrent loop iterations (L2–L4) of frozen Ouro-RLTT, and does a tap
direction trained at one loop transfer unchanged to later loops of the same physical layer?

**Run root:** `artifacts/reports/cross_loop_early_layer_taps_20260720/`
**Budget:** 8 h wall clock, hard stop. Started 2026-07-20 17:27 CEST (unix in `logs/start_time`).
Readout-only: no steering, no backbone training, no paper edits, no artifact overwrites.

## Primary target (preregistered)

CoreContent v2 task-disjoint candidate-quality ranking (corrected/deduped dataset,
`data/corecontent_v2/processed/candidate_groups_deduped.jsonl`), 5 core domains
(coding, reasoning, math, logic, alignment). Split = the existing deterministic
task/prompt-disjoint `split_for` assignment stored per group (train/val/heldout). No new
split is constructed. Hard assertion: zero task_uid crossing between subset splits.

### Bounded subset (extraction budget)

Deterministic sample by SHA-ordered `group_uid` (`stable_int("xloop_early_v1", group_uid)`),
reward-diverse groups only, candidates taken from the cached v2 feature shards (identical
candidate set + order; texts joined by `candidate_uid`):

- train: 250 groups/domain
- val: 60 groups/domain (hyperparameter-grid selection only)
- heldout: 120 groups/domain (fixed final evaluation set, identical across all cells)

### Features

One frozen forward per candidate (`use_cache=False`, `total_ut_steps=4`,
`early_exit_threshold=1.0`), capturing per loop u∈{1..4}:

- layers 8, 16, 24, 36: post-block hidden of `model.model.layers[idx]`, idx = layer−1
  (same convention as the production extractor's 24→23, 36→35);
- layer 47: per-loop boundary states (`output[1]` of the inner model) — the production
  convention for "layer 47".

Pooling: attention-mask-valid mean over all tokens, fp32 (identical to
`BGTransformerFeatureExtractor._masked_mean_pool`); per-domain `max_length` identical to v2
(`alignment 768, coding 1024, math 768, logic 640, reasoning 640`). Text encoded is
`candidate_text` as-is (v2 convention). Stored per candidate: `[layers=5, loops=4, 2048]` fp16.
This is the single preregistered pooling convention; no variants are analyzed.

### Tap and training (identical at every cell)

`AntisymLinearNoNorm`: bias-free linear score on candidate-difference vectors; exact
antisymmetry by construction. Training = v2 `train_pairwise` verbatim: pairwise diffs
(reward_i > reward_j, ≤24 pairs/group), Adam lr, weight_decay 0.01, 60 epochs, grad-clip 1.0,
fixed torch seed. Grid (identical at every cell, selected on **val** macro top-1 only, never
heldout): training_set ∈ {all_core, all_core_balanced, code_reasoning, math_logic,
alignment_only, code_math_logic} × lr ∈ {3e-4, 1e-3} × seed ∈ {0,1,2}.

Scoring/metrics = the locked v2 scoring layer, reimplemented for the 5-layer tensor and
verified against `cc.policy_candidate_scores`/`group_selection_metrics` in tests:
per-candidate mean pairwise margin, z-normalized within group; metrics = macro top-1 oracle,
pairwise accuracy over differing-reward pairs, regret.

## Matrix

**Local refit (Mode B):** fresh tap per cell — layers {8,16,24} × loops {L1..L4} = 12 cells,
plus references L4_36, L4_47.

**Frozen transplant (Mode A):** the selected local-refit tap at source (u, l) evaluated
unchanged (weights, feature convention; no bias/normalization/threshold exist by
construction) at target loops v > u of the same layer: 6 transfers per layer × 3 layers.
No cross-layer transfer in the primary analysis. Reported per transfer: absolute target
performance, delta vs source self-score, delta vs target-local refit, score correlation
(Pearson + Spearman) on identical heldout candidates, pairwise sign agreement, mean
per-group rank correlation.

## Controls (all hard-asserted or reported)

1. Zero task_uid crossing between splits (hard fail).
2. Candidate/label alignment across loops is structural (one tensor per candidate);
   asserted by shape + UID checks per cell.
3. Feature-match: fresh 24/36/47 features vs cached v2 shard features on the heldout subset
   (cosine, max|Δ|) — validates the new extraction path against the production one.
4. Reference reproduction: the saved locked policy `corecontent_v2_policy.pt` (24_L4+36_L4)
   evaluated on the same heldout subset with cached vs fresh features (matched-subset
   comparison; full-population stored value 0.6691 quoted as context only).
5. Label-shuffle: rewards shuffled within train+val groups (fixed seed), full grid rerun at
   representative cells {8_L1, 8_L4, 16_L2, 24_L4}; expect ≈ matched-random.
6. Source self-reproduction: every transplanted tap must reproduce its stored source-cell
   heldout score exactly before target evaluation (hard fail).
7. Frozen-head immutability: weight hash asserted identical before/after transplant eval.
8. Matched-random baseline: analytic tie-aware chance E[top1] = (#reward-max candidates)/n
   per group, plus the deterministic RANDOM policy.
9. No target-label tuning: grid selection uses val only (asserted by construction; test
   checks the selection function never receives heldout groups).

## Statistics

Task-clustered bootstrap, clusters = `task_uid` within domain, macro = unweighted mean of
domain means; 10,000 draws, seed 20260720. Paired deltas: identical draws applied to both
members of each comparison. Raw per-group predictions and scores preserved under
`predictions/`.

## Predeclared interpretation labels

As specified in the task brief: EARLY_READABLE (clustered CI excludes matched chance AND
within 0.03 macro top-1 of the L4_24 local refit or better), DIRECTION_STABLE (above chance,
within 0.03 of target-local refit, positive nontrivial score/rank correlation — actual
correlation reported), ROTATED_BUT_READABLE, LATE_EMERGENT, EARLY_ACTION_SURFACE_CANDIDATE
(readout-only wording), plus the run-level verdict vocabulary from the brief. All continuous
results are reported regardless of labels.

## Degradation plan

Tier 1: layers 16+24, all loops, both modes, L4_36 reference. Tier 2: layer 8. Tier 3:
L4_47 reference + secondary readout (S3B2) only if time permits. Sacrifice order per brief.
At T+7:15 no new model inference; T+8:00 full stop.

## Secondary readout (only if primary complete and time remains)

S3B2 generated-branch correctness on its existing task-grouped source-disjoint split —
explicitly small-n; AUROC + pairwise accuracy + clustered uncertainty. Will be dropped
first under time pressure.
