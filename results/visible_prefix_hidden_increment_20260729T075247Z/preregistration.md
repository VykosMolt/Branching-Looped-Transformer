# Preregistration: visible-prefix versus hidden-state conference gate

Sealed UTC date: 2026-07-29  
Experiment ID: `visible_prefix_hidden_increment_20260729T075247Z`  
Scope: zero-generation analysis of preserved Ouro strict-preanswer artifacts.  
Smallest effect of interest: ±0.02 AUROC for `AUROC(HV) - AUROC(V)`.

This document was written after the label-blind signed inventory and before any
label-dependent head reproduction, classifier fitting, leakage adjudication, or
held-out metric computation. The current arXiv manuscript and its tarball are
outside scope and must not be modified.

## Endpoint hierarchy and estimands

The designated primary endpoint is Horizon Logic prospectively generated
new-only `AUROC(HV) - AUROC(V)`. GSM8K is the key replication endpoint.
Horizon Logic pooled is the secondary precision endpoint. No multiplicity
correction is applied to the single designated primary endpoint.

`S` is the strictly pre-cut shortcut vector:

1. visible-prefix token count (`n_pre_tok`);
2. mean sampled-token log probability over strictly pre-cut tokens;
3. minimum sampled-token log probability over strictly pre-cut tokens.

All three are standardized using training data only. `hit_max_tokens`,
malformedness, parsed-answer fields, final-marker presence, verifier fields,
post-cut text, and future-token statistics are excluded. No additional source
ID, file position, difficulty, or task-identity feature enters `S`.

`T` is the frozen score of the validation-selected visible-only family. `R` is
the raw hidden-only correctness score reconstructed from the original frozen
hidden feature tensor and deterministic original head specification, before any
shortcut fusion.

The nested models are regularized logistic regressions:

- `V = logistic([S, T])`
- `H = logistic([S, R])`
- `HV = logistic([S, T, R])`

No already shortcut-fused hidden score may be used as `R`.

## Inventory-gate decisions

The signed inventory digest is recorded in the sealed protocol manifest.

- Horizon new-only: exact rendered trajectories, stored strict-cut token
  indices, source tasks, tokenizer revision, original task-disjoint splits,
  labels, task IDs, allowed shortcut fields, frozen hidden features, and
  published held-out hidden scores exist. Raw hidden-only scores exist only for
  held-out rows. The deterministic original head implementation, grid, seed,
  frozen features, and published scores permit a reproduction attempt.
- Horizon pooled: same status, using the frozen v2 shard plus the three
  prospective v3 shards.
- GSM8K: hidden features and historical grouped-OOF predictions exist, but
  generated token IDs, rendered candidate trajectories, visible-prefix text,
  and an original train/validation/held-out partition do not. GSM8K is fixed as
  `VISIBLE_PREFIX_NOT_ADJUDICABLE`. No replacement split, approximated prefix,
  regenerated output, or inferred text is allowed.

For Horizon, before visible-model fitting, the original hidden-only head must be
reproduced without held-out fitting. Its held-out scores must agree with the
preserved published scores with maximum absolute probability difference no
greater than `1e-8`, and held-out task order and labels must match exactly.
Failure marks Horizon HV `NOT_ADJUDICABLE`; an H-versus-V fallback is permitted
only if a valid H score and V remain available.

## Source cohorts and immutable splits

Horizon new-only consists only of offsets 170, 340, and 510: 510 tasks and
2,040 preserved candidates before the original malformed-row disposition. Its
task counts are train 266, validation 99, held-out 145. Its sorted
`task_id -> split` digest is
`395e9694b93defe707afd64fa7982986359a8a93865b4ec0d9ff5d6aaa916a55`.

Horizon pooled consists of the frozen v2 offset-0 shard and all three v3
shards: 680 tasks and 2,720 preserved candidates before the original
malformed-row disposition. Its task counts are train 346, validation 130,
held-out 204. Its split digest is
`cfc43af83f0effe677be8b92fb74f3de9bf86dcdada5c15f367d16e0a7dfc906`.

The original Horizon correctness analysis excluded malformed/unverifiable
candidates from the AUROC cohort. That candidate set is preserved as the
historical cohort; malformedness is never a model feature. The audit will
report this selection limitation explicitly.

## Exact prefix reconstruction

The original generation tokenizer is `ByteDance/Ouro-2.6B` revision
`1ed04250da1a9936042725d302e81c8fa2ab5abd`, locally preserved at
`models/ouro_rltt_local`; `tokenizer.json` SHA-256 is
`fcb808fe5e7642f5299be28aea07fc7f6d4f4364c3ac5e408e15a772cbc8fa8d`.

For Horizon, the original prompt is reconstructed byte-for-byte from the
hash-frozen logic source row and the hash-frozen `mcq_prompt` formatter. The
visible prefix is reconstructed from the exact preserved rendered trajectory
by replaying the original strict cut: text strictly before the first
case-insensitive `FINAL\s*ANSWE` match, or the complete rendered trajectory
when no marker exists. This is the original stored cut implementation, not a
new heuristic. The stored `n_pre_tok`, `found_final_marker`, original tokenizer,
prompt token count, and trajectory hash are audited row by row. No span may be
censored.

The classifier input format is exactly:

```text
Prompt:
{original prompt}

Visible reasoning prefix:
{tokens/text strictly before the answer-region cut}
```

## Leakage rules and disposition

The audit checks the visible prefix (not the multiple-choice option list in the
prompt) for:

- case-insensitive literal gold-answer text on token boundaries;
- normalized alphanumeric equivalence;
- explicit symbolic answer equivalence such as `answer/option/choice is C`;
- a forbidden `FINAL\s*ANSWE` marker;
- exact and near-duplicate prompts across task splits;
- task crossing;
- post-cut, parser, malformedness, verifier, future-token, source-ID, and file
  position shortcuts;
- preprocessing fit outside training;
- held-out-informed selection;
- in-sample base scores in fusion;
- hidden features in visible-only models;
- labels entering text preprocessing.

If any candidate in a task contains a genuine gold-answer occurrence, the
entire task is excluded from the leak-free primary sensitivity. The unchanged
cohort is retained only as a historical sensitivity. Individual text spans are
never censored. Exclusions materially change the estimand, and the strict
endpoint becomes `VISIBLE_PREFIX_NOT_ADJUDICABLE`, if more than 20% of tasks
overall or in any split are removed, or if fewer than 50 train, 20 validation,
or 30 held-out tasks remain. Any unrecoverable exact-prefix reconstruction
error or forbidden marker under the reconstructed cut also makes the arm not
adjudicable.

Exact duplicate prompts belonging to distinct tasks are treated taskwise in
the same manner. Near duplicates are defined by character 3–5 gram TF-IDF
cosine similarity at least 0.95; they are reported and receive a secondary
connected-component exclusion sensitivity, but do not alone trigger the hard
gate because synthetic task templates can be legitimately similar.

## Frozen visible encoder and truncation

V2 and V3 use `microsoft/deberta-v3-small` at immutable revision
`a36c739020e01763fe789b4b85e2df55d6180012`, with its same-revision
SentencePiece tokenizer. No model substitution is allowed. V2 and V3 are
`NOT_ADJUDICABLE` if this revision cannot be loaded. The maximum input length is
512 tokens.

The pair input uses 256 prompt tokens and 253 prefix tokens plus three special
tokens. Prompt overflow is truncated after the first 256 prompt tokens. Prefix
overflow keeps the most recent 253 prefix tokens. Capacity is fixed and is not
silently reallocated. The end of the visible reasoning prefix is never
right-truncated. Counts retained and truncation status are recorded per row.
V1 uses the complete exact text without encoder truncation.

V2 pooling is the attention-mask-weighted mean of the final encoder hidden
state, computed in float32 on CPU. Encoder weights remain frozen. Only an
L2-regularized linear logistic head is fitted.

## Visible baseline ladder and fixed grids

All text preprocessing and learned parameters are fit on train only during
validation selection. Selection is performed independently for Horizon
new-only and Horizon pooled because they are separately named arms with
different preserved training cohorts. No selection uses held-out labels or
metrics.

V0 is an L2-logistic model over `S`, with `C` in
`[0.01, 0.1, 1.0, 10.0, 100.0]`.

V1 combines, with equal feature-union weight:

- lowercase word 1–2 gram TF-IDF, Unicode accent stripping, sublinear term
  frequency, maximum 40,000 features;
- lowercase character-within-word-boundary 3–5 gram TF-IDF, sublinear term
  frequency, maximum 60,000 features.

`min_df` is `[1, 2]`; logistic `C` is
`[0.01, 0.1, 1.0, 10.0]`; maximum iterations are 2,000; solver is liblinear;
random seed is 20260729.

V2 uses the frozen mean-pooled embedding and L2 logistic regression with `C` in
`[0.01, 0.1, 1.0, 10.0, 100.0]`, liblinear, 2,000 iterations, seed 20260729.

V3 fine-tunes the same encoder and a binary classification head. The grid is:

- learning rate `[1e-5, 2e-5]`;
- weight decay `[0.01]`;
- batch size `[8]`;
- maximum epochs `3`;
- AdamW betas `(0.9, 0.999)`, epsilon `1e-8`;
- linear warmup fraction `0.1`;
- gradient norm cap `1.0`;
- seeds `[20260729, 20260730, 20260731]`;
- validation-AUROC early stopping with patience one epoch.

Hyperparameters are chosen by mean validation AUROC across the three seeds.
The family score is the mean probability across all three seeds at the selected
hyperparameters; no seed is discarded. All seed-level held-out results are
reported. If main-arm V3 cannot run on the available CPU-only environment, it
is a preregistered execution failure and no easier family is substituted.

Prompt-only and prefix-only controls are run for V1 and V2. V3 prompt-only and
prefix-only controls use the main selected hyperparameters if main V3 completes
within the fixed 12-hour per-arm resource ceiling; otherwise they are reported
as secondary execution failures and cannot affect selection.

T is chosen among adjudicable V1, V2, and V3 by maximum validation AUROC.
V3's validation value is its three-seed ensemble AUROC. Exact ties within
`1e-12` are broken in order V3, V2, V1, then by lexicographically smallest
serialized hyperparameter tuple.

## Hidden readout reproduction and OOF fusion

The original Horizon hidden-only head is standardized hidden features followed
by PCA and L2 logistic regression. The preserved original grid is PCA
components `[16, 24]` and L2 penalty `[1.0, 2.0, 4.0]`, with five task-grouped
folds and seed 20260724. The deterministic original selection/reproduction is
used solely to recover the frozen `R` construction and must pass the published
held-out-score gate.

For higher-level fusion, every learned train/validation base score is
task-grouped out-of-fold. The selected T architecture/hyperparameters and R
architecture/hyperparameters are fixed first. Five deterministic task-grouped
folds are used. No task is scored by a base model trained on that task.
Validation family selection uses a base model trained on train only. After
selection, cross-fitted train+validation T and R scores are created for fusion.

Fusion regularization `C` is selected from
`[0.01, 0.1, 1.0, 10.0, 100.0]`: train OOF rows fit the meta-model and
validation cross-fitted rows select `C` by AUROC. Ties use the smaller `C`.
The final meta-model is fit on all train+validation OOF base scores. Allowed
base components are refit on full train+validation only after all choices are
frozen, and held-out is evaluated once.

## Metrics and statistical procedures

The primary statistic is paired `AUROC(HV) - AUROC(V)` on the unchanged held-out
row pairing. A paired task-clustered percentile bootstrap resamples task IDs
with replacement, retaining every candidate row for each sampled task.
Ten thousand replicates use seed 20260729. Degenerate single-class replicates
are skipped and their count reported. Ninety-five percent percentile intervals
support superiority/direction verdicts.

Equivalence uses a paired task-clustered bootstrap TOST at alpha 0.05:
equivalence passes only when the 90% percentile interval for `HV - V` lies
strictly inside `[-0.02, 0.02]`. Failure to establish superiority is not
equivalence.

Secondary outputs are AUROC for V0/V1/V2/V3/V/H/HV; H−V and HV−V; prompt-only,
prefix-only, and prompt-plus-prefix comparisons; seed-level V3 results; Brier
score, log loss, equal-frequency 10-bin ECE; risk–coverage curves at all unique
candidate coverages; prefix-length quartiles; truncation status; each S feature
quartile; proof depth; source (pooled only); class balance; per-task AUROC where
both classes exist; leave-one-proof-depth-out and leave-one-source-out
influence; task leave-one-out endpoint influence; and 1,000 deterministic
held-out label permutations. Calibration and decision thresholds do not affect
AUROC selection.

Hand audit samples are deterministic SHA-256 ranks under seed 20260729: up to
five positives, five negatives, five malformed/structurally unusual rows, five
truncated rows, and five rows from each available extreme category (visible
margin, hidden margin, H–V disagreement, HV gain, leakage, exact duplicate, and
near duplicate). All selected rows and the rule are recorded.

## Verdict logic

Per-arm and aggregate labels are exactly those in the mission specification.
Superiority requires the lower 95% paired clustered interval above zero.
Visible advantage requires the upper interval below zero. Equivalence requires
the clustered TOST above. Otherwise the arm is unresolved. Fallback H−V labels
are used only when HV is unavailable but H and V are valid, and are never
described as conditional hidden information.

The aggregate is `VISIBLE_PREFIX_AGGREGATE_NOT_ADJUDICABLE` whenever Horizon
new-only is not adjudicable. Mixed adjudicable domain results are never promoted
to a universal conclusion.

## Stopping and failure rules

- No new Ouro generations are permitted.
- Missing exact prefixes, original partitions, or valid hidden reproduction
  stops the affected endpoint.
- Leakage materiality uses only the thresholds above; no post-result threshold
  change is permitted.
- A missing fixed visible encoder revision marks V2/V3 not adjudicable; no
  substitute is allowed.
- A V3 resource failure is recorded before any held-out V3 evaluation and does
  not authorize substitution.
- Held-out labels and metrics remain unopened until this document, source/split
  manifests, model choices, grids, seeds, statistics, and selection rules are
  hash-sealed.
- No manuscript or arXiv tarball file is written.

## Reproduction

The sealed entry point is:

```bash
bash utilities/tests/manual/reproduce_visible_prefix_hidden_increment.sh \
  artifacts/reports/visible_prefix_hidden_increment_20260729T075247Z
```

The sealed protocol manifest records the exact wrapper and implementation
hashes. The final checksum manifest covers every produced output.
