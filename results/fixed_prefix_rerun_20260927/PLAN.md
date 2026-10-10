# Run 1: fixed-prefix rerun of the Horizon pre-answer test

Sealed 2026-09-27, before any generation for this run. The SHA-256 hashes of this file, `answer_detector.py` and `gen_run1.py` are recorded in `SEAL.sha256`. The analysis script's hash is added below before it is run.

## Why

The July 29 audit (`visible_prefix_hidden_increment_20260729T075247Z`) found that the published final-marker cut is not answer-free. The fixed-prefix repair (`fixed_prefix_horizon_20260729T082827Z`) was not adjudicable, because no generated token IDs had been saved. This run generates a fresh cohort that saves them, and tests the published claim at answer-free cuts.

## Data

- **Model:** `shared/models/ouro_rltt_local` (Ouro-RLTT), frozen.
- **Tasks:** 600 Horizon Logic tasks at offsets [2400, 3000) of the hash-ordered, 4,000-capped pool. Every earlier run used [0, 2400).
- **Sampling:** identical to v2/v3. k = 4, T = 0.7, top-p 0.95, at most 448 new tokens, FinalAnswerStop, seed `_sid(20260724, task_uid)`. Same prompt and parser.
- **Saved per candidate:**
  - prompt and generated token IDs, and the log-probability of every sampled token;
  - published labels (parseable, correct);
  - published features: prompt-only, and the final-marker cut;
  - pooled features at N ∈ {0, 16, 32, 48, 64, 96, 128} generated tokens, from the exact IDs. These come from one causal forward pass, checked against separate passes on a subsample (smoke test: max |diff| = 0).
- **Split:** `split_for(task_uid)`, the published deterministic hash split.
- **Labels:** correctness among parseable answers, as published. Unparseable answers are excluded from labels.

## Answer detector

`answer_detector.py`, frozen now.
- It was designed on the v3 training-split texts only, and no labels were used.
- It flags commitments, not mentions: the final-answer marker followed by an option, "answer/option/choice is X", copula conclusions, "cannot be determined", "therefore … option", boxed answers, bare option lines, and the task's statement (or its negation) after "final answer", "therefore", "conclude that" and similar cues.
- Restated option lists and the empty "FINAL ANSWER:" template are not commitments.
- It is applied to `decode(gen_ids[:N])`, with the task's statement.

## Primary endpoint

- **Choosing N\*** (on training candidates only, without labels): the largest N in {16, 32, 48, 64, 96, 128} at which at least 75% of training candidates are answer-free. If no N qualifies, N\* = 16.
- **Cohort:** candidates that are parseable and answer-free at N\* (candidate level).
- **Features:**
  - H: the pooled features at N\* (24,576 dimensions).
  - S: the mean and minimum log-probability of the first N\* generated tokens. The token count is constant at N\*, so it is omitted.
- **Method:** `arm()` from `bg_v2_malformed_sibling_control.py`, unchanged:
  - standardize → PCA → L2-logistic;
  - k_pca ∈ {16, 24}, l2 ∈ {1, 2, 4};
  - 5-fold task-grouped CV on train + val;
  - held-out opened once;
  - 2,000-round paired task-clustered bootstrap, seed 20260724.
- **Endpoint:** AUROC(H + S) − AUROC(S) on the held-out set, with its 95% CI.
- **Adjudicability:** at least 30 held-out tasks and 30 held-out incorrect answers in the primary cohort.
- **Verdicts** (fixed now):
  - `FIXED_PREFIX_INCREMENT_POSITIVE`: CI above zero.
  - `FIXED_PREFIX_INCREMENT_NEGATIVE`: CI below zero.
  - `FIXED_PREFIX_INCREMENT_UNRESOLVED`: CI includes zero.
  - `FIXED_PREFIX_NOT_ADJUDICABLE`: the minimums fail.

## Secondary analyses (reported, no verdicts)

1. **Task level at N\*:** only tasks all of whose candidates are answer-free at N\*.
2. **Every grid N:** the candidate-level endpoint at each N, as a descriptive curve.
3. **Prompt only (N = 0):** H = features at N = 0; S = [prompt token count, proof depth]. This replicates the v4 prompt-only analysis on fresh data.
4. **Final-marker cut, published protocol:** `preanswer_features` against the four published shortcuts. This replicates the published result on fresh data and does not reinterpret it.
5. **Visible-text reader at N\*:**
   - V1 is TF-IDF (word 1–2 grams plus character 3–5 grams, sublinear tf) with logistic regression.
   - Input: "Prompt:\n{prompt}\n\nVisible reasoning prefix:\n{first N\* tokens}".
   - C ∈ {0.01, 0.1, 1, 10}, chosen by 5-fold task-grouped CV on train + val.
   - Reported: AUROC(V1), and the paired AUROC(H + S) − AUROC(V1) with a task-clustered bootstrap.

## Controls

- Slice disjoint from [0, 2400), asserted.
- `split == split_for(task_uid)`, and no task crosses splits.
- Decode round-trip holds, and equivalence checks are ≤ 1e-3.
- A shuffled-label control at N\*.
- Maximum leave-one-task-out influence.
- The detector hash is unchanged.
- Answer-free fractions are reported by split and N.
- **Hand audit:** before any held-out label is opened, 20 deterministic random answer-free training prefixes at N\* are listed and inspected, without labels.

## Interpretation, fixed now

- **POSITIVE:** hidden states predict eventual correctness from answer-free prefixes beyond surface features. This is conditional on the answer not having committed within N\* tokens.
- **UNRESOLVED:** no evidence at this sample size. It is not evidence of absence.
- No reruns with other seeds, cuts or thresholds after seeing results.
