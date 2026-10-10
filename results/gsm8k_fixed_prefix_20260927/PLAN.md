# Run 3: fixed-prefix test of the pre-answer claim on GSM8K

First sealed 2026-09-27 12:28 (hashes kept in `SEAL_v1.sha256`). **Amendment 1** below was frozen on 2026-09-27 at 16:48 CEST, before any cohort data was generated; the only generations so far are smoke tests on problems outside the cohort. `SEAL.sha256` now holds the hashes of this file, `answer_detector_gsm8k.py`, `gen_run3.py` and `analyze_run3.py` as amended. The driver and the analysis refuse to run if any of them has changed.

## Amendment 1 (supersedes the primary endpoint below)

**The Horizon post-hoc analysis informed this amendment.** After Run 1's sealed result, a post-hoc analysis of Run 1's data (`../fixed_prefix_rerun_20260927/posthoc_20260927/README.md`) found three things:

- **Weak baseline.** At a fixed cut, the sealed baseline was token confidence only, at AUROC 0.515.
- **Difficulty.** Proof depth plus prompt length alone reached 0.686. Beyond them the hidden states added +0.087 [+0.005, +0.171].
- **Swamping.** The sealed combined model (`arm()`, PCA over hidden and surface features together) cannot use the surface features: "both" always equalled "hidden".

The amendment answers the question that analysis raised, and was written before any GSM8K cohort data existed.

- **Primary endpoint (replaces the one below).**
  - Surface features S: the mean and minimum sampling log-probability of the first N\* tokens, plus prompt tokens and reference-solution steps. The last two are difficulty known before generation; reference steps play the role of Horizon's proof depth.
  - Combined model: each hidden block is standardized and reduced by PCA (k ∈ {16, 24}); S is appended; everything is standardized; then L2-logistic regression (C ∈ {0.01, 0.1, 1}). Hyperparameters are chosen by 5-fold task-grouped CV on train + val, and the held-out set is opened once.
  - Endpoint: AUROC(PCA(H) + S) − AUROC(S), with a 2,000-round paired task-clustered bootstrap, seed 20260724.
  - Unchanged: N\*, the cohort, the detector, the adjudicability minimums and the verdict labels.
- **Prespecified controls.**
  1. **End-of-prompt state.** The baseline adds PCA of the hidden state at the last prompt token (3 layers × 4 loops), taken before any generated token: the model's own reading of the question. The full model adds PCA(H at N\*) on top. Reported: the increase and its CI.
  2. **Within-question AUROC.** The share of correctly ranked (correct, incorrect) pairs among samples of the same question, where difficulty is constant by construction. It is computed on the primary held-out scores (combined, hidden-only, baseline), with a bootstrap over questions that have mixed outcomes.
  3. **The published cut and the own-answer cut** with reference steps added to the published shortcuts, using the fixed combined model.
- **Headline rule, fixed now:**
  - `READABILITY_BEYOND_DIFFICULTY` if the primary is POSITIVE and control 1's CI is above zero;
  - otherwise `MOSTLY_DIFFICULTY`, meaning pre-answer readability on GSM8K is mostly question difficulty.
  - Control 2 is reported alongside.
  - Even a positive result is an upper bound on a signal from the reasoning process, because generator metadata and the end-of-prompt state may not capture all of the model's own sense of difficulty.
- **Changed secondaries.**
  - The visible-text reader gets the same S appended, and is compared with the primary combined model.
  - The grid, task-level, extended-surface and excluding-reference-number analyses use the fixed combined model with difficulty in S.
  - The prompt-only secondary uses the end-of-prompt state against prompt tokens + reference steps.
  - The original sealed analysis (S = log-probability only, `arm()`) is kept as a secondary for comparison with Horizon Run 1.
- **Also saved per candidate:** the end-of-prompt state.
- **Not applicable to GSM8K:** the verdict tests planned for Horizon. Numeric answers have no verdict classes.

---

The original sealed plan follows. Where it conflicts with Amendment 1, the amendment governs.

## Why

- **The published result.** GSM8K, +0.066 [0.021, 0.112] on 170 problems, from `proto_introspection_within_domain_recapture.py`, June 17.
- **Its cut depended on the label.** The cut was the FINAL ANSWER marker or the first standalone appearance of the reference number, whichever came first. A correct solution always contains the reference number; a wrong one only by chance. So the cut position depends on the label.
- **It cannot be re-analysed.** The published run kept no texts or token IDs, and its feature file (`within_domain_recapture.pt`) is no longer on disk.
- **What this run does.** Run 1 repaired the same claim on Horizon Logic with an answer-free fixed-prefix cut. This run applies that protocol to GSM8K. It also replicates the published protocol on the same fresh problems, to measure how much the reference-dependent cut mattered.

## Data

- **Model:** `shared/models/ouro_rltt_local` (Ouro-RLTT), frozen, all four loops, as published.
- **Problems:**
  - Source: GSM8K test, `elastic_reasoner/data/gsm8k/main/test-00000-of-00001.parquet` (sha256 `ee7b8da9…4f59`). It has 1,319 rows, row 0 is "Janet's ducks" and all references are integers.
  - Cohort: test problems 170–1,318 (the published run used 0–169), sorted by the sha256 of `gsm8k/{index}`. The first 600 are the cohort. Problems after them are used only for smoke tests.
- **Prompt and sampling:** `gsm_prompt`, as published. k = 4, T = 0.7, top-p 0.95, as published. Seed `_sid(20260927, task_id)`.
- **Budget:**
  - At most 448 new tokens. A batch stops only after step 224, and once every candidate has written "FINAL ANSWER: <number>" followed by 8 more tokens.
  - The published run used 224 tokens and no stop. The first 224 tokens here are therefore exactly what the published protocol would see.
  - Ouro-RLTT does not emit EOS (0 of 396 candidates in Run 1), so without a stop every batch would run to the limit.
- **Saved per candidate:**
  - prompt and generated token IDs;
  - the sampling log-probability of each token, and its teacher-forced log-probability and entropy;
  - primary labels;
  - the published-protocol labels, cut, features and shortcut inputs;
  - the own-answer cut and its features;
  - pooled features at N ∈ {0, 16, 32, 48, 64, 96, 128, 160, 192, 256} generated tokens, from one causal forward pass.
- **Checks during generation** (every 25 problems):
  - pooled features at N = 128 against a separate forward pass, max |diff| ≤ 1e-3;
  - teacher-forced statistics over the first max(published cut, 128) tokens against the published `logprob_entropy`, max |diff| ≤ 1e-3.
- **Why the checks use long passes and no mask** (measured in the GPU smoke test, before sealing):
  - On the GPU, a separate pass shorter than about 128 tokens uses a different bf16 attention kernel. At N = 32 this changes the pooled features by up to 0.044 (0.6% of the largest feature). The same code in float32 on the CPU gives 2e-6, and bf16 passes of 159 tokens or more agree exactly. So the check runs at N = 128, and the N = 32 gap is only recorded.
  - The forward pass omits the attention mask, as the published `logprob_entropy` does. Without a mask it reproduces that function to 1e-8; a mask of ones selects another kernel and moves the statistics by up to 0.006.
- **Split:** `split_for(task_id)`, the 50/20/30 hash split used by Run 1.

## Labels

- **Primary.** The candidate's answer is its first committed answer in the whole generation (`first_commitment`): the first "FINAL ANSWER: n", else the first "#### n", else "answer is n", else a boxed number. It is correct if it equals the reference numerically. A candidate that never commits to a number is unparseable and excluded from labels, as in Run 1.
- **Published protocol** (secondary analyses only). `evaluate_output` on the first 224 generated tokens: the last FINAL ANSWER, else the published fallbacks, else the last number. Unparseable counts as incorrect, as published.

## Answer detector

`answer_detector_gsm8k.py` is frozen now. It was written without any Run 3 data; its self-check has 29 cases.

A prefix is answer-bearing if it contains:
- an explicit answer statement with a number: "FINAL ANSWER: n", "#### n", "answer is n", "Answer: n", or a boxed number;
- the candidate's own answer as a standalone number anywhere (the own answer, never the reference answer);
- a number cut off at the end of the prefix whose digits so far begin the own answer.

Not commitments: the empty "FINAL ANSWER: <number>" template, and numbers other than the own answer (intermediate results).

The detector uses only the candidate's own output, never its label.

## Primary endpoint (as in Run 1)

- **Choosing N\*** (without labels): the largest N in {16, …, 256} at which at least 75% of training candidates are answer-free. If no N qualifies, N\* = 16.
- **Cohort:** parseable candidates that are answer-free at N\* (candidate level).
- **Features:**
  - H: the pooled features at N\* (24,576 dimensions).
  - S: the mean and minimum sampling log-probability of the first N\* tokens.
- **Method:** `arm()` from `bg_v2_malformed_sibling_control.py`, unchanged:
  - standardize → PCA → L2-logistic;
  - k_pca ∈ {16, 24}, l2 ∈ {1, 2, 4};
  - 5-fold task-grouped CV on train + val;
  - held-out opened once;
  - 2,000-round paired task-clustered bootstrap, seed 20260724.
- **Endpoint:** AUROC(H + S) − AUROC(S) on the held-out set, with its 95% CI.
- **Adjudicability:** at least 30 held-out tasks and 30 held-out incorrect answers.
- **Verdicts:**
  - `GSM8K_FIXED_PREFIX_INCREMENT_POSITIVE`: CI above zero.
  - `GSM8K_FIXED_PREFIX_INCREMENT_NEGATIVE`: CI below zero.
  - `GSM8K_FIXED_PREFIX_INCREMENT_UNRESOLVED`: CI includes zero.
  - `GSM8K_FIXED_PREFIX_NOT_ADJUDICABLE`: the minimums fail.

## Secondary analyses (reported, no verdicts)

Any analysis below the adjudicability minimums (fewer than 30 held-out tasks or 30 held-out incorrect answers) is reported as not run. This is likely at the largest grid N, where few answers are still answer-free.

1. **Visible-text reader at N\*.** TF-IDF, as in Run 1. Reported: its AUROC, and the paired AUROC(H + S) − AUROC(reader).
2. **Task level at N\*.**
3. **Every grid N.**
4. **Prompt only (N = 0).** S = [prompt tokens, number of reference-solution steps].
5. **Extended surface at N\*.** S plus the teacher-forced mean and last entropy of the tokens read.
6. **Excluding the reference number.** The primary cohort, additionally excluding prefixes that contain the reference number. This checks one asymmetry of the detector: the prefix of a correct candidate can never contain the reference number, while that of a wrong candidate can.
7. **Published protocol on fresh problems:**
   - the published cut, features (prompt + "\n" + the stripped text before the cut, re-tokenized), and labels (224 tokens);
   - the published shortcuts: question characters, prompt tokens, tokens before the cut, and the teacher-forced mean log-probability, mean entropy and last entropy;
   - run through `arm()`, and through the published estimator (out-of-fold, 5-fold task-grouped, shortcuts k_pca = 6 and l2 = 1, combined k_pca = 24 and l2 = 2).
8. **Own-answer cut.** The published cut with the candidate's own 224-token answer in place of the reference answer. Features, shortcuts and labels are the same as in 7, on the same rows. Reported: the published-cut increase minus the own-answer-cut increase, with a task-clustered bootstrap. This measures how much the reference-dependent cut inflated the published estimate.
9. **Published cut position by label:** marker, reference number or neither, crossed with correct or incorrect.
10. **Published label against primary label:** the published 224-token label crossed with the primary label, which shows how often the 224-token budget marked a candidate incorrect because its text was cut off before the answer.

## Controls

- The cohort is disjoint from problems 0–169 (asserted).
- `split == split_for(task_id)`.
- The decode round-trip holds, and the equivalence and teacher-forced checks pass.
- A shuffled-label control at N\*.
- Maximum leave-one-task-out influence.
- The seal check.
- Answer-free fractions are reported by split and N, together with the parseable and token-limit fractions.
- **Hand audit:** before labels are opened, 20 deterministic random answer-free training prefixes at N\* are listed with the problem and the candidate's own answer, and read.

## Logistics

- **Estimated time:** about 6 h on the GPU. The GPU smoke test took 37 s per problem over 3 problems, with a check on every problem; two of the three batches ran past 400 tokens, because the model writes long solutions.
- `run3.sh` refuses to start if another process is on the GPU, or if the seal check fails.
- **Stopping early:** if the run must stop early for GPU time, the analysis uses the complete shards of 100 problems, and this is reported. No result is computed until generation ends, so stopping cannot depend on results.

## Interpretation, fixed now

- **POSITIVE:** on GSM8K, hidden states predict eventual correctness from prefixes that have not stated an answer, beyond surface features.
- **UNRESOLVED:** no evidence at this sample size. It is not evidence of absence.
- **Own-answer comparison:** descriptive. A published-minus-own difference above zero means the reference-dependent cut inflated the published estimate.
- No reruns with other seeds, cuts or thresholds after seeing results.
