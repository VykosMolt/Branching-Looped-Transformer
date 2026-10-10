# Run 4: a second Horizon cohort with clean labels

Sealed 2026-10-03, before any generation for this run. `SEAL.sha256` holds the hashes of this file, `answer_detector.py` (byte-identical to Run 1's sealed detector), `gen_run4.py`, `analyze_run4.py`, `test_strict_answer.py`, and the imported Run 3 analysis (`../gsm8k_fixed_prefix_20260927/analyze_run3.py`). The driver and the analysis refuse to run if any of them changed.

## Why, and what informed this design

Post-hoc analyses of Run 1 informed this design; they are stated plainly here.

1. **Difficulty** (`fixed_prefix_rerun_20260927/posthoc_20260927/`). Most of Run 1's +0.243 increase is question difficulty: proof depth and prompt length alone reach 0.686. The across-question gain beyond difficulty and the end-of-prompt state was +0.056 [−0.011, +0.128]. The within-question AUROC was 0.679 [0.602, 0.755].
2. **Labels** (`ouro_project/opi_posthoc_20261003/`).
   - The published Horizon labels come from `parse_mcq_answer` on the whole generation. It returns the *last* "FINAL ANSWER: letter", else the last "answer is X", else the last "X) " / "X. " anywhere.
   - Ouro-RLTT keeps writing after it answers: on GSM8K, 78% continue past the first FINAL ANSWER and 13% invent a new problem. Horizon reasoning also uses capital proposition symbols, which the last fallback matches.
   - Under a strict first-commitment rule, Run 1's primary keeps its estimate (hidden 0.751, increase +0.239 [+0.075, +0.409]), but its held-out set has only 29 incorrect answers.
3. **Run 3** (GSM8K) was not adjudicable: with 448 tokens Ouro-RLTT solves 94% of the problems.

**What this run does:** it maximizes wrong answers per GPU hour (depth 3 and 4 only), labels each answer by its first commitment, and prespecifies the difficulty-aware primary and both controls.

## Data

- **Tasks:** every depth-3 and depth-4 `synthetic_propositional` task that no earlier run used, 1,164 in total (1,097 depth 3, 67 depth 4), in hash order.
  - Excluded: the capped pool's [0, 3000), used by the v2/v3 cohorts, the Thinking runs and Run 1; and Run 2's 180 depth-4 tasks.
  - Task file sha256 `d5503195…`, as pinned by the July audit.
- **Generation:** identical to Run 1.
  - Ouro-RLTT, frozen, all four loops; `mcq_prompt`.
  - k = 4, T = 0.7, top-p 0.95, at most 448 new tokens, FinalAnswerStop.
  - Seed `_sid(20260724, task_uid)`; `split_for(task_uid)` (50/20/30).
- **Saved per answer:**
  - token IDs and sampling log-probabilities;
  - the strict and the published labels;
  - the final-answer-line cut and its features (published protocol);
  - pooled features at N ∈ {0, 16, 32, 48, 64, 96, 128} from one causal forward pass without an attention mask;
  - the state at the last prompt token.
- **Check during generation:** pooled features at N = 128 against a same-length pass with the later tokens replaced (Amendment 2026-10-04) every 25 tasks, max |diff| ≤ 1e-3.

## Labels (primary)

`strict_answer`, with a 13-case self-check in `test_strict_answer.py`:
- the answer is the first "FINAL ANSWER" (any case, optionally inside `**`) followed directly by a capital option letter, optionally after `**`, `(`, `[`, `<` or quotes, or a boxed letter, whichever comes first;
- failing that, the first "FINAL ANSWER" followed by an option word;
- otherwise the answer is unlabelled and excluded.

The published labels are saved and used only in the comparison secondaries.

## Answer detector, N\*

- **Detector:** Run 1's frozen `answer_detector.py`, applied to `decode(gen_ids[:N])` with the task's statement.
- **N\*:** the largest N in {16, …, 128} at which at least 75% of training answers are answer-free. If no N qualifies, N\* = 16.
- **Hand audit:** before labels are opened, 20 answer-free training prefixes at N\* are listed (label-free) and read.

## Primary endpoint

- **Cohort:** labelled answers that are answer-free at N\*.
- **Surface features S:** the mean and minimum sampling log-probability of the first N\* tokens, plus difficulty known before generation: prompt tokens, proof depth, number of variables, facts, rules, and the answer class (True / False / Unknown) one-hot.
- **Combined model:** Run 3's `proper_arm`. Each hidden block is reduced by PCA (k ∈ {16, 24}); S is appended; then L2-logistic regression (C ∈ {0.01, 0.1, 1}). Selection is by 5-fold task-grouped CV on train + val, and the held-out set is opened once.
- **Endpoint:** AUROC(PCA(H at N\*) + S) − AUROC(S), with a 2,000-round paired task-clustered bootstrap.
- **Adjudicability:** at least 30 held-out tasks and 30 held-out incorrect answers.
- **Verdicts:** `HORIZON2_FIXED_PREFIX_INCREMENT_{POSITIVE, NEGATIVE, UNRESOLVED, NOT_ADJUDICABLE}`.

## Prespecified controls and headline rule

1. **End-of-prompt state in the baseline:** PCA(end-of-prompt state) + S; the full model adds PCA(H at N\*).
2. **Within-question AUROC** of the primary held-out scores (combined, hidden-only, surface).

**Headline rule:**
- `READABILITY_BEYOND_DIFFICULTY` if the primary is POSITIVE and control 1's CI is above zero;
- otherwise `MOSTLY_DIFFICULTY`.
- Even a positive result is an upper bound on a signal from the reasoning process.

## Secondary analyses (reported, no verdicts)

- The TF-IDF reader of prompt + N\* tokens with S appended, against the primary combined model (paired).
- The end-of-prompt state alone against difficulty.
- The Run 1-style analysis (S = log-probability only, `arm()`).
- The grid, every N.
- The final-answer line under the published protocol (published labels, the four published shortcuts, `arm()`).
- The final-answer line under strict labels with difficulty in S (`proper_arm`).
- The published-vs-strict label cross-table.
- Any secondary below the minimums is reported as not run.

## Logistics and interpretation

- **Amended 2026-10-03, before any generation:** the run moves from the laptop to Runpod.
  - The model is downloaded from the private HuggingFace repo `Vykos/Ouro-2.6B-RLTT` into the same local path. Every file was verified against the local checkpoint by SHA-256.
  - Generation is split across parallel processes with `--part i --nparts n`; each process takes every n-th shard. Every task keeps its own seed, so the split does not change what is generated.
  - A GPU smoke test runs on the pod first; the run is aborted on any error or failed equivalence check.
  - Then the label-free audit stage runs; the hand audit and the full stage follow as before.
- **Stopping early:** if the run must stop, the analysis uses complete shards and says so.
- **Rules fixed now:** no reruns with other seeds, cuts or thresholds after seeing results. UNRESOLVED is not evidence of absence.

## Amendment 2026-10-04: the equivalence check (before any cohort data)

- **What failed.** The sealed check compared pooled features at N = 128 from the full pass against a separate pass over the truncated input, with a limit of 1e-3. It cannot pass on the cloud GPUs this run uses:
  - A100: 0.020 and 0.040;
  - RTX PRO 6000 Blackwell: 0.030, under both torch 2.8 and torch 2.12.0.dev20260408;
  - the laptop, where the check was designed, gives exactly 0.
- **Why.** The two passes have different lengths, and the GPUs choose different bfloat16 kernels for them, so the arithmetic is rounded differently. A logic error (later tokens reaching the prefix) would give differences of order 1.
- **The replacement check.** The prefix's pooled features at N = min(128, generated length) must be identical when every token after the prefix is replaced by a filler token, with the total length kept the same. Max |diff| ≤ 1e-3, unchanged. It tests the same property, that later tokens do not affect prefix states, without changing the shapes the kernels see. On the RTX PRO 6000 it gave exactly 0.000000 at N = 32, 64 and 128 (4 answers).
- **Approval.** The user was told and left the choice to me on 2026-10-04 ("Do whatever you want with the others"); the threshold was not loosened. Evidence: `../../../cloud/DECISION_equivalence_check.md`.
- **Where.** Code change: the check block in the generator only. Nothing else in the plan changes.
