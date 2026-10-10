# Run 6: the pre-answer test on GSM8K training-split problems

Sealed 2026-10-03, before any generation for this run. `SEAL.sha256` holds the hashes of this file, `gen_run6.py` and `analyze_run6.py`, and of the Run 3 files they import unchanged: `analyze_run3.py`, `gen_run3.py` and `answer_detector_gsm8k.py`.

## Why

- **Run 3 was not adjudicable.** On 600 GSM8K test problems with 448 tokens and first-commitment labels, Ouro-RLTT solved 94.2%, which left 21 incorrect held-out answers at N\* = 96.
- **The fix is more problems.** The user approved running this on Runpod (2026-10-03).
- **Contamination does not matter here.** The test asks whether the model's own states predict its own correctness; if training data was seen, that raises accuracy but does not invalidate the test.

## Data

- **Problems:** GSM8K training split (`openai/gsm8k`, revision `740312ad…`, parquet sha256 `ea82612e…`, 7,473 problems), sorted by sha256 of `gsm8k-train/{index}`. The first 2,000 are the cohort; problems from the end of the order are used only for smoke tests.
- **Generation:**
  - Prompt `gsm_prompt` (as Run 3); Ouro-RLTT frozen, all four loops.
  - k = 4, T = 0.7, top-p 0.95, at most 448 new tokens. Seed `_sid(20261003, task_id)`; `split_for(task_id)`.
  - A batch stops once every candidate has written "FINAL ANSWER: number" plus 8 tokens.
  - The model comes from the private HF repo `Vykos/Ouro-2.6B-RLTT`, every file SHA-256-verified against the local checkpoint.
- **Saved per answer:** token IDs, sampling log-probabilities, the first committed answer and its label, pooled features at N ∈ {0, 16, 32, 48, 64, 96, 128, 160, 192, 256} from one causal forward pass without a mask, the end-of-prompt state, and the prefix texts.
- **Check during generation:** pooled features at N = 128 against a same-length pass with the later tokens replaced (Amendment 2026-10-04) every 25 problems, max |diff| ≤ 1e-3.

## Labels, detector, N\*

- **Labels:** the candidate's first committed answer, numerically equal to the reference answer (Run 3's primary label). Candidates with no commitment are excluded.
- **Detector:** Run 3's frozen `answer_detector_gsm8k.py`.
- **N\*:** the largest grid N at which at least 75% of training answers are answer-free.
- **Hand audit:** 20 answer-free training prefixes, label-free, read before labels are opened.

## Primary, controls, headline (as Run 3, Amendment 1)

- **Surface features S:** the mean and minimum log-probability of the first N\* tokens, prompt tokens, and reference-solution steps.
- **Endpoint:** `proper_arm` (PCA per hidden block, then S): AUROC(PCA(H at N\*) + S) − AUROC(S) on held-out answers, with a 2,000-round paired task-clustered bootstrap.
- **Adjudicability:** at least 30 held-out tasks and 30 held-out incorrect answers.
- **Verdicts:** `GSM8K_TRAIN_FIXED_PREFIX_{POSITIVE, NEGATIVE, UNRESOLVED, NOT_ADJUDICABLE}`.
- **Controls:**
  1. the end-of-prompt state in the baseline;
  2. within-question AUROC.
- **Headline rule:** `READABILITY_BEYOND_DIFFICULTY` if the primary is POSITIVE and control 1's CI is above zero; otherwise `MOSTLY_DIFFICULTY`.
- **Secondaries:** the TF-IDF reader with S; the end-of-prompt state alone against difficulty; the Run 1-style analysis; the task level; excluding prefixes that contain the reference number; the grid. Any below the minimums is reported as not run.
- **Expected power** (a rough scaling estimate): 2,000 × 4 × 5.8% wrong × 80% answer-free × 30% held-out ≈ 110 incorrect held-out answers.

## Logistics, interpretation

- **Where it runs:** on Runpod, as parallel processes (`--part i --nparts n`); every problem keeps its own seed.
- **Order:** a GPU smoke test first, with abort on any error or failed check; then generation; then the label-free audit stage on the pod; then the hand audit; then the full stage on the pod.
- **Stopping early:** if stopped early, the analysis uses complete shards and says so.
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
