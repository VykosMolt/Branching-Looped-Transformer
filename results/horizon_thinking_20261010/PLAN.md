# Run 8: the answer-free test on a second checkpoint, Ouro-2.6B-Thinking

Prepared 2026-10-10. **Not sealed.** `seal_run8.sh` writes `SEAL.sha256` with the hashes of:
- this file, `gen_run8.py`, `analyze_run8.py`, `modeling_ouro_shim.py` and `test_run8.py`;
- the files the analysis imports unchanged: Run 4's `answer_detector.py`, Run 7's `within_oof.py`, and Run 3's `analyze_run3.py`, `gen_run3.py` and `answer_detector_gsm8k.py` (`analyze_run3` imports the last two).

The analysis refuses to run if any of them changed. No generation for this run exists.

**Notes added 2026-10-18, before sealing.** Run 7 was dropped by the user before it was sealed or run; its `within_oof.py` is used here unchanged and is sealed in this file list by its own hash. The token limit stays at 448, identical to Run 4: a 768-token variant was considered for power but was not affordable with margin (balance $12.34 against an estimated $9.5–13), so this run accepts the UNDERPOWERED risk described below.

## Why

- **Every answer-free result so far is on Ouro-RLTT only.** That covers Run 1, Run 4 and the post-hoc within-question analyses (Run 4: held-out 0.740 [0.636, 0.831]; out of fold 0.709 [0.646, 0.767] on 130 questions).
- **The earlier Thinking tests are not comparable.** The July Thinking runs (`thinking_preanswer_v3_20260726`, 170 questions; `thinking_preanswer_power_v5_20260727`, 900 questions at pool offsets [1500, 2400)) used the final-answer-line cut and the published labels. The larger one was inconclusive (+0.042 [−0.017, +0.104]).
- **What this run does.** It gives Ouro-2.6B-Thinking exactly the inputs Run 4 gave Ouro-RLTT, and it applies the within-question test.

## Data

- **Model:** `ByteDance/Ouro-2.6B-Thinking`, revision `f1edd81e7ac41355db670500ceaf204e0f73af68` (public), frozen, all four loops.
  - `gen_run8.ensure_model` downloads it into `shared/models/ouro_thinking_local` if any file is missing. It then replaces `modeling_ouro.py` with `modeling_ouro_shim.py` and asserts the SHA-256 of every file before the model is loaded.
  - The shim is the upstream file (sha256 `263c097a…`) plus a property and setter for `UniversalTransformerCache.key_cache` / `value_cache`. Without it, `generate()` fails under transformers 4.54.1. It is the same shim the July Thinking runs used (copied from the SSD archive's `ouro_thinking_local`); a `diff` against upstream shows only those lines.
  - **Pins:**
    - `model.safetensors`: `c506a792…0a3080` (5,336,011,242 bytes), the LFS SHA-256 the HF API reports at this revision;
    - the eight small files: the SHA-256 of the files downloaded at this revision on 2026-10-10 (the API gives only git blob hashes for them);
    - `modeling_ouro.py`: the shim's SHA-256.
  - Every shard records the checked hashes. The analysis refuses a shard whose hashes differ.
- **Questions: exactly Run 4's 1,164** (`select_tasks`, unchanged): every depth-3 and depth-4 `synthetic_propositional` task no earlier run used, in hash order. Task file sha256 `d5503195…`.
- **Generation: identical to Run 4.**
  - `mcq_prompt`, without a chat template, as in Run 4 and in the July Thinking runs.
  - k = 4, T = 0.7, top-p 0.95, at most 448 new tokens, FinalAnswerStop.
  - Seed `_sid(20260724, task_uid)`: Run 4's seed rule *and* Run 4's seed, so both checkpoints sample from the same random streams. Splits `split_for(task_uid)`.
  - **Identical inputs, checked before sealing:** the Thinking tokenizer (`tokenizer.json`, `vocab.json` and `merges.txt` byte-identical to RLTT's) gives the same token ids as RLTT's for all 1,164 prompts.
  - The checkpoints differ in their end-of-text token: Thinking's is `<|im_end|>` (id 2), RLTT's `<|endoftext|>` (id 0). Each model stops on its own token, and trailing pad tokens are stripped as in Run 4.
- **Saved per answer:** as Run 4. That is token ids, sampling log-probabilities, the strict and published labels, the final-answer-line cut and its features, pooled features at N ∈ {0, 16, 32, 48, 64, 96, 128} from one causal pass, the end-of-prompt state, and prefix texts.
- **Check during generation:** the same-length equivalence check (Amendment 2026-10-04), every 25 tasks, max |diff| ≤ 1e-3. The analysis asserts it per shard.
- **Smoke test:** Run 4's two depth-2 tasks outside every cohort. Abort on any error or failed check.

## Labels, detector, N\*

- **Labels:** Run 4's `strict_answer`, the first commitment with a letter given priority over an option word. Unlabelled answers are excluded.
- **Detector:** Run 4's frozen `answer_detector.py`, applied to the prefix text with the task's statement.
- **N\*:** Run 4's label-free rule on this checkpoint's own prefixes. It is the largest N in {16, …, 128} at which at least 75% of training answers are answer-free; 16 if none qualifies.
  - If N\* is not 48 (Run 4's N\*), the out-of-fold within-question AUROC at 48 and the RLTT comparison at 48 are also reported as secondaries.
- **Hand audit:** before labels are opened, 20 answer-free training prefixes at N\* are listed (label-free) and read in full, as in Runs 4 and 6.

## Primary endpoint

- **Cohort:** labelled answers that are answer-free at N\*, in all three splits.
- **Probe, folds and endpoint: exactly as Run 7's primary** (`within_oof.py`).
  - Run 4's hidden-state probe (PCA k ∈ {16, 24} on the pooled state at N\*, then L2-logistic regression, C ∈ {0.01, 0.1, 1}) scores every answer out of fold.
  - Outer folds: sha256("oof5|" + task_uid) mod 5. Inner folds for hyperparameters: sha256("inner5|" + task_uid) mod 5. No GroupKFold.
  - Within-question AUROC, pairs-weighted. 95% question-clustered bootstrap, 2,000 draws, seed 20260724.
- **Verdicts, applied in this order:**
  1. `UNDERPOWERED` if fewer than 60 questions have mixed outcomes;
  2. `WITHIN_QUESTION_SIGNAL` if the lower bound > 0.5;
  3. `NO_WITHIN_QUESTION_SIGNAL_ABOVE_0.55` if the upper bound < 0.55;
  4. otherwise `UNRESOLVED`.
- **Expected power** (an estimate, not a measurement). On depth-3 tasks the July Thinking data give 45% strictly labelled answers with 88.5% accuracy; on depth 4, 14% labelled (28 answers).
  - An independence calculation with 76% answer-free answers predicts about 115 mixed questions.
  - The same calculation over-predicted Run 4 by a factor of 1.55 (202 predicted, 130 observed), because difficulty clusters within questions.
  - So **about 70 mixed questions** are expected, against a minimum of 60. UNDERPOWERED is a real possibility.

## Pre-specified secondaries

1. **Run 4's across-question primary and its headline rule,** unchanged:
   - `proper_arm` with S = log-probabilities of the first N\* tokens + prompt tokens, depth, variables, facts, rules and answer class;
   - held-out opened once, minimums 30 held-out tasks and 30 held-out incorrect answers;
   - verdict `HORIZON_THINKING_FIXED_PREFIX_INCREMENT_{POSITIVE, NEGATIVE, UNRESOLVED, NOT_ADJUDICABLE}`;
   - headline `READABILITY_BEYOND_DIFFICULTY` if POSITIVE and the end-of-prompt control's CI is above zero, otherwise `MOSTLY_DIFFICULTY`.
2. **End-of-prompt control** (Run 4's control 1).
3. **Held-out only, within-question** (Run 4's control 2), on the held-out scores of secondary 1.
4. **TF-IDF reader, out of fold, within-question**, plus the paired hidden-minus-reader difference (as Run 7's secondary 3; text = prompt + first N\* tokens).
5. **Commit timing** (as Run 7's secondary 5). The commitment position is the character offset of the `strict_answer` match, as in the 2026-10-04 audit's `commit_oof.py`.
6. **Chronological labels.** The same patterns as `strict_answer`, with the earliest match winning (a letter no longer beats an earlier option word). Reported: how many answers change their label. If the cohort or any label changes, the primary is recomputed.
7. **Comparison with Ouro-RLTT on the same questions:**
   - within-question AUROC (Thinking minus RLTT), out of fold with the same hash folds for both;
   - unpaired: each side is bootstrapped over its own mixed questions (2,000 draws each, seed 20260724), and the interval is taken over the differences;
   - each checkpoint at its own label-free cut: RLTT at Run 4's N\* = 48, Thinking at N\*. If N\* is not 48, both are also compared at 48.
   - **RLTT is recomputed from Run 4's saved shards** (SSD archive: `OURO_COLD_ARCHIVE_20260927/tree/opi/preanswer/horizon_cohort2_20261003/`, 12 shards with the pooled features at every cut and the strict labels). No new generation is needed.
   - The same step reports how many answers have identical prompt ids and seeds in Run 4 and Run 8 (expected: all).
   - It runs on the laptop (`--rltt-data`). Without that flag it is reported as not run.
8. **Run 4's remaining secondaries,** unchanged: the held-out visible reader with S, the end-of-prompt state alone, the Run 1-style analysis, the grid, the final-answer line under the published protocol, the final-answer line with strict labels and difficulty, and the label cross-table.

## Where each stage runs, time, cost

- **One A40 pod, 3 processes** (`cloud/run_job.sh run8 3`). Smoke test, generation, and the label-free audit stage all run on the pod.
  - The smoke test runs online and downloads the model. The generation processes run offline and only verify the hashes.
- **The full stage runs on the laptop CPU,** because the RLTT comparison needs Run 4's shards from the SSD. `analyze_run8.py --stage full --data <SSD dir> --rltt-data <SSD>/…/horizon_cohort2_20261003 --out <run dir>`.
- **Estimate.** Run 4 (RLTT, same tasks) took 10.4 h on one A40 with 3 processes (0.031 tasks/s, `cloud/pods.log`).
  - Thinking runs longer: in the July data, 60% of depth-3 answers hit the 448-token cap, against 36% for RLTT in Run 4.
  - So 1.0–1.4 × Run 4, about 10.5–14.5 h, plus about 0.5 h for setup, the 5.3 GB download and copying back. That is **about $5.5–7.5** at $0.49/h.
  - About 4.2 GB of shards.
- **Stopping early:** if stopped early, the analysis uses complete shards and says so.

## Rules fixed now

- No reruns with other seeds, cuts, folds or thresholds after seeing results. UNRESOLVED is not evidence of absence.
- Any secondary below its minimums is reported as not run.

## Differences from Run 4 (the run this copies)

- **Model:** Ouro-2.6B-Thinking at the pinned revision, with the cache shim and SHA-256 checks of every file. Run 4 used Ouro-RLTT from the private repo, checked by `setup_pod.sh`. Shards `run8_shard_*.pt` record the model and its hashes.
- **Primary endpoint:** out-of-fold within-question AUROC with hash folds. Run 4's primary (the across-question increment) and its headline rule are now secondary 1.
- **New secondaries:** out-of-fold TF-IDF reader and paired difference, commit timing, chronological labels, the comparison with RLTT, and N = 48 if N\* differs.
- **Stages:** the full stage runs on the laptop instead of the pod (for the RLTT comparison). The hand-audit selection salt is 20261010 instead of 20261003.
- **Unchanged:** questions, prompt, sampling, seeds, splits, labels, detector, features, grid, N\* rule, S, estimators, minimums, and the same-length check.
