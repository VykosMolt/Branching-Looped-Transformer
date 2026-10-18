# Run 7: the within-question test on every unused GSM8K problem

**DROPPED 2026-10-10 by the user, before sealing or any generation.** The post-hoc out-of-fold scoring of Run 6 (D1_janiak_agreement) already gave a within-question AUROC of 0.526 [0.447, 0.598] on 93 GSM8K problems with mixed outcomes, which rules out an effect as large as Horizon's; this run would only have narrowed the null. Kept for the record; never sealed or run.

Prepared 2026-10-10. **Not sealed.** `seal_run7.sh` writes `SEAL.sha256` with the hashes of this file, `gen_run7.py`, `analyze_run7.py`, `within_oof.py`, `test_run7.py`, and of the Run 6 and Run 3 files they import unchanged: `gen_run6.py`, `analyze_run6.py`, `gen_run3.py`, `analyze_run3.py`, `answer_detector_gsm8k.py`. The analysis refuses to run if any of them changed. No generation for this run exists.

## Why

- **Run 6's within-question test had no power.** Held-out hidden-state probe 0.482 [0.310, 0.655], from 56 pairs on 20 problems with mixed outcomes. Across questions Run 6 was UNRESOLVED (+0.059 [−0.039, +0.156]), headline MOSTLY_DIFFICULTY.
- **Within a question, difficulty is constant.** The within-question AUROC is the test that difficulty cannot explain. On Horizon it was 0.740 held-out and 0.709 out of fold (Run 4, post hoc).
- **What this run does.** It generates on every GSM8K problem no earlier run used, and it scores every answer out of fold, so all splits contribute pairs.
- **Expected power** (a scaling estimate from Run 6, not a measurement): Run 6 had 20 mixed problems among 551 held-out problems (3.6%). Out of fold over 6,016 problems that gives about 200 mixed problems and 550 pairs. The bootstrap interval would then be about ±0.055 wide on each side (Run 6's ±0.17 × √(20/200)).
  - So `NO_WITHIN_QUESTION_SIGNAL_ABOVE_0.55` is reachable only if the estimate is near 0.49 or lower.
  - With Run 6 pooled in, it is about 265 problems.

## Data

- **Problems: every GSM8K (`openai/gsm8k`, main) problem no earlier run generated on, 6,016 in all.** They are 5,470 training problems and 546 test problems, sorted by sha256 of the task id. The excluded problems:
  - the original 170: test indices 0–169, found in the surviving record `preanswer_oof_predictions.csv` (task ids `gsm8k/0`…`gsm8k/169`, 680 rows);
  - Run 3: 600 cohort problems and 3 smoke problems, test indices 170–1,318 in hash order, positions 0–602;
  - Run 6: 2,000 cohort problems (training split, first 2,000 in hash order) and 3 smoke problems (last 3).
  - The task's request expected about 6,190 problems. The difference is the original 170 (7,473 − 2,000 + 1,319 − 600 = 6,192) and the 6 smoke problems.
- **Overlap check.** `gen_run7.select_tasks` asserts zero overlap with every earlier problem list, by task id and by normalized question text (NFKC, case-folded, non-alphanumeric runs collapsed). It also asserts no duplicate text inside the cohort, and exactly 6,016 problems.
  - Checked before sealing against the archived Run 3 and Run 6 shards (the problem sets in the shards equal the reconstructed lists) and against the 170 in the CSV. Overlap 0 by id and by text.
- **Files.** Read from local copies, with no network:
  - `elastic_reasoner/data/gsm8k/main/train-00000-of-00001.parquet`, sha256 `ea82612e…` (Run 6's pin, revision `740312ad…`);
  - `test-00000-of-00001.parquet`, sha256 `ee7b8da9…` (Run 3's pin).
  - The training file was copied from the SSD archive on 2026-10-10; its hash equals Run 6's pin.
- **Task ids:** `gsm8k-train/{index}` and `gsm8k/{index}`, as Runs 6 and 3. Splits: `split_for(task_id)` (50/20/30).
- **Generation: identical to Run 6.**
  - `gsm_prompt`; Ouro-RLTT frozen, all four loops.
  - k = 4, T = 0.7, top-p 0.95, at most 448 new tokens; the batch stops once every candidate has written "FINAL ANSWER: number" plus 8 tokens.
  - Seed `_sid(20261010, task_id)`: Run 6's rule with this run's date.
  - Weights from the private repo `Vykos/Ouro-2.6B-RLTT`; `cloud/setup_pod.sh` checks the SHA-256 of the three weight files before anything runs (unchanged).
- **Saved per answer:** as Run 6, plus `dataset_split`.
- **Check during generation:** the same-length equivalence check (Amendment 2026-10-04 of Runs 4 and 6). Every 25 problems, the pooled features at N = min(128, length) must not change when the later tokens are replaced, with the total length kept the same. Max |diff| ≤ 1e-3. The analysis asserts it per shard.
- **Smoke test:** 3 problems from Run 6's cohort, so that no unused problem is spent on it. Abort on any error or failed check.

## Labels, detector, cut

- **Labels:** Run 6's. The first committed answer (Run 3's `first_commitment`, patterns in priority order) is numerically equal to the reference answer. Answers with no commitment are excluded.
- **Detector:** Run 3's frozen `answer_detector_gsm8k.py`.
- **Cut: fixed in advance at N = 96**, Run 6's N\*. Reported but not used: what Run 6's label-free rule (the largest N with at least 75% of training answers answer-free) would choose on this run's prefixes, and the fractions.
- **Hand audit:** before labels are opened, 20 answer-free training prefixes at N = 96 are listed (Run 6's `hand_audit`, label-free) and read in full, as in Runs 4 and 6.

## Primary endpoint

- **Cohort:** parseable answers that are answer-free at N = 96, in all three splits.
- **Probe: Run 6's hidden-state probe**, `analyze_run3._fit_score` on the hidden block only:
  - the pooled state at N = 96 (layers 24, 36, 47 × 4 loops × 2,048) is standardized and reduced by PCA (k ∈ {16, 24});
  - then standardized again, and fit with L2-logistic regression (C ∈ {0.01, 0.1, 1});
  - hyperparameters are chosen by the best inner cross-validated AUROC.
- **Out-of-fold scores:**
  - every answer is scored by a probe fit on the other four of 5 folds;
  - the folds are assigned by sha256("oof5|" + task_id) mod 5, so a problem's four answers share a fold;
  - the hyperparameters inside each training set use 5 inner folds by sha256("inner5|" + task_id) mod 5.
  - No GroupKFold. Its tie order depends on the machine: the Run 4 audit could not reproduce the pod's folds on the laptop.
- **PCA computation:** the same PCA as `analyze_run3._pca` (training-set standardization, top singular vectors). The top 24 are computed by ARPACK (`scipy.sparse.linalg.svds`, fixed start vector, tol = 0) instead of a full SVD.
  - On a 14,000 × 24,576 test the subspace agrees with an exact eigendecomposition to 1e-15 (cosine). In `test_run7.py` the probe's predictions match `analyze_run3._fit_score` to 1.8e-7, with the same hyperparameters.
  - A full SVD at this size takes about an hour per fit (139 s at 3,000 rows on the laptop, growing with rows²), and the analysis needs about 150 fits.
- **Endpoint:** within-question AUROC. This is the share of (correct, incorrect) answer pairs from the same problem in which the correct answer scores higher, ties counting ½, weighted by pairs.
  - The interval is a 95% question-clustered bootstrap over problems with mixed outcomes: 2,000 draws, seed 20260724 (`analyze_run3.within_task_auroc`).
- **Verdicts, applied in this order:**
  1. `UNDERPOWERED` if fewer than 100 problems have mixed outcomes;
  2. `WITHIN_QUESTION_SIGNAL` if the lower bound > 0.5;
  3. `NO_WITHIN_QUESTION_SIGNAL_ABOVE_0.55` if the upper bound < 0.55;
  4. otherwise `UNRESOLVED`.
  - An interval inside (0.5, 0.55) therefore reads WITHIN_QUESTION_SIGNAL.

## Pre-specified secondaries (no verdicts, except the across-question rule named below)

1. **Held-out only, within-question:** Run 6's control 2. The held-out scores of the across-question arm (secondary 4) for the combined model, the hidden-only model and S, with the train/val/held-out split as in Run 6.
2. **Combined probe and baseline, out of fold, within-question:**
   - combined = PCA(H) + S;
   - baseline = S, where S is Run 6's: the mean and minimum log-probability of the first 96 tokens, prompt tokens, and reference steps.
   - Same hash folds.
3. **TF-IDF reader, out of fold, within-question.**
   - The text is Run 6's reader text: the prompt plus the first 96 tokens. Word 1–2-grams and character 3–5-grams, text only. C ∈ {0.01, 0.1, 1, 10} by inner CV on the same hash folds (the reader of the 2026-10-04 audit).
   - Also the paired hidden-minus-reader difference on the same pairs, with the same question resamples (2,000 draws, seed 20260724).
4. **Run 6's across-question increment, with Run 6's sealed estimator:**
   - `proper_arm`: GroupKFold for hyperparameters on train + val, held-out opened once, S as above;
   - AUROC(PCA(H) + S) − AUROC(S) with a 2,000-round paired task-clustered bootstrap.
   - The PCA inside it is computed by ARPACK as above (`analyze_run3._pca` is replaced at run time; same subspace).
   - Verdict `GSM8K_REST_FIXED_PREFIX_INCREMENT_{POSITIVE, NEGATIVE, UNRESOLVED, NOT_ADJUDICABLE}`, with Run 6's minimums: 30 held-out problems and 30 held-out incorrect answers.
   - The end-of-prompt control: the baseline adds PCA(end-of-prompt state).
   - Run 6's headline rule: `READABILITY_BEYOND_DIFFICULTY` if the increment is POSITIVE and the control's CI is above zero; otherwise `MOSTLY_DIFFICULTY`.
   - GroupKFold's tie order can differ between machines; the platform, processor and library versions are written to `results.json`.
5. **Commit timing.** The commitment position is the character offset of the match that defines the label. It is a future event, after the cut. Reported on the out-of-fold scores:
   - the share of pairs in which the correct answer commits later;
   - the within-question AUROC of commit position alone;
   - the AUROC of the probe, the combined model, S and the reader, split by which answer commits first (pairs at equal positions reported separately).
6. **Pooled Run 6 + Run 7.** Out-of-fold within-question AUROC of the hidden-state probe on the union of both primary cohorts, with the hash folds over both. Run 6's shards come from the SSD archive. The analysis asserts that the two problem sets are disjoint and that Run 6's problems are Run 6's cohort.
7. **Chronological labels, as a sensitivity analysis.**
   - Same patterns, but the earliest match wins (the primary parser takes the patterns in priority order; in Run 3, 8 of 2,400 answers differ).
   - The answer-free status at 96 is recomputed with the chronological answer.
   - Reported: how many answers change their answer and their label. If the cohort or any label changes, the primary out-of-fold within-question AUROC is recomputed on the chronological cohort.
- **Also reported:** accuracy, parseable fraction, the 448-token cap rate, and problems by dataset split.

## Where each stage runs, time, cost

- **Generation: 3 A40 pods in parallel, 3 processes each** (`cloud/run_job.sh run7 3 <pod> 3`).
  - Pod p runs global parts 3p, 3p + 1 and 3p + 2 of 9 (`--part`, `--nparts 9`), so it takes the shards whose index mod 9 is in {3p, 3p + 1, 3p + 2}.
  - The 61 shards split as 21 / 21 / 19 (2,100 / 2,100 / 1,816 problems).
  - Every problem keeps its own seed, so the split does not change what is generated.
  - Each pod runs its own smoke test first.
- **Estimate.** Run 6 ran at 0.039 problems/s on one A40 with 3 processes: 2,000 problems in 14.25 h (`cloud/pods.log`).
  - About 15.0 / 15.0 / 12.9 h per pod, plus about 0.3 h each for setup, smoke and copying back. That is about 44 pod-hours, or **about $21.5** at $0.49/h, if the throughput matches Run 6.
  - About 26 GB of shards; about 9.2 GB per pod.
- **Audit and full stages: on the laptop CPU, after the shards are copied to the SSD.** Both need every pod's shards, and the pooled secondary needs Run 6's shards, which are only on the SSD.
  - The integrity check needs no network: the problem list comes from the local pinned parquet files. That fixes the Run 6 audit crash, where the offline check tried to download the training file.
  - Commands:
    - `analyze_run7.py --stage audit --data <SSD dir> --out <run dir>`;
    - then the hand audit;
    - then `analyze_run7.py --stage full --data <SSD dir> --run6-data <SSD>/…/gsm8k_train_20261003 --out <run dir>`.
  - Memory: only the N = 96 features and the end-of-prompt state are kept, about 5 GB.
- **Stopping early:** if stopped early, the analysis uses complete shards and says so.

## Rules fixed now

- No reruns with other seeds, cuts, folds or thresholds after seeing results. UNRESOLVED is not evidence of absence.
- Any secondary below its minimums is reported as not run.

## Differences from Run 6 (the run this copies)

- **Problems:** all 6,016 unused GSM8K problems (training and test), not the first 2,000 training problems. They are read from local pinned parquet files instead of `hf_hub_download`, and asserted disjoint from every earlier run.
- **Seed:** 20261010 instead of 20261003 (same rule). Shards `run7_shard_*.pt`. Records add `dataset_split`.
- **Smoke problems:** 3 from Run 6's cohort instead of the end of the hash order.
- **Cut:** N = 96 fixed in advance; Run 6 chose N\* = 96 by the 75% rule, which is now only reported.
- **Primary endpoint:** within-question AUROC of the hidden-state probe, out of fold over hash folds, all splits. Run 6's primary was the held-out across-question increment, now secondary 4. Run 6's within-question control is now secondary 1.
- **New secondaries:** out-of-fold combined and baseline; out-of-fold TF-IDF reader with the paired difference; commit timing; pooled Run 6 + Run 7; chronological labels.
- **Dropped Run 6 secondaries** (not requested, and each would cost about an hour of CPU at this size): the held-out visible reader with S, the end-of-prompt state alone, the Run 1-style analysis, task level, excluding the reference number, and the grid.
- **PCA:** computed by ARPACK instead of a full SVD, with the same subspace.
- **Stages:** generation on 3 pods instead of 1. The audit and full stages run on the laptop instead of the pod, and the integrity check needs no network.
- **Unchanged:** prompt, sampling, stop rule, labels, detector, features, grid, S, the same-length check, the weights check, the hand-audit selection, and the bootstrap seeds.
