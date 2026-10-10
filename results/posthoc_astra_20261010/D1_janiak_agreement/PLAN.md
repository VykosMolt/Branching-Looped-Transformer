# D1: does the pre-answer readout add anything once agreement is in the baseline?

**Post hoc.** Written 2026-10-10, before any number in this directory was computed, but after every sealed result of Runs 1, 4 and 6 had been seen (Run 4: UNRESOLVED / MOSTLY_DIFFICULTY; Run 6: UNRESOLVED / MOSTLY_DIFFICULTY; Run 1: POSITIVE). It answers a promise to Denis Janiak (e-mail of 2026-09-12). Nothing here replaces a sealed result. The SHA-256 of this file is in `PLAN.sha256`; any change after hashing is listed at the end under "Deviations".

## Question

Completed-sibling agreement is information the probe cannot have at the cut. It is the strongest baseline I can give it, not a fair one. Does the strictly pre-answer hidden-state readout still add anything once the baseline holds agreement (and, later, token entropy)? Two targets, kept apart:

- **T1, per-trajectory correctness** (OPI's target): rows are answers.
- **T2, plurality-answer correctness** (Denis's target): rows are questions.

## Data (read only)

| run | role | shards | N\* | label | answer-free rule |
|---|---|---|---|---|---|
| Run 4, Horizon second set | primary, Horizon | `run4_shard_*.pt` | 48 | `gen_run4.strict_answer(generated_text, options) == gold_letter`; unlabelled if `strict_answer` is None (the stored `strict_answer`/`success` fields) | `answer_detector.is_answer_bearing(prefix_texts[N*], task_statement(task_prompt))` is False |
| Run 6, GSM8K train | primary, GSM8K | `run6_shard_*.pt` | 96 | `value(own_answer) == value(gold)`; unparseable if `value(own_answer)` is None (stored `parseable`/`success`) | `answer_detector_gsm8k.is_answer_bearing(prefix_texts[N*], own_answer)` is False |
| Run 1, Horizon first set | secondary | `run1_shard_off*.pt` | 48 | strict, recomputed with `gen_run4.strict_answer` | as Run 4 (same detector file) |

N\* is the sealed value of each run. Shards are read one at a time; a compact cache (metadata plus the pooled features at N\* for answer-free answers) is written to `cache/` and deleted at the end. Run 1 question features (variables, facts, rules) are looked up in the sealed task file by `task_uid`.

## Folds

5 folds, `fold = int(sha256("D1_janiak|" + question_id), 16) % 5`, shared by every readout, target and run. Inner folds for setting selection inside the fixed combined estimator: `int(sha256("D1_janiak_inner|" + question_id), 16) % 5` over the training questions of the outer fold (not GroupKFold, whose tie order is unseeded). All splits (train, val, held-out) are pooled; every score is out of fold.

## Readouts

- **Low-dimensional feature sets:** Denis's `crossfit_logistic_predictions` (train-fold mean imputation, train-fold standardization, `LogisticRegression(max_iter=2000, random_state=42)`), vendored.
- **Hidden states:** the paper's fixed combined estimator as in `analyze_run3._fit_score` / `posthoc_audit_20261004/fastarm.py`: one hidden block (pooled features at N\*, layers 24/36/47 on 4 loops, 24,576 dims), standardized and reduced by PCA fitted on training rows only, k in {16, 24}; L2 logistic, C in {0.01, 0.1, 1}; settings chosen by inner grouped CV AUROC; refit on all training folds; score the test fold.

## 1. Pre-answer probe score

Training rows: labelled answer-free answers. Hidden states alone (no surface features). For each outer fold the model is trained on the other four folds and scores **every answer-free answer** in the held-out fold, labelled or not (the label is not needed to score). Answers that state an answer inside the window get no probe score.

## 2. T1, per-trajectory correctness

Rows: labelled answer-free answers.

- **B0_traj:** completed-trace length (generated tokens), completed-trace mean sampled log-probability (`mean(gen_logprobs)`; these are log-probabilities under the sampling distribution, T = 0.7 and top-p 0.95, as stored), hit-token-limit flag, **vote share** (number of parseable answers among the question's 4 giving this answer, including this one, divided by the number of parseable answers), **agrees-with-plurality** (this answer equals the plurality winner by Denis's rule), plus question features (Run 4 / Run 1: prompt tokens, proof depth, variables, facts, rules, answer class one-hot; Run 6: prompt tokens, reference steps).
- **B1_traj = B0_traj + the OOF probe score**, both through Denis's crossfit logistic with the same folds. Why this form and not the fixed combined estimator with B0_traj appended: (a) B0 and B1 then use the same estimator, so the delta isolates the feature; (b) it is the structure of Denis's own increment (an OOF geometry score added to B0 in a crossfit logistic); (c) T2 must use the stacked form anyway. Known limitation: the probe scores of training rows come from probes that saw the test fold's labels (standard OOF stacking, as in Denis's design); the test row's own score is always out of fold.
- **Reported:** AUROC, AUACC, AURC, Brier, log loss for B0, B1 and the probe alone; paired B1 − B0 deltas for each metric with a **question-clustered** bootstrap (1,000 draws, seed 42; a trajectory-level bootstrap would ignore that siblings share a question); within-question AUROC on mixed (correct, incorrect) pairs for B0, B1 and the probe (`analyze_run3.within_task_auroc`, imported).
- **Orgad-style controls**, agreement = the trajectory's vote share: Pearson and Spearman between probe and vote share; probe AUROC inside agrees-with-plurality = 1 and = 0; probe AUROC inside each vote-share level; AUROC of the OOF linear residual of the probe on (vote share, agrees-with-plurality), scored alone. Strata with fewer than 10 of either class are not reported (Denis's `MIN_STRATUM_CLASS`). Unanimous questions have no mixed outcomes, so no unanimous stratum exists for T1.
- **Sample sizes:** rows, questions, incorrect answers, mixed questions, mixed pairs; also the number of rows whose window contains a strict-rule answer that the detector missed (reported, the cohort is not changed).

## 3. T2, plurality-answer correctness

Rows: questions (all 4 completed answers).

- Outcome: Denis's `_plurality_outcome` (plurality of parseable answers, ties broken by the highest per-answer mean sampled log-probability; an all-unparsed question is an automatic failure, outcome 0).
- **B0** = Denis's four features over the 4 completed answers: length (mean generated tokens), **entropy (omitted in this pass, see 4)**, logprob (mean over answers of mean sampled log-probability), vote_agreement (share of parseable answers giving the winner; 0 if none parseable). Averages over all answers, vote over parseable answers, as `aggregate_prompt_features`.
- **B0q** = B0 + question features. **B1** = B0 + mean probe score over the question's answer-free answers (NaN if none; Denis's train-mean imputation). **B1q** = B0q + the same (added for symmetry).
- **Reported exactly as his report table:** AUACC, AURC, Brier, log loss for every readout; paired deltas B1 − B0 and B1q − B0q for all four metrics, `paired_bootstrap_delta` with 1,000 draws and his seed rule (`42 + 1000 + len(metric) + len(label)`). Populations: `full_population` (all questions) and `valid_plurality` (at least one parseable answer).
- **Orgad controls** (as `controls/orgad_agreement_control.analyze_model`, probe aggregate in place of `rmd_tail_q20`), on both populations: Pearson and Spearman with vote_agreement; marginal AUROCs; AUROC inside the unanimous stratum first, then split, and the per-level table; AUROC of the OOF residual of the probe aggregate on vote_agreement alone, and the mirror; substitution both ways (probe in place of the vote; the vote added on top of the probe), AURC paired deltas, seed 42.
- Sample sizes: questions, incorrect pluralities, automatic failures, questions with no answer-free answer. GSM8K will have few plurality errors; the count is stated and the result is not over-read.

## 4. Entropy

The shards store no per-token entropy. Everything above is run **without entropy** and labelled so. `entropy_pass.py` (not run here; needs the GPU) writes `entropy/entropy_ouro_rltt.npz` with, per (run, question id, candidate index), the mean next-token entropy at temperature 1 over the whole generation and over the first N\* generated tokens. When that file exists, `analysis.py` adds whole-generation entropy to B0_traj and to T2's B0 (Denis's `entropy`), and window entropy to the fair baseline (5), and labels the results "with entropy".

## 5. Fair comparison (reference)

Rows as T1. **F0** = mean and minimum sampled log-probability of the first N\* generated tokens + question features (the paper's surface set S; no completed-trace or sibling information). **F1** = F0 + probe score. Same crossfit logistic, metrics and clustered bootstrap as T1.

## Reading rule (fixed now)

Primary contrasts: AURC delta B1 − B0 for T1 and for T2 (`full_population`) on Run 4 and Run 6 (four contrasts). "The pre-answer readout adds beyond agreement" for a run and target only if the 95% interval of that AURC delta is entirely below zero. Raw intervals are primary; Holm-adjusted p over the four is reported alongside. An interval that includes zero is "not shown", not evidence of absence. Run 1 is secondary. No threshold, cut, fold rule or feature set is changed after results are seen.

## Deviations

(none yet)
