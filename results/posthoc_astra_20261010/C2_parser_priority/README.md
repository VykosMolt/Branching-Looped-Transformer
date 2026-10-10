# C2: format-priority labels vs chronological first commitment

**Post hoc.** Everything here was decided on 2026-10-10, after all results had been seen, in response to the external audit (audit-v4-20261008/REPORT.md, correction 5; claims.md section 7). Nothing here replaces a sealed result.

## Parsers
- Executed parsers, imported unchanged: Horizon `gen_run4.strict_answer` (Run 4 labels and the Run 1 strict-label row); GSM8K `answer_detector_gsm8k.first_commitment` (Run 3 dir; Run 6 imports the same file, sha256 in Run 6's SEAL).
- `chrono_parsers.py`: chronological variants built from the same regex strings and compiled patterns. The label is the answer of the match with the earliest start position; a tie in position keeps the executed priority. The self-test (`python -B chrono_parsers.py`) passes 38 cases. It includes both audit examples (`FINAL ANSWER: True ... FINAL ANSWER: B` → executed B, chronological A; `\boxed{7} ... FINAL ANSWER: 9` → 9 vs 7) and every case of the two sealed self-tests. The parsers differ on 0/13 sealed Horizon cases and on 1/8 sealed GSM8K cases (`Answer: 16 eggs ... FINAL ANSWER: 20`: executed 20, chronological 16).
- By construction both parsers label the same set of answers (a label exists iff any accepted pattern matches), so labelled/unlabelled status cannot change.

## Files and commands (OMP/MKL/OPENBLAS_NUM_THREADS=8, CPU only, `/home/moloch/ouro_project/shared/venv/bin/python -B`)
| Command | Output |
|---|---|
| `python -B chrono_parsers.py` | self-test (stdout) |
| `python -B c2_counts.py run1 run4 run3 run6` | `c2_counts.json`, `c2_examples.md`, `c2_counts.log`; caches `cache/meta_run*.pkl` (deleted at the end) |
| `python -B horizon_pattern_census.py` | `horizon_pattern_census.json` |
| `python -B run3_adjudicability.py` | `run3_adjudicability.json` |
| `python -B c2_run6_rerun.py` | `c2_run6_rerun.json`, `.log` (laptop GroupKFold order: executed gate, chrono labels, chrono labels + cohort) |
| `python -B c2_run6_forced.py` | `c2_run6_forced.json`, `.log` (pod's selected hyperparameters forced) |

Run 6 features come from `../C4_run6_oof_within/r6data.py` (shards loaded one at a time; cache deleted at the end). Run 1 records come from `../C1_first_set_clean/r1data.py`.

## Definitions
- primary = labelled under the executed rule and answer-free at the run's N* with the run's sealed detector (Horizon: N*=48; GSM8K: N*=96). The GSM8K detector also tests for the candidate's own answer, so `c2_counts.json` also reports how many answers would change answer-free status if the chronological own answer were used.
- GSM8K "label differs" is counted two ways: as strings (`'$9.00'` vs `'9.00'`) and as numeric values. Correctness compares values, as the sealed code does.
- `executed_parser_vs_stored_labels_mismatches` = 0 for all runs: recomputing the executed parser reproduces the stored labels exactly.

## Run 6 fold order
The pod's GroupKFold tie order comes from numpy's argsort on an AVX-512 CPU. This laptop has no AVX-512, so that order cannot be rebuilt here. With the laptop order the sealed code reproduces surface, hidden and all within-question point estimates exactly, but it selects combined C=0.1 where the pod selected C=1.0 (increase +0.051 instead of +0.059). Held-out scores depend on the folds only through the selected (k, C). Forcing the pod's selections (`c2_run6_forced.py`) therefore reproduces the published primary and within-question numbers exactly, including the CIs. Both orders are reported, each as a paired executed-vs-chronological comparison.

## Results
See the JSON files; the summary is in the report to the main session.

### Summary of results
Disagreements between the executed (format-priority) and chronological labels, counted on numeric values for GSM8K; string-level counts are in parentheses. Labelled/unlabelled status changed for 0 answers in every run.

| Run | All labelled | label differs | correctness flips | Primary cohort | differs | flips | Primary ∩ held out | differs | flips |
|---|---|---|---|---|---|---|---|---|---|
| Run 1, strict rule | 1,841 | 0 | 0 | 1,339 | 0 | 0 | 399 | 0 | 0 |
| Run 4 | 3,277 | 0 | 0 | 2,376 | 0 | 0 | 735 | 0 | 0 |
| Run 3 (GSM8K fixed cut) | 2,327 | 8 (10) | 7 (6 T→F, 1 F→T) | 1,863 | 6 (8) | 5 (4 T→F, 1 F→T) | 577 | 0 | 0 |
| Run 6 (GSM8K train) | 7,823 | 11 (28) | 10 (all T→F) | 6,514 | 5 (18) | 4 (all T→F) | 1,907 | 1 (6) | 1 (T→F) |

- Horizon: a marker followed by an option word occurs in 0 of 2,396 Run 1 answers and 1 of 4,656 Run 4 answers, and that one comes after a letter (`horizon_pattern_census.json`). The two parsers therefore cannot disagree on these data, and no rerun was needed for Run 1 or Run 4.
- Run 3: no held-out label changes. The primary stays NOT_ADJUDICABLE (168 held-out problems, 21 incorrect, under every variant; `run3_adjudicability.json`), so there is nothing to rerun.
- Run 6 primary (S = lp + difficulty) and held-out within-question:
  - pod hyperparameters forced: executed (= published, exact) +0.059 [−0.039, 0.156], hidden within 0.482 [0.310, 0.655] (56 pairs, 20 problems) → chronological labels +0.061 [−0.034, 0.155], within 0.491 [0.315, 0.667] (57 pairs, 21 problems) → chronological labels and cohort +0.061 [−0.036, 0.156], within 0.464 [0.293, 0.632] (56, 20).
  - laptop GroupKFold order (selection rerun): executed +0.051 [−0.045, 0.147], within 0.482 → chronological labels +0.054 [−0.039, 0.145], within 0.491 → chronological labels and cohort +0.057 [−0.037, 0.151], within 0.464.
  - Verdict UNRESOLVED in every variant.
- Many chronological "first answers" on GSM8K are not commitments: they are mid-reasoning phrases such as "the answer would be 50 - R" or "maybe the answer is 0", which the executed parser's preference for the instructed marker avoids (`c2_examples.md`).

### Caches deleted
`cache/meta_run{1,3,4,6}.pkl` (6–31 MB each) were deleted at the end.
