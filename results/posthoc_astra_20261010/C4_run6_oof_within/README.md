# C4: Run 6 (GSM8K training split) within-question AUROC, out of fold over the whole primary cohort

**Post hoc.** Requested on 2026-10-10 after the Run 6 result had been seen (the held-out within-question test had only 20 mixed problems / 56 pairs). Not preregistered; it does not replace the sealed Run 6 verdict.

## Design (template: `horizon_cohort2_20261003/posthoc_audit_20261004/oof_within.py`, `commit_oof.py`)
- Cohort: the sealed Run 6 primary cohort (parseable, answer-free at N*=96 under the sealed GSM8K detector): 6,514 answers, 1,879 problems, 295 incorrect, all splits.
- Outer folds: 5 folds by `int(sha256(task_id), 16) % 5` (not GroupKFold); sizes 1297 / 1220 / 1354 / 1356 / 1287 answers.
- Probe scores: Run 6's sealed estimator, `analyze_run3._fit_score`, unchanged. Inside each outer-training part it selects k ∈ {16, 24} and C ∈ {0.01, 0.1, 1} by its own inner 5-fold GroupKFold, then refits. hidden = [H96]; combined = [H96] + S; baseline = S. S = log-prob mean/min of the first 96 tokens + prompt tokens + reference steps (`analyze_run6.lp_at`, `s_diff`). Speed only: a one-entry memo around the unchanged `analyze_run3._pca` stops the identical SVD from being computed twice (once for k=16, once for k=24). It returns the identical object for an identical input array, so every number is unchanged.
- TF-IDF reader: the template's reader (prompt + first 96 tokens; word 1–2 + char_wb 3–5 grams; liblinear; C ∈ {0.01, 0.1, 1, 10} by inner GroupKFold; random_state=0). No S is appended, unlike Run 6's sealed held-out reader.
- Within-question AUROC and its question-clustered bootstrap (2,000 draws, seed 20260724): `analyze_run3.within_task_auroc`. Paired hidden − reader: the template's paired bootstrap.
- Commit timing: commit position = character offset, in the whole answer, of the match that gives the label. For the executed parser this is the first match of the highest-priority pattern found; for the chronological parser it is the earliest match.
- `--labels chrono`: labels from the C2 chronological parser (4 primary-cohort labels flip True→False), cohort unchanged.

## Gate
The sealed held-out within-question numbers (hidden 0.482 [0.310, 0.655], combined 0.518, baseline 0.589; 56 pairs, 20 problems) were reproduced in C2. `../C2_parser_priority/c2_run6_forced.json` reproduces them exactly, CIs included, with the pod's selected hyperparameters. `c2_run6_rerun.json` uses the laptop GroupKFold order: all three point estimates and the hidden CI match exactly; the combined CI differs because the laptop selects C=0.1 where the pod selected C=1.0 (see ../C2_parser_priority/README.md).

## Files and commands (OMP/MKL/OPENBLAS_NUM_THREADS=8, CPU only, `/home/moloch/ouro_project/shared/venv/bin/python -B`)
| Command | Output |
|---|---|
| `python -B r6data.py` | `cache/r6_records_f96.pt` (deleted at the end), `r6data.log` |
| `python -B c4_oof_within.py --labels executed` | `c4_executed.json`, `c4_executed.log`, `oof_scores_executed.npy` |
| `python -B c4_oof_within.py --labels chrono` | `c4_chrono.json`, `c4_chrono.log`, `oof_scores_chrono.npy` |

`oof_scores_*.npy`: rows = hidden, combined, baseline, reader; columns = the cohort in record order (shard order, executed cohort).

## Results (all mixed problems, out of fold, all splits)
| | Executed labels | Chronological labels (C2) |
|---|---|---|
| Cohort | 6,514 answers, 1,879 problems, 295 incorrect | 6,514, 1,879, 299 incorrect |
| Mixed problems / pairs | 93 / 253 | 97 / 260 |
| Hidden probe | 0.534 [0.450, 0.613] | 0.527 [0.447, 0.610] |
| Combined (H + S) | 0.522 [0.447, 0.602] | 0.527 [0.451, 0.601] |
| Baseline (S) | 0.494 [0.413, 0.576] | 0.496 [0.419, 0.576] |
| TF-IDF reader | 0.447 [0.367, 0.532] | 0.450 [0.367, 0.531] |
| Hidden − reader (paired) | +0.087 [−0.012, +0.181] | +0.077 [−0.016, +0.175] |
| Across-question AUROC (hidden / combined / baseline / reader) | 0.709 / 0.725 / 0.636 / 0.596 | 0.703 / 0.719 / 0.633 / 0.595 |
| Share of pairs where the correct answer commits later | 0.530 (134/253; 2 ties) | 0.531 (138/260; 2 ties) |
| AUROC of "commits later" alone | 0.534 | 0.535 |
| Hidden probe, correct commits later / earlier | 0.582 (134 pairs) / 0.487 (117) | 0.572 (138) / 0.483 (120) |

Reading: across problems the out-of-fold probe separates correct from incorrect answers (0.709), but within a problem it does not: 0.534, with a CI that includes 0.5. That is the Horizon pattern (0.709 within question on 237 pairs) not replicating on GSM8K, now with 253 pairs from 93 problems rather than 56 pairs from 20. Commit timing barely separates pairs on GSM8K (0.534 vs 0.781 on Horizon).

## Caches deleted
`cache/r6_records_f96.pt` (703 MB) was deleted after both runs.
