# C1: first set (Run 1) without the 15 in-window detector misses

**Post hoc.** Everything here was decided on 2026-10-10, after all Run 1 results had been seen, in response to the external audit (audit-v4-20261008/REPORT.md, correction 2; claims.md section 2). It is a sensitivity analysis and does not replace the sealed Run 1 result.

## Inputs (read only)
- Run 1 shards and sealed code: `/run/media/moloch/ARCH_BACKUP/OURO_COLD_ARCHIVE_20260927/tree/opi/preanswer/fixed_prefix_rerun_20260927/`. `analyze_run1.py`, `answer_detector.py`, `gen_run1.py` and `PLAN.md` match `SEAL.sha256` / `ANALYSIS_SEAL.sha256` (sha256 checked 2026-10-10). The sealed module is imported from the SSD with `python -B`, so nothing is written there.
- Estimators: `opi/preanswer/gsm8k_fixed_prefix_20260927/analyze_run3.py` (sha256 7d8b1d57…, as in Run 4's SEAL); label rule `opi/preanswer/horizon_cohort2_20261003/gen_run4.py` (`strict_answer`, sha256 4944e7b9…, as in Run 4's SEAL).
- Definitions copied from `horizon_cohort2_20261003/posthoc_audit_20261004/r1_inwindow.py`, `opi_posthoc_20261003/strict_labels_run1_final_rule.py`, and Run 1 `posthoc_20260927/{controls,paired_within,reader_within}.py`.

## Files
- `r1data.py`: loads the 6 shards one at a time. Records are identical to `analyze_run1.load()`, except that the feature tensors are dropped apart from `prefix_features[3]` (N=48). Cache: `cache/` (deleted at the end).
- `inwindow_r1.py` → `inwindow_r1.json`, `inwindow_r1.log`: the in-window list, the chronological scan, and the N* rule.
- `c1_analysis.py --exclude none|inwindow` → `c1_none.json`/`.log` (gate), `c1_inwindow.json`/`.log`.
- `c1_oof_reader.py`: the `reader_within.py` reader. `c1_pairs_diag.py` → `c1_pairs_diag.json`/`.log`: a diagnostic that separates the effect of the removed pairs from the effect of refitting.

## Commands (all with OMP/MKL/OPENBLAS_NUM_THREADS=8, CPU only, `/home/moloch/ouro_project/shared/venv/bin/python -B`)
```
python -B inwindow_r1.py            # builds cache/r1_records_f48.pt
python -B c1_analysis.py --exclude none
python -B c1_analysis.py --exclude inwindow
python -B c1_pairs_diag.py
```

## What changes in the modified run
Only the cohort changes. The 15 answers (r1_inwindow.py definition: parseable, not flagged by `is_answer_bearing` at N=48, and `strict_answer(prefix_texts[3])` not None) are removed from the N*=48 primary cohort before anything is fitted. Hyperparameter selection, CV folds, final fits and bootstraps are all rerun with the original code. For the strict-label row, the strict-cohort answers with a `strict_answer` match inside the 48-token window are removed; these are the same 15.

## Results
In-window list (`inwindow_r1.json`): 15 entries, as published. 13 are in train/val (3 of them incorrect); 2 are held out (both correct). In all 15 the in-window answer equals the published label. A chronological scan of the same 1,387 prefixes (any accepted label pattern anywhere in the 48-token prefix) finds exactly these 15 and no others. This holds by construction: `strict_answer` returns a label iff some accepted pattern matches. The N* rule with the misses flagged: training answer-free share at 48 tokens is 0.7573 → 0.7508 (8 training misses; the same 8 when malformed answers are included), still ≥ 0.75, so N* stays 48.

Every gate reproduced the published numbers exactly (to the printed digits; item 1 and the strict row also match the saved JSONs to full precision).

| Item | Published cohort (gate = published) | Without the 15 |
|---|---|---|
| Cohort (all splits) | 1,387 answers, 567 questions, 155 incorrect | 1,372, 566, 152 |
| 1 Sealed primary | surface 0.515, hidden 0.758, both 0.758, +0.243 [0.129, 0.365]; 412 held out, 165 questions, 45 incorrect | 0.504, 0.757, 0.757, +0.253 [0.135, 0.365]; 410, 165, 45; still POSITIVE |
| 2 Strict-label row | 0.523, 0.777, +0.254 [0.105, 0.407]; 399 held out, 163 q, 29 incorrect | 0.511, 0.751, +0.240 [0.077, 0.414]; 397, 163, 29 |
| 3a Within, out of fold | probe 0.679 [0.602, 0.755], reader 0.598, probe − reader +0.081 [0.000, 0.160]; 209 pairs, 108 q | 0.655 [0.568, 0.736], reader 0.611, +0.044 [−0.035, 0.124]; 203 pairs, 106 q |
| 3a, no refit (diag) | n/a | gate scores on the 203 pairs: 0.670 [0.589, 0.748], reader 0.601, +0.069 [−0.014, 0.145] |
| 3b Within, held out | probe 0.719 [0.574, 0.847]; 57 pairs, 30 q | 0.719 [0.574, 0.855]; 57 pairs, 30 q |
| 4 Difficulty (controls.py fixed model) | baseline 0.672, combined 0.745, +0.073 [−0.014, 0.162] | 0.662, 0.736, +0.074 [−0.016, 0.158] |
| 5 Reader across questions | reader 0.641, hidden − reader +0.118 [0.032, 0.206] | 0.626, +0.131 [0.047, 0.221] |

Note: the task brief attributes item 4 to `proper_combo.py`, but the published numbers (0.672, +0.073 [−0.014, 0.162]) come from `controls.py` ("primary_fixed_model_with_difficulty", analyze_run3.proper_arm). `proper_combo.json` gives +0.084 [0.000, 0.171] for the same S, using a different estimator. The item reproduced here is the controls.py one.

## Caches deleted
`cache/r1_records_f48.pt` (252 MB) was deleted at the end.
