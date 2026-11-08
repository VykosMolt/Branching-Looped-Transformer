# T2: the tracking statistic of controls.py, recomputed (OPI v4 review item 1.1)

Post hoc, 2026-11-08. Not preregistered.

`results/fixed_prefix_rerun_20260927/posthoc_20260927/controls.py` (part 3) reports the share of mixed-verdict sibling pairs (two answers to the same question that end in different options) in which a verdict probe on the 48-token hidden states "follows each answer's own verdict". It computes
`mean[(p_i(own_i) > p_j(own_i)) + (p_j(own_j) > p_i(own_j))] / 2`.
Both terms are NumPy bools, and NumPy bool + bool is a logical OR. A pair where both comparisons hold therefore counts 0.5 instead of 1, and the statistic cannot exceed 0.5. The published value, 0.332 [0.295, 0.368], comes from that expression.

`tracking_fix.py` reruns part 3 exactly: the same Run 1 records, cut, folds, model and RNG, with the same draw order. It then asserts that the original expression reproduces the published `controls.json` values for `verdict_on_wrong_answers_trained_on_all` and `verdict_siblings_different_verdicts` (gate: passed). Finally it computes the intended statistic, `mean[(int(a) + int(b)) / 2]`, on the same out-of-fold predictions and the same 2,000 question-clustered bootstrap draws.

| | value | 95% CI |
|---|---:|---|
| Original expression (as published) | 0.332 | [0.295, 0.368] |
| Corrected (chance 0.5) | 0.5444 | [0.4737, 0.6126] |

110 questions and 214 pairs. Both comparisons hold in 42.5% of pairs and neither in 33.6%. The verdict probe does not follow each answer's own eventual option: the corrected share is at chance, not below it. It does point to the correct option, picking it for 69.7% of the wrong answers (unchanged; reproduced in the gate).

Inputs: the Run 1 shards on the archive drive (`OURO_COLD_ARCHIVE_20260927/tree/opi/preanswer/fixed_prefix_rerun_20260927`). Their SHA-256 match the published `CHECKSUMS_OFFLINE.sha256` (6 of 6). Helper modules come from `~/ouro_project/shared` as in the original run. Hashes of every input are in `inputs.sha256`.

Command: `PYTHONDONTWRITEBYTECODE=1 ~/ouro_project/shared/venv/bin/python -u tracking_fix.py` (about 1 minute on CPU). Two runs gave the same `tracking_fix_results.json` SHA-256.
