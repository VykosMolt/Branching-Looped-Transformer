# Post-hoc checks on Run 1 (NOT preregistered), 2026-09-27

Why: the sealed increase at N* = 48 (+0.243) is larger than the final-answer-line increase (+0.095). All numbers below use the sealed primary cohort: parseable answers that are answer-free at 48 tokens (412 held-out answers, 45 incorrect). They do not replace the sealed verdict.

## Baselines (`baselines.py`)

Rows marked "Final line" use the final-answer-line features instead of the 48-token features.

| Baseline S | S | H | Increase [95% CI] |
|---|---|---|---|
| Sealed: log-prob mean/min | 0.515 | 0.758 | +0.243 [+0.129, +0.365] |
| + prompt tokens + proof depth | 0.672 | 0.758 | +0.087 [+0.005, +0.171] |
| Prompt tokens + depth only | 0.686 | 0.758 | +0.072 [-0.013, +0.157] |
| Final line, 4 shortcuts | 0.669 | 0.824 | +0.155 [+0.054, +0.265] |
| Final line, 4 shortcuts + depth | 0.774 | 0.824 | +0.050 [-0.027, +0.132] |
| Prompt-only features, S = lp + depth | 0.672 | 0.625 | -0.047 |

In every sealed-method arm "both" equals "hidden": PCA over [H, S] swamps S.

## Proper combination (`proper_combo.py`)

PCA on H only, then S appended. Combined model AUROC:

| S | Combined | Increase [95% CI] |
|---|---|---|
| log-prob | 0.755 | +0.242 |
| depth + prompt | 0.759 | +0.073 [-0.013, +0.158] |
| log-prob + depth + prompt | 0.756 | +0.084 [+0.000, +0.171] |

## Composition (`leaks.py`)

- The detector dropped 118 of 530 held-out parseable answers. Error rate: 19.5% among dropped, 10.9% among kept.
- Train: 13.5% vs 11.8%. Val: 16.3% vs 9.9%.

## Verdict identity (`leaks.py`, `verdict.py`)

The verdict is the future answer, so these are diagnostics.

- The eventual verdict alone gives AUROC 0.793. Error rates by eventual verdict: Unknown 4%, True 14%, False 25%.
- Verdict from the 48-token hidden states: accuracy 0.682 (majority 0.498); one-vs-rest AUROC True 0.847, False 0.690, Unknown 0.848.
- Correctness AUROC within each eventual verdict:
  - True: 0.746 [0.614, 0.863], n 139, 20 incorrect;
  - False: 0.813 [0.707, 0.900], n 68, 22 incorrect;
  - Unknown: 0.665, n 205, 3 incorrect.
- Proper combination:
  - S = verdict: 0.793 → 0.863, increase +0.070 [+0.006, +0.130];
  - S = verdict + depth + prompt + lp: 0.830 → 0.857, increase +0.027 [-0.032, +0.085].

## Margin to commitment (`leaks.py`)

The sealed model, scored within held-out groups by when the answer first commits:

| First commitment | n | Incorrect | Hidden AUROC |
|---|---|---|---|
| Tokens 49–64 | 30 | 3 | 0.691 |
| Tokens 65–128 | 139 | 13 | 0.877 |
| None through 160 | 183 | 24 | 0.704 |

No cliff near the commitment.

## Unembedding (`logit_lens.py`, GPU)

The model's own next-token distribution at the cut, over 1,387 primary answers (all splits):

- P(eventual verdict) 0.0035 vs P(each other verdict) 0.0032.
- The top token is the eventual verdict for 0.3% of answers, and any verdict for 0.9%.
- Max over the prefix positions: P(eventual) 0.116 vs P(other) 0.067.

## Controls requested 2026-09-27 evening (`controls.py`, `controls.json`)

Run with the fixed combined model of Run 3 amendment 1: PCA per hidden block, then S.

**Final-answer line, 4 shortcuts + prompt tokens + proof depth**

| Cohort | Surface | Hidden | Combined | Increase [95% CI] |
|---|---|---|---|---|
| 510 new | 0.682 | 0.772 | 0.776 | +0.093 [+0.041, +0.150] |
| pooled 680 | 0.655 | 0.763 | 0.761 | +0.106 [+0.054, +0.165] |
| fresh 599 | 0.751 | 0.782 | 0.783 | +0.032 [-0.031, +0.098] |

- First 170: below the 30-incorrect minimum (19).
- Swamping signature, sealed `arm()` on the 510: hidden 0.773, combined 0.772.

**48-token cohort, S = lp + depth + prompt tokens:** surface 0.672, hidden 0.758, combined 0.745, increase +0.073 [-0.014, +0.162].

**Within-question AUROC** (48-token probe, difficulty constant within a pair):

| Scores | Combined | Hidden | Surface |
|---|---|---|---|
| Held-out (30 questions, 57 pairs) | 0.684 [0.529, 0.828] | 0.719 [0.574, 0.847] | 0.526 |
| Out-of-fold, all splits (108 questions, 209 pairs) | 0.665 [0.594, 0.738] | 0.679 [0.602, 0.755] | 0.488 |

**Interim prompt control** (mean-pooled prompt in the baseline, not the end-of-prompt state): baseline 0.633, combined 0.695, increase +0.063 [+0.015, +0.111]. The combined model falls below hidden-only (0.758), so the estimator is noisy with two blocks.

**Verdict dissociation, out-of-fold, 48-token states: fails.**
- On the 155 wrong answers, the argmax is the gold verdict 0.697 [0.616, 0.776] of the time and the model's own verdict only 0.174 [0.108, 0.247].
- Trained on correct answers only: gold 0.742, own 0.161.
- Siblings with different verdicts (110 questions, 214 pairs): the probe tracks each sample's own verdict 0.332 [0.295, 0.368], below chance, unexplained.
- So the "verdict" readable at 48 tokens is problem type (the correct answer), not the model's own commitment.

## End-of-prompt control and reader within question

Scripts: `prompt_end_states.py` (GPU, 93 s), `prompt_control.py`, `reader_within.py`, `paired_within.py`.

**End-of-prompt state in the baseline** (the form Run 3 prespecifies), 48-token cohort, S = lp + depth + prompt tokens:

| Version | Baseline | Combined | Increase [95% CI] |
|---|---|---|---|
| PCA block of the state | 0.698 | 0.754 | +0.056 [-0.011, +0.128] |
| State as one cross-fitted score in S | 0.698 | 0.770 | +0.071 [-0.011, +0.160] |

- The end-of-prompt state adds nothing significant beyond depth and length: +0.011 [-0.055, +0.074] on all parseable answers, +0.021 [-0.051, +0.084] on the 48-token cohort.
- **By the rule the user set, the across-question leftover does not survive the prompt-only probe: pre-answer readability across questions is mostly difficulty.**

**Within-question AUROC** (out-of-fold, 108 questions, 209 pairs):

| Scores | Estimate [95% CI] |
|---|---|
| Hidden-state probe | 0.679 [0.602, 0.755] |
| TF-IDF reader, prompt + 48 tokens | 0.598 [0.520, 0.678] |
| TF-IDF reader, 48 tokens only | 0.584 [0.509, 0.659] |
| Paired, hidden minus reader | +0.081 [0.000, +0.160] |
