# D1 for Denis Janiak: does the pre-answer readout add anything once agreement is in the baseline?

Post hoc (PLAN.md, sha256 `ec6b8e55fa9abf1b...`, written before any result). Ouro-RLTT, 4 sampled answers per question, 5 question-hash folds, everything out of fold. Baselines are **with entropy**. AUACC higher is better, AURC lower is better; deltas are left minus right with 95% bootstrap intervals (1,000 draws). The probe reads only hidden states pooled over the prompt and the first N* generated tokens of an answer that has stated no answer by then; agreement comes from the 4 completed answers.

## Plain reading

Once the baseline holds what the four completed answers say, plus token entropy, the pre-answer readout adds nothing measurable, for either target, on either task. For per-trajectory correctness (T1), B1 − B0 in AURC is −0.0004 [−0.0014, +0.0005] on Horizon (Run 4: 2,376 answer-free answers, 211 incorrect) and +0.0004 [−0.0003, +0.0010] on GSM8K (Run 6: 6,514 answers, 295 incorrect). For plurality correctness (T2, all questions) it is −0.0010 [−0.0042, +0.0014] on Horizon (1,164 questions, 117 incorrect pluralities) and +0.0008 [+0.0000, +0.0017] on GSM8K. GSM8K has only 74 incorrect pluralities, and there the probe slightly worsens the fitted readout. Do not read more into that number. Without entropy the four numbers were nearly the same (−0.0005, +0.0003, −0.0006, +0.0008; `SUMMARY_FOR_DENIS_no_entropy.md`). The readout is not just a re-reading of agreement. Its correlation with vote share is about 0.2. On Horizon it still ranks answers inside the all-four-agree stratum (T1: AUROC 0.69 [0.60, 0.77], 1,855 answers, 36 incorrect; T2 unanimous stratum: 0.60 [0.50, 0.69], 40 incorrect pluralities). But what it carries beyond agreement is too small to help a readout that already has agreement, entropy, length, log-probability and question features. Scored alone, its out-of-fold residual on the vote reaches 0.50 to 0.58 AUROC at the question level. Against the paper's own fair baseline, built from the pre-answer window alone (log-probability, entropy and question features), the probe still adds, though less once window entropy is in: AUROC +0.036 [+0.020, +0.053] on Horizon (+0.047 without entropy) and +0.022 [+0.001, +0.042] on GSM8K (+0.044 without). On the first Horizon set it is +0.005 [−0.019, +0.028]. Within a question, the probe ranks siblings only on Horizon (0.76 [0.70, 0.82]). On GSM8K it is at chance (0.53 [0.45, 0.60]). The one plurality-level gain, on the first Horizon set (B1 − B0 = −0.017 [−0.036, −0.002]), disappears once question features are in the baseline (B1q − B0q = +0.0008), so it is question difficulty. In short, the pre-answer signal is real but weak, and completed-sibling agreement subsumes it. Everything here is post hoc. The entropy step was planned in PLAN.md before any result.

## T1: per-trajectory correctness (rows = answer-free answers)

B0 = completed length, completed mean log-prob, hit-limit flag, vote share, agrees-with-plurality, question features. B1 = B0 + probe score. F0 = the paper's fair baseline (window log-prob mean/min + question features, nothing from completed answers). Intervals are question-clustered.

| run | answers (questions, incorrect) | readout | AUROC | AUACC | AURC | Brier |
|---|---|---|---:|---:|---:|---:|
| run4 N*=48 | 2376 (1067, 211) | B0_traj | 0.928 | 0.987 | 0.012 | 0.040 |
| run4 N*=48 | 2376 (1067, 211) | B1_traj | 0.931 | 0.988 | 0.012 | 0.039 |
| run4 N*=48 | 2376 (1067, 211) | probe_alone | 0.736 | 0.961 | 0.039 | 0.077 |
| run4 N*=48 | 2376 (1067, 211) | F0_window | 0.720 | 0.952 | 0.047 | 0.076 |
| run4 N*=48 | 2376 (1067, 211) | F1_window | 0.756 | 0.962 | 0.037 | 0.075 |
| run6 N*=96 | 6514 (1879, 295) | B0_traj | 0.829 | 0.986 | 0.014 | 0.032 |
| run6 N*=96 | 6514 (1879, 295) | B1_traj | 0.826 | 0.986 | 0.014 | 0.032 |
| run6 N*=96 | 6514 (1879, 295) | probe_alone | 0.670 | 0.973 | 0.027 | 0.043 |
| run6 N*=96 | 6514 (1879, 295) | F0_window | 0.675 | 0.973 | 0.027 | 0.042 |
| run6 N*=96 | 6514 (1879, 295) | F1_window | 0.697 | 0.976 | 0.024 | 0.042 |
| run1 N*=48 | 1339 (556, 101) | B0_traj | 0.907 | 0.986 | 0.013 | 0.044 |
| run1 N*=48 | 1339 (556, 101) | B1_traj | 0.903 | 0.985 | 0.015 | 0.045 |
| run1 N*=48 | 1339 (556, 101) | probe_alone | 0.721 | 0.961 | 0.038 | 0.065 |
| run1 N*=48 | 1339 (556, 101) | F0_window | 0.728 | 0.962 | 0.038 | 0.065 |
| run1 N*=48 | 1339 (556, 101) | F1_window | 0.733 | 0.963 | 0.036 | 0.065 |

| run | contrast | ΔAUROC | ΔAUACC | ΔAURC | ΔBrier |
|---|---|---|---|---|---|
| run4 | B1_traj_minus_B0_traj | +0.003 [-0.003, +0.010] p=0.334 | +0.0004 [-0.0005, +0.0014] p=0.400 | -0.0004 [-0.0014, +0.0005] p=0.400 | -0.0010 [-0.0019, -0.0000] p=0.036 |
| run4 | F1_window_minus_F0_window | +0.036 [+0.020, +0.053] p=0.000 | +0.0100 [+0.0050, +0.0169] p=0.000 | -0.0100 [-0.0169, -0.0050] p=0.000 | -0.0015 [-0.0037, +0.0005] p=0.164 |
| run6 | B1_traj_minus_B0_traj | -0.003 [-0.010, +0.004] p=0.368 | -0.0004 [-0.0010, +0.0003] p=0.282 | +0.0004 [-0.0003, +0.0010] p=0.282 | +0.0000 [-0.0002, +0.0002] p=0.808 |
| run6 | F1_window_minus_F0_window | +0.022 [+0.001, +0.042] p=0.030 | +0.0023 [+0.0003, +0.0044] p=0.012 | -0.0023 [-0.0044, -0.0003] p=0.012 | +0.0000 [-0.0004, +0.0005] p=0.966 |
| run1 | B1_traj_minus_B0_traj | -0.004 [-0.013, +0.005] p=0.394 | -0.0013 [-0.0054, +0.0013] p=0.508 | +0.0013 [-0.0013, +0.0054] p=0.508 | +0.0003 [-0.0015, +0.0026] p=0.754 |
| run1 | F1_window_minus_F0_window | +0.005 [-0.019, +0.028] p=0.684 | +0.0015 [-0.0011, +0.0042] p=0.294 | -0.0015 [-0.0042, +0.0011] p=0.294 | -0.0001 [-0.0021, +0.0021] p=0.974 |

Within-question AUROC (mixed questions only; the vote share differs between a correct and an incorrect sibling, so B0 is strong here by construction):

| run | mixed questions / pairs | B0 | B1 | probe alone | F0 | F1 |
|---|---|---|---|---|---|---|
| run4 | 130 / 237 | 0.844 [0.774, 0.905] | 0.861 [0.797, 0.919] | 0.764 [0.704, 0.822] | 0.599 [0.527, 0.674] | 0.705 [0.640, 0.766] |
| run6 | 93 / 253 | 0.787 [0.698, 0.865] | 0.787 [0.698, 0.865] | 0.526 [0.447, 0.598] | 0.545 [0.467, 0.621] | 0.553 [0.472, 0.631] |
| run1 | 61 / 116 | 0.733 [0.608, 0.846] | 0.750 [0.624, 0.864] | 0.621 [0.519, 0.718] | 0.690 [0.591, 0.785] | 0.681 [0.568, 0.782] |

### T1 Orgad-style controls (agreement = the answer's own vote share)

| run | Pearson / Spearman probe vs vote share | AUROC probe | AUROC vote share | probe in agrees=1 | probe in agrees=0 | residual of probe on (vote share, agrees) |
|---|---|---|---|---|---|---|
| run4 | 0.222 / 0.224 | 0.736 [0.704, 0.768] | 0.887 [0.851, 0.922] | 0.728 [0.669, 0.786] (n=2199, 79 wrong) | 0.619 [0.505, 0.724] (n=177, 132 wrong) | 0.567 [0.519, 0.613] |
| run6 | 0.175 / 0.107 | 0.670 [0.618, 0.724] | 0.755 [0.707, 0.804] | 0.634 [0.564, 0.711] (n=6390, 195 wrong) | 0.550 [0.417, 0.677] (n=124, 100 wrong) | 0.544 [0.480, 0.606] |
| run1 | 0.234 / 0.202 | 0.721 [0.656, 0.785] | 0.881 [0.830, 0.937] | 0.718 [0.607, 0.835] (n=1265, 50 wrong) | 0.586 [0.428, 0.732] (n=74, 51 wrong) | 0.546 [0.460, 0.629] |

Probe AUROC inside each vote-share level (fewer than 10 of either class: not reported):

| run | vote share | answers | incorrect | accuracy | probe AUROC |
|---|---:|---:|---:|---:|---|
| run4 | 0.250 | 42 | 40 | 0.048 | not reported |
| run4 | 0.333 | 83 | 69 | 0.169 | 0.638 [0.472, 0.800] |
| run4 | 0.500 | 121 | 52 | 0.570 | 0.694 [0.617, 0.781] |
| run4 | 0.667 | 149 | 6 | 0.960 | not reported |
| run4 | 0.750 | 126 | 8 | 0.937 | not reported |
| run4 | 1.000 | 1855 | 36 | 0.981 | 0.691 [0.601, 0.766] |
| run6 | 0.250 | 85 | 69 | 0.188 | 0.531 [0.362, 0.699] |
| run6 | 0.333 | 17 | 13 | 0.235 | not reported |
| run6 | 0.500 | 88 | 30 | 0.659 | 0.566 [0.440, 0.680] |
| run6 | 0.667 | 21 | 4 | 0.810 | not reported |
| run6 | 0.750 | 344 | 47 | 0.863 | 0.685 [0.541, 0.826] |
| run6 | 1.000 | 5959 | 132 | 0.978 | 0.594 [0.492, 0.685] |
| run1 | 0.250 | 22 | 21 | 0.045 | not reported |
| run1 | 0.333 | 31 | 21 | 0.323 | 0.757 [0.559, 0.910] |
| run1 | 0.500 | 47 | 20 | 0.574 | 0.528 [0.379, 0.672] |
| run1 | 0.667 | 79 | 14 | 0.823 | 0.731 [0.557, 0.862] |
| run1 | 0.750 | 89 | 8 | 0.910 | not reported |
| run1 | 1.000 | 1071 | 17 | 0.984 | 0.687 [0.444, 0.941] |

## T2: plurality-answer correctness (rows = questions), Denis's table

B0 = length, logprob, vote_agreement over the 4 completed answers plus entropy; B0q = B0 + question features; B1 = B0 + mean probe score over the question's answer-free answers; B1q = B0q + the same.

### run4 / full_population

1164 questions; 117 incorrect pluralities (base accuracy 0.899); automatic failures 24; questions with no answer-free answer (probe imputed) 10.

| model | AUROC | AUACC (higher) | AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.829 | 0.965 | 0.035 | 0.062 | 0.231 |
| B0q | 0.864 | 0.975 | 0.024 | 0.063 | 0.225 |
| B1 | 0.830 | 0.966 | 0.034 | 0.062 | 0.232 |
| B1q | 0.861 | 0.975 | 0.025 | 0.064 | 0.226 |

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | +0.0010 [-0.0014, +0.0040] p=0.472 | -0.0010 [-0.0042, +0.0014] p=0.456 | +0.0004 [-0.0003, +0.0012] p=0.312 | +0.0009 [-0.0011, +0.0029] p=0.328 |
| B1q_minus_B0q | -0.0003 [-0.0009, +0.0004] p=0.300 | +0.0003 [-0.0003, +0.0010] p=0.284 | +0.0004 [-0.0000, +0.0009] p=0.060 | +0.0016 [+0.0003, +0.0029] p=0.016 |

### run4 / valid_plurality

1140 questions; 93 incorrect pluralities (base accuracy 0.918); automatic failures 0; questions with no answer-free answer (probe imputed) 10.

| model | AUROC | AUACC (higher) | AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.783 | 0.965 | 0.034 | 0.063 | 0.233 |
| B0q | 0.829 | 0.975 | 0.024 | 0.063 | 0.224 |
| B1 | 0.786 | 0.966 | 0.033 | 0.063 | 0.234 |
| B1q | 0.825 | 0.974 | 0.025 | 0.063 | 0.226 |

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | +0.0011 [-0.0005, +0.0029] p=0.172 | -0.0011 [-0.0029, +0.0004] p=0.182 | +0.0004 [-0.0006, +0.0014] p=0.466 | +0.0007 [-0.0019, +0.0032] p=0.662 |
| B1q_minus_B0q | -0.0006 [-0.0014, -0.0002] p=0.004 | +0.0006 [+0.0002, +0.0013] p=0.002 | +0.0005 [+0.0001, +0.0009] p=0.024 | +0.0018 [+0.0007, +0.0032] p=0.002 |

### run6 / full_population

2000 questions; 74 incorrect pluralities (base accuracy 0.963); automatic failures 6; questions with no answer-free answer (probe imputed) 108.

| model | AUROC | AUACC (higher) | AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.784 | 0.985 | 0.014 | 0.031 | 0.130 |
| B0q | 0.798 | 0.988 | 0.012 | 0.031 | 0.129 |
| B1 | 0.778 | 0.984 | 0.015 | 0.031 | 0.131 |
| B1q | 0.790 | 0.987 | 0.012 | 0.031 | 0.131 |

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | -0.0008 [-0.0017, +0.0000] p=0.056 | +0.0008 [+0.0000, +0.0017] p=0.042 | +0.0002 [-0.0001, +0.0005] p=0.286 | +0.0011 [+0.0000, +0.0022] p=0.044 |
| B1q_minus_B0q | -0.0008 [-0.0015, -0.0002] p=0.010 | +0.0008 [+0.0002, +0.0015] p=0.008 | +0.0002 [-0.0002, +0.0006] p=0.306 | +0.0013 [-0.0001, +0.0028] p=0.062 |

### run6 / valid_plurality

1994 questions; 68 incorrect pluralities (base accuracy 0.966); automatic failures 0; questions with no answer-free answer (probe imputed) 108.

| model | AUROC | AUACC (higher) | AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.763 | 0.985 | 0.014 | 0.031 | 0.130 |
| B0q | 0.777 | 0.988 | 0.012 | 0.031 | 0.129 |
| B1 | 0.755 | 0.984 | 0.015 | 0.031 | 0.131 |
| B1q | 0.767 | 0.987 | 0.013 | 0.031 | 0.131 |

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | -0.0008 [-0.0017, -0.0001] p=0.020 | +0.0008 [+0.0001, +0.0017] p=0.026 | +0.0002 [-0.0001, +0.0006] p=0.288 | +0.0012 [-0.0000, +0.0023] p=0.054 |
| B1q_minus_B0q | -0.0009 [-0.0015, -0.0002] p=0.002 | +0.0009 [+0.0003, +0.0016] p=0.008 | +0.0002 [-0.0002, +0.0007] p=0.302 | +0.0013 [-0.0001, +0.0029] p=0.058 |

### run1 / full_population

599 questions; 56 incorrect pluralities (base accuracy 0.907); automatic failures 7; questions with no answer-free answer (probe imputed) 3.

| model | AUROC | AUACC (higher) | AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.756 | 0.941 | 0.057 | 0.066 | 0.245 |
| B0q | 0.872 | 0.978 | 0.020 | 0.061 | 0.213 |
| B1 | 0.810 | 0.958 | 0.040 | 0.067 | 0.240 |
| B1q | 0.869 | 0.977 | 0.021 | 0.061 | 0.216 |

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | +0.0174 [+0.0017, +0.0355] p=0.026 | -0.0174 [-0.0356, -0.0021] p=0.024 | +0.0015 [-0.0023, +0.0055] p=0.470 | -0.0052 [-0.0186, +0.0067] p=0.376 |
| B1q_minus_B0q | -0.0008 [-0.0023, +0.0001] p=0.096 | +0.0008 [-0.0001, +0.0027] p=0.108 | +0.0009 [+0.0001, +0.0017] p=0.034 | +0.0025 [-0.0000, +0.0051] p=0.052 |

### run1 / valid_plurality

592 questions; 49 incorrect pluralities (base accuracy 0.917); automatic failures 0; questions with no answer-free answer (probe imputed) 3.

| model | AUROC | AUACC (higher) | AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.721 | 0.941 | 0.057 | 0.066 | 0.248 |
| B0q | 0.852 | 0.979 | 0.019 | 0.061 | 0.215 |
| B1 | 0.783 | 0.959 | 0.040 | 0.068 | 0.243 |
| B1q | 0.848 | 0.978 | 0.020 | 0.062 | 0.218 |

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | +0.0174 [+0.0020, +0.0355] p=0.018 | -0.0174 [-0.0360, -0.0011] p=0.026 | +0.0015 [-0.0022, +0.0055] p=0.466 | -0.0053 [-0.0186, +0.0075] p=0.448 |
| B1q_minus_B0q | -0.0007 [-0.0021, +0.0001] p=0.094 | +0.0007 [-0.0001, +0.0020] p=0.110 | +0.0008 [+0.0000, +0.0017] p=0.048 | +0.0023 [+0.0001, +0.0048] p=0.046 |

### T2 Orgad controls (probe aggregate in place of rmd_tail_q20)

| run | population | n (with probe) | Pearson / Spearman vs vote | AUROC probe | AUROC vote | unanimous stratum | split stratum | probe given vote (residual) | vote given probe (residual) |
|---|---|---|---|---|---|---|---|---|---|
| run4 | full_population | 1164 (1154) | 0.213 / 0.237 | 0.624 [0.566, 0.676] | 0.785 [0.736, 0.834] | 0.596 [0.504, 0.685] (n=908, 40 wrong) | 0.521 [0.437, 0.594] (n=256, 77 wrong) | 0.497 [0.432, 0.556] | 0.757 [0.692, 0.816] |
| run4 | valid_plurality | 1140 (1130) | 0.238 / 0.231 | 0.620 [0.562, 0.679] | 0.730 [0.672, 0.786] | 0.596 [0.504, 0.685] (n=908, 40 wrong) | 0.532 [0.440, 0.622] (n=232, 53 wrong) | 0.517 [0.447, 0.585] | 0.694 [0.618, 0.766] |
| run6 | full_population | 2000 (1892) | 0.154 / 0.101 | 0.657 [0.591, 0.723] | 0.707 [0.647, 0.768] | 0.599 [0.499, 0.698] (n=1796, 38 wrong) | 0.651 [0.554, 0.746] (n=204, 36 wrong) | 0.568 [0.488, 0.647] | 0.672 [0.589, 0.754] |
| run6 | valid_plurality | 1994 (1886) | 0.155 / 0.094 | 0.650 [0.581, 0.723] | 0.681 [0.623, 0.741] | 0.599 [0.499, 0.698] (n=1796, 38 wrong) | 0.649 [0.544, 0.750] (n=198, 30 wrong) | 0.576 [0.497, 0.654] | 0.643 [0.551, 0.723] |
| run1 | full_population | 599 (596) | 0.230 / 0.172 | 0.681 [0.595, 0.758] | 0.764 [0.688, 0.829] | 0.707 [0.562, 0.829] (n=479, 21 wrong) | 0.597 [0.472, 0.718] (n=120, 35 wrong) | 0.576 [0.474, 0.666] | 0.697 [0.596, 0.800] |
| run1 | valid_plurality | 592 (589) | 0.188 / 0.149 | 0.663 [0.568, 0.752] | 0.730 [0.653, 0.802] | 0.707 [0.562, 0.829] (n=479, 21 wrong) | 0.564 [0.432, 0.685] (n=113, 28 wrong) | 0.587 [0.479, 0.688] | 0.656 [0.545, 0.756] |

Substitution both ways (AURC, lower is better; Denis's four contrasts):

| run | population | B1 − B0 | probe for vote − B0 | B1 − (probe for vote) | probe added to voteless |
|---|---|---|---|---|---|
| run4 | full_population | -0.0010 [-0.0042, +0.0016] p=0.476 | +0.0451 [+0.0235, +0.0678] p=0.000 | -0.0462 [-0.0683, -0.0255] p=0.000 | -0.0094 [-0.0211, +0.0028] p=0.106 |
| run4 | valid_plurality | -0.0011 [-0.0030, +0.0004] p=0.158 | +0.0156 [+0.0050, +0.0267] p=0.002 | -0.0167 [-0.0275, -0.0068] p=0.000 | -0.0065 [-0.0130, -0.0014] p=0.014 |
| run6 | full_population | +0.0008 [+0.0000, +0.0016] p=0.034 | +0.0025 [+0.0002, +0.0056] p=0.038 | -0.0017 [-0.0048, +0.0005] p=0.214 | +0.0007 [-0.0000, +0.0013] p=0.054 |
| run6 | valid_plurality | +0.0008 [+0.0001, +0.0017] p=0.020 | +0.0023 [+0.0002, +0.0048] p=0.030 | -0.0015 [-0.0040, +0.0004] p=0.182 | +0.0008 [+0.0001, +0.0016] p=0.020 |
| run1 | full_population | -0.0174 [-0.0353, -0.0019] p=0.016 | +0.0104 [-0.0186, +0.0396] p=0.486 | -0.0278 [-0.0475, -0.0093] p=0.004 | -0.0396 [-0.0692, -0.0111] p=0.006 |
| run1 | valid_plurality | -0.0174 [-0.0356, -0.0008] p=0.026 | +0.0101 [-0.0183, +0.0339] p=0.480 | -0.0275 [-0.0466, -0.0097] p=0.000 | -0.0341 [-0.0592, -0.0086] p=0.008 |

## Primary contrasts (PLAN.md reading rule: AURC delta B1 − B0, interval entirely below zero)

| contrast | ΔAURC | raw p | Holm p | reading |
|---|---|---|---|---|
| run4 T1 | -0.0004 [-0.0014, +0.0005] p=0.400 | 0.400 | 0.846 | not shown |
| run4 T2 full_population | -0.0010 [-0.0042, +0.0014] p=0.456 | 0.456 | 0.846 | not shown |
| run6 T1 | +0.0004 [-0.0003, +0.0010] p=0.282 | 0.282 | 0.846 | not shown |
| run6 T2 full_population | +0.0008 [+0.0000, +0.0017] p=0.042 | 0.042 | 0.168 | not shown |
