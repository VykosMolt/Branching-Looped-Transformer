# D1 for Denis Janiak: does the pre-answer readout add anything once agreement is in the baseline?

Post hoc (PLAN.md, sha256 `ec6b8e55fa9abf1b...`, written before any result). Ouro-RLTT, 4 sampled answers per question, 5 question-hash folds, everything out of fold. Baselines are **WITHOUT entropy (B0 minus entropy; entropy_pass.py not yet run)**. AUACC higher is better, AURC lower is better; deltas are left minus right with 95% bootstrap intervals (1,000 draws). The probe reads only hidden states pooled over the prompt and the first N* generated tokens of an answer that has stated no answer by then; agreement comes from the 4 completed answers.

## Plain reading

Once the baseline holds what the four completed answers say, the pre-answer readout adds nothing measurable, for either target, on either task. For per-trajectory correctness (T1), B1 − B0 in AURC is −0.0005 [−0.0016, +0.0006] on Horizon (Run 4: 2,376 answer-free answers, 211 incorrect) and +0.0003 [−0.0004, +0.0010] on GSM8K (Run 6: 6,514 answers, 295 incorrect). For plurality correctness (T2, all questions) it is −0.0006 [−0.0036, +0.0019] on Horizon (1,164 questions, 117 incorrect pluralities) and +0.0008 [+0.0000, +0.0017] on GSM8K. GSM8K has only 74 incorrect pluralities, and there the probe slightly worsens the fitted readout. Do not read more into that number. The readout is not just a re-reading of agreement. Its correlation with vote share is about 0.2. On Horizon it still ranks answers inside the all-four-agree stratum (T1: AUROC 0.69 [0.60, 0.77], 1,855 answers, 36 incorrect; T2 unanimous stratum: 0.60 [0.50, 0.69], 40 incorrect pluralities). But what it carries beyond agreement is too small to help a readout that already has agreement, length, log-probability and question features. Scored alone, its out-of-fold residual on the vote reaches 0.50 to 0.58 AUROC at the question level. Against the paper's own fair baseline, built from the pre-answer window alone, the probe does add: AUROC +0.047 [+0.029, +0.065] on Horizon and +0.044 [+0.015, +0.074] on GSM8K (+0.012 [−0.012, +0.037] on the first Horizon set). Within a question, though, the probe ranks siblings only on Horizon (0.76 [0.70, 0.82]). On GSM8K it is at chance (0.53 [0.45, 0.60]), so the GSM8K gain is across questions. The one plurality-level gain, on the first Horizon set (B1 − B0 = −0.016 [−0.034, +0.000]), disappears once question features are in the baseline (B1q − B0q = +0.0007), so it is question difficulty. In short, the pre-answer signal is real but weak, and completed-sibling agreement subsumes it. These baselines omit token entropy. `entropy_pass.py` adds it later on a GPU, which should only strengthen the baseline. Everything here is post hoc.

## T1: per-trajectory correctness (rows = answer-free answers)

B0 = completed length, completed mean log-prob, hit-limit flag, vote share, agrees-with-plurality, question features. B1 = B0 + probe score. F0 = the paper's fair baseline (window log-prob mean/min + question features, nothing from completed answers). Intervals are question-clustered.

| run | answers (questions, incorrect) | readout | AUROC | AUACC | AURC | Brier |
|---|---|---|---:|---:|---:|---:|
| run4 N*=48 | 2376 (1067, 211) | B0_traj | 0.927 | 0.987 | 0.012 | 0.040 |
| run4 N*=48 | 2376 (1067, 211) | B1_traj | 0.931 | 0.988 | 0.012 | 0.039 |
| run4 N*=48 | 2376 (1067, 211) | probe_alone | 0.736 | 0.961 | 0.039 | 0.077 |
| run4 N*=48 | 2376 (1067, 211) | F0_window | 0.707 | 0.951 | 0.049 | 0.077 |
| run4 N*=48 | 2376 (1067, 211) | F1_window | 0.753 | 0.962 | 0.038 | 0.076 |
| run6 N*=96 | 6514 (1879, 295) | B0_traj | 0.825 | 0.986 | 0.014 | 0.032 |
| run6 N*=96 | 6514 (1879, 295) | B1_traj | 0.823 | 0.986 | 0.014 | 0.032 |
| run6 N*=96 | 6514 (1879, 295) | probe_alone | 0.670 | 0.973 | 0.027 | 0.043 |
| run6 N*=96 | 6514 (1879, 295) | F0_window | 0.625 | 0.966 | 0.034 | 0.043 |
| run6 N*=96 | 6514 (1879, 295) | F1_window | 0.669 | 0.972 | 0.028 | 0.043 |
| run1 N*=48 | 1339 (556, 101) | B0_traj | 0.913 | 0.988 | 0.012 | 0.045 |
| run1 N*=48 | 1339 (556, 101) | B1_traj | 0.908 | 0.986 | 0.013 | 0.045 |
| run1 N*=48 | 1339 (556, 101) | probe_alone | 0.721 | 0.961 | 0.038 | 0.065 |
| run1 N*=48 | 1339 (556, 101) | F0_window | 0.711 | 0.959 | 0.040 | 0.066 |
| run1 N*=48 | 1339 (556, 101) | F1_window | 0.723 | 0.962 | 0.037 | 0.066 |

| run | contrast | ΔAUROC | ΔAUACC | ΔAURC | ΔBrier |
|---|---|---|---|---|---|
| run4 | B1_traj_minus_B0_traj | +0.004 [-0.003, +0.012] p=0.240 | +0.0005 [-0.0006, +0.0016] p=0.376 | -0.0005 [-0.0016, +0.0006] p=0.376 | -0.0012 [-0.0024, -0.0001] p=0.026 |
| run4 | F1_window_minus_F0_window | +0.047 [+0.029, +0.065] p=0.000 | +0.0109 [+0.0055, +0.0181] p=0.000 | -0.0109 [-0.0181, -0.0055] p=0.000 | -0.0018 [-0.0041, +0.0003] p=0.086 |
| run6 | B1_traj_minus_B0_traj | -0.002 [-0.009, +0.005] p=0.604 | -0.0003 [-0.0010, +0.0004] p=0.482 | +0.0003 [-0.0004, +0.0010] p=0.482 | +0.0000 [-0.0002, +0.0003] p=0.754 |
| run6 | F1_window_minus_F0_window | +0.044 [+0.015, +0.074] p=0.002 | +0.0057 [+0.0024, +0.0093] p=0.002 | -0.0057 [-0.0093, -0.0024] p=0.002 | -0.0000 [-0.0006, +0.0008] p=0.928 |
| run1 | B1_traj_minus_B0_traj | -0.005 [-0.018, +0.006] p=0.438 | -0.0013 [-0.0054, +0.0011] p=0.488 | +0.0013 [-0.0011, +0.0054] p=0.488 | +0.0000 [-0.0021, +0.0027] p=0.962 |
| run1 | F1_window_minus_F0_window | +0.012 [-0.012, +0.037] p=0.322 | +0.0027 [-0.0005, +0.0060] p=0.088 | -0.0027 [-0.0060, +0.0005] p=0.088 | -0.0000 [-0.0024, +0.0025] p=0.992 |

Within-question AUROC (mixed questions only; the vote share differs between a correct and an incorrect sibling, so B0 is strong here by construction):

| run | mixed questions / pairs | B0 | B1 | probe alone | F0 | F1 |
|---|---|---|---|---|---|---|
| run4 | 130 / 237 | 0.844 [0.773, 0.908] | 0.848 [0.781, 0.908] | 0.764 [0.704, 0.822] | 0.578 [0.500, 0.652] | 0.738 [0.674, 0.802] |
| run6 | 93 / 253 | 0.783 [0.693, 0.861] | 0.783 [0.693, 0.861] | 0.526 [0.447, 0.598] | 0.482 [0.401, 0.566] | 0.494 [0.421, 0.571] |
| run1 | 61 / 116 | 0.733 [0.611, 0.850] | 0.733 [0.610, 0.845] | 0.621 [0.519, 0.718] | 0.466 [0.368, 0.563] | 0.603 [0.500, 0.702] |

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

B0 = length, logprob, vote_agreement over the 4 completed answers (entropy omitted); B0q = B0 + question features; B1 = B0 + mean probe score over the question's answer-free answers; B1q = B0q + the same.

### run4 / full_population

1164 questions; 117 incorrect pluralities (base accuracy 0.899); automatic failures 24; questions with no answer-free answer (probe imputed) 10.

| model | AUROC | AUACC (higher) | AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.829 | 0.965 | 0.035 | 0.062 | 0.231 |
| B0q | 0.864 | 0.975 | 0.024 | 0.063 | 0.225 |
| B1 | 0.830 | 0.965 | 0.034 | 0.062 | 0.232 |
| B1q | 0.861 | 0.974 | 0.025 | 0.064 | 0.226 |

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | +0.0006 [-0.0021, +0.0035] p=0.636 | -0.0006 [-0.0036, +0.0019] p=0.626 | +0.0004 [-0.0003, +0.0011] p=0.312 | +0.0009 [-0.0010, +0.0029] p=0.312 |
| B1q_minus_B0q | -0.0006 [-0.0014, +0.0001] p=0.108 | +0.0006 [-0.0000, +0.0013] p=0.064 | +0.0005 [+0.0000, +0.0009] p=0.038 | +0.0017 [+0.0004, +0.0030] p=0.008 |

### run4 / valid_plurality

1140 questions; 93 incorrect pluralities (base accuracy 0.918); automatic failures 0; questions with no answer-free answer (probe imputed) 10.

| model | AUROC | AUACC (higher) | AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.778 | 0.965 | 0.034 | 0.063 | 0.233 |
| B0q | 0.830 | 0.975 | 0.024 | 0.062 | 0.223 |
| B1 | 0.780 | 0.966 | 0.033 | 0.063 | 0.234 |
| B1q | 0.827 | 0.975 | 0.024 | 0.063 | 0.224 |

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | +0.0010 [-0.0011, +0.0032] p=0.352 | -0.0010 [-0.0032, +0.0009] p=0.342 | +0.0004 [-0.0006, +0.0014] p=0.420 | +0.0007 [-0.0018, +0.0031] p=0.588 |
| B1q_minus_B0q | -0.0004 [-0.0007, -0.0000] p=0.028 | +0.0004 [+0.0000, +0.0007] p=0.026 | +0.0004 [+0.0001, +0.0008] p=0.022 | +0.0017 [+0.0006, +0.0029] p=0.000 |

### run6 / full_population

2000 questions; 74 incorrect pluralities (base accuracy 0.963); automatic failures 6; questions with no answer-free answer (probe imputed) 108.

| model | AUROC | AUACC (higher) | AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.785 | 0.985 | 0.014 | 0.031 | 0.130 |
| B0q | 0.798 | 0.988 | 0.011 | 0.031 | 0.130 |
| B1 | 0.778 | 0.985 | 0.015 | 0.031 | 0.131 |
| B1q | 0.788 | 0.987 | 0.012 | 0.031 | 0.131 |

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | -0.0008 [-0.0017, +0.0000] p=0.062 | +0.0008 [+0.0000, +0.0017] p=0.046 | +0.0002 [-0.0001, +0.0006] p=0.284 | +0.0012 [+0.0000, +0.0023] p=0.048 |
| B1q_minus_B0q | -0.0009 [-0.0016, -0.0002] p=0.010 | +0.0009 [+0.0002, +0.0017] p=0.004 | +0.0002 [-0.0002, +0.0007] p=0.334 | +0.0014 [-0.0000, +0.0030] p=0.058 |

### run6 / valid_plurality

1994 questions; 68 incorrect pluralities (base accuracy 0.966); automatic failures 0; questions with no answer-free answer (probe imputed) 108.

| model | AUROC | AUACC (higher) | AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.764 | 0.985 | 0.014 | 0.031 | 0.130 |
| B0q | 0.776 | 0.988 | 0.012 | 0.031 | 0.130 |
| B1 | 0.756 | 0.985 | 0.015 | 0.031 | 0.131 |
| B1q | 0.766 | 0.987 | 0.013 | 0.032 | 0.131 |

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | -0.0008 [-0.0017, -0.0001] p=0.030 | +0.0008 [+0.0001, +0.0017] p=0.032 | +0.0002 [-0.0001, +0.0006] p=0.302 | +0.0012 [+0.0000, +0.0024] p=0.048 |
| B1q_minus_B0q | -0.0008 [-0.0016, -0.0002] p=0.006 | +0.0008 [+0.0002, +0.0016] p=0.010 | +0.0002 [-0.0002, +0.0007] p=0.312 | +0.0014 [-0.0000, +0.0030] p=0.056 |

### run1 / full_population

599 questions; 56 incorrect pluralities (base accuracy 0.907); automatic failures 7; questions with no answer-free answer (probe imputed) 3.

| model | AUROC | AUACC (higher) | AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.763 | 0.943 | 0.056 | 0.065 | 0.243 |
| B0q | 0.872 | 0.978 | 0.020 | 0.060 | 0.212 |
| B1 | 0.816 | 0.959 | 0.039 | 0.067 | 0.238 |
| B1q | 0.869 | 0.978 | 0.021 | 0.061 | 0.215 |

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | +0.0161 [+0.0003, +0.0347] p=0.048 | -0.0161 [-0.0340, +0.0002] p=0.052 | +0.0015 [-0.0021, +0.0055] p=0.442 | -0.0049 [-0.0180, +0.0070] p=0.422 |
| B1q_minus_B0q | -0.0007 [-0.0022, +0.0001] p=0.188 | +0.0007 [-0.0001, +0.0025] p=0.192 | +0.0008 [+0.0001, +0.0016] p=0.028 | +0.0024 [-0.0000, +0.0048] p=0.054 |

### run1 / valid_plurality

592 questions; 49 incorrect pluralities (base accuracy 0.917); automatic failures 0; questions with no answer-free answer (probe imputed) 3.

| model | AUROC | AUACC (higher) | AURC (lower) | Brier (lower) | log loss (lower) |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.729 | 0.943 | 0.055 | 0.066 | 0.246 |
| B0q | 0.851 | 0.979 | 0.019 | 0.061 | 0.215 |
| B1 | 0.789 | 0.959 | 0.039 | 0.068 | 0.241 |
| B1q | 0.848 | 0.978 | 0.020 | 0.062 | 0.217 |

| comparison | AUACC | AURC | Brier | log loss |
| --- | --- | --- | --- | --- |
| B1_minus_B0 | +0.0157 [+0.0003, +0.0340] p=0.046 | -0.0157 [-0.0344, +0.0009] p=0.062 | +0.0015 [-0.0020, +0.0056] p=0.434 | -0.0049 [-0.0180, +0.0077] p=0.484 |
| B1q_minus_B0q | -0.0008 [-0.0025, +0.0001] p=0.132 | +0.0008 [-0.0001, +0.0023] p=0.154 | +0.0008 [+0.0001, +0.0016] p=0.040 | +0.0024 [+0.0001, +0.0048] p=0.040 |

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
| run4 | full_population | -0.0006 [-0.0038, +0.0023] p=0.644 | +0.0464 [+0.0266, +0.0675] p=0.000 | -0.0470 [-0.0683, -0.0274] p=0.000 | -0.0199 [-0.0366, -0.0044] p=0.014 |
| run4 | valid_plurality | -0.0010 [-0.0034, +0.0011] p=0.312 | +0.0164 [+0.0052, +0.0281] p=0.002 | -0.0174 [-0.0289, -0.0068] p=0.000 | -0.0056 [-0.0123, +0.0001] p=0.056 |
| run6 | full_population | +0.0008 [+0.0000, +0.0016] p=0.046 | +0.0032 [+0.0005, +0.0065] p=0.018 | -0.0024 [-0.0057, +0.0002] p=0.080 | +0.0007 [-0.0000, +0.0015] p=0.066 |
| run6 | valid_plurality | +0.0008 [+0.0001, +0.0016] p=0.030 | +0.0029 [+0.0005, +0.0060] p=0.016 | -0.0021 [-0.0052, +0.0002] p=0.082 | +0.0007 [-0.0000, +0.0014] p=0.052 |
| run1 | full_population | -0.0161 [-0.0333, -0.0004] p=0.040 | +0.0123 [-0.0156, +0.0408] p=0.418 | -0.0284 [-0.0481, -0.0103] p=0.004 | -0.0375 [-0.0681, -0.0087] p=0.014 |
| run1 | valid_plurality | -0.0157 [-0.0339, +0.0001] p=0.054 | +0.0151 [-0.0140, +0.0414] p=0.286 | -0.0308 [-0.0515, -0.0117] p=0.000 | -0.0278 [-0.0524, -0.0033] p=0.030 |

## Primary contrasts (PLAN.md reading rule: AURC delta B1 − B0, interval entirely below zero)

| contrast | ΔAURC | raw p | Holm p | reading |
|---|---|---|---|---|
| run4 T1 | -0.0005 [-0.0016, +0.0006] p=0.376 | 0.376 | 1.000 | not shown |
| run4 T2 full_population | -0.0006 [-0.0036, +0.0019] p=0.626 | 0.626 | 1.000 | not shown |
| run6 T1 | +0.0003 [-0.0004, +0.0010] p=0.482 | 0.482 | 1.000 | not shown |
| run6 T2 full_population | +0.0008 [+0.0000, +0.0017] p=0.046 | 0.046 | 0.184 | not shown |
