# Run 8 results (Horizon, Ouro-2.6B-Thinking, Run 4's questions)

N* = 32. Verdict: **UNRESOLVED**. Run 4's across-question verdict: HORIZON_THINKING_FIXED_PREFIX_INCREMENT_UNRESOLVED; headline rule: MOSTLY_DIFFICULTY

- Primary, within-question AUROC of the hidden-state probe, out of fold (5 hash folds): 0.569 [0.462, 0.673] (102 pairs, 74 questions); 1626 answers, 146 incorrect
- Combined (hidden + S): 0.520 [0.410, 0.634] (102 pairs, 74 questions); S only: 0.539 [0.438, 0.644] (102 pairs, 74 questions)
- TF-IDF reader, out of fold: 0.539 [0.434, 0.641] (102 pairs, 74 questions); hidden minus reader +0.029 [-0.100, +0.155]
- Commit timing: correct answer commits later in 0.647 of 102 pairs; commit position alone 0.647 [0.538, 0.752] (102 pairs, 74 questions); by group {'correct_commits_later': {'n_pairs': 66, 'hidden': 0.696969696969697, 'combined': 0.6363636363636364, 'surface': 0.5303030303030303, 'reader': 0.6060606060606061}, 'correct_commits_earlier': {'n_pairs': 36, 'hidden': 0.3333333333333333, 'combined': 0.3055555555555556, 'surface': 0.5555555555555556, 'reader': 0.4166666666666667}, 'same_position': {'n_pairs': 0, 'hidden': None, 'combined': None, 'surface': None, 'reader': None}}
- Within-question, out of fold, at N = 48: 0.575 [0.460, 0.695] (87 pairs, 64 questions)
- Chronological labels: 0 answers change label; cohort [1626, 1626]; hidden within-question unchanged (same cohort and labels)
- Inputs identical to Run 4: {'answers_compared': 4656, 'prompt_ids_equal': 4656, 'seeds_equal': 4656}
- Thinking minus RLTT (own_n_star: N = 32 vs 48): -0.149 [-0.191, -0.104]; RLTT 0.717 [0.654, 0.779] (237 pairs, 130 questions)
- Thinking minus RLTT (both_at_48: N = 48 vs 48): -0.143 [-0.196, -0.082]; RLTT 0.717 [0.654, 0.779] (237 pairs, 130 questions)
- Run 4's primary (across questions, S = log-prob + difficulty): surface 0.689, hidden 0.653, both 0.709, increase +0.021 [-0.034, +0.076]; held-out 464 answers, 278 tasks, 53 incorrect
- End-of-prompt state in the baseline: surface 0.660, hidden 0.653, both 0.683, increase +0.023 [-0.030, +0.081]; held-out 464 answers, 278 tasks, 53 incorrect
- Held-out only, within-question: combined 0.450 [0.286, 0.615] (40 pairs, 26 questions); hidden 0.500 [0.308, 0.682] (40 pairs, 26 questions); surface 0.525 [0.364, 0.698] (40 pairs, 26 questions)
- Shuffled labels 0.459
- Visible reader + S (held-out): AUROC 0.600; combined minus reader +0.109 [+0.031, +0.188]
- End-of-prompt state alone vs difficulty: surface 0.717, hidden 0.707, both 0.718, increase +0.001 [-0.044, +0.043]; held-out 644 answers, 317 tasks, 85 incorrect
- Run 1-style (S = log-prob only, arm()): surface 0.500, hidden 0.653, both 0.653, increase +0.153 [+0.050, +0.253]; held-out 464 answers, 278 tasks, 53 incorrect
- Final-answer line, published protocol: surface 0.701, hidden 0.726, both 0.726, increase +0.024 [-0.025, +0.073]; held-out 699 answers, 324 tasks, 134 incorrect
- Final-answer line, strict labels, difficulty in S: surface 0.757, hidden 0.770, both 0.807, increase +0.049 [+0.001, +0.101]; held-out 644 answers, 317 tasks, 85 incorrect
- Labels, published vs strict: {'published=False|strict=False': 187, 'published=False|strict=None': 188, 'published=False|strict=True': 63, 'published=None|strict=False': 24, 'published=None|strict=None': 2130, 'published=None|strict=True': 91, 'published=True|strict=False': 27, 'published=True|strict=None': 142, 'published=True|strict=True': 1804}
- Accuracy (strict, labelled) 0.892; labelled 0.472; hit 448 0.625

Grid:
- N=16: surface 0.707, hidden 0.672, both 0.724, increase +0.017 [-0.030, +0.061]; held-out 529 answers, 295 tasks, 67 incorrect
- N=32: surface 0.689, hidden 0.653, both 0.709, increase +0.021 [-0.034, +0.076]; held-out 464 answers, 278 tasks, 53 incorrect
- N=48: surface 0.688, hidden 0.696, both 0.738, increase +0.050 [-0.015, +0.116]; held-out 432 answers, 270 tasks, 48 incorrect
- N=64: surface 0.680, hidden 0.673, both 0.722, increase +0.042 [-0.025, +0.105]; held-out 409 answers, 262 tasks, 45 incorrect
- N=96: surface 0.714, hidden 0.729, both 0.791, increase +0.077 [+0.002, +0.158]; held-out 367 answers, 253 tasks, 43 incorrect
- N=128: surface 0.720, hidden 0.721, both 0.767, increase +0.047 [-0.044, +0.150]; held-out 325 answers, 236 tasks, 35 incorrect
