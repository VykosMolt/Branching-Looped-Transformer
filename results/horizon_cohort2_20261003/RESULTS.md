# Run 4 results (Horizon, second cohort, strict labels)

N* = 48. Verdict: **HORIZON2_FIXED_PREFIX_INCREMENT_UNRESOLVED**. Headline rule: **MOSTLY_DIFFICULTY**

- Primary (S = log-prob + difficulty incl. answer class; fixed combined model): surface 0.681, hidden 0.683, both 0.734, increase +0.053 [-0.006, +0.109]; held-out 735 answers, 326 tasks, 59 incorrect
- Control, end-of-prompt state in the baseline: surface 0.709, hidden 0.683, both 0.758, increase +0.049 [+0.008, +0.089]; held-out 735 answers, 326 tasks, 59 incorrect
- Control, within-question AUROC: combined 0.685 [0.577, 0.791] (73 pairs, 43 questions); hidden 0.740 [0.636, 0.831] (73 pairs, 43 questions); surface 0.521 [0.382, 0.662] (73 pairs, 43 questions)
- Shuffled labels 0.541
- Visible reader + S: AUROC 0.720; combined minus reader +0.014 [-0.023, +0.050]
- End-of-prompt state alone vs difficulty: surface 0.709, hidden 0.711, both 0.735, increase +0.026 [-0.010, +0.058]; held-out 996 answers, 345 tasks, 93 incorrect
- Run 1-style (S = log-prob only, arm()): surface 0.530, hidden 0.683, both 0.683, increase +0.153 [+0.047, +0.259]; held-out 735 answers, 326 tasks, 59 incorrect
- Final-answer line, published protocol: surface 0.670, hidden 0.770, both 0.770, increase +0.099 [+0.052, +0.149]; held-out 1008 answers, 347 tasks, 133 incorrect
- Final-answer line, strict labels, difficulty in S: surface 0.751, hidden 0.775, both 0.818, increase +0.067 [+0.030, +0.109]; held-out 996 answers, 345 tasks, 93 incorrect
- Labels, published vs strict: {'published=False|strict=False': 247, 'published=False|strict=None': 118, 'published=False|strict=True': 88, 'published=None|strict=False': 39, 'published=None|strict=None': 1131, 'published=None|strict=True': 162, 'published=True|strict=False': 45, 'published=True|strict=None': 130, 'published=True|strict=True': 2696}
- Accuracy (strict, labelled) 0.899; labelled 0.704; hit 448 0.363

Grid:
- N=16: surface 0.703, hidden 0.619, both 0.720, increase +0.018 [-0.023, +0.059]; held-out 893 answers, 340 tasks, 72 incorrect
- N=32: surface 0.692, hidden 0.661, both 0.719, increase +0.027 [-0.028, +0.081]; held-out 805 answers, 333 tasks, 62 incorrect
- N=48: surface 0.681, hidden 0.683, both 0.734, increase +0.053 [-0.006, +0.109]; held-out 735 answers, 326 tasks, 59 incorrect
- N=64: surface 0.687, hidden 0.698, both 0.738, increase +0.052 [-0.007, +0.112]; held-out 695 answers, 321 tasks, 54 incorrect
- N=96: surface 0.691, hidden 0.707, both 0.730, increase +0.038 [-0.043, +0.120]; held-out 609 answers, 307 tasks, 47 incorrect
- N=128: surface 0.699, hidden 0.759, both 0.792, increase +0.093 [+0.005, +0.185]; held-out 495 answers, 275 tasks, 33 incorrect
