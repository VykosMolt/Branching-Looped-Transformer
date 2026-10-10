# Run 6 results (GSM8K training split)

N* = 96. Verdict: **GSM8K_TRAIN_FIXED_PREFIX_INCREMENT_UNRESOLVED**. Headline rule: **MOSTLY_DIFFICULTY**

- Primary (S = log-prob + prompt tokens + reference steps; fixed combined model): surface 0.620, hidden 0.632, both 0.678, increase +0.059 [-0.039, +0.156]; held-out 1907 answers, 551 tasks, 85 incorrect
- Control, end-of-prompt state in the baseline: surface 0.624, hidden 0.632, both 0.658, increase +0.034 [-0.018, +0.088]; held-out 1907 answers, 551 tasks, 85 incorrect
- Control, within-question AUROC: combined 0.518 [0.362, 0.672] (56 pairs, 20 questions); hidden 0.482 [0.310, 0.655] (56 pairs, 20 questions); surface 0.589 [0.429, 0.732] (56 pairs, 20 questions)
- Shuffled labels 0.491
- Visible reader + S: AUROC 0.625; combined minus reader +0.053 [-0.044, +0.150]
- End-of-prompt state alone vs difficulty: surface 0.607, hidden 0.610, both 0.620, increase +0.012 [-0.093, +0.122]; held-out 2301 answers, 586 tasks, 113 incorrect
- Run 1-style (S = log-prob only, arm()): surface 0.569, hidden 0.630, both 0.630, increase +0.061 [-0.050, +0.173]; held-out 1907 answers, 551 tasks, 85 incorrect
- Task level: surface 0.598, hidden 0.579, both 0.643, increase +0.044 [-0.072, +0.163]; held-out 1495 answers, 383 tasks, 76 incorrect
- Also excluding prefixes with the reference number: surface 0.618, hidden 0.633, both 0.681, increase +0.063 [-0.034, +0.163]; held-out 1906 answers, 550 tasks, 84 incorrect
- Accuracy (parseable) 0.946; parseable 0.978; hit 448 0.110

Grid:
- N=16: surface 0.611, hidden 0.568, both 0.592, increase -0.019 [-0.087, +0.055]; held-out 2276 answers, 586 tasks, 101 incorrect
- N=32: surface 0.599, hidden 0.545, both 0.593, increase -0.006 [-0.099, +0.089]; held-out 2220 answers, 582 tasks, 94 incorrect
- N=48: surface 0.601, hidden 0.584, both 0.610, increase +0.009 [-0.088, +0.105]; held-out 2171 answers, 577 tasks, 94 incorrect
- N=64: surface 0.619, hidden 0.617, both 0.650, increase +0.031 [-0.065, +0.124]; held-out 2102 answers, 569 tasks, 92 incorrect
- N=96: surface 0.620, hidden 0.632, both 0.678, increase +0.059 [-0.039, +0.156]; held-out 1907 answers, 551 tasks, 85 incorrect
- N=128: surface 0.633, hidden 0.645, both 0.689, increase +0.057 [-0.041, +0.155]; held-out 1600 answers, 507 tasks, 78 incorrect
- N=160: surface 0.639, hidden 0.681, both 0.721, increase +0.081 [-0.032, +0.193]; held-out 1191 answers, 420 tasks, 63 incorrect
- N=192: surface 0.699, hidden 0.688, both 0.717, increase +0.018 [-0.070, +0.117]; held-out 859 answers, 322 tasks, 49 incorrect
- N=256: not run, below the minimums (159 held-out tasks, 23 incorrect)
