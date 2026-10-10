# Run 1 results

N* = 48. Verdict: **FIXED_PREFIX_INCREMENT_POSITIVE**

- Primary (candidate level, N*=48): surface 0.515, hidden 0.758, both 0.758, increase +0.243 [+0.129, +0.365]; held-out 412 answers, 165 tasks, 45 incorrect
- Shuffled labels 0.433; max leave-one-task-out influence 0.021
- Visible reader at N*: AUROC 0.641 (C=10.0); hidden+surface minus reader +0.118 [+0.032, +0.206]
- Task level at N*: surface 0.554, hidden 0.753, both 0.753, increase +0.199 [+0.039, +0.360]; held-out 234 answers, 78 tasks, 27 incorrect
- Prompt only: surface 0.632, hidden 0.594, both 0.594, increase -0.038 [-0.111, +0.038]; held-out 530 answers, 169 tasks, 68 incorrect
- Final-marker cut, published protocol: surface 0.687, hidden 0.782, both 0.782, increase +0.095 [+0.012, +0.177]; held-out 530 answers, 169 tasks, 68 incorrect

Grid (candidate level):
- N=16: surface 0.492, hidden 0.693, both 0.693, increase +0.201 [+0.095, +0.310]; held-out 493 answers, 169 tasks, 57 incorrect
- N=32: surface 0.518, hidden 0.771, both 0.771, increase +0.253 [+0.136, +0.364]; held-out 444 answers, 167 tasks, 47 incorrect
- N=48: surface 0.515, hidden 0.758, both 0.758, increase +0.243 [+0.129, +0.365]; held-out 412 answers, 165 tasks, 45 incorrect
- N=64: surface 0.592, hidden 0.784, both 0.784, increase +0.192 [+0.082, +0.302]; held-out 393 answers, 161 tasks, 42 incorrect
- N=96: surface 0.598, hidden 0.755, both 0.755, increase +0.156 [+0.021, +0.287]; held-out 319 answers, 146 tasks, 33 incorrect
- N=128: surface 0.599, hidden 0.725, both 0.725, increase +0.126 [-0.017, +0.276]; held-out 265 answers, 135 tasks, 29 incorrect
