# Run 3 results (GSM8K)

N* = 96. Verdict: **GSM8K_FIXED_PREFIX_NOT_ADJUDICABLE**. Headline rule: **n/a**

- Primary (N*=96; S = log-prob + prompt tokens + reference steps; fixed combined model): not run, below the minimums (168 held-out tasks, 21 incorrect)
- Control, end-of-prompt state in the baseline: not run, below the minimums (168 held-out tasks, 21 incorrect)
- End-of-prompt state alone vs prompt tokens + reference steps: surface 0.445, hidden 0.606, both 0.612, increase +0.167 [+0.042, +0.285]; held-out 740 answers, 190 tasks, 47 incorrect
- Run 1-style (S = log-prob only, sealed arm()): not run, below the minimums (168 held-out tasks, 21 incorrect)
- Task level at N*: not run, below the minimums (119 held-out tasks, 16 incorrect)
- Extended surface at N*: not run, below the minimums (168 held-out tasks, 21 incorrect)
- Also excluding prefixes with the reference number: not run, below the minimums (167 held-out tasks, 20 incorrect)
- Published cut (reference number), 224 tokens: surface 0.803, hidden 0.772, both 0.772, increase -0.031 [-0.081, +0.023]; held-out 764 answers, 191 tasks, 292 incorrect
- Same cut with the candidate's own answer: surface 0.709, hidden 0.811, both 0.811, increase +0.102 [+0.046, +0.157]; held-out 764 answers, 191 tasks, 292 incorrect
- Published cut, difficulty in baseline, fixed combined model: surface 0.817, hidden 0.773, both 0.878, increase +0.061 [+0.036, +0.088]; held-out 764 answers, 191 tasks, 292 incorrect
- Own-answer cut, difficulty in baseline, fixed combined model: surface 0.762, hidden 0.812, both 0.848, increase +0.085 [+0.037, +0.134]; held-out 764 answers, 191 tasks, 292 incorrect
- Published minus own-answer increase: -0.133 [-0.202, -0.065]
- Published cut, published OOF estimator: shortcuts 0.804, both 0.863, increase +0.059 [+0.043, +0.078]; 2400 answers
- Published cut position by label: {'cut_at_gold|correct': 1399, 'cut_at_gold|incorrect': 262, 'cut_at_marker|correct': 51, 'cut_at_marker|incorrect': 67, 'no_marker_no_gold|correct': 4, 'no_marker_no_gold|incorrect': 617}
- Published (224-token) label vs primary label: {'published=False|primary=False': 127, 'published=False|primary=None': 72, 'published=False|primary=True': 747, 'published=True|primary=False': 9, 'published=True|primary=None': 1, 'published=True|primary=True': 1444}
- Accuracy: {'primary_parseable': 0.9415556510528578, 'published_protocol_224': 0.6058333333333333}; parseable 0.970; hit 448 tokens 0.137

Grid (candidate level, S = log-prob + difficulty, fixed combined model):
- N=16: surface 0.542, hidden 0.546, both 0.556, increase +0.014 [-0.109, +0.127]; held-out 728 answers, 190 tasks, 39 incorrect
- N=32: surface 0.550, hidden 0.614, both 0.580, increase +0.029 [-0.089, +0.138]; held-out 702 answers, 188 tasks, 35 incorrect
- N=48: surface 0.580, hidden 0.636, both 0.597, increase +0.017 [-0.117, +0.132]; held-out 685 answers, 185 tasks, 34 incorrect
- N=64: surface 0.550, hidden 0.629, both 0.583, increase +0.034 [-0.117, +0.161]; held-out 657 answers, 183 tasks, 33 incorrect
- N=96: not run, below the minimums (168 held-out tasks, 21 incorrect)
- N=128: not run, below the minimums (157 held-out tasks, 16 incorrect)
- N=160: not run, below the minimums (132 held-out tasks, 16 incorrect)
- N=192: not run, below the minimums (101 held-out tasks, 14 incorrect)
- N=256: not run, below the minimums (51 held-out tasks, 10 incorrect)
