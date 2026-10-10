# C5 results: fine-tuned LM reader vs hidden-state probe (post hoc)

## deberta (microsoft/deberta-v3-large)

**r1** (selected {'lr': 1e-05, 'epochs': 4, 'cv_auroc': 0.6006568575932737})
- Across questions, held-out (412 answers, 165 questions, 45 incorrect), probe `arm_combined` 0.758 vs reader 0.770: probe minus reader -0.012 [-0.081, +0.054]
- Across questions, held-out (412 answers, 165 questions, 45 incorrect), probe `hidden` 0.758 vs reader 0.770: probe minus reader -0.012 [-0.081, +0.054]
- Within questions, held-out pairs (57 pairs, 30 questions): probe 0.719 vs reader 0.754: probe minus reader -0.035 [-0.179, +0.086]; reader alone 0.754 [0.620, 0.873]
- Within questions, all pairs out of fold (209 pairs, 108 questions): probe 0.684 vs reader 0.589: probe minus reader +0.096 [+0.029, +0.163]; reader alone 0.589 [0.507, 0.668]

**r1s** (selected {'lr': 2e-05, 'epochs': 4, 'cv_auroc': 0.613063236047107})
- Across questions, held-out (399 answers, 163 questions, 29 incorrect), probe `arm_combined` 0.777 vs reader 0.523: probe minus reader +0.254 [+0.094, +0.402]
- Across questions, held-out (399 answers, 163 questions, 29 incorrect), probe `hidden` 0.777 vs reader 0.523: probe minus reader +0.254 [+0.095, +0.403]
- Within questions, held-out pairs (37 pairs, 18 questions): probe 0.622 vs reader 0.419: probe minus reader +0.203 [-0.024, +0.450]; reader alone 0.419 [0.261, 0.603]

**r4** (selected {'lr': 5e-06, 'epochs': 4, 'cv_auroc': 0.6614007104732954})
- Across questions, held-out (735 answers, 326 questions, 59 incorrect), probe `hidden` 0.683 vs reader 0.701: probe minus reader -0.017 [-0.089, +0.053]
- Across questions, held-out (735 answers, 326 questions, 59 incorrect), probe `combined` 0.734 vs reader 0.701: probe minus reader +0.033 [-0.038, +0.103]
- Within questions, held-out pairs (73 pairs, 43 questions): probe 0.740 vs reader 0.719: probe minus reader +0.021 [-0.087, +0.121]; reader alone 0.719 [0.604, 0.821]
- Within questions, all pairs out of fold (237 pairs, 130 questions): probe 0.738 vs reader 0.738: probe minus reader +0.000 [-0.057, +0.059]; reader alone 0.738 [0.672, 0.797]

## qwen (Qwen/Qwen2.5-1.5B)

**r1** (selected {'lr': 0.0002, 'epochs': 4, 'cv_auroc': 0.5706988964792432})
- Across questions, held-out (412 answers, 165 questions, 45 incorrect), probe `arm_combined` 0.758 vs reader 0.552: probe minus reader +0.206 [+0.074, +0.324]
- Across questions, held-out (412 answers, 165 questions, 45 incorrect), probe `hidden` 0.758 vs reader 0.552: probe minus reader +0.206 [+0.075, +0.324]
- Within questions, held-out pairs (57 pairs, 30 questions): probe 0.719 vs reader 0.456: probe minus reader +0.263 [+0.057, +0.448]; reader alone 0.456 [0.310, 0.604]
- Within questions, all pairs out of fold (209 pairs, 108 questions): probe 0.684 vs reader 0.557: probe minus reader +0.127 [+0.034, +0.221]; reader alone 0.557 [0.472, 0.637]

**r1s** (selected {'lr': 0.0001, 'epochs': 4, 'cv_auroc': 0.5920458909370201})
- Across questions, held-out (399 answers, 163 questions, 29 incorrect), probe `arm_combined` 0.777 vs reader 0.617: probe minus reader +0.161 [-0.008, +0.313]
- Across questions, held-out (399 answers, 163 questions, 29 incorrect), probe `hidden` 0.777 vs reader 0.617: probe minus reader +0.161 [-0.008, +0.313]
- Within questions, held-out pairs (37 pairs, 18 questions): probe 0.622 vs reader 0.541: probe minus reader +0.081 [-0.132, +0.281]; reader alone 0.541 [0.351, 0.743]

**r4** (selected {'lr': 0.0002, 'epochs': 4, 'cv_auroc': 0.6303661058287089})
- Across questions, held-out (735 answers, 326 questions, 59 incorrect), probe `hidden` 0.683 vs reader 0.651: probe minus reader +0.032 [-0.026, +0.090]
- Across questions, held-out (735 answers, 326 questions, 59 incorrect), probe `combined` 0.734 vs reader 0.651: probe minus reader +0.083 [+0.022, +0.144]
- Within questions, held-out pairs (73 pairs, 43 questions): probe 0.740 vs reader 0.616: probe minus reader +0.123 [+0.016, +0.221]; reader alone 0.616 [0.506, 0.724]
- Within questions, all pairs out of fold (237 pairs, 130 questions): probe 0.738 vs reader 0.643: probe minus reader +0.095 [+0.019, +0.170]; reader alone 0.643 [0.571, 0.712]

