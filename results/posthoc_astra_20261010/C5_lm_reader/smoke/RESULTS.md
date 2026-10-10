# C5 results: fine-tuned LM reader vs hidden-state probe (post hoc)

## deberta (hf-internal-testing/tiny-random-DebertaV2ForSequenceClassification, SMOKE: tiny random model, numbers meaningless)

**r1** (selected {'lr': 1e-05, 'epochs': 3, 'cv_auroc': 0.3636363636363637})
- Across questions, held-out (24 answers, 8 questions, 5 incorrect), probe `arm_combined` 0.905 vs reader 0.789: probe minus reader +0.116 [-0.047, +0.321]
- Across questions, held-out (24 answers, 8 questions, 5 incorrect), probe `hidden` 0.905 vs reader 0.789: probe minus reader +0.116 [-0.047, +0.321]
- Within questions, held-out pairs (9 pairs, 4 questions): probe 0.889 vs reader 0.778: probe minus reader +0.111 [+0.000, +0.333]; reader alone 0.778 [0.333, 1.000]
- Within questions, all pairs out of fold (24 pairs, 12 questions): probe 0.708 vs reader 0.625: probe minus reader +0.083 [-0.100, +0.300]; reader alone 0.625 [0.346, 0.885]

**r1s** (selected {'lr': 5e-06, 'epochs': 2, 'cv_auroc': 0.38541666666666663})
- Across questions, held-out (22 answers, 8 questions, 4 incorrect), probe `arm_combined` 0.722 vs reader 0.847: probe minus reader -0.125 [-0.250, +0.000]
- Across questions, held-out (22 answers, 8 questions, 4 incorrect), probe `hidden` 0.722 vs reader 0.847: probe minus reader -0.125 [-0.250, +0.000]
- Within questions, held-out pairs (9 pairs, 4 questions): probe 0.778 vs reader 0.889: probe minus reader -0.111 [-0.273, +0.000]; reader alone 0.889 [0.667, 1.000]

**r4** (selected {'lr': 5e-06, 'epochs': 2, 'cv_auroc': 0.5619047619047619})
- Across questions, held-out (21 answers, 8 questions, 4 incorrect), probe `hidden` 0.691 vs reader 0.588: probe minus reader +0.103 [-0.476, +0.579]
- Across questions, held-out (21 answers, 8 questions, 4 incorrect), probe `combined` 0.618 vs reader 0.588: probe minus reader +0.029 [-0.417, +0.447]
- Within questions, held-out pairs (7 pairs, 4 questions): probe 0.857 vs reader 0.429: probe minus reader +0.429 [+0.000, +0.857]; reader alone 0.429 [0.000, 0.750]
- Within questions, all pairs out of fold (25 pairs, 12 questions): probe 0.840 vs reader 0.560: probe minus reader +0.280 [-0.045, +0.571]; reader alone 0.560 [0.333, 0.792]

## qwen (trl-internal-testing/tiny-Qwen2ForCausalLM-2.5, SMOKE: tiny random model, numbers meaningless)

**r1** (selected {'lr': 0.0001, 'epochs': 2, 'cv_auroc': 0.47474747474747475})
- Across questions, held-out (24 answers, 8 questions, 5 incorrect), probe `arm_combined` 0.905 vs reader 0.695: probe minus reader +0.211 [-0.053, +0.500]
- Across questions, held-out (24 answers, 8 questions, 5 incorrect), probe `hidden` 0.905 vs reader 0.695: probe minus reader +0.211 [-0.053, +0.500]
- Within questions, held-out pairs (9 pairs, 4 questions): probe 0.889 vs reader 0.889: probe minus reader +0.000 [-0.333, +0.250]; reader alone 0.889 [0.727, 1.000]
- Within questions, all pairs out of fold (24 pairs, 12 questions): probe 0.708 vs reader 0.542: probe minus reader +0.167 [-0.071, +0.409]; reader alone 0.542 [0.304, 0.800]

**r1s** (selected {'lr': 0.0001, 'epochs': 2, 'cv_auroc': 0.42708333333333337})
- Across questions, held-out (22 answers, 8 questions, 4 incorrect), probe `arm_combined` 0.722 vs reader 0.736: probe minus reader -0.014 [-0.348, +0.353]
- Across questions, held-out (22 answers, 8 questions, 4 incorrect), probe `hidden` 0.722 vs reader 0.736: probe minus reader -0.014 [-0.348, +0.353]
- Within questions, held-out pairs (9 pairs, 4 questions): probe 0.778 vs reader 0.778: probe minus reader +0.000 [-0.273, +0.300]; reader alone 0.778 [0.571, 1.000]

**r4** (selected {'lr': 0.0004, 'epochs': 3, 'cv_auroc': 0.5047619047619047})
- Across questions, held-out (21 answers, 8 questions, 4 incorrect), probe `hidden` 0.691 vs reader 0.574: probe minus reader +0.118 [-0.375, +0.574]
- Across questions, held-out (21 answers, 8 questions, 4 incorrect), probe `combined` 0.618 vs reader 0.574: probe minus reader +0.044 [-0.412, +0.583]
- Within questions, held-out pairs (7 pairs, 4 questions): probe 0.857 vs reader 0.857: probe minus reader +0.000 [-0.500, +0.500]; reader alone 0.857 [0.400, 1.000]
- Within questions, all pairs out of fold (25 pairs, 12 questions): probe 0.840 vs reader 0.560: probe minus reader +0.280 [-0.038, +0.586]; reader alone 0.560 [0.261, 0.840]

