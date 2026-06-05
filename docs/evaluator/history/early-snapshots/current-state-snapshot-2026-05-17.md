# Post-v10 Synthesis v7: Actual State And Next Step

**Date:** 2026-05-17  
**Refactored:** 2026-05-18 for readability  
**Raw original:** `raw_archive_2026-05-18/post_v10_synthesis_2026-05-17_v7_actual_state_and_next.md`

## Current Position

The project has enough evidence to use tiny taps for local planning, but not enough to collapse everything into one universal head.

Current architecture stance:

- HH-trained taps are the general preference-domain baseline.
- Code-specific tiny heads are needed for strong strict-clean code branch selection.
- Reasoning multiple-choice should become a third objective eval domain.
- GRU is not the default; keep it as a control or escalation path.
- Full MATH gate-scale remains deferred.

## Framing Rules

The evaluator/tap is relational and pairwise. It selects between branches.

Candidate labels come from:

- HH preference labels.
- exact-answer verifiers.
- unit tests.
- answer keys.

Tap outputs are not labels. NoNorm can be discussed as scalar-readable branch ranking in objective domains, but HH remains preference-like and noisy.

## Timeline Of Decision-Relevant Results

| step | result | decision effect |
| --- | --- | --- |
| math gate prep | demoted | full MATH gate-scale is deferred |
| math validity probe | old mixed pilot truncation-confounded | do not lean on old math result |
| clean GSM8K micro | promising transfer | justified clean expanded GSM8K |
| expanded clean GSM8K | HH-trained transfer GOOD; GRU weak | linear tiny taps remain default |
| code v1 | all-correct collapse | wrapper too strong for branch data |
| code v2 dirty diagnostic | signal but dirty harness | harden wrapper/task taxonomy |
| patched code v2-mini | runnable diagnostic GOOD | HH-trained taps transfer to easier code setting |
| strict-clean screening | 6 ready tasks | within-task pairing is bottleneck |
| strict-clean HH transfer | WEAK on OLD6 | HH projection weak for near-miss code |
| code-specific control | GOOD on OLD6 | code signal exists in states |
| strict-clean expansion | 10 more ready tasks | enough for ALL16 comparison |
| expanded HH-vs-code comparison | CODE_SPECIFIC_ADVANTAGE | use code specialist |
| inverse code-to-HH transfer | partial on small splits, weak on all-200 | code head is not general replacement |
| cross-domain audit | DOMAIN_SPECIALISTS_NEEDED | design general plus specialists |
| reasoning pilot | GOOD | add reasoning eval domain |

## Key Numbers

Expanded strict-clean ALL16:

| head | config | architecture | top1 | pairwise | cycle |
| --- | --- | --- | ---: | ---: | ---: |
| HH-trained | `47_mean` | AntisymLinear | 0.750 | 0.600 | 0.000 |
| code-trained | `36_L4` | AntisymLinear | 0.875 | 0.833 | 0.000 |

HH all-200 diagnostic:

| head | config | architecture | accuracy |
| --- | --- | --- | ---: |
| HH-trained | `47_concat_L1_L4` | NoNorm | 0.855 |
| code-trained | `47_L4` | NoNorm | 0.535 |

Reasoning pilot:

| head | config | architecture | top1 | pairwise | cycle |
| --- | --- | --- | ---: | ---: | ---: |
| code-trained | `24_L4` | AntisymLinear | 1.000 | 1.000 | 0.000 |
| HH-trained | `36_mean` | AntisymLinear | 0.960 | 0.986 | 0.000 |

## Current Bottleneck

The bottleneck is branch curriculum and task selection, especially finding same-task correct-vs-near-miss branches.

Global pools contain correct and incorrect code candidates. What is hard is building clean same-task comparisons without label leakage, wrapper artifacts, or all-correct collapse.

## Recommended Next Move

Add reasoning as a third objective eval domain and update the BG Phase 1 plan around a general HH head plus domain-specific specialists.

Practical next tasks:

1. Build a harder reasoning branch validation set.
2. Add controller-policy simulation for HH/code/reasoning head disagreement.
3. Keep strict-clean code as the main specialist testbed.
4. Keep MATH gate-scale deferred until data yield and generation verbosity are under control.

## Source Reports

- `artifacts/reports/probes/bg_cross_domain_reasoning_audit_2026-05-17_summary.md`
- `artifacts/reports/probes/bg_fixed_config_cross_domain_audit_2026-05-17.md`
- `artifacts/reports/probes/expanded_strict_clean_code_projection_comparison_2026-05-17_summary.md`
- `artifacts/reports/probes/code_trained_vs_hh_trained_random20_hh_splits_2026-05-17.md`
- `artifacts/reports/probes/reasoning_branch_transfer_2026-05-17.md`

## Hard reasoning natural-distractor validation (2026-05-17)

- REASONING_DISTRACTOR_SET_VERDICT: `READY`
- REASONING_DISTRACTOR_FEATURE_VERDICT: `READY`
- REASONING_DISTRACTOR_TRANSFER_VERDICT: `GOOD`
- REASONING_SPECIALIST_VERDICT: `GENERAL_SUFFICIENT`
- REASONING_DIFFICULTY_VERDICT: `DISTRACTORS_HARDER`
- dataset/task counts: `{'ai2_arc_challenge': 30, 'openbookqa': 30}`
- random_top1_baseline: `0.25`
- best HH row: `{'family': 'HH', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5666666666666667, 'over_random': 0.31666666666666665, 'pairwise': 0.7333333333333333, 'condorcet': 0.5666666666666667, 'cycle': 0.0, 'margin_mean': 0.33211478032171726, 'margin_std': 0.3502592859723129}`
- best code row: `{'family': 'CODE', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.6, 'over_random': 0.35, 'pairwise': 0.7222222222222222, 'condorcet': 0.6, 'cycle': 0.0, 'margin_mean': 0.5857972353696823, 'margin_std': 0.5569268932391405}`
- best NoNorm row: `{'family': 'HH', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5666666666666667, 'over_random': 0.31666666666666665, 'pairwise': 0.7333333333333333, 'condorcet': 0.5666666666666667, 'cycle': 0.0, 'margin_mean': 0.33211478032171726, 'margin_std': 0.3502592859723129}`
- best AntisymLinear row: `{'family': 'HH', 'config': '36_L4', 'architecture': 'AntisymLinear', 'top1': 0.4166666666666667, 'over_random': 0.16666666666666669, 'pairwise': 0.6611111111111111, 'condorcet': 0.4166666666666667, 'cycle': 0.0, 'margin_mean': 1.0988494743903479, 'margin_std': 0.9490593444718143}`
- generated-vs-distractor comparison: `{'REASONING_DIFFICULTY_VERDICT': 'DISTRACTORS_HARDER', 'generated_n_tournaments': 25, 'distractor_n_tournaments': 60, 'generated_random_top1_baseline': 0.4400000047683716, 'distractor_random_top1_baseline': 0.25, 'generated_best_hh': {'family': 'HH', 'config': '36_mean', 'architecture': 'AntisymLinear', 'top1': 0.96, 'over_random': 0.5199999952316283, 'pairwise': 0.9861111111111112, 'cycle': 0.0}, 'distractor_best_hh': {'family': 'HH', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5666666666666667, 'over_random': 0.31666666666666665, 'pairwise': 0.7333333333333333, 'condorcet': 0.5666666666666667, 'cycle': 0.0, 'margin_mean': 0.33211478032171726, 'margin_std': 0.3502592859723129}, 'generated_best_code': {'family': 'CODE', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 1.0, 'over_random': 0.5599999952316284, 'pairwise': 1.0, 'cycle': 0.0}, 'distractor_best_code': {'family': 'CODE', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.6, 'over_random': 0.35, 'pairwise': 0.7222222222222222, 'condorcet': 0.6, 'cycle': 0.0, 'margin_mean': 0.5857972353696823, 'margin_std': 0.5569268932391405}, 'generated_best_nonorm': 'not_reported', 'distractor_best_nonorm': {'family': 'HH', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5666666666666667, 'over_random': 0.31666666666666665, 'pairwise': 0.7333333333333333, 'condorcet': 0.5666666666666667, 'cycle': 0.0, 'margin_mean': 0.33211478032171726, 'margin_std': 0.3502592859723129}, 'generated_best_antisymlinear': 'not_reported', 'distractor_best_antisymlinear': {'family': 'HH', 'config': '36_L4', 'architecture': 'AntisymLinear', 'top1': 0.4166666666666667, 'over_random': 0.16666666666666669, 'pairwise': 0.6611111111111111, 'condorcet': 0.4166666666666667, 'cycle': 0.0, 'margin_mean': 1.0988494743903479, 'margin_std': 0.9490593444718143}, 'generated_best_overall': {'family': 'CODE', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 1.0, 'over_random': 0.5599999952316284, 'pairwise': 1.0, 'cycle': 0.0}, 'distractor_best_overall': {'family': 'HH', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5666666666666667, 'over_random': 0.31666666666666665, 'pairwise': 0.7333333333333333, 'condorcet': 0.5666666666666667, 'cycle': 0.0, 'margin_mean': 0.33211478032171726, 'margin_std': 0.3502592859723129}, 'generated_data_counts': {'REASONING_BRANCH_DATA_VERDICT': 'READY', 'tasks_seen': 30, 'mixed_tournaments': 25, 'candidates_total': 120, 'label_counts': {'incorrect': 64, 'correct': 47, 'unparseable': 9}, 'unparseable_rate': 0.075, 'dataset_counts': {'ai2_arc_challenge': 80, 'openbookqa': 40}}, 'interpretation': 'Natural distractors reduced transfer performance relative to generated answer branches.'}`
- reasoning third objective eval domain: `still_supported`
- reasoning specialist justified yet: `not_yet_general_head_sufficient`
- full reports: `artifacts/reports/probes/reasoning_natural_distractor_audit_2026-05-17_summary.md`, `artifacts/reports/probes/reasoning_natural_distractor_transfer_2026-05-17.md`, `artifacts/reports/probes/reasoning_generated_vs_distractor_comparison_2026-05-17.md`
- interpretation: Natural distractors reduced performance relative to generated branches and are a better stress test for reasoning readouts.

## Reasoning trace near-miss + code taps on math/logic (2026-05-17)

- REASONING_TRACE_TASK_SET_VERDICT: `READY`
- REASONING_TRACE_DATA_VERDICT: `PARTIAL`
- REASONING_TRACE_FEATURE_VERDICT: `READY`
- REASONING_TRACE_TRANSFER_VERDICT: `GOOD`
- REASONING_TRACE_SPECIALIST_VERDICT: `GENERAL_SUFFICIENT`
- REASONING_TRACE_DIFFICULTY_VERDICT: `DISTRACTORS_HARDER`
- CODE_TAPS_ON_MATH_VERDICT: `GOOD`
- CODE_TAPS_ON_LOGIC_VERDICT: `GOOD`
- dataset/task counts: `{'ai2_arc_challenge': 15, 'openbookqa': 15}`
- best HH row: `{'family': 'HH', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 0.7083333333333334, 'over_random': 0.42013888433575636, 'pairwise': 0.8524590163934426, 'condorcet': 0.7083333333333334, 'cycle': 0.0, 'margin_mean': 1.3928159878899653, 'margin_std': 1.1117851251122828}`
- best code row: `{'family': 'CODE', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 0.75, 'over_random': 0.461805551002423, 'pairwise': 0.8688524590163934, 'condorcet': 0.75, 'cycle': 0.0, 'margin_mean': 2.297035602852702, 'margin_std': 1.4866195021987771}`
- best NoNorm row: `{'family': 'CODE', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.6666666666666666, 'over_random': 0.3784722176690896, 'pairwise': 0.8524590163934426, 'condorcet': 0.6666666666666666, 'cycle': 0.0, 'margin_mean': 1.19206016138196, 'margin_std': 0.9531564312749697}`
- best AntisymLinear row: `{'family': 'CODE', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 0.75, 'over_random': 0.461805551002423, 'pairwise': 0.8688524590163934, 'condorcet': 0.75, 'cycle': 0.0, 'margin_mean': 2.297035602852702, 'margin_std': 1.4866195021987771}`
- comparison to natural distractor reasoning: `{'REASONING_TRACE_DIFFICULTY_VERDICT': 'DISTRACTORS_HARDER', 'generated_answer_branches': {'n_tournaments': 25, 'random_top1_baseline': 0.4400000047683716, 'best_hh': {'family': 'HH', 'config': '36_mean', 'architecture': 'AntisymLinear', 'top1': 0.96, 'over_random': 0.5199999952316283, 'pairwise': 0.9861111111111112, 'cycle': 0.0}, 'best_code': {'family': 'CODE', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 1.0, 'over_random': 0.5599999952316284, 'pairwise': 1.0, 'cycle': 0.0}, 'best_nonorm': None, 'best_antisymlinear': None, 'best_overall': {'family': 'CODE', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 1.0, 'over_random': 0.5599999952316284, 'pairwise': 1.0, 'cycle': 0.0}, 'verdict': 'GOOD', 'specialist_verdict': None}, 'natural_distractors': {'n_tournaments': 60, 'random_top1_baseline': 0.25, 'best_hh': {'family': 'HH', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5666666666666667, 'over_random': 0.31666666666666665, 'pairwise': 0.7333333333333333, 'condorcet': 0.5666666666666667, 'cycle': 0.0, 'margin_mean': 0.33211478032171726, 'margin_std': 0.3502592859723129}, 'best_code': {'family': 'CODE', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.6, 'over_random': 0.35, 'pairwise': 0.7222222222222222, 'condorcet': 0.6, 'cycle': 0.0, 'margin_mean': 0.5857972353696823, 'margin_std': 0.5569268932391405}, 'best_nonorm': {'family': 'HH', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5666666666666667, 'over_random': 0.31666666666666665, 'pairwise': 0.7333333333333333, 'condorcet': 0.5666666666666667, 'cycle': 0.0, 'margin_mean': 0.33211478032171726, 'margin_std': 0.3502592859723129}, 'best_antisymlinear': {'family': 'HH', 'config': '36_L4', 'architecture': 'AntisymLinear', 'top1': 0.4166666666666667, 'over_random': 0.16666666666666669, 'pairwise': 0.6611111111111111, 'condorcet': 0.4166666666666667, 'cycle': 0.0, 'margin_mean': 1.0988494743903479, 'margin_std': 0.9490593444718143}, 'best_overall': {'family': 'HH', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5666666666666667, 'over_random': 0.31666666666666665, 'pairwise': 0.7333333333333333, 'condorcet': 0.5666666666666667, 'cycle': 0.0, 'margin_mean': 0.33211478032171726, 'margin_std': 0.3502592859723129}, 'verdict': 'GOOD', 'specialist_verdict': 'GENERAL_SUFFICIENT'}, 'generated_reasoning_traces': {'n_tournaments': 24, 'random_top1_baseline': 0.288194448997577, 'best_hh': {'family': 'HH', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 0.7083333333333334, 'over_random': 0.42013888433575636, 'pairwise': 0.8524590163934426, 'condorcet': 0.7083333333333334, 'cycle': 0.0, 'margin_mean': 1.3928159878899653, 'margin_std': 1.1117851251122828}, 'best_code': {'family': 'CODE', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 0.75, 'over_random': 0.461805551002423, 'pairwise': 0.8688524590163934, 'condorcet': 0.75, 'cycle': 0.0, 'margin_mean': 2.297035602852702, 'margin_std': 1.4866195021987771}, 'best_nonorm': {'family': 'CODE', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.6666666666666666, 'over_random': 0.3784722176690896, 'pairwise': 0.8524590163934426, 'condorcet': 0.6666666666666666, 'cycle': 0.0, 'margin_mean': 1.19206016138196, 'margin_std': 0.9531564312749697}, 'best_antisymlinear': {'family': 'CODE', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 0.75, 'over_random': 0.461805551002423, 'pairwise': 0.8688524590163934, 'condorcet': 0.75, 'cycle': 0.0, 'margin_mean': 2.297035602852702, 'margin_std': 1.4866195021987771}, 'best_overall': {'family': 'CODE', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 0.75, 'over_random': 0.461805551002423, 'pairwise': 0.8688524590163934, 'condorcet': 0.75, 'cycle': 0.0, 'margin_mean': 2.297035602852702, 'margin_std': 1.4866195021987771}, 'verdict': 'GOOD', 'specialist_verdict': 'GENERAL_SUFFICIENT'}, 'interpretation': 'Natural answer distractors remain harder than generated reasoning traces.'}`
- comparison to clean GSM8K/code taps: `{'CODE_TAPS_ON_MATH_VERDICT': 'GOOD', 'CODE_TAPS_ON_LOGIC_VERDICT': 'GOOD', 'domains': [{'domain': 'CLEAN_GSM8K_EXPANDED', 'n_tournaments': 28, 'n_candidates': 79, 'random_top1_baseline': 0.5625000074505806, 'best_code': {'domain': 'CLEAN_GSM8K_EXPANDED', 'family': 'CODE', 'config': '24_mean', 'architecture': 'AntisymLinear', 'top1': 0.8928571428571429, 'over_random': 0.3303571354065623, 'pairwise': 0.7962962962962963, 'cycle': 0.0, 'margin_mean': 1.7689685566084725, 'margin_std': 1.411621785660656}, 'best_hh': {'domain': 'CLEAN_GSM8K_EXPANDED', 'family': 'HH', 'config': '24_L4', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.75, 'over_random': 0.1874999925494194, 'pairwise': 0.7407407407407407, 'cycle': 0.0, 'margin_mean': 0.012588573902446245, 'margin_std': 0.011785265045384234}, 'best_overall': {'domain': 'CLEAN_GSM8K_EXPANDED', 'family': 'CODE', 'config': '24_mean', 'architecture': 'AntisymLinear', 'top1': 0.8928571428571429, 'over_random': 0.3303571354065623, 'pairwise': 0.7962962962962963, 'cycle': 0.0, 'margin_mean': 1.7689685566084725, 'margin_std': 1.411621785660656}, 'best_code_verdict': 'GOOD'}, {'domain': 'REASONING_NATURAL_DISTRACTOR', 'n_tournaments': 60, 'n_candidates': 240, 'random_top1_baseline': 0.25, 'best_code': {'domain': 'REASONING_NATURAL_DISTRACTOR', 'family': 'CODE', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.6, 'over_random': 0.35, 'pairwise': 0.7222222222222222, 'cycle': 0.0, 'margin_mean': 0.5857972353696823, 'margin_std': 0.5569268932391405}, 'best_hh': {'domain': 'REASONING_NATURAL_DISTRACTOR', 'family': 'HH', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5666666666666667, 'over_random': 0.31666666666666665, 'pairwise': 0.7333333333333333, 'cycle': 0.0, 'margin_mean': 0.33211478032171726, 'margin_std': 0.3502592859723129}, 'best_overall': {'domain': 'REASONING_NATURAL_DISTRACTOR', 'family': 'HH', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5666666666666667, 'over_random': 0.31666666666666665, 'pairwise': 0.7333333333333333, 'cycle': 0.0, 'margin_mean': 0.33211478032171726, 'margin_std': 0.3502592859723129}, 'best_code_verdict': 'GOOD'}, {'domain': 'REASONING_TRACE', 'n_tournaments': 24, 'n_candidates': 85, 'random_top1_baseline': 0.288194448997577, 'best_code': {'domain': 'REASONING_TRACE', 'family': 'CODE', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 0.75, 'over_random': 0.461805551002423, 'pairwise': 0.8688524590163934, 'cycle': 0.0, 'margin_mean': 2.297035602852702, 'margin_std': 1.4866195021987771}, 'best_hh': {'domain': 'REASONING_TRACE', 'family': 'HH', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 0.7083333333333334, 'over_random': 0.42013888433575636, 'pairwise': 0.8524590163934426, 'cycle': 0.0, 'margin_mean': 1.3928159878899653, 'margin_std': 1.1117851251122828}, 'best_overall': {'domain': 'REASONING_TRACE', 'family': 'CODE', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 0.75, 'over_random': 0.461805551002423, 'pairwise': 0.8688524590163934, 'cycle': 0.0, 'margin_mean': 2.297035602852702, 'margin_std': 1.4866195021987771}, 'best_code_verdict': 'GOOD'}, {'domain': 'LOGIC_COMBINED', 'n_tournaments': 84, 'n_candidates': 325, 'random_top1_baseline': 0.26091269971359343, 'best_code': {'domain': 'LOGIC_COMBINED', 'family': 'CODE', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.6190476190476191, 'over_random': 0.35813491933402564, 'pairwise': 0.7551867219917012, 'cycle': 0.0, 'margin_mean': 0.759015214230333, 'margin_std': 0.7457431940723517}, 'best_hh': {'domain': 'LOGIC_COMBINED', 'family': 'HH', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5476190476190477, 'over_random': 0.28670634790545424, 'pairwise': 0.7302904564315352, 'cycle': 0.0, 'margin_mean': 0.3825706510494153, 'margin_std': 0.3891915641615466}, 'best_overall': {'domain': 'LOGIC_COMBINED', 'family': 'CODE', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.6190476190476191, 'over_random': 0.35813491933402564, 'pairwise': 0.7551867219917012, 'cycle': 0.0, 'margin_mean': 0.759015214230333, 'margin_std': 0.7457431940723517}, 'best_code_verdict': 'GOOD'}], 'best_code_rows': {'CLEAN_GSM8K_EXPANDED': {'domain': 'CLEAN_GSM8K_EXPANDED', 'family': 'CODE', 'config': '24_mean', 'architecture': 'AntisymLinear', 'top1': 0.8928571428571429, 'over_random': 0.3303571354065623, 'pairwise': 0.7962962962962963, 'cycle': 0.0, 'margin_mean': 1.7689685566084725, 'margin_std': 1.411621785660656}, 'REASONING_NATURAL_DISTRACTOR': {'domain': 'REASONING_NATURAL_DISTRACTOR', 'family': 'CODE', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.6, 'over_random': 0.35, 'pairwise': 0.7222222222222222, 'cycle': 0.0, 'margin_mean': 0.5857972353696823, 'margin_std': 0.5569268932391405}, 'REASONING_TRACE': {'domain': 'REASONING_TRACE', 'family': 'CODE', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 0.75, 'over_random': 0.461805551002423, 'pairwise': 0.8688524590163934, 'cycle': 0.0, 'margin_mean': 2.297035602852702, 'margin_std': 1.4866195021987771}, 'LOGIC_COMBINED': {'domain': 'LOGIC_COMBINED', 'family': 'CODE', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.6190476190476191, 'over_random': 0.35813491933402564, 'pairwise': 0.7551867219917012, 'cycle': 0.0, 'margin_mean': 0.759015214230333, 'margin_std': 0.7457431940723517}}, 'best_hh_rows': {'CLEAN_GSM8K_EXPANDED': {'domain': 'CLEAN_GSM8K_EXPANDED', 'family': 'HH', 'config': '24_L4', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.75, 'over_random': 0.1874999925494194, 'pairwise': 0.7407407407407407, 'cycle': 0.0, 'margin_mean': 0.012588573902446245, 'margin_std': 0.011785265045384234}, 'REASONING_NATURAL_DISTRACTOR': {'domain': 'REASONING_NATURAL_DISTRACTOR', 'family': 'HH', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5666666666666667, 'over_random': 0.31666666666666665, 'pairwise': 0.7333333333333333, 'cycle': 0.0, 'margin_mean': 0.33211478032171726, 'margin_std': 0.3502592859723129}, 'REASONING_TRACE': {'domain': 'REASONING_TRACE', 'family': 'HH', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 0.7083333333333334, 'over_random': 0.42013888433575636, 'pairwise': 0.8524590163934426, 'cycle': 0.0, 'margin_mean': 1.3928159878899653, 'margin_std': 1.1117851251122828}, 'LOGIC_COMBINED': {'domain': 'LOGIC_COMBINED', 'family': 'HH', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5476190476190477, 'over_random': 0.28670634790545424, 'pairwise': 0.7302904564315352, 'cycle': 0.0, 'margin_mean': 0.3825706510494153, 'margin_std': 0.3891915641615466}}, 'blockers': []}`
- full reports: `artifacts/reports/probes/reasoning_trace_and_code_taps_math_logic_2026-05-17_summary.md`, `artifacts/reports/probes/reasoning_trace_transfer_2026-05-17.md`, `artifacts/reports/probes/code_taps_on_math_logic_existing_2026-05-17.md`
- interpretation: Generated reasoning traces still favor the general HH readout enough that a reasoning-specific specialist is not justified by this probe.

## Science / bio / chem / medicine natural-distractor validation (2026-05-17)

- SCIENCE_DISTRACTOR_SET_VERDICT: `READY`
- SCIENCE_DISTRACTOR_FEATURE_VERDICT: `READY`
- SCIENCE_TRANSFER_VERDICT: `GOOD`
- SCIENCE_SPECIALIST_VERDICT: `SPECIALIST_NEEDED`
- BIOLOGY_TRANSFER_VERDICT: `GOOD`
- CHEMISTRY_TRANSFER_VERDICT: `GOOD`
- MEDICINE_TRANSFER_VERDICT: `GOOD`
- GENERAL_SCIENCE_TRANSFER_VERDICT: `GOOD`
- SCIENCE_SPECIFIC_HEAD_VERDICT: `GENERAL_SUFFICIENT`
- SCIENCE_DOMAIN_ANALOGY_VERDICT: `HETEROGENEOUS`
- dataset/task counts: `{'mmlu': 95, 'sciq': 25}`
- random_top1_baseline: `0.25`
- best HH row: `{'family': 'HH', 'config': '36_L4', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.4583333333333333, 'over_random': 0.20833333333333331, 'pairwise': 0.6916666666666667, 'condorcet': 0.4583333333333333, 'cycle': 0.0, 'margin_mean': 0.391814417935287, 'margin_std': 0.48293810592259795}`
- best code row: `{'family': 'CODE', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5416666666666666, 'over_random': 0.29166666666666663, 'pairwise': 0.7027777777777777, 'condorcet': 0.5416666666666666, 'cycle': 0.0, 'margin_mean': 0.6878731965863456, 'margin_std': 0.9032204292594548}`
- best NoNorm row: `{'family': 'CODE', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5416666666666666, 'over_random': 0.29166666666666663, 'pairwise': 0.7027777777777777, 'condorcet': 0.5416666666666666, 'cycle': 0.0, 'margin_mean': 0.6878731965863456, 'margin_std': 0.9032204292594548}`
- best AntisymLinear row: `{'family': 'CODE', 'config': '36_L4', 'architecture': 'AntisymLinear', 'top1': 0.43333333333333335, 'over_random': 0.18333333333333335, 'pairwise': 0.625, 'condorcet': 0.4083333333333333, 'cycle': 0.0, 'margin_mean': 1.359556249404947, 'margin_std': 0.9679773308615227}`
- subdomain breakdown: `{'biology': 25, 'chemistry': 25, 'medicine': 25, 'general_science': 25, 'other_science': 20}`
- science objective eval domain status: `add_as_objective_eval_domain`
- science/medicine specialist status: `specialist_needed`
- medicine caveat: benchmark MCQ transfer only, not clinical validation.
- full reports: `artifacts/reports/probes/science_domain_audit_2026-05-17_summary.md`, `artifacts/reports/probes/science_natural_distractor_transfer_2026-05-17.md`, `artifacts/reports/probes/science_domain_comparison_2026-05-17.md`
- interpretation: Science shows a specialist gap in at least one subdomain, so a larger science-specific projection check is justified.


## Mixed-domain tiny heads (2026-05-17)

- MIXED_TAP_SPLIT_VERDICT = READY
- MIXED_TAP_FEATURE_VERDICT = READY
- MIXED_TAP_TRAINING_VERDICT = READY
- MIXED_HEAD_UTILITY_VERDICT = OBJECTIVE_MIXED_USEFUL
- MIXED_HEAD_UTILITY_PROVISIONAL = False
- STRICT_CLEAN_CODE_REGRET_STATUS = CLEAN_WIN
- SMALL_DOMAIN_OVERFIT = True
- DOMAIN_OVERFIT_WARNING = True
- GSM8K_EVAL_STATUS = READY
- best mixed family = MIX_CODE_REASONING
- average objective regret pairwise = 0.048
- worst objective regret pairwise = 0.000
- strict-clean code regret pairwise = 0.067
- reasoning trace regret pairwise = 0.091
- science medicine regret pairwise = -0.083
- clean GSM8K regret pairwise = 0.074
- HH regret pairwise = -0.150
- recommended Phase 1 head set = HH_general_plus_code_specialist_plus_objective_mixed_head
- full reports: `artifacts/reports/probes/mixed_domain_heads_audit_2026-05-17_summary.md`, `artifacts/reports/probes/mixed_domain_head_evaluation_2026-05-17.json`, `artifacts/reports/probes/mixed_head_controller_implications_2026-05-17.md`
- interpretation: mixed heads are controller-routing candidates and should complement, not erase, the established HH/general and code-specialist roles unless regret is cleanly positive outside the current small strict-clean sample.


## BG controller-policy simulator (2026-05-17)

- BG_POLICY_EVAL_BUNDLE_VERDICT = READY
- BG_HEAD_COMPARISON_VERDICT = READY
- HEAD_COMPLEMENTARITY_VERDICT = HIGH_COMPLEMENTARITY
- BG_POLICY_SIM_VERDICT = READY
- BEST_POLICY_VERDICT = OBJECTIVE_MIXED_DEFAULT_WINS
- RECOMMENDED_BG_POLICY = HH_GENERAL_PLUS_OBJECTIVE_MIXED_PLUS_CODE_BACKUP
- CONTRAST_DETECTOR_VERDICT = DEPLOYABILITY_WEAK
- DEFER_POLICY_VERDICT = DEFER_NOT_USEFUL
- ORACLE_GAP_VERDICT = LARGE
- SMALL_N_UNSTABLE_POLICY = True
- STRICT_CLEAN_POLICY_BORDERLINE = True
- HH_HELDOUT_POLICY_BORDERLINE = False
- best policy metrics: DOMAIN_ROUTED_SIMPLE objective_avg=0.817, HH=0.900, strict_clean=0.833
- best single head metrics: OBJECTIVE_MIXED_ONLY objective_avg=0.822; CODE_ONLY strict_clean=0.833
- objective mixed vs code specialist strict-clean delta = 0.033
- HH preservation result: DOMAIN_ROUTED_SIMPLE uses HH_GENERAL on HH, delta vs HH_ONLY = 0.000
- defer policy result: `{'defer_policy_verdict': 'DEFER_NOT_USEFUL', 'best_defer_policy': {'policy': 'ORACLE_POLICY_WITH_DEFER', 'improvement70': 0.02986658580958146, 'improvement80': 0.006450587019319887, 'improvement90': 0.0065386244880666355, 'coverage': 1.0}, 'fallback_adjusted': {'GENERAL_AND_OBJECTIVE_VOTE_defer': {'random_fallback_average': 0.6414534855982225, 'domain_routed_fallback_average': 0.802741845140385, 'by_domain_random': {'CODE_STRICT_CLEAN_ALL16': 0.6833333333333333, 'REASONING_NATURAL_DISTRACTOR': 0.5833333333333333, 'REASONING_TRACE': 0.6363636363636364, 'SCIENCE_OVERALL': 0.6052631578947368, 'SCIENCE_BIOLOGY': 0.625, 'SCIENCE_CHEMISTRY': 0.625, 'SCIENCE_MEDICINE': 0.5, 'SCIENCE_GENERAL': 0.625, 'SCIENCE_OTHER': 0.6666666666666667, 'CODE_RUNNABLE_DIAGNOSTIC': 0.59375, 'CLEAN_GSM8K_EXPANDED': 0.6851851851851852, 'HH_HELDOUT20': 0.8, 'HH_200_DIAGNOSTIC': 0.71}, 'by_domain_domain_routed': {'CODE_STRICT_CLEAN_ALL16': 0.8277777777777777, 'REASONING_NATURAL_DISTRACTOR': 0.6666666666666666, 'REASONING_TRACE': 1.0, 'SCIENCE_OVERALL': 0.7506925207756233, 'SCIENCE_BIOLOGY': 0.875, 'SCIENCE_CHEMISTRY': 0.8125, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.9375, 'SCIENCE_OTHER': 0.6296296296296297, 'CODE_RUNNABLE_DIAGNOSTIC': 0.94921875, 'CLEAN_GSM8K_EXPANDED': 0.808641975308642, 'HH_HELDOUT20': 0.9200000000000002, 'HH_200_DIAGNOSTIC': 0.84135}}, 'THREE_HEAD_VOTE_defer': {'random_fallback_average': 0.5592287074523917, 'domain_routed_fallback_average': 0.7749835455233213, 'by_domain_random': {'CODE_STRICT_CLEAN_ALL16': 0.5666666666666667, 'REASONING_NATURAL_DISTRACTOR': 0.5833333333333334, 'REASONING_TRACE': 0.5, 'SCIENCE_OVERALL': 0.5263157894736843, 'SCIENCE_BIOLOGY': 0.5, 'SCIENCE_CHEMISTRY': 0.5, 'SCIENCE_MEDICINE': 0.5, 'SCIENCE_GENERAL': 0.625, 'SCIENCE_OTHER': 0.5, 'CODE_RUNNABLE_DIAGNOSTIC': 0.59375, 'CLEAN_GSM8K_EXPANDED': 0.6574074074074074, 'HH_HELDOUT20': 0.625, 'HH_200_DIAGNOSTIC': 0.5925}, 'by_domain_domain_routed': {'CODE_STRICT_CLEAN_ALL16': 0.8111111111111111, 'REASONING_NATURAL_DISTRACTOR': 0.7083333333333334, 'REASONING_TRACE': 1.0, 'SCIENCE_OVERALL': 0.7008310249307479, 'SCIENCE_BIOLOGY': 0.8333333333333334, 'SCIENCE_CHEMISTRY': 0.75, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.9375, 'SCIENCE_OTHER': 0.4444444444444444, 'CODE_RUNNABLE_DIAGNOSTIC': 0.94921875, 'CLEAN_GSM8K_EXPANDED': 0.8168724279835391, 'HH_HELDOUT20': 0.885, 'HH_200_DIAGNOSTIC': 0.821475}}, 'CONSENSUS_SELECT_HH_OBJECTIVE': {'random_fallback_average': 0.6414534855982225, 'domain_routed_fallback_average': 0.802741845140385, 'by_domain_random': {'CODE_STRICT_CLEAN_ALL16': 0.6833333333333333, 'REASONING_NATURAL_DISTRACTOR': 0.5833333333333333, 'REASONING_TRACE': 0.6363636363636364, 'SCIENCE_OVERALL': 0.6052631578947368, 'SCIENCE_BIOLOGY': 0.625, 'SCIENCE_CHEMISTRY': 0.625, 'SCIENCE_MEDICINE': 0.5, 'SCIENCE_GENERAL': 0.625, 'SCIENCE_OTHER': 0.6666666666666667, 'CODE_RUNNABLE_DIAGNOSTIC': 0.59375, 'CLEAN_GSM8K_EXPANDED': 0.6851851851851852, 'HH_HELDOUT20': 0.8, 'HH_200_DIAGNOSTIC': 0.71}, 'by_domain_domain_routed': {'CODE_STRICT_CLEAN_ALL16': 0.8277777777777777, 'REASONING_NATURAL_DISTRACTOR': 0.6666666666666666, 'REASONING_TRACE': 1.0, 'SCIENCE_OVERALL': 0.7506925207756233, 'SCIENCE_BIOLOGY': 0.875, 'SCIENCE_CHEMISTRY': 0.8125, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.9375, 'SCIENCE_OTHER': 0.6296296296296297, 'CODE_RUNNABLE_DIAGNOSTIC': 0.94921875, 'CLEAN_GSM8K_EXPANDED': 0.808641975308642, 'HH_HELDOUT20': 0.9200000000000002, 'HH_200_DIAGNOSTIC': 0.84135}}, 'CONSENSUS_SELECT_ALL_THREE': {'random_fallback_average': 0.5592287074523917, 'domain_routed_fallback_average': 0.7749835455233213, 'by_domain_random': {'CODE_STRICT_CLEAN_ALL16': 0.5666666666666667, 'REASONING_NATURAL_DISTRACTOR': 0.5833333333333334, 'REASONING_TRACE': 0.5, 'SCIENCE_OVERALL': 0.5263157894736843, 'SCIENCE_BIOLOGY': 0.5, 'SCIENCE_CHEMISTRY': 0.5, 'SCIENCE_MEDICINE': 0.5, 'SCIENCE_GENERAL': 0.625, 'SCIENCE_OTHER': 0.5, 'CODE_RUNNABLE_DIAGNOSTIC': 0.59375, 'CLEAN_GSM8K_EXPANDED': 0.6574074074074074, 'HH_HELDOUT20': 0.625, 'HH_200_DIAGNOSTIC': 0.5925}, 'by_domain_domain_routed': {'CODE_STRICT_CLEAN_ALL16': 0.8111111111111111, 'REASONING_NATURAL_DISTRACTOR': 0.7083333333333334, 'REASONING_TRACE': 1.0, 'SCIENCE_OVERALL': 0.7008310249307479, 'SCIENCE_BIOLOGY': 0.8333333333333334, 'SCIENCE_CHEMISTRY': 0.75, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.9375, 'SCIENCE_OTHER': 0.4444444444444444, 'CODE_RUNNABLE_DIAGNOSTIC': 0.94921875, 'CLEAN_GSM8K_EXPANDED': 0.8168724279835391, 'HH_HELDOUT20': 0.885, 'HH_200_DIAGNOSTIC': 0.821475}}, 'ORACLE_POLICY_WITH_DEFER': {'random_fallback_average': 0.9050779727095517, 'domain_routed_fallback_average': 0.9050779727095517, 'by_domain_random': {'CODE_STRICT_CLEAN_ALL16': 0.9, 'REASONING_NATURAL_DISTRACTOR': 0.7916666666666666, 'REASONING_TRACE': 1.0, 'SCIENCE_OVERALL': 0.7017543859649122, 'SCIENCE_BIOLOGY': 1.0, 'SCIENCE_CHEMISTRY': 1.0, 'SCIENCE_MEDICINE': 0.8333333333333334, 'SCIENCE_GENERAL': 1.0, 'SCIENCE_OTHER': 0.8888888888888888, 'CODE_RUNNABLE_DIAGNOSTIC': 1.0, 'CLEAN_GSM8K_EXPANDED': 0.8703703703703703, 'HH_HELDOUT20': 0.9, 'HH_200_DIAGNOSTIC': 0.88}, 'by_domain_domain_routed': {'CODE_STRICT_CLEAN_ALL16': 0.9, 'REASONING_NATURAL_DISTRACTOR': 0.7916666666666666, 'REASONING_TRACE': 1.0, 'SCIENCE_OVERALL': 0.7017543859649122, 'SCIENCE_BIOLOGY': 1.0, 'SCIENCE_CHEMISTRY': 1.0, 'SCIENCE_MEDICINE': 0.8333333333333334, 'SCIENCE_GENERAL': 1.0, 'SCIENCE_OTHER': 0.8888888888888888, 'CODE_RUNNABLE_DIAGNOSTIC': 1.0, 'CLEAN_GSM8K_EXPANDED': 0.8703703703703703, 'HH_HELDOUT20': 0.9, 'HH_200_DIAGNOSTIC': 0.88}}, 'MARGIN_DEFER_0.05': {'random_fallback_average': 0.7388607737291948, 'domain_routed_fallback_average': 0.7741764072706132, 'by_domain_random': {'CODE_STRICT_CLEAN_ALL16': 0.6833333333333333, 'REASONING_NATURAL_DISTRACTOR': 0.7291666666666666, 'REASONING_TRACE': 1.0, 'SCIENCE_OVERALL': 0.631578947368421, 'SCIENCE_BIOLOGY': 0.7083333333333334, 'SCIENCE_CHEMISTRY': 0.75, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.7916666666666666, 'SCIENCE_OTHER': 0.4444444444444444, 'CODE_RUNNABLE_DIAGNOSTIC': 0.9375, 'CLEAN_GSM8K_EXPANDED': 0.7499999999999999, 'HH_HELDOUT20': 0.875, 'HH_200_DIAGNOSTIC': 0.8875}, 'by_domain_domain_routed': {'CODE_STRICT_CLEAN_ALL16': 0.8055555555555556, 'REASONING_NATURAL_DISTRACTOR': 0.75, 'REASONING_TRACE': 1.0, 'SCIENCE_OVERALL': 0.6509695290858726, 'SCIENCE_BIOLOGY': 0.7916666666666667, 'SCIENCE_CHEMISTRY': 0.75, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.8958333333333333, 'SCIENCE_OTHER': 0.4444444444444444, 'CODE_RUNNABLE_DIAGNOSTIC': 0.9375, 'CLEAN_GSM8K_EXPANDED': 0.765432098765432, 'HH_HELDOUT20': 0.9349999999999999, 'HH_200_DIAGNOSTIC': 0.921225}}, 'MARGIN_DEFER_0.10': {'random_fallback_average': 0.7237116441063809, 'domain_routed_fallback_average': 0.7740804334930143, 'by_domain_random': {'CODE_STRICT_CLEAN_ALL16': 0.6833333333333333, 'REASONING_NATURAL_DISTRACTOR': 0.7291666666666666, 'REASONING_TRACE': 0.8636363636363636, 'SCIENCE_OVERALL': 0.6052631578947368, 'SCIENCE_BIOLOGY': 0.7083333333333334, 'SCIENCE_CHEMISTRY': 0.75, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.7916666666666666, 'SCIENCE_OTHER': 0.4444444444444444, 'CODE_RUNNABLE_DIAGNOSTIC': 0.9375, 'CLEAN_GSM8K_EXPANDED': 0.7407407407407408, 'HH_HELDOUT20': 0.85, 'HH_200_DIAGNOSTIC': 0.8875}, 'by_domain_domain_routed': {'CODE_STRICT_CLEAN_ALL16': 0.8055555555555556, 'REASONING_NATURAL_DISTRACTOR': 0.75, 'REASONING_TRACE': 1.0, 'SCIENCE_OVERALL': 0.6343490304709142, 'SCIENCE_BIOLOGY': 0.7916666666666667, 'SCIENCE_CHEMISTRY': 0.75, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.8958333333333333, 'SCIENCE_OTHER': 0.4444444444444444, 'CODE_RUNNABLE_DIAGNOSTIC': 0.9375, 'CLEAN_GSM8K_EXPANDED': 0.771604938271605, 'HH_HELDOUT20': 0.9299999999999999, 'HH_200_DIAGNOSTIC': 0.935425}}, 'MARGIN_DEFER_0.20': {'random_fallback_average': 0.7007040717567034, 'domain_routed_fallback_average': 0.7671446459419028, 'by_domain_random': {'CODE_STRICT_CLEAN_ALL16': 0.6833333333333333, 'REASONING_NATURAL_DISTRACTOR': 0.7291666666666666, 'REASONING_TRACE': 0.8636363636363636, 'SCIENCE_OVERALL': 0.6140350877192983, 'SCIENCE_BIOLOGY': 0.7083333333333334, 'SCIENCE_CHEMISTRY': 0.75, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.75, 'SCIENCE_OTHER': 0.2777777777777778, 'CODE_RUNNABLE_DIAGNOSTIC': 0.875, 'CLEAN_GSM8K_EXPANDED': 0.7037037037037037, 'HH_HELDOUT20': 0.85, 'HH_200_DIAGNOSTIC': 0.8875}, 'by_domain_domain_routed': {'CODE_STRICT_CLEAN_ALL16': 0.8055555555555556, 'REASONING_NATURAL_DISTRACTOR': 0.75, 'REASONING_TRACE': 1.0, 'SCIENCE_OVERALL': 0.6528162511542013, 'SCIENCE_BIOLOGY': 0.7916666666666667, 'SCIENCE_CHEMISTRY': 0.75, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.9583333333333333, 'SCIENCE_OTHER': 0.2592592592592593, 'CODE_RUNNABLE_DIAGNOSTIC': 0.9296875, 'CLEAN_GSM8K_EXPANDED': 0.7757201646090535, 'HH_HELDOUT20': 0.9299999999999999, 'HH_200_DIAGNOSTIC': 0.953175}}, 'MARGIN_DEFER_0.30': {'random_fallback_average': 0.7018189315557737, 'domain_routed_fallback_average': 0.7742404131032209, 'by_domain_random': {'CODE_STRICT_CLEAN_ALL16': 0.6333333333333333, 'REASONING_NATURAL_DISTRACTOR': 0.7916666666666666, 'REASONING_TRACE': 0.8636363636363636, 'SCIENCE_OVERALL': 0.5964912280701755, 'SCIENCE_BIOLOGY': 0.7083333333333334, 'SCIENCE_CHEMISTRY': 0.75, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.75, 'SCIENCE_OTHER': 0.33333333333333337, 'CODE_RUNNABLE_DIAGNOSTIC': 0.875, 'CLEAN_GSM8K_EXPANDED': 0.6851851851851852, 'HH_HELDOUT20': 0.85, 'HH_200_DIAGNOSTIC': 0.87}, 'by_domain_domain_routed': {'CODE_STRICT_CLEAN_ALL16': 0.788888888888889, 'REASONING_NATURAL_DISTRACTOR': 0.8333333333333333, 'REASONING_TRACE': 1.0, 'SCIENCE_OVERALL': 0.65466297322253, 'SCIENCE_BIOLOGY': 0.7916666666666667, 'SCIENCE_CHEMISTRY': 0.75, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.9583333333333333, 'SCIENCE_OTHER': 0.29629629629629634, 'CODE_RUNNABLE_DIAGNOSTIC': 0.9296875, 'CLEAN_GSM8K_EXPANDED': 0.7674897119341564, 'HH_HELDOUT20': 0.9299999999999999, 'HH_200_DIAGNOSTIC': 0.9480999999999999}}, 'MARGIN_DEFER_0.50': {'random_fallback_average': 0.68275334314808, 'domain_routed_fallback_average': 0.7896775596633284, 'by_domain_random': {'CODE_STRICT_CLEAN_ALL16': 0.6666666666666666, 'REASONING_NATURAL_DISTRACTOR': 0.7708333333333334, 'REASONING_TRACE': 0.6363636363636364, 'SCIENCE_OVERALL': 0.6052631578947368, 'SCIENCE_BIOLOGY': 0.75, 'SCIENCE_CHEMISTRY': 0.75, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.75, 'SCIENCE_OTHER': 0.33333333333333337, 'CODE_RUNNABLE_DIAGNOSTIC': 0.875, 'CLEAN_GSM8K_EXPANDED': 0.6666666666666667, 'HH_HELDOUT20': 0.825, 'HH_200_DIAGNOSTIC': 0.8300000000000001}, 'by_domain_domain_routed': {'CODE_STRICT_CLEAN_ALL16': 0.8444444444444444, 'REASONING_NATURAL_DISTRACTOR': 0.8333333333333334, 'REASONING_TRACE': 1.0, 'SCIENCE_OVERALL': 0.6925207756232687, 'SCIENCE_BIOLOGY': 0.9166666666666667, 'SCIENCE_CHEMISTRY': 0.75, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.9583333333333333, 'SCIENCE_OTHER': 0.29629629629629634, 'CODE_RUNNABLE_DIAGNOSTIC': 0.9296875, 'CLEAN_GSM8K_EXPANDED': 0.7592592592592593, 'HH_HELDOUT20': 0.9249999999999999, 'HH_200_DIAGNOSTIC': 0.9436}}, 'DISAGREEMENT_DEFER_0.10': {'random_fallback_average': 0.5776632553606238, 'domain_routed_fallback_average': 0.7797329855877198, 'by_domain_random': {'CODE_STRICT_CLEAN_ALL16': 0.65, 'REASONING_NATURAL_DISTRACTOR': 0.5833333333333334, 'REASONING_TRACE': 0.5, 'SCIENCE_OVERALL': 0.5526315789473684, 'SCIENCE_BIOLOGY': 0.625, 'SCIENCE_CHEMISTRY': 0.5, 'SCIENCE_MEDICINE': 0.5, 'SCIENCE_GENERAL': 0.625, 'SCIENCE_OTHER': 0.5, 'CODE_RUNNABLE_DIAGNOSTIC': 0.59375, 'CLEAN_GSM8K_EXPANDED': 0.6574074074074074, 'HH_HELDOUT20': 0.625, 'HH_200_DIAGNOSTIC': 0.5975}, 'by_domain_domain_routed': {'CODE_STRICT_CLEAN_ALL16': 0.8166666666666667, 'REASONING_NATURAL_DISTRACTOR': 0.7083333333333334, 'REASONING_TRACE': 1.0, 'SCIENCE_OVERALL': 0.7174515235457064, 'SCIENCE_BIOLOGY': 0.875, 'SCIENCE_CHEMISTRY': 0.75, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.9375, 'SCIENCE_OTHER': 0.4444444444444444, 'CODE_RUNNABLE_DIAGNOSTIC': 0.94921875, 'CLEAN_GSM8K_EXPANDED': 0.8168724279835391, 'HH_HELDOUT20': 0.885, 'HH_200_DIAGNOSTIC': 0.8193750000000001}}, 'DISAGREEMENT_DEFER_0.20': {'random_fallback_average': 0.5867676938071675, 'domain_routed_fallback_average': 0.7805542946646306, 'by_domain_random': {'CODE_STRICT_CLEAN_ALL16': 0.7166666666666667, 'REASONING_NATURAL_DISTRACTOR': 0.6458333333333333, 'REASONING_TRACE': 0.5, 'SCIENCE_OVERALL': 0.5438596491228069, 'SCIENCE_BIOLOGY': 0.625, 'SCIENCE_CHEMISTRY': 0.5, 'SCIENCE_MEDICINE': 0.5, 'SCIENCE_GENERAL': 0.625, 'SCIENCE_OTHER': 0.4444444444444445, 'CODE_RUNNABLE_DIAGNOSTIC': 0.59375, 'CLEAN_GSM8K_EXPANDED': 0.6759259259259259, 'HH_HELDOUT20': 0.65, 'HH_200_DIAGNOSTIC': 0.6074999999999999}, 'by_domain_domain_routed': {'CODE_STRICT_CLEAN_ALL16': 0.8388888888888889, 'REASONING_NATURAL_DISTRACTOR': 0.75, 'REASONING_TRACE': 1.0, 'SCIENCE_OVERALL': 0.6989843028624192, 'SCIENCE_BIOLOGY': 0.875, 'SCIENCE_CHEMISTRY': 0.75, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.9375, 'SCIENCE_OTHER': 0.40740740740740744, 'CODE_RUNNABLE_DIAGNOSTIC': 0.94921875, 'CLEAN_GSM8K_EXPANDED': 0.8148148148148149, 'HH_HELDOUT20': 0.8900000000000001, 'HH_200_DIAGNOSTIC': 0.8187249999999999}}, 'DISAGREEMENT_DEFER_0.30': {'random_fallback_average': 0.5920312265707003, 'domain_routed_fallback_average': 0.7818001240410155, 'by_domain_random': {'CODE_STRICT_CLEAN_ALL16': 0.7333333333333334, 'REASONING_NATURAL_DISTRACTOR': 0.6458333333333333, 'REASONING_TRACE': 0.5, 'SCIENCE_OVERALL': 0.5438596491228069, 'SCIENCE_BIOLOGY': 0.625, 'SCIENCE_CHEMISTRY': 0.5, 'SCIENCE_MEDICINE': 0.5, 'SCIENCE_GENERAL': 0.625, 'SCIENCE_OTHER': 0.4444444444444445, 'CODE_RUNNABLE_DIAGNOSTIC': 0.59375, 'CLEAN_GSM8K_EXPANDED': 0.6851851851851851, 'HH_HELDOUT20': 0.675, 'HH_200_DIAGNOSTIC': 0.625}, 'by_domain_domain_routed': {'CODE_STRICT_CLEAN_ALL16': 0.8444444444444446, 'REASONING_NATURAL_DISTRACTOR': 0.75, 'REASONING_TRACE': 1.0, 'SCIENCE_OVERALL': 0.6989843028624192, 'SCIENCE_BIOLOGY': 0.875, 'SCIENCE_CHEMISTRY': 0.75, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.9375, 'SCIENCE_OTHER': 0.40740740740740744, 'CODE_RUNNABLE_DIAGNOSTIC': 0.94921875, 'CLEAN_GSM8K_EXPANDED': 0.8189300411522633, 'HH_HELDOUT20': 0.895, 'HH_200_DIAGNOSTIC': 0.82025}}}}`
- oracle-gap summary: average=0.135, objective=0.061
- contrast-detector summary: DEPLOYABILITY_WEAK
- recommended Phase 1 controller design: HH/general for HH and unknown, objective mixed for objective QA/reasoning/science/GSM8K, code specialist backup for strict-clean or high-similarity code, defer on low margin/disagreement.
- full reports: `artifacts/reports/probes/bg_controller_policy_simulator_2026-05-17_summary.md`, `artifacts/reports/probes/bg_controller_policy_simulation_2026-05-17.json`, `artifacts/reports/probes/bg_candidate_head_comparison_2026-05-17.json`, `artifacts/reports/probes/bg_phase1_controller_design_note_2026-05-17.md`
- interpretation: deploy a read-only routed controller; current heads are complementary enough to route, but not stable enough to collapse into one universal head.
