# Math And GSM8K Status

**Original date:** 2026-05-15  
**Refactored:** 2026-05-18 for readability  
**Raw original:** `raw_archive_2026-05-18/math_bg_gate_pilot_2026-05-15.md`

## Current Status

Full MATH gate-scale is deferred. The old mixed math pilot was later found truncation-confounded, so it should not carry current decisions.

The valid math-adjacent evidence now comes mainly from clean GSM8K transfer, plus loop/layer geometry probes.

## What Still Holds

The early/mid-layer tap geometry remains useful:

- layer 24 and 36 pooled states are relatively loop-converged.
- layer 47 carries more loop spread.
- tiny linear pairwise heads can read objective branch-selection signals from pooled states.

The project should not jump to full MATH generation until task yield, truncation control, and answer verification are cleaner.

## What Was Demoted

The old mixed MATH-style branch pilot is not a stable gate result.

Reason:

- validity/truncation probes showed the old math pilot was confounded.
- model verbosity and source-yield constraints made full local MATH gate-scale inefficient.

Current backlog status:

- `MATH gate-scale = BLOCKED/DEFERRED`

## Clean GSM8K Replacement Evidence

Clean GSM8K became the safer math-adjacent objective domain.

Decision-relevant result:

- expanded clean GSM8K transfer was GOOD for tiny linear taps.
- GRU control was weak.
- objective answer-key branch data produced readable signal.

Best fixed-config cross-domain row from the latest audit:

| domain | head | config | architecture | top1 | pairwise | cycle |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `CLEAN_GSM8K_EXPANDED` | CODE | `24_mean` | AntisymLinear | 0.893 | 0.796 | 0.000 |

## Current Interpretation

Clean GSM8K supports the idea that objective domains can contain scalar-readable and pairwise-readable branch signals.

That does not by itself justify full MATH gate-scale. MATH remains a separate higher-cost domain with verification and generation-yield problems.

## Recommended Next For Math

Do not run full MATH gate-scale as the immediate next step.

Only return to MATH after:

1. the controller-policy simulator is built,
2. reasoning multiple-choice is hardened as a third objective eval domain,
3. the generation/parsing/truncation path is explicitly controlled.

## Source Reports

- `artifacts/reports/probes/math_data_validity_2026-05-16.md`
- `artifacts/reports/probes/clean_gsm8k_expanded_transfer_gru_2026-05-16_summary.md`
- `artifacts/reports/probes/bg_fixed_config_cross_domain_audit_2026-05-17.md`
- `artifacts/reports/probes/loop_layer_diagnostics_current_domains_2026-05-17.md`

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
