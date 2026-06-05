# Tap Interface

**Original date:** 2026-05-15  
**Refactored:** 2026-05-18 for readability  
**Raw original:** `raw_archive_2026-05-18/bg_tap_interface_revision_2026-05-15.md`

## Current Status

This note records the active tap interface after the layer 24/36 probe and later cross-domain audits.

The current default is still a tiny pairwise head over pooled hidden states. GRU is not default. Published 5M evaluator retraining is not part of the current local diagnostic path.

## Active Tiny Heads

`AntisymLinear`:

```text
LayerNorm(no affine)(left - right) -> Linear(no bias)
```

`AntisymLinearNoNorm`:

```text
Linear(left - right)
```

Both heads operate on branch pairs. The positive ordering is determined by external labels such as HH chosen/rejected, exact-answer correctness, unit-test correctness, or answer-key correctness.

## Active Feature Inputs

The common pooled-state configs are:

- layer 24: `L1`, `L4`, `mean`.
- layer 36: `L1`, `L4`, `mean`.
- layer 47: `L4`, `mean`, `concat_L1_L4`, `concat_all_loops`.

Feature capture uses masked mean pooling and fp32 pooled features in current probes.

## Layer Policy

Layer 24/36 features are more loop-converged and often stable for objective branch selection.

Layer 47 has more loop spread and can still be useful, especially for HH and all-loop/late-loop settings, but it is not uniformly best across domains.

Latest loop geometry L1/L4 cosine:

| domain | layer 24 | layer 36 | layer 47 |
| --- | ---: | ---: | ---: |
| HH | 0.928 | 0.935 | 0.735 |
| clean GSM8K | 0.911 | 0.930 | 0.631 |
| strict-clean code | 0.917 | 0.926 | 0.723 |
| reasoning | 0.924 | 0.947 | 0.687 |

## Domain Policy

Use a general-plus-specialist setup:

- HH/preference-like domain: HH-trained general head.
- strict-clean code: code-trained specialist head.
- reasoning multiple choice: promote to a third objective eval domain; current pilot supports both HH and code-trained heads, with code-trained best overall.

## NoNorm Policy

NoNorm is useful as a scalar-readable branch-ranking ablation in objective domains. It should not be treated as a universal replacement for AntisymLinear.

Current practical rule:

- Use AntisymLinear as the safer default for hard near-miss and preference-like branch comparisons.
- Keep NoNorm in the fixed-config matrix because it remains competitive and interpretable in objective domains.

## GRU Policy

GRU is not the default. The expanded clean GSM8K run found the linear transfer result GOOD while the GRU control was weak.

Keep GRU for:

- escalation experiments.
- temporal controls.
- future cases where fixed pooled heads fail and trajectory information is clearly needed.

## Required Hygiene

Do not use tap outputs as labels.

Do not train on held-out eval tasks.

For code, candidate labels come from unit tests:

- `correct`: passes all tests.
- `near_miss`: passes at least one but not all granular tests.
- `wrong_code`: parses/runs with expected callable but passes zero tests.
- runtime and malformed candidates are diagnostic unless explicitly included in a diagnostic variant.

For reasoning, labels come from answer keys after parseable final-option extraction.

## Current Source Reports

- `artifacts/reports/probes/bg_head_registry_2026-05-17.md`
- `artifacts/reports/probes/bg_fixed_config_cross_domain_audit_2026-05-17.md`
- `artifacts/reports/probes/loop_layer_diagnostics_current_domains_2026-05-17.md`
- `docs/evaluator/bg_weight_space_merged_taps_plan.md`

## Weight-Space Merge Proposal

Because the active heads are tiny antisymmetric linear readouts, their `linear.weight` vectors can be extracted and compared directly when configs are compatible. The proposed next experiment is to preserve old `coding_reasoning` / `mixed_objective_all` directions and add orthogonalized hidden-branch / bridge residual directions.

This is a diagnostic and candidate compact-selector path only. It does not update old registries, change routing, or replace the fixed old+branch+bridge composite unless the merged heads beat it on heldout survival/final-selection metrics.

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
