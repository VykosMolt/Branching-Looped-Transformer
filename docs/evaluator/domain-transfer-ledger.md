# Domain Transfer Ledger

## CoreContent v2 Addendum (2026-06-06)

After a 27–520× content-selection data expansion (`corecontent_dataset_expansion_refit_v2`),
a crafted content tap beats the v1 broad-objective baseline on heldout — but the cross-domain
transfer picture is nuanced. Per-domain heldout top1 (selected v2 tap vs `mixedhead_MIX_HH_OBJECTIVE`):

| Domain | negatives | v2 blockwise | HH | Δ |
| --- | --- | ---: | ---: | ---: |
| coding | constructed mutants | 0.935 | 0.613 | +0.32 |
| coding | real wrong-problem (relevance) | 0.583 | 0.502 | +0.08 |
| math | constructed perturbations | 0.657 | 0.586 | +0.07 |
| logic | real distractors | 0.460 | 0.353 | +0.11 |
| reasoning | real distractors | 0.681 | 0.677 | +0.00 |
| alignment | real preference pairs | 0.612 | 0.534 | +0.08 |

Transfer conclusions: the broad HH-objective direction transfers *weakly* to constructed
code/math/logic negatives (a crafted tap helps there). On **real** negatives the crafted edge
shrinks to single digits, and on the coding **relevance** test the crafted tap is no better than
a corruption detector — relevance is not linearly accessible in pooled L24/36/47. Mid layer 36
carries the most content signal; layer 47 is prunable. Detail: `content-selection-taps.md`.

## Current Addendum (2026-06-02)

The current branch/action selector framing is DualAnchor:

- `MIX_CODE_REASONING`
- `MIX_OBJECTIVE_ALL`

These taps should be read as pairwise evaluators over hidden-state branch candidates, not absolute scorers.

The latest domain/branch conclusions are:

| Result | Verdict / Metric | Source |
| --- | --- | --- |
| fresh same-content comparison | new two-tap `0.8818` vs old BG `0.8800` | `history/bg-run-notes/dualanchor-two-tap/bg_two_tap_fresh_dataset_comparison_v1.md` |
| HH-RLHF comparison | new two-tap `0.6152` vs old BG `0.5977` | `history/bg-run-notes/dualanchor-two-tap/bg_two_tap_hh_rlhf_comparison_v1.md` |
| layer-native two-tap readiness | `DOMAIN_READY_BRANCH_GAP` | `history/bg-run-notes/dualanchor-two-tap/bg_layer_native_two_tap_readiness_v1.md` |
| constrained layer-native training | `DOMAIN_READY_BRANCH_GAP` | `history/bg-run-notes/dualanchor-two-tap/bg_layer_native_two_tap_constrained_train_v1.md` |
| branch-gap repair | `DUALANCHOR_BRANCH_GAP_REDUCED` | `history/bg-run-notes/dualanchor-two-tap/bg_dualanchor_branch_gap_repair_v1.md` |
| architecture-looped v3 | `READY_WITH_TERMINAL_DEFER` | `dualanchor-architecture-baseline.md` |
| science/reasoning repair v2 | `REASONING_READY_SCIENCE_DIAGNOSTIC` | `science-reasoning-repair.md` |

Updated interpretation:

- The two DualAnchor taps match or slightly beat old BG on fresh same-content domain data.
- They also beat the old BG comparison on the 512-pair HH-RLHF sample, but absolute HH accuracy remains modest.
- Native 24/36/47 layer-local forms preserve domain behavior better than the earlier concat-only framing.
- The remaining weakness is no longer basic survival; it is terminal collapse/final selection on reward-diverse hard slices.
- The active Phase 2a baseline is architecture-looped DualAnchor with terminal defer, not fixed-composite top4 plus a learned final arbiter.
- The latest pre-steering science/reasoning repair locks reasoning as the headline scope
  under survivor-set handoff, but keeps science diagnostic-only because regenerated
  science heldout remained branch-generation weak.

See:

- `evaluator-navigation-map.md`
- `dualanchor-tap-evolution.md`
- `dualanchor-architecture-baseline.md`
- `terminal-selection-and-arbiters.md`
- `science-reasoning-repair.md`

**Refactored:** 2026-05-18 for readability  
**Raw original:** `history/raw_archive_2026-05-18/evaluator_domain_transfer_notes.md`

This file is the compact chronological ledger for domain-transfer results. Full historical text is preserved in the raw archive and in the artifact reports listed below.

## Legacy Conclusion Before DualAnchor

`GENERALIST_SPECIALIST_VERDICT = DOMAIN_SPECIALISTS_NEEDED`

Meaning:

- HH-trained heads remain strongest on HH preference pairs.
- Code-trained heads are materially stronger on strict-clean code.
- Objective branch domains can expose scalar-readable/transitive ranking behavior, especially for NoNorm.
- AntisymLinear remains the safer default for noisy preference-like or hard near-miss comparisons.
- Reasoning branch selection is promising, but needs a harder validation set.
- Hidden-origin branch generation is weak-but-usable after Branch Generator v1, but selector roles should remain composite rather than single universal.
- Weight-space merged taps are a proposed diagnostic/compact-selector follow-up, not an established policy: use old objective/code directions as preservation anchors and add branch/bridge residual directions only if they improve survival/final selection without degrading coding/reasoning.

## Chronological Ledger

| date | result | verdict / key number | artifact |
| --- | --- | --- | --- |
| 2026-06-01 | DualAnchor science/reasoning repair v2 | `REASONING_READY_SCIENCE_DIAGNOSTIC`; science diagnostic-only | `docs/evaluator/science-reasoning-repair.md` |
| 2026-05-14 | layer 24/36 bipartite + Thinking-vs-RLTT | early/mid taps converged; late tap more bipartite | `artifacts/reports/probes/math_layer_geometry_rltt.md` |
| 2026-05-15 | converged tap old-head input probe | old head compatible enough for transfer probes | `artifacts/reports/probes/probe_converged_tap_inputs.md` |
| 2026-05-15 | corrected math BG-gate pilot | trained 24/36 taps viable; gate-scale still open | `docs/evaluator/math-and-gsm8k-status.md` |
| 2026-05-16 | old math data validity | old mixed math pilot truncation-confounded | `artifacts/reports/probes/math_data_validity_2026-05-16.md` |
| 2026-05-16 | clean GSM8K expanded transfer | HH-trained linear transfer GOOD; GRU weak | `artifacts/reports/probes/clean_gsm8k_expanded_transfer_gru_2026-05-16_summary.md` |
| 2026-05-16 | code v1 | all-correct collapse | `artifacts/reports/probes/code_branch_pilot_2026-05-16_summary.md` |
| 2026-05-16 | code v2 dirty diagnostic | transfer signal but dirty harness | `artifacts/reports/probes/code_branch_pilot_v2_2026-05-16_summary.md` |
| 2026-05-16 | wrapper/taskset fixes | prose/status and MBPP wrapper issues fixed | `artifacts/reports/probes/code_branch_v2_harness_agent_fixes_2026-05-16.md` |
| 2026-05-16 | patched code v2-mini | runnable diagnostic GOOD | `artifacts/reports/probes/code_branch_pilot_v2_mini_patched_2026-05-16_summary.md` |
| 2026-05-17 | strict-clean task screening | 6 strict-clean-ready tasks, YELLOW | `artifacts/reports/probes/code_strict_clean_screening_2026-05-17_summary.md` |
| 2026-05-17 | strict-clean HH transfer OLD6 | WEAK, best top1 0.500, pairwise 0.571 | `artifacts/reports/probes/code_strict_clean_transfer_2026-05-17_summary.md` |
| 2026-05-17 | code-specific tiny-head control | GOOD on OLD6 | `artifacts/reports/probes/code_specific_tiny_head_control_2026-05-17.md` |
| 2026-05-17 | strict-clean expansion | 10 additional strict-clean-ready tasks | `artifacts/reports/probes/code_strict_clean_screening_expansion_2026-05-17.md` |
| 2026-05-17 | expanded strict-clean comparison | CODE_SPECIFIC_ADVANTAGE on ALL16 | `artifacts/reports/probes/expanded_strict_clean_code_projection_comparison_2026-05-17_summary.md` |
| 2026-05-17 | inverse code-trained taps on HH | partial small-split transfer; weak all-200 diagnostic | `artifacts/reports/probes/code_trained_taps_on_hh_2026-05-17.md` |
| 2026-05-17 | random 20-pair HH split audit | HH best mean 0.655, code best mean 0.640 | `artifacts/reports/probes/code_trained_vs_hh_trained_random20_hh_splits_2026-05-17.md` |
| 2026-05-17 | fixed-config cross-domain audit | DOMAIN_SPECIALISTS_NEEDED | `artifacts/reports/probes/bg_fixed_config_cross_domain_audit_2026-05-17.md` |
| 2026-05-17 | reasoning branch pilot | READY, 25 mixed tournaments | `artifacts/reports/probes/reasoning_branch_pilot_2026-05-17.md` |
| 2026-05-17 | reasoning transfer | GOOD, best top1 1.000 | `artifacts/reports/probes/reasoning_branch_transfer_2026-05-17.md` |
| 2026-05-17 | loop/layer diagnostics | READY across HH/GSM8K/code/reasoning | `artifacts/reports/probes/loop_layer_diagnostics_current_domains_2026-05-17.md` |
| 2026-05-18 | hidden-origin quota v4 | STILL_DATA_LIMITED; heldout 8 diverse groups / 64 non-tie pairs | `docs/evaluator/hidden-origin-quota-v4.md` |
| 2026-05-18 | hidden-origin Branch Generator v1 | WEAK_BUT_USABLE; heldout quota met, train/val weak | `docs/evaluator/hidden-origin-branch-generator-v1.md` |
| 2026-05-18 | universal branch-content taps v1 | FUSION_NEEDED; old-context good, bridge failed | `docs/evaluator/universal-branch-content-taps-v1.md` |
| 2026-05-18 | weight-space merged taps proposal | proposed; extract old/universal/hidden/bridge vectors and test old-preserving branch-validity residual merges | `docs/evaluator/history/bg-run-notes/weight-merge/bg_weight_space_merged_taps_plan.md` |

## Domain Summary

### HH

HH-trained heads are still the general preference baseline.

Best all-200 diagnostic:

- HH-trained `47_concat_L1_L4 / NoNorm`: accuracy 0.855.
- code-trained `47_L4 / NoNorm`: accuracy 0.535.

Random 20-pair splits softened the gap but did not eliminate it:

- best HH-trained mean: 0.655.
- best code-trained mean: 0.640.

### Clean GSM8K

Clean GSM8K supported HH-trained transfer and helped move the project away from the truncation-confounded old math pilot.

Best fixed-config cross-domain row:

- CODE `24_mean / AntisymLinear`: top1 0.893, pairwise 0.796.

Interpretation:

Objective answer-key domains expose readable branch-selection structure, but full MATH gate-scale remains deferred.

### Runnable-Diagnostic Code

Patched v2-mini produced a runnable diagnostic transfer result.

Best fixed-config cross-domain row:

- CODE `24_L1 / AntisymLinear`: top1 1.000, pairwise 1.000.

HH-trained heads were also usable on this easier code setting.

### Strict-Clean Code

Strict-clean code is the main evidence for domain specialists.

ALL16 result:

- HH-trained best: top1 0.750, pairwise 0.600.
- code-trained best: top1 0.875, pairwise 0.833.

Interpretation:

The states contain a code near-miss branch signal, but HH-trained projection is not strong enough for the cleanest code distinction.

### Reasoning Multiple Choice

The pilot is promising but small.

Data:

- 25 mixed tournaments.
- random top1 baseline 0.440.
- best code-trained top1 1.000, pairwise 1.000.
- best HH-trained top1 0.960, pairwise 0.986.

Interpretation:

Add reasoning as a third objective eval domain, then harden it.

## Current Backlog Status

From `history/bg-run-notes/bg_experiment_backlog_audit_2026-05-17`:

- DONE: 14.
- OBSOLETE: 3.
- ACTIVE: 2 at the time of the audit; both were completed by the broad audit.
- DEFERRED: 2.
- BLOCKED: 1, full MATH gate-scale.

Still useful later:

- controller-policy simulator.
- full HH capture/full split.
- harder reasoning/logic branch validation.

Do not prioritize next:

- GRU as default.
- HH Experiment 2 Redux.
- full MATH gate-scale without better task/data yield.

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

## Read-only BG controller implementation (2026-05-18)

- BG_CONTROLLER_ARTIFACT_VERDICT = READY
- BG_CONTROLLER_REPLAY_VERDICT = PASS
- module path: `src/evaluator/bg_controller.py`
- supported modes: `conservative`, `experimental_vote`, `code_backup`, `diagnostic_all`
- conservative routing: `hh`/`preference`/`unknown` use `hh_general`; `code`/`strict_clean_code`/`reasoning`/`science`/`math`/`gsm8k`/`objective` use `objective_mixed`.
- experimental vote caveat: exposed for validation only; it uses label-free normalized margins and is not the default controller route.
- replay result path: `artifacts/reports/probes/bg_controller_replay_2026-05-18.md`
- usage doc path: `docs/evaluator/controller-usage.md`
- interpretation: the first BG Phase 1 controller layer is ready as a read-only branch-selection component over existing candidate features.

## Read-only transformer BG integration + best-of-N smoke (2026-05-18)

- BG_TRANSFORMER_CAPTURE_INSPECTION_VERDICT = READY
- BG_TRANSFORMER_BEST_OF_N_SMOKE_VERDICT = PASS
- BG_TRANSFORMER_INTEGRATION_VERDICT = PASS
- BG_DEVIL_TASK_INVENTORY_VERDICT = READY
- BG_DEVIL_BEST_OF_N_VERDICT = PASS
- BG_TRANSFORMER_UNIT_TEST_VERDICT = PASS
- module paths: `src/evaluator/bg_transformer_features.py`, `utilities/tests/manual/run_bg_transformer_best_of_n_smoke.py`
- smoke report paths: `artifacts/reports/probes/bg_transformer_best_of_n_smoke_all_2026-05-18.md`, `artifacts/reports/probes/bg_transformer_capture_inspection_2026-05-18.md`
- devil task report path: `artifacts/reports/probes/bg_devil_task_inventory_2026-05-18.md`
- interpretation: the domain-routed BG controller has now been exercised on live generated candidates across reasoning, code, science, unknown/general, and two local devil code prompts; this validates branch-selection plumbing, not standalone answer quality.

## BG steering and partial-trajectory routing suite (2026-05-18)

- BG_STEERING_PREFLIGHT_VERDICT = READY
- BG_STEERING_TASK_SUITE_VERDICT = READY
- BG_BRANCH_POOL_VERDICT = READY
- BG_REACHABILITY_GATE_VERDICT = READY
- BG_PARTIAL_FEATURE_VERDICT = READY
- BG_PARTIAL_ROUTING_VERDICT = NEUTRAL
- BG_COMPUTE_ALLOCATION_VERDICT = INSUFFICIENT
- BG_WRAPPER_MATCHED_VERDICT = SKIPPED
- BG_SOFT_STEERING_VERDICT = STABLE_NO_EFFECT
- BG_LATENT_BRANCH_SELECTION_VERDICT = HELPS
- OVERALL_BG_STEERING_VERDICT = NEUTRAL
- generator reachability result: reasoning, science, and GSM8K passed the gate; direct Ouro code and devil branches did not.
- devil task result: no oracle-success devil branch was produced in the early gate.
- full report paths: `artifacts/reports/probes/bg_steering_suite_2026-05-18/summary.md`, `artifacts/reports/probes/bg_steering_suite_2026-05-18/analysis.md`, `docs/evaluator/steering-and-routing-suite.md`
- interpretation: BG improves selection directionally but not enough for a routing-policy promotion; code remains the clearest generator-reachability bottleneck.

## Local-agent candidate export interface (2026-05-18)

- WRAPPER_CANDIDATE_PATH_INVENTORY_VERDICT = READY
- WRAPPER_CANDIDATE_EXPORT_UNIT_VERDICT = PASS
- WRAPPER_CANDIDATE_EXPORT_SMOKE_VERDICT = SKIPPED
- WRAPPER_CANDIDATE_EXPOSURE_VERDICT = READY
- files modified: `src/local_agent/candidate_export.py`, `src/local_agent/candidate_capture.py`, `src/local_agent/ouro_direct.py`, `src/local_agent/ouro_agent_improved.py`
- API path: `src/local_agent/candidate_capture.py`
- docs/report paths: `docs/evaluator/local-agent-candidate-export.md`, `artifacts/reports/probes/local_agent_candidate_exposure_2026-05-18_summary.md`
- interpretation: future wrapper-matched BG experiments can compare wrapper baseline and BG-selected branches over the same exported candidate set.

## Wrapper-matched BG candidate selection (2026-05-18)

- WRAPPER_TRACE_GENERATION_VERDICT = READY
- WRAPPER_CANDIDATE_EVAL_VERDICT = READY
- WRAPPER_GENERATOR_REACHABILITY_VERDICT = REACHABLE
- WRAPPER_BG_FEATURE_VERDICT = READY
- WRAPPER_MATCHED_BG_VERDICT = NEUTRAL
- BG_VS_RANDOM_VERDICT = NEUTRAL
- BG_VS_STAGE_HEURISTIC_VERDICT = NEUTRAL
- WRAPPER_MATCHED_EXPERIMENT_VERDICT = READY
- devil result: `offline_dynamic_connectivity` and `minimum_xor_paths` remained wrong-code only.
- report paths: `artifacts/reports/probes/wrapper_bg_matched_2026-05-18/summary.md`, `artifacts/reports/probes/wrapper_bg_matched_2026-05-18/bg_selection.md`, `docs/evaluator/wrapper-matched-bg-selection.md`
- interpretation: on wrapper-exported code candidates, BG did not outperform the wrapper's own final selection, random expectation, or a simple stage heuristic.
## BG trajectory prediction sweep (2026-05-18)

BG_TRAJECTORY_PREFLIGHT_VERDICT = `READY`.
BG_TRAJECTORY_TASK_SUITE_VERDICT = `READY`.
BG_TRAJECTORY_PARTIALS_VERDICT = `READY`.
BG_TRAJECTORY_CONTINUATION_VERDICT = `READY`.
BG_TRAJECTORY_PREFIX_FEATURE_VERDICT = `READY`.
BG_TRAJECTORY_PREFIX_SCORE_VERDICT = `READY`.
BG_TRAJECTORY_PREDICTION_VERDICT = `STRONG`.
BEST_PREDICTIVE_CELL = `{'domain': 'reasoning', 'prefix_length': 256, 'head_id': 'mixed::MIX_CODE_REASONING::36_mean::AntisymLinear', 'config': '36_mean', 'architecture': 'AntisymLinear', 'top1_lift': 0.16249999999999998, 'top2_lift': 0.04166666666666663, 'pairwise_accuracy': 0.8536585365853658, 'oracle_success': 0.9, 'n_tasks': 20, 'n_pairwise_comparisons': 41}`.
RECOMMENDED_STEERING_TARGET = `{'domain': 'reasoning', 'prefix_length': 256, 'head_id': 'mixed::MIX_CODE_REASONING::36_mean::AntisymLinear', 'head_config': '36_mean', 'architecture': 'AntisymLinear', 'top1_lift': 0.16249999999999998, 'top2_lift': 0.04166666666666663, 'pairwise_accuracy': 0.8536585365853658, 'oracle_success': 0.9}`.
GENERATOR_REACHABILITY_LIMITED = `false`.
Interpretation: Run a targeted Stage 2 steering-sensitivity probe at the best predictive cell. Measure state movement in the BG-readable direction, output stability, final correctness, and positive-vs-negative-vs-random controls.
Full reports: `artifacts/reports/probes/bg_trajectory_prediction_2026-05-18/summary.md`, `artifacts/reports/probes/bg_trajectory_prediction_2026-05-18/predictive_power.md`, `artifacts/reports/probes/bg_trajectory_prediction_2026-05-18/stage2_recommendation.md`.

## BG Stage 2 layer-hook follow-up (2026-05-18)

BG_STAGE2_PARTIAL_TRACE_AUDIT_VERDICT = READY
BG_LAYERHOOK_FOLLOWUP_PREFLIGHT_VERDICT = READY
BG_LAYERHOOK_FOLLOWUP_TASKS_VERDICT = READY
BG_LAYERHOOK_FOLLOWUP_SWEEP_VERDICT = READY
BG_LAYERHOOK_MECHANICAL_VERDICT = READY
BG_LAYERHOOK_SIGNED_CAUSAL_VERDICT = UNSIGNED_EFFECT
BG_SINGLE_LOOP_POSITION_VERDICT = L1_BETTER
BG_MULTILOOP_VERDICT = MULTILOOP_STRONGER
BG_LAYERHOOK_STABILITY_VERDICT = STABLE_BUT_TINY
BG_FINAL_TASK_LIFT_VERDICT = INSUFFICIENT
BG_LAYERHOOK_FOLLOWUP_VERDICT = READ_ONLY_BG_FOR_NOW
BEST_SINGLE_LOOP_MODE = single_loop_L1
BEST_MULTILOOP_MODE = multi_loop_decayed
MULTILOOP_GAIN_OVER_BEST_SINGLE = 0.0706979167497257
RECOMMENDED_NEXT = keep_BG_as_readout_selector_and_revisit_steering_with_empirical_success_direction_or_training

Interpretation: BG remains more reliable as a readout selector than as an inference-time steering vector under this protocol.

Full reports: `docs/evaluator/stage2-layerhook-followup.md`, `artifacts/reports/probes/bg_stage2_layerhook_followup_2026-05-18/summary.md`, `artifacts/reports/probes/bg_stage2_layerhook_followup_2026-05-18/analysis.md`.

## BG empirical steering direction probe (2026-05-18)

BG_EMPIRICAL_STEERING_PREFLIGHT_VERDICT = READY
BG_EMPIRICAL_DIRECTION_BUILD_VERDICT = READY
BG_EMPIRICAL_STEERING_TASKS_VERDICT = READY
BG_EMPIRICAL_STEERING_SWEEP_VERDICT = READY
BG_EMPIRICAL_DIRECTION_CAUSAL_VERDICT = EMPIRICAL_UNSIGNED_ONLY
BG_EMPIRICAL_VS_RAW_VERDICT = EMPIRICAL_BEATS_RAW
BG_STEERING_DIRECTION_GEOMETRY_VERDICT = RAW_READOUT_NOT_PRODUCTION_DIRECTION
BG_EMPIRICAL_STEERING_STABILITY_VERDICT = DESTABILIZING
BG_EMPIRICAL_FINAL_LIFT_VERDICT = NEGATIVE_LIFT
BG_TINY_STEERING_ADAPTER_VERDICT = NO_BETTER_THAN_STATIC
BG_EMPIRICAL_STEERING_VERDICT = DESTABILIZING
MODE_COVERAGE = {"EMPIRICAL_MEAN_DIFF": {"multi_loop_decayed": {"complete_expected_rows": 48, "intervention_rows": 48, "task_count": 6}, "single_loop_L1": {"complete_expected_rows": 48, "intervention_rows": 48, "task_count": 6}}, "EMPIRICAL_WHITENED_DIFF": {"multi_loop_decayed": {"complete_expected_rows": 48, "intervention_rows": 48, "task_count": 6}, "single_loop_L1": {"complete_expected_rows": 48, "intervention_rows": 48, "task_count": 6}}, "LOGISTIC_SUCCESS_PROBE": {"multi_loop_decayed": {"complete_expected_rows": 48, "intervention_rows": 48, "task_count": 6}, "single_loop_L1": {"complete_expected_rows": 48, "intervention_rows": 48, "task_count": 6}}, "RAW_NONORM_READOUT": {"multi_loop_decayed": {"complete_expected_rows": 48, "intervention_rows": 48, "task_count": 6}, "single_loop_L1": {"complete_expected_rows": 48, "intervention_rows": 48, "task_count": 6}}}
MULTILOOP_DECAYED_VS_L1_DELTA = -0.10560436938609094
Interpretation: empirical directions test whether BG is readout-only or whether calibrated success-space directions can become causal handles.
Full reports: `artifacts/reports/probes/bg_empirical_steering_direction_2026-05-18/summary.md`, `artifacts/reports/probes/bg_empirical_steering_direction_2026-05-18/analysis.md`, `docs/evaluator/empirical-steering-direction.md`.

## BG pre-consolidation control probes (2026-05-18)

- BG_RMS_STEERING_VERDICT = `RMS_UNSIGNED_ONLY`
- BG_RMS_VS_L2_VERDICT = `RMS_MATCHES_L2`
- BG_PROPAGATION_VERDICT = `PROPAGATES_TO_LATER_STATES`
- BG_PROPAGATION_DECAY_PROFILE = `SURVIVES_32_TOKENS`
- BG_TEXT_PREFIX_EXPANSION_VERDICT = `WEAK_POSITIVE`
- BG_CAUSAL_GRADIENT_VERDICT = `GRADIENT_NO_BETTER_THAN_RANDOM`
- BG_INFERENCE_TIME_STEERING_VERDICT = `UNSIGNED_ONLY`
- BG_BRANCH_ALLOCATION_VERDICT = `PROMISING`
- BG_PHASE2_REQUIREMENT_VERDICT = `TRAINING_REQUIRED`
- interpretation: The model can be nudged in BG-readable state space, but tested directions do not provide reliable signed control; Phase 2 training is required.
- reports: `artifacts/reports/probes/bg_preconsolidation_control_probes_2026-05-18/summary.md`, `artifacts/reports/probes/bg_preconsolidation_control_probes_2026-05-18/final_analysis.json`, `docs/evaluator/preconsolidation-control-probes.md`

## BG causal intervention adapter (2026-05-18)

BG_CAUSAL_ADAPTER_PREFLIGHT_VERDICT = READY
BG_CAUSAL_ADAPTER_DATASET_VERDICT = READY
BG_CAUSAL_ADAPTER_IMPLEMENTATION_VERDICT = READY
BG_CAUSAL_ADAPTER_TRAINING_VERDICT = PARTIAL
BG_CAUSAL_ADAPTER_TEACHER_FORCED_VERDICT = ADAPTER_IMPROVES_LOGIT_MARGIN
BG_CAUSAL_ADAPTER_FREE_GEN_VERDICT = TEACHER_FORCED_ONLY
BG_PAIRWISE_CAUSAL_CONTRAST_ADAPTER_VERDICT = SKIPPED
BG_CAUSAL_ADAPTER_LEARNING_VERDICT = LEARNS_LOGIT_CONTROL
BG_CAUSAL_ADAPTER_VS_STATIC_VERDICT = ADAPTER_BEATS_STATIC
BG_CAUSAL_ADAPTER_STABILITY_VERDICT = STABLE
BG_CAUSAL_ADAPTER_GENERATION_TRANSFER_VERDICT = TEACHER_FORCED_ONLY
BG_CAUSAL_ADAPTER_VERDICT = LOCAL_LOGIT_CONTROL_ONLY
TEACHER_FORCED_RESULT_INTERPRETATION = TEACHER_FORCED_SHORTCUT_RISK
FREE_GENERATION_EVAL_COMPLETED = true
KL_ANSWER_POSITION_MASKED = true
INTERVENTION_POSITION_KIND = prefix_last_token

The causal adapter test separates local teacher-forced logit control from actual trajectory transfer; overall verdict is LOCAL_LOGIT_CONTROL_ONLY.

Full reports: `artifacts/reports/probes/bg_causal_intervention_adapter_2026-05-18/summary.md`, `artifacts/reports/probes/bg_causal_intervention_adapter_2026-05-18/analysis.md`, `docs/evaluator/causal-intervention-adapter.md`.

## BG sequence-level adapter / final frozen-backbone steering test (2026-05-18)

- BG_SEQUENCE_PARSER_AUDIT_VERDICT: `READY`
- BG_SEQUENCE_REWARD_DISTRIBUTION_VERDICT: `REWARD_SIGNAL_USABLE`
- BG_SEQUENCE_OPTIMIZER_SANITY_MICRO_VERDICT: `OPTIMIZER_MOVES_ADAPTER`
- BG_SEQUENCE_GPU_THROUGHPUT_VERDICT: `OVERNIGHT_FEASIBLE`
- OVERNIGHT_SEQUENCE_ADAPTER_READINESS: `READY`
- BG_SEQUENCE_ADAPTER_PREFLIGHT_VERDICT: `READY`
- BG_SEQUENCE_ADAPTER_DATASET_VERDICT: `PARTIAL`
- BG_SEQUENCE_ADAPTER_IMPLEMENTATION_VERDICT: `READY`
- BG_SEQUENCE_BASELINE_EVAL_VERDICT: `READY`
- BG_SEQUENCE_OPTIMIZER_SANITY_VERDICT: `OPTIMIZER_CAN_LEARN_TRIVIAL_TARGET`
- BG_SEQUENCE_ADAPTER_TRAINING_VERDICT: `SEQUENCE_REWARD_IMPROVES`
- BG_SEQUENCE_ADAPTER_HELDOUT_VERDICT: `NO_ADAPTER_SPECIFIC_TRANSFER`
- BG_SEQUENCE_ADAPTER_TEACHER_FORCED_DIAG_VERDICT: `NO_LOGIT_MARGIN_EFFECT`
- BG_SEQUENCE_ADAPTER_BG_SCORE_DIAG_VERDICT: `BG_SCORE_MOVES`
- BG_SEQUENCE_ADAPTER_GEOMETRY_VERDICT: `MATCHES_PRIOR_DIRECTIONS`
- BG_SEQUENCE_ADAPTER_LEARNING_VERDICT: `LEARNS_SEQUENCE_REWARD`
- BG_SEQUENCE_ADAPTER_VS_RANDOM_VERDICT: `WORSE_THAN_RANDOM`
- BG_SEQUENCE_ADAPTER_STABILITY_VERDICT: `STABLE`
- BG_SEQUENCE_ADAPTER_TRANSFER_VERDICT: `NO_TRANSFER`
- BG_SEQUENCE_LEVEL_ADAPTER_VERDICT: `NO_FROZEN_BACKBONE_WRITE_PATH`
- FROZEN_BACKBONE_INFERENCE_STEERING_STATUS: `CLOSED_UNDER_TESTED_METHODS`
- STOPPING_RULE_APPLIES: `True`
- STOPPING_RULE_SCOPE: `safe_alpha_leq_0_02_under_tested_optimizers`
- RECOMMENDED_NEXT: `consolidate_phase1_phase1_5_and_design_phase2_training_time_integration`
- STOPPING_RULE_SCOPE: `safe_alpha_leq_0_02_under_tested_optimizers`
- report paths:
  - `artifacts/reports/probes/bg_sequence_adapter_quick_preflight_2026-05-18/summary.md`
  - `artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/preflight.md`
  - `artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/sequence_adapter_dataset.md`
  - `artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/implementation_tests.md`
  - `artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/baseline_eval.md`
  - `artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/optimizer_sanity.md`
  - `artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/sequence_training_report.md`
  - `artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/heldout_free_generation_eval.md`
  - `artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/diagnostics.md`
  - `artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/analysis.md`
  - `artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/summary.md`

## Same-prefix hidden-state branch generation suite (2026-05-18)

- report: `docs/evaluator/hidden-state-branch-generation.md`
- artifacts: `artifacts/reports/probes/bg_hidden_state_branch_generation_2026-05-18/`
- `BG_HIDDEN_BRANCH_FEASIBILITY_VERDICT = HOOK_HIDDEN_ORIGIN_READY`
- `LIVE_BRANCH_METHOD = hook_intervention_per_branch`
- `BG_HIDDEN_BRANCH_GENERATION_VERDICT = HOOK_HIDDEN_ORIGIN_BRANCHES_GENERATED`
- `BG_LATENT_BRANCH_PERSISTENCE_VERDICT = LATENT_BRANCHES_PERSIST_TO_47`
- `BG_HIDDEN_BRANCH_OUTCOME_DATASET_VERDICT = READY`
- `BG_HIDDEN_ORIGIN_BRANCH_SELECTION_VERDICT = NO_HIDDEN_BRANCH_SELECTION_SIGNAL`
- `BG_HIDDEN_BRANCH_L30_L42_GATE_VERDICT = NEEDS_STRONGER_BRANCH_GENERATOR`
- `BG_HIDDEN_BRANCH_ADAPTIVE_THRESHOLD_VERDICT = TOPK_SUFFICIENT`
- `PHASE2_HIDDEN_BRANCH_READINESS = NEEDS_BETTER_BRANCH_EVALUATOR`
- interpretation: frozen BG heads that transfer across cached text/candidate branches did not become reliable selectors for same-prefix hidden-origin branches. This is a hidden-origin evaluator/calibration gap, not evidence that cached candidate transfer was invalid.
## Hidden-origin branch taps (2026-05-18)

- PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS = `DATA_LIMITED`
- tap_eval_verdict = `INSUFFICIENT`
- tap_training_verdict = `READY`
- layer_config_verdict = `INSUFFICIENT`
- geometry_verdict = `ALIGNS_WITH_OLD_TAPS`
- report: `artifacts/reports/probes/bg_hidden_origin_taps_2026-05-18/summary.md`

Generate more hidden-origin branch outcome groups.

## Hidden-origin branch diversity v2 and tap reevaluation (2026-05-18)

- PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_V2 = `WEAK`
- generation_verdict = `READY`
- dataset_verdict = `SMALL_BUT_USABLE`
- training_verdict = `READY`
- eval_verdict = `WEAK_SELECTOR`
- layer_config_verdict = `CONCAT_REQUIRED`
- geometry_verdict = `OLD_GEOMETRY_CONFIRMED`
- report: `artifacts/reports/probes/bg_hidden_origin_diversity_v2_2026-05-18/summary.md`

Either expand once more or proceed only to a small selection-only prototype with the caveat locked in.

## Hidden-origin branch diversity v3 and selector reevaluation (2026-05-18)

- PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_V3 = `STILL_DATA_LIMITED`
- HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE = `v3_hidden_origin_tap`
- diversity_ablation_verdict = `DIVERSITY_IMPROVED`
- driver_verdict = `NON_RANDOM_DIRECTIONS_HELP`
- dataset_verdict = `STILL_DATA_LIMITED`
- training_verdict = `WEAK`
- eval_verdict = `DATA_LIMITED`
- geometry_verdict = `OLD_GEOMETRY_CONFIRMED`
- report: `artifacts/reports/probes/bg_hidden_origin_diversity_v3_2026-05-18/summary.md`

Continue targeted data expansion using the v3 recipe before making selector-readiness claims.

## Hidden-origin branch split salvage and selector reevaluation (2026-05-18)

- `BG_HIDDEN_ORIGIN_SALVAGE_EVAL_VERDICT = WEAK_SELECTOR`
- `BG_HIDDEN_ORIGIN_CV_STABILITY_VERDICT = STABLE_POSITIVE`
- `BG_HIDDEN_ORIGIN_V4_QUOTA_NEED_VERDICT = V4_REQUIRED_HELDOUT_BALANCE`
- `HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE = old_frozen_bg`
- `PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_SALVAGE = WEAK`

Split salvage reused existing v3 branch data only. It reports strict/v3-clean heldout support separately from grouped-CV diagnostics and marks baseline contamination where applicable.
## Hidden-origin branch quota v4 and old-context replay (2026-05-18)

- `BG_HIDDEN_ORIGIN_QUOTA_GENERATION_V4_VERDICT = PARTIAL`
- `BG_HIDDEN_ORIGIN_SELECTOR_EVAL_V4_VERDICT = STILL_DATA_LIMITED`
- `BG_HIDDEN_ORIGIN_OLD_CONTEXT_REPLAY_V4_VERDICT = PARTIAL_MATCH`
- `BG_HIDDEN_ORIGIN_TAP_GEOMETRY_V4_VERDICT = OLD_GEOMETRY_CONFIRMED`
- `HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE_V4 = ensemble`
- `PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_V4 = STILL_DATA_LIMITED`

V4 reserves train/val/heldout task IDs before generation and keeps old-context replay diagnostic-only. Alpha 0.02, sampled labels, and L47 remain excluded from primary readiness claims.

## Hidden-origin Branch Generator v1 (2026-05-18)

Branch Generator v1 was run because v4 confirmed selector geometry but remained heldout-diversity limited. The v1 run tested early L24/L1-style hook perturbations, high-yield non-random directions, a lightweight recipe/CEM schedule, true fork/carry feasibility, and richer outcome diagnostics without training Ouro or changing production routing.

BG_BRANCH_GENERATOR_V1_AUDIT_PLAN_VERDICT = READY
BG_TRUE_FORK_CARRY_PROBE_V1_VERDICT = HOOK_FALLBACK_ONLY
BG_RICH_OUTCOME_SCHEMA_V1_VERDICT = READY
BG_BRANCH_GENERATOR_BASIS_BANK_V1_VERDICT = READY
BG_BRANCH_GENERATOR_PROPOSER_TRAINING_V1_VERDICT = RECIPE_ONLY
BG_BRANCH_GENERATOR_BLACKBOX_SEARCH_V1_VERDICT = WEAK_IMPROVEMENT
BG_BRANCH_GENERATOR_V1_GENERATION_VERDICT = HELDOUT_QUOTA_MET_ONLY
BG_BRANCH_GENERATOR_V1_DIVERSITY_VERDICT = STRONG_IMPROVEMENT
BG_BRANCH_GENERATOR_V1_BEST_METHOD = hs_inspired_controller
BG_BRANCH_GENERATOR_V1_SELECTOR_DATASET_VERDICT = HELDOUT_READY_TRAIN_WEAK
BG_BRANCH_GENERATOR_V1_SELECTOR_TRAINING_VERDICT = WEAK
BG_BRANCH_GENERATOR_V1_SELECTOR_EVAL_VERDICT = WEAK_SELECTOR
BG_BRANCH_GENERATOR_V1_OLD_CONTEXT_REPLAY_VERDICT = PARTIAL_MATCH
BG_BRANCH_GENERATOR_V1_GEOMETRY_VERDICT = OLD_GEOMETRY_CONFIRMED
HIDDEN_ORIGIN_BRANCH_GENERATOR_STATUS_V1 = WEAK_BUT_USABLE
HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE_AFTER_GENERATOR_V1 = v4_hidden_origin_tap

- quota_progress_by_split: `{'all_minimums_met': False, 'heldout': {'behaviorally_diverse_groups': 43, 'behaviorally_diverse_groups_per_100_rows': 6.554878048780488, 'candidate_pairs': 2296, 'groups': 82, 'minimum_met': True, 'non_tie_pairs': 419, 'non_tie_pairs_per_100_rows': 63.8719512195122, 'parse_rate': 0.8262195121951219, 'quota_minimums': {'behaviorally_diverse_groups': 20, 'non_tie_pairs': 120, 'task_ids': 8}, 'reward_diverse_groups': 38, 'stability_rate': 1.0, 'stable_primary_rows': 656, 'task_ids': 10, 'task_ids_with_non_tie_pair_list': ['OpenBookQA/14', 'mmlu/anatomy/12', 'mmlu/anatomy/7', 'mmlu/anatomy/8', 'mmlu/high_school_chemistry/1', 'mmlu/high_school_chemistry/10', 'mmlu/high_school_physics/11', 'sciq/sciq/22'], 'task_ids_with_non_tie_pairs': 8, 'tie_pairs': 1877, 'tie_rate': 0.8175087108013938}, 'train': {'behaviorally_diverse_groups': 35, 'behaviorally_diverse_groups_per_100_rows': 1.4583333333333333, 'candidate_pairs': 8400, 'groups': 300, 'minimum_met': False, 'non_tie_pairs': 473, 'non_tie_pairs_per_100_rows': 19.708333333333332, 'parse_rate': 0.6891666666666667, 'quota_minimums': {'behaviorally_diverse_groups': 60, 'non_tie_pairs': 250, 'task_ids': 24}, 'reward_diverse_groups': 34, 'stability_rate': 1.0, 'stable_primary_rows': 2400, 'task_ids': 40, 'task_ids_with_non_tie_pair_list': ['ARC-Challenge/1', 'ARC-Challenge/17', 'ARC-Challenge/19', 'ARC-Challenge/2', 'OpenBookQA/3'], 'task_ids_with_non_tie_pairs': 5, 'tie_pairs': 7927, 'tie_rate': 0.9436904761904762}, 'val': {'behaviorally_diverse_groups': 1, 'behaviorally_diverse_groups_per_100_rows': 0.1, 'candidate_pairs': 3452, 'groups': 127, 'minimum_met': False, 'non_tie_pairs': 7, 'non_tie_pairs_per_100_rows': 0.7, 'parse_rate': 0.874, 'quota_minimums': {'behaviorally_diverse_groups': 15, 'non_tie_pairs': 60, 'task_ids': 6}, 'reward_diverse_groups': 1, 'stability_rate': 1.0, 'stable_primary_rows': 1000, 'task_ids': 8, 'task_ids_with_non_tie_pair_list': ['mmlu/high_school_chemistry/16'], 'task_ids_with_non_tie_pairs': 1, 'tie_pairs': 3445, 'tie_rate': 0.9979721900347625}}`
- diversity_questions: `{'CEM_or_ES_improved_over_HS': False, 'K6_yield': 0.0, 'K8_remained_useful': True, 'K8_yield': 1.971057884231537, 'L24_remained_better_than_L36': True, 'L24_yield': 1.9863013698630136, 'L36_yield': 1.8485915492957747, 'alpha_0_005_remained_best': False, 'alpha_0_005_yield': 1.8851508120649652, 'alpha_0_01_yield': 2.3026315789473686, 'beat_static_v4_recipe': True, 'cem_yield': 1.957070707070707, 'hs_yield': 2.217741935483871, 'learned_proposer_helped': True, 'non_random_directions_remained_useful': True, 'non_random_yield': 3.75, 'random_yield': 2.1169354838709675, 'static_yield': 1.8333333333333333, 'structured_low_rank_coefficients_helped': False, 'true_behavioral_diversity_not_instability': True, 'true_fork_carry_changed_persistence': False}`
- recommended_next: `Either run a small selection-only prototype with caveat or run targeted generator v1.1 if one recipe clearly remains.`

Selector readiness, if claimed, uses only primary-safe deterministic alpha <= 0.01 heldout rows. Diagnostic alpha 0.02, sampled labels, L47 branches, old-context replay, and auxiliary diagnostics are not readiness support.


## Universal branch-content taps v1 (2026-05-18)

Universal Branch-Content Taps v1 tested whether one tiny hidden-state pairwise evaluator can cover both old content/candidate selection and same-prefix hidden-origin branch survival. It trained only new standalone tap heads and did not alter Ouro, existing BG taps, registries, wrapper/local-agent routing, or production behavior.

BG_UNIVERSAL_TAP_INVENTORY_VERDICT = READY
BG_UNIVERSAL_OLD_CONTENT_DATASET_VERDICT = READY
BG_UNIVERSAL_HIDDEN_BRANCH_DATASET_VERDICT = READY
BG_UNIVERSAL_BRIDGE_DATASET_VERDICT = READY
BG_UNIVERSAL_DATA_EXPANSION_VERDICT = SKIPPED
BG_UNIVERSAL_TAP_DATASET_VERDICT = READY
BG_UNIVERSAL_TAP_TRAINING_VERDICT = READY
BG_UNIVERSAL_OLD_CONTEXT_EVAL_VERDICT = MATCHES_OR_BEATS_OLD_TAPS
BG_UNIVERSAL_HIDDEN_BRANCH_EVAL_VERDICT = SMALL_DEGRADATION
BG_UNIVERSAL_BRIDGE_EVAL_VERDICT = NO_BRIDGE_SIGNAL
BG_UNIVERSAL_LAYERWISE_PRUNING_VERDICT = TOPK_SURVIVAL_ONLY
BG_UNIVERSAL_DOMAIN_GENERALIZATION_VERDICT = REASONING_SCIENCE_ONLY
BG_UNIVERSAL_TAP_GEOMETRY_VERDICT = OLD_GEOMETRY_CONFIRMED
UNIVERSAL_BRANCH_CONTENT_TAP_STATUS = FUSION_NEEDED

- old_content_counts: `{'feature_config_counts': {'24_L4': 462, '24_mean': 462, '30_L4': 0, '36_L4': 462, '36_mean': 462, '42_L4': 0, '47_L4': 462, '47_mean': 462, 'concat_24_30_36': 0, 'concat_24_36': 462, 'concat_24_36_47': 462, 'concat_36_42_47': 0, 'concat_36_47': 462}, 'pairs': 462, 'pairs_by_domain': {'math_simple_arithmetic': 143, 'reasoning': 183, 'science': 136}, 'pairs_by_split': {'heldout': 73, 'train': 295, 'val': 94}, 'pairs_by_type': {'old_content': 462}, 'tasks_by_split': {'heldout': ['ARC-Challenge/0', 'ARC-Challenge/19', 'gsm8k/1', 'gsm8k/12', 'gsm8k/5', 'mmlu/high_school_biology/10', 'mmlu/high_school_biology/12', 'mmlu/high_school_biology/17', 'mmlu/high_school_biology/9'], 'train': ['ARC-Challenge/1', 'ARC-Challenge/11', 'ARC-Challenge/12', 'ARC-Challenge/13', 'ARC-Challenge/14', 'ARC-Challenge/15', 'ARC-Challenge/16', 'ARC-Challenge/17', 'ARC-Challenge/18', 'ARC-Challenge/6', 'ARC-Challenge/8', 'gsm8k/0', 'gsm8k/10', 'gsm8k/11', 'gsm8k/13', 'gsm8k/14', 'gsm8k/16', 'gsm8k/17', 'gsm8k/19', 'gsm8k/2', 'gsm8k/3', 'gsm8k/7', 'gsm8k/8', 'mmlu/high_school_biology/1', 'mmlu/high_school_biology/11', 'mmlu/high_school_biology/15', 'mmlu/high_school_biology/19', 'mmlu/high_school_biology/2', 'mmlu/high_school_biology/3', 'mmlu/high_school_biology/4', 'mmlu/high_school_biology/5', 'mmlu/high_school_biology/7', 'mmlu/high_school_biology/8'], 'val': ['ARC-Challenge/10', 'ARC-Challenge/2', 'ARC-Challenge/3', 'ARC-Challenge/4', 'ARC-Challenge/5', 'ARC-Challenge/7', 'ARC-Challenge/9', 'gsm8k/15', 'gsm8k/9', 'mmlu/high_school_biology/14', 'mmlu/high_school_biology/18']}}`
- hidden_branch_counts: `{'feature_config_counts': {'24_L4': 1753, '24_mean': 1753, '30_L4': 1753, '36_L4': 1753, '36_mean': 1753, '42_L4': 1753, '47_L4': 1753, '47_mean': 1753, 'concat_24_30_36': 1753, 'concat_24_36': 1753, 'concat_24_36_47': 1753, 'concat_36_42_47': 1753, 'concat_36_47': 1753}, 'pairs': 1753, 'pairs_by_domain': {'reasoning': 1074, 'science': 679}, 'pairs_by_split': {'heldout': 483, 'train': 1090, 'val': 180}, 'pairs_by_type': {'hidden_branch': 1753}, 'tasks_by_split': {'heldout': ['OpenBookQA/14', 'OpenBookQA/18', 'mmlu/anatomy/12', 'mmlu/anatomy/7', 'mmlu/anatomy/8', 'mmlu/high_school_chemistry/1', 'mmlu/high_school_chemistry/10', 'mmlu/high_school_physics/11', 'sciq/sciq/22'], 'train': ['ARC-Challenge/1', 'ARC-Challenge/17', 'ARC-Challenge/19', 'ARC-Challenge/2', 'OpenBookQA/3', 'mmlu/anatomy/12', 'mmlu/anatomy/8', 'mmlu/high_school_chemistry/10', 'mmlu/high_school_physics/11', 'sciq/sciq/22'], 'val': ['ARC-Challenge/17', 'mmlu/high_school_chemistry/16']}}`
- bridge_counts: `{'feature_config_counts': {'24_L4': 2142, '24_mean': 2142, '30_L4': 2142, '36_L4': 2142, '36_mean': 2142, '42_L4': 2142, '47_L4': 2142, '47_mean': 2142, 'concat_24_30_36': 2142, 'concat_24_36': 2142, 'concat_24_36_47': 2142, 'concat_36_42_47': 2142, 'concat_36_47': 2142}, 'pairs': 2142, 'pairs_by_domain': {'reasoning': 1316, 'science': 826}, 'pairs_by_split': {'heldout': 580, 'train': 1342, 'val': 220}, 'pairs_by_type': {'bridge': 2142}, 'tasks_by_split': {'heldout': ['OpenBookQA/14', 'OpenBookQA/18', 'mmlu/anatomy/12', 'mmlu/anatomy/7', 'mmlu/anatomy/8', 'mmlu/high_school_chemistry/1', 'mmlu/high_school_chemistry/10', 'mmlu/high_school_physics/11', 'sciq/sciq/22'], 'train': ['ARC-Challenge/1', 'ARC-Challenge/17', 'ARC-Challenge/19', 'ARC-Challenge/2', 'OpenBookQA/3', 'mmlu/anatomy/12', 'mmlu/anatomy/8', 'mmlu/high_school_chemistry/10', 'mmlu/high_school_physics/11', 'sciq/sciq/22'], 'val': ['ARC-Challenge/17', 'mmlu/high_school_chemistry/16']}}`
- recommendation: `Build an explicit composite selector rather than forcing a single universal head.`

Readiness requires old-context, hidden-branch, and bridge support. Cached coding features were inspected but had no non-tie within-task labels, so coding remains coverage-limited.

## Gated branch-content selector v1 (2026-05-18)

Gated/Fusion Branch-Content Selector v1 tested whether old content taps, hidden-origin branch taps, bridge heads, universal heads, and readiness diagnostics can be combined without collapsing all roles into one linear universal tap.

BG_GATED_SELECTOR_INVENTORY_VERDICT = READY
BG_GATED_SELECTOR_EXPERT_SCORES_VERDICT = READY
BG_GATED_SELECTOR_DATASET_VERDICT = READY
BG_GATED_SELECTOR_TRAINING_VERDICT = READY
BG_GATED_SELECTOR_EXPERT_ABLATION_VERDICT = INCONCLUSIVE
BG_GATED_OLD_CONTEXT_EVAL_VERDICT = MATCHES_OR_BEATS_OLD_TAPS
BG_GATED_HIDDEN_BRANCH_EVAL_VERDICT = SMALL_DEGRADATION
BG_GATED_BRIDGE_EVAL_VERDICT = BRIDGE_FIXED
BG_GATED_LAYERWISE_PRUNING_VERDICT = OLD_NEW_COMPOSITE_BEST
BG_GATED_DOMAIN_COVERAGE_VERDICT = MULTIDOMAIN_READY
BG_GATED_CALIBRATION_OOD_VERDICT = CALIBRATION_WEAK
BG_GATED_GEOMETRY_VERDICT = OLD_GEOMETRY_DOMINATES
BG_GATED_AS_OLD_TAP_REPLACEMENT_VERDICT = SAFE_REPLACEMENT_CANDIDATE
GATED_BRANCH_CONTENT_SELECTOR_STATUS = OLD_NEW_COMPOSITE_SUFFICIENT

- recommendation: `Prefer the simpler old+branch+bridge composite over the learned gate for now; keep top-k survival and do not change production routing.`
- no Ouro weights, tokenizer files, checkpoints, old taps, tap registries, wrapper/local-agent routing, or production routing were modified.
- expert/tap scores were used only as input features, not as labels.

## Fixed-composite branch survival policy v1 (2026-05-18)

This run converted the corrected gated selector result into a validation-selected fixed old+branch+bridge survival policy with explicit veto/rescue and missing-expert/OOD fallback.

BG_FIXED_COMPOSITE_SURVIVAL_INVENTORY_VERDICT = READY
BG_FIXED_COMPOSITE_SURVIVAL_DATASET_VERDICT = READY
BG_FIXED_COMPOSITE_SURVIVAL_FEATURES_VERDICT = READY
BG_FIXED_COMPOSITE_SURVIVAL_BASELINES_VERDICT = READY
BG_FIXED_COMPOSITE_OPTIMIZATION_VERDICT = OLD_BRANCH_BRIDGE_SUFFICIENT
BG_FIXED_COMPOSITE_VETO_RESCUE_OPTIMIZATION_VERDICT = READY
BG_FIXED_COMPOSITE_LEARNED_RESCUE_VERDICT = WORSE_THAN_RULES
BG_FIXED_COMPOSITE_MISSING_OOD_POLICY_VERDICT = ROBUST
BG_FIXED_COMPOSITE_SURVIVAL_HELDOUT_EVAL_VERDICT = SURVIVAL_READY
BG_FIXED_COMPOSITE_SURVIVAL_FRONTIER_VERDICT = CLEAR_OPERATING_POINT
BG_FIXED_COMPOSITE_LAYER_ORIGIN_DOMAIN_VERDICT = UNIFORM_POLICY_SUFFICIENT
BG_FIXED_COMPOSITE_OLD_CODE_PRESERVATION_VERDICT = PRESERVED
BG_FIXED_COMPOSITE_SELECTION_ONLY_READINESS_VERDICT = READY
FIXED_COMPOSITE_BRANCH_SURVIVAL_POLICY_STATUS = SURVIVAL_READY

- selected policy: `selected_policy = fixed_composite_conservative_top4; oracle_retention = 0.931; false_prune_rate = 0.069; avg_survivors = 3.873`
- recommendation: `Proceed to a small selection-only Phase 2 prototype using BGV1 branches, the fixed old+branch+bridge composite, the selected conservative top-k survival operating point, and missing/OOD fallback. Keep veto/rescue as a guardrail, not as a replacement for the selected heldout-ready operating point. selected_policy = fixed_composite_conservative_top4; oracle_retention = 0.931; false_prune_rate = 0.069; avg_survivors = 3.873. Do not claim action steering.`
- learned gated selector remains diagnostic; it is not the primary pruning selector.
- no Ouro weights, tokenizer files, checkpoints, old tap registries, wrapper/local-agent routes, or production routing were modified.

## Selection-only Phase 2 prototype v1 (2026-05-18)

SELECTION_ONLY_PHASE2_PROTOTYPE_STATUS = SURVIVAL_READY_FINAL_ARBITER_WEAK

- cached reproduction: `REPRODUCED`
- live/counterfactual prototype: `SURVIVAL_POSITIVE_FINAL_SELECTION_WEAK`
- final arbiter: `FINAL_SELECTION_WEAK`
- steering readiness: `NEEDS_FINAL_ARBITER_FIRST`
- recommendation: Train or evaluate a stronger final arbiter among top4 survivors before steering.
- no action steering was tested; no production routing changed.

## Final arbiter among top4 survivors v1 (2026-05-18)

FINAL_ARBITER_TOP4_STATUS = FINAL_ARBITER_WEAK_BUT_USEFUL
SELECTION_ONLY_PHASE2A_STATUS_AFTER_FINAL_ARBITER = NEEDS_MORE_FINAL_ARBITER_WORK

- heldout eval: `FINAL_ARBITER_WEAK`
- selected model: `listwise_softmax`
- readiness: `FINAL_ARBITER_WEAK_BUT_IMPROVED`
- recommendation: Run a small improved-arbiter v1.1 or proceed only with explicit weak-baseline caveat.
- no action steering was tested.

## Final arbiter among top4 survivors v1.1 (2026-05-18)

FINAL_ARBITER_TOP4_V1_1_STATUS = NO_IMPROVEMENT
SELECTION_ONLY_PHASE2A_STATUS_AFTER_FINAL_ARBITER_V1_1 = NEEDS_DOMAIN_SPECIALIZATION

- split guard: `FRESH_HELDOUT_READY`
- selected model: `tie_aware_rank_listwise`
- heldout eval: `NO_IMPROVEMENT`
- readiness: `NEEDS_REASONING_ARBITER`
- recommendation: Return to expert/bridge signal quality; v1.1 did not improve final selection.
- no action steering was tested.

## Weight-space merged branch-content taps v1 (2026-05-18)

`MERGED_WEIGHT_BRANCH_CONTENT_TAP_STATUS = FINAL_ARBITER_IMPROVES_ONLY`. The run extracted old/content, hidden-branch, and bridge tap directions, aligned them into shared feature coordinates, built residualized merged candidates, and acquired top4 survivor hidden features from cached raw artifacts for final-arbiter rescoring. No action steering or routing change was tested.

Report: `docs/evaluator/history/bg-run-notes/weight-merge/bg_merged_weight_branch_content_taps_v1.md`.

## Merged tap final arbiter integration v1 (2026-05-18)

`MERGED_TAP_FINAL_ARBITER_INTEGRATION_STATUS = MERGED_TOP1_USEFUL`; `SELECTION_ONLY_PHASE2A_STATUS_AFTER_MERGED_TAP_INTEGRATION = USE_MERGED_TAP_TOP1_AS_ARBITER_BUT_NOT_READY`. The selected validation policy was `old_code_reasoning_top1` with fresh-holdout task macro `0.5610`. This is final-arbiter integration only; no action steering or routing change was tested.

Report: `docs/evaluator/history/bg-run-notes/terminal-arbiters/bg_merged_tap_final_arbiter_integration_v1.md`.

## Merged tap final arbiter integration v1.1 (2026-05-18)

`MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_STATUS = DOMAIN_FALLBACK_USEFUL_BUT_REASONING_LIMITED`; `SELECTION_ONLY_PHASE2A_STATUS_AFTER_MERGED_TAP_INTEGRATION_V1_1 = NEEDS_REASONING_ARBITER`. Grouped task-disjoint CV best readiness-eligible policy was `math_universal_reasoning_universal_else_merged` with task macro `0.7736`; merged tap top1 was `0.7581`. This is final-arbiter integration only; no action steering or routing change was tested.

Report: `docs/evaluator/history/bg-run-notes/terminal-arbiters/bg_merged_tap_final_arbiter_integration_v1_1.md`.

## Old-anchored branch-valid taps v1 (2026-05-30)

`BG_OLD_ANCHORED_BRANCH_VALID_TAP_STATUS = OLD_ANCHORED_BRANCH_TAP_USEFUL`. Tested weight-space transplant and constrained fine-tuned copies of old coding/reasoning and mixed-objective anchors. Selected `weight_space_transplant` candidate `transplant::MIX_CODE_REASONING::full_residual::AntisymLinearNoNorm::102` with heldout old/code drops `0.0000` / `0.0000` and branch/bridge gains `0.1594` / `0.2121` vs its matching old anchor. No old taps, registries, Ouro weights, routing, or steering were modified.

Report: `docs/evaluator/history/bg-run-notes/dualanchor-two-tap/bg_old_anchored_branch_valid_taps_v1.md`.

## Two-tap branch selector v1 (2026-05-30)

`BG_TWO_TAP_BRANCH_SELECTOR_STATUS = TWO_TAP_BRANCH_SELECTOR_READY`. Selected `two_tap_equal` using only the transplanted `MIX_CODE_REASONING` and `MIX_OBJECTIVE_ALL` taps. Heldout oracle retention `0.9825`, false prune `0.0175`, avg survivors `4.0000`. No action steering, routing change, Ouro training, or tap-registry update was performed.

Report: `docs/evaluator/history/bg-run-notes/dualanchor-two-tap/bg_two_tap_branch_selector_v1.md`.

## Two-tap full readiness v1 (2026-05-30)

`BG_TWO_TAP_FULL_READINESS_VERDICT = TWO_TAP_PARTIAL_NOT_READY`. Tested the transplanted `MIX_CODE_REASONING` and `MIX_OBJECTIVE_ALL` taps against old-domain references and branch/universal references across cached old-domain and branch datasets. Domain ok `False`; branch ok `False`. No Ouro training, steering, registry update, routing change, or production change was performed.

Report: `docs/evaluator/history/bg-run-notes/dualanchor-two-tap/bg_two_tap_full_readiness_v1.md`.

## Two-tap gap-targeted v2 (2026-05-30)

`BG_TWO_TAP_GAP_TARGETED_V2_STATUS = TWO_TAP_GAP_TARGETED_NOT_READY`. Selected `old_domain_preserve_repair` on validation only. Clean heldout status `TWO_TAP_PARTIAL_NOT_READY`; full replay status `TWO_TAP_PARTIAL_NOT_READY`. Only copied tap vectors were trained; no old taps, registries, Ouro weights, steering, routing, or production behavior were modified.

Report: `docs/evaluator/history/bg-run-notes/dualanchor-two-tap/bg_two_tap_gap_targeted_v2.md`.

## Layer-native two-tap readiness v1 (2026-05-30)

`BG_LAYER_NATIVE_TWO_TAP_READINESS_VERDICT = LAYER_NATIVE_TWO_TAP_PARTIAL_NOT_READY`. Re-tested `MIX_CODE_REASONING` and `MIX_OBJECTIVE_ALL` as native `24_L4`, `36_L4`, and `47_L4` tap bundles instead of concat-only taps. Domain ok `False`; branch ok `False`. No Ouro training, old registry update, steering, wrapper/local-agent execution, or routing change was performed.

Report: `docs/evaluator/history/bg-run-notes/dualanchor-two-tap/bg_layer_native_two_tap_readiness_v1.md`.

## Layer-native two-tap constrained training v1 (2026-05-30)

`BG_LAYER_NATIVE_TWO_TAP_CONSTRAINED_TRAINING_VERDICT = CONSTRAINED_TWO_TAP_NOT_READY`; readiness verdict `LAYER_NATIVE_TWO_TAP_PARTIAL_NOT_READY`. Trained copied `MIX_CODE_REASONING` and `MIX_OBJECTIVE_ALL` layer-local taps at `24_L4`, `36_L4`, and `47_L4` on train-split old-domain plus branch/bridge labels with anchor preservation. Domain ok `False`; branch ok `False`. No Ouro training, old registry update, steering, wrapper/local-agent execution, or routing change was performed.

Report: `docs/evaluator/history/bg-run-notes/dualanchor-two-tap/bg_layer_native_two_tap_constrained_train_v1.md`.

## Layer-native two-tap targeted rehost diagnostic v1 (2026-05-30)

`BG_LAYER_NATIVE_TWO_TAP_TARGETED_REHOST_DIAGNOSTIC_VERDICT = TARGETED_REHOST_DIAGNOSTIC_PASSES`. Diagnostic-only exact source rehost under the two tap identities produced readiness verdict `LAYER_NATIVE_TWO_TAP_READY`. This is not clean readiness because target references came from prior branch failure reports.

Report: `docs/evaluator/history/bg-run-notes/dualanchor-two-tap/bg_layer_native_two_tap_targeted_rehost_diagnostic_v1.md`.

## Two-tap fresh dataset comparison v1 (2026-05-30)

`BG_TWO_TAP_FRESH_DATA_COMPARISON_VERDICT = TWO_TAP_MATCHES_OR_BEATS_OLD_BG_ON_FRESH`. Built a fresh same-content comparison with seed `20260531`: 192 tasks, 742 candidates, 550 correct-vs-wrong pairs across reasoning, science, math/simple arithmetic, and coding. Existing public sources were reused only where prior artifacts were subset samples, and new datasets were added for each domain. Primary new candidates were restricted to the `MIX_CODE_REASONING` and `MIX_OBJECTIVE_ALL` anchors; old-content transplants from other old taps and targeted-rehost diagnostics were excluded. Best old BG accuracy was `0.8800`; best new two-tap accuracy was `0.8818`.

Report: `docs/evaluator/history/bg-run-notes/dualanchor-two-tap/bg_two_tap_fresh_dataset_comparison_v1.md`.

## Two-tap HH-RLHF comparison v1 (2026-05-30)

`BG_TWO_TAP_HH_RLHF_COMPARISON_VERDICT = TWO_TAP_MATCHES_OR_BEATS_ALL_OLD_BG_ON_HH_RLHF`. Sampled 512 Anthropic HH-RLHF test chosen/rejected pairs with seed `20260601`, skipping the likely prior 200-pair local capture slice. Best all-config old BG accuracy was `0.5977`; best strict two-tap accuracy was `0.6152`. This is a positive same-content comparison, but the absolute HH preference accuracy remains modest.

Report: `docs/evaluator/history/bg-run-notes/dualanchor-two-tap/bg_two_tap_hh_rlhf_comparison_v1.md`.

## DualAnchor branch-gap repair v1 (2026-05-30)

`BG_DUALANCHOR_BRANCH_GAP_REPAIR_VERDICT = DUALANCHOR_BRANCH_NOT_READY`; readiness verdict `LAYER_NATIVE_TWO_TAP_PARTIAL_NOT_READY`. Targeted copied `MIX_CODE_REASONING` and `MIX_OBJECTIVE_ALL` layer-native heads at `24_L4`, `36_L4`, and `47_L4` on train-split branch-gap examples with old-domain preservation. Best-candidate domain ok `False`; best-candidate branch ok `False`; selected-bundle branch ok `False`. No Ouro training, old registry update, steering, wrapper/local-agent execution, or routing change was performed.

Report: `docs/evaluator/history/bg-run-notes/dualanchor-two-tap/bg_dualanchor_branch_gap_repair_v1.md`.

## DualAnchor pre-repair fixed-bundle audit v1 (2026-05-30)

`BG_DUALANCHOR_PRE_REPAIR_FIXED_BUNDLE_AUDIT_VERDICT = DUALANCHOR_FIXED_BUNDLE_NOT_READY`. Selected diagnostic fixed bundle `bundle::two_tap_equal::adaptive_balanced_rescue::AntisymLinear::24_36_47`. Ready fixed bundles `0` / `132`; domain-ok fixed bundles `0` / `132`. No training, failed-repair weights, old registry update, steering, or routing change.

Report: `docs/evaluator/history/bg-run-notes/dualanchor-two-tap/bg_dualanchor_pre_repair_fixed_bundle_audit_v1.md`.

## DualAnchor hard-anchor selector v1 (2026-05-30)

`BG_DUALANCHOR_HARD_ANCHOR_SELECTOR_VERDICT = DUALANCHOR_HARD_ANCHOR_NOT_READY`. Selected fixed pre-repair DualAnchor bundle `bundle::two_tap_equal::sparse_old_plus_bridge_30_70_top0p01::AntisymLinear::24_36_47` with validation old-anchor gate `False`. Full replay domain ok `False`; branch ok `False`. No training, failed-repair weights, old registry update, steering, or routing change.

Report: `docs/evaluator/history/bg-run-notes/dualanchor-two-tap/bg_dualanchor_hard_anchor_selector_v1.md`.

## DualAnchor architecture-looped stratified probe v3 (2026-05-31)

Status: `ARCHITECTURE_LOOPED_SURVIVAL_READY_TERMINAL_DEFER_REQUIRED`.

This run scaled the DualAnchor architecture-shaped loop without steering. Taps were active at layers 24, 36, and 47 across loops L1-L4, with only terminal `L4_47` eligible for confidence-gated collapse. It uses cumulative hook approximation at decoder-layer surfaces; it does not claim autoregressive branch-specific KV/cache fork/carry or compute savings.

Headline metrics:

- tasks: `48`
- stage oracle retention: `0.9848484848484849`
- terminal oracle retained: `1.0`
- terminal forced top1 oracle: `0.9166666666666666`
- terminal reward-diverse rate: `0.22916666666666666`
- positive-oracle rate: `0.3541666666666667`

Locked-baseline candidate:

- selector: DualAnchor `MIX_CODE_REASONING` + `MIX_OBJECTIVE_ALL`
- schedule: `L1_24 -> L1_36 -> L1_47 -> L2_24 -> L2_36 -> L2_47 -> L3_24 -> L3_36 -> L3_47 -> L4_24 -> L4_36 -> terminal L4_47`
- threshold: `mean_floor_very_loose`
- budget: `8`
- L47: active in nonterminal loops
- terminal: confidence-gated top1; otherwise defer/keep terminal survivors

Readiness verdict: `READY_WITH_TERMINAL_DEFER`.
No steering was tested.

## DualAnchor convergence hairs and reasoning/science pre-steering probe v1 (2026-05-31)

`DUALANCHOR_CONVERGENCE_HAIRS_RS_STATUS = SCIENCE_BRANCH_GENERATION_WEAK`.
The L30/L42 convergence-hair replay found useful diagnostic structure but did not clear
hard-merge readiness. Representative L30+L42 merge retained terminal oracle on `0.9583`
of tasks with survivor reduction `0.0247`; conservative/hidden-convergence policies were
safer but too small to be useful as hard merges.

Domain transfer remains uneven. Reasoning needs terminal defer rather than forced top1.
Science is not merely a terminal-selection issue: positive-oracle rate was `0.0833`,
science terminal best reward was `0.0500`, and the domain recipe verdict was
`SCIENCE_NEEDS_DIFFERENT_RECIPE`. Branch classification stayed diagnostic-only, and no
steering or production routing change was made.

Report: `docs/evaluator/science-reasoning-repair.md`.
Artifacts: `artifacts/reports/probes/bg_dualanchor_convergence_hairs_reasoning_science_v1_2026-05-31/`.

## DualAnchor science branch recipe and reasoning terminal defer v1 (2026-05-31)

`DUALANCHOR_SCIENCE_RECIPE_REASONING_DEFER_STATUS = PRE_STEERING_READY_WITH_SCIENCE_DIAGNOSTIC`.
Reasoning is ready as the headline Phase 2b domain under the locked DualAnchor
architecture-looped baseline with terminal survivor-set handoff. Forced top1 retained
reasoning oracle `0.8750`, top2 retained `0.9583`, and top5/full handoff retained `1.0000`.

Science remains diagnostic-only. Parser/reward explains part of the weakness, but no
calibration branch recipe improved over baseline, MMLU science remains weak, L47 is not
enough by itself, and soft hairs indicate science often converges to no-good-branch
states. Science should not be a headline steering-ready domain without a regenerated
recipe improvement or validated parser patch.

Report: `docs/evaluator/science-reasoning-repair.md`.
Artifacts: `artifacts/reports/probes/bg_dualanchor_science_branch_recipe_reasoning_defer_v1_2026-05-31/`.


## Autoregressive KV/cache branch-carry validation v1 (2026-06-01)

`autoregressive_kv_branch_carry_v1` validated that Ouro supports generation-time,
branch-specific KV/cache carry (distinct from already-validated prompt-only layer
carry, which used `use_cache=False`). Using the UniversalTransformerCache
(slot = `current_ut*num_hidden_layers + layer_idx`, 4 loops × 48 layers = 192 slots),
the validation ladder Levels 0–5 all pass within bf16:

- L0 cached decode == full recompute (prefill bit-exact; decode RMS ~0.05–0.2, bf16 drift).
- L1 token-boundary fork: K=2/4/8 independent branch caches, no cross-branch contamination.
- L2 batched branches == independent == full recompute.
- L3 prune/reorder survivors (subset, order changes, 8→4→2 / 8→3 / 4→1) with aligned lineage.
- L4 current-token layer perturbation (L24/36/47, loop-targeted) carries via branch cache.
- L5 prompt-internal perturbation yields a valid branch-specific cache; negative control
  (unperturbed cache vs perturbed recompute, RMS ≈ 3.0) confirms the branch cache is required.
- L6 partial-cache splice: shared/affected slot boundary logic validated (Option A reproduces
  the full branch cache bit-exactly), but **diagnostic only — no compute savings claimed**.

**Status: `AUTOREGRESSIVE_KV_BRANCH_CARRY_STATUS = PROMPT_INTERNAL_BRANCH_CACHE_VALID`;
`LEVEL6_PARTIAL_SPLICE_STATUS = PARTIAL_SPLICE_DIAGNOSTIC_ONLY`.** Equivalence standard:
cached logits match full recompute of the identical sequence within bf16, with exact top-1/token
agreement except at model-intrinsic argmax near-ties. No steering, no training, no compute-savings
claim. Details: `docs/evaluator/kv-cache-branch-carry.md`; artifacts under
`artifacts/reports/probes/bg_autoregressive_kv_branch_carry_v1_2026-06-01/`.


## Partial cache splice v2 (2026-06-01)

`partial_cache_splice_v2` turned the v1 Level 6 diagnostic (copy-affected slots, no
saving) into a real compute-saving suffix-recompute splice. Key obstacle: the
UniversalTransformerCache stores K/V but not the inter-layer residual stream, so the
residual hidden at the perturbation boundary (loop u, layer L output) is captured
during a minimal shared-prefix prefill; an additive boundary perturbation is then
applied without a forward, and ONLY the suffix (loop u layers L+1.., loops u+1..) is
recomputed. The spliced branch cache is **bit-exact** vs a full perturbed-prompt
reference (all 192 slots; prefill logits RMS 0; continuation bit-for-bit).

Empirical hook timing confirmed: perturbing a layer output leaves the boundary slot
unaffected; first affected slot = (u, L+1); changed set == downstream_only theory.
Validated single-branch, multi-branch (K=2/4, independent, no contamination),
batched+prune/reorder, and left-padded (explicit position_ids). Measured compute
savings (baseline K full prefills vs prefix + K suffixes): per-branch layer-pass
saving 13%/38%/63%/88% for boundary loops 0/1/2/3 (layer 24); at loop 2 layer 24,
K-scaling gives 32%/47%/55% fewer passes for K=2/4/8 (wall-clock tracks). Saving is
amortized (needs K≥2; K=1 does prefix+suffix == full). Copy-affected (Mode A) saves
nothing.

**Status: `PARTIAL_CACHE_SPLICE_V2_STATUS = PARTIAL_SPLICE_COMPUTE_SAVING_VALID`**
(upgrades v1's `LEVEL6_PARTIAL_SPLICE_STATUS = PARTIAL_SPLICE_DIAGNOSTIC_ONLY`).
Compute-saving branch-carry CAN now be claimed (amortized, equivalence-validated);
production readiness CANNOT. No steering, no training. GPU guard confirmed the MMLU
science-repair process was not active and its artifacts were untouched. Details:
`docs/evaluator/kv-cache-branch-carry.md`; artifacts under
`artifacts/reports/probes/bg_partial_cache_splice_v2_2026-06-01/`.

## Branch training + logic expansion + terminal v1 (2026-06-06)

- Status: `LOGIC_EXPANSION_READY_TRAINING_NOT_READY`; policy decision: `KEEP_EXTERNAL_DUALANCHOR_CORECONTENT_BASELINE`.
- Goal: move from external branch selection (DualAnchor + CoreContent_v2) toward **model-internal branching**. Primary deliverable = the branch-training **data + evaluation harness** (the L/O training is a bounded 300-step LoRA proof-of-capability on Ouro-RLTT, not a converged model). External verifiers are the only ground truth; DualAnchor/CoreContent are teachers/baselines; science diagnostic-only; steering not run.
- **Logic expanded** to ~33247 train groups across 10 verifier-backed families (synthetic propositional/proof-depth/syllogism/z3-constraint + LogiQA/ReClor/RuleTaker/FOLIO/LSAT/logical-entailment).
- **Branch-pool reachability@4** (Ouro-RLTT generations, external-labeled): {'coding': 0.4333, 'logic': 0.7333, 'math': 0.8333, 'reasoning': 0.95}. Math went 0.31→0.83 once given a brutal tool-free answer-forcing prompt + 1400 math budget + early-stop + LaTeX/sympy verifier; coding ~0.43 after fixing a function-name prompt gap (genuine, not artifact).
- **Experiment 1 (integrated terminal, H)**: within real DualAnchor top-5 survivor sets, CoreContent_v2 0.6584 > DualAnchor forced-top1 0.3787 (retention 1.0) → **selection, not survival, is the terminal bottleneck**; the composed external baseline (DualAnchor survival → CoreContent_v2 ranking → survivor handoff) is valid.
- **DualAnchor-as-teacher (J)**: useful branch-policy teacher (oracle-retention lift over random +0.11 coding/reasoning) but ~random on logic (+0.03) → logic branch quality must come from verifier reward, not teacher distillation.
- **Trained vs Ouro-RLTT-no-adapter (O)**: macro positive_oracle@K {'base': 0.708, 'sft': 0.688}, lift -0.02, diversity lift 0.45 (`MODEL_INTERNAL_BRANCHING_PARTIAL`).
- Data-hygiene fixes caught by hand-audit: math-LaTeX verifier false-negatives, coding function-name mismatch, and a hendrycks_math `task_uid` collision (row indices repeat across 7 subjects → re-keyed by prompt-hash).
- No Ouro base/tokenizer/checkpoint/registry overwrite; adapters saved only under `artifacts/models/branch_training_logic_expansion_v1/`; pure/transplanted/CoreContent_v2 artifacts untouched.
- Artifacts: `artifacts/reports/probes/branch_training_logic_expansion_terminal_v1_2026-06-06`; data `data/branch_training_logic_expansion_v1/`.
