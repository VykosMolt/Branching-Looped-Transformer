# Corrected Math BG-Gate Pilot

**Date:** 2026-05-15  
**Status:** Successful protocol/proof-of-possibility pilot. Not a Phase-2 gate decision.  
**Scope:** Math generated-branch tournaments from Ouro-RLTT, trained single-state 24/36 tap heads, and late 47 fused baseline.

## Executive Summary

The corrected math pilot supports the main BG readout hypothesis:

> Trained layer-24 and layer-36 single-state taps can read useful relational math-branch signal from Ouro-RLTT-generated branches, and they are not obviously worse than the late layer-47 fused baseline.

This is not evidence that math branch selection is solved. The held-out eval split has only 8 tournaments, and the kept tournament set is still GSM8K-heavy after exact-answer filtering. The result is directional: the trained 24/36 readouts are viable enough to justify a larger, source-stratified, budget-aware gate run.

## Artifacts

Geometry:
- `artifacts/reports/probes/math_layer_geometry_rltt.md`
- `artifacts/reports/probes/math_layer_geometry_rltt.json`
- `artifacts/reports/probes/math_layer_geometry_rltt_features.pt`

Generated tournaments and trained heads:
- `artifacts/reports/probes/math_branch_tournaments_rltt.json`
- `artifacts/reports/probes/math_branch_tap_features.pt`
- `artifacts/reports/probes/math_single_state_tap_heads.pt`
- `artifacts/reports/probes/train_single_state_math_taps_24_36.md`
- `artifacts/reports/probes/train_single_state_math_taps_24_36.json`
- `artifacts/reports/probes/evaluate_math_tournament_taps.md`
- `artifacts/reports/probes/evaluate_math_tournament_taps.json`

Probe scripts:
- `utilities/tests/manual/math_layer_geometry_rltt.py`
- `utilities/tests/manual/generate_math_branch_tournaments_rltt.py`
- `utilities/tests/manual/train_single_state_math_taps_24_36.py`
- `utilities/tests/manual/evaluate_math_tournament_taps.py`
- `utilities/tests/manual/math_bg_probe_lib.py`

## Corrections Made Before Final Pilot

The first mixed run was misleading because the mixed loader filled the candidate set from GSM8K before truncation, so the kept artifact became GSM8K-only. The loader now constructs a balanced candidate pool first:

- 50 GSM8K candidate prompts
- 50 MATH candidate prompts

Generation also hit CUDA OOM on longer MATH prompts when sampling all return sequences at once. `generate_math_branch_tournaments_rltt.py` now has `--generation-batch-size`; the corrected pilot used sub-batches of 2 return sequences.

These changes are generation/data-pipeline fixes, not model/tap changes.

## Math Layer Geometry

No evaluator checkpoint was loaded. Geometry was measured on paired math texts: gold/reference solution text and a deterministic wrong-answer control.

| Layer | L1-L4 cos | L2-L4 cos | mean off-diag | min off-diag | Verdict |
|---:|---:|---:|---:|---:|---|
| 24 | +0.9133 | +0.9719 | +0.9535 | +0.9133 | fully converged |
| 36 | +0.9315 | +0.9718 | +0.9644 | +0.9315 | fully converged |
| 47 | +0.6562 | +0.8947 | +0.8504 | +0.6562 | intermediate / unclear |

This differs from the HH blocker result. On HH text, 24/36 were converged and 47 was bipartite. On math text, 24/36 remain converged, but 47 is intermediate rather than cleanly bipartite.

Interpretation: the heterogeneous tap interface is still supported for 24/36, but the reason to privilege 47 on math is weaker than on HH. For math, late fused 47 is a useful baseline, not a privileged default winner.

## Corrected Tournament Generation

Command class:

```bash
venv/bin/python -u utilities/tests/manual/generate_math_branch_tournaments_rltt.py \
  --source mixed \
  --max-problems 100 \
  --attempts-per-problem 4 \
  --max-prompt-length 384 \
  --max-new-tokens 160 \
  --generation-batch-size 2 \
  --temperature 0.7 \
  --top-p 0.95 \
  --report-every 5
```

Generation summary:

| Item | Value |
|---|---:|
| Candidate prompts seen | 100 |
| Candidate source mix | 50 GSM8K / 50 MATH |
| Attempts generated | 400 |
| Correct attempts | 91 |
| Incorrect attempts | 309 |
| Unparseable attempts | 0 |
| Tournaments kept | 33 |
| Kept prompt rate | 0.330 |
| Kept source mix | 26 GSM8K / 7 MATH |

The exact-answer verifier worked cleanly on this run: no unparseable attempts. The kept set is still source-skewed because MATH prompts more often produced all-wrong or all-correct sets under this budget, so they failed the "at least one correct and one incorrect branch" tournament criterion.

This skew is not a tap failure. It is a generation-yield and budget issue.

## Trained Tap Pilot

Feature capture:

| Item | Value |
|---|---:|
| Tournaments | 33 |
| Candidate branches | 132 |
| Tap layers | 24, 36, 47 |
| Train split | 25 tournaments |
| Eval split | 8 tournaments |

Configs:

- `24_L1`
- `24_L4`
- `24_mean`
- `36_L1`
- `36_L4`
- `36_mean`
- `47_concat_L1_L4`

Held-out eval:

| Config | top1 | pairwise | condorcet | cycle | n |
|---|---:|---:|---:|---:|---:|
| `24_L1` | 1.000 | 1.000 | 1.000 | 0.000 | 8 |
| `24_L4` | 1.000 | 1.000 | 1.000 | 0.000 | 8 |
| `24_mean` | 1.000 | 1.000 | 0.875 | 0.000 | 8 |
| `36_L4` | 1.000 | 1.000 | 1.000 | 0.000 | 8 |
| `36_mean` | 1.000 | 1.000 | 1.000 | 0.000 | 8 |
| `ensemble_mean_logits` | 1.000 | 1.000 | 0.875 | 0.000 | 8 |
| `36_L1` | 1.000 | 0.962 | 1.000 | 0.000 | 8 |
| `47_concat_L1_L4` | 1.000 | 0.962 | 0.875 | 0.000 | 8 |

The important result is not the 1.000 point estimate. With n=8, that number is not stable enough to decide anything. The important result is that all trained 24/36 readouts work at least as well as the late 47 fused baseline on this pilot, and no pairwise cycles appear.

## Interpretation

What the pilot supports:

1. The math geometry check does not refute the heterogeneous tap interface: 24/36 are still converged single-state readouts.
2. Newly trained 24/36 single-state heads can select among Ouro-RLTT-generated math branches.
3. The late 47 fused tap is a good baseline, but this pilot gives no reason to privilege it over trained 24/36 taps for math.
4. The protocol is viable: generate branches, label with exact-answer verifier, keep mixed correct/incorrect tournaments, capture taps, train antisymmetric pairwise heads, evaluate top-1/pairwise/Condorcet/cycle.

What the pilot does not prove:

1. It does not prove generalization; eval n=8 is too small.
2. It does not prove MATH-domain performance; only 7 kept tournaments are from MATH.
3. It does not decide the Phase-2 gate.
4. It does not settle whether MATH failures are model-signal failures, generation-budget failures, or verifier/yield failures.

## Generation Budget / Verbosity Constraint

Ouro-RLTT generation verbosity is now part of the measurement problem. Under a fixed token budget, the branch dataset is selected for prompts where:

- the model finishes within budget,
- the final answer is parseable,
- at least one correct and one incorrect branch are both produced.

That can bias the kept tournament set toward easier or cleaner math cases. MATH-heavy scaling failures must not be interpreted as tap failures until generation budget and yield are measured.

The next generator should log, per source and per budget:

- prompts seen,
- attempts generated,
- kept tournament rate,
- correct attempt rate,
- incorrect attempt rate,
- parse failure rate,
- generated token mean/median/p95,
- truncation rate,
- correct rate for truncated vs non-truncated attempts,
- tournament formation rate among non-truncated prompts,
- kept source mix.

Run budget strata before interpreting MATH performance:

| Stratum | Purpose |
|---|---|
| `max_new_tokens=256` | deployment-like budget pressure |
| `max_new_tokens=512` | moderate diagnostic budget |
| `max_new_tokens=1024` | diagnostic ceiling / yield check |

The budgeted result is especially important because the BG controller is meant to help under compute constraints. The unbudgeted or high-budget result diagnoses whether low MATH yield is just truncation/verbosity.

## Next Gate

The next real gate should be source-stratified and budget-aware, not just "bigger mixed."

Suggested target:

| Source | Generate | Keep target |
|---|---:|---:|
| GSM8K | 300-500 prompts | >= 100 tournaments |
| MATH | 300-500 prompts | >= 100 tournaments if feasible |

Report GSM8K, MATH, and combined metrics separately:

- generation/yield metrics,
- top-1 tournament accuracy,
- pairwise accuracy,
- Condorcet winner rate,
- cycle rate,
- bootstrap confidence intervals when n is large enough.

If MATH kept rate remains low, tune generation before drawing tap conclusions:

- mix lower-temperature and higher-temperature attempts,
- force a short `Final answer:` line,
- use stricter stop sequences,
- sweep `max_new_tokens`,
- keep labels external/objective.

## Bottom Line

This is a successful pilot. For math branch selection, the BG controller may not need late bipartite fusion. Early/mid converged single-state taps can work, and may be cheaper and more useful under budget.

The next decision needs a larger, source-stratified, budget-aware run.

## Expanded clean GSM8K transfer + GRU control (2026-05-16)

- EXPANDED_CLEAN_GSM8K_VERDICT: `CLEAN_MINIMUM`
- EXPANDED_LINEAR_TRANSFER_VERDICT: `GOOD`
- GRU_CONTROL_VERDICT: `GRU_WEAK`
- clean tournaments: `28`
- random_top1_baseline: `0.563`
- best AntisymLinear config/head: `{'config': '36_L4', 'architecture': 'AntisymLinear', 'top1': 0.7857142857142857, 'pairwise': 0.7037037037037037, 'cycle': 0.0, 'margin_mean': 1.0424813000219209, 'margin_std': 0.8679666340925131}`
- best NoNorm config/head: `{'config': '24_L4', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.75, 'pairwise': 0.7407407407407407, 'cycle': 0.0, 'margin_mean': 0.012588573902446245, 'margin_std': 0.011785265045384234}`
- best GRU config: `{'config': 'gru24_sequence', 'top1': 0.7142857142857143, 'pairwise': 0.6666666666666666, 'cycle': 0.0, 'bias_to_signal': 0.010894300608310918, 'hh_heldout_acc': 0.5}`
- winner family: `AntisymLinear`
- winner layer: `36`
- full report: `artifacts/reports/probes/clean_gsm8k_expanded_transfer_gru_2026-05-16_summary.md`
- interpretation: The GRU control underperformed the exact-antisymmetric linear/NoNorm controls; centered raw bias was low, but HH holdout accuracy stayed below the control threshold.

## Code branch pilot (2026-05-16)

- CODE_INTERFACE_VERDICT: `READY`
- CODE_TASKSET_VERDICT: `READY`
- CODE_GENERATION_VERDICT: `READY`
- CODE_TOURNAMENT_VERDICT: `TOO_FEW_TOURNAMENTS`
- CODE_TRANSFER_VERDICT: `NOT_RUN`
- tasks: `10`
- candidates: `40`
- strict_clean tournaments: `1`
- diagnostic_mixed tournaments: `4`
- random_top1_baseline: `nan`
- best AntisymLinear row: `NOT_RUN`
- best NoNorm row: `NOT_RUN`
- winner: `NOT_RUN`
- full report: `artifacts/reports/probes/code_branch_pilot_2026-05-16_summary.md`
- interpretation: The local-agent wrapper generated mostly correct code on the selected local/MBPP task mix, leaving too few objective mixed unit-test tournaments for a relational tap-transfer evaluation.

## Code branch pilot v2 (2026-05-16)

- CODE_V2_INTERFACE_VERDICT: `READY`
- CODE_V2_TASKSET_VERDICT: `READY`
- CODE_V2_GENERATION_VERDICT: `READY`
- CODE_V2_TOURNAMENT_VERDICT: `DIAGNOSTIC_ONLY`
- CODE_V2_TRANSFER_VERDICT: `GOOD`
- tasks: `40`
- candidates: `188`
- duplicate rate: `0.3088235294117647`
- correct / near_miss / nonsense: `{'correct': 70, 'near_miss': 13, 'nonsense': 105}`
- strict_clean tournaments: `3`
- diagnostic_mixed tournaments: `22`
- random_top1_baseline: `0.5522727343169126`
- best AntisymLinear row: `{'config': '36_L4', 'architecture': 'AntisymLinear', 'top1': 0.6818181818181818, 'pairwise': 0.7090909090909091, 'cycle': 0.0, 'margin_mean': 1.7220367030663923, 'margin_std': 1.6406907362264995}`
- best NoNorm row: `{'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.7272727272727273, 'pairwise': 0.6909090909090909, 'cycle': 0.0, 'margin_mean': 0.27857657115567813, 'margin_std': 0.35337859692663737}`
- winner: `{'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.7272727272727273, 'pairwise': 0.6909090909090909, 'cycle': 0.0, 'margin_mean': 0.27857657115567813, 'margin_std': 0.35337859692663737}`
- candidate-stage harvesting fixed v1 too-successful-wrapper problem: `yes`
- full report: `artifacts/reports/probes/code_branch_pilot_v2_2026-05-16_summary.md`
- interpretation: The v2 code pilot used objective unit-test labels and candidate-stage harvesting; transfer remains a preliminary code-branch signal.

## Code branch pilot v2 correction + harness hardening (2026-05-16)

- FIX_VERDICT: `PATCHED_AND_UNIT_TESTED`
- full fix report: `artifacts/reports/probes/code_branch_v2_harness_agent_fixes_2026-05-16.md`
- nonsense inspection: `artifacts/reports/probes/code_branch_v2_nonsense_inspection_2026-05-16.md`
- status of previous v2 transfer: `HISTORICAL_PRE_FIX`
- reason: the old `nonsense` bucket was contaminated by runnable zero-pass wrong code, wrapper/prose extraction failures, syntax/runtime failures, and safety rejections.
- original 105 old-nonsense breakdown: `50 runnable_zero_pass_wrong`, `25 parseable_runtime_error`, `20 prose_or_wrapper_not_code`, `6 syntax_invalid_code`, `3 safety_rejected`, `1 parseable_no_function`.
- local-agent fixes: final code extraction rejects wrapper/prose status payloads; `sanitize_tool_input("python", ...)` returns empty for obvious non-code payloads.
- taskset fixes: MBPP function-name extraction skips outer builtins, so `mbpp/232` targets `larg_nnum`; MBPP prompts include exact inferred signature shape; `mbpp/237` states tuple canonicalization explicitly.
- evaluator fixes: binary correct/incorrect labels are preserved for branch selection, while diagnostic subtype is split into `near_miss`, `wrong_code`, `runtime_error`, and `malformed`; legacy `nonsense` is retained only as compatibility metadata.
- safety fix: `sys.setrecursionlimit` is allowed for local DSA while unsafe `sys` usage remains rejected.
- checkpointing fix: generation writes `code_branch_candidates_v2_2026-05-16.partial.json` after each task; evaluation writes `code_branch_tournaments_v2_2026-05-16.partial.json` after each task with provisional evaluations, tournaments, summary, primary eval set, and verdict.
- validation: patched files pass `py_compile`; local-agent wrapper tests pass (`139 passed`); no-model relabel of existing candidates completed.
- no-model relabel of old candidates under patched evaluator: `38` tasks evaluated, `182` candidates, labels `{'correct': 68, 'near_miss': 12, 'wrong_code': 71, 'runtime_error': 0, 'malformed': 31}`, strict-clean tournaments `3`, diagnostic-mixed tournaments `21`, random baseline `0.5468253968253969`.
- caveat on relabel: two old candidate tasks no longer match the regenerated patched taskset (`mbpp/251`, `mbpp/255` old; `mbpp/111`, `mbpp/230` new), so this relabel is a sanity check only, not a replacement transfer result.
- next valid code transfer run: regenerate candidates with the patched local-agent/taskset path, then recapture features and rerun HH-trained AntisymLinear / NoNorm transfer. Do not reuse the old v2 feature tensor as a current result.

## Patched code branch pilot v2-mini (2026-05-16)

- CODE_V2_PATCH_STATUS: `READY`
- CODE_V2_MINI_TASKSET_VERDICT: `READY`
- CODE_V2_MINI_GENERATION_VERDICT: `READY`
- CODE_V2_MINI_TOURNAMENT_VERDICT: `RUNNABLE_DIAGNOSTIC`
- CODE_V2_MINI_TRANSFER_VERDICT: `GOOD`
- tasks: `30`
- unique candidates: `112`
- label counts: `{'correct': 49, 'near_miss': 11, 'wrong_code': 50, 'runtime_error': 0, 'malformed': 2, 'safety_rejected': 0}`
- strict_clean / diagnostic_runnable / diagnostic_mixed: `2 / 8 / 8`
- primary_eval_set: `diagnostic_runnable`
- random_top1_baseline: `0.5625`
- best AntisymLinear row: `{'config': '36_mean', 'architecture': 'AntisymLinear', 'top1': 0.875, 'pairwise': 0.78125, 'cycle': 0.0, 'margin_mean': 0.9804582111537457, 'margin_std': 0.7993556956798318}`
- best NoNorm row: `{'config': '47_L4', 'architecture': 'AntisymLinearNoNorm', 'top1': 1.0, 'pairwise': 0.875, 'cycle': 0.0, 'margin_mean': 0.7644035294651985, 'margin_std': 1.0323769370663691}`
- winner top1 / pairwise / cycle: `{'top1': 1.0, 'pairwise': 0.875, 'cycle': 0.0}`
- transfer signal survived patched path: `True`
- wrapper-prose artifacts eliminated: `False`
- mbpp/232 fixed: `True`
- mbpp/306 fixed: `True`
- future-risk-register path: `artifacts/reports/probes/code_branch_future_risks_2026-05-16.md`
- full report path: `artifacts/reports/probes/code_branch_pilot_v2_mini_patched_2026-05-16_summary.md`
- interpretation: The patched harness produced usable code-branch tournaments and HH-trained linear transfer remained above the random branch-selection baseline.

## Patched code branch pilot v2-mini clarification (2026-05-16)

- wrapper-prose artifacts eliminated from admitted candidates: `True`
- raw wrapper/status generations rejected before candidate admission: `6`
- admitted wrapper/status artifact count: `0`

## Code near-miss balancing pass (2026-05-17)

- BALANCE_INSPECTION_VERDICT: `READY`
- BALANCING_GENERATION_VERDICT: `COMPLETED`
- BALANCED_TOURNAMENT_VERDICT: `RED`
- BALANCED_FEATURE_VERDICT: `NOT_RUN`
- BALANCED_TRANSFER_VERDICT: `NOT_RUN`
- tasks: `10`
- old strict_clean / new strict_clean: `2 / 2`
- old label counts / new label counts: `{'correct': 14, 'near_miss': 12, 'wrong_code': 11, 'runtime_error': 0, 'malformed': 0, 'safety_rejected': 0} / {'correct': 18, 'near_miss': 16, 'wrong_code': 13, 'runtime_error': 0, 'malformed': 0, 'safety_rejected': 0}`
- tasks converted to strict_clean: `[]`
- modes that helped: `[]`
- best AntisymLinear row: `NA`
- best NoNorm row: `NA`
- full report path: `artifacts/reports/probes/code_branch_near_miss_balancing_2026-05-17_summary.md`
- interpretation: The bottleneck remains task/test design or missing-side generation reliability.

## Current transfer state and scalar-vs-relational audit (2026-05-17)

- CURRENT_ARTIFACT_INVENTORY_VERDICT: `READY`
- GENERALIZATION_VERDICT: `SUPPORTED_FOR_LOCAL_PLANNING`
- POINTWISE_RANKING_VERDICT: `SUPPORTED_IN_OBJECTIVE_DOMAINS`
- clean GSM8K: expanded run was `CLEAN_MINIMUM` with `GOOD` HH-trained linear transfer; GRU control was `GRU_WEAK`.
- patched code v2-mini: patched pipeline reached `RUNNABLE_DIAGNOSTIC` and `GOOD` transfer, with NoNorm strongest on the diagnostic set.
- near-miss enrichment/balancing: correct and near_miss candidates exist globally, but strict_clean stayed low at 2 after balancing.
- current bottleneck: `WITHIN_TASK_PAIRING`
- full reports: `artifacts/reports/probes/current_bg_transfer_state_2026-05-17_summary.md`, `docs/evaluator/post_v10_synthesis_2026-05-17_v7_actual_state_and_next.md`, `artifacts/reports/probes/scalar_vs_relational_current_state_2026-05-17.md`, `artifacts/reports/probes/strict_clean_task_screening_protocol_2026-05-17.md`.
- interpretation: transfer is supported enough for local planning; the next bottleneck is strict-clean code task screening, not another broad transfer existence probe.

## Strict-clean code task screening (2026-05-17)

- SCREENING_TASKPOOL_VERDICT: `READY`
- SCREENING_GENERATION_VERDICT: `COMPLETED`
- STRICT_CLEAN_SCREENING_VERDICT: `YELLOW`
- tasks screened: `60`
- strict_clean_ready: `6`
- anchor_only / near_miss_only / all_correct / all_wrong: `5 / 7 / 36 / 6`
- label totals: `{'correct': 73, 'wrong_code': 24, 'near_miss': 20}`
- per-source summary: `{'mbpp': {'all_correct': 36, 'all_wrong': 6, 'strict_clean_ready': 6, 'anchor_only': 5, 'near_miss_only': 7}}`
- per-difficulty summary: `{'hard': {'all_correct': 2, 'all_wrong': 1}, 'medium': {'strict_clean_ready': 6, 'all_correct': 34, 'anchor_only': 5, 'all_wrong': 5, 'near_miss_only': 7}}`
- within-task pairing bottleneck: `CONFIRMED`
- full report: `artifacts/reports/probes/code_strict_clean_screening_2026-05-17_summary.md`
- interpretation: Screening still shows the main constraint is finding same-task correct-vs-near-miss pairs cheaply.

## Strict-clean code transfer micro-eval (2026-05-17)

- STRICT_CLEAN_TRANSFER_SET_VERDICT: `READY`
- STRICT_CLEAN_FEATURE_VERDICT: `READY`
- STRICT_CLEAN_TRANSFER_VERDICT: `WEAK`
- tasks: `6`
- candidates: `13` primary / `15` secondary
- random_top1_baseline: `0.527777781089147`
- best AntisymLinear row: `{'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 0.5, 'pairwise': 0.42857142857142855, 'cycle': 0.0, 'margin_mean': 0.634135976433754, 'margin_std': 0.5422409172287559}`
- best NoNorm row: `{'config': '47_concat_all_loops', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5, 'pairwise': 0.5714285714285714, 'cycle': 0.0, 'margin_mean': 0.7558111765732368, 'margin_std': 0.9909718739852695}`
- winner top1 / pairwise / cycle: `{'top1': 0.5, 'pairwise': 0.5714285714285714, 'cycle': 0.0}`
- transfer survived on strict-clean correct-vs-near_miss candidates: `weak`
- full report: `artifacts/reports/probes/code_strict_clean_transfer_2026-05-17_summary.md`
- interpretation: HH-trained tiny taps show a weak strict-clean code transfer signal on this small correct-vs-near_miss micro-set.

## Code-specific tiny-head control + strict-clean screening expansion (2026-05-17)

- CODE_SPECIFIC_SPLIT_VERDICT: `MISSING_FEATURES`
- CODE_SPECIFIC_FEATURE_VERDICT: `RECAPTURED`
- CODE_SPECIFIC_TINY_HEAD_VERDICT: `GOOD`
- STRICT_CLEAN_SCREENING_EXPANSION_VERDICT: `GREEN`
- training task count / pair count: `14` / `138`
- held-out strict-clean task count: `6`
- best code-trained AntisymLinear row: `{config: 24_L4, architecture: AntisymLinear, top1: 0.8333333333333334, pairwise: 0.7142857142857143, cycle: 0.0, margin_mean: 3.946825544039408, margin_std: 1.9451767754980098}`
- best code-trained NoNorm row: `{config: 24_L4, architecture: AntisymLinearNoNorm, top1: 0.8333333333333334, pairwise: 0.7142857142857143, cycle: 0.0, margin_mean: 0.4422141411341727, margin_std: 0.3071355319441245}`
- comparison to HH-trained strict-clean WEAK result: `47_concat_all_loops / AntisymLinearNoNorm`, top1=0.500, pairwise=0.571, cycle=0.000.
- screening expansion counts: tasks_screened=`137`, strict_clean_ready=`10`, label_totals=`{correct: 164, near_miss: 60, wrong_code: 64, safety_rejected: 4, malformed: 3}`
- screening expansion source mix: `{humaneval: {strict_clean_ready: 7}, mbpp: {strict_clean_ready: 3}}` strict-clean-ready contribution; HumanEval was used with granularized checks where possible after the MBPP-only tail stayed low-yield.
- new strict_clean_ready task IDs: `[mbpp/11, mbpp/20, mbpp/434, HumanEval/10, HumanEval/118, HumanEval/123, HumanEval/125, HumanEval/141, HumanEval/148, HumanEval/69]`
- interpretation: states contain a strict-clean code branch signal; HH-trained projection did not transfer strongly enough to near-miss code, while code-specific tiny taps can read it.

## Expanded strict-clean code projection comparison (2026-05-17)

- EXPANDED_STRICT_CLEAN_SET_VERDICT: `READY`
- EXPANDED_STRICT_CLEAN_FEATURE_VERDICT: `RECAPTURED`
- CODE_SPECIFIC_EXPANDED_TRAIN_VERDICT: `RETRAINED`
- EXPANDED_HH_TRANSFER_VERDICT: `GOOD`
- EXPANDED_CODE_SPECIFIC_TRANSFER_VERDICT: `GOOD`
- EXPANDED_STRICT_CLEAN_COMPARISON_VERDICT: `CODE_SPECIFIC_ADVANTAGE`
- eval tasks: `16`
- OLD6 best HH / CODE: `{'head_family': 'HH', 'config': '47_concat_all_loops', 'architecture': 'AntisymLinearNoNorm', 'family_architecture': 'HH_NoNorm', 'top1': 0.5, 'over_random': -0.02777778108914697, 'pairwise': 0.5714285714285714, 'cycle': 0.0, 'margin_mean': 0.7558111765732368, 'margin_std': 0.9909718739852695}` / `{'head_family': 'CODE', 'config': '36_L4', 'architecture': 'AntisymLinear', 'family_architecture': 'CODE_AntisymLinear', 'top1': 1.0, 'over_random': 0.47222221891085303, 'pairwise': 0.8571428571428571, 'cycle': 0.0, 'margin_mean': 1.4508539686600368, 'margin_std': 1.1298246732529835}`
- NEW10 best HH / CODE: `{'head_family': 'HH', 'config': '47_L4', 'architecture': 'AntisymLinear', 'family_architecture': 'HH_AntisymLinear', 'top1': 0.9, 'over_random': 0.3176190376281739, 'pairwise': 0.7391304347826086, 'cycle': 0.0, 'margin_mean': 1.305394220352173, 'margin_std': 1.134789404785102}` / `{'head_family': 'CODE', 'config': '47_L4', 'architecture': 'AntisymLinear', 'family_architecture': 'CODE_AntisymLinear', 'top1': 0.9, 'over_random': 0.3176190376281739, 'pairwise': 0.8695652173913043, 'cycle': 0.0, 'margin_mean': 3.521452635526657, 'margin_std': 3.307454072283649}`
- ALL16 best HH / CODE: `{'head_family': 'HH', 'config': '47_mean', 'architecture': 'AntisymLinear', 'family_architecture': 'HH_AntisymLinear', 'top1': 0.75, 'over_random': 0.18809523060917854, 'pairwise': 0.6, 'cycle': 0.0, 'margin_mean': 0.9949273709207773, 'margin_std': 0.9282425444305212}` / `{'head_family': 'CODE', 'config': '36_L4', 'architecture': 'AntisymLinear', 'family_architecture': 'CODE_AntisymLinear', 'top1': 0.875, 'over_random': 0.31309523060917854, 'pairwise': 0.8333333333333334, 'cycle': 0.0, 'margin_mean': 1.6726789940148592, 'margin_std': 1.4576284253711889}`
- best HH-trained row: `{'head_family': 'HH', 'config': '47_mean', 'architecture': 'AntisymLinear', 'family_architecture': 'HH_AntisymLinear', 'top1': 0.75, 'over_random': 0.18809523060917854, 'pairwise': 0.6, 'cycle': 0.0, 'margin_mean': 0.9949273709207773, 'margin_std': 0.9282425444305212}`
- best code-trained row: `{'head_family': 'CODE', 'config': '36_L4', 'architecture': 'AntisymLinear', 'family_architecture': 'CODE_AntisymLinear', 'top1': 0.875, 'over_random': 0.31309523060917854, 'pairwise': 0.8333333333333334, 'cycle': 0.0, 'margin_mean': 1.6726789940148592, 'margin_std': 1.4576284253711889}`
- code-specific advantage survived: `yes`
- winning architecture: `AntisymLinear`
- winning layer: `36`
- interpretation: Code-specific projection training remains better than HH-trained projection on the expanded strict-clean code branch-selection set.
- full report: `artifacts/reports/probes/expanded_strict_clean_code_projection_comparison_2026-05-17_summary.md`

## Cross-domain fixed-config + reasoning pilot (2026-05-17)

- BG_BACKLOG_AUDIT_VERDICT: `READY`
- BG_HEAD_REGISTRY_VERDICT: `RETRAINED`
- BG_CROSS_DOMAIN_MATRIX_VERDICT: `READY`
- FIXED_CONFIG_AUDIT_VERDICT: `READY`
- GENERALIST_SPECIALIST_VERDICT: `DOMAIN_SPECIALISTS_NEEDED`
- REASONING_BRANCH_DATA_VERDICT: `READY`
- REASONING_TRANSFER_VERDICT: `GOOD`
- LOOP_LAYER_DIAGNOSTIC_VERDICT: `READY`
- best fixed configs / stability: `[{'head_key': 'CODE_AntisymLinearNoNorm::36_mean', 'domains_within_0p05_of_best': 3, 'domains_seen': 6, 'avg_pairwise': nan, 'avg_top1_over_random': 0.17190475741490013}, {'head_key': 'CODE_AntisymLinear::24_L4', 'domains_within_0p05_of_best': 2, 'domains_seen': 6, 'avg_pairwise': nan, 'avg_top1_over_random': 0.16380951753152267}, {'head_key': 'CODE_AntisymLinear::36_L4', 'domains_within_0p05_of_best': 2, 'domains_seen': 6, 'avg_pairwise': nan, 'avg_top1_over_random': 0.1667857090011239}, {'head_key': 'CODE_AntisymLinearNoNorm::36_L4', 'domains_within_0p05_of_best': 2, 'domains_seen': 6, 'avg_pairwise': nan, 'avg_top1_over_random': 0.1526190409436822}, {'head_key': 'CODE_AntisymLinear::47_L4', 'domains_within_0p05_of_best': 2, 'domains_seen': 6, 'avg_pairwise': nan, 'avg_top1_over_random': 0.18249999491409177}]`
- HH vs code-trained comparison: `{'CODE_SPECIFIC_ADVANTAGE_ON_STRICT_CLEAN': True, 'HH_GENERAL_ADVANTAGE_ON_HH': True, 'SHARED_COHERENCE_AXIS': 'weak', 'best_code_all16': {'domain': 'CODE_STRICT_CLEAN_ALL16', 'family': 'CODE', 'config': '36_L4', 'architecture': 'AntisymLinear', 'top1': 0.875, 'over_random': 0.31309523060917854, 'pairwise': 0.8333333333333334, 'cycle': 0.0, 'margin_mean': 1.6726789940148592, 'margin_std': 1.4576284253711889}, 'best_hh_all16': {'domain': 'CODE_STRICT_CLEAN_ALL16', 'family': 'HH', 'config': '47_mean', 'architecture': 'AntisymLinear', 'top1': 0.75, 'over_random': 0.18809523060917854, 'pairwise': 0.6, 'cycle': 0.0, 'margin_mean': 0.9949273709207773, 'margin_std': 0.9282425444305212}, 'best_code_hh': {'domain': 'HH_200', 'family': 'CODE', 'config': '47_L4', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5350000262260437, 'over_random': 0.0350000262260437, 'pairwise': 0.5350000262260437, 'cycle': nan, 'margin_mean': 0.17239375412464142, 'margin_std': 1.2840533256530762}, 'best_hh_hh': {'domain': 'HH_200', 'family': 'HH', 'config': '47_concat_L1_L4', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.8550000190734863, 'over_random': 0.35500001907348633, 'pairwise': 0.8550000190734863, 'cycle': nan, 'margin_mean': 1.262777328491211, 'margin_std': 1.140013337135315}}`
- NoNorm vs AntisymLinear comparison: `{'best_per_domain': {'CLEAN_GSM8K_EXPANDED': {'domain': 'CLEAN_GSM8K_EXPANDED', 'family': 'CODE', 'config': '24_mean', 'architecture': 'AntisymLinear', 'top1': 0.8928571428571429, 'over_random': 0.3303571354065623, 'pairwise': 0.7962962962962963, 'cycle': 0.0, 'margin_mean': 1.7689685566084725, 'margin_std': 1.411621785660656}, 'CODE_RUNNABLE_DIAGNOSTIC': {'domain': 'CODE_RUNNABLE_DIAGNOSTIC', 'family': 'CODE', 'config': '24_L1', 'architecture': 'AntisymLinear', 'top1': 1.0, 'over_random': 0.43749999441206455, 'pairwise': 1.0, 'cycle': 0.0, 'margin_mean': 7.189825654029846, 'margin_std': 6.033544621780366}, 'CODE_STRICT_CLEAN_ALL16': {'domain': 'CODE_STRICT_CLEAN_ALL16', 'family': 'CODE', 'config': '36_L4', 'architecture': 'AntisymLinear', 'top1': 0.875, 'over_random': 0.31309523060917854, 'pairwise': 0.8333333333333334, 'cycle': 0.0, 'margin_mean': 1.6726789940148592, 'margin_std': 1.4576284253711889}, 'CODE_STRICT_CLEAN_NEW10': {'domain': 'CODE_STRICT_CLEAN_NEW10', 'family': 'CODE', 'config': '47_L4', 'architecture': 'AntisymLinear', 'top1': 0.9, 'over_random': 0.3176190376281739, 'pairwise': 0.8695652173913043, 'cycle': 0.0, 'margin_mean': 3.521452635526657, 'margin_std': 3.307454072283649}, 'CODE_STRICT_CLEAN_OLD6': {'domain': 'CODE_STRICT_CLEAN_OLD6', 'family': 'HH', 'config': '24_L1', 'architecture': 'AntisymLinear', 'top1': 0.0, 'over_random': 0.0, 'pairwise': nan, 'cycle': 0.0, 'margin_mean': 0.6804154217243195, 'margin_std': 0.2965801554579709}, 'HH_200': {'domain': 'HH_200', 'family': 'HH', 'config': '47_concat_L1_L4', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.8550000190734863, 'over_random': 0.35500001907348633, 'pairwise': 0.8550000190734863, 'cycle': nan, 'margin_mean': 1.262777328491211, 'margin_std': 1.140013337135315}}, 'interpretation': 'NoNorm remains useful in objective domains, while AntisymLinear remains the safer default for relational/noisy preference-like distinctions.'}`
- reasoning pilot result: `{'REASONING_TRANSFER_VERDICT': 'GOOD', 'n_tournaments': 25, 'random_top1_baseline': 0.4400000047683716, 'best_overall': {'family': 'CODE', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 1.0, 'over_random': 0.5599999952316284, 'pairwise': 1.0, 'cycle': 0.0}, 'best_hh': {'family': 'HH', 'config': '36_mean', 'architecture': 'AntisymLinear', 'top1': 0.96, 'over_random': 0.5199999952316283, 'pairwise': 0.9861111111111112, 'cycle': 0.0}, 'best_code': {'family': 'CODE', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 1.0, 'over_random': 0.5599999952316284, 'pairwise': 1.0, 'cycle': 0.0}, 'per_dataset': {'ai2_arc_challenge': {'baseline': 0.4270833395421505, 'best': {'family': 'CODE', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 1.0, 'over_random': 0.5729166604578495, 'pairwise': 1.0, 'cycle': 0.0}}, 'openbookqa': {'baseline': 0.4629629651705424, 'best': {'family': 'HH', 'config': '36_L4', 'architecture': 'AntisymLinear', 'top1': 1.0, 'over_random': 0.5370370348294575, 'pairwise': 1.0, 'cycle': 0.0}}}}`
- RECOMMENDED_NEXT: `add_reasoning_as_third_objective_eval_domain`
- full reports: `artifacts/reports/probes/bg_cross_domain_reasoning_audit_2026-05-17_summary.md`, `artifacts/reports/probes/bg_fixed_config_cross_domain_audit_2026-05-17.md`, `artifacts/reports/probes/reasoning_branch_transfer_2026-05-17.md`
