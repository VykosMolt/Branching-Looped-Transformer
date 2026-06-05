# Post-v10 synthesis v7 - actual state and next step

**Date:** 2026-05-17
**Status:** current working synthesis after clean GSM8K, code pilots, harness fixes, near-miss enrichment, and balancing.

## 1. Executive Opinion

HH-trained tiny taps now have enough preliminary evidence of transfer to clean objective generated branches for local planning. The remaining bottleneck is not whether a signal exists; it is branch-curriculum quality, especially producing same-task strict-clean alternatives for code.

## 2. Ground Rules / Framing

The evaluator remains relational, pairwise, and branch-selection oriented. Candidate labels come from external answer verifiers or unit tests. Tap/evaluator scores are not labels. Do not frame the system as `score(x)=quality`.

NoNorm adds a nuance: `score(a,b)=w*(a-b)=u(a)-u(b)`, so objective correctness domains can be scalar-readable without refuting the original HH relational/noisy preference finding.

## 3. Math Gate-Prep Demotion

The full MATH gate-prep path is demoted as a local blocker. MATH generation remains budget- and verbosity-shaped under the local setup. GSM8K and code are currently better objective-branch domains for testing transfer and curriculum quality.

## 4. Math Pilot Validity Probe

- DATA_VALIDITY: `TRUNCATION_CONFOUNDED`
- Main reason: old pilot had truncation and difficulty-composition confounds.
- Parser was not the issue: extractable answer was reported as 100%.
- Old `TRANSFER_POOR` is suspect/confounded, not a final negative result.

## 5. Clean GSM8K Micro

- CLEAN_GSM8K_VERDICT: `CLEAN`
- CLEAN_TRANSFER_VERDICT: `PRELIM_GOOD`
- tournaments: `5`
- best row: `{'config': '47_L4', 'architecture': 'AntisymLinearNoNorm', 'top1': 1.0, 'pairwise': 0.9166666666666666, 'cycle': 0.0}`

## 6. Expanded Clean GSM8K + GRU Control

- EXPANDED_CLEAN_GSM8K_VERDICT: `CLEAN_MINIMUM`
- EXPANDED_LINEAR_TRANSFER_VERDICT: `GOOD`
- GRU_CONTROL_VERDICT: `GRU_WEAK`
- prompts processed: `80`
- attempts generated: `892`
- clean tournaments: `28`
- kept branches: `79`
- random top1 baseline: `0.5625000074505806`
- near-miss fraction: `0.5588235294117647`
- best AntisymLinear: `{'config': '36_L4', 'architecture': 'AntisymLinear', 'top1': 0.7857142857142857, 'pairwise': 0.7037037037037037, 'cycle': 0.0}`
- best NoNorm: `{'config': '24_L4', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.75, 'pairwise': 0.7407407407407407, 'cycle': 0.0}`
- best GRU: `{'config': 'gru24_sequence', 'architecture': 'NA', 'top1': 0.7142857142857143, 'pairwise': 0.6666666666666666, 'cycle': 0.0}`

Interpretation: HH-trained taps transfer to clean GSM8K. GRU was above random but did not justify added temporal complexity.

## 7. Code v1 All-Correct Collapse

- CODE_TOURNAMENT_VERDICT: `TOO_FEW_TOURNAMENTS`
- CODE_TRANSFER_VERDICT: `NOT_RUN`
- label counts: `{'correct': 32, 'nonsense': 7, 'near_miss': 1}`
- Issue: wrapper/final route was too successful; 32/40 candidates passed all tests, leaving too few mixed tournaments.

## 8. Code v2 Dirty Diagnostic Result

- CODE_V2_TOURNAMENT_VERDICT: `DIAGNOSTIC_ONLY`
- CODE_V2_TRANSFER_VERDICT: `GOOD`
- Status: historical/pre-fix diagnostic only.
- Manual bucket breakdown: runnable_zero_pass_wrong=50, parseable_runtime_error=25, prose_or_wrapper_not_code=20, syntax_invalid_code=6, safety_rejected=3, parseable_no_function=1.

## 9. Wrapper / Taskset / Harness Fixes

- FIX_VERDICT: `PATCHED_AND_UNIT_TESTED`
- Wrapper/prose/status outputs are rejected as code.
- MBPP function-name and signature issues were fixed.
- Labels are split into `correct`, `near_miss`, `wrong_code`, `runtime_error`, `malformed`, and `safety_rejected`.
- `sys.setrecursionlimit` is allowed while unsafe sys/file/network/process usage remains rejected.
- Validation reported `py_compile` success and local-agent wrapper tests: 139 passed.

## 10. Patched Code v2-Mini

- CODE_V2_MINI_TOURNAMENT_VERDICT: `RUNNABLE_DIAGNOSTIC`
- CODE_V2_MINI_TRANSFER_VERDICT: `GOOD`
- tasks: `30`
- unique candidates: `112`
- labels: `{'correct': 49, 'near_miss': 11, 'wrong_code': 50, 'runtime_error': 0, 'malformed': 2, 'safety_rejected': 0}`
- strict/diagnostic_runnable/diagnostic_mixed: `2 / 8 / 8`
- random top1 baseline: `0.5625`
- best AntisymLinear: `{'config': '36_mean', 'architecture': 'AntisymLinear', 'top1': 0.875, 'pairwise': 0.78125, 'cycle': 0.0}`
- best NoNorm: `{'config': '47_L4', 'architecture': 'AntisymLinearNoNorm', 'top1': 1.0, 'pairwise': 0.875, 'cycle': 0.0}`

Interpretation: the transfer signal survived cleanup, but the primary set was runnable-diagnostic, not strict-clean.

## 11. Independent 10-Task Near-Miss Enrichment

- outcome: `MISS`
- tasks: `10`
- unique candidates: `37`
- labels: `{'correct': 14, 'near_miss': 12, 'wrong_code': 11}`
- strict_clean: `2`
- Interpretation: harness cleanliness was good and correct/near-miss candidates existed globally, but same-task pairing was weak.

## 12. Near-Miss Balancing Pass

- BALANCE_INSPECTION_VERDICT: `READY`
- BALANCING_GENERATION_VERDICT: `COMPLETED`
- BALANCED_TOURNAMENT_VERDICT: `RED`
- BALANCED_TRANSFER_VERDICT: `NOT_RUN`
- before labels: `{'correct': 14, 'near_miss': 12, 'wrong_code': 11, 'runtime_error': 0, 'malformed': 0, 'safety_rejected': 0}`
- after labels: `{'correct': 18, 'near_miss': 16, 'wrong_code': 13, 'runtime_error': 0, 'malformed': 0, 'safety_rejected': 0}`
- strict_clean stayed: `2 -> 2`
- Interpretation: the global pool improved, but no tasks converted; within-task pairing is the bottleneck.

## 13. Generalization Verdict

`GENERALIZATION_VERDICT = SUPPORTED_FOR_LOCAL_PLANNING`

Signal generalization is preliminarily supported for local planning. It is not a proof of full gate-scale transfer, MATH transfer under local budget, or controller effects.

## 14. Scalar / Pointwise-Ranking Verdict

`POINTWISE_RANKING_VERDICT = SUPPORTED_IN_OBJECTIVE_DOMAINS`

NoNorm suggests objective correctness domains can be scalar-readable. This does not refute the original HH relational finding; HH preference remains predominantly relational/noisy.

## 15. Current Bottleneck: Within-Task Pairing / Task Curriculum

`STRICT_CLEAN_BOTTLENECK = WITHIN_TASK_PAIRING`

Code strict-clean near-miss generation remains unsolved. The next data problem is screening tasks for natural correct-vs-near-miss alternatives before spending full generation budget.

## 16. Recommended Next Move

`RECOMMENDED_NEXT = task_screening_for_strict_clean_ready_code_tasks`

Run a cheap task-screening protocol for strict-clean-ready code tasks: one strong anchor attempt plus two near-miss-seeking attempts per task, labeled only by unit tests, then scale only tasks that already show same-task pairing.

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
