# BG / Tap Transfer Handoff After v4

**Date:** 2026-05-17  
**Scope:** Everything important that happened after the v4 synthesis / AntisymLinear pivot, including math validity, clean GSM8K, code branch pilots, wrapper/harness fixes, strict-clean screening, strict-clean transfer, and the current active Codex run.  
**Status:** Current working handoff for continuation in a fresh chat or with another agent.

---

## 0. Executive summary

The project is no longer in the phase of proving that a branch-selection signal exists. Clean GSM8K and patched code pilots support enough preliminary transfer for local planning. The active bottleneck is now **branch curriculum**, especially generating same-task strict-clean correct-vs-near-miss code contrasts.

Current best interpretation:

> Ouro-RLTT hidden states contain branch-selection information that tiny heads can read. In objective domains like GSM8K and code, the signal is often scalar-readable with `AntisymLinearNoNorm`; this does **not** refute the original HH-RLHF finding that social/preference geometry is predominantly relational/noisy. The BG controller remains a selector over alternatives, not a pointwise judge.

Current default head family:

```python
AntisymLinear:
    LayerNorm(elementwise_affine=False)(left - right) -> Linear(bias=False)

AntisymLinearNoNorm:
    Linear(bias=False)(left - right)
```

Current default tap interface:

```text
layer 24: single-state 2048-dim tap
layer 36: single-state 2048-dim tap
layer 47: domain-dependent late tap, often L4 / concat / all-loops
```

GRU is **not** default. It remains a capacity/temporal aggregation control only.

Current active bottleneck:

```text
Within-task pairing.
```

The generator can produce correct code candidates and near-miss candidates globally, but it often fails to produce both for the **same task**, which is required for strict-clean branch-selection tournaments.

---

## 1. Ground rules preserved from v4

The v4 synthesis pivot remains the foundation:

1. The tap/evaluator is **relational/pairwise**.
2. Do not write or imply `score(x) = quality`.
3. Branch labels must come from external/objective verifiers: exact-answer checkers, unit tests, environment scores, etc.
4. Evaluator/tap scores must never become labels.
5. Generated-branch tournaments are the BG-relevant gate, not HH centered accuracy.
6. Phase 2 / cloud / backbone modification remains off the table until local branch-selection evidence is much cleaner.

Important nuance added after v4:

- In objective domains, `AntisymLinearNoNorm` can behave like a scalar utility readout:

```python
score(a, b) = w · (a - b) = u(a) - u(b)
```

This is acceptable and sometimes useful. It does not invalidate the relational framing for HH because HH preference remains noisier and less pointwise-accessible.

---

## 2. What changed after v4 — chronological run history

### 2.1 Math gate-prep was demoted

The initial post-v4 plan still treated math gate-prep as the next major blocker. That changed because of local compute constraints and Ouro-RLTT verbosity.

Problem:

- MATH generation under local budgets is expensive and often confounded by truncation / yapping.
- Running large MATH gate-prep locally was not worth the cost.
- GSM8K can be made clean with strict wrappers; MATH remains a stress-test rather than the immediate training/eval domain.

Outcome:

```text
MATH full gate-prep: deferred / not local priority.
GSM8K clean micro + expanded: used instead.
Code branch tournaments: promoted as primary local branch-curriculum domain.
```

---

### 2.2 Math pilot validity probe

A post-hoc validity probe was run on the old math pilot.

Top-level result:

```text
DATA_VALIDITY = TRUNCATION_CONFOUNDED
```

Key findings:

- The old pilot JSON had 33 tournaments.
- It did not store output token counts or truncation flags directly, so completion lengths were recovered with the local Ouro-RLTT tokenizer.
- Parser was **not** the problem: `has_extractable_answer = 100%`, stored vs recomputed parser outputs had 0 mismatches.
- The confound was truncation and difficulty composition.
- Incorrect GSM8K branches were especially truncation-heavy.

Important table from the run:

| source | correctness | n | trunc_rate | extractable_answer |
|---|---:|---:|---:|---:|
| GSM8K | correct | 49 | 55.1% | 100.0% |
| GSM8K | incorrect | 55 | 80.0% | 100.0% |
| MATH | correct | 10 | 30.0% | 100.0% |
| MATH | incorrect | 18 | 11.1% | 100.0% |

Near-miss / nonsense:

| source | incorrect | near_miss | nonsense | near_miss_frac |
|---|---:|---:|---:|---:|
| GSM8K | 55 | 9 | 46 | 16.4% |
| MATH | 18 | 11 | 7 | 61.1% |

Interpretation:

- The old `TRANSFER_POOR` result was suspect and should not drive architecture decisions.
- The old math-trained 1.000 results were likely inflated by malformed/easy-after-filtering effects.
- This justified regenerating a tiny clean GSM8K set instead of trusting the old pilot.

---

### 2.3 Clean GSM8K extreme micro

A tiny clean GSM8K benchmark was generated with aggressive final-answer wrappers.

Top-level result:

```text
CLEAN_GSM8K_VERDICT = CLEAN
CLEAN_TRANSFER_VERDICT = PRELIM_GOOD
```

Key facts:

- 5 kept clean GSM8K tournaments.
- 15 prompts.
- 168 generated attempts.
- Kept branches were 100% parseable.
- Kept branches had 100% `FINAL ANSWER` present.
- 0 kept incorrect branches hit max_new_tokens.
- Random top-1 baseline: 0.633.
- Best HH-trained row:
  - `47_L4 / AntisymLinearNoNorm`
  - top1 = 1.000
  - pairwise = 0.917
  - cycle = 0.000

Interpretation:

- The old negative HH→math transfer result did not reproduce on a clean set.
- This was too small for final conclusions but enough to justify expansion.

---

### 2.4 Expanded clean GSM8K + GRU control

The clean GSM8K benchmark was expanded.

Top-level result:

```text
EXPANDED_CLEAN_GSM8K_VERDICT = CLEAN_MINIMUM
EXPANDED_LINEAR_TRANSFER_VERDICT = GOOD
GRU_CONTROL_VERDICT = GRU_WEAK
```

Key metrics:

| metric | value |
|---|---:|
| prompts processed | 80 |
| attempts generated | 892 |
| clean tournaments | 28 |
| kept branches | 79 |
| random top-1 baseline | 0.563 |
| near-miss fraction | 0.559 |
| clean verdict | CLEAN_MINIMUM |

Best transfer rows:

| family | config | top1 | pairwise | cycle |
|---|---|---:|---:|---:|
| AntisymLinear | 36_L4 | 0.786 | 0.704 | 0.000 |
| NoNorm | 24_L4 | 0.750 | 0.741 | 0.000 |
| GRU | gru24_sequence | 0.714 | 0.667 | 0.000 |

GRU nuance:

- GRU was not useless. It selected above random.
- It failed the control’s purpose: it did not show temporal loop aggregation adds value over simpler exact-antisymmetric heads.
- Bias was low, but HH holdout accuracy stayed weak.
- Interpretation: temporal aggregation does not justify added complexity on this clean GSM8K control.

Interpretation:

- HH-trained taps transfer to clean GSM8K.
- NoNorm competitiveness suggests objective math correctness can be scalar-readable at some taps.
- GRU remains a control/escalation, not a default.

---

## 3. Code branch work after v4

### 3.1 Code branch pilot v1 — wrapper too successful

Top-level result:

```text
CODE_INTERFACE_VERDICT = READY
CODE_TASKSET_VERDICT = READY
CODE_GENERATION_VERDICT = READY
CODE_TOURNAMENT_VERDICT = TOO_FEW_TOURNAMENTS
CODE_TRANSFER_VERDICT = NOT_RUN
```

Key metrics:

| item | value |
|---|---:|
| tasks | 10 |
| sources | 4 local_dsa / 6 MBPP |
| candidates | 40 |
| direct_final | 30 |
| first_tool_code | 10 |
| correct | 32 |
| near_miss | 1 |
| nonsense | 7 |
| strict-clean tournaments | 1 |
| diagnostic-mixed tournaments | 4 |

Interpretation:

- The local-agent wrapper was too successful on the task mix.
- Most candidates were correct or duplicate/near-duplicate correct implementations.
- This was not a tap failure. It was a candidate-diversity failure.

---

### 3.2 Code branch pilot v2 pre-fix — transfer signal but dirty harness

Code v2 increased task count and difficulty and harvested pre-final/failure-stage candidates.

Top-level result:

```text
CODE_V2_INTERFACE_VERDICT = READY
CODE_V2_TASKSET_VERDICT = READY
CODE_V2_GENERATION_VERDICT = READY
CODE_V2_TOURNAMENT_VERDICT = DIAGNOSTIC_ONLY
CODE_V2_TRANSFER_VERDICT = GOOD
```

Key metrics:

| Metric | Value |
|---|---:|
| tasks | 40 |
| source mix | 6 local_dsa / 14 MBPP / 20 HumanEval |
| difficulty mix | 2 devil / 5 hard / 33 medium |
| unique candidates | 188 |
| duplicate rate | 30.9% |
| correct / near_miss / nonsense | 70 / 13 / 105 |
| strict-clean tournaments | 3 |
| diagnostic-mixed tournaments | 22 |
| primary eval candidates captured | 108 |
| random top-1 baseline | 0.552 |

Best rows:

| family | config | top1 | over baseline | pairwise | cycle |
|---|---|---:|---:|---:|---:|
| AntisymLinearNoNorm | 36_mean | 0.727 | +0.175 | 0.691 | 0.000 |
| AntisymLinear | 36_L4 | 0.682 | +0.130 | 0.709 | 0.000 |
| AntisymLinear | 47_concat_all_loops | 0.636 | +0.084 | 0.764 | 0.000 |

Manual inspection found the `nonsense` bucket was too broad:

| bucket | count | meaning |
|---|---:|---|
| runnable_zero_pass_wrong | 50 | valid code shape, ran, failed all tests |
| parseable_runtime_error | 25 | code parsed but crashed / wrong arity / NameError |
| prose_or_wrapper_not_code | 20 | wrapper status/prose admitted as candidate |
| syntax_invalid_code | 6 | incomplete or invalid Python |
| safety_rejected | 3 | rejected by safety policy |
| parseable_no_function | 1 | Python parses but no function |

Main issues:

1. `mbpp/232` taskset bug: parsed function name as `set`, but tests call `larg_nnum`.
2. `repaired_final` contained dirty wrapper/status outputs.
3. HumanEval zero-pass failures were too coarse.
4. `mbpp/306` had signature ambiguity.
5. `sys.setrecursionlimit` was over-rejected.

Interpretation:

- Pre-fix code v2 was promising but dirty.
- It should be treated as historical/pre-fix evidence only.

---

### 3.3 Wrapper/taskset/harness fixes

Top-level result:

```text
FIX_VERDICT = PATCHED_AND_UNIT_TESTED
```

Local-agent fixes:

- Reject wrapper/prose outputs like `[Max steps reached ...]` as code.
- `sanitize_tool_input("python", ...)` returns empty for obvious non-code payloads.
- `final_python_code_for_answer(...)` extracts real Python code blocks and refuses prose/status text.

Harness fixes:

- MBPP function extraction skips outer builtins, so `mbpp/232` targets `larg_nnum`, not `set`.
- MBPP prompts include inferred exact signature shape, fixing cases like `mbpp/306`.
- Added tuple-canonicalization hint for `mbpp/237`.
- Labels split into:
  - `correct`
  - `near_miss`
  - `wrong_code`
  - `runtime_error`
  - `malformed`
- Legacy `nonsense` retained only for compatibility.
- Safe `sys.setrecursionlimit` allowed for DSA while unsafe sys usage remains rejected.
- Candidate generation now writes partial JSON after each task and supports resume.
- Evaluation writes partial tournament JSON after each task with provisional tournaments and verdict.

Validation:

- `py_compile` passed.
- Local-agent wrapper tests passed: `139 passed`.

Interpretation:

- Old code v2 transfer became historical/pre-fix.
- Any valid code-domain result should be produced through the patched path.

---

### 3.4 Patched code v2-mini

A small patched rerun was executed through the fixed wrapper/taskset path.

Top-level result:

```text
CODE_V2_PATCH_STATUS = READY
CODE_V2_MINI_TASKSET_VERDICT = READY
CODE_V2_MINI_GENERATION_VERDICT = READY
CODE_V2_MINI_TOURNAMENT_VERDICT = RUNNABLE_DIAGNOSTIC
CODE_V2_MINI_TRANSFER_VERDICT = GOOD
```

Key metrics:

| metric | value |
|---|---:|
| tasks | 30 |
| unique candidates | 112 |
| labels | 49 correct / 11 near_miss / 50 wrong_code / 0 runtime_error / 2 malformed / 0 safety_rejected |
| tournaments | 2 strict_clean / 8 diagnostic_runnable / 8 diagnostic_mixed |
| random top-1 baseline | 0.5625 |

Best rows:

| family | config | top1 | pairwise | cycle |
|---|---|---:|---:|---:|
| NoNorm | 47_L4 | 1.000 | 0.875 | 0.000 |
| AntisymLinear | 36_mean | 0.875 | 0.781 | 0.000 |

Interpretation:

- Transfer signal survived cleanup.
- Primary evaluation was runnable-diagnostic, not strict-clean.
- The strongest row was NoNorm `47_L4`, suggesting a scalar-readable late-state projection for this code diagnostic set.
- Intermediate layer 36 remained competitive through `36_mean / AntisymLinear`.

---

### 3.5 Independent 10-task near-miss enrichment

This run targeted near-miss yield specifically.

Top-level outcome:

```text
Outcome = MISS
```

Metrics:

| metric | value |
|---|---:|
| tasks | 10 |
| unique candidates | 37 |
| correct | 14 |
| near_miss | 12 |
| wrong_code | 11 |
| malformed | 0 |
| safety_rejected | 0 |
| strict_clean | 2 |

Interpretation:

- Harness was clean.
- Correct and near-miss candidates were both produced globally.
- The failure was within-task pairing: too many tasks had only one side of the desired pair.

---

### 3.6 Near-miss balancing pass

A targeted balancing pass tried to generate the missing side for one-sided tasks.

Top-level result:

```text
BALANCE_INSPECTION_VERDICT = READY
BALANCING_GENERATION_VERDICT = COMPLETED
BALANCED_TOURNAMENT_VERDICT = RED
BALANCED_FEATURE_VERDICT = NOT_RUN
BALANCED_TRANSFER_VERDICT = NOT_RUN
```

Counts:

```text
before: 14 correct / 12 near_miss / 11 wrong_code
after:  18 correct / 16 near_miss / 13 wrong_code
strict_clean stayed 2
```

Interpretation:

- The pass added useful candidates globally.
- It did not convert any tasks to strict-clean.
- The bottleneck is not candidate quality globally; it is **same-task pairing**.

---

### 3.7 Strict-clean task screening

A screening pass was run to find tasks that naturally produce correct + near-miss under cheap generation modes.

Top-level result:

```text
SCREENING_TASKPOOL_VERDICT = READY
SCREENING_GENERATION_VERDICT = COMPLETED
STRICT_CLEAN_SCREENING_VERDICT = YELLOW
RECOMMENDED_NEXT = run_small_transfer_or_screen_more_tasks
```

Key metrics:

| metric | value |
|---|---:|
| tasks screened | 60 |
| candidates generated | 178 |
| primary unique candidates | 117 |
| strict_clean_ready tasks | 6 |
| label totals | 73 correct / 20 near_miss / 24 wrong_code |
| task classes | 6 strict_clean_ready / 5 anchor_only / 7 near_miss_only / 36 all_correct / 6 all_wrong / 0 malformed |

Strict-clean-ready tasks found:

```text
mbpp/100
mbpp/129
mbpp/283
mbpp/291
mbpp/391
mbpp/392
```

Interpretation:

- Within-task pairing bottleneck confirmed.
- The bottleneck is not absolute: six tasks were strict-clean-ready.
- Most tasks are all-correct, meaning the wrapper remains strong for the screened task pool.

---

### 3.8 Strict-clean transfer micro-eval

The six strict-clean-ready tasks were used for a transfer micro-eval.

Top-level result:

```text
STRICT_CLEAN_TRANSFER_SET_VERDICT = READY
STRICT_CLEAN_FEATURE_VERDICT = READY
STRICT_CLEAN_TRANSFER_VERDICT = WEAK
RECOMMENDED_NEXT = screen_more_strict_clean_tasks_or_consider_code_specific_training
```

Primary strict-clean set:

| metric | value |
|---|---:|
| tasks | 6 |
| primary candidates | 13 |
| correct / near_miss | 7 / 6 |
| random top-1 baseline | 0.528 |
| best overall | 47_concat_all_loops / AntisymLinearNoNorm |
| top1 | 0.500 |
| pairwise | 0.571 |
| cycle | 0.000 |

Secondary diagnostic with wrong_code included:

| metric | value |
|---|---:|
| candidates | 15 |
| random top-1 baseline | 0.472 |
| best overall | 24_L4 / AntisymLinear |
| top1 | 0.500 |
| pairwise | 0.556 |
| cycle | 0.000 |

Interpretation:

- HH-trained taps are weak on strict-clean correct-vs-near-miss code tasks.
- Runnable-diagnostic transfer is GOOD, but strict near-miss transfer is not solved.
- This is the first result cutting against the optimistic read.

---

## 4. Current active Codex prompt

The active Codex prompt is the **two-step code-domain prompt**:

1. **Code-specific tiny-head training control**
2. **Strict-clean task-screening expansion**

Purpose:

```text
A. Is strict-clean code weakness caused by HH-trained projection mismatch?
   -> Train AntisymLinear / NoNorm on existing code-generated branches,
      excluding the 6 strict-clean eval tasks.

B. Can we find more strict-clean-ready code tasks?
   -> Screen additional tasks cheaply for correct + near_miss within the same task.
```

Important guardrails:

```text
- no MATH
- no GRU
- no HH Experiment 2 Redux
- no published 5M evaluator
- no huge code expansion
- no git commands
- no model/checkpoint modifications
```

Verdicts expected:

```text
CODE_SPECIFIC_SPLIT_VERDICT
CODE_SPECIFIC_FEATURE_VERDICT
CODE_SPECIFIC_TINY_HEAD_VERDICT
STRICT_CLEAN_SCREENING_EXPANSION_VERDICT
```

Interpretation of possible outcomes:

- If code-specific tiny heads are GOOD on strict-clean eval: hidden states contain the signal; HH projection did not transfer strongly enough to strict near-miss code.
- If code-specific tiny heads remain WEAK/POOR: strict near-miss code signal may require more data, better pooling, richer heads, or more controlled branch generation.
- If screening expansion finds more strict-clean-ready tasks: proceed to a larger strict-clean transfer eval.
- If screening remains low-yield: task/test design is the active bottleneck.

---

## 5. Current conclusions

### 5.1 Generalization status

Current verdict:

```text
GENERALIZATION_VERDICT = SUPPORTED_FOR_LOCAL_PLANNING
```

Reason:

- Clean GSM8K expanded transfer was GOOD.
- Patched code v2-mini transfer was GOOD on runnable-diagnostic code branches.
- Old negative math transfer was confounded by truncation and difficulty composition.

This is not a full proof of universal generalization. It is enough to plan locally around branch-selection signal existing.

### 5.2 Pointwise / scalar-readable status

Current verdict:

```text
POINTWISE_RANKING_VERDICT = SUPPORTED_IN_OBJECTIVE_DOMAINS
```

Reason:

- NoNorm was competitive or winning in clean GSM8K and patched code.
- NoNorm is transitive and scalar-readable:

```python
score(a, b) = w · (a - b) = u(a) - u(b)
```

Important caveat:

This does **not** refute the original HH relational/noisy preference framing. It suggests objective domains such as math/code correctness can be more scalar-readable than HH preference.

### 5.3 Strict-clean bottleneck

Current verdict:

```text
STRICT_CLEAN_BOTTLENECK = WITHIN_TASK_PAIRING
```

Reason:

- Correct, near_miss, and wrong_code candidates exist globally.
- Strict-clean remains low because correct and near_miss candidates often do not appear for the same task.
- Balancing old tasks failed; task screening worked better.

### 5.4 GRU status

Current verdict:

```text
GRU = CONTROL / ESCALATION ONLY
```

Reason:

- On expanded clean GSM8K, GRU was above random but underperformed simpler heads.
- It did not demonstrate added value from temporal loop aggregation.
- It should not be Phase-1 default.

---

## 6. Recommended next work after the active Codex run

Wait for the active Codex run.

Depending on result:

### If code-specific tiny heads are GOOD and screening expansion finds more strict-clean tasks

Then run a larger strict-clean eval comparing:

```text
HH-trained taps
code-specific taps
AntisymLinear
AntisymLinearNoNorm
24/36/47 configs
```

Goal:

```text
Confirm whether code-specific projection fixes strict-clean weakness.
```

### If code-specific tiny heads are GOOD but screening finds few tasks

Then task source/test granularity remains bottleneck. Improve task sources before more modeling.

### If code-specific tiny heads are WEAK/POOR but screening finds more tasks

Rerun strict-clean transfer with larger n before concluding architecture failure.

### If both code-specific heads and screening fail

Then strict-clean code selection needs either:

- richer pooling/features,
- richer head architecture,
- more controlled task/test design,
- or a different domain.

---

## 7. Practical advice for future agents

Do not rerun old MATH gate-prep.  
Do not trust old confounded math transfer.  
Do not rerun dirty pre-fix code v2.  
Do not run GRU unless explicitly testing capacity.  
Do not use static datasets as tap-training examples directly.

Use datasets as:

```text
task prompts + tests + external labels
```

not as:

```text
arbitrary static hidden-state training pairs
```

Training/evaluation candidates should be generated by Ouro-RLTT or the local-agent wrapper, then labeled externally.

---

## 8. One-line handoff

The BG/tap project has preliminary generalization evidence on clean GSM8K and patched code, but strict near-miss code selection remains weak. The active blocker is not signal existence; it is building a branch curriculum that reliably produces same-task correct-vs-near-miss contrasts. Current active work tests whether code-specific tiny heads fix strict-clean weakness and whether more strict-clean-ready tasks can be screened.

