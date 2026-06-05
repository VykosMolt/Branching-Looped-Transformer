# BG / Tap Transfer Handoff After v7

**Date:** 2026-05-17  
**Scope:** Everything that happened after `bg_after_v4_handoff_2026-05-17.md` / the v7 synthesis: code-specific control, strict-clean screening expansion, expanded strict-clean projection comparison, inverse transfer to HH, random HH split audit, broad cross-domain fixed-config audit, reasoning pilot, and loop/layer diagnostics.  
**Status:** Current working state after the broad BG/tap audit. This supersedes the prior handoff as the continuation document.

---

## 0. Executive update

The project is now beyond the question **“does a branch-selection signal generalize at all?”**

The current best answer is:

> HH-trained tiny taps generalize enough for local planning on clean GSM8K, runnable-diagnostic code, and some reasoning branches. However, hard same-task strict-clean code contrasts benefit materially from **domain-specific projection training**. The practical BG architecture should therefore use a **general HH-trained selector plus domain-specialist selectors** for hard objective domains.

Current headline verdicts:

```text
GENERALIST_SPECIALIST_VERDICT = DOMAIN_SPECIALISTS_NEEDED
REASONING_TRANSFER_VERDICT = GOOD
RECOMMENDED_NEXT = add_reasoning_as_third_objective_eval_domain
```

The major active shift since the previous handoff:

```text
Before:
  The strict-clean bottleneck looked mostly like task/curriculum difficulty.

Now:
  Task/curriculum still matters, but strict-clean code states are readable.
  The weak HH-trained result was mainly projection mismatch.
  Code-specific tiny heads substantially improve strict-clean code selection.
```

---

## 1. Ground rules that still hold

The original v4/v7 framing still stands:

1. The BG/tap is a **branch-selection mechanism**, not a pointwise judge.
2. Do not write or imply `score(x) = quality`.
3. Candidate labels come only from external/objective sources:
   - HH chosen/rejected labels,
   - exact-answer verifiers,
   - unit tests,
   - answer keys,
   - environment outcomes.
4. Tap scores must never become labels.
5. Generated branches are evaluation/training examples only after external labeling.
6. GRU remains a control/escalation, not default.
7. AntisymLinear and AntisymLinearNoNorm remain the default tiny-head family.

Tiny head family:

```python
AntisymLinear:
    LayerNorm(elementwise_affine=False)(left - right) -> Linear(bias=False)

AntisymLinearNoNorm:
    Linear(bias=False)(left - right)
```

Interpretation of NoNorm:

```python
score(a, b) = w · (a - b) = u(a) - u(b)
```

So NoNorm is a transitive / scalar-readable branch-ranking ablation. Its success in objective domains does **not** refute the original HH relational/noisy preference framing.

---

## 2. Completed since the previous handoff

### 2.1 Code-specific tiny-head control

The prompt after the earlier strict-clean WEAK result tested whether strict-clean code weakness was caused by state unreadability or by HH-trained projection mismatch.

Top-line results:

```text
CODE_SPECIFIC_SPLIT_VERDICT = MISSING_FEATURES
CODE_SPECIFIC_FEATURE_VERDICT = RECAPTURED
CODE_SPECIFIC_TINY_HEAD_VERDICT = GOOD
STRICT_CLEAN_SCREENING_EXPANSION_VERDICT = GREEN
RECOMMENDED_NEXT = evaluate_code_specific_and_hh_trained_taps_on_expanded_strict_clean_set
```

Code-specific control details:

```text
training split: 14 tasks, 138 primary pairwise pairs
held-out strict-clean eval: 6 tasks
best code-trained AntisymLinear: 24_L4, top1 0.833, pairwise 0.714, cycle 0.000
best code-trained NoNorm:       24_L4, top1 0.833, pairwise 0.714, cycle 0.000
```

Interpretation:

> The strict-clean code signal is present in the states. The earlier HH-trained strict-clean WEAK result was not evidence that the states lack the signal; it was evidence that the HH-trained projection was not the best readout for hard code near-miss contrasts.

This was a crucial disambiguation.

---

### 2.2 Strict-clean screening expansion

The same run also screened more tasks for strict-clean readiness.

Results:

```text
cumulative tasks screened: 137
new strict-clean-ready tasks: 10
STRICT_CLEAN_SCREENING_EXPANSION_VERDICT = GREEN
```

New strict-clean-ready task IDs:

```text
mbpp/11
mbpp/20
mbpp/434
HumanEval/10
HumanEval/118
HumanEval/123
HumanEval/125
HumanEval/141
HumanEval/148
HumanEval/69
```

Notes:

- The run switched to HumanEval after the MBPP tail stayed low-yield.
- HumanEval checks were granularized where possible.
- This confirmed the previous diagnosis: strict-clean tasks exist, but they require screening.
- The within-task pairing bottleneck is real but tractable.

---

### 2.3 Expanded strict-clean code projection comparison

After obtaining the new 10 strict-clean-ready tasks, Codex ran the expanded strict-clean comparison.

Top-line results:

```text
EXPANDED_STRICT_CLEAN_SET_VERDICT = READY
EXPANDED_STRICT_CLEAN_FEATURE_VERDICT = RECAPTURED
CODE_SPECIFIC_EXPANDED_TRAIN_VERDICT = RETRAINED
EXPANDED_HH_TRANSFER_VERDICT = GOOD
EXPANDED_CODE_SPECIFIC_TRANSFER_VERDICT = GOOD
EXPANDED_STRICT_CLEAN_COMPARISON_VERDICT = CODE_SPECIFIC_ADVANTAGE
RECOMMENDED_NEXT = update_BG_phase1_plan_for_domain_specific_projection_training
```

ALL16 primary eval:

```text
random top1 baseline: 0.562
```

Best HH-trained:

```text
47_mean / AntisymLinear
top1    0.750
pairwise 0.600
cycle   0.000
```

Best code-trained:

```text
36_L4 / AntisymLinear
top1    0.875
pairwise 0.833
cycle   0.000
```

Interpretation:

> HH-trained taps do transfer to expanded strict-clean code, but code-specific taps are substantially better. The signal exists in the states; hard code near-miss comparisons benefit from domain-specific projection training.

This result changed the Phase-1 BG architecture implication.

Old tentative view:

```text
Maybe HH-trained taps can be the universal default everywhere.
```

Updated view:

```text
HH-trained taps are a useful general selector, but hard objective domains need domain-specialist projections.
```

---

### 2.4 Inverse transfer: code-trained taps on HH

A small inverse-transfer probe evaluated whether code-specific heads transfer back to HH preference pairs.

Initial held-out 20-pair result:

```text
CODE_TO_HH_TRANSFER_VERDICT = GOOD
HH_BASELINE_VERDICT = GOOD
```

Held-out split:

```text
best HH-trained:   36_mean / AntisymLinear, accuracy 0.700
best code-trained: 24_L1 / AntisymLinearNoNorm, accuracy 0.800
```

But all-200 diagnostic gave a much weaker code-trained picture:

```text
best HH-trained all-200:   0.885
best code-trained all-200: 0.535
```

Immediate interpretation:

> The held-out 20 result was interesting, but likely split-sensitive. Code-trained heads did not obviously replace HH-trained heads as general preference selectors.

---

### 2.5 Random 20-pair HH split audit

To check whether the held-out 20-pair inverse transfer was a fluke, Codex evaluated 10 random 20-pair HH splits.

Setup:

- Code heads trained once from saved expanded code features.
- HH heads retrained per split on the complementary 180 HH pairs.
- No generation, no new captures.

Results:

| family | mean acc | std | min | max |
|---|---:|---:|---:|---:|
| best HH-trained | 0.655 | 0.099 | 0.550 | 0.850 |
| best code-trained | 0.640 | 0.077 | 0.550 | 0.800 |
| HH AntisymLinear | 0.630 | 0.090 | 0.500 | 0.750 |
| HH NoNorm | 0.620 | 0.095 | 0.550 | 0.850 |
| CODE AntisymLinear | 0.625 | 0.056 | 0.550 | 0.700 |
| CODE NoNorm | 0.600 | 0.092 | 0.500 | 0.800 |

Interpretation at the time:

> Code-specific heads are not pure narrow code specialists. They carry some general candidate-coherence signal. But best-config selection is potentially inflating results, so fixed-config analysis is required.

This motivated the broad fixed-config audit.

---

## 3. Broad BG/tap audit

Codex then ran the broad BG/tap generalization audit. It did no git commands or commits.

Top-line results:

```text
BG_BACKLOG_AUDIT_VERDICT = READY
BG_HEAD_REGISTRY_VERDICT = RETRAINED
BG_CROSS_DOMAIN_MATRIX_VERDICT = READY
FIXED_CONFIG_AUDIT_VERDICT = READY
GENERALIST_SPECIALIST_VERDICT = DOMAIN_SPECIALISTS_NEEDED
REASONING_BRANCH_DATA_VERDICT = READY
REASONING_TRANSFER_VERDICT = GOOD
LOOP_LAYER_DIAGNOSTIC_VERDICT = READY
RECOMMENDED_NEXT = add_reasoning_as_third_objective_eval_domain
```

---

### 3.1 Backlog audit

Backlog audit status:

```text
DONE:      14 experiments
OBSOLETE:   3
ACTIVE:     2
DEFERRED:   2
BLOCKED:    1
```

Important classifications:

- Full MATH gate-scale: blocked/deferred by local budget and model verbosity.
- GRU as default: obsolete.
- Controller-policy simulator: deferred.
- Full HH capture/full split: deferred.
- Fixed-config audit and reasoning pilot: active in this run, now completed.

---

### 3.2 Head registry

The run built/retrained a reusable tiny-head registry.

Registry:

```text
40 tiny heads total
20 HH-trained heads
20 code-trained heads
10 configs each
2 architectures: AntisymLinear and AntisymLinearNoNorm
```

No new code candidates were generated. Heads were trained/retrained from cached artifacts only.

Important implication:

> We now have a reusable basis for cross-domain comparison between general HH-trained and code-trained specialist projections.

---

### 3.3 Cross-domain evaluation matrix

The cross-domain matrix loaded all 6 requested eval sets with feature coverage:

```text
HH_200:                      200 HH pairs
CLEAN_GSM8K_EXPANDED:        28 tournaments, 79 candidates
CODE_RUNNABLE_DIAGNOSTIC:     8 tournaments, 37 candidates
CODE_STRICT_CLEAN_OLD6:       6 tournaments, 13 candidates
CODE_STRICT_CLEAN_NEW10:     10 tournaments, 33 candidates
CODE_STRICT_CLEAN_ALL16:     16 tournaments, 46 candidates
```

This matrix is now the best cross-domain snapshot of the project.

---

### 3.4 Fixed-config result

Headline:

```text
GENERALIST_SPECIALIST_VERDICT = DOMAIN_SPECIALISTS_NEEDED
```

The broad audit found that no single projection is stable enough to be the only default across domains.

Best strict-clean code, ALL16:

```text
code-trained 36_L4 / AntisymLinear
top1    0.875
pairwise 0.833
cycle   0.000
```

Best HH-trained on ALL16:

```text
HH-trained 47_mean / AntisymLinear
top1    0.750
pairwise 0.600
cycle   0.000
```

Best HH_200:

```text
HH-trained 47_concat_L1_L4 / AntisymLinearNoNorm
accuracy/pairwise 0.855
```

Best code-trained on HH_200:

```text
code-trained 47_L4 / AntisymLinearNoNorm
accuracy/pairwise 0.535
```

Interpretation:

> Code-trained heads are clearly better on strict-clean code. HH-trained heads remain clearly better on HH preference pairs. This supports a general HH head plus domain specialists, not replacing the HH head with a code head.

This resolves the inverse-transfer ambiguity.

---

## 4. Reasoning pilot

A new small reasoning/multiple-choice branch pilot was run using ARC-Challenge and OpenBookQA.

Data verdict:

```text
REASONING_BRANCH_DATA_VERDICT = READY
```

Pilot stats:

```text
tasks seen:        30
mixed tournaments: 25
candidates:        120
correct:           47
incorrect:         64
unparseable:        9
unparseable rate:  7.5%
```

Feature capture used CUDA and captured 92 parseable candidates.

---

### 4.1 Reasoning transfer result

```text
REASONING_TRANSFER_VERDICT = GOOD
```

Overall:

```text
random top1 baseline: 0.440
best overall: CODE 24_L4 / AntisymLinear
top1    1.000
pairwise 1.000
cycle   0.000
```

Best HH:

```text
HH 36_mean / AntisymLinear
top1    0.960
pairwise 0.986
cycle   0.000
```

Per dataset:

```text
ARC-Challenge best: CODE 24_L4 / AntisymLinear, top1/pairwise 1.000
OpenBookQA best:   HH 36_L4 / AntisymLinear, top1/pairwise 1.000
```

Interpretation:

> Multiple-choice reasoning is now a promising third objective eval domain. The pilot is small and likely somewhat easy after filtering, so it is not final. But it is strong enough to promote reasoning into the active evaluation set.

Important nuance:

- Reasoning seems to show strong transfer from both HH-trained and code-trained heads.
- Code wins ARC-Challenge in this pilot.
- HH wins OpenBookQA in this pilot.
- This further supports the generalist + specialist routing idea.

---

## 5. Loop/layer diagnostics across domains

Loop geometry was available for 4 domains:

```text
HH
clean GSM8K
strict-clean code
reasoning
```

Recurring pattern:

```text
layers 24/36: more loop-converged
layer 47: more L1-to-L4 spread
```

Layer 47 L1/L4 cosine:

```text
HH:                0.735
clean GSM8K:       0.631
strict-clean code: 0.723
reasoning:         0.687
```

Interpretation:

> The heterogeneous tap interface is reinforced. Layers 24/36 are practical single-state control taps. Layer 47 remains more trajectory-sensitive and sometimes benefits from fused/all-loop readouts.

This matches earlier findings:

- 24/36 are converged checkpoints.
- 47 is domain-dependent and more trajectory-spread.

---

## 6. Caveats from the broad audit

Do not ignore these.

### 6.1 CODE_STRICT_CLEAN_OLD6 baseline anomaly

The standalone `CODE_STRICT_CLEAN_OLD6` row in the fixed-config matrix has an anomalous random baseline of `0.0`.

Do not lean on that row.

Use instead:

```text
expanded strict-clean OLD6/NEW10/ALL16 breakdown
```

from the expanded strict-clean comparison.

### 6.2 NaN aggregate fields

Some aggregate average fields in the fixed-config report show `nan` because HH pair-style rows do not have tournament cycle values.

Use:

```text
per-domain rows
verdicts
fixed-config tables
```

not raw aggregate averages.

### 6.3 Reasoning pilot may be easy

Reasoning transfer was extremely strong, but the pilot is small and may be easy-after-filtering.

Do not claim reasoning is solved.

The correct claim:

> Reasoning is promising enough to add as a third objective evaluation domain.

---

## 7. Updated architecture interpretation

### 7.1 Generalist + specialists

The BG Phase-1 plan should now include:

```text
general / preference head:
  HH-trained selector

code specialist:
  code-specific selector, currently 36_L4 / AntisymLinear for strict-clean code

reasoning specialist:
  provisional; needs harder reasoning validation
```

The controller should not be one universal scalar.

It should route between:

```text
general selector
code specialist
reasoning specialist candidate
NoNorm easy-prune selector
AntisymLinear hard-comparison selector
```

---

### 7.2 NoNorm vs AntisymLinear

Refined rule after the broad audit:

```text
NoNorm:
  useful scalar-readable/transitive ranking in objective domains;
  strong for easier/runnable diagnostic selection;
  sometimes best on HH/config subsets but not universal.

AntisymLinear:
  safer for hard relational or near-miss distinctions;
  best current strict-clean code specialist: code-trained 36_L4 / AntisymLinear;
  strong on reasoning pilot.
```

Do not choose one globally.

Use both as complementary head families.

---

### 7.3 Layer policy

Current empirical layer pattern:

```text
24/36:
  converged intermediate control states;
  often strong for objective domains;
  36_L4 and 36_mean repeatedly matter.

47:
  trajectory-spread / late semantic state;
  important for HH and some broad/general comparisons;
  all-loop/concat configs may help in trajectory-spread domains.
```

The current best strict-clean code head is:

```text
code-trained 36_L4 / AntisymLinear
```

The current best HH_200 head in the fixed-config audit is:

```text
HH-trained 47_concat_L1_L4 / AntisymLinearNoNorm
```

Reasoning pilot best overall:

```text
code-trained 24_L4 / AntisymLinear
```

---

## 8. Updated project state

Current conclusions:

```text
1. Branch-selection signal exists and generalizes enough for local planning.
2. Clean GSM8K and patched code support preliminary objective-domain transfer.
3. Strict-clean code signal is present in states.
4. Code-specific projection training substantially improves strict-clean code selection.
5. HH-trained heads remain best for HH preference pairs.
6. Reasoning is a promising third objective evaluation domain.
7. No single projection is stable enough to be universal.
8. BG Phase 1 should use general + specialist heads.
```

Current bottleneck has shifted.

Previous bottleneck:

```text
Can we produce strict-clean same-task contrasts?
```

Current bottleneck:

```text
How should the BG controller route among general and specialist heads,
and how do we validate this on harder reasoning / broader strict-clean sets?
```

Strict-clean task screening remains useful, but it is no longer the only active line.

---

## 9. Recommended next work

### 9.1 Immediate spec update

Update the BG Phase-1 plan to explicitly include:

```text
1. HH/general head.
2. Code specialist head.
3. Reasoning specialist candidate.
4. NoNorm easy-prune option.
5. AntisymLinear hard-comparison option.
6. Domain routing when domain is known.
7. Disagreement / margin-based defer when general and specialist disagree.
```

### 9.2 Harder reasoning validation

Because reasoning pilot was GOOD but probably easy, the next empirical domain work should be a harder reasoning pilot.

Options:

```text
ARC-Challenge hard subset
OpenBookQA filtered for low agreement
CommonsenseQA
BBH only if parser reliable
generated reasoning distractor branches
```

Goal:

```text
Determine whether reasoning transfer remains GOOD when branches are harder and less answer-format trivial.
```

### 9.3 Controller-policy simulator

Now that the generalist/specialist result is established, a controller-policy simulator becomes more timely.

It should test policies such as:

```text
general_only
specialist_only
domain_routed
NoNorm_easy_prune_then_Antisym_hard_select
disagreement_defer
margin_threshold_defer
general_specialist_vote
```

But this should come after the spec update and maybe one harder reasoning validation pass.

### 9.4 Do not run these next

Do not prioritize:

```text
full MATH gate-scale
GRU default architecture
published 5M evaluator retraining
blind code generation expansion
more balancing of old one-sided tasks
```

---

## 10. One-line handoff

The BG/tap project now has a broad cross-domain result: HH-trained heads remain best for HH preference, code-trained heads are materially better for strict-clean code, and reasoning looks promising as a third objective domain. The architecture should move to **general + domain-specialist tiny heads**, with AntisymLinear/NoNorm kept as complementary readouts and controller policy now becoming the next design problem.

