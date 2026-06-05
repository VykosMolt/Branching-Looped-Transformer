# Post-v10 synthesis addendum — clean GSM8K transfer, GRU control, and code-branch next step

**Date:** 2026-05-16  
**Status:** Internal working addendum. Extends `post_v10_synthesis_2026-05-15_v4.md` with everything new after the AntisymLinear pivot and math gate-prep discussion.  
**Scope:** Clean GSM8K regeneration, validity probe, HH→GSM8K transfer, GRU control, and the local-agent code wrapper notes that motivate the next code-branch tournament probe.  
**High-level update:** The old MATH/GSM8K pilot was truncation-confounded. A clean GSM8K-only local-budget benchmark fixes that confound and shows preliminary HH-trained tap transfer. Temporal GRU control does not beat the simpler exact-antisymmetric linear heads. The next serious domain should be **code branch tournaments**, using the local-agent wrapper and unit-test labels.

---

## 0. Current operational status

The project should **not** treat the old `TRANSFER_POOR` result as final. That result was measured on a pilot set later shown to be truncation-confounded. The clean GSM8K follow-up produced a usable local-budget benchmark and a positive preliminary transfer result.

The current best operational summary is:

```text
Old pilot validity:         TRUNCATION_CONFOUNDED
Clean GSM8K expanded set:   CLEAN_MINIMUM
Linear transfer:            GOOD
GRU control:                GRU_WEAK
Recommended next:           code_branch_dataset_or_scale_clean_gsm8k_once_more
```

My interpretation:

> HH-trained tiny taps show real preliminary transfer to clean GSM8K branch selection. The earlier negative result was mostly an artifact of evaluating on truncated / easy-after-filtering branches. GRU-style temporal aggregation is above random but does not justify its extra complexity on the clean expanded GSM8K set.

---

## 1. Data-validity probe: the old math pilot was truncation-confounded

A post-hoc validity probe inspected the original `math_branch_tournaments_rltt.json` pilot instead of assuming it was a clean wrong-reasoning dataset.

### Verdict

```text
DATA_VALIDITY = TRUNCATION_CONFOUNDED
```

### Important details

The JSON had 33 tournaments and candidate fields:

```text
attempt_index
source
completion
candidate_text
extracted_answer
is_correct
```

It did **not** store `output_tokens` or `truncated`, so completion lengths were recovered with the local Ouro-RLTT tokenizer. No model was loaded, no GPU was used, and no generation was run.

### Truncation by source and correctness

| source | correctness | n | trunc_rate | extractable_answer |
|---|---:|---:|---:|---:|
| GSM8K | correct | 49 | 55.1% | 100.0% |
| GSM8K | incorrect | 55 | 80.0% | 100.0% |
| MATH | correct | 10 | 30.0% | 100.0% |
| MATH | incorrect | 18 | 11.1% | 100.0% |

Parser health was fine: `has_extractable_answer = 100%` for every source/correctness cell, and stored vs recomputed parser outputs had 0 mismatches.

The problem was **truncation and difficulty composition**, especially in GSM8K:

```text
overall incorrect truncation = 63.0%
overall incorrect near-miss fraction = 27.4%
```

### Near-miss / nonsense split

| source | incorrect | near_miss | nonsense | near_miss_frac |
|---|---:|---:|---:|---:|
| GSM8K | 55 | 9 | 46 | 16.4% |
| MATH | 18 | 11 | 7 | 61.1% |

The MATH subset was surprisingly coherent, but tiny. The kept pilot set was GSM8K-heavy, and its incorrect GSM8K branches were mostly truncated.

### Interpretation

The old math-trained heads scoring 1.000 on the pilot may have learned a “complete/coherent answer beats truncated/incomplete answer” direction. The old HH→math `TRANSFER_POOR` result was therefore not clean evidence against HH-trained transfer.

The old result should be renamed mentally as:

```text
TRANSFER_POOR_ON_CONFOUNDED_PILOT
```

or simply:

```text
TRANSFER_UNRESOLVED
```

---

## 2. Fast HH→math transfer inventory result

The inventory probe found enough artifacts for a small transfer test, but the transfer result used the confounded pilot above.

### Inventory verdict

```text
INVENTORY_VERDICT = SUFFICIENT
```

### HH captures

| file | layers | loops | examples | usable |
|---|---:|---:|---:|---|
| `hh_layer_states_200_rltt.pt` | 24, 36, 47 | 4 | 200 | Y |
| `hh_layer_states_200_thinking.pt` | 24, 36, 47 | 4 | 200 | Y |
| `hh_loop_states_200_rltt.pt` | NA | 4 | 200 | N |
| `hh_loop_states_200_thinking.pt` | NA | 4 | 200 | N |

### Tournament data

| file | n | source mix | features | labels | usable |
|---|---:|---|---|---|---|
| `math_branch_tap_features.pt` | 33 | 26 GSM8K / 7 MATH | Y | Y | Y |
| `math_branch_tournaments_rltt.json` | 33 | 26 GSM8K / 7 MATH | N | Y | N |

### Transfer result on confounded pilot

```text
TRANSFER_VERDICT = TRANSFER_POOR
```

But after the validity probe, this verdict is not decisive.

Comparison used the saved math-head eval split, `n = 8`.

| config | HH AntisymLinear | math-trained | delta | verdict | HH NoNorm |
|---|---:|---:|---:|---|---:|
| 24_L1 | 0.500 | 1.000 | 0.500 | TRANSFER_POOR | 0.500 |
| 24_L4 | 0.500 | 1.000 | 0.500 | TRANSFER_POOR | 0.375 |
| 24_mean | 0.500 | 1.000 | 0.500 | TRANSFER_POOR | 0.375 |
| 36_L1 | 0.375 | 1.000 | 0.625 | TRANSFER_POOR | 0.375 |
| 36_L4 | 0.375 | 1.000 | 0.625 | TRANSFER_POOR | 0.625 |
| 36_mean | 0.375 | 1.000 | 0.625 | TRANSFER_POOR | 0.625 |
| 47_L4 | 0.250 | NA | NA | NO_MATH_BASELINE | 0.250 |
| 47_mean | 0.125 | NA | NA | NO_MATH_BASELINE | 0.125 |
| 47_concat_L1_L4 | 0.500 | 1.000 | 0.500 | TRANSFER_POOR | 0.125 |
| 47_concat_all_loops | 0.250 | NA | NA | NO_MATH_BASELINE | 0.125 |

Best HH-trained all-tournament config was `47_concat_L1_L4`:

```text
combined top-1 = 0.424
GSM8K top-1    = 0.462
MATH top-1     = 0.286
```

NoNorm cycle rate was 0.000 for every config.

### Interpretation after validity probe

The negative transfer result was measured on a confounded pilot. It should not be used to conclude that HH-trained taps fail to transfer to math or GSM8K.

---

## 3. Clean GSM8K extreme microbenchmark

A stricter GSM8K-only regeneration was run to answer a narrower question:

> Can we get a tiny, clean, parseable, non-truncated GSM8K tournament set under the real local budget, and do HH-trained taps show any preliminary transfer on that clean set?

### Verdict

```text
CLEAN_GSM8K_VERDICT = CLEAN
CLEAN_TRANSFER_VERDICT = PRELIM_GOOD
```

### Generation summary

The micro run produced:

```text
5 kept clean GSM8K tournaments
15 prompts
168 generated attempts
```

Kept branches were:

```text
100% parseable
100% FINAL ANSWER present
0 kept incorrect branches hit max_new_tokens
```

### Transfer result

```text
random top-1 baseline = 0.633
best HH-trained row = 47_L4 / AntisymLinearNoNorm
top-1 = 1.000
pairwise = 0.917
cycle = 0.000
```

Several AntisymLinear rows also cleared the preliminary-good threshold, including:

```text
36_L1
36_L4
36_mean
47_concat_all_loops
```

### Interpretation

The microbenchmark rescued the HH→GSM8K transfer question from the old confounded negative, but `n=5` was too small for a real conclusion. It justified an expanded clean GSM8K run.

---

## 4. Expanded clean GSM8K + HH-trained transfer

The expanded clean GSM8K run scaled the microbenchmark while retaining the strict answer-wrapper regime.

### Verdicts

```text
EXPANDED_CLEAN_GSM8K_VERDICT = CLEAN_MINIMUM
EXPANDED_LINEAR_TRANSFER_VERDICT = GOOD
```

### Generation summary

| metric | value |
|---|---:|
| prompts processed | 80 |
| attempts generated | 892 |
| clean tournaments | 28 |
| kept branches | 79 |
| random top-1 baseline | 0.563 |
| near-miss fraction | 0.559 |
| clean verdict | CLEAN_MINIMUM |

The run reached the minimum acceptable threshold but not the full 30 target:

```text
target clean tournaments = 30
actual clean tournaments = 28
```

The set is clean:

```text
kept branches parseable = 100%
FINAL ANSWER present = 100%
kept incorrect branches hit max_new_tokens = 0
```

### Best transfer rows

| family | config | top1 | pairwise | cycle |
|---|---|---:|---:|---:|
| AntisymLinear | 36_L4 | 0.786 | 0.704 | 0.000 |
| NoNorm | 24_L4 | 0.750 | 0.741 | 0.000 |

### Interpretation

This is the main positive result since v4.

The earlier `TRANSFER_POOR` does **not** reproduce once the branch dataset is clean. HH-trained exact-antisymmetric linear heads show a meaningful preliminary transfer signal on generated GSM8K branches:

```text
AntisymLinear 36_L4 top-1:
  0.786 vs random baseline 0.563
  +22.3 pp over random

NoNorm 24_L4 top-1:
  0.750 vs random baseline 0.563
  +18.7 pp over random
```

Pairwise performance is also meaningful:

```text
AntisymLinear 36_L4 pairwise = 0.704
NoNorm 24_L4 pairwise        = 0.741
```

The result supports the current project hypothesis:

> Ouro-RLTT organizes branch-selection signal in hidden states strongly enough that tiny exact-antisymmetric taps trained on HH can transfer to clean generated GSM8K branch selection.

It does **not** prove transfer to all math, MATH, code, or reasoning. It is a clean GSM8K result.

---

## 5. NoNorm competitiveness and scalar-readable GSM8K signal

The NoNorm baseline is no longer just a control. It was competitive and had the best pairwise score:

```text
24_L4 / NoNorm:
  top-1   = 0.750
  pairwise = 0.741
  cycle   = 0.000
```

NoNorm is transitive by construction:

```python
score(a, b) = w · (a - b) = u(a) - u(b)
```

This suggests that clean GSM8K branch correctness may be at least partly readable as a scalar projection at intermediate layers. That does **not** refute the broader relational framing, especially for HH preference. It does suggest that in some objective-label domains, particularly GSM8K, a scalar-readable utility direction may be sufficient.

The practical conclusion:

```text
Keep both AntisymLinear and NoNorm in branch-selection evaluations.
If NoNorm remains competitive, use it as the simpler transitive controller for that tap/domain.
If AntisymLinear wins meaningfully, the LayerNorm relational comparator is adding value.
```

---

## 6. GRU control result

A small temporal GRU control was trained/evaluated on the same expanded clean GSM8K set.

### Verdict

```text
GRU_CONTROL_VERDICT = GRU_WEAK
```

### Best GRU result

| family | config | top1 | pairwise | cycle |
|---|---|---:|---:|---:|
| GRU | gru24_sequence | 0.714 | 0.667 | 0.000 |

### Why `GRU_WEAK` does not mean “GRU useless”

The GRU selected correct branches above random. It was not degenerate.

The weak verdict means:

> The GRU failed the control’s purpose: showing that temporal loop aggregation adds value over the much simpler exact-antisymmetric linear heads.

Concrete comparison:

| metric | best linear / NoNorm | best GRU | readout |
|---|---:|---:|---|
| top-1 | 0.786 AntisymLinear 36_L4 | 0.714 gru24_sequence | GRU is ~7.1 pp lower |
| pairwise | 0.741 NoNorm 24_L4 | 0.667 gru24_sequence | GRU is ~7.4 pp lower |
| cycle | 0.000 | 0.000 | no issue |
| bias_to_signal | N/A | 0.011 best GRU | bias was low |
| HH holdout acc | linear heads varied | best GRU 0.550, top GRU row 0.500 | below 0.60 threshold |

The weak verdict came from two facts:

1. It underperformed the simpler heads by more than 5 pp.
2. It did not learn the HH held-out split strongly.

This was **not** a raw-score bias failure:

```text
GRU bias_to_signal = 0.005 to 0.013
centered/raw metrics similar
```

### Interpretation

On the expanded clean GSM8K set, temporal loop aggregation did not justify its added capacity. The branch-selection signal was already readable by simpler AntisymLinear/NoNorm heads, especially:

```text
36_L4 / AntisymLinear
24_L4 / NoNorm
```

This does not rule out GRUs elsewhere. It means GRU should remain a **capacity/temporal control**, not a Phase-1 default.

---

## 7. Current architecture implications

The current best read is:

```text
Phase-1 default head family:
  AntisymLinear + AntisymLinearNoNorm

GRU:
  control/escalation only

Published 5M evaluator:
  historical / comparison control

MATH large generation:
  not a current blocker under local compute

Code:
  next serious generated-branch domain
```

### Tap implications

The clean GSM8K run strengthens the 24/36 intermediate tap story:

```text
AntisymLinear winner: 36_L4
NoNorm winner:        24_L4
GRU winner:           gru24_sequence
```

Layer 47 is not the clear winner on clean GSM8K. The useful transfer signal is appearing at intermediate layers.

This supports the heterogeneous BG design:

| tap | current role |
|---|---|
| 24 | early/intermediate scalar-readable or relational branch signal |
| 36 | strong intermediate relational branch signal |
| 47 | useful late baseline, not privileged for clean GSM8K |

### Operational implication

The next benchmark should not ask whether a large temporal head can rescue the system. The next benchmark should ask whether the same simple tap family transfers to **code**, where labels are cleaner and near-miss structure is easier to define.

---

## 8. Local-agent code wrapper inventory

The code-domain branch dataset should use the local-agent wrapper rather than raw unconstrained text generation, because the wrapper already turns local generations into executable code and test evidence.

### Core code-generation path

| Stage | File | What it does |
|---|---|---|
| Prompt contract | `src/local_agent/ouro_prompts.py:26` | Tells the model to emit complete runnable code, no TODOs/scaffolds/prose inside code. |
| Tool syntax | `src/local_agent/ouro_prompts.py:36` | Defines `[Action]: python` / `[Input]: <complete source>` format. |
| Parse action | `src/local_agent/ouro_agent_improved.py:476` | Extracts JSON or bracketed `[Action]` / `[Input]` tool calls. |
| Sanitize code | `src/local_agent/ouro_policies.py:285` | Pulls code out of markdown fences, removes late FINAL ANSWER, observations, verifier text, etc. |
| Forced tool-tested coding | `src/local_agent/ouro_agent_improved.py:2202` | Injects policy requiring Python-tool execution and runtime checks before final answer. |
| Code prefill + stops | `src/local_agent/ouro_agent_improved.py:6450` | Uses code-action assistant prefill, clamps UT steps, and stops at observation/system/verifier/final markers. |
| Runtime-check gate | `src/local_agent/ouro_agent_improved.py:6522` | Rejects definition-only or check-free hard-code tool inputs. |
| Execute + final grounded output | `src/local_agent/ouro_agent_improved.py:6535` | Executes tool, then builds final answer from observed working code. |

The direct-code route has a parallel wrapper:

```text
src/local_agent/ouro_direct.py:118
```

It normalizes incomplete fences, forces `FINAL ANSWER:\n```python`, verifies code blocks, and can repair incomplete coding answers.

### Relevant local-agent settings

| Setting | Default | File | Purpose |
|---|---:|---|---|
| `DIRECT_MAX_TOKENS` | 1536 | `src/local_agent/ouro_config.py:296` | Direct answer budget. |
| `DIRECT_TEMPERATURE` | 0.0 | `src/local_agent/ouro_config.py:297` | Deterministic direct code. |
| `CODE_FAST_PREFILL` | True | `src/local_agent/ouro_config.py:303` | Enables fast code prefill behavior. |
| `CODE_FIRST_PASS_TOKENS` | 512 | `src/local_agent/ouro_config.py:304` | First-pass code budget. |
| `CODE_REPAIR_TOKENS` | 384 | `src/local_agent/ouro_config.py:305` | Repair budget for incomplete code. |
| `CODE_FIRST_PASS_UT_STEPS` | 2 | `src/local_agent/ouro_config.py:306` | Reasoning/token step budget for first code action. |
| `CODE_REPAIR_UT_STEPS` | 2 | `src/local_agent/ouro_config.py:307` | UT budget for repair. |
| `REQUIRE_HARD_CODE_VERIFICATION` | True | `src/local_agent/ouro_config.py:310` | Requires verification before accepting hard-code answers. |
| `HARD_CODE_TOOL_MODE` | True | `src/local_agent/ouro_config.py:311` | Forces hard coding tasks through executable tool evidence. |
| `HARD_CODE_AGENT_MAX_TOKENS` | 1536 | `src/local_agent/ouro_config.py:313` | Max agent budget for hard-code mode. |
| `HARD_CODE_FIRST_ACTION_TOKENS` | 768 | `src/local_agent/ouro_config.py:314` | First tool-call budget for hard-code mode. |
| `PYTHON_TOOL_TIMEOUT_SEC` | 15.0 | `src/local_agent/ouro_config.py:320` | Python execution timeout. |
| `HARD_CODE_TASK_WALLCLOCK_SEC` | 120.0 | `src/local_agent/ouro_config.py:322` | Overall hard-code wallclock cap. |

Backend detail:

```text
src/local_agent/ouro_backend.py:787
sampled generations use top_p = 0.7 whenever temperature > 0
```

### Effective code prompt

There is no single standalone `CODE_SYSTEM_PROMPT`.

The effective code prompt is assembled from:

```text
SYSTEM_PROMPT
+ OURO_SOLVER_POSTURE
+ direct code-first additions OR hard-code tool-tested policy context
```

Base system prompt is in:

```text
src/local_agent/ouro_prompts.py:32
```

The code-relevant posture is in:

```text
src/local_agent/ouro_prompts.py:3
```

Direct code-first route appends in:

```text
src/local_agent/ouro_direct.py:625
```

High-risk algorithmic code policy context is added in:

```text
src/local_agent/ouro_agent_improved.py:2202
```

### Implication for branch dataset construction

The code dataset should not use “raw model text → hope it is code” generation.

It should collect candidates from the local-agent pipeline:

```text
prompt contract
code/action prefill
tool-call parser
sanitizer
runtime-check gate
execution evidence
final grounded code
```

But for branch-selection evaluation, we must be careful not to let the wrapper erase all failures through repair. Candidate branches should include multiple stages:

```text
raw direct code block
first tool-call code
post-repair code, if any
final grounded code
```

Unit tests provide labels. The wrapper can produce both good and bad branches; the branch selector should learn to select the branch that passes the tests.

---

## 9. Code-domain branch dataset: next recommended experiment

### Why code next

Code is a better next generated-branch domain than MATH under current constraints:

1. External labels are cheap and objective: unit tests.
2. Near-miss is operationally clean:
   - near-miss = compiles/runs and passes some but not all tests;
   - nonsense = syntax error, runtime error before tests, or passes 0 tests.
3. The local-agent wrapper already forces executable evidence.
4. The model’s verbosity is less harmful because code tasks end in code/tool execution rather than long math explanations.
5. The clean GSM8K run already showed HH-trained taps can transfer to a generated reasoning branch task. The next question is whether that extends to code.

### Proposed code pilot size

Start small:

```text
10 HumanEval/MBPP-style tasks
4 candidates per task
target: 5+ mixed tournaments
max wallclock: local-budget, not overnight
```

Scale only if the pilot produces mixed pass/fail tournaments.

### Candidate labels

Use unit tests, not evaluator scores.

Candidate classes:

| class | definition |
|---|---|
| correct | passes all tests |
| near-miss | compiles/runs and passes at least one but not all tests |
| nonsense | syntax error, import/runtime failure before tests, or passes zero tests |

### Tournament keep rule

A code tournament is kept if:

```text
at least one correct branch
at least one incorrect branch
at least two total clean runnable branches OR one correct branch plus one failure branch with explicit failure mode
```

For transfer evaluation, report both:

```text
strict clean tournaments: only runnable candidates
mixed diagnostic tournaments: include syntax/runtime failures as nonsense
```

### Evaluation plan

Train or reuse HH-trained heads from the existing 200 HH capture:

```text
AntisymLinear
AntisymLinearNoNorm
```

Configs:

```text
24_L1
24_L4
24_mean
36_L1
36_L4
36_mean
47_L4
47_mean
47_concat_L1_L4
47_concat_all_loops
```

Metrics:

```text
random_top1_baseline
top-1 tournament accuracy
pairwise accuracy
Condorcet winner rate
cycle rate
margin distribution
pass-rate breakdown
near-miss/nonsense breakdown
per-task family breakdown if available
```

GRU is not a default. It should only be rerun as a control if the linear heads fail or if code trajectories expose a specific temporal aggregation need.

---

## 10. Revised experiment backlog

The old “MATH gate-prep is next” statement is superseded locally by the clean-GSM8K and code-domain evidence.

Current recommended backlog:

| # | Experiment | Status / priority |
|---:|---|---|
| 1 | Data-validity probe on old math pilot | Done; old pilot truncation-confounded |
| 2 | HH→math transfer on confounded pilot | Done; downgraded to unresolved/confounded |
| 3 | Clean GSM8K microbenchmark | Done; preliminary transfer good |
| 4 | Expanded clean GSM8K + GRU control | Done; linear transfer good, GRU weak |
| 5 | Code branch tournament pilot | **Next** |
| 6 | Optional clean GSM8K scale to 50 tournaments | Useful if code is delayed |
| 7 | Larger pooled HH capture | Only if needed after code/GSM8K evidence |
| 8 | MATH stress-test under strict wrapper | Qualitative / small only |
| 9 | Full MATH gate-prep | Deferred under current compute constraints |
| 10 | HH Experiment 2 Redux | Deferred; not driving current work |
| 11 | Phase-1 BG training | After code / broader generated-branch evidence |

---

## 11. Current working conclusion

The clean GSM8K run changes the project state substantially.

Before it, there were two plausible readings:

```text
A. HH-trained tiny taps do not transfer to generated math branches.
B. The old transfer benchmark was invalid because branches were truncated.
```

The clean run supports B.

Current best claim:

> HH-trained exact-antisymmetric linear taps transfer preliminarily to clean generated GSM8K branch selection. The strongest signal appears at intermediate single-state taps (`36_L4` for AntisymLinear, `24_L4` for NoNorm). Temporal GRU aggregation is above random but does not outperform the simpler heads.

Practical next step:

> Build a code-branch tournament pilot using the existing local-agent code wrapper, unit-test labels, and the same HH-trained AntisymLinear/NoNorm transfer evaluation.

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
- full reports: `artifacts/reports/probes/current_bg_transfer_state_2026-05-17_summary.md`, `docs/evaluator/current-state.md`, `artifacts/reports/probes/scalar_vs_relational_current_state_2026-05-17.md`, `artifacts/reports/probes/strict_clean_task_screening_protocol_2026-05-17.md`.
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
