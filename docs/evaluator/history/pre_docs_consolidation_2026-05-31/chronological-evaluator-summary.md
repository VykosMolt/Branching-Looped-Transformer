# BG Evaluator Chronological Summary for a DSA Presentation

Date: 2026-05-27

This document summarizes the evaluator/BG work in chronological order for a reader who is comfortable with algorithms and systems, but not necessarily with modern machine-learning internals.

Source scope: this is based on the Markdown documentation under `docs/evaluator`, including the current root docs, `history/`, and `raw_archive_2026-05-18/`. The raw archive preserves exact earlier documents; the root docs are the current canonical version.

## Executive Summary

The project is trying to build a small controller that can look at the hidden states of a looped language model and choose which answer branch is more likely to succeed.

In DSA terms, think of a language model as generating a search tree of possible continuations. Each partial answer is a node or path prefix. The BG evaluator is a learned ranking heuristic that tries to compare two paths and say which one is more promising. The key question is not "can the model produce text?" It is "can we read useful branch-quality information from the model's internal states, early enough to guide search?"

The main result so far is:

- The signal is real and readable from frozen Ouro-RLTT internal states.
- The useful object is mostly pairwise comparison, not absolute scoring.
- Small exact-antisymmetric linear heads often work better than larger GRU-style temporal aggregation for this project.
- The signal transfers across several objective domains: code, reasoning, science, and clean arithmetic/GSM8K.
- The transformer-native trajectory-prediction sweep was strong: BG scores over partial prefixes predicted which branches would later finish correctly.
- Direct hidden-state steering on the frozen backbone is closed under the tested safe-alpha methods: hooks write cleanly, perturbations propagate, but no static direction or adapter produced reliable held-out free-generation transfer.
- Same-prefix hidden-origin branches can be generated with the hook fallback and can produce different downstream outcomes, but frozen BG taps do not yet select those hidden-origin branches better than random.

Most important current result:

| Item | Value |
| --- | --- |
| Strong readout verdict | `BG_TRAJECTORY_PREDICTION_VERDICT = STRONG` |
| Steering closure verdict | `BG_SEQUENCE_LEVEL_ADAPTER_VERDICT = NO_FROZEN_BACKBONE_WRITE_PATH` |
| Frozen-backbone status | `FROZEN_BACKBONE_INFERENCE_STEERING_STATUS = CLOSED_UNDER_TESTED_METHODS` |
| Hidden-origin status | `PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_V3 = STILL_DATA_LIMITED` |
| Best trajectory-prediction cell | `MIX_CODE_REASONING / 36_mean / AntisymLinear`, reasoning, 256 tokens |
| Best cell lift / pairwise accuracy | +0.1625 top-1 lift / 0.8537 pairwise |
| Recommended next step | consolidate Phase 1/1.5 and design Phase 2 training-time integration |

The practical interpretation is: BG is now more than a finished-answer reranker. It can read useful information from partial trajectories. But on the frozen Ouro-RLTT backbone, the readable direction has not become a reliable write/control handle under the tested methods. The next serious work is Phase 2 training-time integration or a branch-native hidden-origin evaluator, not another simple static steering direction.

## Plain-Language Mental Model

### The components

Ouro-RLTT:

A local looped transformer model. It runs several internal loop iterations while producing hidden states. The active local path is:

`/home/moloch/ouro_project/models/ouro_rltt_local`

Candidate or branch:

One possible answer attempt. For example, four branches can be generated for the same reasoning question, and the evaluator tries to rank them.

Prefix:

The beginning of a generated branch, for example the first 32, 64, 128, or 256 tokens. A prefix is a partial path, not a final answer.

Continuation:

A generation that starts from a prefix and continues to a final answer. This lets us label whether the prefix was "solution-bearing."

BG evaluator:

A small branch-selection module. "BG" is used in the basal-ganglia-inspired sense: a controller that chooses among competing actions or branches. In this project it is not a full model. It is a small head reading hidden states.

Tap:

A place where hidden states are read from the transformer. The current taps are layers 24, 36, and 47, across loop iterations L1-L4.

Feature tensor:

For each prompt/candidate pair, the live transformer feature extractor returns:

```text
[layers=3, loops=4, hidden=2048]
```

This means: three layers, four loop states per layer, and 2048 hidden dimensions.

Head:

A small scoring function placed on top of captured features. The active families are:

- `AntisymLinear`: LayerNorm over the difference, then a bias-free linear score.
- `AntisymLinearNoNorm`: direct bias-free linear score over the difference.

Pairwise comparator:

The evaluator compares candidate A to candidate B. If the score is positive, A is preferred; if negative, B is preferred. This is closer to a comparison function in a sorting/ranking algorithm than to a standalone classifier.

Oracle success:

For a task with several generated branches, oracle success asks: "Did any branch solve it?" It is an upper bound on what any selector could achieve. If no branch is correct, no reranker can select a correct branch.

Top-1 lift:

How much better BG's top choice is than random selection. A lift of +0.10 means BG is 10 percentage points better than choosing a branch uniformly at random.

Reachability:

Whether the generator produced at least one viable answer candidate. Low reachability means the generator failed, not necessarily the evaluator.

## Chronological Story

## 1. Starting Point: A Pairwise Evaluator Signal Exists

The early work started from a published-style result: a pairwise evaluator could read preference information from hidden states and distinguish chosen vs rejected HH-RLHF answers.

The first question was narrow:

> Where inside the looped transformer does this evaluator signal live?

Early probes tested loop positions, layers, normalization, and whether a GRU over the loop axis was necessary. The goal was not yet a full controller. It was to understand the signal's location and shape.

Important findings:

- Pairwise comparison was the right primitive.
- The signal was not simply "candidate A has high absolute quality." It was relational: candidate A compared to candidate B.
- Some apparent high scores were artifacts. In particular, masked-zero variants produced degenerate constant-sign behavior and were demoted.
- The GRU was not clearly helping. Simple mean pooling or single-state readouts often performed as well or better.
- The project learned to check score distributions and flip tests, not just raw accuracy.

Why this mattered:

From an algorithms viewpoint, the evaluator needed to be a reliable comparison operator. If a comparator gives high apparent accuracy only because of positional bias or constant-sign output, it is not usable for sorting, ranking, or branch selection.

## 2. May 11-12: Locus Probes and the First Architecture Lessons

The raw locus memo records several probe passes over loop and layer states.

What was tested:

- Which loop iteration carried the strongest signal.
- Whether applying model normalization multiple times improved readout.
- Whether feeding all loop states into a GRU improved over simpler pooling.
- Whether masked/zeroed inputs were valid or artifacts.
- Whether the same configuration worked on HH and math-like distributions.

What was found:

- Proper normalization made a large difference.
- Loop states all carried useful information once normalization artifacts were controlled.
- The GRU was mildly counterproductive in several settings.
- Masked-zero "100%" style findings were not trusted after flip tests.
- The pairwise signal survived across several probes, but the exact best loop/layer framing changed as artifacts were removed.

Practical lesson:

The comparator had to be exact, antisymmetric, and checked under candidate swap. If A beats B, then B should lose to A. This led directly to the later `AntisymLinear` family.

## 3. May 14: Reframing From "Best Layer" to "Branch-Selection Controller"

The next synthesis reframed the goal.

Old framing:

> Can a pairwise evaluator read preference from Ouro loop states?

New framing:

> Can we build a relational, pairwise, multi-tap branch-selection controller, validated on externally labeled tournaments over Ouro-RLTT-generated branches?

This was a major pivot. HH preference accuracy became a diagnostic, not the main success criterion. The main target became branch selection over generated attempts.

Why this mattered:

In an algorithmic search setting, a heuristic is only useful if it improves decisions among actual generated branches. A high score on a static preference dataset is not enough.

The Phase-2 gate was therefore changed to generated-branch tournament performance:

- Generate multiple branches for the same problem.
- Label branches using an external verifier or answer key.
- Score/rank branches using BG.
- Measure whether BG picks better branches than random or baseline policies.

## 4. May 14: L1/L4 and Layer-Geometry Work

Early architecture assumed that layers 24, 36, and 47 might all benefit from the same L1/L4 loop fusion strategy.

That assumption was tested.

What was found:

- Layer 47 has meaningful loop spread. It can benefit from fused or all-loop readouts.
- Layers 24 and 36 are more loop-converged. Their L1-L4 states are very similar.
- Therefore a uniform "always fuse L1 and L4" interface was not right.

This produced the heterogeneous tap interface:

| Layer | Role | Typical readout |
| --- | --- | --- |
| 24 | early/mid converged checkpoint | usually single-state, often L4 or mean |
| 36 | mid converged checkpoint | usually single-state, often L4 or mean |
| 47 | late/trajectory-spread checkpoint | L4, L1+L4, or all loops |

Key layer-47 L1/L4 cosine examples:

| Domain | L1-L4 cosine at layer 47 |
| --- | ---: |
| HH | 0.735 |
| clean GSM8K | 0.631 |
| strict-clean code | 0.723 |
| reasoning MCQ | 0.687 |

Interpretation:

Layers 24/36 behave more like stable checkpoints. Layer 47 preserves more trajectory history. This matters because a controller reading early/mid/late evidence should not force one input format everywhere.

## 5. May 15: AntisymLinear Pivot

The next important design change was the move away from a large published-style evaluator head toward tiny exact-antisymmetric heads.

The core scoring rule became:

```text
score(A, B) = linear(feature(A) - feature(B))
```

with an optional LayerNorm on the difference:

```text
AntisymLinear:
    LayerNorm(left - right) -> Linear(no bias)

AntisymLinearNoNorm:
    Linear(left - right)
```

Why this was attractive:

- It enforces pairwise antisymmetry by construction.
- It is small and easy to audit.
- It removes the need for swap augmentation and symmetry penalties.
- It directly matches the comparator role.

Important interpretation:

`NoNorm` often behaves like a scalar utility readout. It can work well when correctness is relatively transitive.

`AntisymLinear` keeps a more relational geometry. It can help on hard near-miss comparisons where two candidates look locally similar but one has a subtle bug.

## 6. May 15-16: Math and GSM8K Hygiene

The project then tested whether HH-trained evaluator signal transferred to math-like generated branches.

The first broad MATH-style pilot was demoted because of a serious confound:

- Generated math attempts were often too verbose.
- Many were truncated by token limits.
- Truncation correlated with correctness.
- A selector could accidentally learn "not truncated" rather than "mathematically correct."

So the project moved to cleaner GSM8K/simple arithmetic settings.

Clean GSM8K findings:

- Clean GSM8K became the safer math proxy.
- HH-trained tiny linear heads transferred well enough for local planning.
- GRU control was weak.
- Exact numeric parsing/verifying gave cleaner labels.

Representative current-state result:

| Dataset | Finding |
| --- | --- |
| clean GSM8K expanded | GOOD linear transfer |
| GRU control | weak |
| full MATH gate-scale | deferred |

Why this mattered:

This prevented an invalid conclusion. The project avoided treating a dataset artifact as a reasoning signal.

## 7. May 16-17: Code Branch Pilots and Harness Fixes

Code became the hardest domain.

The first code pilot had an all-correct or insufficient-mixed-pair collapse, so it could not teach the evaluator much. A later code v2 pilot showed signal but exposed harness and sanitization issues.

The code harness was patched and retested.

Important stages:

- Initial code pilot: too few useful tournaments.
- Code v2: diagnostic signal but dirty harness.
- Patched v2-mini: runnable diagnostic with GOOD transfer.
- Strict-clean code: focus on correct vs near-miss code where both candidates are plausible.

Strict-clean code was especially important because it tests high-local-similarity discrimination:

- Both candidates may compile.
- Both may look similar.
- One subtle difference makes one fail.

Key expanded strict-clean result:

| Head family | Best config | Top-1 | Pairwise |
| --- | --- | ---: | ---: |
| HH-trained | `47_mean / AntisymLinear` | 0.750 | 0.600 |
| Code-trained | `36_L4 / AntisymLinear` | 0.875 | 0.833 |

Interpretation:

The code signal exists in the hidden states, but HH-trained preference heads are not the best projection for hard code near-misses. Code-specific or mixed-objective projections are materially better.

## 8. May 17: General vs Specialist Heads

After code, GSM8K, reasoning, and HH tests, the project asked:

> Do we need one universal head, one specialist per domain, or a smaller set of projection types?

Several transfer tests were run.

HH inverse-transfer result:

- HH-trained heads remained best for HH preference.
- Code-trained heads had some general signal but did not replace HH heads.
- On HH all-200, HH-trained best was about 0.855, while code-trained best was about 0.535.
- Random 20-pair splits showed code heads can transfer somewhat, but not enough to replace HH.

This led to the contrast-type principle:

| Contrast type | Meaning | Preferred head |
| --- | --- | --- |
| Preference/coherence | Which answer is more helpful, coherent, or preference-aligned? | HH general |
| Objective branch correctness | Which branch is more likely correct under an answer key/test/verifier? | mixed objective |
| High-local-similarity near-miss | Same task, very similar candidates, subtle difference | code specialist backup |

This is more precise than "one head per domain." The important distinction is the type of comparison, not just whether the task is code, science, or reasoning.

## 9. May 17: Reasoning and Science Transfer

Reasoning and science became the main reachable non-code objective domains.

Reasoning:

- Initial reasoning pilot used ARC-Challenge and OpenBookQA style multiple-choice tasks.
- Results were very strong, but the first pilot may have been too easy after filtering.
- Later natural-distractor and trace-style validations still supported positive transfer.
- Conclusion: no separate reasoning specialist was justified yet.

Science:

- Science tasks were tested across biology, chemistry, medicine, and general science.
- Results were heterogeneous by subdomain.
- Chemistry sometimes behaved more HH-like.
- Biology/general science often favored objective or code-trained projections.
- Conclusion: science does not require a new specialist yet, but subdomain behavior is real.

Why this mattered:

Reasoning and science were reachable under direct Ouro generation, unlike hard direct code. They became the proper headline domains for transformer-native trajectory tests.

## 10. May 17-18: Mixed-Domain Heads and Locked v8.1 Controller Policy

The project then consolidated many small heads into a deployable Phase-1 controller.

The locked head set:

| Role | Head | Training source | Layer config | Architecture |
| --- | --- | --- | --- | --- |
| HH general | `hh_general` | HH-trained | `47_concat_L1_L4` | `AntisymLinearNoNorm` |
| Objective mixed | `objective_mixed` | MIX_CODE_REASONING | `36_L4` | `AntisymLinearNoNorm` |
| Code specialist backup | `code_specialist_backup` | code-trained | `36_L4` | `AntisymLinear` |

The conservative routing rule:

```text
HH / preference / unknown -> hh_general
code / reasoning / science / math / GSM8K / objective -> objective_mixed
code_specialist_backup -> retained for diagnostics and fallback, not primary route
```

The controller-policy simulator found high complementarity:

- HH and objective heads make different errors.
- A vote/ensemble mode can extract more signal, but calibration is not yet trustworthy.
- Conservative domain routing was locked as default.
- Experimental vote mode exists for validation, not production default.

Important v8.1 conclusion:

The Phase-1 architecture is a two-production-head controller with one specialist backup.

## 11. May 18: Read-Only BG Controller Implementation

The locked policy was implemented in:

```text
src/evaluator/bg_controller.py
```

Verdicts:

| Verdict | Result |
| --- | --- |
| `BG_CONTROLLER_ARTIFACT_VERDICT` | READY |
| `BG_CONTROLLER_IMPLEMENTATION_VERDICT` | READY |
| `BG_CONTROLLER_UNIT_TEST_VERDICT` | PASS |
| `BG_CONTROLLER_REPLAY_VERDICT` | PASS |

The controller replay matched the cached simulator exactly in conservative mode:

```text
max pairwise difference = 0.000000
```

Interpretation:

The engineering implementation reproduced the intended routing and scoring behavior. This made BG usable as a read-only branch-selection component over already-captured features.

## 12. May 18: Live Transformer Feature Capture

Next, BG needed to work on live model outputs rather than only cached artifacts.

Feature extraction was implemented in:

```text
src/evaluator/bg_transformer_features.py
```

The feature extractor takes:

```text
Prompt:
{prompt}

Candidate:
{candidate}
```

and returns:

```text
[3, 4, 2048]
```

It does not include:

- answer key
- correctness label
- verifier output
- BG score
- selected index

Verdicts:

| Verdict | Result |
| --- | --- |
| `BG_TRANSFORMER_CAPTURE_INSPECTION_VERDICT` | READY |
| `BG_TRANSFORMER_BEST_OF_N_SMOKE_VERDICT` | PASS |
| `BG_TRANSFORMER_INTEGRATION_VERDICT` | PASS |
| `BG_TRANSFORMER_UNIT_TEST_VERDICT` | PASS |

Interpretation:

BG became connected to live Ouro-RLTT hidden states in a read-only way. This is the transformer-native path.

## 13. May 18: Steering and Partial-Trajectory Routing Suite

The next question was whether BG could help during generation, not just after final answers.

The steering/routing suite tested:

- branch pool generation
- reachability
- partial trajectory routing
- compute allocation
- soft hidden-state steering
- text-prefix branch selection

Important verdicts:

| Verdict | Result |
| --- | --- |
| `BG_PARTIAL_ROUTING_VERDICT` | NEUTRAL |
| `BG_COMPUTE_ALLOCATION_VERDICT` | INSUFFICIENT |
| `BG_SOFT_STEERING_VERDICT` | STABLE_NO_EFFECT |
| `BG_LATENT_BRANCH_SELECTION_VERDICT` | HELPS |
| `OVERALL_BG_STEERING_VERDICT` | NEUTRAL |

Partial routing metrics:

| Metric | Value |
| --- | ---: |
| oracle success rate | 0.793 |
| random top-1 expected | 0.560 |
| BG top-1 | 0.603 |
| BG top-1 lift | +0.043 |
| random top-2 expected | 0.698 |
| BG top-2 | 0.741 |
| BG top-2 lift | +0.043 |

Interpretation:

The partial-routing signal was directionally positive but below the predeclared +0.05 threshold. That is why the verdict was NEUTRAL, not positive.

Reachability result:

| Domain | Oracle success |
| --- | ---: |
| code | 0.000 |
| devil code | 0.000 |
| reasoning | 1.000 |
| science | 1.000 |
| GSM8K | 1.000 |

This separated generator failure from evaluator failure. BG cannot select a correct code branch if direct Ouro produced no correct code branch.

Soft steering:

Tiny activation nudges were stable, but had no clean directional effect. This means inference-time steering was not validated yet.

Text-prefix branch selection:

A small pilot showed `HELPS`, but n was only 8 tasks, so it was promising rather than conclusive.

## 14. May 18: Wrapper Candidate Export and Wrapper-Matched BG

The local-agent wrapper can generate tool-using code candidates. The project exposed these candidates via an opt-in trace interface:

```text
src/local_agent/candidate_export.py
src/local_agent/candidate_capture.py
```

The wrapper export does not change wrapper behavior by default. It only records candidates for later analysis.

The wrapper-matched experiment compared:

- wrapper final choice
- BG conservative selection over the same candidate pool
- random candidate
- stage heuristic
- oracle candidate

Headline wrapper-matched results:

| Metric | Value |
| --- | ---: |
| matched evaluable tasks | 15 |
| wrapper final pass rate | 0.400 |
| BG conservative pass rate | 0.400 |
| random expected pass rate | 0.400 |
| stage heuristic pass rate | 0.400 |
| oracle reachability | 0.400 |

Verdicts:

| Verdict | Result |
| --- | --- |
| `WRAPPER_MATCHED_BG_VERDICT` | NEUTRAL |
| `BG_VS_RANDOM_VERDICT` | NEUTRAL |
| `BG_VS_STAGE_HEURISTIC_VERDICT` | NEUTRAL |
| `WRAPPER_ORACLE_GAP_VERDICT` | SMALL |

Interpretation:

The wrapper produced more reachable code candidates than direct Ouro, but BG did not improve over wrapper final, random expectation, or the stage heuristic on this small matched set. The wrapper oracle gap was small, so there were few missed correct branches for BG to rescue.

Architectural correction:

The wrapper is now treated as a candidate-generator diagnostic, not the architectural target. The main target is transformer-native BG:

```text
Ouro-RLTT loop states -> BG taps -> trajectory ranking/routing -> possible future steering/training
```

## 15. May 18: Stage-1 BG Trajectory Prediction Sweep

This is the strongest recent result and the current architectural milestone.

Question:

> At which prefix length, layer/config, head, and domain does BG score best predict whether a partial trajectory will lead to a successful final answer?

This was read-only:

- no steering
- no activation modification
- no weight modification
- no tokenizer/checkpoint modification
- no training
- no wrapper use

Task suite:

| Domain | Tasks |
| --- | ---: |
| reasoning MCQ | 20 |
| science MCQ | 20 |
| GSM8K/simple arithmetic | 20 |

Generation:

- K = 4 branches per task.
- Generate a full 256-token trajectory once per branch.
- Slice prefixes at 32, 64, 128, and 256 tokens.
- Continue every prefix once to a final answer.
- Evaluate final answers using external parsers/verifiers.

This creates labels for prefixes after the fact. The evaluator labels are not fed back into BG scoring.

Verdicts:

| Stage | Verdict |
| --- | --- |
| preflight | READY |
| task suite | READY |
| partial generation | READY |
| continuation/evaluation | READY |
| feature capture | READY |
| prefix scoring | READY |
| predictive analysis | STRONG |

Best predictive cell:

| Field | Value |
| --- | --- |
| domain | reasoning |
| prefix length | 256 |
| head/config | `mixed::MIX_CODE_REASONING::36_mean::AntisymLinear` |
| top-1 success | 0.850 |
| random top-1 expected | 0.6875 |
| top-1 lift | +0.1625 |
| top-2 success | 0.900 |
| oracle success | 0.900 |
| pairwise predictive accuracy | 0.8537 |
| AUC margin-success | 0.7265 |
| n tasks | 20 |

Interpretation:

When BG ranks reasoning prefixes at 256 tokens using the 36_mean AntisymLinear head, it is much better than random at identifying branches that will finish correctly. The top-2 result reaches the oracle in that cell, meaning the correct branch is captured in BG's top two whenever any branch is correct.

## 16. Latest Heatmap and Trend Results

The full trajectory-prediction heatmap shows that the result is broad, not a single isolated lucky cell.

Best cell by domain and prefix:

| domain | prefix | best config | top-1 lift | pairwise accuracy | oracle |
| --- | ---: | --- | ---: | ---: | ---: |
| GSM8K | 32 | `47_concat_L1_L4 / AntisymLinear` | +0.125 | 0.778 | 1.000 |
| GSM8K | 64 | `24_L4 / AntisymLinear` | +0.162 | 0.724 | 1.000 |
| GSM8K | 128 | `47_L4 / AntisymLinearNoNorm` | +0.150 | 0.765 | 1.000 |
| GSM8K | 256 | `36_L4 / AntisymLinear` | +0.200 | 0.750 | 1.000 |
| reasoning | 32 | `36_L4 / AntisymLinear` | +0.200 | 0.550 | 0.900 |
| reasoning | 64 | `36_mean / AntisymLinearNoNorm` | +0.262 | 0.638 | 1.000 |
| reasoning | 128 | `47_concat_L1_L4 / AntisymLinear` | +0.238 | 0.655 | 0.950 |
| reasoning | 256 | `36_mean / AntisymLinear` | +0.162 | 0.854 | 0.900 |
| science | 32 | `47_L4 / AntisymLinear` | +0.188 | 0.811 | 0.900 |
| science | 64 | `36_mean / AntisymLinearNoNorm` | +0.125 | 0.750 | 0.900 |
| science | 128 | `47_concat_all_loops / AntisymLinear` | +0.025 | 0.647 | 0.950 |
| science | 256 | `36_L4 / AntisymLinearNoNorm` | +0.138 | 0.545 | 0.900 |

Prefix-length trend:

The trend is not monotonic.

| Domain | Top-1 peak | Pairwise peak | Interpretation |
| --- | --- | --- | --- |
| GSM8K | 256 tokens | 32 tokens | arithmetic often needs late completion, but early ranking signal exists |
| reasoning | 64 tokens | 256 tokens | early commitment is visible, but pairwise ordering is strongest late |
| science | 32 tokens | 32 tokens | useful signal appears very early |

Domain breakdown:

| Domain | Strong cells | Cells with top-1 lift >= +0.10 | Cells with pairwise >= 0.65 |
| --- | ---: | ---: | ---: |
| GSM8K | 90 | 52 | 77 |
| reasoning | 152 | 137 | 78 |
| science | 126 | 111 | 72 |

Config trend:

| Config | Strong cells | Cells with top-1 lift >= +0.10 |
| --- | ---: | ---: |
| `36_L4` | 91 | 79 |
| `36_mean` | 72 | 62 |
| `24_L4` | 69 | 55 |
| `47_concat_L1_L4` | 55 | 38 |
| `47_concat_all_loops` | 46 | 37 |
| `47_L4` | 35 | 29 |

Architecture trend:

| Architecture | Strong cells | Cells with top-1 lift >= +0.10 |
| --- | ---: | ---: |
| `AntisymLinear` | 179 | 146 |
| `AntisymLinearNoNorm` | 189 | 154 |

Operating envelope:

| Metric | Count |
| --- | ---: |
| strong cells | 368 |
| cells with top-1 lift >= +0.10 | 300 |
| cells with top-2 lift >= +0.10 | 40 |
| cells with pairwise accuracy >= 0.65 | 227 |

Interpretation:

The high-performing region is wide. Reasoning is the best current headline domain, but science and GSM8K also show strong cells. `36_mean` is important, but it is not the only useful configuration. `NoNorm` wins more total strong cells, while the single best predictive cell uses `AntisymLinear`.

## 17. May 18-22: Steering Closure on the Frozen Backbone

After the trajectory-prediction sweep, the project tested whether BG-readable directions could also act as write/control directions during generation.

The validated write surface:

- decoder-layer hooks are mechanically clean,
- zero-alpha hook runs match no-hook generation,
- perturbation size scales predictably,
- perturbations propagate to later hidden states and logits,
- no broad CUDA/NaN/Inf instability appeared under the safe-alpha envelope.

The steering result:

| Test | Verdict |
| --- | --- |
| raw NoNorm readout steering | `UNSIGNED_EFFECT` |
| empirical success directions | `EMPIRICAL_UNSIGNED_ONLY` |
| RMS-calibrated steering | `RMS_UNSIGNED_ONLY` |
| causal gradient probe | `GRADIENT_NO_BETTER_THAN_RANDOM` |
| inference-time steering summary | `UNSIGNED_ONLY` |
| Phase 2 requirement | `TRAINING_REQUIRED` |

The central geometry result is:

```text
readout geometry != empirical-success geometry != local logit-control geometry
```

Concrete cosine checks were near-orthogonal:

| Direction pair | Cosine |
| --- | ---: |
| adapter proxy vs raw NoNorm | -0.000553 |
| adapter proxy vs empirical mean diff | -0.004294 |
| raw NoNorm vs empirical mean diff | +0.101002 |

Interpretation:

BG can read which trajectories are promising, and hooks can write into the model, but the directions that read success are not the directions that reliably cause success. This is the core reason frozen-backbone steering failed under the tested methods.

## 18. May 22-23: Causal and Sequence-Level Adapter Tests

The project then tested whether a learned adapter could supply the missing write path while keeping Ouro and BG heads frozen.

Teacher-forced causal adapter:

| Verdict | Result |
| --- | --- |
| `BG_CAUSAL_ADAPTER_TEACHER_FORCED_VERDICT` | `ADAPTER_IMPROVES_LOGIT_MARGIN` |
| `BG_CAUSAL_ADAPTER_FREE_GEN_VERDICT` | `TEACHER_FORCED_ONLY` |
| `BG_CAUSAL_ADAPTER_VERDICT` | `LOCAL_LOGIT_CONTROL_ONLY` |

This adapter learned weak local answer-logit control under teacher forcing, but that did not transfer into reliable free-generation task improvement.

Sequence-level adapter:

| Verdict | Result |
| --- | --- |
| `BG_SEQUENCE_ADAPTER_TRAINING_VERDICT` | `SEQUENCE_REWARD_IMPROVES` |
| `BG_SEQUENCE_ADAPTER_HELDOUT_VERDICT` | `NO_ADAPTER_SPECIFIC_TRANSFER` |
| `BG_SEQUENCE_ADAPTER_VS_RANDOM_VERDICT` | `WORSE_THAN_RANDOM` |
| `BG_SEQUENCE_LEVEL_ADAPTER_VERDICT` | `NO_FROZEN_BACKBONE_WRITE_PATH` |
| `FROZEN_BACKBONE_INFERENCE_STEERING_STATUS` | `CLOSED_UNDER_TESTED_METHODS` |

The stopping rule applies under the tested safe-alpha envelope (`alpha <= 0.02`) and tested optimizers. This does not prove that no future training method can ever steer Ouro, but it closes the simple frozen-backbone inference-time path.

## 19. May 27: Same-Prefix Hidden-Origin Branch Work

The hidden-origin branch suite separated real same-prefix hidden-state branches from ordinary text/candidate branch artifacts.

The method used hook-hidden-origin branches:

- same prompt and token prefix,
- perturbation at an internal layer/loop,
- branch continuation through Ouro,
- downstream MCQ outcome scoring.

True autoregressive fork/carry remains blocked because there is no validated API for resuming generation from a copied internal hidden/cache state.

Key verdicts:

| Item | Verdict |
| --- | --- |
| hidden-branch feasibility | `HOOK_HIDDEN_ORIGIN_READY` |
| branch generation | `HOOK_HIDDEN_ORIGIN_BRANCHES_GENERATED` |
| latent persistence | `LATENT_BRANCHES_PERSIST_TO_47` |
| outcome dataset | `READY` |
| frozen tap selection | `NO_HIDDEN_BRANCH_SELECTION_SIGNAL` |
| L30/L42 gates | `NEEDS_STRONGER_BRANCH_GENERATOR` |
| adaptive thresholds | `TOPK_SUFFICIENT` |
| Phase 2 hidden-branch readiness | `NEEDS_BETTER_BRANCH_EVALUATOR` |

Important measurements:

| Metric | Value |
| --- | ---: |
| branch records | 112 |
| branch groups | 28 |
| safe-alpha groups | 24 |
| behaviorally diverse safe groups | 6 |
| diversity rate | 0.25 |
| random top-1 success | 0.625 |
| frozen BG pairwise winner top-1 | 0.583 |

Interpretation:

Safe same-prefix hidden-origin perturbations can create measurable geometric and behavioral branch variation. But the canonical frozen BG taps that work for text/candidate branches are not calibrated selectors for these same-prefix hidden-origin branches.

## 20. May 27: Hidden-Origin Tap Training and Diversity Expansion

Three hidden-origin selector passes followed.

V1 hidden-origin taps:

- Status: `DATA_LIMITED`
- Pair labels came only from same-group downstream reward comparisons.
- Tie rate was very high (`0.940`).
- Best heldout behaviorally diverse result looked good, but only over 3 groups, so it was not load-bearing.

V2 diversity expansion:

- Status: `WEAK`
- Dataset became `SMALL_BUT_USABLE`.
- Primary pairs increased to 170.
- Tie rate remained high (`0.932`).
- Best new policy was a weak selector on behaviorally diverse heldout groups.
- Layer result: `CONCAT_REQUIRED`.

V3 diversity and selector reevaluation:

- Status: `STILL_DATA_LIMITED`
- Diversity improved.
- Non-random directions helped.
- Primary pairs increased to 536.
- Primary tie rate improved but remained high (`0.881`).
- Best available selector is `v3_hidden_origin_tap`.
- Heldout evidence remained too small for selector-readiness claims.

Current hidden-origin conclusion:

Hidden-origin branch generation is real but the selector is not ready. Continue targeted data expansion using the V3 recipe before claiming a Phase 2 hidden-origin branch evaluator.

## 21. What We Learned Overall

### 1. BG is primarily a comparator

The stable unit is not "score one answer in isolation." It is "compare candidate A to candidate B." This is why exact antisymmetry became central.

### 2. Small heads are enough to read substantial signal

Tiny linear heads can extract useful branch-selection information from frozen hidden states. This is important because it suggests the information is already organized in the model's representation, not only learned by a large evaluator.

### 3. Layer choice matters

Layer 36 is repeatedly strong for objective branch selection. Layer 47 is important for HH/preference and trajectory-spread readouts. Layer 24 also contributes in several objective settings.

### 4. Domain reachability matters as much as ranking

If no generated branch solves the problem, a selector cannot create a solution. This explains why direct code/devil experiments were not good headline tests: direct Ouro did not produce correct code branches there.

### 5. Objective domains share signal

Code, reasoning, science, and clean arithmetic are not identical, but they share enough objective branch-quality structure for mixed heads to work.

### 6. HH preference geometry is distinct

HH preference/coherence requires a different projection than objective correctness. A universal objective head does not replace the HH general head.

### 7. Wrapper integration is useful but not the main architecture

The wrapper can expose candidates and improve code reachability, but the architectural target is reading and eventually influencing transformer trajectories directly.

### 8. Read-only trajectory prediction now works

The Stage-1 trajectory sweep shows BG can predict from partial prefixes which branches are likely to complete correctly. This is the key bridge from finished-answer reranking to trajectory-level control.

### 9. Frozen-backbone inference steering is closed under tested methods

The system can read useful directions and can be perturbed mechanically, but static directions, empirical directions, teacher-forced adapters, and sequence-level adapters did not produce reliable held-out free-generation transfer under the tested safe-alpha envelope. The remaining steering path is no longer "try another static direction"; it is Phase 2 training-time integration or a better branch-native evaluator.

## 22. What Did Not Work or Was Demoted

GRU as default:

The GRU was repeatedly weak or mildly counterproductive relative to simpler heads. It remains an ablation/control, not the default.

Masked-zero variants:

These produced degenerate results and were treated as artifacts.

Full MATH generation:

Full MATH-style generation was confounded by verbosity/truncation and deferred. Clean GSM8K is the current math proxy.

Direct Ouro code/devil generation:

Reachability was too low. No selector can rescue a candidate pool with zero correct branches.

Wrapper-matched BG as deployment target:

Neutral result. Useful diagnostic, not the architectural target.

Tiny soft steering:

Stable but no effect. It did not show output improvement or reliable directional movement.

Static and empirical hidden-state steering:

Hooks worked, but raw NoNorm, mean-diff, whitened, logistic, RMS-calibrated, and gradient directions produced unsigned or random-comparable effects rather than reliable signed control.

Frozen-backbone adapters:

The causal adapter improved teacher-forced answer-logit margins but did not transfer to free generation. The sequence-level adapter improved training reward but did not beat non-adapter/random heldout baselines.

Frozen BG taps on hidden-origin branches:

Same-prefix hidden-origin branches were real, but the old frozen taps did not select good hidden-origin branches better than random.

Compute allocation arm:

Insufficient due to time/cap constraints. It needs a cheaper cached design or longer dedicated run.

## 23. What Worked

Exact pairwise framing:

The comparator framing survived many probes and matched the branch-selection problem.

Heterogeneous taps:

The layers should not all be treated the same. Layers 24/36 are converged checkpoints; layer 47 carries more loop-spread information.

AntisymLinear family:

Small exact-antisymmetric heads are effective and auditable.

Clean GSM8K:

Provided a valid arithmetic proxy without the truncation confound.

Strict-clean code specialist:

Showed that code near-miss signal exists in states and benefits from code-specific projection.

Reasoning/science:

Reachable and positive, making them suitable headline domains for transformer-native tests.

Controller implementation:

The locked conservative controller was implemented and replay-matched.

Live read-only feature capture:

The transformer-native path now works without modifying model weights or activations.

Trajectory prediction:

The latest Stage-1 sweep delivered a strong positive result.

Layer-hook write surface:

The intervention surface is mechanically valid: zero-alpha equivalence passed, perturbations propagate to later hidden states and logits, and safe-alpha runs were stable.

Same-prefix hidden-origin branch generation:

Hook-origin branches can persist geometrically to layer 47 and sometimes change downstream outcomes, which makes them a valid Phase 2 object even though the selector is not ready.

## 24. Current Architecture State

The current Phase-1 architecture:

```text
Ouro-RLTT generated candidates
    -> read hidden states at layers 24, 36, 47 across loops L1-L4
    -> build [3, 4, 2048] features
    -> score candidate pairs with locked BG heads
    -> route by domain in conservative mode
```

Locked conservative heads:

| Route | Head |
| --- | --- |
| HH/preference/unknown | `hh_general`: HH-trained, `47_concat_L1_L4`, `AntisymLinearNoNorm` |
| objective domains | `objective_mixed`: MIX_CODE_REASONING, `36_L4`, `AntisymLinearNoNorm` |
| backup/diagnostic | `code_specialist_backup`: code-trained, `36_L4`, `AntisymLinear` |

Important nuance:

The best trajectory-prediction cell used `36_mean / AntisymLinear`, not the locked finished-candidate objective route `36_L4 / AntisymLinearNoNorm`. This suggests trajectory-level operation may need its own optimized configuration, even though v8.1 remains correct for finished-candidate selection.

## 25. Why This Matters Algorithmically

For a DSA audience, the project can be framed as search control.

Language generation creates a branching process. At every step, there are possible continuations. If we can score partial paths, we can:

- choose which branches to continue,
- allocate compute to promising branches,
- prune bad branches,
- compare candidate solutions,
- eventually steer the model toward better internal states.

The BG evaluator is analogous to a learned heuristic function for branch selection. It is not guaranteed admissible like A* heuristics, and it is not a formal proof of correctness. It is an empirical ranking heuristic. The experiments measure whether that heuristic correlates with final success.

The current readout result says:

> Yes, for reasoning/science/GSM8K partial trajectories, the heuristic has predictive power.

The steering result says:

> On the frozen backbone, under tested safe-alpha methods, this heuristic only ranks branches after they exist.

The next algorithmic question is how to train a write path or branch-native hidden-origin evaluator so the branch process itself becomes controllable.

## 26. Recommended Next Work

The targeted Stage-2 steering-sensitivity probe has been run. Its follow-ups closed the simple frozen-backbone inference-time steering path under the tested methods.

Current recommended work:

1. Consolidate Phase 1 and Phase 1.5 as read-only BG selection, routing, trajectory prediction, and branch allocation.
2. Design Phase 2 as training-time integration, not another static frozen-backbone steering sweep.
3. Keep the readout controller intact: `hh_general`, `objective_mixed`, and `code_specialist_backup` remain useful for selection.
4. Expand hidden-origin branch outcome data using the V3 recipe before claiming selector readiness.
5. Build a branch-native evaluator/calibration target for same-prefix hidden-origin branches.
6. Implement branch-aware Ouro forward/cache support if true hidden-state fork/carry is required.

Backbone regularization is not required for the read-only selector. It becomes relevant only if the goal is action steering: aligning readout and production geometry during training, or training an equivalent write-path adapter as part of Phase 2.

## 27. One-Slide Version

Problem:

We want a controller that chooses among possible reasoning branches generated by a looped transformer.

Method:

Read hidden states from layers 24, 36, and 47; compare candidate branches with small antisymmetric heads; evaluate choices using external answer keys, tests, and verifiers.

Main findings:

- Pairwise comparison is the right primitive.
- Tiny linear heads read useful branch-quality signal.
- Objective branch signal transfers across reasoning, science, GSM8K, and code.
- HH preference signal is distinct from objective correctness signal.
- Direct code generation is reachability-limited, so non-code domains are the right transformer-native headline.
- Wrapper-matched BG was neutral and is now diagnostic only.
- The latest Stage-1 trajectory sweep is strong: BG predicts from partial prefixes which branches will finish correctly.
- Frozen-backbone inference-time steering is closed under tested safe-alpha methods.
- Same-prefix hidden-origin branches are real but need a better selector/evaluator.

Current best result:

Reasoning at 256 tokens, `MIX_CODE_REASONING / 36_mean / AntisymLinear`: top-1 lift +0.1625 over random, pairwise accuracy 0.8537, oracle success 0.900.

Next question:

How do we train a write path or hidden-origin branch evaluator so BG can move from read-only selection to actual trajectory control?

## 28. Files to Cite

Current canonical docs:

- `docs/evaluator/chronological-evaluator-summary.md`
- `docs/evaluator/current-state.md`
- `docs/evaluator/domain-transfer-ledger.md`
- `docs/evaluator/evaluator-locus-summary.md`
- `docs/evaluator/tap-interface.md`
- `docs/evaluator/controller-usage.md`
- `docs/evaluator/transformer-integration.md`
- `docs/evaluator/steering-and-routing-suite.md`
- `docs/evaluator/trajectory-prediction-sweep.md`
- `docs/evaluator/steering-findings.md`
- `docs/evaluator/sequence-level-adapter.md`
- `docs/evaluator/hidden-state-branch-generation.md`
- `docs/evaluator/hidden-origin-diversity-v3.md`
- `docs/evaluator/wrapper-matched-bg-selection.md`
- `docs/evaluator/phase1-routing-policy-locked.md`
- `docs/evaluator/phase1-progress-addendum.md`

Recent reports:

- `artifacts/reports/probes/bg_trajectory_prediction_2026-05-18/summary.md`
- `artifacts/reports/probes/bg_trajectory_prediction_2026-05-18/predictive_power.md`
- `artifacts/reports/probes/bg_trajectory_prediction_2026-05-18/stage2_recommendation.md`
- `artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/summary.md`
- `artifacts/reports/probes/bg_causal_intervention_adapter_2026-05-18/summary.md`
- `artifacts/reports/probes/bg_hidden_state_branch_generation_2026-05-18/summary.md`
- `artifacts/reports/probes/bg_hidden_origin_diversity_v3_2026-05-18/summary.md`

Historical docs:

- `docs/evaluator/history/`
- `docs/evaluator/raw_archive_2026-05-18/`

## 29. Final Interpretation

The work has moved through three levels.

First, it established that hidden states contain pairwise preference/correctness information. This was the evaluator-locus stage.

Second, it turned that signal into a small, auditable branch-selection controller over generated candidates. This was the BG controller and domain-transfer stage.

Third, it showed that the signal appears before final answers are complete. This was the trajectory-prediction stage.

Fourth, it tested whether that readable signal is already a write/control handle on the frozen backbone. It is not, under the tested methods.

The project is therefore past "can we read anything useful?" and also past the simple version of "can we steer it at inference time?"

The next stage is Phase 2: training-time integration or a branch-native hidden-origin evaluator that makes the control target explicit.
