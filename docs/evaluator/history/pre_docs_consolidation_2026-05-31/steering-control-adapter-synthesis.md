# BG Steering / Control / Adapter Synthesis — 2026-05-18

## Executive summary

The current evidence supports a clean architectural split:

- **BG readout / selection / trajectory prediction is validated.**
- **Layer-hook writing into Ouro is mechanically valid and stable.**
- **Perturbations propagate to later hidden states and logits.**
- **Static inference-time steering directions do not provide reliable signed control.**
- **A tiny teacher-forced causal adapter learned weak local logit control, but did not show detectable adapter-specific free-generation transfer.**
- **The adapter free-generation eval was low-resolution: most heldout tasks were invariant, and the apparent lift came from one task that also responded to raw/random perturbations.**
- **The strongest mechanistic geometry result is now three-way orthogonality: readout geometry, empirical-success geometry, and local logit-control geometry are distinct.**
- **The next credible path is one serious sequence-level / black-box adapter optimization attempt, then Phase 2 backbone or adapter regularization if that fails.**

The core finding remains:

> **Readout geometry is not production/write geometry.**

More precisely, the current evidence suggests:

> **Readout geometry, empirical-success geometry, and local logit-control geometry are distinct.**

BG taps can read trajectory quality. They can select among candidate branches. Ouro can be perturbed through hooks. But the tested directions and tiny local adapter do not yet provide a dependable action-steering write path.

---

## Current top-level verdicts

```text
BG_TRAJECTORY_PREDICTION_VERDICT = STRONG
BG_LAYERHOOK_MECHANICAL_VERDICT = READY
BG_LAYERHOOK_SIGNED_CAUSAL_VERDICT = UNSIGNED_EFFECT
BG_EMPIRICAL_DIRECTION_CAUSAL_VERDICT = EMPIRICAL_UNSIGNED_ONLY
BG_RMS_STEERING_VERDICT = RMS_UNSIGNED_ONLY
BG_PROPAGATION_VERDICT = PROPAGATES_TO_LATER_STATES
BG_PROPAGATION_DECAY_PROFILE = SURVIVES_32_TOKENS
BG_LOGIT_EFFECT_VERDICT = LOGITS_SHIFT_DIRECTIONALLY
BG_TEXT_PREFIX_EXPANSION_VERDICT = WEAK_POSITIVE
BG_CAUSAL_GRADIENT_VERDICT = GRADIENT_NO_BETTER_THAN_RANDOM
BG_INFERENCE_TIME_STEERING_VERDICT = UNSIGNED_ONLY
BG_BRANCH_ALLOCATION_VERDICT = PROMISING
BG_PHASE2_REQUIREMENT_VERDICT = TRAINING_REQUIRED
BG_CAUSAL_ADAPTER_VERDICT = LOCAL_LOGIT_CONTROL_ONLY
```

---

## Core mechanistic geometry finding

The steering/adaptation stack now supports a stronger statement than simply “readout is not steering.”

There are at least three distinct geometries:

```text
1. Readout geometry
   Raw NoNorm / BG predictive directions that rank or score trajectory quality.

2. Empirical-success geometry
   Mean-difference, whitened-difference, and logistic success directions derived from successful vs failed branches.

3. Local logit-control geometry
   The trained causal adapter's average/proxy delta direction, which weakly improves teacher-forced answer logits.
```

The measured cosines are near-orthogonal:

```text
cos(adapter proxy, raw NoNorm) = -0.000553
cos(adapter proxy, empirical mean diff) = -0.004294
cos(raw NoNorm, empirical mean diff) = +0.101002
```

Interpretation:

> **Readout, empirical-success separation, and local logit-control occupy distinct directions. None yet corresponds to a reliable trajectory-level action-steering direction.**

This is one of the central paper-worthy findings. It generalizes the earlier readout/production mismatch into a broader **readout ≠ empirical-success ≠ local-logit-control** geometry split.

---

## Stage 1 — trajectory prediction

Stage 1 established that BG/tap heads strongly predict partial trajectory success.

```text
BG_TRAJECTORY_PREDICTION_VERDICT = STRONG
GENERATOR_REACHABILITY_LIMITED = false
```

Broad operating envelope:

```text
strong cells total: 368
cells with top1 lift >= +0.10: 300
cells with pairwise accuracy >= 0.65: 227
```

Best diagnostic peak:

```text
domain: reasoning
prefix length: 256
head/config: mixed::MIX_CODE_REASONING::36_mean::AntisymLinear
top1 lift: +0.1625
pairwise accuracy: 0.8537
oracle success: 0.900
```

Domain and config trends:

```text
reasoning strong cells: 152
science strong cells: 126
GSM8K strong cells: 90

36_L4 strong cells: 91
36_mean strong cells: 72
24_L4 strong cells: 69

AntisymLinearNoNorm strong cells: 189
AntisymLinear strong cells: 179
```

Interpretation:

> BG-readable trajectory viability is broad across objective domains and multiple layer/config regions. This is readout power, not yet steering power.

---

## Stage 2 layer-hook steering — raw NoNorm direction

Layer-hook steering was mechanically validated.

```text
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
```

Mechanical validation:

```text
zero-alpha hook equivalence: pass
hook loop identity: current_ut
cache intervention mode: disabled
CUDA errors: 0
NaN/Inf activations: 0
safety_status: OK for all rows
```

Expected hook accounting for 128-token continuations:

```text
single_loop_L1:
  hook_forward_call_count = 512
  hook_modifications = 128

single_loop_L4:
  hook_forward_call_count = 512
  hook_modifications = 128

multi_loop_uniform:
  hook_forward_call_count = 512
  hook_modifications = 512

multi_loop_decayed:
  hook_forward_call_count = 512
  hook_modifications = 512
```

Interpretation:

> The hook writes into the model correctly. Early and multi-loop intervention schedules produce stronger aggregate movement than L4-only. But the raw NoNorm direction is not a reliable signed steering vector.

---

## Empirical steering direction follow-up

Empirical directions were tested to determine whether the issue was the raw NoNorm vector specifically.

Verdicts:

```text
BG_EMPIRICAL_STEERING_PREFLIGHT_VERDICT = READY
BG_EMPIRICAL_DIRECTION_BUILD_VERDICT = READY
BG_EMPIRICAL_STEERING_TASKS_VERDICT = READY
BG_EMPIRICAL_STEERING_SWEEP_VERDICT = READY
BG_EMPIRICAL_DIRECTION_CAUSAL_VERDICT = EMPIRICAL_UNSIGNED_ONLY
BG_EMPIRICAL_VS_RAW_VERDICT = EMPIRICAL_BEATS_RAW
BG_STEERING_DIRECTION_GEOMETRY_VERDICT = RAW_READOUT_NOT_PRODUCTION_DIRECTION
BG_EMPIRICAL_STEERING_STABILITY_VERDICT = DESTABILIZING  # caused by one outlier row, not broad instability
BG_EMPIRICAL_FINAL_LIFT_VERDICT = NEGATIVE_LIFT
BG_TINY_STEERING_ADAPTER_VERDICT = NO_BETTER_THAN_STATIC
BG_EMPIRICAL_STEERING_VERDICT = DESTABILIZING
```

Direction geometry:

```text
cos(raw, mean_diff) = 0.1010
cos(raw, whitened_diff) = 0.0823
cos(raw, logistic_probe) = 0.0122
```

Interpretation:

> Empirical success directions learned something different from the raw NoNorm readout vector, but still did not become clean production-space steering handles. This directly supports the readout/production geometry mismatch.

---

## RMS-calibrated steering

RMS calibration tested the objection that L2-normalized perturbations were effectively microscopic.

Effective RMS comparison:

```text
L2 alpha 0.01: mean effective RMS 0.000138
L2 alpha 0.02: mean effective RMS 0.000276

RMS alpha 0.005: mean effective RMS 0.004063
RMS alpha 0.01:  mean effective RMS 0.008125
RMS alpha 0.02:  mean effective RMS 0.016250
```

Full sweep:

```text
total rows: 618
zero-baseline rows: 54
intervention rows / forward passes: 564 / 564
CUDA errors: 0
NaN/Inf activations: 0
empty outputs: 0
cache mode: disabled for all rows
loop identity: current_ut for all intervention rows
safety outliers: 1
```

The one safety outlier:

```text
row: 233
direction: EMPIRICAL_MEAN_DIFF
mode: single_loop_L1
alpha: 0.01
condition: negative
task: ARC-Challenge/1
reason: parse failure plus repetition_rate = 0.383
CUDA: false
NaN/Inf: false
empty output: false
effective RMS: 0.010006
activation RMS change: 0.010006
z-score change: -0.9634
```

Correct stability interpretation:

> One outlier in roughly 600 rows should be treated as stable with an isolated repetition/parse outlier, not broad destabilization.

Best RMS signed-looking cell:

```text
RMS_NORMALIZED / EMPIRICAL_WHITENED_DIFF / single_loop_L1 / alpha 0.02

positive z mean: +0.0249
negative z mean: -0.4109
random z mean:   +0.0046
random std:       0.3849
signed: yes
strong signed: false
```

Overall condition means:

```text
positive z mean: -0.2400
negative z mean: -0.2334
random z mean:   -0.2425
```

Interpretation:

> RMS scaling fixed the effective-perturbation-size issue and showed that meaningful RMS perturbations are mechanically stable, but it still did not produce reliable signed BG steering.

---

## Propagation / decay map

This was the most important positive hidden-write result.

```text
BG_PROPAGATION_VERDICT = PROPAGATES_TO_LATER_STATES
BG_PROPAGATION_DECAY_PROFILE = SURVIVES_32_TOKENS
BG_LOGIT_EFFECT_VERDICT = LOGITS_SHIFT_DIRECTIONALLY
```

Details:

```text
row count: 280
selected directions: EMPIRICAL_MEAN_DIFF, RAW_NONORM_READOUT
alpha: 0.01
max hidden delta RMS: 0.06056
mean cosine to direction: 0.07594
max logit KL: 0.00209
```

Interpretation:

> The write surface is alive. Perturbations propagate through later hidden states and affect logits. The failure is not “hooks do nothing.” The failure is that tested directions do not line up with reliable desired control.

---

## Text-prefix branch selection expansion

Text-prefix branch allocation remains the best practical control path so far.

```text
BG_TEXT_PREFIX_EXPANSION_VERDICT = WEAK_POSITIVE
BG_BRANCH_ALLOCATION_VERDICT = PROMISING
```

Expansion results:

```text
cached non-code tasks: 40
BG top1 lift: +0.04375
BG top2 lift: +0.02917
pairwise branch-ranking accuracy: 0.5672
oracle gap: 0.025
```

Domain breakdown:

```text
GSM8K top1 lift: +0.100
reasoning top1 lift: +0.017
science top1 lift: +0.033
```

Interpretation:

> Branch selection is modest but consistently directionally useful and deployable without hidden-state writes. This should be preserved as Phase 1.5 control.

---

## Causal-gradient probe

The local gradient probe tested whether a task-local causal direction exists at all.

```text
BG_CAUSAL_GRADIENT_VERDICT = GRADIENT_NO_BETTER_THAN_RANDOM
```

Metrics:

```text
row count: 18
positive z mean: -0.1287
negative z mean: -0.1287
random z mean:   -0.1534
signed causal signature: false
```

Gradient geometry:

```text
ARC-Challenge/0:
  cosine grad/raw = 0.3603
  cosine grad/empirical = 0.0619

ARC-Challenge/1:
  cosine grad/raw = 0.3069
  cosine grad/empirical = 0.0115

ARC-Challenge/10:
  cosine grad/raw = 0.4527
  cosine grad/empirical = -0.0129
```

Interpretation:

> Even local BG-score gradients did not beat random under this probe. This pushes the next training objective away from BG-score-only optimization and toward output/logit or sequence-level objectives.

---

## Causal intervention adapter

The causal adapter experiment trained only a small adapter, with Ouro and BG heads frozen.

Top verdicts:

```text
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
```

Execution and dataset:

```text
device: cuda:0
GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU

domains: reasoning, science
reasoning prefix: 64 tokens
science prefix: 32 tokens
GSM8K: excluded from adapter training

train tasks: 24
validation tasks: 8
heldout tasks: 8

train examples: 96
validation examples: 32
heldout examples: 32

domain examples:
  reasoning = 80
  science = 80
```

Adapter:

```text
trained variant: LowRankDeltaAdapter(rank=32)
params: 137,248
max allowed params: 2,000,000

target layer: 36
mode: multi_loop_decayed
position: prefix_last_token
alpha values: 0.005, 0.01, 0.02
use_cache: false
```

Methodological checks:

```text
Ouro parameters frozen: true
BG heads frozen: true
adapter only trained: true
KL answer position masked: true
sampled-generation backprop: false
BG score primary loss: false
intervention applied before FINAL ANSWER suffix, not directly at answer-token position
```

Training results:

```text
max_epochs: 20
epochs_completed: 5
early stopping: triggered
best_epoch: 2
training rows: 96
validation rows: 32

best validation margin lift: +0.003174
```

Epoch table:

```text
epoch  train margin lift  best val alpha  val margin lift  val accuracy      KL    delta RMS
1      -0.000853          0.020           +0.003127        0.969          0.008910  0.012494
2      +0.000957          0.010           +0.003174        0.969          0.005205  0.006249
3      +0.001088          0.020           +0.000215        0.969          0.014911  0.012495
4      +0.003035          0.005           -0.002949        0.969          0.009258  0.003125
5      +0.005476          0.010           +0.002989        0.969          0.003697  0.006252
```

Teacher-forced heldout eval:

```text
heldout rows: 32
BG_CAUSAL_ADAPTER_TEACHER_FORCED_VERDICT = ADAPTER_IMPROVES_LOGIT_MARGIN

best method: trained_adapter alpha 0.005
margin: 2.135497
margin lift: +0.010405
accuracy: 0.875
KL: 0.003590
delta RMS: 0.003124
```

Comparison:

```text
trained_adapter:
  best alpha: 0.005
  best margin lift: +0.010405
  accuracy: 0.875

random_same_rms:
  best alpha: 0.020
  best margin lift: +0.007234
  accuracy: 0.875

empirical_mean_diff:
  best alpha: 0.010
  best margin lift: +0.005706
  accuracy: 0.875

raw_nonorm:
  best alpha: 0.005
  best margin lift: +0.003982
  accuracy: 0.875

baseline:
  margin lift: 0.000000
  accuracy: 0.875
```

Free-generation eval:

```text
heldout tasks: 8
BG_CAUSAL_ADAPTER_FREE_GEN_VERDICT = TEACHER_FORCED_ONLY

baseline success: 0.500
trained_adapter best success: 0.625
raw_nonorm best success: 0.625
random_same_rms best success: 0.625
```

Free-generation interpretation:

> The trained adapter improved over baseline, but raw and random perturbations improved by the same amount. Therefore the free-generation lift was not adapter-specific.

Important caveat:

> The free-generation eval was low-resolution, not a decisive positive or negative transfer test. Of 8 heldout tasks, most were invariant to all interventions, and the apparent 0.500 → 0.625 lift came mainly from one task pattern where trained adapter, raw NoNorm, and random perturbations all helped. The correct conclusion is **no detectable adapter-specific free-generation transfer**, not “the adapter produced a real free-generation effect.”

Task-level pattern:

```text
ARC-Challenge/0:
  baseline and all interventions correct.

ARC-Challenge/1:
  baseline and all interventions failed / failed parse.

ARC-Challenge/10:
  baseline and all interventions failed / failed parse.

ARC-Challenge/11:
  baseline failed parse;
  at alpha 0.01 and 0.02, trained adapter, raw NoNorm, and random all became correct.

mmlu/high_school_biology/0:
  baseline and all interventions correct.

mmlu/high_school_biology/1:
  baseline and all interventions wrong.

mmlu/high_school_biology/10:
  baseline and all interventions correct.

mmlu/high_school_biology/11:
  baseline and all interventions correct.
```

Adapter stability:

```text
CUDA errors: 0
NaN/Inf rows: 0
empty output rate: 0
repetition rate: 0
baseline parse: 0.625
intervention parse around alpha 0.01/0.02: 0.750
```

Adapter geometry:

```text
cos(adapter proxy, raw NoNorm) = -0.000553
cos(adapter proxy, empirical mean diff) = -0.004294
cos(raw NoNorm, empirical mean diff) = +0.101002
```

Interpretation:

> The adapter found a weak local teacher-forced logit-control direction, and that direction is essentially orthogonal to both the raw BG readout vector and empirical success direction. This is not just “readout ≠ production”; it suggests **readout, empirical-success separation, and local logit-control are three distinct geometries**. The adapter did not find a trajectory-level write path. The correct verdict is LOCAL_LOGIT_CONTROL_ONLY, not PROMISING_WRITE_PATH.

---

## Final consolidated interpretation

The completed stack now supports the following:

```text
BG readout / candidate selection: validated.
BG partial trajectory prediction: validated.
Text-prefix branch allocation: promising.
Layer-hook intervention surface: validated.
Hidden-state perturbations propagate and affect logits.
Static inference-time steering: unsigned / unreliable.
RMS-calibrated static steering: unsigned / unreliable.
Empirical and gradient directions: unsigned / no better than random.
Tiny classifier-style adapter: no better than static.
Teacher-forced causal adapter: local logit control only.
Free-generation adapter transfer: no detectable adapter-specific effect in a low-resolution n=8 eval.
True trajectory-level write path: not yet achieved.
```

The most important geometry update:

```text
readout geometry != empirical-success geometry != local logit-control geometry
```

The core architecture conclusion:

> **The model is BG-readable and writable, but not yet BG-steerable.**

Or more precisely:

> **Ouro hidden states expose trajectory-quality information that BG taps can read and use for selection. Layer-hook writes propagate through the model. But the tested write directions and small local adapter do not create reliable semantic/action steering. Achieving true action steering likely requires sequence-level controller training or Phase 2 training-time integration.**

---

## Architecture implications

### Phase 1 — validated

```text
BG readout
BG candidate selection
BG trajectory prediction
```

### Phase 1.5 — promising

```text
text-prefix branch allocation
BG-guided branch selection / compute routing
```

### Static inference-time steering — tested and not supported

```text
raw NoNorm vectors
empirical mean/whitened/logistic directions
RMS-normalized static directions
local BG gradients
```

### Adapter inference-time steering — local only

```text
tiny teacher-forced adapter learns weak correct-option logit control
does not create adapter-specific free-generation lift
```

### Phase 2 — required for true action steering

Possible next routes:

```text
1. Sequence-level / black-box optimized intervention adapter
   Optimize actual generation behavior, not only teacher-forced logits.

2. DPO/RL-style trajectory-level adapter training
   Use generated trajectories and verifier rewards/preferences.

3. Backbone or adapter regularization
   Train the model/controller to preserve BG-readability while becoming receptive to BG modulation.

4. Plastic adapter / consolidation architecture later
   Add controlled learning modules and replay, with BG taps frozen as monitoring instruments.
```

---

## Recommended next work

The intended next step is Option 1:

> **Sequence-level / black-box-finetuned adapter.**

This should be treated as the **final serious frozen-backbone inference-time steering attempt**.

Suggested target:

```text
Train or optimize an adapter using final generation behavior / verifier reward / trajectory preference,
not only teacher-forced answer-token logits.
```

Design constraints:

```text
Ouro frozen initially.
BG taps frozen.
No sampled-generation backprop.
Use black-box optimization, REINFORCE, ES, DPO-style trajectory preference, or bandit-style update.
Keep strong regression gates:
  - BG tap preservation
  - base capability preservation
  - free-generation stability
  - heldout trajectory success
```

Stopping rule:

> If the sequence-level / black-box adapter fails to produce adapter-specific free-generation improvement, stop searching for more frozen-backbone inference-time steering variants and move to Option 2.

After that, move to Option 2:

> **Phase 2 training-time integration / backbone or controller regularization.**

This should explicitly preserve BG tap functionality during training.

---

## Key report paths

```text
artifacts/reports/probes/bg_preconsolidation_control_probes_2026-05-18/summary.md
artifacts/reports/probes/bg_preconsolidation_control_probes_2026-05-18/final_analysis.json
artifacts/reports/probes/bg_preconsolidation_control_probes_2026-05-18/rms_row_level_analysis.md
artifacts/reports/probes/bg_preconsolidation_control_probes_2026-05-18/propagation_decay_analysis.md
artifacts/reports/probes/bg_preconsolidation_control_probes_2026-05-18/text_prefix_expansion_analysis.md
artifacts/reports/probes/bg_preconsolidation_control_probes_2026-05-18/causal_gradient_probe.md

artifacts/reports/probes/bg_causal_intervention_adapter_2026-05-18/summary.md
artifacts/reports/probes/bg_causal_intervention_adapter_2026-05-18/analysis.md
artifacts/reports/probes/bg_causal_intervention_adapter_2026-05-18/training_report.md
artifacts/reports/probes/bg_causal_intervention_adapter_2026-05-18/teacher_forced_eval.md
artifacts/reports/probes/bg_causal_intervention_adapter_2026-05-18/free_generation_eval.md
docs/evaluator/causal-intervention-adapter.md
```
