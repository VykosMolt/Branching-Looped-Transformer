# BG Steering: findings and architectural status

**Date:** 2026-05-18
**Status:** Topic document consolidating all BG steering work to date. Feeds the eventual v9 synthesis. Companion to `phase1-progress-addendum.md`.
**Scope:** What "steering" means in this project, what was tested, what was found, and what the result forces architecturally.

## 0. One-paragraph summary

BG readouts are predictive of trajectory quality, and the layer-hook intervention mechanism is mechanically clean, but the directions BG reads from are not production-space steering directions. Across five direction-finding methods (raw NoNorm readout, empirical mean-diff, whitened mean-diff, logistic success probe, tiny learned adapter), tiny activation nudges produced movement in BG-readable space that was mostly unsigned and often comparable to random directions. No method crossed the strong signed-causal threshold, and final task lift was not demonstrated. The conclusion is **READ_ONLY_BG**: on the frozen Ouro-RLTT backbone, at inference time, BG is a readout/selector but not a reliable control handle. Making BG steerable requires either backbone regularization (align readout and production geometry) or an end-to-end-trained intervention adapter (learn a write-path) — both untested.

## 1. What "steering" means here, and what it does not

The project's vocabulary has used "steering" loosely. Three distinct things must be kept separate:

**Selection (best-of-N).** Generate N candidates, score each with BG, pick the best. The model produces options; BG picks one externally. This is what the BG controller does and what the CLT paper validated (95.2% on HH-RLHF). It is not steering — the model was not influenced, a selector chose among its outputs.

**Output steering.** Modify the model's output distribution to bias toward desirable outputs — logit-level intervention, contrastive decoding, classifier-free guidance. The model's behavior changes but only at the output layer.

**Action steering (the architectural target).** Modify what the model computes internally so its actions reflect the modification. Hidden-state intervention at intermediate layers, or train-time integration. The model's computation itself changes; outputs follow as a downstream consequence. This is the basal-ganglia-equivalent capability the brain-ontology framing implies.

The BG project to date had validated **selection**. Stage 2 tested whether **action steering** is achievable at inference time on the frozen backbone. The answer is no, by the methods tested.

## 2. The progression that led here

Steering was always the planned destination, not a retrofit. The staged path:

1. **Stage 0 — validate signal exists.** CLT paper: frozen Ouro loop states contain extractable preference information. Selection-time only. (Done.)
2. **Stage 1 — map where signal lives.** Cross-domain audits, simulator, trajectory prediction sweep. Found broad predictive envelope: 368 strong cells across (domain × prefix × config), pairwise accuracy up to 0.85. Still selection-time. (Done.)
3. **Stage 2 — test whether signal admits intervention.** This document. Does the direction BG reads from also serve as a control handle when written to? (Done — answer: no.)
4. **Phase 2 — make signal usable for trajectory shaping.** Train-time integration. (Not started; informed by Stage 2 result.)

Each stage was the prerequisite for the next. Selection (Stages 0-1) established that there is something worth steering. Stage 2 tested whether the readable signal is also a writable handle. It is not, on the frozen backbone.

## 3. What was tested in Stage 2

### Mechanisms

**Layer-hook injection (Mechanism A).** Temporary forward hooks on a target Ouro decoder layer. The hook returns `h + delta` as the layer output during normal generation. Downstream layers receive the modified state.
- Verdict: `LAYER_HOOK_INJECTION_VERDICT = READY`
- Zero-alpha equivalence: passed (zero-alpha hook produces identical output to no-hook generation under deterministic settings).
- Loop identity source: `current_ut`.
- `use_cache=False`.
- RMS movement scaled linearly with alpha: 0.005 → 0.000104, 0.01 → 0.000207, 0.02 → 0.000414.
- No CUDA errors, no NaN/Inf, hook forward/modification counts matched exactly.

**Latent loop-boundary fork (Mechanism B).** Run Ouro for partial loop steps, capture loop-boundary hidden state, perturb copies, continue remaining loop steps with `use_cache=False`.
- Verdict: `LATENT_LOOP_BOUNDARY_FORK_VERDICT = BLOCKED`
- Reason: generation could not be cleanly resumed from a post-loop-boundary state without validated cache/state forking. Partial loop computation was possible; clean resumption was not.

So all interpretable steering results come from Mechanism A. The loop-boundary mechanism remains untested due to implementation blockers, not due to a negative result.

### Intervention schedules

Four schedules tested per target:
- `single_loop_L1` — early one-shot steering.
- `single_loop_L4` — late one-shot steering.
- `multi_loop_uniform` — sustained steering at L1+L2+L3+L4, equal per-loop alpha.
- `multi_loop_decayed` — sustained steering at L1-L4 with increasing alpha toward L4 (0.25/0.50/0.75/1.00 × alpha).

### Directions

Five direction-finding methods, in increasing sophistication:
- `RAW_NONORM_READOUT` — the Stage 1 best NoNorm head weight vector.
- `EMPIRICAL_MEAN_DIFF` — difference of means between successful and unsuccessful trajectory features.
- `EMPIRICAL_WHITENED_DIFF` — mean-diff accounting for feature covariance.
- `LOGISTIC_SUCCESS_PROBE` — trained logistic classifier on success.
- Tiny learned adapter — 526K params, frozen features.

### Controls

Per condition: zero baseline, positive direction, negative direction, random same-norm direction. Alpha sweep {0.0, 0.005, 0.01, 0.02}. Alpha capped at 0.02 throughout.

## 4. What was found

### The central finding: readout geometry ≠ production geometry

The direction BG reads from to *discriminate* candidates is nearly orthogonal to the direction that would *move* trajectories toward success.

| Direction | Heldout AUC (as success predictor) | Cosine to raw NoNorm readout |
|---|---:|---:|
| RAW_NONORM_READOUT | 0.267 | 1.000 |
| EMPIRICAL_MEAN_DIFF | 0.583 | 0.101 |
| EMPIRICAL_WHITENED_DIFF | 0.583 | 0.082 |
| LOGISTIC_SUCCESS_PROBE | 0.600 | 0.012 |

The raw NoNorm readout — which achieved strong *pairwise predictive* accuracy in Stage 1 — has heldout AUC of 0.267 as a success-predictor direction in this held-out setting (below chance). The directions that actually predict success are near-orthogonal to it (cosine 0.01-0.10).

This is `BG_STEERING_DIRECTION_GEOMETRY_VERDICT = RAW_READOUT_NOT_PRODUCTION_DIRECTION`. It is the single most important architectural finding from the steering work. The axis along which a feature is linearly readable is not the axis along which intervening changes that feature — a known interpretability asymmetry, now confirmed for BG on Ouro.

### No method produced signed causal control

**Raw NoNorm direction (narrowed T1 follow-up, reasoning @ 64, layer 36):**
- `BG_LAYERHOOK_SIGNED_CAUSAL_VERDICT = UNSIGNED_EFFECT`
- Movement in BG-readable space existed, but positive and negative directions did not reliably separate from random.
- Only one cell (multi_loop_decayed, alpha 0.01) showed weak signed-looking behavior; none were strong.
- `BG_SINGLE_LOOP_POSITION_VERDICT = L1_BETTER` (single_loop_L1 beat single_loop_L4 among single-loop schedules).
- `BG_MULTILOOP_VERDICT = MULTILOOP_STRONGER` (multi_loop_decayed best schedule; gain over best single-loop = 0.071).
- `BG_LAYERHOOK_STABILITY_VERDICT = STABLE_BUT_TINY`.
- `BG_FINAL_TASK_LIFT_VERDICT = INSUFFICIENT`.

**Empirical directions:**
- `BG_EMPIRICAL_DIRECTION_CAUSAL_VERDICT = EMPIRICAL_UNSIGNED_ONLY`.
- `BG_EMPIRICAL_VS_RAW_VERDICT = EMPIRICAL_BEATS_RAW` (empirical gain over raw = 0.167 in aggregate).
- `BG_EMPIRICAL_FINAL_LIFT_VERDICT = NEGATIVE_LIFT`.
- `BG_EMPIRICAL_STEERING_VERDICT = DESTABILIZING` (formally, but driven by one repetition-heavy random-control row out of 432; not broad hook instability).
- Best empirical direction (EMPIRICAL_WHITENED_DIFF) was better in aggregate but did not become reliable signed control.

**Tiny learned adapter:**
- `BG_TINY_STEERING_ADAPTER_VERDICT = NO_BETTER_THAN_STATIC`.
- 526K params, heldout AUC 0.767 *as a classifier*.
- Still did not produce signed causal steering *as an intervention*.

### The pattern across methods

Sophistication increased (raw readout → mean-diff → whitened → logistic probe → learned adapter), classifier quality improved (AUC 0.27 → 0.77), but signed causal steering never emerged. The problem is not "wrong static direction." The relationship between hidden-state perturbation and trajectory outcome is not captured by any single direction at the tested intervention scale (alpha ≤ 0.02).

### Notable secondary findings

These hold even though the overall verdict is negative, and are worth preserving:
- **L1 intervention beats L4** among single-loop schedules. Consistent with the intuition that early intervention has more loops to propagate through.
- **Multi-loop decayed beats best single-loop** (gain 0.071) for the raw direction. Sustained loop-wide modulation does shift refinement more than one-shot, even if not to signed-causal threshold. The architectural intuition about multi-loop modulation was directionally correct, just not sufficient.
- **Empirical directions beat raw readout** in aggregate (gain 0.167). The success direction is findable and better than the readout direction — it's just still not a reliable control handle.

## 5. Why the adapter result does not close the question

The tiny adapter was trained as a **classifier on frozen features** (predict success from features), then its learned weight used as an intervention direction. This is the readout-vs-production mismatch in a more sophisticated form — it learned to *read*, then was repurposed to *write*. Same geometry problem.

An adapter trained **end-to-end through the forward pass** with a task-success objective (apply intervention → continue generation → measure success → backprop the success signal through the intervention) is a different and untested thing. It would learn to *write* directly. The distinction:
- Tested (failed): objective = "predict success from features."
- Untested: objective = "produce an intervention that causes success."

The negative adapter result therefore rules out "readout-style adapter" but not "write-path adapter." This matters for choosing the next experiment.

## 6. What this forces architecturally

### The frozen backbone is BG-readable but not BG-steerable

The current Ouro-RLTT backbone encodes trajectory quality in a way BG can read, but its hidden states are not organized so that pushing along the readable axis changes generation. The basal-ganglia-equivalent capability does not exist at inference time on the frozen backbone, by the methods tested.

### The v8.1 Phase 1 architecture is unaffected

BG as a best-of-N selector is validated and stands unchanged. This result narrows what BG can do (selection yes, inference-time steering no); it does not contradict any prior selection result. The two-head + backup architecture from v8.1, the conservative routing, the contrast-type principle — all unchanged.

### The v8.1 Phase 2 framing sharpens

v8.1 §7 framed Phase 2 as "preserve readable directions during continued pretraining." This is now insufficient. The Stage 2 result shows readable ≠ steerable. Phase 2 must do one of:

**Option A — Backbone regularization to align geometry.** Train Ouro so the BG-readable axis *becomes* a causal axis — readout and production geometry aligned by construction. Cleaner architecturally (the backbone itself becomes steerable, no bolt-on), but expensive (cloud compute, continued pretraining) and uncertain (training may not be able to align the geometries without degrading base capability).

**Option B — Causal intervention adapter (differentiable proxy).** Train a small module that maps the current hidden state to a production-space intervention vector. The training objective cannot be naive "backprop task success through generation" — sampled token generation and discrete pass/fail verification are both non-differentiable. Instead, use a differentiable proxy under teacher forcing:
- *Primary target: correct-answer logit margin.* Apply the adapter during a forward pass over fixed (teacher-forced) continuations; optimize the logit of the correct answer token / correct-option margin. This directly tests whether intervention can shift the model's output distribution toward correctness — genuine write-path capability.
- *Alternative target: pairwise causal contrast.* Given success/fail continuation pairs, train the adapter so intervention moves hidden states toward the successful continuation's later geometry.
- *Diagnostic-only target: later BG score.* Risks circularity — optimizing the adapter to increase the very readout signal whose steerability is in question may just relearn the readout direction rather than a production direction. Report but do not treat as the load-bearing result.
- *Later option: two-stage.* Differentiable proxy first, optional black-box (ES/REINFORCE) fine-tune on actual task reward second.

Cheaper than backbone regularization, locally testable, and directly tests "can a learned write-path convert BG-readable information into causal control." The failed tiny adapter does not rule this out because it was trained as a frozen-feature classifier (learned to read), not as a causal intervention with a differentiable production objective (learned to write).

## 7. Recommended next experiment

**Option B before Option A.** The causal intervention adapter is the cheaper test of the more tractable hypothesis, and its result determines whether the expensive Option A is justified.

- If the adapter produces causal control on a differentiable production proxy (correct-answer logit margin) that transfers to actual task lift → inference-time steering via a learned write-path is achievable, Phase 2 backbone work may be unnecessary.
- If it fails → inference-time steering on the frozen backbone is conclusively unachievable by tested methods, and Phase 2 backbone regularization is justified as the only remaining path.

Design sketch:
- Small adapter (~500K params) mapping current hidden state → intervention vector, applied via the validated layer-hook surface.
- Frozen throughout: Ouro weights, BG heads, tokenizer, checkpoints. Only the adapter trains.
- Training objective: differentiable proxy under teacher forcing. Primary = correct-answer logit margin; alternative = pairwise causal contrast toward successful-continuation geometry; diagnostic-only = later BG score (circularity risk). NOT naive backprop through sampled generation or discrete task success.
- Evaluation: after training the adapter on the differentiable proxy, test whether the learned intervention produces signed causal sensitivity AND final task lift on held-out tasks — using the same three metrics as Stage 2 v3.
- Critical difference from the failed tiny adapter: that one was a frozen-feature success classifier repurposed as a direction. This one is trained as a causal write-path with a differentiable production objective.

`RECOMMENDED_NEXT = train_causal_intervention_adapter_with_differentiable_proxy_before_phase2_backbone`

## 8. Verdict ledger (steering work)

```
LAYER_HOOK_INJECTION_VERDICT            = READY
LATENT_LOOP_BOUNDARY_FORK_VERDICT       = BLOCKED (clean resumption not implementable)
BG_LAYERHOOK_MECHANICAL_VERDICT         = READY
BG_LAYERHOOK_SIGNED_CAUSAL_VERDICT      = UNSIGNED_EFFECT
BG_SINGLE_LOOP_POSITION_VERDICT         = L1_BETTER
BG_MULTILOOP_VERDICT                    = MULTILOOP_STRONGER (but sub-threshold)
BG_LAYERHOOK_STABILITY_VERDICT          = STABLE_BUT_TINY
BG_EMPIRICAL_DIRECTION_CAUSAL_VERDICT   = EMPIRICAL_UNSIGNED_ONLY
BG_EMPIRICAL_VS_RAW_VERDICT             = EMPIRICAL_BEATS_RAW
BG_STEERING_DIRECTION_GEOMETRY_VERDICT  = RAW_READOUT_NOT_PRODUCTION_DIRECTION
BG_EMPIRICAL_FINAL_LIFT_VERDICT         = NEGATIVE_LIFT
BG_TINY_STEERING_ADAPTER_VERDICT        = NO_BETTER_THAN_STATIC
OVERALL_BG_STEERING_VERDICT             = READ_ONLY_BG
```

Key continuous measurements:
```
RAW_NONORM_READOUT heldout AUC          = 0.267
best empirical direction heldout AUC    = 0.583 (whitened) / 0.600 (logistic)
tiny adapter heldout AUC                = 0.767
best empirical cosine to raw NoNorm     = 0.082
empirical gain over raw (aggregate)     = 0.167
multiloop gain over best single-loop    = 0.071
RMS movement at alpha 0.02              = 0.000414 (tiny, controlled, linear)
```

## 9. What is settled and what remains open

### Settled

- BG readouts are predictive of trajectory quality (Stages 0-1).
- BG is a valid best-of-N selector (CLT, simulator, controller).
- Layer-hook intervention is mechanically clean (zero-alpha equivalence, linear RMS scaling).
- Raw NoNorm readout directions are not production-space steering directions.
- Readout geometry and production geometry are near-orthogonal.
- No tested static direction or readout-style adapter produces signed causal steering at alpha ≤ 0.02.
- The frozen backbone is BG-readable but not BG-steerable at inference time.

### Open

- Causal intervention adapter (differentiable production proxy, e.g. correct-answer logit margin) — untested, the cheapest remaining inference-time option. Distinct from the failed frozen-feature classifier adapter.
- Latent loop-boundary fork — blocked by resumption implementation, not by a negative result; could be revisited if clean cache/state forking is built.
- Phase 2 backbone regularization to align readout/production geometry — untested, the expensive path, justified only if Option B fails.
- Whether higher alpha (>0.02) would produce signed control — deliberately not tested; deferred unless tiny-alpha methods show promise (they did not).
- Whether multi-layer simultaneous intervention (24+36+47) behaves differently — deferred from Stage 2; reserve for follow-up.

## 10. One-paragraph thesis

On the frozen Ouro-RLTT backbone, BG is a readout and selector but not an inference-time control handle. The directions along which BG reads trajectory quality are near-orthogonal to the directions along which intervention would improve trajectories, and no static direction, empirical success direction, or readout-style adapter tested produces reliable signed causal steering at safe activation-nudge magnitudes. Making BG into an action-steering handle — the basal-ganglia-equivalent capability the brain-ontology framing requires — is not free from the trained-for-prediction model; it must be trained in, either by aligning the backbone's readout and production geometry through regularization, or by learning a dedicated write-path adapter. The next experiment tests the cheaper of these (a causal intervention adapter trained on a differentiable production proxy such as correct-answer logit margin under teacher forcing — not a naive backprop-through-sampled-generation setup, which is non-differentiable) before committing to the more expensive (backbone regularization).
