# BG Steering Consolidation — inference-time closure and Phase 2 framing

**Date:** 2026-05-18
**Status:** Consolidation document. Locks the completed frozen-backbone inference-time steering investigation. Supersedes the standalone `steering-findings.md` by incorporating the final sequence-level adapter result. Feeds the eventual v9 synthesis. Cross-references `phase1-progress-addendum.md`.
**Parent spec:** `phase1-routing-policy-locked.md` (v8.1, Phase 1 architecture locked, unchanged).

## 0. One-paragraph conclusion

The frozen Ouro-RLTT backbone is BG-readable and mechanically writable, but not BG-steerable at inference time. Seven intervention methods — spanning static directions, calibrated directions, local gradients, and learned adapters trained with both differentiable-proxy and black-box-reward objectives — were tested. None produced reliable adapter-specific action steering at safe intervention magnitudes (alpha/effective-RMS ≤ 0.02). Two independently-trained adapters, optimized against entirely different objectives, converged to the same control direction (cosine 0.95), and that direction does not transfer to free generation. The readout direction, the empirical-success direction, and the controllable-intervention direction are mutually near-orthogonal: reading trajectory quality and writing trajectory control are separate geometries in this architecture. The inference-time steering search is **closed under tested methods**. Action steering — the basal-ganglia-equivalent capability — is not free from a model trained only for prediction; it must be trained in. The project pivots to Phase 2 training-time integration.

## 1. Scope of this document

This locks one specific question: **can the frozen Ouro-RLTT backbone be steered at inference time using BG-readable information?** The answer is no, under tested methods at safe magnitudes.

This does NOT touch:
- Phase 1 validated capabilities (BG readout, candidate selection, trajectory prediction) — unchanged.
- Phase 1.5 promising capabilities (text-prefix branch allocation, BG-guided compute routing) — unchanged.
- The v8.1 locked Phase 1 head architecture and routing — unchanged.
- The CLT paper's headline (95.2% HH-RLHF readout/selection) — unchanged.
- Hunter-Seeker ARC agent or Ouro depth expansion — separate tracks, untouched.

The closure is narrow and precisely scoped. It rules out a specific class of methods, not the architecture.

## 2. The seven-method closure

The inference-time steering investigation tested seven methods of increasing sophistication. All failed to produce reliable signed/adapter-specific action steering on the frozen backbone at alpha ≤ 0.02.

| # | Method | Result |
|---|---|---|
| 1 | Raw NoNorm readout-vector steering | UNSIGNED_ONLY |
| 2 | Empirical mean-diff / whitened / logistic success directions | EMPIRICAL_UNSIGNED_ONLY |
| 3 | RMS-calibrated static directions | RMS_UNSIGNED_ONLY |
| 4 | Local BG-score gradient | GRADIENT_NO_BETTER_THAN_RANDOM |
| 5 | Tiny classifier-style adapter | NO_BETTER_THAN_STATIC |
| 6 | Teacher-forced causal adapter (differentiable logit-margin proxy) | LOCAL_LOGIT_CONTROL_ONLY |
| 7 | Sequence-level adapter (black-box REINFORCE reward) | NO_FROZEN_BACKBONE_WRITE_PATH |

The progression matters. Each method addressed a hypothesis about *why* the previous one failed:

- Methods 1-3 tested whether the failure was the *choice of static direction* (raw vs empirical vs RMS-calibrated). It was not — all static directions failed.
- Method 4 tested whether a *task-local* direction exists. It did not (no better than random).
- Method 5 tested whether a *learned* direction (classifier) helps. It did not.
- Method 6 tested whether a *differentiable causal objective* (teacher-forced logit margin) finds a write-path. It found only local logit control, no free-generation transfer.
- Method 7 tested whether a *black-box sequence-reward objective* finds a write-path, removing the teacher-forcing shortcut. It learned validation reward but did not transfer to heldout, and converged to the same direction as method 6.

The hypothesis space for "frozen-backbone inference-time steering, but we just haven't found the right direction/objective yet" is now substantially exhausted.

### Per-method supporting data (preserved in full)

These numbers are retained so this document is self-contained and the closure can be defended without re-opening the underlying reports.

**Method 1 — raw NoNorm (narrowed T1 follow-up, reasoning @ 64, layer 36):**
```
BG_LAYERHOOK_SIGNED_CAUSAL_VERDICT = UNSIGNED_EFFECT
BG_SINGLE_LOOP_POSITION_VERDICT    = L1_BETTER
BG_MULTILOOP_VERDICT               = MULTILOOP_STRONGER
MULTILOOP_GAIN_OVER_BEST_SINGLE    = 0.0707
BEST_SINGLE_LOOP_MODE              = single_loop_L1
BEST_MULTILOOP_MODE                = multi_loop_decayed
zero-alpha hook equivalence: pass; loop identity: current_ut; cache: disabled; CUDA/NaN errors: 0
```
Movement existed in BG-readable space; positive and negative did not reliably separate from random.

**Methods 2-3 — empirical directions and RMS calibration.** RMS calibration directly addressed the objection that L2-normalized perturbations were microscopic. Effective-RMS comparison:
```
L2  alpha 0.01: effective RMS 0.000138
L2  alpha 0.02: effective RMS 0.000276
RMS alpha 0.005: effective RMS 0.004063
RMS alpha 0.01:  effective RMS 0.008125
RMS alpha 0.02:  effective RMS 0.016250
```
RMS scaling raised effective perturbation magnitude by ~60x at alpha 0.02 (0.000276 → 0.0163) and confirmed that meaningfully-sized perturbations are mechanically stable — but still produced no reliable signed steering. The "nudge was too small" objection is therefore closed: the failure persists at perturbation magnitudes two orders larger.

Empirical-direction full sweep: 618 rows, 564 intervention forward passes, 0 CUDA errors, 0 NaN/Inf, 0 empty outputs, cache disabled throughout, loop identity current_ut for all rows. Best RMS signed-looking cell (RMS_NORMALIZED / EMPIRICAL_WHITENED_DIFF / single_loop_L1 / alpha 0.02): positive z +0.0249, negative z −0.4109, random z +0.0046 (random std 0.3849); signed yes, strong-signed false. Overall condition means: positive −0.2400, negative −0.2334, random −0.2425.

Aggregate empirical-vs-raw: `EMPIRICAL_BEATS_RAW`, empirical gain over raw 0.167; `BG_EMPIRICAL_FINAL_LIFT_VERDICT = NEGATIVE_LIFT`.

**Stability caveat on the empirical run.** The formal `BG_EMPIRICAL_STEERING_STABILITY_VERDICT = DESTABILIZING` was driven by a single outlier row, not broad instability:
```
row 233: EMPIRICAL_MEAN_DIFF / single_loop_L1 / alpha 0.01 / negative / ARC-Challenge/1
  reason: parse failure + repetition_rate 0.383
  CUDA: false, NaN/Inf: false, empty: false
  effective RMS 0.010006, z-score change -0.9634
```
One destabilizing row in ~600 should be read as stable-with-an-isolated-outlier, not broad hook instability:
```
Formal analyzer verdict:          DESTABILIZING
Practical stability interpretation: stable with one isolated output-quality outlier (1 row / ~600)
```
The DESTABILIZING label in the method table above is the formal analyzer verdict; the practical reality is stable.

**Method 4 — local BG-score gradient.** `GRADIENT_NO_BETTER_THAN_RANDOM`. 18 rows, positive z −0.1287, negative z −0.1287, random z −0.1534, signed-causal signature false. Notably, the gradient direction had non-trivial alignment to raw NoNorm (cosine 0.30–0.45 across ARC-Challenge tasks: 0.3603, 0.3069, 0.4527) yet still did not beat random — even the *locally-optimal* BG-score-ascent direction is not a steering direction. This is what pushed subsequent objectives away from BG-score optimization toward output/logit and sequence-level targets.

**Method 6 — teacher-forced causal adapter.** `LOCAL_LOGIT_CONTROL_ONLY`. LowRankDeltaAdapter(rank=32), 137,248 params. Best val margin lift +0.0032 (epoch 2, early stopped at epoch 5); teacher-forced heldout `ADAPTER_IMPROVES_LOGIT_MARGIN` with best margin lift +0.0104 (beating random +0.0072, empirical +0.0057, raw +0.0040). Free-gen `TEACHER_FORCED_ONLY`: baseline 0.500 → all interventions 0.625, but the single moved task (ARC-Challenge/11) responded identically to random — i.e. the free-gen eval was effectively a null measurement at n=8 (5/8 tasks invariant). Methodological checks: KL answer-position masked true; intervention applied before FINAL ANSWER suffix, not at the answer token. Adapter geometry: cos to raw NoNorm −0.0006, cos to empirical mean-diff −0.0043.

**Method 7 — sequence-level adapter (final test).** Detailed in §4.

## 3. The geometry finding (central mechanistic result)

The single most important architectural result from the steering investigation is that the relevant directions are mutually near-orthogonal, and the one stable controllable direction is not a steering direction.

### Readout ≠ empirical-success ≠ controllable-intervention

```
cos(raw NoNorm readout, empirical mean-diff)        = 0.101
cos(raw NoNorm readout, empirical whitened-diff)    = 0.082
cos(raw NoNorm readout, logistic success probe)     = 0.012
cos(sequence adapter delta, raw NoNorm readout)     = 0.006
cos(teacher-forced adapter delta, raw NoNorm)       ≈ 0.000 (orthogonal)
cos(teacher-forced adapter delta, empirical mean)   ≈ 0.000 (orthogonal)
```

The readout direction (what BG reads to discriminate trajectory quality) is near-orthogonal to the empirical-success direction (what separates successful from failed trajectories) which is near-orthogonal to the controllable-intervention direction (what an adapter learns to push).

### The controllable direction is stable across objectives but non-steering

```
cos(sequence-level adapter delta, teacher-forced adapter proxy) = 0.951
```

This is the convergent-evidence result. Two adapters trained against completely different objectives —
- method 6: teacher-forced correct-answer logit margin (differentiable proxy),
- method 7: black-box sequence-level generation reward (REINFORCE),

converged to essentially the same direction (cosine 0.95). And that shared direction produces only local logit control, not free-generation transfer. The frozen backbone admits a stable, consistent intervention direction across objectives — but that direction is a local-logit-control direction, not an action-steering direction.

### Interpretation

This is a clean instance of the reading-vs-writing asymmetry known in interpretability: the direction along which a feature is linearly *readable* is generally not the direction along which *intervening* changes the feature in the model's own representation. Here the asymmetry is sharp and multi-way:
- BG can read trajectory quality (readout direction).
- There exists a direction that empirically separates success/failure (empirical direction).
- There exists a stable direction an adapter can push for local logit control (controllable direction).
- None of these three is the same vector, and none produces action steering.

The frozen backbone's hidden states *encode* trajectory quality (readable) but are not *organized* so that any tested intervention converts that encoding into generation control.

## 4. The final experiment (method 7) in detail

The sequence-level adapter run was designed as the final, clean test, with confound-elimination built in.

### Confounds eliminated

- **Parser usable:** parse rate 1.000 on baseline generations.
- **Reward had variance:** REWARD_SIGNAL_USABLE, 0.25 nonzero-variance rate (REINFORCE/ES had signal to optimize).
- **Optimizer proven capable:** OPTIMIZER_CAN_LEARN_TRIVIAL_TARGET. On an artificial "always answer A" target, the full training harness moved reward 0.500 → 0.667 with 50 nonzero gradient updates and adapter param delta L2 = 12.3. The optimizer demonstrably works.
- **GPU throughput feasible:** ~16s per uncached generation, overnight-feasible.

### Training succeeded; transfer failed

- Training: SEQUENCE_REWARD_IMPROVES. Validation reward 0.0875 → 0.4625 (+0.375 lift) over 175 updates, best at update 50.
- Heldout: NO_ADAPTER_SPECIFIC_TRANSFER. On 12 heldout tasks × 4 samples = 48 samples per method:

| Method | Heldout success |
|---|---:|
| no intervention | 0.438 |
| random same-RMS | 0.458 |
| raw NoNorm static | 0.375 |
| teacher-forced adapter | 0.521 |
| trained sequence adapter | 0.479 |

The trained adapter beat no-intervention by +0.042 but lost to the prior teacher-forced adapter checkpoint (0.479 vs 0.521) and was classified WORSE_THAN_RANDOM on the adapter-specificity analysis (random/static moved 5 tasks; adapter-only moved 2).

### Why the val-vs-heldout gap is itself diagnostic

The +0.375 validation lift collapsing to no heldout transfer is the signature of overfitting val-set-specific behavior, not learning a generalizable write-path. With a proven-capable optimizer and adequate capacity (137K params), a generalizable inference-time write-path would be expected to leave at least some adapter-specific heldout signal. Instead, the adapter learned validation-set-specific behavior that did not generalize. This is consistent with "no generalizable frozen-backbone write-path to find."

### Why the closure is clean

Because the optimizer was proven capable on a trivial target (OPTIMIZER_CAN_LEARN_TRIVIAL_TARGET) AND the adapter learned validation reward (SEQUENCE_REWARD_IMPROVES) AND heldout still showed no adapter-specific transfer, the null result cannot be attributed to optimization failure. The stopping-rule disambiguation fires: NO_FROZEN_BACKBONE_WRITE_PATH with OPTIMIZER_CAN_LEARN licenses the architectural closure claim.

## 5. The closure verdict and its scope

```
BG_SEQUENCE_LEVEL_ADAPTER_VERDICT       = NO_FROZEN_BACKBONE_WRITE_PATH
FROZEN_BACKBONE_INFERENCE_STEERING_STATUS = CLOSED_UNDER_TESTED_METHODS
STOPPING_RULE_APPLIES                   = true
STOPPING_RULE_SCOPE                     = safe_alpha_leq_0_02_under_tested_optimizers
RECOMMENDED_NEXT                        = consolidate_phase1_phase1_5_and_design_phase2_training_time_integration
```

### What is closed

Frozen-backbone inference-time action steering using BG-readable information, via static directions, calibrated directions, local gradients, or learned adapters (differentiable-proxy or black-box-reward), at safe intervention magnitudes (alpha/effective-RMS ≤ 0.02).

### What is NOT closed

- Steerability at larger, destabilizing intervention magnitudes (deliberately not tested; a method requiring destabilizing perturbations would not be useful, but it is not ruled out).
- Steerability after backbone or controller training (Phase 2).
- Steerability with a future architecture that explicitly trains a write-path during pretraining.

### Stopping-rule discipline

This was pre-committed: if the final sequence-level test failed with the optimizer proven capable, stop searching for inference-time variants and move to Phase 2. The discipline exists to prevent indefinite search through inference-time method variants. Seven methods is sufficient evidence. The closure should be treated as settled and not re-litigated in future sessions unless a genuinely new mechanism (not a variant of a tested one) is proposed.

## 6. Architectural status after closure

### Phase 1 — validated, unchanged

```
BG readout                      (CLT: 95.2% HH-RLHF pairwise)
BG candidate selection          (simulator, controller; replay-exact)
BG trajectory prediction        (Stage 1: STRONG, 368 strong cells, up to 85% pairwise)
```

### Phase 1.5 — promising, unchanged

```
text-prefix branch allocation   (HELPS / PROMISING; deployable without hidden-state writes)
BG-guided compute routing       (modest but directionally useful)
```

Text-prefix branch allocation expansion (40 cached non-code tasks):
```
BG top1 lift:                +0.04375
BG top2 lift:                +0.02917
pairwise branch-ranking acc: 0.5672
oracle gap:                  0.025
domain top1 lift: GSM8K +0.100, science +0.033, reasoning +0.017
```
Modest but consistently directionally useful, and deployable without any hidden-state writes. This is the steering-adjacent capability that DID work, and it should be preserved as a Phase 1.5 deliverable.

### Inference-time steering — closed

```
static directions               UNSIGNED_ONLY
calibrated directions           UNSIGNED_ONLY
local gradients                 NO_BETTER_THAN_RANDOM
classifier adapter              NO_BETTER_THAN_STATIC
differentiable-proxy adapter    LOCAL_LOGIT_CONTROL_ONLY
black-box-reward adapter        NO_FROZEN_BACKBONE_WRITE_PATH
```

### Mechanical write surface — alive (this is why Phase 2 is plausible)

```
layer-hook intervention         mechanically valid, zero-alpha equivalent, stable
perturbation propagation        PROPAGATES_TO_LATER_STATES, SURVIVES_32_TOKENS
logit effect                    LOGITS_SHIFT_DIRECTIONALLY
```

Propagation/decay specifics (280 rows, directions EMPIRICAL_MEAN_DIFF + RAW_NONORM_READOUT, alpha 0.01):
```
max hidden delta RMS:   0.06056
mean cosine to direction: 0.07594
max logit KL:           0.00209
```

The write surface is not inert. Perturbations do propagate and do affect logits. The failure is not "writes do nothing" — it is "no tested direction/objective converts the readable signal into reliable control." This distinction is what keeps Phase 2 (training a write-path in) plausible: there is a live surface to train against.

## 7. Phase 2 framing (corrected and sharpened)

### Why v8.1 §7 is insufficient

v8.1 §7 framed Phase 2 as "preserve readable directions during continued pretraining." The steering investigation establishes that this is necessary but not sufficient. Readable ≠ steerable. Preserving the readout direction during training does not make it a control direction — they are near-orthogonal and the orthogonality is the problem.

### What Phase 2 must actually do

Phase 2 must **actively align readout geometry with production/control geometry**, or **train a dedicated write-path into the backbone**. Two non-exclusive routes:

**Route A — Backbone regularization for geometry alignment.** Continued pretraining with an objective that makes the BG-readable direction *become* a causal control direction. The training signal must reward "intervention along the BG-readable axis changes generation in the BG-predicted way," not merely "the BG-readable axis remains readable." This is a more demanding objective than v8.1's "preserve readouts." Expensive (cloud A100/H100, continued pretraining on 20-50B tokens per the original v4 Phase 2 sketch).

**Route B — Trained write-path module integrated during backbone training.** Rather than aligning the whole backbone's geometry, co-train a write-path module *with* the backbone (not frozen, as in the failed inference-time adapters) so the backbone learns to be receptive to the module's interventions. The inference-time adapter experiments failed because the backbone was frozen and not trained to respond to the adapter. Co-training removes that constraint.

The key distinction from the closed inference-time work: in Phase 2, the backbone is NOT frozen. The inference-time investigation established that a frozen backbone has no write-path; Phase 2 tests whether training installs one.

### BG taps must remain frozen evaluation instruments during Phase 2

During Phase 2, the BG readout heads must remain frozen and be used as measurement and preservation instruments, not as trainable targets. Phase 2 training may alter the backbone (Route A) or write-path modules (Route B), but the readout heads should be used to monitor whether BG-readable structure is preserved or made more causally useful — not updated to chase the moving representation.

This matters because otherwise Phase 2 could "succeed" by moving the readout target rather than making the model steerable. If the taps are co-trained with the backbone, the readout direction drifts toward whatever the backbone happens to produce, and "steerability" becomes circular: the model would look steerable because the measuring instrument moved to agree with it. Freezing the taps as fixed instruments keeps the steerability claim honest — a Phase 2 success must be the backbone becoming responsive to a *fixed* BG-readable axis, not the axis relocating to a controllable region.

### Multi-loop and loop-position considerations carried forward

From the Stage 2 secondary findings (which held even under the negative headline):
- L1 intervention beat L4 among single-loop schedules (early intervention has more loops to propagate through).
- multi_loop_decayed beat best single-loop (sustained loop-wide modulation shifts refinement more than one-shot).

These were directionally correct but insufficient to reach signed-causal threshold on the frozen backbone. For Phase 2, they suggest the training objective should make the loops receptive to multi-loop steering (the loop-wide modulation hypothesis), and that the backbone should be trained to integrate L1-injected steering through its refinement iterations rather than suppress it. A Phase 2 backbone trained to amplify multi-loop BG-direction modulation (rather than the frozen backbone's tendency to refine perturbations away) is the architecturally-motivated target.

### Head-config note for trajectory-level Phase 2

Stage 1 found that the trajectory-prediction peak config (36_mean / AntisymLinear) differs from the v8.1-locked finished-candidate head (36_L4 / NoNorm). If Phase 2 deploys BG at the trajectory level, the trajectory-level config may be the relevant one. The v8.1 locked architecture remains correct for finished-candidate selection; trajectory-level operation is a separate config question.

### Phase 2 success criterion

A Phase 2 success requires both:
1. frozen BG taps retain readout/ranking performance (no degradation of the validated Phase 1 capability), and
2. interventions using BG/controller signals produce heldout behavioral improvement not matched by random/static controls.

Both conditions are necessary. Condition 1 alone is just Phase 1 preserved. Condition 2 without condition 1 is the circular false-success (steerable only because the taps drifted). Only both together demonstrate that a fixed BG-readable axis has been made causally useful.

## 8. The honest project narrative

Where the project stands:

- BG reads trajectory quality well, across domains, at finished-candidate and partial-trajectory levels. **Validated.**
- BG selects among candidates well (best-of-N). **Validated and deployable.**
- BG-guided branch allocation is modestly useful and deployable without hidden-state writes. **Promising.**
- The frozen backbone cannot be steered at inference time by any tested method. **Established.**
- Making BG an action-steering handle — the basal-ganglia-equivalent capability the brain-ontology framing requires — needs training-time integration. **The path forward.**

The brain-ontology thesis is partially supported and partially open. The "BG taps read cortical state" half is strongly validated. The "BG modulates cortical computation" half is not achievable on the frozen backbone and is the explicit target of Phase 2. The honest current claim is: the project has built a strong evaluator and trajectory-quality readout for Ouro, and has rigorously established that converting that readout into action steering requires training the backbone, not just reading it.

## 9. Paper implications

The steering investigation is a publishable systematic negative/mechanistic result, distinct from and complementary to the CLT readout/selection paper.

Proposed framing: BG-style linear readouts strongly predict trajectory quality in a looped transformer (85% pairwise across a broad operating envelope), but the readout directions are near-orthogonal to any direction that causally steers generation. Seven intervention methods were tested, spanning static directions, calibrated directions, local gradients, and learned adapters trained with both differentiable-proxy and black-box-reward objectives. None produced reliable inference-time action steering at safe magnitudes. Two independently-trained adapters converged to the same non-steering control direction (cosine 0.95). Reading trajectory quality and writing trajectory control are separate geometries in this architecture.

Why it's worth publishing: the reading-vs-writing asymmetry is underreported precisely because clean negative results rarely get written up. The seven-method systematic sweep plus the cross-objective geometry convergence make this a rigorous instance. CLT covers what BG can read; this covers what it cannot write on a frozen backbone.

## 10. What is locked vs open

### Locked (settled, do not re-litigate)

- Frozen-backbone inference-time steering is closed under tested methods at alpha ≤ 0.02.
- Readout, empirical-success, and controllable-intervention directions are mutually near-orthogonal.
- The controllable-intervention direction is stable across objectives (cosine 0.95) but non-steering.
- v8.1 Phase 1 architecture, routing, and CLT result are unchanged by all of this.
- The mechanical write surface is alive (perturbations propagate and affect logits).

### Open (Phase 2 and beyond)

- Whether backbone regularization (Route A) can align readout/production geometry.
- Whether a co-trained write-path module (Route B) installs a control handle.
- Whether Phase 2 can install steerability while preserving BG readout performance — the central danger is making the model steerable but destroying the taps (or, worse, appearing steerable only because the taps drifted to agree with the trained backbone). Readout preservation under fixed taps is a required success criterion, not an afterthought.
- Whether multi-loop steering training makes the loops receptive to BG modulation.
- Whether steerability exists at larger destabilizing magnitudes (deliberately untested).
- Trajectory-level vs finished-candidate head configs for Phase 2 deployment.
- Latent loop-boundary fork (blocked by resumption implementation, not a negative result; revisit if clean cache/state forking is built).

## 11. Recommended next work

1. **Consolidate Phase 1 / Phase 1.5 as the validated deliverable.** BG readout, selection, trajectory prediction, and branch allocation are the project's validated capabilities. These can anchor the CLT paper and a deployment story independent of steering.

2. **Design Phase 2 training-time integration.** Choose Route A (backbone regularization for geometry alignment) or Route B (co-trained write-path module) or a staged combination. The design must target geometry alignment / write-path installation, not merely readout preservation. Backbone is unfrozen in Phase 2 — this is the key difference from the closed inference-time work.

3. **Optionally, write the steering negative-result paper** using §3 and §9 as the core. The cross-objective geometry convergence is the central figure.

`RECOMMENDED_NEXT = consolidate_phase1_phase1_5_and_design_phase2_training_time_integration`

## 12. File index

**This consolidation:** `docs/evaluator/steering-consolidation.md`
**Supersedes:** `steering-findings.md` (standalone, pre-final-experiment)
**Companion:** `phase1-progress-addendum.md`
**Parent spec:** `phase1-routing-policy-locked.md`

**Experiment reports (steering arc):**
- Steering suite: `artifacts/reports/probes/bg_steering_suite_2026-05-18/`
- Stage 1 trajectory prediction: `artifacts/reports/probes/bg_trajectory_prediction_2026-05-18/`
- Stage 2 layer-hook + followup: `artifacts/reports/probes/bg_stage2_steering_2026-05-18/`, `artifacts/reports/probes/bg_stage2_layerhook_followup_2026-05-18/`
- Empirical direction + RMS + adapter probes: `artifacts/reports/probes/bg_empirical_steering_direction_2026-05-18/`, `artifacts/reports/probes/bg_preconsolidation_control_probes_2026-05-18/`
- Teacher-forced causal adapter: `artifacts/reports/probes/bg_causal_intervention_adapter_2026-05-18/`
- Sequence-level adapter (final): `artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/`

**Code:**
- `src/evaluator/bg_steering_hook.py`
- `src/evaluator/bg_causal_adapter.py`
- `src/evaluator/bg_sequence_adapter.py`
- `src/evaluator/bg_controller.py`, `src/evaluator/bg_transformer_features.py`

**Docs:**
- `docs/evaluator/trajectory-prediction-sweep.md`
- `docs/evaluator/causal-intervention-adapter.md`
- `docs/evaluator/sequence-level-adapter.md`
- `docs/evaluator/current-state.md`, `docs/evaluator/domain-transfer-ledger.md`

## 13. v9 synthesis trigger

This consolidation should be folded into a v9 synthesis when Phase 2 design is chosen and the first Phase 2 experiment is specified. Until then, v8.1 remains the canonical Phase 1 spec, the v8.1 addendum tracks progress since v8.1, and this document locks the inference-time steering closure. v9 should formally incorporate: Phase 1.5 (branch allocation) as a validated tier, the inference-time-steering closure, the geometry findings, and the corrected Phase 2 framing (geometry alignment / write-path installation, backbone unfrozen).

## Same-prefix hidden-state branch generation suite (2026-05-18)

Report: `docs/evaluator/hidden-state-branch-generation.md`

Artifacts: `artifacts/reports/probes/bg_hidden_state_branch_generation_2026-05-18/`

Verdicts:

- `BG_HIDDEN_BRANCH_FEASIBILITY_VERDICT = HOOK_HIDDEN_ORIGIN_READY`
- `LIVE_BRANCH_METHOD = hook_intervention_per_branch`
- `BG_HIDDEN_BRANCH_GENERATION_VERDICT = HOOK_HIDDEN_ORIGIN_BRANCHES_GENERATED`
- `BG_LATENT_BRANCH_PERSISTENCE_VERDICT = LATENT_BRANCHES_PERSIST_TO_47`
- `BG_HIDDEN_BRANCH_OUTCOME_DATASET_VERDICT = READY`
- `BG_HIDDEN_ORIGIN_BRANCH_SELECTION_VERDICT = NO_HIDDEN_BRANCH_SELECTION_SIGNAL`
- `BG_HIDDEN_BRANCH_L30_L42_GATE_VERDICT = NEEDS_STRONGER_BRANCH_GENERATOR`
- `BG_HIDDEN_BRANCH_ADAPTIVE_THRESHOLD_VERDICT = TOPK_SUFFICIENT`
- `PHASE2_HIDDEN_BRANCH_READINESS = NEEDS_BETTER_BRANCH_EVALUATOR`

Interpretation:

This suite does not reopen the frozen-backbone action-steering closure. It tests branch generation and selection evidence only. Same-prefix hook-hidden-origin branches can persist geometrically and sometimes produce different outcomes, but frozen BG taps did not select good hidden-origin branches better than random. True fork/carry still needs branch-aware Ouro cache/state handling.
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

Report: `docs/evaluator/bg_merged_weight_branch_content_taps_v1.md`.

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
The L30/L42 convergence-hair replay did not clear hard-merge readiness; use hairs as
soft diagnostics only for now. Terminal confidence/defer remains required, especially on
reasoning hard slices, and science needs a different branch-generation recipe before it
can be treated as a headline steering domain. Branch classification remained
diagnostic-only; no steering, routing change, compute-savings claim, or fork/carry claim
was introduced.

Report: `docs/evaluator/bg_dualanchor_convergence_hairs_reasoning_science_v1.md`.

## DualAnchor science branch recipe and reasoning terminal defer v1 (2026-05-31)

`DUALANCHOR_SCIENCE_RECIPE_REASONING_DEFER_STATUS = PRE_STEERING_READY_WITH_SCIENCE_DIAGNOSTIC`.
Phase 2b steering comparison can proceed with reasoning as the headline domain and
science as a diagnostic slice only. The locked baseline remains v3 DualAnchor looped
survival with confidence/defer terminal handoff and soft-only convergence hairs. Science
headline readiness needs regenerated branch-recipe improvement or a validated parser
patch. No steering was run in this probe.

Report: `docs/evaluator/bg_dualanchor_science_branch_recipe_reasoning_defer_v1.md`.
