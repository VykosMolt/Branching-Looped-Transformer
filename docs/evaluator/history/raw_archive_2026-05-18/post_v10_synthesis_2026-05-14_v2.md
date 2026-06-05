# Post-v10 synthesis — Experiment 2 spec + Ouro-RLTT-BG architecture (v2, locked)

**Date:** 2026-05-14
**Status:** Locked plan after two-pass review with GPT-5.5 Pro. Supersedes the 2026-05-14 draft of the same name.
**Scope:** Everything decided after v10 of the locus memo. The memo itself ends at v10; this document picks up there and captures the L1 ablation result, the α-sweep, the Experiment 2 architecture decision, and the Ouro-RLTT-BG continued-training design — with two rounds of external review folded in.
**Reframing relative to v1:** the project is no longer "find the best loop/layer to read preference from." The project is "build a relational, pairwise, multi-tap branch-selection controller, validated on externally-labeled tournaments over Ouro-RLTT-generated branches." HH centered accuracy is a diagnostic, not the headline metric.

---

## 0. The reframing

The original framing was: *Can a pairwise evaluator read preference from Ouro loop states?*

The current framing is: *Can a looped model expose relational branch-selection signals at multiple internal depths, and can those signals become a read-only basal-ganglia controller for latent thought/action selection?*

Three sentences worth banning from the codebase and the next paper:

1. "`score(x)` means quality." It does not. The object is `score(a, b)`, ideally debiased as `score(a, b) − score(b, a)`. The evaluator is a comparator, not a judge.
2. "The evaluator's preference signal." There is no preference signal in single-candidate hidden states (the 21.75% below-chance independent probe in the original CLT paper established this). What exists is a *relational* signal between candidates.
3. "Loop 2 is special." Refuted by v6 onwards. All four loops carry comparable signal under proper readout; v10 settles that L2-L4 are functionally one state and L1 is a different state.

The basal-ganglia metaphor is appropriate only when implemented as selection/gating over alternatives, not judgment over individuals. Early tap proposes/prunes, mid tap detects uncertainty/disagreement, late tap selects. Not "average all taps and hope."

---

## 1. L1 ablation result (2026-05-14)

**Script:** `utilities/tests/manual/probe_l1_ablation.py`, zero-shot, swap-balanced, fp32, no model forward, 8.9 s wall.

| variant | canonical | centered | flip |
|---|---:|---:|---:|
| L4-only (baseline) | 0.950 | 0.595 | 0.175 |
| L1-only | 0.915 | 0.575 | 0.180 |
| L2-only | 0.950 | 0.600 | 0.155 |
| mean(L1, L4) | 0.940 | **0.620** | 0.150 |
| natural[L1-4] (ref) | 0.950 | 0.595 | 0.180 |

No degeneracy (flip rates 0.15–0.18, well clear of the constant-output failure mode).

**Verdict:** L1 carries readable relational signal → Experiment 2 should add a fused (L1, L4) arm.

The verdict's two legs are not equally strong:

- **Weak leg.** L1-only centered 0.575 clears the ≥ 0.55 bar but only just, below both L4 (0.595) and L2 (0.600).
- **Strong/load-bearing leg.** mean(L1, L4) centered 0.620 beats every single-loop variant by +2.0 to +4.5 pp, costing only 1 pp canonical. Fusion of the bipartite endpoints helps the antisymmetric signal — exactly the metric Experiment 2 optimizes.

Two v10 consistency checks held: L2-only ≈ L4-only (0.600 vs 0.595, confirming L2-L4 are one functional state) and natural[L1-4] = L4-only to 4 decimals.

### Mechanistic interpretation

L1 and L4 carry **partially independent relational signal**. If L1 were fully redundant with L4, averaging would equal max(L1, L4). If L1 were noise, averaging would hurt L4. Instead averaging *exceeds* both — L1 picks up a relational projection L4 has lost or smoothed, even though L1 alone is a weaker comparator. The bipartite trajectory is not "L1 different, L2-L4 same" but "L1 and L4 are complementary readouts of the same preference."

This tightens v10. v10 established L1 was geometrically distinct from the L2-L4 cluster; the ablation establishes the distinctness is useful, and specifically useful on the antisymmetric reading where Experiment 2 lives.

---

## 2. α-sweep result (2026-05-14)

Sweep over `α · h_L1 + (1−α) · h_L4` on the same 200 saved pairs, frozen head, ~10 s wall.

| α | canonical | centered | bias/sig |
|---:|---:|---:|---:|
| 0.00 | 0.950 | 0.595 | 1.416 |
| 0.30 | 0.945 | 0.595 | 1.468 |
| 0.35 | 0.950 | 0.615 | 1.472 |
| 0.45 | 0.945 | **0.620** | 1.474 |
| 0.50 | 0.940 | 0.620 | 1.472 |
| 0.55 | 0.930 | 0.610 | 1.468 |
| 1.00 | 0.915 | 0.575 | 1.404 |

**No α beats the unweighted mean's 0.620.** Peak gain over mean: 0.00 pp. The curve is a step (L4-dominated ≤ 0.30 at 0.595) up to a broad plateau (0.35–0.50 at ~ 0.615–0.620), then monotone decline toward L1. α = 0.50 reproduces 0.620 exactly, internal consistency confirmed.

### Interpretation

Falls into the "win isn't about weighting" branch. **A 2048-dim weighted mean cannot exceed the simple average → the +2.5 pp must come from the head seeing L1 and L4 as separate slots.** Concat/diff do structurally different work than any 2048-dim mixer.

- **bias_to_signal is flat across the sweep** (1.40–1.47). Weighting doesn't touch the symmetric-offset problem; λ_sym is doing orthogonal work. The two levers don't fight.
- **Plateau is broad** (0.35–0.50 all ≈ 0.615–0.620), not a peak — "the optimum is a region, not a tuned weight."

### Concat ≈ diff equivalence

`L4 − L1` is a linear combination of {L1, L4}, so `Linear(4096, …)` on `[L1; L4]` can represent `[L4; L4−L1]` exactly and vice-versa — same function class, same capacity. The diff parameterization isn't more informative; what it changes is inductive bias and optimization conditioning under LayerNorm (which is nonlinear and breaks the equivalence). Worth running both; expect close results; divergence > 1 pp is a LayerNorm conditioning story and is directly actionable for BG tap parameterization (see §4).

---

## 3. Experiment 2 — debiased evaluator retrain spec (pre-registered)

**Framing.** Experiment 2 produces a publishable evaluator result on its own. It is *not* the Phase-2 gate. HH centered_acc is a diagnostic for antisymmetric comparator health. The Phase-2 go/no-go is generated-branch tournament performance (§5).

### Backbone (frozen)

Ouro-2.6B-Thinking, loaded from `ByteDance/Ouro-2.6B-Thinking`. Frozen. bf16 forward (12 GB VRAM), states cast to fp32 on capture, written to disk, model unloaded before head training. Matches v10 numerical contract.

**What we keep per token:** L1 and L4 only. L2 and L3 are dropped (v10 + ablation confirm functional redundancy). Halves disk footprint at 50k examples.

### Input pipeline (the structural change)

Published head:
```
h_L4 [B, T, 2048] → AttentionPool(2048) → LayerNorm(2048) → Linear(2048, 512) → ...
```

Experiment 2 **primary arm (concat):**
```
h_L1 [B, T, 2048] ─┐
                    ├─ concat dim=-1 → x [B, T, 4096]
h_L4 [B, T, 2048] ─┘

x → AttentionPool(4096) → LayerNorm(4096) → Linear(4096, 512) → ...
```

4096-dim propagates only as far as the first Linear, which projects to 512. From there, head is identical to published.

**Parameter count:** AttentionPool +~2k; LayerNorm +2× scale/bias; Linear(4096, 512) 2 097 152 vs 1 048 576. Net ~+1.05M. Head grows from ~5M to ~6M total.

**Diff arm:** identical shape, concat replaced by `[h_L4; h_L4 − h_L1]`. Tests LayerNorm conditioning effect.

**Weighted-mean arm (demoted negative control):** stays 2048-dim, learned scalar α, init logit(0.5) = 0. Confirms 4096-dim arms aren't just rediscovering averaging.

**L4-only control:** published architecture verbatim, retrained with new objective. Isolates L1-fusion contribution from debiased-objective contribution.

### Head body (unchanged from published)

After first Linear → 512-dim, body is identical across arms. GRU's loop-axis dimension is 1 (one fused slot per pair member), not 4 — v9 per-loop GRU is collapsed by v10's bipartite finding + ablation. Head total: ~6M for concat/diff arms, ~5M for weighted-mean and L4-only.

### Pairwise scoring (new protocol)

```
s_cr = score(chosen, rejected)
s_rc = score(rejected, chosen)
```

Swap augmentation. Effective batch doubles per pair, same per-ordering compute as published.

### Loss

```
L_pref  = BCE(σ(s_cr), 1) + BCE(σ(-s_rc), 1)
L_sym   = (s_cr + s_rc)² · λ_sym             # symmetric-offset penalty
L_total = L_pref + L_sym.mean()
```

λ_sym swept over **{0.1, 0.3, 1.0}**.

### Metrics (logged every eval)

- `canonical_acc = P(s_cr > 0)` — published metric, diagnostic only.
- `centered_acc = P(s_cr − s_rc > 0)` — primary checkpointing metric.
- `bias_to_signal = |mean(s_cr + s_rc)| / std(s_cr − s_rc)` — target < 0.5.
- `flip_pos_rate`, `flip_strict_sign_reversal` — degeneracy diagnostics.
- `pos_rate`, `std(s_cr)`, `std(s_rc)` — distribution sanity.

Checkpoint by **best centered_acc on held-out HH test (8 552 examples)**.

### Training configuration

- Dataset: HH-RLHF, 50k train / 8 552 test (same split as published).
- States: from overnight Ouro-Thinking capture (L1, L4, fp32, per token).
- Optimizer: AdamW, lr 1e-4, weight decay 0.01.
- EBS = 32, GRAD_ACCUM_STEPS = 1.
- Epochs: 5, checkpoint per epoch. Expect epoch 2 best; 4-5 overfit.
- Precision: fp32 throughout head. AMP/bf16 breaks GRU.
- Seeds: 42 primary; 7 and 13 if borderline.

### Sweep design

4 arms × 3 λ_sym = **12 runs**, reported as a 4×3 matrix.

| Arm | λ_sym = 0.1 | λ_sym = 0.3 | λ_sym = 1.0 |
|---|---|---|---|
| concat [L1; L4] | (canon, centered, bias/sig) | … | … |
| diff [L4; L4 − L1] | … | … | … |
| weighted-mean (learned α) | … | … | … |
| L4-only control | … | … | … |

### Tiered success criteria (paper-grade evaluator)

**Frame:** these tiers grade Experiment 2 as a *publishable evaluator paper*. They do **not** decide Phase 2. A strong-tier HH result without Phase-2 gate firing (§5) is no-go for Phase 2 cloud.

- **Excellent:** centered ≥ 0.85 *and* bias/sig < 0.5
- **Strong:** centered ≥ 0.75 *and* bias/sig < 0.8
- **Useful-HH:** centered ≥ 0.68 *and* bias not dominant (bias/sig < 1.0). HH-side only; says nothing about BG.
- **Weak:** centered ≤ 0.64 or bias dominant

(A separate **Useful-for-BG** outcome — centered ≥ 0.68 *and* a generated-branch tournament gain over baseline — lives entirely in §5 and grades the BG architecture, not Experiment 2 as an evaluator paper.)

**Secondary findings worth recording regardless:**
- Concat vs diff gap (< 2 pp expected; > 1 pp in either direction informs BG tap design, §4).
- Weighted-mean vs concat/diff gap (> 1 pp in favor of 4096-dim arms confirms "separated slots").
- λ_sym sensitivity profile per arm.

### Pre-registered outcome probabilities (paper tier)

- Excellent: ~25%
- Strong: ~40%
- Useful-HH: ~25%
- Weak: ~10%

Down-weighted from the v1 synthesis's "60% all-green at the 0.85 threshold," which was unrealistically optimistic given the 0.604 → 0.85 jump.

### What's explicitly *not* in Experiment 2

- No per-loop GRU (collapsed by bipartite finding).
- No iterated norm K-sweep (v10 refuted gamma-attractor).
- No L2 or L3 states (redundant).
- No multi-tap-over-layers (lives in §4 BG architecture).
- No basal-ganglia controller (separate phase, §4).

---

## 4. Ouro-RLTT-BG — continued-training architecture

The new Ouro variant. Continues from Ouro-RLTT, bakes the evaluator into the forward pass at three layers (24, 36, 47) as a basal-ganglia-style controller. The evaluator becomes part of the model's compute graph at every forward pass.

This operationalizes v9's "Final evaluator insertion interface" — multi-tap controller for visibility, late tap makes the call — refined by v10 + L1 ablation + two rounds of review.

### Base backbone

**Ouro-RLTT** from `/home/moloch/ouro_project/models/ouro_rltt_local`. 48 shared layers, R = 4 loops, 2048 hidden dim.

**Freezing strategy:** LoRA rank 16 on attention and MLP projections at all layers, **full unfreeze at layers 24, 36, 47**.

LoRA preserves RLTT-trained competence at intermediate layers; tap-layer unfreeze lets them develop readable relational signal under auxiliary pressure.

### Evaluator taps

Three taps at layers 24, 36, 47. Each tap is the Experiment 2 winning architecture — reads (L1, L4) of *its own layer*, fuses via the Experiment 2 winning parameterization (concat or diff, decided by Experiment 2), produces a scalar score with antisymmetry pressure.

Per tap:
```
h_layer_l_loop_1 [B, T, 2048] ─┐
                                ├─ winning_fusion → x [B, T, 4096]
h_layer_l_loop_4 [B, T, 2048] ─┘

x → AttentionPool(4096) → LayerNorm(4096) → Linear(4096, 512) → ... → scalar score
```

Initialized from the **winning Experiment 2 checkpoint**. Three taps × ~6M = ~18M evaluator parameters; trivial vs 2.6B backbone.

**Winning fusion rule:**
- If Experiment 2 diff beats concat by ≥ 1 pp centered: taps use `concat(h_L4, h_L4 − h_L1)`.
- If Experiment 2 concat beats diff by ≥ 1 pp centered: taps use `concat(h_L1, h_L4)`.
- If gap < 1 pp: use concat as default (simpler, no implicit "refinement" framing).

### Open question before commit: layer 24/36 bipartite probe (BLOCKING)

The L1↔L4 bipartite finding was measured at the boundary (effectively layer 47 post-block). It is **not yet known** whether layers 24 and 36 have the same bipartite loop geometry. If they don't, the (L1, L4) parameterization at those taps may be wrong.

Same shape as v10's loop geometry probe but reading layer 24 and 36 hidden states. ~15 min compute, reuses infrastructure. **Must run before Phase 1 commits to the tap input shape.**

### Per-layer Thinking-vs-RLTT probe (BLOCKING)

v10 settled that boundary states are essentially identical between Thinking and RLTT on HH text (mean cos 0.9992). But intermediate layers (24, 36) might differ — the final norm equalizes the boundary, not the intermediate layers. The v9 layer recommendations were measured under Thinking; if intermediate layers differ between backbones, the v9 layer choices need re-validation on RLTT. ~30 min compute. **Must run before Phase 1.**

### Controller logic (read-only baked-in)

Three taps fire on every forward pass; scores are auxiliary outputs alongside logits.

```python
early_score = tap_24(h_24_L1, h_24_L4)   # filter / fork pressure
mid_score   = tap_36(h_36_L1, h_36_L4)   # consistency / disagreement
late_score  = tap_47(h_47_L1, h_47_L4)   # final selector

control_signal = {
    "early": early_score,
    "mid":   mid_score,
    "late":  late_score,
    "early_mid_disagreement": abs(early_score - mid_score),
    "mid_late_disagreement":  abs(mid_score - late_score),
    "confidence": abs(late_score),
}
```

**Read-only:** controller exposes signals; consumed externally by branch-selection / rollout / generation. Active steering (residual-stream modification) is a Phase 3+ follow-on, not in scope here.

### Training objective

```
L_total = L_LM + λ_eval · L_eval + λ_align · L_align    # λ_align = 0 in Phase 1
```

**L_LM** — standard next-token prediction on a pretraining mix. Initial proposal: FineWeb-Edu (1×), OpenWebMath (2×), NuminaMath-CoT (3×), StarCoder (1.5×). Validate against what RLTT base was actually trained on before committing.

**L_eval — mixed-domain relational loss** (this is the change from v1 of the synthesis):

```
L_eval = w_HH   · L_eval_HH
       + w_math · L_eval_math
       + w_code · L_eval_code
       + w_reas · L_eval_reasoning
```

Per channel, per tap:
```
L_eval_channel_l = BCE(σ(s_cr_l), 1) + BCE(σ(-s_rc_l), 1) + λ_sym · (s_cr_l + s_rc_l)²
```

**Channel data sources:**
- **HH:** HH-RLHF preference pairs, swap-augmented (Experiment 2 protocol).
- **Math:** Ouro-RLTT-generated attempts on GSM8K + verifier-clean MATH problems, paired correct-vs-incorrect, labels from exact-answer verifier.
- **Code:** Ouro-RLTT-generated patches on HumanEval / MBPP / similar, paired pass-vs-fail, labels from unit-test execution.
- **Reasoning:** Ouro-RLTT-generated attempts on selected reasoning benchmarks (specific datasets TBD), labels from answer keys.

**Critical:** math/code/reasoning pairs must be from *Ouro-RLTT's own generated attempts*, not static datasets. The deployment distribution is generated branches; static-dataset pairs don't match it.

**Channel weighting:** swept as hyperparameter. Default options:
- Equal per-channel batches (simple, may overweight smaller channels per token).
- Inverse-sqrt channel-size weighting (balances per-channel signal).
- Equal per-token (may overweight HH given availability).

**L_align — Phase 1: disabled (λ_align = 0).**

Phase 1 trains independent taps under L_eval only. Tap disagreement is measured empirically, not regularized. The four-way decomposition decides whether L_align enters Phase 2:

| Disagreement predicts | Implication | Phase 2 action |
|---|---|---|
| Branch ambiguity | Disagreement is metacognitive signal | No L_align; use disagreement as branch policy input |
| Late-tap error | Disagreement is corrective | No L_align; use disagreement as confidence correction |
| Need for more rollout | Disagreement is compute-allocation signal | No L_align; use disagreement as compute scheduler input |
| None (noise) | Disagreement is regularizable | Add L_align with `|s_cr_late|` weighting |

This measurement is part of Phase 1 evaluation, not a separate phase.

### Phase split

**Phase 1 — Local, frozen RLTT backbone, tap warm-start.**
- Capture (L1, L4) at layers 24, 36, 47 over HH train + math/code/reasoning generated-branch pairs.
- Train three independent tap heads with mixed-domain L_eval, no L_align, no LM loss.
- ~12h overnight per channel-weighting config.
- Outputs: three trained tap checkpoints, per-tap centered_acc per channel, agreement profile.

**Phase 2 — Cloud, full objective, joint training.**
- A100/H100 rental. Init from Phase 1 taps + Ouro-RLTT backbone.
- LoRA rank 16 everywhere, full unfreeze on 24/36/47.
- Full L_total objective. λ_align decided by Phase 1 disagreement analysis.
- 20-50B tokens continued pretraining on pretraining mix interleaved with preference pairs.
- Checkpoint by **generated-branch tournament accuracy + per-tap centered_acc + LM perplexity**.
- Cost: scale of the SOLAR-expansion plan in memory.

**Phase 3 — Controller policy.**
- Design branching / rollout / decoding policy that uses `control_signal`.
- Distinct enough to be its own design pass.

### Open questions before Phase 1 starts

1. **Layer-24/36 bipartite probe.** Blocking.
2. **Per-layer Thinking-vs-RLTT comparison.** Blocking.
3. **Generated-branch dataset construction.** See §5. Blocking for Phase 1 evaluation, not for Phase 1 training (training can start on HH-only and add channels as data lands).
4. **L_LM mix validation against RLTT base.** Blocking for Phase 2; not blocking for Phase 1.
5. **Phase 2 cloud budget.** Blocking for Phase 2 kickoff.

---

## 5. Generated-branch tournament dataset and Phase-2 gate

This is the single most important addition to v2 of the synthesis. The Phase-2 go/no-go is generated-branch tournament performance, not HH centered_acc.

### Dataset construction protocol

**Generator:** Ouro-RLTT (not Ouro-Thinking — the controller's deployment distribution is RLTT-generated branches).

**Generation parameters (fixed for reproducibility):**
- Temperature: 0.7
- Nucleus (top-p): 0.95
- Attempts per prompt: 2-4
- Keep prompt only if at least one correct *and* one incorrect attempt produced (otherwise can't form pair / tournament)

**Domains:**
- **Math:** GSM8K, verifier-clean MATH problems. Labels from exact-answer verifier. Restrict to canonical-answer problems for Phase 1; proof-style is out of scope until Phase 3.
- **Code:** HumanEval, MBPP. Labels from unit-test execution.
- **Reasoning:** Selected MCQ benchmarks (e.g., specific BBH subsets where verifier is unambiguous). Labels from answer keys.
- **Optional ARC/games:** environment score or solved-state progress (later phase; not in initial dataset).

**Critical: labels must be external/objective.** Not evaluator-derived. Not LLM-judge-derived (introduces preference correlation contamination from training distribution overlap). Verifier or execution only.

**Kept-prompt rate logged per domain** — diagnoses whether the model is too easy (mostly correct, few branch sets formed) or too hard (mostly wrong, can't pair) for each domain.

### Pilot vs gate-set sizes

- **Pilot (protocol shakedown):** 100-500 examples total. Used to validate protocol, generation parameters, label pipeline. Insufficient for a Phase-2 gate decision.
- **Phase-2 gate set:** **≥ 1000 tournaments total AND ≥ 300 per domain** that Phase 2 cares about. The per-domain floor is required because the disjunctive gate (below) includes a per-domain criterion (C), and per-domain CI must be meaningful.

### Evaluation metric panel (tournament-level)

Reported per controller variant and per domain:

1. **Top-1 tournament accuracy** — controller picks the correct branch in a tournament of K candidates.
2. **Condorcet winner rate** — fraction of tournaments where the controller's pick beats every other candidate in pairwise comparison.
3. **Cycle rate** — fraction of tournaments where the pairwise comparator produces preference cycles (A > B, B > C, C > A). High cycle rate is a structural failure mode invisible to centered_acc.
4. **Margin calibration** — does |late_score| predict correctness? Reliability diagram + ECE. Required because the BG controller policy uses |late| as confidence; if it's not calibrated, the policy is built on sand.
5. **Per-domain breakdown** — all of the above, split by math/code/reasoning.

### Phase-2 go/no-go gate (disjunctive)

**Proceed to Phase 2 if at least one of the following holds:**

**A. Generated-branch tournament gain.**
- Multi-tap or trained-tap controller improves over layer-47 single-tap baseline by ≥ 2-3 pp on top-1 tournament accuracy.
- Paired bootstrap CI on the within-prompt difference **excludes zero** (95% CI).
- Required n: ≥ 1000 paired tournaments.

**B. Disagreement informativeness.**
- Early/mid/late tap disagreement predicts at least one of:
  - Branch ambiguity (operationalized: tournament margin |best − second_best| inverse-correlated with disagreement).
  - Late-tap error (operationalized: disagreement classifies "late picked wrong" vs "late picked right").
  - Need for additional rollout (operationalized: disagreement correlates with "another sample would change the answer").
  - Improved tournament policy (operationalized: disagreement-conditioned controller beats unconditioned controller).
- AUC point estimate ≥ 0.65 *and* lower 95% CI bound ≥ 0.60 on the relevant predictive task.

**C. Mixed-domain transfer.**
- Trained taps improve at least two target domains on top-1 tournament accuracy.
- Per-domain bootstrap CI excludes zero in each of the two domains.
- No material degradation on the strongest pre-training domain.
- No material LM-sanity-check degradation.

**Explicit non-criterion: Do not proceed to Phase 2 solely because HH centered_acc is high.** HH centered_acc is a diagnostic for evaluator health, not a Phase-2 gate.

### Statistical hygiene across all three gates

- Gates A and C use paired bootstrap (same prompts through baseline and treatment), not independent bootstrap. Paired CIs are substantially tighter and the comparison structure supports them.
- Gate B uses AUC with bootstrap CI on the AUC estimate itself.
- All CIs reported at 95%. All metrics reported with bootstrap intervals, not just point estimates.
- If pilot data only (n < 1000): treat as directional evidence, not a gate. Lock the protocol, expand to gate scale, then decide.

---

## 6. Sequencing (revised)

1. **Layer-24/36 bipartite probe.** ~15 min. Blocking for BG tap parameterization.
2. **Per-layer Thinking-vs-RLTT comparison.** ~30 min. Blocking for confirming v9 layer choices on RLTT.
3. **Full-split Ouro-Thinking capture pass.** ~9h overnight. Unblocks Experiment 2.
4. **Experiment 2 sweep.** 12 runs, overnight after capture. Produces winning concat/diff parameterization and λ_sym choice. Publishable on its own; grades on the Experiment 2 tiered criteria.
5. **In parallel with 1-4: build generated-branch tournament dataset.**
   - Pilot first (100-500 examples) to lock generation parameters and label pipeline.
   - Then expand to gate scale (≥ 1000 total, ≥ 300 per domain).
   - Generator: Ouro-RLTT. Labels: external verifier/execution.
6. **Full-split Ouro-RLTT capture pass at layers 24/36/47.** ~12h overnight. After steps 1-2 confirm tap parameterization and layer choices.
7. **Ouro-RLTT-BG Phase 1.** Three parallel debiased tap retrains on RLTT states with mixed-domain L_eval, no L_align. Overnight per channel-weighting config.
8. **Phase 1 evaluation.** On HH (diagnostic) *and* generated-branch tournaments (primary). Includes the four-way disagreement decomposition that decides whether L_align enters Phase 2.
9. **Phase-2 go/no-go.** Apply disjunctive gate (§5). Cloud budget confirmed.
10. **Paper write-up: CLT v2.** Covers v10 + Experiment 2 + ablation + Phase 1 results. Independent of Phase 2.
11. **Phase 2 cloud run.** Gated on step 9.
12. **Phase 3 controller policy design.** Gated on Phase 2.

Steps 1, 2, 5 can run in parallel with each other. Step 5 (dataset construction) is the longest-lead item; start it immediately.

---

## 7. Open items / TODOs

- [ ] Run layer-24/36 bipartite probe (sequencing step 1).
- [ ] Run per-layer Thinking-vs-RLTT probe (step 2).
- [ ] Run full-split Ouro-Thinking capture pass (step 3).
- [ ] Execute Experiment 2 12-run sweep (step 4).
- [ ] **Build generated-branch tournament dataset** — pilot first, then gate-scale. Generator = Ouro-RLTT, labels = external verifiers/execution. **Long-lead item; start now.**
- [ ] Identify specific reasoning benchmarks for the reasoning channel (BBH subsets candidate).
- [ ] Validate RLTT-base pretraining mix against the proposed FineWeb-Edu + OpenWebMath + NuminaMath-CoT + StarCoder recipe.
- [ ] Get cloud-compute quote for Phase 2 (20-50B tokens, 2.6B backbone + LoRA + 3 unfrozen layers).
- [ ] Run full-split RLTT capture at layers 24/36/47 (step 6, after 1-2).
- [ ] Phase 1 tap retrain with mixed-domain L_eval (step 7).
- [ ] Phase 1 evaluation including four-way disagreement decomposition (step 8).
- [ ] Apply Phase-2 disjunctive gate (step 9).
- [ ] Commit this document to repo as `architecture_spec_v2_2026-05-14.md` once reviewed.

---

## 8. Summary

**The strongest current empirical claim:** Ouro contains trajectory-distributed relational evidence about candidate quality, readable across loop endpoints and raw layer taps, with L1 and L4 carrying complementary relational projections and late raw layers serving as the strongest final selectors *as measured on HH-RLHF text*. Transfer to Ouro-RLTT-generated branches is the immediate next empirical question.

**The current architecture target:** a relational, pairwise, multi-tap branch-selection controller. The controller does not judge candidate quality in isolation. It selects among alternatives, exposes early/mid/late disagreement as a possible control signal, and is validated on externally-labeled tournaments over Ouro-RLTT-generated branches. HH centered accuracy remains a unit test for antisymmetric comparator health, not the decision criterion for scaling to Phase 2.

**The biggest risks:**
- *Scientific:* HH-trained relational signal may not transfer to generated branches. Mitigation: build generated-branch dataset early; gate Phase 2 on tournament performance.
- *Engineering:* Phase 2 cloud cost; preserving LM competence while forcing tap readability. Mitigation: LoRA + selective unfreeze; per-tap centered_acc + LM perplexity as joint checkpointing criteria.
- *Conceptual:* Treating disagreement as noise rather than control signal (would collapse the multi-tap controller into a regularized single-tap). Mitigation: Phase 1 explicitly tests this with the four-way decomposition before L_align is added.
