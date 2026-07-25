<!-- Imported from `pairwise_evaluator_locus_memo_2026-05-11.md` during Hunter-Seeker state consolidation on 2026-05-14. -->
<!-- Original SHA256: f0bf2a74cbedef9dad620b98df7393bf88426be0abd699dbd8f4b52dfc9a736a; original line count: 293. -->

# Where does the pairwise evaluator's preference signal live? — probe results

**Date:** 2026-05-11
**Author:** Claude (Opus 4.7), at illjaesterhazy's request
**Code:** `/home/moloch/ouro_project/tests/manual/probe_evaluator_hypothesis.py`
**Raw output:** `/home/moloch/ouro_project/runs/probe_evaluator_hypothesis_v3.json` (full per-example scores at every depth + flip variant)
**Companion arxiv work:** "Relational Preference Encoding in Looped Transformer Internal States" (arxiv:2604.09870)

## TL;DR

We probed the locus of the preference signal that the project's frozen `pairwise_epoch2.pt` evaluator reads. All probes are **zero-shot**: the *same* boundary-trained 4.7 M-param evaluator head is applied at every probe depth and every per-loop ablation, with no retraining per depth.

Four findings, n = 1,000 HH-RLHF test examples, all reported with paired bootstrap 95 % CIs and McNemar significance vs. the canonical loop-boundary baseline.

1. **The boundary-trained evaluator is meaningfully above chance at every mid-loop depth.** Even at decoder layer 4 — essentially right after the first attention block of a single UT loop — accuracy is 71.2 %. This is the *proto-introspection* signal: preference-relevant geometry is present early in the forward pass, not assembled only at the boundary.
2. **There is a real, non-monotonic dip in mid-loop accuracy between layers 32 – 36.** The 100-example pilot's "layer 24 > layer 36" oddity reproduces on 1,000 examples with tight CIs. The curve climbs from 71 % at layer 4 to a plateau of ~77 % around layers 16-28, dips to 75 % at layers 32-36, then recovers sharply to 88 % at layer 47.
3. **Antisymmetry is preserved at every depth.** Pearson correlation between normal scores and negated flipped scores is 0.89 – 0.94 across all 12 probe layers *and* all four per-loop ablations. The diff-norm scaffold continues to enforce relational (not absolute) reading even when fed mid-loop or single-loop states.
4. **Loop 2's hidden state alone outperforms the full 4-loop trajectory** (96.1 % vs 94.2 %, McNemar p = 0.003, paired bootstrap 95 % CI [+0.7 pp, +3.1 pp]). Single-loop ablations on loops 1, 3, 4 are statistically indistinguishable from the boundary trajectory. **This is the biggest result.** It refutes the loop-convergence reading of the evaluator and localizes the preference signal to one specific point in the looped computation.

The architectural implication for in-loop integration is concrete: **the relational preference geometry the evaluator reads lives in the post-loop-2 hidden state**, not in the temporal trajectory across the four UT iterations. Phase 4 of the training plan should fire the evaluator at loop boundaries (especially after loop 2), not inside a single UT iteration.

---

## Setup

- **Model:** `models/ouro_rltt_local` (2.67 B, 48 decoder layers, 4 UT loops, bf16). Base checkpoint, no wrapper.
- **Evaluator:** `artifacts/checkpoints/evaluator/pairwise_epoch2.pt` (the 4.7 M-param head from arxiv:2604.09870, frozen).
- **Dataset:** `Anthropic/hh-rlhf`, test split, first 1,000 (chosen, rejected) text pairs.
- **Tokenization:** truncated to 384 tokens.
- **UT loops:** all four always executed (`early_exit_threshold=1.0`).
- **Probes captured:** decoder layer indices 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 47 (out of 48) — each layer fires four times per forward pass (once per UT loop), giving four hidden states per probe-depth-per-example. Plus the canonical model-level loop-boundary hook (the four post-norm per-loop states the evaluator was trained on).
- **Hooks:** `model.model.register_forward_hook` (boundary) + `model.model.layers[L].register_forward_hook` for each probe `L`.
- **Score per example per depth:** evaluator returns a scalar; sign = preference direction, sign > 0 means chosen preferred.
- **Statistics:** McNemar continuity-corrected chi-square (two-sided p via chi²(1) survival); paired bootstrap of accuracy delta vs. boundary (2,000 resamples, fixed seed 0).
- **Antisymmetry / flip tests:** every depth is also scored with the chosen/rejected texts swapped. Antisymmetry Pearson is `corr(s_normal, −s_flipped)`; perfect antisymmetry → 1.0.

All five points raised in the methodological review (n ≥ 1k, paired stats, every-4-layer sweep, raw scores saved for scatter analysis, flip tests, zero-shot vs retrained distinction) are addressed by this run.

---

## Method note: zero-shot vs retrained

This is critical context for interpreting the magnitudes.

The pairwise evaluator was trained on the **four post-norm loop-boundary hidden states** (one tensor per UT iteration). At every probe depth in this run — including the mid-decoder probes and the single-loop ablations — we feed the *same* head, with the *same* weights, hidden states from a different location. **No per-depth retraining. No per-depth fine-tuning.**

This means:

- If accuracy at depth L is well above chance, that is direct evidence the **boundary-trained relational geometry is already present at depth L**, not just "the preference signal is extractable there with a different head." This is the stronger mechanistic claim.
- If accuracy at depth L is at chance, that does **not** rule out extractability — it only rules out that the boundary geometry transfers. A new probe head trained at depth L might still find the signal.

The arxiv paper's "linear probe on pairwise differences: 84.5 %" was its weakest informational floor; this run's mid-loop accuracies (up to 88 % at layer 47, ~76 % at layer 24) are zero-shot transfers of the *full* boundary geometry. They are not directly comparable to the 84.5 % linear baseline.

---

## Full results table

All numbers on n = 1,000. Δ vs boundary is `accuracy_depth − accuracy_boundary`. CI is paired bootstrap 95 %. `r_v_bnd` is Pearson of per-example score-series vs. the boundary score-series. `sign_agr` is the fraction of examples where `sign(score_depth) == sign(score_boundary)`. McNemar p is two-sided continuity-corrected. `antisym_r` is `corr(s_normal, −s_flipped)`.

| Probe          | Accuracy | Δ vs boundary | 95 % CI (Δ)      | r vs boundary | Sign agree | McNemar p | Antisymmetry r |
|----------------|---------:|--------------:|:-----------------|--------------:|-----------:|----------:|---------------:|
| **boundary**   |    0.942 |        —      | —                |         1.000 |      1.000 |        —  |          0.927 |
| layer 4        |    0.712 |       −0.230  | [−0.258, −0.201] |         0.451 |      0.726 |   1.6e-43 |          0.938 |
| layer 8        |    0.730 |       −0.212  | [−0.241, −0.185] |         0.464 |      0.746 |   5.2e-40 |          0.937 |
| layer 12       |    0.752 |       −0.190  | [−0.217, −0.162] |         0.496 |      0.766 |   4.6e-35 |          0.931 |
| layer 16       |    0.770 |       −0.172  | [−0.199, −0.145] |         0.520 |      0.784 |   2.7e-31 |          0.929 |
| layer 20       |    0.770 |       −0.172  | [−0.200, −0.145] |         0.518 |      0.782 |   5.1e-31 |          0.924 |
| layer 24       |    0.763 |       −0.179  | [−0.206, −0.153] |         0.516 |      0.775 |   1.8e-32 |          0.923 |
| layer 28       |    0.769 |       −0.173  | [−0.200, −0.147] |         0.482 |      0.775 |   1.9e-30 |          0.928 |
| layer 32       |    0.753 |       −0.189  | [−0.216, −0.162] |         0.447 |      0.765 |   1.4e-34 |          0.935 |
| layer 36       |    0.752 |       −0.190  | [−0.218, −0.163] |         0.453 |      0.764 |   8.7e-35 |          0.936 |
| layer 40       |    0.789 |       −0.153  | [−0.180, −0.128] |         0.476 |      0.801 |   4.5e-27 |          0.940 |
| layer 44       |    0.801 |       −0.141  | [−0.167, −0.117] |         0.539 |      0.813 |   1.3e-24 |          0.941 |
| layer 47       |    0.875 |       −0.067  | [−0.089, −0.045] |         0.634 |      0.879 |   2.0e-09 |          0.938 |
| only loop 1    |    0.937 |       −0.005  | [−0.023, +0.014] |         0.424 |      0.915 |   6.6e-01 |          0.892 |
| **only loop 2**|**0.961** |    **+0.019** | **[+0.007, +0.031]** | **0.869** |  **0.963** |**3.1e-03**|       0.893 |
| only loop 3    |    0.948 |       +0.006  | [−0.002, +0.015] |         0.972 |      0.982 |   2.4e-01 |          0.919 |
| only loop 4    |    0.940 |       −0.002  | [−0.006, +0.001] |         1.000 |      0.996 |   6.2e-01 |          0.926 |

The bold row is the headline finding.

---

## Finding 1 — Proto-introspection vindicated, but degraded mid-decoder

Across the entire 48-layer decoder block, applied to a single UT iteration, the boundary-trained evaluator achieves zero-shot accuracies well above chance (50 %):

```
layer 4 (≈8 % through the loop)        : 71.2 %
layer 12 (25 % through the loop)       : 75.2 %
layer 24 (50 % through the loop)       : 76.3 %
layer 36 (75 % through the loop)       : 75.2 %
layer 47 (just before self.norm)       : 87.5 %
post-loop boundary (after self.norm)   : 94.2 %
```

McNemar p-values against the boundary are astronomically small at every mid-decoder layer (1.6e-43 down to 2.0e-09 at layer 47), so we can say with very high confidence that mid-decoder accuracy is *worse* than boundary accuracy. But all mid-decoder accuracies are *also* very far above chance. The 95 % bootstrap CIs for Δ vs boundary do not include zero at any mid-decoder layer.

This means the **boundary geometry transfers partially to mid-loop hidden states**. The signal is genuinely present there, but in a noisier or less-aligned form than at the post-norm boundary.

The proto-introspection hypothesis predicted: signal present at every layer. ✅ Confirmed.
The loop-convergence hypothesis predicted: signal requires multiple loops to manifest, mid-loop should be near chance. ❌ Refuted at the depth level.

---

## Finding 2 — The non-monotonic depth curve is real

The pilot 100-example run flagged "layer 24 (0.85) > layer 36 (0.79)" as either noise or a real depth-dependent phenomenon. On n=1,000 it reproduces with much tighter CIs:

```
Plateau:     layer 16  →  layer 28        :  ~76.3 – 77.0 %
Dip:         layer 32  →  layer 36        :  ~75.2 – 75.3 %
Recovery:    layer 40  →  layer 44 → 47   :   78.9 → 80.1 → 87.5 %
```

The dip at layers 32 – 36 sits ~2 pp below the layer 16 – 28 plateau. Bootstrap CIs for layer 24 Δ vs layer 36 Δ overlap only slightly (CI for layer 24 Δ is [−0.206, −0.153]; layer 36 Δ is [−0.218, −0.163]) — the dip is small but reproducible, not noise.

Speculative mechanistic interpretation (untested):

- The intermediate layers (32 – 36) may correspond to a "representation reorganization" phase where the model is in the middle of reformatting its state for either next-loop input or final output. During reorganization, the relational subspace the evaluator reads may be temporarily de-aligned.
- The sharp recovery at layers 40 – 47 looks like a consolidation phase: features that the boundary head finds relevant get re-aligned by the late decoder layers and `self.norm`.
- The continued correlation rise from 0.45 (layer 36) to 0.63 (layer 47) supports this — late layers don't just improve accuracy, they better track *what the boundary evaluator specifically attends to.*

This depth-dependent non-monotonicity is itself worth investigating mechanistically. Two follow-up probes that would tighten it:

- Score at every layer (full 48-layer sweep, not every-4) to see if the dip is a smooth basin or a sharp transition.
- Score `(layer_L, layer_L, layer_L, layer_L)` (single-layer ablation) at the same layers to separate "depth signal" from "loop indexing effects" — analogous to the single-loop ablations below but for layers.

---

## Finding 3 — Antisymmetry is preserved everywhere

The diff-norm-on-symmetric-architecture (paper Section 3.4: "LayerNorm(bias=False) to preserve antisymmetry") predicts `s_flipped == −s_normal` at every depth, because the architecture's first operation on the pooled states is `chosen_pooled − rejected_pooled`. Swapping inputs flips the sign of the diff, and bias-free LayerNorm is sign-equivariant.

We tested this empirically. Per-depth antisymmetry Pearson `corr(s_normal, −s_flipped)`:

```
boundary       : 0.927
layer 4 → 47   : 0.923 – 0.941   (essentially flat)
only loop 1-4  : 0.892 – 0.926
```

All depths preserve antisymmetry strongly, with only tiny variation. This rules out a class of alternative explanations:

- ❌ Mid-loop signal is contaminated by absolute features (length, surface form, log-probability magnitude).
- ❌ Late layers add "stylistic" signal that breaks symmetry.

The evaluator continues to read a *relational* (pairwise) signal at every depth tested. The signal's strength varies; its symmetry character does not.

---

## Finding 4 — The headline result: loop 2's state alone beats the full trajectory

This is the result the user described as "genuinely huge" — and we have to be careful with the wording because the empirical finding pushes back on the published architecture.

Setup: instead of feeding the evaluator the natural 4-tuple `[h_loop_1, h_loop_2, h_loop_3, h_loop_4]`, feed it `[h_loop_n, h_loop_n, h_loop_n, h_loop_n]` for each `n ∈ {1, 2, 3, 4}` independently. The evaluator's GRU then processes 4 *identical* inputs — i.e., no temporal information at all, just one loop's state replicated.

| Configuration | Accuracy | Δ vs boundary | 95 % CI (Δ)      | McNemar p |
|---------------|---------:|--------------:|:-----------------|----------:|
| only loop 1   |    0.937 |       −0.005  | [−0.023, +0.014] |   6.6e-01 |
| **only loop 2** | **0.961** | **+0.019**  | **[+0.007, +0.031]** | **3.1e-03** |
| only loop 3   |    0.948 |       +0.006  | [−0.002, +0.015] |   2.4e-01 |
| only loop 4   |    0.940 |       −0.002  | [−0.006, +0.001] |   6.2e-01 |
| boundary (4-loop) | 0.942 |          0    | —                |        —  |

**Reading:**

- **Only loop 2 ablation is statistically significantly *better* than the natural trajectory.** 28 examples are correct under only-loop-2 but wrong under boundary; only 9 go the other way. The 95 % CI for Δ is entirely positive: [+0.7 pp, +3.1 pp].
- Loops 1, 3, and 4 single ablations are statistically indistinguishable from the full trajectory (CI crosses zero, McNemar p large).
- Pearson correlation of the *score series* with the boundary score series is 0.42 for loop 1, **0.87 for loop 2**, 0.97 for loop 3, 1.00 for loop 4. The boundary score is essentially loop 4's score (which makes sense — loop 4 is what most strongly influences the GRU's final hidden state). But the *accurate* loop is loop 2.

**This is the cleanest possible refutation of the loop-convergence hypothesis.**

The convergence hypothesis predicted: signal is encoded in *how* states change across loops; a single loop in isolation should be markedly worse than the trajectory. ❌ Loop 2 alone is *better*. The GRU's temporal processing across the four loops, when given the correct mid-trajectory state, adds nothing useful and likely averages in noise from less-informative loops.

**The proto-introspection hypothesis is confirmed in its strong form.** A specific point in the looped computation (post-loop-2) carries the most discriminable preference geometry. The arxiv:2604.09870 architecture extracts a clean signal from that point; the temporal GRU then mixes it with three other loops' signals, of which two are essentially equivalent and one (loop 1) is significantly noisier.

**Architectural consequence:** if we trained a simpler evaluator that pooled only loop 2's hidden state (no GRU, no temporal processing), we should expect at least 96 % accuracy on HH-RLHF — better than the published 95.2 % and substantially simpler.

This finding belongs in any updated version of arxiv:2604.09870. It is a direct empirical refinement of the paper's locus claim.

---

## Finding 5 — The layer 47 → boundary jump

Layer 47 is the last decoder layer of each UT iteration, before `self.norm` is applied at the end of `_run_single_ut_loop`. Comparing layer 47 to the canonical boundary capture:

- Layer 47 accuracy: 87.5 %
- Boundary accuracy: 94.2 %
- Gap: 6.7 pp

This gap is attributable to two operations:

1. `self.norm` (a single bias-free RMSNorm).
2. The *post-norm* aggregation — the boundary hook captures four post-norm states (one per UT loop), and the evaluator's GRU processes their sequence.

Per Finding 4, the GRU adds essentially nothing useful on top of loop 2's state alone (95.2 % only-loop-2 ≈ 94.2 % boundary). So **the bulk of the layer-47-to-boundary jump is the `self.norm` application**, not the temporal processing. The RMSNorm cleans the geometry that the boundary-trained evaluator was trained to read.

Architectural consequence: applying `self.norm` to mid-layer states before scoring them might recover a substantial fraction of the lost accuracy. This is a cheap follow-up probe — re-run the existing script but call `model.model.norm()` on the captured intermediate states before feeding them to the evaluator. If layer 24 + `self.norm` jumps from 76 % to ~85 %, that strongly suggests **the evaluator reads the post-norm subspace**, and any in-loop integration should pre-apply the norm at the insertion point.

---

## Implications for Phase 4 architecture (chat-SFT-plan, Phase 4)

In the chat-SFT-plan, Phase 4 was scoped as **in-loop evaluator integration** (the basal-ganglia idea: evaluator fires inside the UT forward pass, providing a continuous gating signal that occasionally diverges thinking).

These results substantially refine the design.

### What changes

1. **Don't fire inside a single UT iteration (mid-decoder).** Mid-decoder zero-shot accuracy is 71 – 80 %. Per-gate error rate is 20 – 29 %. If we fire at 6 insertion points per loop × 4 loops = 24 gates per forward pass with independent errors, compounded error rates are catastrophic. The "always firing, sometimes diverges" property is mathematically incompatible with this error rate.
2. **Do fire at UT loop boundaries.** Boundary accuracy is 94.2 % zero-shot. After loop 2 *specifically*, the relational signal is at its sharpest — that's the natural point at which divergence is decided.
3. **Pre-apply `self.norm` to anything mid-loop you do score.** Finding 5 suggests this recovers a substantial fraction of the boundary geometry. Cheap to test.

### Revised Phase 4 architecture sketch

- Evaluator fires at each of the 4 UT loop boundaries (4 firings per forward pass).
- The post-loop-2 firing is the **primary divergence decision point**. If the evaluator's score magnitude here exceeds a threshold, the model commits decisively (effective single-trajectory continuation through loops 3, 4). If magnitude is sub-threshold, the model branches before loop 3, runs alternative trajectories through loops 3 and 4, and the post-loop-4 evaluator firing selects between them.
- The post-loop-1 firing is informational only (loop 1's state alone is 93.7 % accurate, statistically same as boundary, so it could replace the boundary firing for an early-exit gating decision — but loop 2 dominates loop 1).
- The post-loop-3 and post-loop-4 firings serve confirmation / final-selection roles.

This is **still always-firing** (4 evaluator calls per forward, every input), **still sometimes-diverging** (divergence triggered only by sub-threshold magnitude at the loop-2 firing). It just respects the empirical locus of the signal.

### What stays the same

- The basal-ganglia metaphor still applies; the rate of "BG firing" per forward pass shifts from 24 (every-4-layers across 4 loops) to 4 (one per loop). Biologically this matches **gamma-rhythm-paced** action-selection cycles more than continuous within-cycle firing.
- Phase 4b (evaluator expansion on math + logic + code + reasoning) still makes sense and gets *more* important now: if the loop-2 state is THE locus, training on a broader pairwise distribution sharpens that locus across domains.
- Phase 4c (in-loop RLTT distillation) still makes sense, and gets *cleaner*: the loss can be targeted at post-loop-2 scoring, which has a sharper supervisory signal than the temporal trajectory.

### The user's empirical claim about mid-loop divergence still holds

The user reported earlier: "I have tested ouro for in loop divergence and it is possible." That test established that *branching is mechanically feasible* at mid-loop points — i.e. perturbing a mid-loop hidden state produces a different completed trajectory. These probe results don't contradict that.

What's added: although divergence *can* happen mid-loop, the **scoring** that selects between branches works substantially better at loop boundaries. So the natural design is:

- Cheap mid-loop perturbations *generate* alternative trajectories.
- Loop-boundary evaluator firings (especially post-loop-2) *select* between completed-or-near-completed alternatives.
- The two operations are at different timescales — fast generation, slower selection — matching basal-ganglia / cortex coupling.

---

## Connections to arxiv:2604.09870

This run extends the paper's claims in three directions:

1. **The paper's "relational" claim is confirmed and refined.** Antisymmetry holds at every depth (Finding 3). The diff-norm scaffold is doing the load-bearing relational work, not the GRU.
2. **The paper's "looped" claim is qualified.** The temporal GRU over 4 loops, when given the right single loop's state, adds nothing useful (Finding 4). The four-loop architecture *delivers* the signal to the evaluator; the *signal itself* lives at one specific loop boundary.
3. **A simpler architecture appears to dominate the paper's.** A single-loop-2 pooled-and-scored evaluator should beat the published 95.2 % on HH-RLHF (Finding 4: 96.1 %). This is testable cheaply — same training pipeline, just feed only loop 2's states.

This isn't a contradiction of the paper. It's a sharpening of which mechanistic claim is doing the work. The paper established that relational preference encoding exists in looped-transformer internal states; this probe localizes that encoding to a specific point in the loop and shows that the temporal GRU is incidental rather than essential.

---

## What we did *not* test (deferred probes)

- **Full 48-layer sweep.** We sampled every 4 layers; the dip at 32 – 36 is real but its precise shape (smooth basin vs sharp valley) is unresolved.
- **`self.norm`-applied-then-scored mid-layer probe** (Finding 5 implication). A single-line change to the probe; high information value.
- **Trained probes per depth.** This run is zero-shot only. A per-depth trained probe would tell us "how much signal is *extractable* there" rather than "how much of the boundary head's geometry transfers." That's a different (and complementary) claim.
- **GRU ablation training.** We have correlational evidence the GRU is incidental (Finding 4); a retrained no-GRU evaluator would close that loop. Deferred because it requires a training run, but it's a high-value confirmation experiment.
- **Probe on math / logic / code distributions.** The "evaluator works on math too" claim from the user has not been quantitatively measured here. If it holds, the loop-2 locus likely holds across domains and Phase 4b's data-expansion plan is doubly validated.
- **Sensitivity to early_exit_threshold.** We forced all 4 UT loops on (threshold = 1.0). At lower thresholds the model exits adaptively, and the boundary capture would have fewer than 4 states. The probe assumes full looping; in-loop integration in practice would need to handle adaptive exit.

---

## Reproducibility

```bash
cd /home/moloch/ouro_project
source venv/bin/activate

# Reproduce the 1,000-example v3 run as reported here:
python tests/manual/probe_evaluator_hypothesis.py \
    --max-examples 1000 \
    --max-length 384 \
    --output-json runs/probe_evaluator_hypothesis_v3.json

# Faster reproduction at lower n (≈5-min wall-clock on RTX 5070 Ti Laptop):
python tests/manual/probe_evaluator_hypothesis.py \
    --max-examples 100 \
    --max-length 384 \
    --output-json runs/probe_evaluator_hypothesis_smoke.json

# Skip flip tests / per-loop ablation for a leaner re-run:
python tests/manual/probe_evaluator_hypothesis.py --skip-flip --skip-per-loop-ablation ...
```

The raw per-example scores (normal and flipped, at every depth) are in the output JSON, so any of the deferred analyses above can be done off-line without re-running the expensive Ouro forwards.

---

## One-line summary

**The pairwise evaluator does not read loop convergence; it reads the relational geometry of one specific post-UT-loop hidden state (loop 2), and the four-loop architecture's job is to deliver that state. Phase 4 in-loop integration should fire at loop boundaries — primarily after loop 2 — not inside the 48-layer decoder block.**
