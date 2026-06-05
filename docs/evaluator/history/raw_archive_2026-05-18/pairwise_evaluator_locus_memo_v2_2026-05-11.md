# Pairwise evaluator locus — v2 (post-ablation-table, with v4-redo, v5–v9, and v10 cross-backbone update)

**Date:** 2026-05-11 (multi-pass, same day) — extended through 2026-05-14 with the loop-geometry / cross-backbone chapter (v10)
**Author:** Claude (Opus 4.7)
**Code:**
- `utilities/evaluator/probes/probe_evaluator_hypothesis.py` (v3 — first probe pass, depth sweep + per-loop ablation + flip)
- `utilities/evaluator/probes/probe_evaluator_v4_ablations.py` (v4 — 26-config ablation table)
- `utilities/evaluator/probes/probe_evaluator_v4_redo_masked.py` (v4-redo — degeneracy verification + iterated-norm + positional bias)
- `utilities/evaluator/probes/probe_loop_geometry_hh.py` (v10 — Experiments 1A + 1B, Thinking vs RLTT loop-state geometry on HH text)
**Raw output:**
- `artifacts/reports/evaluator/probe_evaluator_hypothesis_v3.json` (depth sweep + flip)
- `artifacts/reports/evaluator/probe_evaluator_v4_ablations.json` (full 26-config table)
- `artifacts/reports/evaluator/probe_v4_redo.json` (degeneracy verification + iterated-norm sweep)
- `artifacts/reports/evaluator/probe_loop_geometry_hh.json` (v10 — geometry numbers + four verdicts)
- `artifacts/reports/evaluator/hh_loop_states_200_thinking.pt` (v10 — captured per-token Thinking states, fp32, 2.5 GB)
- `artifacts/reports/evaluator/hh_loop_states_200_rltt.pt` (v10 — captured per-token RLTT states, fp32, 2.5 GB)
**Predecessor:** `pairwise_evaluator_locus_memo_2026-05-11.md` (v1 — the probe-1 findings this extends)
**Paper:** arxiv:2604.09870 "Relational Preference Encoding in Looped Transformer Internal States"
**Subsumed standalone notes:** `evaluator_domain_transfer_notes.md` (the loop-geometry chapter was originally drafted there; merged into this memo as v10 on 2026-05-14).

---

## Revision note (2026-05-12)

This memo intentionally preserves the earlier v4/v5 interpretation text below, including the now-known faulty reading of `sign_flip_rate`. The later addendum at the end supersedes it. Methodological honesty matters here: the path was "accuracy climbed, we interpreted falling sign-flip-rate as cleaner antisymmetry, then centered/debiased probes showed that interpretation was backwards." Do not silently erase the mistake.

New scripts/results added after the original memo:

- `utilities/evaluator/probes/analyze_pairwise_bias_decomposition.py`
- `utilities/evaluator/probes/probe_hh_norm_mechanism_centered.py`
- `utilities/evaluator/probes/probe_branch_selection_sim.py`
- `utilities/evaluator/probes/probe_evaluator_hendrycks_math.py` updated to fall back to `EleutherAI/hendrycks_math/*`
- `artifacts/reports/evaluator/bias_decomposition_final.json`
- `artifacts/reports/evaluator/probe_hh_norm_mechanism_centered.json`
- `artifacts/reports/evaluator/probe_evaluator_hendrycks_math.json`
- `artifacts/reports/evaluator/probe_branch_selection_sim.json`

---

## TL;DR (final, after three probe passes)

Three zero-shot probes on 1,000 HH-RLHF test pairs, evaluating the boundary-trained `pairwise_epoch2.pt` (94.2 % canonical accuracy) on a battery of input configurations. The headline keeps moving up.

1. **New best single configuration: `loop2_quadnorm` at 97.9 % accuracy.** Loop 2's boundary state with `OuroRMSNorm` applied **four** times, replicated 4× into the evaluator. Beats `loop2_triplenorm` (97.1 %), `loop2_doublenorm` (96.8 %), `only_loop_2` (96.1 %), and boundary (94.2 %). Δ = +3.7 pp vs boundary. **No fixed point reached yet** — accuracy is still climbing at four norm iterations, with std falling (1.057 → 0.712) and sign-flip-rate dropping (0.291 → 0.091). Iterating the RMSNorm keeps revealing cleaner geometry.
2. **The "loop 2 is uniquely special" claim weakens with proper norming.** Applying double-norm to every loop individually: loop 1 → 96.6 %, loop 2 → 96.8 %, loop 3 → 94.8 %, loop 4 → 95.2 %. Loop 2 retains a marginal advantage over loop 1 (+0.2 pp) but the gap is much smaller than the un-normed comparison (loop 2 alone 96.1 % vs loop 1 alone 93.7 % was +2.4 pp). **Most of the "loop 2 wins" effect was a norm-alignment artifact masking similar latent-quality across loops 1, 2, 4. Loop 3 is the laggard.**
3. **The GRU is mildly counterproductive.** Mean-pool-over-loops (`mean_all_replicated`, no GRU) hits 95.5 % vs boundary's 94.2 % (Δ = +1.3 pp, McNemar p = 0.031). Simple mean beats the trained 2-layer GRU on the same per-loop pooled inputs. Mid-layer self-norm partially recovers raw layer accuracy (5-9 pp).
4. **Positional bias is essentially nil.** Putting any loop's state at position 1 with mean-loops context produces 95.6 % regardless of which loop (1, 2, 3, or 4). Position-of-input doesn't matter to the evaluator.
5. **Degeneracy formally confirmed via flip test.** The masked-zero variants (`mask_zero_only_loop2`, `mask_zero_loop2_front`) reported 100 % accuracy on n=1,000 in v4. The v4-redo flip-test shows flip_pos_rate=1.000 and sign_flip_rate=0.000 — the architecture outputs the SAME positive sign for both `(chosen, rejected)` and `(rejected, chosen)`. Not signal. Bias-degeneracy that exploits HH-RLHF's chosen-first ordering convention. **Methodological lesson logged: always check pos_rate, std, and flip behavior alongside accuracy.**

The updated working architecture for Phase 4 in-loop integration: **fire the evaluator at end-of-loop-2, apply OuroRMSNorm four times to the boundary state, replicate into 4 slots, score.** No GRU, no temporal aggregation, no multi-loop input needed. Expected accuracy: ≥ 97.9 % (this is what we measured zero-shot; a retrain on the same architecture should match or exceed).

---

## Setup (delta from v1)

Same model, same evaluator checkpoint, same dataset, same n = 1,000, same statistics apparatus. v4 changes:

- **5 probe layers** (4, 12, 24, 36, 47) instead of the every-4 sweep — to free compute for the 21 new non-layer configurations.
- **26 configurations scored per example pair**, all zero-shot (same frozen `pairwise_epoch2.pt` head everywhere).
- **`model.model.norm`** (`OuroRMSNorm(2048,)`) is now exposed for ad-hoc normalization of mid-loop or boundary states before scoring.
- **Flip tests** on 7 subset configurations (boundary, only_loop_2, layer_24, layer_24_normed, layer_47_normed, mean_all_replicated, loop2_doublenorm). Antisymmetry Pearson computed for each.

Wall-clock: ~25 min on the RTX 5070 Ti Laptop (8 GB peak VRAM in bf16).

---

## Full ablation table

Sorted by accuracy, descending. Δ vs boundary is `accuracy_config − accuracy_boundary`. CI is paired bootstrap 95 %. `r_vs_bnd` is Pearson of per-example score-series vs boundary's. McNemar is two-sided continuity-corrected.

| Configuration                | Accuracy | Δ vs boundary | 95 % CI (Δ)       | r vs boundary | McNemar p | Antisym r |
|------------------------------|---------:|--------------:|:------------------|--------------:|----------:|----------:|
| ⚠ loop2_then_zeros          | **1.000**|       +0.058  | [+0.044, +0.073]  |        +0.128 |    7.2e-14|        —  |
| ⚠ mask_zero_only_loop2      | **1.000**|       +0.058  | [+0.044, +0.073]  |        +0.077 |    7.2e-14|        —  |
| ⚠ mask_zero_loop2_front     | **1.000**|       +0.058  | [+0.044, +0.073]  |        +0.128 |    7.2e-14|        —  |
| **loop2_doublenorm**         | **0.968**|       +0.026  | [+0.014, +0.038]  |        +0.857 |    7.7e-05|     0.871 |
| only_loop_2                  |    0.961 |       +0.019  | [+0.007, +0.031]  |        +0.869 |    3.1e-03|     0.893 |
| trunc_to_loop2  [h1,h2,h2,h2]|    0.959 |       +0.017  | [+0.006, +0.029]  |        +0.872 |    6.8e-03|        —  |
| mean_all_replicated          |    0.955 |       +0.013  | [+0.002, +0.024]  |        +0.901 |    3.1e-02|     0.894 |
| mean_loops_23_replicated     |    0.952 |       +0.010  | [+0.001, +0.019]  |        +0.944 |    6.6e-02|        —  |
| mean_loops_234_replicated    |    0.951 |       +0.009  | [+0.001, +0.018]  |        +0.975 |    6.6e-02|        —  |
| pair_1_2_alt   [h1,h2,h1,h2] |    0.950 |       +0.008  | [−0.003, +0.019]  |        +0.876 |    2.2e-01|        —  |
| only_loop_3                  |    0.948 |       +0.006  | [−0.002, +0.015]  |        +0.972 |    2.4e-01|        —  |
| pair_2_3_alt   [h2,h3,h2,h3] |    0.945 |       +0.003  | [−0.005, +0.011]  |        +0.973 |    6.3e-01|        —  |
| **boundary** (canonical)     |    0.942 |        —      | —                 |        +1.000 |        —  |     0.927 |
| layer_47_normed (= boundary) |    0.942 |       +0.000  | [+0.000, +0.000]  |        +1.000 |    1.0e+00|     0.927 |
| pair_2_4_alt   [h2,h4,h2,h4] |    0.942 |       +0.000  | [−0.003, +0.003]  |        +1.000 |    4.8e-01|        —  |
| start_from_loop2 [h2,h3,h4,h4]|   0.941 |       −0.001  | [−0.004, +0.002]  |        +1.000 |    1.0e+00|        —  |
| only_loop_4                  |    0.940 |       −0.002  | [−0.006, +0.001]  |        +1.000 |    6.2e-01|        —  |
| only_loop_1                  |    0.937 |       −0.005  | [−0.023, +0.014]  |        +0.424 |    6.6e-01|        —  |
| layer_47 (raw, pre-norm)     |    0.875 |       −0.067  | [−0.089, −0.045]  |        +0.634 |    2.0e-09|        —  |
| layer_36_normed              |    0.845 |       −0.097  | [−0.120, −0.075]  |        +0.630 |    3.7e-15|        —  |
| layer_24_normed              |    0.815 |       −0.127  | [−0.151, −0.101]  |        +0.530 |    1.2e-20|     0.899 |
| layer_12_normed              |    0.798 |       −0.144  | [−0.170, −0.119]  |        +0.467 |    3.6e-23|        —  |
| layer_4_normed               |    0.776 |       −0.166  | [−0.194, −0.139]  |        +0.457 |    8.5e-28|        —  |
| layer_24 (raw)               |    0.763 |       −0.179  | [−0.206, −0.153]  |        +0.516 |    1.8e-32|     0.923 |
| layer_12 (raw)               |    0.752 |       −0.190  | [−0.217, −0.162]  |        +0.496 |    4.6e-35|        —  |
| layer_36 (raw)               |    0.752 |       −0.190  | [−0.218, −0.163]  |        +0.453 |    8.7e-35|        —  |
| layer_4 (raw)                |    0.712 |       −0.230  | [−0.258, −0.201]  |        +0.451 |    1.6e-43|        —  |

⚠ = numerical degeneracy, not signal (see methodological note below).

---

## Finding 1 — Loop-2 + double-norm is the new best zero-shot configuration (0.968)

The `loop2_doublenorm` configuration applies `model.model.norm` (OuroRMSNorm, eps=1e-6) to the loop-2 boundary state once **on top of** the norm already applied inside `_run_single_ut_loop` to produce that boundary state, then replicates the result 4× into the evaluator's expected input shape.

Results:

| Config             | Accuracy | Δ vs boundary | 95 % CI       | McNemar p | n_config_correct_boundary_wrong | n_boundary_correct_config_wrong |
|--------------------|---------:|--------------:|:--------------|----------:|--------------------------------:|--------------------------------:|
| loop2_doublenorm   |    0.968 |       +0.026  | [+0.014, +0.038] |   7.7e-05 |                              35 |                               9 |
| only_loop_2 (v1)   |    0.961 |       +0.019  | [+0.007, +0.031] |   3.1e-03 |                              28 |                               9 |
| boundary           |    0.942 |        0      | —             |        —  |                              —  |                              —  |

Interpretation: loop 2's already-normed boundary state still carries some residual non-aligned components that a second `OuroRMSNorm` further suppresses. The geometry the evaluator was trained to read appears to be a "twice-canonicalized" RMSNorm subspace; the original training pipeline never exposed it to the loop-2 state in this form, but it transfers cleanly.

This sharpens the v1 thesis. The earlier statement was:

> *The preference geometry peaks at the loop where Ouro's recurrent computation effectively terminates for the given input.*

The corrected version is:

> *The preference geometry peaks at the loop where Ouro's recurrent computation effectively terminates, after RMSNorm alignment. For HH-RLHF this is post-loop-2; an additional explicit norm sharpens it further.*

The +2.6 pp improvement over boundary is small absolute but distributionally large — McNemar identifies 35 examples that flip from "boundary wrong" to "loop2_doublenorm correct," against just 9 the other way. These are likely cases where the boundary trajectory's GRU averaging mixed in noise from loops 3 / 4.

---

## Finding 2 — The GRU is mildly counterproductive

GPT raised this as a hypothesis to test. The result on three mean-based configurations:

| Config                       | Accuracy | Δ vs boundary | 95 % CI         | McNemar p |
|------------------------------|---------:|--------------:|:----------------|----------:|
| mean_all_replicated          |    0.955 |       +0.013  | [+0.002, +0.024]|    3.1e-02|
| mean_loops_23_replicated     |    0.952 |       +0.010  | [+0.001, +0.019]|    6.6e-02|
| mean_loops_234_replicated    |    0.951 |       +0.009  | [+0.001, +0.018]|    6.6e-02|

All three mean configurations beat the natural GRU sequence (boundary) by ~1 pp. The all-loops mean is statistically significant at p < 0.05; the two-loop and three-loop means are right at the edge.

This is direct evidence that the published GRU-over-loops aggregation isn't doing useful work — a parameter-free mean over the same per-loop pooled vectors does at least as well, sometimes better.

The mechanism is consistent with the loop-2-dominance finding: when the GRU temporally processes the 4-loop sequence, loop 2's clean signal is mixed with loop 1's noisier signal and loops 3-4's slight degradation. Mean-pool spreads the contribution equally; GRU compounds across positions and ends up overweighting the final (loop-4) hidden state by virtue of recency. Loop 2's dominance leaks through both, but more cleanly through mean.

**Combined with Finding 1**, the architecturally optimal zero-shot configuration is:

```
score = scorer(layernorm(norm(h_loop_2_c) - norm(h_loop_2_r)) → linear → identity-or-mean)
```

i.e., just the loop-2-boundary state, an extra norm, the diff-norm scaffold, the projection, and the scorer. No GRU. No multi-loop aggregation. Expected accuracy: ≥ 97 %.

A from-scratch retrain at this architecture is **Pass B** below.

---

## Finding 3 — `self.norm` partially closes the mid-layer gap (~5-9 pp)

Each mid-layer probe was scored twice: once on the raw layer output, once on `model.model.norm(output)`. The normed variants are uniformly better, but never reach boundary.

| Layer | Raw accuracy | Normed accuracy | Δ (norm helps) | Boundary − normed |
|------:|-------------:|----------------:|---------------:|------------------:|
|     4 |        0.712 |           0.776 |        +0.064  |          +0.166   |
|    12 |        0.752 |           0.798 |        +0.046  |          +0.144   |
|    24 |        0.763 |           0.815 |        +0.052  |          +0.127   |
|    36 |        0.752 |           0.845 |        +0.093  |          +0.097   |
|    47 |        0.875 |           0.942  |        +0.067  |          +0.000   |

Two observations:

1. **Norm always helps**, by 5-9 pp at every depth. The `OuroRMSNorm` projection is the *single largest single-operation improvement* across the decoder block. This is consistent with a "the evaluator was trained to read the post-norm geometric region" story — applying that same norm to off-distribution mid-layer states partially aligns them.
2. **Norm doesn't close the gap.** Even after norming, layer 24 is 12.7 pp below boundary. The late decoder layers (32 — 47) are still doing semantic refinement that isn't recoverable by post-hoc normalization of an earlier layer's state.

Layer 47 is a sanity check: `layer_47_normed` matches `boundary` exactly (mean = +1.6403, std = 1.0572, identical raw scores) because the boundary capture *is* `self.norm(layer_47_output)` — they're the same tensor modulo dtype, and we've measured both paths producing identical scores. The hook placement is correct.

Layer 36 has the largest norm-induced jump (+9.3 pp from 0.752 → 0.845). This is consistent with the v1 finding that layers 32-36 are in a "representation reorganization" phase whose state is *meaningfully misaligned* with the evaluator's expected geometric region — norm helps a lot there because there's the most misalignment to undo. By layer 47, most of the alignment is already done by the model, so norm only adds 6.7 pp.

---

## Finding 4 — Pair / truncation / start-from configurations: loop 2 dominance, again

Several configurations were designed to test whether *combining* loop 2 with another loop helps the evaluator do better than loop 2 alone:

- `pair_1_2_alt = [h1, h2, h1, h2]`: 0.950 (worse than only_loop_2 = 0.961). Adding loop 1 dilutes.
- `pair_2_3_alt = [h2, h3, h2, h3]`: 0.945 (worse). Adding loop 3 dilutes.
- `pair_2_4_alt = [h2, h4, h2, h4]`: 0.942 (matches boundary). Adding loop 4 dilutes.
- `trunc_to_loop2 = [h1, h2, h2, h2]`: 0.959 (close to only_loop_2). Loop 1 hurts slightly less when it's a minority.
- `start_from_loop2 = [h2, h3, h4, h4]`: 0.941 (matches boundary). Replacing loop 1 with loop 2 is neutral.

Loop 2 is most informative *replicated alone*. Mixing it with other loops in any configuration we tried either neutralizes the gain or partially dilutes it. The information loss isn't catastrophic — the worst-mixed cases just regress to boundary — but the cleanest signal is the un-mixed one.

Combined with Finding 2, this is consistent. The GRU's natural-sequence processing partially overweights loop 4 by recency; mixing loop 2 with later loops via any pattern just gives the GRU more loop-4-like signal to overweight.

---

## Methodological caution — masked-zero variants are numerical degeneracy, not signal

The original GPT-suggested probe `[0, h_loop_2, 0, 0]` (loop 2's state at one position, zeros elsewhere) was intended to test whether the evaluator could still extract a preference signal when most loop positions are masked. We added two variants of the same idea (`[h_loop_2, 0, 0, 0]` and `loop2_then_zeros`, which is the same thing).

All three reported 100.0 % accuracy on n = 1,000. The instinct was to celebrate. The score distributions instead are:

| Config                  | mean   | std   | min    | max    | pos_rate |
|-------------------------|-------:|------:|-------:|-------:|---------:|
| boundary                | +1.640 | 1.057 | −1.490 | +4.391 |    0.942 |
| only_loop_2             | +1.514 | 0.910 | −1.053 | +4.091 |    0.961 |
| **mask_zero_only_loop2**| +1.660 | **0.225** | +1.017 | +2.276 | **1.000** |
| **mask_zero_loop2_front**| +1.741 | **0.220** | +1.142 | +2.367 | **1.000** |
| **loop2_then_zeros**    | +1.741 | **0.220** | +1.142 | +2.367 | **1.000** |

The masked-zero variants have score range [+1.02, +2.37] — never even close to zero, let alone negative. std is ~5× smaller than boundary's. The "100 % accuracy" is because **every score is positive**, and HH-RLHF orders pairs (chosen, rejected) by convention. "Always say chosen wins" hits 100 %.

The mechanism is straightforward, in retrospect: `nn.Linear(... bias=True)` after `nn.LayerNorm(... bias=False)` outputs the projection bias on zero-diff input. The GRU then sees three positions of pure-bias input plus one position with a small variation from the loop-2 signal. The scorer's downstream bias terms push the output reliably positive. The signal contribution from the one informative position is tiny relative to the bias offsets.

**The 100 % is not extracting preference signal from h_loop_2 alone in those configurations.** It's the evaluator's structural bias collapsing the output distribution to positive.

Lesson: **always check the score distribution (mean, std, pos_rate, range), not just the accuracy.** A constant-sign predictor is an accuracy hack, not a finding. We need to verify against the flipped condition too — if the same configuration also returns positive scores on `(rejected, chosen)` input, the architecture is broken. (We didn't flip-test these specific configurations; that would have caught it immediately. Adding to the v4-followup probe wishlist.)

This caution applies to the published architecture too: any time a "zero-shot transfer" hits suspiciously high accuracy, check positivity rate.

---

## Architectural recommendation for Phase 4 (refined again)

Combining v1 and v4 findings, the in-loop integration for the basal-ganglia design should be:

1. **Evaluator fires at end-of-loop-2** (per Ouro's natural recurrence — the first probe pass that puts the preference signal in its sharpest geometric form).
2. **State is `OuroRMSNorm`-applied once more** before being fed to the evaluator. (Finding 1: doublenorm gains +0.7 pp over single-norm boundary loop 2.)
3. **No GRU. Use mean-pool or just-loop-2 architecture.** (Finding 2: mean beats GRU; loop-2-only beats both.)
4. **The "always firing" property remains** — score is computed every forward pass at end-of-loop-2. The "sometimes diverges" property is gated by `|score|` magnitude, same as before.
5. **For inputs that need more compute** (Ouro's exit_pdf low at loop 2), the evaluator can also fire at end-of-loop-3 and end-of-loop-4 — but the loop-2 firing is primary and sufficient for most preference judgments.

This is mechanistically the cleanest formulation. It's also computationally cheap: a loop-2-only evaluator that fires after the second UT iteration means the model can short-circuit loops 3-4 entirely for shallow tasks.

---

## Pass B — Retraining variants from scratch

GPT's items 1, 2, 3 propose retraining evaluator variants to confirm by direct training what we've measured zero-shot:

| Variant            | Architecture                                          | Expected accuracy |
|--------------------|-------------------------------------------------------|------------------:|
| **loop-2-only no-GRU**| Pool loop 2 → diff-norm → Linear → scorer (no GRU) |             ≥ 96.8 % |
| **mean-pool**      | Pool each loop → diff-norm per loop → mean → scorer  |             ≈ 95.5 — 96.5 % |
| **learned softmax**| Pool each loop → diff-norm per loop → softmax(w)·proj → scorer | ≥ 96 % (learned to converge on loop 2) |
| **control: re-trained GRU**| Same as published architecture, retrained on identical data | ≈ 95 % (matches published 95.2 %) |

The hypothesis the zero-shot probe makes: a loop-2-only no-GRU evaluator, trained on HH-RLHF, will hit ≥ 96.8 % — better than the published 95.2 % and architecturally simpler.

To run this fairly, all four variants should share:

- Same AttentionPool initialization (random or copied from `pairwise_epoch2.pt`).
- Same training data (HH-RLHF train split, 50 % swap protocol).
- Same loss (`−logsigmoid(target · score) + 1e-4 · score²`).
- Same epochs, batch size, optimizer, learning rate.
- Evaluation on the same HH-RLHF test split (8,552 pairs).

Cost: ~1 hour per variant on a single GPU (small model, small per-loop pooled-state cache). Total ~4 hours wall-clock plus a few hours of data caching (HH-RLHF train forward through Ouro is the bottleneck).

I'll write the Pass B training pipeline next.

---

## What's still not tested

- **Full 48-layer sweep with norm-applied variants.** We have 5 layers × 2 (raw/normed). A full sweep would resolve the layer-36-normed-jump pattern more cleanly.
- **Math / ARC / code distributions.** The user reported loop-2 dominance across task types; this needs a separate probe with non-HH-RLHF (chosen, rejected) pairs.
- **`loop2_triplenorm`, etc.** Where does the norm-applied accuracy converge? Three norms? Four? If it plateaus at some N, that's evidence the geometry has a stable fixed point.
- **Flip-test on the masked-zero variants** to formally close the loop on the methodological caution (the score distribution already does, but flip would too).
- **AttentionPool variants.** We've left AttentionPool untouched. What if it's also suboptimal?
- **Sensitivity to seq_length.** All probes at max_length=384. Long-context behavior?

---

## Reproducibility

```bash
cd /home/moloch/ouro_project
source venv/bin/activate

# v3 probe (12-layer depth sweep + per-loop ablation + flip):
python utilities/evaluator/probes/probe_evaluator_hypothesis.py --max-examples 1000

# v4 probe (this memo's results — 26-configuration ablation table):
python utilities/evaluator/probes/probe_evaluator_v4_ablations.py --max-examples 1000
```

Raw scores for both runs are saved per-example, so any of the deferred analyses can be done off-line.

---

## One-line summary (after v4-redo, supersedes earlier summaries)

**The pairwise evaluator's preference signal lives in an iteratively-RMSNorm-aligned post-loop hidden state. Loop 2 retains a slight edge but loops 1 and 4 are nearly equal under proper norming. The "loop-2-is-special" v1 claim was largely a norm-alignment artifact. Iterated norm keeps helping monotonically (single → double → triple → quad: 96.1 % → 96.8 % → 97.1 % → 97.9 %), no fixed point at 4 applications. The GRU is mildly counterproductive vs mean-pool. The masked-zero "100 %" finding was a sign-degeneracy artifact (confirmed via flip-test). Phase 4 in-loop architecture should fire at end-of-loop-2, apply 4× OuroRMSNorm to the boundary state, replicate into 4 slots, score.**

---

# v4-redo update (later same day, 2026-05-11)

This section consolidates the third probe pass, which (a) formally verifies the v4 degeneracy claim using flip tests, (b) replaces literal-zero masks with neutral baselines so LayerNorm has non-zero diff input, (c) extends the doublenorm finding by iterating norm 3× and 4×, (d) tests positional bias.

## Finding 6 — Iterated RMSNorm keeps helping: loop2_quadnorm hits 97.9 %

Applying `OuroRMSNorm` to loop 2's boundary state multiple times in succession, then replicating 4× into the evaluator, produces monotonically increasing accuracy:

| Configuration              | Accuracy | std   | sign_flip_rate | antisym r |
|----------------------------|---------:|------:|---------------:|----------:|
| boundary (no extra norm)   |    0.942 | 1.057 |          0.291 |     0.927 |
| only_loop_2 (= 1× extra implicit) |  0.961 | 0.910 |    0.188 |     0.893 |
| only_loop_2_doublenorm     |    0.968 | 0.816 |          0.164 |     0.871 |
| loop2_triplenorm           |    0.971 | 0.762 |          0.128 |     0.851 |
| **loop2_quadnorm**         |**0.979** | 0.712 |          0.091 |     0.827 |

Direct comparison (McNemar pair counts, vs `only_loop_2`):
- `loop2_quadnorm` correct & `only_loop_2` wrong: **21**
- `only_loop_2` correct & `loop2_quadnorm` wrong: **3**

7:1 ratio in favor of quadnorm. Δ = +1.8 pp vs only_loop_2 (95 % CI definitely excludes zero given the McNemar counts).

The std monotonically falls (1.057 → 0.712) as we apply more norms — the score distribution tightens around its mean. Sign-flip-rate on the flip test also falls (0.291 → 0.091) — the architecture becomes more antisymmetric in behavior as we feed more-canonical input.

**No fixed point at 4 applications.** Each additional norm produced ≥ 0.3 pp of accuracy gain. The geometric alignment the evaluator was trained to read appears to be an iterative attractor — repeatedly applying RMSNorm pulls states deeper into it. The natural follow-up: where does this plateau? Could probe pentanorm, hexanorm, etc. until the marginal gain flattens.

This is the **single largest accuracy result** in any zero-shot configuration tested across the three probe passes.

## Finding 7 — Per-loop double-norm: the loop-2-specialness shrinks

Applying double-norm individually to each loop:

| Per-loop variant            | Accuracy | Single-loop baseline | Δ from norm | std   |
|-----------------------------|---------:|---------------------:|------------:|------:|
| only_loop_1_doublenorm      |    0.966 |                0.937 |       +2.9  | 0.775 |
| only_loop_2_doublenorm      |    0.968 |                0.961 |       +0.7  | 0.816 |
| only_loop_3_doublenorm      |    0.948 |                0.948 |       +0.0  | 0.926 |
| only_loop_4_doublenorm      |    0.952 |                0.940 |       +1.2  | 0.956 |

Two observations that **reshape the v1 narrative**:

1. **Norm helps every loop, with the smallest absolute gain for loop 2** — because loop 2's single-norm baseline is already best, there's less room to improve. Loop 1's huge +2.9 pp jump shows that loops 1 and 4 had clean preference geometry that was *masked* by suboptimal RMSNorm alignment.
2. **After proper norming, loops 1, 2, and 4 are all in the 95-97 % band**, with loop 2 keeping only a marginal 0.2 pp lead over loop 1. Loop 3 is the laggard, gaining nothing from extra norm.

The v1 claim — "the preference geometry is uniquely concentrated at loop 2" — has to soften. The corrected reading:

> *The preference geometry is approximately equally accessible at loops 1, 2, and 4 once properly RMSNorm-aligned. Loop 3 is a representational transit point where the geometry is less accessible. Loop 2 has a small genuine advantage at the canonical norm count (the model's `_run_single_ut_loop` applies one norm), making it the natural single-shot endpoint for the published evaluator — but the magnitude of that advantage is much smaller than the un-normed comparison suggested.*

The user's prior cross-task observation ("loop 4 was still the one with most of the work, but loop 2 was often close") is consistent with this: under default norming, loop 4 tends to win on harder tasks (where the model uses all 4 loops), loop 2 wins on easier tasks (where the model converges early); under proper iterated norming, both reveal similar latent quality.

## Finding 8 — Formal verification of the masked-zero degeneracy

The v4 ablation reported 100 % accuracy on three zero-masked variants. v4-redo flip-tests them with the same configurations to verify whether this is signal or constant-sign degeneracy.

Result on `mask_zero_only_loop2` ([0, h2, 0, 0]):
- Normal pos_rate: 1.000 (every score positive)
- **Flipped pos_rate: 1.000** (scoring (rejected, chosen) instead of (chosen, rejected): every score still positive)
- **sign_flip_rate: 0.000** (the sign NEVER flips under input swap)
- std: 0.225 (vs boundary's 1.057 — score distribution is tight around a positive mean)
- antisym r: +0.634 (the underlying antisymmetric component is hiding inside a dominant positive bias offset)

Same pattern for `mask_zero_loop2_front`. **Confirmed: the 100 % is a sign-degeneracy artifact**, not signal extraction.

The mechanism is now understood with a bit more precision than v2 described:

- The architecture is approximately antisymmetric: `corr(s_normal, -s_flipped) = +0.634` even for the degenerate config (would be exactly +1.0 if perfectly antisymmetric, ~0 if uncorrelated).
- But the score = antisymmetric_signal + constant_positive_bias. When 3 of 4 input positions are zero, the antisymmetric signal is small (because the diff-norm gives near-zero diffs at zero positions, and the GRU sees mostly the constant projection bias). The bias offset dominates. Both `(chosen, rejected)` and `(rejected, chosen)` produce scores around +1.7 with small variation.
- HH-RLHF orders pairs (chosen first); "always positive" = 100 % accuracy mechanically. The architecture isn't extracting preference signal — it's outputting a near-constant value that happens to match the dataset's chosen-first convention.

This is the kind of failure mode that escapes accuracy-only evaluation. **In any future probe, report at minimum: accuracy, pos_rate, std, and flip-pos-rate** to catch this class of artifact.

## Finding 9 — Positional bias is essentially nil

We tested whether the evaluator weighs the input positions (0, 1, 2, 3) differently — for instance, whether the GRU's hidden state at the end of the sequence carries more influence than the start. Tested by putting `h_loop_N` at one of the four positions, filling the others with `mean(h_loops_1..4)` as a neutral context.

| Configuration                 | Accuracy |
|-------------------------------|---------:|
| loop_2_at_pos_0_mean_ctx      |    0.955 |
| loop_2_at_pos_2_mean_ctx      |    0.955 |
| loop_2_at_pos_3_mean_ctx      |    0.958 |
| loop_1_at_pos_1_mean_ctx      |    0.956 |
| loop_3_at_pos_1_mean_ctx      |    0.956 |
| loop_4_at_pos_1_mean_ctx      |    0.956 |

All six configurations land in 0.955 – 0.958. Position-of-informative-state doesn't matter. And **which loop is placed at the informative position doesn't matter much either** — loops 1, 2, 3, 4 all hit 0.956 when placed at position 1 with mean context.

This further refines Finding 7: in the "single informative loop + neutral context" setup, even loop 3 (the laggard in pure replication) participates in the same accuracy band as the others. The "loop 3 is worse" finding is specific to the replicated-only configuration, not a general statement about loop 3's geometry.

## Finding 10 — Context choice matters; mean-context is benign, single-loop context hurts

Configurations where loop 2 is "embedded" in different contexts:

| Context for loop 2        | Configuration                        | Accuracy | Δ vs only_loop_2 (0.961) |
|---------------------------|--------------------------------------|---------:|-------------------------:|
| Pure replication          | only_loop_2                          |    0.961 |                    0     |
| Loop 1 surrounding it     | loop2_in_loop1_ctx [h1,h2,h1,h1]     |    0.934 |                   −0.027 |
| Loop 4 surrounding it     | loop2_in_loop4_ctx [h4,h2,h4,h4]     |    0.941 |                   −0.020 |
| Mean-of-loops context     | loop2_in_mean_ctx [μ,h2,μ,μ]         |    0.955 |                   −0.006 |
| All four norm-applied     | loop2_quadnorm                       |    0.979 |                   +0.018 |

When loop 2 is replicated in all 4 slots, accuracy is best (without iterated norming). When it's surrounded by other loops, accuracy *drops*: loop 1 context hurts most (−2.7 pp), loop 4 hurts less (−2.0 pp), mean context hurts least (−0.6 pp). When it's iterated-normed-then-replicated, accuracy *rises* (+1.8 pp).

Interpretation: the GRU's temporal processing isn't neutral — it can actively introduce noise when given heterogeneous-quality loops to aggregate. Pure replication minimizes that interference (the GRU has nothing to "aggregate" because all inputs are equal). Iterated norm tightens the input further, pushing the GRU output to be even more loop-2-flavored.

This confirms (and sharpens) v2's Finding 2: the published GRU-over-loops aggregation is the source of the boundary→only_loop_2 gap. Heterogeneous inputs to the GRU make things worse; homogeneous inputs make things better.

## Final architectural recommendation for Phase 4 (after three probe passes)

Single best zero-shot configuration tested: **`loop2_quadnorm` at 97.9 % on HH-RLHF**.

Architecture:
```python
# Inside the looped transformer's forward pass, after loop 2 completes:
h = h_loop_2_post_norm        # what _run_single_ut_loop returns after its self.norm
h = model.model.norm(h)        # second norm (loop2_doublenorm: 96.8%)
h = model.model.norm(h)        # third  norm (loop2_triplenorm:  97.1%)
h = model.model.norm(h)        # fourth norm (loop2_quadnorm:    97.9%)

# Score (no GRU, no temporal aggregation):
pooled_c = attention_pool(h_chosen, mask_c)
pooled_r = attention_pool(h_rejected, mask_r)
diff = pooled_c - pooled_r
normed = layernorm(diff)
projected = linear(normed)
score = scorer(projected)
```

This formulation:
- Fires the evaluator at end-of-loop-2 (matches Ouro's natural recurrence, supports "exit early for shallow tasks" if combined with the model's existing `exit_pdf`).
- Drops the GRU and per-loop aggregation entirely (eliminates the GRU's mild counterproductivity).
- Iterates RMSNorm 4× to canonicalize the geometric region.
- Cost: 4 extra norm forwards per evaluator call (each is O(seq_len × hidden_dim) — negligible vs the model's own forward).

Expected accuracy on HH-RLHF after retraining on this architecture from scratch: **≥ 97.9 %** (the zero-shot floor; a properly trained version should match or exceed).

The "always firing, sometimes diverges" basal-ganglia property remains: at every forward pass, evaluator fires with this configuration; divergence is gated by `|score|` magnitude. For confident inputs, single-trajectory commit. For ambiguous inputs, branch.

## What v4-redo did *not* test (next probes worth running)

- **pentanorm, hexanorm**: where does the iterated-norm gain plateau? Cheap to add.
- **Iterated norm on other loops**: if `loop1_quadnorm` also hits ≥ 97 %, the "loop 2 specialness" claim disappears entirely. Cheap.
- **GRU ablation retrain**: Pass B, deferred for time. Predicted accuracy: loop2-only-no-GRU retrained ≥ 97.9 %, matching the zero-shot quadnorm result.
- **Math / ARC distributions**: still the strongest external-validity probe. Same machinery, different dataset.

---

# Three-probe roadmap recap

| Pass | What was tested                                            | Headline finding                                              | Best accuracy |
|------|------------------------------------------------------------|---------------------------------------------------------------|--------------:|
| v3   | Depth sweep (12 layers) + per-loop ablation + flip         | Loop 2 alone beats full trajectory                            |        0.961  |
| v4   | 26-config ablation table with norm + pairs + means         | Loop2_doublenorm + mean-pool > GRU + norm helps mid-layer     |        0.968  |
| v4-redo | Degeneracy verification + iterated norm + positional      | Loop2_quadnorm 97.9 %, per-loop doublenorm flattens loop-2 advantage, masked-zero formally degenerate |        0.979  |

Each pass refined or reframed the previous one. The architecture recommendation for Phase 4 has tightened across passes:

- v1 (after v3): "fire at end of loop 2, drop the GRU"
- v2 (after v4): "fire at end of loop 2, apply norm twice, drop the GRU"
- final (after v4-redo): "fire at end of loop 2, apply norm four times, drop the GRU, no per-loop aggregation"

The accuracy floor for the recommended architecture has gone 96.1 % → 96.8 % → 97.9 %.

## What this means for the published paper (arxiv:2604.09870)

The paper's claim was "Relational preference encoding exists in looped-transformer internal states; pairwise evaluator reads it at 95.2 %." That claim survives intact — the relational geometry is real and antisymmetry holds at every depth (Finding 3 from v3, Finding 9 from v4-redo). What we've added:

1. **The signal locus is sharper than the paper documented.** It lives in an iteratively-RMSNorm-aligned subspace of loops 1/2/4's hidden states. Loop 2 has a slight edge under canonical norming, but loops 1 and 4 are not far behind.
2. **The temporal GRU is incidental, not load-bearing.** Mean-pool beats it; replication beats it further; iterated-norm-then-replicate beats it most. The original architecture was carrying unnecessary temporal baggage.
3. **A from-scratch retraining of `loop2_quadnorm + no GRU` should publish at ≥ 97.9 % on HH-RLHF**, improving the paper's headline by ~2.7 pp at substantially reduced architectural complexity.
4. **The methodology lesson** — always pair accuracy with pos_rate / std / flip-test — applies retroactively to any evaluator-based experiment. Constant-sign degeneracy is a publishable footgun.

---

# v5 update (2026-05-11, fourth probe pass)

This pass extends the iterated-norm sweep beyond x4 and applies quadnorm individually to each loop. n = 1,000 HH-RLHF test, same machinery, fully zero-shot.

## Finding 11 — No plateau in iterated norm: x8 hits 99.2 %

Extending the v4-redo norm-iteration sweep from x4 to x8 on loop 2's boundary state:

| Configuration       | Accuracy | std   | sign_flip_rate | antisymmetry r |
|---------------------|---------:|------:|---------------:|---------------:|
| only_loop_2 (= x1)  |    0.961 | 0.910 |          0.188 |          0.893 |
| loop2_norm_x2       |    0.968 | 0.816 |          0.164 |          0.871 |
| loop2_norm_x3       |    0.971 | 0.762 |          0.128 |          0.851 |
| loop2_norm_x4       |    0.979 | 0.712 |          0.091 |          0.827 |
| loop2_norm_x5       |    0.981 | 0.665 |          0.069 |          0.798 |
| loop2_norm_x6       |    0.986 | 0.618 |          0.048 |          0.765 |
| loop2_norm_x7       |    0.989 | 0.575 |          0.037 |          0.730 |
| **loop2_norm_x8**   |**0.992** | 0.535 |          0.029 |          0.694 |

**No plateau reached.** Accuracy climbs monotonically from x1 (96.1 %) to x8 (99.2 %), with each additional norm adding +0.3 to +0.7 pp. The score distribution std falls monotonically (0.910 → 0.535) and the sign-flip-rate falls monotonically (0.188 → 0.029) — the architecture's outputs become more antisymmetric in behavior as more norms are applied.

Total improvement vs boundary (0.942): **+5.0 pp** at x8. Total improvement vs un-normed only_loop_2 (0.961): **+3.1 pp**.

The antisymmetry_r (corr(s_normal, -s_flipped)) does decline (0.893 → 0.694), but this is not a contradiction — it reflects that with iterated norming, the score distribution tightens around a narrower magnitude range, making the swap-induced negation correlate less perfectly even while the *sign-flip behavior* gets cleaner (sign_flip_rate 0.029 = sign flips 97 % of the time on swap, very nearly antisymmetric in sign).

This refutes the v4-redo speculation that x4 was near a fixed point. The geometric attractor that RMSNorm pulls states toward is steeper than was visible at x4.

## Finding 12 — All four loops are equal under quadnorm

Applying 4× extra norm to each loop's boundary state individually, replicated 4× into the evaluator:

| Configuration              | Accuracy | Single-loop baseline (1× norm) | Norm gain |
|----------------------------|---------:|-------------------------------:|----------:|
| only_loop_1_quadnorm       |    0.979 |                          0.937 |   +4.2 pp |
| only_loop_2_quadnorm       |    0.981 |                          0.961 |   +2.0 pp |
| only_loop_3_quadnorm       |    0.979 |                          0.948 |   +3.1 pp |
| only_loop_4_quadnorm       |    0.980 |                          0.940 |   +4.0 pp |

**Spread: 0.979 – 0.981. All four loops are statistically equivalent under quadnorm**, differing by at most 0.2 pp. The "loop 2 is the unique locus of preference geometry" thesis from v1 is now fully refuted:

- The 2.4 pp gap between loop 2 and loops 1/4 in un-normed scoring was almost entirely a norm-alignment artifact.
- Loop 3 — the laggard in un-normed comparison (0.948 in v4-redo vs other loops at 0.937-0.961) — fully catches up under quadnorm.
- The preference geometry exists with equivalent quality across all four UT loop boundary states. The model carries the same preference-relevant subspace through every iteration.

Loop 2's marginal edge (+0.2 pp over loop 1) at quadnorm is within statistical noise. **Under proper iterated norming, any loop's boundary state works equally well as the evaluator's input.**

## Finding 13 — Mean of loops with quadnorm beats individual quadnorm

| Configuration                   | Accuracy |
|---------------------------------|---------:|
| only_loop_2_quadnorm            |    0.981 |
| mean_loops_quadnorm             |    0.984 |
| quadnorm_each_natural_seq (GRU) |    0.979 |

Applying quadnorm to the mean of all 4 loops' boundary states and replicating 4× gives 98.4 % — marginally better than any individual loop's quadnorm (98.0 – 98.1 %). The mean-then-quadnorm aggregation captures slightly more information than any single loop, but the gain is small (+0.3 pp).

Critically: feeding the GRU four quadnormed boundary states in their natural sequence (`quadnorm_each_natural_seq`) gives 0.979 — *worse* than the simple replicated quadnorm (0.981 – 0.984). **Even with cleaned inputs, the GRU's temporal processing remains slightly counterproductive.** This is a stronger confirmation of v4's GRU-counterproductivity finding: the issue isn't that the GRU is given noisy inputs; it's the temporal aggregation operation itself.

## Updated architectural recommendation

The final recommendation, after four probe passes:

```python
# Inside the looped transformer's forward pass, at end of loop 2 (or any loop):
h = h_loop_boundary   # any of h_loop_1, h_loop_2, h_loop_3, h_loop_4 -- they're equivalent under quadnorm

# Iterated RMSNorm canonicalization. Each application adds 0.3-0.7 pp.
# x8 reaches 99.2% on HH-RLHF test (zero-shot). Diminishing returns past x8 not yet measured.
for _ in range(8):
    h = model.model.norm(h)

# Score (no GRU, no temporal aggregation):
pooled_c = attention_pool(h_chosen, mask_c)
pooled_r = attention_pool(h_rejected, mask_r)
diff = pooled_c - pooled_r
normed = layernorm(diff)
projected = linear(normed)
score = scorer(projected)
```

For absolute best zero-shot on HH-RLHF, fire on the mean of loops:

```python
h_mean = mean([h_loop_1, h_loop_2, h_loop_3, h_loop_4])
for _ in range(4):
    h_mean = model.model.norm(h_mean)
# ... score as above
```

But this requires waiting for all 4 loops to complete. The single-loop approach gives 99.2 % zero-shot at x8 with the option to fire after loop 2 (or even loop 1), enabling early-exit inference — which is the basal-ganglia property we wanted from the start.

## Revised one-line summary (after v5, on HH-RLHF only)

**On HH-RLHF chat preference, the pairwise evaluator's preference signal is uniformly accessible across all four UT loop boundary states, but lives in an attractor that RMSNorm iteratively pulls states into. Eight applications of OuroRMSNorm on any loop's boundary state plus replication-into-4-slots gives 99.2 % zero-shot accuracy — within rounding of ceiling, with no plateau yet identified. The temporal GRU is consistently slightly counterproductive. Loop-2-specialness was a 1×-norm artifact; under proper iterated norming, all loops are equivalent.**

---

# Math-distribution probe update (2026-05-11, fifth probe pass)

Tests whether the v3 / v4 / v5 findings generalize beyond HH-RLHF chat. Probe machinery is identical; the dataset is GSM8K (1,000 test examples) with synthetic (chosen, rejected) pairs constructed by replacing the final numerical answer (after "#### N") with a perturbed wrong number. Same length, same vocabulary, same reasoning structure on both sides — only the final answer differs.

**This is a "math correctness" probe**, not a "math preference" probe. It tests whether the evaluator can distinguish correct from incorrect math reasoning.

## Headline: ceiling on math, with three reversals

| Configuration              | Math accuracy | Chat accuracy (v4-redo / v5) | Δ math vs chat |
|----------------------------|--------------:|-----------------------------:|---------------:|
| boundary                   |    **1.000**  |                        0.942 |        +0.058  |
| only_loop_1                |         0.993 |                        0.937 |        +0.056  |
| only_loop_2                |    **1.000**  |                        0.961 |        +0.039  |
| only_loop_3                |    **1.000**  |                        0.948 |        +0.052  |
| only_loop_4                |    **1.000**  |                        0.940 |        +0.060  |
| loop2_norm_x2 (doublenorm) |         0.995 |                        0.968 |        +0.027  |
| loop2_norm_x4 (quadnorm)   |         0.989 |                        0.979 |        +0.010  |
| loop2_norm_x5              |         0.989 |                        0.981 |        +0.008  |
| only_loop_1_quadnorm       |    **1.000**  |                        0.979 |        +0.021  |
| only_loop_2_quadnorm       |         0.989 |                        0.981 |        +0.008  |
| only_loop_3_quadnorm       |         0.999 |                        0.979 |        +0.020  |
| only_loop_4_quadnorm       |         0.999 |                        0.980 |        +0.019  |
| mean_replicated            |    **1.000**  |                        0.955 |        +0.045  |

Three reversals vs the chat picture:

1. **The boundary architecture hits ceiling on math** (100.0 % accuracy). The "GRU-aggregated four-loop sequence" is fine here. Chat headroom (boundary 94.2 → quadnorm 97.9) doesn't exist on math because there's no headroom left at the boundary.
2. **All four single-loop ablations effectively tie at boundary** (99.3 – 100 %). On chat at 1× norm, loop 2 led by 2.4 pp. On math at 1× norm, the spread is 0.7 pp. The "loop 2 wins" effect is largely a chat-specific artifact.
3. **Iterated norm slightly HURTS on math.** loop2_norm_x4 = 0.989 (down 1.1 pp from boundary's 1.000). loop2_norm_x5 = 0.989. The chat-favorable norm iteration is mildly counterproductive on math discrimination.

## Finding 14 — The math distribution generalizes the evaluator, but in a different regime

The evaluator does what the user said it does informally: it discriminates math correctness without retraining. Zero-shot, the boundary head correctly picks the correct-final-answer text 100 % of the time on GSM8K with synthetic answer-perturbation pairs.

But the *mechanism* differs from chat. On chat, the relational signal is moderate-strength relative to the architecture's positive bias offset (sign_flip_rate ≈ 0.29 at boundary — only 29 % of examples have the relational signal magnitude exceeding the bias). On math, the relational signal is stronger (sign_flip_rate ≈ 0.47 at boundary — 47 % exceed the bias). But the architectural bias offset *also* contributes substantially to the math accuracy: pos_rate is 1.000 on math but flipped_pos_rate is approximately 0.53, meaning when you swap inputs, the score stays positive in 53 % of examples.

What this means concretely:

- **Math is two effects stacked.** A real relational signal (the evaluator does notice the correct number is correct), plus a positive bias offset that pushes ambiguous cases to positive. Both align with HH-RLHF's chosen-first convention applied to GSM8K's standardized ordering, giving the 100 % headline.
- **A truly hard math discrimination task** — pairs where the wrong reasoning is qualitatively similar to the right reasoning — would likely show less ceiling behavior. The current "final-answer-swap" perturbation is too easy.
- **The reading on chat from earlier passes still stands.** Chat is the harder task; iterated norm produces real gains there.

## Finding 15 — Iterated norm trades bias offset for cleaner relational signal

Combining the chat and math results suggests a unified mechanism. Iterated `OuroRMSNorm` doesn't make the relational signal *better*; it makes the architecture's output *less dependent on the bias offset*. On math:

| Configuration       | sign_flip_rate | std   |
|---------------------|---------------:|------:|
| boundary            |          0.473 | 0.546 |
| loop2_norm_x2       |          0.248 | 0.569 |
| loop2_norm_x3       |          0.166 | 0.554 |
| loop2_norm_x4       |          0.111 | 0.530 |
| loop2_norm_x5       |          0.066 | 0.501 |
| only_loop_4_quadnorm|          0.030 | 0.477 |

sign_flip_rate falls monotonically with iterated norm (0.473 → 0.030) — the architecture becomes nearly perfectly antisymmetric. Meanwhile std also falls (0.546 → 0.477) — the score distribution tightens around the true signal.

But for *accuracy*, this is a wash on math because the bias-offset was already getting math right with high probability (every score positive ≥ chosen-first answer). Removing the bias removes a useful crutch. The accuracy stays at ceiling (well, 98.9 % – 100 %) but doesn't improve.

**The unified statement, across chat and math:**

> Iterated RMSNorm produces a cleaner antisymmetric architecture by canonicalizing the input geometry. On distributions where the relational signal is moderate and the bias offset is doing meaningful "tiebreaker" work (chat), this is a clear win — sharper signal, +5 pp accuracy. On distributions where signal is strong and bias is happily aligned with the dataset ordering convention (math with easy perturbations), it's a small wash — the bias offset was carrying the easy cases at no cost, and removing it doesn't add headroom because there's no headroom to add.

## Finding 16 — Loop 1 is consistently slightly worse, on both distributions

Loop 1's per-loop accuracy at 1× norm:
- Chat: 0.937 (lowest of the four)
- Math: 0.993 (lowest of the four, despite ceiling on others)

But under quadnorm:
- Chat: 0.979 (tied with loops 2-4)
- Math: 1.000 (tied with loops 3, 4 at 0.999-1.000)

Loop 1's marginal disadvantage at 1× norm is consistent across both tasks. Under proper iterated norming, loop 1 catches up fully. This is consistent with the v5 framing: the difference between loops at 1× norm reflects how much canonicalization each loop's state has already had via the model's natural `self.norm` application inside `_run_single_ut_loop` — loop 1's state may be slightly farther from the attractor's center.

## Implications for Phase 4 architecture

Math results add a constraint to the architectural recommendation:

- The chat-derived "iterate `OuroRMSNorm` 8 times" prescription is **chat-specific**. On math discrimination, it slightly hurts.
- The robust recommendation is therefore: **fire the evaluator at a UT loop boundary; apply iterated norm adaptively based on task difficulty**.

In the basal-ganglia framing, this maps onto: easier inputs need less canonicalization (the model's recurrent state already has sufficient alignment); harder inputs benefit from more iterated norming (we pull the geometry deeper into the attractor to resolve ambiguity).

Phase 4 should probably parameterize the norm count as a task-difficulty-dependent hyperparameter, possibly tied to the model's own `exit_pdf` confidence:

```python
# Pseudocode for adaptive iterated norm
h = h_loop_boundary
n_norm_apps = adaptive_count(exit_pdf, task_difficulty)  # e.g., 0 for confident easy, 8 for unconfident hard
for _ in range(n_norm_apps):
    h = model.model.norm(h)
score = evaluator(h × 4, ...)
```

The "always firing, sometimes diverges" property is preserved. The new wrinkle: also "always firing, sometimes canonicalizes more deeply" — easy cases get a fast, lightly-normed read; hard cases get a slow, deeply-normed read.

## Caveats on the math probe

- **Synthetic perturbation is too easy.** Final-answer-swap leaves everything else intact; the diff signal is concentrated in one or two tokens. A harder math probe would use (correct reasoning, wrong reasoning) pairs where the *steps* differ, not just the final number. That would likely push boundary accuracy below ceiling and re-create the headroom we have on chat.
- **GSM8K's ordering convention matches HH-RLHF's.** We always put correct (chosen) first. A randomized-order math probe would tease apart "is the architecture picking signal" from "does the bias offset agree with the ordering."
- **n = 1000 is small for ceiling-level effects.** At 1.000 vs 0.999, we can't statistically distinguish single-pp differences. Larger n would clarify which configurations are genuinely at ceiling vs nearly so.

## Updated one-line summary (after math probe, final)

**The pairwise evaluator generalizes from HH-RLHF chat to GSM8K math correctness at zero-shot ceiling (100 % at boundary). The "loop 2 specialness" found on chat largely disappears on math even at 1× norm — all loops carry equivalent correctness-discrimination signal. Iterated RMSNorm gives large gains on hard discrimination tasks (chat: +5 pp from x1 to x8) but is a small wash on easy ones (math: −1 pp from boundary to x4-x5). The mechanism appears to be that iterated norm trades a useful positive-bias-offset for a cleaner antisymmetric output; this is a net win when the relational signal needs disambiguation and a slight loss when the bias was already correctly aligned.**

---

# Five-probe roadmap recap (final)

| Pass | Dataset | What was tested | Headline finding | Best accuracy |
|------|---------|------------------|-------------------|--------------:|
| v3 | HH-RLHF | Depth sweep (12 layers) + per-loop ablation + flip | Loop 2 alone beats full trajectory | 0.961 |
| v4 | HH-RLHF | 26-config ablation table with norm + pairs + means | Loop2_doublenorm + mean-pool > GRU + norm helps mid-layer | 0.968 |
| v4-redo | HH-RLHF | Degeneracy verification + iterated norm + positional | Loop2_quadnorm 97.9%, masked-zero degenerate confirmed | 0.979 |
| v5 | HH-RLHF | Iterated norm sweep to x8 + per-loop quadnorm | Loop2_norm_x8 = 99.2%, ALL loops equal under quadnorm | 0.992 |
| math | GSM8K (synthetic pairs) | Same configurations on math correctness | Ceiling on math, norm slightly hurts, loop 2 specialness was chat-specific | 1.000 |

Each pass refined the previous one. The story shifted four times:

- v3: "loop 2 is the locus" → 
- v4: "loop 2 + double-norm is the locus" → 
- v4-redo + v5: "any loop + iterated norm is the locus, GRU is incidental" → 
- math: "any loop is fine on easy tasks; iterated norm only helps on hard discrimination."

The methodology lesson (always check pos_rate / std / flip-test) was confirmed empirically twice (v4 masked-zero degeneracy on chat; math probe showing bias offset doing real work on easy discrimination).

---

# v6 correction and control-readiness update (2026-05-12)

This section supersedes the v5 and GSM8K interpretations above while preserving them as part of the research record.

The new review point was correct: the memo interpreted `sign_flip_rate` backwards. The code defines it as:

```python
strict_sign_reversal_rate = mean(sign(score(chosen, rejected)) != sign(score(rejected, chosen)))
```

So a lower value means **less** strict sign reversal, not more. In the masked-zero degeneracy section, `sign_flip_rate = 0.000` was already correctly understood as "the sign never flips." The v5 claim that `loop2_norm_x8` with `sign_flip_rate = 0.029` "flips 97% of the time" was wrong. It flips only 2.9% of the time.

The corrected high-level conclusion:

> Iterated `OuroRMSNorm` makes canonical chosen-first HH-RLHF accuracy climb dramatically, but it does not make the evaluator more antisymmetric. At high norm counts it appears to amplify a symmetric positive offset relative to the antisymmetric relational signal. `loop2_norm_x8 = 99.2%` is real as a canonical chosen-first classifier result, but not as a robust pairwise comparator result.

## Test 1 — Offline bias decomposition on existing v5 HH scores

Raw v5 scores were re-analyzed offline using:

```bash
venv/bin/python utilities/evaluator/probes/analyze_pairwise_bias_decomposition.py \
  artifacts/reports/evaluator/probe_v5_extended.json \
  artifacts/reports/evaluator/probe_evaluator_math.json \
  --output-json artifacts/reports/evaluator/bias_decomposition_all.json \
  --output-md artifacts/reports/evaluator/bias_decomposition_all.md
```

Key HH rows:

| config | canonical_acc | centered_acc | flipped_acc | strict_sign_reversal | antisym_corr | bias_to_signal |
|---|---:|---:|---:|---:|---:|---:|
| boundary | 0.942 | 0.604 | 0.233 | 0.291 | 0.927 | 3.42 |
| only_loop_2 | 0.961 | 0.588 | 0.149 | 0.188 | 0.893 | 5.09 |
| loop2_norm_x4 | 0.979 | 0.585 | 0.070 | 0.091 | 0.827 | 6.77 |
| loop2_norm_x8 | 0.992 | 0.547 | 0.021 | 0.029 | 0.694 | 14.39 |
| mean_loops_quadnorm | 0.984 | 0.579 | 0.054 | 0.070 | 0.814 | 7.94 |
| quadnorm_each_natural_seq | 0.979 | 0.591 | 0.089 | 0.110 | 0.841 | 6.04 |

The decisive number is `centered_acc = mean(score(chosen,rejected) - score(rejected,chosen) > 0)`.

`loop2_norm_x8` is worse than boundary on centered accuracy: 0.547 vs 0.604. It is better only on canonical chosen-first accuracy. Bias-to-signal rises from 3.42 at boundary to 14.39 at x8.

The right decomposition is:

```python
s_cr = score(chosen, rejected)
s_rc = score(rejected, chosen)

bias = 0.5 * (s_cr + s_rc)
antisym = 0.5 * (s_cr - s_rc)
centered_score = s_cr - s_rc
```

A true pairwise comparator should mostly preserve the sign of `centered_score`. A dataset-order classifier can succeed with positive `bias` even when `antisym` is weak.

## Test 2 — x8 across all loops on HH-RLHF

New script:

```bash
venv/bin/python utilities/evaluator/probes/probe_hh_norm_mechanism_centered.py \
  --max-examples 1000 \
  --output-json artifacts/reports/evaluator/probe_hh_norm_mechanism_centered.json
```

All four loops are still equivalent under total x8 for canonical accuracy, but all are poor centered comparators:

| config | canonical_acc | centered_acc | flipped_acc | strict_sign_reversal | antisym_corr | bias_to_signal |
|---|---:|---:|---:|---:|---:|---:|
| loop1 total_x8 | 0.988 | 0.530 | 0.023 | 0.035 | 0.698 | 19.64 |
| loop2 total_x8 | 0.992 | 0.547 | 0.021 | 0.029 | 0.694 | 14.39 |
| loop3 total_x8 | 0.993 | 0.547 | 0.024 | 0.031 | 0.694 | 12.49 |
| loop4 total_x8 | 0.991 | 0.551 | 0.022 | 0.031 | 0.697 | 12.21 |
| mean-then total_x8 | 0.992 | 0.543 | 0.022 | 0.030 | 0.697 | 14.09 |
| natural-seq total_x8 | 0.991 | 0.552 | 0.024 | 0.033 | 0.698 | 11.77 |

This finishes off the "loop 2 special" thesis. Loop 2 is not special under the relevant normalization/readout tests. It remains an engineering convenience only: a cheap early boundary where the signal is available.

## Test 3 — Norm mechanism ablation

The v5 memo described iterated norm as an attractor. That metaphor is too vague. Mechanically, repeated `OuroRMSNorm` mostly behaves like repeated application of the learned final-norm weights `gamma`, with RMS rescaling:

```python
h = gamma * h / rms(h)
```

Ablation on loop 2:

| config | canonical_acc | centered_acc | flipped_acc | strict_sign_reversal | antisym_corr | bias_to_signal | score_std |
|---|---:|---:|---:|---:|---:|---:|---:|
| learned k0 | 0.961 | 0.588 | 0.149 | 0.188 | 0.893 | 5.09 | 0.910 |
| learned k3 | 0.979 | 0.585 | 0.070 | 0.091 | 0.827 | 6.77 | 0.712 |
| learned k7 / total_x8 | 0.992 | 0.547 | 0.021 | 0.029 | 0.694 | 14.39 | 0.535 |
| pure RMS k7 | 0.957 | 0.558 | 0.124 | 0.167 | 0.886 | 6.38 | 0.894 |
| gamma-only k7 | 0.990 | 0.547 | 0.026 | 0.036 | 0.714 | 14.48 | 0.537 |
| learned k8 | 0.993 | 0.550 | 0.017 | 0.024 | 0.657 | 16.09 | 0.501 |
| gamma-only k8 | 0.991 | 0.546 | 0.019 | 0.028 | 0.677 | 15.91 | 0.504 |

Pure RMS does not reproduce the effect. Gamma-only almost exactly reproduces the high-canonical / low-centered x8 behavior. The result is therefore not "normalization finds a preference fixed point." It is:

> Ouro's learned final norm weights act as a readout lens / diagonal power filter. Repeated application sharpens the canonical chosen-first direction, but also increases symmetric positive bias dominance.

## Test 4 — HH K sweep ranked by centered accuracy

When ranked by centered/debiased behavior, high norm counts are not best. The top learned HH configurations were:

| config | canonical_acc | centered_acc | flipped_acc | strict_sign_reversal | bias_to_signal |
|---|---:|---:|---:|---:|---:|
| loop4 k0 | 0.940 | 0.606 | 0.225 | 0.285 | 3.57 |
| loop4 k2 | 0.961 | 0.605 | 0.165 | 0.204 | 4.36 |
| natural seq k0 | 0.942 | 0.604 | 0.233 | 0.291 | 3.42 |
| natural seq k2 | 0.958 | 0.603 | 0.175 | 0.217 | 4.20 |
| loop3 k3 | 0.971 | 0.603 | 0.110 | 0.139 | 5.46 |

For HH branch control, the measured centered optimum is low/moderate K, not x8. x8 remains valuable as a diagnostic of the canonical chosen-first classifier direction, but it should not be used raw for branch selection.

## Test 5 — Harder Hendrycks MATH middle-step perturbation

The existing harder math probe was used, with its loader patched to fall back to `EleutherAI/hendrycks_math` subject configs when `lighteval/MATH` and `hendrycks/competition_math` are unavailable:

```bash
venv/bin/python utilities/evaluator/probes/probe_evaluator_hendrycks_math.py \
  --max-examples 1000 \
  --output-json artifacts/reports/evaluator/probe_evaluator_hendrycks_math.json
```

Dataset: 1,000 examples sampled from Hendrycks MATH level >= 4, rejected branch built by perturbing a middle-step number rather than only the final answer.

| config | canonical_acc | centered_acc | flipped_acc | strict_sign_reversal | antisym_corr | bias_to_signal |
|---|---:|---:|---:|---:|---:|---:|
| boundary | 0.999 | 0.882 | 0.751 | 0.752 | 0.799 | 0.84 |
| only_loop_2 | 0.999 | 0.868 | 0.647 | 0.648 | 0.781 | 0.97 |
| loop2_norm_x4 | 0.999 | 0.809 | 0.279 | 0.280 | 0.614 | 1.67 |
| loop2_norm_x8 | 0.998 | 0.850 | 0.163 | 0.165 | 0.396 | 1.78 |
| mean_replicated | 1.000 | 0.880 | 0.736 | 0.736 | 0.816 | 0.87 |
| mean_then_norm_x8 | 0.999 | 0.873 | 0.249 | 0.250 | 0.438 | 1.51 |
| norm_x8_each_natural_seq_GRU | 0.998 | 0.882 | 0.422 | 0.424 | 0.482 | 1.32 |

The stronger math probe changes the earlier GSM8K interpretation:

- Canonical math accuracy remains near ceiling.
- Centered accuracy is much higher than HH, so the evaluator really does read correctness-like relational signal on math.
- Iterated norm still reduces strict sign reversal and antisymmetry correlation.
- Boundary / mean / natural sequence are already strong; x8 is not required.

The earlier GSM8K final-answer-swap probe was too easy. Hendrycks middle-step perturbation is still synthetic, but it is a better probe of correctness geometry.

## Test 6 — Branch-selection simulation

A direct Phase-4-style tournament was run:

```bash
venv/bin/python utilities/evaluator/probes/probe_branch_selection_sim.py \
  --max-examples 250 \
  --output-json artifacts/reports/evaluator/probe_branch_selection_sim.json
```

Each example had four unordered candidates: one correct solution plus three middle-step-perturbed wrong branches. Correct position was randomized: `[62, 59, 68, 61]`.

Two selection rules:

```python
raw_winner = argmax_i sum_j score(branch_i, branch_j)
debiased_winner = argmax_i sum_j 0.5 * (score(branch_i, branch_j) - score(branch_j, branch_i))
```

Results:

| config | raw_select_acc | debiased_select_acc | raw/debiased agreement | raw_margin | debiased_margin |
|---|---:|---:|---:|---:|---:|
| boundary_natural | 0.896 | 0.888 | 0.992 | 4.304 | 3.810 |
| only_loop_2 | 0.848 | 0.832 | 0.976 | 3.389 | 3.014 |
| loop2_total_x8 | 0.796 | 0.788 | 0.960 | 1.474 | 1.392 |
| loop3_extra_k3 | 0.828 | 0.816 | 0.984 | 2.254 | 2.000 |
| loop4_extra_k0 | 0.896 | 0.892 | 0.996 | 4.273 | 3.790 |
| loop4_extra_k2 | 0.848 | 0.840 | 0.980 | 2.516 | 2.208 |
| mean_replicated | 0.884 | 0.884 | 0.996 | 4.253 | 3.769 |
| mean_total_x8 | 0.864 | 0.856 | 0.988 | 1.861 | 1.753 |
| natural_seq_total_x8 | 0.864 | 0.864 | 0.992 | 2.144 | 2.011 |

This is the practical control result:

- x8 is bad for branch selection despite its near-ceiling canonical HH score.
- Boundary natural, loop4 k0, and mean replicated are the best tested branch selectors.
- Raw and debiased winners agree at high rates in this math simulation because the correct branch is usually strongly separable, but the debiased rule remains the safer control primitive for arbitrary branch comparisons.

## Updated interpretation across the three papers

### Scaling Latent Reasoning via Looped Language Models

The Ouro paper frames loop count as latent compute depth. These probes support that loop boundary states are meaningful readout points, but the preference/correctness signal is not localized to a single loop. It is carried through the recurrent trajectory. The apparent loop-2 privilege was mostly the evaluator/readout/norm condition, not a unique mechanistic phase transition.

### RLTT

RLTT's "reward the latent thought trajectory" framing fits better than the older loop-2-locus story. The signal is trajectory-distributed: all UT loop boundaries can expose quality/correctness information when read through an appropriate lens. That is consistent with process-level supervision shaping the latent path, not merely the final answer.

### Relational Preference Encoding

The relational preference paper survives the strongest. The evaluator is reading a real pairwise relational geometry in hidden states. But the new correction matters: raw score is not identical to relational preference. Raw score decomposes into antisymmetric relational signal plus symmetric bias. Accuracy-only evaluation conflates the two whenever the dataset order is canonical.

The robust object is:

```python
preference(a, b) = 0.5 * (score(a, b) - score(b, a))
```

not:

```python
score(a, b)
```

## Proto-introspection interpretation

The evidence supports a limited, technical form of proto-introspection:

> Ouro's latent states contain internally readable evidence about answer quality / preference / correctness before final decoding, and a lightweight evaluator can extract that evidence relationally.

But "set in stone" is too strong. What is now set:

- The evaluator is not reading loop-2-specialness.
- The GRU is not load-bearing.
- x8 raw score is not a robust preference comparator.
- The learned final-norm gamma acts as a strong readout lens.
- The useful signal is distributed across loop boundary states.
- Branch control needs centered/debiased pairwise scores.

What is not set:

- Whether the signal should be called "human preference", "correctness", "internal value", or "answer-quality geometry" in every domain.
- Which K is best outside the tested distributions.
- Whether a retrained no-bias/debiased evaluator would preserve x8's canonical gain without the symmetric offset.

The cautious phrase is:

> latent relational self-evaluation / proto-introspection signal

not:

> solved introspection or fixed internal preference scalar.

## Phase 4 recommendation after v6

Do not use raw `score(a,b)` for branch selection. Use the antisymmetric component:

```python
def score_pair_debiased(h_a, mask_a, h_b, mask_b, k_norm):
    h_a_k = h_a
    h_b_k = h_b
    for _ in range(k_norm):
        h_a_k = model.model.norm(h_a_k)
        h_b_k = model.model.norm(h_b_k)

    s_ab = evaluator(h_a_k, mask_a, h_b_k, mask_b)
    s_ba = evaluator(h_b_k, mask_b, h_a_k, mask_a)
    return 0.5 * (s_ab - s_ba)
```

For tournament branch selection:

```python
winner = argmax_i sum_j score_pair_debiased(branch_i, branch_j, k_norm)
```

Use low/moderate K first:

- HH centered best tested: loop4 k0/k2, natural seq k0/k2, loop3 k3.
- Math branch selection best tested: boundary natural, loop4 k0, mean replicated.
- x8 is diagnostic, not production control, unless retraining or centering fixes the bias problem.

The next most valuable test is still retraining, but the target has changed:

1. Train a centered/debiased evaluator objective or explicitly swap-balanced objective.
2. Penalize symmetric offset: encourage `score(a,b) + score(b,a) ≈ 0`.
3. Evaluate by centered accuracy and branch-selection tournaments, not canonical chosen-first accuracy.

## Corrected one-line summary after v6

The evaluator's canonical chosen-first accuracy improves dramatically when Ouro loop boundary states are repeatedly passed through learned `OuroRMSNorm`, reaching 99.2% on a 1,000-pair HH-RLHF probe at `loop2_norm_x8`. Quadnorm/x8 tests remove loop-2 specialness: all loop boundary states expose comparable signal under the same readout. However, falling strict sign-reversal rates, falling antisymmetry correlations, and centered-accuracy collapse show that repeated learned norm mostly sharpens a canonical chosen-first direction while amplifying symmetric positive bias. For Phase 4 control, use the antisymmetric centered score `score(a,b) - score(b,a)`, and choose norm depth by centered/branch-selection validation rather than canonical HH accuracy.

# v7 addendum - all-layer taps and cached coding/reasoning/logic probes

Date: 2026-05-12

This addendum preserves the v6 correction and adds two new blocks of evidence:

1. A full HH all-layer centered sweep across transformer layers.
2. Actual selected-tap branch-selection runs on hard math plus cached coding, reasoning, logic, and thinking benchmarks.

## Dataset download status

The compact evaluation dataset downloader cached:

| tag | domain | rows | status |
|---|---|---:|---|
| humaneval | coding | 164 | OK |
| mbpp_sanitized | coding | 257 | OK |
| mbpp_plain | coding | 257 | OK fallback alias |
| arc_challenge | reasoning | 299 | OK |
| arc_easy | reasoning | 570 | OK control |
| bbh_boolean | reasoning | 250 | OK |
| bbh_multistep | reasoning | 250 | OK |
| bbh_logical_five | logic | 250 | OK |
| bbh_tracking | logic | 250 | OK |
| strategyqa | thinking | 687 | OK |
| logiqa_eleuther / logiqa_alt | logic | - | failed: HF dataset scripts unsupported by installed datasets version |
| gpqa_diamond | thinking | - | failed: gated dataset, auth required |

This means the non-math run is not "just math." It includes coding, ARC science reasoning, BBH symbolic/logic tasks, and StrategyQA.

## Test 7 - HH all-layer sweep

Command:

```bash
venv/bin/python utilities/evaluator/probes/probe_all_layers_centered.py \
  --max-examples 1000 \
  --score-chunk-size 32 \
  --output-json artifacts/reports/evaluator/probe_all_layers_centered.json
```

Analysis:

```bash
venv/bin/python utilities/evaluator/probes/analyze_all_layer_taps.py \
  --input-json artifacts/reports/evaluator/probe_all_layers_centered.json \
  --output-json artifacts/reports/evaluator/probe_all_layers_analysis.json \
  --output-md artifacts/reports/evaluator/probe_all_layers_analysis.md
```

Key HH result:

| config | canonical_acc | centered_acc |
|---|---:|---:|
| layer_45_natural_normed | 0.893 | 0.609 |
| layer_47_natural_normed / boundary | 0.942 | 0.604 |
| layer_42_natural_normed | 0.887 | 0.603 |
| layer_41_natural_normed | 0.903 | 0.603 |
| layer_46_natural_normed | 0.908 | 0.602 |
| layer_40_natural_normed | 0.900 | 0.601 |
| layer_36_natural_normed | 0.845 | 0.576 |
| layer_24_natural_normed | 0.815 | 0.563 |

The first natural normed layer above 0.600 centered accuracy is layer 40. Layer 24 is early and nontrivial, but on HH it is not the strongest control/readout layer. The all-layer sweep argues for a multi-tap controller:

- layer 20/24: early scout/intervention taps
- layer 36: mid-late transition tap
- layers 40-47: strongest readout/final gate taps

Layer 24 is promising because it is early, not because it is globally best.

## Test 8 - actual selected layer taps on hard MATH

Command:

```bash
venv/bin/python utilities/evaluator/probes/probe_layer_tap_math_actual.py \
  --pair-examples 1000 \
  --branch-examples 250 \
  --output-json artifacts/reports/evaluator/probe_layer_tap_math_actual.json
```

Selected hard-MATH results:

| config | canonical_acc | centered_acc | raw_branch_acc | debiased_branch_acc |
|---|---:|---:|---:|---:|
| boundary_natural | 0.997 | 0.890 | 0.904 | 0.900 |
| layer_20_natural_raw | 0.994 | 0.879 | 0.860 | 0.860 |
| layer_24_natural_raw | 0.998 | 0.890 | 0.864 | 0.864 |
| layer_36_natural_raw | 0.995 | 0.890 | 0.900 | 0.900 |
| layer_40_natural_raw | 0.995 | 0.892 | 0.908 | 0.908 |
| layer_45_natural_raw | 0.998 | 0.895 | 0.912 | 0.912 |
| layer_46_natural_raw | 1.000 | 0.897 | 0.912 | 0.912 |
| layer_47_natural_raw | 0.998 | 0.900 | 0.908 | 0.908 |
| layer_24_natural_normed | 0.992 | 0.852 | 0.820 | 0.812 |
| layer_36_natural_normed | 0.994 | 0.886 | 0.884 | 0.880 |
| layer_47_natural_normed | 0.997 | 0.890 | 0.904 | 0.900 |

Hard math strengthens the "late raw readout" result. Layers 40-47 raw beat or match boundary in branch tournaments. Layer 36 raw is also strong. Layer 24 raw is excellent pairwise but weaker as a four-way selector, so it should be treated as an early intervention/signal tap rather than the final selector.

## Test 9 - cached coding, reasoning, logic, and thinking run

Command:

```bash
venv/bin/python utilities/evaluator/probes/probe_layer_tap_cached_domains.py \
  --max-examples-per-dataset 120 \
  --output-json artifacts/reports/evaluator/probe_layer_tap_cached_domains.json
```

Each item was converted into one correct candidate and one or more wrong candidates:

- HumanEval/MBPP: reference solution vs mutated code.
- ARC: correct answer choice vs wrong choices.
- BBH: target answer vs wrong target/perturbed answers.
- StrategyQA: yes/no answer vs opposite answer.

Aggregate over 1,080 examples:

| config | canonical_acc | centered_acc | raw_branch_acc | debiased_branch_acc |
|---|---:|---:|---:|---:|
| boundary_natural | 0.971 | 0.757 | 0.658 | 0.656 |
| boundary_loop_4_replicated | 0.976 | 0.754 | 0.662 | 0.655 |
| layer_20_natural_raw | 0.891 | 0.786 | 0.701 | 0.705 |
| layer_24_natural_raw | 0.893 | 0.788 | 0.704 | 0.706 |
| layer_36_natural_raw | 0.875 | 0.774 | 0.696 | 0.697 |
| layer_40_natural_raw | 0.884 | 0.776 | 0.696 | 0.698 |
| layer_45_natural_raw | 0.901 | 0.776 | 0.696 | 0.700 |
| layer_47_natural_raw | 0.927 | 0.777 | 0.696 | 0.695 |
| layer_36_natural_normed | 0.894 | 0.771 | 0.690 | 0.692 |
| layer_47_natural_normed | 0.971 | 0.757 | 0.658 | 0.656 |

Top aggregate branch selectors:

| config | raw_branch_acc | debiased_branch_acc |
|---|---:|---:|
| layer_24_natural_raw | 0.704 | 0.706 |
| layer_20_natural_raw | 0.701 | 0.705 |
| layer_45_natural_raw | 0.696 | 0.700 |
| layer_40_natural_raw | 0.696 | 0.698 |
| layer_36_natural_raw | 0.696 | 0.697 |
| layer_47_natural_raw | 0.696 | 0.695 |
| layer_36_natural_normed | 0.690 | 0.692 |
| boundary_loop_4_replicated | 0.662 | 0.655 |
| boundary_natural | 0.658 | 0.656 |

Important per-dataset branch-selection results:

| dataset | best tested config | raw_branch_acc | debiased_branch_acc | note |
|---|---|---:|---:|---|
| HumanEval | layer_45_natural_raw | 0.917 | 0.925 | raw intra-layer taps dominate boundary |
| MBPP | layer_36_natural_raw / layer_47_natural_raw | 0.967 | 0.967 | coding signal is very strong |
| ARC-Challenge | layer_36_natural_normed | 0.875 | 0.875 | normed mid-late tap wins |
| ARC-Easy | layer_40_natural_normed | 1.000 | 1.000 | easy control near ceiling |
| BBH Boolean | layer_36_natural_raw | 0.767 | 0.767 | moderate symbolic signal |
| BBH Multistep | layer_24_natural_raw | 0.450 | 0.442 | weak, below useful control threshold |
| BBH Logical Five | layer_24_natural_normed | 0.750 | 0.750 | early/mid taps beat boundary |
| BBH Tracking | layer_45_natural_normed | 0.283 | 0.275 | very weak, likely candidate construction/task difficulty issue |
| StrategyQA | layer_40_natural_raw | 0.633 | 0.633 | weak-to-moderate yes/no relational signal |

This cross-domain run changes the engineering recommendation:

- Boundary has high canonical accuracy, but it is not the best branch selector outside the original HH-style setup.
- Raw intra-layer taps often beat normalized boundary states for branch selection.
- Layer 24 is genuinely useful as an early raw tap across coding/reasoning aggregate, but it is not sufficient alone.
- Layer 36 is the best mid-late intervention candidate.
- Layers 40-47 are the best final readout selector region, especially for hard math and coding.
- The BBH tracking and multistep failures are important: the evaluator is not a general reasoning oracle. It reads some relational correctness geometry well and other reasoning geometries poorly.

## Revised Phase 4 layer placement recommendation

The controller should not be inserted at only one layer.

Use a multi-tap basal-ganglia-style policy:

1. Early scout taps: layers 20 and 24 raw states.
2. Mid controller tap: layer 36 raw/normed, task-selected.
3. Final selector taps: layers 40, 45/46, and 47 raw states.
4. Boundary state as a baseline/fallback, not the default control surface.

A practical first policy:

```python
tap_set = {
    "early": [20, 24],
    "mid": [36],
    "late": [40, 45, 47],
    "boundary": ["loop_boundary"],
}
```

For each branch tournament, score multiple taps and either:

- vote by per-tap tournament winner, or
- learn a small calibrator over tap scores using branch-selection validation.

The evidence does not support hardcoding layer 24 as "the" basal ganglia insertion point. It supports layer 24 as the earliest useful probe, with layers 36-47 carrying the stronger final selection signal.

## Updated collective interpretation after v7

The strongest current interpretation is:

> Ouro's recurrent transformer carries relational correctness/preference evidence across many layers and loop boundaries. The evidence becomes readable early enough to support intervention, but final branch selection is strongest in late raw layer states and boundary states depending on domain. The evaluator is a latent relational comparator, not a scalar reward model, and its best control use is multi-candidate branch selection rather than independent candidate scoring.

The three-paper synthesis now looks cleaner:

- Ouro: looped latent computation creates a trajectory with readable intermediate states.
- RLTT: the useful signal is process/trajectory distributed, not only final-token behavior.
- Relational preference encoding: the readout is fundamentally comparative; pair and tournament evaluations are the correct unit.

The "proto-introspection" claim should remain limited:

> The model exposes internally readable comparative evidence about candidate branches.

not:

> The model has a fixed pointwise self-knowledge scalar.

What is set in stone after these tests is not the exact layer or exact norm count. What is set is the ontology: relational branch comparison over latent trajectories. The implementation should therefore be multi-layer, relational, and tournament-based.

## v8 evaluator placement decision - pre-ensemble

The immediate placement answer from v7 is:

> Do not put the evaluator at one layer only. Use a two-stage/multi-tap layout.

Recommended first implementation:

| role | layer/tap | state | reason |
|---|---:|---|---|
| early gate | 24 | raw natural loop sequence | best aggregate non-math branch selector; early enough for intervention |
| optional early backup | 20 | raw natural loop sequence | nearly tied with layer 24 on aggregate branch selection |
| mid check | 36 | raw natural loop sequence, sometimes normed by task | strong coding/ARC/boolean tap and useful transition point |
| final selector | 45/46 | raw natural loop sequence | strongest hard-MATH and HumanEval selector region |
| final backup | 47 | raw natural loop sequence | boundary-adjacent readout; strong hard-MATH/MBPP |
| boundary | loop boundary | natural sequence | baseline/fallback, not the default control surface |

Operational placement:

- Early gate: raw layer 24.
- Mid check: raw layer 36.
- Final selector: raw layers 45/46/47.
- Boundary: baseline/fallback, not primary.

All layer taps are raw hidden states after the decoder block, before final `model.model.norm`, using the natural 4-loop sequence for that same layer.

Domain-specific placement from current evidence:

| domain/result block | primary tap | secondary taps | interpretation |
|---|---:|---|---|
| Hard MATH | 45/46 raw | 40/47 raw, boundary | final readout dominates; layer 24 is pairwise-good but weaker for four-way selection |
| Cached non-math aggregate | 24 raw | 20 raw, 36 raw, 45 raw | early raw taps beat boundary overall |
| Coding | 36 or 45 raw | 20/24 raw, 47 raw | coding signal is strong across raw taps; late raw taps are best final selectors |

Same placement in implementation-table form:

| Domain | Primary eval tap | Secondary tap | Use |
|---|---:|---:|---|
| Hard MATH | 45/46 raw | 40/47 raw | final branch tournament |
| Cached non-math aggregate | 24 raw | 20, 36, 45 raw | early branch filter + late confirmation |
| Coding | 36 or 45 raw | 20/24, 47 raw | candidate pruning and final selection |

Practical architecture:

```python
early_score = tournament(layer=24, raw=True)
mid_score   = tournament(layer=36, raw=True)
late_score  = tournament(layer=45_or_46, raw=True)
```

The next four evaluator-first steps are now:

1. Run a multi-tap ensemble test over `[20, 24, 36, 40, 45, 47] + boundary`.
2. Compare per-tap voting, summed raw tournament scores, summed debiased tournament scores, and a simple calibrated fusion.
3. Use harder candidate generation for code/reasoning, so the evaluator is not only detecting trivial answer/code mutations.
4. Convert the best ensemble result into the basal-ganglia insertion interface.

The current prior before running that test:

```python
early = [20, 24]
mid = [36]
late = [40, 45, 47]
fallback = ["boundary"]

default_interface = {
    "gate": 24,
    "confirm": 36,
    "select": [45, 47],
}
```

This is a hypothesis to test, not the final interface.

## v9 multi-tap ensemble result

Command:

```bash
venv/bin/python utilities/evaluator/probes/probe_multitap_ensemble.py \
  --max-examples-per-dataset 120 \
  --include-math \
  --math-examples 250 \
  --layers 20,24,36,40,45,46,47 \
  --output-json artifacts/reports/evaluator/probe_multitap_ensemble.json
```

This run executed the four evaluator-first follow-ups:

1. Multi-tap ensemble over `[20, 24, 36, 40, 45, 46, 47] + boundary`.
2. Comparison of per-tap winners, raw vote, debiased vote, summed raw, summed debiased, and held-out calibrated fusion.
3. Harder synthetic candidate generation for code: operator, loop-bound, condition, numeric, and return-expression mutations are preferred before constant-return fallbacks.
4. Basal-ganglia insertion interface updated from the measured result.

Dataset coverage:

| dataset | examples |
|---|---:|
| HumanEval | 120 |
| MBPP sanitized | 120 |
| ARC-Challenge | 120 |
| ARC-Easy | 120 |
| BBH Boolean | 120 |
| BBH Multistep | 120 |
| BBH Logical Five | 120 |
| BBH Tracking | 120 |
| StrategyQA | 120 |
| hard MATH | 250 |

Total: 1,330 candidate-set tournaments.

### Aggregate result

Held-out split, 665 examples:

| method | accuracy |
|---|---:|
| tap_raw:layer_47_natural_raw | 0.755 |
| tap_debiased:layer_47_natural_raw | 0.752 |
| tap_raw:layer_46_natural_raw | 0.740 |
| vote_raw | 0.740 |
| tap_debiased:layer_45_natural_raw | 0.738 |
| tap_debiased:layer_46_natural_raw | 0.738 |
| vote_debiased | 0.738 |
| tap_raw:layer_45_natural_raw | 0.737 |
| calibrated_debiased | 0.737 |
| sum_raw | 0.735 |
| sum_debiased | 0.735 |
| arch_24_36_46 | 0.734 |
| arch_24_36_45 | 0.728 |
| tap_debiased:boundary_natural | 0.716 |
| tap_debiased:layer_24_natural_raw | 0.710 |

Important correction:

> Multi-tap ensembling did not beat the best single late tap on aggregate. The best held-out aggregate selector is raw layer 47. Voting/summing/calibrated fusion are competitive but not superior.

This does not invalidate the multi-stage interface. It changes what each stage should do:

- Layer 24 is an early branch filter / intervention probe, not the final selector.
- Layer 36 is a mid-course check / disagreement detector.
- Layers 45/46/47 are the final selector region.
- Layer 47 is the current best default final tap on the mixed held-out aggregate.

### Per-dataset held-out result

| dataset | best / notable result | accuracy |
|---|---|---:|
| hard MATH | tap_raw:layer_47_natural_raw | 0.888 |
| hard MATH | sum_debiased | 0.872 |
| HumanEval | tap_raw:boundary_natural / tap_raw:layer_47 / tap_debiased:layer_47 | 0.917 |
| MBPP | several taps incl. boundary/l20/l24/l47 | 0.917 |
| ARC-Challenge | vote/sum/arch all tied | 0.933 |
| ARC-Easy | layer 24/46/47 raw/debiased | 0.983 |
| BBH Boolean | layer 20/24 raw/debiased | 0.750 |
| BBH Logical Five | layer 45 debiased | 0.733 |
| BBH Multistep | layer 47 raw/debiased | 0.517 |
| BBH Tracking | layer 46 raw/debiased | 0.283 |
| StrategyQA | vote raw/debiased | 0.683 |

BBH multistep and tracking remain the main failures. The evaluator does not currently provide reliable branch control for those task geometries.

### Calibration result

The simple held-out calibration was not useful enough to justify complexity:

| method | held-out accuracy |
|---|---:|
| best single tap, layer 47 raw | 0.755 |
| calibrated debiased fusion | 0.737 |
| sum debiased | 0.735 |

The learned global weights were nearly flat:

| tap | weight |
|---|---:|
| layer_45_natural_raw | 0.129 |
| layer_46_natural_raw | 0.128 |
| layer_36_natural_raw | 0.126 |
| layer_47_natural_raw | 0.126 |
| layer_40_natural_raw | 0.126 |
| layer_24_natural_raw | 0.124 |
| boundary_natural | 0.120 |
| layer_20_natural_raw | 0.120 |

This suggests a simple accuracy-weighted calibrator is too weak. If calibration is revisited, use a real learned fusion head on held-out branch tournaments, not scalar validation-accuracy weights.

## Final evaluator insertion interface after v9

Use a staged controller, but make the final decision late.

Default active evaluator layers are exactly three:

```python
active_layers = [24, 36, 47]
```

Layers 45/46 are not part of the default active controller. They remain diagnostic/fallback taps only.

```python
early_probe = evaluator_tournament(layer=24, raw=True)   # filter / fork pressure
mid_probe   = evaluator_tournament(layer=36, raw=True)   # consistency / disagreement check
late_score  = evaluator_tournament(layer=47, raw=True)   # default final selector
alt_late    = evaluator_tournament(layer=45_or_46, raw=True)  # domain fallback
```

Recommended first production-ish policy:

```python
if early_probe.confident_bad(branch):
    prune_or_downweight(branch)

if mid_probe.disagrees_strongly_with_late(branch):
    keep_branch_for_extra_rollout_or_recompare(branch)

winner = argmax_i late_score.sum_pairwise_wins(branch_i)
```

For hard math:

```python
final_selector = layer_47_raw
fallbacks = [45_raw, 46_raw, boundary]
```

For coding:

```python
final_selector = layer_47_raw
fallbacks = [45_raw, 36_raw, 24_raw]
```

For mixed non-math / reasoning:

```python
early_filter = layer_24_raw
mid_check = layer_36_raw
final_selector = layer_47_raw
```

The basal-ganglia metaphor should be implemented as:

> early tap proposes/prunes, mid tap detects uncertainty/disagreement, late tap selects.

not:

> average all taps and hope the ensemble is better.

The central engineering conclusion after v9:

> Put the evaluator into the loop at multiple taps for control visibility, but let raw layer 47 make the default final branch-selection call. Layer 24 remains important because it is early enough to affect the trajectory, not because it is the best selector.

---

# v10 — Loop geometry on HH text: Thinking vs RLTT cross-backbone (2026-05-14)

**Source script:** `utilities/evaluator/probes/probe_loop_geometry_hh.py`
**Source memo:** `handoff_claude_code_2026-05-14.md` §6 (Experiments 1A + 1B)
**Inputs:** 200 HH-RLHF test pairs (chosen + rejected), seed=42 sampling, max_length=384,
  mean-pool over valid tokens for geometry.
**Backbones:** loaded sequentially in bf16 (12 GB VRAM cannot hold a fp32 2.6B Ouro);
  captured states immediately cast to fp32. Tokenizer shared (`models/ouro_rltt_local`)
  to ensure identical input IDs into both models.
**Head:** `artifacts/checkpoints/evaluator/pairwise_epoch2.pt` — the published 5M-param GRU.
**Outputs:**
- `artifacts/reports/evaluator/probe_loop_geometry_hh.json` — full numbers
- `artifacts/reports/evaluator/hh_loop_states_200_thinking.pt` — per-token states + masks (Thinking, fp32, 2.5 GB)
- `artifacts/reports/evaluator/hh_loop_states_200_rltt.pt` — same, RLTT (2.5 GB)

This chapter takes the v3–v9 picture (which was Thinking-only) and asks the
two questions the handoff laid out:

- **1A.** Is the loop trajectory genuinely *converged* on text inputs, or
  *trajectory-distributed*? The earlier 0.08 cosine measurement was on
  off-distribution GridEncoder embeddings; this is the same measurement on
  HH-RLHF text.
- **1B.** Does RLTT's trajectory-wide credit assignment leave a *measurable
  geometric signature* in the boundary states that Thinking does not have,
  even though zero-shot transfer of the published head is ~0.2 pp? This is
  the W1/W2/W3 verdict that decides which backbone the debiased Experiment-2
  retrain should target.

## Headline numbers

### 1A — Per-backbone intrinsic geometry

Cross-loop pair cosines, mean-pooled over valid tokens, pooled across chosen + rejected:

| Pair | Thinking | RLTT |
|---|---:|---:|
| L1↔L2 | +0.8651 | +0.8707 |
| L1↔L3 | +0.7605 | +0.7706 |
| **L1↔L4** | **+0.7230** | **+0.7350** |
| L2↔L3 | +0.9754 | +0.9764 |
| L2↔L4 | +0.9590 | +0.9608 |
| L3↔L4 | +0.9961 | +0.9961 |
| mean off-diag | +0.8798 | +0.8849 |

Within-loop cos(h_chosen, h_rejected), Thinking: L1 0.963, L2 0.955, L3 0.955, L4 0.955 — chosen and rejected are near-collinear at every loop; the relational signal is a small projection orthogonal to that.

Per-loop L2 norm (Thinking): 25.18 / 25.42 / 25.13 / 25.22 — flat across loops.

Effective dimensionality (top-32 cumulative variance ratio over n=400 stacked pooled vectors), both backbones: ≈ 0.60 across all four loops.

Diff-vector cross-loop cosines, Thinking — `cos(h_chosen − h_rejected at L_i, same at L_j)`:

| Pair | Thinking |
|---|---:|
| d1↔d2 | +0.8946 |
| d1↔d3 | +0.8811 |
| **d1↔d4** | **+0.8748** |
| d2↔d3 | +0.9753 |
| d2↔d4 | +0.9656 |
| d3↔d4 | +0.9920 |

K-norm pair-cos sweep, mean off-diag (apply `model.model.norm` K times then mean-pool):

| K | Thinking mean | Thinking min | RLTT mean | RLTT min |
|---:|---:|---:|---:|---:|
| 0 | +0.8798 | +0.7230 | +0.8849 | +0.7350 |
| 1 | +0.8851 | +0.7362 | +0.8907 | +0.7494 |
| 2 | +0.8860 | +0.7389 | +0.8925 | +0.7542 |
| 4 | +0.8807 | +0.7295 | +0.8902 | +0.7513 |
| 8 | +0.8577 | +0.6858 | +0.8744 | +0.7226 |

Cosines do **not** climb toward 1.0 as K increases — they peak at k=2 and recede slightly.
Section 6 of the handoff predicted that x8 norm would collapse all loops to near-cosine-1.0;
that prediction is refuted on text.

### 1B — Cross-backbone geometry

| Quantity | Value |
|---|---:|
| Mean cos(h_T, h_R), all loops pooled | **0.9992** |
| Min  cos(h_T, h_R), per-loop minimum (L1) | 0.9985 |
| Per-loop L1 / L2 / L3 / L4 | 0.9985 / 0.9993 / 0.9995 / 0.9994 |

Score agreement using the published `pairwise_epoch2.pt` head on both backbones'
captured states (n=200):

| Quantity | Value |
|---|---:|
| Pearson(score_T, score_R) | **0.9906** |
| Mean \|Δscore\| | 0.0707 |
| Median \|Δscore\| | 0.0457 |
| Decision agreement rate | **0.9950** |
| canonical_acc Thinking (200-pair subsample) | 0.9500 |
| canonical_acc RLTT (200-pair subsample) | 0.9450 |
| score_T mean ± std | 1.602 ± 0.964 |
| score_R mean ± std | 1.596 ± 0.978 |

The 0.5 pp canonical-accuracy gap on this 200-pair subsample is consistent with
the ~0.2 pp full-test-set transfer gap referenced in the handoff — within sampling
noise. The published 95.2% on the full 8,552-test split maps to 0.950 here.

The readout-direction decomposition (handoff §6 Experiment 1B step h) was **skipped**:
with mean cos(h_T, h_R) = 0.9992, the deltas are too small to support a meaningful
aligned/orthogonal decomposition. The W1 verdict applies cleanly without it.

## The four verdicts

1. **Intrinsic-geometry (Thinking):** *loops are converged on text* (mean off-diag 0.880 > 0.85 threshold). **Caveat below.**
2. **Intrinsic-geometry (RLTT):** *loops are converged on text* (mean off-diag 0.885).
3. **Gamma-attractor (Thinking):** *not confirmed.* k8 min off-diag = 0.686 (verdict requires > 0.95).
4. **Gamma-attractor (RLTT):** *not confirmed.* k8 min off-diag = 0.723.
5. **Cross-backbone:** **W1 — RLTT did not move loop-boundary geometry.** mean cos(h_T, h_R) = 0.9992 > 0.99 threshold.
6. **Experiment-2 direction:** **train the new debiased head on Thinking activations.** RLTT activations would not give the head anything visible that Thinking doesn't already expose.

### Caveat on the intrinsic verdict

The simple "mean off-diag cosine > 0.85" rule fires "converged," but the
distribution is bimodal: L2-L4 are very tight (cos > 0.96 pairwise) while
**L1 sits apart from L4 at 0.72** on Thinking and 0.74 on RLTT. That L1↔L4
number lands inside the handoff's prediction band of [0.3, 0.8] — i.e. the
intent of the prediction (mild refinement, not full convergence) is in fact
corroborated, just not in the way the threshold rule reads it. A more
faithful summary is: **L1 is geometrically distinct from the L2-L4 cluster;
L2 onward is essentially a single state under mean-pooling on text.**

The diff-vector picture matches: d1↔d4 = 0.87 is the loosest pairwise diff
cosine, and d2↔d4 onward are above 0.97. The relational direction picked up
at loop 1 is meaningfully different from the relational direction picked up
by the post-loop-1 cluster, even though the chosen-vs-rejected absolute
cosine stays near 0.96 throughout.

This is consistent with the v8/v9 multi-tap finding that loop 1 / earlier
layers do something separate from later loops/layers, and with the
proto-introspection framing in the v6 update: **trajectory-distributed,
not trajectory-converged**.

## Predictions vs outcomes (handoff §6, Experiment 1A)

| Prediction | Outcome |
|---|---|
| L1↔L4 cosine on HH text in [0.3, 0.8] | **Confirmed** — Thinking 0.72, RLTT 0.74. |
| Cross-loop diff-vector cosines higher than within-loop chosen/rejected cosines | **Mixed.** d2-d4 cosines (0.97-0.99) exceed within-loop CR cosine (≈0.96), but d1-d4 = 0.87 is lower than CR (≈0.96). The L1 diff direction is a separate axis from the L2-L4 diff direction. |
| After x8 norm, all loops collapse to cos ≈ 1.0 (gamma-attractor confirmation) | **Refuted.** Cosines peak at k=2 and recede slightly by k=8 (k8 min = 0.69 / 0.72). The learned final-norm gamma does not act as a uniform attractor pulling all loops together on HH text. The v6 "gamma-power-filter" mechanism may still hold for individual states, but it is not the mechanism that produced the prior x4/x8 metric jumps in the v4–v5 sweeps. |

## Cross-backbone summary (handoff §6, Experiment 1B)

| Verdict logic | Met? |
|---|---|
| Mean cos > 0.99 across all loops | **Yes** (min L1 = 0.9985) — **W1**. |
| W2 (mean cos < 0.99 AND Pearson > 0.99 AND ‖orth‖ ≫ ‖aligned‖) | Not applicable; W1 fired first. |
| W3 (mean cos < 0.99 AND Pearson < 0.99) | Not applicable. |

W1 implication: **the new debiased head should be trained on Thinking activations.**
RLTT activations are not expected to unlock a head-invisible subspace because no such
subspace exists at the boundary-state level on HH text — the two backbones produce
essentially the same boundary states (per-loop cos ≥ 0.998).

This *also* tells us something about RLTT itself on the HH distribution: RLTT's
trajectory-wide credit assignment did not measurably reshape the boundary-state
geometry on text inputs. That is consistent with the ~0.2 pp full-test-set transfer
gap reported in the handoff. The trajectory-wide signal RLTT shapes shows up in
math/AIME numbers, not in HH boundary geometry — at least not at the
per-token-mean-pool level this probe measures.

## Method notes

- Float32 throughout the head pipeline; the model forward runs in bf16
  because two 2.6B Ouro backbones in fp32 do not fit in 12 GB VRAM
  (confirmed with the user before launch). Captured states are cast to
  fp32 before any geometry math or head call. Matches the existing probe
  stack (`probe_hh_norm_mechanism_centered.py`) and is numerically
  consistent with the GRU head's training dtype.
- Sequential backbone load: Thinking, save .pt, free, then RLTT, save
  .pt, free. All cross-backbone math runs from the saved .pt files in a
  third pass without holding either model in VRAM.
- Tokenizer: a single Ouro tokenizer (the one shipped with
  `models/ouro_rltt_local`) is used for both backbones to ensure
  identical input IDs.
- Geometry pooling: mean over mask-valid tokens, to avoid biasing the raw
  geometry by the head's learned attention pool.
- Score agreement: head run unmodified on each backbone's per-token
  states. Head's attention pool runs internally as in production.
- Capture wall time: ~84 s per backbone (200 examples × 2 prompts =
  400 forwards each at ~3 fwd/s on the 5070 Ti Laptop). One-time
  Thinking download was ~14 min.

## What v10 enables for the rest of the program

- **Backbone choice for Experiment 2:** Thinking. (Confirmed by W1.)
- **State source:** the saved `hh_loop_states_200_thinking.pt` is sized for
  sanity tests, not for the 50k-train / 8,552-test retrain. The retrain
  needs a fresh capture pass over the full split, but using the same hook
  + numerical contract this probe established.
- **Numerical contract for retrain:** bf16 model forward, fp32 captured
  states, fp32 head training, swap augmentation, λ_sym ∈ {0.1, 0.3, 1.0}
  symmetric-offset penalty, track centered_acc as the first-class metric
  per handoff §6 Experiment 2 spec.

## Open follow-up before Experiment 2 (non-blocking)

The intrinsic-verdict caveat opens a sub-question: **does the head benefit
from L1's distinct projection, or is L1 noise relative to the head's
readout?** A small follow-up probe — run the head on states with L1
zeroed/replaced by L2 and measure the per-loop contribution to canonical
and centered scores — would tell us whether the multi-tap intuition from
v8/v9 has a within-loop analogue. **Resolved below (L1 ablation, 2026-05-14).**

## L1 ablation (2026-05-14)

**Script:** `utilities/evaluator/probes/probe_l1_ablation.py`
**Raw output:** `artifacts/reports/evaluator/probe_l1_ablation.json`
**Protocol:** zero-shot, swap-balanced, fp32, no model forward. Pure
post-hoc analysis on the saved v10 Thinking states
(`hh_loop_states_200_thinking.pt`, 200 HH-RLHF pairs, per-token L1..L4).
The published `pairwise_epoch2.pt` head is run unmodified; each variant is
the named loop state replicated into all four of the head's input slots
(the `only_loop_N` convention from v3–v6). `natural[L1-4]` = the natural
`[L1,L2,L3,L4]` sequence, reference only, not in the verdict.

`bias_to_signal` here is `|mean(s_cr + s_rc)| / std(s_cr − s_rc)` (the
formula specified for this probe; note it differs from the
`|mean(bias)|/|mean(antisym)|` definition used in the v6 tables, so these
numbers are not directly comparable to the 3.42 / 14.39 figures above).

| variant | canonical | centered | bias/sig | pos_rate | strict_sign_rev |
|---|---:|---:|---:|---:|---:|
| L4-only* (baseline) | 0.9500 | 0.5950 | 1.416 | 0.950 | 0.175 |
| L1-only* | 0.9150 | 0.5750 | 1.404 | 0.915 | 0.180 |
| L2-only* | 0.9500 | 0.6000 | 1.506 | 0.950 | 0.155 |
| **mean(L1,L4)*** | 0.9400 | **0.6200** | 1.472 | 0.940 | 0.150 |
| natural[L1-4] (ref) | 0.9500 | 0.5950 | 1.398 | 0.950 | 0.180 |

\* verdict-relevant. No degeneracy: flip rates 0.15–0.18 and pos_rates
0.91–0.95 are well clear of the constant-output failure mode
(strict_sign_reversal < 0.05 AND pos_rate > 0.95).

**Verdict: L1 carries readable relational signal — Experiment 2 should add
an `early_tap=True` option that ingests L1 alongside L4.**

Both verdict legs fired, but they are not equally strong and the honest
reading matters for the Experiment 2 spec:

- **Weak leg:** L1-only centered_acc = 0.5750 clears the ≥ 0.55 bar, but
  only just, and it is *below* both L4-only (0.5950) and L2-only (0.6000).
  L1 alone is a slightly *worse* comparator than the L2-L4 cluster — it is
  not noise, but it is not independently strong either. On its own this
  leg would be borderline.
- **Strong leg (the real result):** `mean(L1, L4)` centered_acc = 0.6200
  beats every single-loop variant — L4-only by +2.50 pp, L2-only by
  +2.00 pp, L1-only by +4.50 pp — while only giving up 1.0 pp of canonical
  accuracy (0.940 vs 0.950). Fusing the two geometrically-distinct
  endpoints of the bipartite trajectory measurably improves the
  *antisymmetric* (debiased) signal, which is exactly the metric
  Experiment 2 optimizes. This is the load-bearing finding.

Two consistency checks against v10 land cleanly: **L2-only (0.600) ≈
L4-only (0.595)** confirms the v10 result that L2-L4 are one functional
state for the head; and **natural[L1-4] (0.595) = L4-only (0.595)** to
four decimals, consistent with the v10 score-agreement step.

**Implication for the Experiment 2 spec.** The handoff's recipe (debiased
retrain on the published L4-only architecture) is no longer the only
option on the table. The recommended modification:

- Add an `early_tap` mode that feeds the head a fused L1⊕L4 input (mean,
  or a learned 2-vector combine) instead of L4-replicated.
- Keep the L4-only arm as the control so the fusion gain is measured under
  the debiased objective, not just zero-shot — the +2.5 pp here is on the
  *pointwise-trained* head and is a floor, the same way the v6 hard-MATH
  0.882 was a floor.
- Everything else in the handoff §6 Experiment 2 spec (swap augmentation,
  λ_sym ∈ {0.1, 0.3, 1.0} symmetric-offset penalty, centered_acc as the
  first-class checkpoint metric, Thinking backbone per W1) is unchanged.

This is a spec change, not a stop. Experiment 2 should run the fused arm
and the L4-only arm side by side and keep whichever has the higher
*centered* accuracy under the debiased objective.

### L1↔L4 weighted-mean α sweep (2026-05-14)

**Script:** `utilities/evaluator/probes/probe_l1_alpha_sweep.py`
**Raw output:** `artifacts/reports/evaluator/probe_l1_alpha_sweep.json`

Derisks the Experiment-2 arm design. Question: is the mean(L1,L4) fusion
gain about *mixing weight* (a 2048-dim weighted mean captures it → the
learned-α arm is high-prior) or about the head seeing L1 and L4 as
*separate slots* (only a 4096-dim concat/diff arm captures it → those arms
are high-prior)? Sweep of `fusion = α·L1 + (1−α)·L4`, frozen head,
swap-balanced, zero-shot, 200 saved Thinking pairs. α=0 ⇒ L4-only,
α=1 ⇒ L1-only, α=0.5 ⇒ mean (consistency check against the 0.620 above).

| α | canonical | centered | bias/sig |
|---:|---:|---:|---:|
| 0.00 (L4) | 0.950 | 0.595 | 1.416 |
| 0.10 | 0.950 | 0.595 | 1.436 |
| 0.20 | 0.945 | 0.595 | 1.456 |
| 0.30 | 0.945 | 0.595 | 1.468 |
| 0.35 | 0.950 | 0.615 | 1.472 |
| 0.40 | 0.950 | 0.610 | 1.474 |
| **0.45** | 0.945 | **0.620** | 1.474 |
| **0.50 (mean)** | 0.940 | **0.620** | 1.472 |
| 0.55 | 0.930 | 0.610 | 1.468 |
| 0.60 | 0.930 | 0.610 | 1.464 |
| 0.70 | 0.930 | 0.605 | 1.451 |
| 0.80 | 0.925 | 0.595 | 1.436 |
| 0.90 | 0.930 | 0.590 | 1.420 |
| 1.00 (L1) | 0.915 | 0.575 | 1.404 |

α=0.50 reproduces centered 0.620 **exactly** — internal consistency with
the L1-ablation `mean(L1,L4)` row holds. The curve is a step from an
L4-dominated regime (α ≤ 0.30 ≈ 0.595) up to a broad plateau (α 0.35–0.50
≈ 0.615–0.620), then a monotone decline toward L1. **No α beats the
unweighted mean's 0.620** — peak gain over mean is 0.00 pp, below the
0.5 pp "meaningful" bar.

`bias_to_signal` is essentially flat across the whole sweep (1.40–1.47)
and α=0.50 is the lowest among the two near-optimal αs (1.472) — i.e.
weighting does not help the symmetric-offset problem either, so the
debiased objective's λ_sym penalty is doing work orthogonal to L1 fusion.

**Verdict (matches the pre-registered "flat ≈ 0.620" case): the fusion
gain is NOT about mixing weight.** A 2048-dim weighted mean cannot exceed
the simple average, so the +2.5 pp the L1-ablation found has to come from
the head seeing L1 and L4 as *separate* signals. Concat/diff do
structurally different work than any 2048-dim mixer can.

**Decided Experiment-2 arm priority for tonight:**

> **concat [L1;L4] > diff [L4; L4−L1] > weighted-mean (seed α=0.50) > L4-only control**

per λ_sym ∈ {0.1, 0.3, 1.0}, checkpoint by centered_acc, Thinking backbone
(W1), 4 arms × 3 λ = 12 runs. (Recall the concat≈diff equivalence caveat:
linearly the same function class; any divergence between them is a
LayerNorm-induced conditioning effect, not extra capacity.) The
weighted-mean arm is kept but demoted — it is cheap and a useful negative
control; if it matches concat/diff under the debiased objective, that
*reopens* the "it was just averaging" hypothesis under retraining.

