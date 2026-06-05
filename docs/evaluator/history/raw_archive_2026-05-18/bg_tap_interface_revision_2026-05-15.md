# BG Tap Interface Revision After Layer 24/36 Probe

**Date:** 2026-05-15  
**Status:** Supersedes the uniform tap-interface assumption in `post_v10_synthesis_2026-05-14_v2.md` section 4.  
**Scope:** Tap parameterization only. The BG controller program, generated-branch tournament gate, and mixed-domain Phase 1 training objective remain active.

## Executive Summary

The blocking layer-24/36 geometry probe invalidates the assumption that all BG taps should use the same `(L1, L4)` fused input.

The old assumption was:

```python
tap24 = fused_L1_L4(h24_L1, h24_L4)
tap36 = fused_L1_L4(h36_L1, h36_L4)
tap47 = fused_L1_L4(h47_L1, h47_L4)
```

The probe result says the geometry is layer-specific:

| Layer | L1-L4 cosine | L2-L4 cosine | mean off-diag | Verdict |
|---:|---:|---:|---:|---|
| 24 | +0.9284 | +0.9790 | +0.9632 | fully converged |
| 36 | +0.9347 | +0.9822 | +0.9682 | fully converged |
| 47 | +0.7350 | +0.9608 | +0.8849 | bipartite |

The revised interface is therefore heterogeneous:

```python
early_score = tap24_single(h24_loop_k)
mid_score   = tap36_single(h36_loop_k)
late_score  = tap47_fused(h47_L1, h47_L4)
```

Layer 24 and 36 taps should be 2048-dim single-state heads unless a direct ablation shows that loop fusion helps. Layer 47 remains a 4096-dim fused endpoint head because the bipartite L1-vs-L2/L3/L4 structure persists there.

The 1000-example old-head-compatible follow-up probe does **not** select a meaningful 24/36 loop input. Centered accuracy is near chance across the candidate constructions:

| Layer | Best old-head row | centered | canonical | Note |
|---:|---|---:|---:|---|
| 24 | L4 replicated | 0.512 | 0.763 | tied with natural/L2/mean within noise |
| 36 | L4 replicated | 0.513 | 0.760 | tied with mean/L1/natural within noise |
| 47 | mean replicated | 0.586 | 0.944 | old-head-compatible control only |

Therefore L4 at 24/36 is a conservative capture/default choice, not an empirical winner. The important conclusion is that 24/36 need newly trained 2048-dim pairwise heads; do not rely on zero-shot transfer from the boundary-trained published head.

## What Is Blocked

Blocked:

> Every tap at layers 24, 36, and 47 reads `(L1, L4)` because the L1/L4 bipartite structure exists at every tap layer.

The probe shows this is false. The v10/L1-L4 complementarity finding is a late-layer/readout-layer property, not a universal property of all tapped layers.

Not blocked:

- The BG controller concept.
- The active layer set `[24, 36, 47]`.
- The late layer-47 selector role.
- The generated-branch tournament gate.
- Phase 1 mixed-domain `L_eval` with `L_align = 0`.
- The need to measure early/mid/late disagreement before deciding whether alignment is useful.

## Mechanistic Reading

Layers 24 and 36 are not exposing useful "trajectory endpoint contrast" between loop 1 and loop 4. Their loop states mostly cluster together, so their value is as intermediate layer-depth checkpoints:

- What does the representation look like halfway through the layer stack?
- What does it look like three-quarters through?
- Does the intermediate judgment disagree with the late final judgment?

Layer 47 is different. It still has a bipartite endpoint structure:

- L1 is geometrically distinct.
- L2/L3/L4 form the tight late-loop cluster.
- The prior L1/L4 fusion result remains relevant at this late readout layer.

The updated mental model:

| Tap | Geometry | Control role |
|---:|---|---|
| 24 | converged intermediate state | early prune / viability signal |
| 36 | converged intermediate state | mid-course uncertainty / consistency signal |
| 47 | bipartite final trajectory state | final selector |

This is cleaner than the uniform design: early and mid taps read state-of-processing checkpoints, while the late tap reads final trajectory contrast.

## Revised Tap Spec

### Tap 24

Default shape:

```python
x24 = h24_loop_k       # [B, T, 2048]
early_score = tap24_single(x24)
```

Candidate loop choices:

- `h24_L1`: latency candidate for the trained-head ablation.
- `h24_L2`: optional latency/compromise candidate if storage and time allow.
- `h24_L4`: conservative capture/default choice if implementation simplicity matters.
- `mean(h24_L1, h24_L2, h24_L3, h24_L4)`: offline control, probably unnecessary if convergence is real.

Do not use `concat(h24_L1, h24_L4)` by default. At layer 24 the two inputs are near-duplicates, and concat likely wastes parameters or adds conditioning noise.

The old published head should not be used to choose among these candidates: its best layer-24 centered accuracy was 0.512, with the top four rows effectively tied.

### Tap 36

Default shape:

```python
x36 = h36_loop_k       # [B, T, 2048]
mid_score = tap36_single(x36)
```

Candidate loop choices:

- `h36_L1` or `h36_L2`: latency candidates for the trained-head ablation.
- `h36_L4`: conservative capture/default choice if implementation simplicity matters.
- `mean(h36_L1, h36_L2, h36_L3, h36_L4)`: offline control.

Do not use `concat(h36_L1, h36_L4)` by default unless a direct ablation shows a real centered or tournament gain.

The old published head should not be used to choose among these candidates: its best layer-36 centered accuracy was 0.513, with L4/mean/L1/natural effectively tied.

### Tap 47

Default shape remains fused:

```python
if experiment2_diff_wins:
    x47 = torch.cat([h47_L4, h47_L4 - h47_L1], dim=-1)  # [B, T, 4096]
else:
    x47 = torch.cat([h47_L1, h47_L4], dim=-1)           # [B, T, 4096]

late_score = tap47_fused(x47)
```

Layer 47 is the only default tap that inherits the Experiment 2 concat/diff decision directly.

## Phase 1 Update

Old Phase 1:

```python
tap24_fused_4096
tap36_fused_4096
tap47_fused_4096
```

Revised Phase 1:

```python
tap24_single_2048
tap36_single_2048
tap47_fused_4096
```

Training still uses:

- frozen RLTT backbone for local Phase 1 tap training,
- mixed-domain relational `L_eval`,
- swap augmentation,
- symmetric-offset penalty,
- `L_align = 0`,
- generated-branch tournament evaluation as the primary BG gate.

Experiment 2 still matters, but mainly for:

- the layer-47 fused architecture,
- concat-vs-diff conditioning,
- lambda_sym choice,
- debiased evaluator training protocol.

Experiment 2 should not be treated as an initializer for all three taps unless 24/36 single-state heads are explicitly adapted from the L4-only control arm.

## Completed Old-Head Follow-Up Probe

The cheap old-head-compatible tap probe has now run at n=1000:

For each layer in `{24, 36, 47}`, it compared:

| Variant | Purpose |
|---|---|
| `L1_replicated` | earliest loop state through the old four-loop head shape |
| `L2_replicated` | early but likely converged state |
| `L4_replicated` | late-loop single-state baseline |
| `mean_replicated` | convergence sanity/control |
| `natural_seq` | old natural four-loop sequence control |

Metrics:

- canonical accuracy,
- centered accuracy,
- bias/signal,
- flip diagnostics,
- raw vs debiased agreement.

Key result:

> The old boundary-trained head basically cannot read HH-centered relational signal from layers 24/36 zero-shot. It is order-sensitive there, but not aligned enough to choose the branch reliably.

Do not run more old-head zero-shot probes on 24/36 to choose a loop. The next useful step is to train small single-state 2048-dim heads for layer 24 and 36 candidates, then compare HH centered diagnostics and generated-branch tournament performance.

Suggested trained-head comparison:

| Layer | Required candidates | Optional candidate |
|---:|---|---|
| 24 | L1, L4, mean(L1..L4) | L2 |
| 36 | L1, L4, mean(L1..L4) | L2 |

## Corrected Math BG-Gate Pilot

A corrected math generated-branch pilot has now run; full details are in `math_bg_gate_pilot_2026-05-15.md`.

The final corrected pilot fixed two data-pipeline issues before evaluation:

- mixed sampling now builds a balanced GSM8K/MATH candidate pool before truncation;
- branch generation now supports small return-sequence sub-batches to avoid CUDA OOM on longer MATH prompts.

Corrected generation used 100 candidate prompts, balanced 50 GSM8K / 50 MATH. Exact-answer filtering kept 33 tournaments: 26 GSM8K and 7 MATH. The run generated 91 correct and 309 incorrect attempts, with 0 unparseable attempts.

Training small antisymmetric pairwise heads on the corrected tournaments produced a strong but tiny held-out readout:

| Config | eval top1 | eval pairwise | eval condorcet | cycle |
|---|---:|---:|---:|---:|
| `24_L1` | 1.000 | 1.000 | 1.000 | 0.000 |
| `24_L4` | 1.000 | 1.000 | 1.000 | 0.000 |
| `24_mean` | 1.000 | 1.000 | 0.875 | 0.000 |
| `36_L4` | 1.000 | 1.000 | 1.000 | 0.000 |
| `36_mean` | 1.000 | 1.000 | 1.000 | 0.000 |
| `36_L1` | 1.000 | 0.962 | 1.000 | 0.000 |
| `47_concat_L1_L4` | 1.000 | 0.962 | 0.875 | 0.000 |

Interpretation:

- This validates 24/36 single-state trained heads as viable math branch selectors.
- This does not decide the final readout choice; eval n=8 is too small.
- Layer 47 is not privileged on this pilot. It remains a late baseline, not a proven winner.
- The next run must be source-stratified and budget-aware, because MATH kept-yield is strongly shaped by generation verbosity and token budget.

## Revised Sequencing

1. Treat the two completed blocking probes as resolved:
   - Layer 24/36 bipartite probe: resolved, old uniform fusion assumption failed.
   - Per-layer Thinking-vs-RLTT comparison: resolved, v9 layer choices transfer.
2. Treat the old-head-compatible tap probe as resolved: 24/36 are near chance under the old checkpoint, so L4 is not a decisive winner.
3. Treat the corrected math trained-head pilot as positive but not decisive: 24/36 single-state heads are viable, but the gate-scale source-stratified follow-up still selects the final readout.
4. Update any Phase 1 capture/training code to support heterogeneous tap specs.
5. Continue Experiment 2 for the layer-47 fused head and debiased objective.
6. Build the generated-branch tournament dataset with source-stratified and budget-aware math reporting.
7. Start RLTT Phase 1 only after the trained 24/36 single-state readout choice is selected.

## Revised One-Liner

RLTT exposes converged intermediate control states at layers 24/36 and a bipartite final comparison state at layer 47. The BG controller should therefore use single-state early/mid taps and a fused L1/L4 late tap, with disagreement across layers, not L1/L4 fusion at every layer, as the main control signal.
