# DualAnchor Architecture Baseline

Updated: 2026-05-31

This document is the memorable entry point for the architecture-shaped DualAnchor branch/prune baseline.

## Status

`DUALANCHOR_ARCHITECTURE_LOOPED_V3_STATUS = ARCHITECTURE_LOOPED_SURVIVAL_READY_TERMINAL_DEFER_REQUIRED`

`BG_DUALANCHOR_PHASE2A_READINESS_V3_VERDICT = READY_WITH_TERMINAL_DEFER`

## What Is Locked

DualAnchor now means the two old-anchored, branch-valid tap identities:

- `MIX_CODE_REASONING`
- `MIX_OBJECTIVE_ALL`

They are treated as unified branch/action/content selectors. The earlier split framing of "old taps for content" plus "branch taps for branching" is no longer the leading architecture.

The architecture-shaped branch/prune loop is:

`L1_24 -> L1_36 -> L1_47 -> L2_24 -> L2_36 -> L2_47 -> L3_24 -> L3_36 -> L3_47 -> L4_24 -> L4_36 -> terminal L4_47`

At each nonterminal stage:

- current survivors spawn perturbation children,
- inherited survivors remain candidates,
- DualAnchor scores candidates pairwise at that loop/layer,
- `mean_floor_very_loose` thresholding runs,
- disagreement/clean/diversity rescue can preserve candidates,
- hard budget is `8`,
- survivors flow to the next stage,
- after nonterminal `*_47`, survivors loop forward to the next loop's layer 24.

At terminal `L4_47`:

- forced top1 is diagnostic,
- confidence-gated top1 is allowed only when the gate fires,
- otherwise the terminal survivor set is deferred/retained.

## V3 Metrics

Source artifact:

`artifacts/reports/probes/bg_dualanchor_architecture_looped_stratified_probe_v3_2026-05-31/`

Headline:

| Metric | Value |
| --- | ---: |
| tasks | 48 |
| domains | 24 reasoning / 24 science |
| rows generated/evaluated | 3454 |
| nonterminal stage decisions | 528 |
| stage oracle retention | 0.9848 |
| terminal oracle retained | 1.0000 |
| forced terminal top1 oracle | 0.9167 |
| forced terminal top1 reward | 0.2625 |
| terminal best-survivor reward | 0.3167 |
| reward-diverse rate | 0.2292 |
| positive-oracle rate | 0.3542 |
| local false prunes | 8 |
| false-prune recovery | 8/8 |

Hard slices:

| Slice | Count | Terminal Oracle Retained | Forced Top1 Oracle | Forced Top1 Reward | Best Reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 48 | 1.0000 | 0.9167 | 0.2625 | 0.3167 |
| reasoning | 24 | 1.0000 | 0.8750 | 0.4833 | 0.5833 |
| science | 24 | 1.0000 | 0.9583 | 0.0417 | 0.0500 |
| positive-oracle | 17 | 1.0000 | 0.8824 | 0.8706 | 1.0000 |
| reward-diverse | 11 | 1.0000 | 0.6364 | 0.3091 | 0.5455 |
| positive and reward-diverse | 6 | 1.0000 | 0.6667 | 0.6333 | 1.0000 |

Interpretation:

Survival is strong. Terminal forced top1 is inflated by tie-heavy tasks and is weak on reward-diverse hard slices. The correct locked baseline is therefore not unconditional terminal top1; it is terminal confidence/defer.

## L47 Result

Candidate-tree replay ablation:

| Ablation | Oracle Retained vs Full | Best Reward | Forced Top1 Oracle |
| --- | ---: | ---: | ---: |
| full L47 enabled | 1.0000 | 0.3167 | 0.9167 |
| L2_47 only | 0.1875 | 0.0667 | 0.2500 |
| no L47 perturbation | 0.0417 | -0.1000 | 0.0417 |
| nonterminal L47 disabled | 0.0417 | -0.1000 | 0.0417 |
| terminal-only L47 | 0.0417 | -0.1000 | 0.0417 |

Earlier-loop L47 is not diagnostic-only. It must stay active unless regenerated ablations later contradict the replay result.

## Lineage Result

| Metric | Value |
| --- | ---: |
| child count | 3406 |
| child improved parent | 0.0390 |
| child tied parent | 0.9281 |
| child worsened parent | 0.0329 |
| mean child-parent delta | +0.00464 |

Recursive perturbation is a useful local search process, not a large-jump generator. Most children tie parents; improvements slightly exceed harm.

## Boundary Conditions

The prompt-only carry equivalence probe validated decoder-layer carry at layers 24/36/47 under the cumulative-hook approximation:

`BG_DUALANCHOR_PROMPT_CARRY_REFERENCE_V3_VERDICT = PRIOR_VALIDATION_ACCEPTED`

This does not claim autoregressive branch-specific KV/cache fork-carry. It also does not claim compute savings.

No steering was tested.

## Source Docs

- `history/bg-run-notes/dualanchor-two-tap/bg_dualanchor_architecture_looped_stratified_probe_v3.md`
- `dualanchor-tap-evolution.md`
- `branch-generation-and-survival.md`
- `terminal-selection-and-arbiters.md`



## Autoregressive KV/cache branch-carry validation v1 (2026-06-01)

`autoregressive_kv_branch_carry_v1` validated that Ouro supports generation-time,
branch-specific KV/cache carry (distinct from already-validated prompt-only layer
carry, which used `use_cache=False`). Using the UniversalTransformerCache
(slot = `current_ut*num_hidden_layers + layer_idx`, 4 loops × 48 layers = 192 slots),
the validation ladder Levels 0–5 all pass within bf16:

- L0 cached decode == full recompute (prefill bit-exact; decode RMS ~0.05–0.2, bf16 drift).
- L1 token-boundary fork: K=2/4/8 independent branch caches, no cross-branch contamination.
- L2 batched branches == independent == full recompute.
- L3 prune/reorder survivors (subset, order changes, 8→4→2 / 8→3 / 4→1) with aligned lineage.
- L4 current-token layer perturbation (L24/36/47, loop-targeted) carries via branch cache.
- L5 prompt-internal perturbation yields a valid branch-specific cache; negative control
  (unperturbed cache vs perturbed recompute, RMS ≈ 3.0) confirms the branch cache is required.
- L6 partial-cache splice: shared/affected slot boundary logic validated (Option A reproduces
  the full branch cache bit-exactly), but **diagnostic only — no compute savings claimed**.

**Status: `AUTOREGRESSIVE_KV_BRANCH_CARRY_STATUS = PROMPT_INTERNAL_BRANCH_CACHE_VALID`;
`LEVEL6_PARTIAL_SPLICE_STATUS = PARTIAL_SPLICE_DIAGNOSTIC_ONLY`.** Equivalence standard:
cached logits match full recompute of the identical sequence within bf16, with exact top-1/token
agreement except at model-intrinsic argmax near-ties. No steering, no training, no compute-savings
claim. Details: `docs/evaluator/kv-cache-branch-carry.md`; artifacts under
`artifacts/reports/probes/bg_autoregressive_kv_branch_carry_v1_2026-06-01/`.


## Partial cache splice v2 (2026-06-01)

`partial_cache_splice_v2` turned the v1 Level 6 diagnostic (copy-affected slots, no
saving) into a real compute-saving suffix-recompute splice. Key obstacle: the
UniversalTransformerCache stores K/V but not the inter-layer residual stream, so the
residual hidden at the perturbation boundary (loop u, layer L output) is captured
during a minimal shared-prefix prefill; an additive boundary perturbation is then
applied without a forward, and ONLY the suffix (loop u layers L+1.., loops u+1..) is
recomputed. The spliced branch cache is **bit-exact** vs a full perturbed-prompt
reference (all 192 slots; prefill logits RMS 0; continuation bit-for-bit).

Empirical hook timing confirmed: perturbing a layer output leaves the boundary slot
unaffected; first affected slot = (u, L+1); changed set == downstream_only theory.
Validated single-branch, multi-branch (K=2/4, independent, no contamination),
batched+prune/reorder, and left-padded (explicit position_ids). Measured compute
savings (baseline K full prefills vs prefix + K suffixes): per-branch layer-pass
saving 13%/38%/63%/88% for boundary loops 0/1/2/3 (layer 24); at loop 2 layer 24,
K-scaling gives 32%/47%/55% fewer passes for K=2/4/8 (wall-clock tracks). Saving is
amortized (needs K≥2; K=1 does prefix+suffix == full). Copy-affected (Mode A) saves
nothing.

**Status: `PARTIAL_CACHE_SPLICE_V2_STATUS = PARTIAL_SPLICE_COMPUTE_SAVING_VALID`**
(upgrades v1's `LEVEL6_PARTIAL_SPLICE_STATUS = PARTIAL_SPLICE_DIAGNOSTIC_ONLY`).
Compute-saving branch-carry CAN now be claimed (amortized, equivalence-validated);
production readiness CANNOT. No steering, no training. GPU guard confirmed the MMLU
science-repair process was not active and its artifacts were untouched. Details:
`docs/evaluator/kv-cache-branch-carry.md`; artifacts under
`artifacts/reports/probes/bg_partial_cache_splice_v2_2026-06-01/`.
