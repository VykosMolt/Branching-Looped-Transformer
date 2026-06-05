# Branch Generation And Survival

Updated: 2026-05-31

This document collapses the hidden-origin branch-generation, universal/gated selector, fixed-composite survival, selection-only prototype, and DualAnchor survival line.

## Current Conclusion

Branch generation is useful as search expansion. The strongest current survival mechanism is the architecture-looped DualAnchor branch/prune loop:

`ARCHITECTURE_LOOPED_SURVIVAL_READY_TERMINAL_DEFER_REQUIRED`

Before DualAnchor, the strongest cached selection-only survival policy was:

`FIXED_COMPOSITE_BRANCH_SURVIVAL_POLICY_STATUS = SURVIVAL_READY`

The fixed-composite policy remains an important reference point, but it is no longer the leading architecture-shaped baseline.

## Historical Milestones

| Stage | Status | Interpretation |
| --- | --- | --- |
| hidden-origin taps / v2 / v3 | weak or data-limited | Hidden-origin branch signal existed, but selector data was weak. |
| split salvage / quota v4 | still data-limited | Heldout support improved, but branch selector readiness was not established. |
| Branch Generator v1 | `WEAK_BUT_USABLE` | Heldout diversity improved; train/val remained weak. |
| universal branch-content taps v1 | `FUSION_NEEDED` | Old-context behavior survived, but bridge/materialization failed. |
| gated/composite selector v1 | `OLD_NEW_COMPOSITE_SUFFICIENT` | Composite old+new/bridge signals beat a naive universal replacement. |
| fixed-composite survival policy v1 | `SURVIVAL_READY` | Top4 survival cleared readiness: retention high, false prune low. |
| selection-only prototype v1 | `SURVIVAL_READY_FINAL_ARBITER_WEAK` | Top4 contained good branches; final choice among survivors was weak. |
| DualAnchor architecture-looped v3 | `READY_WITH_TERMINAL_DEFER` | Repeated all-loop branch/prune survival scaled; final collapse remains gated/deferred. |

## Fixed-Composite Reference

Selected operating point:

`fixed_composite_conservative_top4`

Heldout:

- oracle retention: `0.931`
- false prune: `0.069`
- avg survivors: `3.873`

Old-context/coding preservation:

- old-context/coding retention: `1.000`
- coding retention: `1.000`
- coding false prune: `0.000`

Interpretation:

Top4 was the first safe operating point. Top1/top2/top3 were too aggressive for primary branch survival.

## Selection-Only Prototype Reference

Selection-only Phase 2 prototype v1:

- fixed top4 oracle retention: `0.9514`
- false prune: `0.0486`
- avg survivors: `3.9109`
- task macro best-selected reward: `0.9453`
- task macro final reward: `0.6672`
- status: `SURVIVAL_READY_FINAL_ARBITER_WEAK`

The bottleneck was final selection, not survival.

## DualAnchor Architecture-Looped Survival

V3 all-loop run:

- tasks: `48`
- nonterminal stage decisions: `528`
- stage oracle retention: `0.9848`
- terminal oracle retained: `1.0000`
- false prunes: `8`
- false-prune recovery: `8/8`
- budget: `8`

Interpretation:

Survival is strong under the architecture-shaped loop. Local false-prunes can recover because later perturbation stages remain active.

## Perturbation And Lineage

The earlier normalized perturbation audit corrected the naive "perturbed wins 80%" claim:

- perturbed candidate fraction in that slice: `0.875`
- observed forced-top1 perturbed rate: `0.7986`
- lift over count-share random: `-0.0764`

The stronger result was oracle expansion:

- max perturbed reward > clean reward: `46.8%`
- max perturbed reward = clean reward: `53.2%`
- max perturbed reward < clean reward: `0.0%`

V3 lineage:

- child improved parent: `0.0390`
- child tied parent: `0.9281`
- child worsened parent: `0.0329`
- mean child-parent delta: `+0.00464`

Interpretation:

Perturbation is useful as local search expansion. It mostly preserves parent quality and occasionally improves it.

## L47

V3 replay ablation says full L47 is required:

- full L47 retained oracle: `1.0000`
- nonterminal L47 disabled retained oracle: `0.0417`
- terminal-only L47 retained oracle: `0.0417`

This is a replay ablation over generated lineages, not a regenerated dynamics proof. Still, earlier-loop L47 should stay active until a regenerated ablation contradicts it.

## Boundaries

No action steering has been tested in this branch-generation/survival line.

Prompt-only decoder-layer carry was validated for cumulative-hook equivalence, but this does not prove autoregressive branch-specific KV/cache fork-carry or compute savings.

## Source Run Notes

- `history/bg-run-notes/branch-generation/bg_hidden_state_branch_generation.md`
- `history/bg-run-notes/branch-generation/bg_hidden_origin_taps.md`
- `history/bg-run-notes/branch-generation/bg_hidden_origin_split_salvage.md`
- `history/bg-run-notes/branch-generation/bg_hidden_origin_diversity_v2.md`
- `history/bg-run-notes/branch-generation/bg_hidden_origin_diversity_v3.md`
- `history/bg-run-notes/branch-generation/bg_hidden_origin_quota_v4.md`
- `history/bg-run-notes/branch-generation/bg_hidden_origin_branch_generator_v1.md`
- `history/bg-run-notes/survival-selection/bg_universal_branch_content_taps_v1.md`
- `history/bg-run-notes/survival-selection/bg_gated_branch_content_selector_v1.md`
- `history/bg-run-notes/survival-selection/bg_fixed_composite_branch_survival_policy_v1.md`
- `history/bg-run-notes/survival-selection/bg_selection_only_phase2_prototype_v1.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_dualanchor_architecture_looped_stratified_probe_v3.md`



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
