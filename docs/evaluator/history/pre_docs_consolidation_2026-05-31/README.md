# Evaluator Docs

Updated: 2026-05-31

This directory has three layers:

1. Current entry-point docs with memorable names.
2. Source run notes, mostly `history/bg-run-notes/bg_*` files.
3. Exact archives under `history/`.

No source data was intentionally removed during the 2026-05-31 cleanup. Exact pre-consolidation root Markdown files were copied to:

`history/pre_docs_consolidation_2026-05-31/`

Checksum manifest:

`history/pre_docs_consolidation_2026-05-31/manifest.sha256`

The older 2026-05-18 raw archive is now under:

`history/raw_archive_2026-05-18/`

## Start Here

- `evaluator-navigation-map.md` is the compact navigation map.
- `current-state.md` is the active current-state ledger.
- `dualanchor-architecture-baseline.md` is the active Phase 2a branch-selection baseline.
- `dualanchor-tap-evolution.md` explains how the old two taps became DualAnchor.
- `branch-generation-and-survival.md` consolidates hidden-origin, fixed-composite, and architecture-looped survival.
- `terminal-selection-and-arbiters.md` consolidates final-arbiter and terminal confidence work.
- `domain-transfer-ledger.md` tracks domain-transfer results.
- `phase1-controller-and-routing.md` consolidates Phase 1 routing/controller docs.
- `steering-and-adapters.md` consolidates steering and adapter boundary docs.
- `interfaces-and-tools.md` consolidates tap interface and tooling docs.
- `chronological-evaluator-summary.md` is the long narrative summary.

## Current Bottom Line

The active Phase 2a baseline candidate is:

`DUALANCHOR_ARCHITECTURE_LOOPED_V3_STATUS = ARCHITECTURE_LOOPED_SURVIVAL_READY_TERMINAL_DEFER_REQUIRED`

`BG_DUALANCHOR_PHASE2A_READINESS_V3_VERDICT = READY_WITH_TERMINAL_DEFER`

Locked candidate:

- selector: DualAnchor `MIX_CODE_REASONING` + `MIX_OBJECTIVE_ALL`
- schedule: `L1_24 -> L1_36 -> L1_47 -> L2_24 -> L2_36 -> L2_47 -> L3_24 -> L3_36 -> L3_47 -> L4_24 -> L4_36 -> terminal L4_47`
- threshold: `mean_floor_very_loose`
- budget: `8`
- L47: active in nonterminal loops
- terminal: confidence-gated top1; otherwise defer/keep terminal survivors

V3 metrics:

| Metric | Value |
| --- | ---: |
| tasks | 48 |
| domains | 24 reasoning / 24 science |
| stage oracle retention | 0.9848 |
| terminal oracle retained | 1.0000 |
| forced terminal top1 oracle | 0.9167 |
| reward-diverse forced top1 oracle | 0.6364 |
| false-prune recovery | 8/8 |

Interpretation:

Survival is strong. Terminal forced top1 is not the primary policy because reward-diverse hard slices remain weak. The locked baseline is terminal-defer-ready, not unconditional-terminal-ready.

No steering was tested. No production routing changed. No autoregressive branch-specific KV/cache fork-carry or compute-savings claim is made.

## Source Run Notes

Run-specific source docs were moved under `history/bg-run-notes/`. They are useful for exact provenance, but they are not the main reading path.

DualAnchor/two-tap sources:

- `history/bg-run-notes/bg_old_anchored_branch_valid_taps_v1.md`
- `history/bg-run-notes/bg_two_tap_branch_selector_v1.md`
- `history/bg-run-notes/bg_two_tap_full_readiness_v1.md`
- `history/bg-run-notes/bg_two_tap_fresh_dataset_comparison_v1.md`
- `history/bg-run-notes/bg_two_tap_hh_rlhf_comparison_v1.md`
- `history/bg-run-notes/bg_layer_native_two_tap_readiness_v1.md`
- `history/bg-run-notes/bg_layer_native_two_tap_constrained_train_v1.md`
- `history/bg-run-notes/bg_dualanchor_architecture_looped_stratified_probe_v3.md`

Branch generation/survival sources:

- `history/bg-run-notes/bg_hidden_origin_branch_generator_v1.md`
- `history/bg-run-notes/bg_universal_branch_content_taps_v1.md`
- `history/bg-run-notes/bg_gated_branch_content_selector_v1.md`
- `history/bg-run-notes/bg_fixed_composite_branch_survival_policy_v1.md`
- `history/bg-run-notes/bg_selection_only_phase2_prototype_v1.md`

Terminal/final-arbiter sources:

- `history/bg-run-notes/bg_final_arbiter_top4_survivors_v1.md`
- `history/bg-run-notes/bg_final_arbiter_top4_survivors_v1_1.md`
- `history/bg-run-notes/bg_merged_tap_final_arbiter_integration_v1_1.md`

Older Phase 1/steering sources:

- `history/phase1-controller-and-routing/post_v10_synthesis_2026-05-18_v8.1_routing_locked.md`
- `history/bg-run-notes/bg_steering_consolidation_2026-05-18.md`
- `history/steering-and-adapters/steering-consolidation.md`
- `history/steering-and-adapters/sequence-level-adapter.md`
- `history/interfaces-and-tools/transformer-integration.md`
- `history/interfaces-and-tools/tap-interface.md`

## Naming Policy

Human-facing docs should use memorable names. New exact run notes may keep `history/bg-run-notes/bg_*` names, but they should point back to the memorable entry point that owns the interpretation.

When old underscore/hyphen duplicate names exist, the hyphenated root names are canonical for human-facing docs:

- `current-state.md`
- `domain-transfer-ledger.md`
