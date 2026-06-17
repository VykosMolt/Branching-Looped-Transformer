# Evaluator Docs

Updated: 2026-06-04

This directory has three layers:

1. Current entry-point docs with memorable names.
2. Source run notes sorted by topic under `history/bg-run-notes/`.
3. Exact archives under `history/`.

Scope note: these docs preserve research provenance from a larger local worktree.
The public repository's active code surface is the top-level `evaluator/`,
`evaluator_core/`, and `probes/` folders. Historical references to older paths,
external wrappers, source-worktree routing, action-loop systems, or other
non-public components are provenance notes only; they are not dependencies or
current claims of this repository.

No source data was intentionally removed during the 2026-05-31 cleanup. A preservation snapshot of the pre-cleanup root Markdown files is kept at:

`history/pre_docs_consolidation_2026-05-31/`

Checksum manifest:

`history/pre_docs_consolidation_2026-05-31/manifest.sha256`

The older 2026-05-18 raw archive is now under:

`history/raw_archive_2026-05-18/`

## Start Here

**Navigate**

- `evaluator-navigation-map.md` — the compact "what do I read for X" map. Start here.
- `current-state.md` — the active current-state ledger (verdicts, status).
- `chronological-evaluator-summary.md` — the long narrative ("DSA") summary, end to end.

**Active baselines and decisions**

- `dualanchor-architecture-baseline.md` — the active Phase 2a branch-selection baseline.
- `science-reasoning-repair.md` — pre-steering domain decision (reasoning headline-ready, science diagnostic).
- `bg_core_domain_tap_audit_dualanchor_readiness_v1.md` — core-domain (coding/reasoning/math/logic/alignment) tap audit; DualAnchor locked unchanged, core domains ready for Phase 2b, science diagnostic-only.
- `domain-transfer-ledger.md` — domain-transfer results ledger.

**Mechanisms and foundations**

- `evaluator-locus-summary.md` — pairwise-locus / readout-geometry summary (v2-v10 foundation through the current baseline).
- `flip-test-interpretation.md` — what the 95.2% pairwise accuracy (vs ~65% pointwise) actually means.
- `dualanchor-tap-evolution.md` — how the old two taps became DualAnchor.
- `branch-generation-and-survival.md` — hidden-origin, fixed-composite, and architecture-looped survival.
- `terminal-selection-and-arbiters.md` — final-arbiter and terminal-confidence work.
- `kv-cache-branch-carry.md` — generation-time KV/cache branch-carry (v1 ladder) + compute-saving suffix-recompute splice (v2).
- `s1-branch-carry-reference-loop.md` — S1: full inject->carry->prune->loop-back->terminal reference loop (5-gate validation), and the frozen-model capability result (reachability-neutral / locally closed; two walls -> training is the next lever).

**Historical consolidations**

- `phase1-controller-and-routing.md` — Phase 1 routing/controller docs.
- `steering-and-adapters.md` — steering and adapter boundary docs.
- `interfaces-and-tools.md` — tap interface and tooling docs.

## Current Bottom Line

The active Phase 2a baseline candidate is:

`DUALANCHOR_ARCHITECTURE_LOOPED_V3_STATUS = ARCHITECTURE_LOOPED_SURVIVAL_READY_TERMINAL_DEFER_REQUIRED`

`BG_DUALANCHOR_PHASE2A_READINESS_V3_VERDICT = READY_WITH_TERMINAL_DEFER`

Latest pre-steering domain decision (v3, 2026-06-04):

`MMLU_SCIENCE_BRANCH_PARSER_REPAIR_V3_STATUS = SCIENCE_PARTIALLY_REPAIRED`
`BG_PRE_STEERING_DOMAIN_DECISION_V3_VERDICT = READY_FOR_STEERING_REASONING_PLUS_PARTIAL_SCIENCE`

Reasoning is the headline domain under confidence-gated top1 else survivor-set handoff.
Science is now partially repaired: MMLU anatomy is a candidate partial/secondary headline;
chemistry, physics, and SciQ remain excluded/diagnostic. See `science-reasoning-repair.md`.
(Supersedes v2 `REASONING_READY_SCIENCE_DIAGNOSTIC` on the science half.)

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

No steering was tested for the Phase 2a baseline, and no production routing changed.

Separately, the generation-time cache substrate is now validated (see
`kv-cache-branch-carry.md`):

`AUTOREGRESSIVE_KV_BRANCH_CARRY_STATUS = PROMPT_INTERNAL_BRANCH_CACHE_VALID` and
`PARTIAL_CACHE_SPLICE_V2_STATUS = PARTIAL_SPLICE_COMPUTE_SAVING_VALID`. Autoregressive
branch-specific KV/cache carry is validated, and amortized compute-saving branch-carry
(K≥2 branches via suffix recompute) is the one place a compute-savings claim is made.
This is a mechanical cache result, not steering or a production routing change.

## Source Run Notes

Run-specific source docs were moved under topic folders in `history/bg-run-notes/`. They are useful for exact provenance, but they are not the main reading path.

DualAnchor/two-tap sources:

- `history/bg-run-notes/dualanchor-two-tap/bg_old_anchored_branch_valid_taps_v1.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_two_tap_branch_selector_v1.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_two_tap_full_readiness_v1.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_two_tap_fresh_dataset_comparison_v1.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_two_tap_hh_rlhf_comparison_v1.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_layer_native_two_tap_readiness_v1.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_layer_native_two_tap_constrained_train_v1.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_dualanchor_architecture_looped_stratified_probe_v3.md`

Science/reasoning repair sources (owned by `science-reasoning-repair.md`):

- `history/bg-run-notes/science-reasoning/bg_dualanchor_science_reasoning_repair_v2.md`
- `history/bg-run-notes/science-reasoning/bg_dualanchor_convergence_hairs_reasoning_science_v1.md`
- `history/bg-run-notes/science-reasoning/bg_dualanchor_science_branch_recipe_reasoning_defer_v1.md`

KV/cache branch-carry sources (owned by `kv-cache-branch-carry.md`):

- `history/bg-run-notes/kv-cache/bg_autoregressive_kv_branch_carry_v1.md`
- `history/bg-run-notes/kv-cache/autoregressive_kv_branch_carry_validation_note.md`
- `history/bg-run-notes/kv-cache/bg_partial_cache_splice_v2.md`

Branch generation/survival sources:

- `history/bg-run-notes/branch-generation/bg_hidden_origin_branch_generator_v1.md`
- `history/bg-run-notes/survival-selection/bg_universal_branch_content_taps_v1.md`
- `history/bg-run-notes/survival-selection/bg_gated_branch_content_selector_v1.md`
- `history/bg-run-notes/survival-selection/bg_fixed_composite_branch_survival_policy_v1.md`
- `history/bg-run-notes/survival-selection/bg_selection_only_phase2_prototype_v1.md`

Terminal/final-arbiter sources:

- `history/bg-run-notes/terminal-arbiters/bg_final_arbiter_top4_survivors_v1.md`
- `history/bg-run-notes/terminal-arbiters/bg_final_arbiter_top4_survivors_v1_1.md`
- `history/bg-run-notes/terminal-arbiters/bg_merged_tap_final_arbiter_integration_v1_1.md`

Older Phase 1/steering sources:

- `history/phase1-controller-and-routing/post_v10_synthesis_2026-05-18_v8.1_routing_locked.md`
- `history/bg-run-notes/steering/bg_steering_consolidation_2026-05-18.md`
- `history/steering-and-adapters/steering-consolidation.md`
- `history/steering-and-adapters/sequence-level-adapter.md`
- `history/interfaces-and-tools/transformer-integration.md`
- `history/interfaces-and-tools/tap-interface.md`

## Naming Policy

Human-facing docs should use memorable names. New exact run notes may keep `history/bg-run-notes/bg_*` names, but they should point back to the memorable entry point that owns the interpretation.

When old underscore/hyphen duplicate names exist, the hyphenated root names are canonical for human-facing docs:

- `current-state.md`
- `domain-transfer-ledger.md`
