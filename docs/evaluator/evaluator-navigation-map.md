# Evaluator Navigation Map

Updated: 2026-06-04

This is the current navigation layer for evaluator, branch-generation, tap, and DualAnchor work. Detailed run notes live under `history/`; new readers should start from the canonical docs below.

No data was intentionally removed during this consolidation. Exact pre-consolidation root Markdown files were copied to:

`docs/evaluator/history/pre_docs_consolidation_2026-05-31/`

The checksum manifest is:

`docs/evaluator/history/pre_docs_consolidation_2026-05-31/manifest.sha256`

## Current Entry Points

| Need | Read |
| --- | --- |
| Current state and active verdicts | `current-state.md` |
| Pairwise locus / readout-geometry summary | `evaluator-locus-summary.md` |
| Short navigation map | `evaluator-navigation-map.md` |
| DualAnchor architecture-looped baseline | `dualanchor-architecture-baseline.md` |
| DualAnchor/two-tap evolution | `dualanchor-tap-evolution.md` |
| Branch generation, survival, and hidden-origin history | `branch-generation-and-survival.md` |
| Terminal selection and final-arbiter history | `terminal-selection-and-arbiters.md` |
| Domain-transfer ledger | `domain-transfer-ledger.md` |
| Science/reasoning pre-steering domain decision | `science-reasoning-repair.md` |
| Core-domain tap audit + DualAnchor readiness (Phase 2b) | `core-domain-tap-audit.md` |
| Content / final-selection taps (CoreContent) | `content-selection-taps.md` |
| CoreContent dataset expansion + refit v2 (detail) | `corecontent-dataset-expansion-v2.md` |
| Branch training + logic expansion (toward model-internal branching) | `branch-training-logic-expansion.md` |
| Generation-time KV/cache branch-carry + compute-saving splice | `kv-cache-branch-carry.md` |
| What the 95.2% pairwise accuracy means (flip test) | `flip-test-interpretation.md` |
| Long chronological story | `chronological-evaluator-summary.md` |
| Phase 1 controller/routing | `phase1-controller-and-routing.md` |
| Steering/adapters boundary | `steering-and-adapters.md` |
| Tap interfaces and tools | `interfaces-and-tools.md` |

## Current Bottom Line

The active Phase 2a baseline candidate is now:

- DualAnchor selector: `MIX_CODE_REASONING` + `MIX_OBJECTIVE_ALL`.
- Architecture loop: `L1_24 -> L1_36 -> L1_47 -> L2_24 -> L2_36 -> L2_47 -> L3_24 -> L3_36 -> L3_47 -> L4_24 -> L4_36 -> terminal L4_47`.
- Threshold: `mean_floor_very_loose`.
- Budget: `8`.
- L47: active in nonterminal loops.
- Terminal policy: confidence-gated top1 only; otherwise defer or keep terminal survivors.
- Status: `ARCHITECTURE_LOOPED_SURVIVAL_READY_TERMINAL_DEFER_REQUIRED`.
- Readiness: `READY_WITH_TERMINAL_DEFER`.

The Phase 2a baseline itself is still selection/branching only: no steering, no production routing change.

Latest pre-steering domain decision (v3, 2026-06-04):

- Status: `MMLU_SCIENCE_BRANCH_PARSER_REPAIR_V3_STATUS = SCIENCE_PARTIALLY_REPAIRED`.
- Decision: `BG_PRE_STEERING_DOMAIN_DECISION_V3_VERDICT = READY_FOR_STEERING_REASONING_PLUS_PARTIAL_SCIENCE`.
- Reasoning: headline-ready under confidence-gated top1 else top5/full survivor-set handoff (`REASONING_BASELINE_STILL_READY`).
- Science: partially repaired — MMLU anatomy is a candidate partial/secondary headline (`ANATOMY_REPAIRED`, heldout positive-oracle 0.333 / terminal-best 0.20 on 3 tasks); chemistry, physics, and SciQ remain excluded/diagnostic (chem/physics parse collapsed to 0.0). Heldout overall `SCIENCE_RECIPE_IMPROVED_BUT_WEAK`.
- Supersedes v2 `REASONING_READY_SCIENCE_DIAGNOSTIC` on the science half.
- Report: `science-reasoning-repair.md`.

Generation-time cache substrate (separate mechanical track, not steering/routing):

- `AUTOREGRESSIVE_KV_BRANCH_CARRY_STATUS = PROMPT_INTERNAL_BRANCH_CACHE_VALID` (v1, ladder L0–L5).
- `PARTIAL_CACHE_SPLICE_V2_STATUS = PARTIAL_SPLICE_COMPUTE_SAVING_VALID` (v2): amortized
  compute-saving branch-carry via suffix recompute is now validated (needs K≥2 branches).
  This is the only place a compute-saving claim is made; it is not a production routing change.
- Report: `kv-cache-branch-carry.md`.

## Canonical Source Groups

### DualAnchor / Two-Tap

Use `dualanchor-tap-evolution.md` for the consolidated interpretation.

Source run notes moved to `history/bg-run-notes/dualanchor-two-tap/`:

- `history/bg-run-notes/dualanchor-two-tap/bg_old_anchored_branch_valid_taps_v1.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_two_tap_branch_selector_v1.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_two_tap_full_readiness_v1.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_two_tap_fresh_dataset_comparison_v1.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_two_tap_hh_rlhf_comparison_v1.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_two_tap_gap_targeted_v2.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_layer_native_two_tap_readiness_v1.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_layer_native_two_tap_constrained_train_v1.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_layer_native_two_tap_targeted_rehost_diagnostic_v1.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_dualanchor_branch_gap_repair_v1.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_dualanchor_pre_repair_fixed_bundle_audit_v1.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_dualanchor_hard_anchor_selector_v1.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_dualanchor_architecture_looped_stratified_probe_v3.md`

(Science/reasoning repair has its own group below.)

### Branch Generation And Survival

Use `branch-generation-and-survival.md` for the consolidated interpretation.

Source run notes under `history/bg-run-notes/branch-generation/` and `history/bg-run-notes/survival-selection/`:

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

### Terminal Selection / Arbiters

Use `terminal-selection-and-arbiters.md` for the consolidated interpretation.

Source run notes:

- `history/bg-run-notes/terminal-arbiters/bg_final_arbiter_top4_survivors_v1.md`
- `history/bg-run-notes/terminal-arbiters/bg_final_arbiter_top4_survivors_v1_1.md`
- `history/bg-run-notes/weight-merge/bg_merged_weight_branch_content_taps_v1.md`
- `history/bg-run-notes/terminal-arbiters/bg_merged_tap_final_arbiter_integration_v1.md`
- `history/bg-run-notes/terminal-arbiters/bg_merged_tap_final_arbiter_integration_v1_1.md`
- `history/bg-run-notes/dualanchor-two-tap/bg_dualanchor_architecture_looped_stratified_probe_v3.md`

### Steering And Prior Phase 1 Work

Use `phase1-controller-and-routing.md`, `steering-and-adapters.md`, and `interfaces-and-tools.md` for the consolidated historical baseline and boundary docs. Source notes were moved under `history/`.

- `history/phase1-controller-and-routing/post_v10_synthesis_2026-05-18_v8.1_routing_locked.md`
- `history/bg-run-notes/steering/bg_steering_consolidation_2026-05-18.md`
- `history/steering-and-adapters/steering-consolidation.md`
- `history/steering-and-adapters/sequence-level-adapter.md`
- `history/interfaces-and-tools/transformer-integration.md`
- `history/interfaces-and-tools/tap-interface.md`

### Science / Reasoning Repair

Use `science-reasoning-repair.md` for the consolidated pre-steering domain decision.

- `history/bg-run-notes/science-reasoning/bg_dualanchor_science_reasoning_repair_v2.md`
- `history/bg-run-notes/science-reasoning/bg_dualanchor_convergence_hairs_reasoning_science_v1.md`
- `history/bg-run-notes/science-reasoning/bg_dualanchor_science_branch_recipe_reasoning_defer_v1.md`

### Generation-Time KV/Cache Branch-Carry

Use `kv-cache-branch-carry.md` for the consolidated cache branch-carry + compute-saving result.

- `history/bg-run-notes/kv-cache/bg_autoregressive_kv_branch_carry_v1.md`
- `history/bg-run-notes/kv-cache/autoregressive_kv_branch_carry_validation_note.md`
- `history/bg-run-notes/kv-cache/bg_partial_cache_splice_v2.md`

## Duplicate Name Policy

Some old docs existed in both underscore and hyphen variants. The hyphenated names are now canonical in the root:

- `current-state.md`
- `domain-transfer-ledger.md`

The underscore duplicates were moved to `history/compatibility-duplicates/`. `history/bg-run-notes/bg_*` run notes and older hyphen-only topic summaries were moved under `history/`.

## Branch training + logic expansion + terminal v1 (2026-06-06)

- Status: `LOGIC_EXPANSION_READY_TRAINING_NOT_READY`; policy decision: `KEEP_EXTERNAL_DUALANCHOR_CORECONTENT_BASELINE`.
- Goal: move from external branch selection (DualAnchor + CoreContent_v2) toward **model-internal branching**. Primary deliverable = the branch-training **data + evaluation harness** (the L/O training is a bounded 300-step LoRA proof-of-capability on Ouro-RLTT, not a converged model). External verifiers are the only ground truth; DualAnchor/CoreContent are teachers/baselines; science diagnostic-only; steering not run.
- **Logic expanded** to ~33247 train groups across 10 verifier-backed families (synthetic propositional/proof-depth/syllogism/z3-constraint + LogiQA/ReClor/RuleTaker/FOLIO/LSAT/logical-entailment).
- **Branch-pool reachability@4** (Ouro-RLTT generations, external-labeled): {'coding': 0.4333, 'logic': 0.7333, 'math': 0.8333, 'reasoning': 0.95}. Math went 0.31→0.83 once given a brutal tool-free answer-forcing prompt + 1400 math budget + early-stop + LaTeX/sympy verifier; coding ~0.43 after fixing a function-name prompt gap (genuine, not artifact).
- **Experiment 1 (integrated terminal, H)**: within real DualAnchor top-5 survivor sets, CoreContent_v2 0.6584 > DualAnchor forced-top1 0.3787 (retention 1.0) → **selection, not survival, is the terminal bottleneck**; the composed external baseline (DualAnchor survival → CoreContent_v2 ranking → survivor handoff) is valid.
- **DualAnchor-as-teacher (J)**: useful branch-policy teacher (oracle-retention lift over random +0.11 coding/reasoning) but ~random on logic (+0.03) → logic branch quality must come from verifier reward, not teacher distillation.
- **Trained vs Ouro-RLTT-no-adapter (O)**: macro positive_oracle@K {'base': 0.708, 'sft': 0.688}, lift -0.02, diversity lift 0.45 (`MODEL_INTERNAL_BRANCHING_PARTIAL`).
- Data-hygiene fixes caught by hand-audit: math-LaTeX verifier false-negatives, coding function-name mismatch, and a hendrycks_math `task_uid` collision (row indices repeat across 7 subjects → re-keyed by prompt-hash).
- No Ouro base/tokenizer/checkpoint/registry overwrite; adapters saved only under `artifacts/models/branch_training_logic_expansion_v1/`; pure/transplanted/CoreContent_v2 artifacts untouched.
- Artifacts: `artifacts/reports/probes/branch_training_logic_expansion_terminal_v1_2026-06-06`; data `data/branch_training_logic_expansion_v1/`.
