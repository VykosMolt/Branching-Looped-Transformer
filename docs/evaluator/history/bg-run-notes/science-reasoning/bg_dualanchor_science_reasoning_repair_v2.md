# DualAnchor Science And Reasoning Repair V2

Date: 2026-06-01
Output root: `artifacts/reports/probes/bg_dualanchor_science_reasoning_repair_v2_2026-06-01/`

## Status

`DUALANCHOR_SCIENCE_REASONING_REPAIR_V2_STATUS = REASONING_READY_SCIENCE_DIAGNOSTIC`

`BG_PRE_STEERING_READINESS_V2_VERDICT = READY_FOR_STEERING_WITH_SCIENCE_DIAGNOSTIC`

Reasoning is ready as the headline steering-comparison domain only with the locked
terminal handoff policy. Science remains diagnostic/excluded from headline steering:
the regenerated branch-recipe calibration found some recipe signal, but heldout did not
validate a science repair.

No steering, Ouro training, tokenizer/checkpoint edit, tap-registry update, wrapper or
local-agent execution, Hunter-Seeker execution, ARC action loop, production routing
change, hard convergence-hair merge, compute-savings claim, or autoregressive fork/carry
claim was made.

## Component Verdicts

| Component | Verdict |
| --- | --- |
| Inventory | `BG_SCIENCE_REASONING_REPAIR_V2_INVENTORY_VERDICT = PARTIAL` |
| Task suite | `BG_SCIENCE_REASONING_REPAIR_V2_TASK_SUITE_VERDICT = SCIENCE_LIMITED` |
| Parser candidates | `BG_SCIENCE_PARSER_PATCH_CANDIDATES_VERDICT = READY` |
| Parser validation | `BG_SCIENCE_PARSER_VALIDATION_VERDICT = PARSER_PATCH_DIAGNOSTIC_ONLY` |
| Science recipe plan | `BG_SCIENCE_BRANCH_RECIPE_V2_PLAN_VERDICT = SOURCE_SPECIFIC_READY` |
| Science recipe calibration | `BG_SCIENCE_RECIPE_V2_CALIBRATION_VERDICT = SCIENCE_RECIPE_FOUND` |
| Science heldout | `BG_SCIENCE_RECIPE_V2_HELDOUT_VERDICT = SCIENCE_BRANCH_GENERATION_STILL_WEAK` |
| Source-specific science | `BG_SCIENCE_V2_SOURCE_SPECIFIC_VERDICT = MMLU_CHEM_ANATOMY_BLOCKED` |
| Parser recommendation | `BG_SCIENCE_PARSER_PATCH_RECOMMENDATION_VERDICT = ROBUST_DIAGNOSTIC_ONLY` |
| Reasoning terminal handoff | `BG_REASONING_TERMINAL_HANDOFF_V2_VERDICT = REASONING_HANDOFF_LOCKED` |
| Reasoning hard slice | `BG_REASONING_HARD_SLICE_V2_VERDICT = REASONING_READY_WITH_HANDOFF` |
| Science L47/layer ablation | `BG_SCIENCE_L47_LAYER_ABLATION_V2_VERDICT = L2_47_HELPS` |
| Perturbation escalation | `BG_SCIENCE_PERTURBATION_ESCALATION_V2_VERDICT = ESCALATION_NO_HELP` |
| Soft hairs | `BG_SOFT_HAIRS_V2_VERDICT = SCIENCE_NO_GOOD_WARNING_USEFUL` |
| Integrated repair | `BG_SCIENCE_REASONING_INTEGRATED_REPAIR_V2_VERDICT = SCIENCE_STILL_BLOCKED` |

## Science Result

The task suite remained source-limited: `31` science tasks were selected, with only `7`
science heldout tasks. OpenBookQA and biology were missing from the requested science
source set.

Calibration found source-specific recipe signal:

- `L2_47_emphasis`: positive-oracle `0.2500`, terminal best reward `0.2500` on `4` tasks.
- `L47_heavy`: positive-oracle `0.2500`, terminal best reward `0.2500` on `4` tasks.
- `baseline_v3_regenerated`: positive-oracle `0.1667`, terminal best reward `0.1667` on `12` tasks.

Heldout did not validate the repair:

- heldout recipe: `baseline_v3_regenerated`
- heldout task count: `7`
- selected-parser positive-oracle rate: `0.0000`
- selected-parser terminal best reward: `-0.0571`
- reward-diverse rate: `0.5714`
- terminal oracle retained: `1.0000`

Source diagnosis:

| Source | Heldout/task count | Best recipe | Readiness |
| --- | ---: | --- | --- |
| `mmlu_anatomy` | `3` | `baseline_v3_regenerated` | `BRANCH_GENERATION_BLOCKED` |
| `mmlu_high_school_chemistry` | `2` | `baseline_v3_regenerated` | `BRANCH_GENERATION_BLOCKED` |
| `mmlu_high_school_physics` | `1` | `baseline_v3_regenerated` | `DATA_LIMITED` |
| `sciq` | `1` | `baseline_v3_regenerated` | `DATA_LIMITED` |

The parser patch remains diagnostic-only. The selected candidate did not clear the
calibration gate, and robust parsing must not replace strict reward without stronger
false-positive validation.

## Reasoning Result

Reasoning terminal handoff is locked:

| Policy | Tasks | Oracle retained | First selected oracle | Defer rate |
| --- | ---: | ---: | ---: | ---: |
| `dualanchor_confidence_gated` | `24` | `1.0000` | `0.8750` | `0.9583` |
| `dualanchor_terminal_top5` | `24` | `1.0000` | `0.8750` | `0.9583` |
| `terminal_defer_all` | `24` | `1.0000` | `0.8750` | `0.9583` |
| `dualanchor_terminal_top2` | `24` | `0.9583` | `0.8750` | `0.9583` |
| `dualanchor_forced_top1` | `24` | `0.8750` | `0.8750` | `0.9583` |

The locked terminal policy remains confidence-gated top1 only when confidence is strong;
otherwise top5/full survivor-set handoff. Reasoning is headline-ready under that handoff,
not under unconditional terminal top1.

## Locked Baseline

Use the existing DualAnchor architecture-looped baseline if proceeding to steering
comparison:

- schedule: `L1_24 -> L1_36 -> L1_47 -> L2_24 -> L2_36 -> L2_47 -> L3_24 -> L3_36 -> L3_47 -> L4_24 -> L4_36 -> terminal L4_47`
- selector: `MIX_CODE_REASONING + MIX_OBJECTIVE_ALL`
- threshold: `mean_floor_very_loose`
- budget: `8`
- L47: active in nonterminal loops
- terminal: confidence-gated top1; otherwise top5/full survivor-set handoff
- convergence hairs: soft-only monitoring
- reasoning scope: headline
- science scope: diagnostic/excluded from headline
- steering: not run in this probe

## Primary Artifacts

- `summary.md`
- `analysis.md`
- `science_recipe_v2_calibration.md`
- `science_recipe_v2_heldout.md`
- `science_source_specific.md`
- `parser_patch_recommendation.md`
- `reasoning_terminal_handoff_v2.md`
- `reasoning_hard_slice_v2.md`
- `science_l47_layer_ablation_v2.md`
- `science_perturbation_escalation_v2.md`
- `soft_hairs_v2.md`
- `integrated_repair_eval.md`
- `pre_steering_readiness_v2.md`



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
claim. Details: `docs/evaluator/bg_autoregressive_kv_branch_carry_v1.md`; artifacts under
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
`docs/evaluator/bg_partial_cache_splice_v2.md`; artifacts under
`artifacts/reports/probes/bg_partial_cache_splice_v2_2026-06-01/`.
