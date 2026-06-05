# DualAnchor convergence hairs and reasoning/science pre-steering probe v1

Date: 2026-05-31

Artifact root:
`artifacts/reports/probes/bg_dualanchor_convergence_hairs_reasoning_science_v1_2026-05-31/`

## Top verdicts

- `BG_CONVERGENCE_HAIRS_RS_INVENTORY_VERDICT = PARTIAL`
- `BG_CONVERGENCE_HAIR_DATASET_VERDICT = READY`
- `BG_CONVERGENCE_HAIR_POLICY_DEF_VERDICT = READY`
- `BG_CONVERGENCE_HAIR_REPLAY_EVAL_VERDICT = INSUFFICIENT`
- `BG_CONVERGENCE_HAIR_REGENERATED_VERDICT = SKIPPED`
- `BG_TIE_DECOMPOSITION_DIAGNOSTIC_VERDICT = SCIENCE_TIES_ARE_NO_GOOD_BRANCH`
- `BG_REASONING_HARD_SLICE_VERDICT = REASONING_NEEDS_TERMINAL_DEFER`
- `BG_SCIENCE_HARD_SLICE_VERDICT = SCIENCE_BRANCH_GENERATION_WEAK`
- `BG_DOMAIN_BRANCH_RECIPE_DIAGNOSTIC_VERDICT = SCIENCE_NEEDS_DIFFERENT_RECIPE`
- `BG_TERMINAL_DEFER_POLICY_VERDICT = CONFIDENCE_TOP1_READY_WITH_DEFER`
- `BG_PRE_STEERING_READINESS_VERDICT = NEEDS_SCIENCE_BRANCH_RECIPE_FIRST`
- `DUALANCHOR_CONVERGENCE_HAIRS_RS_STATUS = SCIENCE_BRANCH_GENERATION_WEAK`

## Why this was run

The v3 DualAnchor architecture-looped probe showed strong survival but many branch ties.
This probe tested whether L30/L42 convergence hairs could safely reduce redundant branch
carry before steering work, while separately diagnosing reasoning and science hard slices.

This was not steering. No Ouro weights, tokenizer files, checkpoints, tap registries,
production routing, wrapper/local-agent behavior, or Hunter-Seeker execution were changed.

## Architecture context

The locked selector remains DualAnchor:

- `MIX_CODE_REASONING`
- `MIX_OBJECTIVE_ALL`

The locked baseline schedule remains:

`L1_24 -> L1_36 -> L1_47 -> L2_24 -> L2_36 -> L2_47 -> L3_24 -> L3_36 -> L3_47 -> L4_24 -> L4_36 -> terminal L4_47`

The v3 baseline uses `mean_floor_very_loose`, budget `8`, active nonterminal L47, and
terminal confidence-gated top1 with defer/survivor-set handoff otherwise.

## Branch classification boundary

Branch classification and tie taxonomy were diagnostic only. They were not used as a
runtime architecture component and are not part of the convergence-hair merge policy.

Allowed runtime-style hair signals in this probe were hidden-state distance, DualAnchor
score or margin proxies, anchor disagreement, lineage/root/parent relation, and
stability/OOD guards. Final reward, oracle labels, parsed answers, output text similarity,
and diagnostic tie labels were evaluation-only or report-only signals.

## L30/L42 hair mechanism

The probe found L30/L42 hidden states in the existing v3 `.pt` rows:

- L30 hairs between L24 and L36.
- L42 hairs between L36 and L47.
- Hair rows covered `L1_30..L4_30` and `L1_42..L4_42`.
- Logits were not available at the hair surfaces, so logit-gated policies were limited.

The dataset contains `2592` hair candidate rows and `8400` same-task same-hair pair rows
across the 48-task v3 suite. This replay dataset is suitable for safety diagnostics, but
it is not a compute-savings claim.

## Merge safety result

Hard merge was not cleared.

Representative L30+L42 merge retained terminal oracle on `0.9583` of tasks, with
false-merge rate `0.0208` and survivor reduction `0.0247`. L30-only and L42-only
representative merge each retained `0.9792`, below the `>= 0.98` candidate threshold,
and neither reached meaningful survivor reduction. Conservative and hidden-convergence
policies preserved oracle retention, but survivor reduction was below 1 percent.

The useful hard-merge gate was therefore not met: no non-diagnostic policy achieved
terminal oracle retained `>= 0.98`, false merge `<= 0.05`, and survivor reduction `>= 0.10`.

Recommended boundary: use L30/L42 hairs as soft monitoring/diagnostic clusters only
unless a later regenerated run confirms hard merge safety.

## Tie decomposition

The tie decomposition verdict is `SCIENCE_TIES_ARE_NO_GOOD_BRANCH`.

Reward ties are not a sufficient merge signal. The reward-tie diagnostic control was
unsafe and collapsed oracle availability in replay. Science tie-heavy cases were mostly
all-zero or no-good-branch cases rather than useful convergence. Reward/preference ties
must not be converted into arbitrary branch classes for runtime architecture.

## Reasoning hard slice

Reasoning remains viable with terminal defer but not forced final collapse.

- Reasoning tasks: `24`
- Terminal best reward: `0.5833`
- Forced top1 reward: `0.4833`
- Forced top1 oracle: `0.8750`
- Reasoning reward-diverse forced top1 oracle: `0.5000`
- Reasoning positive+reward-diverse forced top1 oracle: `0.5000`

The reasoning verdict is `REASONING_NEEDS_TERMINAL_DEFER`. Top2 improves oracle
retention, and full terminal survivor-set handoff preserves all observed reasoning
oracle options in this dataset.

## Science hard slice and parser audit

Science is branch-generation limited in this suite.

- Science tasks: `24`
- Science positive-oracle rate: `0.0833`
- Science reward-diverse rate: `0.2083`
- Science terminal best reward: `0.0500`
- Science forced top1 reward: `0.0417`
- Science forced top1 oracle: `0.9583`

The parser audit found parse success rate `0.7610` over science candidate outputs, with
`413` `no_mcq_letter` parse failures and `195` cases where a standalone correct option
letter pattern appeared despite parser failure or mismatch. Parser quality should still
be watched, but the main science failure is that most science tasks never produce a
correct terminal survivor.

The science verdict is `SCIENCE_BRANCH_GENERATION_WEAK`.

## Domain branch recipe

The domain recipe diagnostic verdict is `SCIENCE_NEEDS_DIFFERENT_RECIPE`.

Reasoning positive-oracle rate was `0.6250`; science positive-oracle rate was `0.0833`.
This gap is too large to treat science as merely a terminal-selection problem. Science
needs a branch-generation recipe audit before it can be used as a headline steering
domain.

## Terminal defer policy

The terminal policy verdict is `CONFIDENCE_TOP1_READY_WITH_DEFER`.

Across all 48 tasks, confidence-gated top1 plus defer preserved oracle availability
under the survivor-set semantics. Forced top1 remained weaker than terminal survivor-set
handoff. Do not remove the terminal confidence gate.

## Pre-steering readiness

The pre-steering readiness verdict is `NEEDS_SCIENCE_BRANCH_RECIPE_FIRST`.

Recommended locked baseline, if moving into steering-adjacent work, is:

- schedule: `L1_24 -> L1_36 -> L1_47 -> L2_24 -> L2_36 -> L2_47 -> L3_24 -> L3_36 -> L3_47 -> L4_24 -> L4_36 -> terminal L4_47`
- selector: DualAnchor `MIX_CODE_REASONING` + `MIX_OBJECTIVE_ALL`
- threshold: `mean_floor_very_loose`
- budget: `8`
- L47: active in nonterminal loops
- terminal: confidence-gated top1; otherwise terminal defer / survivor-set handoff
- convergence hairs: soft-only diagnostics unless regenerated execution later confirms hard merge safety
- lineage: required
- steering: not run in this probe

## Non-claims

This probe does not claim steering readiness for science as a headline domain, production
routing readiness, compute savings, autoregressive branch-specific KV/cache fork-carry,
unconditional terminal top1 readiness, or branch classification as runtime architecture.

## Primary files

- `summary.md`
- `analysis.md`
- `inventory.md`
- `convergence_hair_dataset.md`
- `convergence_hair_policies.md`
- `convergence_hair_replay_eval.md`
- `convergence_hair_regenerated.md`
- `tie_decomposition_diagnostic.md`
- `reasoning_hard_slice.md`
- `science_hard_slice.md`
- `domain_branch_recipe.md`
- `terminal_defer_policy.md`
- `pre_steering_readiness.md`

## DualAnchor science branch recipe and reasoning terminal defer v1 (2026-05-31)

Follow-up status: `DUALANCHOR_SCIENCE_RECIPE_REASONING_DEFER_STATUS = PRE_STEERING_READY_WITH_SCIENCE_DIAGNOSTIC`.

The next pre-steering probe kept convergence hairs soft-only and tested the remaining
science/reasoning blockers. Science parser/reward was partly responsible, but no replay
recipe improved over baseline on calibration and heldout remained
`SCIENCE_BRANCH_GENERATION_STILL_WEAK`. MMLU science was the weak source family. Reasoning
terminal policy was locked to confidence-gated top1 else top5/full survivor handoff.

Domain scope for Phase 2b steering comparison is reasoning headline with science as a
diagnostic slice only. No steering or production change was made.

Report: `docs/evaluator/bg_dualanchor_science_branch_recipe_reasoning_defer_v1.md`.

## DualAnchor science and reasoning repair v2 (2026-06-01)

Follow-up status: `DUALANCHOR_SCIENCE_REASONING_REPAIR_V2_STATUS = REASONING_READY_SCIENCE_DIAGNOSTIC`.

The v2 repair run kept L30/L42 convergence hairs soft-only. The hair verdict is
`BG_SOFT_HAIRS_V2_VERDICT = SCIENCE_NO_GOOD_WARNING_USEFUL`: useful as no-good-branch
warning diagnostics, but not hard merge gates. Reasoning terminal handoff is locked.
Science still does not clear headline readiness after regenerated heldout.

Report: `docs/evaluator/bg_dualanchor_science_reasoning_repair_v2.md`.


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
