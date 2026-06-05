# DualAnchor science branch recipe and reasoning terminal defer v1

Date: 2026-05-31

Artifact root:
`artifacts/reports/probes/bg_dualanchor_science_branch_recipe_reasoning_defer_v1_2026-05-31/`

## Top verdicts

- `BG_SCIENCE_RECIPE_REASONING_DEFER_INVENTORY_VERDICT = PARTIAL`
- `BG_SCIENCE_RECIPE_REASONING_TASK_SUITE_VERDICT = SCIENCE_LIMITED`
- `BG_SCIENCE_PARSER_REWARD_AUDIT_VERDICT = PARSER_PARTLY_RESPONSIBLE`
- `BG_SCIENCE_BRANCH_RECIPE_PLAN_VERDICT = DIRECTION_LIMITED`
- `BG_SCIENCE_BRANCH_RECIPE_CALIBRATION_VERDICT = SCIENCE_PARSER_DOMINATES`
- `BG_SCIENCE_BRANCH_RECIPE_HELDOUT_VERDICT = SCIENCE_BRANCH_GENERATION_STILL_WEAK`
- `BG_SCIENCE_SOURCE_BREAKDOWN_VERDICT = MMLU_SCIENCE_WEAK`
- `BG_REASONING_TERMINAL_DEFER_VERDICT = REASONING_FULL_SURVIVOR_HANDOFF_REQUIRED`
- `BG_REASONING_BRANCH_GENERATION_SANITY_VERDICT = REASONING_BASELINE_SUFFICIENT`
- `BG_SCIENCE_L47_INTERACTION_VERDICT = L47_NOT_ENOUGH_FOR_SCIENCE`
- `BG_SOFT_HAIRS_SCIENCE_REASONING_VERDICT = SCIENCE_CONVERGES_TO_NO_GOOD_BRANCH`
- `BG_SCIENCE_PERTURBATION_ESCALATION_VERDICT = DIAGNOSTIC_ONLY`
- `BG_PRE_STEERING_DOMAIN_DECISION_VERDICT = READY_FOR_STEERING_WITH_SCIENCE_DIAGNOSTIC`
- `DUALANCHOR_SCIENCE_RECIPE_REASONING_DEFER_STATUS = PRE_STEERING_READY_WITH_SCIENCE_DIAGNOSTIC`

## Why this run was needed

The prior convergence-hair pre-steering probe left one main blocker: science branch
generation. Architecture-looped survival was ready, hard L30/L42 convergence-hair merge
was not ready, reasoning needed terminal defer/survivor handoff, and science was too weak
for headline steering.

This run checked whether the remaining blocker was parser/reward, branch recipe, source
specificity, L47 interaction, or terminal policy. It did not run steering.

## Prior result carried forward

- Hard convergence-hair merge remains off the baseline.
- L30/L42 hairs remain soft diagnostics only.
- DualAnchor selector remains `MIX_CODE_REASONING` + `MIX_OBJECTIVE_ALL`.
- The locked architecture-looped schedule and `mean_floor_very_loose` threshold remain
  unchanged.
- Terminal behavior remains confidence-gated top1, otherwise defer / survivor-set handoff.

## Task suite

The local audit plan contained only `31` eligible science MCQ tasks and `27` eligible
reasoning tasks. The suite selected all `31` science tasks and the top `24` reasoning
tasks. Because the requested science minimum was `32`, the suite verdict is
`SCIENCE_LIMITED`.

Only `48` selected tasks had v3-generated candidate trees, so branch recipe testing in
this pass is replay diagnostic unless otherwise stated.

## Science parser audit

The parser/reward audit verdict is `PARSER_PARTLY_RESPONSIBLE`.

- strict parse success rate: `0.7610`
- strict positive-oracle rate: `0.0833`
- robust letter+text diagnostic positive-oracle rate: `0.2500`
- strict terminal best reward: `0.0500`
- robust diagnostic terminal best reward: `0.2250`
- no-MCQ-letter failures: `413`
- missed-correct candidate count: `195`
- robust false-positive risk proxy: `0.7899`

Parser weaknesses are real, especially for chemistry/anatomy-style outputs, but the
robust diagnostic has high ambiguity risk. Strict parser remains primary until a parser
patch is separately validated.

## Science branch recipes tested

The recipe plan covered baseline v3, L24-heavy, L36-heavy, L47-heavy, L2_47-emphasis,
DualAnchor-aligned replay, bridge/materialization, random orthogonal control, alpha
variants, more-children same-budget, budget-10 diagnostic, perturbation escalation, and
parser-aware diagnostic scoring.

Only recipes represented in the v3 candidate tree had replay evidence. Several requested
families require regenerated branch trees and are marked direction/data limited.

## Selected science recipe

No replay recipe improved over baseline on calibration. The best recipe record falls back
to `baseline_v3`; `selected = false`.

Calibration science rows had:

- positive-oracle rate: `0.0000`
- reward-diverse rate: `0.0588`
- terminal best reward: `-0.0471`
- robust parser diagnostic terminal best reward: `0.0824`

The calibration verdict is `SCIENCE_PARSER_DOMINATES` because parser-corrected diagnostics
improve the strict score, but no generation recipe improved the candidate tree.

## Science heldout result

Heldout science replay still has verdict `SCIENCE_BRANCH_GENERATION_STILL_WEAK`.

Heldout baseline rows had:

- task count: `7`
- positive-oracle rate: `0.2857`
- reward-diverse rate: `0.5714`
- terminal best reward: `0.2857`
- forced top1 reward: `0.2857`
- robust parser diagnostic terminal best reward: `0.5714`

This heldout subset is too small and no calibration-selected recipe improved over
baseline, so it does not unlock science headline readiness.

## Science source breakdown

The source breakdown verdict is `MMLU_SCIENCE_WEAK`.

- `sciq`: positive-oracle `1.0000`, terminal best reward `1.0000`
- `mmlu_high_school_physics`: positive-oracle `0.2000`, terminal best reward `0.2000`
- `mmlu_high_school_chemistry`: positive-oracle `0.0000`, terminal best reward `-0.0286`
- `mmlu_anatomy`: positive-oracle `0.0000`, terminal best reward `-0.0545`

Science weakness is not uniform. SciQ is viable in this small sample, while MMLU anatomy
and chemistry remain weak and parser-sensitive.

## Reasoning terminal defer

The reasoning terminal verdict is `REASONING_FULL_SURVIVOR_HANDOFF_REQUIRED`.

- forced top1 oracle retained: `0.8750`
- top2 oracle retained: `0.9583`
- top5/full survivor oracle retained: `1.0000`
- confidence-gated survivor semantics retained oracle: `1.0000`
- defer rate: `0.9583`

Reasoning should use confidence-gated top1 only when confident; otherwise hand off the
terminal survivor set. Top2 is not sufficient on this dataset.

## Reasoning branch sanity

The reasoning branch-generation sanity verdict is `REASONING_BASELINE_SUFFICIENT`.
Reasoning does not need the science recipe diagnostics applied to generation. The weakness
is terminal selection/handoff, not branch generation.

## L47/science interaction

The L47 verdict is `L47_NOT_ENOUGH_FOR_SCIENCE`. L47 remains required in the locked
baseline because previous v3 ablations showed L47 is necessary for survival, but L47-heavy
replay by itself does not solve science.

## Soft convergence-hair monitoring

The soft-hair verdict is `SCIENCE_CONVERGES_TO_NO_GOOD_BRANCH`. L30/L42 hairs remain
useful as monitoring signals for science no-good-branch collapse, but they remain
diagnostic only and are not hard merge gates.

## Perturbation escalation

The perturbation escalation verdict is `DIAGNOSTIC_ONLY`. This pass audited trigger
conditions over v3 science stage rows; it did not run escalated generation.

## Pre-steering domain decision

The domain decision verdict is `READY_FOR_STEERING_WITH_SCIENCE_DIAGNOSTIC`.

Recommended domain scope:

- reasoning: headline domain
- science: diagnostic slice only

Science is not headline-ready until a regenerated branch recipe or validated parser patch
improves heldout behavior.

## Locked baseline

Use this baseline if moving to Phase 2b steering comparison:

- schedule: `L1_24 -> L1_36 -> L1_47 -> L2_24 -> L2_36 -> L2_47 -> L3_24 -> L3_36 -> L3_47 -> L4_24 -> L4_36 -> terminal L4_47`
- selector: DualAnchor `MIX_CODE_REASONING` + `MIX_OBJECTIVE_ALL`
- threshold: `mean_floor_very_loose`
- budget: `8`
- L47: active in nonterminal loops
- terminal: confidence-gated top1; otherwise defer / survivor-set handoff
- convergence hairs: soft-only diagnostics
- science recipe: diagnostic/excluded unless regenerated recipe improves heldout

## Explicit no-steering statement

This run did not test steering, train steering modules, apply steering, train Ouro, modify
Ouro weights, modify tokenizer/checkpoint files, update tap registries, run wrapper or
local-agent code, execute Hunter-Seeker modules, run ARC action loops, run MATH
generation, claim compute savings, claim autoregressive fork/carry, deploy hard
convergence-hair merge, or change production routing.

## DualAnchor science and reasoning repair v2 (2026-06-01)

Follow-up status: `DUALANCHOR_SCIENCE_REASONING_REPAIR_V2_STATUS = REASONING_READY_SCIENCE_DIAGNOSTIC`.

The regenerated v2 run supersedes this v1 diagnostic as the active pre-steering
science/reasoning decision. It kept the same locked DualAnchor looped baseline and
confirmed reasoning headline readiness with confidence-gated top1 else top5/full
survivor-set handoff. Science remains diagnostic/excluded: calibration found some
source-specific recipe signal, but heldout was
`BG_SCIENCE_RECIPE_V2_HELDOUT_VERDICT = SCIENCE_BRANCH_GENERATION_STILL_WEAK`, MMLU
chemistry/anatomy were branch-generation blocked, and parser changes remain
diagnostic-only.

Report: `docs/evaluator/bg_dualanchor_science_reasoning_repair_v2.md`.
Artifacts: `artifacts/reports/probes/bg_dualanchor_science_reasoning_repair_v2_2026-06-01/`.
