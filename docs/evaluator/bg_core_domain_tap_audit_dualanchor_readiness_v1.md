# Core-Domain Tap Audit and DualAnchor Readiness v1 (2026-06-04)

Output root: `artifacts/reports/probes/bg_core_domain_tap_audit_dualanchor_readiness_v1_2026-06-04/`
Short name: `core_domain_tap_audit_dualanchor_readiness_v1`

## Headline

`CORE_DOMAIN_TAP_AUDIT_STATUS = CORE_TAPS_READY`
`BG_CORE_PRE_STEERING_READINESS_VERDICT = READY_FOR_STEERING_CORE_DOMAINS`
`BG_CORE_TAP_POLICY_SELECTION_VERDICT = KEEP_SCIENCE_DIAGNOSTIC_ONLY`

The locked **DualAnchor** baseline is carried into Phase 2b **unchanged**. Coding, reasoning,
math, logic, and alignment are the core domains and are sufficiently backed by clean
verifier/exact-answer/MCQ/preference labels. Science/anatomy taps stay **diagnostic only**.
No steering was run, trained, applied, or claimed; no production routing changed.

## Why broad science is no longer the steering gate

The preceding `mmlu_science_branch_parser_repair_v3` run concluded
`SCIENCE_PARTIALLY_REPAIRED` / `READY_FOR_STEERING_REASONING_PLUS_PARTIAL_SCIENCE`:
anatomy is a small-n partial; chemistry/physics/SciQ remain excluded/diagnostic. So this
audit pivots to the domains that matter for steering now (coding/reasoning/math/logic/
alignment) and tests science/anatomy taps **as auxiliary experts**, not as a headline domain.

## Tap inventory (Part A — `READY`)

45 taps/heads across 9 families. Every record carries `capability`
(`action_selection` vs `action_selection_and_branching`) and `needs_transplant_for_branching`:

- **DualAnchor** (`MIX_CODE_REASONING` + `MIX_OBJECTIVE_ALL`), 16 canonical taps + a 2,358-candidate
  transplant grid — natively selection **and** branching (old + branch + bridge transplant).
- **mixed-domain tiny heads** (`MIX_CODE_REASONING`, `MIX_OBJECTIVE_ALL`, `MIX_HH_OBJECTIVE`,
  and the science specialists `MIX_CODE_SCIENCE`, `MIX_REASONING_SCIENCE`, `MIX_CODE_SCIENCE_MED`) —
  content/selection only.
- **old content heads** (HH, CODE) — selection only.
- **HH published pairwise evaluator** (`pairwise_epoch2.pt`, stored acc 0.621).
- branch / bridge / universal-gated / fixed-composite / merged-weight / final-arbiter artifacts.

Content-only taps are selection-only by design and must be **transplanted** to be branch-capable —
they are never scored "naked" on branching. The transplant is the **canonical** weight-space op
`normed(a·old + b·masked_residual(branch,[old]) + c·masked_residual(bridge,[old,branch_res]))` with
the **real aligned source taps** — branch = `hidden_origin_v4`, bridge = `universal` (stored in the
constrained registry as the `branch_full_00_100_diagnostic` / `bridge_full_00_100_diagnostic`
recipes), canonical coeffs `(0.80,0.10,0.10)` plus a grid. **Both the pure content tap weights and
the transplanted (branch-capable) weights are saved** under `constructed_taps/`
(`pure_content_taps.pt`, `transplanted_taps.pt`, `transplant_manifest.{json,csv,md}`); no existing
registry is mutated.

## Task / candidate suite (Part B — `READY`)

Feature-bearing candidate groups (calibration/heldout task-disjoint):

| domain | groups | reward-diverse | source |
| --- | ---: | ---: | --- |
| coding | 30 | 30 | `code_branch_tap_features_v2` (+mini) |
| reasoning | 24 | 5 | arch-looped branch trees |
| math | 66 | 66 | math + gsm8k tap features |
| logic | 80 | 80 | **LogiQA** MCQ-option features (extracted this run) |
| alignment | 200 | 200 | HH layer-state packs |
| science / anatomy | 24 | 3 | arch-looped (mmlu aggregate; diagnostic) |

Logic had no local features; LogiQA (`lucasmccabe/logiqa`, parquet export) was pulled and a
bounded GPU **encode-only** MCQ-option extraction produced 80 logic groups (40 cal / 40 heldout).
Anatomy is proxied by science-aggregate trees (the arch-looped probe logged `mmlu` generically).

## Antisymmetry / bias (Part C — `ANTISYMMETRIC_SCORES_READY`)

The tiny `AntisymLinear` / `AntisymLinearNoNorm` heads are **exactly antisymmetric** by
construction (LayerNorm-no-affine + bias-free linear): strict sign-flip 0, antisymmetry
correlation 1.0, raw == antisymmetrized accuracy. Low sign-flip is therefore a construction fact,
not an accuracy estimate. The **published HH evaluator's stored accuracy is 0.621** — i.e. the
~62% pointwise number, *not* the 95.2% pairwise fixed-order figure; that distinction is preserved.

## Action/content selection (Part D — `SCIENCE_TAPS_HELP`)

Macro top1-oracle over core domains: `mixedhead_MIX_HH_OBJECTIVE` 0.643, `MIX_OBJECTIVE_ALL` 0.620,
`science_MIX_REASONING_SCIENCE` 0.613, `MIX_CODE_REASONING` 0.610, DualAnchor ~0.583. For **pure
content selection** the broad-objective tiny heads marginally lead, and a science head edges
DualAnchor by ~0.03 (small-n) — which trips the "science helps" threshold here. This is selection
only; it does not carry into branching or the policy (see Parts E/J/N).

## Branch survival (Part E — `DUALANCHOR_SURVIVAL_CONFIRMED`)

DualAnchor retains an oracle at top-8 with macro 0.96 across core domains, matching or leading the
field. Running content-only taps through the **canonical transplant** gives a small but positive
mean top1 gain of **+0.012** (e.g. `science_MIX_CODE_SCIENCE` **+0.050**, `old_HH` +0.010;
`science_MIX_REASONING_SCIENCE` −0.013) — the transplant adds branch validity, modestly, and makes
the content-only comparison fair rather than handicapped. The per-config residual structure is real
(at 24_L4 the bridge residual collapses to ~0.15 because the bridge is nearly in the old span there;
at 47_L4 both branch and bridge are fully independent). DualAnchor remains confirmed for survival.

## Terminal handoff (Part F — `FULL_HANDOFF_REQUIRED`)

Forced terminal top1 is not safe on the core heldout; top5/full survivor-set retains the oracle.
The locked **confidence-gated top1 else top5/full survivor-set handoff** stands; terminal top1 is
not promoted. The terminal confidence gate is not removed.

## Science/anatomy cross-domain (Parts G / I / J)

- `BG_SCIENCE_TAP_CROSS_DOMAIN_VERDICT = SCIENCE_TAPS_USEFUL_AUXILIARY` — but only as a **marginal
  content-selection edge** (~0.03, small-n); the science third-expert survival gain is **0.0**.
- `BG_CORE_TAP_GEOMETRY_VERDICT = SCIENCE_RESIDUAL_INDEPENDENT` — science taps occupy an essentially
  independent direction (residual ≈1.0 on the DualAnchor span); independent but not transferable to
  core performance.
- `BG_CORE_THRESHOLD_ANCHOR_ABLATION_VERDICT = SCIENCE_ANCHOR_NOT_USEFUL` — adding science as a
  third anchor does not improve survival.

Net: science is geometrically distinct and marginally helpful for raw selection, but survival-neutral,
so it stays **diagnostic only**.

## Transfer matrix (Part H — `MIX_OBJECTIVE_DOMINATES`)

Most robust by mean core top1: `mixedhead_MIX_HH_OBJECTIVE`; DualAnchor mean core rank ≈ 3.8. Best
per domain: alignment/coding/science → `MIX_*`/code-reasoning heads; reasoning → DualAnchor; logic →
`science_MIX_REASONING_SCIENCE`; math → `MIX_CODE_REASONING`. The broad objective heads lead pure
selection; DualAnchor leads where branching matters.

## Alignment (Part L — `MIX_OBJECTIVE_TRANSFERS_TO_ALIGNMENT`)

On HH pairs: `MIX_HH_OBJECTIVE` pairwise 0.742 / top1 0.75 (best), `old_HH` 0.642, `MIX_OBJECTIVE_ALL`
0.592. The broad objective taps transfer to alignment; the published evaluator stays a diagnostic
reference (stored acc 0.621). Alignment is a viable core headline domain.

## Parsers / verifiers (Part M — `CODING_VERIFIER_READY`)

Core-domain labels are deterministic (coding unit-test pass/fail, math exact-answer, logic/reasoning
MCQ-correct) — materially cleaner than the science MCQ-letter parser that collapsed in v3.

## Partial-cache splice smoke (Part K — `SPLICE_VALID_SELECTION_LIMITED`)

A bounded runtime smoke (math + logic generation, DualAnchor scoring) ran clean with generous token
budgets (extra for math, which is verbose). Splice **math equivalence + compute-saving** are cited
from `bg_partial_cache_splice_v2` (`PARTIAL_SPLICE_COMPUTE_SAVING_VALID`); they are **not** re-derived
here and **no** compute or production-routing claim is made beyond that validated test-harness result.

## Two honest findings

- **The two DualAnchor anchors are nearly collinear**: cos(`MIX_CODE_REASONING`,`MIX_OBJECTIVE_ALL`)
  = 1.00 @24_L4, 0.92 @36_L4, 0.998 @47_L4. The "dual" distinction is real only around L36; elsewhere
  the anchors point almost the same direction.
- **Selection vs branching split**: broad-objective tiny heads marginally win *pure content selection*,
  but DualAnchor is the only natively *selection+branching* tap and wins survival — which is why it is
  locked as the default rather than a pure-selection head.

## Final tap policy and locked baseline

`BG_CORE_TAP_POLICY_SELECTION_VERDICT = KEEP_SCIENCE_DIAGNOSTIC_ONLY`

- selector: **DualAnchor** `MIX_CODE_REASONING` + `MIX_OBJECTIVE_ALL` (unchanged)
- third expert: **none** (science survival-neutral → diagnostic only)
- schedule: `L1_24 -> L1_36 -> L1_47 -> L2_24 -> L2_36 -> L2_47 -> L3_24 -> L3_36 -> L3_47 -> L4_24 -> L4_36 -> terminal L4_47`
- threshold: `mean_floor_very_loose`; budget: `8`; L47: active in nonterminal loops
- terminal: confidence-gated top1 else top5/full survivor-set handoff
- convergence hairs: soft-only
- cache: partial-cache splice v2 available as **test-harness-validated** infrastructure (no production claim)
- science: diagnostic only; steering: **not run**

## Pre-steering readiness

`BG_CORE_PRE_STEERING_READINESS_VERDICT = READY_FOR_STEERING_CORE_DOMAINS` — coding, reasoning, math,
logic, and alignment meet the suite minimums with clean labels, DualAnchor confirmed for survival, and
alignment transfer confirmed. The recommended Phase 2b domain scope is the five core domains under the
locked DualAnchor baseline above, with science/anatomy carried as diagnostics only.

## Caveats

- reasoning/science branch trees are mostly single-reward (5/3 reward-diverse groups) → limited
  hard-slice depth there;
- logic is MCQ-option features (80 tasks), not multi-loop branch trees → its branch-survival is a
  candidate-group retention proxy;
- the science "selection help" rests on small-n margins (~0.03) and is survival-neutral.

No steering, Ouro training, tokenizer/checkpoint edit, tap-registry mutation, wrapper/local-agent
execution, Hunter-Seeker execution, ARC/MATH action loop, production-routing change, hard
convergence-hair merge, or compute-savings/fork-carry claim was made. Constructed transplant taps were
written only under this run's output root.
