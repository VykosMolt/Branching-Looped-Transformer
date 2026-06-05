# Evaluator Locus Summary

**Original date:** 2026-05-11, extended through 2026-05-14
**Refactored:** 2026-05-18 for readability
**Refreshed and promoted to root:** 2026-06-03 — brought current through the DualAnchor architecture-looped v3 baseline, the science/reasoning repair v2 decision, and the in-progress (paused) MMLU science branch + parser repair v3 run.
**Raw original:** `history/raw_archive_2026-05-18/pairwise_evaluator_locus_memo_v2_2026-05-11.md`
**Prior archived copy:** `history/interfaces-and-tools/evaluator-locus-summary.md` (also snapshotted under `history/pre_docs_consolidation_2026-05-31/`).

The original file was a multi-pass 1,791-line research log covering v2 through v10. This root copy keeps the decision-relevant state in one place and is the canonical, up-to-date locus summary. No historical text was intentionally deleted; use the raw archive for full details.

## Current Distillation

The evaluator signal is best understood as a pairwise branch-selection readout over latent states, not as an absolute standalone scalar target. That framing has held from the original locus work through the current DualAnchor architecture-looped baseline: the production-relevant quantity is a *relational* margin between candidate branches at chosen decoder loci, normalized within the live candidate set.

Key historical findings that still matter:

- Mid/late loop and layer choices matter.
- Masked-zero variants were a numerical degeneracy, not a usable signal.
- Normalization and centered/difference parameterizations helped isolate relational signal.
- GRU/temporal aggregation was not justified as the default.
- Later cross-domain work moved the default toward tiny AntisymLinear and NoNorm heads over pooled features.
- The v10 Thinking-vs-RLTT loop geometry made decoder layers **24 / 36 / 47** the central loci; these are now the per-loop tap points in the architecture-looped baseline.

## Evolution Of The Locus Work

| pass | key point | current status |
| --- | --- | --- |
| v2-v4 | pairwise locus and loop ablations | historical foundation |
| v4-redo | iterated RMSNorm improved HH readout | useful mechanistic context |
| v5 | loop identity became less special under stronger normalization | supports pooled/simple heads |
| math probe | math looked different from HH | superseded by later validity caveats |
| v6 | bias decomposition and branch simulation | supports relational framing |
| v7 | all-layer/cached coding/reasoning/logic probes | historical bridge to domain transfer |
| v8-v9 | evaluator placement and multi-tap ensemble | historical design context |
| v10 | Thinking vs RLTT loop geometry | layer 24/36/47 geometry became central |
| DualAnchor / two-tap | old two taps fused into `MIX_CODE_REASONING` + `MIX_OBJECTIVE_ALL` | active selector (see `dualanchor-tap-evolution.md`) |
| architecture-looped v3 | 24/36/47 geometry became a per-loop survival schedule | active Phase 2a baseline candidate |

## Current Architecture Implication

The current local diagnostic path uses tiny heads:

- `AntisymLinear`: `LayerNorm(no affine)(left - right) -> Linear(no bias)`.
- `AntisymLinearNoNorm`: `Linear(left - right)`.

GRU and larger learned evaluators are no longer default for these probes.

## From Locus Geometry To The Architecture-Looped Baseline

The 24/36/47 loci are now consumed as a per-loop pairwise survival schedule rather than a single readout point. The active Phase 2a branch-selection baseline candidate is:

- **Selector:** DualAnchor `MIX_CODE_REASONING` + `MIX_OBJECTIVE_ALL` (the fused descendants of the old two taps).
- **Schedule:** `L1_24 -> L1_36 -> L1_47 -> L2_24 -> L2_36 -> L2_47 -> L3_24 -> L3_36 -> L3_47 -> L4_24 -> L4_36 -> terminal L4_47`.
- **Threshold:** `mean_floor_very_loose` (per-stage mean floor with loop/layer offsets; z-normalized per tap, then averaged).
- **Budget:** hard budget `8`.
- **L47:** active in nonterminal loops (do not disable earlier-loop L47 as default).
- **Terminal:** confidence-gated top1 **only** when confidence is strong; otherwise defer / keep the terminal survivor set (top5/full handoff).
- **Convergence hairs (L30/L42):** soft-only monitoring; no hard merge in the baseline.

Status: `DUALANCHOR_ARCHITECTURE_LOOPED_V3_STATUS = ARCHITECTURE_LOOPED_SURVIVAL_READY_TERMINAL_DEFER_REQUIRED`,
`BG_DUALANCHOR_PHASE2A_READINESS_V3_VERDICT = READY_WITH_TERMINAL_DEFER`.

This remains selection/branching only. No steering, production-routing change, autoregressive branch-specific KV/cache fork-carry claim, or compute-savings claim is made. See `dualanchor-architecture-baseline.md` and `branch-generation-and-survival.md`.

## Current Domain Status (2026-06-04)

The locus work alone did not settle domain generalization; later domain-transfer and source-specific probes did. The current pre-steering domain decision (v3, 2026-06-04):

- **Reasoning — headline-ready** under the locked terminal handoff (confidence-gated top1 else top5/full survivor-set handoff), **not** under unconditional terminal top1.
- **Science — partially repaired.** MMLU **anatomy** crosses into partial headline-readiness (`ANATOMY_REPAIRED`); chemistry, physics, and SciQ remain excluded/diagnostic. Decision: `READY_FOR_STEERING_REASONING_PLUS_PARTIAL_SCIENCE`. (v2 had science diagnostic-only; v3 supersedes the science half.)

Latest decided status (v2, 2026-06-01):

- `DUALANCHOR_SCIENCE_REASONING_REPAIR_V2_STATUS = REASONING_READY_SCIENCE_DIAGNOSTIC`
- `BG_PRE_STEERING_READINESS_V2_VERDICT = READY_FOR_STEERING_WITH_SCIENCE_DIAGNOSTIC`
- Reasoning terminal: `BG_REASONING_TERMINAL_HANDOFF_V2_VERDICT = REASONING_HANDOFF_LOCKED`.
- Science source-specific: `BG_SCIENCE_V2_SOURCE_SPECIFIC_VERDICT = MMLU_CHEM_ANATOMY_BLOCKED`.
- Parser: `BG_SCIENCE_PARSER_PATCH_RECOMMENDATION_VERDICT = ROBUST_DIAGNOSTIC_ONLY` (robust letter+text parser stays diagnostic; do not silently replace strict reward).
- Weak-source layer signal: `BG_SCIENCE_L47_LAYER_ABLATION_V2_VERDICT = L2_47_HELPS`; perturbation escalation gave no help; soft hairs warn that science converges to a no-good branch.

See `science-reasoning-repair.md` for the full v2 component table.

### MMLU science branch + parser repair v3 (complete, 2026-06-04)

The source-specific follow-up completed (calibration 131/131, 0 errors; downstream pipeline 0 stage failures). Output root: `artifacts/reports/probes/bg_mmlu_science_branch_parser_repair_v3_2026-06-01/`. Run note: `history/bg-run-notes/science-reasoning/bg_mmlu_science_branch_parser_repair_v3.md`.

- **Result:** `SCIENCE_PARTIALLY_REPAIRED` → `READY_FOR_STEERING_REASONING_PLUS_PARTIAL_SCIENCE`. MMLU **anatomy** repaired (heldout positive-oracle 0.333 / selected-parser terminal-best 0.20, parse 1.0, on 3 tasks); chemistry and physics still blocked with parse collapsed to 0.0; SciQ parses but wrong. Heldout overall `SCIENCE_RECIPE_IMPROVED_BUT_WEAK` (7 tasks).
- **Locked component verdicts:** inventory `MMLU_TASKS_LIMITED`; task suite `CHEM_ANATOMY_LIMITED`; parser build `READY`; adversarial parser `ROBUST_PARSER_STILL_DIAGNOSTIC`; prompt format `FORMAT_FIX_HELPS`; recipe plan `SOURCE_SPECIFIC_READY`; calibration `SOURCE_SPECIFIC_RECIPE_FOUND`; L47 `SOURCE_SPECIFIC_L47`; budget/breadth `BUDGET8_SUFFICIENT`; soft-hair `CHEM_ANATOMY_NO_GOOD_CONFIRMED`; reasoning guardrail `REASONING_BASELINE_STILL_READY`.
- **Caveat:** anatomy's gain rests on 3 heldout tasks and the soft-hair still warns; treat it as a candidate partial/secondary headline, not a settled one. Robust parser stays diagnostic-only.

## Numbers To Use Now

### Historical domain-transfer readouts (still valid)

HH all-200 diagnostic:

- HH-trained `47_concat_L1_L4 / NoNorm`: accuracy 0.855.
- code-trained `47_L4 / NoNorm`: accuracy 0.535.

Strict-clean code ALL16:

- HH-trained `47_mean / AntisymLinear`: top1 0.750, pairwise 0.600.
- code-trained `36_L4 / AntisymLinear`: top1 0.875, pairwise 0.833.

Reasoning pilot:

- best code-trained `24_L4 / AntisymLinear`: top1 1.000, pairwise 1.000.
- best HH-trained `36_mean / AntisymLinear`: top1 0.960, pairwise 0.986.

### Current architecture-looped v3 branch-selection metrics

48 tasks (24 reasoning / 24 science):

- stage oracle retention 0.9848
- terminal oracle retained 1.0000
- forced terminal top1 oracle 0.9167
- reward-diverse forced top1 oracle 0.6364
- false-prune recovery 8/8

Survival is strong; the limiter is reward-diverse hard slices at the terminal, which is why the baseline is terminal-defer-ready, not unconditional-terminal-ready.

### v2 reasoning terminal handoff (the lock)

| Policy | Tasks | Oracle retained | First selected oracle | Defer rate |
| --- | ---: | ---: | ---: | ---: |
| `dualanchor_confidence_gated` | 24 | 1.0000 | 0.8750 | 0.9583 |
| `dualanchor_terminal_top5` | 24 | 1.0000 | 0.8750 | 0.9583 |
| `terminal_defer_all` | 24 | 1.0000 | 0.8750 | 0.9583 |
| `dualanchor_terminal_top2` | 24 | 0.9583 | 0.8750 | 0.9583 |
| `dualanchor_forced_top1` | 24 | 0.8750 | 0.8750 | 0.9583 |

### v2 science heldout (did not validate repair)

- heldout recipe `baseline_v3_regenerated`, 7 tasks
- selected-parser positive-oracle rate 0.0000
- selected-parser terminal best reward -0.0571
- reward-diverse rate 0.5714
- terminal oracle retained 1.0000

## Current Recommendation

Use a general-plus-specialist readout policy inside the locked architecture-looped baseline:

- HH-trained general head for HH/preference-like comparisons.
- code-trained specialist for strict-clean code.
- reasoning is a validated third objective domain at the headline level **only with the survivor-set terminal handoff**.
- keep science as a diagnostic domain until the v3 MMLU run produces source-specific heldout evidence; do not headline science or promote the robust parser without that validation.

Do not use the long historical memo as the current operating spec without also reading:

- `current-state.md`
- `evaluator-navigation-map.md`
- `dualanchor-architecture-baseline.md`
- `domain-transfer-ledger.md`
- `science-reasoning-repair.md`
- `interfaces-and-tools.md`

## Scope And Constraints

This summary covers selection/branching diagnostics only. It makes no steering, Ouro-training, tokenizer/checkpoint, tap-registry, production-routing, hard convergence-hair merge, compute-savings, or autoregressive fork/carry claim. The terminal confidence gate stays in place; earlier-loop L47 stays on by default; the robust science parser stays diagnostic until validated.
