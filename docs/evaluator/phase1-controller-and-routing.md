# Phase 1 Controller And Routing

Updated: 2026-05-31

This is the consolidated entry point for the Phase 1 read-only BG controller, routing policy, and trajectory-prediction work.

## Status

Phase 1 remains a read-only selector/routing layer over frozen Ouro hidden states.

The Phase 1 routing lock is historical but still valid as the production-routing boundary:

- no production routing change was made by later DualAnchor work,
- no steering claim follows from Phase 1 routing,
- Phase 2a DualAnchor selection is an experimental branch/prune baseline, not production routing.

## Phase 1 Locked Policy

From the v8.1 synthesis:

- HH/preference/unknown domains route to `hh_general`.
- Code, reasoning, science, math, and objective tasks route to `objective_mixed_primary`.
- `code_specialist_backup` remains for ablation, tie-break, low-margin, and disagreement cases.

The old strict-clean code assumption was corrected: objective-mixed was at least as good as the code specialist at the then-current sample size.

## Head Roles

| Role | Function |
| --- | --- |
| `hh_general` | HH/preference-like comparisons and unknown preference-shaped tasks. |
| `objective_mixed_primary` | Default objective-domain branch selection across code, reasoning, science, and math. |
| `code_specialist_backup` | Backup and ablation reference for strict-clean code or low-margin cases. |

## Trajectory Prediction

The trajectory-prediction work established that BG reads useful branch-quality information before final answers are complete.

Current remembered verdict:

`BG_TRAJECTORY_PREDICTION_VERDICT = STRONG`

Interpretation:

BG is not just a finished-answer reranker. It can read partial trajectory quality from frozen hidden states.

## Relationship To DualAnchor

DualAnchor (`MIX_CODE_REASONING` + `MIX_OBJECTIVE_ALL`) grew out of the objective-mixed/code-reasoning line. It is now the active Phase 2a branch/action selector, but it does not replace Phase 1 production routing.

Use:

- `dualanchor-architecture-baseline.md` for the active Phase 2a baseline.
- `dualanchor-tap-evolution.md` for the old-tap to DualAnchor transition.

## Source Notes Moved To History

Detailed source docs were moved under `history/phase1-controller-and-routing/`.

Exact pre-consolidation root copies are also preserved under:

`history/pre_docs_consolidation_2026-05-31/`

