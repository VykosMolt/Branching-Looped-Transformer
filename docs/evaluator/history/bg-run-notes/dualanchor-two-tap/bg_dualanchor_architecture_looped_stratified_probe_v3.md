<!-- docs-consolidation-source-note -->
> Consolidation note (2026-05-31): this is a source run note. The current consolidated interpretation is in `dualanchor-architecture-baseline.md`. Exact pre-consolidation text is archived under `docs/evaluator/history/pre_docs_consolidation_2026-05-31/`.

# DualAnchor architecture-looped stratified probe v3

This document records the v3 architecture-shaped DualAnchor branch/prune probe.

## DualAnchor architecture-looped stratified probe v3 (2026-05-31)

Status: `ARCHITECTURE_LOOPED_SURVIVAL_READY_TERMINAL_DEFER_REQUIRED`.

This run scaled the DualAnchor architecture-shaped loop without steering. Taps were active at layers 24, 36, and 47 across loops L1-L4, with only terminal `L4_47` eligible for confidence-gated collapse. It uses cumulative hook approximation at decoder-layer surfaces; it does not claim autoregressive branch-specific KV/cache fork/carry or compute savings.

Headline metrics:

- tasks: `48`
- stage oracle retention: `0.9848484848484849`
- terminal oracle retained: `1.0`
- terminal forced top1 oracle: `0.9166666666666666`
- terminal reward-diverse rate: `0.22916666666666666`
- positive-oracle rate: `0.3541666666666667`

Locked-baseline candidate:

- selector: DualAnchor `MIX_CODE_REASONING` + `MIX_OBJECTIVE_ALL`
- schedule: `L1_24 -> L1_36 -> L1_47 -> L2_24 -> L2_36 -> L2_47 -> L3_24 -> L3_36 -> L3_47 -> L4_24 -> L4_36 -> terminal L4_47`
- threshold: `mean_floor_very_loose`
- budget: `8`
- L47: active in nonterminal loops
- terminal: confidence-gated top1; otherwise defer/keep terminal survivors

Readiness verdict: `READY_WITH_TERMINAL_DEFER`.
No steering was tested.

## Artifact Root

`/home/moloch/ouro_project/artifacts/reports/probes/bg_dualanchor_architecture_looped_stratified_probe_v3_2026-05-31`

## DualAnchor convergence hairs and reasoning/science pre-steering probe v1 (2026-05-31)

Follow-up replay on the v3 candidate tree found available L30/L42 hidden states and built
a convergence-hair dataset with `2592` candidate rows and `8400` pair rows. Hard merge was
not cleared: representative L30+L42 merge retained terminal oracle `0.9583` with survivor
reduction `0.0247`, and no non-diagnostic policy met the safe/useful hard-merge bar.
Hairs should remain soft diagnostics unless regenerated execution later confirms safety.

Reasoning still needs terminal confidence/defer. Science is branch-generation weak
(`positive_oracle_rate = 0.0833`, terminal best reward `0.0500`) and needs a domain branch
recipe pass before headline steering. No steering, routing, compute-savings, fork/carry,
or runtime branch-classification claim was made.

Report: `docs/evaluator/bg_dualanchor_convergence_hairs_reasoning_science_v1.md`.

## DualAnchor science branch recipe and reasoning terminal defer v1 (2026-05-31)

Follow-up status: `DUALANCHOR_SCIENCE_RECIPE_REASONING_DEFER_STATUS = PRE_STEERING_READY_WITH_SCIENCE_DIAGNOSTIC`.
The locked v3 schedule, DualAnchor two-tap selector, mean-floor very-loose threshold,
budget 8, nonterminal L47, terminal confidence/defer, and soft-only convergence hairs are
preserved. Reasoning is headline-ready for Phase 2b with survivor-set handoff; science is
diagnostic-only because no calibration recipe improved over baseline and MMLU science
remains weak. No steering, routing change, hard hair merge, compute-savings claim, or
fork/carry claim was made.

Report: `docs/evaluator/bg_dualanchor_science_branch_recipe_reasoning_defer_v1.md`.
