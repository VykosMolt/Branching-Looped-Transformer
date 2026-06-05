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
