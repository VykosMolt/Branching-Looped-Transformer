# Branching Looped Transformer

Branching Looped Transformer is a research repository for evaluator and
branch-selection work on Princeton's Ouro-RLTT looped transformer. It studies
whether candidate branches can be ranked from frozen loop-state activations, and
uses that signal to build pairwise evaluators, tap heads, and branch-prune
selection baselines.

This is not a full runtime or tool-using agent stack. The repository focuses on
read-only model-state capture, relational scoring, selection policies, and the
experiments that support those claims.

## What Is Here

- `evaluator/`: read-only BG controller code, Ouro-RLTT feature capture,
  hidden-branch utilities, and adapter/intervention experiments.
- `evaluator_core/`: the pairwise evaluator architecture and frozen evaluator
  anchor utilities.
- `probes/`: opt-in measurement scripts for loop-state geometry, pairwise
  scoring, branch selection, and domain-transfer diagnostics.
- `docs/evaluator/`: consolidated evaluator documentation.
- `docs/evaluator/history/`: source run notes and historical archives.
- `LICENSE`: Apache License 2.0.

## Current Baseline

The current documentation centers on a frozen Ouro-RLTT branch-selection stack:

- feature capture from layers 24, 36, and 47 across four loop iterations;
- pairwise relational scoring over candidate branch states;
- a DualAnchor selection baseline using `MIX_CODE_REASONING` and
  `MIX_OBJECTIVE_ALL`;
- architecture-looped branch/prune survival with terminal confidence-gated or
  deferred handoff;
- no claimed production routing change or model-weight change.

For the detailed state, start with:

- `docs/evaluator/README.md`
- `docs/evaluator/evaluator-navigation-map.md`
- `docs/evaluator/current-state.md`
- `docs/evaluator/chronological-evaluator-summary.md`

## Local Inputs

Running the probes expects local model/checkpoint paths such as
`models/ouro_rltt_local` and
`artifacts/checkpoints/evaluator/pairwise_epoch2.pt`. Model weights, evaluator
checkpoints, generated reports, and large run outputs are not stored on GitHub.

## License

Apache License 2.0. See `LICENSE`.
