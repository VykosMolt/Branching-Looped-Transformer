# Docs Consolidation Manifest

Date: 2026-05-31

Purpose: make the evaluator docs readable without losing prior reports.

## Data Preservation

A preservation snapshot of root Markdown files in `docs/evaluator/` was written to:

`docs/evaluator/history/pre_docs_consolidation_2026-05-31/`

Checksum file:

`docs/evaluator/history/pre_docs_consolidation_2026-05-31/manifest.sha256`

No report artifact under `artifacts/reports/probes/` was removed.

## New Memorable Entry Points

- `evaluator-navigation-map.md`
- `dualanchor-architecture-baseline.md`
- `dualanchor-tap-evolution.md`
- `branch-generation-and-survival.md`
- `terminal-selection-and-arbiters.md`

## Updated Entry Points

- `README.md`
- `current-state.md`
- `domain-transfer-ledger.md`
- `chronological-evaluator-summary.md`
- `post_v10_synthesis_2026-05-18_v8.1_routing_locked.md`

## Collapse Policy

The old `bg_*` files remain source run notes sorted under `history/bg-run-notes/`. They are not the main reading path.

The root docs are now organized as:

1. Navigation/current state.
2. Consolidated memorable docs.
3. Source run notes.
4. Historical archives.

Future docs should prefer memorable names for human-facing summaries, and reserve `bg_*` names for exact run notes.
