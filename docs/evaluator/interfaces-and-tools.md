# Interfaces And Tools

Updated: 2026-05-31

This is the consolidated entry point for tap interfaces, transformer integration notes, local tooling notes, and low-level evaluator mechanics.

## Tap Interface

The evaluator taps are relational pairwise heads over hidden-state features. Treat them as pairwise evaluators, not absolute scorers.

Labels must come from external sources:

- HH chosen/rejected labels,
- answer keys,
- exact-answer verifiers,
- unit tests,
- parser/verifier labels.

Tap scores are features or decisions, not labels.

## Current Layer/Loop Context

The active DualAnchor architecture uses taps at:

- layer 24,
- layer 36,
- layer 47,
- loops L1-L4.

Terminal collapse is only considered at final `L4_47`, and even there only through confidence gating.

## Cumulative-Hook Boundary

Prompt-only decoder-layer carry equivalence was validated for cumulative-hook replay at layers 24/36/47.

This does not prove:

- autoregressive branch-specific KV/cache fork-carry,
- production Hunter-Seeker execution,
- compute savings.

## Source Notes Moved To History

Detailed source docs were moved under `history/interfaces-and-tools/`.

Exact pre-consolidation root copies are also preserved under:

`history/pre_docs_consolidation_2026-05-31/`

