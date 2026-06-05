# Steering And Adapters

Updated: 2026-05-31

This is the consolidated entry point for the steering, adapter, layer-hook, and frozen-backbone write-path investigations.

## Status

Frozen-backbone inference steering is closed under the tested safe-alpha methods.

Remembered verdicts:

- `BG_SEQUENCE_LEVEL_ADAPTER_VERDICT = NO_FROZEN_BACKBONE_WRITE_PATH`
- `FROZEN_BACKBONE_INFERENCE_STEERING_STATUS = CLOSED_UNDER_TESTED_METHODS`

Interpretation:

Hooks and perturbations can be applied and can propagate, but no tested static direction or frozen-backbone adapter produced reliable heldout free-generation control. This remains a boundary condition, not a production steering capability.

## What Was Ruled Out

The tested frozen-backbone routes did not establish:

- reliable action steering,
- a production write path,
- a trained steering corridor,
- true autoregressive branch-specific KV/cache fork-carry,
- compute savings.

## Relationship To Current Phase 2a

The DualAnchor architecture-looped v3 result is selection/branching only.

No steering was tested there. The readiness verdict `READY_WITH_TERMINAL_DEFER` means the selection baseline can be used as a candidate locked baseline for later steering comparison, not that steering already exists.

## Source Notes Moved To History

Detailed source docs were moved under `history/steering-and-adapters/`.

Exact pre-consolidation root copies are also preserved under:

`history/pre_docs_consolidation_2026-05-31/`

