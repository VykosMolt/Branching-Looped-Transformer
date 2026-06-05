<!-- docs-consolidation-source-note -->
> Consolidation note (2026-05-31): this is a source run note. The current consolidated interpretation is in `dualanchor-tap-evolution.md`. Exact pre-consolidation text is archived under `docs/evaluator/history/pre_docs_consolidation_2026-05-31/`.

# Layer-Native Two-Tap Constrained Training v1

BG_LAYER_NATIVE_TWO_TAP_CONSTRAINED_TRAINING_VERDICT = CONSTRAINED_TWO_TAP_PARTIAL
BG_LAYER_NATIVE_TWO_TAP_READINESS_VERDICT = DOMAIN_READY_BRANCH_GAP

Copied `MIX_CODE_REASONING` and `MIX_OBJECTIVE_ALL` layer-local heads were trained on actual train-split old-domain and branch/bridge labels with old-anchor penalties. No Ouro weights or old registries were changed.

## Result

- domain ok: `True`
- branch ok: `False`
- status: `CONSTRAINED_TWO_TAP_PARTIAL`

Report: `artifacts/reports/probes/bg_layer_native_two_tap_constrained_train_v1_2026-05-30/constrained_train_eval.md`.
