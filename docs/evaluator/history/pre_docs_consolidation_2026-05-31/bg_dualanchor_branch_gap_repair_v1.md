# DualAnchor Branch-Gap Repair v1

BG_DUALANCHOR_BRANCH_GAP_REPAIR_VERDICT = DUALANCHOR_BRANCH_GAP_REDUCED
BG_LAYER_NATIVE_TWO_TAP_READINESS_VERDICT = DOMAIN_READY_BRANCH_GAP

`DualAnchor` is the branch-valid `MIX_CODE_REASONING` + `MIX_OBJECTIVE_ALL` tap pair evaluated as layer-native heads at `24_L4`, `36_L4`, and `47_L4`.

## Result

- selected bundle: `bundle::two_tap_equal::sparse_old_plus_branch_30_70_top0p1::AntisymLinear::24_36_47`
- best-candidate domain ok: `True`
- best-candidate branch ok: `False`
- selected-bundle domain ok: `False`
- selected-bundle branch ok: `False`
- status: `DUALANCHOR_BRANCH_GAP_REDUCED`

## Interpretation

This pass targeted the remaining BGV1, hidden-origin-v3/salvage, and universal-bridge branch gaps while preserving old-domain behavior. It trained only copied tap vectors under a new artifact path.

Report: `artifacts/reports/probes/bg_dualanchor_branch_gap_repair_v1_2026-05-30/dualanchor_branch_gap_repair.md`.
