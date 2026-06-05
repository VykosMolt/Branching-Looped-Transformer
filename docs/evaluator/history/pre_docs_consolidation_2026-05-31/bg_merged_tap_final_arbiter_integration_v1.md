# Merged Tap Final Arbiter Integration v1

MERGED_TAP_FINAL_ARBITER_INTEGRATION_STATUS = MERGED_TOP1_USEFUL
SELECTION_ONLY_PHASE2A_STATUS_AFTER_MERGED_TAP_INTEGRATION = USE_MERGED_TAP_TOP1_AS_ARBITER_BUT_NOT_READY

This cached run integrated the selected merged weight tap as a final-arbiter expert among fixed-composite top4 survivors. It did not train Ouro, run action steering, change routing, or run new generation.

## Result

- selected policy: `old_code_reasoning_top1`
- fresh holdout task macro: `0.5610`
- merged tap top1: `0.7314`
- fixed-composite top1: `0.5610`
- majority-rank: `0.5353`
- oracle best survivor: `0.9583`
- readiness: `USE_MERGED_TAP_TOP1_AS_ARBITER_BUT_NOT_READY`

The merged tap remains useful as a final-arbiter signal, but this run does not clear Phase 2a readiness if the validation-selected integrated policy misses the 0.75 task-macro target or domain guardrails.

## Files

- integration eval: `artifacts/reports/probes/bg_merged_tap_final_arbiter_integration_v1_2026-05-18/integration_eval.md`
- readiness: `artifacts/reports/probes/bg_merged_tap_final_arbiter_integration_v1_2026-05-18/selection_readiness.md`
- summary: `artifacts/reports/probes/bg_merged_tap_final_arbiter_integration_v1_2026-05-18/summary.md`

## Merged tap final arbiter integration v1.1 (2026-05-18)

`MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_STATUS = DOMAIN_FALLBACK_USEFUL_BUT_REASONING_LIMITED`; `SELECTION_ONLY_PHASE2A_STATUS_AFTER_MERGED_TAP_INTEGRATION_V1_1 = NEEDS_REASONING_ARBITER`. Grouped task-disjoint CV best readiness-eligible policy was `math_universal_reasoning_universal_else_merged` with task macro `0.7736`; merged tap top1 was `0.7581`. This is final-arbiter integration only; no action steering or routing change was tested.

Report: `docs/evaluator/bg_merged_tap_final_arbiter_integration_v1_1.md`.
