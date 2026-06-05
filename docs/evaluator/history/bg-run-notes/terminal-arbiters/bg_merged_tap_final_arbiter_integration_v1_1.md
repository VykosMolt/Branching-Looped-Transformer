<!-- docs-consolidation-source-note -->
> Consolidation note (2026-05-31): this is a source run note. The current consolidated interpretation is in `terminal-selection-and-arbiters.md`. Exact pre-consolidation text is archived under `docs/evaluator/history/pre_docs_consolidation_2026-05-31/`.

# Merged Tap Final Arbiter Integration v1.1

MERGED_TAP_FINAL_ARBITER_INTEGRATION_V1_1_STATUS = DOMAIN_FALLBACK_USEFUL_BUT_REASONING_LIMITED
SELECTION_ONLY_PHASE2A_STATUS_AFTER_MERGED_TAP_INTEGRATION_V1_1 = NEEDS_REASONING_ARBITER

This run tested the complete follow-up probe set after merged-tap integration v1: grouped task-disjoint CV, domain oracle diagnostics, domain-gated/fallback rules, reasoning failures, and math fallback behavior. It used cached top4 survivor features only. No action steering, routing change, wrapper/local-agent code, Hunter-Seeker execution, or new generation was run.

## Main Result

- grouped-CV verdict: `DOMAIN_FALLBACK_USEFUL_REASONING_WEAK`
- domain diagnostic verdict: `DOMAIN_SPECIALIZATION_WEAK_SIGNAL`
- reasoning probe verdict: `REASONING_BLOCKER_REMAINS`
- math fallback verdict: `MATH_UNIVERSAL_FALLBACK_USEFUL`
- readiness: `NEEDS_REASONING_ARBITER`
- best readiness-eligible policy: `math_universal_reasoning_universal_else_merged`
- best readiness-eligible grouped-CV task macro: `0.7736`
- best diagnostic probe policy: `outer_domain_oracle_diagnostic`
- best diagnostic probe grouped-CV task macro: `0.8200`
- merged top1 grouped-CV task macro: `0.7581`

## Files

- grouped CV: `artifacts/reports/probes/bg_merged_tap_final_arbiter_integration_v1_1_2026-05-18/grouped_cv_eval.md`
- domain oracle diagnostic: `artifacts/reports/probes/bg_merged_tap_final_arbiter_integration_v1_1_2026-05-18/domain_oracle_diagnostic.md`
- reasoning failure probe: `artifacts/reports/probes/bg_merged_tap_final_arbiter_integration_v1_1_2026-05-18/reasoning_failure_probe.md`
- math fallback probe: `artifacts/reports/probes/bg_merged_tap_final_arbiter_integration_v1_1_2026-05-18/math_fallback_probe.md`
- readiness: `artifacts/reports/probes/bg_merged_tap_final_arbiter_integration_v1_1_2026-05-18/selection_readiness.md`
- summary: `artifacts/reports/probes/bg_merged_tap_final_arbiter_integration_v1_1_2026-05-18/summary.md`
