# BG Two-Tap Fresh Dataset Comparison v1

This probe compares the old frozen BG taps and the new layer-native two-tap candidates on the same fresh domain candidate content.

BG_TWO_TAP_FRESH_DATA_COMPARISON_VERDICT = TWO_TAP_MATCHES_OR_BEATS_OLD_BG_ON_FRESH

No Ouro weights, tokenizer files, checkpoints, tap registries, production routing, wrapper/local-agent code, Hunter-Seeker modules, or steering modules were modified or executed.

## Dataset Policy

Previously used public sources were sampled again only because the prior local artifacts were subset samples rather than full-dataset passes. The runner used a non-42 seed and excluded prior task IDs where the earlier artifact exposed them. New public datasets were also added for the same domains.

Primary new two-tap candidates were restricted to the `MIX_CODE_REASONING` and `MIX_OBJECTIVE_ALL` anchors. Diagnostic old-content transplants from other old taps were excluded from the primary readiness comparison.

## Result

- tasks: `192`
- candidates: `742`
- domains: `{'reasoning': 48, 'science': 48, 'math_simple_arithmetic': 48, 'coding': 48}`
- best old BG overall accuracy: `0.88`
- best new two-tap overall accuracy: `0.8818181818181818`
- delta: `0.0018181818181818299`
- best old BG: `source::old_mixed_domain::MIX_CODE_SCIENCE::36_L4::AntisymLinearNoNorm::23::row=23`
- best new two-tap: `layer_native::MIX_OBJECTIVE_ALL::old_plus_bridge_50_50::36_L4::AntisymLinearNoNorm`

Targeted rehost candidates remain diagnostic and are not counted in primary readiness.

## Artifacts

- `artifacts/reports/probes/bg_two_tap_fresh_dataset_comparison_v1_2026-05-30/fresh_candidate_dataset.json`
- `artifacts/reports/probes/bg_two_tap_fresh_dataset_comparison_v1_2026-05-30/fresh_candidate_features.pt`
- `artifacts/reports/probes/bg_two_tap_fresh_dataset_comparison_v1_2026-05-30/fresh_comparison_summary.md`
- `artifacts/reports/probes/bg_two_tap_fresh_dataset_comparison_v1_2026-05-30/fresh_comparison_pair_rows.csv`
- `artifacts/reports/probes/bg_two_tap_fresh_dataset_comparison_v1_2026-05-30/two_tap_fresh_dataset_comparison_v1.pt`

