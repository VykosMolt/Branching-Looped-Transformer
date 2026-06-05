# BG Two-Tap HH-RLHF Comparison v1

This probe compares the old frozen BG taps and the strict layer-native two-tap candidates on fresh Anthropic HH-RLHF chosen/rejected pairs.

BG_TWO_TAP_HH_RLHF_COMPARISON_VERDICT = TWO_TAP_MATCHES_OR_BEATS_ALL_OLD_BG_ON_HH_RLHF

- pairs: `512`
- best old BG accuracy: `0.59765625`
- best layer-native old BG accuracy: `0.59765625`
- best new two-tap accuracy: `0.615234375`
- delta: `0.017578125`
- best old BG: `source::old_mixed_domain::MIX_OBJECTIVE_ALL::36_L4::AntisymLinear::62::row=62`
- best layer-native old BG: `source::old_mixed_domain::MIX_OBJECTIVE_ALL::36_L4::AntisymLinear::62::row=62`
- best new two-tap: `trained_layer_native::MIX_CODE_REASONING::adaptive_branch_from_val::layer_native_sparse__MIX_CODE_REASONING__sparse_old_plus_bridge_50_50_top0p2__24_L4__AntisymLine::24_L4::AntisymLinearNoNorm`

No Ouro weights, tokenizer files, checkpoints, old tap registries, production routing, wrapper/local-agent code, Hunter-Seeker modules, or steering modules were modified or executed.

Artifacts:

- `artifacts/reports/probes/bg_two_tap_hh_rlhf_comparison_v1_2026-05-30/hh_rlhf_comparison_summary.md`
- `artifacts/reports/probes/bg_two_tap_hh_rlhf_comparison_v1_2026-05-30/hh_rlhf_features.pt`
- `artifacts/reports/probes/bg_two_tap_hh_rlhf_comparison_v1_2026-05-30/two_tap_hh_rlhf_comparison_v1.pt`

