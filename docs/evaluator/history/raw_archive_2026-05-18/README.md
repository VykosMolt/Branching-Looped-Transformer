# Evaluator Docs

Evaluator and domain-transfer notes live here.

- `pairwise_evaluator_locus_memo_v2_2026-05-11.md`: current evaluator/domain
  transfer memo, including layer taps, branch selection, loop geometry, and
  cross-domain probe results.
- `post_v10_synthesis_2026-05-14_v2.md`: locked pre-probe Experiment 2 and
  Ouro-RLTT-BG architecture spec.
- `post_v10_synthesis_2026-05-14_v3_after_layer_geometry.md`: current
  synthesis after the layer-24/36 blocker resolved; updates BG Phase 1 to
  heterogeneous tap interfaces.
- `bg_tap_interface_revision_2026-05-15.md`: active addendum after the
  layer-24/36 probe. Supersedes the uniform `(L1, L4)` tap-interface
  assumption in the locked post-v10 synthesis.
- `math_bg_gate_pilot_2026-05-15.md`: corrected math generated-branch BG-gate
  pilot. Records the 24/36 trained-head result, the small/eval caveat, and the
  budget-aware next-gate requirements.
- `evaluator_domain_transfer_notes.md`: short result note for the blocking
  layer-24/36, Thinking-vs-RLTT, and converged-tap old-head probes.

Source lives in `../../src/evaluator_core/`; probes and post-RLTT bundles live
under `../../utilities/evaluator/`.
