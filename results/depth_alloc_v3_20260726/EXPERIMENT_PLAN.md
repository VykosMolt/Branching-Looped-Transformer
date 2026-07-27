# Tap-gated loop allocation — pre-registered plan (SEALED)

Sealed 2026-07-26 before any full-slice generation. Script sha256 at seal: 2fd933a8747baca7b0a0b4a35f2485b801ad297177607e228a7bc8bf776cfb46
(bg_v3_depth_alloc.py; the module docstring carries the full design and the
fixed verdict labels).

Grounding (read before design): RLTT paper (arXiv 2602.10520) — trained at
fixed 4 loops from Ouro-2.6B-Thinking, adaptive halting explicitly sacrificed
and named as future work; Ouro paper (arXiv 2510.25741) — per-token exit gate
sigma(Linear(h)) with survival-chain pdf, all released checkpoints carry
trained gates, RL post-training documented to break gate calibration.

Key design choices, sealed:
- Task slice [680, 860) of the sealed hash-ordered pool; sealed split_for.
- Fixed-depth generation TABLE (d=1..4, k=2, max_new=448, T=0.7, per-(task,
  depth) seeds); all policies are row-selectors over the same table, so
  policy comparisons are exact-compute-matched by construction.
- Tap allocator: ridge on loop-1 prompt features (layers 8/16/24 concat),
  labels = minimal sufficient depth, trained on train+val tasks only.
- Native-gate policy: survival-chain exit pdf from the model's own gate
  logits, mean-pooled over prompt tokens, expected-step score, thresholded.
- Random-histogram control gates any positive claim (sealed in docstring).
- Heldout opened once by the analyze stage.
- Models: thinking26 (well-calibrated gate; hardest baseline) and rltt
  (gate retained but post-RL; the interesting arm). Interpretation commitments
  as per verdict labels in the script; an imprecise result is reported as such.
