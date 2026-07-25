# PAUSED 2026-07-20 ~18:25 CEST (user request — resume at night)

Elapsed at pause: ~1h05m of the 8h budget (`logs/start_time`, `logs/pause_time`).

Extraction checkpoint: 370/2150 groups (1,763/8,592 candidates) durable across 3 shards;
manifest-based resume, ~38 min GPU remaining at ~3.0 cand/s.

## Resume sequence (all from repo root, scripts in utilities/tests/manual/)

1. `venv/bin/python -u utilities/tests/manual/bg_xloop_early_v1_extract.py >> artifacts/reports/cross_loop_early_layer_taps_20260720/logs/extract.log 2>&1`
   (idempotent: subset/integrity/cached-ref steps are skipped, extraction resumes from manifest)
2. `bg_xloop_early_v1_train_eval.py` — refit matrix + frozen transplants + shuffle controls
3. `bg_xloop_early_v1_controls.py` — feature-match + locked-policy reproduction (final pass)
4. `bg_xloop_early_v1_stats.py` — 10k task-clustered bootstrap + verdict labels
5. `bg_xloop_early_v1_plots.py` — 4 figures (SVG+PNG)
6. `bg_xloop_early_v1_tests.py` — targeted test suite
7. optional Tier-3: `bg_xloop_early_v1_s3b2.py` (secondary readout, ~1 min GPU)
8. write RESULTS.md, PAPER1_INTEGRATION_NOTE.md, RUN_MANIFEST.json, SHA256SUMS, final report

Already validated: scoring equivalence vs production (exact), extraction path
EXTRACTION_PATH_VALIDATED on partial data (cos 1.000000 at 24/36/47; locked v2 policy
identical on cached vs fresh coding-heldout features). See EXPERIMENT_PLAN.md for the
preregistered design and ARTIFACT_AUDIT.md for the audit trail.
