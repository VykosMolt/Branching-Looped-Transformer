# D0: AURC and AUACC for the published abstention comparison (promise to Denis Janiak)

Post hoc, 2026-10-10. `aurc_auacc.py` reads the preserved held-out predictions
(`results/horizon_power_v3_20260726/raw_predictions_heldout_v3.json`, public repo commit 6c99dab)
and first reproduces the published AUARC deltas (coverage 0.50-1.00) exactly with the original
`bg_v3_selective_prediction.py` (assert). It then computes Denis Janiak's AUACC (full coverage,
trapezoid over k/n) and AURC (= 1 - 1/n - AUACC) on the same rows, with the same paired
task-clustered bootstrap (2,000 rounds, seed 20260724).

Command: `PYTHONDONTWRITEBYTECODE=1 ~/ouro_project/shared/venv/bin/python aurc_auacc.py`
Output: `aurc_auacc_results.json`.

Caveat: these scores come from the final-answer-line readout of version 3, which is NOT
pre-answer (the reasoning before the cut often states the conclusion; paper v4 Section 3.2).
