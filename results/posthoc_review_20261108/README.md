# Post-hoc checks for OPI v4 after a full review (2026-11-08)

These checks were decided after the results, in response to a full read of the manuscript. None was preregistered. Each one first reproduces the published numbers exactly with the original code, as a gate, and only then computes the corrected values. Every subfolder has its own README, script, results and SHA256SUMS.

- **`T1_fair_ties/`** — top-1 accuracy with credit split among candidates tied at the top score. The original code credits the first-listed candidate, which in the tied groups is the correct one. Covers every public prediction file: Ouro-RLTT, Ouro-2.6B, Ouro-2.6B-Thinking, Ouro-1.4B and Huginn.
- **`T2_tracking_fix/`** — the tracking statistic of `fixed_prefix_rerun_20260927/posthoc_20260927/controls.py`, part 3. The original summed two NumPy bools, which acts as a logical OR. The published value was 0.332; with the fix it is 0.544 [0.474, 0.613], at chance.
- **`T3_fair_ties_archive/`** — the same tie rule applied to the series whose per-group rows were kept offline: the moves to earlier loops, the shuffled-label controls, the length baselines, the transfer between checkpoints, and the MiniCPM comparison.

`CHECKSUMS.sha256` lists every file.
