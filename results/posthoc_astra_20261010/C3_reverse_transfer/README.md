# C3: backward transfer of cross-loop ranking probes (post hoc)

**This analysis is post hoc.** It was designed on 2026-10-10, after the original results were known, in response to correction 1 of the external audit (`~/Documents/Research/ouro/Operational-Proto-Introspection/audit-v4-20261008/REPORT.md`). It was not pre-registered. The 2026-07-20 run was only read; nothing in it, or on the external SSD, was modified.

The question is whether loop-1 ranking probes do badly at loop 4 because loop 1 uses different coordinates or because the loop-1 signal is weak. The test applies probes trained at later loops, unchanged, at earlier loops, and compares them with the probe trained at the target loop.

## Source
- Run: `/run/media/moloch/ARCH_BACKUP/OURO_COLD_ARCHIVE_20260729/historical_experiments/paper_v3_audit_evidence/artifacts/reports/cross_loop_early_layer_taps_20260720/`. The features were read in place, so no copy was made on the laptop.
- Code: the original modules `bg_xloop_early_v1_{common,train_eval,stats}.py` in `/home/moloch/ouro_project/shared/utilities/tests/manual/` are imported unchanged. Their path constants are pointed at the archived run, and only loader functions are called, never the writers.
- Environment: torch 2.12.0.dev20260407+cu128 (the same version as the original run), CPU only.

## Commands
```
cd /home/moloch/ouro_project/opi/posthoc_astra_20261010/C3_reverse_transfer
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" \
  /home/moloch/ouro_project/shared/venv/bin/python -u reverse_transfer.py 2>&1 | tee reverse_transfer.log
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" \
  /home/moloch/ouro_project/shared/venv/bin/python -u thread_check.py 2>&1 | tee thread_check.log
```

## Files
- `reverse_transfer.py`: the full analysis, covering stage 0 (checksums), gate 1a (refit, reported), gate 1b (saved weights, blocking), stage 2 (backward transfers with the paired bootstrap) and stage 3 (4x4 matrices and cosines).
- `reverse_transfer_results.json`: every number. Keys: `stage0`, `gate_1a_refit`, `gate_1b_saved_weights`, `transfers` (forward and backward, each with `paired_delta_vs_target_refit` and `paired_delta_vs_source_self`), `matrix_top1`, `cosine`, `weight_norm`, `refit_weight_sensitivity_backward_top1`, `bootstrap_reproduces_saved_forward_deltas`.
- `reverse_transfer.log`: the log of the final run.
- `thread_check.py` and `thread_check.log`: test whether a refit reproduces the saved weights bit for bit, as a function of the BLAS thread count.
- `gate_attempt1_8threads.{log,_results.json}` and `gate_attempt2_4threads.{log,_results.json}`: the two earlier runs, kept unedited. Both used an earlier version of `reverse_transfer.py` in which the refit gate was blocking. Both failed that gate, as described below, and stopped before stage 2.

## Gate history
1. **Run with 8 threads.** The selected settings, all 36 grid validation scores, held-out macro top-1 and every forward-transfer top-1, Spearman and sign-agreement value match the saved record. The weights differ from the saved ones at float precision (relative difference about 4e-7). As a result, the 6-decimal scores in all 12 refit files and all 18 transfer files differ.
2. **Thread diagnosis** (`thread_check.log`). With 1 to 4 threads the refit is bitwise identical for cells trained on `code_reasoning`. Cell 8_L3 is the only selected cell trained on the larger unbalanced `all_core` set, and it differs by at most 3.2e-8 at every thread count up to 8. Larger thread counts were not tried, to keep the laptop quiet.
3. **Run with 4 threads.** 11 of 12 refits are bitwise exact. 8_L3 and the transfer 8:L3->L4 are not, although their per-group top-1, macro top-1 and agreement statistics are identical. **Strictly, the refit gate is not exact.**
4. **Saved-weight gate.** The original selected weights are stored in the run's `xloop_taps.pt`. They match the saved weight hashes and reproduce all 12 `refit_*` files and all 18 forward `transfer_*` files exactly, including all saved agreement statistics. Stage 2 uses these weights. The original paired bootstrap, re-run with the backward series added, reproduces all 18 saved forward deltas (`bootstrap_stats.json`) exactly. As a sensitivity check, all 18 backward top-1 values computed with the 4-thread refit weights are identical to those from the saved weights.

## Definitions
- Bootstrap: clustered by task_uid within domain, unweighted macro mean over domains, 10,000 draws, seed 20260720, with the same draws used for every series. This is `bg_xloop_early_v1_stats.bootstrap`, called unchanged.
- Spearman and sign agreement: `bg_xloop_early_v1_train_eval.score_agreement`, comparing the transferred probe's held-out scores with the scores of the probe trained at the target loop, on the same candidates. This is the computation used for the forward moves.
- Cosine: cosine between the raw weight vectors of the selected probes, computed in the unnormalized feature space.
