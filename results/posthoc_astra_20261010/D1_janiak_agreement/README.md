# D1: pre-answer readout vs completed-sibling agreement (for Denis Janiak)

**Post hoc**, 2026-10-10. Plan: `PLAN.md`, hashed in `PLAN.sha256` (`ec6b8e55…`, 11:58) before any result was computed; `analysis.py` refuses to run if the plan changed. Promised to Denis Janiak in the e-mail of 2026-09-12. Results: `SUMMARY_FOR_DENIS.md` (tables plus a plain reading) and `results.json` (everything).

## Files
- `PLAN.md`, `PLAN.sha256`: the analysis plan, fixed before results.
- `janiak_vendor.py`: Denis's functions copied verbatim from reasoning-geometry commit 26c48be (MIT; `LICENSE_reasoning-geometry`). `test_vendor.py` checks that every copied function's source matches the original, and that the restated per-question aggregation matches his `aggregate_prompt_features`, by importing his code.
- `extract.py`: reads the shards (read only, one at a time, the sealed loaders' asserts) into `cache/` (deleted at the end).
- `analysis.py`: probe, T1, T2, Orgad controls, fair comparison; writes `results.json`, `probe_scores_<run>.npz` (per-answer OOF probe score, label, fold) and `SUMMARY_FOR_DENIS.md` (tables generated; the plain reading comes from `reading.md`).
- `entropy_pass.py`: GPU pass for token entropy, **run 2026-10-10 15:30–16:53** on the RTX 5070 Ti Laptop (user-approved). It took 82.8 min at 1,239 tok/s for 6.35M tokens (15,052 answers), at batches of 4,096 tokens with no OOM. `test_entropy_logic.py`: CPU check of its token alignment with a toy model.
- `entropy/entropy_ouro_rltt.npz` (3.4 MB, sha256 73fa6ec3…): per (run, question id, candidate index), the whole-generation and window entropy, plus `lp_check_mad`. Median 0.0097, 99th percentile 0.033, maximum 0.125: bf16 noise, no misalignment. `entropy/parts/` holds the per-shard parts (2.6 MB).
- **Results with entropy (current):** `results.json`, `SUMMARY_FOR_DENIS.md`, `reading.md`, `analysis_with_entropy.log`. **Without entropy:** `results_no_entropy.json`, `SUMMARY_FOR_DENIS_no_entropy.md`, `reading_no_entropy.md`, `analysis_no_entropy.log` (= `analysis.log`). `compare_entropy.py` prints the side-by-side table (saved as `logs/entropy_comparison.md`). PLAN.md is unchanged: entropy entered the baselines exactly as its section 4 says. The probe scores are identical in both versions, because the probe uses hidden states only.
- `analysis.log`: log of the first (no-entropy) run. `logs/`: extraction logs (`extract_r6_r1.log` ends in the first Run 1 attempt's KeyError, a task-file lookup bug fixed before `extract_r1.log`) and a trial Run 4 analysis whose numbers are identical to the final run.
- `cache/` (1.2 GB: metadata plus pooled features at N*) was deleted at the end; `extract.py` rebuilds it.
- `reading.md`: the plain-reading paragraph, written after the results and included in the summary.

## Commands
CPU only, `OMP_NUM_THREADS=8` (also MKL/OPENBLAS), `CUDA_VISIBLE_DEVICES=""`, Python `/home/moloch/ouro_project/shared/venv/bin/python -B`, from this directory:
```
python -B test_vendor.py                 # vendored == Denis's code (needs his checkout; path is the default argument)
python -B extract.py run4                # ~2 min each; run6 reads 8.7 GB
python -B extract.py run6
python -B extract.py run1
python -B analysis.py run4 run6 run1     # ~45 min at load average ~59 on 24 cores (Run 4 alone: 4 min unloaded)
python -B test_entropy_logic.py          # CPU, seconds
```
Entropy, later, on the GPU (resumable; prints tokens/s after each shard):
```
python -B entropy_pass.py --limit-shards 1 --runs run4    # smoke test: one shard (~200k tokens)
python -B entropy_pass.py                                 # all runs; writes entropy/entropy_ouro_rltt.npz
python -B extract.py run4; python -B extract.py run6; python -B extract.py run1   # cache was deleted
python -B analysis.py run4 run6 run1                      # picks entropy up automatically, labels it
```
`entropy_pass.py` needs the shards on the SSD and the weights in `shared/models/ouro_rltt_local` (sha256-checked as `cloud/setup_pod.sh` does). Before re-running the analysis with entropy, keep a copy of the entropy-free `results.json` and `SUMMARY_FOR_DENIS.md`; the re-run overwrites them.

## Inputs (read only)
- Shards: `/run/media/moloch/ARCH_BACKUP/OURO_COLD_ARCHIVE_20260927/tree/opi/preanswer/{horizon_cohort2_20261003/run4_shard_*, gsm8k_train_20261003/run6_shard_*, fixed_prefix_rerun_20260927/run1_shard_off*}.pt`.
- Labels and detectors: `horizon_cohort2_20261003/{gen_run4.strict_answer, answer_detector}` (Run 1's sealed detector is byte-identical, checked in `extract.py`); `gsm8k_fixed_prefix_20260927/answer_detector_gsm8k`.
- Estimator: `horizon_cohort2_20261003/posthoc_audit_20261004/fastarm.py` (the fixed combined estimator, `analyze_run3._fit_score` with PCA via the Gram matrix); `analyze_run3.within_task_auroc`.
- Cohort check: the T1 rows reproduce the sealed primary cohorts exactly on the held-out split (Run 4: 735 answers, 59 incorrect; Run 6: 1,907 answers, 85 incorrect).
