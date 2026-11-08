# T3: fair tie handling for the archived (non-public) ranking series

Post hoc reanalysis, 2026-11-08, for OPI v4. Not part of any preregistered protocol. It continues
T1 (`../T1_fair_ties/`), which re-scored the public prediction files. T3 covers the series whose
per-group rows were never public. Their inputs are on the external archive drive, which was mounted
read-only (`ro` in `findmnt`) and was only read.

## Rule

T1's rule, imported from T1's `fair_ties.py` (`fair_top1`, unchanged):

    top1_fair = #(candidates tied at the group's max score with reward >= the group's max reward)
                / #(candidates tied at the max)

Ties are exact equality of the scores the original rule ranked. For the xloop series these are the
6-decimal rounded scores in the prediction rows, as in T1. For the length baseline they are integer
lengths. For task 5 they are the unrounded float scores, because nothing was saved there; a 6-dp
variant and a MiniCPM-specific sensitivity are also reported (see RESULTS.md).

## Command

    cd /home/moloch/ouro_project/opi/posthoc_review_20261108/T3_fair_ties_archive
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" \
      PYTHONDONTWRITEBYTECODE=1 /home/moloch/ouro_project/shared/venv/bin/python -I -u t3_fair_ties_archive.py

Environment: Python 3.14.9, torch 2.12.0.dev20260407+cu128 (the original runs' version), numpy 2.4.4.
CPU only, 4 threads (`torch.set_num_threads(4)` in the script). The run takes about 2 minutes. The
CoreContent-v2 features are loaded as float32, which is about 10 GB of RAM (estimated, not measured).

Code comes from git commit `b8c95e31198315fae4dc574376009d70265a1695` via `git archive` into a
temporary directory, so the working tree is never read. The xloop and family modules are taken from
`probes/` and `tools/paper1_v3/`. The task-5 modules are extracted in their original tree layout
(`<tmp>/shared/utilities/tests/manual`, `<tmp>/shared/tools/paper_verification`, `<tmp>/src/evaluator`).
They are imported unmodified; only their path constants are pointed at the archive, and every writer
is redirected to memory. The script writes only `results.json`, `RESULTS.md` and `SHA256SUMS` in
this folder.

## Gates (each must reproduce the published numbers with the original first-max rule)

All seven gates pass, with 0 mismatches each.

1. **C3 reverse transfer.** The saved `xloop_taps.pt` weights match the saved `weight_sha`. With them,
   the original `eval_cell` regenerates all 12 `refit_*` and 18 forward `transfer_*` files exactly
   (canonical JSON, every row), and the forward agreement statistics equal `train_eval_results.json`.
   The original `bootstrap` (seed 20260720, 10,000 task-clustered draws) reproduces all 18 forward
   deltas in `bootstrap_stats.json`. All 36 transfer entries of C3's `reverse_transfer_results.json`
   are reproduced field by field, including the 18 backward moves: top-1, by-domain values, Pearson,
   Spearman, sign agreement and group rank correlation, both paired deltas, the CIs and p>0.
   C3 itself ran with 4 threads; C3's thread note concerns refits only, and this task uses the saved weights.
2. **Shuffled-label controls.** The shuffled weights were never saved, so they are refit through the
   original code path: `train_eval.shuffle_labels` on train then val with
   `torch.Generator().manual_seed(20260720)`, then `fit_cell_grid` (36-point grid, val-only
   selection) and `eval_cell` on the heldout set. Checked: `heldout_macro_top1`, `by_domain` and
   `chance_macro` (4 dp, as stored) equal `train_eval_results.json["shuffle_controls"]`, and the
   printed selection line (best val, tset, lr, seed) and heldout line appear verbatim in
   `logs/train_eval.log`. The gate can only be as precise as that record. Thread check: at 8 threads
   the shuffled weights differ (different `weight_sha`), but every original and fair macro value is
   identical and the gate still passes.
3. **Length baseline.** No code for `length_baseline.json` exists in the commit or in the run, so the
   measure was found by reproduction. Score = +len or -len of `candidate_text` from
   `subset_groups.jsonl`, top-1 by the original first-max `group_selection_metrics`, domain means over
   the 600 heldout groups, unweighted macro. Characters, UTF-8 bytes and stripped characters all
   reproduce both macros and all 10 per-domain values exactly, and all three give identical fair
   values. The last-max rule does not reproduce them: it gives 0.2200 and 0.1767.
4. **Cross-checkpoint transfer.** The original `bg_v3_family_xloop.stage_crossmodel` (sealed RLTT
   `xloop_taps.pt` on each checkpoint's heldout features; `write_json` captured in memory;
   `eval_cell` wrapped only to keep its rows) reproduces `crossmodel_rltt_transfer.json` for base26
   and thinking26 in every cell and field. The archive copy is byte-identical to the commit copy.
5. **MiniCPM 0.568.** Stored features from the 2026-07-29 archive (the 2026-09-27 copy has only the
   manifest) plus the saved channels in `minicpm2_sft_corecontent_v2_task_disjoint.pt`, scored with
   the original `evaluate()` and the split lines copied verbatim from `train_corecontent`
   (train-clean). This reproduces `split_audit`, `blockwise_val` and `blockwise_heldout` exactly
   (unrounded floats).
   **Ouro 0.631.** CoreContent-v2 features plus `ouro_corecontent_task_disjoint_policy.pt`, with the
   original `assigned_split` and `eval_channels_macro`, reproduce `split_audit`, group counts and the
   validation and heldout per-domain and macro values exactly (unrounded floats).

Planted-bug check (run once on a scratch copy, not part of the script): with bootstrap seed +1 the
task-1 gate fails (98 mismatches). With shuffle seed +1, the task-2 gate fails (32). Swapping to
last-max fails the task-3 gate (9). Pointing the sealed root at the checkpoint's own taps fails
task 4 (54 and 52). The fair refit values for the 12 RLTT target cells in task 1 equal T1's
independent values, orig and fair, for all 18 moves.

Reproducibility: two runs from scratch with 4 threads gave byte-identical `results.json` (SHA-256 in
`SHA256SUMS`).

## Inputs and SHA-256

`ARCH` = `/run/media/moloch/ARCH_BACKUP`. Every file read is listed with its SHA-256 in
`results.json["inputs_sha256"]` (269 files: 182 feature shards, 60 prediction files, 27 others). Every commit file read or imported is listed in
`results.json["commit_files_sha256"]`. The non-shard inputs:

    f561184ce64a0d46d4a2c3a894b7fa9511192d03e8cb596c79333a4f5ff04bad  /home/moloch/ouro_project/opi/posthoc_review_20261108/T1_fair_ties/fair_ties.py
    RLTT run: ARCH/OURO_COLD_ARCHIVE_20260729/historical_experiments/paper_v3_audit_evidence/artifacts/reports/cross_loop_early_layer_taps_20260720/
    f99c75c3b15448b67157fc212e94f5ac42229e4cf3375147726ef45643ba520a  bootstrap_stats.json
    7288502e90e62cddc26314fb1d03690adbfb8c34404644fa9a30b61a19e1fca6  features/feature_manifest.json
    ccc805adc78b57c775319be9525b414851ef268b9205889dda6810a62aac15c8  length_baseline.json
    07f8063846b10f7b6376c4297126f74517fd2bb43fa2e04ca293f41aeb8b4e05  logs/train_eval.log
    b3715594b8f106f0c3b1da1df014a6ee202fa0c4a36fdd4be80e06fa3a0feef5  subset_groups.jsonl
    7a391c824758438e22d496ee7ac212d9b0c8b1c49ebd4af360206f82bd2b4545  train_eval_results.json
    f50b5c0da8ead0a998d8a6b499a54fccd36d9fa05da17d0392d0fa2be7f5fd29  xloop_taps.pt
    (+ predictions/refit_*.json x14 and transfer_*.json x18, in results.json)
    C3: ARCH/OURO_COLD_ARCHIVE_20260927/tree/opi/posthoc_astra_20261010/C3_reverse_transfer/
    3e83384e13ce3a64119167692c321350e8068361317e73458191f7c460f358b6  reverse_transfer_results.json
    f4ca284e74c99acdabd073ec865791c57d6ba642e414b4c96c56dcbd4b8b9b8e  reverse_transfer.py (read, not imported)
    10a7916732ad25fd32e5034c9a8a60092e6e08a382bf35881782103d81cdba83  README.md
    Family: ARCH/OURO_COLD_ARCHIVE_20260927/tree/opi/localization/family_xloop_v3_20260726/
    b86a39b864c1c01c967c7158d58b2e90a930641642fdf0e71cbd12a21faafcb6  base26/crossmodel_rltt_transfer.json
    00a9927d7d4c4311f81aa34010bed3afd1a081017b151d16585e7d9e115b721e  base26/features/feature_manifest.json
    bade77bb7511236fc7dc5455b81071ebe260dbbaf5889b49a544802da9ae3108  base26/model_plan.json
    d92fa1c877cd758164e59d87a49017d0964749cd7be2b706505382b8badd3cae  base26/train_eval_results.json
    7f1a37251bac56af883c409569a8b503afb99e3c0473f9eb28aae03c924cc65c  thinking26/crossmodel_rltt_transfer.json
    404e6ce08c6c9ef94a44b537193d321a864479dacdb236b9ff9c24ffc4fa0e4a  thinking26/features/feature_manifest.json
    d8548c9f23869288bc6a9bb26ce21e6e51b80020b5e48fa053caee08bd5f1def  thinking26/model_plan.json
    5d22201649298e241fe775337e6112ccfe2325f1da795b457e165a6cfb1c9efa  thinking26/train_eval_results.json
    (+ {base26,thinking26}/predictions/refit_*.json x14 each, in results.json)
    MiniCPM: ARCH/OURO_COLD_ARCHIVE_20260729/historical_experiments/paper_v3_audit_evidence/artifacts/reports/paper_verification/nonlooped_architecture_control_20260710_235252/
    19a95e0f6bcdc1ad5f07e2a4f9e24a54c40b7d324a393bc250b46e75da2e7e8a  corecontent_control_results_task_disjoint.json (same hash in the 20260927 copy and the commit)
    ab5f8b92515f8426876709c66a754f45f66e95f45798e2ba676afe82328197cf  corecontent_features/feature_manifest.json (same hash in the 20260927 copy)
    dc134f0ce07e611ce1475838f5f97688d34c27249ecddd4591c55da959bce919  minicpm2_sft_corecontent_v2_task_disjoint.pt
    Ouro task-disjoint: ARCH/OURO_COLD_ARCHIVE_20260927/tree/
    093f37cfa77a09ef8a8bdcfe977fc5612dbef2c8ed5ff9141299d20a99eb4376  opi/verification/paper_verification/powered_clean_probe_and_corecontent_20260711/ouro_corecontent_task_disjoint_policy.pt
    8e9db6c3fbf9df9ad75716a970203e7931bf83d080d05a692417bbdb51f836e7  opi/verification/paper_verification/powered_clean_probe_and_corecontent_20260711/ouro_corecontent_task_disjoint_results.json (same hash as the commit)
    8d5e09a350f364690a48725d6b9a148f94d1bd0412c9bb7fe9b5bdad57f0c01c  shared/data/corecontent_v2/features/feature_manifest.json

Feature shards (182 in all) are listed individually in `results.json`. These are the RLTT run (18),
base26 (18), thinking26 (18), MiniCPM (64) and CoreContent-v2 (64). The manifests store each shard's
SHA-256 (CoreContent-v2 stores its first 16 hex digits). Each gate checks every shard against its
manifest, and all 182 match.

Main commit files (SHA-256): `probes/bg_xloop_early_v1_common.py` bf44f9b9...6086,
`_stats.py` 8c4bd31c...31c9, `_train_eval.py` 214ab1bc...10dc, `_controls.py` fdd36f07...c4fe (read
only), `tools/paper1_v3/bg_v3_family_xloop.py` 61e1b447...a119,
`tools/paper_verification/nonlooped_transformer_control.py` 0e971cf8...4767,
`rerun_ouro_corecontent_task_disjoint.py` 7c16182d...f664. Full hashes for these and the 24 modules in
the task-5 import chain are in `results.json`.

## Not covered, and caveats

- **Task 2 precision.** The shuffle gate is exact only at the 4 dp of the record, because the
  shuffled weights were never saved. The refit weights are thread-dependent at float precision. The
  original and fair values are thread-independent between 4 and 8 threads.
- **Task 3 measure.** Characters, bytes and stripped characters cannot be told apart from the record;
  they give identical fair values, so the choice does not matter.
- **Task 5 tie definition (judgment call, flagged).** MiniCPM's score averages two z-scored channels.
  In a 2-candidate group each channel's z-scores are exactly (-1, +1), so when the two channels
  disagree, both candidates score exactly 0 in exact arithmetic. 400 such ties survive in float32
  (exact equality). 211 more are broken by float noise of up to about 2e-5 and appear only at 6 dp.
  RESULTS.md gives three values: exact equality (the stated rule), 6-dp rounding, and all
  disagreement groups treated as ties. Ouro averages three channels, so a sign disagreement alone
  cannot cancel there.
- **Bootstrap CIs** are reported only where the published numbers have them (task 1). Tasks 2-5
  publish points only, so only points are compared.
- **The family checkpoints' own shuffle controls** (`train_eval_results.json["shuffle_controls"]` for
  base26, thinking26 and ouro14b) were not requested and are not covered.
