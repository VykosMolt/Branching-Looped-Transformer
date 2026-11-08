# T1: fair tie handling for candidate-ranking top-1

Post hoc reanalysis, 2026-11-08, for OPI v4. Not part of the preregistered protocol.

## Why

`group_selection_metrics` (in `probes/bg_xloop_early_v1_common.py`) scores top-1 by taking
the first index of the maximum score. Several dataset builders put the correct candidate
first (`::pos` before `::negN`), so a tie at the maximum is always credited as correct.
This reanalysis gives a tied group fractional credit instead:

    top1_fair = #(candidates tied at the group's max saved score with reward >= max reward)
                / #(candidates tied at the max)

The rule uses exact equality of the saved 6-decimal scores. Pairwise accuracy already gave
0.5 credit to ties, so it is unchanged.

## Inputs (read-only, git commit `b8c95e31198315fae4dc574376009d70265a1695` of `/home/moloch/ouro_project`)

- Ouro-RLTT: `results/cross_loop_early_layer_taps_20260720/` (14 refit + 18 transfer prediction files, `bootstrap_stats.json`, `train_eval_results.json`)
- Ouro-2.6B base / 2.6B-Thinking / 1.4B: `results/family_xloop_v3_20260726/{base26,thinking26,ouro14b}/` (32 files each; `bootstrap_stats.json` or `bootstrap_stats_v3.json`; `model_plan.json`)
- Huginn-0125: `results/huginn_probe_v3_20260726/` (25 files, `bootstrap_stats_v3.json`)
- Code, imported unmodified from the commit: `probes/bg_xloop_early_v1_common.py`, `probes/bg_xloop_early_v1_stats.py`, `tools/paper1_v3/bg_v3_family_xloop.py`, `tools/paper1_v3/bg_v3_huginn_probe.py`. Also read: `probes/bg_xloop_early_v1_controls.py` and the MiniCPM control file.

The script extracts these with `git archive` into a temporary directory. It never reads the
working tree, which has unrelated local changes. The SHA-256 of every non-prediction input
is recorded in `results.json` under `inputs_sha256`.

## Command

    cd /home/moloch/ouro_project/opi/posthoc_review_20261108/T1_fair_ties
    /home/moloch/ouro_project/shared/venv/bin/python -I fair_ties.py

This uses Python 3.14 with numpy 2.4.4 and runs in about 20 s on CPU. The script runs:

1. **Gate A**, per file and per row: top-1 recomputed from the saved scores with the original
   `group_selection_metrics` (first-max) must equal the stored `top1_oracle`.
2. **Gate B**, per run: the original stats entry point (`bg_xloop_early_v1_stats.main` for
   RLTT, base26 and thinking26; `bg_v3_family_xloop.generic_stats` for ouro14b and Huginn,
   dispatched through the original `stage_stats`), with `RUN_ROOT` and `PRED_ROOT` redirected,
   must reproduce the published stats file exactly. The comparison covers macro points, CIs,
   paired deltas (point, CI, p>0), verdict labels, the best-early cell, and the draws, seed
   and cluster settings. Only `elapsed_seconds` is excluded. On any mismatch the script
   writes `gate_failure.json` and exits 1.
3. **Fair rerun**: the fair rows are written as prediction files with
   `top1_oracle := top1_fair`. The same original stats code then runs on them, with the same
   seed (20260720), the same 10,000 draws and the same task-clustered resampling. The
   loop-1 to loop-4 rises that the generic stats files do not publish (ouro14b, Huginn; also
   loop 1 to 8 for Huginn) are computed from the same bootstrap samples. They are marked
   "not in published file".
4. **Tie inventory** per file, with the evidence for whether ties are duplicates or
   coincidences, and an estimate of how many 6-decimal coincidences chance would produce.

Outputs: `results.json` (everything; deterministic, keys sorted), `RESULTS.md` (tables
generated from it), `SHA256SUMS` (written by the script; covers this README too).

Each gate was checked once with a planted bug (not part of the script). Replacing first-max
with last-max makes Gate A fail. Changing the bootstrap seed by +1 makes Gate B fail, with
94 mismatches on RLTT.

## Not covered

- **Cross-model RLTT-tap transfer** (`crossmodel_rltt_transfer.json`, base26/thinking26):
  only macro top-1 per cell is stored, with no per-group rows, so it cannot be recomputed.
- **MiniCPM-2B non-looped control** (0.568): the file stores only per-domain and macro
  aggregates, with no per-group scores, so it cannot be checked.
- **Length baseline** (0.323 longer-first, 0.370 shorter-first): no code in the commit
  produces `length_baseline.json`. `bg_xloop_early_v1_controls.py` only runs the
  extraction-path controls (feature match, locked-policy reproduction) and computes no
  length ranking, so its tie handling cannot be inspected. Candidate texts and lengths are
  not in the commit either. It cannot be recomputed from public data.
- **Reverse transfers and shuffled-label rows**: not public, and not reconstructed.
- **`s3b2_*` prediction files** (secondary generated-branch readout): out of scope.
