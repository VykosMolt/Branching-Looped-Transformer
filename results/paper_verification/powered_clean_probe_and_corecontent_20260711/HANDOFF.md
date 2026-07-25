# Powered clean probe and corrected CoreContent handoff

Last updated: 2026-07-12, Europe/Zagreb.

## One-command continuation

From `/home/moloch/ouro_project`, run:

```bash
bash tools/paper_verification/continue_powered_hh_probe.sh
```

The driver sets the project-local Hugging Face cache and offline dataset mode. It skips completed stages automatically. Remaining work, in order:

1. Resume Thinking extraction from completed 500-pair shards (Base extract+fit already done and will be skipped).
2. Fit/evaluate Thinking.
3. Extract RLTT features on the identical pair indices.
4. Fit/evaluate RLTT.

Do not start a second copy of the driver or extraction command while one is active.

## Paused state (2026-07-12)

- Base: **COMPLETE** — extraction (80 shards, 40,000 pairs), manifest, fit, and results all written.
  - Base result: eval accuracy **0.55525**, 95% pair-bootstrap CI **[0.544375, 0.56625]**, train accuracy 0.6916875, `exact_antisymmetry=true`, `max_swap_sum_abs=0`.
- Thinking extraction: **13,000 / 40,000 pairs complete** (26 shards, `diffs_0000.pt` through `diffs_0025.pt`).
- Shard size: 500 source pairs.
- The interrupted in-memory partial shard was not written and will be recomputed.
- Thinking completion manifest does not yet exist; this is expected.
- RLTT extraction: not started.
- Driver and model process were stopped cleanly; GPU memory was released.

Resume logic counts rows in completed shards, so the next Thinking position is 13,000. Existing shards are reused and never overwritten.

### Cache fix applied 2026-07-12 (do not undo)

The extraction script passes `cache_dir=artifacts/hf_cache` directly, so hub-format model dirs must sit at the top level of `artifacts/hf_cache/` (NOT under `artifacts/hf_cache/hub/`). `ByteDance/Ouro-2.6B-Thinking` was only cached in `~/.cache/huggingface/hub`, which made the Thinking stage fail offline. Fixed with a symlink, verified at the pinned revision:

```
artifacts/hf_cache/models--ByteDance--Ouro-2.6B-Thinking -> ~/.cache/huggingface/hub/models--ByteDance--Ouro-2.6B-Thinking
```

If that symlink is missing, recreate it before running the driver.

## Fixed experimental protocol

- Dataset: `Anthropic/hh-rlhf`, split `train`, cached offline.
- Source pairs: 40,000, selected without replacement.
- Shared pair-index file: `hh_train_indices_40000.json`.
- Pair-selection seed: `20260711`.
- Pair-disjoint split seed: `20260711`.
- Training/evaluation pairs: 32,000 / 8,000.
- Candidate truncation: 384 tokens.
- Feature: concatenated mean-pooled representations from four post-final-norm Ouro loop boundaries.
- Feature width: 8,192.
- Probe: bias-free L-BFGS logistic linear readout.
- Difference: raw/NoNorm `h_chosen - h_rejected`.
- Swap scoring: exact `s(B,A) = -s(A,B)`.
- CI: 10,000-draw bootstrap over held-out source pairs.
- Forward dtype: bfloat16.
- Saved feature dtype: float16.
- Quantization: none.
- No Ouro training, checkpoint writes, generation sweep, or paper modification.

Backbones:

- Base: `ByteDance/Ouro-2.6B`, revision `1ed04250da1a9936042725d302e81c8fa2ab5abd`.
- Thinking: `ByteDance/Ouro-2.6B-Thinking`, revision `f1edd81e7ac41355db670500ceaf204e0f73af68`.
- RLTT: `/home/moloch/ouro_project/models/ouro_rltt_local`.

## Runtime expectation

Observed Thinking throughput before pause was approximately 3.8 pairs/second. Thinking has 27,000 pairs left, or roughly 2 hours of extraction. RLTT should require approximately three hours if throughput is similar. Allow additional time for two L-BFGS fits (each under a minute) and bootstrap evaluation. Expected remaining wall time is approximately 5–6 hours.

Every extraction is resumable at the most recent 500-pair shard if interrupted.

## Corrected Ouro CoreContent result — already complete

Artifact: `ouro_corecontent_task_disjoint_results.json`.

- Historical stored-split held-out macro top-1: `0.6691`.
- Corrected task-disjoint held-out macro top-1: **`0.6309983905`**.
- Difference: approximately `-0.0381`.
- Stored task IDs crossing splits: 195.
- Corrected effective task IDs crossing splits: 0.
- Unique tasks: 28,833.
- Corrected split: 23,054 train / 2,948 validation / 2,831 held-out tasks.

Corrected held-out top-1 by domain:

| Domain | Groups | Top-1 |
|---|---:|---:|
| Coding | 182 | 0.895604 |
| Reasoning | 269 | 0.598513 |
| Math | 298 | 0.640940 |
| Logic | 212 | 0.405660 |
| Alignment | 2,564 | 0.614275 |
| Macro | — | **0.630998** |

Conclusion: the historical `0.6691` does not stand unchanged, but the Ouro CoreContent signal remains substantial under the corrected task-disjoint split.

## Expected powered-probe outputs

For each of `base`, `thinking`, and `rltt`:

- `hh_<backbone>_40000_diffs/manifest.json`
- `hh_<backbone>_40000_pair_disjoint_probe.pt`
- `hh_<backbone>_40000_pair_disjoint_results.json`

All paths are under:

`/home/moloch/ouro_project/artifacts/reports/paper_verification/powered_clean_probe_and_corecontent_20260711/`

## Code

- Extraction/fitting: `tools/paper_verification/powered_hh_pair_disjoint_probe.py`
- One-command continuation: `tools/paper_verification/continue_powered_hh_probe.sh`
- Corrected CoreContent refit: `tools/paper_verification/rerun_ouro_corecontent_task_disjoint.py`

## Completion checklist

After the continuation command exits successfully:

1. Confirm all three result JSON files exist and report 40,000 source pairs, 32,000 training pairs, and 8,000 evaluation pairs.
2. Confirm `exact_antisymmetry=true` and `max_swap_sum_abs=0` for every backbone.
3. Compare clean accuracies and pair-bootstrap CIs across Base, Thinking, and RLTT.
4. Do not use the historical orientation-row split as the headline.
5. Combine these results with corrected Ouro CoreContent `0.6310` and non-looped MiniCPM task-disjoint `0.5680`.
6. Produce the final Markdown/JSON paper-verification report without editing draft v3.16.

## Manual recovery commands

If a specific stage must be resumed manually, use the same environment variables set by the driver:

```bash
export HF_HOME=/home/moloch/ouro_project/artifacts/hf_cache
export HF_DATASETS_CACHE=/home/moloch/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1
```

Then, for example, resume Thinking extraction:

```bash
venv/bin/python tools/paper_verification/powered_hh_pair_disjoint_probe.py extract \
  --backbone thinking --pairs 40000 \
  --output-dir artifacts/reports/paper_verification/powered_clean_probe_and_corecontent_20260711
```

Do not run `fit` until that backbone has all 40,000 saved differences and its manifest reports `COMPLETE`.
