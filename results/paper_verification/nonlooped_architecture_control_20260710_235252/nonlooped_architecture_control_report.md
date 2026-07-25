# Non-looped architecture control

Generated 2026-07-11 (Europe/Zagreb). This report tests whether the readable quality/preference signal is specific to Ouro's looped architecture or is also present in an ordinary deep Transformer. No Ouro weights or checkpoints were modified, and no model was trained or used for broad generation. Only frozen-feature extraction and lightweight linear-probe fitting were performed.

## Executive result

The current evidence does **not** support attributing the readable signal to the looped architecture alone.

1. On CoreContent-v2, a conventional 40-layer SFT Transformer supports a clean task-disjoint linear quality probe: held-out macro top-1 **0.5680** and macro pairwise **0.7237**. Thus a substantial linearly readable content/process-quality signal exists without recurrent/looped blocks.
2. On HH, the historical approximately 84.5% antisymmetric-linear result is reproduced in every backbone only under an orientation-row split that leaks the opposite orientation of the same pair across train and evaluation. With a pair-disjoint split, Base Ouro, Ouro-RLTT, and non-looped MiniCPM are all statistically compatible with chance.
3. Ouro's prior CoreContent-v2 selected-policy held-out macro (**0.6691**) is higher than the non-looped control's corrected **0.5680**, but this numerical gap must not be interpreted as a controlled architecture effect: the backbone, parameterization, feature dimensionality, split provenance, and training history differ. The justified conclusion is availability, not equivalence or causal attribution.

## Model and environment

| Field | Value |
|---|---|
| Control backbone | `openbmb/MiniCPM-2B-sft-bf16` |
| Revision | `4ec16344ac13e6ef5010aeecaa533369ac8eb53c` |
| Weight SHA-256 | `0b0c993ace78c5983373948c636b0e587fcf1ac6f2e0f980bf7d735fe7dc52f8` |
| Architecture | Conventional non-looped decoder Transformer, 40 distinct physical blocks |
| Hidden size | 2304 |
| Dtype / quantization | bfloat16 / none |
| Device | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| PyTorch | 2.12.0.dev20260407+cu128 |
| Transformers | 4.54.1 |
| Feature taps | Physical post-block layers 24 and 36 |
| Pooling | Mean over mask-valid tokens |
| Inputs | Raw CoreContent-v2 `candidate_text` |

## CoreContent-v2 control

The frozen control backbone was evaluated using the recovered CoreContent-v2 protocol: the existing 36,279-group/98,228-candidate corpus, raw/NoNorm candidate differences, bias-free linear scoring, domain-blockwise selection, and the same learning-rate/seed grid. Feature extraction produced 64 resumable shards with zero encoding errors.

The extraction summary says `PARTIAL` because `group_uid` is not globally unique: 36,279 records contain 28,833 unique `group_uid` strings. Direct shard accounting confirms that all **36,279 groups and 98,228 candidates** were saved. This is a manifest bookkeeping defect, not missing features.

An integrity audit also found that 195 repeated task IDs crossed the source file's stored splits. The primary result therefore uses a new deterministic SHA-256 task assignment (seed 20260711): 23,054 unique training tasks, 2,948 validation tasks, and 2,831 held-out tasks, with **zero task IDs crossing splits**. The exact stored-split replication scored 0.5780 top-1 / 0.7323 pairwise; the corrected task-disjoint run scored 0.5680 / 0.7237 and is the headline result.

Best validation configuration: layer 24, all-core training, learning rate 0.001, seed 2.

| Domain | Held-out groups | Top-1 | Pairwise |
|---|---:|---:|---:|
| Coding | 182 | 0.9121 | 0.9645 |
| Math | 298 | 0.3893 | 0.7458 |
| Logic | 212 | 0.3255 | 0.5833 |
| Reasoning | 269 | 0.5204 | 0.7206 |
| Alignment | 2,564 | 0.6927 | 0.6045 |
| **Domain macro** | — | **0.5680** | **0.7237** |

The signal is real but heterogeneous: coding is particularly strong, while logic top-1 remains weak. This establishes that linear readability is available in a non-looped SFT Transformer; it does not establish that the underlying representation or learning mechanism is identical to Ouro's.

## HH antisymmetric-linear audit

All powered runs used the same 1,000 cached Anthropic HH pairs and an exact swap-safe score `w^T(h_chosen - h_rejected)`, with no bias and no normalization. The clean test assigned whole unordered pairs to either training or evaluation (800/200). The diagnostic historical split independently assigned the two oriented rows, allowing a pair's exact negative to cross the boundary.

| Backbone | Model/revision | Leaking row split | Clean pair-disjoint | Clean 95% CI |
|---|---|---:|---:|---:|
| Base Ouro | `ByteDance/Ouro-2.6B` / `1ed04250da1a9936042725d302e81c8fa2ab5abd` | 0.8550 | 0.5100 | [0.4400, 0.5800] |
| Ouro-RLTT | local `models/ouro_rltt_local` | 0.8650 | 0.5300 | [0.4600, 0.6000] |
| MiniCPM-SFT | `openbmb/MiniCPM-2B-sft-bf16` / `4ec16344ac13e6ef5010aeecaa533369ac8eb53c` | 0.8575 | 0.4950 | [0.4250, 0.5650] |

Every scorer is exactly antisymmetric under candidate swap. Exact antisymmetry is therefore necessary but insufficient for a clean evaluation: it constrains the scoring function but does not prevent train/test duplication through negated orientations.

The near-identical leaking accuracies explain why the evaluator genuinely returned the expected approximately 85% numbers. They are reproducible, but they measure orientation-pair memorization under that split rather than generalization to unseen HH pairs.

## Verdict and paper consequence

- `NONLOOPED_CORECONTENT_SIGNAL = PRESENT`
- `HH_ANTISYM_84P5_STATUS = REPRODUCED_WITH_PAIR_ORIENTATION_LEAKAGE`
- `HH_CLEAN_PAIR_DISJOINT_STATUS = CHANCE_COMPATIBLE_ALL_BACKBONES`
- `ARCHITECTURE_CAUSAL_ATTRIBUTION = NOT_SUPPORTED`
- `TRAINING_STAGE_LOCALIZATION = NOT_REPRODUCED_IN_CLEAN_PAIR_DISJOINT_UNITS`

Recommended wording:

> A frozen linear probe trained on task-disjoint CoreContent-v2 data also reads process/content quality from a conventional 40-layer SFT Transformer (held-out domain-macro top-1 56.8%; pairwise 72.4%). Linear readability is therefore not unique to Ouro's looped architecture. Conversely, the earlier approximately 84.5% HH antisymmetric-probe result does not survive pair-disjoint evaluation: Base Ouro, Ouro-RLTT, and the non-looped control score 51.0%, 53.0%, and 49.5%, respectively. The earlier high result arose because opposite orientations of the same preference pair could cross the train/test boundary. These controls support broad availability of quality-relevant linear features, but do not identify looped architecture or RL-style training as their causal source.

Do not retain the Base-versus-Thinking-versus-RLTT localization claim as main-text evidence unless it is rebuilt on a powered pair-disjoint dataset and succeeds. Do not cite the approximately 84.5% HH number as clean relational generalization.

## Remaining limitations

- This is not an architecture-matched causal intervention. MiniCPM and Ouro differ in data, objectives, width, feature geometry, and optimization history.
- The non-looped control establishes presence, not that all Transformers contain the signal.
- A causal architecture claim would require otherwise matched looped and non-looped models trained on the same corpus/objective, ideally from comparable initialization and compute.
- The duplicate `group_uid` namespace should be repaired or replaced by a composite record key before future resumable CoreContent extraction.

## Primary artifacts

- `corecontent_control_results_task_disjoint.json` (primary)
- `corecontent_control_results.json` (stored-split replication)
- `minicpm2_sft_corecontent_v2.pt`
- `hh_linear_probe_results_1000.json`
- `ouro_base_hh_linear_probe_1000.json`
- `ouro_rltt_hh_linear_probe_1000.json`
- `corecontent_features/feature_manifest.json`
- `tools/paper_verification/nonlooped_transformer_control.py`
