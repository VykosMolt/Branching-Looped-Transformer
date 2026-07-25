# Fixed-order evaluator decomposition addendum

This follow-up runs the four requested diagnostics on the same 200 HH-RLHF pairs used by the Fable hardening check. It uses the original `pairwise_epoch2.pt` checkpoint unchanged. No model or evaluator was trained, no generation was run, and no checkpoint or paper file was modified.

## Executive result

The evaluator's approximately 95% canonical-order number is real and reproducible, but it is **not localized to Thinking/RLTT** in this controlled run:

| Backbone | Canonical `(chosen,rejected)` | Swapped correctness | Strict antisymmetric accuracy | Both orders prefer first |
|---|---:|---:|---:|---:|
| Base Ouro-2.6B | **0.950** | 0.110 | 0.580 | 0.840 |
| Ouro-2.6B-Thinking | **0.950** | 0.130 | 0.595 | 0.820 |
| Ouro-RLTT | **0.945** | 0.130 | 0.600 | 0.815 |

Thus:

- the historical 95.2%/95.0% canonical values reproduce on this slice;
- the claimed base collapse to 24% does **not** reproduce—the base model scores 95.0%;
- the canonical score is dominated by a positive symmetric/order component on all three backbones;
- neither learned attention pooling nor difference normalization explains a base-specific collapse.

Verdicts:

- `HISTORICAL_BASE_24_STATUS = NOT_REPRODUCED_NO_ARTIFACT_FOUND`
- `FIXED_ORDER_MECHANISM = SYMMETRIC_ORDER_PRIOR_DOMINATES_ALL_BACKBONES`
- `POOLING_ABLATION = ATTENTION_NOT_REQUIRED_FOR_CANONICAL_HIGH_ACCURACY`
- `NORMALIZATION_ABLATION = NORMALIZATION_MITIGATES_ORDER_DOMINANCE`
- `LOCALIZATION_IMPLICATION = NO_CROSS_BACKBONE_LOCALIZATION_IN_ORIGINAL_OR_LINEAR_UNITS_ON_CONTROLLED_SLICE`

## 1. Token-level capture

Thinking and RLTT use their existing token-level captures:

- `artifacts/reports/evaluator/hh_loop_states_200_thinking.pt`
- `artifacts/reports/evaluator/hh_loop_states_200_rltt.pt`

Base token states were captured in memory and immediately scored from:

- model: `ByteDance/Ouro-2.6B`
- revision: `1ed04250da1a9936042725d302e81c8fa2ab5abd`
- dataset: `Anthropic/hh-rlhf`, test split
- selection: the identical 200 sorted indices from `default_rng(seed=42)`
- tokenizer: `/home/moloch/ouro_project/models/ouro_rltt_local`
- truncation: 384 tokens
- loops: four forced loops, `early_exit_threshold=1.0`
- backbone forward: bf16; captured states/evaluator: fp32; no quantization

The base states were not persisted, avoiding another approximately 2.6 GB artifact. Per-pair scores and decompositions are saved in `evaluator_decomposition_rows.csv`.

Evaluator checkpoint:

- `artifacts/checkpoints/evaluator/pairwise_epoch2.pt`
- SHA-256: `3630c2092eca8db13239f763bc9c212f4b673866e47f811c3095efc57409ec96`

## 2. Both-order scoring and decomposition

For every pair:

\[
s_\text{canonical}=s(\text{chosen},\text{rejected}),\qquad
s_\text{swapped}=s(\text{rejected},\text{chosen}),
\]

\[
s_\text{anti}=\frac{s_\text{canonical}-s_\text{swapped}}{2},\qquad
s_\text{order}=\frac{s_\text{canonical}+s_\text{swapped}}{2}.
\]

Original architecture (`attention_with_norm`):

| Backbone | Canonical acc. | 95% CI | Swapped acc. | Antisym acc. | 95% CI | Sign-flip rate | `mean(s_order)` | `mean|s_order| / mean|s_anti|` |
|---|---:|---|---:|---:|---|---:|---:|---:|
| Base | 0.950 | [0.920,0.975] | 0.110 | 0.580 | [0.510,0.650] | 0.160 | 1.259 | 1.828 |
| Thinking | 0.950 | [0.920,0.980] | 0.130 | 0.595 | [0.530,0.665] | 0.180 | 1.293 | 1.652 |
| RLTT | 0.945 | [0.910,0.975] | 0.130 | 0.600 | [0.530,0.670] | 0.185 | 1.292 | 1.622 |

The positive order component is nearly the same across all three backbones. It is larger in absolute magnitude than the antisymmetric component, and 81.5–84.0% of pairs are scored as preferring the first argument in **both** orders. This explains why canonical chosen-first accuracy is high and swapped correctness is very low.

The negative canonical-vs-swapped correlations (base −0.872, Thinking −0.900, RLTT −0.902) show that content still changes the scores; the evaluator is not constant. But its positive symmetric offset overwhelms the sign on most pairs.

## 3. Pooling ablation

Replacing learned attention pooling with masked mean pooling while preserving trained difference normalization, projection, GRU, nonlinear scorer, and all weights gives:

| Backbone | Attention canonical / antisym | Mean canonical / antisym | Change in canonical | Change in antisym |
|---|---:|---:|---:|---:|
| Base | 0.950 / 0.580 | 0.950 / 0.550 | 0.000 | −0.030 |
| Thinking | 0.950 / 0.595 | 0.935 / 0.580 | −0.015 | −0.015 |
| RLTT | 0.945 / 0.600 | 0.935 / 0.565 | −0.010 | −0.035 |

Learned attention pooling is therefore not responsible for either the approximately 95% canonical result or a cross-backbone separation. Mean pooling preserves essentially the same fixed-order behavior.

## 4. Normalization ablation

Bypassing only the evaluator's trained difference `LayerNorm` increases, rather than removes, the order-dominated canonical result:

| Pooling | Backbone | With norm canonical / antisym | Without norm canonical / antisym | Order/antisym ratio with → without norm |
|---|---|---:|---:|---:|
| attention | Base | 0.950 / 0.580 | 0.990 / 0.595 | 1.83 → 3.35 |
| attention | Thinking | 0.950 / 0.595 | 0.990 / 0.600 | 1.65 → 3.18 |
| attention | RLTT | 0.945 / 0.600 | 0.990 / 0.610 | 1.62 → 3.11 |
| mean | Base | 0.950 / 0.550 | 0.995 / 0.560 | 1.97 → 3.28 |
| mean | Thinking | 0.935 / 0.580 | 0.990 / 0.550 | 1.89 → 3.20 |
| mean | RLTT | 0.935 / 0.565 | 0.990 / 0.560 | 1.85 → 3.14 |

Normalization is not causing the order bias. It partially restrains it: without normalization, canonical accuracy rises to 99–99.5%, sign-flip rates fall to 3.5–4.5%, and the symmetric/order component becomes roughly three times the antisymmetric component.

## 5. What this says about the historical 24%

The 24% base number is not supported by any surviving repository artifact found in a targeted search. Its only occurrences are in draft v3.16, whose own claim table labels it **“User-confirmed; base-eval artifact path pending.”** The controlled rerun using the pinned current base revision, the original evaluator checkpoint, identical tokenizer/input pairs, and both-order scoring yields 95%, not 24%.

Plausible explanations for the historical number now narrow to:

1. a different base checkpoint or model revision was evaluated;
2. a different feature hook/locus, tokenizer, truncation, early-exit setting, or pair orientation was used;
3. the 24% came from a different metric or evaluator checkpoint;
4. the historical number was transcribed or remembered incorrectly.

The present evidence does **not** distinguish among those possibilities because the original command, artifact, and per-pair scores are absent. What it does establish is that the available canonical evaluator checkpoint does not produce a base-specific collapse under the controlled current protocol.

## 6. Paper implication

The previous recommendation stands and is strengthened: remove/TODO §3.7's training-stage localization claim and its abstract/contribution echoes. The controlled evidence now shows no localization under either:

- the reconstructed bias-free antisymmetric linear probe: 83.75% / 83.75% / 83.75%; or
- the original high-capacity evaluator in canonical order: 95.0% / 95.0% / 94.5%.

The original evaluator's canonical accuracy may still be reported as historical discovery-stage behavior, but its strict antisymmetric accuracy on this slice is only 58–60% and it cannot support a reasoning-stage localization claim.

