# Evaluator Locus Summary

**Original date:** 2026-05-11, extended through 2026-05-14  
**Refactored:** 2026-05-18 for readability  
**Raw original:** `raw_archive_2026-05-18/pairwise_evaluator_locus_memo_v2_2026-05-11.md`

The original file was a multi-pass 1,791-line research log covering v2 through v10. This refactored version keeps the decision-relevant state in one place. No historical text was intentionally deleted; use the raw archive for full details.

## Current Distillation

The evaluator signal is best understood as a pairwise branch-selection readout over latent states, not as an absolute standalone scalar target.

Key historical findings that still matter:

- Mid/late loop and layer choices matter.
- Masked-zero variants were a numerical degeneracy, not a usable signal.
- Normalization and centered/difference parameterizations helped isolate relational signal.
- GRU/temporal aggregation was not justified as the default.
- Later cross-domain work moved the default toward tiny AntisymLinear and NoNorm heads over pooled features.

## Evolution Of The Locus Work

| pass | key point | current status |
| --- | --- | --- |
| v2-v4 | pairwise locus and loop ablations | historical foundation |
| v4-redo | iterated RMSNorm improved HH readout | useful mechanistic context |
| v5 | loop identity became less special under stronger normalization | supports pooled/simple heads |
| math probe | math looked different from HH | superseded by later validity caveats |
| v6 | bias decomposition and branch simulation | supports relational framing |
| v7 | all-layer/cached coding/reasoning/logic probes | historical bridge to domain transfer |
| v8-v9 | evaluator placement and multi-tap ensemble | historical design context |
| v10 | Thinking vs RLTT loop geometry | layer 24/36/47 geometry became central |

## Current Architecture Implication

The current local diagnostic path uses tiny heads:

- `AntisymLinear`: `LayerNorm(no affine)(left - right) -> Linear(no bias)`.
- `AntisymLinearNoNorm`: `Linear(left - right)`.

GRU and larger learned evaluators are no longer default for these probes.

## Current Domain Implication

The locus work alone did not settle domain generalization. Later domain-transfer probes did:

- HH-trained taps transfer to clean GSM8K.
- HH-trained taps transfer to runnable-diagnostic code.
- HH-trained taps are weaker on strict-clean code.
- code-specific tiny heads are substantially better on strict-clean code.
- code-specific heads show partial small-split inverse transfer to HH, but not enough to replace HH-trained heads.
- reasoning multiple-choice transfer is promising but still early.

## Numbers To Use Now

HH all-200 diagnostic:

- HH-trained `47_concat_L1_L4 / NoNorm`: accuracy 0.855.
- code-trained `47_L4 / NoNorm`: accuracy 0.535.

Strict-clean code ALL16:

- HH-trained `47_mean / AntisymLinear`: top1 0.750, pairwise 0.600.
- code-trained `36_L4 / AntisymLinear`: top1 0.875, pairwise 0.833.

Reasoning pilot:

- best code-trained `24_L4 / AntisymLinear`: top1 1.000, pairwise 1.000.
- best HH-trained `36_mean / AntisymLinear`: top1 0.960, pairwise 0.986.

## Current Recommendation

Use a general-plus-specialist readout policy:

- HH-trained general head for HH/preference-like comparisons.
- code-trained specialist for strict-clean code.
- add reasoning as a third objective eval domain after harder validation.

Do not use the long historical memo as the current operating spec without also reading:

- `current-state.md`
- `domain-transfer-ledger.md`
- `tap-interface.md`
