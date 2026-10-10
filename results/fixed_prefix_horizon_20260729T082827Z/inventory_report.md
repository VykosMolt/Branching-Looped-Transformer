# Fixed-prefix Horizon inventory report

Inventory timestamp: `2026-07-29T08:44:08.662540+00:00`

## Decision

`FIXED_PREFIX_HORIZON_NOT_ADJUDICABLE`

The repaired experiment stops at the pre-preregistration inventory gate.
Complete generated-token ID sequences are preserved for
**0 of
2720 candidates**. All four canonical Horizon shard
schemas preserve rendered `generated_text` and the scalar `n_gen_tok`, but no
generated-token ID list or full sequence tensor.

The canonical generator moved `out.sequences` to a local `seqs` list, decoded
`new_ids_full` to text, and wrote only `generated_text` plus token counts and
derived features. The temporary IDs were deleted after each task. The v2
terminal pool copies the same record schema and adds `full_features`; it does
not restore token IDs. The v3 shards inherit the identical generator.

Retokenizing the rendered continuation is forbidden and cannot recover the
sampled tokenization uniquely. Regenerating from seeds is both prohibited and
not an exact artifact reconstruction. Therefore no candidate can be replayed
as `prompt token IDs + preserved generated token IDs`, no fixed boundary can
be selected, and no answer detector, replay, feature extraction, or
visible-versus-hidden fitting is authorized.

## Independently relevant identity finding

The historical environment establishes the local checkpoint path,
Transformers 4.54.1, four-loop inference, and repository Git head. Current
checkpoint and tokenizer hashes are recorded. However, the historical
SHA256SUMS files did not stamp the model weights or the untracked generator and
feature-extractor source files. Exact replay identity is therefore not
cryptographically established from the historical package, independently of
the fatal missing-token condition.

## What is preserved

- Exact task identities, candidate indices and record ordering: present.
- Original deterministic task partitions: present and reproduced with zero
  crossing.
- Rendered continuations and generated-token counts: present.
- Eventual correctness labels, parsed committed answers and deterministic
  verifier-equivalent fields: present and internally consistent.
- Original prompts: reconstructible byte-for-byte from the hash-recorded task
  source and formatter; prompt token counts agree with the stored counts.
- Canonical feature definition: layers 24/36/47, four recurrent passes,
  attention-mask mean pooling over the replay input, `[3,4,2048]` float32
  features, followed by standardized PCA and L2 logistic regression.

## Required disposition

No preregistration was sealed because the gate precedes preregistration. Files
with downstream required names are terminal status artifacts, not fitted
results. No model was loaded, no continuation was generated or retokenized,
no replay was attempted, and no predictive performance was inspected.
