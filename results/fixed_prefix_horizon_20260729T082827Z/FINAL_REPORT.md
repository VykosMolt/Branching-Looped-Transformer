# Final report: repaired fixed-prefix Horizon experiment

## Verdict

`FIXED_PREFIX_HORIZON_NOT_ADJUDICABLE`

The repaired experiment is non-adjudicable at the mandatory artifact inventory
gate. This is not an underpowered result, a null result, evidence of
equivalence, or evidence for either hidden or visible advantage.

## Why the final-marker experiment was invalid

The prior sealed audit found that the final-marker prefix already contained
the normalized answer in 451/503 scorable new-only tasks and 598/669 pooled
tasks. “Before the final marker” was therefore not a genuinely answer-free
boundary. Those historical hidden-versus-shortcut results remain only
non-conditional sensitivities and are not reused as evidence for the repaired
question.

## Inventory gate

The four canonical, historically checksum-matched Horizon shards contain 2,720
candidates: 2,040 prospectively generated v3/new-only candidates plus 680
original v2 candidates. Exact prompts, prompt token counts, task IDs,
candidate indices/order, deterministic splits, rendered continuations,
eventual labels, committed parsed answers, generation metadata, and canonical
hidden-feature definitions are present.

The fatal field is absent:

- candidates with complete preserved generated-token IDs: **0/2,720**;
- candidates with only rendered continuation text and `n_gen_tok`: **2,720/2,720**;
- token ID fields in shard records: none;
- token ID fields in light indexes: none;
- token ID fields in the v2 derived terminal pool: none.

The canonical generator converted `out.sequences` to a transient list, decoded
each `new_ids_full` sequence into `generated_text`, stored only the text and
counts, then deleted the transient IDs. Retokenizing the rendered text cannot
establish the sampled tokenization uniquely and is explicitly forbidden.
Regenerating from seeds is generation, can drift, and is also forbidden.

The current tokenizer matches `ByteDance/Ouro-2.6B` revision
`1ed04250da1a9936042725d302e81c8fa2ab5abd`, and all reconstructed prompt token
counts match. However, the historical package did not stamp checkpoint weight
or custom-code hashes; it records only the local path, environment and Git
head. This independently prevents fully authenticated replay identity.

## Boundary selection and detector

No N was selected. The fixed grid `[16, 32, 48, 64, 96, 128]` was never evaluated because its
inputs do not exist. Accordingly:

- strict-clean retention at every checkpoint: not estimable;
- training retained-class counts: not estimable;
- projected held-out adequacy: not estimable;
- answer-detector false positives/negatives: not estimable;
- hand detector audit: not run.

No detector code was frozen or corrected, no answer strings were censored, and
the contaminated final-marker cut was not reused.

## Replay and predictive results

Replay was not attempted and the model was not loaded. Maximum absolute replay
error, RMS error and token agreement are undefined. No hidden feature was
extracted.

Therefore all of the following are not estimable for both Horizon new-only and
Horizon pooled:

- AUROC(V0), AUROC(V1), AUROC(V2), AUROC(V3);
- AUROC(V), AUROC(H), AUROC(HV);
- H−V0, H−V and HV−V;
- calibration, Brier score, ECE, AUARC and risk–coverage;
- checkpoint curves, clustered confidence intervals and equivalence.

No visible family won validation. V3 seeds 20260729, 20260730 and 20260731 were
all `NOT_RUN_INVENTORY_GATE_FAILURE`, not technical resource failures. Practical
equivalence was not tested; visible advantage and conditional hidden increment
were not tested.

## Cohort integrity

- primary new-only tasks/candidates:
  510/2040; task splits train
  266, validation
  99, held-out
  145;
- original historical component tasks/candidates:
  170/680; task splits train
  80, validation
  31, held-out
  59;
- secondary pooled tasks/candidates:
  680/2720; task splits train
  346, validation
  130, held-out
  204;
- zero task crossing: true;
- zero duplicate candidate IDs: true;
- zero candidate crossing: true;
- exact source offset/task ordering reproduced for all four shards: true;
- verifier-equivalent label mismatches: 0;
- malformed-field equivalence mismatches: 0.

The primary new-only and secondary pooled endpoints receive the same
`FIXED_PREFIX_HORIZON_NOT_ADJUDICABLE`. No pooled precision estimate exists and pooling cannot replace the
missing primary result.

## Limitations and required repair

This experiment can be made adjudicable only by prospectively preserving the
exact prompt IDs, generated IDs, per-token log probabilities, tokenizer and
checkpoint hashes, custom model code hash, deterministic split, and verifier
record for every continuation. Existing rendered text cannot be upgraded into
the missing token artifact without changing the estimand.

The outcome neither restores nor rejects the original strict pre-answer
interpretation. It leaves that interpretation unsupported by the existing
Horizon artifacts.

## Artifacts, hashes and runtime

Artifact root:
`/home/moloch/ouro_project/artifacts/reports/fixed_prefix_horizon_20260729T082827Z`

- inventory report SHA-256: `990b6e6f590ca0487d51a9f1b8db92c33bfc7cbf89964bb804f520b94e900797`
- candidate token manifest SHA-256: `bfaf1833d19b14faaaf8a4989db855cad156e4ef23a19bd2a2960abefa66b43b`
- inventory-to-package runtime: 6.2 seconds

The checksum manifest records every payload hash. One-command reproduction of
the inventory gate and terminal package:

```bash
bash utilities/tests/manual/reproduce_fixed_prefix_horizon_gate.sh /home/moloch/ouro_project/artifacts/reports/fixed_prefix_horizon_20260729T082827Z
```

No generation, continuation retokenization, replay, fitting, or manuscript
modification occurs in that command.
