# Final report: visible-prefix versus hidden-state conference gate

## Outcome

The conference endpoint is **not adjudicable**. This is not a null result and
not evidence that visible and hidden constructions are equivalent.

The preserved GSM8K strict-preanswer cohort lacks exact candidate trajectories
or prefix text and lacks an original train/validation/held-out partition. The
Horizon cohorts preserve exact rendered trajectories and splits, but they fail
the required strict visible-prefix contract: the pre-marker reasoning prefix
literally or symbolically states the normalized gold conclusion in
451 of 503 new-only tasks
(89.7%) and 598 of
669 pooled tasks
(89.4%). Taskwise removal leaves only
22 train,
11 validation, and
19 held-out
new-only tasks. This exceeds the sealed 20% materiality threshold in every
split and falls below the sealed minimum task counts.

Consequently, fitting a strong visible reader on the unchanged Horizon text
would answer a different, answer-contaminated question. Fitting on the small
post-exclusion residue would also change the estimand materially. The sealed
stopping rule therefore prohibited V1/V2/V3, T, R reproduction, V/H/HV fusion,
held-out endpoint evaluation, bootstrapping, equivalence testing, calibration,
and selective-prediction analyses.

## Direct answers to the conference questions

1. **Does visible prompt-plus-prefix text explain the published hidden-state
   effect?** Not adjudicable from these preserved strict cohorts. Horizon
   visible prefixes often contain the semantic answer, while GSM8K exact
   prefixes are absent.
2. **Do hidden states retain a statistically supported conditional increment
   beyond the strongest visible reader?** Not established; HV−V was not validly
   estimable.
3. **Are visible and hidden constructions practically equivalent?** Not
   established. The clustered equivalence test was not run.
4. **Does visible text outperform the hidden construction?** Not established.
5. **Was HV fully adjudicable?** No, in every arm.
6. **Which visible family won validation?** None; selection was prohibited by
   the upstream gate.
7. **V3 seed-level results:** seeds 20260729, 20260730, and 20260731 were all
   `NOT_RUN_UPSTREAM_VISIBLE_PREFIX_GATE`; this was not a resource failure.

## Endpoint hierarchy and verdicts

| Arm | Endpoint status | Verdict |
|---|---|---|
| Horizon Logic new-only (designated primary) | HV−V not estimable | `VISIBLE_PREFIX_NOT_ADJUDICABLE` |
| GSM8K (replication) | exact prefix and original split absent | `VISIBLE_PREFIX_NOT_ADJUDICABLE` |
| Horizon Logic pooled (secondary precision) | HV−V not estimable | `VISIBLE_PREFIX_NOT_ADJUDICABLE` |
| Aggregate | designated primary not adjudicable | `VISIBLE_PREFIX_AGGREGATE_NOT_ADJUDICABLE` |

No fallback H−V verdict is assigned: GSM8K has no valid V construction, and
Horizon's leak-free V estimand is materially changed and under the minimum
split sizes. A fallback comparison would not restore the requested conditional
question.

## Historical sensitivity, not conference evidence

The unchanged historical analyses are retained, as required, but they do not
answer HV−V:

- GSM8K grouped OOF: shortcuts AUROC
  0.731, hidden+shortcuts
  0.797, delta
  +0.066, task-clustered 95% CI
  [+0.021, +0.112]. This has
  no preserved exact visible prefix and no original held-out split.
- Horizon new-only historical four-shortcut result: shortcuts AUROC
  0.677, hidden+shortcuts
  0.773, delta
  +0.095, 95% CI
  [+0.038,
  +0.157].
- Horizon pooled historical four-shortcut result: shortcuts AUROC
  0.652, hidden+shortcuts
  0.763, delta
  +0.111, 95% CI
  [+0.056,
  +0.169].

The Horizon historical construction includes `hit_max_tokens`, which this
conference protocol forbids as post-cut generation metadata. None of these
historical deltas is conditional on a strong visible-prefix reader.

## Leakage and reconstruction

No forbidden `FINAL ANSWER` marker survived in a reconstructed Horizon prefix,
and no prompt token-count reconstruction error was found. The gold occurrences
are genuine stored-prefix content, not a tokenizer/rendering repair error.
The hand audit confirms that reasoning commonly states `True`, `False`, or
`Unknown` before the final marker. Structurally unusual prefixes sometimes
contain incomplete marker fragments such as `FINAL ANSW`; these do not satisfy
the frozen original `FINAL\s*ANSWE` cut regex and are recorded without repair.
No individual span was censored.

There were
0 exact
cross-split duplicate-prompt groups and
5 cross-split
near-duplicate pairs at the sealed 0.95 character-TFIDF threshold. File-order
held-out AUROC was 0.470 new-only and
0.485 pooled; pooled source-ID AUROC
was 0.470. These diagnostics did not
enter any model.

## Truncation

Under the frozen DeBERTa-v3-small allocation, new-only held-out truncation was
21.7% (96/443);
pooled held-out truncation was 21.8%
(134/614). All truncation was prefix overflow;
zero prompts exceeded the 256-token prompt allocation. Because no classifier
was fitted, truncation-performance sensitivity is not estimable.

## Domain-specific limitations

- GSM8K is blocked by missing exact visible candidate text and missing original
  three-way partitions.
- Horizon is blocked by pervasive genuine answer content before the stored
  final-answer marker.
- The historical Horizon scorable cohort was itself selected using
  post-generation malformedness; this selection is preserved only for
  historical comparability and is not used as a feature.
- Raw hidden-only train/validation scores were not preserved. Frozen hidden
  features and a deterministic reproduction specification exist, but the
  reproduction was not run after the earlier terminal leakage gate.
- Re-encoding the exact rendered trajectories and applying the stored token
  count did not byte-match the original char-cut prefix on
  364 new-only and
  489 pooled scorable candidates.
  This reflects loss of the original generated token IDs and reinforces the
  need for prospective token-ID preservation. The original hidden feature
  input itself remains exactly reconstructible because the frozen generator
  implementation built it from the preserved rendered text and the frozen
  char-level cut function.
- A new prospective cohort with an answer-semantic cut or a separately
  annotated answer-free reasoning boundary is required. This package does not
  authorize generation and therefore cannot repair the missing estimand.

## Artifacts and reproducibility

Artifact root:
`/home/moloch/ouro_project/artifacts/reports/visible_prefix_hidden_increment_20260729T075247Z`

Key seals:

- inventory JSON SHA-256:
  `879fd166fd2d2512dd4f2527a3cc483808bfaf946adc400c143c14e4b6a6c84a`
- preregistration SHA-256:
  `1cf90fe1c26ca462c261489dba207af82828d6ff648b15d0a930d0db76f629b0`
- exact-prefix manifest SHA-256:
  `1d0cf9ab9d0bb8d4c9194c4b7177ea4034d4a64e23b194e2dd5df9b91d77a7ee`
- leakage audit SHA-256:
  `8f1fece96268227fd5fb46f7cc22e3b5d0519ef42e36df494f19edd36183d566`

The checksum manifest contains the hash of every payload file. The
inventory-to-final-package runtime was 971.5 seconds; preserved
source-generation runtimes remain in the source payloads and are not counted
as new runtime.

Reproduction from the sealed artifact root:

```bash
bash utilities/tests/manual/reproduce_visible_prefix_hidden_increment_evidence_package.sh   /home/moloch/ouro_project/artifacts/reports/visible_prefix_hidden_increment_20260729T075247Z
```

No frozen manuscript or arXiv tarball file was modified.
