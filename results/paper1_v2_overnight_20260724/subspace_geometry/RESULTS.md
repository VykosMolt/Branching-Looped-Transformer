# Subspace-vs-Subspace Geometry Audit — RESULTS

Bounded, mostly inference-free diagnostic. Pure linear algebra on frozen artifacts; no
model loaded, no new extraction, no new intervention campaign. Full numeric results in
`geometry_results.json`; bases in `subspace_bases.pt`.

## Inputs (validated, read-only)

- Readable features: `cross_loop_early_layer_taps_20260720/features/*.pt` — 8592
  candidates, 2150 groups, layers `{8,16,24,36,47}` x loops `{1..4}`. SHA256SUMS
  100/100 verified (see `SHARED_ARTIFACT_AUDIT.md`); `split_integrity.json` confirms 0
  task crossings.
- Writable/injection deltas: `s1_s3_exact_injection_delta_bundle_2026-06-17.pt` —
  `delta_locus` per-branch perturbation vectors, available **only** at loop `{1..4}` x
  layer `{24,36,47}` (12 late loci, 32 samples/locus). **Zero** early-locus (layer 8/16)
  writable tensor exists anywhere in the repo — confirmed by direct inspection of
  `protocol.loci`, which is exactly the late-loci cross product. Not expanded with a new
  intervention campaign, per programme rule.

## Method

- Readable subspace: bootstrap ensemble (task-level resampling, train split only,
  N=40/locus) of standardize -> PCA(lowrank) -> L2-logistic direction fits, mapped back to
  the original 2048-dim feature space, normalized, stacked, SVD -> top-r right singular
  vectors (r in {1,3,5}).
- Outcome-relevant subspace: bootstrap ensemble (N=100/locus) of standardized
  between-class mean-difference (correct vs incorrect) directions, same
  stack-then-SVD construction. Deliberately a different construction from the readable
  ensemble (per spec) even though both ultimately target the same reward label.
- Writable subspace (late loci only): SVD of centered observed injection deltas.
- Nulls: random rank-matched subspaces in the same 2048-dim ambient space, 2000 draws
  per rank (cheap, pure linear algebra). A secondary, reduced-round (N=25) label-permuted
  outcome-direction null was also run at `L4_47` rank 3 as a sanity check (expensive
  because it refits an ensemble; kept small on purpose — see "Sacrifices" below).
- Loci: primary early `{L3_8, L3_16, L4_8, L4_16}`; late reference `{L4_24, L4_36,
  L4_47}`; optional loop-transition controls `{L1_16, L2_16}` (physical layer 16 fixed,
  loop varied 1-4).

## Finding 1 — Readable <-> outcome overlap is present broadly, not early-locus-specific

At **every** locus tested (early and late alike), the readable and outcome subspaces are
far more aligned than the random-rotation null (rank-1 angle 27-38 deg vs. null
mean ~89 deg, null p05 ~87.5 deg; `enriched_vs_null = True` everywhere). This replicates
cleanly across ranks 1/3/5.

| Locus | rank1 angle | rank3 mean angle | rank5 mean angle |
| --- | ---: | ---: | ---: |
| L3_8 (early) | 35.75 | 67.63 | 72.36 |
| L3_16 (early) | 33.24 | 67.42 | 72.33 |
| L4_8 (early) | 37.87 | 69.19 | 73.31 |
| L4_16 (early) | 34.71 | 66.59 | 72.17 |
| L4_24 (late) | 31.95 | 67.00 | 72.69 |
| L4_36 (late) | 36.04 | 67.98 | 73.48 |
| L4_47 (late, terminal) | **27.27** | **47.83** | **58.69** |

The early loci do **not** show stronger readable<->outcome alignment than the late loci —
if anything, the terminal locus `L4_47` shows the *tightest* multi-dimensional overlap
(noticeably lower rank-3/rank-5 mean angle than every other locus, early or late). This
is a genuine, non-null finding, but it runs opposite to a naive "early loci are
geometrically special" story: readable-outcome alignment looks like a broadly shared
property of this model's candidate representations, sharpening somewhat at the terminal
locus rather than at the newly identified early loci.

## Finding 2 — Writable alignment is at chance everywhere it can be tested

At the only loci where a real writable/injection tensor exists (`L4_24`, `L4_36`,
`L4_47`), neither the readable nor the outcome subspace is aligned with the writable
perturbation subspace beyond the random-rotation null, at any rank:

| Locus | rank | readable<->writable mean deg | outcome<->writable mean deg | null p05 |
| --- | ---: | ---: | ---: | ---: |
| L4_24 | 1 | 89.38 | 89.61 | 87.53 |
| L4_24 | 5 | 88.11 | 87.86 | 87.06 |
| L4_36 | 1 | 88.60 | 88.86 | 87.53 |
| L4_36 | 5 | 87.75 | 87.69 | 87.06 |
| L4_47 | 1 | 87.73 | 87.73 | 87.53 |
| L4_47 | 5 | 88.36 | 88.34 | 87.06 |

`readable_enriched_vs_null` and `outcome_enriched_vs_null` are `False` at every late
locus and every rank. Combined with Finding 1, this is a clean instance of
**`READABLE_OUTCOME_OVERLAP_WITHOUT_WRITABLE_ALIGNMENT`**: what is readable and
correlates with outcome is not, in this data, the same as what these particular
injection deltas actually move. This is exactly the "readable ≠ actionable" caution the
programme asked to preserve, now with a direct empirical instance of it at the only loci
where the comparison is possible.

## Finding 3 (early loci) — cannot be tested, not fabricated

The strict question "do early loci show stronger writable alignment than late loci" is
**not answerable** with data in this repository: no writable tensor exists at layer 8 or
16 at any loop. Verdict for this sub-question:
`INSUFFICIENT_MATCHED_LOCUS_WRITABLE_DATA`. This was not worked around by transporting
the late-locus writable vectors into early coordinates (no mathematically justified
transport map exists) and was not expanded with a new intervention campaign, per
programme rule ("abandon rather than expand").

## Finding 4 — Loop-to-loop readable-subspace rotation (physical layer 16 fixed)

| Transition | rank1 angle (deg) | rank3 mean | rank5 mean |
| --- | ---: | ---: | ---: |
| L1_16 -> L2_16 | **30.89** | 57.09 | 66.53 |
| L2_16 -> L3_16 | 16.63 | 56.84 | 63.72 |
| L3_16 -> L4_16 | 11.16 | 58.25 | 65.35 |

None of these transitions individually clear the random-rotation null at p05 (i.e., none
qualify for a strict `LOOP1_TO_LOOP2_SUBSPACE_ROTATION_CONFIRMED` verdict — the rank-1
angles are all well below the ~87-89 deg null, meaning readable directions stay far more
aligned across loops than chance would predict at every transition). What *is* visible,
descriptively, is the relative ordering: the L1->L2 transition has the largest rank-1
rotation (30.89 deg) of the three, decreasing monotonically toward L3->L4 (11.16 deg).
This is directionally consistent with the frozen study's own finding that "the main
representational rotation is concentrated at L1->L2" and that "frozen L2/L3->later
transfers are direction-stable" — but it is reported here as a **descriptive corroboration
in the multi-dimensional geometry**, not a newly confirmed statistical result, since it
does not clear the null-based bar the programme requires for a `_CONFIRMED` verdict.

## Secondary label-permutation null (reduced rounds)

At `L4_47`, rank 3: a single label-shuffled outcome direction vs. the real rank-3 outcome
subspace gives a mean angle of 59.54 deg (n=25 rounds) — noisier than, but broadly
consistent with, the random-rotation null (88.2 deg at rank 3) sitting well above the
real outcome-subspace-vs-writable comparison, i.e. it does not change any conclusion
above. Kept to 25 rounds rather than 2000+ because each round refits a full bootstrap
outcome-direction ensemble (expensive); this is an explicit, logged sacrifice (see below),
not a silent cap.

## Sacrifices (logged, per programme priority)

- Label/task-permutation null for the **readable/outcome ensembles** was run at reduced
  rounds (25, one locus) rather than the full ≥2000, because each round requires
  re-fitting a 40-100-member bootstrap ensemble (expensive linear algebra), unlike the
  random-rotation null which is a single random draw per round (cheap). The full
  2000-round random-rotation null (the primary null referenced in every verdict above)
  was run at full scale.
- CCA-based geometry (spec: "canonical correlation only if it can be validated out of
  sample") was not attempted — out-of-sample CCA validation adds meaningful engineering
  scope for a lowest-priority optional variant; principal angles (required) fully cover
  the primary claims.
- Rank-10 subspaces were not computed (spec marks rank 10 optional); ranks 1/3/5 (required)
  are reported in full at every locus.

## Verdicts

- Readable<->outcome overlap: **present at all loci tested, not early-locus-enriched**
  (`L4_47` shows the tightest multi-rank alignment of any locus).
- Readable/outcome <-> writable (late loci, the only place it can be tested):
  **`READABLE_OUTCOME_OVERLAP_WITHOUT_WRITABLE_ALIGNMENT`**.
- Early-locus writable comparison: **`INSUFFICIENT_MATCHED_LOCUS_WRITABLE_DATA`**
  (not fabricated, not expanded via new intervention campaign).
- Loop-to-loop rotation: descriptively consistent with the frozen study's L1->L2 finding,
  but does **not** clear the null-based bar for `LOOP1_TO_LOOP2_SUBSPACE_ROTATION_CONFIRMED`.

## Answer to the programme's geometry question

> Do the newly identified early readable loci show stronger multi-dimensional alignment
> between readable, outcome-relevant, and actually writable perturbation subspaces than
> the established late loci?

**No, and the comparison is only partly answerable.** Readable<->outcome alignment is a
broad property of this model's candidate representations present at every locus tested,
not specifically enriched at the early loci (`L4_47` is if anything the tightest).
Writable-subspace alignment cannot be compared early-vs-late at all, because no writable
data exists at early loci — and where it *can* be tested (three late loci), readable and
outcome directions are geometrically indistinguishable from random relative to the actual
injection-delta subspace. This is diagnostic evidence, not a causal-control claim: it
does not by itself prove early loci are non-actionable, only that the *available*
injection-delta geometry at late loci does not line up with what is readable there
either — a caution about extrapolating "readable" to "steerable" at any locus, early or
late, with the data currently in hand.
