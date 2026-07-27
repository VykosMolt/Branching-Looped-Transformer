# Powered Thinking-checkpoint pre-answer replication (SEALED PLAN)

Sealed: 2026-07-27 ~22:40, BEFORE any generation. Author: Claude (session 1fd122fa),
authorized by the user relaying Sol's design (2026-07-27) with three corrections
incorporated: operational SESOI justification, full predeclared verdict logic, and
a formal simulation-based sample size.

## Question

Does the strict pre-answer increment replicate on Ouro-2.6B-Thinking? The pilot
(`thinking_preanswer_v3_20260726`, sealed verdict `THINKING_PREANSWER_NULL`) was
inconclusive/underpowered: +0.027, 95% CI [−0.13, +0.21], ≈20 held-out negatives.
This run is the powered, sealed successor. It is a pre-answer readability
experiment; the separate C2 allocation null neither informs nor is informed by it
beyond rate estimation for sizing.

## Design (all fixed before launch)

- Model, prompts, sampling, seeds, strict cut, and the 448-token cap: identical to
  the pilot (matched protocol; the cap is deliberately retained despite its known
  malformedness cost on Thinking).
- Tasks: hash-ordered pool offsets **[1500, 2400)** — 900 tasks, disjoint from every
  prior run ([0,1500) fully consumed per the task-slice ledger). k = 4.
- Splits: sealed split_for(task_uid), task-level, deterministic.
- PRIMARY endpoint: new-only cohort incremental AUROC, combined minus the
  adversarial 5-shortcut composite (includes the malformed-sibling count), paired
  task-clustered bootstrap 2000 rounds seed 20260727; 95% CI for the
  replication/negative verdicts and 90% CI for TOST.
- SECONDARY (predeclared, never described as independent replication): pooled with
  the pilot cohort, as a precision estimate.
- Sealed malformedness controls: (i) drop-tasks-with-≥75%-malformed control arm
  (the pilot's rule, fixed here before generation); (ii) hidden-only
  malformedness-prediction AUROC diagnostic.
- Stage-A gate: the analysis must first reproduce the pilot's published increments
  and intervals to 1e-9 from preserved artifacts (verified passing at seal time,
  |d| = 0.0).

## Sample size (formal simulation, sealed with the plan)

`bg_v5_thinking_power_sim.py`, 500 reps/cell, seed 20260727, plug-in task-level
resampling of the pilot's raw held-out predictions through the EXACT task-clustered
estimator; alternative injected by calibrated positive-class shift (achieved
population increment +0.09523); results in `sizing_simulation_v5.json`:

| rates | N | E[heldout neg] | median 95% half-width | power(+0.095) | P(TOST | true 0) |
|---|---|---|---|---|---|
| pilot | 600 | 71 | 0.088 | 0.642 | 0.000 |
| **pilot** | **900** | **106** | **0.072** | **0.816** | 0.000 |
| pilot | 1200 | 141 | 0.063 | 0.906 | 0.012 |
| C2-d4 sensitivity | 900 | 146 | 0.063 | — | — |

**N = 900** is sealed: power ≥ 0.8 at the RLTT-replication-sized alternative within
the overnight budget (~9.8 h at the pilot's measured 39 s/task). C2-d4 rates enter
only this sensitivity row (its d4 arm is protocol-matched; d1–d3 are not and are
excluded from all sizing).

## SESOI and its justification (operational, per Sol's correction)

Smallest effect of interest: **±0.05 AUROC** — not "half the RLTT effect" (it is
~76% of GSM8K's +0.066); it is sealed as the smallest increment that would
materially change the paper's cross-checkpoint conclusion, i.e. an effect below it
would leave the "established on RLTT; open elsewhere" framing intact either way.

**Sealed honesty note on equivalence power:** at every feasible N the 90% CI
half-width floor (~0.052–0.060) exceeds the ±0.05 margin, so P(TOST passes | true
effect 0) ≈ 0–0.08. A true null will therefore almost certainly return UNRESOLVED,
not PRACTICAL_EQUIVALENCE. The equivalence verdict is retained in the logic but this
run is not powered to reach it; establishing genuine absence at this margin would
need N ≳ 1600. No verdict may be re-labelled in light of this note.

## Sealed verdict logic (checked in this order)

1. NEGATIVE_TRANSFER: 95% CI upper bound < 0.
2. POSITIVE_REPLICATION: 95% CI lower bound > 0; flag PRACTICALLY_SMALL = true if
   the point estimate < +0.05 (reported as "positive but practically small", not a
   full-magnitude replication). If the TOST condition also holds, the verdict
   remains POSITIVE_REPLICATION with the flag.
3. PRACTICAL_EQUIVALENCE: 90% CI entirely within [−0.05, +0.05].
4. UNRESOLVED: none of the above.

No stopping rules: generation runs to the sealed 900 tasks regardless of interim
class counts; the analysis runs once. No budget, slice, seed, shortcut-set, or
margin changes after launch.

## Recorded implementation note (pre-launch, not an amendment to endpoints)

The sealed pilot generator hard-codes its own slice guard (MAX_OFFSET_EXCL = 170)
and OUT_ROOT. The v5 wrapper overrides these two module attributes at call time —
the sealed file itself is untouched — and imposes its own strictly stronger guard
(offset ≥ 1500). Generation is sharded (150-task resumable chunks).
