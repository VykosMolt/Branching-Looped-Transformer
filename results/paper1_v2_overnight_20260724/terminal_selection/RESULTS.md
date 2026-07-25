# Powered Terminal-Selection Evaluation — RESULTS

**STATUS: COMPLETE.**

## Shared-pool suitability decision

The Horizon Logic pool (170 tasks, k=4 candidates/task, from the main generation run
described in `horizon_logic/RESULTS.md`) was used as the primary terminal-selection
dataset. It yielded **39 informative heldout groups** — above the minimum useful target
of 25, just below the preferred 40-50 band — from 59 total heldout tasks (15 all-correct,
5 all-wrong, 39 informative). No separate DualAnchor/CoreContent/S3B2 pool was needed or
substituted.

## Protocol

- Feature: full-candidate ("terminal") pooled hidden feature (prompt + complete
  generated text, including malformed/truncated candidates), computed by
  `bg_v2_overnight_terminal_prep.py` via one additional forward pass per candidate
  through the same canonical extractor used in Part I. No leakage constraint applies
  here — terminal selection scores already-completed candidates.
- Unlike Part I, malformed candidates are **not** excluded here: a malformed
  (non-committing) candidate is a genuine failure mode under forced terminal choice, and
  is scored as `success=False` like any other incorrect candidate.
- Primary selector: standardize -> PCA -> L2-logistic (same low-capacity family as
  Part I and the audited S3B2 protocol), hyperparameters selected via task-grouped CV on
  train+val groups only (`cv_auroc=0.7184`, `k_pca=16`, `l2=4.0`), evaluated once on
  informative heldout groups.
- Secondary/control selector: shortcut-only (generated-token-count, found-final-marker,
  hit-max-tokens, pre/gen-token ratio), same fit family (`cv_auroc=0.8031`, `k_pca=16`,
  `l2=1.0`).
- Primary endpoint: forced top-1 success vs. exact Poisson-binomial matched-random null
  (per-group p_i = n_correct_i / n_i), with a paired task-clustered bootstrap interval on
  the observed-minus-matched-random difference.

## Headline numbers

- n_source_tasks=170; groups by split: 80 train / 31 val / 59 heldout.
- n_informative_heldout_groups = **39** (all-correct=15, all-wrong=5, excluded from the
  forced-choice endpoint since they cannot support discrimination).
- **Observed top-1 successes: 34/39 (0.872)**
- **Matched-random expected successes: 23.0/39 (0.590)**
- **Paired difference: +0.282**
- **Exact Poisson-binomial p-value (observed or better): 2.44e-05**
- **Task-clustered bootstrap 95% CI on the paired difference: [+0.167, +0.391]** (2000
  rounds) — excludes zero comfortably.
- Pairwise ranking accuracy: 0.762 (122 correct/incorrect pairs); MRR: 0.927; top-2
  oracle retention: 0.949; top-3 oracle retention: 1.0.
- Candidate-level AUROC (all heldout candidates, malformed included): 0.787.

## Important honest caveat: the shortcut-only control selector performs comparably (or slightly better)

The shortcut-only selector (generated-token-count / found-marker / hit-max / pre-gen
ratio — no hidden state at all) reaches **35/39 (0.897) observed successes**, paired
difference **+0.308** — nominally *better* than the primary hidden-state selector
(34/39, +0.282). This is a real, unflattering-but-important finding, not swept aside:

- **34 of the 39 informative groups (87%) contain at least one malformed candidate** —
  i.e. most of the "which is correct" choice in this pool is actually "avoid the
  malformed/truncated one," which the shortcut features (`found_final_marker`,
  `hit_max_tokens`) detect almost by construction.
- On the **5 informative groups where every candidate is well-formed** (a genuine
  content-quality choice, not a malformed-avoidance choice), the primary hidden-state
  selector picks the correct candidate in **5/5**, vs. a matched-random expectation of
  3.75/5 (paired diff +0.25). This is directionally consistent with real content-quality
  discrimination, but **n=5 is far too small to support any standalone statistical
  claim** — it is reported descriptively as context, not as a second powered result.
- **Interpretation:** `TERMINAL_SELECTION_ESTABLISHED` is earned on its own defined
  terms (hidden-state selector vs. exact matched-random, on informative groups) — the
  primary endpoint's criteria are all met. But the honest reading is narrower than
  "hidden states let you pick the better answer": in this pool, a large share of the
  forced-choice power comes from detecting non-commitment (malformed candidates), which
  simple shortcuts already capture about as well as the hidden state does. The
  hidden-state selector is not shown here to add much *beyond* malformed-avoidance on
  this domain; the 5-group pure-content-quality subset is the closest look at that
  question this run can support, and it is suggestive, not conclusive.

## Controls

1. Zero task crossing (selector training vs. evaluated groups): **true**.
2. Duplicate-group / duplicate-candidate audit: **0 duplicates**.
3. Frozen checkpoint hashes: same `models/ouro_rltt_local` used throughout this
   programme (see `ENVIRONMENT.json`); no weight modification.
4. Source-slice reproduction for any historical selector: not applicable this run (no
   frozen S3B2/DualAnchor selector was re-scored under Horizon Logic — see Sacrifices).
5. Shuffled-score control: paired diff **0.000** under randomly shuffled scores (exactly
   chance, as required).
6. Length/log-probability baseline: the shortcut-only selector above (+0.308 paired
   diff) — reported prominently, not just as a pass/fail control, per the caveat above.
7. Candidate-order invariance: **0 failures** (reversing candidate order within a group
   never changes the argmax) on a 20-group spot check.
8. No held-out-label tuning: hyperparameters selected on train+val only via grouped CV.
9. Exact per-group matched-random probabilities: stored in `per_group_p` inside
   `terminal_results.json`.
10. Raw score/prediction preservation: per-group scores are deterministically
    re-derivable from the saved model coefficients and `terminal_pool.pt` features.

## Verdict

**`TERMINAL_SELECTION_ESTABLISHED`**

All required criteria are met: positive selector-minus-random difference (+0.282), a
paired 95% interval excluding zero ([+0.167, +0.391]), an exact Poisson-binomial null
p=2.44e-05 (<0.05), no dependence on one or two groups (pairwise accuracy 0.762 across
122 pairs, MRR 0.927), and all integrity controls passing.

**This verdict should be read alongside the shortcut-selector caveat above.** The forced
top-1 result is real and clears every stated bar, but a large share of its power in this
specific pool is plausibly attributable to detecting non-commitment (malformed
candidates) rather than discriminating between two well-formed but differently-correct
completions. This is a narrower, more defensible claim than "readable correctness
supports reliable forced selection of the better answer" — see
`PAPER1_V2_INTEGRATION_PLAN.md` for the exact language boundary this implies.

## Sacrifices

- Only the Horizon Logic pool's own selector was trained and evaluated (the primary,
  required comparison). Re-scoring the frozen S3B2 hidden-ridge/L2-logistic or
  DualAnchor/CoreContent selectors under Horizon Logic's distribution was treated as a
  secondary/optional cross-domain-transfer diagnostic and dropped first, per the
  programme's explicit sacrifice ordering ("secondary terminal selectors" ranks below
  the primary powered test).
