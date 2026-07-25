# Paper 1 v2 Integration Plan

This programme does not edit Paper 1 (`ouro_paper_draft_v3_6.md`) or Paper 2. This file
states what would merit inclusion, at what narrowness, and where, if the user later asks
for the manuscript to be updated. **All three experiments are now complete; numbers
below are final, from `MASTER_RESULTS.md` and each experiment's own `RESULTS.md`.**

## 1. Horizon Logic (second-domain strict pre-answer) → Section 5 addendum

**Proposed location:** immediately after Section 5.2 (GSM8K incremental-AUROC result),
as a new "5.3 A second domain: Horizon Logic" subsection, replacing the
`proto_introspection_second_domain_preflight_2026-06-17.md` conclusion that both prior
candidate second domains (SVAMP, MATH) were rejected at pre-flight.

**Narrowest supported claim (final, from `horizon_logic/RESULTS.md`, verdict
`SECOND_DOMAIN_PREANSWER_REPLICATION`):** "On Horizon Logic — synthetic propositional
entailment tasks with proof_depth 2-4 as an explicit reasoning-horizon control and a
deterministic truth-table verifier — the strict pre-answer protocol used for GSM8K
replicates: a within-domain paired increment (hidden+shortcuts minus shortcuts) of
**+0.141**, task-clustered 95% CI **[+0.004, +0.290]** (170 tasks, 680 candidates, 510
scorable after excluding malformed candidates from the label set). The lower CI bound is
close to zero and the domain has a substantial class imbalance (89% success among
scorable candidates) — this is a real, controls-passing replication, not a
high-precision one. Headroom-normalized increment: 0.340."

**What must NOT be claimed:** that the absolute AUROC equals or should be compared
in magnitude to GSM8K's 0.797 (Horizon Logic: shortcut 0.585, hidden 0.726, combined
0.726); that a lower absolute AUROC constitutes a failed replication; that this one
second domain establishes cross-domain universality; that the malformed-vs-clean
separability finding (hidden AUROC 0.713 for detecting malformed candidates, close to
the 0.726 correctness AUROC) has been fully disentangled from the correctness signal —
control #11 (increment survives after dropping high-malformed-share tasks) rules out
the simplest form of that concern but the nuance should be preserved in any paper text,
not smoothed into a clean claim.

**Figures/tables:** one combined bar chart (shortcut vs. hidden vs. hidden+shortcuts,
GSM8K columns next to Horizon Logic columns) + the paired-increment forest-style
interval, matching the spec's "Minimal figures" #1-2.

## 2. Terminal selection → Section 6/commitment-gap section

**Proposed location:** Section 6 (the forced-choice/commitment-gap section, currently
citing `sel@oracle 0.6250` on both readers with "Pending live-repo confirmation; no
matched random baseline imported" per the paper's own TODO at line ~1546). This
programme's exact-Poisson-binomial matched-random baseline is precisely the missing
piece flagged there.

**Narrowest supported claim (final, from `terminal_selection/RESULTS.md`, verdict
`TERMINAL_SELECTION_ESTABLISHED`):** "On 39 informative, task-disjoint, reward-diverse
Horizon Logic groups, a low-capacity hidden-state selector achieves forced top-1
**34/39 (0.872)** vs. an exact matched-random expectation of **23.0/39 (0.590)**
(Poisson-binomial p=**2.44e-05**), a paired difference of **+0.282** with task-clustered
95% CI **[+0.167, +0.391]**."

**Required companion caveat, not optional:** a shortcut-only control selector (no hidden
state — generated-length/marker-found/hit-max features only) reaches **35/39 (0.897)**,
nominally *better* than the hidden-state selector. 87% of the 39 informative groups
contain at least one malformed candidate, so most of the forced-choice power here is
plausibly "avoid the non-committing candidate," which shortcuts detect about as well as
hidden states do. On the 5 groups where every candidate is well-formed (a genuine
content-quality choice), the hidden selector is 5/5 vs. matched-random 3.75/5 —
suggestive of real content discrimination, but nowhere near powered enough (n=5) to
stand alone. **Any paper text using this result must carry this caveat alongside the
headline number** — do not report the 34/39 vs. matched-random result without also
stating that the shortcut control performs comparably.

**What must NOT be claimed:** that a null result proves selection is impossible in
general (moot here — this is a positive result); that "readable correctness supports
reliable forced selection of the better answer" — the more defensible claim is
"selection beats matched-random, but a large share of that margin is attributable to
detecting non-commitment rather than discriminating well-formed-but-wrong from correct";
that this domain-matched selector result implies the frozen S3B2 selector would be
equally strong out of domain (not tested this run); that this closes the commitment-gap
question raised in Section 6 rather than narrowing it with one additional, properly-
powered, appropriately-caveated data point.

**Figures/tables:** observed-vs-matched-random bar/interval plot; a
confidence-vs-coverage or MRR/top-k-oracle-retention table, matching "Minimal figures"
#3-4.

## 3. Early-layer geometry → Section 8/9 (readout-control boundary)

**Proposed location:** Section 8/9, as a new subsection after the existing frozen-branch
steering results, using the already-integrated early-layer localization result (from
`cross_loop_early_layer_taps_20260720/`, already noted in that section per
`PAPER1_INTEGRATION_NOTE.md`) as the jumping-off point for this run's *geometric*
diagnostic.

**Narrowest supported claim (from `subspace_geometry/RESULTS.md`, COMPLETE):**
"Readable and outcome-relevant subspaces are substantially aligned (well beyond a
random-rotation null) at every locus tested, early and late alike — the terminal locus
L4_47 shows the tightest multi-dimensional alignment of any locus, not the newly
identified early loci. At the three late loci where a real writable/injection-delta
tensor exists, neither the readable nor the outcome subspace is aligned with the
writable subspace beyond the random-rotation null at any tested rank — a direct
instance of `READABLE_OUTCOME_OVERLAP_WITHOUT_WRITABLE_ALIGNMENT`. No writable tensor
exists at any early locus (layer 8 or 16) in this repository, so the early-vs-late
writable-alignment comparison itself is `INSUFFICIENT_MATCHED_LOCUS_WRITABLE_DATA` —
this was not manufactured via a new intervention campaign. Loop-to-loop rotation
(physical layer 16, fixed) shows the L1→L2 transition with the largest rank-1 rotation
of the three tested transitions, directionally consistent with (but not a new
statistical confirmation of) the earlier frozen-transfer finding that the main
representational rotation concentrates at L1→L2."

**What must NOT be claimed:** that this geometric misalignment *explains* or *proves*
why frozen steering fails (Section 8's existing causal-intervention results are the
actual evidence for that); that geometric alignment (where found) would itself imply
actionability; that the early loci are geometrically special (the data says the
opposite for readable-outcome overlap, and the writable comparison is simply
unavailable at the early loci, not favorable).

**Figures/tables:** null-normalized alignment-by-locus bar chart; loop-to-loop subspace
distance line chart, matching "Minimal figures" #5-6.

## Cross-cutting language changes if the manuscript is edited later

- **Abstract/conclusion:** at most one added clause noting a second-domain replication
  attempt and its qualified outcome; do not upgrade any existing universal claim.
- **Evidence-status map:** add three new rows (Horizon Logic pre-answer, terminal
  selection powered test, early-locus geometry) with their specific verdict strings from
  `MASTER_RESULTS.md`, not paraphrases.
- **Limitations:** the geometry audit is diagnostic, not causal; the terminal-selection
  result is domain-matched (Horizon Logic-trained selector), not evidence the frozen
  S3B2/DualAnchor/CoreContent selectors would perform the same way out of domain.
- **Readout-control boundary interpretation:** all three verdicts together **strengthen**
  the boundary framing, not weaken or leave it unchanged. The geometry audit gives a
  direct empirical instance of readable-outcome overlap without writable alignment at
  the late loci (Finding 2). The terminal-selection result adds a complementary,
  more surprising instance from a different angle: even where selection *does* beat
  matched-random, a shortcut-only control matches or exceeds the hidden-state selector,
  and most of the power traces to detecting non-commitment rather than genuine
  content-quality discrimination — i.e., the readable signal that helps here is closer
  to "did the process complete" than "is the content good," which is itself a further
  narrowing of what "readable" buys you under forced commitment. The Horizon Logic
  replication (Section 5) is the one result in the opposite rhetorical direction — it
  shows the pre-answer readout side of the boundary generalizes to a second domain —
  but it is a within-domain incremental-validity claim, not a control/actionability
  claim, so it does not cut against the boundary framing; it strengthens the "readable"
  half while the other two results strengthen the "not straightforwardly controllable"
  half.

## Explicitly forbidden language (repeated from the run's evidence-discipline rules)

Do not write, in any paper edit motivated by this programme's results: "the model uses
its own signal"; "early readability proves actionability"; "geometric alignment proves
causal control"; "failure to select proves selection is impossible"; "a lower absolute
logic AUROC is a failed replication"; "this establishes cross-domain universality."
