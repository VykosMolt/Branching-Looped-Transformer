# V3 overnight programme — results summary (2026-07-26, run 02:52–08:49)

Three pre-registered experiments, all completed, all gates passed.
Plans were written before execution (EXPERIMENT_PLAN.md in each run root).

## 1. Horizon Logic power extension — horizon_power_v3_20260726
VERDICT: REPLICATED_AND_ROBUST_TO_MALFORMED_SIBLING_SHORTCUT

- Stage A gate: published v2 numbers reproduced to <1e-9 before any new data touched.
- 510 new tasks (offsets 170–680 of the sealed hash-sorted pool), 2,040 candidates,
  protocol byte-identical (path-only wrapper over the sealed generator).
- Heldout negatives 19 -> 84 (pooled).
- New-only (independent replication, disjoint tasks):
  4-shortcut +0.0953 [ +0.0376, +0.1569 ];  5-shortcut (adversarial, incl.
  malformed-sibling count) +0.0910 [ +0.0345, +0.1531 ] — both exclude zero.
- Pooled: 4sc +0.1114 [ +0.0563, +0.1689 ]; 5sc +0.1074 [ +0.0528, +0.1643 ].
- Calibration: v2 point estimate (+0.141) was high; powered estimate ≈ +0.11.
- Controls: shuffled-label 0.5415; LOTO max influence 0.0081; split == split_for(uid)
  for all 2,720 records; v2/v3 task sets disjoint.

## 2. Within-family replication — family_xloop_v3_20260726
The §4.4 pattern (early layers weak on pass 1, parity by loops 3–4) replicates on
ALL THREE additional checkpoints, on the identical sealed 2,150-group subset:

- base26  (Ouro-2.6B):        8_L1 0.415 -> 8_L4 0.630; 16_L1 0.430 -> 16_L4 0.648
- thinking26 (2.6B-Thinking): 8_L1 0.408 -> 8_L4 0.613; 16_L1 0.422 -> 16_L4 0.623
- ouro14b (1.4B, 24 layers, mapped cells {4,8,12,18,23}): every early layer
  0.46–0.47 at L1 -> 0.60–0.62 by L3/L4; all trend contrasts exclude zero.
  Scale nuance: 1.4B's early-loop cells EXCEED its late refs (18_L4 0.568,
  23_L4 0.560) — the late-basis-as-gold-standard framing is 2.6B-specific.
- Frozen RLTT->sibling tap transfer (2.6B geometry): |delta| <= 0.028 at every
  cell on both siblings; score correlations 0.97–0.99 (Pearson/Spearman).
  The readable geometry is family-shared, not installed by RLTT.

## 3. Huginn depth-recurrence probe — huginn_probe_v3_20260726
Out-of-family (3.5B depth-recurrent, prelude/core/coda, hidden 5280), same subset,
sealed tap machinery on the recurrence-step axis (steps 1..8, one forward each):

- Readability RISES with recurrence depth: step 1 macro top-1 0.332 -> step 8
  0.438 (chance 0.282). Step8−Step1 = +0.107 [ +0.058, +0.157 ], excludes zero.
- Magnitude is smaller than Ouro's (0.44 vs 0.63 at depth) — reported as a
  qualitative class-generality result, not a quantitative match.
- Step-transfer: step1-sourced taps transfer to later steps at parity or with
  small losses (only L1->L5 excludes zero at −0.06) — Huginn shows no sharp
  Ouro-style L1 rotation; its step-1 representation is already
  direction-stable. A real architectural difference, reported as such.

## Paper-1 v3 integration (manuscript work, separate pass)
- §5 / abstract: powered Horizon increment (+0.11 [+0.05, +0.17] pooled 5sc;
  new-only +0.09) replaces the wide v2 interval; malformed-sibling caveat resolved.
- §4.4 + new subsection: family replication table; RLTT-tap transfer; 1.4B nuance.
- New §4.x or appendix: Huginn probe as class-generality evidence.
- Limitations rewrite: "single frozen model" -> "single family + one
  out-of-family depth-recurrent replication of the trend".

## 4. Thinking pre-answer attempt — thinking_preanswer_v3_20260726 (addendum)
VERDICT: THINKING_PREANSWER_NULL (sealed labels; correct reading = UNRESOLVED).
Same task slice/protocol as RLTT main; sealed max_new=448 held. Malformed rate
0.4559 vs RLTT 0.2500 -> 370 scorable, 130 heldout, ~20 negatives. Increments
+0.0168 [-0.1440,+0.2061] (4sc), +0.0273 [-0.1285,+0.2106] (5sc): the interval
contains both zero and the RLTT-sized effect. Reported in §5.4/§10.2/§11 as an
attempted, underpowered cross-checkpoint test; a pre-registered longer-budget
re-run is the identified follow-up. No post-hoc budget change was made.
