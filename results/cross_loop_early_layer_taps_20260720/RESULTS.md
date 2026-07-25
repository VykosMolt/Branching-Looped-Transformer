# Results — cross-loop early-layer tap experiment (2026-07-20)

**Run verdict:** `READABILITY_SHIFTS_EARLIER_WITH_RECURRENT_REFINEMENT`
(with `EARLY_READABLE_DIRECTION_STABLE` for L2+ source loops and
`EARLY_READABLE_DIRECTION_ROTATES` for L1 source loops).

Readout-only. No steering, no intervention, no backbone training, no paper edit, no
existing artifact overwritten.

## One-paragraph answer

After recurrent refinement, Ouro-RLTT's CoreContent process-quality signal **is** readable
earlier in the shared physical stack. At the first loop (L1) a linear quality tap at
physical layers 8, 16, and 24 is weak (macro top-1 0.41–0.44, barely above the label-shuffle
floor). By the third and fourth loops the **same early layers** reach 0.61–0.63 — within
noise of the layer-24 readout at the same loop (0.618) and of the historical L4_36 (0.647)
and L4_47 (0.620) references. The direction that carries this signal is **coordinate-stable
across loops once at least one full pass has completed**: a tap trained at L2 or L3 and
applied frozen to a later loop of the same physical layer retains 97–100% of the
locally-refit performance with Spearman score correlation 0.90–0.99. The exception is the
first loop: an L1-trained tap transferred to later loops loses 0.11–0.23 macro top-1
(Spearman 0.53–0.85) — the representation rotates sharply across the L1→L2 transition, then
stabilizes. So the readable quality direction both moves earlier in depth and becomes
loop-transferable, but only after the first recurrent iteration.

## Setup (as executed)

- **Target:** CoreContent v2 corrected/deduped candidate-quality ranking, 5 core domains
  (coding, reasoning, math, logic, alignment). Existing deterministic task/prompt-disjoint
  split; **0 task_uid crossings** among the 250 train / 60 val / 120 heldout groups per
  domain (`split_integrity.json`; 1,250 / 300 / 600 groups total). 195 hendrycks_math
  task-ID-colliding tasks excluded (`excluded_crossing_tasks.json`).
- **Features:** one frozen forward per candidate (`use_cache=False`, 4 UT steps, no early
  exit), capturing physical layers {8, 16, 24, 36, 47} × loops {L1–L4}, masked-mean pooled,
  `[5,4,2048]` fp16. 8,592 candidates. Model `models/ouro_rltt_local` (config sha
  `7d6764dbc8210d02`), Transformers 4.54.1.
- **Tap:** `AntisymLinearNoNorm` (bias-free, exactly antisymmetric), trained by the locked
  v2 `train_pairwise` procedure; 36-point grid (6 training-sets × 2 lr × 3 seed) selected on
  **val** macro top-1 only. Scoring = the locked v2 per-candidate mean-pairwise-margin,
  z-normalized within group (verified bit-exact against `cc.policy_candidate_scores`).
- **Statistics:** task-clustered bootstrap, 10,000 draws, seed 20260720, identical draws for
  every paired comparison. Matched chance = tie-aware E[top-1] per group (macro 0.282).

## Primary result 1 — local-refit matrix (Mode B): readability moves earlier over loops

Heldout macro top-1 (task-clustered 95% CI); every cell's CI excludes matched chance
(0.282) with p(>chance)=1.0.

| layer \ loop | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| **8**  | 0.410 [.373,.447] | 0.568 [.531,.605] | **0.633 [.597,.670]** | **0.622 [.585,.658]** |
| **16** | 0.440 [.412,.469]* | 0.583 [.547,.620] | **0.617 [.580,.653]** | **0.628 [.592,.665]** |
| **24** | 0.428 [.392,.465]* | 0.568 [.531,.605] | 0.610 [.573,.647] | 0.618 [.580,.655] |

References: **L4_36 = 0.647 [.610,.683]**, **L4_47 = 0.620 [.583,.657]**. *(CI shown per
cell; values are point ± bootstrap.)*

- **Loop trend is strong and significant at every layer:** L4−L1 = +0.212 (layer 8),
  +0.188 (16), +0.190 (24); all CIs exclude 0.
- **EARLY_READABLE cells** (above chance AND within 0.03 of L4_24, both loops ≥ L3):
  **8_L3, 8_L4, 16_L3, 16_L4** all qualify. 8_L2 and 16_L2 are above chance but ~0.04–0.05
  short of L4_24, so they do not clear the strict margin.
- The best late-loop early-layer cell (8_L3 = 0.633) is statistically tied with L4_24
  (Δ +0.015, CI [−0.022, +0.050]) and with L4_36 (Δ −0.013, CI [−0.050, +0.023]).
- **Not late-emergent:** the signal is present at layer 8 by L3, not confined to 24/36/47.

## Primary result 2 — frozen-transfer matrix (Mode A): direction is stable after L1

Per physical layer, tap trained at source loop u, applied **frozen** to target loop v>u.
Δ vs target-local refit and Spearman score correlation on identical heldout candidates:

| transfer | target top-1 | Δ vs refit | Spearman | pairwise sign | label |
|---|---|---|---|---|---|
| 8: L1→L4 | 0.410 | −0.212 | 0.527 | 0.713 | ROTATED |
| 8: L2→L4 | 0.628 | +0.007 | 0.939 | 0.928 | **STABLE** |
| 8: L3→L4 | 0.628 | +0.007 | 0.914 | 0.924 | **STABLE** |
| 16: L1→L4 | 0.497 | −0.132 | 0.775 | 0.820 | ROTATED |
| 16: L2→L4 | 0.628 | 0.000 | 0.951 | 0.933 | **STABLE** |
| 16: L3→L4 | 0.630 | +0.002 | 0.985 | 0.967 | **STABLE** |
| 24: L1→L4 | 0.510 | −0.108 | 0.771 | 0.837 | ROTATED |
| 24: L2→L4 | 0.610 | −0.008 | 0.903 | 0.915 | **STABLE** |
| 24: L3→L4 | 0.637 | +0.018 | 0.954 | 0.954 | **STABLE** |

(Full 18-transfer matrix in `train_eval_results.json` / `bootstrap_stats.json`.)

- **All 9 L2+ → later transfers are DIRECTION_STABLE** at every layer (within 0.03 of target
  refit, Spearman 0.90–0.99). Every transplanted tap reproduced its source-loop score
  exactly before target evaluation (hard assertion), and every frozen weight hash was
  unchanged across evaluation.
- **All 9 L1 → later transfers are ROTATED_BUT_READABLE**: target-local refit succeeds but
  the frozen L1 direction underperforms it materially and correlates less (Spearman as low
  as 0.53 at layer 8). The coordinate rotation is concentrated in the L1→L2 transition.
- **Mechanistic reading:** shared weights across loops are semantically stationary only from
  L2 onward; the first pass uses a distinct coordinate system for the same information.

## Controls

- **Extraction path validated** (`extraction_controls.json`): fresh vs cached-production
  features on 2,434 identical heldout candidates — min per-cell cosine **0.998** at layers
  24/36/47; the locked v2 policy (24_L4+36_L4) scores **0.687 cached vs 0.685 fresh** on the
  same subset (Δ 0.002). Layers 8/16 are confirmed genuinely new loci (cosine <0.999 vs any
  cached layer).
- **Label-shuffle** (rewards permuted within groups, full grid rerun): 8_L1 0.350, 8_L4
  0.427, 16_L2 0.403, 24_L4 0.337 vs matched chance 0.282. The shuffle floor sits modestly
  above naive chance — driven by the alignment domain's 2-candidate 0.5 pairwise floor and
  36-point val-selection optimism — but the real cells beat their shuffle counterparts by
  +0.06 (weak 8_L1) up to +0.28 (strong cells 8_L4/16_L2/24_L4). Crucially the weak L1
  early-layer cell is barely above its own shuffle, confirming L1 early readability is
  genuinely near-absent, while the strong late-loop cells are far above shuffle.
- **Length baseline** (`length_baseline.json`): ranking by candidate text length gives macro
  0.323 (best-signed 0.370) — the taps are not length proxies.
- Reproduction, zero-crossing, alignment, frozen-head immutability, deterministic-refit,
  no-heldout-tuning, and bootstrap-clustering checks: **20/20 tests pass**
  (`code_tests/test_results.json`).

## Secondary readout — S3B2 generated-branch correctness (exploratory, small-n)

16 tasks / 160 saved generated branches; leave-one-task-out local refit + frozen transfer of
the CoreContent taps. **Explicitly under-powered** (coding has 0 positives).

- **Local refit corroborates the loop/layer shift:** LOTO AUROC rises with loop and layer —
  layer 16 reaches 0.654 (L3) / 0.651 (L4), layer 24 L4 = 0.681, layer 47 L4 = 0.704; layer
  8 stays weak (0.49–0.58). So the "later loops read better" pattern reproduces at layer 16,
  though layer 8 does not carry generated-branch correctness.
- **Frozen CoreContent direction does NOT transfer to generated branches:** transfer AUROC
  0.35–0.48 (near/below 0.5) at every cell — consistent with the known v2 finding that the
  CoreContent tap is a corruption/quality detector rather than a plausible-wrong-branch
  ranker. This is a distribution-shift result, not evidence against the primary finding.

## What this does and does not establish

- **Does:** process-quality information is linearly readable at physical layers 8 and 16
  during loops L3–L4 at parity with the historical mid/late 24/36/47 basis; the carrying
  direction is loop-stable from L2 onward; readability increases monotonically across loops.
- **Does not:** no intervention was performed. 8_L3, 8_L4, 16_L3, 16_L4 are labelled
  `EARLY_ACTION_SURFACE_CANDIDATE (readout-only)` — they are readable, at shallow depth, with
  substantial same-loop downstream depth remaining (40 and 32 layers) — but nothing here
  shows they are controllable, actionable, or steerable. The detection→selection gap is
  untouched.

## Figures

- `figures/fig1_local_refit_heatmap.{svg,png}` — loop × layer readability heatmap.
- `figures/fig2_transfer_matrices.{svg,png}` — per-layer transplant + rotation-gap matrices.
- `figures/fig3_readability_vs_depth.{svg,png}` — readability vs remaining same-loop depth.
- `figures/fig4_transplant_gap.{svg,png}` — target-refit minus frozen transplant by pair.
