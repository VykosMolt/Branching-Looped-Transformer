# Paper Pending-Items Resolution — Live-Repo Artifact Pinning Pass

**Draft in:** `ouro_paper_draft_v3_6.md` → **Patched:** `ouro_paper_draft_v3_7_pending_items_pinned.md`
**Run:** 2026-07-03/04 · **Scope:** artifact pinning only — no training, no checkpoint edits, no artifact
deletion/moves, no expensive regeneration, no new scientific claims. Numbers already in the draft were
verified against live artifacts and either pinned (with exact paths) or marked `NOT_FOUND`; nothing
uncertain was upgraded to verified.

---

## 1. Executive Summary

**Items verified & pinned (7 of 9 artifact items):** 1 (strict pre-answer GSM8K), 2 (S3B2 correctness),
3 (orthogonality null audit), 4 (S1 K-matched frozen-fork), 5 (seven-method steering closure),
8 (pre-DualAnchor scaffold lineage), 9 (branch/carry substrate). All numbers matched the draft exactly;
where both MD and JSON exist they agree. **Item 10 (citations): all 10 arXiv IDs web-verified.**

**Items marked `NOT_FOUND` (2), preserved as historical/user-confirmed, NOT upgraded:**
6 (base-model 24% training-stage localization eval) and 7 (2.7× / 47-of-100 math-transfer origin). Both
were searched thoroughly on 2026-07-03/04 and are absent from the live artifact tree and docs (evidently
purged, as the operator anticipated). The scientific claims are left intact but explicitly flagged as not
artifact-backed.

**Item 11 (second powered domain): feasibility only — not run.** A live negative pre-flight already
exists (`REJECTED_AT_PREFLIGHT_BOTH_CANDIDATES`); see §"Second powered pre-answer domain feasibility".

**One notable correction:** the S3B2 evaluation **does** contain its own matched random-selection control
(`RANDOM` sel@oracle **0.5833**), which v3.6 stated it did not. This is the S3B2 pool's *own* control, not
the forbidden S3B1 0.58 import (they coincide only because both refit over the same pool). It makes the
selection wall starker, and is added with airtight provenance.

**One provenance bug fixed:** v3.6 pointed the 0.951 two-adapter convergence at `bg_steering_suite_2026-05-18`;
the real source is `bg_sequence_level_adapter_2026-05-18/diagnostics.json`.

**arXiv blockers remaining (materially affect strength, not just presentation):**
- **B1.** Base-model 24% localization (§3.7) is a *headline* result but has **no live artifact**
  (`NOT_FOUND`). Either re-run to pin a base-vs-Thinking-vs-RLTT artifact, or soften the "one of the
  strongest results" framing. **Highest-risk remaining item** because it is featured, not a caveat.
- **B2.** Single powered pre-answer domain. The pre-flight shows a clean second domain is genuinely hard
  for this model; decision needed (see feasibility note). Non-blocking if the narrow claim is accepted.
- **B3 (minor, out of scope here).** 512-slice HH antisymmetrization (0.9668/0.6367) not pinned in this
  pass; a companion audit artifact exists and should be checked next.

**Not blockers:** math-transfer origin (already non-load-bearing), citation metadata (done), all of
items 1/3/4/5/8/9 (pinned).

---

## 2. Status Table

| Pending item | Status | Primary artifact path | Draft action | Risk |
|---|---|---|---|---|
| **1. Strict pre-answer GSM8K** | ✅ VERIFIED (MD+JSON agree) | `artifacts/reports/proto_introspection/proto_introspection_within_domain_preanswer_specificity_2026-06-17.{md,json}` | §5.2 + App I: replaced TODO with pinned paths; cut definition confirmed | **Low** — safe as primary evidence |
| **2. S3B2 correctness + random baseline** | ✅ VERIFIED; **matched RANDOM baseline found (0.5833)** | `…/final_prewrite_hardening_2026-06-17/s3b2_..._refit_2026-06-17.{md,json}`; `…/final_engineering_expansion_2026-06-17/s3b2_..._expanded_2026-06-17.{md,json}` | §6.3/§6.4/App E.2: corrected "no baseline" → native RANDOM 0.5833 w/ provenance | **Low–Med** — sanity-check the 0.5833↔0.58 coincidence note |
| **3. Orthogonality null audit** | ✅ VERIFIED (full precision + draw counts) | `…/s1_s3_exact_injection_orthogonality_2026-06-17/s1_s3_exact_injection_orthogonality_null_audit_2026-06-17.{md,json}` | §8.4 + App J: draw counts (1e6 / 10k / 10k) resolved; kept as failed explanation | **Low** — non-load-bearing by design |
| **4. S1 K-matched frozen-fork** | ✅ VERIFIED (live-run) | `artifacts/reports/probes/mpn_s1_baseline_2026-06-13/s1_report_2026-06-17.md` (+ `s1_4b_kmatched_sampling.json`, `s1_4a_fork_param_screen.json`, `s1_4_reference_loop.json`) | §8.2 + App G: exact figures 0.75/0.611/0.0 + params pinned | **Low** |
| **5. Seven steering methods** | ✅ VERIFIED; provenance corrected | `docs/evaluator/history/steering-and-adapters/steering-consolidation.md` §2; `bg_causal_intervention_adapter_2026-05-18/analysis.json`; `bg_sequence_level_adapter_2026-05-18/diagnostics.json` | §8.1 + App H: 0.951 pointer fixed; enumeration + cosines pinned | **Low** |
| **6. Base/Thinking/RLTT localization** | ⚠️ `NOT_FOUND` (base 24% / RLTT 95.0% eval); 95.2% is historical Kirin | — (no live artifact) | §3.7 Table 1: provenance note; §10.1 row + checklist marked NOT_FOUND / user-confirmed | **HIGH** — featured result, not artifact-backed |
| **7. Math-transfer 2.7× / 47-of-100** | ⚠️ `NOT_FOUND` (purged) | — (no live artifact) | §3.6/§1.5: "unarchived" → explicit NOT_FOUND; kept non-load-bearing | **Low** — already non-load-bearing |
| **8. Pre-DualAnchor scaffold** | ✅ VERIFIED (doc/snapshot) | `docs/evaluator/branch-generation-and-survival.md` (L36–63) | Already snapshot-backed; numbers confirmed exact | **Low** |
| **9. Branch/carry substrate** | ✅ VERIFIED (doc) | `docs/evaluator/kv-cache-branch-carry.md` | Confirmed; compute savings kept appendix-only | **Low** |
| **10. Citations** | ✅ VERIFIED (web, 10/10) | arXiv + `huggingface.co/ByteDance/Ouro-2.6B-Thinking` | References: author lists + HF URL resolved | **Low** |
| **11. Second powered domain** | ▫️ Feasibility only (not run) | `…/proto_introspection_second_domain_preflight_2026-06-17.md` | §11 + Known-TODOs cite the negative pre-flight | **Med** — decision item |

---

## 3. Detailed Findings

### Item 1 — Strict pre-answer GSM8K (load-bearing) — VERIFIED

**Artifacts (MD + JSON both exist and agree):**
- `artifacts/reports/proto_introspection/proto_introspection_within_domain_preanswer_specificity_2026-06-17.md`
- `artifacts/reports/proto_introspection/proto_introspection_within_domain_preanswer_specificity_2026-06-17.json`
- Raw metrics: `artifacts/reports/proto_introspection/within_domain_specificity_results.json`
- Features: `artifacts/reports/proto_introspection/within_domain_recapture.pt` (+ `_index.json`)
- Scripts: `utilities/tests/manual/proto_introspection_within_domain_recapture.py`, `…within_domain_analysis.py`

**Every draft figure matches (draft value → artifact value):**

| Quantity | Draft | Artifact (MD / JSON) |
|---|---|---|
| tasks / examples | 170 / 680 | 170 / 680 |
| hidden AUROC | 0.745 [0.707, 0.783] | 0.745 / JSON 0.7451, CI [0.7071, 0.7829] |
| length AUROC | 0.687 | 0.687 / 0.6865 |
| logprob/entropy AUROC | 0.569 | 0.569 / 0.5692 |
| length+logprob AUROC | 0.731 | 0.731 / 0.7312 |
| hidden+all AUROC | 0.797 | 0.797 / 0.797 |
| incremental (hidden+all − composite) | +0.066, CI [+0.017, +0.114] | +0.066 / JSON 0.0658, CI [+0.017, +0.114] |

- **Strict cut definition present?** Yes, explicitly (artifact §4): pre-answer = text before
  `min(FINAL-ANSWER marker, first occurrence of the gold answer value)`; the answer number is excluded by
  construction so it "cannot leak." GSM8K pre-answer median 163 tokens.
- **Confidence intervals present?** Yes (per-feature and incremental; group bootstrap over tasks, 1000 rounds).
- **Safe as primary proto-introspection evidence?** **Yes.** Verdict constants:
  `TIMING=PREANSWER_TIMING_CONTROL_PASS`, `SPECIFICITY=HIDDEN_BEATS_SHORTCUTS_WITHIN_DOMAIN`,
  `WEAK_CLAIM=NARROW_DEFENSIBLE_CLAIM_ONLY`. Caveats (retained, already in draft): single powered domain;
  hidden-alone (0.745) ties the length+logprob composite (Δ +0.014, ns) — the load-bearing effect is the
  **incremental** +0.066.

### Item 2 — S3B2 generated-branch correctness + random baseline — VERIFIED (+ baseline found)

**Artifacts:**
- `artifacts/reports/proto_introspection/final_prewrite_hardening_2026-06-17/s3b2_generated_branch_correctness_refit_2026-06-17.{md,json}` (L2 logistic)
- `artifacts/reports/proto_introspection/final_engineering_expansion_2026-06-17/s3b2_generated_branch_correctness_expanded_2026-06-17.{md,json}` (expanded hidden ridge)
- Pool: `artifacts/reports/probes/mpn_s3b_2026-06-17/s3b1_loop_pools.pt` (16 task groups, 8 oracle-present, 160 candidates; leave-one-task-out)

**Metrics found (all match):** L2 logistic AUROC **0.7515** / pairwise **0.6835** / sel@oracle **0.6250**;
expanded hidden-ridge AUROC **0.7755** / pairwise **0.7338** / sel@oracle **0.6250**. Metadata-only
controls weak (length-only AUROC 0.1094, domain-only 0.4334, provenance-only 0.0754). High-margin
abstention non-rescuing (margin-q90: coverage 0.125, sel@oracle-acted 0.0).

**Matched random baseline — EXISTS.** The expanded artifact's "Baseline Comparison" table (and the refit
JSON) contains a native `RANDOM` control **on this exact pool**: **sel@oracle 0.5833**, separability/AUROC
0.4625. So the 0.6250 forced-choice selector sits only marginally above random on this small pool
(sel@oracle CI [0.25, 0.875]).
- **Value & provenance:** 0.5833 — `s3b2_..._expanded_2026-06-17.md` Baseline Comparison → `RANDOM`;
  `..._expanded_2026-06-17.json` → `baseline_comparison.RANDOM.sel_at_oracle`; also `..._refit_2026-06-17.json`
  `RANDOM` row (0.5833).
- **This is NOT the S3B1 0.58 import.** It is the S3B2 pool's own control; the two values coincide only
  because S3B1 and S3B2 refit over the *same* candidate pool (`s3b1_loop_pools.pt`). The draft now states
  this explicitly (footnote `[^s3b2-prov]`, §6.4, Appendix E.2) so it cannot be mistaken for the forbidden
  import. **Please sanity-check this editorial call** — it satisfies your instruction "add it only if the
  S3B2 artifact itself contains one," and it makes the negative result stronger, but the numeric coincidence
  with 0.58 is worth a second look.

### Item 3 — Orthogonality / null audit — VERIFIED (full precision, draw counts)

**Artifacts:**
- `artifacts/reports/proto_introspection/s1_s3_exact_injection_orthogonality_2026-06-17/s1_s3_exact_injection_orthogonality_null_audit_2026-06-17.{md,json}`
- Delta bundle: `…/s1_s3_exact_injection_delta_bundle_2026-06-17.pt`

| Quantity | Draft | Artifact |
|---|---|---|
| ambient D | 24576 | 24576 |
| injection span rank | 344 | 344 (raw positive-SV rank) |
| relative-1e-3 rank | 246 | 246 |
| entropy effective rank | 6.427472 | 6.427472 |
| participation ratio | 5.062494 | 5.062494 |
| observed projection | 0.018296247348189354 | 0.018296247348189354 |
| k/D chance | 0.013997395833333334 | 0.013997395833333334 |
| observed/expected | 1.3071179500845975 | 1.3071179500845975 |
| random-direction null | 1,000,000 draws | 1,000,000 |
| observed percentile | 99.9907 | 99.9907 |
| p_right | 0.000093 | 9.3e-05 |
| global shuffled mean / pct | 0.022990 / 3.44 | 0.022990 / 3.44 (**10,000 draws**) |
| domain-stratified mean / pct | 0.022702 / 0.0 | 0.022702 / 0.0 (**10,000 draws**) |

- **Shuffled draw counts present?** Yes: both shuffled-label nulls use **10,000** draws; random-direction
  null uses **1,000,000**.
- **Is domain-stratified 0.0 percentile strongly supportable?** From 10,000 draws, observed (0.018296)
  falls below the null's 1st percentile (q01 = 0.020509) — i.e. below all 10k draws. This supports "below
  the 1st percentile / at the low end," but given the tiny effect (~0.0044) and a single 1-D projection
  from one regenerated bundle, the **conservative "at the low end" phrasing is retained** (per your rule).
- **Both nulls correctly represented?** Yes. The two nulls disagree (random-direction: observed *above*
  chance; shuffled-label: observed *below* mean). Verdict `ORTHOGONALITY_LOAD_BEARING_STATUS =
  DO_NOT_USE_AS_ORTHOGONALITY_EVIDENCE` — kept as a **failed simple explanation only**, not load-bearing.

### Item 4 — S1 frozen-fork / K-matched — VERIFIED (live-run, not regenerated)

**Artifacts (all under `artifacts/reports/probes/mpn_s1_baseline_2026-06-13/`):**
`s1_report_2026-06-17.md`, `s1_4_reference_loop.json`, `s1_4a_fork_param_screen.json`,
`s1_4b_kmatched_sampling.json`.

- **Reference loop (S1.4):** 4 tasks (1/domain), K=2, budget 4, α=0.02, last-token, greedy; five
  bit-exactness gates pass; `oracle_over_survivors = base_acc = selected_acc = 0.25`; mean survivors 4.0.
- **Fork screen (S1.4a):** loop-1 loci L24/L36/L47, K=4, MNT=96, prompt+answer scoring,
  α∈{.02,.05,.10} × token-range∈{last, last-8, second-half} + loop-4 sentinel; **greedy 0.0 new-correct
  everywhere**, sample 0.5–0.75 (sampling RNG dominates).
- **K-matched (S1.4b):** N=12 plain samples/task (=3 loci × K=4), temp 0.7, top_p 0.95, MNT 96 (base
  decoded greedy at MNT 160 and 96 to remove truncation confound). **plain-sampling oracle 0.75**,
  **sample-fork oracle 0.611**, sample_fork − plain = −0.139, **greedy-fork new-correct 0.0**,
  selected_acc 0.0. Verdict `FROZEN_FORK_CLOSED__SAMPLING_EXPLAINS_SCREEN__S3_IS_LEVER`.
- **Interpretation match:** sampled-fork gains explained by sampling; deterministic/greedy forks produce
  no reliable new correct answers; frozen branch/carry mechanically valid but reachability-neutral under
  tested settings. All confirmed. Verdict is explicitly scoped **LOCAL** (under tested regimes).

### Item 5 — Seven steering methods / closure — VERIFIED (provenance corrected)

- **"Seven" is exactly artifact/doc-backed.** `docs/evaluator/history/steering-and-adapters/steering-consolidation.md`
  §2 "The seven-method closure" enumerates seven methods of increasing sophistication, matching the draft:
  (1) raw NoNorm readout, (2) empirical success-mean-difference, (3) RMS-calibrated static,
  (4) local BG-score gradient, (5) classifier/logistic adapter, (6) teacher-forced causal adapter,
  (7) sequence-level REINFORCE (`REINFORCE_score_function`) adapter.
- **Cosines** (`bg_causal_intervention_adapter_2026-05-18/analysis.json`, `geometry`):
  adapter/raw −0.000553 (→ −0.0006), adapter/empirical −0.004294 (→ −0.0043), raw/empirical +0.101002
  (→ +0.101). Match.
- **0.951 two-adapter convergence** = `bg_sequence_level_adapter_2026-05-18/diagnostics.json`
  → `cosine_to_teacher_forced_adapter_proxy = 0.951094388961792` (also stated in `steering-consolidation.md`
  line 118). **v3.6's pointer to `bg_steering_suite_2026-05-18` was wrong and is corrected.**
- **Closure supports "no reliable signed control"?** Yes: `RMS_UNSIGNED_ONLY`,
  `GRADIENT_NO_BETTER_THAN_RANDOM`, `LOCAL_LOGIT_CONTROL_ONLY`, `NO_FROZEN_BACKBONE_WRITE_PATH`,
  `CLOSED_UNDER_TESTED_METHODS`; propagation `SURVIVES_32_TOKENS`; safe envelope α ≤ 0.02.
- **Numeric results safe to include:** the three cosines, the 0.951 convergence, teacher-forced margin
  lift +0.0104, and the verdict strings. All pinned.

### Item 6 — Base / reasoning / RL training-stage localization — NOT_FOUND

- **95.2% (Thinking):** historical Kirin (2026) epoch-2 figure (documented; epoch curve 83.3→95.2→62.4).
- **base 24% (below chance) and RLTT 95.0%:** **no live artifact.** Repeated searches (grep over
  `docs/`, `artifacts/`, all `*.md`, `.codex`) found no base-vs-Thinking-vs-RLTT cross-model evaluation.
  The project's frozen backbone throughout is the Thinking/RLTT model (`models/ouro_rltt_local`); the
  untuned base Ouro-2.6B checkpoint appears not to have been retained or evaluated in-repo.
- **Same evaluator across all three? / metric type?** Per the draft's framing it is the fixed Kirin
  evaluator applied unchanged, and the metric is pairwise-preference accuracy on the HH test split — but
  because there is no artifact, these details are **user-confirmed, not verifiable here.**
- **Can the paper safely claim "looping alone insufficient; reasoning/RL localizes the signal"?** The
  *claim* is coherent and consistent with the base-model paper, but with **no artifact it should not be
  presented as one of the strongest artifact-backed results.** Draft now carries a provenance note
  (§3.7 Table 1) marking 24%/95.0% user-confirmed-only and not upgraded. **Recommended: re-run to pin, or
  soften the "one of the strongest results in the paper" framing before arXiv.**

### Item 7 — Math-transfer origin (2.7× / 47-of-100) — NOT_FOUND

- **No artifact exists.** The only occurrences of "47 of 100" / "2.7×" in the entire repo are in the
  paper draft itself. The `chronological-evaluator-summary.md` §6 documents only the *later*, separately
  truncation-confounded broad MATH pilots (which motivated the move to clean GSM8K) — **not** the origin
  47/100 / 2.7× numbers. The git-deleted `docs/project/pairwise_evaluator_locus_memo_v2_2026-05-11.md`
  concerns locus probes, not this ratio.
- **Denominator/baseline recoverable?** No.
- **Safe to include as a number?** Only as clearly-labeled historical/lab-notes motivation. The draft
  already treats it as non-load-bearing; v3.7 upgrades "unarchived" → explicit `NOT_FOUND (purged)`.
- **Confirmed:** it should remain historical/unarchived origin-motivation only.

### Item 8 — Pre-DualAnchor scaffold lineage — VERIFIED (doc/snapshot)

**Artifact:** `docs/evaluator/branch-generation-and-survival.md` (L36–63); corroborated in
`terminal-selection-and-arbiters.md`, `current-state.md`.

| Quantity | Draft | Doc |
|---|---|---|
| fixed_composite_conservative_top4 oracle retention | 0.931 | 0.931 |
| false prune | 0.069 | 0.069 |
| avg survivors | 3.873 | 3.873 |
| old-context/coding retention | 1.000 | 1.000 |
| coding false prune | 0.000 | 0.000 |
| selection-only Phase 2 top4 retention | 0.9514 | 0.9514 |
| false prune | 0.0486 | 0.0486 |
| avg survivors | 3.9109 | 3.9109 |
| best-selected reward | 0.9453 | 0.9453 |
| final reward | 0.6672 | 0.6672 |
| status | `SURVIVAL_READY_FINAL_ARBITER_WEAK` | same |

All exact. **Safe as lineage / historical-supporting evidence** (already labeled snapshot-backed).

### Item 9 — Branch/carry substrate mechanical checks — VERIFIED (doc)

**Artifact:** `docs/evaluator/kv-cache-branch-carry.md` (corroborated by memories
[[autoregressive-kv-branch-carry-v1]], [[partial-cache-splice-v2]]).

- 4 loops × 48 layers = **192 slots** (`slot = current_ut*48 + layer_idx`). ✓
- Zero-perturbation fork: **prefill bit-exact (RMS 0)**; cached decode small bf16 drift
  (RMS ~0.05–0.2, max-abs < 1.0). ✓
- No-carry negative control: RMS ≈ 3.0. ✓
- Suffix-recompute splice: **bit-exact** vs full perturbed-prompt reference across all 192 slots
  (prefill logits RMS 0, continuation bit-for-bit). ✓
- Compute savings: loop 0 **13%**, loop 1 **38%**, loop 2 **63%**, loop 3 **88%**;
  K-scaling (loop 2, L24) K=2 **32%**, K=4 **47%**, K=8 **55%**; **amortized over K ≥ 2**. ✓
- **Main-body safe:** 192 slots, prefill bit-exactness, no-carry RMS≈3.0, suffix-splice bit-exactness.
  **Appendix-only (kept there):** compute-savings percentages (amortized, K≥2). The draft correctly does
  **not** overclaim zero-perturbation bit-exactness beyond prefill (it states prefill RMS 0 + decode
  drift). No change needed beyond confirmation.

### Item 10 — Citation metadata — VERIFIED (web, 2026-07-04)

| Citation | arXiv | Verified title / authors | Draft |
|---|---|---|---|
| Lindsey 2026 | 2601.01828 | *Emergent Introspective Awareness…* — Jack Lindsey (single) | ✓ |
| Binder et al. 2024 | 2410.13787 | *Looking Inward…* — Binder, Chua, Korbak, Sleight, Hughes, Long, Perez, Turpin, Evans | ✓ |
| Comşa & Shanahan 2025 | 2506.05068 | *Does It Make Sense to Speak of Introspection…* — Iulia M. Comsa, Murray Shanahan | ✓ |
| Latent Introspection | 2602.20031 | *…Detect Prior Concept Injections* — Pearson-Vogel, Vanek, Douglas, Kulveit | ✓ |
| Coconut | 2412.06769 | *…Continuous Latent Space* — Hao, Sukhbaatar, Su, Li, Hu, Weston, Tian | ✓ |
| Williams & Tureci RLTT | 2602.10520 | *Prioritize the Process…* — Jonathan Williams, Esin Tureci | ✓ |
| Kirin 2026 (prior paper) | 2604.09870 | *Relational Preference Encoding…* — Jan Kirin (single) | ✓ |
| Zhu et al. (Ouro) | 2510.25741 | *Scaling Latent Reasoning…* — 33 authors incl. Bengio, Eshraghian (full list added) | resolved TODO |
| RewardBench (Lambert) | 2403.13787 | Lambert, Pyatkin, Morrison, Miranda, Lin, Chandu, Dziri, Kumar, Zick, Choi, Smith, Hajishirzi | resolved TODO |
| HH-RM source | 2408.05094 | *Unlocking Decoding-time Controllability…* — Fu, Hou, McAuley, Yan | resolved TODO |

- Ouro HF model card: `https://huggingface.co/ByteDance/Ouro-2.6B-Thinking` (base `…/Ouro-2.6B`; project
  page `https://ouro-llm.github.io/`). **No citation TODOs remain** (only optional BibTeX formatting).

---

## 4. Second Powered Pre-Answer Domain Feasibility

*(Item 11 — feasibility only; no second domain was run, per the hard rule.)*

A live pre-flight already answers most of this:
`artifacts/reports/proto_introspection/proto_introspection_second_domain_preflight_2026-06-17.md`
(`SECOND_DOMAIN_STRENGTHENER_VERDICT = REJECTED_AT_PREFLIGHT_BOTH_CANDIDATES`).

- **Candidate domains examined.** (a) **ARC-Challenge/reasoning** — already captured but *underpowered*
  (105 tasks < 150) and front-loaded (pre-answer median 14 tokens ≈ prompt-only), so it corroborates but
  cannot be the second powered domain. (b) **SVAMP** — REJECTED: 4/8 emit `Final Answer:\boxed{N}` in the
  first ~3 tokens (empty pre-answer), and the GSM8K parser mismatches `\boxed{}` output (≈60% mislabels).
  (c) **Hendrycks MATH (L1–3 numeric)** — REJECTED even with a hardened `\boxed{}` extractor: front-loading
  is solved (pre-answer median 160–223 tokens) but the **base rate is near-degenerate** — for this model,
  math failure = *truncation* (yaps past the token budget without committing), not a wrong answer, so
  parseable samples are ~21/22 correct → no negative class → success/failure probe impossible. Keeping
  truncated-as-failure turns the label into "predicts running out of budget" = a length artifact.
- **Available artifacts/datasets.** GSM8K (the working powered domain); ARC-Challenge (underpowered);
  local HF caches for SVAMP and MATH. The recapture+analysis code is reusable.
- **Can it reuse the GSM8K pre-answer audit code?** **Yes** —
  `utilities/tests/manual/proto_introspection_within_domain_recapture.py` +
  `…within_domain_analysis.py`, plus the hardened
  `utilities/tests/manual/proto_introspection_math_recapture.py` (balanced-brace `\boxed{}` extractor,
  numeric Fraction compare, strict pre-answer cut, truncated-sample dropping). Same design (task-grouped
  CV, group bootstrap, hidden vs length vs logprob vs composite incremental validity).
- **Likely compute cost.** The GSM8K powered run was ~4,127 s for 170 tasks (K=4) on a single ~12 GiB
  GPU — a comparable ≥150-task run is a single bounded GPU session, not a cloud job. The *risk* is
  recapture, but the analysis is cheap.
- **Cleanest control design.** Reuse the exact GSM8K protocol; the binding constraint is finding a domain
  that satisfies **both** (i) answer-last with genuine pre-answer reasoning **and** (ii) a balanced
  parseable correct/incorrect mix. GSM8K uniquely hits this; trivial domains front-load, hard domains
  truncate.
- **Expected risk of an ambiguous result.** **High.** Both obvious candidates already fail for principled,
  model-specific reasons. A usable second domain likely needs a *curated* set (e.g., medium-difficulty
  numeric word problems where the model both reasons and commits) or acceptance of a messier label.
- **Recommendation.** Treat the second domain as **not a cheap win**. Options, in order: (1) accept the
  narrow single-powered-domain claim as-is (defensible; already caveated) and ship; (2) invest in a
  curated medium-difficulty numeric domain designed to dodge both failure modes; (3) power the existing
  ARC/reasoning secondary to ≥150 tasks while acknowledging its front-loading limits what it tests.
  Given the pre-flight, option (1) is reasonable for arXiv, with the second domain as follow-up work.

---

## 5. Exact Patch Summary (`ouro_paper_draft_v3_7_pending_items_pinned.md`)

**What changed (provenance / claim-strength only; no structural or scientific changes):**
- **§5.2** and **Appendix I**: replaced the load-bearing `[TODO: pin…]` block with pinned artifact paths;
  confirmed all figures against MD+JSON; stated the strict-cut definition and CIs.
- **§6.3 footnote `[^s3b2-prov]`, §6.4, Appendix E.2**: corrected the false "we report no random baseline"
  → the S3B2 expanded artifact's **own** RANDOM control (sel@oracle 0.5833), with airtight provenance and
  an explicit distinction from the forbidden S3B1 0.58 import; added metadata-control and abstention
  numbers; pinned both refit + expanded paths.
- **§8.2** and **Appendix G**: replaced K-matched `[TODO]` with exact figures (0.75 / 0.611 / 0.0) and
  full parameters; pinned S1 artifact paths; noted live-run (not regenerated).
- **§8.1** and **Appendix H**: **corrected the 0.951 provenance** (→ `bg_sequence_level_adapter_2026-05-18/
  diagnostics.json`); pinned the seven-method enumeration (`steering-consolidation.md` §2) and the three
  cosines; listed all steering artifact paths.
- **§8.4** and **Appendix J**: resolved the shuffled-draw-count `[TODO]` (1e6 / 10k / 10k); added
  full-precision values; kept "at the low end" phrasing and the failed-explanation status.
- **§3.6 / §1.5**: math-transfer origin "unarchived" → explicit **`NOT_FOUND` (purged)**; non-load-bearing.
- **§3.7 Table 1** (new provenance note), **§10.1 localization row**, **Claims Checklist**: base 24% /
  RLTT 95.0% marked **`NOT_FOUND` / user-confirmed only, not upgraded**; 95.2% flagged as historical Kirin.
- **§8/§10.1/Claims Checklist**: statuses for items 1/2/3/4/5/13/15 changed from "pending" to
  "**live-repo pinned**" with paths.
- **References**: full Zhu et al. (33 authors) + HF URL; full Lambert (RewardBench) list; Fu et al.
  (2408.05094) authors — all web-verified 2026-07-04.
- **Banner** + new **Revision Notes for v3.7**: summarize the pass; **§11 Limitations**: "Provenance items
  pending" → mostly resolved, with the two NOT_FOUND exceptions and a pin of the negative second-domain
  pre-flight.

**TODOs that remain (intentionally, out of this pass's scope):**
- 512-slice HH antisymmetrization (0.9668/0.6367) — not one of the 11 items; companion artifact
  `…/antisymmetrized_hh_pairwise_audit_2026-06-17.{md,json}` exists and should be checked next; full-8,552
  strict rerun still desirable.
- Base-model 24% localization artifact (re-run to pin) — see B1.
- Production: figures/tables, reproducibility-commit pin, LaTeX conversion (Appendix K), BibTeX formatting.

**Claims downgraded (provenance only — no numbers changed, none deleted):**
- Base/RLTT training-stage localization (§3.7): artifact-backed-implied → **user-confirmed, NOT_FOUND**.
- Math-transfer 2.7× (§3.6): "unarchived, flag for pinning" → explicit **NOT_FOUND (purged)**,
  non-load-bearing.
- (No positive claim was upgraded; the S3B2 addition strengthens a **negative** result.)

---

## 6. Final Recommendation

**Order of operations before arXiv:**
1. **Resolve B1 (base-model localization).** This is the only *featured* claim now standing on
   user-confirmation with no artifact. Either re-run a base-vs-Thinking-vs-RLTT evaluation to pin an
   artifact, or soften §3.7's "one of the strongest results in the paper" to match its evidentiary status.
   *Highest-leverage remaining action.*
2. **Decide the second powered domain (B2).** Given the negative pre-flight, the narrow single-domain
   claim is defensible for arXiv; a second domain is worthwhile but not cheap (see feasibility §4). This
   is a judgment call, not a blocker.
3. **Then proceed to the figure/table pass.** Items 1/2/3/4/5/8/9 are pinned and stable; the figures
   (AUROC bars, domain-transfer matrix, S3B2 detection-vs-selection, two-null histogram, cache schematic)
   can be built against the pinned artifacts.
4. **Optional next artifact search:** the 512-slice HH antisymmetrization companion artifact (out of scope
   here) to close the last `[TODO]` in the Claims Checklist.

**Net:** the paper's load-bearing evidence (strict pre-answer GSM8K) and its negative-result spine
(S3B2 selection wall, K-matched frozen-fork, seven-method steering closure, two-null orthogonality) are
now **live-repo pinned and internally consistent**. The two `NOT_FOUND` items are correctly quarantined as
non-artifact-backed. Recommend **proceeding to the figure/table pass after deciding B1**, rather than a
second artifact sweep.
