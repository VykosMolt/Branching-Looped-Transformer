# Post-v10 synthesis v8 — locked Phase 1 architecture

**Date:** 2026-05-18
**Status:** Supersedes v4, v5, v7, and the v7 handoff as the canonical BG architecture spec.
**Scope:** Locks Phase 1 head set, projection-direction framing, and routing principle. Names controller-policy simulator as the next experimental question. Defers prior open items explicitly.

## 0. The locked thesis

The BG Phase 1 architecture uses **two production heads plus one backup specialist**, distinguished by the kind of discrimination the projection captures rather than by domain label.

> **HH preference geometry and objective branch geometry are empirically distinct projection directions.** A general HH-trained head reads preference/coherence structure that objective-domain training cannot substitute. A mixed objective head reads cross-candidate structure that generalizes across code, reasoning, science, and clean-math objective tasks. High-local-similarity code (strict-clean correct vs near-miss) remains the hardest known contrast and retains a dedicated specialist as backup.

This replaces three prior framings:

- v4's "AntisymLinear as universal Phase 1 default" — too coarse; projection direction matters more than head architecture.
- v7's "general HH + per-domain specialists" — overshoots; objective domains share enough structure that one mixed head suffices.
- The "one universal head" hypothesis tested by MIX_HH_OBJECTIVE — refuted by HH heldout regret of -0.150 to -0.250 across all mixed objective heads.

The decision principle for adding future specialists is the **contrast type**, not the domain label:

- Preference/coherence contrasts (HH-style, chemistry MCQ-style): HH general head.
- Cross-candidate objective contrasts (code, reasoning, science, math): objective mixed head.
- High-local-similarity contrasts (strict-clean code near-miss specifically): code specialist backup, route or ensemble.

## 1. Locked Phase 1 head set

| Role | Head | Training source | Layer config | Architecture | Production deployment |
|---|---|---|---|---|---|
| HH general | `hh_general` | HH-RLHF 200 pairs, Thinking backbone | 47_concat_L1_L4 | AntisymLinearNoNorm | Default for unknown/preference/coherence contrasts |
| Objective mixed | `objective_mixed_primary` | CODE + REASONING_NATURAL + REASONING_TRACE (MIX_CODE_REASONING) | 36_L4 | AntisymLinearNoNorm | Default for code, reasoning, science, GSM8K |
| Code specialist | `code_specialist_backup` | Code-only training, 14 task split, 138 pairs | 36_L4 | AntisymLinear | Backup for strict-clean code; route or ensemble |

### Performance summary on validated eval domains

| Eval domain | hh_general | objective_mixed | code_specialist | Best |
|---|---:|---:|---:|---|
| HH_200 (full) | 0.855 | 0.880* | 0.535 | hh_general (*caveat) |
| HH heldout20 | 0.850 | 0.700 | 0.640 | hh_general |
| CODE_STRICT_CLEAN_ALL16 (pairwise) | 0.600 | 0.867 | 0.833 | objective_mixed |
| CODE_RUNNABLE_DIAGNOSTIC | strong | strong | strong | objective_mixed (marginal) |
| CLEAN_GSM8K_EXPANDED | strong | 0.870 | 0.796 | objective_mixed |
| REASONING_NATURAL_DISTRACTOR | 0.733 | 0.792 | 0.722 | objective_mixed |
| REASONING_TRACE | 0.852 | 1.000 | 0.869 | objective_mixed |
| SCIENCE_OVERALL | 0.692 | 0.702 | 0.703 | objective_mixed (tie) |
| SCIENCE_BIOLOGY | 0.600 | code-favoring | 0.840 | code-trained variants |
| SCIENCE_CHEMISTRY | 0.773 | HH-favoring | 0.640 | hh_general |

*The HH_200 full-set result for MIX_HH_OBJECTIVE at 0.880 is contaminated by training-data exposure; the heldout20 result at 0.700 is the trustworthy estimate for that head. MIX_CODE_REASONING's HH performance is the relevant comparison here and runs at -0.150 on heldout.

### Why these specific configurations

**47_concat_L1_L4 / AntisymLinearNoNorm for hh_general:** Fixed-config audit identified this as the best HH_200 row. NoNorm wins on HH preference, suggesting HH structure is more transitive/scalar than was assumed before code/reasoning data came in. Layer 47 with L1+L4 fusion is the bipartite-trajectory readout that v10 established as optimal for HH preference.

**36_L4 / AntisymLinearNoNorm for objective_mixed_primary:** MIX_CODE_REASONING's best config across the strict-clean code, reasoning trace, and clean GSM8K evals. Layer 36 (converged intermediate) suits objective discrimination because the trajectory is settled but pre-final-readout. NoNorm wins because objective correctness is more scalar-readable than HH preference.

**36_L4 / AntisymLinear for code_specialist_backup:** Best code-trained config on strict-clean ALL16. AntisymLinear preferred over NoNorm here because hard near-miss discrimination requires the full relational geometry, not just scalar utility. Retained because at n=16 the +0.034 gap between mixed (0.867) and specialist (0.833) is within noise; specialist may be marginally better in deployment.

## 2. Architectural principle (locked)

The contrast-type rule:

| Contrast type | Operational definition | Head |
|---|---|---|
| Preference/coherence | Candidates differ in "how plausibly explained" or "how well-formed." HH, chemistry MCQ, persuasive writing | hh_general |
| Cross-candidate objective | Candidates differ in correctness against external verifier. Code, reasoning, science, math. Candidates are *different things* not perturbations of the same thing | objective_mixed_primary |
| High-local-similarity | Candidates share most structure, differ in one locally-subtle element. Strict-clean code (same task, both compile, one bug different) is the canonical case | code_specialist_backup, with route/ensemble |

The rule predicts that future high-local-similarity domains will need their own specialists. The rule predicts that future preference-coherence domains will route to hh_general regardless of nominal domain. The rule predicts that most objective discrimination tasks across novel domains will be handled by objective_mixed_primary.

The principle is empirically grounded in:

- HH heldout regret of -0.150 to -0.250 across all five mixed objective heads, demonstrating HH projection is distinct.
- MIX_CODE_REASONING's positive average objective regret of +0.033 across four objective domains, demonstrating objective projection generalizes.
- Strict-clean code being the only domain where the mixed head and the code specialist are within noise of each other on the hardest contrast, demonstrating the specialist's continued value at the local-similarity extreme.

## 3. Open architectural question — controller routing policy

Phase 1 deploys three heads. The controller decides which head's score to use, or how to combine them, for each branch-selection query.

The decision is genuinely open and is the next experimental question. Candidate policies to test:

1. **Default-and-specialist routing.** Use objective_mixed by default. Route to code_specialist when domain is known to be code AND candidates appear structurally similar (heuristic: both candidates compile or both produce parseable output). Use hh_general when domain is HH/preference or unknown.

2. **Margin-based deferral.** Run objective_mixed first. If |margin| is small, ensemble with code_specialist and hh_general. Use the head with the largest absolute margin, or vote across the three.

3. **Disagreement-based ensemble.** Run all three heads. If agreement is high (all three pick the same candidate), proceed. If agreement is low, defer to additional rollouts or abstain.

4. **Domain-tagged routing.** Require the caller to specify domain. Route deterministically: domain == code → both objective_mixed and code_specialist; domain == HH → hh_general; domain in {reasoning, science, math} → objective_mixed; domain unknown → hh_general.

5. **Single-head baselines.** objective_mixed only, hh_general only, code_specialist only — these are the experimental controls that determine whether routing/ensembling adds value over picking one head.

The controller-policy simulator is the experiment that tests these. Each policy is evaluated on the cross-domain eval matrix using the existing head registry; no new training, no new generation. The success criterion is "best policy beats best single-head baseline by ≥3 pp pairwise on average across domains AND doesn't regress on any single domain by more than 5 pp."

This is the next experiment after v8 lands.

## 4. Backbone, taps, and architectural constants

These remain locked from v4/v7 and do not change:

- **Backbone:** Ouro-RLTT, 2.6B params, R=4 loop iterations, frozen for Phase 1.
- **Tap interface:** Heterogeneous. Layers 24/36 single-state (converged trajectory), layer 47 fused or all-loops (bipartite/trajectory-spread). Empirically validated across HH, GSM8K, code, reasoning, science.
- **Head family:** AntisymLinear (LayerNorm-no-affine over diff, Linear-no-bias) and AntisymLinearNoNorm (Linear-no-bias over diff). Antisymmetric by construction. No swap augmentation, no λ_sym, no GRU.
- **Pooling:** Masked mean over valid tokens. Learned attention pooling deferred indefinitely as it never empirically beat mean pooling.
- **Numerical contract:** bf16 forward, fp32 captured states, fp32 head training.

Layer 47 trajectory spread per domain (loop L1-L4 cosine):

| Domain | L1-L4 cos |
|---|---:|
| HH | 0.735 |
| Clean GSM8K | 0.631 |
| Strict-clean code | 0.723 |
| Reasoning MCQ | 0.687 |

Layers 24 and 36 are converged across all four domains (cosine ≥0.91). The heterogeneous tap design is empirically supported.

## 5. Validated evaluation domains

The following are validated for cross-domain BG evaluation. Routing decisions and architectural changes are tested against these:

- **HH_200** (200 HH preference pairs, Thinking and RLTT captures)
- **CLEAN_GSM8K_EXPANDED** (28 tournaments, 79 candidates, exact-answer verifier labels)
- **CODE_RUNNABLE_DIAGNOSTIC** (8 tournaments, 37 candidates, unit-test labels, mixed correct/near-miss/wrong_code/nonsense)
- **CODE_STRICT_CLEAN_ALL16** (16 tournaments, 46 candidates, unit-test labels, strict correct-vs-near-miss only)
- **REASONING_NATURAL_DISTRACTOR** (60 questions, 240 candidates, ARC-Challenge + OpenBookQA, official answer keys)
- **REASONING_TRACE** (24 tournaments, 85 candidates, generated reasoning traces with answer-key labels)
- **SCIENCE_OVERALL** (120 questions, 480 candidates, MMLU + SciQ across biology/chemistry/medicine/general)
- **SCIENCE_BIOLOGY / CHEMISTRY / MEDICINE / GENERAL** (25/25/25/25 question subdomain splits)

Notably *not* validated: MATH (gate-scale deferred under local compute; clean GSM8K serves as the math proxy). Logic (single combined pilot only). The current eval matrix is sufficient for Phase 1 architecture decisions.

## 6. Deferred or settled items

### Deferred-not-blocked

- **MATH gate-scale dataset construction.** Compute-hostile under local budget due to Ouro-RLTT verbosity. GSM8K serves as the math proxy. Resume if cloud compute becomes available.
- **Full-split HH capture (50k/8552).** 200-example captures sufficient for current architecture decisions. Resume if publication-strategy work activates and Experiment 2 Redux needs proper-power confirmation.
- **Harder reasoning natural distractors / generated near-misses.** Current 60-question MCQ + 24-tournament trace set is enough for "reasoning routes to objective_mixed." Resume if reasoning-specific specialist becomes a hypothesis worth retesting.
- **Larger strict-clean code eval (>20 tasks).** n=16 makes the +0.034 mixed-vs-specialist gap noisy. Could screen another 100-150 tasks to expand. Marginal value for current architecture.
- **Code+math mixed head.** GSM8K performed well under code-trained projection; explicit mix could be tested. Not load-bearing.

### Settled

- **AntisymLinear vs published 5M-param GRU head.** AntisymLinear at ~4k params reaches ~0.855 on HH_200 vs published head's 0.95. Provisional Path 2 (capacity-vs-structure contrast within 10pp). Formal Experiment 2 Redux on full-split deferred.
- **GRU temporal aggregation.** Underperforms simpler heads on clean GSM8K. Obsolete as default. Retained as ablation reference only.
- **LayerNorm vs NoNorm.** Both useful. NoNorm wins on objective domains (transitive scalar utility is enough); LayerNorm wins on hard relational/near-miss (HH preference, strict-clean code). Both retained in head family.
- **Heterogeneous tap interface.** Layer 24/36 single-state, layer 47 fused. Empirically validated across HH, GSM8K, code, reasoning, science.
- **Mixed-domain training.** MIX_CODE_REASONING is useful; MIX_HH_OBJECTIVE dilutes HH; per-domain specialists not needed beyond code. Settled by 2026-05-17 mixed-head experiment.
- **Reasoning specialist.** Not needed. Multiple-choice reasoning, generated trace selection, and natural distractor selection all route to objective_mixed (or hh_general for chemistry-like coherence cases).
- **Science specialist.** Not needed. Heterogeneous subdomain behavior handled by routing to existing heads.

### Obsolete

- v4's mixed-domain L_eval design across HH/math/code/reasoning channels with channel weighting. Subsumed by simpler "HH + objective mixed" two-head training.
- v4's Experiment 2 Redux 12-run λ_sym sweep. Architectural antisymmetry obviates λ_sym. Provisional Path 2 by side-effect at 200 examples.
- The "one specialist per domain" framing from v7. Replaced by contrast-type principle.
- Math gate-prep at <1024 tokens. Truncation-confounded; GSM8K is the math proxy.

## 7. Phase 2 implications

Phase 1 trains tiny heads on captured features from a frozen backbone. Phase 2 was originally specified in v4 as full joint training of taps and backbone with mixed-domain L_eval over 20-50B tokens on cloud compute.

The v8 architecture changes Phase 2 substantially:

**Phase 2 becomes a backbone-regularization pass, not a tap-training pass.** The taps already work as cheap linear probes on frozen states; what changes in Phase 2 is ensuring the backbone *preserves* readable HH and objective projections under L_LM pressure during continued pretraining. The L_eval objective shrinks to a regularizer rather than a primary loss.

This is a much cheaper Phase 2 compute envelope than the original v3/v4 spec. The cloud quote should reflect:

- LoRA rank 16 across most layers
- Full unfreeze on layers 24, 36, 47
- 5-20B tokens (not 20-50B; the regularization target is "don't break the projections," not "learn new ones")
- L_LM with HH-projection-preservation + objective-projection-preservation regularizers, weights TBD
- Validation against the eval matrix from §5 at each checkpoint

The actual Phase 2 cloud quote and L_LM mix validation are listed in the next-action backlog below.

## 8. Next actions, in order

1. **Controller-policy simulator.** Build infrastructure to evaluate routing/ensemble policies on the cross-domain eval matrix using the locked three-head registry. Test the five policies in §3. Produce a verdict: best policy, regret table, and recommendation.

2. **Phase 2 cloud quote.** Get pricing for the reframed Phase 2 (smaller compute envelope, regularization-focused). A100 or H100 rental, 5-20B tokens of LoRA + selective-unfreeze continued pretraining.

3. **Phase 2 L_LM mix validation.** Validate the FineWeb-Edu / OpenWebMath / NuminaMath-CoT / StarCoder mix against Ouro-RLTT baseline before committing to the full pretraining run.

4. **Optional polish, low priority:** Bootstrap CIs on the mixed-head regret table (1-2 hours, doesn't change the architecture). Larger strict-clean code eval (1-2 weeks of screening, may tighten the mixed-vs-specialist comparison). Full-split HH Experiment 2 Redux (overnight capture + 4-6h training, settles Path 2 verdict).

5. **Eventually, if publication activates:** Write the BG/tap interpretability paper. Headline finding: Ouro-RLTT linearly organizes branch-selection signal in two distinct projection directions (preference/coherence and objective discrimination); tiny exact-antisymmetric probes at ~4k parameters read both directions; the strict-clean code regime requires a third specialist projection for high-local-similarity discrimination.

## 9. One-paragraph thesis

The BG controller for Phase 1 uses an HH-trained head for preference/coherence contrasts and a code+reasoning mixed-trained head for objective contrasts, with a pure code specialist retained as backup for the strict-clean code regime where local-similarity discrimination is the hardest known case. The architectural principle is that contrast type, not domain label, determines projection requirements: HH preference geometry is empirically distinct from objective discrimination geometry, but objective domains (code, reasoning, science, math) share enough structure that one mixed head suffices. The open architectural question is the controller's routing policy among the three heads, and that's the next experimental question. Everything else — backbone choice, tap layers, head family, pooling, numerical contract — is locked.

## 10. Documents superseded

This v8 supersedes:

- `post_v10_synthesis_2026-05-15_v4.md`
- `post_v10_synthesis_2026-05-16_v5_clean_gsm8k_code_next.md`
- `post_v10_synthesis_2026-05-17_v7_actual_state_and_next.md`
- `bg_after_v4_handoff_2026-05-17.md`
- `bg_after_v7_handoff_2026-05-17.md`
- `bg_tap_interface_revision_2026-05-15.md`

The historical documents remain in the archive for the paper trail. v8 is the canonical reference going forward.

## 11. What's not changed by v8

For clarity:

- The locus memo v3-v10 remains the authoritative reference for HH-RLHF readout work prior to the BG pivot.
- The CLT paper (95.2% on HH-RLHF with 5M-param GRU head) remains the published reference; v8 doesn't republish anything.
- The math BG-gate pilot from 2026-05-15 remains in the historical record, with the validity caveats from the truncation-confounded pilot already documented.
- The Hunter-Seeker ARC agent work and Ouro depth expansion plans are separate tracks, not affected by v8.
