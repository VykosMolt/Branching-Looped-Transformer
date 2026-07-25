# Proto-Introspection Targeted Controls (A / B / C)

**Date:** 2026-06-17 (run 2026-06-22)
**Goal:** run the three remaining controls (pre-answer leakage, shortcut baselines, task-grouped heldout) needed to decide whether the *weak* proto-introspection claim can move from "substantially supported" to "paper-grade with caveats."
**Scope:** read-only w.r.t. model weights. One bounded GPU recapture (no checkpoint changes, no training, no S3A). Correctness labels are EXTERNAL only (gold answer key / verifier); tap/evaluator scores were never used as labels.

---

## 1. Executive Summary

All three controls were run; an optional fourth (Barbados) was not (not instrumented, avoided a rabbit hole). The controls are **deflationary but clarifying**: they show the previously head-lined trajectory/branch-success "STRONG" result was substantially inflated by **answer leakage** (the original best cell was the 256-token prefix, which is 95% answer-leaked) and by **task memorization** under random splits (random CV over-states AUROC by +0.108 vs task-grouped). After removing both, a **genuine but modest pre-answer signal survives**: hidden states captured strictly before the answer token (a clean recapture, leak-rate 0.0) predict eventual external-verifier success at AUROC 0.690 (95% CI [0.574, 0.789]), essentially equal to the full-answer ceiling of 0.698 — i.e., the predictive information is present *before* the answer is emitted, and the leaked-answer states themselves sit at chance (existing-data leaked subsets AUROC 0.48–0.50). **However**, this pre-answer signal does **not** clearly exceed trivial shortcuts: a domain+length baseline matches it (preanswer hidden 0.690 vs shortcut 0.702), the combined model adds nothing on clean data (Δ −0.013), and within-domain the effect shrinks to AUROC 0.55–0.58 with confidence intervals that all span 0.5 (n = 30–48 per domain). The prompt-only cut (question representation before any generation) is at chance within domain (0.42–0.51) and is beaten by the domain shortcut. Net: **timing is PARTIAL** (pre-answer signal is real and not leakage-derived, but modest), **specificity is PARTIAL leaning negative** (hidden does not outperform shortcuts on clean states; it only beats shortcuts when it can read the leaked answer), and **grouped-heldout is PARTIAL** (signal survives above chance, 0.624, but drops materially and does not beat shortcuts; branch-ranking utility is leakage-dependent). The role-separation (P5) and control-boundary (P7) pillars are unaffected and remain strong. **Conclusion: the weak claim is supported in a narrowed form (readable pre-answer process-quality signal exists), but the targeted controls did not lift the success-prediction leg to paper-grade specificity; the prior evidence matrix's P1/P2 ratings need revision and one powered within-domain control is the cheapest path to settle specificity.**

What this does NOT touch: the separate HH-RLHF relational-preference result (95.2% pairwise) is a different task (preference between two complete candidates, not pre-answer success prediction) and is not re-tested here. No claim of consciousness, self-awareness, autonomous self-control, or capability gain is made.

---

## 2. Artifact Inventory

**Existing artifacts used (read-only):**
- `artifacts/reports/probes/bg_trajectory_prediction_2026-05-18/prefix_features.pt` — 960 prefix feature records `[3 layers × 4 loops × 2048]`.
- `artifacts/reports/probes/bg_trajectory_prediction_2026-05-18/continued_prefixes.json` — per-(task,branch,prefix) external `is_correct`, `prefix_text`, `prefix_token_count`, `parse_failed`.
- `artifacts/reports/probes/bg_trajectory_prediction_2026-05-18/task_suite.json` — 60 tasks (reasoning/science/gsm8k) with task_id grouping.

**New artifacts created:**
- `artifacts/reports/proto_introspection/preanswer_recapture.pt` (+ `_index.json`) — bounded GPU recapture: 120 tasks (reasoning 40, science 50, gsm8k 30), K=4 external-verified samples/task, pooled BG features at 5 capture cuts; 2508 s, 0 errors. Model `models/ouro_rltt_local`, tap layers 24/36/47, 4 loops, `force_all_loops=True`, seed 20260617, temp 0.7 / top_p 0.95, max_new 192 (MCQ) / 224 (gsm8k).
- `artifacts/reports/proto_introspection/controls_analysis_results.json` — all probe metrics.
- Scripts: `utilities/tests/manual/proto_introspection_preanswer_recapture.py`, `utilities/tests/manual/proto_introspection_controls_analysis.py`.

**Missing / not run:**
- `MISSING: Barbados loop-state predictability — consequence: no second-architecture cross-check of the introspection claim. Not instrumented for loop-state dumps; running it would be a rabbit hole. Marked NOT_RUN; not invented.`
- `MISSING: powered within-domain pre-answer probe (≥150 tasks/domain) — consequence: within-domain pre-answer AUROC (0.55–0.58) cannot be declared individually significant; this is the one control that could move specificity off PARTIAL.`

---

## 3. Control A — Pre-Answer Leakage Audit

### Setup
Two complementary tests. **(A1)** A fresh bounded recapture: for each task, capture pooled hidden states at cuts ranging from strictly pre-answer to fully leaked, label by an external verifier over K=4 samples, and probe each cut with a tiny standardize→PCA(24)→L2-logistic-regression probe under task-grouped CV (inherently 1 example/task for the recapture). **(A2)** A leakage-stratified re-analysis of the *existing* 960 prefix features: at each prefix length, split prefixes into "pre-answer" (no FINAL-ANSWER marker and the model's parsed answer not present) vs "leaked," and probe `is_correct` separately.

### Datasets / features / cuts
Recapture cuts and measured leak rate (fraction whose generated part already contains the FINAL ANSWER marker):

| cut | what it sees | leak rate | label |
|---|---|---:|---|
| `prompt_only` | question only, 0 generated tokens | 0.000 | task maj-correct (sample-independent) |
| `gen16` | prompt + first 16 generated tokens | 0.400 | sample-0 correct |
| `gen32` | prompt + first 32 generated tokens | 0.458 | sample-0 correct |
| `preanswer` | prompt + reasoning up to just before FINAL ANSWER | 0.000 | sample-0 correct |
| `full` | prompt + full sample-0 answer | 0.692 | sample-0 correct |

Note: for MCQ (reasoning/science) the model emits "FINAL ANSWER: X" within the first ~16 tokens, so `gen16`/`gen32` are already 40–46% leaked; only `prompt_only` and `preanswer` are clean (leak 0.0).

### Results

**A1 — recapture cuts (grouped CV, n=120 / 113):**

| cut | clean | AUROC | 95% CI | per-domain AUROC (reas/sci/gsm) | n | base |
|---|---|---:|---|---|---:|---:|
| prompt_only | ✅ | 0.574 | [0.473, 0.680] | 0.43 / 0.51 / 0.42 | 120 | 0.583 |
| gen16 | ❌ (0.40) | 0.585 | [0.474, 0.691] | 0.23 / 0.45 / 0.63 | 120 | 0.375 |
| gen32 | ❌ (0.46) | 0.603 | [0.498, 0.702] | 0.32 / 0.44 / 0.62 | 120 | 0.375 |
| **preanswer** | ✅ | **0.690** | **[0.574, 0.789]** | 0.55 / 0.58 / 0.56 | 113 | 0.363 |
| full (ceiling) | ❌ (0.69) | 0.698 | [0.595, 0.798] | 0.57 / 0.57 / 0.77 | 120 | 0.375 |

Two readings:
- **Positive:** the clean `preanswer` cut (leak 0.0) reaches AUROC 0.690, statistically indistinguishable from the leaked `full` ceiling (0.698). The predictive signal is therefore essentially **fully present before the answer token** — it is not produced by reading the committed answer.
- **Caution:** `prompt_only` (before any generation) is at chance *within domain* (0.42–0.51); its pooled 0.574 is a domain-base-rate artifact. The `preanswer` pooled 0.690 is also partly domain-base-rate (within-domain only 0.55–0.58).

**A2 — existing-feature leakage stratification:**

| prefix | leak rate | clean pre-answer AUROC | leaked AUROC |
|---|---:|---|---|
| 32 | 0.463 | **0.654** [0.550, 0.759] (n=129) | 0.480 [0.367, 0.592] (n=111) |
| 64 | 0.538 | **0.726** [0.618, 0.820] (n=111) | 0.503 [0.384, 0.626] (n=129) |
| 128 | 0.708 | 0.523 [0.364, 0.670] (n=70) | 0.548 [0.443, 0.652] (n=170) |
| 256 | 0.954 | underpowered (n=11) | 0.693 [0.619, 0.765] (n=229) |

Striking: in the existing data the predictive signal lives in the **pre-answer** subset (p32 0.654, p64 0.726), while the **leaked** subset is at chance (0.48–0.50). This independently confirms A1: the signal is pre-answer, not from the committed answer. (Caveat: the clean subset is a non-random selection — tasks where the model has not yet emitted an answer by 32–64 tokens — so it carries mild selection bias.)

### Verdict
**PREANSWER_TIMING_CONTROL_PARTIAL.** Pre-answer hidden states predict eventual success above chance (preanswer 0.690, CI excludes 0.5) and as well as the full-answer ceiling, and leaked-answer states are at chance — so the surviving signal is genuinely pre-answer, not leakage. But the magnitude is modest, largely a domain-difficulty proxy, and within-domain it is underpowered (0.55–0.58, CIs span 0.5). Not STRONG/PASS; clearly not a FAIL.

---

## 4. Control B — Shortcut Baselines

### Baselines used
Reviewer-oriented, kept simple. For each prediction task we fit (a) the hidden-state probe, (b) a shortcut-only logistic model, and (c) a combined model (PCA-24 of hidden concatenated with the un-PCA'd standardized shortcut features, so the few shortcut dims are not washed out). Shortcut features:
- **recapture pre-answer cuts:** domain (one-hot), question length (chars), prompt token count — i.e., only features available before the answer.
- **existing data:** domain, prefix length, prefix token count, prefix char length, parse-failed flag, leak flag.

### Hidden-state vs shortcut comparison (grouped CV)

| dataset / cut | hidden | shortcut | combined | Δ(hidden−shortcut) | Δ(combined−shortcut) |
|---|---:|---:|---:|---:|---:|
| recapture prompt_only (clean) | 0.574 | **0.658** | 0.565 | −0.084 | −0.093 |
| recapture preanswer (clean) | 0.690 | **0.702** | 0.689 | −0.012 | −0.013 |
| existing p32 clean | 0.654 | **0.698** | 0.693 | −0.044 | −0.005 |
| existing p32 all | 0.606 | 0.657 | **0.735** | −0.051 | +0.078 |
| existing all prefixes | 0.624 | 0.637 | **0.761** | −0.013 | +0.124 |

### Results / interpretation
- On every **clean / pre-answer** test (recapture prompt_only, recapture preanswer, existing p32-clean), the hidden-state probe **does not beat** the trivial shortcut, and the combined model adds essentially nothing (Δ ≤ ±0.013). I.e., the pre-answer hidden signal is not demonstrably beyond domain/length.
- The hidden state only adds large signal beyond shortcuts when **leakage is included**: on existing all-prefixes the combined model jumps to 0.761 (Δ +0.124 over shortcut), but that gain is dominated by the 95%-leaked p256 cases where the hidden state encodes the committed answer. This is a leakage artifact, not pre-answer process-quality.
- Crisp statement: **hidden features outperform shortcuts only when they can read the (leaked) answer; on genuinely pre-answer states they do not.**

### Verdict
**SHORTCUT_BASELINE_CONTROL_PARTIAL (leans negative / underpowered).** On clean pre-answer states the hidden probe does not outperform domain/length shortcuts; within-domain there is a weak consistent positive trend (AUROC 0.55–0.58 across all three domains) but it is not individually significant at current power. Specificity is not established at paper-grade.

---

## 5. Control C — Task-Grouped Heldout

### Grouping method
Branch-success probe over all 960 existing prefix features (`is_correct` label), comparing 5-fold **random** CV against 5-fold **task-grouped** CV (group = task_id; 60 groups, so no task appears in both train and val). Pairwise branch-ranking accuracy is also computed under grouped CV, by prefix length.

### Original vs grouped results

| split | AUROC | 95% CI | acc@0.5 |
|---|---:|---|---:|
| random | 0.732 | [0.699, 0.766] | 0.732 |
| **task-grouped** | **0.624** | [0.586, 0.662] | 0.709 |
| Δ (random − grouped) | **0.108** | — | — |

Pairwise branch-ranking accuracy under grouped CV, by prefix length:

| prefix | pairwise acc (grouped) | n pairs | leak rate |
|---|---:|---:|---:|
| 32 | 0.474 | 113 | 0.463 |
| 64 | 0.509 | 108 | 0.538 |
| 128 | 0.553 | 123 | 0.708 |
| 256 | **0.737** | 118 | 0.954 |

### Results / interpretation
- The branch-success signal **survives** task-grouping above chance (grouped AUROC 0.624, CI excludes 0.5), but random splits overstated it by +0.108 — task identity leaked across folds.
- Under grouped CV, hidden (0.624) ≈ shortcut (0.637) — i.e., the surviving grouped signal also does not beat shortcuts.
- Pairwise branch-ranking (the application the original report headlined at 0.854) is **at chance at clean short prefixes** (p32 0.474, p64 0.509) and strong **only at the 95%-leaked p256** (0.737). The original 0.854 was a leakage + memorization composite.

### Verdict
**GROUPED_HELDOUT_CONTROL_PARTIAL.** Signal remains above chance under task-grouped splits but drops materially (~0.11) and does not exceed shortcuts; branch-ranking utility is leakage-dependent.

---

## 6. Optional Control D — Barbados

**Not run.** The Barbados modular-expr replication harness is instrumented for capability replication, not loop-state dumps; wiring loop-state capture + a success/failure probe would be a separate build and a rabbit hole relative to the bounded scope here. No Barbados proto-introspection result is asserted or invented.

**Verdict: BARBADOS_PROTO_INTROSPECTION_NOT_RUN.**

---

## 7. Updated Evidence Pillar Table

| Pillar | Prior | Updated | Basis of change |
|---|---|---|---|
| **P1 Prediction** | STRONG | **PARTIAL** (split) | Preference (HH 95.2%) unaffected and STRONG; but the *own-computation success* leg, previously credited via trajectory 0.854, was leakage+memorization-inflated. Clean pre-answer AUROC ≈0.69 pooled / 0.55–0.58 within-domain. |
| **P2 Timing** | PARTIAL | **PARTIAL** (sharpened) | Leakage quantified (46%→95% by prefix). Clean pre-answer signal confirmed present (preanswer 0.690 ≈ full ceiling 0.698; leaked subsets at chance) but modest/shortcut-confounded. |
| **P3 Specificity** | PARTIAL | **PARTIAL→WEAK** | On clean pre-answer states hidden does not outperform domain/length shortcuts (Δ ≤ ±0.013); only beats shortcuts via leakage. |
| **P4 Utility** | weak-yes/strong-no | **weak-yes/strong-no** (unchanged) | Survivor-set retention (DualAnchor) intact; note the trajectory top-1 lift cited earlier was on leakage-affected data. |
| **P5 Role separation** | STRONG | **STRONG** (unchanged) | Not tested by these controls; validity/content/correctness separation stands. |
| **P6 Cross-domain** | PARTIAL | **PARTIAL** (unchanged) | Recapture shows a weak but directionally-consistent pre-answer signal across all three domains (within-domain 0.55–0.58). |
| **P7 Control boundary** | STRONG | **STRONG** (unchanged) | Not tested by these controls; readout≠control stands. |

---

## 8. Paper-Readiness Table

| Claim | Readiness | Evidence | Caveat / wording |
|---|---|---|---|
| Weak proto-introspection exists in Ouro-RLTT hidden states | **READY_WITH_CAVEAT** | preanswer 0.690 (CI excl. 0.5); leaked-states at chance; role-sep + control-boundary | State as: readable pre-answer process-quality signal exists but is modest and partly domain-confounded. |
| Hidden states predict process quality | **READY_WITH_CAVEAT** | HH preference 95.2% (separate); preanswer 0.690 pooled | Distinguish preference (strong) from own-computation success (modest, shortcut-confounded). |
| Signal appears before final-answer leakage | **READY_WITH_CAVEAT** | preanswer (leak 0.0) ≈ full ceiling; leaked subsets at chance; A2 p64-clean 0.726 | True and clean, but effect size modest; within-domain underpowered. |
| Signal is not reducible to trivial shortcuts | **NOT_READY** | clean cuts: hidden ≤ shortcut; combined Δ ≤ ±0.013 | Do not assert; hidden does not beat domain/length on clean states. |
| Signal transfers at least partially across domains | **READY_WITH_CAVEAT** | within-domain 0.55–0.58 across reasoning/science/gsm8k | Directionally consistent but individually underpowered. |
| Signal is useful for external readout/scaffolding | **READY_WITH_CAVEAT** | survivor-set retention 0.95–1.0 (separate); trajectory lift was leakage-affected | Keep weak-utility wording; do not cite the leaked trajectory pairwise as utility. |
| Readout does not imply control | **READY** | no frozen write path; fork closed | Unchanged. |
| Training-time integration is required for control | **READY_WITH_CAVEAT** | control boundary + S3A design | Unchanged; S3A not run. |

---

## 9. Final Verdict Constants

```
PROTO_INTROSPECTION_WEAK_CLAIM_VERDICT = SUBSTANTIALLY_SUPPORTED_REQUIRES_MORE_CONTROLS
PREDICTION_EVIDENCE_VERDICT            = HIDDEN_STATES_PREDICT_PROCESS_QUALITY   # preference strong; own-computation-success modest + shortcut-confounded
TIMING_EVIDENCE_VERDICT                = PREANSWER_TIMING_CONTROL_PARTIAL
SPECIFICITY_EVIDENCE_VERDICT           = SHORTCUT_BASELINE_CONTROL_PARTIAL       # leans negative: hidden does not beat shortcuts on clean states
GROUPED_HELDOUT_VERDICT                = GROUPED_HELDOUT_CONTROL_PARTIAL
UTILITY_EVIDENCE_VERDICT               = WEAK_UTILITY_SUPPORTED_STRONG_CAPABILITY_UTILITY_NOT_PROVEN
ROLE_SEPARATION_VERDICT                = VALIDITY_CONTENT_CORRECTNESS_SEPARATED
CONTROL_BOUNDARY_VERDICT               = READOUT_SUPPORTED_CONTROL_NOT_SOLVED
BARBADOS_REPLICATION_VERDICT           = BARBADOS_PROTO_INTROSPECTION_NOT_RUN
NEXT_STEP_VERDICT                      = RETURN_TO_EVIDENCE_MATRIX               # revise P1/P2; optionally run one powered within-domain pre-answer control
```

---

## 10. Final Recommendation

- **Can paper writing begin?** Not yet for the headline "process-quality prediction" claim. The controls *weakened* rather than confirmed the success-prediction leg: the strong trajectory numbers were leakage+memorization artifacts, and the genuine pre-answer signal that remains does not beat trivial shortcuts on clean states. A paper that leads with "looped hidden states predict the model's own success" is **not** supportable at paper-grade today.
- **Which claims are safe now?** (1) A clean pre-answer signal exists: hidden states before the answer token predict eventual success as well as states after it (preanswer ≈ full ceiling; leaked states at chance) — stated with its modest effect size. (2) Role separation (validity/content/correctness). (3) The control boundary (readout ≠ control). (4) The HH relational-preference result (95.2%), framed as preference between complete candidates, not pre-answer introspection.
- **Which must remain caveated / not claimed?** Do not claim the signal is beyond trivial shortcuts (it is not, on clean states). Do not re-use the trajectory pairwise 0.854 as evidence (leakage). Do not claim consciousness, self-awareness, autonomous self-control, or capability gain.
- **What remains before drafting?** Either (a) **revise the evidence matrix** (P1 → split preference/success; P2 → sharpened PARTIAL; P3 → WEAK) and write a deliberately *narrow* paper; or (b) run **one powered within-domain pre-answer control** (≥150 tasks/domain, per-domain CIs, plus a logprob/entropy baseline) to test whether the within-domain 0.55–0.58 trend is real and beyond shortcuts. The latter is the single highest-value next experiment and is locally feasible (a larger version of the recapture run here).

**Bottom line:** `NEXT_STEP_VERDICT = RETURN_TO_EVIDENCE_MATRIX`. The targeted controls did their job — they caught that the prior "STRONG" prediction/timing rating rested on leakage and memorization, and they isolated a real-but-modest pre-answer signal that is not yet shown to be beyond trivial shortcuts. Honest status: weak proto-introspection is **substantially supported in a narrowed form, but requires more (powered, within-domain) controls before the success-prediction leg is paper-grade.**
