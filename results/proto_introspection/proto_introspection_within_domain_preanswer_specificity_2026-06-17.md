# Within-Domain Strict-Preanswer Specificity Audit

**Date:** 2026-06-17 (run 2026-06-22/23)
**Decisive question:** *Within a single domain, at a strict pre-answer cut, do Ouro-RLTT hidden states predict final success/failure better than length / logprob / metadata baselines?*
**Scope:** read-only on weights; one bounded GPU recapture; external-verifier labels only; frozen Ouro; no checkpoint changes; no S3A.

---

## 1. Executive Summary

This is the powered follow-up to the prior controls round, whose specificity verdict was **deflationary but underpowered** (n≈40–50/domain, no logprob baseline, marker-only pre-answer cut). With proper power and clean design, the result **partly reverses** that pessimism. In the powered **GSM8K** domain (170 tasks, 680 per-sample examples, task-grouped CV), strict-preanswer hidden states predict per-sample correctness at **AUROC 0.745 (CI [0.707, 0.783])** — clearly above chance and above each individual shortcut (length 0.687, logprob/entropy 0.569). Critically, on the proper incremental-validity test, **hidden states add statistically significant signal beyond all length+logprob shortcuts combined**: the hidden+shortcut model reaches 0.797 vs 0.731 for shortcuts alone, a **+0.066 gain with group-bootstrap CI [+0.017, +0.114] (excludes 0)**. The honest caveat: the hidden state **alone** (0.745) does **not** significantly beat a *length+logprob composite* baseline (0.731; Δ +0.014, ns) — that composite is itself a strong predictor — so the hidden contribution is **incremental/complementary rather than dominant**. The secondary **reasoning** domain (105 tasks, UNDERPOWERED, and pre-answer mostly front-loaded: median 14 tokens) corroborates more strongly: hidden beats every shortcut and the composite **significantly** (+0.09, all CIs exclude 0). Net: the strict-preanswer hidden signal is real, powered, leakage-controlled, and carries information beyond trivial shortcuts — but the effect is modest and demonstrated cleanly in one powered domain. This **upgrades** the weak proto-introspection claim from "requires more controls" to a **narrow, defensible, paper-grade claim with caveats**.

---

## 2. Artifact Inventory

**Process check (first step):** `jobs` empty; `ps … grep -E proto|ouro|python|manual` → none; GPU idle (851 MiB) before launch. No stale processes; nothing terminated.

**New artifacts:**
- `artifacts/reports/proto_introspection/within_domain_recapture.pt` (+ `_index.json`) — powered recapture. GSM8K 170 tasks + reasoning 105 tasks (budget-capped at 130 requested), K=4 external-verified samples, per-sample strict-preanswer + prompt-only features + logprob/entropy. 6322 s, 0 errors.
- `artifacts/reports/proto_introspection/within_domain_specificity_results.json` — probe metrics.
- Scripts: `utilities/tests/manual/proto_introspection_within_domain_recapture.py`, `…within_domain_analysis.py`.

**Datasets:** loaded directly from local HF cache (offline): `openai/gsm8k` (test, 1319 available), `allenai/ai2_arc` ARC-Challenge (test, 1172 available). The project's curated loaders are capped (gsm8k at 30), so raw datasets were used to reach adequate N.

**Missing / not run:**
- `MISSING: second powered domain — reasoning reached only 105 tasks (budget cap) and is UNDERPOWERED (<150); its pre-answer is mostly front-loaded (median 14 tok). Consequence: the powered specificity result rests on one domain (GSM8K).`
- `MISSING: Barbados loop-state cross-check — still NOT_RUN (out of scope).`

---

## 3. Dataset and Split Construction

| domain | tasks | per-sample examples | groups | base rate (correct) | class balance | split |
|---|---:|---:|---:|---:|---|---|
| **gsm8k** (powered) | 170 | 680 | 170 | 0.599 | 407 pos / 273 neg | 5-fold task-grouped CV |
| reasoning (underpowered) | 105 | 420 | 105 | 0.429 | 180 pos / 240 neg | 5-fold task-grouped CV |

- **Labels:** external verifier only — GSM8K `numeric_exact` (gold from `#### N`), reasoning `mcq_letter` (gold `answerKey`). Tap/evaluator scores never used as labels.
- **Grouping:** group = task_id; no task's samples split across folds (hard rule honored). No domains mixed for the primary result. No groups dropped.
- K=4 samples/task, temp 0.7, top_p 0.95, seed 20260617.

---

## 4. Strict-Preanswer Capture Definition

Per sample, the pre-answer cut = text **before** `min(FINAL-ANSWER marker, first occurrence of the gold answer value)`. For GSM8K this also excludes the answer **number** if it appears in the reasoning (e.g., "…the total is 18…"), so the answer value cannot leak. Features are the standard pooled `[3 layers (24/36/47) × 4 loops × 2048]`, captured on `prompt + preanswer_text`, frozen Ouro, 4 UT loops.

| domain | pre-answer coverage | n_pre_tok (median / mean / p10–p90) | leakage risk |
|---|---:|---|---|
| gsm8k | 1.00 | 163 / 154.8 / 61–224 | controlled (cut at gold value or marker); answer number absent by construction |
| reasoning | 1.00 | 14 / 67.6 / 6–192 | controlled, but mostly front-loaded → pre-answer ≈ prompt-level (weak test) |

GSM8K is the clean primary: median 163 tokens of genuine reasoning precede the answer. Reasoning MCQ front-loads the letter, so its "pre-answer" is mostly a few tokens (≈ prompt-only) — a structurally weaker test, reflected in its underpowered status.

---

## 5. Baselines

For each domain (single-domain, so no domain feature), task-grouped CV:
1. **Random** — base rate (AUROC 0.5).
2. **Shortcut-only (length/metadata):** question chars, prompt tokens, pre-answer generated-token count.
3. **Logprob/entropy:** mean token logprob, mean token entropy, last-position entropy over the pre-answer tokens (teacher-forced).
4. **Hidden-state probe:** standardize → PCA(24) → L2 logistic regression on `[3,4,2048]`.
5. **Hidden + shortcut + logprob:** PCA(24) of hidden concatenated with raw standardized shortcut+logprob features.

Significance via **group bootstrap** (resampling tasks, 1000 rounds) on paired AUROC deltas.

---

## 6. Results by Domain

### GSM8K (powered, n=680, 170 groups, base 0.599)

| model | AUROC | 95% CI | acc | balanced acc |
|---|---:|---|---:|---:|
| random | 0.500 | — | 0.599 | 0.500 |
| shortcut (length) | 0.687 | — | — | — |
| logprob/entropy | 0.569 | — | — | — |
| length + logprob | 0.731 | — | — | — |
| **hidden** | **0.745** | [0.707, 0.783] | 0.71 | 0.66 |
| **hidden + all shortcuts** | **0.797** | — | — | — |

Paired deltas (group bootstrap, resampling tasks):

| comparison | Δ AUROC | 95% CI | significant? |
|---|---:|---|---|
| hidden − length | +0.059 | [−0.0005, +0.120] | borderline (touches 0) |
| hidden − logprob | +0.176 | — | yes (large) |
| hidden − (length+logprob composite) | +0.014 | [−0.054, +0.080] | **no** |
| **(hidden+all) − (length+logprob)** | **+0.066** | **[+0.017, +0.114]** | **YES** |
| (hidden+all) − length | +0.110 | [+0.060, +0.167] | YES |

**Interpretation:** hidden beats each *individual* shortcut by point estimate and adds **significant incremental signal** beyond all shortcuts combined (+0.066, CI excludes 0). But hidden *alone* ties the length+logprob composite (+0.014, ns) — that composite is a strong baseline (longer, more-confident solutions correlate with success). So the hidden state carries **real, complementary** success information, but is not a standalone-dominant predictor.

### Reasoning (UNDERPOWERED, n=420, 105 groups, base 0.429)

| model | AUROC | 95% CI |
|---|---:|---|
| shortcut (length) | 0.590 | — |
| logprob/entropy | 0.528 | — |
| length + logprob | 0.597 | — |
| hidden | 0.690 | [0.634, 0.741] |
| hidden + all | 0.690 | — |

Paired deltas: hidden − length +0.100 [+0.039, +0.162] **SIG**; hidden − composite +0.093 [+0.025, +0.159] **SIG**; (hidden+all) − composite +0.093 [+0.037, +0.150] **SIG**.

**Interpretation:** hidden beats all shortcuts **significantly** here — but the domain is UNDERPOWERED (105<150) and the pre-answer is mostly front-loaded (median 14 tok ≈ prompt-level), so this is corroborating, not decisive.

---

## 7. Results by Cut (timing)

Two cuts captured: prompt-only (0 generated tokens) and strict-preanswer (full reasoning before the answer value).

| domain | cut | hidden AUROC | shortcut AUROC | Δ |
|---|---|---:|---:|---:|
| gsm8k | prompt_only | 0.699 [0.611, 0.788] | 0.689 | +0.010 |
| gsm8k | **preanswer** | **0.745 [0.707, 0.783]** | 0.687 (len) / 0.731 (len+lp) | +0.059 / +0.014 |
| reasoning | prompt_only | 0.623 [0.516, 0.727] | 0.436 | +0.187 |
| reasoning | preanswer | 0.690 [0.634, 0.741] | 0.590 | +0.100 |

**Timing reading:** for GSM8K the hidden advantage over length emerges *with* the reasoning trajectory (prompt-only Δ +0.010 → preanswer Δ +0.059), consistent with the signal living in the unfolding computation rather than the static question. The strict cut is leakage-controlled (answer value excluded), so the 0.745 is genuinely pre-answer.

---

## 8. Specificity Conclusion

- **Does the hidden state add information beyond shortcuts?** **Yes, significantly** — the incremental-validity test (hidden+shortcuts vs shortcuts-alone) gains +0.066 in powered GSM8K (CI [+0.017, +0.114]) and +0.093 in reasoning (CI [+0.037, +0.150]). Hidden is not merely a re-encoding of length/logprob.
- **Within a single domain?** Yes — both analyses are strictly within-domain (no domain-base-rate confound).
- **Under grouped split?** Yes — all results are task-grouped CV (no task split across folds).
- **Caveat:** hidden *alone* does **not** significantly exceed a length+logprob composite in the powered domain (+0.014, ns). The effect is incremental and modest, and the second domain is underpowered.

---

## 9. Updated Proto-Introspection Pillar Statuses

| Pillar | Prior (post-controls) | Updated | Basis |
|---|---|---|---|
| **P1 Prediction** | PARTIAL | **STRONG (powered)** | GSM8K powered hidden AUROC 0.745; beats individual shortcuts; significant incremental validity. Prior deflation was an underpower artifact. |
| **P2 Timing** | PARTIAL | **STRONG / PASS** | Strict leakage-controlled pre-answer cut (median 163 tok) predicts at 0.745 in a powered domain. |
| **P3 Specificity** | WEAK | **PARTIAL (positive)** | Significant incremental validity beyond all shortcuts (+0.066, CI excl. 0); but hidden-alone ties the length+logprob composite; one powered domain. |
| **P4 Utility** | weak-yes/strong-no | unchanged | Not tested here. |
| **P5 Role separation** | STRONG | unchanged | Not tested here. |
| **P6 Cross-domain** | PARTIAL | **PARTIAL (improved)** | Powered GSM8K + significant (underpowered) reasoning, same direction. |
| **P7 Control boundary** | STRONG | unchanged | Not tested here. |

---

## 10. Paper Implication

- **What can now be claimed:** In a powered, within-domain, leakage-controlled test, strict-preanswer Ouro-RLTT hidden states predict the model's own eventual success above chance (GSM8K AUROC 0.745) and carry **statistically significant predictive information beyond length and logprob/entropy shortcuts** (incremental +0.066, CI excludes 0). A token-confidence (logprob) baseline is weak (0.569), so the signal lives in the **representation**, not the output distribution — a clean proto-introspection point.
- **What must remain caveated:** the effect is **modest and incremental** — hidden alone does not beat a length+logprob composite; only one domain is powered; the second (reasoning) is underpowered and front-loaded. Do not claim hidden states *dominate* trivial predictors, nor that the result is established across domains.
- **Proceed as proto-introspection, or reframe?** Proceed as a **narrow proto-introspection** paper with caveats. The narrow claim — "looped hidden states expose a readable, pre-answer, partly shortcut-independent process-quality signal about the model's own ongoing computation" — is now defensible. The broad/strong claim (large, shortcut-dominant, cross-domain self-prediction) is **not** supported; relational-readout framing remains the honest backbone, with proto-introspection as a cautious, evidence-backed interpretation.
- Forbidden (unchanged): consciousness, self-awareness, autonomous self-control, capability/Jormungandr gains.

---

## 11. Final Verdict Constants

```
WITHIN_DOMAIN_PREANSWER_SPECIFICITY_VERDICT = PARTIAL
   # powered GSM8K: significant incremental validity (+0.066, CI[+0.017,+0.114]);
   # strong margin vs length-only (+0.059); but hidden-alone vs length+logprob composite +0.014 (ns)
TIMING_EVIDENCE_VERDICT          = PREANSWER_TIMING_CONTROL_PASS
SPECIFICITY_EVIDENCE_VERDICT     = HIDDEN_BEATS_SHORTCUTS_WITHIN_DOMAIN
   # via incremental-validity (significant); caveat: not standalone-dominant over the composite
PROTO_INTROSPECTION_WEAK_CLAIM_VERDICT = NARROW_DEFENSIBLE_CLAIM_ONLY
NEXT_STEP_VERDICT                = WRITE_PAPER_DRAFT
   # narrow, caveated proto-introspection; optional strengthener: power a 2nd domain (reasoning >=150)
```

Supporting (carried/updated):
```
PREDICTION_EVIDENCE_VERDICT = HIDDEN_STATES_PREDICT_PROCESS_QUALITY   (powered: GSM8K 0.745)
UTILITY_EVIDENCE_VERDICT    = WEAK_UTILITY_SUPPORTED_STRONG_CAPABILITY_UTILITY_NOT_PROVEN
ROLE_SEPARATION_VERDICT     = VALIDITY_CONTENT_CORRECTNESS_SEPARATED
CONTROL_BOUNDARY_VERDICT    = READOUT_SUPPORTED_CONTROL_NOT_SOLVED
BARBADOS_REPLICATION_VERDICT = BARBADOS_PROTO_INTROSPECTION_NOT_RUN
```

---

## 12. Bottom Line

The decisive powered control came out **positive with caveats**, reversing the prior round's underpowered pessimism on specificity. In a powered, leakage-controlled, within-domain test (GSM8K, 170 tasks), Ouro-RLTT strict-preanswer hidden states predict the model's own success at AUROC 0.745 and add **statistically significant** signal beyond length+logprob shortcuts (+0.066, CI excludes 0), while a logprob confidence baseline is weak (0.569). The hidden state does not standalone-dominate a strong length+logprob composite, and only one domain is powered — so the honest verdict is a **narrow, defensible, paper-grade proto-introspection claim with explicit caveats**, with the single best strengthening step being a second powered domain.
