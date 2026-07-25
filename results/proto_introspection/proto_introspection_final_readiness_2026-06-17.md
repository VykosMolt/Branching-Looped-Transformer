# Proto-Introspection — Final Readiness (post-controls)

**Date:** 2026-06-17 (run 2026-06-22)
**Inputs:** `proto_introspection_evidence_matrix_2026-06-17.{md,json}` (prior) + `proto_introspection_controls_2026-06-17.{md,json}` (this round).
**One-line status:** the targeted controls were run and were **deflationary** — the headline trajectory/success-prediction result was leakage- and memorization-inflated; a genuine but modest pre-answer signal survives that is **not yet shown to be beyond trivial shortcuts**.

---

## Headline verdict

```
PROTO_INTROSPECTION_WEAK_CLAIM_VERDICT = SUBSTANTIALLY_SUPPORTED_REQUIRES_MORE_CONTROLS
NEXT_STEP_VERDICT                      = RETURN_TO_EVIDENCE_MATRIX
```

The weak claim is supported in a **narrowed** form: *looped-model hidden states contain a readable, genuinely pre-answer process-quality signal* (it predicts success as well before the answer token as after it, and committed-answer states are at chance). But the **specificity** leg — that this signal is not just a trivial domain/difficulty proxy — is **not established**, and the **success-prediction** numbers previously rated STRONG were inflated by answer leakage (best original cell = 256-token prefix, 95% leaked) and task memorization (random CV +0.108 over task-grouped).

---

## Controls scoreboard

| Control | Verdict | Why |
|---|---|---|
| A — pre-answer leakage | **PREANSWER_TIMING_CONTROL_PARTIAL** | Clean preanswer cut (leak 0.0) AUROC 0.690 ≈ full-answer ceiling 0.698; leaked subsets at chance (0.48–0.50). Signal is pre-answer, but modest and domain-confounded; within-domain 0.55–0.58 (CIs span 0.5). |
| B — shortcut baselines | **SHORTCUT_BASELINE_CONTROL_PARTIAL** (leans negative) | On clean states hidden ≤ domain/length shortcut (preanswer Δ −0.012; prompt_only Δ −0.084); combined adds ~0. Hidden beats shortcut only via leakage (all-prefixes combined 0.761). |
| C — task-grouped heldout | **GROUPED_HELDOUT_CONTROL_PARTIAL** | Survives grouping (0.624 > 0.5) but random overstated by +0.108; under grouping hidden ≈ shortcut; pairwise ranking strong only at 95%-leaked p256. |
| D — Barbados | **BARBADOS_PROTO_INTROSPECTION_NOT_RUN** | Not instrumented for loop-state dumps; out of bounded scope; not invented. |

---

## Is weak proto-introspection paper-ready?

**Not in its strong/headline form.** A paper that claims "looped hidden states predict the model's own success" is not supportable today, because on clean (leakage-free) states that prediction does not beat a trivial domain/length baseline.

**A deliberately narrow paper is defensible now**, built only on the legs that survived:
- A clean **pre-answer** result: hidden states before the answer token predict eventual external-verifier success as well as states after it (preanswer 0.690 ≈ full ceiling 0.698; existing-data leaked subsets at chance) — reported with its modest effect size and the explicit caveat that within-domain it is underpowered and not beyond domain-difficulty.
- **Role separation** (validity / content / correctness) — strong, untouched.
- **Control boundary** (readout exists, frozen-backbone control does not) — strong, untouched.
- The **HH relational-preference** result (95.2% pairwise) — framed as preference between two *complete* candidates, explicitly **not** a pre-answer introspection claim.

---

## What must NOT be claimed

- That the signal is **beyond trivial shortcuts** (it is not, on clean pre-answer states).
- The trajectory **pairwise 0.854** as evidence (leakage + memorization artifact).
- Consciousness, self-awareness, human-like introspection.
- Autonomous self-control, or that the model internally uses the signal for control.
- Jormungandr / capability improvement.

---

## What remains before drafting (cheapest path first)

1. **Revise the evidence matrix** (the literal `NEXT_STEP`): P1 → split into *preference* (STRONG, unaffected) vs *own-computation success* (PARTIAL, leakage-corrected); P2 → sharpened PARTIAL; P3 → WEAK. This is a documentation pass, no compute.
2. **One powered within-domain pre-answer control** (optional but high-value): scale the recapture to ≥150 tasks/domain, compute per-domain AUROC CIs, and add a **logprob/entropy** baseline. This is the single experiment that could move specificity from PARTIAL to PASS (or cleanly to FAIL). Locally feasible — it is just a larger version of `proto_introspection_preanswer_recapture.py` (the 120-task run took ~42 min).
3. Only after (2) decide between `WRITE_PAPER_DRAFT` (narrow claim) and a broader claim.

**Further GPU work needed?** Only for the optional powered control (item 2). Revising the matrix and writing the narrow claim need no further GPU.

---

## Artifacts

- Report: `artifacts/reports/proto_introspection/proto_introspection_controls_2026-06-17.md`
- JSON: `artifacts/reports/proto_introspection/proto_introspection_controls_2026-06-17.json`
- Raw metrics: `artifacts/reports/proto_introspection/controls_analysis_results.json`
- Recapture features: `artifacts/reports/proto_introspection/preanswer_recapture.pt` (+ `_index.json`)
- Scripts: `utilities/tests/manual/proto_introspection_preanswer_recapture.py`, `proto_introspection_controls_analysis.py`
- Prior (unchanged): `artifacts/reports/proto_introspection/proto_introspection_evidence_matrix_2026-06-17.{md,json}`
