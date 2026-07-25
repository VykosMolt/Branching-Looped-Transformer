# Second-Domain Strengthener — Pre-Flight Findings (NEGATIVE)

**Artifact:** `artifacts/reports/proto_introspection/proto_introspection_second_domain_preflight_2026-06-17.md`
**Work dated:** 2026-06-17 · **Run:** 2026-06-23 · **Scope:** bounded GPU pre-flight only; frozen Ouro; no checkpoint changes; no S3A; external labels only.

## Goal
The paper package listed an optional strengthener: *power a second strict-pre-answer domain with **real** pre-answer reasoning* (≥150 tasks), to confirm the GSM8K incremental-validity effect (hidden+all − length/logprob composite = +0.066 [+0.017,+0.114]) replicates and lift the specificity pillar toward a clean PASS. The prior secondary (ARC-Challenge) failed because MCQ answers are **front-loaded** (pre-answer median 14 tokens — no reasoning to probe).

## Process check
`jobs` empty; no stale recapture/training processes (only an unrelated searxng resource-tracker); GPU idle (654 / 12227 MiB). No action needed.

## Verdict
```
SECOND_DOMAIN_STRENGTHENER_VERDICT = REJECTED_AT_PREFLIGHT_BOTH_CANDIDATES
CROSS_DOMAIN_VERDICT               = PARTIAL_NOT_FULLY_ESTABLISHED  (unchanged; now better characterized)
ACTION                             = STOP_AND_DOCUMENT (user decision 2026-06-23)
```
Two candidate domains were tested with full manual inspection of generations. Both fail for **principled, model-specific reasons** — not for lack of effort. The cross-domain limitation stands, now with a concrete explanation of *why* a clean second domain is hard for this model.

---

## Candidate 1 — SVAMP (easy math word-problems): REJECTED
Probe: 8 problems, `gsm_prompt`, max_new 256, temp 0.7. Two independent fatal flaws (full generations inspected):

1. **Front-loading.** In 4/8 the model emits `Final Answer: \boxed{N}` as the *first 3 tokens*, then reasons — so the strict pre-answer cut is empty (`n_pre_tok = 3`). SVAMP is too trivial; the model blurts the answer immediately, exactly the ARC failure mode. Only 2/8 reasoned-then-answered (n_pre 165, 192).
2. **Verifier mismatch.** The model outputs `\boxed{N}`, but the GSM8K parser (`parse_gsm_answer`) matches only `FINAL ANSWER: <bare-number>` and otherwise grabs the **last number in the text**. Result: 5/8 correct answers mislabeled incorrect (gold 27→parsed "62"; 4→"19"; 31→"$26"; 720→"24"; 64→"450"). Labels would be ~60% wrong.

Plus 1 truncation and 1 fully degenerate generation (hallucinated a fake multiple-choice list). **SVAMP is unusable.**

## Candidate 2 — MATH (hendrycks, Level 1-3 numeric): REJECTED
Built a dedicated **hardened** recapture (`utilities/tests/manual/proto_introspection_math_recapture.py`): balanced-brace `\boxed{}` extractor, numeric-only gold (Fraction compare), boxed-first verifier, strict pre-answer cut = before min(first `\boxed`, FINAL marker, gold-value occurrence), and **truncated/non-numeric samples dropped** (label cannot be trusted). Two smokes, all generations inspected:

| Smoke | Levels / subjects | kept / dropped | sample_acc | preanswer median |
|---|---|---|---|---|
| 1 | L1-2, all 7 subjects | 9 kept / 7 dropped | 0.889 | 160 tok |
| 2 | L2-3, 5 selected subjects | 13 kept / 7 dropped (35%) | **1.00** | 223 tok |

**Front-loading was solved** — MATH forces genuine long pre-answer reasoning (median 160-223 tokens, like GSM8K's 163). But a **deeper, fatal confound surfaced:**

> **For this model, failure on math = truncation, not wrong answers.** The model either (a) solves the problem and commits a `\boxed{}` answer (≈always correct), or (b) yaps past the token budget without ever committing (truncated). Across both smokes, parseable samples were ~21/22 correct. Dropping truncated samples therefore drops virtually all the failures → **parseable base rate ≈ 1.0 → no negative class → a success/failure probe is impossible.**

Keeping truncated-as-failure does not rescue it: a truncated sample's pre-answer is the full 512-token run, so the label becomes "predict whether the model runs out of budget" — a length/truncation artifact, not a pre-answer process-quality signal, and it would be trivially captured by the length shortcut baseline. Either way MATH cannot deliver a balanced, clean success/failure label for this model. This is the deep form of the operator's warning that *"this model likes to yap extensively on math."*

---

## Why a clean second domain is hard for this model (the general lesson)
A usable strict-pre-answer success/failure domain needs **both**:
1. **Answer-last with real reasoning** — so the pre-answer cut contains genuine process (rules out front-loaded MCQ: ARC, SVAMP-trivial), **and**
2. **The model commits a parseable answer in a balanced correct/incorrect mix** — so there is a negative class (rules out domains where the model's failure mode is non-commitment/truncation: MATH).

GSM8K hits the sweet spot (hard enough to force reasoning, but the model commits a numeric answer → base rate 0.599, balanced). Trivial domains front-load; hard domains truncate. The window is narrow, which is itself the explanation for why cross-domain establishment is incomplete.

## Impact on the paper
- `CROSS_DOMAIN_VERDICT` stays **PARTIAL_NOT_FULLY_ESTABLISHED** — unchanged, but now defensibly characterized: the second-domain gap is a measured property of the model's commit/truncate behavior, not an untested hole.
- The powered primary (GSM8K, 170 tasks, hidden 0.745, incremental +0.066 SIG) and the underpowered ARC secondary remain the evidence base.
- No claim changes. No numbers change. This is a documented **negative finding**, added to the package limitations and honesty ledger.

## Artifacts
- Script (kept, reusable, hardened): `utilities/tests/manual/proto_introspection_math_recapture.py`
- Throwaway smoke captures removed: `math_recapture_smoke.pt` / `_index.json` (not load-bearing).
- SVAMP probe was a scratch script (not committed to the repo tree).
- No full recapture was run for either candidate (both rejected at pre-flight before spend).
