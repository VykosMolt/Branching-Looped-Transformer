# Proto-Introspection Paper — Short Handoff

**For:** the writer (human or model) drafting the paper.
**Full package:** `proto_introspection_paper_writing_package_2026-06-17.md` (+ `.json`). Read that for the complete plan; this is the one-page brief.

## The claim (use exactly this scope)
> **Looped-model hidden states expose readable, pre-answer, partly shortcut-independent process-quality signals about the model's own ongoing computation.** Operational, not psychological.

Recommended thesis: *In a frozen looped LM (Ouro-RLTT), intermediate hidden states carry readable, pre-answer process-quality information predicting eventual success in a powered domain and adding significant information beyond length/logprob shortcuts — but readout does not yield autonomous control or capability gains.*

Recommended title: **Readable Process-Quality Signals in Looped-Transformer Hidden States.**

## The three numbers that carry the paper
1. **Relational readout:** HH pairwise **0.952**; pointwise-linear control **0.2175** (below chance); flip ρ≈**−0.94** → signal is relational, not an artifact.
2. **Pre-answer prediction (cleanest):** GSM8K, 170 powered tasks, strict cut excluding gold value → hidden **AUROC 0.745** [0.707, 0.783], task-grouped CV.
3. **Specificity:** hidden+all beats length+logprob composite by **+0.066** [+0.017, +0.114] (SIG). **Caveat — state it plainly:** hidden-alone (0.745) only **ties** the composite (0.731; Δ+0.014, ns). Significant *incremental* validity, not domination.

## Hard do-nots (these sink the paper if violated)
- No consciousness / self-awareness / sentience / mental states / self-understanding.
- No "hidden states dominate all trivial predictors" — they tie the composite alone.
- No established cross-domain claim — second domain (ARC, 105 tasks) is **UNDERPOWERED**.
- No autonomous control / steering-works claim — every frozen intervention failed (K-matched sampling 0.75 ≥ fork 0.611; greedy-fork new-correct 0.0; steering closed).
- No Jormungandr/branching capability gain — mechanism validated, capability not shown; S3A not run.
- Don't cite trajectory **0.854** as a clean headline — it was 95% answer-leaked (task-grouped drops to 0.624). Use it only as caveated motivation; the clean number is 0.745.
- Use "evidence/supports," never "proof."

## Results order
R1 relational readout → R2 trajectory (caveated) → R3 strict pre-answer (cleanest) → R4 specificity (incremental SIG; ties composite) → R5 role separation → R6 readout/control boundary.

## Readiness
```
PAPER_READINESS_VERDICT = READY_TO_DRAFT_WITH_CAVEATS
PROTO_INTROSPECTION_CLAIM_SCOPE = NARROW_OPERATIONAL_WEAK_FORM
NEXT_STEP_VERDICT = BEGIN_PAPER_DRAFT
```
Optional strengthener (not blocking): power a second strict-pre-answer domain (reasoning ≥150 tasks with real pre-answer reasoning) to lift specificity toward a clean PASS.
