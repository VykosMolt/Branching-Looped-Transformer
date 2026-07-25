# Content-Selection Taps (CoreContent)

Updated: 2026-06-06

Memorable entry point for the **content / final-selection** tap line. This is the component
that *ranks within an already-handed-off candidate set* — it is **not** branch survival
(that is DualAnchor, see `dualanchor-architecture-baseline.md`) and **not** branch
generation. Detailed run write-ups:

- `corecontent-dataset-expansion-v2.md` — the v2 dataset-expansion + refit run (this doc's headline).
- `core-domain-tap-audit.md` — the core-domain tap audit + DualAnchor readiness that produced the constructed taps and clean labels.

## What content selection is

Each problem is a **group of candidate answers**, one correct/preferred and the rest
wrong/rejected. The frozen Ouro model digests each candidate (one read-only forward, 4
loops), we mean-pool hidden states to `(3 layers [24,36,47] × 4 loops × 2048)`, and a small
**antisymmetric linear tap** `w` ranks them via `layernorm(stateᵢ − stateⱼ)·w`. Win metric =
`top1_oracle` (the top-scored candidate is a max-reward one). Same digest-and-compare engine
as the HH-RLHF preference evaluator, generalized to five domains:

| Domain | positive | negatives |
| --- | --- | --- |
| alignment | dataset **chosen** response | dataset **rejected** response (real) |
| reasoning / logic | correct MCQ option | dataset distractors (real) |
| math | exact correct answer | numeric perturbations (constructed) |
| coding | canonical solution | deterministic mutants (constructed) |

The model never decides correctness (labels come from datasets / parsers / verifiers), and
tap scores are never used as labels.

## Lineage and bottom line

1. **v1 (tap crafting):** every crafted content tap lost to the broad-objective baseline
   `mixedhead_MIX_HH_OBJECTIVE`. Diagnosed cause: tiny per-domain data (coding 30, reasoning
   5, math 66, logic 80, alignment 200 reward-diverse groups).
2. **Core-domain tap audit:** locked DualAnchor for survival (unchanged), produced clean
   verifier/exact/MCQ/preference labels, kept science/anatomy diagnostic-only.
3. **v2 (dataset expansion + refit, 2026-06-05/06):** expanded the starved domains 27–520×
   (coding 1,733 · reasoning 2,600 · math 3,200 · logic 2,199 · alignment ~26,000
   reward-diverse), re-extracted frozen features (64 shards, 4.87 GB, 0 errors), and refit
   small taps. A crafted tap finally **beat the baseline on the untouched heldout**.

### Headline (heldout core macro top1)

| Policy | macro | 95% CI |
| --- | ---: | --- |
| **CoreContent v2 blockwise (24/36/47)** | **0.6691** | 0.645–0.690 |
| domain-gated router | 0.6610 | |
| `mixedhead_MIX_HH_OBJECTIVE` (v1 winner) | 0.5525 | 0.526–0.577 |

`CORECONTENT_DATASET_EXPANSION_REFIT_V2_STATUS = V2_CORECONTENT_READY`;
`BG_CORECONTENT_V2_PHASE2B_READINESS_VERDICT = READY_FOR_PHASE2B_WITH_V2_CORECONTENT`.

### Honest caveats (follow-up stress tests)

The +0.117 headline is partly an artifact of **constructed negatives**:

- **Real-negative domains only** (reasoning/logic/alignment): blockwise 0.585 vs HH 0.521 →
  **+0.063** (CIs barely disjoint). Constructed domains (coding/math) carry the rest.
- **Coding negative hardness:** the tap is robust to syntax + semantic mutants (~0.94), but
  against **real, compiling solutions to *other* problems** (a relevance test) it collapses to
  **0.58** — it learned *corruption detection*, not prompt-relevance. HH's low raw coding
  score is mostly its own failure on syntactically-broken code (HH = 0.85 on compile-valid).
- **Relevance retrain (hardening):** retraining coding *with* wrong-problem negatives did **not**
  close the gap (wrong-problem 0.593 → 0.583), so prompt-code relevance is **not linearly
  accessible** in these pooled L24/36/47 features — an intrinsic ceiling, not a data artifact.
- **Layer 47 is dead weight:** a 2-channel 24+36 tap equals/beats the 3-channel one (0.674 ≥
  0.669) and is slightly better on real negatives.

## Locked selector (re-locked after hardening)

- **Content / final selection:** `CoreContent_v2_blockwise_pruned_24_36` — 2-channel
  antisymmetric tap, layers **24 + 36** (layer 47 pruned). Fallback: `mixedhead_MIX_HH_OBJECTIVE`.
  Artifact: `artifacts/reports/probes/bg_corecontent_dataset_expansion_refit_v2_2026-06-04/corecontent_v2_policy.pt`.
- **Branch survival:** DualAnchor (`MIX_CODE_REASONING` + `MIX_OBJECTIVE_ALL`) — **unchanged**.
- **Terminal:** top5/full survivor-set handoff retained; the content selector ranks *within* it.
- **Science / anatomy:** diagnostic-only (science-aux gave +0.015 core, not promoted).

### Realistic-deployment note

At branch-selection time the real candidate pool is *plausible complete attempts*, which is
closer to the wrong-problem (0.58) regime than the mutation (0.94) regime. So the coding win
should be treated cautiously; the cleaner generalization signals are **alignment** (real
preference pairs, +0.08) and **math** (+0.07), with **reasoning** flat.

## Constraints honored

No steering trained/applied/claimed. No Ouro training, no weight/tokenizer/checkpoint edits,
no tap-registry mutation. `pure_content_taps.pt` / `transplanted_taps.pt` untouched. DualAnchor
branch survival unchanged; terminal survivor-set handoff retained; science/anatomy diagnostic-only.
