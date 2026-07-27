# Frozen conversion #5 — LoRA direction-outcome binding pilot (S3A pilot) (SEALED PLAN)

Sealed: 2026-07-27 ~03:00, BEFORE any training or eval generation. Author: Claude
(session 1fd122fa), under the standing authorization for the frozen-conversion
programme ("Id honestly also like to do the LoRa attempt"). Explicitly a bounded
DIAGNOSTIC: the smallest training-time test of the paper's closing hypothesis
(§8.5/§12) that the readout–control boundary exists because "the frozen model was
never trained to make its injectable branches outcome-distinct."

## Question

Can a short LoRA pass BIND a fixed writable direction to verifier outcomes — so that
injecting +αd at the §8 locus, which does nothing on the frozen model (§8.1–8.2),
raises verified success after training? This does not test S3A's full tournament;
it tests its precondition: direction-outcome coupling is trainable at all.

## Design (all fixed before running)

- Base: models/ouro_rltt_local, frozen; PEFT LoRA r=16, alpha 32, dropout 0.05,
  target_modules all-linear, bf16, gradient checkpointing (the repo's proven
  300-step recipe from train_branching_sft_v1.py), bs=1, grad-accum 8, max 400
  steps, lr 1e-4 cosine.
- Injection convention (verbatim §8 machinery): HiddenDeltaLayerHook at layer 24,
  loop 1, single token = last prompt token; α = 0.02 (SAFE_ALPHA_CAP), delta scaled
  by local hidden RMS. Direction d: the top curated basis-bank direction for L24
  (family priority high_yield_recipe_direction → …, perturbation_usable,
  heldout_leakage_free, unit-RMS 2048-d), recorded by id in the results file.
- Training data: the #2 tournament table (slice [1080,1260), 6 verified candidates
  per task) — no new collection. Example = (prompt, candidate text); condition:
  verified-correct candidates train with hook +αd active; verified-incorrect with
  hook −αd active. CE on candidate tokens only. Class-balanced sampling.
- CONTROL ADAPTER: identical everything, but the hook sign per example is assigned
  by a seeded coin flip (decoupled from correctness). Isolates direction-binding
  from generic SFT-on-solutions and from adapter-plus-hook drift.
- Eval (task-disjoint, slice **[1260, 1350)** = 90 fresh tasks, disjoint from all
  prior runs): per task, K=4 samples per arm, temp 0.7/top_p 0.95, max_new 320,
  FinalAnswerStop, marker-gated parse, external verifier only. Arms:
  1. binding-LoRA + inject +αd at prefill
  2. binding-LoRA clean (no hook)
  3. control-LoRA + inject +αd
  4. control-LoRA clean
  5. frozen base + inject +αd  (the §8 negative, re-measured on this slice)
  6. frozen base clean
  Same per-task seeds across arms (paired). Optional secondary: binding-LoRA − αd
  (should not raise success if binding is directional).
- Gates before eval: zero-delta hook forward == no-hook forward (bit-identical) on
  3 probe prompts, adapter loads cleanly, training loss finite throughout.

## Sealed endpoints

Primary: Δ_bind = success(arm1) − success(arm2), task-clustered bootstrap 2000
rounds seed 20260727. Comparators: Δ_ctrl = success(arm3) − success(arm4) and
Δ_frozen = success(arm5) − success(arm6).
Success criterion for binding: Δ_bind > 0 with CI excluding zero AND Δ_bind
exceeds Δ_ctrl (paired difference-of-differences CI excluding zero).

## Sealed verdict labels

- BINDING_TRAINABLE (criterion met)
- BINDING_NOT_DETECTED (Δ_bind CI spans zero or ≤ Δ_ctrl)
- ADAPTER_IMPROVES_UNCONDITIONALLY (arm2 − arm6 CI excludes zero and Δ_bind CI
  spans zero: the adapter helps but injection adds nothing — reported as its own
  informative outcome, not spun as binding)
- TRAINING_UNSTABLE (non-finite loss or gate failure; recorded, not retried with
  new hyperparameters)

Budget: one training run per adapter (max 400 steps each), one eval pass; no
hyperparameter search, no direction re-picks, no extra steps after launch.
