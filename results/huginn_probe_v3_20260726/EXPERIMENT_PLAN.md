# Huginn depth-recurrence readability probe (v3) — pre-registered plan

Date: 2026-07-26, before feature extraction.

## Question
Outside the Ouro family: does candidate-quality readability at a fixed capture
locus improve with recurrence depth in Huginn-0125 (3.5B depth-recurrent,
prelude-2 / core-4-recurred / coda-2, hidden 5280)?

## Design
Same 2,150-group CoreContent subset (sha-verified), same splits. One frozen
forward per candidate at num_steps=8; capture = output of the last core block
at each recurrence step (masked mean pooled, fp16). Steps map onto the sealed
machinery's loop axis ([1, 8, 5280] per candidate); the sealed pairwise tap
training, 36-point grid, val-only selection and clustered bootstrap
(seed 20260720, 10k draws) run unchanged.

## Endpoints
1. Refit macro top-1 across steps 1..8: the Ouro-analogue prediction is a
   rising, then saturating trend (readability increases with recurrent
   refinement). The null is a flat profile.
2. Frozen step-transfer (taps trained at step u scored at step v>u): the
   Ouro-analogue prediction is early-step rotation followed by stability.

## Scope commitments
This is a class-generality probe, not a full replication: one capture locus,
no generation-side experiments, no steering claims. Reported per-cell with CIs
regardless of direction.
