# Within-family cross-loop replication (v3) — pre-registered plan

Date: 2026-07-26, before extraction of thinking26/ouro14b (base26 extraction
in progress at write time; no train/eval run yet for any model).

## Question
Do the v2 findings (candidate quality weakly readable at early layers on the
first recurrent pass, at parity with the late basis by loops 3-4; L1-sourced
taps rotated, L2/L3-sourced taps stable) hold across the Ouro family, or are
they RLTT-specific?

## Models
base26 (ByteDance/Ouro-2.6B @1ed0425), thinking26 (Ouro-2.6B-Thinking
@f1edd81), ouro14b (ByteDance/Ouro-1.4B, layer set mapped proportionally from
the sealed {8,16,24,36,47}/48 scheme; fresh taps only).

## Held fixed (sealed run inheritance)
The identical 2,150-group subset (sha-verified copy), split function, tap
class, pairwise training procedure, 36-point grid with val-only selection,
EARLY_MARGIN=0.03, bootstrap seed 20260720 / 10,000 draws.

## Endpoints
1. Per model: loop trend at early layers (L1 vs L3/L4 refit deltas) and parity
   labels vs the model's own late references — the v2 EARLY_READABLE pattern
   replicates if the qualifying-cell structure reappears.
2. crossmodel stage (2.6B geometry only): sealed RLTT taps scored frozen on
   the sibling's features — macro top-1 delta vs the sibling's own local refit
   plus Spearman agreement. Prediction registered by the author: taps transfer
   to thinking26 approximately intact; base26 unknown.

## Interpretation commitments
Any cell that fails to replicate is reported as such per-model; no averaging
across checkpoints.
