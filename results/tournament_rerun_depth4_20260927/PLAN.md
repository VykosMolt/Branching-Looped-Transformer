# Run 2: prefix-prune tournament rerun on a pool that can differentiate policies

Sealed 2026-09-27, before any generation for this run. The SHA-256 hashes of this file and `run2_tournament.py` are recorded in `SEAL.sha256`.

## Why

The sealed tournament (`opi/branching/tournament_v4_20260727/EXPERIMENT_PLAN.md`) returned `POOL_CANNOT_DIFFERENTIATE`. On its depth 2–4 tasks, the any-correct-of-6 rate was 0.978, above the 0.95 ceiling of its feasibility check. In that run, any-correct-of-6 was 0.991 at depth 2 (110 tasks), 0.981 at depth 3 (53) and 0.882 at depth 4 (17).

## What is unchanged

Everything in the sealed design, via `bg_v4_tournament.py` imported unchanged:
- **Generation:** B = 6 as two seeded rounds of three; T = 0.7, top-p 0.95, at most 320 new tokens, FinalAnswerStop, seed 20260727 via `_sid`.
- **Scoring:** P = 80, S = 2, the frozen DualAnchor locked channels, and cohort z-scored pairwise margins.
- **Policies:** tournament, best-of-3, best-of-4, and 500 random-prune replicates.
- **Statistics and decisions:** exact token accounting, 2,000-round paired task-clustered bootstrap, spend guard, feasibility floor [0.15, 0.95], and the sealed verdict labels and their order.

## What changes

1. **Task set.** The first 180, in hash order, of the depth-4 Horizon tasks that no earlier run used. These are the tasks outside the capped pool's [0, 3000), which earlier runs and Run 1 occupy; 247 such tasks exist.
2. **Token IDs.** Prompt and generated token IDs are also saved.

## Interpretation, fixed now

- The sealed labels apply as written.
- If the pool again fails the feasibility check, the result is reported as such, with no further retries.
- The prefix-score AUROC and the survivor oracle retention are reported alongside and never override the verdict.
