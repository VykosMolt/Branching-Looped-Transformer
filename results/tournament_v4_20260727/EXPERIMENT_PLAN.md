# Frozen conversion #2 — matched-budget prefix-prune tournament (SEALED PLAN)

Sealed: 2026-07-27 ~02:45, BEFORE any generation. Author: Claude (session 1fd122fa),
under the standing authorization for the frozen-conversion programme.

## Question

Depth allocation (#1) returned a double null: tap features cannot route *loop count*
per task better than a random histogram. Selective prediction (#3) is positive: the
readout converts to a deployment gain when it selects among already-executed
computations. The remaining untested actuator between them: can the frozen survival
readout steer compute MID-GENERATION — prune weak branch prefixes early and spend the
saved tokens finishing strong ones — and beat spending the same token budget on
independent full candidates? This is the frozen, inference-only shadow of the S3A
training-time tournament the paper motivates.

Prior support (fixed BEFORE this design, small-n): bg_trajectory_prediction_2026-05-18
verdict STRONG — prefix-stage tap margins predict continuation success (20 tasks,
registry heads). This experiment is its powered, sealed, task-disjoint successor with
exact budget accounting.

## Design (all fixed before running)

- Model: models/ouro_rltt_local, sampling identical to v2 horizon generation
  (temperature 0.7, top_p 0.95, max_new 320, FinalAnswerStop).
- Tasks: logic_tasks.jsonl synthetic_propositional depth 2-4, hash-ordered pool
  (cap 4000, never raised), slice offsets **[1080, 1260)** — 180 tasks, disjoint from
  ALL prior runs ([0,170) v2, [170,680) v3, [680,860) depth-alloc, [860,1080) wf-pool).
  No selector training occurs, so all 180 tasks are evaluation; split_for tags are
  recorded for provenance only.
- Data collection (one pass): per task, B=6 candidates run to completion (marker
  stop or 320). SEED = 20260727. AMENDMENT 2 (pre-data, 2026-07-27 ~05:45): the
  originally specified single k=6 batch OOMs the 12 GB card (first launch failed
  with zero usable records; a code bug in the same launch also meant no records
  were ever written). B=6 is therefore drawn as 2 seeded rounds of 3
  (num_return_sequences=3, per-round seed _sid(SEED, task_uid, round)); sampling
  parameters unchanged. No generated data existed before this amendment.
  Per candidate: full text, exact generated-token count, marker/parse/success labels
  (marker-gated parse, external verifier only), pooled [3,4,2048] features for
  (a) prompt + first-P-token prefix text and (b) prompt + full text. P = 80.
- Frozen scorer (the ONLY head, no alternates, chosen from prior data): the paper's
  locked DualAnchor recipe — dualanchor_role_channels() for roles MIX_CODE_REASONING
  + MIX_OBJECTIVE_ALL, arch AntisymLinear, configs {24,36,47}_L4 (6 channels), from
  the constrained bank pinned in bg_core_tap_audit_v1_common. Candidate score =
  per-channel mean antisymmetric margin vs cohort (score_diff on config_vector
  slices), z-scored per channel within cohort, averaged across channels. Cohort =
  the candidates being compared at that decision (never across tasks/stages).
- Policies simulated over the SAME candidate table (exact token accounting,
  tok(c) = generated tokens of candidate c, pre(c) = min(tok(c), P)):
  1. TOURNAMENT: score 6 prefixes; keep top S=2; final answer = tap-pick among the
     2 completed survivors (full-feature scores). Cost = sum_6 pre(c) + sum_2 max(0, tok(c)-P).
  2. BASELINE best-of-3: candidates with draw indices {0,1,2}; final = tap-pick among
     their completed texts. Cost = sum_3 tok(c). Cap-level design match:
     6*80 + 2*240 = 960 = 3*320; actual per-task costs reported for both arms.
  2b. BASELINE best-of-4: draw indices {0,1,2,3}, same rule. Pre-launch amendment
     (added before any generation, after a synthetic smoke test exposed the issue):
     early-terminating candidates break the cap-level match — the tournament's
     ACTUAL spend can exceed best-of-3's (prefix cost of 6 is paid in full while
     full candidates terminate below cap). Best-of-4's actual spend strictly
     exceeds the tournament's whenever every continuation is a suffix of a
     candidate, so it brackets the tournament from above.
  3. RANDOM-PRUNE control: tournament structure, survivors chosen uniformly at
     random (500 replicates, torch.Generator seed 20260727), final = tap-pick among
     the 2. Gate: tournament success must exceed the replicate p95.
  4. Oracle ceilings (reported, not verdicts): any-success of 6, of {0,1,2}, of the
     tap-chosen survivors (survivor oracle retention).

## Sealed endpoints

Primary: paired per-task success difference TOURNAMENT − BASELINE, task-clustered
bootstrap 2000 rounds seed 20260727. Gate (checked first): TOURNAMENT vs
RANDOM-PRUNE replicate distribution — mean success must exceed the p95 of the 500
random-prune replicates, else the allocation is not signal-driven.
Feasibility floor: if any-success-of-6 (pool ceiling) < 0.15 or > 0.95 the pool
cannot differentiate policies.

## Sealed verdict labels

- POOL_CANNOT_DIFFERENTIATE (ceiling out of [0.15, 0.95])
- PRUNING_NOT_SIGNAL_DRIVEN (tournament <= random-prune p95)
- TOURNAMENT_BEATS_MATCHED_BASELINE (gate passed, paired diff vs best-of-3 > 0 with
  CI excluding zero, AND — spend guard — if the tournament's actual mean spend
  exceeds best-of-3's by more than 5%, the tournament must also be >= best-of-4
  (paired diff >= 0); otherwise the verdict is TOURNAMENT_BEATS_BUT_SPEND_MISMATCH)
- TOURNAMENT_BEATS_BUT_SPEND_MISMATCH (beats best-of-3 but fails the spend guard)
- TOURNAMENT_MATCHES_MATCHED_BASELINE (gate passed, CI spans zero)
- TOURNAMENT_WORSE_THAN_MATCHED_BASELINE (gate passed, diff < 0, CI excludes zero)

Reported alongside, never overriding: survivor oracle retention, both arms' actual
token spends, prefix-score vs final-success AUROC (the powered replication of the
May-18 STRONG cell, on 1080 candidates).
No post-hoc head swaps, no S or P changes, no replicate-count changes after launch.
