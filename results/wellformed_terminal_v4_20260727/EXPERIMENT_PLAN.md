# Frozen conversion #4 — all-well-formed terminal-selection pool (SEALED PLAN)

Sealed: 2026-07-27 ~02:00, BEFORE any generation. Author: Claude (session 1fd122fa),
under the standing authorization to run the frozen-conversion programme.

## Question

§6.5 of the paper established forced terminal selection above an exact matched-random
null, then showed the margin is mostly malformed-output avoidance: a hidden-free
shortcut selector captures it equally well, leaving content-sensitive commitment
unresolved on five well-formed groups. This experiment removes malformedness from the
pool BY CONSTRUCTION: every candidate in every group commits to a parseable FINAL
ANSWER. Any selection margin that survives is content-sensitive by elimination;
any margin that vanishes quantifies how much of §6.5 was form.

This is queue item #1 of §12.4 ("the single item that would most change the paper's
standing"). Both outcomes are informative and will be reported.

## Design (all fixed before running)

- Model: models/ouro_rltt_local (canonical frozen RLTT), extractor
  BGTransformerFeatureExtractor unchanged (pooled [3,4,2048], layers {24,36,47} x loops 1-4).
- Task source: logic_tasks.jsonl, category synthetic_propositional, proof_depth 2-4,
  hash-ordered pool capped at 4000 (NEVER raise the cap), slice offsets **[860, 1080)**
  (disjoint from v2 main [0,170), v3 extension [170,680), depth-alloc [680,860)).
  220 tasks; prospective split via sealed split_for(task_uid): 113 train / 46 val / 61 heldout.
- Generation: identical sampling to v2 (temperature 0.7, top_p 0.95, max_new 320,
  FinalAnswerStop early stop), in ROUNDS of 4 candidates. Per-round seed
  _sid(SEED, task_uid, round). SEED = 20260727.
- **Well-formed** := found_final_marker AND parsed answer is not None (same
  marker-gated parse as v2). Form only — correctness is NEVER consulted when
  filtering or ordering candidates.
- Keep the FIRST 4 well-formed candidates in (round, within-round index) order;
  stop sampling a task once 4 are kept or MAX_ROUNDS=4 rounds (16 draws) are spent.
  Tasks ending with 2-3 well-formed candidates stay in the pool (Poisson-binomial
  null handles variable group size); tasks with <2 are dropped and counted.
- Features: full_features only (prompt + full generated text), computed for KEPT
  candidates only.
- Selector protocol: verbatim v2 terminal-selection analysis — standardize->PCA->
  L2-logistic, hyperparams by grouped CV on train+val only; heldout opened once.
  Shortcut control uses the identical 4-dim terminal_shortcut_vec (its
  found_final_marker coordinate is constant 1.0 in this pool by construction —
  that is the point, not a bug).

## Sealed endpoints

Primary: forced top-1 success on informative heldout groups vs the exact
matched-random baseline (Poisson-binomial, P(X >= observed)), plus task-clustered
bootstrap (2000 rounds, seed 20260727) on the paired difference.
Secondary (the §6.5 resolution): paired hidden-vs-shortcut top-1 difference on the
same informative heldout groups, task-clustered bootstrap, same rounds/seed.
Power floor: MIN_USEFUL = 25 informative heldout groups.

## Sealed verdict labels

- WF_INFORMATIVE_YIELD_TOO_LOW (< 25 informative heldout groups)
- WF_CONTENT_SELECTION_ESTABLISHED (paired diff > 0, bootstrap excludes zero,
  exact p < 0.05, order-invariance and duplicate controls clean)
- WF_CONTENT_SELECTION_ABOVE_RANDOM_UNDERPOWERED (diff > 0, CI spans zero)
- WF_CONTENT_SELECTION_NOT_ABOVE_RANDOM (diff <= 0)
Secondary flag (reported alongside, never overriding the primary):
- HIDDEN_BEATS_SHORTCUT / HIDDEN_MATCHES_SHORTCUT / SHORTCUT_ABOVE_HIDDEN
  (sign of the paired delta and whether its CI excludes zero).

## What would count as resolving the §6.5 hedge

- ESTABLISHED + HIDDEN_BEATS_SHORTCUT: content-sensitive commitment is real and
  hidden-specific.
- ESTABLISHED + HIDDEN_MATCHES_SHORTCUT: content selection is real but surface-
  visible (length/logit statistics carry it).
- NOT_ABOVE_RANDOM: the §6.5 margin was malformedness, fully; the commitment gap
  stands as stated.
No post-hoc re-slicing, no added selector families, no budget change after launch.

## SEALED EXTENSION 1 (declared 2026-07-27 ~05:40, before any selector fitting)

The primary run hit the pre-registered power gate: 16/60 informative heldout groups
(floor 25) because the all-well-formed pool's candidate success rate is 0.856 and
most groups are all-correct. The analysis exited AT the gate — no selector was
trained, no heldout selection outcome was computed or seen. Extension, declared
before any such computation:
- 150 additional tasks, slice offsets **[1350, 1500)** (disjoint from all prior
  runs including tournament [1080,1260) and LoRA eval [1260,1350)); identical
  generation protocol, seeds _sid(SEED, task_uid, round) unchanged (task-keyed).
- Analysis pooled over primary + extension cohorts (per-cohort counts reported);
  identical endpoints, power floor, and verdict labels. If the pooled yield is
  still < 25, the verdict remains WF_INFORMATIVE_YIELD_TOO_LOW and no further
  extension is run (the finding is then "the RLTT well-formed failure mode is too
  rare on this domain to power the question at feasible budget").
- Runs after the LoRA pilot to respect the GPU chain.
