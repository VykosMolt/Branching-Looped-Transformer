# Capability parity and migration ledger

This ledger distinguishes code-path parity from benchmark parity.  Hunter-Seeker
v2 implements the active behavioral surfaces below, but it does not yet claim
to reproduce the legacy assisted `6/4/2` result or to improve the strict
autonomous public-game score.  Those are empirical evaluation tasks, not facts
that can be established by unit tests.

Status meanings:

- **Implemented**: present, exercised, and causally tested in v2.
- **Consolidated**: capability retained through a smaller mechanism.
- **Plugin**: explicit optional boundary; absent from autonomous policy unless
  configured.
- **Removed**: no supported behavioral capability or invalid premise.

| Legacy capability | v2 status | Compact mechanism |
|---|---|---|
| Observation/action/outcome adapters | Implemented | Immutable categorical adapters; arbitrary 2-D shapes; ARC animation-gap cache; clickless and positional domains. |
| Public lifecycle | Implemented and corrected | `begin_run -> act -> env.step -> observe -> callbacks`; one successful action creates one transition before any terminal handling. |
| Grid encoder | Consolidated | Deterministic compact grid features by default; injected frozen representations supported. |
| Ouro loop recurrence | Plugin | `OuroLoopRepresentationBackend` reads frozen early/middle/late loop states through a small connector and collapses them into one `Representation`. It never owns the controller, and this path is not itself RLTT. |
| Action prior | Implemented | State-independent outcome-trained empirical task/global action frequencies; explicit teacher distillation updates the same runtime route. It cannot express a route by construction. |
| State-conditioned student policy | New mechanism (2026-07-18) | `student.py`: deterministic nonlinear features over the current global/spatial representation plus candidate action, balanced teacher batches, task-local negative outcomes, explicit positive-only transfer, and one named `student_policy` score term. Student-mode `act` has no teacher lookup. The multi-representation trajectory API is unit-tested but not yet wired into the agent/backbone runtime and is not RLTT. |
| Spatial click predictor | Consolidated | Object-centric centroids, topology reachability, and learned target affordances rank compact click proposals. No full 64×64 decoder is required. |
| Action-conditioned next-frame model | Consolidated | Bootstrap latent/object dynamics predicts change, progress, value, hazard, terminal, and latent/object deltas. Full-frame decoding is optional. |
| Transition ranker | Consolidated | Auditable `ScoreTerm` composition over learned prediction, evidence, topology, affordance, taps, planning, risk, and efficiency. |
| Beam search and transpositions | Implemented | Bounded receding-horizon search; exact graph first, learned model second; uncertainty penalizes imagined branches. |
| Root-only expensive cognition | Implemented | Full object/event/topology perception at the real root; speculative nodes use immutable compact state. |
| Scene parsing | Implemented | Deterministic 4-connected categorical components and translation-invariant signatures. |
| Object tracking and beliefs | Implemented | Stable tracks, velocity, disappearance/reappearance, pruning, and outcome-grounded controllability/hazard/reward beliefs. Click targets ground through `_target_signature`; clickless domains ground through control attribution (below). |
| Ego/control-set localization | Ported 2026-07-17 (stage 1; behaviorally neutral so far) | `ego.py` `ControlAttribution`: influence = normalized mutual information between the executed action and a track's quantized displacement response (count-based causal-action-influence analog), with signature bootstrap at half weight. Multi-body control falls out of the per-track threshold. Outputs: grounds `ObjectState.controllable`; ego-protected contact-hazard attribution into the affordance route; one named bounded term `ego_motion_hazard`. Replaces legacy `NONCLICK_MOVED`/`COUPLED_MOVED` heuristics: drifters and static objects score exactly zero by construction. `enabled=False` is a tested exact no-op. |
| Structured events | Implemented | Appearance, disappearance, movement, transformation, contact, frame change, progress, hazard, and terminal events on the current transition. |
| Hunter/Seeker exploration | Consolidated | One ensemble-disagreement signal: optimistic for safe real probes, conservative in imagination, adjusted by learning progress and competence. |
| Cheap topology | Implemented | Free-space components, largest reachable region, frontier fraction, object adjacency, and reachable object set. |
| Full region/gateway graph | Plugin/future | The exact observed state graph is active; a heavier inferred gateway graph is not in the policy baseline. |
| Multi-affordance objectivity | Consolidated | Directly grounded controllability, hazard, and reward estimates; blocking/reachability is represented by topology. |
| Symbolic heuristic/residual | Consolidated | Named topology, affordance, graph, evidence, and executable-plan terms; no opaque residual is needed in the baseline. |
| Online learning | Implemented | Immediate dynamics/prior/affordance/student update with finite parameter-delta tests; disabling online learning gates all four routes. |
| Replay learning | Implemented | Provenance-separated bounded replay with explicit teacher fraction and no action-lookup API. |
| Trusted exact-state action lookup | Plugin and isolated | `TrajectoryTeacher` behind `TeacherAccessGuard`; autonomous/student `act` performs zero teacher reads. |
| Observation learning heads | Consolidated | Forward effect/change/terminal prediction in the dynamics ensemble; permanence in tracking; inverse-action diagnostic derived from the event ledger. |
| Recent exact transition memory | Consolidated | Exact state graph plus canonical evidence ledger. |
| Generic engrams | Consolidated | Typed evidence records with task/stage/transfer scope, provenance, confidence, polarity, effect, and counterevidence. |
| Terminal/prototype/basin memories | Consolidated | Task-scoped negative evidence, exact graph outcomes, loop/no-change rates, and uncertainty-aware risk. |
| Protected-contact memory | Consolidated | Contact events plus object-signature hazard evidence and topology. |
| Click cooldown | Consolidated | Exact state/action no-change evidence penalizes sterile repeated clicks. |
| Coarse episode memory | Removed from policy | The legacy path was written and reported but not read by policy. Compact reports derive from the canonical ledger. |
| Safety scoring | Implemented | Expected hazard/terminal, object hazard, scoped evidence, and reachability terms. |
| Risk patcher/all-risky recovery | Implemented | One final risk arbiter applies to greedy, beam, teacher-assisted, and epsilon-random choices; all-risky selects least risk. |
| Model-basin sampler | Consolidated | Ensemble doubt, graph cycles/self-loops, no-change rate, and exact successor evidence. |
| Post-veto module | Removed | The active legacy module was diagnostic; real intervention is represented by the single arbiter. |
| Phase teacher pretraining | Plugin | NPZ trajectory teacher can train the prior, dynamics, and state-conditioned student routes while remaining absent from autonomous/student action selection and graph/evidence lookup. |
| Runtime phase templates and solved prefix | Plugin | Exact-state compatibility assistance is available only in `compat_assisted`; it is never reported as autonomous. |
| Passive self-model | Consolidated | `CompetenceState` contains calibrated operational quantities with direct effects on exploration and planning horizon. |
| Context-token injection | Removed | The legacy projector had no demonstrated gradient/behavior route. |
| Temporal context route | Removed pending evidence | No zero-initialized route is admitted without parameter-delta and decision-causality tests. |
| Cortex monitor | Consolidated | Dynamics error, disagreement, prediction error, stagnation, and learning gain are direct telemetry. |
| Evaluator/CLT anchor | Removed | Invalid fixed-order premise; no default policy capability was lost. |
| Role-specific taps | Implemented | Calibrated pointwise taps, strict two-order antisymmetric pair taps, and survival-only top-k retention. |
| Executable world model | New plugin | Replay-verified rule models ranked by coverage, error, and complexity; planning is blocked before verification. |
| Exogenous memory identity | New mechanism (2026-07-17) | `exogenous.py`: cells whose change trajectories replay identically across episodes with differing action histories are masked from the durable-memory state hash, restoring graph/evidence reuse under environment clocks. Transactional identity stays exact; `enabled=False` is a tested exact no-op. No legacy counterpart existed. |
| Relational goal hypotheses | New mechanism (2026-07-17) | `hypotheses.py`: structural goal-relation proposals (equal, mirror, self-mirror, uniform, lattice-canonical equal) scored as bounded potential-difference shaping; completion-gated promotion, contradiction refutation, per-(task, stage) scope, tested exact no-op when disabled. No legacy counterpart existed — the legacy agent had no goal-inference mechanism either. |
| Checkpointing | Implemented | Strict versioned JSON, full config/model/backend/adapter plus tap-bundle/executable-registry signatures, replay/knowledge/runtime/student state, atomic rollback on failed load, weights-only mode, and first-resume decision tests. |
| Measurement/event dumps | Implemented | Typed bounded traces and derived summaries rather than an enormous hand-maintained schema. |
| General dense vision | Not claimed | Both systems are currently strongest on categorical 2-D domains; v2 exposes representation and adapter protocols for future backends. |

## Causal acceptance already covered

- Terminal and completing actions own their transitions.
- Duplicate observation cannot duplicate learning or memory.
- Speculative search cannot mutate durable perception, graph, evidence, or
  model parameters.
- Negative evidence is exactly neutral across unrelated tasks.
- Random exploration passes through the same risk gate as score selection.
- Dynamics weights change on observed data and learned good actions beat
  observed death actions in a controlled task.
- Teacher distillation changes actual runtime parameters while leaving the
  autonomous exact graph/evidence empty.
- A trained state-conditioned student changes the full agent's selected action;
  removing exactly its score term supplies a live candidate-set counterfactual.
- Pairwise tap output is exactly antisymmetric.
- Executable planning is unavailable until replay verification passes.
- Save/load reproduces the first resumed action and score.
- Nonfinite model outputs become finite, conservative values.
- Odd-shaped clickless integration and a live one-step ARC run both produce
  exactly one transition per action.

## Remaining empirical work

The code rebuild is ready for controlled evaluation.  The next evidence gates
are:

1. matched-action-budget public ARC runs for legacy strict mode and v2;
2. task-order reversal to test memory scope;
3. teacher-train then teacher-disabled retention: **state-conditioned v2
   evaluated 2026-07-18; v3 bandwidth fix 2026-07-19 produced the first
   teacher-free ls20 level completions (4/6 distilled episodes versus 0/18
   undistilled); v4 same day closed online erosion (episode-2 degeneration
   gone), added per-task fitted bandwidths (tr87 fragments restored), and
   fixed the wa30 action-inventory collapse; the four-arm attribution run
   (same day) showed the head alone reproduces the full distilled stack's
   result exactly (4/141/80 vs 4/141/80; head-off 0/0/0); v5 with the
   1024-wide head and 32x32 representation reached 6/6 ls20 completions**
   — this gate is closed;
4. GridFeature versus frozen Ouro multi-tap ablation: **offline C1 portion
   evaluated 2026-07-19** (Ouro wins tr87 91 vs 84 and ls20 45 vs 42, exact
   tie on wa30; ~400x encode cost); a live retention comparison remains
   open and is currently low-value since the tie falls on the one game
   whose retention is unsolved;
5. exact graph only versus graph plus dynamics ensemble;
6. uncertainty exploration versus a matched random probe budget;
7. taps versus no taps with matched candidate count and compute;
8. directional-action affordance grounding: **implemented and causally
   tested 2026-07-17** (`ego.py`; tests in
   `utilities/tests/unit/test_hunter_seeker_v2_ego.py`); autonomous gain not
   yet demonstrated — see the experiment note below.

First single-run smoke evidence (2026-07-16, autonomous v2 from scratch,
200-step budget, default config, no teacher, no checkpoint): `ls20` 0 levels
(death at step 144), `tr87` 0 levels (death at step 128), `wa30` 0 levels
(death at step 200).  The legacy strict topology-trio baseline (1/2/0 levels)
is **not** a matched comparison: it ran the trained `sprint4` checkpoint with
the Ouro backbone and pre-loaded trusted trajectories.  These runs establish
only that from-scratch v2 does not clear trio levels inside 200 steps; the
matched-budget comparison of gate 1 remains open.

### Named experiment: exogenous filter on/off trio (2026-07-17)

`utilities/tests/manual/run_hs_v2_exogenous_onoff_trio_v1.py`, matched
200-step budget, seed 0, 4 persistent-agent episodes per (game, arm);
artifacts under
`artifacts/reports/hunter_seeker_v2/exogenous_onoff_trio_20260717/`.

Result: **the memory objective is met on the clock-bearing game; behavior is
unchanged on all three.**  On `tr87` (per-step countdown bar) the mask locks
onto 64 cells and stays stable; graph revisits by episode 4 are 247 with the
filter versus 95 without (2.6×), with 267 versus 421 state nodes.  On `ls20`
a 137-cell mask formed and was then fully retracted by the contradiction
path — conservative unmasking works, at the cost of orphaned-node churn.  On
`wa30` large animated regions mask (272 cells) while exact states already
recur naturally, so filtering mostly churns identity there.  No arm cleared
a stage: restored memory reuse does not by itself supply the goal inference
these games demand.  Known follow-up: mask growth/retraction changes memory
ids and orphans earlier nodes; a hysteresis or confirmation-depth policy
would reduce churn.

### Named experiment: goal hypotheses on/off (2026-07-17)

`utilities/tests/manual/run_hs_v2_hypotheses_onoff_v1.py`, matched 200-step
budget, seeds {0, 1}, 2 persistent-agent episodes per (game, arm) on
template-matched games (sc25, lf52, m0r0, ka59) plus tr87 as a
template-mismatch control; artifacts under
`artifacts/reports/hunter_seeker_v2/hypotheses_onoff_20260717/` and the
post-lattice sc25 rerun under `hypotheses_onoff_sc25_lattice_20260717/`.

Result: **installed and live, not yet decisive.**  Six hypotheses propose
per game and track every transition; after the lattice-canonical template
was added, inspection confirms the true sc25 key/board relation is among
the proposals (Φ = 0.222 across a 2.1× scale and different palettes).
Trajectories diverge measurably with the term active (sc25 seed 0: death
at 62 versus 71), but no arm completed a stage, so no promotion has
occurred and the potential prior remains exploration-grade.  Bottleneck
analysis: fuses of 53–77 steps give the per-(action, target) delta prior
too little support to steer precise clicks within one life.  The follow-ups
that would compound: hypothesis potentials evaluated inside beam-search
rollouts (not only at the root), fuse-aware urgency, and click-proposal
coverage keyed to hypothesis mismatch cells.

### Named experiment: student distillation retention v1 (2026-07-18, gate 3)

`utilities/tests/manual/run_hs_v2_student_retention_v1.py`: student mode,
distilled versus undistilled arms, identical seeds, matched 500-step
budgets, 3 persistent episodes, zero teacher action-selection reads in
every row (guard-audited).  Artifacts under
`artifacts/reports/hunter_seeker_v2/student_retention_20260718/`.

Historical result: **the then-available distillation targets retained action
frequencies, not routes.**
On tr87 seed 0 the distilled arm's episode-1 route overlap is 0.755 versus
0.038 baseline — a real, teacher-free behavioral transfer — but the
transfer is the task-level action prior saturating on the route's dominant
action (prior(action 1) = 1.0), not state-conditioned routing: no arm
completed a level, and an undistilled seed happened to reach the same
overlap by converging on the same dominant action itself.  The distillation
targets in that v1 stack were structurally incapable of route retention: the
action prior is state-independent by design and the linear dynamics ensemble
was too coarse to encode per-state action choice.  This finding motivated the
state-conditioned student head below.  It is a scoped historical conclusion,
not a claim that all distillation is frequency-only.

### Named experiment: state-conditioned student retention v2 (2026-07-18)

The same harness now runs experiment
`hs_v2_state_conditioned_student_retention_v2`: games {ls20, tr87, wa30},
seeds {0, 1}, three persistent-agent episodes, 500-step caps, epsilon zero,
and student mode in both arms.  The arms begin from identical canonical paired
observation signatures and differ before the schedule only by the
teacher-distilled stack's offline `distill_teacher` call; arm order alternates.
Source, environment,
and trajectory hashes remained unchanged, and every one of the 36 rows
recorded zero teacher action-selection attempts and zero teacher
action-selection reads.  Artifact:
`artifacts/reports/hunter_seeker_v2/student_retention_20260718T125224.600923Z_0849a1e20968/summary.json`.

The primary metric fixes a state-only longest ordered alignment before it
looks at actions or successors; positional and action-frequency overlaps are
diagnostics only.  On tr87, the teacher-distilled stack produced 18 correct
action-and-successor rows among 24 ordered aligned states, versus 0 among 9
for the baseline (18/636 versus 0/636 of the shared positional opportunity;
strict-prefix steps 18 versus 0).  The sharpest frequency control is tr87
seed 1, episode 1: the undistilled arm reached 0.755 positional action overlap
but zero ordered transitions, while the teacher-distilled stack had lower positional
overlap (0.613) and four correct ordered transitions.  On ls20 the totals were
14/20 versus 3/9 ordered aligned states (14/324 versus 3/324 opportunity;
strict-prefix steps 14 versus 3).  On wa30 both arms scored zero correct
ordered transitions.

Result: **the state-conditioned stack retains short teacher-free route
fragments on tr87 and ls20, but retention is not universal and does not yet
solve a level.**  No arm produced a level completion, game completion, or
positive progress.  Removing exactly the `student_policy` term from each
realized candidate set changed 502/768 tr87 choices in the teacher-distilled stack
versus 1/768 in baseline; on wa30 it still changed 1058/1200 choices despite
zero retained transitions.  Together with the crafted full-agent test that
changes only the trained head and flips `Decision.action`, this establishes a
live causal policy route, not universal route competence.  The two-arm rollout
distills the prior and dynamics routes as well as the student head, so broad
behavioral differences belong to the distilled stack; a third head-off
rollout is still required for a head-only performance attribution.  This is
state-conditioned behavior distillation, **not RLTT**.

### Named experiment: state-conditioned student retention v3 — bandwidth fix (2026-07-19)

Diagnosis of the v2 near-zero result traced it to the student feature map's
RBF bandwidth, not to the head, the score arbitration, or the distillation
targets.  At `state_feature_scale=80` the implied kernel width (~0.0125) sat
two orders of magnitude below the measured ls20 latent geometry (median
pairwise distance 0.386, adjacent 0.133), so even adjacent recorded states
produced orthogonal features (cosine −0.11) and the head degenerated into a
4-row hash table: 31/54 argmax agreement on its own training states and, in a
live term-breakdown probe, a dominant but wrong `student_policy` term (the
agent followed the head faithfully; the head was wrong).  An offline sweep
(scales 2.5/5/8/12) showed the expected two-sided failure: ≤2.5 re-collapses
toward the dominant-action classifier (far-state cosine 0.63), ≥80 is a
fingerprint.  The default is now `state_feature_scale=8.0`.

Rerun under the matched v2 protocol (games {ls20, tr87, wa30}, seeds {0, 1},
3 persistent episodes, 500-step caps, epsilon zero, zero teacher
action-selection reads in all 36 rows; source/environment/trajectory hashes
unchanged).  Artifact:
`artifacts/reports/hunter_seeker_v2/student_retention_20260719T004554.610980Z_0b661181afff/summary.json`.

Result: **first teacher-free level completions.  On ls20 the
teacher-distilled stack completed level 1 in episodes 1 and 3 at both seeds
(4/6 episodes, progress 1.0, 79 steps, strict route prefix 14, 29/33 correct
ordered action-and-successor transitions); the undistilled stack completed
nothing anywhere (0/18 episodes across games).**  Known limits, all
replicated identically at both seeds: (1) ls20 episode 2 degenerates into a
498/500-step single-action loop — online outcome updates accumulated on top
of the distilled rows drift the head mid-schedule before episode 3's fresh
death-reset trajectory recovers it; online-update/distillation interaction is
an open defect.  (2) tr87 retained nothing at scale 8 whereas the v2 hash
regime retained 18 ordered fragments: tr87's geometry is inverted (adjacent
latent distance 0.026 versus 1.14 median pairwise), so its route revisits
near-identical states that only near-fingerprint bandwidths separate (offline
C1: 94/106 at scale 80 versus 43/106 at scale 8).  No single global bandwidth
serves both games; a per-task bandwidth fitted from distillation-time latent
distances is the principled follow-up.  A two-scale (8/80 split-dimension)
feature map was prototyped and rejected: it halves per-regime capacity and
the fingerprint block destroys ls20 locality (adjacent cosine 0.15).
(3) wa30 remains at zero progress in both arms.  This remains behavior
distillation, **not RLTT**.

### Named experiment: state-conditioned student retention v4 — blocker fixes (2026-07-19)

Three v3 defects were mechanistically diagnosed and fixed at their layer,
then the matched protocol was rerun (same games/seeds/episodes/caps; all 36
rows zero teacher action-selection reads; source/environment/trajectory
hashes unchanged).  Artifact:
`artifacts/reports/hunter_seeker_v2/student_retention_20260719T082439.740060Z_5425924bae77/summary.json`.

1. **Online erosion (v3 defect 1).**  Neutral committed outcomes regressed
   the same trained rows toward a target of zero, washing out distilled
   margins mid-schedule.  Fix: online updates are weighted by
   outcome-target magnitude (`online_target_weighting`); neutral steps are
   informationless for a preference head and now leave parameters unchanged
   while deaths and completions keep full weight.  Result: the
   deterministic episode-2 single-action degeneration is gone; the
   distilled stack completed ls20 level 1 in episodes 2 and 3 at both seeds
   (79–81 steps, strict prefixes 10–14, 25–28 correct ordered
   transitions) with episode 1 the only non-completion — the schedule now
   improves across episodes instead of collapsing.
2. **Per-task bandwidth (v3 defect 2).**  Distillation now fits each
   task's RFF scale from measured geometry: the kernel argument is
   anchored at the low quantile of latent distances between
   differently-labeled demonstration states (`_fit_task_scale`; fitted
   ls20 9.6, tr87 63.6, wa30 5.2, persisted in checkpoints).  The rule
   recovers each game's empirically-best fixed scale from geometry alone,
   and tr87 fragment retention returned: strict prefixes 2–4 with 3–4/5
   correct ordered transitions in all six distilled episodes versus zero
   in every undistilled episode.
3. **Action-inventory collapse (wa30 diagnosis).**  wa30's zero was not
   capacity at first order: `from_npz` declared no action inventory, the
   safe-action provider contributes only actions 1–4, so action 5 entered
   distillation with 158 positive and zero negative examples and won every
   argmax (offline C1 10.2%, all-a5).  `from_npz` now declares the
   inventory (explicit or inferred from the demonstrator's used actions),
   which also makes NPZ state ids live-comparable (exact-index teacher
   matches are reachable again).  Offline C1 rose 3.4x to 34.2% with
   argmax spread across all five actions — but live wa30 retention remains
   zero at both seeds.  The residual is real capacity/geometry: 1,556
   route states against five linear rows (Bayes ceiling 98.7% memoryless),
   with a 124-step level-1 route.  wa30 is the standing case for a richer
   student (larger `state_dim`, sequence-level or route-value training),
   not for further bandwidth or inventory work.

Undistilled baseline: 0 completions and 0 strict-prefix steps in all 18
episodes.  This remains behavior distillation, **not RLTT**.

### Named experiment: four-arm student attribution (2026-07-19)

`utilities/tests/manual/run_hs_v2_student_attribution_v1.py` decomposes the
v4 "distilled stack" effect with four persistent-agent arms under the same
paired seeded protocol (metrics/alignment/seeding imported from the
retention runner; trio, seeds {0, 1}, 3 episodes, 500-step caps; zero
teacher action-selection reads in all 72 rows; the head-only donor's single
offline read is reported separately).  Arms: full distilled stack; the same
distilled stack with the head's `runtime_weight` set to zero (head pressure
off, distilled prior/dynamics/replay still active); a fresh agent receiving
only the donor-distilled head object; and the undistilled baseline.
Artifact:
`artifacts/reports/hunter_seeker_v2/student_attribution_20260719T083710.666069Z_396e635e093b/summary.json`.

Result: **the trained head is the entire retention mechanism.**  Aggregate
(level completions / ordered correct transitions / strict prefix steps):
full stack 4 / 141 / 80; head-only 4 / 141 / 80 — identical totals, with
the same episode-level completion pattern (ls20 episodes 2–3 at both
seeds); head-off 0 / 0 / 0; undistilled 0 / 3 / 3 (one incidental episode-1
fragment).  The distilled prior/dynamics/replay contribute nothing
measurable without the head term and add nothing measurable on top of it.
Counterfactual choice-change rates corroborate: 0.70–1.00 in the two head
arms, exactly 0.00 in the head-off arm.  Practical consequences: (1) the
v2–v4 caveat about stack-level attribution is closed — head-level causal
claims are now performance-level claims; (2) retention experiments can
iterate on the head alone via the donor-transplant harness; (3) the wa30
capacity limit is a head/representation problem, since no other distilled
component supplies any route signal.

### Named experiment: gate-4 student representation ablation (2026-07-19)

`utilities/tests/manual/run_hs_v2_student_repr_ablation_v1.py` compares
GridFeature against the frozen Ouro loop-state backend
(`OuroLoopRepresentationBackend`: sprint4 GridEncoder tokens ->
Ouro-2.6B-Thinking, frozen, bf16 -> early/middle/late loop taps -> the same
32+8x8 Representation shape) as input to the identical student head — same
StudentPolicyConfig, same 96-value state width, same RFF map, same
distillation path; feature content is the only variable.  Compute is
deliberately unmatched and reported per encode (~0.6 ms vs ~230–250 ms,
about 400x).  Offline protocol: distill each trusted route, then C1
recorded-state argmax agreement.  Backbone frozen throughout; no action
selection.  Artifact:
`artifacts/reports/hunter_seeker_v2/student_repr_ablation_20260719T085434.253048Z_ff9d5062de5d/summary.json`.

Result: **Ouro loop taps win exactly where geometry was the diagnosed
failure and tie exactly where capacity is.**  tr87 84/106 -> 91/106 with
the minimum pairwise latent distance improved ~30x (0.00047 -> 0.014) and
a smaller fitted bandwidth (51.6 vs 63.6) — the taps genuinely decompress
tr87's aliased states.  ls20 42/54 -> 45/54.  wa30: 532/1556 for **both**
backends (margins differ only in the 4th decimal) — at a fixed 96-value
width, feature content is irrelevant to wa30, so its bottleneck is the
width itself plus head capacity, not the upstream encoder.

A follow-up capacity probe (GridFeature, post-inventory-fix) confirmed the
head side: raising the RFF dimension lifts wa30 C1 monotonically —
`state_dim` 128/512/1024 -> 34.2%/43.2%/48.1% (fitted scale stable at 5.2;
extra epochs +1.4pt).  The earlier "512 does not help" observation predated
the action-inventory fix and was an artifact of it.  On that evidence the
default `state_dim` is now 1024: offline C1 improved on every game (ls20
42->48/54, tr87 84->87/106) at ~1.6 ms per candidate score, and the
wa30-only live retention rerun at the new width
(`student_retention_20260719T094252.093570Z_7d725c5a025a`) moved wa30 off
zero for the first time — the distilled arm produced a correct
1-step strict route prefix with a correct ordered successor in **all six
episodes at both seeds** (undistilled: zero in all six).  That is
consistency, not competence: a 124-step level-1 route at ~48% per-state
accuracy has effectively zero completion probability, and wa30's states
are visited once each (1,518 unique of 1,556), so the re-entry basin that
let ls20 complete despite imperfect offline fit does not exist.  Closing
wa30 needs either a wider-than-96-value representation (connector width),
or per-step accuracy near the Bayes ceiling, or Sol's sequence-level /
route-value endgame.

The representation-width follow-up (same day) closed most of that gap's
cheap end.  Spatial resolution — not global-latent width — is the axis
that pays: raising `GridFeatureBackend.spatial_shape` from 8x8 to 32x32
improved offline fit on every game (ls20 48->50/54, tr87 87->89/106, wa30
749->917/1556), while full 64x64 resolution is game-dependent (tr87 goes
near-ceiling at 104/106 — beating the Ouro taps' 91 — but ls20 regresses
to 37/54 because unpooled timer pixels dominate distances; wa30 62.0%).
Widening the global latent to 128 *hurt* (wa30 49.7% vs 52.3% at width 32).
The default is now `spatial_shape=(32, 32)`; the RFF basis is memoized at
module level (regeneration dominated wide-representation scoring; caching
is pure, unpersisted, and keeps the bases out of the observe-transaction
deep copy), and `_fit_task_scale` computes distances via the Gram identity
(the broadcasted difference tensor needed ~75 GiB at 1,556 x 4,128).  The
fitted-bandwidth floor of 2.0 was checked at 32x32 and is at the optimum
(fixed 0.5/1.0/2.0 -> 47.6%/58.1%/58.9%).

Retention v5 at the new defaults
(`student_retention_20260719T110034.514671Z_58e74e328f8d`, matched
protocol, integrity clean): **ls20 6/6 level completions — every episode
at both seeds, 79 steps, strict prefix 14, no episode-1 wander and no
degeneration** (v4: 4/6); tr87 uniform 2-step prefixes in all six
distilled episodes; wa30 uniform 1-step prefixes with improved ordered
matches (11 vs 6).  Undistilled arm: zero completions, 3 incidental
prefix steps.  The teacher-train-then-teacher-free retention gate is now
saturated on ls20; wa30's residual is squarely the no-recovery route
structure.  Standing conclusions:
GridFeature remains the runtime default (the 400x encode cost buys +5–7
points of offline C1 only where retention already works); wa30's path is
capacity (wider RFF head, then wider representation width), not a backend
swap; the tr87 separability gain is the first quantified evidence that
frozen loop-state taps carry structure GridFeature lacks, which is the
Lane-B motivation if loop features are ever wanted live.

### Named experiment: checkpoint-carry into teacher-free play (2026-07-19)

`utilities/tests/manual/run_hs_v2_checkpoint_carry_v1.py` tests whether the
mechanisms validated in replay-adjacent settings compose into new
capability.  Phase A: an agent acquires knowledge under compat-assisted
conditions (one live-teacher episode whose completions promote relational
hypotheses and populate the graph/evidence, then offline head
distillation), and is checkpointed.  Phase B: the checkpoint is loaded into
an agent built with `teacher=None` — teacher-free by construction, not
merely by audit — which plays paired seeded episodes against a `fresh`
(no-checkpoint) arm.  Games ls20 (run0 route) and tr87 (merged run0+run1);
wa30 excluded (assisted play never completed it, nothing to carry).
Artifacts:
`artifacts/reports/hunter_seeker_v2/checkpoint_carry_20260719T163903.668719Z_78b7c3681db9`
(default) and `...T165411.523477Z_fd9adbed5efc` (clean-head control).

Phase-A ordering is load-bearing and was corrected during the run:
distilling *before* the assisted episode trains a head that overrides the
live teacher on games it fits imperfectly, collapsing tr87 acquisition from
3 completions to 0.  Assisted-first (untrained head during acquisition,
distill after) restores it: ls20 completes level 1 with 3 verified
hypotheses; tr87 completes 3 levels with 18 verified hypotheses and a
422-example head.

Result: **checkpoint-carry of the retention capability works, but it is
bounded to the walked levels and the carried goal-inference structure is not
load-bearing.**  Teacher-free, the `carried` arm completed 4/6 ls20 episodes
(level 1) versus 0/6 for `fresh` and 0/12 across all wa30-free episodes;
`fresh` never completed anything.  Two hard limits: (1) *no unwalked-level
generalization* — ls20 carried completes exactly the walked level 1 (reaches
stage 2) then dies at the level-2 boundary, never making level-2 progress;
(2) *carried hypotheses do not drive* — tr87 carried, despite 18 verified
relational hypotheses and a head covering three levels of route, completed
0 teacher-free episodes, dying at the 128-step fuse identically to `fresh`.
The relational hypotheses are potential-difference shaping terms
(Ng-Harada-Russell), which nudge but cannot override a mispredicting head,
and tr87's head fidelity (~80%) is below what the fuse tolerates unaided.

Mechanism note: the ls20 `carried` first episode fails at both seeds
(retention v5's head-only agent completed episode 1), and a clean-head
re-distill (`--reset-head-before-distill`) left it failing — so the cause is
not head online-drift but the *carried non-head world-state* (graph /
evidence / exogenous mask) adding score terms that pull off the head's route
on the first teacher-free episode; ep2–3 recover.  Carrying full world-state
is therefore not strictly better than carrying the head alone.

Conclusion for the goal-inference main quest: verified relational
hypotheses are acquired and persisted faithfully but remain **shaping, not
planning** — they do not convert to teacher-free progress on a game whose
head mispredicts.  The identified next step is to make a verified hypothesis
emit a concrete subgoal (the target cell/configuration that minimizes Phi)
as a candidate/plan into beam search, rather than only reshaping scores, and
re-test tr87 carried.

### Goal inference is invariant detection, not goal detection (2026-07-19)

A diagnostic pass on why carried hypotheses never drive teacher-free play
(above) established that the relational hypothesis engine could not
represent goals at all.  Evidence: (1) teacher-free tr87 `candidate_signal`
spread across actions is 0.003-0.029 versus a student term spanning +/-2.7 —
the goal signal is uninformative, not merely outweighed, because the only
action-model is a per-(action, target-signature) delta prior with no support
on fresh states; (2) verified hypotheses on both ls20 (3) and tr87 (18) have
initial potential <= 0.1 (ls20 max 0.080, tr87 max 0.095) — they were
already satisfied at proposal and never drove anything (12/18 tr87 are
spurious `self_mirror_v`); (3) root cause `hypotheses.py`: proposals were
gated `0 < potential <= max_initial_mismatch`, so a relation is only ever
proposed once already near-satisfied — a goal, violated until solved, is
never proposed while it matters and by the time it is achieved is
indistinguishable from a background invariant; (4) contrasting each level's
initial and solved frames, zero structural relations go violated->satisfied
on ls20 (27 candidates) or tr87 (120) — the goals are out of the vocabulary
entirely, which was region-geometric identity only.

### Object-relational goal primitive (`reach`) + goal-from-contrast (2026-07-19)

`hypotheses.py` now adds a `reach` kind: an object-relational potential that
is 0 when the cells of colour A are 8-adjacent to the cells of colour B and
rises with their normalized separation (Ng-Harada-Russell over inter-object
distance; frame-pure and cacheable like the geometric kinds).  A new
`propose_goal_hypotheses` path is the only way goals enter: at a completion
it admits a relation only if it was violated at the level's initial frame
(potential >= `goal_contrast_floor`, default 0.3) and satisfied at the
completion frame, tagging survivors `origin="goal_contrast"` and verified.
The engine caches each level's initial frame and mines goals on progression;
`enable_goal_contrast=False` and `enabled=False` are exact no-ops.  Unit
tests cover the adjacency potential, violated->satisfied detection, and both
no-op guards; full suite 655 pass.

Validation on the trio: the mechanism **detects a genuine reach goal on
tr87** (colours 1->4 go potential 1.0 -> 0.0), proving it works.  On ls20 it
correctly surfaces **nothing**, because ls20's goal is block *consumption*
(the value-11 cell count falls 84 -> 64 -> 28 as blocks are pushed onto
targets and convert), not adjacency between two persistent objects.  So
`reach` is a correct, domain-general capability that captures the large
avatar-reaches-target class, but does not by itself unblock the trio, whose
goals are other types (ls20 consumption/coverage, tr87 a learned legend
mapping).

Load-bearing test on a dedicated venue (`tu93`, an avatar maze; value 4 is a
one-cell avatar, value 14 the stationary target, value 9 a co-moving marker
the goal path correctly excludes because it is never violated).  Undirected
exploration makes no progress — reach-Phi(avatar->target) stays flat at
0.875.  Two findings: (1) *detection* is validated live — the mechanism
identifies avatar 4 -> target 14 as the violated relation and tracks its
potential; (2) *load-bearing* is real but weak.  A greedy controller that
learns an action->displacement model and moves to minimize predicted
distance does **not** beat random on the maze (mean min-Phi 0.65 vs 0.49) —
greedy distance reduction cannot route around walls.  The full agent given
the reach goal steers slightly better than without it via the existing
graph-successor `candidate_signal` (best min-Phi 0.281 vs 0.500 over 8
persistent episodes) but solves neither.  The gap is mechanism, not
representation: decisively load-bearing maze navigation needs the reach
potential used as an A*-style heuristic to direct frontier expansion over
the persistent state graph, plus more exploration budget than tu93's ~50-step
fuse allows in a few episodes.  That graph-directed reach search is the
identified next build; a count/coverage primitive (for ls20) remains the
other open goal-vocabulary follow-up.

### Named experiment: compat-assisted trio under legacy information conditions (2026-07-18)

`utilities/tests/manual/run_hs_v2_assisted_trio_v1.py`: v2 in
``compat_assisted`` mode with `TrajectoryTeacher` over the same
`trusted_topology_trio_20260513` recordings the legacy strict baseline
loaded, epsilon 0, 500-step budget.  Two real defects were found and fixed
to make the comparison possible: `stable_frame_hash` was storage-dtype
sensitive (int8 recordings could never match live int64 frames; identity is
now content-canonical), and the recorder's stage numbering disagrees with
the live environment by one step at level boundaries (an exact-content
fallback index in `TrajectoryTeacher` now covers this).

Result: **v2 clears levels at or above the legacy baseline given the same
information.**  ls20: 1 level (all 54 route steps followed; legacy strict:
1 level).  tr87: 5 levels with merged runs, 3 with run0 alone (legacy
strict: 2).  wa30: 0 (legacy: 0; the recorded route desyncs at step 15 —
open question).  First frames match recordings exactly, confirming
environment determinism.  The completions also exercised the hypothesis
promotion path on real games for the first time: 3 verified goal hypotheses
on ls20 and 26 on tr87.  Assisted results are reported as assisted and are
never mixed with autonomous claims.  Earlier zero-progress results were
strict-autonomous-from-scratch — a condition the legacy agent never
demonstrated either.

### Named experiment: compound goal/memory stack on/off (2026-07-17)

`utilities/tests/manual/run_hs_v2_compound_onoff_v1.py` after the three
compounding follow-ups (hypothesis potentials inside beam rollouts with
exact-graph successor chaining, fuse-aware urgency from masked exogenous
cells, mismatch-keyed click proposals).  Matched 200-step budgets, seeds
{0, 1}, 3 persistent episodes, sc25/lf52/tr87/ls20; artifacts under
`artifacts/reports/hunter_seeker_v2/compound_onoff_20260717/`.

Result: **every follow-up demonstrably functions in live games; no stage
cleared yet.**  The fuse sensor reads exactly 1.0 at the death step of
episode 3 on lf52/tr87 (mask forms at episode 3, so pressure activates
then); graph reuse under chaining+filtering reaches 113–118 revisits on
tr87 versus 32–79 without; mismatch-keyed clicks lead the proposal list on
sc25.  Behavior remains within a few steps of baseline and progress is
zero everywhere.  Current bottleneck hypothesis, from the sc25 renders:
these games demand *composite* interactions (for example selecting a color
from a gauge before painting a cell), which single-click mismatch
targeting cannot express; representing interaction sequences — or a
compat-assisted teacher demonstration — is the next capability layer.

### Named experiment: ego on/off trio (2026-07-17)

`utilities/tests/manual/run_hs_v2_ego_onoff_trio_v1.py`, matched 200-step
budget, seeds {0, 1}, 3 persistent-agent episodes per (game, arm); artifacts
under `artifacts/reports/hunter_seeker_v2/ego_onoff_trio_3ep_20260717/`.

Result: **behaviorally neutral**.  No arm cleared a stage; `tr87` died at
exactly step 128 in all 12 episodes in both arms; `ls20`/`wa30` differed only
marginally.  The sensor itself is demonstrably active (up to 113 tracked
rows, 600 attribution updates) and costs about 0.5 s per 200-step episode.
Interpretation: within-episode, terminal contact attribution arrives too late
to help, and across episodes the trio deaths do not appear to be repeated
ego-contact deaths at these budgets.  Per the evidence rules the component
keeps its observable target and proven causal route, but its default-on
status is justified only as capability-parity restoration with demonstrated
non-regression — flip `EgoConfig.enabled` to exclude it.  Follow-ups that
could show real gain: contact-death-dominated games, longer budgets, more
episodes, and the gate-1 matched legacy comparison.

Results must report autonomous, student, and compatibility-assisted modes
separately.
