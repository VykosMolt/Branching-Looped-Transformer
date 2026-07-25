# Hunter-Seeker v2 — Session Handoff (2026-07-19)

One-line status: **student-head retention is saturated on ls20 (6/6 teacher-free
completions); goal inference was root-caused (it detected invariants, not goals)
and a new object-relational `reach` goal primitive was built and detection-validated;
a `Hunter-Seeker` GitHub repo is staged locally but not pushed (gh auth expired).**

All active development is in `/home/moloch/ouro_project` (this is where the live
ARC-AGI-3 game environments are, under `data/arc_agi3/environment_files/`). A
separate extracted repo lives at `/home/moloch/Hunter-Seeker` — see §1.

---

## 1. Repo, git, and environment state

- **Working tree**: `src/hunter_seeker_v2/` remains **uncommitted** in `ouro_project`
  (it has been untracked the whole project; all of today's edits are in the working
  tree). Tests: full unit+integration suite is **green at 655 passed**, with one
  pre-existing unrelated failure (`utilities/tests/unit/test_causal_correctness.py::
  test_attention_pool_all_zero_mask_returns_zero_not_nan` — an evaluator-track
  AttentionPool NaN test, not a Hunter-Seeker test).
- **Extracted repo** at `/home/moloch/Hunter-Seeker`: built + committed (1 commit,
  181 files, 24 MB, its V2 suite 143 green). Structure: `src/hunter_seeker_v2`,
  `utilities/tests/{unit,integration,manual}`, `docs/hunter_seeker_v2`,
  `data/trajectories/...` (npz+json only), `archived/v1/hunter_seeker_core` (whole
  legacy dir) + `archived/v1/tests`. Large binaries deliberately gitignored (the
  116 MB GridEncoder checkpoint exceeds GitHub's 100 MB limit; the 85 MB `.pt`
  trajectory caches are regenerable).
  - **IMPORTANT — the staged repo is STALE.** It was copied *before* the goal-inference
    work, so it is **missing the `reach` primitive** (hypotheses.py, contracts.py
    HypothesisConfig fields, the reach unit tests) and possibly other late edits.
    Re-sync the changed V2 files from `ouro_project` into `/home/moloch/Hunter-Seeker`
    before pushing.
- **Push is BLOCKED**: `gh` token is invalid (HTTP 401 Bad credentials). User must run
  `! gh auth login -h github.com`. Then push with:
  `gh repo create Hunter-Seeker --private --source=/home/moloch/Hunter-Seeker --remote=origin --push`
  (intended **private**).

---

## 2. What shipped this session (in order)

1. **Diagnosed the zero-autonomous-retention defect**: the student head's RFF state
   map used `state_feature_scale=80`, an RBF bandwidth ~2 orders of magnitude past the
   measured latent geometry, so states were mutually orthogonal (a hash table). Fixed
   the default to 8, then to a **per-task fitted bandwidth** (`_fit_task_scale`).
2. **Fixed three retention blockers** (`student.py`, `teacher.py`): online-erosion
   (neutral outcomes regressed distilled margins → `online_target_weighting` scales
   updates by |target|); tr87 aliasing (per-task bandwidth); wa30 action-inventory
   collapse (`from_npz` now declares the action inventory so distillation's negative
   set covers every action, and NPZ state ids become live-comparable — F1).
3. **Capacity scaling**: default `state_dim` 128→1024 and `GridFeatureBackend.spatial_shape`
   8×8→32×32 (spatial resolution is the paying axis; latent width past 32 hurts).
   Added module-level RFF basis memoization and a Gram-identity distance in the scale fit.
4. **Retention v5**: ls20 **6/6 teacher-free level completions** at both seeds
   (undistilled 0/6). Gate 3 (teacher-train → teacher-disabled retention) is saturated
   on ls20.
5. **Four-arm attribution experiment** (`run_hs_v2_student_attribution_v1.py`): the
   trained student head is the **entire** retention mechanism (head-only reproduces the
   full stack exactly, 4/141/80; head-off 0/0/0).
6. **Gate-4 representation ablation** (`run_hs_v2_student_repr_ablation_v1.py`): frozen
   Ouro-2.6B loop taps vs GridFeature as student input. Ouro wins where geometry was the
   failure (tr87 91 vs 84, ls20 45 vs 42) and ties exactly on wa30 (capacity, not
   features). GridFeature stays the runtime default.
7. **Checkpoint-carry experiment** (`run_hs_v2_checkpoint_carry_v1.py`): knowledge
   acquired under assisted conditions survives save/load into a `teacher=None` agent
   and drives teacher-free completions (carried 4/6 ls20 vs fresh 0/6). Two hard limits:
   no unwalked-level generalization, and carried verified hypotheses do **not** drive
   (tr87 18 verified → 0 completions).
8. **Goal-inference root cause**: the hypothesis engine detected **invariants, not
   goals**. Proposals were gated on the relation being already ~satisfied
   (`hypotheses.py`, the `0 < potential <= max_initial_mismatch` gate), so a goal
   (violated until solved) was never proposed while it mattered. Both games' "verified"
   hypotheses had initial-Φ ≤ 0.1 (already satisfied). Contrasting each level's initial
   vs solved frame, **zero** structural relations went violated→satisfied — the trio's
   goals are out of the region-geometric vocabulary entirely.
9. **Object-relational `reach` goal primitive** (user-chosen direction): a potential
   that is 0 when color-A cells are 8-adjacent to color-B cells, plus a
   `propose_goal_hypotheses` path that admits a relation only if violated at the level's
   initial frame and satisfied at completion (the only way goals can enter). Detection
   validated live (tr87 colors 1→4 Φ 1.0→0.0; tu93 avatar→target correctly identified;
   ls20 correctly nothing — its goal is block *consumption*). Unit-tested; suite 655.
10. **Load-bearing test on tu93** (avatar maze, found via the 25-game survey):
    detection works, but making reach *drive* is weak — greedy distance-reduction can't
    route around maze walls (no better than random), and the full agent with the goal
    steers only slightly via the graph-successor `candidate_signal` (best min-Φ 0.281 vs
    0.500). The gap is mechanism, not representation.

---

## 3. Key findings that steer future work

- **The student head is the whole retention mechanism.** Prior/dynamics/replay/exogenous
  contribute nothing measurable to route retention; iterate on the head. Head-level
  causal claims are now performance-level claims.
- **Retention is bounded by per-game head fidelity and does not generalize to unwalked
  levels.** ls20 completes the walked level then dies at the next boundary. wa30 never
  completes: 1,556 route states with single visits (no re-entry basin), a genuine
  capacity/sequence problem, not features (gate-4 tie).
- **Goal inference was invariant detection, not goal detection.** Now fixed at the
  proposal layer (goal-from-contrast). But the region-geometric + `reach` vocabulary
  still does not cover the trio's actual goals: ls20 = block consumption/coverage,
  tr87 = a learned legend mapping. `reach` covers the avatar-reaches-target class.
- **`reach` is correct and detectable but not yet decisively load-bearing on mazes.**
  It needs to be used as an A*-style heuristic directing frontier expansion over the
  persistent state graph, not a bounded score nudge, plus more exploration budget than
  tu93's ~50-step fuse gives in a few episodes.

---

## 4. Code changes reference (all in `src/hunter_seeker_v2`, uncommitted)

- `student.py`: `state_feature_scale` 80→8; `state_dim` 128→1024; new config
  `fit_state_feature_scale`, `online_target_weighting`; `_fit_task_scale` (Gram-identity
  distances, per-task RBF fit persisted in `task_scales`); module-level `_RFF_BASIS_CACHE`
  memoization; `online_target_weighting` scales committed updates by |outcome target|.
- `models.py`: `GridFeatureBackend.spatial_shape` default 8×8→32×32.
- `teacher.py`: `from_npz(available_actions=...)` explicit-or-inferred inventory; state ids
  now hash actions+progress; successor id uses same identity.
- `contracts.py`: `HypothesisConfig` gained `enable_goal_contrast` (True), `goal_contrast_floor`
  (0.3), `reach_cell_cap` (400), with validation.
- `hypotheses.py`: new `reach` kind + `_reach_potential`; `Hypothesis` gained `value_a`,
  `value_b`, `origin`, `reach_cell_cap`; refactored enumeration into
  `_region_relation_candidates` / `_reach_relation_candidates`; new `propose_goal_hypotheses`
  (violated→satisfied contrast); engine caches per-scope initial frame in `ensure_proposals`
  and mines goals on `progressed` in `observe_transition`; `summary` adds `goals`/`reach_goals`;
  serialization carries the new fields.
- Tests: `utilities/tests/unit/test_hunter_seeker_v2_teacher_index_regressions.py` (new);
  reach tests appended to `test_hunter_seeker_v2_hypotheses.py`; one persistence-test fixture
  pinned to an explicit 8×8 backend so the config check ordering survived the default change.
- New manual runners: `run_hs_v2_student_attribution_v1.py`, `run_hs_v2_student_repr_ablation_v1.py`,
  `run_hs_v2_checkpoint_carry_v1.py` (the ablation runner imports the legacy GridEncoder; in the
  extracted repo it adds `archived/v1` to `sys.path`).

---

## 5. Open threads & recommended next steps (ordered)

1. **Graph-directed `reach` search** (the payoff for the primitive): use reach-Φ as an
   A*-style heuristic to direct beam-search frontier expansion over the persistent state
   graph, and validate on **tu93** with a larger persistent-episode budget. This is the
   step that would turn validated detection into an autonomous maze solve.
2. **Count/coverage goal primitive** for the ls20 goal type (block consumption): the other
   open goal-vocabulary gap.
3. **Sync + push the GitHub repo** (re-copy today's V2 changes into `/home/moloch/Hunter-Seeker`,
   then push once `gh` is re-authed).
4. **Deferred behind retention (now unblocked)**: RLTT Lanes — Lane A loop-convergence
   sensors / three-way Base/Thinking/RLTT gate-4 ablation (the gate-4 tr87 tap-separability
   gain is its standing motivation), Lane B cached tap features, Lane C branch-loop
   imagination. See `hs-v2-rltt-lanes.md`. RLTT objective itself stays parked until the
   student retains routes broadly.
5. **wa30 completion**: a research project (sequence-level / route-value training), parked.
6. **Legend/key-mapping goal primitive** for tr87/sk48/sb26 — hardest, high value, deferred.

---

## 6. Operational gotchas

- **Harness Bash flakiness**: intermittent "claude native binary not installed" errors on
  piped greps/heredocs. Workaround used throughout: write outputs to log files with plain
  Python scripts and `Read` them; avoid `| grep` chains.
- **ARC env seeding**: use `run_hs_v2_student_retention_v1._run_seeded_arc_episode` /
  `_episode_seed` for reproducible per-(seed,episode) environments; the trio's seeded
  environments match the recordings (0-cell first-frame diff).
- **Distill-before-play footgun**: in compat-assisted acquisition, distilling the head
  *before* the assisted episode trains a head that overrides the live teacher on
  imperfect-fit games (tr87 3 levels → 0). Always assisted-episode-first, then distill.
- **Checkpoint config-equality**: a checkpoint saved with a different `state_dim`/backend
  won't load (full config equality check). Old 128-wide / 8×8 checkpoints are now stale.
- **tu93 mechanics** (venue): value 4 = 1-cell avatar, value 14 = stationary target,
  value 9 = co-moving marker (correctly excluded, never violated); pure 1,2,3,4 movement;
  ~50-step fuse. Undirected baseline = flat reach-Φ 0.875.

---

## 7. Artifacts index (`artifacts/reports/hunter_seeker_v2/`)

- `student_retention_20260719T004554...` — v3 (first teacher-free completions, 4/6 ls20)
- `student_retention_20260719T082439...` — v4 (blocker fixes)
- `student_retention_20260719T110034...` — v5 (ls20 6/6, the saturated result)
- `student_retention_20260719T094252...` — wa30-only at state_dim 1024
- `student_attribution_20260719T083710...` — four-arm (head is the whole mechanism)
- `student_repr_ablation_20260719T085434...` — gate-4 (Ouro taps vs GridFeature)
- `checkpoint_carry_20260719T163903...` (default) and `...T165411...` (clean-head control)

Memory: `~/.claude/projects/-home-moloch-ouro-project/memory/hunter-seeker-v2-rebuild.md`
holds the full running history; `hs-v2-rltt-lanes.md` holds the RLTT plan.
