<!-- Source: PROJECT_STATE_HUNTER_SEEKER.md lines 195-1021 before the 2026-05-14 split. -->
<!-- Source chunk SHA256: a0ca7c23e1db03fdafac0490aedac8394ddc3db6ed3d0425a888ed64bb341160 -->

### 0.0.1 Observation Learning Completion Pass - 2026-05-05

This pass completes the first real observation-learning loop and wires it into the self-model without adding domain/game-specific route logic.

Changed files:

- `claude_sandbox/observation_learning_codex.py`
- `claude_sandbox/arc_agent_pairwise_stockfish_codex.py`
- `claude_sandbox/arc_agent_hunter_seeker_codex.py`
- `claude_sandbox/train_arc_codex.py`
- `tests/unit/test_codex_sandbox.py`

Observation buffer state:

- `ObservationReplayBuffer` is now serial-aware in sampled batches.
- Samples now return `serials`, so watched/unlabeled rows can be updated in place.
- New buffer label APIs:
  - `update_action_label(...)`;
  - `promote_action_labels(...)`;
  - `label_summary()`.
- Unlabeled/video rows remain unlabeled until inverse-action confidence is high enough.
- Promotion only changes action-label metadata and reindexes the row. It does not rewrite frames or invent progress/terminal outcomes.

Observation model state:

- Existing heads remain:
  - `ChangedMaskHead`;
  - `EffectSummaryHead`;
  - `InverseActionModel`.
- New structural heads:
  - `TopologyDeltaHead`;
  - `FramePermanenceEncoder`;
  - `ObservationNextFrameHead`.
- New generic target/loss helpers:
  - `OBS_TOPOLOGY_DELTA_KEYS`;
  - `topology_delta_targets_from_effects(...)`;
  - `build_topology_delta_targets(...)`;
  - `frame_permanence_contrastive_loss(...)`.
- The observation loss now includes:
  - changed-mask BCE;
  - observation-only next-frame CE;
  - effect-summary SmoothL1;
  - topology-delta BCE;
  - frame-permanence contrastive loss;
  - inverse-action CE on known/high-confidence labels;
  - click-location CE when click labels exist;
  - inverse-label confidence loss.
- Historical note: at this point this was still observation-only learning. It did not update the ranker/action-prior unless normal `train_step()` policy losses were also active. This caveat is superseded by `0.0.3` below: observation pretraining now has policy-visible action-world-model, action-prior, and ranker updates when high-confidence known action labels are present.

Inverse-action promotion state:

- New agent method: `_infer_unlabeled_observation_actions(...)`.
- The confidence used for promotion is:
  - inverse action softmax max probability;
  - multiplied by the inverse model's label-confidence head.
- Default threshold: `0.90`.
- Promoted sources are tagged with `inverse_inferred`.
- Inferred-action effect engrams are conservative:
  - source-visible;
  - confidence is bounded to `<= 0.35`;
  - outcome is only `change` or `noop`;
  - no terminal/progress claim is fabricated from the inverse label alone.

Engram state:

- `TransitionEffectEngramMemory` now exposes source diagnostics via `source_counts()`.
- Recall diagnostics now include:
  - `obs_engram_inferred_action_support`;
  - `obs_engram_inferred_action_match_count`.
- Risk-aware aggregation remains unchanged in spirit:
  - positive evidence does not erase negative terminal/hazard support;
  - cross-action evidence remains diagnostic unless extremely high confidence;
  - no broad action/color/game blacklist was introduced.

Candidate diagnostics state:

- Observation candidate rows now expose topology/event predictions:
  - `obs_topology_delta_max_prob`;
  - `obs_topology_delta_best_label`;
  - `obs_topology_delta_terminal_prob`;
  - `obs_topology_delta_progress_prob`;
  - `obs_topology_delta_moved_value_prob`;
  - `obs_topology_delta_noop_prob`.
- These are score-visible diagnostics. They feed self-model pressure and bounded memory/recovery context, not hard overrides.

Observation pretraining state:

- New agent method: `pretrain_observation_learning(...)`.
- New harness flags:
  - `--observation_pretrain_iters`;
  - `--observation_pretrain_infer_every`;
  - `--observation_video_npz`;
  - `--observation_video_key`.
- `--observation_video_npz` accepts `.npy`/`.npz` frame arrays shaped like `[T,H,W]` or `[T,H,W,C]`.
- This is a generic watched-frame hook. Non-ARC domains should decode their own footage into arrays and use the same agent observation API.
- Historical note: this was true for the first observation-learning completion pass. It is superseded by `0.0.3`: observation pretraining now also trains the action-conditioned world model and positive known-action policy path, while remaining separate from the harmful same-game `--game_policy_pretrain_iters 300` configuration.

Self-model wiring state:

- Existing self-eval slots now carry richer observation-learning state:
  - observation loss pressure includes changed/effect/topology/permanence/inverse losses;
  - observation activity includes buffer size, known/unlabeled balance, update count, and inverse-action promotions;
  - expected-outcome error still comes from predicted-vs-real successor diagnostics;
  - terminal pressure includes observation engram, recent-exact memory, and topology-delta terminal probability;
  - memory progress includes observation engram, recent-exact progress, and topology-delta progress probability;
  - candidate uncertainty rises when watched/unlabeled transitions remain low-confidence or conflict flags are present.
- Thought signature now blends changed-mask quality with inverse-action confidence/accuracy, so the self-model receives a continuity signal for "how well my observation machinery is understanding transitions."

Checkpoint state:

- New observation heads are saved/loaded:
  - `topology_delta_head`;
  - `frame_permanence_encoder`;
  - `observation_nextframe_head`.
- New inverse-action promotion counters are saved/loaded:
  - `observation_inferred_action_attempts`;
  - `observation_inferred_action_promotions`;
  - `observation_inferred_action_confidence_ema`.
- Old checkpoints remain load-tolerant. Missing new observation heads start fresh.

Verification:

- `py_compile` passed for:
  - `claude_sandbox/observation_learning_codex.py`;
  - `claude_sandbox/arc_agent_pairwise_stockfish_codex.py`;
  - `claude_sandbox/arc_agent_hunter_seeker_codex.py`;
  - `claude_sandbox/train_arc_codex.py`;
  - `tests/unit/test_codex_sandbox.py`.
- Focused observation/memory tests passed: `21 passed, 175 deselected`.
- Full unit file passed: `196 passed`.

Next empirical step:

- Run RLTT with trusted trajectories loaded, policy pretrain disabled, and observation-only pretraining enabled before returning to `wa30`.
- Suggested first diagnostic shape:

```bash
/home/moloch/ouro_project/venv/bin/python -m claude_sandbox.train_arc_codex --agent hunter_seeker --games ls20 tr87 wa30 --backbone_mode ouro --ouro_model_path models/ouro_rltt_local --checkpoint checkpoints_running/sprint4_encoder_reverted.pt --load_trajs claude_sandbox/trusted_plus_expanded --pretrain_iters 0 --observation_pretrain_iters 100 --observation_pretrain_infer_every 5 --game_policy_pretrain_iters 0 --batch_size 64 --train_every 0 --max_steps 180 --n_runs 1 --eps 0.0 --no_replay --running_checkpoint "" --checkpoint_dir claude_sandbox/checkpoints_observation_pretrain_trio --save_trajs_dir claude_sandbox/solved_sequences_expanded --dump_events_dir claude_sandbox/perf_event_dumps_observation_pretrain_trio --thermal_guard --thermal_check_every 5 --step_timing_every 10 --slow_step_seconds 15
```

Interpretation target:

- If exact-support behavior recovers while observation diagnostics improve, keep observation pretraining as the source-of-info path and avoid game-policy pretrain as a default.
- If `wa30` still collapses near phase 65, inspect whether observation topology/event predictions and inferred-action support see the divergence before step 249.
- If observation losses improve but behavior does not, the next patch should connect observation recall to candidate generation/action-effect recovery, still bounded and score-visible.

### 0.0.2 Topology Trio Regression - 2026-05-05

Latest RLTT topology trio command:

```bash
/home/moloch/ouro_project/venv/bin/python -m claude_sandbox.train_arc_codex --agent hunter_seeker --games ls20 tr87 wa30 --backbone_mode ouro --ouro_model_path models/ouro_rltt_local --checkpoint checkpoints_running/sprint4_encoder_reverted.pt --load_trajs claude_sandbox/trusted_plus_expanded --pretrain_iters 0 --game_policy_pretrain_iters 300 --batch_size 128 --lr 0.001 --train_every 0 --max_steps 180 --n_runs 1 --eps 0.0 --no_replay --running_checkpoint "" --checkpoint_dir claude_sandbox/checkpoints_topology_trio_exact_support --save_trajs_dir claude_sandbox/solved_sequences_expanded --dump_events_dir claude_sandbox/perf_event_dumps_topology_trio_exact_support --thermal_guard --thermal_check_every 5 --step_timing_every 10 --slow_step_seconds 15
```

Result:

- `ls20`: cleared level 1 only, then mechanism death at step `85`; score `3.5714285714285716`.
- `tr87`: cleared `0` levels, mechanism death at step `128`; score `0.0`.
- `wa30`: cleared `0` levels by step `180`; score `0.0`.
- GPU was genuinely active during live `wa30` (`~100%` utilization, about `6.9GB` VRAM, `51-54C`).

Interpretation:

- Runtime solved-prefix replay is still disabled under `--no_replay`.
- Exact-state trusted action support is live and score-visible. It recovered `ls20` level 1 without runtime replay, so the patch is real behavior rather than a prefix follower.
- The 300-iteration same-game pretrain configuration did not recover the old topology trio target (`ls20` 3, `tr87` 4, `wa30` 1). It likely disturbs ranking/prior behavior enough that the next comparison should run with `--game_policy_pretrain_iters 0`.
- `tr87` only had exact support on the opening state and selected an unobserved action despite a bounded exact-state penalty; after that it was off-demo.
- `wa30` had exact support for the first several states but diverged early and did not rediscover the old level-0 completion.

Patch after this run:

- `trusted_state_action_exact_match` now survives Hunter-Seeker score-component compaction in lean/candidate/full traces.
- Verification: `py_compile` passed for the modified core files; focused unit tests passed with `4 passed, 190 deselected`.

Next empirical check:

- Run the same topology trio with RLTT, exact-state support enabled, runtime replay disabled, and `--game_policy_pretrain_iters 0`.
- If that recovers more of the old behavior, keep game pretraining diagnostic/optional and avoid treating it as a default topology fix.
- If that still fails, inspect the first divergence against trusted exact states, especially `tr87` step 1 and `wa30` steps 1-10, before adding any new behavior patch.

### 0.0.3 Observation-Policy Completion + Topology Trio Matrix - 2026-05-13

This entry supersedes the 2026-05-05 observation caveat. Observation pretraining is no longer merely behavior-visible through observation heads; when known/high-confidence action labels are available, it now updates the action-conditioned successor model and the policy-side prior/ranker path.

Code state:

- `ObservationNextFrameHead` is now action/click-conditioned.
- `pretrain_observation_learning(...)` now trains:
  - observation heads;
  - action-conditioned world model from known observed actions;
  - action prior on positive known-action transitions;
  - transition ranker on positive-vs-contrast known-action transitions;
  - loop-pooler path when enabled.
- New pretrain diagnostics:
  - `observation_pretrain_action_world_model_updates`;
  - `observation_pretrain_action_world_model_loss`;
  - `observation_pretrain_policy_updates`;
  - `observation_pretrain_policy_prior_loss`;
  - `observation_pretrain_policy_ranker_loss`;
  - `observation_pretrain_policy_prior_updates`;
  - `observation_pretrain_policy_ranker_updates`.
- Bounded recovery/ES parameter surface was added with neutral defaults. The full ES outer loop was not implemented because `es_integration_plan.md` says the first step is visible bounded gates and diagnostics, not an active evolutionary training loop.
- `arc-agi==0.9.6` and `arcengine==0.9.3` were installed into the project venv.

Verification:

- `py_compile` passed for:
  - `claude_sandbox/observation_learning_codex.py`;
  - `claude_sandbox/arc_agent_pairwise_stockfish_codex.py`;
  - `claude_sandbox/arc_agent_hunter_seeker_codex.py`;
  - `claude_sandbox/train_arc_codex.py`.
- Targeted observation tests passed:
  - `test_pairwise_observation_learning_trains_and_scores_diagnostics`;
  - `test_pairwise_observation_inverse_promotes_video_transitions`;
  - `test_observation_effect_trust_drops_after_bad_successor_calibration`.
- GPU-visible launch shape for these runs used `timeout ... env ... venv/bin/python`; CUDA preflight reported `NVIDIA GeForce RTX 5070 Ti Laptop GPU`.
- No GPU compute processes remained after the final run.

Trusted data / command conditions:

- Topology-only trusted slice:
  - `claude_sandbox/trusted_topology_trio_20260513`;
  - copied only `ls20`, `tr87`, and `wa30` trajectories from the expanded trusted pool.
- Shared serious-run settings:
  - RLTT model: `models/ouro_rltt_local`;
  - checkpoint: `checkpoints_running/sprint4_encoder_reverted.pt`;
  - `--pretrain_iters 0`;
  - `--game_policy_pretrain_iters 0`;
  - `--batch_size 32`;
  - `--train_every 10`;
  - `--post_level_reinforce_iters 2`;
  - `--max_steps 500`;
  - `--n_runs 1`;
  - `--eps 0.0`;
  - `--no_replay`;
  - `--enable_agent_phase_policy`.

Important replay/phase distinction:

- `--no_replay` disables raw solved-prefix replay, not all training data.
- Trusted-buffer training and sibling-pair auxiliary data remain active by design.
- The historical topology trio scores were not strict topology-only. They depended on replay-derived phase/action templates being active.
- Therefore there are three different measurements:
  1. strict/default phase-disabled topology-only policy;
  2. phase/action policy active but live training off;
  3. phase/action policy active with live training and optional observation pretrain.

Strict topology-only / phase policy disabled:

- Event root:
  - `claude_sandbox/perf_event_dumps_topology_trio_500x2_topology_only_20260513`.
- Config:
  - `--train_every 0`;
  - no `--enable_agent_phase_policy`;
  - `--no_replay`;
  - `--max_steps 500`;
  - intended `n_runs=2`, stopped after the issue was clear.
- Observed behavior:
  - `ls20` run 1: level 1 at step `13`, mechanism death at step `91`;
  - `ls20` run 2: measurement reported `0` levels, mechanism death;
  - `tr87` run 1: `2` levels, mechanism death;
  - `tr87` run 2: `2` levels, mechanism death;
  - `wa30` partial: `0` levels, protected-terminal-starvation death in the measurement.
- Interpretation:
  - strict topology-only is not the historical topology-trio behavior;
  - disabling phase/action templates exposes that the live default action policy still lacks route/action-sequence competence;
  - this is not evidence that topology cannot see the board.

Phase/action policy active, no live training:

- Event root:
  - `claude_sandbox/perf_event_dumps_topology_trio_500x2_phase_policy_20260513`.
- Config:
  - `--enable_agent_phase_policy`;
  - `--train_every 0`;
  - `--no_replay`;
  - `--max_steps 500`.
- Completed `ls20` run:
  - levels at run-relative steps `13,136,175,267,321,429`;
  - `6` levels by the `500` step cap;
  - score `72.6235484742101`;
  - no ranker/prior/next-frame training updates.
- Duplicate run was stopped after matching the same route.
- Interpretation:
  - phase/action templates alone recover and exceed the earlier `ls20` smoke behavior;
  - this is still not raw solved-prefix replay, but it is replay-derived phase/action policy scaffolding;
  - this proves the old topology-trio performance was being carried by phase/action route competence, not by pure topology.

Phase/action policy active + live training, no observation pretrain:

- Event roots:
  - `claude_sandbox/perf_event_dumps_topology_trio_500x2_phase_policy_train_b32_20260513` for `ls20`;
  - `claude_sandbox/perf_event_dumps_topology_trio_1each_phase_policy_train_b32_20260513` for `tr87` and `wa30`.
- Results, run-relative:

| game | levels | level steps | terminal | observation updates | last obs loss | inverse action acc |
|---|---:|---|---|---:|---:|---:|
| `ls20` | 6 | `13,136,175,267,321,429` | none before cap | 62 | 0.3506 | 0.9688 |
| `tr87` | 4 | `37,67,106,135` | death `263`, `topology` | 34 | 0.4251 | 0.9688 |
| `wa30` | 2 | `125,183` | death `283`, `mechanism` | 66 | 0.5261 | 0.9063 |

- Losses were active during play:
  - ranker loss;
  - action-prior loss;
  - next-frame loss;
  - observation loss;
  - objectivity loss.
- Interpretation:
  - with all active components processing and learning, raw solved-prefix replay is not required for the topology trio smoke;
  - `ls20` and `tr87` recover the historical good routes;
  - `wa30` remains at the known two-level ceiling and dies in the same late mechanism/nonviable-basin region;
  - the remaining issue is route/action-effect competence after the phase/action scaffold runs out, not a basic replay-training failure.

Phase/action policy active + live training + observation pretrain:

- A 100-iteration observation-pretrain attempt was stopped before episode execution because the new policy/ranker pretrain path is expensive: each iteration performs Ouro encodes for policy-side current/next states. It was GPU-bound, not OOM or hung.
- Completed shorter comparison:
  - event root: `claude_sandbox/perf_event_dumps_topology_trio_1each_phase_policy_train_obs20_b32_20260513`;
  - checkpoints: `claude_sandbox/checkpoints_topology_trio_1each_phase_policy_train_obs20_b32_20260513`;
  - `--observation_pretrain_iters 20`;
  - `--observation_pretrain_infer_every 5`;
  - otherwise same live-training settings.
- Observation-pretrain diagnostics:
  - `iters=20`;
  - `updates=20`;
  - `action_wm_updates=20`;
  - `policy_updates=20`;
  - `policy_prior_loss=1.7181`;
  - `policy_ranker_loss=0.7215`;
  - `promotions=0`;
  - `known=2591`;
  - `unlabeled=0`;
  - `engram=2591`.
- Results, run-relative:

| game | levels | level steps | terminal | observation updates | last obs loss | inverse action acc |
|---|---:|---|---|---:|---:|---:|
| `ls20` | 6 | `13,136,175,267,321,429` | none before cap | 82 | 0.4411 | 0.8750 |
| `tr87` | 4 | `37,67,106,135` | death `263`, `self_model` | 116 | 0.4193 | 0.9063 |
| `wa30` | 2 | `125,183` | death `283`, `mechanism` | 148 | 0.3058 | 0.9688 |

Critical interpretation:

- Observation pretrain is now finished in the narrow sense that it is policy-visible:
  - it trains observation heads;
  - it trains the action-conditioned successor model;
  - it trains action prior/ranker on known positive observation transitions.
- However, `obs20` did not change the actual topology-trio route:
  - `ls20` exact same six level steps;
  - `tr87` exact same four level steps and death step;
  - `wa30` exact same two level steps and death step.
- The only `tr87` behavior difference was failure classification changing from `topology` to `self_model`; the route and step boundary did not improve.
- `wa30` observation metrics improved strongly by the end of the run, but behavior did not. That means the observation learner is processing and learning, but its learned signal is not yet generating a better candidate/route out of the level-3 mechanism basin.
- The 100-iteration pretrain cost is now too high for routine smoke use unless the policy-side Ouro encodes are cached, batched more efficiently, or sampled less frequently.

What the runs told us:

1. Replay layer 1 is not the main training problem in these measurements.
   - With `--no_replay`, phase/action policy plus live learning reproduces the known routes.
   - The harmful result was the old same-game `--game_policy_pretrain_iters 300`, not trusted buffers or sibling-pair training.
2. The topology trio name is misleading if it implies pure topology.
   - Historical good behavior used phase/action route templates.
   - Strict phase-disabled topology-only policy is weak and collapses into repeated directional action patterns.
3. Observation learning is necessary but not sufficient.
   - The new obs-policy pretrain path is active and measurable.
   - In this trio it improves/changes internal diagnostics more than behavior.
   - The missing piece is using observed action effects to produce or prefer better live candidate continuations.
4. `ls20` is healthy under the actual active-component regime.
   - `6` levels by `500` steps with or without obs20 pretrain.
   - This is stronger than the old "3 levels" smoke expectation and matches the known extended route through step `429`.
5. `tr87` is healthy up to the known four-level route, then enters a slow recovery/failure window.
   - The post-level-4 slow region remains expensive.
   - Obs20 changes diagnostic classification but not the outcome.
6. `wa30` remains the real ceiling.
   - Current ceiling remains `2` levels, death at run-relative step `283`, mechanism failure.
   - This is not stale-avatar topology.
   - The problem is still phase/action-effect competence around the late nonviable/mechanism basin.

Next work from this evidence:

1. Do not re-enable raw solved-prefix replay as the explanation for training health. Keep `--no_replay` serious runs, but keep trusted-buffer and sibling-pair training active.
2. Keep phase/action policy as an ablation/bridge, but treat it as scaffolding rather than final cognition.
3. Optimize observation-policy pretrain:
   - cache current/next Ouro encodes for known-action observation rows;
   - avoid doing two full Ouro passes for every policy pretrain iteration when the trusted row set is static;
   - then retry `obs100` if needed.
4. Connect observation/action-effect learning to candidate generation and recovery:
   - state/effect-conditioned branch diversification;
   - contradiction handling when predicted/observed effect mismatches;
   - learned positive action-effect proposals;
   - bounded novelty/diversity terms already exposed through the neutral recovery parameter surface.
5. Continue to avoid:
   - game-id logic;
   - color/label blacklists;
   - broad action blacklists;
   - hand-coded `wa30` routes.

### 0.0.4 Phase Policy As Teacher, Not Permanent Policy - 2026-05-14

Goal for the next session:

- Make the agent learn from the phase/action policy the way a toddler learns from guided action:
  - the guide can show what to do;
  - the child observes action/effect/outcome structure;
  - the guide is gradually removed;
  - success means autonomous action without the scaffold.

Current interpretation:

- Phase/action policy should not be treated as final cognition.
- It should become a teacher/scaffold that produces high-confidence demonstrations and candidate preferences.
- The student should be the ordinary live agent:
  - action prior;
  - transition ranker;
  - action-conditioned world model;
  - observation/engram/action-effect memory;
  - recovery candidate generation.
- Final evaluation must run with phase policy disabled, raw solved-prefix replay disabled, and live learning active.

Compatibility with observation learning:

- This is compatible with the current observation-learning state.
- The current observation pretrain path already supports known-action transitions and now updates:
  - observation heads;
  - action-conditioned successor model;
  - action prior;
  - transition ranker;
  - loop-pooler path when enabled.
- A phase-policy teacher transition can be represented as:
  - pre-state;
  - teacher action/click;
  - next-state;
  - outcome/effect summary;
  - teacher confidence/margin;
  - candidate alternatives when available.
- This is conceptually observation learning, not replay, if the phase policy is used only to generate/label experience and is absent at autonomous evaluation.

Required next implementation direction:

1. Add a phase-teacher collection/distillation path.
   - Let phase policy select or label actions during a teacher pass.
   - Store those transitions as high-confidence known-action observation rows.
   - Store teacher-chosen candidates and rejected alternatives for pairwise ranker training.
2. Add scaffold fading.
   - Start with teacher-forced action labels.
   - Add teacher dropout/student takeover.
   - Add DAgger-style correction data: when the student reaches a state, ask the teacher what it would have done, then train on the discrepancy.
3. Train student components, not phase machinery.
   - Prior learns plausible action in the current state.
   - Ranker learns chosen transition vs alternatives.
   - Action-conditioned world model learns action effects.
   - Observation/engram memory learns support, contradiction, and effect signatures.
4. Evaluate autonomy explicitly.
   - Teacher/phase policy disabled.
   - Raw solved-prefix replay disabled.
   - Trusted buffers and sibling-pair training may remain active as training-data curation.

Metrics to add/track:

- `phase_teacher_student_action_agreement`;
- `phase_teacher_rank_margin`;
- `phase_teacher_student_rank_gap`;
- `phase_teacher_distill_prior_loss`;
- `phase_teacher_distill_ranker_loss`;
- `phase_teacher_distill_world_model_loss`;
- `route_retention_without_phase_policy`;
- `recovery_after_student_divergence`;
- `effect_prediction_error_on_teacher_transitions`.

Success criterion:

- Train with phase policy available as a teacher/scaffold.
- Then run strict phase-disabled topology trio with:
  - `--no_replay`;
  - live training active;
  - `--game_policy_pretrain_iters 0`;
  - phase policy disabled.
- The agent should retain the learned route/action-effect competence instead of collapsing into repeated directional actions.

Guardrails:

- Do not expose game id, level id, or phase id to the final student policy at inference.
- Do not hard-code `wa30`, `ls20`, or `tr87` routes.
- Do not add color/label/action blacklists.
- Keep all teacher-derived effects source-visible and auditable.
- Treat phase policy as a temporary developmental scaffold, not an architectural dependency.

Implementation started 2026-05-14:

- Added `hunter_seeker/phase_teacher.py`.
- Added `pretrain_phase_teacher_policy(...)`, exposed by
  `train_arc.py --phase_teacher_pretrain_iters`.
- Added `phase_teacher_label_for_state(...)` as the explicit future
  DAgger/student-takeover hook: exact label when the live pre-state signature
  matches, abstention otherwise.
- Exact same-game solved action templates and click/target templates now become
  teacher examples only when their before-state signature exactly matches an
  expert trusted transition with the same action.
- Mismatched, weak, missing-signature, or untrusted templates are counted as
  teacher abstentions instead of producing hard labels.
- The distillation path trains ordinary student surfaces:
  - `ActionPriorHead`;
  - `TransitionRanker` via teacher positive vs rejected action/click
    alternatives;
  - spatial click target head for teacher click targets;
  - action-conditioned world model unless `policy_train_only` is active;
  - observation heads from recorded phase-teacher before/after rows.
- Runtime phase policy remains opt-in through `enable_agent_phase_policy`; the
  new teacher path does not select live actions.

Regression diagnosis and scope guard - 2026-05-14:

- The first teacher-training topology trio showed behavior-visible learning but
  regressed `tr87` from the previous strict smoke's `2` clears to `1` clear.
- The regression did not reproduce in a fresh single-game `tr87`
  teacher+observation diagnostic:
  - artifact root: `artifacts/reports/diagnostic_tr87_teacher_obs_20260514/`;
  - `tr87` cleared `2` levels by step `67`.
- It did reproduce in a two-game `ls20 -> tr87` phase-teacher-only ablation:
  - artifact root: `artifacts/reports/diagnostic_ls20_tr87_phase_teacher_only_20260514/`;
  - no initial observation pretrain was used;
  - `tr87` diverged after level 1 and cleared only `1` level.
- Root cause:
  - observation-effect terminal/negative engram memory was task-agnostic;
  - an `ls20` terminal memory matched the trusted `tr87` post-level-1
    continuation by effect/state similarity;
  - the false terminal recall injected negative support and caused
    all-risky/risk-patcher selection to override the trusted continuation.
- Fix:
  - `TransitionEffectEngram` now stores a generic scope;
  - terminal/hazard recalls are ignored when stored scope and current scope do
    not match;
  - progress/change recalls remain cross-scope so positive effect transfer is
    still possible;
  - online terminal/model-basin labels write scoped sources
    (`terminal_observed:<scope>`, `model_basin_sampler:<scope>`).
- Guardrail:
  - scope is a memory-isolation key for negative safety evidence, not a policy
    input and not a route rule;
  - it should prevent cross-task death poisoning without exposing game identity
    to the action prior/ranker as a behavioral feature.
- Verification:
  - full test suite before the patch: `619 passed, 1 skipped, 18 subtests passed`;
  - scoped engram and phase-teacher tests after the patch: `5 passed`;
  - post-fix behavior smoke was interrupted before completion and still needs a
    rerun.

Encoder-before-ES ordering decision:

- Train new encoder weights before applying active evolutionary-algorithm behavior.
- Rationale:
  - representation learning and ES policy/search adaptation should not be confounded in the same experiment;
  - if both change at once, regressions or gains cannot be attributed cleanly;
  - active ES could adapt around unstable encoder geometry and select behavior that exploits transient representation quirks rather than durable action-effect competence.
- Safe before encoder training:
  - neutral ES/recovery parameter surface;
  - diagnostics;
  - config plumbing;
  - off-by-default ES scaffolding;
  - score-component visibility.
- Should wait until after the encoder is trained and checkpointed:
  - active ES outer loop;
  - mutating recovery/search weights;
  - population selection over behavior policies;
  - ES-driven candidate steering.
- Required encoder-training discipline:
  - keep the behavior stack stable while training the encoder;
  - use CLT/Ouro anchoring so the encoder does not drift out of Ouro-compatible geometry;
  - train on observation/action-effect/teacher data where useful;
  - after training, freeze/checkpoint the encoder and run loop-signature/ranker/topology-trio baselines before activating ES.

### 0.0.5 Hunter-Seeker Refactor Start - 2026-05-14

Decision:

- Begin a total readability/navigation refactor before further algorithmic work.
- First phase is behavior-preserving extraction only:
  - move cohesive definitions out of `arc_agent_hunter_seeker_codex.py`;
  - re-export old public names from the legacy module;
  - keep tests, scripts, and imports compatible;
  - do not change policy behavior while moving code.

Extraction completed so far:

- New package:
  - `claude_sandbox/hunter_seeker/`.
- New map:
  - `claude_sandbox/hunter_seeker/README.md`.
- Extracted modules:
  - `hunter_seeker/events.py`
    - `EffectType`;
    - `FailureType`;
    - `EventType`;
    - `Event`;
    - `EventLog`.
  - `hunter_seeker/scene.py`
    - `_ndimage_label` with scipy fallback;
    - `SceneObject`;
    - `SceneParser`.
  - `hunter_seeker/heads.py`
    - affordance constants;
    - `ObjectActionabilityHead`;
    - `SymbolicPlannerHead`;
    - `ObjectivitySample`;
    - `SymbolicTransitionSample`;
    - `pool_object_features`.
  - `hunter_seeker/objects.py`
    - `_OBJECT_TYPES`, `_N_TYPES`, `_EVIDENCE`, `_TYPE_CLICK_BONUS`;
    - `ObjectRecord`;
    - `TrackRecord`;
    - `ColorPriorTable`;
    - `ObjectTable`;
    - ego/control-set tracking.
  - `hunter_seeker/recovery.py`
    - `RecoveryContext`;
    - `RiskArbitrationDecision`;
    - `RecoveryPolicyParams`.
  - `hunter_seeker/memory.py`
    - `EngramRecord`.
  - `hunter_seeker/smoke_tests.py`
    - legacy in-file smoke tests moved out of the runtime agent file;
    - `python claude_sandbox/arc_agent_hunter_seeker_codex.py` still runs them through a thin runner.

Compatibility status:

- `claude_sandbox.arc_agent_hunter_seeker_codex` still exports the extracted names.
- Manual checks confirmed:
  - EventLog terminal-protection behavior;
  - scene parser separates touching different-color components;
  - head output shapes;
  - object belief/evidence update behavior;
  - object table scene ingestion and track creation;
  - recovery/memory dataclass behavior;
  - legacy import surface for `HunterSeekerAgent`, `EventLog`, `SceneParser`, `ObjectActionabilityHead`, `ObjectTable`, `ObjectRecord`, `TrackRecord`, `ColorPriorTable`, `EngramRecord`, and `RecoveryPolicyParams`.
- `py_compile` passed for:
  - `claude_sandbox/hunter_seeker/__init__.py`;
  - `claude_sandbox/hunter_seeker/events.py`;
  - `claude_sandbox/hunter_seeker/scene.py`;
  - `claude_sandbox/hunter_seeker/heads.py`;
  - `claude_sandbox/hunter_seeker/objects.py`;
  - `claude_sandbox/hunter_seeker/recovery.py`;
  - `claude_sandbox/hunter_seeker/memory.py`;
  - `claude_sandbox/hunter_seeker/smoke_tests.py`;
  - `claude_sandbox/arc_agent_hunter_seeker_codex.py`;
  - `claude_sandbox/arc_agent_pairwise_stockfish_codex.py`;
  - `claude_sandbox/observation_learning_codex.py`;
  - `claude_sandbox/train_arc_codex.py`.
- Historical note for this refactor slice: `pytest` was not installed in the
  venv yet, so focused pytest slices were not run at that point. This is
  superseded by the 2026-05-14 pytest install and verification below.

Current size:

- Before extraction:
  - `arc_agent_hunter_seeker_codex.py`: `26441` lines.
- After current extraction:
  - `arc_agent_hunter_seeker_codex.py`: `22123` lines.
  - `hunter_seeker/events.py`: `168` lines.
  - `hunter_seeker/scene.py`: `215` lines.
  - `hunter_seeker/heads.py`: `150` lines.
  - `hunter_seeker/objects.py`: `2182` lines.
  - `hunter_seeker/recovery.py`: `121` lines.
  - `hunter_seeker/memory.py`: `27` lines.
  - `hunter_seeker/smoke_tests.py`: `1401` lines.

Next extraction targets:

1. `topology.py`
   - free-space topology;
   - reachability;
   - local-contact risk helpers;
   - topology cache signatures.
2. `memory.py`
   - terminal outcome memory;
   - protected-terminal memory;
   - engram recall;
   - coarse episode summaries.
3. `recovery.py`
   - all-risky recovery;
   - live recovery;
   - risk patcher;
   - selector logic.
4. `phase_teacher.py`
   - future phase-policy-as-teacher collection/distillation;
   - must remain separate from default live policy.

Refactor invariants:

- Behavior-preserving moves first.
- No game-id, level-id, phase-id exposure to final student policy.
- Keep legacy imports working until downstream code is migrated.
- Keep all behavior-affecting terms visible in diagnostics.
- Run compile/import checks after every slice.

### 0.0.6 Hunter-Seeker Refactor Sweep - 2026-05-14

Status:

- Continued the behavior-preserving extraction from the legacy agent file.
- `claude_sandbox.arc_agent_hunter_seeker_codex` remains the compatibility entry point.
- `HunterSeekerAgent` is now a thin orchestration class composed from extracted mixins.
- No intentional policy change was made in this sweep.

Additional extracted modules:

- `hunter_seeker/topology.py`
  - free-space topology;
  - reachability scoring;
  - topology cache signatures.
- `hunter_seeker/symbolic.py`
  - symbolic transition summaries;
  - symbolic planner-head buffer/training;
  - short-horizon action-effect memory.
- `hunter_seeker/safety.py`
  - local contact safety;
  - protected-terminal context;
  - terminal-risk positive-pressure suppression.
- `hunter_seeker/terminal_memory.py`
  - exact terminal context keys;
  - terminal outcome prototypes;
  - cross-action terminal basin penalties;
  - terminal-memory diagnostics.
- `hunter_seeker/objectivity.py`
  - CNN feature pooling for object affordance heads;
  - click/directional affordance sample buffering;
  - affordance label derivation from event logs;
  - objectivity-head training and scoring.
- `hunter_seeker/phase_policy.py`
  - phase target/action helper logic;
  - recovery/escalation/reseed bookkeeping;
  - phase state alignment/trust diagnostics;
  - default policy remains gated by `enable_agent_phase_policy`.
- `hunter_seeker/risk_arbitration.py`
  - risk patcher;
  - all-risky recovery;
  - live/observation recovery trace selectors;
  - risk arbitration decision records.
- `hunter_seeker/post_veto.py`
  - online hybrid trace summaries;
  - post-veto candidate generation diagnostics.
- `hunter_seeker/self_model.py`
  - neural self-model feature construction;
  - pending self-model loss consumption;
  - advance/predict ordering.
- `hunter_seeker/measurement.py`
  - failure summaries;
  - measurement summaries;
  - event dumps.
- `hunter_seeker/persistence.py`
  - belief persistence;
  - solved trajectory loading;
  - checkpoint save/load.

Current size after this sweep:

- `arc_agent_hunter_seeker_codex.py`: `5285` lines.
- Extracted package:
  - `events.py`: `168`;
  - `scene.py`: `215`;
  - `heads.py`: `150`;
  - `objects.py`: `2182`;
  - `memory.py`: `1389`;
  - `topology.py`: `503`;
  - `symbolic.py`: `802`;
  - `safety.py`: `1487`;
  - `terminal_memory.py`: `1351`;
  - `objectivity.py`: `665`;
  - `phase_policy.py`: `5943`;
  - `risk_arbitration.py`: `1872`;
  - `post_veto.py`: `599`;
  - `self_model.py`: `1207`;
  - `measurement.py`: `582`;
  - `persistence.py`: `757`;
  - `recovery.py`: `121`;
  - `smoke_tests.py`: `1401`.

Verification:

- `py_compile` passed for:
  - all files under `claude_sandbox/hunter_seeker/*.py`;
  - `claude_sandbox/arc_agent_hunter_seeker_codex.py`;
  - `claude_sandbox/arc_agent_pairwise_stockfish_codex.py`;
  - `claude_sandbox/observation_learning_codex.py`;
  - `claude_sandbox/train_arc_codex.py`.
- Focused import checks passed for the new mixins:
  - topology;
  - symbolic;
  - safety;
  - engram/coarse memory;
  - phase policy;
  - terminal memory;
  - objectivity;
  - risk arbitration;
  - post-veto diagnostics;
  - self-model;
  - measurement;
  - persistence.
- Historical note for this slice: `pytest` was still not installed in the venv,
  so no pytest slice was run at that point. This is superseded by the
  2026-05-14 pytest install and verification below.

Local-agent self-model comparison:

- The local wrapper self-model is at `/home/moloch/local_agent/ouro_self_model.py`.
- Related wrapper modules:
  - `/home/moloch/local_agent/ouro_self_predictor.py`;
  - `/home/moloch/local_agent/ouro_vector_memory.py`;
  - `/home/moloch/local_agent/ouro_self_context.py`;
  - `/home/moloch/local_agent/ouro_self_store.py`.
- Useful ideas to port into the neural self-model:
  - bounded outcome-prediction/calibration buckets;
  - explicit decision-surface diagnostics;
  - loop-disagreement calibration tied to later outcomes;
  - vector/semantic recall as a weak memory channel.
- Do not port directly:
  - prompt/context construction;
  - hard-code route policies;
  - local-agent task-specific route priors;
  - text-contamination sanitizers, except as a persistence hygiene pattern.
- Recommended neural form:
  - add a small auxiliary outcome/calibration head on top of `SelfModel.h_t` and existing self-eval/thought features;
  - train it from real structural transition outcomes: terminal boundary, progress, stagnation, recovery after instability, and predicted/observed topology mismatch;
  - expose its prediction in diagnostics first;
  - only later allow a weak bounded score term after ablations show calibration.

Implemented continuation on 2026-05-14:

- `claude_sandbox/self_model.py` now owns a zero-init `SelfOutcomePredictor` inside `SelfModel`, so checkpoint persistence remains under `self_model.state_dict()`.
- The auxiliary labels are domain-neutral:
  - `terminal_boundary`;
  - `transition_progress`;
  - `stagnation`;
  - `recovery_after_instability`;
  - `topology_mismatch`.
- `HunterSeekerAgent.step()` now consumes pending self-model losses after the real transition has been decoded into event/outcome targets, but still before `super().step()` can run ranker training and mutate shared self-model parameters.
- The outcome head is diagnostic/training-only. It does not feed candidate scoring or phase/action policy.
- `measurement_summary()["self_model"]` now reports outcome prediction, target, loss EMA, and update count.
- Older checkpoints that predate `outcome_predictor` load compatible self-model weights with the new head initialized fresh.

Stockfish refactor note:

- Stockfish has now been split into `claude_sandbox/stockfish/` while
  preserving `arc_agent_pairwise_stockfish_codex.py` as the compatibility
  constructor/wiring entry point.
- Completed stockfish extractions on 2026-05-14:
  - new package `claude_sandbox/stockfish/`;
  - neural heads and loop poolers moved to `claude_sandbox/stockfish/models.py`;
  - Ouro/backbone loading, loop diagnostics, dense-frame conversion, and
    adapter fallback actions moved to `claude_sandbox/stockfish/encoding.py`;
  - observation-learning and observation-pretrain helpers moved to
    `claude_sandbox/stockfish/observation.py`;
  - checkpoint and solved-trajectory persistence moved to
    `claude_sandbox/stockfish/persistence.py`;
  - search dataclasses moved to `claude_sandbox/stockfish/search_types.py`;
  - replay storage/samplers moved to `claude_sandbox/stockfish/replay.py`;
  - reset/step/run lifecycle and solved-prefix bookkeeping moved to
    `claude_sandbox/stockfish/runtime.py`;
  - candidate generation, scoring, search caches, and beam selection moved to
    `claude_sandbox/stockfish/search.py`;
  - model-basin diagnostics later moved one step further to
    `claude_sandbox/stockfish/model_basin.py`;
  - ranker/action-prior/world-model/anchor training moved to
    `claude_sandbox/stockfish/training.py`;
  - saliency/top-k/click-normalization helpers moved to `claude_sandbox/stockfish/utils.py`;
  - `arc_agent_pairwise_stockfish_codex.py` is down to about `0.7k` lines
    from `10.3k`.
- Audit fixes from the same pass:
  - restored legacy re-exports from `arc_agent_pairwise_stockfish_codex.py`
    (`SearchNode`, pending-search dataclasses, `LEGACY_ARC_GRID_SIZE`,
    quality constants, saliency/top-k helpers), because Hunter-Seeker/tests
    still import them from the compatibility module;
  - added explicit imports for `DEVICE`, `ObjectRecord`, and `TrackRecord` in
    `hunter_seeker/persistence.py`, which previously relied on monolith globals;
  - updated the in-file smoke expectation for non-click level completion: it
    must stay unattributed unless an exit was already localized.
- Verification:
  - `py_compile` passed for the refactored stockfish package, Hunter-Seeker
    compatibility module, Hunter-Seeker mixins, and self-model;
  - import smoke passed for pairwise/Hunter-Seeker compatibility exports;
  - `timeout 60 venv/bin/python claude_sandbox/arc_agent_hunter_seeker_codex.py`
    passed all legacy smoke tests;
  - historical note: `pytest` was not installed in the venv at that point, so
    pytest-based tests could not run. This is superseded by the 2026-05-14
    pytest install and verification below.
- Current navigation map:
  - see `PROJECT_ARCHITECTURE_MAP.md`.
