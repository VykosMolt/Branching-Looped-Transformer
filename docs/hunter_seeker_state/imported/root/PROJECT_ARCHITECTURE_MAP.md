<!-- Imported from `PROJECT_ARCHITECTURE_MAP.md` during Hunter-Seeker state consolidation on 2026-05-14. -->
<!-- Original SHA256: cf006800c00b842333ae2391bc256a0c5c545de852de2218aff040c9111411bc; original line count: 575. -->

# Ouro Project Architecture Map

Last updated: 2026-05-14

This is the navigation document for the active Hunter-Seeker/Ouro code path.
It is intentionally practical: where the code lives, how data moves, what owns
which responsibility, what was verified, and where the current risks are.

## Active Entry Points

- `claude_sandbox/train_arc.py`
  - Main ARC harness.
  - Builds adapters, constructs either `stockfish` or `hunter_seeker`, loads
    checkpoints/trajectories, handles pretraining, runs games, saves outputs.
  - ARC-specific environment wiring belongs here.

- `claude_sandbox/stockfish/agent.py`
  - Compatibility constructor for the pairwise search/ranker agent.
  - Now mostly module wiring and neural component construction.
  - Re-exports old public names used by tests/Hunter-Seeker:
    `PairwiseARCSearchAgent`, `SearchNode`, pending search dataclasses,
    `TransitionReplayBuffer`, `TransitionRanker`, `LEGACY_ARC_GRID_SIZE`,
    quality constants, saliency helpers.
  - Also exposes domain-neutral aliases:
    `PairwiseSearchAgent` and `DEFAULT_SYMBOLIC_GRID_SIZE`.

- `claude_sandbox/hunter_seeker/agent.py`
  - Compatibility constructor and Hunter-Seeker composition shell.
  - Owns constructor/state initialization plus the mixin stack; runtime
    behavior is now split across named modules in `hunter_seeker/`.

- `PROJECT_STATE_HUNTER_SEEKER.md`
  - Historical/current project state.
  - Use this for empirical context and what previous runs meant.

- `claude_sandbox/hunter_seeker/README.md`
  - Refactor map for the Hunter-Seeker package.

## Runtime Data Flow

1. Harness creates domain adapters.
   - ARC harness uses `ArcObservationAdapter` and `ArcActionAdapter`.
   - Core agents read adapter metadata (`n_values`, `n_actions`,
     `click_action_idx`) rather than hardcoding action/color dimensions.

2. `agent.step(obs)` extracts the current frame/actions through adapters.

3. The frame is encoded.
   - Dense symbolic/video conversion lives in `stockfish/encoding.py`.
   - `GridEncoder` tokenizes frames.
   - If `backbone_mode="ouro"`, Ouro/RLTT loop states are read and optionally
     pooled by `LoopStatePooler` or `AttnResLoopPooler`.
   - If `backbone_mode="encoder_only"`, the encoder CLS is used directly.

4. Candidate generation proposes actions/clicks.
   - Base stockfish candidate search lives in `stockfish/search.py`.
   - Hunter-Seeker adds object/probe/phase-aware candidates in
     `hunter_seeker/candidate_generation.py`.

5. Candidate scoring combines:
   - learned transition ranker;
   - learned action prior;
   - spatial predictor;
   - world-model successor confidence;
   - observation-effect memory;
   - model-basin diagnostics;
   - Hunter-Seeker object belief, topology, terminal memory, engrams,
     objectivity head, symbolic/action-effect memory, phase/recovery terms, and
     risk arbitration. Hunter-Seeker score assembly lives in
     `hunter_seeker/scoring.py`; base model-basin diagnostics live in
     `stockfish/model_basin.py`.

6. The selected action is executed by the harness/environment.

7. On the next observation, the previous transition is committed to buffers:
   - online replay buffer;
   - observation buffer;
   - object/event memories;
   - terminal/progress bookkeeping.

8. Training runs according to harness knobs.
   - `train_every=0` disables online training.
   - `policy_train_only=True` skips world-model/observation auxiliary updates.
   - observation pretraining can update observation heads and, when known
     actions exist, policy-visible action/world-model/ranker/action-prior paths.

## Stockfish Package

Directory: `claude_sandbox/stockfish/`

- `models.py`
  - `ActionPriorHead`
  - `SpatialClickPredictor`
  - `NextFramePredictor`
  - `TransitionRanker`
  - `LoopStatePooler`
  - `AttnResLoopPooler`

- `encoding.py`
  - Ouro/RLTT model loading.
  - Loop diagnostics and exit-gate diagnostics.
  - `encode_and_think_batch`.
  - Dense frame conversion helpers.
  - Adapter-routed fallback action ids.

- `search.py`
  - Spatial prediction and candidate generation.
  - World-model successor prediction.
  - Ranker symbolic feature construction.
  - Candidate scoring.
  - Search encode cache and transposition table support.
  - Beam search and top-level `select_action`.

- `model_basin.py`
  - Model-basin diagnostic sampling.
  - Root candidate survival/collapse summaries.
  - Trace synchronization for model-basin fields.

- `training.py`
  - Sibling hard-negative insertion.
  - Ranker training.
  - Action-prior training.
  - World-model/encoder auxiliary training.
  - CLT anchor diagnostics/training.
  - Per-game policy pretraining.
  - Trusted/online blended batch helpers.
  - `train_step`.

- `observation.py`
  - Observation transition ingestion.
  - Video-frame ingestion.
  - Inverse-action promotion for unlabeled observations.
  - Observation heads training.
  - Observation-pretrain policy/world-model bridge.
  - Candidate observation diagnostics and trust components.

- `replay.py`
  - `_Transition`.
  - `TransitionReplayBuffer`.
  - Trusted/online/auxiliary sampling logic.
  - Ranking-pair, quality-gap, terminal-failure, sibling-pair, spatial, expert,
    and next-frame samplers.

- `persistence.py`
  - Solved trajectory loading.
  - Trusted trajectory CLS cache signature checks.
  - Checkpoint save/load for stockfish-level modules.

- `runtime.py`
  - Per-game reset.
  - Solved sequence bookkeeping.
  - Level-complete and run-end quality relabeling.
  - Base `step`.

- `search_types.py`
  - `SearchNode`
  - `PendingSearchExpansion`
  - `PendingCandidatePrediction`

- `utils.py`
  - `DEFAULT_GRID_SIZE`
  - `visual_saliency`
  - `topk_points_2d`
  - `normalize_clicks`

## Hunter-Seeker Package

Directory: `claude_sandbox/hunter_seeker/`

- `events.py`
  - Event/failure/effect enums and `EventLog`.

- `scene.py`
  - Connected-component scene parser with scipy fallback.
  - `SceneObject`.

- `objects.py`
  - `ObjectRecord`, `TrackRecord`, `LabelPriorTable`, `ObjectTable`.
  - `ColorPriorTable` remains as a compatibility alias.
  - Object/track/scene records expose `label` aliases while keeping old
    `color` storage for event dumps and checkpoint compatibility.
  - Ego/control-set tracking.
  - Track matching, object belief updates, terminal evidence, exit evidence.

- `heads.py`
  - Object affordance head.
  - Symbolic planner head.
  - Objectivity and symbolic sample dataclasses.

- `candidate_generation.py`
  - Hunter-mode weight and intrigue threshold helpers.
  - Directional/click action helper methods.
  - Hunter-Seeker candidate generation override.

- `scoring.py`
  - Hunter-Seeker candidate score assembly.
  - Terminal/prototype/basin penalty integration into candidate traces.
  - Object belief, symbolic, phase, safety, engram, and observation-effect
    score components.

- `action_selection.py`
  - Hunter-Seeker `beam_search_action` and `select_action` overrides.
  - Risk patcher/live recovery selector integration around base beam search.

- `runtime_lifecycle.py`
  - Hunter-Seeker `step`.
  - `on_level_complete`, `on_game_over`, and `reset_for_new_game`.
  - Runtime event commitment, terminal callback attribution, and reset
    bookkeeping.

- `objectivity.py`
  - Affordance sample buffering.
  - Label derivation from events.
  - Objectivity-head training/scoring.

- `topology.py`
  - Free-space topology, reachability, frontier objects, topology caches.
  - This is domain-neutral spatial structure, not ARC route logic.

- `symbolic.py`
  - Symbolic transition summaries.
  - Action-effect memory.
  - Symbolic planner-head training and ranker symbolic features.

- `memory.py`
  - Engram memory.
  - Coarse episode memory.
  - Similarity/risk/benefit recall primitives.

- `terminal_memory.py`
  - Terminal action/context keys.
  - Terminal outcome prototypes and basin penalties.
  - Terminal diagnostics.

- `safety.py`
  - Contact/protected-terminal risk helpers.
  - Local safety scoring.

- `phase_policy.py`
  - Phase-template bookkeeping.
  - Legacy phase-guidance/recovery/escalation/reseed logic.
  - Passive calibration and score-component compaction.
  - This is the largest remaining extracted module and the next readability
    target.

- `risk_arbitration.py`
  - Post-score risk patching.
  - All-risky recovery.
  - Live/observation recovery selectors.
  - Candidate diagnostic write-through.

- `post_veto.py`
  - Post-veto diagnostics.
  - Online trace summaries.

- `self_model.py`
  - Hunter-Seeker integration for `claude_sandbox/self_model.py`.
  - Affective/event bundle construction.
  - Self-model pending-loss ordering.
  - Outcome target construction and temporal-context recomputation.

- `measurement.py`
  - Measurement summary.
  - Failure summaries.
  - Event dump writing.

- `persistence.py`
  - Hunter-Seeker checkpoint extensions.
  - Belief persistence.
  - Solved trajectory loading override.
  - Handles compatibility for old checkpoints.

- `recovery.py`
  - `RecoveryContext`
  - `RiskArbitrationDecision`
  - `RecoveryPolicyParams`

- `smoke_tests.py`
  - Legacy in-file smoke suite used by
    `python claude_sandbox/hunter_seeker/agent.py`.

## Self-Model Core

File: `claude_sandbox/self_model.py`

- `AffectiveState`
  - Scalar affect vector from recent agent events.

- `AgentEventBundle`
  - Compact event/state bundle used to build self-model inputs.

- `SelfModel`
  - GRU hidden state.
  - Context-token projector.
  - Zero-init outcome predictor.

- `SelfOutcomePredictor`
  - Predicts:
    - `terminal_boundary`
    - `transition_progress`
    - `stagnation`
    - `recovery_after_instability`
    - `topology_mismatch`

- `CortexMonitor`
  - Optional GRU over Ouro loop-signature dynamics.

- `TemporalContextAggregator`
  - Combines self-model hidden state, object/track summaries, event EMA,
    loop-delta, self-eval summaries, thought signatures, and cortex features
    into ranker temporal features.

## Training And Pretraining Modes

- Online training:
  - Controlled by `--train_every`.
  - Runs `train_step` from `stockfish/training.py`.
  - Updates ranker/action prior, and unless `policy_train_only`, spatial,
    next-frame/world model, encoder when unfrozen, and observation heads.

- Trusted trajectory loading:
  - `--load_trajs`.
  - Populates trusted replay.
  - Can run pretrain iterations.
  - Cached CLS can avoid repeated Ouro encodes when signatures match.

- Game policy pretrain:
  - `--game_policy_pretrain_iters`.
  - Same-game supervised adaptation before each game.
  - Trains exact-state action contrast, ranker pairs, action prior, and
    optionally world model.
  - It is not runtime prefix replay.

- Observation pretrain:
  - `--observation_pretrain_iters`.
  - Trains observation heads.
  - Can promote high-confidence inverse-action labels.
  - With known actions, also updates action-conditioned world model, action
    prior, and ranker path.

- Anchor training:
  - `--anchor_train_every` for live anchor updates.
  - `--pretrain_anchor_every` for pretrain anchor updates.
  - Requires Ouro and an unfrozen encoder.
  - `--anchor_pair_smoke` is diagnostic only.

- Replay/prefix policy:
  - `--enable_solved_prefix_policy` opts into solved-prefix runtime action
    following.
  - `--no_replay` disables Hunter-Seeker replay follower behavior.
  - Default intended path is learning from trajectories, not following them as
    an autopilot.

## Diagnostics And Outputs

- `measurement_summary()`
  - Main structured agent diagnostics.
  - Includes object/topology/terminal/self-model/anchor/observation summaries.

- Event dumps:
  - `--dump_events_dir`.
  - Written by `hunter_seeker/measurement.py`.

- Candidate traces:
  - Search traces come from `stockfish/search.py`.
  - Hunter-Seeker compaction/diagnostic enrichment lives mostly in
    `hunter_seeker/phase_policy.py`, `risk_arbitration.py`, and
    `post_veto.py`.

- Checkpoints:
  - Stockfish base checkpointing in `stockfish/persistence.py`.
  - Hunter-Seeker extensions in `hunter_seeker/persistence.py`.

- Run outputs:
  - `runs/` when regenerated by manual probes.
  - `claude_sandbox/perf_event_dumps_*` when regenerated by training/eval runs.
  - `claude_sandbox/checkpoints_*` only for explicitly requested new runs.
  - `solved_sequences/`
  - `trusted_trajs/`
  - Historical run/checkpoint outputs were purged on 2026-05-14; old state
    references to those paths are provenance only.

## Verification From This Refactor/Audit Pass

Passed:

- `venv/bin/python -m py_compile` across the refactored stockfish package,
  Hunter-Seeker compatibility module, Hunter-Seeker mixins, and self-model.
- Import smoke for:
  - `HunterSeekerAgent`
  - `PairwiseARCSearchAgent`
  - compatibility exports from `stockfish/agent.py`
  - `claude_sandbox.stockfish.__all__`
- Missing-global bytecode probe across stockfish and Hunter-Seeker mixins.
- Legacy smoke suite:
  - `timeout 60 venv/bin/python claude_sandbox/hunter_seeker/agent.py`

Additional verification after pytest install:

- `venv/bin/python -m pytest tests/unit/test_domain_neutral_aliases.py tests/unit/test_topology_sprint5.py tests/unit/test_codex_sandbox.py -q`
  passed (`206 passed`).
- `venv/bin/python -m pytest tests/unit -q --ignore=tests/unit/test_local_agent_wrapper.py`
  passed (`426 passed`) after the mixin extraction.
- `timeout 60 venv/bin/python claude_sandbox/hunter_seeker/agent.py`
  passed.
- `timeout 60 venv/bin/python -m claude_sandbox.hunter_seeker.agent`
  passed.
- `timeout 60 venv/bin/python claude_sandbox/arc_agent_hunter_seeker_codex.py`
  passed.

## Audit Findings Fixed In This Pass

1. Compatibility re-export regression.
   - Problem: `hunter_seeker/agent.py` and tests still import legacy
     names from `stockfish/agent.py`.
   - Fix: restored re-exports for search dataclasses, quality constants,
     `LEGACY_ARC_GRID_SIZE`, and saliency/top-k helpers.

2. Split-module missing globals.
   - Problem: `hunter_seeker/persistence.py` referenced `DEVICE`,
     `ObjectRecord`, and `TrackRecord` as if it still lived in the monolith.
   - Fix: added explicit imports.

3. Stale smoke expectation.
   - Problem: the smoke suite expected non-click level completion to invent exit
     evidence.
   - Fix: updated it to match the current conservative rule: non-click
     completion stays unattributed unless an exit was already localized.

4. Dead imports in extracted stockfish modules.
   - Removed unused imports from `stockfish/search.py`,
     `stockfish/observation.py`, and `hunter_seeker/persistence.py`.

5. Large-file extraction.
   - Hunter-Seeker runtime/search overrides moved out of `hunter_seeker/agent.py`
     into `candidate_generation.py`, `scoring.py`, `action_selection.py`, and
     `runtime_lifecycle.py`.
   - Stockfish model-basin diagnostics moved from `stockfish/search.py` into
     `stockfish/model_basin.py`.

## Remaining Risks And Grounded Improvements

1. Centralize constants/device.
   - `_QUALITY_LEVEL_SCALE`, `_QUALITY_CHANGE_BONUS`, and `DEVICE` are currently
     duplicated across compatibility/mixin modules.
   - Best next step: `stockfish/constants.py` with compatibility re-exports from
     `stockfish/agent.py`.

2. Continue shrinking the large extracted behavior modules.
   - `hunter_seeker/agent.py` is now constructor/composition-sized at about
     `672` lines.
   - Next readability targets are the behavior-heavy extracted modules:
     - `hunter_seeker/runtime_lifecycle.py` (`step`, terminal callbacks, reset);
     - `hunter_seeker/scoring.py` (large score-component assembly);
     - `hunter_seeker/phase_policy.py` (legacy phase/recovery machinery).

3. Split `phase_policy.py`.
   - It is about 5.9k lines and mixes:
     - phase-template storage;
     - progress/recovery/escalation/reseed;
     - state-alignment diagnostics;
     - candidate score compaction;
     - trusted/future option logic.
   - Suggested target modules:
     - `phase_templates.py`
     - `phase_recovery.py`
     - `phase_alignment.py`
     - `phase_compaction.py`

4. Replay sampler performance.
   - `stockfish/replay.py` has improved pool caches, but several samplers still
     perform repeated Python loops and `np.stack` over frame arrays.
   - If training becomes the bottleneck, maintain indexed pools by:
     - game id;
     - level;
     - terminal/nonterminal;
     - changed/nonchanged;
     - quality bucket.

5. Training CPU/GPU transfer pressure.
   - Ranker training repeatedly calls `.cpu().numpy()` to compute symbolic
     features and to re-encode frames.
   - Grounded improvement: cache symbolic feature tensors alongside replay
     transitions when frame pairs are inserted, invalidating only when the
     symbolic feature schema changes.

6. Search compute pressure.
   - `stockfish/search.py` already batches frontier encodes and has a search
     encode cache.
   - Remaining high-cost areas are model-basin diagnostics and observation
     candidate diagnostics.
   - Keep diagnostics cadence-controlled; do not make them unconditional in
     topology-trio runs.

7. Constructor logging is noisy.
   - Agent construction prints full parameter summaries every instantiation.
   - This is useful in long runs but noisy in tests/smokes.
   - Add a `verbose_init` or harness-level logging flag later.

8. Broad `except Exception` remains common.
   - Some are intentional compatibility guards.
   - Critical paths should prefer counted diagnostics over silent pass:
     persistence migration, terminal-memory scoring, phase recovery, and
     observation promotion.

9. Compatibility names are partly ARC-flavored.
   - New neutral names exist for the main public boundary:
     `PairwiseSearchAgent`, `DEFAULT_SYMBOLIC_GRID_SIZE`, `LabelPriorTable`,
     `label_records`, `tracks_by_label`, and `loop_pooler_gate`.
   - `PairwiseARCSearchAgent`, `LEGACY_ARC_GRID_SIZE`, `ColorPriorTable`, and
     JSON/checkpoint `color` fields remain for existing callers and old data.

10. Broader tests still need selective reruns after large refactors.
    - Pytest is now installed in the venv.
    - The smoke suite catches many regressions but is not a substitute for the
      full unit/integration suite.

## Where To Make Common Changes

- Change action/frame adapters:
  - `claude_sandbox/observation_adapters.py`
  - `claude_sandbox/action_adapters.py`
  - Harness construction in `train_arc.py`

- Change base candidate generation:
  - `stockfish/search.py`

- Change Hunter-Seeker candidate additions:
  - `hunter_seeker/candidate_generation.py`
  - `hunter_seeker/objectivity.py`
  - `hunter_seeker/phase_policy.py`

- Change candidate score terms:
  - Base learned scores: `stockfish/search.py`
  - Observation-effect terms: `stockfish/observation.py`
  - Object/topology/symbolic/terminal terms:
    `hunter_seeker/scoring.py`, `hunter_seeker/topology.py`,
    `hunter_seeker/symbolic.py`, `hunter_seeker/terminal_memory.py`
  - Risk overrides: `hunter_seeker/risk_arbitration.py`
  - Model-basin diagnostics: `stockfish/model_basin.py`

- Change replay/training:
  - Replay storage/sampling: `stockfish/replay.py`
  - Train step and supervised pretraining: `stockfish/training.py`
  - Observation pretraining: `stockfish/observation.py`

- Change checkpoint format:
  - Base stockfish: `stockfish/persistence.py`
  - Hunter-Seeker additions: `hunter_seeker/persistence.py`

- Change self-model behavior:
  - Core neural definitions: `self_model.py`
  - Agent integration: `hunter_seeker/self_model.py`
  - Runtime call ordering: `hunter_seeker/runtime_lifecycle.py`

- Change event/measurement output:
  - Event definitions: `hunter_seeker/events.py`
  - Measurement summary/event dump: `hunter_seeker/measurement.py`
  - Trace compaction: `hunter_seeker/phase_policy.py`

## Refactor Invariants

- Keep behavior-preserving extractions separate from algorithmic changes.
- Preserve compatibility imports from legacy entry points unless all callers are
  migrated in the same patch.
- Keep ARC-specific harness/environment logic in `train_arc.py` or ARC
  adapters, not in neural/core cognition modules.
- Keep topology as domain-neutral spatial structure.
- Keep replay/prefix following opt-in; default policy should learn from stored
  trajectories rather than execute them directly.
- Run at least:
  - `py_compile`;
  - compatibility import smoke;
  - missing-global probe after each large extraction;
  - legacy smoke suite when Hunter-Seeker behavior surfaces are touched.
