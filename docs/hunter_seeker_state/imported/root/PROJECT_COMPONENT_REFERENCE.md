<!-- Imported from `PROJECT_COMPONENT_REFERENCE.md` during Hunter-Seeker state consolidation on 2026-05-14. -->
<!-- Original SHA256: 6621b0b61e675240a205bc5c3bf87f2a71faf7756784540bee3c644b38d3ec07; original line count: 598. -->

# Ouro Project Component Reference

Last updated: 2026-05-14

This file is the component-level companion to `PROJECT_ARCHITECTURE_MAP.md`.
Use it when you need to know which file owns a behavior, which names are legacy
compatibility names, and where generated artifacts live.

## Naming Boundary

- Domain-neutral names should be preferred in core cognition code:
  `PairwiseSearchAgent`, `LabelPriorTable`, `label`, `label_records`,
  `tracks_by_label`, `DEFAULT_SYMBOLIC_GRID_SIZE`, and `loop_pooler_gate`.
- Compatibility names remain valid where old callers, tests, checkpoints, or
  ARC event dumps still use them:
  `PairwiseARCSearchAgent`, `ColorPriorTable`, `color`,
  `LEGACY_ARC_GRID_SIZE`, and `gru_gate`.
- ARC-specific names are allowed in ARC adapters and the ARC harness:
  `ArcObservationAdapter`, `ArcActionAdapter`, and `train_arc.py`.

## Runtime Entry Points

### `claude_sandbox/train_arc.py`

ARC-AGI-3 training and smoke harness. It builds ARC adapters, creates either
the stockfish baseline or Hunter-Seeker agent, loads checkpoints/trajectories,
runs pretraining, executes games, dumps events, and saves artifacts. It is the
right place for ARC environment wiring, thermal logging, artifact naming, and
game-loop changes.

Important helpers:

- `_enable_live_stdout()`: line-buffer long GPU runs.
- `dump_frame(...)`: save debug frame arrays.
- `_current_frame_from_obs(...)`: adapter-aware frame extraction.
- `_wait_for_thermal_budget(...)`: GPU/CPU thermal guard.
- `_save_run_artifacts(...)`: per-run checkpoint and sequence outputs.
- `_print_step_log(...)`: live step diagnostics.
- `_print_run_summary(...)`: end-of-run diagnostics.
- `_dump_measurement_summary_safe(...)`: structured Hunter-Seeker summaries.
- `_dump_event_log_safe(...)`: event dump writer.
- `load_partial_checkpoint_for_sandbox(...)`: tolerant checkpoint load.
- `train_on_game(...)`: one-game run loop.
- `main()`: CLI construction.

### `claude_sandbox/stockfish/agent.py`

Compatibility constructor and public import surface for the pairwise stockfish
agent. The behavior lives in `claude_sandbox/stockfish/`.

Important public names:

- `PairwiseSearchAgent`: preferred neutral agent name.
- `PairwiseARCSearchAgent`: legacy alias/name kept for old code.
- `DEFAULT_SYMBOLIC_GRID_SIZE`: preferred neutral grid-size alias.
- `LEGACY_ARC_GRID_SIZE`: legacy compatibility alias.
- `DEVICE`, quality constants, saliency helpers, search dataclasses, replay
  buffer, ranker/head classes.

### `claude_sandbox/hunter_seeker/agent.py`

Compatibility constructor and Hunter-Seeker composition shell. It wires all
Hunter-Seeker mixins onto the pairwise agent and owns constructor/state
initialization.

- `HunterSeekerAgent.__init__`

Runtime behavior lives in named mixins under `claude_sandbox/hunter_seeker/`.
New cohesive behavior should move into those modules unless it is strictly
constructor glue.

## Adapter And Encoder Layer

### `claude_sandbox/observation_adapters.py`

Observation boundary between an environment and the generic agent core.

- `ObservationAdapter`: protocol for frame extraction, segmented frames, dense
  encoder input, frame shape, and available actions.
- `ArcObservationAdapter`: ARC observation adapter. ARC-specific by design.
- `MockSymbolicObservation`: simple test observation object.
- `MockSymbolicAdapter`: non-ARC symbolic test adapter.

### `claude_sandbox/action_adapters.py`

Action boundary between generic action ids and environment actions.

- `ActionTypeHead`: action-class logits from a CLS token.
- `ClickHead`: spatial click logits from patch tokens.
- `ActionHead`: combined action and click head.
- `ActionAdapter`: protocol for decoding, bootstrap actions, safe fallback
  actions, and action-head construction.
- `ArcActionAdapter`: ARC action adapter. ARC-specific by design.
- `MockSymbolicActionAdapter`: symbolic test adapter.

### `claude_sandbox/grid_encoder.py`

Domain-neutral symbolic/grid visual encoder. It consumes adapter-sized integer
label grids.

- `PatchEmbedding`: patchifies and embeds dense grids.
- `SinusoidalPositionalEncoding2D`: dynamic 2D positions.
- `LocalPatternExtractor`: local convolutional feature extractor.
- `GridEncoder`: transformer-style grid encoder and Ouro token interface.
- `pad_grids_to_batch(...)`: pad variable grid sizes for batching.

## Stockfish Package

Directory: `claude_sandbox/stockfish/`

### `models.py`

Neural heads and Ouro loop-state poolers.

- `ActionPriorHead`: supervised action prior from encoded state.
- `SpatialClickPredictor`: click-change/spatial action surface.
- `NextFramePredictor`: action-conditioned successor-frame model.
- `TransitionRanker`: pairwise candidate successor ranker.
- `LoopStatePooler`: GRU-style Ouro loop-state reducer.
- `AttnResLoopPooler`: attention/residual Ouro loop-state reducer.

### `encoding.py`

Backbone and frame encoding.

- Loads Ouro/RLTT when requested.
- Builds dense tensors through adapters.
- Runs `encode_and_think_batch(...)`.
- Produces loop diagnostics and exit-gate diagnostics.
- Handles adapter-routed fallback action ids.

### `search.py`

Base candidate generation, scoring, and search.

- Spatial candidate generation.
- World-model successor prediction.
- Ranker symbolic feature construction.
- Search encode cache.
- Transposition table support.
- Beam search and top-level action selection.

### `model_basin.py`

Model-basin diagnostics for base search.

- Bounded root candidate sampling.
- Survivor/collapse/risk summaries.
- Model-basin trace component synchronization.

### `training.py`

Online and supervised training paths.

- Sibling hard-negative insertion.
- Ranker updates.
- Action-prior updates.
- Spatial and next-frame/world-model updates.
- Encoder auxiliary updates when unfrozen.
- CLT anchor diagnostics/training.
- Game-policy pretraining.
- Trusted/online batch blending.
- `train_step(...)`.

### `observation.py`

Observation-learning integration into the stockfish agent.

- Observation transition ingestion.
- Video-frame ingestion.
- Inverse-action promotion for unlabeled observations.
- Observation head training.
- Observation-pretrain bridge into action-conditioned world model, action
  prior, and ranker paths when known/high-confidence action labels exist.
- Candidate observation diagnostics and trust components.

### `replay.py`

Transition storage and sampling.

- `_Transition`: replay row.
- `TransitionReplayBuffer`: online/trusted/auxiliary replay buffer.
- Samplers for ranking pairs, quality gaps, terminal failures, sibling pairs,
  spatial examples, expert actions, and next-frame examples.

### `persistence.py`

Stockfish-level persistence.

- Solved trajectory loading.
- Trusted trajectory CLS cache signatures.
- Base checkpoint save/load for encoder, ranker, prior, spatial model,
  next-frame model, observation heads, and optimizer state.

### `runtime.py`

Base per-game runtime.

- Per-game reset.
- Solved sequence bookkeeping.
- Level-complete relabeling.
- Run-end quality relabeling.
- Base `step(...)`.
- Emits `loop_pooler_gate` and legacy `gru_gate` diagnostics.

### `search_types.py`

Search dataclasses.

- `SearchNode`
- `PendingSearchExpansion`
- `PendingCandidatePrediction`

### `utils.py`

Small search utilities.

- `DEFAULT_GRID_SIZE`
- `visual_saliency(...)`
- `topk_points_2d(...)`
- `normalize_clicks(...)`

## Hunter-Seeker Package

Directory: `claude_sandbox/hunter_seeker/`

### `events.py`

Event taxonomy and bounded event log.

- `EffectType`
- `FailureType`
- `EventType`
- `Event`
- `EventLog`

### `scene.py`

Connected-component scene parser.

- `_ndimage_label`: scipy label function with pure-Python fallback.
- `SceneObject`: parsed component. Exposes `label` alias over legacy `color`.
- `SceneParser`: cached parser and point-object lookup.

### `objects.py`

Object/track belief system and ego/control-set tracking.

- `ObjectRecord`: per-label type belief, interaction history, intrigue terms.
- `TrackRecord`: per-instance object track. Exposes `label` alias over `color`.
- `LabelPriorTable`: game-persistent affordance prior by symbolic label.
- `ColorPriorTable`: compatibility alias for `LabelPriorTable`.
- `ObjectTable`: episode-local tracks plus persistent label priors.
- Ego/control-set tracking: controlled tracks, confidence, ambiguity, action to
  motion signatures.
- Event detection: moved, appeared, disappeared, transformed, contact, level
  completion, death attribution.

### `heads.py`

Hunter-Seeker neural heads and sample types.

- Affordance constants.
- `ObjectActionabilityHead`
- `SymbolicPlannerHead`
- `ObjectivitySample`
- `SymbolicTransitionSample`
- `pool_object_features(...)`

### `objectivity.py`

Objectivity-head integration.

- Buffers affordance samples.
- Derives labels from object events.
- Trains the objectivity head.
- Scores object/click candidates.

### `candidate_generation.py`

Hunter-Seeker candidate proposal.

- `CandidateGenerationMixin`
- `hunter_weight()` and intrigue threshold helpers.
- Click/directional action helper methods.
- Object/probe/phase-aware candidate generation override.

### `scoring.py`

Hunter-Seeker candidate score assembly.

- `CandidateScoringMixin`
- Integrates object belief, topology, symbolic, terminal, engram, safety,
  phase/recovery, observation-effect, and model trust terms.
- Writes score-component diagnostics into candidate traces.

### `action_selection.py`

Hunter-Seeker action selection overrides.

- `ActionSelectionMixin`
- Beam-search wrapping.
- Random-action safety tracing.
- Risk patcher/live recovery selector integration.

### `runtime_lifecycle.py`

Hunter-Seeker runtime lifecycle.

- `RuntimeLifecycleMixin`
- `step(...)`
- `on_level_complete(...)`
- `on_game_over(...)`
- `reset_for_new_game(...)`
- Transition event commitment, terminal attribution, self-model call ordering,
  and per-run reset bookkeeping.

### `topology.py`

Domain-neutral spatial topology.

- `TopologyMixin`
- Free-space topology.
- Reachability/frontier objects.
- Topology cache helpers.
- Candidate topology diagnostics.

### `symbolic.py`

Symbolic transition summaries and action-effect memory.

- `SymbolicTransitionMixin`
- Object/topology summary over predicted successors.
- Symbolic planner-head training.
- Ranker symbolic features.
- Action-effect memory and recovery terms.

### `memory.py`

Engram and coarse episode memory.

- `EngramRecord`
- `EngramMemoryMixin`
- `CoarseEpisodeMemoryMixin`
- Cue construction, similarity, risk/benefit recall, optimism suppression, and
  memory summaries.

### `terminal_memory.py`

Terminal outcome memory.

- `TerminalOutcomeMemoryMixin`
- Terminal action/context keys.
- Terminal outcome prototypes.
- Basin penalties.
- Terminal diagnostics.

### `safety.py`

Local risk and protected-terminal safety.

- `SafetyMixin`
- Local contact hazard scoring.
- Uncertain contact scoring.
- Protected overlap/context memory.
- Positive-pressure suppression.

### `phase_policy.py`

Legacy phase-template and recovery machinery.

- `PhasePolicyMixin`
- Phase-template storage and lookup.
- Recovery/escalation/reseed bookkeeping.
- Passive calibration.
- State-alignment diagnostics.
- Score-component compaction.

Default policy should remain phase-disabled unless the harness explicitly opts
in for ablation.

### `risk_arbitration.py`

Risk patcher and recovery selection.

- `RiskArbitrationMixin`
- Live recovery trace scoring.
- All-risky recovery.
- Candidate diagnostic write-through.
- Post-score replacement of unsafe selected actions.

### `post_veto.py`

Post-veto and online trace diagnostics.

- `PostVetoDiagnosticsMixin`
- Hybrid trace summaries.
- Post-veto candidate diagnostics.

### `self_model.py`

Hunter-Seeker integration for the neural self-model.

- `SelfModelMixin`
- Track summary construction.
- Agent event bundle construction.
- Self-eval summary construction.
- Thought-signature construction.
- Temporal feature recomputation.
- Pending-loss ordering and outcome targets.
- Advance/predict loop for self-model state.

### `measurement.py`

Measurement and event dump output.

- `MeasurementMixin`
- Failure summaries.
- Measurement summaries.
- Event dump serialization.

### `persistence.py`

Hunter-Seeker checkpoint extensions.

- Belief persistence.
- Same-game belief restore.
- Solved trajectory loading override.
- Checkpoint save/load for Hunter-Seeker-specific state.

### `recovery.py`

Small recovery dataclasses.

- `RecoveryContext`
- `RiskArbitrationDecision`
- `RecoveryPolicyParams`

### `smoke_tests.py`

Legacy smoke suite still run by:

```bash
venv/bin/python claude_sandbox/hunter_seeker/agent.py
```

## Observation Learning Core

File: `claude_sandbox/observation_learning.py`

- `ObservationTransition`: before/after observation row with optional action,
  click, outcome, source, and confidence metadata.
- `segment_video_frames(...)`: split videos into frame transitions.
- `to_discrete_frame_tensor(...)`: convert dense/RGB-like frames to symbolic
  tensors.
- `to_discrete_frame_pair_tensors(...)`: paired before/after conversion.
- `state_vector_from_frame(...)`: generic state statistics.
- `build_effect_targets(...)`: generic effect target vector.
- `topology_delta_targets_from_effects(...)`: topology deltas from effect
  targets.
- `build_topology_delta_targets(...)`: topology-delta target construction.
- `effect_vector_from_frames(...)`: before/after effect vector.
- `ObservationReplayBuffer`: serial-aware observation replay with known,
  unlabeled, click, topology, and contrastive samplers.
- `TransitionFrameEncoder`: shared before/after frame encoder.
- `ChangedMaskHead`: changed-cell prediction.
- `EffectSummaryHead`: generic effect summary prediction.
- `TopologyDeltaHead`: structural delta prediction.
- `FramePermanenceEncoder`: contrastive frame permanence encoder.
- `ObservationNextFrameHead`: observation-only frame-to-frame predictor.
- `InverseActionModel`: infer action/click labels from before/after frames.
- `TransitionEffectEngram`: stored effect memory record.
- `TransitionEffectEngramMemory`: effect recall and source diagnostics.
- `RecentExactTransition`: exact recent transition row.
- `RecentExactTransitionWindow`: exact transition window and diagnostics.
- `frame_permanence_contrastive_loss(...)`
- `changed_mask_iou(...)`

## Self-Model Core

File: `claude_sandbox/self_model.py`

- `AffectiveState`: scalar affect vector from recent events.
- `AgentEventBundle`: compact event/state input bundle.
- Empty-vector helpers: track summary, self-eval summary, thought signature,
  event summary, track delta.
- `SelfModelGRU`: recurrent self-state.
- `ContextTokenProjector`: maps self-state into Ouro/context-token space.
- `SelfOutcomePredictor`: predicts terminal boundary, transition progress,
  stagnation, recovery after instability, and topology mismatch.
- `SelfModel`: combined recurrent self-model.
- `compute_loop_signature(...)`: loop-state dynamics signature.
- `CortexMonitor`: optional recurrent monitor over loop signatures.
- `TemporalContextAggregator`: combines self-state, track summaries, event EMA,
  loop deltas, self-eval summaries, thought signatures, and cortex features into
  ranker temporal features.

## Anchor And Evaluator

### `claude_sandbox/anchor_loss.py`

- `FrozenCLTAnchor`: frozen evaluator anchor for preference consistency.
- `synth_loop_states(...)`: synthetic loop-state pairs for anchor smoke tests.

### `claude_sandbox/pairwise_evaluator.py`

- `AttentionPool`: masked attention pooling over hidden states.
- `PairwiseEvaluator`: chosen/rejected sequence preference evaluator.
- `validate_hook_output(...)`: hook-output shape validation.
- `test_pairwise()`: direct smoke.

## Diagnostics And Reports

- `summarize_event_dumps.py`: aggregate event dump roots and trusted-prefix
  agreement summaries.
- `compare_ladder_summaries.py`: compare ladder step summaries and alarms.
- `focus_game_timeline_report.py`: per-game timeline and predictor report.
- `focus_game_hybrid_report.py`: hybrid focus-game report.
- `event_dump_sprint6_ablate.py`: event-dump feature ablations.
- `branch_basin_audit.py`: replay a trace and branch counterfactual suffixes.
- `online_trace_run_report.py`: mock online trace report.
- `sandbox_sweep_validate.py`: mock env validation sweep.
- `live_arc_diagnostic.py`: live ARC diagnostic runner with checkpoint loading.

## Models And Ouro/RLTT

- `models/ouro_rltt_local/`: local converted Ouro/RLTT checkpoint used by
  Hunter-Seeker probes.
- `ouro_rltt/`: local remote-code/model definition mirror.
- `tools/convert_rltt_fsdp_to_hf.py`: conversion helper.
- `tools/run_post_rltt_probe_bundle.py`: post-RLTT probe bundle helper.
- `tools/launch_ouro_ui.sh`: local UI launcher.

## Tests

- `tests/unit/`: fast unit coverage for sandbox components, self-model,
  topology, anchor loss, ranker behavior, and alias compatibility.
- `tests/integration/`: integration tests for Codex/Hunter-Seeker and
  self-model wiring.
- `tests/reports/`: report-script regression tests.
- `tests/manual/`: long/manual probes for evaluator, layer taps, branch
  selection, spatial RLTT, and dataset downloads.

## Artifacts And Data

- `trusted_trajs/`: trusted trajectory inputs.
- `solved_sequences/`: solved action sequences.
- `claude_sandbox/trusted_plus_expanded/`: expanded trusted trajectory set.
- `claude_sandbox/trusted_topology_trio_20260513/`: topology-trio trusted data.
- `claude_sandbox/perf_event_dumps_*`: generated event dump roots from runs;
  historical dumps were purged on 2026-05-14.
- `claude_sandbox/checkpoints_*`: generated run checkpoint roots; only
  `claude_sandbox/checkpoints_encoder_retrain/` is retained by default.
- `claude_sandbox/arc_debug_frames/`: saved debug frames.
- `runs/`: generated local-agent and research probe outputs; historical
  outputs were purged on 2026-05-14.

There is no `local_agent/` source directory under this workspace root. The
local-agent state is documented in `PROJECT_STATE_LOCAL_AGENT.md`; related
outputs live under `runs/local_agent_*`, and local-agent test/probe helpers live
under `tools/test_local_agent_*` and `tools/run_local_agent_self_model_probe.py`.

## Common Change Targets

- Change observation conversion:
  `observation_adapters.py`, `grid_encoder.py`.
- Change action decoding or fallback policy:
  `action_adapters.py`.
- Change base search:
  `stockfish/search.py`; model-basin diagnostics in `stockfish/model_basin.py`.
- Change base training:
  `stockfish/training.py`, `stockfish/replay.py`.
- Change observation pretraining:
  `stockfish/observation.py`, `observation_learning.py`.
- Change Hunter-Seeker object beliefs:
  `hunter_seeker/objects.py`, `hunter_seeker/objectivity.py`.
- Change Hunter-Seeker candidate proposals:
  `hunter_seeker/candidate_generation.py`.
- Change Hunter-Seeker candidate scoring:
  `hunter_seeker/scoring.py`.
- Change Hunter-Seeker action selection:
  `hunter_seeker/action_selection.py`.
- Change Hunter-Seeker step/terminal/reset lifecycle:
  `hunter_seeker/runtime_lifecycle.py`.
- Change topology:
  `hunter_seeker/topology.py`.
- Change symbolic action-effect logic:
  `hunter_seeker/symbolic.py`.
- Change terminal/risk behavior:
  `hunter_seeker/terminal_memory.py`, `hunter_seeker/safety.py`,
  `hunter_seeker/risk_arbitration.py`.
- Change phase-policy ablations:
  `hunter_seeker/phase_policy.py`.
- Change self-model:
  `self_model.py`, `hunter_seeker/self_model.py`.
- Change event output:
  `hunter_seeker/events.py`, `hunter_seeker/measurement.py`,
  `summarize_event_dumps.py`.
