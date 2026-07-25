<!-- Imported from `claude_sandbox/hunter_seeker/README.md` during Hunter-Seeker state consolidation on 2026-05-14. -->
<!-- Original SHA256: 5006f4e3ce3d2c3f8b36d5d4bf6215bdc469bce205379ec1ab40a30873b7ef38; original line count: 151. -->

# Hunter-Seeker Refactor Map

The compatibility entry point remains:

`claude_sandbox.hunter_seeker.agent`

This package holds extracted implementation slices.  Refactor order is
behavior-preserving first: move cohesive code out of the legacy file, re-export
the old public names, and keep policy behavior unchanged until tests and smokes
say the new structure is stable.

Current modules:

- `events.py`
  - `EffectType`
  - `FailureType`
  - `EventType`
  - `Event`
  - `EventLog`

- `scene.py`
  - `_ndimage_label` with scipy fallback
  - `SceneObject`
  - `SceneParser`

- `heads.py`
  - affordance constants
  - `ObjectActionabilityHead`
  - `SymbolicPlannerHead`
  - `ObjectivitySample`
  - `SymbolicTransitionSample`
  - `pool_object_features`

- `objects.py`
  - object records, track records, label priors, object table, ego/control-set
    tracking
  - `LabelPriorTable` is the neutral name; `ColorPriorTable` remains as a
    compatibility alias for old callers/checkpoints.
  - records expose `label` aliases while preserving old `color` storage fields.

- `recovery.py`
  - `RecoveryContext`
  - `RiskArbitrationDecision`
  - `RecoveryPolicyParams`

- `memory.py`
  - `EngramRecord`
  - `EngramMemoryMixin`
  - `CoarseEpisodeMemoryMixin`

- `topology.py`
  - `TopologyMixin`
  - free-space topology, reachability, topology cache helpers

- `symbolic.py`
  - `SymbolicTransitionMixin`
  - symbolic transition summaries, symbolic planner-head training, action-effect memory

- `safety.py`
  - `SafetyMixin`
  - local contact safety and protected-terminal risk helpers

- `terminal_memory.py`
  - `TerminalOutcomeMemoryMixin`
  - exact terminal keys, terminal prototypes, basin penalties, diagnostics

- `objectivity.py`
  - `ObjectivityMixin`
  - affordance sample buffering, label derivation, objectivity-head training/scoring

- `candidate_generation.py`
  - `CandidateGenerationMixin`
  - hunter-weight/intrigue helpers and candidate proposal override

- `scoring.py`
  - `CandidateScoringMixin`
  - Hunter-Seeker score-component assembly and candidate trace diagnostics

- `action_selection.py`
  - `ActionSelectionMixin`
  - beam/random action selection wrappers and risk-patcher handoff

- `runtime_lifecycle.py`
  - `RuntimeLifecycleMixin`
  - `step`, terminal callbacks, and reset bookkeeping

- `phase_policy.py`
  - `PhasePolicyMixin`
  - phase-template helper logic and recovery/escalation bookkeeping
  - default policy remains gated by `enable_agent_phase_policy`

- `risk_arbitration.py`
  - `RiskArbitrationMixin`
  - all-risky recovery, risk patcher, trace arbitration selectors

- `post_veto.py`
  - `PostVetoDiagnosticsMixin`
  - online hybrid trace summaries and post-veto candidate diagnostics

- `self_model.py`
  - `SelfModelMixin`
  - neural self-model feature construction, pending-loss ordering, outcome calibration targets, advance/predict loop

- `measurement.py`
  - `MeasurementMixin`
  - failure summaries, measurement summaries, event dumps

- `persistence.py`
  - `PersistenceMixin`
  - belief persistence, trajectory loading, checkpoint save/load

- `smoke_tests.py`
  - legacy in-file smoke test runner, still callable via
    `python -m claude_sandbox.hunter_seeker.agent`

Next extraction targets:

- `phase_teacher.py`
  - future phase-policy-as-teacher collection/distillation, separate from live
    policy.

- `agent.py`
  - current `HunterSeekerAgent` constructor/composition shell.
  - keep constructor glue here; move behavior into named mixins.

- `stockfish/`
  - separate refactor target for `stockfish/agent.py`.
  - `models.py` now holds the stockfish neural heads and loop poolers.
  - `encoding.py` now holds Ouro/backbone loading, loop diagnostics, dense-frame
    conversion, and adapter-routed fallback actions.
  - `observation.py` now holds observation-learning/pretraining helpers.
  - `persistence.py` now holds checkpoint and solved-trajectory persistence.
  - `replay.py` now holds `TransitionReplayBuffer` and its samplers.
  - `runtime.py` now holds reset/step/run lifecycle and solved-prefix bookkeeping.
  - `search.py` now holds candidate generation, scoring, search-cache handling,
    and beam selection.
  - `model_basin.py` now holds model-basin diagnostic sampling and trace fields.
  - `search_types.py` now holds search-node dataclasses.
  - `training.py` now holds ranker/action-prior/world-model/anchor training.
  - `utils.py` now holds saliency/click-normalization/top-k helpers.
  - `stockfish/agent.py` is now the compatibility constructor
    and wiring shell.
  - Keep this separate from Hunter-Seeker behavior-preserving extraction.

Refactor invariants:

- Do not change behavior while moving code.
- Keep the old `arc_agent_hunter_seeker_codex` shim import-compatible.
- Keep game/level/phase scaffolding out of default policy.
- Keep every behavior-affecting term source-visible in diagnostics.
- Run compile/import checks after each slice.
