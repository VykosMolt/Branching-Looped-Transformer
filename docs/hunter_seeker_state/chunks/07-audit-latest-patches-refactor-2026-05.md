<!-- Source: PROJECT_STATE_HUNTER_SEEKER.md lines 9002-9788 before the 2026-05-14 split. -->
<!-- Source chunk SHA256: 7fa3f51e70af0d610a5bd60b46a35f02ec9bb10421cd3591f27baaa6c0390bfb -->

## Architecture Audit (2026-05-04)

Scope:

- Read the active `claude_sandbox`, `tools`, and `tests` architecture after the RLTT pre-acquisition hardening pass.
- Ran Claude Code in a terminal with the same audit task. It returned a compact summary rather than a long report; current-code claims were re-checked locally before being accepted.
- Excluded quarantined backup files from active-code conclusions.

Verification:

- `py_compile` passed for active `claude_sandbox/*.py`, `tools/*.py`, and tests.
- Full CPU test suite passed:
  - `430 passed, 1 skipped`.
- GPU visibility check:
  - RTX 5070 Ti Laptop GPU;
  - `43 C`;
  - `1%` utilization;
  - `809 MiB / 12227 MiB`.

Current active-code shape:

- Active code surface is about `43,424` Python lines outside tests.
- Tests are about `16,915` Python lines.
- Largest active bodies:
  - `HunterSeekerAgent`: `20,621` lines;
  - `PairwiseARCSearchAgent`: `6,360` lines;
  - `_run_smoke_tests`: `1,391` lines;
  - `TransitionReplayBuffer`: `1,206` lines;
  - `HunterSeekerAgent.step`: `1,169` lines;
  - `ObjectTable`: `1,147` lines;
  - `HunterSeekerAgent.score_candidates`: `1,036` lines;
  - `HunterSeekerAgent.on_game_over`: `1,033` lines;
  - `_compact_score_components`: `554` lines.
- Active code still has about `213` broad `except Exception` / bare-except handlers. Many are diagnostic guards, but several silent-drop paths should expose counters.

Claude Code audit cross-check:

- Claude reported that five prior audit issues are already fixed in the current tree:
  - encoder padding/mask issue;
  - cortex-off temporal aggregator bias bleed;
  - broad `env.step` swallowing in the train harness;
  - backup `.bak` relocation;
  - `d_model % 4` positional-encoding guard.
- Claude's one claimed current HIGH issue, "loop-pooler eval-mode missing try/finally", is not present in the current file: the pooler eval path now saves `pooler_was_training` and restores it in `finally`.
- Accepted current Claude-side concerns after local verification:
  - anchor coefficient has no CLI range guard;
  - risk-patcher `is_risky` and `chosen_has_risk` use partially divergent thresholds;
  - pending-state rollback logic is duplicated in the train harness;
  - diagnostics/memory silent-drop paths need counters.

High-confidence bugs / fixes to do next:

1. `SceneParser.parse()` fallback cache key is shape-unsafe.
   - Current fallback uses `hash(frame.astype(np.uint8).tobytes())` when no `frame_hash` is supplied.
   - Two different frame shapes with identical bytes can return the wrong cached scene.
   - Most production calls pass `agent._hash_frame(frame)`, but the public parser API, smoke tests, and direct tests can still hit this.
   - Fix:
     - include shape, dtype, and bytes in the fallback hash;
     - add a regression with same bytes / different shape.

2. Hunter checkpoint save/load is behind the base pairwise checkpoint surface.
   - Base pairwise saves `loop_pooler_kind` and `aux_world_model_optimizer`.
   - Hunter save currently omits both.
   - Base pairwise load skips pooler weights when `loop_pooler_kind` mismatches.
   - Hunter load directly loads `loop_pooler` without the saved-kind mismatch guard.
   - Fix:
     - mirror base keys in Hunter save;
     - load `aux_world_model_optimizer`;
     - use the base loop-pooler kind mismatch guard;
     - add a checkpoint round-trip regression.

3. GPU runs still need a fail-fast CUDA requirement.
   - Core agent device selection remains `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`.
   - If CUDA visibility breaks, a long run can silently fall back to CPU.
   - Fix:
     - add `--require_cuda` or explicit `--device cuda` handling to run/probe harnesses;
     - log `torch.cuda.get_device_name(0)` at run start;
     - fail before training when CUDA is required but unavailable.

Medium-priority correctness / observability items:

- `RecentExactTransitionWindow.add()` and diagnostics currently swallow all exceptions and return no signal. Add skip/failure counters to measurement summaries.
- Hunter checkpoint load swallows failed `transition_effect_engram_records` and `recent_exact_transition_records` restores. Add a warning/counter instead of silent `pass`.
- `AttentionPool` in `evaluator_pairwise_codex.py` can produce NaNs if any row has an all-zero attention mask. Normal tokenizer paths should be nonempty, but a defensive fallback/test is cheap.
- `utilities/evaluator/run_post_rltt_probe_bundle.py` records return codes/durations but does not capture child stdout/stderr into durable logs. Failed probes can lose the useful error context.
- Anchor coefficients should be bounded or at least rejected outside a sane interval; negative or huge values currently parse.
- Risk-patcher all-risky aggregation should use one helper for chosen and aggregate risk classification so threshold drift does not create inconsistent recovery behavior.

Performance / architecture targets:

- Topology is now functionally much cleaner, but `_compute_free_space_topology()` still has avoidable Python work:
  - full object-table signature construction for cache keys;
  - Python BFS plus optional connected-component labeling over the same passable grid;
  - distance-map work even when only reachable masks are needed.
- Safe next optimizations:
  - give `ObjectTable` a version counter for topology/cache invalidation;
  - split reachable-mask and distance-map modes;
  - use the labeled component containing the avatar as the reachable set when distances are not requested;
  - compute region graph and branch/basin diagnostics only on cadence or under pressure.
- Replay / observation memory scans are still O(N) over bounded buffers. This is acceptable at current caps but should be indexed before larger video/general-domain memories.
- `EventLog` and measurement summary helpers still re-scan current-run events in several places; cache per measurement build.

Domain-agnostic cleanup still open:

- The live architecture is much less ARC-bound than v17, but naming is still color-heavy:
  - `ColorPriorTable`;
  - `ObjectRecord.color`;
  - `patch_color_head`;
  - `tracked_colors`;
  - many event payload keys.
- This should become `label` / `entity_label` terminology during the later cleanup pass.
- `SceneParser(min_area=2)` is still an ARC-ish default. For general symbolic/video domains, a single-cell component may be meaningful. Make the minimum object area adapter-provided before broader transfer work.
- Measurement/report scripts intentionally contain game ids such as `ls20`, `ft09`, and `wa30`; those are acceptable in diagnostics, not in core cognition.

Deferred structural cleanup:

- Do not start the large Hunter-Seeker split yet; that remains deferred until topology and observational learning are both stable.
- The first safe extraction later should be:
  - move `_run_smoke_tests` out of the agent file into tests;
  - extract score-component compaction into a small diagnostics module;
  - extract checkpoint save/load surfaces;
  - then split phase-state, terminal memory, and topology trackers.

Immediate recommended patch order if we patch from this audit:

1. Fix `SceneParser` fallback hash and add the regression.
2. Fix Hunter checkpoint parity with base pairwise checkpointing.
3. Add fail-fast CUDA requirement/logging for GPU runs.
4. Add memory-load/diagnostic silent-drop counters.
5. Add `AttentionPool` all-zero-mask guard.

No behavior/topology calibration patch was enacted during this audit.

## Post-Audit Correctness / Safe-Optimization Patch (2026-05-04)

Implemented immediately after the architecture audit.

Correctness fixes:

- `SceneParser.parse()` fallback cache key is now shape/dtype aware when callers omit `frame_hash`.
  - It no longer hashes raw bytes alone.
  - Regression added for same bytes / different shape.
- `PairwiseARCSearchAgent._hash_frame()` is also shape/dtype aware.
  - This matters because production parser calls generally pass `agent._hash_frame(frame)`.
- Hunter checkpoint parity with base pairwise checkpointing is fixed:
  - saves `loop_pooler_kind`;
  - saves `aux_world_model_optimizer`;
  - loads `aux_world_model_optimizer` when optimizer state is being restored;
  - skips pooler weights with a visible warning when saved/current `loop_pooler_kind` mismatch.
- Recent exact transition diagnostics now expose guard counters:
  - `obs_recent_exact_add_exception_count`;
  - `obs_recent_exact_diagnostics_exception_count`.
- Hunter memory restore no longer silently ignores failed observation/engram memory loads:
  - failed `transition_effect_engram_records` restore increments `_memory_load_failure_counts` and prints a warning;
  - failed `recent_exact_transition_records` restore does the same.
- `AttentionPool` now handles all-zero attention-mask rows without producing NaNs.
  - All-invalid rows return a zero pooled vector.
- ARC train harness now fails fast on missing CUDA by default.
  - Use `--allow_cpu` for an intentional CPU run.
  - Run start logs CUDA device name and device count when CUDA is visible.
- Anchor coefficients are range-checked at CLI parse time:
  - `--anchor_coefficient` and `--pretrain_anchor_coefficient` must be finite and in `[0.0, 1.0]`.
- Risk-patcher aggregate and chosen-candidate risk classification now share one helper:
  - `_risk_patcher_candidate_has_risk()`;
  - avoids threshold drift between `all_candidates_risky` and `chosen_has_risk`.

Safe operational/performance cleanups:

- `_clear_agent_pending_transition()` centralizes pending-state rollback in `train_arc_codex.py`.
  - This removes duplicate rollback code in recoverable and unexpected `env.step` exception paths.
- `_load_ouro()` now passes:
  - `low_cpu_mem_usage=True`;
  - `local_files_only=True` when `HF_HUB_OFFLINE=1` or `TRANSFORMERS_OFFLINE=1`.
  - This keeps RLTT/Ouro loading aligned with offline acquisition runs and reduces CPU memory pressure.
- `utilities/evaluator/run_post_rltt_probe_bundle.py` now writes durable child-process logs:
  - `output_dir/logs/NN_script_name.log`;
  - manifest entries include `log_path` and a `log_tail`;
  - failed probes no longer lose stdout/stderr context.

Safety decision:

- Deeper topology cache restructuring was not enacted in this patch.
  - `ObjectTable` version-counter invalidation and reachable-mask/distance-map splitting are plausible speedups, but they touch cache invalidation and topology semantics.
  - Keep those for a measured behavior pass, not a blind audit-cleanup patch.

Verification:

- Focused regressions:
  - `6 passed, 313 deselected`.
- Full `py_compile` over active sandbox/tools/tests passed.
- Probe bundle dry-run passed and wrote a manifest under `/tmp/ouro_probe_bundle_dryrun_audit`.
- Train harness guard checks:
  - invalid `--anchor_coefficient 2` rejects at parse time;
  - `CUDA_VISIBLE_DEVICES=''` without `--allow_cpu` rejects at parse time.
- Full CPU test suite:
  - `435 passed, 1 skipped`.
- GPU status after verification:
  - RTX 5070 Ti Laptop GPU;
  - `38 C`;
  - `14%` utilization at sampling;
  - `969 MiB / 12227 MiB`.

Current state:

- Audit-noticed concrete bugs are patched.
- Behavior/topology calibration was not changed.
- The remaining useful optimization work is measured topology/cache work, not safe blind cleanup.

## RLTT Mixed Regression + Local Agent Wrapper Update (2026-05-04)

RLTT is now the default Ouro backbone for current regressions.

Mixed CUDA regression:

- Command shape:
  - `/home/moloch/ouro_project/venv/bin/python -m claude_sandbox.train_arc_codex --agent hunter_seeker --games ls20 tr87 wa30 --backbone_mode ouro --ouro_model_path models/ouro_rltt_local --checkpoint checkpoints_running/sprint4_encoder_reverted.pt --load_trajs claude_sandbox/trusted_plus_expanded --pretrain_iters 1 --max_steps 180 --n_runs 1 --eps 0.0 --no_replay --running_checkpoint "" --checkpoint_dir claude_sandbox/checkpoints_rltt_regression_mixed_cuda --save_trajs_dir claude_sandbox/solved_sequences_expanded --dump_events_dir claude_sandbox/perf_event_dumps_rltt_regression_mixed_cuda --thermal_guard --thermal_check_every 5`
- Device/thermal:
  - CUDA-visible RTX 5070 Ti Laptop GPU;
  - thermal guard active;
  - sampled GPU temperature stayed around `52-55 C`;
  - GPU utilization was generally high during model work.
- Artifacts:
  - event root: `claude_sandbox/perf_event_dumps_rltt_regression_mixed_cuda`;
  - checkpoints: `claude_sandbox/checkpoints_rltt_regression_mixed_cuda`;
  - saved checkpoints:
    - `arc_ls20_run1.pt`;
    - `arc_tr87_run1.pt`;
    - `arc_wa30_run1.pt`.

Results:

- `ls20`, `max_steps=180`:
  - levels completed: `3`;
  - level completion steps from stdout/event log: `13`, `136`, `175`;
  - score: `21.428571428571427`;
  - event dump: `claude_sandbox/perf_event_dumps_rltt_regression_mixed_cuda/ls20/run_1.json`;
  - measurement: `claude_sandbox/perf_event_dumps_rltt_regression_mixed_cuda/ls20/measurement_run_1.json`.
- `tr87`, `max_steps=180`:
  - levels completed: `4`;
  - level completion steps: `37`, `67`, `106`, `135`;
  - level 3 improved from `97` to `39` actions during this run;
  - score: `47.61904761904762`;
  - event dump: `claude_sandbox/perf_event_dumps_rltt_regression_mixed_cuda/tr87/run_1.json`;
  - measurement: `claude_sandbox/perf_event_dumps_rltt_regression_mixed_cuda/tr87/measurement_run_1.json`.
- `wa30`, `max_steps=180`:
  - levels completed: `1`;
  - level 1 completed at step `125`;
  - score: `0.7169422222222223`;
  - event dump: `claude_sandbox/perf_event_dumps_rltt_regression_mixed_cuda/wa30/run_1.json`;
  - measurement: `claude_sandbox/perf_event_dumps_rltt_regression_mixed_cuda/wa30/measurement_run_1.json`.

Interpretation:

- This run does not show a broad topology/code regression.
- Important comparison for `ls20`:
  - older `topology_norm_ls20_long_gpu/run_5` completed the same first three levels at steps `13`, `136`, and `175`;
  - its level 4 came later at step `267`;
  - the current RLTT mixed regression was capped at `180`, so it could not test whether RLTT/current code still reaches level 4.
- Therefore the apparent `ls20` drop from the earlier level-4 topology result is currently a step-budget artifact, not evidence of RLTT damage or a bad patch.
- `tr87` still reaches live level 4 under RLTT/current code, and the run produced a genuine level-3 action-count improvement.
- `wa30` at `180` steps is only an early-behavior smoke. It cannot test the known long level-3 ceiling, because previous notes established that useful wa30 level-3 validation needs `max_steps>=500`.

Recommended next evidence:

1. Like-for-like long `ls20` RLTT check with `max_steps>=300` if we want to prove RLTT/current code still reaches the old level-4 topology result.
2. Long `wa30` RLTT run with `max_steps>=500`, preferably `n_runs=3`, only after deciding whether to spend the time.
3. Treat the current mixed run as a regression smoke, not as a final ceiling measurement.

Local-agent wrapper material from this mixed regression note has been split out to `PROJECT_STATE_LOCAL_AGENT.md`.

## RLTT Loop-Architecture Calibration Patch (2026-05-04)

Implemented after returning from the local wrapper side task.

Goal:

- Take advantage of RLTT as a looped/recurrent model, not a static encoder.
- Keep behavior changes conservative: expose and calibrate loop signals first; do not add a hard game-specific policy rule.

Files changed:

- `claude_sandbox/arc_agent_pairwise_stockfish_codex.py`
- `claude_sandbox/arc_agent_hunter_seeker_codex.py`
- `claude_sandbox/train_arc_codex.py`
- `tests/unit/test_causal_correctness.py`

Implemented:

- Added `ouro_total_ut_steps` as a first-class agent/runtime config.
  - Constructor argument on `PairwiseARCSearchAgent` / inherited by Hunter-Seeker.
  - CLI flag: `--ouro_total_ut_steps`.
  - Valid range: integer `[1, 256]`.
  - Default remains `4`.
  - `_load_ouro()` now sets `model.config.total_ut_steps` when the loaded Ouro/RLTT config exposes it.
  - `early_exit_threshold` remains fixed at `1.0`; early-exit/gate telemetry is diagnostic-first.
- Added explicit RLTT runtime signature:
  - `model_id`;
  - resolved local model path when the path exists;
  - `total_ut_steps`;
  - `early_exit_threshold`;
  - `use_cache=False`;
  - whether the loop pooler consumes all loop states.
- Changed trajectory CLS cache schema to `trajectory_cls_cache_v3_loop_config`.
  - Cached CLS tensors are now invalidated when RLTT model/depth/runtime config differs.
  - This prevents treating RLTT loop-state features as static encoder features.
  - Old v2 caches without loop runtime metadata are intentionally stale under this schema.
- Promoted loop/refinement telemetry into durable diagnostics:
  - helper: `_ouro_loop_diagnostics()`;
  - per-step `info["rltt_loop_diagnostics"]`;
  - Hunter `measurement_summary()["rltt_loop_diagnostics"]`;
  - candidate `score_components` now include:
    - `ouro_total_ut_steps`;
    - `ouro_loop_count`;
    - `ouro_confidence`;
    - `ouro_expected_exit`;
    - `ouro_survival`;
    - `ouro_exit_pdf`;
    - `ouro_exit_pdf_mass`;
    - `loop_delta`;
    - `loop_delta_ema`;
    - `loop_delta_ratio`;
    - `loop_refinement_pressure`;
    - `loop_compute_pressure`.
- Exit-gate diagnostic state now also tracks:
  - final survival;
  - per-loop exit-PDF mass;
  - observed loop count.
- Reset now clears all per-run RLTT loop diagnostic fields.
- Train config print now includes `ouro_total_ut_steps`.

Interpretation:

- `loop_delta` and `loop_delta_ema` remain the main downstream confidence proxy.
- RLTT exit-gate confidence and expected exit are recorded and already participate only through the conservative trust multiplier path; they are not a broad override.
- `loop_compute_pressure` is a compact diagnostic pressure signal combining:
  - low Ouro confidence;
  - late expected exit;
  - unusual loop-delta ratio.
- This gives the self-model, score traces, and measurement summaries access to "how much thinking the model seemed to need" without hardcoding any ARC/game behavior.

Verification:

- `py_compile` passed for:
  - `claude_sandbox/arc_agent_pairwise_stockfish_codex.py`;
  - `claude_sandbox/arc_agent_hunter_seeker_codex.py`;
  - `claude_sandbox/train_arc_codex.py`;
  - `tests/unit/test_causal_correctness.py`.
- Focused RLTT-loop tests:
  - `4 passed, 143 deselected`.
- Full causal-correctness file:
  - `147 passed`.
- CLI help shows `--ouro_total_ut_steps`.
- Invalid CLI value rejects:
  - `--ouro_total_ut_steps 0` fails at parse time.
- Constructor smoke:
  - `PairwiseARCSearchAgent(backbone_mode="encoder_only", ouro_total_ut_steps=7)` reports runtime signature with `total_ut_steps=7`;
  - remote model ids no longer get misleadingly resolved to a local cwd path.
- GPU status after verification:
  - around `44 C`;
  - low utilization;
  - about `733 MiB / 12227 MiB` used.

Operational note:

- A GPU launch attempted with an environment-prefix command saw `torch.cuda.is_available() == False` under the sandboxed invocation.
- Direct invocation through the approved project venv Python sees CUDA correctly.
- For future GPU runs here, prefer commands that begin with `/home/moloch/ouro_project/venv/bin/python` and pass local paths directly, rather than prefixing the command with `env ...`.

## Agent Phase-Policy Cleanup / Domain Boundary Patch (2026-05-05)

Reason:

- The late `wa30` work accumulated replay-derived phase/level machinery inside Hunter-Seeker.
- Some of that machinery was useful for diagnosing the trusted-trajectory discontinuity, but it had become active agent policy.
- Current rule: level/game specificity is allowed in ARC harnesses, measurements, fixtures, and ablation switches, but not in default live agent behavior.

Implemented:

- Added `enable_agent_phase_policy=False` as the Hunter-Seeker default.
  - CLI opt-in: `--enable_agent_phase_policy`.
  - This is now explicitly described as legacy ablation scaffolding, not default cognition.
- Default live policy no longer allows replay-derived phase/level templates to:
  - inject learned phase click candidates;
  - reorder directional action proposals;
  - add phase action or phase target score bonuses;
  - trigger exact phase guidance beam overrides;
  - trigger live recovery selection from phase traces;
  - activate phase continuity recovery windows;
  - update recovery/escalation/reseed/exact-veto phase-control state.
- Phase diagnostics/state can still exist in memory and checkpoints, but they are score-no-op / report-no-op unless the harness opts into `--enable_agent_phase_policy`.
- `on_level_complete()` no longer clears level-indexed phase/recovery maps on the default path; that cleanup also belongs to the opt-in phase-policy ablation.
- Click cooldown memory no longer keys by ARC level.
  - `_click_target_key(level, ...)` keeps the old argument for compatibility.
  - The key now uses a coarse stable scene-context hash plus target label/location.
  - This keeps cooldowns local to similar board states without baking `levels_completed` into policy.
- Removed the phase-mode scaling from `_directional_topology_bonus`.
  - Directional topology remains domain-general reachability/contact/topology scoring.
  - Recovery/escalation/reseed modes no longer amplify it by default.
- Fixed a gating bug introduced by this cleanup:
  - `phase_control_progress_signal` now defaults to the domain-general `progress_signal`;
  - engram outcome collection no longer references a phase-only variable when phase policy is disabled.
- Score components now expose `agent_phase_policy_enabled` so traces can prove whether the old machinery was active.

Current topology state after this cleanup:

- Active/default:
  - free-space topology, reachability, frontier, local contact, hazard-aware safety scoring;
  - controlled-avatar / controlled-set tracking;
  - terminal memory and engram recall;
  - risk patcher and model-basin diagnostics;
  - observation-effect terminal risk;
  - self-model/evaluator diagnostic signals.
- Default-disabled / legacy ablation only:
  - phase target templates;
  - phase action templates;
  - exact trusted phase guidance;
  - phase continuity recovery;
  - phase recovery/escalation/reseed/exact-veto state machine;
  - phase semantic calibration.
- Still present but should be extracted or deleted after the next regression pass:
  - the large `_phase_*` helper mass inside `arc_agent_hunter_seeker_codex.py`;
  - phase-state checkpoint payloads;
  - phase-specific score fields retained for old tests and ablation traces.

Verification:

- `py_compile` passed for:
  - `claude_sandbox/arc_agent_hunter_seeker_codex.py`;
  - `claude_sandbox/train_arc_codex.py`;
  - `tests/unit/test_codex_sandbox.py`.
- Focused phase-policy regressions:
  - initial patch: `3 passed, 180 deselected`;
  - after trace-compaction fix: `4 passed, 181 deselected`.
- Broader topology/phase/risk/engram/hazard slice:
  - `106 passed, 78 deselected`.
- Full sandbox unit file:
  - initial patch: `184 passed`;
  - after trace-compaction fix: `185 passed`.
- CLI help shows `--enable_agent_phase_policy`.

GPU smoke:

- First default-disabled RLTT smoke:
  - command used direct project venv Python, not an env-prefix wrapper;
  - CUDA preflight reported `NVIDIA GeForce RTX 5070 Ti Laptop GPU`;
  - RLTT loaded on `cuda:0`;
  - trusted trajectory bundle loaded: `14090` transitions;
  - phase templates were built in memory: `3477` target descriptors and `9408` action descriptors;
  - config intentionally omitted `--enable_agent_phase_policy`;
  - `wa30`, `max_steps=80`, `n_runs=1`, `eps=0.0`, `no_replay=True`, `train_every=0`;
  - event dump: `claude_sandbox/perf_event_dumps_phase_policy_default_smoke_wa30_rltt/wa30/run_1.json`;
  - measurement dump: `claude_sandbox/perf_event_dumps_phase_policy_default_smoke_wa30_rltt/wa30/measurement_run_1.json`;
  - GPU reached about `99%` utilization during live steps and stayed cool, around low-to-mid `50 C`.
- The first smoke found an instrumentation gap:
  - internal score components had `agent_phase_policy_enabled`;
  - compact event dumps did not preserve it.
- Trace-compaction fix:
  - `agent_phase_policy_enabled` is now retained in lean, candidate, and full compact score components;
  - action trace entries now also expose top-level `agent_phase_policy_enabled`.
- Second default-disabled RLTT smoke after trace fix:
  - `wa30`, `max_steps=20`, same RLTT/checkpoint/trusted/no-replay setup;
  - event dump: `claude_sandbox/perf_event_dumps_phase_policy_default_smoke_tracefix_wa30_rltt/wa30/run_1.json`;
  - trace audit:
    - `20/20` top-level action trace entries had `agent_phase_policy_enabled=False`;
    - `160/160` compact score component dicts had `agent_phase_policy_enabled=False`;
    - `0` score component dicts had `agent_phase_policy_enabled=True`;
    - `0` missing flag values;
    - `0` nonzero phase policy terms across `phase_target_bonus`, `phase_action_bonus`, `recovery_penalty`, `escalation_bonus`, `live_recovery_bonus`;
    - `0` retained `phase_template` / `phase_action_template` payloads.

Topology trio RLTT smoke:

- Command:
  - direct project venv Python;
  - `--games ls20 tr87 wa30`;
  - RLTT `models/ouro_rltt_local`;
  - checkpoint `checkpoints_running/sprint4_encoder_reverted.pt`;
  - `--load_trajs claude_sandbox/trusted_plus_expanded`;
  - `--pretrain_iters 0`;
  - `--max_steps 180`;
  - `--n_runs 1`;
  - `--eps 0.0`;
  - `--no_replay`;
  - `--train_every 0`;
  - no `--enable_agent_phase_policy`.
- CUDA/GPU:
  - CUDA preflight succeeded;
  - RLTT loaded on `cuda:0`;
  - live steps reached about `99-100%` GPU utilization;
  - GPU stayed cool, roughly `50-55 C` during live play.
- Artifacts:
  - checkpoints: `claude_sandbox/checkpoints_phase_policy_default_topology_trio_rltt/`;
  - event dumps: `claude_sandbox/perf_event_dumps_phase_policy_default_topology_trio_rltt/`.

Results:

- `ls20`:
  - `0` levels completed;
  - game over at step `129`;
  - current-run failure: `self_model`;
  - action distribution: `{4: 4, 2: 125}`;
  - dump: `claude_sandbox/perf_event_dumps_phase_policy_default_topology_trio_rltt/ls20/run_1.json`.
- `tr87`:
  - `0` levels completed;
  - game over at step `128` relative to the `tr87` run;
  - current-run failure: `mechanism`;
  - action distribution: `{4: 1, 3: 1, 1: 53, 2: 73}`;
  - dump: `claude_sandbox/perf_event_dumps_phase_policy_default_topology_trio_rltt/tr87/run_1.json`.
- `wa30`:
  - `0` levels completed;
  - no death by the `180` step cap;
  - action distribution: `{2: 2, 4: 1, 1: 176, 3: 1}`;
  - dump: `claude_sandbox/perf_event_dumps_phase_policy_default_topology_trio_rltt/wa30/run_1.json`.

Phase-policy audit for topology trio:

- `ls20`:
  - `129/129` top-level trace entries had `agent_phase_policy_enabled=False`;
  - `903/903` compact score component dicts had `agent_phase_policy_enabled=False`;
  - no nonzero phase policy terms;
  - no retained phase templates.
- `tr87`:
  - `128/128` top-level trace entries had `agent_phase_policy_enabled=False`;
  - `896/896` compact score component dicts had `agent_phase_policy_enabled=False`;
  - no nonzero phase policy terms;
  - no retained phase templates.
- `wa30`:
  - `180/180` top-level trace entries had `agent_phase_policy_enabled=False`;
  - `1440/1440` compact score component dicts had `agent_phase_policy_enabled=False`;
  - no nonzero phase policy terms;
  - no retained phase templates.

Interpretation:

- The phase-policy cleanup works: replay-derived phase templates can be loaded and built without affecting default action choice.
- But the default phase-disabled live policy is currently weak on the topology trio.
- Compared with earlier mixed RLTT evidence where `tr87` reached multiple levels, this run shows that the removed phase/replay scaffold had been carrying real route/action-sequence competence.
- The failure pattern is not "topology cannot compute"; the agent mostly collapses into repeated directional actions:
  - `ls20`: almost all action `2`;
  - `wa30`: almost all action `1`;
  - `tr87`: mostly actions `1/2`.
- This points to missing learned action-effect / route-policy competence, not a reason to re-enable game/level/phase templates in the agent.
- Next useful patch should stay domain-general:
  1. add a low-level anti-stall/action-diversity pressure based on repeated no-progress directional loops;
  2. make it state/effect based, not game/level based;
  3. prefer observational/inverse-action learning as the real solution;
  4. use the trio traces to decide whether the anti-stall patch should be score-level, candidate-generation-level, or self-model/evaluator pressure.

Tests added/updated:

- Default scoring ignores phase action templates even when helper recall finds a match.
- Harness opt-in enables phase action bonus for ablation.
- Click target keys are identical across different ARC level identifiers for the same scene/target.
- Candidate generation ignores exact phase ordering by default.
- Candidate generation still orders exact phase actions when `enable_agent_phase_policy=True`.

Next work:

1. Run a short RLTT GPU smoke with the default phase policy disabled and inspect whether `agent_phase_policy_enabled=False` is present through traces.
2. Run a longer `wa30` probe only after confirming the default-disabled path is stable.
3. If behavior does not regress, move the legacy phase subsystem out of the agent core into a harness/diagnostic module or delete it in pieces.
4. Continue observational/action-effect learning as the real route out of the `wa30` ceiling.
5. Continue the naming cleanup: replace agent-internal `color` wording with `label`/`value` where it is not literally rendered color.

Local-agent wrapper notes from 2026-05-06 onward have been split out to `PROJECT_STATE_LOCAL_AGENT.md`.

## Self-Model Boundary Notes From Local-Agent Split (2026-05-06)

Wrapper boundary:

- Do not wire the local terminal wrapper itself into Hunter-Seeker's self-model.
- The useful architectural move is to wire the same structured signals into the embodied agent self-model:
  - engram recall support/conflict;
  - evaluator/verifier outcome;
  - retry/correction pressure;
  - action-source and trust/mismatch state;
  - recent loop/cortex signatures.
- Keep this low-bandwidth and diagnostic-first. The self-model should receive "what my decision machinery just did and how well it worked", not arbitrary wrapper prose.

Hunter-Seeker neural self-model status:

- The real neural self-model scaffold is present and wired, but it is not yet proven as the finished active agency layer.
- Implemented components:
  - `claude_sandbox/self_model.py` contains `SelfModel`, `AffectiveState`, `AgentEventBundle`, `CortexMonitor`, and `TemporalContextAggregator`;
  - `SelfModel.build_input()` consumes loop delta, affect, track summary, `self_eval_summary`, and `thought_signature`;
  - Hunter-Seeker constructs the self-model in `passive`, `inject`, and `inject_aux_grad` modes;
  - `_compute_self_eval_summary()` feeds evaluator/risk/memory/topology/observation diagnostics into the self-model;
  - `_compute_thought_signature()` feeds domain-general internal-computation continuity signals;
  - `_self_model_advance_and_predict()` advances affect + GRU and produces temporal features/context token;
  - replay/ranker snapshots carry self-model hidden state, self-eval summaries, and thought signatures for temporal-feature reconstruction.
- Remaining limitations:
  - default training still runs with `--self_model_mode off` unless explicitly enabled;
  - context-token injection is identity-start/zero-init and needs training evidence before it can matter;
  - loop delta is still partly represented by a scalar proxy expanded through a fixed pattern, not a learned full loop-state projection;
  - affective decay/excitation values remain hand-coded;
  - evaluator/self-diagnosis signals are inputs to the self-model, but the self-model does not yet own the evaluator loop as an internal controller;
  - active self-model benefit still needs a controlled ladder/regression with `passive` and `inject_aux_grad`.
- Current interpretation:
  - Hunter-Seeker has the serious neural self-model substrate wired and tested;
  - it is not yet the final "thing making choices and noticing its own issues";
  - the next meaningful proof step is empirical, not more naming or scaffold work.

## Domain-Neutral Naming Cleanup - 2026-05-14

This implements the deferred naming hygiene called out in the pre/post-ladder
notes and later state entries: move agent-internal terminology toward
label/value/entity language where the integer is a symbolic observation label,
while keeping old `color` names alive where event dumps, checkpoints, tests, or
ARC-specific adapters still depend on them.

Completed in this pass:

- `LabelPriorTable` is now the neutral public class for game-persistent
  affordance priors.
- `ColorPriorTable` remains a direct compatibility alias.
- `HunterSeekerAgent` now owns `label_prior_table`; `color_prior_table` remains
  an alias for old checkpoint and helper code.
- `SceneObject`, `ObjectRecord`, and `TrackRecord` expose `label` properties
  backed by the legacy `color` storage field.
- `ObjectTable` now exposes neutral helper names:
  - `label_records`;
  - `_ensure_record_for_label(...)`;
  - `_centroid_for_label(...)`;
  - `tracks_by_label(...)`;
  - `object_intrigue_score_by_label(...)`.
- `PairwiseSearchAgent` is now the neutral public alias for
  `PairwiseARCSearchAgent`.
- `DEFAULT_SYMBOLIC_GRID_SIZE` is now the neutral public alias for
  `LEGACY_ARC_GRID_SIZE`.
- Runtime diagnostics now emit `loop_pooler_gate`; `gru_gate` remains as a
  compatibility alias.
- The ARC harness now imports/constructs `PairwiseSearchAgent` while remaining
  explicitly ARC-specific at the environment/adapter boundary.

Intentionally left compatible:

- JSON/event/checkpoint payloads can still contain `color`, because old runs and
  persistence loaders depend on that key.
- `ArcObservationAdapter` and `ArcActionAdapter` keep their ARC names because
  those classes are legitimately ARC-specific adapters.
- Scene parsing still reads literal ARC palette values as `color` internally in
  some paths. New API-facing code should prefer `label` unless it is describing
  a rendered palette value.

Verification:

- `venv/bin/python -m py_compile` passed for the renamed modules and entry
  points.
- `venv/bin/python -m pytest tests/unit/test_domain_neutral_aliases.py tests/unit/test_topology_sprint5.py tests/unit/test_codex_sandbox.py -q`
  passed (`206 passed`).
- `timeout 60 venv/bin/python claude_sandbox/arc_agent_hunter_seeker_codex.py`
  passed the legacy Hunter-Seeker smoke suite.

## Neutral File Rename Pass - 2026-05-14

Active implementation files now live at neutral paths. The old ARC/Codex names
remain as thin compatibility shims so historical commands, checkpoints, and
older tests keep working.

Renamed implementation paths:

- `claude_sandbox/arc_agent_hunter_seeker_codex.py` ->
  `claude_sandbox/hunter_seeker/agent.py`
- `claude_sandbox/arc_agent_pairwise_stockfish_codex.py` ->
  `claude_sandbox/stockfish/agent.py`
- `claude_sandbox/train_arc_codex.py` ->
  `claude_sandbox/train_arc.py`
- `claude_sandbox/action_adapters_codex.py` ->
  `claude_sandbox/action_adapters.py`
- `claude_sandbox/observation_adapters_codex.py` ->
  `claude_sandbox/observation_adapters.py`
- `claude_sandbox/grid_encoder_codex.py` ->
  `claude_sandbox/grid_encoder.py`
- `claude_sandbox/observation_learning_codex.py` ->
  `claude_sandbox/observation_learning.py`
- `claude_sandbox/evaluator_pairwise_codex.py` ->
  `claude_sandbox/pairwise_evaluator.py`

Compatibility shims still present:

- The old files import/re-export the new modules.
- `python claude_sandbox/arc_agent_hunter_seeker_codex.py` still runs the
  legacy smoke suite.
- `python -m claude_sandbox.train_arc_codex` still calls
  `claude_sandbox.train_arc.main()`.

Import guidance from here forward:

- Use `claude_sandbox.hunter_seeker.agent` for `HunterSeekerAgent`.
- Use `claude_sandbox.stockfish.agent` for `PairwiseSearchAgent` and stockfish
  compatibility exports.
- Use unsuffixed module names for adapters, encoder, observation learning,
  evaluator, and the ARC harness.
- Treat older references earlier in this state file as historical unless a
  command explicitly depends on the compatibility shim.

Verification for the rename pass:

- `venv/bin/python -m py_compile` passed for all new implementation files and
  old shim files.
- `venv/bin/python -m pytest tests/unit/test_domain_neutral_aliases.py tests/unit/test_topology_sprint5.py tests/unit/test_codex_sandbox.py -q`
  passed (`206 passed`).
- `venv/bin/python -m pytest tests/unit -q --ignore=tests/unit/test_local_agent_wrapper.py`
  passed (`426 passed`).
- `timeout 60 venv/bin/python claude_sandbox/arc_agent_hunter_seeker_codex.py`
  passed the old shim smoke suite.
- `timeout 60 venv/bin/python -m claude_sandbox.hunter_seeker.agent` passed the
  new module smoke suite.
- `timeout 60 venv/bin/python claude_sandbox/hunter_seeker/agent.py` passed the
  direct-script smoke suite.

## Hunter-Seeker / Stockfish Mixin Extraction - 2026-05-14

After the neutral file rename, the remaining large Hunter-Seeker behavior
methods were moved out of `claude_sandbox/hunter_seeker/agent.py` into named
mixins. This pass was behavior-preserving: no policy/scoring algorithm was
intentionally changed.

New Hunter-Seeker module boundaries:

- `claude_sandbox/hunter_seeker/agent.py`
  - now constructor/composition shell (`684` lines);
  - owns `HunterSeekerAgent.__init__` and mixin wiring.
- `claude_sandbox/hunter_seeker/candidate_generation.py`
  - `CandidateGenerationMixin`;
  - hunter-weight/intrigue helpers, action helper methods, and
    `generate_candidates(...)`.
- `claude_sandbox/hunter_seeker/scoring.py`
  - `CandidateScoringMixin`;
  - `score_candidates(...)` and score-component trace writes.
- `claude_sandbox/hunter_seeker/action_selection.py`
  - `ActionSelectionMixin`;
  - `beam_search_action(...)` and `select_action(...)`.
- `claude_sandbox/hunter_seeker/runtime_lifecycle.py`
  - `RuntimeLifecycleMixin`;
  - `step(...)`, `on_level_complete(...)`, `on_game_over(...)`, and
    `reset_for_new_game(...)`.

Stockfish refactor continuation:

- `claude_sandbox/stockfish/search.py` is now `1425` lines and keeps candidate
  generation, successor prediction, scoring, encode cache, transposition table,
  beam search, and select-action logic.
- `claude_sandbox/stockfish/model_basin.py` now holds
  `StockfishModelBasinMixin` and model-basin diagnostic sampling/trace fields.
- `StockfishSearchMixin` inherits `StockfishModelBasinMixin`, so public
  behavior remains reachable through the same agent class.

Static tests that inspect source paths were updated to follow the extracted
owners:

- score/terminal trace checks now read `hunter_seeker/scoring.py`;
- step/on-game-over checks now read `hunter_seeker/runtime_lifecycle.py`.

Verification for this extraction:

- `venv/bin/python -m py_compile` passed for the new Hunter-Seeker mixins,
  `hunter_seeker/agent.py`, `stockfish/search.py`, `stockfish/model_basin.py`,
  `stockfish/agent.py`, and the updated causal static tests.
- Import/MRO smoke confirmed `HunterSeekerAgent` instances are instances of
  `CandidateGenerationMixin`, `CandidateScoringMixin`, and
  `RuntimeLifecycleMixin`; `StockfishSearchMixin` subclasses
  `StockfishModelBasinMixin`.
- Focused suite with refactor-aware static checks passed (`213 passed`).
- `venv/bin/python -m pytest tests/unit -q --ignore=tests/unit/test_local_agent_wrapper.py`
  passed (`426 passed`).
- `timeout 60 venv/bin/python -m claude_sandbox.hunter_seeker.agent` passed.
- `timeout 60 venv/bin/python claude_sandbox/hunter_seeker/agent.py` passed.
- `timeout 60 venv/bin/python claude_sandbox/arc_agent_hunter_seeker_codex.py`
  passed.

Map files updated:

- `PROJECT_ARCHITECTURE_MAP.md`;
- `PROJECT_COMPONENT_REFERENCE.md`;
- `claude_sandbox/hunter_seeker/README.md`.

Next cleanup targets at that point were lifecycle and phase-policy splitting.
Those targets were completed later in this same chunk under
`Hunter-Seeker Phase/Lifecycle Split - 2026-05-14`. `scoring.py` remains a
candidate for a later terminal/engram/phase score-writer extraction.

## Root Utilities Consolidation - 2026-05-14

The root-level `tests/` and `tools/` directories were moved under
`utilities/` to keep the repository root smaller and more navigable:

- `utilities/tests/` now owns Hunter-Seeker/core unit, integration, report, and
  local-agent wrapper suites.
- `utilities/evaluator/` now owns evaluator/domain-transfer probes and the
  post-RLTT diagnostic bundle.
- `utilities/tools/` now owns shared conversion tools.

Current entry points were updated:

- `pytest.ini` collects `utilities/tests/`; local-agent wrapper tests now live
  under `utilities/tests/local_agent/`.
- `pytest.ini` also adds `src/local_agent/` to `pythonpath`, because the
  local-agent tests intentionally import wrapper modules as top-level names
  such as `ouro_server` and `ouro_memory_safety`.
- `utilities/evaluator/run_post_rltt_probe_bundle.py` resolves the repo root from
  the new location and launches manual probes through `utilities/evaluator/probes/`.
- `PROJECT_TREE_MAP.md`, `PROJECT_COMPONENTS.md`, `README.md`, and the utility
  README files describe the new layout.
- Historical state chunks and imported memos may still contain old root
  `tests/` or `tools/` command examples; treat those as preserved historical
  context unless a newer map or wrapper says otherwise.

Compatibility fixes found while proving full pytest after the move:

- `run_task_mode(...)` now accepts both the current primary-only positional
  form and the older secondary-manager positional form still used by some
  local-agent harnesses.
- Local-agent tool harnesses now unpack the current six-value
  `create_agent_runtime()` return shape and keep `qwen_mgr = None` only as a
  compatibility placeholder.
- `ouro_model_managers.py` restores the `KEEP_BACKENDS_LOADED_MODE` setting,
  `effective_keep_backends_loaded()`, and a minimal `MultiBackendModelManager`
  facade for tests/probes that inject fake primary/secondary managers.
- Local self-model context again surfaces compact policy-decision and internal
  signature lines expected by wrapper self-model tests.

Verification after consolidation:

- Full active Python compile passed for `src/`, `utilities/`, and `compat/`.
- The old utility and artifact/data symlinks were removed; legacy path
  replacements are documented under `compat/legacy_paths/`.
- Root directory no longer has top-level `tests/` or `tools/`.
- `venv/bin/python -m pytest -q` passed:
  `613 passed, 1 skipped, 18 subtests passed`.
- `utilities/evaluator/run_post_rltt_probe_bundle.py --dry-run --offline ...`
  emitted `utilities/evaluator/probes/...` child commands with `--device cuda`.

## Output Placement Hardening - 2026-05-14

Follow-up cleanup after the utilities move made active default write paths match
the reorganized tree:

- Manual probe defaults now write under `artifacts/reports/evaluator/` instead of
  recreating root `runs/`.
- Local-agent harness defaults now write under
  `artifacts/reports/local_agent/`.
- Local-agent writable state now defaults to `artifacts/local_agent/`; existing
  `src/local_agent/projects/` and `src/local_agent/browser_profiles/` runtime
  data were moved there.
- The root `sitecustomize.py` import hook was removed. The project venv now
  carries both `src/` and `src/local_agent/` in
  `venv/lib/python3.14/site-packages/ouro_project_src.pth`, and `pytest.ini`
  carries the same source roots for tests.
- Pytest cache writing is disabled with `-p no:cacheprovider`; `.gitignore`
  also ignores `.pytest_cache/`, `__pycache__/`, and accidental root `runs/`.

Current rule of thumb:

- active source under `src/`;
- tests and shared tools under `utilities/`;
- generated reports under `artifacts/reports/`;
- ARC SDK environment caches under `data/arc_agi3/environment_files/`;
- local-agent runtime state under `artifacts/local_agent/`;
- no active default should create root `runs/`, root `tests/`, root `tools/`,
  or source-tree local-agent runtime directories.

## Hunter-Seeker Phase/Lifecycle Split - 2026-05-14

The deferred post-refactor splits for lifecycle and phase policy have now been
applied as behavior-preserving mixin extractions.

Runtime lifecycle:

- `runtime_lifecycle.py` now owns `step(...)` orchestration only (`1343`
  lines).
- `runtime_terminal.py` owns level-complete and game-over terminal callbacks,
  including protected-terminal starvation diagnostics (`1502` lines).
- `runtime_reset.py` owns run reset bookkeeping (`171` lines).
- MRO smoke confirmed `HunterSeekerAgent.reset_for_new_game` resolves to
  `RuntimeResetMixin`, and `on_level_complete` / `on_game_over` resolve to
  `RuntimeTerminalMixin`.

Phase policy:

- `phase_policy.py` is now target/click phase-policy glue and mixin
  composition (`292` lines).
- `phase_templates.py` owns phase-template storage, lookup, branch/source
  bookkeeping, compaction, and trusted-trajectory template building (`1447`
  lines).
- `phase_recovery.py` owns recovery/escalation/reseed windows, live-effect
  recovery memory, nonviable basin pressure, and semantic calibration (`1326`
  lines).
- `phase_state.py` owns frame/object/component signatures, phase-state
  alignment trust, and exact/approx resync (`1577` lines).
- `score_components.py` owns selected-score lookup, passive calibration, and
  score-component compaction (`1297` lines).

Architecture check:

- These extractions keep the old behavior surfaces on `HunterSeekerAgent`
  through normal Python MRO rather than compatibility wrappers.
- The split follows the active conceptual boundaries in the component map:
  templates, recovery, phase-state alignment, score diagnostics, step
  lifecycle, terminal lifecycle, and reset lifecycle.
- The remaining large Hunter-Seeker behavior modules are now `objects.py`,
  `risk_arbitration.py`, `safety.py`, `memory.py`, `terminal_memory.py`, and
  `scoring.py`; `scoring.py` is still the main candidate for a later
  terminal/engram/phase score-writer extraction.

Verification run for this split:

- `py_compile` passed for the new phase/lifecycle modules, scoring, action
  selection, post-veto, and `agent.py`.
- Focused score-component tests passed: `5 passed`.
- Runtime lifecycle/terminal tests passed: `8 passed`.
- Phase-state alignment tests passed: `12 passed`.
- Phase recovery tests passed: `7 passed`.
- Phase-template tests passed after fixing one stale static reference to the
  old `PhasePolicyMixin` class name: `8 passed`.
- Protected-terminal starvation helpers were moved from `phase_templates.py`
  into `runtime_terminal.py` after architecture review; focused terminal tests
  passed: `5 passed`.
- Full active pytest passed after the final split:
  `613 passed, 1 skipped, 18 subtests passed`.

## Post-Refactor Deferred Cleanup - 2026-05-14

The grounded cleanup items that were explicitly waiting on the structural
Hunter-Seeker refactor have now been applied where they were behavior-preserving
or boundary-cleaning.

Terminal scoring split:

- `terminal_scoring.py` now owns generic terminal-outcome candidate scoring:
  exact context keys, latent prototype vectors, basin penalties, terminal
  candidate/action key maps, and terminal penalty diagnostics.
- `scoring.py` remains the single owner of final score assembly and score-trace
  writes; it now calls `_score_terminal_outcome_candidate(...)` and adds the
  returned terminal penalty.
- A real extraction bug was caught by tests and fixed: non-click candidates now
  initialise `target_key=None` and `cooldown_penalty=0.0` before score trace
  writing.

Outcome/progress adapter:

- Added `outcome_adapters.py` with `OutcomeAdapter`, `ArcOutcomeAdapter`, and
  `MockSymbolicOutcomeAdapter`.
- Stockfish now accepts an `outcome_adapter` and routes normal runtime
  transition quality, completed-stage quality, unfinished-run weak labels, and
  trusted-trajectory quality through that adapter.
- ARC behavior is preserved: the old `levels_completed`, level-quality scale,
  change bonus, completion bonus, and weak unfinished labels are now in
  `ArcOutcomeAdapter` instead of being hardwired into normal Stockfish runtime
  paths.
- Legacy constants and fallbacks remain for old imports/checkpoints, but new
  code should consume the adapter boundary.

Topology region graph extraction:

- `topology_regions.py` now owns the optional full region/gateway graph builder.
- `topology.py` keeps the hot reachable/frontier topology path and calls the
  region builder only when full topology and `topology_region_graph_enabled`
  are requested.
- This preserves the Sprint 5 full-topology test contract while making clear
  that the region/gateway graph is diagnostic/future-work code, not active
  runtime policy.

Architecture status after this pass:

- The main post-refactor cleanup seams called out by the older notes are now
  covered: naming hygiene, phase/lifecycle split, terminal-scoring split,
  region-graph extraction, outcome/progress boundary, `pad_grids_to_batch`
  vectorization, action enum-map caching, output placement hardening, and
  `loop_pooler_gate` logging.
- Work still intentionally not started: sleep/consolidation, Ouro-confidence
  trust redesign, ES outer-loop implementation, and broad self-model control
  ownership. Those are research/experiment steps, not refactor cleanup.

Verification:

- Focused terminal/scoring tests passed: `7 passed`, then `4 passed` after the
  final static-test update.
- Outcome/domain-neutral targeted tests passed: `7 passed`.
- Topology/model-basin/symbolic focused tests passed:
  `29 passed, 174 deselected`.
- Full active pytest passed after all post-refactor cleanup:
  `615 passed, 1 skipped, 18 subtests passed`.

## Post-Refactor Topology Trio Smoke - 2026-05-14

Command shape:

- direct project venv Python outside the sandbox so PyTorch can see CUDA;
- `--agent hunter_seeker`;
- `--games ls20 tr87 wa30`;
- `--backbone_mode ouro`;
- `--ouro_model_path models/ouro_rltt_local`;
- `--checkpoint artifacts/checkpoints/running/sprint4_encoder_reverted.pt`;
- `--load_trajs data/trajectories/trusted_topology_trio_20260513`;
- `--pretrain_iters 0`;
- `--max_steps 500`;
- `--n_runs 1`;
- `--eps 0.0`;
- `--no_replay`;
- `--train_every 0`;
- no `--enable_agent_phase_policy`.

CUDA/GPU:

- Sandbox Python still reports CUDA unavailable, so long GPU runs must use the
  approved direct `venv/bin/python` path outside the sandbox.
- CUDA preflight in the run succeeded:
  `NVIDIA GeForce RTX 5070 Ti Laptop GPU`.
- RLTT loaded on `cuda:0`.
- `nvidia-smi` during `ls20` showed the Python process using about `5458 MiB`
  GPU memory.

Artifacts:

- checkpoints:
  `artifacts/checkpoints/topology_trio_post_refactor_smoke/`;
- event/measurement dumps:
  `artifacts/reports/topology_trio_post_refactor_smoke/`;
- improved trajectory artifacts:
  `artifacts/trajectories/topology_trio_post_refactor_smoke/`.

Results:

- `ls20`:
  - `1` level completed;
  - level 1 completed at step `13`;
  - game over at step `91`;
  - current-run failure: `mechanism`;
  - selected action distribution from `action_trace`: `{1: 24, 2: 23, 3: 20, 4: 24}`;
  - `agent_phase_policy_enabled=False` on all `91/91` top-level action-trace entries.
- `tr87`:
  - `2` levels completed;
  - level 1 completed at step `37`;
  - level 2 completed at step `67`;
  - game over at step `195`;
  - current-run failure: `mechanism`;
  - selected action distribution: `{1: 96, 2: 32, 3: 31, 4: 36}`;
  - `risk_patcher_beam` selected `6` of `195` steps;
  - `agent_phase_policy_enabled=False` on all `195/195` action-trace entries.
- `wa30`:
  - `0` levels completed;
  - game over at step `200`;
  - current-run failure: `protected_terminal_starvation`;
  - selected action distribution: `{1: 62, 2: 14, 3: 57, 4: 59, 5: 8}`;
  - topology-death diagnostic: raw pool `2`, candidates `0`, both adjacent
    candidates rejected as protected avatar/exit;
  - `agent_phase_policy_enabled=False` on all `200/200` action-trace entries.

Interpretation:

- The post-refactor architecture did not break basic RLTT/topology execution:
  CUDA path works, event/measurement dumps write, checkpoints save, and all
  three games run through Hunter-Seeker scoring without exceptions.
- Compared with the earlier 180-step phase-disabled trio, this is better on
  `ls20` and `tr87`:
  - `ls20` now clears level 1 instead of zero levels;
  - `tr87` now clears two levels instead of zero.
- It is still below the older trusted/phase-assisted behavior:
  - `ls20` is not near the earlier level-3/4 expectation;
  - `tr87` dies after level 2;
  - `wa30` still fails before solving level 1.
- The action distributions are no longer the extreme one-action collapse seen
  in the earlier phase-disabled smoke, especially on `ls20` and `wa30`.
- The main remaining issue is not a refactor regression. It is still the
  domain-general learning/control problem already identified:
  observation/action-effect knowledge and route competence are insufficient
  when replay-derived phase policy is disabled.
- `wa30` specifically remains blocked by protected-terminal/ego-control
  ambiguity: the terminal postmortem sees only protected avatar-like adjacent
  candidates, so it classifies the death as protected-terminal starvation
  rather than a normal topology/hazard mechanism.

Verification after the smoke:

- Full active pytest still passed:
  `615 passed, 1 skipped, 18 subtests passed`.

Follow-up placement hardening from the smoke:

- The smoke revealed that raw `arc_agi.Arcade()` still defaults to creating a
  root `environment_files/` directory.
- Added `arc_runtime.py` and routed `train_arc.py`, `live_arc_diagnostic.py`,
  and `branch_basin_audit.py` through `make_arcade()`.
- Active ARC environment caches now stay under
  `data/arc_agi3/environment_files/`; ARC recordings go under
  `artifacts/recordings/arc_agi3/`.
- The accidental root `environment_files/` cache from the smoke was removed.
- Full active pytest passed again after this harness-path fix:
  `615 passed, 1 skipped, 18 subtests passed`.
