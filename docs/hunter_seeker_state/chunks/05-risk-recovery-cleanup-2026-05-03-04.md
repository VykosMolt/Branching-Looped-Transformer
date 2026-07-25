<!-- Source: PROJECT_STATE_HUNTER_SEEKER.md lines 6464-8031 before the 2026-05-14 split. -->
<!-- Source chunk SHA256: 1059ea3c31f9963a870e4772210c916804f1ace3f0b46e08b489fc1c262cfe36 -->

## Flow/speed pass, effective batching, and adjacent-risk patch (2026-05-03)

- User intent for this pass:
  - Make topology/search runtime flow much faster.
  - Keep the system domain/game agnostic.
  - Do not start the 18k-line Hunter-Seeker refactor yet.
  - Increase effective GPU batch size where possible.
  - After the optimization pass, enact the next patch suggested by the run evidence.
- Files changed:
  - `claude_sandbox/arc_agent_pairwise_stockfish_codex.py`
  - `claude_sandbox/arc_agent_hunter_seeker_codex.py`
  - `claude_sandbox/train_arc_codex.py`
  - `claude_sandbox/test_codex_sandbox.py`
  - `PROJECT_STATE.md`

### Implemented optimizations

1. Beam frontier batching:
   - Added `PendingSearchExpansion` / `PendingCandidatePrediction`.
   - `_run_beam_search_pass()` now expands a whole frontier through `_expand_search_frontier()` instead of encoding every beam node independently.
   - Successor CLS encoding is batched across same-shaped frontier predicted frames via `_encode_search_frames_batched()`.
   - New frontier prediction counters:
     - `search_frontier_prediction_batches`
     - `search_frontier_prediction_nodes`
     - `search_frontier_prediction_candidates`
   - These counters are included in `info` and Hunter action traces.

2. Effective batch-size increase:
   - Added `_predict_candidate_successors_frontier()`.
   - Same-shaped beam nodes now batch current-frame feature extraction and next-frame prediction:
     - source frames are stacked;
     - encoder `encode_features()` runs once per same-shaped frontier group;
     - features are `repeat_interleave`d per candidate;
     - `NextFramePredictor` runs one larger candidate batch.
   - This is domain-neutral:
     - grouping is by frame shape;
     - dense conversion still goes through the observation adapter boundary;
     - no ARC/game/color/action assumptions are introduced.

3. Adaptive beam/depth cleanup:
   - Adaptive beam width no longer expands to `beam_width * 2`.
   - Runtime width now scales between `beam_width * adaptive_beam_min_fraction` and `beam_width * adaptive_beam_max_multiplier`.
   - Defaults:
     - `adaptive_beam_min_fraction = 0.50`
     - `adaptive_beam_max_multiplier = 1.00`
     - `adaptive_depth_score_margin = 0.45`
   - `beam_search_action()` can stop iterative deepening early when the depth-1 root margin is already decisive.
   - Added diagnostics:
     - `adaptive_beam_cap`
     - `adaptive_depth_stop_reason`

4. Stricter model-basin scheduling for Hunter:
   - Hunter now widens `model_basin_diag_every` to at least `20`.
   - Pressure windows/latches still exist and can still force deeper diagnostics.
   - Pairwise defaults remain test-compatible.

5. ObjectTable lookup map:
   - Added `ObjectTable._track_lookup_cache`.
   - `_find_track_for_obj()` now memoizes scene-object-to-track lookup by:
     - object id;
     - color;
     - quantized centroid;
     - area.
   - Cache clears on reset, scene update, and track creation.
   - This avoids broad stale reuse across predicted scenes while cutting repeated object-track scans.

6. Cheaper planning copies for predicted symbolic summaries:
   - `ObjectRecord.planning_copy(share_weights=False)`.
   - `TrackRecord.planning_copy(share_belief_weights=False)`.
   - `ObjectTable.planning_copy(share_belief_weights=False)`.
   - `_symbolic_transition_summary()` now uses `current_table.planning_copy(share_belief_weights=True)` because predicted-scoring copies update positions/centroids, not belief weights.
   - The default still deep-copies weights, so existing tests and mutation-safe callers preserve the old aliasing contract.

7. Directory-level trajectory CLS cache:
   - Added `_traj_cls_bundle.pt` support in `load_solved_trajectories()`.
   - The loader still accepts existing per-file `*_traj_cls.pt` caches.
   - When per-file caches are loaded or generated, the bundle is updated.
   - The bundle validates each entry by trajectory basename plus `{n, size, mtime_ns}`.
   - Verified during runs:
     - first optimized run wrote `claude_sandbox/trusted_plus_expanded/_traj_cls_bundle.pt`;
     - later GPU runs printed `Loading bundled CLS cache` and loaded bundled CLS for all 31 trajectories.

8. Reduced GPU/CPU tensor churn:
   - `_frames_to_dense_tensor()` now keeps low-rank torch frame tensors as tensors instead of round-tripping through CPU numpy.
   - Higher-rank/raw/video tensors still use adapter conversion.

9. Trace dump shrink:
   - Hunter action trace cap reduced from `512` to `256` entries.
   - Top-candidate score components now use `detail="candidate"`:
     - scoreboard/risk fields only;
     - no large phase templates, terminal context payloads, or broad diagnostic dicts per candidate.
   - Chosen action score components use `detail="lean"`:
     - still includes hazard, local contact, engram, model-basin, risk-patcher, phase, and observation-effect terms needed to inspect decisions.
   - The old `_compact_score_components()` default remains the fuller compactor for direct tests and non-routine diagnostics.
   - Size impact:
     - before candidate lean mode: `4.65 MiB` for a 140-step dump;
     - after candidate lean mode: `0.762 MiB` for a 40-step dump, about `0.019 MiB/step`.

10. Matplotlib/startup friction:
   - `train_arc_codex.py` now sets:
     - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
     - `MPLCONFIGDIR=/tmp/matplotlib`
   - This avoids sandbox/home-directory matplotlib cache noise from transitive imports.

### Next behavior patch enacted from run evidence

- Run evidence:
  - The long 340-step smoke reached level 2 and died at step 283 again.
  - This run was not the final trusted GPU verification because its command shape did not show a Python CUDA process in `nvidia-smi`, but the trace was still useful.
  - Failure type was now `mechanism`, not topology.
  - Death postmortem:
    - `last_action = 2`
    - accepted adjacent candidate color `2`, object id `9`
    - belief hazard was low (`0.0833`) but uncertainty/risk conflict was present
    - rejected adjacent objects were protected/avatar-like
  - Local contact scoring at fatal step mostly scored the target/protected-overlap object, not the risky adjacent neighbor in the movement direction.
- Patch:
  - Added bounded local adjacent-risk scoring inside `_local_contact_hazard_components()`.
  - New score components:
    - `local_contact_adjacent_risk_penalty`
    - `local_contact_adjacent_risk`
    - `local_contact_adjacent_risk_hazard`
    - `local_contact_adjacent_risk_wall`
    - `local_contact_adjacent_risk_unknownness`
    - `local_contact_adjacent_risk_avatar`
    - `local_contact_adjacent_risk_alignment`
    - `local_contact_adjacent_risk_reward_gate`
    - `local_contact_adjacent_risk_conflict`
    - `local_contact_adjacent_risk_active`
    - `local_contact_adjacent_risk_obj_id`
    - `local_contact_adjacent_risk_track_id`
  - Penalty is bounded:
    - `local_contact_adjacent_risk_penalty >= -0.18`
  - It is domain/game agnostic:
    - uses only local scene adjacency, directional geometry, object-table beliefs, unknownness, and reward relief;
    - no hardcoded game id, color id, route, or action blacklist.
  - It is wired into:
    - `safety_penalty`;
    - candidate score adjustment;
    - risk patcher hazard-risk aggregation;
    - live-recovery risk gate;
    - lean/candidate score-component traces.
- Test added:
  - `test_local_contact_adjacent_risk_penalizes_directional_neighbor()`

### Verification

- Py-compile:
  - `arc_agent_pairwise_stockfish_codex.py`
  - `arc_agent_hunter_seeker_codex.py`
  - `train_arc_codex.py`
  - passed.
- Focused tests:
  - `claude_sandbox/test_codex_sandbox.py -k "planning_copy or model_basin or directional_topology or observation"`
    - `25 passed, 126 deselected`
  - `claude_sandbox/test_causal_correctness.py -k "ObjectTable or topology or engram or beam_search or observation"`
    - `16 passed, 112 deselected`
  - local-contact/action-trace tests after adjacent-risk patch:
    - `13 passed, 139 deselected`
- Full suite:
  - after all patches:
    - `389 passed, 1 skipped`
- GPU verification:
  - Direct venv command shape was required for CUDA visibility.
  - `nvidia-smi` verified `/home/moloch/ouro_project/venv/bin/python` as a compute process at about `340 MiB` CUDA memory.
  - 120-step frontier-batch smoke:
    - dump: `claude_sandbox/perf_event_dumps_frontier_batch_wa30_gpu`
    - completed 120 steps
    - frontier counters:
      - `search_frontier_prediction_batches` sum `134`
      - `search_frontier_prediction_nodes` sum `163`
      - `search_frontier_prediction_candidates` sum `815`
  - 140-step adjacent-risk GPU smoke:
    - dump: `claude_sandbox/perf_event_dumps_adjacent_risk_wa30_gpu`
    - completed 140 steps and level 1
    - frontier counters:
      - batches sum `140`
      - nodes sum `140`
      - candidates sum `700`
  - 40-step final lean-trace CUDA smoke:
    - dump: `claude_sandbox/perf_event_dumps_trace_lean_wa30_gpu`
    - completed 40 steps
    - CUDA verified, then no Python compute process remained in final `nvidia-smi`
    - dump size `0.762 MiB`
    - frontier counters:
      - `search_frontier_prediction_batches` sum `41`
      - `search_frontier_prediction_nodes` sum `43`
      - `search_frontier_prediction_candidates` sum `215`

### Current state and next likely issues

- Current topology runtime state:
  - root scoring still uses cheap topology, symbolic summaries, safety, engram, and phase/risk diagnostics;
  - non-root speculative beam/model-basin scoring stays lightweight;
  - full topology remains available for tests/future diagnostics but is not the active policy bottleneck.
- Current search runtime state:
  - root/depth frontier expansion now batches next-frame prediction and successor CLS encoding across same-shaped beam nodes;
  - current 40/120/140-step wa30 smokes mostly had one active frontier node early, so the batching counters are modest there;
  - the benefit should grow when beam depth/frontier width genuinely branches.
- Current failure diagnosis:
  - level 3 still needs a longer post-patch GPU probe;
  - the old death trace indicates the problem is now more mechanism/policy-learning than topology representation;
  - model-basin repeatedly reported survivor-collapse/non-viable basins (`model_basin_risk=0.65`, `viable_fraction=0`, `collapse_fraction=1`) with low path trust, but because all candidates were risky this did not by itself create a better alternative.
- Recommended next probes:
  1. Run a longer GPU probe past step 283 with the adjacent-risk patch.
  2. Inspect whether `local_contact_adjacent_risk_*` activates in the level-3 death window.
  3. If death remains, inspect whether all-candidates-risky periods should trigger a bounded recovery/search-mode change rather than only score penalties.
  4. Consider gradient accumulation or larger `--batch_size` for training separately from search batching; the search batching patch increases effective inference batch size, not learner batch size.

## All-Candidates-Risky Recovery Selector (2026-05-03)

### Why this patch exists

- The prior long wa30 trace showed a late level-3 failure where the ordinary risk patcher often reported:
  - `risk_patcher_reason = all_candidates_risky` or `no_lower_risk_candidate`;
  - model-basin survivor collapse:
    - `model_basin_risk ~= 0.65`;
    - `model_basin_viable_fraction = 0`;
    - `model_basin_collapse_fraction = 1`;
    - low `model_basin_min_path_trust`.
- The existing risk patcher could reroute when one candidate was clearly lower-risk, but it had no second-stage choice rule for flat-risk/all-bad candidate sets.
- This is not a topology-representation fix and not a game-specific fix. It is a bounded least-trapped selector over diagnostics the agent already computes.

### Code changes

File changed:

- `claude_sandbox/arc_agent_hunter_seeker_codex.py`

New helper methods:

- `_all_risky_recovery_candidate_terms()`
- `_all_risky_recovery_select_candidate()`
- `_all_risky_recovery_write_candidate_diag()`

Behavior:

- The selector only runs inside `_risk_patcher_select_from_trace()` when:
  - the root candidate set is already classified as all risky;
  - the ordinary lower-risk reroute found no safer candidate;
  - there is generic risk/basin evidence, such as high risk score, basin risk, collapse, or nonviable low-trust basin.
- It does not blacklist actions, colors, games, levels, or routes.
- It scores candidates using generic diagnostics only:
  - existing risk-patcher score and primary/hazard/terminal risk;
  - `model_basin_root_risk`, `model_basin_risk`, `model_basin_viable_fraction`, `model_basin_survival_fraction`, `model_basin_collapse_fraction`, `model_basin_min_path_trust`, `model_basin_best_survivor_score`, `model_basin_best_viable_score`;
  - live/observation recovery support when present;
  - positive pressure, so optimism cannot silently dominate the least-trapped choice.
- Score budget is bounded:
  - starts from the existing risk-patcher budget;
  - can widen only to at most `0.42` under high risk/collapse evidence.
- Candidate eligibility remains conservative:
  - no candidate can be selected if it is too far below the chosen score budget;
  - a candidate with direct hard risk cannot replace a non-direct-risk chosen action;
  - terminal and basin risk are not allowed to grow materially unless the chosen candidate is itself in a nonviable basin.

New score components:

- `all_risky_recovery_active`
- `all_risky_recovery_selected`
- `all_risky_recovery_reason`
- `all_risky_recovery_candidate_count`
- `all_risky_recovery_score_budget`
- `all_risky_recovery_score_gap`
- `all_risky_recovery_margin`
- `all_risky_recovery_selected_margin`
- `all_risky_recovery_selection_score`
- `all_risky_recovery_risk_score`
- `all_risky_recovery_primary_risk`
- `all_risky_recovery_hazard_risk`
- `all_risky_recovery_terminal_risk`
- `all_risky_recovery_basin_risk`
- `all_risky_recovery_viable_fraction`
- `all_risky_recovery_survival_fraction`
- `all_risky_recovery_collapse_fraction`
- `all_risky_recovery_min_path_trust`
- `all_risky_recovery_survivor_score`
- `all_risky_recovery_live_support`
- `all_risky_recovery_positive_pressure`
- `all_risky_recovery_collapse_pressure`
- `all_risky_recovery_nonviable_basin`

Selection methods added:

- `all_risky_recovery_beam`
- `all_risky_recovery_random`

Compaction:

- All new terms are included in score-component compaction where relevant:
  - lean chosen traces;
  - candidate traces;
  - full/default score compaction.

### Tests

File changed:

- `claude_sandbox/test_codex_sandbox.py`

New/updated tests:

- `test_all_risky_recovery_uses_basin_survivor_signal_when_risk_is_flat()`
- `test_all_risky_recovery_stays_inactive_without_broad_risk()`
- existing phase-template/score-compaction test now asserts `all_risky_recovery_*` trace terms are preserved.

Verification run:

- Py-compile passed:
  - `/home/moloch/ouro_project/venv/bin/python -m py_compile claude_sandbox/arc_agent_hunter_seeker_codex.py claude_sandbox/test_codex_sandbox.py`
- Focused tests:
  - `/home/moloch/ouro_project/venv/bin/python -m pytest claude_sandbox/test_codex_sandbox.py -q -k "all_risky_recovery or risk_patcher or compact_phase_template_keeps_state_fields"`
  - result: `10 passed, 144 deselected`
- Full focused sandbox file:
  - `/home/moloch/ouro_project/venv/bin/python -m pytest claude_sandbox/test_codex_sandbox.py -q`
  - result: `154 passed`
- Full active sandbox suite:
  - `/home/moloch/ouro_project/venv/bin/python -m pytest -q claude_sandbox`
  - result: `391 passed, 1 skipped`
- Root-level `pytest -q` is currently not a meaningful active-suite command:
  - it collects archived legacy test files under `archive/`;
  - failures are import/module-name collisions and missing legacy module imports, not this patch.

### CUDA / run verification

- CUDA availability check:
  - `torch.cuda.is_available() = True`
  - `torch.version.cuda = 12.8`
  - `torch.cuda.device_count() = 1`
- Deliberate CUDA allocation:
  - `/home/moloch/ouro_project/venv/bin/python -c "... torch.empty(..., device='cuda') ..."`
  - `nvidia-smi` showed Python PID `343150` as a compute process using about `1178 MiB`.
- Important command-shape correction:
  - shell-wrapped env-prefix commands can complete but may not be CUDA-confirmed in `nvidia-smi`;
  - direct venv command prefix is the reliable GPU protocol:
    - `/home/moloch/ouro_project/venv/bin/python -m claude_sandbox.train_arc_codex ...`
- Non-CUDA-confirmed functional smoke:
  - command used shell/env prefix;
  - dump: `claude_sandbox/perf_event_dumps_all_risky_recovery_gpu/wa30/run_1.json`
  - completed `160` steps and level 1;
  - score `0.7169422222222223`;
  - action trace:
    - `selection_method = beam_search` for all 160 steps;
    - `all_risky_recovery_active_steps = 0`;
    - risk reasons: `no_high_risk_chosen=145`, `trusted_exact_phase_guidance=15`.
  - Treat this as a functional smoke only, not as the trusted CUDA run.
- CUDA-confirmed smoke:
  - direct venv command prefix;
  - `nvidia-smi` showed Python PID `343255` as a compute process, around `1000 MiB`;
  - dump: `claude_sandbox/perf_event_dumps_all_risky_recovery_cuda_check/wa30/run_1.json`
  - completed `40` steps;
  - action trace:
    - `selection_method = beam_search` for all 40 steps;
    - `all_risky_recovery_active_steps = 0`;
    - risk reasons: `no_high_risk_chosen=39`, `trusted_exact_phase_guidance=1`.

### Current interpretation

- The patch is implemented and unit-tested.
- Short smokes did not reach an all-candidates-risky / nonviable-basin window, so they correctly did not exercise the selector.
- The real behavior test is still a longer wa30 GPU probe that reaches the old level-3 failure window around step `276-283`.
- If the selector fires there and survives, the next question is whether it finds genuine recovery or only delays failure.
- If it does not fire, inspect:
  - whether all-candidates-risky classification is absent after adjacent-risk scoring;
  - whether model-basin pressure terms are missing from the selected root;
  - whether the useful recovery action is absent from the candidate set rather than merely scored badly.

### Future cleanup: active test tree

- Move active tests out of `claude_sandbox/` into a dedicated test tree after the current topology/observation work is stable.
- Suggested structure:
  - `tests/unit/`
  - `tests/integration/`
  - `tests/reports/`
  - `tests/fixtures/`
- Add pytest configuration so active test runs collect `tests/` and/or explicit `claude_sandbox` tests, while ignoring `archive/`.
- This would fix the current root-level collection failure and make `claude_sandbox` less cluttered without changing behavior.

## Phase-Template Continuity / wa30 Level-3 Ceiling (2026-05-03)

### Correction to the step-283 diagnosis

- We had already concluded, correctly, that step `283` is checkmate-like:
  - by the terminal frame, no root action is genuinely safe enough;
  - patching the fatal-step selector is treating the symptom.
- The new inspection confirms the divergence begins earlier, around level 3 phase `65` / live step `249`.
- Before the continuity patch, the agent treated the next trusted action as exact even though its stored pre-state did not match the live pre-state.
- After the continuity patch, stale exact trust is gone, but the agent still dies because it has no reliable policy after the trusted source discontinuity.

### Trusted trajectory discontinuity found

File inspected:

- `claude_sandbox/trusted_plus_expanded/wa30_run0_traj.npz`

Important wa30 level-3 break:

- phase `64` action `1`:
  - before: `64x64:4096:4e693552:e88f6e9fe4e092a5`
  - after: `64x64:4096:5f9f94c6:d4bbd91a715b795d`
- phase `65` action `4`:
  - before: `64x64:4096:78e03506:7467e0df5ee51b75`
  - after: `64x64:4096:21971a1c:50126f917330e1ca`
- The phase-64 after-state does not appear later as a before-state in the trusted file.
- The phase-65 before-state is a disconnected segment start, not the next state reachable from phase 64.
- Cell gap at this boundary: `234`.

Other continuity gaps found in the same trusted file:

- level `3`, phase `65`, action `4`, gap `234`
- level `3`, phase `165`, action `4`, gap `272`
- level `4`, phase `51`, action `1`, gap `225`
- level `5`, phase `125`, action `2`, gap `320`
- level `5`, phase `250`, action `4`, gap `320`
- level `5`, phase `375`, action `4`, gap `320`
- level `7`, phase `125`, action `4`, gap `176`
- level `9`, phase `70`, action `4`, gap `304`

### Code changes now in `claude_sandbox`

Changed file:

- `claude_sandbox/arc_agent_hunter_seeker_codex.py`

Continuity metadata is now built for phase action and target templates:

- `source_file`
- segmented `source`, e.g. `wa30_run0_traj.npz#seg0`, `#seg1`
- `continuity_segment`
- `continuity_break_before`
- `continuity_gap_cell_count`
- `continuity_prev_after_signature`
- `continuity_prev_source_index`
- `continuity_source_index`

Exact phase trust now respects continuity:

- `_phase_action_template_is_exact_trusted(..., state_aligned=None)` rejects templates with `continuity_break_before=True` unless an explicit live pre-state match is supplied.
- `_phase_action_template_pre_state_matches()` compares the template before-state signature against the current live frame.
- Phase bonus for a continuity-break template is capped to a weak hint (`<= 0.08 * confidence`) unless the live pre-state matches.
- Continuity-break mismatch diagnostics are included in score components:
  - `phase_state_template_continuity_break_before`
  - `phase_state_template_continuity_segment`
  - `phase_state_template_continuity_gap_cell_count`
  - `phase_state_continuity_pre_state_aligned`
  - `phase_state_continuity_break_penalty`
- These terms are preserved in compact/lean/candidate score-component traces.

Immediate source-continuity recovery is now active:

- `_phase_action_continuity_gap_diag()` detects when the active trusted source has no current continuation and the current phase contains a discontinuous segment start.
- `_maybe_activate_phase_continuity_recovery()` now:
  - retires the active/preferred branch source;
  - activates recovery and reseed immediately;
  - clears the unguided plateau counter;
  - emits `phase_continuity_gap` in the action trace.

Disconnected segment quarantine is now active:

- `_phase_action_incompatible_sources` tracks per-run/per-level demonstration segments whose start state mismatched the live state.
- `_phase_action_templates_for()` filters incompatible sources from:
  - exact options;
  - future options;
  - recovery options;
  - reseed options;
  - terminal suffix options;
  - branch selection.
- This is not an action/color/game blacklist. It is source-trust quarantine keyed by `(game_id, level, source_segment)`.
- The checkpoint state now preserves `phase_action_incompatible_sources`.

No hardcoding was added:

- no game-id rule;
- no color-id rule;
- no action blacklist;
- no `wa30`-specific branch;
- no ARC-specific terminal constant.

### Tests

Changed file:

- `claude_sandbox/test_codex_sandbox.py`

New focused tests:

- `test_phase_action_templates_mark_trajectory_continuity_breaks`
- `test_phase_action_continuity_break_is_weak_until_pre_state_matches`
- `test_phase_continuity_gap_activates_recovery_when_active_source_ends`
- `test_phase_continuity_gap_does_not_recover_when_segment_state_matches`

Verification:

- Py-compile:
  - `/home/moloch/ouro_project/venv/bin/python -m py_compile claude_sandbox/arc_agent_hunter_seeker_codex.py claude_sandbox/test_codex_sandbox.py`
  - passed
- Focused continuity tests:
  - `/home/moloch/ouro_project/venv/bin/python -m pytest -q claude_sandbox/test_codex_sandbox.py -k "continuity_gap or continuity_break or phase_action_templates_record_expected_no_change"`
  - result: `5 passed, 153 deselected`
- Full active sandbox suite:
  - `/home/moloch/ouro_project/venv/bin/python -m pytest -q claude_sandbox`
  - result after this patch: `395 passed, 1 skipped`

### CUDA-confirmed runs

All long probes below used direct venv Python and were confirmed in `nvidia-smi` as GPU compute processes.

1. Pre-continuity-fix long probe:

- Dump:
  - `claude_sandbox/perf_event_dumps_all_risky_recovery_long_wa30_gpu/wa30/run_1.json`
- Result:
  - level 1 completed at step `125`;
  - level 2 completed at step `183`;
  - game over at step `283`;
  - `FAILURE_TYPE=mechanism`;
  - death: `color 2 hazard↑`.
- Diagnosis:
  - step `249` was already a no-positive-candidate basin;
  - exact trusted phase action was mismatched;
  - the real source was the trusted trajectory discontinuity at phase `65`.

2. Continuity metadata / exact-trust gate probe:

- Dump:
  - `claude_sandbox/perf_event_dumps_phase_continuity_wa30_gpu/wa30/run_1.json`
- Result:
  - still died at step `283`;
  - `FAILURE_TYPE=mechanism`.
- Important behavior:
  - at step `249`, the disconnected phase-65 action-4 template was no longer exact;
  - `phase_action_bonus=0.08`;
  - `phase_state_pre_alignment_penalty=-0.48`;
  - `continuity_break_before=True`;
  - `continuity_gap_cell_count=234`;
  - no exact-veto recovery fired.
- Remaining problem:
  - the agent entered unguided fallback too slowly and still fell into the terminal basin.

3. Immediate continuity-gap recovery probe:

- Dump:
  - `claude_sandbox/perf_event_dumps_phase_continuity_recovery_wa30_gpu/wa30/run_1.json`
- Result:
  - still died at step `283`;
  - `FAILURE_TYPE=mechanism`.
- Important behavior:
  - step `249` activated `phase_continuity_gap`;
  - recovery and reseed activated immediately;
  - action sequence changed materially after the gap;
  - phase progress reached about `80`.
- Remaining problem:
  - recovery still used future templates from the disconnected segment, which are not valid continuations from the live state.

4. Disconnected-segment quarantine probe:

- Dump:
  - `claude_sandbox/perf_event_dumps_phase_continuity_quarantine_wa30_gpu/wa30/run_1.json`
- Result:
  - still died at step `283`;
  - `FAILURE_TYPE=mechanism`.
- Important behavior:
  - step `249` activated continuity-gap recovery;
  - `incompatible_source_count=1`;
  - phase bonuses from the disconnected segment were removed;
  - after the gap, `phase_action_bonus=0.0` for the chosen candidates;
  - recovery became genuinely live/topology/model driven rather than disconnected replay driven.
- Remaining problem:
  - the agent still cannot find a productive continuation from the live phase-64 after-state.
  - phase progress mostly stalls around `65-66`;
  - live recovery eventually deactivates;
  - terminal death still occurs at step `283`.

### Current interpretation

- The original late step-283 framing was wrong as a patch target.
- The trusted phase stream bug was real and is now guarded:
  - stale exact trust is fixed;
  - discontinuous segments are detected;
  - unmatched segment starts are quarantined.
- This did not break wa30 level 3.
- That means the remaining ceiling is probably not a simple topology scoring bug.
- The current failure is better described as:
  - trusted replay has a missing continuous demonstration after level-3 phase `64`;
  - after that gap, the agent lacks a learned action-effect policy strong enough to recover from the live state.
- Topology is still doing useful safety/diagnostic work, but the agent is no longer failing because topology cannot see the board. It is failing because, once replay is invalid, the live policy cannot plan the needed continuation.

### Next work / cleanup implications

- Do not keep tuning the step-283 selector unless a future trace shows a genuinely safer candidate was present and rejected.
- Do not add wa30-specific action sequences or route rules.
- Strong next evidence options:
  1. collect or reconstruct a continuous trusted trajectory through wa30 level 3 after phase `64`;
  2. run a trajectory-continuity audit over all trusted files and report segment counts/gaps before trusting replay;
  3. move to the observation-learning/action-effect path so the agent can learn from watching transitions instead of needing exact replay after a source gap.
- After wa30 level 3 is eventually broken, audit which late recovery patches were symptom treatments:
  - all-candidates-risky recovery selector;
  - repeated basin/recovery pressure terms;
  - exact-veto recovery windows;
  - some phase-stall/reseed machinery.
- Keep the continuity/source-quarantine patch regardless; it is a permanent data-contract fix, not a wa30 workaround.

## Patch Necessity Audit Before Test Cleanup (2026-05-03)

### Method

- Audited code surface in `claude_sandbox/arc_agent_hunter_seeker_codex.py`.
- Parsed recent CUDA event dumps for actual activation/effect evidence:
  - `perf_event_dumps_topology_contact4_wa30_gpu`
  - `perf_event_dumps_phase_state_resync_wa30_gpu`
  - `perf_event_dumps_effect_diag_wa30_gpu`
  - `perf_event_dumps_observation_long_wa30_gpu`
  - `perf_event_dumps_obs_terminal_risk_2run_wa30_gpu`
  - `perf_event_dumps_pressure_basin_2run_wa30_gpu2`
  - `perf_event_dumps_all_risky_recovery_long_wa30_gpu`
  - `perf_event_dumps_phase_continuity_wa30_gpu`
  - `perf_event_dumps_phase_continuity_recovery_wa30_gpu`
  - `perf_event_dumps_phase_continuity_quarantine_wa30_gpu`
- Classification standard:
  - `Keep`: fixes a real data/contract/safety bug or remains broadly useful.
  - `Keep but simplify later`: valid behavior, but implementation has grown too large or overlapping.
  - `Diagnostic-first`: useful to understand failures, but should not carry strong behavioral authority.
  - `Bloat candidate`: active or complex, but evidence says it treated symptoms or did not improve outcome.

### Keep: permanent structural fixes

1. Continuity-aware phase templates and source quarantine

- Evidence:
  - trusted wa30 file has real discontinuities;
  - stale exact trust previously drove phase-65 guidance from an impossible pre-state;
  - patch correctly removed exact trust and emitted continuity diagnostics.
- Classification: `Keep`.
- Reason:
  - this is a data-contract fix, not a wa30 workaround.
  - It prevents any solved trajectory with discontinuous segments from being treated as a single continuous behavioral source.

2. State-aligned exact phase trust

- Evidence:
  - pre-state mismatch was the original control-flow gap around step `249`;
  - exact guidance must only be exact when the stored before-state matches the live before-state.
- Classification: `Keep`.
- Simplification note:
  - keep the pre-state gate and diagnostics;
  - later review whether the separate exact-veto recovery window is still needed.

3. Hazard-aware safety scoring

- Evidence:
  - earlier GPU probe showed `reachable_hazard_delta > 0` with `safety_penalty == 0`;
  - patched probe showed `hazard_reachable_penalty < 0` and safety wired correctly.
- Classification: `Keep`.
- Reason:
  - this fixed a real safety-score bug.
  - Penalty remains bounded and appears in `score_components`.

4. Core topology/contact instrumentation and protected-avatar poisoning fixes

- Evidence:
  - earlier topology runs fixed avatar/exit poisoning and protected terminal evidence mistakes.
  - Current wa30 failure is not "topology cannot see the board"; topology is producing useful local/contact/hazard diagnostics.
- Classification: `Keep`.
- Review note:
  - `local_contact_source=avatar_overlap_protected_context` at fatal steps still deserves later audit because over-protection can mask risk, but the general protected-context mechanism is not bloat.

5. Active score-component instrumentation

- Evidence:
  - the continuity diagnosis, hazard bug, and post-veto analysis were only possible because score components preserved root/candidate terms.
- Classification: `Keep`.
- Reason:
  - diagnostics are part of the science here.
  - Do not remove compact trace terms until a replacement report surface exists.

### Keep but simplify later

1. Risk patcher core

- Evidence:
  - active in many runs, e.g. `topology_contact4_wa30_gpu` had `149` risk-patcher selections over `600` traced steps.
  - It catches direct or relatively lower-risk alternatives.
- Classification: `Keep but simplify later`.
- Reason:
  - the core idea is valid: prevent obviously risky scored choices from winning through positive pressure.
  - The implementation is too entangled with all-risky recovery, observation risk, basin risk, terminal risk, and score budgets.
- Future cleanup:
  - keep one generic "risk arbitration" selector;
  - remove special late-wa30 branches from it once observation/action-effect learning can supply better candidates.

2. Engram recall and observation-effect recall

- Evidence:
  - engram/observation signals fire in traces, but in wa30 they usually remain weak or diagnostic:
    - observation negative support exists before death;
    - `obs_effect_terminal_risk` often remains `0` because similarity is not high enough;
    - engram bias is bounded and sometimes suppresses bad immediate actions.
- Classification: `Keep but simplify later`.
- Reason:
  - these are aligned with the architecture's memory direction.
  - The behavior is intentionally conservative and not the current level-3 fix.
- Future cleanup:
  - keep support aggregation and conflict diagnostics;
  - revisit overlapping terminal/engram/observation memory stores during the eventual Hunter Seeker refactor.

3. Model-basin sampler

- Evidence:
  - late wa30 traces repeatedly show `model_basin_risk=0.65`, `viable_fraction=0`, `collapse_fraction=1`.
  - However the same "nonviable/collapsed" diagnostic appears early in otherwise successful play, e.g. around step `41`, so it is not reliable enough as a hard behavioral guard.
- Classification: `Diagnostic-first`.
- Reason:
  - useful as a basin warning and trace explainer;
  - not strong enough to decide policy alone.
- Future cleanup:
  - keep the sampler as a measurement/diagnostic path;
  - avoid adding more behavioral weight to it until its false-positive rate is calibrated across games.

4. Phase recovery / reseed / escalation scaffold

- Evidence:
  - after continuity quarantine, phase bonuses from the bad segment were removed and recovery became genuinely live-driven.
  - It still failed, but the scaffold gave a controlled way to stop trusting stale replay.
- Classification: `Keep but simplify later`.
- Reason:
  - some recovery mode is necessary when exact replay falls out of alignment;
  - the current implementation has accumulated overlapping modes.
- Future cleanup:
  - unify recovery, reseed, escalation, and live recovery into one recovery-state machine.

### Bloat candidates / likely symptom patches

1. All-candidates-risky recovery selector

- Evidence:
  - active in the old long run (`12` selected steps) and later continuity runs (`5-22` selected steps depending on patch state).
  - It changed late actions but did not escape wa30 level 3.
  - It often operates when every candidate is already in a nonviable/collapsed model basin.
- Classification: `Bloat candidate`.
- Current recommendation:
  - do not extend it further.
  - After the next stable behavior source exists, remove or fold it into a much smaller generic risk-arbitration rule.

2. Live-recovery selector as a separate selector

- Evidence:
  - selected only a few steps in the long probes (`1-6` typical), changed actions, did not solve.
  - Its signal overlaps with observation-effect recall, risk patcher, and phase recovery.
- Classification: `Bloat candidate`.
- Current recommendation:
  - keep live effect statistics for diagnostics/training;
  - later fold the separate selector into the unified recovery-state machine or remove it.

3. Exact-veto recovery window

- Evidence:
  - it was useful before the continuity bug was understood;
  - continuity-aware trust now prevents the exact mismatched template from being exact in the first place.
- Classification: `Bloat candidate`.
- Current recommendation:
  - keep pre-state mismatch diagnostics and penalties;
  - later test whether the explicit exact-veto recovery window can be deleted without regression.

4. Repeated pressure-window/model-basin latch tuning

- Evidence:
  - helped expose collapse earlier, but did not solve;
  - risks turning a diagnostic into a behavior driver.
- Classification: `Bloat candidate`.
- Current recommendation:
  - keep cheap/cached diagnostics;
  - avoid further pressure-window behavior patches.

### Current architectural conclusion

- The good patches are the ones that enforce truthful contracts:
  - state alignment;
  - trajectory continuity;
  - bounded hazard safety;
  - explicit score-component diagnostics.
- The suspicious patches are the ones that try to recover from an unrecoverable candidate set without adding new competence:
  - all-risky selector;
  - standalone live-recovery selector;
  - exact-veto recovery window;
  - repeated basin pressure behavior.
- The remaining wa30 level-3 ceiling is not evidence that topology needs more patching.
- The remaining ceiling is evidence that, after replay becomes invalid, the agent needs better action-effect learning or a continuous demonstration source.

### Cleanup order after this audit

1. Do the test tree cleanup first; make root test collection sane.
2. Then add feature flags or a small ablation harness for bloat candidates before deleting behavior:
   - `enable_all_risky_recovery_selector`
   - `enable_live_recovery_selector`
   - `enable_exact_veto_recovery_window`
   - `enable_model_basin_behavioral_pressure`
3. Run a small mixed regression before deletion:
   - `wa30` long probe;
   - `ls20 tr87 wa30` mixed smoke.
4. Only remove bloat candidates after ablation shows no regression.

## Test Tree Cleanup (2026-05-03)

### What changed

- Active tests were moved out of `claude_sandbox/` into a dedicated root `tests/` tree:
  - `tests/unit/` for component and behavior tests;
  - `tests/integration/` for cross-module integration tests;
  - `tests/reports/` for diagnostic/report tooling tests.
- `claude_sandbox/` now holds runtime/source code and supporting data, not root-level test files.
- Added root `pytest.ini`:
  - `testpaths = tests`;
  - collects `test_*.py` and `*_test.py`;
  - ignores archived/data/runtime directories such as `archive`, `runs`, `solved_sequences`, `trusted_trajs`, and `environment_files`.

### Verification

- Pytest collection from repo root:
  - `396 tests collected`.
- Core compile check:
  - `claude_sandbox/arc_agent_hunter_seeker_codex.py`;
  - `claude_sandbox/train_arc_codex.py`;
  - `claude_sandbox/self_model.py`;
  - `claude_sandbox/observation_adapters_codex.py`;
  - `claude_sandbox/action_adapters_codex.py`.
- Full root suite:
  - `395 passed, 1 skipped in 11.56s`.

### Current cleanup state

- Test organization is now sane enough for normal root-level `pytest`.
- No behavior patches were removed during this cleanup.
- The patch-necessity audit above is the basis for the next ablation/removal pass after wa30 behavior is measured again.

## Self-Model Wiring, New-Encoder wa30 Probe, and Ladder Seeding (2026-05-03)

### Latest wa30 new-encoder validation

- GPU command used direct venv Python and `nvidia-smi` confirmed CUDA residency.
- Seed checkpoint:
  - `claude_sandbox/checkpoints_encoder_retrain/encoder_anchor_candidate_20260503.pt`.
- Validation output:
  - `claude_sandbox/checkpoints_wa30_new_encoder_protected_evidence2_3run_gpu/`.
  - `claude_sandbox/perf_event_dumps_wa30_new_encoder_protected_evidence2_3run_gpu/`.
- Run results:
  - run 1: `levels_completed=2`, died at step `283`, `failure_counts_current_run={"mechanism": 1}`;
  - run 2: `levels_completed=1`, died at step `195`, `failure_counts_current_run={"mechanism": 1}`;
  - run 3: `levels_completed=0`, died at step `200`, `failure_counts_current_run={"mechanism": 1}`.
- Protected-terminal evidence state:
  - run 2 now uses `source=adjacent_protected_terminal_override`;
  - `terminal_protected_evidence_override=True`;
  - `terminal_candidate_starvation=False`;
  - `protected_terminal_starvation=False`;
  - `timeout_like_terminal=False`;
  - `protected_terminal_context=True`.
- Interpretation:
  - the original protected-terminal starvation/instrumentation gap is fixed;
  - wa30 still fails because the chosen actions do not receive enough trustworthy terminal/mechanism risk before the terminal transition;
  - this is no longer a justification for another wa30/topology-specific patch.

### Permanent code changes landed in this pass

1. Phase-template and topology cleanup from the prior run was kept:
   - weak/current-phase hints now require pre-state alignment;
   - underpowered weak phase transitions block phase progress;
   - underpowered continuity-bridge sources are marked incompatible;
   - chosen action traces now expose root `score_components` as well as `chosen_score_components`.
2. Protected-terminal postmortem evidence was fixed:
   - protected terminal context disables the late timeout-like evidence skip;
   - when all adjacent candidates were filtered as protected, the postmortem path now falls back to weak evidence from raw protected adjacent objects instead of writing no hazard evidence.
3. Self-model wiring was expanded:
   - `SELF_EVAL_SUMMARY_NAMES` now includes:
     - `evaluator_pair_support`;
     - `expected_outcome_error`;
     - `observation_terminal_pressure`;
     - `memory_risk_pressure`;
     - `memory_progress_pressure`;
     - `topology_pressure`;
     - `candidate_uncertainty`;
     - `compute_confidence_pressure`.
   - `AgentEventBundle` carries those same fields into affect.
   - Affect now reacts to evaluator support, expected/actual mismatch, observation terminal pressure, memory risk/progress, topology pressure, candidate uncertainty, and Ouro/compute confidence pressure.
   - This is still input-side/self-model wiring only: no new direct policy score term and no game/domain hardcoding.
4. `run_ablation_ladder.sh` now accepts `START_CHECKPOINT`.
   - Step 0 remains the frozen historical baseline.
   - Steps 1-7 can be seeded from a common active checkpoint with `--checkpoint START_CHECKPOINT --weights_only --reset_optimizer`.
   - This prevents a "new encoder ladder" from accidentally comparing fresh random starts.

### Verification

- Py-compile:
  - `claude_sandbox/self_model.py`;
  - `claude_sandbox/arc_agent_hunter_seeker_codex.py`;
  - `claude_sandbox/arc_agent_pairwise_stockfish_codex.py`;
  - `claude_sandbox/train_arc_codex.py`.
- Focused tests:
  - self-model/measurement slice: `7 passed, 123 deselected`;
  - observation/topology/protected slice: `26 passed, 144 deselected`;
  - ladder/report slice: `14 passed`.
- Full root suite:
  - `414 passed, 1 skipped in 11.76s`.
- GPU smoke:
  - command loaded `encoder_anchor_candidate_20260503.pt` with `--weights_only --reset_optimizer`;
  - old self-model state was correctly rejected because the self-eval input dimension changed;
  - `ls20`, `max_steps=80`, `n_runs=1`, `self_model_mode=passive`;
  - solved level 1 at step `13`;
  - checkpoint: `claude_sandbox/checkpoints_self_model_eval_wiring_smoke_gpu/arc_ls20_run1.pt`;
  - measurement: `claude_sandbox/perf_event_dumps_self_model_eval_wiring_smoke_gpu/ls20/measurement_run_1.json`;
  - expanded `self_model.self_eval_summary` fields are present in the measurement JSON.

### Current interpretation

- Topology is not the current wa30 bottleneck.
- The protected-terminal evidence bugs are fixed.
- The phase-continuity/source-quarantine fixes are permanent data-contract fixes.
- The remaining wa30 ceiling is policy/action-effect competence after trusted replay becomes invalid.
- The next correct work is checkpoint generation/ladder measurement from the new encoder candidate, then return to wa30 with those weights.

### Next run to launch

- Run the full seeded ladder with:
  - `START_CHECKPOINT=claude_sandbox/checkpoints_encoder_retrain/encoder_anchor_candidate_20260503.pt`;
  - direct venv Python via `claude_sandbox/run_ablation_ladder.sh`;
  - `ANCHOR_BATCH_SIZE=1`;
  - no smoke pairs except where explicitly in quick/VRAM mode.
- After the ladder, run `compare_ladder_summaries.py`, inspect anchor/self-model/AttnRes diagnostics, then promote the best checkpoint before returning to wa30.

## Direct Seeded Ladder Progress and NaN Hardening (2026-05-04 local)

### Direct GPU rule

- The shell ladder wrapper was observed loading Ouro on CPU under the sandbox path.
- Direct module execution with `/home/moloch/ouro_project/venv/bin/python -m claude_sandbox.train_arc_codex` was verified with `nvidia-smi`.
- Current rule for this environment:
  - do not use the shell wrapper for GPU ladder work;
  - run each ladder step directly with venv Python;
  - verify Ouro steps with `nvidia-smi` after launch.

### Runtime bugs fixed during direct ladder setup

1. Click-objectivity inference bug:
   - failure:
     - `ft09` crashed in `_score_candidates_objectivity`;
     - cause was `torch.sigmoid(logits).cpu().numpy()` on a grad-tracking tensor.
   - fix:
     - objectivity scoring now runs under `torch.no_grad()`;
     - tensor conversion uses `.detach().cpu().numpy()`.
   - verification:
     - `py_compile` passed;
     - focused tests passed;
     - short `ft09` smoke passed through the previous crash point.
2. Passive self-model NaN contamination:
   - failure:
     - first Step 2 run reached late `wa30` with `total_score=NaN`, `transition_score_normed=NaN`, `risk_patcher_score_gap=NaN`;
     - measurement also showed `self_model_loss_ema=NaN`, `aggregator_fuse_weight_norm=NaN`, `temporal_feature_norm=NaN`;
     - `Avg ranker loss: nan`, so the Step 2 checkpoint was contaminated and discarded.
   - root cause:
     - score components can contain diagnostic NaNs;
     - `_compute_self_eval_summary()` accepted non-finite numeric values as floats;
     - those values entered the self-model/temporal aggregator and then the ranker temporal path.
   - fix:
     - self-eval score ingestion now ignores non-finite values;
     - self-eval vectors are `nan_to_num` clipped to `[0, 1]`;
     - temporal buffer snapshots sanitize temporal features, self-model hidden state, track summary, event summary, loop delta, and self-eval summary;
     - off-policy temporal recompute sanitizes all snapshot tensors and outputs;
     - candidate scoring sanitizes proposal scores, temporal features, ranker/prior logits, confidence gates, gated transition scores, and totals;
     - ranker training skips non-finite score/loss/gradient updates instead of stepping contaminated optimizers;
     - self-model event-prediction loss skips non-finite loss/gradient updates.
   - tests added:
     - self-eval NaN/Inf guard;
     - temporal recompute NaN/Inf guard.
   - verification:
     - py-compile:
       - `claude_sandbox/arc_agent_hunter_seeker_codex.py`;
       - `claude_sandbox/arc_agent_pairwise_stockfish_codex.py`.
     - focused suite:
       - `309 passed`.

### Direct seeded ladder results so far

- Run namespace:
  - `new_encoder_self_eval_direct2_20260503`.
- Common seed checkpoint for active steps:
  - `claude_sandbox/checkpoints_encoder_retrain/encoder_anchor_candidate_20260503.pt`.
- Common games:
  - `ls20 ft09 r11l tr87 wa30`.
- Common run shape:
  - `max_steps=400`;
  - `n_runs=3`;
  - `eps=0.15`;
  - `load_trajs=claude_sandbox/trusted_plus_expanded`;
  - `pretrain_iters=1`.

#### Step 1: encoder-only, self-model off

- Checkpoint:
  - `claude_sandbox/checkpoints_running/new_encoder_self_eval_direct2_20260503/ladder_step_1.pt`.
- Event dumps:
  - `claude_sandbox/ablation_event_dumps/new_encoder_self_eval_direct2_20260503/step_1/`.
- Comparator summary:
  - `runs=15`;
  - `levels_completed_total=14`;
  - per-run levels:
    - `4, 0, 1, 1, 0, 1, 1, 1, 1, 2, 2, 0, 0, 0, 0`.
- Interpretation:
  - clean encoder-only baseline from the new encoder seed;
  - no self-model/anchor/AttnRes attribution active.

#### Step 2: encoder-only, passive self-model

- Checkpoint:
  - `claude_sandbox/checkpoints_running/new_encoder_self_eval_direct2_20260503/ladder_step_2.pt`.
- Event dumps:
  - `claude_sandbox/ablation_event_dumps/new_encoder_self_eval_direct2_20260503/step_2/`.
- Artifact scan:
  - `15` measurement JSON files present;
  - `rg "\\bNaN\\b|Infinity|-Infinity"` over Step 2 dumps/checkpoints returned no matches.
- Comparator summary against Step 1:
  - `runs=15`;
  - `levels_completed_total=19`;
  - per-run levels:
    - `0, 0, 0, 1, 0, 1, 5, 4, 3, 1, 2, 2, 0, 0, 0`.
  - self-model diagnostics:
    - `self_model_loss_ema=0.0326`;
    - `aggregator_fuse_weight_norm=2.13`;
    - `temporal_feature_norm=7.23`;
    - `self_model_gru_grad_norm=0.00311`.
  - comparator alarms:
    - none.
- Interpretation:
  - passive self-model wiring is now numerically stable in this sweep;
  - Step 2 improved total levels vs Step 1, mostly through `r11l`;
  - `ft09` regressed relative to Step 1 in this seed/run ordering;
  - `wa30` remains at `0/0/0` in encoder-only passive mode;
  - no Step 2 checkpoint contamination remains after the NaN hardening rerun.

### Current ladder state

- Step 1, Step 2, and Step 3 are clean resume points.
- Current best level total in this direct ladder is still Step 2 (`19`).
- Step 3 proves the full Ouro path is wired and GPU-valid, but it is not yet a policy improvement over encoder-only passive self-model in this seed.
- Next empirical step before blindly continuing the ladder:
  - inspect the Step 3 Ouro diagnostics, especially the near-constant `Δloop=6.107` and tiny GRU pooler gate;
  - decide whether this is only an expected identity-start/low-training signal or an instrumentation/feature-flow bug;
  - only then proceed to Step 3.5 (`Ouro + anchor only`) or patch the feature flow first.

#### Thermal guard patch for long GPU runs

- Motivation:
  - first Step 3 attempt was killed manually after GPU reached about `84C` and CPU package/core reached about `94C`;
  - the accidental `env ... python` relaunch stayed inside the sandbox, hid CUDA/NVML, and loaded Ouro on CPU.
- Harness changes in `claude_sandbox/train_arc_codex.py`:
  - defaults now set `ARC_API_URL=offline`;
  - defaults now limit CPU math threads with `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4`, and `NUMEXPR_NUM_THREADS=4`;
  - added opt-in `--thermal_guard`;
  - added GPU/CPU threshold/resume/check/sleep flags;
  - GPU temp is read through `nvidia-smi` with `/usr/bin/nvidia-smi` fallback;
  - CPU temp is read from `sensors`, parsing CPU package/core/Tctl/Tdie only;
  - `train_on_game()` checks the guard before expensive agent steps and pauses/resumes without killing the process.
- Step 3 launch used:
  - `--thermal_guard`;
  - `--thermal_gpu_max_c 82`;
  - `--thermal_cpu_max_c 90`;
  - `--thermal_gpu_resume_c 68`;
  - `--thermal_cpu_resume_c 76`;
  - `--thermal_check_every 1`;
  - `--thermal_sleep_seconds 60`.
- Runtime behavior:
  - verified Ouro loaded on `cuda:0`;
  - verified `nvidia-smi` saw about `6.2-6.9 GB` VRAM during the run;
  - guard repeatedly paused at GPU `83-84C` or CPU over budget and resumed after cooldown to about GPU `55-56C`, CPU `60-63C`;
  - one external CPU check saw `92C/94C` during a long step before the internal guard reached its next checkpoint; the guard caught the next boundary and paused safely;
  - final idle temp after completion was about GPU `41C`.

#### Step 3: Ouro baseline, GRU pooler, self-model/cortex/anchor off

- Command shape:
  - direct venv Python module execution;
  - `--backbone_mode ouro`;
  - `--self_model_mode off`;
  - `--cortex_monitor_mode off`;
  - `--loop_pooler_kind gru`;
  - no anchor flags.
- Checkpoint:
  - `claude_sandbox/checkpoints_running/new_encoder_self_eval_direct2_20260503/ladder_step_3.pt`.
- Event dumps:
  - `claude_sandbox/ablation_event_dumps/new_encoder_self_eval_direct2_20260503/step_3/`.
- Artifact scan:
  - `15` measurement JSON files present;
  - text scan over Step 3 dumps/per-game artifacts found no `NaN`/`Infinity`;
  - checkpoint finite scan found `471` floating tensors, `26,861,444` floating values, `0` non-finite tensors.
- Comparator summary against Steps 1 and 2:
  - `runs=15`;
  - `levels_completed_total=17`;
  - per-run levels:
    - `2, 0, 0, 1, 1, 1, 1, 2, 5, 2, 0, 2, 0, 0, 0`.
  - anchor attempts/successes:
    - `0/0` as expected for no-anchor Step 3.
  - self-model diagnostics:
    - `0`/absent as expected for self-model off.
  - failure counts:
    - `mechanism=71`;
    - `planner=49`;
    - no topology count in comparator output for this step.
  - comparator alarms:
    - none.
- Interpretation:
  - Step 3 is numerically clean and checkpointable;
  - Step 3 underperforms Step 2 by total levels (`17` vs `19`);
  - Step 3 improves over Step 1 (`17` vs `14`), but the added compute is not justified by this result alone;
  - `wa30` remains blocked at `0/0/0`, with deaths around step `200` in all three Step 3 runs;
  - live logs repeatedly showed `Δloop=6.107` and a very small loop-pooler gate (`~0.006` by the end), suggesting the current Ouro path is either still identity-start/undertrained or not injecting useful per-candidate variation into selection.

#### Post-Step 3 diagnosis and permanent fixes

- Terminal outcome memory zero-delta detection was too narrow:
  - the old exact-memory softening path only inspected latent `d=` sketch deltas;
  - Step 3 diagnostics showed many terminal keys with zero latent delta but nonzero predicted-frame delta;
  - those keys were incorrectly treated as all-zero terminal contexts.
- Fix:
  - `_terminal_context_key_looks_zero_delta()` now checks both latent `d=` and predicted-frame `f=` sketches;
  - if either available delta-bearing sketch has nonzero delta, the terminal context is not treated as zero-delta;
  - exact terminal-memory penalties therefore remain active for candidate frames whose visible successor changed even when the latent delta sketch is zero.
- Architectural bug found in the Ouro/encoder feature path:
  - `GridEncoder` prepended a learned CLS token after patch processing, but there is no transformer block before Ouro;
  - encoder-only CLS consumers therefore saw an effectively constant CLS token across different frames;
  - a diagnostic on `16` distinct trusted frames found `encoder_cls_feature_std_mean=0.0` and `encoder_cls_pair_l2_mean=0.0` before the fix.
- Fix:
  - `GridEncoder.forward()` now makes CLS content-conditioned with a conservative mean patch summary:
    - `cls = cls_token + tanh(cls_summary_scale) * mean(projected_patch_tokens)`;
    - `cls_summary_scale` starts at `0.10`, so the change is small but immediately nonzero;
    - the full patch token sequence is unchanged for Ouro.
- Post-fix diagnostics:
  - encoder-only diagnostic with the Step 3 checkpoint now reports `encoder_cls_feature_std_mean=0.0416` and `encoder_cls_pair_l2_mean=2.639`;
  - small Ouro GPU diagnostic reports varied final CLS distances (`final_cls_pair_l2_mean=15.9998`, `final_cls_feature_std_mean=0.2424`);
  - this confirms the constant-CLS feature-flow bug is gone.
- Trajectory CLS cache invalidation was tightened:
  - per-file `*_traj_cls.pt` caches and `_traj_cls_bundle.pt` entries now require a versioned signature containing trajectory metadata and a model-output signature;
  - the model signature covers encoder weights plus loop-pooler weights/mode when the loop pooler is active;
  - stale caches from the pre-content-CLS encoder are rejected and re-encoded instead of silently training from constant cached tokens;
  - missing or invalid caches are still self-healing: live encoding writes both per-file and bundled cache entries.
- Trajectory CLS cache reuse was then made less wasteful:
  - the signature now separates schema/structure compatibility from exact weight identity;
  - future launches can accept schema-compatible caches when the trajectory, backbone mode, loop-pooler mode, and output structure still match;
  - exact matches remain preferred, but a new seed checkpoint no longer has to throw away all trusted-trajectory CLS caches only because a small amount of training changed weights;
  - incompatible schemas, incompatible trajectory payloads, or incompatible feature shapes are still rejected and re-encoded.
- Thermal guard update:
  - default ceilings were raised by the requested `+2.5C`;
  - current defaults are `--thermal_gpu_max_c 84.5` and `--thermal_cpu_max_c 92.5`;
  - resume thresholds remain `68C` GPU and `76C` CPU unless explicitly overridden.
- Verification after these patches:
  - `py_compile` passed for `arc_agent_pairwise_stockfish_codex.py`, `train_arc_codex.py`, `grid_encoder_codex.py`, and `arc_agent_hunter_seeker_codex.py`;
  - focused unit suite passed: `311 passed`.

#### Current state after the feature-flow fix

- The Step 1/2/3 artifacts remain valid historical results for the old forward path.
- They should not be treated as final evidence for the fixed architecture because the encoder CLS and trajectory cache semantics changed.
- Content-CLS-compatible seed regeneration completed:
  - source checkpoint: `claude_sandbox/checkpoints_encoder_retrain/encoder_anchor_candidate_20260503.pt`;
  - output checkpoint: `claude_sandbox/checkpoints_encoder_retrain/encoder_content_cls_candidate_20260504.pt`;
  - mode: `--backbone_mode ouro`, `--weights_only`, `--reset_optimizer`, `--pretrain_only`, `--pretrain_iters 10`, `--unfreeze_encoder_for_anchor`, thermal guard enabled;
  - trusted buffer loaded `14090` transitions with expert fraction `0.896`;
  - final logged losses at iteration `10/10` included `ranker_loss=0.7052`, `prior_loss=1.9013`, `nextframe_loss=1.0221`, `observation_loss=1.4379`, `changed_mask_iou=0.6891`;
  - checkpoint save completed cleanly and the GPU returned to idle afterward.
- Post-save verification:
  - `py_compile` passed for the touched sandbox modules;
  - focused unit suite passed: `311 passed`;
  - recursive checkpoint finite scan found `481` floating tensors, `30,514,986` floating values, and `0` non-finite tensors;
  - cache-load smoke from the new checkpoint loaded all `31` solved trajectories and `14090` transitions from bundled CLS cache entries marked `schema-compatible`;
  - the smoke showed no live `Encoding ... through Ouro` trajectory rebuild, so the cache-reuse optimization is working for the next ladder launch.
- Next checkpoint agenda:
  - restart the direct ladder under a new namespace rather than extending `new_encoder_self_eval_direct2_20260503`;
  - use `encoder_content_cls_candidate_20260504.pt` as the new seed checkpoint;
  - keep thermal guard enabled and continue running through direct venv Python so CUDA/NVML remain visible.

#### Self-model future work: domain-general thought continuity

- Current limitation:
  - the self-model and temporal aggregator exist in the shared pairwise-agent surface;
  - however, the richest self-diagnosis/event stream is still Hunter-Seeker-centric;
  - without Hunter-Seeker, the system has temporal features, but it does not yet have a clean standalone memory of "what my thinking has been doing" across Ouro calls.
- Desired direction:
  - make the self model carry a domain-general `thought_signature` every step;
  - the signature should be produced from internal computation, not game-specific state:
    - Ouro loop-state deltas and convergence;
    - loop-pooler or attention weights;
    - evaluator/ranker disagreement;
    - action-prior entropy and selected-action confidence;
    - next-frame / changed-mask prediction error;
    - anchor/evaluator pressure when available;
    - score-component conflict and non-finite/instability diagnostics.
  - feed a compact zero-init projection of this thought signature into the temporal context path so it can provide continuity across decisions;
  - keep it diagnostic-first or very low-weight initially until ablated.
- Architecture rule:
  - Hunter-Seeker should add object/topology/engram/self-diagnosis evidence on top of this channel;
  - Hunter-Seeker must not be required for the basic self-model continuity mechanism.
- Choice-ownership goal:
  - the self model should eventually carry a small trace of why the agent chose the selected branch:
    - winning score components;
    - strongest rejected alternative;
    - uncertainty/conflict flags;
    - remembered failure or success pressure;
    - post-action surprise when the world disagrees with prediction.
  - This is the path toward the evaluator and self-diagnosis feeling like the same system making choices and noticing its own issues, rather than a diagnostic sidecar.
- Additional self-model ideas to evaluate after the current checkpoint ladder:
  - maintain a compact recent-decision trace ring: chosen candidate, closest rejected alternative, uncertainty, conflict, remembered pressure, and post-action surprise;
  - add a domain-general agency/controllability estimate from predicted effect vs observed effect, inverse-action confidence, and whether the chosen action changed the world as intended;
  - add a calibration ledger: predicted value, evaluator/ranker disagreement, realized outcome, and whether the self-diagnosis was right about the failure mode;
  - add a compute-control signal that can eventually request deeper evaluation, cheaper evaluation, or recovery mode based on uncertainty and repeated low-progress loops;
  - add a memory-trust gate for engram/topology/evaluator evidence so the model can learn when a remembered warning is reliable rather than treating memory as a permanent blacklist;
  - add loop-trap/fatigue features from repeated low-delta Ouro loops, repeated rejected-action structure, and repeated no-effect predictions;
  - keep every new channel identity-start, ablatable, logged, and diagnostic-first until a ladder proves that it improves behavior.

#### Self-model implementation outline before the fresh ladder

- Scope decision:
  - do this before restarting the content-CLS ladder;
  - do not start encoder acquisition / RLTT-weight work as part of this patch;
  - do not let the self model directly override action selection in this pass.
- Current code reality:
  - `SelfModel.build_input()` already consumes loop-delta, affect, track summary, and a `24`-dim `self_eval_summary`;
  - Hunter-Seeker already populates `self_eval_summary` with safety pressure, hazard pressure, terminal memory, engram conflict, observation loss/activity, evaluator pair support, expected-outcome error, memory risk/progress, topology pressure, candidate uncertainty, and compute-confidence pressure;
  - temporal-context features are already snapshot into the replay buffer and can be recomputed with gradient during ranker training;
  - the remaining gap is that thought continuity is not a named shared substrate: it is implicit in Hunter-Seeker score components instead of being a domain-general `thought_signature`.
- Patch sequence:
  1. Add a shared `THOUGHT_SIGNATURE_NAMES` schema in `self_model.py`, plus `empty_thought_signature()` and `THOUGHT_SIGNATURE_DIM`.
  2. Extend `SelfModel` input construction with an optional `thought_signature` tensor:
     - default to zeros when absent;
     - sanitize NaN/Inf;
     - preserve batch broadcasting;
     - keep the context-token projector zero-init so Ouro injection remains identity-start.
  3. Keep `self_eval_summary` for choice/evaluator/risk diagnostics and reserve `thought_signature` for domain-general internal-computation continuity.
  4. Initial thought-signature fields should be compact and game-agnostic:
     - loop-delta relative to EMA;
     - Ouro confidence and expected-exit pressure;
     - loop-pooler/attention gate or entropy when present;
     - action-prior entropy / confidence when available;
     - score margin and candidate uncertainty;
     - strongest rejected alternative pressure;
     - predicted-vs-observed effect error / post-action surprise;
     - observation-learning activity and changed-mask quality;
     - evaluator/ranker disagreement when available;
     - non-finite or skipped-update pressure.
  5. Add snapshot plumbing:
     - `_Transition.thought_signature_snapshot`;
     - `TransitionReplayBuffer.push(...)`;
     - ranking-pair sampling surfaces;
     - Hunter-Seeker `_last_thought_signature_snapshot`;
     - `_recompute_temporal_features(...)`.
  6. Add a Hunter-Seeker builder such as `_compute_thought_signature()` that reads only generic internal signals and already-existing diagnostics.
  7. Feed `thought_signature` into `_self_model_advance_and_predict()` and off-policy temporal-feature recomputation.
  8. Add measurement diagnostics:
     - named `thought_signature` values under `measurement_summary["self_model"]`;
     - `thought_signature_norm`;
     - optional last decision trace: chosen score, margin, uncertainty, strongest rejected pressure, post-action surprise.
  9. Tests:
     - `SelfModel.build_input()` defaults the thought signature to zero/no-op;
     - explicit thought signature appears in the correct tail slice;
     - batched thought signatures broadcast/validate correctly;
     - replay buffer preserves/sanitizes the snapshot;
     - `_recompute_temporal_features()` accepts old transitions with no thought signature;
     - measurement summary emits named thought-signature diagnostics when self-model is active;
     - focused `py_compile` and self-model/causal tests pass.
- Deferred beyond this patch:
  - self-value head / pairwise internal evaluator distillation;
  - evaluator as a hot-path uncertainty tool;
  - direct self-value contribution to action score;
  - broader OutcomeAdapter/progress abstraction;
  - Hunter-Seeker structural refactor.

#### Self-model thought-signature patch implemented

- Implemented files:
  - `claude_sandbox/self_model.py`;
  - `claude_sandbox/arc_agent_pairwise_stockfish_codex.py`;
  - `claude_sandbox/arc_agent_hunter_seeker_codex.py`;
  - `tests/unit/test_self_model.py`;
  - `tests/unit/test_causal_correctness.py`.
- Shared self-model changes:
  - added `THOUGHT_SIGNATURE_NAMES`, `THOUGHT_SIGNATURE_DIM`, and `empty_thought_signature()`;
  - extended `SelfModel.build_input()` with optional `thought_signature`;
  - absent thought signatures default to all-zero, preserving old call sites and old replay entries;
  - NaN/Inf thought inputs are sanitized before concatenation;
  - `ContextTokenProjector` remains zero-init, so Ouro context injection remains identity-start.
- Initial thought-signature schema:
  - `loop_delta_rel`;
  - `ouro_low_confidence`;
  - `ouro_expected_exit_pressure`;
  - `loop_pooler_gate_abs`;
  - `loop_pooler_attention_entropy_norm`;
  - `action_prior_entropy`;
  - `action_prior_confidence`;
  - `score_margin_pressure`;
  - `candidate_uncertainty`;
  - `strongest_rejected_pressure`;
  - `post_action_surprise`;
  - `observation_activity`;
  - `changed_mask_quality`;
  - `evaluator_ranker_disagreement`;
  - `nonfinite_skip_pressure`;
  - `skipped_update_pressure`.
- Replay / training plumbing:
  - `_Transition` now carries `thought_signature_snapshot`;
  - `TransitionReplayBuffer.push()` snapshots and sanitizes it;
  - `_to_tensors()` emits it only when all sampled transitions have compatible snapshots;
  - ranking-pair and sibling-pair samplers forward it as `{pos,neg}_thought_signature_snapshot`;
  - `_train_ranker()` passes it into Hunter-Seeker's off-policy temporal-feature recomputation;
  - old transitions without the field still fall back to zero/default behavior.
- Hunter-Seeker wiring:
  - added `_compute_thought_signature()`;
  - the builder reads generic internal diagnostics only: loop/Ouro confidence, loop-pooler gate/attention entropy, action-prior entropy/confidence, candidate margin pressure, strongest rejected alternative pressure, observation surprise/activity, changed-mask quality, evaluator/ranker disagreement pressure, non-finite pressure, and skipped-update pressure;
  - `_self_model_advance_and_predict()` feeds the signature into `SelfModel.build_input()` and snapshots it for replay;
  - `measurement_summary()["self_model"]` now emits named `thought_signature` values plus `thought_signature_norm`;
  - `measurement_summary()["self_model"]["choice_ownership_trace"]` now exposes chosen score, score margin, candidate uncertainty, strongest rejected pressure, and post-action surprise.
- Verification:
  - `py_compile` passed for `self_model.py`, `arc_agent_pairwise_stockfish_codex.py`, and `arc_agent_hunter_seeker_codex.py`;
  - focused suite passed: `340 passed`;
  - no ladder, encoder-acquisition, checkpoint-generation, or GPU training run was started as part of this patch.

## Cleanup Items 1-4 Checkpoint (2026-05-04)

### Item 1: active test tree checked

- The active-test cleanup was already done:
  - root `pytest.ini` collects from `tests/`;
  - active tests live under `tests/unit/`, `tests/integration/`, and `tests/reports/`;
  - no active loose `test_*.py` files remain in `claude_sandbox/`.
- The older "Future cleanup: active test tree" note is now historical and superseded by the `Test Tree Cleanup (2026-05-03)` section above.

### Item 2: low-risk performance cleanup implemented

- Changed file:
  - `claude_sandbox/arc_agent_hunter_seeker_codex.py`
- `SceneParser._compute_adjacency()` now deduplicates component-pair contacts before entering the Python loop:
  - the horizontal/vertical NumPy masks are unchanged;
  - adjacency output is still `Dict[int, Set[int]]`;
  - repeated border pixels between the same two components now produce one loop iteration instead of one iteration per touching pixel.
- This is behavior-preserving and domain/game agnostic:
  - no game ids;
  - no color ids;
  - no ARC-level rules;
  - no change to topology semantics.
- Added a focused regression test:
  - `tests/unit/test_causal_correctness.py::test_scene_parser_adjacency_dedupes_repeated_border_pairs`.

### Item 3: behavior-preserving architecture cleanup implemented

- Changed files:
  - `claude_sandbox/arc_agent_pairwise_stockfish_codex.py`;
  - `claude_sandbox/arc_agent_hunter_seeker_codex.py`.
- Added `LEGACY_ARC_GRID_SIZE = 64` in the shared pairwise module.
- Replaced remaining generic `grid_h/grid_w=64` fallback defaults in pairwise scoring/click-normalization surfaces with the named constant.
- Hunter-Seeker imports the same constant for remaining pre-observation frame-size fallbacks and direct `score_candidates()` defaults.
- The intent is explicit now:
  - runtime/domain adapters should pass real dimensions;
  - the `64` fallback remains only for legacy ARC compatibility and old direct-call tests;
  - no behavior change was intended.

### Item 4: topology/wa30 cleanup audit

- Current topology state:
  - topology is still useful and should not be removed wholesale;
  - cheap runtime topology and cached symbolic summaries remain the right active path;
  - full topology remains available for tests and future diagnostics;
  - the current wa30 level-3 ceiling is not evidence that topology cannot represent the board.
- Permanent fixes to keep:
  - trajectory continuity/source quarantine;
  - state-aligned exact phase trust;
  - hazard-aware safety scoring;
  - protected-avatar/protected-terminal poisoning fixes;
  - populated score-component traces.
- Keep but simplify later:
  - risk patcher core should become one smaller generic risk-arbitration layer;
  - phase recovery/reseed/escalation should become one recovery-state machine;
  - engram recall and observation-effect recall should keep support/conflict diagnostics but be reviewed for memory-store overlap;
  - model-basin sampling should remain diagnostic-first until calibrated across games.
- Bloat candidates to ablate before deletion:
  - all-candidates-risky recovery selector;
  - standalone live-recovery selector;
  - exact-veto recovery window;
  - repeated pressure-window/model-basin latch tuning.
- Do not delete those bloat candidates yet:
  - current evidence says some were symptom treatments, but removal needs feature flags or an ablation harness plus a small mixed regression;
  - fresh ladder/new-checkpoint work should happen before more topology behavior patches.

## Performance / Architecture Cleanup Pass (2026-05-04)

### Implemented

- `TransitionReplayBuffer` now maintains serial-based sampling indexes:
  - real transitions;
  - changed transitions;
  - expert transitions;
  - click transitions;
  - terminal and nonterminal transitions;
  - game, level, game+level, and sibling-group pools.
- Hot samplers now use those indexes instead of rebuilding pools from full-buffer scans:
  - `sample_expert_action_batch`;
  - `sample_spatial_batch`;
  - `sample_nextframe_batch`;
  - `sample_ranking_pairs`;
  - `sample_terminal_failure_pair_batch`;
  - `sample_smoke_pair_batch`;
  - `sample_sibling_pairs`.
- `sample_quality_gap_pair_batch()` no longer performs an unconditional O(N^2) pair scan:
  - small buffers (`<=256` eligible real transitions) still use exhaustive search for deterministic tests and exact behavior;
  - large buffers use bounded indexed pair proposals;
  - returned batches now include `pos_quality` and `neg_quality` diagnostics.
- `EventLog.append()` now has an O(1) common eviction path:
  - it tracks terminal-event count;
  - when the oldest event is nonterminal, eviction is `popleft()`;
  - the protected terminal scan remains only for the uncommon case where terminal anchors sit at the left edge.
- Model-basin diagnostic rollout queue now uses `collections.deque.popleft()` instead of `list.pop(0)`.
- Added `_trace_candidate_risk_records()` in Hunter-Seeker:
  - deduplicates root trace entries by `(action, click_x, click_y)`;
  - precomputes the shared risk/recovery fields once;
  - `live_recovery` and `risk_patcher` now consume the same canonical candidate-risk records.
- Added cleanup/ablation switches with defaults preserving current behavior:
  - `enable_live_recovery_selector`;
  - `enable_all_risky_recovery_selector`;
  - `enable_exact_veto_recovery_window`.
- Added focused tests:
  - replay-buffer index eviction/terminal marking;
  - quality-gap positive-vs-negative quality diagnostics;
  - event-log nonterminal eviction with terminal preservation.

### Audit After This Pass

- Remaining performance targets:
  - `ranking_pair_diagnostics()` still scans and does pair counting, but it is diagnostic/failure-path work rather than hot-path sampling.
  - Observation replay/buffer samplers likely need the same serial-index treatment as `TransitionReplayBuffer`.
  - `ObjectTable.update_from_scene()` still has object-to-track matching that can become O(objects * tracks); acceptable for ARC-sized scenes, future issue for richer domains/video.
  - Post-veto candidate-generation diagnostics still sort multiple row views; diagnostic only, but can be top-k/heapified later.
  - Score-component dict copying remains a real Python overhead source; the clean fix is a compact internal candidate diagnostic record and dump-time materialization, not more ad hoc dict pruning.
- Remaining architecture cleanup targets:
  - phase recovery/reseed/escalation/exact-veto should still be unified into one recovery-state machine later;
  - risk patcher/live recovery/model-basin pressure should still collapse into one risk-arbitration layer after ablation;
  - color/entity naming and OutcomeAdapter/progress abstraction remain deferred;
  - Hunter-Seeker refactor remains deferred until topology and observation-learning behavior are stable.

### Verification

- Py-compile passed for:
  - `claude_sandbox/arc_agent_pairwise_stockfish_codex.py`;
  - `claude_sandbox/arc_agent_hunter_seeker_codex.py`;
  - `tests/unit/test_causal_correctness.py`.
- Focused replay/risk/model-basin suites passed:
  - replay/quality-gap/event/risk slice: `15 passed`;
  - model-basin/all-risky/live-recovery/exact-veto slice: `21 passed`;
  - broader unit slice: `355 passed`.
- Full root suite passed:
  - `424 passed, 1 skipped`.

## Cleanup Targets Completed Except Big Split (2026-05-04)

### Implemented

- Completed the smaller cleanup targets requested after the first performance pass; the large Hunter-Seeker split remains intentionally deferred until topology and observation learning stabilize.
- `ObservationReplayBuffer` now maintains serial-based indexes:
  - known-action transitions;
  - unlabeled watched transitions;
  - click-labeled transitions;
  - visual/topology-delta transitions.
- Observation samplers now use those indexes instead of rebuilding eligible pools by scanning the full buffer on every sample:
  - `sample_known_action_batch`;
  - `sample_unlabeled_transition_batch`;
  - `sample_click_transition_batch`;
  - `sample_topology_delta_batch`;
  - `sample_object_contrastive_batch` via the topology-delta sampler.
- The observation buffer unregisters evicted serials from every index, so stale rows cannot be sampled after FIFO eviction.
- Hunter-Seeker chosen-score trace assembly now creates one per-step selected-score snapshot and reuses it for:
  - `chosen_candidate.score_components`;
  - `chosen_score_components`;
  - legacy `score_components`.
- Added `_phase_recovery_context_flags()` as the shared recovery/risk context helper:
  - reports recovery, exact-veto, reseed, escalation, forced-veto, low-trust, and drift state in one place;
  - used by live-recovery scoring/selection;
  - remains domain/game agnostic.
- Added `_score_components_recovery_context()` for score-component dictionaries:
  - basin-collapse behavior gating now reads one canonical recovery-context helper instead of repeating partial condition sets.
- Objectivity and symbolic planner train-time buffers no longer materialize full deque copies for tiny random batches:
  - added `_sample_deque_by_indices()`;
  - `_train_objectivity_head()` and `_train_symbolic_planner_head()` now fetch sampled rows in one pass.

### Audit Result

- Enacted cleanup targets:
  - observation-buffer sampler indexing;
  - score-component dict churn reduction in the chosen trace hot path;
  - shared recovery/risk context helpers;
  - objectivity/symbolic deque sampling cleanup discovered during audit.
- Still deferred:
  - the big Hunter-Seeker split;
  - deeper recovery-state-machine consolidation;
  - risk-patcher/live-recovery/model-basin arbitration collapse after ablation;
  - domain-neutral naming pass: rename color/colour-centric variables and APIs toward label/value/entity terminology where the field is not literally a rendered color;
  - object-to-track matching acceleration for richer non-ARC domains/video;
  - top-k/heapified post-veto diagnostic sorting.

### Verification

- Py-compile passed for:
  - `claude_sandbox/observation_learning_codex.py`;
  - `claude_sandbox/arc_agent_hunter_seeker_codex.py`;
  - `tests/unit/test_codex_sandbox.py`.
- Focused tests passed:
  - observation/chosen-score/live-recovery/all-risky/model-basin slice: `24 passed`;
  - replay/score/risk/recovery/basin/observation causal slice: `19 passed`.
- Full root suite passed:
  - `425 passed, 1 skipped`.

## Thin Recovery / Risk-Arbitration Consolidation (2026-05-04)

### Implemented

- Added a behavior-preserving typed recovery context layer:
  - `RecoveryContext`;
  - `_phase_recovery_context()`;
  - compatibility wrapper `_phase_recovery_context_flags()`.
- `RecoveryContext` centralizes:
  - recovery active;
  - exact-veto active;
  - reseed active;
  - escalation active;
  - forced pre-state veto;
  - low state trust;
  - state drift;
  - state-alignment trust and mismatch streak.
- Live-recovery effect scoring and live-recovery selection now consume the typed recovery context instead of reassembling partial flag sets.
- Added a behavior-preserving arbitration diagnostic layer:
  - `RiskArbitrationDecision`;
  - `_record_risk_arbitration_decision()`;
  - `_record_risk_arbitration_noop()`.
- Current selectors still make the same decisions:
  - no recovery/reseed/escalation/exact-veto semantics were deleted;
  - no risk-patcher/live-recovery/model-basin winner semantics were changed;
  - existing feature flags still control the same behavior.
- The new arbitration surface records which subsystem last participated:
  - `model_basin` writes diagnostic-only arbitration records after model-basin sampling;
  - `live_recovery` writes active/inactive recovery-selection records;
  - `risk_patcher` and `all_risky_recovery` write active/inactive risk-arbitration records.
- Arbitration diagnostics are now exposed in:
  - step `info["risk_arbitration"]`;
  - action-trace entries;
  - `measurement_summary()["risk_arbitration"]`.

### Next Work

- After fresh current-architecture checkpoints are acquired, run ablations before deleting behavior:
  - `enable_live_recovery_selector=False`;
  - `enable_all_risky_recovery_selector=False`;
  - `enable_exact_veto_recovery_window=False`;
  - model-basin pressure/latch reductions.
- If ablations show the late selectors are still necessary, collapse them into one explicit risk-arbitration layer using `RiskArbitrationDecision` as the public diagnostic surface.
- If ablations show any selector is symptom bloat from the old encoder/wa30 failure, remove it behind tests rather than carrying it into the Hunter-Seeker split.
- Do not perform the large Hunter-Seeker split until topology and observation-learning behavior are stable.

### Verification

- Py-compile passed for:
  - `claude_sandbox/arc_agent_hunter_seeker_codex.py`;
  - `tests/unit/test_causal_correctness.py`;
  - `tests/unit/test_codex_sandbox.py`.
- Focused recovery/risk/model-basin/measurement suite passed:
  - `49 passed`.
- Full root suite passed:
  - `426 passed, 1 skipped`.

