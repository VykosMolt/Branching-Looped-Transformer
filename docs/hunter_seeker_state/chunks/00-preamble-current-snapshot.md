<!-- Source: PROJECT_STATE_HUNTER_SEEKER.md lines 1-194 before the 2026-05-14 split. -->
<!-- Source chunk SHA256: a0a72acb2698f07cbb3a00c0e90b5528b260bc2793baa7bef8d8d886edd278ef -->

# Ouro Project — Hunter-Seeker / Ouro Core State

Split from the former combined `PROJECT_STATE.md` on 2026-05-07. This file is for Hunter-Seeker/Ouro core state: ARC topology, wa30/tr87/ls20 regressions, RLTT evaluator/backbone work, observational learning, engram memory, ladder status, and neural self-model work. Local-agent wrapper state now lives in `PROJECT_STATE_LOCAL_AGENT.md`.

*Johann Hirschner (VykosMolt) · April 2026*

> **Version note:** This document supersedes the v17c state. Sprint 4 (multi-head affordance model) is complete with directional-buffer extension. The checkpoint system has gained a cumulative-provenance chain plus reset/weights-only modes. A domain-adapter refactor has decoupled Hunter Seeker's perception and action layers from ARC-specific assumptions, making the cognitive stack portable across environments while the ARC training harness stays ARC-specific. Two research directions have been deferred and explicitly named (encoder-cognition feedback, video-based world-model pretraining).

> **Latest empirical update:** see the 2026-05-14 teacher-training topology trio entry near the end of this snapshot. It supersedes the older observation-only pretrain caveat and records the current RLTT topology-trio interpretation.

> **Storage cleanup update - 2026-05-14:** old checkpoint roots, event dumps, probe outputs, quarantine folders, and raw consolidation artifacts were purged. Historical sections may still name deleted paths for provenance; those paths are not expected to exist unless they are regenerated. The intentionally retained runtime set is:
> - `models/ouro_rltt_local/`;
> - `checkpoints_running/sprint4_encoder_reverted.pt`;
> - `artifacts/checkpoints/evaluator/pairwise_epoch2.pt`;
> - `claude_sandbox/checkpoints_encoder_retrain/encoder_content_cls_candidate_20260504.pt`;
> - active trusted trajectory/training inputs: `trusted_trajs/`, `claude_sandbox/trusted_plus_expanded/`, `claude_sandbox/trusted_topology_trio_20260513/`, `claude_sandbox/trusted_ls20_smoke_20260513/`, and `claude_sandbox/solved_sequences_expanded/`.
>
> Removed storage classes include `checkpoints_v17b/`, archived checkpoint forests, all non-retained `claude_sandbox/checkpoints*` roots, stale pairwise/evaluator epochs except epoch 2, `ouro_rltt/consolidated_clean.pt`, `archive/`, `runs/`, `_tree_cleanup/`, `claude_sandbox/perf_event_dumps_*`, `claude_sandbox/ablation_event_dumps/`, `claude_sandbox/ablation_runs/`, and cleanup quarantines. Post-clean verification left only the three checkpoint roots above; repo footprint was about `14G`.

## Contents

- `0. Canonical Synthesis - 2026-04-29`: current working truth, source list, core philosophy, Ouro paper implications, architecture map, empirical state, roadmap, and active measurement questions.
- `0.17 Pre/Post Ladder Canonical Detail`: ladder blockers, shipped fixes, anchor fallback validation, VRAM constraints, and post-ladder cleanup priorities.
- `0.18 Claude Sandbox Architecture Audit - 2026-04-29`: current code audit findings, test status, and implementation direction for `claude_sandbox`.
- `0.19 Terminal Predframe 8-Run Ladder - 2026-04-29`: GPU ladder result, comparison to bad terminal-memory probes, random-bypass patch, and next empirical step.
- `1-8`: identity, core idea, philosophical position, file state, setup, training commands, completed sprints, and current v17 behavior.
- `9-18`: checkpoint policy, basal-ganglia/self-model design, CLT paper integration, and future-sprint architecture.
- `19-23`: adapter refactor, domain transfer, sleep/consolidation plans, v17c outcomes, and deferred research directions.
- `24-28`: later diagnostics, ranker/belief questions, encoder drift analysis, and mitigation paths.

---

## 0.0 Current Snapshot - 2026-05-05 RLTT wa30 Ego/Topology

This is the latest working state after the avatar/control-set patch, the protected-overlap/contact patch, the zero-offset recovery patch, the nonviable-basin phase-control patch, and the current RLTT `wa30` probes.

Active model/runtime:

- Current Ouro backbone for Hunter-Seeker probes is the converted local RLTT model at `models/ouro_rltt_local`.
- `transformers` must remain pinned at `4.54.1`.
- Direct project-venv Python invocation sees CUDA correctly. Prefer commands beginning with `/home/moloch/ouro_project/venv/bin/python`; avoid environment-prefix launch shapes that previously hid CUDA from torch.
- The train harness now line-buffers stdout, so long GPU runs expose live step/event progress rather than appearing stuck.
- Post-level reinforcement is configurable with `--post_level_reinforce_iters`; default is now `2`, not the earlier hardcoded `8`, to avoid long RLTT post-level stalls.

Verification after the latest code patch:

- `py_compile` passed for:
  - `claude_sandbox/arc_agent_hunter_seeker_codex.py`;
  - `claude_sandbox/arc_agent_pairwise_stockfish_codex.py`;
  - `claude_sandbox/train_arc_codex.py`;
  - `tests/unit/test_causal_correctness.py`.
- Focused causal-correctness/unit pass after the latest phase-control diagnostics patch: `155 passed`.
- `nvidia-smi --query-compute-apps` showed no active compute processes after the last run.

Implemented topology/ego changes:

- `ObjectTable` now maintains an explicit ego/control set, not just a single inferred avatar:
  - primary ego track id;
  - controlled track ids;
  - per-track confidence;
  - ego ambiguity;
  - last action/centroid/alignment diagnostics;
  - bounded remembered action-to-motion signatures.
- Directional action attribution is now ego-aware:
  - `NONCLICK_MOVED` evidence applies only to tracks whose motion aligns with the issued directional action or whose stable learned signature says they are controlled;
  - other moved objects get `COUPLED_MOVED`, so mirrored/passive objects are not incorrectly promoted into "the avatar" just because they moved.
- Multi-avatar support is explicitly represented:
  - same-direction controlled bodies can be admitted together;
  - mirrored/chiral bodies can be admitted only after repeated stable action-to-motion signatures;
  - `controlled_tracks()` is used by terminal topology, not only `avatar_track()`.
- `on_game_over()` now evaluates terminal/contact topology from all controlled avatar scene entries:
  - controlled avatar object ids and track ids are protected from hazard poisoning;
  - terminal adjacency/frontier diagnostics consider the whole controlled set;
  - the measurement summary records `ego_localization`.
- Score components now expose the new behavior:
  - `ego_track_id`;
  - `ego_confidence`;
  - `ego_ambiguity`;
  - `ego_last_alignment`;
  - `ego_controlled_track_count`;
  - `action_offset_dy`;
  - `action_offset_dx`;
  - `action_zero_offset`;
  - `all_risky_recovery_zero_offset_contact_pressure`;
  - `all_risky_recovery_zero_offset_contact_penalty`.
- Local-contact risk now treats `avatar_overlap_protected_context` as a bounded diagnostic/safety signal when protected-overlap, unknown-contact, or conflict evidence is present. This keeps the signal score-visible and avoids broad action/color/game blacklists.
- All-risky recovery now downweights zero-offset/contact candidates in nonviable/collapse basins. The GPU check confirmed the penalty fires and prevents the prior zero-offset terminal action from being selected, but it does not solve phase 65.
- Phase-control recovery now treats nonviable/collapse basin evidence as control failure rather than progress:
  - weak local topology/reward movement is capped for phase recovery/escalation/reseed updates when the selected candidate's model-basin diagnostics say `viable_fraction=0`, `collapse_fraction=1`, and low `min_path_trust`;
  - persistent nonviable basin pressure activates recovery after `2` steps, escalation after `3`, and reseed after `6`;
  - this is domain-general and uses only model-basin/control diagnostics;
  - future score traces expose `phase_control_nonviable_basin_pressure`, `phase_control_nonviable_basin_active`, and related scalar components through candidate `score_components`.

GPU probe 1 after ego/control-set patch:

- Event root: `claude_sandbox/perf_event_dumps_rltt_wa30_ego_cuda`.
- Config: RLTT `models/ouro_rltt_local`, `wa30`, `max_steps=350`, `n_runs=1`, `eps=0.0`, no replay.
- Result: death at step `283`, levels completed `2`, score `5.161386666666667`.
- Important fix confirmed:
  - terminal topology no longer blamed the stale old avatar candidate;
  - `ego_localization.track_id=31`, color `14`, centroid approximately `[26.0, 61.0]`, confidence `1.0`;
  - terminal topology used the bottom-row controlled body, not the earlier stale color-12 track.
- Remaining issue in this run:
  - fatal selected action still had `safety_penalty=0`;
  - model-basin diagnostics already said nonviable/collapse (`viable_fraction=0`, `collapse_fraction=1`, `model_basin_risk=0.65`);
  - the score stack still selected into the terminal basin.

GPU probe 2 after protected-overlap/contact + reduced reinforcement:

- Event root: `claude_sandbox/perf_event_dumps_rltt_wa30_ego_contact_cuda`.
- Config: RLTT `models/ouro_rltt_local`, `wa30`, `max_steps=290`, `n_runs=1`, `eps=0.0`, no replay, `post_level_reinforce_iters=2`.
- Result: death at step `283`, levels completed `2`, score `5.161386666666667`.
- Confirmed changes:
  - post-level reinforcement used `2` extra iterations and no longer produced the earlier long stall;
  - all-risky recovery engaged near the failure window;
  - ego localization expanded to controlled tracks `[31, 82]`, confidence about `0.92`, ambiguity about `0.05`.
- Terminal failure classification changed:
  - adjacent object colors were `[0, 14]`;
  - both adjacent objects were protected as controlled/avatar context;
  - no terminal hazard candidate was accepted;
  - failure type is better described as protected terminal starvation, not stale-avatar poisoning.
- Timeline diagnosis:
  - the first chosen candidate with `viable_fraction=0`, `collapse_fraction=1`, and `model_basin_risk=0.65` appears around step `249`;
  - from roughly step `249` onward the chosen path remains in a nonviable/collapse basin;
  - by step `283`, there is no evidence that a locally safe terminal action exists.

GPU probe 3 after zero-offset all-risky penalty:

- Event root: `claude_sandbox/perf_event_dumps_rltt_wa30_ego_zero_offset_cuda`.
- Config: RLTT `models/ouro_rltt_local`, `wa30`, `max_steps=290`, `n_runs=1`, `eps=0.0`, no replay, `post_level_reinforce_iters=2`.
- Result: death at step `283`, levels completed `2`, score `5.161386666666667`.
- The zero-offset/contact penalty fired as intended:
  - `all_risky_recovery_zero_offset_contact_penalty` appeared on `25` candidate rows;
  - the previous fatal zero-offset action was no longer selected at terminal;
  - final selected action was directional action `2`, not zero-offset action `5`.
- Failure remained protected terminal starvation:
  - terminal adjacent objects were controlled/protected avatar context;
  - no accepted terminal hazard candidate;
  - ego localization remained correct with controlled tracks `[31, 82]`.
- Trace diagnosis:
  - phase progress advanced to `65` at step `249` and then froze there through death;
  - steps `249-283` were almost entirely all-risky/nonviable;
  - the issue is therefore phase-65 nonviable-basin escape, not the specific zero-offset action.
- Performance:
  - the step `250` to `275` interval was very slow despite only a `290` max-step run;
  - the late basin remains the major compute wall.

GPU probe 4 after phase-control nonviable-basin patch:

- Event root: `claude_sandbox/perf_event_dumps_rltt_wa30_phase_control_cuda`.
- Config: RLTT `models/ouro_rltt_local`, `wa30`, `max_steps=290`, `n_runs=1`, `eps=0.0`, no replay, `post_level_reinforce_iters=2`.
- Result: death at step `283`, levels completed `2`, score `5.161386666666667`.
- The patch changed behavior but did not break the ceiling:
  - step `275` selected through `all_risky_recovery_beam`, whereas the previous run was still ordinary beam at that checkpoint;
  - recovery/escalation/reseed were active through the phase-65 window;
  - the terminal failure moved from protected terminal starvation to mechanism death.
- Terminal diagnosis:
  - last action: directional action `3`;
  - accepted terminal candidate: color `4`, track `71`;
  - terminal printed `color 4 hazard↑` with hazard about `0.85`, unknownness about `0.34`, weight about `0.48`;
  - failure type: `mechanism`;
  - ego localization was still correct: controlled avatar track `31`, color `14`, centroid about `[25.5, 61.0]`.
- Important instrumentation note:
  - this run was launched before the final diagnostic write-through patch, so `phase_control_*` terms were not present in the dumped candidate score components;
  - after the run, `_risk_patcher_write_candidate_diag()` was patched so future traces write nonviable-basin phase-control pressure into candidate `score_components`;
  - CPU verification after that diagnostic patch is `155 passed`.

Current interpretation:

- The previous avatar-perspective bug is fixed. The agent's topology now tracks the controlled body/bodies rather than generic movement.
- The current `wa30` level-3 ceiling is not terminal topology being unable to "see" the avatar at step `283`.
- The run appears to be checkmated earlier, with the important divergence occurring before or around phase `65` at step `249`.
- The newest patch can change the terminal failure class, but it does not yet produce a viable branch out of phase `65`.
- The next real behavior patch should target phase-65 route competence / action-effect recovery before the terminal window, not stronger terminal hazard poisoning at step `283`.
- There is now renewed mechanism evidence on color `4` track `71`, but this should not be turned into a broad color/action blacklist. It should feed bounded mechanism/engram/action-effect diagnostics only.
- Keep the constraints:
  - no game ids;
  - no color ids;
  - no broad action blacklists;
  - no `wa30` route logic;
  - every new safety/recovery term must remain bounded and visible in `score_components`.

Immediate next work:

1. Inspect the phase-65 trace across `ego_zero_offset_cuda` and `phase_control_cuda` to identify why the active recovery/reseed stack cannot produce a viable branch.
2. Add a performance patch for the late all-risky basin. The `250-275` window remains too slow and will block iteration.
3. Check the new mechanism death candidate with bounded diagnostics:
   - compare color/track mechanism evidence before and after the phase-control run;
   - verify hazard-aware scoring and engram/action-effect recall see the color-4 mechanism without creating a color blacklist.
4. If another behavior patch is justified, make it a domain-general phase/action-effect competence patch:
   - e.g. recovery branch diversification, candidate novelty, or action-effect contradiction handling;
   - no `wa30` route logic.
5. After `wa30` level 3 is broken, run the planned patch-necessity audit and remove symptom patches that are not generally useful.

2026-05-14 phase-teacher follow-up implementation:

- Item 3, protected-terminal / ego-contact ambiguity, is now implemented as a generic score and memory signal:
  - `SafetyMixin._ego_contact_ambiguity_components()` distinguishes clean protected self/body contacts from ambiguous protected adjacent contacts using only ego confidence, ego ambiguity, controlled-track count, learned contact beliefs, unknownness, and conflict pressure;
  - candidate scoring writes `ego_contact_ambiguity_*` into score components and applies only a small bounded penalty when real ambiguity/risk is present;
  - protected-terminal context memory includes the ego/contact signal, with compatibility padding so older stored vectors are not discarded just because the feature vector grew;
  - terminal postmortems now report `ego_contact_*` diagnostics for accepted/rejected local candidates and scale protected-adjacent terminal evidence by ambiguity/risk rather than treating every protected adjacent object identically.
- Item 2, observation/action-effect learning feeding policy, is now materially tighter:
  - observation candidate diagnostics expose `obs_effect_model_signal`, `obs_inverse_action_match_confidence`, `obs_effect_policy_signal`, and `obs_effect_recovery_model_gate`;
  - positive observation-effect recovery now requires an effect-model signal, while same-action terminal/negative recall can still apply terminal pressure;
  - inverse-action confidence can raise the recovery action gate when the visual before/after effect supports the candidate action;
  - observation policy pretraining weights ActionPrior/TransitionRanker supervision by action confidence and transition quality.
- Verification so far:
  - targeted protected-terminal and observation-recovery tests passed;
  - full repository test suite passed with `619 passed, 1 skipped`.

Topology trio smoke after item 3 + item 2 changes:

- Command shape:
  - `venv/bin/python -m hunter_seeker_core.train_arc`;
  - `--games ls20 tr87 wa30`;
  - `--backbone_mode ouro --ouro_model_path models/ouro_rltt_local`;
  - `--checkpoint artifacts/checkpoints/running/sprint4_encoder_reverted.pt`;
  - `--load_trajs data/trajectories/trusted_topology_trio_20260513`;
  - `--max_steps 500 --n_runs 1 --eps 0.0 --no_replay --train_every 0`;
  - phase policy remained disabled; no online ranker/prior/world-model updates occurred.
- Artifacts:
  - event/measurement dumps: `artifacts/reports/topology_trio_after_ego_obs_20260514/`;
  - checkpoints: `artifacts/checkpoints/topology_trio_after_ego_obs_20260514/`;
  - saved improved trajectory snippets: `artifacts/trajectories/topology_trio_after_ego_obs_20260514/`.
- Results:
  - `ls20`: `1` level completed, game over at step `91`, failure type `mechanism`, action counts `{1: 24, 2: 27, 3: 17, 4: 23}`, phase-policy rows `0`.
  - `tr87`: `2` levels completed, game over at step `192`, failure type `mechanism`, action counts `{1: 94, 2: 34, 3: 29, 4: 35}`, phase-policy rows `0`.
  - `wa30`: `0` levels completed, game over at step `200`, failure type `protected_terminal_starvation`, action counts `{1: 49, 2: 15, 3: 62, 4: 67, 5: 7}`, phase-policy rows `0`.
- Interpretation:
  - the item 3/2 changes did not regress the strict replay-off topology baseline;
  - `wa30` now records the new ego/contact fields in the terminal diagnostic and correctly identifies the adjacent protected candidates as controlled-body context, so no hazard evidence is invented;
  - this smoke still does not test the teacher-training path, because `train_every=0` and no phase-teacher/observation pretrain was requested.

Teacher-training topology trio smoke after item 3 + item 2 changes:

- Command shape:
  - `venv/bin/python -m hunter_seeker_core.train_arc`;
  - `--games ls20 tr87 wa30`;
  - `--backbone_mode ouro --ouro_model_path models/ouro_rltt_local`;
  - `--checkpoint artifacts/checkpoints/running/sprint4_encoder_reverted.pt`;
  - `--load_trajs data/trajectories/trusted_topology_trio_20260513`;
  - `--phase_teacher_pretrain_iters 100`;
  - `--observation_pretrain_iters 20 --observation_pretrain_infer_every 5`;
  - `--max_steps 500 --n_runs 1 --eps 0.0 --no_replay --train_every 0`;
  - phase policy remained disabled at runtime; no online ranker/prior/world-model updates occurred.
- Artifacts:
  - event/measurement dumps: `artifacts/reports/topology_trio_phase_teacher_obs_20260514/`;
  - checkpoints: `artifacts/checkpoints/topology_trio_phase_teacher_obs_20260514/`;
  - trajectories: `artifacts/trajectories/topology_trio_phase_teacher_obs_20260514/`.
- Pretraining diagnostics:
  - observation pretrain: `20` updates, observation loss `2.0870`, effect loss `0.0271`, topology loss `0.3010`, inverse loss `1.5688`, policy prior loss `1.7307`, policy ranker loss `0.6879`;
  - `ls20` phase teacher: `569` examples, `44` abstained, agreement `0.375`, ranker loss `0.6903`, prior loss `1.3932`, world-model loss `0.8021`;
  - `tr87` phase teacher: `420` examples, `1` abstained, agreement `0.562`, ranker loss `0.4920`, prior loss `0.9760`, world-model loss `1.1088`;
  - `wa30` phase teacher: `1555` examples, `1` abstained, agreement `0.188`, ranker loss `0.6877`, prior loss `1.7529`, world-model loss `1.4685`.
- Results:
  - `ls20`: `1` level completed, level 1 at step `13`, game over at step `79`, failure type `mechanism`, action counts `{1: 16, 2: 23, 3: 19, 4: 21}`, selection methods `{beam_search: 79}`, phase-policy rows `0`.
  - `tr87`: `1` level completed, level 1 at step `37`, game over at step `165`, failure type `mechanism`, action counts `{1: 43, 2: 25, 3: 46, 4: 51}`, selection methods `{beam_search: 88, all_risky_recovery_beam: 3, risk_patcher_beam: 74}`, phase-policy rows `0`.
  - `wa30`: `0` levels completed, game over at step `200`, failure type `mechanism`, action counts `{1: 49, 2: 39, 3: 49, 4: 40, 5: 23}`, selection methods `{beam_search: 200}`, phase-policy rows `0`.
- Interpretation:
  - the teacher path is active and behavior-visible, but it is not yet sufficient autonomous policy learning;
  - it improves first-level speed on `ls20` (`13` steps vs `21` actions in the trusted sequence and `91`-step death in the previous strict smoke), but still dies after a single clear;
  - `tr87` regressed from the previous strict smoke's `2` clears to `1` clear, suggesting the current teacher objective can sharpen early route imitation while failing to preserve later continuation competence;
  - `wa30` changed from protected-terminal starvation to ordinary mechanism death, confirming the ego/contact ambiguity work changed terminal attribution, but the policy still cannot make level-1 progress;
  - the strongest next issue is not runtime phase routing or replay. It is teacher-to-policy transfer after reset/state shift: the agent needs student-on-policy correction data, better post-level teacher labels, and effect-conditioned candidate preferences rather than only pre-run exact-template distillation.

Regression inspection after the teacher-training smoke:

- Full repository tests before the follow-up patch passed: `619 passed, 1 skipped, 18 subtests passed`.
- The `tr87` regression was localized to the first post-level-1 continuation:
  - strict post-refactor / strict item-2+3 runs keep the trusted continuation sequence and clear level 2;
  - full teacher+observation topology trio diverges immediately after `tr87` level 1, around trace row `38`, where all-risky/risk-patcher arbitration starts selecting off-route actions.
- Fresh single-game diagnostic:
  - command shape: same teacher+observation setup but `--games tr87 --max_steps 120`;
  - artifacts: `artifacts/reports/diagnostic_tr87_teacher_obs_20260514/`;
  - result: `tr87` cleared `2` levels by step `67`;
  - interpretation: `tr87` phase-teacher distillation is not intrinsically broken when started from a fresh agent.
- Two-game ablation:
  - command shape: `--games ls20 tr87 --phase_teacher_pretrain_iters 100 --observation_pretrain_iters 0 --max_steps 120`;
  - artifacts: `artifacts/reports/diagnostic_ls20_tr87_phase_teacher_only_20260514/`;
  - result: `ls20` cleared `1` level; subsequent `tr87` cleared only `1` level and diverged after level 1;
  - interpretation: the regression reproduces without the initial global observation pretrain, so the trigger is cross-game carried agent state from `ls20` plus per-game phase-teacher/observation-effect updates.
- Concrete failure mechanism found:
  - the bad `ls20 -> tr87` trace recalled a terminal/negative observation-effect memory from `ls20` on the trusted `tr87` continuation;
  - at the bad `tr87` post-clear row, the chosen trusted action had `obs_engram_namespace_terminal_count=1`, `obs_engram_namespace_terminal_best_similarity≈0.853`, and `engram_negative_support≈0.998`;
  - the fresh `tr87` run had no terminal recall at the same point;
  - that cross-task negative recall lowered the trusted continuation enough for all-risky recovery to select an off-route action.
- Patch applied:
  - `TransitionEffectEngram` now carries a generic `scope`;
  - terminal and hazard observation-effect recalls are skipped when stored scope and current scope differ;
  - progress/change memories can still generalize across scope;
  - online terminal and model-basin labels now write scoped sources such as `terminal_observed:<scope>` and `model_basin_sampler:<scope>`.
- Tests after the patch:
  - `venv/bin/python -m py_compile src/hunter_seeker_core/observation_learning.py src/hunter_seeker_core/stockfish/observation.py` passed;
  - `venv/bin/python -m pytest -q utilities/tests/unit/test_observation_engram_scope.py utilities/tests/unit/test_phase_teacher.py` passed: `5 passed`.
- Remaining verification gap:
  - the post-fix `ls20 -> tr87` behavior diagnostic was started but interrupted before producing results;
  - rerun `diagnostic_ls20_tr87_phase_teacher_scopefix_20260514` or the full topology trio before treating the behavioral regression as closed.
