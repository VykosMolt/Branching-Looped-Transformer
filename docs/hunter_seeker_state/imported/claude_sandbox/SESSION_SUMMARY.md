<!-- Imported from `claude_sandbox/SESSION_SUMMARY.md` during Hunter-Seeker state consolidation on 2026-05-14. -->
<!-- Original SHA256: ce59c0bcdc3ed87c2888d0a66371488ccf8438377d962446f2930c6b4ed355ec; original line count: 1654. -->

# Codex Sandbox Session Summary

## 2026-04-24 Architecture Pass: Consistency And Hot-Path Audit

### Confirmed Issues Found

- **Directional centroid order bug** had existed in the new topology helpers:
  - `SceneObject.centroid` and `TrackRecord.centroid` are `(x, y)`
  - `_directional_target_object(...)` and `_directional_topology_bonus(...)` had initially unpacked them as `(y, x)`
  - this was a real correctness issue, not just trace noise
- **Compact trace inconsistency**:
  - `recovery_penalty` and `escalation_bonus` were missing from compact score serialization
  - this made recovery/escalation traces harder to interpret than the full in-memory score dict
- **Directional topology bonus double-counting**:
  - `generate_candidates(...)` already injects `directional_topology_bonus` into proposal scores so good movement actions survive top-k clipping
  - `score_candidates(...)` was then adding the same bonus again for non-click actions
  - because proposal score already flows into `_score_candidates_core(...)`, this was a real scoring inconsistency
- **Redundant topology recomputation**:
  - `generate_candidates(...)` computed the same free-space topology twice on the same `(frame, scene, planning_table)`
  - `_symbolic_transition_summary(...)` also recomputed the **current** topology once per candidate even though it is constant across the whole scoring batch

### Fixes Made

- Fixed the `(x, y)` centroid-order bug in both directional helpers.
- Added a focused regression test:
  - `test_directional_target_object_uses_xy_centroid_order`
- Added missing compact-trace scalar fields:
  - `recovery_penalty`
  - `escalation_bonus`
  - `directional_topology_bonus`
- Removed directional-topology double counting:
  - the bonus still shapes proposal scores in `generate_candidates(...)`
  - it is still recorded in score components for tracing
  - but it is no longer added a second time in `score_candidates(...)`
- Added a regression test:
  - `test_directional_topology_bonus_is_not_double_counted_in_final_scoring`
- Reused current-topology results instead of recomputing them redundantly:
  - one topology per `generate_candidates(...)` call
  - one current-topology per `score_candidates(...)` batch
  - `_symbolic_transition_summary(...)` now accepts optional `current_topology`

### Verification

- Focused suite passes:
  - `/home/moloch/ouro_project/venv/bin/python -m pytest codex_sandbox/test_codex_sandbox.py codex_sandbox/test_codex_integration.py`
  - Result: `33 passed`
- A fresh post-XY-fix `wa30` 90-step rerun completed before the no-double-count patch:
  - dump: `codex_sandbox/perf_event_dumps_topology_bonus_trace_cpu_smoke_xyfix/wa30/run_1.json`
  - score: `0.0`
  - change rate: `0.730`
  - interpretation:
    - the centroid-order correction did not regress the planner into inactivity
    - but it also did not solve `wa30` on its own
- A fresh post-no-double-count smoke completed:
  - `codex_sandbox/checkpoints_arc_topology_bonus_trace_cpu_smoke_xyfix_nodouble`
  - `codex_sandbox/perf_event_dumps_topology_bonus_trace_cpu_smoke_xyfix_nodouble`
  - score: `0.0`
  - change rate: `0.764`
  - important trace confirmation:
    - directional candidates now show `proposal_score ~= directional_topology_bonus`
    - this is what we want after the accounting fix, because the topology term should shape proposal ranking once, not be added again later
  - at the first recovery step (`65`), the top movement candidates showed:
    - action `1`: proposal `-0.0399`, topology `-0.0399`
    - action `2`: proposal `+0.1011`, topology `+0.1011`
    - action `3`: proposal `+0.0682`, topology `+0.0682`
    - action `4`: proposal `+0.1319`, topology `+0.1319`
  - by step `90`, the agent was already in escalation and still remained movement-active

### Current Read

- The most important architecture mistakes found in this pass were **not** broad roadmap issues.
- They were local but meaningful:
  - coordinate-order correctness
  - score-term accounting correctness
  - hot-path redundant topology work
- The broader roadmap conclusion still stands:
  - no game-specific logic was added
  - planner work remains generic/topology/object-memory based
  - the remaining open problem is constructive post-template control on `wa30`, not missing sandbox hygiene

## 2026-04-24 `wa30` Root-Cause Update: Branch Trust And Destructive Reset

### Diagnosis Update

- The older `wa30` story ("post-template fallback is weak") was only partially true.
- The stronger diagnosis after the latest trace work is:
  - **branch-failure credit was too sticky**
  - **and stale-branch handling was too destructive**
- Concretely:
  - same-game phase-action sources accumulate `branch_failures`
  - before this pass, they did **not** earn trust back when template-backed actions kept producing real frame changes
  - when a stale pocket finally triggered `_abandon_phase_action_branch(...)`, the code reset `phase_progress` to `0`
- On a long directional level like `wa30` level 1, that is the wrong behavior:
  - a few old misses can poison the only coherent same-game source
  - and when the branch stalls locally, resetting the whole level plan to phase `0` throws away valid accumulated context

### Why This Matters For `wa30`

- Trusted `wa30` level 1 is long:
  - the solved trajectory reaches level `1` completion around step `124`
- The old planner state was effectively too impatient for that:
  - earlier runs lost same-game action authority around phase `17`
  - then fell into recovery / escalation / reseed
  - or, after the first trust-decay patch, looped back into early-sequence actions because phase progress had been reset to `0`

## 2026-04-24 Phase-Branch Success Decay

### Change Made

- Added `_reward_phase_action_branch_success(...)`.
- Behavior:
  - when a chosen `phase_action_template` leads to a real frame change or stronger progress signal, the source's prior `branch_failures` are decayed
  - this keeps the "retire genuinely bad branches" mechanism
  - but stops old failures from living forever while the same source continues to be productive

### Verification

- Added focused regression test:
  - `test_phase_action_branch_success_decays_failure_credit`
- Focused suite passes:
  - `/home/moloch/ouro_project/venv/bin/python -m pytest codex_sandbox/test_codex_sandbox.py codex_sandbox/test_codex_integration.py`
  - Result after this and the follow-up fix below: `43 passed`

### Intermediate Live Result

- 90-step smoke:
  - dump: `codex_sandbox/perf_event_dumps_phase_branch_decay_cpu_smoke/wa30/run_1.json`
  - score: `0.0`
  - change rate: `0.652`
- What it proved:
  - template authority no longer died at phase `17`
  - but the run exposed a second problem:
    - after a stale pocket, the planner restarted at phase `0`
    - then replayed early same-game sequence fragments instead of recovering locally
- Trace signature:
  - `n_phase_steps = 90`
  - same-game action templates persisted through the whole run
  - but around step `65` the planner restarted near phase `0` with `branch_failures = 1.0`

## 2026-04-24 Local Recovery Instead Of Full Phase Reset

### Change Made

- Updated `_abandon_phase_action_branch(...)`.
- Old behavior:
  - record source failure
  - set `phase_progress[level] = 0`
- New behavior:
  - record source failure
  - keep the level anchored at the stalled local phase
  - activate bounded `recovery` there instead of wiping the whole level plan

### Rationale

- This is generic planner behavior, not ARC-specific:
  - a local option stall should trigger nearby search, not total amnesia
  - especially on long same-game directional sequences

### Verification

- Added/updated focused tests:
  - `test_abandon_phase_action_branch_preserves_local_phase_and_enters_recovery`
  - `test_stale_phase_action_branch_preserves_local_phase_and_enters_recovery`
- Focused suite passes:
  - `/home/moloch/ouro_project/venv/bin/python -m pytest codex_sandbox/test_codex_sandbox.py codex_sandbox/test_codex_integration.py`
  - Result: `43 passed in 3.05s`

### Live Result

- 90-step smoke:
  - dump: `codex_sandbox/perf_event_dumps_phase_local_recovery_cpu_smoke/wa30/run_1.json`
  - score: no level complete event in dump; effectively still `0.0`
  - change rate: `0.756`
  - deaths: `0`
- This is the strongest `wa30` planner result in this pass even though it still does not solve:
  - same-game phase-action guidance survives through the full 90-step window
  - it no longer collapses around phase `17`
  - it no longer snaps back to phase `0`
  - by step `89`, the chosen candidate is still same-game phase-guided
  - the trace reaches approximately phase `85/86`
  - `branch_failures` remain at `0.0` through the late same-game segment
  - the first visible `recovery_action_option` only appears at step `90`

### Current Read

- This changes the `wa30` diagnosis materially:
  - the planner can now carry constructive same-game option authority much deeper into the level
  - the main blocker is no longer the old early phase collapse
  - the next decisive question is whether a longer run now converts that deeper phase carryover into actual level completion

## 2026-04-24 Longer Follow-Up Run

### Status

- A deterministic longer follow-up run was launched to test whether the new local-recovery behavior clears `wa30` level 1 once the run exceeds the old 90-step ceiling.
- Target directories:
  - `codex_sandbox/checkpoints_arc_phase_local_recovery_long_cpu_smoke`
  - `codex_sandbox/checkpoints_arc_phase_local_recovery_long2_cpu_smoke`
  - corresponding `perf_event_dumps_*` directories
- At the time of this summary update, no completed longer-run dump was available yet, so no trustworthy longer-horizon result is recorded here.

## 2026-04-24 Escalation Context Gating

### Why This Was Next

- The corrected `wa30` traces showed:
  - recovery was already reasonably topology-aligned
  - escalation was still too willing to reward *least-repeated* directions even when their topology support was weak or negative
- This was a genuine planner issue:
  - not hardcoding
  - not missing data
  - not a trace artifact
- In other words, escalation had diversity, but not enough directional plausibility.

### Change Made

- Added `_phase_escalation_context_bonus(...)`.
- Design:
  - keep the existing action-diversity bonus from `_phase_escalation_action_bonus(...)`
  - but scale that bonus by current directional topology support
  - weak/negative topology support strongly damps escalation novelty
  - strong positive support preserves or slightly boosts escalation novelty
- Important constraint:
  - this does **not** reintroduce raw topology double-counting
  - the topology term still enters proposal scoring once
  - the new logic only gates the *escalation* bonus using that signal

### Verification

- Added regression test:
  - `test_phase_escalation_context_bonus_downweights_topology_opposed_moves`
- Focused suite passes:
  - `/home/moloch/ouro_project/venv/bin/python -m pytest codex_sandbox/test_codex_sandbox.py codex_sandbox/test_codex_integration.py`
  - Result: `34 passed`

### Live Status

- A fresh `wa30` smoke was launched into:
  - `codex_sandbox/checkpoints_arc_topology_bonus_trace_cpu_smoke_xyfix_nodouble_ctxesc`
  - `codex_sandbox/perf_event_dumps_topology_bonus_trace_cpu_smoke_xyfix_nodouble_ctxesc`
- The first run attempt behaved oddly at the process/session layer and did not leave a completed dump within the current pass, so no trustworthy new empirical number is recorded yet.

### Current Read

- This was the right next planner change:
  - it keeps escalation exploratory
  - but stops it from overriding the generic topology model with flat action novelty
- If this helps, the expected trace signature is:
  - fewer escalation picks with negative/near-zero topology support
  - more escalation picks among topology-backed directional candidates

## 2026-04-24 Trace Completeness Fix

### Problem Found

- The compact action trace only stored the first four ranked candidates as `top_candidates`.
- In beam-search segments, the actually chosen action could fall outside that slice.
- That made post-hoc diagnosis ambiguous:
  - the dump showed the top-ranked shortlist
  - but not always the candidate the planner ultimately executed

### Fix Made

- Updated action-trace emission so each step now stores:
  - `top_candidates` = compact top-4 slice for readability
  - `chosen_candidate` = the compact entry for the actual executed action/click tuple
- Fallback behavior:
  - if the chosen tuple somehow cannot be matched, the first compact candidate is used as a safe fallback rather than leaving the field absent

### Verification

- Added focused regression test:
  - `test_action_trace_keeps_chosen_candidate_even_if_not_in_top_slice`
- Focused suite passes:
  - `/home/moloch/ouro_project/venv/bin/python -m pytest codex_sandbox/test_codex_sandbox.py codex_sandbox/test_codex_integration.py`
  - Result: `35 passed`

### Why It Matters

- This does not change planner behavior.
- It removes a real observability gap that was blocking clean analysis of recovery/escalation behavior, especially when action `5` or a deeper beam-selected move was chosen outside the visible top-4 shortlist.

## 2026-04-24 Recovery / Escalation Failure Memory

### Why This Was Next

- After the topology-accounting and escalation-context fixes, the remaining `wa30` pattern was:
  - post-template behavior was active
  - but recovery/escalation still revisited low-value actions too easily
  - mere frame change was being treated too similarly to constructive progress
- This is a Sprint 5/6 issue:
  - topology exists
  - planner state machine exists
  - but the agent still needed short-horizon memory of which actions had just failed *in this phase window*

### Change Made

- Added generic per-level failure memory for:
  - `recovery`
  - `escalation`
- The memory is action-index keyed and updates from the existing `progress_signal`.
- First version:
  - repeated low-progress actions accumulated stronger penalties / bonus damping
- Follow-up softening:
  - `frame_changed=False` increments failure memory strongly
  - `frame_changed=True` but `progress_signal` low increments it only weakly
- Architectural intent:
  - distinguish true no-ops from world-changing but not-yet-productive actions
  - keep this generic, not ARC-specific

### Lifecycle / Consistency Fixes

- Wired the new phase-failure state through:
  - `__init__`
  - `reset_for_new_game(...)`
  - `on_level_complete(...)`
  - checkpoint save/load
- This fixed a real consistency gap: behavior had the new state before the object was fully declaring and persisting it.

### Verification

- Added focused tests:
  - `test_phase_recovery_action_penalty_tracks_unproductive_action_memory`
  - `test_phase_escalation_bonus_tracks_unproductive_action_memory`
  - `test_phase_recovery_failure_memory_is_softer_for_world_changing_actions`
  - `test_phase_escalation_failure_memory_is_softer_for_world_changing_actions`
- Focused suite passes:
  - `/home/moloch/ouro_project/venv/bin/python -m pytest codex_sandbox/test_codex_sandbox.py codex_sandbox/test_codex_integration.py`
  - Result: `39 passed`

### Live Results

- First failure-memory run:
  - dump: `codex_sandbox/perf_event_dumps_phase_failmem_cpu_smoke/wa30/run_1.json`
  - score: `0.0`
  - change rate: `0.742`
- Softened failure-memory run:
  - dump: `codex_sandbox/perf_event_dumps_phase_failmem_soft_cpu_smoke/wa30/run_1.json`
  - score: `0.0`
  - change rate: `0.753`

### Comparative Read

- Against the no-double-count baseline:
  - recovery repeats dropped from `4` → `2`
  - recovery chosen topology mean improved slightly from `0.0868` → `0.0879`
- Against the first blunt failure-memory version:
  - recovery became less punitive (`rec_mean -0.3059` → `-0.1651`)
  - recovery stayed more movement-constructive (`changes 14` → `15`)
  - escalation removed repeat actions again (`1` → `0`)
- Bottom line:
  - the softened version is better than the first hard failure-memory pass
  - it preserves the anti-loop effect without suppressing world-changing actions as aggressively
  - but `wa30` is still unsolved, so the planner frontier remains open

## 2026-04-24 Attention Residuals (arXiv:2603.15031) Assessment

### Source

- Primary source used:
  - `arXiv:2603.15031` — *Attention Residuals*

### Paper Claim

- The paper argues that standard PreNorm residual accumulation uses fixed unit-weight aggregation across depth, causing hidden-state growth and contribution dilution.
- It replaces that with learned, content-dependent depth-wise aggregation over prior layer outputs.
- A scalable `Block AttnRes` variant reduces the cost of full across-depth attention.

### Judgment For This Project

- Interesting and plausibly useful **for backbone architecture work**.
- Not the right next change for current Sprint 5/6 sandbox work.

### Why

- Current active work is on:
  - sandbox planner logic
  - topology-guided control
  - symbolic transition summaries
  - option/recovery/escalation/reseed behavior
- `Attention Residuals` instead targets:
  - the transformer backbone’s internal residual routing
  - hidden-state dilution with depth
  - large-scale pretraining efficiency/quality

### Practical Fit

- For the current project, it is most relevant to:
  - a future non-frozen backbone experiment
  - or a successor to the frozen Ouro substrate
- It is **not** a drop-in improvement for the current sandbox ARC planner.
- If adapted conceptually at all in the present architecture, the closer analogue would be:
  - learned depth-wise mixing over loop states / layer summaries
  - not rewriting the current planner around it

### Bottom Line

- Worth remembering for later backbone research.
- Not the bottleneck for current Sprint 5/6 progress.

## 2026-04-24 Post-Escalation Reseed Follow-up

### Problem Found

- The new `phase_reseed` state activated correctly after recovery and escalation on `wa30`, but it initially had no effect on scoring.
- Live encoder-only `wa30` smoke (`codex_sandbox/perf_event_dumps_branch_reseed_cpu_smoke/wa30/run_1.json`) showed:
  - recovery segment: steps `65-81`
  - escalation segment: steps `82-96`
  - reseed segment: steps `97-109`
- However, every reseed-step top candidate still had:
  - `phase_action_bonus = 0.0`
  - no `phase_action_template`
- Root cause:
  - `wa30` has only one action-template source at level 1: `wa30_run0_traj.npz`
  - by the time reseed activates, that source is already marked exhausted with branch failure count `3.0`
  - the first reseed implementation still filtered exhausted action-template sources, so the reseed window reopened no action options at all

### Fix Made

- Updated `arc_agent_hunter_seeker_codex.py` so `phase_reseed` can reopen a retired action-template source as a **very weak hint** when no non-exhausted source exists for that phase window.
- Important constraint preserved:
  - reseed does **not** restore hard branch authority
  - it only allows low-confidence `reseed_action_option` hints
  - retired sources are explicitly tagged with `retired_source = True`
  - confidence is capped lower than ordinary reseed hints (`<= 0.08 / (1 + phase_offset)`)

### Verification

- Added focused regression test:
  - `test_phase_reseed_can_reopen_retired_action_source_as_weak_hint`
- Focused suite now passes:
  - `/home/moloch/ouro_project/venv/bin/python -m pytest codex_sandbox/test_codex_sandbox.py codex_sandbox/test_codex_integration.py`
  - Result: `29 passed in 2.88s`

### Current State

- The planner state machine now has three bounded post-template phases:
  - recovery
  - escalation
  - reseed
- The structural `wa30` blocker that left reseed empty has been removed.
- A full follow-up `wa30` smoke was started into:
  - `codex_sandbox/checkpoints_arc_branch_reseed_bonus_cpu_smoke`
  - `codex_sandbox/perf_event_dumps_branch_reseed_bonus_cpu_smoke`
- That run did not produce a completed dump within the session window, so no fresh performance number is recorded yet.

## 2026-04-24 Planning-Copy Optimization

### Claude Note Assessment

- Claude's main performance claim was correct:
  - `_symbolic_transition_summary(...)` was paying for full `ObjectTable` deep copies in hot planning paths
  - this is plausibly a real throughput drag, especially under beam expansion
- I do **not** think it was the primary behavioral blocker.
  - The option/recovery/reseed logic issues were more important for actual policy construction.
  - But this copy cost was still worth fixing because it slows iteration and likely reduces effective search throughput.

### Fix Made

- Added explicit lightweight planning clones in `arc_agent_hunter_seeker_codex.py`:
  - `ObjectRecord.planning_copy()`
  - `TrackRecord.planning_copy()`
  - `ColorPriorTable.planning_copy()`
  - `ObjectTable.planning_copy()`
- Replaced the hot-path full copies with `planning_copy()` in:
  - `generate_candidates(...)`
  - `_symbolic_transition_summary(...)`
  - the pre-step `table_before` snapshot used for symbolic transition measurement
- Design choice:
  - did **not** override global `copy.deepcopy(...)` semantics for `ObjectTable`
  - kept this optimization explicit and local to planning paths
  - preserves safety around real-transition bookkeeping and tests that still expect normal copy semantics elsewhere

### What The Lightweight Copy Keeps

- Track identity and geometry:
  - `track_id`, `color`, `centroid`, `area`, `age`, `miss_count`, `velocity`
- Belief state needed for symbolic/topology reasoning:
  - `_weights`, `n_interactions`, `tested_actions`, `last_centroid`, `sterile_count`
- Color prior arrays for new-track initialisation during hypothetical updates

### What It Drops

- Heavy interaction-history deques:
  - `effect_history`
  - `sharpening_history`
- Event-log / real-transition snapshot state:
  - not needed for hypothetical successor summaries

### Verification

- Added focused regression test:
  - `test_object_table_planning_copy_preserves_beliefs_without_aliasing`
- Focused suite now passes:
  - `/home/moloch/ouro_project/venv/bin/python -m pytest codex_sandbox/test_codex_sandbox.py codex_sandbox/test_codex_integration.py`
  - Result: `30 passed in 2.85s`

### Quick Benchmark

- Small synthetic microbenchmark on a populated 5-track table:
  - `copy.deepcopy(object_table)` ≈ `0.098 ms`
  - `object_table.planning_copy()` ≈ `0.070 ms`
  - speedup ≈ `1.41x`
- This benchmark is smaller/cleaner than real ARC planning loads, so it should be treated as directional only.
- The reason it still matters:
  - the hot path performs many such copies per step under beam search
  - savings should compound with candidate count and track count

## 2026-04-24 Topology-Directed Directional Scoring

### Why This Was Next

- After recovery / escalation / reseed work, the remaining `wa30` weakness was:
  - once template authority ended, the agent still largely fell back to generic directional ranking
  - the roadmap wanted stronger topology-guided movement, and Claude's note also correctly pointed out that explicit directional BFS/path steering was still thin
- User constraint remains explicit:
  - no game-specific hardcoding
  - ARC is only the current cradle for Hunter Seeker, not the intended terminal domain
  - all changes in this section were kept generic and domain-agnostic

### Changes Made

- Added `_directional_topology_bonus(...)` in `arc_agent_hunter_seeker_codex.py`.
- The bonus is generic and domain-agnostic. It uses only:
  - local wall / hazard / reward-like beliefs at the target step cell
  - distance reduction toward a known exit, if one exists
  - distance reduction toward topology frontier objects
  - distance reduction toward reachable reward-like objects
  - whether the step lands inside currently reachable free space
- Recovery-stage scaling:
  - modestly amplified during recovery / escalation / reseed windows
- Wired the bonus into:
  - `generate_candidates(...)` for non-click actions
  - final `score_candidates(...)` bookkeeping as `directional_topology_bonus`
- Follow-up trace fix:
  - added `directional_topology_bonus` to compact score-component serialization
  - future event dumps will now actually expose the signal during recovery / escalation / reseed analysis

### Verification

- Added focused regression test:
  - `test_directional_topology_bonus_prefers_exitward_motion`
- Extended score-component coverage so Sprint-6 component traces now expect:
  - `directional_topology_bonus`
- Focused suite passes:
  - `/home/moloch/ouro_project/venv/bin/python -m pytest codex_sandbox/test_codex_sandbox.py codex_sandbox/test_codex_integration.py`
  - Result: `31 passed in 2.90s`

### Smoke Status

- Started a fresh encoder-only `wa30` smoke into:
  - `codex_sandbox/checkpoints_arc_topology_bonus_cpu_smoke`
  - `codex_sandbox/perf_event_dumps_topology_bonus_cpu_smoke`
- The run did not leave a completed dump within the session window, so no trustworthy live `wa30` outcome is recorded yet.

### Follow-up Trace Confirmation

- A shorter trace-focused encoder-only run completed successfully:
  - dump: `codex_sandbox/perf_event_dumps_topology_bonus_trace_cpu_smoke/wa30/run_1.json`
  - score: `0.0`
  - change rate: `0.730`
- Important result:
  - `directional_topology_bonus` is now live in compact traces
  - at the recovery step that appeared in this shorter run (`step 90`), the top directional candidates had distinct topology bonuses:
    - action `1`: `+0.061`
    - action `2`: `+0.135`
    - action `3`: `-0.037`
    - action `4`: `+0.100`
  - the chosen action was `2`, i.e. the direction with the strongest positive topology bonus
- Interpretation:
  - the new generic topology-directed movement term is not dead code
  - it is already influencing directional ranking in the intended post-template regime
- Limitation:
  - in this 90-step run, template steering lasted until step `89`, so only the first recovery step was visible
  - a longer 120-step follow-up was launched to inspect the full recovery / escalation / reseed sequence with the same trace field available

## 2026-04-23

### Working Scope

- Active workspace is `~/ouro_project/codex_sandbox/`.
- Keep Codex-authored variants, diagnostics, and notes inside this folder unless explicitly asked otherwise.
- Do not modify the user's original project files while sandbox work is still experimental.
- Use v17b checkpoints as the trusted encoder/checkpoint source when resolving outdated notes in `ouro_project_state.md`.

### Reconstructed State

- The master state document is useful for architecture context, but parts are outdated.
- The decisive stale-doc item that still matters: Sprint 4 encoder drift made the trained encoder incompatible with Ouro-space.
- Sandbox code already defaults to `freeze_encoder=True` in `arc_agent_pairwise_stockfish_codex.py`.
- `checkpoints_running/sprint4_encoder_reverted.pt` exists and is likely the Sprint 4 state with a v17b encoder restored.
- Sandbox contains experimental Sprint 5 and Sprint 6 work:
  - Sprint 5 free-space topology via avatar-rooted flood fill.
  - Sprint 6 symbolic transition summaries, symbolic planner head, ranker symbolic features, online trace mode diagnostics.
  - Offline event-dump ablation and focus-game timeline/hybrid reports.

### Tests / Diagnostics Run

- Sandbox tests:
  - Command: `/home/moloch/ouro_project/venv/bin/python -m pytest codex_sandbox/test_codex_sandbox.py codex_sandbox/test_codex_integration.py codex_sandbox/test_event_dump_ablate.py codex_sandbox/test_focus_game_timeline_report.py codex_sandbox/test_online_trace_report.py`
  - Result: `16 passed in 7.10s`.
- Live mock online trace:
  - Command: `/home/moloch/ouro_project/venv/bin/python codex_sandbox/online_trace_run_report.py --episodes 3 --steps 12 --seed 0`
  - Result: live selection path is active; `beam_search=23`, `random=13`, `trace_mode_counts={"topology": 23}`, `symbolic_update_count=3`, event log populated.
- Offline Sprint 6 event-dump ablation:
  - Command: `/home/moloch/ouro_project/venv/bin/python codex_sandbox/event_dump_sprint6_ablate.py --seeds 0 1 2`
  - Result: `n_runs=124`, label counts `mechanism=80`, `planner=19`, `topology=25`.
  - Best current report: `hybrid14_mean_metrics`: accuracy `0.7634`, mechanism recall `0.8773`, planner recall `0.4167`, topology recall `0.8381`.
  - Interpretation: hybrid features improve topology and planner separation compared with expanded14, so live ARC diagnostics should focus on whether online trace labels match actual failures.

### Recommended Next Step

Run a short real ARC diagnostic, not an overnight sweep.

## 2026-04-23 Live ARC Diagnostic

### New File

- Added `live_arc_diagnostic.py`.
- Purpose: sandbox-only live ARC runner that imports `codex_sandbox.*`, not the main project agents.
- It performs selective checkpoint loading because `sprint4_encoder_reverted.pt` predates the sandbox symbolic-ranker input width.
- Compatible modules loaded fully: encoder, action prior, spatial predictor, next-frame predictor, patch color head, objectivity head, loop pooler.
- Ranker loaded partially: `10/13` checkpoint tensors matched; symbolic-input-dependent ranker layers remain freshly initialized.

### Command

```bash
/home/moloch/ouro_project/venv/bin/python \
  codex_sandbox/live_arc_diagnostic.py \
  --games ls20 ft09 \
  --checkpoint checkpoints_running/sprint4_encoder_reverted.pt \
  --max_steps 120 \
  --n_runs 1 \
  --eps 0.15 \
  --dump_events_dir codex_sandbox/live_arc_event_dumps \
  --out codex_sandbox/live_arc_diagnostic_summary.json
```

### Result

- ARC SDK attempted remote metadata fetch, failed DNS due restricted/offline environment, then successfully fell back to cached local game files:
  - `environment_files/ls20/9607627b/ls20.py`
  - `environment_files/ft09/0d8bbf25/ft09.py`
- Summary written to `codex_sandbox/live_arc_diagnostic_summary.json`.
- Event dumps written to:
  - `codex_sandbox/live_arc_event_dumps/ls20/run_1.json`
  - `codex_sandbox/live_arc_event_dumps/ft09/run_1.json`

### ls20

- Ran full `120` steps, terminal state `GameState.NOT_FINISHED`, `levels_completed=0`.
- Selection counts: `beam_search=104`, `random=16`.
- Trace modes: `topology=104`, `unknown=16`.
- Event log: `94` events: moved `34`, contact `57`, appeared `2`, transformed `1`.
- Failure summary empty because no terminal death occurred within 120 steps.
- Interpretation: online trace strongly classifies ls20 as topology during live play, matching the known topology-game framing, but no death/failure anchor was reached in this short run.

### ft09

- Ended at step `39` with `GameState.GAME_OVER`, `levels_completed=0`.
- Selection counts: `beam_search=35`, `random=4`.
- Trace modes: `mechanism=35`, `unknown=4`.
- Failure summary: `mechanism=1`.
- Event log: `91` events: appeared `6`, contact `32`, death `1`, disappeared `31`, moved `21`.
- Death attribution: click death on color `9`, track `24`, marked hazard with `hazard=0.97`, `unknownness=0.08`.
- Interpretation: live trace mode and terminal failure type agree on mechanism. This is a useful positive signal that the online Sprint 6 trace classifier is not just post-hoc fitting event dumps.

### Next Step

- Run the same diagnostic on `wa30` and `r11l`.
- `wa30` should be another topology probe with a terminal anchor if it dies quickly.
- `r11l` should test the mechanism/planner split on a click game with known prior ambiguity.
- If those agree, the next implementation step is to make the live diagnostic output a compact CSV/table view and then decide whether to port the sandbox runner logic into the main harness or keep it isolated for another round.

## 2026-04-23 Performance Work

### Training Harness

- Copied the root ARC training harness into `codex_sandbox/train_arc_codex.py`.
- Patched it to import sandbox agents/adapters and default all outputs to sandbox paths.
- Added partial checkpoint loading so older v17b/Sprint4 checkpoints can initialize matching modules while leaving the newer symbolic-ranker input layers fresh.

### Fixes Made

- `arc_agent_pairwise_stockfish_codex.py`: fixed frozen-encoder warmup gating. With `freeze_encoder=True`, encoder update count never advances, so the ranker/action-prior/depth logic was previously blocked from training/acting as intended.
- `arc_agent_hunter_seeker_codex.py`: fixed `ObjectTable()` construction by passing `ColorPriorTable()`.
- `arc_agent_hunter_seeker_codex.py`: strengthened hazard click scoring with a larger hazard type penalty and a continuous safety penalty based on track hazard/wall belief.

### Verification

- Sandbox test suite before training work: `16 passed`.
- Targeted sandbox tests after warmup/ObjectTable fixes: `12 passed`.
- Targeted sandbox tests after safety-scoring patch: `12 passed`.

### Actual Performance Runs

- No-replay training/eval on `ft09 r11l` from `checkpoints_running/sprint4_encoder_reverted.pt`, with solved-sequence pretraining, still scored `0`:
  - `ft09`: died at steps `41` and `37`, both mechanism deaths.
  - `r11l`: died at step `60` in both runs, mechanism deaths.
- Safety-scoring continuation run still scored `0`:
  - `ft09`: died at steps `35` and `39`.
  - `r11l`: died at step `60` in both runs.
- Replay-enabled run from `codex_sandbox/checkpoints_running/perf_ft09_r11l.pt` produced nonzero score via solved prefixes:
  - `ft09` run 1: completed level 1 at step `43`, died at step `75`, score `4.761904761904762`.
  - `ft09` run 2: completed level 1 at step `43`, died at step `75`, score `4.761904761904762`.
  - `r11l` run 1: completed level 1 at step `4`, died at step `64`, score `4.761904761904762`.
  - `r11l` run 2: died at step `60` without completing a level in that episode, reported cumulative score `4.761904761904762`.

### Current Interpretation

- Actual combined capability is nonzero when replay/prefixes are allowed.
- Novel no-replay policy performance is still the blocker: it identifies hazards, but lethal click candidates still survive selection.
- The next code-level move should be candidate-generation safety, not more diagnostics or only ranker scoring:
  - suppress high-hazard/high-wall click candidates before top-k truncation when safer candidates exist;
  - inject safe object-centroid alternatives from unknown/non-hazard/reward-like tracks;
  - log per-candidate belief/safety components in search traces to verify that the selected action was truly considered safe.

### Follow-up Candidate Safety Patch

- Added upstream click-target safety helpers in `arc_agent_hunter_seeker_codex.py`:
  - `_click_target_safety(...)`
  - `_is_suppressed_click_target(...)`
- `generate_candidates(...)` now injects visible non-suppressed track centroids as safe alternatives before top-k clipping.
- Known hazard/wall click targets are ordered after safe candidates before `click_candidates` truncation.
- Tightened hazard suppression so `hazard >= 0.45` suppresses the target even when the same track also looks exit-like. This addresses the `r11l` exit-vs-hazard ambiguity.

### Verification After Candidate Safety Patch

- Targeted tests still pass:
  - `/home/moloch/ouro_project/venv/bin/python -m pytest codex_sandbox/test_codex_sandbox.py codex_sandbox/test_codex_integration.py`
  - Result: `12 passed`.
- Short no-replay eval from `codex_sandbox/checkpoints_running/perf_ft09_r11l.pt`, after safety filter:
  - `ft09`: died step `38`, score `0`.
  - `r11l`: died step `60`, score `0`.
- Pretrained no-replay eval from v17b/Sprint4 checkpoint with solved trajectories and the safety filter:
  - pretraining loaded `2876` trusted transitions from `26` solved trajectories and ran `40` iterations;
  - `ft09`: died step `44`, score `0`;
  - `r11l`: died step `60`, score `0`.
- Focused `r11l` no-replay eval after unconditional hazard suppression:
  - died step `60`, score `0`.

### Updated Diagnosis

- The patch is structurally correct but insufficient for no-replay performance.
- The remaining failure is earlier than final scoring: the agent repeatedly clicks mechanism-critical objects and only becomes confident they are hazards at death time.
- In `r11l`, the same color/object can become exit-like and hazard-like, so color/track belief alone is not enough. The policy needs phase/context-conditioned mechanism state or replay/prefix execution to avoid treating a previously productive click target as safe in a later state.
- Immediate useful next move: add chosen-action trace logging for pre-click safety state and candidate list, then implement a per-level/per-phase hazard memory or cooldown for repeated high-frequency click targets that have stopped producing progress.

## 2026-04-23 Continued Performance Work

### Changes Made

- Added per-target click memory in `arc_agent_hunter_seeker_codex.py`.
  - Keyed by `(level, color, coarse_x, coarse_y)` so the same color can still be useful elsewhere.
  - Tracks attempts, sterile attempts, productive attempts, and cooldown expiry.
  - Death gives a strong cooldown; repeated nonproductive clicks give a weaker cooldown.
  - Cooldown penalty is applied both during candidate generation and final candidate scoring.
- Added durable chosen-action trace logging.
  - Event dumps now include `action_trace`.
  - Each trace entry records selected action, trace mode, chosen click safety, cooldown penalty, target stats, and top candidate score components.
- Added no-object click suppression/penalty.
  - Empty/no-track click targets are suppressed when alternatives exist and receive a final scoring penalty.
- Fixed evaluation epsilon handling in `train_arc_codex.py`.
  - Removed the hard floor `max(0.10, ...)`.
  - `--eps 0.0` now actually runs deterministic/no-random evaluation.
- Added `click_target_stats` to Hunter Seeker checkpoints so cooldown safety memory survives between commands.

### Verification

- Syntax check:
  - `/home/moloch/ouro_project/venv/bin/python -m py_compile codex_sandbox/train_arc_codex.py codex_sandbox/arc_agent_hunter_seeker_codex.py`
  - Passed.
- Targeted tests:
  - `/home/moloch/ouro_project/venv/bin/python -m pytest codex_sandbox/test_codex_sandbox.py codex_sandbox/test_codex_integration.py`
  - Result: `12 passed`.

### Performance Checks

- `r11l`, no-replay, 2 runs, epsilon floor still present at `0.10`, with cooldown:
  - run 1: died step `60`, score `0`;
  - run 2: game over step `60`, score `0`;
  - behavior changed away from the original repeated color-2 pattern, but no level completion.
- `r11l`, no-replay, action-trace verification:
  - event dump contains `action_trace` with `60` step entries.
  - trace showed cooldown can reach `-3.0`, but when every candidate is poor the planner still picks a cooled target.
- `r11l`, no-replay, no-object suppression, epsilon `0.10`:
  - run 1: died step `60`, score `0`;
  - run 2: died step `60`, score `0`.
- `r11l`, no-replay, deterministic `--eps 0.0` after epsilon-floor fix:
  - run 1: died step `60`, score `0`;
  - run 2: died step `60`, score `0`.

### Current Diagnosis

## 2026-04-23 Branch Planner Follow-up

### Branch-Consistency Work

- `arc_agent_hunter_seeker_codex.py` now keeps directional phase options on a single solved branch per level instead of mixing per-phase votes across different trusted runs.
- Added branch failure accounting and persisted `phase_action_branch_failures` in running checkpoints.
- Added cross-level preferred-source carryover so a game that solves one level on a given trusted run will prefer that same source on the next level.
- Fixed the one-step lag after `on_level_complete`, which had been causing the first action of the next level to miss its phase template.
- Added `chosen_phase_action_template` to `action_trace` entries so event dumps show exactly which solved template branch was used.
- Added a generic stale-branch restart guard: after repeated sterile non-click actions on the same branch, the level phase is reset and the branch is marked failed instead of blindly advancing the phase.
- Added a generic branch-choice guard in `choose_branch(...)`: when candidate solved branches disagree, exact action agreement at the current phase now outranks deeper minority branches. This is not `ls20`-specific; it uses majority action support plus branch-failure count before depth/confidence tie-breakers.

### Regression Coverage

- Focused tests now cover:
  - branch consistency within a level
  - preferred-source carryover across levels
  - immediate level bookkeeping after `on_level_complete`
  - stale branch restart behavior
  - majority-action branch preference before depth
- Verification:
  - `py_compile` passed
  - `pytest codex_sandbox/test_codex_sandbox.py codex_sandbox/test_codex_integration.py` passed with `17 passed`
  - report tests remained green with `4 passed`

### Live CUDA Results

- `tr87` smoke after branch carryover + level-lag fix:
  - completed levels 1, 2, 3 at steps `37`, `67`, `106`
  - score `28.571428571428573`
- Combined trusted trajectory source created at `codex_sandbox/trusted_plus_expanded/`, merging `trusted_trajs/` with improved `tr87` trajectories from sandbox runs.
- CUDA sweep on `ls20 tr87 wa30` using the merged trusted set:
  - `tr87`: completed levels 1, 2, 3, 4 at steps `37`, `67`, `106`, `135`; score `47.61904761904762`
  - `wa30`: completed level 1 at step `150`; score `0.4978765432098765`
  - `ls20`: no completion in that sweep; source cycling exposed a branch-selection problem rather than a representation problem
- CUDA smoke after the stale-branch restart guard:
  - `tr87` stayed at 4 completed levels
  - `ls20` still failed to complete, but change rate improved, confirming the guard changed behavior without solving the initial branch pick
- CUDA smoke after the majority-action branch-vote guard:
  - command used `codex_sandbox/checkpoints_arc_branch_vote_guard_cuda_smoke` and `codex_sandbox/perf_event_dumps_branch_vote_guard_cuda_smoke`
  - `ls20` completed level 1 at step `13`
  - final score `3.5714285714285716`
  - saved:
    - `codex_sandbox/solved_sequences_expanded/ls20_level1_run1.json`
    - `codex_sandbox/solved_sequences_expanded/ls20_run1_traj.npz`

### Interpretation

- `tr87` is now firmly in the "planner helped" column: the branch planner stopped splicing solved runs and started carrying a coherent source across levels.
- `wa30` tolerated the same planner changes and produced a level completion.
- `ls20` only started moving once branch choice stopped favoring a deeper minority trajectory over the two shorter trusted runs that agreed on the opening actions.
- The remaining planner work should stay generic:
  - keep branch selection biased toward strong local action agreement
  - only switch branches after explicit failure evidence
  - avoid game-specific hacks

### Broader CUDA Sweep After Branch-Vote Guard

- Command outputs:
  - checkpoints: `codex_sandbox/checkpoints_arc_branch_vote_guard_cuda_sweep2`
  - dumps: `codex_sandbox/perf_event_dumps_branch_vote_guard_cuda_sweep2`
- Results:
  - `ls20`: completed levels 1, 2, 3 at steps `13`, `136`, `175`; score `21.428571428571427`
  - `tr87`: completed levels 1, 2, 3, 4 at steps `37`, `67`, `106`, `135`; score `47.61904761904762`
  - `wa30`: completed `0` levels; score `0.0`
- New saved trajectories:
  - `codex_sandbox/solved_sequences_expanded/ls20_level1_run2.json`
  - `codex_sandbox/solved_sequences_expanded/ls20_run2_traj.npz`
  - `codex_sandbox/solved_sequences_expanded/tr87_level3_run3.json`
  - `codex_sandbox/solved_sequences_expanded/tr87_run3_traj.npz`

### Dump Review

- `ls20`:
  - `3` level-complete events in the dump.
  - Action trace sources were mostly coherent on `ls20_run2_traj.npz` (`167` traced actions), with only a short opening contribution from `ls20_run1_traj.npz` (`13` traced actions).
  - This is a large shift from the earlier source-cycling failure mode.
- `tr87`:
  - `4` level-complete events.
  - All traced phase-action selections came from `tr87_run1_traj.npz` (`180` traced actions).
  - Confirms the branch planner is now stable on the strongest solved source for that game.
- `wa30`:
  - `0` level-complete events.
  - Still stayed coherent on `wa30_run0_traj.npz` for all `180` traced actions.
  - This suggests the remaining failure on `wa30` is not source splicing; it is more likely a bad branch progression / branch-abandon decision inside one long coherent source.

### Updated Direction

- The branch-vote guard appears to have solved the initial branch-pick problem on `ls20` without harming `tr87`.
- `wa30` is now the best generic planner target because it already stays on one source; the next move should examine when a coherent branch should be abandoned or re-phased, not how to pick an initial source.

## 2026-04-23 Exhausted-Branch Guard

### Goal

- Keep the planner generic and reusable beyond ARC.
- Once an option/source has been retried and failed repeatedly at the same level, stop giving it hard steering authority.
- This is option-management logic, not domain-specific game knowledge.

### Changes Made

- Added helper methods in `arc_agent_hunter_seeker_codex.py`:
  - `_phase_action_branch_failures_for(level, source)`
  - `_phase_action_branch_is_exhausted(level, source)`
- Exhausted branches now drop out of:
  - active-source reuse
  - preferred-source reuse
  - new branch selection in `choose_branch(...)`
  - exact solved-template fallback
  - future solved-action options
- Current threshold is generic and local to planner memory: if the same source has failed `3` times at the same level, the planner stops using that source for action steering on that level.

### Regression Coverage

- Added `test_phase_action_exhausted_branch_stops_steering_without_alternative`.
- Focused tests now pass with `18 passed`.

### CUDA Smoke Result

- New host-GPU smoke:
  - checkpoints: `codex_sandbox/checkpoints_arc_branch_exhaust_guard_cuda_smoke`
  - dumps: `codex_sandbox/perf_event_dumps_branch_exhaust_guard_cuda_smoke`
- `wa30` result:
  - levels completed: `0`
  - score: `0.0`
  - change rate: `0.520` (down from `0.918` in the prior branch-vote sweep)

### Dump Interpretation

- `wa30` action trace length: `180`
- Steps with a chosen phase-action template: `64`
- All template-steered steps still came from `wa30_run0_traj.npz`, but only while branch failures were `0.0`, `1.0`, or `2.0`.
- After the source became exhausted, phase-action steering disappeared from the trace.
- This confirms the new guard is functioning as intended:
  - the agent no longer overcommits indefinitely to a repeatedly failing option;
  - the remaining weakness is what the planner does after retiring an option.

### Updated Direction

- The next generic planner step should add a recovery mode after option retirement, for example:
  - maintain a lightweight progress score over recent transitions;
  - if retired-option fallback is not improving progress, re-seed phase search from nearby alternatives or from weak non-option policy guidance;
  - keep this framed as generic option retirement / recovery rather than ARC-specific mechanics.

## 2026-04-23 Recovery Mode After Option Retirement

### Goal

- Add a generic post-retirement planner mode rather than falling straight from option-following into undirected fallback.
- Keep it domain-agnostic:
  - no ARC-specific mechanics;
  - just option retirement, recent progress tracking, and short-horizon action diversification.

### Changes Made

- Added level-local recovery state in `arc_agent_hunter_seeker_codex.py`:
  - `phase_recovery_steps`
  - `phase_recovery_recent_actions`
  - `phase_recovery_progress_ema`
- Recovery activates automatically when the same action-branch source reaches the exhaustion threshold (`3` failures at the same level).
- While recovery is active:
  - phase target templates are suppressed for that level;
  - phase action templates are suppressed for that level;
  - repeated recent actions receive a recovery penalty in candidate scoring.
- Recovery exits automatically on meaningful progress, using a generic progress signal built from:
  - level completion
  - topology improvement
  - positive reachable-reward delta
- Recovery state is now cleared on level completion and persisted through checkpoints.

### Regression Coverage

- Added tests for:
  - exhausted branch activates recovery and stops steering
  - repeated recent actions are penalized more than alternatives during recovery
- Focused tests passed with `19 passed`.

### CUDA Smoke Result

- New host-GPU smoke:
  - checkpoints: `codex_sandbox/checkpoints_arc_branch_recovery_cuda_smoke`
  - dumps: `codex_sandbox/perf_event_dumps_branch_recovery_cuda_smoke`
- `wa30` result:
  - levels completed: `0`
  - score: `0.0`
  - change rate: `0.536`

### Comparison Against Exhausted-Branch-Only Guard

- Previous exhausted-branch-only smoke on `wa30`:
  - change rate: `0.520`
  - template-steered steps: `64 / 180`
  - post-template action mix: `3:73, 1:36, 2:7`
- Recovery-mode smoke on `wa30`:
  - change rate: `0.536`
  - template-steered steps: `64 / 180`
  - post-template action mix: `3:77, 1:30, 4:4, 2:3, 5:2`

### Interpretation

- Recovery mode is functioning:
  - the planner retires the exhausted option;
  - templates stay off during recovery;
  - fallback action selection becomes slightly more diverse.
- But the effect is still too weak to convert `wa30` into a solved level.
- The next generic planner move should likely target recovery reseeding, not just recovery diversification:
  - after retirement, branch to nearby weak alternatives or alternative abstract action modes instead of only penalizing repetition inside fallback.

## 2026-04-23 Recovery Trace Instrumentation

### Trace Fields Added

- Added explicit per-step recovery-state fields to `action_trace`:
  - `phase_recovery_active`
  - `phase_recovery_steps_left`
  - `phase_recovery_progress_ema`

### Verification

- Focused tests remained green: `23 passed`.

### CUDA Smoke Result

- New host-GPU smoke:
  - checkpoints: `codex_sandbox/checkpoints_arc_branch_recovery_trace_cuda_smoke`
  - dumps: `codex_sandbox/perf_event_dumps_branch_recovery_trace_cuda_smoke`
- `wa30` result stayed:
  - levels completed: `0`
  - score: `0.0`
  - change rate: `0.536`

### Trace Finding

- Recovery activated exactly once:
  - segment from step `65` through step `81`
  - duration `17` steps
- The last template-steered step was `64`, so recovery starts immediately after template steering ends.
- Recovery steps counted down cleanly from `17` to `1`.
- `phase_recovery_progress_ema` stayed `0.0` for the entire recovery segment.
- After step `81`, recovery ended and the planner fell back to ordinary non-recovery behavior.

### Interpretation

- Recovery is not exiting early.
- Recovery is not secretly succeeding and then getting cleared.
- The actual remaining weakness is post-recovery behavior:
  - no meaningful progress occurs during the recovery window;
  - once the window expires, fallback still does not re-acquire a productive option.
- The next generic planner step should target recovery escalation after a zero-progress recovery window, not more instrumentation.

- Random exploration was not the main issue; deterministic no-replay still fails.
- Cooldown and no-object suppression improve diagnostics and alter behavior, but they do not create a constructive action sequence.
- The trace shows the planner often reaches a state where all candidate clicks are bad/cooldowned, but ARC exposes only click action on `r11l`; without a learned phase policy or replay prefix, it still must click something and eventually dies.
- Next technical move should be constructive policy, not more penalties:
  - learn/use phase-conditioned target sequencing from solved trajectories;
  - make no-replay still allowed to use learned sequence abstractions, but not raw replay;
  - or add an options layer that predicts "which object class/phase target next" rather than scoring single-step click saliency.

## 2026-04-23 Phase-Option Sequencing

### Constraint

- User explicitly warned: no hardcoded ARC specifications.
- The implemented path follows that: no game-specific constants, no manually encoded ARC rules, and no raw solved-prefix replay in `--no_replay`.
- The new guidance is data-derived from solved trajectory files as generic phase target descriptors.

### Changes Made

- Added generic phase-option memory to `arc_agent_hunter_seeker_codex.py`.
  - `_phase_target_templates`: `{game_id -> level -> phase -> descriptors}` built from `*_traj.npz`.
  - Descriptors store clicked target color, coarse 8x8 zone, area, border flag, and whether it was a parsed scene object.
  - Raw click coordinates are not replayed; live candidates are generated from current-frame pixels/objects matching the descriptor.
- Corrected template extraction for trajectory `levels`.
  - The `.npz` stores `level_after`, so the level-completion action must be assigned to the previous level.
- Added `_phase_progress` as an option pointer.
  - It advances only when a selected click matched the current phase descriptor.
  - It is not tied to total click count anymore.
- Added `_phase_template_candidates(...)`.
  - Injects live candidates from the current phase descriptor.
  - Supports both parsed objects and no-object/background-color click targets, because some solved clicks land on pixels the scene parser treats as background.
- Phase templates are saved/loaded in checkpoints as domain memory.

### Verification

- Syntax and targeted tests after the phase-option patch:
  - `/home/moloch/ouro_project/venv/bin/python -m py_compile codex_sandbox/arc_agent_hunter_seeker_codex.py`
  - `/home/moloch/ouro_project/venv/bin/python -m pytest codex_sandbox/test_codex_sandbox.py codex_sandbox/test_codex_integration.py`
  - Result: `12 passed`.

### Performance Result

- Deterministic no-replay `r11l` with phase options:
  - Command used `--load_trajs solved_sequences --pretrain_iters 5 --eps 0.0 --no_replay`.
  - Phase target templates built: `7` descriptors from solved trajectories.
  - `r11l` run 1:
    - completed level 1 at step `4`;
    - died at step `64`;
    - score `4.761904761904762`.
  - `r11l` run 2:
    - completed level 1 at step `4`;
    - died at step `14`;
    - score `4.761904761904762`.
- Trace confirmation:
  - `codex_sandbox/perf_event_dumps_r11l_phase_options/r11l/run_1.json` has `4` phase-target-selected clicks at steps `1-4`.
  - `codex_sandbox/perf_event_dumps_r11l_phase_options/r11l/run_2.json` also has `4` phase-target-selected clicks at steps `1-4`.
  - The selected clicks are descriptor matches, not exact replay; e.g. phase 2 picked `(35,19)` vs solved JSON `(36,21)`.

### Updated Diagnosis

- Constructive sequencing fixed the level-1 no-replay problem for `r11l`.
- The remaining failure is post-level-1 generalization:
  - only level 1 has a solved phase template for `r11l`;
  - after level 1, the policy falls back to generic candidate scoring/cooldowns and dies.
- Next useful move:
  - generalize phase templates across games/levels by descriptor similarity rather than requiring same `game_id`;
  - or mine more solved/improved level templates from live successful prefixes into `solved_sequences`;
  - keep the abstraction data-derived and avoid any hand-authored ARC/game rules.

## 2026-04-23 Continuation Template Follow-up

### User Constraint

- User explicitly reiterated: no hardcoded ARC specifications.
- Work stayed data-derived:
  - no game-specific constants;
  - no manually encoded level solutions;
  - no exact solved-prefix replay under `--no_replay`.

### Changes Made

- Extended phase template mining to include lower-confidence continuation descriptors.
  - If a trajectory completed at least one level, clicks after the last completed level become `template_kind="continuation"`.
  - Continuation confidence is lower than solved template confidence.
  - These are still coarse descriptors: color, 8x8 zone, area, border flag, scene-object flag.
- Added phase-template failure memory.
  - If a selected phase template leads to death, its descriptor key gets a failure count.
  - Future matches receive a penalty.
  - Failure memory is persisted in checkpoints as `phase_template_failures`.
- Kept solved templates strong and continuation templates weak/failure-suppressible.

### Verification

- Syntax and targeted tests:
  - `/home/moloch/ouro_project/venv/bin/python -m py_compile codex_sandbox/arc_agent_hunter_seeker_codex.py`
  - `/home/moloch/ouro_project/venv/bin/python -m pytest codex_sandbox/test_codex_sandbox.py codex_sandbox/test_codex_integration.py`
  - Result: `12 passed`.

### Performance Result

- Deterministic no-replay `r11l`, continuation templates:
  - Built `30` descriptors from solved trajectories.
  - Run 1: completed level 1 at step `4`, died at step `42`, score `4.761904761904762`.
  - Run 2: completed level 1 at step `4`, died at step `64`, score `4.761904761904762`.
- Deterministic no-replay `r11l`, continuation failure suppression:
  - Run 1: completed level 1 at step `4`, died at step `42`, score `4.761904761904762`.
  - Run 2: completed level 1 at step `4`, died at step `64`, score `4.761904761904762`.

### Current Diagnosis

- Phase options are a real performance improvement over earlier no-replay score `0`.
- Continuation descriptors do not solve post-level-1 because the source trajectory itself did not solve level 2.
- More penalties will not create the missing policy.
- Next useful work should focus on acquiring or deriving stronger post-level templates:
  - run broader deterministic sweeps to find any new level completions and save improved prefixes;
  - mine successful level transitions into phase templates automatically;
  - evaluate whether cross-game descriptor similarity can propose candidates only when same-game templates are missing, with low confidence and failure suppression.

## 2026-04-23 Cross-Game Template Fallback

### Changes Made

- Added `_phase_templates_for(...)`.
  - Same-game solved templates still have priority.
  - If same-game solved templates are missing, the agent can mix same-game continuation descriptors with very weak solved descriptors from other games at the same level/phase.
  - Cross-game descriptors are tagged `source_scope="cross_game"` and capped at low confidence.
- This remains data-derived:
  - no game-specific conditionals;
  - no ARC rule assumptions;
  - no hand-authored solutions.

### Verification

- Syntax and targeted tests:
  - `/home/moloch/ouro_project/venv/bin/python -m py_compile codex_sandbox/arc_agent_hunter_seeker_codex.py`
  - `/home/moloch/ouro_project/venv/bin/python -m pytest codex_sandbox/test_codex_sandbox.py codex_sandbox/test_codex_integration.py`
  - Result: `12 passed`.

### Performance Result

- Deterministic no-replay `r11l`, cross-game fallback:
  - Run 1: completed level 1 at step `4`, died at step `42`, score `4.761904761904762`.
  - Run 2: completed level 1 at step `4`, died at step `64`, score `4.761904761904762`.
- Conclusion:
  - Cross-game fallback did not improve score.
  - It also did not break the established level-1 phase-option improvement.

### Current Next Move

- The useful path is to acquire stronger post-level data, not to keep increasing heuristic pressure.
- Recommended next implementation:
  - run a broader sweep with `--save_trajs_dir codex_sandbox/solved_sequences_expanded`;
  - automatically ingest any new level-completion prefixes as phase templates;
  - keep cross-game fallback weak and failure-suppressed.

## 2026-04-23 Sprint Validation / Bug Fix Pass

### Changes Made

- Optimized Hunter Seeker diagnostic event dumps.
  - Added compact score-component serialization for `chosen_click_diag` and `action_trace.top_candidates`.
  - Preserved high-signal fields: total/proposal score, belief/safety/cooldown/phase bonuses, target key, compact phase template, symbolic probabilities, and compact symbolic deltas.
  - Did not change candidate scoring or search behavior.
- Fixed sandbox checkpoint loading for old v17b/Sprint4 checkpoints.
  - `train_arc_codex.py` now falls back to partial same-shape tensor loading when strict load fails due architecture drift.
  - This is generic shape-compatible loading, not ARC/game-specific logic.

### Verification

- Syntax:
  - `/home/moloch/ouro_project/venv/bin/python -m py_compile codex_sandbox/train_arc_codex.py codex_sandbox/arc_agent_hunter_seeker_codex.py`
  - Result: passed.
- Full sandbox tests:
  - `/home/moloch/ouro_project/venv/bin/python -m pytest codex_sandbox/test_codex_sandbox.py codex_sandbox/test_codex_integration.py codex_sandbox/test_event_dump_ablate.py codex_sandbox/test_focus_game_timeline_report.py codex_sandbox/test_online_trace_report.py`
  - Result: `16 passed in 64.35s`.
- Sprint report scripts:
  - `online_trace_run_report.py --episodes 3 --steps 12 --seed 0`: ran successfully; `selection_counts={"beam_search": 21, "random": 15}`, `trace_mode_counts={"topology": 21}`.
  - `event_dump_sprint6_ablate.py --seeds 0 1 2`: ran successfully; `n_runs=124`, hybrid14 accuracy `0.7634408602150536`, topology recall `0.8380952380952381`, planner recall `0.4166666666666667`.

### Performance Smoke

- Deterministic no-replay `r11l`, encoder-only smoke from `checkpoints_running/sprint4_encoder_reverted.pt`:
  - Command used `--backbone_mode encoder_only --load_trajs solved_sequences --pretrain_iters 1 --max_steps 20 --eps 0.0 --no_replay`.
  - Strict checkpoint load failed on symbolic-ranker width mismatch, then the new partial-load fallback loaded compatible tensors successfully.
  - Completed level 1 at step `4`, score `4.761904761904762`.
  - Event dump: `codex_sandbox/perf_event_dumps_r11l_compact_trace_encoder_only/r11l/run_1.json`.
  - Compact dump size was `99K` versus an older comparable `r11l_phase_options` dump at `699K`.

### Notes

- A full Ouro-backed smoke run was attempted with offline HuggingFace settings, but the tool session did not return useful progress in time. Encoder-only smoke validates the sandbox code changes and the v17b checkpoint fallback; full actual-performance sweeps should be run with cached/Ouro load confirmed or with a longer timeout.
- Current performance diagnosis is unchanged: phase options reliably recover `r11l` level 1 without raw replay, but post-level improvement requires more successful post-level data/templates rather than stronger hand-tuned penalties.

## 2026-04-23 Stockfish Option-Layer Revamp

### Motivation

- Live trusted-trajectory sweep showed the current Stockfish layer is mostly a local action ranker, not a constructive planner.
- `trusted_trajs` is the correct trajectory source, not `solved_sequences`.
  - Trusted load: `31` trajectories, `14081` transitions, expert fraction `0.894`.
  - Phase templates built from trusted trajectories: `3477` descriptors.
- `ft09` trace exposed an important phase-option bug:
  - early phase-template clicks with `frame_changed=False` still advanced the phase pointer;
  - after preventing unconditional advancement, stale unchanged phase clicks could repeat forever without a bounded escape.

### Changes Made

- Added a lightweight option layer above raw Stockfish scoring in `arc_agent_hunter_seeker_codex.py`.
  - `_phase_option_templates_for(...)` returns exact current descriptors first.
  - If exact descriptors have no live candidate, it proposes weak same-game future-phase descriptors within `_phase_option_window`.
  - Future options are tagged `source_scope="same_game_future_option"` and confidence-decayed by phase distance.
  - Option candidates carry their bonus through final scoring via `_phase_option_candidate_bonus`, so the ranker does not wash out the option proposal.
- Phase descriptors now persist `level`, `phase`, and compact traces include `phase_offset`.
- Fixed phase progress:
  - phase pointer advances immediately only after productive/frame-changing/level-progressing template actions;
  - sterile template clicks are counted per `(game, level, phase)`;
  - after 3 unchanged template attempts, the descriptor gets failure memory and the phase pointer advances to avoid deadlock.
- Persisted `phase_sterile_attempts` in checkpoints.

### Verification

- Syntax:
  - `/home/moloch/ouro_project/venv/bin/python -m py_compile codex_sandbox/arc_agent_hunter_seeker_codex.py`
  - Passed.
- Focused tests:
  - `/home/moloch/ouro_project/venv/bin/python -m pytest codex_sandbox/test_codex_sandbox.py codex_sandbox/test_codex_integration.py`
  - Result: `12 passed in 8.84s`.
- Report tests before the stale-phase patch:
  - `/home/moloch/ouro_project/venv/bin/python -m pytest codex_sandbox/test_event_dump_ablate.py codex_sandbox/test_focus_game_timeline_report.py codex_sandbox/test_online_trace_report.py`
  - Result: `4 passed in 71.94s`.

### Smoke Notes

- A short isolated smoke into `codex_sandbox/perf_event_dumps_option_revamp_smoke` loaded trusted trajectories correctly.
- `ft09` smoke still scored `0`; its trace showed repeated sterile phase-0 clicks, which directly motivated the final sterile-phase escape patch.
- The smoke wrapper stayed open during `r11l`, so do not use that run as a performance result.
- The user-launched live sweep continued separately and is the better source of full performance results.

## 2026-04-23 Trusted Sweep Review + Directional Options

### Completed Sweep Review

- Log reviewed: `codex_sandbox/sweep_expanded_trusted_20260423_1630.log`.
- Data load was correct:
  - `31` trusted trajectories;
  - `14081` trusted transitions;
  - trusted expert fraction `0.894`;
  - click phase templates `3477`.
- Per-game result summary:
  - `r11l`: all 3 runs completed level 1, then died; scores `2.56`, `4.76`, `4.76`.
  - `ft09`: run 2 completed 2 levels and saved a new level-1 prefix; run 3 inherited cumulative score `8.19` but died.
  - `tr87`, `wa30`, `ls20`: no completions in that sweep despite high frame-change rates.
- New saved artifact:
  - `codex_sandbox/solved_sequences_expanded/ft09_level1_run0.json`
  - `codex_sandbox/solved_sequences_expanded/ft09_run0_traj.npz`

### Diagnosis

- Click games benefited from phase target templates.
- Directional/topology games had zero click templates because they use movement actions, so Stockfish was still acting as a local movement ranker.
- Movement traces showed long high-change sequences with no level completion, matching the "ranker, not planner" diagnosis.

### Changes Made

- Added non-click/action phase templates mined from the same trusted `.npz` files.
  - `_phase_action_templates` stores `{game -> level -> phase -> action descriptors}`.
  - `trusted_trajs` currently builds `9399` action descriptors.
- Added `_phase_action_bonus(...)` so directional candidates receive an option bonus during scoring.
- Added `_last_chosen_phase_action_template` and phase-progress updates for non-click actions.
- Added consensus filtering in `_phase_action_templates_for(...)`.
  - If multiple solved trajectories disagree at the same phase, only the majority action gets the full option bonus.
  - Traces include `vote_count` and `variant_count`.
- Fixed level-boundary progress attribution.
  - A level-completing action now advances the phase pointer for the template's own level, not the newly entered level.
  - This prevents level 2 from inheriting level 1's phase index.
- Persisted `phase_action_templates` in checkpoints.

### Verification

- Syntax:
  - `/home/moloch/ouro_project/venv/bin/python -m py_compile codex_sandbox/arc_agent_hunter_seeker_codex.py`
  - Passed.
- Focused tests:
  - `/home/moloch/ouro_project/venv/bin/python -m pytest codex_sandbox/test_codex_sandbox.py codex_sandbox/test_codex_integration.py`
  - Result: `12 passed`.
- Report tests:
  - `/home/moloch/ouro_project/venv/bin/python -m pytest codex_sandbox/test_event_dump_ablate.py codex_sandbox/test_focus_game_timeline_report.py codex_sandbox/test_online_trace_report.py`
  - Result: `4 passed`.

### Performance Smoke

- Command family: encoder-only, `trusted_trajs`, `--no_replay`, isolated output dirs.
- `ls20` directional consensus smoke:
  - completed level 1 at step `13`;
  - score `3.5714285714285716`;
  - improved stored level-1 sequence from `21 -> 13` actions;
  - event dump: `codex_sandbox/perf_event_dumps_action_level_fix_smoke/ls20/run_1.json`.
- Trace confirms phase action templates drive movement:
  - level 1 phases use same-game solved templates with `phase_action_bonus=0.90`;
  - after level completion, level 2 restarts at `phase=0` rather than inheriting phase 14.

### Current Next Move

- Run a broader post-patch topology sweep on `ls20 tr87 wa30` with `--save_trajs_dir codex_sandbox/solved_sequences_expanded`.
- If additional improved prefixes appear, ingest them back into trusted/action-template mining.

## 2026-04-23 Branch-Consistent Directional Options

### Run Log Review

- Live topology sweep loaded the correct trusted source:
  - click phase templates: `3477`;
  - action phase templates: `9399`;
  - trusted transitions: `14081`, expert fraction `0.894`.
- `ls20` improved materially under directional options:
  - both observed runs completed levels 1 and 2;
  - score `10.599542886873792`;
  - level 1 completed at step `13`, level 2 at step `137`.
- `tr87` still failed twice before this patch:
  - both observed runs died at step `128`;
  - failure type `mechanism`;
  - action trace was a hybrid of solved runs rather than one coherent solved branch.

### Diagnosis

- The earlier action-option consensus voted independently per phase.
- For `tr87` level 1, trusted solved branches disagree:
  - `tr87_run0_traj.npz`: depth `15`, starts `2,2,4,2,2,4,...`;
  - `tr87_run1_traj.npz`: depth `37`, starts `4,4,1,2,4,2,...`.
- Per-phase voting can splice those into a path no solved trajectory actually followed.

### Changes Made

- Added branch-consistent action options:
  - `_phase_action_branch` keeps the selected source trajectory per level during a run.
  - `_phase_action_branch_failures` persists branch failures across runs.
  - `_phase_action_templates_for(...)` now prefers the active branch and chooses a solved source branch before returning exact/future templates.
- Branches are abandoned on:
  - directional death;
  - repeated sterile phase-action attempts.
- Level completion clears only the completed level's active branch.
- Trace compaction now includes action-template `source`, `branch_depth`, and `branch_failures`.
- Checkpoints now persist `phase_action_branch_failures`.

### Verification

- Syntax:
  - `/home/moloch/ouro_project/venv/bin/python -m py_compile codex_sandbox/arc_agent_hunter_seeker_codex.py`
  - Passed.
- Focused tests:
  - `/home/moloch/ouro_project/venv/bin/python -m pytest codex_sandbox/test_codex_sandbox.py codex_sandbox/test_codex_integration.py`
  - Result: `12 passed in 9.58s`.
- Report tests:
  - `/home/moloch/ouro_project/venv/bin/python -m pytest codex_sandbox/test_event_dump_ablate.py codex_sandbox/test_focus_game_timeline_report.py codex_sandbox/test_online_trace_report.py`
  - Result: `4 passed in 4.07s`.

### Operational Notes

- A patched `tr87` environment smoke was started, but it hung before step output or dump creation; it was terminated.
- The direct full-agent diagnostic also triggered a blocked Hugging Face config lookup; it was terminated.
- Host process list was checked after termination; no stale `train_arc_codex.py` or diagnostic Python process remained.

## 2026-04-23 Cross-Level Branch Carryover + Level-Lag Fix

### Diagnosis

- The first successful branch-consistent `tr87` smoke completed level 1 at step `37`, but stalled on level 2.
- Trace showed two separate issues:
  - after level completion, one action executed with no level-2 phase template because `_last_levels_completed` lagged by one step;
  - after completing level 1 with `tr87_run1_traj.npz`, level 2 could independently switch to `tr87_run0_traj.npz`, mixing trajectory branches across levels.

### Changes Made

- Added `_phase_action_preferred_source`.
  - A successful source trajectory is preferred across later levels in the same run.
  - Branch failure clears the preferred source so another branch can be tried.
- Updated `on_level_complete(...)` to immediately advance `_last_levels_completed`.
  - This removes the one-step post-completion phase lag.
- Prevented same-level fallback to another source after a preferred branch is exhausted.
- Added chosen action-template diagnostics directly to each `action_trace` entry.
- Added regression tests:
  - branch consistency within a level;
  - preferred source carryover across levels;
  - immediate `_last_levels_completed` update on level completion.

### Verification

- Syntax:
  - `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/moloch/ouro_project/venv/bin/python -m py_compile codex_sandbox/arc_agent_hunter_seeker_codex.py codex_sandbox/test_codex_sandbox.py`
  - Passed.
- Focused tests:
  - `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/moloch/ouro_project/venv/bin/python -m pytest codex_sandbox/test_codex_sandbox.py codex_sandbox/test_codex_integration.py`
  - Result: `15 passed in 2.16s`.

### CUDA Smoke

- The normal sandbox cannot see NVML/CUDA, but host-side escalated commands can:
  - `nvidia-smi` saw `NVIDIA GeForce RTX 5070 Ti Laptop GPU`;
  - project PyTorch reported `cuda_available=True`, `device_count=1`.
- Final CUDA-visible `tr87` smoke:
  - command family: encoder-only, trusted trajectories, `--no_replay`, `max_steps=120`;
  - event dump: `codex_sandbox/perf_event_dumps_branch_options_tr87_cuda3/tr87/run_1.json`;
  - checkpoint: `codex_sandbox/checkpoints_arc_branch_options_tr87_cuda3/arc_tr87_run1.pt`;
  - completed level 1 at step `37`;
  - completed level 2 at step `67`;
  - completed level 3 at step `106`;
  - score `28.571428571428573`;
  - saved improved level-3 artifacts:
    - `codex_sandbox/solved_sequences_expanded/tr87_level3_run0.json`;
    - `codex_sandbox/solved_sequences_expanded/tr87_run0_traj.npz`.

### Current Next Move

- Run a CUDA-visible multi-game topology sweep on `ls20 tr87 wa30`.
- Re-ingest the newly saved `tr87` trajectory into the trusted/action-template source or an explicit expanded source set, then rerun `tr87` to test deeper levels.

## 2026-04-24 Speculative Branch Purity + Terminal Overrun Recovery

### Diagnosis

- Hunter Seeker phase-action scoring was mutating durable branch state while evaluating candidates.
  - `_phase_action_bonus(...)` called `_phase_action_templates_for(...)`, and that helper could set `_phase_action_branch` / `_phase_action_preferred_source`.
  - Because scoring is also used by speculative beam expansion, hypothetical branches could become real before the selected action was known.
- Long branch traces can consume the known source sequence without completing the current level/domain task.
  - After the last exact source phase, action templates became empty and control fell back to generic loops.
  - This is a generic terminal-overrun failure mode, not an ARC-specific rule issue.

### Changes Made

- Added an explicit `commit` flag to `_phase_action_templates_for(...)`.
  - Direct callers keep the old committing behavior by default.
  - Scoring now calls it with `commit=False`.
- Added `_commit_phase_action_template(...)`.
  - Durable branch binding now happens only for the action actually emitted to the environment.
  - Weak recovery/reseed/terminal-overrun options do not hard-commit branch authority.
- Added weak terminal-overrun suffix options.
  - If a source branch has no current/future phase left, the planner can expose a low-confidence suffix window from the same branch.
  - Negative phase offsets are penalized by absolute distance, so old suffix actions do not get an accidental score boost.
- Kept prior stale-exact behavior intact while preventing terminal-overrun abandonment from rewinding local phase progress.
- Added regression tests for scoring purity and terminal-overrun suffix recovery.

### Verification

- Syntax:
  - `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/moloch/ouro_project/venv/bin/python -m py_compile codex_sandbox/arc_agent_hunter_seeker_codex.py codex_sandbox/test_codex_sandbox.py`
  - Passed.
- Full sandbox test set:
  - `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/moloch/ouro_project/venv/bin/python -m pytest codex_sandbox/test_codex_sandbox.py codex_sandbox/test_codex_integration.py codex_sandbox/test_online_trace_report.py codex_sandbox/test_focus_game_timeline_report.py codex_sandbox/test_event_dump_ablate.py`
  - Result: `49 passed in 4.11s`.

### Current Next Move

- Run a longer first-domain smoke to see whether terminal-overrun suffixes reduce post-source looping.
- If it helps, generalize the same read-only/commit split to any other template or memory path used from speculative scoring.

## 2026-04-24 Recovery Semantics + Retired Hint Windowing

### Diagnosis

- The first terminal-overrun smoke confirmed that suffix templates could appear, but weak hints still earned progress from raw frame changes.
  - That allowed low-confidence replay-like hints to repeat even when they did not improve topology or complete a level/task.
- A later trace showed the opposite failure after semantic gating:
  - once the active source was exhausted, recovery could spend many steps with no live action templates;
  - adding retired-source recovery hints helped expose options, but stale retired hints were treated like fresh branch failures and kept refreshing recovery.
- Probe-budget and available-action state also had speculative writes from candidate generation.
  - Those fields describe the selected real action and live environment, so they belong in `step(...)`, not in speculative beam expansion.

### Changes Made

- Weak/displaced action hints now require semantic progress.
  - Exact same-phase solved actions can still advance on raw frame change.
  - `same_game_future_action_option`, `recovery_action_option`, `reseed_action_option`, `recovery_retired_action_option`, and terminal-overrun hints require level/topology progress.
- Candidate generation no longer mutates selected-probe flags or available-action memory.
  - Selected Hunter probe flags are set after the actual chosen action is known.
  - Available-action memory is updated from the real observation at the start of `step(...)`.
- Recovery can reopen exhausted same-source action hints as very low-confidence `recovery_retired_action_option` candidates.
  - These hints are explicitly marked `retired_source`.
  - They do not hard-commit branch authority.
- Abandoning a stale retired hint now advances the local phase window without adding branch-failure credit or refreshing recovery.
- Action traces now include:
  - `phase_progress`;
  - active/preferred phase-action sources;
  - active branch failure count;
  - sterile attempt count;
  - selected Hunter probe flags;
  - compact `retired_source` / `terminal_overrun` template flags.
- Added regression tests for:
  - speculative candidate-generation purity;
  - weak-hint semantic progress gating;
  - retired-source recovery hints;
  - retired hint abandonment without recovery refresh.

### Verification

- Syntax:
  - `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/moloch/ouro_project/venv/bin/python -m py_compile codex_sandbox/arc_agent_hunter_seeker_codex.py codex_sandbox/test_codex_sandbox.py`
  - Passed.
- Full sandbox test set:
  - `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/moloch/ouro_project/venv/bin/python -m pytest codex_sandbox/test_codex_sandbox.py codex_sandbox/test_codex_integration.py codex_sandbox/test_online_trace_report.py codex_sandbox/test_focus_game_timeline_report.py codex_sandbox/test_event_dump_ablate.py`
  - Result: `54 passed in 9.35s`.
- CPU smoke:
  - command family: encoder-only, trusted-plus-expanded trajectories, `--no_replay`, `max_steps=100`;
  - event dump: `codex_sandbox/perf_event_dumps_phase_retired_abandon_cpu_smoke/wa30/run_1.json`;
  - checkpoint: `codex_sandbox/checkpoints_phase_retired_abandon_cpu_smoke/arc_wa30_run1.pt`;
  - result: no level complete, score `0.0`;
  - change rate `0.879`;
  - chosen action scopes: `same_game=14`, `recovery_action_option=11`, `recovery_retired_action_option=14`, `reseed_action_option=14`;
  - local phase advanced to `31` instead of staying pinned at `17`, confirming retired-hint abandonment moves the recovery window.

### Current Next Move

- The remaining issue is no longer replay stickiness; it is post-recovery planning quality.
- Next generic target: make recovery/escalation choose actions from predicted semantic progress, not just diversity/topology bonuses, so fallback can construct progress after learned hints expire.

## 2026-04-24 Semantic Fallback + Unguided Plateau Recovery

### Diagnosis

- After weak replay/action hints expired, Hunter Seeker could return to unguided repeated actions even while no live recovery source remained.
- The first semantic-fallback smoke showed the right signal direction, but the bonus was too large under very low world-model confidence (`WM` around `0.013` to `0.019`), so predicted symbolic successors could dominate despite being unreliable.
- Trace dumps did not preserve enough confidence-gating detail to audit whether a selected action came from the ranker, heuristic proposal, low-confidence model prediction, or recovery semantics.

### Changes Made

- Added `_phase_fallback_semantic_bonus(...)`.
  - Active only in recovery/escalation/reseed.
  - Rewards predicted generic progress signals: reachable/frontier growth, reward/exit reachability, exit progress, new colors, and moved tracks.
  - Penalizes predicted hazard expansion and missing avatar state.
  - Dampens actions that already failed in recovery/escalation memory.
- Added world-model confidence gating to fallback semantics.
  - Near-zero confidence still leaves a small exploratory prior, but cannot overrule stronger provenance/topology signals.
  - The current trust curve is `0.08 + 0.92 * sqrt(wm_confidence)`.
- Added an unguided plateau watchdog.
  - If no phase guidance is active and repeated unguided actions fail to produce level-scale progress, escalation is re-entered.
  - The counter resets when guidance returns or high progress is observed.
- Persisted and traced `phase_unguided_steps` / recent unguided action memory.
- Compact action traces now retain `wm_confidence`, `delta_trust`, `effective_confidence`, transition/prior scores, and gated transition score for selected candidates.
- Added regression tests for fallback semantic direction, inactive-mode gating, confidence damping, and unguided plateau activation/reset.

### Verification

- Full sandbox test set:
  - `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/moloch/ouro_project/venv/bin/python -m pytest codex_sandbox/test_codex_sandbox.py codex_sandbox/test_codex_integration.py codex_sandbox/test_online_trace_report.py codex_sandbox/test_focus_game_timeline_report.py codex_sandbox/test_event_dump_ablate.py`
  - Result: `59 passed in 4.43s`.
- CPU smoke:
  - command family: encoder-only, trusted-plus-expanded trajectories, `--no_replay`, `max_steps=100`;
  - event dump: `codex_sandbox/perf_event_dumps_phase_semantic_wmtrust_cpu_smoke/wa30/run_1.json`;
  - checkpoint: `codex_sandbox/checkpoints_phase_semantic_wmtrust_cpu_smoke/arc_wa30_run1.pt`;
  - result: no level complete, score `0.0`;
  - change rate `0.869`;
  - chosen action scopes: `same_game=14`, `recovery_action_option=11`, `recovery_retired_action_option=8`, `reseed_action_option=20`;
  - local phase advanced to `32`;
  - fallback semantic bonus was active but bounded: positive on `79/100` steps, negative on `1/100`, min/max `-0.0215/0.1191`, mean `0.0622`;
  - recovery/escalation/reseed coverage stayed alive through the tail instead of dropping permanently back to unguided loops.

### Current Next Move

- The immediate failure boundary is now planning quality under low-confidence symbolic successors, not branch-state corruption or stale-hint pinning.
- Next generic target: calibrate fallback semantics against observed post-action progress by feeding trace outcomes back into per-action/per-mode trust, so low-confidence predictions become useful only after they repeatedly match real progress.

## 2026-04-24 Outcome-Calibrated Fallback Semantics

### Diagnosis

- Confidence damping bounded fallback semantics, but every low-confidence symbolic prediction still started with the same authority.
- The planner needed a generic post-action feedback loop:
  - if a recovery/escalation/reseed action predicted semantic progress and the next transition did not produce level/topology progress, future bonuses for that action/mode should shrink;
  - if it repeatedly predicted progress that did materialize, future bonuses should recover and eventually strengthen.

### Changes Made

- Added game/level/mode/action semantic trust memory:
  - key shape: `game_id -> level -> recovery|escalation|reseed -> action`;
  - stored in checkpoints as `phase_semantic_trust`.
- Added `_phase_semantic_calibration_factor(...)`.
  - Default factor is `0.75`.
  - Repeated false positives can push it down toward `0.25`.
  - Repeated observed progress can push it up toward `1.20`.
- Added `_update_phase_semantic_calibration(...)`.
  - Runs after the real environment transition, before recovery/escalation/reseed timers mutate.
  - Uses the existing generic `progress_signal`, expected-progress estimate, and selected candidate fallback bonus.
  - Does not encode ARC rules or task-specific action scripts.
- Fallback semantic scoring now multiplies by the calibration factor in addition to recovery mode, action-failure dampening, and world-model confidence.
- Fallback semantic scoring now uses Stockfish's effective confidence gate (`wm_confidence * delta_trust`) when available, so symbolic fallback cannot bypass Ouro loop-convergence trust.
- Compact traces now include `phase_semantic_mode` and `phase_semantic_calibration`.
- Added a regression test proving repeated false-positive predictions reduce the bonus and repeated true progress restores it.

### Verification

- Full sandbox test set:
  - `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/moloch/ouro_project/venv/bin/python -m pytest codex_sandbox/test_codex_sandbox.py codex_sandbox/test_codex_integration.py codex_sandbox/test_online_trace_report.py codex_sandbox/test_focus_game_timeline_report.py codex_sandbox/test_event_dump_ablate.py`
  - Result: `60 passed in 4.30s`.
- CPU smoke:
  - command family: encoder-only, trusted-plus-expanded trajectories, `--no_replay`, `max_steps=100`;
  - event dump: `codex_sandbox/perf_event_dumps_phase_semantic_effective_conf_cpu_smoke/wa30/run_1.json`;
  - checkpoint: `codex_sandbox/checkpoints_phase_semantic_effective_conf_cpu_smoke/arc_wa30_run1.pt`;
  - result: no level complete, score `0.0`;
  - change rate `0.848`;
  - events `310`;
  - tracked colors `7`;
  - chosen action scopes: `same_game=14`, `recovery_action_option=11`, `recovery_retired_action_option=8`, `reseed_action_option=17`;
  - local phase progress reached `25`;
  - fallback semantic bonus stayed bounded: positive on `76/100` steps, negative on `4/100`, min/max `-0.0114/0.0893`, mean `0.0327`;
  - selected candidate calibration range `0.384..1.0`, mean `0.659`;
  - selected world-model confidence range `0.0094..0.0187`, effective confidence range `0.0050..0.0186`;
  - Hunter weight remained active late (`0.433` at step 100; max `1.0`) instead of staying pinned in low-intrigue replay fallback.

### Current Next Move

- The remaining boundary is not state purity, branch recovery, or confidence gating.
- Next generic target: improve the symbolic successor model/calibration itself. The trace still reports very low effective confidence and large predicted reachable/hazard deltas, so planning is being guided by a weak predictor even after damping.
