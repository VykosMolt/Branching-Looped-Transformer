<!-- Source: PROJECT_STATE_HUNTER_SEEKER.md lines 2890-6463 before the 2026-05-14 split. -->
<!-- Source chunk SHA256: 2c6183bc66751919818cf1af11bec172cae5918455a4df2786a46d7a37bc0bf1 -->

## Engramme / Large Memory Models context — implementation note

Public context, April 2026:
Engramme is presenting "Large Memory Models" as a memory layer for human digital-life / memorome retrieval. The useful public idea is not simply a larger context window, but memory that is lifelong, proactive, and associative.

Treat this as architectural inspiration only. Public technical detail is sparse, so do not assume a specific algorithmic breakthrough.

### Relevance to Hunter Seeker / Ouro ARC

The useful idea is not "store more logs." The useful idea is proactive associative recall.

Existing Hunter Seeker systems already contain proto-engram material:

- event log and object-table histories;
- terminal outcome memory and terminal prototypes;
- topology death diagnostics;
- solved/trusted trajectory prefixes;
- phase/action templates;
- object tracks and color/type beliefs;
- loop-state deltas and Ouro recurrence signatures;
- self-model / affective context token machinery.

### Implementation direction

Build a small engram layer above the existing memories rather than replacing them.

Minimal abstraction:

- `EngramRecord`: compact state/action/outcome memory record.
- `cue_vector`: current state/action/topology/loop signature.
- `outcome_type`: progress, no_change, terminal, mechanism, topology, self_model.
- `risk`: terminal/hazard scalar.
- `reward`: progress/solved scalar.
- `support_count` and `counterevidence_count`.

Decision-time recall should retrieve top-k relevant records using:

1. terminal/context-key similarity;
2. object/topology similarity;
3. loop-state delta similarity;
4. action/click compatibility;
5. recent phase/template compatibility.

Retrieved engrams should feed action arbitration as explicit score components:

- `engram_terminal_penalty`
- `engram_hazard_penalty`
- `engram_progress_bonus`
- `engram_probe_bonus`
- `engram_conflict_flag`

Important constraint:
Engram memory must not become broad action blacklisting. It should be contextual and counterevidence-aware.

### Immediate connection to current debug work

The chosen-action debug probe showed a likely hazard-arbitration hole: `reachable_hazard_delta` can be nonzero while `safety_penalty` remains `0.0`.

If death-probe diagnostics confirm the fatal chosen action has `reachable_hazard_delta > 0` and `safety_penalty == 0`, the next patch should add hazard-aware safety scoring, then later generalize it into the engram recall layer.

Minimal first implementation:

- `collect_engram_record(step_info, score_components, outcome)`
- `retrieve_engram_biases(current_candidate_context)`
- `apply_engram_biases_to_score_components()`
- `summarize_engram_memory()`

The first backend can reuse existing `terminal_outcome_memory`, prototypes, topology-death diagnostics, phase templates, object tracks, and loop-state signatures. A learned retriever can come later.

<!-- TOPOLOGY_ENGRAMS_NOTE_START -->

### Topology engrams — why this matters specifically for ARC/Hunter Seeker

The current ls20 death probes suggest a failure mode that terminal memory alone does not solve:

- terminal memory can fire globally;
- but the selected fatal candidate can still receive terminal_outcome_penalty = 0.0;
- repeated deaths occur in similar local adjacency/frontier situations.

This means the agent is not merely forgetting that one exact action/context was bad. It is failing to recognize a local spatial danger basin across slightly different primitive actions, tracks, or context keys.

Terminal memory says:

    action X in this exact/prototype transition context killed me

A topology engram should say:

    this local adjacency/frontier configuration near this object class killed me,
    even when the exact action, track id, or coordinate differs

This is especially relevant for ARC-style environments because danger often aliases through structure:

- same color but different object tracks;
- same local object relation but different absolute coordinates;
- same trap geometry but different primitive action;
- same frontier/adjacency basin but different step;
- terminal outcome caused by spatial relation, not by action identity alone.

Minimal topology engram record:

- source: adjacent | topology_frontier | fallback_interacted
- outcome_type: mechanism | topology | terminal | progress
- last_action
- avatar descriptor
- adjacent_colors
- frontier object descriptors
- candidate object descriptors:
  - color
  - area_bin
  - adjacency_count
  - relative centroid to avatar
  - belief_avatar / belief_exit / belief_hazard / belief_collectible
  - track protection flags
- symbolic_summary:
  - reachable_delta
  - reachable_hazard_delta
  - reachable_reward_delta
  - moved_track_count
  - frontier_delta
  - exit_path_delta
- loop_signature / loop_delta summary
- valence: negative for death/failure, positive for progress
- support_count
- counterevidence_count

Decision-time use:

For each candidate action, retrieve similar topology engrams and compute explicit score components:

- engram_topology_penalty
- engram_hazard_penalty
- engram_progress_bonus
- engram_conflict_flag

Design constraint:
Topology engrams must not become color-global or action-global blacklists. They should generalize only through local spatial/object structure.

Bad behavior to avoid:

    never press action 1
    never approach color 3

Desired behavior:

    avoid this kind of local adjacency/frontier basin unless there is strong counterevidence

Connection to current debug thread:
The same-color candidate diagnostic should not be blindly flipped into a rejection rule, because it was intentionally made track-local to avoid color-global false protection. The better fix is a topology-engram layer that can remember local danger basins while still allowing same-color objects to diverge by track and evidence.

Immediate implementation path:

1. Keep existing terminal memory as exact/prototype action-context memory.
2. Add instrumentation for cross-action terminal prototype similarity.
3. If cross-action similarity is high, add conservative cross-direction terminal fallback.
4. If cross-action similarity is weak, add topology-local engram recall using topology_death diagnostics and chosen_action_debug score components.
5. Feed topology engram recall into score components as:
   - engram_topology_penalty
   - engram_hazard_penalty
   - engram_progress_bonus
   - engram_conflict_flag

This gives Hunter Seeker proactive associative recall over spatial failure modes, which is the useful part of the Engramme/LMM idea for ARC.

<!-- TOPOLOGY_ENGRAMS_NOTE_END -->

<!-- ENGRAMME_LMM_CONTEXT_END -->


---

## Hazard arbitration handoff - 2026-04-29

Current diagnosis from ls20 death probes:

Terminal memory is no longer the primary suspect.

Observed probes:
- `chosen_action_debug_death_probe`
- `cross_action_terminal_diag_probe`

Repeated result:
- deaths remain `mechanism` failures;
- deaths are `directional/adjacent`;
- color 3 repeatedly receives hazard evidence;
- selected fatal candidates often have `terminal_outcome_penalty = 0.0`;
- terminal memory can fire globally on other candidates/actions, but does not reliably attach to the fatal selected candidate.

Cross-action terminal fallback was tested diagnostically and does not look safe yet.

Example:
- fatal action: 4
- best cross-action terminal prototype: action 2
- similarity: about 0.551

This is too weak for a safe cross-direction terminal fallback. Do not patch broad cross-action terminal avoidance yet, because it could become directional fear / action blacklisting.

More important exposed bug:

    reachable_hazard_delta can be positive while safety_penalty remains 0.0

Concrete fatal traces from `cross_action_terminal_diag_probe`:

Run 1:
- `reachable_hazard_delta = 34.948`
- `reachable_reward_delta = 164.070`
- `safety_penalty = 0.0`
- `terminal_outcome_penalty = 0.0`
- death: mechanism / directional adjacent / color 3 hazard

Run 2:
- `reachable_hazard_delta = 1.177`
- `reachable_reward_delta = 7.459`
- `safety_penalty = 0.0`
- `terminal_outcome_penalty = 0.0`
- death: mechanism / directional adjacent / color 3 hazard

Interpretation:
The scorer sees reachable hazard mass, but the safety channel does not brake the action. Hazard plus reward is being treated as activity/progress rather than as a hazard/reward conflict.

Next recommended patch:
Add conservative hazard-aware safety scoring in `claude_sandbox/arc_agent_hunter_seeker_codex.py`, around the `score_candidates` / `_score_candidates_core` section where `symbolic_summary`, `safety_penalty`, and `total_score` are assembled.

Suggested scoring logic:

    hazard = max(0.0, reachable_hazard_delta)
    reward = max(0.0, reachable_reward_delta)
    hazard_ratio = hazard / (hazard + reward + 1e-6)

    if hazard > 0:
        hazard_reachable_penalty =
            -min(0.35, 0.12 * log1p(hazard) * (0.35 + hazard_ratio))

    safety_penalty += hazard_reachable_penalty

Design intent:
This is not "hazard seen, never act." It is a brake for hazard/reward conflict.

Expected behavior:
- huge reward with some hazard: moderate penalty, not hard veto;
- small hazard with reward: small penalty;
- hazard without convincing reward/progress: stronger penalty.

Add diagnostics to score components:
- `hazard_reachable_penalty`
- `hazard_reachable_ratio`
- `hazard_reachable_delta`
- `hazard_reward_delta`

Implementation status (2026-04-29):
- Hazard-aware reachable safety scoring is implemented as the real behavior patch in `claude_sandbox/arc_agent_hunter_seeker_codex.py`.
- The hazard term is bounded, is weaker when reachable reward is large, and is included in `safety_penalty` for both click and directional candidates.
- Conservative engram recall plumbing is in place with explicit `score_components` diagnostics. Same-action contextual recall can affect score, while cross-action recall remains diagnostic unless similarity and evidence are both very high. Total engram bias is bounded to `[-0.25, 0.20]`.
- `engram_records` are persisted in checkpoints and surfaced in `measurement_summary()["engram_memory"]`.
- Post-ladder cleanup item `ArcActionAdapter` enum-map caching is implemented.

Follow-up status after GPU probe (2026-04-29):
- Original `safety_penalty == 0` bug is confirmed fixed by GPU probe.
- Hazard reachable penalty was strengthened to cap at `0.50` with the same reward-sensitive shape.
- Engram recall now aggregates positive and negative support separately, sets `engram_conflict_flag` when both match, suppresses progress bonus under conflict, and records `engram_positive_support`, `engram_negative_support`, `engram_positive_best_similarity`, and `engram_negative_best_similarity`.
- Selected-action instrumentation now falls back to live `_last_score_components` when the returned `search_trace` has empty/`None` score components, and records `score_components_source`.

GPU probe `hazard_engram_gpu_probe_post_followup` (2 runs, ls20, CUDA):
- Both runs still died to `mechanism` at steps 130/129.
- Selected-score instrumentation is now fixed: run 1 used `score_components_source = last_score_components`; run 2 used `score_components_source = search_trace`, with non-null chosen score components.
- Hazard scoring is clearly active:
  - run 1 fatal selected action: `hazard_reachable_delta = 12.36`, `hazard_reward_delta = 58.44`, `hazard_reachable_penalty = safety_penalty = -0.245`.
  - run 2 fatal selected action: `hazard_reachable_delta = 4.997`, `hazard_reward_delta = 23.41`, `hazard_reachable_penalty = safety_penalty = -0.170`.
- Risk-aware engram aggregation worked for one final run-2 candidate: action 4 had `engram_conflict_flag = True`, `engram_negative_support = 0.0509`, `engram_best_outcome = mechanism`, and `engram_total_bias = -0.0162`.
- The selected fatal run-2 action was action 2 and still had only positive engram support: `engram_negative_support = 0.0`, `engram_progress_bonus = +0.0560`, `engram_total_bias = +0.0560`.
- Checkpoint engram inspection shows the mechanism record exists but remains action 4/source adjacent and has accumulated heavy counterevidence (`support_count = 2`, `counterevidence_count ≈ 64.1`). This explains why high-sim cross-action risk did not attach to selected action 2 despite `engram_cross_action_best_similarity ≈ 0.953`.
- Next patch target: make counterevidence less eager for terminal/mechanism engrams, and treat high-hazard `progress`/`nonterminal_observed` records as mixed/risky rather than purely positive when `risk` or `hazard` is high.

Counterevidence/mixed-risk patch status (2026-04-29):
- Generic nonterminal observations no longer wash out terminal/mechanism/topology engrams. Terminal-like engrams now accept only small same-action, high-sim, low-hazard counterevidence.
- High-hazard `progress` and `nonterminal_observed` engrams now act as mixed evidence: they can contribute positive support, but also contribute negative/hazard support and trigger `engram_conflict_flag`.
- GPU probe `hazard_engram_gpu_probe_counterpolicy` still did not solve `ls20`, but deaths moved later (run 1 step 155, run 2 step 160) and the fatal chosen candidates now receive max engram risk: `engram_total_bias = -0.25`, `engram_conflict_flag = True`, `engram_negative_support > 4`, `engram_cross_action_penalty_enabled = True`.
- The mechanism engram now survives with `counterevidence_count = 0.0` instead of being erased.
- Remaining issue: even max bounded risk plus hazard can be outweighed by local optimism terms. Follow-up patch adds `engram_optimism_suppression`, `engram_risk_ratio`, and `engram_optimism_pressure` to suppress already-added directional/escalation/fallback/symbolic/progress bonuses under high-sim engram risk, capped at `0.35`.

Do not remove terminal memory.
Terminal memory remains useful as exact/prototype action-context memory. It is just not solving this specific failure because the selected fatal candidate often receives zero terminal penalty.

Connection to topology engrams:
If hazard-aware safety scoring helps but does not solve adjacent color-3 deaths, the next larger step is topology-local engram recall: remember local spatial danger basins, not just exact action/context terminal outcomes.

Risk-patcher status (2026-04-30):
- Implemented a domain/game-agnostic selection patcher in `HunterSeekerAgent`.
- It reads only generic score components already produced by the scorer: hazard penalties, terminal penalties, engram penalties, optimism suppression, positive-pressure terms, and total score.
- It does not hardcode game ids, colors, action names, ARC directions, reset/undo behavior, or action blacklists.
- It activates only when the selected candidate is risky and a lower-risk candidate is close enough in score. In all-candidates-risky states, it can choose the lower-risk candidate even if the normal total score is slightly worse.
- Random exploration now consults the same scored risk trace once terminal or engram memory exists, so random probes no longer bypass terminal/engram risk evidence.
- New diagnostics are emitted in `score_components`: `risk_patcher_risk_score`, `risk_patcher_primary_risk`, `risk_patcher_hazard_risk`, `risk_patcher_terminal_risk`, `risk_patcher_engram_risk`, `risk_patcher_optimism_risk`, `risk_patcher_positive_pressure`, `risk_patcher_score_gap`, `risk_patcher_score_budget`, `risk_patcher_risk_margin`, `risk_patcher_candidate_count`, `risk_patcher_active`, `risk_patcher_selected`, `risk_patcher_all_candidates_risky`, and `risk_patcher_reason`.
- CPU verification: `py_compile` passed; focused hazard/engram/risk-patcher/action-adapter tests passed (`13 passed, 92 deselected`); full `claude_sandbox` suite passed with the historical external event-dump fixture test skipped when its fixture root is absent (`265 passed, 1 skipped`).

GPU probe `hazard_engram_gpu_probe_risk_patcher` (2 runs, ls20, CUDA):
- Both runs still died to `mechanism`, but the failure mode changed.
- Run 1 died at step 169. The risk patcher fired heavily (`46` risk-patcher selections) and was active on the final steps. The fatal candidate had `risk_patcher_active = True`, `risk_patcher_selected = True`, `risk_patcher_reason = all_candidates_risky`, `risk_patcher_risk_score ≈ 0.909`, `hazard_reachable_penalty ≈ -0.317`, `engram_total_bias = -0.25`, and `engram_optimism_suppression ≈ -0.195`.
- Run 2 died at step 160. The risk patcher fired earlier (`65` risk-patcher selections), but the final fatal action had positive score and no current risk signal: `risk_patcher_reason = no_high_risk_chosen`, `risk_patcher_risk_score = 0.0`, `hazard_reachable_penalty = 0.0`, `terminal_outcome_penalty = 0.0`, `engram_total_bias ≈ +0.088`, and `engram_conflict_flag = False`.
- Interpretation: the all-bad optimism failure is now patched and visible. Remaining failures include blind-positive terminal contexts where the hazard/topology/engram diagnostics do not mark the candidate as risky before death.
- Do not respond by hardcoding `ls20`, color `3`, or movement actions. The next useful patch would need a generic topology-local risk cue or better pre-terminal context recall, with cross-action penalties still conservative.

Topology-local engram / risk-budget follow-up (2026-04-30):
- Implemented a conservative topology-local engram cue. `EngramRecord` now stores `topology_cue_vector`, and selected candidates preserve a small action-context cue for later nonterminal/terminal engram collection.
- The cue is domain/game agnostic: local current/predicted occupancy around the tracked avatar, local change structure, adjacent-track belief aggregates, normalized avatar/click position, and the existing bounded symbolic cue. It does not use game ids, color ids, action names, or broad action/color blacklists.
- Recall remains conservative. Same-action high-sim topology-source terminal records can penalize. Cross-action topology recall is diagnostic unless local cue similarity is extremely high (`>= 0.985`) and evidence is strong (`>= 0.75`). Engram total bias remains bounded.
- New score-component diagnostics are emitted and compacted: `engram_topology_local_cue_present`, `engram_topology_local_similarity`, `engram_topology_local_support`, `engram_topology_local_penalty`, and `engram_topology_local_match_count`.
- Added focused tests for topology-local terminal recall and unrelated-candidate no-op behavior.
- First GPU probe after this patch, `hazard_engram_gpu_probe_topology_local`, still died at steps 129/130. The cue was present, but topology-local support stayed `0.0`; the fatal candidates were still explained mostly by hazard/engram risk and optimism pressure, not by local-topology recall.
- That probe exposed a narrow risk-patcher issue: very high-risk selected candidates were sometimes not replaced because the score budget was just too tight. Follow-up patch widened the score budget only when existing generic risk components show extreme risk (`risk_score >= 0.95` or `primary_risk >= 0.60`) and allows a smaller risk-margin threshold only in high-risk cases. This still does not inspect game ids, colors, action names, or ARC directions.
- CPU verification after the follow-up patch: `py_compile` passed; focused hazard/engram/risk-patcher/compact tests passed (`20 passed, 88 deselected`); full `claude_sandbox` suite passed (`268 passed, 1 skipped`).
- Second GPU probe, `hazard_engram_gpu_probe_risk_budget`, still died at steps 129/130. Run 1 showed the widened budget working: `risk_patcher_selected` was true on the final risky/all-candidates-risky stretch and the budget reached `0.30`. Run 2's final fatal action was different: risk was only moderate by current diagnostics (`risk_patcher_risk_score ≈ 0.397`, `hazard_reachable_penalty ≈ -0.092`, `engram_total_bias = -0.25`, `engram_negative_best_similarity ≈ 0.886`, `engram_topology_local_support = 0.0`), so no narrow safe next patch is justified from this run alone.
- Current interpretation: the safety/engram/risk-patcher wiring is real, but `ls20` can still reach terminal mechanism states whose pre-terminal trace only looks mildly risky under the current generic features. The next useful work is diagnostic: improve pre-terminal context discrimination or topology-local cue quality without lowering recall thresholds into broad directional fear.

Topology/scoring normalization patch (2026-04-30):
- Implemented the follow-up in order:
  - Directional-death attribution now prefers local avatar-adjacent object ids first, then distance-gated free-space frontier ids, then fallback. This removes the remaining case where a global frontier object could beat the immediate local death relation.
  - Topology-death diagnostics now include adjacent object ids, local frontier ids, frontier distance gate, and nearest frontier distance.
  - Topology-local engram cue is now relation-first: adjacent track hazard/wall/reward/unknown aggregates, local hazard/reward proximity, action target relation, bounded symbolic ratios/densities, and a small low-weight 3x3 occupancy/change patch. It still avoids game ids, color ids, action names, broad action blacklists, and broad color blacklists.
  - Topology-local recall now emits decomposition diagnostics: `engram_topology_local_relation_similarity`, `engram_topology_local_summary_similarity`, `engram_topology_local_occupancy_similarity`, and `engram_topology_local_position_similarity`.
  - Symbolic topology scoring now uses track-deduped counts, density deltas, and hazard/reward ratios for the heuristic. Raw object deltas are retained as diagnostics, but giant reachable mass no longer produces a large progress bonus by itself.
  - New normalized symbolic terms are present in compact `score_components.symbolic_summary`: `reachable_track_delta`, `frontier_track_delta`, `reachable_reward_density_delta`, `reachable_hazard_density_delta`, `reachable_reward_ratio`, `reachable_hazard_ratio`, `reachable_hazard_ratio_delta`, `frontier_hazard_density_delta`, and `frontier_hazard_ratio`.
  - The learned symbolic planner head feature dimension was intentionally kept stable for checkpoint compatibility; the new normalized terms feed the heuristic and diagnostics, not the saved head shape.
- CPU verification:
  - `py_compile` passed for `arc_agent_hunter_seeker_codex.py` and `test_causal_correctness.py`.
  - Focused causal tests passed: `23 passed, 87 deselected`.
  - Full sandbox suite passed: `270 passed, 1 skipped`.
- GPU probe `topology_norm_cuda_probe` / event dir `claude_sandbox/perf_event_dumps_topology_norm_cuda_probe` (2 runs, `ls20`, CUDA, `max_steps=160`, `eps=0.0`):
  - Both runs completed level 1 at step 13 and level 2 at step 136.
  - Neither run died under the 160-step cap. This is a major change from the previous step-129/130 terminal deaths.
  - Selected candidate instrumentation is populated again; `chosen_candidate.score_components` includes hazard, engram, risk-patcher, topology-local, and normalized symbolic diagnostics.
  - Run 1 had 7 selected candidates with hazard/risk signal; early level-1 moves got hazard penalties around `-0.09` to `-0.16`, and late run risk was small (`-0.027` to `-0.032`). Run 2 had no selected high-risk candidates.
  - Topology-local cue presence is visible in selected score components, but topology-local support remained `0.0` because no terminal topology engrams were collected in these non-death runs. The new decomposition diagnostics are available for the next death case if one reappears.
  - Measurement still reports high terminal context `zero_delta_count`; since the probe did not die, no terminal exact/prototype penalties were exercised. Treat that as residual diagnostic debt, not a blocker for the current patch.
- Current interpretation:
  - The concrete topology/scoring bug chain that produced the earlier `ls20` terminal deaths is plausibly fixed or at least substantially mitigated.
  - No next behavioral patch is justified from this GPU result alone. The next decision should be based on a longer/multi-game run, not another immediate safety penalty increase.

Current topology code state, detailed handoff (2026-04-30):
- Free-space topology is still generic and game-agnostic:
  - `_compute_free_space_topology()` builds a passable mask from dominant background plus traversable objects.
  - Object traversability is belief-based: objects are treated as traversable unless their track belief says wall/hazard strongly enough.
  - The avatar start point comes from the avatar track/object if available, then nearest passable fallback.
  - It returns reachable mask/object ids, frontier object ids, region map, region sizes, object-region assignments, gateway object ids, region adjacency, BFS distance map, and BFS max distance.
  - It does not use game ids, color ids, action names, or ARC-specific direction semantics beyond the action adapter's direction offsets.
- Directional topology bonus is still a small optimistic steering term:
  - `_directional_topology_bonus()` scores the one-step target cell/object for wall/hazard/reward/unknown belief, exit-distance progress, reachable-cell status, frontier-distance progress, and reward-like-object progress.
  - It remains clipped to `[-0.90, 0.90]`.
  - It is not a safety veto. Hazard-aware reachable scoring, engram risk, terminal memory, and the risk patcher are the braking paths.
- Symbolic topology scoring is now normalized:
  - `_symbolic_transition_summary()` still emits raw object deltas for diagnostics, but belief masses are deduped by track before hazard/reward/unknown aggregation.
  - New normalized terms include track-count deltas, hazard/reward density deltas, hazard/reward ratios, hazard-ratio delta, frontier hazard density delta, and frontier hazard ratio.
  - `_symbolic_bonus_from_summary()` now scores bounded track/density/ratio terms instead of letting raw reachable-object mass dominate.
  - Hazard ratio damps reward/reachability optimism, and positive hazard density/ratio deltas add bounded negative pressure.
  - The learned `SymbolicPlannerHead` input dimension is intentionally unchanged for checkpoint compatibility; normalized topology terms affect the heuristic and diagnostics, not the saved head shape.
- Hazard-aware safety scoring is active and bounded:
  - `_hazard_reachable_safety_components()` still computes a reward-sensitive penalty from positive reachable hazard delta.
  - Penalty is bounded at `0.50`, weakens when reward is huge, and is included in `safety_penalty`.
  - Required score components remain present: `hazard_reachable_penalty`, `hazard_reachable_ratio`, `hazard_reachable_delta`, `hazard_reward_delta`.
- Directional death attribution is now local-first:
  - In `on_game_over()`, directional deaths collect the best avatar object, immediate adjacent object ids, and topology frontier ids.
  - Candidate source order is now:
    1. adjacent object ids from the avatar object's actual scene adjacency;
    2. frontier object ids only if they are within a distance gate from the avatar object;
    3. interacted non-safe fallback tracks.
  - Adjacent attribution no longer expands by color. Same-color non-adjacent components are not updated.
  - Safe filtering remains track-local: avatar/exit-like protected tracks are skipped, but a protected same-color track does not protect the whole color.
  - Evidence is deduped by track id before applying hazard updates.
- Topology-local engram cue is conservative and relation-first:
  - `_engram_topology_local_cue()` now builds a 63-dimensional cue with named sections:
    - relation: 18 values for adjacent belief aggregates, hazard/reward proximity, and target-object relation;
    - summary: 14 bounded symbolic topology/history values;
    - occupancy: 27 low-weight current/predicted/change samples from a 3x3 avatar-local patch;
    - position: 4 low-weight normalized avatar/target coordinates.
  - It avoids game ids, color ids, action names, broad action blacklists, and broad color blacklists.
  - Relation and summary sections dominate; occupancy is intentionally low weight so pixel-layout drift does not swamp semantic local risk.
- Topology-local engram recall remains conservative:
  - Same-action high-sim terminal topology-source records can penalize.
  - Cross-action topology recall is diagnostic unless local cue similarity is extremely high and record evidence is strong.
  - Engram total bias remains bounded to `[-0.25, 0.20]`.
  - Topology-local support will remain zero until terminal/topology-source engrams exist; in the successful 2-run probe, no deaths meant no new terminal topology memories.
- Current topology diagnostics in selected/top candidates:
  - Hazard/safety: `hazard_reachable_penalty`, `hazard_reachable_ratio`, `hazard_reachable_delta`, `hazard_reward_delta`, `safety_penalty`.
  - Symbolic normalized topology: `reachable_track_delta`, `frontier_track_delta`, `reachable_reward_density_delta`, `reachable_hazard_density_delta`, `reachable_reward_ratio`, `reachable_hazard_ratio`, `reachable_hazard_ratio_delta`, `frontier_hazard_density_delta`, `frontier_hazard_ratio`.
  - Engram recall: positive/negative support, best similarities, conflict flag, total bias, optimism suppression.
  - Topology-local recall decomposition: relation, summary, occupancy, and position similarities.
  - Risk patcher: risk score, risk components, positive pressure, score gap/budget/margin, selected/active/all-candidates-risky flags, reason.
  - Directional death postmortem: adjacent object ids/colors, local frontier ids, frontier gate/nearest distance, raw/candidate/rejected pools, protected tracks/colors, source.
- Important residual diagnostics:
  - Terminal context `zero_delta_count` remains high in measurements, even though unique key diversity is high. The current successful probe did not exercise terminal penalties because it did not die.
  - Topology-local decomposition diagnostics are visible in score components, but support stayed zero in the successful run because there were no terminal topology records to retrieve.

How to find the next issues (recommended order):
1. Extend `ls20` before changing code:
   - Run `max_steps=240` or `320`, `n_runs=5`, `eps=0.0`.
   - Purpose: determine whether the old step-129/130 death is gone and whether level-3+ introduces a new failure.
   - Success signal: repeated level 1/2 clears and no terminal deaths.
   - Failure signal: any death with populated `chosen_candidate.score_components`; inspect the final 10 selected candidates plus `topology_death`.
2. Then run a small keyboard/topology multi-game smoke:
   - Suggested games: `ls20 tr87 wa30`.
   - Use `max_steps=200` to `240`, `n_runs=3`.
   - Purpose: detect whether the normalized topology scoring helps `ls20` but harms other keyboard/topology games.
3. Then run a broader mixed smoke:
   - Suggested games: `ls20 ft09 r11l tr87 wa30 su15`.
   - Use `max_steps=180` to `240`, `n_runs=2` or `3`.
   - Purpose: catch click-game regressions from shared symbolic scoring/risk-patcher changes.
4. Do not add another behavior patch before one of those runs produces a concrete failure trace.
   - Current evidence says the immediate topology/scoring issue is mitigated.
   - A premature penalty increase would risk turning conservative safety into broad directional fear.
5. Diagnostics are probably sufficient for the next run.
   - The useful next tooling would be a dump summarizer, not more live score fields: aggregate per-run deaths, final selected components, hazard/risk/engram distributions, risk-patcher selections, and topology-death source/candidate pools.
   - Add new live diagnostics only if the next failure has `None`/missing selected components, zero topology-death postmortem, or ambiguous cue decomposition.

Post-ladder deferred cleanup check (2026-04-30):
- Ordered self-model pending-loss consumption is present: `HunterSeekerAgent.step()` consumes pending self-model event loss before the base/ranker path can mutate shared parameters, and the focused causal tests cover the safe/no-op and ordering behavior.
- `ActionAdapter.safe_action_indices()` is now implemented and the empty-action fallback paths in the base pairwise agent and Hunter Seeker use the adapter hook instead of assuming `range(n_actions)` directly.
- ARC and mock adapters preserve prior behavior by returning their full action ontology; future domains can override the hook with domain-specific safe fallback actions without changing planner code.
- The historical `focus_game_timeline_report` test now skips when the external `event_dumps/sprint4_overnight` fixture root is absent, so the sandbox suite is runnable from the tidied checkout.
- The remaining items in the pre/post-ladder notes are intentionally deferred or broad hygiene: OutcomeAdapter/progress abstraction, color/entity terminology rename, legacy ARC default cleanup, Ouro-confidence trust redesign, sleep/consolidation, `pad_grids_to_batch` vectorization, stable frame hashes only if persistent caches need them, and generic `loop_pooler_gate` logging migration.

wa30 contact/topology cleanup and GPU result (2026-04-30):
- Additional permanent fixes landed after the topology-normalization probe:
  - `ColorPriorTable.update_from_track()` no longer promotes avatar-only evidence into a color-level affordance prior. Avatar evidence is treated as instance/track-local; if any color-level record is still created from mixed evidence, the avatar component is damped before storage. This prevents all same-colored components from becoming "the avatar" after reset.
  - Local contact hazard scoring is now an explicit bounded safety path. `_local_contact_hazard_components()` scores the target/contact object near the avatar and emits `local_contact_hazard_penalty`, `local_contact_hazard`, `local_contact_wall`, `local_contact_reward`, `local_contact_avatar`, `local_contact_unknownness`, `local_contact_alignment`, `local_contact_distance`, `local_contact_obj_id`, `local_contact_track_id`, `local_contact_out_of_bounds`, and `local_contact_source`.
  - Local contact penalty is included in `safety_penalty`, compact score components, and the risk patcher. It also handles avatar-overlap target cells by looking through to adjacent contact, and action 5/no-op can now receive adjacent-contact hazard penalty instead of early-returning.
  - Directional death attribution has a narrow `adjacent_terminal_override` for cases where all adjacent candidates were filtered as safe/avatar-like. The override can accept a single adjacent terminal object even if it is not in the action direction, because some terminal frames place the fatal/contact object behind or beside the avatar.
  - Directional death attribution now accepts already-known high-hazard fallback tracks even when they have zero interaction count.
  - A late low-hazard timeout guard was added: if the run is late (`run_relative_step >= 150`), attribution exists only through the low-prior adjacency override, and the max terminal candidate hazard prior is below `0.20`, the death is classified as `planner` and hazard evidence is skipped. New topology-death diagnostics include `terminal_candidate_prior_hazard_max`, `terminal_run_relative_step`, `timeout_like_terminal`, and `timeout_like_reason`.
  - Domain-general cleanup continued: train/live bootstrap paths use `action_adapter.bootstrap_action(GameAction)` instead of hardcoded first actions; no-frame fallback uses adapter safe indices; terminal cross-action diagnostics use adapter directional ids; action/context coordinate hints and BFS summary distances normalize by actual frame/grid size rather than a fixed 64.
- CPU verification after these changes:
  - `py_compile` passed.
  - Focused causal tests passed: `119 passed`.
  - Full `claude_sandbox` suite passed: `279 passed, 1 skipped`.
- GPU probe `topology_contact4_wa30_gpu` / event dir `claude_sandbox/perf_event_dumps_topology_contact4_wa30_gpu`:
  - Command used CUDA through `/home/moloch/ouro_project/venv/bin/python -m claude_sandbox.train_arc_codex`.
  - `nvidia-smi` showed the live Python process as a compute process using about `634MiB`; low utilization was expected because this probe is Python environment/game-loop heavy rather than dense GPU training.
  - All 3 runs ended at environment step 200 with `levels_completed = 0`.
  - All 3 deaths were classified as `planner`, not mechanism/topology. Cumulative failure counts were `planner=1`, `planner=2`, `planner=3`.
  - In all 3 runs, `topology_death.source = adjacent_terminal_override`, `timeout_like_terminal = true`, and `timeout_like_reason = late_low_hazard_adjacency_override`.
  - Terminal candidate hazard priors were effectively zero (`~1e-10`), so hazard evidence was correctly skipped instead of poisoning object/color topology memory.
  - Chosen score components were populated again (`score_components_source = search_trace`), including hazard, local-contact, engram, topology-local, normalized symbolic, and risk-patcher fields.
  - Run 1 final selected candidate still had real risk diagnostics (`hazard_reachable_delta = 2.455`, `hazard_reward_delta = 6.566`, `hazard_reachable_penalty = safety_penalty = -0.139`, `engram_conflict_flag = true`, `engram_total_bias = -0.25`), but the terminal attribution itself was timeout-like and low-hazard.
  - Runs 2 and 3 final selected candidates had no positive reachable hazard delta and no local-contact hazard penalty; the death was still the same step-200 timeout classification.
- Current diagnosis:
  - The original topology/hazard poisoning bug is fixed for this wa30 trace. The system no longer learns a fake hazard from the step-200 terminal frame when the candidate object has only avatar-like/low-hazard evidence.
  - The current wa30 failure is planner/progress: the environment itself has a countdown-style terminal condition around step 200, so increasing `max_steps` beyond 200 does not help this game unless the agent reaches progress sooner.
  - This result does not justify another safety/topology penalty increase. The braking paths are wired; the remaining issue is that the policy/search/reseed/phase machinery is not finding the productive sequence before timeout.
- How to find the next issue:
  1. Inspect the wa30 chosen-action timeline against trusted trajectories. Find the first divergence from a trusted/progress-producing sequence and check whether `phase_action_template`, reseed, escalation, risk patcher, or symbolic scoring made the choice.
  2. Add or use a dump summarizer that aggregates per-run selected actions, level/progress events, risk-patcher selections, phase/reseed template use, final 10 score components, and topology-death postmortem. More live score fields are not needed yet; better aggregation is.
  3. For wa30 specifically, increase run count rather than max step count: `n_runs=5` or more at `max_steps=240` is useful, but the real deadline is the environment's step-200 timer.
  4. For regression, run a mixed GPU smoke over `ls20 tr87 wa30` after any planner/reseed patch. `ls20` should continue clearing level 1/2 under the normalized topology patch, and wa30 should remain planner-classified rather than poisoning topology.
  5. Do not hardcode wa30, colors, action ids, or the step-200 timer into the cognitive core. Any fix should be expressed as domain-general progress/reseed/planner behavior or adapter-provided outcome/progress abstraction.

Phase-guidance / engram / terminal-risk cleanup after wa30 GPU inspection (2026-04-30):
- Code changes landed after the `topology_contact4_wa30_gpu` result:
  - Exact trusted phase-action guidance is now defined centrally by `_phase_action_template_is_exact_trusted()`: solved template, same-game source scope, phase offset `0`, not retired, confidence at least `0.80`.
  - Exact same-game templates can explicitly encode expected no-observation-change transitions via `expected_observation_changed = False`. `_phase_action_template_made_progress()` now treats those exact trusted no-op/wait transitions as phase progress when the observation does not change.
  - `_build_phase_target_templates()` records `expected_observation_changed` and `expected_level_progress` from trajectory `frames_after` / `levels` for both click and directional phase templates.
  - `_compact_phase_template()` now preserves `expected_observation_changed`, legacy `expected_frame_changed`, and `expected_level_progress`. This fixed a real bug: the template builder had the no-op metadata, but compacting the selected score components stripped it before `_last_chosen_phase_action_template` was used on the next step.
  - Exact trusted phase actions are ordered first in `generate_candidates()` through `_phase_action_candidate_order_priority()`. This is not a score term; it protects exact guidance from the parent beam search's pre-score predicted-frame dedupe, which otherwise kept the first action-id candidate and could clip the phase action before `phase_action_bonus` existed.
  - The risk patcher now respects exact trusted phase guidance unless there is hard direct risk. Broad hazard/engram risk, a soft terminal prototype, or a single low-confidence prototype match cannot veto exact trusted guidance by itself.
  - Hard direct risk still vetoes exact phase guidance: exact terminal evidence, local contact hazard penalty, high local hazard, or a terminal prototype only when the prototype is matched with similarity at least `0.95` and count at least `2`.
  - `on_game_over()` no longer records phase-action branch failure unconditionally. `_phase_action_branch_failure_should_record()` only retires branch evidence for mechanism/topology terminal causality, and never for timeout-like/planner/self-model outcomes.
- Why these are domain-general:
  - No color ids, action ids, game ids, wa30 names, or ARC timer constants were added.
  - The changes are expressed as contracts over trusted demonstration metadata, candidate dedupe, terminal evidence strength, and failure causality.
  - Weak recovery/reseed/future templates remain low-weight/diagnostic; the strong protection applies only to exact same-game solved templates.
- CPU verification after the final soft-prototype patch:
  - `py_compile` passed for `arc_agent_hunter_seeker_codex.py`, `test_codex_sandbox.py`, and `test_causal_correctness.py`.
  - Focused phase/risk tests passed.
  - Full sandbox suite passed: `287 passed, 1 skipped`.
- New regression tests added:
  - Risk patcher keeps exact trusted phase guidance when only broad hazard/engram risk is present.
  - Risk patcher can still override exact guidance with direct terminal risk.
  - A single soft terminal prototype match does not count as direct risk.
  - A repeated high-similarity terminal prototype still counts as hard direct risk.
  - Phase branch failure records only real terminal causality, not planner/self-model/timeout endings.
  - Exact phase actions are ordered before untemplated actions so pre-score dedupe cannot erase them.
  - Compacted phase templates preserve expected no-observation-change metadata and still advance on expected no-op transitions.
- GPU probe sequence and interpretation:
  1. `phase_trust_guard_wa30_gpu`, `max_steps=240`, `n_runs=3`:
     - Result: `levels_completed = 0` for all runs.
     - Failure profile: `planner=1`, `topology=2`.
     - Inspection showed first meaningful remaining issue was exact phase guidance getting clipped or bypassed after early phases.
  2. `phase_dedupe_guard_wa30_gpu`, `max_steps=240`, `n_runs=3`:
     - Result: `levels_completed = 0` for all runs.
     - Failure profile: `planner=3`.
     - Improvement: topology/mechanism poisoning disappeared in that probe.
     - Inspection exposed that no-op/wait metadata was being stripped by compacting, so exact no-change phase actions did not advance phase.
  3. `phase_noop_compact_wa30_gpu`, `max_steps=240`, `n_runs=3`:
     - Result: still `levels_completed = 0`.
     - Failure profile: `mechanism=2`, `planner=1`.
     - Inspection showed run 3 followed the trusted sequence until step 25, then the risk patcher replaced exact phase action `2` because of one terminal prototype match at similarity about `0.806`. That was too soft to veto trusted guidance.
  4. `phase_soft_proto_guard_wa30_gpu`, `max_steps=240`, `n_runs=3`:
     - Result: run 1 ended planner at step 200; runs 2 and 3 completed levels 1 and 2.
     - Run 2: level 1 completed at step 125, level 2 completed at step 183, final `levels_completed = 2`.
     - Run 3: level 1 completed at step 125, level 2 completed at step 183, final `levels_completed = 2`.
     - This is the first clean wa30 evidence in this pass that the phase/topology/risk stack can make real progress rather than only avoid bad deaths.
  5. `phase_soft_proto_long_wa30_gpu`, `max_steps=500`, `n_runs=1`:
     - Result: unlucky planner/timeout-like death at step 200 before level 1 completion.
     - It does not invalidate the 3-run result; it shows the start is still stochastic/fragile and longer validation needs multiple runs or a checkpoint carried forward from a successful level-2 run.
- Trusted wa30 trajectory length check:
  - Level 1: 124 actions.
  - Level 2: 58 actions.
  - Level 3: 257 actions.
  - Therefore a 240-step cap is intrinsically too short to evaluate level-3 completion once level 2 finishes around step 183. A clean level-3 test needs roughly `max_steps >= 450`, preferably `500`.
- Current topology state after these patches:
  - The topology core is still not using wa30-specific or ARC-specific rule tables.
  - Hazard-aware scoring, local contact scoring, terminal/death attribution, and topology-local engram fields remain in `score_components`.
  - The original safety/hazard bug remains fixed: positive reachable-hazard deltas produce bounded negative `hazard_reachable_penalty` / `safety_penalty`.
  - Timeout-like low-hazard terminal frames are still classified as planner and do not poison hazard memory.
  - The remaining observed failure is not a topology math bug. It is planner/phase robustness: exact guidance can now carry successful runs through two levels, but some runs still drift or time out before level 1.
- Current recommended next work:
  1. Add a dump summarizer before further hand inspection. It should aggregate actions vs trusted prefix, first divergence, level-complete steps, selected exact/weak/no-template counts, risk-patcher replacements, final score components, and topology-death postmortem.
  2. Run `wa30` with `max_steps=500`, `n_runs=3` or `5`, not a single long run, to separate stochastic start failures from real level-3 failure.
  3. If one long run reaches level 3 and then fails, inspect the first divergence after trusted phase 183 before patching.
  4. Run mixed regression `ls20 tr87 wa30` after any next planner/phase patch.
  5. Do not add another safety or topology penalty from the current evidence. The remaining bottleneck is productive sequence following under planner/reseed/risk-patcher interaction.

Phase-exact override validation, summarizer, and current run state (2026-04-30):
- Added `claude_sandbox/summarize_event_dumps.py` plus `claude_sandbox/test_summarize_event_dumps.py`.
  - The summarizer aggregates current-run levels, terminal events, failure counts, trusted-prefix divergence, phase-template usage, exact phase override reasons, risk-patcher reasons, final score components, and topology-death postmortems.
  - It filters current-run events with `run_start_step < step_number <= run_end_step` and positive `run_relative_step`, so terminal events from prior runs are no longer counted as the current run.
  - It prefers `failure_counts_current_run` when present and keeps cumulative failure counts separate.
- Measurement instrumentation was corrected in `HunterSeekerAgent`:
  - `measurement_summary()` now emits `failure_counts_current_run`, `event_log_counts_current_run`, and `event_log_size_current_run`.
  - Current-run helpers isolate event/failure summaries from cumulative logs.
  - `reset_for_new_game()` clears stale `_last_topology_death_diag`, so successful same-game runs no longer inherit an old topology death postmortem.
- The exact phase-guidance patch is now the active code state:
  - `_abandon_phase_action_branch()` no longer records source-level branch failure from local sterile search. Source retirement is reserved for terminal causal evidence, not failed local lookahead.
  - `_phase_exact_guidance_select_from_trace()` can override beam-lookahead selection when an exact same-game trusted phase candidate appears in the root trace and has no hard direct risk.
  - `_phase_exact_guidance_write_candidate_diag()` writes the override diagnostics back into `score_components`.
  - Hard direct risk still wins: exact terminal evidence, hard terminal prototype evidence, or local contact hazard can veto exact guidance.
  - New compact score-component fields: `phase_exact_override_active`, `phase_exact_override_selected`, `phase_exact_override_reason`, `phase_exact_override_score_gap`, and `phase_exact_override_candidate_count`.
- CPU verification after this work:
  - `py_compile` passed for touched `claude_sandbox` modules/tests.
  - Focused phase/risk/causal/summarizer tests passed.
  - Full sandbox suite passed: `291 passed, 1 skipped`.
- Pre-patch diagnostic run `phase_summarizer_long_wa30_gpu`:
  - `max_steps=500`, `n_runs=3`, `wa30`.
  - All 3 runs failed planner/timeout before level completion.
  - Summarizer showed exact trusted phase guidance barely engaged: `exact_trusted=28`, `weak_template=175`, `no_template=397`.
  - Runs 2 and 3 diverged from trusted action at step 2. Run 1 diverged at step 14 when the exact trusted action was present but beam lookahead selected a different first action.
  - Diagnosis: this was a phase-guidance/search-selection problem, not topology poisoning.
- Post-patch GPU run `phase_exact_override_wa30_gpu`:
  - Command used CUDA through `/home/moloch/ouro_project/venv/bin/python -m claude_sandbox.train_arc_codex`.
  - `nvidia-smi` during execution showed the Python process as a compute process using about `634MiB`. After completion, `nvidia-smi` showed no remaining Python compute process.
  - Result at `max_steps=240`, `n_runs=3`: all runs completed level 1 at step 125 and level 2 at step 183. No death/game-over occurred under the cap.
  - Summarizer output: `levels total=6`, `per_run=[2, 2, 2]`, no current-run failures, `phase_templates exact_trusted=720`, `weak_template=0`, `no_template=0`.
  - Exact phase override reasons: `trusted_exact_phase_guidance=24`, `already_exact_phase_guidance=696`.
  - Risk-patcher reasons: `trusted_exact_phase_guidance=156`, `no_high_risk_chosen=564`.
  - Trusted-prefix check: every run matched the trusted trajectory for all 240 observed actions (`min=max=240`, `mean_ratio=1`).
  - Final score components remained populated, including phase-action bonus, engram bias, hazard/safety fields, and risk-patcher reason.
- Current interpretation:
  - The original safety/hazard bug is fixed.
  - The topology poisoning path is not the current blocker for this trace.
  - The phase-guidance/search bug exposed by the long failed run is fixed at the 240-step validation horizon.
  - The 240-step cap cannot test level 3, because trusted `wa30` level 3 needs 257 actions after level 2 completes around step 183.
  - No new behavior patch is justified from this successful run. The next issue should be discovered by running `wa30` at `max_steps=500`, `n_runs=3` or `5`, summarizing first divergence after step 183 if level 3 fails.

Future work note from `hunter_seeker_additional_components.md` (2026-04-30):
- The new root markdown proposes two major future components:
  1. Observation learning:
     - Canonical `ObservationTransition` records over `frame_t -> frame_t+1`, with optional known or inferred action/click labels.
     - `ObservationReplayBuffer` for known-action transitions, unlabeled transitions, click transitions, topology deltas, and object-contrastive batches.
     - `EventSegmenter` for turning raw video into stable before/after transitions instead of training on every raw frame.
     - `ChangedMaskHead`, `InverseActionModel`, `TopologyDeltaHead`, object-permanence contrastive loss, and action-conditioned/observation-only forward dynamics.
     - Encoder drift protection is explicit: start with frozen encoder plus trainable observation adapter/heads, and only unfreeze selectively after held-out probes show no damage.
     - Observation-derived priors may influence candidate recall/actionability/topology mechanism estimates, but must never override terminal or hazard guards.
  2. Internalized evaluator / self-value:
     - Standardize candidate futures with a `CandidateFuture` record instead of ad-hoc score dictionaries.
     - Encode candidate futures and train `SelfValueHead` / `PairwiseSelfPreferenceHead` from outcome, evaluator, safety, and observation teachers.
     - Add `SelfState` logging for chosen action, expected outcome, actual outcome, surprise, confidence, hazard/progress estimates, memory support, and evaluator disagreement.
     - Roll out in stages: offline only, shadow mode, small score contribution, tie-break/uncertainty use, then normal contribution only after validation.
     - Preserve the core rule that learned self-value must not override hard terminal/death/hazard guards.
- My assessment:
  - This document is architecturally aligned with the project. It turns the current hand-coded event/topology/engram machinery into learnable transition understanding without abandoning the brain-ontological shape.
  - The strongest immediate part is observation learning from known-action trajectories: it can improve candidate recall, inverse-action priors, changed-mask localization, topology delta prediction, and mechanism familiarity while keeping the encoder frozen.
  - The self-value proposal is correct in direction because the external evaluator should become a teacher/critic rather than a permanent hot-path authority. Pairwise training and shadow rollout match the CLT finding that relational preference is the useful signal.
  - The main risk is scope. Implementing the full document now would distract from the current phase/topology validation and could introduce another poorly calibrated scoring source. It should be staged after the current `wa30` long validation and mixed regression.
  - Recommended future order:
    1. Finish current evidence loop: `wa30 max_steps=500 n_runs=3/5`, then mixed `ls20 tr87 wa30`.
    2. Add only the data schema first: `ObservationTransition`, `ObservationReplayBuffer`, and `CandidateFuture`, with tests and no scoring effect.
    3. Feed trusted/known-action trajectories into the observation buffer and train/evaluate `ChangedMaskHead` plus `InverseActionModel` offline.
    4. Add `TopologyDeltaHead` from deterministic object/topology targets.
    5. Add self-value in shadow mode only, with pairwise flip tests and terminal/hazard preference tests before any live scoring weight.
    6. Consider video/event segmentation only after known-action observation learning is stable.

Long `wa30` GPU validation and phase-stall patch (2026-04-30):
- The requested long GPU validation was run outside the sandbox so PyTorch actually attached to the NVIDIA GPU.
  - Initial sandboxed launch was terminated because `nvidia-smi` showed no Python compute process and `ps` showed the process burning CPU.
  - Relaunch used `/home/moloch/ouro_project/venv/bin/python` directly with escalated execution. `nvidia-smi` showed Python as a `C` compute process on GPU 0, first around `488MiB`, then around `634MiB`.
- Long pre-stall-patch run `phase_exact_long_wa30_gpu2`:
  - Command: `wa30`, `max_steps=500`, `n_runs=3`, `eps=0.0`, `no_replay`.
  - Result: all 3 runs completed level 1 at step 125 and level 2 at step 183, then died at step 283 during level 3.
  - Summarizer: `levels total=6`, `per_run=[2,2,2]`, `failures topology=3`, `phase_templates exact_trusted=849`, `weak_template=0`, `no_template=0`.
  - Trusted action prefix before first divergence was exactly `252` in all three runs.
  - First divergence in all three: run step 253, actual action `5`, expected trusted action `3`.
  - Inspection showed the exact phase action at phase 68 was action `5`; it was expected to change the observation, but one live attempt produced no visible frame change. The agent therefore kept phase progress at 68 and repeated action `5`, inserting an extra action and putting the rest of level 3 one phase late.
- Phase-stall patch landed:
  - `_phase_action_templates_for()` now detects when an exact same-source solved action has a sterile attempt on the current phase.
  - If that exact action was expected to change the observation and a next same-source solved phase exists, it exposes two bounded options:
    - `same_game_stalled_retry`: a downgraded retry of the current phase action.
    - `same_game_stall_escape_action_option`: the next same-source phase action, treated as a non-hard-trust recovery hint.
  - `same_game_stall_escape_action_option` is intentionally not exact trusted guidance. Terminal/hazard guards and the risk patcher can still veto it.
  - Stall escape consumes the next phase slot even when the escape action itself has no visible frame change; this prevents the system from repeating the escape action and drifting another step.
  - New compact diagnostics in `score_components`: `phase_stall_attempts`, `phase_stall_escape_bonus`, and `phase_stall_retry_penalty`.
  - Compact phase templates now preserve `stalled_phase`, `stalled_action`, and `stall_attempts`.
- CPU verification:
  - `py_compile` passed for `arc_agent_hunter_seeker_codex.py` and `test_codex_sandbox.py`.
  - Focused phase tests passed: `17 passed, 47 deselected`.
  - Full sandbox suite passed after the final stall patch: `293 passed, 1 skipped`.
- GPU validation of the stall patch:
  1. `phase_stall_escape_wa30_gpu`, `max_steps=500`, `n_runs=1`:
     - Step 253 selected the stall-escape action `3` instead of repeating action `5`.
     - However the escape action was still progress-gated by visible frame change, so it repeated and triggered recovery. This produced first divergence at step 254 and still died at step 283.
     - Failure was classified as planner/timeout-like low-hazard in that run, which confirmed topology poisoning was not the right explanation.
  2. `phase_stall_escape2_wa30_gpu`, `max_steps=500`, `n_runs=1`:
     - Step 253 selected action `3` through `same_game_stall_escape_action_option`.
     - Phase progress advanced to phase 70 at the next step, and the live action prefix matched the trusted action sequence through the terminal step: `trusted_prefix=283`, `mean_ratio=1`.
     - The run still died at step 283, classified as topology.
     - Summarizer: `levels total=2`, `per_run=[2]`, `failures topology=1`, `phase_templates exact_trusted=282`, `weak_template=1`, `no_template=0`.
- Current diagnosis after the stall patch:
  - The action-sequence bug is fixed: the agent no longer repeats phase-68 action `5`, and it can maintain the trusted action prefix through step 283.
  - The remaining failure is not a simple phase-action selection bug. It is state divergence under an action-matched trusted prefix.
  - The critical evidence is phase 68 / run step 252: the trusted trajectory records action `5` as producing a 13-cell frame change, but the live run produced no event for that action. Following the subsequent trusted actions exactly was not enough to restore the trusted state, and the live state later died at step 283.
  - Therefore the next problem is state/observation alignment, not another hazard penalty and not an ARC-specific timer hack.
- Current recommended next patch:
  1. Add state-aware phase alignment diagnostics before changing scoring again:
     - Compare live phase state against trusted trajectory state, not only trusted action prefix.
     - At minimum, record per-step expected-vs-actual frame-change count for exact phase templates.
     - Prefer a compact, domain-general state signature: frame hash or object/topology signature from the trusted frame, plus live object/topology signature.
  2. Add phase resynchronization based on observed state signatures:
     - If exact action prefix is intact but an expected-change action has no live effect, do not assume the phase counter alone is enough.
     - Search nearby same-source phases for the trusted frame/signature that best matches the live observation, then continue from that phase.
     - Keep this diagnostic-first or low-weight until it proves better than action-index phase tracking.
  3. Only after state-alignment diagnostics identify the mismatch should behavior change again.
     - Possible behavior patch: a bounded `phase_state_alignment_bonus` or phase-resync selector using trusted object/topology signatures.
     - Do not add game ids, action hardcodes, color hardcodes, or step-283 special cases.
  4. Re-run `wa30 max_steps=500 n_runs=1` after state-alignment instrumentation, then `n_runs=3` only if the first run no longer dies at the same state divergence.

State-aware phase alignment diagnostics and exact-resync guard (2026-05-01):
- Implemented the state-alignment patch requested after the phase-stall run.
- Code changes in `claude_sandbox/arc_agent_hunter_seeker_codex.py`:
  - Added deterministic trusted-frame state signatures with `_phase_state_signature_key()`. The key is shape plus CRC32 over the discrete uint8 frame, so it is stable across Python processes and checkpoint/event dumps.
  - Added `_phase_frame_change_count()` for exact expected-vs-actual changed-cell counts.
  - `_build_phase_target_templates()` now records, for both click and non-click phase templates:
    - `frame_signature`
    - `after_frame_signature`
    - `expected_frame_change_count`
    - existing `expected_observation_changed`
    - existing `expected_level_progress`
  - `_compact_phase_template()` preserves `frame_signature`, `after_frame_signature`, and `expected_frame_change_count`.
  - `_phase_state_score_diag()` writes state-alignment terms into candidate `score_components`:
    - `phase_state_signature_available`
    - `phase_state_expected_frame_changed`
    - `phase_state_expected_frame_change_count`
    - `phase_state_template_phase`
    - `phase_state_template_action`
    - `phase_state_template_phase_offset`
  - Transition-time diagnostics now compare the pending live transition against the trusted template via `_phase_state_transition_diag()`:
    - level, source, source scope, current phase, template phase/action
    - expected vs actual frame-changed flag
    - expected vs actual changed-cell count
    - before/after exact signature matches
    - expected/live before/after signature strings
    - exact resync search result
  - `_phase_state_resync_from_frame()` searches nearby same-source solved phases for an exact before/after frame-signature match.
  - `_phase_state_apply_resync()` can apply a resync only for exact trusted same-game templates, only within the bounded phase window, and only when the live frame exactly matches a nearby trusted before/after signature. No approximate or color/action/game hardcode was added.
  - `step()` now emits `phase_state_alignment` and per-level `phase_state_resync_count` into the info dict and action trace.
  - `reset_for_new_game()` clears phase-state alignment/resync counters.
- Code changes in `claude_sandbox/summarize_event_dumps.py`:
  - The summarizer now reports `phase_state` aggregate counts:
    - `available`
    - `exact_matches`
    - `mismatches`
    - `resync_matched`
    - `resync_applied`
  - It also prints the first phase-state mismatch per run, including selected action, processed/pending action, processed phase, expected/actual change counts, before/after signature matches, and whether resync applied.
  - For old dumps that do not yet contain `phase_state_template_action`, it infers the processed action from the prior trace entry, because `phase_state_alignment` describes the pending transition consumed before the current trace entry selects its next action.
- New tests added/updated:
  - Phase templates record stable signatures and expected changed-cell counts from `frames_after`.
  - Exact same-source state resync finds a nearby after-frame signature.
  - Resync applies only to exact trusted templates and does not apply to weak/stall-escape hints.
  - Phase-state terms survive compact score-component serialization.
  - Summarizer reports phase-state aggregate counts and first mismatch details.
- CPU verification after the final summarizer tweak:
  - `py_compile` passed for touched agent/summarizer/test files.
  - Focused phase/summarizer tests passed: `21 passed, 47 deselected`.
  - Full sandbox suite passed: `296 passed, 1 skipped`.
- GPU validation run `phase_state_resync_wa30_gpu`:
  - Command used `/home/moloch/ouro_project/venv/bin/python -m claude_sandbox.train_arc_codex` outside the sandbox with CUDA-visible execution.
  - `nvidia-smi` verified the run was actually on GPU: Python compute PID `152394`, initially about `488MiB`, later about `634MiB`.
  - `ps` showed the same PID active during the run rather than a dead/stalled process.
  - Event dir: `claude_sandbox/perf_event_dumps_phase_state_resync_wa30_gpu`.
  - Result: `wa30`, `max_steps=500`, `n_runs=1`.
    - Level 1 completed at step 125.
    - Level 2 completed at step 183.
    - Death at step 283.
    - Failure classified as `topology`.
  - Summarizer output:
    - `levels total=2`, `per_run=[2]`
    - `failures topology=1`
    - `phase_templates exact_trusted=282`, `weak_template=1`, `no_template=0`
    - `phase_state available=282`, `exact_matches=248`, `mismatches=34`, `resync_matched=246`, `resync_applied=0`
    - First phase-state mismatch:
      - trace step 250
      - selected next action `2`
      - processed/pending action `4`
      - processed trusted phase `65`
      - expected changed cells `65`
      - actual changed cells `18`
      - before signature match `False`
      - after signature match `False`
      - exact resync `False`
  - Terminal topology postmortem:
    - last action `4`
    - `source=adjacent`
    - `raw_pool_count=3`
    - `candidate_count=0`
    - `rejected_count=3`
    - adjacent colors `[2, 4, 9]`
    - protected colors `[0, 2, 4, 9, 12, 14]`
    - `timeout_like_terminal=false`
    - `terminal_candidate_prior_hazard_max=0`
  - Final selected score components were populated:
    - `action=4`
    - `total_score=1.366`
    - `safety_penalty=0`
    - `hazard_reachable_penalty=0`
    - `engram_total_bias=+0.1597`
    - `phase_action_bonus=+0.9`
    - `risk_patcher_reason=no_high_risk_chosen`
- Current diagnosis after this patch:
  - The earlier phase-stall/action-prefix bug remains fixed. The agent no longer repeats the stalled phase-68 action and can keep exact phase guidance through the terminal point.
  - The new diagnostics show state divergence starts before the previously obvious phase-68 stall. The first observed exact state mismatch is the transition processed at trace step 250: trusted phase 65 action `4` expected a 65-cell frame change, but live play changed only 18 cells and no exact nearby trusted frame matched.
  - This means the live state has already left the trusted state manifold before the later step-253 stall symptom. Action-prefix equality is therefore not enough evidence that the agent is actually in the trusted trajectory state.
  - The conservative exact-frame resync did the right thing: it did not apply a false correction when no nearby trusted exact state matched.
  - The terminal topology classification is now downstream of state drift and candidate starvation. At death, all adjacent terminal-attribution candidates were rejected as protected/avatar-like; the chosen action had no positive hazard penalty and no local-contact hazard signal. This is not evidence for increasing hazard/topology penalties again.
  - Current topology state remains:
    - hazard-aware safety scoring is wired and bounded;
    - local-contact hazard scoring is in `safety_penalty` and `score_components`;
    - topology/terminal postmortem diagnostics are populated;
    - low-hazard timeout-like deaths are still protected from poisoning hazard memory;
    - the current wa30 failure is better explained as state-misaligned exact guidance plus terminal candidate starvation than as a missing direct hazard penalty.
- Recommended next patch:
  1. Add a domain-general phase-state mismatch tracker:
     - track mismatch streak per level/source/phase branch;
     - expose `phase_state_mismatch_streak` and `phase_state_alignment_trust` in `score_components`;
     - if the live before-signature no longer matches trusted state for multiple exact phases, downgrade exact hard guidance into weak/recovery guidance instead of continuing to trust the action index blindly.
  2. Add approximate state signatures before using approximate resync:
     - object/topology signature from connected components, normalized object counts, coarse centroid/area buckets, avatar/topology cues, and reachable/frontier summaries;
     - compare live state against nearby same-source trusted phases;
     - only allow approximate resync with very high confidence and diagnostic fields explaining the match.
  3. Improve topology candidate-starvation diagnostics:
     - summarize why each terminal-adjacent candidate was rejected;
     - count "all adjacent candidates protected/avatar-like" cases;
     - keep this diagnostic unless repeated evidence shows the attribution itself is wrong.
  4. Re-run `wa30 max_steps=500 n_runs=1` after the mismatch-trust patch. If it no longer dies at the same state-divergent point, run `n_runs=3`.
  5. Run mixed regression `ls20 tr87 wa30` after any behavior patch.
  6. Continue avoiding game ids, color/action hardcodes, and step-specific rules. The correct abstraction is state-aligned trust, not wa30-specific recovery.

Post-ladder phase-state trust and recovery softening patch (2026-05-01):
- Implemented the next patch after the exact-state-resync run. This is still domain/game agnostic: no game ids, colors, action ids, or step-specific rules were added.
- Code changes in `claude_sandbox/arc_agent_hunter_seeker_codex.py`:
  - Added per-game/level/source phase-state trust memory:
    - `_phase_state_mismatch_streaks`
    - `_phase_state_alignment_trust`
  - `reset_for_new_game()` clears the trust memory; `on_level_complete()` drops completed-level entries.
  - Added coarse object-state signatures:
    - `_phase_state_object_signature_vector()`
    - `_phase_state_object_signature_similarity()`
  - Phase templates now carry `object_signature_vector` and `after_object_signature_vector` in addition to exact CRC frame signatures.
  - Added approximate state resync diagnostics via `_phase_state_approx_resync_from_frame()`.
  - Approximate resync remains conservative:
    - exact trusted templates only;
    - similarity must be at least `0.9975`;
    - not applied before the source already has at least two mismatch streak counts;
    - bounded phase delta;
    - no broad action/color blacklist.
  - Added state-trust downgrade path:
    - `_phase_state_trust_key()`
    - `_phase_state_mismatch_streak_for()`
    - `_phase_state_alignment_trust_for()`
    - `_phase_state_guidance_should_downgrade()`
    - `_phase_state_trust_adjust_template()`
    - `_phase_state_update_alignment_trust()`
  - Alignment trust is currently `1.0 - 0.25 * mismatch_streak`, clipped to `[0, 1]`.
  - Exact guidance downgrades after `mismatch_streak >= 3` or `trust < 0.35`.
  - Downgraded exact templates become `same_game_state_misaligned_action_option` with capped confidence and stop receiving exact-trusted protection.
  - Approximate object alignment (`similarity >= 0.985`) can waive exact CRC/count drift only when the count mismatch is not severe.
  - Severe expected-vs-actual transition-count mismatch overrides approximate alignment:
    - `phase_state_count_mismatch_delta >= 3`
    - and `phase_state_count_mismatch_ratio >= 0.20`
  - Severe count mismatches can trigger recovery even when object-level similarity is high.
  - New transition diagnostics:
    - `phase_state_approx_aligned`
    - `phase_state_count_mismatch_delta`
    - `phase_state_count_mismatch_ratio`
    - `phase_state_severe_count_mismatch`
    - `phase_state_mismatch_streak`
    - `phase_state_alignment_trust`
    - `phase_state_guidance_downgraded`
  - New/updated score-component terms include:
    - `phase_state_mismatch_streak`
    - `phase_state_alignment_trust`
    - `phase_state_trust_penalty`
    - `phase_state_approx_similarity`
    - `phase_state_guidance_downgraded`
  - Recovery hints were softened so recovery is guidance rather than a replay rail:
    - `recovery_action_option` additive bonus reduced from `+0.34` to `+0.16`.
    - `recovery_retired_action_option` reduced from `+0.18` to `+0.12`.
    - `terminal_recovery_action_option` reduced from `+0.30` to `+0.18`.
- Code changes in `claude_sandbox/summarize_event_dumps.py`:
  - Phase-state summaries now include:
    - approximate resync matched/applied counts;
    - approximate-aligned counts;
    - severe-count-mismatch counts;
    - downgraded counts;
    - first mismatch count delta/ratio/severity details.
  - Topology death summaries include candidate-starvation diagnostics:
    - `terminal_candidate_starvation`
    - `terminal_all_adjacent_candidates_protected`
    - `terminal_rejected_safe_reason_counts`
- Tests added/updated:
  - exact guidance downgrades only after the new trust threshold;
  - low approximate similarity plus repeated exact mismatch activates recovery;
  - high approximate similarity with only tiny count drift does not false-downgrade;
  - severe count mismatch overrides approximate alignment and activates recovery;
  - approximate resync remains high-confidence only;
  - recovery action bonus is bounded below the old replay-rail strength.
- CPU verification for the final patch:
  - `py_compile` passed for touched agent/summarizer/test files.
  - Focused tests passed: `38 passed, 35 deselected`.
  - Full sandbox suite passed: `301 passed, 1 skipped`.
- GPU run 1 after the approximate-trust patch, `phase_state_trust2_wa30_gpu`:
  - CUDA verified with `nvidia-smi`: Python compute process allocated GPU memory.
  - Result: level 1 complete at step 125, level 2 complete at step 183, death at step 283.
  - Failure: `topology`.
  - Summary:
    - `phase_templates exact_trusted=282, weak_template=1, no_template=0`
    - `phase_state available=282, exact_matches=248, mismatches=34`
    - `approx_aligned=278`
    - `downgraded=0`
  - Diagnosis:
    - The early false downgrade was fixed.
    - But exact guidance stayed trusted through severe state drift from trace step 250 onward.
    - First mismatch: processed phase 65/action 4, expected 65 changed cells, actual 18, approximate similarity 1.0.
    - Terminal still had topology candidate starvation: raw adjacent pool 3, candidate count 0, all rejected as protected/avatar-like.
- GPU run 2 after severe-count mismatch gating, `phase_state_severe_trust_wa30_gpu`:
  - CUDA verified with `nvidia-smi`.
  - Result: level 1 complete at step 125, level 2 complete at step 183, death at step 283.
  - Failure: `topology`.
  - Summary:
    - `phase_templates exact_trusted=251, weak_template=29, no_template=3`
    - `phase_state available=279, exact_matches=248, mismatches=31`
    - `severe_count_mismatches=3`
    - `downgraded=1`
  - The severe gate fired correctly:
    - step 250: streak 1, trust 0.75
    - step 251: streak 2, trust 0.50
    - step 252: streak 3, trust 0.25, downgraded true, recovery activated
  - Remaining problem:
    - Recovery hints still carried too much phase bonus (`+0.4696` at terminal), so recovery remained partially replay-driven.
    - Terminal topology starvation persisted: raw adjacent pool 3, candidate count 0.
- GPU run 3 after recovery hint softening, `phase_recovery_soft_wa30_gpu`:
  - CUDA verified with `nvidia-smi`: Python compute process allocated GPU memory.
  - Result: level 1 complete at step 125, level 2 complete at step 183, death at step 283.
  - Failure changed from `topology` to `mechanism`.
  - Summary:
    - `phase_templates exact_trusted=251, weak_template=22, no_template=10`
    - `phase_state available=273, exact_matches=248, mismatches=25`
    - `severe_count_mismatches=3`
    - `downgraded=1`
    - terminal `candidate_count=1`
    - `terminal_candidate_starvation=false`
    - `terminal_all_adjacent_candidates_protected=false`
  - Terminal behavior:
    - Final action was `1`, not a phase-template action.
    - Final selected score:
      - `total_score=0.6084`
      - `phase_action_bonus=0`
      - `safety_penalty=0`
      - `hazard_reachable_penalty=0`
      - `engram_total_bias=+0.1629`
      - `risk_patcher_reason=no_high_risk_chosen`
    - Topology death attribution accepted one local candidate:
      - accepted object id `10`
      - color `2`
      - track `66`
      - prior hazard `0.10`
      - same color had protected tracks, but this specific track was not protected.
    - Directional death evidence was applied:
      - `color 2 hazard↑`
      - hazard became about `0.83`
      - failure classified as `mechanism`
- Current topology/phase state after these patches:
  - Exact phase guidance no longer blindly persists through severe state drift.
  - Approximate object-state recall is diagnostic/conservative and does not by itself blacklist actions or colors.
  - Severe state-count divergence now activates recovery at the actual drift point.
  - Recovery hints are bounded and no longer dominate the candidate list by themselves.
  - Topology terminal attribution is no longer starving on the current terminal state after recovery softening; it can identify a specific same-color-but-not-protected track as the plausible hazard.
  - Hazard-aware reachable safety remains wired and bounded.
  - Local-contact hazard scoring is wired into `safety_penalty` and score components, but its current threshold only penalizes stronger local hazard priors:
    - local contact penalty starts from hazard above about `0.18`;
    - risk patcher treats direct local hazard as hard only around `>=0.35` or penalty `>=0.05`.
  - The latest death had local hazard prior `0.10`, so the first exposure still received no safety penalty before the death. That is now a mechanism-learning case, not a topology starvation bug.
- Current diagnosis:
  - The main post-ladder topology bug found by the run was not a missing region computation. It was a trust/recovery interaction:
    1. state drift starts at step 250;
    2. exact phase guidance needed to downgrade there;
    3. recovery hints were initially too strong;
    4. once softened, topology attribution worked and learned a concrete hazard candidate.
  - The remaining open question is whether that new hazard evidence prevents the same failure on subsequent runs. A one-run probe cannot answer that because it dies on first exposure before hazard memory can help.
- Next diagnostics / future work:
  1. Run `wa30` on GPU with `n_runs=3`, current code, `max_steps=500`, no replay, to check whether run 2 or run 3 avoids the newly learned track/color-2 hazard.
  2. If repeated runs still die on a low-prior local-contact hazard, add a small, bounded, reward-aware local-contact uncertainty penalty:
     - domain agnostic;
     - directional local-contact only;
     - no color/action/game blacklist;
     - weaker when reward/exit/collectible belief is high;
     - all new scoring terms must appear in `score_components`.
  3. Run a mixed GPU regression after the next behavior patch: `ls20 tr87 wa30`.
  4. Add a summarizer flag for "mechanism death with local_contact_hazard below penalty threshold" so this class of failure is easier to spot.

Post-ladder terminal memory, engram, and local-contact ambiguity updates (2026-05-02):
- Implemented since the previous state entry, all in `claude_sandbox/arc_agent_hunter_seeker_codex.py` unless noted:
  - Terminal/topology engram source alignment:
    - Added `_engram_topology_sources()`.
    - Topology-local engram paths now treat `adjacent_terminal_override` as a topology source alongside `topology_frontier`, `adjacent`, and `fallback`.
    - This lets terminal override records participate in later topology-local recall instead of being stranded as generic terminal records.
  - Terminal basin memory:
    - Added `_terminal_outcome_basin_penalty()`.
    - Cross-action terminal-context penalty is bounded to `>= -0.28`.
    - It only activates for high-confidence terminal contexts:
      - extremely high cross-action similarity (`>=0.96` with enough evidence), or
      - multi-action terminal-context support (`action_count >= 2`, `best_sim >= 0.88`, `support >= 0.70`).
    - It does not create action, color, game, or broad mechanism blacklists.
    - New score/compact terms include:
      - `terminal_basin_penalty`
      - `terminal_basin_match`
      - `terminal_basin_similarity`
      - `terminal_basin_support`
      - `terminal_basin_match_count`
      - `terminal_basin_action_count`
      - `terminal_basin_matched`
    - Risk patcher now counts terminal basin risk and has a direct-risk helper for hard terminal basin matches.
  - Terminal memory persistence:
    - `save_checkpoint()` now persists `terminal_outcome_memory` and `terminal_outcome_prototypes`.
    - `load_checkpoint()` restores them with type guards.
  - Phase-state trust for weak/recovery templates:
    - `_phase_state_score_diag()` now applies a bounded negative `phase_state_trust_penalty` for templates with `streak >= 2` and `trust < 0.70`, not only exact downgraded templates.
    - The penalty is `-0.12 * min(3, streak) * (1 - trust)`.
  - Positive engram state-trust gate:
    - Added `_engram_state_trust_suppression_components(...)`.
    - Positive engram optimism is suppressed when phase-state mismatch streak is high and trust is low.
    - Suppression is bounded to `>= -0.20`.
    - New score terms:
      - `engram_state_trust_suppression`
      - `engram_state_trust_gate`
    - Risk patcher counts this as optimism risk.
  - Repeated terminal topology engram recall:
    - Terminal topology counterevidence scale reduced from `0.05` to `0.025`.
    - Topology-source terminal counterevidence capped at `0.40 * support_count`.
    - Topology-local recall evidence floor lowered from `0.65` to `0.60` only for repeated terminal records (`outcome_type` terminal and `support_count >= 2`).
    - This fixed the observed case where `support_count=2`, `counterevidence_count ~=0.96`, and similarity `>0.98` still produced zero topology-local penalty.
  - Low-prior local-contact ambiguity patch:
    - `_local_contact_hazard_components()` now separates:
      - `local_contact_known_hazard_penalty`
      - `local_contact_ambiguity_penalty`
      - `local_contact_ambiguity`
      - `local_contact_ambiguity_reward_gate`
      - `local_contact_ambiguity_active`
    - The ambiguity penalty targets directional contact with low/moderate local hazard belief that is below the known-hazard threshold.
    - It is bounded to `>= -0.12`, included in total `local_contact_hazard_penalty`, and therefore included in `safety_penalty`.
    - It is weaker when reward/exit/collectible belief is high.
    - It is suppressed for avatar-like/high-exit contacts.
    - It is not a hard direct-risk veto: exact guidance direct-risk checks now use `local_contact_known_hazard_penalty` when available, falling back to the old total only for older traces.
    - All new terms are compacted into `score_components`.
- Tests added/updated in `claude_sandbox/test_codex_sandbox.py`:
  - `adjacent_terminal_override` feeds topology-local engram recall.
  - Repeated topology terminal engram survives bounded counterevidence.
  - Engram state-trust suppression only suppresses positive bias when misaligned.
  - Terminal basin penalty requires high-confidence cross-action support and does not penalize unrelated candidates.
  - Local-contact ambiguity:
    - penalizes low-prior directional contact;
    - is bounded;
    - is weaker for high reward contact;
    - is not a hard direct-risk veto;
    - appears in compact score components.
- CPU verification after the latest local-contact ambiguity patch:
  - `py_compile` passed for touched files.
  - Focused tests passed: `18 passed, 63 deselected`.
  - Full sandbox suite passed: `310 passed, 1 skipped`.
- GPU probes run after terminal/engram patches, before the latest ambiguity patch:
  - `terminal_basin_wa30_n3_gpu`:
    - levels `[2, 2, 2]`
    - failures: `mechanism=1`, `planner=2`
    - terminal basin did not fire on chosen actions because cross-action context similarity was below the high-confidence threshold.
    - Diagnosis: phase trust penalty worked, but positive engram optimism could still cancel stale source trust.
  - `state_trust_engram_wa30_n3_gpu`:
    - levels `[2, 2, 2]`
    - failures: `planner=1`, `topology=2`
    - run 2 and run 3 showed candidate starvation:
      - `source=adjacent`
      - `candidate_count=0`
      - raw adjacent pool protected/rejected
    - Repeated topology-local similarity was high (`~0.983-0.985`) but support was zero because terminal counterevidence/flooring was too strict.
  - `topology_repeated_engram_wa30_n3_gpu`:
    - levels `[2, 2, 2]`
    - failures: `mechanism=1`, `planner=2`
    - Topology-local repeated terminal recall fired on the repeated bad candidate:
      - example action 3 score became about `-0.461`
      - `terminal_outcome_penalty=-0.322`
      - `engram_total_bias=-0.180`
      - `engram_topology_local_support=0.656`
      - `risk_score=0.802`
    - Remaining failure changed to a low-prior local mechanism/hazard exposure:
      - accepted adjacent candidate existed;
      - prior hazard was about `0.10-0.126`;
      - local-contact known hazard threshold did not penalize it before death.
- Latest GPU probe after the local-contact ambiguity patch:
  - Command target:
    - `wa30`
    - `max_steps=500`
    - `n_runs=3`
    - `eps=0`
    - `no_replay`
    - dump dir `claude_sandbox/perf_event_dumps_local_contact_ambiguity_wa30_gpu`
  - CUDA was verified with `nvidia-smi`:
    - Python training process allocated GPU memory during the run.
    - After completion, no training Python process remained on the GPU.
  - Summary:
    - levels `[2, 2, 2]`
    - failures: `planner=1`, `topology=2`
    - `phase_templates exact_trusted=753, weak_template=46, no_template=50`
    - `phase_state available=799, exact_matches=744, mismatches=55`
    - `severe_count_mismatches=9`
    - `downgraded=3`
  - First phase-state mismatch remains stable across all three runs:
    - trace step 250
    - selected next action `2`
    - processed/pending action `4`
    - processed trusted phase `65`
    - expected changed cells `65`
    - actual changed cells `18`
    - approximate similarity `1.0`
    - severe count mismatch true
    - initial trust `0.75`, then later downgrade/recovery
  - Run 1:
    - death at step 283
    - failure classified as `planner`
    - `source=adjacent_terminal_override`
    - raw pool `1`, candidate `1`
    - timeout-like terminal true because the candidate was late and low-hazard.
    - Final chosen action `1`, score about `0.283`.
    - Local-contact ambiguity did not fire because the scorer saw avatar overlap, not uncertain hazard contact.
  - Run 2:
    - death at step 283
    - failure classified as `topology`
    - `source=adjacent`
    - raw pool `3`, candidate `0`, rejected `3`
    - `terminal_candidate_starvation=true`
    - `terminal_all_adjacent_candidates_protected=true`
    - rejected reasons:
      - `protected_track_avatar_or_exit=3`
      - `avatar_belief_gt_0.50=2`
    - Final chosen action `2`, score about `-0.222`.
    - `local_contact_source=avatar_overlap_target`
    - `local_contact_ambiguity_penalty=0`
    - `engram_total_bias=+0.044`
    - `risk_patcher_reason=all_candidates_risky`
    - action 1 was avoided because terminal prototype/engram risk fired; action 2 had no terminal penalty.
  - Run 3:
    - death at step 283
    - failure classified as `topology`
    - `source=adjacent`
    - raw pool `2`, candidate `0`, rejected `2`
    - `terminal_candidate_starvation=true`
    - `terminal_all_adjacent_candidates_protected=true`
    - rejected reasons:
      - `protected_track_avatar_or_exit=2`
      - `avatar_belief_gt_0.50=2`
    - Final chosen action `5`, score about `0.305`.
    - `local_contact_source=adjacent_hazard_contact`
    - `local_contact_hazard` approximately zero because the adjacent object was believed avatar-like.
    - `engram_total_bias=+0.2`; no negative support matched this action/context.
- Current diagnosis after latest probe:
  - The low-prior mechanism hazard gap has been patched, but this 3-run probe no longer primarily presents as that mechanism failure.
  - The current blocker is protected-candidate terminal starvation after phase-state drift:
    1. live state leaves the trusted trajectory at step 250;
    2. severe mismatch downgrades phase guidance and enters recovery;
    3. recovery candidates often predict large reachable/reward collapse;
    4. terminal postmortem sees adjacent objects, but every raw adjacent candidate is protected as avatar/exit/avatar-like;
    5. therefore no hazard evidence is applied and topology death is recorded.
  - The local-contact ambiguity term did not fire on the fatal choices because the local scorer saw avatar-overlap/protected-avatar contacts, not an uncertain non-avatar contact.
  - Terminal recall is behaving conservatively as intended:
    - action 1 was penalized in run 2 by terminal prototype and engram risk;
    - action 2/action 5 contexts only had about `0.72` terminal-basin similarity to prior terminal actions, below the high-confidence cross-action threshold;
    - this is correct under the current rule: no broad directional cross-action blacklist.
  - Do not "fix" this by broadening terminal basin thresholds or adding action/color/game-specific bans.
- Current code state for topology and safety:
  - Hazard-aware reachable safety is wired, bounded, and included in `score_components`.
  - Local-contact known hazard scoring is wired into `safety_penalty`.
  - Local-contact ambiguity scoring is wired into `safety_penalty`, bounded, reward-aware, and diagnostic-visible.
  - Terminal memory has exact/prototype/basin paths, but basin remains high-confidence only.
  - Engram recall is risk-aware and aggregates positive/negative support, but still intentionally bounded.
  - Phase-state trust downgrades exact guidance after severe state drift; recovery no longer receives old replay-rail-scale bonuses.
  - Protected terminal candidates are still not used for hazard evidence. This avoids poisoning avatar/exit tracks, but it currently leaves some wa30 terminal states as topology-starvation/no-learning events.
- Next likely patch, if continuing behavior work:
  1. Add a domain-general protected-starvation terminal diagnostic/handling path:
     - detect late terminal states with raw adjacent candidates but all candidates protected as avatar/exit/avatar-like;
     - classify them separately from true topology failure, for example `protected_terminal_starvation`;
     - record terminal/context memory without applying hazard evidence to protected tracks;
     - keep this out of color/action/game hardcodes.
  2. Add a bounded topology-collapse/reward-collapse penalty or diagnostic:
     - use existing symbolic summary fields such as large negative `reachable_delta`, `reachable_reward_delta`, `exit_path_delta`, and `disappear_delta`;
     - make it reward/progress-aware and bounded;
     - include all new terms in `score_components`;
     - be careful because the final candidates often all share similar collapse summaries, so this may be more diagnostic than decisive.
  3. Improve local-contact diagnostics for avatar-overlap/protected-adjacent cases:
     - expose whether a candidate's local contact was skipped because the target was the avatar/protected;
     - summarize adjacent protected object beliefs in score traces, not only terminal postmortem.
  4. Mixed GPU regression after the next behavior patch: `ls20 tr87 wa30`.
- Additional-components / self-model future work:
  - `hunter_seeker_additional_components.md` is directionally right, but it should come after the current safety/state-alignment stack is stable.
  - Preferred order:
    1. Freeze Ouro and add small adapters.
    2. Build `ObservationTransition` and an observation replay buffer.
    3. Train `ChangedMaskHead` first because it directly teaches "what changed" and supports topology/actionability diagnostics.
    4. Add inverse action as a shadow diagnostic before policy use; only use inferred actions for policy if confidence is high.
    5. Add topology delta prediction after changed-mask/object-delta signals are reliable.
    6. Add self-value/internalized outcome prediction only after enough reliable transition/outcome data exists.
  - Evaluator and self-diagnosis should eventually wire into the self model as teacher/critic signals:
    - evaluator judgments can become supervised labels or calibration targets for the self model;
    - self-diagnosis can become structured self-state/outcome targets;
    - neither should become a brittle hot-path oracle that overrides local evidence every step.

Post-ladder protected-starvation, engram effective-bias, and phase-resync cleanup (2026-05-02):
- Code changes landed:
  - Added `FailureType.PROTECTED_TERMINAL_STARVATION`.
  - Added a protected terminal starvation classifier for directional terminal states where:
    - the raw adjacent terminal pool is non-empty;
    - all adjacent terminal candidates are protected as avatar/exit/avatar-like;
    - no object receives hazard evidence;
    - the failure is not timeout-like.
  - Protected-starvation outcomes now participate in terminal/context memory, terminal basin memory, engram collection/retrieval, and phase-branch failure recording as terminal causality, while still avoiding broad action/color blacklists.
  - Added source-level phase-state low-trust diagnostics for unguided/no-template candidates:
    - `phase_state_global_mismatch_streak`
    - `phase_state_global_alignment_trust`
    - `phase_state_global_source_count`
    - `phase_state_global_recovery_active`
    - `phase_state_global_low_trust`
    - `phase_state_global_gate_applied`
  - Added effective engram accounting:
    - `engram_effective_total_bias`
    - `engram_effective_progress_bonus`
    - risk-patcher positive-pressure diagnostics now use effective engram progress rather than raw progress optimism.
  - Fixed a zero-trust parsing bug:
    - `phase_state_alignment_trust=0.0` and `phase_state_global_alignment_trust=0.0` are no longer coerced to `1.0`.
    - This matters because engram state-trust suppression now actually fires at full distrust.
  - Approximate phase-state resync was made behaviorally usable but guarded:
    - A second consecutive severe count mismatch with near-perfect object-state similarity can apply approximate resync while the phase delta is still within the safe window.
    - Repeated backward approximate resync to the same target phase is treated as a phase loop, not proof of alignment.
    - Repeated backward resync now downgrades phase trust and re-enters recovery instead of letting exact guidance continue forever.
- Tests:
  - `py_compile` passed for `arc_agent_hunter_seeker_codex.py` and `test_codex_sandbox.py`.
  - Focused phase/engram tests passed.
  - Full CPU suite passed: `316 passed, 1 skipped`.
- GPU probes:
  1. `perf_event_dumps_protected_starvation_wa30_gpu`
     - CUDA verified with `nvidia-smi`.
     - Levels `[2, 2, 2]`.
     - Failures: `planner=2`, `protected_terminal_starvation=1`.
     - Confirmed the new protected-starvation classification works and no hazard evidence is applied to protected/avatar/exit tracks.
  2. `perf_event_dumps_phase_global_gate_wa30_gpu`
     - CUDA verified with `nvidia-smi`.
     - Levels `[2, 2, 2]`.
     - Failures: `planner=3`.
     - Global phase-state low-trust gate fired on fatal candidates, including an unguided candidate with `phase_state_global_gate_applied=true`.
  3. `perf_event_dumps_effective_engram_wa30_gpu2`
     - CUDA verified with `nvidia-smi`.
     - Levels `[2, 2, 2]`.
     - Failures: `mechanism=1`, `planner=1`, `protected_terminal_starvation=1`.
     - Effective engram fields were present:
       - run 1: raw `engram_total_bias=+0.157`, suppression `-0.147`, effective total `+0.0098`;
       - run 3: raw progress `+0.093`, effective progress `0.0`, effective total `-0.127`.
  4. `perf_event_dumps_approx_resync_wa30_gpu`
     - CUDA verified with `nvidia-smi`.
     - Levels `[2, 2, 2]`.
     - Failures: `protected_terminal_starvation=3`.
     - Approximate resync applied `33` times and prevented trust downgrade.
     - Diagnosis: the first approximate-resync patch created a 64-65-66 phase loop and was too optimistic.
  5. `perf_event_dumps_approx_resync_guard_wa30_gpu`
     - CUDA verified with `nvidia-smi`.
     - Levels `[2, 2, 2]`.
     - Failures: `planner=2`, `mechanism=1`.
     - Approximate resync applied only `6` times; trust downgraded `3` times.
     - The repeated-resync guard broke the bad exact-guidance loop.
  6. `perf_event_dumps_trust_zero_wa30_gpu_smoke`
     - CUDA verified with `nvidia-smi`.
     - One-run smoke after the zero-trust parsing fix.
     - Levels `[2]`.
     - Failure: `planner=1`.
     - Final chosen score components showed the intended self-state braking:
       - `phase_state_global_low_trust=true`
       - `phase_state_mismatch_streak=4`
       - `phase_state_alignment_trust=0.0`
       - `engram_total_bias=+0.102`
       - `engram_state_trust_suppression=-0.102`
       - `engram_effective_total_bias=0.0`
       - `engram_effective_progress_bonus=0.0`
- Current topology / phase diagnosis:
  - The stable divergence is now very clear:
    - first phase-state mismatch at run-relative step `250`;
    - selected next action `2`;
    - processed/pending trusted action `4`;
    - processed trusted phase `65`;
    - expected frame-change count `65`;
    - actual frame-change count `18`;
    - approximate object-state similarity `1.0`;
    - severe count mismatch true.
  - First action divergence is now consistently step `251`: actual `1`, trusted expected `4`.
  - At step `254`, the repeated backward approximate resync is detected and downgraded:
    - `phase_state_approx_resync_repeat_count=2`
    - `phase_state_approx_resync_repeat_downgraded=true`
    - `phase_state_mismatch_streak=4`
    - `phase_state_alignment_trust=0.0`
  - This is no longer a missing-topology-postmortem bug:
    - topology death diagnostics are populated;
    - protected terminal starvation is distinct from true topology failure;
    - terminal timeout-like low-hazard cases are planner-classified;
    - mechanism cases apply hazard evidence to accepted candidates;
    - protected/avatar/exit tracks are still not poisoned.
  - The remaining wa30 ceiling is phase/action-effect competence after state drift, not a reason to increase broad hazard/topology penalties.
- Current code state:
  - Hazard-aware reachable safety remains bounded and diagnostic-visible in `score_components`.
  - Local-contact known hazard and ambiguity scoring remain bounded and reward-aware.
  - Engram recall remains conservative:
    - no broad action blacklist;
    - no broad color blacklist;
    - cross-action similarity is diagnostic except very high-confidence local topology recall;
    - penalties are bounded and visible.
  - Effective engram diagnostics now show what the self-state gate actually lets through.
  - Phase exact guidance is still trusted when aligned, but repeated approximate resync loops are now self-diagnosed as misalignment.
- Next useful work:
  1. Add a compact summarizer field for repeated approximate-resync loops:
     - count repeat downgrades;
     - report the repeated target phase;
     - report the first post-downgrade action sequence.
  2. Investigate the level-3 action-effect mismatch at phase `65`:
     - compare trusted and live object deltas around steps `248-255`;
     - this likely needs observation/action-effect learning rather than another safety penalty.
  3. Mixed GPU regression after any next behavior patch:
     - `ls20 tr87 wa30`
     - verify `ls20` still clears early levels and `wa30` remains planner/mechanism/protected-starvation classified without topology poisoning.
  4. Observation-learning future work from `hunter_seeker_additional_components.md` remains the right direction:
     - `ChangedMaskHead` first;
     - inverse action only as shadow diagnostic until confident;
     - topology/action-effect delta prediction after changed-mask/object-delta reliability.
  5. Evaluator and self-diagnosis should be internalized into the self model:
     - the evaluator should become a teacher/critic for self-model calibration, not a permanent external oracle;
     - self-diagnosis should become part of the self-state that notices mismatch, surprise, hazard/progress uncertainty, memory conflict, and evaluator disagreement;
     - action choice should eventually feel mediated by the self model's own confidence/risk/contradiction estimates, with those estimates modulating the ranker and recovery logic.

Post-ladder action-effect diagnostics and underpowered-effect resync guard (2026-05-02):
- Code changes landed:
  - Added trusted-vs-live phase action-effect diagnostics in `arc_agent_hunter_seeker_codex.py`.
  - New transition diagnostics:
    - `phase_state_effect_vector_available`
    - `phase_state_expected_effect_norm`
    - `phase_state_actual_effect_norm`
    - `phase_state_effect_norm_ratio`
    - `phase_state_effect_delta_norm`
    - `phase_state_effect_similarity`
    - `phase_state_expected_before_object_similarity`
    - `phase_state_expected_after_object_similarity`
    - `phase_state_effect_count_ratio`
    - `phase_state_effect_count_shortfall`
    - `phase_state_effect_underpowered`
    - `phase_state_severe_effect_mismatch`
    - `phase_state_effect_resync_blocked`
  - These diagnostics are domain/game/action/color agnostic:
    - they compare stored trusted before/after state signatures and live before/after state signatures;
    - they use frame-change counts and coarse object-signature deltas;
    - they do not encode game ids, action bans, color bans, or wa30-specific rules.
  - Approximate phase resync now refuses to apply when a severe trusted/live count mismatch is also an underpowered action-effect mismatch.
    - Block reason: `phase_state_approx_resync_block_reason="underpowered_effect"`.
    - This directly addresses the bad case where coarse object similarity was `1.0` while the actual action effect was much smaller than trusted.
  - Severe underpowered action-effect mismatches now accelerate phase-state distrust:
    - mismatch streak increments by `2` instead of `1`;
    - this gets the source into recovery earlier without adding game-specific logic.
  - `summarize_event_dumps.py` now reports:
    - effect mismatch counts;
    - underpowered effect counts;
    - severe effect mismatch counts;
    - effect-resync blocked counts;
    - approximate-resync repeat downgrade counts/max.
- Tests:
  - Added a unit test that transition diagnostics mark an underpowered action effect.
  - Added a unit test that underpowered action effects block approximate resync and activate recovery.
  - Added summarizer fixture/assertions for effect mismatch and effect-resync blocked fields.
  - `py_compile` passed.
  - Focused tests passed:
    - phase-state tests: `13 passed, 76 deselected`;
    - summarizer test: `1 passed`.
  - Full CPU suite passed: `318 passed, 1 skipped`.
- GPU probe before the behavior guard:
  - Dump root: `claude_sandbox/perf_event_dumps_effect_diag_wa30_gpu`.
  - CUDA verified with `nvidia-smi`; training process was visible as a CUDA process using about `488-634 MiB`.
  - Runs: `3`.
  - Levels: `[2, 2, 2]`.
  - Failures: `planner=2`, `protected_terminal_starvation=1`.
  - First stable phase-state mismatch:
    - run-relative step `250`;
    - selected action `2`;
    - processed trusted action `4`;
    - processed trusted phase `65`;
    - expected frame-change count `65`;
    - actual frame-change count `18`;
    - exact signatures mismatch;
    - approximate object-state similarity `1.0`;
    - severe count mismatch true;
    - effect similarity `0.0`;
    - expected effect norm `0.04714`;
    - actual effect norm `0.03125`;
    - effect count ratio `0.2769`;
    - effect shortfall `47`;
    - effect underpowered true.
  - Aggregate effect diagnostics:
    - phase-state available `803`;
    - exact matches `744`;
    - mismatches `59`;
    - approximate resync applied `6`;
    - repeated-resync downgrade `3`;
    - severe count mismatches `12`;
    - effect mismatches `48`;
    - effect underpowered `48`.
  - Diagnosis from this run:
    - the old approximate-resync path was still too optimistic;
    - it allowed backward phase resync from the underpowered action-effect state because coarse object similarity was perfect;
    - the new effect diagnostic provided the missing reason to block that resync.
- GPU probe after the behavior guard:
  - Dump root: `claude_sandbox/perf_event_dumps_effect_resync_guard_wa30_gpu`.
  - CUDA verified with `nvidia-smi`; training process was visible as CUDA PID `199160` using about `488 MiB`, and no training process remained after completion.
  - Runs: `3`.
  - Levels: `[2, 2, 2]`.
  - Failures: `mechanism=2`, `protected_terminal_starvation=1`.
  - Aggregate phase/effect results:
    - phase-state available `785`;
    - exact matches `744`;
    - mismatches `41`;
    - approximate resync applied `0`;
    - repeated-resync downgrade `0`;
    - severe count mismatches `6`;
    - severe effect mismatches `6`;
    - effect mismatches `39`;
    - effect underpowered `32`;
    - effect resync blocked `6`;
    - guidance downgraded `3`.
  - First mismatch after the guard:
    - still at run-relative step `250`;
    - processed action `4`, trusted phase `65`;
    - expected change `65`, actual change `18`;
    - effect underpowered true;
    - severe effect mismatch true;
    - approximate resync blocked true;
    - alignment trust drops to `0.5` immediately instead of being reset by approximate resync.
  - Step window after the guard:
    - step `250`: selected action `2`, trust `0.5`, streak `2`, block reason `underpowered_effect`;
    - step `251`: processed action `2`, another severe underpowered effect, trust `0.0`, recovery active;
    - the old backward loop to phase `64` no longer happens.
- Current diagnosis after the latest GPU probe:
  - The phase-resync bug is materially improved:
    - no approximate resync applications in the post-guard run;
    - no repeat-resync loops;
    - the agent self-diagnoses the source as untrusted by step `251`.
  - The remaining wa30 failure at step `283` is now different:
    - runs 1-2 die as real `mechanism` failures;
    - run 3 dies as `protected_terminal_starvation`.
  - The mechanism deaths show a late adjacent-contact hazard gap:
    - final local contact hazard was only about `0.096-0.10`;
    - after death the mechanism postmortem reports color hazard around `0.83`;
    - `safety_penalty` was only about `-0.037` to `-0.040`;
    - `hazard_reachable_delta` was not positive, so reachable-hazard safety did not fire;
    - the candidate still carried reward/avatar belief, so the local-contact penalty stayed weak.
  - This is not a topology logging bug anymore:
    - topology death diagnostics are populated;
    - mechanism vs protected-starvation is distinguished;
    - protected tracks are not poisoned with hazard evidence;
    - terminal state and score components are visible.
  - The next likely issue is late mechanism/contact risk calibration under uncertainty, not another phase-resync patch.
- Next useful work:
  1. Add a conservative, domain-general uncertain-adjacent-contact risk term:
     - trigger from local contact hazard/reward/avatar/unknownness mixture, not color/action/game ids;
     - only when phase/global state trust is low or recovery is active;
     - bound it tightly and expose every term in `score_components`;
     - make it weaker when reward evidence is very high.
  2. Add a diagnostic summarizer for final-contact belief drift:
     - compare final chosen `local_contact_*` values against postmortem mechanism hazard;
     - report cases where pre-death hazard belief was low but postmortem hazard became high.
  3. Run a longer GPU probe after the next patch:
     - `wa30`, at least `5` runs, because mechanism evidence may need multiple deaths to propagate;
     - mixed regression `ls20 tr87 wa30` after wa30 improves.
  4. Keep observation-learning as the real long-term answer:
     - `ChangedMaskHead` and action-effect prediction should eventually explain why trusted action `4` at phase `65` produces only `18/65` expected changed cells in live state.

Post-ladder protected-overlap local-contact patch and GPU result (2026-05-02):
- Implemented the next behavior/diagnostic patch after the uncertain-contact 5-run probe showed repeated `adjacent_terminal_override` / `protected_terminal_starvation` deaths with final candidates still looking partly like avatar/protected contacts.
- Code changes in `claude_sandbox/arc_agent_hunter_seeker_codex.py`:
  - `_local_contact_hazard_components()` now detects avatar-overlap/protected-adjacent context instead of reporting these candidates as zero-risk:
    - it checks avatar-adjacent objects and a one-step shifted avatar mask;
    - it records protected/avatar/exit-like local context without applying hazard evidence to protected tracks;
    - it emits `local_contact_source=avatar_overlap_protected_context` when the target cell is still avatar-like but protected adjacent context exists.
  - New local-contact diagnostics in `score_components`:
    - `local_contact_protected_context`
    - `local_contact_protected_context_count`
    - `local_contact_protected_context_avatar`
    - `local_contact_protected_context_reward`
    - `local_contact_protected_context_hazard`
    - `local_contact_protected_context_unknownness`
    - `local_contact_protected_overlap_penalty`
    - `local_contact_protected_overlap_risk`
    - `local_contact_protected_overlap_state_gate`
    - `local_contact_protected_overlap_reward_gate`
    - `local_contact_protected_overlap_avatar_gate`
    - `local_contact_protected_overlap_context_gate`
    - `local_contact_protected_overlap_active`
  - Added `_local_contact_protected_overlap_risk_components()`:
    - domain/game/action/color agnostic;
    - only active under low phase/global state trust or recovery;
    - reward-aware, so strong exit/collectible evidence weakens it;
    - bounded to `>= -0.12`;
    - contributes through `safety_penalty`;
    - not a hard direct-risk veto for exact guidance.
  - Risk patcher now counts protected-overlap local risk in `hazard_risk`.
  - Added `_risk_patcher_has_soft_contact_risk()` so low-trust uncertain/protected local-contact risk can reroute to a close lower-risk candidate without becoming a hard blacklist.
  - `_compact_score_components()` and `_score_components_have_signal()` preserve the new fields.
- Code changes in `claude_sandbox/summarize_event_dumps.py`:
  - Summaries now include protected-overlap local-contact fields.
  - `final_contact_drift` now reports:
    - `local_contact_protected_overlap_penalty`;
    - `local_contact_protected_context`;
    - `local_contact_protected_context_count`;
    - aggregate `protected_overlap_active`.
- Tests:
  - `py_compile` passed for the touched agent/summarizer/test files.
  - Focused local-contact/risk-patcher tests passed: `12 passed, 84 deselected`.
  - Summarizer tests passed: `2 passed`.
  - Full CPU suite passed: `326 passed, 1 skipped`.
- GPU validation:
  - An initial sandboxed run was killed because it was CPU-only and not visible to `nvidia-smi`.
  - Valid CUDA run:
    - dump root: `claude_sandbox/perf_event_dumps_protected_overlap_wa30_gpu2`;
    - checkpoint dir: `claude_sandbox/checkpoints_arc_protected_overlap_wa30_gpu2`;
    - CUDA verified with `nvidia-smi`: Python compute PID `206327`, about `634 MiB`;
    - after completion, `nvidia-smi` showed no Python compute process remaining.
  - Result:
    - runs: `5`;
    - levels: `[2, 2, 2, 2, 2]`;
    - failures: `planner=2`, `protected_terminal_starvation=2`, `mechanism=1`.
  - Aggregate summary:
    - `phase_state available=1315`;
    - `exact_matches=1240`;
    - `mismatches=75`;
    - `approx_resync_applied=0`;
    - `severe_count_mismatches=10`;
    - `severe_effect_mismatches=10`;
    - `effect_underpowered=75`;
    - `effect_resync_blocked=10`;
    - `downgraded=5`.
  - First state/effect mismatch is still identical in all five runs:
    - run-relative step `250`;
    - selected next action `2`;
    - processed trusted action `4`;
    - trusted phase `65`;
    - expected changed cells `65`;
    - actual changed cells `18`;
    - exact before/after signatures mismatch;
    - approximate object similarity `1.0`;
    - effect similarity `0.0`;
    - effect count ratio `0.2769`;
    - underpowered effect true;
    - effect resync blocked true.
  - Final-contact drift:
    - `protected_overlap_active=5`;
    - `uncertain_penalty_active=1`;
    - `mechanism_low_prior=1`;
    - `protected_starvation=2`.
  - Example final chosen score components:
    - run 1 planner: `safety_penalty=-0.1005`, `local_contact_uncertain_penalty=-0.0382`, `local_contact_protected_overlap_penalty=-0.0623`, protected context count `3`.
    - run 2 protected-starvation: `local_contact_source=avatar_overlap_protected_context`, `local_contact_protected_overlap_penalty=-0.0241`, protected context count `1`.
    - run 4 planner: `local_contact_protected_overlap_penalty=-0.0866`, protected context count `3`, but `risk_patcher_reason=no_lower_risk_candidate`.
    - run 5 protected-starvation: risk patcher selected a lower-risk action, but it was still terminal-bound.
- Current diagnosis:
  - The instrumentation/scoring gap is fixed: avatar-overlap/protected-adjacent contacts are no longer invisible, and all new terms appear in `score_components`.
  - The behavior is still not solved. The agent still reaches the same step-283 terminal boundary after the stable underpowered action-effect mismatch at step 250.
  - Stronger local-contact or terminal penalties are not justified from this run alone:
    - late candidates mostly all carry terminal/protected-overlap risk;
    - rerouting chooses among bad late candidates rather than producing a genuinely safe recovery;
    - broadening terminal basin/cross-action thresholds would risk becoming an action blacklist.
  - The current bottleneck is earlier action-effect competence and recovery policy after state drift, not missing topology postmortem, not missing score components, and not a simple hazard-penalty scale issue.
- Next useful work:
  1. Add deeper action-effect diagnostics around run-relative steps `248-255`:
     - compare trusted/live object deltas, moved objects, disappear/appear components, and reachable topology deltas for the underpowered phase-65 action;
     - report which expected changed components failed to move/change in live state.
  2. Keep the next behavior patch tied to action-effect/recovery competence:
     - either a diagnostic-first `ChangedMaskHead` / observation-transition buffer;
     - or a conservative recovery candidate source based on observed live action effects, not trusted phase index replay.
  3. Do not respond by increasing broad terminal/protected penalties or adding action/color/game hardcodes.
  4. Mixed GPU regression `ls20 tr87 wa30` remains needed after the next actual behavior patch.
  5. Evaluator/self-diagnosis wiring into the self model remains future work: the self-state should eventually own mismatch/surprise/risk/evaluator disagreement instead of relying on an external diagnostic layer.

Post-ladder component action-effect diagnostics and exact pre-state override veto (2026-05-02):
- Implemented the next action-effect/state-alignment patch in `claude_sandbox/arc_agent_hunter_seeker_codex.py`.
- Component-level transition diagnostics:
  - Added component descriptors for phase before/after frames.
  - Trusted/live transition comparison now tracks changed, moved, disappeared, appeared, missing, static, and shortfall components.
  - New phase-state diagnostics include:
    - `phase_state_component_delta_available`
    - `phase_state_expected_changed_components`
    - `phase_state_live_changed_components`
    - `phase_state_component_changed_ratio`
    - `phase_state_expected_moved_components`
    - `phase_state_live_moved_components`
    - `phase_state_expected_disappeared_components`
    - `phase_state_live_disappeared_components`
    - `phase_state_expected_appeared_components`
    - `phase_state_live_appeared_components`
    - `phase_state_expected_changed_area`
    - `phase_state_live_changed_area`
    - `phase_state_component_area_ratio`
    - `phase_state_expected_component_failure_count`
    - `phase_state_expected_component_missing_count`
    - `phase_state_expected_component_static_count`
    - `phase_state_expected_component_move_shortfall_count`
    - `phase_state_component_failure_examples`
  - Live symbolic/topology deltas are now emitted beside the phase-state effect diagnostics:
    - `phase_state_live_reachable_delta`
    - `phase_state_live_frontier_delta`
    - `phase_state_live_exit_path_delta`
    - `phase_state_live_reachable_reward_delta`
    - `phase_state_live_reachable_hazard_delta`
    - `phase_state_live_moved_track_count`
    - `phase_state_live_disappeared_track_count`
    - `phase_state_live_appeared_track_count`
  - Disappearances are handled correctly: a matching expected/live disappearance is not falsely marked as a static failure.
- Raw-template retention bug fixed:
  - The selected raw phase-action template is retained internally for post-step diagnostics.
  - Event dumps still receive compact templates so traces do not balloon.
  - Click templates have the same raw/compact split.
  - This fixed the previous `component_delta_available=0` instrumentation gap.
- Exact trusted phase guidance now has a pre-state alignment gate:
  - `_phase_state_score_diag()` compares the candidate's trusted before-frame signature against the current live frame before awarding exact phase guidance.
  - New score components:
    - `phase_state_pre_signature_available`
    - `phase_state_pre_signature_match`
    - `phase_state_pre_object_similarity`
    - `phase_state_pre_alignment_penalty`
    - `phase_state_pre_alignment_downgraded`
  - Exact trusted candidates with mismatched pre-state signatures receive a bounded penalty, roughly canceling the exact phase bonus rather than hard-banning the action.
  - `_phase_exact_guidance_select_from_trace()` now vetoes exact override when all exact candidates are pre-state mismatched.
  - Veto reason: `phase_exact_override_reason="pre_state_mismatch_veto"`.
  - The risk patcher no longer treats pre-state-mismatched exact guidance as protected guidance.
- Summarizer updates in `claude_sandbox/summarize_event_dumps.py`:
  - Aggregates component diagnostics and pre-state alignment score components.
  - `first_phase_state_mismatches` and `first_phase_state_effect_mismatches` now include component counts, component failure examples, and live topology deltas.
  - Final score summary includes `phase_state_pre_alignment_penalty`.
- Tests:
  - Added/updated tests for component delta diagnostics, matching disappearance handling, exact-template pre-state mismatch penalty, exact-guidance pre-state veto, compact score fields, and summarizer component fields.
  - Full CPU suite recorded after the exact-veto patch: `330 passed, 1 skipped`.
  - Final focused verification after the GPU run:
    - `py_compile` passed for touched files.
    - Phase/exact-guidance focused tests: `19 passed, 81 deselected`.
    - Summarizer tests: `2 passed`.
- GPU probes during this patch:
  1. `perf_event_dumps_component_diag_wa30_gpu`
     - CUDA verified by `nvidia-smi`; Python was visible as a compute process.
     - Runs: `5`.
     - Levels: `[2, 2, 2, 2, 2]`.
     - Failures: `planner=2`, `protected_terminal_starvation=3`.
     - Component diagnostics were unavailable because compacted templates had dropped `component_delta_summary`.
     - This exposed the raw-template retention bug.
  2. `perf_event_dumps_component_diag_fix_wa30_gpu`
     - CUDA verified by `nvidia-smi`.
     - Runs: `1`.
     - Levels: `[2]`.
     - Failure: `protected_terminal_starvation`.
     - Component diagnostics became available.
     - At the known step-250 mismatch:
       - expected changed components `4`;
       - live changed components `3`;
       - expected moved components `3`;
       - live moved components `0`;
       - component failures `4`;
       - missing components `2`;
       - move-shortfall components `2`;
       - expected changed area `33`;
       - live changed area `18`;
       - component area ratio `0.545`.
     - Inspection showed the selected exact phase template at step 249 was for trusted phase `65`, but the live current frame did not match that template's before signature.
  3. `perf_event_dumps_prestate_gate_wa30_gpu`
     - CUDA verified by `nvidia-smi`.
     - Runs: `5`.
     - Levels: `[2, 2, 2, 2, 2]`.
     - Failures: `mechanism=1`, `planner=2`, `protected_terminal_starvation=2`.
     - The pre-state penalty appeared on selected exact templates, but exact override still forced mismatched trusted guidance:
       - step 249 had `phase_action_bonus=0.9`;
       - `phase_state_pre_alignment_penalty=-0.94`;
       - `phase_state_pre_signature_match=false`;
       - override reason still `trusted_exact_phase_guidance`.
     - This proved the penalty alone was insufficient because the exact override path bypassed normal score arbitration.
  4. `perf_event_dumps_prestate_override_veto_wa30_gpu`
     - CUDA verified by `nvidia-smi`: Python compute PID `212964`, about `634 MiB`.
     - After completion, `nvidia-smi` showed no Python compute process remaining.
     - Runs: `1`.
     - Levels: `[2]`.
     - Failure: `protected_terminal_starvation` at step `283`.
     - Exact override veto behavior is confirmed:
       - step `247`: exact guidance aligned, `phase_action_bonus=0.9`, pre-state match true.
       - step `248`: exact guidance aligned, `phase_action_bonus=0.9`, pre-state match true.
       - step `249`: selected action `2`, no phase template/bonus, `phase_exact_override_reason=pre_state_mismatch_veto`.
       - steps `250-252`: no phase template/bonus, `phase_exact_override_reason=pre_state_mismatch_veto`.
     - The original fatal exact-replay action is no longer forced at the divergence point.
- Current diagnosis:
  - The old step-250 exact-guidance bug is fixed:
    - stale exact trusted phase guidance can no longer override when the trusted before-state does not match the live current state;
    - the risk patcher no longer protects such stale exact guidance;
    - selected score components and phase-state diagnostics are populated.
  - The remaining failure is now recovery competence after trusted replay is vetoed:
    - first trusted-prefix divergence is step `249` (`actual=2`, expected trusted action `4`);
    - first phase-state mismatch appears later at step `258`;
    - mismatch still references trusted phase `65` / trusted action `4`, but the agent is now intentionally off-trace after the veto;
    - the component diagnostic at that later mismatch reports expected changed components `4`, live changed components `2`, and two missing expected components;
    - live topology deltas there are zero, so this is not evidence for increasing hazard penalties.
  - The final terminal state remains protected-starvation:
    - raw adjacent pool `3`, candidates `0`, rejected `3`;
    - all adjacent terminal candidates are protected/avatar-or-exit-like;
    - no hazard evidence is applied to protected tracks;
    - final chosen `safety_penalty=-0.02163` comes from protected-overlap context, not reachable hazard.
  - This is not an ARC/game-specific fix. The patch is based on state signature alignment, component-level before/after effect comparison, and bounded score terms. It does not hardcode games, actions, colors, or broad blacklists.
- Current topology/code state after this patch:
  - Hazard-aware reachable safety remains wired and bounded.
  - Local-contact known hazard, local-contact ambiguity, and protected-overlap contact scoring remain wired into `safety_penalty` and `score_components`.
  - Terminal/topology postmortems are populated for mechanism, adjacent override, timeout-like low-hazard planner cases, and protected-terminal-starvation cases.
  - Protected terminal candidates are still not used to apply hazard evidence to protected/avatar/exit tracks.
  - Topology-local engram recall remains conservative and bounded; state-trust suppression prevents positive engram optimism from overriding a distrusted phase source.
  - Phase-state diagnostics now distinguish:
    - exact signature mismatch;
    - approximate object similarity;
    - underpowered action effects;
    - component-level missing/static/move-shortfall failures;
    - live symbolic/topology deltas.
  - Exact trusted guidance is now conditional on live pre-state alignment.
- Next useful work:
  1. Do not raise broad hazard, terminal, topology, or engram penalties from this run.
  2. Add a conservative recovery candidate source after exact guidance is vetoed:
     - based on observed live action effects and local transition history;
     - not based on replaying the stale trusted phase index;
     - still subject to terminal/hazard/local-contact guards.
  3. Add a summarizer section for post-veto recovery windows:
     - first `pre_state_mismatch_veto`;
     - next 10 actions;
     - whether any candidate had positive live topology/progress deltas;
     - whether all available late candidates were terminal/protected-overlap risky.
  4. Longer GPU probe after a recovery patch:
     - `wa30`, `n_runs>=3`, `max_steps=500`;
     - then mixed GPU regression `ls20 tr87 wa30`.
  5. Observation-learning remains the right larger direction:
     - `ChangedMaskHead` first;
     - inverse action model only as shadow diagnostic until reliable;
     - self-value/shadow evaluator only after hard terminal/hazard preference tests.
  6. Evaluator/self-diagnosis should be wired into the self model later:
     - the self-state should own pre-state mismatch, action-effect surprise, component failures, memory conflict, hazard/progress uncertainty, and evaluator disagreement;
     - this should make the system's recovery choice feel mediated by its own confidence/risk/contradiction state rather than by an external post-hoc diagnostic layer.

Post-ladder live-effect recovery patch and GPU result (2026-05-02):
- Implemented the conservative recovery-candidate patch in `claude_sandbox/arc_agent_hunter_seeker_codex.py`.
- Purpose:
  - after exact trusted phase guidance is vetoed by a live pre-state mismatch, let the agent prefer candidates supported by recent observed live effects;
  - keep the patch domain/game agnostic;
  - avoid replaying stale trusted phase indices as recovery;
  - avoid broad action, color, object, or game blacklists.
- Exact-veto recovery state:
  - added per-level exact-veto recovery windows via `_phase_exact_veto_steps`;
  - `_phase_exact_guidance_select_from_trace()` now activates a short recovery window when all exact trusted candidates are vetoed by pre-state mismatch;
  - new emitted diagnostics:
    - `phase_exact_veto_active`
    - `phase_exact_veto_steps_left`
    - `phase_state_global_exact_veto_active`
    - `phase_state_global_exact_veto_steps_left`
    - `phase_recovery_activated_by="pre_state_mismatch_veto"`
    - `phase_recovery_activated_level`
- Live effect memory:
  - added `_live_recovery_effect_stats`, keyed by `(game, level, action)` for non-click actions only;
  - click actions are deliberately unsupported by this memory path to avoid broad click blacklists or broad click optimism;
  - memory stores exponential moving averages for:
    - progress;
    - visible change;
    - topology/progress deltas;
    - reachable reward deltas;
    - reachable hazard deltas;
    - failure/harm signals;
    - count and last step.
  - live effect stats are persisted in checkpoints as `live_recovery_effect_stats`.
- Live recovery scoring:
  - added `_live_recovery_summary_terms()` and `_live_recovery_effect_components()`;
  - scoring combines observed live history with the candidate's current symbolic transition summary;
  - actual score impact is gated to recovery/escalation/reseed/exact-veto/low-trust contexts;
  - the term is bounded to approximately `[-0.24, +0.30]`;
  - all new terms are present in score components and compacted event dumps:
    - `live_recovery_active`
    - `live_recovery_bonus`
    - `live_recovery_potential_score`
    - `live_recovery_selection_score`
    - `live_recovery_history_score`
    - `live_recovery_predicted_score`
    - `live_recovery_count`
    - `live_recovery_progress_ema`
    - `live_recovery_change_ema`
    - `live_recovery_topology_ema`
    - `live_recovery_reward_ema`
    - `live_recovery_hazard_ema`
    - `live_recovery_failure_ema`
    - `live_recovery_uncertainty`
    - `live_recovery_state_gate`
    - `live_recovery_risk_gate`
    - `live_recovery_reason`
- Live recovery selector:
  - added `_live_recovery_select_from_trace()` after exact phase guidance selection and before the risk patcher;
  - selector is active only under exact-veto/recovery/escalation/reseed pressure;
  - it refuses candidates with stale exact pre-state mismatch or hard direct risk;
  - it can replace the current beam choice only when a live-supported candidate is close enough in score, has enough recovery margin, and is not worse on direct terminal risk;
  - new selector diagnostics:
    - `live_recovery_selector_active`
    - `live_recovery_selector_selected`
    - `live_recovery_selector_reason`
    - `live_recovery_selector_candidate_count`
    - `live_recovery_selector_score_budget`
    - `live_recovery_selector_score_gap`
    - `live_recovery_selector_recovery_margin`
    - `live_recovery_selector_risk_score`
    - plus selected/replaced action/click fields in the step info when it fires.
- Summarizer updates:
  - `claude_sandbox/summarize_event_dumps.py` now includes the live-recovery and exact-veto fields in summary score components.
- CPU verification:
  - `py_compile` passed for touched files.
  - Focused live-recovery/phase tests after the initial patch: `6 passed, 97 deselected`.
  - Full CPU suite after the initial patch: `333 passed, 1 skipped`.
  - After the loss-aware regression fix:
    - focused tests: `7 passed, 97 deselected`;
    - full CPU suite: `334 passed, 1 skipped`.
- First GPU probe, before loss-aware correction:
  - Dump dir: `claude_sandbox/perf_event_dumps_live_recovery_wa30_gpu`.
  - CUDA was verified by `nvidia-smi`; Python was visible as a compute process at roughly `488-634 MiB`.
  - Command shape:
    - `wa30`, `n_runs=3`, `max_steps=500`, `eps=0.0`, `--no_replay`, trusted-plus-expanded trajectories, encoder-only checkpoint.
  - Result:
    - levels: `[2, 2, 2]`;
    - failures: `planner=2`, `protected_terminal_starvation=1`;
    - first divergence in runs 2 and 3 selected action `5` at step `249` through `selection_method="live_recovery_beam"`.
  - Diagnosis from this failed probe:
    - live recovery was wired and capable of selecting;
    - but `_live_recovery_summary_terms()` did not penalize large negative `reachable_delta` / `reachable_reward_delta`;
    - action `5` at the divergence had very negative reachability/reward deltas, yet received a positive live-recovery potential;
    - this was a real scoring bug in the new patch, not a topology bug.
- Loss-aware correction:
  - `_live_recovery_summary_terms()` now subtracts generic losses for negative reachable, frontier, reward, and exit-path deltas;
  - added regression test `test_live_recovery_predicted_score_penalizes_reachability_loss()`;
  - the fix is still game/domain agnostic: it only uses symbolic transition deltas already computed by topology/symbolic scoring.
- Second GPU probe, after the loss-aware correction:
  - Dump dir: `claude_sandbox/perf_event_dumps_live_recovery_loss_wa30_gpu`.
  - CUDA was verified by `nvidia-smi`; Python was visible as a compute process at roughly `488-634 MiB`.
  - After completion, `nvidia-smi` showed no Python compute process remaining.
  - Result:
    - runs: `3`;
    - levels: `[2, 2, 2]`;
    - failures: `planner=2`, `protected_terminal_starvation=1`;
    - phase templates: `exact_trusted=745`, `weak_template=66`, `no_template=38`;
    - phase-state exact matches: `744`;
    - phase-state mismatches: `66`;
    - effect mismatches: `63`;
    - effect-underpowered count: `61`;
    - exact override reasons included `pre_state_mismatch_veto=3`;
    - live recovery selector did not fire after the loss-aware fix.
  - First divergences after the corrected patch:
    - run 1: step `249`, actual action `2`, trusted expected action `4`, method `beam_search`;
    - run 2: step `249`, actual action `2`, trusted expected action `4`, method `beam_search`;
    - run 3: step `250`, actual action `4`, trusted expected action `2`, method `beam_search`.
  - Step-249 corrected behavior:
    - chosen candidate had `live_recovery_potential_score=-0.24`;
    - `live_recovery_predicted_score` was strongly negative;
    - `phase_exact_override_reason="pre_state_mismatch_veto"`;
    - no candidate had enough positive live recovery support to be selected.
  - Step-250 to step-255 behavior:
    - destructive reachability/reward-loss candidates now receive `live_recovery_bonus=-0.24` in the recovery window;
    - this confirms the original live-recovery over-optimism is fixed.
  - Remaining terminal:
    - run 1: planner/adjacent-terminal path, final safety from protected-overlap;
    - run 2: protected-terminal-starvation;
    - run 3: planner/adjacent-terminal path, final safety from protected-overlap;
    - deaths still cluster around step `283`.
- Current diagnosis after the live-effect patch:
  - Exact stale trusted replay is no longer being forced.
  - Live recovery scoring is now conservative and loss-aware.
  - The recovery selector does not fire after the corrected patch because the post-veto candidate set does not contain a genuinely supported productive recovery candidate.
  - This is not a penalty-scale problem:
    - broad hazard/topology/engram penalties should not be increased from this run;
    - the late candidates are mostly reachability/reward destructive or terminal/protected-overlap risky;
    - topology diagnostics are populated and are not showing hidden positive topology that the scorer ignores.
  - The stable remaining failure is action-effect competence after phase-65 drift:
    - expected component changes from trusted phase `65` are not reproduced live;
    - repeated diagnostics show missing expected components and move-shortfall components;
    - the agent has enough diagnostics to know stale replay is distrusted, but the candidate generator/model does not yet propose an actually productive recovery action.
- Current topology/code state after this patch:
  - Hazard-aware safety scoring remains wired and bounded.
  - Engram recall remains conservative and bounded.
  - Phase exact guidance is conditional on live pre-state alignment.
  - Exact-veto recovery now exists as a stateful condition.
  - Live recovery memory can bias or penalize existing candidates under recovery pressure.
  - Live recovery does not invent new candidates; it can only score/select among the candidate trace the current planner already produced.
  - This is the key limitation exposed by the corrected GPU run.
- Next useful work:
  1. Add the post-veto recovery-window summarizer section originally deferred:
     - first `pre_state_mismatch_veto`;
     - next 10 actions;
     - candidate recovery scores;
     - positive/negative live topology deltas;
     - terminal/protected-overlap rejection pressure.
  2. Add candidate-generation diagnostics around post-veto windows:
     - which actions/clicks are present in the beam;
     - which are absent;
     - whether the symbolic simulator predicts any positive local transition at all.
  3. Start the observation-learning path from `hunter_seeker_additional_components.md`:
     - `ChangedMaskHead` first as a shadow diagnostic;
     - inverse action model second, still shadow-only until reliable;
     - use those diagnostics to identify which candidate effects the current symbolic/topology stack cannot predict.
  4. Only after shadow transition diagnostics are reliable, add a conservative recovery candidate generator:
     - propose candidates from learned/observed local transition effects;
     - keep terminal/hazard/local-contact guards hard;
     - keep it diagnostic-first before giving it strong selection power.
  5. Longer GPU regression once candidate-generation diagnostics exist:
     - `wa30`, `n_runs>=5`, `max_steps>=600`;
     - then mixed `ls20 tr87 wa30` to make sure the recovery machinery remains domain agnostic.

Post-veto candidate-generation diagnostics and representation diagnosis (2026-05-02):
- Implemented diagnostic-only post-veto candidate availability instrumentation in `claude_sandbox/arc_agent_hunter_seeker_codex.py`.
- Purpose:
  - answer whether level-3 recovery fails because the useful action is absent from the beam, present but scored badly, or present but predicted as no-op/harmful;
  - keep the diagnostic domain/game agnostic;
  - do not add a new behavior selector before the representation question is answered.
- New agent diagnostics:
  - `_post_veto_candidate_generation_diag()` emits a bounded summary for exact-veto windows.
  - It stores:
    - full-trace candidate count and unique action/click count;
    - action histogram;
    - positive / negative / no-op / risky candidate counts;
    - direct-risk, terminal-risk, hazard-risk candidate counts;
    - exact trusted aligned vs pre-state-mismatched counts;
    - chosen candidate rank and prediction class;
    - best candidate by score;
    - best candidate by live-recovery score;
    - best candidate by generic predicted progress;
    - top candidates by each of those three views.
  - Generic predicted-progress score uses already-emitted symbolic/topology fields only:
    - reachable/frontier/exit/reward deltas;
    - hazard deltas;
    - moved/appeared/disappeared counts;
    - symbolic model probabilities;
    - live recovery potential;
    - existing symbolic/topology bonuses and safety penalties.
  - This diagnostic does not hardcode ARC, `wa30`, colors, actions, or object identities.
- New summarizer support in `claude_sandbox/summarize_event_dumps.py`:
  - `post_veto_recovery` section per run;
  - aggregate counts for runs with post-veto windows;
  - window size;
  - trusted expected action present/absent counts;
  - positive/no-positive candidate steps;
  - exact-mismatch steps;
  - risky-or-harmful-only steps;
  - live recovery selector reasons;
  - chosen prediction classes.
- Tests:
  - `py_compile` passed for touched files.
  - Focused tests: `8 passed, 98 deselected`.
  - Full CPU suite: `106 passed`.
  - New tests:
    - `test_post_veto_candidate_generation_diag_classifies_beam_support()`;
    - `test_post_veto_recovery_summary_reports_expected_action_availability()`.
- GPU probe:
  - Correct GPU run: `claude_sandbox/perf_event_dumps_post_veto_diag_wa30_gpu2`.
  - Command shape:
    - `wa30`, `n_runs=3`, `max_steps=500`, `eps=0.0`, `--no_replay`, trusted-plus-expanded trajectories, encoder-only checkpoint.
  - CUDA verification:
    - initial direct approved Python run showed `/home/.../venv/bin/python` in `nvidia-smi` as a compute process;
    - memory rose from about `488 MiB` to `634 MiB`;
    - after completion `nvidia-smi` showed no Python compute process.
  - Note:
    - an earlier attempt with shell env assignments launched inside the filesystem sandbox and was CPU-bound; it was terminated and ignored.
- GPU result:
  - runs: `3`;
  - levels: `[2, 2, 2]`;
  - failures: `planner=2`, `protected_terminal_starvation=1`;
  - deaths still occur around step `283`;
  - phase templates: `exact_trusted=744`, `weak_template=62`, `no_template=43`;
  - phase exact override reasons:
    - `already_exact_phase_guidance=716`;
    - `trusted_exact_phase_guidance=28`;
    - `pre_state_mismatch_veto=3`.
- New post-veto diagnosis:
  - Post-veto windows found in all 3 runs.
  - Window steps inspected: `30`.
  - Trusted expected action was present in the candidate set on all `30/30` post-veto steps.
  - Trusted expected action was absent on `0/30` steps.
  - Positive candidate steps: `0/30`.
  - No-positive-candidate steps: `30/30`.
  - Risky-or-harmful-only steps: `30/30`.
  - Live recovery selector reason: `no_supported_recovery_candidate=30`.
  - Chosen prediction class: `harmful=30`.
  - First post-veto step in each run:
    - run 1:
      - step `249`;
      - chosen action `5`;
      - trusted expected action `4` was present;
      - candidate count `5`;
      - all 5 candidates harmful;
      - chosen action `5`: `reachable_delta=-23`, `reachable_reward_delta=-6.514`, `live_recovery_score=-0.24`;
      - best predicted progress was action `2`, still harmful: `reachable_delta=-20`, `reachable_reward_delta=-4.803`, `live_recovery_score=-0.24`;
      - trusted expected action `4` was exact-trusted but pre-state mismatched and harmful.
    - run 2:
      - step `249`;
      - chosen action `5`;
      - trusted expected action `4` present;
      - all 5 candidates harmful;
      - chosen action `5`: `reachable_delta=-27`, `reachable_reward_delta=-7.926`, `live_recovery_score=-0.24`;
      - best predicted progress was action `2`, still harmful.
    - run 3:
      - step `249`;
      - chosen action `5`;
      - trusted expected action `4` present;
      - all 5 candidates harmful;
      - trusted expected action `4` was also the best predicted-progress candidate, but still harmful and pre-state mismatched.
- Updated diagnosis:
  - The immediate level-3 blocker is not candidate absence.
  - The expected trusted action is in the beam, but it is pre-state-mismatched and the current symbolic/topology model predicts harmful reachability/reward loss.
  - Every available candidate in the post-veto window is predicted harmful.
  - That makes the current blocker representational in the practical sense:
    - the system can notice stale replay;
    - it can see all available options look bad under the current symbolic/topology model;
    - it cannot yet represent which transition effect would repair the live state.
  - A pure scoring/penalty patch is unlikely to solve this now:
    - increasing hazard, topology, terminal, or engram penalties would only choose among all-negative candidates;
    - choosing the least harmful candidate is available as a fallback, but prior probes with action `2` still died and this does not address the missing transition understanding.
  - The next real patch should be representation/transition diagnostic-first, not another score-scale adjustment.
- Current code state:
  - Post-veto diagnostics are diagnostic-only.
  - No new behavior selector was added in this patch.
  - Exact-veto and live-recovery behavior from the prior patch remains unchanged.
  - `top_candidates` in action traces remains a compact trace slice and can omit the chosen candidate; `chosen_candidate` and `post_veto_candidate_generation` should be used for serious post-veto inspection.
- Next useful work from this result:
  1. Start `ChangedMaskHead` as a shadow transition diagnostic:
     - predict changed cells/components for each candidate;
     - compare predicted changed mask to actual post-step changed mask;
     - record whether trusted expected action looks harmful only because symbolic reachability is misreading the transition.
  2. Add inverse-action shadow diagnostics after changed-mask quality is measurable:
     - given before/after, predict which action caused the transition;
     - use it to detect action-effect aliasing and underpowered-effect confusion.
  3. Add a post-veto "all candidates harmful" fallback only after representation diagnostics:
     - if implemented, it should choose least predicted harm under hard terminal/hazard guards;
     - it should be treated as a safety fallback, not as a solution to level 3.
  4. Add candidate-effect oracle diagnostics from trusted trajectories:
     - when expected trusted action is present but pre-state mismatched, compare current predicted effect with nearest trusted local effect;
     - keep it diagnostic-only so stale replay does not regain override authority.

Architecture audit notes from Opus review (2026-05-02):
- These are future-work/audit notes, not behavior changes in this patch.
- `action_adapters_codex.py`:
  - `ArcActionAdapter.decode()` fallback can map unknown indices toward action `0` / reset-like behavior.
  - That is acceptable for the ARC adapter, but the fallback policy should eventually be adapter-specific rather than living as a domain-neutral default.
- `grid_encoder_codex.py`:
  - dynamic patch padding and dynamic 2D sinusoidal position encoding are considered sound;
  - the backwards-compat re-export of action heads is in migration limbo and should either be declared stable or removed with callers updated.
- `self_model.py`:
  - the split between `SelfModel` and `CortexMonitor` is architecturally right;
  - identity-start initialization is consistently respected;
  - `AgentEventBundle` is a good explicit coupling boundary;
  - fixed affective decay/excitation constants are a limitation:
    - currently hand-coded and outside autograd;
    - eventually should be learned, derived, or explicitly defended;
  - binary `avatar_identified` stress input should eventually become graded confidence to avoid brittle affective flips;
  - `TemporalContextAggregator` should be reviewed for clean cortex-off ablations:
    - when `cortex_feature is None`, the cortex projection bias may still create a learned constant channel;
    - either zero that bias deliberately or bypass cortex projection to exact zeros in cortex-off mode.
- `train_arc_codex.py`:
  - partial checkpoint loading and forced encoder freeze are sound;
  - terminal anchor retry is sound;
  - baseline mode disabling weight saving is sound;
  - broad `except Exception` around `env.step()` is too permissive:
    - it can hide real bugs behind logs;
    - add a narrowed exception class set and/or per-run exception threshold.
- `arc_agent_pairwise_stockfish_codex.py`:
  - identity-start temporal ranker head is architecturally consistent;
  - `AttnResLoopPooler` design is good and paper-relevant;
  - future diagnostic:
    - inspect `last_attn_weights` distributions empirically;
    - if weights are peaked/input-dependent, this supports the residual-loop specialization story;
    - if near-uniform, attention is not adding much over averaging.
  - `TransitionReplayBuffer` is a future refactor target:
    - pair-sourcing strategies should eventually be extracted from the monolithic buffer into separate strategy classes.

Observation-learning shadow patch and audit (2026-05-02):
- Implemented the first generic observation-learning layer from `hunter_seeker_additional_components.md`.
- New file: `claude_sandbox/observation_learning_codex.py`.
- New generic transition substrate:
  - `ObservationTransition` stores before/after frames, optional action/click labels, action confidence, terminal/progress labels, source, and expert flag.
  - `ObservationReplayBuffer` stores known-action and unlabeled watched transitions.
  - Sampling methods now include:
    - `sample_known_action_batch()`;
    - `sample_unlabeled_transition_batch()`;
    - `sample_click_transition_batch()`;
    - `sample_topology_delta_batch()` as first-pass visual-change sampling;
    - `sample_object_contrastive_batch()` as conservative positive-pair scaffolding until real object deltas exist.
  - `segment_video_frames()` converts raw frame/video streams into sparse before/after transitions.
  - `to_discrete_frame_tensor()` accepts dense game grids and RGB/video-like arrays; the current RGB path quantizes to grayscale bins for a domain-neutral first pass.
- New shadow heads:
  - `ChangedMaskHead` predicts/localizes changed cells for a before/after transition.
  - `EffectSummaryHead` predicts a generic 9-dimensional effect summary:
    - changed fraction;
    - changed-any flag;
    - mean absolute delta;
    - changed bounding-box center/extent;
    - terminal flag;
    - progress delta.
  - `InverseActionModel` predicts:
    - action logits;
    - click heatmap;
    - label-confidence logit.
  - All three operate over generic dense before/after frames rather than ARC-specific colors/actions.
- New transition-effect engram memory:
  - `TransitionEffectEngramMemory` stores generic effect vectors plus optional action/outcome/source.
  - Empty memory returns no-op zero diagnostics.
  - Recall aggregates top-k support and reports:
    - `obs_engram_support`;
    - positive/negative support;
    - positive/negative best similarities;
    - match count;
    - same-action count;
    - conflict flag.
  - It is diagnostic-only and does not add penalties or bonuses to candidate scores.
- Agent integration in `claude_sandbox/arc_agent_pairwise_stockfish_codex.py`:
  - `PairwiseARCSearchAgent` now owns:
    - `observation_buffer`;
    - `transition_effect_engram_memory`;
    - `changed_mask_head`;
    - `effect_summary_head`;
    - `inverse_action_model`;
    - `observation_optimizer`.
  - Trusted trajectories are copied into the observation buffer during `load_solved_trajectories()`.
  - Live executed transitions are copied into the observation buffer in `step()`.
  - `ingest_observation_video()` is the public hook for watched/unlabeled frame streams.
  - `_train_observation_learning()` trains the shadow heads during `train_step()`.
  - Observation heads are checkpointed and optionally loaded from checkpoints.
  - `score_candidates()` / `_score_candidates_core()` now accepts `current_frame`.
  - Candidate score components now include diagnostic observation terms:
    - `obs_changed_mask_mass`;
    - `obs_actual_changed_fraction`;
    - `obs_effect_changed_fraction_pred`;
    - `obs_inverse_action_prob`;
    - `obs_inverse_label_confidence`;
    - all `obs_engram_*` support/conflict fields.
  - These terms are currently zero-behavior: they appear in score components but do not change `total_score`.
- Hunter-Seeker trace/log integration:
  - `HunterSeekerAgent.score_candidates()` forwards the real current frame into the base observation diagnostics.
  - Compact score components preserve the new `obs_*` scalar fields and `obs_engram_conflict_flag`.
  - Action trace entries now include observation buffer/update/engram counts and observation-loss metrics when present.
  - `train_arc_codex.py` now prints `ObsL` in periodic step logs.
- Tests:
  - `py_compile` passed for:
    - `claude_sandbox/observation_learning_codex.py`;
    - `claude_sandbox/arc_agent_pairwise_stockfish_codex.py`;
    - `claude_sandbox/arc_agent_hunter_seeker_codex.py`;
    - `claude_sandbox/train_arc_codex.py`;
    - `claude_sandbox/test_codex_sandbox.py`.
  - Full sandbox CPU test suite:
    - `110 passed`.
  - New/covered tests:
    - video segmenter emits unlabeled transitions;
    - observation buffer supports unlabeled/click/topology/object-contrastive sampling;
    - changed-mask/effect/inverse heads handle dynamic non-64x64 shapes;
    - transition-effect engram memory returns no-op when empty and does not penalize unrelated vectors;
    - pairwise agent trains observation heads and emits candidate observation diagnostics.
- GPU / CUDA verification:
  - Direct CUDA check:
    - `torch.cuda.is_available() == True`;
    - device is `NVIDIA GeForce RTX 5070 Ti Laptop GPU`;
    - direct agent instantiation reports `DEVICE cuda`;
    - ranker and observation-head parameters are on `cuda:0`.
  - A deliberate CUDA allocation appeared in `nvidia-smi` as `/home/.../venv/bin/python`, confirming the sandbox-visible Python can use the GPU.
  - ARC smoke command:
    - game `wa30`;
    - `n_runs=1`;
    - `max_steps=80`;
    - `eps=0.0`;
    - encoder-only checkpoint;
    - trusted-plus-expanded trajectories;
    - output: `claude_sandbox/perf_event_dumps_observation_learning_smoke_gpu`.
  - Smoke result:
    - run completed without crash;
    - levels completed: `0` within the short 80-step smoke;
    - checkpoint saved: `claude_sandbox/checkpoints_observation_learning_smoke_gpu/arc_wa30_run1.pt`;
    - chosen score components in the measurement dump include the new `obs_*` fields.
  - Note:
    - the live ARC smoke did not show sustained GPU utilization in `nvidia-smi` when sampled after completion; the direct CUDA/device checks confirm the model tensors are on CUDA, but this short encoder-only smoke has brief kernels and is not a good utilization benchmark.
- Current architectural state after this patch:
  - Observation learning is present and trainable, but still shadow-only.
  - It does not yet repair level 3, choose actions, override topology, or propose candidates.
  - Changed-mask learning currently uses before/after frames, so it is best interpreted as effect-localization/diagnostic supervision, not yet as a latent-only predictive effect model.
  - The inverse-action head is trained only on known-action labels; unlabeled/video transitions are not used for policy training.
  - Video support exists as a generic ingestion path, but robust video learning still needs:
    - better event stabilization/flicker filtering;
    - a real visual adapter instead of grayscale binning;
    - high-confidence inferred-action quarantine before any policy use.
  - Transition-effect engram recall is intentionally conservative:
    - no broad action blacklist;
    - no score penalties/bonuses;
    - cross-action similarity is diagnostic context only;
    - as of the 2026-05-14 teacher-training regression inspection, terminal
      and hazard recalls are task-scoped so negative safety evidence from one
      environment cannot poison another environment's trusted continuation.
      Progress/change recalls remain transferable across tasks.
  - The effect vector is still coarse; early smoke shows high positive observation-engram support on many candidates, so this memory should remain diagnostic until topology/object/event deltas are richer.
- Architecture audit findings from this implementation pass:
  1. `train_arc_codex.py` still catches broad `Exception` around `env.step()`.
     - This remains a real risk: repeated SDK/network/runtime bugs can be hidden behind logs.
     - Permanent fix: narrow expected environment exceptions and add a per-run exception threshold.
  2. Observation metrics are now in action traces and score components, but `measurement_summary()` does not yet aggregate observation-learning losses or buffer counts.
     - Permanent fix: add an observation-learning measurement block.
  3. The observation heads are three separate transition encoders.
     - This is clean for first-pass isolation, but compute can be reduced later by sharing the transition encoder.
  4. The current video path is deliberately simple.
     - Good enough to establish the generic watching interface.
     - Not yet enough for serious natural video learning.
  5. `SelfModel` still does not own the evaluator/self-diagnosis loop.
     - User intent: evaluator and self-diagnosis should eventually feel like the same system making choices and noticing its own issues.
     - Future work: expose evaluator/self-diagnosis signals through `AgentEventBundle` and/or the temporal context aggregator, rather than only as external logs.
  6. Opus audit items remain valid:
     - adapter fallback policy should be made explicitly adapter-owned;
     - cortex-off aggregation should avoid a learned constant cortex channel;
     - AttnRes loop-pooler attention distributions should be measured empirically;
     - replay pair-sourcing should eventually be extracted from the monolithic buffer.
- Next useful work:
  1. Add an observation-learning summary block to `measurement_summary()`:
     - buffer size;
     - update count;
     - last/EMA observation loss;
     - changed-mask IoU;
     - inverse-action accuracy;
     - observation-engram support/conflict rates.
  2. Add actual post-step observation diagnostics:
     - compare candidate-predicted changed/effect summaries against the real next transition after `env.step()`;
     - this is the bridge between shadow heads and the level-3 representational diagnosis.
  3. Add topology/object/event delta targets to the observation path:
     - moved/disappeared/appeared/split/merge-like summaries;
     - reachability/frontier/reward/hazard deltas where a domain adapter can provide them;
     - keep the base interface domain-neutral.
  4. Only after those diagnostics are reliable, consider using observation-effect memory for conservative candidate generation or recovery selection under hard terminal/hazard guards.

Observation outcome calibration and predicted-effect trust gate (2026-05-02):
- Implemented the next observation-learning patch after the shadow heads.
- Purpose:
  - compare candidate-predicted effects against the real transition after `env.step()`;
  - expose when predicted successor frames are unreliable;
  - prevent predicted-frame symbolic/topology effects from dominating behavior when the world model is empirically hallucinating.
- New post-step diagnostics:
  - `PairwiseARCSearchAgent` now keeps a private one-step `_last_full_search_trace` with predicted frames.
  - `train_arc_codex.py` calls `record_observation_outcome_diagnostics()` after `env.step()` when before/after frames are available.
  - The diagnostic compares the chosen candidate's predicted successor to the real successor and records:
    - `obs_real_changed_fraction`;
    - `obs_predicted_changed_fraction`;
    - `obs_predicted_changed_fraction_error`;
    - `obs_predicted_effect_l1`;
    - `obs_predicted_frame_cell_accuracy`;
    - `obs_predicted_frame_exact_match`;
    - `obs_head_changed_fraction_pred`;
    - `obs_head_changed_fraction_error`;
    - carried-through inverse-action and observation-engram diagnostics.
  - Hunter-Seeker action traces now attach `observation_outcome_diag` for every step where before/after frames exist.
- New measurement summary block:
  - `measurement_summary()["observation_learning"]` now reports:
    - observation buffer size;
    - transition-effect engram size;
    - update count;
    - loss EMA;
    - last training diagnostics;
    - last outcome diagnostics;
    - outcome diagnostic count;
    - mean predicted changed-fraction error;
    - mean head changed-fraction error;
    - mean predicted effect L1;
    - mean predicted-frame cell accuracy;
    - changed-any match rate;
    - exact predicted-frame match rate.
- New predicted-effect trust gate:
  - `_observation_effect_trust_components()` combines:
    - candidate-local `effective_confidence` / world-model confidence;
    - empirical recent predicted-vs-real changed-fraction error.
  - If there is not enough outcome history, calibration is `1.0`.
  - Once enough history exists, calibration is `clip(1 - mean_error, 0.05, 1.0)`.
  - `predicted_effect_trust = candidate_wm_confidence * calibration`.
  - This remains domain/game agnostic and is based only on generic before/after effect accuracy.
- Behavioral gating added in `HunterSeekerAgent.score_candidates()`:
  - predicted-frame-derived terms are now multiplied by `predicted_effect_trust`:
    - `hazard_reachable_penalty`;
    - `symbolic_bonus`;
    - `symbolic_model_bonus`;
    - fallback semantic scoring receives `predicted_effect_trust` instead of raw candidate WM confidence.
  - Raw values remain in score components:
    - `hazard_reachable_penalty_raw`;
    - `symbolic_bonus_raw`;
    - `symbolic_model_bonus_raw`;
    - `predicted_effect_trust`;
    - `predicted_effect_candidate_wm_confidence`;
    - `predicted_effect_calibration`;
    - `predicted_effect_calibration_mean_error`;
    - `predicted_effect_calibration_count`.
  - Local contact safety, terminal memory, phase guidance, and current-state protections are not gated by this patch.
  - This is the first behavior patch using observation-learning, but it is conservative: it only reduces trust in predicted successor effects when the model has recent evidence of bad effect prediction.
- Tests:
  - `py_compile` passed for touched files.
  - Focused observation tests: `5 passed, 107 deselected`.
  - Full sandbox suite: `112 passed`.
  - New tests:
    - post-step observation outcome diagnostics compare predicted vs real effects and update action trace;
    - predicted-effect trust drops after repeated bad successor calibration.
- CUDA/GPU smoke:
  - Command:
    - `wa30`, `n_runs=1`, `max_steps=60`, `eps=0.0`, encoder-only checkpoint, trusted-plus-expanded trajectories.
  - `nvidia-smi` confirmed `/home/.../venv/bin/python` as a compute process using about `976 MiB`.
  - Output:
    - `claude_sandbox/perf_event_dumps_observation_effect_trust_smoke_gpu`;
    - checkpoint `claude_sandbox/checkpoints_observation_effect_trust_smoke_gpu/arc_wa30_run1.pt`.
  - Run completed without crash.
  - Short smoke result:
    - levels completed: `0` in 60 steps;
    - observation updates: `7`;
    - outcome diagnostics: `60`;
    - mean predicted changed-fraction error: `0.981856`;
    - mean predicted effect L1: `0.354690`.
  - Last chosen candidate diagnostics after trust gate:
    - raw candidate WM/effective confidence: `0.03461`;
    - calibration mean error: `0.98193`;
    - calibration floor: `0.05`;
    - `predicted_effect_trust = 0.00173`;
    - raw hazard penalty: `-0.3339995`;
    - effective hazard penalty: `-0.0005780`;
    - raw symbolic bonus: `+0.066136`;
    - effective symbolic bonus: `+0.000114`;
    - raw symbolic model bonus: `-0.009626`;
    - effective symbolic model bonus: `-0.0000167`.
- Important diagnosis:
  - This confirms the previous "all candidates harmful" post-veto result was at least partly polluted by a very unreliable next-frame predictor.
  - The world model currently predicts near full-frame changes for transitions whose real effect is around one percent of cells.
  - Therefore, topology/symbolic scoring over predicted frames cannot be trusted as a primary behavioral signal until effect prediction is calibrated.
  - The topology code may still be internally consistent; the current failure mode is now more specifically:
    - predicted successor representation is bad;
    - symbolic/topology analysis of that hallucinated successor becomes bad;
    - the planner then reasons coherently over a false successor.
- Current code state:
  - Observation learning is no longer purely passive:
    - it now gates predicted-frame-derived symbolic/hazard scoring according to empirical effect accuracy.
  - The gate is self-calibrating per run and candidate-local.
  - It does not invent actions or override hard local/terminal protections.
  - It gives score traces the raw and effective values so we can audit whether behavior changed for the right reason.
- Next useful work:
  1. Run a longer GPU regression with this gate:
     - `wa30`, `n_runs=3-5`, `max_steps>=500`;
     - inspect whether level-3 death still occurs and whether post-veto candidates remain all-negative after predicted-effect gating.
  2. Add a next-frame predictor diagnostic block:
     - per-run changed-fraction overprediction;
     - cell accuracy;
     - entropy confidence vs actual effect error;
     - by-action breakdown.
  3. Train/improve the next-frame/effect model:
     - the current next-frame predictor is too inaccurate for topology-on-prediction planning;
     - changed-mask/effect heads should be trained longer and measured before they influence candidate generation.
  4. If the longer run improves candidate scoring but still lacks productive candidates, only then start conservative observation-effect candidate generation.

Observation diagnostic cadence, video-safe paired quantization, and audit follow-up (2026-05-02):
- Implemented a performance/diagnostic cleanup after the observation trust-gate patch.
- Live candidate observation diagnostics are now cadence-bound and candidate-capped:
  - `PairwiseARCSearchAgent.observation_candidate_diag_every = 10`;
  - `PairwiseARCSearchAgent.observation_candidate_diag_max_candidates = 32`;
  - `_live_observation_candidate_diagnostics()` runs the neural observation heads only on cadence and only for the strongest proposal-score subset when there are many root candidates;
  - direct `_observation_candidate_diagnostics()` remains available for utilities/evaluator/probes probes and still runs the heads immediately.
- New diagnostic score-component fields:
  - `obs_candidate_diag_active`;
  - `obs_candidate_diag_skipped`;
  - `obs_candidate_diag_reason`;
  - `obs_candidate_diag_cadence`;
  - `obs_candidate_diag_max_candidates`.
  - Compact traces preserve the scalar cadence/cap fields; the full in-memory score components keep the reason string.
- Observation/video bug fixed:
  - before/after frames are now discretized with one shared scale through `to_discrete_frame_pair_tensors()`;
  - this matters for video: separate quantization could make a uniform dark frame and a uniform bright frame both collapse to bin zero, falsely producing a no-change transition;
  - `effect_vector_from_frames()`, observation training, and live candidate diagnostics now use paired conversion;
  - adapter-based conversion is still attempted first by concatenating the paired batch and splitting the dense result.
- Observation replay optimization:
  - `ObservationTransition` now stores `changed_fraction` at push time;
  - `sample_topology_delta_batch()` uses the cached value instead of rescanning full before/after frames;
  - `ObservationReplayBuffer.tail(count)` replaces direct agent access to the private `_buf` for newly ingested video rows.
- Tests after this patch:
  - `py_compile` passed for touched files;
  - focused observation tests: `5 passed, 107 deselected`;
  - full sandbox suite: `112 passed`.
- GPU/CUDA run notes:
  - The run must be launched outside the default sandbox to exercise CUDA; `nvidia-smi` should show `/home/.../venv/bin/python` as a compute process.
  - For ARC local/offline runs, the correct environment knob is `OPERATION_MODE=offline`, not `ARC_API_URL=offline`.
  - Use `PYTHONUNBUFFERED=1` for these probes; otherwise the run can be progressing while stdout looks silent.
- CUDA smoke after cadence patch:
  - command shape: `OPERATION_MODE=offline PYTHONUNBUFFERED=1 ... --games wa30 --max_steps 120 --n_runs 1 --eps 0.0 --no_replay`;
  - output: `claude_sandbox/perf_event_dumps_observation_diag_cadence_smoke_gpu4`;
  - checkpoint: `claude_sandbox/checkpoints_observation_diag_cadence_smoke_gpu4/arc_wa30_run1.pt`;
  - `nvidia-smi` showed compute PID using about `980 MiB`;
  - run completed without crash;
  - levels completed: `0`;
  - observation updates: `13`;
  - outcome diagnostics: `120`;
  - mean predicted changed-fraction error: `0.9203816731770833`;
  - mean predicted effect L1: `0.3539492958535751`;
  - compact score rows inspected: `600`;
  - observation candidate diagnostics active rows: `55`;
  - skipped rows: `545`;
  - `predicted_effect_trust` range/mean over compact rows: min `0.001038`, max `0.031256`, mean `0.004097`.
- CUDA smoke after paired-quantization patch:
  - command shape: `OPERATION_MODE=offline PYTHONUNBUFFERED=1 ... --games wa30 --max_steps 80 --n_runs 1 --eps 0.0 --no_replay`;
  - output: `claude_sandbox/perf_event_dumps_observation_pair_quant_smoke_gpu`;
  - checkpoint: `claude_sandbox/checkpoints_observation_pair_quant_smoke_gpu/arc_wa30_run1.pt`;
  - `nvidia-smi` showed compute PID using about `980 MiB`;
  - run completed without crash;
  - levels completed: `0`;
  - observation updates: `9`;
  - outcome diagnostics: `80`;
  - mean predicted changed-fraction error: `0.9775421142578125`;
  - mean predicted effect L1: `0.3631638174876571`;
  - compact score rows inspected: `400`;
  - observation candidate diagnostics active rows: `35`;
  - skipped rows: `365`;
  - `predicted_effect_trust` range/mean over compact rows: min `0.001251`, max `0.029571`, mean `0.004637`.
- Current diagnosis after the latest runs:
  - CUDA wiring is real and verified.
  - The observation heads train and report `ObsL`.
  - The cadence patch prevents candidate observation diagnostics from becoming an every-candidate/every-step neural tax.
  - The next-frame/world-model successor remains the dominant representational problem:
    - real changed fraction is usually around `0.008`;
    - predicted changed fraction remains around `0.91`;
    - predicted frame cell accuracy is roughly `0.08-0.09` in the last chosen examples;
    - therefore predicted-successor topology/symbolic reasoning remains untrustworthy and is correctly suppressed by the trust gate.
  - The trust gate is doing its job: predicted-frame-derived hazard/symbolic effects are nearly zeroed when recent outcome calibration is bad.
  - This does not solve `wa30` by itself because the reliable remaining controls are current-state/contact/phase/terminal logic plus weak ranker priors; the agent is not yet learning a good action-effect policy from observation.
- Audit findings / optimizations still open:
  1. `train_arc_codex.py` still catches broad `Exception` around `env.step()`.
     - Permanent fix remains: narrow expected environment exceptions and add a per-run exception threshold.
  2. `tr87_run0_traj_cls.pt` cache is stale:
     - cached length `97`, trajectory length `106`;
     - every trusted-load startup re-encodes those 106 transitions through Ouro;
     - regenerate that cache to remove needless startup work.
  3. Observation heads still use three separate `TransitionFrameEncoder` instances.
     - Good for isolation, but inefficient;
     - future optimization: shared transition encoder with three lightweight heads, or a shared frozen feature path plus separate heads.
  4. Main replay-buffer samplers still perform many whole-buffer scans.
     - Observation topology-delta sampling now caches change fraction, but `TransitionReplayBuffer` still has several scan-heavy pair/action samplers;
     - future optimization: maintain small index lists for real/expert/click/changed/auxiliary subsets.
  5. Compact action traces do not retain `obs_candidate_diag_reason`.
     - Not behavior-critical, but useful if cadence/cap behavior needs event-dump inspection without full in-memory traces.
  6. Natural-video learning still needs a real visual adapter.
     - Paired quantization fixes the immediate false-no-change bug, but grayscale bins are still only a crude generic bridge.
  7. `SelfModel` still does not own evaluator/self-diagnosis.
     - User intent remains: the evaluator and self-diagnosis should feel like the system making choices and noticing issues itself.
     - Future wiring should expose evaluator disagreement, calibration error, outcome prediction error, and recovery success/failure through the self-model/temporal context path.
- Recommended next empirical step:
  - run a longer unbuffered offline CUDA regression:
    - `wa30`, `n_runs=3`, `max_steps=500`, `eps=0.0`;
    - inspect whether it remains alive past the old level-3 death zone and whether candidate selection is now limited by successor hallucination rather than topology/hazard scoring.
  - Add a next-frame diagnostic block before tuning topology again:
    - predicted vs real changed fraction by action;
    - predicted frame cell accuracy by action;
    - next-frame entropy/confidence vs actual effect error;
    - chosen vs top rejected candidate effect error.
  - The next behavior patch should target action-effect learning / successor prediction quality, not stronger topology penalties.

Long wa30 offline CUDA regression after observation trust/cadence patch (2026-05-02):
- Run command shape:
  - `OPERATION_MODE=offline PYTHONUNBUFFERED=1 ... /home/moloch/ouro_project/venv/bin/python -m claude_sandbox.train_arc_codex`;
  - `--agent hunter_seeker --games wa30 --checkpoint checkpoints_running/sprint4_encoder_reverted.pt --backbone_mode encoder_only`;
  - `--load_trajs claude_sandbox/trusted_plus_expanded --pretrain_iters 1 --max_steps 500 --n_runs 3 --eps 0.0 --no_replay`;
  - outputs under `claude_sandbox/perf_event_dumps_observation_long_wa30_gpu` and `claude_sandbox/checkpoints_observation_long_wa30_gpu`.
- CUDA verification:
  - launched outside the default sandbox;
  - `nvidia-smi` showed `/home/.../venv/bin/python` as an active compute process;
  - memory use was roughly `976-1002 MiB`;
  - checkpoints were written for all three runs:
    - `arc_wa30_run1.pt`;
    - `arc_wa30_run2.pt`;
    - `arc_wa30_run3.pt`.
- Outcome:
  - all three runs completed levels 1 and 2;
  - all three died at run-relative step `283`;
  - run 1 death payload: `failure_type=protected_terminal_starvation`, `last_action=4`;
  - run 2 death payload: `failure_type=protected_terminal_starvation`, `last_action=2`;
  - run 3 death payload: `failure_type=planner`, `last_action=5`;
  - console score was `5.161386666666667` for each run.
- Observation-learning summary:
  - run 1:
    - `updates=45`;
    - `outcome_diag_count=256`;
    - `mean_predicted_changed_fraction_error=0.4152965545654297`;
    - `mean_predicted_effect_l1=0.25670243219065014`;
    - `mean_predicted_frame_cell_accuracy=0.572504997253418`;
    - fatal-window `predicted_effect_trust` mean about `0.0673`.
  - run 2:
    - `updates=89`;
    - `outcome_diag_count=256`;
    - `mean_predicted_changed_fraction_error=0.025646209716796875`;
    - `mean_predicted_effect_l1=0.1637239249976119`;
    - `mean_predicted_frame_cell_accuracy=0.9624671936035156`;
    - fatal-window `predicted_effect_trust` mean about `0.1318`.
  - run 3:
    - `updates=133`;
    - `outcome_diag_count=256`;
    - `mean_predicted_changed_fraction_error=0.023896217346191406`;
    - `mean_predicted_effect_l1=0.15882076090201735`;
    - `mean_predicted_frame_cell_accuracy=0.9644441604614258`;
    - fatal-window `predicted_effect_trust` mean about `0.1890`.
- Interpretation:
  - the successor/world-model calibration problem improves substantially online by runs 2 and 3;
  - successor hallucination is no longer the best explanation for the repeated level-3 death at step 283;
  - `hazard_reachable_penalty` and `hazard_reachable_penalty_raw` are zero in the fatal window of all three runs, so this is not the older "hazard scoring wired but too weak" failure;
  - the current blocker is topology/protection attribution around terminal contact and protected tracks.
- Fatal topology details:
  - runs 1 and 2:
    - topology source: `adjacent`;
    - adjacent colors: `[2, 4, 9]`;
    - protected colors: `[0, 2, 4, 9, 12, 14]`;
    - `raw_pool_count=3`;
    - `candidate_count=0`;
    - all adjacent terminal candidates were rejected as safe/protected;
    - `terminal_candidate_starvation=True`;
    - `terminal_all_adjacent_candidates_protected=True`;
    - `protected_terminal_starvation=True`.
  - run 3:
    - topology source: `adjacent_terminal_override`;
    - adjacent colors: `[14]`;
    - `raw_pool_count=1`;
    - `candidate_count=1`;
    - terminal adjacency override accepted a protected/avatar-like adjacent candidate;
    - the terminal was then classified as timeout-like planner failure because prior hazard was nearly zero;
    - `failure_type=planner`.
- Final chosen score diagnostics from `measurement_run_*.json`:
  - run 1 final selected action has `local_contact_source=avatar_overlap_protected_context`, protected-overlap penalty about `-0.0241`, `hazard_reachable_penalty=0`, `directional_topology_bonus=+0.0116`, `phase_action_bonus=+0.2896`, and `engram_progress_bonus=+0.1581`;
  - run 2 final selected action came through `risk_patcher_beam`; protected-overlap penalty about `-0.0216`, `hazard_reachable_penalty=0`, and the risk patcher saw all candidates as risky;
  - run 3 final selected action has protected-overlap penalty about `-0.0433`, `directional_topology_bonus=+0.0955`, `phase_action_bonus=+0.1748`, `engram_conflict_flag=True`, `engram_negative_support=0.1609`, and effective engram bias negative after optimism suppression;
  - these confirm the trace has the right scoring ingredients, but the protection/topology interpretation is still not behaviorally decisive.
- Diagnostic gap still open:
  - `measurement_summary.chosen_action_debug.last_score_components` is populated and useful;
  - the per-step `action_trace` rows in the long-run dumps still do not carry `chosen_candidate` / compact score components for the final step;
  - the action trace therefore remains weaker than the measurement fallback for post-run inspection.
- Current diagnosis:
  - do not make the next patch a generic stronger hazard penalty;
  - do not hardcode `wa30`, colors, or game-specific terminal causes;
  - the real bug is that "terminal happened while all adjacent terminal candidates were classified protected/safe" is being treated mostly as missing hazard evidence or planner failure;
  - domain-agnostically, that should be interpreted as evidence that the protection model was overconfident in that local context.
- Next behavior patch direction:
  1. Add a protected-terminal-starvation / protected-overlap terminal risk signal:
     - when a terminal occurs and all adjacent terminal candidates were filtered by protected/avatar/exit safety gates, record a bounded topology/protection risk rather than skipping the evidence;
     - key it by generic context features: action, local contact source, object role beliefs, adjacency/contact, protection reasons, terminal result, and state/topology cue vectors;
     - do not key by game id, ARC level, color constants, or any `wa30` special case.
  2. Feed that risk into scoring:
     - candidate score components need explicit fields for the new risk and support;
     - the risk should be bounded and conservative, similar in spirit to the existing protected-overlap penalty;
     - phase-action or positive progress engrams should not erase repeated protected-terminal evidence in the same context.
  3. Make terminal attribution consistent:
     - `protected_terminal_starvation` should remain distinct from true hazard contact and from generic planner timeout;
     - `adjacent_terminal_override` should not silently downgrade a protected terminal into a low-hazard planner failure when the diagnostic reason is "all adjacent candidates were filtered as safe."
  4. Fix the remaining action-trace instrumentation gap:
     - ensure compact selected score components are written into `action_trace` even when `search_trace` is absent or compacted away;
     - keep `measurement_summary.chosen_action_debug.last_score_components` as the final fallback.

Protected-terminal context patch and CUDA result (2026-05-02):
- Implemented the next topology/protection patch in `claude_sandbox/arc_agent_hunter_seeker_codex.py`.
- New behavior:
  - `adjacent_terminal_override` with reason `all_adjacent_candidates_filtered_as_safe` is now classified as `protected_terminal_starvation` when the accepted candidate still carries protected/avatar/exit safe evidence;
  - timeout-like protected adjacency is no longer demoted to generic `planner` before the protected-terminal taxonomy can see it;
  - true hazard evidence is still not applied to protected tracks in that case, so the patch does not poison avatar/exit-like tracks with hazard evidence.
- Added a domain-agnostic protected-terminal context memory:
  - stored in `_protected_terminal_context_memory`;
  - persisted in checkpoints as `protected_terminal_context_memory`;
  - keyed by generic local-contact score diagnostics, phase-state trust diagnostics, primitive direction geometry, and topology cue sketch;
  - does not key by game id, level id, color constants, object ids, or `wa30`-specific rules.
- New scoring/diagnostic terms:
  - `protected_terminal_context_penalty`;
  - `protected_terminal_context_risk`;
  - `protected_terminal_context_support`;
  - `protected_terminal_context_best_similarity`;
  - `protected_terminal_context_match_count`;
  - `protected_terminal_context_action_count`;
  - `protected_terminal_context_counter_count`;
  - `protected_terminal_context_terminal_count`;
  - `protected_terminal_context_active`;
  - `protected_terminal_context_reason`.
- Risk/trace integration:
  - protected-terminal context penalty contributes to `safety_penalty`;
  - protected-terminal penalty is included in risk-patcher terminal risk;
  - active protected-terminal context can count as direct terminal risk;
  - compact score components preserve all new protected-terminal terms;
  - per-step `action_trace` now always gets `chosen_score_components` when `_last_chosen_score_components` has signal, fixing the earlier final-step trace gap.
- Tests:
  - `py_compile` passed for touched files;
  - focused tests for protected terminal context and trace fallback: `6 passed, 110 deselected`;
  - broader focused terminal/risk/topology sets:
    - `30 passed, 86 deselected` in `test_codex_sandbox.py`;
    - `59 passed, 64 deselected` in `test_causal_correctness.py`;
  - full sandbox suite: `346 passed, 1 skipped`.
- Main CUDA regression:
  - command shape:
    - `OPERATION_MODE=offline PYTHONUNBUFFERED=1 ... -m claude_sandbox.train_arc_codex`;
    - `--games wa30 --max_steps 500 --n_runs 3 --eps 0.0 --no_replay`;
    - checkpoint input: `checkpoints_running/sprint4_encoder_reverted.pt`;
    - outputs:
      - `claude_sandbox/perf_event_dumps_protected_terminal_context_wa30_gpu`;
      - `claude_sandbox/checkpoints_protected_terminal_context_wa30_gpu`.
  - `nvidia-smi` confirmed CUDA process `/home/.../venv/bin/python` at about `976-982 MiB`.
  - all three runs still died at run-relative step `283`;
  - all three still completed levels 1 and 2 before dying.
- Main CUDA outcomes:
  - run 1:
    - failure: `mechanism`;
    - source: `adjacent`;
    - last action: `3`;
    - accepted adjacent candidate color `4`, track `71`;
    - `protected_terminal_starvation=False`;
    - final chosen score components now present in both measurement and action trace.
  - run 2:
    - failure: `mechanism`;
    - source: `adjacent`;
    - last action: `5`;
    - accepted adjacent candidates color `4` and color `2`;
    - `protected_terminal_starvation=False`.
  - run 3:
    - failure: `protected_terminal_starvation`;
    - source: `adjacent_terminal_override`;
    - last action: `1`;
    - timeout-like terminal still true, but protected-terminal classification now wins;
    - protected context memory recorded one entry after terminal.
- Observation-learning status in the main CUDA run:
  - run 1 mean predicted changed-fraction error: `0.380716`;
  - run 2 mean predicted changed-fraction error: `0.029071`;
  - run 3 mean predicted changed-fraction error: `0.023843`;
  - frame cell accuracy by runs 2/3 remains around `0.959-0.964`;
  - successor calibration is still improving online and is not the dominant step-283 explanation.
- Continuation diagnostic:
  - loaded `claude_sandbox/checkpoints_protected_terminal_context_wa30_gpu/arc_wa30_run3.pt`;
  - ran `wa30`, `n_runs=2`, `max_steps=500`, offline CUDA;
  - outputs:
    - `claude_sandbox/perf_event_dumps_protected_terminal_context_cont_wa30_gpu`;
    - `claude_sandbox/checkpoints_protected_terminal_context_cont_wa30_gpu`.
  - CUDA confirmed at about `976 MiB`.
  - continuation run 1:
    - completed levels 1 and 2;
    - died at run-relative step `283`;
    - failure: `protected_terminal_starvation`;
    - source: `adjacent`;
    - `candidate_count=0`;
    - final chosen action had `protected_terminal_context_best_similarity=0.689`, below active threshold;
    - protected context memory size after death: `2`.
  - continuation run 2:
    - died earlier at run-relative step `200`;
    - levels completed: `0`;
    - failure: `protected_terminal_starvation`;
    - source: `adjacent_terminal_override`;
    - final chosen action had:
      - `protected_terminal_context_support=0.214`;
      - `protected_terminal_context_best_similarity=0.932`;
      - `protected_terminal_context_penalty=0.0`;
      - `terminal_outcome_penalty=-0.0247`;
      - `phase_action_bonus=+0.2896`;
      - `directional_topology_bonus=+0.1450`;
      - `engram_effective_total_bias=+0.1363`;
      - total score about `+0.9983`.
- Diagnosis after the protected-terminal patch:
  - The old instrumentation gap is fixed: final action traces now carry selected score components.
  - The protected-terminal taxonomy bug is fixed for timeout-like adjacency override: it no longer disappears into generic planner failure.
  - The patch does not clear level 3.
  - Remaining issue is more general:
    - terminal/protected evidence is recorded, but it is still conservative and often inactive before the next death;
    - when weak terminal evidence is present, positive phase/topology/engram terms can still dominate by a large score margin;
    - the risk patcher only crosses limited score gaps unless terminal risk is already active/hard.
  - This must not become a run-3-specific fix:
    - during training, online failures may seed memory;
    - post-training/evaluation should rely on trained/persisted evidence or distilled evaluator/ranker/self-model behavior, not on dying in an earlier evaluation run.
- Next topology work:
  1. Generalize protected-terminal evidence into training/evaluation:
     - terminal protected-context examples should be learned from training sweeps/checkpoints or distilled into the evaluator/ranker/self-model path;
     - evaluation should not require a fresh run to die before the agent has the warning.
  2. Calibrate protected-terminal support:
     - support values like `0.214` with similarity around `0.93` are diagnostic but inactive today;
     - decide whether a small bounded cross-action protected-terminal penalty should activate at high similarity/support, while avoiding broad directional blacklists.
  3. Reconcile positive pressure vs terminal evidence:
     - phase-action, topology, and positive engram bonuses can overwhelm terminal/protected risk;
     - the scorer/risk patcher needs a domain-agnostic rule for when terminal evidence should cross a larger score gap.
  4. Keep the refactor deferred until topology is genuinely stable:
     - `arc_agent_hunter_seeker_codex.py` is now over 18k lines and should be split after the topology behavior is closed;
     - likely extraction seams: terminal memory/protected-terminal context, local contact safety, phase guidance/recovery, engram memory, observation learning, and dump/measurement plumbing.

Post-protected-context cleanup, phase-state suppression, and confirmed CUDA run (2026-05-03):
- Important run-procedure finding:
  - in this environment, launching the long ARC command through `/bin/bash -lc ...` can make `torch.cuda.is_available()` evaluate false inside the training process;
  - direct invocation through the approved venv Python prefix sees CUDA correctly;
  - use command shapes that start with `/home/moloch/ouro_project/venv/bin/python -m claude_sandbox.train_arc_codex ...` for GPU regressions;
  - `nvidia-smi` confirmed the direct run on PID `264985`, with the Python process holding about `982-1002 MiB`.
- Code state before this run:
  - protected-terminal context memory and taxonomy are still present:
    - protected terminal evidence is recorded under generic local-contact/topology/phase-state cue vectors;
    - it contributes bounded `protected_terminal_context_*` score components and risk-patcher terminal risk;
    - action traces carry compact chosen score components, so the previous final-step instrumentation gap is fixed.
  - protected-track risk conflict is active:
    - weak/conflicted avatar/exit/protected evidence no longer makes a terminal-adjacent object absolutely safe when unknownness and hazard/wall evidence disagree;
    - topology death diagnostics now expose belief hazard/wall/unknownness and `protected_track_risk_conflict`.
  - escalation optimism suppression is active:
    - local contact risk can suppress escalation/recovery positive pressure through `escalation_risk_suppression`;
    - all terms are emitted in `score_components`.
  - protected-terminal context relevance includes `avatar_overlap_target`, so this context is recordable/retrievable instead of invisible to protected-terminal memory.
  - phase-state pre-state mismatch suppression is active:
    - positive phase/recovery/reseed/exact-template pressure is partially suppressed when the template before-state does not match the live state;
    - emitted fields include `phase_state_bonus_suppression`, `phase_state_bonus_pressure`, `phase_state_bonus_gate`, `phase_state_bonus_active`, and `phase_state_bonus_reason`;
    - risk patcher counts that suppression as optimism risk.
- Latest confirmed GPU regression:
  - command:
    - `/home/moloch/ouro_project/venv/bin/python -m claude_sandbox.train_arc_codex --agent hunter_seeker --games wa30 --checkpoint checkpoints_running/sprint4_encoder_reverted.pt --backbone_mode encoder_only --load_trajs claude_sandbox/trusted_plus_expanded --pretrain_iters 1 --max_steps 500 --n_runs 3 --eps 0.0 --no_replay --running_checkpoint "" --checkpoint_dir claude_sandbox/checkpoints_avatar_overlap_context_wa30_gpu --save_trajs_dir claude_sandbox/solved_sequences_expanded --dump_events_dir claude_sandbox/perf_event_dumps_avatar_overlap_context_wa30_gpu`
  - outputs:
    - `claude_sandbox/perf_event_dumps_avatar_overlap_context_wa30_gpu`;
    - `claude_sandbox/checkpoints_avatar_overlap_context_wa30_gpu`.
  - all three runs completed levels 1 and 2, then died at run-relative step `283`;
  - failure counts: `mechanism=3`;
  - protected-terminal starvation stayed gone in this run.
- Aggregate dump summary:
  - levels: total `6`, per-run `[2, 2, 2]`;
  - phase templates: `exact_trusted=744`, `weak_template=56`, `no_template=49`;
  - phase-state alignment:
    - `available=797`;
    - `exact_matches=744`;
    - `mismatches=53`;
    - `effect_mismatches=53`;
    - `effect_underpowered=53`;
    - `component_failures=206`;
    - `component_missing=112`;
    - `component_static=90`;
    - `component_move_shortfall=4`.
  - phase exact override reasons:
    - `already_exact_phase_guidance=737`;
    - `pre_state_mismatch_veto=8`;
    - `trusted_exact_phase_guidance=7`.
  - final contact drift:
    - `mechanism_low_prior=0`;
    - `protected_starvation=0`;
    - `uncertain_penalty_active=0`;
    - `protected_overlap_active=2`.
  - post-veto recovery:
    - inspected `30` recovery-window steps across the three runs;
    - `positive_steps=0`;
    - `no_positive_steps=30`;
    - `risky_only_steps=30`;
    - selector reason was `no_supported_recovery_candidate=30`;
    - chosen classes were `harmful=28`, `risky=2`.
- First phase-state mismatches remain consistent across runs:
  - run 1:
    - first mismatch at step `251`;
    - selected action `3`, processed trusted action `4`, phase `65`;
    - expected changed cells `65`, actual changed cells `7`;
    - component failures include missing expected value `12`, missing value `0`, and a move-shortfall component.
  - run 2:
    - first mismatch at step `251`;
    - selected action `2`, processed trusted action `4`, phase `65`;
    - expected changed cells `65`, actual changed cells `33`;
    - component failures include missing expected value `12` and missing value `7`.
  - run 3:
    - first mismatch at step `252`;
    - selected action `3`, processed trusted action `2`, phase `66`;
    - expected changed cells `64`, actual changed cells `33`;
    - component failure is missing expected value `12`.
- Final chosen-score diagnostics:
  - run 1 final chosen:
    - action `1`;
    - total score about `0.2299`;
    - risk score about `0.1017`;
    - risk reason `no_high_risk_chosen`;
    - safety penalty about `-0.0216`;
    - engram bias about `+0.1862`;
    - `phase_state_bonus_suppression=-0.1068`;
    - `phase_state_pre_signature_match=false`.
  - run 2 final chosen:
    - action `4`;
    - total score about `0.2956`;
    - risk score about `0.1559`;
    - risk reason `no_lower_risk_candidate`;
    - safety penalty about `-0.0811`;
    - engram bias `+0.2`;
    - `phase_state_bonus_suppression=-0.0997`;
    - `phase_state_pre_signature_match=false`.
  - run 3 final chosen:
    - action `3`;
    - total score about `-0.1072`;
    - risk score about `0.0587`;
    - risk reason `all_candidates_risky`;
    - safety penalty `0`;
    - engram bias `+0.2`;
    - `phase_state_bonus_suppression=-0.0782`;
    - `phase_state_pre_signature_match=false`.
- Current topology diagnosis:
  - topology instrumentation is now doing the useful thing:
    - terminal deaths are attributed with populated adjacent/topology diagnostics;
    - protected-terminal starvation is not silently hidden in this latest run;
    - selected score components are present in action traces;
    - phase-state mismatch/effect-underpower/component failures are visible.
  - The remaining wa30 level-3 failure is not primarily "topology cannot see the hazard";
    - it is selecting among candidates whose own symbolic/topology summaries are all bad;
    - after the veto window, the agent has no candidate it predicts as positive;
    - risk patcher is choosing least-bad candidates, not discovering a productive recovery.
  - Phase-state suppression fixed a real optimism bug, but by itself cannot solve the failure because the candidate set/model still lacks a believable good action-effect path.
- Current observation-learning diagnosis:
  - the shadow observation path is present and training:
    - observation buffer;
    - `ChangedMaskHead`;
    - effect-summary head;
    - inverse-action model;
    - transition-effect engram memory;
    - post-step observation outcome diagnostics;
    - cadence-bound candidate observation diagnostics.
  - This is the right next direction because the repeated mismatch is action-effect/representation shaped:
    - expected trusted effects around steps `251-252` do not occur in live state;
    - value `12` repeatedly appears as a missing expected moved component;
    - the current policy has diagnostics for stale/mismatched replay, but not yet a strong learned way to propose the correct replacement effect.
  - Keep observation-derived behavior conservative:
    - video/unlabeled transitions should train changed/effect/topology/inverse diagnostics;
    - inferred actions should not train policy unless inverse confidence is high;
    - observation priors must remain behind terminal/hazard guards.
- Future work carried forward from `hunter_seeker_additional_components.md`:
  1. Continue observation learning beyond the current shadow heads:
     - robust `EventSegmenter` for video streams;
     - object/topology/event delta targets, not just changed masks and coarse effect summaries;
     - inverse-action confidence calibration before using inferred labels for policy;
     - candidate-effect oracle diagnostics from trusted trajectories.
  2. Add conservative observation-effect candidate generation only after diagnostics show the observation path can identify useful alternatives in the post-veto window.
  3. Internalize evaluator and self-diagnosis into the self model:
     - evaluator disagreement, outcome-prediction error, phase-state mismatch, memory conflict, and recovery failure should enter the self-state/temporal context;
     - the long-term shape should feel like the same system making choices and noticing its own mistakes, not an external logger overriding it.
  4. Keep Hunter Seeker refactor deferred:
     - do not start the 18k-line split until topology behavior and observation-learning behavior are stable enough that refactoring will not bury active bugs.
- Latest test state before this run:
  - after the phase-state suppression patch, the full local suite passed: `358 passed, 1 skipped`;
  - no code behavior was changed after that test run, only this project-state update.

Observation-effect recall bridge, post-veto recovery scoring, and CUDA result (2026-05-03):
- Implemented the first three requested items:
  1. Observation/action-effect patch:
     - `claude_sandbox/observation_learning_codex.py` now extends `OBS_EFFECT_KEYS` from 9 to 15 features.
     - New generic effect descriptors:
       - changed component count;
       - value appearance mass;
       - value disappearance mass;
       - active value-set Jaccard;
       - same-value centroid shift;
       - moved-value fraction.
     - Added `state_vector_from_frame()`:
       - quantized value histogram;
       - state entropy;
       - active-value fraction;
       - non-dominant component count;
       - non-dominant fraction.
     - This is domain/game agnostic: no ARC game id, color id, object id, action name, or `wa30` rule is used.
  2. Post-veto candidate-effect recall:
     - `TransitionEffectEngramMemory` now stores optional state vectors and serializes/deserializes records.
     - Recall now combines effect similarity with state similarity when both are present.
     - Recall reports separate support:
       - `obs_engram_same_action_positive_support`;
       - `obs_engram_same_action_negative_support`;
       - `obs_engram_cross_action_positive_support`;
       - `obs_engram_cross_action_negative_support`;
       - `obs_engram_progress_score`;
       - best effect/state similarities and best outcome.
     - Pairwise/Hunter checkpoints now persist observation heads, observation optimizer state, observation-loss counters, and transition-effect engram records.
     - Shape-mismatched old observation heads load best-effort and restart incompatible heads instead of crashing.
  3. Conservative recovery candidate scoring/generation:
     - Hunter scoring now converts observation-effect recall into bounded recovery pressure:
       - `obs_effect_recovery_score`;
       - `obs_effect_recovery_bonus`;
       - support/state/risk/action gates;
       - reason string.
     - Positive observation-effect recovery is gated by:
       - same-action support, or only weak cross-action support at very high similarity;
       - recovery/low-trust/pre-state-mismatch state context;
       - terminal/hazard/local-contact risk guards.
     - Bonus is bounded at `[-0.12, +0.18]`.
     - It is included in `score_components`, compact traces, post-veto candidate diagnostics, and risk-patcher diagnostics.
     - Live recovery selector can now choose observation-supported candidates and emits:
       - `observation_effect_recovery_supported`;
       - `pre_state_veto_observation_effect_recovery`.
- Tests run:
  - compile:
    - `/home/moloch/ouro_project/venv/bin/python -m py_compile claude_sandbox/observation_learning_codex.py claude_sandbox/arc_agent_pairwise_stockfish_codex.py claude_sandbox/arc_agent_hunter_seeker_codex.py claude_sandbox/test_codex_sandbox.py`
    - passed.
  - focused tests:
    - `/home/moloch/ouro_project/venv/bin/python -m pytest claude_sandbox/test_codex_sandbox.py -k "observation or live_recovery or post_veto or compact_score_components or risk_patcher" -q`
    - `19 passed, 112 deselected`.
  - broader local CPU suite:
    - `/home/moloch/ouro_project/venv/bin/python -m pytest claude_sandbox/test_codex_sandbox.py claude_sandbox/test_causal_correctness.py -q`
    - `256 passed`.
- GPU/CUDA run procedure correction:
  - Direct venv Python without any leading environment assignment sees CUDA:
    - `torch.cuda.is_available() == True`;
    - device: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`.
  - In this runner, adding even a harmless environment assignment before the Python command makes CUDA invisible:
    - examples tested: `MPLCONFIGDIR=/tmp ...`, `ARC_API_URL=offline ...`, `HF_HUB_OFFLINE=1 ...`;
    - all returned `torch.cuda.is_available() == False`.
  - The first `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ARC_API_URL=offline ...` training attempt was killed because it was CPU-bound.
  - Correct GPU run used no env prefix:
    - `/home/moloch/ouro_project/venv/bin/python -m claude_sandbox.train_arc_codex --agent hunter_seeker --games wa30 --checkpoint checkpoints_running/sprint4_encoder_reverted.pt --backbone_mode encoder_only --load_trajs claude_sandbox/trusted_plus_expanded --pretrain_iters 1 --max_steps 500 --n_runs 3 --eps 0.0 --no_replay --running_checkpoint "" --checkpoint_dir claude_sandbox/checkpoints_observation_effect_recall_wa30_gpu --save_trajs_dir claude_sandbox/solved_sequences_expanded --dump_events_dir claude_sandbox/perf_event_dumps_observation_effect_recall_wa30_gpu`
  - CUDA verified by `nvidia-smi`:
    - Python compute PID `272564`;
    - initial allocation about `340 MiB`;
    - process was launched by the direct venv Python command.
  - Because no env prefix was used, the ARC SDK fetched metadata from the API and then loaded cached `wa30-ee6fef47` from `environment_files/wa30/ee6fef47/wa30.py`.
- GPU run outputs:
  - dumps:
    - `claude_sandbox/perf_event_dumps_observation_effect_recall_wa30_gpu`;
  - checkpoints:
    - `claude_sandbox/checkpoints_observation_effect_recall_wa30_gpu`;
  - all three runs completed levels 1 and 2, then died at run-relative step `283`;
  - failure counts: `mechanism=3`;
  - score each run: `5.161386666666667`.
- Aggregate dump summary after the patch:
  - levels: total `6`, per-run `[2, 2, 2]`;
  - phase templates:
    - `exact_trusted=744`;
    - `weak_template=63`;
    - `no_template=42`.
  - phase-state:
    - `available=806`;
    - `exact_matches=744`;
    - `mismatches=62`;
    - `effect_mismatches=60`;
    - `effect_underpowered=56`;
    - `component_failures=251`;
    - `component_missing=155`;
    - `component_static=95`;
    - `component_move_shortfall=1`.
  - phase exact override reasons:
    - `already_exact_phase_guidance=743`;
    - `pre_state_mismatch_veto=11`;
    - `trusted_exact_phase_guidance=1`.
  - post-veto recovery:
    - `30` recovery-window steps inspected;
    - `positive_steps=7`;
    - `no_positive_steps=23`;
    - `risky_only_steps=23`;
    - selector reasons:
      - `no_supported_recovery_candidate=28`;
      - `observation_effect_recovery_supported=2`.
    - chosen classes:
      - `harmful=9`;
      - `mixed=17`;
      - `productive=4`.
- Important diagnosis from the GPU run:
  - Observation recall is wired and visible in traces.
  - It does influence behavior:
    - run 2 selected `live_recovery_beam` at step `250`;
    - observation-supported recovery selector fired in the post-veto window;
    - run 3 had `pre_state_veto_observation_effect_recovery` at step `278`.
  - It still does not solve level 3 because recall is currently positive-dominated:
    - many candidate recalls show high positive support from progress/demo records;
    - negative support is often `0.0`;
    - the chosen or best-observation candidates still have strongly negative symbolic/topology deltas.
  - Example final-step state:
    - run 3 final chosen action `5`;
    - total score about `0.108`;
    - `safety_penalty=-0.4654`;
    - `obs_effect_recovery_bonus=+0.0077`;
    - `obs_effect_recovery_score=+0.0426`;
    - `obs_engram_positive_support=0.6521`;
    - `obs_engram_negative_support=0.0`;
    - `engram_total_bias=+0.2`;
    - risk patcher reason `no_lower_risk_candidate`.
  - Example observation-supported recovery:
    - run 3 step `278`;
    - selection method `live_recovery_beam`;
    - reason `pre_state_veto_observation_effect_recovery`;
    - chosen action `3`;
    - `obs_effect_recovery_bonus=+0.1400`;
    - `obs_effect_recovery_score=+0.7779`;
    - still predicted by symbolic/topology as losing reachability/reward heavily:
      - `reachable_delta=-28`;
      - `reachable_reward_delta=-8.119`;
      - `reachable_hazard_delta=-2.448`.
- Current conclusion:
  - The first three patches are enacted and tested.
  - The behavior patch works mechanically, but the learned observation memory is too optimistic to be trusted as a strong policy source.
  - The repeated level-3 death now looks like a representation/action-effect learning problem, not a missing topology diagnostic:
    - topology sees the bad deltas;
    - phase-state sees stale/mismatched replay;
    - risk patcher sees risk but often has no lower-risk candidate within budget;
    - observation recall can identify similar transitions but cannot yet distinguish "generic progress-looking change" from "this candidate is actually useful in the current state."
- Next patch direction, still domain agnostic:
  1. Add risk-aware observation recall calibration before increasing observation influence:
     - downweight expert/demo `progress` records when their effect vector is broad and not tied to same-action/state-specific successful outcomes;
     - make `change` a neutral outcome, not positive support;
     - add terminal/failure counterevidence from real deaths into the same observation-effect memory with enough state context to match future fatal candidates.
  2. Add candidate-effect quality labels for observation memory:
     - success/progress should mean actual level progress, reward/reachability improvement, or post-action recovery success;
     - mere frame change or expert membership should not automatically produce policy-positive support.
  3. Keep observation-effect bonuses conservative until negative support appears in the post-veto window:
     - current positive recall can fire on candidates whose symbolic deltas are all bad;
     - the next diagnostics should track positive/negative support by source (`trusted`, `live`, `terminal`, `video`) and outcome quality.
  4. Then rerun:
     - `wa30`, `n_runs=3`, `max_steps=500`, direct venv Python with no env prefix;
     - inspect whether step-249..283 observation recall gains negative support and whether observation-supported recovery stops selecting candidates with large negative reachability/reward deltas.

Claude sandbox audit fixes and current wa30 basin status (2026-05-03):
- Source audit read:
  - `CLAUDE_SANDBOX_AUDIT_2026-05-03.md`.
  - Scope fixed in this pass: concrete correctness bugs, low-risk diagnostics gaps, stale artifacts, and small cache/perf hazards.
  - Large structural refactors remain deferred until topology/observation-learning behavior is stable.
- Implemented Claude audit fixes:
  1. `grid_encoder_codex.py`
     - `SinusoidalPositionalEncoding2D` now rejects invalid dimensions (`d_model <= 0` or not divisible by 4) instead of failing later in tensor math.
     - `encode_for_ouro()` now returns padding-aware attention masks.
     - Auto-padded patch tokens are masked when their patch contains only `pad_value`.
     - Binary cell masks are converted into patch masks with max-pooling semantics.
     - The backwards-compatible action-head re-export is documented as stable compatibility, not an unresolved migration window.
  2. `action_adapters_codex.py`
     - `ClickHead` docs now match the real fixed-ladder implementation.
     - `ActionHead.select_action()` no longer prints a lost warning when `available_actions` contains no valid in-range action; it records `invalid_available_actions=True` and safely samples unmasked.
     - `ArcActionAdapter` keeps a small multi-enum cache instead of thrashing between one cached enum class.
     - Unknown action decode fallback now consults adapter `safe_action_indices()` rather than hardcoding reset/action `0`.
  3. `self_model.py`
     - `TemporalContextAggregator` now bypasses `cortex_proj` entirely when `cortex_feature is None`, appending true zeros and preventing cortex-off ablation drift through learned projection bias.
     - Module docs now state that affect/streak/GRU hidden state are not in `state_dict`; checkpointing must use `state_snapshot()` / `load_state_snapshot()`.
  4. `train_arc_codex.py`
     - Partial checkpoint loading now checks dtype as well as tensor shape and reports `dtype_skipped`.
     - `env.step()` no longer swallows arbitrary exceptions:
       - expected recoverable environment exceptions are limited and counted;
       - after three recoverable exceptions the run aborts;
       - unexpected exceptions clear pending attribution state and re-raise.
     - Observation-outcome diagnostics now preserve all finite scalar/bool/short-string fields under `info["observation_outcome_diag"]`, while keeping `obs_*` numeric fields directly visible for compact logs.
  5. `arc_agent_pairwise_stockfish_codex.py`
     - Replay-buffer `temporal_features` snapshots are detached, CPU-copied, float-cast, and cloned before storage.
     - Malformed/mixed temporal-feature batches are skipped cleanly instead of throwing opaque `torch.stack` shape errors.
     - Pending cortex-signature prediction now records the step it was produced on and refuses to train against stale pending signatures after skipped steps.
     - `encode_and_think_batch(train_pooler=True)` restores the pooler training mode in `finally`.
     - Candidate-successor confidence is clamped to `[0, 1]`.
     - BFS/structural click candidates are clamped to grid bounds before emission.
     - Observation candidate diagnostics guard malformed predicted batches.
     - No-click adapters no longer silently fall back to ARC click index `6`.
     - The transient candidate cache is bounded at 256 entries and still cleared at beam-search boundaries.
  6. `arc_agent_hunter_seeker_codex.py`
     - No-click adapters now return `-1` for `_click_action_idx()` instead of silently assuming ARC click index `6`.
     - Phase-state frame signatures now include shape, byte length, CRC32, and a 64-bit BLAKE2b digest.
     - Topology-local engram similarity skips records whose topology-cue vector length differs from the current cue instead of silently padding/truncating.
     - Phase-state mismatch streaks, alignment trust, and approximate-resync repeat counts are now saved and loaded in Hunter checkpoints.
     - Measurement summary avoids repeated current-run event-log scans by computing the current-run entries once and passing them through helper calls.
     - Observation-effect recall risk gating was strengthened: `safety_penalty` contributes to the recall risk estimate, and positive recall is suppressed more aggressively under moderate safety/local/terminal risk.
     - Risk patcher budget now expands under high positive pressure plus high candidate risk, but remains a selector guard rather than a broad action blacklist.
- Cleanup enacted:
  - Moved legacy Hunter backup files into `claude_sandbox/_cleanup_quarantine/`:
    - `arc_agent_hunter_seeker_codex.py.bak_chosen_action_debug_20260429_152657`;
    - `arc_agent_hunter_seeker_codex.py.bak_soft_terminal_proto_threshold_20260429_151558`;
    - `arc_agent_hunter_seeker_codex.py.bak_cross_action_terminal_diag_20260429_153945`.
  - Quarantined stale `tr87_run0_traj_cls.pt` caches:
    - `trusted_trajs/tr87_run0_traj_cls.pt`;
    - `claude_sandbox/trusted_plus_expanded/tr87_run0_traj_cls.pt` renamed in quarantine to avoid collision.
  - The subsequent GPU load regenerated the 106-transition `tr87` CLS cache instead of loading the stale 97-transition cache.
- Tests after the audit fixes:
  - Compile:
    - `/home/moloch/ouro_project/venv/bin/python -m py_compile claude_sandbox/grid_encoder_codex.py claude_sandbox/action_adapters_codex.py claude_sandbox/self_model.py claude_sandbox/arc_agent_pairwise_stockfish_codex.py claude_sandbox/train_arc_codex.py claude_sandbox/arc_agent_hunter_seeker_codex.py claude_sandbox/test_causal_correctness.py claude_sandbox/test_cortex_monitor.py claude_sandbox/test_self_model_integration.py claude_sandbox/test_codex_sandbox.py claude_sandbox/branch_basin_audit.py`
    - passed.
  - Focused slices:
    - action/grid/adapter/risk-patcher tests: `6 passed, 122 deselected`;
    - cortex monitor: `14 passed`;
    - self-model integration: `22 passed`;
    - Hunter observation/engram/phase/risk compact trace slice: `9 passed, 122 deselected`;
    - temporal-feature and no-click/observation-diagnostic slices passed.
  - Full CPU suite:
    - `/home/moloch/ouro_project/venv/bin/python -m pytest claude_sandbox -q`
    - `368 passed, 1 skipped`.
- GPU verification and run:
  - Direct venv Python still sees CUDA:
    - `torch.cuda.is_available() == True`;
    - device: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`.
  - `nvidia-smi` verified the run as a compute process:
    - `/home/moloch/ouro_project/venv/bin/python`;
    - initial allocation about `340 MiB`, later about `980 MiB`.
  - Command used:
    - `/home/moloch/ouro_project/venv/bin/python -m claude_sandbox.train_arc_codex --agent hunter_seeker --games wa30 --checkpoint checkpoints_running/sprint4_encoder_reverted.pt --backbone_mode encoder_only --load_trajs claude_sandbox/trusted_plus_expanded --pretrain_iters 1 --max_steps 500 --n_runs 3 --eps 0.0 --no_replay --running_checkpoint "" --checkpoint_dir claude_sandbox/checkpoints_audit_fixes_wa30_gpu --save_trajs_dir claude_sandbox/solved_sequences_expanded --dump_events_dir claude_sandbox/perf_event_dumps_audit_fixes_wa30_gpu`
  - Result:
    - all three runs cleared levels 1 and 2;
    - all three died at run-relative step `283`;
    - failures: `mechanism=3`;
    - score per run: `5.161386666666667`;
    - dumps in `claude_sandbox/perf_event_dumps_audit_fixes_wa30_gpu`.
- Current topology/diagnostic state after the audit fixes:
  - Topology is not blind anymore:
    - terminal deaths carry adjacent/topology attribution;
    - final-contact drift diagnostics are populated;
    - selected score components are present;
    - phase-state mismatch/effect-underpower/component-failure fields are visible;
    - risk patcher reasons are present in compact traces.
  - The remaining wa30 level-3 failure is a delayed terminal-basin problem:
    - run 1, run 2, and run 3 all enter a state where no 6-step continuation survives after the actual step-277 transition;
    - depth-7 branch audit from the state before step 277 found `0/78,125` surviving branches in all three current runs;
    - therefore the current death is not fixable by only changing the final action at step 283 or by slightly reranking step 277.
  - Fresh branch-basin audit outputs:
    - `branch_basin_audit_focus_run_1.json`;
    - `branch_basin_audit_focus_run_2.json`;
    - `branch_basin_audit_focus_run_3.json`;
    - `branch_basin_audit_depth7_run_1.json`;
    - `branch_basin_audit_depth7_run_2.json`;
    - `branch_basin_audit_depth7_run_3.json`.
  - A broader depth-7 sweep over many earlier run-3 prefixes was intentionally stopped because it is too expensive with the current exhaustive tool; the focused audits already show the useful state: the issue must be detected earlier than the last visible safety decision.
- Current interpretation:
  - The Claude audit fixes landed and the local suite is clean.
  - The previous real behavior patch (hazard-aware safety, conservative engram/observation recall, risk-aware patcher budget) is wired but not enough for wa30 level 3.
  - The next behavior bottleneck is horizon/recovery representation:
    - topology can score local badness;
    - phase-state can see replay mismatch;
    - observation recall can surface effect similarity;
    - none of these yet produce an earlier "this prefix is entering a terminal basin within ~7 steps" signal strong enough to alter behavior before the basin becomes unavoidable.
- Future work carried forward:
  1. Add a cheaper, diagnostic-first finite-horizon basin sampler:
     - sample or beam-search action continuations from live prefixes;
     - report `basin_survival_fraction`, `basin_terminal_depth_min`, and `basin_best_level_gain`;
     - keep it diagnostic/very low weight at first.
  2. Use basin diagnostics to train/label observation-effect memory:
     - terminal-basin entry should create negative effect records before actual `GAME_OVER`;
     - positive support should require actual recovery/progress, not generic frame change.
  3. Keep phase/topology/observation recovery domain agnostic:
     - no game ids, color ids, or wa30-specific routes;
     - all new behavior must flow through generic frame/object/topology/effect features.
  4. Defer larger architecture cleanup:
     - `TransitionReplayBuffer` pair-sourcing strategy extraction;
     - observation buffer sampler indexing;
     - `PhaseStateTracker` class extraction;
     - Hunter Seeker file split/refactor.
  5. Do not start the Hunter refactor until topology behavior and observational learning behavior are stable enough that refactoring will not bury active bugs.

Model-basin sampler, observation terminal labels, and CUDA probes (2026-05-03):
- Implemented the deferred finite-horizon basin sampler as a conservative diagnostic-first patch.
- Code changes:
  1. `arc_agent_pairwise_stockfish_codex.py`
     - Added model-basin diagnostic settings:
       - `model_basin_diag_enabled=True`;
       - `model_basin_diag_every=10`;
       - `model_basin_diag_depth=2`;
       - `model_basin_diag_branch_width=1`;
       - `model_basin_diag_root_width=1`;
       - `model_basin_diag_max_expansions=4`;
       - `model_basin_diag_label_threshold=0.78`;
       - `model_basin_diag_label_max_confidence=0.25`.
     - Added `_annotate_model_basin_diagnostics()` and bounded root sampling over predicted successor states.
     - Added root-risk diagnostics on every root candidate, but recursive sampling now runs only on cadence or when a root already has terminal-like evidence.
     - Added score-component fields:
       - `model_basin_active`, `model_basin_sampled`, `model_basin_reason`;
       - `model_basin_horizon`, `model_basin_cadence`, `model_basin_branch_width`, `model_basin_root_width`;
       - `model_basin_expansions`, `model_basin_leaf_count`, `model_basin_terminal_count`, `model_basin_survivor_count`;
       - `model_basin_survival_fraction`, `model_basin_terminal_fraction`;
       - `model_basin_min_terminal_depth`, `model_basin_terminal_depth_mean`;
       - `model_basin_best_survivor_score`, `model_basin_mean_leaf_risk`, `model_basin_max_path_risk`;
       - `model_basin_root_risk`, `model_basin_risk`, `model_basin_penalty`;
       - `model_basin_label_confidence`, `model_basin_label_candidate`;
       - `model_basin_all_terminal`, `model_basin_any_survivor`, `model_basin_depth_limit_hit`.
     - Added `terminal_observed` observation-outcome labeling:
       - when the training harness sees `GAME_OVER`, the real before/after transition is added to `transition_effect_engram_memory` as outcome `terminal` with confidence `1.0`;
       - diagnostics: `obs_terminal_observed_label_added`, `obs_terminal_observed_label_confidence`;
       - summary fields: `terminal_observed_label_rate`, `mean_terminal_observed_label_confidence`.
     - Model-basin pseudo-labels remain low-confidence and bounded:
       - only when sampled terminal fraction is at least `0.80`;
       - label confidence capped at `0.25`;
       - no direct score change from `model_basin_penalty` yet.
  2. `arc_agent_hunter_seeker_codex.py`
     - Calls `_annotate_model_basin_diagnostics()` before phase/risk/live-recovery selectors so trace selectors can read the new fields.
     - Risk patcher reads `model_basin_risk` at low weight:
       - `+0.35 * basin_risk` in risk score only;
       - `risk_patcher_basin_risk` emitted in score components;
       - basin risk is not a hard direct-risk veto.
     - Compact score components now retain all model-basin scalar, boolean, and reason fields.
  3. `observation_learning_codex.py`
     - `terminal_basin` and `model_terminal_basin` are now negative recall outcomes.
  4. `train_arc_codex.py`
     - Adds `info["terminal_observed"]` before observation-outcome diagnostics when the returned environment state contains `GAME_OVER`.
  5. `test_codex_sandbox.py`
     - Added coverage that:
       - `terminal_basin` recall is negative and conflicts with positive support;
       - model-basin pseudo-labeling adds a bounded negative observation-effect record;
       - real terminal observation adds a strong negative observation-effect record.
- Important tuning correction:
  - The first implementation sampled too broadly and generated 32 low-confidence basin labels in a 180-step nonterminal smoke.
  - The live patch was tightened:
    - terminal-like classification now requires actual terminal-memory/protected-terminal evidence, not generic hazard or negative score;
    - label terminal fraction threshold raised to `0.80`;
    - recursive sampling budget reduced to cadence-10, root-1, branch-1, max-4 expansions.
- Verification:
  - Compile passed:
    - `/home/moloch/ouro_project/venv/bin/python -m py_compile claude_sandbox/train_arc_codex.py claude_sandbox/arc_agent_pairwise_stockfish_codex.py claude_sandbox/arc_agent_hunter_seeker_codex.py claude_sandbox/observation_learning_codex.py claude_sandbox/test_codex_sandbox.py`
  - Focused tests:
    - `/home/moloch/ouro_project/venv/bin/python -m pytest claude_sandbox/test_codex_sandbox.py -q`
    - `134 passed`.
  - Full CPU suite:
    - `/home/moloch/ouro_project/venv/bin/python -m pytest claude_sandbox -q`
    - `371 passed, 1 skipped`.
- CUDA verification:
  - Host-level runs were launched with escalated direct venv Python so `nvidia-smi` could see the CUDA process.
  - `nvidia-smi` verified `/home/moloch/ouro_project/venv/bin/python` as a compute process, initially around `340 MiB`, later around `980 MiB`.
  - A sandboxed run was stopped because it was CPU-bound/hidden from `nvidia-smi`; verified GPU runs should be launched outside the sandbox with the direct venv Python command.
- GPU smoke results:
  1. `80`-step conservative smoke:
     - command wrote to `claude_sandbox/perf_event_dumps_model_basin_smoke2_wa30_gpu`;
     - run completed normally at max steps;
     - `trace_len=80`;
     - chosen traces with model-basin fields: `80/80`;
     - recursive samples: `8/80`;
     - basin labels: `0`;
     - max chosen `model_basin_risk=0.4786`;
     - reasons: `ran=8`, `not_sampled=72`.
  2. `320`-step level-3 probe:
     - command wrote to `claude_sandbox/perf_event_dumps_model_basin_320_wa30_gpu`;
     - levels 1 and 2 cleared;
     - death still occurred at run-relative step `283`;
     - failure type: `mechanism`;
     - score: `5.161386666666667`.
- What the 320-step trace says:
  - The death is still the same delayed level-3 failure.
  - Risk patcher becomes active before death:
    - step `275`: `risk_patcher_beam`, reason `high_risk_chosen`;
    - steps `277` and `278`: `risk_patcher_beam`, reason `all_candidates_risky`.
  - The fatal chosen action at step `283` still has no meaningful pre-terminal risk:
    - `safety_penalty=0`;
    - `hazard_reachable_penalty=0`;
    - `terminal_outcome_penalty=0`;
    - `terminal_basin_penalty=0`;
    - `model_basin_risk=0`;
    - `risk_patcher_risk_score=0`;
    - `obs_engram_negative_support=0`;
    - `engram_negative_support=0`.
  - Therefore the current sampler did not prevent death because the chosen fatal transition looked safe to all current pre-terminal diagnostics.
  - The new actual-terminal observation label should improve the next run after the death is observed, because the real fatal before/after effect now enters observation-effect memory as negative support.
- Current interpretation after this patch:
  - The code now has the intended diagnostic surfaces and the observation-effect memory can learn from real terminal observations.
  - The model-basin sampler is conservative enough not to spam labels in ordinary nonterminal segments.
  - The remaining first-run failure is not solved by current pre-terminal evidence: it is a missing representation/signal problem for the fatal directional transition, not a simple weighting problem.
- Next likely work:
  1. Run at least a 2-run wa30 GPU probe after the actual-terminal observation label patch:
     - run 1 may still die at step 283;
     - run 2 should reveal whether `terminal_observed` observation-effect memory gives negative support before repeating the fatal action.
  2. Inspect run-2 final trace for:
     - `obs_engram_negative_support`;
     - `obs_engram_conflict_flag`;
     - `obs_effect_recovery_bonus`;
     - `terminal_observed_label_rate`;
     - whether risk patcher sees observation risk.
  3. If run 2 still repeats the fatal transition with zero observation risk:
     - strengthen same-action terminal observation recall, still bounded;
     - consider always sampling the selected root at very low depth near all-candidates-risk windows only;
     - do not add action/color/game blacklists.

Risk-aware observation recall probe and current wa30 basin status (2026-05-03):
- Implemented the follow-up observation recall patch after the `terminal_observed` GPU probe showed sparse negative terminal records being crowded out by many positive/progress records.
- Code changes now present in `claude_sandbox`:
  1. `observation_learning_codex.py`
     - `TransitionEffectEngramMemory.recall()` now collects top matches separately for all/positive/negative records and unions them before computing support.
     - Positive and negative supports are reported independently, so sparse terminal/mechanism records cannot disappear just because progress records dominate the global top-k.
     - Added diagnostics:
       - `obs_engram_positive_match_count`;
       - `obs_engram_negative_match_count`.
  2. `arc_agent_hunter_seeker_codex.py`
     - Observation-effect terminal recall is now wired into score components as a bounded, conservative term:
       - `obs_effect_terminal_risk`;
       - `obs_effect_terminal_penalty`;
       - `obs_effect_terminal_reason`.
     - Same-action negative terminal recall can apply a bounded penalty when confidence is high.
     - Cross-action negative recall remains diagnostic unless confidence is very high.
     - The terminal-effect penalty is capped at `0.25` and is included in `safety_penalty` only through explicit score components.
     - Risk patcher diagnostics now include observation terminal risk, but observation recall is not an action/color/game blacklist.
  3. `arc_agent_pairwise_stockfish_codex.py`
     - Model-basin/risk diagnostics forward the new observation terminal-risk fields.
     - Observation outcome diagnostics now retain positive/negative match counts and terminal-risk fields for later trace inspection.
  4. `test_codex_sandbox.py`
     - Added regression coverage that negative observation records are not crowded out by positive top-k records.
     - Added regression coverage that terminal observation penalties are bounded and high-confidence-only.
- Verification after the patch:
  - Py-compile passed for the edited observation, pairwise, Hunter Seeker, and test files.
  - Focused sandbox tests:
    - `/home/moloch/ouro_project/venv/bin/python -m pytest claude_sandbox/test_codex_sandbox.py -q`
    - `136 passed`.
  - Full CPU suite:
    - `/home/moloch/ouro_project/venv/bin/python -m pytest claude_sandbox -q`
    - `373 passed, 1 skipped`.
- GPU probe:
  - Command wrote to:
    - `claude_sandbox/perf_event_dumps_obs_terminal_risk_2run_wa30_gpu`;
    - `claude_sandbox/checkpoints_arc_obs_terminal_risk_2run_wa30_gpu`.
  - `nvidia-smi` verified the active compute process:
    - PID `301195`;
    - `/home/moloch/ouro_project/venv/bin/python`;
    - about `982 MiB` CUDA memory during the run.
  - Both runs cleared levels 1 and 2, then died on level 3 at run-relative step `283` with `FAILURE_TYPE=mechanism`.
- Important run-2 diagnostics:
  - The aggregation fix worked diagnostically:
    - by run-relative step `249`, `obs_engram_conflict_flag=True`;
    - `obs_engram_negative_support` is nonzero before the bad window, usually around `0.09` to `0.14`;
    - negative observation match counts are present (`2` to `3` matches in the inspected window).
  - The conservative behavioral gate did not fire:
    - `obs_engram_negative_best_similarity` stayed around `0.77` to `0.79`;
    - `obs_effect_terminal_risk=0`;
    - `obs_effect_terminal_penalty=0`;
    - `risk_patcher_observation_risk=0`.
  - This is the intended conservative behavior: weak cross-action negative observation recall is visible but does not become a broad action penalty.
  - Topology/local engram risk did fire on the obviously bad immediate action:
    - around run-relative `275-278`, action `3` carried strong generic engram risk and was scored down hard;
    - the agent selected lower-risk-looking alternatives instead.
  - Those lower-risk alternatives still led to terminal death, which means the remaining failure is sequence/basin-level, not a missing final-contact penalty.
- Current interpretation:
  - Step `283` is the terminal symptom, not the useful decision point.
  - The branch-basin audits and the repeated GPU traces agree that the agent is already effectively checkmated around run-relative `276-277`; previous depth-7 audits from the state before step `277` found `0/78,125` surviving branches.
  - The new observation recall patch confirms negative terminal evidence exists before death, but its best similarity is too weak to safely use as behavior.
  - The immediate topology system is not blind anymore:
    - local hazard/topology scoring is populated;
    - generic engram risk can suppress a repeatedly bad immediate action;
    - risk patcher diagnostics are present;
    - model-basin fields are present.
  - The unsolved issue is earlier basin-entry detection and recovery:
    - the planner needs to detect that a prefix is entering a terminal basin several steps before `GAME_OVER`;
    - or it needs a genuinely productive recovery candidate source when exact trusted guidance is vetoed and the available candidates are all harmful/mixed.
- Next work, still domain agnostic:
  1. Do not lower observation terminal thresholds just to fix wa30; the current observation signal is diagnostic but not high-confidence enough.
  2. Improve the finite-horizon basin path:
     - sample the selected root more reliably in all-candidates-risk / exact-veto windows;
     - consider a deeper, bounded diagnostic horizon for selected roots only;
     - keep any behavior very low weight until traces prove the model-basin signal is reliable.
  3. Improve pre-checkmate recovery representation:
     - use post-veto/all-candidates-harmful diagnostics as an early warning;
     - identify whether a productive candidate ever exists before step `276`;
     - if not, add a conservative learned/observed-effect recovery candidate source rather than only reranking the existing bad candidate set.
  4. Keep all behavior generic:
     - no wa30 route logic;
     - no game ids, color ids, or action blacklists;
     - all signals must come from frame/effect/topology/object/engram diagnostics.

Pressure-gated model-basin cleanup and current topology state (2026-05-03):
- Implemented the post-ladder finite-horizon basin cleanup in `claude_sandbox`.
- Current code state:
  1. `arc_agent_pairwise_stockfish_codex.py`
     - Model-basin diagnostics now track viable/collapsed continuations, not just terminal/survivor fractions:
       - `model_basin_viable_count`;
       - `model_basin_collapse_count`;
       - `model_basin_viable_fraction`;
       - `model_basin_collapse_fraction`;
       - `model_basin_best_viable_score`;
       - `model_basin_mean_leaf_trust`;
       - `model_basin_min_path_trust`.
     - Predicted-effect trust is part of model-basin scoring. Low-trust hallucinated survivors no longer count as viable just because their predicted score is positive.
     - Pressure windows use deeper bounded lookahead:
       - pressure depth: `6`;
       - pressure root width: `5`;
       - pressure max expansions: `8`.
     - The sampler now treats a nonterminal predicted leaf as viable only when:
       - predicted score is above threshold;
       - path risk is below threshold;
       - path trust is at least `0.18`.
     - Exact-veto / pre-state-mismatch recovery now triggers pressure-horizon sampling immediately, even if the shallow root score still looks superficially okay. This removed the step-249/250 instrumentation lag.
     - Important correction: raw cadence-discovered basin collapse is diagnostic-only unless the agent is already in recovery/pressure context. The first attempt made collapse behavior-active from step 0/10 and derailed trusted phase replay.
  2. `arc_agent_hunter_seeker_codex.py`
     - Risk patcher now reads basin collapse as behavior-relevant only when the collapse is in a pressure/recovery context:
       - `model_basin_reason in {"pressure_window", "pressure_window_after_collapse"}`;
       - or phase/exact-veto/live-recovery context is active.
     - Diagnostic raw collapse remains visible through the `model_basin_*` fields, but `risk_patcher_basin_collapse` is no longer set for plain cadence `model_basin_reason="ran"` cases.
     - Exact trusted phase guidance is not vetoed by raw low-trust basin collapse outside pressure/recovery. This preserves the solved-demonstration path early in the run.
  3. `test_codex_sandbox.py`
     - Added regression tests for:
       - nonviable survivor collapse;
       - trust-required viability;
       - pressure-window deep sampling;
       - exact-veto recovery triggering pressure sampling off cadence;
       - cadence collapse escalating only when recovery is active;
       - cadence collapse remaining diagnostic-only without recovery;
       - risk patcher treating pressure-context basin collapse as risk;
       - diagnostic-only basin collapse staying low weight;
       - exact phase guidance not overriding pressure-context basin collapse.
- CPU verification:
  - Py-compile passed for the edited pairwise, Hunter Seeker, and test files.
  - Focused sandbox suite after final patch:
    - `/home/moloch/ouro_project/venv/bin/python -m pytest claude_sandbox/test_codex_sandbox.py -q`
    - `147 passed`.
  - Full sandbox suite after final patch:
    - `/home/moloch/ouro_project/venv/bin/python -m pytest claude_sandbox -q`
    - `384 passed, 1 skipped`.
- CUDA probes and conclusions:
  1. `claude_sandbox/perf_event_dumps_collapse_risk_2run_wa30_gpu`
     - GPU verified via direct venv Python.
     - Both runs still died at run-relative step `283`.
     - From about step `250`, pressure sampling showed:
       - `model_basin_viable_fraction=0.0`;
       - `model_basin_collapse_fraction=1.0`;
       - low `model_basin_min_path_trust`;
       - `risk_patcher_basin_collapse=1.0`.
     - Selector usually reported `all_candidates_risky` or `no_lower_risk_candidate`.
  2. Over-aggressive cadence-collapse version:
     - Dump: `claude_sandbox/perf_event_dumps_cadence_collapse_2run_wa30_gpu`.
     - Run was intentionally stopped after run 1 because it regressed badly.
     - Run 1 died at step `200`, before the old level-3 failure.
     - Cause: raw model-basin collapse was behavior-active from early cadence steps (`0`/`10`) before recovery context was present. That derailed trusted phase replay.
     - Permanent fix applied afterward: basin collapse is behavior-active only in pressure/recovery context.
  3. Corrected pressure-gated version:
     - Dump: `claude_sandbox/perf_event_dumps_pressure_gated_2run_wa30_gpu`.
     - GPU verified with PID `314440`, direct venv Python, about `980 MiB` CUDA memory.
     - Run 1:
       - completed 2 levels;
       - died at step `283`;
       - failure type `mechanism`.
     - Run 2:
       - completed 2 levels;
       - died at step `283`;
       - failure type `protected_terminal_starvation`.
     - This restored the non-regressed baseline behavior while keeping model-basin collapse diagnostics/risk in the late recovery window.
  4. Exact-veto pressure trigger:
     - Dump: `claude_sandbox/perf_event_dumps_pressure_trigger_1run_wa30_gpu`.
     - GPU verified with PID `315854`, direct venv Python, about `982 MiB` CUDA memory.
     - Result:
       - completed 2 levels;
       - died at step `283`;
       - failure type `mechanism`.
     - Step `249` now samples pressure horizon immediately:
       - `model_basin_diag_reason=pressure_window`;
       - `model_basin_diag_sampled_count=5`;
       - `model_basin_viable_fraction=0.0`;
       - `model_basin_collapse_fraction=1.0`.
     - The earlier sampling changed some late actions but did not escape the terminal basin.
- Current topology/behavior diagnosis:
  - Topology is no longer blind in this failure.
  - The late window now has populated diagnostics:
    - local contact/protected-overlap scoring;
    - hazard-aware safety terms;
    - observation-effect recall;
    - engram recall;
    - model-basin viability/collapse/trust;
    - risk patcher all-candidates-risk metadata.
  - From step `249` onward, the model-basin sampler often reports every top root as collapsed under trust-gated horizon-6 lookahead.
  - The selector is usually not missing an obvious safer candidate:
    - `risk_patcher_all_candidates_risky=True`;
    - `risk_patcher_reason` is usually `all_candidates_risky` or `no_lower_risk_candidate`;
    - sampled candidates all carry `model_basin_viable_fraction=0.0`.
  - The exact-veto one-step lag is fixed diagnostically, but not behaviorally sufficient.
  - The repeated step-`283` death is still the terminal symptom. The useful decision point is earlier, before the available root candidates have all collapsed.
- Current assessment:
  - Further penalty tuning is unlikely to solve this by itself.
  - The next real bottleneck is a productive recovery candidate source / better action-effect representation, not another stronger reranker over the same bad candidates.
  - The next patch should stay domain agnostic and should likely target observation/action-effect learning:
    - use learned/observed effects to propose recovery candidates when exact-veto + model-basin collapse indicate the current candidate set is bad;
    - keep model-basin labels conservative and diagnostic-first;
    - do not use game ids, colors, route scripts, broad action bans, or ARC-specific rules.

Exact-guidance observation-risk veto and non-selected pre-state pressure fix (2026-05-03):
- Implemented two small generic fixes found by inspecting the latest GPU traces.
- Fix 1: exact trusted phase guidance now treats high-confidence observation-terminal recall as hard direct risk.
  - File: `claude_sandbox/arc_agent_hunter_seeker_codex.py`.
  - `_risk_patcher_has_direct_risk()` now includes:
    - `obs_effect_terminal_risk`;
    - `obs_effect_terminal_penalty`;
    - reasons `same_action_negative_recall` and `high_conf_cross_action_negative_recall`.
  - This prevents exact replay from overriding bounded, high-confidence observation terminal evidence.
  - Weak cross-action observation recall remains diagnostic-only unless it already passed the high-confidence terminal gate.
- Fix 2: model-basin pressure sampling now scans all root candidates for recovery/pre-state activation, not only the currently selected root.
  - File: `claude_sandbox/arc_agent_pairwise_stockfish_codex.py`.
  - The previous code could still miss the live step-249 trigger:
    - the selected beam root had no pre-state downgrade yet;
    - a non-selected exact trusted root had `phase_state_pre_alignment_downgraded=True`;
    - `_annotate_model_basin_diagnostics()` ran before phase-exact guidance wrote `pre_state_mismatch_veto`;
    - result: step 249 stayed `model_basin_diag_reason=cadence`, sampled `0` roots, and pressure sampling began at step 250.
  - New trace-wide pressure fields:
    - `model_basin_recovery_window`;
    - `model_basin_pressure_evidence`;
    - `model_basin_trace_recovery_context`;
    - `model_basin_trace_recovery_activation_pressure`.
  - These fields are emitted into score components/compact traces.
  - This is not game-specific: the trigger is based only on generic phase-state/pre-state/recovery diagnostics.
- Tests added:
  - exact guidance vetoes observation terminal risk;
  - non-selected pre-state-veto roots trigger model-basin pressure sampling.
- CPU verification:
  - Py-compile passed for pairwise, Hunter Seeker, and tests.
  - Focused tests:
    - `/home/moloch/ouro_project/venv/bin/python -m pytest claude_sandbox/test_codex_sandbox.py -k "model_basin or phase_exact_guidance or observation_effect_terminal" -q`
    - `14 passed, 135 deselected`.
  - Full sandbox suite after final diagnostics-field patch:
    - `/home/moloch/ouro_project/venv/bin/python -m pytest claude_sandbox -q`
    - `386 passed, 1 skipped`.
- GPU probes:
  1. Observation-terminal exact-veto probe:
     - dump: `claude_sandbox/perf_event_dumps_obs_terminal_exact_veto_2run_wa30_gpu`;
     - checkpoint dir: `claude_sandbox/checkpoints_obs_terminal_exact_veto_2run_wa30_gpu`;
     - CUDA verified by `nvidia-smi`:
       - PID `317772`;
       - direct `/home/moloch/ouro_project/venv/bin/python`;
       - about `982 MiB` CUDA memory.
     - Result:
       - run 1: cleared levels 1 and 2, died at step `283`, failure `mechanism`;
       - run 2: cleared levels 1 and 2, died at step `283`, failure `mechanism`.
     - The observation-terminal exact-veto patch fixed the direct-risk control-flow gap, but did not solve wa30.
  2. Non-selected pre-state pressure verification:
     - dump: `claude_sandbox/perf_event_dumps_nonselected_veto_pressure_1run_wa30_gpu`;
     - checkpoint dir: `claude_sandbox/checkpoints_nonselected_veto_pressure_1run_wa30_gpu`;
     - CUDA verified by `nvidia-smi`:
       - PID `319756`;
       - direct `/home/moloch/ouro_project/venv/bin/python`;
       - about `982 MiB` CUDA memory.
     - Result:
       - cleared levels 1 and 2;
       - died at step `283`;
       - failure `mechanism`.
     - Step `249` is now correctly pressure-sampled in the live trace:
       - `model_basin_diag_reason=pressure_window`;
       - `model_basin_diag_sampled_count=5`;
       - chosen `model_basin_reason=pressure_window`;
       - chosen `model_basin_viable_fraction=0.0`;
       - chosen `model_basin_collapse_fraction=1.0`;
       - chosen `model_basin_risk=0.65`;
       - all top shown roots at step 249 also had `model_basin_viable_fraction=0.0` and `model_basin_collapse_fraction=1.0`.
- Current diagnosis after these fixes:
  - The step-249 model-basin instrumentation lag is now actually fixed in the live trace.
  - The remaining death is not caused by exact guidance overriding the new observation terminal-risk term.
  - At the first exact-veto/pre-state-mismatch window, the bounded horizon-6 model-basin sampler already sees every sampled root as collapsed.
  - Risk patcher still usually reports `all_candidates_risky` or `no_lower_risk_candidate`.
  - This strengthens the prior conclusion: the system now sees the bad basin, but the existing root candidate set does not contain an obvious productive recovery path.
- Next work:
  1. Do not increase penalties again as the primary move.
  2. Add a genuinely productive, domain-general recovery source:
     - learned/observed effect proposals;
     - action-effect sequence memory;
     - or a stronger observation transition model that can predict which action sequence repairs the current state.
  3. Keep this diagnostic-first and bounded:
     - no wa30 route logic;
     - no game ids;
     - no color/action blacklists;
     - no broad cross-action terminal bans.

Topology compute cleanup and vestige audit (2026-05-03):
- Motivation:
  - The wa30 topology/model-basin probes became too expensive.
  - A 120-step verification run showed the process holding CUDA but spending most time in Python-side environment/search/topology/instrumentation work, not GPU math.
  - Topology had started as "diagnostic-first", but it was being charged repeatedly inside root scoring, deeper beam expansion, and model-basin speculative rollout.
- Implemented compute cleanup:
  1. Model-basin pressure latch/cache:
     - File: `claude_sandbox/arc_agent_pairwise_stockfish_codex.py`.
     - Added short pressure-collapse latch:
       - `model_basin_diag_pressure_latch_ttl = 4`;
       - `_model_basin_pressure_latch`.
     - Reuses collapsed pressure-window basin diagnostics only when:
       - pressure window is active;
       - available action set is identical;
       - sampled root action/click keys are identical;
       - predicted-root frame hashes and rounded root scores match;
       - all sampled roots were collapsed/non-viable with low trust.
     - New diagnostics:
       - `model_basin_cache_hit`;
       - `model_basin_reused_latched`;
       - `model_basin_latch_ttl_remaining`;
       - `model_basin_latch_root_count`;
       - `model_basin_sample_budget_used`;
       - `model_basin_diag_cache_hit`;
       - `model_basin_diag_reused_latched`;
       - `model_basin_diag_sample_budget_used`;
       - `model_basin_diag_latch_ttl_remaining`.
     - Latch clears on game reset and after training updates.
  2. Cheap topology mode for runtime scoring:
     - File: `claude_sandbox/arc_agent_hunter_seeker_codex.py`.
     - `_compute_free_space_topology(..., full=False)` now returns only the fields runtime policy actually uses:
       - `reachable_mask`;
       - `reachable_obj_ids`;
       - `frontier_obj_ids`;
       - avatar/start metadata;
       - empty placeholders for region graph fields.
     - Runtime calls in candidate generation, symbolic summaries, reachability scoring, and directional-death attribution now pass `full=False`.
     - Full topology still exists for direct Sprint 5 tests/future diagnostics.
     - Full topology is no longer refreshed every real step unless the self-model is active and the snapshot cadence fires.
  3. Removed duplicate flood-fill work:
     - Full topology now fills `bfs_distance` during the main reachable flood-fill instead of running a second BFS.
     - The old per-object `_mask_adjacent_to_reachable()` helper was removed.
     - Runtime now computes one `reachable_contact` mask per topology call and tests each object against it.
  4. Per-action planning caches:
     - Added `_topology_cache` and `_symbolic_summary_cache`, cleared at the start of each Hunter beam action and on reset.
     - Added trace counters:
       - `topology_cache_hits`;
       - `topology_cache_misses`;
       - `symbolic_summary_cache_hits`;
       - `symbolic_summary_cache_misses`.
  5. Root-only Hunter/topology scoring for speculative search:
     - Pairwise `_expand_search_node()` now sets `_hs_runtime_root_expansion` from `enable_observation_diagnostics`.
     - Hunter root expansion keeps full Hunter object/topology/safety scoring.
     - Non-root speculative beam nodes and model-basin rollout nodes skip Hunter topology/object scoring and use the learned ranker/world model proposal surface.
     - Non-root click proposals are capped by `deep_search_click_candidates = 2`.
     - This is domain/game agnostic: no game id, route, color, or action blacklist.
  6. Batched symbolic planner head:
     - The symbolic planner head is now run once per candidate batch instead of once per candidate.
     - This removes many tiny GPU launches/CPU syncs inside `score_candidates()`.
  7. Trajectory CLS cache writeback:
     - File: `claude_sandbox/arc_agent_pairwise_stockfish_codex.py`.
     - If a `*_traj_cls.pt` cache is missing, live encoding now writes it after fallback encoding.
     - `claude_sandbox/trusted_plus_expanded/tr87_run0_traj_cls.pt` was created during the next run.
     - Future runs should not repeatedly print/perform `Encoding 106 transitions through Ouro` for `tr87_run0`.
- Current topology code state:
  - Runtime policy topology is cheap and uses only reachable/frontier information.
  - Full region graph support remains in `_compute_free_space_topology(full=True)` because `claude_sandbox/test_topology_sprint5.py` asserts that public/default contract.
  - The region graph/gateway fields are currently not consumed by policy:
    - `region_map`;
    - `n_regions`;
    - `avatar_region_id`;
    - `region_sizes`;
    - `object_region`;
    - `region_objects`;
    - `gateway_obj_ids`;
    - `region_adjacency`.
  - `bfs_distance` is currently consumed only by the self-model track summary.
  - Therefore the region graph/gateway block is effectively diagnostic/test/future-work code, not active runtime decision logic.
  - If this remains unused after the self-model/evaluator wiring work, it should be extracted or deleted during the later Hunter-Seeker refactor.
- Vestigial/wrong-direction findings:
  1. Recursive Hunter topology inside model-basin rollout was the wrong direction:
     - it recursively paid for object parsing, topology summaries, engram local cues, and symbolic scoring;
     - model-basin should diagnose learned-successor basin risk, not rerun the whole root policy stack at every speculative node.
     - Fixed by root-only Hunter scoring for speculative search.
  2. Region graph/gateway construction is not currently used by policy:
     - useful as a tested diagnostic surface;
     - expensive and misleading if assumed to be the active reason for current behavior.
  3. `_mask_adjacent_to_reachable()` was redundant after the reachable-contact-mask optimization and has been removed.
  4. Per-candidate symbolic-head inference was an avoidable launch/sync tax and is now batched.
  5. Re-encoding missing trajectory CLS caches was pure startup waste and is now self-healing.
- Verification:
  - Py-compile passed for modified files.
  - Focused tests:
    - `/home/moloch/ouro_project/venv/bin/python -m pytest claude_sandbox/test_codex_sandbox.py -k "topology or model_basin or symbolic" -q`
    - `19 passed, 132 deselected`.
  - Full sandbox suite:
    - `/home/moloch/ouro_project/venv/bin/python -m pytest claude_sandbox -q`
    - `388 passed, 1 skipped`.
  - GPU verification:
    - 120-step direct venv smoke after root-only budget patch:
      - dump: `claude_sandbox/perf_event_dumps_topology_fast_budget_smoke_wa30_gpu`;
      - CUDA verified by `nvidia-smi`, PID `328861`, direct venv Python, about `340 MiB` CUDA memory during startup;
      - completed 120 steps and wrote event/measurement dumps;
      - first run also wrote the missing `tr87_run0_traj_cls.pt` cache.
    - Final 25-step direct venv CUDA check after all cleanup:
      - dump: `claude_sandbox/perf_event_dumps_topology_fast_cuda_check_wa30_gpu`;
      - CUDA verified by `nvidia-smi`, PID `330908`, direct venv Python, about `340 MiB` CUDA memory;
      - completed 25 steps and wrote event/measurement dumps.
  - Final 25-step trace counters:
    - `topology_cache_hits` sum: `35`;
    - `topology_cache_misses` sum: `150`;
    - `symbolic_summary_cache_hits` sum: `0`;
    - `symbolic_summary_cache_misses` sum: `125`;
    - `model_basin_diag_sampled_count` sum: `3`;
    - `model_basin_diag_sample_budget_used` sum: `3`.
  - 120-step trace counters from the completed smoke:
    - `model_basin_diag_reason`: `cadence` on 108 steps, `ran` on 12 steps;
    - `model_basin_diag_sampled_count` sum: `12`;
    - no pressure-window latch hits in the first 120 steps because the pressure window had not been reached.
- Remaining speed audit:
  1. The main remaining runtime cost is now likely beam/search expansion and environment interaction, not the topology region graph.
  2. `beam=6x2` still generates many candidate-successor predictions and Ouro encodes; batching across the whole frontier instead of per-node expansion is the next large optimization candidate.
  3. Root symbolic summaries still miss cache hits because each predicted frame is usually unique; this is expected and means the summary cache is mostly protecting transpositions/repeats, not normal root scoring.
  4. Object-table track lookup is still O(objects * tracks) in several places; a per-scene object-id-to-track map would be a clean later optimization.
  5. Event dumps are large (`run_1.json` was about 9.9 MiB for the earlier 120-step smoke); compact trace modes or sampled diagnostics may matter for long GPU probes.
  6. The timed `/usr/bin/time` smoke is not considered a valid GPU timing because `nvidia-smi` did not catch the process; direct venv runs are the reliable GPU protocol.
