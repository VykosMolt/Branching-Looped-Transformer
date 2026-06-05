# Post-v8.1 progress addendum

**Date:** 2026-05-18
**Status:** Tracking document, not a synthesis revision. Captures what has happened between v8.1 locking and the present moment. When the next synthesis revision is warranted (v8.2 or v9), this addendum becomes one of its inputs.
**Parent:** `phase1-routing-policy-locked.md`

## 0. Why this exists separately from v8.1

v8.1 locked the BG Phase 1 architecture (two-head with code-specialist backup, conservative domain routing default, experimental vote mode for validation). The §3a routing policy was locked; §8 next-actions pointed at deployment integration.

What happened next was *not* what §8 anticipated. Instead of clean Phase 1 deployment integration, the project explored several adjacent questions in sequence:

1. **BG controller implementation** — built the read-only deployable module.
2. **Wrapper candidate exposure** — built non-breaking candidate-export interface to local-agent.
3. **Overnight steering/routing suite** — first transformer-native test of partial routing, compute allocation, text-prefix branching, soft hidden-state steering.
4. **Wrapper-matched BG experiment** — ran the wrapper-matched comparison (still in progress as of this doc).
5. **Architectural correction** — recognized that wrapper integration is a side tool, not the architectural target; refocused on transformer-native BG.
6. **Stage 1 trajectory prediction sweep** — established BG signal at the partial-trajectory level.
7. **Stage 2 steering sensitivity probe** — designed (this prompt is queued; not yet run).

This addendum captures findings and design decisions from steps 1-6. Step 7 is queued and pending execution.

## 1. BG controller implementation (settled)

Built `src/evaluator/bg_controller.py` as a thin read-only Python module per v8.1 §3 / §12 spec.

**Verdicts at completion:**
- `BG_CONTROLLER_ARTIFACT_VERDICT = READY`
- `BG_CONTROLLER_IMPLEMENTATION_VERDICT = READY`
- `BG_CONTROLLER_UNIT_TEST_VERDICT = PASS`
- `BG_CONTROLLER_REPLAY_VERDICT = PASS`

**Key finding:** Conservative-mode replay against the cached policy simulator produced max pairwise difference of 0.000000 — the controller exactly reproduces the simulator's behavior on the validated eval bundle. This is the strongest possible correctness signal for the routing logic.

Experimental_vote mode showed max diff 0.500 vs simulator vote, which is expected because the controller uses calibration std from artifacts (or fallback 1.0) while the simulator used per-eval-domain std. The divergence is calibration approximation, not a bug. Vote mode remains experimental pending deployment-traffic-derived calibration.

**Architecture decisions confirmed by implementation:**
- The three locked heads load correctly from existing registry artifacts (`bg_head_registry_2026-05-17.pt` for hh_general and code_specialist_backup; `mixed_domain_tiny_heads_2026-05-17.pt` for objective_mixed_primary).
- Strict-clean code correctly routes to objective_mixed in conservative mode (v8.1's §3 correction holds in deployment).
- Four modes supported: conservative, experimental_vote, code_backup, diagnostic_all.

This artifact is production-ready for any downstream system that wants BG selection on already-captured candidate features. Files: `src/evaluator/bg_controller.py`, `utilities/tests/manual/test_bg_controller_unit.py`, `utilities/tests/manual/replay_bg_controller_on_cached_eval.py`.

## 2. Live transformer feature capture (settled)

Built `src/evaluator/bg_transformer_features.py` to extract pooled features from raw text via Ouro-RLTT forward pass. Returns `[layers=3, loops=4, hidden=2048]` shape per candidate, matching the format the BG heads expect.

Smoke tests passed. The module is now the connecting tissue between raw model output and BG controller scoring. Without it, BG would only work on pre-computed feature artifacts; with it, BG works on live generated candidates.

## 3. Wrapper candidate exposure (settled, but reframed)

Built `src/local_agent/candidate_export.py` and `src/local_agent/candidate_capture.py` as opt-in candidate trace infrastructure for the local-agent wrapper.

**Verdicts at completion:**
- `WRAPPER_CANDIDATE_PATH_INVENTORY_VERDICT = READY`
- `WRAPPER_CANDIDATE_EXPORT_UNIT_VERDICT = PASS` (139 wrapper tests still passing)
- `WRAPPER_CANDIDATE_EXPORT_SMOKE_VERDICT = SKIPPED` (gated by env var, not executed by default)
- `WRAPPER_CANDIDATE_EXPOSURE_VERDICT = READY`

**Reframing:** The wrapper integration was initially treated as a high-priority deployment target. It is now correctly understood as **a candidate-generator tool, not the architectural target**. The wrapper's role is to produce realistic code candidates that BG can be evaluated against — analogous to how MBPP/HumanEval are tools for evaluating code generation. The wrapper is not the destination.

The candidate export infrastructure remains useful as a tool for future evaluation work. It does not represent architectural progress in the way v8.1 anticipated.

## 4. Overnight steering/routing suite (settled)

First broad-scope transformer-native BG experiment. Tested partial trajectory routing, compute allocation, wrapper-matched selection, soft hidden-state steering, text-prefix branch selection.

**Verdicts:**
- `BG_PARTIAL_ROUTING_VERDICT = NEUTRAL` (+0.043 lift, below the +0.05 threshold; directionally positive but sub-threshold)
- `BG_COMPUTE_ALLOCATION_VERDICT = INSUFFICIENT` (interrupted by wall cap after 3 tasks)
- `BG_WRAPPER_MATCHED_VERDICT = SKIPPED` (wrapper multi-candidate interface not yet exposed at time of suite)
- `BG_SOFT_STEERING_VERDICT = STABLE_NO_EFFECT` (tiny α 0.005-0.02, mechanically stable, no clean directional improvement)
- `BG_LATENT_BRANCH_SELECTION_VERDICT = HELPS` (text-prefix branching, 8 tasks, cleared threshold but small n)
- `OVERALL_BG_STEERING_VERDICT = NEUTRAL`

**Key findings:**

*Generator reachability is heterogeneous.* Code oracle success: 0.000. Devil oracle success: 0.000. Reasoning/science/GSM8K oracle success: 1.000. Direct Ouro code generation does not produce viable candidates at the difficulty level tested; BG cannot select winning code branches because no code branches win.

*Partial routing has directional signal but small effect.* +0.043 lift on both top1 and top2 (consistent across two independent metrics suggests real small effect rather than noise). NEUTRAL by pre-registered threshold; deployable conclusion is "BG is weakly predictive at partial-trajectory level but not strong enough to drive default policy."

*Text-prefix branch selection at HELPS is the most promising positive finding.* Short prefix scoring + continuation under matched compute. Different mechanically from partial-routing (shorter prefixes, explicit branching structure). Underpowered at n=8 but cleared threshold.

*Soft steering at tiny α is mechanically stable.* No degeneration, no destabilization. Also no directional improvement — positive direction performed similar to random and negative controls. Tiny-α intervention is safe but doesn't move trajectories at that scale.

**Files:** `artifacts/reports/probes/bg_steering_suite_2026-05-18/*`.

## 5. Architectural correction (load-bearing)

Mid-stream during step 4-5, an important framing correction occurred:

> The wrapper is a tool, not the destination. The BG architecture lives in:
> Ouro-RLTT loop states → BG taps at 24/36/47 → trajectory routing / compute allocation → eventually activation-level steering or training regularizer.
> Not in: local-agent wrapper final answer reranker.

This corrected the prior framing that treated wrapper integration as the deployment milestone. The actual target has always been transformer-native BG operation. The wrapper was over-weighted as a candidate source because it produces real code candidates; this was a category error about what "real" means in the project's architectural goals.

**Operational consequences:**
- Wrapper-matched experiment (in progress) will be filed as candidate-generator diagnostic, not architectural milestone.
- Future BG experiments target transformer trajectories directly, not wrapper-produced candidates.
- The wrapper exposure infrastructure remains useful as a tool but does not anchor the project's deployment strategy.

This correction matters for interpreting subsequent results. Stage 1 (next section) was designed under the corrected framing; its strong positive result is therefore an architectural milestone in a way the wrapper experiment would not be.

## 6. Stage 1 BG trajectory prediction sweep (load-bearing positive result)

First targeted transformer-native experiment under the corrected framing. Tested whether BG signal at partial trajectories predicts which trajectories will complete to correct final answers.

**Verdicts:**
- `BG_TRAJECTORY_PREDICTION_VERDICT = STRONG`
- `GENERATOR_REACHABILITY_LIMITED = false`
- `RECOMMENDED_NEXT = run_targeted_BG_steering_sensitivity_probe_at_best_cell`

**Headline cell:**
- domain: reasoning
- prefix length: 256
- config: MIX_CODE_REASONING / 36_mean / AntisymLinear
- top1 lift: +0.1625 over random
- pairwise accuracy: 0.8537
- oracle success: 0.900

**Operating envelope (the breadth result):**
- 368 strong cells total across (domain × prefix_length × config) space
- 300 cells with top1 lift ≥ +0.10
- 227 cells with pairwise accuracy ≥ 0.65
- All three domains contribute strong cells (reasoning 152, science 126, GSM8K 90)
- Multiple configs contribute (36_L4: 91, 36_mean: 72, 24_L4: 69, plus layer 47 variants)
- Both architectures contribute (NoNorm: 189, AntisymLinear: 179)

**Non-monotonic prefix-length trend:**
- Reasoning top1 lift peaks at 64 tokens (early commitment detection)
- Science top1 lift peaks at 32 tokens (very early)
- GSM8K top1 lift peaks at 256 tokens (late completion needed)

This is a domain-dependent pattern: abstract/categorical reasoning benefits from early-trajectory BG scoring; computational/arithmetic tasks need near-completion scoring.

**Architectural implications:**

*BG operates at the trajectory level, not just the finished-candidate level.* The CLT paper established BG works on finished pairwise selection (95.2%). This experiment establishes BG also works on partial-trajectory pairwise selection (85.4% pairwise on the best cell). This is the first transformer-native validation of the broader BG architecture thesis.

*Trajectory-level BG operations may need different head configurations than finished-candidate operations.* The locked production head (objective_mixed = MIX_CODE_REASONING / 36_L4 / AntisymLinearNoNorm) is NOT the same as the trajectory-prediction peak (MIX_CODE_REASONING / 36_mean / AntisymLinear). If Phase 2 ever deploys BG at the trajectory level, 36_mean and/or AntisymLinear may be the relevant choices, not the v8.1-locked 36_L4 / NoNorm. The locked v8.1 architecture remains correct for finished-candidate selection; trajectory-level operation is a separate configuration question.

*Pairwise accuracy 0.85 with top1 success 0.39 means BG is good at ranking but not perfect at single-best selection.* At oracle 0.900, random_top1 ≈ 0.225, BG_top1 ≈ 0.388. BG extracts most of the available pairwise signal but leaves substantial absolute-selection accuracy unextracted. For Stage 2 design: intervention pushes toward "BG-preferred direction" which is a higher-ranked branch, not necessarily the single correct branch.

**Files:** `artifacts/reports/probes/bg_trajectory_prediction_2026-05-18/*`, `docs/evaluator/trajectory-prediction-sweep.md`.

## 7. Stage 2 steering sensitivity probe (queued, not yet run)

Designed but not yet executed. The prompt targets four cells (reasoning @ 64, reasoning @ 256, science @ 32, GSM8K @ 256) using NoNorm directions for intervention with AntisymLinear used only as diagnostic readout. Three orthogonal metrics: causal sensitivity, output stability, final correctness. Four conditions per (target, α): zero baseline, positive direction, negative direction, random control. α sweep at {0.0, 0.005, 0.01, 0.02}.

Pre-registered verdicts route to one of five outcomes:
- PROMISING_HANDLE_FOUND (causal + stable + final lift): expand sweep
- CAUSAL_BUT_NO_TASK_LIFT (causal + stable, no final lift): Phase 2 design priority
- READ_ONLY_BG (no causal effect, stable): lock v8.1, plan Phase 2 training
- DESTABILIZING (any α breaks generation): abandon inference-time steering
- INSUFFICIENT: fix experimental design

The most likely outcome on prior is CAUSAL_BUT_NO_TASK_LIFT or READ_ONLY_BG, both of which point toward Phase 2 training as the next major work block.

## 8. Current architecture state vs v8.1

What v8.1 §0 said:
> The BG Phase 1 architecture uses two production heads with one specialist backup, deployed via a domain-routed controller in conservative mode, with an experimental vote mode available for validation.

This remains correct and is unchanged. The Phase 1 architecture is locked.

What v8.1 §3 left open:
> The decision is genuinely open and is the next experimental question.

This was resolved by the controller-policy simulator (which fired before v8.1 was written but informed it) and operationalized by the controller implementation. Conservative mode and experimental_vote mode are both available. Vote mode validation is now blocked on deployment-scale traffic, which requires integration with a real downstream system.

What v8.1 §7 said about Phase 2:
> The Phase 2 L_eval regularizer must preserve head complementarity, not just individual head performance.

This remains correct and is strengthened by Stage 1's finding. If Phase 2 training improves individual head accuracy but reduces head complementarity, the deployable policy degrades. The v8.1 L_eval_complementarity constraint should be kept.

**New addition to Phase 2 considerations from Stage 1:**

If Phase 2 wants BG to operate at the trajectory level (not just finished-candidate selection), the head configurations may need to be retrained or the L_eval objective needs to include trajectory-level pairwise discrimination as a training signal. The current heads were trained on finished-candidate data; trajectory-level prediction works but possibly suboptimally.

## 9. Open questions, in order of priority

1. **Stage 2 outcome:** does BG have a causal handle on trajectories, or is it read-only? (Queued, pending execution.)

2. **Phase 2 design specifics:** depending on Stage 2, what does the L_eval objective look like? If READ_ONLY_BG: train heads to be steerable. If CAUSAL_BUT_NO_TASK_LIFT: train backbone to amplify intervention propagation.

3. **Vote mode validation at deployment scale:** requires real query traffic. Depends on which downstream system gets integrated with the controller.

4. **Margin calibration from deployment data:** currently approximate eval-domain std. Becomes feasible once deployment generates query traffic.

5. **Text-prefix branch selection expansion:** the HELPS result on 8 tasks deserves validation at proper power. Cheaper than Stage 2 to run.

6. **Code generation reachability:** direct Ouro code at oracle 0.000 is the largest underexplored domain. Either improve generator (out of scope for BG work) or accept that BG operates only on non-code domains for now.

## 10. What this addendum does NOT change

- v8.1's locked Phase 1 head set (HH_GENERAL + OBJECTIVE_MIXED_PRIMARY + CODE_SPECIALIST_BACKUP) is unchanged.
- v8.1's conservative routing rules are unchanged.
- v8.1's contrast-type architectural principle is unchanged.
- The CLT paper's headline result (95.2% on HH) is unchanged.
- The Hunter-Seeker ARC agent and Ouro depth expansion remain separate tracks.

## 11. Next synthesis revision trigger

This addendum should be folded into a v8.2 or v9 synthesis when:

- Stage 2 result lands AND
- That result either confirms PROMISING_HANDLE_FOUND (warrants expanded steering work) or settles READ_ONLY_BG (locks v8.1 as final Phase 1 architecture and frees Phase 2 design)

Until then, v8.1 remains the canonical reference and this addendum is the progress log.

## 12. File index

**v8.1 spec:** `docs/evaluator/phase1-routing-policy-locked.md`

**This addendum:** `docs/evaluator/phase1-progress-addendum.md`

**BG controller:** `src/evaluator/bg_controller.py`, `src/evaluator/bg_transformer_features.py`

**Wrapper candidate exposure:** `src/local_agent/candidate_export.py`, `src/local_agent/candidate_capture.py`

**Recent experiment reports:**
- Steering suite: `artifacts/reports/probes/bg_steering_suite_2026-05-18/*`
- Trajectory prediction: `artifacts/reports/probes/bg_trajectory_prediction_2026-05-18/*`
- Stage 2 (pending): `artifacts/reports/probes/bg_stage2_steering_2026-05-18/*` (will be populated when Stage 2 runs)

**Docs:**
- `docs/evaluator/controller-usage.md`
- `docs/evaluator/transformer-integration.md`
- `docs/evaluator/steering-and-routing-suite.md`
- `docs/evaluator/trajectory-prediction-sweep.md`
- `docs/evaluator/local-agent-candidate-export.md`
