# Post-v10 synthesis v8.1 — BG Phase 1 controller policy locked

**Date:** 2026-05-18
**Status:** Supersedes v8 by adding the BG controller-policy simulator result and locking §3. Otherwise identical to v8.
**Scope:** Closes the open routing question from v8 §3. Updates §1 head set role, §2 architectural principle, §3 routing policy, §7 Phase 2 implications, §8 next actions. Adds new §3a on experimental vote mode. Updates the §9 thesis paragraph.

## 0. The locked thesis (updated)

The BG Phase 1 architecture uses **two production heads with one specialist backup**, deployed via a domain-routed controller in conservative mode, with an experimental vote mode available for validation.

> **HH preference geometry and objective branch geometry are empirically distinct projection directions.** A general HH-trained head reads preference/coherence structure. A mixed objective head reads cross-candidate structure that generalizes across code, reasoning, science, and math objective tasks. The two projections carry **complementary signal** — a magical oracle router would gain over 8 pp pairwise over any deployable single-head policy by picking the right head per case. A simple margin-vote between the two extracts most of this complementarity (oracle gap of 0.009 vs 0.080+ for domain routing) but requires margin calibration that's not yet trustworthy at current eval sizes. Conservative deployment uses domain routing; experimental validation tests the vote scheme.

The contrast-type principle from v8 §2 holds unchanged. v8.1 adds the empirical complementarity finding as an explicit architectural constant.

## 1. Locked Phase 1 head set (updated)

| Role | Head | Training source | Layer config | Architecture | Conservative deployment |
|---|---|---|---|---|---|
| HH general | `hh_general` | HH-RLHF 200 pairs, Thinking backbone | 47_concat_L1_L4 | AntisymLinearNoNorm | HH preference, chemistry-like semantic MCQ if domain evidence favors HH, unknown/preference-shaped tasks |
| Objective mixed primary | `objective_mixed_primary` | CODE + REASONING_NATURAL + REASONING_TRACE (MIX_CODE_REASONING) | 36_L4 | AntisymLinearNoNorm | Default for code, reasoning, science, math objective tasks **including strict-clean code** |
| Code specialist backup | `code_specialist_backup` | Code-only training, 14 task split, 138 pairs | 36_L4 | AntisymLinear | Strict-clean code ablation reference; retained for tie-break / low-margin / disagreement fallback. **Not the primary strict-clean route.** |

**Correction from v8:** the strict-clean code role of `code_specialist_backup` was overstated. The simulator showed objective_mixed handles strict-clean as well as or better than code specialist at current n. The specialist is retained for borderline cases and ablation, but the primary strict-clean route is `objective_mixed_primary`.

### Performance summary on validated eval domains (from simulator)

| Eval domain | hh_general | objective_mixed | code_specialist | Conservative policy | GENERAL_AND_OBJECTIVE_VOTE |
|---|---:|---:|---:|---:|---:|
| HH heldout20 | 0.900 | 0.700 | 0.450 | 0.900 | 0.950 |
| CODE_STRICT_CLEAN_ALL16 (pairwise) | 0.600 | 0.867 | 0.833 | 0.867* | 0.900 |
| Cross-domain objective average | 0.577 | 0.822 | 0.728 | 0.817 | 0.893 |
| Overall pairwise | — | — | — | ~0.85 | 0.891 |
| Oracle gap on objective | LARGE (>0.08) | MODERATE | LARGE | LARGE (>0.08) | 0.009 |

*Strict-clean cell under conservative policy is 0.867 because objective_mixed routes there; code_specialist would deliver 0.833 if routed.

The conservative policy underperforms the vote scheme by ~4 pp average pairwise. The vote scheme is at oracle ceiling. But the vote scheme's strict-clean 0.900 at n=16 has CI ±10 pp, and its margin calibration uses eval-domain std rather than a separate validation distribution, so the result is not robust enough for default deployment.

## 2. Architectural principle (locked, unchanged from v8)

The contrast-type rule from v8 §2 holds. v8.1 adds one corollary:

**Head complementarity is empirically large.** The simulator's HEAD_COMPLEMENTARITY_VERDICT = HIGH_COMPLEMENTARITY result combined with ORACLE_GAP_VERDICT = LARGE means HH_GENERAL and OBJECTIVE_MIXED_PRIMARY make different errors on different cases. Cross-candidate objective discrimination and preference-coherence discrimination capture genuinely orthogonal signal. Future routing or calibration work that better extracts this complementarity is potentially valuable.

## 3. Locked BG Phase 1 routing policy (conservative)

The simulator's BEST_POLICY_VERDICT was OBJECTIVE_MIXED_DEFAULT_WINS. The RECOMMENDED_BG_POLICY was HH_GENERAL_PLUS_OBJECTIVE_MIXED_PLUS_CODE_BACKUP.

### Routing rules (conservative mode)

```
def route(query):
    if query.domain == "HH" or query.domain == "preference" or query.domain == "unknown":
        return hh_general(query)
    elif query.domain in ["code", "code_strict_clean", "reasoning", "science", "math", "objective"]:
        return objective_mixed_primary(query)
    else:
        return hh_general(query)
```

That's the entire conservative policy. No margin checks, no deferral, no contrast detection, no voting. Just domain → head.

**Science subdomain caveat.** Conservative mode routes all science queries to objective_mixed regardless of subdomain. The science probe showed heterogeneous subdomain behavior (biology→code-favoring, chemistry→HH-favoring, medicine close, general science→code-favoring), but routing by subdomain requires either deployment-time subdomain detection or per-subdomain confidence scoring, neither of which exists yet. Treat subdomain preferences as observed routing biases, not as a permanent ontology. A future calibrated router (e.g., the experimental vote mode after deployment validation) may extract more of the heterogeneity automatically.

### Why this and not the alternatives

**Why not DOMAIN_ROUTED_SIMPLE with strict-clean → code_specialist?** Because objective_mixed handles strict-clean as well as or better than code_specialist on current data (0.867 vs 0.833 pairwise). Routing strict-clean to code_specialist would *lose* 3 pp on average without compensating gains.

**Why not contrast routing?** CONTRAST_DETECTOR_VERDICT = DEPLOYABILITY_WEAK. Feature-cosine contrast detector achieved F1 0.800 but false-positive rate 1.000 — it can't distinguish locally-similar candidates from unrelated ones. The policy would route everything to code_specialist regardless of whether candidates are actually near-miss.

**Why not deferral?** DEFER_POLICY_VERDICT = DEFER_NOT_USEFUL. Margin-based and disagreement-based deferral failed to clear the +0.05 selective accuracy improvement threshold at 70% coverage. Either heads are well-calibrated enough that low-confidence predictions are still useful, or fallback strategies don't outperform the abandoned predictions.

**Why not OBJECTIVE_MIXED_ONLY?** Because it loses HH heldout by 0.200 (0.700 vs 0.900 for hh_general). HH preference is a distinct projection that objective mixing cannot replicate.

**Why not vote scheme by default?** Because it's at oracle ceiling (0.009 gap) but strict-clean is small-n borderline and margin calibration uses approximate std rather than validation distribution. Worth validating, not worth defaulting to before validation.

### Code specialist backup role

The code_specialist_backup is retained in the registry, deployed for:
- Ablation comparisons in future analysis
- Tie-break cases when objective_mixed and hh_general disagree and both have low margin (future vote mode)
- Diagnostic fallback if objective_mixed performance degrades in deployment

It is *not* part of the conservative routing path. It exists for failure modes and research, not production decisions.

## 3a. Experimental vote mode (added in v8.1)

The simulator surfaced GENERAL_AND_OBJECTIVE_VOTE_margin as the highest-performing deployable policy by a substantial margin: overall pairwise 0.891 vs conservative 0.817, oracle gap 0.009 vs 0.080+.

The controller should expose this as a non-default mode:

```
def route(query, mode="conservative"):
    if mode == "conservative":
        # see §3 above
        if query.domain == "HH" or "preference" or "unknown":
            return hh_general(query)
        elif query.domain in ["objective", ...]:
            return objective_mixed_primary(query)
    
    elif mode == "experimental_vote":
        score_hh = hh_general(query)
        score_obj = objective_mixed_primary(query)
        
        if sign(score_hh) == sign(score_obj):
            return score with larger |score| (agreement, pick stronger margin)
        else:
            # disagreement: pick higher normalized margin
            z_hh = score_hh / std_hh_validation
            z_obj = score_obj / std_obj_validation
            return whichever head has higher abs(z)
```

The vote mode is the **validation target**, not the default. Two reasons to expose it:

1. **Measurement.** Deployment-scale data lets us test whether the 0.891 result holds at higher n than the simulator's eval sets. If it replicates at 200+ queries, the vote scheme becomes the default in a future v8.2.
2. **Calibration.** The margin std used by the vote scheme is currently approximate (eval-domain std rather than separate validation distribution). Deployment provides natural validation data for proper calibration.

The experimental_vote mode is **not** recommended for production decisions where errors are costly. Reserve for validation queries, A/B testing, and research evaluation.

## 4. Backbone, taps, and architectural constants (unchanged from v8)

[Identical to v8 §4. Backbone: Ouro-RLTT. Tap interface: heterogeneous (24/36 single-state, 47 fused). Head family: AntisymLinear / AntisymLinearNoNorm. Pooling: masked mean. Numerical: bf16 forward, fp32 captured/training.]

## 5. Validated evaluation domains (extended from v8)

The simulator added an integrated eval bundle covering 13 domains with full feature coverage and 160 scored heads. The domain list is unchanged from v8 §5 (HH_200, CLEAN_GSM8K_EXPANDED, CODE_RUNNABLE_DIAGNOSTIC, CODE_STRICT_CLEAN_ALL16, REASONING_NATURAL_DISTRACTOR, REASONING_TRACE, SCIENCE × 5 subdomains).

What's new in v8.1: the eval bundle is now a reusable asset (`bg_policy_sim_eval_bundle_2026-05-17.{json,pt,md}`) that future policy iterations can replay against. Any new policy candidate can be evaluated on the same data the conservative and vote policies were evaluated on, ensuring apples-to-apples comparison.

## 6. Deferred or settled items (updated)

### Newly settled in v8.1

- **Phase 1 routing policy.** Domain-routed conservative is the default. Resolved by simulator.
- **Code specialist as primary strict-clean route.** Refuted. Objective_mixed handles strict-clean at least as well at current n.
- **Contrast-based routing.** Not deployable. Feature-cosine detector fails on precision.
- **Deferral-based routing.** Not useful at current eval sizes; doesn't clear improvement threshold.

### Newly added to deferred-not-blocked

- **Vote mode validation.** GENERAL_AND_OBJECTIVE_VOTE_margin needs replication at deployment scale. Cheap to do once controller is integrated. Highest-EV follow-up experiment.
- **Margin calibration.** Currently uses eval-domain std. Proper calibration needs a separate validation distribution. Becomes feasible once deployment generates query traffic.
- **Oracle gap reduction.** The LARGE oracle gap on conservative routing means substantial signal is being left on the table. Better routing schemes that extract this complementarity are valuable. Beyond vote mode, possible directions: learned routing classifier, head-confidence-weighted ensemble, query-feature-based routing.

### Previously deferred, status unchanged

- MATH gate-scale dataset construction (compute-hostile under local budget)
- Full-split HH capture (200-example sufficient for current decisions)
- Harder reasoning generated near-misses (current set sufficient)
- Larger strict-clean code eval (n=16 borderline; expansion would tighten verdicts)
- Code+math mixed head (low priority; GSM8K already works under existing heads)

### Settled (unchanged from v8)

- AntisymLinear vs published GRU head (provisional Path 2)
- GRU temporal aggregation (obsolete as default)
- LayerNorm vs NoNorm (both retained, both useful)
- Heterogeneous tap interface (validated)
- Mixed-domain training (settled)
- Reasoning specialist (not needed)
- Science specialist (not needed)

## 7. Phase 2 implications (updated)

v8's framing of Phase 2 as backbone-regularization-pass holds. v8.1 adds one constraint:

**The Phase 2 L_eval regularizer must preserve head complementarity, not just individual head performance.** The simulator's LARGE oracle gap shows the heads contain genuinely orthogonal signal. If Phase 2 regularization preserves HH performance and objective_mixed performance independently but allows their disagreement structure to collapse (e.g., both projections drift toward a common direction), the future vote-mode validation would fail despite individual heads looking fine.

Operational implication: the Phase 2 L_eval should include a disagreement-preservation term. Something like:

```
L_eval_complementarity = E[|sigmoid(hh_score) - sigmoid(obj_score)|]
                         (computed over branches where labels disagree about which head is correct)
```

The goal is to keep the heads making *different* errors on different cases, not to make both heads more accurate in correlated ways. This is a real Phase 2 design constraint that wasn't in v8.

Phase 2 compute envelope and L_LM mix unchanged from v8 §7.

## 8. Next actions, in order

1. **Implement the BG controller deployment layer.** Two modes: conservative (default), experimental_vote. Thin read-only Python module that takes query + domain hint → returns head pairwise score. ~200-400 lines of code. No new training, no new generation.

2. **Integrate the controller with existing inference paths.** Connect to wherever branch selection actually happens in deployment (likely Hunter-Seeker or future BG-using agent). This is real engineering work, not research.

3. **Run validation queries through both modes.** Once integrated, collect 200-500 deployment queries scored under both conservative and experimental_vote. Compare. If vote consistently outperforms conservative, write v8.2 promoting vote to default.

4. **Margin calibration based on deployment data.** Use the validation queries to compute proper per-head margin std, replacing the approximate eval-domain std currently used in vote mode.

5. **Phase 2 cloud quote and L_LM mix validation.** Unchanged from v8 §8. Now includes complementarity preservation as a constraint.

6. **Optional polish, low priority.** Bootstrap CIs on simulator results, larger strict-clean code eval, full-split HH Experiment 2 Redux. None block deployment.

## 9. One-paragraph thesis (updated)

The BG controller for Phase 1 uses an HH-trained head for preference/coherence contrasts and a code+reasoning mixed-trained head for objective contrasts, deployed via a simple domain-routed controller. A pure code specialist is retained as backup/ablation reference but not as the primary strict-clean code route — objective_mixed handles strict-clean as well or better at current eval n. The two production heads carry complementary signal: a magical oracle router would gain over 8 pp pairwise over the conservative policy. A simple margin-vote between the heads extracts most of this complementarity (oracle gap 0.009) and is available as an experimental controller mode pending validation at deployment scale and proper margin calibration. Phase 2 backbone regularization must preserve the head-complementarity structure, not just individual head accuracy.

## 10. Documents superseded

This v8.1 supersedes:

- `phase1-architecture-locked.md` (the immediate parent)

Plus all v8's superseded documents (v4, v5, v7, handoffs, tap interface revision).

The v8 document remains in the archive; v8.1 is the canonical reference.

## 11. What's not changed by v8.1

- The locus memo, CLT paper, math BG-gate pilot status unchanged.
- The Hunter-Seeker ARC agent and Ouro depth expansion remain separate tracks.
- The architectural constants from v8 §4 unchanged.
- The validated eval domains from v8 §5 unchanged (just consolidated into the reusable simulator bundle).
- v8's contrast-type principle unchanged. The principle is unaffected by the vote-mode finding; voting is a complementarity-extraction mechanism, not a new contrast type.

## 12. Quick reference (added in v8.1)

For agents/engineers integrating the controller:

```
HEADS:
  hh_general            = HH-trained, 47_concat_L1_L4 / AntisymLinearNoNorm
  objective_mixed       = MIX_CODE_REASONING, 36_L4 / AntisymLinearNoNorm
  code_specialist_backup = code-trained, 36_L4 / AntisymLinear

ROUTING (conservative, default):
  domain in {HH, preference, unknown}  → hh_general
  domain in {code, strict_clean_code, reasoning, science, math, objective}
                                       → objective_mixed
  
ROUTING (experimental_vote):
  always compute both hh_general and objective_mixed
  return the head with higher normalized margin

DEFER: false (not implemented)
CONTRAST_DETECTOR: not used (deployability weak)
```

## Read-only BG controller implementation (2026-05-18)

- BG_CONTROLLER_ARTIFACT_VERDICT = READY
- BG_CONTROLLER_REPLAY_VERDICT = PASS
- module path: `src/evaluator/bg_controller.py`
- supported modes: `conservative`, `experimental_vote`, `code_backup`, `diagnostic_all`
- conservative routing: `hh`/`preference`/`unknown` use `hh_general`; `code`/`strict_clean_code`/`reasoning`/`science`/`math`/`gsm8k`/`objective` use `objective_mixed`.
- experimental vote caveat: exposed for validation only; it uses label-free normalized margins and is not the default controller route.
- replay result path: `artifacts/reports/probes/bg_controller_replay_2026-05-18.md`
- usage doc path: `docs/evaluator/controller-usage.md`
- interpretation: the first BG Phase 1 controller layer is ready as a read-only branch-selection component over existing candidate features.

## Read-only transformer BG integration + best-of-N smoke (2026-05-18)

- BG_TRANSFORMER_CAPTURE_INSPECTION_VERDICT = READY
- BG_TRANSFORMER_BEST_OF_N_SMOKE_VERDICT = PASS
- BG_TRANSFORMER_INTEGRATION_VERDICT = PASS
- BG_DEVIL_TASK_INVENTORY_VERDICT = READY
- BG_DEVIL_BEST_OF_N_VERDICT = PASS
- BG_TRANSFORMER_UNIT_TEST_VERDICT = PASS
- module paths: `src/evaluator/bg_transformer_features.py`, `utilities/tests/manual/run_bg_transformer_best_of_n_smoke.py`

- smoke report paths: `artifacts/reports/probes/bg_transformer_best_of_n_smoke_all_2026-05-18.md`, `artifacts/reports/probes/bg_transformer_capture_inspection_2026-05-18.md`
- devil task report path: `artifacts/reports/probes/bg_devil_task_inventory_2026-05-18.md`
- interpretation: v8.1 now has a read-only live transformer integration that preserves conservative routing and produces complete traces; experimental vote remains diagnostic and direct Ouro code-generation quality is not promoted by this smoke.

## BG steering and partial-trajectory routing suite (2026-05-18)

- BG_STEERING_PREFLIGHT_VERDICT = READY
- BG_STEERING_TASK_SUITE_VERDICT = READY
- BG_BRANCH_POOL_VERDICT = READY
- BG_REACHABILITY_GATE_VERDICT = READY
- BG_PARTIAL_FEATURE_VERDICT = READY
- BG_PARTIAL_ROUTING_VERDICT = NEUTRAL
- BG_COMPUTE_ALLOCATION_VERDICT = INSUFFICIENT
- BG_WRAPPER_MATCHED_VERDICT = SKIPPED
- BG_SOFT_STEERING_VERDICT = STABLE_NO_EFFECT
- BG_LATENT_BRANCH_SELECTION_VERDICT = HELPS
- OVERALL_BG_STEERING_VERDICT = NEUTRAL
- generator reachability result: non-code objective domains were reachable; direct code generation was not reliably reachable.
- devil task result: no passing devil branch was generated in the reachability gate.
- full report paths: `artifacts/reports/probes/bg_steering_suite_2026-05-18/summary.md`, `artifacts/reports/probes/bg_steering_suite_2026-05-18/analysis.md`, `docs/evaluator/steering-and-routing-suite.md`
- interpretation: the locked v8.1 conservative controller is mechanically usable for partial routing, but the first full comparison does not justify promoting BG steering/allocation beyond neutral; text-prefix selection is promising only as a small pilot.
## BG trajectory prediction sweep (2026-05-18)

BG_TRAJECTORY_PREFLIGHT_VERDICT = `READY`.
BG_TRAJECTORY_TASK_SUITE_VERDICT = `READY`.
BG_TRAJECTORY_PARTIALS_VERDICT = `READY`.
BG_TRAJECTORY_CONTINUATION_VERDICT = `READY`.
BG_TRAJECTORY_PREFIX_FEATURE_VERDICT = `READY`.
BG_TRAJECTORY_PREFIX_SCORE_VERDICT = `READY`.
BG_TRAJECTORY_PREDICTION_VERDICT = `STRONG`.
BEST_PREDICTIVE_CELL = `{'domain': 'reasoning', 'prefix_length': 256, 'head_id': 'mixed::MIX_CODE_REASONING::36_mean::AntisymLinear', 'config': '36_mean', 'architecture': 'AntisymLinear', 'top1_lift': 0.16249999999999998, 'top2_lift': 0.04166666666666663, 'pairwise_accuracy': 0.8536585365853658, 'oracle_success': 0.9, 'n_tasks': 20, 'n_pairwise_comparisons': 41}`.
RECOMMENDED_STEERING_TARGET = `{'domain': 'reasoning', 'prefix_length': 256, 'head_id': 'mixed::MIX_CODE_REASONING::36_mean::AntisymLinear', 'head_config': '36_mean', 'architecture': 'AntisymLinear', 'top1_lift': 0.16249999999999998, 'top2_lift': 0.04166666666666663, 'pairwise_accuracy': 0.8536585365853658, 'oracle_success': 0.9}`.
GENERATOR_REACHABILITY_LIMITED = `false`.
Interpretation: Run a targeted Stage 2 steering-sensitivity probe at the best predictive cell. Measure state movement in the BG-readable direction, output stability, final correctness, and positive-vs-negative-vs-random controls.
Full reports: `artifacts/reports/probes/bg_trajectory_prediction_2026-05-18/summary.md`, `artifacts/reports/probes/bg_trajectory_prediction_2026-05-18/predictive_power.md`, `artifacts/reports/probes/bg_trajectory_prediction_2026-05-18/stage2_recommendation.md`.

## BG Stage 2 layer-hook follow-up (2026-05-18)

BG_STAGE2_PARTIAL_TRACE_AUDIT_VERDICT = READY
BG_LAYERHOOK_FOLLOWUP_PREFLIGHT_VERDICT = READY
BG_LAYERHOOK_FOLLOWUP_TASKS_VERDICT = READY
BG_LAYERHOOK_FOLLOWUP_SWEEP_VERDICT = READY
BG_LAYERHOOK_MECHANICAL_VERDICT = READY
BG_LAYERHOOK_SIGNED_CAUSAL_VERDICT = UNSIGNED_EFFECT
BG_SINGLE_LOOP_POSITION_VERDICT = L1_BETTER
BG_MULTILOOP_VERDICT = MULTILOOP_STRONGER
BG_LAYERHOOK_STABILITY_VERDICT = STABLE_BUT_TINY
BG_FINAL_TASK_LIFT_VERDICT = INSUFFICIENT
BG_LAYERHOOK_FOLLOWUP_VERDICT = READ_ONLY_BG_FOR_NOW
BEST_SINGLE_LOOP_MODE = single_loop_L1
BEST_MULTILOOP_MODE = multi_loop_decayed
MULTILOOP_GAIN_OVER_BEST_SINGLE = 0.0706979167497257
RECOMMENDED_NEXT = keep_BG_as_readout_selector_and_revisit_steering_with_empirical_success_direction_or_training

Interpretation: BG remains more reliable as a readout selector than as an inference-time steering vector under this protocol.

Full reports: `docs/evaluator/stage2-layerhook-followup.md`, `artifacts/reports/probes/bg_stage2_layerhook_followup_2026-05-18/summary.md`, `artifacts/reports/probes/bg_stage2_layerhook_followup_2026-05-18/analysis.md`.

## BG empirical steering direction probe (2026-05-18)

BG_EMPIRICAL_STEERING_PREFLIGHT_VERDICT = READY
BG_EMPIRICAL_DIRECTION_BUILD_VERDICT = READY
BG_EMPIRICAL_STEERING_TASKS_VERDICT = READY
BG_EMPIRICAL_STEERING_SWEEP_VERDICT = READY
BG_EMPIRICAL_DIRECTION_CAUSAL_VERDICT = EMPIRICAL_UNSIGNED_ONLY
BG_EMPIRICAL_VS_RAW_VERDICT = EMPIRICAL_BEATS_RAW
BG_STEERING_DIRECTION_GEOMETRY_VERDICT = RAW_READOUT_NOT_PRODUCTION_DIRECTION
BG_EMPIRICAL_STEERING_STABILITY_VERDICT = DESTABILIZING
BG_EMPIRICAL_FINAL_LIFT_VERDICT = NEGATIVE_LIFT
BG_TINY_STEERING_ADAPTER_VERDICT = NO_BETTER_THAN_STATIC
BG_EMPIRICAL_STEERING_VERDICT = DESTABILIZING
MODE_COVERAGE = {"EMPIRICAL_MEAN_DIFF": {"multi_loop_decayed": {"complete_expected_rows": 48, "intervention_rows": 48, "task_count": 6}, "single_loop_L1": {"complete_expected_rows": 48, "intervention_rows": 48, "task_count": 6}}, "EMPIRICAL_WHITENED_DIFF": {"multi_loop_decayed": {"complete_expected_rows": 48, "intervention_rows": 48, "task_count": 6}, "single_loop_L1": {"complete_expected_rows": 48, "intervention_rows": 48, "task_count": 6}}, "LOGISTIC_SUCCESS_PROBE": {"multi_loop_decayed": {"complete_expected_rows": 48, "intervention_rows": 48, "task_count": 6}, "single_loop_L1": {"complete_expected_rows": 48, "intervention_rows": 48, "task_count": 6}}, "RAW_NONORM_READOUT": {"multi_loop_decayed": {"complete_expected_rows": 48, "intervention_rows": 48, "task_count": 6}, "single_loop_L1": {"complete_expected_rows": 48, "intervention_rows": 48, "task_count": 6}}}
MULTILOOP_DECAYED_VS_L1_DELTA = -0.10560436938609094
Interpretation: empirical directions test whether BG is readout-only or whether calibrated success-space directions can become causal handles.
Full reports: `artifacts/reports/probes/bg_empirical_steering_direction_2026-05-18/summary.md`, `artifacts/reports/probes/bg_empirical_steering_direction_2026-05-18/analysis.md`, `docs/evaluator/empirical-steering-direction.md`.

## BG pre-consolidation control probes (2026-05-18)

- BG_RMS_STEERING_VERDICT = `RMS_UNSIGNED_ONLY`
- BG_RMS_VS_L2_VERDICT = `RMS_MATCHES_L2`
- BG_PROPAGATION_VERDICT = `PROPAGATES_TO_LATER_STATES`
- BG_PROPAGATION_DECAY_PROFILE = `SURVIVES_32_TOKENS`
- BG_TEXT_PREFIX_EXPANSION_VERDICT = `WEAK_POSITIVE`
- BG_CAUSAL_GRADIENT_VERDICT = `GRADIENT_NO_BETTER_THAN_RANDOM`
- BG_INFERENCE_TIME_STEERING_VERDICT = `UNSIGNED_ONLY`
- BG_BRANCH_ALLOCATION_VERDICT = `PROMISING`
- BG_PHASE2_REQUIREMENT_VERDICT = `TRAINING_REQUIRED`
- interpretation: The model can be nudged in BG-readable state space, but tested directions do not provide reliable signed control; Phase 2 training is required.
- reports: `artifacts/reports/probes/bg_preconsolidation_control_probes_2026-05-18/summary.md`, `artifacts/reports/probes/bg_preconsolidation_control_probes_2026-05-18/final_analysis.json`, `docs/evaluator/preconsolidation-control-probes.md`

## BG causal intervention adapter (2026-05-18)

BG_CAUSAL_ADAPTER_PREFLIGHT_VERDICT = READY
BG_CAUSAL_ADAPTER_DATASET_VERDICT = READY
BG_CAUSAL_ADAPTER_IMPLEMENTATION_VERDICT = READY
BG_CAUSAL_ADAPTER_TRAINING_VERDICT = PARTIAL
BG_CAUSAL_ADAPTER_TEACHER_FORCED_VERDICT = ADAPTER_IMPROVES_LOGIT_MARGIN
BG_CAUSAL_ADAPTER_FREE_GEN_VERDICT = TEACHER_FORCED_ONLY
BG_PAIRWISE_CAUSAL_CONTRAST_ADAPTER_VERDICT = SKIPPED
BG_CAUSAL_ADAPTER_LEARNING_VERDICT = LEARNS_LOGIT_CONTROL
BG_CAUSAL_ADAPTER_VS_STATIC_VERDICT = ADAPTER_BEATS_STATIC
BG_CAUSAL_ADAPTER_STABILITY_VERDICT = STABLE
BG_CAUSAL_ADAPTER_GENERATION_TRANSFER_VERDICT = TEACHER_FORCED_ONLY
BG_CAUSAL_ADAPTER_VERDICT = LOCAL_LOGIT_CONTROL_ONLY
TEACHER_FORCED_RESULT_INTERPRETATION = TEACHER_FORCED_SHORTCUT_RISK
FREE_GENERATION_EVAL_COMPLETED = true
KL_ANSWER_POSITION_MASKED = true
INTERVENTION_POSITION_KIND = prefix_last_token

The causal adapter test separates local teacher-forced logit control from actual trajectory transfer; overall verdict is LOCAL_LOGIT_CONTROL_ONLY.

Full reports: `artifacts/reports/probes/bg_causal_intervention_adapter_2026-05-18/summary.md`, `artifacts/reports/probes/bg_causal_intervention_adapter_2026-05-18/analysis.md`, `docs/evaluator/causal-intervention-adapter.md`.

## BG sequence-level adapter / final frozen-backbone steering test (2026-05-18)

- BG_SEQUENCE_PARSER_AUDIT_VERDICT: `READY`
- BG_SEQUENCE_REWARD_DISTRIBUTION_VERDICT: `REWARD_SIGNAL_USABLE`
- BG_SEQUENCE_OPTIMIZER_SANITY_MICRO_VERDICT: `OPTIMIZER_MOVES_ADAPTER`
- BG_SEQUENCE_GPU_THROUGHPUT_VERDICT: `OVERNIGHT_FEASIBLE`
- OVERNIGHT_SEQUENCE_ADAPTER_READINESS: `READY`
- BG_SEQUENCE_ADAPTER_PREFLIGHT_VERDICT: `READY`
- BG_SEQUENCE_ADAPTER_DATASET_VERDICT: `PARTIAL`
- BG_SEQUENCE_ADAPTER_IMPLEMENTATION_VERDICT: `READY`
- BG_SEQUENCE_BASELINE_EVAL_VERDICT: `READY`
- BG_SEQUENCE_OPTIMIZER_SANITY_VERDICT: `OPTIMIZER_CAN_LEARN_TRIVIAL_TARGET`
- BG_SEQUENCE_ADAPTER_TRAINING_VERDICT: `SEQUENCE_REWARD_IMPROVES`
- BG_SEQUENCE_ADAPTER_HELDOUT_VERDICT: `NO_ADAPTER_SPECIFIC_TRANSFER`
- BG_SEQUENCE_ADAPTER_TEACHER_FORCED_DIAG_VERDICT: `NO_LOGIT_MARGIN_EFFECT`
- BG_SEQUENCE_ADAPTER_BG_SCORE_DIAG_VERDICT: `BG_SCORE_MOVES`
- BG_SEQUENCE_ADAPTER_GEOMETRY_VERDICT: `MATCHES_PRIOR_DIRECTIONS`
- BG_SEQUENCE_ADAPTER_LEARNING_VERDICT: `LEARNS_SEQUENCE_REWARD`
- BG_SEQUENCE_ADAPTER_VS_RANDOM_VERDICT: `WORSE_THAN_RANDOM`
- BG_SEQUENCE_ADAPTER_STABILITY_VERDICT: `STABLE`
- BG_SEQUENCE_ADAPTER_TRANSFER_VERDICT: `NO_TRANSFER`
- BG_SEQUENCE_LEVEL_ADAPTER_VERDICT: `NO_FROZEN_BACKBONE_WRITE_PATH`
- FROZEN_BACKBONE_INFERENCE_STEERING_STATUS: `CLOSED_UNDER_TESTED_METHODS`
- STOPPING_RULE_APPLIES: `True`
- STOPPING_RULE_SCOPE: `safe_alpha_leq_0_02_under_tested_optimizers`
- RECOMMENDED_NEXT: `consolidate_phase1_phase1_5_and_design_phase2_training_time_integration`
- STOPPING_RULE_SCOPE: `safe_alpha_leq_0_02_under_tested_optimizers`
- report paths:
  - `artifacts/reports/probes/bg_sequence_adapter_quick_preflight_2026-05-18/summary.md`
  - `artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/preflight.md`
  - `artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/sequence_adapter_dataset.md`
  - `artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/implementation_tests.md`
  - `artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/baseline_eval.md`
  - `artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/optimizer_sanity.md`
  - `artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/sequence_training_report.md`
  - `artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/heldout_free_generation_eval.md`
  - `artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/diagnostics.md`
  - `artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/analysis.md`
  - `artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/summary.md`

## Same-prefix hidden-state branch generation suite (2026-05-18)

Report: `docs/evaluator/hidden-state-branch-generation.md`

Artifacts: `artifacts/reports/probes/bg_hidden_state_branch_generation_2026-05-18/`

Verdicts:

- `BG_HIDDEN_BRANCH_FEASIBILITY_VERDICT = HOOK_HIDDEN_ORIGIN_READY`
- `LIVE_BRANCH_METHOD = hook_intervention_per_branch`
- `BG_HIDDEN_BRANCH_GENERATION_VERDICT = HOOK_HIDDEN_ORIGIN_BRANCHES_GENERATED`
- `BG_LATENT_BRANCH_PERSISTENCE_VERDICT = LATENT_BRANCHES_PERSIST_TO_47`
- `BG_HIDDEN_BRANCH_OUTCOME_DATASET_VERDICT = READY`
- `BG_HIDDEN_ORIGIN_BRANCH_SELECTION_VERDICT = NO_HIDDEN_BRANCH_SELECTION_SIGNAL`
- `BG_HIDDEN_BRANCH_L30_L42_GATE_VERDICT = NEEDS_STRONGER_BRANCH_GENERATOR`
- `BG_HIDDEN_BRANCH_ADAPTIVE_THRESHOLD_VERDICT = TOPK_SUFFICIENT`
- `PHASE2_HIDDEN_BRANCH_READINESS = NEEDS_BETTER_BRANCH_EVALUATOR`

Interpretation:

Phase 1 cached branch transfer remains valid as text/candidate branch selection evidence, but it does not establish latent fork viability. The new same-prefix suite generated hook-hidden-origin branches and found measurable geometric persistence plus some outcome diversity, but frozen taps did not select good hidden-origin branches better than random. This adds a Phase 2 constraint: hidden-origin branch selection needs evaluator/calibration work, and true fork/carry still requires branch-aware Ouro cache/state handling.
## Hidden-origin branch taps (2026-05-18)

- PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS = `DATA_LIMITED`
- tap_eval_verdict = `INSUFFICIENT`
- tap_training_verdict = `READY`
- layer_config_verdict = `INSUFFICIENT`
- geometry_verdict = `ALIGNS_WITH_OLD_TAPS`
- report: `artifacts/reports/probes/bg_hidden_origin_taps_2026-05-18/summary.md`

Generate more hidden-origin branch outcome groups.

## Hidden-origin branch diversity v2 and tap reevaluation (2026-05-18)

- PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_V2 = `WEAK`
- generation_verdict = `READY`
- dataset_verdict = `SMALL_BUT_USABLE`
- training_verdict = `READY`
- eval_verdict = `WEAK_SELECTOR`
- layer_config_verdict = `CONCAT_REQUIRED`
- geometry_verdict = `OLD_GEOMETRY_CONFIRMED`
- report: `artifacts/reports/probes/bg_hidden_origin_diversity_v2_2026-05-18/summary.md`

Either expand once more or proceed only to a small selection-only prototype with the caveat locked in.

## Hidden-origin branch diversity v3 and selector reevaluation (2026-05-18)

- PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_V3 = `STILL_DATA_LIMITED`
- HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE = `v3_hidden_origin_tap`
- diversity_ablation_verdict = `DIVERSITY_IMPROVED`
- driver_verdict = `NON_RANDOM_DIRECTIONS_HELP`
- dataset_verdict = `STILL_DATA_LIMITED`
- training_verdict = `WEAK`
- eval_verdict = `DATA_LIMITED`
- geometry_verdict = `OLD_GEOMETRY_CONFIRMED`
- report: `artifacts/reports/probes/bg_hidden_origin_diversity_v3_2026-05-18/summary.md`

Continue targeted data expansion using the v3 recipe before making selector-readiness claims.

