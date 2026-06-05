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

## Hidden-origin branch split salvage and selector reevaluation (2026-05-18)

- `BG_HIDDEN_ORIGIN_SALVAGE_EVAL_VERDICT = WEAK_SELECTOR`
- `BG_HIDDEN_ORIGIN_CV_STABILITY_VERDICT = STABLE_POSITIVE`
- `BG_HIDDEN_ORIGIN_V4_QUOTA_NEED_VERDICT = V4_REQUIRED_HELDOUT_BALANCE`
- `HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE = old_frozen_bg`
- `PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_SALVAGE = WEAK`

Split salvage reused existing v3 branch data only. It reports strict/v3-clean heldout support separately from grouped-CV diagnostics and marks baseline contamination where applicable.
## Hidden-origin branch quota v4 and old-context replay (2026-05-18)

- `BG_HIDDEN_ORIGIN_QUOTA_GENERATION_V4_VERDICT = PARTIAL`
- `BG_HIDDEN_ORIGIN_SELECTOR_EVAL_V4_VERDICT = STILL_DATA_LIMITED`
- `BG_HIDDEN_ORIGIN_OLD_CONTEXT_REPLAY_V4_VERDICT = PARTIAL_MATCH`
- `BG_HIDDEN_ORIGIN_TAP_GEOMETRY_V4_VERDICT = OLD_GEOMETRY_CONFIRMED`
- `HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE_V4 = ensemble`
- `PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_V4 = STILL_DATA_LIMITED`

V4 reserves train/val/heldout task IDs before generation and keeps old-context replay diagnostic-only. Alpha 0.02, sampled labels, and L47 remain excluded from primary readiness claims.

## Hidden-origin Branch Generator v1 (2026-05-18)

Branch Generator v1 was run because v4 confirmed selector geometry but remained heldout-diversity limited. The v1 run tested early L24/L1-style hook perturbations, high-yield non-random directions, a lightweight recipe/CEM schedule, true fork/carry feasibility, and richer outcome diagnostics without training Ouro or changing production routing.

BG_BRANCH_GENERATOR_V1_AUDIT_PLAN_VERDICT = READY
BG_TRUE_FORK_CARRY_PROBE_V1_VERDICT = HOOK_FALLBACK_ONLY
BG_RICH_OUTCOME_SCHEMA_V1_VERDICT = READY
BG_BRANCH_GENERATOR_BASIS_BANK_V1_VERDICT = READY
BG_BRANCH_GENERATOR_PROPOSER_TRAINING_V1_VERDICT = RECIPE_ONLY
BG_BRANCH_GENERATOR_BLACKBOX_SEARCH_V1_VERDICT = WEAK_IMPROVEMENT
BG_BRANCH_GENERATOR_V1_GENERATION_VERDICT = HELDOUT_QUOTA_MET_ONLY
BG_BRANCH_GENERATOR_V1_DIVERSITY_VERDICT = STRONG_IMPROVEMENT
BG_BRANCH_GENERATOR_V1_BEST_METHOD = hs_inspired_controller
BG_BRANCH_GENERATOR_V1_SELECTOR_DATASET_VERDICT = HELDOUT_READY_TRAIN_WEAK
BG_BRANCH_GENERATOR_V1_SELECTOR_TRAINING_VERDICT = WEAK
BG_BRANCH_GENERATOR_V1_SELECTOR_EVAL_VERDICT = WEAK_SELECTOR
BG_BRANCH_GENERATOR_V1_OLD_CONTEXT_REPLAY_VERDICT = PARTIAL_MATCH
BG_BRANCH_GENERATOR_V1_GEOMETRY_VERDICT = OLD_GEOMETRY_CONFIRMED
HIDDEN_ORIGIN_BRANCH_GENERATOR_STATUS_V1 = WEAK_BUT_USABLE
HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE_AFTER_GENERATOR_V1 = v4_hidden_origin_tap

- quota_progress_by_split: `{'all_minimums_met': False, 'heldout': {'behaviorally_diverse_groups': 43, 'behaviorally_diverse_groups_per_100_rows': 6.554878048780488, 'candidate_pairs': 2296, 'groups': 82, 'minimum_met': True, 'non_tie_pairs': 419, 'non_tie_pairs_per_100_rows': 63.8719512195122, 'parse_rate': 0.8262195121951219, 'quota_minimums': {'behaviorally_diverse_groups': 20, 'non_tie_pairs': 120, 'task_ids': 8}, 'reward_diverse_groups': 38, 'stability_rate': 1.0, 'stable_primary_rows': 656, 'task_ids': 10, 'task_ids_with_non_tie_pair_list': ['OpenBookQA/14', 'mmlu/anatomy/12', 'mmlu/anatomy/7', 'mmlu/anatomy/8', 'mmlu/high_school_chemistry/1', 'mmlu/high_school_chemistry/10', 'mmlu/high_school_physics/11', 'sciq/sciq/22'], 'task_ids_with_non_tie_pairs': 8, 'tie_pairs': 1877, 'tie_rate': 0.8175087108013938}, 'train': {'behaviorally_diverse_groups': 35, 'behaviorally_diverse_groups_per_100_rows': 1.4583333333333333, 'candidate_pairs': 8400, 'groups': 300, 'minimum_met': False, 'non_tie_pairs': 473, 'non_tie_pairs_per_100_rows': 19.708333333333332, 'parse_rate': 0.6891666666666667, 'quota_minimums': {'behaviorally_diverse_groups': 60, 'non_tie_pairs': 250, 'task_ids': 24}, 'reward_diverse_groups': 34, 'stability_rate': 1.0, 'stable_primary_rows': 2400, 'task_ids': 40, 'task_ids_with_non_tie_pair_list': ['ARC-Challenge/1', 'ARC-Challenge/17', 'ARC-Challenge/19', 'ARC-Challenge/2', 'OpenBookQA/3'], 'task_ids_with_non_tie_pairs': 5, 'tie_pairs': 7927, 'tie_rate': 0.9436904761904762}, 'val': {'behaviorally_diverse_groups': 1, 'behaviorally_diverse_groups_per_100_rows': 0.1, 'candidate_pairs': 3452, 'groups': 127, 'minimum_met': False, 'non_tie_pairs': 7, 'non_tie_pairs_per_100_rows': 0.7, 'parse_rate': 0.874, 'quota_minimums': {'behaviorally_diverse_groups': 15, 'non_tie_pairs': 60, 'task_ids': 6}, 'reward_diverse_groups': 1, 'stability_rate': 1.0, 'stable_primary_rows': 1000, 'task_ids': 8, 'task_ids_with_non_tie_pair_list': ['mmlu/high_school_chemistry/16'], 'task_ids_with_non_tie_pairs': 1, 'tie_pairs': 3445, 'tie_rate': 0.9979721900347625}}`
- diversity_questions: `{'CEM_or_ES_improved_over_HS': False, 'K6_yield': 0.0, 'K8_remained_useful': True, 'K8_yield': 1.971057884231537, 'L24_remained_better_than_L36': True, 'L24_yield': 1.9863013698630136, 'L36_yield': 1.8485915492957747, 'alpha_0_005_remained_best': False, 'alpha_0_005_yield': 1.8851508120649652, 'alpha_0_01_yield': 2.3026315789473686, 'beat_static_v4_recipe': True, 'cem_yield': 1.957070707070707, 'hs_yield': 2.217741935483871, 'learned_proposer_helped': True, 'non_random_directions_remained_useful': True, 'non_random_yield': 3.75, 'random_yield': 2.1169354838709675, 'static_yield': 1.8333333333333333, 'structured_low_rank_coefficients_helped': False, 'true_behavioral_diversity_not_instability': True, 'true_fork_carry_changed_persistence': False}`
- recommended_next: `Either run a small selection-only prototype with caveat or run targeted generator v1.1 if one recipe clearly remains.`

Selector readiness, if claimed, uses only primary-safe deterministic alpha <= 0.01 heldout rows. Diagnostic alpha 0.02, sampled labels, L47 branches, old-context replay, and auxiliary diagnostics are not readiness support.


## Universal branch-content taps v1 (2026-05-18)

Universal Branch-Content Taps v1 tested whether one tiny hidden-state pairwise evaluator can cover both old content/candidate selection and same-prefix hidden-origin branch survival. It trained only new standalone tap heads and did not alter Ouro, existing BG taps, registries, wrapper/local-agent routing, or production behavior.

BG_UNIVERSAL_TAP_INVENTORY_VERDICT = READY
BG_UNIVERSAL_OLD_CONTENT_DATASET_VERDICT = READY
BG_UNIVERSAL_HIDDEN_BRANCH_DATASET_VERDICT = READY
BG_UNIVERSAL_BRIDGE_DATASET_VERDICT = READY
BG_UNIVERSAL_DATA_EXPANSION_VERDICT = SKIPPED
BG_UNIVERSAL_TAP_DATASET_VERDICT = READY
BG_UNIVERSAL_TAP_TRAINING_VERDICT = READY
BG_UNIVERSAL_OLD_CONTEXT_EVAL_VERDICT = MATCHES_OR_BEATS_OLD_TAPS
BG_UNIVERSAL_HIDDEN_BRANCH_EVAL_VERDICT = SMALL_DEGRADATION
BG_UNIVERSAL_BRIDGE_EVAL_VERDICT = NO_BRIDGE_SIGNAL
BG_UNIVERSAL_LAYERWISE_PRUNING_VERDICT = TOPK_SURVIVAL_ONLY
BG_UNIVERSAL_DOMAIN_GENERALIZATION_VERDICT = REASONING_SCIENCE_ONLY
BG_UNIVERSAL_TAP_GEOMETRY_VERDICT = OLD_GEOMETRY_CONFIRMED
UNIVERSAL_BRANCH_CONTENT_TAP_STATUS = FUSION_NEEDED

- old_content_counts: `{'feature_config_counts': {'24_L4': 462, '24_mean': 462, '30_L4': 0, '36_L4': 462, '36_mean': 462, '42_L4': 0, '47_L4': 462, '47_mean': 462, 'concat_24_30_36': 0, 'concat_24_36': 462, 'concat_24_36_47': 462, 'concat_36_42_47': 0, 'concat_36_47': 462}, 'pairs': 462, 'pairs_by_domain': {'math_simple_arithmetic': 143, 'reasoning': 183, 'science': 136}, 'pairs_by_split': {'heldout': 73, 'train': 295, 'val': 94}, 'pairs_by_type': {'old_content': 462}, 'tasks_by_split': {'heldout': ['ARC-Challenge/0', 'ARC-Challenge/19', 'gsm8k/1', 'gsm8k/12', 'gsm8k/5', 'mmlu/high_school_biology/10', 'mmlu/high_school_biology/12', 'mmlu/high_school_biology/17', 'mmlu/high_school_biology/9'], 'train': ['ARC-Challenge/1', 'ARC-Challenge/11', 'ARC-Challenge/12', 'ARC-Challenge/13', 'ARC-Challenge/14', 'ARC-Challenge/15', 'ARC-Challenge/16', 'ARC-Challenge/17', 'ARC-Challenge/18', 'ARC-Challenge/6', 'ARC-Challenge/8', 'gsm8k/0', 'gsm8k/10', 'gsm8k/11', 'gsm8k/13', 'gsm8k/14', 'gsm8k/16', 'gsm8k/17', 'gsm8k/19', 'gsm8k/2', 'gsm8k/3', 'gsm8k/7', 'gsm8k/8', 'mmlu/high_school_biology/1', 'mmlu/high_school_biology/11', 'mmlu/high_school_biology/15', 'mmlu/high_school_biology/19', 'mmlu/high_school_biology/2', 'mmlu/high_school_biology/3', 'mmlu/high_school_biology/4', 'mmlu/high_school_biology/5', 'mmlu/high_school_biology/7', 'mmlu/high_school_biology/8'], 'val': ['ARC-Challenge/10', 'ARC-Challenge/2', 'ARC-Challenge/3', 'ARC-Challenge/4', 'ARC-Challenge/5', 'ARC-Challenge/7', 'ARC-Challenge/9', 'gsm8k/15', 'gsm8k/9', 'mmlu/high_school_biology/14', 'mmlu/high_school_biology/18']}}`
- hidden_branch_counts: `{'feature_config_counts': {'24_L4': 1753, '24_mean': 1753, '30_L4': 1753, '36_L4': 1753, '36_mean': 1753, '42_L4': 1753, '47_L4': 1753, '47_mean': 1753, 'concat_24_30_36': 1753, 'concat_24_36': 1753, 'concat_24_36_47': 1753, 'concat_36_42_47': 1753, 'concat_36_47': 1753}, 'pairs': 1753, 'pairs_by_domain': {'reasoning': 1074, 'science': 679}, 'pairs_by_split': {'heldout': 483, 'train': 1090, 'val': 180}, 'pairs_by_type': {'hidden_branch': 1753}, 'tasks_by_split': {'heldout': ['OpenBookQA/14', 'OpenBookQA/18', 'mmlu/anatomy/12', 'mmlu/anatomy/7', 'mmlu/anatomy/8', 'mmlu/high_school_chemistry/1', 'mmlu/high_school_chemistry/10', 'mmlu/high_school_physics/11', 'sciq/sciq/22'], 'train': ['ARC-Challenge/1', 'ARC-Challenge/17', 'ARC-Challenge/19', 'ARC-Challenge/2', 'OpenBookQA/3', 'mmlu/anatomy/12', 'mmlu/anatomy/8', 'mmlu/high_school_chemistry/10', 'mmlu/high_school_physics/11', 'sciq/sciq/22'], 'val': ['ARC-Challenge/17', 'mmlu/high_school_chemistry/16']}}`
- bridge_counts: `{'feature_config_counts': {'24_L4': 2142, '24_mean': 2142, '30_L4': 2142, '36_L4': 2142, '36_mean': 2142, '42_L4': 2142, '47_L4': 2142, '47_mean': 2142, 'concat_24_30_36': 2142, 'concat_24_36': 2142, 'concat_24_36_47': 2142, 'concat_36_42_47': 2142, 'concat_36_47': 2142}, 'pairs': 2142, 'pairs_by_domain': {'reasoning': 1316, 'science': 826}, 'pairs_by_split': {'heldout': 580, 'train': 1342, 'val': 220}, 'pairs_by_type': {'bridge': 2142}, 'tasks_by_split': {'heldout': ['OpenBookQA/14', 'OpenBookQA/18', 'mmlu/anatomy/12', 'mmlu/anatomy/7', 'mmlu/anatomy/8', 'mmlu/high_school_chemistry/1', 'mmlu/high_school_chemistry/10', 'mmlu/high_school_physics/11', 'sciq/sciq/22'], 'train': ['ARC-Challenge/1', 'ARC-Challenge/17', 'ARC-Challenge/19', 'ARC-Challenge/2', 'OpenBookQA/3', 'mmlu/anatomy/12', 'mmlu/anatomy/8', 'mmlu/high_school_chemistry/10', 'mmlu/high_school_physics/11', 'sciq/sciq/22'], 'val': ['ARC-Challenge/17', 'mmlu/high_school_chemistry/16']}}`
- recommendation: `Build an explicit composite selector rather than forcing a single universal head.`

Readiness requires old-context, hidden-branch, and bridge support. Cached coding features were inspected but had no non-tie within-task labels, so coding remains coverage-limited.

## Gated branch-content selector v1 (2026-05-18)

Gated/Fusion Branch-Content Selector v1 tested whether old content taps, hidden-origin branch taps, bridge heads, universal heads, and readiness diagnostics can be combined without collapsing all roles into one linear universal tap.

BG_GATED_SELECTOR_INVENTORY_VERDICT = READY
BG_GATED_SELECTOR_EXPERT_SCORES_VERDICT = READY
BG_GATED_SELECTOR_DATASET_VERDICT = READY
BG_GATED_SELECTOR_TRAINING_VERDICT = READY
BG_GATED_SELECTOR_EXPERT_ABLATION_VERDICT = INCONCLUSIVE
BG_GATED_OLD_CONTEXT_EVAL_VERDICT = MATCHES_OR_BEATS_OLD_TAPS
BG_GATED_HIDDEN_BRANCH_EVAL_VERDICT = SMALL_DEGRADATION
BG_GATED_BRIDGE_EVAL_VERDICT = BRIDGE_FIXED
BG_GATED_LAYERWISE_PRUNING_VERDICT = OLD_NEW_COMPOSITE_BEST
BG_GATED_DOMAIN_COVERAGE_VERDICT = MULTIDOMAIN_READY
BG_GATED_CALIBRATION_OOD_VERDICT = CALIBRATION_WEAK
BG_GATED_GEOMETRY_VERDICT = OLD_GEOMETRY_DOMINATES
BG_GATED_AS_OLD_TAP_REPLACEMENT_VERDICT = SAFE_REPLACEMENT_CANDIDATE
GATED_BRANCH_CONTENT_SELECTOR_STATUS = OLD_NEW_COMPOSITE_SUFFICIENT

- recommendation: `Prefer the simpler old+branch+bridge composite over the learned gate for now; keep top-k survival and do not change production routing.`
- no Ouro weights, tokenizer files, checkpoints, old taps, tap registries, wrapper/local-agent routing, or production routing were modified.
- expert/tap scores were used only as input features, not as labels.

## Fixed-composite branch survival policy v1 (2026-05-18)

This run converted the corrected gated selector result into a validation-selected fixed old+branch+bridge survival policy with explicit veto/rescue and missing-expert/OOD fallback.

BG_FIXED_COMPOSITE_SURVIVAL_INVENTORY_VERDICT = READY
BG_FIXED_COMPOSITE_SURVIVAL_DATASET_VERDICT = READY
BG_FIXED_COMPOSITE_SURVIVAL_FEATURES_VERDICT = READY
BG_FIXED_COMPOSITE_SURVIVAL_BASELINES_VERDICT = READY
BG_FIXED_COMPOSITE_OPTIMIZATION_VERDICT = OLD_BRANCH_BRIDGE_SUFFICIENT
BG_FIXED_COMPOSITE_VETO_RESCUE_OPTIMIZATION_VERDICT = READY
BG_FIXED_COMPOSITE_LEARNED_RESCUE_VERDICT = WORSE_THAN_RULES
BG_FIXED_COMPOSITE_MISSING_OOD_POLICY_VERDICT = ROBUST
BG_FIXED_COMPOSITE_SURVIVAL_HELDOUT_EVAL_VERDICT = SURVIVAL_READY
BG_FIXED_COMPOSITE_SURVIVAL_FRONTIER_VERDICT = CLEAR_OPERATING_POINT
BG_FIXED_COMPOSITE_LAYER_ORIGIN_DOMAIN_VERDICT = UNIFORM_POLICY_SUFFICIENT
BG_FIXED_COMPOSITE_OLD_CODE_PRESERVATION_VERDICT = PRESERVED
BG_FIXED_COMPOSITE_SELECTION_ONLY_READINESS_VERDICT = READY
FIXED_COMPOSITE_BRANCH_SURVIVAL_POLICY_STATUS = SURVIVAL_READY

- selected policy: `selected_policy = fixed_composite_conservative_top4; oracle_retention = 0.931; false_prune_rate = 0.069; avg_survivors = 3.873`
- recommendation: `Proceed to a small selection-only Phase 2 prototype using BGV1 branches, the fixed old+branch+bridge composite, the selected conservative top-k survival operating point, and missing/OOD fallback. Keep veto/rescue as a guardrail, not as a replacement for the selected heldout-ready operating point. selected_policy = fixed_composite_conservative_top4; oracle_retention = 0.931; false_prune_rate = 0.069; avg_survivors = 3.873. Do not claim action steering.`
- learned gated selector remains diagnostic; it is not the primary pruning selector.
- no Ouro weights, tokenizer files, checkpoints, old tap registries, wrapper/local-agent routes, or production routing were modified.

## Selection-only Phase 2 prototype v1 (2026-05-18)

SELECTION_ONLY_PHASE2_PROTOTYPE_STATUS = SURVIVAL_READY_FINAL_ARBITER_WEAK

- cached reproduction: `REPRODUCED`
- live/counterfactual prototype: `SURVIVAL_POSITIVE_FINAL_SELECTION_WEAK`
- final arbiter: `FINAL_SELECTION_WEAK`
- steering readiness: `NEEDS_FINAL_ARBITER_FIRST`
- recommendation: Train or evaluate a stronger final arbiter among top4 survivors before steering.
- no action steering was tested; no production routing changed.

## Final arbiter among top4 survivors v1 (2026-05-18)

FINAL_ARBITER_TOP4_STATUS = FINAL_ARBITER_WEAK_BUT_USEFUL
SELECTION_ONLY_PHASE2A_STATUS_AFTER_FINAL_ARBITER = NEEDS_MORE_FINAL_ARBITER_WORK

- heldout eval: `FINAL_ARBITER_WEAK`
- selected model: `listwise_softmax`
- readiness: `FINAL_ARBITER_WEAK_BUT_IMPROVED`
- recommendation: Run a small improved-arbiter v1.1 or proceed only with explicit weak-baseline caveat.
- no action steering was tested.

## Final arbiter among top4 survivors v1.1 (2026-05-18)

FINAL_ARBITER_TOP4_V1_1_STATUS = NO_IMPROVEMENT
SELECTION_ONLY_PHASE2A_STATUS_AFTER_FINAL_ARBITER_V1_1 = NEEDS_DOMAIN_SPECIALIZATION

- split guard: `FRESH_HELDOUT_READY`
- selected model: `tie_aware_rank_listwise`
- heldout eval: `NO_IMPROVEMENT`
- readiness: `NEEDS_REASONING_ARBITER`
- recommendation: Return to expert/bridge signal quality; v1.1 did not improve final selection.
- no action steering was tested.

## Weight-space merged taps proposal (2026-05-18)

Post-v10 routing remains locked. A proposed diagnostic follow-up is `bg_weight_space_merged_taps_v1`: extract old objective/code, universal, hidden-branch, and bridge tiny-head directions; build old-preserving branch-validity residual merges; evaluate against the fixed old+branch+bridge composite and final-arbiter datasets.

- planning doc: `docs/evaluator/bg_weight_space_merged_taps_plan.md`
- status: not yet run.
- no routing change, no old tap overwrite, and no action steering claim.

## Weight-space merged branch-content taps v1 (2026-05-18)

`MERGED_WEIGHT_BRANCH_CONTENT_TAP_STATUS = FINAL_ARBITER_IMPROVES_ONLY`. The run extracted old/content, hidden-branch, and bridge tap directions, aligned them into shared feature coordinates, built residualized merged candidates, and acquired top4 survivor hidden features from cached raw artifacts for final-arbiter rescoring. No action steering or routing change was tested.

Report: `docs/evaluator/bg_merged_weight_branch_content_taps_v1.md`.

## DualAnchor architecture-looped stratified probe v3 (2026-05-31)

Status: `ARCHITECTURE_LOOPED_SURVIVAL_READY_TERMINAL_DEFER_REQUIRED`.

This run scaled the DualAnchor architecture-shaped loop without steering. Taps were active at layers 24, 36, and 47 across loops L1-L4, with only terminal `L4_47` eligible for confidence-gated collapse. It uses cumulative hook approximation at decoder-layer surfaces; it does not claim autoregressive branch-specific KV/cache fork/carry or compute savings.

Headline metrics:

- tasks: `48`
- stage oracle retention: `0.9848484848484849`
- terminal oracle retained: `1.0`
- terminal forced top1 oracle: `0.9166666666666666`
- terminal reward-diverse rate: `0.22916666666666666`
- positive-oracle rate: `0.3541666666666667`

Locked-baseline candidate:

- selector: DualAnchor `MIX_CODE_REASONING` + `MIX_OBJECTIVE_ALL`
- schedule: `L1_24 -> L1_36 -> L1_47 -> L2_24 -> L2_36 -> L2_47 -> L3_24 -> L3_36 -> L3_47 -> L4_24 -> L4_36 -> terminal L4_47`
- threshold: `mean_floor_very_loose`
- budget: `8`
- L47: active in nonterminal loops
- terminal: confidence-gated top1; otherwise defer/keep terminal survivors

Readiness verdict: `READY_WITH_TERMINAL_DEFER`.
No steering was tested.

