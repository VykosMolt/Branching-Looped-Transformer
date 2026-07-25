# Proto-Introspection Evidence Matrix (Ouro-RLTT)

**Date:** 2026-06-17 (compiled 2026-06-22)
**Scope:** read-only audit of existing evaluator / tap / trajectory / branching artifacts. No new experiments were run. No model checkpoints were modified.
**Purpose:** turn the existing evidence into a rigorous, bounded *weak proto-introspection* proof package, identify the minimal remaining controls, and state exactly what may and may not be written in the paper.

---

## 1. Executive Summary

Weak proto-introspection — *the claim that intermediate hidden states of the looped Ouro model carry readable information about the quality / likely success of the model's own ongoing computation* — is **substantially supported but not yet airtight**. Five of seven pillars are at STRONG or PARTIAL with load-bearing artifacts and clean controls; the two gating weaknesses are timing (pre-answer leakage not isolated by a dedicated control) and shortcut-specificity (length / logprob / prompt-family baselines never run as explicit comparators for the trajectory predictor). Prediction is strong: the frozen pairwise relational evaluator reaches 95.2% fixed-order HH-RLHF preference accuracy and a tiny antisymmetric prefix head predicts which not-yet-completed branch will succeed at pairwise accuracy 0.854 (top-1 lift +0.162 over random, oracle 0.90) on held reasoning tasks. Specificity is strong on the relational axis (pairwise 95.2% vs pointwise-nonlinear ~65% vs pointwise-linear 21.75% below chance; antisymmetry ρ≈−0.94) but PARTIAL on trivial-shortcut controls. Timing is PARTIAL: the predictor reads prefix-position loop states and predicts the outcome of continuations not yet generated, with positive lift even at 32–64-token prefixes, but no control yet isolates states captured strictly before any answer token can leak the label. Role separation (validity / content / correctness) is empirically clean. The causal/control boundary is itself strong evidence: readout exists while naive frozen-backbone steering is closed under all tested methods — readout and control are demonstrably different problems. **No new expensive GPU experiments are required before a first paper draft can begin; the three minimal controls (pre-answer leakage, shortcut baselines, task-grouped heldout) are mostly CPU re-analyses of already-captured features and can run in parallel with writing.** The weak claim is paper-ready *with caveats*; the strong claims (capability gain, autonomous self-control) are not, and must not be made.

---

## 2. Operational Definition

**Proto-introspection (operational, weak):** intermediate hidden states of a looped model contain *readable* information about the *quality, stability, uncertainty, likely success/failure, or branch viability* of the model's *own ongoing computation*, before — or not reducible to — external final judgment.

The allowed claim is exactly:

> **Looped-model hidden states expose readable process-quality signals about the model's own ongoing computation.**

This definition requires four things, each of which is tested below:
1. **Readable internal signal** — a tiny external readout recovers the signal from hidden states (P1).
2. **About the model's own ongoing computation** — the target is the success/quality of the model's own (possibly unfinished) trajectory, not a static text label (P1, P4).
3. **Present in intermediate hidden states** — signal exists at intermediate loops / prefix positions, not only at the final output token (P2).
4. **Not reducible to final external judgment** — signal appears before or independently of the external verifier's terminal correctness call (P2, P5).

**Explicitly out of scope / NOT claimed:**
- **Consciousness** — not claimed, not measured, not implied.
- **Self-awareness** — not claimed.
- **Verbal self-report** — the model is not asked to describe itself; signal is read from activations by an external probe.
- **Generic answer correctness** — correctness is an *external verifier* axis, kept separate from readout (see P5).
- **External evaluator capability** — a strong external probe is not the same as the model "knowing" anything; we claim only that the information is *present and readable*.
- **Autonomous internal use of the signal for self-control** — explicitly NOT claimed; the control boundary (P7) shows the opposite under current methods.

---

## 3. Evidence Pillar Matrix

| Pillar | Status | One-line | Allowed claim |
|---|---|---|---|
| **P1 Prediction** | **STRONG** | Frozen hidden states predict preference (95.2%) and not-yet-finished branch success (pair acc 0.854, +0.162 top-1 lift). | Hidden states predict process quality / success above strong baselines. |
| **P2 Timing** | **PARTIAL** | Prefix/loop states predict outcome of continuations not yet generated; positive lift at 32–64-tok prefixes; no strict pre-answer-token control yet. | Signal is present in intermediate loop/prefix states (partial pre-answer evidence). |
| **P3 Specificity** | **PARTIAL** | Relational vs pointwise + antisymmetry controls are strong; length/logprob/prompt-family baselines not yet run for the trajectory predictor. | Signal is relational and order-sensitive, not a constant/degenerate or pair-order artifact; trivial-shortcut controls incomplete. |
| **P4 Utility** | **PARTIAL** (weak utility yes, strong no) | Taps rank/retain/diagnose; survivor-set retention high; but forced-top1 count-normalized lift ≈0 and fork adds nothing over K-matched sampling. | Signal is useful for external readout / ranking / retention / diagnosis; end-to-end capability gain not shown. |
| **P5 Role separation** | **STRONG** | Validity (DualAnchor), content (CoreContent), correctness (verifier) empirically + conceptually separated. | The three axes are distinct; tap scores are not correctness labels. |
| **P6 Cross-domain** | **PARTIAL** | Signal transfers across HH / GSM8K / code / reasoning / science, but needs domain-specific heads; one universal head is weak. | Signal is not HH-only; it transfers partially across objective domains with domain-specialized readouts. |
| **P7 Control boundary** | **STRONG (as a boundary)** | Readout exists; frozen-backbone steering closed under all tested methods. | Proto-introspective readout is supported; autonomous self-control is not solved and requires training-time integration. |

### P1 — Prediction
- **Status:** STRONG.
- **Supporting artifacts:** `artifacts/reports/evaluator/probe_loop_geometry_hh.json`; `artifacts/reports/probes/bg_trajectory_prediction_2026-05-18/{summary.json,predictive_power.md}`; `artifacts/checkpoints/evaluator/pairwise_epoch2.pt`; `docs/evaluator/flip-test-interpretation.md`.
- **Key numbers:** HH pairwise preference acc 95.2% (8141/8552 full test); 200-subset canonical 0.95 (Thinking) / 0.945 (RLTT); pairwise linear diff probe 84.5%. Trajectory best cell (reasoning, 256-tok prefix): pairwise acc 0.854, top-1 lift +0.162, top-2 lift +0.042, oracle 0.90, AUC(margin→success) 0.727, Spearman 0.364, top-1-lift bootstrap CI [0.0625, 0.2625], n_tasks 20 / 41 pairwise comparisons. 368 "strong" cells; 300 cells with top-1 lift ≥ 0.10.
- **Caveats:** trajectory n per cell is small (20 tasks; 41 comparisons in the best cell). Preference accuracy is fixed-order (see P3). The trajectory head reads prefix states but cells with longer prefixes may include partial answer (see P2).
- **Exact claim allowed:** "Frozen Ouro hidden states contain readable information that predicts both human-preference ordering and the eventual success of an ongoing branch, recoverable by a tiny antisymmetric linear/relational readout."
- **Missing controls:** shortcut baselines (P3-B) and task-grouped heldout (P3-C) to convert "predicts" into "predicts beyond trivial features."

### P2 — Timing
- **Status:** PARTIAL.
- **Supporting artifacts:** `bg_trajectory_prediction_2026-05-18/{task_suite.md,continued_prefixes.md,preflight.md,prefix_features.pt}` (features `[3 layers × 4 loops × 2048]` captured at the prefix position); `artifacts/reports/evaluator/probe_l1_ablation.json` (L1-only loop carries readable signal).
- **Key numbers:** prefixes 32/64/128/256 tokens, 4 branches/task, 960 prefixes (evaluable rate 1.0). Positive top-1 lift at short prefixes: reasoning@64 +0.262, science@32 +0.188, gsm8k@256 +0.200; reasoning peak pairwise acc 0.854 at 256. L1-only (first loop) canonical 0.915 / centered 0.575; signal is present at the first loop, not only the last.
- **Caveats:** the predictor reads states at a prefix *position* and predicts the success of *continuations not yet generated* — that is genuine pre-completion prediction. **But** for short tasks a 256-token prefix can already contain the answer, and no dedicated control isolates states captured strictly *before the first answer token*. Loop-geometry shows L24/L36 fully converge across loops and only L47 is bipartite (L1 distinct), so "intermediate loop" signal at converged layers is weaker than at L47.
- **Exact claim allowed:** "Predictive signal is present in intermediate loop states and at partial-trajectory prefixes, including short (32–64-token) prefixes that precede most of the answer — i.e., the signal is not solely a post-hoc readout of the finished answer." (Stated as *partial* pre-answer evidence, not airtight.)
- **Missing controls:** **P2/P3-A pre-answer leakage audit** (the single most important missing control) — re-probe at a strictly pre-answer cut and confirm lift survives.

### P3 — Specificity
- **Status:** PARTIAL (STRONG on relational/antisymmetry axes; trivial-shortcut baselines incomplete).
- **Supporting artifacts:** `docs/evaluator/flip-test-interpretation.md`; `artifacts/reports/evaluator/probe_l1_ablation.json` (swap-balanced protocol); `probe_loop_geometry_hh.json` (score range, antisymmetry).
- **Key numbers:** pairwise nonlinear 95.2% vs pointwise-nonlinear ~65% vs pointwise-linear 21.75% (below chance, inverted polarity) → signal is *relational*, not single-candidate quality. Antisymmetry ρ≈−0.94; scores span [−1.51, +4.56] (not a constant-output degenerate head). Strict sign-flip 25% at epoch-2 reflects scorer *bias*, not accuracy collapse (epoch-4/5: sign-flip ~96% as bias dissipates and acc collapses to 67.2%/62.4%). Antisymmetrized/centered accuracy ≈0.595–0.62 (the honest bidirectional-comparator number once bias is removed).
- **Caveats:** the trajectory predictor has NOT been compared against length, answer-token-count, prompt-family, candidate-source, or logprob/entropy baselines; pair-order is controlled (antisymmetric construction + swap-balanced) but domain/length shortcuts are not yet excluded for the success predictor.
- **Exact claim allowed:** "The preference signal is relational and order-sensitive (not a pointwise quality readout, not a constant-output artifact, not a pair-order artifact). Trivial-shortcut controls for the branch-success predictor are not yet complete."
- **Missing controls:** P3-B shortcut baselines; P3-C task-grouped heldout.

### P4 — Utility
- **Status:** PARTIAL — **weak utility supported, strong (capability) utility NOT proven**.
- **Supporting artifacts:** `bg_dualanchor_perturbation_lift_v1_2026-05-31/perturbation_lift.md`; `docs/evaluator/current-state.md` (v3 architecture-looped metrics, Experiment 1/H); `artifacts/reports/probes/mpn_s1_baseline_2026-06-13/s1_4b_kmatched_sampling.json`; CoreContent v2 refit.
- **Key numbers (weak utility, supported):** survivor-set retention — stage oracle retention 0.985, terminal oracle retained 1.0 (v3 48-task); false-prune recovery 8/8; perturbation oracle retained 0.95, clean never sole oracle. Trajectory top-1 lift +0.162. CoreContent_v2 in-dist 0.6691 [0.645–0.690] > MIX_HH 0.5525 within real DualAnchor top-5 survivor sets; Experiment 1/H: CoreContent 0.658 > MIX_HH 0.552 > DualAnchor forced-top1 0.379 (oracle retention 1.0) → selection, not survival, is the terminal bottleneck.
- **Key numbers (strong utility, NOT proven):** forced-terminal top-1 perturbed rate 0.799 vs count-share 0.875 → count-normalized lift **−0.076**; S1 K-matched: greedy fork 0 new-correct, plain-sampling oracle 0.75 ≥ sample-fork oracle 0.611 (`fork_adds_over_sampling=false`); bounded LoRA training net reachability flat (macro 0.708→0.688).
- **Caveats:** forced top-1 is a *diagnostic*; survivor-set retention is the correct headline. CoreContent's in-dist edge is ~half constructed-negative artifact (real-negative edge +0.063) and drops to ~0.417 on generated/OOD branches.
- **Exact claim allowed:** "The readable signal is useful for external ranking, retention, survivor-set pruning, and diagnosis (weak utility). It has not been shown to produce end-to-end capability gains over plain sampling (strong utility unproven)."
- **Missing controls:** none required for the *weak* utility claim; the strong claim is a training-time question (S3A) out of scope here.

### P5 — Role separation
- **Status:** STRONG.
- **Supporting artifacts:** `artifacts/reports/probes/mpn_s3_closure_2026-06-17/s3_closure_and_sokac_validity_bundle.md`; `docs/evaluator/{content-selection-taps.md,dualanchor-architecture-baseline.md,current-state.md}`; `s1_4_gate3_prune_integration.json`.
- **Key numbers:** DualAnchor confirmed for survival (`DUALANCHOR_SURVIVAL_CONFIRMED`) but forced-top1 selection weak (0.379); CoreContent for content selection (0.658) but content→correctness transfer fails OOD (0.417); S3B-1 existing taps near-chance on generated-branch correctness. Gate-3 shows DualAnchor (6ch) and CoreContent (3ch) produce finite, varying, *distinct* per-branch scores.
- **Caveats:** none material; this separation is the project's most robustly established result.
- **Exact claim allowed:** "Validity/survivability (DualAnchor), relational content quality (CoreContent), and correctness (external verifier) are empirically and conceptually distinct axes; tap scores are not correctness labels and were never used as such."
- **Missing controls:** none.

### P6 — Cross-domain / generality
- **Status:** PARTIAL.
- **Supporting artifacts:** `bg_cross_domain_eval_matrix_2026-05-17.md`; `bg_fixed_config_cross_domain_audit_2026-05-17.md`; `docs/evaluator/current-state.md`.
- **Key numbers:** HH 0.855; CLEAN_GSM8K 0.778–0.893; CODE_STRICT_CLEAN_ALL16 0.875 (code-trained) / 0.750 (HH-trained); reasoning trajectory 0.854; science 0.811. `GENERALIST_SPECIALIST_VERDICT = DOMAIN_SPECIALISTS_NEEDED`; `SHARED_COHERENCE_AXIS = weak`.
- **Caveats:** a single universal head is weak; HH→code transfer is weak; code-specific heads win on code; science is diagnostic-only; coding-correctness transfer fails.
- **Exact claim allowed:** "Readable process-quality signal is not limited to HH preference; it appears across objective domains (math, reasoning, code, science), but is best read by domain-specialized heads rather than one universal probe."
- **Missing controls:** none required for the partial claim; broader generality would need more domains/heads (out of scope).

### P7 — Causal / control boundary
- **Status:** STRONG (as a *boundary* result).
- **Supporting artifacts:** `docs/evaluator/steering-and-adapters.md`; `s1_4b_kmatched_sampling.json`; S3 closure bundle.
- **Key numbers:** `BG_SEQUENCE_LEVEL_ADAPTER_VERDICT = NO_FROZEN_BACKBONE_WRITE_PATH`; `FROZEN_BACKBONE_INFERENCE_STEERING_STATUS = CLOSED_UNDER_TESTED_METHODS`; frozen fork closed (greedy fork 0 new-correct).
- **Caveats:** "closed under tested methods" is a bounded negative, not a proof of impossibility; training-time integration (S3A) is untested.
- **Exact claim allowed:** "Hidden states are readable (readout exists), but no tested frozen-backbone steering/adapter produced reliable write-control — proto-introspective *readout* and autonomous *control* are different problems; control requires training-time integration."
- **Missing controls:** none for the boundary claim.

---

## 4. Existing Evidence Inventory

| Artifact | Report | JSON | Script | Claim supported | Load-bearing? | Exists? |
|---|---|---|---|---|---|---|
| HH pairwise evaluator / loop geometry | — | `artifacts/reports/evaluator/probe_loop_geometry_hh.json` | (evaluator src) | P1, P3 (preference acc, antisymmetry, cross-backbone) | YES | YES |
| Evaluator head checkpoint | — | `artifacts/checkpoints/evaluator/pairwise_epoch2.pt` | — | P1 (the trained readout) | YES | YES |
| Flip-test interpretation | `docs/evaluator/flip-test-interpretation.md` | — | — | P3 (relational vs pointwise; antisymmetry; bias) | YES | YES |
| L1 ablation (loop localization) | — | `artifacts/reports/evaluator/probe_l1_ablation.json` | `utilities/tests/manual/probe_l1_ablation.py` (deleted from worktree; report retained) | P2 (L1 carries signal), P3 (swap-balanced) | YES | JSON YES; script MISSING |
| L1 alpha sweep | — | `artifacts/reports/evaluator/probe_l1_alpha_sweep.json` | `utilities/tests/manual/probe_l1_alpha_sweep.py` (deleted) | P1/P2 (fusion is not about mixing weight) | auxiliary | JSON YES; script MISSING |
| Bipartite layer probe | — | `artifacts/reports/evaluator/probe_bipartite_layers_24_36.json` | — | P2 (L24/36 converged, L47 bipartite) | YES | YES |
| Trajectory prediction suite | `bg_trajectory_prediction_2026-05-18/predictive_power.md` | `.../summary.json` | `utilities/tests/manual/*bg_trajectory*` | P1, P2 (prefix→success) | YES | YES |
| Trajectory prefix features | `.../prefix_features.md` | `.../prefix_features.pt`, `.../prefix_scores.json` | — | P2 (re-usable for pre-answer control) | YES (for controls) | YES |
| Cross-domain eval matrix | `bg_cross_domain_eval_matrix_2026-05-17.md` | `.../.json` | — | P6 | YES | YES |
| Fixed-config cross-domain audit | `bg_fixed_config_cross_domain_audit_2026-05-17.md` | `.../.json` | — | P6 (specialists needed) | YES | YES |
| DualAnchor perturbation lift | `bg_dualanchor_perturbation_lift_v1_2026-05-31/perturbation_lift.md` | `.../perturbation_lift.json` | — | P4 (count-normalized utility), P5 | YES | YES |
| DualAnchor architecture-looped v3 | `docs/evaluator/dualanchor-architecture-baseline.md` | (probe dir) | — | P4 (survivor retention), P5 | YES | YES |
| CoreContent v2 refit | `docs/evaluator/corecontent-dataset-expansion-v2.md` | `bg_corecontent_dataset_expansion_refit_v2_2026-06-04/` | — | P4, P5 (content axis, OOD break) | YES | YES |
| S1 K-matched sampling | — | `mpn_s1_baseline_2026-06-13/s1_4b_kmatched_sampling.json` | — | P4 (strong utility not proven), P7 | YES | YES |
| Steering / adapters | `docs/evaluator/steering-and-adapters.md` | (steering probe dirs) | — | P7 (no write path) | YES | YES |
| S3 closure + Šokac bundle | `mpn_s3_closure_2026-06-17/s3_closure_and_sokac_validity_bundle.md` | `.../s3_closure_verdicts.json` | — | P4, P5, P7 (engineering closure) | YES | YES |

**MISSING / partial:**
- `MISSING: utilities/tests/manual/probe_l1_ablation.py and probe_l1_alpha_sweep.py — consequence: the L1/loop-localization and alpha-sweep result JSONs survive and are load-bearing, but the generating scripts were removed from the worktree; re-deriving these exact numbers would require reconstructing the scripts from git history. The reported JSON values stand; reproducibility of the script path is weakened, not the result.`
- `MISSING: dedicated pre-answer-leakage control artifact — consequence: the P2 timing claim cannot yet be stated as airtight; this is the single highest-value missing control (see §11-A).`
- `MISSING: shortcut-baseline comparison for the trajectory predictor — consequence: P3 cannot exclude length/logprob/prompt-family shortcuts for branch-success prediction (see §11-B).`
- `MISSING: Barbados loop-state success/failure predictability — consequence: no small-looped-transformer cross-check of the introspection claim exists; this is optional and must NOT be invented (see §11-D).`

---

## 5. Prediction Evidence

- **What hidden-state features predicted quality/success?** Pooled per-loop hidden states at tap layers {24, 36, 47}, each `[loops × 2048]`, read either as a single loop slot or pooled. For preference: ordered-pair difference of pooled chosen/rejected states fed through a no-affine LayerNorm → linear projection → GRU over loops → scalar. For branch success: a tiny antisymmetric head over prefix-position features `[3 layers × 4 loops × 2048]`.
- **Which layers/loops were strongest?** Preference signal is readable from every loop including L1 (L1-only canonical 0.915). For trajectory success the strong cells concentrate at configs `36_L4`, `36_mean`, `24_L4` and at prefix 256 for reasoning; layer 47 carries the loop-distinct (bipartite) geometry while L24/L36 are loop-converged.
- **Was the predictor tiny/linear/pairwise?** Yes — the load-bearing predictors are `AntisymLinear` (LayerNorm(no affine)(left−right)→Linear(no bias)) or `AntisymLinearNoNorm` (Linear(left−right)); the only non-linear element is the evaluator's GRU over the 4 loops. All are tiny relative to the 2.6B backbone.
- **Was Ouro frozen?** Yes. All P1 evidence uses a frozen backbone; only the small readout heads are trained. RLTT did not move loop-boundary geometry (cross-backbone mean cos 0.999, score Pearson 0.991, decision agreement 0.995 between Thinking and RLTT).
- **What was the baseline?** Preference: random 0.50; pointwise-linear 21.75% (below chance) and pointwise-nonlinear ~65% are the *informative* baselines showing relational access dominates. Trajectory: random top-1 expected (e.g., 0.6875 in the best cell) — the claim is the *lift* (+0.162), not the raw rate.
- **Did prediction generalize beyond HH?** Yes, partially: GSM8K 0.778–0.893, code strict-clean up to 0.875, reasoning 0.854, science 0.811 — but with domain-specialized heads (P6).

**Linear probe vs evaluator vs taps:**
- **Linear probe** = one learned direction over pooled pair-difference vectors (HH pairwise linear diff probe 84.5%).
- **Evaluator** = richer relational readout: attention-pooled chosen/rejected per loop → differences → GRU aggregation over loops → scalar (95.2%).
- **Taps** = tiny specialized antisymmetric readouts at known {layer, loop} states, used for branch ranking/survival (the trajectory and DualAnchor/CoreContent heads).

---

## 6. Timing / Leakage Evidence

- **Results that use prefix/intermediate/loop states:** trajectory prediction (prefix positions 32/64/128/256, features captured at the prefix point, predicting continuations not yet generated); L1 ablation (first-loop-only readout, canonical 0.915); loop-geometry (per-loop states L1–L4).
- **Results that may still include final-answer leakage:** the longer trajectory prefixes (128/256) on short tasks (GSM8K, MCQ reasoning/science) can already contain the answer; the HH preference evaluator reads states over prompt+*full* response, so it is post-answer for the preference axis.
- **Results that are pre-answer or partially pre-answer:** short prefixes (32/64) precede most of the answer for these tasks and still yield positive top-1 lift (reasoning@64 +0.262, science@32 +0.188); L1-only readout shows signal at the first loop. This is *partial* pre-answer evidence.
- **Exact control still needed:** capture/relabel hidden states at a strictly-pre-answer cut (no answer token emitted yet) and confirm pairwise accuracy and top-1 lift remain above the random and shortcut baselines. The existing `prefix_features.pt` + `continued_prefixes`/`prefix_scores` largely permit this as a CPU re-analysis at the shortest prefixes.

**Required verdict:**

> **TIMING_SIGNAL_PARTIAL_PREFIX_EVIDENCE**

(We do not claim TIMING_SIGNAL_STRONG_PREANSWER_CONTROLLED because the dedicated strictly-pre-answer control has not been run; we do not claim TIMING_SIGNAL_NOT_YET_CONTROLLED because short-prefix lift and first-loop readout provide real partial pre-answer evidence.)

---

## 7. Specificity / Shortcut Controls

| Control | Status | Evidence |
|---|---|---|
| Pair order / flip antisymmetry | **PASS** | ρ≈−0.94; antisymmetric construction; swap-balanced L1-ablation protocol. |
| Pointwise vs pairwise | **PASS** | pairwise 95.2% vs pointwise-nonlinear ~65% vs pointwise-linear 21.75% (inverted). |
| Train/test split | **PARTIAL** | HH uses the standard train/test split; trajectory uses fixed task suites with small n and no task-grouped CV yet. |
| Prompt-family leakage | **MISSING** | no explicit prompt-family heldout for the trajectory predictor (see §11-C). |
| Length / style shortcuts | **MISSING** | no length / answer-token-count baseline run as a comparator (see §11-B). |
| Domain shortcuts | **PARTIAL** | per-domain results are reported separately, but no within-task domain-shortcut ablation for the success predictor. |
| Candidate-source shortcuts | **PARTIAL** | DualAnchor perturbation analysis is count-normalized against candidate base rate (good), but the trajectory predictor has no candidate-source control. |
| Logprob / entropy baseline | **MISSING** | hidden-state probe never compared against a model-logprob/entropy success predictor (see §11-B). |
| Raw-text baseline | **MISSING** | no simple text-feature baseline for branch success. |
| Verifier / gold label separation | **PASS** | all labels from external verifiers / HH chosen-rejected / answer keys; tap scores never used as labels (P5). |

**Specificity verdict:** SHORTCUT_CONTROLS_PARTIAL — strong on the relational/antisymmetry/label-separation axes, incomplete on trivial-feature (length/logprob/prompt-family) baselines for the branch-success predictor.

---

## 8. Utility Evidence

**Weak utility (SUPPORTED):**
- Hidden-state taps rank/retain/diagnose candidates: trajectory top-1 lift +0.162; survivor-set retention — stage oracle 0.985, terminal oracle retained 1.0 (v3), perturbation oracle retained 0.95; false-prune recovery 8/8.
- Branch loop can consume tap scores live (Gate-3: DualAnchor 6ch + CoreContent 3ch produce finite, varying, distinct per-branch scores during a live encode→group→score pass).
- CoreContent_v2 in-dist content selection 0.6691 > MIX_HH 0.5525; within survivor top-5, CoreContent 0.658 > MIX_HH 0.552 > DualAnchor forced-top1 0.379.

**Strong utility (NOT PROVEN):**
- End-to-end capability gain over sampling is not demonstrated: forced-terminal top-1 count-normalized lift −0.076; S1 K-matched shows fork adds nothing beyond stochastic decoding (`fork_adds_over_sampling=false`, greedy fork 0 new-correct, plain sampling 0.75 ≥ fork 0.611); bounded LoRA net reachability flat.

**Utility verdict:** WEAK_UTILITY_SUPPORTED — STRONG_CAPABILITY_UTILITY_NOT_PROVEN. The two must not be collapsed; the strong utility question is a training-time (S3A) lever, out of scope for this proto-introspection package.

---

## 9. Role Separation (corrected S3B-1 interpretation)

- **DualAnchor = validity / survivability**, NOT correctness. Confirmed for survival; its forced-top1 *selection* is weak (0.379) and its count-normalized perturbation lift is ≈0 — it keeps good candidates alive, it does not rank correctness.
- **CoreContent = relational content quality**, NOT correctness. In-dist 0.6691; content→correctness transfer fails OOD (0.417); coding tap is a corruption detector, not a wrong-answer detector.
- **Verifier / gold = correctness truth.** All correctness labels come from external verifiers, unit tests, answer keys, HH chosen/rejected.
- **Branch-correctness selector = a separate future axis** (S3B-2), not yet built; S3B-1 shows existing taps are near-chance on generated-branch correctness.
- **Forced top-1 = diagnostic only**, unless an architecture explicitly commits to it.
- **Survivor-set retention = the proper metric** for pipeline survival tests (and it is strong: 0.95–1.0).

This separation is the project's most robust result and underwrites the proto-introspection claim's *honesty*: we read process quality / validity / preference, and we explicitly do **not** equate those with correctness.

---

## 10. Causal / Control Boundary

- Steering / adapters / layer hooks did **not** provide reliable frozen-backbone write control: `BG_SEQUENCE_LEVEL_ADAPTER_VERDICT = NO_FROZEN_BACKBONE_WRITE_PATH`, `FROZEN_BACKBONE_INFERENCE_STEERING_STATUS = CLOSED_UNDER_TESTED_METHODS`; the frozen fork is closed (greedy fork 0 new-correct).
- This does **not** refute proto-introspection. Readability of a signal and the ability to act on it via static intervention are different problems.
- It shows readout and control are different: the information is present and recoverable, but the frozen backbone does not provide a corridor to use it for autonomous self-direction.
- Strong self-control would require **training-time integration** (S3A branch-tournament RLTT continuation), which is explicitly the next, separate lever and is not claimed here.

**Control-boundary verdict:** PROTO_INTROSPECTION_READOUT_SUPPORTED_CONTROL_NOT_SOLVED.

---

## 11. Minimal Remaining Controls

Only the smallest necessary controls are listed. These are not a campaign; the first three are mostly CPU re-analyses of already-captured features.

### A. Pre-answer leakage audit (HIGHEST VALUE)
- **Exact question:** does branch-success prediction survive when the readout sees *only* hidden states captured strictly before any answer token is emitted?
- **Input artifacts:** `bg_trajectory_prediction_2026-05-18/{prefix_features.pt, prefix_scores.json, continued_prefixes.json, task_suite.json}`; possibly a tiny re-capture at a defined pre-answer cut.
- **GPU needed?** Mostly no — re-analysis of existing prefix features at the shortest (32/64) prefixes with an explicit "answer-not-yet-emitted" filter is CPU. A small GPU re-capture (tens of minutes) is only needed if a cleaner pre-answer cut than prefix-32 is required.
- **Estimated cost:** CPU hours, or <1 GPU-hour if re-capture is needed.
- **Pass/fail:** PASS if pairwise acc and top-1 lift stay above random and the shortcut baselines (B) at the strictly-pre-answer cut.
- **Effect on claim:** converts P2 from PARTIAL → STRONG; upgrades the timing verdict to STRONG_PREANSWER_CONTROLLED.

### B. Shortcut baseline audit
- **Exact question:** does the hidden-state probe beat simple baselines — answer/branch length, domain, prompt family, candidate source, and (if available) model logprob/entropy?
- **Input artifacts:** the trajectory features + scores + task suite; logprobs from the existing continuations if logged, else a tiny re-score.
- **GPU needed?** No for length/domain/prompt-family/text baselines; possibly a tiny GPU re-score for logprob/entropy if not already logged.
- **Estimated cost:** CPU hours (+ optional <1 GPU-hour for logprob).
- **Pass/fail:** PASS if the hidden-state probe's lift exceeds every simple baseline's lift.
- **Effect on claim:** converts P3 from PARTIAL → STRONG (or, on failure, honestly downgrades the prediction claim).

### C. Grouped / task-heldout audit
- **Exact question:** does prediction survive when train/test are split by task / prompt family (no task appears in both)?
- **Input artifacts:** `prefix_scores.json` + `task_suite.json` (task_id is available, e.g., `ARC-Challenge/k`, `gsm8k/k`, `mmlu/.../k`).
- **GPU needed?** No — grouped cross-validation re-fit of the tiny head on existing features.
- **Estimated cost:** CPU minutes–hours.
- **Pass/fail:** PASS if pairwise accuracy stays above chance under task-grouped heldout.
- **Effect on claim:** strengthens generalization for P1/P3/P6.

### D. Optional Barbados replication
- **Exact question:** do loop states of the small Barbados looped transformer predict final exactness / boundary failure / repair?
- **Input artifacts:** the Barbados modular-expr replication harness (capability-replication only at present; **no loop-state introspection result currently exists** — do not invent one).
- **GPU needed?** Yes (small).
- **Estimated cost:** modest GPU; only worth it if Barbados is already instrumented to dump loop states.
- **Pass/fail:** PASS if loop-state probe predicts final exactness above chance.
- **Effect on claim:** would add a second-architecture cross-check (nice-to-have, not gating). **Optional.**

---

## 12. Paper Claim Readiness

| Claim | Readiness | Evidence | Caveat | Wording allowed |
|---|---|---|---|---|
| Weak proto-introspection exists in Ouro-RLTT hidden states | **READY_WITH_CAVEAT** | P1 STRONG, P2 PARTIAL, P5 STRONG | timing not airtight until §11-A | "Looped-model hidden states expose readable process-quality signals about the model's own ongoing computation." |
| Signal predicts quality/success | **READY** | 95.2% preference; trajectory pair acc 0.854 / +0.162 | small n in trajectory cells | "A tiny frozen-backbone readout predicts preference ordering and eventual branch success." |
| Signal appears in intermediate loop states | **READY_WITH_CAVEAT** | L1-only 0.915; prefix 32/64 lift; loop-geometry | converged layers carry less loop-distinct info; pre-answer control pending | "Predictive signal is present in intermediate loop and partial-trajectory states, including short pre-answer prefixes." |
| Signal is not merely a pair-order artifact | **READY** | ρ≈−0.94; swap-balanced; pointwise vs pairwise | — | "The signal is relational and order-sensitive, not a pair-order or constant-output artifact." |
| Signal is not fully explained by trivial shortcuts | **NOT_READY** | antisymmetry controls only | length/logprob/prompt-family baselines missing | (do not assert until §11-B passes) |
| Signal is useful for external readout/scaffolding | **READY** | retention 0.95–1.0; top-1 lift; live tap consumption | weak utility only | "The signal is useful for external ranking, retention, and diagnosis." |
| Signal does not yet imply autonomous self-control | **READY** | no frozen write path; fork closed | bounded negative, not impossibility | "Readout exists; autonomous self-control is not solved under tested methods." |
| Training-time integration is needed for control | **READY_WITH_CAVEAT** | control boundary + S3A design | S3A not yet run | "Acting on the signal autonomously appears to require training-time integration." |

---

## 13. Final Verdict Constants

```
PROTO_INTROSPECTION_WEAK_CLAIM_VERDICT = SUBSTANTIALLY_SUPPORTED_REQUIRES_TARGETED_CONTROLS
PREDICTION_EVIDENCE_VERDICT            = HIDDEN_STATES_PREDICT_PROCESS_QUALITY
TIMING_EVIDENCE_VERDICT                = TIMING_SIGNAL_PARTIAL_PREFIX_EVIDENCE
SPECIFICITY_EVIDENCE_VERDICT           = SHORTCUT_CONTROLS_PARTIAL
UTILITY_EVIDENCE_VERDICT               = WEAK_UTILITY_SUPPORTED_STRONG_CAPABILITY_UTILITY_NOT_PROVEN
ROLE_SEPARATION_VERDICT                = VALIDITY_CONTENT_CORRECTNESS_SEPARATED
CONTROL_BOUNDARY_VERDICT               = READOUT_SUPPORTED_CONTROL_NOT_SOLVED
NEXT_STEP_VERDICT                      = RUN_MINIMAL_PREFIX_AND_SHORTCUT_CONTROLS_THEN_WRITE_PAPER
```

Pillar status rollup: P1 STRONG · P2 PARTIAL · P3 PARTIAL · P4 PARTIAL(weak-yes/strong-no) · P5 STRONG · P6 PARTIAL · P7 STRONG.

---

## 14. Final Recommendation

- **Is the weak proto-introspection paper claim ready now?** Yes, **with caveats**. The core claim ("looped-model hidden states expose readable process-quality signals about the model's own ongoing computation") is supported by strong prediction, role-separation, and control-boundary evidence, plus partial timing and specificity evidence. It is honestly stated as *weak* proto-introspection with two named open controls.
- **What exact controls should be run before writing the timing and specificity claims as airtight?** §11-A (pre-answer leakage), §11-B (shortcut baselines), §11-C (task-grouped heldout). These are mostly CPU re-analyses of already-captured trajectory features and can run in parallel with drafting.
- **What can be written immediately?** The prediction, role-separation, and control-boundary sections; the operational definition; the relational/antisymmetry specificity controls; the weak-utility (retention/ranking) results; and the explicit boundary that control is not solved. The narrative that *readout precedes/differs from control* is fully supported now.
- **What should NOT be claimed?** Consciousness; self-awareness; verbal self-understanding; autonomous internal use of the signal for self-control; end-to-end capability/Jormungandr improvement; airtight pre-answer timing (until §11-A); shortcut-free specificity (until §11-B). Do not use tap scores as correctness labels; do not conflate validity, content, and correctness.

**Bottom line:** `NEXT_STEP_VERDICT = RUN_MINIMAL_PREFIX_AND_SHORTCUT_CONTROLS_THEN_WRITE_PAPER` — paper writing can begin **in parallel** with the three small controls; no new expensive GPU experiments are required for the first draft.
