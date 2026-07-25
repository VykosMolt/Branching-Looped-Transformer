# Complete Ouro-RLTT / BG / Proto-Introspection Project Handoff for Claude

**Prepared for:** Claude or another external research assistant/model.  
**Prepared by:** ChatGPT from the current project conversation, preserved project memory, and currently mounted project documents.  
**Date prepared:** 2026-07-01.  
**Purpose:** Give Claude the missing context for the Ouro-RLTT hidden-state evaluator / BG / branch-control / proto-introspection project without requiring it to reconstruct the chronology from scattered run notes.

---

## 0. Accuracy / provenance / anti-fabrication note

This file deliberately separates what is known from what should be re-verified.

**I am not inventing new results here.** The numerical claims below come from one of three places:

1. The currently mounted source documents in `/mnt/data`, especially:
   - `paper.pdf`
   - `README_ouro_rltt_evaluator_project.md`
   - `current-state.md`
   - `chronological-evaluator-summary.md`
   - `content-selection-taps.md`
   - `corecontent-dataset-expansion-v2.md`
   - `core-domain-tap-audit.md`
   - `domain-transfer-ledger.md`
   - `dualanchor-tap-evolution.md`
   - `branch-generation-and-survival.md`
   - `branch-training-logic-expansion.md`
   - `kv-cache-branch-carry.md`
   - `steering-and-adapters.md`
   - `evaluator-locus-summary.md`
   - `flip-test-interpretation.md`
   - `evaluator_pairwise.py`
2. Stable project memory from the long-running conversation.
3. User-reported final deliverable summaries for the proto-introspection controls and paper-writing package, where the user stated the numbers were verified against source artifacts.

Some older uploaded files have expired in the ChatGPT environment. This handoff is therefore not a literal full repository dump. It is a consolidated memory/context handoff. If Claude has access to the local repository, it should verify exact artifact paths and numbers before using them in a paper.

**Claim discipline:** do not convert “supported,” “validated in test harness,” “partial,” “diagnostic,” or “local” into “proven,” “solved,” “production-ready,” or “capability-improving.” The project has both strong positive readout results and important negative control/steering results.

---

## 1. One-paragraph identity of the project

This project studies whether a frozen looped language model, **Ouro-RLTT / Ouro-2.6B-Thinking**, exposes readable information in its intermediate hidden states about the quality, viability, preference, or likely success of its own ongoing computation. It began as a relational HH-RLHF pairwise evaluator over hidden states, then became a broader BG-style hidden-state tap/controller project: tiny pairwise taps, domain-transfer audits, DualAnchor branch-survival policies, CoreContent terminal/content selection, trajectory prediction, frozen steering closure, hidden-origin branch generation, KV/cache branch-carry, live branch/carry/prune/loop-back scaffolds, branch-training datasets, and now a narrow **operational proto-introspection** paper. The current paper is not merely “pre-answer probing” and not merely an S3A compute proposal. It is a systems + interpretability paper about building and validating a hidden-state readout/control-signal stack in a looped LM, then drawing the boundary between readable process-quality state and trained autonomous control.

---

## 2. Core thesis and current paper identity

### 2.1 The current thesis

The current paper thesis should be something like:

> In Ouro-RLTT, intermediate hidden states expose a family of readable process-quality signals. These signals support relational preference prediction, domain-transfer quality readouts, role-specialized taps for content and branch survivability, and strict pre-answer success prediction. They can be consumed by a live internal branch/carry/prune scaffold, but they do not by themselves yield autonomous control or capability gains. This supports a narrow operational notion of proto-introspection: readable internal information about the model’s own ongoing computation, not consciousness or self-awareness.

### 2.2 What the paper is **not**

It is not just:

- a reward-model paper;
- a probe-only paper;
- a pre-answer GSM8K audit paper;
- a philosophical consciousness paper;
- a “Jormungandr works” paper;
- an S3A compute request pretending to be a result;
- a branch-carry engineering appendix with no interpretability claim.

### 2.3 What the paper **is**

It is a paper about a complete stack:

1. Relational hidden-state preference/quality readouts.
2. Tiny layer/loop taps.
3. Domain transfer and role specialization.
4. DualAnchor survival versus CoreContent terminal/content selection.
5. Live internal branch/carry/prune substrate.
6. Frozen steering/branching negative results.
7. Strict pre-answer process-quality audit.
8. Synthesis into weak operational proto-introspection.
9. Future path: S3B branch-correctness selector and S3A branch-tournament RLTT/training-time integration.

---

## 3. Operational definition of proto-introspection

Use the word “proto-introspection” only operationally.

Recommended definition:

> An intermediate model state is proto-introspective if it contains readable information about the quality, stability, uncertainty, likely success/failure, or branch viability of the model’s own ongoing computation before external final judgment.

This does **not** imply:

- consciousness;
- self-awareness;
- subjective experience;
- explicit self-understanding;
- human-like metacognition;
- autonomous self-control;
- that the model internally uses the signal;
- that branching/Jormungandr already improves capability.

The empirical claim is about **readable process-quality information** inside the model’s ongoing computation.

---

## 4. User / project context relevant to Claude

- The user is Johann / Jan, a Software Engineering student in Croatia, working independently.
- The user wants critical, non-sycophantic research support. Do not be vague, do not flatter, and do not rewrite the whole project unless asked.
- The local machine is constrained: RTX 5070 Ti with 12GB VRAM. This is enough for extraction, taps, small LoRA pilots, bounded recaptures, and analysis, but not for serious 2.6B backbone continuation/training.
- The local project folder is generally `/home/moloch/ouro_project`.
- The user has repeatedly emphasized that S3A/backbone-scale training requires outside compute/collaboration.
- The paper is being written now. The current draft should treat the engineering stack as core main-body material.
- External contact update: Arnau Padres replied positively at a high level, said the direction is interesting/promising and looped models look natural for branching, but MELT access would require internal legal processes and is unlikely before a paper is accepted.

---

## 5. Naming / conceptual map

### 5.1 Ouro-RLTT / Ouro-2.6B-Thinking

A looped/recurrent transformer used as the frozen backbone. In documents it is often referred to as Ouro-RLTT or Ouro-2.6B-Thinking. It produces multiple internal loop states. Hidden states are captured from layers/loops.

Known feature shapes:

- Hidden dimension: `2048`.
- Common compact feature tensor: `[layers=3, loops=4, hidden=2048]`, usually layers `24`, `36`, `47` and loops `L1-L4`.
- Ouro has 4 loops × 48 layers = 192 UniversalTransformerCache slots.

### 5.2 BG

“BG” is a basal-ganglia-inspired label for branch selection / controller readouts. It is not a separate full model. It is a hidden-state tap/controller line.

### 5.3 Tap

A tiny readout head over hidden-state features. Usually pairwise/relational. Often an antisymmetric or low-capacity head over `feature_A - feature_B`.

### 5.4 DualAnchor

A branch-survival / validity / survivability policy. It should not be called a correctness selector.

Current DualAnchor baseline:

- `MIX_CODE_REASONING + MIX_OBJECTIVE_ALL`.
- Schedule: `L1_24 -> L1_36 -> L1_47 -> L2_24 -> L2_36 -> L2_47 -> L3_24 -> L3_36 -> L3_47 -> L4_24 -> L4_36 -> terminal L4_47`.
- Threshold: `mean_floor_very_loose`.
- Hard budget: `8`.
- Terminal policy: confidence-gated top1 if confident; otherwise defer/keep terminal survivors.

### 5.5 CoreContent

Content / final-choice selection tap. It ranks within an already-handed-off candidate/survivor set. It is not branch survival and not a universal correctness oracle.

Current selected content selector:

- `CoreContent_v2_blockwise_pruned_24_36` / `CoreContent_v2_blockwise` depending artifact wording.
- Fallback: `mixedhead_MIX_HH_OBJECTIVE`.

### 5.6 S1 / S3 / S3A / S3B

These labels are internal engineering phases, not the same thing as the professor/Šokac validity package.

- **S1**: mechanically validate frozen branch/carry/prune/loop-back / internal branch substrate. Result: mechanism works, frozen reachability gain null.
- **S3B / S3B-pre**: branch-correctness selector trained/refit on verifier-labeled generated branch candidate pools. This addresses the selection wall on generated branches.
- **S3A**: branch-tournament RLTT / training-time integration. Train loop dynamics/backbone/adapters at scale so internal branches become outcome-distinct and useful. This likely needs outside compute.
- **S3C**: future integrated comparison after S3A: base greedy, K-matched sampling, frozen S1, S3A-trained branch/carry, S3A+selector, etc.

### 5.7 Jormungandr / BLT

The broader future direction: trained branching looped transformers / internal branch selection/control. Current paper should motivate this but must not claim it works yet.

### 5.8 Barbados

Separate adjacent tiny looped-transformer proof-of-framework project. It is a “wind tunnel” for looped latent computation / Jormungandr-like ideas. Current evidence is modest and should not be mixed into the Ouro paper unless used as future work.

---

## 6. Chronology of the project

### 6.1 Initial relational preference encoding paper / evaluator

The original paper was titled **“Relational Preference Encoding in Looped Transformer Internal States.”** It used Ouro-2.6B-Thinking hidden loop states to predict HH-RLHF preferences.

Key verified paper/pdf numbers:

- Base model: Ouro-2.6B-Thinking, frozen.
- Trainable evaluator: about 5M parameters.
- Dataset: Anthropic HH-RLHF chosen/rejected pairs.
- Pairwise evaluator test accuracy: **95.2%** on **8,552** unseen examples.
- L-BFGS linear pairwise-difference probe: **84.5%**.
- Best nonlinear independent / pointwise evaluator: about **65%** test accuracy.
- Linear independent / pointwise classification: **21.75%**, below chance/inverted polarity.
- Epoch behavior: test accuracy rose from **83.3% epoch 1** to **95.2% epoch 2**, then degraded to **62.4% epoch 5** while misleading/deflated training metric increased.
- Flip/antisymmetry correlation stable roughly **ρ = −0.92 to −0.97** across epochs.
- Degenerate pairwise failure mode was discovered: constant output could yield 100% if protocol was flawed; flip test became mandatory.

Interpretation:

- The preference signal is strongly relational.
- Pairwise difference geometry is much more powerful than pointwise scoring.
- Scorer bias can cause low strict sign-flip rate even when relational component reverses properly.
- The evaluator is a strong fixed-order pairwise relational evaluator with scorer bias, not necessarily a perfectly zero-centered antisymmetric comparator.

### 6.2 Evaluator architecture lineage

The pairwise evaluator architecture in `evaluator_pairwise.py` includes:

- `AttentionPool` over token dimension.
- Pairwise differences between chosen and rejected pooled hidden states.
- `LayerNorm(hidden_dim, bias=False)` on differences.
- Linear projection to GRU hidden dimension.
- 2-layer GRU over loop-state sequence.
- Scorer on concatenation of GRU final state and final projected state.

The code comment says “architecture that reached 70%,” but the later paper/current project interpretation separates older architecture comments from the final pairwise result. The stable load-bearing result is the 95.2% pairwise evaluator in the paper and flip-test documents.

### 6.3 AntisymLinear / tiny tap pivot

After the evaluator, the project moved toward smaller taps:

- A tap reads frozen hidden features.
- Pairwise score often uses `feature_A - feature_B`.
- Exact/near-exact antisymmetric tiny heads often beat or matched bigger temporal/GRU approaches for practical branch selection.
- This made the claim stronger: if a tiny head works, the relevant geometry is already in Ouro’s states rather than being built by a large external network.

### 6.4 Clean GSM8K/code/reasoning transfer phase

Early transfer experiments had confounds (truncation, wrapper issues, dirty branch data). After repairs:

- Clean GSM8K and code pilots supported preliminary HH-trained tap transfer.
- Local-agent wrappers and unit-test labels became important for code.
- Same-task strict-clean correct-vs-near-miss code contrasts were identified as hard and needed domain-specific training.
- Reasoning branch pilots showed good transfer under natural distractors.

Example remembered/current-state numbers:

- Clean GSM8K expanded random top1 baseline around **0.563**; selected best rows in fixed config reached around **0.893 top1 / 0.796 pairwise** for a code-trained head at `24_mean` in one fixed-config audit.
- Reasoning pilot random baseline around **0.440**.
- Some reasoning subsets hit top1/pairwise **1.000 / 1.000** in small/pilot settings; treat as small-n diagnostic, not broad proof.

### 6.5 Cross-domain fixed-config audit / generalist-specialist verdict

The project tested whether HH-trained and code-trained taps transfer across:

- HH;
- clean GSM8K;
- code runnable diagnostics;
- strict-clean code;
- reasoning;
- science subsets.

Relevant verdicts from current-state:

- `BG_CROSS_DOMAIN_MATRIX_VERDICT = READY`
- `FIXED_CONFIG_AUDIT_VERDICT = READY`
- `GENERALIST_SPECIALIST_VERDICT = DOMAIN_SPECIALISTS_NEEDED`
- `REASONING_TRANSFER_VERDICT = GOOD`

Interpretation:

- Transfer exists and is practically useful.
- A single universal tap is not enough.
- Domain specialists and role-specific heads are needed.
- This is a role/distribution specificity story, not a collapse of the readout thesis.

### 6.6 Trajectory prediction sweep

The transformer-native trajectory-prediction sweep showed that BG scores over partial prefixes predicted which branches would later finish correctly.

Key remembered numbers from current-state / chronological summary:

- `BG_TRAJECTORY_PREDICTION_VERDICT = STRONG`.
- Best cell: `MIX_CODE_REASONING / 36_mean / AntisymLinear`, reasoning, 256-token prefix.
- Best cell lift / pairwise accuracy: **+0.1625 top-1 lift / 0.8537 pairwise**.
- User/project memory also says reasoning@256 oracle about **0.90** and 368 strong cells, but this should be verified before paper use.

Important caveat after later controls:

- Do not use the 0.854-ish trajectory number as the cleanest proto-introspection headline because older trajectory/prefix results may include leakage/task-shortcut inflation.
- Use it as supporting evidence.
- The cleanest anti-skeptic proto-introspection result is the later GSM8K strict pre-answer audit.

### 6.7 Steering closure (May 18)

The project tested whether hidden-state readout directions could become direct frozen write/control directions.

Major verdicts:

- `FROZEN_BACKBONE_INFERENCE_STEERING_STATUS = CLOSED_UNDER_TESTED_METHODS`.
- `BG_SEQUENCE_LEVEL_ADAPTER_VERDICT = NO_FROZEN_BACKBONE_WRITE_PATH`.
- Static steering directions did not provide reliable signed control.
- Perturbations propagate to later states/logits, but propagation is not useful signed behavioral control.
- Tiny causal adapter learned local teacher-forced logit control only; free-generation transfer was absent/weak.
- Sequence-level adapter improved sequence reward during training but no adapter-specific heldout free-generation transfer; worse than random in some comparisons.
- The strongest mechanistic interpretation: readout geometry, empirical-success geometry, and local logit-control geometry are distinct / orthogonal-ish.

Key adapter/steering verdicts from current-state:

- `BG_RMS_STEERING_VERDICT = RMS_UNSIGNED_ONLY`
- `BG_PROPAGATION_VERDICT = PROPAGATES_TO_LATER_STATES`
- `BG_PROPAGATION_DECAY_PROFILE = SURVIVES_32_TOKENS`
- `BG_INFERENCE_TIME_STEERING_VERDICT = UNSIGNED_ONLY`
- `BG_PHASE2_REQUIREMENT_VERDICT = TRAINING_REQUIRED`
- `BG_CAUSAL_ADAPTER_VERDICT = LOCAL_LOGIT_CONTROL_ONLY`
- `BG_SEQUENCE_ADAPTER_TRANSFER_VERDICT = NO_TRANSFER`

Interpretation:

- Readout exists.
- Mechanical writing into hidden states exists.
- But frozen inference-time steering/control does not solve the problem.
- This becomes part of the paper’s readout-control boundary.

### 6.8 Hidden-origin branch generation

The same-prefix hidden-origin branch suite tried to generate branch diversity from hidden-state perturbations/forks.

Relevant current-state verdicts:

- `PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_V3 = STILL_DATA_LIMITED`
- `PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_V4 = STILL_DATA_LIMITED`
- `HIDDEN_ORIGIN_BRANCH_GENERATOR_STATUS_V1 = WEAK_BUT_USABLE`
- `UNIVERSAL_BRANCH_CONTENT_TAP_STATUS = FUSION_NEEDED`

Interpretation:

- Same-prefix hidden-origin branches can be generated and sometimes produce different downstream outcomes.
- Frozen taps do not robustly select generated hidden-origin branches better than random without the right distribution/training.
- This pointed toward generated-branch-specific selectors and training-time integration.

### 6.9 DualAnchor evolution and branch-survival baseline

DualAnchor evolved from two-tap/anchor experiments into the active branch-survival policy.

Current locked candidate details:

- DualAnchor selector: `MIX_CODE_REASONING + MIX_OBJECTIVE_ALL`.
- Schedule: `L1_24 -> L1_36 -> L1_47 -> L2_24 -> L2_36 -> L2_47 -> L3_24 -> L3_36 -> L3_47 -> L4_24 -> L4_36 -> terminal L4_47`.
- Threshold: `mean_floor_very_loose`.
- Hard budget: `8`.
- Lineage logging required.
- Terminal confidence-gated top1, otherwise survivor handoff.

V3 metrics (48 tasks, 24 reasoning / 24 science):

| Metric | Value |
| --- | ---: |
| stage oracle retention | 0.9848 |
| terminal oracle retained | 1.0000 |
| forced terminal top1 oracle | 0.9167 |
| forced terminal top1 reward | 0.2625 |
| terminal best-survivor reward | 0.3167 |
| reward-diverse rate | 0.2292 |
| positive-oracle rate | 0.3542 |
| false-prune recovery | 8/8 |

Interpretation:

- Survival/pruning is strong.
- Terminal top1 decision is still a bottleneck.
- Terminal survivor-set handoff is the safe default.
- Science was partial/diagnostic; reasoning was headline-ready under locked terminal handoff.

### 6.10 Science / reasoning repair

Science was repeatedly problematic. Reasoning was stronger.

Current science/reasoning repair v3 summary:

- `MMLU_SCIENCE_BRANCH_PARSER_REPAIR_V3_STATUS = SCIENCE_PARTIALLY_REPAIRED`
- `BG_PRE_STEERING_DOMAIN_DECISION_V3_VERDICT = READY_FOR_STEERING_REASONING_PLUS_PARTIAL_SCIENCE`
- Reasoning: headline-ready.
- MMLU anatomy: partial/secondary candidate.
- Anatomy heldout positive-oracle **0.333**, selected-parser terminal-best **0.20**, parse **1.0 on 3 tasks**.
- Chemistry/physics/SciQ excluded or diagnostic; chem/physics parse collapsed to 0.0.
- Overall heldout science remained weak; tiny heldout caveats.

### 6.11 Core-domain tap audit

Core-domain tap audit added/confirmed:

- `CORE_DOMAIN_TAP_AUDIT_STATUS = CORE_TAPS_READY`
- `BG_CORE_PRE_STEERING_READINESS_VERDICT = READY_FOR_STEERING_CORE_DOMAINS`
- `BG_CORE_TAP_POLICY_SELECTION_VERDICT = KEEP_SCIENCE_DIAGNOSTIC_ONLY`
- Core domains ready: coding, reasoning, math, logic, alignment.
- Science/anatomy diagnostic-only.
- Tiny heads exactly antisymmetric.
- HH evaluator distinction preserved: roughly 62–65% pointwise vs 95.2% pairwise.

### 6.12 CoreContent v2 dataset expansion and refit

CoreContent v1 was data-limited. v2 expanded the dataset and fixed that.

Key numbers from `current-state.md` / `corecontent-dataset-expansion-v2.md`:

- Status: `V2_CORECONTENT_READY`.
- Original v1 reward-diverse coverage: alignment 200, logic 80, math 66, coding 30, reasoning 5.
- v2 expanded data:
  - alignment: ~25,993 / ~26,000;
  - coding: 1,733;
  - logic: 2,199;
  - math: 3,200;
  - reasoning: 2,600.
- Feature storage: **4.87 GB**, **64 shards**, **0 errors**.
- Heldout best: `CoreContent_v2_blockwise` = **0.6691** with CI about **[0.645, 0.690]**.
- Baseline `mixedhead_MIX_HH_OBJECTIVE` = **0.5525** with CI about **[0.526, 0.577]**.
- Edge: +0.117 on constructed heldout.
- Real-negative-domain edge (reasoning/logic/alignment): about **+0.063**.
- Coding tap is partly a corruption detector: about **0.94 vs mutants**, but only about **0.58 vs real wrong-problem code**.
- Relevance negatives did not fix coding relevance; relevance not linearly accessible in pooled L24/36/47.
- Layer 47 is dead weight for the selected pruned content tap.

Interpretation:

- CoreContent is a content/final-selection component.
- It is not branch survival.
- It is not a universal generated-branch correctness selector.
- It improved with scale, but has honest distribution caveats.

### 6.13 Branch training + logic expansion + terminal v1 (June 7)

This was the first step toward model-internal branching and S3 training.

Key status:

- `BRANCH_TRAINING_LOGIC_EXPANSION_STATUS = LOGIC_EXPANSION_READY_TRAINING_NOT_READY`
- `BRANCH_TRAINING_POLICY_DECISION = KEEP_EXTERNAL_DUALANCHOR_CORECONTENT_BASELINE`

Dataset / harness:

- Logic-expanded branch-training dataset.
- **48.5k tasks**, **10 verifier-backed families**, **33.2k train**.
- 5 training views.
- Correctness from external verifiers only.
- DualAnchor/CoreContent remain teachers/baselines.
- Science diagnostic-only.
- Steering not run.

Experiment H:

- Within real DualAnchor top-5 survivor sets:
  - CoreContent_v2: **0.658** top1 oracle.
  - MIX_HH: **0.552**.
  - DualAnchor forced-top1: **0.379**.
  - Oracle retention: **1.0**.
- Interpretation: selection, not survival, is terminal bottleneck.

Reachability @4:

- reasoning: **0.95**;
- math: **0.83**;
- logic: **0.73**;
- coding: **0.43**.

Important harness lesson:

- Math jumped **0.31 -> 0.83** after tool-free answer-forcing prompt + 1400-token budget + early-stop + LaTeX/SymPy verifier.
- This showed some apparent “model limitation” was actually harness/verifier/prompting limitation.

DualAnchor-as-teacher:

- Useful branch-policy teacher for coding/reasoning: retention lift over random about **+0.11**.
- Near-random on logic: about **+0.03**.
- Logic needs verifier reward, not teacher distillation.

Bounded training / LoRA proof-of-capability:

- 300-step bf16 LoRA on Ouro-RLTT, not converged.
- Behavior changed.
- Branch diversity: **+0.45**.
- Coding parse: **0.72 -> 0.94**.
- Math: **0.75 -> 0.92**.
- Macro reachability: **0.708 -> 0.688**, lift **−0.02**.
- Reasoning/logic regressed.
- External baseline remains locked.
- Teacher distillation M + verifier-reward RL N at scale are next separate run.

Interpretation:

- Local training can move behavior/diversity, but not enough to improve net reachability.
- This is proof-of-capability, not solved training.
- Serious S3A needs compute.

### 6.14 Offline verifier-backed branch generator v2 rounds 1–4 (June 13)

Goal: train useful branch generator offline-first before online RL. DualAnchor still prunes; CoreContent ranks.

Key policy:

- `OFFLINE_BRANCH_GENERATOR_POLICY = KEEP_EXTERNAL_DUALANCHOR_CORECONTENT_BASELINE`
- `K_CANARY_SELECTION_VERDICT = NO_ADAPTER_IMPROVES_ON_CANARY`
- Fixed canary: 610 groups (paired 30/domain + 110 alignment).

Rounds:

- r1 meta-text: macro **−0.244**. Rendered branches were strategy narrations, not solutions.
- r2 executed logic: **−0.052**. Logic repaired to 0.000; math/coding still bare.
- r3 rationale math + canonical code: **−0.112**. Math recovered to −0.133, but canonical-solution coding backfired to **−0.467**.
- r4 train/eval format alignment: **−0.059**. Root cause: training/eval rendering mismatch. v5 prompts made byte-identical to eval-side rendering.

Round 4 finding:

- First C-arm to clear degeneration alarm.
- Branch chars 0.70× base vs 0.26–0.40× for r1–r3.
- Reasoned-code data cut coding **−0.467 -> −0.067** at 0.92× length.
- All generation domains within ~2/30 of base; no collapse.
- The r1–r3 “training degrades generator” verdict was largely a format-mismatch + bad-coding-data confound.
- Corrected reading: offline SFT+DPO is reachability-neutral and non-degenerate, but does not add reachability where base is already strong (logic/math/reasoning 0.83–0.87).
- Real levers are low-base domains and verifier-reward RL, not more SFT on saturated domains.

### 6.15 Autoregressive KV/cache branch-carry validation (June 1)

This is a major technical substrate contribution.

Status v1:

- `AUTOREGRESSIVE_KV_BRANCH_CARRY_STATUS = PROMPT_INTERNAL_BRANCH_CACHE_VALID`
- `LEVEL6_PARTIAL_SPLICE_STATUS = PARTIAL_SPLICE_DIAGNOSTIC_ONLY`

UniversalTransformerCache:

- Slot = `current_ut * num_hidden_layers + layer_idx`.
- 4 loops × 48 layers = **192 slots**.

Validation ladder levels 0–5 all passed within bf16:

- L0 cached decode == full recompute: prefill bit-exact; decode RMS ~0.05–0.2 due bf16 drift.
- L1 token-boundary fork: K=2/4/8 independent branch caches, no cross-branch contamination.
- L2 batched branches == independent == full recompute.
- L3 prune/reorder survivors: examples 8→4→2, 8→3, 4→1 with aligned lineage.
- L4 current-token layer perturbation at L24/36/47, loop-targeted, carries via branch cache.
- L5 prompt-internal perturbation yields valid branch-specific cache; negative control RMS ≈ 3.0 confirms branch cache required.
- L6 v1 only diagnostic: boundary logic validated but no compute savings.

### 6.16 Partial cache splice v2 (June 1)

Status:

- `PARTIAL_CACHE_SPLICE_V2_STATUS = PARTIAL_SPLICE_COMPUTE_SAVING_VALID`

Key obstacle:

- UniversalTransformerCache stores K/V but not inter-layer residual stream.
- Solution: capture residual hidden at perturbation boundary during minimal shared-prefix prefill, apply additive boundary perturbation without a forward, recompute only suffix.

Validation:

- Spliced branch cache bit-exact vs full perturbed-prompt reference across all 192 slots.
- Prefill logits RMS 0; continuation bit-for-bit.
- Hook timing: perturbing a layer output leaves boundary slot unaffected; first affected slot is `(u, L+1)`; changed set == downstream-only theory.
- Validated single-branch, multi-branch (K=2/4), batched+prune/reorder, left-padded with explicit `position_ids`.

Compute savings:

- Per-branch layer-pass saving at layer 24: **13% / 38% / 63% / 88%** for boundary loops 0/1/2/3.
- At loop 2 layer 24, K-scaling gives **32% / 47% / 55% fewer passes** for K=2/4/8.
- Saving is amortized; needs K≥2.
- K=1 prefix+suffix == full.
- Copy-affected Mode A saves nothing.

Interpretation:

- Compute-saving branch-carry can be claimed in test harness, amortized and equivalence-validated.
- Production readiness cannot be claimed.
- No steering/training claim.

### 6.17 S1 branch/carry/prune/loop-back closure

S1 was the full frozen-model branch/carry/prune/loop-back mechanism.

Mechanically validated gates include:

- alpha=0 rederive identity: bit-exact, maxabs 0.0, 48/48 tokens.
- single-locus alpha>0 splice equivalence.
- two-locus alpha>0 chaining equivalence.
- live prune path: branch text → `[3,4,2048]` → DualAnchor/CoreContent scoring.
- structural lineage invariants.
- monotone 12-locus schedule.
- full 12-locus reference loop.

Reference run:

- 4 tasks.
- K=2.
- budget=4.
- alpha=0.02.
- last-token.
- greedy.
- Correctness preserving / zero loss.
- `oracle_over_survivors = base_acc = selected_acc = 0.25`.

Divergence examples:

- logic: 0/12 diverged, full collapse, wrong→wrong.
- math: 10/12 diverged, terminal spread 0, right→right reconverged.
- reasoning: 9/12 diverged, terminal spread ~2.31, wrong→wrong.
- coding: 11/12 diverged, spread ~0.81, wrong→wrong.

S1.4a fork parameter screen:

- 18 cells.
- Single-locus from clean root.
- loop-1 L24/L36/L47 + loop-4 sentinel.
- K=4.
- prompt+answer scoring.
- alpha ∈ {0.02, 0.05, 0.10}.
- token_range ∈ {last, last-8, second-half}.
- decode ∈ {greedy, sample(0.7/0.95)}.
- Greedy cells: no new correct; divergence/reconvergence only.
- Sample cells: new-correct@base_missed 0.5–0.75, loci_div 1.0, reconv 0.0, but nearly identical across alpha/token_range: sampling RNG dominates.

S1.4b K-matched sampling baseline:

- 12 plain samples/task, temp 0.7/top_p 0.95.
- Plain sampling oracle 0.75.
- Sample fork 0.611.
- Greedy fork 0 new-correct.
- Sampling explains fork+sampling gains.

Final verdict:

- `FROZEN_FORK_CLOSED` / frozen branch/carry locally reachability-neutral under tested regimes.

Interpretation:

- Mechanism works.
- Frozen injected branches do not produce reachability gain over sampling.
- This is a strong boundary result and justification for training-time integration, not a failure to hide.

### 6.18 S3B-0 refit sanity

Existing reusable assets:

- `data/corecontent_v2/features/*.pt`
- `data/branch_training_logic_expansion_v1/train_v*/branch_set_dpo.jsonl`
- `data/corecontent_v2/processed/candidate_groups_deduped.jsonl`
- math/code branch tournament JSONs
- training machinery `M.train_pairwise` / `M.train_listwise`

Results:

- ORACLE sel@oracle: **1.000**, regret 0, top4_ret 1.
- CoreContent_v2_blockwise frozen: **0.6691** sel@oracle, regret 0.331, top4_ret 0.997.
- S3B0_listwise refit: **0.6512**.
- S3B0_pairwise refit: **0.6399**.
- MIX_HH: **0.5526**.
- MIX_OBJECTIVE_ALL: **0.3833**.
- DualAnchor: **0.3787**.
- RANDOM: **0.2709**.

Interpretation:

- Training path works.
- Refit near polished CoreContent.
- S3B-0 sanity passes.
- Do not chase tiny 0.029 gap unless validation-only capped sweep.

### 6.19 S3B-1 generated branch pool transfer corrected interpretation

The first interpretation was flawed because it conflated validity/content/correctness and had aggregation bugs.

Corrected deliverables remembered:

- `artifacts/reports/probes/mpn_s3b_2026-06-17/s3b1_corrected_addendum_2026-06-17.md`
- `s3b1_loop_pool_transfer.json`

Verdict constants:

- `S3B1_DIRECT_CORRECTNESS_TRANSFER_VERDICT = EXISTING_TAPS_NEAR_CHANCE_ON_GENERATED_BRANCH_CORRECTNESS`
- `S3B1_DUALANCHOR_INTERPRETATION = NOT_EVALUATED_ON_PRIMARY_VALIDITY_ROLE`
- `S3B1_CORECONTENT_INTERPRETATION = CONTENT_TO_CORRECTNESS_TRANSFER_FAILS_ON_GENERATED_BRANCHES`
- `S3B1_PIPELINE_VERDICT = FULL_DUALANCHOR_TO_CORECONTENT_PIPELINE_NOT_TESTED`
- `S3B2_RECOMMENDATION = GENERATE_POWERED_BRANCH_POOL_DATASET_BEFORE_REFIT`

Corrected numbers:

- Best real selector MIX_HH sel@oracle: **0.667**.
- Random: **0.583**.
- Oracle: **1.0**.
- Separability for real taps: about **0.49–0.57**.
- Random separability: **0.46**.
- Oracle separability: **1.0**.
- CoreContent_v2 in-distribution: **0.6691** → generated transfer **0.417**.
- Generated transfer carried by math (1.0), collapses reasoning (0.25) and logic (0).
- Usable pools: 8 (math 2, reasoning 4, logic 2; coding 0/4 oracle-present).

Bugs fixed:

- Coding zero-oracle pools incorrectly averaged as 0.0, deflating selectors and ORACLE.
- ORACLE separability incorrectly counted as “best separability”; ORACLE/RANDOM now excluded from best-real.

Interpretation:

- Existing taps are near chance on generated-branch correctness.
- This is not proof DualAnchor is bad; DualAnchor was not tested on its main validity/survival role there.
- CoreContent-to-correctness transfer fails OOD on generated branches.
- Need generated-branch correctness selector trained on verifier-labeled generated pools.

### 6.20 Proto-introspection evidence matrix (June 17)

Deliverables user reported:

- `artifacts/reports/proto_introspection/proto_introspection_evidence_matrix_2026-06-17.md`
- `artifacts/reports/proto_introspection/proto_introspection_evidence_matrix_2026-06-17.json`

Validated: 8 verdict constants, 7 pillars all with status, 10 specificity controls, 4 minimal controls, 8 paper-readiness rows.

Pillar statuses from user report:

- P1 Prediction — STRONG initially.
  - HH pairwise preference 95.2%.
  - Prefix→branch-success pair acc 0.854.
  - Top1 lift +0.162.
  - Oracle 0.90 reasoning@256.
  - Frozen backbone, tiny antisymmetric readout.
- P2 Timing — PARTIAL.
  - Continuations not-yet-generated predicted from prefix/loop states.
  - Positive lift at 32–64-token prefixes and L1-only 0.915.
  - No strictly-pre-answer-token control yet.
- P3 Specificity — PARTIAL.
  - Strong relational/antisymmetry: pairwise 95.2 vs pointwise 65 vs pointwise-linear 21.75; rho about −0.94.
  - length/logprob/prompt-family baselines missing.
- P4 Utility — weak yes / strong no.
  - Retention 0.95–1.0, ranking lift, live tap consumption.
  - Forced top1 count-normalized lift −0.076 and fork adds nothing over K-matched sampling.
- P5 Role separation — STRONG.
  - Validity/content/correctness empirically distinct; tap scores never used as labels.
- P6 Cross-domain — PARTIAL.
  - Transfers across HH/GSM8K/code/reasoning/science, but domain-specialized heads needed.
- P7 Control boundary — STRONG.
  - Readout exists; frozen-backbone steering closed under tested methods. Readout ≠ control.

Verdict:

- `PROTO_INTROSPECTION_WEAK_CLAIM_VERDICT = SUBSTANTIALLY_SUPPORTED_REQUIRES_TARGETED_CONTROLS`
- Timing partial; specificity shortcut controls partial; utility weak supported, strong not proven; control readout supported, control not solved.

### 6.21 Proto-introspection controls and first deflationary result

A first controls run found:

- preanswer hidden AUROC ≈ **0.690**;
- full-answer hidden AUROC ≈ **0.698**;
- shortcut AUROC ≈ **0.702**.

Interpretation at that point:

- Signal was not simply final-answer leakage; pre-answer close to full-answer.
- But shortcut baseline matched or slightly beat hidden.
- Broad shortcut-free claim not proven.
- Narrow claim viable with caveats.

This led to the decisive within-domain powered audit.

### 6.22 Within-domain strict-preanswer specificity audit (June 17)

Deliverables user reported:

- `artifacts/reports/proto_introspection/proto_introspection_within_domain_preanswer_specificity_2026-06-17.md`
- `artifacts/reports/proto_introspection/proto_introspection_within_domain_preanswer_specificity_2026-06-17.json`

Process check:

- jobs empty;
- no stale proto/ouro/python processes;
- GPU idle at 851 MiB before launch;
- nothing terminated.

Domains:

- GSM8K powered primary: **170 tasks**, **680 per-sample examples**.
  - Chosen because answer-last format allows genuine pre-answer reasoning.
  - Median pre-answer reasoning: **163 tokens**.
  - Strict pre-answer cut excludes gold value.
- Reasoning/ARC secondary: **105 tasks**, underpowered, budget-capped.
  - Pre-answer mostly front-loaded; median 14 tokens.

Key AUROC / delta table:

| Domain | Hidden | Length | Logprob | Len+logprob | Hidden+all | Incremental hidden gain beyond shortcuts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GSM8K powered | 0.745 [0.707, 0.783] | 0.687 | 0.569 | 0.731 | 0.797 | +0.066 CI [+0.017,+0.114], significant |
| Reasoning underpowered | 0.690 | 0.590 | 0.528 | 0.597 | 0.690 | +0.093 CI [+0.037,+0.150], significant but underpowered |

Important caveat:

- In GSM8K, hidden alone vs length+logprob composite is +0.014 and not significant.
- The hidden contribution is incremental/complementary, not standalone-dominant.
- Logprob confidence alone is weak (0.569), so the signal is not just output-distribution confidence.

Verdict constants:

- `WITHIN_DOMAIN_PREANSWER_SPECIFICITY_VERDICT = PARTIAL`
- `TIMING_EVIDENCE_VERDICT = PREANSWER_TIMING_CONTROL_PASS`
- `SPECIFICITY_EVIDENCE_VERDICT = HIDDEN_BEATS_SHORTCUTS_WITHIN_DOMAIN`
- `PROTO_INTROSPECTION_WEAK_CLAIM_VERDICT = NARROW_DEFENSIBLE_CLAIM_ONLY`
- `NEXT_STEP_VERDICT = WRITE_PAPER_DRAFT`

Interpretation:

- Weak proto-introspection is paper-ready for a narrow/caveated claim.
- The prior specificity pessimism was at least partly underpower/confounding.
- Defensible claim: looped hidden states expose a readable, pre-answer, partly shortcut-independent process-quality signal about the model’s own ongoing computation.
- Do not claim broad cross-domain shortcut-free proof or hidden dominance over all simple predictors.

### 6.23 Paper-writing package (June 17)

User reported the package is complete and verified against artifacts.

Deliverables:

- `artifacts/reports/proto_introspection/proto_introspection_paper_writing_package_2026-06-17.md`
- `artifacts/reports/proto_introspection/proto_introspection_paper_writing_package_2026-06-17.json`
- `artifacts/reports/proto_introspection/proto_introspection_paper_handoff_short_2026-06-17.md`

Final recommended conservative thesis from package:

> In a frozen looped LM (Ouro-RLTT), intermediate hidden states carry readable, pre-answer process-quality information about the model's own ongoing computation — predicting eventual success in a powered domain and adding statistically significant information beyond length/log-probability shortcuts — but this readable signal does not, on its own, yield autonomous control or capability gains.

Verdicts:

- `PAPER_READINESS_VERDICT = READY_TO_DRAFT_WITH_CAVEATS`
- `PROTO_INTROSPECTION_CLAIM_SCOPE = NARROW_OPERATIONAL_WEAK_FORM`
- `NEXT_STEP_VERDICT = BEGIN_PAPER_DRAFT`

Package locked in:

- Relational readout 0.952; pointwise-linear control 0.2175; rho≈−0.94.
- Strict pre-answer GSM8K AUROC 0.745 [0.707,0.783].
- Significant incremental specificity +0.066 [+0.017,+0.114].
- Honesty ledger:
  - trajectory 0.854 was leakage-inflated or at least not clean enough as headline;
  - hidden-alone ties length+logprob composite;
  - specificity rests on one powered primary domain;
  - frozen branching/steering produced no control or capability gain.

### 6.24 Writing started

The current paper draft began with title/abstract/intro. The user rejected a structure that pushed engineering to appendices. The correct structure makes the engineering stack central.

Preferred title direction:

> Operational Proto-Introspection in Looped Language Models: Relational Taps, Branch Signals, and the Readout-Control Boundary

The title should be distinct from the earlier **Relational Preference Encoding...** paper. Avoid generic titles like “Readable Process-Quality Signals in Looped-Transformer Hidden States” because they sound too similar to the old paper and understate the branch/tap/control engineering.

---

## 7. Main result blocks and how to write them

### 7.1 Result block 1: Relational hidden-state quality signal

Purpose:

- Establish that frozen Ouro-RLTT hidden states contain strong relational preference/quality signal.

Use:

- HH 95.2 pairwise.
- L-BFGS 84.5.
- pointwise 65.
- pointwise-linear 21.75.
- flip rho −0.92 to −0.97 / about −0.94.

Do not overclaim:

- This is not by itself proto-introspection.
- It establishes relational hidden-state quality geometry.

### 7.2 Result block 2: Tiny taps and layer/loop localization

Purpose:

- Show the signal compresses into small local readouts rather than needing a large evaluator.

Use:

- layer 24/36/47;
- loops L1-L4;
- `[3,4,2048]` features;
- AntisymLinear / NoNorm / blockwise heads;
- exact antisymmetric pairwise taps.

Claim:

- Low-capacity taps imply signal is in hidden geometry.

### 7.3 Result block 3: Domain transfer and mixed objectives

Purpose:

- Show process-quality geometry is not only HH.

Use:

- clean GSM8K;
- code runnable diagnostics;
- strict-clean code;
- reasoning natural distractors;
- science partial/diagnostic;
- CoreContent v2 expansion;
- mixed objective heads.

Claim:

- Transfer exists, but role/domain specificity matters.

### 7.4 Result block 4: Role separation

Purpose:

- Prevent collapse of preference/content/validity/correctness/control into one “evaluator.”

Key distinctions:

- DualAnchor = branch survival / validity / survivability.
- CoreContent = content/final-choice ranking.
- Generated-branch correctness selector = separate future S3B axis.
- Gold/verifier/exact correctness = labels, never tap scores.

Claim:

- Role separation is one of the strongest conceptual results.

### 7.5 Result block 5: Executable internal branching

Purpose:

- Show readouts can be consumed by a live scaffold.

Use:

- KV/cache branch-carry;
- UniversalTransformerCache 192 slots;
- pruning/reordering;
- suffix recompute splice;
- S1 gates;
- full branch/carry/prune/loop-back scaffold.

Claim:

- Internal branch-control substrate is real, not theoretical.

### 7.6 Result block 6: Frozen branching/steering null

Purpose:

- Establish readout/control boundary.

Use:

- steering closure;
- adapter failures;
- S1 frozen branch/carry null;
- K-matched sampling deconfound;
- LoRA proof-of-capability flat reachability.

Claim:

- Readable signal does not automatically imply frozen control or capability gain.

### 7.7 Result block 7: Strict pre-answer audit

Purpose:

- Anti-skeptic control: signal appears before answer and adds information beyond shortcuts.

Use:

- GSM8K 170 tasks / 680 samples;
- strict pre-answer cut;
- hidden AUROC 0.745;
- hidden+all 0.797;
- length+logprob 0.731;
- incremental +0.066 significant.

Claim:

- Process-quality signal is pre-answer and partly shortcut-independent.

---

## 8. Current paper structure recommended

Do not push engineering to appendices. Use something like:

1. **Introduction**
   - looped models;
   - hidden process signals;
   - why engineering + interpretability;
   - contributions.

2. **Operational Proto-Introspection**
   - definition;
   - non-mentalistic disclaimers;
   - why a family of engineering readouts is stronger than one probe.

3. **Ouro-RLTT as a Looped Hidden-State Substrate**
   - loops/layers;
   - hidden state features;
   - loop/layer/tap map.

4. **From Linear Probe to Relational Evaluator**
   - pairwise difference;
   - HH 0.952;
   - pointwise controls;
   - flip tests.

5. **Tiny Taps and Layer/Loop Localization**
   - low-capacity taps;
   - AntisymLinear;
   - role of layers 24/36/47 and L1-L4.

6. **Domain Transfer and Mixed Objective Readouts**
   - transfer across HH/math/reasoning/code/science;
   - generalist/specialist conclusion;
   - domain caveats.

7. **Role-Specialized Signals: DualAnchor, CoreContent, Correctness**
   - survival vs content vs correctness;
   - selection wall;
   - S3B need.

8. **Executable Internal Branching**
   - KV/cache branch-carry;
   - suffix recompute splice;
   - live branch/carry/prune/loop-back.

9. **Frozen Branching and Steering: The Readout-Control Boundary**
   - S1 closure;
   - steering/adapters closure;
   - K-matched sampling;
   - LoRA proof-of-capability.

10. **Strict Pre-Answer Process-Quality Audit**
    - GSM8K powered audit;
    - shortcut baselines;
    - hidden+all incremental gain;
    - underpowered ARC secondary.

11. **Synthesis: Why This Supports Weak Operational Proto-Introspection**
    - connect all result blocks.

12. **Limitations**
    - all caveats.

13. **Future Work**
    - S3B-pre;
    - S3A;
    - robustness term;
    - Barbados;
    - Jormungandr.

14. **Conclusion**

Appendices:

- A: probe/evaluator formulas.
- B: tap architecture details.
- C: domain-transfer tables.
- D: branch-carry/splice validation.
- E: strict pre-answer audit details.
- F: steering/adapter negatives.
- G: artifact/reproducibility index.

---

## 9. S3A and future compute pitch

### 9.1 Why S3A is justified

Frozen branch/carry was mechanically valid but reachability-neutral. This does not mean branch-control is conceptually wrong. It means the frozen model has not learned to make injected hidden branches outcome-distinct and useful.

S3A should train the loop dynamics / backbone / adapters with verifier-labeled branch tournaments so that:

- injected/internal branches become outcome-distinct;
- branch-survival and terminal selection signals align with verifier correctness;
- the model learns to use process-quality signals, not merely expose them.

### 9.2 Orthogonality / injection-outcome subspace audit

Claude previously suggested a cheap local audit:

- define injection span from frozen branch injection/carry deltas;
- define outcome direction from verifier-correct vs verifier-incorrect hidden features;
- define natural sampling span from K-sampling variation;
- measure projection fraction, principal angles, CCA/subspace overlap, classification on projected/residual features.

Possible interpretations:

- If outcome direction mostly outside injection span: frozen null explained by subspace misalignment; S3A is strongly justified.
- If outcome inside injection span but frozen still fails: bottleneck likely nonlinear dynamics/magnitude/decoding/instability.
- If injection span ≈ sampling span: frozen branching is mostly a worse sampler; S3A must prove it creates a new trained control manifold.

This audit is high-value for compute pitch but not a blocker before writing.

### 9.3 Robustness term for S3A

Important concern:

If S3A trains the model to make L24/L36 perturbations propagate into outcome-different generations, it might make readout axes brittle/noisy. A heldout tap-accuracy preservation gate might miss this.

Add robustness gate/term:

- perturb BG-readable axes randomly during training;
- require branch ranking/verifier ordering to remain stable;
- measure readout stability under small hidden perturbations;
- preserve margin and calibration, not only accuracy;
- compare clean vs perturbed readout rankings.

Suggested criterion:

> BG-readable-axis robustness: after training, hidden-state taps must preserve ranking under small random perturbations/noise in readout-relevant axes, not merely retain heldout accuracy on clean features.

---

## 10. Exact claim safety table

| Claim | Status | Notes |
| --- | --- | --- |
| Ouro hidden states encode relational preference/quality | Safe | HH 95.2, L-BFGS 84.5, pointwise weaker, flip correlation strong. |
| Signal is mostly relational, not pointwise | Safe | Pointwise 65 / pointwise-linear 21.75 contrast. |
| Tiny taps can read useful hidden-state process signals | Safe | Multiple tap/domain audits; low-capacity heads. |
| Taps transfer across domains | Safe with caveat | Transfer partial; specialists needed. |
| DualAnchor is a branch survival signal | Safe | V3 retention strong. |
| DualAnchor is a correctness selector | Unsafe | Do not say this. |
| CoreContent improves final/content selection | Safe with caveat | Stronger on intended distribution; OOD generated correctness weak. |
| Existing taps solve generated-branch correctness | Unsafe | S3B-1 says near-chance / OOD failure. |
| KV/cache branch-carry is validated | Safe | Test harness; compute-saving splice valid. |
| Branch/carry is production-ready | Unsafe | Not claimed. |
| Frozen branch/carry improves capability | Unsafe | K-matched sampling explains gains. |
| Readout/control boundary established | Safe | Steering/adapters/S1 nulls. |
| Strict pre-answer process-quality signal exists in GSM8K | Safe with caveat | 0.745 AUROC; one powered primary domain. |
| Hidden states dominate all shortcuts | Unsafe | Hidden-alone ties length+logprob composite. |
| Hidden states add significant information beyond shortcuts | Safe | Hidden+all vs shortcuts +0.066 CI positive. |
| Proto-introspection in weak operational form | Safe with caveat | Define operationally. |
| Consciousness/self-awareness | Forbidden | Never claim. |
| Autonomous self-control | Forbidden | Not shown. |
| Jormungandr capability gain | Forbidden | Future work. |

---

## 11. Known exact numbers and verdicts to preserve

### Relational evaluator / paper

- Pairwise evaluator: **95.2%** on **8,552** unseen HH-RLHF examples.
- L-BFGS pairwise-difference probe: **84.5%**.
- Best independent nonlinear pointwise evaluator: **65%**.
- Linear independent classification: **21.75%** below chance/inverted polarity.
- Epoch 1: **83.3%**.
- Epoch 2: **95.2%**.
- Epoch 5: **62.4%**.
- Flip correlation: **ρ = −0.92 to −0.97**, often summarized as ~−0.94.

### DualAnchor V3

- Tasks: 48 total, 24 reasoning / 24 science.
- Stage oracle retention: **0.9848**.
- Terminal oracle retained: **1.0000**.
- Forced terminal top1 oracle: **0.9167**.
- Forced terminal top1 reward: **0.2625**.
- Terminal best-survivor reward: **0.3167**.
- Reward-diverse rate: **0.2292**.
- Positive-oracle rate: **0.3542**.
- False-prune recovery: **8/8**.

### CoreContent v2

- Expanded data: coding 1,733; reasoning 2,600; math 3,200; logic 2,199; alignment ~26k.
- Feature storage: **4.87 GB**, **64 shards**, **0 errors**.
- CoreContent v2 blockwise: **0.6691 [0.645, 0.690]**.
- Mixed HH/objective baseline: **0.5525 [0.526, 0.577]**.
- Real-negative edge: **+0.063**.
- Coding mutant/corruption detector: ~**0.94** vs mutants, ~**0.58** vs real wrong-problem code.

### Branch training / reachability

- Dataset: **48.5k tasks**, **10 verifier-backed families**, **33.2k train**.
- CoreContent in real DualAnchor top5 survivors: **0.658**.
- MIX_HH: **0.552**.
- DualAnchor forced top1: **0.379**.
- Oracle retention: **1.0**.
- Reachability@4: reasoning **0.95**, math **0.83**, logic **0.73**, coding **0.43**.
- Math fix: **0.31 -> 0.83**.
- DualAnchor teacher lift: +0.11 coding/reasoning, +0.03 logic.
- 300-step bf16 LoRA: diversity +0.45; coding parse 0.72→0.94; math 0.75→0.92; macro reachability 0.708→0.688.

### Offline generator v2

- r1: −0.244.
- r2: −0.052.
- r3: −0.112; coding −0.467.
- r4: −0.059.
- r4 branch chars: 0.70× base; r1–r3 0.26–0.40×.
- coding r3/r4: −0.467 → −0.067 at 0.92× length.

### KV/cache and splice

- UniversalTransformerCache slots: **4 loops × 48 layers = 192 slots**.
- L1 fork K=2/4/8 validated.
- Prune examples: 8→4→2, 8→3, 4→1.
- Negative control RMS ≈ **3.0**.
- Partial splice v2 per-branch layer-pass savings: **13% / 38% / 63% / 88%** for boundary loops 0/1/2/3 at layer 24.
- Loop 2 layer 24 K-scaling fewer passes: **32% / 47% / 55%** for K=2/4/8.

### S1

- alpha=0 identity: maxabs **0.0**, **48/48** tokens.
- Reference: 4 tasks, K=2, budget=4, alpha=0.02, last-token, greedy.
- `oracle_over_survivors = base_acc = selected_acc = 0.25`.
- logic divergence 0/12; math 10/12; reasoning 9/12; coding 11/12.
- K-matched sampling: plain sampling oracle **0.75**, sample fork **0.611**, greedy fork **0** new-correct.

### S3B

- CoreContent_v2 frozen: **0.6691**.
- S3B0 listwise: **0.6512**.
- S3B0 pairwise: **0.6399**.
- MIX_HH: **0.5526**.
- MIX_OBJECTIVE_ALL: **0.3833**.
- DualAnchor: **0.3787**.
- RANDOM: **0.2709**.
- Generated branch transfer best real selector MIX_HH: **0.667**.
- Random: **0.583**.
- Real tap separability: **0.49–0.57**.
- CoreContent generated transfer: **0.417**.

### Proto-introspection strict preanswer

- GSM8K: 170 tasks, 680 samples.
- Median pre-answer reasoning: 163 tokens.
- Hidden AUROC: **0.745 [0.707, 0.783]**.
- Length: **0.687**.
- Logprob: **0.569**.
- Length+logprob: **0.731**.
- Hidden+all: **0.797**.
- Incremental gain: **+0.066 CI [+0.017, +0.114]**.
- Reasoning secondary: 105 tasks, hidden 0.690, length+logprob 0.597, hidden+all 0.690, increment +0.093 CI [+0.037,+0.150], underpowered.

---

## 12. Problems / issues faced and lessons learned

### 12.1 Pairwise metric trap

Pairwise training metrics were misleading due to antisymmetry/swap protocols and scorer bias. The project discovered degenerate constant-output and metric-deflation failure modes. Flip tests became mandatory.

Lesson:

- Pairwise evaluators need explicit swap/flip diagnostics.
- Accuracy alone can be meaningless under bad protocols.

### 12.2 Pointwise vs pairwise

Pointwise models underperformed badly relative to pairwise readouts. This forced a conceptual pivot: quality/preference is represented relationally.

Lesson:

- Use pairwise difference geometry for candidate selection.

### 12.3 Truncation / dirty branch artifacts

Early transfer looked worse due to truncation and dirty generated branches. Clean GSM8K/code repairs changed interpretation.

Lesson:

- Model limitation and harness limitation can be confounded.

### 12.4 Code relevance / corruption detector issue

CoreContent/code taps sometimes detected mutants/corruption rather than true relevance/correctness against real wrong-problem code.

Lesson:

- Constructed negatives can overstate usefulness.
- Real-negative stress tests are essential.

### 12.5 Science parsing and tiny-N science

Science/MMLU repeatedly had parser and small-n issues. Anatomy partially repaired but science remains diagnostic.

Lesson:

- Science should not be headline domain until stronger.

### 12.6 Steering/control failure

Frozen read/write paths were mechanically valid but not behaviorally controllable under safe tested methods.

Lesson:

- Readout is not control.
- Training-time integration is required.

### 12.7 S1 frozen branch null

Branch/carry/prune scaffold worked, but deterministic injected frozen branches did not beat sampling.

Lesson:

- Mechanism real, frozen reachability null.
- S3A needed to make branches outcome-distinct.

### 12.8 Training/eval format mismatch

Offline generator rounds r1–r3 looked like training degraded the generator; r4 revealed much was format mismatch and bad coding data.

Lesson:

- Prompt/rendering byte-identity matters.
- Do not overinterpret training negatives before format alignment.

### 12.9 Pre-answer shortcut controls

Initial broad pre-answer audit showed shortcut baselines could match hidden features. Powered within-domain audit rescued a narrower claim.

Lesson:

- Claim only partly shortcut-independent, not shortcut-dominant.
- Use strict preanswer + within-domain + grouped splits where possible.

---

## 13. Adjacent Barbados / BLT context

Barbados is a separate tiny-looped-transformer proof-of-framework project, not the main Ouro paper.

Relevant current memory:

- Barbados = minimal looped transformer / “smallest snake” wind-tunnel for Jormungandr.
- Exact GPT→looped conversion was done.
- Recurrence finetuning improved in-band COPY/REVERSE vs continued GPT under matched budget.
- No length extrapolation.
- Boundary failures remain.
- Low-rate shift/damping ineffective.
- Modular-expression edge appears in easy strata only:
  - Track A modular-expression 2× scale at 49.152M tokens, 5 matched seeds:
    - Barbados L4 ID 18.59% ± 3.67 vs GPT continued 15.78% ± 3.23;
    - paired Δ +2.81 ± 1.06;
    - Barbados > GPT in 5/5 seeds.
    - Edge confined to easy strata: one-op Δ +12.81 pts, addition-only Δ +13.91, multi-op Δ +0.81, multiplication-containing Δ +0.38 noise.
    - Near/far remain ~6–8%; loop refinement persists; complexity generalization still fails.
- Track B pointer-chase implemented and validated; one-loop equivalence 0.0; details incomplete in current memory.

Use in Ouro paper only as future work or adjacent motivation, not as a main result.

---

## 14. Current outreach / external compute context

### 14.1 Professor / local academic context

User presented the work to a professor/doctor Džambić for ~75 minutes. The professor was supportive, may connect the user with an ML professional, possibly Mistral. This is relevant for compute/collaboration framing but not a paper result.

### 14.2 Arnau Padres / MELT contact

Arnau replied:

- Ideas look really interesting.
- They have not explored this direction themselves, but it seems promising.
- Looped models look like a natural playground for branching.
- He wonders how much has been explored in standard transformers.
- MELT access likely not available until after a paper is accepted due internal legal processes.

Interpretation:

- Positive high-level engagement.
- Not a MELT access path yet.
- Paper first, access later.

---

## 15. What Claude should do with this handoff

### Do

- Treat the engineering stack as central to the paper.
- Keep claims scoped.
- Preserve negative results.
- Use the strict preanswer audit as the cleanest anti-shortcut control.
- Use S1/S3/steering failures to define the readout-control boundary.
- Explicitly separate survival, content, correctness, preference, validity, and control.
- Help write sections with full technical detail and caveats.
- Suggest figures/tables that make the stack legible.
- Recommend checks when a number needs verification.

### Do not

- Collapse the project into a simple “probe hidden states” paper.
- Push branch engineering into appendices only.
- Claim consciousness/self-awareness.
- Claim autonomous self-control.
- Claim Jormungandr capability gain.
- Claim hidden states dominate all shortcuts.
- Claim S3A was run.
- Treat DualAnchor as correctness selector.
- Treat CoreContent OOD failure as global CoreContent failure.
- Hide that frozen branching/steering failed to improve capability.
- Hide that hidden-alone ties length+logprob composite in the powered audit.

---

## 16. Suggested first paper-writing tasks for Claude

1. Write Section 2: **Operational Proto-Introspection** with very strict non-mentalistic framing.
2. Rewrite the Introduction so it frames the paper as systems + interpretability, not probe-only.
3. Draft Section 4: **From Linear Probe to Relational Evaluator**, using the verified evaluator numbers.
4. Draft Section 8: **Executable Internal Branching**, using KV/cache branch-carry and splice details.
5. Draft Section 9: **Readout-Control Boundary**, combining steering closure + S1 null + LoRA pilot.
6. Create figure/table plan:
   - Fig 1: looped hidden-state/tap overview.
   - Fig 2: pairwise relational geometry.
   - Fig 3: tap/role map.
   - Fig 4: DualAnchor survival vs CoreContent selection.
   - Fig 5: KV/cache branch-carry scaffold.
   - Fig 6: frozen branch null vs sampling.
   - Fig 7: GSM8K strict preanswer AUROC bars.
   - Table 1: claim safety table.
   - Table 2: verdict chronology.
   - Table 3: roles/labels/features.

---

## 17. Shortest possible correct summary

The project began with a strong relational HH-RLHF hidden-state evaluator in frozen Ouro-RLTT: 95.2% pairwise vs much weaker pointwise controls. It then compressed the signal into tiny layer/loop taps, showed partial cross-domain transfer, separated branch survival from content/correctness, built DualAnchor survival and CoreContent selection, validated generation-time KV/cache branch-carry and compute-saving suffix splice, built a live branch/carry/prune scaffold, and found that frozen steering/branching does not yield capability gains beyond sampling. A strict preanswer GSM8K audit showed hidden states predict future success before the answer and add significant information beyond length/logprob shortcuts. Therefore the defensible claim is weak operational proto-introspection: Ouro-RLTT exposes readable process-quality information about its own ongoing computation, but does not yet show autonomous self-control. S3A is the future training step to turn readout into control.

---

## 18. Local files/source index used for this handoff

Mounted docs/files available in this session included:

- `/mnt/data/paper.pdf`
- `/mnt/data/evaluator_pairwise.py`
- `/mnt/data/content-selection-taps.md`
- `/mnt/data/flip-test-interpretation.md`
- `/mnt/data/interfaces-and-tools.md`
- `/mnt/data/evaluator-navigation-map.md`
- `/mnt/data/README.md`
- `/mnt/data/science-reasoning-repair.md`
- `/mnt/data/chronological-evaluator-summary.md`
- `/mnt/data/core-domain-tap-audit.md`
- `/mnt/data/corecontent-dataset-expansion-v2.md`
- `/mnt/data/kv-cache-branch-carry.md`
- `/mnt/data/evaluator-locus-summary.md`
- `/mnt/data/dualanchor-tap-evolution.md`
- `/mnt/data/domain-transfer-ledger.md`
- `/mnt/data/current-state.md`
- `/mnt/data/branch-generation-and-survival.md`
- `/mnt/data/terminal-selection-and-arbiters.md`
- `/mnt/data/branch-training-logic-expansion.md`
- `/mnt/data/steering-and-adapters.md`
- `/mnt/data/README_ouro_rltt_evaluator_project.md`
- `/mnt/data/README_ouro_project.md`
- several older synthesis/handoff docs from May 2026.

Some older uploaded files expired in the ChatGPT environment, and the actual local repo may contain more exact reports/JSONs than were visible here. Claude should verify exact artifact paths and numbers in `/home/moloch/ouro_project` before final paper use.

