# Fable hardening checks

Run timestamp: `20260710_175017`  
Project: `/home/moloch/ouro_project`  
Draft consulted read-only: `/home/moloch/Downloads/ouro_paper_draft_v3_16.md`

No Ouro model was trained, no checkpoint was modified, no broad generation was run, and the paper draft was not edited.

## 1. Executive summary

### Check 1: antisymmetric-linear localization

**The claimed base/Thinking/RLTT localization did not reproduce in the clean linear units.** The closest recoverable reconstruction of the approximately 84.5% probe scored **83.75% on all three backbones**: base Ouro-2.6B, Ouro-2.6B-Thinking, and local Ouro-RLTT. All scores were exactly swap-safe (`swap_consistency=1.0`, `max |s(A,B)+s(B,A)|=0`). The base features are demonstrably different from the Thinking features, so this is not a path-alias or identical-feature error; the learned linear decisions genuinely transfer.

The old 24% / 95.2% / 95.0% pattern therefore remains a result in the discredited fixed-order high-capacity evaluator units and cannot support §3.7's main-text localization claim.

- `ANTISYM_LINEAR_PROBE_STATUS = RECONSTRUCTED_LINEAR_PROBE`
- `LOCALIZATION_BACKBONE_STATUS = ALL_THREE_BACKBONES_EVALUATED`
- `LOCALIZATION_RESULT_STATUS = LOCALIZATION_NOT_REPRODUCED`
- `PAPER_ACTION_LOCALIZATION = REMOVE_OR_TODO`

### Check 2: task-clustered strict pre-answer CI

**The primary statistical result survives task clustering.** Recreating the exact grouped-CV out-of-fold predictions from the raw 170-task/680-example artifact gives:

- hidden+all AUROC: **0.796973**
- length+logprob AUROC: **0.731183**
- incremental ΔAUROC: **+0.065790**
- paired task-clustered bootstrap, 10,000 draws: **95% CI [+0.020707, +0.112254]**
- skipped degenerate draws: **0**
- excludes zero: **yes**

The historical report explicitly calls its [+0.0172,+0.1144] interval a group bootstrap, but the exact paired cluster-bootstrap routine and original OOF predictions were not preserved. Thus the historical execution path is not code-auditable even though the new clustered recomputation independently verifies the paper-critical significance claim.

- `PREANSWER_RAW_PREDICTIONS_STATUS = RAW_PREDICTIONS_FOUND`
- `PREANSWER_CI_ORIGINAL_STATUS = ORIGINAL_CI_PROVENANCE_UNKNOWN`
- `PREANSWER_TASK_CLUSTERED_RESULT = TASK_CLUSTERED_CI_EXCLUDES_ZERO`
- `PAPER_ACTION_PREANSWER = KEEP_PRIMARY_CLAIM_VERIFIED`

## 2. Check 1 — antisymmetric-linear localization

### 2.1 Search and recovery

The approximately 84.5% number is documented throughout the draft and evaluator history, but no saved weight vector was found. The closest executable source is the tracked `probe_test.py`, currently deleted in the dirty worktree but inspected read-only with `git show HEAD:probe_test.py`.

Recovered setup:

| Item | Recovered value |
|---|---|
| Training backbone | `ByteDance/Ouro-2.6B-Thinking` |
| Data | `Anthropic/hh-rlhf`, chosen/rejected pairs |
| Candidate pooling | mean over mask-valid tokens |
| Locus | post-final-norm loop-boundary / layer 47 |
| Loops | L1, L2, L3, L4 concatenated |
| Feature width | `4 × 2048 = 8192` |
| Difference | raw/NoNorm `h_left - h_right` |
| Probe | full-batch L-BFGS logistic linear readout |
| Clean scoring used here | bias-free `wᵀ(h_left-h_right)` |
| Swap property | exact: `s(B,A) = -s(A,B)` |
| Seed/split | symmetric `+Δ/-Δ` rows, `RandomState(42)`, 80/20 row split |

The reconstruction uses the saved 200-pair Thinking capture, full-batch `torch.optim.LBFGS` with strong-Wolfe line search, and the `C=1`-equivalent penalty `||w||²/(2n)`. Its Thinking accuracy is 83.75%, within 0.75 percentage points of the documented approximately 84.5% result. Accordingly this is marked `RECONSTRUCTED_LINEAR_PROBE`, not `SAVED_PROBE_FOUND`.

Important split caveat: the historical construction splits the positive and negative orientations at row level, so opposite orientations of a source pair can cross train/eval. The 80 evaluation rows represent 74 unique source pairs. The CI below clusters duplicate orientations by source pair. This caveat is another reason not to use this small replication as a new positive headline.

Probe artifact: `reconstructed_antisym_linear_probe.pt`.

### 2.2 Backbones and captures

| Backbone | Exact model/path | Revision or immutable identity | Evaluation source | Status |
|---|---|---|---|---|
| Base Ouro 2.6B | `ByteDance/Ouro-2.6B` | `1ed04250da1a9936042725d302e81c8fa2ab5abd`; model SHA-256 `2fdf9805…c1bba` | fresh bounded pooled capture, `hh_boundary_pooled_200_base.pt` | loaded successfully |
| Ouro 2.6B Thinking | `ByteDance/Ouro-2.6B-Thinking` | `f1edd81e7ac41355db670500ceaf204e0f73af68`; weight blob `c506a792…a3080` | saved `artifacts/reports/evaluator/hh_layer_states_200_thinking.pt` | reused |
| Ouro RLTT | `/home/moloch/ouro_project/models/ouro_rltt_local` | no embedded revision; shard SHA-256 values recorded in JSON | saved `artifacts/reports/evaluator/hh_layer_states_200_rltt.pt` | reused |

Base weights were downloaded only to `/home/moloch/ouro_project/artifacts/hf_cache`. All three models use the same local Ouro tokenizer/input IDs, fixed 384-token truncation, four loops with `early_exit_threshold=1.0`, bf16 forward, fp32 captures, and no quantization.

Runtime: Python 3.14.6; Transformers 4.54.1; Torch 2.12.0.dev20260407+cu128; CUDA 12.8; NVIDIA GeForce RTX 5070 Ti Laptop GPU.

### 2.3 Results

| Backbone | Model ID/path | n ordered eval rows (unique pairs) | Feature shape | Probe status | Strict antisym accuracy | 95% CI, clustered by source pair | Swap consistency | Fixed-order diagnostic | Notes |
|---|---|---:|---|---|---:|---|---:|---|---|
| Base Ouro 2.6B | `ByteDance/Ouro-2.6B` | 80 (74) | `[200,8192]` | reconstructed | **0.8375** | [0.7349,0.9351] | 1.000 | not computed | same weights/split |
| Ouro 2.6B Thinking | `ByteDance/Ouro-2.6B-Thinking` | 80 (74) | `[200,8192]` | reconstructed | **0.8375** | [0.7349,0.9351] | 1.000 | not computed | fit backbone |
| Ouro RLTT local | `models/ouro_rltt_local` | 80 (74) | `[200,8192]` | reconstructed | **0.8375** | [0.7349,0.9351] | 1.000 | not computed | unchanged transfer |

Every evaluated row has `score(B,A) = -score(A,B)` exactly. Predictions use `sign(score(A,B))`; no canonical-order prior or fixed-order result enters the headline.

### 2.4 Identity/path audit

The equality in accuracy is not caused by loading the same checkpoint three times:

| Comparison | Mean cosine of candidate-difference vectors | Score Pearson | Mean absolute score delta | Decision agreement | Max absolute feature delta |
|---|---:|---:|---:|---:|---:|
| base vs Thinking | 0.8765 | 0.9118 | 0.5777 | 0.995 | 2.0173 |
| Thinking vs RLTT | 0.9434 | 0.9989 | 0.0691 | 1.000 | 0.5074 |

The representations differ, especially base vs Thinking, while the reconstructed linear decision boundary remains stable. This directly contradicts the claimed collapse to 24% in these clean units.

### 2.5 Check 1 verdict

All three backbones were evaluated, but the expected localization pattern was absent. The appropriate action is to remove/TODO §3.7's main-text claim rather than translate the old fixed-order values into clean units. A definitive positive retest would require either the missing original weight vector plus its full held-out set or a larger reconstruction with an independent source-pair split.

Supporting tables: `localization_results.json`, `localization_rows.csv`.

### 2.6 Follow-up: original evaluator decomposition

A requested follow-up applied the original `pairwise_epoch2.pt` evaluator in both orders and crossed attention/mean pooling with difference normalization on/off. The original canonical evaluator also shows **no base collapse** on the controlled 200-pair slice: base 0.950, Thinking 0.950, RLTT 0.945. Strict antisymmetric accuracies are only 0.580, 0.595, and 0.600, respectively; swapped correctness is 0.110–0.130. The symmetric/order component dominates every backbone. Mean pooling retains the behavior, while removing normalization makes the order dominance stronger.

No surviving repository artifact contains the historical base 24% result; draft v3.16 labels it “User-confirmed; base-eval artifact path pending.” Full results are in `evaluator_decomposition.md`, `evaluator_decomposition.json`, and the paired CSV tables. This strengthens `LOCALIZATION_NOT_REPRODUCED`.

## 3. Check 2 — task-clustered strict pre-answer CI

### 3.1 Artifact paths

- Raw recapture: `artifacts/reports/proto_introspection/within_domain_recapture.pt`
  - SHA-256: `f0e44d7e859cb08c6423a1422f0899fe62a40ebec13dcb1a457952d5a04fab82`
- Historical report: `artifacts/reports/proto_introspection/proto_introspection_within_domain_preanswer_specificity_2026-06-17.{md,json}`
- Preserved analysis: `utilities/tests/manual/proto_introspection_within_domain_analysis.py`
- Preserved probe/bootstrap utilities: `utilities/tests/manual/proto_introspection_controls_analysis.py`
- Recreated predictions: `preanswer_oof_predictions.csv`
- New bootstrap result: `preanswer_task_clustered_ci.json`
- Influence table: `preanswer_leave_one_task_out.csv`

The raw artifact contains 170 GSM8K task IDs, four samples per task, strict-pre-answer hidden features, length/logprob features, external correctness labels, and sample IDs. Therefore no generation or model loading was required.

### 3.2 Historical CI provenance

The June MD and JSON explicitly say that paired deltas used a group bootstrap resampling tasks and report `+0.0658 [0.0172,0.1144]`. However:

- the preserved `within_domain_analysis.py` recreates grouped-CV scores but does not implement a paired clustered delta interval;
- its imported `bootstrap_auroc_ci` resamples candidate rows for marginal AUROC intervals;
- no original OOF score table or bootstrap draw table was saved.

Consequently, the narrative provenance supports task clustering, but the exact historical run cannot be independently code-audited. The conservative constant is `ORIGINAL_CI_PROVENANCE_UNKNOWN`. This does not undermine the primary significance claim because it was recomputed below with an explicit auditable task-clustered routine.

### 3.3 Recomputed method

1. Recreate the two out-of-fold score vectors using the preserved five-fold task-grouped CV (`group=task_id`, seed 20260617).
2. For each of 10,000 bootstrap draws, sample 170 task IDs with replacement.
3. Include all four candidate rows for every sampled occurrence of a task.
4. Compute AUROC for hidden+all and length+logprob on the same draw.
5. Store the paired difference `AUROC(hidden+all)-AUROC(length+logprob)`.
6. Use the 2.5th and 97.5th percentiles; skip one-class draws (none occurred).

Bootstrap seed: `20260710`.

### 3.4 Bootstrap result

| Quantity | Result |
|---|---:|
| Tasks / examples | 170 / 680 |
| Positive / negative | 407 / 273 |
| hidden+all AUROC | 0.796973 |
| length+logprob AUROC | 0.731183 |
| Full-data ΔAUROC | **+0.065790** |
| Cluster-bootstrap mean | +0.065770 |
| Cluster-bootstrap std | 0.023464 |
| 95% percentile CI | **[+0.020707,+0.112254]** |
| Requested / valid / skipped draws | 10,000 / 10,000 / 0 |
| Excludes zero | **yes** |

The candidate-level diagnostic gives the narrower interval [+0.03158,+0.10035] and is not used as the primary result. Leave-one-task-out deltas range from +0.0560 to +0.0706; the maximum absolute change from the full-data delta is 0.00979.

### 3.5 Check 2 verdict

The point estimate reproduces exactly to displayed precision and the correct task-clustered interval excludes zero. §5's primary claim is safe, with the new auditable interval replacing the historically pinned endpoints if the paper is patched.

## 4. Exact recommended paper patches

### §3.7 downgrade/removal wording

> In a targeted replication using the bias-free antisymmetric linear difference probe, the same held-out HH accuracy was obtained for base Ouro-2.6B, Ouro-2.6B-Thinking, and Ouro-RLTT (83.75% for each; exact swap consistency 1.0). Thus the earlier 24%/95.2%/95.0% separation is specific to the fixed-order high-capacity evaluator and does not establish training-stage localization in the clean relational units. We therefore do not claim that reasoning/RL training installs the linearly readable relational preference signal; resolving training-stage localization requires a larger independently pair-split antisymmetric replication.

Recommended scope: remove the current §3.7 localization claim, plus corresponding abstract/contribution/discussion references. If historical context is retained, place the fixed-order result in an appendix and label it non-load-bearing.

### §5 verified wording

> At the strict pre-answer cut, adding hidden features to the length-and-log-probability baseline raises AUROC from 0.731 to 0.797 (incremental ΔAUROC +0.066). A paired task-clustered bootstrap over 170 GSM8K problems (10,000 draws; all four candidates retained per sampled problem) gives a 95% CI of [+0.021, +0.112], which excludes zero. The improvement is therefore statistically significant under task-clustered resampling, while remaining a modest incremental effect demonstrated in one powered domain.

## 5. Remaining blockers

1. The original approximately 84.5% weight vector was not saved; Check 1 is necessarily a reconstruction.
2. The historical `+Δ/-Δ` row split allows opposite orientations of a pair to cross train/eval. A larger independent pair-level split is needed for a definitive localization study.
3. The exact code/draws producing the historical [+0.0172,+0.1144] pre-answer interval were not preserved. The new task-clustered result verifies significance but yields the independently reproducible interval [+0.0207,+0.1123].
4. The local RLTT checkpoint has no embedded upstream revision; immutable shard hashes are recorded in the JSON report.
