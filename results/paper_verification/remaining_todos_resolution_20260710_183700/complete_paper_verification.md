# Complete paper verification and remaining TODO resolution

Draft inspected read-only: `/home/moloch/Downloads/ouro_paper_draft_v3_16.md`  
No Ouro training, checkpoint mutation, or broad generation was performed.

## Executive verdict

- **Training-stage localization:** not reproduced in clean antisymmetric-linear units. Base, Thinking, and RLTT each score 0.8375 in the recovered reconstruction. Remove the 24%/95.2%/95.0% localization claim as a strong result.
- **Strict pre-answer GSM8K:** verified. ΔAUROC = +0.065790 with fresh paired task-clustered 95% CI [+0.020707, +0.112254], excluding zero.
- **Full HH evaluator swap audit:** fixed-order 0.9478; strict antisymmetric 0.6391 (95% CI [0.6291, 0.6493]). The fixed-order result is substantially order-artifacted.
- **S3B2 matched random:** observed 5/8 = 0.625 versus native uniform-random expectation 0.3625; exact P(X≥5) = 0.0869. Above expectation, not significant at 0.05.
- **S1 and steering spot-checks:** verified exactly against source JSON/diagnostic artifacts.
- **Second powered domain:** closed as a documented preflight rejection, not silently left untested.

## 1. Why the evaluator produced the old-looking high numbers

The evaluator genuinely outputs high canonical-order accuracy, but that is not the same measurement as an antisymmetric relational probe. On the controlled 200-pair cross-backbone slice, the original attention+normalization head gives canonical accuracies 0.950 (base), 0.950 (Thinking), and 0.945 (RLTT), while strict antisymmetrized accuracies are only 0.580, 0.595, and 0.600. Swapped-direction correctness is 0.110–0.130 and the symmetric order component has mean about +1.26 to +1.29, demonstrating a strong learned preference for the first argument. Attention pooling is not required for the canonical high result, and normalization mitigates rather than creates the order dominance.

This explains how the evaluator could surface ~95% numbers: chosen was canonically placed first, and the high-capacity fixed-order head exploited that regularity. It does **not** explain or recover the historical base=24% cell. The current controlled application of the same saved head gives base=95%, and no artifact for the 24% run was found. That cell most likely depended on an unrecovered mismatch in extraction locus, preprocessing/tokenization, dataset ordering, or checkpoint/run identity; the exact cause cannot be proved without the missing artifact. The safe conclusion is that the historical cross-stage localization pattern is not reproducible, not that the evaluator never produced the logged numbers.

## 2. Fable checks

### Antisymmetric-linear localization

The original ~84.5% weights were not saved, so the closest documented probe was reconstructed: raw/NoNorm candidate differences, layer-47 loop-boundary means for L1–L4 concatenated (8192 dimensions), bias-free score `w·(h_left-h_right)`, L-BFGS logistic fit, exact swap antisymmetry. All three backbones score 0.8375 on 80 held-out ordered rows (74 source pairs), source-pair-clustered CI [0.7349, 0.9351], with swap consistency 1.0. Features are not identical: Thinking–RLTT score Pearson is 0.9989 and Base–Thinking 0.9118, but decisions are essentially unchanged.

Caveat: the reconstructed historical protocol splits positive/negative orientations at row level, so opposite orientations of a source pair can cross train/evaluation. This is a failed targeted replication, not a definitive proof of equality across stages.

### Task-clustered pre-answer CI

Raw data were recovered for 170 tasks × 4 candidates = 680 examples (407 positive, 273 negative). Five-fold task-grouped out-of-fold predictions were recreated, then 170 task IDs were sampled with replacement in each of 10,000 paired bootstrap draws, retaining all four candidates for every sampled task occurrence. No draw was skipped.

- hidden+all AUROC: 0.796973
- length+logprob AUROC: 0.731183
- ΔAUROC: +0.065790
- task-clustered 95% CI: [+0.020707, +0.112254]
- bootstrap SD: 0.023464
- leave-one-task-out Δ range: [0.056000, 0.070557]

The original code path for the historical [+0.017, +0.114] interval is not fully preserved, but the fresh task-clustered recomputation independently verifies the significance claim.

## 3. HH strict swap audit

The pinned 512-pair artifact gives fixed-order 0.9668 and strict antisymmetric 0.6367 (95% CI [0.5957, 0.6777]). The new full-test audit evaluates all 8,552 HH test pairs in both orders with local `models/ouro_rltt_local`, `pairwise_epoch2.pt`, max length 768, bfloat16 backbone, float32 head, no quantization, and exact dataset indices.

- fixed-order accuracy: 0.947848 (95% CI [0.943054, 0.952526])
- swapped-direction accuracy: 0.195393
- strict antisymmetric accuracy: 0.639149 (95% CI [0.629093, 0.649322])
- strict sign-flip rate: 0.247544
- normal/flipped score correlation: -0.924682
- symmetric/antisymmetric absolute-magnitude ratio: 1.499648

The run used resume-safe inference segments recorded in JSON; all 8,552 rows are unique dataset indices. Model shard, evaluator, library, dtype, device, seed, and cache revision metadata are pinned in the full audit JSON.

## 4. Other empirical TODOs

### S3B2 native matched-random control

The eight oracle-present groups contain 10 candidates each, with positive counts 7, 1, 3, 2, 4, 2, 8, and 2. Uniform within-group random success therefore averages 0.3625 (expected 2.9/8), not 0.625 and not the inherited S3B1 macro 0.5833. The exact heterogeneous Bernoulli tail for at least five successes is 0.0869024. Report this as above random expectation but statistically unresolved at N=8.

### S1 K-matched control

Exact artifact values: four tasks; 12 plain samples/task; temperature 0.7; top-p 0.95; 96-token sampling budget (160-token long reference). Plain-sampling oracle is 0.750, sampled-fork screen mean is 0.611 (fork minus sampling −0.139), and greedy fork produces 0 new-correct tasks. Status: `FROZEN_FORK_CLOSED__SAMPLING_EXPLAINS_SCREEN__S3_IS_LEVER`.

### Steering closure

Exactly seven methods are enumerated in the consolidation artifact. Geometry values are adapter-vs-empirical −0.0042941, adapter-vs-raw −0.0005530, raw-vs-empirical +0.1010016, and independently trained adapter convergence +0.9510944. The draft's method count and cosine are verified.

### Second powered pre-answer domain

Status: `REJECTED_AT_PREFLIGHT_BOTH_CANDIDATES`; `CROSS_DOMAIN_VERDICT = PARTIAL_NOT_FULLY_ESTABLISHED`; action `STOP_AND_DOCUMENT`. SVAMP failed strict-cut validity and parser/label checks. MATH supplied long reasoning, but parseable generations were ~21/22 correct and failures were largely truncations, making the negative class either absent or a length artifact. No full recapture was warranted.

### Historical math-transfer origin

The 47/100 and ~2.7× values remain unarchived. No raw denominator table was recovered. They may remain only as explicitly non-load-bearing historical motivation; otherwise remove the exact multiplier.

## 5. Production TODOs

- Six requested figures/tables are generated under `figures/`: pre-answer AUROC, domain-transfer matrix, CoreContent/DualAnchor roles, cache/branch schematic, S3B2 detection-vs-selection, and the two-null orthogonality audit.
- All five explicit project-relative paths in the draft resolve. Of 23 backticked artifact-file tokens, all resolve by exact/suffix search except the genuinely absent companion `changelog.md`.
- Reproducibility state is pinned to Git HEAD `e4776dd41a85cad699ac36f309b5986ab48bd171` plus `git_status_short.txt`. No commit was created because the shared worktree has 1067 dirty entries.
- Markdown-to-LaTeX conversion is deferred: the instruction forbids modifying the draft, and v3.16 first needs the evidence patches below accepted.

## 6. Exact recommended wording

### §3.4 full audit

A strict swap audit on the full 8,552-pair HH-RLHF test set confirms that the original evaluator's high canonical-order accuracy is substantially order-artifacted. Fixed-order accuracy is 0.948, whereas strict antisymmetrized accuracy is 0.639 (95% pair-bootstrap CI [0.629, 0.649]). We therefore report the historical 95.2% only as fixed-order discovery-stage accuracy, not as relational discrimination.

### §3.7 replacement

In a targeted replication using a bias-free antisymmetric linear difference probe, the same held-out HH accuracy was obtained for base Ouro-2.6B, Ouro-2.6B-Thinking, and Ouro-RLTT (83.75% for each; exact swap consistency 1.0). Thus the earlier 24%/95.2%/95.0% separation is specific to an unrecovered historical fixed-order evaluation and does not establish training-stage localization in clean relational units. We therefore do not claim that reasoning/RL training installs the linearly readable relational preference signal. Because the original probe weights were unavailable and the reconstructed row split permits opposite orientations of a source pair to cross train and evaluation, resolving training-stage localization requires a larger independently pair-split antisymmetric replication.

### §5

At the strict pre-answer cut, adding hidden features to the length-and-log-probability baseline raises AUROC from 0.731 to 0.797 (incremental ΔAUROC +0.066). A paired task-clustered bootstrap over 170 GSM8K problems (10,000 draws; all four candidates retained per sampled problem) gives a 95% CI of [+0.021, +0.112], which excludes zero. The improvement is statistically significant under task-clustered resampling, while remaining a modest incremental effect demonstrated in one powered domain.

### §6.4 S3B2

Among the eight oracle-present S3B2 groups, the forced selector succeeds in 5/8 groups (0.625). Under uniform random choice within each group's ten-candidate pool, the group-specific success probabilities imply a matched expectation of 0.3625 (2.9/8 hits), not 0.625. The exact Poisson-binomial probability of at least five random successes is p=0.0869. The selector therefore exceeds the matched-random point expectation, but N=8 does not establish a statistically significant selection advantage; correctness is decodable, while reliable forced selection remains unestablished.

### §8.1 steering

Across seven pinned frozen-backbone intervention methods, no method produced reliable signed behavioral steering at safe magnitudes. The teacher-forced adapter proxy is nearly orthogonal to the empirical success direction (−0.0043) and raw NoNorm readout (−0.0006), while raw and empirical directions have cosine +0.1010. Independently trained teacher-forced and sequence-level adapters converge to cosine 0.951094, but the shared direction does not transfer to adapter-specific held-out free-generation control.

### §8.2 matched sampling

In the four-task K-matched control, plain stochastic decoding used 12 samples per task (temperature 0.7, top-p 0.95, 96-token budget) and reached oracle 0.750. Sampled forks reached 0.611 (−0.139 versus plain sampling), while deterministic/greedy forks produced zero new-correct tasks. Frozen branch injection therefore adds no demonstrated reachability beyond matched stochastic decoding in this bounded test.

### Second-domain limitation

Cross-domain pre-answer generality remains partial. A powered second-domain run was rejected at preflight: SVAMP front-loaded answers and failed parser/label checks, while parseable MATH generations were approximately 21/22 correct and the remaining failures were predominantly truncations, leaving no clean negative class. No full second-domain run was performed.

### Historical math-transfer note

The historical 47/100 Hendrycks MATH observation is retained only as explicitly unarchived, non-load-bearing motivation. Its raw denominator table and the exact single-shot baseline underlying the reported ~2.7× ratio were not recovered; the exact multiplier should be removed if publication policy requires every number to be artifact-reproducible.

## 7. Remaining blockers

- Original 84.5% probe weights and historical base=24% evaluation artifact were not recovered.
- The historical math-transfer denominator table and ~2.7x baseline artifact were not recovered.
- The powered second pre-answer domain has no clean accepted dataset after two failed preflights.
- The draft references a missing companion changelog.md.
- A publication commit was not created because the shared worktree contains 1067 pre-existing dirty entries; HEAD and the status manifest are pinned instead.
- Markdown-to-LaTeX conversion is deferred because the no-draft-modification constraint remains active and v3.16 contains claims now requiring evidence patches.
