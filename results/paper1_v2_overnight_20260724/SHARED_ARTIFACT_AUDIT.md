# Shared Artifact Audit — Paper 1 v2 Overnight Programme

Generated at programme start (`ENVIRONMENT.json` has exact timestamp). Read-only audit;
nothing in this section modifies any existing artifact.

## Frozen early-layer geometry input (`cross_loop_early_layer_taps_20260720/`)

- `SHA256SUMS`: **100/100 entries verified** against on-disk files (`ok=100 bad=0 missing=0`).
- `split_integrity.json`: `zero_crossing_count = 0` across train/val/heldout
  (1250/300/600 groups, 8592 candidates total). Split is the existing deterministic
  v2 task/prompt-disjoint split — no new split constructed here.
- `feature_manifest.json`: layers `[8, 16, 24, 36, 47]` x loops `[1,2,3,4]`, hidden 2048,
  fp16 storage / fp32 pooling compute, 18 shards, 2150 groups / 8592 candidates.
- Confirmed frozen: this run only **reads** `features/`, `predictions/refit_*.json`,
  `s3b2_secondary.json`, `xloop_taps.pt`. No retraining, no rewriting.

## Horizon Logic protocol discovery

No literal "Horizon Logic" artifact exists under any name (`grep -ri horizon.logic` across
the repo returns nothing). What exists instead, and what this programme uses:

- `data/branch_training_logic_expansion_v1/processed/logic_tasks.jsonl` — **48,536** logic
  tasks across 10 verifier-backed families (`proofwriter_deduction` 13400,
  `synthetic_propositional` 10000, `mcq_logical_reading` 9802, `synthetic_fol` 5000,
  `logical_entailment` 3162, `synthetic_constraint_game` 3000, `ruletaker_deduction` 2621,
  `fol_entailment` 811, `lsat_logical_reasoning` 510, `lsat_analytical_reasoning` 230).
  Each record carries `task_uid`, `task_prompt`, `options`, `answer_key`, `gold_answer`,
  a **deterministic verifier** (`truth_table` / `forward_chaining` / `finite_model_checking`
  / `z3`), and a `proof_depth` field usable as the reasoning-**horizon** control knob.
  `synthetic_propositional` has proof_depth 1–4 (n=3437/4094/1994/475) with a genuine
  True/False/Unknown label mix (not front-loadable MCQ, not gold-collapsed like
  `proofwriter_deduction`, whose forward-chaining construction always yields gold=`True`
  and is therefore unsuitable alone as a pre-answer success/failure domain).
- `utilities/tests/manual/branch_training_logic_v1_common.py` — the generator source for
  the above (kept for reference; not re-run, since the task pool already exists on disk).
- `docs/evaluator/proto_introspection_second_domain_preflight_2026-06-17.md` (via
  `artifacts/reports/proto_introspection/`) — **critical prior lesson**: SVAMP and MATH
  were both tried and rejected as GSM8K's second domain. SVAMP front-loads answers
  (median pre-answer 3 tokens in 4/8 probes); MATH's failure mode is truncation, not wrong
  answers, so it yields no negative class. The usable-domain criterion established there
  ("answer-last with real reasoning" + "model commits a parseable answer in a balanced
  mix") is the design target for Horizon Logic's task/depth selection and is re-verified
  empirically in the pilot rather than assumed.
- `utilities/tests/manual/proto_introspection_preanswer_recapture.py` +
  `proto_introspection_controls_analysis.py` — the **audited canonical strict pre-answer
  protocol** (cut = text before `FINAL\s*ANSWE` marker, feature = flattened pooled
  `[3,4,2048]` hidden state at layers `{24,36,47}` x loops `{1..4}`, fit family =
  standardize -> PCA(k=24) -> L2-logistic, grouped-by-task 5-fold OOF CV, AUROC + 1000-round
  bootstrap CI). This programme reuses this fit family and cut convention verbatim for
  Horizon Logic, rather than inventing a new methodology.
- `src/evaluator/bg_transformer_features.py` (`BGTransformerFeatureExtractor`) and
  `utilities/tests/manual/bg_steering_suite_lib.py` (`OuroTextGenerator`, `mcq_prompt`,
  `parse_mcq_answer`, `evaluate_output`) — the shared generation/feature-extraction
  machinery reused unmodified for Horizon Logic generation.

## Terminal-selection candidate pools

- `docs/evaluator/terminal-selection-and-arbiters.md`: current policy is
  `ARCHITECTURE_LOOPED_SURVIVAL_READY_TERMINAL_DEFER_REQUIRED` — terminal collapse is the
  known weak point; DualAnchor architecture-looped v3 numbers (forced top1 oracle 0.9167,
  forced top1 reward 0.2625) are the most relevant prior baseline for what "terminal
  selection is hard" looks like on this project's own domains.
- Given the shared-pool preference in the spec, the Horizon Logic candidate pool (4
  candidates/task, external truth-table/forward-chaining verifier labels) is evaluated
  directly for terminal-selection suitability (informative-group yield) before falling
  back to any separate DualAnchor/CoreContent pool. See `terminal_selection/RESULTS.md`
  for the suitability decision actually made.

## Writable/injection tensors (for the geometry audit)

- `artifacts/reports/proto_introspection/s1_s3_exact_injection_orthogonality_2026-06-17/
  s1_s3_exact_injection_delta_bundle_2026-06-17.pt` is the only real observed
  perturbation/injection delta bundle found in the repo (`delta_locus` tensor,
  `[344, 2048]`, `EXACT_PROTOCOL_REGENERATED` status).
- Its `protocol.loci` list is the cross product of loop `{1,2,3,4}` x physical layer
  `{24, 36, 47}` — **12 late loci, zero early loci**. This matches
  `cross_loop_early_layer_taps_20260720/ARTIFACT_AUDIT.md`'s own conclusion: "S3B2 and
  branch-survival packs exist only at 24/36/47 loci... deferred behind the primary matrix."
- **No writable/perturbation tensor exists anywhere in the repo at physical layer 8 or 16**
  (confirmed by grep across `artifacts/` and `data/` for early-layer injection/carry
  artifacts, and by the loci list above). Per the programme's own rule ("do not open a new
  extraction or intervention campaign... abandon rather than expand"), the early-vs-late
  **writable**-subspace comparison is therefore reported as
  `INSUFFICIENT_MATCHED_LOCUS_WRITABLE_DATA` rather than fabricated or transported across
  coordinate systems. The readable<->outcome overlap and loop-to-loop rotation analyses,
  which need only the frozen readable/outcome artifacts (available at all five layers),
  are still run in full — see `subspace_geometry/RESULTS.md`.
