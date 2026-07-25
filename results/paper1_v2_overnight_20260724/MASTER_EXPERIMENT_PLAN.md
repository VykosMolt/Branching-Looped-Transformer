# Paper 1 v2 Overnight Programme — Master Experiment Plan

Programme root: `artifacts/reports/paper1_v2_overnight_20260724/`
Start time: see `ENVIRONMENT.json`. Hard wall-clock ceiling: 9 hours.

## Three experiments, priority order

1. **Horizon Logic strict pre-answer study** (`horizon_logic/`) — second-domain
   replication of the strict pre-answer protocol. Primary endpoint: paired
   task-clustered increment `AUROC(hidden+shortcuts) - AUROC(shortcuts)` on a heldout
   split opened once.
2. **Powered terminal-selection evaluation** (`terminal_selection/`) — forced top-1 vs.
   exact Poisson-binomial matched-random baseline on informative, task-disjoint,
   reward-diverse groups drawn from the same Horizon Logic pool (shared-pool design).
3. **Subspace-vs-subspace geometry audit** (`subspace_geometry/`) — bounded,
   inference-free diagnostic reusing the frozen `cross_loop_early_layer_taps_20260720/`
   artifact (COMPLETE — see `subspace_geometry/RESULTS.md`).

## Shared-pool decision

Per the spec's preference, one Horizon Logic generation pool (task source:
`data/branch_training_logic_expansion_v1/processed/logic_tasks.jsonl`, category
`synthetic_propositional`, proof_depth 2-4 as the reasoning-horizon control knob) is
used for both the pre-answer study and, if it turns out reward-diverse enough, the
terminal-selection evaluation. See `terminal_selection/RESULTS.md` for the actual
suitability decision made once the pool was generated.

## Protocol reuse (no new methodology invented)

- Generation/feature machinery: `src/evaluator/bg_transformer_features.py`
  (`BGTransformerFeatureExtractor`, canonical pooled `[3,4,2048]` features at layers
  `{24,36,47}` x loops `{1..4}`) and `utilities/tests/manual/bg_steering_suite_lib.py`
  (`mcq_prompt`, `parse_mcq_answer`).
- Strict pre-answer cut convention: text before the first `FINAL\s*ANSWE` marker, per
  `proto_introspection_preanswer_recapture.py`.
- Statistical fit family: standardize -> PCA -> L2-logistic, task-grouped 5-fold CV,
  AUROC + bootstrap CI, per `proto_introspection_controls_analysis.py` (imported and
  reused directly, not reimplemented).

## New scripts written this run (prefix `bg_v2_overnight_*`, all under
`utilities/tests/manual/`)

- `bg_v2_overnight_horizon_generate.py` — generation + feature extraction for the
  Horizon Logic pool (prompt_only and preanswer cuts).
- `bg_v2_overnight_horizon_analysis.py` — Part I AUROC analysis.
- `bg_v2_overnight_terminal_prep.py` — adds full-candidate ("terminal") pooled features
  to the same pool for Part II.
- `bg_v2_overnight_terminal_selection.py` — Part II selector + exact matched-random test.
- `bg_v2_overnight_geometry_audit.py` — Part III (complete).

## Known engineering issue encountered and fixed

The first pilot attempt hit CUDA OOM after task 1: `out` (containing all per-step
`output_scores` tensors from `model.generate`) was never dereferenced between loop
iterations, so GPU memory accumulated task-over-task until every subsequent generate()
call failed. Fixed by moving per-step logits to CPU immediately, deleting `out` inside
the per-task loop, and calling `torch.cuda.empty_cache()` after each task. Verified fixed
via an 8-task x 4-candidate smoke test with GPU memory returning to baseline afterward.
See `logs/horizon_pilot.log` for the original failure and the fixed rerun.

## Deliverables checklist

See `MASTER_RESULTS.md` for the final consolidated numbers and verdicts, and
`PAPER1_V2_INTEGRATION_PLAN.md` for what should/should not enter the paper.
