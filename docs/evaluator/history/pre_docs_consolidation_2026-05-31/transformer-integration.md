# Read-only BG Transformer Integration

This layer connects live Ouro-RLTT candidate generation and hidden-state capture to the read-only BG controller. It generates bounded candidate completions, captures transformer states, pools features into `[layers=3, loops=4, hidden=2048]`, and selects among candidates with `BGController`.

It differs from cached controller replay: replay scores existing saved features, while this path captures fresh live model features for prompt + candidate text.

## Feature Format

- Model: `/home/moloch/ouro_project/models/ouro_rltt_local`
- Layer order: `[24, 36, 47]`
- Loop order: `[L1, L2, L3, L4]`
- Pooling: masked mean over valid input tokens
- Returned dtype/device: CPU `float32`
- Candidate feature shape: `[3, 4, 2048]`

Feature input text is deterministic:

```text
Prompt:
{prompt}

Candidate:
{candidate}
```

No labels, verifier outputs, selected index, BG scores, tests, logs, or chain-of-thought tags are included in the feature text.

## How To Run

```bash
venv/bin/python -u utilities/tests/manual/run_bg_transformer_best_of_n_smoke.py \
  --prompt "Write a Python function `is_palindrome(s)` that returns True if the string is a palindrome ignoring spaces and case." \
  --domain-hint code \
  --n-candidates 3 \
  --max-new-tokens 192 \
  --temperature 0.7 \
  --top-p 0.95 \
  --mode conservative \
  --device cuda \
  --output artifacts/reports/probes/bg_transformer_best_of_n_smoke_code_2026-05-18.json
```

The harness writes JSON and Markdown traces. It also writes a rolling aggregate at:

- `artifacts/reports/probes/bg_transformer_best_of_n_smoke_all_2026-05-18.json`
- `artifacts/reports/probes/bg_transformer_best_of_n_smoke_all_2026-05-18.md`

## Supported Domain Hints

- `hh`
- `preference`
- `unknown`
- `code`
- `strict_clean_code`
- `reasoning`
- `science`
- `math`
- `gsm8k`
- `objective`

Conservative mode is the default. It routes `hh`, `preference`, and `unknown` to `hh_general`; objective domains route to `objective_mixed`, including `strict_clean_code`.

`experimental_vote` is available only as a diagnostic/validation mode. It is not the default route.

## Devil Task Caveat

The devil code task smoke uses existing local prompts and tests when found. Test results are diagnostic only and never affect BG selection. Direct Ouro generation may be much weaker than the local-agent wrapper on hard code prompts, so failing devil tests does not imply a controller integration failure.

## Not Implemented

- hidden-state steering
- latent branching
- contrast routing
- deferral
- local-agent wrapper integration
- model editing or checkpoint modification
- training or evaluator replacement

## Known Risks

- The capture path is model-version-specific because it depends on Ouro-RLTT loop outputs and decoder-layer hook positions.
- Token budget can be consumed by verbose completions.
- CUDA memory depends on candidate count and max-new-token budget.
- `experimental_vote` still depends on approximate margin calibration.
- Direct Ouro generation is not expected to solve hard local code tasks reliably without the existing agentic repair path.

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
- generator reachability result: non-code objective branches were often reachable; code/devil branches remained limited without wrapper repair.
- devil task result: no passing devil branch appeared in the early reachability gate.
- full report paths: `artifacts/reports/probes/bg_steering_suite_2026-05-18/summary.md`, `artifacts/reports/probes/bg_steering_suite_2026-05-18/analysis.md`, `docs/evaluator/steering-and-routing-suite.md`
- interpretation: live feature capture is stable enough for routing experiments, but better candidate generation or wrapper-exposed candidate sets are needed before stronger BG deployment claims.

## Local-agent candidate export interface (2026-05-18)

- WRAPPER_CANDIDATE_PATH_INVENTORY_VERDICT = READY
- WRAPPER_CANDIDATE_EXPORT_UNIT_VERDICT = PASS
- WRAPPER_CANDIDATE_EXPORT_SMOKE_VERDICT = SKIPPED
- WRAPPER_CANDIDATE_EXPOSURE_VERDICT = READY
- files modified: `src/local_agent/candidate_export.py`, `src/local_agent/candidate_capture.py`, `src/local_agent/ouro_direct.py`, `src/local_agent/ouro_agent_improved.py`
- API path: `src/local_agent/candidate_capture.py`
- report path: `artifacts/reports/probes/local_agent_candidate_exposure_2026-05-18_summary.md`
- interpretation: transformer BG selection can now be paired with wrapper-produced candidate artifacts in a later matched experiment, rather than relying only on direct Ouro generation.

## Wrapper-matched BG candidate selection (2026-05-18)

- WRAPPER_TRACE_GENERATION_VERDICT = READY
- WRAPPER_CANDIDATE_EVAL_VERDICT = READY
- WRAPPER_GENERATOR_REACHABILITY_VERDICT = REACHABLE
- WRAPPER_BG_FEATURE_VERDICT = READY
- WRAPPER_MATCHED_BG_VERDICT = NEUTRAL
- BG_VS_RANDOM_VERDICT = NEUTRAL
- BG_VS_STAGE_HEURISTIC_VERDICT = NEUTRAL
- WRAPPER_MATCHED_EXPERIMENT_VERDICT = READY
- devil result: no devil candidate was correct; both wrapper final and BG selected wrong-code candidates.
- report paths: `artifacts/reports/probes/wrapper_bg_matched_2026-05-18/summary.md`, `artifacts/reports/probes/wrapper_bg_matched_2026-05-18/bg_selection.md`, `docs/evaluator/wrapper-matched-bg-selection.md`
- interpretation: live BG feature capture works on exported wrapper candidates, but the current wrapper candidate pools produced a neutral selector result.
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

