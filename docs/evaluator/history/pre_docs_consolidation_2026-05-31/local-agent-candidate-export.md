# Local-Agent Candidate Export

## Purpose

The local-agent candidate export interface exposes wrapper branch artifacts for later BG wrapper-matched experiments. It is an opt-in trace path for candidates that the wrapper already creates: direct answers, tool inputs, failed tool attempts, repairs, and grounded final code.

This does not run BG, select candidates, assign labels, or change the wrapper's normal output.

## Schema

The lightweight schema lives in `src/local_agent/candidate_export.py`.

- `CandidateArtifact`: one branch-like artifact with raw text, raw tool input, sanitized code, final code, tool status, stage, and metadata.
- `CandidateTrace`: a per-task trace containing the prompt hash, route, candidates, selected wrapper candidate if known, and final answer.

The schema deliberately does not import Torch, Transformers, or model code.

## Candidate Stages

Supported stage strings:

- `direct_final`
- `direct_short_budget`
- `sampled_direct`
- `first_tool_code`
- `first_failed_tool_code`
- `first_repair_code`
- `repaired_final`
- `final_grounded_code`
- `tool_verified_code`
- `rejected_malformed`
- `rejected_wrapper_status`
- `unknown`

## API Examples

Direct route:

```python
from src.local_agent.candidate_capture import run_direct_with_candidates

trace = run_direct_with_candidates(
    prompt="Write a Python function add_one(x) that returns x + 1.",
    task_id="example/add_one",
    model_mgr=model_mgr,
)
```

Agent/tool route:

```python
from src.local_agent.candidate_capture import run_agent_with_candidates

trace = run_agent_with_candidates(
    prompt="Write a Python function add_one(x) that returns x + 1.",
    task_id="example/add_one",
    model_mgr=model_mgr,
    project=project,
    history=[],
)
```

Direct wrapper opt-in:

```python
answer, trace = direct_answer(
    model_mgr,
    prompt,
    return_candidate_trace=True,
    task_id="example/add_one",
)
```

Agent wrapper opt-in:

```python
answer, trace = run_agent_task(
    model_mgr,
    project,
    history,
    prompt,
    return_candidate_trace=True,
    task_id="example/add_one",
)
```

## Safety Notes

Default behavior is unchanged. The wrapper returns its original string result unless `return_candidate_trace=True` or an explicit `CandidateTrace` is supplied.

Candidate traces store tool success and observations for diagnostics, but those fields must not be used as BG labels or routing inputs. BG wrapper-matched experiments should capture candidate text/code only for feature extraction and run tests only after selection.

Tool safety checks, sanitization, runtime guards, timeouts, and hard-code policies remain in the existing wrapper path.

## BG Experiment Use

The next BG wrapper-matched experiment can use the exported candidate set as a shared pool:

1. Run the normal wrapper with candidate trace enabled.
2. Extract candidate artifacts from the trace.
3. Build BG feature input from prompt plus candidate text/code only.
4. Compare wrapper baseline, random candidate, and BG-selected candidate over the same candidate set.
5. Run unit tests or verifiers only after candidate selection for evaluation.

## Non-Goals

This interface does not:

- change wrapper behavior by default,
- run BG,
- use labels,
- select candidates,
- change prompts for generation,
- train or modify models,
- bypass wrapper safety policies.

## Wrapper-matched BG candidate selection (2026-05-18)

- WRAPPER_TRACE_GENERATION_VERDICT = READY
- WRAPPER_CANDIDATE_EVAL_VERDICT = READY
- WRAPPER_GENERATOR_REACHABILITY_VERDICT = REACHABLE
- WRAPPER_BG_FEATURE_VERDICT = READY
- WRAPPER_MATCHED_BG_VERDICT = NEUTRAL
- BG_VS_RANDOM_VERDICT = NEUTRAL
- BG_VS_STAGE_HEURISTIC_VERDICT = NEUTRAL
- WRAPPER_MATCHED_EXPERIMENT_VERDICT = READY
- devil result: both local devil tasks remained unsolved, with wrapper and BG selecting `wrong_code`.
- report paths: `artifacts/reports/probes/wrapper_bg_matched_2026-05-18/summary.md`, `artifacts/reports/probes/wrapper_bg_matched_2026-05-18/analysis.md`, `docs/evaluator/wrapper-matched-bg-selection.md`
- interpretation: the candidate export interface worked for a matched BG experiment; the result supports logging/calibration rather than enabling BG as a wrapper reranker yet.
