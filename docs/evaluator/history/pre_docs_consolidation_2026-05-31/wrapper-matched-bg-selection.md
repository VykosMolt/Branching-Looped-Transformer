# Wrapper-Matched BG Candidate Selection

## Purpose

This experiment compares the local-agent wrapper's normal final output against BG selection over the same exported wrapper candidate trace. It tests BG as a selector over wrapper-quality code candidates, not as a generator, solver, verifier, or repair loop.

## Fairness Constraints

- Wrapper baseline and BG use the same generated candidate trace.
- BG selection uses only prompt plus candidate code-like text.
- Unit-test labels are computed only after trace generation and feature capture.
- Random and stage-heuristic baselines sample or choose from the same trace.
- Oracle is post-hoc only.
- Wrapper final candidates must be present in the trace for headline comparison.

No wrapper behavior, prompts, model weights, tokenizers, checkpoints, or evaluator heads were modified.

## Candidate Export Dependency

The experiment uses:

- `src/local_agent/candidate_export.py`
- `src/local_agent/candidate_capture.py`
- `src/local_agent/ouro_agent_improved.py` with `return_candidate_trace=True`

Candidate traces include direct/tool/repair/status artifacts already produced by the wrapper.

## Candidate Stages

Observed stages:

- `final_grounded_code`
- `first_tool_code`
- `first_failed_tool_code`
- `first_repair_code`
- `rejected_wrapper_status`

For BG features, candidates were filtered to code-like text in this order:

1. `final_code`
2. `sanitized_code`
3. code-like `raw_tool_input`
4. extractable code from `raw_text`

## BG Policies

Primary:

- `BG_CONSERVATIVE`, domain hint `code`

Diagnostics:

- `BG_CODE_BACKUP`
- `BG_EXPERIMENTAL_VOTE`
- `DIAGNOSTIC_ALL`

Baselines:

- `WRAPPER_FINAL`
- `RANDOM_CANDIDATE`
- `STAGE_HEURISTIC_FINAL`
- `ORACLE_CANDIDATE`

## Results

- WRAPPER_TRACE_GENERATION_VERDICT = READY
- WRAPPER_CANDIDATE_EVAL_VERDICT = READY
- WRAPPER_GENERATOR_REACHABILITY_VERDICT = REACHABLE
- WRAPPER_BG_FEATURE_VERDICT = READY
- WRAPPER_MATCHED_BG_VERDICT = NEUTRAL
- BG_VS_RANDOM_VERDICT = NEUTRAL
- BG_VS_STAGE_HEURISTIC_VERDICT = NEUTRAL
- WRAPPER_ORACLE_GAP_VERDICT = SMALL
- WRAPPER_MATCHED_EXPERIMENT_VERDICT = READY

Headline metrics:

- matched evaluable tasks: `15`
- wrapper final pass rate: `0.400`
- BG conservative matched pass rate: `0.400`
- BG conservative non-devil pass rate: `0.400`
- random expected pass rate: `0.400`
- stage heuristic pass rate: `0.400`
- oracle reachability rate: `0.400`

## Interpretation

Wrapper-quality candidate generation was materially more reachable than direct Ouro code generation, but the correct candidates were mostly cases where the wrapper, BG, random expectation, and stage heuristic all selected or had access to the same successful branch. BG did not improve the wrapper final choice in this matched run.

The wrapper oracle gap was small on the matched set, so there were few opportunities for BG to rescue a missed correct branch.

## Devil Tasks

Both devil tasks produced code-like wrapper candidates, but none passed the local tests:

- `offline_dynamic_connectivity`: wrapper `wrong_code`, BG `wrong_code`
- `minimum_xor_paths`: wrapper `wrong_code`, BG `wrong_code`

These remain generator/repair bottlenecks, not BG-selection wins or losses.

## What Remains Open

- Generate more diverse wrapper candidate sets where the wrapper final is not already the oracle candidate.
- Add a matched repair-candidate pool with multiple independently viable branches.
- Use BG for logging/calibration before making it an optional wrapper reranker.
- Re-run on larger N after ensuring candidate diversity, not just duplicated `first_tool_code`/`final_grounded_code` pairs.

## Reports

- `artifacts/reports/probes/wrapper_bg_matched_2026-05-18/summary.md`
- `artifacts/reports/probes/wrapper_bg_matched_2026-05-18/analysis.md`
- `artifacts/reports/probes/wrapper_bg_matched_2026-05-18/bg_selection.md`
