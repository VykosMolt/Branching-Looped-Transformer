# Read-only BG Controller Usage

The BG controller is a read-only pairwise branch-selection layer. It loads existing trained heads and scores already-captured pooled candidate features.

It is not a reward model, not model steering, not generation, not a verifier replacement, and not a candidate feature capture path. Heads produce pairwise branch-selection evidence only; candidate labels are for replay/evaluation and are never used for routing.

## Heads

- `hh_general`: HH-trained `47_concat_L1_L4 / AntisymLinearNoNorm`; used for HH, preference, and unknown/default semantic preference routing.
- `objective_mixed`: MIX_CODE_REASONING `36_L4 / AntisymLinearNoNorm`; default objective selector for code, strict-clean code, reasoning, science, math, GSM8K, and objective routing.
- `code_specialist_backup`: code-trained `36_L4 / AntisymLinear`; backup and diagnostic head, not the conservative default route.

## Modes

- `conservative`: default v8.1 policy.
- `experimental_vote`: validation/research mode using HH and objective mixed normalized margins.
- `code_backup`: explicit code specialist backup scoring.
- `diagnostic_all`: returns scores from all three locked heads.

## Domain Hints

Supported normalized hints:

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

Conservative routing:

- `hh`, `preference`, `unknown` -> `hh_general`
- `code`, `strict_clean_code`, `reasoning`, `science`, `math`, `gsm8k`, `objective` -> `objective_mixed`

## Example

```python
from src.evaluator.bg_controller import BGController

controller = BGController.from_artifacts(device="cpu")

result = controller.select_best(candidate_features, domain_hint="code")
result_vote = controller.select_best(
    candidate_features,
    domain_hint="code",
    mode="experimental_vote",
)
```

`candidate_features` must be a list of tensors or a tensor batch shaped `[N, 3, 4, 2048]`. Each candidate pooled tensor is `[layers=3, loops=4, hidden=2048]`, with layer positions corresponding to 24, 36, and 47.

## Caveats

- `experimental_vote` is not the default. It is exposed for validation and research.
- Contrast routing is not implemented.
- Deferral routing is not implemented.
- Medical/science benchmark behavior is benchmark MCQ transfer only, not clinical validation.
- Old dirty artifacts should not be used as controller inputs.
- The controller selects among existing branches; it does not treat a single branch score as standalone quality.

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
- generator reachability result: code/devil branches were limited under direct Ouro; reasoning/science/GSM8K were reachable.
- devil task result: no passing devil branch was available for BG to select.
- full report paths: `artifacts/reports/probes/bg_steering_suite_2026-05-18/summary.md`, `artifacts/reports/probes/bg_steering_suite_2026-05-18/analysis.md`, `docs/evaluator/steering-and-routing-suite.md`
- interpretation: use BGController as a read-only selector; this suite does not support treating it as a solver or as a hidden-state steering mechanism.
