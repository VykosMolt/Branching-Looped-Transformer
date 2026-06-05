# MMLU Science Branch + Parser Repair v3

Date: 2026-06-04
Output root: `artifacts/reports/probes/bg_mmlu_science_branch_parser_repair_v3_2026-06-01/`
Consolidated interpretation: `../../../science-reasoning-repair.md`

Narrow, source-specific follow-up to v2: can MMLU science (chemistry / anatomy / physics,
with SciQ as a control) be repaired enough to become a Phase 2b headline steering domain,
or must it stay excluded/diagnostic? Calibration was 131 recipe-task pairs (paused by user
at 66/131 on 2026-06-03, resumed 2026-06-04, completed 131/131, 0 errors); the full
downstream pipeline then ran with 0 stage failures.

## Status

`MMLU_SCIENCE_BRANCH_PARSER_REPAIR_V3_STATUS = SCIENCE_PARTIALLY_REPAIRED`

`BG_PRE_STEERING_DOMAIN_DECISION_V3_VERDICT = READY_FOR_STEERING_REASONING_PLUS_PARTIAL_SCIENCE`

`BG_MMLU_SCIENCE_V3_READINESS_VERDICT = SCIENCE_PARTIAL_HEADLINE_READY`

Science moved from v2's diagnostic-only to **partial** headline-readiness, driven entirely
by **MMLU anatomy**. Chemistry, physics, and SciQ stay excluded/diagnostic. Reasoning is
unaffected and still headline-ready.

No steering, Ouro training, tokenizer/checkpoint edit, tap-registry update, wrapper or
local-agent execution, Hunter-Seeker execution, ARC action loop, MATH generation,
production routing change, hard convergence-hair merge, compute-savings claim, or
autoregressive fork/carry claim was made. The robust parser stays diagnostic-only.

## Component verdicts

| Component | Verdict |
| --- | --- |
| Inventory | `MMLU_TASKS_LIMITED` |
| Task suite | `CHEM_ANATOMY_LIMITED` |
| Parser build | `READY` |
| Parser adversarial | `ROBUST_PARSER_STILL_DIAGNOSTIC` |
| Prompt format | `FORMAT_FIX_HELPS` |
| Recipe plan | `SOURCE_SPECIFIC_READY` |
| Recipe calibration | `SOURCE_SPECIFIC_RECIPE_FOUND` |
| Recipe heldout | `SCIENCE_RECIPE_IMPROVED_BUT_WEAK` |
| Source failure | `ANATOMY_REPAIRED` |
| L47 ablation | `SOURCE_SPECIFIC_L47` |
| Budget / breadth | `BUDGET8_SUFFICIENT` |
| Soft hair (no-good) | `CHEM_ANATOMY_NO_GOOD_CONFIRMED` |
| Reasoning guardrail | `REASONING_BASELINE_STILL_READY` |
| Readiness | `SCIENCE_PARTIAL_HEADLINE_READY` |
| Pre-steering decision | `READY_FOR_STEERING_REASONING_PLUS_PARTIAL_SCIENCE` |

## Heldout result (selected parser)

7 heldout tasks total (still data-limited). Overall `baseline_v3_regenerated`: positive-oracle
0.1429, terminal-best 0.0571, reward-diverse 0.0, terminal oracle retained 1.0 — up from v2's
0.0000 / -0.0571.

| Source | tasks | positive-oracle | terminal-best | parse | classification |
| --- | ---: | ---: | ---: | ---: | --- |
| `mmlu_anatomy` | 3 | 0.3333 | 0.2000 | 1.00 | REPAIRED |
| `mmlu_high_school_chemistry` | 2 | 0.0000 | 0.0000 | 0.00 | blocked + parse fail |
| `mmlu_high_school_physics` | 1 | 0.0000 | 0.0000 | 0.00 | blocked + parse fail |
| `sciq` | 1 | 0.0000 | -0.2000 | 1.00 | parses but wrong |

## Interpretation and caveats

- **Anatomy is the repair.** It was branch-generation-blocked (0.0) in v2 and now clears
  positive-oracle 0.333 / selected-parser terminal-best 0.20 with clean parsing — enough for
  the readiness stage to call partial headline-readiness and the decision stage to include
  partial science.
- **Chemistry and physics remain dead and now fail to parse** (parse_success 0.0 in heldout),
  so they are excluded; SciQ parses but is wrong. These are not promoted.
- **Small sample:** anatomy's result rests on 3 heldout tasks; treat "partial headline" as a
  candidate, not a settled headline.
- **Soft-hair still warns:** `CHEM_ANATOMY_NO_GOOD_CONFIRMED` — the convergence diagnostic
  flags chem+anatomy branches converging to no-good, so anatomy's gain is real but fragile.
- **Parser:** strict reward stays primary; robust letter+text parser remains diagnostic-only
  pending stronger false-positive validation.
- **Reasoning unaffected:** `REASONING_BASELINE_STILL_READY`; budget 8 remains sufficient
  (`BUDGET8_SUFFICIENT`); weak-source L47 emphasis helps source-specifically
  (`SOURCE_SPECIFIC_L47`).

## Primary artifacts

Under `artifacts/reports/probes/bg_mmlu_science_branch_parser_repair_v3_2026-06-01/`:
`summary.json` / `analysis.json`, `recipe_v3_calibration.json`, `recipe_v3_heldout.json`,
`source_failure_analysis.json`, `l47_v3_ablation.json`, `budget_breadth_v3.json`,
`soft_hair_no_good_v3.json`, `reasoning_guardrail.json`, `science_v3_readiness.json`,
`pre_steering_domain_decision_v3.json`, `selected_parser_v3.json`, `selected_prompt_format.json`.
