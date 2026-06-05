# Science / Reasoning Repair (pre-steering domain decision)

Updated: 2026-06-04

This is the consolidated pre-steering domain-readiness decision for the DualAnchor
architecture-looped baseline: **is reasoning and/or science ready to be a headline
steering-comparison domain?** It merges the v3 MMLU repair (2026-06-04) and the v2
decision (2026-06-01) with their two v1 precursors (2026-05-31). Exact run notes are
archived under `history/bg-run-notes/science-reasoning/`.

## Bottom line (current — v3, 2026-06-04)

`MMLU_SCIENCE_BRANCH_PARSER_REPAIR_V3_STATUS = SCIENCE_PARTIALLY_REPAIRED`

`BG_PRE_STEERING_DOMAIN_DECISION_V3_VERDICT = READY_FOR_STEERING_REASONING_PLUS_PARTIAL_SCIENCE`

`BG_MMLU_SCIENCE_V3_READINESS_VERDICT = SCIENCE_PARTIAL_HEADLINE_READY`

- **Reasoning** unchanged: headline-ready under the locked terminal handoff
  (`REASONING_BASELINE_STILL_READY`).
- **Science is now partially repaired** — **MMLU anatomy** crosses into partial
  headline-readiness (heldout positive-oracle 0.333, selected-parser terminal-best 0.20,
  parse 1.0 on 3 tasks; `ANATOMY_REPAIRED`). **Chemistry, physics, and SciQ stay
  excluded/diagnostic** (heldout 0.0; chemistry/physics parse collapsed to 0.0). Overall
  heldout is `SCIENCE_RECIPE_IMPROVED_BUT_WEAK`.
- **Caveats:** tiny heldout (7 tasks; anatomy = 3); robust parser stays diagnostic-only
  (`ROBUST_PARSER_STILL_DIAGNOSTIC`); the soft-hair convergence diagnostic still warns
  chem+anatomy converge to no-good (`CHEM_ANATOMY_NO_GOOD_CONFIRMED`) — so anatomy's gain
  is real but fragile. Do not promote chemistry, physics, or SciQ.

This supersedes the **science** half of the v2 decision below (diagnostic-only → partial
headline for anatomy). The reasoning half is unchanged.

## Prior bottom line (v2, 2026-06-01)

`DUALANCHOR_SCIENCE_REASONING_REPAIR_V2_STATUS = REASONING_READY_SCIENCE_DIAGNOSTIC`

`BG_PRE_STEERING_READINESS_V2_VERDICT = READY_FOR_STEERING_WITH_SCIENCE_DIAGNOSTIC`

- **Reasoning** is headline-ready, but **only** under the locked terminal handoff
  policy (confidence-gated top1, else top5/full survivor-set handoff) — not under
  unconditional terminal top1.
- **Science** stayed diagnostic / excluded from headline steering. Regenerated
  branch-recipe calibration found some recipe signal, but heldout did not validate a
  science repair (branch generation remained weak). The v3 run below re-tested this
  source-specifically and partially cleared it for anatomy.

No steering, Ouro training, tokenizer/checkpoint edit, tap-registry update, wrapper or
local-agent execution, Hunter-Seeker execution, ARC action loop, production routing
change, hard convergence-hair merge, compute-savings claim, or autoregressive
fork/carry claim was made in this work.

## v3 MMLU repair verdicts (2026-06-04)

Source-specific MMLU follow-up (chemistry / anatomy / physics; SciQ control). Calibration
131/131 (0 errors), full downstream pipeline 0 stage failures. Run note:
`history/bg-run-notes/science-reasoning/bg_mmlu_science_branch_parser_repair_v3.md`;
artifacts under `artifacts/reports/probes/bg_mmlu_science_branch_parser_repair_v3_2026-06-01/`.

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

Heldout per source (selected parser, 7 tasks total): `mmlu_anatomy` 3 tasks → positive-oracle
0.333 / terminal-best 0.20 / parse 1.0 (REPAIRED); `mmlu_high_school_chemistry` 2 → 0.0 / 0.0 /
parse 0.0; `mmlu_high_school_physics` 1 → 0.0 / 0.0 / parse 0.0; `sciq` 1 → 0.0 / -0.20 / parse
1.0. Overall up from v2 (0.0 / -0.0571 → 0.143 / 0.057), driven by anatomy.

## v2 component verdicts (2026-06-01)

| Component | Verdict |
| --- | --- |
| Inventory | `PARTIAL` |
| Task suite | `SCIENCE_LIMITED` |
| Parser candidates | `READY` |
| Parser validation | `PARSER_PATCH_DIAGNOSTIC_ONLY` |
| Science recipe plan | `SOURCE_SPECIFIC_READY` |
| Science recipe calibration | `SCIENCE_RECIPE_FOUND` |
| Science heldout | `SCIENCE_BRANCH_GENERATION_STILL_WEAK` |
| Source-specific science | `MMLU_CHEM_ANATOMY_BLOCKED` |
| Parser recommendation | `ROBUST_DIAGNOSTIC_ONLY` |
| Reasoning terminal handoff | `REASONING_HANDOFF_LOCKED` |
| Reasoning hard slice | `REASONING_READY_WITH_HANDOFF` |
| Science L47/layer ablation | `L2_47_HELPS` |
| Perturbation escalation | `ESCALATION_NO_HELP` |
| Soft hairs | `SCIENCE_NO_GOOD_WARNING_USEFUL` |
| Integrated repair | `SCIENCE_STILL_BLOCKED` |

### Science result

Source-limited suite: 31 science tasks selected, only 7 science heldout tasks
(OpenBookQA and biology missing from the requested source set). Calibration found
source-specific recipe signal (`L2_47_emphasis` and `L47_heavy`: positive-oracle
0.2500 on 4 tasks; `baseline_v3_regenerated`: 0.1667 on 12 tasks), but heldout did
**not** validate it (selected-parser positive-oracle 0.0000, terminal best reward
-0.0571 on 7 tasks; terminal oracle retained 1.0000). Per-source: `mmlu_anatomy`,
`mmlu_high_school_chemistry` → `BRANCH_GENERATION_BLOCKED`; physics, sciq →
`DATA_LIMITED`. The parser patch stays diagnostic-only — robust parsing must not
replace strict reward without stronger false-positive validation.

### Reasoning result

Terminal handoff is locked. On 24 reasoning tasks, oracle is fully retained under
confidence-gated, top5, and defer-all policies (defer rate 0.9583, first-selected
oracle 0.8750); forced terminal top1 drops to 0.8750 oracle. So reasoning is
headline-ready **with** the handoff, not under unconditional terminal top1.

### Locked baseline (for any steering comparison)

- schedule: `L1_24 -> L1_36 -> L1_47 -> L2_24 -> L2_36 -> L2_47 -> L3_24 -> L3_36 -> L3_47 -> L4_24 -> L4_36 -> terminal L4_47`
- selector: `MIX_CODE_REASONING + MIX_OBJECTIVE_ALL`; threshold `mean_floor_very_loose`; budget `8`
- L47: active in nonterminal loops; terminal: confidence-gated top1 else top5/full survivor-set handoff
- convergence hairs: soft-only monitoring; reasoning: headline; science: diagnostic/excluded; steering: not run

## v1 precursors (2026-05-31)

These two probes set up the v2 decision and are archived in full under
`history/bg-run-notes/science-reasoning/`.

**Convergence hairs + reasoning/science pre-steering probe v1**
(`bg_dualanchor_convergence_hairs_reasoning_science_v1`): established that science
ties are not a good branch (`SCIENCE_TIES_ARE_NO_GOOD_BRANCH`) and that reasoning
needs terminal defer (`REASONING_NEEDS_TERMINAL_DEFER`). Convergence-hair dataset and
policy definitions were `READY`, but replay-eval was `INSUFFICIENT` and the
regenerated hair was `SKIPPED` — motivating the v2 regeneration.

**Science branch recipe + reasoning terminal defer v1**
(`bg_dualanchor_science_branch_recipe_reasoning_defer_v1`): found the science parser
partly responsible for weak reward (`PARSER_PARTLY_RESPONSIBLE`,
`SCIENCE_PARSER_DOMINATES`), the recipe direction limited (`DIRECTION_LIMITED`), and
heldout science branch generation still weak (`SCIENCE_BRANCH_GENERATION_STILL_WEAK`,
`MMLU_SCIENCE_WEAK`). This is the result v2 re-tested with source-specific recipes
and still could not clear.

## Source run notes

Archived (exact provenance) under `history/bg-run-notes/science-reasoning/`:

- `bg_mmlu_science_branch_parser_repair_v3.md` (v3 decision, 2026-06-04; primary artifacts in
  `artifacts/reports/probes/bg_mmlu_science_branch_parser_repair_v3_2026-06-01/`)
- `bg_dualanchor_science_reasoning_repair_v2.md` (v2 decision; primary artifacts in
  `artifacts/reports/probes/bg_dualanchor_science_reasoning_repair_v2_2026-06-01/`)
- `bg_dualanchor_convergence_hairs_reasoning_science_v1.md`
- `bg_dualanchor_science_branch_recipe_reasoning_defer_v1.md`

The `mmlu_science_branch_parser_repair_v3` run is now complete: MMLU anatomy is partially
repaired and a candidate partial/secondary science headline; chemistry, physics, and SciQ
remain excluded/diagnostic.
