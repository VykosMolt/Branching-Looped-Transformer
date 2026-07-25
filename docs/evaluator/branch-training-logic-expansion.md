# Branch Training + Logic Expansion + Terminal v1

Updated: 2026-06-07 · Run: `branch_training_logic_expansion_terminal_v1` (Parts A–R)

`BRANCH_TRAINING_LOGIC_EXPANSION_STATUS = LOGIC_EXPANSION_READY_TRAINING_NOT_READY`
`BRANCH_TRAINING_POLICY_DECISION = KEEP_EXTERNAL_DUALANCHOR_CORECONTENT_BASELINE`

## Goal

First step from **external** branch selection (DualAnchor survival + CoreContent_v2 terminal
ranking) **toward model-internal branching** — teaching Ouro-RLTT to produce and manage diverse
solution branches itself, with **DualAnchor as a teacher, not a permanent crutch**. The explicit
primary deliverable is the **branch-training data + evaluation harness**; converged training is a
separate, multi-day effort. **External verifiers are the only ground truth**; DualAnchor/CoreContent
are teachers/baselines, never correctness labels; science diagnostic-only; steering not run.

## Primary deliverable (achieved)

- **Logic expanded** to **48,536 tasks (33,247 train / 11,017 heldout) across 10 verifier-backed
  families**: synthetic propositional (10k, truth-table), proof-depth deduction (13.4k incl. real
  ProofWriter), syllogisms (5k, finite-model), z3 constraint games (3k), + LogiQA/ReClor/RuleTaker/
  FOLIO/LSAT/logical-entailment. Synthetic items verified at generation.
- **Branch pools:** 33,350 cheap candidate groups (114,824 attempts) + **240 model-generated,
  domain-balanced pools** (Ouro-RLTT, externally labeled). `F=LOGIC_LABELS_READY`, `G=LEAKAGE_FOUND_FIXED`.
- **5 training views** (`K=TRAINING_DATA_READY`): branch_format_sft 177 · diversity_sft 240 ·
  policy_distillation 10,209 · final_self_selection 177 · verifier_reward_rl 240 · preference_dpo 194.

## Experiment 1 — integrated terminal (H = `CORECONTENT_IMPROVES_TERMINAL`)

Within real DualAnchor top-5 survivor sets (heldout, external labels):

| selector | top1 oracle |
| --- | ---: |
| oracle | 1.000 |
| **CoreContent_v2_blockwise** | **0.658** |
| mixedhead_MIX_HH_OBJECTIVE | 0.552 |
| DualAnchor forced-top1 | 0.379 |
| random survivor | 0.293 |

Survivor oracle retention **1.0** → **selection, not survival, is the terminal bottleneck**; the
composed external baseline (DualAnchor survival → CoreContent_v2 ranking → survivor handoff) is valid.

## Experiment 3 — reachability (I = `LOGIC_REACHABILITY_READY`)

Ouro-RLTT generated pools, externally labeled, positive-oracle@4: **reasoning 0.95 · math 0.83 ·
logic 0.73 · coding 0.43** (overall @1 0.60, @4 0.74, parse 0.955). **Math went 0.31 → 0.83** once
given a tool-free answer-forcing prompt (lifted from the local-agent's posture, *no tools/experts*)
+ 1400-token math budget + cheap early-stop + LaTeX/sympy verifier. Where @4 is high (logic/reasoning)
selection is the lever; where low, the generator is.

## DualAnchor as teacher (J = `TEACHER_MISMATCH_ON_LOGIC`)

Oracle-retention of DualAnchor pruning vs **random** (lift): reasoning +0.11, coding +0.11, math
+0.05, **logic +0.03**. → Useful branch-policy teacher except on logic ⇒ **logic branch quality must
come from verifier reward (N), not teacher distillation (M)**. These are policy labels, never correctness.

## Bounded training (L `SFT_TRAINED`, O `MODEL_INTERNAL_BRANCHING_PARTIAL`)

bf16 LoRA on Ouro-RLTT's UT arch (30.3M params, 1.12%), 300 steps, loss 0.34 — **proof-of-capability,
not converged**. Trained vs no-adapter, heldout:

| domain | base oracle@K | sft oracle@K | diversity base→sft | parse base→sft |
| --- | ---: | ---: | --- | --- |
| math | 0.75 | **0.92** | 1.9→2.7 | 1.0→1.0 |
| coding | 0.33 | **0.42** | 2.2→2.6 | **0.72→0.94** |
| logic | 0.83 | 0.75 | 2.4→2.6 | 0.97→0.97 |
| reasoning | 0.92 | 0.67 | 2.2→2.7 | 1.0→0.97 |
| **macro** | **0.708** | **0.688 (−0.02)** | **2.17→2.62 (+0.45)** | |

The SFT clearly **changed branching behavior** (+0.45 diversity, much cleaner coding output, +math/coding)
but **net reachability is flat** with reasoning/logic regressing — the classic 300-step/~600-example
LoRA signature. It does not beat the external baseline → **keep DualAnchor + CoreContent_v2**.

## Rigor / artifacts fixed by hand-audit

All sampled failures were verified as **genuine model errors** after fixing three artifacts the audits
surfaced: math-LaTeX false-negatives (`41π/4 == \frac{41\pi}{4}` via sympy), coding function-name
mismatch (inject the required `def` name → coding 0.05→0.43; a pre-screen proved **0 buggy MBPP tests**),
and a **hendrycks_math `task_uid` collision** (row indices repeat across 7 subjects → re-keyed by prompt-hash).

## Locked state & constraints

- Branch survival: **DualAnchor** (`MIX_CODE_REASONING` + `MIX_OBJECTIVE_ALL`) — unchanged (teacher + baseline).
- Content/final selection: **CoreContent_v2_blockwise_pruned_24_36** — unchanged (H confirmed it beats DualAnchor forced-top1 within survivors).
- Terminal: top5/full survivor-set handoff. Science: diagnostic only. Steering: not run/claimed.
- No Ouro base/tokenizer/checkpoint/registry overwrite; adapters only under `artifacts/models/branch_training_logic_expansion_v1/`; pure/transplanted/CoreContent_v2 artifacts untouched; no git.

## Next (separate run)

For a real internal-branching model: **M (DualAnchor teacher distillation) + N (verifier-reward RL) at
scale, to convergence** — the multi-day GPU job. J predicts M helps coding/reasoning and N is the lever
for logic. Reusable on disk: gen pools (`data/branch_training_logic_expansion_v1/processed/gen_shards/`),
SFT adapter, and the 5 training views (`…/train/`).

Artifacts: `artifacts/reports/probes/branch_training_logic_expansion_terminal_v1_2026-06-06/`.
