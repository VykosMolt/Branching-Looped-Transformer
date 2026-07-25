# Interfaces And Tools

Updated: 2026-05-31

This is the consolidated entry point for tap interfaces, transformer integration notes, local tooling notes, and low-level evaluator mechanics.

## Tap Interface

The evaluator taps are relational pairwise heads over hidden-state features. Treat them as pairwise evaluators, not absolute scorers.

Labels must come from external sources:

- HH chosen/rejected labels,
- answer keys,
- exact-answer verifiers,
- unit tests,
- parser/verifier labels.

Tap scores are features or decisions, not labels.

## Current Layer/Loop Context

The active DualAnchor architecture uses taps at:

- layer 24,
- layer 36,
- layer 47,
- loops L1-L4.

Terminal collapse is only considered at final `L4_47`, and even there only through confidence gating.

## Cumulative-Hook Boundary

Prompt-only decoder-layer carry equivalence was validated for cumulative-hook replay at layers 24/36/47.

This does not prove:

- autoregressive branch-specific KV/cache fork-carry,
- production Hunter-Seeker execution,
- compute savings.

## Source Notes Moved To History

Detailed source docs were moved under `history/interfaces-and-tools/`.

Exact pre-consolidation root copies are also preserved under:

`history/pre_docs_consolidation_2026-05-31/`

## Branch training + logic expansion + terminal v1 (2026-06-06)

- Status: `LOGIC_EXPANSION_READY_TRAINING_NOT_READY`; policy decision: `KEEP_EXTERNAL_DUALANCHOR_CORECONTENT_BASELINE`.
- Goal: move from external branch selection (DualAnchor + CoreContent_v2) toward **model-internal branching**. Primary deliverable = the branch-training **data + evaluation harness** (the L/O training is a bounded 300-step LoRA proof-of-capability on Ouro-RLTT, not a converged model). External verifiers are the only ground truth; DualAnchor/CoreContent are teachers/baselines; science diagnostic-only; steering not run.
- **Logic expanded** to ~33247 train groups across 10 verifier-backed families (synthetic propositional/proof-depth/syllogism/z3-constraint + LogiQA/ReClor/RuleTaker/FOLIO/LSAT/logical-entailment).
- **Branch-pool reachability@4** (Ouro-RLTT generations, external-labeled): {'coding': 0.4333, 'logic': 0.7333, 'math': 0.8333, 'reasoning': 0.95}. Math went 0.31→0.83 once given a brutal tool-free answer-forcing prompt + 1400 math budget + early-stop + LaTeX/sympy verifier; coding ~0.43 after fixing a function-name prompt gap (genuine, not artifact).
- **Experiment 1 (integrated terminal, H)**: within real DualAnchor top-5 survivor sets, CoreContent_v2 0.6584 > DualAnchor forced-top1 0.3787 (retention 1.0) → **selection, not survival, is the terminal bottleneck**; the composed external baseline (DualAnchor survival → CoreContent_v2 ranking → survivor handoff) is valid.
- **DualAnchor-as-teacher (J)**: useful branch-policy teacher (oracle-retention lift over random +0.11 coding/reasoning) but ~random on logic (+0.03) → logic branch quality must come from verifier reward, not teacher distillation.
- **Trained vs Ouro-RLTT-no-adapter (O)**: macro positive_oracle@K {'base': 0.708, 'sft': 0.688}, lift -0.02, diversity lift 0.45 (`MODEL_INTERNAL_BRANCHING_PARTIAL`).
- Data-hygiene fixes caught by hand-audit: math-LaTeX verifier false-negatives, coding function-name mismatch, and a hendrycks_math `task_uid` collision (row indices repeat across 7 subjects → re-keyed by prompt-hash).
- No Ouro base/tokenizer/checkpoint/registry overwrite; adapters saved only under `artifacts/models/branch_training_logic_expansion_v1/`; pure/transplanted/CoreContent_v2 artifacts untouched.
- Artifacts: `artifacts/reports/probes/branch_training_logic_expansion_terminal_v1_2026-06-06`; data `data/branch_training_logic_expansion_v1/`.
