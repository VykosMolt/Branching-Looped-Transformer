# Branching Looped Transformer

Evaluator probes and branch-selection scaffolding for reading hidden-state
signals in Princeton Ouro-RLTT loop states.

> **Current status:** Phase 2a branch survival is ready under DualAnchor with
> terminal survivor-set handoff. Autoregressive KV/cache branch-carry and
> partial-cache splice are validated in test harnesses. Steering/model training
> has not started. Broad science is diagnostic; the active core domains are
> coding, reasoning, math, logic, and alignment/preference.

Branching Looped Transformer is a research repository for evaluator and
branch-selection work on **Princeton Ouro-RLTT**, a looped transformer whose
internal loop states can be read by lightweight evaluators. The central
hypothesis is that frozen looped-transformer hidden states contain
**relationally readable signals** about preference, answer quality, branch
viability, and trajectory promise.

The current system is best understood as a latent branch selector:

```text
Ouro-RLTT hidden states
    -> relational taps / pairwise evaluators
    -> branch survival and content-selection policies
    -> terminal survivor-set handoff
    -> future steering or training-time write path
```

This repository is research code. It is not production routing, not a safety
product, not a trained steering system, and not a tool-using runtime.

## Table Of Contents

- [Executive Summary](#executive-summary)
- [What Is Currently Claimed](#what-is-currently-claimed)
- [What Is Not Claimed](#what-is-not-claimed)
- [Architecture Overview](#architecture-overview)
- [Current Locked Baseline](#current-locked-baseline)
- [Major Results](#major-results)
- [Current Project State](#current-project-state)
- [Repository Layout](#repository-layout)
- [Quickstart](#quickstart)
- [Key Files](#key-files)
- [Glossary](#glossary)
- [Caveats](#caveats)
- [Citation / Paper](#citation--paper)
- [License](#license)
- [Short Handoff](#short-handoff)

## Executive Summary

The project has converged on a clean separation of roles:

| Role | Current mechanism | Status |
| --- | --- | --- |
| Branch survival / pruning | DualAnchor = `MIX_CODE_REASONING` + `MIX_OBJECTIVE_ALL` | Locked for Phase 2a / Phase 2b baseline |
| Terminal collapse | Top5/full survivor-set handoff | Required; forced top1 is unsafe on hard slices |
| Content/final choice | Broad objective/content selector family | Separate from branch survival |
| Runtime branch carry | Autoregressive KV/cache branch-carry + partial-cache splice | Validated in test harnesses |
| Steering/write path | Not started | Future work |
| Science | Anatomy partial; chemistry/physics/SciQ diagnostic | Not a broad headline domain |
| Core domains | Coding, reasoning, math, logic, alignment/preference | Ready for core-domain Phase 2b prep |

The most important design decision is:

```text
DualAnchor keeps good latent branches alive.
A separate content selector ranks final survivors.
Terminal top5/full handoff prevents unsafe early collapse.
```

## What Is Currently Claimed

### 1. Frozen Ouro-RLTT loop states contain relational preference signal

The pairwise HH-RLHF evaluator achieved **95.2% fixed-order test accuracy** on
heldout HH-RLHF examples when scoring `chosen` against `rejected`. This is not
the same as pointwise scoring: independent pointwise evaluators were much
weaker. The correct interpretation is that Ouro-RLTT loop states encode
preference primarily **relationally**, not as a clean absolute scalar on one
response.

### 2. DualAnchor survives architecture-shaped branching

The active branch/prune selector is:

```text
MIX_CODE_REASONING + MIX_OBJECTIVE_ALL
```

The architecture-shaped loop uses taps at layers 24, 36, and 47 across loops
L1-L4. The strongest current survival result retains terminal oracle branches
under repeated perturb-score-prune stages, but terminal forced top1 remains
unsafe on hard/reward-diverse slices.

### 3. Terminal survivor-set handoff is required

The project does not currently trust unconditional terminal top1. The safe
terminal policy is:

```text
keep top5/full terminal survivor set;
let the final selector rank or choose within that survivor set.
```

Confidence-gated top1 is diagnostic/optional unless recalibrated.

### 4. Autoregressive branch-specific KV/cache carry is validated

The cache validation ladder passed through prompt-internal branch-specific
cache continuation. A later partial-cache splice run upgraded the diagnostic
splice into a real suffix-recompute method with measured amortized compute
savings for K>=2 branches.

### 5. Core domains are the active scope

The current core-domain scope is:

```text
coding
reasoning
math
logic
alignment / preference
```

Science is kept diagnostic unless a narrow source-specific result is explicitly
being studied.

## What Is Not Claimed

This repository does **not** currently claim:

- production routing readiness;
- trained steering or action steering;
- model fine-tuning success;
- that Ouro-RLTT has been modified or trained;
- broad science competence;
- unconditional terminal top1 readiness;
- human-like introspection or consciousness;
- that branch selection has already been trained into the model;
- that every result generalizes beyond the tested model, datasets, and probes.

The current claim is narrower:

> Frozen looped-transformer states expose readable relational signals. External
> taps can read those signals well enough to guide branch survival and
> candidate ranking. The write/steering path remains future work.

## Architecture Overview

### Conceptual system

```text
                 +-------------------------------+
                 | Princeton Ouro-RLTT           |
                 | looped transformer backbone   |
                 +---------------+---------------+
                                 |
                                 v
                  layers 24 / 36 / 47, loops L1-L4
                                 |
              +------------------+------------------+
              |                                     |
              v                                     v
     DualAnchor branch survival          objective/content scoring
     pairwise tap policy                  for terminal survivors
              |                                     |
              v                                     v
  perturb -> score -> threshold -> prune  rank terminal survivors
              |                                     |
              +------------------+------------------+
                                 |
                                 v
                     top5/full survivor handoff
                                 |
                                 v
                     future steering/write path
```

### DualAnchor loop schedule

```text
L1_24 -> L1_36 -> L1_47
L2_24 -> L2_36 -> L2_47
L3_24 -> L3_36 -> L3_47
L4_24 -> L4_36 -> terminal L4_47
```

At each nonterminal stage:

```text
1. current survivors spawn perturbation children
2. inherited survivors remain candidates
3. DualAnchor scores candidates pairwise
4. mean_floor_very_loose thresholding runs
5. rescue guards can preserve candidates
6. hard budget = 8
7. survivors flow to the next stage
```

At terminal `L4_47`:

```text
forced top1: diagnostic only
top5/full survivor-set handoff: safe default
```

## Current Locked Baseline

```text
Branch selector:
  DualAnchor = MIX_CODE_REASONING + MIX_OBJECTIVE_ALL

Loop/layer schedule:
  L1_24 -> L1_36 -> L1_47
  L2_24 -> L2_36 -> L2_47
  L3_24 -> L3_36 -> L3_47
  L4_24 -> L4_36 -> terminal L4_47

Threshold:
  mean_floor_very_loose

Budget:
  8

L47:
  active in nonterminal loops

Terminal:
  top5/full survivor-set handoff

Science:
  diagnostic except narrow partial anatomy result

Steering:
  not started
```

## Major Results

### Relational HH-RLHF evaluator

The pairwise evaluator uses hidden-state differences across Ouro-RLTT loop
states. It is not a pointwise scorer. The core pattern is:

```text
pairwise nonlinear evaluator:       95.2% fixed-order HH-RLHF test accuracy
pairwise linear difference probe:   84.5%
independent nonlinear evaluator:    ~65%
independent linear classifier:      21.75%, inverted polarity
```

The flip test revealed a positive scalar offset in the raw scorer. Low strict
sign-flip rate is therefore a bias/calibration issue, not a collapse of the
95.2% fixed-order result.

### DualAnchor architecture-looped survival

The architecture-looped v3 run established strong repeated branch survival:

```text
tasks:                         48
stage oracle retention:         0.9848
terminal oracle retained:       1.0000
forced terminal top1 oracle:    0.9167
reward-diverse top1 oracle:     0.6364
false-prune recovery:           8/8
```

Interpretation:

```text
Survival is strong.
Terminal forced top1 is not the primary policy.
```

### Science/reasoning repair

Science is no longer treated as a gate for the whole project.

Current science state:

```text
MMLU anatomy:     partial / small-n repaired
MMLU chemistry:   diagnostic / excluded
MMLU physics:     diagnostic / excluded
SciQ:             diagnostic / excluded in latest source-specific run
```

Reasoning remains headline-ready under survivor-set handoff.

### Core-domain tap audit

The core-domain audit pivoted the project toward:

```text
coding
reasoning
math
logic
alignment
```

It found:

```text
DualAnchor:           best locked survival mechanism
broad objective taps: better pure content selectors
science/anatomy:      diagnostic only
terminal:             full/top5 handoff required
```

### CoreContent v1

CoreContent v1 attempted to craft a better content/final selector. It did not
beat the broad-objective baseline on heldout.

Heldout macro top1:

```text
mixedhead_MIX_HH_OBJECTIVE     0.6270
CoreContent_weight_merge       0.6217
DualAnchor                     0.6205
MIX_CODE_REASONING_only        0.6190
MIX_OBJECTIVE_ALL_only         0.6155
CoreContent_linear_best        0.6079
CoreContent_domain_gated       0.6034
CoreContent_listwise           0.5590
```

Interpretation:

```text
Further content-tap work needs larger and cleaner data,
not more complex heads on the old split.
```

### KV/cache branch-carry and partial splice

Validated cache capabilities:

```text
ordinary cached decode
token-boundary cache fork
batched branch cache
prune/reorder survivor cache
current-token perturb carry
prompt-internal perturb branch cache
partial-cache suffix-recompute splice
```

Partial splice result:

```text
PARTIAL_CACHE_SPLICE_V2_STATUS = PARTIAL_SPLICE_COMPUTE_SAVING_VALID
```

This enables a test-harness claim of amortized compute-saving branch-carry for
multiple branches sharing a prompt. It does not imply production readiness.

## Current Project State

```text
Branch survival:
  DualAnchor locked

Content/final selection:
  separate broad objective/content selector family

Terminal:
  top5/full survivor handoff required

Core domains:
  coding / reasoning / math / logic / alignment

Science:
  diagnostic, with anatomy partial small-n

Cache substrate:
  autoregressive branch-carry and compute-saving splice validated in test harnesses

Steering:
  not started
```

## Repository Layout

Representative layout:

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── evaluator/
│   ├── bg_controller.py
│   ├── bg_transformer_features.py
│   ├── bg_hidden_branching.py
│   ├── bg_causal_adapter.py
│   ├── bg_sequence_adapter.py
│   └── bg_steering_hook.py
├── evaluator_core/
│   ├── pairwise_evaluator.py
│   ├── anchor_loss.py
│   ├── COMPONENTS.md
│   └── README.md
├── probes/
│   ├── evaluate_pairwise_rltt.py
│   ├── flip_pairwise_rltt.py
│   ├── layer_state_capture.py
│   ├── probe_loop_geometry_hh.py
│   ├── probe_bipartite_layers_24_36.py
│   ├── probe_cross_backbone_layers.py
│   └── README.md
└── docs/evaluator/
    ├── README.md
    ├── current-state.md
    ├── evaluator-navigation-map.md
    ├── dualanchor-architecture-baseline.md
    ├── dualanchor-tap-evolution.md
    ├── branch-generation-and-survival.md
    ├── terminal-selection-and-arbiters.md
    ├── kv-cache-branch-carry.md
    ├── science-reasoning-repair.md
    ├── bg_core_domain_tap_audit_dualanchor_readiness_v1.md
    ├── flip-test-interpretation.md
    └── history/
```

### Important docs

| File | Purpose |
| --- | --- |
| `docs/evaluator/evaluator-navigation-map.md` | Where to start reading |
| `docs/evaluator/current-state.md` | Current verdict ledger |
| `docs/evaluator/dualanchor-architecture-baseline.md` | Active branch/prune baseline |
| `docs/evaluator/bg_core_domain_tap_audit_dualanchor_readiness_v1.md` | Core-domain readiness audit |
| `docs/evaluator/kv-cache-branch-carry.md` | Autoregressive cache and splice validation |
| `docs/evaluator/flip-test-interpretation.md` | HH evaluator flip-test interpretation |
| `docs/evaluator/science-reasoning-repair.md` | Science/reasoning domain decision |
| `docs/evaluator/terminal-selection-and-arbiters.md` | Terminal handoff and final-arbiter history |

## Quickstart

This repository assumes local access to a compatible Ouro-RLTT checkpoint and,
for many probes, a CUDA-capable machine. Exact dependency pins vary by local
environment.

```bash
git clone https://github.com/VykosMolt/Branching-Looped-Transformer.git
cd Branching-Looped-Transformer
python -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Most probe scripts expect local paths such as:

```text
models/ouro_rltt_local
artifacts/checkpoints/evaluator/pairwise_epoch2.pt
```

Model weights, evaluator checkpoints, generated reports, and large run outputs
are not stored on GitHub.

For a dependency-light smoke check of the evaluator module:

```bash
python -m evaluator_core.pairwise_evaluator
```

For live Ouro-RLTT probes, read the script arguments before rerunning. Many
scripts are archived research probes and expect local checkpoints or reports.

## Key Files

| File | Role |
| --- | --- |
| `evaluator_core/pairwise_evaluator.py` | Pairwise evaluator architecture |
| `evaluator_core/anchor_loss.py` | Frozen evaluator anchor/scoring utilities |
| `evaluator/bg_transformer_features.py` | Read-only Ouro-RLTT feature capture |
| `evaluator/bg_controller.py` | Read-only BG branch-selection controller |
| `evaluator/bg_hidden_branching.py` | Hidden-branch perturbation utilities |
| `evaluator/bg_causal_adapter.py` | Adapter/intervention utility classes |
| `evaluator/bg_sequence_adapter.py` | Sequence-level adapter helpers |
| `evaluator/bg_steering_hook.py` | Guarded intervention hooks used in experiments |
| `probes/evaluate_pairwise_rltt.py` | HH-RLHF pairwise evaluation on local loop states |
| `probes/flip_pairwise_rltt.py` | Argument-swap / scorer-bias diagnostic |
| `probes/layer_state_capture.py` | Loop/layer state capture utilities |
| `probes/probe_loop_geometry_hh.py` | Loop geometry diagnostic |
| `probes/probe_bipartite_layers_24_36.py` | Layer 24/36/47 loop geometry probe |
| `probes/probe_cross_backbone_layers.py` | Cross-backbone layer comparison probe |

## Glossary

### Ouro-RLTT

The looped transformer backbone used throughout this project. The relevant
research object is not just its final text output, but its internal loop/layer
hidden states.

### Loop States

Intermediate hidden states produced by Ouro-RLTT's iterative computation. The
project reads these states at selected layers and loops.

### Tap

A lightweight evaluator head over hidden-state features. Most taps are pairwise
relational comparators.

### DualAnchor

The current branch-survival selector:

```text
MIX_CODE_REASONING + MIX_OBJECTIVE_ALL
```

### CoreContent

The content/final-choice selector family. CoreContent is deliberately separated
from branch survival.

### Branch Survival

Keeping potentially useful latent branches alive through perturb/prune stages.

### Terminal Handoff

Retaining top5/full terminal survivors rather than forcing one final candidate
too early.

### Partial-Cache Splice

A compute-saving branch-carry method that shares unaffected prefix cache slots
and recomputes only the downstream suffix after a branch perturbation.

## Caveats

- The central results are currently from one model family: Princeton Ouro-RLTT
  and local compatible variants.
- Many experiments are probe-scale, not production-scale.
- Science remains uneven and source-specific.
- Terminal final-choice remains a bottleneck; survivor handoff is required.
- Steering/write-path training remains future work.
- Large generated outputs can be expensive to reproduce and are not included in
  this repository.

## Citation / Paper

The conceptual foundation is summarized in the included documentation as:

```text
Relational Preference Encoding in Looped Transformer Internal States
Jan Kirin, April 2026
```

The main takeaway:

```text
Preference and quality signals in Ouro-RLTT loop states are much more
accessible relationally than pointwise, supporting pairwise hidden-state
readouts as the core primitive.
```

## License

Apache License 2.0. See `LICENSE`.

## Short Handoff

```text
DualAnchor is locked for branch survival.
Terminal top5/full survivor handoff is mandatory.
Final content selection is separate from branch survival.
Core domains are coding, reasoning, math, logic, and alignment.
Science is diagnostic except anatomy partial small-n.
Autoregressive cache branch-carry and compute-saving partial splice are validated in test harnesses.
Steering has not started.
```
