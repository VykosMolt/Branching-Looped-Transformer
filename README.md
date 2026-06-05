[README_ouro_project.md](https://github.com/user-attachments/files/28645982/README_ouro_project.md)
# Ouro Research: Relational Evaluators, DualAnchor Branching, and CoreContent Selection

> **Current status:** Phase 2a branch survival is ready under DualAnchor with terminal survivor-set handoff. Autoregressive KV/cache branch-carry and amortized partial-cache splice are validated in a test harness. Steering/model training has **not** started. Broad science is no longer a blocking headline domain; the current core domains are coding, reasoning, math, logic, and alignment.

This repository contains an experimental research stack around **Ouro RLTT**, a looped transformer whose internal loop states can be read by lightweight evaluators. The project’s central hypothesis is that frozen looped-transformer hidden states contain **relationally readable signals** about preference, answer quality, branch viability, and trajectory promise.

The current system is best understood as a latent branch controller:

```text
Ouro hidden states
    ↓
relational taps / pairwise evaluators
    ↓
branch survival and content-selection policies
    ↓
terminal survivor-set handoff
    ↓
future steering / training-time write path
```

The project is still research code. It is not production routing, not a safety product, and not a trained steering system.

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [What Is Currently Claimed](#what-is-currently-claimed)
- [What Is Not Claimed](#what-is-not-claimed)
- [Architecture Overview](#architecture-overview)
- [Current Locked Baseline](#current-locked-baseline)
- [Major Results](#major-results)
- [Current Project State](#current-project-state)
- [Repository Layout](#repository-layout)
- [Quickstart](#quickstart)
- [Key Scripts](#key-scripts)
- [Data and Artifact Policy](#data-and-artifact-policy)
- [Recommended Next Run](#recommended-next-run)
- [Glossary](#glossary)
- [Caveats](#caveats)
- [License](#license)

---

## Executive Summary

The project has converged on a clean separation of roles:

| Role | Current mechanism | Status |
| --- | --- | --- |
| Branch survival / pruning | **DualAnchor** = `MIX_CODE_REASONING` + `MIX_OBJECTIVE_ALL` | Locked for Phase 2a / Phase 2b baseline |
| Terminal collapse | Top5/full survivor-set handoff | Required; forced top1 is unsafe |
| Content/final choice | `mixedhead_MIX_HH_OBJECTIVE` | Current baseline until CoreContent v2 refit |
| Runtime branch-carry | Autoregressive KV/cache branch-carry + partial-cache splice | Validated in test harness |
| Steering/write path | Not started | Future work |
| Science | Anatomy partial; chemistry/physics/SciQ diagnostic | Not a broad headline domain |
| Core domains | Coding, reasoning, math, logic, alignment | Ready for core-domain Phase 2b prep |

The project’s most important current design decision is:

```text
DualAnchor keeps good latent branches alive.
A separate content selector chooses/ranks final survivors.
Terminal top5/full handoff prevents unsafe collapse.
```

---

## What Is Currently Claimed

### 1. Frozen Ouro loop states contain relational preference signal

The pairwise HH-RLHF evaluator achieved **95.2% fixed-order test accuracy** on heldout HH-RLHF examples when scoring `chosen` vs `rejected`. This is not the same as pointwise scoring: independent pointwise evaluators were much weaker. The correct interpretation is that Ouro’s loop states encode preference primarily **relationally**, not as a clean absolute scalar on one response.

### 2. DualAnchor survives architecture-shaped branching

The active branch/prune selector is DualAnchor:

```text
MIX_CODE_REASONING + MIX_OBJECTIVE_ALL
```

The architecture-shaped loop uses taps at layers 24, 36, and 47 across loops L1–L4. The strongest current survival result retains terminal oracle branches under repeated perturb-score-prune stages, but terminal forced top1 remains unsafe on hard/reward-diverse slices.

### 3. Terminal survivor-set handoff is required

The project does not currently trust unconditional terminal top1. The safe terminal policy is:

```text
keep top5/full terminal survivor set;
let the content selector rank/choose within the survivor set.
```

Confidence-gated top1 is diagnostic/optional only unless recalibrated.

### 4. Autoregressive branch-specific KV/cache carry is validated

The cache validation ladder passed through prompt-internal branch-specific cache continuation. A later partial-cache splice run upgraded the diagnostic splice into a real suffix-recompute method with measured amortized compute savings for K≥2 branches.

### 5. Core domains are the active scope

The current core-domain scope is:

```text
coding
reasoning
math
logic
alignment / preference
```

Science is kept diagnostic unless a narrow source-specific result is explicitly being studied.

---

## What Is Not Claimed

This repository does **not** currently claim:

- production routing readiness;
- trained steering or action steering;
- model fine-tuning success;
- that Ouro has been modified or trained;
- broad science competence;
- unconditional terminal top1 readiness;
- human-like introspection or consciousness;
- that branch selection has already been trained into the model;
- that every result generalizes beyond the tested model, datasets, and probes.

The current claim is narrower:

> Frozen looped-transformer states expose readable relational signals. External taps can read those signals well enough to guide branch survival and candidate ranking. The write/steering path remains future work.

---

## Architecture Overview

### Conceptual system

```text
                 ┌───────────────────────────────┐
                 │ Ouro-2.6B-Thinking             │
                 │ looped transformer backbone    │
                 └───────────────┬───────────────┘
                                 │ hidden states
                                 ▼
                  layers 24 / 36 / 47, loops L1-L4
                                 │
              ┌──────────────────┴──────────────────┐
              │                                     │
              ▼                                     ▼
     DualAnchor branch survival           CoreContent / objective
     pairwise tap policy                   final-content selector
              │                                     │
              ▼                                     ▼
  perturb → score → threshold → prune       rank terminal survivors
              │                                     │
              └──────────────────┬──────────────────┘
                                 ▼
                     top5/full survivor handoff
                                 │
                                 ▼
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
7. survivors flow to next stage
```

At terminal `L4_47`:

```text
forced top1: diagnostic only
top5/full survivor-set handoff: safe default
```

---

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

Content/final selection:
  mixedhead_MIX_HH_OBJECTIVE baseline

Science:
  diagnostic only

Steering:
  not started
```

---

## Major Results

### Relational HH-RLHF evaluator

The pairwise evaluator uses hidden-state differences across Ouro loop states. It is not a pointwise scorer. The core pattern is:

```text
pairwise nonlinear evaluator:       95.2% fixed-order HH-RLHF test accuracy
pairwise linear difference probe:   84.5%
independent nonlinear evaluator:    ~65%
independent linear classifier:      21.75%, inverted polarity
```

The flip test revealed a positive scalar offset in the raw scorer. Low strict sign-flip rate is therefore a bias/calibration issue, not a collapse of the 95.2% fixed-order result.

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
Terminal top1 is not the primary policy.
```

### Science/reasoning repair

Science is no longer treated as a gate for the whole project.

Current science state:

```text
MMLU anatomy:   partial / small-n repaired
MMLU chemistry: diagnostic / excluded
MMLU physics:   diagnostic / excluded
SciQ:           diagnostic / excluded in latest source-specific run
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
DualAnchor:          best locked survival mechanism
broad objective taps: better pure content selectors
science/anatomy:     diagnostic only
terminal:            full/top5 handoff required
```

### CoreContent v1

CoreContent v1 attempted to craft a better content/final selector. It did not beat the broad-objective baseline on heldout.

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

Selected content policy:

```text
mixedhead_MIX_HH_OBJECTIVE
```

Interpretation:

```text
Further content-tap work needs larger data, not more complex heads on the old split.
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

This enables a test-harness claim of **amortized compute-saving branch-carry** for multiple branches sharing a prompt. It does not imply production readiness.

---

## Current Project State

```text
Branch survival:
  DualAnchor locked

Content/final selection:
  mixedhead_MIX_HH_OBJECTIVE baseline

Terminal:
  top5/full survivor handoff required

Core domains:
  coding / reasoning / math / logic / alignment

Science:
  diagnostic, with anatomy partial small-n

Cache substrate:
  autoregressive branch-carry and compute-saving splice validated in test harness

Steering:
  not started

Next blocker:
  expanded core-domain content-selection data
```

---

## Repository Layout

Representative layout:

```text
.
├── docs/evaluator/
│   ├── current-state.md
│   ├── evaluator-navigation-map.md
│   ├── dualanchor-architecture-baseline.md
│   ├── dualanchor-tap-evolution.md
│   ├── branch-generation-and-survival.md
│   ├── terminal-selection-and-arbiters.md
│   ├── kv-cache-branch-carry.md
│   ├── science-reasoning-repair.md
│   ├── bg_core_domain_tap_audit_dualanchor_readiness_v1.md
│   ├── flip-test-interpretation.md
│   └── history/
│
├── artifacts/reports/probes/
│   └── <run-specific reports and outputs>
│
├── constructed_taps/
│   ├── pure_content_taps.pt
│   ├── transplanted_taps.pt
│   └── transplant_manifest.{json,csv,md}
│
├── utilities/tests/manual/
│   └── <probe / audit / training scripts>
│
├── modeling_ouro_patched.py
├── evaluator_pairwise.py
├── train_pairwise_fast.py
├── evaluate_pairwise.py
├── flip.py
├── latent_beam_search.py
├── grid_encoder.py
├── train_arc.py
├── solve_games.py
├── vesper.py
├── vesper_agent.py
├── arafel_agent_fixed.py
└── arafel_deepthink.py
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

---

## Quickstart

> This project assumes a CUDA-capable machine and local access to model weights / Hugging Face downloads. Exact dependency pins may vary by environment.

```bash
git clone <repo-url>
cd <repo>
python -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install torch transformers datasets accelerate numpy pandas pyarrow scikit-learn tqdm
```

If using bitsandbytes / local quantized agents:

```bash
pip install bitsandbytes sentencepiece
```

If using ARC tools:

```bash
pip install arc-agi
```

### Load Ouro and inspect loop states

```bash
python ouro_inspect.py
```

### Evaluate the HH pairwise evaluator

```bash
python evaluate_pairwise.py
```

### Run the flip-test diagnostic

```bash
python flip.py
```

### Run latent beam search prototype

```bash
python latent_beam_search.py --mode validate --n_prompts 10 --n_branches 4
```

### Run ARC/Vesper experimental agent

```bash
python vesper_agent.py --game ls20
```

---

## Key Scripts

| Script | Role |
| --- | --- |
| `modeling_ouro_patched.py` | Patched Ouro model/cache implementation |
| `ouro_inspect.py` | Inspect Ouro loop-state outputs |
| `evaluator_pairwise.py` | Pairwise evaluator architecture |
| `train_pairwise_fast.py` | Pairwise evaluator training |
| `evaluate_pairwise.py` | HH-RLHF evaluator evaluation |
| `flip.py` | Evaluator argument-swap / scorer-bias diagnostic |
| `latent_beam_search.py` | Prompt/loop-level latent branch-search prototype |
| `grid_encoder.py` | ARC grid-to-token encoder |
| `train_arc.py` | ARC training agent experiments |
| `solve_games.py` / `solver.py` | ARC-AGI-3 solver utilities |
| `vesper.py` / `vesper_agent.py` | Connectome-prior sensorimotor agent |
| `arafel_agent_fixed.py` | Local terminal reasoning agent |
| `arafel_deepthink.py` | Hidden-state recycling backend for local agent |

---

## Data and Artifact Policy

Large outputs are intentionally separated from source code.

Recommended git policy:

```text
Commit:
  source scripts
  docs
  small manifests
  small configuration files

Do not commit directly:
  .pt checkpoints
  huge feature shards
  raw Hugging Face dataset caches
  generated model outputs
  large artifacts under artifacts/reports/probes
```

Use `.gitignore` or Git LFS for:

```text
artifacts/
data/
features/
checkpoints/
constructed_taps/*.pt
*.safetensors
*.gguf
```

The project’s run reports should preserve:

```text
verdicts
metrics
artifact paths
scripts run
failure cases
non-claims
```

---

## Recommended Next Run

The next major run is:

```text
CoreContent Dataset Expansion + Content Tap Refit v2
```

Purpose:

```text
1. Pull much larger datasets for coding, reasoning, math, logic, and alignment.
2. Build clean candidate groups.
3. Validate parsers/verifiers.
4. Extract frozen Ouro/tap features.
5. Refit simple CoreContent selectors.
6. Compare against mixedhead_MIX_HH_OBJECTIVE.
```

If the expanded dataset produces a new policy that beats `mixedhead_MIX_HH_OBJECTIVE`, promote it as the content/final selector.

If it does not, lock:

```text
Branch survival:
  DualAnchor

Content/final selection:
  mixedhead_MIX_HH_OBJECTIVE

Terminal:
  top5/full survivor handoff
```

Then proceed to Phase 2b steering planning on the core domains.

---

## Glossary

### Ouro

The looped transformer backbone used throughout this project. The relevant research object is not just its final text output, but its internal loop/layer hidden states.

### Loop states

Intermediate hidden states produced by Ouro’s iterative computation. The project reads these states at selected layers and loops.

### Tap

A lightweight evaluator head over hidden-state features. Most taps are pairwise relational comparators.

### DualAnchor

The current branch-survival selector:

```text
MIX_CODE_REASONING + MIX_OBJECTIVE_ALL
```

### CoreContent

The content/final-choice selector family. CoreContent is deliberately separated from branch survival.

### Branch survival

Keeping potentially useful latent branches alive through perturb/prune stages.

### Terminal handoff

Retaining top5/full terminal survivors rather than forcing one final candidate too early.

### Proto-introspection

The claim that the model’s hidden states expose readable traces of its own emerging judgments. This is **not** a claim of consciousness or full self-understanding.

### Partial-cache splice

A compute-saving branch-carry method that shares unaffected prefix cache slots and recomputes only the downstream suffix after a branch perturbation.

---

## Caveats

- The project’s central results are currently from one model family: Ouro-2.6B-Thinking / local variants.
- Many experiments are probe-scale or artifact-scale, not production-scale.
- Science remains uneven and source-specific.
- Terminal final-choice remains a bottleneck; survivor handoff is required.
- Steering/write-path training remains future work.
- Generated artifacts can be large; plan disk usage accordingly.
- The local Python/tool agents are not security sandboxes.

---

## Citation / Paper

The conceptual foundation is summarized in the included paper:

```text
Relational Preference Encoding in Looped Transformer Internal States
Jan Kirin, April 2026
```

The main takeaway:

```text
Preference and quality signals in Ouro loop states are much more accessible relationally
than pointwise, supporting pairwise hidden-state readouts as the core primitive.
```

---

## License

No license is declared here. Add an explicit license before public release.

---

## Short Handoff

```text
DualAnchor is locked for branch survival.
Terminal top5/full survivor handoff is mandatory.
mixedhead_MIX_HH_OBJECTIVE is the current content/final selector.
Core domains are coding, reasoning, math, logic, and alignment.
Science is diagnostic only except anatomy partial small-n.
Autoregressive cache branch-carry and compute-saving partial splice are validated in test harness.
Steering has not started.
Next: CoreContent Dataset Expansion + Content Tap Refit v2.
```
