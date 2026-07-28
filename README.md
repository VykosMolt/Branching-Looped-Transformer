# Branching Looped Transformer

Evaluator probes and branch-selection scaffolding for reading hidden-state
signals in Princeton Ouro-RLTT loop states.

> **Current status (paper v3, 2026-07-27):** the readout--control boundary has been
> *located* rather than merely observed. Five sealed, pre-registered conversions asked
> where a frozen readout turns into an outcome gain. **Decision-level use converts:**
> calibrated abstention improves the risk--coverage trade-off in all four sealed arms
> across both domains, and forced terminal selection clears an exact matched-random null
> even on a pool where every candidate is well-formed by construction (27/32 vs 64.8%
> expected, exact *p* = 0.0086) -- which resolves the content-sensitivity question the
> v2 pool could not. **Generative control does not:** directional steering is an
> established negative, the four-task fork screen is bounded, per-task recurrent-depth
> allocation is a sealed null (neither the tap nor the model's own exit gate beats a
> matched-histogram random control, on both checkpoints), and a minimal sign-conditioned
> LoRA returns a bounded no-detected-gain result. Phase 2a branch survival remains locked
> under DualAnchor, and the branch-carry substrate and bit-exact splice remain validated.
> Full training-time integration is still the next lever and has not been run.

> ## ⚠️ Correction notice (2026-07-25)
>
> Several figures quoted in this README were corrected by a project-wide evaluation audit
> (source-item leakage across row-level splits; a presentation-order prior in the
> fixed-order pairwise evaluator). The correction of record for the published figures is
> the **erratum to [arXiv:2604.09870](https://arxiv.org/abs/2604.09870) (v2)**.
>
> | Result | Reported here | Corrected |
> |---|---:|---:|
> | Pairwise nonlinear evaluator | 95.2% fixed-order | **0.6392** (strict antisymmetrized) |
> | Pairwise linear difference probe | 84.5% | **0.5653** |
> | Independent linear classifier | 21.75%, "inverted polarity" | **0.5418** — *above* chance |
> | CoreContent v2 macro top-1 | 0.6691 | **0.6310** (task-disjoint) |
> | DualAnchor stage oracle retention | 0.9848 | **0.9697** (task-disjoint) |
>
> Every effect survived decontamination with a modest loss; one training-stage claim did
> not reproduce and is retracted. See [`papers/`](papers/) for the full account and
> [`results/`](results/) for the artifacts behind the current values.


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
- [Papers And Published Artifacts](#papers-and-published-artifacts)
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

### S1 branch-carry reference loop (frozen model)

The full inject -> carry -> prune -> loop-back -> terminal loop was assembled as a
correctness reference and validated with a five-gate ladder:

```text
gate 1  alpha=0 re-derive plumbing                    bit-exact (logit maxabs 0.0)
gate 2  alpha>0 multi-locus chaining                  bit-exact
          (2a splice fork == hook replay;
           2b re-derive-through-ancestor == all-hooks-live)
gate 3  live DualAnchor + CoreContent tap prune        pass (gibberish bottomed)
gate 4  no-original-root-reuse (lineage invariant)     held across 12 loci x 4 tasks
gate 5  lineage/splice-stack schedule sanity           held across 12 loci x 4 tasks
```

The per-survivor branch-specific re-derivation uses the single fork primitive
`LayerOutputPerturbHook(token_range)` at the canonical last-token suffix
(causal-suffix-safe), with each lineage perturbation reproduced exactly on replay.
The 12-locus reference loop runs end to end with **zero correctness loss**
(`oracle_over_survivors == base_acc`).

Frozen-model capability result (single-locus fork-parameter screen across
alpha {.02,.05,.10} x token_range {last, last-8, second-half} x decode
{greedy, sample}, loop-1 loci + a loop-4 sentinel, plus a K-matched plain-sampling
baseline):

```text
greedy / deterministic fork:   0.0 new-correct on base-missed tasks (every cell)
fork + sampling:               0.611 oracle
K-matched plain sampling:      0.75  oracle   (>= fork+sampling)
```

Two cleanly separated walls (mapping to two distinct future training jobs):

```text
Wall A  generation / reachability:
  deterministic frozen injection/carry creates no new correct branches;
  sampling creates some; fork+sampling does NOT beat K-matched plain sampling.

Wall B  selection / conversion:
  even when correct candidates exist, the current content tap does not reliably
  rank them first for GENERATED branch candidates (not a global tap failure).
```

Scope: this is a **local** verdict under tested perturbation regimes, not a claim
that no frozen branch regime can ever work. The positive deliverable is the
validated measurement harness itself; it converts the branch-carry idea from an
architectural sketch into an instrument ready to test whether training makes
branches outcome-distinct and selector-readable. Probes:
`probes/mpn_s1_4_reference_loop.py`, `probes/mpn_s1_5_divergence_ablation.py`,
`probes/mpn_s1_4b_kmatched_sampling.py`, and the gate probes; full write-up in
`docs/evaluator/s1-branch-carry-reference-loop.md`.

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

Branch loop:
  full inject->carry->prune->loop-back->terminal reference loop validated (5 gates);
  frozen-model capability is reachability-neutral, locally closed under tested regimes

Frozen conversions (paper v3):
  abstention           -> converts (all four sealed arms)
  content selection    -> converts (all-well-formed pool; hidden-vs-surface open)
  loop allocation      -> sealed null (tap and native gate, both checkpoints)
  prefix-prune tourney -> pool could not be powered (0.978 any-of-6 ceiling)
  trained binding      -> bounded no-detected-gain result (one direction, one locus)

Cross-checkpoint:
  Thinking pre-answer transfer remains unresolved after a powered replication:
  +0.042 [-0.017, +0.104] on 900 fresh tasks, verdict UNRESOLVED -- rules out an
  RLTT-sized effect on Thinking, leaves a smaller one open

Next lever:
  training-time integration (outcome-distinct branches + correctness-readable selector)

Steering:
  not started
```

## Papers and published artifacts

`papers/` contains the manuscripts this repository supports:

| File | Pages | Contents |
|---|---:|---|
| `kirin2026_paper1_v3.pdf` | 72 | **Current version.** *Operational Proto-Introspection in Looped Language Models* — process-quality taps, the executable branching substrate, and the readout–control boundary, now located between decision-level use and generative control by five sealed conversions (§8.6). |
| `kirin2026_paper1_v3_source.tar.gz` | — | Self-contained XeLaTeX source for the v3 PDF (figures, styles, bibliography); builds in three passes with no network access. |
| `figures_v3/` | — | The v3 figure PDFs and `make_figures.py`, which regenerates every one of them from the values quoted in the paper. |
| `kirin2026_paper1_v1.pdf` | 49 | Superseded first version, retained for provenance. Its magnitudes predate the audit summarised above; read v3 for any current claim. |
| `kirin2026_paper2.pdf` | 13 | *Two Evaluation Traps in Constructed-Row Pipelines* — source-item leakage, presentation-order shortcuts, and the audit protocol behind the corrections above. |

`results/` contains the experiment records the papers cite, minus model weights and
extracted feature tensors. Paths map onto the papers' artifact index by dropping the
`artifacts/` prefix — the index's `artifacts/reports/<run>/` is `results/<run>/` here.

| Directory | What it holds |
|---|---|
| `results/cross_loop_early_layer_taps_20260720/` | Cross-loop early-layer localization: the loop × layer refit matrix, the 18-transfer frozen transplant matrix, bootstrap statistics, controls, per-cell predictions, figures, and `SHA256SUMS`. |
| `results/paper1_v2_overnight_20260724/` | Horizon Logic strict pre-answer study, the powered terminal-selection evaluation, and the subspace-vs-subspace geometry audit, each with its run manifest, integrity records, and results JSON. |
| `results/paper_verification/` | The re-audits behind the corrected values: powered pair-disjoint probes, the full 8,552-pair antisymmetry audit, the task-disjoint CoreContent refit, and the non-looped architecture control. |
| `results/proto_introspection/` | Strict pre-answer GSM8K records and the artifact reconciliation index. |
| `results/horizon_power_v3_20260726/` | Powered Horizon Logic extension: 510 prospectively sealed task-disjoint tasks, raising the held-out negative class from ≈19 to 84. Pooled increment +0.111 [+0.056, +0.169]; the new cohort alone independently replicates at +0.095, and both survive the adversarial malformed-sibling composite that the original 170-task cohort did not. |
| `results/family_xloop_v3_20260726/` | Within-family replication of the recurrent-depth readability trend on base-2.6B, Thinking-2.6B, and the 1.4B (layers mapped proportionally), plus frozen cross-checkpoint tap transfer at parity (\|Δ\| ≤ 0.028, score correlation 0.97–0.99). |
| `results/huginn_probe_v3_20260726/` | Out-of-family probe on Huginn-0125, a depth-recurrent architecture: readability rises with recurrence depth (step 8 − step 1 = +0.107 [+0.058, +0.157]) at a lower absolute level, with no comparable first-pass rotation detected. |
| `results/thinking_preanswer_v3_20260726/` | Sealed pre-answer attempt on the Thinking sibling. Reported as **inconclusive / underpowered**, not null: +0.027 with a 95% CI of [−0.13, +0.21] that contains both zero and the RLTT-sized effect, power lost to 45.6% truncation-malformedness at the deliberately matched 448-token budget. |
| `results/selective_prediction_v3_20260726/` | **Conversion C1** — selective prediction (abstention). Computed entirely from preserved out-of-fold predictions with no new generation; ΔAUARC excludes zero in all four sealed arms across both domains. |
| `results/depth_alloc_v3_20260726/` | **Conversion C2** — tap-gated recurrent-depth allocation on RLTT and Thinking. Fixed-depth tables let every policy be scored as row selection at exactly matched compute; verdict `ALLOCATION_NOT_SIGNAL_DRIVEN` on both checkpoints, for the tap *and* the model's own exit gate. |
| `results/tournament_v4_20260727/` | **Conversion C3** — matched-budget prefix-prune tournament. Hit its pre-registered feasibility floor (0.978 any-of-6 ceiling), so no claim is made in either direction; the internals are preserved for the harder-domain re-run the design now awaits. |
| `results/wellformed_terminal_v4_20260727/` | **Conversion C4** — terminal selection on a pool where every candidate is well-formed *by construction*, so malformedness cannot carry the margin. 27/32 informative held-out groups against 64.8% matched-random (exact *p* = 0.0086), establishing content-sensitive selection; the hidden-versus-surface increment remains unresolved at this sample size. Includes the sealed extension declared before any selector was fit. |
| `results/lora_s3a_pilot_20260727/` | **Conversion C5** — sign-conditioned LoRA binding pilot: one curated writable direction, one locus, ≤400 steps, against a coin-flipped twin adapter. Binding not detected; scope is deliberately narrow, so this bounds the cheapest form of binding rather than training-time integration generally. |
| `results/thinking_preanswer_power_v5_20260727/` | **Complete.** The powered Thinking-only pre-answer replication, pre-registered in this repository before it ran: the sealed plan, the formal sizing simulation behind N = 900, and now the result. 900 fresh task-disjoint tasks yielded 99 held-out negatives against a pre-registered target of 80–100; the new-cohort increment is **+0.042, 95% CI [−0.017, +0.104]**, sealed verdict **`UNRESOLVED`**. Roughly three times tighter than the pilot interval and still crossing zero: the run rules out a Thinking effect as large as RLTT's (+0.095, which it was powered at 0.816 to detect) without establishing either the effect or its practical absence at the ±0.05 margin. |
| `results/v3_overnight_20260726_logs/` | Orchestration logs and the summary of record for the v3 overnight programme. |

Excluded by design: `*.pt` tensors, extracted feature shards, datasets, and model
weights. Every published run directory ships the `SHA256SUMS` covering its **full**
contents, so the omitted files are still enumerable and verifiable at their source.

### Access to the Ouro-RLTT weights

The Ouro-RLTT weights used as this project's primary experimental backbone are not
redistributed in this repository. I can share them **for research purposes on a
case-by-case basis** — including, explicitly, for anyone who wants to check this work.

Message me and say who you are, what you are working on, and why you need them. Checking
or attempting to falsify a result in these papers is a perfectly good reason, and I would
rather hand the weights to someone auditing the numbers than have the claims go untested.
Requests are answered individually and at my discretion.

`tools/` holds the verification scripts named in the papers' reproducibility index, and
`probes/` now also carries the scripts that produced the two newest runs
(`bg_xloop_early_v1_*`, `bg_v2_overnight_*`).

`tools/paper1_v3/` carries the 22 generators, analysers, and orchestrators behind the v3
results above — the Horizon power extension, the family and Huginn replications, the
Thinking attempt, all five frozen conversions, and the pre-registered v5 power
replication including its sizing simulation. Each sealed run's `EXPERIMENT_PLAN.md` names
the scripts that produced it, and the analysis stages carry the Stage-A gates that
reproduce prior published numbers to 1e-9 before touching new data.

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
│   ├── mpn_s1_4_reference_loop.py        # S1 branch-carry reference loop
│   ├── mpn_s1_5_divergence_ablation.py   # S1.4a fork-parameter screen
│   ├── mpn_s1_4b_kmatched_sampling.py    # S1.4b K-matched sampling baseline
│   ├── mpn_s1_4_*.py                     # gate ladder (re-derive / chaining / prune)
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
    ├── s1-branch-carry-reference-loop.md
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
| `docs/evaluator/s1-branch-carry-reference-loop.md` | S1 full branch loop: gates, reference loop, frozen-model fork screen + sampling baseline |
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
| `probes/mpn_s1_4_reference_loop.py` | S1 full inject->carry->prune->loop-back->terminal reference loop |
| `probes/mpn_s1_5_divergence_ablation.py` | S1.4a single-locus fork-parameter screen |
| `probes/mpn_s1_4b_kmatched_sampling.py` | S1.4b K-matched plain-sampling baseline |

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
