<!-- Imported from `es_integration_plan.md` during Hunter-Seeker state consolidation on 2026-05-14. -->
<!-- Original SHA256: 9f8045d1830ce923aed14f9a9015d03c3bbfce93ecfff5691e6575a94f433546; original line count: 217. -->

# ES Integration Plan for Hunter Seeker

**Status:** Planning · **Date:** 2026-05-08 · **Author:** Johann Hirschner (synthesis with Claude Opus 4.7 + GPT 5.5 Pro)

**Sources:**
- Salimans et al. 2017, *Evolution Strategies as a Scalable Alternative to Reinforcement Learning* (arXiv:1703.03864)
- Qiu et al. 2025, *Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning* (arXiv:2509.24372)
- Sarkar et al. 2025, *Evolution Strategies at the Hyperscale* / EGGROLL (arXiv:2511.16652)
- `PROJECT_STATE_HUNTER_SEEKER.md` snapshot of 2026-05-05 (wa30 ego/topology RLTT)

---

## 0. TL;DR

ES is useful to Hunter Seeker as an outer loop over **recovery / candidate-generation gates and self-model gates**, optimized against whole-episode ARC-AGI-3 reward with deterministic evaluation. It is *not* useful as a full-parameter Ouro fine-tuner on the laptop GPU, and the surface-level target (the ActionHead in the older `train_arc.py`) was the wrong target — the live blocker is post-collapse recovery proposal generation around wa30 phase 65 / step 249-251, not action-head scoring.

The plan is gated by two hard prerequisites:

1. The 250-275 all-risky-basin performance patch must land. ES multiplies rollout cost; if a single deterministic wa30 rollout is already slow in that window, ES is wall-clock dead before sample efficiency matters.
2. The observation/action-effect learning loop must be producing nontrivial signal (inverse-action promotions, engram inferred-action support, topology-delta predictions before the phase-65 divergence). ES over gates that select among meaningless proposals optimizes noise.

Once those prerequisites hold, ES is well-matched to the project's stated next-work in `PROJECT_STATE_HUNTER_SEEKER.md` line 180: "recovery branch diversification, candidate novelty, or action-effect contradiction handling."

---

## 1. What the three papers actually contribute

### 1.1 Salimans et al. 2017 — the conceptual skeleton

Not an LLM paper. The contribution is the black-box optimization pattern: perturb policy parameters, run complete episodes, score the whole rollout, update toward perturbations that performed better. Seed-replay communication makes it parallel-friendly across workers (only scalar rewards transmitted).

Key findings that transfer to Hunter Seeker:

- **Whole-episode parameter perturbation suits sparse, delayed, long-horizon rewards.** ARC-AGI-3 score is squared per level and weighted by level index; this is exactly the regime ES handles better than per-step credit assignment.
- **Random-init brittleness is real.** Section 2.2: vanilla Gaussian perturbations on randomly-initialized CNN policies tended to encode constant-action policies regardless of input. They fixed it with virtual batch normalization. *Implication for HS:* if any ES target is a randomly-initialized nonlinear module, expect collapse to constant outputs unless bootstrapped or stabilized. Tuning bounded gates around already-meaningful machinery doesn't have this problem.
- **Antithetic sampling and rank-shaped fitness are practical necessities, not optional.** Qiu et al. omitted them to demonstrate vanilla ES; we cannot afford that on laptop hardware.
- **Sample efficiency is unflattering.** ES used 3-10× more data than A3C on Atari, won 23/51 games, lost 28/51. The sample-efficiency hype is mostly downstream of the LLM paper, not the original.

### 1.2 Qiu et al. 2025 — the LLM fine-tuning case

Full-parameter ES on Qwen2.5 (0.5B-7B) and LLaMA-3 (1B-8B). Headline: ES outperforms PPO/GRPO on Countdown across all model sizes, with population N=30 (vs 10,000+ in older neuroevolution work) and <20% of RL's sample evaluations.

The implementation details that transfer:

- **Seed-only noise storage.** Workers reconstruct perturbations from seeds; never store the noise tensor.
- **Layer-level in-place perturb / restore.** Memory peak is one layer's worth of float, not the whole population.
- **Z-score reward normalization per generation.** Removes reward-scale sensitivity.
- **Greedy decoding during evaluation.** Implementation point 5: *"the perturbed models use greedy decoding to generate the responses for reward evaluations. As a result, the perturbed models are evaluated deterministically, so that all performance differences come from the exploration in parameter space instead of action space."* Critical for Hunter Seeker: ES rollouts must use ε=0 argmax, not the ε=0.5 used in normal training, otherwise parameter-space variance contaminates with action-space variance.
- **Hyperparameters used:** N=30, σ=0.001, α=5×10⁻⁴ for Countdown; σ ∈ {0.0005, 0.001, 0.0015}, α=σ/2 for conciseness.
- **Conciseness fine-tuning works on TWO PROMPTS.** Table 4. ES can do meaningful behavior change with a tiny training set — relevant for per-game solver work.

The most interesting unverified claim: their parameter-magnitude shift histograms (Section A.5) show that on Countdown, ES updates look statistically indistinguishable from random walk; only on Qwen2.5-7B conciseness do they see systematic deviation toward "abundant small magnitude edits." Hypothesis: large pretrained models fine-tune through many small distributed changes — the LoRA prior. *Implication:* ES on a small randomly-initialized head behaves very differently from ES on a pretrained large model. The Qiu recipe does not transfer verbatim to fresh modules.

The pivotal sentence for this project (Section 5):
> *"Use ES to perform unsupervised fine-tuning based on internal behaviors of LLMs, such as confidence calculated based on semantic entropy and semantic density. Such fine-tuning cannot be done with RL, since action space exploration does not change the internal representations of LLMs."*

This is the structural fit between ES's mechanism and CLT's central thesis. The CLT evaluator scores Ouro loop states — a property of internal representations, not actions. RL with action-space exploration cannot directly optimize for "make Ouro's loop states better evaluators." ES can.

### 1.3 Sarkar et al. 2025 (EGGROLL) — low-rank ES at scale

Replaces the dense Gaussian perturbation E ∈ ℝ^(m×n) with a low-rank product √(1/r)·AB^T where A ∈ ℝ^(m×r), B ∈ ℝ^(n×r), r ≪ min(m,n). Memory drops from O(mn) to O(r(m+n)), forward pass cost from O(mn) to O(r(m+n)). 100× speedup at billion-parameter scale.

Two findings that surprised me on actually reading the paper (vs the abstract):

- **Low-rank can outperform full-rank ES on small RL nets.** Their RL experiments use 3-layer × 256-neuron MLPs; EGGROLL beat full-rank OpenES on 7/16 environments. They attribute it to "large networks are difficult to optimize for Open ES and lend themselves well to low rank updates." Low-rank is sometimes a better optimizer, not just a faster approximation.
- **Recurrent architecture parallelism.** Section 6.3: RWKV-7 fine-tuning is faster under EGGROLL than under transformers because RWKV has constant state size — KV-cache memory becomes population-evaluation memory. Ouro is a recurrent looped transformer; the same property may apply, though the engineering friction is real.

Strategic role for Hunter Seeker: **not the first ES implementation, but the right structure for any later Ouro-adjacent perturbation work.** Vanilla ES on bounded gating parameters is the right Phase 2.5; EGGROLL on small Ouro-adjacent LoRA modules is a Phase 4+ option.

---

## 2. Project-state alignment

`PROJECT_STATE_HUNTER_SEEKER.md` independently endorses ES targets matching what the papers fit best. Line 179-181:

> *"If another behavior patch is justified, make it a domain-general phase/action-effect competence patch: e.g. recovery branch diversification, candidate novelty, or action-effect contradiction handling; no wa30 route logic."*

Verified live elements that any ES integration must respect:

- `obs_effect_recovery_bonus` is a real score component (lines 5022, 5113, 5124, 5391) with observed values in `+0.0077` to `+0.1400` range.
- `model_basin_risk`, `viable_fraction`, `collapse_fraction` are real diagnostics (line 94: `viable_fraction=0`, `collapse_fraction=1`, `model_basin_risk=0.65` at the failure window).
- Recovery / escalation / reseed timing thresholds at 2 / 3 / 6 steps (line 79).
- Encoder is locked to v17b-compatible geometry (lines 505-514). Cosine vs v17b ≈ −0.0093 for the drifted encoder. Do not ES-train the encoder without explicit CLT anchoring.
- Self-model substrate is wired but unproven as the active agency layer (line 8925); default training runs with `--self_model_mode off`.
- Hard constraints (lines 166-170): no game ids, no color ids, no broad action blacklists, no wa30 route logic, every new safety/recovery term must remain bounded and visible in `score_components`.

---

## 3. Risks and prerequisites

### 3.1 Hard prerequisites (blocking)

**Perf patch on the late all-risky basin.** Line 175: *"the 250-275 window remains too slow and will block iteration."* ES with antithetic pairs at population N=16 across ls20/tr87/wa30 will compound this into multi-hour generations. The perf patch is a true prerequisite, not a parallel concern.

**Observation/action-effect learning cold-start.** Section 0.0.1's loop has supervised losses on changed-mask, next-frame CE, topology-delta, permanence contrastive, inverse-action, click-label, and confidence. ES tunes gates that select among proposals; if the proposals are uninformative, ES selects among noise. Go/no-go diagnostics:

- Inverse-action confidence/promotions are nonzero or clearly improving.
- `obs_engram_inferred_action_support` appears in relevant traces.
- Topology/event predictions show signal before phase-65 divergence.
- Observation losses are not flat/random.

### 3.2 Soft risks

- **Wall-clock dominance.** Even after the perf patch, ES will be rollout-bound. Episode caps must be enforced; antithetic pairs do not halve rollout cost.
- **Random-init brittleness.** Tuning a fresh MLP from random init replicates Salimans' Atari-CNN failure mode. Stick to bounded continuous weights around existing mechanisms on the first pass.
- **Reward hacking via loophole gates.** ES "intrinsically optimizes a solution distribution" (Qiu Section 5), making point hacks harder, but gate-level reward hacking (e.g. setting all gates to zero so the agent does nothing and avoids death penalty) is still possible. Step cost and progress reward must be in the fitness function.

---

## 4. Phased plan

### Phase 0 — Prerequisites
1. Perf patch on 250-275 wa30 window.
2. Observation/action-effect learning go/no-go diagnostics passing.

### Phase 1 — Add bounded gates
Add these as bounded continuous parameters, all visible in `score_components`:

- `nonviable_pressure_scale`
- `recovery_branch_diversity_scale`
- `candidate_novelty_scale`
- `obs_effect_contradiction_scale`
- `obs_engram_support_scale`
- `model_basin_risk_scale`
- `repeat_state_penalty_scale`
- `reseed_candidate_temperature`

**Hold the 2/3/6 recovery/escalation/reseed timing constants fixed on the first pass.** Those are control-flow thresholds. Letting ES move them on the first pass means ES failures are indistinguishable from threshold-tuning failures. Continuous weight tuning around fixed structural triggers is the right scope.

### Phase 2 — ES on those gates
Implementation: `claude_sandbox/es_outer_loop_codex.py` (do not write yet).

**Staged harness:**

- **Stage A — ES smoke**: ls20 + tr87, max_steps=180, population=8 (4 antithetic pairs). Purpose: verify plumbing, reward normalization, deterministic eval.
- **Stage B — wa30 windowed**: deterministic replay/prefix to just before phase-65 failure window, max continuation ~40-70 steps, population=8 or 16. Purpose: optimize recovery from the actual nonviable basin without paying for the first 248 steps every rollout.
- **Stage C — mixed validation**: ls20 + tr87 + wa30, population=16, max_steps as in current probes (180 / 180 / 290). Purpose: reject wa30-only hacks/regressions.

**ES eval mode (strict):**
- ε = 0.0, deterministic argmax action selection.
- Fixed env seeds / fixed replay prefix.
- Antithetic perturbations.
- Rank-normalized rewards.
- Early termination on obvious repeated collapse.
- All new terms visible in `score_components`.
- Constraint enforcement: no game ids, no color ids, no broad action blacklists, no wa30 route logic.

**Fitness function:**
```
R = env_score
  + level_completion
  + productive_recovery_after_nonviable_pressure
  + candidate_set_diversity_that_survives_rollout
  + λ * CLT_pairwise_tournament_winrate
  - death
  - repeated_state_loop
  - invalid/noop pressure
  - terminal-pattern repetition
  - step_cost
```

**Hyperparameter starting point** (scaled from Qiu Countdown):
- N = 16 (8 antithetic pairs)
- σ = 0.02 for scalar gates, 0.001-0.005 for any small-NN weights
- α = σ / 2
- Antithetic sampling: yes
- Rank-normalized rewards: yes
- Fitness shaping: yes (rank transformation)

### Phase 3 — CLT pairwise auxiliary
Add `λ * CLT_pairwise_tournament_winrate` to the fitness. The pairwise tournament is between two complete trajectories under the frozen CLT evaluator, leveraging the relational-not-absolute finding from the CLT paper (84.5% pairwise vs 21.75% independent). Random pairings within each generation are sufficient for an auxiliary signal — O(N) cost rather than the O(N²) of a full bracket.

This is the closest match to Qiu et al.'s "ES on internal-representation reward" suggestion (Section 5) and the structural fit with CLT's central thesis.

### Phase 4 — Self-model active variants
Controlled ladder/regression with `passive` vs `inject_aux_grad` modes. The self-model's evaluator/risk/memory/topology/observation diagnostics are inputs but the self-model does not yet own the evaluator loop as an internal controller. ES on self-model gate parameters is the natural empirical proof step the file says is missing.

### Phase 5+ — Ouro-adjacent LoRA via EGGROLL
Only after Phases 0-4 produce signal worth amplifying. Small rank-1 adapters on a few Ouro MLP layers, fitness = ARC level completion, encoder still frozen. This preserves the modular CLT thesis while allowing principled task-specific tweaks.

---

## 5. Implementation notes (deferred until after Phase 0)

When `es_outer_loop_codex.py` becomes worth writing:

- **`RecoveryPolicyParams` as a frozen dataclass.** Serialize into and read out of `score_components` on every step so parameter values are visible in the same dump format as everything else. ES experiments become inspectable with the same tooling as existing `claude_sandbox/perf_event_dumps_*` probes.
- **Seed-based noise reconstruction.** Per Qiu Algorithm 2 / Salimans Algorithm 2: store random seeds per worker, reconstruct perturbations on demand for both forward evaluation and update aggregation.
- **Layer-level in-place perturb / restore.** Even though the parameter vector is small, follow the pattern for memory hygiene.
- **Greedy decoding flag must be enforced inside the ES eval loop, not by convention.** A separate code path that asserts ε=0 during ES rollouts and `train_every=0` during ES generations.
- **Per-game reporting is mandatory.** The file already shows that policy pretrain and exact support can help one game while damaging another (Section 0.0). Aggregate fitness across the trio but track per-game deltas.

---

## 6. What this plan does *not* do

- Full-parameter ES on Ouro-2.6B. Ruled out by VRAM and the modular CLT thesis. Qiu makes it plausible in principle but not on this hardware.
- ES on the GridEncoder. Encoder drift to effective orthogonality vs v17b is documented; current rule is freeze unless CLT anchoring is explicitly active and measured.
- Replacement of observation pretraining with ES. Observation losses are differentiable and supervised; backprop wins there. ES sits *above* them, deciding how their outputs affect recovery and action choice.
- Tuning of terminal penalty constants as the main move. The file already says the system sees the bad basin; the missing piece is productive recovery source generation.
- Implementation of EGGROLL before vanilla ES has produced signal. The strongest EGGROLL foundation-model runs lean on RWKV/recurrent infrastructure; engineering friction with Ouro's transformer machinery is real.

---

## 7. Open questions

- **Qiu v2 ARC-AGI claim.** GPT 5.5 Pro's earlier response cited a static ARC-AGI experiment with Qwen2.5-14B (N=50, σ=0.001, α=0.0003, 1500 iterations) in Qiu et al. v2. The v1 paper does not contain this experiment. Verify v2 before citing internally.
- **ES on the encoder via CLT-anchored fitness.** The file's deferred research direction "encoder-cognition feedback" overlaps with the Qiu Section 5 internal-representation reward idea. Worth a separate plan once Phase 3 has data.
- **Pairwise CLT tournament cost vs signal tradeoff.** Random pairings give O(N) cost. Single-elimination bracket gives O(N log N). Round-robin gives O(N²). The right choice depends on how noisy the CLT signal is on out-of-distribution rollouts (the CLT was trained on HH-RLHF, not ARC trajectories).

---

## 8. Decision

**Next artifact is the perf patch on the 250-275 wa30 window, not `es_outer_loop_codex.py`.** ES becomes worth implementing only after both prerequisites hold.

The papers are useful. The plan is right. The sequencing has moved.
