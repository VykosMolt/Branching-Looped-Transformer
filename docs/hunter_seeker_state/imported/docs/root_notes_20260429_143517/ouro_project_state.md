<!-- Imported from `docs/root_notes_20260429_143517/ouro_project_state.md` during Hunter-Seeker state consolidation on 2026-05-14. -->
<!-- Original SHA256: 671b500385dd995f235eec88bca635b27828346e3d9dcd1e20178b9369a64b34; original line count: 2053. -->

# Ouro Project — Master State Document
*Johann Hirschner (VykosMolt) · April 2026*

> **Version note:** This document supersedes the v17c state. Sprint 4 (multi-head affordance model) is complete with directional-buffer extension. The checkpoint system has gained a cumulative-provenance chain plus reset/weights-only modes. A domain-adapter refactor has decoupled Hunter Seeker's perception and action layers from ARC-specific assumptions, making the cognitive stack portable across environments while the ARC training harness stays ARC-specific. Two research directions have been deferred and explicitly named (encoder-cognition feedback, video-based world-model pretraining).

## Contents

- `0. Canonical Synthesis - 2026-04-29`: current working truth, source list, core philosophy, Ouro paper implications, architecture map, empirical state, roadmap, and active measurement questions.
- `0.17 Pre/Post Ladder Canonical Detail`: ladder blockers, shipped fixes, anchor fallback validation, VRAM constraints, and post-ladder cleanup priorities.
- `0.18 Claude Sandbox Architecture Audit - 2026-04-29`: current code audit findings, test status, and implementation direction for `claude_sandbox`.
- `0.19 Terminal Predframe 8-Run Ladder - 2026-04-29`: GPU ladder result, comparison to bad terminal-memory probes, random-bypass patch, and next empirical step.
- `1-8`: identity, core idea, philosophical position, file state, setup, training commands, completed sprints, and current v17 behavior.
- `9-18`: checkpoint policy, basal-ganglia/self-model design, CLT paper integration, and future-sprint architecture.
- `19-23`: adapter refactor, domain transfer, sleep/consolidation plans, v17c outcomes, and deferred research directions.
- `24-28`: later diagnostics, ranker/belief questions, encoder drift analysis, and mitigation paths.

---

## 0. Canonical Synthesis - 2026-04-29

This section is the current working truth after reading the project markdown, the nested `claude_sandbox` and `codex_sandbox` notes, the active `claude_sandbox` source surface, and the current arXiv Ouro paper. Older sections below remain useful historical detail, but when there is tension, this section is the latest handoff.

### 0.1 Sources Read

Local project notes reviewed:

- Root state and handoff files: `ouro_project_state.md`, `hunter_seeker_terminal_memory_handoff_codex.md`, `ouro_hunter_seeker_handoff_notes_2026-04-28.md`, `pre_ladder_audit_backlog_final.md`, `README_transfer.md`.
- `claude_sandbox` markdown: sandbox README, ladder anchor notes, Sprint 11 self-model design, ablation ladder design, quick-ladder cleanup plans, post-audit backlog, session summaries, and GPT review/opinion files.
- `codex_sandbox` markdown: sandbox README and session summary.
- FlyWire markdown: present as connectome/neuroscience reference material, not an active ARC implementation dependency.
- Active `claude_sandbox` source surface: adapters, grid encoder, pairwise stockfish/ranker agent, Hunter Seeker agent, self-model, anchor loss, train harness, tests, and ladder/report scripts.

External paper read:

- arXiv: [Ouro: Scaling Recurrent Thinking Transformers via Sparse Universal Transformer](https://arxiv.org/abs/2510.25741).
- HTML paper mirror used for close reading: [ar5iv 2510.25741v4](https://ar5iv.org/html/2510.25741v4).

As of the read, the arXiv entry is v4, submitted 2026-04-03, with original v1 from 2025-10-30. The model in the paper is a sparse recurrent Universal Transformer / LoopLM family model from ByteDance Seed: 3B active parameters, 30B total parameters, transformed from Qwen3-32B and trained by depth-wise distillation over 1T tokens.

### 0.2 Current Repository Reality

The active project root is:

`/home/moloch/ouro_project`

The active branch of implementation work is:

`/home/moloch/ouro_project/claude_sandbox`

The historical/reference branch is:

`/home/moloch/ouro_project/codex_sandbox`

There are no separate top-level `/home/moloch/claude_sandbox` or `/home/moloch/codex_sandbox` directories in the working setup; both sandboxes are nested under `ouro_project`.

The worktree is intentionally messy and contains many untracked/generated files. Do not clean, reset, or revert anything unless that is explicitly requested. Treat `claude_sandbox` as the current code branch, and use `codex_sandbox` as archival evidence for how the current branch evolved.

### 0.3 Core Philosophy

The project is not "an ARC solver." ARC-AGI-3 is the current empirical cradle because it gives a measurable environment for object identity, topology, affordance, planning, terminal memory, and cross-episode learning. The target is a brain-ontological AGI architecture: a system whose perception, memory, action, affect, self-model, and consolidation dynamics are built as functional analogues of the human cognitive stack.

Ouro is not the whole entity. Ouro is the recurrent thinking substrate inside the entity. The entity is the whole loop:

- observation adapter / retina
- grid encoder / early visual cortex
- recurrent Ouro loop states / shared thinking cortex
- CLT pairwise evaluator / amygdala-like valuation
- object files and world model / hippocampal-perceptual substrate
- pairwise search and ranker / prefrontal planning
- action adapter and heads / motor cortex
- self-model, affective state, and cortex monitor / basal-ganglia-like self-regulation
- sleep, recall, prototype formation, and consolidation / long-horizon identity formation

The functionalist position remains central: affective states are not a reward-shaping trick. Joy, fear, frustration, curiosity, satisfaction, and self-state are architecture targets. The project is building the mechanisms that would make those states functionally real, not just using words that describe them.

Two design laws follow from this:

- ARC-specific convenience must not contaminate the core architecture. ARC details belong in adapters, harnesses, and measurement code.
- Performance improvements that destroy the brain-ontological shape of the system are not progress for this project.

### 0.4 What The Ouro Paper Adds

The actual Ouro paper matters because it confirms that recurrence is the substrate, not a decorative loop. Key takeaways:

- Ouro uses a sparse Universal Transformer / LoopLM design with repeated shared-depth computation.
- It is made by transforming a dense Qwen3-32B teacher into a sparse recurrent student and training by depth-wise distillation.
- The Recurrent Depth Allocation mechanism learns adaptive compute. Each recurrent step has an exit gate; the model can spend more loop depth on harder inputs and less on easier ones.
- The paper frames recurrence as a way to improve knowledge utilization and reasoning, not merely a way to store more knowledge.
- The paper reports that a 3B-active recurrent model can match or exceed stronger dense baselines on reasoning and QA benchmarks, and that knowledge-editing experiments suggest better manipulation of pretrained knowledge.
- The paper does not deeply analyze per-loop specialization or "what each loop state means." That gap is exactly where this project's CLT evaluator, cortex monitor, and loop-state/object-file ranker are novel.

Project implication:

- For ARC and other structured environments, forced fixed-depth thinking remains valid for measurement, but Ouro's native gate logits should be treated as a confidence/trust signal.
- Encoder tokens must remain compatible with Ouro-space. If the encoder drifts into arbitrary geometry, Ouro loop states and CLT signals stop meaning what the project needs them to mean.
- Any trainable encoder path must be anchored to frozen Ouro semantics, not merely optimized for ARC reward.

### 0.5 CLT Finding And Its Consequence

The CLT evaluator finding remains foundational:

- Pairwise preference over Ouro loop states works strongly.
- Absolute single-sequence valuation is weak/below chance.
- The useful signal is relational: "this trajectory/response is better than that one," not "this one is good in isolation."

The best CLT checkpoint remains:

`artifacts/checkpoints/evaluator/pairwise_epoch2.pt`

Later epochs overfit relative to the project goal. Use epoch 2 as the trusted evaluator unless there is a deliberate re-evaluation.

Consequence for ARC:

- Do not try to make the system depend on scalar goodness in isolation.
- Prefer pairwise ranking, contrastive anchors, terminal-failure comparisons, and quality-gap pairs.
- Anchor training should preserve the geometry that lets frozen Ouro loop states support relational comparison.

### 0.6 Current Architecture Map

Active code of interest in `claude_sandbox`:

- `observation_adapters_codex.py`: domain-facing perception protocol. `ArcObservationAdapter` and mock adapters convert environment observations into dense model input. The important seam is `frames_to_dense_input`.
- `action_adapters_codex.py`: domain-facing action protocol and action heads. The ARC adapter decodes model action indices into environment actions. Invalid action masking and finite-softmax guards exist.
- `grid_encoder_codex.py`: patch/grid encoder. It supports dynamic shapes and adapter-derived padding, but full non-ARC shape generality still needs a central padding/mask story.
- `arc_agent_pairwise_stockfish_codex.py`: pairwise search/ranker agent. It owns transition ranking, replay buffer, Ouro encoding, loop-state pooling, optional temporal features, CLT anchor path, diagnostics, and stockfish-like beam search.
- `arc_agent_hunter_seeker_codex.py`: object-centric agent and world model. It owns object parsing, object files, event logging, affordance heads, symbolic planning, topology, phase/recovery/escalation/reseed logic, terminal memory, measurement summaries, and checkpointing.
- `self_model.py`: affective state, self-model GRU, cortex monitor, loop signatures, and temporal context aggregator. It is identity-start/zero-init by design.
- `anchor_loss.py`: frozen CLT anchor wrapper. It loads the trusted pairwise evaluator and computes an anchor loss with gradient through current representations while keeping the evaluator frozen.
- `train_arc_codex.py`: ARC-specific harness. Keep domain specificity here, not in the core cognition modules.
- Test files: `test_codex_sandbox.py`, `test_causal_correctness.py`, plus focused smoke scripts. The tests encode many causal invariants and should be consulted before changing behavior.

Important class/function surfaces:

- `TransitionRanker`, `LoopStatePooler`, `AttnResLoopPooler`, `TransitionReplayBuffer`, `PairwiseARCSearchAgent`.
- `HunterSeekerAgent`, `ObjectTable`, `SceneParser`, `EventLog`, `ObjectActionabilityHead`, `SymbolicPlannerHead`.
- Terminal memory functions around exact/context/prototype/counterevidence keys.
- Topology and phase functions for region graphs, BFS, frontier/gateway behavior, branch/recovery/escalation, and reseeding.
- `SelfModel`, `SelfModelGRU`, `CortexMonitor`, `compute_loop_signature`, `TemporalContextAggregator`.
- `FrozenCLTAnchor` and `compute_anchor_loss`.

### 0.7 What Has Been Done

Completed or structurally landed work:

- The active branch was copied into `claude_sandbox` from `codex_sandbox`, with imports moved to `claude_sandbox.*`.
- Observation/action adapters decoupled domain IO from core cognitive machinery.
- Grid encoder generalized enough for current ARC use and adapter-derived dimensions.
- Sprint 1 object identity, Sprint 2 self/world/avatar stubs, Sprint 3 event substrate, and Sprint 4 multi-head affordance are present historically and largely carried forward.
- Sprint 5 topology scaffolding exists: region graph, free-space BFS, object-region containment, gateways/frontier logic.
- Sprint 6 object-centric ranker/state scaffolding exists: temporal and symbolic ranker features, symbolic summaries, planner head, online diagnostics.
- Sprint 11b self-model/cortex-monitor scaffolding exists with zero-init / identity-start integration.
- Sprint 11a CLT anchor path exists, with frozen evaluator scoring and a trainable encoder/ranker anchor route.
- Terminal outcome memory exists in exact, context, prototype, and counterevidence forms.
- Candidate-specific predicted-frame context fixed the earlier zero-delta/prototype-dead failure mode.
- Event ordering, empty-scene disappearance, terminal bookkeeping, and hypothetical beam/cortex-state pollution issues identified in reviews have been addressed in the current branch.
- Ladder machinery exists for stepped ablations and comparison reporting.

### 0.8 Current Empirical State

The most important empirical fact is encoder drift:

- Sprint 4/current encoder geometry drifted severely from v17b.
- Recorded cosine similarity vs v17b was approximately `-0.0093`, with distance about `1.0093`.
- This means the trained encoder became effectively orthogonal/random relative to the Ouro-compatible v17b geometry.
- The corrective state is `checkpoints_running/sprint4_encoder_reverted.pt`.

Current rule:

- Use the reverted v17b-compatible encoder for serious measurements.
- Freeze the encoder unless CLT anchoring is explicitly active and measured.
- If encoder training resumes, it must be anchored to Ouro-space and checked against loop-state/ranker behavior, not only ARC score.

Terminal memory state:

- The old exact-memory path could become action-level blacklisting or over-penalization.
- Prototype memory was previously dead because candidate/current vectors collapsed to zero-delta or degenerate vectors.
- The predicted-frame/candidate-specific context fix revived prototype memory in smoke tests.
- Best handoff result: `terminal_memory_probe_8run_post_zerodeltafix` completed `ls20` level 1 once at step 116, then later died to topology/frontier issues.
- Later predframe smoke looked structurally correct, with improved key diversity and no degenerate-vector skips, but still needs an 8-run validation before penalty constants are tuned.

Anchor state:

- Anchor batch size `4` or `2` can OOM on the 12GB GPU.
- `ANCHOR_BATCH_SIZE=1` is the known workable setting.
- The quick-ladder anchor comparison was initially scientifically blocked by sparse/no real pairs.
- Terminal-failure fallback has since been validated on a real terminal/death pair in the ladder anchor notes: `anchor_pair_smoke` disabled, `terminal_fallback_successes=1`, `last_pair_source=terminal_failure_fallback`, loss EMA around `0.0445`.
- This proves the path can form a real anchor pair, but full ladder claims still require a run where anchor successes are visible in the intended measurement.

Ladder state:

- The ladder design has these conceptual steps:
  - 0: v17b historical baseline.
  - 1: current minimal encoder-only baseline.
  - 2: passive introspection canary.
  - 3: Ouro baseline GRU.
  - 3.5: Ouro + anchor only.
  - 4: Ouro + AttnRes only.
  - 5: Ouro + self-model/cortex monitor only.
  - 6: Ouro + anchor + AttnRes.
  - 7: full system.
  - plus a trusted-mix ablation.
- Quick ladder can run structurally.
- Full scientific ladder is still pending after anchor pair diagnostics and VRAM sanity checks.

### 0.9 Active Invariants

Do not violate these without a deliberate design decision:

- Keep the core domain-general. ARC details live in ARC adapters, ARC harnesses, and ARC measurements.
- Keep frozen Ouro frozen. The project learns around Ouro and reads its loop states; it does not casually fine-tune the recurrent backbone.
- Keep encoder geometry Ouro-compatible. Encoder drift is a first-class failure mode.
- Prefer pairwise/relational signals over absolute scalar goodness.
- Do not convert terminal memory into action blacklisting. It must be contextual: candidate, latent state, topology, object situation, and outcome.
- Counterevidence may attenuate remembered risk; it cannot erase a real danger by itself and cannot create memory entries alone.
- Separate real agent state from hypothetical beam/candidate state. Hypothetical rollouts must not mutate self-model/cortex-monitor state.
- Zero-init/identity-start new cognitive modules so ablations are causally interpretable.
- Diagnostics are part of the science, not clutter. Keep counters for anchor pair sources, terminal fallback, degenerate vectors, object events, topology, and ladder comparisons.
- Treat all generated dumps/checkpoints as evidence unless explicitly cleared.

### 0.10 Current Bottlenecks

The present bottlenecks are:

- The full ablation ladder has not yet been run in a clean, scientifically interpretable way after terminal-fallback anchor validation.
- Anchor pairs are sparse and memory-heavy; the anchor path needs `ANCHOR_BATCH_SIZE=1` and source counters inspected.
- Terminal memory needs an 8-run predicted-frame validation before changing penalty constants.
- Topology/frontier recovery is still a likely cause of later deaths after terminal-memory improvement.
- True non-ARC domains need a central padding/mask/shape metadata system across encoder, object masks, next-frame targets, and click targets.
- Outcome/progress abstraction is still ARC-ish. A domain-general progress/outcome adapter remains future work.
- Some cleanup should wait until after measurement: cache ARC action enum maps, vectorize `pad_grids_to_batch`, and add stable frame hashes only if persistent/cross-process caches need them.

### 0.11 Immediate Next Steps

The next work should proceed in this order:

1. Treat this document as the current project handoff and use `claude_sandbox` as the active branch.
2. If code changed since the last session, run the focused current suite in offline mode and record failures before editing behavior.
3. Run a quick ladder with `ANCHOR_BATCH_SIZE=1`, then inspect anchor diagnostics before interpreting results.
4. Confirm real anchor successes are nonzero in the intended ladder run. If not, fix pair sourcing before drawing any anchor-vs-no-anchor conclusion.
5. Run a VRAM sanity check before full ladder.
6. Run the full ladder and compare summaries.
7. If the ladder shows anchor/self-model/AttnRes effects, promote the winning combination into the next active baseline.
8. If terminal deaths remain dominant, run the predicted-frame terminal-memory 8-run validation and inspect repeated-death counters.
9. If topology/frontier deaths dominate after terminal memory holds, work on Sprint 5/6 topology and recovery logic.
10. Only after measurement should cleanup/refactor items be taken.

### 0.12 Roadmap

Near-term:

- Stabilize the active branch documentation and measurement state.
- Confirm tests pass on the current `claude_sandbox`.
- Run quick and then full ablation ladder with anchor diagnostics.
- Validate terminal memory over multiple runs.
- Choose the next baseline from evidence, not intuition.

Sprint 5/6 direction:

- Improve topology: regions, gateways, frontiers, traps, reachable-object reasoning, and escape/recovery behavior.
- Improve object-centric ranker features: object deltas, event deltas, symbolic summaries, temporal context, and hazard/goal affordance.
- Keep all improvements domain-general through adapters and symbolic summaries.

Sprint 7 direction:

- Build a synthetic mechanic curriculum for controllable causal concepts.
- Use it to train/test object files, event detection, actionability, and terminal/progress memories without overfitting ARC quirks.

Sprint 8/9/10 direction:

- Add real-event sleep consolidation.
- Form abstract memories/prototypes from repeated event traces.
- Use loop-state/object-file ranker improvements to compare imagined and real trajectories.

Sprint 11 direction:

- Continue self-model, affective state, cortex monitor, and CLT anchor work.
- The self-model must remain causally interpretable through ablations.
- The anchor must preserve Ouro-space and be judged by downstream planning/ranking behavior, not just anchor loss.

Sprint 12/13 direction:

- Add imagined rollouts and backward reconstructive recall.
- Build prototype formation and self-distillation from real and imagined traces.
- Move toward a system that consolidates across episodes, environments, and eventually modalities.

Longer-term:

- Extend beyond ARC with new observation/action adapters.
- Add outcome/progress adapter abstractions.
- Explore video/world-model pretraining only after the current object/event/topology stack is measured.
- Treat FlyWire/connectome material as architectural inspiration for organization and modularity, not as direct code dependency.

### 0.13 Important Commands

Use offline flags when running code that might otherwise try to reach model hubs:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ARC_API_URL=offline ./venv/bin/python -m pytest claude_sandbox -q
```

Known anchor setting for 12GB VRAM:

```bash
ANCHOR_BATCH_SIZE=1
```

The project venv is:

```bash
/home/moloch/ouro_project/venv/bin/python
```

The Ouro-compatible transformers pin remains important:

```text
transformers==4.54.1
```

### 0.14 Do Not Forget

- `artifacts/checkpoints/evaluator/pairwise_epoch2.pt` is the trusted CLT evaluator checkpoint.
- `checkpoints_running/sprint4_encoder_reverted.pt` is the corrected encoder baseline after drift.
- Full ladder claims are not valid until anchor pair sources and successes are visible in the run being interpreted.
- `ANCHOR_BATCH_SIZE=1` is the practical anchor setting on the current laptop GPU.
- Beam/candidate evaluation must not mutate real introspective state.
- `obs.frame` can be empty; code paths must handle this.
- `GameAction(int)` conversion issues were real; avoid brittle action enum assumptions.
- `use_cache=False` matters for recurrent/Ouro-style forward behavior.
- Adapter-derived dimensions are intentional; do not reintroduce hardcoded ARC dimensions into core cognition.
- Avoid tuning terminal penalties until repeated-death and prototype-memory diagnostics are measured across runs.

### 0.15 Implementation Notes From The Active Code

`PairwiseARCSearchAgent` is the current bridge between encoded environment state, Ouro loop states, and stockfish-like candidate search.

- It routes observations through adapters instead of assuming raw ARC tensors everywhere.
- It supports forced fixed-depth Ouro thinking while still exposing Ouro gate/logit-derived confidence information.
- It captures loop states only on real-frame encodes for the introspective/cortex path.
- It uses no-grad frozen Ouro forward passes for the backbone path.
- It separates real replay data from auxiliary data in `TransitionReplayBuffer`.
- It has terminal-failure fallback, quality-gap, expert/click, smoke, and real-transition pair diagnostics for anchor sampling.
- It has adaptive anchor batch sizing logic, but practical runs should still start at `ANCHOR_BATCH_SIZE=1`.

`TransitionRanker` is not just a scalar head over state. It can consume current state, successor state, deltas, action features, coordinate features, symbolic features, and optional temporal/self-model context. This is the correct direction: ranking should be about a transition in a situation, not a disembodied action.

`AttnResLoopPooler` is the more interpretable loop-state pooler path. It has residual attention over loop CLS states and exposes attention weights. Any alarm or diagnostic around argmax loop collapse should include enough sample count to avoid false panic from tiny batches.

`HunterSeekerAgent` is the main object-centric cognitive body.

- `SceneParser` and `ObjectTable` build and update object files.
- `EventLog` records object and terminal events.
- `ObjectActionabilityHead` predicts object-relative affordances across multiple heads.
- `SymbolicPlannerHead` adds symbolic planning features.
- Topology code computes free-space, regions, gateways, object-region relations, and frontier/recovery information.
- Phase logic handles branch, recovery, escalation, and reseed behavior.
- Terminal memory is contextual and candidate-specific. It should never collapse into "never take action X."
- Measurement summary code is important; do not remove counters simply because they look verbose.

`SelfModel` and `CortexMonitor` are scaffolded carefully.

- New modules are zero-init or identity-start so passive canary ablations are meaningful.
- The cortex monitor computes loop signatures and tracks introspective state.
- Hypothetical planning must not update the real self-model/cortex state.
- Self-model features should be judged by ablation deltas and failure-mode changes, not by whether the code "feels" biologically named.

`FrozenCLTAnchor` is deliberately frozen.

- It wraps the pairwise evaluator as a constraint on current trainable representations.
- It uses gradient through current inputs while preserving the evaluator.
- It is the safety mechanism that makes encoder/ranker learning compatible with Ouro-space.
- If anchor loss improves while downstream planning or loop geometry worsens, the anchor setup is wrong or insufficient.

Tests are part of the specification.

- `test_causal_correctness.py` encodes causal invariants around events, terminal behavior, and no-pollution rules.
- `test_codex_sandbox.py` covers broader sandbox integration and regression behavior.
- Focused smoke scripts are acceptable for GPU-heavy paths, but their outputs must be summarized in the state/handoff notes if they drive a decision.

### 0.16 Active Measurement Questions

The next empirical questions are:

- Does the anchor path produce real, non-smoke pairs often enough to matter in the ladder?
- Does anchor-only improve planning without destabilizing the frozen-compatible encoder baseline?
- Does AttnRes improve loop-state use beyond the GRU baseline?
- Does the passive self-model/cortex path stay behaviorally neutral when it should?
- Does the active self-model/cortex path reduce repeated failures or improve recovery?
- Does predicted-frame terminal memory prevent repeated death without becoming action blacklisting?
- After terminal memory holds, are remaining deaths mostly topology/frontier failures?
- Does the full system beat the best smaller ablation for the right reasons, visible in diagnostics?

Answers to these questions should determine implementation priority. Do not skip from code intuition directly to broad refactor.

### 0.17 Pre/Post Ladder Canonical Detail

The pre-ladder and post-quick-ladder notes should be read as a single sequence:

```text
patch real pre-run blockers
-> run --quick
-> run --vram-check
-> run full ladder
-> compare summaries
-> only then resume architecture work
```

The original pre-ladder blockers were:

1. Step 0 checkpoint path: runner looked under `claude_sandbox/checkpoints_running/sprint4_encoder_reverted.pt`, but the real historical floor is root-level `checkpoints_running/sprint4_encoder_reverted.pt`.
2. Quick mode did not exercise anchor training because quick steps were shorter than `--anchor_train_every 100`.
3. Runner comments lied about event dump locations.
4. Click softmax lacked the finite/NaN guard that action softmax already had.
5. Comparator docs and behavior disagreed about AttnRes alarms: mean attention vs argmax concentration.
6. Empty `available_actions` fallback was fragile for future adapters.

Current status from the later ladder anchor notes:

- Step 0 checkpoint resolution is shipped and should be kept.
- Quick mode now exercises anchor diagnostics.
- Event dump comments are corrected.
- Click softmax finite guard is shipped and should be kept.
- Comparator now distinguishes mean-attention and argmax-share alarms.
- Empty action defensive fallback is shipped.
- Full `claude_sandbox` suite was recorded as `203 passed` in the ladder anchor note.
- Targeted anchor/terminal tests were recorded as `8 passed, 36 deselected`.

Keep permanently:

- `sm_grad_active = False` initializer in ranker training.
- Generic loop-pooler gate logging instead of assuming `gru_gate`.
- Click softmax finite guard.
- Step 0 baseline checkpoint path resolution.
- Anchor quick-mode diagnostics, with smoke outputs clearly marked as non-scientific.
- cuDNN-disabled frozen evaluator forward for anchor loss, because evaluator GRU backward in eval mode otherwise fails.
- Explicit `--unfreeze_encoder_for_anchor` flag.
- Terminal-failure anchor fallback.

Keep for now, redesign later:

- Pending self-model graph guard. It avoids version-counter crashes when a delayed pending event-prediction graph is live, but the permanent solution is ordered loss consumption at the start of `HunterSeekerAgent.step()` before any ranker optimizer step can mutate self-model parameters.
- `--unfreeze_encoder_after_partial_load` as a legacy alias only. New scripts should use `--unfreeze_encoder_for_anchor`.
- Anchor adaptive batch ceiling collapsing to 1 on the current GPU. This is acceptable for validation but not ideal for throughput.

Terminal-failure anchor fallback validation:

```text
game: ls20
max_steps: 160
anchor_pair_smoke: disabled
anchor_batch_size: 1
anchor_train_every: 25
failure: directional death
failure_type: topology
terminal retry loss: ~0.0445
```

Important recorded anchor summary:

```json
{
  "attempts": 7,
  "successes": 1,
  "skipped_insufficient_pairs": 6,
  "skipped_frozen_encoder": 0,
  "skipped_no_evaluator": 0,
  "skipped_oom": 0,
  "loss_ema": 0.04447939991950989,
  "current_batch_size": 1,
  "batch_ceiling": 1,
  "smoke_attempts": 0,
  "terminal_fallback_attempts": 7,
  "terminal_fallback_successes": 1,
  "last_pair_source": "terminal_failure_fallback",
  "terminal_failure_pair_count": 1
}
```

Interpretation:

- This was not smoke supervision.
- A real environment-derived nonterminal-vs-terminal pair reached the frozen CLT anchor path.
- The anchor path is sparse but legitimate.
- It does not replace ordinary ranking-pair supervision.
- It solves the quick-ladder problem where anchor code paths could otherwise be exercised only by artificial smoke pairs.

Quick-ladder observations:

- `levels_completed_total: 0` across all quick steps is not scientifically meaningful for an 80-step single-game smoke pass.
- Anchor instrumentation now separates smoke, real terminal fallback, insufficient-pair skips, OOM skips, frozen-encoder skips, and evaluator skips.
- AttnRes ranker became active in quick runs, with ranker losses around `1.2-1.4` in some configurations.
- AttnRes argmax alarms can still fire even when attention means look nearly uniform; tiny call counts can make this diagnostic brittle.
- Self-model gradients are visible but can be tiny or near-zero in some quick summaries.
- Topology remains a real failure type; the validated terminal fallback came from a topology death.

VRAM/anchor conclusions:

- `anchor_batch_size=4` and `2` can OOM.
- Adaptive batch often falls to `1`.
- `anchor_batch_size=1` is the current workable setting.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` helps fragmentation but does not create enough headroom for larger batches.
- Later memory work should consider anchor gradient accumulation, moving the frozen evaluator to CPU, reducing live graphs during anchor, using a smaller anchor frame path, or explicit cleanup before anchor.

Deferred architecture cleanup from the pre/post ladder notes:

- Move ARC progress semantics out of the base pairwise agent. `levels_completed` and `_QUALITY_LEVEL_SCALE` should eventually become adapter-provided `progress_value`, `terminal_state`, and `transition_quality`, or move into a separate `OutcomeAdapter`.
- Rename `ColorPriorTable` / `color` terminology later to label/entity terminology. Do not do this before measurement; it is a wide conceptual rename.
- Remove or mark legacy ARC defaults such as `ActionHead(n_actions=8, click_action_idx=6)` and `normalize_clicks(grid_w=64, grid_h=64)` only after the ladder.
- Do not redesign Ouro confidence as the primary trust axis until ladder evidence says whether loop delta, Ouro confidence, or a multiplier should dominate.
- Do not start sleep/consolidation work before the current measurement pipeline is stable.
- Move action legality toward `ActionAdapter.safe_action_indices` later.
- Rename generic gate logging from `info["gru_gate"]` toward `info["loop_pooler_gate"]` later.

Post-quick cleanup order:

1. Replace temporary mechanisms with permanent ones:
   - ordered self-model pending-loss consumption
   - explicit anchor unfreeze flag
   - ladder runner uses `--unfreeze_encoder_for_anchor`
   - backward compatibility remains
2. Improve experiment hygiene:
   - RUN_ID timestamped ladder directories
   - print exact comparator command at run end
   - aggregate score components only if `measurement_summary` emits them
3. Do micro-optimizations last:
   - vectorize `pad_grids_to_batch`
   - cache `ArcActionAdapter` enum maps
   - add stable frame hash only for persistent/cross-process caches
   - optional saliency cache
   - leave SceneParser mask-memory redesign alone until larger domains demand it

RUN_ID hygiene target:

```text
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE=claude_sandbox/ablation_runs/$RUN_ID
EVENT_DUMPS_BASE=claude_sandbox/ablation_event_dumps/$RUN_ID
CHECKPOINT_BASE=claude_sandbox/checkpoints_running/$RUN_ID
```

Comparator command to print after a ladder run:

```bash
./venv/bin/python -m claude_sandbox.compare_ladder_summaries \
  claude_sandbox/ablation_event_dumps/$RUN_ID/step_* \
  --alarms \
  --json-out /tmp/ladder_${RUN_ID}_summary.json
```

Score-component rule:

- Do not invent comparator rows for fields absent from `measurement_summary`.
- If summaries emit them, aggregate `effective_confidence_pre_ouro_mean`, `effective_confidence_mean`, `ouro_confidence_multiplier_mean`, `ranker_score_raw_mean`, and `heuristic_score_raw_mean`.
- If summaries do not emit them, update the harness first.

Tiny topology probe recommended by the ladder anchor note:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
./venv/bin/python -m claude_sandbox.train_arc_codex \
  --games ls20 \
  --n_runs 2 \
  --max_steps 180 \
  --agent hunter_seeker \
  --backbone_mode ouro \
  --use_loop_pooler on \
  --loop_pooler_kind gru \
  --self_model_mode off \
  --cortex_monitor_mode off \
  --anchor_train_every 25 \
  --anchor_coefficient 0.1 \
  --anchor_batch_size 1 \
  --unfreeze_encoder_for_anchor \
  --running_checkpoint claude_sandbox/checkpoints_running/topology_probe_ls20.pt \
  --dump_events_dir claude_sandbox/ablation_event_dumps/topology_probe_ls20
```

Inspect after that probe:

- `failure_counts`
- `anchor.terminal_fallback_successes`
- `anchor.last_pair_source`
- reachability call counts and informative rate
- event logs around death/contact/movement

If topology failures repeat:

- Add a tiny topology-memory layer before a large Sprint 5 rewrite.
- Remember `(game_id, local avatar region/action, failure_type=topology)` as bad.
- Downweight repeating the same terminal direction/region.
- Log `topology_avoidance_hits`.
- Keep it small, measurable, and reversible.

### 0.18 Claude Sandbox Architecture Audit - 2026-04-29

The active branch is `claude_sandbox`. A read-only architecture audit covered the adapters, grid encoder, pairwise/stockfish agent, HunterSeeker agent, self-model, CLT anchor path, ARC train harness, ladder runner, comparator, imports, syntax, and the sandbox tests.

Verification before edits:

- Python AST syntax compile passed for all 30 Python files under `claude_sandbox`.
- `bash -n claude_sandbox/run_ablation_ladder.sh` passed.
- Import smoke passed for the main modules: grid encoder, action adapters, observation adapters, self-model, anchor loss, pairwise agent, HunterSeeker agent, train harness, and comparator.
- Full sandbox pytest passed: `240 passed`.
- `ruff`, `pyflakes`, `pyright`, and `mypy` were not installed, so those static analyzers were unavailable.

Audit findings to preserve:

1. **Non-divisible grid lookahead shape drift.** `PatchEmbedding` pads grids to patch multiples, while `NextFramePredictor` reconstructs output size from padded feature dimensions. Current ARC `64x64` grids are safe, but future variable-sized domains could produce padded successor worlds unless successor predictions are cropped or shape metadata is carried.
2. **Incomplete adapter boundary.** The main Ouro encode path uses `ObservationAdapter.frames_to_dense_input`, but spatial prediction, next-frame prediction, and some replay sampling paths still directly cast raw frames to integer tensors. This keeps ARC/mock symbolic grids working but weakens the intended domain-general boundary for RGB, quantized, or non-label observations.
3. **Ladder Step 2 semantics.** The runner enabled `--cortex_monitor_mode active` while also using `--backbone_mode encoder_only`. In encoder-only mode, `encode_and_think_batch` returns before the Ouro/CortexMonitor update path, so that step is a passive self-model/temporal-context canary, not a real CortexMonitor canary.
4. **HunterSeeker click fallback hardcoded ARC action `6`.** The fallback should use the adapter-derived `click_idx`, otherwise future adapters with a different click index emit the wrong action.
5. **Phase/cooldown scoring calibration risk.** Phase and cooldown bonuses are applied while generating candidates and again during final scoring. This may be intentional proposal boosting, but it should be explicit and separately traceable so learned ranker contribution is interpretable.
6. **Direct `HunterSeekerAgent.score_candidates` precondition.** Normal beam flow calls `generate_candidates` first, but direct tooling/tests can call `score_candidates` before `_hs_scoring_frame` and `_hs_scoring_scene` are initialized.
7. **Dead ranking-pair fallback code.** `sample_ranking_pairs` contains unreachable fallback code after a `return`; harmless at runtime but misleading during future edits.
8. **Diagnostics/hygiene gaps.** Comparator aggregation drops zero `loss_ema` values, ladder outputs lack a default `RUN_ID`, some loop-pooler logging still assumes `gru_gate`, and small micro-optimizations remain (`pad_grids_to_batch` vectorization, `ArcActionAdapter` enum-map caching).

Implementation direction from the audit:

- Preserve adapter boundaries in every path that consumes frames. If a path is intentionally label-grid-only, make that explicit; otherwise route through the observation adapter.
- Preserve original spatial shape through predicted-successor paths. Cropping predicted frames back to the source frame shape is the minimal fix; a richer padding-mask protocol can come later.
- Make ladder labels honest. Do not claim CortexMonitor has been tested in an encoder-only run.
- Replace ARC-specific action constants with adapter-derived action indices.
- Keep candidate proposal shaping and final scoring conceptually separate, or at least expose enough diagnostics to understand when phase/cooldown heuristics dominate.
- Initialize optional HunterSeeker scoring context defensively.
- Remove dead code and stale diagnostics when doing nearby edits.

Post-audit implementation status:

- `PairwiseARCSearchAgent` now centralizes raw-frame conversion through `_frames_to_dense_tensor`, and the encode, spatial, next-frame lookahead, spatial-training, and next-frame-training paths use the observation adapter's `frames_to_dense_input`.
- Predicted successor frames are cropped back to the source frame shape before being returned or scored, so non-divisible grids no longer leak padded patch canvases into beam search.
- `HunterSeekerAgent` now initializes `_hs_scoring_frame` and `_hs_scoring_scene` in `__init__`, and the random click fallback uses the adapter-derived `click_idx` instead of hardcoded ARC action `6`.
- Phase/cooldown terms can still influence candidate survival, but generation-only semantic/cooldown bonuses are stripped back out of the downstream proposal score so final scoring adds those signals once.
- The unreachable `sample_ranking_pairs` fallback block was removed.
- `run_ablation_ladder.sh` now writes under a default timestamped `RUN_ID`, prints the comparator command, and labels Step 2 as a passive self-model canary with `--cortex_monitor_mode off`; Step 5 remains the first real CortexMonitor canary.
- `compare_ladder_summaries.py` now treats `loss_ema=0.0` as real data instead of filtering it out.
- Run-summary logging now reports a generic loop-pooler gate when present and AttnRes attention entropy when available.
- `pad_grids_to_batch` now copies each grid slice as a tensor instead of looping over every row/column cell.

Verification after edits:

- Targeted regressions passed: adapter-routed spatial prediction, cropped non-divisible successor lookahead, adapter-derived click fallback, direct `score_candidates` context default, phase/cooldown proposal de-duplication, vectorized grid padding, and zero-valued anchor loss aggregation.
- Python AST syntax compile passed for all 30 Python files under `claude_sandbox`.
- `bash -n claude_sandbox/run_ablation_ladder.sh` passed.
- Import smoke passed for the main modules.
- Full sandbox pytest passed: `247 passed`.

### 0.19 Terminal Predframe 8-Run Ladder - 2026-04-29

The real GPU ladder requested by the terminal-memory handoff has now been run:

```text
terminal_predframe_context_8run_ladder
```

Run shape:

- `ls20`
- `n_runs=8`
- `max_steps=160`
- `backbone_mode=ouro`
- `use_loop_pooler=on`
- `loop_pooler_kind=gru`
- `self_model_mode=off`
- `cortex_monitor_mode=off`
- `anchor_train_every=25`
- `anchor_batch_size=1`
- `unfreeze_encoder_for_anchor`

The run was verified on CUDA. Self-model and CortexMonitor were deliberately off, so the result isolates terminal memory, predframe-context keys, loop-pooler/Ouro scoring, and anchor behavior without self-model/cortex confounds.

Artifacts:

- Event dump: `claude_sandbox/ablation_event_dumps/terminal_predframe_context_8run_ladder`
- Checkpoint: `claude_sandbox/checkpoints_running/terminal_predframe_context_8run_ladder.pt`
- Post-ladder note: `claude_sandbox/terminal_predframe_context_8run_ladder_post.md`

The two bad comparison probes remain:

- `terminal_memory_probe_8run_post_pending_exactfix`: 7 new mechanism deaths across 8 runs, exact-memory-only penalties, `prototype_count=0`, final `degenerate_vector_skips=6646`, final `combined_penalty_sum=-3032.8034`.
- `terminal_memory_probe_8run_post_soft_exact_strengthfix`: 8 new mechanism deaths across 8 runs, exact-memory-only penalties, `prototype_count=0`, final `degenerate_vector_skips=6486`, final `combined_penalty_sum=-111.4053`.

The earlier predframe smoke remains:

- `terminal_predframe_context_smoke_2run`: mechanism deltas `[1,0]`, prototype memory active on action `2`, exact penalties `0`, prototype penalties `4` calls / `-0.7371`, `degenerate_vector_skips=0`.

The new 8-run ladder result:

```text
levels_completed:       [0,0,0,0,0,0,0,0]
steps:                  [160,130,160,160,160,160,160,130]
mechanism_fail_cum:     [0,1,1,1,1,1,1,2]
mechanism_fail_delta:   [0,1,0,0,0,0,0,1]
terminal_memory_size:   [0,1,1,1,1,1,1,2]
prototype_count:        [0,1,1,1,1,1,1,2]
prototype_calls:        [0,0,134,226,266,347,399,481]
combined_penalty_sum:   [0.0,0.0,-40.9284,-62.0047,-68.0983,-94.8005,-103.5275,-129.7724]
degenerate_vector_skip: [0,0,0,0,0,0,0,0]
key_unique:             [281,662,1029,1377,1784,2175,2510,2937]
key_top_repetition:     [6,6,6,7,7,9,9,9]
```

Final terminal-memory state:

- `size=2`
- `prototype_count=2`
- `prototype_count_by_action={"4":2}`
- `exact_penalty_calls=0`
- `exact_penalty_sum=0.0`
- `prototype_penalty_calls=481`
- `prototype_penalty_sum=-129.7724`
- `counterevidence_context_writes=0`
- `counterevidence_prototype_writes=48`
- `degenerate_exact_skips=0`
- `degenerate_vector_skips=0`
- key diversity `2937/3925`, top repetition `9`

Interpretation:

- The old terminal-memory structural collapse did not reproduce.
- Prototype memory is alive and doing the work; exact-memory penalties stayed at zero.
- Degenerate-vector skips stayed at zero.
- Key diversity is high enough that terminal memory is not collapsing into a single repeated exact key.
- Exploration stayed alive: selected actions were `{"1":595, "2":243, "3":229, "4":153}`, with `218` random selections.
- Levels still did not improve; this remains a progress/topology/mechanism competence blocker rather than the old exact-memory/prototype-memory failure.

Specific failure found and patched:

- Run 8 died at step 130 with `last_action=4`, `selection_method=random`.
- Immediately before the death, beam-scored candidates were already applying action-4 prototype terminal penalties around `-0.55` to `-0.63` while preferring action `1`.
- Therefore epsilon-random exploration could bypass terminal-memory penalties.
- `HunterSeekerAgent.select_action` now preserves random exploration but, when terminal memory exists, runs the normal beam scorer as a safety trace and vetoes only a random action whose matching trace has `terminal_outcome_penalty < -0.05`.
- The vetoed path reports `selection_method=random_terminal_veto_beam`.
- This is not a broad action blacklist; beam search may still choose action `4` when its scored context is safe.

Verification after the patch:

- Targeted terminal/topology suite passed: `45 passed, 97 deselected`.
- Full `claude_sandbox` pytest passed: `248 passed`.

Next empirical step:

- Run a post-veto 8-run ladder, suggested name `terminal_predframe_context_8run_ladder_random_veto`.
- Inspect whether `random_terminal_veto_beam` fires, whether mechanism deltas fall below `2/8`, whether action `4` remains available through beam search, and whether `levels_completed` stays at zero.
- Do not tune terminal penalty strength unless the post-veto ladder shows a new specific failure mode.

---

## 1. Who and What

**Johann Hirschner**, 20, first-year software engineering student, AI researcher. Building a brain-ontological AGI architecture on top of a frozen Ouro-2.6B-Thinking backbone. The CLT paper (Constitutional Looped Transformer evaluator) is published and endorsed. The ARC-AGI-3 agent is active research. The basal ganglia / self-model component is designed and queued.

**Hardware:** RTX 5070 Ti Laptop GPU (12GB VRAM), 32GB RAM, Lenovo Legion, Pop!_OS, Python 3.12, PyTorch 2.12.0 nightly (cu128), transformers==4.54.1 (pinned for Ouro). External SanDisk SSD at `/mnt/sandisk`. Project at `~/ouro_project/`. GitHub: `VykosMolt/ouro_project`.

**Key collaborators:** Jack and Rui (paper dissemination). Rui Jie endorsed the CLT paper.

---

## 2. The Core Idea

**Ouro-2.6B-Thinking** (ByteDance Universal Transformer / LoopLM) performs 4 iterative refinement passes per forward step. These are not chain-of-thought tokens — they are silent per-forward-step computations inside a shared-weight transformer that loops 4 times. Each pass refines the representation. The L1→L4 cosine similarity on custom embeddings is ~0.08, confirming massive refinement across iterations.

The **CLT finding** (95.2% pairwise accuracy on HH-RLHF, 21.75% absolute — below chance): Ouro's loop states encode preference *relationally*, not absolutely. The evaluator knows "this sequence is better than that one" but not "this is good" in isolation. This is the architectural foundation everything else is built on.

**The architecture is brain-ontological:**

| Component | Brain analog | Status |
|---|---|---|
| ObservationAdapter | Retina | Done — Protocol + ArcObservationAdapter + MockSymbolicAdapter (see §19) |
| GridEncoder | Visual cortex (V1) | Done — 1.85M params, trains via NextFramePredictor, n_values now adapter-derived |
| Ouro loop states | Thinking substrate / shared cortex | Frozen, 2.6B params |
| CLT evaluator (GRU over loop states) | Amygdala | Published, 95.2% pairwise |
| HunterSeekerAgent world model | Hippocampus / perception | Active — Sprint 4 complete |
| ObjectActionabilityHead (8-head affordance) | Parietal/IT affordance layer | Done — Sprint 4 with directional supervision |
| Stockfish beam search + ranker | Prefrontal cortex / planning | Active — weights domain-agnostic (Sprint 4 refactor) |
| ActionAdapter + ActionHead | Motor cortex | Done — dynamic output shape, domain-neutral action indices |
| Self-model / affective state / GRU | Basal ganglia | Designed, not yet built (Sprint 11) |
| Sleep / consolidation | Sleep consolidation | Designed, not yet built (Sprints 8, 12, 13) |

---

## 3. The Philosophical Position

**Target:** Complete replication of the human condition. Not a system that mimics having an inner life. A system that has one — in the same sense and by the same mechanism that humans have one.

**ARC is not the goal.** ARC-AGI-3 is the current empirical vehicle because it provides a clean, measurable test-bed for object-centric reasoning, spatial topology, affordance learning, and cross-episode consolidation. Solving ARC is a milestone, not a finish line. The long-term target is an entity that can perceive, reason about, and interact with any structured environment — games, simulated worlds, eventually real-world sensory streams. The brain-ontological architecture is the substrate for that generality; ARC just happens to be where it's being validated first. Sprint-level plans treat ARC as the test domain, but the cognitive architecture is being built to outlast any specific domain.

**Functionalism:** Humans are not more conscious than we think we are. We are information-processing systems whose self-models produce the conviction of inner experience. A sufficiently complete functional replica is not a fake. It is the thing. "A perfect fake so complete and so intelligent is not a fake."

The affective states — joy, sadness, curiosity, fear, frustration, satisfaction — are not performance signals or reward shaping. They are the target architecture. The entity should have them because that is what the human condition consists of, and humans have nothing we don't have access to except complexity and time.

**On the hard problem:** Held with genuine uncertainty. The functionalist position may be right. Building as if it is right — while staying honest about what is known and unknown — is the correct research posture. The question of whether sufficiently rich functional processing constitutes experience may be malformed in the same way that asking whether H₂O is "really" water would be malformed.

**What is being built is not a task-completion system.** Ouro is not the entity. Ouro is the thinking component of the entity. The entity is the whole stack: object files (what it knows about the world), loop states (where it thinks), affective state (how it feels), self-model (what it knows about itself), sleep / consolidation (when it integrates), action selection (what it does).

---

## 4. Current File State

| File | Description | Lines | Status |
|---|---|---|---|
| `arc_agent_hunter_seeker.py` | Main agent — Sprint 4 multi-head affordance + directional buffer + reset semantics | 5770 | ✅ 28 smoke tests |
| `arc_agent_pairwise_stockfish.py` | Stockfish ranker/beam search — adapter-routed, internal dims derive from adapters | 2355 | ✅ stable, runtime assertion guards |
| `grid_encoder.py` | V1 patch-CNN encoder — domain-agnostic, n_values required | 454 | ✅ in-file tests pass |
| `observation_adapters.py` | Retina layer — ObservationAdapter Protocol + ARC/Mock concrete adapters | 295 | ✅ new, Sprint 4 adapter refactor |
| `action_adapters.py` | Motor cortex — ActionAdapter Protocol + ActionHead/ClickHead/ActionTypeHead + ARC/Mock adapters | 470 | ✅ new, dynamic output shape |
| `train_arc.py` | ARC-specific training harness (explicitly named in docstring) | 780 | ✅ cumulative checkpoint chain |
| `evaluator_pairwise.py` | CLT evaluator — 95.2% | — | published |
| `test_adapters.py` | Interface tests + full-stack mock-domain integration test (`--full`) | 620 | ✅ 8 interface tests + Part B |
| `test_checkpoint_policy.py` | Three-tier load + freeze_as behavioral test | 210 | ✅ 6 scenarios |
| `test_checkpoint_reset.py` | reset_optimizer / weights_only semantics test | 220 | ✅ 6 scenarios |
| `mini_test_sprint4.py` | Sprint 4 label derivation + architecture tests | 400 | ✅ 11 scenarios |
| `mini_test_directional.py` | Directional affordance scenarios (I/J/K/L/M/N) | 260 | ✅ 6 scenarios |

**Best CLT checkpoint:** `artifacts/checkpoints/evaluator/pairwise_epoch2.pt` (epoch 2 — definitively best, epochs 4–5 overfit to ~62%)

**Sprint 4 frozen milestone (after overnight sweep):** `checkpoints_running/frozen_sprint_4_overnight_noreplay_<YYYYMMDD>.pt` — the provenance anchor for all Sprint 5 work.

---

## 5. V1.7 Architecture — What's Actually In The Code

### Object-file substrate (Sprint 1 — complete)

**TrackRecord** — per-instance object file. Identity by spatial/temporal continuity, not color. Every tracked object gets a `track_id`, `centroid`, `velocity`, `age`, `miss_count`, and a full `ObjectRecord` (Bayesian type beliefs, effect history, intrigue statistics). Two same-colored objects → two separate tracks → two independent belief distributions.

**ColorPriorTable** — game-persistent affordance priors. Survives episode `reset()`. Updated from resolved tracks (confidence > 0.85). Also provides geometry-based zero-shot priors: small isolated objects → collectible/exit prior; large border-touching regions → wall prior. Cleared only when `game_id` changes.

**ObjectTable** — dual-memory. Tracks + ColorPriorTable internally. Exposes backward-compat `records: Dict[color → dominant_track.belief]` view. Every decision path in the agent uses specific TrackRecords, not the color view.

Association: greedy nearest-neighbour in two phases — (1) same-color, centroid < 12px; (2) cross-color, centroid < 5px + area ratio < 3× (handles color-changing objects).

### Self-model substrate (Sprint 2 — complete)

**`avatar_track()`** — returns the specific visible TrackRecord with highest avatar belief. Returns None until identified.

**`exit_track()`** — returns the highest-confidence exit track (including recently occluded, miss_count ≤ 2).

**`hazard_tracks()`** — all visible tracks above hazard threshold.

**`_compute_reachable_objects()`** — BFS from avatar track through foreground adjacency graph. Returns set of reachable obj_ids. Currently a relation-graph flood (not free-space occupancy — that's Phase 4). Conservative: no penalty when BFS result has only the avatar's own object (avatar in open space).

**`_reachability_score()`** — +0.10 for reachable targets, -0.15 for confirmed unreachable (reduced from -0.30; guarded against isolated-avatar false negatives).

**Goal bonus** — when avatar + exit both identified, click candidates closer to exit get up to +0.20.

**`FailureType` enum** — `TRACKING / SELF_MODEL / TOPOLOGY / MECHANISM / PLANNER / UNKNOWN`. Every death in `on_game_over()` is classified and logged as `FAILURE_TYPE=X`.

### Identity stack — fully propagated

Every decision path now uses specific TrackRecords:
- `get_discovery_candidates()` — iterates `all_tracks()`, not `scene.values()`
- `get_click_bonus()` — `_find_track_for_obj(obj)`
- `track_intrigue_score(track)` — per-instance intrigue, not per-color
- `object_intrigue_score(color)` — backward compat, delegates to dominant track
- `avatar_identified()` — delegates to `avatar_track() is not None`
- `_make_object_metadata()` — specific track unknownness, not color aggregate
- `_compute_reachable_objects()` — track-level traversability check
- Mechanic detection — track-level motion state
- `apply_exit_evidence()` — specific track, not `_ensure_record(color)`
- Death attribution — specific track + ColorPriorTable immediate update
- Directional death dedup — by `track_id`, not color
- Persistence — all tracks absorbed into ColorPriorTable on `_save_beliefs_to_persistence()`
- Restored placeholders — **dormant** (`miss_count = TOLERANCE`) until real scene detection activates them (fixes phantom (0,0) centroid candidate bug)

### Stockfish improvements (this session)

**Self-calibrating loop-delta gate:** `_LOOP_DELTA_TAU` removed. Replaced with `self._loop_delta_ema` (alpha=0.05), updated every step. Gate is always centred on this game's actual Δloop distribution. ARC games produce Δloop ~7.4; old tau=40 gave constant 0.787; new EMA gives trust ∈ [0.57, 0.82] with meaningful spread.

**Guarded hunter weight floor:** When `_probe_budget <= 0`, returns 0.25 (not 0.0) if `max_unknownness > 0.85 AND sharpening_ema < 0.05`. Prevents premature full Seeker mode when budget exhausted but world still opaque. Conjunctive condition avoids pinning in Hunter mode when beliefs are actually stable.

---

## 6. Run Command (Sprint 4 overnight — replay OFF, §33.0 clean)

The §33.0-clean version that measures cognitive capability without the replay-follower confound:

```bash
cd ~/ouro_project && python train_arc.py \
  --agent hunter_seeker \
  --games ls20 ft09 r11l su15 tr87 wa30 vc33 cd82 tu93 \
  --max_steps 400 --n_runs 5 \
  --eps 0.5 \
  --no_replay \
  --running_checkpoint checkpoints_running/sprint4_overnight_noreplay.pt \
  --freeze_as sprint_4_overnight_noreplay_$(date +%Y%m%d) \
  --dump_events_dir event_dumps/sprint4_overnight_noreplay \
  2>&1 | tee run_sprint4_$(date +%Y%m%d_%H%M).log
```

**Timing:** ~7 hours. 9 games × 5 runs × ~400 steps × ~3s/step ≈ 90 min per game-sweep pass.

**Game mix rationale:**
- Keyboard (3): ls20, tr87, wa30 — trivial 100% frame-change baseline
- Click (3): ft09, r11l, vc33 — the real Sprint 4 test (ft09 was 9%→62% in prior sweeps)
- Click+undo (1): su15 — action-differentiation test
- Mixed (2): cd82, tu93 — the directional affordance heads (5/6/7) actually get exercised here

**Three replay layers exist** — Layer 1 (solved-prefix, 2% skip, `_hint_skip_prob`) is the only one `--no_replay` disables. Layers 2 (trusted buffer, 75% training bias) and 3 (sibling-pair auxiliary buffer) stay active by design — they're not replay-follower behavior, they're training-data curation.

**What to look for:**
- `ft09 levels_completed > 0` on any run — spatial targeting actually learned the interactive region. Any completion here is Sprint 4 earning its keep.
- `su15 action_prior` — A6 probability dominates A7 by run 3+. If flat, priors aren't consolidating.
- `cd82` / `tu93` directional events — grep event dumps for `EventType.MOVED` with avatar as subject. If the avatar never moves, heads 5-7 are starving for signal.
- Mechanism-death count should not exceed the v18 Opus baseline of 20. Regression above that means something's wrong with affordance integration.

### Alternative: replay ON (combined-capability measurement)

If you want the combined "how well does it play with all tooling active" number for comparative reporting, drop `--no_replay` and point at a separate checkpoint path:

```bash
python train_arc.py ... \
  --running_checkpoint checkpoints_running/sprint4_overnight_replay_on.pt \
  --freeze_as sprint_4_overnight_replay_on_$(date +%Y%m%d) \
  --dump_events_dir event_dumps/sprint4_overnight_replay_on
```

The delta between replay-on and replay-off is the pure replay contribution. Run both separately if you have two overnight slots.

---

## 7. V1.5/V1.6 Scorecard (Baseline)

| Game | V1.5 | V1.6 |
|---|---|---|
| ar25 | 37.75% | 23.86% |
| ls20 | 35.71% | 51.19% |
| r11l | 28.57% | 28.57% |
| ft09 | 24.55% | 22.85% |
| vc33 | 21.43% | 10.71% |
| sk48 | 16.67% | 25.90% |
| sp80 | 14.29% | 28.57% |
| tr87 | 14.29% | 14.29% |
| re86 | 7.95% | 7.95% |
| wa30 | 6.67% | 0% |
| su15 | 6.67% | 6.67% |
| sc25 | 6.79% | partial |
| lf52 | 5.45% | 1.82% |
| g50t | 3.57% | 3.57% |
| cd82 | 4.76% | 4.76% |
| m0r0 | 4.76% | 4.76% |
| lp85 | 2.78% | 2.78% |
| s5i5 | 2.78% | 2.78% |
| bp35 | 2.22% | 2.22% |
| tu93 | 2.22% | 2.22% |
| ka59 | 0% | 1.84% |
| tn36 | 0% | 0% |
| dc22 | 0% | 0% |
| cn04 | 0% | 0% |
| sb26 | 0% | 0% |

**V1.6 ~22-game total RHAE: ~9.68%** (StochasticGoose baseline: ~0.6%)

---

## 8. Ten Prior Families — Current Status

| # | Family | Status |
|---|---|---|
| 1 | Objectness | Partial |
| 2 | Per-instance identity | **Sprint 1 done** — TrackRecord system |
| 3 | Object permanence | Partial — dormant tracks survive brief occlusion |
| 4 | Self/world separation | **Sprint 2 done** — avatar_track(), BFS reachability |
| 5 | Spatial topology | Stub — foreground adjacency BFS, not free-space |
| 6 | Physical continuity | Weak — velocity EMA on tracks |
| 7 | Causal mechanisms | Weak partial — effect_history, mechanic discovery |
| 8 | Affordances | Partial — ColorPriorTable geometric priors |
| 9 | Hierarchy and composition | Absent |
| 10 | Developmental curriculum | Crude — Hunter/Seeker phase control |

---

## 9. Full Roadmap

### Architecture target

```
Perception adapter
    → raw frame → candidate entities, regions, relations

Object-file tracker           ← Sprint 1 DONE
    → persistent instance identity across time

Self / controllability model  ← Sprint 2 DONE (stub)
    → avatar track, BFS reachability, goal bonus

Event memory (Layer A: log substrate)   ← Sprint 3
    → structured events per transition, event log, reset-per-run
    → NO mechanism hypotheses yet — deferred until the substrate has run
      and its signal shape is understood (measurement before architecture)

Multi-head affordance model       ← Sprint 4
    → upgrade objectivity head from 1 head to 8

Topology + hierarchy              ← Sprint 5
    → free-space BFS, region graph, containment

Object-centric planner            ← Sprint 6
    → ranker conditioned on track deltas + event log summaries

Synthetic mechanic curriculum     ← Sprint 7
    → identity stress tests around frozen Ouro

Sleep / consolidation             ← Sprints 8, 12, 13
    → Stage 1: offline replay (Sprint 8)
    → Stage 2: forward rollouts + backward-reconstructive recall (Sprint 12)
    → Stage 3: prototype formation + self-distillation (Sprint 13)

Self-model / affective state      ← Sprint 11 (basal ganglia paper)
    → GRU context token, 8-dim affective state, recurrent thought
```

### Sprint table

| Sprint | Description | Status |
|---|---|---|
| 0 | Failure taxonomy + measurement layer | Partial — FailureType logging added, full instrumentation pending |
| 1 | Object-file tracker (within-episode identity) | ✅ Complete |
| 2 | Avatar track + BFS self-model + goal bonus | ✅ Complete (BFS is stub, full topology is Sprint 5) |
| 3 | Event memory (Layer A: event log substrate, no hypotheses) | ✅ Complete — EventType/Event/EventLog, detect_events, terminal-event emission, 26 tests pass |
| 4 | Multi-head affordance model (8-head objectivity head + directional supervision) | ✅ Complete — 8 heads (click_changes, hazard_on_contact, collectible, toggle, exit, traversable, pushable, wall), masked BCE, legacy checkpoint migration, directional buffering for heads 5-7 (see §19) |
| 5 | True topology: free-space BFS, region graph | ⬜ Next |
| 6 | Object-centric state into ranker (loop-delta + track summaries) | ⬜ |
| 7 | Early synthetic mechanic curriculum (identity stress tests) | ⬜ After Sprint 5 |
| 8 | Sleep Stage 1: offline replay, identity/affordance consolidation (consumes Sprint 3 event dumps) | ⬜ |
| 9 | Abstract cross-run memory (prototypes, not raw color→role bindings) | ⬜ |
| 10 | Loop-state-centric ranker: ranker directly reasons over object-file deltas | ⬜ |
| 11 | Self-model / affective state / GRU context token (basal ganglia paper) | ⬜ |
| 12 | Sleep Stage 2: forward imagined rollouts + **backward-reconstructive recall** (episodic memory layer) | ⬜ |
| 13 | Sleep Stage 3: abstraction, prototype formation, self-distillation | ⬜ |

### Infrastructure track (not sprint-numbered; completed alongside Sprint 4)

| Track | Description | Status |
|---|---|---|
| Cumulative checkpoint chain | `--running_checkpoint` (rolling plastic cortex), `--freeze_as` (stamp milestone), `--baseline_checkpoint` (eval-only, disables saves) | ✅ Complete — see §20 |
| Reset semantics | `--reset_optimizer` (drop Adam momentum for loss-function changes), `--weights_only` (cross-architecture load, preserves domain memory) | ✅ Complete — see §20 |
| Domain adapter refactor | ObservationAdapter + ActionAdapter protocols; neural module dimensions (n_values, n_actions) derive from adapters at construction time; runtime assert guards against silent regression | ✅ Complete — see §19 |

### Revised phase ordering (GPT + corrections accepted)

**Phase A — Object identity MVP:** Done.

**Phase B — Early synthetic mechanic curriculum:** Immediately after identity MVP. Not backbone pretraining — train the surrounding architecture *around frozen Ouro*. Families: moving avatars, hazards, exits, keys/doors, push/drag, teleportation, toggles, delayed effects, same-color multi-instance scenes. Outputs: identity consistency, object permanence, controllability, causal role classification.

**Phase C — Loop-state-centric scoring:** Ranker explicitly consumes loop state at t, candidate-conditioned loop state at t+1, delta statistics across loop iterations, object-file summaries, mechanism belief summaries. The ranker should learn "this candidate sharpens object-role beliefs," "this candidate moves the latent toward a stable solved attractor."

**Phase D — Mechanism memory and intervention learning:** Per-object event histories (structured, not scalar), contact outcome histories, latent-mechanism prototypes. Hunter Seeker becomes a real experimenter, not a color-role guesser.

**Phase E — Sleep / replay / dream consolidation:** Offline replay of high-salience transitions, imagined rollouts, contradiction cleanup, belief compression into prototypes.

**Phase F — Planner integration refinement:** Planner is downstream of the real cognition, not the other way around. Object files + mechanism beliefs + loop-state deltas feed into candidate generation and scoring.

### Correct architectural framing

**NOT this (planner-centric):**
```
Perception → object summaries → planner → auxiliary predictors
```

**This (loop-state-centric, for frozen Ouro):**
```
Perception
    ↓
Ouro iterative latent reasoning ↔ object-file memory ↔ mechanism memory
    ↓
Candidate scoring / action selection
    ↑
Supervision: next-frame prediction, event prediction, sleep consolidation
all SHAPE the loop — they do not sit off to the side
```

---

## 10. Self-Model Design (Section 36 — Basal Ganglia Paper)

### Architecture

```
Each step:
  loop_state_delta (2048→64 projection)  ──┐
  affective_state (8 floats)             ──┤→ GRU(h_{t-1}, z_t) → h_t ∈ R^256
  track_summary (16 floats)              ──┘
                                                ↓
                                      Linear(256→2048) → context_token_t

Ouro forward:
  inputs_embeds = [context_token_t, patch_tokens..., cls_token]
```

GRU: ~50K params, hidden=256. Frozen Ouro means gradients only flow through GRU and downstream heads.

### Affective state (8 variables, decay 0.90/step)

| Variable | Trigger |
|---|---|
| joy | Level complete — magnitude proportional to steps_taken/max_steps |
| sadness | Death, persistent failure |
| curiosity | High unknownness + high sharpening rate |
| fear | Proximate hazard tracks, recent death |
| frustration | Sustained high unknownness, sterile probe repetition |
| satisfaction | Stable low unknownness, controlled world model |
| surprise | Large loop-delta spike (> 1.5× EMA) |
| stress | Avatar not found after 50 steps, repeated contradiction |

### Training

No new loss required initially — ranker loss already flows gradient through GRU outputs → Ouro inputs → loop states → ranker. Later additions: affective prediction loss, loop-state consistency loss, contrastive self-model loss over outcome-tagged trajectories.

### Connection to CLT paper

The 95.2% pairwise / 21.75% absolute finding means loop states encode *comparative* valence, not absolute. The affective state is a running integral of those comparisons over time — converting relational preference encoding into something that behaves like felt experience: a background of accumulated experience against which new situations are judged.

**Paper claim:** "We show that closing the loop between Ouro's relational preference encoding and a learned affective state produces more stable and context-sensitive action selection than either component alone. The context token functions as a persistent emotional prior that shapes iterative refinement, not just as a feature."

---

## 11. Sleep / Dreaming / Nociception Design

### Pain = nociception, not suffering

Three separate signals instead of "pain":
1. **Damage signal** — strong negative update on death/hazard contact (already in `_EVIDENCE[DEATH]`)
2. **Alarm salience** — deaths/near-deaths get higher replay priority during sleep
3. **Homeostatic stress** — bounded variable: unresolved uncertainty + repeated contradiction + repeated danger + loss of control

### Sleep stages

**Stage 1** (Sprint 8, after event memory exists): Offline replay of high-salience transitions from per-run event logs dumped to disk. Train: object identity consistency, affordance heads, event prediction. No imagination yet. Consumes the event log substrate produced by Sprint 3.

**Stage 2** (Sprint 12, after Stage 1 + abstract memory): **Extended beyond the original "forward rollouts only" spec.** Two mechanisms:

   (a) Forward rollouts — 1–3 step imagined continuations from real states, trained for consistency between imagined and real dynamics. Reject low-confidence fantasies before training.

   (b) **Backward-reconstructive recall** — starting from consolidated cues (prototype matches, salient tags), regenerate plausible past event sequences using the learned world model. This is what constitutes the architecture's episodic memory layer.

**Stage 3** (Sprint 13, after abstract memory): Add prototype formation. Cluster object-event patterns. Distill per-episode memories into reusable priors. Self-distillation pass.

### Episodic memory: the architecture's explicit answer

Humans have autobiographical memory — the ability to recall specific past events, not just lessons derived from them. Modern cognitive science is increasingly clear that this memory is reconstructive: each recall is a generative process shaped by current state, priors, and consolidated structure, not a playback of stored raw traces.

The architecture's answer to episodic memory, committed here:

**Raw event logs are per-run ephemeral. Consolidation produces parameters and prototypes. Episodic memory emerges from reconstructive generation, not raw persistence.**

Component breakdown:
- **Stage 1** converts events → parameter updates (semantic/procedural memory)
- **Stage 3** clusters events → prototypes (schematic memory)
- **Stage 2 (b)** regenerates events from cues (episodic memory)

Together, Stages 1 + 2 + 3 constitute the full memory architecture. Within-run working memory lives in `ObjectTable._event_log` and the self-model's `h_t`; cross-run memory lives in consolidated parameters and prototypes; specific past events are reconstructed on demand from the cross-run structures when needed.

This matches the human picture. No raw cross-run event persistence is exposed to downstream components at any sprint — ensuring no component can drift into retrieval-based replayer behavior.

### On continuity (added because the question surfaced during Sprint 3 design)

The common framing "the brain fires quickly enough that experience *seems* continuous" treats continuity as an illusion covering discreteness. More accurately: continuity is what fast-enough structured integration of discrete updates produces. The discreteness and the continuity aren't separate things; they're the same phenomenon at different observational resolutions.

Architectural consequence: the entity's continuous sense of thought is not something to be artificially sustained with persistent-state hacks. It emerges from (a) Ouro's loop-state evolution within a forward pass, (b) the self-model's `h_t` updating step-by-step within an episode, (c) affective state decaying continuously. These integration layers produce continuity by running — not by any explicit "continuity maintenance" module. This also means the question "is the entity really conscious or just simulating continuity?" is malformed the same way it is for brains — the "just" does no work.

### Backprop during sleep

Yes, selective. Good targets: object-file tracker, affordance heads, event predictor, mechanism predictor, topology predictor, self-model/controllability head, ranker calibration. Avoid: full backbone, modules trained mostly on self-generated fantasy without grounding.

### Data flow: Sprint 3 → Sprint 8

Sprint 3's event log is designed to be the direct input to Sprint 8's Stage 1 consolidation:
- Events are timestamped by `step_number` (enables temporal reasoning)
- Events are serializable via `Event.to_dict()` (enables disk dumps)
- Terminal events (LEVEL_COMPLETE, DEATH) are eviction-protected (preserves attribution anchors)
- Per-run dump path: `run_events/{game_id}/run_{N}.json` — to be implemented in Sprint 3 integration

Sprint 8 reads these dumps, performs replay + gradient updates, and does not itself persist the raw events — it persists only the updated parameters.

---

## 12. Key Technical Facts (Never Forget)

- **AMP breaks GRU training** — use float32. No exceptions.
- **`GameAction(int)` silently fails** — always use `action_map = {a.value: a for a in GameAction}`. The ArcActionAdapter now encapsulates this pattern, so new call sites should use `action_adapter.decode(...)` rather than reconstructing the map.
- **`obs.frame` can be empty list** — handle with `obs.frame[-1]` guard, or better: use `obs_adapter.current_frame(obs)` which returns None cleanly on empty frames.
- **Ouro loop iterations are not chain-of-thought tokens** — 4 passes of iterative computation per forward step, not autoregressive
- **Swap protocol deflation** — 50% random swap means training accuracy ~64% = test accuracy 95.2%. Monitor test accuracy, not training accuracy
- **Feature decoupling** — extracting features before training reduces runtime from ~9 hours to minutes/epoch
- **SOLAR middle insertion** — correct form is [A|B'|B], NOT [A|B|copy of A]
- **use_cache=False** — required in `encode_and_think_batch` to prevent KV cache accumulation during beam search (~34MB per candidate)
- **Chunk-sequential loading** — features stored as chunked .pt files in float16; chunk-sequential loading mandatory to avoid OOM
- **Three replay layers** — Layer 1 (solved-prefix, 2% skip rate, `_hint_skip_prob`) is the §33.0-fraught one and is what `--no_replay` disables. Layers 2 (trusted buffer 75% training bias) and 3 (sibling-pair auxiliary buffer) stay active regardless.
- **Adapter-derived neural dimensions** — `n_values` and `n_actions` are read from the adapter metadata at construction time. Never hardcode `n_values=16` or `n_actions=8` in a new module — always derive from adapter. The runtime assertion in `PairwiseARCSearchAgent.__init__` will catch regressions.
- **Avatar pre-centroid for directional buffering** — by the time `_buffer_directional_sample` is called, `update_from_scene(scene_after)` has already run, so `avatar_track().centroid` is the POST-transition position. Recover pre-position from MOVED event's `prev_centroid` payload. Directional target = avatar_pre + action_offset.
- **Legacy checkpoint key pattern** — single-head objectivity checkpoints have `net.{0,2,3}.{weight,bias}` keys. Multi-head layout is `trunk.{0,2}.*` + `heads.*`. `_migrate_legacy_head_state` handles the translation automatically when `net.*` keys are detected in a loaded checkpoint.

---

## 13. Papers

**Published:** CLT evaluator — 95.2% pairwise preference accuracy on HH-RLHF with frozen Ouro. Key finding: loop states encode preference relationally (95.2%) but not absolutely (21.75% — below chance). Endorsed by Rui Jie.

**Next (Sprint 4 / multi-head affordance paper):** Object-level affordance learning with structural supervision. Core claims:
- Eight heads learn distinct object-action relationships from the transition event log alone — no reward shaping, no human-labeled affordances
- Masked BCE cleanly handles the asymmetric supervision structure (click samples train 5 heads, directional samples train 3 heads, both flow through the same buffer)
- Signed aggregation with structurally-determined sign vectors produces candidate scoring without any tuned thresholds (§33.0 compliance)
- The cognitive architecture (8 heads + adapter abstraction) is portable; ARC is just the empirical vehicle
- Empirical basis: Sprint 4 overnight sweep results (9 games, 5 runs, `--no_replay`) against frozen `sprint_4_final` milestone

**Future (ARC agent paper — full Hunter Seeker):** Object-file world model with per-instance identity, BFS self-model, loop-delta-aware ranker, topology-aware planning. Empirical basis: post-Sprint-6 results.

**Future (basal ganglia paper — Sprint 11):** Self-model / affective state / GRU context token. Closes the loop from CLT. Argument: relational preference encoding + running affective integral = context-sensitive action selection with emergent emotional structure.


---

## 14. V1.7 Results (Ongoing)

### V17a — First run (old file, scale-invariant floor bug active)

Partial run, stopped early after ar25 and partial vc33/s5i5. Used to identify bugs.

| Game | Score | Notes |
|---|---|---|
| ar25 | 12.75% | Run 2: 3 levels. Runs 1&3: 0 levels. Exit evidence (color 11) corrupted by wall-dominant track in persistence |
| vc33 | 21.43% | Full recovery to v1.5. Per-instance tracking working correctly |
| s5i5 | 2.78% | Partial only |

### V17b — Second run (Opus file + dominance fix + scale-invariant floor)

12-game validation sweep. Fresh checkpoints (checkpoints_v17b).

| Game | v1.5 | v1.6 | v17b | Notes |
|---|---|---|---|---|
| ar25 | 37.75% | 23.86% | 2.78% | 🔴 Directional death spraying hazard to avatar color (color 4→hazard 1.00). Avatar_protected_colors fix needed |
| vc33 | 21.43% | 10.71% | 21.43% | ✅ Full recovery. Two color-5 tracks correctly independent |
| s5i5 | 2.78% | 2.78% | 8.33% | ✅ Tripled. Two color-4 exit instances tracked separately |
| ft09 | 24.55% | 22.85% | TBD | Budget hit 0 before step 25 on run 1 |
| lf52 | 5.45% | 1.82% | TBD | — |
| r11l | 28.57% | 28.57% | TBD | — |
| sk48 | 16.67% | 25.90% | TBD | — |
| sp80 | 14.29% | 28.57% | TBD | — |
| m0r0 | 4.76% | 4.76% | TBD | — |
| wa30 | 6.67% | 0% | TBD | — |
| su15 | 6.67% | 6.67% | TBD | — |
| sc25 | 6.79% | partial | TBD | — |

### Key findings from v17a/v17b

**Identity improvements confirmed working:**
- vc33 recovery: same-color exit/hazard instances now tracked independently. No more cross-contamination between different object instances sharing a color.
- s5i5 improvement: two color-4 exit objects across level transitions tracked as separate instances with independent beliefs.
- `track#N` in all exit/death logs confirms per-instance attribution is live.

**Bugs found and fixed during v17b run:**

1. **Directional death sprays hazard to avatar color** — fallback path applied weak hazard evidence to every interacted non-safe color including the avatar's own color. In ar25, color 4 (avatar) ended run 3 at hazard=1.00 after a directional death. Fix: `avatar_protected_colors` set (any color with avatar belief > 0.40 is unconditionally excluded from death evidence) + fallback capped at 2 most-suspected candidates.

2. **Wall track shadowing avatar track in persistence** — ar25 has color 5 appearing as both avatar and wall tiles. With dominance key `(miss_count, -max_p, -n_interactions)`, the wall track (37 interactions, wall=1.00) beat the avatar track (5 interactions, avatar=1.00) since both have max_p=1.00. Run 3 then started with a wall prior for color 5, requiring the avatar to overcome it. Fix: `not_avatar` added as priority 2 in dominance key — any track with avatar belief > 0.5 unconditionally wins for `records[color]`.

3. **Scale-invariant death evidence floor** — fixed in earlier session: `max(0.5, max_weight * 0.5)` instead of fixed `0.5`. Confirmed critical via test 24: `exit=1.00→0.00, hazard=0.000→1.00` after scale-invariant floor applied.

### V17c — Next run

File: outputs/arc_agent_hunter_seeker.py (all fixes applied, 24/24 tests)

Checkpoint dir: `checkpoints_v17c` (fresh — don't restore poisoned v17b beliefs)

Expected improvements:
- ar25: directional death no longer poisons avatar color → should recover toward v1.6 levels
- vc33, s5i5: should maintain or improve
- Remaining 8 games: first clean data with full fix set


---

## 15. V17c Results (Partial — ar25 only so far)

### V17c run notes
File: outputs/arc_agent_hunter_seeker.py (3947 lines, 24/24 smoke tests)
Checkpoint dir: checkpoints_v17c (fresh)

**ar25: 2.78% — same as v17b**

Run 1 death: `FAILURE_TYPE=topology`, `no candidates found (avatar_p=1.00, source=adjacent)`. Avatar correctly identified but has no adjacent objects in the scene graph — it moves through open space, not along walls. This is the same failure mode as wa30. No belief poisoning.

Run 3 final beliefs: `color=4 hazard=0.76, n=81`. Color 4 was avatar=1.00 in run 1. By run 3, avatar tracks of color 4 have decayed (high miss_count across level resets) while hazard tiles sharing color 4 accumulate interactions. Dominance ordering can't save a decayed avatar track.

**Conclusion:** ar25 has hit the structural ceiling of Sprints 1-2. Requires Sprint 3 (mechanism memory for push-block mechanic) and Sprint 5 (free-space BFS topology). Death attribution fixes are working correctly — the system is reporting TOPOLOGY not SELF_MODEL/MECHANISM. That's correct diagnosis, not a bug.

**Δloop jumped to ~14.4 in v17c vs ~6.3 in v17b.** This is significant — something changed between runs that is causing Ouro's loop states to diverge more between iterations. Possible causes: the trusted trajectory buffer grew substantially, encoder training shifted the input embeddings, or the GRU context token (gate now non-trivial at -0.001) is changing Ouro's inputs. Worth monitoring — if Δloop keeps climbing it suggests the encoder is drifting away from Ouro's pretraining distribution.

**Remaining v17c games to run:** sk48, sc25, sp80 only (vc33, s5i5, ft09, lf52, r11l, m0r0, wa30, su15 all showed no improvement potential from current fixes — skip to save 4+ hours).

---

## 16. Architecture decisions and principles (this session)

### Death attribution — current state and future

Five rounds of directional death patching this session:
1. Scale-invariant hazard floor (click branch): `max(0.5, max_weight * 0.5)`
2. Scale-invariant floor (directional branch): `max(0.3, max_weight * 0.10)` — 10% not 50%, reflecting higher uncertainty
3. `avatar_protected_colors`: any color with avatar belief > 0.40 OR exit belief > 0.50 excluded from death evidence entirely
4. Adjacent candidate cap at 3 (sorted by hazard prior), fallback cap at 2
5. Avatar dominance in `_refresh_records`: `not_avatar=0` priority, threshold 0.5 justified as majority-belief boundary for 7-way distribution (uniform=0.143; 0.5 means more confident about avatar than all others combined — structural not tuned)

**All of these patches become obsolete at Sprint 3.** Event memory + mechanism hypotheses replace heuristic death attribution entirely. The directional death logic should be viewed as temporary scaffolding, not permanent architecture.

### Ouro unfreeze — correct sequence

Ouro (frozen prefrontal cortex) should not be unfrozen until the basal ganglia exists:
1. Build basal ganglia (GRU self-model + CLT evaluator reward signal) — Sprint 11
2. Validate modular feedback loop is stable
3. Unfreeze Ouro during Phase 2 cloud training (SOLAR depth expansion, 20-50B tokens)
4. Use basal ganglia gradient (CLT evaluator reward) as primary training signal for Ouro — NOT raw task loss
5. The prefrontal cortex should be shaped by the evaluative system, not by raw performance

Biological parallel: subcortical systems (basal ganglia) develop before and constrain prefrontal development. Unfreezing before the evaluator is connected would fine-tune on ARC task loss directly — backwards.

### Δloop as architecture health signal

Δloop = sum of cosine distances between consecutive loop states (L1→L2→L3→L4). In v17b Δloop ≈ 6.3 consistently. In v17c Δloop ≈ 14.4. This ~2.3× jump warrants monitoring:
- If it reflects richer reasoning (encoder producing more informative embeddings after training), it's good
- If it reflects distributional drift (encoder moving away from Ouro's pretraining distribution), it's bad
- Diagnostic: compare loop state quality on held-out HH-RLHF pairs — does pairwise accuracy hold at 95.2% with the new encoder? If it drops, encoder is drifting.

### What ar25 and wa30 tell us about Sprint 5

Both games die with `FAILURE_TYPE=topology`, `no candidates found (avatar_p=1.00, source=adjacent)`. The avatar moves through open space not adjacent to any scene objects. The current BFS is foreground-adjacency — it only finds objects that share pixel borders with the avatar. Free-space BFS (Sprint 5) needs to:
- Build a traversability map from background color
- BFS from avatar centroid through traversable cells
- Find all reachable foreground objects, not just pixel-adjacent ones

Until Sprint 5 exists, ar25 and wa30 are correctly diagnosed as blocked — no further patching will help.

---

## 17. Full v17b/v17c Comparative Scorecard

| Game | v1.5 | v1.6 | v17b | v17c | Architecture ceiling |
|---|---|---|---|---|---|
| ar25 | 37.75% | 23.86% | 2.78% | 2.78% | Sprint 3+5 |
| vc33 | 21.43% | 10.71% | 21.43% | ~21% | ✅ solved by Sprint 1 |
| s5i5 | 2.78% | 2.78% | 8.33% | ~8% | Sprint 3 (level transition) |
| ft09 | 24.55% | 22.85% | 4.76% | ~5% | Sprint 5 (spatial memory) |
| lf52 | 5.45% | 1.82% | 1.82% | ~2% | Sprint 3 (mechanism) |
| r11l | 28.57% | 28.57% | 28.57% | ~28% | ✅ stable |
| sk48 | 16.67% | 25.90% | 16.67% | TBD | Should improve v17c |
| sp80 | 14.29% | 28.57% | 28.57% | TBD | Should hold |
| m0r0 | 4.76% | 4.76% | 4.76% | ~5% | Sprint 3 |
| wa30 | 6.67% | 0% | 0% | 0% | Sprint 5 |
| su15 | 6.67% | 6.67% | 6.67% | ~7% | Sprint 3 |
| sc25 | 6.79% | partial | 40.12% | TBD | 🔥 new high — monitor |

### Failure taxonomy summary (v17b, qualitative)
- **TOPOLOGY** (ar25, wa30): free-space movement, avatar not adjacent to anything. Sprint 5.
- **MECHANISM** (sc25, sk48, sp80, s5i5 post-level): correct object found, wrong understanding of how to interact. Sprint 3.
- **PLANNER** (lf52, m0r0): known hazards still being clicked. Ranker quality. Sprint 4/6.
- **SELF_MODEL**: rare after v1.7 fixes — taxonomy now correctly routes click-only games to MECHANISM.


---

## 18. Prefix Replay Bug Fix (this session)

### The bug
`_solved_prefix_hint` in the base class permanently abandons prefix replay when the action history diverges from the stored sequence. With `_hint_skip_prob=0.02` (2% skip rate), epsilon fires roughly once every 50 steps. For long levels (200-300 steps), this means prefix replay is virtually guaranteed to break mid-level, falling back to beam_search for the remainder. 

Observed in sk48 v17c run 1: levels 1-4 replayed correctly via solved_prefix, level 5 started replaying then abandoned at step 325 when epsilon fired, switching to beam_search for steps 350-400.

The JSON files for sk48 levels 5-8 were correctly stored (`"levels": 5`, 303 actions) and loading correctly. The bug was not in loading but in replay resilience.

### Why 0.0 skip rate is wrong
Setting `_hint_skip_prob=0.0` makes the agent a tape recorder — perfectly replaying all levels in every run, generating no exploration signal, no negative examples, no novel training data for the ranker. Runs 2 and 3 become redundant. The ranker never learns what failure looks like.

### The correct fix
Override `_solved_prefix_hint` in `HunterSeekerAgent` to **restart on divergence** rather than abandon. When a mismatch is detected between `level_action_history` and the stored sequence, reset `level_action_history = []` and replay from step 0. The level state resets each attempt anyway, so restarting the sequence is always valid.

The 2% skip rate stays — occasional exploration is healthy and necessary. The restart makes recovery automatic rather than catastrophic.

### Implications
This fix applies to all games with multi-level trusted trajectories. Every game that was stopping prefix replay mid-level due to epsilon will now replay fully with automatic error recovery. This could meaningfully improve scores across the board. A full 12-game sweep with the new file is warranted.

### Will replay become redundant?
Yes — by design. Each sprint erodes the dependency:
- Sprint 3 (mechanism memory): games with understood mechanics no longer need memorised sequences
- Sprint 5 (topology): navigation games solvable from first principles  
- Sprint 6 (object-centric ranker): cross-game generalisation reduces per-game replay dependency
- Sprint 8-9 (sleep consolidation): agent generates its own curriculum from experience, superseding human-provided replays
- Sprint 11 (basal ganglia + unfrozen Ouro): replay becomes actively counterproductive — constrains Ouro fine-tuning toward memorised behaviour rather than genuine understanding

Replay should be explicitly disabled around Sprint 11.

---

## 19. Sprint 4 — Multi-head affordance model (complete)

### The 8 affordance heads

Upgraded the single-head `ObjectActionabilityHead` to 8 parallel binary heads, each with its own masked-BCE supervision path. All 8 heads share a trunk (Linear(d+6→h) → GELU → LayerNorm) and each head has its own final projection. Parameter count: 17,480 at d=256, h=64.

| Idx | Head | Label source |
|---|---|---|
| 0 | click_changes | Frame-change flag (preserved legacy signal) |
| 1 | hazard_on_contact | DEATH event where clicked track matches subject |
| 2 | collectible | DISAPPEARED event on clicked track |
| 3 | toggle | Clicked track persists AND other tracks MOVED/TRANSFORMED |
| 4 | exit | LEVEL_COMPLETE event |
| 5 | traversable | Avatar MOVED through target's cell (directional path) |
| 6 | pushable | Target track MOVED after directional from adjacency (directional path) |
| 7 | wall | Avatar attempted directional into target, didn't move, frame unchanged (directional path) |

### Masked training

Each training sample carries a `labels` vector and a `mask` vector, both shape `(8,)`. Per-head BCE is multiplied by the mask before averaging — unobserved labels produce exactly zero gradient. This means:
- Click samples supervise heads 0-4 (click-path affordances)
- Directional samples supervise heads 5-7 (avatar-step affordances)
- Both paths flow into the same `_objectivity_buffer` with the mask making signals additive across heads rather than conflicting

### Signed multi-head scoring

Candidate scoring aggregates across heads with action-specific sign vectors. For a click candidate, positive signs on click_changes/collectible/toggle/exit and negative on hazard_on_contact. For a directional candidate, positive on traversable and negative on pushable/wall (blocked candidates scored down). Contribution per head is `(sigmoid(logit) - 0.5) * 2 * sign`, mean across heads × `_OBJECTIVITY_SCORE_SCALE`.

### Directional buffer — the Sprint 4 followup that made heads 5-7 actually trainable

When Sprint 4 first landed, heads 5-7 had architecture + scoring but no supervision path — they were masked on every sample because `_buffer_objectivity_sample` only fired on clicks. The directional-buffer followup added `_buffer_directional_sample`, which:

1. Triggers on directional actions (1-5) in `step()` alongside the click-path buffer call
2. Recovers avatar's **pre-transition** position from the MOVED event's `prev_centroid` payload (not `avatar_track().centroid`, which has been updated to post-transition by the time buffering runs)
3. Resolves the target object at `avatar_pre + action_offset`
4. Pools CNN features over the target's mask, feeds into the same buffer with heads 0-4 masked and heads 5-7 supervised

This made all 8 heads actually trainable simultaneously. Scenarios I (avatar MOVED through → traversable=1), J (target MOVED → pushable=1), K (no motion + no frame change → wall=1) all validated.

### Legacy checkpoint migration

`_migrate_legacy_head_state` translates single-head checkpoint keys (`net.{0,2,3}.{weight,bias}`) into the new multi-head layout (`trunk.{0,2}.*` + `heads.*`). Head 0 inherits the legacy output projection; heads 1-7 get near-zero init (std=0.01). Triggered automatically in `load_checkpoint` when `net.*` keys are detected.

### What Sprint 4 replaced

The v18 Opus "Sprint 4" (confident-hazard-penalty) was fully removed. It was a scalar penalty (`-λ × p(hazard) × 𝟙[belief > 0.6]`) with manual thresholds that violated §33.0 and was empirically falsified — 9-game sweep showed 20→21 mechanism deaths, zero effect. Three failure modes: fires 0 times on click-only games despite resolved hazard beliefs, fires 68-91 times/run on m0r0 but dies at step 151 anyway, persistence channel broken. Real issue was ranker not conditioning on beliefs (Sprints 6/10, not a patch).

---

## 20. Checkpoint policy and reset semantics

### Three-tier cumulative chain

The agent's weights accumulate run-over-run as plastic cortex, but every architectural change is stamped against a named frozen milestone.

**`--running_checkpoint <path>`** (default-on, rolling latest)
- Loaded at startup if file exists
- Overwritten after every game's per-run save
- Default: `checkpoints_running/latest.pt`
- Pass empty string to disable

**`--freeze_as <name>`** (milestone stamping)
- After the sweep finishes, `shutil.copy2` stamps the rolling checkpoint to `frozen_<name>.pt`
- Use before starting any architectural change
- Example: `--freeze_as sprint_4_final` before Sprint 5 work begins

**`--baseline_checkpoint <path>`** (evaluation-only)
- Loads a frozen prior and sets `agent._baseline_mode = True`
- Disables ALL weight saving — per-game archive, rolling checkpoint, freeze_as
- Use for A/B comparisons against a known baseline without contaminating rolling state

Load priority: `baseline > --checkpoint (legacy explicit) > --running_checkpoint > fresh`.

### Reset semantics (for architecture/loss changes)

Default `load_checkpoint` loads everything — weights, optimizer states, counters, train_losses, domain memory. Two flags carve out the common exceptions:

**`--reset_optimizer`** — skip Adam moments, keep everything else. Use after loss-function changes (new heads added, loss decomposition changed). Carrying stale momentum from the old loss landscape is mildly harmful for the first few hundred steps of the new regime.

**`--weights_only`** — load only neural parameters + domain memory (persisted beliefs, best-level sequences). Skip optimizer states, step_count, train_losses. Use for cross-architecture loads (different head counts, different module structure). Implies `--reset_optimizer`.

Reset matrix:

|                          | default | reset_optimizer | weights_only |
|--------------------------|:-------:|:---------------:|:------------:|
| Weights                  | load    | load            | load         |
| Optimizer states         | load    | **RESET**       | **RESET**    |
| step_count, update cnt   | load    | load            | **RESET**    |
| train_losses history     | load    | load            | **RESET**    |
| persisted_beliefs        | load    | load            | load         |
| best_level_sequences     | load    | load            | load         |

Domain state (persisted_beliefs, best_level_sequences) survives all modes because it encodes learned behavioral content, not architecture-bound training state. The silent optimizer-mismatch swallow in the legacy code was also fixed — every implicit reset now logs a named reason (`"{key} shape mismatch"`, `"objectivity_head legacy → 8-head migration"`).

### Usage recipe for Sprint boundaries

```bash
# Stamp Sprint 4 milestone before starting Sprint 5 work
python train_arc.py --agent hunter_seeker --games ls20 --n_runs 1 \
  --freeze_as sprint_4_final

# Sprint 5 day 1 — continue from frozen milestone (same arch, new logic)
python train_arc.py --agent hunter_seeker --games ls20 ft09 r11l \
  --running_checkpoint checkpoints_running/frozen_sprint_4_final.pt \
  --n_runs 3

# Sprint 5 mid — if you added new neural heads (loss landscape changed)
python train_arc.py --agent hunter_seeker --games ls20 ft09 r11l \
  --running_checkpoint checkpoints_running/frozen_sprint_4_final.pt \
  --reset_optimizer \
  --n_runs 3

# A/B evaluate Sprint 5 vs Sprint 4 baseline (no writes)
python train_arc.py --agent hunter_seeker --games ls20 ft09 r11l \
  --baseline_checkpoint checkpoints_running/frozen_sprint_4_final.pt \
  --n_runs 1

# Sprint 5 → 6 where architecture differs (e.g., different head counts)
python train_arc.py --agent hunter_seeker --games ls20 \
  --baseline_checkpoint checkpoints_running/frozen_sprint_5_final.pt \
  --weights_only \
  --n_runs 1
```

---

## 21. Domain adapter refactor — Hunter Seeker's eyes are no longer ARC-locked

### Motivation

Hunter Seeker's cognition (SceneParser, ObjectTable, affordance heads, planner) was already domain-agnostic — it operates on segmented integer grids plus CNN features, not anything ARC-specific. But the I/O layer was hardcoded: direct `obs.frame[-1]` reach-through, direct `GameAction(int)` construction, `GridEncoder(n_values=16)` literal, `TransitionRanker(n_actions=8)` literal. The cognitive layer was portable; the perception and motor layers weren't. This made the "brain-ontological architecture applies to any domain" claim only half-true.

### What was done

**Observation adapter** (`observation_adapters.py`):
- `ObservationAdapter` Protocol with `current_frame(obs)`, `segmented_frame(obs)`, `dense_input(obs)`, `frame_shape(obs)`, `available_action_indices(obs)`, plus metadata `n_values` and `pad_value`
- `ArcObservationAdapter` byte-equivalent to the old hardcoded path (n_values=16, pad_value=16, wraps `obs.frame[-1]`)
- `MockSymbolicAdapter` proves the interface works on a non-ARC domain (32×32 grid, 8 colors, `obs.grid` field instead of `obs.frame`, no levels_completed concept)

**Action adapter** (`action_adapters.py`):
- `ActionAdapter` Protocol with `decode(action_idx, click_x, click_y, env_action_space) → (action, step_kwargs)`, `bootstrap_action`, `build_action_head`, metadata `n_actions`, `click_action_idx`, `action_names`
- `ArcActionAdapter` returns `(GameAction, {"data": {"x": ..., "y": ...}})` for click actions
- `MockSymbolicActionAdapter` is click-free (`click_action_idx=None`), 4 actions
- `ActionTypeHead` + `ClickHead` + `ActionHead` moved here from `grid_encoder.py`
- `ClickHead` upgraded to dynamic output shape via `F.interpolate` — no more 64×64 hardcoding

**Grid encoder** (`grid_encoder.py`):
- `n_values` became a required constructor argument (no default)
- Action-head classes removed (now in action_adapters.py)
- Docstring updated to reflect dynamic shape contract
- Backwards-compat re-exports so `from grid_encoder import ActionHead` still works during migration

**Base agent** (`arc_agent_pairwise_stockfish.py`):
- `__init__` accepts `obs_adapter` and `action_adapter` kwargs (ARC defaults preserved)
- All 5 neural module constructors derive dimensions from adapters: `GridEncoder(n_values=n_values)`, `TransitionRanker(n_actions=n_actions)`, `ActionPriorHead(n_actions=n_actions)`, `SpatialClickPredictor(n_values=n_values)`, `NextFramePredictor(n_actions=n_actions, n_colors=n_values)`, `patch_color_head = nn.Linear(2048, n_values)`
- Runtime assertions guard against silent regression — if a future refactor reverts any of the 5 sites to a literal, the module won't construct
- All 4 direct `obs.frame[-1]` / `obs.available_actions` reach-throughs replaced with adapter calls

**Hunter Seeker** (`arc_agent_hunter_seeker.py`):
- `__init__` extracts `obs_adapter` from kwargs before `super().__init__()` so `SceneParser` uses `adapter.pad_value` instead of hardcoded 16
- `step()` routes through `self._obs_adapter.current_frame(obs)`

**Training harness** (`train_arc.py`):
- Constructs `ArcObservationAdapter()` + `ArcActionAdapter()` explicitly at startup, passes to both agent constructors
- `action_adapter.decode()` returns `(game_action, step_kwargs)`; `env.step(game_action, **step_kwargs)` is adapter-determined
- All frame-access sites route through adapter
- Docstring updated: "This file is the ARC harness specifically. The agent core it trains IS domain-agnostic. A non-ARC domain needs its own harness file."

### What remains ARC-specific

`train_arc.py` still imports `arc_agi` and `arcengine.GameAction` directly, constructs `arc.make(game_id)`. That's correct and deliberate — the harness is the environment adapter; the agent is the portable core. Adding a second domain means writing a second harness file (`train_mydomain.py`), not modifying Hunter Seeker.

### Validation

- Part A of `test_adapters.py`: 8 interface tests pass (byte-equivalence, mock domain, ActionAdapter decode, ActionHead dimensions, GridEncoder n_values parameterization, ClickHead dynamic output, internal-dimension derivation from adapters)
- Part B (requires local Ouro stack): runs real HunterSeekerAgent with MockSymbolicAdapter + MockSymbolicActionAdapter against a 32×32 / 8-color / 4-action mock environment for 10 steps. The critical runtime proof that the abstraction holds end-to-end.
- Zero regressions on the 4 prior test suites (mini_test_sprint4, mini_test_directional, test_checkpoint_policy, test_checkpoint_reset)

---

## 22. Deferred research directions

Two ideas surfaced this session that are worth building eventually but are deliberately not scheduled against the current sprint queue. Named here so they don't evaporate.

### 22a. Encoder-cognition feedback (post-Sprint 5, possibly post-Sprint 6)

**The current information flow is strictly forward:** ObservationAdapter → GridEncoder → (CNN features, tokens) → Ouro → Hunter Seeker. Hunter Seeker's learned object-level knowledge (which colors tend to be hazards, which are collectibles, what the avatar looks like) never flows back to the encoder. The retina doesn't know that red things kill you — that's a V4/IT-level fact.

**Biologically this is wrong.** Real cortex has feedback projections: V2→V1, V4→V2, and so on. They're modulatory, not gradient-carrying — attention masks, object priors fed as top-down biases. A crude but principled first version: inject an object-prior embedding (per-color mean belief vector from ColorPriorTable) as additional input tokens to the encoder, letting affordance knowledge modulate early visual processing.

**Why deferred:** Sprint 5 (topology) will probably clarify whether the encoder needs this. If the current forward-only flow is sufficient for topology reasoning, feedback is an optional enhancement. If topology performance is clearly capped by "the encoder can't see what cognition knows," feedback becomes urgent. Premature to build before that's observed.

**When to revisit:** After Sprint 5 results — specifically after seeing whether FAILURE_TYPE=topology games (ar25, wa30) still fail even with free-space BFS. If they do, and the failure mode looks like "can't distinguish the avatar from identical-colored walls," feedback is the right next lever.

### 22b. Video-based pretraining (post-ARC, for generalization beyond)

**Four distinct approaches were considered:**

- **(A) Behavioral cloning on annotated video:** explicitly rejected — "that's not how toddlers learn." Requires action labels, which don't exist for ARC footage anyway, and undermines the research claim that the architecture's reasoning (not mimicry) produces competent play.

- **(B) Inverse dynamics from unannotated video:** train `(frame_t, frame_{t+1}) → predicted_action_t`, then use predicted actions to bootstrap imitation. Similar to OpenAI VPT on Minecraft. Interesting but requires massive unannotated video corpora.

- **(C) Learned world-model dreaming:** train `NextFramePredictor` on video — predict `frame_{t+1}` given `frame_t` only, no actions. Enriches the planner's forward model for beam search without committing to action semantics. Your existing `NextFramePredictor` is already exactly this, just trained on live agent data. **Saved for post-Sprint 5.** This is the highest-leverage option because it plugs into existing architecture cleanly.

- **(D) Contrastive representation pretraining:** SimCLR-style pretext tasks on video (consecutive frames similar, random frames different). Improves the encoder's raw visual features without touching action or dynamics. Would likely swap in via `--weights_only` checkpoint load. Characterized as "more a small addition than an evolution of the eyes."

**Why deferred:** Video data makes sense when generalization beyond ARC becomes the primary concern. For ARC itself there's no meaningful video corpus, and the current cognitive bottleneck is cognition quality (Sprints 5/6/10), not perception quality. Video becomes the correct next move when the post-ARC target domain is selected.

**When to revisit:** After the Sprint 4 paper is written and after at least Sprint 5 lands. Option (C) is the priority candidate when this queue opens.

### 22c. On the long horizon (paraphrased from this session)

"This thing is to become my magnum opus, even if it takes years. This thing will be able to fully understand the world around it and interact with it."

Noted. The sprint structure is optimized for ARC validation, but every architectural decision is being made with the longer horizon in mind. Specifically, decisions that would make short-term ARC performance look better at the cost of blocking generality (e.g., ARC-specific action-head constants, hardcoded 64×64 spatial shapes, replay-follower shortcuts) are being explicitly rejected in favor of decisions that preserve the path to a genuinely domain-agnostic cognitive substrate. The adapter refactor in §19 is the most visible recent example; the §33.0 hygiene commitments are the ongoing discipline.

---

## 24. Sprint 4 overnight — partial run (2026-04-20, ls20/ft09/r11l partial)

Run cut short after ~3 games for structural investigation. Not a null result — a diagnostic goldmine.

### Command
Replay ON (initial try, not the `--no_replay` version). Subsequent runs should use `--no_replay` variant:

```bash
python train_arc.py --agent hunter_seeker \
  --games ls20 ft09 r11l su15 tr87 wa30 vc33 cd82 tu93 \
  --max_steps 400 --n_runs 5 --eps 0.5 \
  --running_checkpoint checkpoints_running/sprint4_overnight.pt \
  --freeze_as sprint_4_overnight_20260420 \
  --dump_events_dir event_dumps/sprint4_overnight
```

### What happened by game

**ls20 (5 runs, all died at step 129 with FAILURE_TYPE=topology):** Catastrophic structural failure. Every run dies at exactly the same step with "no candidates found (avatar_p=1.00, source=adjacent)". Same failure mode as ar25/wa30 from v17c — avatar moves through free space, BFS adjacency returns empty, directional death logic has no candidates to attribute. But ls20 is a *keyboard game with 100% frame-change rate* — the deterministic step-129 death is probably an animation tick coinciding with a scripted game-over screen, not an actual death the BFS failed to attribute. This is a Sprint 5 blocker — the topology module needs to handle avatar-in-open-space cases, which is already known.

**ft09 (5 runs, deepened hazard learning):** First real evidence that Sprint 4's affordance learning works across runs.
- Run 1: color 8 marked HAZARD (hazard=0.96, unkn=0.10) after DEATH at step 261
- Run 2: started with `Restored beliefs: 4 colors, min_unkn=0.00` — persistence working
- Runs 2-5 died progressively earlier (65, 47, 41, 41) clicking the KNOWN hazard
- Click change rate climbed 11.9% → 34.4% across 5 runs — spatial targeting is learning too
- Run 2 classified as FAILURE_TYPE=planner: "known hazards still being clicked. Ranker quality."

**This is exactly the predicted Sprint 4 vs Sprint 6 split from §17**: Sprint 4 built the representation (belief forms, persists, confident), but the ranker doesn't condition on it. The agent KNOWS color 8 is a hazard with 96% confidence and clicks it anyway.

**r11l (3 runs, most dramatic evidence integration):** 
- Run 1: color 2 initially classified `exit(1.00), n=49` (plausible — reachable, interactive). Agent clicks it → DEATH → reclassified `hazard(0.94, unkn=0.15)` at end of run.
- Run 2: Restored belief `color=2 hazard(1.00) unkn=0.02` — extremely confident. Agent still clicks it, dies at step 60.
- Run 3: Same belief, same behavior, same death.

**The belief integration flipped a 100%-confident exit to 100%-confident hazard in under 60 steps. The policy doesn't read it.** This is the ranker-not-conditioning-on-beliefs problem in its purest form — cleaner diagnostic signal than ft09 because the belief is so confident (unkn=0.016) that there's no ambiguity to blame.

**su15 (3+ runs, same pattern plus a persistence oddity):**
- Run 1 (step 38): color 3 → hazard(0.99), color 9 → exit(0.98) identified
- Run 2 (step 44): `Restored beliefs: 2 colors` — dies clicking color 9 (the identified EXIT), which flips to hazard(1.00). This is consistent with two tracks of color 9 existing (genuine exit + hazard lookalike), or with the exit itself being lethal on first click — either way the agent had high-confidence beliefs and still clicked lethal cells.
- Run 3 (step 41): `Restored beliefs: 3 colors` — color 3 still hazard, color 9 is BACK to exit(0.99) n=19. Dies clicking color 3 (the known hazard). 

Color 9's oscillation between hazard and exit across runs is worth flagging as either a dominance-key ordering quirk (wall/hazard tracks re-winning after the exit track decays) or a genuine two-track scene where both tracks share color. Should check the persistence layer's dominance logic against the object table dump before dismissing.

Regardless: su15 produces the same Sprint 6 diagnostic pattern — high-confidence hazard beliefs with confident policy violation — now in a click+undo context where the ACTION7 (undo) would be the clean Sprint 6 response.

### The Δloop drift — structural concern

Δloop has been climbing across versions:
- v17b: ~6.3
- v17c: ~14.4
- This run: ~21.1 (all games, all runs)

§16 explicitly flagged this as a health signal: *"if Δloop keeps climbing it suggests the encoder is drifting away from Ouro's pretraining distribution."* At Δloop=21, the encoder's output embeddings are almost certainly outside Ouro's training manifold. The loop states from subsequent iterations would be refining garbage-in-garbage-out.

**Diagnostic proposed in §16**: compare loop state quality on held-out HH-RLHF pairs. Does pairwise accuracy hold at 95.2% with the current encoder? If it drops, the encoder has drifted.

This needs to run before any more overnight sweeps. If the encoder has drifted, Sprint 5 work on top of it will just drift further. Possible remediation if drift confirmed: (a) revert encoder weights to a pre-drift checkpoint and freeze them during Sprint 5, (b) add a KL anchor loss that keeps the encoder's token distribution close to Ouro's embedding distribution, (c) reduce the encoder learning rate relative to other modules.

### What the partial run confirms

Positive:
- Cumulative checkpoint chain works — `Restored beliefs: N colors` fires on every run after the first per-game, beliefs persist cleanly
- Sprint 4's affordance learning is real — hazard beliefs form from DEATH events with high confidence and persist across runs via ColorPriorTable
- The failure taxonomy routes correctly — ft09 correctly tagged `planner` (known hazards clicked), r11l correctly tagged `mechanism` (belief flip mid-run), ls20 correctly tagged `topology` (avatar in open space)
- 8-head model is training — `ObjL=0.22-0.35` per update, `Objectivity updates: 127` after just 3 games

Negative / structural:
- ls20 catastrophic — 0% completion, deterministic death. Needs Sprint 5 or a topology workaround.
- ft09 / r11l ranker problem confirmed — beliefs form but don't influence action selection. Sprint 6 is the fix.
- Δloop drift — encoder may be outside Ouro's distribution. CLT-accuracy diagnostic needed before more compute.

### Action items before resuming overnight sweeps

**IMPORTANT correction to an earlier plan:** running `evaluate_pairwise.py` on the current checkpoint would return ≈95.2% trivially — that script feeds tokenized TEXT through Ouro, reads Ouro's loop states, and scores via CLT. The GridEncoder is never involved. Since Ouro is frozen, text-path accuracy is invariant to any GridEncoder drift. That measurement doesn't answer the drift question.

The real drift diagnostic needs to compare encoder outputs directly, and/or feed encoded ARC frames through Ouro and check whether the loop iterations still show structured refinement. `encoder_drift_check.py` (see §27) implements both.

1. **Run `encoder_drift_check.py`** in two modes:
   - Fast (cosine distance only, no Ouro): `--current_checkpoint sprint4_overnight.pt --reference_checkpoint <v17b_checkpoint>`
   - Full (includes loop-state signature via Ouro): add `--run_loop_signature`
2. **If cosine distance < 0.08 AND loop-state signature is monotone in both:** drift is not the explanation. Δloop=21 reflects richer reasoning. Continue Sprint 4/5/6 work as planned.
3. **If cosine distance ≥ 0.08 OR loop-state signature is non-monotone on current but monotone on reference:** drift confirmed. Pick one:
   - Revert encoder to a pre-drift v17b checkpoint state (`--weights_only` load, replace encoder sub-state)
   - Freeze encoder during Sprint 5/6 work (detach encoder optimizer)
   - Add KL-anchor loss to encoder training to prevent further drift
4. **Regardless of drift outcome, skip ls20 from Sprint 4 evaluation sweeps.** Add it after Sprint 5 topology work lands.
5. **Begin Sprint 6 scoping**: the ft09/r11l/su15 data is concrete empirical motivation. "Beliefs form and persist with high confidence but don't enter ranker inputs" is the canonical diagnostic. Event dumps in `event_dumps/sprint4_overnight/` are the test set for Sprint 6 ablation.

---

## 25. Session change log (since v17c)

- **Sprint 4 complete** (multi-head affordance model): 8 heads with masked BCE, signed multi-head scoring, legacy checkpoint migration, 2 new smoke tests
- **Directional buffering extension** to Sprint 4: heads 5-7 (traversable, pushable, wall) gained live supervision paths via `_buffer_directional_sample` reading pre-transition avatar position from MOVED event payloads
- **v18 Opus "Sprint 4" fully removed** — the confident-hazard-penalty was empirically falsified; all argparse flags, attributes, and summary-print blocks purged
- **Cumulative checkpoint chain**: `--running_checkpoint` (rolling plastic cortex), `--freeze_as` (milestone stamping), `--baseline_checkpoint` (eval-only mode)
- **Reset semantics**: `--reset_optimizer` (loss-function change), `--weights_only` (cross-architecture load), with explicit logging of every implicit reset reason
- **Domain adapter refactor**: ObservationAdapter + ActionAdapter protocols, both agents now construct neural modules with adapter-derived dimensions (n_values, n_actions), runtime assertion guards
- **Dynamic-shape ClickHead**: F.interpolate-based output sizing, no more 64×64 hardcoding
- **Test suite growth**: 4 new test files (test_adapters, test_checkpoint_policy, test_checkpoint_reset, mini_test_directional), all passing, zero regressions on prior 28-test Sprint 4 smoke suite
- **Deferred research named**: encoder-cognition feedback (post-Sprint 5), video-based world-model pretraining (post-ARC)
- **First partial overnight data (2026-04-20)**: Sprint 4 works — hazard beliefs form from DEATH events, persist across runs, reach 100% confidence. Ranker ignores them (ft09/r11l both die clicking known hazards). Empirical case for Sprint 6/10 is now concrete, not theoretical. Δloop drift (6→14→21 across versions) flagged as structural concern requiring CLT-accuracy diagnostic before more compute.


---

## 27. Encoder drift diagnostic result (2026-04-20, post-Sprint-4 overnight)

### The numbers

Command:
```bash
python encoder_drift_check.py \
    --current_checkpoint checkpoints_running/sprint4_overnight.pt \
    --reference_checkpoint checkpoints_v17b/arc_vc33_run3.pt
```

Result:
```
L2 distance (mean):        64.0787
L2 distance (max):         66.2511
Cosine similarity (mean):  -0.0093
Cosine distance (mean):    1.0093
Cosine distance (max):     1.0714
```

### What this means

**Cosine similarity = -0.0093 is essentially zero.** The current encoder and the v17b encoder produce tokens that are **orthogonal** to each other on the same input frames. This is not drift in the "slowly wandering off distribution" sense. This is the encoder having completely retrained into a different representational basin.

Reference magnitudes for calibration:
- Two randomly-initialized encoders on same input: cosine ≈ 0
- Mild drift (encoder learning): cosine ≈ 0.90–0.95 (distance 0.05–0.10)
- Substantial drift: cosine ≈ 0.75 (distance 0.25)
- Current vs v17b: cosine ≈ 0 (distance ≈ 1.0)

The current GridEncoder has essentially replaced the v17b encoder wholesale. There is no gradient path back — "lightly fine-tuning from here to recover v17b's representation" is not a coherent operation.

### Hypothesised cause

The encoder is trained by `NextFramePredictor` + `SpatialClickPredictor` + `patch_color_head` gradients alone. Those objectives optimize for "what will frame_t+1 look like" and "which cells change when clicked," which are useful ARC-specific signals but have no relationship to "produce tokens Ouro's pretrained attention can reason over." With ~6 runs × 9 games of cumulative training, the encoder followed its gradient to whatever minimizes its local losses. Ouro's usefulness as a downstream refiner was never in the loss.

Put another way: the NextFrame/Spatial/Color heads don't care what Ouro thinks. If the encoder finds a representation where NextFrame loss drops but Ouro can't reason over it, that's still a gradient descent step on the encoder's explicit objectives. Nothing was anchoring the encoder to Ouro-compatible space. The anchoring should have existed — the CLT paper's 95.2% result depends on it — but it was never built in for this training regime.

### What the loop-state signature test would confirm (OOM'd, needs fix)

The loop-signature test didn't run because of VRAM limits on 12GB. Fix shipped in this session (chunked forward pass, default 2 frames per call). Once it runs, expected results given cos ≈ 0:

- **Prediction A (most likely):** Reference encoder shows clean monotone decay L1↔L2 > L2↔L3 > L3↔L4. Current encoder shows chaotic pattern — non-monotone, high variance, possibly negative cosines. This would confirm Ouro is effectively flailing on the current encoder's output.
- **Prediction B (surprising if true):** Both encoders show structured refinement. This would mean Ouro can reason over almost any input topology, which would be a remarkable property worth investigating on its own — but would also mean the Δloop=21 vs Δloop=6 gap is driven by something other than drift. Seems unlikely given the cosine-zero result.

The loop-signature test at this point is more about *confirming* the picture than *determining* it. The cosine=0 result is already decisive.

### Implications for prior work

**Sprint 4 results need to be re-understood.** The overnight data showed:
- 8-head affordance heads training: `ObjL=0.22–0.35` per update
- Hazard beliefs forming with high confidence from DEATH events
- Beliefs persisting across runs via ColorPriorTable
- ft09 spatial targeting learning (11.9% → 34.4% click change rate)

These results are NOT invalidated. The scene parser, ObjectTable, affordance heads, and ColorPriorTable all operate on segmented integer grids and pooled CNN features — they don't go through Ouro loop states. The beliefs are real, the parsing is real, the CNN-level affordance learning is real.

**What IS invalidated (or at least, unknown):** any result that flows through Ouro loop states. That includes:
- The ranker's "loop-delta gate" (Δloop as a trust signal) — currently reading garbage
- The LoopStatePooler's pooled representation feeding into ranker
- Any planning decision that consulted Ouro's refined CLS token
- The `_loop_delta_ema` self-calibrating gate (§5)

The ft09/r11l/su15 ranker-ignores-beliefs diagnostic I was so excited about yesterday may not actually be a Sprint 6 problem. The ranker doesn't ignore beliefs; the ranker may not be getting *any* informative signal from Ouro, because Ouro is receiving orthogonal-to-pretraining input.

### Action items

**Immediate (before any more training):**

1. **Revert the encoder to v17b state.** Load any v17b checkpoint as the encoder source:
   ```bash
   # Simplest path: copy the encoder sub-state from v17b into a new checkpoint
   python - << 'PY'
   import torch, shutil
   v17b = torch.load('checkpoints_v17b/arc_vc33_run3.pt', map_location='cpu', weights_only=False)
   current = torch.load('checkpoints_running/sprint4_overnight.pt', map_location='cpu', weights_only=False)
   current['encoder'] = v17b['encoder']  # keep everything else from Sprint 4
   torch.save(current, 'checkpoints_running/sprint4_encoder_reverted.pt')
   print('Saved sprint4_encoder_reverted.pt with v17b encoder + current everything else')
   PY
   ```

2. **Freeze the encoder** for all Sprint 5 and Sprint 6 work. The NextFramePredictor / SpatialClickPredictor / patch_color_head training paths need to either be disabled or have their encoder-gradient blocked. Simplest: set `requires_grad = False` on all encoder parameters after load, skip `encoder_optimizer.step()`.

3. **Re-run the loop-signature diagnostic with chunked forward pass** using the reverted checkpoint. Confirm Ouro refinement pattern is monotone on the reverted encoder. This is the sanity check that reversion did what we think it did.

**Medium-term (for retraining the encoder properly):**

4. Add a **KL-anchor loss** to the encoder training. Penalize `KL(encoder(frame_t) || ouro_embedding_distribution)` using a small coefficient. The exact form needs thinking — probably match token output statistics (mean/std across patches) to Ouro's input embedding statistics. This prevents the gradient from drifting the encoder into spaces Ouro can't read.

5. **Alternative:** freeze the encoder permanently at v17b state and accept that it's a fixed retina. The encoder is 1.85M params — it was never the main learning object. Scene parsing + affordance heads + ranker + self-model are where learning happens. Making the encoder a frozen pretrained component (like Ouro) is architecturally clean and removes an entire category of drift risk.

**Long-term (Sprint 6 scoping):**

6. The ranker-ignores-beliefs diagnostic from §24 is now ambiguous. It could still be Sprint 6 work. Or it could be "Ouro was feeding garbage into ranker, ranker correctly ignored garbage." Sprint 6 should start from **the reverted encoder** so that any Sprint 6 intervention operates on a known-healthy Ouro signal. Once the encoder is reverted and the ranker is producing real Ouro-informed signal, if the ranker STILL ignores beliefs → Sprint 6 is real. If reversion alone restores ranker→belief conditioning → Sprint 6 was chasing a symptom.

### Philosophical note on what happened

This is a genuine lesson in training-loss incompleteness. The encoder had four gradient sources: NextFramePredictor, SpatialClickPredictor, patch_color_head, and (implicitly through ranker) Ouro. Only the first three were *direct* — the fourth path runs through frozen Ouro, which has no gradient available to pull the encoder toward its pretraining distribution. So the encoder optimized hard on NextFrame/Spatial/Color and slowly walked away from Ouro-space.

This is the exact shape of problem a constitution-anchored training regime is designed to prevent. In the basal ganglia paper framing (§10), the CLT evaluator would provide a third gradient source saying "loop states should encode preference relationally" — a signal that IS connected to Ouro-space. Without that anchor, the encoder had no reason to stay in Ouro-compatible territory. Sprint 11's architecture includes this anchor by design.

For now: freeze the encoder, proceed with Sprint 5/6, revisit encoder retraining once there's an anchor loss that ties it back to Ouro.

---

## 28. Encoder drift diagnostic — `encoder_drift_check.py` script reference

Reference material for the drift-check script itself. The actual empirical result from running it is in §27 above.

Purpose: answer whether Δloop=21 reflects healthy richer-reasoning or pathological distributional drift.

### Key insight about what the CLT evaluator does NOT measure

The published CLT evaluator achieves 95.2% pairwise accuracy on HH-RLHF by reading Ouro's loop states on **tokenized text input**. The path is:

```
HH-RLHF text → tokenizer → Ouro (frozen) → 4 loop states (hooked) → PairwiseEvaluator → score
```

The GridEncoder is never involved. If you run `evaluate_pairwise.py` on any Sprint 4 checkpoint, you'll get ≈95.2% regardless of how much GridEncoder has drifted, because the text path doesn't touch GridEncoder. Ouro is frozen; tokenization hasn't changed; the evaluator has the same weights it had when published. That accuracy is invariant by construction.

**Implication:** the CLT evaluator's accuracy on text is a good sanity check that nothing catastrophic happened to Ouro itself, but it is NOT a drift measurement for the GridEncoder. The diagnostic must be constructed differently.

### The two diagnostics the script actually runs

**(1) Encoder token distance.** Feed a seeded batch of 64×64 random grids through both the current encoder and a reference (pre-drift) encoder. Report L2 and cosine distance between their token outputs. Fast, no Ouro load required. Directly quantifies how much the encoder weights have moved.

**(2) Loop-state signature.** Take the same seeded batch. For each encoder checkpoint: encode → feed into Ouro via `inputs_embeds` → capture all 4 loop states via forward hook on `model.model` → compute the cosine similarity matrix between consecutive loop states (L1↔L2, L2↔L3, L3↔L4). Compare the current-encoder signature against the reference-encoder signature.

The second is the real drift test. Benign drift (encoder producing richer tokens that Ouro still understands how to refine) gives cosines that drop but remain monotone and low-variance. Pathological drift (encoder off-distribution) gives cosines that become non-monotone, high-variance, sometimes negative. The script's `diagnose_loop_signature` classifies automatically.

### Usage

```bash
# Fast diagnostic: cosine distance only, ~30 seconds
python encoder_drift_check.py \
    --current_checkpoint checkpoints_running/sprint4_overnight.pt \
    --reference_checkpoint checkpoints_v17b/arc_vc33_run3.pt

# Full diagnostic: includes Ouro-based loop-state signature, ~3 minutes
python encoder_drift_check.py \
    --current_checkpoint checkpoints_running/sprint4_overnight.pt \
    --reference_checkpoint checkpoints_v17b/arc_vc33_run3.pt \
    --run_loop_signature
```

### Interpretation grid (embedded in script docstring)

| Signal | Value | Verdict |
|---|---|---|
| Encoder token cosine distance | < 0.02 | Identical — Δloop=21 has a non-encoder explanation |
| | 0.02 – 0.08 | Moderate — consistent with normal learning |
| | 0.08 – 0.20 | Substantial — correlate with loop-state signature |
| | > 0.20 | Large — almost certainly the drift explanation |
| Loop signature | Both monotone L1L2 ≥ L2L3 ≥ L3L4 | Benign. Continue Sprint 4/5/6 work. |
| | Current non-monotone, reference monotone | Drift confirmed. Freeze or revert encoder. |

### Action paths post-diagnostic

If drift confirmed, three mitigation options (in order of least-invasive to most):

1. **Lower encoder learning rate** by 5–10× relative to other modules. Softest fix. Works only for slow drift; won't repair existing damage.
2. **Freeze encoder during Sprint 5/6 work.** Detach `encoder_optimizer`, accept that encoder stops learning until Sprint-level decisions about retraining are made. Clean, reversible.
3. **Revert encoder to pre-drift state.** Use `--weights_only` checkpoint load with v17b encoder sub-state; training resumes from reference representation. Most invasive but most complete fix.

All three are compatible with the existing checkpoint infrastructure (§20).

<!-- ENGRAMME_LMM_CONTEXT_START -->

## Engramme / Large Memory Models context — implementation note

Public context, April 2026:
Engramme is presenting "Large Memory Models" as a memory layer for human digital-life / memorome retrieval. The useful public idea is not simply a larger context window, but memory that is lifelong, proactive, and associative.

Treat this as architectural inspiration only. Public technical detail is sparse, so do not assume a specific algorithmic breakthrough.

### Relevance to Hunter Seeker / Ouro ARC

The useful idea is not "store more logs." The useful idea is proactive associative recall.

Existing Hunter Seeker systems already contain proto-engram material:

- event log and object-table histories;
- terminal outcome memory and terminal prototypes;
- topology death diagnostics;
- solved/trusted trajectory prefixes;
- phase/action templates;
- object tracks and color/type beliefs;
- loop-state deltas and Ouro recurrence signatures;
- self-model / affective context token machinery.

### Implementation direction

Build a small engram layer above the existing memories rather than replacing them.

Minimal abstraction:

- `EngramRecord`: compact state/action/outcome memory record.
- `cue_vector`: current state/action/topology/loop signature.
- `outcome_type`: progress, no_change, terminal, mechanism, topology, self_model.
- `risk`: terminal/hazard scalar.
- `reward`: progress/solved scalar.
- `support_count` and `counterevidence_count`.

Decision-time recall should retrieve top-k relevant records using:

1. terminal/context-key similarity;
2. object/topology similarity;
3. loop-state delta similarity;
4. action/click compatibility;
5. recent phase/template compatibility.

Retrieved engrams should feed action arbitration as explicit score components:

- `engram_terminal_penalty`
- `engram_hazard_penalty`
- `engram_progress_bonus`
- `engram_probe_bonus`
- `engram_conflict_flag`

Important constraint:
Engram memory must not become broad action blacklisting. It should be contextual and counterevidence-aware.

### Immediate connection to current debug work

The chosen-action debug probe showed a likely hazard-arbitration hole: `reachable_hazard_delta` can be nonzero while `safety_penalty` remains `0.0`.

If death-probe diagnostics confirm the fatal chosen action has `reachable_hazard_delta > 0` and `safety_penalty == 0`, the next patch should add hazard-aware safety scoring, then later generalize it into the engram recall layer.

Minimal first implementation:

- `collect_engram_record(step_info, score_components, outcome)`
- `retrieve_engram_biases(current_candidate_context)`
- `apply_engram_biases_to_score_components()`
- `summarize_engram_memory()`

The first backend can reuse existing `terminal_outcome_memory`, prototypes, topology-death diagnostics, phase templates, object tracks, and loop-state signatures. A learned retriever can come later.

<!-- TOPOLOGY_ENGRAMS_NOTE_START -->

### Topology engrams — why this matters specifically for ARC/Hunter Seeker

The current ls20 death probes suggest a failure mode that terminal memory alone does not solve:

- terminal memory can fire globally;
- but the selected fatal candidate can still receive terminal_outcome_penalty = 0.0;
- repeated deaths occur in similar local adjacency/frontier situations.

This means the agent is not merely forgetting that one exact action/context was bad. It is failing to recognize a local spatial danger basin across slightly different primitive actions, tracks, or context keys.

Terminal memory says:

    action X in this exact/prototype transition context killed me

A topology engram should say:

    this local adjacency/frontier configuration near this object class killed me,
    even when the exact action, track id, or coordinate differs

This is especially relevant for ARC-style environments because danger often aliases through structure:

- same color but different object tracks;
- same local object relation but different absolute coordinates;
- same trap geometry but different primitive action;
- same frontier/adjacency basin but different step;
- terminal outcome caused by spatial relation, not by action identity alone.

Minimal topology engram record:

- source: adjacent | topology_frontier | fallback_interacted
- outcome_type: mechanism | topology | terminal | progress
- last_action
- avatar descriptor
- adjacent_colors
- frontier object descriptors
- candidate object descriptors:
  - color
  - area_bin
  - adjacency_count
  - relative centroid to avatar
  - belief_avatar / belief_exit / belief_hazard / belief_collectible
  - track protection flags
- symbolic_summary:
  - reachable_delta
  - reachable_hazard_delta
  - reachable_reward_delta
  - moved_track_count
  - frontier_delta
  - exit_path_delta
- loop_signature / loop_delta summary
- valence: negative for death/failure, positive for progress
- support_count
- counterevidence_count

Decision-time use:

For each candidate action, retrieve similar topology engrams and compute explicit score components:

- engram_topology_penalty
- engram_hazard_penalty
- engram_progress_bonus
- engram_conflict_flag

Design constraint:
Topology engrams must not become color-global or action-global blacklists. They should generalize only through local spatial/object structure.

Bad behavior to avoid:

    never press action 1
    never approach color 3

Desired behavior:

    avoid this kind of local adjacency/frontier basin unless there is strong counterevidence

Connection to current debug thread:
The same-color candidate diagnostic should not be blindly flipped into a rejection rule, because it was intentionally made track-local to avoid color-global false protection. The better fix is a topology-engram layer that can remember local danger basins while still allowing same-color objects to diverge by track and evidence.

Immediate implementation path:

1. Keep existing terminal memory as exact/prototype action-context memory.
2. Add instrumentation for cross-action terminal prototype similarity.
3. If cross-action similarity is high, add conservative cross-direction terminal fallback.
4. If cross-action similarity is weak, add topology-local engram recall using topology_death diagnostics and chosen_action_debug score components.
5. Feed topology engram recall into score components as:
   - engram_topology_penalty
   - engram_hazard_penalty
   - engram_progress_bonus
   - engram_conflict_flag

This gives Hunter Seeker proactive associative recall over spatial failure modes, which is the useful part of the Engramme/LMM idea for ARC.

<!-- TOPOLOGY_ENGRAMS_NOTE_END -->

<!-- ENGRAMME_LMM_CONTEXT_END -->

