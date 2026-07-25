<!-- Source: PROJECT_STATE_HUNTER_SEEKER.md lines 1022-1750 before the 2026-05-14 split. -->
<!-- Source chunk SHA256: d575b3620f151d4d9e5e3df93fcfa5df92e30046110aa1b6662bc40903e3bca8 -->

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

