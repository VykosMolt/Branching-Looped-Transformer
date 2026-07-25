<!-- Imported from `claude_sandbox/post_quick_ladder_cleanup_plan.md` during Hunter-Seeker state consolidation on 2026-05-14. -->
<!-- Original SHA256: b6b65fa69ce6cfe886bde1dbbc39f0a94869130258eb21621a433c209db92b28; original line count: 706. -->

# Post-Quick-Ladder Cleanup Plan: Deferred Items 2, 3, and 5

This file captures the plan for the backlog groups the user asked about:

```text
2. "Defer until after ladder"
3. "Micro-optimizations and cleanup ideas"
5. "Keep for now, redesign later"
```

Important framing:

```text
Do not do everything at once.
Do not mix architecture refactors with micro-optimizations.
Do not touch sleep, Ouro-trust redesign, or broad renames before the current measurement pipeline is stable.
```

The recommended order is:

```text
A. Do item group 5 first:
   Replace temporary safety guards with clean permanent mechanisms.

B. Do selected parts of item group 2:
   Improve experiment hygiene and measurement reliability.

C. Do item group 3 last:
   Pure QoL/performance cleanup only.
```

---

## A. Group 5 — "Keep for now, redesign later"

These are the patches that currently unblock the ladder, but should eventually become cleaner mechanisms.

### A1. Pending self-model graph guard

Current state:

```python
_pending_event_pred = getattr(self, "_pending_event_pred", None)
if (
    sm_grad_active
    and _pending_event_pred is not None
    and getattr(_pending_event_pred, "requires_grad", False)
):
    sm_grad_active = False
```

This guard is acceptable for the quick ladder because it prevents `_train_ranker()` from stepping `self_model_optimizer` while a delayed pending event-prediction graph is still live.

But it is not the ideal permanent design.

### Why the current guard exists

The self-model path stores a delayed event-prediction graph across steps:

```text
step t:
  create _pending_event_pred graph

between t and t+1:
  _train_ranker() may step self_model_optimizer

step t+1:
  _self_model_step() tries to backward through the pending graph
  but the self-model parameters have already been mutated
  → PyTorch version-counter / inplace-modification error
```

The guard avoids the crash by preventing the ranker path from mutating self-model parameters when a pending graph is live.

### Permanent solution

The permanent fix should be ordering, not suppression.

Recommended design:

```python
def step(self, obs):
    self._self_model_consume_pending_loss_if_any()
    result = super().step(obs)
    self._self_model_advance_and_create_next_prediction(...)
    return result
```

Detailed requirements:

```text
1. At the beginning of HunterSeekerAgent.step(), before super().step(obs),
   consume any pending self-model event loss.

2. Backward/step self_model_optimizer there.

3. Clear _pending_event_pred immediately after consumption.

4. Only then call super().step(obs), which may train the ranker.

5. After super().step(obs), advance the self-model state and create the next
   pending event prediction for the following transition.

6. Once this works, the guard in _train_ranker can remain only as a defensive
   fallback/assert/log, not as the normal control path.
```

### Suggested Claude instruction

```text
Refactor self-model update ordering so no live pending self-model event graph
survives across any optimizer step that can mutate self-model parameters.

Implement:
1. At the beginning of HunterSeekerAgent.step(), before super().step(obs),
   consume any pending self-model event loss.
2. Backward/step self_model_optimizer there.
3. Clear _pending_event_pred immediately after consumption.
4. Only then call super().step(obs), which may train the ranker.
5. After super().step(obs), advance/create the next pending event prediction
   for the following transition.

Once this is done, remove or keep the _pending_event_pred guard in _train_ranker
only as a defensive assert/log, not as the normal control path.
```

---

## A2. Anchor unfreeze flag

Current state:

```text
--unfreeze_encoder_after_partial_load
```

This flag is being reused to make anchor training unfreeze the encoder even when no partial-load fallback happened.

Behaviorally, this is correct for the ladder:

```text
anchor training needs trainable encoder parameters
```

But the flag name is wrong. It describes an implementation detail, not the actual intent.

### Permanent solution

Add a new explicit flag:

```text
--unfreeze_encoder_for_anchor
```

or:

```text
--train_encoder_for_anchor
```

Recommended behavior:

```text
If anchor_train_every > 0 and --unfreeze_encoder_for_anchor is set:
  - set agent.freeze_encoder = False
  - set encoder parameters requires_grad=True
  - print a clear line saying the encoder is unfrozen for anchor training

If anchor_train_every > 0 and encoder is frozen and the new flag is not set:
  - fail loudly or warn loudly
  - do not pretend anchor training is active

Keep --unfreeze_encoder_after_partial_load as a backward-compatible alias for now.
```

### Runner update

Update `run_ablation_ladder.sh` so anchor steps use:

```bash
--unfreeze_encoder_for_anchor
```

instead of the historical:

```bash
--unfreeze_encoder_after_partial_load
```

while preserving the old flag as an alias inside `train_arc_codex.py`.

### Suggested Claude instruction

```text
Add a new explicit CLI flag:
  --unfreeze_encoder_for_anchor

Behavior:
- If anchor_train_every > 0 and --unfreeze_encoder_for_anchor is set:
    unfreeze encoder and set encoder params requires_grad=True.
- If anchor_train_every > 0 and encoder is frozen and the new flag is not set:
    fail loudly or warn loudly, but do not silently pretend anchor training is active.
- Keep --unfreeze_encoder_after_partial_load as a backwards-compatible alias for now.
- Update run_ablation_ladder.sh to use --unfreeze_encoder_for_anchor instead of the old flag.
```

---

## B. Group 2 — "Defer until after ladder"

Not all of group 2 should be done immediately. Some are true architecture sprints, not QoL.

Recommended split:

```text
Do soon:
  - RUN_ID timestamped ladder dirs
  - comparator command printout
  - score_components aggregation only if summaries already emit the fields

Defer:
  - progress/outcome adapter
  - color terminology rename
  - removing all legacy ARC defaults
  - Ouro confidence as primary trust axis
  - sleep/consolidation
```

---

## B1. RUN_ID timestamped ladder directories

Current problem:

The runner uses fixed paths:

```text
claude_sandbox/ablation_runs/step_*
claude_sandbox/ablation_event_dumps/step_*
claude_sandbox/checkpoints_running/ladder_step_*.pt
```

This is okay for one clean run, but repeated runs can mix old and new outputs or reuse stale checkpoints.

### Permanent solution

Add:

```bash
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
```

Then nest outputs under:

```text
claude_sandbox/ablation_runs/$RUN_ID/
claude_sandbox/ablation_event_dumps/$RUN_ID/
claude_sandbox/checkpoints_running/$RUN_ID/
```

Print the run ID at start:

```bash
echo "RUN_ID: $RUN_ID"
```

At the end, print the exact comparator command for that run:

```bash
python -m claude_sandbox.compare_ladder_summaries \
  claude_sandbox/ablation_event_dumps/$RUN_ID/step_* \
  --alarms \
  --json-out /tmp/ladder_${RUN_ID}_summary.json
```

### Suggested Claude instruction

```text
Add RUN_ID timestamped ladder directories.

Requirements:
- RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
- OUTPUT_BASE=claude_sandbox/ablation_runs/$RUN_ID
- EVENT_DUMPS_BASE=claude_sandbox/ablation_event_dumps/$RUN_ID
- CHECKPOINT_BASE=claude_sandbox/checkpoints_running/$RUN_ID
- Print RUN_ID at script start.
- Print exact comparator command at script end.
- Preserve existing --quick and --vram-check behavior.
```

---

## B2. Comparator score-components aggregation

The runner currently mentions comparing:

```text
effective_confidence_pre_ouro
effective_confidence
ranker_score_raw
heuristic_score_raw
```

But these fields may live only in per-step traces, not in `measurement_summary`.

### Rule

Do not add comparator rows for fields that the harness does not emit.

### If measurement_summary already emits these fields

Add aggregation rows for:

```text
effective_confidence_pre_ouro_mean
effective_confidence_mean
ouro_confidence_multiplier_mean
ranker_score_raw_mean
heuristic_score_raw_mean
```

### If measurement_summary does not emit these fields

First update the harness to summarize them into `measurement_summary`.

Potential summary block:

```json
"score_components": {
  "effective_confidence_pre_ouro_mean": 0.0,
  "effective_confidence_mean": 0.0,
  "ouro_confidence_multiplier_mean": 0.0,
  "ranker_score_raw_mean": 0.0,
  "heuristic_score_raw_mean": 0.0
}
```

Then update the comparator.

### Suggested Claude instruction

```text
Add score_components aggregation only if measurement_summary already emits those fields.
If the fields are not present, do not invent comparator rows. Stop and report which harness fields need to be emitted first.
```

---

## B3. Move ARC progress semantics out of base agent

This is a real architecture leak, but it should be a separate session because it changes scoring semantics.

Current leak:

```text
levels_completed
_QUALITY_LEVEL_SCALE
```

These concepts are ARC-shaped and live too close to the core.

### Permanent direction

Add an outcome/progress abstraction.

Option A: extend `ObservationAdapter`.

```python
class ObservationAdapter(Protocol):
    ...
    def progress_value(self, obs) -> float: ...
    def terminal_state(self, obs) -> bool: ...
    def transition_quality(self, prev_obs, obs, frame_changed: bool, info: dict) -> float: ...
```

Option B: separate `OutcomeAdapter`.

```python
class OutcomeAdapter(Protocol):
    def progress_value(self, obs) -> float: ...
    def terminal_state(self, obs) -> bool: ...
    def transition_quality(self, prev_obs, obs, frame_changed: bool, info: dict) -> float: ...
```

ARC implementation can use:

```text
levels_completed
level completion bonuses
frame-change bonuses
```

But the base agent should only consume:

```text
transition_quality
progress_value
terminal_state
```

### Suggested Claude instruction

```text
Do not do this in the same session as QoL runner fixes.

In a separate session, introduce an OutcomeAdapter or extend ObservationAdapter
with progress_value(), terminal_state(), and transition_quality().

Move levels_completed / _QUALITY_LEVEL_SCALE usage out of the base agent and
into the ARC-specific adapter/harness layer.

Preserve current ARC behavior exactly.
```

---

## B4. Rename `ColorPriorTable` / `color` terminology

This is conceptual hygiene, not a ladder blocker.

Long-term rename:

```text
color → label / object_label / segment_id
ColorPriorTable → LabelPriorTable / EntityPriorTable
color_beliefs → label_beliefs
```

Reason:

```text
For ARC, "color" works.
For the domain-general architecture, color should not be a privileged concept.
The adapter emits integer labels; the core should reason over labels/entities.
```

Do this only after the ladder and after other functional changes are stable, because it is a wide rename with little immediate value.

---

## B5. Remove legacy ARC defaults from generic modules

Examples:

```text
ActionHead(n_actions=8, click_action_idx=6)
normalize_clicks(grid_w=64, grid_h=64)
```

These defaults are okay as backward-compatible legacy defaults, but new domain-general paths should pass explicit values.

Long-term cleanup:

```text
- make n_actions required where practical
- make click_action_idx adapter-owned
- make grid_w/grid_h required in normalize_clicks
- or clearly mark defaults as legacy compatibility only
```

Do not do this before measurement unless it becomes a real bug.

---

## B6. Ouro confidence as primary trust axis

Do not change this now.

Current state:

```text
Ouro confidence / exit confidence is captured and blended.
It is not yet the primary trust axis.
```

This is a research change, not cleanup.

Wait for ladder results before deciding whether:

```text
loop_delta remains primary
Ouro confidence becomes primary
Ouro confidence is only a multiplier
```

---

## B7. Sleep / consolidation

Definitely not now.

Suggested staging remains:

```text
Sprint 7  = synthetic mechanic curriculum
Sprint 9a = abstract memory scaffold / prototype API
Sprint 10 = loop-state-centric / object-file-delta ranker
Sprint 8  = grounded offline replay / consolidation
Sprint 12 = imagined rollouts + backward reconstruction
Sprint 13 = abstraction / prototype formation / self-distillation
```

Rules:

```text
Sprint 8 may only train from real recorded transitions/events.
Sprint 12 may imagine/reconstruct but must validate/filter.
Sprint 13 may compress into prototypes only after replay/reconstruction is stable.
```

---

## C. Group 3 — Micro-optimizations and cleanup ideas

These are QoL/performance cleanups. They should not be mixed with architecture changes.

Recommended batch:

```text
1. Vectorize pad_grids_to_batch.
2. Cache ArcActionAdapter enum map.
3. Add stable hash helper only if persistent/cross-process hashes are needed.
4. Add optional saliency cache by frame hash.
5. Leave SceneParser mask memory alone unless larger domains become real.
```

---

## C1. Vectorize `pad_grids_to_batch`

If this still uses Python row/column loops, replace with tensor slicing:

```python
arr = torch.as_tensor(g, dtype=torch.long)
batch[i, :h, :w] = arr
```

This is safe and simple.

### Suggested Claude instruction

```text
Vectorize pad_grids_to_batch without changing behavior.
Keep pad_value required.
Add/keep tests for:
- different H/W grids
- non-multiple-of-patch-size padding
- pad_value preservation
- dtype long
```

---

## C2. Cache `ArcActionAdapter` enum map

If `decode()` rebuilds the enum map every call:

```python
{a.value: a for a in env_action_space}
```

cache it by enum class identity or by `id(env_action_space)` if stable.

This is tiny and not urgent.

### Suggested Claude instruction

```text
Cache the GameAction enum value map in ArcActionAdapter.decode().
Do not change behavior.
Keep fallback behavior if env_action_space is unusual.
```

---

## C3. Stable frame hash helper

Current `hash(frame.tobytes())` is fine for in-process caches, but Python hash is process-salted.

Do not replace everything unless needed.

If future code needs persistent or cross-process cache keys, add:

```python
import hashlib

def stable_frame_hash(frame: np.ndarray) -> str:
    return hashlib.blake2b(frame.tobytes(), digest_size=16).hexdigest()
```

Use only where persistence/cross-process comparison matters.

---

## C4. Optional saliency cache

`visual_saliency(frame)` is cheap at ARC scale.

If candidate generation repeatedly recomputes saliency for the same frame, cache by frame hash.

Do not add complex cache invalidation.

---

## C5. SceneParser cache memory

Current SceneParser stores object masks and caches parsed scenes.

This is fine for ARC.

For larger domains, consider:

```text
- bounding-box-local masks
- one global label map plus object metadata
- configurable cache size based on spatial resolution
```

Do not change before a larger domain requires it.

---

## Recommended Claude execution plan

### Pass 1 — Replace temporary mechanisms with permanent ones

Do:

```text
- ordered self-model pending-loss consumption
- explicit --unfreeze_encoder_for_anchor flag
- update ladder runner to use new flag
- keep backward compatibility
```

Do not do:

```text
- outcome/progress adapter
- color rename
- sleep
- Ouro trust redesign
- micro-opts
```

Validation:

```bash
python3 -m py_compile claude_sandbox/*.py
./claude_sandbox/run_ablation_ladder.sh --quick
python -m claude_sandbox.compare_ladder_summaries   claude_sandbox/ablation_event_dumps/step_*   --alarms   --json-out /tmp/ladder_quick_summary.json
```

### Pass 2 — Experiment hygiene

Do:

```text
- RUN_ID timestamped ladder dirs
- print exact comparator command
- add score_components aggregation only if already emitted
```

Do not do:

```text
- core architecture changes
```

Validation:

```bash
./claude_sandbox/run_ablation_ladder.sh --quick
find claude_sandbox/ablation_event_dumps -name 'measurement_run_*.json' | head
```

### Pass 3 — Micro-optimizations

Do:

```text
- vectorize pad_grids_to_batch
- cache ArcActionAdapter enum map if trivial
```

Maybe do later:

```text
- stable frame hash helper
- saliency cache
```

Do not do:

```text
- SceneParser memory redesign unless larger domains require it
```

---

## One-line summary

```text
Do group 5 first, selected group 2 second, group 3 last.
Keep true architecture changes isolated from measurement/QoL changes.
```
