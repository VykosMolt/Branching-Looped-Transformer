<!-- Imported from `claude_sandbox/post_quick_ladder_cleanup_plan_v4.md` during Hunter-Seeker state consolidation on 2026-05-14. -->
<!-- Original SHA256: 6332625344aec3167affe991984bca65f8e089f19148d270164dece8d47dfee9; original line count: 1337. -->

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


---

## Anchor VRAM bottleneck after cuDNN fix

Discovered during the next `--quick` run after the frozen-evaluator/cuDNN anchor fix.

### Symptom

Anchor training no longer fails with:

```text
cudnn RNN backward can only be called in training mode
```

Instead, anchor training reaches the real training path and OOMs:

```text
[Anchor-Train] OOM with batch_size=4 ... halving adaptive batch 4 → 2
[Anchor-Train] OOM with batch_size=2 ... halving adaptive batch 2 → 1
```

This means the cuDNN/eval-mode GRU issue is fixed and the anchor is now actually attempting to train.

### Interpretation

This is progress, but it exposes the next bottleneck:

```text
anchor batch 4/2 is too large for the current 11.5–12 GB GPU memory budget
```

The log showed very little free VRAM remaining:

```text
~91 MB free before batch_size=4 OOM
~77 MB free before batch_size=2 OOM
```

So the immediate issue is memory budget, not anchor correctness.

### Immediate runner fix

Make anchor batch size configurable from the runner.

Add near the other defaults in `run_ablation_ladder.sh`:

```bash
ANCHOR_BATCH_SIZE="${ANCHOR_BATCH_SIZE:-4}"
```

For quick mode, force a small default unless the user explicitly overrides it:

```bash
if [[ "$QUICK" == "1" ]]; then
    GAMES="ls20"
    MAX_STEPS=80
    N_RUNS=1
    ANCHOR_EVERY=10
    ANCHOR_BATCH_SIZE="${ANCHOR_BATCH_SIZE:-1}"
fi
```

Then replace hardcoded anchor calls like:

```bash
--anchor_batch_size 4
```

with:

```bash
--anchor_batch_size "$ANCHOR_BATCH_SIZE"
```

in all anchor-enabled steps:

```text
VRAM check
Step 3.5
Step 6
Step 7
```

### Important environment fix

Set the CUDA allocator config in the bash runner before Python starts:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Reason:

```text
Several Python files set PYTORCH_CUDA_ALLOC_CONF internally, but if torch has already been imported,
that can be too late. The runner is the correct place to set allocator behavior before Python startup.
```

### Anchor OOM retry cleanup

When anchor training catches OOM and retries with a smaller adaptive batch, it should clear memory before retrying.

Patch shape:

```python
import gc

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

Use this inside the anchor OOM handling path before retrying or before returning to the main training loop.

Reason:

```text
The log showed free VRAM dropping after each failed attempt.
Even if PyTorch normally releases failed allocations, explicit cleanup makes adaptive retry behavior safer.
```

### Verification

Let quick mode continue and check whether adaptive batch size 1 succeeds.

Success criteria:

```text
- no cuDNN RNN backward error
- no frozen-encoder anchor warning
- anchor.attempts > 0
- anchor.successes > 0
- current_batch_size appears in measurement_summary.anchor
- if OOM occurs at 4/2, batch 1 eventually succeeds
```

If batch size 1 succeeds:

```text
Use ANCHOR_BATCH_SIZE=1 for quick and probably for --vram-check on this GPU.
Proceed to --vram-check before the full ladder.
```

If batch size 1 also OOMs:

```text
Full anchor training cannot run on this 11.5–12 GB card without deeper memory mitigation.
Do not run the full ladder until anchor.successes > 0 has been observed.
```

### Possible deeper memory mitigations if batch 1 OOMs

Only consider these if batch size 1 fails:

```text
1. Use torch.cuda.empty_cache() + gc.collect() before every anchor attempt.
2. Run anchor less frequently.
3. Reduce anchor trajectory/token retention.
4. Use mixed precision for anchor forward if numerically safe.
5. Move frozen evaluator to CPU for scoring/backward is likely too slow and may not solve gradient-to-input needs cleanly.
6. Split anchor scoring into microbatches with gradient accumulation if batch > 1 is needed later.
7. Disable nonessential training heads during anchor step to reduce live graph memory.
```

### Classification

Status: pre-full-ladder blocker until verified.

Reason:

```text
The quick ladder can structurally finish even with anchor OOMs, but anchor-vs-no-anchor comparisons are not trustworthy unless anchor.successes > 0.
```

Action:

```text
Patch runner batch configurability + allocator export + OOM cleanup.
Rerun --quick.
Then run comparator and inspect anchor fields.
Only proceed to --vram-check/full ladder after anchor succeeds at least once.
```


---

## Anchor batch-size result and higher-batch options

Update after the post-cuDNN quick run:

```text
anchor batch 4 → OOM
anchor batch 2 → OOM
anchor batch 1 → succeeded
```

Interpretation:

```text
The anchor path is now real:
- frozen evaluator/cuDNN issue is fixed
- encoder unfreeze flag is wired correctly
- anchor attempts reach the actual training path
- batch size 1 can succeed on the current ~11.5–12 GB GPU
```

For this GPU, use:

```bash
ANCHOR_BATCH_SIZE=1 ./claude_sandbox/run_ablation_ladder.sh --quick
ANCHOR_BATCH_SIZE=1 ./claude_sandbox/run_ablation_ladder.sh --vram-check
```

and likely for the full ladder unless the VRAM check shows more headroom.

### Why batch size is so expensive

Anchor training is not just a small evaluator pass.

It requires:

```text
encoder/grid tokens
→ frozen Ouro forward with gradients through inputs
→ loop-state trajectory extraction
→ frozen CLT/GRU evaluator forward
→ backward through evaluator + Ouro path into encoder/projector/self-model inputs
```

Even though Ouro and the evaluator weights are frozen, gradients still need to flow through their forward computations to reach the trainable input-side modules. That means activations must be kept, so memory use is much closer to training than inference.

### What batch size 1 means

Batch size 1 is acceptable for correctness and smoke testing.

It verifies:

```text
anchor.successes > 0
loss_ema exists
current_batch_size is meaningful
encoder gradients can flow
anchor-vs-no-anchor ladder steps are not fake/no-op
```

But it is noisy. If anchor is important, the long-term goal should be either:

```text
- higher real batch size
- or higher effective batch size via microbatch gradient accumulation
```

### Best way to get a higher effective batch

Preferred: microbatch gradient accumulation.

Instead of trying to fit batch 4 at once:

```text
microbatch 1
microbatch 1
microbatch 1
microbatch 1
accumulate gradients
optimizer.step()
```

This gives an effective batch of 4 with roughly batch-1 peak VRAM.

Patch direction:

```text
--anchor_batch_size controls effective batch size
--anchor_microbatch_size controls peak memory
```

Example desired behavior:

```text
anchor_batch_size=4
anchor_microbatch_size=1
→ process 4 pairs as four separate forward/backward passes
→ divide loss by 4 or otherwise average gradients
→ single optimizer.step()
```

This is the most principled way to improve anchor stability without needing more VRAM.

### Other ways to try a real higher batch

These may help, but are secondary to microbatching.

#### 1. Clear memory before anchor attempts

Before anchor training:

```python
import gc
gc.collect()
torch.cuda.empty_cache()
```

Also clear memory after OOM before retrying.

This reduces fragmentation and prevents failed OOM attempts from poisoning the next attempt.

#### 2. Set allocator config before Python starts

In `run_ablation_ladder.sh`:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

This must happen before Python imports torch. Setting it inside Python files can be too late.

#### 3. Run anchor at lower frequency

This does not increase batch size, but reduces total memory-pressure events and runtime:

```bash
ANCHOR_EVERY=50
ANCHOR_EVERY=100
```

Useful for full ladder stability if anchor batch 1 is slow but works.

#### 4. Reduce competing live graphs before anchor

Anchor should run when unrelated training graphs have been consumed/cleared.

Guideline:

```text
do ranker/nextframe/self-model optimizer steps
clear temporary tensors
then run anchor
```

Do not keep ranker/self-model pending graphs live across anchor.

#### 5. Use mixed precision for anchor if numerically safe

Potential option:

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    ...
```

or fp16 if bf16 is unavailable.

Caution:

```text
Anchor is a stability/alignment signal.
Check loss scale, NaNs, and whether evaluator scores remain sane before trusting mixed precision.
```

#### 6. Gradient checkpointing through Ouro

Potentially high-value but more invasive.

Since frozen Ouro must still pass gradients to inputs, activation memory is large. Gradient checkpointing trades compute for memory.

Possible direction:

```python
ouro_model.gradient_checkpointing_enable()
```

or manual checkpointing around the looped forward if the custom Ouro wrapper supports it.

Caution:

```text
This can slow anchor training substantially and may interact with custom loop-state extraction.
Treat as a separate experiment.
```

#### 7. Reduce anchor token count / trajectory payload

If anchor currently retains more tokens/states than necessary, inspect whether it can score only the loop states actually required by the evaluator.

Caution:

```text
Do not change the evaluator input semantics silently.
If token pooling or state selection changes, record it as a different anchor variant.
```

#### 8. Disable nonessential heads during anchor

During anchor training, only modules that should receive anchor gradients need to participate.

Potential cleanup:

```text
ensure action prior / nextframe / objectivity heads are not retaining graphs
zero grads with set_to_none=True
detach unrelated tensors
```

### Less recommended options

#### CPU evaluator offload

Moving the frozen evaluator to CPU might save some VRAM, but it is likely slow and awkward because gradients through evaluator inputs must still connect back to GPU tensors.

Not recommended unless GPU memory is impossible and speed does not matter.

#### Switching evaluator to train mode

Avoid this. It can bypass the cuDNN eval-mode issue, but activates dropout and makes anchor scores noisy. The cuDNN-disabled forward is cleaner.

### Practical recommendation for now

For the immediate ladder:

```text
Use ANCHOR_BATCH_SIZE=1.
Run --quick until anchor.successes > 0 is visible.
Then run --vram-check with ANCHOR_BATCH_SIZE=1.
Only then run the full ladder.
```

For the next QoL/performance patch:

```text
Implement anchor microbatch gradient accumulation:
effective anchor_batch_size can be 4 or 8,
peak anchor_microbatch_size stays 1.
```

This is the best path to a stronger anchor signal on a 12 GB GPU.


---

## Quick-ladder comparator result: anchor fires but has no eligible pairs

Update after running:

```bash
python3 -m claude_sandbox.compare_ladder_summaries \
  claude_sandbox/ablation_event_dumps/step_* \
  --alarms \
  --json-out /tmp/ladder_quick_summary.json
```

### Comparator result

Anchor-enabled steps reported:

```text
Step 3.5:
  attempts = 7
  successes = 0
  skipped_insufficient_pairs = 7
  skipped_frozen_encoder = 0
  skipped_no_evaluator = 0
  skipped_oom = 0
  current_batch_size = 1
  batch_ceiling = 1

Step 6:
  attempts = 7
  successes = 0
  skipped_insufficient_pairs = 7
  skipped_frozen_encoder = 0
  skipped_no_evaluator = 0
  skipped_oom = 0
  current_batch_size = 1
  batch_ceiling = 1

Step 7:
  attempts = 7
  successes = 0
  skipped_insufficient_pairs = 7
  skipped_frozen_encoder = 0
  skipped_no_evaluator = 0
  skipped_oom = 0
  current_batch_size = 1
  batch_ceiling = 1
```

### Interpretation

This is an important distinction.

The anchor is no longer failing because of:

```text
- frozen encoder
- missing evaluator
- cuDNN/eval-mode GRU backward
- CUDA OOM
```

Those issues appear resolved for quick mode with `ANCHOR_BATCH_SIZE=1`.

The current problem is:

```text
anchor attempts are firing, but every attempt skips because no eligible chosen/rejected pairs are available.
```

So the anchor call path is alive, but the anchor loss is still not actually being applied.

### Current status

```text
✅ anchor call path works
✅ encoder unfreezes correctly
✅ batch size 1 is accepted
✅ no OOM / cuDNN / frozen-encoder issue
✅ anchor counters are being written and aggregated
❌ anchor successes = 0
❌ anchor loss is not being applied yet
❌ anchor-vs-no-anchor comparison is not scientifically valid yet
```

### Likely cause

The anchor sampler likely reuses the ordinary ranking-pair source. In a short 80-step quick run, especially with no solved trusted trajectories and limited quality variation, there may be no valid chosen/rejected pairs by the time anchor fires.

This can mean either:

```text
1. The run is simply too short for valid anchor pairs.
2. The pair-quality criteria are too strict for Hunter Seeker's early online buffer.
3. The anchor sampler is looking in the wrong buffer or not considering the right eligible transitions.
4. The sampler correctly excludes auxiliary/sibling predicted transitions, leaving no true real-transition pairs.
```

The distinction matters.

### Scientific caution

Do not silently train anchor on junk pairs just to make `successes > 0`.

In particular, be careful with sibling/auxiliary candidates:

```text
- They may be useful for a separate anchor-ablation.
- They should not silently replace real chosen/rejected trajectory pairs.
- If they are model-predicted rather than real environment transitions, they can contaminate the intended ground-truth anchor signal.
```

For real anchor training, preferred pair sources are:

```text
1. Real online ranking pairs with enough quality separation.
2. Trusted trajectory pairs.
3. Explicitly labeled diagnostic/smoke pairs only when the run is marked as smoke-test, not scientific training.
```

### Required next verification

The next question is:

```text
Are there no pairs because quick mode is too short,
or because the anchor sampler is using the wrong buffer / criteria?
```

Add diagnostics around anchor pair sampling:

```text
- online buffer length
- trusted buffer length
- auxiliary/sibling buffer length if applicable
- candidate pair count before filtering
- candidate pair count after filtering
- reason pair sampling failed
- whether the sampler attempted online, trusted, or fallback source
```

### Full-ladder readiness

Do not run the full ladder yet.

The quick ladder is structurally valid, but anchor-vs-no-anchor comparisons remain invalid until:

```text
anchor.successes > 0
anchor.loss_ema appears
anchor skipped_insufficient_pairs no longer equals attempts
```

### AttnRes alarm note

The comparator also reported argmax-share alarms such as:

```text
Step 4: L1 = 100% of 8 calls
Step 6: L1 = 100% of 4 calls
Step 7: L3 = 100% of 4 calls
```

This is not yet strong evidence of AttnRes collapse because the sample counts are tiny.

The mean attention and entropy look healthy:

```text
attn_l1/l2/l3/l4 means are roughly balanced
attn_entropy ≈ 1.38–1.39, close to log(4)
```

Recommendation:

```text
Keep the mean-attention and entropy alarms as-is.
For argmax-share alarms, require a minimum count before firing, e.g. 20 or 50 calls.
```

Otherwise quick-smoke runs produce noisy 100% argmax alarms from only a handful of observations.

### Updated readiness status

```text
Quick ladder structure: passed
Measurement dumps: passed
AttnRes execution: passed
Self-model execution: passed
Anchor call path: passed
Anchor actual training: not yet passed
Full ladder: blocked until anchor pair availability is diagnosed/fixed
```
