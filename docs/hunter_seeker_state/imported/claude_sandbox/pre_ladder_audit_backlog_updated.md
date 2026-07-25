<!-- Imported from `claude_sandbox/pre_ladder_audit_backlog_updated.md` during Hunter-Seeker state consolidation on 2026-05-14. -->
<!-- Original SHA256: e2f889266478147f55bf08b3f436bae2077bef8df485fdfd3728b6daba99e0a7; original line count: 750. -->

# Pre-Ladder Audit Notes and Deferred Cleanup

Status: notes to preserve the remaining issues and follow-up ideas before the ablation ladder run.

The current priority is still:

```text
patch the real pre-run blockers
→ run --quick
→ run --vram-check
→ run the full ladder
→ compare summaries
→ only then resume architecture work
```

This file is for everything that should not be lost, but should mostly not block the ladder.

---

## Immediate pre-ladder patch items

These are worth doing before the ladder because they affect whether the run is trustworthy.

### 1. Fix Step 0 checkpoint path

The runner currently looks for the v17b / Sprint-4 reference checkpoint under:

```text
claude_sandbox/checkpoints_running/sprint4_encoder_reverted.pt
```

but the actual file is at root-level:

```text
checkpoints_running/sprint4_encoder_reverted.pt
```

If this is not fixed, Step 0 will skip and the ladder loses the historical floor.

Recommended behavior:

```text
- Respect V17B_CHECKPOINT env override if set.
- Otherwise check root-level checkpoints_running first.
- Optionally fall back to claude_sandbox/checkpoints_running.
- Print the resolved path before Step 0.
```

### 2. Make `--quick` exercise anchor training

`--quick` currently uses roughly 80 steps, while anchor steps use:

```text
--anchor_train_every 100
```

That means quick mode probably never fires anchor training, so it does not validate anchor counters, anchor JSON fields, or adaptive anchor batch behavior.

Recommended fix:

```text
ANCHOR_EVERY=100 by default
ANCHOR_EVERY=10 in --quick mode
```

Then use `$ANCHOR_EVERY` for Steps 3.5, 6, and 7.

### 3. Fix stale runner comment

The runner comment says measurement JSONs land under:

```text
--checkpoint_dir/<game>/
```

but they actually land under:

```text
--dump_events_dir/<game>/measurement_run_<N>.json
```

This is only a comment bug, but it is exactly the kind of doc lie that wastes time later.

### 4. Add finite guard to click softmax

`ActionHead.select_action` now has a finite/NaN guard for action probabilities, but the click softmax path should have the same protection.

Recommended pattern:

```python
click_probs = F.softmax(flat_logits / temperature, dim=-1)
if not torch.isfinite(click_probs).all():
    click_probs = torch.ones_like(flat_logits) / flat_logits.numel()
```

or equivalent shape-safe fallback.

### 5. Align comparator AttnRes alarm docs and behavior

`compare_ladder_summaries.py` says it checks:

```text
any-loop argmax-share > 80%
```

but the implementation checks:

```text
attn_l{i}_mean > 0.80
```

Both are useful, but they are not the same.

Recommended fix:

```text
- Keep mean-attention > 0.80 alarm.
- Add argmax-share > 0.80 alarm if argmax counts exist.
- Update the docstring to mention both.
```

### 6. Optional but surgical: empty `available_actions` fallback

ARC currently does not return an empty available-action set, so this is dormant for ARC. But for the “no ARC assumptions in core” rule, the core should not crash if a future adapter returns an empty list.

Known fragile pattern:

```python
np.random.choice(list(available_actions))
```

Recommended minimal fallback:

```python
available_actions = list(available_actions)
if not available_actions:
    available_actions = list(range(self._action_adapter.n_actions))
```

Do this only if it stays surgical. Do not redesign the action adapter before the ladder.

---

## Defer until after ladder

These are real, but should not be patched before the first ladder unless the run exposes them directly.

### 1. Progress semantics still leak ARC into the base agent

The base pairwise agent still uses ARC-shaped concepts such as:

```text
levels_completed
_QUALITY_LEVEL_SCALE
```

This is a real architecture leak. The ARC harness is allowed to know ARC, but the core should eventually consume environment-neutral progress/outcome signals.

Long-term replacement:

```python
obs_adapter.progress_value(obs)
obs_adapter.terminal_state(obs)
obs_adapter.transition_quality(prev_obs, obs, frame_changed, info)
```

or a separate `OutcomeAdapter`.

Do not fix this immediately before the ladder. It changes the meaning of ranker supervision and would contaminate the comparison.

### 2. Rename `ColorPriorTable` / `color` terminology

For ARC, “color” works. For the general architecture, this should eventually become:

```text
color → label / object_label / segment_id
ColorPriorTable → LabelPriorTable / EntityPriorTable
```

This is conceptual hygiene, not a pre-run blocker. Renaming now risks dumb breakage and gives no ladder value.

### 3. Remove legacy ARC defaults from generic modules

Some defaults are still ARC-shaped, even if real adapter paths pass correct values:

```text
ActionHead(n_actions=8, click_action_idx=6)
normalize_clicks(grid_w=64, grid_h=64)
```

Long-term, make these required args or mark them as legacy defaults only.

This is not a ladder blocker as long as adapters pass explicit values.

### 4. RUN_ID / timestamped ladder directories

The runner uses fixed paths like:

```text
ladder_step_3.pt
ablation_event_dumps/step_3
ablation_runs/step_3
```

This is okay for a single clean run, but repeated runs can reuse stale checkpoints or mix outputs.

Before the full run, manual cleanup is acceptable:

```bash
rm -f claude_sandbox/checkpoints_running/ladder_step_*.pt
rm -rf claude_sandbox/ablation_event_dumps/step_*
rm -rf claude_sandbox/ablation_runs/step_*
```

Long-term improvement:

```text
RUN_ID=$(date +%Y%m%d_%H%M%S)
outputs/checkpoints/event dumps all nested under RUN_ID
```

Nice to have, not required before first ladder.

### 5. Comparator score-component aggregation

The runner mentions comparing things like:

```text
effective_confidence_pre_ouro
effective_confidence
ranker_score_raw
heuristic_score_raw
```

but those fields live in per-step traces, not necessarily in `measurement_summary`.

Do not add this to the comparator until the harness actually emits those fields in summaries.

Future direction:

```text
- Add score_components summary means to measurement_summary.
- Then aggregate them in compare_ladder_summaries.py.
```

### 6. `anchor.loss_ema == 0.0` filtering

The comparator currently ignores zero `loss_ema` values. That may be intended to skip inactive defaults, but it can bias results if zero is ever real.

Not urgent. The key anchor fields are more important:

```text
attempts
successes
success_rate
skips
current_batch_size
batch_ceiling
```

### 7. Ouro confidence as primary trust axis

Ouro exit confidence is currently captured and blended into the trust gate. It is not yet the primary trust axis.

This should remain deferred until ladder data exists. Do not redesign the trust gate before seeing whether the current blend helps or hurts.

Future question:

```text
Should S_t(x) / Ouro exit confidence replace loop_delta as the main trust signal,
or should it remain a multiplier?
```

### 8. Sleep / consolidation system

Sleep should not be bundled into the current ladder phase.

Suggested staging:

```text
Sprint 7  = synthetic mechanic curriculum
Sprint 9a = abstract memory scaffold / prototype API
Sprint 10 = loop-state-centric / object-file-delta ranker
Sprint 8  = grounded offline replay / consolidation
Sprint 12 = imagined rollouts + backward reconstruction
Sprint 13 = abstraction / prototype formation / self-distillation
```

The key rule:

```text
Sprint 8 may only train from real recorded transitions/events.
Sprint 12 may imagine/reconstruct but must validate/filter.
Sprint 13 may compress into prototypes only after replay/reconstruction is stable.
```

---

## Micro-optimizations and cleanup ideas

These should not block the ladder.

### 1. Vectorize `pad_grids_to_batch`

If it still uses Python row/column loops, replace with tensor slicing:

```python
arr = torch.as_tensor(g, dtype=torch.long)
batch[i, :h, :w] = arr
```

### 2. Cache `ArcActionAdapter` enum map

If `decode()` rebuilds an enum map every call, cache it by enum class identity. Tiny optimization only.

### 3. Stable frame hash

Python `hash(frame.tobytes())` is fine for in-process caches, but process-salted and not stable across runs.

If cache keys ever need to persist or compare across processes, use:

```text
hashlib.blake2b
xxhash
```

### 4. SceneParser cache memory

SceneParser caches full object masks. Fine for ARC; potentially expensive for larger domains.

Future improvement:

```text
- store bounding-box-local masks
- or store one label map plus object metadata
```

### 5. Cache visual saliency by frame hash

`visual_saliency(frame)` is cheap at ARC scale, but if reused heavily during candidate generation, cache by frame hash.

---

## Current run order

Once the immediate patch items are done:

```bash
./claude_sandbox/run_ablation_ladder.sh --quick
```

Then verify measurement summaries exist:

```bash
find claude_sandbox/ablation_event_dumps -name 'measurement_run_*.json' | head
```

Then run the comparator:

```bash
python -m claude_sandbox.compare_ladder_summaries   claude_sandbox/ablation_event_dumps/step_*   --alarms   --json-out /tmp/ladder_quick_summary.json
```

If quick works:

```bash
./claude_sandbox/run_ablation_ladder.sh --vram-check
```

Then, after checking anchor batch behavior and VRAM stability:

```bash
./claude_sandbox/run_ablation_ladder.sh
```

---

## Design law to preserve

ARC is the empirical vehicle, not the architecture.

Allowed to be ARC-specific:

```text
train_arc_codex.py
ArcObservationAdapter
ArcActionAdapter
ARC game loading / arc.make(game_id)
ARC palette/frame dumping
ARC-specific evaluation scripts
```

Not allowed to be ARC-specific:

```text
GridEncoder
loop-state poolers
self-model
pairwise/evaluator anchor
object/event memory
affordance machinery
ranker interface
candidate scoring logic
action/observation contracts
topology/mechanism abstractions
```

The final architecture should continue moving toward:

```text
domain-specific adapter/harness
        ↓
canonical observations/actions/events
        ↓
domain-general cognition around Ouro/RLTT loop states
```

not:

```text
ARC quirks
        ↓
agent brain
```


---

## Temporary vs permanent patch classification

Added after the quick-ladder smoke run exposed the `sm_grad_active`, pending self-model graph, AttnRes logging, and anchor-unfreeze issues.

### Keep permanently

These are real robustness/correctness fixes, not hacks.

#### 1. Initialize `sm_grad_active = False`

Status: permanent.

Reason:

```text
_train_ranker() uses sm_grad_active later even when the self-model gradient path does not activate.
A local variable used in a later conditional must have a default value.
```

Permanent solution:

```python
def _train_ranker(...):
    sm_grad_active = False
    ...
```

#### 2. Generic loop-pooler gate logging

Status: permanent.

Reason:

```text
The logging path assumed every loop pooler exposes `gru_gate`.
GRU pooler exposes `gru_gate`; AttnRes pooler exposes `attn_gate`.
Both are legitimate poolers.
```

Permanent solution:

```python
if hasattr(self.loop_pooler, "gru_gate"):
    gate_tensor = self.loop_pooler.gru_gate
elif hasattr(self.loop_pooler, "attn_gate"):
    gate_tensor = self.loop_pooler.attn_gate
else:
    gate_tensor = None

if gate_tensor is not None:
    info["gru_gate"] = float(torch.tanh(gate_tensor).item())
```

Longer-term naming cleanup:

```text
info["loop_pooler_gate"] instead of info["gru_gate"]
```

Keep `gru_gate` as a backward-compatible alias until the training/logging scripts are updated.

#### 3. Finite guards for action and click softmax

Status: permanent.

Reason:

```text
NaN/Inf logits should not crash action selection.
If the policy head produces invalid probabilities, fallback to uniform sampling.
```

Permanent solution:

```python
if not torch.isfinite(action_probs).all():
    action_probs = uniform_over_valid_actions

if not torch.isfinite(click_probs).all():
    click_probs = uniform_over_click_map
```

#### 4. Empty `available_actions` fallback

Status: permanent.

Reason:

```text
ARC normally returns non-empty available actions, but future adapters may accidentally return empty lists.
The domain-general core should not crash on np.random.choice([]).
```

Permanent solution:

```python
available_actions = list(available_actions)
if not available_actions:
    available_actions = list(range(self._action_adapter.n_actions))
```

Longer-term solution:

```text
Add an ActionAdapter method such as safe_action_indices(obs) or fallback_action_index().
The core should ask the adapter for a guaranteed non-empty action set.
```

#### 5. Step 0 checkpoint resolution

Status: permanent.

Reason:

```text
The v17b historical-floor checkpoint lives at root-level checkpoints_running/.
The runner must not silently skip Step 0 due to looking only inside claude_sandbox/.
```

Permanent solution:

```text
Resolution order:
1. explicit V17B_CHECKPOINT env var
2. project-root checkpoints_running/sprint4_encoder_reverted.pt
3. sandbox-local fallback
4. loud warning if missing
```

#### 6. Quick-mode anchor frequency

Status: permanent.

Reason:

```text
--quick is supposed to smoke-test the ladder mechanics.
If quick mode uses max_steps=80 but anchor_train_every=100, it never tests anchor counters, anchor JSON, or adaptive batch fields.
```

Permanent solution:

```text
ANCHOR_EVERY=100 by default
ANCHOR_EVERY=10 in --quick mode
```

#### 7. Comparator alarm doc/implementation alignment

Status: permanent.

Reason:

```text
The comparator should not claim to check argmax-share collapse while only checking mean-attention collapse.
Both diagnostics are useful but distinct.
```

Permanent solution:

```text
Keep both:
- mean attention weight > 0.80
- argmax share > 0.80 when argmax counts exist
```

---

## Keep for now, redesign later

These patches are acceptable for unblocking the ladder, but the final architecture should replace them with cleaner mechanisms.

### 1. Pending self-model event graph guard in `_train_ranker()`

Status: temporary safety guard.

Current patch:

```python
_pending_event_pred = getattr(self, "_pending_event_pred", None)
if (
    sm_grad_active
    and _pending_event_pred is not None
    and getattr(_pending_event_pred, "requires_grad", False)
):
    sm_grad_active = False
```

Why it exists:

```text
The self-model path stores a delayed pending event-prediction graph across steps.
_train_ranker() can also step self_model_optimizer.
If ranker training mutates self-model params before the delayed pending graph is consumed,
the next sm_loss.backward() hits a PyTorch version-counter error.
```

Why this is only a guard:

```text
It prevents the optimizer collision, but it does not solve the deeper scheduling problem:
live autograd graphs should not be kept across steps where another optimizer may mutate their parameters.
```

Permanent solution options:

```text
Option A — strict ordering:
  At the beginning of each agent step:
    consume pending self-model event loss
    backward/step self_model_optimizer
    clear pending graph
  Then allow ranker training.

Option B — split self-model update:
  _self_model_consume_pending_loss()
  _self_model_advance_state_and_predict_next()
  Make sure no other optimizer step can happen between graph creation and graph consumption.

Option C — single owner:
  Only one training routine owns self_model_optimizer.
  Ranker loss contributes to self-model through an accumulated loss queue, but stepping happens in one central place.

Option D — detach pending prediction:
  Store pending predictions detached from the graph.
  This avoids the error, but removes event-prediction gradient through the self-model, so it is not preferred unless the event prediction is diagnostic-only.
```

Recommended permanent direction:

```text
Use Option A or B.
Do not carry live self-model graphs across arbitrary ranker optimizer steps.
The event-prediction loss should be consumed before any ranker path can mutate self-model parameters.
```

For the current ladder:

```text
Keep the guard.
It is safer than crashing and does not stop ranker training; it only prevents ranker loss from stepping self-model while a pending event graph is live.
```

### 2. Anchor unfreeze via `--unfreeze_encoder_after_partial_load`

Status: semi-temporary interface.

Current behavior:

```text
If anchor training is requested and --unfreeze_encoder_after_partial_load is passed,
the train harness unfreezes the encoder even for fresh agents where no partial-load fallback fired.
```

Why it exists:

```text
Anchor training is a no-op if the encoder is frozen.
The old flag name only mentions partial-load fallback, but Step 3.5/6/7 need the encoder trainable even when the agent starts fresh.
```

Why the interface is ugly:

```text
The flag name describes an implementation detail, not the actual training intent.
```

Permanent solution:

Add a clearer explicit flag:

```text
--unfreeze_encoder_for_anchor
```

or:

```text
--train_encoder_for_anchor
```

or a general pair:

```text
--freeze_encoder
--no_freeze_encoder
```

Recommended final behavior:

```text
if anchor_train_every > 0:
    if not encoder_trainable:
        either:
          - auto-unfreeze when --unfreeze_encoder_for_anchor is present
          - or fail loudly with an actionable error
```

Do not silently run anchor training with a frozen encoder.

For now:

```text
Keep using the current flag to unblock the ladder.
After the ladder, rename or add the clearer flag and deprecate the partial-load-specific meaning.
```

---

## Permanent cleanup checklist after ladder

After the quick/vram/full ladder sequence, revisit these so the smoke-test patches become clean architecture:

```text
1. Rename info["gru_gate"] → info["loop_pooler_gate"] with backward-compatible alias.
2. Replace pending-event graph guard with ordered self-model loss consumption.
3. Add explicit --unfreeze_encoder_for_anchor / --train_encoder_for_anchor flag.
4. Add ActionAdapter.safe_action_indices(obs) or fallback_action_index().
5. Move ARC progress semantics out of the base agent into an outcome/progress adapter.
```

Do not do items 1–5 before the current ladder unless another smoke-test crash requires it.
