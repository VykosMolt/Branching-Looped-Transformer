<!-- Imported from `docs/root_notes_20260429_143517/hunter_seeker_terminal_memory_handoff_codex.md` during Hunter-Seeker state consolidation on 2026-05-14. -->
<!-- Original SHA256: 7d3ae68755980cf7af4e6aa3020cf1525c41093b26fcca72a6d5fc962311d4bf; original line count: 606. -->

# Hunter Seeker Terminal-Memory Handoff for Codex

Date/context: 2026-04-28 to 2026-04-29  
Project path: `~/ouro_project`  
Primary file: `claude_sandbox/arc_agent_hunter_seeker_codex.py`  
Primary test file: `claude_sandbox/test_causal_correctness.py`  
Main game under test: `ls20`  
Current best probe: `terminal_predframe_context_smoke_2run`

---

## 1. Executive summary

We were debugging Hunter Seeker's generic terminal-outcome memory. The goal is to let the agent remember that a selected latent action-context led to a terminal negative outcome, then downweight similar future candidates without hard-coding ARC, ls20, colors, hazards, walls, exits, avatars, or topology-specific labels.

The previous versions passed many unit tests but failed in live ls20 ladders. They relied almost entirely on exact terminal keys. Prototype terminal memory never activated because terminal context vectors were degenerate: `current_cls` and candidate `successor_cls` often had zero delta for every candidate. This caused thousands of `degenerate_vector_skips`, low key diversity, and repeated mechanism deaths.

The current patch direction is promising: terminal context key/vector construction now includes candidate-specific predicted-frame information, visible in keys like `a=2|d=...|f=...|p=-1,-1`. After this, key diversity improved dramatically, prototype memory started writing and matching, and the 2-run smoke showed run 1 died once but run 2 did not add another death (`failure_counts` stayed `{'mechanism': 1}`). This is not solved, but it is the first structurally correct terminal-memory result in this sequence.

Do not tune penalty constants further until the current predframe/context patch is validated with an 8-run ladder.

---

## 2. Constraints and coding style

The user explicitly asked: do not throw patches around blindly. Ask for snippets/definitions and diagnose before patching. Prefer small, targeted edits. Do not rewrite the file wholesale unless explicitly instructed.

Terminal memory must remain domain-agnostic. Avoid adding any of these concepts inside generic terminal-memory helpers: `ls20`, `arc-agi`, `hazard`, `avatar`, `wall`, `exit`, `color`, `topology_frontier`. Existing tests statically check that `_remember_terminal_outcome_counterevidence` does not smuggle those terms.

Keep all existing features unless a change is justified by a failing diagnostic.

---

## 3. Terminal memory design intent

The system has two terminal-memory paths:

1. **Exact terminal memory**
   - Stores exact opaque action-context keys.
   - Should be a weak warning, not a veto.
   - Should not become an action-level blacklist.
   - Counterevidence can attenuate it but must not create terminal memory entries by itself.

2. **Prototype terminal memory**
   - Stores normalized context vectors and matches future candidates by similarity.
   - Intended to be the stronger/generalizing terminal-risk path.
   - Should activate only when candidate-specific context exists.

Counterevidence behavior:

- On nonterminal transition after selected action, `_remember_terminal_outcome_counterevidence(...)` is called with the selected context key/vector.
- If terminal memory already exists, counterevidence attenuates risk with a capped count.
- If terminal memory does not exist, pending counterevidence is stored and later merged if that context becomes terminal.
- Counterevidence alone must not create a terminal-memory entry.
- Real terminal risk must not be erased to zero.

---

## 4. Earlier bad result: `terminal_memory_probe_8run_post_pending_exactfix`

Probe path:

```bash
claude_sandbox/ablation_event_dumps/terminal_memory_probe_8run_post_pending_exactfix
```

Result summary:

- 8 runs on `ls20`.
- `levels_completed: 0` throughout.
- Failure count climbed almost every run:
  - run 1: `{'mechanism': 1}`
  - run 2: `{'mechanism': 2}`
  - run 3: `{'mechanism': 3}`
  - run 4: `{'mechanism': 4}`
  - run 5: `{'mechanism': 5}`
  - run 6: `{'mechanism': 6}`
  - run 7: `{'mechanism': 7}`
  - run 8: `{'mechanism': 7}`
- Exact memory grew to 4 entries.
- Exact penalties became huge in aggregate.
- Prototype memory never activated:
  - `prototype_count: 0`
  - `prototype_penalty_calls: 0`
  - `prototype_penalty_sum: 0.0`
- Degenerate vector skips reached thousands:
  - run 8: `degenerate_vector_skips: 6646`

Representative run 8 terminal stats:

```text
exact_size: 4
prototype_count: 0
exact_penalty_calls: 2444
exact_penalty_sum: -3032.8034
prototype_penalty_calls: 0
combined_penalty_calls: 2444
combined_penalty_sum: -3032.8034
counterevidence_context_writes: 588
degenerate_vector_skips: 6646
```

Interpretation: exact memory was doing all the work, prototype memory was dead, and the agent still kept dying.

---

## 5. Soft exact memory / nonzero exact strengthening

We patched `_terminal_outcome_penalty` to make exact terminal memory weak and bounded. The intended policy:

- keep exact terminal writes;
- preserve pending/safe counterevidence;
- cap counterevidence;
- heavily soften zero-delta exact penalties;
- keep exact penalties bounded even after repeated deaths.

A unit test then failed:

```text
test_terminal_exact_counterevidence_cannot_zero_real_terminal_risk
assert -0.06578947368421052 < -0.25
```

Cause: the nonzero exact terminal branch had become too weak; continuation evidence could reduce a real terminal key into numerical noise.

We strengthened the nonzero exact branch to approximately:

```python
base = 0.90 + 0.15 * max(0, terminal_count - 1)
max_abs_penalty = 1.25
risk_floor = 0.35
```

After that, targeted terminal tests passed. But this did not solve live behavior because live candidate transitions still produced zero-delta vectors and prototype memory remained dead.

---

## 6. Result of `terminal_memory_probe_8run_post_soft_exact_strengthfix`

Probe path:

```bash
claude_sandbox/ablation_event_dumps/terminal_memory_probe_8run_post_soft_exact_strengthfix
```

Run outcome:

- 8 runs on `ls20`.
- `levels_completed: 0` throughout.
- It still died every run:
  - run 8: `failure_counts: {'mechanism': 8}`
- Exact penalties were much softer than before:
  - run 8 `combined_penalty_sum: -111.4053`, not thousands.
  - min penalties around `-0.133`.
- Prototype memory still never activated:
  - `prototype_count: 0`
  - `prototype_penalty_calls: 0`
  - `prototype_penalty_sum: 0.0`
- Degenerate vector skips still huge:
  - run 8: `degenerate_vector_skips: 6486`

Interpretation: exact softening prevented huge exact penalties, but did not solve repeated terminal mechanisms. The core issue was upstream context construction, not penalty strength.

---

## 7. Key discovery: action traces, not events, contain terminal score components

Initial event-only search found no terminal keys because terminal score components are not stored in `events`. Dump structure is:

```text
root dict:
  game_id
  step_count_at_dump
  run_start_step
  step_count_in_run
  event_log_capacity
  events
  action_trace
```

Terminal fields live in `action_trace`, inside `top_candidates` / `chosen_candidate`, under `score_components`.

Once we walked `action_trace`, the old soft-exact run showed:

```text
terminal_exact_penalty: 4111
terminal_prototype_penalty: 4111
terminal_outcome_penalty: 4111
terminal_counter_count: 4111
terminal_context_key: 4111
terminal_prototype_matched: 4111
unique_terminal_context_keys: 17
total_terminal_context_key_occurrences: 4111
negative penalties: 2563
min penalty: -0.13344
prototype matched: false everywhere
```

Interpretation: terminal fields existed, but prototype never matched and keys had poor diversity.

---

## 8. Key-diversity diagnostics

We added measurement-only key-diversity diagnostics to `measurement_summary()['terminal_outcome_memory']`:

```text
key_diversity.total_count
key_diversity.unique_count
key_diversity.top_repetition
key_diversity.zero_delta_count
key_diversity.top_examples
key_diversity.by_action
```

Purpose: verify whether exact keys are real candidate contexts or collapsed action signatures. Keep this diagnostic for now.

Probe before predframe patch:

```bash
PROBE=terminal_key_diversity_smoke_2run
```

Run 1:

```text
total_count: 368
unique_count: 4
top_repetition: 92
zero_delta_count: 368
by_action: each action had total_count=92 unique_count=1 top_repetition=92
prototype_count: 0
degenerate_vector_skips: 736
```

Run 2:

```text
total_count: 764
unique_count: 24
top_repetition: 129
zero_delta_count: 764
prototype_count: 0
degenerate_vector_skips: 1528
```

Interpretation: keys were mostly one/few repeated signatures per primitive action; latent delta section was zero; prototype vector was degenerate.

---

## 9. Shape probe: root cause of dead prototypes

A temporary `TerminalShapeProbe` was inserted in `score_candidates` near terminal context construction. It printed:

```text
[TerminalShapeProbe] current_cls (2048,) successor_cls (4, 2048) candidates 4
  cand 0 action 1 delta_rms 0.0 succ_norm 45.2545
  cand 1 action 2 delta_rms 0.0 succ_norm 45.2545
  cand 2 action 3 delta_rms 0.0 succ_norm 45.2545
  cand 3 action 4 delta_rms 0.0 succ_norm 45.2545
```

This repeated across steps.

Core finding:

- `current_cls` is a single `(2048,)` vector.
- `successor_cls` is `(4, 2048)`, but candidate rows were identical / effectively equal to current latent state.
- Candidate-specific latent delta was zero for every candidate.
- `_terminal_action_context_vector` correctly refused these as degenerate.
- Prototype terminal memory could not learn.
- Exact keys based only on current/successor/delta collapsed.

Therefore the fix needed candidate-specific context from a different source. `predicted_frames[i]` was the natural candidate-specific signal already available in `score_candidates`.

Important: remove or disable temporary `TerminalShapeProbe` prints before normal ladder runs:

```bash
grep -n "TerminalShapeProbe\|_terminal_shape_probe_prints" claude_sandbox/arc_agent_hunter_seeker_codex.py
```

---

## 10. Current patch direction: predicted-frame / candidate-specific terminal context

The current patch appears to extend terminal context keys/vectors with candidate-specific predicted-frame information. Keys now look like:

```text
a=2|d=...|f=...|p=-1,-1
```

Interpretation:

- `a=` primitive action id.
- `d=` latent current/successor/delta sketch.
- `f=` predicted-frame or frame-derived candidate sketch.
- `p=` click position; for non-click actions often `-1,-1`.

This is the correct structural fix because the latent successor path was not candidate-informative.

Subtlety:

- `key_diversity.zero_delta_count` is still equal to total count because the `d=` latent delta section remains zero.
- That is okay for now because `f=` / predicted-frame context adds candidate specificity.
- `_looks_zero_delta_key` in `_terminal_outcome_penalty` probably still only inspects `d=`, so exact penalties remain weak. This is acceptable because prototype memory is now the main risk path.
- Do not strengthen exact penalties just because `zero_delta_count` is high. Exact zero-delta penalties were previously dangerous.

---

## 11. Current best result: `terminal_predframe_context_smoke_2run`

Probe path:

```bash
claude_sandbox/ablation_event_dumps/terminal_predframe_context_smoke_2run
```

Command used:

```bash
PROBE=terminal_predframe_context_smoke_2run

rm -rf "claude_sandbox/ablation_event_dumps/$PROBE"
rm -f "claude_sandbox/checkpoints_running/$PROBE.pt"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
./venv/bin/python -m claude_sandbox.train_arc_codex \
  --games ls20 \
  --n_runs 2 \
  --max_steps 160 \
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
  --running_checkpoint "claude_sandbox/checkpoints_running/$PROBE.pt" \
  --dump_events_dir "claude_sandbox/ablation_event_dumps/$PROBE"
```

High-level result:

- Run 1 died at step 129:
  - `failure_counts: {'mechanism': 1}`
- Run 2 reached max step / did not add another death:
  - measurement run 2 still `failure_counts: {'mechanism': 1}`
- `levels_completed` still 0.
- This is not solved, but it is a real improvement over repeated death every run.

Run 1 terminal stats:

```text
size: 1
prototype_count: 1
exact_penalty_calls: 0
prototype_penalty_calls: 0
combined_penalty_calls: 0
counterevidence_context_writes: 0
counterevidence_prototype_writes: 0
degenerate_exact_skips: 0
degenerate_vector_skips: 0
prototype_count_by_action: {"2": 1}
```

Run 1 key diversity:

```text
total_count: 258
unique_count: 238
top_repetition: 3
zero_delta_count: 258
```

Run 2 terminal stats:

```text
size: 1
prototype_count: 1
exact_penalty_calls: 0
exact_penalty_sum: 0.0
prototype_penalty_calls: 4
prototype_penalty_sum: -0.7370763348706075
combined_penalty_calls: 4
combined_penalty_sum: -0.7370763348706075
counterevidence_context_writes: 0
counterevidence_prototype_writes: 2
degenerate_exact_skips: 0
degenerate_vector_skips: 0
prototype_count_by_action: {"2": 1}
prototype_counterevidence_by_action: {"2": 1}
prototype_penalty_by_action: action 2, calls 4, sum -0.7371, min_penalty -0.19836, max_similarity 0.831865
```

Run 2 key diversity:

```text
total_count: 678
unique_count: 558
top_repetition: 8
zero_delta_count: 678
```

Action-trace terminal field summary:

```text
total_key_occurrences: 870
unique_keys: 260
negative penalties: 6
min penalty: -0.1983639906602074
prototype matches: 6
prototype similarity min/max: -1.0 / 0.8318651914596558
```

Example prototype match:

```json
{
  "action": 2,
  "click_x": -1,
  "click_y": -1,
  "terminal_exact_penalty": 0.0,
  "terminal_prototype_penalty": -0.1983639906602074,
  "terminal_outcome_penalty": -0.1983639906602074,
  "terminal_prototype_matched": true,
  "terminal_prototype_similarity": 0.8318651914596558,
  "terminal_counter_count": 0.05,
  "terminal_risk_scale": 0.9960159362549801
}
```

Interpretation:

- Prototype terminal memory is alive.
- Degenerate vector issue is fixed in this smoke.
- Key diversity is no longer collapsed.
- Prototype penalties are modest and sparse but real.
- Run 2 did not repeat the death, which is encouraging but not enough evidence.

---

## 12. Relationship to pre/post ladder markdowns

The pre/post ladder markdowns are the broader measurement framework. This terminal-memory work is a targeted sub-experiment inside that ladder approach.

The **pre-ladder MD** should represent baseline behavior before the current terminal-memory/predframe fix. It should emphasize:

- repeated ls20 mechanism deaths;
- no prototype terminal memory;
- high `degenerate_vector_skips`;
- low key diversity / high top repetition;
- exact memory either too strong or too weak but structurally not solving the issue;
- 0 levels completed.

The **post-ladder MD** should be generated after a real 8-run predframe-context ladder. It should compare against:

- `terminal_memory_probe_8run_post_pending_exactfix`
- `terminal_memory_probe_8run_post_soft_exact_strengthfix`
- `terminal_predframe_context_smoke_2run`
- any pre-terminal-memory baseline ladder MDs already made

The current 2-run smoke is not the post-ladder. It only says the patch is worth laddering.

The post-ladder MD should answer:

1. Did terminal memory prevent repeated terminal mechanisms?
2. Did it do so via prototype memory rather than exact action-blacklist behavior?
3. Did exploration remain alive?
4. Did levels completed improve?
5. Did terminal memory suppress only a risky action-context or broadly suppress an entire primitive action?
6. Did prototype similarity/penalty behavior remain sane over multiple runs?

Recommended comparison fields:

```text
levels_completed per run
failure_counts and per-run delta
death count per run
run termination step
terminal_outcome_memory.size
prototype_count
prototype_count_by_action
exact_penalty_calls/sum
prototype_penalty_calls/sum
combined_penalty_calls/sum
counterevidence_context_writes
counterevidence_prototype_writes
degenerate_vector_skips
key_diversity.unique_count
key_diversity.top_repetition
prototype matches and max similarity
action distribution
selected-action terminal penalties
```

---

## 13. Recommended next steps

### Step 1: remove temporary shape probe prints

```bash
cd ~/ouro_project
grep -n "TerminalShapeProbe\|_terminal_shape_probe_prints" claude_sandbox/arc_agent_hunter_seeker_codex.py
```

If present, remove/disable only that debug print block. Keep key-diversity diagnostics.

### Step 2: run targeted tests

```bash
cd ~/ouro_project

./venv/bin/python -m pytest -q \
  claude_sandbox/test_causal_correctness.py \
  --import-mode=importlib \
  -k "terminal or compact_score_components"
```

Expected: all pass.

If tests fail, add targeted tests around predframe context rather than deleting existing tests.

Suggested tests:

- Latent zero-delta + different predicted frames should produce different terminal keys.
- Latent zero-delta + predicted frame context should produce non-None terminal vectors.
- Counterevidence still must not create terminal memory entries by itself.
- Prototype counterevidence still attenuates but cannot erase terminal risk.
- Generic terminal-memory code remains domain-agnostic.

### Step 3: run an 8-run predframe ladder

```bash
cd ~/ouro_project

PROBE=terminal_predframe_context_8run_ladder

rm -rf "claude_sandbox/ablation_event_dumps/$PROBE"
rm -f "claude_sandbox/checkpoints_running/$PROBE.pt"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
./venv/bin/python -m claude_sandbox.train_arc_codex \
  --games ls20 \
  --n_runs 8 \
  --max_steps 160 \
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
  --running_checkpoint "claude_sandbox/checkpoints_running/$PROBE.pt" \
  --dump_events_dir "claude_sandbox/ablation_event_dumps/$PROBE"
```

### Step 4: summarize and write post-ladder MD

Use the existing terminal summary script, pointed at:

```text
claude_sandbox/ablation_event_dumps/terminal_predframe_context_8run_ladder
```

Then update/create the post-ladder markdown with the comparison described above.

---

## 14. Red flags to monitor

Terminal-memory red flags:

- `prototype_count == 0` after a death.
- `degenerate_vector_skips` rising with candidate count.
- `key_diversity.unique_count` very low.
- `key_diversity.top_repetition` huge.
- `prototype_penalty_calls == 0` after later similar candidates.
- `terminal_prototype_similarity` always `-1.0`.
- Penalties applied broadly to all candidates of an action.
- Death count still increases every run despite prototype penalties.

Representation / anchor red flags:

- In one earlier run, `Δloop` jumped from about `12` to about `42` in run 2. This may be anchor/encoder drift or representation scale shift.
- Monitor `Δloop`, anchor loss, GRU gate, WM confidence, change rate, and sibling count.
- Do not patch terminal memory to solve representation drift.

---

## 15. Current recommended stance

The predframe/candidate-specific terminal context patch appears to fix the structural bug that killed prototype terminal memory. It should be ladder-tested before more tuning.

Do not change penalty constants yet. The next meaningful question is not “should the penalty be stronger?” but:

> Does prototype terminal memory now reliably prevent repeated terminal mechanisms across an 8-run ladder without becoming a primitive-action blacklist?

