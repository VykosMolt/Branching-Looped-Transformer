<!-- Imported from `docs/root_notes_20260429_143517/ouro_hunter_seeker_handoff_notes_2026-04-28.md` during Hunter-Seeker state consolidation on 2026-05-14. -->
<!-- Original SHA256: d0d5eb05e71db1d0dac62b8d4ad0ef7eea0036c8739557b6cd8c230490edc91f; original line count: 949. -->

# Ouro / Hunter Seeker / ARC-AGI-3 Handoff Notes

**Date:** 2026-04-28  
**Project root assumed:** `~/ouro_project`  
**Main active file:** `claude_sandbox/arc_agent_hunter_seeker_codex.py`  
**Primary harness:** `claude_sandbox.train_arc_codex`  
**Primary test file:** `claude_sandbox/test_causal_correctness.py`  
**Current empirical focus:** `ls20`, terminal-outcome memory, topology/hazard/death handling, and avoiding action-level blacklists.

---

## Executive state

Hunter Seeker is currently in a partially-working Sprint 4/Sprint 5 boundary state.

The most important recent empirical result is that after the terminal-memory context-key/prototype fixes, an 8-run `ls20` probe finally completed level 1 once:

```text
terminal_memory_probe_8run_post_zerodeltafix
Run 4: *** LEVEL 1 COMPLETED at step 116 ***
Score: 0.12846101579751995
Solved levels: [1]
```

This matters because earlier terminal-memory runs were either zero-completion or got trapped by terminal memory becoming an action-level blacklist. The current system can sometimes solve level 1, but it still repeatedly dies later, especially from topology/frontier hazards involving color 5.

The immediate patch point is the prototype counterevidence cap. A test failed because prototype counterevidence reached `25.0` after 500 writes, while the intended invariant caps it at `<= 8.0`.

---

## Project-level goal

The project is an Ouro-backed ARC-style AGI agent.

High-level goal:

1. Use ARC-AGI-3 as the empirical vehicle.
2. Use Hunter Seeker as the environment-understanding front-end.
3. Use Ouro loop states as the reasoning substrate.
4. Avoid ARC-specific hand-coded solutions in the final architecture.
5. Build an agent that learns mechanics by probing, remembering outcomes, and using object/topology/event structure.

Terms:

- **Hunter Seeker** = exploratory environment-understanding agent.
- **Stockfish** = solver-heavy/current pairwise ARC approach.
- **Vesper** = later parallel architecture/paper direction, not the immediate implementation priority.

Canonical roadmap context from the user's project notes:

- ARC is the main empirical vehicle.
- Hunter Seeker is the required environment-understanding front-end.
- Ouro loop states are the main reasoning substrate.
- Vesper is later.
- Near-term ARC steps: failure-taxonomy instrumentation, self/world separation, synthetic curriculum around frozen Ouro, then event/mechanism/topology layers.

---

## Current architecture snapshot

Current run command pattern:

```bash
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
  --running_checkpoint <checkpoint.pt> \
  --dump_events_dir <dump_dir>
```

Printed architecture:

```text
Pairwise ARC Search Agent:
  GridEncoder:      1,845,024
  TransitionRanker: 3,351,366
  ActionPrior:      530,696
  SpatialPredictor: 145,137
  NextFramePredictor: 506,144
  LoopStatePooler:  3,677,185
  Backbone mode: ouro
  Train every: 10, batch: 16, beam: 6x2
  ObjectActionabilityHead: 17,480 params (8 heads)
  SymbolicPlannerHead: 776 params
  SceneParser + ObjectTable: 0 params (deterministic)
  Hunter Seeker unknownness threshold: 0.7
```

Core components:

### GridEncoder

- Encodes ARC grids into latent features.
- Historical critical issue: encoder drift. The user previously restored encoder weights to v1.7b after drift.
- Anchor training is currently used as insurance against drift.

### Ouro backbone

- Uses `Ouro-2.6B-Thinking`.
- Current probes force full loop depth:
  - `early_exit_threshold=1.0`
  - `T=4 forced`
- Ouro loop states feed the agent heads and loop pooler.

### LoopStatePooler

- Enabled with `--use_loop_pooler on`.
- Current kind: `gru`.
- Logs repeatedly show `GRU gate: 0.0000`; this may indicate the GRU gate is inactive, diagnostic-only, or saturated. This should be investigated later, but not during the terminal-memory patch.

### TransitionRanker / ActionPrior / SpatialPredictor

- Current probes often show:
  - `No ranker updates`
  - `No prior updates`
  - `No spatial updates`
- These experiments mainly test terminal memory, objectivity, next-frame, anchor, event, and topology behavior.

### NextFramePredictor

- Trains intermittently.
- Loss tends to drop over multi-run probes.
- Example latest run: average nextframe loss moved roughly from 2.5-ish to 1.3-ish depending on run.

### ObjectActionabilityHead

- 8 heads, 17,480 params.
- Objectivity updates happen regularly.
- Objectivity buffer grows into hundreds of entries.
- Color beliefs include avatar/wall/hazard/exit.

### SymbolicPlannerHead

- Tiny head, 776 params.
- Diagnostics exist, but not the main current bottleneck.

### SceneParser + ObjectTable

- Deterministic.
- Maintains event log, object/event structure, and count summaries.
- Event log capacity usually 500.

### Terminal outcome memory

This is the main active patch area.

It contains:

- exact terminal context memory,
- terminal prototype memory,
- terminal counterevidence,
- score component tracing,
- measurement summaries.

---

## Sprint status

### Sprint 0 — measurement / diagnostics

Status: **mostly complete, still expanding.**

Working:

- Per-run measurement dumps: `measurement_run_N.json`
- Event dumps: `run_N.json`
- Measurement summaries include:
  - `levels_completed`
  - `failure_counts`
  - `event_log_counts`
  - `terminal_outcome_memory`
  - topology death info
  - anchor diagnostics
  - self-model diagnostics when enabled
- Terminal trace diagnostics include:
  - `terminal_context_key`
  - `terminal_exact_penalty`
  - `terminal_prototype_penalty`
  - `terminal_outcome_penalty`
  - `terminal_prototype_similarity`
  - `terminal_counter_count`
  - `terminal_risk_scale`
  - `terminal_prototype_matched`
  - `terminal_prototype_match`

Gap:

- Need a standard analyzer script for exact/prototype penalty breakdown, action suppression, key diversity, would-win-without-penalty, and level-completion/death correlation.

### Sprint 3 — event substrate

Status: **substantially complete.**

Working event types:

- `moved`
- `contact`
- `appeared`
- `transformed`
- `disappeared`
- `death`
- `level_complete`

Terminal events are emitted from `on_level_complete` and `on_game_over`. Failure taxonomy exists and mechanism deaths accumulate.

Known gap:

- Death evidence is now visible, but terminal memory still needs calibration so it avoids repeated deaths without suppressing useful actions.

### Sprint 4 — multi-head affordances / object actionability

Status: **partially complete.**

Working:

- Objectivity updates occur.
- Color/object beliefs are tracked.
- Exit marking works at least once:
  ```text
  [HunterSeeker] Color 8 track#10 marked EXIT (action=1, unkn=0.24)
  ```

Known issues:

- Duplicate color entries can exist with different object beliefs.
- Hazard beliefs can become very strong.
- Still not enough for robust later-level solving.
- `click_change_rate` is `0.000` in current `ls20` traces.

### Sprint 5 — topology

Status: **started / active.**

Evidence:

- Death source includes `topology_frontier`.
- Recent deaths:
  ```text
  DEATH (directional/topology_frontier): color 5 hazard↑
  ```

Known issue:

- After level 1 is solved, the agent repeatedly dies from color 5 topology/frontier hazards.
- Topology can identify the issue but does not yet robustly avoid it.

### Self-model sprint

Status: **disabled in current experiments.**

Current command uses:

```bash
--self_model_mode off
--cortex_monitor_mode off
```

### Anchor / encoder alignment

Status: **active insurance mode.**

Current command uses:

```bash
--anchor_train_every 25
--anchor_coefficient 0.1
--anchor_batch_size 1
--unfreeze_encoder_for_anchor
```

Anchor loss generally falls to tiny values after a few runs, with occasional spikes. Keep anchor diagnostics. Do not remove this until encoder drift is fully ruled out.

---

## Terminal-outcome memory subsystem

### Purpose

The subsystem is meant to remember:

> An action in a similar latent transition context caused a terminal negative outcome before; downweight repeating it.

It must not become:

> Action 2 is bad forever.

or:

> Action 1 is bad in every state.

Terminal memory should be:

- generic,
- domain-agnostic,
- latent-context-conditioned,
- action-conditioned,
- weak enough not to hard veto,
- strong enough to prevent repeated deaths.

### Main functions in `arc_agent_hunter_seeker_codex.py`

- `_terminal_action_context_key`
- `_terminal_action_context_vector`
- `_terminal_outcome_context_key_is_degenerate`
- `_terminal_context_key_is_degenerate`
- `_remember_terminal_outcome_context`
- `_terminal_outcome_penalty`
- `_remember_terminal_outcome_prototype`
- `_remember_terminal_outcome_counterevidence`
- `_terminal_outcome_prototype_penalty`
- `_terminal_outcome_bump_action_diag`
- `_terminal_outcome_prototype_counts_by_action`
- `_terminal_outcome_diag_dict`
- `_compact_score_components`
- `measurement_summary`
- `dump_events_for_sleep`

---

## Terminal memory patch history

### 1. Initial terminal tracing / boolfix

Probe:

```text
terminal_trace_diag_smoke_boolfix
```

Findings:

- Run 1: no terminal memory.
- Run 2: first death stores prototype for action 3.
- Run 3: action 3 gets prototype penalty.
- Context keys had collapsed to all-zero deltas:
  ```text
  a=3|d=0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
  ```

Problem:

- Terminal memory became action-level or near-action-level.

### 2. Trace search confirmed score path is active

Recursive trace search found fields such as:

```text
terminal_prototype_penalty: -1.249...
terminal_prototype_similarity: 0.99995...
terminal_counter_count: small
terminal_risk_scale: near 1.0
terminal_prototype_matched: true
```

This proved the scoring path worked. The bug was context collapse, not dead code.

### 3. Action-level collapse analysis

For `terminal_memory_probe_8run_post_trace_boolfix`:

- Run 3:
  - action 2 penalized 105 times.
  - action 2 would have won without penalty 71 times.
- Run 4:
  - action 2 and action 3 penalized 128 times each.
  - action 3 would have won without penalty 93 times.
- Run 7:
  - action 3 would have won without penalty 140 times.

Problem:

- The terminal memory was suppressing entire action IDs.

### 4. `_terminal_outcome_context_key_is_degenerate` parser fix

A test failed:

```text
test_terminal_context_key_does_not_collapse_when_transition_delta_is_zero
```

The helper considered a nonzero mixed key degenerate.

Fix:

- Removed regex dependency.
- Degenerate only if:
  - no key,
  - no `d=`,
  - payload is empty/unavailable/degenerate,
  - no parseable ints,
  - all parsed ints are zero.

Validation:

```text
degenerate: False
zero_degenerate: True
8 passed, 63 deselected
71 passed in 3.12s
```

### 5. Robust `_terminal_action_context_key`

A later patch temporarily produced:

```text
a=2|d=unavailable
```

Fix:

- Replaced `_terminal_action_context_key` robustly.
- The current key includes:
  - current-state sketch,
  - successor-state sketch,
  - delta sketch.
- Uses 16 bins each, 48 total values.
- Sketching normalizes sampled values and clips bins to `[-4, 4]`.

Manual validation:

```text
key: a=2|d=-2,-1,-1,-1,-1,-1,0,0,0,0,1,1,1,1,1,2,-2,-1,-1,-1,-1,-1,0,0,0,0,1,1,1,1,1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
degenerate: False
prototype_vector: None
```

### 6. Zero-delta prototype guard

Purpose:

- Prevent prototype memory from becoming an absolute-state/action blacklist.
- If successor and current are effectively identical, prototype vectors are dominated by current/successor state, not transition information.

Current behavior:

```python
delta_rms = sqrt(mean(delta_np * delta_np))
if not finite(delta_rms) or delta_rms < 1e-6:
    self._terminal_outcome_degenerate_vector_skips += 1
    return None
```

Effect in latest run:

```text
prototype_count: 0
prototype_count_by_action: {}
last_match: {}
last_counter: {}
```

Prototype memory is currently effectively disabled for zero-delta contexts, which removed the worst prototype blacklist behavior.

### 7. Chunk-summary prototype vector

Current `_terminal_action_context_vector` uses chunk statistics instead of fixed index sampling.

Reason:

- Fixed index sampling collapses when sampled dimensions are stationary.
- Chunk summaries look across the whole latent.

This only matters when `delta_rms` is nonzero enough.

### 8. Exact terminal memory penalties

Current exact penalty logic:

```python
count = entry["count"]
counter_count = entry.get("counter_count", 0)
risk_scale = max(0, (count - counter_count) / max(1, count + counter_count))
return -min(4.0, 1.50 + 0.50 * (count - 1)) * risk_scale
```

Latest behavior:

- Prototype count = 0.
- Exact memory carries the combined penalty.
- Combined penalty grows strongly:
  - Run 2: 102 calls, sum -153.0
  - Run 3: 288 calls, sum -432.0
  - Run 4: 519 calls, sum -778.5
  - Run 8: 1662 calls, sum -2637.0

Risk:

- Exact memory may still become too strong and broad.
- It did not prevent level 1 completion in the latest probe, but it now penalizes many/all actions over time.

### 9. Prototype counterevidence cap needed

Current failing test:

```text
test_terminal_counterevidence_cannot_zero_real_terminal_risk
```

Failure:

```text
assert match["counter_count"] <= 8.0
E assert 25.00000000000022 <= 8.0
```

Cause:

- Prototype counterevidence increments by `0.05`.
- 500 writes produce `25.0`.
- Test expects cap `<= 8.0`.

Patch needed:

- Cap prototype counterevidence at `8.0`.
- Do not necessarily cap exact-key counterevidence the same way.

---

## Current tests and invariants

Run:

```bash
cd ~/ouro_project

./venv/bin/python -m pytest -q   claude_sandbox/test_causal_correctness.py   --import-mode=importlib
```

Targeted run:

```bash
./venv/bin/python -m pytest -q   claude_sandbox/test_causal_correctness.py   --import-mode=importlib   -k "terminal or compact_score_components"
```

Important invariants:

1. Exact terminal context must not become action-only for zero-delta states.
2. `_terminal_action_context_key` must not return `d=unavailable` for valid tensors.
3. Zero/all-zero context keys are degenerate.
4. Mixed current/successor/delta keys with nonzero bins are not degenerate.
5. `_terminal_action_context_vector(... cur, succ=cur ...)` returns `None`.
6. Prototype counterevidence cannot erase real terminal risk.
7. Prototype counterevidence should be capped at `<= 8.0`.
8. `risk_scale` should remain `>= 0.35` for real terminal-risk prototypes.
9. `_compact_score_components` must preserve terminal fields.

---

## Empirical run chronology

### `terminal_trace_diag_smoke_boolfix`

Directory:

```text
claude_sandbox/ablation_event_dumps/terminal_trace_diag_smoke_boolfix
```

Key observations:

- Run 1: no deaths, no prototypes.
- Run 2: death action 3 adjacent, one prototype for action 3.
- Run 3:
  - terminal penalty applied to action 3,
  - counterevidence for action 3 appears,
  - combined penalty calls: 96,
  - combined penalty sum: -117.62,
  - level completions: 0.

Conclusion:

- Terminal memory storage/scoring worked, but context collapsed.

### `terminal_memory_probe_8run_post_trace_boolfix`

Directory:

```text
claude_sandbox/ablation_event_dumps/terminal_memory_probe_8run_post_trace_boolfix
```

Result:

- 0 level completions across 8 runs.
- Mechanism deaths reached 3.
- Prototype count reached 3 by run 8.
- Context diversity was 1 per action.
- Keys looked all-zero:
  ```text
  a=2|d=0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
  ```
- Combined penalty by run 8:
  ```text
  calls: 1419
  sum: -1662.65
  ```

Conclusion:

- Terminal memory became an action blacklist.

### `terminal_memory_probe_8run_post_contextkeyfix`

Directory:

```text
claude_sandbox/ablation_event_dumps/terminal_memory_probe_8run_post_contextkeyfix
```

Result:

- 0 level completions.
- Deaths increased to 6 by run 8.
- Key diversity improved.
- Prototype memory punished actions 1 and 4 broadly.
- Prototype count by run 8:
  ```text
  {'1': 3, '4': 2, '3': 1}
  ```
- Combined penalty by run 8:
  ```text
  calls: 911
  sum: -948.93
  ```

Conclusion:

- Context key fix helped, but prototype memory still overgeneralized due to absolute-state dominance.

### `terminal_memory_probe_8run_post_zerodeltafix`

Directory:

```text
claude_sandbox/ablation_event_dumps/terminal_memory_probe_8run_post_zerodeltafix
```

Result:

- Run 4 completed level 1 at step 116.
- Score became `0.12846101579751995`.
- Solved levels: `[1]`.
- Later runs used `solved_prefix`.
- Later deaths shifted to color 5, often topology frontier.
- Prototype count stayed 0.
- Exact memory carried all combined penalties.

Important run 4 excerpt:

```text
*** LEVEL 1 COMPLETED at step 116 ***
[HunterSeeker] Color 8 track#10 marked EXIT (action=1, unkn=0.24)
Stored level 1 sequence: 116 actions
Reinforcement update (8 extra iterations)...
```

Measurement summary:

- Run 1:
  - exact_size 1
  - prototype_count 0
- Run 4:
  - levels_completed 1
  - exact_size 3
  - combined penalty calls 519
- Run 8:
  - exact_size 4
  - prototype_count 0
  - combined penalty calls 1662
  - combined penalty sum -2637.0

Conclusion:

- Robust exact keys + zero-delta prototype skip is the best current state.
- It produced first level completion in these terminal-memory probes.
- Remaining bottleneck is calibrated exact memory plus topology/frontier hazard avoidance.

---

## Important files and directories

Main files:

```text
claude_sandbox/arc_agent_hunter_seeker_codex.py
claude_sandbox/train_arc_codex.py
claude_sandbox/test_causal_correctness.py
```

Recent dump directories:

```text
claude_sandbox/ablation_event_dumps/terminal_trace_diag_smoke_boolfix
claude_sandbox/ablation_event_dumps/terminal_memory_probe_8run_post_trace_boolfix
claude_sandbox/ablation_event_dumps/terminal_memory_probe_8run_post_contextkeyfix
claude_sandbox/ablation_event_dumps/terminal_memory_probe_8run_post_zerodeltafix
```

Recent checkpoints:

```text
claude_sandbox/checkpoints_running/terminal_memory_probe_8run_post_trace_boolfix.pt
claude_sandbox/checkpoints_running/terminal_memory_probe_8run_post_contextkeyfix.pt
claude_sandbox/checkpoints_running/terminal_memory_probe_8run_post_zerodeltafix.pt
claude_sandbox/checkpoints_arc_pairwise/arc_ls20_runN.pt
```

Backups created by patch scripts include names like:

```text
claude_sandbox/arc_agent_hunter_seeker_codex.py.bak_fix_terminal_context_degenerate_no_regex_1777364000
claude_sandbox/arc_agent_hunter_seeker_codex.py.bak_terminal_proto_zero_delta_1777365388
claude_sandbox/test_causal_correctness.py.bak_terminal_proto_zero_delta_1777365388
claude_sandbox/arc_agent_hunter_seeker_codex.py.bak_terminal_context_key_builder_robust_1777365520
```

---

## Immediate next patch

Apply the prototype counterevidence cap.

Patcher:

```bash
cd ~/ouro_project

cat > /tmp/fix_prototype_counterevidence_cap.py <<'PY'
from pathlib import Path
import time

p = Path("claude_sandbox/arc_agent_hunter_seeker_codex.py")
s = p.read_text(encoding="utf-8")

stamp = int(time.time())
b = p.with_suffix(p.suffix + f".bak_proto_counter_cap_{stamp}")
b.write_text(s, encoding="utf-8")
print("backup:", b)

old = '''                            old_counter = float(best_entry.get("counter_count", 0.0) or 0.0)
                            new_counter = float(max(0.0, old_counter) + proto_increment)
                            best_entry["counter_count"] = new_counter
                            proto_counter_count = new_counter
                            terminal_count = int(best_entry.get("count", 1))
                            wrote_proto = proto_increment > 0.0
'''

new = '''                            old_counter = float(best_entry.get("counter_count", 0.0) or 0.0)

                            # Prototype counterevidence is intentionally capped.
                            # It is fuzzy/nearest-neighbor evidence, so it may attenuate
                            # but must not grow without bound and erase real terminal risk.
                            proto_counter_cap = 8.0
                            new_counter = float(
                                min(proto_counter_cap, max(0.0, old_counter) + proto_increment)
                            )

                            best_entry["counter_count"] = new_counter
                            proto_counter_count = new_counter
                            terminal_count = int(best_entry.get("count", 1))
                            wrote_proto = (new_counter > max(0.0, old_counter) + 1e-12)
'''

if old not in s:
    raise SystemExit("ABORT: expected prototype counterevidence block not found")

s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")
print("patched prototype counterevidence cap")
PY

./venv/bin/python /tmp/fix_prototype_counterevidence_cap.py

./venv/bin/python -m pytest -q   claude_sandbox/test_causal_correctness.py   --import-mode=importlib   -k "terminal or compact_score_components"

./venv/bin/python -m pytest -q   claude_sandbox/test_causal_correctness.py   --import-mode=importlib
```

If the exact block is not found, inspect `_remember_terminal_outcome_counterevidence` around the prototype update and manually cap only the prototype `counter_count`.

---

## Next empirical experiment after patch

Use a fresh name:

```text
terminal_memory_probe_8run_post_countercap
```

Command:

```bash
cd ~/ouro_project

rm -rf claude_sandbox/ablation_event_dumps/terminal_memory_probe_8run_post_countercap
rm -f claude_sandbox/checkpoints_running/terminal_memory_probe_8run_post_countercap.pt

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./venv/bin/python -m claude_sandbox.train_arc_codex   --games ls20   --n_runs 8   --max_steps 160   --agent hunter_seeker   --backbone_mode ouro   --use_loop_pooler on   --loop_pooler_kind gru   --self_model_mode off   --cortex_monitor_mode off   --anchor_train_every 25   --anchor_coefficient 0.1   --anchor_batch_size 1   --unfreeze_encoder_for_anchor   --running_checkpoint claude_sandbox/checkpoints_running/terminal_memory_probe_8run_post_countercap.pt   --dump_events_dir claude_sandbox/ablation_event_dumps/terminal_memory_probe_8run_post_countercap
```

Compare to `post_zerodeltafix`:

- Did it complete level 1 again?
- Which run/step?
- Did deaths shift from color 3 to color 5 again?
- Did exact penalty calls explode across all actions?
- Did prototype count stay 0 or did nonzero-delta prototypes appear?
- Did solved_prefix start being used after completion?
- Did GRU gate remain 0?
- Did anchor losses stay sane?
- Did `combined_penalty_by_action` become broad action suppression again?
- Did failure count improve relative to 6 mechanism failures?

---

## Strong recommendation for next algorithmic patch

After counterevidence cap, do **not** immediately add terminal penalty strength.

The latest run already shows strong exact penalties. Consider making exact terminal memory more conservative if post-countercap still over-suppresses:

Current:

```python
return -min(4.0, 1.50 + 0.50 * (count - 1)) * risk_scale
```

Safer option:

```python
base = min(2.0, 0.50 + 0.50 * max(0, count - 1))
return -base * risk_scale
```

Rationale:

- A single death should create caution, not a hard reroute.
- Repeated terminal evidence should matter.
- Current exact memory can rapidly create `-1.5` or worse penalties.

Potential exact counterevidence rule:

- Add small exact counterevidence only if the same exact context key is followed by survival for several steps or a positive event.
- Do not let immediate one-step nonterminal continuation erase terminal risk.

---

## Current known issues

1. **Exact memory may be too punitive.**
   - Latest run 8 combined penalty: 1662 calls, sum -2637.0.
   - Eventually penalizes all actions.

2. **Prototype memory is currently mostly disabled.**
   - `prototype_count: 0` in latest probe.
   - This avoids prototype blacklists but loses fuzzy generalization.

3. **Topology frontier deaths remain.**
   - Especially color 5 after level 1.

4. **GRU gate appears stuck at 0.**
   - Needs separate investigation.

5. **No ranker/prior/spatial updates in current probes.**
   - These experiments do not validate full learning.

6. **Online buffer / sibling crowding remains a known concern.**
   - Sibling counts rise steadily.
   - Historical concern: auxiliary sibling entries may crowd real online transitions.

7. **Environment fragility.**
   - Use project venv.
   - Do not use system python for tests.
   - Do not update Transformers/Torch/CUDA casually.
   - Keep Ouro offline flags for runs.

---

## Coding style constraints

- Do not rewrite whole files unless explicitly requested.
- Prefer targeted patchers with backups.
- Preserve all existing features unless explicitly removing something.
- Run tests after every patch.
- Use `./venv/bin/python`, not system Python.
- Do not update Transformers.
- Do not change Torch/CUDA/Ouro dependencies casually.
- Keep diagnostics.
- Be blunt and research-partner style.

---

## Handoff checklist

Next collaborator should:

1. Apply prototype counterevidence cap.
2. Run targeted terminal tests.
3. Run full `test_causal_correctness.py`.
4. Run `terminal_memory_probe_8run_post_countercap`.
5. Summarize:
   - level completions,
   - failures,
   - exact/prototype penalties,
   - action suppression,
   - topology deaths,
   - key diversity.
6. If completion disappears, inspect exact penalty breadth and soften exact penalty.
7. If completion persists but later deaths remain, move to topology-frontier avoidance.
8. Do not re-enable broad prototype memory until it is transition-dominant instead of absolute-state-dominant.
9. Keep all terminal diagnostics.

---

## Minimal mental model

The agent currently behaves like this:

1. It explores `ls20`.
2. It detects mechanics/events and classifies colors/objects.
3. It dies and stores terminal context.
4. Terminal memory modifies candidate scores.
5. Earlier, this memory blacklisted actions because context collapsed.
6. Now, exact context is more specific, and prototype memory skips zero-delta cases.
7. This allowed one level-1 completion.
8. The agent then uses solved_prefix and carries solved level `[1]`.
9. It still dies from later hazards/topology frontiers.
10. The current challenge is calibrating terminal memory so it prevents repeated deaths without suppressing useful actions.

---

## One-line current status

Hunter Seeker is no longer obviously blocked by terminal-memory action blacklisting; after robust exact context keys and zero-delta prototype suppression it can solve `ls20` level 1 once, but exact terminal penalties are now very strong and topology-frontier deaths remain the next empirical bottleneck.
