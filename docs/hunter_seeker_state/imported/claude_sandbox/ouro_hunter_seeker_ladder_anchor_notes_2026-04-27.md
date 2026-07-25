<!-- Imported from `claude_sandbox/ouro_hunter_seeker_ladder_anchor_notes_2026-04-27.md` during Hunter-Seeker state consolidation on 2026-05-14. -->
<!-- Original SHA256: 4176d9190dcbdfe9a3047182922c49a95538a54e303c5f7fae386e87c9fbee84; original line count: 416. -->

# Ouro / Hunter Seeker — Pre-Ladder Audit & Anchor-Fallback Notes

_Last updated: 2026-04-27_

This file consolidates the current ladder/audit state, fixes already shipped, deferred cleanup work, quick-ladder findings, and the validated terminal-failure anchor fallback.

---

## Current status

The quick ladder now runs through the relevant ablation steps and the `claude_sandbox` test suite is passing.

Latest validation:

```text
Targeted anchor/terminal tests: 8 passed, 36 deselected
Full claude_sandbox suite: 203 passed
```

The most important new result is that anchor training now has a real terminal/death/failure fallback path. This path successfully backpropagated from a real environment-derived terminal failure pair with smoke pairs disabled.

---

## Immediate pre-ladder patch items — shipped

These were the original ladder blockers / papercuts and are now considered shipped.

1. **Step 0 checkpoint path fixed**
   - Runner now resolves the historical baseline from root-level:
     `checkpoints_running/sprint4_encoder_reverted.pt`
   - Old bad path:
     `claude_sandbox/checkpoints_running/sprint4_encoder_reverted.pt`
   - This restored the historical-floor reference.

2. **Quick mode now exercises anchor-related code**
   - Original issue: `--quick` capped at 80 steps while anchor cadence was hard-coded at 100, so anchor counters stayed zero.
   - Current state: quick mode has explicit anchor diagnostics and, when needed, diagnostic smoke mode for the quick runner only.

3. **Runner event-dump comment fixed**
   - Comment now matches reality:
     events land under `--dump_events_dir/<game>/`, not `--checkpoint_dir/<game>/`.

4. **Click softmax finite guard added**
   - `ActionHead` already guarded action logits.
   - Click logits now also guard against NaN/Inf softmax output before multinomial sampling.

5. **Comparator alarm/doc alignment**
   - Comparator doc/code now distinguish mean-attention alarms from argmax-share alarms.
   - Argmax-share concentration alarms are now present and useful.

6. **Empty action defensive fallback**
   - `np.random.choice(list(available_actions))` no longer assumes non-empty action sets.
   - ARC adapter currently never returns empty actions, but this avoids a future adapter contract leak.

---

## Keep permanently

These changes are worth keeping rather than treating as temporary ladder hacks.

1. **`sm_grad_active = False` initializer**
   - Prevents `UnboundLocalError` in `_train_ranker`.
   - Keep as a real correctness guard.

2. **Generic loop-pooler gate logging**
   - Avoids assuming every pooler has `gru_gate`.
   - AttnRes pooler no longer crashes when summary/logging asks for the gate.

3. **Click softmax finite guard**
   - Same crash class as action softmax.
   - Keep.

4. **Step 0 baseline checkpoint path resolution**
   - Real runner correctness fix.
   - Keep.

5. **Anchor quick-mode diagnostics**
   - Keep the ability to prove anchor code paths in short runs.
   - But smoke-pair outputs must stay clearly marked as non-scientific.

6. **cuDNN-disabled anchor evaluator forward**
   - Prevents the frozen evaluator GRU backward error:
     `cudnn RNN backward can only be called in training mode`.
   - Keep.

7. **Explicit anchor unfreeze flag**
   - `--unfreeze_encoder_for_anchor` is the canonical flag.
   - `--unfreeze_encoder_after_partial_load` remains as a legacy alias only.
   - Anchor warning now makes frozen-encoder no-op obvious.

8. **Terminal-failure anchor fallback**
   - Validated with smoke disabled.
   - This is now a real sparse supervision path, not just a smoke-test hook.

---

## Keep for now, redesign later

These are acceptable for the current ladder, but should eventually be cleaned up.

1. **Pending-event graph guard / ordered self-model loss consumption**
   - A guard was needed to avoid graph/version issues.
   - Longer-term, self-model losses should be consumed in a cleaner ordered way so stale graph references do not exist.

2. **`--unfreeze_encoder_after_partial_load` legacy alias**
   - Keep only for backwards compatibility.
   - New scripts should use `--unfreeze_encoder_for_anchor`.

3. **Anchor adaptive batch ceiling currently often collapses to 1**
   - On the current 11.47 GiB GPU setup, anchor batch sizes of 4 and 2 OOMed in some ladder configurations.
   - Batch size 1 works.
   - This is acceptable for validation, but throughput/quality would improve if we reduce memory pressure.

---

## Terminal-failure anchor fallback — validated

Status: **validated and keep**.

A v3 patch added a real terminal/death/failure fallback path for anchor training. This is no longer only a quick smoke mechanism. When ordinary ranking pairs are unavailable, Hunter Seeker can now mark the most recent real environment transition after `on_game_over()` as terminal/death/failure metadata, then run one post-terminal anchor retry using a real nonterminal-vs-terminal pair.

Validation run:

```text
Game: ls20
max_steps: 160
anchor_pair_smoke: disabled
anchor_batch_size: 1
anchor_train_every: 25
failure: directional death
failure_type: topology
terminal retry loss: 0.0445
```

Measurement summary:

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
  "success_streak": 1,
  "buffer_len": 154,
  "trusted_buffer_len": 0,
  "auxiliary_buffer_len": 308,
  "candidate_pair_count": 154,
  "last_skip_reason": null,
  "smoke_attempts": 0,
  "terminal_fallback_attempts": 7,
  "terminal_fallback_successes": 1,
  "last_pair_source": "terminal_failure_fallback",
  "terminal_failure_pair_count": 1
}
```

Failure counts:

```json
{
  "topology": 1
}
```

Interpretation:

- `smoke_attempts = 0`, so this was not smoke supervision.
- `terminal_fallback_successes = 1`, so a real terminal/death fallback pair was used.
- `last_pair_source = "terminal_failure_fallback"`, confirming the real fallback path.
- `loss_ema > 0`, confirming anchor loss actually backpropagated.
- The six skipped attempts are expected: scheduled anchor calls occurred before any terminal/death transition existed.
- This should be treated as a sparse but legitimate real supervision channel, complementary to normal ranking pairs.

Remaining caveat:

This fallback only learns from terminal/death/failure outcomes, so it is sparse. It does not replace normal ranking-pair supervision. It does, however, solve the quick-ladder problem where anchor training could be structurally exercised without relying on arbitrary smoke pairs.

---

## Quick-ladder observations

Latest quick-ladder comparison showed:

```text
levels_completed_total: 0 across all steps
```

That is not shocking for an 80-step single-game quick run. Treat it as a smoke/instrumentation pass, not a scientific performance result.

Useful signals from quick ladder:

1. **Anchor instrumentation now distinguishes smoke vs real fallback**
   - Smoke path should only be used for quick diagnostic proof.
   - Real terminal fallback is now validated separately.

2. **AttnRes ranker becomes active**
   - AttnRes configurations showed ranker losses around ~1.2–1.4 in some quick runs.
   - This means the ranker path is alive, but not necessarily useful yet.

3. **AttnRes argmax alarms still fire**
   - Attention means look nearly uniform, but argmax winner can concentrate on one loop.
   - This suggests the attention distribution may be too flat/noisy or the argmax diagnostic is too brittle with tiny call counts.
   - Do not overinterpret from a one-game quick run.

4. **Self-model gradients are visible but sometimes tiny**
   - Passive and inject modes report self-model loss/grad metrics.
   - Full-system quick runs can show zeroed/near-zero self-model norms in some measurements.
   - Needs longer run or direct self-model probe before judging.

5. **Topology remains a real failure type**
   - The validated anchor fallback used a topology death.
   - This reinforces Sprint 5 topology as the right next architectural target.

---

## Known issue: anchor batch size and VRAM

Observed:

```text
[Anchor-Train] OOM with batch_size=4
[Anchor-Train] OOM with batch_size=2
adaptive batch falls to 1
```

Current conclusion:

- `anchor_batch_size=1` is viable.
- The current GPU memory headroom is extremely tight with Ouro loaded.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` helps fragmentation but does not magically create enough headroom.
- Higher batch sizes probably require structural memory reductions.

Possible ways to allow larger anchor batches later:

1. **Run anchor less often but accumulate gradients**
   - Keep microbatch size 1.
   - Accumulate N anchor microsteps before optimizer step.
   - This gives effective batch >1 without peak-memory spike.

2. **Move frozen evaluator to CPU**
   - If evaluator forward is not too slow, CPU evaluator can save GPU memory.
   - This may be acceptable for sparse anchor calls.

3. **Use smaller anchor frame encoding path**
   - Avoid full Ouro/loop path where not strictly needed.
   - Distill terminal preference into a smaller encoder-side auxiliary head.

4. **Temporarily disable nonessential heads during anchor step**
   - Ensure nextframe/spatial/self-model graphs are not retained.
   - Use aggressive `torch.no_grad()` for every module not receiving anchor gradient.

5. **Explicit cleanup before anchor**
   - `torch.cuda.empty_cache()` may help fragmentation, but should not be relied on as a core solution.
   - Better to reduce live tensors and retained graph references.

Recommended short-term choice:

- Keep `anchor_batch_size=1`.
- Do not chase batch size before extracting a real comparison table and doing the next topology probe.

---

## Deferred architecture cleanup

These are real issues but should not block the current ladder/quick validation.

1. **Rename `info["gru_gate"]` to generic `info["loop_pooler_gate"]`**
   - Current generic gate handling works.
   - Cleanup naming later to avoid GRU-specific assumptions.

2. **Move action legality to `ActionAdapter.safe_action_indices`**
   - Avoid scattered action-set guards.
   - Especially important for future non-ARC adapters.

3. **Remove ARC progress semantics from base agent**
   - Current `levels_completed` / `_QUALITY_LEVEL_SCALE` is an architecture leak.
   - Long-term adapter API should expose:
     - `progress_value`
     - `terminal_state`
     - `transition_quality`
   - This matters for no-game-specific-hacks discipline.

4. **Comparator score-components aggregation**
   - `score_components` are per-step traces, not in `measurement_summary`.
   - Needs harness support first.

5. **RUN_ID timestamped output dirs**
   - Nice quality-of-life improvement.
   - Prevents stale `ladder_step_*.pt` and dump reuse.
   - Not blocking if manual `rm` is done before runs.

6. **ColorPriorTable / ActionHead / normalize_clicks naming hygiene**
   - Cleanup only.
   - Not a current correctness blocker.

7. **Micro-optimizations**
   - Vectorize padding.
   - Cache enum maps.
   - Stable hashing.
   - Not blocking.

---

## Suggested next steps

### Next quick action: extract the comparison table cleanly

Run the comparator after any ladder/probe:

```bash
cd ~/ouro_project

./venv/bin/python -m claude_sandbox.compare_ladder_summaries \
  claude_sandbox/ablation_event_dumps/step_* \
  --alarms \
  --json-out /tmp/ladder_quick_summary.json
```

Then preserve the output in the notes. This is the fastest way to avoid making decisions from raw logs.

### Next scientific action: run a tiny topology-focused probe

Because the validated terminal fallback produced `failure_type=topology`, the fastest useful next experiment is not more anchor plumbing. It is a topology probe.

Goal:

- Find whether deaths/stalls are caused by reachability/path topology or object mechanism confusion.
- Measure whether Hunter Seeker can learn “this path/action region is terminal” after one death.

Suggested short probe:

```bash
cd ~/ouro_project

rm -rf claude_sandbox/ablation_event_dumps/topology_probe_ls20
rm -f claude_sandbox/checkpoints_running/topology_probe_ls20.pt

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

Then inspect:

```bash
python3 - <<'PY'
import json
from pathlib import Path

root = Path("claude_sandbox/ablation_event_dumps/topology_probe_ls20")
for p in sorted(root.glob("*/measurement_run_*.json")):
    data = json.loads(p.read_text())
    ms = data.get("measurement_summary", {})
    print("\n", p)
    print("failure_counts:", ms.get("failure_counts"))
    print("anchor:", ms.get("anchor"))
    print("reach:", {
        "calls": ms.get("reach_calls_total"),
        "informative": ms.get("reach_calls_informative"),
        "rate": ms.get("reach_informative_rate"),
    })
PY
```

Good signs:

- `terminal_fallback_successes > 0`
- repeated topology failures decrease or become more specifically classified
- reachability metrics are nonzero and informative
- event log contains death events with correlated preceding movement/contact events

### Next code action after that: topology memory

If the topology probe confirms repeated topology deaths, add a tiny topology-memory layer before full Sprint 5:

- remember `(game_id, local avatar region/action, failure_type=topology)` as bad
- downweight repeating the same terminal direction/region
- log `topology_avoidance_hits`
- keep it tiny and measurable

Do **not** jump into a huge topology architecture rewrite until this probe says the failure is repeatable and measurable.

---

## Current recommendation

Do this order:

1. Save this file into the repo.
2. Run full `claude_sandbox` tests once after any further edit.
3. Run the 2-run topology probe above.
4. Extract the small JSON summary.
5. If topology repeats, implement tiny topology memory.
6. Only then run a broader ladder again.

The important win from today is that the anchor system is no longer fake: it has a validated real terminal-failure supervision path.
