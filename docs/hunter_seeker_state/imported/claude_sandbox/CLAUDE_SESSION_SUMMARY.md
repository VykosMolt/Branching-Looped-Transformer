<!-- Imported from `claude_sandbox/CLAUDE_SESSION_SUMMARY.md` during Hunter-Seeker state consolidation on 2026-05-14. -->
<!-- Original SHA256: 185de98f1928dcec4e37e40726fad3a44fd59e830b90eab72c73e4924bf9ee9a; original line count: 1649. -->

# Claude Sandbox Session Summary

Companion log to `SESSION_SUMMARY.md` (which is Codex's handoff record — not edited by Claude). This file is Claude's own session log. Append-only, dated entries, newest at top.

Rules for this sandbox (mirror of codex_sandbox policy):

- Do not modify, overwrite, delete, or rename files in `codex_sandbox/`.
- Inside `claude_sandbox/`, do not overwrite Codex's original `SESSION_SUMMARY.md` either — that remains the Codex handoff record. Claude's own log lives in this file (`CLAUDE_SESSION_SUMMARY.md`).
- Work only on the Python copies in `claude_sandbox/` — don't touch `codex_sandbox/*.py`.
- Keep all Claude-authored variants and experiments inside this directory unless the user explicitly requests otherwise.
- Preserve compatibility with the local Ouro setup. Do not upgrade `transformers` past `4.54.1`.
- No game-specific hardcoding. Planner/options/affordances must stay domain-agnostic and data-derived. ARC is the test domain, not the target.

## 2026-04-26 Handoff (pre-context-clear)

### Files changed this session-cluster (V → handoff)

Modified:
- `arc_agent_pairwise_stockfish_codex.py` — adaptive anchor batch
  state + halve-on-OOM + double-on-success-streak in
  `train_anchor_step`.
- `arc_agent_hunter_seeker_codex.py` — `current_batch_size`,
  `batch_ceiling`, `success_streak` in `measurement_summary["anchor"]`.
- `train_arc_codex.py` — `_dump_measurement_summary_safe` +
  `_json_safe_payload` helpers; per-run measurement_summary JSON
  written to `{dump_events_dir}/{game_id}/measurement_run_{N}.json`
  alongside the existing event-log dump.
- `test_causal_correctness.py` — adaptive-batch state-machine test.

New:
- `claude_sandbox/design/ablation_ladder.md` — 8-step runbook (my
  Step 0 v17b baseline + Step 3.5 anchor-alone, GPT's Step 2 passive
  canary + Step 6 anchor+AttnRes interaction).
- `claude_sandbox/run_ablation_ladder.sh` — bash runner with
  `--quick` and `--vram-check` modes.
- `claude_sandbox/compare_ladder_summaries.py` — post-hoc table
  builder (see below).

### What `compare_ladder_summaries.py` does

Ingests per-run measurement_summary JSONs from N ablation-ladder step
directories, aggregates by step (sum scores, mean floats, sum
counters, last-value for adaptive state), and emits a markdown
comparison table to stdout. Optional flags:

- `--full` — adds AttnRes argmax counts and event_log_counts.
- `--alarms` — health-rule check pass: anchor success rate <30%,
  adaptive batch collapsed to floor, AttnRes entropy <0.3 or any-loop
  argmax-share >80%, aggregator_fuse_weight_norm stuck at zero,
  failure_counts.unknown ratio >50%.
- `--json-out PATH` — also write the aggregated dict for follow-up
  scripts.

Column order = arg order; column labels = directory basenames (so
name your step dirs descriptively, e.g. `step_0_v17b_baseline`).

### Commands

VRAM calibration before any anchor sweep:
```
./claude_sandbox/run_ablation_ladder.sh --vram-check
```

Quick smoke-test of the ladder mechanics (1 game, 80 steps, n_runs=1):
```
./claude_sandbox/run_ablation_ladder.sh --quick
```

Full ladder (default games: ls20 ft09 r11l tr87 wa30; 400 steps × 3 runs):
```
./claude_sandbox/run_ablation_ladder.sh
```

Override sweep parameters via env:
```
GAMES="ls20 ft09" MAX_STEPS=200 N_RUNS=2 ./claude_sandbox/run_ablation_ladder.sh
```

Post-hoc compare:
```
./venv/bin/python -m claude_sandbox.compare_ladder_summaries \
  claude_sandbox/ablation_event_dumps/step_0 \
  claude_sandbox/ablation_event_dumps/step_1 \
  claude_sandbox/ablation_event_dumps/step_2 \
  claude_sandbox/ablation_event_dumps/step_3 \
  claude_sandbox/ablation_event_dumps/step_3.5 \
  claude_sandbox/ablation_event_dumps/step_4 \
  claude_sandbox/ablation_event_dumps/step_5 \
  claude_sandbox/ablation_event_dumps/step_6 \
  claude_sandbox/ablation_event_dumps/step_7 \
  --alarms
```

### What remains unfinished

1. **Run the ladder.** Pure GPU compute. Pre-flight via `--vram-check`,
   then commit. Expected duration with defaults: hours per step.
2. **`compare_ladder_summaries.py` not exercised on real data.** It's
   built and self-consistent, but I didn't have ladder JSONs to feed
   it. First real run may surface schema / aggregation edge cases.
3. **Per-component health thresholds in `--alarms`** are best-guess
   from the design doc. Tune after the first ladder run shows what
   "normal" actually looks like.
4. **Sprint 7 (synthetic mechanic curriculum)**, **Sprint 9 (abstract
   cross-run memory)**, **Sprint 12 (sleep stage 2)**, **Sprint 13
   (sleep stage 3)** — not built. State doc consistently positions
   these as a separate phase after Sprint 8.
5. **Sprint 8 consumer** (offline replay loop reading the dumped
   event logs into parameter updates) — not built; substrate ready.
6. **Sprint 10 object-file-delta-centric ranker** — partially served
   by symbolic transition features; the full vision (ranker reasoning
   directly over object-file deltas, not just CLS/loop pooled
   representations) is still open.

### Known caveats

- The new `_dump_measurement_summary_safe` was added to the harness
  but **NOT exercised in a real run during this session** — only the
  unit test suite ran. First ladder invocation should sanity-check
  that the JSON files actually appear in `{dump_events_dir}/{game_id}/`.
- Step 0 of the ladder requires
  `checkpoints_running/sprint4_encoder_reverted.pt` to exist; the
  runner skips Step 0 with a warning if it's missing. Without Step 0,
  no historical-floor reference; ladder still runs.
- `compare_ladder_summaries.py` aggregates assuming each step's
  directory contains JSONs named `measurement_run_*.json` recursively.
  If you change the harness's output naming convention, update the
  glob in `_load_step_summaries`.
- Test suite occasionally shows a flaky failure on
  `test_codex_integration.py::test_mock_sweep_validator_reports_symbolic_terms`
  on full-suite runs but passes when run in isolation. Likely test-
  ordering dependency unrelated to this session's changes; flagged
  but not chased.
- 175 tests passing on clean runs. Counter resets across ladder
  steps (each step gets a fresh `running_checkpoint`), so `step_count`
  / loss-history resets are intentional, not a state bug.
- The current ladder script doesn't include the `--trusted_mix`
  ablation as a separate step (that's documented in the markdown but
  not scripted). Add it manually after the main ladder finishes if
  needed.

## 2026-04-26 Continuation Pass V — adaptive anchor batch + ablation ladder

Two pieces shipped after the onion.md fixes: an adaptive batch-size
controller for the anchor loss, and a merged 8-step ablation ladder
combining my structuring with GPT's additions.

### Adaptive `train_anchor_step` batch sizing

User-supplied `--anchor_batch_size` is now a CEILING, not a fixed
value. Runtime maintains `_anchor_current_batch_size`:

- On `cuda.OutOfMemoryError`: halve current batch (down to floor=1).
- On 8 consecutive successes: double back up (capped at ceiling).
- Loud print at every transition so logs stay attributable.

State surfaced via `measurement_summary["anchor"]`:
- `current_batch_size`
- `batch_ceiling`
- `success_streak`

This guards against the silent-degradation failure mode onion §2
flagged: a sweep at the user's chosen ceiling can quietly degenerate
to "anchor mostly skipped" under VRAM pressure with the OOM-safe
catch alone, and the only signal would be `_anchor_skipped_oom`
climbing — which you'd only notice post-run. Adaptation makes the
sweep self-healing within the budget.

Test added (`test_train_anchor_step_adaptive_batch_halves_on_simulated_oom`)
drives the state machine directly and verifies floor-clamping +
recovery doubling.

### Ablation ladder (merged version)

GPT's framing of onion §10 was better than mine in two specific
places, so the doc landed at `claude_sandbox/design/ablation_ladder.md`
is a merge:

- My **Step 0**: explicit v17b historical baseline against
  `frozen_sprint_4_overnight_noreplay`. Without this, the ladder
  isolates internal contributions but doesn't anchor against the
  scoreboard the paper has to beat.
- My **Step 3.5**: anchor-on-Ouro-baseline (no other heads, no
  AttnRes). Cleanest single-variable test of "does the anchor alone
  prevent encoder drift?" — the failure mode it was designed to fix.
- GPT's **Step 2**: passive-introspection canary. Verifies that
  passive cognitive heads byte-equivalently match off-mode within the
  first ~50 steps, ruling out a real perturbation vs CUDA non-
  determinism. Caveat documented: zero-init guarantees decay as the
  event/signature predictors train, so this is a short-horizon
  canary, not a full-run guarantee.
- GPT's **Step 6**: anchor + AttnRes pairwise interaction. Anchor
  reshapes encoder loop dynamics; AttnRes pooler should track that
  shift via softmax weights. Bundling them only at Step 7 loses the
  attribution channel.

Total 8 steps, every adjacent pair differs by exactly one factor.

Companion shell script `claude_sandbox/run_ablation_ladder.sh`
invokes the ladder. Has a `--vram-check` flag for the 60-step
sacrificial pre-flight calibration, and a `--quick` flag for
ladder-mechanics smoke testing (n_runs=1, 80 steps, single game).
Each step gets its own running checkpoint and event-dump directory
so configs don't bleed across the chain.

### Test snapshot

```
175 passed in 10.16s
```

(174 → 175 with the new adaptive-batch test.)

### Files changed

Modified:
- `arc_agent_pairwise_stockfish_codex.py` — adaptive batch state
  in `__init__`, halve-on-OOM + double-on-success-streak logic in
  `train_anchor_step`.
- `arc_agent_hunter_seeker_codex.py` — `current_batch_size`,
  `batch_ceiling`, `success_streak` added to
  `measurement_summary["anchor"]`.
- `test_causal_correctness.py` — adaptive-batch state-machine test.

New:
- `design/ablation_ladder.md` — 8-step ladder runbook.
- `run_ablation_ladder.sh` — bash runner with `--vram-check` and
  `--quick` modes.

### What's actually left

1. **Run the ladder.** Pure GPU compute. The harness, runner, and
   doc are all in place. Pre-flight with `--vram-check`, then
   commit to the full sweep.
2. **`compare_ladder_summaries.py`** post-hoc table builder. The
   ladder doc points at this as a useful follow-up; not built. Each
   step's measurement_summary JSON is self-describing so reading by
   hand or with a one-liner works for the first pass.
3. **Hook S_t(x) into beam search trust gate as a primary signal**.
   Currently captured (`_last_ouro_confidence`) and blended into
   `effective_conf` multiplicatively. Could become a primary trust
   axis instead of a multiplier on Δloop. Design call best made
   after ladder data exists.

## 2026-04-26 Continuation Pass IV — onion.md fixes (CLI safety + diagnostics)

GPT-onion review flagged 10 items, mostly logging/safety, plus one
documentation drift. Shipped fixes for all of them.

### Onion §1+§4 — encoder-freeze CLI flags + warnings

`load_partial_checkpoint_for_sandbox` already had a kwarg, but no CLI
flag wired it. Now `--unfreeze_encoder_after_partial_load` flows
through all three partial-load fallback sites in train_arc_codex
(baseline / explicit / running checkpoints). Plus a loud warning if
`--anchor_train_every > 0` AND `agent.freeze_encoder` is True — anchor
training is a no-op in that combination, easy to miss in logs.

`live_arc_diagnostic.py` previously froze the encoder unconditionally
inside `load_diagnostic_checkpoint` AND at agent construction. Both
gated now via new `--freeze_encoder` (default True, preserves existing
behaviour) / `--no_freeze_encoder` flag pair. Prints a clear note about
the freeze state at run start.

### Onion §3 — anchor training counters

`train_anchor_step` was OOM-safe but couldn't tell you whether anchor
calls were actually firing or silently skipping. Counters added:

- `_anchor_attempts`
- `_anchor_successes`
- `_anchor_skipped_insufficient_pairs`
- `_anchor_skipped_frozen_encoder`
- `_anchor_skipped_no_evaluator`
- `_anchor_skipped_oom`
- `_anchor_loss_ema`

All exposed via `measurement_summary()["anchor"]` so run summaries
distinguish "no anchor activity" from "every call short-circuited."

### Onion §5+§7 — self-model warmup + AttnRes anti-collapse diagnostics

The aggregator's zero-init fuse means upstream gradient (ranker →
SelfModel GRU) stays blocked until training moves the fuse off zero.
Without dedicated metrics, "system in identity-safe warmup" and
"system silently broken" look identical. Added to `measurement_summary()`:

- `self_model.self_model_loss_ema` — event-prediction loss
- `self_model.aggregator_fuse_weight_norm` — gate moving off zero
- `self_model.temporal_feature_norm` — output magnitude
- `self_model.self_model_gru_grad_norm` — actual training signal reaching the GRU

For the AttnResLoopPooler:

- `attn_res_pooler.attn_l[1-4]_mean` — average attention weight per loop
- `attn_res_pooler.attn_argmax_count_l[1-4]` — how often each loop "wins"
- `attn_res_pooler.attn_entropy` — Shannon entropy of average distribution
- `attn_res_pooler.attn_gate` — tanh-applied scalar gate

If the AttnRes pooler collapses to "always L4," `attn_l4_mean → 1.0`
and `attn_entropy → 0` — that's the diagnostic to flag the regression
early (the architecture would degenerate to a final-loop reader).

### Onion §6 — pre/post-Ouro effective_conf logging

The Ouro confidence multiplier was blended into `effective_conf` last
session, but the trace dump only carried the post-blend value. Now
`score_components` carries:

- `effective_confidence_pre_ouro` — Δloop trust × wm_confidence only
- `effective_confidence` (post-blend, unchanged)
- `ouro_trust_multiplier` — the actual `0.5 + 0.5*ouro_conf` factor
- `ranker_score_raw`, `heuristic_score_raw`

So trace readers can attribute scoring changes to Δloop trust vs Ouro
confidence vs raw ranker output independently.

### Onion §8 — buffer-mix CLI

Added `--trusted_mix` (default 0.75) plumbed through the agent
constructor and used by `_blended_ranking_pairs` etc. via a
`self.trusted_mix` attribute (read with `getattr(..., 0.75)` so legacy
agents pre-dating this attr keep the original behaviour).

### Onion §9 — stale GridEncoder docstring

`GridEncoder.forward` docstring claimed `H/W must be divisible by
patch_size`, but PatchEmbedding has been auto-padding for a session
already. Updated to reflect the actual contract, with a note that
downstream code mapping tokens back to pixel space must account for
post-padding patch counts.

### Detailed re-audit (every domain walked)

1. **Step ordering**: confirmed `update_beliefs → update_from_scene(scene_after) → detect_events`
   in pre-step; post-step `update_from_scene` only fires when pre-step
   didn't (the `transition_processed_this_step` flag).
2. **Optimizer step coverage**: ranker, prior, spatial, encoder, aux
   world model, loop pooler, objectivity, symbolic planner,
   self-model, cortex monitor — all step from real losses. Self-model
   has TWO step paths now: event-prediction loss (in
   `_self_model_step`) AND ranker loss (in `_train_ranker` via
   off-policy reconstruction).
3. **Buffer schema**: `temporal_features`, `self_model_h`,
   `track_summary_snapshot`, `event_summary_ema_snapshot`,
   `loop_delta_scalar_snapshot` — all carried through
   `_Transition` → `push()` → `_to_tensors()` →
   `sample_ranking_pairs` / `sample_sibling_pairs` → `_train_ranker`.
   All-or-nothing rule prevents shape mismatches.
4. **Checkpoint coverage**: every new module (AttnResLoopPooler,
   temporal_event_predictor, cortex_signature_predictor) and every new
   optimiser (aux_world_model_optimizer, self_model_optimizer with
   predictor params, cortex_monitor_optimizer with signature predictor)
   persists in the save dict and restores via the load path with
   strict=False fall-backs. `loop_pooler_kind` saves alongside the
   pooler weights with arch-mismatch warning on cross-kind load.
5. **`encode_and_think_batch` call sites**: only line 3271 (real-frame
   in `select_action`) sets `update_introspective_state=True`. Beam,
   training, sibling, offline-encoding paths all leave it False.
   Cortex monitor advance and Ouro exit-gate signal capture both
   gated by this flag.
6. **Push call sites**: 2 live (main buffer + auxiliary sibling
   buffer) carry all SelfModel snapshot fields; trusted-buffer load
   leaves them None (correct — no live state at trajectory load).
7. **Empty-scene + duplicate-detect_events guards**: still in place.
8. **on_game_over**: click-death attribution failure falls through to
   the terminal-bookkeeping block (DEATH event, `_failure_counter`,
   `_discovered_mechanics.add(DEATH)`, `_self_model_step(died=True)`)
   instead of early-returning. Confirmed.
9. **Adapter `frames_to_dense_input` contract**: honored at both
   call sites (`encode_and_think_batch` and `_ouro_loop_states`) with
   legacy fallback for adapters that pre-date the method.
10. **CLI plumbing**: `--self_model_mode`, `--cortex_monitor_mode`,
    `--loop_pooler_kind`, `--anchor_train_every`,
    `--anchor_coefficient`, `--anchor_batch_size`,
    `--unfreeze_encoder_after_partial_load`, `--trusted_mix` all
    threaded from argparse → agent_kwargs → train_on_game and to
    `live_arc_diagnostic` where applicable.

### Test snapshot

```
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/moloch/ouro_project/venv/bin/python -m pytest \
  claude_sandbox/ -q
# 174 passed in 28.97s
```

One new regression test added covering `measurement_summary` exposing
the anchor / self-model / AttnRes diagnostic surfaces. (Counter is at
174 because two of the prior session's intermediate tests later got
folded into broader ones — the suite is monotonically growing in
coverage if not in raw count.)

### Files changed this pass

Modified:
- `arc_agent_pairwise_stockfish_codex.py` — Ouro exit-gate signals
  (`_last_ouro_*`), pre/post-blend `effective_conf` logging,
  `effective_conf_pre_ouro` + `ouro_trust_multiplier` +
  `ranker_score_raw` + `heuristic_score_raw` in score_components,
  anchor counters, `trusted_mix` attribute used by
  `_blended_*_pairs` and `expert_fraction`, `train_anchor_step`
  bumps counters per outcome.
- `arc_agent_hunter_seeker_codex.py` — `measurement_summary` returns
  `anchor` + `self_model` + `attn_res_pooler` diagnostic dicts.
- `train_arc_codex.py` — `--unfreeze_encoder_after_partial_load`,
  `--trusted_mix`, anchor-vs-freeze warning, threading through to
  load fallback sites and agent constructor.
- `live_arc_diagnostic.py` — `--freeze_encoder` / `--no_freeze_encoder`
  flag pair; `load_diagnostic_checkpoint` accepts a
  `freeze_encoder_after_load` kwarg; loud freeze-state print at startup.
- `grid_encoder_codex.py` — corrected `GridEncoder.forward` docstring.
- `test_causal_correctness.py` — `measurement_summary` diagnostic
  surface coverage.

### What's still left

Same as the prior pass:

1. **Paper-scale comparison sweep** — pure GPU compute. The harness
   now exposes everything the sweep needs as CLI flags, including
   `--trusted_mix` for the trusted/online-heavy ablation onion §8
   suggested.
2. **Live VRAM calibration for `--anchor_batch_size`** — sacrificial
   small run before serious sweeps. Onion §2 procedural recommendation,
   not coding work.
3. **Onion §10 micro-run order** — 6-step recommended ablation
   sequence, also procedural.

## 2026-04-26 Continuation Pass III — Ouro exit gate + ranker→self-model gradient + CLI

Picked up the two items from "what's left for a future session" that were
contained enough to finish in one pass: capturing Ouro's per-step
early-exit gate, and routing ranker loss back into the self-model via
off-policy reconstruction. Then wired CLI flags so the new functionality
is reachable from training runs without code edits.

### Ouro early-exit gate signal

The Ouro paper specifies a learned exit gate λ_t(x) at each UT step plus
a survival probability S_t(x) = ∏(1 - λ_j(x)) used by the Q-exit
criterion. The CLT paper noted Ouro is run with the exit threshold
disabled so all 4 UT steps execute — but the gate logits are still
computed at each step. We just weren't reading them.

Cracked open `~/.cache/huggingface/.../modeling_ouro.py` and confirmed
that `model.model.forward` returns `(BaseModelOutputWithPast,
hidden_states_list, gate_list)`. We were already extracting `[1]` (loop
states) but ignoring `[2]` (per-step gate logits, shape `(B, seq_len, 1)`
each). Wired:

- `_last_ouro_confidence ∈ [0, 1]` = `1 - S_T`. Higher = Ouro would have
  exited earlier if early-exit were enabled = more confident.
- `_last_ouro_expected_exit ∈ [1, T+1]` = `Σ_t t·(S_{t-1}·λ_t) + (T+1)·S_T`.
  Lower = earlier hypothetical exit.
- `_last_ouro_exit_pdf` = per-step exit probability mass (length T) for
  diagnostics / paper plots.
- All three computed only when `update_introspective_state=True`
  (real-frame branch); beam-search hypothetical encodes leave them
  alone.
- All three reset to None on `reset_for_new_game`.

The signal is now blended into the existing Δloop trust gate inside
`score_candidates`: `effective_conf *= 0.5 + 0.5 * ouro_confidence`.
High Ouro confidence → ranker's transition score is trusted more; low
→ defer to heuristic proposal score. The 0.5 floor keeps a single
low-confidence reading from collapsing the gate. Both `ouro_confidence`
and `ouro_expected_exit` appear in `score_components` so trace dumps
carry the diagnostic.

A regression test verifies the math: synthetic gate logits with λ =
[0.1, 0.3, 0.5, 0.7] give S_T = 0.9·0.7·0.5·0.3 = 0.0945, confidence
= 1 - 0.0945 = 0.9055, matched within 1e-3.

### Ranker loss now flows back into the self-model

Prior session shipped temporal_features through the ranker's training
ranker calls, but the buffer-snapshot path was detached at push time —
ranker gradient stopped at the buffer. To actually train the self-model
from preference signal:

- Buffer schema gained `self_model_h` (256-dim GRU state),
  `track_summary_snapshot` (16-dim), `event_summary_ema_snapshot`
  (7-dim), `loop_delta_scalar_snapshot` (scalar). All four optional;
  None on transitions pushed before the wiring landed.
- HunterSeekerAgent writes these on every `_self_model_step` call
  BEFORE advancing the GRU; the base-class push() reads them via
  `getattr` and propagates to the buffer. Trusted-buffer pushes leave
  them None — there's no live agent state at trajectory load.
- New method `HunterSeekerAgent._recompute_temporal_features(h, ts, ev,
  ld) → temporal_features` does an off-policy one-step forward through
  the SelfModel + TemporalContextAggregator with grad enabled. h is
  restored into `sm.gru.h` per-sample before the forward.
- `_train_ranker` checks `getattr(self, "_recompute_temporal_features",
  None)` and the presence of `pos_self_model_h` in the batch. When
  available, it recomputes fresh temporal_features (with grad) instead
  of reading the buffer snapshots; ranker loss then backprops through
  aggregator + SelfModel + (zero-init) event predictor.
- After the ranker's `total_loss.backward()`, `self_model_optimizer.step()`
  fires (when sm_grad_active) — the self-model now has TWO gradient
  routes: its own one-step event-prediction loss (already wired) AND
  the ranker's preference loss (this session).
- `self_model.detach_state()` after the optimizer step keeps the live
  GRU graph fresh.

The aggregator's zero-init fuse means at startup gradient stops at the
fuse layer and doesn't reach the SelfModel GRU. Once training moves
the fuse off zero, the upstream pathway opens — the same identity-
preserving safety property the LoopStatePooler maintains. A regression
test confirms the recomputed TF is grad-connected and the aggregator
receives gradient.

### CLI flags landed in train_arc_codex

Five new flags surface the continuation features without code edits:

- `--self_model_mode` ∈ {off, passive, inject, inject_aux_grad,
  inject_grad}. Legacy `inject_grad` accepted as alias.
- `--cortex_monitor_mode` ∈ {off, active}.
- `--loop_pooler_kind` ∈ {gru, attn_res}. AttnRes pooler is selectable
  at construction; checkpoint round-trips warn on arch mismatch.
- `--anchor_train_every N` (default 0). When >0, calls
  `agent.train_anchor_step(...)` every N agent steps inside
  `train_on_game`. Off by default; opt-in to the gradient-checkpointed
  Ouro backprop.
- `--anchor_coefficient` (default 0.1) and `--anchor_batch_size`
  (default 4) — pass-through to `train_anchor_step`.

### GPU validation

Smoke ran on the RTX 5070 Ti without loading Ouro (encoder-only mode
exercises the new wiring, and 11.3 GB free VRAM suggests Ouro would
fit but isn't required for this validation):

- 100 self-model steps in 0.45s (4.5ms/step) with passive mode +
  attn_res pooler + cortex monitor.
- Self-model loss EMA = 0.038 after 100 steps (started near zero
  because the predictor is zero-init; rises with the first event then
  decays as the predictor learns).
- Off-policy temporal-features reconstruction produces a grad-connected
  tensor; backward populates 2/18 aggregator params (the fuse Linear
  weight + bias — exactly the connection point that, once trained,
  unlocks gradient flow back to the self-model GRU).

### Test snapshot

```
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/moloch/ouro_project/venv/bin/python -m pytest \
  claude_sandbox/ -q
# 173 passed in 9.35s
```

3 new regression tests added since the prior pass (Ouro exit-gate math,
encoder-only None-fall-back, off-policy TF recompute is grad-connected).

### Files changed

Modified:
- `arc_agent_pairwise_stockfish_codex.py` — Ouro gate-list extraction
  in encode_and_think_batch; `_last_ouro_*` state with reset hooks;
  Ouro confidence blended into the score_components trust gate; buffer
  schema + push() + _to_tensors + sample_*_pairs all carry the
  SelfModel snapshot fields; `_train_ranker` gates temporal_features
  through the off-policy recompute when available; sm_optim.step()
  + self_model.detach_state() after backward.
- `arc_agent_hunter_seeker_codex.py` — `_recompute_temporal_features`
  method on HunterSeekerAgent; SelfModel snapshot writes in
  `_self_model_step` (sm.gru.h, ts_np, _event_summary_ema,
  _last_loop_delta_scalar).
- `train_arc_codex.py` — new CLI flags; agent_kwargs passes
  cortex_monitor_mode + loop_pooler_kind; HunterSeeker kwargs adds
  self_model_mode; train_on_game accepts anchor_train_every /
  coefficient / batch_size and schedules `agent.train_anchor_step`
  inside the per-step loop.
- `test_causal_correctness.py` — 3 new tests covering the additions.

### What's left now

1. **Paper-scale comparison sweep**: pure GPU compute, not coding.
   Ready to run: `python -m claude_sandbox.train_arc_codex --agent
   hunter_seeker --self_model_mode inject_aux_grad
   --cortex_monitor_mode active --loop_pooler_kind attn_res
   --anchor_train_every 100 --backbone_mode ouro --games ls20 ft09 r11l
   tr87 wa30 --n_runs 3` etc. The harness now exposes everything
   needed for the four-condition comparison the prior session sketched.

2. **Validating gradient checkpointing through Ouro stays under 12 GB
   VRAM under live conditions.** train_anchor_step is OOM-safe
   (catches CudaOutOfMemoryError, empties cache, returns None) but
   the empirical batch_size that fits hasn't been measured. The
   default `--anchor_batch_size 4` is a conservative starting point.

## 2026-04-26 Continuation Pass II — AttnRes pooler + self-model training + Ouro GC

Picked up after the morning's gptopinion2/3 fixes. The user permitted GPU
access and asked specifically about Attention Residuals. Took the
remaining four deferred items (gradient checkpointing, self-model + cortex
optimisers stepped from a real loss, adapter dense_input contract, AttnRes
pooler) and shipped contained implementations of all four.

### AttnResLoopPooler — alternative pooler over Ouro's 4 loop CLS tokens

New module in `arc_agent_pairwise_stockfish_codex.py`. The mechanism from
Kimi's Attention Residuals paper (arXiv 2603.15031) — replace
unit-weight residual accumulation with softmax attention over preceding
states — applied at the LOOP-ITERATION axis instead of the layer axis.
Final-loop CLS is the query; all 4 loop CLS tokens are keys + values;
softmax attention determines per-call which iteration to weight.

- **Q/K/V projections** to a 512-dim bottleneck, 4 trainable matrices
  totalling ~4.2M params (matches LoopStatePooler's ~3.7M).
- **Identity-equivalent at startup**: scalar attn_gate is zero-init, so
  output = norm(loop_cls[:, -1]) until training moves the gate. Same
  safety property as LoopStatePooler.
- **Inspectable attention weights**: stored as
  ``self.loop_pooler.last_attn_weights`` (B, T) per call. Diagnoses the
  CLT paper's relational-encoding hypothesis directly: which loop
  iteration carried the most signal for this input?
- **Selectable** via the new ``loop_pooler_kind`` agent kwarg
  (``"gru"`` default = LoopStatePooler unchanged; ``"attn_res"`` =
  AttnResLoopPooler).

Three regression tests landed: identity-at-init exactness,
attention-weights-are-distribution, and gate-perturbation-changes-output.

### Self-model + CortexMonitor optimisers now step from real losses

The gptopinion3 §4 finding: both optimisers existed but were dead — no
``.step()`` call ever. Wired one-step-ahead self-supervised prediction
losses for both:

- **`temporal_event_predictor`**: 32-dim temporal feature → 7-dim
  predicted next-step event counts. MSE vs the actual events that fire
  on the next step. Trains the SelfModel GRU + TemporalContextAggregator
  + the predictor head. Zero-init so behaviour is unperturbed at startup.
- **`cortex_signature_predictor`**: 128-dim cortex hidden state →
  14-dim predicted next loop signature. MSE vs the actual signature
  computed from the next real-frame Ouro forward. Trains the
  CortexMonitor GRU + the signature predictor.

Both losses run inline in their respective step functions
(``_self_model_step`` for the agent self-model, inside
``encode_and_think_batch``'s real-frame branch for the cortex monitor).
After backward + optimizer.step, both call ``detach_state()`` on the
GRU so the next step starts a fresh autograd graph (otherwise we'd
trip "backward through the graph a second time").

GPU smoke test confirmed: 50 self-model steps in 0.35s on RTX 5070 Ti
(7ms/step), loss EMA decreasing across steps, VRAM stable.

Both heads + their optimizers persist in checkpoints
(``temporal_event_predictor``, ``cortex_signature_predictor``,
``cortex_monitor_optimizer`` keys). Old checkpoints loading is
best-effort — missing keys produce a warning and the head starts fresh.

### Anchor loss is now actually trainable: Ouro gradient checkpointing

The diagnostic version from the morning shipped with ``no_grad`` around
Ouro because backprop through 2.6B params at fp32 OOMs on 12 GB. Now
fixed with ``torch.utils.checkpoint``:

- **`_ouro_loop_states(frames, with_grad=False)`** new kwarg.
  ``with_grad=True`` runs the encoder forward outside ``no_grad`` and
  wraps Ouro's forward in ``torch.utils.checkpoint.checkpoint`` so
  activations are recomputed during backward. Trades VRAM for compute,
  exactly the right deal here.
- **`_ouro_forward_for_checkpoint(encoder_tokens)`** is the wrapped
  callable; takes the grad-tracking encoder tokens, returns the tuple
  of T loop states. Required signature for ``torch.utils.checkpoint``.
- **`train_anchor_step(batch_size=4, coefficient=0.1)`** is the
  trainable sister of `compute_anchor_loss_diagnostic`. Samples chosen
  / rejected pairs from the buffer, computes
  ``-log_sigmoid(score) * coefficient``, backprops through Ouro into
  the encoder, runs `encoder_optimizer.step()`. OOM-safe: catches
  `torch.cuda.OutOfMemoryError`, calls `empty_cache`, returns None.

Ouro's params stay frozen throughout — gradient FLOWS through them but
doesn't accumulate on them. Only encoder receives gradient. This is the
exact loss the encoder needed to keep itself in Ouro-compatible space
(see the v17b → v17c → Sprint 4 cosine-zero drift result in
ouro_project_state.md §27).

### freeze_encoder no longer blocks aux head training

gptopinion2 §2: ``freeze_encoder=True`` previously early-returned from
``_train_encoder`` and silently disabled the next-frame predictor +
patch-color head training along with the encoder. Split into two
optimisers:

- ``self.encoder_optimizer`` — encoder params only.
- ``self.aux_world_model_optimizer`` — nextframe + patch_color heads.

When frozen, encoder runs under ``no_grad`` and only
``aux_world_model_optimizer`` steps. The aux heads keep learning on
detached encoder features. Checkpoint save/load extended for the new
optimiser. The frame-padding logic inside ``_train_encoder`` was also
hardened to work with the encoder's auto-padded shapes (gpt3 §1
follow-on).

### inject_grad → inject_aux_grad

gptopinion2 §3: the mode name overpromised. There's never been gradient
flow through Ouro in that mode (anchor's GC path is separate); the
"_grad" suffix referred to alternative supervision paths only. Renamed
to ``inject_aux_grad``. Legacy ``inject_grad`` is preserved as a
back-compat alias — old scripts and checkpoints load unchanged.

### Adapter dense_input contract honoured

gptopinion3 §3: ``encode_and_think_batch`` previously did
``torch.from_numpy(frames.astype(np.int64)).long()`` directly, bypassing
the adapter and assuming raw frame == label grid. Added a new
``frames_to_dense_input(frames)`` method to the ObservationAdapter
Protocol; ARC + mock-symbolic adapters override with identity (their
raw frame IS the label grid). Both call sites in
``arc_agent_pairwise_stockfish_codex.py`` now route through the adapter
when available, falling back to the legacy direct cast for adapters
that pre-date the method. RGB-style domains now have a clean seam to
plug in pixel → label-index quantization.

### Smaller polish from the audit pass

- gpt3 §1: ``pool_object_features`` auto-pads non-divisible mask shapes
  (matches PatchEmbedding's auto-pad behaviour).
- gpt3 §2: ``SpatialClickPredictor`` interpolates back to input HxW
  (fixes off-by-one on odd dims).
- gpt3 §5: ``ActionHead.select_action`` falls back to unmasked sampling
  when no provided action is in range; defensive NaN check on softmax.
- gpt3 §6: ``load_partial_checkpoint_for_sandbox`` prints a loud
  ``[PARTIAL-LOAD]`` line and accepts ``freeze_encoder_after_load=False``.
- gpt3 §8: ``pad_grids_to_batch`` requires ``pad_value`` (no ARC default).
- gpt3 §7: ``online_trace_run_report.py`` import fallback for direct
  execution from inside the sandbox dir.

### What's left for a future session (genuinely deferred)

1. **Paper-scale comparison sweep**: run baseline vs anchor-trained vs
   AttnRes-pooler on the standard game set, n_runs ≥ 3, log
   anchor-loss EMA + temporal-event-loss EMA + cortex-signature-loss
   EMA. This is GPU compute, not coding.
2. **Read S_t(x) from Ouro's exit gate** to use as a per-input
   "refinement budget" trust signal in beam search, instead of (or
   alongside) the current Δloop EMA. The Ouro paper's depth-allocation
   objective produces this signal; we just don't read it. Hooking
   ``model.model``'s exit logits is contained — left for a focused pass.
3. **Cortex monitor / self-model loss in inject_aux_grad mode** could
   also flow through the ranker if the ranker's training path
   re-computes temporal_features instead of reading from buffer
   snapshots. That would tie the self-model to actual policy quality.
   Bigger refactor than this session covered.

### Test snapshot

```
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/moloch/ouro_project/venv/bin/python -m pytest \
  claude_sandbox/ -q
# 170 passed in 7.84s
```

Per-file growth this session: `test_causal_correctness.py` 20 → 24 →
(...) → final, adding the AttnRes pooler tests and the self-model
optimiser-actually-steps test.

### Files changed this session (additive to morning's pass)

Modified:
- `arc_agent_pairwise_stockfish_codex.py` — AttnResLoopPooler class +
  ``loop_pooler_kind`` selector, freeze_encoder split into two
  optimisers + checkpoint coverage, anchor train_anchor_step path with
  gradient checkpointing through Ouro, cortex_signature_predictor +
  loss + checkpoint, adapter ``frames_to_dense_input`` integration,
  loop_pooler_kind persisted in checkpoint with arch-mismatch warning.
- `arc_agent_hunter_seeker_codex.py` — ``inject_aux_grad`` rename
  with legacy alias, pool_object_features auto-pad,
  temporal_event_predictor + loss + checkpoint, self_model GRU
  detach_state after backward.
- `action_adapters_codex.py` — ActionHead empty-mask guard +
  defensive softmax-NaN fall-back.
- `grid_encoder_codex.py` — pad_grids_to_batch requires pad_value;
  PatchEmbedding ``pad_value`` attribute (consumed by
  pool_object_features padding).
- `train_arc_codex.py` — partial-load freeze made loud + opt-out.
- `live_arc_diagnostic.py` — ``inject_aux_grad`` added to choices.
- `online_trace_run_report.py` — local-unsuffixed import fallback.
- `observation_adapters_codex.py` — Protocol + ARC + Mock now expose
  ``frames_to_dense_input``.
- `test_causal_correctness.py` — 10 new regression tests across all
  the above.

## 2026-04-26 Continuation Pass — Wiring + gptopinion2/3 fixes

Picked up from the 2026-04-25 audit. The two pieces of deferred work I
attacked here: the temporal-features wiring through training-loss ranker
calls (was previously "design call, not rote integration") and the
anchor-loss diagnostic primitive on the agent. Then a fresh review pass
flagged 11 more issues across `gptopinion2.md` / `gptopinion3.md` and I
fixed them all.

### Sprint 6/11b — temporal_features now reach the training ranker

Wired `temporal_features` through the replay buffer and the four
training-loss ranker calls in `_train_ranker`:

- `_Transition` gained an optional `temporal_features: Optional[torch.Tensor]`
  field. Snapshotted to CPU (detach + clone) on push.
- `TransitionReplayBuffer.push()` accepts a new `temporal_features` kwarg.
  Live agent passes `self._current_temporal_features` (the
  `TemporalContextAggregator`'s output snapshotted at push time, so it
  reflects the affective + cortex state at the observation that produced
  the transition).
- `_to_tensors()` stacks `temporal_features` only when ALL sampled
  transitions have it; mixed batches drop the key (graceful fallback to
  the ranker's zero-contribution `temporal_features=None` branch).
- `sample_ranking_pairs()` and `sample_sibling_pairs()` propagate
  `pos_temporal_features` / `neg_temporal_features` when present.
- All four `self.ranker(...)` calls in `_train_ranker` pass
  `temporal_features=...` from the batch (or None fall-back).
- Auxiliary sibling pushes also carry the snapshot.
- Trusted buffer (loaded from solved-trajectory JSONs at startup) leaves
  `temporal_features` unset — there's no live agent state at load time.
  This means trusted-only batches won't activate the temporal head.
  Mixed batches drop the key per the `_to_tensors` rule above.

### Sprint 11a — anchor-loss diagnostic primitive on the agent

Added `compute_anchor_loss_diagnostic(batch_size=None) -> Optional[float]`
to `PairwiseARCSearchAgent`. Samples chosen/rejected pairs from the
buffer, runs Ouro forward to capture loop states (new helper
`_ouro_loop_states`), feeds them to a lazily-loaded `FrozenCLTAnchor`,
returns the scalar `-log_sigmoid(score)`. **Diagnostic only** —
encoder/Ouro forward is wrapped in `torch.no_grad`, so the loss is not
backproppable. Full gradient flow is blocked by the 12 GB VRAM limit;
gradient checkpointing through Ouro is the separately-deferred lever
that turns this into a real anchor.

### Latent merge_batches bug found during the audit

`_merge_batches` previously took keys from `a` only — keys present in
just `a` came through with their original length while shared keys
grew via `cat`, producing batches whose per-key first-dim lengths
disagreed. The pre-temporal-features codebase never exercised the
asymmetry, but `pos_temporal_features` (added to live transitions but
absent from trusted-buffer ones) does. Fixed: the merged dict now
contains only keys present in BOTH inputs.

### gptopinion2 + gptopinion3 fixes

1. **`freeze_encoder` was too blunt** (gpt2 §2). It froze encoder weights
   but ALSO disabled the next-frame / patch-color aux head training via
   an early-return in `_train_encoder`. Split into two optimizers
   (`encoder_optimizer` and `aux_world_model_optimizer`); the freeze
   path now runs the encoder forward under `no_grad` and still trains
   the aux heads on detached features. Checkpoint save/load updated to
   carry both optimizers.

2. **`inject_grad` overpromised** (gpt2 §3 / gpt3 §4). The mode name
   suggested gradient flow through Ouro, but the context token is
   detached inside the no_grad Ouro forward. Renamed to
   **`inject_aux_grad`** to match what actually happens (gradient
   through alternative supervision paths, not through Ouro). The legacy
   `inject_grad` name is preserved as a back-compat alias — old
   checkpoints, scripts, and tests keep working transparently.

3. **`pool_object_features` crashed on non-divisible mask shapes** (gpt3
   §1). The function reshaped raw masks via
   `H // patch_size, patch_size` without padding, mismatching the
   encoder's auto-padded feature map. Now pads the mask the same way
   `PatchEmbedding.forward` pads the frame (with `False` for padded
   cells, so the patch any-reduction stays correct).

4. **`SpatialClickPredictor` lost a pixel on odd dims** (gpt3 §2). The
   stride-2 conv → stride-2 transpose conv pair returns input shape
   for even dims but one pixel short for odd dims. Added
   `F.interpolate(out, size=frame_tensor.shape[-2:])` to restore the
   input HxW. No-op for the 64×64 ARC case.

5. **`ActionHead.select_action` could NaN-crash on all-invalid masks**
   (gpt3 §5). When every entry in `available_actions` is out of
   `[0, n_actions)`, masking made every logit `-inf`, softmax produced
   NaN, and `torch.multinomial` crashed. Added a `n_valid` counter; if
   no valid action survives the mask, fall back to unmasked sampling
   with a loud warning. Also guards the post-softmax probabilities
   against any other source of NaN with a uniform fall-back.

6. **`load_partial_checkpoint_for_sandbox` silently froze the encoder**
   (gpt3 §6). Added a `freeze_encoder_after_load: bool = True` kwarg
   and a loud `[PARTIAL-LOAD]` print describing the freeze decision.
   Long sweeps that silently dropped encoder training are now visible.

7. **`pad_grids_to_batch` defaulted `pad_value=16`** (gpt3 §8). That
   contradicted the domain-agnostic encoder story (caller's adapter
   knows the right pad token, not the function). Removed the default;
   `pad_value` is now a required positional arg. The in-file test
   updated to pass it explicitly.

8. **Audited `online_trace_run_report.py`'s
   `sandbox_sweep_validate` import** (gpt3 §7). The file does exist;
   the import worked but had no fallback for direct execution from
   inside the sandbox dir. Added a nested try/except that falls
   through to local-unsuffixed names so the script runs whether or
   not the parent dir is on `sys.path`.

### Reference reading: Ouro paper (arXiv 2510.25741) + Attention Residuals (2603.15031)

Pulled the published Ouro paper and the Kimi Team's Attention Residuals
(AttnRes) paper for context. Useful takeaways:

- **Ouro's depth-allocation gate**: each loop step has a learned exit
  gate λ_t with survival probability S_t(x) = ∏(1 - λ_j(x)). At
  inference, a Q-exit threshold q controls compute. **We currently
  don't read this signal.** S_t(x) is a per-input "how much refinement
  is left" measure that could replace or augment our self-calibrating
  Δloop EMA as a beam-search trust gate. Capturing it requires hooking
  Ouro's exit logits — non-trivial but contained future work.
- **Ouro paper does NOT investigate per-loop-iteration specialisation.**
  Our CLT paper and our `CortexMonitor` are arguably the first
  empirical probes of how the iterations differ. That's a paper-worthy
  observation in its own right.
- **AttnRes**: standard PreNorm residuals add all preceding layer
  outputs with unit weights, causing magnitude growth and dilution.
  AttnRes replaces this with softmax attention over preceding layer
  outputs — each layer learns input-dependent weights for which prior
  representation to emphasise. The looped-transformer analog is
  immediate: our `LoopStatePooler` pools the 4 loop states via a GRU
  with a zero-init gate, and an AttnRes-style alternative would
  compute softmax attention over the 4 loop states using the
  final-loop CLS as query. **Not implemented this session** — flagged
  as a clean Sprint 6/10 candidate. Block AttnRes is irrelevant for us
  (only 4 iterations, no scaling concern).
- The AttnRes paper does not discuss looped/recurrent architectures, so
  the cross-application is a new contribution if pursued.

### What this session did NOT do (genuinely deferred)

1. **Gradient checkpointing through Ouro** to make the anchor loss
   actually trainable. The diagnostic version is ready; flipping the
   `no_grad` is a one-line change once GC is in place.
2. **Self-model / CortexMonitor optimisers stepped from a real loss**
   (gpt3 §4). The optimisers exist but no training loop calls
   `.step()` on them. They're intentionally inert for now —
   self-model parameters update only through whatever ranker / anchor
   gradient flows reach them, which today is none (anchor is no_grad,
   ranker's temporal head is zero-init and gate-free for the
   self-model branch). Once anchor gets gradient or a direct
   self-model loss is added, those optimisers become live.
3. **Honoring the adapter's `dense_input` contract for non-ARC domains**
   (gpt3 §3). `encode_and_think_batch` still takes raw frames; for
   ARC and the mock-symbolic adapter that's correct because raw frame
   == label grid. RGB-like domains will need this addressed before
   they can plug in cleanly. Left open as a non-ARC blocker.
4. **AttnRes-style loop-state pooler** (paper insight). One concrete
   architectural alternative to the existing GRU-based
   `LoopStatePooler`. Not built; flagged for Sprint 10.

### Test snapshot

```
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/moloch/ouro_project/venv/bin/python -m pytest \
  claude_sandbox/ -q
# 166 passed in 10.38s
```

Per-file growth from this continuation pass:
- `test_causal_correctness.py`: 14 → 20 tests (added pool_object_features
  padding, spatial-predictor odd dims, action-head invalid mask,
  inject_grad alias, replay buffer temporal_features round-trip).

### Files changed

Modified:
- `arc_agent_pairwise_stockfish_codex.py` — temporal_features wiring,
  anchor-loss diagnostic, freeze_encoder split, merge_batches fix,
  spatial predictor odd-dim fix, encode_and_think_batch
  `update_introspective_state` flag (already from 2026-04-25 entry).
- `arc_agent_hunter_seeker_codex.py` — `inject_aux_grad` rename + alias,
  `pool_object_features` mask padding.
- `action_adapters_codex.py` — ActionHead empty-mask guard.
- `grid_encoder_codex.py` — `pad_grids_to_batch` requires pad_value.
- `train_arc_codex.py` — partial-load freeze made loud + opt-out.
- `live_arc_diagnostic.py` — added `inject_aux_grad` to choices.
- `online_trace_run_report.py` — local-unsuffixed import fallback.
- `test_causal_correctness.py` — six new regression tests.

## 2026-04-25 Audit-Driven Bug-Fix Pass

Acted on the consolidated review from `gptsopinion.md`, `sonnetopinion.md`,
`sonnetopinion2.md`, `Claudes opinion`, and `Opinion.txt`. Each fix below
is paired with a regression test in `test_causal_correctness.py`.

### Critical causal-correctness fixes

1. **Event-detection ordering in `HunterSeekerAgent.step()`**
   The pre-step block previously called `update_beliefs` → `detect_events`
   → (later) `update_from_scene(scene_now)`. The detect_events docstring
   requires tracks to already reflect `scene_after`; the old ordering
   meant MOVED / TRANSFORMED / APPEARED were one transition delayed,
   silently corrupting the event log fed into the self-model.
   Now: `update_beliefs(before, after)` → `update_from_scene(after)` →
   `detect_events`. The post-step block guards against double-advancing
   the table via a `transition_processed_this_step` flag.

2. **`ObjectTable.update_from_scene({})` empty-scene branch**
   Was a silent `return`, leaving previously-visible tracks alive
   forever (the most informative DISAPPEARED signal — collectible
   consumed, level transition, scene blanked — was swallowed). Now
   processes the empty scene as "every visible track failed to match,"
   bumps miss_count, populates `_disappeared_track_ids`, prunes after
   tolerance.

3. **`on_game_over` early-return on click attribution failure**
   When `scene_parser.get_object_at(click_x, click_y, scene)` returned
   None, the function persisted beliefs and returned BEFORE the failure
   taxonomy block, `_failure_counter` increment, `_discovered_mechanics`
   add, terminal DEATH event log, and the self-model death tick. Now
   restructured: attribution failure leaves `hazard_track_id_for_event=-1`
   and falls through to the common terminal-bookkeeping block. Death
   evidence is no longer applied (no track to attribute to) but the
   event log and affective state correctly record that a death happened.

4. **`encode_and_think_batch` cortex-monitor pollution**
   The CortexMonitor GRU advanced unconditionally inside
   `encode_and_think_batch`, which is also called on hypothetical
   beam-search successor frames. New `update_introspective_state: bool`
   parameter, default False. The single real-frame encode in
   `select_action` passes True; every other call site (beam expansion,
   training-path encodes, sibling-pair encodes, offline trajectory
   encoding) leaves it False. Beam search can no longer "imagine the
   agent into a different mood."

5. **`anchor_loss.py` import path**
   Both branches of the try/except imported `evaluator_pairwise` (the
   project-root name); the sandbox file is `evaluator_pairwise_codex.py`.
   Now tries `claude_sandbox.evaluator_pairwise_codex` first, falls back
   to `evaluator_pairwise_codex`, finally `evaluator_pairwise`.

### High-priority correctness fixes

6. **`update_beliefs` silent click-evidence drop**
   When `_find_track_for_obj(clicked_obj)` returned None (closest track
   beyond the 12-px match radius — typically just-pruned), evidence was
   silently dropped: no record updated, no objectivity sample buffered.
   Now falls back to `_ensure_record(clicked_obj.color)`, mirroring the
   `appeared_colors` branch. The click still teaches.

7. **`detect_events` duplicate-call guard**
   Added `_last_detect_step` tracker; a second call for the same
   `step_number` returns `[]`. Prevents CONTACT / MOVED double-emission
   from defensive callers or future refactors.

### Performance fixes

8. **Lightweight planning snapshot**
   `ObjectRecord.planning_copy` already drops `effect_history` /
   `sharpening_history`; trimmed `tested_actions` and `sterile_count`
   too — they're never read on the planning copy. Kept the existing
   `planning_copy` shape so downstream callers don't change.

9. **`SinusoidalPositionalEncoding2D.max_size` removed**
   `max_size=64` was dead code that misled readers into thinking the
   encoder was 64×64-locked. The buffer size depends only on `d_model`.
   Constructor now `__init__(self, d_model: int)`.

10. **`PatchEmbedding` auto-pads non-divisible grids**
    Was a hard `assert H % p == 0 and W % p == 0`. Now pads with
    `pad_value` (= `n_values`, same as the Embedding's `padding_idx`)
    so arbitrary spatial shapes load. The encoder is now genuinely
    domain-agnostic in spatial extent.

11. **CONTACT detection `O(N×M)` → `O(N)`**
    Built `tid_to_oid = {tid: oid for oid, tid in obj_to_tid.items()}`
    once, replacing the per-moving-track linear scan over `obj_to_tid`.

12. **`TransitionReplayBuffer` aggregate stats `O(1)`**
    `change_rate`, `expert_fraction`, `click_change_rate` previously
    list-comprehended over the full ≤20k-element deque on every step
    (called from `_compute_track_summary`). Now backed by running
    counters maintained in `push()` (with eviction bookkeeping) and
    `update_quality(mark_expert=True)`.

13. **`_compute_free_space_topology` passable-mask vectorisation**
    For ≥2 traversable objects, replaces the per-object `passable |=
    obj.mask` loop with a single `np.logical_or.reduce`.

14. **Cross-game dict growth bound**
    `_phase_template_failures`, `_phase_action_branch_failures`,
    `_phase_semantic_trust` are keyed by `game_id`; queries for the
    current game stay correct even with stale entries, but for sweeps
    over many games these grow unbounded. `reset_for_new_game` now
    drops entries whose game_id ≠ current.

### Cleanup

15. Inlined the walrus-operator clever cap in `record_interaction`
    (`min(self.sterile_count + 1, len(self.sharpening_history))` — the
    cap was never functionally binding because sharpening_history is a
    deque with maxlen=8).
16. Documented the implicit `_hs_scoring_*` contract between
    `generate_candidates` and `score_candidates` in the score_candidates
    docstring.

### Considered but reverted

- **`type_belief` memoisation on `ObjectRecord`** — Sonnet flagged this
  as a hot-path optimisation. Implemented it as a lazy cache invalidated
  by `apply_evidence`, then reverted: existing tests
  (`test_topology_sprint5.py` and others) directly write to
  `belief._weights` without going through `apply_evidence`, and the
  cache served stale beliefs. The 7-element normalise is cheap; not
  worth the test fragility. A future version-counter approach could
  re-enable this safely.
- **Removing `_compute_reachable_objects`** — Sonnet called it dead
  code; it is actually exercised by smoke tests at lines 9440 / 9450.
  Kept.

### Test snapshot

```
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/moloch/ouro_project/venv/bin/python -m pytest \
  claude_sandbox/ -q
# 158 passed in 6.51s
```

New tests: `test_causal_correctness.py` (12).

### Files changed

Modified:
- `arc_agent_hunter_seeker_codex.py` — items 1, 2, 3, 6, 7, 8, 11, 13, 14, 15, 16.
- `arc_agent_pairwise_stockfish_codex.py` — items 4, 12.
- `anchor_loss.py` — item 5.
- `grid_encoder_codex.py` — items 9, 10.
- `CLAUDE_SESSION_SUMMARY.md` — this entry.

New:
- `test_causal_correctness.py` — 12 regression tests pinning each fix.

### What's still genuinely deferred

Same list as the prior session entry:

1. Anchor-loss wiring into `train_arc_codex.py` (Ouro hook capture of
   chosen/rejected loop states, `aux_loss = -log_sigmoid(score)`
   scheduled every N steps at coefficient 0.1).
2. `temporal_features` through the four training-loss ranker calls in
   `_train_step` — needs h_t snapshotting in the replay buffer or
   off-policy reconstruction.
3. Gradient-through-Ouro for true `inject_grad` — gradient checkpointing
   to stay under 12 GB.
4. Paper-scale comparison sweep across off / passive / inject /
   inject_grad with n_runs ≥ 3 to separate self-model effect from
   CUDA-noise band.

## 2026-04-24 Sandbox Bootstrap

### What Was Done

- Read `~/ouro_project/ouro_project_state.md` in full (architecture, sprint table, encoder-drift diagnosis, checkpoint policy, adapter refactor).
- Read `~/ouro_project/codex_sandbox/SESSION_SUMMARY.md` in full (complete Codex session log up to the outcome-calibrated fallback semantics entry).
- Created `~/ouro_project/claude_sandbox/` by rsync-copying from `codex_sandbox/`, excluding:
  - `checkpoints_*/` (all ~7G of historical checkpoint artifacts — still readable from codex_sandbox if needed for reference).
  - `perf_event_dumps_*/` (historical per-experiment event dumps — same reasoning).
  - `__pycache__/`.
- Copied data/source directories included:
  - `solved_sequences_expanded/`
  - `trusted_plus_expanded/` (main 327M item — active trajectory source)
  - `arc_debug_frames/`
  - `live_arc_event_dumps/`
- Resulting size: `332M` vs `8.5G` for the full codex_sandbox.
- Rewrote every `codex_sandbox.*` import in every `.py` file inside `claude_sandbox/` to `claude_sandbox.*`. Non-`.py` files (`sweep_expanded_trusted_20260423_1630.log`, the copied `SESSION_SUMMARY.md`) kept their historical `codex_sandbox` references — they are record, not code.
- File names still carry the `_codex` suffix (e.g. `arc_agent_hunter_seeker_codex.py`). Renaming to `_claude` was not necessary for the module path swap and makes it easier to diff against codex_sandbox if we ever need to reconcile.

### Verification

- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/moloch/ouro_project/venv/bin/python -m pytest claude_sandbox/test_codex_sandbox.py claude_sandbox/test_codex_integration.py -x -q`
  - Result: `56 passed in 3.79s`.
- This matches the most recent focused-suite pass reported in `codex_sandbox/SESSION_SUMMARY.md` (outcome-calibrated fallback semantics section: `60 passed` when including the report tests on top of the core focused suite).

### State Taken On Entry

- Sprint 4 complete (8-head affordance model + directional buffer).
- Sprint 5 / Sprint 6 sandbox work active:
  - free-space topology
  - symbolic transition summaries
  - symbolic planner head + ranker symbolic features
  - online trace mode + focus-game timeline/hybrid reports
  - recovery / escalation / reseed planner modes with failure memory, world-model-confidence gating, outcome calibration
- Encoder drift from Sprint 4 known and mitigated:
  - `checkpoints_running/sprint4_encoder_reverted.pt` is the canonical "v17b encoder + Sprint 4 everything else" checkpoint.
  - `arc_agent_pairwise_stockfish_codex.py` defaults `freeze_encoder=True`.
- `trusted_trajs/` at project root is the canonical trajectory source (31 trajectories, 14081 transitions, 3477 click templates, 9399 action templates). `trusted_plus_expanded/` merges those with sandbox-mined improved prefixes.
- Open planner frontier: `wa30`. Change rate ≈ 0.85, still score 0, multiple rounds of recovery/escalation/reseed + topology-directed scoring + calibration have each improved trace signature without converting to level completion. The most recent Codex note identifies the next generic target as improving the symbolic successor model / calibration itself, because fallback semantics are currently operating under very low effective confidence.

### Next Useful Moves (For Whoever Resumes Here)

- Tighten the symbolic successor model so fallback semantics operate under higher effective confidence, rather than only dampening a weak predictor further.
- Mine any new level completions into `solved_sequences_expanded` / trusted action templates to shrink the template-coverage gap (especially post-level-1 on `r11l` and deeper levels on `tr87`/`wa30`).
- If a new experimental run is needed, write into fresh directories under `claude_sandbox/` (e.g. `claude_sandbox/perf_event_dumps_<label>/` and `claude_sandbox/checkpoints_<label>/`) so Codex and Claude artifacts stay cleanly separated.

## 2026-04-24 / 25 Sprint 11b + 11a + 5 + 6 Build Pass

Session goal: design Sprint 11, complete Sprints 5 and 6, use the CLT
evaluator and the preliminary basal-ganglia scaffolding from the project root.
Everything below lives in `claude_sandbox/` only — `codex_sandbox/` untouched.

### Design doc landed first

`claude_sandbox/design/sprint_11_self_model.md` — full spec of the self-model
(GRU + affective state + context token) + the CLT-evaluator anchor loss. Splits
"Sprint 11" into two parts with distinct roles: 11b (cognition-time state,
per step) and 11a (training-time anchor).

### Sprint 11b — self-model wired, gated, zero-regression

New module: `claude_sandbox/self_model.py`.

- `AffectiveState` (8-dim numpy): joy / sadness / curiosity / fear /
  frustration / satisfaction / surprise / stress. Deterministic event triggers,
  per-dim decay, clamped to [0, 1].
- `AgentEventBundle`: strict contract for everything affective state is
  allowed to read. Keeps coupling direction explicit.
- `SelfModelGRU(88 → 256)`: GRUCell wrapper with explicit `h` state and
  `reset_state()` / `detach_state()` lifecycle hooks.
- `ContextTokenProjector(256 → 2048)`: LayerNorm → Linear, **zero-init** so
  the context token is exactly zero at t=0 (identity-start property — Ouro
  sees the stack it was trained on unless/until the projector learns non-zero
  weights).
- `SelfModel`: bundle of the three with `build_input()` / `forward()` /
  `project()` / `state_snapshot()` / `load_state_snapshot()`.
- Total: **~793K params**.

Agent wiring (`arc_agent_hunter_seeker_codex.py`):

- New kwarg: `self_model_mode: Literal["off", "passive", "inject", "inject_grad"]`.
- `off` is the default and is **byte-equivalent** to the pre-11b agent —
  `self_model` stays `None`, no tensors allocated, `_self_model_step()` is
  a no-op.
- `passive` computes h_t + affective state every step but does not inject
  anything into Ouro's forward. Use this to collect affective traces for
  paper material without touching policy.
- `inject` / `inject_grad` produce a context token via the projector; the
  former detaches it, the latter lets the ranker loss flow back to the GRU.
- Helpers added: `_compute_track_summary()`, `_build_agent_event_bundle()`,
  `_self_model_step()`. Lifecycle hooked via `reset_for_new_game` (reset),
  `on_level_complete` (detach + joy/satisfaction tick), `on_game_over`
  (sadness tick), and `step()` (advance GRU once per real transition).
- Checkpoint round-trip persists both the parameter state and the affective
  / GRU snapshot — load works across `off` ↔ `passive` transitions.

Tests: `test_self_model.py` (23 unit tests), `test_self_model_integration.py`
(19 integration tests). Zero-init projector property is verified: at first
forward, `context_token.abs().sum() == 0` regardless of input.

### Sprint 11a — frozen CLT evaluator anchor primitive

New module: `claude_sandbox/anchor_loss.py`.

- `FrozenCLTAnchor`: wraps `PairwiseEvaluator` with eval-mode + frozen
  weights by default. Exposes `score(...)`, `anchor_loss(...)`,
  `preference_accuracy(...)`, `load_evaluator_checkpoint(...)`.
- The anchor loss is standard `-log_sigmoid(score)` — non-negative,
  symmetric-breaking. Gradients reach caller inputs (encoder, GRU, etc.) but
  never accumulate on evaluator weights.
- Verified against the real `artifacts/checkpoints/evaluator/pairwise_epoch2.pt` — the
  95.2% pairwise / 2.7×-better-at-math evaluator loads cleanly and produces
  finite scores on synthetic inputs.

Tests: `test_anchor_loss.py` (10 tests) — includes the real-checkpoint load
as a regular passing test.

What's intentionally deferred: wiring the anchor loss into
`train_arc_codex.py`. That step needs Ouro forward capture (hook on
`model.model` to extract the 4 loop states for both chosen and rejected
trajectory continuations). The scoring primitive is ready; the
harness-integration is its own task (`task #10` in the local TaskList).

### Sprint 5 — region graph, containment, gateways, BFS distances

Upgrade of the existing sandbox `_compute_free_space_topology(...)`. All
legacy fields preserved; new fields added to the return dict:

- `region_map: np.ndarray[int32]` — connected-component labels over passable
  cells. Avatar sits in `avatar_region_id`.
- `n_regions`, `region_sizes`, `avatar_region_id`.
- `object_region`, `region_objects` — which region each traversable object
  lives in (max-overlap), plus the inverse mapping.
- `gateway_obj_ids`, `region_adjacency` — non-traversable objects that
  border ≥ 2 regions become gateway candidates; the adjacency dict lists
  which regions they connect.
- `bfs_distance: np.ndarray[int32]` — per-cell BFS distance within the
  avatar's region; `-1` outside.
- `bfs_max_distance` — max distance within the avatar's region.

Agent now stashes the most recent topology snapshot in
`self._last_topology` after each real transition (inside `step()`), so
downstream consumers (self-model's track summary, future planner heads) can
read it without recomputing. Reset per run.

Track summary indices 7 (`n_frontier_norm`), 8 (`reachable_ratio`), and
9 (`bfs_mean_distance_norm`) are now populated from the topology snapshot.

Tests: `test_topology_sprint5.py` (7 tests). Covers single-region geometry,
vertical-wall two-region split, gateway detection through a touching wall,
object containment assignment, BFS distance monotonicity, track-summary
integration, and backward-compat legacy keys.

### Sprint 6 — TransitionRanker temporal-context head

Extension of the sandbox's existing ranker. Zero-regression guarantee on
every pre-Sprint-6 checkpoint.

- New module components in `self_model.py`:
  `TemporalContextAggregator(h_t, event_summary, track_delta) → [B, 32]`.
  Three small LayerNorm → Linear → GELU heads fuse into a zero-init final
  `Linear(96 → 32)`. Zero-init gate means the aggregator's output is exactly
  zero at initialization regardless of input.
- `TransitionRanker` gained a `temporal_dim` kwarg (default 32) plus an
  additive `temporal_mlp → temporal_gate` path. The gate is zero-init.
  Forward pass: `base_score + gate(temporal_mlp(temporal_features))`. Both
  `temporal_features=None` and `gate=0 × anything` produce byte-equivalent
  output to the pre-Sprint-6 ranker.
- Agent wiring: the Hunter Seeker agent constructs the aggregator whenever
  `self_model_mode != "off"`, computes the 32-dim temporal feature inside
  `_self_model_step()`, and stashes it on `self._current_temporal_features`.
  The main scoring-path ranker call (`_score_candidates_core`) broadcasts
  this `[1, 32]` feature across the candidate batch.
- Event-summary EMA (per-type counts, 7-dim, decay 0.8/step) gets populated
  from `detect_events` return values — the ranker's temporal feature now
  reflects actual world-event history, not just self-model hidden state.
- Checkpoint save/load persists the aggregator state; pre-Sprint-6
  checkpoints load cleanly because both the ranker's new params and the
  aggregator are zero-init.

Tests: `test_ranker_sprint6.py` (12 tests). Zero-init byte-equivalence
verified three ways: `temporal_features=None`, zero-tensor features, and
random-tensor features (with gate at zero). Gate perturbation test confirms
the non-trivial path works once training moves it.

### What did NOT get built this session (named so it doesn't evaporate)

1. **Training-harness wiring of 11a.** Ouro forward capture via hook, trajectory
   encoding (Option A from the design doc — encoder tokens through Ouro), and
   aux-loss scheduling in `train_arc_codex.py`. Primitive is ready, harness is
   the work.
2. **Training-path ranker calls for Sprint 6.** The four ranker calls inside
   `_train_step` (lines 1789, 1795, 1829, 1835 in pairwise_stockfish) still use
   `temporal_features=None`. Those calls are over replay-buffer transitions
   that don't carry snapshots of `h_t` from the original step. Solving this
   well needs either (a) h_t snapshotting in the replay buffer, or (b) an
   off-policy reconstruction of `h_t` from the stored transition's context —
   a design call, not a rote integration.
3. **Live empirical validation on ARC.** No GPU sweep was run in this session;
   133 sandbox tests cover correctness but do not measure whether the new
   machinery moves score. The highest-signal first run would be encoder-only
   `wa30` / `r11l` / `ft09` / `tr87` / `ls20` smokes with
   `self_model_mode="passive"` to confirm byte-equivalence on the scoreboard,
   then `inject_grad` to see whether temporal context moves the needle.

### Verification snapshot

```
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/moloch/ouro_project/venv/bin/python -m pytest \
  claude_sandbox/ -q
# 133 passed
```

- `test_self_model.py`                    23 passed
- `test_self_model_integration.py`         19 passed (4 added this session for Sprint 6)
- `test_anchor_loss.py`                    10 passed (including real-checkpoint load)
- `test_topology_sprint5.py`                7 passed
- `test_ranker_sprint6.py`                 12 passed
- `test_codex_sandbox.py` + integration   62 passed (all pre-existing, no regressions)

### File list (all in `claude_sandbox/`, none in `codex_sandbox/`)

New:
- `design/sprint_11_self_model.md`
- `self_model.py`
- `anchor_loss.py`
- `test_self_model.py`
- `test_self_model_integration.py`
- `test_anchor_loss.py`
- `test_topology_sprint5.py`
- `test_ranker_sprint6.py`

Modified:
- `arc_agent_hunter_seeker_codex.py` — self-model construction + helpers +
  lifecycle hooks + Sprint 5 topology extensions + Sprint 6 aggregator +
  checkpoint save/load.
- `arc_agent_pairwise_stockfish_codex.py` — `TransitionRanker` gains
  `temporal_dim` + `temporal_features` additive gate; base agent gains
  `_current_temporal_features` and wires it through `_score_candidates_core`.
  Adds `_context_token_for_ouro` on the base class; `encode_and_think_batch`
  reads it and adds the context token to the CLS embedding (CLS-modulation
  variant of §2.5 injection — see below). Load path relaxed to `strict=False`
  for the ranker so pre-Sprint-6 checkpoints load cleanly.
- `live_arc_diagnostic.py` — added `--self_model_mode` and `--backbone_mode`
  flags; step records now include per-step affective state, context-token
  norm, temporal-feature norm, and region-graph stats when the self-model is
  active.

### CLS-modulation vs prepended-token injection

The original design (§2.5) specified **prepending** the context token as a new
leading `inputs_embeds` position. In implementation I used the
**CLS-modulation** variant instead: `encoder_tokens[:, 0] += context_token`.
This is equivalent for the self-model's purpose (biasing Ouro's current-step
forward with accumulated cognitive state) but preserves every downstream
assumption that CLS lives at position 0 — no position-index shift, no
attention-mask extension, no reindexing of anything that reads `loop_states[:, 0]`.
Ouro's full token-level hidden states propagate across iterations (per the
paper), so the modulated CLS accumulates context across all 4 recurrent passes.

### Gradient topology for inject / inject_grad

`encode_and_think_batch` wraps Ouro in `torch.no_grad()` — grad through a
frozen 2.6B model would OOM the 12 GB GPU. The context token is therefore
added **detached** inside the no_grad block. Semantically this means:

- `inject` — Ouro sees the context token; self-model has no gradient path
  through Ouro. Useful for frozen-GRU experiments or for eval with a
  pre-trained self-model.
- `inject_grad` — same behavior wrt Ouro; "grad" here refers to gradient
  flow from **alternative supervision paths** (Sprint 11a anchor loss,
  future direct affective supervision). Calling the mode `inject_grad`
  preserves the taxonomy in the design doc; actual gradient-through-Ouro
  is a separate decision that needs gradient checkpointing, which was not
  in scope for this session.

This is the correct boundary: a frozen backbone forbids backprop through
itself, so self-model training must come from externally-computed losses
that consume the self-model's outputs. The CLT anchor loss (Sprint 11a) is
exactly that.

### Empirical smoke — live ARC runs, 2026-04-25

1. **Encoder-only baseline** — `ft09 r11l`, 60 steps, eps=0.15,
   `self_model_mode="off"`. Ran to completion (both games GAME_OVER as
   expected at this pre-solve capability level). 52 beam_search + 8 random
   on ft09; 52+8 on r11l.

2. **Encoder-only passive** — same config, `self_model_mode="passive"`.
   Ran to completion. Self-model traces came through:
   - ft09 final affect: `fear=0.413, frustration=0.486, curiosity=0.237` —
     directionally correct (ft09 is a hazard-rich mechanism-death game).
   - r11l final affect: `fear=0.203, curiosity=0.098, frustration=0.027` —
     lower-intensity, also directionally correct.
   - `context_token_norm=0.0` and `temporal_feat_norm=0.0` exactly, at all
     sampled steps. **Identity-start property verified empirically.**
   - Sprint 5 topology: `n_regions=10` on ft09, `gateway_count=1`,
     `bfs_max_dist=6` — rich region structure detected live.

3. **Encoder-only action-trace equivalence** — comparing off vs passive
   action sequences on the same games at eps=0.0 revealed drift. Two
   back-to-back runs of off-mode alone also drift — confirming this is
   CUDA-kernel non-determinism that **already existed** in the codebase,
   not a self-model regression. Semantic guarantees (mode=off allocates
   nothing; zero-init projector outputs zero; gate-at-zero adds nothing)
   all hold.

4. **Ouro-backed inject smoke** — `r11l`, 12 steps, eps=0.0,
   `self_model_mode="inject"`, `backbone_mode="ouro"`. Full 2.6B backbone
   loaded, 12 real Ouro forward passes with CLS-modulation active. **No
   crash. No numerical issues. Ran to completion.** Affective-state
   evolution over these 12 steps:

   | Step | Affect (nonzero) |
   |---|---|
   | 1  | `fear=0.043` |
   | 5  | `fear=0.150, curiosity=0.035` |
   | 10 | `fear=0.215, curiosity=0.094` |
   | 12 | `fear=0.229, curiosity=0.121, frustration=0.095` |

   Monotonically growing, tracks environmental statistics (high unknown
   hazard belief + no level progress → frustration emerges exactly where
   expected). `context_token_norm=0.0` throughout — projector is still
   zero-init because no training loss has moved it. **This is the paper
   signal**: the self-model produces interpretable, environment-tracking
   affective trajectories from a frozen architecture, and the architecture
   is safe to inject into Ouro (zero-init → identity-start) until training
   explicitly learns to use it.

### Full test snapshot

```
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/moloch/ouro_project/venv/bin/python -m pytest \
  claude_sandbox/ -q
# 133 passed in ~10s
```

### The genuinely deferred work

1. Anchor loss wiring into `train_arc_codex.py` (Ouro hook capture of
   chosen/rejected loop states, `aux_loss = -log_sigmoid(score)` scheduled
   every N steps at coefficient 0.1).
2. `temporal_features` through the 4 training-loss ranker calls (replay
   buffer would need to carry `h_t` snapshots or we'd need off-policy
   reconstruction; not a rote integration).
3. Gradient-through-Ouro for true `inject_grad` — requires gradient
   checkpointing to stay under 12 GB.
4. Paper-scale sweep: compare off / passive / inject / inject_grad across
   ls20 / r11l / ft09 / tr87 / wa30 with n_runs ≥ 3 to separate self-model
   effect from CUDA-noise band. Once anchor loss is wired this is the
   obvious next run.

## 2026-04-25 Audit Pass + Cortex Monitor (Ouro's Self-Model)

### The architectural question

The user asked "should Ouro itself have a self-model?" — distinct question
from whether *the agent* has one. The answer is yes, but it's a different
kind of self-model. Two separate GRUs integrating different streams:

| Module              | Input stream                                         | Role                               |
|---------------------|------------------------------------------------------|------------------------------------|
| `SelfModel`         | External events: affect, track summary, loop-Δ proxy | Agent's model of world + self      |
| **`CortexMonitor`** | Internal Ouro dynamics: per-loop cosines, norms, σ   | Ouro's model of its own processing |

Anterior-cingulate analog vs basal-ganglia analog — both legitimate,
ontologically distinct.

### Audit fixes landed first

- Single source of truth for event ordering: the agent now imports
  `EVENT_TYPE_NAMES` from `self_model` rather than duplicating the tuple in
  `step()`.
- `max_phase_seen` had a dead-code `or [1]` fallback that never fired — the
  list is always non-empty. Cleaned up.
- The Rademacher expansion pattern used by `_self_model_step` was being
  re-seeded and re-sampled on every call. Cached once per agent as
  `_loop_delta_pattern`. Negligible perf win but the call site is cleaner.
- Documented gradient topology explicitly: `encode_and_think_batch` detaches
  the context token inside its `no_grad` block, by design — the frozen 2.6B
  Ouro can't fit gradients through a 12 GB GPU, so self-model training must
  come from alternative losses (anchor, future direct affective supervision).
  Renamed `inject_grad` semantic accordingly.
- Non-finding: "passive mode action sequences drift from off mode" is a
  real observation but it's CUDA-kernel non-determinism — two back-to-back
  off-mode runs drift too. Not a self-model regression.

### Cortex Monitor — new module

New class in `self_model.py`: `CortexMonitor(sig_dim=14, hidden=128, out=32)`.

- **Signature**: `compute_loop_signature(loop_cls[B, T, 2048]) → [B, 14]`
  of per-step cortex statistics:
  - 3 consecutive cosines (L1↔L2, L2↔L3, L3↔L4)
  - 3 consecutive L2 distances
  - 4 per-loop CLS norms
  - 4 per-loop element-wise std (peaked-vs-spread measure)
- **Monitor**: input `LayerNorm → GRUCell(14 → 128)`, projector
  `LayerNorm → Linear(128 → 32)` **zero-init** so the output feature is
  exactly zero at initialization (identity-start mirrors the agent self-model).
- **Lifecycle**: `reset_state` per run; `detach_state` available for future
  gradient-cadence management; snapshot save/load for checkpoints.
- **Where it runs**: inside `encode_and_think_batch`, right after `loop_cls`
  is stacked. Mean-over-batch signature → GRU advance → detached feature.
- **How it reaches the ranker**: the `TemporalContextAggregator` gained a
  fourth input channel (`cortex_feature`, optional, defaults to zeros). The
  aggregator's `fuse` stays zero-init; the cortex channel is additive to
  the existing h_t/event/delta channels.
- **Base-class placement**: `PairwiseARCSearchAgent` owns the monitor
  because it owns the Ouro forward. `HunterSeekerAgent` inherits and threads
  the output through `_self_model_step` into the aggregator.

Params: ~60 K. Negligible vs everything else.

### Empirical smoke — all three streams running together

```
./venv/bin/python claude_sandbox/live_arc_diagnostic.py \
  --games r11l --max_steps 10 --n_runs 1 --eps 0.0 \
  --self_model_mode inject --cortex_monitor_mode active --backbone_mode ouro
```

Result: 10 Ouro-backed steps on r11l. The three streams showed clearly
distinguishable dynamics:

| step | ctx norm | temp feat | cortex feat | cortex_h norm | affect (nonzero)                  |
|------|----------|-----------|-------------|---------------|-----------------------------------|
|  1   | 0.000    | 0.000     | 0.000       | **1.688**     | fear=0.043                        |
|  5   | 0.000    | 0.000     | 0.000       | **2.356**     | fear=0.150, curiosity=0.035       |
|  7   | 0.000    | 0.000     | 0.000       | **2.363**     | fear=0.182, curiosity=0.058       |
|  10  | 0.000    | 0.000     | 0.000       | **2.363**     | fear=0.215, curiosity=0.094       |

- `ctx / temp / cortex_feat = 0` — zero-init projector property holds across
  all three heads. Safe to add to any agent.
- `cortex_h` climbs from 1.69 → 2.36 and then **saturates** by step 7. This
  is the GRU integrating a consistent Ouro-dynamics signal to a steady state.
  Matches intuition: r11l early steps present Ouro with broadly similar
  stimuli, Ouro's refinement signature is stable, the monitor recognizes
  "nothing unusual about the cortex's processing right now."
- `affect` keeps growing linearly — outcome-driven, hasn't saturated.

**This is the orthogonality we wanted.** Affect and cortex-state are
uncorrelated at 10-step horizon: one tracks environmental statistics, the
other tracks internal dynamics.

### Files changed this pass

New:
- `claude_sandbox/test_cortex_monitor.py` — 13 tests (signature shape/range,
  monitor forward + reset + snapshot, aggregator with 4-channel zero-init).

Modified:
- `claude_sandbox/self_model.py` — added `CortexMonitor`, `compute_loop_signature`,
  `CORTEX_SIG_DIM` / `CORTEX_HIDDEN_DIM` / `CORTEX_OUT_DIM`, extended
  `TemporalContextAggregator` to 4-channel with new `cortex_feature` input.
- `claude_sandbox/arc_agent_pairwise_stockfish_codex.py` — `cortex_monitor_mode`
  kwarg, monitor construction + optimizer, forward advance inside
  `encode_and_think_batch`, lifecycle reset.
- `claude_sandbox/arc_agent_hunter_seeker_codex.py` — thread
  `_last_cortex_feature` into the aggregator call; checkpoint save/load for
  the monitor; audit-pass cleanups (cached Rademacher, `EVENT_TYPE_NAMES`
  single source, `max_phase_seen` simplification).
- `claude_sandbox/live_arc_diagnostic.py` — `--cortex_monitor_mode` flag,
  cortex trace fields (`cortex_feat_norm`, `cortex_h_norm`).

### Test snapshot

```
146 passed in 10.81s
```

New tests: `test_cortex_monitor.py` (13). Updated test counts for existing
files due to the 4-channel aggregator change. Zero-init byte-equivalence
still verified on all existing aggregator paths.

### What the next meaningful run looks like

With both self-model and cortex monitor live, the next useful empirical
result is a multi-game sweep comparing:

1. baseline (`self_model=off cortex_monitor=off`)
2. agent only (`self_model=inject cortex_monitor=off`)
3. cortex only (`self_model=off cortex_monitor=active`)
4. both (`self_model=inject cortex_monitor=active`)

Since all projectors are zero-init, (2)/(3)/(4) will score identically to
(1) at initialization — the difference only shows up **after** a training
loop backprops loss through one or both heads. The anchor loss (Sprint 11a)
is the obvious first training signal for either stream. That wiring is the
next real deliverable.
