<!-- Imported from `claude_sandbox/design/ablation_ladder.md` during Hunter-Seeker state consolidation on 2026-05-14. -->
<!-- Original SHA256: 2d2cd9918ec97c5cf7250c8f12620625c772e472863845e4a9e4c96ee2b9ce63; original line count: 233. -->

# Ablation Ladder — what to run before claiming the new stack helps

GPT-onion §10 sketched a 6-step micro-run sequence to isolate component
contributions. This file is the merged version after a back-and-forth
with GPT-onion: my additions of **Step 0** (explicit historical floor)
and **Step 3.5** (anchor-alone drift test); GPT's additions of
**Step 2** (passive-introspection canary against CUDA non-determinism)
and **Step 6** (anchor + AttnRes pairwise interaction).

Total: 8 steps. Designed so each adjacent pair differs by exactly one
factor, giving every score delta a single attributable cause.

## The eight steps

Every step uses the same `--games`, `--max_steps`, `--n_runs`, `--eps`
so deltas come from the configuration, not the workload. Pin
`--running_checkpoint claude_sandbox/checkpoints_running/ladder_step_N.pt`
per step so the rolling chain doesn't bleed across configurations.
Pre-flight VRAM calibration (see end of file) before any step that
sets `--anchor_train_every > 0`.

### Step 0 — v17b historical baseline

```
python -m claude_sandbox.train_arc_codex \
  --baseline_checkpoint checkpoints_running/frozen_sprint_4_overnight_noreplay.pt \
  --backbone_mode ouro --self_model_mode off --cortex_monitor_mode off \
  --loop_pooler_kind gru
```

**Probes**: Sprint-4 frozen historical floor. Anything later in the
ladder that doesn't beat this hasn't earned its keep.
**Look for**: levels-completed totals; that's the only number the
rest of the ladder has to beat.

### Step 1 — current minimal baseline

```
... --backbone_mode encoder_only --self_model_mode off
    --cortex_monitor_mode off --loop_pooler_kind gru
```

**Probes**: harness sanity, dependencies, no-Ouro inference path.
**Look for**: zero crashes, finite ranker losses. If this fails,
nothing else matters.

### Step 2 — passive introspection canary

```
... --backbone_mode encoder_only --self_model_mode passive
    --cortex_monitor_mode active --loop_pooler_kind gru
```

**Probes**: do passive cognitive heads byte-equivalently match Step 1
within the first ~50 steps? Zero-init guarantees say yes; CUDA
non-determinism (flagged in the prior session summary) is the only
plausible perturbation source. This step empirically rules in/out
the byte-equivalence claim.

**Look for**:
- Action-sequence prefix match against Step 1 over the first ~30 steps.
  Drift earlier than that = real perturbation, not CUDA noise.
- `self_model.aggregator_fuse_weight_norm` should stay near zero
  early; it WILL drift off zero as the event-prediction and
  signature-prediction losses train. Past ~100 steps the canary loses
  its grip on byte-equivalence — by design — because the cognitive
  heads then start contributing real signal through the ranker.

This step is the *short-horizon* sanity check. It's NOT a full-run
guarantee that passive == off; it's a confirmation that no SECRET
inference perturbation snuck into the passive path.

### Step 3 — Ouro baseline (GRU pooler, no cognitive heads)

```
... --backbone_mode ouro --self_model_mode off
    --cortex_monitor_mode off --loop_pooler_kind gru
```

**Probes**: byte-equivalent reproduction of the pre-Sprint-11 Ouro
pipeline. Should match Step 0's score floor on the same checkpoint
modulo CUDA non-determinism.
**Look for**: score parity with Step 0; substantially worse means
recent stack work has regressed Ouro inference quality.

### Step 3.5 — Ouro + anchor only (encoder unfrozen)

```
... --backbone_mode ouro --self_model_mode off
    --cortex_monitor_mode off --loop_pooler_kind gru \
    --anchor_train_every 100 --anchor_coefficient 0.1 --anchor_batch_size 4 \
    --unfreeze_encoder_after_partial_load
```

**Probes**: does the anchor loss alone prevent encoder drift? This is
the cleanest single-variable test of Sprint-11a's purpose.
The encoder MUST be trainable here — otherwise the anchor is a no-op
and the harness will print a warning. If you used a partial-load
fallback, pass `--unfreeze_encoder_after_partial_load`.

**Look for**:
- `anchor.successes / anchor.attempts` ratio > 0.5. Lower = buffer
  not filling fast enough or batch size too large for VRAM.
- `anchor.current_batch_size` stays at the ceiling (= adaptive
  backoff didn't fire). If it dropped to 1, you have VRAM pressure;
  drop the ceiling for the real sweep.
- `anchor.loss_ema` decreasing across steps; rising = encoder
  drifting AWAY from anchor space.
- End-of-run encoder cosine vs v17b (`encoder_drift_check.py`).
  Anchor-on should keep cosine ≥ 0.5; anchor-off historically
  collapsed to ≈ 0.

### Step 4 — Ouro + AttnRes pooler only

```
... --backbone_mode ouro --self_model_mode off
    --cortex_monitor_mode off --loop_pooler_kind attn_res
```

**Probes**: pooler architecture in isolation. Identity-init guarantees
match to Step 3 at run start; divergence comes from training the
AttnRes gate.
**Look for** (`attn_res_pooler` block of `measurement_summary`):
- `attn_l[1-4]_mean` stays away from one-hot. Healthy: roughly 0.25
  each at init, drifts to a non-degenerate distribution.
- `attn_entropy` > 0.5 (one-hot → 0).
- `attn_argmax_count_l[1-4]` no single loop > 80%. If l4 wins ≥ 95%,
  AttnRes has degenerated to a final-loop reader (a pathology — at
  that point it's strictly worse than the GRU pooler).

### Step 5 — Ouro + self-model + cortex monitor only

```
... --backbone_mode ouro --self_model_mode inject_aux_grad
    --cortex_monitor_mode active --loop_pooler_kind gru
```

**Probes**: cognitive-head contribution independent of pooler choice
and anchor. The aggregator's zero-init fuse keeps temporal_features at
zero until the event-prediction and ranker losses move it.
**Look for** (`self_model` block):
- `aggregator_fuse_weight_norm` non-zero by mid-run; if still ≈ 0,
  the cognitive heads are inert.
- `self_model_loss_ema` non-trivially decreasing.
- `self_model_gru_grad_norm` non-zero. Zero = ranker loss isn't
  reaching the GRU through the off-policy reconstruction path.
- Affective-state traces in `dump_events_dir`: fear/curiosity/
  frustration should respond to events, not flat.

### Step 6 — Ouro + anchor + AttnRes (no cognitive heads)

```
... --backbone_mode ouro --self_model_mode off
    --cortex_monitor_mode off --loop_pooler_kind attn_res \
    --anchor_train_every 100 --anchor_coefficient 0.1 --anchor_batch_size 4 \
    --unfreeze_encoder_after_partial_load
```

**Probes**: pairwise interaction. An anchor-trained encoder might
produce loop dynamics where the per-iteration signal distribution
shifts; the AttnRes pooler should track that shift via its softmax
weights. This step tells you whether they cooperate or fight.
**Look for**: AttnRes attention-weight distribution from Step 4 and
anchor cosine-vs-v17b from Step 3.5 — does one degrade in the
combined setting? Specifically:
- `attn_l[1-4]_mean` distribution should NOT diverge wildly from
  Step 4's. If it does, anchor is reshaping loop signal in a way
  AttnRes hasn't learned to track.
- `anchor.loss_ema` should reach a similar floor as Step 3.5. If
  meaningfully higher, AttnRes's training is interfering with
  anchor's encoder shaping.

### Step 7 — Full system

```
... --backbone_mode ouro --self_model_mode inject_aux_grad
    --cortex_monitor_mode active --loop_pooler_kind attn_res \
    --anchor_train_every 100 --anchor_coefficient 0.1 --anchor_batch_size 4 \
    --unfreeze_encoder_after_partial_load
```

**Probes**: everything together. Score difference from Step 0 is what
the paper claims.
**Look for**: all diagnostics from Steps 3.5, 4, 5, 6 simultaneously.
The interpretive question — "if Step 7 beats Step 0 but the
cognitive-head diagnostics from Step 5 look healthy in isolation but
flat here, is the AttnRes pooler eating the gradient?" — is the kind
of analysis this ladder enables.

## Trusted/online ablation (orthogonal axis)

After Steps 0–7 at the default `--trusted_mix 0.75`, repeat Step 7 at:

- `--trusted_mix 0.5`  — balanced
- `--trusted_mix 0.25` — online-heavy

This is onion §8's ablation. If Step 7 results converge across mix
ratios, the new machinery is genuinely helping; if scores collapse at
0.25, the system is mostly replaying trusted curriculum and the new
heads are window dressing.

## Pre-flight: VRAM calibration

Before launching any step that uses `--anchor_train_every > 0`, run a
60-step sacrificial check:

```
... --anchor_train_every 10 --anchor_batch_size 4 \
    --max_steps 60 --n_runs 1 --games ls20
```

Tail the log for `[Anchor-Train] OOM` lines. The adaptive backoff
halves the batch on each OOM and recovers after 8 successes — check
`anchor.current_batch_size` at the end of the run summary's
`measurement_summary` JSON. If it's stable at the ceiling, you're
fine. If it floored at 1 with frequent OOMs, drop
`--anchor_batch_size` to 2 for the real sweep.

## Comparison post-run

Each step's `measurement_summary` JSON is self-describing. Useful
top-line numbers to extract per step into a comparison table:

- `levels_completed` totals (the score)
- `failure_counts` (which layer is failing)
- `anchor.successes` / `anchor.attempts` ratio
- `anchor.loss_ema`
- `self_model.aggregator_fuse_weight_norm`
- `self_model.self_model_loss_ema`
- `attn_res_pooler.attn_entropy` (Step 4, 6, 7)
- Per-step `effective_confidence_pre_ouro` vs `effective_confidence`
  (mean across run) — diagnoses Ouro-confidence contribution to
  scoring (onion §6).
