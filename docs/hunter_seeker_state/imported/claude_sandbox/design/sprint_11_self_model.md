<!-- Imported from `claude_sandbox/design/sprint_11_self_model.md` during Hunter-Seeker state consolidation on 2026-05-14. -->
<!-- Original SHA256: 00e51d110c99f6bf557674ad7cf5385e84a9665e7e522803a11dd20469fb5463; original line count: 231. -->

# Sprint 11 — Self-model, affective state, context token, CLT anchor loss

This is the design that unlocks the basal-ganglia paper (paper #3 in the pipeline) and also fixes the Sprint-4 encoder-drift failure mode by providing a constitution-anchored gradient path.

Reference: `ouro_project_state.md` §10. This doc is the concrete engineering spec.

## 1. The two components that share the name "Sprint 11"

There are two distinct things lumped under "Sprint 11" in the current state doc. They must be implemented separately because they serve different roles:

| Part | Role | When it runs |
|---|---|---|
| **11b — Self-model** | Cognition-time state: a running GRU that integrates loop-state delta + affective state + track summary into a context token injected into each Ouro forward | Every agent step |
| **11a — CLT anchor loss** | Training-time regularizer: frozen CLT evaluator scores (chosen trajectory vs counterfactual trajectory) as an auxiliary loss that ties encoder/GRU gradients back to Ouro-compatible space | Selected training steps |

Implementation order: 11b first (self-contained, testable, additive), then 11a on top (needs somewhere meaningful for the anchor gradient to flow through).

## 2. 11b architecture

### 2.1 Inputs per step

```
loop_state_delta     ∈ R^64   (Ouro L1↔L4 cosine-difference projection, already exists in ranker path)
affective_state      ∈ R^8    (this sprint — see §3)
track_summary        ∈ R^16   (compact object-table summary — see §4)
                     ─────
total                ∈ R^88
```

### 2.2 Module

```python
class SelfModelGRU(nn.Module):
    def __init__(self, input_dim=88, hidden_dim=256, num_layers=1):
        super().__init__()
        self.cell = nn.GRUCell(input_dim, hidden_dim)
        self.register_buffer("hidden_dim_buf", torch.tensor(hidden_dim))
        self.reset_state()

    def reset_state(self, batch=1, device=None, dtype=None):
        self.h = torch.zeros(batch, self.cell.hidden_size, device=device, dtype=dtype)

    def step(self, z_t: torch.Tensor) -> torch.Tensor:
        self.h = self.cell(z_t, self.h)
        return self.h

    def detach_state(self):
        self.h = self.h.detach()
```

GRUCell not GRU: single-step, explicit state, easier to checkpoint and to reset at episode boundary.

### 2.3 Context token projector

```python
class ContextTokenProjector(nn.Module):
    def __init__(self, hidden_dim=256, ouro_dim=2048):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, ouro_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(h))
```

Zero-init the projector so on first forward pass, the context token is exactly zero and Ouro sees the stack it's always seen. This is the **identity-start property**: at t=0 of training, the self-model produces zero effect on Ouro. Every deviation from zero is a learned signal, gated by the ranker gradient.

### 2.4 Parameter count

- GRUCell(88→256): `3 * (88 + 256 + 1) * 256 ≈ 265K`
- LayerNorm(256): 512
- Linear(256→2048): 526K
- **Total: ~791K params.** Negligible vs 2.6B Ouro.

### 2.5 Injection into Ouro forward

```
Standard Ouro call:
  inputs_embeds = [cls, patch_0, ..., patch_N]

Self-model on:
  inputs_embeds = [context_token, cls, patch_0, ..., patch_N]
```

`context_token` is prepended as the first position. Attention mask extended by one on the left. Position IDs extended by one. This is identical in pattern to how `AutoModelForCausalLM` accepts `inputs_embeds` with a leading instruction-encoded embedding.

Critical detail: the context token is *not* an input that Ouro was trained on. But because the projector zero-inits, it starts at all-zeros, which is in-distribution (it's a plausible pad-like embedding). Learning moves it gradually.

### 2.6 State lifecycle

- Episode start (level 1 step 0 of each run) → `reset_state()`
- Level transition → `detach_state()` (carry memory forward but don't backprop through the boundary)
- Step → update `h_t = GRUCell(z_t, h_{t-1})`
- Episode end / death → reset

## 3. Affective state (8 signals)

Each is a float in [0, 1], decayed per step, excited by events. Definitions faithful to §10 of the state doc.

| Idx | Name         | Decay | Excitation rule |
|-----|--------------|-------|-----------------|
| 0   | joy          | 0.90  | LEVEL_COMPLETE: +clip(1 - steps_taken/max_steps, 0, 1). Clamp ≤1.0 |
| 1   | sadness      | 0.90  | DEATH: +0.6. Sterile run-end: +0.2 |
| 2   | curiosity    | 0.90  | + 0.5 * max(0, max_unknownness - 0.6) * sharpening_rate_ema |
| 3   | fear         | 0.85  | Hazard track within reachable set: +0.3 * hazard_belief_of_nearest |
| 4   | frustration  | 0.90  | Sustained high unknownness > 0.8 for > 10 steps: +0.05 per step |
| 5   | satisfaction | 0.92  | Low unknownness < 0.2 for > 10 steps: +0.05 per step. LEVEL_COMPLETE: +0.4 |
| 6   | surprise     | 0.80  | Δloop > 1.5 * ema: +0.5 * (Δloop/ema - 1.5) clipped to 1.0 |
| 7   | stress       | 0.95  | Avatar not found after 50 steps: +0.1. Repeated contradictions: +0.05 each |

Implementation as a dataclass with a `step()` method that takes an event bundle and returns the updated 8-float tensor. All coefficients and thresholds are structural (e.g. 0.6 = "high uncertainty threshold", 1.5 = "1.5× EMA spike"), not free tunables — §33.0 compliance.

## 4. Track summary (16 floats)

Compact summary of the object table for the GRU input. Computed once per step.

```
idx  0: n_visible_tracks / 20                    (normalized count, capped)
idx  1: mean_unknownness over visible tracks
idx  2: max_unknownness over visible tracks
idx  3: avatar_identified (0 or 1)
idx  4: exit_identified (0 or 1)
idx  5: n_hazard_tracks / 10                     (normalized count, capped)
idx  6: n_collectible_tracks / 10
idx  7: n_frontier_tracks / 10                   (from free-space topology)
idx  8: reachable_object_count / n_visible       (ratio, 0 if n_visible=0)
idx  9: bfs_mean_distance_normalized             (0..1)
idx 10: n_interactions_of_dominant_track / 100   (saturated)
idx 11: age_of_dominant_track / 200              (saturated)
idx 12: last_frame_change_rate (EMA)             (0..1)
idx 13: last_progress_signal                     (already exists, clip to [-1, 1])
idx 14: phase_progress_normalized                (phase / max_phase_seen_for_level)
idx 15: phase_recovery_active (0 or 1)
```

All values are scene-derivable. No new tracking state needed. Lives in `_compute_track_summary()` helper on the agent.

## 5. Training signal for 11b

### Primary (always-on once 11b is enabled)

Ranker loss flows back through:
- `ranker(..., cls_token_from_ouro, ...)` → Ouro loop states → Ouro `inputs_embeds` → `context_token` → `ContextTokenProjector` → `SelfModelGRU.h`

Because Ouro is frozen, gradients pass through but don't update Ouro. They DO update the projector and the GRU cell. This is the same gradient topology as the existing grid encoder → Ouro → ranker path, just with one more learnable module at the start.

### Auxiliary (Sprint 11a, added next)

See §7.

## 6. Four modes of operation

| Mode | self-model updated | context token computed | context token injected | context token gradient | Use case |
|---|---|---|---|---|---|
| `off` (default) | no | no | no | — | Existing behavior. Zero regression risk. |
| `passive` | yes | yes | no | — | Collect affective traces and h_t diagnostics without touching Ouro forward. Empirical baseline for "what would the self-model look like?" |
| `inject` | yes | yes | yes | no (detached) | Ouro sees context token, but no gradient flows back. Tests whether injection alone affects behavior. |
| `inject_grad` | yes | yes | yes | yes | Full Sprint 11b. Ranker gradient trains the self-model. |

`inject_grad` is the target. The other modes exist for clean ablation.

## 7. 11a — CLT evaluator anchor loss

### 7.1 Motivation

Sprint 4 encoder drift (cosine=0 vs v17b reference) happened because the encoder had no gradient anchor to Ouro-compatible space. Its direct losses (NextFramePredictor, SpatialClickPredictor, patch_color_head) optimize for "describe the frame well," not "produce tokens Ouro's attention can reason over." Sprint 11a adds that anchor.

### 7.2 Form

At selected training steps:
1. Sample a chosen trajectory segment from the current run (state_t → action_chosen → state_{t+1}).
2. Sample a counterfactual action from the rejected set (second-ranked candidate, or a same-state action that led to worse outcome in replay).
3. Encode both trajectory segments into text-like sequences of Ouro loop states using a canonical trajectory encoding (described below).
4. Feed both through Ouro (no text, just `inputs_embeds` with the encoder's tokens + optional context token).
5. Extract the 4 loop states from both forward passes.
6. Pass to `PairwiseEvaluator(chosen_states, chosen_mask, rejected_states, rejected_mask)` → scalar preference score.
7. Use `-log_sigmoid(score)` as aux loss (chosen > rejected).
8. Backprop into encoder, GRU, projector. **Not into Ouro** (frozen). **Not into evaluator** (frozen — this is the anchor).

### 7.3 Canonical trajectory encoding

The CLT evaluator was trained on HH-RLHF text pairs, so it expects loop states over a natural-language-like sequence. We don't have natural language here. Two workable approaches:

**Option A — encoder tokens directly.** The grid encoder already outputs a sequence of tokens (patches). Use them as the Ouro input, capture loop states, pass to evaluator. The evaluator has never seen these token distributions, but if encoder outputs are Ouro-compatible (the whole point of the anchor), they should produce structured loop-state trajectories that the evaluator's GRU can read.

**Option B — translate trajectory to text.** Serialize (state, action, outcome) into a short textual description, tokenize, feed to Ouro as text. This is the path of least resistance for evaluator compatibility but adds a tokenization step per anchor call.

Pick Option A initially. It's the one that validates the "encoder in Ouro-space" claim by construction: if the evaluator's loop-state GRU produces a meaningful preference signal over grid-encoder tokens, the encoder is ipso facto Ouro-compatible.

If Option A's preference gradient is noisy or sign-flipped, fall back to Option B.

### 7.4 Budget

Anchor loss is expensive — each call is two extra Ouro forwards + one evaluator forward. Run it at most 1 in every N training steps (N=8 initial). Coefficient starts small (0.1) to avoid dominating the primary ranker loss. Tune later.

### 7.5 What success looks like

- Encoder cosine distance from v17b reference stops growing. Ideally shrinks during Sprint 5/6 work.
- Loop-state signature (L1↔L2 > L2↔L3 > L3↔L4 monotone) remains clean across training.
- Validation: unfreezing the encoder during a small training run should no longer produce cosine-0 drift within a few hundred steps.

## 8. What this does NOT do yet

These are explicitly deferred to later sprints/papers, named here so they don't sneak in:

- **Explicit affective loss.** No loss function trains affective state directly. It's driven by deterministic event triggers. A future extension could add a consistency loss (the GRU's internal representation of affect should correlate with the explicit affective state).
- **Contrastive self-model loss over outcome-tagged trajectories.** Future work (basal ganglia paper §3).
- **Loop-state consistency loss.** Future work, tied to Sprint 12 (sleep Stage 2).
- **Affective state as input to action selection.** The current design lets Ranker + Ouro use `h_t` through the context token, but nothing explicitly reads the 8-dim affective vector as a policy input. Could add.
- **Basal-ganglia-style candidate tournament.** Arafel-at-the-action-level — use the frozen CLT evaluator to do pairwise candidate scoring at action-selection time, not just as a training anchor. Powerful, but separate from the self-model and best designed after 11b is stable.

## 9. Integration checklist

- [ ] `claude_sandbox/self_model.py` — standalone module with tests.
- [ ] `claude_sandbox/arc_agent_hunter_seeker_codex.py` — add `self_model_mode` kwarg, wire into `__init__`, `reset_for_new_game`, `step()`, and checkpoint save/load.
- [ ] `claude_sandbox/arc_agent_pairwise_stockfish_codex.py` — when `self_model_mode in {"inject", "inject_grad"}`, prepend `context_token` to `inputs_embeds` before Ouro forward. Extend attention mask.
- [ ] `claude_sandbox/test_self_model.py` — new test file. 8+ tests: GRU init, GRU step shape, projector zero-init property, affective state decay, affective state excitation, track summary ranges, end-to-end state propagation across N steps, reset semantics.
- [ ] `claude_sandbox/test_self_model_integration.py` — gated integration test: run HunterSeekerAgent with `self_model_mode="passive"` for 10 mock steps, verify h_t and affective state update as expected, verify Ouro forward produces identical output as mode="off".
- [ ] `claude_sandbox/CLAUDE_SESSION_SUMMARY.md` — log the change.
- [ ] Anchor loss: `claude_sandbox/anchor_loss.py` — helper that wraps PairwiseEvaluator calls, produces the aux loss. Added to `train_arc_codex.py` under a flag.

## 10. Success criteria for the sprint as a whole

1. `self_model_mode="off"` runs are byte-equivalent to current sandbox runs (zero regression).
2. `self_model_mode="passive"` runs produce interpretable affective traces (joy spikes on level complete, fear spikes near hazards, etc.).
3. `self_model_mode="inject_grad"` runs show non-zero context-token magnitude growing over training and non-trivial ranker-loss reduction beyond the off baseline. Measurement: if this runs for ≥ 500 steps and `context_token.norm()` stays at 0, training isn't producing a useful signal.
4. Anchor loss (11a) keeps encoder cosine distance from v17b bounded during subsequent unfreeze experiments.

Paper-level: the Sprint 11 paper writes itself once (2) and (3) are demonstrated — "closing the loop between Ouro's relational preference encoding and a learned affective state produces more stable and context-sensitive action selection."
