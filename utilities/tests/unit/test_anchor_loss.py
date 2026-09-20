"""Unit tests for evaluator_core.anchor_loss (Sprint 11a).

Uses synthetic loop-state fixtures — does not require Ouro. The evaluator
checkpoint load is exercised optionally when the canonical checkpoint exists
locally.
"""
from __future__ import annotations

import os

import pytest
import torch

from evaluator_core.anchor_loss import (
    DEFAULT_EVALUATOR_CHECKPOINT,
    FrozenCLTAnchor,
    synth_loop_states,
)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke: instantiation, frozen defaults
# ─────────────────────────────────────────────────────────────────────────────


def test_anchor_instantiates_without_checkpoint():
    anchor = FrozenCLTAnchor()
    # Evaluator should be frozen by default
    for p in anchor.evaluator.parameters():
        assert not p.requires_grad


def test_anchor_eval_mode_by_default():
    anchor = FrozenCLTAnchor()
    assert not anchor.evaluator.training


def test_anchor_unfrozen_when_freeze_false():
    anchor = FrozenCLTAnchor(freeze=False)
    # At least some params should be trainable
    assert any(p.requires_grad for p in anchor.evaluator.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# Score / loss shape
# ─────────────────────────────────────────────────────────────────────────────


def test_score_returns_scalar_logit_per_batch_item():
    anchor = FrozenCLTAnchor()
    batch, seq_len = 3, 16
    chosen = synth_loop_states(batch, seq_len, seed=0)
    rejected = synth_loop_states(batch, seq_len, seed=1)
    mask = torch.ones(batch, seq_len)
    score = anchor.score(chosen, mask, rejected, mask)
    assert score.shape == (batch, 1)


def test_anchor_loss_is_scalar_by_default():
    anchor = FrozenCLTAnchor()
    chosen = synth_loop_states(2, 8, seed=0)
    rejected = synth_loop_states(2, 8, seed=1)
    mask = torch.ones(2, 8)
    loss = anchor.anchor_loss(chosen, mask, rejected, mask)
    assert loss.dim() == 0


def test_anchor_loss_per_sample_under_none_reduction():
    anchor = FrozenCLTAnchor()
    chosen = synth_loop_states(3, 8, seed=0)
    rejected = synth_loop_states(3, 8, seed=1)
    mask = torch.ones(3, 8)
    loss = anchor.anchor_loss(chosen, mask, rejected, mask, reduction="none")
    assert loss.shape == (3, 1)


def test_anchor_loss_is_nonnegative():
    """log_sigmoid(x) ≤ 0, so -log_sigmoid(x) ≥ 0 always."""
    anchor = FrozenCLTAnchor()
    for seed in range(5):
        chosen = synth_loop_states(2, 8, seed=seed)
        rejected = synth_loop_states(2, 8, seed=seed + 100)
        mask = torch.ones(2, 8)
        loss = anchor.anchor_loss(chosen, mask, rejected, mask, reduction="mean")
        assert loss.item() >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Gradient flow — gradients reach caller inputs but NOT evaluator weights
# ─────────────────────────────────────────────────────────────────────────────


def test_anchor_loss_gradients_reach_input_states_but_not_evaluator():
    anchor = FrozenCLTAnchor()
    batch, seq_len = 2, 8
    chosen = [t.detach().clone().requires_grad_(True) for t in synth_loop_states(batch, seq_len, seed=0)]
    rejected = [t.detach().clone().requires_grad_(True) for t in synth_loop_states(batch, seq_len, seed=1)]
    mask = torch.ones(batch, seq_len)

    loss = anchor.anchor_loss(chosen, mask, rejected, mask)
    loss.backward()

    # Gradients should have reached the input tensors
    assert all(t.grad is not None and t.grad.abs().sum() > 0 for t in chosen)
    assert all(t.grad is not None and t.grad.abs().sum() > 0 for t in rejected)

    # No gradient should have accumulated on evaluator params (they are frozen)
    for p in anchor.evaluator.parameters():
        assert p.grad is None


# ─────────────────────────────────────────────────────────────────────────────
# Preference accuracy helper is bounded in [0, 1]
# ─────────────────────────────────────────────────────────────────────────────


def test_score_disables_cudnn_during_evaluator_forward_but_restores_after():
    """pre_ladder_audit_backlog_final §"Quick-ladder anchor issue":
    cuDNN refuses RNN backward in eval mode, but the frozen evaluator
    contains a GRU. The score() call must wrap the forward in
    cudnn.flags(enabled=False) so anchor backward through the frozen
    GRU works on GPU; the surrounding cuDNN state must be restored.

    Test verifies (CPU-safe):
      · the wrapper actually toggles cudnn during the forward
      · cuDNN's enabled-state is restored to whatever it was before
      · gradients still reach the input states (the substantive contract)
      · evaluator parameters stay frozen and in eval mode
    """
    anchor = FrozenCLTAnchor()
    pre = torch.backends.cudnn.enabled
    pre_training = anchor.evaluator.training

    # Capture the cudnn state observed inside the forward path. Wrap the
    # evaluator with a hook that records `torch.backends.cudnn.enabled`
    # at forward-time; that's the moment the wrapper must have flipped it.
    observed: dict = {}

    def _hook(_module, _inputs, _output):
        observed["cudnn_enabled_during_forward"] = torch.backends.cudnn.enabled

    handle = anchor.evaluator.register_forward_hook(_hook)
    try:
        chosen = [t.detach().clone().requires_grad_(True) for t in synth_loop_states(2, 4, seed=0)]
        rejected = [t.detach().clone().requires_grad_(True) for t in synth_loop_states(2, 4, seed=1)]
        mask = torch.ones(2, 4)
        loss = anchor.anchor_loss(chosen, mask, rejected, mask)
        loss.backward()
    finally:
        handle.remove()

    assert observed.get("cudnn_enabled_during_forward") is False, (
        "evaluator forward must run with cuDNN disabled to permit "
        "RNN backward through the frozen eval-mode GRU"
    )
    # The flag context manager must restore prior state on exit.
    assert torch.backends.cudnn.enabled == pre, (
        "cudnn.enabled must be restored to its pre-call value"
    )
    # Eval mode preserved (no dropout activation, no train-mode side effects).
    assert anchor.evaluator.training == pre_training
    # Frozen-weights contract preserved: parameters didn't accumulate grad.
    for p in anchor.evaluator.parameters():
        assert p.grad is None
    # Substantive contract: gradient reaches the inputs.
    assert all(t.grad is not None and t.grad.abs().sum() > 0 for t in chosen)
    assert all(t.grad is not None and t.grad.abs().sum() > 0 for t in rejected)


def test_preference_accuracy_bounded():
    anchor = FrozenCLTAnchor()
    chosen = synth_loop_states(8, 12, seed=0)
    rejected = synth_loop_states(8, 12, seed=1)
    mask = torch.ones(8, 12)
    acc = anchor.preference_accuracy(chosen, mask, rejected, mask)
    assert 0.0 <= acc <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Optional: exercise real checkpoint load if available
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not os.path.exists(DEFAULT_EVALUATOR_CHECKPOINT),
    reason=f"{DEFAULT_EVALUATOR_CHECKPOINT} not present — skipping real-load test",
)
def test_real_evaluator_checkpoint_loads_cleanly():
    anchor = FrozenCLTAnchor(checkpoint_path=DEFAULT_EVALUATOR_CHECKPOINT)
    # After loading, the evaluator should still be frozen
    for p in anchor.evaluator.parameters():
        assert not p.requires_grad
    # And produce nontrivial scores on synthetic input (exact value not
    # meaningful since inputs are random, but the forward pass must work).
    chosen = synth_loop_states(2, 16, seed=7)
    rejected = synth_loop_states(2, 16, seed=11)
    mask = torch.ones(2, 16)
    score = anchor.score(chosen, mask, rejected, mask)
    assert torch.isfinite(score).all()
