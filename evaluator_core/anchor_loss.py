"""Sprint 11a — CLT evaluator anchor loss.

Why: the Sprint-4 encoder drift (cosine=0 vs v17b) happened because the
encoder had no gradient anchor pulling it toward Ouro-compatible space. Its
direct losses (NextFramePredictor, SpatialClickPredictor, patch_color_head)
optimize for "describe the frame well," not "produce tokens Ouro's attention
can reason over." Sprint 11a adds that anchor.

How: at selected training steps, we score (chosen trajectory, rejected
trajectory) pairs through the frozen CLT evaluator. The evaluator expects
lists of 4 Ouro loop-state tensors per sample — we produce those via Ouro
forward on encoder tokens (+ optional context token). The loss is
``-log_sigmoid(score)`` for the preference "chosen > rejected." The anchor
is the evaluator + frozen Ouro; backprop updates encoder, projector, and
self-model GRU.

What this file is: the *scoring* primitive. Gathering chosen/rejected
trajectory pairs and running Ouro live is the training harness's job (see
``train_arc.py`` integration for Sprint 11a). Keeping the scoring
primitive separate means it is independently testable with synthetic inputs
and the evaluator load path is exercised without a full training setup.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


from evaluator_core.pairwise_evaluator import PairwiseEvaluator


# Default checkpoint — the epoch-2 Pairwise evaluator is the canonical 95.2%
# pairwise, 2.7×-better-at-math variant (state doc §13). Epochs 4-5 overfit.
DEFAULT_EVALUATOR_CHECKPOINT = "artifacts/checkpoints/evaluator/pairwise_epoch2.pt"


class FrozenCLTAnchor(nn.Module):
    """Wraps the frozen pairwise evaluator + exposes a clean anchor-loss API.

    The evaluator parameters are frozen by default — this module is an
    *anchor*, not a learnable module. Gradients flow through the evaluator's
    forward pass to reach caller inputs (encoder, self-model, etc.), but the
    evaluator's own weights never update.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        freeze: bool = True,
    ):
        super().__init__()
        self.evaluator = PairwiseEvaluator()
        self._freeze = freeze

        if checkpoint_path is not None:
            self.load_evaluator_checkpoint(checkpoint_path, map_location=device)

        if device is not None:
            self.to(device)

        if self._freeze:
            self._apply_freeze()

    def _apply_freeze(self) -> None:
        self.evaluator.eval()
        for p in self.evaluator.parameters():
            p.requires_grad = False

    def load_evaluator_checkpoint(
        self, path: str, map_location: Optional[torch.device] = None
    ) -> None:
        """Load pretrained evaluator weights. Checkpoint format matches
        ``evaluator_pairwise.py``: a dict with a ``'model_state_dict'`` entry,
        or a plain state dict."""
        state = torch.load(path, map_location=map_location or "cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            self.evaluator.load_state_dict(state["model_state_dict"])
        else:
            self.evaluator.load_state_dict(state)
        if self._freeze:
            self._apply_freeze()

    def score(
        self,
        chosen_states: Sequence[torch.Tensor],
        chosen_mask: torch.Tensor,
        rejected_states: Sequence[torch.Tensor],
        rejected_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return raw pairwise preference logit. Positive => chosen preferred.

        Shapes:
            chosen_states, rejected_states: list of 4 tensors each of shape
                [batch, seq_len, 2048] — the 4 Ouro loop states for the
                chosen / rejected sequence.
            chosen_mask, rejected_mask: [batch, seq_len] attention masks.

        Returns:
            [batch, 1] score tensor.
        """
        # The evaluator contains a GRU and stays frozen + in eval mode (anchor,
        # not learnable). On GPU, cuDNN refuses RNN backward in eval mode with:
        #   RuntimeError: cudnn RNN backward can only be called in training mode
        # Anchor loss explicitly needs gradient flow THROUGH the frozen GRU
        # into the evaluator's inputs (encoder/Ouro tokens upstream). Disable
        # cuDNN for just this forward so the non-cuDNN backward path is used;
        # evaluator weights remain frozen, dropout stays off (eval mode), and
        # only the GRU's RNN backward kernel changes. CPU paths are unaffected.
        with torch.backends.cudnn.flags(enabled=False):
            return self.evaluator(
                list(chosen_states), chosen_mask, list(rejected_states), rejected_mask
            )

    def anchor_loss(
        self,
        chosen_states: Sequence[torch.Tensor],
        chosen_mask: torch.Tensor,
        rejected_states: Sequence[torch.Tensor],
        rejected_mask: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """``-log_sigmoid(score)`` — standard pairwise preference loss.

        Drops to zero when score → +∞ (chosen trivially preferred).
        Grows large when score is negative (chosen dispreferred — anchor
        pulls encoder/GRU toward producing more-preferred chosen states).

        ``reduction`` follows PyTorch convention: ``'mean'`` | ``'sum'`` | ``'none'``.
        """
        score = self.score(chosen_states, chosen_mask, rejected_states, rejected_mask)
        # Preference target: chosen > rejected.
        per_sample = -F.logsigmoid(score)
        if reduction == "mean":
            return per_sample.mean()
        if reduction == "sum":
            return per_sample.sum()
        if reduction == "none":
            return per_sample
        raise ValueError(f"unknown reduction: {reduction!r}")

    def preference_accuracy(
        self,
        chosen_states: Sequence[torch.Tensor],
        chosen_mask: torch.Tensor,
        rejected_states: Sequence[torch.Tensor],
        rejected_mask: torch.Tensor,
    ) -> float:
        """Fraction of (chosen, rejected) pairs where score > 0. Diagnostic."""
        with torch.no_grad():
            score = self.score(chosen_states, chosen_mask, rejected_states, rejected_mask)
            return float((score > 0).float().mean().item())


def synth_loop_states(
    batch: int,
    seq_len: int,
    hidden_dim: int = 2048,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
) -> List[torch.Tensor]:
    """Produce a plausible-shape list of 4 loop-state tensors for tests.

    Not used in training — this is a fixture helper so unit tests don't need
    to stand up Ouro. Returns 4 tensors each [batch, seq_len, hidden_dim].
    """
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    states = []
    for _ in range(4):
        t = torch.randn(batch, seq_len, hidden_dim, generator=gen)
        if device is not None:
            t = t.to(device)
        states.append(t)
    return states


__all__ = [
    "DEFAULT_EVALUATOR_CHECKPOINT",
    "FrozenCLTAnchor",
    "synth_loop_states",
]
