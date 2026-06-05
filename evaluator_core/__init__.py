"""Evaluator and domain-transfer primitives."""

from evaluator_core.anchor_loss import (
    DEFAULT_EVALUATOR_CHECKPOINT,
    FrozenCLTAnchor,
    synth_loop_states,
)
from evaluator_core.pairwise_evaluator import (
    AttentionPool,
    PairwiseEvaluator,
    validate_hook_output,
)

__all__ = [
    "AttentionPool",
    "DEFAULT_EVALUATOR_CHECKPOINT",
    "FrozenCLTAnchor",
    "PairwiseEvaluator",
    "synth_loop_states",
    "validate_hook_output",
]
