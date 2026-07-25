# Evaluator Components

- **PairwiseEvaluator** (`pairwise_evaluator.py`): low-rank attention pooling,
  recurrent comparison over loop-state sequences, and scalar preference logit.
- **Hook validation** (`validate_hook_output`): shared guard for captured
  hidden-state/loop-state tensors.
- **FrozenCLTAnchor** (`anchor_loss.py`): frozen evaluator wrapper that lets
  gradients flow through evaluator reads into caller-owned representations while
  keeping evaluator parameters fixed.
- **Canonical checkpoint**:
  `artifacts/checkpoints/evaluator/pairwise_epoch2.pt`.

Domain-transfer experiments, layer taps, branch-selection probes, and post-RLTT
diagnostic bundles are owned by `utilities/evaluator/`.
