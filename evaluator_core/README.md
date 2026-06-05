# Evaluator Core

Core evaluator primitives for Branching Looped Transformer.

- `pairwise_evaluator.py`: frozen pairwise/CLT relational preference head.
- `anchor_loss.py`: frozen-evaluator anchor loss and scoring utilities for
  caller-owned representations while keeping evaluator parameters fixed.

Long-running probes and domain-transfer scripts live under `../probes/`.
Evaluator notes live under `../docs/evaluator/`.
