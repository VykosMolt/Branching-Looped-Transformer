# Evaluator Core

Evaluator source lives here instead of under `hunter_seeker_core` so the
relational evaluator and domain-transfer work are a separate project surface.

- `pairwise_evaluator.py`: frozen pairwise/CLT relational preference head.
- `anchor_loss.py`: frozen-evaluator anchor loss used by Hunter-Seeker training
  and diagnostics.

Long-running probes and domain-transfer scripts live under
`../../utilities/evaluator/`; evaluator notes live under `../../docs/evaluator/`.
