# Signed inventory report

- Inventory timestamp: `2026-07-29T07:59:35.356511+00:00`
- Experiment ID: `visible_prefix_hidden_increment_20260729T075247Z`
- Scope: label-blind schema, path, partition, and checksum inventory.
- Attestation: no correctness values or held-out metrics were read or computed by this inventory program.

## Gate decisions

### horizon_new_only

- Inventory status: `CONDITIONALLY_COMPLETE`
- HV inventory gate: `PENDING_EXACT_HEAD_REPRODUCTION`
- Exact-prefix inventory gate: `PASS`
- Original partition gate: `PASS`
- Reason: Exact rendered trajectories, stored token cut indices, source prompts, tokenizer revision, frozen hidden features, splits, labels, task IDs, and held-out published scores exist. Hidden-only train/validation scores must be cross-fitted from the preserved original head specification.

### gsm8k

- Inventory status: `INCOMPLETE`
- HV inventory gate: `FAIL`
- Exact-prefix inventory gate: `FAIL`
- Original partition gate: `FAIL`
- Reason: The strict-preanswer artifact preserves hidden features and cut lengths but not generated token IDs, rendered trajectories, or visible-prefix text. The original evaluation used grouped cross-validation and did not preserve a train/validation/held-out partition. Exact prefixes and the required partition cannot be reconstructed without changing the estimand.

### horizon_pooled

- Inventory status: `CONDITIONALLY_COMPLETE`
- HV inventory gate: `PENDING_EXACT_HEAD_REPRODUCTION`
- Exact-prefix inventory gate: `PASS`
- Original partition gate: `PASS`
- Reason: The pooled v2+v3 arm has the same complete rendered-text, cut, split, feature, and reproduction evidence as the new-only arm.

## Hidden-score decision rule

Horizon raw hidden-only scores are preserved only for held-out rows. Frozen hidden feature tensors exist for train, validation, and held-out rows, and the deterministic original head implementation, grids, seed, and published held-out scores are preserved. The sealed run must reconstruct the original head without using held-out labels and pass exact/numerical agreement before any visible-model fitting. Failure marks Horizon HV `NOT_ADJUDICABLE`.

GSM8K has frozen hidden features and historical out-of-fold predictions, but has neither exact rendered candidate trajectories nor the required original train/validation/held-out partition. It is stopped at inventory as `VISIBLE_PREFIX_NOT_ADJUDICABLE`; no approximated prefix or replacement split is allowed.

## Signature

The detached SHA-256 signature is written to `inventory_report.sha256`. The signer identity is `Codex /root`, using a deterministic content digest rather than an unavailable private-key signature.
