# Hidden-Origin Split Salvage

Split salvage audited the earlier hidden-origin branch data after v3 heldout looked too sparse. The main finding was that hidden-origin selector signal existed, but the strict heldout split was not balanced enough for readiness.

## Verdicts

```text
BG_HIDDEN_ORIGIN_SPLIT_AUDIT_VERDICT = READY
BG_HIDDEN_ORIGIN_EVAL_MODE_VERDICT = WEAK_ONLY
BG_HIDDEN_ORIGIN_SALVAGE_DATASET_VERDICT = READY
BG_HIDDEN_ORIGIN_SALVAGE_TRAINING_VERDICT = READY
BG_HIDDEN_ORIGIN_SALVAGE_EVAL_VERDICT = WEAK_SELECTOR
BG_HIDDEN_ORIGIN_CV_STABILITY_VERDICT = STABLE_POSITIVE
BG_HIDDEN_ORIGIN_V4_QUOTA_NEED_VERDICT = V4_REQUIRED_HELDOUT_BALANCE
HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE = old_frozen_bg
PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_SALVAGE = WEAK
```

## Main Finding

The existing v3 data was not empty and did contain ranking signal:

- behaviorally diverse groups: `65`
- support tasks with non-tie pairs: `14`
- non-tie pairs: `536`
- tie rate on reward-signal support: `0.709`

But strict/clean heldout was inadequate:

- behaviorally diverse groups: `6`
- support tasks with non-tie pairs: `1`
- non-tie pairs: `15`
- tie rate: `0.989`

## Implication

The blocker was heldout balance, not total absence of signal. This directly motivated quota v4: reserve train/val/heldout task IDs before generation, generate per split, and stop only when split quotas are met.

Legacy report:

- `docs/evaluator/bg_hidden_origin_split_salvage.md`

