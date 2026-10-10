# Leakage audit

The automated audit was run only after the preregistration and model-selection rules were hash-sealed. No classifier was fitted in this stage.

## GSM8K

`VISIBLE_PREFIX_NOT_ADJUDICABLE`: preserved GSM8K rows contain neither generated token IDs, exact rendered trajectories, nor visible-prefix text, and no original train/validation/held-out partition exists

## horizon_new_only

- Gate: `VISIBLE_PREFIX_NOT_ADJUDICABLE`
- Original scorable tasks/candidates: 503/1557
- Leaking tasks: 451 (89.7%)
- Candidate literal-gold hits: 861
- Candidate symbolic-equivalence hits: 153
- Forbidden marker hits: 0
- Reason: genuine normalized-gold leakage requires task exclusions exceeding the predeclared 20% materiality threshold

## horizon_pooled

- Gate: `VISIBLE_PREFIX_NOT_ADJUDICABLE`
- Original scorable tasks/candidates: 669/2067
- Leaking tasks: 598 (89.4%)
- Candidate literal-gold hits: 1144
- Candidate symbolic-equivalence hits: 215
- Forbidden marker hits: 0
- Reason: genuine normalized-gold leakage requires task exclusions exceeding the predeclared 20% materiality threshold

## Duplicate-prompt audit

- Exact cross-split duplicate groups: 0
- Near-duplicate threshold: 0.95
- Cross-split near-duplicate pairs: 5

No individual visible span was censored. Genuine violations are disposed taskwise exactly as preregistered.
