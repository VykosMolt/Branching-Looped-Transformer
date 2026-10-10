# Preregistration status: not reached

This is a terminal gate record, not a predictive-analysis preregistration.

The inventory gate ran before detector implementation, checkpoint selection,
replay, validation fitting, and held-out inspection. It found exact preserved
generated-token sequences for 0/2,720 Horizon candidates. The mission's
stopping rule therefore assigned `FIXED_PREFIX_HORIZON_NOT_ADJUDICABLE` and prohibited all downstream
work.

- Inventory report SHA-256: `990b6e6f590ca0487d51a9f1b8db92c33bfc7cbf89964bb804f520b94e900797`
- Fixed grid that was not evaluated: `[16, 32, 48, 64, 96, 128]`
- Selected primary N: none
- Detector frozen: no
- Replay performed: no
- Model selection performed: no
- Held-out predictive metrics opened: no

Retokenizing rendered continuations, regenerating from seeds, or inventing a
boundary would violate the repaired estimand. This file exists only so the
terminal evidence package has an explicit preregistration disposition.
