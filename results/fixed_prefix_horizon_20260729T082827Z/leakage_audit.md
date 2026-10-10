# Leakage and integrity audit

`FIXED_PREFIX_HORIZON_NOT_ADJUDICABLE`

The inventory confirms zero task crossing, zero duplicate candidate IDs and
zero candidate crossing. The original shard checksums match their historical
manifests. The repaired boundary leakage audit cannot be run because no exact
first-N generated-token sequence exists. Continuations were not retokenized.

All post-boundary leakage checks are `NOT_APPLICABLE_NO_N`; no feature matrix,
preprocessor, base score or fusion score was constructed. Replay mismatch is
undefined because replay was prohibited. Checkpoint identity is additionally
not cryptographically pinned by the historical package.
