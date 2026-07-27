# Horizon Logic power extension (v3) — pre-registered plan

Date: 2026-07-26. Written BEFORE the generation shards were run.

## Motivation
The published v2 increment (+0.141, CI [+0.004, +0.290]) rests on ~19 heldout
negatives, and the malformed-sibling shortcut control (paper1_v2_malformed_
sibling_control_20260725) pushed the interval across zero (+0.122,
[-0.014, +0.270]). This run adds power; it does not change the protocol.

## Design
- 3 new generation shards: offsets 170/340/510 of the SAME hash-sorted
  4000-task pool (v2 main used [0,170)); k=4, max_new=448, seed=20260724,
  identical prompts/sampling/cut/features (bg_v3_horizon_power_generate.py is
  a path-only wrapper over the sealed generator).
- Task-level split remains the sealed deterministic prospective function of
  task_uid; disjointness with v2 asserted.

## Pre-registered endpoints (bg_v3_horizon_power_analysis.py)
1. STAGE A GATE: v2-only reanalysis must reproduce auroc_results.json to 1e-9
   or the run aborts.
2. new_only_4sc / pooled_4sc: nested increment under the published 4-shortcut
   composite (new-only = independent replication; pooled = combined).
3. new_only_5sc / pooled_5sc: the same under the STRONGER 5-shortcut composite
   including the malformed-sibling count. THE HEADLINE ENDPOINT for v3 is
   pooled_5sc: the increment must exclude zero against the adversarial
   baseline to be claimed as robust.
4. Verdict labels fixed in the script before running:
   REPLICATED_AND_ROBUST / REPLICATED_ON_PUBLISHED_PROTOCOL_BUT_NOT_ROBUST /
   DID_NOT_REPLICATE_AT_INCREASED_POWER / MIXED_SEE_ARMS.

## Controls
Task-set disjointness; split == split_for(uid) for every record; shuffled-label;
LOTO max influence; malformed-share task-drop; per-shard composition counts.

## Interpretation commitments
- If pooled_5sc excludes zero: report as the corrected headline (v3 abstract).
- If not: v3 reports the pre-answer second-domain result as NOT robust to the
  malformed-sibling baseline at increased power — no averaging, no retrying
  with different seeds, no post-hoc shortcut removal.
