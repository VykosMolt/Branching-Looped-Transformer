# Artifact audit — existing features vs recapture (2026-07-20)

## What was inspected

- `docs/evaluator/evaluator-locus-summary.md` — locus programme v2–v10 history; canonical
  loci are 24/36/47 throughout; no 8/16 capture mentioned.
- `docs/evaluator/core-domain-tap-audit.md` + CoreContent v2 section — tap inventory (45
  taps, 9 families): every tap config is over layers {24,36,47} (plus 47_mean/concat
  variants). No early-layer taps exist.
- `docs/evaluator/chronological-evaluator-summary.md`, `interfaces-and-tools.md`,
  `dualanchor-tap-evolution.md` (via locus summary + audit doc cross-references) — same
  conclusion; the v7 "all-layer/cached" probes are all-*loci-of-24/36/47*, not all
  physical layers.
- `artifacts/reports/probes/bg_corecontent_dataset_expansion_refit_v2_2026-06-04/` — the
  current corrected CoreContent line: `feature_extraction_plan.json`, shard manifest,
  `heldout_eval_v2.json`, `followups*.json`, `corecontent_v2_policy.pt` (locked policy =
  AntisymLinearNoNorm channels 24_L4 + 36_L4; layer 47 pruned as dead weight).
- `data/corecontent_v2/features/` — 4.6 GB, 64+ shards; per-candidate tensors are
  **(3,4,2048) = layers[24,36,47] × loops[1..4]**, fp16, masked-mean pooled, produced by
  `src/evaluator/bg_transformer_features.BGTransformerFeatureExtractor` on the local
  Ouro-RLTT checkpoint. Loop-separated: yes. Layers 8/16: **absent**.
- Filesystem sweep for `*all_layer*`, `*layer8*`, `*l8*` style caches under `artifacts/`
  and `data/`: nothing relevant.

## Decision

No compatible layer-8/16 features exist under any checkpoint. Bounded recapture is
required and was scoped to a deterministic subset of the CoreContent v2 corrected dataset
(2,150 groups / 8,592 candidates; 250 train / 60 val / 120 heldout groups per core
domain), capturing layers {8,16,24,36,47} × loops {1..4} in a single frozen forward per
candidate. Layers 24/36/47 are recaptured alongside so the new extraction path can be
validated feature-by-feature against the production cache (see
`extraction_controls.json`) — reused cached features serve as the reference control, not
as training input.

## Validation of the reused artifacts (cached v2 shards)

- checkpoint identity: same local model dir (`models/ouro_rltt_local`, config sha
  `7d6764dbc8210d02`), same wrapper class, same tokenizer files (unchanged on disk since
  the v2 run per git status; model dir is untracked/frozen).
- extraction locus + loop ordering: verified in code
  (`bg_transformer_features.py`: hooks on `layers[23]`, `layers[35]`, loop-boundary list
  for 47; append order = loop order).
- pooling: masked mean over valid tokens, fp32 → fp16 storage.
- split identity: per-group `split` field from the deterministic v2 `split_for`; subset
  integrity re-asserted in `split_integrity.json` (0 crossings).
- label alignment: rewards joined by `candidate_uid`, asserted equal between the
  processed jsonl and the shard records.

## Data-quality findings surfaced by this audit (new)

1. **195 `hendrycks_math` task_uids span multiple splits** in the v2 processed dataset —
   a task-ID collision (row index reused across subject configs), not prompt leakage
   (identical prompts were deduped in v2). These task_uids are excluded from this run's
   subset (`excluded_crossing_tasks.json`); they make task-clustered inference and
   task-disjointness claims unreliable for the affected math rows. Worth fixing upstream
   in any future v3 dataset build.
2. **~6.5k duplicate group records across v2 feature shards** (resume overlap in the v2
   extraction). Harmless for v2's own aggregate metrics only if duplicates are rare per
   split; this run dedupes by first occurrence (deterministic shard order).

## Secondary readouts (S3B2 / DualAnchor survival / strict pre-answer / HH 40k)

- S3B2 and branch-survival packs exist only at 24/36/47 loci; strict pre-answer (170-task)
  and HH 40k artifacts likewise have no early-layer capture. All would need fresh model
  passes; deferred behind the primary matrix per the degradation plan (HH 40k recapture
  explicitly out of budget).
