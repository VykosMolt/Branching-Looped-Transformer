# Scratch notes (folded into MASTER_RESULTS.md at the end)

## Geometry audit (COMPLETE)
- Readable<->outcome overlap: enriched vs null at ALL loci (rank1 27-38deg vs ~89deg null).
  L4_47 (terminal) shows tightest multi-rank alignment, not the early loci.
- Writable data exists ONLY at late loci (loop1-4 x layer{24,36,47}); zero early-locus
  writable tensor anywhere in repo -> INSUFFICIENT_MATCHED_LOCUS_WRITABLE_DATA for the
  strict early-vs-late writable comparison.
- At late loci: readable/outcome vs writable = at-chance everywhere (86-90deg, null p05
  ~87.5deg) -> READABLE_OUTCOME_OVERLAP_WITHOUT_WRITABLE_ALIGNMENT.
- Loop rotation L16: L1->L2 30.89deg > L2->L3 16.63deg > L3->L4 11.16deg (rank1);
  directionally consistent with frozen study's L1->L2 rotation claim but does not clear
  the random-rotation null -> reported descriptively, not as _CONFIRMED.

## Engineering issue
- First pilot attempt OOM'd after task 1 (out.scores never freed between generate()
  calls). Fixed: move logits to CPU, del out, empty_cache per task. Verified via 8-task
  smoke test.

## Horizon Logic pilot (IN PROGRESS as of this note)
- category=synthetic_propositional, proof_depth 2-4, k=4 candidates/task.
- Smoke (8 tasks): n_success=18/32=0.5625, n_malformed=7/32=0.219, mean_n_pre_tok=145/320.
