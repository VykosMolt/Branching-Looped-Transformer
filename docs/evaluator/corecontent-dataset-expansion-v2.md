# CoreContent Dataset Expansion + Content Tap Refit v2

## CoreContent dataset expansion and refit v2 (2026-06-04)

- Status: `V2_CORECONTENT_READY`.
- Why: v1 kept the broad-objective baseline (mixedhead_MIX_HH_OBJECTIVE); bottleneck was dataset scale, not head design. v1 reward-diverse coverage {'alignment': 200, 'logic': 80, 'math': 66, 'coding': 30, 'reasoning': 5}.
- Data expanded (reward-diverse): {'alignment': 25993, 'coding': 1733, 'logic': 2199, 'math': 3200, 'reasoning': 2600} ; feature storage 4.87 GB, 64 shards. Datasets: coding(mbpp/apps/verifiable/humaneval), math(gsm8k/hendrycks/svamp), logic(logiqa), reasoning(arc/openbookqa/commonsenseqa/strategyqa), alignment(hh/ultrafeedback/shp/pku).
- Parser/verifier: CORE_LABELS_CLEAN; dedup/leakage: LEAKAGE_FOUND_FIXED.
- Heldout: best v2 `CoreContent_v2_blockwise` = 0.6691 vs mixedhead_MIX_HH_OBJECTIVE 0.5525 (verdict V2_CORECONTENT_READY).
- Selected content selector: `CoreContent_v2_blockwise` (LOCK_V2_CORECONTENT_POLICY). Phase 2b: READY_FOR_PHASE2B_WITH_V2_CORECONTENT.
- No steering trained/applied/claimed. No Ouro training, no weight/tokenizer/checkpoint edits, no tap-registry mutation. pure_content_taps.pt / transplanted_taps.pt untouched. Science/anatomy diagnostic-only; terminal survivor-set handoff retained; DualAnchor branch survival unchanged.
- Artifacts: `artifacts/reports/probes/bg_corecontent_dataset_expansion_refit_v2_2026-06-04`.

### Top-line verdicts

    BG_CORECONTENT_V2_INVENTORY_VERDICT = READY
    BG_CORECONTENT_V2_DATASET_PULL_VERDICT = ALIGNMENT_PULL_LARGE
    BG_CORECONTENT_V2_SCHEMA_NORMALIZATION_VERDICT = READY
    BG_CORECONTENT_V2_CANDIDATE_GROUPS_VERDICT = READY
    BG_CORECONTENT_V2_PARSER_VERIFIER_VERDICT = CORE_LABELS_CLEAN
    BG_CORECONTENT_V2_DEDUP_LEAKAGE_VERDICT = LEAKAGE_FOUND_FIXED
    BG_CORECONTENT_V2_FEATURE_PLAN_VERDICT = READY
    BG_CORECONTENT_V2_FEATURE_EXTRACTION_VERDICT = READY
    BG_CORECONTENT_V2_DATASET_BALANCE_VERDICT = LARGE_CORE_DATA_READY
    BG_CORECONTENT_V2_BASELINES_VERDICT = MIX_OBJECTIVE_ALL_STRONG
    BG_CORECONTENT_V2_LINEAR_PAIRWISE_VERDICT = LINEAR_IMPROVED_WITH_DATA
    BG_CORECONTENT_V2_LISTWISE_VERDICT = LISTWISE_READY
    BG_CORECONTENT_V2_DOMAIN_GATED_VERDICT = DOMAIN_GATED_READY
    BG_CORECONTENT_V2_WEIGHT_MERGE_VERDICT = BROAD_OBJECTIVE_STILL_BEST
    BG_CORECONTENT_V2_SCIENCE_AUX_VERDICT = SCIENCE_AUX_HELPS_CORE
    BG_CORECONTENT_V2_HELDOUT_VERDICT = V2_CORECONTENT_READY
    BG_CORECONTENT_V2_DOMAIN_ERROR_VERDICT = CORE_DOMAINS_IMPROVED
    BG_CORECONTENT_V2_CALIBRATION_ABLATION_VERDICT = ROBUST
    BG_CORECONTENT_V2_POLICY_SELECTION_VERDICT = LOCK_V2_CORECONTENT_POLICY
    BG_CORECONTENT_V2_PHASE2B_READINESS_VERDICT = READY_FOR_PHASE2B_WITH_V2_CORECONTENT
    CORECONTENT_DATASET_EXPANSION_REFIT_V2_STATUS = V2_CORECONTENT_READY

## Hardening addendum — relevance retrain + L47 prune (2026-06-06)

- Final re-locked content selector: **CoreContent_v2_blockwise_pruned_24_36** (2-channel tap, layers 24+36; layer 47 pruned as dead weight).
- Follow-up stress tests showed the original headline (+0.117) was ~half a constructed-negative artifact: on real-negative domains (reasoning/logic/alignment) the edge over MIX_HH_OBJECTIVE is +0.063; and the coding tap, trained only on canonical-vs-mutant, was a *corruption detector* — it dropped to 0.58 against real wrong-problem solutions (relevance).
- After retraining coding with wrong-problem relevance negatives and pruning L47: coding wrong-problem top1 0.593 -> 0.5829; coding mutation top1 0.9146 -> 0.8744; hardened core macro 0.5979 -> 0.5977; real-neg macro 0.6022 -> 0.6106.
- No steering, no Ouro training, no registry mutation; pure/transplanted taps untouched; science diagnostic-only; DualAnchor branch survival unchanged; terminal survivor-set handoff retained.
