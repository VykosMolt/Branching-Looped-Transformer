# Final Engineering Expansion Summary

## Executive Summary

S3B2 was expanded on the saved 160-candidate generated-branch pool. The registered hardening result remains AUROC 0.7515, pairwise 0.6835, sel@oracle 0.6250. The expanded hidden ridge probe gives AUROC 0.7755, pairwise 0.7338, sel@oracle 0.6250. Metadata controls are weak and abstention does not solve terminal selection.

The exact June-17 S1/S3 injection tensors were still not found. A recovered adjacent hidden-origin real-delta artifact was audited and gives projection fraction 0.001151 with residual 0.998849. That supports but does not prove the S1.4 subspace-mismatch explanation.

Full drafting can begin with caveated orthogonality.

## S3B2 Expanded Analysis

Inputs: `artifacts/reports/probes/mpn_s3b_2026-06-17/s3b1_loop_pools.pt`, `artifacts/reports/probes/mpn_s3b_2026-06-17/s3b1_pool_texts.json`, `artifacts/reports/probes/mpn_s3b_2026-06-17/s3b1_loop_pool_transfer.json`. Counts: 160 candidates, 16 task groups, 29 positives, 8 oracle-present groups.

| metric | prior L2 logistic | expanded hidden ridge |
| --- | --- | --- |
| AUROC | 0.7515 | 0.7755 |
| pairwise | 0.6835 | 0.7338 |
| sel@oracle | 0.6250 | 0.6250 |
| top2 | 0.7500 | 0.7500 |
| regret | 0.3750 | 0.3750 |

## S1/S3 Real Orthogonality Audit

Exact S1.4 tensors found: no. Adjacent real-delta artifact audited: `artifacts/reports/probes/bg_hidden_origin_diversity_v2_2026-05-18/diverse_hidden_origin_branches.pt`. Audited rows: 542; domains: {'reasoning': 163, 'science': 379}.

| metric | value |
| --- | --- |
| proj outcome onto real injection span | 0.001151 |
| residual outside real injection span | 0.998849 |
| proj outcome onto sampling span rank matched | 0.165930 |
| real injection effective rank | 82.9919 |
| sampling effective rank | 24.4224 |

## Paper Impact

| claim | impact | reason |
| --- | --- | --- |
| generated-branch correctness is partially readable | strengthened | Prior S3B2 logistic AUROC 0.7515 plus expanded hidden ridge AUROC 0.7755; metadata-only controls weak. |
| terminal selection remains wall | strengthened | sel@oracle remains 0.6250 on 8 oracle-present groups; high-margin abstention does not rescue selection. |
| S3A needs verifier-labeled branch training | strengthened | Transfer taps and S3B2 partial readout do not solve terminal selection on generated branches. |
| subspace misalignment explains frozen null | still caveated | Exact June-17 S1.4 tensors are missing; adjacent real-delta audit is supportive but not a strict S1.4 proof. |
| readout-control boundary | strengthened with caveat | Readout signal is measurable while frozen control and selection remain unsolved; orthogonality mechanism is not load-bearing. |

## Exact Paper Insertion Text

**S3B2 selection wall section.** On the saved generated-branch pools, a task-grouped S3B2 refit confirms that branch correctness is partially readable from hidden states. The registered L2 logistic audit reached AUROC 0.7515, pairwise accuracy 0.6835, and sel@oracle 0.6250; an expanded grouped ridge probe gives the same forced-selection conclusion while metadata-only controls fail. Thus the failure is not simply a domain/provenance/length shortcut story. The readout exists, but it is not a reliable terminal arbiter on this small shifted pool.

**Orthogonality/subspace caveat section.** The exact June-17 S1.4 clean, injected, carried, and rederived tensors were not persisted, so the paper should not present a strict S1.4 orthogonality proof. A recovered adjacent hidden-origin intervention artifact does contain real nonzero injection deltas and labels; in that side audit, the correctness direction has projection fraction 0.00115 into the real injection span, with residual 0.99885. This is consistent with a subspace-mismatch explanation but remains scoped as adjacent engineering evidence.

**S3A motivation/future-work section.** The engineering picture is therefore coherent but caveated: frozen branch/carry mechanics are valid, K-matched sampling closes the frozen fork as a capability source, generated-branch correctness is partially readable, and terminal selection remains the bottleneck. S3A should be framed as the next training-time test: learn verifier-labeled branch geometry on the generated-branch distribution, then test whether learned write/read geometry can beat both base decoding and K-matched sampling.

## Final Verdict Constants

| constant | value |
| --- | --- |
| FINAL_ENGINEERING_EXPANSION_VERDICT | BEGIN_DRAFT_WITH_CAVEATED_ORTHOGONALITY |
| S3B2_EXPANDED_VERDICT | GENERATED_BRANCH_CORRECTNESS_SIGNAL_CONFIRMED_BUT_SELECTION_UNSOLVED |
| REAL_INJECTION_ORTHOGONALITY_VERDICT | REAL_INJECTION_DELTAS_FOUND_AND_AUDITED |
| EXACT_S1_S3_TENSOR_STATUS | EXACT_JUNE17_S1_S3_INJECTION_DELTAS_NOT_FOUND |
| S3A_GEOMETRY_CLAIM | SAFE_TO_CLAIM_PROXY_CONSISTENT_WITH_MISALIGNMENT_ONLY |
