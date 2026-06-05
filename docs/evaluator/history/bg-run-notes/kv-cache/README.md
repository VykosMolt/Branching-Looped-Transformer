# KV/Cache Branch-Carry Run Notes

Exact run notes for generation-time KV/cache branch-carry and the compute-saving
splice, moved out of the evaluator docs root on 2026-06-04. The consolidated
interpretation is `../../../kv-cache-branch-carry.md`.

- `bg_autoregressive_kv_branch_carry_v1.md` — v1 formal report (2026-06-01): ladder
  L0-L5 validated; `AUTOREGRESSIVE_KV_BRANCH_CARRY_STATUS = PROMPT_INTERNAL_BRANCH_CACHE_VALID`.
- `autoregressive_kv_branch_carry_validation_note.md` — v1 interpretation note ("what
  changed"); predates v2 so it still describes the splice as diagnostic-only.
- `bg_partial_cache_splice_v2.md` — v2 report (2026-06-04): real suffix-recompute splice;
  `PARTIAL_CACHE_SPLICE_V2_STATUS = PARTIAL_SPLICE_COMPUTE_SAVING_VALID`.

Artifacts: `artifacts/reports/probes/bg_autoregressive_kv_branch_carry_v1_2026-06-01/`
and `artifacts/reports/probes/bg_partial_cache_splice_v2_2026-06-01/`.
