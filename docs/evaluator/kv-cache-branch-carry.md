# Generation-Time KV/Cache Branch-Carry (and compute-saving splice)

Updated: 2026-06-04

This consolidates two runs that together establish autoregressive, branch-specific
KV/cache carry on Ouro and a real compute-saving splice on top of it:

- **v1** `autoregressive_kv_branch_carry_v1` (2026-06-01) — validated the cache-carry
  ladder Levels 0–5; the partial splice was diagnostic-only.
- **v2** `partial_cache_splice_v2` (2026-06-04) — implemented a true suffix-recompute
  splice, upgrading the diagnostic to a measured compute saving.

Exact run notes are archived under `history/bg-run-notes/kv-cache/`.

## Bottom line

| Status key | Value |
| --- | --- |
| `AUTOREGRESSIVE_KV_BRANCH_CARRY_STATUS` | `PROMPT_INTERNAL_BRANCH_CACHE_VALID` (v1, L0–L5) |
| `PARTIAL_CACHE_SPLICE_V2_STATUS` | `PARTIAL_SPLICE_COMPUTE_SAVING_VALID` (v2; upgrades v1's `LEVEL6 = DIAGNOSTIC_ONLY`) |

**What can be claimed:** autoregressive token-boundary / batched / prune-reorder /
current-token-perturb / prompt-internal-perturb branch-carry, all validated; and
**amortized compute-saving branch-carry** via suffix recompute (needs K≥2 branches
sharing a prompt). **What cannot:** production readiness; steering of any kind; any
claim beyond the validated levels.

## Why this mattered

Prompt-only layer carry (validated earlier by the architecture-looped DualAnchor
probes, which run `use_cache=False`) is **not** the same as generation-time,
branch-specific KV/cache carry. The latter requires each branch to keep its own
`past_key_values`, `cache_position`, `attention_mask`, `position_ids`,
`generated_ids`, and lineage aligned across autoregressive decode steps.

## Cache structure (Ouro UniversalTransformerCache)

`OuroForCausalLM`, bf16, eager attention; `total_ut_steps = 4`,
`num_hidden_layers = 48` → **192 distinct cache slots**, with
`cache_slot = current_ut * num_hidden_layers + layer_idx` (confirmed; slot audit =
`SLOT_MAPPING_CONFIRMED`). Each (loop, layer) owns a distinct KV slot; prefill
populates all 192; a decode step appends one token to every slot; `reorder_cache`
reorders the batch dim of every populated slot.

## Equivalence standard (important)

Cached decode logits are compared (in float32) to a full no-cache recomputation of
the **identical** running sequence:

- **Prefill is bit-exact** (RMS 0).
- **Decode shows small bf16 drift** (RMS ~0.05–0.2, max-abs < 1.0) from
  shape-dependent cuBLAS accumulation (q=1 vs q=seq). Strict 1e-4 equality is only
  reachable at prefill.
- **Top-1 / token sequences match** except at model-intrinsic argmax near-ties at
  the bf16 noise floor (counted, not hidden). A top-1 disagreement is treated as a
  cache failure only when the logit gap exceeds the bf16 band.

## v1 — validation ladder (L0–L5 pass; L6 diagnostic)

| Level | Result |
| --- | --- |
| L0 cached decode | matches full recompute (prefill bit-exact; decode bf16 drift) |
| L1 token-boundary fork | K=2/4/8 independent branch caches; no cross-branch contamination |
| L2 batched branches | batched == independent == full recompute |
| L3 prune/reorder survivors | subset, order changes, 8→4→2 / 8→3 / 4→1; lineage aligned |
| L4 current-token perturb | layers 24/36/47, loop-targeted; carries via branch cache |
| L5 prompt-internal perturb | branch-specific cache; negative control (unperturbed cache vs perturbed recompute, RMS ≈ 3.0) confirms the branch cache is required |
| L6 partial splice (v1) | slot-boundary logic valid (copy-affected reproduces the full cache bit-exactly) but **diagnostic only — no compute saving** |

Supporting: padding/mask/cache_position = `PADDING_MASK_SAFE` (left-padded batched
decode needs explicit per-row `position_ids`); DualAnchor integration smoke =
`BRANCH_PRUNE_CARRY_SMOKE_VALID`; failure analysis = `NO_MAJOR_FAILURES`.

## v2 — real compute-saving suffix-recompute splice

**Key obstacle:** the KV cache stores K/V but not the inter-layer residual stream, so
naively a perturbed branch cache forces a full re-prefill. **Solution:** also capture
the residual hidden at the perturbation boundary (output of loop u, layer L) during a
minimal shared-prefix prefill; for an additive boundary perturbation reconstruct
`H_boundary_perturbed = H_boundary + delta` with no forward, and recompute **only the
suffix** (loop u layers L+1.., loops u+1..). Implemented as test-only orchestration
over `model.model.layers / rotary_emb / norm / lm_head` — no weight edits, no
permanent model surgery.

- **Hook timing (empirical):** perturbing a layer *output* leaves the boundary slot
  unaffected; first affected slot = `(u, L+1)`; the changed set equals the
  `downstream_only` theory. Over-sharing an affected slot diverges (negative control).
- **Equivalence:** the spliced branch cache is **bit-exact** vs a full
  perturbed-prompt reference (all 192 slots, prefill logits RMS 0, continuation
  bit-for-bit). Validated single-branch, multi-branch (K=2/4, independent storage, no
  contamination), batched + prune/reorder, and left-padded (with explicit
  position_ids; RoPE's relativity makes a single-prompt left-pad shift harmless).
- **Measured compute saving** (baseline = K full perturbed prefills; splice = one
  shared prefix prefill + K suffix recomputes):

  | boundary (loop, layer 24) | per-branch passes saved |
  | --- | ---: |
  | loop 0 | 13% |
  | loop 1 | 38% |
  | loop 2 | 63% |
  | loop 3 | 88% |

  K-scaling at (loop 2, layer 24): K=2 → 32%, K=4 → 47%, K=8 → 55% fewer layer-passes
  (wall-clock tracks ~33/48/56%). Saving is **amortized** — K=1 does prefix+suffix ==
  full work. The copy-affected oracle (Mode A) saves nothing and is excluded.

## No-steering statement

Both runs performed no steering (no vectors, no action steering, no trained corridor,
no steering as a tested condition, no steering claim), no Ouro training, no
weight/tokenizer/checkpoint edits, no BG tap or registry changes, no wrapper /
local-agent, no Hunter-Seeker imports/instantiation, no git. Perturbations are a test
harness to create branch-specific caches, not behavioural interventions. v2's GPU
guard confirmed the MMLU science-repair process was inactive and left its artifacts
untouched.

## Source run notes & artifacts

Archived under `history/bg-run-notes/kv-cache/`:

- `bg_autoregressive_kv_branch_carry_v1.md` (v1 formal report)
- `autoregressive_kv_branch_carry_validation_note.md` (v1 interpretation note)
- `bg_partial_cache_splice_v2.md` (v2 compute-saving report)

Artifacts: `artifacts/reports/probes/bg_autoregressive_kv_branch_carry_v1_2026-06-01/`
and `artifacts/reports/probes/bg_partial_cache_splice_v2_2026-06-01/`. Reusable
test-only helpers live in `utilities/tests/manual/bg_autoregressive_cache_{common,helpers}_v1.py`
and `utilities/tests/manual/bg_partial_cache_splice_v2_common.py`.
