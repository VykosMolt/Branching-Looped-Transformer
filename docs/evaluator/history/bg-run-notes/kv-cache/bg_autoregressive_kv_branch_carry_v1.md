# Autoregressive KV/Cache Branch-Carry Validation v1 (2026-06-01)

Short name: `autoregressive_kv_branch_carry_v1`

## Final status

**`AUTOREGRESSIVE_KV_BRANCH_CARRY_STATUS = PROMPT_INTERNAL_BRANCH_CACHE_VALID`**
(validation ladder Levels 0–5 all pass within bf16).
**`LEVEL6_PARTIAL_SPLICE_STATUS = PARTIAL_SPLICE_DIAGNOSTIC_ONLY`** — no compute-savings claim is made.

| Stage | Verdict |
|---|---|
| Inventory | READY |
| Cache helpers (unit tests) | READY |
| Level 0 — cached decode | CACHE_NUMERIC_DRIFT_SMALL |
| Level 1 — token-boundary fork | BRANCH_CACHE_NUMERIC_DRIFT_SMALL |
| Level 2 — batched branches | BATCHED_NUMERIC_DRIFT_SMALL |
| Level 3 — prune/reorder survivors | PRUNE_REORDER_NUMERIC_DRIFT_SMALL |
| Level 4 — current-token perturb | CURRENT_TOKEN_PERTURB_NUMERIC_DRIFT_SMALL |
| Level 5 — prompt-internal perturb | PROMPT_INTERNAL_NUMERIC_DRIFT_SMALL |
| Level 6 — partial-cache splice | SPLICE_SLOT_LOGIC_VALID_BUT_NO_COMPUTE_SAVING |
| Padding/mask/cache_position | PADDING_MASK_SAFE |
| Loop/layer slot audit | SLOT_MAPPING_CONFIRMED |
| DualAnchor integration smoke | BRANCH_PRUNE_CARRY_SMOKE_VALID |
| Failure analysis | NO_MAJOR_FAILURES |

Artifacts: `artifacts/reports/probes/bg_autoregressive_kv_branch_carry_v1_2026-06-01/`.

## Why autoregressive cache branch-carry needed validation

Prompt-only layer carry (already validated by the architecture-looped DualAnchor
probes) is **not** the same as generation-time, branch-specific KV/cache carry.
Existing latent-beam / loop-branching code runs with `use_cache=False`, so it
validates prompt/loop hidden-state perturbation behaviour but cannot prove that a
branch can keep its own `past_key_values`, `cache_position`, `attention_mask`,
`generated_ids`, and lineage aligned across autoregressive decode steps. This run
tests exactly that.

## Distinction from prompt-only layer carry

- **Prompt-only carry** (prior work): perturb/observe hidden states inside a single
  prompt forward; no incremental KV cache; `use_cache=False`.
- **Autoregressive branch-carry** (this run): prefill once, fork branch-specific
  caches, then *continue generation* step-by-step with `use_cache=True`, keeping
  each branch's cache/mask/positions/lineage consistent, and proving each cached
  branch equals full recomputation of that exact branch.

## UniversalTransformerCache slot structure

- Model: `OuroForCausalLM`, bf16, eager attention; `total_ut_steps = 4`,
  `num_hidden_layers = 48` → **192 distinct cache slots**.
- Slot formula (confirmed in `OuroAttention.forward`):
  `cache_slot = current_ut * num_hidden_layers + layer_idx`.
- Each (loop, layer) owns a distinct KV slot; prefill populates all 192; a decode
  step appends one token to every slot. `reorder_cache` reorders the batch
  dimension of every populated slot. Layer 47's output feeds the next loop's
  layer 0 via the UT recurrence.

## Cache clone/fork helpers (`bg_autoregressive_cache_helpers_v1.py`)

`BranchState` dataclass + `clone_universal_cache` (deep-clone, independent
storage, preserves `_seen_tokens`/`max_cache_size`), `expand_universal_cache_for_branches`,
`compare_universal_caches`, `summarize_universal_cache`,
`assert_branch_state_consistent`, `make_branch_states_from_prefill`,
`prune_branch_states`/`reorder_cache_and_branch_state`, `concat_generated_token`,
`update_attention_mask`, `compute_cache_slot_order`,
`affected_cache_slots_for_boundary`, `maybe_splice_cache_prefix`. All 16 unit
tests pass (clone isolation, expand independence, reorder/prune, slot mapping,
branch-state consistency).

## Equivalence standard (important)

The cache-correctness property tested is: **cached decode logits match a full
no-cache recomputation of the identical running sequence, within tolerance**, with
all comparisons in float32.

- **Prefill is bit-exact** (RMS = 0.0): a `use_cache=True` prompt forward equals the
  `use_cache=False` forward exactly.
- **Decode shows small bf16 drift** (RMS ≈ 0.05–0.2, max-abs < 1.0): the cached
  query-length-1 path and the full query-length-N path take shape-dependent cuBLAS
  matmul-accumulation orders, differing by ~1–2 bf16 ULP. Strict 1e-4 logit
  equality is **not** achievable in bf16 decode — only at prefill.
- **Top-1 / token sequences match exactly** except at *model-intrinsic argmax
  near-ties*: when the top-2 logits are within the bf16 drift band, the tiny
  cached-vs-full difference can flip the argmax. These are numerical ties, not
  cache errors (11 such flips across the whole run; 0 real mismatches above the
  noise floor). A top-1 disagreement is only treated as a cache failure when the
  logit gap exceeds the bf16 band.

## Cached decode equivalence (Level 0)

Prefill bit-exact; cached decode for 1/2/4/8 steps matches full recompute within
bf16; `model.generate` greedy == manual cached greedy == full-recompute greedy
(modulo near-ties). Structural checks (cache seq length, slot count = 192,
attention-mask length, cache_position) all correct.

## Token-boundary branch carry (Level 1)

Prefill once, clone the root cache into K = 2/4/8 independent branches seeded by
distinct top-K first tokens. Each branch's cached logits match full recompute of
that exact branch within bf16. Branch caches have **independent storage**; a
cross-branch contamination test (mutating one branch by extra decode steps) leaves
siblings **bit-identical**. Branch cache lengths advance independently.

## Batched branch carry (Level 2)

K branches expanded across the batch dimension of one cache. With token sequences
fixed via independent greedy then replayed batched, batched logits equal both the
independent per-branch caches and full recompute within bf16. Batch dimension,
sequence lengths, slot count, and finiteness all correct.

## Prune/reorder carry (Level 3)

Survivor selection via `reorder_cache(beam_idx)` in lockstep with the BranchState
list, `input_ids`, `attention_mask`, `generated_ids`, and lineage. Subset
selection, **order changes** (e.g. survivors `[3,1,6]`), and multi-round pruning
(8→4→2, 8→3, 4→1) all preserve survivor caches: the live reordered-cache
continuation logits match full recompute of each survivor's exact sequence within
bf16, with correct lineage/order alignment.

## Current-token perturbation carry (Level 4)

After prefill, clone a branch cache, apply a deterministic hidden perturbation to
the **current generated token** at decoder layer 24/36/47 (UT loop targeted via
`current_ut`), and continue generation. A full recomputation that re-applies the
same perturbation at the same fixed absolute position and (loop, layer) matches the
cached perturbed branch within bf16. A zero-perturbation control reduces to Level 0.
The perturbation is injected at the **first** UT loop so it propagates into many
downstream cache slots (genuine branch-specific carry); the cached-vs-full drift
stays bf16-scale regardless of perturbation magnitude (the perturbation is identical
in both paths).

## Prompt-internal perturbation cache (Level 5)

Perturbing a span of internal prompt positions during prefill produces a
branch-specific cache that continues correctly and matches full perturbed
recompute within bf16 (max RMS ≈ 0.07). The **negative control** — continuing the
same tokens from the *unperturbed* prompt cache — diverges sharply from the
perturbed recompute (RMS ≈ 3.0), confirming the **branch-specific prompt cache is
required** and cannot be substituted by the unperturbed cache. Validates
correctness, **not** compute savings (the full perturbed prompt is recomputed).

## Partial-cache splice diagnostic (Level 6)

For a perturbation at (loop u, layer L), slots before the boundary are provably
shareable (root and branch slots are bit-identical there). An Option-A boundary
splice (root for shared slots, perturbed branch for affected slots) reproduces the
full perturbed branch cache **bit-exactly** and continues identically; the
conservative splice (share only loops < u) is also exact; and an aggressive
over-share (claiming only the last loop is affected) **diverges**, confirming the
affected slots are genuinely branch-specific. Hypothetical shareable fraction at
loop 0: ~13% (L24), ~19% (L36), ~25% (L47).

This validates the **shared/affected slot boundary logic** only. Option A copies the
affected slots from a full perturbed prefill, so **it does not save compute**. Real
compute saving (Option B — recomputing only the affected suffix) would require model
surgery and is **not** implemented here.

## Padding/mask/cache_position stress (Part J)

Single unpadded prompt, length-1 decode, and reorder/prune after padding are all
safe; `cache_position` values are correct. Left-padded mixed-length batched
prefill matches independent unpadded recompute within bf16 **when explicit per-row
`position_ids` are supplied** — Ouro derives `position_ids` from a single
`cache_position` when none is passed, so left-padded batched **decode** requires
explicit per-row positions. This is a usage requirement, not a cache bug.

## Final status and claims allowed

- ✅ Autoregressive token-boundary branch-carry (Level 1).
- ✅ Batched branch-carry (Level 2).
- ✅ Prune/reorder survivor carry (Level 3).
- ✅ Current-token layer-perturbation branch-carry during generation (Level 4).
- ✅ Prompt-internal branch-specific cache validity (Level 5).
- ❌ Compute-saving branch-carry — **NOT claimed** (Level 6 diagnostic only).

## Explicit no-steering statement

This run performed **no steering** of any kind: no steering vectors, no action
steering, no trained steering corridor, no steering as an experimental condition,
and no steering claim. No Ouro training, no weight/tokenizer/checkpoint edits, no
BG tap or tap-registry updates, no wrapper/local-agent execution, no Hunter-Seeker
imports or `HunterSeekerAgent` instantiation, no `train_arc.py`, no ARC/MATH
generation, no git commands. DualAnchor scoring in the integration smoke was
deliberately **skipped** (a deterministic logprob proxy was used) so no tap
registry was touched.

## Explicit no-compute-savings statement

**No compute-savings claim is made.** Level 6 validates only that the shared/affected
cache-slot boundary is correct (Option A reproduces the full perturbed branch cache);
it does not implement suffix recomputation and therefore saves no compute. A
compute-saving claim would require an Option-B implementation that avoids recomputing
shared slots *and* passes full-recompute equivalence.


## Partial cache splice v2 (2026-06-01)

`partial_cache_splice_v2` turned the v1 Level 6 diagnostic (copy-affected slots, no
saving) into a real compute-saving suffix-recompute splice. Key obstacle: the
UniversalTransformerCache stores K/V but not the inter-layer residual stream, so the
residual hidden at the perturbation boundary (loop u, layer L output) is captured
during a minimal shared-prefix prefill; an additive boundary perturbation is then
applied without a forward, and ONLY the suffix (loop u layers L+1.., loops u+1..) is
recomputed. The spliced branch cache is **bit-exact** vs a full perturbed-prompt
reference (all 192 slots; prefill logits RMS 0; continuation bit-for-bit).

Empirical hook timing confirmed: perturbing a layer output leaves the boundary slot
unaffected; first affected slot = (u, L+1); changed set == downstream_only theory.
Validated single-branch, multi-branch (K=2/4, independent, no contamination),
batched+prune/reorder, and left-padded (explicit position_ids). Measured compute
savings (baseline K full prefills vs prefix + K suffixes): per-branch layer-pass
saving 13%/38%/63%/88% for boundary loops 0/1/2/3 (layer 24); at loop 2 layer 24,
K-scaling gives 32%/47%/55% fewer passes for K=2/4/8 (wall-clock tracks). Saving is
amortized (needs K≥2; K=1 does prefix+suffix == full). Copy-affected (Mode A) saves
nothing.

**Status: `PARTIAL_CACHE_SPLICE_V2_STATUS = PARTIAL_SPLICE_COMPUTE_SAVING_VALID`**
(upgrades v1's `LEVEL6_PARTIAL_SPLICE_STATUS = PARTIAL_SPLICE_DIAGNOSTIC_ONLY`).
Compute-saving branch-carry CAN now be claimed (amortized, equivalence-validated);
production readiness CANNOT. No steering, no training. GPU guard confirmed the MMLU
science-repair process was not active and its artifacts were untouched. Details:
`docs/evaluator/bg_partial_cache_splice_v2.md`; artifacts under
`artifacts/reports/probes/bg_partial_cache_splice_v2_2026-06-01/`.
