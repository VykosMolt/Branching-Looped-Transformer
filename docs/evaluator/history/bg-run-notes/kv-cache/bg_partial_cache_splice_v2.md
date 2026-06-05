# Partial cache splice v2 — compute-saving branch-carry (2026-06-01)

Short name: `partial_cache_splice_v2`

## Final status

**`PARTIAL_CACHE_SPLICE_V2_STATUS = PARTIAL_SPLICE_COMPUTE_SAVING_VALID`**

The v2 suffix-recompute splice is implemented, is **bit-exact** against a full
perturbed-prompt reference, and yields **real, measured compute savings** that grow
with branch count K and perturbation-boundary loop depth.

| Stage | Verdict |
|---|---|
| Inventory + GPU guard | READY (science-repair NOT active) |
| Reproduce v1 Level 6 | REPRODUCED |
| Boundary dependency theory | BOUNDARY_THEORY_READY |
| Hook timing (empirical) | DOWNSTREAM_ONLY_AFFECTED |
| Suffix recompute impl | SUFFIX_RECOMPUTE_IMPLEMENTED |
| Single-branch splice | SINGLE_BRANCH_SPLICE_VALID |
| Multi-branch splice | MULTI_BRANCH_SPLICE_VALID |
| Batched + prune/reorder | BATCHED_PRUNE_SPLICE_VALID |
| Position/padding stress | POSITION_PADDING_SAFE |
| Compute accounting | REAL_COMPUTE_SAVING_MEASURED |
| Architecture-looped smoke | ARCH_LOOP_SPLICE_SMOKE_VALID |
| Failure analysis | NO_MAJOR_FAILURES |

Artifacts: `artifacts/reports/probes/bg_partial_cache_splice_v2_2026-06-01/`.

## Why v2 was needed

v1's Level 6 proved the shared/affected slot-boundary *logic* but copied affected
slots from a full perturbed prefill — `PARTIAL_SPLICE_DIAGNOSTIC_ONLY`, no compute
saving. v2 implements an actual suffix recompute that avoids the full perturbed
prefill.

## Prior v1 result

`AUTOREGRESSIVE_KV_BRANCH_CARRY_STATUS = PROMPT_INTERNAL_BRANCH_CACHE_VALID`;
`LEVEL6_PARTIAL_SPLICE_STATUS = PARTIAL_SPLICE_DIAGNOSTIC_ONLY`.

## Branch-carry vs compute-saving splice — the key obstacle

The `UniversalTransformerCache` stores **K/V only, not the inter-layer residual
stream**. To build a perturbed branch cache you need the residual hidden state at
the perturbation boundary, which the KV cache does not hold — so naively you must
re-run the prompt from the embeddings (no saving). v2's solution: **capture the
residual boundary hidden during the (shared) root prefill**. For an additive
boundary perturbation, `H_boundary_perturbed = H_boundary + delta` is reconstructed
with no forward, and only the suffix is recomputed.

## Boundary slot theory + empirical hook timing

The perturbation hook fires at a decoder layer **output**, i.e. *after* that
layer's K/V was written. So for a boundary at (loop u, layer L):
- **Shared (reusable from root):** all slots in loops < u, plus loop u layers ≤ L
  (including the boundary slot itself).
- **Affected (branch-specific):** loop u layers > L, plus all later loops.

Part D confirmed this empirically across all loops/layers: zero perturbation
changes nothing; the boundary slot is unaffected; the first changed slot is exactly
`(u, L+1)`; and the full changed set equals the `downstream_only` theory. The
`aggressive` policy (sharing an affected slot) diverges, as predicted.

## Suffix recompute implementation (Mode B)

Test-only orchestration over the model's own modules (`model.model.layers`,
`rotary_emb`, `norm`, `lm_head`) — **no permanent model surgery, no weight edits**:
1. **Minimal shared-prefix prefill** runs loops 0..u−1 plus loop u layers 0..L,
   writing only the shared slots and capturing `H_boundary` (= output of (u, L)).
2. `H_boundary_perturbed = H_boundary + delta` (additive, matches the hook's
   RMS-scaled vector exactly).
3. **Suffix recompute** runs loop u layers L+1..47 then loops u+1..3 from
   `H_boundary_perturbed`, writing the affected slots into a fresh cache.
4. **Merge** shared (root) + affected (suffix) → the branch cache.

## Equivalence results

The spliced branch cache is **bit-exact** vs a full perturbed-prompt reference
(max-abs = 0 on all 192 slots; prefill next-token logits RMS = 0; greedy
continuation matches bit-for-bit). Validated single-branch (layers 24/36/47,
loops 0–3, α∈{0,0.5,1.0}), multi-branch (K=2,4, independent storage, no
contamination), batched + prune/reorder, and left-padded (with explicit
position_ids; RoPE relativity makes a single-prompt left-pad shift harmless). The
aggressive over-share negative control diverges.

## Compute accounting (measured)

Baseline (no splice) = K full perturbed prefills. Splice = one shared prefix
prefill + K suffix recomputes. Layer-passes (per branch, deterministic):

| boundary | suffix passes | full passes | per-branch save |
|---|---|---|---|
| loop 0, layer 24 | 167 | 192 | 13% |
| loop 1, layer 24 | 119 | 192 | 38% |
| loop 2, layer 24 | 71 | 192 | 63% |
| loop 3, layer 24 | 23 | 192 | 88% |
| loop 3, layer 47 | 0 | 192 | 100% |

K-scaling at boundary (loop 2, layer 24): K=1 → 0% (prefix+suffix == full); K=2 →
32%; K=4 → 47%; K=8 → 55% fewer layer-passes (wall-clock saving tracks: ~33/48/56%).
Saving grows with K (prefix amortized) and with boundary loop depth.

**Mode A (copy-affected) saves nothing** and is excluded from the claim.

## Final claims allowed

- ✅ **Compute-saving branch-carry** via suffix-recompute splice — equivalence
  passes and savings are measured. Saving is **amortized**: it requires K ≥ 2
  branches sharing one prompt (K=1 does prefix+suffix == full work).
- ✅ Splice correctness / equivalence to full perturbed reference (bit-exact).
- ❌ **Production readiness** — NOT claimed (research probe; small prompts/branches).
- The implementation handles additive boundary perturbations; broader perturbation
  forms and batched-decode position handling are future work.

## Explicit no-steering statement

No steering of any kind (no steering vectors, no action steering, no trained
corridor, no steering as a tested condition, no steering claim). No Ouro training,
no weight/tokenizer/checkpoint edits, no BG tap or registry changes, no
wrapper/local-agent, no Hunter-Seeker imports/instantiation, no `train_arc.py`, no
ARC/MATH generation, no git. DualAnchor scoring in the smoke was skipped (logprob
proxy). Perturbations are a test harness to create branch-specific caches, not
interventions on behaviour.

## Explicit science-repair non-interference statement

The Part A GPU/process guard verified the MMLU science-repair (`recipe_v3`
calibration) process was **NOT active** (stale PID pointers, dead PIDs, GPU idle,
no matching process) before any GPU work. Its output root
(`bg_mmlu_science_branch_parser_repair_v3_2026-06-01/`) and all its artifacts were
**not touched**. If that run is resumed, this probe does not share its artifacts.
