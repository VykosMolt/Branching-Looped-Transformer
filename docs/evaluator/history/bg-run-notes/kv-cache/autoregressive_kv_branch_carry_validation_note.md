# Autoregressive KV/Cache Branch-Carry Validation v1 — Interpretation Note

## Status

`AUTOREGRESSIVE_KV_BRANCH_CARRY_STATUS = PROMPT_INTERNAL_BRANCH_CACHE_VALID`

The autoregressive KV/cache branch-carry ladder has now passed through Level 5. This means branch-specific KV/cache continuation is validated for ordinary cached decoding, token-boundary cache forks, batched branch caches, pruning/reordering survivors, current-token perturbations during decode, and prompt-internal perturbations that produce branch-specific caches.

The remaining caveat is compute saving:

`LEVEL6_PARTIAL_SPLICE_STATUS = PARTIAL_SPLICE_DIAGNOSTIC_ONLY`

The partial splice test validated slot-boundary logic, but it did not implement a true compute-saving path because affected slots were copied from a full perturbed prefill. Therefore, branch-carry is validated; compute-saving branch-carry is not yet validated.

## What Changed

Before this run, the project state was:

```text
prompt-only layer carry:
  validated

autoregressive KV/cache branch-carry:
  not validated
```

After this run, the state is:

```text
prompt-only layer carry:
  validated

autoregressive token-boundary branch carry:
  validated

batched branch cache carry:
  validated

prune/reorder survivor cache carry:
  validated

current-token perturb cache carry:
  validated

prompt-internal perturb branch-specific cache:
  validated

partial-cache compute-saving splice:
  not yet validated; diagnostic only
```

This removes a major mechanical caveat around generation-time branch continuation. The architecture-shaped branch loop no longer rests only on prompt-only cumulative-hook evidence; there is now a validated autoregressive cache-carry substrate through prompt-internal perturbation.

## Validated Ladder

### Level 0 — Ordinary cached decode

Ordinary cached decoding matches full recomputation within bf16 decode tolerance.

Important detail:

```text
prefill:
  bit-exact, RMS = 0

bf16 decode:
  expected numerical drift around 0.05–0.2 RMS
```

The old strict `1e-4` tolerance is unrealistic for bf16 decode because q=1 cached decode and q=sequence full recomputation use different numerical paths. The correct criterion is bounded drift plus stable behavior.

### Level 1 — Token-boundary branch fork

Shared prefill cache can be cloned into independent branch caches. Branch caches do not contaminate each other.

### Level 2 — Batched branch caches

Batched branch continuation matches independent branch continuation within bf16 numerical tolerance.

### Level 3 — Prune/reorder survivor cache

Survivor cache pruning and reordering works. Cache state, token histories, and lineage remain aligned after subset/order changes and multi-round pruning.

### Level 4 — Current-token perturbation

A hidden perturbation applied during an autoregressive decode step can be carried forward through the branch-specific cache.

### Level 5 — Prompt-internal perturbation branch cache

A branch born from a prompt/internal hidden perturbation can produce a valid branch-specific cache and continue autoregressively.

The negative control matters: using the unperturbed cache for a perturbed branch produced a large mismatch, approximately RMS ≈ 3.0. This confirms that the branch-specific cache is genuinely required, not decorative.

### Level 6 — Partial-cache splice diagnostic

The splice boundary logic is promising but not yet a compute-saving implementation.

The diagnostic showed:

```text
shared/affected slot logic:
  valid diagnostically

roughly shareable slots:
  13–25%

over-sharing:
  diverges

compute savings:
  not claimed
```

The current Level 6 copied affected slots from a full perturbed prefill. That validates slot accounting, but not actual compute-saving branch carry.

## Calibration Decisions

### bf16 tolerance

Future cache equivalence tests should distinguish prefill from decode:

```text
prefill:
  strict / bit-exact comparison is reasonable

bf16 autoregressive decode:
  require bounded numerical drift,
  top-token stability,
  generated-token equivalence,
  and no real mismatches above numerical noise
```

### Perturbation alpha

The original small perturbation alphas, `0.001–0.01`, were sub-noise for this cache-carry validation because residual RMS was around `0.1–0.5`. The validation used `alpha >= 1.0` to demonstrate mechanically meaningful carryable perturbations.

This does not mean all future branch-generation alphas should be `1.0+`. It means cache-carry perturbation tests must calibrate alpha against actual residual RMS rather than reuse old hidden-origin alpha values blindly.

### Padding / mask requirement

For left-padded batched decode, explicit per-row `position_ids` are required. Ouro otherwise derives position behavior from a shared `cache_position`, which can be unsafe for mixed-length rows.

Standing rule:

```text
For left-padded batched branch decode:
  pass explicit per-row position_ids.
```

## What Can Now Be Claimed

It is now fair to claim:

```text
Autoregressive branch-specific KV/cache carry is validated through prompt-internal branch-specific cache.
```

More specifically:

```text
ordinary cached decode works
independent token-boundary branch cache fork works
batched branch cache works
prune/reorder survivor cache works
current-token perturb cache carry works
prompt-internal perturb branch cache works
```

## What Cannot Yet Be Claimed

Do not claim:

```text
compute-saving branch carry
partial-cache splice compute savings
production routing readiness
steering
trained write-path control
autoregressive branch compute reduction
```

The correct compute-saving statement is:

```text
partial-cache slot-boundary logic is diagnostically validated;
actual compute-saving splice remains future work.
```

## Architectural Implication

This result upgrades the Phase 2 runtime substrate.

Before:

```text
Architecture-looped branching was validated mainly as prompt-only / cumulative-hook behavior.
```

Now:

```text
Architecture-looped branching has a validated autoregressive branch-cache continuation substrate.
```

That makes the eventual Phase 2 runtime architecture much more credible. It does not solve science, terminal final choice, or steering, but it removes the major mechanical caveat around carrying branches through generation.

## Current Project State

```text
DualAnchor architecture-looped survival:
  ready

Reasoning:
  ready with terminal survivor-set handoff

Science:
  repair still running / not yet resolved

Autoregressive KV/cache branch-carry:
  validated through prompt-internal branch-specific cache

Partial-cache compute-saving splice:
  diagnostic only

Steering:
  not started
```

## Recommended Next Cache-Specific Step

The next cache-specific experiment should be:

```text
Partial Cache Splice v2
```

Goal:

```text
reuse prefix cache slots before a perturbation boundary,
recompute only the affected downstream suffix,
continue generation,
and verify exact equivalence to full perturbed-branch recomputation.
```

Only if that passes should the project claim:

```text
PARTIAL_SPLICE_COMPUTE_SAVING_VALID
```
