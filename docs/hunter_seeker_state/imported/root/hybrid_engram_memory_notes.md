<!-- Imported from `hybrid_engram_memory_notes.md` during Hunter-Seeker state consolidation on 2026-05-14. -->
<!-- Original SHA256: 64e1f965a6fdeb0a691e9b5cf0b4012a1fa30980d8f33bcac1f1c0bfb2eb0476; original line count: 825. -->

# DeepSeek V4 Hybrid Attention → Hunter Seeker Engram Memory Notes

## Executive summary

DeepSeek V4’s “hybrid attention” is useful to this project, but **not as a direct attention transplant**.

The useful lesson is:

```text
Do not give the agent the whole past.

Give it:
1. exact recent context,
2. coarse compressed global/episode memory,
3. sparse fine-grained retrieval of relevant old memories.
```

For Hunter Seeker, this maps naturally onto the existing **engram / LMM context memory layer**. Since topology engrams already exist, the right move is **not** to add a new DeepSeek-style module. The right move is to evolve the existing engram layer into a **multi-resolution engram memory system**.

---

## Important clarification: DeepSeek V4 and coding

DeepSeek V4 is not “incapable of coding.” The opposite is more likely true: coding, long-context codebase work, and agentic workflows are probably among its main strengths.

The caution is different:

```text
Strong at coding/reasoning != safe to trust as an ungrounded factual oracle.
```

So for code, it is useful. For facts, repo claims, or current-world claims, it should still be grounded by retrieval, tests, search, or tool output.

---

## What DeepSeek V4 “hybrid attention” actually is

From the public inference implementation, the mechanism is not one magical new attention block. It is a **memory/attention schedule** combining:

```text
recent exact tokens
+ compressed old tokens
+ sparse retrieval over compressed old tokens
```

The main pieces are:

### 1. Local sliding window

DeepSeek keeps a recent exact context window.

In V4-Pro, the relevant config uses:

```text
window_size = 128
```

Meaning the most recent local context remains available at high resolution.

For language/code, this preserves local syntax and immediate coherence.

For Hunter Seeker, the equivalent is:

```text
recent exact frames/actions/topology states
```

Probably much smaller than 128. ARC episodes are dense, so a first useful value would likely be:

```text
N = 8 to 32 transitions
```

---

### 2. HCA — heavily compressed attention

DeepSeek uses very aggressive compression in some layers:

```text
compress_ratio = 128
```

This means many old tokens are folded into a much smaller compressed memory stream.

Conceptually:

```text
1,000,000 raw tokens
→ about 7,812 compressed memory entries
```

This is good for broad global context, not exact details.

For Hunter Seeker, this maps to:

```text
coarse episode summaries
coarse mechanism summaries
global trajectory summaries
```

Examples:

```text
“this region behaved like a teleporter system”
“red objects became hazardous after switch activation”
“the player moved through a dependency chain”
“this level contains a key-lock topology”
```

---

### 3. CSA — compressed sparse attention

DeepSeek also uses less aggressive compression in other layers:

```text
compress_ratio = 4
index_topk = 1024
```

The old context is compressed less severely, then a learned indexer retrieves the most relevant compressed blocks.

Conceptually:

```text
old context
→ 4x compressed blocks
→ score blocks against current query
→ retrieve top-k
→ attend only to selected old blocks
```

This is the part most similar to the existing topology engram system.

For Hunter Seeker:

```text
current state/action query
→ retrieve similar remembered topology/mechanism/hazard cases
→ feed retrieved evidence into action scoring
```

---

### 4. Learned compression, not simple averaging

DeepSeek’s compressor is not just average pooling. It learns a softmax-gated weighted summary over a block.

The equivalent for Hunter Seeker should not be:

```text
average all frame embeddings in a chunk
```

It should be:

```text
learn which events in the chunk matter
```

The compressor should prefer transitions with high causal/event relevance:

```text
object changed
topology changed
hazard appeared
terminal happened
world-model prediction error spiked
mechanism activated
new object became reachable
click caused delayed change
```

---

## Translation into Hunter Seeker terms

DeepSeek’s pattern translates cleanly:

```text
DeepSeek local window
= exact recent transitions / frames / actions

DeepSeek HCA
= coarse compressed episode or mechanism memory

DeepSeek CSA
= fine sparse engram recall

DeepSeek top-k indexer
= retrieve relevant topology / hazard / mechanism memories

DeepSeek sparse attention
= action scorer receives compact retrieved evidence
```

The project should treat this as a **memory architecture lesson**, not a transformer architecture patch.

---

## Current state: topology engrams already exist

Since the system already uses an **engram / LMM context memory layer for topology**, the project has already implemented the most relevant piece:

```text
current state
→ retrieve similar topology memories
→ bias/scaffold scoring
```

This is roughly analogous to DeepSeek’s **CSA-style sparse recall**.

The missing piece is explicit **multi-resolution structure**:

```text
1. exact recent memory
2. fine topology/hazard/mechanism engrams
3. coarse compressed episode/mechanism summaries
```

---

## Recommended architecture: Hybrid Engram Memory

Do **not** add a separate “DeepSeekHybridAttention” module.

Instead, extend the existing engram memory into:

```text
HybridEngramMemory
```

or, if keeping names more project-native:

```text
EngramMemory
```

with explicit sub-stores.

---

## Proposed memory stores

### 1. LocalExactTransitionWindow

Stores the last few transitions exactly.

Suggested first values:

```text
capacity = 8 to 32
```

Each entry should contain:

```python
{
    "frame_embed": ...,
    "object_table_embed": ...,
    "topology_embed": ...,
    "action": ...,
    "click_xy": ...,
    "predicted_next_embed": ...,
    "actual_next_embed": ...,
    "delta_embed": ...,
    "changed": bool,
    "terminal": bool,
    "hazard_score": float,
    "progress_score": float,
    "world_model_error": float,
}
```

Purpose:

```text
Preserve immediate causal context without compression.
```

Useful for:

```text
undoing recent mistakes
detecting loops
tracking delayed consequences
maintaining local mechanism state
```

---

### 2. FineTopologyEngrams

This is the existing topology engram path.

It should store high-resolution remembered topology cases:

```python
{
    "query_key": topology_state_or_state_action_embedding,
    "topology_signature": ...,
    "action": ...,
    "outcome": ...,
    "support_type": "positive" | "negative" | "neutral",
    "hazard_delta": ...,
    "terminal": bool,
    "progress_delta": ...,
    "similarity_metadata": ...,
}
```

Purpose:

```text
Sparse recall of similar topology/action situations.
```

This is the DeepSeek CSA analogue.

---

### 3. Hazard / Terminal Engrams

This should be separate or at least separately aggregated.

Do not bury negative evidence inside generic topology memory.

Store:

```python
{
    "state_key": ...,
    "action": ...,
    "terminal": bool,
    "hazard_delta": float,
    "death_like": bool,
    "reachable_hazard_delta": float,
    "evidence_strength": float,
}
```

Purpose:

```text
Prevent optimistic action scoring from overriding known danger.
```

This is especially important because the project already found a failure mode where:

```text
reachable_hazard_delta can be nonzero
while safety_penalty remains 0.0
```

The memory system should make that harder to repeat.

---

### 4. MechanismEngrams

These are not just states. They are causal summaries.

Examples:

```text
clicking object A opened path B
stepping on tile X changed player color
object Y only moved after switch Z
red hazard became active after mechanism trigger
door topology changed after key interaction
```

Suggested structure:

```python
{
    "precondition_key": ...,
    "action_or_event": ...,
    "effect_key": ...,
    "affected_objects": ...,
    "topology_before": ...,
    "topology_after": ...,
    "delay": int,
    "confidence": float,
}
```

Purpose:

```text
Recall mechanisms, not just visual similarity.
```

This is probably more important for ARC than raw frame similarity.

---

### 5. CoarseEpisodeSummaries

This is the DeepSeek HCA analogue.

Every K steps, compress a chunk into a coarse summary.

Suggested first values:

```text
K = 4 or 8
```

Not 128. ARC trajectories are much shorter and denser than text.

The summary should include:

```python
{
    "summary_embed": ...,
    "dominant_mechanism": ...,
    "objects_introduced": ...,
    "objects_removed": ...,
    "topology_changes": ...,
    "hazards_seen": ...,
    "terminal_risk_seen": ...,
    "best_progress_events": ...,
    "failed_actions": ...,
}
```

Purpose:

```text
Give the agent coarse global context without forcing it to inspect every old step.
```

Examples:

```text
“we already explored the left-side key-lock chain”
“click actions changed objects but directional actions stalled”
“this episode found hazards near the lower-right structure”
```

---

## Memory query path during action scoring

When scoring a candidate action, build a query from:

```text
current frame embedding
current object table
current topology state
candidate action
candidate click coordinates, if any
predicted next frame/context
reachable hazard features
```

Then retrieve:

```text
recent exact transitions
top-k fine topology engrams
top-k hazard/terminal engrams
top-k mechanism engrams
relevant coarse episode summaries
```

Then aggregate evidence into a compact context:

```python
memory_context = {
    "recent_exact": ...,

    "positive_support": ...,
    "negative_support": ...,

    "topology_match_score": ...,
    "mechanism_match_score": ...,
    "hazard_match_score": ...,

    "terminal_risk_support": ...,
    "progress_support": ...,

    "coarse_episode_context": ...,

    "engram_positive_best_similarity": ...,
    "engram_negative_best_similarity": ...,

    "engram_conflict_flag": bool,
}
```

This context should feed into:

```text
TransitionRanker
ObjectActionabilityHead
hazard-aware safety scoring
terminal-memory arbitration
predicted-frame-context scoring
Hunter/Seeker action arbitration
```

---

## Critical rule: engrams must not become optimism bias

Bad pattern:

```text
similar topology found
→ add progress bonus
```

Good pattern:

```text
similar topology/action previously led to progress
→ positive support

similar topology/action previously led to death/stall/hazard
→ negative support

positive and negative both present
→ conflict flag
→ suppress optimistic bonus
→ increase caution / require stronger evidence
```

The memory layer should always separate:

```text
positive support
negative support
conflict / ambiguity
```

Do not collapse everything into one scalar too early.

---

## Conflict handling

If both positive and negative memories are retrieved:

```python
if positive_support > threshold and negative_support > threshold:
    engram_conflict_flag = True
    suppress_optimistic_progress_bonus = True
```

If high-confidence negative recall exists:

```python
if negative_support_high and action_match_high:
    increase_safety_penalty()
    reduce_action_score()
```

If negative recall is weak or cross-action only:

```python
keep diagnostic
do not hard-block action
```

This matches the existing direction:

```text
high-confidence cross-action negative recall can penalize
low-evidence cross-action recall stays diagnostic-only
```

---

## Where this helps most

### 1. Topology memory

The existing topology engrams become more robust if they are no longer the only memory scale.

They become the **fine sparse recall** layer.

---

### 2. Hazard / terminal recall

This is probably the highest immediate value.

The system needs strong memory for:

```text
this looked promising but killed me
this object was reachable but hazardous
this click caused terminal risk later
this route created a dead state
```

---

### 3. Predicted-frame-context scoring

Candidate actions can be scored not only by predicted immediate change, but by whether the predicted next frame resembles:

```text
past progress states
past terminal states
past mechanism activation states
past stalled states
```

---

### 4. Observation learning from video

This is where DeepSeek’s pattern becomes very relevant.

For video, the agent cannot retain every frame equally.

It needs:

```text
recent exact frames
+ compressed event chunks
+ sparse recall of similar mechanisms
```

A video-observation pipeline should look like:

```text
video frames
→ frame/object/topology embeddings
→ event segmentation
→ local exact buffer
→ compressed event memories
→ sparse recall during action prediction / mechanism inference
```

This can be done mostly through the encoder + Hunter Seeker memory side, without requiring Ouro to process giant histories.

---

## What not to do

Do not patch DeepSeek-style attention into Ouro.

Reasons:

```text
1. Ouro is fragile and dependency-sensitive.
2. The project already depends on stable transformers/Ouro behavior.
3. ARC trajectories are structured episodes, not million-token documents.
4. The bottleneck is action/world/event memory, not raw token attention.
5. You are not training a 60-layer giant from scratch.
```

Also do not rewrite the architecture around this idea.

This should be an incremental extension of the existing engram/LMM context memory layer.

---

## Implementation plan for Codex

### Goal

Extend the existing topology engram/LMM context memory layer into a multi-resolution engram memory system inspired by DeepSeek V4’s hybrid attention pattern.

### Constraints

```text
Do not modify Ouro.
Do not modify the core GridEncoder unless plumbing is required.
Do not rewrite Hunter Seeker.
Do not replace existing topology engrams.
Preserve existing diagnostics and scoring paths.
Add this as an incremental memory/scoring improvement.
```

### Components to add or extend

```text
1. LocalExactTransitionWindow
2. CoarseEpisodeSummary / MechanismSummary compressor
3. Separate positive and negative engram aggregation
4. Hazard/terminal engram retrieval path
5. Conflict flag logic
6. Compact RetrievedMemoryContext object
```

### Integration points

Feed retrieved memory context into:

```text
action scoring
hazard-aware safety penalty
terminal-memory arbitration
progress scoring
predicted-frame-context scoring
topology/mechanism diagnostics
```

### First-pass parameters

```text
local_window_capacity = 16
episode_summary_chunk_size = 4 or 8
topology_top_k = 8
hazard_top_k = 8
mechanism_top_k = 8
coarse_summary_top_k = 4
```

### First-pass behavior

```text
1. Keep exact recent transitions.
2. Keep existing topology engram retrieval.
3. Add negative/hazard support aggregation.
4. Add coarse summaries every K steps.
5. Add conflict flag if positive and negative support coexist.
6. Suppress optimistic progress bonuses under conflict.
7. Penalize candidate actions only when negative recall is high-confidence and action-relevant.
8. Log all evidence components.
```

---

## Suggested diagnostics

Add or preserve these logs:

```text
memory_context_source
recent_exact_count
topology_engrams_retrieved
hazard_engrams_retrieved
mechanism_engrams_retrieved
coarse_summaries_retrieved

engram_positive_support
engram_negative_support
engram_positive_best_similarity
engram_negative_best_similarity
engram_conflict_flag

terminal_memory_support
hazard_memory_support
mechanism_memory_support
coarse_episode_support

optimistic_bonus_suppressed
safety_penalty_from_memory
score_components_source
```

---

## Minimal Codex prompt

```text
Extend the existing engram/LMM context memory layer into a multi-resolution memory system.

Do not add a new architecture and do not modify Ouro or the encoder unless strictly needed for plumbing.

Keep the existing topology engram retrieval as the fine-grained sparse recall path.

Add:
1. a small exact recent-transition window,
2. coarse compressed episode/mechanism summaries,
3. separate positive and negative evidence aggregation,
4. conflict detection that suppresses optimistic bonuses,
5. explicit hazard/terminal recall features that feed action scoring,
6. detailed diagnostics for positive/negative support, best similarities, conflict flag, memory-derived safety penalty, and memory-context source.

The design should behave like DeepSeek V4’s useful memory pattern translated to Hunter Seeker:
local exact context + fine sparse recall + coarse compressed context.

The implementation must preserve existing behavior unless retrieved negative evidence or conflict explicitly changes scoring.
```

---

## Final verdict

DeepSeek V4’s hybrid attention is useful to this project as **confirmation and refinement** of the engram-memory direction.

The direct lesson is not:

```text
add hybrid attention
```

It is:

```text
turn engrams into multi-resolution memory:
local exact + fine sparse + coarse compressed
```

Topology engrams are already the fine sparse layer.

The next valuable step is to add:

```text
coarse episode/mechanism summaries
explicit hazard/terminal negative recall
conflict-aware scoring
exact recent transition memory
```

This is most relevant to:

```text
topology memory
terminal memory
hazard recall
mechanism discovery
predicted-frame-context scoring
video observation learning
delayed-effect action reasoning
```
