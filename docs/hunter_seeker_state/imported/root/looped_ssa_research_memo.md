<!-- Imported from `looped_ssa_research_memo.md` during Hunter-Seeker state consolidation on 2026-05-14. -->
<!-- Original SHA256: 2ec8821beaf332ccff2e04bd3b85494f0d14362eb50c912c1a64f8d3b82f2433; original line count: 1401. -->

# Looped-SSA / Ouro-NSA++ Research Memo

**Date:** 2026-05-07  
**Status:** Speculative architecture memo + concrete experiment plan  
**Core phrase:** **NSA++ with constant fanout**  
**Core research question:** Do recurrent sparse-attention models genuinely refine evidence routes across latent loops, or do they mostly reuse the first selected evidence set?

---

## 0. Executive summary

This memo consolidates the discussion around SubQ-style sparse attention, Ouro-style looped latent reasoning, RLTT/LoopRPT-style trajectory training, and the resulting research direction: **looped sparse attention as an interpretable evidence-selection substrate**.

The converged architecture hypothesis is:

```text
SubQ / SSA ≈ NSA++ with constant fanout

NSA++ = local window
      + compressed/global index stream
      + selected full-resolution remote blocks
      + bounded or hierarchical selector
      + route reuse across layers
      + exact softmax over selected positions

constant fanout ≈ ~12K–16K selected token-equivalents per query/layer group
```

The converged Ouro translation is:

```text
Looped-SSA = Ouro-style recurrent latent reasoning
             over an NSA++ sparse evidence substrate

where loops refine:
    1. the latent answer state
    2. the selected evidence route

but under a bounded total evidence budget K_total.
```

The most important correction to the naive version is that looped retrieval is probably **front-loaded**, not symmetric. The realistic controller is:

```text
Loop 1: broad route; high recall; spend most of K_total.
Loop 2: bounded top-up; follow pointers/coreference/contradictions.
Loop 3: smaller refinement; verify/conflict resolution.
Loop 4: mostly reuse evidence; stabilize answer or exit.
```

The strongest first paper is **not** “build a better long-context model.” It is:

> **Do sparse-attention routes refine across recurrent loops under retrieval pressure?**

The single highest-value metric is:

```text
route IoU across loops, conditional on task difficulty
```

If route IoU stays high across all tasks, looped retrieval is mostly illusory: the model routes once, then reasons over the same evidence. If route IoU drops on hard tasks while accuracy improves, iterative retrieval is real. If route IoU drops but accuracy does not improve, routes churn rather than refine.

The practical sequencing recommendation is:

```text
1. Finish the current ARC/agent work.
2. Get the CLT/Ouro paper public on arXiv.
3. Build an NSA-lite side implementation only.
4. Wait for SubQ's technical report/model card.
5. Run the route-dynamics experiment with sharpened hypotheses.
```

---

## 1. Source status and epistemic framing

At the time of writing, SubQ has public claims and benchmark tables, but not a full reproducible architecture disclosure. The official SubQ technical post says a comprehensive model card is coming soon, and the product/launch material frames SSA as Subquadratic Sparse Attention with long-context scaling claims.[^subq-ssa] The launch post advertises a research result at 12 million tokens and says the architecture reduces attention compute by almost 1,000× versus dense attention at that scale.[^subq-intro]

So this memo separates:

```text
Evidence:
    Public numbers, papers, source descriptions.

Inference:
    What those numbers strongly imply.

Speculation:
    What exact mechanism is most likely.

Research plan:
    What can be tested without knowing SubQ's internals.
```

The architectural hypothesis is not “we know what SubQ did.” It is:

> Given the public FLOP ratios, speed claims, and adjacent literature, the most likely design family is native trainable sparse attention with constant fanout, compressed indexing, selected full-resolution blocks, and aggressive route reuse.

---

## 2. The SubQ/SSA hypothesis

### 2.1 The load-bearing arithmetic: constant fanout

SubQ’s public technical post gives attention-FLOP reductions of approximately:

```text
128K context: 8× attention-FLOP reduction
1M context:   62.5× attention-FLOP reduction
```

If sparse attention does exact attention over a selected subset, then:

```text
dense attention cost / sparse attention cost ≈ full_context_length / selected_budget
```

So the implied selected budget is:

```text
128K / 8       ≈ 16K selected token-equivalents
1,000K / 62.5  ≈ 16K selected token-equivalents
```

The 12M-token research claim says attention compute is reduced by “almost 1,000×,” implying roughly:

```text
12M / 1000 ≈ 12K selected token-equivalents
```

This is the architectural skeleton. Whatever the exact selector is, SubQ’s numbers look like **constant fanout**, not merely constant sparsity.

### 2.2 Constant sparsity vs constant fanout

This distinction is the central axis.

```text
Constant sparsity:
    attend to a fixed fraction of the context
    e.g. 1% of tokens

    selected_tokens ∝ n
    reduction factor ≈ constant

Constant fanout:
    attend to a fixed number of token-equivalents
    e.g. ~16K tokens

    selected_tokens ≈ constant
    reduction factor grows with n
```

SubQ’s reduction factor grows from 8× at 128K to 62.5× at 1M. That is constant-fanout behavior.

### 2.3 Best one-line description

```text
SubQ SSA ≈ NSA++ with constant fanout.
```

Where:

```text
NSA++ = NSA-family sparse attention
        + bounded/hierarchical compressed index
        + exact attention over selected blocks
        + aggressive route reuse
        + custom kernels
```

And:

```text
constant fanout = each query or layer group sees only ~12K–16K selected token-equivalents,
                  regardless of whether the full context is 128K, 1M, or 12M.
```

### 2.4 Why vanilla NSA is close but insufficient

Native Sparse Attention (NSA) is the closest published macro-template. The NSA paper decomposes attention into three branches:

```text
1. compressed/coarse attention
2. selected/fine attention
3. sliding-window local attention
```

The NVIDIA cuDNN NSA frontend exposes the same primitive set: selection, compression, top-k, and sliding-window attention, with Blackwell-oriented implementations for key pieces.[^nvidia-nsa]

But vanilla fixed-stride NSA has a scaling problem. If the compressed branch has one compressed token per fixed-size block, then:

```text
compressed_tokens = n / stride
```

If every query attends densely over all compressed tokens:

```text
compressed_branch_cost ≈ n × (n / stride)
```

With fixed stride, that is still quadratic up to a constant factor.

At 1M tokens, stride 64 gives:

```text
1,000,000 / 64 ≈ 15,625 compressed tokens
```

That fits the inferred ~16K fanout budget.

At 12M tokens, the same stride gives:

```text
12,000,000 / 64 ≈ 187,500 compressed tokens
```

That no longer fits the ~12K–16K budget. So fixed-stride vanilla NSA cannot by itself explain the 12M claim.

### 2.5 How the compressed/index stream might stay bounded

There are four plausible fixes.

#### Option A: fixed-budget adaptive compression

Hold the number of compressed summaries constant:

```text
compressed_budget = C
compression_stride = n / C
```

Example:

```text
C = 2K compressed summaries

128K context → stride ≈ 64
1M context   → stride ≈ 500
12M context  → stride ≈ 6000
```

This fits the FLOP curve, but it creates a quality cliff. At 12M tokens, each compressed summary may represent thousands of original tokens. That is brittle for exact evidence, variable names, legal exceptions, hidden constraints, code references, or adversarial distractors.

#### Option B: multiscale hierarchy

Use multiple levels:

```text
coarse summaries → medium summaries → fine block summaries → full-resolution selected K/V
```

A query first searches the coarse level, then refines into a small set of medium/fine regions, then reads full-resolution K/V only from selected blocks.

This is more robust than one-level adaptive compression and still compatible with constant fanout.

#### Option C: learned semantic ANN/hash/tree routing

Use learned route embeddings or hash codes to search only relevant regions. HashAttention, for example, casts pivotal-token identification as a recommendation problem and maps queries/keys into Hamming space to find selected tokens efficiently.[^hashattention]

This is conceptually close, but pure token-level hashing is less likely to be the main production mechanism because random token-level KV access is hard to make fast. A block/page-level semantic index is more plausible.

#### Option D: route reuse across layers

IndexCache shows that DSA-style sparse selections are highly similar across consecutive layers, so sparse top-k indices can be computed only on a subset of “Full” layers and reused by “Shared” layers. It reports removing up to 75% of indexer computations with negligible quality degradation.[^indexcache]

This is probably load-bearing for any 1M–12M context serving story. If the selector is recomputed independently at every layer and every loop, even a hierarchical selector becomes expensive.

### 2.6 Most likely actual mechanism

The strongest current bet is:

```text
SubQ SSA = NSA-family macroarchitecture
           + bounded/multiscale compressed index
           + selected full-resolution remote blocks
           + local sliding attention
           + cross-layer route reuse
           + exact softmax over selected positions
           + custom Blackwell/B200 sparse kernels
```

Pseudo-architecture:

```text
Prefill:
    split context into blocks/pages
    compute full-resolution K/V pages
    compute compressed summaries at one or more scales
    build route metadata / hierarchy / index

For each query or layer group:
    include local window and current block
    query bounded/hierarchical index
    retrieve candidate remote blocks
    optionally refine inside candidates
    cap selected full-resolution evidence at K ≈ 12K–16K token-equivalents
    run exact softmax over selected positions

Across layers:
    compute selector on anchor layers
    reuse or lightly refine routes on intervening layers
```

### 2.7 Alternative ancestors in the design space

The exact ancestry is probably not clean. It is better to think of SubQ as sitting in the convex hull of several published directions.

#### NSA: native trainable sparse attention

Closest macro-template: compression + selection + local window.[^nsa]

Strengths:

```text
trainable end-to-end
hardware-aware
three-path structure already matches the guessed architecture
```

Weakness:

```text
vanilla dense compressed branch grows with n unless bounded/hierarchical/adaptive
```

#### MoBA: mixture of block attention

MoBA selects top-k KV blocks per query, applying a mixture-of-experts-like routing idea to attention.[^moba]

Strengths:

```text
block-level sparse attention
production relevance via Kimi long-context requests
simple conceptual fit
```

Weakness:

```text
plain MoBA often behaves like constant sparsity rather than constant fanout
and flat block scoring can still become expensive
```

#### DSA: learned fine-grained sparse attention

DeepSeek Sparse Attention reduces core attention by selecting top-k relevant KV entries, but the indexer itself can retain an O(L²) component.[^deepseek]

Strengths:

```text
fine-grained selection
production-grade sparse attention family
```

Weakness:

```text
flat indexer can move, not remove, the quadratic bottleneck
```

#### HISA: hierarchical replacement for flat DSA indexing

HISA replaces flat token scanning with a block-level coarse filter followed by token-level refinement, directly attacking the DSA indexer bottleneck.[^hisa]

Strengths:

```text
exactly the kind of hierarchy needed for bounded/sublinear routing
```

Weakness:

```text
still a component, not a full end-to-end long-context model architecture
```

#### IndexCache: cross-layer index reuse

IndexCache exploits cross-layer redundancy in sparse selections, reducing indexer cost by reusing top-k selections across nearby layers.[^indexcache]

Strengths:

```text
large practical multiplier
likely essential at million-token scale
```

Weakness:

```text
does not by itself solve initial route selection
```

#### HashAttention / HiP-style methods

HashAttention and HiP-style hierarchical pruning suggest possible selector mechanisms: learned semantic retrieval, Hamming-space routing, or hierarchical pruning.[^hashattention][^hip]

Strengths:

```text
show that sublinear or cheaper candidate selection is plausible
```

Weakness:

```text
may be hard to combine with production GPU-friendly contiguous block reads
```

### 2.8 Falsifiers for the SubQ hypothesis

The technical report should answer:

```text
1. Does the selector ever score O(n) candidates per query?
2. Is compressed attention dense over all compressed tokens?
3. Are selected indices reused across layers?
4. Is the selected-token budget constant, constant-sparsity, or adaptive?
5. Are there any full-context dense attention layers in the deployed inference path?
6. Are speedups attention-only, prefill-only, or end-to-end?
7. Does 12M mean accepted context, functional retrieval, or robust reasoning?
```

If the selector scores O(n) candidates per query, or if full dense attention layers exist at 1M–12M context, then “fully subquadratic” becomes marketing. The model may still be valuable, but the claim shifts from structural improvement to cheaper-quadratic engineering.

---

## 3. Ouro, RLTT, and LoopRPT: what transfers

### 3.1 Ouro gives adaptive latent depth

Ouro/LoopLMs perform iterative computation in latent space before token emission. They reuse a shared transformer stack over recurrent steps and use an early-exit mechanism so simple cases can stop earlier while harder cases spend more internal compute.[^ouro]

For this memo, Ouro contributes the **depth axis**:

```text
How many latent refinement loops should the model spend before emitting?
```

SubQ/SSA contributes the **breadth axis**:

```text
Which evidence positions should the model read from the huge context?
```

The combined model has two adaptive compute dimensions:

```text
breadth: selected evidence budget K

depth: latent recurrent loops R
```

### 3.2 RLTT gives trajectory-level credit for latent loops

RLTT exists because standard RL objectives such as GRPO assign credit only to the final latent state, even though a LoopLM produces a sequence of latent states before token generation. RLTT distributes reward over the full latent thought trajectory and is evaluated on Ouro-2.6B-Thinking.[^rltt]

For dense Ouro, the trajectory is:

```text
h¹ → h² → h³ → h⁴ → token
```

For Looped-SSA, the trajectory becomes:

```text
(h¹, selected_evidence¹)
→ (h², selected_evidence²)
→ (h³, selected_evidence³)
→ (h⁴, selected_evidence⁴)
→ token
```

That makes the route trajectory a first-class object.

### 3.3 LoopRPT shapes latent steps during pretraining

LoopRPT reframes next-token prediction as a next-token reasoning task for LoopLMs, assigning reinforcement signals directly to latent steps using an EMA teacher reference and noisy latent rollouts.[^looprpt]

Important correction:

```text
LoopRPT proper:
    same-model EMA teacher
    noisy latent rollouts
    stability depends partly on teacher-student similarity

High-fanout or dense teacher for sparse routes:
    this is distillation, not vanilla LoopRPT
```

A Looped-SSA training recipe may use both, but they should not be conflated.

---

## 4. The Looped-SSA architecture

### 4.1 Naive translation

The naive version is:

```text
Take Ouro.
Replace dense attention with NSA++ sparse attention.
Keep recurrent loops.
Keep exit gate.
Train with RLTT.
```

Pseudo-code:

```text
h_i^0 = embedding/context state

for r in 1..R:
    C_i^r = sparse_router(h_i^{r-1}, memory_index)
    h_i^r = shared_sparse_transformer(h_i^{r-1}, C_i^r)
    logits_i^r = LMHead(h_i^r)
    exit_i^r = ExitGate(h_i^r)

emit from the first r whose exit probability crosses threshold
```

But this version is too optimistic because it ignores total route budget.

### 4.2 The K_total problem

If each loop independently gets a 16K selected-evidence budget, then four loops silently become:

```text
4 × 16K = 64K selected token-equivalents
```

That breaks the constant-fanout story.

The real constraint must be:

```text
|U_R| ≤ K_total
```

where:

```text
U_R = union of selected evidence across all loops
K_total ≈ 12K–16K token-equivalents, or whatever budget the serving target allows
```

This makes route control central.

### 4.3 Front-loaded breadth, bounded top-ups

The realistic loop schedule is not symmetric iterative search. It is front-loaded:

```text
Loop 1:
    broad route
    high recall
    spend most of K_total

Loop 2:
    bounded top-up
    follow pointers/coreference/symbols
    add small amount of new evidence

Loop 3:
    verify
    resolve conflicts
    add even smaller top-up, if needed

Loop 4:
    mostly reuse evidence
    stabilize answer or exit
```

Example budget schedule:

```text
K_total = 16K token-equivalents

Loop 1: 10K–12K broad evidence
Loop 2: +2K–4K top-up
Loop 3: +0.5K–2K verification/conflict evidence
Loop 4: +0K–0.5K escape hatch only
```

This is closer to:

```text
one broad route + iterative evidence refinement
```

than:

```text
full global search at every loop
```

### 4.4 Route reuse vs iterative refinement

There is a real tension:

```text
Heavy route reuse:
    efficient
    but loops mostly reason over the same evidence
    resembles vanilla Ouro over a sparse evidence set

Fresh rerouting every loop:
    higher-quality search
    but kills efficiency
```

The plausible resolution is layered:

```text
coarse global route:
    computed once or rarely
    reused across layers and loops

fine top-up route:
    small, bounded, loop-dependent
    allowed to change with latent state
```

This gives:

```text
Loop 1: choose candidate pool P₁ and initial evidence U₁
Loop 2: refine P₁ into P₂; add small U₂ \ U₁
Loop 3: refine/verify; add tiny U₃ \ U₂
Loop 4: mostly reason over U₃
```

### 4.5 Static memory + active loop state

At million-token scale, the full context should not be looped naively. Instead:

```text
Long context:
    encoded/indexed once during prefill
    stored as static multiscale sparse memory

Active state:
    current question/token states
    generated prefix states
    latent scratch/working state
    route history embedding

Loops:
    operate mainly on active state
    read from static memory through sparse selected routes
```

This avoids cache explosion.

### 4.6 Proposed Looped-SSA block

```text
Inputs:
    active hidden state h_r
    static memory M
    previous candidate pool P_{r-1}
    previous selected evidence union U_{r-1}
    remaining budget B_r

Step:
    if r == 1:
        P_r = broad_route(h_r, M)
        ΔU_r = select_blocks(P_r, budget=large)
    else:
        P_r = refine_route(h_r, P_{r-1}, M)
        ΔU_r = topup_or_prune(P_r, U_{r-1}, budget=small)

    U_r = budgeted_union(U_{r-1}, ΔU_r, K_total)

    h_{r+1} = shared_transformer_loop(
        h_r,
        local_window,
        compressed_summaries,
        full_resolution_KV[U_r]
    )

    logits_r = LMHead(h_{r+1})
    exit_r = ExitGate(h_{r+1}, route_confidence, output_entropy, budget_remaining)
```

### 4.7 Complexity target

Good version:

```text
attention cost:
    O(n · K_total · T_eff) or better, depending on batching/layer grouping

selector cost:
    O(n log n) or O(n) prefill/index construction
    O(log n), O(1), or bounded candidate scoring per active query/loop
    heavily reused across layers

memory:
    O(n) static memory
    + O(K_total) active selected evidence
    + small per-loop active state
```

Dangerous version:

```text
O(n · K · R · L)
+ selector recomputed per layer per loop
+ full index copied per loop
+ union fanout grows with loop count
```

The good version requires:

```text
1. bounded union fanout
2. route reuse across layers
3. front-loaded route budget
4. static long-context memory
5. looped active-state refinement
```

---

## 5. Training Looped-SSA

### 5.1 What RLTT can train cleanly

RLTT is clean for latent states because each loop exposes a differentiable token distribution:

```text
p(y | h¹), p(y | h²), ..., p(y | hᴿ)
```

A trajectory-level objective can directly weight those loop distributions.

Use RLTT for:

```text
latent reasoning trajectory
exit timing
output distribution stabilization
underthinking vs overthinking
```

### 5.2 Why route training is harder

Route selection is usually discrete:

```text
top-k blocks
selected pages
evicted evidence
candidate-pool pruning
```

Naively extending RLTT to routes is hard because:

```text
Gumbel-softmax top-k:
    scaling and memory issues at long context

REINFORCE:
    high variance, worse with huge action spaces

Dense-teacher attention imitation:
    needs a teacher capable of good long-context retrieval

Final-answer reward only:
    cannot distinguish bad routing from bad reasoning
```

So route training is likely the actual bottleneck.

### 5.3 More realistic route-training stack

The route module should be trained mostly as a ranking/indexing system, not purely with RL.

Recommended stack:

```text
1. Dense/high-fanout teacher imitation at manageable lengths
2. Synthetic tasks with known gold evidence blocks
3. Contrastive block-ranking losses
4. Counterfactual route ablation labels
5. Route dropout and distractor training
6. Small-action RL for controller decisions only
```

Use RL for decisions like:

```text
continue vs exit
reuse vs reroute
broad rescan vs local refinement
increase budget vs save budget
evict vs keep block
```

Do not use RL to directly choose 16K tokens out of 12M.

### 5.4 Route supervision vs route emergence

There are two possible research stories:

```text
Emergent route refinement:
    loops naturally change routes under task pressure

Supervised route refinement:
    route supervision is required before loops use retrieval productively
```

Both are interesting. The first experiment should test emergence first, then add supervision as the key follow-up.

### 5.5 Avoiding premature route collapse

RLTT encourages earlier latent states to align with final answers. That can be dangerous in retrieval tasks because the first plausible evidence may be local but wrong.

Failure mode:

```text
Loop 1 finds plausible local distractor.
RLTT rewards early answer stabilization.
Model exits or stops routing.
Remote decisive evidence is never selected.
```

Countermeasures:

```text
early-loop route entropy bonus
anti-locality penalty on tasks requiring remote evidence
route dropout
near-distractor training
counterfactual evidence ablations
compute penalty only after evidence sufficiency is high
```

---

## 6. Interpretability thesis

### 6.1 Why Looped-SSA is more interpretable than dense Ouro

Dense Ouro exposes continuous latent states:

```text
h¹ → h² → h³ → h⁴
```

These can be probed, but the evidence path is implicit.

Looped-SSA exposes two coupled trajectories:

```text
latent trajectory:
    h¹ → h² → h³ → h⁴

route trajectory:
    selected_blocks¹ → selected_blocks² → selected_blocks³ → selected_blocks⁴
```

That route trajectory is a discrete evidence trace.

This enables sharper failure decomposition:

```text
Did the model miss the right evidence?
Did it select the right evidence but ignore it?
Did it select a local distractor first?
Did a later loop overwrite remote evidence?
Did it exit before evidence refinement?
Did routes churn without improving evidence quality?
```

### 6.2 The strongest publishable framing

The strongest framing is not:

> Can we build a better long-context architecture?

It is:

> **Can recurrent sparse-attention models expose their reasoning process as a sequence of selected evidence sets, and do those evidence sets genuinely refine under task pressure?**

This sits at the intersection of:

```text
Ouro / LoopLMs:
    latent recurrent reasoning

RLTT / LoopRPT:
    trajectory-level training

SubQ / NSA++:
    sparse selected evidence

Mechanistic interpretability:
    observable route trajectories + probeable latent states
```

### 6.3 Relation to CLT-style latent probing

CLT-style latent-feature analysis should be a second paper, not part of the first experiment.

Suggested split:

```text
Paper 1:
    Route Dynamics in Looped Sparse-Attention Models
    Question: Do routes refine across loops?

Paper 2:
    Coupled Latent and Evidence Trajectories in Looped Sparse Models
    Question: How do discrete route trajectories align with continuous latent relational/preference features?
```

This avoids overstuffing the first paper.

---

## 7. Minimal experiment: route dynamics across loops

### 7.1 Core hypothesis

```text
H1: Under retrieval pressure, recurrent loops produce meaningful route refinement.

H0: Recurrent loops mostly reuse the first selected evidence set;
    any improvement comes from latent refinement over fixed evidence.
```

### 7.2 Core metric: route IoU across loops

For selected block sets:

```text
IoU(r, r+1) = |S_r ∩ S_{r+1}| / |S_r ∪ S_{r+1}|
```

Measure IoU conditional on task difficulty.

Interpretation:

```text
Easy tasks + high IoU:
    expected; no rerouting needed

Hard tasks + lower IoU + higher accuracy:
    evidence of real iterative retrieval

Hard tasks + high IoU:
    looped retrieval is mostly illusory

Hard tasks + low IoU + low accuracy:
    routes churn rather than refine
```

### 7.3 Difficulty axis

Use tightly controlled synthetic tasks rather than broad benchmarks.

Recommended axis:

```text
D0: 1-hop, no distractors
D1: 2-hop, no distractors
D2: 2-hop + near distractors
D3: 3-hop + near distractors
D4: remote decisive evidence contradicts local plausible answer
```

Optional dimensions:

```text
evidence distance
number of distractors
number of hops
surface-form overlap
coreference depth
amount of irrelevant context
```

But keep the first version narrow.

### 7.4 Model variants

Keep ablations minimal.

```text
A. One-shot sparse route, no loop rerouting

B. Looped model with frozen loop-1 route

C. Looped model with bounded top-up rerouting
```

This isolates the key causal question:

```text
Does allowing routes to change across loops help?
```

Route supervision should be a second axis only after the basic effect is measured.

Optional follow-up:

```text
D. Looped model with bounded top-up rerouting + gold/synthetic route supervision
```

### 7.5 Model scale

A single-GPU toy setup is enough for the first result.

Suggested scale:

```text
parameters: 30M–150M
context:    2K–16K initially
loops:      2–4
blocks:     small fixed pages, e.g. 64–256 tokens
attention:  NSA-lite
```

NSA-lite can be:

```text
local sliding window
+ compressed block summaries
+ selected block attention
```

No need for production kernels at first. Correctness and route observability matter more than speed.

### 7.6 Minimal measurements

Primary:

```text
route IoU across loops vs task difficulty
answer accuracy vs task difficulty
```

Secondary only if cheap:

```text
gold-evidence recall
route precision
new-evidence fraction per loop
exit loop distribution
```

Defer:

```text
CLT-style latent probes
causal scrubbing
large benchmark suite
multi-size scaling law
```

### 7.7 Expected result patterns

#### Pattern A: Real iterative retrieval

```text
easy tasks:
    high route IoU
    high accuracy

hard tasks:
    lower route IoU
    new evidence appears in later loops
    accuracy improves over frozen-route baseline
```

Interpretation:

```text
loops refine evidence under pressure
```

#### Pattern B: Illusory iterative retrieval

```text
all tasks:
    high route IoU
    bounded-top-up model ≈ frozen-route model
```

Interpretation:

```text
loops reason over fixed evidence;
retrieval is basically one-shot
```

#### Pattern C: Route churn

```text
hard tasks:
    low route IoU
    no accuracy gain
    low gold-evidence recall
```

Interpretation:

```text
routes change, but not usefully
```

#### Pattern D: Supervision required

```text
unsupervised top-up:
    churn or frozen routes

route-supervised top-up:
    lower IoU on hard tasks
    higher gold recall
    higher accuracy
```

Interpretation:

```text
route refinement is possible but does not emerge reliably without supervision
```

### 7.8 Why a negative result is publishable

A negative result would show:

```text
Looped sparse attention does not automatically create iterative retrieval.
It may route once and then perform latent reasoning over fixed evidence.
```

That would be valuable because many architectural intuitions assume loops naturally become search. This experiment directly tests that assumption.

---

## 8. Suggested implementation sketch

### 8.1 Toy task generator

Generate contexts with blocks like:

```text
Block 017:
    The key for ALPHA is in Block 481.

Block 481:
    The object referenced by ALPHA is DEVICE-X9.

Block 932:
    DEVICE-X9 has final color: cobalt.
```

Question:

```text
What is the final color for ALPHA?
```

Difficulty knobs:

```text
number of hops
block distance
near distractors
coreference substitutions
local plausible contradiction
surface-form overlap
```

### 8.2 NSA-lite routing

Simplified mechanism:

```text
1. Divide context into blocks.
2. Compute block summaries by pooling hidden states.
3. Route active query state to top-k blocks.
4. Attention reads local window + selected blocks.
5. For loops >1, allow only Δk new blocks.
```

For toy experiments, the route can initially be:

```text
score(block_j) = q_route · summary_j
```

Then:

```text
selected_blocks_r = top_k(score)
```

Bounded top-up:

```text
new_blocks_r = top_delta_k(blocks not already in U_{r-1})
U_r = U_{r-1} ∪ new_blocks_r
```

### 8.3 Frozen-route baseline

```text
Loop 1:
    select U_1

Loops 2..R:
    reuse U_1 exactly
```

This distinguishes:

```text
latent recurrence benefit
```

from:

```text
route refinement benefit
```

### 8.4 Bounded top-up model

```text
Loop 1:
    select k_1 blocks

Loop r > 1:
    select Δk_r new blocks
    keep union under K_total
```

Budget example:

```text
K_total = 16 blocks
k_1 = 12
Δk_2 = 3
Δk_3 = 1
Δk_4 = 0
```

### 8.5 Reporting table

A minimal paper can report:

| Difficulty | Variant | Accuracy | IoU 1→2 | IoU 2→3 | Gold recall loop 1 | Gold recall final |
|---|---:|---:|---:|---:|---:|---:|
| D0 | frozen | | | | | |
| D0 | top-up | | | | | |
| D4 | frozen | | | | | |
| D4 | top-up | | | | | |

The central plot:

```text
x-axis: task difficulty
left y-axis: route IoU across loops
right y-axis or separate chart: accuracy improvement over frozen-route baseline
```

---

## 9. Research sequencing

The current line of work should not immediately displace shorter-horizon priorities.

Suggested sequence:

```text
1. ARC/agent work
   Finish the current agent/encoder thread and get real frame-change signal.

2. CLT/Ouro paper
   Get the existing looped-latent interpretability work public on arXiv.

3. NSA-lite side implementation
   A few weekends only: verify that toy selected-block routing and route logging work.

4. Wait for SubQ report/model card
   Use the eventual technical details to sharpen the architecture hypothesis.

5. Full route-dynamics experiment
   Run the clean IoU-vs-difficulty experiment.

6. CLT + route trajectory follow-up
   Probe alignment between latent relational features and discrete evidence routes.
```

Reasoning:

```text
The CLT paper being public strengthens the authority for the Looped-SSA interpretability framing.
The NSA-lite side track builds useful infrastructure without committing six months immediately.
SubQ's eventual report may reveal which sparse primitive is actually worth mirroring.
```

---

## 10. Frozen claims and open questions

### 10.1 Frozen claims

```text
1. SubQ's public FLOP ratios strongly imply constant fanout, not constant sparsity.

2. The best one-line architecture hypothesis is:
       NSA++ with constant fanout.

3. Vanilla NSA is close but insufficient at 12M unless its compressed stream is bounded,
   adaptive, hierarchical, or aggressively reused.

4. Ouro translates naturally as adaptive depth over a sparse evidence substrate.

5. RLTT translates cleanly to latent states, but not cleanly to discrete routes.

6. Route training is the main technical bottleneck.

7. The best first experiment is route IoU across loops conditional on task difficulty.

8. The strongest research framing is interpretability:
       looped sparse models expose both latent trajectories and evidence-selection trajectories.
```

### 10.2 Open technical questions

```text
SubQ/SSA:
    Does the selector score O(n) candidates per query?
    Are there full dense-attention layers?
    How does compressed budget scale with n?
    Are routes reused across layers?
    Is 12M functional or nominal context?

Looped-SSA:
    Do routes refine under task pressure?
    Is route refinement emergent or supervision-dependent?
    How should K_total be allocated across loops?
    How much route reuse is compatible with quality?
    Can controller RL train budget decisions without high variance?

Interpretability:
    Do route changes precede latent answer changes?
    Can failures be classified into route failures vs integration failures?
    Do latent relational features align with selected evidence paths?
```

---

## 11. A compact abstract for the eventual paper

> **Route Dynamics in Looped Sparse-Attention Models**  
> Sparse attention models reduce long-context compute by selecting a small subset of evidence for each query, while looped language models improve reasoning by iteratively refining latent states before token generation. Combining these ideas suggests an appealing hypothesis: recurrent loops may enable iterative retrieval, where later loops select different evidence after earlier reasoning reveals what matters. We test this hypothesis in controlled multi-hop retrieval environments using a toy looped sparse-attention transformer with observable selected-block routes. We measure route intersection-over-union across recurrent loops as task difficulty varies, comparing one-shot routing, frozen-route recurrence, and bounded top-up rerouting. The central question is whether route changes under retrieval pressure correspond to improved accuracy, or whether looped sparse models primarily reuse their first selected evidence set. This provides a mechanistic test of whether looped sparse attention exposes genuine evidence-refinement trajectories or merely latent refinement over fixed evidence.

---

## 12. Bibliography / source notes

[^subq-ssa]: Subquadratic, “How SSA Makes Long Context Practical,” May 2026. https://subq.ai/how-ssa-makes-long-context-practical

[^subq-intro]: Subquadratic, “Introducing SubQ: The First Fully Subquadratic LLM,” May 2026. https://subq.ai/introducing-subq

[^venturebeat]: VentureBeat, “Miami startup Subquadratic claims 1,000x AI efficiency gain with SubQ model; researchers demand independent proof,” May 2026. https://venturebeat.com/technology/miami-startup-subquadratic-claims-1-000x-ai-efficiency-gain-with-subq-model-researchers-demand-independent-proof

[^ouro]: Zhu et al., “Scaling Latent Reasoning via Looped Language Models,” arXiv:2510.25741. https://arxiv.org/abs/2510.25741

[^rltt]: Jonathan and Tureci, “Prioritize the Process, Not Just the Outcome: Rewarding Latent Thought Trajectories Improves Reasoning in Looped Language Models,” arXiv:2602.10520. https://arxiv.org/abs/2602.10520

[^looprpt]: Tang et al., “LoopRPT: Reinforcement Pre-Training for Looped Language Models,” arXiv:2603.19714. https://arxiv.org/abs/2603.19714

[^nsa]: Yuan et al., “Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention,” arXiv:2502.11089. https://arxiv.org/abs/2502.11089

[^nvidia-nsa]: NVIDIA cuDNN Frontend, “Native Sparse Attention (NSA),” documentation. https://docs.nvidia.com/deeplearning/cudnn/frontend/v1.18.0/fe-oss-apis/nsa.html

[^moba]: Lu et al., “MoBA: Mixture of Block Attention for Long-Context LLMs,” arXiv:2502.13189. https://arxiv.org/abs/2502.13189

[^deepseek]: DeepSeek-AI, “DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models,” arXiv:2512.02556. https://arxiv.org/abs/2512.02556

[^hisa]: Xu et al., “HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention,” arXiv:2603.28458. https://arxiv.org/abs/2603.28458

[^indexcache]: Bai et al., “IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse,” arXiv:2603.12201. https://arxiv.org/abs/2603.12201

[^hashattention]: Desai et al., “HashAttention: Semantic Sparsity for Faster Inference,” arXiv:2412.14468. https://arxiv.org/abs/2412.14468

[^hip]: Lee et al., “HiP Attention: Sparse Sub-Quadratic Attention with Hierarchical Attention Pruning,” arXiv:2406.09827. https://arxiv.org/abs/2406.09827
