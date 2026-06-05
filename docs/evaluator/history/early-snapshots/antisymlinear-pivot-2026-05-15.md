# Post-v10 synthesis — Experiment 2 Redux + Ouro-RLTT-BG architecture (v4, after AntisymLinear pivot)

**Date:** 2026-05-15
**Status:** Locked plan after the math BG-gate pilot, the layer 24/36 geometry blocker resolution, and the AntisymLinear architecture pivot. Supersedes v3.
**Scope:** Everything decided after v10 of the locus memo. Captures L1 ablation, α-sweep, layer 24/36 geometry, math BG-gate pilot, the AntisymLinear architecture decision, and the publication-strategy decision tree.
**Reframing relative to v3:** the project's load-bearing claim shifts from "debiased pairwise training fixes the chosen-first failure mode" to "Ouro-RLTT linearly organizes relational branch-selection signal in its loop/layer states; tiny exact-antisymmetric probes suffice to read it." Whether that claim survives gate scale is the next experimental question.

---

## Current operational priority (2026-05-15)

**Publication-path selection is not currently driving the project.** This changes which experiment runs next compared to what §3 and §6 might suggest in isolation.

> **Next blocking experiment:** source/budget/difficulty-stratified math generated-branch *gate-prep* (50 GSM8K + 50 MATH per budget × 3 budgets), with lib extensions completed first.
>
> **HH Experiment 2 Redux (§3):** deferred architecture triage. Useful later for paper write-up and as an additional diagnostic if gate-scale math/code/reasoning results are ambiguous. Not blocking current BG work.
>
> **Proceeding rule:** do not scale math tap sweeps (AntisymLinear vs NoNorm full matrix) until gate-prep confirms the kept tournament distribution contains enough near-miss-dominant tournaments to actually test branch-selection discrimination. The math pilot's uniform 1.000 result is consistent with an easy-after-filtering kept set; gate-prep must rule that out before scaling.
>
> **Three-tier proceeding criterion (per source):**
> - ≥ 30% near-miss-dominant kept tournaments in at least one budget stratum → scale to gate-scale (300-500/source).
> - 15-30% near-miss-dominant after generation-parameter tuning iteration → accept and scale; report the hard-fraction metrics prominently.
> - < 15% near-miss-dominant after tuning → structural problem; do not scale. Investigate generator/verifier, or pivot to code domain where unit-test partial-pass naturally produces near-miss cases.

**Near-miss vs nonsense classification (operational definitions, math domain):**

*Note: this classification is a diagnostic only. Dataset labels remain exact-answer verifier (correct/incorrect). The near-miss split is solely for difficulty assessment on the kept set.*

- **Near-miss wrong branch:** final answer wrong, but output is parseable and structurally math-like, AND at least one of:
  - numeric answer within tolerance of gold answer (relative error ≤ 0.10 when |gold| > 1; absolute error ≤ 1 when |gold| ≤ 1; never divide by zero);
  - reasoning trace contains at least one nontrivial intermediate number from the gold/reference solution (nontrivial = not 0, 1, -1, or numbers already appearing in the prompt);
  - final answer differs by a small arithmetic slip or simple sign/factor error.
- **Nonsense wrong branch:** unparseable, truncated before final answer, no recognizable math structure, wrong setup, or wildly unrelated final answer.

**Intermediate-value extraction is heuristic, especially for MATH.** GSM8K has `<<lhs=rhs>>` markers that make extraction reliable. MATH does not; the conservative heuristic is: extract integers and decimals from reference solution, extract same from candidate reasoning, near-miss if at least one nontrivial reference number appears in candidate reasoning AND the final answer is parseable but wrong. If intermediate-value extraction returns empty or fails, fall back to numeric tolerance only and flag the tournament metadata with `classifier_fallback = True`.

Tournament-level classification:
- "near_miss_dominant" if at least half of incorrect branches are near-miss;
- "nonsense_dominant" otherwise;
- "trivial" if no incorrect branches (should not occur in kept set; flag if it does).

**Verdict reporting format.** Gate-prep reports must explicitly print, on their own lines near the top of the diagnostic:
```
GATE_PREP_VERDICT_GSM8K = GREEN | YELLOW | RED | YELLOW_INSUFFICIENT_N | RED_INSUFFICIENT_YIELD
GATE_PREP_VERDICT_MATH  = GREEN | YELLOW | RED | YELLOW_INSUFFICIENT_N | RED_INSUFFICIENT_YIELD
GATE_PREP_VERDICT       = GREEN | YELLOW | RED | YELLOW_INSUFFICIENT_N | RED_INSUFFICIENT_YIELD
```
This makes the verdict machine-parseable and the headline output of any gate-prep session.

Minimum denominator rule: a source-budget cell can only produce a GREEN verdict if it has at least 10 kept tournaments. If its near-miss-dominant fraction is high but `kept_tournaments < 10`, mark the cell `YELLOW_INSUFFICIENT_N` rather than GREEN. If a source has fewer than 10 kept tournaments across all budget strata combined, mark the source `RED_INSUFFICIENT_YIELD` regardless of near-miss fraction. Overall verdict is the worst of GSM8K and MATH, ordered GREEN > YELLOW > YELLOW_INSUFFICIENT_N > RED > RED_INSUFFICIENT_YIELD.

The full experiment backlog (current state, completion required for all, ordering flexible within tracks):

| # | Experiment | Status |
|---|---|---|
| 1 | Layer 24/36 bipartite probe | ✅ Done |
| 2 | Per-layer Thinking-vs-RLTT comparison | ✅ Done |
| 3 | Converged-layer old-head readout probe | ✅ Done |
| 4 | Corrected math BG-gate pilot | ✅ Done |
| 5 | **Math gate-prep pass** | **Next** |
| 6 | Math gate-scale tournament dataset | After 5 |
| 7 | AntisymLinear vs NoNorm at math gate scale | After 6 |
| 8 | Experiment 2 Redux (HH layer-47 triage) | Deferred |
| 9 | Code branch tournament dataset + tap evaluation | Track B |
| 10 | Reasoning branch tournament dataset + tap evaluation | Track B |
| 11 | Full-split Ouro-Thinking capture (blocks #8) | When #8 runs |
| 12 | Full-split Ouro-RLTT capture at layers 24/36/47 | After 7/9/10 |
| 13 | BG Phase 1 mixed-domain L_eval training | After 12 |
| 14 | Phase 1 evaluation + four-way disagreement decomposition | After 13 |
| 15 | L_LM mix validation against RLTT base | Phase 2 prep |
| 16 | Cloud-compute quote for Phase 2 | Phase 2 prep |

---

## 0. The reframing

The current framing: *Ouro-RLTT exposes trajectory-distributed relational evidence about candidate quality in its loop/layer states. The BG controller's role is selection over alternatives, not judgment over individuals. The tap heads do not need to compute correctness — they need to linearly read a relational direction the backbone has already organized.*

Three sentences banned from the codebase and the next paper:

1. "`score(x)` means quality." It does not. The object is `score(a, b)`, antisymmetric by construction in the locked architecture. The evaluator is a comparator, not a judge.
2. "The evaluator's preference signal." There is no preference signal in single-candidate hidden states (the 21.75% below-chance independent probe in the original CLT paper established this). What exists is a *relational* signal between candidates.
3. "Loop 2 is special." Refuted by v6 onwards. All four loops carry comparable signal under proper readout; v10 settles that L2-L4 are functionally one state at the boundary on HH text. The math-domain geometry differs (see §4).

The basal-ganglia metaphor is appropriate only when implemented as selection/gating over alternatives, not judgment over individuals. Early tap proposes/prunes, mid tap detects uncertainty/disagreement, late tap selects. Not "average all taps and hope." Not "compute correctness from scratch."

---

## 1. L1 ablation result (2026-05-14)

**Status:** Historical. Measured against the published GRU head, not the AntisymLinear family. May not transfer to the new architecture — included for paper-trail completeness and because it informed the bipartite framing v10 introduced.

**Script:** `utilities/evaluator/probes/probe_l1_ablation.py`, zero-shot, swap-balanced, fp32, no model forward, 8.9 s wall.

| variant | canonical | centered | flip |
|---|---:|---:|---:|
| L4-only (baseline) | 0.950 | 0.595 | 0.175 |
| L1-only | 0.915 | 0.575 | 0.180 |
| L2-only | 0.950 | 0.600 | 0.155 |
| mean(L1, L4) | 0.940 | **0.620** | 0.150 |
| natural[L1-4] (ref) | 0.950 | 0.595 | 0.180 |

**Verdict at the time:** L1 carries readable relational signal → fuse (L1, L4).

**Caveat under v4:** the +2.5 pp gain from mean(L1, L4) was measured against a 5M-param head with chosen-first bias. AntisymLinear at 47_L4 alone might already saturate the linear-readable signal at layer 47, in which case the L1 fusion finding becomes a property of the published head's failure mode rather than a property of Ouro's loop geometry. The Experiment 2 Redux (§3) tests this directly by including 47_L4 and 47_concat_L1_L4 as separate input parameterizations of the AntisymLinear head.

---

## 2. α-sweep result (2026-05-14)

**Status:** Historical. Same caveat as §1 — measured against the published head.

| α | canonical | centered | bias/sig |
|---:|---:|---:|---:|
| 0.00 | 0.950 | 0.595 | 1.416 |
| 0.45 | 0.945 | **0.620** | 1.474 |
| 1.00 | 0.915 | 0.575 | 1.404 |

No α beat the unweighted mean's 0.620. The plateau (α ∈ [0.35, 0.50]) suggested the gain came from "separated slots" rather than weighted mixing. Under the AntisymLinear regime this argument simplifies: input parameterization is `concat(L1, L4)` (4096-dim) vs `L4` alone (2048-dim), and the head is the same exact-antisymmetric linear comparator. Whether concat wins over L4-only is now a parameter-of-the-head question, not a learned-weight question.

---

## 3. Experiment 2 Redux — architecture triage on HH at layer 47

**Status:** Pre-registered, replaces v3's 12-run λ_sym sweep.

**Framing:** the v3 Experiment 2 spec built around λ_sym and a 6M-param debiased head is obsolete. The math BG-gate pilot demonstrated that a 4k-param exact-antisymmetric linear comparator (AntisymLinear) reaches 1.000 top-1 on held-out math tournaments at converged layers. The decisive question for the project is whether AntisymLinear works on HH-RLHF too.

**Decision driven by this experiment:**
- Whether the v4 evaluator paper uses AntisymLinear as the headline or the published debiased head.
- Whether the project ships as one paper, two papers, or one paper with a contrast framing (see §6 publication-strategy decision tree).
- Whether the BG taps use AntisymLinear at all three layers (default if the experiment succeeds) or only at 24/36 (if AntisymLinear fails on HH but works at converged layers).

### Backbone

Ouro-2.6B-Thinking, frozen, loaded from `ByteDance/Ouro-2.6B-Thinking`. Forward in bf16 (12 GB VRAM), states cast to fp32 on capture. Matches v10 numerical contract.

### Captured states

L1 and L4 at layer 47 only, per token, fp32, over the full HH-RLHF split (50k train + 8552 test). Capture pass is the same overnight ~9h job from v3; whatever states are already on disk get reused.

### Head architectures under test

**A. AntisymLinear (the new default):**
```python
class AntisymLinearHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.linear = nn.Linear(dim, 1, bias=False)
    def forward(self, left, right):
        return self.linear(self.norm(left - right)).squeeze(-1)
```

Antisymmetric by construction: `score(b, a) = −score(a, b)` exactly (modulo fp32 precision). No symmetric-offset failure mode possible. Approximately 2k params at d=2048, 4k at d=4096.

**B. AntisymLinearNoNorm (transitivity-ablation baseline):**
```python
class AntisymLinearNoNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 1, bias=False)
    def forward(self, left, right):
        return self.linear(left - right).squeeze(-1)
```

Factors as `w · (a − b) = u(a) − u(b)` for scalar utility `u(x) = w · x`. Transitive by construction — cycles `A > B > C > A` are impossible. Same parameter count as AntisymLinear.

**C. Published-style debiased head (the v3 default, now a comparison control):**

5M-param head from the CLT paper architecture, retrained with swap augmentation and symmetric-offset penalty λ_sym = 0.3. Optimizer config and training schedule per v3 §3. This is the v3 plan as it would have run, included here as the "this is what we would have shipped" comparison.

**D. AntisymMLP (escalation-only, run only if A is weak):**
```python
class AntisymMLPHead(nn.Module):
    def __init__(self, dim, hidden=256):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, 1, bias=False)
    def forward(self, left, right):
        h = self.norm(left - right)
        h = torch.tanh(self.up(h))  # tanh is odd; preserves antisymmetry through depth
        return self.down(h).squeeze(-1)
```

~500k params at d=2048, hidden=256. Antisymmetry preserved through depth because `tanh` is odd: `tanh(−x) = −tanh(x)`. Other activations (ReLU, GELU, SiLU, or `x − silu(−x)` which I previously misclaimed was odd) silently break antisymmetry and must not be used.

### Input parameterizations at layer 47

For each architecture, test:

1. **L4 only** (2048-dim) — minimal baseline.
2. **concat(L1, L4)** (4096-dim) — endpoint contrast.
3. **concat(L4, L4 − L1)** (4096-dim) — refinement-direction parameterization.

For AntisymLinear and AntisymLinearNoNorm specifically: this is a 2 architectures × 3 parameterizations × 1 hyperparameter sweep = **6 runs**. Plus the published debiased head as a single run with its v3 config. **7 runs total**, all small (AntisymLinear trains in minutes; published head in hours).

The v3 weighted-mean arm is dropped — the α-sweep showed no α beats mean, and "is weighted-mean useful?" is no longer the question.

### Loss

For A, B, D: `BCEWithLogitsLoss(head(chosen, rejected), 1)`. No swap augmentation (antisymmetry is architectural). No λ_sym (no symmetric offset possible).

For C (published head): swap augmentation + λ_sym = 0.3, per v3.

**Sanity check (all heads):** log `head(a, b) + head(b, a)` per batch. Should be machine-zero for A, B, D. Nonzero for C indicates the symmetric-offset penalty hasn't fully driven bias to zero.

### Metrics

- `canonical_acc = P(score(c, r) > 0)`
- `centered_acc = P(score(c, r) − score(r, c) > 0)` — primary checkpointing metric for C; identical to canonical_acc for A, B, D by construction
- `bias_to_signal = |mean(s_cr + s_rc)| / std(s_cr − s_rc)` — should be ≈ 0 for A, B, D; reported as ~3.42 for the published head pre-debiasing, target < 0.5 for C
- `cycle_rate` on triplets sampled from the test set — should be 0 for B (transitive by construction), nonzero permitted for A and D, nonzero expected for C
- `flip_strict_sign_reversal` — degeneracy diagnostic
- Parameter count — reported per head as a first-class metric

Checkpoint by best centered_acc on held-out HH test.

### Pre-registered outcome interpretations

The four outcomes that route to four different paper strategies:

| Outcome on HH | Interpretation | Paper path (see §6) |
|---|---|---|
| AntisymLinear strictly dominates published-debiased | Capacity-as-bias-correction obviated by architecture | Single paper, AntisymLinear as headline |
| AntisymLinear competitive (within ~10pp) | Capacity-vs-structure contrast | Single paper with contrast framing |
| AntisymLinear substantially worse (>10pp gap) | Linear comparator insufficient for HH noise | Two papers split |
| AntisymLinear fails on generated branches too | v3 plan as written | v3 publication strategy |

### Pre-registered outcome probabilities

- AntisymLinear strictly dominates HH: ~15%
- AntisymLinear competitive but not dominant: ~50%
- AntisymLinear substantially worse on HH: ~25%
- AntisymLinear fails on both HH and generated branches: ~10%

The "competitive but not dominant" case is most likely because: HH-RLHF has substantial label noise (the v6 error analysis showed the published head correctly disagrees with the human annotator on safety-aligned answers in many examples), and a 4k-param linear probe is genuinely capacity-limited compared to a 5M-param GRU head, so some gap is expected. The "strictly dominates" case is possible only if the entire capacity of the published head was doing bias-correction work — which is plausible given the v6/v9 findings but not guaranteed.

### What's explicitly *not* in Experiment 2 Redux

- No λ_sym sweep (architectural antisymmetry obviates).
- No swap augmentation for A/B/D (architectural antisymmetry obviates).
- No per-loop GRU (v10 + bipartite finding collapse it).
- No iterated norm K-sweep (v10 refuted gamma-attractor).
- No basal-ganglia controller (separate, §4).

---

## 4. Ouro-RLTT-BG — continued-training architecture

**Status:** Heterogeneous tap interface locked. AntisymLinear is the Phase-1 default head family across all taps. Experiment 2 Redux (§3) may inform HH/evaluator-track comparisons later but does not block current BG gate-prep. Phase 1 captures and training spec are locked at the family level.

### Base backbone

**Ouro-RLTT** from `/home/moloch/ouro_project/models/ouro_rltt_local`. 48 shared layers, R = 4 loops, 2048 hidden dim.

**Freezing strategy:** LoRA rank 16 on attention and MLP projections at all layers, **full unfreeze at layers 24, 36, 47**.

LoRA preserves RLTT-trained competence at intermediate layers; tap-layer unfreeze lets them develop readable relational signal under auxiliary pressure.

### Layer-24/36/47 geometry (resolved 2026-05-15)

| Layer | L1-L4 cos | L2-L4 cos | mean off-diag | Verdict |
|---:|---:|---:|---:|---|
| 24 | +0.9284 | +0.9790 | +0.9632 | fully converged |
| 36 | +0.9347 | +0.9822 | +0.9682 | fully converged |
| 47 | +0.7350 | +0.9608 | +0.8849 | bipartite |

Conclusion: 24/36 use single-state 2048-dim heads; 47 uses 4096-dim fused heads.

### Math-vs-HH layer-47 geometry divergence

On math text, the layer-47 trajectory is *more* spread than on HH:

| Layer 47 | HH | Math |
|---|---:|---:|
| L1-L4 cos | 0.735 | **0.656** |
| L2-L4 cos | 0.961 | **0.895** |

L1↔L4 is *further apart* on math (0.656 < 0.735), and L2↔L4 is also further apart (0.895 < 0.961). The whole trajectory is more distributed on math, not less.

Mechanistic reading: RLTT trained the model to actually use the loop trajectory for math computation. The v10 finding that Thinking and RLTT are identical at HH boundary states (cos 0.9992) does not preclude a math-domain difference; it just measured HH. This is the closest the project has come to a mechanistic signature for RLTT's math gains.

**Implication for BG tap design:** on math, the late-tap fusion of L1/L4 alone may be insufficient. Math gate-scale needs to also test `47_mean` (mean of all four loops) and `47_concat_all_loops` (concat of L1, L2, L3, L4, 8192-dim input). At AntisymLinear's parameter scale, this is 8192 weights — trivial.

### Evaluator taps (heterogeneous, locked)

| Tap | Input interface | Default head | Role |
|---:|---|---|---|
| 24 | `tap24(h_24_loop_k)` single-state 2048-dim | AntisymLinear (locked) | early prune / viability signal |
| 36 | `tap36(h_36_loop_k)` single-state 2048-dim | AntisymLinear (locked) | mid-course uncertainty / consistency signal |
| 47 (HH) | `tap47(h_47_L1, h_47_L4)` fused 4096-dim | AntisymLinear (locked) | final selector |
| 47 (math) | also `tap47(mean(L1..L4))` 2048-dim, `tap47(concat(L1..L4))` 8192-dim | AntisymLinear | final selector with trajectory-spread option |

For all three taps, the parallel `AntisymLinearNoNorm` baseline runs alongside `AntisymLinear` to measure transitivity (cycles permitted under AntisymLinear, impossible under NoNorm).

If Experiment 2 Redux shows AntisymLinear failing on HH but succeeding on generated branches, the heterogeneous-architecture option is: AntisymLinear at 24/36 (where it provably works on math), published-debiased head at 47 (where it provably works on HH). That's only locked if §3 forces it; the default is AntisymLinear everywhere.

### Pooling

**Default:** masked mean pool over valid tokens. Applied per-tap before the head.

**Escalation:** learned attention pooling, only if mean pooling underperforms. Mean pool is the right starting point because (a) it doesn't introduce additional learned parameters that could overfit, (b) it doesn't introduce additional failure modes, (c) the math pilot's pooling (which worked) was masked mean.

### Controller logic (read-only baked-in)

Three taps fire on every forward pass; scores are auxiliary outputs alongside logits.

```python
early_score = tap24(h_24_loop_k)               # 2048-dim single-state
mid_score   = tap36(h_36_loop_k)               # 2048-dim single-state
late_score  = tap47(h_47_L1, h_47_L4)          # 4096-dim fused
# math-only optional: late_alt = tap47(concat(L1..L4))  # 8192-dim trajectory-spread

control_signal = {
    "early": early_score,
    "mid":   mid_score,
    "late":  late_score,
    "early_mid_disagreement": abs(early_score - mid_score),
    "mid_late_disagreement":  abs(mid_score - late_score),
    "confidence": abs(late_score),
}
```

**Read-only.** Active steering (residual-stream modification) is Phase 3+, not in scope.

### Training objective

```
L_total = L_LM + λ_eval · L_eval + λ_align · L_align    # λ_align = 0 in Phase 1
```

**L_LM** — standard next-token prediction. Mix proposal: FineWeb-Edu (1×), OpenWebMath (2×), NuminaMath-CoT (3×), StarCoder (1.5×). Validate against RLTT base before Phase 2.

**L_eval — mixed-domain relational loss:**

```
L_eval = w_HH   · L_eval_HH
       + w_math · L_eval_math
       + w_code · L_eval_code
       + w_reas · L_eval_reasoning
```

Per channel, per tap:
```
L_eval_channel_l = BCEWithLogits(head_l(correct, incorrect), 1)
```

No λ_sym needed — antisymmetry is architectural.
Log `head_l(a, b) + head_l(b, a)` per batch as numerical sanity (should be machine-zero).

**Channel data sources:**
- **HH:** HH-RLHF preference pairs.
- **Math:** Ouro-RLTT-generated attempts on GSM8K + verifier-clean MATH problems, paired correct-vs-incorrect, labels from exact-answer verifier.
- **Code:** Ouro-RLTT-generated patches, paired pass-vs-fail, labels from unit-test execution.
- **Reasoning:** Ouro-RLTT-generated attempts on selected reasoning benchmarks, labels from answer keys.

**Critical:** math/code/reasoning pairs must be from Ouro-RLTT's own generated attempts, not static datasets. The deployment distribution is generated branches.

**Channel weighting:** swept as hyperparameter. Defaults: equal per-channel batches; inverse-sqrt channel-size weighting; equal per-token.

**L_align — Phase 1: disabled (λ_align = 0).**

Phase 1 trains independent taps under L_eval only. Tap disagreement is measured empirically. The four-way decomposition decides whether L_align enters Phase 2:

| Disagreement predicts | Implication | Phase 2 action |
|---|---|---|
| Branch ambiguity | Disagreement is metacognitive signal | No L_align; use disagreement as branch policy input |
| Late-tap error | Disagreement is corrective | No L_align; use disagreement as confidence correction |
| Need for more rollout | Disagreement is compute-allocation signal | No L_align; use disagreement as compute scheduler input |
| None (noise) | Disagreement is regularizable | Add L_align with `|s_cr_late|` weighting |

### Implication of AntisymLinear holding for Phase 2

**This is a substantive Phase 2 reframe worth being explicit about.**

If AntisymLinear holds at gate scale (the BG taps are linear probes over already-organized backbone signal), then Phase 2's job is not "train the taps to work" but "ensure the backbone *continues* to expose linearly-readable relational signal at layers 24/36/47 as it updates under L_LM pressure."

L_eval becomes a *regularizer on the backbone*, not a training signal for the taps. The taps can stay frozen (or updated slowly with much lower lr) during Phase 2; what changes is the constraint on the backbone's hidden state geometry. The basal-ganglia metaphor sharpens further: the controller reads a signal that already exists; Phase 2 ensures the signal doesn't get washed away by other training pressure.

This changes the per-tap centered_acc + LM perplexity joint checkpointing criterion from "do the taps maintain their accuracy" to "does the backbone preserve linearly-readable structure under L_LM updates."

If AntisymLinear fails at gate scale, the v3 Phase 2 framing applies — the taps are full evaluators that need joint training with the backbone.

### Phase split

**Phase 1 — Local, frozen RLTT backbone, AntisymLinear tap training.**
- Capture heterogeneous tap states over HH train + math/code/reasoning generated-branch pairs at layers 24, 36, 47.
- Train three independent heterogeneous tap heads with mixed-domain L_eval, no L_align, no LM loss.
- Plus AntisymLinearNoNorm baseline at every config.
- ~12h overnight per channel-weighting config.
- Outputs: tap checkpoints, per-tap centered_acc per channel, agreement profile, cycle-rate per config.

**Phase 2 — Cloud, full objective, joint training.**
- A100/H100 rental. Init from Phase 1 taps + Ouro-RLTT backbone.
- LoRA rank 16 everywhere, full unfreeze on 24/36/47.
- Full L_total objective. λ_align decided by Phase 1 disagreement analysis.
- 20-50B tokens continued pretraining on pretraining mix interleaved with preference pairs.
- Checkpoint by generated-branch tournament accuracy + per-tap centered_acc + LM perplexity.

**Phase 3 — Controller policy.** Design branching / rollout / decoding policy that uses `control_signal`.

---

## 5. Generated-branch tournament dataset and Phase-2 gate

This is the single most important addition from v2-v3. The Phase-2 go/no-go is generated-branch tournament performance, not HH centered_acc.

### Dataset construction protocol

**Generator:** Ouro-RLTT (not Ouro-Thinking).

**Generation parameters (fixed for reproducibility):**
- Temperature: 0.7
- Nucleus (top-p): 0.95
- Attempts per prompt: 2-4
- Budget strata: max_new_tokens ∈ {160, 256, 512, 1024} — multiple budgets per source
- Keep prompt only if at least one correct *and* one incorrect attempt produced

**Domains:**
- **Math:** GSM8K + verifier-clean MATH problems. Labels from exact-answer verifier.
- **Code:** HumanEval, MBPP. Labels from unit-test execution.
- **Reasoning:** Selected MCQ benchmarks. Labels from answer keys.
- **Optional ARC/games:** environment score (later phase).

**Critical: labels must be external/objective.** Verifier or execution only. Not evaluator-derived. Not LLM-judge-derived.

### Generation-process metadata (mandatory)

For every kept tournament, log:

- Source (GSM8K / MATH / HumanEval / MBPP / specific reasoning benchmark)
- Budget (max_new_tokens used)
- Attempts generated, attempts correct, attempts incorrect, attempts unparseable
- Per-attempt: output token length, truncation flag
- Tournament formation budget: max_new_tokens value at which this prompt first yielded a mixed correct/incorrect set

**Difficulty proxies per kept tournament:**

- Attempts-to-first-correct (1 = correct on first try; higher = harder)
- Correct fraction among attempts (0 < x < 1; closer to 0 or 1 = easier filtering)
- Near-miss vs nonsense classification for incorrect branches (near-miss = wrong answer differs from correct by small margin in solution space; nonsense = wildly wrong, fails all tests, or doesn't parse)

The near-miss vs nonsense split is the key difficulty proxy. Tournaments where all incorrect branches are nonsense are trivial (any comparator beats nonsense vs correct). Tournaments where incorrect branches are near-misses require the comparator to actually discriminate. The gate-scale evaluation must report top-1 separately on near-miss-dominant and nonsense-dominant splits.

### Difficulty-balanced sampling at gate scale

The dataset should be *sampled* to be difficulty-balanced after filtering, not just *reported* with difficulty metadata. The target is a kept-tournament set where near-miss-dominant and nonsense-dominant tournaments are present in similar proportion. If near-miss-dominant tournaments are scarce, the gate-scale dataset construction iterates: generate more candidate prompts, keep those that produce near-miss-dominant tournaments preferentially, until the difficulty distribution is balanced.

### Pilot vs gate-set sizes

- **Pilot (protocol shakedown):** 100-500 examples total, single budget. Used to validate protocol, generation parameters, label pipeline.
- **Phase-2 gate set:** ≥ 1000 tournaments total AND ≥ 300 per domain, across at least three budget strata, with difficulty-balanced sampling.

### Evaluation metric panel (tournament-level)

Reported per controller variant per domain per difficulty stratum:

1. Top-1 tournament accuracy (controller picks correct branch in tournament of K candidates)
2. Pairwise accuracy
3. Condorcet winner rate
4. **Cycle rate** (structurally required given AntisymLinear permits cycles)
5. Margin calibration (does |score| predict correctness)
6. Per-source / per-budget / per-difficulty-split breakdown

### Phase-2 go/no-go gate (disjunctive)

**Proceed to Phase 2 if at least one of the following holds:**

**A. Generated-branch tournament gain.**
- Multi-tap or trained-tap controller improves over layer-47 single-tap baseline by ≥ 2-3 pp on top-1 tournament accuracy.
- Paired bootstrap CI on within-prompt difference excludes zero (95% CI).
- Required n: ≥ 1000 paired tournaments.

**B. Disagreement informativeness.**
- Early/mid/late tap disagreement predicts at least one of: branch ambiguity, late-tap error, need for additional rollout, improved tournament policy.
- AUC point estimate ≥ 0.65 AND lower 95% CI bound ≥ 0.60.

**C. Mixed-domain transfer.**
- Trained taps improve at least two target domains on top-1 tournament accuracy.
- Per-domain bootstrap CI excludes zero in each of the two domains.
- No material degradation on the strongest pre-training domain.
- No material LM-sanity-check degradation.

**Explicit non-criterion:** Do not proceed to Phase 2 solely because HH centered_acc is high. HH centered_acc is a diagnostic for comparator health, not a Phase-2 gate.

### Statistical hygiene

- Gates A and C use paired bootstrap (same prompts through baseline and treatment).
- Gate B uses AUC with bootstrap CI on the AUC estimate.
- All CIs at 95%, all metrics with bootstrap intervals.
- Pilot data only (n < 1000): directional evidence, not a gate.

### Corrected math pilot result (2026-05-15)

Pilot complete, recorded in `docs/evaluator/math-and-gsm8k-status.md`. 33 kept tournaments (26 GSM8K, 7 MATH), 25 train / 8 eval. All AntisymLinear configs at 24/36/47 hit 1.000 top-1 with zero cycles on the n=8 eval split. Treated as protocol/proof-of-possibility, not a gate decision. The uniform 1.000 result probably reflects easy-after-filtering tournaments rather than genuine signal saturation — the gate-scale follow-up must include difficulty-balanced sampling to test discrimination on hard tournaments.

---

## 6. Publication-strategy decision tree

The Experiment 2 Redux outcome routes to one of four publication paths. Pre-registered now to control post-hoc narrative drift.

### Path 1 — Single paper, AntisymLinear as headline (if AntisymLinear strictly dominates HH)

Story: *"Ouro-RLTT linearly organizes relational branch-selection signal in its loop/layer states. Tiny exact-antisymmetric probes — 4096 parameters at the layer-47 fused readout — suffice to read it. The 5M-parameter published head's apparent strength was substantially capacity-as-bias-correction; an architectural choice obviates the entire debiasing apparatus."*

Single paper, AntisymLinear is the primary artifact, published-debiased head appears as a contrast. The v3 evaluator-paper track is dropped into a single subsection of this paper.

Probability: ~15%.

### Path 2 — Single paper with capacity-vs-structure contrast framing (if AntisymLinear competitive within ~10pp)

Story: *"A 4096-parameter exact-antisymmetric linear comparator reaches X% centered accuracy on HH-RLHF. A 5M-parameter GRU-based head with swap-augmented debiased training reaches X+δ%. The gap is small relative to the parameter ratio. Most of the published head's capacity is performing bias-correction work that an architectural choice could have prevented. Both architectures are useful: AntisymLinear at deployment scale, published-debiased where the last few accuracy points matter."*

Single paper, both architectures included, the framing is the contribution. The v3 evaluator-paper track and the BG-controller track fold into one paper with two architectural exhibits.

Probability: ~50%. Most likely path.

### Path 3 — Two papers split (if AntisymLinear substantially worse on HH, >10pp gap)

Paper A — HH evaluator paper, debiased published head as the headline. The v3 Experiment 2 spec essentially as written, just with AntisymLinear as a "we tried this and it was much weaker on noisy preference data" comparison.

Paper B — BG branch-selection paper. AntisymLinear at gate-scale generated branches, the read-only basal-ganglia controller, the multi-tap disagreement analysis. The v3 BG-controller track as its own paper.

Probability: ~25%.

### Path 4 — v3 plan as currently written (if AntisymLinear fails on both HH and generated branches)

Single paper, debiased published head as the headline. AntisymLinear becomes a small ablation in the appendix showing what doesn't work. The v3 spec applies verbatim.

Probability: ~10%.

### What this commits the project to

The Experiment 2 Redux is the load-bearing experiment *for the publication strategy decision*, when publication strategy becomes a driving priority. It is **not** currently the load-bearing next experiment — math gate-prep is (see operational priority box at top). The Experiment 2 Redux runs cheaply (~6 small AntisymLinear runs + 1 published-debiased run, total maybe 4-6 hours of compute), and pre-registering the decision tree now means the eventual outcome interprets cleanly rather than being read post-hoc into whatever narrative is convenient — but the actual run is deferred to whenever publication-path selection becomes a priority.

---

## 7. Sequencing (revised)

1. **Layer-24/36 bipartite probe.** Complete. Result: 24/36 converged, 47 bipartite.
2. **Per-layer Thinking-vs-RLTT comparison.** Complete. Result: v9 layer choices transfer.
3. **Converged-layer old-head readout probe.** Complete. Result: 24/36 old-head centered_acc is near chance; new heads must be trained, can't transfer the published checkpoint.
4. **Math BG-gate pilot.** Complete. Result: AntisymLinear at 24/36 single-state matches 47 fused on tiny held-out math eval. Proof-of-possibility for the AntisymLinear architecture.
5. **Math gate-prep pass.** Source/budget/difficulty-stratified math generated-branch generation (50 GSM8K + 50 MATH per budget × 3 budgets), with lib extensions completed first. Verifies kept-set is hard enough to discriminate taps before scaling. ~2-4 hours compute. **Next.**
6. **Build generated-branch tournament dataset at gate scale.** Source-stratified, budget-stratified, difficulty-balanced sampling. 300-500 prompts per source after gate-prep verdict is GREEN or acceptable YELLOW. Long-lead; sequenced after step 5.
7. **AntisymLinear vs NoNorm tap matrix at math gate scale.** Full sweep across 24/36 single-state configs (L1, L4, mean) and 47 fused configs (L4, mean, concat_L1_L4, concat_all_loops), with AntisymLinear and AntisymLinearNoNorm in parallel. Per-source/per-budget/per-difficulty-split metrics.
8. **Code branch tournament dataset + tap evaluation.** Clone the math pattern for HumanEval/MBPP with unit-test labels. Track B parallel to math gate-scale.
9. **Reasoning branch tournament dataset + tap evaluation.** Selected MCQ benchmarks with answer keys. Track B parallel.
10. **Experiment 2 Redux — HH architecture triage.** Deferred. Pre-registered in §3 and §6. Runs when publication-path selection becomes a priority, or as an additional diagnostic if gate-scale results are ambiguous. ~4-6 hours compute when run. Depends on full-split Ouro-Thinking capture (~9h overnight) as a prerequisite.
11. **Full-split Ouro-RLTT capture at heterogeneous taps 24/36/47.** ~12h overnight. After tap input choices are settled from steps 7-9.
12. **Ouro-RLTT-BG Phase 1.** AntisymLinear heterogeneous taps trained on captured states with mixed-domain L_eval. Plus AntisymLinearNoNorm baseline. Overnight per channel-weighting config.
13. **Phase 1 evaluation on HH + generated branches.** Includes four-way disagreement decomposition deciding whether L_align enters Phase 2.
14. **Phase-2 go/no-go gate.** Apply disjunctive criteria (§5).
15. **Paper write-up.** Per the publication-strategy decision tree (§6).
16. **Phase 2 cloud run.** Gated on step 14.
17. **Phase 3 controller policy design.** Gated on Phase 2.

Step 5 (math gate-prep) is the load-bearing next decision and runs in hours. Step 6 (gate-scale dataset) is the longest-lead item once gate-prep clears. Step 10 (Experiment 2 Redux) is deferred backlog, not blocking BG work.

---

## 8. Open items / TODOs

- [x] Run layer-24/36 bipartite probe (step 1).
- [x] Run per-layer Thinking-vs-RLTT probe (step 2).
- [x] Run converged-layer old-head readout probe for layers 24/36 (step 3).
- [x] Run math BG-gate pilot (step 4, corrected).
- [ ] **Run math gate-prep pass (step 5). Load-bearing next experiment.** Lib extensions (AntisymLinearNoNorm, new 47 configs, cycle rate, margin calibration, near-miss classifier) completed first.
- [ ] Build gate-scale generated-branch tournament dataset (step 6) — source-stratified, budget-stratified, difficulty-balanced. Sequenced after gate-prep verdict.
- [ ] Run AntisymLinear vs NoNorm tap matrix at math gate scale (step 7).
- [ ] Code branch tournament dataset + tap evaluation (step 8). Track B.
- [ ] Reasoning branch tournament dataset + tap evaluation (step 9). Track B.
- [ ] *Deferred:* Experiment 2 Redux — HH architecture triage (step 10). Runs when publication-path selection becomes a priority, or as additional diagnostic if gate-scale results are ambiguous. Requires full-split Ouro-Thinking capture as prerequisite.
- [ ] Identify specific reasoning benchmarks for the reasoning channel (BBH subsets candidate).
- [ ] Validate RLTT-base pretraining mix against FineWeb-Edu + OpenWebMath + NuminaMath-CoT + StarCoder recipe.
- [ ] Get cloud-compute quote for Phase 2 (20-50B tokens, 2.6B backbone + LoRA + 3 unfrozen layers).
- [ ] Full-split RLTT capture at heterogeneous taps 24/36/47 (step 11).
- [ ] Phase 1 heterogeneous AntisymLinear tap retrain with mixed-domain L_eval (step 12).
- [ ] Phase 1 evaluation including four-way disagreement decomposition (step 13).
- [ ] Apply Phase-2 disjunctive gate (step 14).

---

## 9. Summary

**The strongest current empirical claim:** Ouro-RLTT linearly organizes relational branch-selection signal in its loop/layer states. At layers 24 and 36, the loops converge to a single effective state; at layer 47 on HH text, the trajectory is bipartite (L1 vs L2-L4 cluster); on math text, the layer-47 trajectory is more uniformly spread. Tiny exact-antisymmetric linear comparators (AntisymLinearHead, ~2-8k parameters per tap) can read this signal on Ouro-RLTT-generated math branches in the corrected pilot. Whether the same architecture suffices on HH-RLHF preference data and gate-scale generated branches is the next experimental question.

**The current architecture target:** a relational, pairwise, heterogeneous multi-tap branch-selection controller. AntisymLinearHead as the default head family across all taps. Heterogeneous tap inputs: single-state 2048-dim at converged layers 24/36, fused 4096-dim at bipartite layer 47, optional 8192-dim trajectory-spread variant for math. AntisymLinearNoNorm as the parallel transitivity-ablation baseline at every config. No λ_sym, no swap augmentation — antisymmetry is architectural. Mixed-domain L_eval over HH + math + code + reasoning channels with externally-labeled generated branches.

**The publication strategy:** four paths pre-registered against the Experiment 2 Redux outcome on HH. Most likely (~50%) is a single paper with a capacity-vs-structure contrast framing — both AntisymLinear and the debiased published head as architectural exhibits, the relative parameter efficiency as the contribution.

**The biggest risks:**
- *Scientific:* AntisymLinear works on tiny math pilot but fails on noisy HH preference data or hard generated branches. Mitigation: Experiment 2 Redux runs immediately; gate-scale dataset uses difficulty-balanced sampling.
- *Engineering:* Phase 2 cloud cost; preserving LM competence while forcing tap readability. Mitigation: LoRA + selective unfreeze; if AntisymLinear holds, L_eval becomes backbone regularizer (much lighter constraint) rather than tap training signal.
- *Conceptual:* Treating disagreement as noise rather than control signal. Mitigation: Phase 1 explicitly tests the four-way decomposition before L_align is added.

## Expanded clean GSM8K transfer + GRU control (2026-05-16)

- EXPANDED_CLEAN_GSM8K_VERDICT: `CLEAN_MINIMUM`
- EXPANDED_LINEAR_TRANSFER_VERDICT: `GOOD`
- GRU_CONTROL_VERDICT: `GRU_WEAK`
- clean tournaments: `28`
- random_top1_baseline: `0.563`
- best AntisymLinear config/head: `{'config': '36_L4', 'architecture': 'AntisymLinear', 'top1': 0.7857142857142857, 'pairwise': 0.7037037037037037, 'cycle': 0.0, 'margin_mean': 1.0424813000219209, 'margin_std': 0.8679666340925131}`
- best NoNorm config/head: `{'config': '24_L4', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.75, 'pairwise': 0.7407407407407407, 'cycle': 0.0, 'margin_mean': 0.012588573902446245, 'margin_std': 0.011785265045384234}`
- best GRU config: `{'config': 'gru24_sequence', 'top1': 0.7142857142857143, 'pairwise': 0.6666666666666666, 'cycle': 0.0, 'bias_to_signal': 0.010894300608310918, 'hh_heldout_acc': 0.5}`
- winner family: `AntisymLinear`
- winner layer: `36`
- full report: `artifacts/reports/probes/clean_gsm8k_expanded_transfer_gru_2026-05-16_summary.md`
- interpretation: The GRU control underperformed the exact-antisymmetric linear/NoNorm controls; centered raw bias was low, but HH holdout accuracy stayed below the control threshold.

## Code branch pilot (2026-05-16)

- CODE_INTERFACE_VERDICT: `READY`
- CODE_TASKSET_VERDICT: `READY`
- CODE_GENERATION_VERDICT: `READY`
- CODE_TOURNAMENT_VERDICT: `TOO_FEW_TOURNAMENTS`
- CODE_TRANSFER_VERDICT: `NOT_RUN`
- tasks: `10`
- candidates: `40`
- strict_clean tournaments: `1`
- diagnostic_mixed tournaments: `4`
- random_top1_baseline: `nan`
- best AntisymLinear row: `NOT_RUN`
- best NoNorm row: `NOT_RUN`
- winner: `NOT_RUN`
- full report: `artifacts/reports/probes/code_branch_pilot_2026-05-16_summary.md`
- interpretation: The local-agent wrapper generated mostly correct code on the selected local/MBPP task mix, leaving too few objective mixed unit-test tournaments for a relational tap-transfer evaluation.

## Code branch pilot v2 (2026-05-16)

- CODE_V2_INTERFACE_VERDICT: `READY`
- CODE_V2_TASKSET_VERDICT: `READY`
- CODE_V2_GENERATION_VERDICT: `READY`
- CODE_V2_TOURNAMENT_VERDICT: `DIAGNOSTIC_ONLY`
- CODE_V2_TRANSFER_VERDICT: `GOOD`
- tasks: `40`
- candidates: `188`
- duplicate rate: `0.3088235294117647`
- correct / near_miss / nonsense: `{'correct': 70, 'near_miss': 13, 'nonsense': 105}`
- strict_clean tournaments: `3`
- diagnostic_mixed tournaments: `22`
- random_top1_baseline: `0.5522727343169126`
- best AntisymLinear row: `{'config': '36_L4', 'architecture': 'AntisymLinear', 'top1': 0.6818181818181818, 'pairwise': 0.7090909090909091, 'cycle': 0.0, 'margin_mean': 1.7220367030663923, 'margin_std': 1.6406907362264995}`
- best NoNorm row: `{'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.7272727272727273, 'pairwise': 0.6909090909090909, 'cycle': 0.0, 'margin_mean': 0.27857657115567813, 'margin_std': 0.35337859692663737}`
- winner: `{'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.7272727272727273, 'pairwise': 0.6909090909090909, 'cycle': 0.0, 'margin_mean': 0.27857657115567813, 'margin_std': 0.35337859692663737}`
- candidate-stage harvesting fixed v1 too-successful-wrapper problem: `yes`
- full report: `artifacts/reports/probes/code_branch_pilot_v2_2026-05-16_summary.md`
- interpretation: The v2 code pilot used objective unit-test labels and candidate-stage harvesting; transfer remains a preliminary code-branch signal.

## Code branch pilot v2 correction + harness hardening (2026-05-16)

- FIX_VERDICT: `PATCHED_AND_UNIT_TESTED`
- full fix report: `artifacts/reports/probes/code_branch_v2_harness_agent_fixes_2026-05-16.md`
- nonsense inspection: `artifacts/reports/probes/code_branch_v2_nonsense_inspection_2026-05-16.md`
- status of previous v2 transfer: `HISTORICAL_PRE_FIX`
- reason: the old `nonsense` bucket was contaminated by runnable zero-pass wrong code, wrapper/prose extraction failures, syntax/runtime failures, and safety rejections.
- original 105 old-nonsense breakdown: `50 runnable_zero_pass_wrong`, `25 parseable_runtime_error`, `20 prose_or_wrapper_not_code`, `6 syntax_invalid_code`, `3 safety_rejected`, `1 parseable_no_function`.
- local-agent fixes: final code extraction rejects wrapper/prose status payloads; `sanitize_tool_input("python", ...)` returns empty for obvious non-code payloads.
- taskset fixes: MBPP function-name extraction skips outer builtins, so `mbpp/232` targets `larg_nnum`; MBPP prompts include exact inferred signature shape; `mbpp/237` states tuple canonicalization explicitly.
- evaluator fixes: binary correct/incorrect labels are preserved for branch selection, while diagnostic subtype is split into `near_miss`, `wrong_code`, `runtime_error`, and `malformed`; legacy `nonsense` is retained only as compatibility metadata.
- safety fix: `sys.setrecursionlimit` is allowed for local DSA while unsafe `sys` usage remains rejected.
- checkpointing fix: generation writes `code_branch_candidates_v2_2026-05-16.partial.json` after each task; evaluation writes `code_branch_tournaments_v2_2026-05-16.partial.json` after each task with provisional evaluations, tournaments, summary, primary eval set, and verdict.
- validation: patched files pass `py_compile`; local-agent wrapper tests pass (`139 passed`); no-model relabel of existing candidates completed.
- no-model relabel of old candidates under patched evaluator: `38` tasks evaluated, `182` candidates, labels `{'correct': 68, 'near_miss': 12, 'wrong_code': 71, 'runtime_error': 0, 'malformed': 31}`, strict-clean tournaments `3`, diagnostic-mixed tournaments `21`, random baseline `0.5468253968253969`.
- caveat on relabel: two old candidate tasks no longer match the regenerated patched taskset (`mbpp/251`, `mbpp/255` old; `mbpp/111`, `mbpp/230` new), so this relabel is a sanity check only, not a replacement transfer result.
- next valid code transfer run: regenerate candidates with the patched local-agent/taskset path, then recapture features and rerun HH-trained AntisymLinear / NoNorm transfer. Do not reuse the old v2 feature tensor as a current result.

## Patched code branch pilot v2-mini (2026-05-16)

- CODE_V2_PATCH_STATUS: `READY`
- CODE_V2_MINI_TASKSET_VERDICT: `READY`
- CODE_V2_MINI_GENERATION_VERDICT: `READY`
- CODE_V2_MINI_TOURNAMENT_VERDICT: `RUNNABLE_DIAGNOSTIC`
- CODE_V2_MINI_TRANSFER_VERDICT: `GOOD`
- tasks: `30`
- unique candidates: `112`
- label counts: `{'correct': 49, 'near_miss': 11, 'wrong_code': 50, 'runtime_error': 0, 'malformed': 2, 'safety_rejected': 0}`
- strict_clean / diagnostic_runnable / diagnostic_mixed: `2 / 8 / 8`
- primary_eval_set: `diagnostic_runnable`
- random_top1_baseline: `0.5625`
- best AntisymLinear row: `{'config': '36_mean', 'architecture': 'AntisymLinear', 'top1': 0.875, 'pairwise': 0.78125, 'cycle': 0.0, 'margin_mean': 0.9804582111537457, 'margin_std': 0.7993556956798318}`
- best NoNorm row: `{'config': '47_L4', 'architecture': 'AntisymLinearNoNorm', 'top1': 1.0, 'pairwise': 0.875, 'cycle': 0.0, 'margin_mean': 0.7644035294651985, 'margin_std': 1.0323769370663691}`
- winner top1 / pairwise / cycle: `{'top1': 1.0, 'pairwise': 0.875, 'cycle': 0.0}`
- transfer signal survived patched path: `True`
- wrapper-prose artifacts eliminated: `False`
- mbpp/232 fixed: `True`
- mbpp/306 fixed: `True`
- future-risk-register path: `artifacts/reports/probes/code_branch_future_risks_2026-05-16.md`
- full report path: `artifacts/reports/probes/code_branch_pilot_v2_mini_patched_2026-05-16_summary.md`
- interpretation: The patched harness produced usable code-branch tournaments and HH-trained linear transfer remained above the random branch-selection baseline.

## Patched code branch pilot v2-mini clarification (2026-05-16)

- wrapper-prose artifacts eliminated from admitted candidates: `True`
- raw wrapper/status generations rejected before candidate admission: `6`
- admitted wrapper/status artifact count: `0`

## Code near-miss balancing pass (2026-05-17)

- BALANCE_INSPECTION_VERDICT: `READY`
- BALANCING_GENERATION_VERDICT: `COMPLETED`
- BALANCED_TOURNAMENT_VERDICT: `RED`
- BALANCED_FEATURE_VERDICT: `NOT_RUN`
- BALANCED_TRANSFER_VERDICT: `NOT_RUN`
- tasks: `10`
- old strict_clean / new strict_clean: `2 / 2`
- old label counts / new label counts: `{'correct': 14, 'near_miss': 12, 'wrong_code': 11, 'runtime_error': 0, 'malformed': 0, 'safety_rejected': 0} / {'correct': 18, 'near_miss': 16, 'wrong_code': 13, 'runtime_error': 0, 'malformed': 0, 'safety_rejected': 0}`
- tasks converted to strict_clean: `[]`
- modes that helped: `[]`
- best AntisymLinear row: `NA`
- best NoNorm row: `NA`
- full report path: `artifacts/reports/probes/code_branch_near_miss_balancing_2026-05-17_summary.md`
- interpretation: The bottleneck remains task/test design or missing-side generation reliability.

## Current transfer state and scalar-vs-relational audit (2026-05-17)

- CURRENT_ARTIFACT_INVENTORY_VERDICT: `READY`
- GENERALIZATION_VERDICT: `SUPPORTED_FOR_LOCAL_PLANNING`
- POINTWISE_RANKING_VERDICT: `SUPPORTED_IN_OBJECTIVE_DOMAINS`
- clean GSM8K: expanded run was `CLEAN_MINIMUM` with `GOOD` HH-trained linear transfer; GRU control was `GRU_WEAK`.
- patched code v2-mini: patched pipeline reached `RUNNABLE_DIAGNOSTIC` and `GOOD` transfer, with NoNorm strongest on the diagnostic set.
- near-miss enrichment/balancing: correct and near_miss candidates exist globally, but strict_clean stayed low at 2 after balancing.
- current bottleneck: `WITHIN_TASK_PAIRING`
- full reports: `artifacts/reports/probes/current_bg_transfer_state_2026-05-17_summary.md`, `docs/evaluator/current-state.md`, `artifacts/reports/probes/scalar_vs_relational_current_state_2026-05-17.md`, `artifacts/reports/probes/strict_clean_task_screening_protocol_2026-05-17.md`.
- interpretation: transfer is supported enough for local planning; the next bottleneck is strict-clean code task screening, not another broad transfer existence probe.

## Strict-clean code task screening (2026-05-17)

- SCREENING_TASKPOOL_VERDICT: `READY`
- SCREENING_GENERATION_VERDICT: `COMPLETED`
- STRICT_CLEAN_SCREENING_VERDICT: `YELLOW`
- tasks screened: `60`
- strict_clean_ready: `6`
- anchor_only / near_miss_only / all_correct / all_wrong: `5 / 7 / 36 / 6`
- label totals: `{'correct': 73, 'wrong_code': 24, 'near_miss': 20}`
- per-source summary: `{'mbpp': {'all_correct': 36, 'all_wrong': 6, 'strict_clean_ready': 6, 'anchor_only': 5, 'near_miss_only': 7}}`
- per-difficulty summary: `{'hard': {'all_correct': 2, 'all_wrong': 1}, 'medium': {'strict_clean_ready': 6, 'all_correct': 34, 'anchor_only': 5, 'all_wrong': 5, 'near_miss_only': 7}}`
- within-task pairing bottleneck: `CONFIRMED`
- full report: `artifacts/reports/probes/code_strict_clean_screening_2026-05-17_summary.md`
- interpretation: Screening still shows the main constraint is finding same-task correct-vs-near-miss pairs cheaply.

## Strict-clean code transfer micro-eval (2026-05-17)

- STRICT_CLEAN_TRANSFER_SET_VERDICT: `READY`
- STRICT_CLEAN_FEATURE_VERDICT: `READY`
- STRICT_CLEAN_TRANSFER_VERDICT: `WEAK`
- tasks: `6`
- candidates: `13` primary / `15` secondary
- random_top1_baseline: `0.527777781089147`
- best AntisymLinear row: `{'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 0.5, 'pairwise': 0.42857142857142855, 'cycle': 0.0, 'margin_mean': 0.634135976433754, 'margin_std': 0.5422409172287559}`
- best NoNorm row: `{'config': '47_concat_all_loops', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5, 'pairwise': 0.5714285714285714, 'cycle': 0.0, 'margin_mean': 0.7558111765732368, 'margin_std': 0.9909718739852695}`
- winner top1 / pairwise / cycle: `{'top1': 0.5, 'pairwise': 0.5714285714285714, 'cycle': 0.0}`
- transfer survived on strict-clean correct-vs-near_miss candidates: `weak`
- full report: `artifacts/reports/probes/code_strict_clean_transfer_2026-05-17_summary.md`
- interpretation: HH-trained tiny taps show a weak strict-clean code transfer signal on this small correct-vs-near_miss micro-set.

## Code-specific tiny-head control + strict-clean screening expansion (2026-05-17)

- CODE_SPECIFIC_SPLIT_VERDICT: `MISSING_FEATURES`
- CODE_SPECIFIC_FEATURE_VERDICT: `RECAPTURED`
- CODE_SPECIFIC_TINY_HEAD_VERDICT: `GOOD`
- STRICT_CLEAN_SCREENING_EXPANSION_VERDICT: `GREEN`
- training task count / pair count: `14` / `138`
- held-out strict-clean task count: `6`
- best code-trained AntisymLinear row: `{config: 24_L4, architecture: AntisymLinear, top1: 0.8333333333333334, pairwise: 0.7142857142857143, cycle: 0.0, margin_mean: 3.946825544039408, margin_std: 1.9451767754980098}`
- best code-trained NoNorm row: `{config: 24_L4, architecture: AntisymLinearNoNorm, top1: 0.8333333333333334, pairwise: 0.7142857142857143, cycle: 0.0, margin_mean: 0.4422141411341727, margin_std: 0.3071355319441245}`
- comparison to HH-trained strict-clean WEAK result: `47_concat_all_loops / AntisymLinearNoNorm`, top1=0.500, pairwise=0.571, cycle=0.000.
- screening expansion counts: tasks_screened=`137`, strict_clean_ready=`10`, label_totals=`{correct: 164, near_miss: 60, wrong_code: 64, safety_rejected: 4, malformed: 3}`
- screening expansion source mix: `{humaneval: {strict_clean_ready: 7}, mbpp: {strict_clean_ready: 3}}` strict-clean-ready contribution; HumanEval was used with granularized checks where possible after the MBPP-only tail stayed low-yield.
- new strict_clean_ready task IDs: `[mbpp/11, mbpp/20, mbpp/434, HumanEval/10, HumanEval/118, HumanEval/123, HumanEval/125, HumanEval/141, HumanEval/148, HumanEval/69]`
- interpretation: states contain a strict-clean code branch signal; HH-trained projection did not transfer strongly enough to near-miss code, while code-specific tiny taps can read it.

## Expanded strict-clean code projection comparison (2026-05-17)

- EXPANDED_STRICT_CLEAN_SET_VERDICT: `READY`
- EXPANDED_STRICT_CLEAN_FEATURE_VERDICT: `RECAPTURED`
- CODE_SPECIFIC_EXPANDED_TRAIN_VERDICT: `RETRAINED`
- EXPANDED_HH_TRANSFER_VERDICT: `GOOD`
- EXPANDED_CODE_SPECIFIC_TRANSFER_VERDICT: `GOOD`
- EXPANDED_STRICT_CLEAN_COMPARISON_VERDICT: `CODE_SPECIFIC_ADVANTAGE`
- eval tasks: `16`
- OLD6 best HH / CODE: `{'head_family': 'HH', 'config': '47_concat_all_loops', 'architecture': 'AntisymLinearNoNorm', 'family_architecture': 'HH_NoNorm', 'top1': 0.5, 'over_random': -0.02777778108914697, 'pairwise': 0.5714285714285714, 'cycle': 0.0, 'margin_mean': 0.7558111765732368, 'margin_std': 0.9909718739852695}` / `{'head_family': 'CODE', 'config': '36_L4', 'architecture': 'AntisymLinear', 'family_architecture': 'CODE_AntisymLinear', 'top1': 1.0, 'over_random': 0.47222221891085303, 'pairwise': 0.8571428571428571, 'cycle': 0.0, 'margin_mean': 1.4508539686600368, 'margin_std': 1.1298246732529835}`
- NEW10 best HH / CODE: `{'head_family': 'HH', 'config': '47_L4', 'architecture': 'AntisymLinear', 'family_architecture': 'HH_AntisymLinear', 'top1': 0.9, 'over_random': 0.3176190376281739, 'pairwise': 0.7391304347826086, 'cycle': 0.0, 'margin_mean': 1.305394220352173, 'margin_std': 1.134789404785102}` / `{'head_family': 'CODE', 'config': '47_L4', 'architecture': 'AntisymLinear', 'family_architecture': 'CODE_AntisymLinear', 'top1': 0.9, 'over_random': 0.3176190376281739, 'pairwise': 0.8695652173913043, 'cycle': 0.0, 'margin_mean': 3.521452635526657, 'margin_std': 3.307454072283649}`
- ALL16 best HH / CODE: `{'head_family': 'HH', 'config': '47_mean', 'architecture': 'AntisymLinear', 'family_architecture': 'HH_AntisymLinear', 'top1': 0.75, 'over_random': 0.18809523060917854, 'pairwise': 0.6, 'cycle': 0.0, 'margin_mean': 0.9949273709207773, 'margin_std': 0.9282425444305212}` / `{'head_family': 'CODE', 'config': '36_L4', 'architecture': 'AntisymLinear', 'family_architecture': 'CODE_AntisymLinear', 'top1': 0.875, 'over_random': 0.31309523060917854, 'pairwise': 0.8333333333333334, 'cycle': 0.0, 'margin_mean': 1.6726789940148592, 'margin_std': 1.4576284253711889}`
- best HH-trained row: `{'head_family': 'HH', 'config': '47_mean', 'architecture': 'AntisymLinear', 'family_architecture': 'HH_AntisymLinear', 'top1': 0.75, 'over_random': 0.18809523060917854, 'pairwise': 0.6, 'cycle': 0.0, 'margin_mean': 0.9949273709207773, 'margin_std': 0.9282425444305212}`
- best code-trained row: `{'head_family': 'CODE', 'config': '36_L4', 'architecture': 'AntisymLinear', 'family_architecture': 'CODE_AntisymLinear', 'top1': 0.875, 'over_random': 0.31309523060917854, 'pairwise': 0.8333333333333334, 'cycle': 0.0, 'margin_mean': 1.6726789940148592, 'margin_std': 1.4576284253711889}`
- code-specific advantage survived: `yes`
- winning architecture: `AntisymLinear`
- winning layer: `36`
- interpretation: Code-specific projection training remains better than HH-trained projection on the expanded strict-clean code branch-selection set.
- full report: `artifacts/reports/probes/expanded_strict_clean_code_projection_comparison_2026-05-17_summary.md`

## Cross-domain fixed-config + reasoning pilot (2026-05-17)

- BG_BACKLOG_AUDIT_VERDICT: `READY`
- BG_HEAD_REGISTRY_VERDICT: `RETRAINED`
- BG_CROSS_DOMAIN_MATRIX_VERDICT: `READY`
- FIXED_CONFIG_AUDIT_VERDICT: `READY`
- GENERALIST_SPECIALIST_VERDICT: `DOMAIN_SPECIALISTS_NEEDED`
- REASONING_BRANCH_DATA_VERDICT: `READY`
- REASONING_TRANSFER_VERDICT: `GOOD`
- LOOP_LAYER_DIAGNOSTIC_VERDICT: `READY`
- best fixed configs / stability: `[{'head_key': 'CODE_AntisymLinearNoNorm::36_mean', 'domains_within_0p05_of_best': 3, 'domains_seen': 6, 'avg_pairwise': nan, 'avg_top1_over_random': 0.17190475741490013}, {'head_key': 'CODE_AntisymLinear::24_L4', 'domains_within_0p05_of_best': 2, 'domains_seen': 6, 'avg_pairwise': nan, 'avg_top1_over_random': 0.16380951753152267}, {'head_key': 'CODE_AntisymLinear::36_L4', 'domains_within_0p05_of_best': 2, 'domains_seen': 6, 'avg_pairwise': nan, 'avg_top1_over_random': 0.1667857090011239}, {'head_key': 'CODE_AntisymLinearNoNorm::36_L4', 'domains_within_0p05_of_best': 2, 'domains_seen': 6, 'avg_pairwise': nan, 'avg_top1_over_random': 0.1526190409436822}, {'head_key': 'CODE_AntisymLinear::47_L4', 'domains_within_0p05_of_best': 2, 'domains_seen': 6, 'avg_pairwise': nan, 'avg_top1_over_random': 0.18249999491409177}]`
- HH vs code-trained comparison: `{'CODE_SPECIFIC_ADVANTAGE_ON_STRICT_CLEAN': True, 'HH_GENERAL_ADVANTAGE_ON_HH': True, 'SHARED_COHERENCE_AXIS': 'weak', 'best_code_all16': {'domain': 'CODE_STRICT_CLEAN_ALL16', 'family': 'CODE', 'config': '36_L4', 'architecture': 'AntisymLinear', 'top1': 0.875, 'over_random': 0.31309523060917854, 'pairwise': 0.8333333333333334, 'cycle': 0.0, 'margin_mean': 1.6726789940148592, 'margin_std': 1.4576284253711889}, 'best_hh_all16': {'domain': 'CODE_STRICT_CLEAN_ALL16', 'family': 'HH', 'config': '47_mean', 'architecture': 'AntisymLinear', 'top1': 0.75, 'over_random': 0.18809523060917854, 'pairwise': 0.6, 'cycle': 0.0, 'margin_mean': 0.9949273709207773, 'margin_std': 0.9282425444305212}, 'best_code_hh': {'domain': 'HH_200', 'family': 'CODE', 'config': '47_L4', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5350000262260437, 'over_random': 0.0350000262260437, 'pairwise': 0.5350000262260437, 'cycle': nan, 'margin_mean': 0.17239375412464142, 'margin_std': 1.2840533256530762}, 'best_hh_hh': {'domain': 'HH_200', 'family': 'HH', 'config': '47_concat_L1_L4', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.8550000190734863, 'over_random': 0.35500001907348633, 'pairwise': 0.8550000190734863, 'cycle': nan, 'margin_mean': 1.262777328491211, 'margin_std': 1.140013337135315}}`
- NoNorm vs AntisymLinear comparison: `{'best_per_domain': {'CLEAN_GSM8K_EXPANDED': {'domain': 'CLEAN_GSM8K_EXPANDED', 'family': 'CODE', 'config': '24_mean', 'architecture': 'AntisymLinear', 'top1': 0.8928571428571429, 'over_random': 0.3303571354065623, 'pairwise': 0.7962962962962963, 'cycle': 0.0, 'margin_mean': 1.7689685566084725, 'margin_std': 1.411621785660656}, 'CODE_RUNNABLE_DIAGNOSTIC': {'domain': 'CODE_RUNNABLE_DIAGNOSTIC', 'family': 'CODE', 'config': '24_L1', 'architecture': 'AntisymLinear', 'top1': 1.0, 'over_random': 0.43749999441206455, 'pairwise': 1.0, 'cycle': 0.0, 'margin_mean': 7.189825654029846, 'margin_std': 6.033544621780366}, 'CODE_STRICT_CLEAN_ALL16': {'domain': 'CODE_STRICT_CLEAN_ALL16', 'family': 'CODE', 'config': '36_L4', 'architecture': 'AntisymLinear', 'top1': 0.875, 'over_random': 0.31309523060917854, 'pairwise': 0.8333333333333334, 'cycle': 0.0, 'margin_mean': 1.6726789940148592, 'margin_std': 1.4576284253711889}, 'CODE_STRICT_CLEAN_NEW10': {'domain': 'CODE_STRICT_CLEAN_NEW10', 'family': 'CODE', 'config': '47_L4', 'architecture': 'AntisymLinear', 'top1': 0.9, 'over_random': 0.3176190376281739, 'pairwise': 0.8695652173913043, 'cycle': 0.0, 'margin_mean': 3.521452635526657, 'margin_std': 3.307454072283649}, 'CODE_STRICT_CLEAN_OLD6': {'domain': 'CODE_STRICT_CLEAN_OLD6', 'family': 'HH', 'config': '24_L1', 'architecture': 'AntisymLinear', 'top1': 0.0, 'over_random': 0.0, 'pairwise': nan, 'cycle': 0.0, 'margin_mean': 0.6804154217243195, 'margin_std': 0.2965801554579709}, 'HH_200': {'domain': 'HH_200', 'family': 'HH', 'config': '47_concat_L1_L4', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.8550000190734863, 'over_random': 0.35500001907348633, 'pairwise': 0.8550000190734863, 'cycle': nan, 'margin_mean': 1.262777328491211, 'margin_std': 1.140013337135315}}, 'interpretation': 'NoNorm remains useful in objective domains, while AntisymLinear remains the safer default for relational/noisy preference-like distinctions.'}`
- reasoning pilot result: `{'REASONING_TRANSFER_VERDICT': 'GOOD', 'n_tournaments': 25, 'random_top1_baseline': 0.4400000047683716, 'best_overall': {'family': 'CODE', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 1.0, 'over_random': 0.5599999952316284, 'pairwise': 1.0, 'cycle': 0.0}, 'best_hh': {'family': 'HH', 'config': '36_mean', 'architecture': 'AntisymLinear', 'top1': 0.96, 'over_random': 0.5199999952316283, 'pairwise': 0.9861111111111112, 'cycle': 0.0}, 'best_code': {'family': 'CODE', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 1.0, 'over_random': 0.5599999952316284, 'pairwise': 1.0, 'cycle': 0.0}, 'per_dataset': {'ai2_arc_challenge': {'baseline': 0.4270833395421505, 'best': {'family': 'CODE', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 1.0, 'over_random': 0.5729166604578495, 'pairwise': 1.0, 'cycle': 0.0}}, 'openbookqa': {'baseline': 0.4629629651705424, 'best': {'family': 'HH', 'config': '36_L4', 'architecture': 'AntisymLinear', 'top1': 1.0, 'over_random': 0.5370370348294575, 'pairwise': 1.0, 'cycle': 0.0}}}}`
- RECOMMENDED_NEXT: `add_reasoning_as_third_objective_eval_domain`
- full reports: `artifacts/reports/probes/bg_cross_domain_reasoning_audit_2026-05-17_summary.md`, `artifacts/reports/probes/bg_fixed_config_cross_domain_audit_2026-05-17.md`, `artifacts/reports/probes/reasoning_branch_transfer_2026-05-17.md`

## Hard reasoning natural-distractor validation (2026-05-17)

- REASONING_DISTRACTOR_SET_VERDICT: `READY`
- REASONING_DISTRACTOR_FEATURE_VERDICT: `READY`
- REASONING_DISTRACTOR_TRANSFER_VERDICT: `GOOD`
- REASONING_SPECIALIST_VERDICT: `GENERAL_SUFFICIENT`
- REASONING_DIFFICULTY_VERDICT: `DISTRACTORS_HARDER`
- dataset/task counts: `{'ai2_arc_challenge': 30, 'openbookqa': 30}`
- random_top1_baseline: `0.25`
- best HH row: `{'family': 'HH', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5666666666666667, 'over_random': 0.31666666666666665, 'pairwise': 0.7333333333333333, 'condorcet': 0.5666666666666667, 'cycle': 0.0, 'margin_mean': 0.33211478032171726, 'margin_std': 0.3502592859723129}`
- best code row: `{'family': 'CODE', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.6, 'over_random': 0.35, 'pairwise': 0.7222222222222222, 'condorcet': 0.6, 'cycle': 0.0, 'margin_mean': 0.5857972353696823, 'margin_std': 0.5569268932391405}`
- best NoNorm row: `{'family': 'HH', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5666666666666667, 'over_random': 0.31666666666666665, 'pairwise': 0.7333333333333333, 'condorcet': 0.5666666666666667, 'cycle': 0.0, 'margin_mean': 0.33211478032171726, 'margin_std': 0.3502592859723129}`
- best AntisymLinear row: `{'family': 'HH', 'config': '36_L4', 'architecture': 'AntisymLinear', 'top1': 0.4166666666666667, 'over_random': 0.16666666666666669, 'pairwise': 0.6611111111111111, 'condorcet': 0.4166666666666667, 'cycle': 0.0, 'margin_mean': 1.0988494743903479, 'margin_std': 0.9490593444718143}`
- generated-vs-distractor comparison: `{'REASONING_DIFFICULTY_VERDICT': 'DISTRACTORS_HARDER', 'generated_n_tournaments': 25, 'distractor_n_tournaments': 60, 'generated_random_top1_baseline': 0.4400000047683716, 'distractor_random_top1_baseline': 0.25, 'generated_best_hh': {'family': 'HH', 'config': '36_mean', 'architecture': 'AntisymLinear', 'top1': 0.96, 'over_random': 0.5199999952316283, 'pairwise': 0.9861111111111112, 'cycle': 0.0}, 'distractor_best_hh': {'family': 'HH', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5666666666666667, 'over_random': 0.31666666666666665, 'pairwise': 0.7333333333333333, 'condorcet': 0.5666666666666667, 'cycle': 0.0, 'margin_mean': 0.33211478032171726, 'margin_std': 0.3502592859723129}, 'generated_best_code': {'family': 'CODE', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 1.0, 'over_random': 0.5599999952316284, 'pairwise': 1.0, 'cycle': 0.0}, 'distractor_best_code': {'family': 'CODE', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.6, 'over_random': 0.35, 'pairwise': 0.7222222222222222, 'condorcet': 0.6, 'cycle': 0.0, 'margin_mean': 0.5857972353696823, 'margin_std': 0.5569268932391405}, 'generated_best_nonorm': 'not_reported', 'distractor_best_nonorm': {'family': 'HH', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5666666666666667, 'over_random': 0.31666666666666665, 'pairwise': 0.7333333333333333, 'condorcet': 0.5666666666666667, 'cycle': 0.0, 'margin_mean': 0.33211478032171726, 'margin_std': 0.3502592859723129}, 'generated_best_antisymlinear': 'not_reported', 'distractor_best_antisymlinear': {'family': 'HH', 'config': '36_L4', 'architecture': 'AntisymLinear', 'top1': 0.4166666666666667, 'over_random': 0.16666666666666669, 'pairwise': 0.6611111111111111, 'condorcet': 0.4166666666666667, 'cycle': 0.0, 'margin_mean': 1.0988494743903479, 'margin_std': 0.9490593444718143}, 'generated_best_overall': {'family': 'CODE', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 1.0, 'over_random': 0.5599999952316284, 'pairwise': 1.0, 'cycle': 0.0}, 'distractor_best_overall': {'family': 'HH', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5666666666666667, 'over_random': 0.31666666666666665, 'pairwise': 0.7333333333333333, 'condorcet': 0.5666666666666667, 'cycle': 0.0, 'margin_mean': 0.33211478032171726, 'margin_std': 0.3502592859723129}, 'generated_data_counts': {'REASONING_BRANCH_DATA_VERDICT': 'READY', 'tasks_seen': 30, 'mixed_tournaments': 25, 'candidates_total': 120, 'label_counts': {'incorrect': 64, 'correct': 47, 'unparseable': 9}, 'unparseable_rate': 0.075, 'dataset_counts': {'ai2_arc_challenge': 80, 'openbookqa': 40}}, 'interpretation': 'Natural distractors reduced transfer performance relative to generated answer branches.'}`
- reasoning third objective eval domain: `still_supported`
- reasoning specialist justified yet: `not_yet_general_head_sufficient`
- full reports: `artifacts/reports/probes/reasoning_natural_distractor_audit_2026-05-17_summary.md`, `artifacts/reports/probes/reasoning_natural_distractor_transfer_2026-05-17.md`, `artifacts/reports/probes/reasoning_generated_vs_distractor_comparison_2026-05-17.md`
- interpretation: Natural distractors reduced performance relative to generated branches and are a better stress test for reasoning readouts.

## Reasoning trace near-miss + code taps on math/logic (2026-05-17)

- REASONING_TRACE_TASK_SET_VERDICT: `READY`
- REASONING_TRACE_DATA_VERDICT: `PARTIAL`
- REASONING_TRACE_FEATURE_VERDICT: `READY`
- REASONING_TRACE_TRANSFER_VERDICT: `GOOD`
- REASONING_TRACE_SPECIALIST_VERDICT: `GENERAL_SUFFICIENT`
- REASONING_TRACE_DIFFICULTY_VERDICT: `DISTRACTORS_HARDER`
- CODE_TAPS_ON_MATH_VERDICT: `GOOD`
- CODE_TAPS_ON_LOGIC_VERDICT: `GOOD`
- dataset/task counts: `{'ai2_arc_challenge': 15, 'openbookqa': 15}`
- best HH row: `{'family': 'HH', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 0.7083333333333334, 'over_random': 0.42013888433575636, 'pairwise': 0.8524590163934426, 'condorcet': 0.7083333333333334, 'cycle': 0.0, 'margin_mean': 1.3928159878899653, 'margin_std': 1.1117851251122828}`
- best code row: `{'family': 'CODE', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 0.75, 'over_random': 0.461805551002423, 'pairwise': 0.8688524590163934, 'condorcet': 0.75, 'cycle': 0.0, 'margin_mean': 2.297035602852702, 'margin_std': 1.4866195021987771}`
- best NoNorm row: `{'family': 'CODE', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.6666666666666666, 'over_random': 0.3784722176690896, 'pairwise': 0.8524590163934426, 'condorcet': 0.6666666666666666, 'cycle': 0.0, 'margin_mean': 1.19206016138196, 'margin_std': 0.9531564312749697}`
- best AntisymLinear row: `{'family': 'CODE', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 0.75, 'over_random': 0.461805551002423, 'pairwise': 0.8688524590163934, 'condorcet': 0.75, 'cycle': 0.0, 'margin_mean': 2.297035602852702, 'margin_std': 1.4866195021987771}`
- comparison to natural distractor reasoning: `{'REASONING_TRACE_DIFFICULTY_VERDICT': 'DISTRACTORS_HARDER', 'generated_answer_branches': {'n_tournaments': 25, 'random_top1_baseline': 0.4400000047683716, 'best_hh': {'family': 'HH', 'config': '36_mean', 'architecture': 'AntisymLinear', 'top1': 0.96, 'over_random': 0.5199999952316283, 'pairwise': 0.9861111111111112, 'cycle': 0.0}, 'best_code': {'family': 'CODE', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 1.0, 'over_random': 0.5599999952316284, 'pairwise': 1.0, 'cycle': 0.0}, 'best_nonorm': None, 'best_antisymlinear': None, 'best_overall': {'family': 'CODE', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 1.0, 'over_random': 0.5599999952316284, 'pairwise': 1.0, 'cycle': 0.0}, 'verdict': 'GOOD', 'specialist_verdict': None}, 'natural_distractors': {'n_tournaments': 60, 'random_top1_baseline': 0.25, 'best_hh': {'family': 'HH', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5666666666666667, 'over_random': 0.31666666666666665, 'pairwise': 0.7333333333333333, 'condorcet': 0.5666666666666667, 'cycle': 0.0, 'margin_mean': 0.33211478032171726, 'margin_std': 0.3502592859723129}, 'best_code': {'family': 'CODE', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.6, 'over_random': 0.35, 'pairwise': 0.7222222222222222, 'condorcet': 0.6, 'cycle': 0.0, 'margin_mean': 0.5857972353696823, 'margin_std': 0.5569268932391405}, 'best_nonorm': {'family': 'HH', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5666666666666667, 'over_random': 0.31666666666666665, 'pairwise': 0.7333333333333333, 'condorcet': 0.5666666666666667, 'cycle': 0.0, 'margin_mean': 0.33211478032171726, 'margin_std': 0.3502592859723129}, 'best_antisymlinear': {'family': 'HH', 'config': '36_L4', 'architecture': 'AntisymLinear', 'top1': 0.4166666666666667, 'over_random': 0.16666666666666669, 'pairwise': 0.6611111111111111, 'condorcet': 0.4166666666666667, 'cycle': 0.0, 'margin_mean': 1.0988494743903479, 'margin_std': 0.9490593444718143}, 'best_overall': {'family': 'HH', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5666666666666667, 'over_random': 0.31666666666666665, 'pairwise': 0.7333333333333333, 'condorcet': 0.5666666666666667, 'cycle': 0.0, 'margin_mean': 0.33211478032171726, 'margin_std': 0.3502592859723129}, 'verdict': 'GOOD', 'specialist_verdict': 'GENERAL_SUFFICIENT'}, 'generated_reasoning_traces': {'n_tournaments': 24, 'random_top1_baseline': 0.288194448997577, 'best_hh': {'family': 'HH', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 0.7083333333333334, 'over_random': 0.42013888433575636, 'pairwise': 0.8524590163934426, 'condorcet': 0.7083333333333334, 'cycle': 0.0, 'margin_mean': 1.3928159878899653, 'margin_std': 1.1117851251122828}, 'best_code': {'family': 'CODE', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 0.75, 'over_random': 0.461805551002423, 'pairwise': 0.8688524590163934, 'condorcet': 0.75, 'cycle': 0.0, 'margin_mean': 2.297035602852702, 'margin_std': 1.4866195021987771}, 'best_nonorm': {'family': 'CODE', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.6666666666666666, 'over_random': 0.3784722176690896, 'pairwise': 0.8524590163934426, 'condorcet': 0.6666666666666666, 'cycle': 0.0, 'margin_mean': 1.19206016138196, 'margin_std': 0.9531564312749697}, 'best_antisymlinear': {'family': 'CODE', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 0.75, 'over_random': 0.461805551002423, 'pairwise': 0.8688524590163934, 'condorcet': 0.75, 'cycle': 0.0, 'margin_mean': 2.297035602852702, 'margin_std': 1.4866195021987771}, 'best_overall': {'family': 'CODE', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 0.75, 'over_random': 0.461805551002423, 'pairwise': 0.8688524590163934, 'condorcet': 0.75, 'cycle': 0.0, 'margin_mean': 2.297035602852702, 'margin_std': 1.4866195021987771}, 'verdict': 'GOOD', 'specialist_verdict': 'GENERAL_SUFFICIENT'}, 'interpretation': 'Natural answer distractors remain harder than generated reasoning traces.'}`
- comparison to clean GSM8K/code taps: `{'CODE_TAPS_ON_MATH_VERDICT': 'GOOD', 'CODE_TAPS_ON_LOGIC_VERDICT': 'GOOD', 'domains': [{'domain': 'CLEAN_GSM8K_EXPANDED', 'n_tournaments': 28, 'n_candidates': 79, 'random_top1_baseline': 0.5625000074505806, 'best_code': {'domain': 'CLEAN_GSM8K_EXPANDED', 'family': 'CODE', 'config': '24_mean', 'architecture': 'AntisymLinear', 'top1': 0.8928571428571429, 'over_random': 0.3303571354065623, 'pairwise': 0.7962962962962963, 'cycle': 0.0, 'margin_mean': 1.7689685566084725, 'margin_std': 1.411621785660656}, 'best_hh': {'domain': 'CLEAN_GSM8K_EXPANDED', 'family': 'HH', 'config': '24_L4', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.75, 'over_random': 0.1874999925494194, 'pairwise': 0.7407407407407407, 'cycle': 0.0, 'margin_mean': 0.012588573902446245, 'margin_std': 0.011785265045384234}, 'best_overall': {'domain': 'CLEAN_GSM8K_EXPANDED', 'family': 'CODE', 'config': '24_mean', 'architecture': 'AntisymLinear', 'top1': 0.8928571428571429, 'over_random': 0.3303571354065623, 'pairwise': 0.7962962962962963, 'cycle': 0.0, 'margin_mean': 1.7689685566084725, 'margin_std': 1.411621785660656}, 'best_code_verdict': 'GOOD'}, {'domain': 'REASONING_NATURAL_DISTRACTOR', 'n_tournaments': 60, 'n_candidates': 240, 'random_top1_baseline': 0.25, 'best_code': {'domain': 'REASONING_NATURAL_DISTRACTOR', 'family': 'CODE', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.6, 'over_random': 0.35, 'pairwise': 0.7222222222222222, 'cycle': 0.0, 'margin_mean': 0.5857972353696823, 'margin_std': 0.5569268932391405}, 'best_hh': {'domain': 'REASONING_NATURAL_DISTRACTOR', 'family': 'HH', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5666666666666667, 'over_random': 0.31666666666666665, 'pairwise': 0.7333333333333333, 'cycle': 0.0, 'margin_mean': 0.33211478032171726, 'margin_std': 0.3502592859723129}, 'best_overall': {'domain': 'REASONING_NATURAL_DISTRACTOR', 'family': 'HH', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5666666666666667, 'over_random': 0.31666666666666665, 'pairwise': 0.7333333333333333, 'cycle': 0.0, 'margin_mean': 0.33211478032171726, 'margin_std': 0.3502592859723129}, 'best_code_verdict': 'GOOD'}, {'domain': 'REASONING_TRACE', 'n_tournaments': 24, 'n_candidates': 85, 'random_top1_baseline': 0.288194448997577, 'best_code': {'domain': 'REASONING_TRACE', 'family': 'CODE', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 0.75, 'over_random': 0.461805551002423, 'pairwise': 0.8688524590163934, 'cycle': 0.0, 'margin_mean': 2.297035602852702, 'margin_std': 1.4866195021987771}, 'best_hh': {'domain': 'REASONING_TRACE', 'family': 'HH', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 0.7083333333333334, 'over_random': 0.42013888433575636, 'pairwise': 0.8524590163934426, 'cycle': 0.0, 'margin_mean': 1.3928159878899653, 'margin_std': 1.1117851251122828}, 'best_overall': {'domain': 'REASONING_TRACE', 'family': 'CODE', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 0.75, 'over_random': 0.461805551002423, 'pairwise': 0.8688524590163934, 'cycle': 0.0, 'margin_mean': 2.297035602852702, 'margin_std': 1.4866195021987771}, 'best_code_verdict': 'GOOD'}, {'domain': 'LOGIC_COMBINED', 'n_tournaments': 84, 'n_candidates': 325, 'random_top1_baseline': 0.26091269971359343, 'best_code': {'domain': 'LOGIC_COMBINED', 'family': 'CODE', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.6190476190476191, 'over_random': 0.35813491933402564, 'pairwise': 0.7551867219917012, 'cycle': 0.0, 'margin_mean': 0.759015214230333, 'margin_std': 0.7457431940723517}, 'best_hh': {'domain': 'LOGIC_COMBINED', 'family': 'HH', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5476190476190477, 'over_random': 0.28670634790545424, 'pairwise': 0.7302904564315352, 'cycle': 0.0, 'margin_mean': 0.3825706510494153, 'margin_std': 0.3891915641615466}, 'best_overall': {'domain': 'LOGIC_COMBINED', 'family': 'CODE', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.6190476190476191, 'over_random': 0.35813491933402564, 'pairwise': 0.7551867219917012, 'cycle': 0.0, 'margin_mean': 0.759015214230333, 'margin_std': 0.7457431940723517}, 'best_code_verdict': 'GOOD'}], 'best_code_rows': {'CLEAN_GSM8K_EXPANDED': {'domain': 'CLEAN_GSM8K_EXPANDED', 'family': 'CODE', 'config': '24_mean', 'architecture': 'AntisymLinear', 'top1': 0.8928571428571429, 'over_random': 0.3303571354065623, 'pairwise': 0.7962962962962963, 'cycle': 0.0, 'margin_mean': 1.7689685566084725, 'margin_std': 1.411621785660656}, 'REASONING_NATURAL_DISTRACTOR': {'domain': 'REASONING_NATURAL_DISTRACTOR', 'family': 'CODE', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.6, 'over_random': 0.35, 'pairwise': 0.7222222222222222, 'cycle': 0.0, 'margin_mean': 0.5857972353696823, 'margin_std': 0.5569268932391405}, 'REASONING_TRACE': {'domain': 'REASONING_TRACE', 'family': 'CODE', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 0.75, 'over_random': 0.461805551002423, 'pairwise': 0.8688524590163934, 'cycle': 0.0, 'margin_mean': 2.297035602852702, 'margin_std': 1.4866195021987771}, 'LOGIC_COMBINED': {'domain': 'LOGIC_COMBINED', 'family': 'CODE', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.6190476190476191, 'over_random': 0.35813491933402564, 'pairwise': 0.7551867219917012, 'cycle': 0.0, 'margin_mean': 0.759015214230333, 'margin_std': 0.7457431940723517}}, 'best_hh_rows': {'CLEAN_GSM8K_EXPANDED': {'domain': 'CLEAN_GSM8K_EXPANDED', 'family': 'HH', 'config': '24_L4', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.75, 'over_random': 0.1874999925494194, 'pairwise': 0.7407407407407407, 'cycle': 0.0, 'margin_mean': 0.012588573902446245, 'margin_std': 0.011785265045384234}, 'REASONING_NATURAL_DISTRACTOR': {'domain': 'REASONING_NATURAL_DISTRACTOR', 'family': 'HH', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5666666666666667, 'over_random': 0.31666666666666665, 'pairwise': 0.7333333333333333, 'cycle': 0.0, 'margin_mean': 0.33211478032171726, 'margin_std': 0.3502592859723129}, 'REASONING_TRACE': {'domain': 'REASONING_TRACE', 'family': 'HH', 'config': '24_L4', 'architecture': 'AntisymLinear', 'top1': 0.7083333333333334, 'over_random': 0.42013888433575636, 'pairwise': 0.8524590163934426, 'cycle': 0.0, 'margin_mean': 1.3928159878899653, 'margin_std': 1.1117851251122828}, 'LOGIC_COMBINED': {'domain': 'LOGIC_COMBINED', 'family': 'HH', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5476190476190477, 'over_random': 0.28670634790545424, 'pairwise': 0.7302904564315352, 'cycle': 0.0, 'margin_mean': 0.3825706510494153, 'margin_std': 0.3891915641615466}}, 'blockers': []}`
- full reports: `artifacts/reports/probes/reasoning_trace_and_code_taps_math_logic_2026-05-17_summary.md`, `artifacts/reports/probes/reasoning_trace_transfer_2026-05-17.md`, `artifacts/reports/probes/code_taps_on_math_logic_existing_2026-05-17.md`
- interpretation: Generated reasoning traces still favor the general HH readout enough that a reasoning-specific specialist is not justified by this probe.

## Science / bio / chem / medicine natural-distractor validation (2026-05-17)

- SCIENCE_DISTRACTOR_SET_VERDICT: `READY`
- SCIENCE_DISTRACTOR_FEATURE_VERDICT: `READY`
- SCIENCE_TRANSFER_VERDICT: `GOOD`
- SCIENCE_SPECIALIST_VERDICT: `SPECIALIST_NEEDED`
- BIOLOGY_TRANSFER_VERDICT: `GOOD`
- CHEMISTRY_TRANSFER_VERDICT: `GOOD`
- MEDICINE_TRANSFER_VERDICT: `GOOD`
- GENERAL_SCIENCE_TRANSFER_VERDICT: `GOOD`
- SCIENCE_SPECIFIC_HEAD_VERDICT: `GENERAL_SUFFICIENT`
- SCIENCE_DOMAIN_ANALOGY_VERDICT: `HETEROGENEOUS`
- dataset/task counts: `{'mmlu': 95, 'sciq': 25}`
- random_top1_baseline: `0.25`
- best HH row: `{'family': 'HH', 'config': '36_L4', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.4583333333333333, 'over_random': 0.20833333333333331, 'pairwise': 0.6916666666666667, 'condorcet': 0.4583333333333333, 'cycle': 0.0, 'margin_mean': 0.391814417935287, 'margin_std': 0.48293810592259795}`
- best code row: `{'family': 'CODE', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5416666666666666, 'over_random': 0.29166666666666663, 'pairwise': 0.7027777777777777, 'condorcet': 0.5416666666666666, 'cycle': 0.0, 'margin_mean': 0.6878731965863456, 'margin_std': 0.9032204292594548}`
- best NoNorm row: `{'family': 'CODE', 'config': '36_mean', 'architecture': 'AntisymLinearNoNorm', 'top1': 0.5416666666666666, 'over_random': 0.29166666666666663, 'pairwise': 0.7027777777777777, 'condorcet': 0.5416666666666666, 'cycle': 0.0, 'margin_mean': 0.6878731965863456, 'margin_std': 0.9032204292594548}`
- best AntisymLinear row: `{'family': 'CODE', 'config': '36_L4', 'architecture': 'AntisymLinear', 'top1': 0.43333333333333335, 'over_random': 0.18333333333333335, 'pairwise': 0.625, 'condorcet': 0.4083333333333333, 'cycle': 0.0, 'margin_mean': 1.359556249404947, 'margin_std': 0.9679773308615227}`
- subdomain breakdown: `{'biology': 25, 'chemistry': 25, 'medicine': 25, 'general_science': 25, 'other_science': 20}`
- science objective eval domain status: `add_as_objective_eval_domain`
- science/medicine specialist status: `specialist_needed`
- medicine caveat: benchmark MCQ transfer only, not clinical validation.
- full reports: `artifacts/reports/probes/science_domain_audit_2026-05-17_summary.md`, `artifacts/reports/probes/science_natural_distractor_transfer_2026-05-17.md`, `artifacts/reports/probes/science_domain_comparison_2026-05-17.md`
- interpretation: Science shows a specialist gap in at least one subdomain, so a larger science-specific projection check is justified.


## Mixed-domain tiny heads (2026-05-17)

- MIXED_TAP_SPLIT_VERDICT = READY
- MIXED_TAP_FEATURE_VERDICT = READY
- MIXED_TAP_TRAINING_VERDICT = READY
- MIXED_HEAD_UTILITY_VERDICT = OBJECTIVE_MIXED_USEFUL
- MIXED_HEAD_UTILITY_PROVISIONAL = False
- STRICT_CLEAN_CODE_REGRET_STATUS = CLEAN_WIN
- SMALL_DOMAIN_OVERFIT = True
- DOMAIN_OVERFIT_WARNING = True
- GSM8K_EVAL_STATUS = READY
- best mixed family = MIX_CODE_REASONING
- average objective regret pairwise = 0.048
- worst objective regret pairwise = 0.000
- strict-clean code regret pairwise = 0.067
- reasoning trace regret pairwise = 0.091
- science medicine regret pairwise = -0.083
- clean GSM8K regret pairwise = 0.074
- HH regret pairwise = -0.150
- recommended Phase 1 head set = HH_general_plus_code_specialist_plus_objective_mixed_head
- full reports: `artifacts/reports/probes/mixed_domain_heads_audit_2026-05-17_summary.md`, `artifacts/reports/probes/mixed_domain_head_evaluation_2026-05-17.json`, `artifacts/reports/probes/mixed_head_controller_implications_2026-05-17.md`
- interpretation: mixed heads are controller-routing candidates and should complement, not erase, the established HH/general and code-specialist roles unless regret is cleanly positive outside the current small strict-clean sample.


## BG controller-policy simulator (2026-05-17)

- BG_POLICY_EVAL_BUNDLE_VERDICT = READY
- BG_HEAD_COMPARISON_VERDICT = READY
- HEAD_COMPLEMENTARITY_VERDICT = HIGH_COMPLEMENTARITY
- BG_POLICY_SIM_VERDICT = READY
- BEST_POLICY_VERDICT = OBJECTIVE_MIXED_DEFAULT_WINS
- RECOMMENDED_BG_POLICY = HH_GENERAL_PLUS_OBJECTIVE_MIXED_PLUS_CODE_BACKUP
- CONTRAST_DETECTOR_VERDICT = DEPLOYABILITY_WEAK
- DEFER_POLICY_VERDICT = DEFER_NOT_USEFUL
- ORACLE_GAP_VERDICT = LARGE
- SMALL_N_UNSTABLE_POLICY = True
- STRICT_CLEAN_POLICY_BORDERLINE = True
- HH_HELDOUT_POLICY_BORDERLINE = False
- best policy metrics: DOMAIN_ROUTED_SIMPLE objective_avg=0.817, HH=0.900, strict_clean=0.833
- best single head metrics: OBJECTIVE_MIXED_ONLY objective_avg=0.822; CODE_ONLY strict_clean=0.833
- objective mixed vs code specialist strict-clean delta = 0.033
- HH preservation result: DOMAIN_ROUTED_SIMPLE uses HH_GENERAL on HH, delta vs HH_ONLY = 0.000
- defer policy result: `{'defer_policy_verdict': 'DEFER_NOT_USEFUL', 'best_defer_policy': {'policy': 'ORACLE_POLICY_WITH_DEFER', 'improvement70': 0.02986658580958146, 'improvement80': 0.006450587019319887, 'improvement90': 0.0065386244880666355, 'coverage': 1.0}, 'fallback_adjusted': {'GENERAL_AND_OBJECTIVE_VOTE_defer': {'random_fallback_average': 0.6414534855982225, 'domain_routed_fallback_average': 0.802741845140385, 'by_domain_random': {'CODE_STRICT_CLEAN_ALL16': 0.6833333333333333, 'REASONING_NATURAL_DISTRACTOR': 0.5833333333333333, 'REASONING_TRACE': 0.6363636363636364, 'SCIENCE_OVERALL': 0.6052631578947368, 'SCIENCE_BIOLOGY': 0.625, 'SCIENCE_CHEMISTRY': 0.625, 'SCIENCE_MEDICINE': 0.5, 'SCIENCE_GENERAL': 0.625, 'SCIENCE_OTHER': 0.6666666666666667, 'CODE_RUNNABLE_DIAGNOSTIC': 0.59375, 'CLEAN_GSM8K_EXPANDED': 0.6851851851851852, 'HH_HELDOUT20': 0.8, 'HH_200_DIAGNOSTIC': 0.71}, 'by_domain_domain_routed': {'CODE_STRICT_CLEAN_ALL16': 0.8277777777777777, 'REASONING_NATURAL_DISTRACTOR': 0.6666666666666666, 'REASONING_TRACE': 1.0, 'SCIENCE_OVERALL': 0.7506925207756233, 'SCIENCE_BIOLOGY': 0.875, 'SCIENCE_CHEMISTRY': 0.8125, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.9375, 'SCIENCE_OTHER': 0.6296296296296297, 'CODE_RUNNABLE_DIAGNOSTIC': 0.94921875, 'CLEAN_GSM8K_EXPANDED': 0.808641975308642, 'HH_HELDOUT20': 0.9200000000000002, 'HH_200_DIAGNOSTIC': 0.84135}}, 'THREE_HEAD_VOTE_defer': {'random_fallback_average': 0.5592287074523917, 'domain_routed_fallback_average': 0.7749835455233213, 'by_domain_random': {'CODE_STRICT_CLEAN_ALL16': 0.5666666666666667, 'REASONING_NATURAL_DISTRACTOR': 0.5833333333333334, 'REASONING_TRACE': 0.5, 'SCIENCE_OVERALL': 0.5263157894736843, 'SCIENCE_BIOLOGY': 0.5, 'SCIENCE_CHEMISTRY': 0.5, 'SCIENCE_MEDICINE': 0.5, 'SCIENCE_GENERAL': 0.625, 'SCIENCE_OTHER': 0.5, 'CODE_RUNNABLE_DIAGNOSTIC': 0.59375, 'CLEAN_GSM8K_EXPANDED': 0.6574074074074074, 'HH_HELDOUT20': 0.625, 'HH_200_DIAGNOSTIC': 0.5925}, 'by_domain_domain_routed': {'CODE_STRICT_CLEAN_ALL16': 0.8111111111111111, 'REASONING_NATURAL_DISTRACTOR': 0.7083333333333334, 'REASONING_TRACE': 1.0, 'SCIENCE_OVERALL': 0.7008310249307479, 'SCIENCE_BIOLOGY': 0.8333333333333334, 'SCIENCE_CHEMISTRY': 0.75, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.9375, 'SCIENCE_OTHER': 0.4444444444444444, 'CODE_RUNNABLE_DIAGNOSTIC': 0.94921875, 'CLEAN_GSM8K_EXPANDED': 0.8168724279835391, 'HH_HELDOUT20': 0.885, 'HH_200_DIAGNOSTIC': 0.821475}}, 'CONSENSUS_SELECT_HH_OBJECTIVE': {'random_fallback_average': 0.6414534855982225, 'domain_routed_fallback_average': 0.802741845140385, 'by_domain_random': {'CODE_STRICT_CLEAN_ALL16': 0.6833333333333333, 'REASONING_NATURAL_DISTRACTOR': 0.5833333333333333, 'REASONING_TRACE': 0.6363636363636364, 'SCIENCE_OVERALL': 0.6052631578947368, 'SCIENCE_BIOLOGY': 0.625, 'SCIENCE_CHEMISTRY': 0.625, 'SCIENCE_MEDICINE': 0.5, 'SCIENCE_GENERAL': 0.625, 'SCIENCE_OTHER': 0.6666666666666667, 'CODE_RUNNABLE_DIAGNOSTIC': 0.59375, 'CLEAN_GSM8K_EXPANDED': 0.6851851851851852, 'HH_HELDOUT20': 0.8, 'HH_200_DIAGNOSTIC': 0.71}, 'by_domain_domain_routed': {'CODE_STRICT_CLEAN_ALL16': 0.8277777777777777, 'REASONING_NATURAL_DISTRACTOR': 0.6666666666666666, 'REASONING_TRACE': 1.0, 'SCIENCE_OVERALL': 0.7506925207756233, 'SCIENCE_BIOLOGY': 0.875, 'SCIENCE_CHEMISTRY': 0.8125, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.9375, 'SCIENCE_OTHER': 0.6296296296296297, 'CODE_RUNNABLE_DIAGNOSTIC': 0.94921875, 'CLEAN_GSM8K_EXPANDED': 0.808641975308642, 'HH_HELDOUT20': 0.9200000000000002, 'HH_200_DIAGNOSTIC': 0.84135}}, 'CONSENSUS_SELECT_ALL_THREE': {'random_fallback_average': 0.5592287074523917, 'domain_routed_fallback_average': 0.7749835455233213, 'by_domain_random': {'CODE_STRICT_CLEAN_ALL16': 0.5666666666666667, 'REASONING_NATURAL_DISTRACTOR': 0.5833333333333334, 'REASONING_TRACE': 0.5, 'SCIENCE_OVERALL': 0.5263157894736843, 'SCIENCE_BIOLOGY': 0.5, 'SCIENCE_CHEMISTRY': 0.5, 'SCIENCE_MEDICINE': 0.5, 'SCIENCE_GENERAL': 0.625, 'SCIENCE_OTHER': 0.5, 'CODE_RUNNABLE_DIAGNOSTIC': 0.59375, 'CLEAN_GSM8K_EXPANDED': 0.6574074074074074, 'HH_HELDOUT20': 0.625, 'HH_200_DIAGNOSTIC': 0.5925}, 'by_domain_domain_routed': {'CODE_STRICT_CLEAN_ALL16': 0.8111111111111111, 'REASONING_NATURAL_DISTRACTOR': 0.7083333333333334, 'REASONING_TRACE': 1.0, 'SCIENCE_OVERALL': 0.7008310249307479, 'SCIENCE_BIOLOGY': 0.8333333333333334, 'SCIENCE_CHEMISTRY': 0.75, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.9375, 'SCIENCE_OTHER': 0.4444444444444444, 'CODE_RUNNABLE_DIAGNOSTIC': 0.94921875, 'CLEAN_GSM8K_EXPANDED': 0.8168724279835391, 'HH_HELDOUT20': 0.885, 'HH_200_DIAGNOSTIC': 0.821475}}, 'ORACLE_POLICY_WITH_DEFER': {'random_fallback_average': 0.9050779727095517, 'domain_routed_fallback_average': 0.9050779727095517, 'by_domain_random': {'CODE_STRICT_CLEAN_ALL16': 0.9, 'REASONING_NATURAL_DISTRACTOR': 0.7916666666666666, 'REASONING_TRACE': 1.0, 'SCIENCE_OVERALL': 0.7017543859649122, 'SCIENCE_BIOLOGY': 1.0, 'SCIENCE_CHEMISTRY': 1.0, 'SCIENCE_MEDICINE': 0.8333333333333334, 'SCIENCE_GENERAL': 1.0, 'SCIENCE_OTHER': 0.8888888888888888, 'CODE_RUNNABLE_DIAGNOSTIC': 1.0, 'CLEAN_GSM8K_EXPANDED': 0.8703703703703703, 'HH_HELDOUT20': 0.9, 'HH_200_DIAGNOSTIC': 0.88}, 'by_domain_domain_routed': {'CODE_STRICT_CLEAN_ALL16': 0.9, 'REASONING_NATURAL_DISTRACTOR': 0.7916666666666666, 'REASONING_TRACE': 1.0, 'SCIENCE_OVERALL': 0.7017543859649122, 'SCIENCE_BIOLOGY': 1.0, 'SCIENCE_CHEMISTRY': 1.0, 'SCIENCE_MEDICINE': 0.8333333333333334, 'SCIENCE_GENERAL': 1.0, 'SCIENCE_OTHER': 0.8888888888888888, 'CODE_RUNNABLE_DIAGNOSTIC': 1.0, 'CLEAN_GSM8K_EXPANDED': 0.8703703703703703, 'HH_HELDOUT20': 0.9, 'HH_200_DIAGNOSTIC': 0.88}}, 'MARGIN_DEFER_0.05': {'random_fallback_average': 0.7388607737291948, 'domain_routed_fallback_average': 0.7741764072706132, 'by_domain_random': {'CODE_STRICT_CLEAN_ALL16': 0.6833333333333333, 'REASONING_NATURAL_DISTRACTOR': 0.7291666666666666, 'REASONING_TRACE': 1.0, 'SCIENCE_OVERALL': 0.631578947368421, 'SCIENCE_BIOLOGY': 0.7083333333333334, 'SCIENCE_CHEMISTRY': 0.75, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.7916666666666666, 'SCIENCE_OTHER': 0.4444444444444444, 'CODE_RUNNABLE_DIAGNOSTIC': 0.9375, 'CLEAN_GSM8K_EXPANDED': 0.7499999999999999, 'HH_HELDOUT20': 0.875, 'HH_200_DIAGNOSTIC': 0.8875}, 'by_domain_domain_routed': {'CODE_STRICT_CLEAN_ALL16': 0.8055555555555556, 'REASONING_NATURAL_DISTRACTOR': 0.75, 'REASONING_TRACE': 1.0, 'SCIENCE_OVERALL': 0.6509695290858726, 'SCIENCE_BIOLOGY': 0.7916666666666667, 'SCIENCE_CHEMISTRY': 0.75, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.8958333333333333, 'SCIENCE_OTHER': 0.4444444444444444, 'CODE_RUNNABLE_DIAGNOSTIC': 0.9375, 'CLEAN_GSM8K_EXPANDED': 0.765432098765432, 'HH_HELDOUT20': 0.9349999999999999, 'HH_200_DIAGNOSTIC': 0.921225}}, 'MARGIN_DEFER_0.10': {'random_fallback_average': 0.7237116441063809, 'domain_routed_fallback_average': 0.7740804334930143, 'by_domain_random': {'CODE_STRICT_CLEAN_ALL16': 0.6833333333333333, 'REASONING_NATURAL_DISTRACTOR': 0.7291666666666666, 'REASONING_TRACE': 0.8636363636363636, 'SCIENCE_OVERALL': 0.6052631578947368, 'SCIENCE_BIOLOGY': 0.7083333333333334, 'SCIENCE_CHEMISTRY': 0.75, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.7916666666666666, 'SCIENCE_OTHER': 0.4444444444444444, 'CODE_RUNNABLE_DIAGNOSTIC': 0.9375, 'CLEAN_GSM8K_EXPANDED': 0.7407407407407408, 'HH_HELDOUT20': 0.85, 'HH_200_DIAGNOSTIC': 0.8875}, 'by_domain_domain_routed': {'CODE_STRICT_CLEAN_ALL16': 0.8055555555555556, 'REASONING_NATURAL_DISTRACTOR': 0.75, 'REASONING_TRACE': 1.0, 'SCIENCE_OVERALL': 0.6343490304709142, 'SCIENCE_BIOLOGY': 0.7916666666666667, 'SCIENCE_CHEMISTRY': 0.75, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.8958333333333333, 'SCIENCE_OTHER': 0.4444444444444444, 'CODE_RUNNABLE_DIAGNOSTIC': 0.9375, 'CLEAN_GSM8K_EXPANDED': 0.771604938271605, 'HH_HELDOUT20': 0.9299999999999999, 'HH_200_DIAGNOSTIC': 0.935425}}, 'MARGIN_DEFER_0.20': {'random_fallback_average': 0.7007040717567034, 'domain_routed_fallback_average': 0.7671446459419028, 'by_domain_random': {'CODE_STRICT_CLEAN_ALL16': 0.6833333333333333, 'REASONING_NATURAL_DISTRACTOR': 0.7291666666666666, 'REASONING_TRACE': 0.8636363636363636, 'SCIENCE_OVERALL': 0.6140350877192983, 'SCIENCE_BIOLOGY': 0.7083333333333334, 'SCIENCE_CHEMISTRY': 0.75, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.75, 'SCIENCE_OTHER': 0.2777777777777778, 'CODE_RUNNABLE_DIAGNOSTIC': 0.875, 'CLEAN_GSM8K_EXPANDED': 0.7037037037037037, 'HH_HELDOUT20': 0.85, 'HH_200_DIAGNOSTIC': 0.8875}, 'by_domain_domain_routed': {'CODE_STRICT_CLEAN_ALL16': 0.8055555555555556, 'REASONING_NATURAL_DISTRACTOR': 0.75, 'REASONING_TRACE': 1.0, 'SCIENCE_OVERALL': 0.6528162511542013, 'SCIENCE_BIOLOGY': 0.7916666666666667, 'SCIENCE_CHEMISTRY': 0.75, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.9583333333333333, 'SCIENCE_OTHER': 0.2592592592592593, 'CODE_RUNNABLE_DIAGNOSTIC': 0.9296875, 'CLEAN_GSM8K_EXPANDED': 0.7757201646090535, 'HH_HELDOUT20': 0.9299999999999999, 'HH_200_DIAGNOSTIC': 0.953175}}, 'MARGIN_DEFER_0.30': {'random_fallback_average': 0.7018189315557737, 'domain_routed_fallback_average': 0.7742404131032209, 'by_domain_random': {'CODE_STRICT_CLEAN_ALL16': 0.6333333333333333, 'REASONING_NATURAL_DISTRACTOR': 0.7916666666666666, 'REASONING_TRACE': 0.8636363636363636, 'SCIENCE_OVERALL': 0.5964912280701755, 'SCIENCE_BIOLOGY': 0.7083333333333334, 'SCIENCE_CHEMISTRY': 0.75, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.75, 'SCIENCE_OTHER': 0.33333333333333337, 'CODE_RUNNABLE_DIAGNOSTIC': 0.875, 'CLEAN_GSM8K_EXPANDED': 0.6851851851851852, 'HH_HELDOUT20': 0.85, 'HH_200_DIAGNOSTIC': 0.87}, 'by_domain_domain_routed': {'CODE_STRICT_CLEAN_ALL16': 0.788888888888889, 'REASONING_NATURAL_DISTRACTOR': 0.8333333333333333, 'REASONING_TRACE': 1.0, 'SCIENCE_OVERALL': 0.65466297322253, 'SCIENCE_BIOLOGY': 0.7916666666666667, 'SCIENCE_CHEMISTRY': 0.75, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.9583333333333333, 'SCIENCE_OTHER': 0.29629629629629634, 'CODE_RUNNABLE_DIAGNOSTIC': 0.9296875, 'CLEAN_GSM8K_EXPANDED': 0.7674897119341564, 'HH_HELDOUT20': 0.9299999999999999, 'HH_200_DIAGNOSTIC': 0.9480999999999999}}, 'MARGIN_DEFER_0.50': {'random_fallback_average': 0.68275334314808, 'domain_routed_fallback_average': 0.7896775596633284, 'by_domain_random': {'CODE_STRICT_CLEAN_ALL16': 0.6666666666666666, 'REASONING_NATURAL_DISTRACTOR': 0.7708333333333334, 'REASONING_TRACE': 0.6363636363636364, 'SCIENCE_OVERALL': 0.6052631578947368, 'SCIENCE_BIOLOGY': 0.75, 'SCIENCE_CHEMISTRY': 0.75, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.75, 'SCIENCE_OTHER': 0.33333333333333337, 'CODE_RUNNABLE_DIAGNOSTIC': 0.875, 'CLEAN_GSM8K_EXPANDED': 0.6666666666666667, 'HH_HELDOUT20': 0.825, 'HH_200_DIAGNOSTIC': 0.8300000000000001}, 'by_domain_domain_routed': {'CODE_STRICT_CLEAN_ALL16': 0.8444444444444444, 'REASONING_NATURAL_DISTRACTOR': 0.8333333333333334, 'REASONING_TRACE': 1.0, 'SCIENCE_OVERALL': 0.6925207756232687, 'SCIENCE_BIOLOGY': 0.9166666666666667, 'SCIENCE_CHEMISTRY': 0.75, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.9583333333333333, 'SCIENCE_OTHER': 0.29629629629629634, 'CODE_RUNNABLE_DIAGNOSTIC': 0.9296875, 'CLEAN_GSM8K_EXPANDED': 0.7592592592592593, 'HH_HELDOUT20': 0.9249999999999999, 'HH_200_DIAGNOSTIC': 0.9436}}, 'DISAGREEMENT_DEFER_0.10': {'random_fallback_average': 0.5776632553606238, 'domain_routed_fallback_average': 0.7797329855877198, 'by_domain_random': {'CODE_STRICT_CLEAN_ALL16': 0.65, 'REASONING_NATURAL_DISTRACTOR': 0.5833333333333334, 'REASONING_TRACE': 0.5, 'SCIENCE_OVERALL': 0.5526315789473684, 'SCIENCE_BIOLOGY': 0.625, 'SCIENCE_CHEMISTRY': 0.5, 'SCIENCE_MEDICINE': 0.5, 'SCIENCE_GENERAL': 0.625, 'SCIENCE_OTHER': 0.5, 'CODE_RUNNABLE_DIAGNOSTIC': 0.59375, 'CLEAN_GSM8K_EXPANDED': 0.6574074074074074, 'HH_HELDOUT20': 0.625, 'HH_200_DIAGNOSTIC': 0.5975}, 'by_domain_domain_routed': {'CODE_STRICT_CLEAN_ALL16': 0.8166666666666667, 'REASONING_NATURAL_DISTRACTOR': 0.7083333333333334, 'REASONING_TRACE': 1.0, 'SCIENCE_OVERALL': 0.7174515235457064, 'SCIENCE_BIOLOGY': 0.875, 'SCIENCE_CHEMISTRY': 0.75, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.9375, 'SCIENCE_OTHER': 0.4444444444444444, 'CODE_RUNNABLE_DIAGNOSTIC': 0.94921875, 'CLEAN_GSM8K_EXPANDED': 0.8168724279835391, 'HH_HELDOUT20': 0.885, 'HH_200_DIAGNOSTIC': 0.8193750000000001}}, 'DISAGREEMENT_DEFER_0.20': {'random_fallback_average': 0.5867676938071675, 'domain_routed_fallback_average': 0.7805542946646306, 'by_domain_random': {'CODE_STRICT_CLEAN_ALL16': 0.7166666666666667, 'REASONING_NATURAL_DISTRACTOR': 0.6458333333333333, 'REASONING_TRACE': 0.5, 'SCIENCE_OVERALL': 0.5438596491228069, 'SCIENCE_BIOLOGY': 0.625, 'SCIENCE_CHEMISTRY': 0.5, 'SCIENCE_MEDICINE': 0.5, 'SCIENCE_GENERAL': 0.625, 'SCIENCE_OTHER': 0.4444444444444445, 'CODE_RUNNABLE_DIAGNOSTIC': 0.59375, 'CLEAN_GSM8K_EXPANDED': 0.6759259259259259, 'HH_HELDOUT20': 0.65, 'HH_200_DIAGNOSTIC': 0.6074999999999999}, 'by_domain_domain_routed': {'CODE_STRICT_CLEAN_ALL16': 0.8388888888888889, 'REASONING_NATURAL_DISTRACTOR': 0.75, 'REASONING_TRACE': 1.0, 'SCIENCE_OVERALL': 0.6989843028624192, 'SCIENCE_BIOLOGY': 0.875, 'SCIENCE_CHEMISTRY': 0.75, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.9375, 'SCIENCE_OTHER': 0.40740740740740744, 'CODE_RUNNABLE_DIAGNOSTIC': 0.94921875, 'CLEAN_GSM8K_EXPANDED': 0.8148148148148149, 'HH_HELDOUT20': 0.8900000000000001, 'HH_200_DIAGNOSTIC': 0.8187249999999999}}, 'DISAGREEMENT_DEFER_0.30': {'random_fallback_average': 0.5920312265707003, 'domain_routed_fallback_average': 0.7818001240410155, 'by_domain_random': {'CODE_STRICT_CLEAN_ALL16': 0.7333333333333334, 'REASONING_NATURAL_DISTRACTOR': 0.6458333333333333, 'REASONING_TRACE': 0.5, 'SCIENCE_OVERALL': 0.5438596491228069, 'SCIENCE_BIOLOGY': 0.625, 'SCIENCE_CHEMISTRY': 0.5, 'SCIENCE_MEDICINE': 0.5, 'SCIENCE_GENERAL': 0.625, 'SCIENCE_OTHER': 0.4444444444444445, 'CODE_RUNNABLE_DIAGNOSTIC': 0.59375, 'CLEAN_GSM8K_EXPANDED': 0.6851851851851851, 'HH_HELDOUT20': 0.675, 'HH_200_DIAGNOSTIC': 0.625}, 'by_domain_domain_routed': {'CODE_STRICT_CLEAN_ALL16': 0.8444444444444446, 'REASONING_NATURAL_DISTRACTOR': 0.75, 'REASONING_TRACE': 1.0, 'SCIENCE_OVERALL': 0.6989843028624192, 'SCIENCE_BIOLOGY': 0.875, 'SCIENCE_CHEMISTRY': 0.75, 'SCIENCE_MEDICINE': 0.4166666666666667, 'SCIENCE_GENERAL': 0.9375, 'SCIENCE_OTHER': 0.40740740740740744, 'CODE_RUNNABLE_DIAGNOSTIC': 0.94921875, 'CLEAN_GSM8K_EXPANDED': 0.8189300411522633, 'HH_HELDOUT20': 0.895, 'HH_200_DIAGNOSTIC': 0.82025}}}}`
- oracle-gap summary: average=0.135, objective=0.061
- contrast-detector summary: DEPLOYABILITY_WEAK
- recommended Phase 1 controller design: HH/general for HH and unknown, objective mixed for objective QA/reasoning/science/GSM8K, code specialist backup for strict-clean or high-similarity code, defer on low margin/disagreement.
- full reports: `artifacts/reports/probes/bg_controller_policy_simulator_2026-05-17_summary.md`, `artifacts/reports/probes/bg_controller_policy_simulation_2026-05-17.json`, `artifacts/reports/probes/bg_candidate_head_comparison_2026-05-17.json`, `artifacts/reports/probes/bg_phase1_controller_design_note_2026-05-17.md`
- interpretation: deploy a read-only routed controller; current heads are complementary enough to route, but not stable enough to collapse into one universal head.
