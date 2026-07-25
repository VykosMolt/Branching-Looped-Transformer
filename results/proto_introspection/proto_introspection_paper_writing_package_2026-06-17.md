# Proto-Introspection Paper-Writing Package

**Artifact:** `artifacts/reports/proto_introspection/proto_introspection_paper_writing_package_2026-06-17.md`
**Companion JSON:** `artifacts/reports/proto_introspection/proto_introspection_paper_writing_package_2026-06-17.json`
**Short handoff:** `artifacts/reports/proto_introspection/proto_introspection_paper_handoff_short_2026-06-17.md`
**Work dated:** 2026-06-17 · **Compiled:** 2026-06-23 · **Scope:** synthesis only — no new experiments, no checkpoint changes, no S3A.

> This is the last synthesis step before drafting. It converts the settled evidence into a precise, paper-ready package a writer (human or model) can hand-execute into a draft. Every number here was verified against the cited artifact JSON/MD on 2026-06-23; nothing is from memory alone. Where an artifact is missing, it is marked `MISSING:` with the consequence.

---

## 1. Executive summary

The paper should claim, in operational terms only, that **looped language-model hidden states expose readable, pre-answer, partly shortcut-independent process-quality signals about the model's own ongoing computation.** The evidence is a multi-pillar bundle on a single frozen model (Ouro-RLTT, a Universal Transformer looped 4× with hidden-state taps at layers 24/36/47). The strongest supporting block is relational: a tiny antisymmetric pairwise readout recovers human-preference ordering at 95.2% and is provably order-sensitive (sign-flip exact-0 by construction, flip-correlation ρ≈−0.94), with a pointwise-linear control collapsing to 21.75% (below chance) — so the signal is genuinely relational, not a pointwise or pair-order artifact. The decisive recent addition is a **strict pre-answer within-domain specificity audit**: on 170 powered GSM8K tasks (680 task-grouped per-sample examples, cut strictly before the gold answer value so it cannot leak), frozen hidden states predict eventual correctness at AUROC 0.745 [0.707, 0.783], and hidden+shortcuts beats the length+logprob shortcut composite by +0.066 (group-bootstrap CI [+0.017, +0.114], significant). This is what upgrades the timing pillar to a genuine pre-answer control. What remains caveated is the standalone-dominance question: hidden-alone (0.745) only **ties** the length+logprob composite (0.731; Δ +0.014, ns) — the hidden signal is significant *incremental* validity, not a knockout of all trivial predictors, and that caveat must be stated plainly, not buried. The signal generalizes in direction to a second (underpowered) reasoning/ARC domain but cross-domain shortcut-free establishment is **not** complete. Crucially, **readout is not control**: every tested frozen-backbone intervention (steering, frozen branch/carry fork) failed to convert the readable signal into reliable write-control or new correct answers (K-matched plain sampling 0.75 ≥ sample-fork 0.611; greedy fork new-correct 0.0), so the paper draws a clean readout/control boundary and explicitly does **not** claim consciousness, self-awareness, autonomous self-control, or any Jormungandr/branching capability gain. The paper is ready to draft now because the one missing decisive control (strict pre-answer specificity in a powered domain) has been run and is positive-leaning-significant; the remaining gaps are scope limitations, not blockers. Drafting should foreground the pre-answer hidden-state *process* signal — not let the result collapse into generic reward modeling or into philosophy.

---

## 2. Final paper thesis

- **Conservative (RECOMMENDED):** *In a frozen looped language model (Ouro-RLTT), intermediate hidden states carry readable, pre-answer process-quality information about the model's own ongoing computation — predicting eventual success in a powered domain and adding statistically significant information beyond length/log-probability shortcuts — but this readable signal does not, on its own, yield autonomous control or capability gains.*
- **Slightly stronger:** *Looped-transformer hidden states expose an operational form of weak proto-introspection: a tiny external readout recovers relational quality ordering (95.2%) and strict pre-answer success signal (AUROC 0.745) that is partly shortcut-independent, establishing that the model's own trajectory states encode their own prospects before external judgment.*
- **Title-safe:** *Readable Process-Quality Signals in Looped-Transformer Hidden States.*
- **Abstract-safe:** *We give an operational definition of weak proto-introspection for looped LMs and show, on frozen Ouro-RLTT, that intermediate hidden states predict the model's own pre-answer success beyond simple shortcuts, while readout does not imply control.*

---

## 3. Contribution list

1. **An operational definition of weak proto-introspection for looped LMs** — readable internal *process-quality* signal about the model's own ongoing computation, defined to exclude mentalistic interpretation.
2. **Relational hidden-state readout evidence in Ouro-RLTT** — a tiny antisymmetric pairwise probe recovers human-preference ordering (95.2%), with antisymmetry/pointwise controls proving the signal is relational and not a pair-order or pointwise artifact.
3. **A strict pre-answer within-domain specificity audit** — leakage-controlled, task-grouped, powered (GSM8K, 170 tasks) evidence that hidden states predict eventual success (0.745) and add *significant incremental* information beyond length+logprob shortcuts (+0.066, CI [+0.017,+0.114]).
4. **Role separation** — validity/survivability (DualAnchor), relational content quality (CoreContent), and external correctness are empirically distinct; tap scores are never correctness labels.
5. **An empirical readout/control boundary** — readable signals do not automatically yield steering or capability gains; every tested frozen intervention failed, motivating training-time integration rather than more inference-time steering.

---

## 4. Claim safety table

| Claim | Status | Evidence | Wording allowed | Wording forbidden |
|---|---|---|---|---|
| Hidden states encode relational preference/quality | **SAFE** | HH pairwise 0.952; pointwise-linear 0.2175; ρ≈−0.94 | "a tiny relational readout recovers human-preference ordering" | "the model knows which answer is better" |
| Pre-answer hidden states predict future success (GSM8K) | **SAFE** | strict cut, AUROC 0.745 [0.707,0.783], grouped CV, 170 tasks | "predict eventual correctness before the answer is emitted" | "predicts its own success" in a mentalistic sense |
| Hidden states add significant information beyond shortcuts | **SAFE_WITH_CAVEAT** | (hidden+all)−composite +0.066 [+0.017,+0.114] SIG | "adds significant *incremental* information beyond length/logprob" | "dominates / beats all trivial predictors" |
| Signal is partly shortcut-independent | **SAFE_WITH_CAVEAT** | incremental gain sig; hidden-alone ties composite (Δ+0.014 ns) | "partly shortcut-independent" | "shortcut-free" |
| Signal is cross-domain | **SAFE_WITH_CAVEAT** | GSM8K powered + ARC underpowered, same direction | "appears in a second domain (underpowered)" | "established cross-domain" |
| Signal is useful | **SAFE_WITH_CAVEAT** | retention 0.9848/1.0; ranking; diagnostics | "useful for external ranking/retention/diagnosis (weak utility)" | "improves task capability" |
| Readout implies control | **UNSAFE** | steering closed; fork 0.0 new-correct; sampling ≥ fork | — | "readable ⇒ controllable / steerable" |
| Proto-introspection means consciousness | **UNSAFE** | n/a (definitional) | "operational, non-psychological" | "consciousness / self-awareness / sentience / mental states" |
| Branching/Jormungandr capability gain | **UNSAFE** | frozen fork reachability-neutral; S3A not run | "mechanism validated; capability not demonstrated" | "branching improves capability" |

---

## 5. Evidence table

| Evidence block | Artifact path | Key result | Supports | Caveat | Paper placement |
|---|---|---|---|---|---|
| Original evaluator / relational preference | `artifacts/reports/evaluator/probe_loop_geometry_hh.json`; `artifacts/checkpoints/evaluator/pairwise_epoch2.pt` | HH pairwise **0.952** (8141/8552); subset thinking 0.95 / RLTT 0.945 | P1, P3 | fixed-order preference; external evaluator ≠ self-knowledge | R1, Fig 2, Table 1 |
| Linear probe & pairwise difference | `artifacts/reports/evaluator/probe_l1_ablation.json`; matrix JSON | pairwise-nonlinear 0.952; pointwise-nonlinear 0.65; **pointwise-linear 0.2175** (below chance); linear-diff probe 0.845; L1-only canonical 0.915 | P1, P3 | generating script `probe_l1_ablation.py` removed from worktree (JSON survives) | R1, §Methods, Appendix |
| Flip / antisymmetry | `docs/evaluator/flip-test-interpretation.md`; s3 closure item D | strict sign-flip exact 0 (by construction); flip-correlation **ρ≈−0.94** | P3 | structural+empirical; not a held-out generalization test | R1, Fig 2 |
| Taps / CoreContent / DualAnchor role separation | `artifacts/reports/probes/mpn_s3_closure_2026-06-17/s3_closure_verdicts.json`; `docs/evaluator/content-selection-taps.md`; `docs/evaluator/dualanchor-architecture-baseline.md` | CoreContent in-dist **0.6691** [0.6448,0.6902]; DualAnchor survival retention **0.9848** / terminal **1.0** / forced-top1 0.379 | P4, P5 | forced-top1 is diagnostic; survivor-retention is the headline | R5, Table 1 |
| Trajectory / prefix prediction | `artifacts/reports/probes/bg_trajectory_prediction_2026-05-18/summary.json` + `predictive_power.md` | best cell reasoning@256: pairwise **0.8536**, top1-lift **+0.1625** [0.0625,0.2625], oracle 0.90 | P1, P2 | **LEAKAGE: 256-prefix 95% answer-leaked; task-grouped AUROC drops to 0.624; clean pre-answer 0.690.** Do NOT use as the clean headline | R2 (caveated), Appendix |
| Strict pre-answer GSM8K audit | `artifacts/reports/proto_introspection/proto_introspection_within_domain_preanswer_specificity_2026-06-17.{md,json}`; `within_domain_recapture.pt` | hidden **0.745** [0.707,0.783]; length 0.687; logprob 0.569; len+logprob 0.731; hidden+all 0.797; **incremental +0.066 [+0.017,+0.114] SIG** | P1, P2, P3 | hidden-alone ties composite (Δ+0.014 ns); one powered domain | R3, R4, Fig 3, Table 3 |
| Reasoning/ARC secondary audit | same JSON, `reasoning` block | hidden 0.690 [0.634,0.741]; len+logprob 0.597; incremental +0.093 [+0.037,+0.150] SIG | P1, P6 | **UNDERPOWERED (105<150); pre-answer front-loaded, median 14 tok** | R3 (secondary), Table 3 |
| Steering / control failure | `docs/evaluator/steering-and-adapters.md`; `mpn_s1_baseline_2026-06-13/s1_4b_kmatched_sampling.json` | frozen steering CLOSED_UNDER_TESTED_METHODS; K-matched plain sampling **0.75 ≥** sample-fork 0.611 (Δ−0.139); greedy-fork new-correct **0.0** | P7 | bounded negative under tested methods, not impossibility | R6, Fig 4, Table 4 |
| S1/S3 branch closure | `mpn_s3_closure_2026-06-17/s3_closure_and_sokac_validity_bundle.md` + `s3_closure_verdicts.json` | mechanism validated; frozen branching reachability-neutral; S3A cloud-only, not run | P4, P5, P7 | engineering background/limitation, not the proto-introspection proof | R6, Appendix, Limitations |

---

## 6. Operational definition section (draft text)

> **"Proto-introspection" is used operationally, not psychologically.** An intermediate model state is *proto-introspective* if it contains readable information about the quality, stability, uncertainty, likely success/failure, or branch-viability of the model's own ongoing computation, recoverable by a small external readout *before* external final judgment.

**What this does imply:**
- The model's own intermediate trajectory states encode their own prospects.
- A small, frozen-backbone readout can recover that information.
- The information is present before the final answer/token and is not reducible to the final external label.

**What this does NOT imply:**
- Not consciousness, sentience, subjective awareness, or mental states.
- Not verbal self-report or that the model "understands itself."
- Not that the model autonomously *uses* the signal to control itself (readout ≠ control).
- Not generic answer correctness reducible to trivial output statistics (we test against shortcut baselines).

---

## 7. Methods map

- **Backbone:** frozen Ouro-RLTT (`models/ouro_rltt_local`) — Ouro-2.6B-Thinking + RLTT, 48 layers, Universal-Transformer recurrence with `total_ut_steps=4`, `early_exit_threshold=1.0` (`force_all_loops=True`), hidden 2048. No checkpoint modification anywhere in the paper.
- **Looped hidden states:** taps at layers **{24, 36, 47}** across **4 UT loops** → pooled feature tensor **[3 layers × 4 loops × 2048]** per text.
- **Hidden-state extraction:** `BGTransformerFeatureExtractor.encode_text_to_pooled_features` (`src/evaluator/bg_transformer_features.py`).
- **Pairwise difference readouts:** antisymmetric `layernorm(i−j)` evaluator; sign-flip exact-0 by construction; trained head `pairwise_epoch2.pt`.
- **Linear probes:** standardize → PCA(24) → L2 logistic regression (LBFGS), pure-torch.
- **Evaluator:** HH human-preference pairwise head, reported at full-test and thinking/RLTT subsets.
- **Tiny taps:** DualAnchor (validity/survivability, 6ch), CoreContent (relational content quality, 3ch) — read hidden states, never raw text, never used as correctness labels.
- **Strict pre-answer probe:** per sample, cut text at `min(FINAL-ANSWER marker, first occurrence of gold answer value)` so the gold value cannot leak; capture pre-answer + prompt-only `[3,4,2048]`; K=4 externally-verified samples per task.
- **Shortcut baselines:** random; length metadata (question chars, prompt tokens, pre-answer tokens); logprob/entropy (teacher-forced mean logprob + mean/last entropy); length+logprob composite; hidden; hidden+all. Combined model PCA's the hidden block but passes shortcut dims through un-PCA'd.
- **Grouped / task-heldout evaluation:** 5-fold **task-grouped** CV (group = task_id); no task's samples split across folds; AUROC (Mann–Whitney tie-corrected); bootstrap CIs; **group-bootstrap paired delta CIs** (resampling tasks — the honest unit).

---

## 8. Main results plan (presentation order)

- **R1 — Relational hidden-state quality signal.** Evaluator pairwise 0.952; pointwise-linear control 0.2175; antisymmetry ρ≈−0.94. Establishes a readable, relational, order-sensitive quality signal.
- **R2 — Intermediate/prefix trajectory signal (CAVEATED).** Prefix→branch-success 0.8536 at reasoning@256, **but** flag leakage (95% answer-leaked at 256; grouped AUROC 0.624). Present only as motivation, with the leakage correction inline; clean pre-answer signal survives at 0.690.
- **R3 — Strict pre-answer process-quality signal (CLEANEST RESULT).** GSM8K powered: hidden 0.745 [0.707,0.783] at a leakage-controlled pre-answer cut (median 163 reasoning tokens). Secondary ARC underpowered but same direction.
- **R4 — Specificity.** Hidden+all beats length+logprob composite by +0.066 [+0.017,+0.114] (SIG); but hidden-alone (0.745) ties the composite (0.731; Δ+0.014 ns). Logprob alone weak (0.569) ⇒ signal lives in the representation, not output confidence. State both the significant incremental gain and the tie, equally prominently.
- **R5 — Role separation.** Validity (DualAnchor retention 0.9848/1.0) vs content (CoreContent in-dist 0.6691) vs correctness (external verifier) are distinct; generated-branch correctness transfer fails (0.417 < random).
- **R6 — Readout/control boundary.** Steering closed under tested methods; K-matched sampling 0.75 ≥ fork 0.611; greedy-fork new-correct 0.0 ⇒ readable signal is not sufficient for autonomous control; motivates training-time branch-tournament RLTT (S3A), not more frozen steering.

---

## 9. Figure & table plan

- **Figure 1 — Conceptual diagram:** looped Universal Transformer, taps at L24/36/47 × 4 loops, external readout recovering process-quality before final judgment. *Source:* schematic from §7 methods map + model config in s3 closure JSON.
- **Figure 2 — Pairwise hidden-state difference & antisymmetric readout:** 0.952 pairwise vs 0.2175 pointwise-linear; ρ≈−0.94 flip behaviour. *Source:* `probe_loop_geometry_hh.json`, `flip-test-interpretation.md`, matrix JSON P1/P3.
- **Figure 3 — Strict pre-answer AUROC bars:** hidden / length / logprob / length+logprob / hidden+all, with hidden CI; GSM8K (powered) and ARC (underpowered, flagged). *Source:* within-domain specificity JSON `results`.
- **Figure 4 — Readout vs control boundary:** readout successes (0.952, 0.745, retention 0.9848) vs control failures (fork new-correct 0.0, sampling ≥ fork, steering closed). *Source:* within-domain JSON + s3 closure JSON `key_metrics`.
- **Table 1 — Evidence pillar matrix:** P1–P7 with status and headline number. *Source:* matrix JSON `pillars` updated by within-domain JSON `pillar_status_updated`.
- **Table 2 — Claim safety table:** §4 verbatim. *Source:* this package.
- **Table 3 — Strict pre-answer audit metrics:** full per-domain AUROCs, paired deltas, CIs, n, base rate, pre-answer token distribution. *Source:* within-domain specificity JSON.
- **Table 4 — Limitations & forbidden interpretations:** §14 + §4 UNSAFE rows. *Source:* this package + s3 closure `hard_rules_honored`.

---

## 10. Abstract candidates

**Conservative.**
> Whether a language model's intermediate computation carries readable information about its own prospects is usually studied only through final outputs. We give an *operational* (non-psychological) definition of weak proto-introspection for looped transformers and test it on a frozen Ouro-RLTT model whose Universal-Transformer loops expose intermediate hidden states. A tiny relational readout recovers human-preference ordering at 95.2%, and — at a strict pre-answer cut that excludes the gold answer — frozen hidden states predict eventual GSM8K success at AUROC 0.745, adding statistically significant information beyond length and log-probability shortcuts (incremental +0.066, CI [+0.017,+0.114]). The hidden signal does not dominate a strong length+log-probability composite on its own, and no tested frozen intervention converts the readable signal into control. We therefore claim only that looped-model hidden states expose readable process-quality signals about the model's own ongoing computation — an operational result, not a claim about consciousness, self-awareness, or autonomous control.

**Balanced.**
> Looped transformers recompute hidden states across recurrent steps, raising the question of whether those states carry readable information about the quality of the model's own ongoing computation before any answer is produced. On frozen Ouro-RLTT we show a small antisymmetric readout recovers relational quality ordering (95.2%; pointwise-linear control 21.75%, below chance; flip-correlation ρ≈−0.94), and a strict, leakage-controlled, task-grouped pre-answer probe predicts GSM8K success at AUROC 0.745 [0.707,0.783] over 170 powered tasks, with a significant incremental gain over shortcut baselines. A second reasoning domain corroborates the direction (underpowered). We separate validity, content, correctness, and control, and show that readable signal does not yield steering or capability gains. We frame this as an *operational* weak proto-introspection result — readable internal process-quality information — explicitly not consciousness or self-control.

**Ambitious but safe.**
> Do looped language models carry readable, pre-answer evidence about whether their own computation is going to succeed? Using frozen Ouro-RLTT, we operationally define and test weak proto-introspection. Intermediate hidden states support relational quality readout at 95.2% and predict eventual success at a strict pre-answer cut (AUROC 0.745) with statistically significant information beyond length/log-probability shortcuts. Yet the same readable states resist every frozen intervention we test, establishing a sharp readout/control boundary. The result is operational and empirical — readable process-quality signal about the model's own ongoing computation — and we explicitly disclaim any mentalistic interpretation: no consciousness, no self-awareness, no autonomous self-control, no capability gain from branching.

---

## 11. Title candidates

*Conservative:*
1. Readable Process-Quality Signals in Looped-Transformer Hidden States
2. Pre-Answer Hidden States Predict Success in a Frozen Looped Language Model
3. An Operational Audit of Weak Proto-Introspection in Ouro-RLTT
4. Hidden-State Readouts of a Looped Model's Own Ongoing Computation
5. What Looped Hidden States Reveal Before the Answer: A Specificity Audit

*Balanced:*
6. Readout, Not Control: Process-Quality Signals in Looped Transformers
7. Weak Proto-Introspection in Looped Language Models: Evidence and Boundaries
8. Predicting a Model's Own Success From Its Intermediate Hidden States

*Ambitious but safe:*
9. The Model Knows Its Work (Operationally): Pre-Answer Process Signals in Looped Transformers
10. Looking Inward Without Looking Conscious: An Operational Proto-Introspection Study

**Recommended:** Title 1 (conservative, matches the recommended thesis).

---

## 12. Introduction argument (paragraph-by-paragraph)

1. **Why looped transformers matter.** Universal-Transformer / latent-reasoning models recompute hidden states across recurrent steps; those steps are a natural place for *process* information to live, distinct from the final token.
2. **Why hidden states may contain process signals.** If recurrence does iterative refinement, intermediate states may encode how well the computation is going — uncertainty, stability, likely success — before the answer is emitted.
3. **Why "introspection" needs an operational definition.** The word invites mentalistic overreach. We define it operationally (readable process-quality signal recoverable by an external probe before final judgment) and bound what it does/does not mean (§2 / §6).
4. **What Ouro-RLTT allows us to test.** A frozen looped model with explicit tap points (L24/36/47 × 4 loops) lets us read intermediate states without training the backbone, and an external verifier gives clean correctness labels.
5. **What we find.** Relational quality readout (95.2%); strict pre-answer success prediction (0.745) with significant incremental validity over shortcuts; role separation; and a hard readout/control boundary.
6. **What we do not claim.** Not consciousness/self-awareness/control/capability; not shortcut-domination (hidden-alone ties the composite); not established cross-domain. State the caveats up front.

---

## 13. Related-work buckets (no citations invented — buckets only)

- Looped transformers / latent reasoning / RLTT (Universal Transformers, recurrent-depth reasoning, reasoning-as-test-time-training).
- Probing hidden states (linear probes, diagnostic classifiers, what-do-representations-encode).
- Preference modeling and reward models (pairwise preference, RLHF reward heads).
- Mechanistic interpretability / representation geometry (directions, antisymmetry, concept readouts).
- Uncertainty / confidence estimation (logprob/entropy calibration, self-consistency confidence).
- Process supervision (process reward models, step-level correctness).
- Tree-of-thought / external search vs internal branching (search over reasoning vs in-model branch carry).
- Steering / activation intervention limits (activation addition, representation engineering, and their failure modes).

> Do not invent citations; list the bucket and the likely citation need only.

---

## 14. Limitations section (draft list)

- Primary specificity result rests on **one powered strict-pre-answer domain** (GSM8K, 170 tasks).
- Secondary reasoning/ARC domain is **UNDERPOWERED** (105<150) and pre-answer front-loaded (median 14 tokens), so it corroborates direction only.
- **A second powered domain was attempted and rejected at pre-flight** (2026-06-23; `proto_introspection_second_domain_preflight_2026-06-17.md`). SVAMP front-loads (pre-answer median 3 tokens) and the model's `\boxed{}` output mismatched the GSM8K verifier; MATH (L1-3) forces real reasoning but the model's failure mode is **truncation, not wrong answers**, so parseable samples are ~100% correct (no negative class). A clean strict-pre-answer domain needs *both* answer-last reasoning *and* a balanced commit/fail mix — GSM8K hits this window; trivial domains front-load and hard domains truncate. The cross-domain gap is therefore a measured property of the model, not an untested hole.
- Hidden-alone (0.745) **does not dominate** the length+logprob composite (0.731; Δ+0.014, ns); the win is significant *incremental* validity, not standalone dominance.
- Shortcut independence is **partial**, not complete.
- Cross-domain shortcut-free proof is **not** complete.
- Readout is **not causal control**; the control boundary is established, not crossed.
- No consciousness/self-awareness/sentience/mental-state claim (definitional exclusion).
- No Jormungandr/branching capability improvement (frozen branching reachability-neutral; fork new-correct 0.0).
- **S3A (training-time branch-tournament RLTT) was not run** (cloud/backbone scale).
- Frozen branch/carry is a **negative result** (K-matched sampling ≥ fork); reported, not hidden.
- Generating scripts for two L1 probe JSONs were removed from the worktree (`MISSING:` below); result JSONs survive, script reproducibility weakened.
- Evaluator preference is fixed-order human-preference data; external evaluator capability is not equated with model self-knowledge.

---

## 15. Discussion framing

- **Why the result is still interesting despite caveats:** it isolates a *pre-answer*, leakage-controlled, task-grouped signal that adds significant information beyond the obvious shortcuts — a cleaner test than prior trajectory work that was leakage-inflated.
- **Why logprob being weak (0.569) matters:** the signal is in the *representation*, not output confidence; this is the difference between proto-introspection and calibration.
- **Why hidden+all (0.797) improvement matters:** complementary information — hidden states carry something the shortcuts do not, even if shortcuts are individually strong.
- **Why the readout/control separation is scientifically useful:** it cleanly factors "the information exists" from "the model can act on it," preventing the common overclaim that readable ⇒ steerable.
- **Why this motivates training-time branch-tournament RLTT (S3A) rather than more frozen steering:** frozen interventions are closed under tested methods; the lever is training-time loop-dynamics integration, which is the honest next experiment.

---

## 16. Paper outline

- **Abstract** — conservative candidate (§10); explicit operational disclaimer.
- **1 Introduction** — §12 paragraphs 1–6; preview numbers 0.952, 0.745, +0.066; state non-claims.
- **2 Operational Definition** — §6 verbatim; does/does-not-imply.
- **3 Background: Ouro-RLTT and Hidden-State Readouts** — model config (s3 closure JSON), tap points, pairwise evaluator, taps.
- **4 Methods** — §7 methods map; strict pre-answer cut construction; shortcut baselines; task-grouped CV + group-bootstrap deltas.
- **5 Results** — R1–R5 (§8). Fig 2, Fig 3, Table 1, Table 3.
- **6 Readout vs Control** — R6; steering/fork failures; Fig 4, Table 4.
- **7 Limitations** — §14.
- **8 Discussion** — §15.
- **9 Conclusion** — restate conservative thesis + readout/control boundary + next step (training-time integration).
- **Appendices** — §17.

---

## 17. Appendix plan

- **A — Evaluator architecture / linear-probe details:** antisymmetric `layernorm(i−j)`, `pairwise_epoch2.pt`, PCA(24)+L2-logreg pipeline.
- **B — Tap definitions:** DualAnchor (6ch, validity), CoreContent (3ch, content); never correctness labels.
- **C — Shortcut-baseline details:** length features, teacher-forced logprob/entropy, composite construction, combined-model PCA handling.
- **D — Strict pre-answer cut construction:** `min(FINAL-marker, gold-value occurrence)`; K=4 verified samples; per-sample vs prompt-only labeling.
- **E — Branch/S3 engineering closure:** S1 mechanism gates, frozen reachability-neutral, K-matched sampling deconfound, S3A design (not run).
- **F — Additional negative results:** trajectory leakage correction (0.854→0.624 grouped), steering closure, generated-branch correctness transfer fail.
- **G — Artifact / reproducibility index:** §20.

---

## 18. Canonical wording bank (reusable sentences)

- **Operational definition:** "We use *proto-introspection* operationally, not psychologically: an intermediate state is proto-introspective if it carries readable information about the quality, stability, uncertainty, or likely success of the model's own ongoing computation, recoverable by a small external readout before external final judgment."
- **Weak claim:** "Looped-model hidden states expose readable process-quality signals about the model's own ongoing computation."
- **Pre-answer result:** "At a strict pre-answer cut that excludes the gold answer value, frozen hidden states predict eventual GSM8K success at AUROC 0.745 [0.707, 0.783] under task-grouped cross-validation over 170 powered tasks."
- **Shortcut caveat:** "Hidden states add statistically significant *incremental* information beyond length and log-probability shortcuts (+0.066, CI [+0.017, +0.114]); on their own they tie, but do not beat, a strong length+log-probability composite (Δ +0.014, ns)."
- **Readout/control boundary:** "These signals are readable, but no tested frozen-backbone intervention converted them into reliable control or new correct answers; readout and autonomous control are distinct problems."
- **No-consciousness disclaimer:** "This is an operational, empirical result about readable internal information; we make no claim about consciousness, self-awareness, sentience, mental states, or verbal self-understanding."
- **Jormungandr limitation:** "Frozen internal branching is reachability-neutral under the tested regimes and does not demonstrate any capability improvement; the branching mechanism is validated but not shown to add capability."
- **Future work:** "The readout/control gap motivates training-time branch-tournament RLTT (S3A), a cloud/backbone-scale experiment, rather than further inference-time steering."

---

## 19. Final paper readiness verdict

```
PAPER_READINESS_VERDICT           = READY_TO_DRAFT_WITH_CAVEATS
PROTO_INTROSPECTION_CLAIM_SCOPE   = NARROW_OPERATIONAL_WEAK_FORM
TIMING_VERDICT                    = STRICT_PREANSWER_SIGNAL_SUPPORTED
SPECIFICITY_VERDICT               = PARTLY_SHORTCUT_INDEPENDENT_SIGNIFICANT_INCREMENTAL_SIGNAL
CROSS_DOMAIN_VERDICT              = PARTIAL_NOT_FULLY_ESTABLISHED
UTILITY_VERDICT                   = WEAK_READOUT_UTILITY_SUPPORTED_STRONG_CONTROL_NOT_PROVEN
CONTROL_VERDICT                   = READOUT_CONTROL_BOUNDARY_ESTABLISHED
BRANCHING_VERDICT                 = MECHANISM_VALIDATED_CAPABILITY_NOT_PROVEN
NEXT_STEP_VERDICT                 = BEGIN_PAPER_DRAFT
```

---

## 20. Artifact index

**Present and load-bearing:**
- `artifacts/reports/proto_introspection/proto_introspection_evidence_matrix_2026-06-17.{md,json}` — pillar matrix, operational definition.
- `artifacts/reports/proto_introspection/proto_introspection_within_domain_preanswer_specificity_2026-06-17.{md,json}` — strict pre-answer audit (primary specificity result).
- `artifacts/reports/proto_introspection/within_domain_recapture.pt` (+ `within_domain_recapture_index.json`) — GSM8K 170 + ARC 105 recapture features.
- `artifacts/reports/proto_introspection/within_domain_specificity_results.json` — analysis output.
- `artifacts/reports/proto_introspection/proto_introspection_controls_2026-06-17.{md,json}` + `preanswer_recapture.pt` (+ index) + `controls_analysis_results.json` + `proto_introspection_final_readiness_2026-06-17.md` — Phase-1 controls (leakage correction).
- `artifacts/reports/probes/mpn_s3_closure_2026-06-17/s3_closure_and_sokac_validity_bundle.md` + `s3_closure_verdicts.json` — engineering closure, role separation, control boundary, S3A design.
- `artifacts/reports/evaluator/probe_loop_geometry_hh.json` — preference subset accuracies (thinking 0.95 / RLTT 0.945).
- `artifacts/checkpoints/evaluator/pairwise_epoch2.pt` — evaluator head.
- `artifacts/reports/probes/bg_trajectory_prediction_2026-05-18/summary.json` + `predictive_power.md` — trajectory headline (caveated; leakage-corrected).
- `artifacts/reports/probes/mpn_s1_baseline_2026-06-13/s1_4b_kmatched_sampling.json` — sampling deconfound / fork negative result.
- `docs/evaluator/flip-test-interpretation.md`, `docs/evaluator/steering-and-adapters.md`, `docs/evaluator/dualanchor-architecture-baseline.md`, `docs/evaluator/content-selection-taps.md` — methods/role-separation docs.
- Scripts: `utilities/tests/manual/proto_introspection_within_domain_recapture.py`, `proto_introspection_within_domain_analysis.py`, `proto_introspection_preanswer_recapture.py`, `proto_introspection_controls_analysis.py`; `src/evaluator/bg_transformer_features.py`.

**Present and load-bearing (negative finding):**
- `artifacts/reports/proto_introspection/proto_introspection_second_domain_preflight_2026-06-17.md` — documents the rejected second-domain attempt (SVAMP front-loading + verifier mismatch; MATH truncation-failure selection → degenerate base rate). Script kept: `utilities/tests/manual/proto_introspection_math_recapture.py` (hardened MATH recapture).

**Missing (with consequence):**
- `MISSING: utilities/tests/manual/probe_l1_ablation.py and probe_l1_alpha_sweep.py` — consequence: the result JSONs (`probe_l1_ablation.json`, `probe_l1_alpha_sweep.json`) survive and are load-bearing, but the generating scripts were removed from the worktree; the 0.2175 pointwise-linear and L1-only 0.915 numbers stand, script reproducibility is weakened.
- `MISSING: second powered strict-pre-answer domain` (ARC reached 105<150, front-loaded; a fresh attempt via SVAMP and MATH was rejected at pre-flight 2026-06-23 — see negative-finding report above) — consequence: powered specificity rests on one domain; CROSS_DOMAIN_VERDICT stays PARTIAL, now with a measured explanation of why a clean second domain is hard for this model.
- `MISSING: Barbados loop-state success/failure predictability` — consequence: no small-looped-transformer cross-check; optional, must NOT be invented.
- `MISSING: artifacts/logs/mpn_s3b/` — consequence: S3B-0/S3B-1 have no dedicated run log (JSONs are the record); low impact.
- `MISSING: offsite/cloud copy of S0 backup` — consequence: single-disk failure risk for the pre-S3A weight backup; weakens S3A safety precondition, not a paper claim.

---

### Honesty ledger (do not bury)
1. The trajectory **0.854** headline was leakage+memorization-inflated (256-prefix 95% answer-leaked; task-grouped AUROC 0.624). The clean, defensible number is the strict pre-answer GSM8K **0.745**.
2. Hidden-alone **ties** the length+logprob composite (Δ +0.014, ns). The significant result is the **incremental** gain of hidden+all over the composite (+0.066, CI [+0.017,+0.114]).
3. The powered specificity result is **one domain**; the ARC secondary is **underpowered**, and a fresh second-domain attempt (SVAMP, MATH) was **rejected at pre-flight** for principled model-specific reasons (front-loading / truncation-failure selection) — documented, not silently dropped.
4. Frozen branching/steering produced **no** control and **no** capability gain — reported, not hidden.
