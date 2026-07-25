# Research basis and live design ledger

This rebuild was designed after reading the complete Erratum and the full
`ouro_paper_draft_v6_4(1).md`, then comparing their corrected claims with
current primary research and 2026 ARC-AGI-3 systems.  The design is not frozen
to either paper.  A mechanism stays only while it has an observable target, a
causal route to behavior, and an ablation that justifies its complexity.

## Corrected local evidence

The governing local sources are:

- `/home/moloch/Documents/Research/kirin2026_erratum/paper.tex`
- `/home/moloch/Documents/Research/ouro_paper_draft_v6_4(1).md`

They change Hunter-Seeker's premise in four decisive ways:

1. The fixed-order 95.2% evaluator is not a valid universal reward or encoder
   anchor.  Strict antisymmetrized accuracy is 0.6392.
2. Pointwise preference signal is above chance, while the clean relational
   advantage is modest.  Pointwise progress, hazard, and value heads are
   therefore legitimate.
3. Role-specialized taps can read useful process information, but readout,
   branch survival, final selection, and steering are distinct capabilities.
4. External outcomes and verifiers remain authoritative.  A readable hidden
   direction is not evidence that the base model uses it or can be controlled
   through it.

The v2 consequence is simple: taps are typed sensors; the environment-trained
controller owns actions; the legacy CLT anchor is absent.

## Research that changed the implementation

| Source | Mechanism used in v2 | What was not copied |
|---|---|---|
| [ARC-AGI-3 technical report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf) | Hash-addressed directed state graph, explicit terminals/no-change edges, cycle and merge handling, action efficiency as a first-class cost. | The benchmark's privileged hidden environment state; v2 hashes observations and learned state only. |
| [ARC-AGI-3 preview learnings](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings) | Prune known loops/no-change actions and back-label distance after observed progress. | Large brute-force action budgets. |
| [ARC Prize 2026 Milestone 1](https://arcprize.org/blog/arc-prize-2026-milestone-1) | Lightweight generic harness, legal-action guards, explicit feature toggles, and executable hypotheses. | Winner-specific prompts, game heuristics, or model assumptions. |
| [Executable World Models for ARC-AGI-3](https://arxiv.org/abs/2605.05138) and its [released implementation](https://github.com/astroseger/arc-3-agents-baseline1) | Optional rule-model registry with replay verification, coverage/error/complexity ranking, and a hard planning gate. | An LLM coding-agent dependency in the deterministic core. |
| [DINO-WM](https://proceedings.mlr.press/v267/zhou25t.html) | Frozen spatial/token features and decoder-optional latent dynamics. | Goal-image planning and the DINOv2 dependency. |
| [Rewarding Latent Thought Trajectories](https://arxiv.org/abs/2602.10520) (Williams & Tureci) | The student API accepts normalized representation weights, applies the same teacher label or absolute online outcome target to every nonzero-weight representation, and designates one terminal representation for scoring. This is an interface boundary for a future action-policy analogue, not an implementation of the paper's objective. | Calling a one-representation GridFeature head “RLTT,” treating a hidden-state tap as RLTT training, or sampling actions from every loop. In the paper the terminal loop samples the output while one group-normalized outcome advantage trains every loop distribution; terminal KL remains terminal-loop only. |
| [Objects Matter / OC-STORM](https://arxiv.org/abs/2501.16443) | Explicit small-object branch alongside global features; connected components are first-class dynamics inputs. | Prompted video segmentation, because ARC already supplies categorical grids. |
| [Efficient Exploration with an Object-Centric Abstraction](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1f8b45077e53a39a8b657101bb1a09b4-Abstract-Conference.html) | Object-level state abstraction and action-effect learning for long-horizon exploration. | Domain-specific crafting/item semantics. |
| [COPlanner](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4a32a646254d2e37fc74a38d65796552-Abstract-Conference.html) | The same epistemic uncertainty is a bonus for safe real probes and a penalty in imagined rollouts. | Its continuous-control policy-learning stack. |
| [Learning to Explore in POMDPs with Informational Rewards](https://proceedings.mlr.press/v235/xie24a.html) | Mechanic/affordance discovery is a learnable information target, with hindsight-compatible event records. | A separately trained information-reward network in the first implementation. |
| [Plan2Explore](https://proceedings.mlr.press/v119/sekar20a.html) | Bootstrap disagreement as a practical epistemic estimate. | Reward-free pretraining as a required phase. |
| [Beyond Noisy-TVs: Learning Progress Monitoring](https://openreview.net/forum?id=wzm38DRLhC) | Expected model improvement and realized learning progress temper raw novelty. | Treating prediction error itself as intrinsic reward. |
| [SafeDreamer](https://proceedings.iclr.cc/paper_files/paper/2024/hash/ece182f93af26c64187ba3f7dfd4309a-Abstract-Conference.html) | Separate progress/value and hazard cost, followed by one risk constraint. | A physical-safety interpretation or fixed near-zero risk tolerance. |
| [MrSteve](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2af7168a1f19e0ae61134f89eb238e57-Abstract-Conference.html) | One structured what/where/when event memory with derived retrieval views. | Minecraft navigation and language-memory machinery. |
| [Chunked-TD](https://proceedings.mlr.press/v235/ramesh24b.html) | Kept as a future optimization for predictable trajectory chunks. | It is not used to conceal missing terminal transitions; transaction correctness came first. |
| [Practice Makes Perfect](https://arxiv.org/abs/2402.15025) | Operational competence: model error, calibration, success estimate, learning gain, stagnation, and risk budget. | Anthropomorphic or consciousness claims. |
| [Causal Influence Detection](https://arxiv.org/abs/2106.03443) (Seitzer et al.) | Control attribution (`ego.py`): a track's influence score is the count-based normalized mutual information between the executed action and its quantized displacement response — the discrete analog of their conditional-mutual-information controllability measure. | The learned network estimator and continuous-control integration; categorical tracked objects admit an exact plug-in estimate. |
| [Contingency-Aware Exploration](https://openreview.net/forum?id=HyxGB2AcY7) (Choi et al.) | Locating the controllable observation element as a first-class, compact target that policy may consume. | The attentive dynamics model and pixel-region formulation; v2 already has object tracks. |
| [AXIOM](https://arxiv.org/abs/2505.24784) | Gradient-free count-based object models; action-conditioned per-object response discovery; outcome attribution through object interaction/adjacency identity (the ego-protected contact-hazard route). | The full Bayesian mixture machinery and pixel-level generative model. |
| Exogenous-state decomposition ([Dietterich et al., ICML 2018](https://arxiv.org/abs/1806.01584); [Efroni et al., ICLR 2022](https://arxiv.org/abs/2110.08847); [Lamb et al., 2022](https://arxiv.org/abs/2207.08229)) | Memory identity filtering (`exogenous.py`): observation cells whose change trajectories replay identically across episodes with differing action histories are exogenous and excluded from the durable-memory state hash.  Different action histories are the intervention that certifies action-independence. | Learned inverse-dynamics encoders and latent decompositions; categorical cells admit a direct event-trajectory comparison.  Transactional identity is never filtered. |
| Potential-based reward shaping (Ng, Harada & Russell, ICML 1999) | Goal hypotheses (`hypotheses.py`) influence policy only through potential *differences* of candidate relations (equality, mirror, uniformity, lattice-canonical equality), keeping the pressure bounded and removable. | Assuming the potential is the true value function; unverified hypotheses carry exploration-grade weight only, and promotion requires an observed completion at satisfied potential. |

## Current synthesis

The smallest defensible system found by this review is:

```text
categorical observation
  -> exact state graph
  -> connected objects/tracks/events/topology
  -> frozen grid or Ouro multi-tap representation
  -> small bootstrap action-conditioned dynamics ensemble
  -> bounded receding-horizon search
  -> state-conditioned student preferences
  -> explicit score terms
  -> one risk arbiter
  -> action
```

One append-only transition ledger supplies replay, terminal evidence, learned
affordances, inverse-effect diagnostics, route fragments, and reports.  It
replaces memories that previously stored overlapping, inconsistently scoped
versions of the same transition.

## Applied example: relational goal hypotheses (2026-07-17)

Motivating observation: the 25-game render survey found the uniform blocker
is goal inference — nearly every game displays its goal as an on-screen
relation (template, legend, key, symmetry, recipe) and the agent has no
mechanism to represent one.

1. Observed target: hypothesis potentials are computed from committed
   frames; the policy signal is built from *realized* potential deltas per
   action, and promotion/refutation from actual completion events.
2. Nonzero updates: proposals form on structural relations (including the
   lattice-canonical template verified to detect sc25's true key/board pair
   at Φ=0.222 despite a 2.1× scale and different palettes) and delta counts
   grow on synthetic fix/break transitions.
3. Decision route: `test_hypothesis_term_flips_the_selected_action` flips
   the selected action through the named bounded `hypothesis_potential`
   term against opposing prior pressure.
4. Matched-compute autonomous comparison:
   `utilities/tests/manual/run_hs_v2_hypotheses_onoff_v1.py`; results in
   `CAPABILITY_PARITY.md` — trajectories diverge but no completions yet.
5. Scope: hypotheses and their delta statistics are per (task, stage);
   nothing transfers across tasks.
6. Not a pairwise reader; not applicable.
7. Simpler mechanism check: the evidence ledger scores past outcomes, not
   counterfactual goal relations; the executable registry demands exact
   successor-hash prediction (unworkable under environment clocks without a
   frame-prediction stack); no existing component proposes goal candidates
   at all.  Epistemic guardrails: unverified hypotheses carry
   exploration-grade weight (`unverified_scale`), promotion requires a real
   completion at satisfied potential, contradiction refutes, and
   `enabled=False` is a tested exact no-op.

## Implemented: state-conditioned student policy (gate 3 follow-up)

The first retention experiment exposed a structural frequency prior: its
distilled arm could reproduce a dominant action but could not represent the
state-to-action route.  `student.py` now supplies a candidate-conditioned
policy function over the current representation.  Offline demonstrations are
balanced by action index to counter collapse onto the dominant action; a
deterministic nonlinear feature map consumes flattened global and spatial
features without storing teacher state ids.  Crafted tests recover minority
turns, but neither the balancing nor the feature map guarantees that every
route state is separable.  The feature map's `state_feature_scale` is an RBF
bandwidth and must track the latent inter-state distance scale: the original
value of 80 sat two orders of magnitude past the measured ls20 geometry,
reducing the head to a hash table (v2 experiment's near-zero retention); the
default is now 8, which produced the first teacher-free level completions
(v3, 2026-07-19).  Because the right bandwidth is game-geometry dependent
(tr87's adjacent latent distance is 0.026 versus ls20's 0.13), distillation
now fits a per-task scale by anchoring the kernel argument at a low quantile
of latent distances between differently-labeled demonstration states
(`_fit_task_scale`; fitted ls20 9.6, tr87 63.6, wa30 5.2 — recovering each
game's empirically-best fixed scale from geometry alone).  Two further
retention defects are closed at the same layer: committed online outcomes
are weighted by outcome-target magnitude, so neutral steps (target zero) no
longer erode distilled margins by regressing trained rows toward zero, and
`TrajectoryTeacher.from_npz` always declares an action inventory (explicit
or inferred from the demonstrator's used actions) so distillation's
alternative set covers every action — an action absent from that set
received no negative updates and the head provably collapsed onto it
(wa30's action 5: 158 positive examples, zero negative, won every argmax).
Committed online outcomes otherwise continue updating the
same task-local rows, and only explicit positive rows may transfer across
tasks.  Runtime influence is one named `student_policy` term (a bounded raw
head score multiplied by a bounded runtime weight) followed by the ordinary
risk arbiter, and action selection has no teacher reference or lookup path.

The interface also accepts `StudentRepresentationTrajectory`: teacher labels
and realized outcome credit update every normalized representation weight,
while scoring reads the designated terminal representation.  With the current
agent integration this is state-conditioned distillation, **not RLTT**: both
GridFeature and the Ouro connector supply one `Representation`, and only direct
student API tests exercise multiple representations.  Wiring loop states into
that API would still be insufficient for an RLTT claim, which requires the
paper's policy-gradient semantics and per-loop action distributions.

## Ouro loop-state lanes after the full RLTT-paper audit (2026-07-18)

Ranked plan for exploiting the frozen Ouro looped backbone after reading the
complete Williams–Tureci paper, including its appendices and qualitative
examples.  The GridFeature student gate is now implemented.  The next three
lanes remain frozen-backbone plus small probes; full-parameter RLTT training
remains a separate experiment rather than being smuggled in as readout work.

1. **Loop-convergence sensors (gate 4 first).**  The weight-tied loop
   yields an iterate sequence per state; the convergence shape
   (`‖late−middle‖ / ‖middle−early‖`) becomes an UNCERTAINTY-role
   calibrated tap, and small linear value/hazard probes trained on the
   agent's own transition ledger fill the pointwise tap slot.  Evaluated
   through the GridFeature-versus-Ouro-taps ablation.
2. **Tap latents as dynamics features.**  Affordable only through the
   exogenous-identity cache: tap outputs are cached per masked
   `memory_id`, so the backbone runs on genuinely new endogenous states
   rather than every frame.
3. **Branch-loop imagination substrate.**  Validated KV branch-carry and
   partial cache splice spawn latent branches; survival taps feed the
   `SurvivalRetainer` (retention only), content taps feed bounded terms.
   The v2 tap contracts were designed for exactly this consumer role.

If the policy itself is later trained with genuine RLTT, the implementation
must retain the paper's defining credit route in action-policy form: sample an
action from the terminal loop, compute one group-normalized outcome advantage,
apply it to every loop's action distribution with normalized loop weights
(uniform, progressive, or exit-probability weighting), and keep KL
regularization on the terminal loop only.  The paper's ablations suggest the
existence of all-loop credit matters more than the exact normalized weighting
scheme; that is the hypothesis to test, not a license to label ordinary
hidden-state supervision as RLTT.

## Watchlist, not dependencies

The following are relevant but intentionally not in the compact baseline:

- giant V-JEPA/V-JEPA 2 style video models;
- state-space long-memory replacements for the explicit event ledger;
- very recent object-centric MCTS systems such as ObjectZero;
- latent reward optimization without an independent outcome verifier;
- online encoder or policy adaptation before the dynamics/readout route is
  calibrated.

They should enter only through a named experiment and causal acceptance test.

## Evidence required for future changes

A proposed component must answer:

1. What observed target trains or calibrates it?
2. Does it receive nonzero updates on a synthetic test?
3. Can a controlled change in the component alter a decision?
4. Does autonomous task performance improve against a matched-compute baseline?
5. Does the gain survive task/source-disjoint evaluation?
6. For pairwise readers, is the public score strictly antisymmetric in both
   orders?
7. Can the same behavior be obtained through a simpler exact graph, event
   aggregate, or small pointwise head?

If the last answer is yes, the smaller mechanism wins.

## Applied example: control attribution (2026-07-17)

The ego/control-set port answers the seven questions as follows:

1. Observed target: committed transitions' MOVED/DISAPPEARED events paired
   with the executed action index; no labels, no speculation.
2. Nonzero updates: `test_control_attribution_learns_action_coupled_track_...`
   drives distinct displacement responses and asserts the influence score
   separates a controlled track from a drifter and a static object.
3. Decision causality: `test_ego_motion_hazard_term_changes_the_selected_action`
   flips the selected action through the named `ego_motion_hazard` term with
   all shared terms perturbed by an order of magnitude less.
4. Matched-compute autonomous comparison: the named experiment
   `utilities/tests/manual/run_hs_v2_ego_onoff_trio_v1.py` (matched budget,
   same seed, `EgoConfig.enabled` as the only difference); results recorded in
   `CAPABILITY_PARITY.md`.
5. Task/source-disjoint: negative attribution stays scoped through the
   existing evidence rules; signature bootstrap carries at half weight only.
6. Not a pairwise reader; not applicable.
7. Simpler mechanism check: the exact graph is state-specific and cannot
   generalize object hazard; scene-signature evidence has no per-object
   attribution; the affordance head exists but had no clickless grounding
   route.  The port therefore grounds the existing head; the only new state
   is the action-to-motion counts, which nothing else supplies.

## Applied example: exogenous memory identity (2026-07-17)

Motivating observation: rendering `tr87` showed a per-step countdown bar, so
every observation hashes to a fresh state and hash-addressed memory never
sees a revisit — the state graph, no-change penalties, distance labels, and
cross-episode reuse were all silently inert in clock-bearing environments.

1. Observed target: per-cell change-event trajectories of committed
   transitions plus the executed action sequence, compared across episodes.
2. Nonzero updates: the mask forms on a synthetic ticker after two episodes
   with differing action histories, and provably does not form when the
   action histories are identical (no intervention, no evidence).
3. Decision route: masked memory identity restores graph revisits
   (`test_agent_graph_reuses_states_across_tick_variants`: revisits > 0 with
   the filter, exactly 0 without), which feeds exact predictions, known-edge
   bonuses, and no-change penalties back into scoring.
4. Matched-compute autonomous comparison:
   `utilities/tests/manual/run_hs_v2_exogenous_onoff_trio_v1.py`; results in
   `CAPABILITY_PARITY.md`.
5. Scope: masks are learned per task and shape; nothing transfers across
   tasks.
6. Not a pairwise reader; not applicable.
7. Simpler mechanism check: no existing mechanism can restore state identity
   under environment clocks — adapter-level cropping would be hand-coded per
   environment (rejected as domain-specific), and object-level abstraction
   is a far larger change.  Known limitations, stated: episode-1 memory is
   inherently unfiltered; a lifetime-monotone clock that never resets across
   episodes evades the reproducibility test; a time-deterministic hazard
   animation is exogenous by this test and will be merged in memory, leaving
   object beliefs and evidence as the safety carrier.
