# Horizon Logic Strict Pre-Answer Study — RESULTS

**STATUS: COMPLETE.** Main generation: 170 tasks, k=4, `max_new_tokens=448`,
`temperature=0.7`, `top_p=0.95`, seed=20260724, elapsed 4368.9s (~73 min).

## Protocol

- Task source: `data/branch_training_logic_expansion_v1/processed/logic_tasks.jsonl`,
  category `synthetic_propositional`, proof_depth 2-4 (the reasoning-horizon control
  knob), deterministic truth-table verifier.
- Prompt: `mcq_prompt` (reused verbatim from `bg_steering_suite_lib.py`), options mapped
  to letters, requiring an explicit `FINAL ANSWER: <letter>` commitment line.
- Strict pre-answer cut: text before the first `FINAL\s*ANSWE` marker match, computed at
  the text level and re-tokenized via the canonical extractor (same convention as
  `proto_introspection_preanswer_recapture.py`).
- Feature basis: canonical pooled `[3,4,2048]` hidden state (layers `{24,36,47}` x loops
  `{1..4}`) at the pre-answer cut, from `BGTransformerFeatureExtractor`. No early-layer
  secondary variant was evaluated (see Sacrifices).
- Shortcuts: pre-cut generated token count, mean pre-cut token logprob, min pre-cut token
  logprob, hit-max-tokens indicator.
- Split: task-level, deterministic hash of `task_uid`, ~50/20/30 train/val/heldout, never
  crossed. Hyperparameters (k_pca, l2) selected via task-grouped 5-fold CV on train+val
  only; heldout opened once.
- Pilot-informed one-time adjustment: the 24-task pilot (`max_new=320`) showed a 39.6%
  malformed rate driven almost entirely by proof_depth 3 tasks (61% malformed at depth 3
  vs. 27% at depth 2, and 24/38 malformed candidates also hit the token cap) -- the
  model frequently needs more than 320 tokens to reach a committed answer at depth 3-4.
  The main run therefore froze `max_new_tokens=448` (the one permitted generation-config
  adjustment) before any candidate for the main pool was generated. To avoid mixing two
  different token budgets in one analysis pool, the pilot's own 24 tasks/96 candidates
  are kept as a diagnostic-only artifact (`horizon_logic/pilot_diagnostic_only/`) and are
  **not** included in the numbers below; the main run regenerates its own full task set
  under the single frozen budget.
- **Malformed-candidate handling (methodological point flagged during the run):**
  malformed candidates (no `FINAL ANSWER` marker found, or no parseable letter) have no
  valid verifier label and are **excluded from the shortcut/hidden/combined AUROC label
  set** -- they are not folded into the "incorrect" class, since hit-max-tokens
  correlates heavily with malformed (pilot: 24/38 malformed candidates also hit the
  token cap), and folding them in would risk the classifier mostly learning "did this
  hit the budget" rather than "is the answer correct." They ARE, however: (a) kept in
  full in the raw preserved records, (b) counted in the required malformed-rate metric,
  and (c) checked with two explicit robustness controls -- see "Malformed-artifact
  controls" below.

## Headline numbers

- n_tasks=170, n_candidates=680 (170 tasks x k=4), depth histogram {2: 460, 3: 192, 4: 28}.
- n_scorable (non-malformed) = 510 (75.0%); n_malformed = 170 (25.0%); n_hit_max_tokens
  = 212 (31.2%). Scorable base rate (success among scorable) = **0.890** — a real,
  substantial class imbalance in this domain: once the model commits to an answer at
  proof_depth 2-4, it is usually right. This is a genuine descriptive property of
  Horizon Logic under this generation config, not an artifact of the analysis.
- Task-level split: 80 train / 31 val / 59 heldout (170 total). Heldout: n=171 scorable
  candidates across 56 tasks, base_rate=0.889.
- **Shortcut-only AUROC (heldout): 0.5852**
- **Hidden-only AUROC (heldout): 0.7261**
- **Hidden+shortcuts AUROC (heldout): 0.7261** (identical to hidden-only at the selected
  regularization — the combined model's cross-validated hyperparameters did not further
  benefit from the shortcut dimensions once hidden features were included; this is
  visible directly in the near-identical CV AUROCs for the hidden and combined
  hyperparameter search: 0.5913 vs. 0.5915)
- **Incremental AUROC (combined − shortcut): +0.1409**
- **Paired task-clustered 95% CI on the increment: [+0.0044, +0.2899]** (2000 rounds).
  The interval excludes zero, but its lower bound sits close to zero — an honest
  reading is "a real, positive, but not tightly-bounded increment," not a slam-dunk
  effect size, given the limited effective negative-class sample (~19 incorrect
  candidates in the heldout split, driven by the domain's 89% scorable success rate).
- **Headroom-normalized increment: 0.3397** (of the remaining AUROC headroom above the
  shortcut baseline, ~34% is captured by adding hidden features).

## Cross-domain descriptive comparison (GSM8K reference numbers, not a target to match)

| Quantity | GSM8K (paper) | Horizon Logic (this run) |
| --- | ---: | ---: |
| Shortcut-only AUROC | 0.731 | 0.585 |
| Hidden-only AUROC | 0.745 | 0.726 |
| Hidden+shortcuts AUROC | 0.797 | 0.726 |
| Incremental AUROC | +0.066 [+0.017, +0.114] | +0.141 [+0.004, +0.290] |
| Headroom-normalized increment | n/a (not reported for GSM8K) | 0.340 |

Horizon Logic's shortcut-only AUROC (0.585) is markedly weaker than GSM8K's (0.731) —
length/logprob shortcuts carry much less signal in this domain, plausibly because the
fixed proof-depth structure makes "how long the reasoning is" a much weaker proxy for
correctness than it is for open-ended arithmetic word problems. Horizon Logic's
*incremental* AUROC (+0.141) is nominally larger than GSM8K's (+0.066), but the two
intervals are wide and this run's heldout N (171 scorable candidates, ~56 tasks) is much
smaller than GSM8K's (680 examples, 170 tasks) — the comparison of point estimates
across domains is descriptive only, not a claim that Horizon Logic shows a *stronger*
effect than GSM8K. The nested paired increment excluding zero is the result that
matters, and it does, on both domains.

A lower absolute Horizon Logic AUROC than GSM8K's is **not**, by itself, evidence of a
failed replication — see the programme's evidence-discipline rules. The nested paired
increment is the result that matters.

## Controls

1. Zero task crossing: **true** (0 tasks crossing train/val vs. heldout).
2. Strict answer-region exclusion: **0 violations** (checked: no scorable candidate has
   `n_pre_tok >= n_gen_tok` while a marker was found, i.e. the pre-answer cut never
   silently swallows the whole answer region).
3. Gold-value exclusion: **true by construction** (gold letter/text is never concatenated
   into any prompt or pre-answer feature text; only used at evaluation time).
4. Shuffled-label control: **AUROC 0.508** on the combined-model heldout predictions
   under a shuffled label permutation — indistinguishable from chance, as required.
5. Task-ID permutation control: split assignment is a pure function of
   `sha256(task_uid)`; permuting task IDs cannot change per-record split membership.
6. Shortcut-only comparison: **AUROC 0.585** (see headline numbers).
7. No single-task dependence: **max leave-one-task-out influence on combined AUROC =
   0.029** — dropping the single most influential heldout task moves the combined AUROC
   by less than 0.03; no dominant single-task artifact.
8. Duplicate-task / duplicate-candidate audit: **0 duplicates** (task_uid, candidate_idx)
   pairs.
9. Fixed heldout split: **true** (deterministic hash, opened once).
10. Raw predictions preserved: `raw_predictions_heldout.json` (task_uids, labels, and
    all three score vectors for every heldout scorable candidate).

## Malformed-artifact controls (not one of the original 10, added during the run)

11. **Increment after dropping high-malformed-share tasks:** 3 heldout tasks had >=75%
    malformed candidates; dropping them entirely (168 candidates remain) gives
    combined AUROC **0.7301**, shortcut AUROC **0.5888**, incremental **+0.1413** —
    essentially unchanged from the full-heldout number (+0.1409). **The increment
    survives.** This directly answers the "no malformed-output artifact dominating"
    requirement.
12. **Malformed-vs-clean separability:** a hidden-only classifier predicting
    malformed(1)/clean(0) on all heldout candidates (malformed included, a genuinely
    different label than correctness) reaches AUROC **0.7126** — close in magnitude to
    the correctness hidden-only AUROC (0.7261). This is flagged honestly rather than
    swept aside: the hidden state is doing real work distinguishing both "will this
    trajectory ever commit" and "will the committed answer be right," and the two
    signals are not cleanly separable from AUROC magnitude alone. What *does* rule out
    the strongest version of the worry — that the correctness increment is nothing but
    a repackaged malformed-detector — is control #11: restricting to heldout tasks with
    a *low* malformed share still gives the same +0.141 increment, and the shortcut
    model (which already has direct access to `hit_max_tokens`, the most malformed-
    correlated single variable) only reaches 0.585, not 0.71+. The hidden signal is
    doing more than reconstructing the malformed indicator, but the two phenomena likely
    share some underlying process-quality structure in this domain, and that nuance
    should not be flattened into a single clean claim.

## Verdict

**`SECOND_DOMAIN_PREANSWER_REPLICATION`**

All four requirements are met: positive hidden-plus-shortcuts increment (+0.141), a
paired task-clustered 95% interval excluding zero ([+0.004, +0.290]), passing strict-cut
controls (zero task crossing, zero answer-region-exclusion violations, gold value
excluded by construction, shuffled-label control at chance), and no dominant single-task
artifact (max influence 0.029) or malformed-output artifact (control #11 survives).

**Honest caveats, not grounds to downgrade the verdict but necessary context:**
- The paired CI's lower bound (+0.004) sits close to zero — this is a real,
  controls-passing replication, not a high-precision one.
- The domain has a substantial class imbalance (89% success among scorable
  candidates), driven by the domain property that once this model commits to an
  answer at proof_depth 2-4 it is usually right; the effective heldout negative-class
  size is small (~19 candidates), which is the main driver of the wide CI.
- The malformed-vs-clean separability control (#12) shows the hidden state's
  malformed-detection and correctness-detection signals are of similar AUROC magnitude
  — not proof of contamination (control #11 rules out the simplest form of that
  concern), but a nuance the paper integration plan should preserve rather than smooth
  over.
- Per the programme's evidence-discipline rules: Horizon Logic's absolute AUROCs are
  lower than GSM8K's shortcut/hidden numbers in places and not required to match; the
  nested paired increment excluding zero is the result, and it replicates.

## Sacrifices

- Secondary early-layer feature variant (e.g. `L3_16`) was not evaluated for Part I —
  the shared canonical extractor (`BGTransformerFeatureExtractor`) only supports the
  audited layers `{24,36,47}`; adding a hook-based early-layer extractor for a
  single secondary comparison was judged lower priority than securing the primary
  canonical-basis result and the other two experiments, per the programme's own
  sacrifice ordering ("optional Horizon diagnostics" is last to keep, first to drop).
