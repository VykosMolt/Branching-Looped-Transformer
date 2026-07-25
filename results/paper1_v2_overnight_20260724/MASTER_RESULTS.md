# Paper 1 v2 Overnight Programme — MASTER RESULTS

Programme root: `artifacts/reports/paper1_v2_overnight_20260724/`
Start: 2026-07-24 19:15:40 UTC. This report written: 2026-07-24 ~21:35 UTC.
**Total wall-clock: ~2h20m of a 9-hour hard ceiling.** All three primary experiments
completed with real, honestly-reported verdicts; the programme stopped once all
required deliverables were secured rather than consuming the remaining budget on
optional variants (per the priority/sacrifice ordering in the original spec).

## Environment (see `ENVIRONMENT.json`, `INITIAL_GIT_STATUS.txt`, `FINAL_GIT_STATUS.txt`)

- Repo: `/home/moloch/ouro_project`, branch `main`, HEAD `e4776dd41a85cad699ac36f309b5986ab48bd171`
  (unchanged throughout — no commits made).
- Model: `models/ouro_rltt_local` (OuroForCausalLM, 48 physical layers, 4 recurrent
  loops, hidden 2048), untouched (no weight modification; verified via file
  timestamps and git status).
- Python 3.14.6, torch 2.12.0.dev20260407+cu128, CUDA 12.8, transformers 4.54.1
  (matches expected), tokenizers 0.21.4. GPU: RTX 5070 Ti Laptop, 12227 MiB.
- Worktree was dirty at start (extensive pre-existing untracked/deleted files from
  prior sessions) and was preserved as-is; no `git reset`/`git clean`, no destructive
  checkout, no commits. **The only diff between `INITIAL_GIT_STATUS.txt` and
  `FINAL_GIT_STATUS.txt` is the 7 new `bg_v2_overnight_*.py` scripts and the new
  programme-root directory** — verified by direct diff, not assumed.
- The frozen `cross_loop_early_layer_taps_20260720/` artifact's `SHA256SUMS` were
  verified 100/100 matching **both before and after** this programme's geometry audit
  ran against it — it was read-only throughout.
- No paper file (`ouro_paper_draft_v3_6.md`, Paper 2, or any arXiv source package) was
  edited, and none was even present as a tracked file to accidentally modify.

## Engineering issue encountered and fixed

The first Horizon Logic generation attempt hit CUDA OOM after one task: `out` (the
`model.generate(..., output_scores=True, return_dict_in_generate=True)` return value,
holding every per-step logit tensor) was never dereferenced before the next task's
`generate()` call, so GPU memory accumulated across tasks until allocation failed.
Fixed by moving per-step logits to CPU immediately, `del`-ing `out`, and calling
`torch.cuda.empty_cache()` at the end of each task; verified via an isolated 8-task
smoke test with GPU memory returning to baseline. A second false alarm (apparent
multi-minute "hangs" on task 1 in two subsequent launches) turned out to be a stdout
block-buffering artifact of running under `nohup > logfile` — not an actual stall;
switching to `python3 -u` (unbuffered) resolved the visibility issue and confirmed the
underlying generation pipeline was healthy throughout (~15-40s/task, no thermal
throttling: GPU clock stayed at a healthy ~2000-2400 MHz).

## Shared generation

One Horizon Logic candidate pool was generated and used for **both** Part I and Part II
(the preferred shared-pool design held up — no fallback to a separate pool was needed).

- Task source: `data/branch_training_logic_expansion_v1/processed/logic_tasks.jsonl`,
  category `synthetic_propositional`, proof_depth 2-4, deterministic truth-table
  verifier.
- Pilot (24 tasks, k=4, `max_new_tokens=320`): revealed a 39.6% malformed rate driven by
  proof_depth 3 (61% malformed vs. 27% at depth 2) — the model frequently needs more
  than 320 tokens to reach a committed answer at higher depth. One permitted one-time
  adjustment made: `max_new_tokens` raised to 448 for the main run. The pilot's own 24
  tasks/96 candidates were kept as a diagnostic-only artifact
  (`horizon_logic/pilot_diagnostic_only/`) and excluded from the final analysis pool, to
  avoid mixing two different token budgets in one dataset.
- Main run (170 tasks, k=4, `max_new_tokens=448`, temp=0.7, top_p=0.95, seed=20260724):
  **680 candidates**, 454 raw successes, **170 malformed (25.0%)**, 212 hit-max-tokens
  (31.2%), elapsed 4368.9s (~73 min). Task split: 80 train / 31 val / 59 heldout.

## Part I — Horizon Logic strict pre-answer study

**Verdict: `SECOND_DOMAIN_PREANSWER_REPLICATION`**

- Malformed candidates (no valid verifier label) were **excluded from the correctness
  label set** for this analysis (not folded into "incorrect"), leaving **510 scorable
  candidates** (75.0%) — comfortably above the 350-trajectory minimum. This choice was
  made specifically to avoid the classifier learning "did this hit the token budget"
  instead of "is the answer correct," given hit-max-tokens correlates heavily with
  malformed.
- Shortcut-only AUROC (heldout): **0.585**. Hidden-only: **0.726**. Hidden+shortcuts:
  **0.726**. Incremental AUROC: **+0.141**. Paired task-clustered 95% CI:
  **[+0.004, +0.290]** (excludes zero). Headroom-normalized increment: **0.340**.
- All strict-cut and integrity controls passed (zero task crossing, zero answer-region
  exclusion violations, gold value excluded by construction, shuffled-label control at
  chance 0.508, max single-task influence 0.029, zero duplicates). Two additional
  malformed-artifact controls (added mid-run after review) both passed: the increment
  survives dropping high-malformed-share tasks (+0.141 → +0.141), and while a
  malformed-vs-clean hidden-state detector reaches a similarly high AUROC (0.713),
  restricting to low-malformed-share tasks does not change the correctness increment.
- **Honest caveat:** the CI's lower bound (+0.004) is close to zero, and the domain has
  a strong class imbalance (89% success among scorable candidates) that limits
  effective negative-class sample size. This is a real, controls-passing replication,
  not a high-precision one.

## Part II — powered terminal-selection evaluation

**Verdict: `TERMINAL_SELECTION_ESTABLISHED`**

- 39 informative heldout groups (15 all-correct, 5 all-wrong, excluded from the endpoint)
  out of 59 heldout tasks — above the 25-group minimum.
- Observed top-1: **34/39 (0.872)**. Matched-random expectation: **23.0/39 (0.590)**.
  Exact Poisson-binomial p (observed-or-better): **2.44e-05**. Paired difference:
  **+0.282**, task-clustered 95% CI **[+0.167, +0.391]**. Pairwise ranking accuracy
  0.762 (122 pairs), MRR 0.927, top-2 oracle retention 0.949, top-3 1.0.
- All controls passed (zero crossing, zero duplicates, shuffled-score control exactly
  at chance, zero candidate-order-invariance failures, no held-out-label tuning).
- **Required companion caveat:** a shortcut-only control selector (length/marker-found/
  hit-max features, no hidden state) reaches 35/39 (0.897) — nominally *better* than the
  hidden-state selector. 87% of informative groups contain a malformed candidate, so
  most of the forced-choice power here is plausibly "detect non-commitment," which
  shortcuts capture as well as hidden states do. On the 5 groups with only well-formed
  candidates, the hidden selector is 5/5 vs. matched-random 3.75/5 — suggestive of
  genuine content discrimination but far too small (n=5) to stand alone. The verdict is
  earned on its own defined terms, but the paper-facing claim must carry this caveat.

## Part III — subspace-vs-subspace geometry audit (already complete before this update)

**Verdicts:** readable<->outcome overlap present at all loci (not early-locus-enriched;
`L4_47` shows the tightest alignment); `READABLE_OUTCOME_OVERLAP_WITHOUT_WRITABLE_ALIGNMENT`
at the three late loci with real writable data; `INSUFFICIENT_MATCHED_LOCUS_WRITABLE_DATA`
for the early-vs-late writable comparison itself (no early-locus writable tensor exists
anywhere in the repo, and this was not manufactured); loop-to-loop rotation descriptively
consistent with (but not a new statistical confirmation of) the frozen study's L1->L2
finding. Elapsed: 233.3s (pure linear algebra, no model loaded). Full detail in
`subspace_geometry/RESULTS.md`.

## Controls summary (all three experiments)

Every required control category passed: task-level split integrity (zero crossings,
verified independently at each stage), duplicate audits, shuffled-label/score sanity
checks (all landed at chance), no single-task or single-group dominance, fixed/frozen
splits opened once, raw predictions preserved (`raw_predictions_heldout.json`,
`terminal_pool.pt`, `subspace_bases.pt`). The two malformed-artifact controls added
mid-run (per coordinator review) both passed and are documented in
`horizon_logic/RESULTS.md`.

## Omitted analyses (and why — all explicit sacrifices, not silent gaps)

1. Secondary early-layer feature variant for Horizon Logic (e.g. `L3_16`) — the shared
   canonical extractor only supports layers `{24,36,47}`; adding a new hook-based
   extractor for one secondary comparison was lower priority than securing the three
   primary results.
2. Cross-domain transfer test of the frozen S3B2/DualAnchor selector under Horizon
   Logic's distribution — treated as a secondary terminal-selector comparison, ranked
   below the primary powered test in the sacrifice order.
3. CCA-based geometry variant and rank-10 subspaces — both explicitly marked optional in
   the spec; principal angles at ranks 1/3/5 (required) were run in full at every locus.
4. Full 2000-round label-permutation null for the geometry readable/outcome ensembles —
   reduced to 25 rounds (one locus) because each round refits a 40-100-member bootstrap
   ensemble; the primary random-rotation null (2000 rounds, cheap) was run at full scale
   and is what every geometry verdict above is actually referenced against.

None of these omissions affect the primary verdicts; all are logged here and in each
experiment's own RESULTS.md rather than silently dropped.

## Final verdicts, restated together

| Experiment | Verdict |
| --- | --- |
| Horizon Logic pre-answer | `SECOND_DOMAIN_PREANSWER_REPLICATION` |
| Terminal selection | `TERMINAL_SELECTION_ESTABLISHED` (with shortcut-comparability caveat) |
| Subspace geometry | `READABLE_OUTCOME_OVERLAP_WITHOUT_WRITABLE_ALIGNMENT` (late loci); `INSUFFICIENT_MATCHED_LOCUS_WRITABLE_DATA` (early-vs-late writable comparison) |

## Answers to the programme's three questions

**1. Does strict pre-answer hidden-state information replicate beyond GSM8K on Horizon
Logic when judged by within-domain incremental value rather than absolute scalar
equality?** **Yes.** The paired increment (+0.141, CI [+0.004,+0.290], excluding zero)
replicates the qualitative GSM8K finding on a second domain with a different verifier
family and a different reasoning structure (proof-depth-controlled propositional
entailment rather than arithmetic). The absolute AUROCs differ from GSM8K's (as
expected and explicitly not required to match), and the CI is wide — an honest
replication, not a strong one.

**2. Does readable correctness support reliable forced terminal selection on a powered,
task-disjoint, reward-diverse pool?** **Yes, on its own defined statistical terms** (34/39
vs. 23.0/39 matched-random, exact p=2.44e-05), **but with an important qualification**:
a shortcut-only selector performs comparably or better on this pool, and most of the
discriminative power traces to detecting non-commitment (malformed candidates) rather
than distinguishing well-formed-but-wrong from correct content. The narrower, more
defensible claim is "forced selection beats matched-random," not "the model's hidden
states let you pick the better answer" in the strong sense.

**3. Are early readable loci geometrically better aligned with writable and
outcome-relevant subspaces than established late loci?** **No** for readable-outcome
overlap (present everywhere tested, tightest at the late terminal locus `L4_47`, not
enriched at the early loci), and **not answerable** for the writable comparison (no
early-locus writable tensor exists in this repository, and none was manufactured to
answer the question). Where the writable comparison *can* be run (three late loci),
neither readable nor outcome subspaces align with it beyond chance — a clean empirical
instance of "readable is not the same as writable," diagnostic only, not a causal-control
claim.

## Master artifact root and verification

`artifacts/reports/paper1_v2_overnight_20260724/`. SHA-256 verification: see
`SHA256SUMS` (generated last, after this report, covering every deliverable file in the
programme root) and `FINAL_GIT_STATUS.txt`.
