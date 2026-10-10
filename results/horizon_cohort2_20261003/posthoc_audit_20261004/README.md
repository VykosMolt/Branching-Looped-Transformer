# Post-hoc audit of the second set (Run 4), 2026-10-04. NOT preregistered.

Independent validity audit (eval-auditor agent) plus re-verification by the main session. The sealed verdict stands as executed (UNRESOLVED / MOSTLY_DIFFICULTY); nothing here replaces it.

- **Fold sensitivity** (`robust.json`, `robust.py`): the sealed code's GroupKFold tie order is not seeded. Seed 0 reproduces the pod exactly (0.6809 / 0.7342, +0.0533 [−0.0059, 0.1088]; control +0.0489 [0.0077, 0.0890]; within-question 0.7397). Over 31 tie orders (30 seeds + the laptop's): primary +0.052 to +0.064, CI > 0 in 16/31; end-of-prompt control +0.038 to +0.056, CI > 0 in 22/31; both (headline READABILITY_BEYOND_DIFFICULTY) 12/31; within-question hidden 0.726 to 0.740 in 31/31.
- **Within-question, out of fold, all splits** (`oof_within.json`): 130 mixed questions, 237 pairs. Hidden 0.709 [0.646, 0.767], combined 0.726, surface 0.570, TF-IDF reader 0.759 [0.699, 0.815]; hidden − reader −0.051 [−0.117, 0.017].
- **Held-out within-question reader** (`reader_ho.py`, re-run 2026-10-04): 0.753 [0.638, 0.851] on the 73 pairs; hidden − reader −0.014 [−0.137, 0.116].
- **Commit timing** (`commit_oof.py`, re-run): in 185/237 pairs the correct answer states its answer later; "commits later" alone AUROC 0.781. Hidden probe 0.762 on those 185 pairs vs 0.519 on the other 52 (reader 0.805 vs 0.596).
- **Within-question by cut** (`grid_wq.json`, held-out): hidden 0.631 / 0.714 / 0.740 / 0.714 / 0.679 / 0.607 at 16 / 32 / 48 / 64 / 96 / 128 tokens (107 / 91 / 73 / 70 / 53 / 28 pairs); CIs above 0.5 from 16 to 96.
- **Alternative baselines, laptop folds** (`checks2.json`): mean-pooled prompt in the baseline +0.051 [0.013, 0.086]; both prompt summaries +0.053 [0.010, 0.097]; fixed estimator with log-probabilities only +0.144 [0.043, 0.244]; excluding answers with an in-window "**Final Answer**" commitment (9 of 2,376 cohort answers; 15 of 1,387 in Run 1): primary +0.0505 [−0.010, 0.108], within-question 0.721 [0.615, 0.816] (68 pairs).
- **Phrases in the first 48 tokens** (main session, cohort = labelled answers that state no answer by 48 tokens, all splits, 2,376): "parse the facts" 120 answers, 98.3% correct vs 90.7% without; "final answer" 464 answers, 88.1% vs 91.8%.
- **Run 2 transfer** (`transfer_r2.json`): within-question 0.63–0.67 on 27 pairs, CIs span 0.5; inconclusive.

## Files and inputs
- `oof_scores.npy`, `repro_primary.pkl`, `inwindow.pkl`: inputs saved by the audit; `extract4.py` rebuilds the feature cache (`r4cache/`, ~2 GB, not kept) from the SSD shards.
- `commit_oof_output.txt`, `reader_ho_output.txt`: re-run outputs.
- Phrase counts: across questions only; within questions the phrases barely separate correct from incorrect (AUROC 0.53 for "parse the facts", 0.55 for absence of "final answer"; fact-check re-derivation). Not used in the paper.
- Fact-check 2026-10-04: the "**Final Answer**" exclusion moves the alternative-fold primary from +0.0615 [0.0155, 0.1088] to +0.0505 [−0.0101, 0.1081] (interval crosses zero); the first set's primary was not re-run without its 15.
