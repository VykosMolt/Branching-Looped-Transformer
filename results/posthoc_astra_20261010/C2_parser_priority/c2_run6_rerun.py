"""Post hoc (2026-10-10, Astra audit correction 5): Run 6 primary and held-out within-question AUROC with executed vs
chronological labels, using the sealed estimators unchanged (analyze_run3.proper_arm / within_task_auroc via
analyze_run6's S, feat_at, answer_free_table, choose_n_star, verdict). Same fold algorithm (laptop sklearn GroupKFold)
in every condition, so the comparison is paired.
Conditions:
  executed           : gate; the sealed labels (first_commitment), must reproduce the published Run 6 numbers.
  chrono_labels      : labels (own answer, success) from first_commitment_chrono; cohort as executed (answer-free test
                       with the executed own answer). Only the labels change.
  chrono_all         : chronological own answer also used by the answer-free detector (cohort recomputed).
Writes c2_run6_rerun.json and run6 scores to cache/ (for C4 none needed)."""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
C4 = os.path.join(os.path.dirname(HERE), "C4_run6_oof_within")
sys.path.insert(0, C4)
import r6data  # noqa: E402

r6data.setup_paths()
import analyze_run6 as A6  # noqa: E402  sealed
import analyze_run3 as A3  # noqa: E402  sealed
from answer_detector_gsm8k import value  # noqa: E402
from chrono_parsers import first_commitment, first_commitment_chrono  # noqa: E402

N = 96
S = lambda r: torch.cat([A6.lp_at(N)(r), A6.s_diff(r)])
recs = r6data.load()
for r in recs:
    r["own_exec"] = r["own_answer"]
    assert first_commitment(r["generated_text"]) == r["own_answer"]
    r["own_chrono"] = first_commitment_chrono(r["generated_text"])

out = {}
for cond in sys.argv[1:] or ["executed", "chrono_labels", "chrono_all"]:
    lab_key = "own_exec" if cond == "executed" else "own_chrono"
    det_key = "own_chrono" if cond == "chrono_all" else "own_exec"
    for r in recs:
        r["own_answer"] = r[det_key]
    A6.answer_free_table(recs)
    n_star, frac = A6.choose_n_star(recs)
    for r in recs:
        own = r[lab_key]
        r["parseable"] = value(own) is not None
        r["success"] = value(own) is not None and value(own) == value(r["gold"])
    primary = [r for r in recs if r["parseable"] and r["free"][N]]
    assert all(r["prefix_features"] for r in primary)
    res, sc = A3.proper_arm(f"primary_{cond}", primary, [A6.feat_at(N)], S)
    wq = {k: A3.within_task_auroc(sc[k].numpy(), sc["y"].numpy(), sc["tasks"]) for k in ("combined", "hidden", "shortcut")}
    out[cond] = {"n_star_rule": n_star, "train_share_96": frac[N], "n_cohort": len(primary),
                 "n_incorrect_cohort": sum(not r["success"] for r in primary), "verdict": A6.verdict(res),
                 "surface": res["auroc_shortcut_only"], "hidden": res["auroc_hidden_only"], "both": res["auroc_hidden_plus_shortcuts"],
                 "increase": res["incremental_auroc_combined_minus_shortcut"],
                 "ci95": res["paired_task_clustered_bootstrap_combined_minus_shortcut"]["ci95"],
                 "n_heldout": res["n_heldout"], "n_heldout_tasks": res["n_heldout_tasks"], "n_heldout_negatives": res["n_heldout_negatives"],
                 "hyperparams": res["hyperparams"], "within_question_heldout": wq}
    print(cond, json.dumps(out[cond], default=float), flush=True)
    json.dump(out, open(os.path.join(HERE, "c2_run6_rerun.json"), "w"), indent=2, default=float)
