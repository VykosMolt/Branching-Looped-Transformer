"""Post hoc (2026-10-10): Run 6 with the as-run (pod) hyperparameter selections forced. The pod's GroupKFold tie order
(numpy argsort on an AVX-512 CPU) cannot be rebuilt on this laptop (no AVX-512), and the held-out scores depend on the
folds only through the selected (k, C). So the final step of analyze_run3._fit_score (_design on train+val, then
LogisticRegression(C)) is run with the pod's selections from results.json: baseline C=0.1; hidden k=16, C=0.1;
combined k=16, C=1.0. Gate: executed labels must reproduce the published primary and within-question numbers exactly.
Then the same forced configs with chronological labels (cohort as executed, and cohort recomputed). Writes c2_run6_forced.json."""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "C4_run6_oof_within"))
import r6data  # noqa: E402

r6data.setup_paths()
import analyze_run6 as A6  # noqa: E402
import analyze_run3 as A3  # noqa: E402
from answer_detector_gsm8k import value  # noqa: E402
from bg_v2_overnight_horizon_analysis import bootstrap_task_clustered  # noqa: E402
from chrono_parsers import first_commitment_chrono  # noqa: E402
from proto_introspection_controls_analysis import auroc  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

pub = json.load(open(r6data.R6 + "/results.json"))["primary"]["hyperparams"]
CFG = {k: (tuple(pub[k]["k"]), pub[k]["C"]) for k in ("baseline", "hidden", "combined")}
N = 96
S_fn = lambda r: torch.cat([A6.lp_at(N)(r), A6.s_diff(r)])
recs = r6data.load()
for r in recs:
    r["own_exec"], r["own_chrono"] = r["own_answer"], first_commitment_chrono(r["generated_text"])


def final(blocks, S, y, tv, ho, cfg):  # the last three lines of analyze_run3._fit_score, with best = cfg
    itv = np.where(tv)[0]
    Xtr, Xte = A3._design(itv, np.where(ho)[0], blocks, S, cfg[0])
    return LogisticRegression(C=cfg[1], max_iter=5000).fit(Xtr, y[itv]).predict_proba(Xte)[:, 1]


out = {"forced_configs": {k: [list(v[0]), v[1]] for k, v in CFG.items()}}
for cond in ("executed", "chrono_labels", "chrono_all"):
    lab_key = "own_exec" if cond == "executed" else "own_chrono"
    for r in recs:
        r["own_answer"] = r["own_chrono" if cond == "chrono_all" else "own_exec"]
    A6.answer_free_table(recs)
    for r in recs:
        own = r[lab_key]
        r["parseable"], r["success"] = value(own) is not None, value(own) is not None and value(own) == value(r["gold"])
    rows = [r for r in recs if r["parseable"] and r["free"][N]]
    y = np.array([int(r["success"]) for r in rows]); g = np.array([r["task_id"] for r in rows])
    tv = np.array([r["split"] in ("train", "val") for r in rows]); ho = np.array([r["split"] == "heldout" for r in rows])
    S = np.stack([S_fn(r).float().numpy() for r in rows])
    H = [np.stack([r["prefix_features"][r6data.GI].flatten().float().numpy() for r in rows])]
    sb = final([], S, y, tv, ho, CFG["baseline"])
    sh = final(H, np.zeros((len(rows), 0)), y, tv, ho, CFG["hidden"])
    sc = final(H, S, y, tv, ho, CFG["combined"])
    del H
    T = lambda v: torch.tensor(v, dtype=torch.double)
    yh, th = T(y[ho]), list(g[ho])
    out[cond] = {"n_cohort": len(rows), "n_heldout": int(ho.sum()), "n_heldout_negatives": int((y[ho] == 0).sum()),
                 "surface": auroc(T(sb), yh), "hidden": auroc(T(sh), yh), "both": auroc(T(sc), yh),
                 "increase": auroc(T(sc), yh) - auroc(T(sb), yh),
                 "ci95": bootstrap_task_clustered(T(sc), T(sb), yh, th, rounds=2000)["ci95"],
                 "within_question_heldout": {k: A3.within_task_auroc(v, y[ho], th) for k, v in (("combined", sc), ("hidden", sh), ("shortcut", sb))}}
    print(cond, json.dumps(out[cond], default=float), flush=True)
json.dump(out, open(os.path.join(HERE, "c2_run6_forced.json"), "w"), indent=2, default=float)
