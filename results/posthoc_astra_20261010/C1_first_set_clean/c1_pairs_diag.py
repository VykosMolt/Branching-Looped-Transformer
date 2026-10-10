"""Post hoc diagnostic (2026-10-10): is the change in Run 1's out-of-fold within-question numbers after excluding the 15
due to the removed pairs or to refitting on new folds? Recomputes the gate's (published-cohort) out-of-fold hidden and
reader scores with the same code as c1_analysis.py item 3a, then scores them on the 203 pairs that remain after the
exclusion (no refit). Writes c1_pairs_diag.json."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import r1data  # noqa: E402
r1data.setup_paths()
sys.path.insert(1, r1data.R3)
import analyze_run1 as A1  # noqa: E402
import analyze_run3 as A3  # noqa: E402
from bg_steering_suite_lib import mcq_prompt  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
sys.path.insert(0, HERE)
from c1_oof_reader import oof_reader  # noqa: E402

recs = r1data.load()
for r in recs:
    r["task_id"] = r["task_uid"]
A1.answer_free_table(recs)
bad = {(d["task_uid"], d["candidate_idx"]) for d in json.load(open(os.path.join(HERE, "inwindow_r1.json")))["inwindow"]}
P = [r for r in recs if not r["malformed"] and r["free"][48]]
y = np.array([int(r["success"]) for r in P]); g = np.array([r["task_id"] for r in P])
H = np.stack([A1.feat_at(48)(r).flatten().float().numpy() for r in P])
hid = np.zeros(len(P))
for a_, b_ in GroupKFold(5).split(H, y, g):
    tv = np.zeros(len(P), bool); tv[a_] = True
    hid[b_] = A3._fit_score([H], np.zeros((len(P), 0)), y, g, tv, ~tv)[0]
X = [f"Prompt:\n{mcq_prompt(r['task_prompt'], r['options'])}\n\nVisible reasoning prefix:\n{r['prefix_texts'][3]}" for r in P]
rdr = oof_reader(X, y, g)
keep = np.array([(r["task_uid"], r["candidate_idx"]) not in bad for r in P])
out = {}
for name, m in (("all_209", np.ones(len(P), bool)), ("without_15_no_refit", keep)):
    yy, gg = y[m], list(g[m])
    by = {}
    for i, t in enumerate(gg):
        by.setdefault(t, []).append(i)
    mixed = sorted(t for t, ix in by.items() if 0 < yy[ix].mean() < 1)
    pairs = {t: [(p, n) for p in by[t] if yy[p] == 1 for n in by[t] if yy[n] == 0] for t in mixed}
    wq = lambda s, ts: np.mean([1.0 if s[p] > s[n] else 0.5 if s[p] == s[n] else 0.0 for t in ts for p, n in pairs[t]])
    h, rd = hid[m], rdr[m]
    rng = np.random.default_rng(20260724)
    d = sorted(wq(h, ts) - wq(rd, ts) for ts in (rng.choice(mixed, len(mixed)) for _ in range(2000)))
    out[name] = {"hidden": A3.within_task_auroc(h, yy, gg), "reader": A3.within_task_auroc(rd, yy, gg),
                 "diff": wq(h, mixed) - wq(rd, mixed), "ci95": [d[50], d[1949]]}
    print(name, json.dumps(out[name], default=float), flush=True)
json.dump(out, open(os.path.join(HERE, "c1_pairs_diag.json"), "w"), indent=2, default=float)
