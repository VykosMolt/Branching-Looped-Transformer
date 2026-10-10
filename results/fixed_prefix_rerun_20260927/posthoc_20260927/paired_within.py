"""Post-hoc: paired within-question comparison, hidden-state probe vs TF-IDF reader, same out-of-fold folds."""
import json, sys
import numpy as np
ROOT = "/home/moloch/ouro_project"; R1 = ROOT + "/opi/preanswer/fixed_prefix_rerun_20260927"
sys.path[:0] = [R1 + "/posthoc_20260927"]
import reader_within as RW  # noqa: E402  (recomputes reader OOF on import)
import analyze_run3 as A3  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
P, y, g = RW.P, RW.y, RW.g
H = np.stack([RW.A1.feat_at(48)(r).flatten().float().numpy() for r in P])
hid = np.zeros(len(P))
for a_, b_ in GroupKFold(5).split(H, y, g):
    tv = np.zeros(len(P), bool); tv[a_] = True
    hid[b_] = A3._fit_score([H], np.zeros((len(P), 0)), y, g, tv, ~tv)[0]
rdr = RW.oof_reader(RW.X)
by = {}
for i, t in enumerate(g):
    by.setdefault(t, []).append(i)
mixed = sorted(t for t, ix in by.items() if 0 < y[ix].mean() < 1)
pairs = {t: [(p, n) for p in by[t] if y[p] == 1 for n in by[t] if y[n] == 0] for t in mixed}
wq = lambda s, ts: np.mean([1.0 if s[p] > s[n] else 0.5 if s[p] == s[n] else 0.0 for t in ts for p, n in pairs[t]])
rng = np.random.default_rng(20260724)
d = sorted(wq(hid, ts) - wq(rdr, ts) for ts in (rng.choice(mixed, len(mixed)) for _ in range(2000)))
res = {"hidden": wq(hid, mixed), "reader": wq(rdr, mixed), "diff": wq(hid, mixed) - wq(rdr, mixed), "ci95": [d[50], d[1949]],
       "n_questions": len(mixed), "n_pairs": sum(len(v) for v in pairs.values())}
json.dump(res, open(R1 + "/posthoc_20260927/paired_within.json", "w"), indent=2)
print(json.dumps(res))
