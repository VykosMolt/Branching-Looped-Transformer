"""Promise to Denis Janiak (e-mail 2026-09-12): report AURC and AUACC next to the AUARC increment.
Post hoc. Same rows and same paired task-clustered bootstrap as bg_v3_selective_prediction.py
(2,000 rounds, seed 20260724). AUACC and AURC follow Denis's incremental_abstention.py
(_auacc: trapezoid over coverage k/n, k = 1..n; AURC = 1 - 1/n - AUACC)."""
import json, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, "/home/moloch/ouro_project/shared/utilities/tests/manual")
import bg_v3_selective_prediction as SP  # original AUARC code, for the gate

RAW = "/home/moloch/blt-records-v4/results/horizon_power_v3_20260726/raw_predictions_heldout_v3.json"

def auacc(scores, y):  # Denis's _auacc
    order = np.argsort(-scores, kind="stable"); v = y[order]
    cov = np.arange(1, len(v) + 1) / len(v); acc = np.cumsum(v) / np.arange(1, len(v) + 1)
    return float(np.trapezoid(acc, cov))

def aurc(scores, y): return float((1 - 1 / len(y)) - auacc(scores, y))

def clustered(sa, sb, y, tasks, fn):
    rng = np.random.default_rng(SP.SEED); by = defaultdict(list)
    for i, t in enumerate(tasks): by[t].append(i)
    tl = sorted(by); d = []
    for _ in range(SP.ROUNDS):
        idx = np.concatenate([by[tl[i]] for i in rng.choice(len(tl), len(tl), replace=True)])
        if y[idx].mean() in (0.0, 1.0): continue
        d.append(fn(sa[idx], y[idx]) - fn(sb[idx], y[idx]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"a": fn(sa, y), "b": fn(sb, y), "delta": fn(sa, y) - fn(sb, y), "ci": [float(lo), float(hi)]}

hp = json.load(open(RAW)); out = {}
for arm in ("pooled_4sc", "pooled_5sc", "new_only_5sc"):
    a = hp[arm]; y = np.array(a["y"], float); t = a["tasks"]
    comb = np.array(a["scores"]["combined"], float); sc = np.array(a["scores"]["shortcut"], float)
    g = SP.clustered_delta(comb, sc, y, t, SP.COVERAGES)
    out[arm] = {"n": len(y), "n_tasks": len(set(t)), "n_incorrect": int((1 - y).sum()),
                "gate_auarc_0.50_1.00": {"delta": g["delta_auarc"], "ci": g["delta_auarc_ci"]},
                "auacc_full": clustered(comb, sc, y, t, auacc), "aurc_full": clustered(comb, sc, y, t, aurc)}
published = {"pooled_4sc": (0.0124, [0.0048, 0.0207]), "pooled_5sc": (0.0119, [0.0044, 0.0196]), "new_only_5sc": (0.0093, [0.0006, 0.0184])}
for arm, (d, ci) in published.items():
    gg = out[arm]["gate_auarc_0.50_1.00"]
    assert abs(gg["delta"] - d) < 1e-9 and all(abs(x - y) < 1e-9 for x, y in zip(gg["ci"], ci)), (arm, gg)
print("gate: published AUARC deltas reproduced exactly")
json.dump(out, open("aurc_auacc_results.json", "w"), indent=1)
for arm, r in out.items():
    A, R = r["auacc_full"], r["aurc_full"]
    print(f"{arm}: n={r['n']} tasks={r['n_tasks']} incorrect={r['n_incorrect']} | AUACC {A['a']:.4f} vs {A['b']:.4f} d={A['delta']:+.4f} [{A['ci'][0]:+.4f},{A['ci'][1]:+.4f}] | AURC {R['a']:.4f} vs {R['b']:.4f} d={R['delta']:+.4f} [{R['ci'][0]:+.4f},{R['ci'][1]:+.4f}]")
