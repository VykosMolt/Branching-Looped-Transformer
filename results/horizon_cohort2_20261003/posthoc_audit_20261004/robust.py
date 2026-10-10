"""Post-hoc robustness: the sealed Run 4 estimators under 30 equally valid GroupKFold tie orders (same algorithm)."""
from rows import *
import json
from fastarm import Arm
from sklearn.model_selection import GroupKFold
from proto_introspection_controls_analysis import auroc
from bg_v2_overnight_horizon_analysis import bootstrap_task_clustered
rows = primary
y = np.array([int(r["success"]) for r in rows]); g = np.array([r["task_id"] for r in rows])
tv = np.array([r["split"] in ("train", "val") for r in rows]); itv, iho = np.where(tv)[0], np.where(~tv)[0]
S = np.stack([S48(r).float().numpy() for r in rows])
H = np.stack([np.asarray(F["f48"][r["i"]], dtype=np.float32) for r in rows])
PE = np.stack([np.asarray(F["pend"][r["i"]], dtype=np.float32) for r in rows])
yh = torch.tensor(y[iho]).double(); th = list(g[iho])
def gkf(groups, seed):
    ug, gi = np.unique(groups, return_inverse=True); n = np.bincount(gi)
    idx = np.lexsort((np.random.default_rng(seed).permutation(len(n)), -n))
    w = np.zeros(5); g2f = np.zeros(len(ug))
    for k in idx:
        f = np.argmin(w); w[f] += n[k]; g2f[k] = f
    lab = g2f[gi]
    return [(np.where(lab != f)[0], np.where(lab == f)[0]) for f in range(5)]
final_cache = {}
out = []
for s in [-1] + list(range(30)):
    folds = list(GroupKFold(5).split(itv, y[itv], g[itv])) if s < 0 else gkf(g[itv], s)
    arm = Arm([H, PE], S, y, folds, itv, iho); arm.cache.update(final_cache)
    T = lambda p: torch.tensor(p, dtype=torch.double)
    pb, hb = arm.fit_score([], True); pc, hc = arm.fit_score([0], True); ph, hh = arm.fit_score([0], False)
    qb, gb = arm.fit_score([1], True); qc, gc = arm.fit_score([1, 0], True)
    final_cache.update({k: v for k, v in arm.cache.items() if k[0] == "final"})
    b1 = bootstrap_task_clustered(T(pc), T(pb), yh, th, rounds=2000)["ci95"]
    b2 = bootstrap_task_clustered(T(qc), T(qb), yh, th, rounds=2000)["ci95"]
    wq_h = A3.within_task_auroc(ph, y[iho], th); wq_c = A3.within_task_auroc(pc, y[iho], th)
    rec = {"seed": s, "base": round(auroc(T(pb), yh), 4), "comb": round(auroc(T(pc), yh), 4), "hid": round(auroc(T(ph), yh), 4),
           "inc": round(auroc(T(pc), yh) - auroc(T(pb), yh), 4), "ci": [round(b1[0], 4), round(b1[1], 4)],
           "pend_base": round(auroc(T(qb), yh), 4), "pend_comb": round(auroc(T(qc), yh), 4),
           "pend_inc": round(auroc(T(qc), yh) - auroc(T(qb), yh), 4), "pend_ci": [round(b2[0], 4), round(b2[1], 4)],
           "wq_hidden": round(wq_h["estimate"], 4), "wq_hidden_ci": [round(x, 3) for x in wq_h["ci95"]],
           "wq_comb": round(wq_c["estimate"], 4), "hp": {"b": hb["C"], "c": (hc["k"], hc["C"]), "h": (hh["k"], hh["C"]),
           "pb": (gb["k"], gb["C"]), "pc": (gc["k"], gc["C"])}}
    out.append(rec); print(json.dumps(rec), flush=True)
json.dump(out, open("robust.json", "w"), indent=1)
inc = np.array([r["inc"] for r in out]); lo = np.array([r["ci"][0] for r in out])
pinc = np.array([r["pend_inc"] for r in out]); plo = np.array([r["pend_ci"][0] for r in out])
wq = np.array([r["wq_hidden"] for r in out])
print("primary inc: median %.4f range [%.4f, %.4f]; CI lower >0 in %d/%d" % (np.median(inc), inc.min(), inc.max(), (lo > 0).sum(), len(lo)))
print("pend inc: median %.4f range [%.4f, %.4f]; CI lower >0 in %d/%d" % (np.median(pinc), pinc.min(), pinc.max(), (plo > 0).sum(), len(plo)))
print("within-question hidden: median %.4f range [%.4f, %.4f]" % (np.median(wq), wq.min(), wq.max()))
