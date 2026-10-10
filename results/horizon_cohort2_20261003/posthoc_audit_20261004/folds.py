from rows import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
rows = primary
y = np.array([int(r["success"]) for r in rows]); g = np.array([r["task_id"] for r in rows])
tv = np.array([r["split"] in ("train", "val") for r in rows]); itv = np.where(tv)[0]
S = np.stack([S48(r).float().numpy() for r in rows])
def gkf(groups, order):
    ug, gi = np.unique(groups, return_inverse=True); n = np.bincount(gi)
    idx = order(n)
    w = np.zeros(5); g2f = np.zeros(len(ug))
    for k in idx:
        f = np.argmin(w); w[f] += n[k]; g2f[k] = f
    lab = g2f[gi]
    return [(np.where(lab != f)[0], np.where(lab == f)[0]) for f in range(5)]
def cv_base(folds):
    out = {}
    for C in (0.01, 0.1, 1.0):
        oof = np.zeros(len(itv))
        for a_, b_ in folds:
            tr, te = itv[a_], itv[b_]
            mu, sd = S[tr].mean(0), S[tr].std(0) + 1e-6
            m = LogisticRegression(C=C, max_iter=5000).fit((S[tr] - mu) / sd, y[tr])
            oof[b_] = m.predict_proba((S[te] - mu) / sd)[:, 1]
        out[C] = round(roc_auc_score(y[itv], oof), 6)
    return out
orders = {
  "sklearn default (laptop)": None,
  "stable argsort, reversed": lambda n: np.argsort(n, kind="stable")[::-1],
  "stable on -n (ties by group index asc)": lambda n: np.argsort(-n, kind="stable"),
}
for name, o in orders.items():
    folds = list(GroupKFold(5).split(itv, y[itv], g[itv])) if o is None else gkf(g[itv], o)
    print(name, cv_base(folds))
rng = np.random.default_rng(0)
for s in range(6):
    perm_tie = lambda n, s=s: np.lexsort((np.random.default_rng(s).permutation(len(n)), -n))
    print("random tie order", s, cv_base(gkf(g[itv], perm_tie)))
print("published pod: baseline C=0.1 cv 0.710469 selected")
