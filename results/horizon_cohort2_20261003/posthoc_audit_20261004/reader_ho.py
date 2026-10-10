from rows import *
import pickle
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion
from bg_steering_suite_lib import mcq_prompt
rows = primary
y = np.array([int(r["success"]) for r in rows]); g = np.array([r["task_id"] for r in rows])
tv = np.array([r["split"] in ("train", "val") for r in rows]); itv, iho = np.where(tv)[0], np.where(~tv)[0]
X = np.array([f"Prompt:\n{mcq_prompt(r['task_prompt'], r['options'])}\n\nVisible reasoning prefix:\n{r['prefix_texts'][3]}" for r in rows], dtype=object)
vec = lambda: FeatureUnion([("w", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), strip_accents="unicode", sublinear_tf=True, max_features=40000)),
                            ("c", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=60000))])
best = None
for C in (0.01, 0.1, 1.0, 10.0):
    oof = np.zeros(len(itv))
    for a_, b_ in GroupKFold(5).split(itv, y[itv], g[itv]):
        v = vec().fit(X[itv[a_]]); m = LogisticRegression(C=C, solver="liblinear", max_iter=2000, random_state=0).fit(v.transform(X[itv[a_]]), y[itv[a_]])
        oof[b_] = m.predict_proba(v.transform(X[itv[b_]]))[:, 1]
    a = roc_auc_score(y[itv], oof); best = (a, C) if best is None or a > best[0] else best
v = vec().fit(X[itv]); m = LogisticRegression(C=best[1], solver="liblinear", max_iter=2000, random_state=0).fit(v.transform(X[itv]), y[itv])
s = m.predict_proba(v.transform(X[iho]))[:, 1]
print("C", best, "held-out across AUROC", round(roc_auc_score(y[iho], s), 3))
print("held-out within-question reader:", A3.within_task_auroc(s, y[iho], list(g[iho])))
d = pickle.load(open("repro_primary.pkl", "rb")); h = d["sc"]["hidden"]
assert list(d["ho_idx"]) == [rows[i]["i"] for i in iho]
by = {}
for k, t in enumerate(g[iho]): by.setdefault(t, []).append(k)
yy = y[iho]; mixed = sorted(t for t, ix in by.items() if 0 < yy[ix].mean() < 1)
pairs = {t: [(p, n) for p in by[t] if yy[p] == 1 for n in by[t] if yy[n] == 0] for t in mixed}
wq = lambda sc, ts: np.mean([1.0 if sc[p] > sc[n] else 0.5 if sc[p] == sc[n] else 0.0 for t in ts for p, n in pairs[t]])
rng = np.random.default_rng(20260724)
dd = sorted(wq(h, ts) - wq(s, ts) for ts in (rng.choice(mixed, len(mixed)) for _ in range(2000)))
print("paired hidden - reader on the 73 held-out pairs: %.3f [%.3f, %.3f]" % (wq(h, mixed) - wq(s, mixed), dd[50], dd[1949]))
