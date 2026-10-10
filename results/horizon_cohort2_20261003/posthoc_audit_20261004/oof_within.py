"""Post-hoc: Run 4 within-question AUROC from nested out-of-fold scores over ALL splits (as Run 1's 209-pair analysis),
hidden-state probe vs TF-IDF reader (text only, as Run 1's reader_within), paired by question."""
from rows import *
import json
from fastarm import Arm
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion
from bg_steering_suite_lib import mcq_prompt
rows = primary
y = np.array([int(r["success"]) for r in rows]); g = np.array([r["task_id"] for r in rows])
H = np.stack([np.asarray(F["f48"][r["i"]], dtype=np.float32) for r in rows])
S = np.stack([S48(r).float().numpy() for r in rows])
X = np.array([f"Prompt:\n{mcq_prompt(r['task_prompt'], r['options'])}\n\nVisible reasoning prefix:\n{r['prefix_texts'][3]}" for r in rows], dtype=object)
def vec():
    return FeatureUnion([("w", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), strip_accents="unicode", sublinear_tf=True, max_features=40000)),
                         ("c", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=60000))])
oof = {k: np.zeros(len(rows)) for k in ("hidden", "combined", "surface", "reader")}
for a_, b_ in GroupKFold(5).split(H, y, g):
    inner = list(GroupKFold(5).split(a_, y[a_], g[a_]))
    arm = Arm([H], S, y, inner, a_, b_)
    oof["hidden"][b_] = arm.fit_score([0], False)[0]; oof["combined"][b_] = arm.fit_score([0], True)[0]; oof["surface"][b_] = arm.fit_score([], True)[0]
    best = None
    for C in (0.01, 0.1, 1.0, 10.0):
        inn = np.zeros(len(a_))
        for c_, d_ in inner:
            v = vec().fit(X[a_][c_]); m = LogisticRegression(C=C, solver="liblinear", max_iter=2000, random_state=0).fit(v.transform(X[a_][c_]), y[a_][c_])
            inn[d_] = m.predict_proba(v.transform(X[a_][d_]))[:, 1]
        a = roc_auc_score(y[a_], inn); best = (a, C) if best is None or a > best[0] else best
    v = vec().fit(X[a_]); m = LogisticRegression(C=best[1], solver="liblinear", max_iter=2000, random_state=0).fit(v.transform(X[a_]), y[a_])
    oof["reader"][b_] = m.predict_proba(v.transform(X[b_]))[:, 1]
    print("fold done", flush=True)
res = {k: A3.within_task_auroc(v, y, list(g)) for k, v in oof.items()}
res["across_auroc"] = {k: roc_auc_score(y, v) for k, v in oof.items()}
by = {}
for i, t in enumerate(g): by.setdefault(t, []).append(i)
mixed = sorted(t for t, ix in by.items() if 0 < y[ix].mean() < 1)
pairs = {t: [(p, n) for p in by[t] if y[p] == 1 for n in by[t] if y[n] == 0] for t in mixed}
wq = lambda s, ts: np.mean([1.0 if s[p] > s[n] else 0.5 if s[p] == s[n] else 0.0 for t in ts for p, n in pairs[t]])
rng = np.random.default_rng(20260724)
d = sorted(wq(oof["hidden"], ts) - wq(oof["reader"], ts) for ts in (rng.choice(mixed, len(mixed)) for _ in range(2000)))
res["paired_hidden_minus_reader"] = {"diff": wq(oof["hidden"], mixed) - wq(oof["reader"], mixed), "ci95": [d[50], d[1949]]}
# question-weighted variant (each mixed question counts once)
qw = lambda s: np.mean([np.mean([1.0 if s[p] > s[n] else 0.5 if s[p] == s[n] else 0.0 for p, n in pairs[t]]) for t in mixed])
res["question_weighted"] = {k: qw(v) for k, v in oof.items()}
np.save("oof_scores.npy", np.stack([oof[k] for k in ("hidden", "combined", "surface", "reader")]))
json.dump(res, open("oof_within.json", "w"), indent=1, default=float)
print(json.dumps(res, default=float))
