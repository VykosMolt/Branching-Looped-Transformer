"""Post hoc (2026-10-10, requested after the Astra audit): Run 6 (GSM8K training split) within-question AUROC from
out-of-fold scores over EVERY answer of the primary cohort (labelled, answer-free at N*=96), all splits, following
horizon_cohort2_20261003/posthoc_audit_20261004/oof_within.py and commit_oof.py.

Changes from the template, by request: outer folds = 5 folds assigned by int(sha256(task_id), 16) % 5 (not GroupKFold).
Probe scores use Run 6's sealed estimator unchanged: analyze_run3._fit_score (PCA per hidden block then S; k in {16,24},
C in {0.01,0.1,1}; inner 5-fold task-grouped CV on the outer-training part, exactly as in the sealed code) with
  hidden   = [H96], no S          (proper_arm's 'hidden')
  combined = [H96] + S            (proper_arm's 'combined')
  baseline = S only               (proper_arm's 'shortcut'); S = lp mean/min at 96 + prompt tokens + reference steps.
TF-IDF reader: the template's reader (prompt + first 96 tokens, word 1-2 + char 3-5 grams, C chosen by inner grouped CV,
liblinear, random_state=0), no S. Within-question AUROC and the question-clustered bootstrap (2,000 draws, seed
20260724) from analyze_run3.within_task_auroc; paired hidden-minus-reader as in oof_within.py.
Commit timing: commit position = character offset in the whole answer of the match that gives the label
(executed: the first match of the highest-priority pattern found; chronological: the earliest match).
--labels executed | chrono (chrono: labels from first_commitment_chrono, cohort as executed).
Writes c4_<labels>.json and oof_scores_<labels>.npy (rows in cohort order: hidden, combined, baseline, reader)."""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r6data  # noqa: E402

r6data.setup_paths()
import analyze_run6 as A6  # noqa: E402  sealed
import analyze_run3 as A3  # noqa: E402  sealed
import answer_detector_gsm8k as G  # noqa: E402
from bg_steering_suite_lib import gsm_prompt  # noqa: E402
from chrono_parsers import first_commitment, first_commitment_chrono  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import FeatureUnion  # noqa: E402

# Speed only: _fit_score calls _pca(B[tr]) once per k in {16, 24} on the same training rows, i.e. the same SVD twice.
# A one-entry memo around the unchanged sealed _pca returns the identical closure for an identical input array
# (keyed by a hash of the array bytes), so every number is the same as without it.
_pca_sealed, _memo = A3._pca, {}


def _pca_memo(X, k=24):
    key = (X.shape, k, hashlib.sha1(np.ascontiguousarray(X).view(np.uint8)).hexdigest())
    if key not in _memo:
        _memo.clear()
        _memo[key] = _pca_sealed(X, k)
    return _memo[key]


A3._pca = _pca_memo
ap = argparse.ArgumentParser()
ap.add_argument("--labels", choices=["executed", "chrono"], required=True)
LAB = ap.parse_args().labels
N, GI = 96, r6data.GI


def commit_pos(text, chrono):
    hits = sorted((m.start(), prio, m.group(1).strip()) for prio, pat in enumerate(G.PATTERNS.values()) for m in [pat.search(text)] if m)
    if chrono:
        return hits[0][0]
    ex = first_commitment(text)
    return next(h[0] for h in sorted(hits, key=lambda h: h[1]) if h[2] == ex)  # first match of the winning pattern


recs = r6data.load()
A6.answer_free_table(recs)  # executed own answer -> executed cohort
assert A6.choose_n_star(recs)[0] == N
if LAB == "chrono":
    for r in recs:
        own = first_commitment_chrono(r["generated_text"])
        r["parseable"], r["success"] = G.value(own) is not None, G.value(own) is not None and G.value(own) == G.value(r["gold"])
rows = [r for r in recs if r["parseable"] and r["free"][N]]
y = np.array([int(r["success"]) for r in rows]); g = np.array([r["task_id"] for r in rows])
fold = np.array([int(hashlib.sha256(t.encode()).hexdigest(), 16) % 5 for t in g])
S = np.stack([torch.cat([A6.lp_at(N)(r), A6.s_diff(r)]).float().numpy() for r in rows])
H = np.stack([r["prefix_features"][GI].flatten().float().numpy() for r in rows])
X = np.array([f"Prompt:\n{gsm_prompt(r['question'])}\n\nVisible reasoning prefix:\n{r['prefix_texts'][GI]}" for r in rows], dtype=object)
P = np.array([commit_pos(r["generated_text"], LAB == "chrono") for r in rows])
del recs
print(f"cohort {len(rows)}, incorrect {int((y == 0).sum())}, questions {len(set(g))}, fold sizes {np.bincount(fold).tolist()}", flush=True)


def vec():
    return FeatureUnion([("w", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), strip_accents="unicode", sublinear_tf=True, max_features=40000)),
                         ("c", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=60000))])


oof = {k: np.zeros(len(rows)) for k in ("hidden", "combined", "baseline", "reader")}
hp = []
zeros = np.zeros((len(rows), 0), np.float32)
for k in range(5):
    tv, ho = fold != k, fold == k
    oof["hidden"][ho], h1 = A3._fit_score([H], zeros, y, g, tv, ho)
    oof["combined"][ho], h2 = A3._fit_score([H], S, y, g, tv, ho)
    oof["baseline"][ho], h3 = A3._fit_score([], S, y, g, tv, ho)
    a_, b_ = np.where(tv)[0], np.where(ho)[0]
    inner = list(GroupKFold(5).split(a_, y[a_], g[a_]))
    best = None
    for C in (0.01, 0.1, 1.0, 10.0):
        inn = np.zeros(len(a_))
        for c_, d_ in inner:
            v = vec().fit(X[a_][c_]); m = LogisticRegression(C=C, solver="liblinear", max_iter=2000, random_state=0).fit(v.transform(X[a_][c_]), y[a_][c_])
            inn[d_] = m.predict_proba(v.transform(X[a_][d_]))[:, 1]
        a = roc_auc_score(y[a_], inn); best = (a, C) if best is None or a > best[0] else best
    v = vec().fit(X[a_]); m = LogisticRegression(C=best[1], solver="liblinear", max_iter=2000, random_state=0).fit(v.transform(X[a_]), y[a_])
    oof["reader"][b_] = m.predict_proba(v.transform(X[b_]))[:, 1]
    hp.append({"fold": k, "hidden": h1, "combined": h2, "baseline": h3, "reader_C": best[1], "reader_cv": best[0]})
    print("fold", k, "done", json.dumps(hp[-1], default=float), flush=True)
np.save(os.path.join(HERE, f"oof_scores_{LAB}.npy"), np.stack([oof[k] for k in ("hidden", "combined", "baseline", "reader")]))

res = {"labels": LAB, "n_cohort": len(rows), "n_incorrect": int((y == 0).sum()), "n_questions": len(set(g)),
       "fold_sizes": np.bincount(fold).tolist(), "hyperparams": hp}
res["within"] = {k: A3.within_task_auroc(v, y, list(g)) for k, v in oof.items()}
res["across_auroc"] = {k: roc_auc_score(y, v) for k, v in oof.items()}
by = {}
for i, t in enumerate(g):
    by.setdefault(t, []).append(i)
mixed = sorted(t for t, ix in by.items() if 0 < y[ix].mean() < 1)
pairs = {t: [(p, n) for p in by[t] if y[p] == 1 for n in by[t] if y[n] == 0] for t in mixed}
wq = lambda s, ts: np.mean([1.0 if s[p] > s[n] else 0.5 if s[p] == s[n] else 0.0 for t in ts for p, n in pairs[t]])
rng = np.random.default_rng(20260724)
d = sorted(wq(oof["hidden"], ts) - wq(oof["reader"], ts) for ts in (rng.choice(mixed, len(mixed)) for _ in range(2000)))
res["paired_hidden_minus_reader"] = {"diff": wq(oof["hidden"], mixed) - wq(oof["reader"], mixed), "ci95": [d[50], d[1949]]}
# commit timing (commit_oof.py)
allp = [pq for t in mixed for pq in pairs[t]]
f = lambda s, ps: float(np.mean([1 if s[p] > s[n] else .5 if s[p] == s[n] else 0 for p, n in ps])) if ps else None
later = [(p, n) for p, n in allp if P[p] > P[n]]
earlier = [(p, n) for p, n in allp if P[p] < P[n]]
ties = [(p, n) for p, n in allp if P[p] == P[n]]
res["commit_timing"] = {"n_pairs": len(allp), "share_correct_commits_later": len(later) / len(allp), "auroc_commits_later_alone": f(P, allp),
                        "correct_commits_later": {"n_pairs": len(later), **{k: f(v, later) for k, v in oof.items()}},
                        "correct_commits_earlier": {"n_pairs": len(earlier), **{k: f(v, earlier) for k, v in oof.items()}},
                        "same_position": {"n_pairs": len(ties)}}
json.dump(res, open(os.path.join(HERE, f"c4_{LAB}.json"), "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k != "hyperparams"}, default=float))
