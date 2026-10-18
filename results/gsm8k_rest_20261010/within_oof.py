"""Within-question, out-of-fold analysis shared by Run 7 (GSM8K) and Run 8 (Horizon, Ouro-Thinking); sealed with both.

- Folds: 5 outer folds by sha256("oof5|" + task_id) % 5; inside each outer training set, 5 inner folds for the
  hyperparameter choice by sha256("inner5|" + task_id) % 5. No GroupKFold (whose tie order depends on the machine).
- Probe: analyze_run3's hidden-state probe (each hidden block standardized and reduced by PCA, k in {16, 24}; S
  appended for the combined model; standardized; L2-logistic regression, C in {0.01, 0.1, 1}; best inner-CV AUROC).
  `fit_score` is analyze_run3._fit_score with the folds passed in and the PCA cached per (fold, block).
- PCA: analyze_run3._pca (standardize with training statistics, top right singular vectors), with the top 24
  computed by ARPACK (scipy svds, fixed start vector, tol=0) instead of a full SVD. Same subspace to ~1e-15 (cosine,
  14,000 x 24,576 test); a full SVD at that size takes about an hour per fit on 7-24 CPUs.
- Within-question AUROC: share of (correct, incorrect) answer pairs to the same question ranked correctly, ties 1/2,
  pairs-weighted, 2,000-draw question-clustered bootstrap with seed 20260724 (analyze_run3.within_task_auroc).
"""
from __future__ import annotations

import hashlib
import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "opi/preanswer/gsm8k_fixed_prefix_20260927"))
from analyze_run3 import within_task_auroc  # noqa: E402,F401  (re-exported: the sealed estimator)

SEED, ROUNDS, K = 20260724, 2000, 5


def hash_fold(task_id: str, salt: str = "oof5") -> int:
    return int(hashlib.sha256(f"{salt}|{task_id}".encode()).hexdigest(), 16) % K


def hash_folds(tasks, salt: str):
    lab = np.array([hash_fold(t, salt) for t in tasks])
    return [(np.where(lab != f)[0], np.where(lab == f)[0]) for f in range(K)]


def pca(X, k: int = 24):
    """analyze_run3._pca with the top k right singular vectors from ARPACK (exact to tolerance, deterministic)."""
    from scipy.sparse.linalg import svds
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = ((X - mu) / sd).astype(np.float64)
    _, s, vt = svds(Z, k=k, v0=np.ones(min(Z.shape)), tol=0, maxiter=100 * min(Z.shape))
    vt = vt[np.argsort(-s)]
    return lambda Q, kk: ((Q - mu) / sd) @ vt[:kk].T


def fit_score(blocks, S, y, itr, ite, inner, cache: dict, key):
    """analyze_run3._fit_score: hyperparameters by CV over `inner` (positions within itr), refit on itr, score ite."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    def proj(tag, b, rows):
        if (key, tag, b) not in cache:
            cache[(key, tag, b)] = pca(blocks[b][rows])
        return cache[(key, tag, b)]

    def design(tag, tr, te, ks):
        A = [proj(tag, b, tr)(blocks[b][tr], k) for b, k in enumerate(ks)] + [S[tr]]
        B = [proj(tag, b, tr)(blocks[b][te], k) for b, k in enumerate(ks)] + [S[te]]
        A, B = np.concatenate(A, 1), np.concatenate(B, 1)
        mu, sd = A.mean(0), A.std(0) + 1e-6
        return (A - mu) / sd, (B - mu) / sd

    grid = [(ks, C) for ks in itertools.product((16, 24), repeat=len(blocks)) for C in (0.01, 0.1, 1.0)]
    oof = {cfg: np.zeros(len(itr)) for cfg in grid}
    for fi, (a, b) in enumerate(inner):
        tr, te = itr[a], itr[b]
        for ks in {cfg[0] for cfg in grid}:
            Xtr, Xte = design(fi, tr, te, ks)
            for C in (0.01, 0.1, 1.0):
                oof[(ks, C)][b] = LogisticRegression(C=C, max_iter=5000).fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
    cv = {cfg: roc_auc_score(y[itr], oof[cfg]) for cfg in grid}
    best = max(grid, key=lambda c: cv[c])
    Xtr, Xte = design("final", itr, ite, best[0])
    p = LogisticRegression(C=best[1], max_iter=5000).fit(Xtr, y[itr]).predict_proba(Xte)[:, 1]
    return p, {"k": list(best[0]), "C": best[1], "cv_auroc": cv[best]}


def oof_scores(blocks, S, y, tasks, arms=("hidden", "combined", "surface")):
    """Every answer scored by a model fit on the other four hash folds. hidden = PCA(H); combined = PCA(H) + S;
    surface = S only."""
    tasks, y = np.asarray(tasks), np.asarray(y)
    none = np.zeros((len(y), 0))
    spec = {"hidden": (blocks, none), "combined": (blocks, S), "surface": ([], S)}
    out, hp = {a: np.full(len(y), np.nan) for a in arms}, {a: [] for a in arms}
    for f, (itr, ite) in enumerate(hash_folds(tasks, "oof5")):
        cache, inner = {}, hash_folds(tasks[itr], "inner5")
        for a in arms:
            out[a][ite], h = fit_score(*spec[a], y, itr, ite, inner, cache, f)
            hp[a].append(h)
    assert all(np.isfinite(v).all() for v in out.values())
    return out, hp


def reader_oof(texts, y, tasks):
    """TF-IDF reader of the visible text (word 1-2 + char 3-5 grams), text only, scored out of fold with the same
    hash folds; C in {0.01, 0.1, 1, 10} by inner CV (the 2026-10-04 audit's reader, oof_within.py)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import FeatureUnion

    def vec():
        return FeatureUnion([
            ("w", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), strip_accents="unicode", sublinear_tf=True, max_features=40000)),
            ("c", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=60000))])

    def lr(C):
        return LogisticRegression(C=C, solver="liblinear", max_iter=2000, random_state=0)
    X, y, tasks = np.asarray(texts, dtype=object), np.asarray(y), np.asarray(tasks)
    out = np.full(len(y), np.nan)
    for itr, ite in hash_folds(tasks, "oof5"):
        inner = hash_folds(tasks[itr], "inner5")
        inn = {C: np.zeros(len(itr)) for C in (0.01, 0.1, 1.0, 10.0)}
        for a, b in inner:
            v = vec().fit(X[itr][a])
            Xa, Xb = v.transform(X[itr][a]), v.transform(X[itr][b])
            for C in inn:
                inn[C][b] = lr(C).fit(Xa, y[itr][a]).predict_proba(Xb)[:, 1]
        C = max(inn, key=lambda c: roc_auc_score(y[itr], inn[c]))
        v = vec().fit(X[itr])
        out[ite] = lr(C).fit(v.transform(X[itr]), y[itr]).predict_proba(v.transform(X[ite]))[:, 1]
    return out


def _pairs(y, tasks):
    by = {}
    for i, t in enumerate(tasks):
        by.setdefault(t, []).append(i)
    mixed = sorted(t for t, ix in by.items() if 0 < np.mean([y[i] for i in ix]) < 1)
    return mixed, {t: [(p, n) for p in by[t] if y[p] == 1 for n in by[t] if y[n] == 0] for t in mixed}


def _win(s, p, n):
    return 1.0 if s[p] > s[n] else 0.5 if s[p] == s[n] else 0.0


def _boot(stat, mixed, rng):
    return sorted(stat(rng.choice(mixed, len(mixed))) for _ in range(ROUNDS))


def paired_diff(s1, s2, y, tasks) -> dict:
    """Within-question AUROC of s1 minus s2 on the same pairs; the same question resamples for both."""
    mixed, pairs = _pairs(np.asarray(y), list(tasks))
    w = {t: (sum(_win(s1, p, n) - _win(s2, p, n) for p, n in pairs[t]), len(pairs[t])) for t in mixed}
    stat = lambda ts: sum(w[t][0] for t in ts) / sum(w[t][1] for t in ts)
    d = _boot(stat, mixed, np.random.default_rng(SEED))
    return {"diff": stat(mixed), "ci95": [d[int(0.025 * ROUNDS)], d[int(0.975 * ROUNDS)]], "n_mixed_tasks": len(mixed)}


def unpaired_diff(a: dict, b: dict) -> dict:
    """Within-question AUROC of a minus b (dicts with s, y, tasks), independent question resamples for each."""
    rng, sides = np.random.default_rng(SEED), []
    for x in (a, b):
        mixed, pairs = _pairs(np.asarray(x["y"]), list(x["tasks"]))
        w = {t: (sum(_win(x["s"], p, n) for p, n in pairs[t]), len(pairs[t])) for t in mixed}
        stat = lambda ts, w=w: sum(w[t][0] for t in ts) / sum(w[t][1] for t in ts)
        sides.append((stat(mixed), _boot(stat, mixed, rng)))
    d = sorted(u - v for u, v in zip(sides[0][1], sides[1][1]))
    return {"diff": sides[0][0] - sides[1][0], "a": sides[0][0], "b": sides[1][0],
            "ci95": [d[int(0.025 * ROUNDS)], d[int(0.975 * ROUNDS)]]}


def commit_timing(scores: dict, y, tasks, pos) -> dict:
    """pos = where the answer's label commitment starts (character offset in the whole generation, a future event
    beyond the cut). Share of pairs where the correct answer commits later; within-question AUROC of commit position
    alone; each score's pair-level AUROC split by which answer commits first."""
    y, pos = np.asarray(y), np.asarray(pos, dtype=float)
    mixed, pairs = _pairs(y, list(tasks))
    allp = [pn for t in mixed for pn in pairs[t]]
    groups = {"correct_commits_later": [(p, n) for p, n in allp if pos[p] > pos[n]],
              "correct_commits_earlier": [(p, n) for p, n in allp if pos[p] < pos[n]],
              "same_position": [(p, n) for p, n in allp if pos[p] == pos[n]]}
    return {"n_pairs": len(allp), "share_correct_commits_later": len(groups["correct_commits_later"]) / max(len(allp), 1),
            "auroc_commit_position_alone": within_task_auroc(pos, y, list(tasks)),
            "by_group": {g: {"n_pairs": len(ps), **{k: (float(np.mean([_win(s, p, n) for p, n in ps])) if ps else None)
                                                    for k, s in scores.items()}} for g, ps in groups.items()}}


def verdict(wq: dict, min_mixed: int) -> str:
    """Fixed in both PLANs. Applied in this order, so a CI inside (0.5, 0.55) reads WITHIN_QUESTION_SIGNAL."""
    if wq.get("n_mixed_tasks", 0) < min_mixed:
        return "UNDERPOWERED"
    lo, hi = wq["ci95"]
    return "WITHIN_QUESTION_SIGNAL" if lo > 0.5 else "NO_WITHIN_QUESTION_SIGNAL_ABOVE_0.55" if hi < 0.55 else "UNRESOLVED"
