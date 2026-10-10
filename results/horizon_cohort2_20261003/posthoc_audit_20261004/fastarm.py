"""Equivalent of analyze_run3.proper_arm/_fit_score with PCA cached per (fold, block); top-24 PCs via the Gram matrix."""
import itertools, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def pca24(X):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    G = Z.astype(np.float64) @ Z.T.astype(np.float64)
    w, U = np.linalg.eigh(G)
    w, U = w[::-1][:24], U[:, ::-1][:, :24]
    vt = ((U.T @ Z.astype(np.float64)) / np.sqrt(w)[:, None]).astype(np.float32)
    return lambda Q, kk: ((Q - mu) / sd) @ vt[:kk].T

class Arm:
    def __init__(self, blocks, S, y, folds_tv, itv, iho):
        self.blocks, self.S, self.y, self.itv, self.iho = blocks, S, y, itv, iho
        self.folds = folds_tv  # list of (train positions, test positions) within itv
        self.cache = {}
    def _p(self, key, b, tr):
        if (key, b) not in self.cache:
            self.cache[(key, b)] = pca24(self.blocks[b][tr])
        return self.cache[(key, b)]
    def design(self, key, tr, te, bidx, ks, useS):
        A, B = [], []
        for b, k in zip(bidx, ks):
            f = self._p(key, b, tr); A.append(f(self.blocks[b][tr], k)); B.append(f(self.blocks[b][te], k))
        if useS:
            A.append(self.S[tr]); B.append(self.S[te])
        A, B = np.concatenate(A, 1), np.concatenate(B, 1)
        mu, sd = A.mean(0), A.std(0) + 1e-6
        return (A - mu) / sd, (B - mu) / sd
    def fit_score(self, bidx, useS, force=None):
        grid = [(ks, C) for ks in itertools.product((16, 24), repeat=len(bidx)) for C in (0.01, 0.1, 1.0)]
        y, itv = self.y, self.itv
        if force is None:
            oof = {cfg: np.zeros(len(itv)) for cfg in grid}
            for fi, (a_, b_) in enumerate(self.folds):
                tr, te = itv[a_], itv[b_]
                for ks in {c[0] for c in grid}:
                    Xtr, Xte = self.design(fi, tr, te, bidx, ks, useS)
                    for C in (0.01, 0.1, 1.0):
                        oof[(ks, C)][b_] = LogisticRegression(C=C, max_iter=5000).fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
            cv = {cfg: roc_auc_score(y[itv], oof[cfg]) for cfg in grid}
            best = max(grid, key=lambda c: cv[c])
        else:
            best, cv = force, {force: float("nan")}
        Xtr, Xte = self.design("final", itv, self.iho, bidx, best[0], useS)
        p = LogisticRegression(C=best[1], max_iter=5000).fit(Xtr, y[itv]).predict_proba(Xte)[:, 1]
        return p, {"k": list(best[0]), "C": best[1], "cv": cv[best]}
