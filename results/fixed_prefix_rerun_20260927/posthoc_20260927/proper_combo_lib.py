import numpy as np, torch
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from bg_v2_overnight_horizon_analysis import bootstrap_task_clustered

def _model(nh, ns, k, C, use_h):
    steps = [("s", StandardScaler(), list(range(nh, nh + ns)))]
    if use_h:
        steps.insert(0, ("h", make_pipeline(StandardScaler(), PCA(n_components=k, random_state=0)), list(range(nh))))
    return Pipeline([("ct", ColumnTransformer(steps)), ("lr", LogisticRegression(C=C, max_iter=5000))])

def _select_fit(X, y, g, tv, nh, ns, use_h):
    grid = [(k, C) for k in ((16, 24) if use_h else (0,)) for C in (0.01, 0.1, 1.0)]
    best = None
    for k, C in grid:
        oof = np.zeros(tv.sum())
        for tr, va in GroupKFold(5).split(X[tv], y[tv], g[tv]):
            m = _model(nh, ns, k, C, use_h).fit(X[tv][tr], y[tv][tr]); oof[va] = m.predict_proba(X[tv][va])[:, 1]
        a = roc_auc_score(y[tv], oof)
        if best is None or a > best[0]:
            best = (a, k, C)
    return _model(nh, ns, best[1], best[2], use_h).fit(X[tv], y[tv]).predict_proba(X[~tv])[:, 1]

def fit_select(H, S, y, g, tv):
    X = np.concatenate([H, S], 1); nh, ns = H.shape[1], S.shape[1]
    s_s = _select_fit(X, y, g, tv, nh, ns, False)
    s_c = _select_fit(X, y, g, tv, nh, ns, True)
    b = bootstrap_task_clustered(torch.tensor(s_c).double(), torch.tensor(s_s).double(), torch.tensor(y[~tv]).double(), list(g[~tv]), rounds=2000)
    return s_s, s_c, b
