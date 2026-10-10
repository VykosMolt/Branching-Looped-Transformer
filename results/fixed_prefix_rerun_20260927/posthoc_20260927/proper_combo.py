"""Post-hoc: a combined model that can use the surface features (PCA on hidden only, then append S)."""
import sys, json, torch, numpy as np
R1 = "/home/moloch/ouro_project/opi/preanswer/fixed_prefix_rerun_20260927"
sys.path.insert(0, R1)
import analyze_run1 as A
from bg_v2_overnight_horizon_analysis import bootstrap_task_clustered
from proto_introspection_controls_analysis import auroc
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

recs = A.load(); A.answer_free_table(recs)
N = 48
primary = [r for r in recs if not r["malformed"] and r["free"][N]]
lp = A.prefix_lp_vec(N)
S_sets = {"lp": lambda r: lp(r), "depth+prompt": lambda r: torch.tensor([float(r["prompt_tok"]), float(r["proof_depth"])]),
          "lp+depth+prompt": lambda r: torch.cat([lp(r), torch.tensor([float(r["prompt_tok"]), float(r["proof_depth"])])])}
H = torch.stack([A.feat_at(N)(r).flatten().float() for r in primary]).numpy()
y = np.array([int(r["success"]) for r in primary])
g = np.array([r["task_id"] if "task_id" in r else r["task_uid"] for r in primary])
tv = np.array([r["split"] in ("train", "val") for r in primary]); ho = ~tv

def fit_select(X, cols_h, cols_s, grid):
    best = None
    for k, C in grid:
        steps = []
        if cols_h:
            steps.append(("h", make_pipeline(StandardScaler(), PCA(n_components=k, random_state=0)), cols_h))
        if cols_s:
            steps.append(("s", StandardScaler(), cols_s))
        model = Pipeline([("ct", ColumnTransformer(steps)), ("lr", LogisticRegression(C=C, max_iter=5000))])
        oof = np.zeros(tv.sum())
        for tr, va in GroupKFold(5).split(X[tv], y[tv], g[tv]):
            model.fit(X[tv][tr], y[tv][tr]); oof[va] = model.predict_proba(X[tv][va])[:, 1]
        a = roc_auc_score(y[tv], oof)
        if best is None or a > best[0]:
            best = (a, k, C)
    _, k, C = best
    steps = []
    if cols_h:
        steps.append(("h", make_pipeline(StandardScaler(), PCA(n_components=k, random_state=0)), cols_h))
    if cols_s:
        steps.append(("s", StandardScaler(), cols_s))
    model = Pipeline([("ct", ColumnTransformer(steps)), ("lr", LogisticRegression(C=C, max_iter=5000))]).fit(X[tv], y[tv])
    return model.predict_proba(X[ho])[:, 1], best

out = {}
grid_h = [(k, C) for k in (16, 24) for C in (0.01, 0.1, 1.0)]
s_h, best_h = fit_select(H, list(range(H.shape[1])), [], grid_h)
print(f"hidden only: AUROC {roc_auc_score(y[ho], s_h):.3f} (cv {best_h[0]:.3f}, k={best_h[1]}, C={best_h[2]})", flush=True)
for name, f in S_sets.items():
    S = torch.stack([f(r).float() for r in primary]).numpy()
    X = np.concatenate([H, S], 1)
    hc, sc_ = list(range(H.shape[1])), list(range(H.shape[1], X.shape[1]))
    s_s, _ = fit_select(X, [], sc_, [(0, C) for C in (0.01, 0.1, 1.0)])
    s_c, best_c = fit_select(X, hc, sc_, grid_h)
    yt = torch.tensor(y[ho]).double(); tt = list(g[ho])
    b = bootstrap_task_clustered(torch.tensor(s_c).double(), torch.tensor(s_s).double(), yt, tt, rounds=2000)
    out[name] = {"surface": roc_auc_score(y[ho], s_s), "combined": roc_auc_score(y[ho], s_c), "boot": b, "best": best_c}
    print(f"S = {name}: surface {roc_auc_score(y[ho], s_s):.3f} combined(PCA(H)+S) {roc_auc_score(y[ho], s_c):.3f} "
          f"increase {roc_auc_score(y[ho], s_c) - roc_auc_score(y[ho], s_s):+.3f} [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}]", flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=2, default=float)
