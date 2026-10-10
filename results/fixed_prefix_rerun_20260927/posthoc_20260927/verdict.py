"""Post-hoc: is the 48-token probe reading the eventual verdict? (a) predict the verdict from hidden states;
(b) correctness AUROC within each eventual verdict; (c) proper combination with verdict (+ difficulty) in S."""
import sys, json, torch, numpy as np
R1 = "/home/moloch/ouro_project/opi/preanswer/fixed_prefix_rerun_20260927"
sys.path.insert(0, R1)
import analyze_run1 as A
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
sys.path.insert(0, "/tmp/claude-1000/-home-moloch-Documents-Research-ouro-Operational-Proto-Introspection/07561c8b-32be-4f56-b8e6-e1f42d49fe0a/scratchpad/posthoc")
from proper_combo_lib import fit_select  # noqa

recs = A.load(); A.answer_free_table(recs)
N = 48
P = [r for r in recs if not r["malformed"] and r["free"][N]]
V = ["True", "False", "Unknown"]
v = np.array([V.index(r["options"][r["parsed_answer"]].strip().rstrip(".").capitalize()) for r in P])
H = torch.stack([A.feat_at(N)(r).flatten().float() for r in P]).numpy()
y = np.array([int(r["success"]) for r in P]); g = np.array([r["task_uid"] for r in P])
tv = np.array([r["split"] in ("train", "val") for r in P]); ho = ~tv
rng = np.random.default_rng(20260724)

# (a) predict the eventual verdict from hidden states at 48 tokens
clf = make_pipeline(StandardScaler(), PCA(24, random_state=0), LogisticRegression(C=0.1, max_iter=5000)).fit(H[tv], v[tv])
pv = clf.predict_proba(H[ho])
print(f"(a) verdict from hidden: held-out accuracy {accuracy_score(v[ho], pv.argmax(1)):.3f} (majority {np.bincount(v[ho]).max() / ho.sum():.3f}); "
      f"one-vs-rest AUROC " + ", ".join(f"{V[k]} {roc_auc_score(v[ho] == k, pv[:, k]):.3f}" for k in range(3)))

# (b) the sealed-style hidden probe's held-out scores, AUROC within each eventual verdict
res, sc, y_ho, t_ho = A.run_arm("primary", P, A.feat_at(N), A.prefix_lp_vec(N))
s_h = sc["hidden"].numpy(); vh = v[ho]; th = g[ho]
def boot_auc(s, yy, tt, rounds=2000):
    u = np.unique(tt); idx = {t: np.where(tt == t)[0] for t in u}; vals = []
    for _ in range(rounds):
        pick = np.concatenate([idx[t] for t in rng.choice(u, len(u))])
        if 0 < yy[pick].mean() < 1:
            vals.append(roc_auc_score(yy[pick], s[pick]))
    return np.percentile(vals, [2.5, 97.5])
print(f"(b) all held-out: hidden AUROC {roc_auc_score(y[ho], s_h):.3f}")
for k in range(3):
    m = vh == k
    if 0 < y[ho][m].mean() < 1:
        lo, hi = boot_auc(s_h[m], y[ho][m], th[m])
        print(f"    eventual verdict {V[k]}: n {m.sum()} incorrect {int((y[ho][m] == 0).sum())} hidden AUROC {roc_auc_score(y[ho][m], s_h[m]):.3f} [{lo:.3f}, {hi:.3f}]")
    else:
        print(f"    eventual verdict {V[k]}: n {m.sum()}, one class only")

# (c) proper combination with verdict in S
onehot = np.eye(3)[v]
lp = np.stack([A.prefix_lp_vec(N)(r).numpy() for r in P]); dif = np.array([[r["prompt_tok"], r["proof_depth"]] for r in P], float)
for name, S in (("verdict", onehot), ("verdict + depth + prompt + lp", np.concatenate([onehot, dif, lp], 1))):
    s_s, s_c, b = fit_select(H, S, y, g, tv)
    print(f"(c) S = {name}: surface {roc_auc_score(y[ho], s_s):.3f} combined(PCA(H)+S) {roc_auc_score(y[ho], s_c):.3f} "
          f"increase {roc_auc_score(y[ho], s_c) - roc_auc_score(y[ho], s_s):+.3f} [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}]", flush=True)
