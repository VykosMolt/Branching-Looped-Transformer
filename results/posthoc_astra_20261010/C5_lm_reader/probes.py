"""CPU only. (1) Reproduce the published TF-IDF readers on the extracted texts (gate: proves the texts, cohort,
order and splits are the ones the TF-IDF reader saw). (2) Recompute the published probes' held-out scores with the
published code (gate: published AUROC). (3) Score the hidden-state probe out of fold over the hash folds.
Writes data/probe_scores.json and data/repro_checks.json. Exits non-zero if any gate fails."""
import argparse
import itertools
import json
import sys

from common import DATA, R1, R4CODE, ROOT, cohort, fold_of, lib_paths, load_rows, N_FOLDS

sys.dont_write_bytecode = True
lib_paths()
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.pipeline import FeatureUnion  # noqa: E402
import analyze_run3 as A3  # noqa: E402
from bg_v2_malformed_sibling_control import arm  # noqa: E402
from bg_v2_overnight_horizon_analysis import SEED  # noqa: E402

# Published values (results.json of each run, controls.json, strict_labels_run1_final_rule.json).
PUB = {
    "r1_tfidf": 0.6405086285195277, "r1_arm_combined": 0.7582803511958826, "r1_arm_hidden": 0.758340902210112,
    "r1_A3_hidden_wq_heldout": 0.7192982456140351, "r1s_arm_combined": 0.7771668219944082,
    "r4_tfidf_plus_S": 0.7200882559422325, "r4_hidden": 0.6833567345301373, "r4_combined": 0.7342041921572561,
    "r4_wq_heldout_hidden": 0.7397260273972602,
}
R4_HP = {"hidden": (24, 1.0), "combined": (24, 1.0)}  # Run 4 results.json primary.hyperparams (pod run)
TOL = 1e-9


def vec():
    return FeatureUnion([
        ("w", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), strip_accents="unicode", sublinear_tf=True, max_features=40000)),
        ("c", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=60000))])


def tfidf_reader(rs, y, S=None):
    """Body of analyze_run1.visible_reader (S=None) / analyze_run4.visible_reader (S given), text = stored text."""
    from scipy.sparse import csr_matrix, hstack
    tv = [i for i, r in enumerate(rs) if r["split"] in ("train", "val")]
    ho = [i for i, r in enumerate(rs) if r["split"] == "heldout"]
    X = [r["text"] for r in rs]
    uniq = sorted({rs[i]["task_uid"] for i in tv})
    order = [uniq[j] for j in torch.randperm(len(uniq), generator=torch.Generator().manual_seed(SEED)).tolist()]
    fold = {u: k % 5 for k, u in enumerate(order)}
    folds = [fold[rs[i]["task_uid"]] for i in tv]

    def add_s(M, fit, use):
        if S is None:
            return M
        mu, sd = S[fit].mean(0), S[fit].std(0) + 1e-6
        return hstack([M, csr_matrix((S[use] - mu) / sd)]).tocsr()
    best = None
    for C in (0.01, 0.1, 1.0, 10.0):
        oof = np.zeros(len(tv))
        for k in range(5):
            tr = [tv[j] for j, f in enumerate(folds) if f != k]
            va_pos = [j for j, f in enumerate(folds) if f == k]
            va = [tv[j] for j in va_pos]
            v = vec().fit([X[i] for i in tr])
            m = LogisticRegression(C=C, solver="liblinear", max_iter=2000, random_state=20260729)
            m.fit(add_s(v.transform([X[i] for i in tr]), tr, tr), y[tr])
            oof[va_pos] = m.predict_proba(add_s(v.transform([X[i] for i in va]), tr, va))[:, 1]
        a = roc_auc_score(y[tv], oof)
        best = (a, C) if best is None or a > best[0] else best
    v = vec().fit([X[i] for i in tv])
    m = LogisticRegression(C=best[1], solver="liblinear", max_iter=2000, random_state=20260729)
    m.fit(add_s(v.transform([X[i] for i in tv]), tv, tv), y[tv])
    s = m.predict_proba(add_s(v.transform([X[i] for i in ho]), tv, ho))[:, 1]
    return s, {"C": best[1], "cv_auroc": best[0], "auroc": roc_auc_score(y[ho], s)}


def arm_scores(H, S, y, g, tv, ho):
    """analyze_run1.run_arm: arm() from bg_v2_malformed_sibling_control, unchanged."""
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()
    yt = torch.tensor(y, dtype=torch.float)
    res = arm("c5_repro", T(H[tv]), T(S[tv]), yt[tv], list(g[tv]), T(H[ho]), T(S[ho]), yt[ho], list(g[ho]))
    return {k: v.numpy() for k, v in res["_scores"].items()}


def fixed_fit(H, S, y, train, test, k, C):
    """analyze_run3 estimator (PCA on hidden block fit on train, then S, standardized) with a given config."""
    Xtr, Xte = A3._design(np.where(train)[0], np.where(test)[0], [H], S, [k])
    return LogisticRegression(C=C, max_iter=5000).fit(Xtr, y[train]).predict_proba(Xte)[:, 1]


def hash_fold_oof(H, y, fold, split):
    """Hidden-only probe. Config (k, C) by 5-fold CV over the hash folds within train+val (pooled OOF AUROC,
    first-best in grid order, as _fit_score); then each hash fold over all splits scored by a fit on the other four."""
    tv = np.isin(split, ("train", "val"))
    Z = np.zeros((len(y), 0))
    cv = {}
    for k, C in itertools.product((16, 24), (0.01, 0.1, 1.0)):
        oof = np.full(len(y), np.nan)
        for f in range(N_FOLDS):
            oof[tv & (fold == f)] = fixed_fit(H, Z, y, tv & (fold != f), tv & (fold == f), k, C)
        cv[(k, C)] = roc_auc_score(y[tv], oof[tv])
    best = max(cv, key=lambda c: cv[c])
    oof = np.zeros(len(y))
    for f in range(N_FOLDS):
        oof[fold == f] = fixed_fit(H, Z, y, fold != f, fold == f, *best)
    return oof, {"k": best[0], "C": best[1], "cv_auroc": cv[best], "cv_table": {f"{k}|{C}": a for (k, C), a in cv.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", required=True)
    a = ap.parse_args()
    rows = load_rows()
    out, checks, fails = {}, {}, []

    def gate(name, got, want, tol=TOL, protocol_gate=True, note=None):
        ok = abs(got - want) <= tol
        checks[name] = {"got": got, "published": want, "ok": ok, "protocol_gate": protocol_gate, **({"note": note} if note else {})}
        print(f"[{'gate' if protocol_gate else 'info'}] {name}: got {got:.12f} published {want:.12f} {'OK' if ok else 'FAIL'}", flush=True)
        if not ok and protocol_gate:
            fails.append(name)

    feats = {run: np.load(f"{a.feat_dir}/{run}.npz") for run in ("r1", "r4")}
    for c in ("r1", "r1s", "r4"):
        rs, y, g, split, fold = cohort(rows, c)
        F = feats[c[:2]]
        idx = np.array([r["feat_idx"] for r in rs])
        H = F["H"][idx]
        LP = F["lp"][idx]
        tv, ho = np.isin(split, ("train", "val")), split == "heldout"
        keys = [r["key"] for r in rs]
        kho = [k for k, m in zip(keys, ho) if m]
        o = out[c] = {"heldout": {}, "oof": {}, "info": {}}
        if c == "r1":
            s, info = tfidf_reader(rs, y)
            gate("r1_tfidf_heldout_auroc", info["auroc"], PUB["r1_tfidf"])
            o["heldout"]["tfidf"] = dict(zip(kho, s.tolist())); o["info"]["tfidf"] = info
        if c == "r4":
            S = np.concatenate([LP, F["sdiff"][idx]], 1)  # analyze_run4: lp_at(48) + s_diff
            s, info = tfidf_reader(rs, y, S)
            gate("r4_tfidf_plus_S_heldout_auroc", info["auroc"], PUB["r4_tfidf_plus_S"])
            o["heldout"]["tfidf_plus_S"] = dict(zip(kho, s.tolist())); o["info"]["tfidf_plus_S"] = info
            sh = fixed_fit(H, np.zeros((len(y), 0)), y, tv, ho, *R4_HP["hidden"])
            sc = fixed_fit(H, S, y, tv, ho, *R4_HP["combined"])
            gate("r4_hidden_heldout_auroc", roc_auc_score(y[ho], sh), PUB["r4_hidden"])
            gate("r4_combined_heldout_auroc", roc_auc_score(y[ho], sc), PUB["r4_combined"])
            gate("r4_hidden_wq_heldout", A3.within_task_auroc(sh, y[ho], list(g[ho]))["estimate"], PUB["r4_wq_heldout_hidden"])
            o["heldout"]["hidden"] = dict(zip(kho, sh.tolist())); o["heldout"]["combined"] = dict(zip(kho, sc.tolist()))
        else:
            sa = arm_scores(H, LP, y, g, tv, ho)
            gate(f"{c}_arm_combined_heldout_auroc", roc_auc_score(y[ho], sa["combined"]), PUB[f"{c}_arm_combined"])
            if c == "r1":
                # Not a protocol gate and not used by C5. On 2026-10-10 this gave 0.758280 (selection picked l2=4):
                # the CV AUROCs of l2 = 1/2/4 lie within 4e-5 and differ from the 2026-09-27 run by ~1e-5, so the
                # pick flips; refit at the published k=16, l2=1 it gives 0.758340902210112, the published value.
                gate("r1_arm_hidden_heldout_auroc", roc_auc_score(y[ho], sa["hidden"]), PUB["r1_arm_hidden"], protocol_gate=False,
                     note="selection near-tie; equals published when refit at published k=16, l2=1; unused by C5")
            sh, hp = A3._fit_score([H], np.zeros((len(y), 0)), y, g, tv, ho)  # controls.py / proper_arm hidden-only
            if c == "r1":
                gate("r1_A3_hidden_wq_heldout", A3.within_task_auroc(sh, y[ho], list(g[ho]))["estimate"], PUB["r1_A3_hidden_wq_heldout"])
            o["heldout"]["arm_combined"] = dict(zip(kho, sa["combined"].tolist()))
            o["heldout"]["arm_hidden"] = dict(zip(kho, sa["hidden"].tolist()))
            o["heldout"]["hidden"] = dict(zip(kho, sh.tolist())); o["info"]["hidden_hp"] = hp
        if c in ("r1", "r4"):
            oof, info = hash_fold_oof(H, y, fold, split)
            o["oof"]["hidden"] = dict(zip(keys, oof.tolist())); o["info"]["oof_hidden"] = info
            print(f"[{c}] hash-fold OOF probe config {info['k']}, {info['C']}", flush=True)
    assert all(fold_of(r["task_uid"]) == r["fold"] for r in rows)
    json.dump(out, open(DATA / "probe_scores.json", "w"))
    json.dump({"checks": checks, "failed": fails}, open(DATA / "repro_checks.json", "w"), indent=1)
    print("FAILED GATES:" if fails else "all gates passed", fails)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
