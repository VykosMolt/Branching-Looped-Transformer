"""Post hoc (2026-10-10, Astra audit correction 2): Run 1 analyses with and without the in-window detector misses.

--exclude none      : gate, reproduces the published numbers with the unmodified analysis code.
--exclude inwindow  : the same, with every primary-cohort answer whose 48-token prefix contains an accepted label
                      pattern (inwindow_r1.json: the 15) removed from the cohort before anything is fitted.
                      Hyperparameter selection, folds and bootstraps are refit as in the original code.
Items (code reused unchanged; only the cohort changes):
 1. sealed primary: analyze_run1.run_arm on the N*=48 cohort (as analyze_run1.main)
 5. TF-IDF reader across questions: analyze_run1.visible_reader (as analyze_run1.main)
 4. difficulty check: analyze_run3.proper_arm with S = lp + prompt tokens + proof depth (posthoc_20260927/controls.py,
    'primary_fixed_model_with_difficulty'); its held-out hidden scores give the held-out within-question AUROC
 3. within-question, out of fold over all splits: hidden via analyze_run3._fit_score in GroupKFold(5) (controls.py,
    paired_within.py), TF-IDF reader via reader_within.oof_reader (copied verbatim below), paired bootstrap (paired_within.py)
 2. strict-label row: labels from gen_run4.strict_answer on the whole generation, then analyze_run1.run_arm
    (opi_posthoc_20261003/strict_labels_run1_final_rule.py); with --exclude inwindow, strict-cohort answers whose
    48-token prefix contains a strict_answer match are removed.
Writes c1_<exclude>.json."""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r1data  # noqa: E402

r1data.setup_paths()
sys.path.insert(1, r1data.R3)
import analyze_run1 as A1  # noqa: E402  sealed (SSD, read-only; sha256 checked in README)
import analyze_run3 as A3  # noqa: E402  sealed Run 3 estimators
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "C2_parser_priority"))
from chrono_parsers import strict_answer  # noqa: E402  = gen_run4.strict_answer, unchanged
from bg_steering_suite_lib import mcq_prompt  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import FeatureUnion  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--exclude", choices=["none", "inwindow"], required=True)
EXC = ap.parse_args().exclude
N = 48
key = lambda r: (r["task_uid"], r["candidate_idx"])


def brief(res):
    b = res["paired_task_clustered_bootstrap_combined_minus_shortcut"]["ci95"]
    return {"surface": res["auroc_shortcut_only"], "hidden": res["auroc_hidden_only"], "both": res["auroc_hidden_plus_shortcuts"],
            "increase": res["incremental_auroc_combined_minus_shortcut"], "ci95": b, "n_heldout": res["n_heldout"],
            "n_heldout_tasks": res["n_heldout_tasks"], "n_heldout_negatives": res["n_heldout_negatives"],
            "n_trainval": res["n_trainval"], "hyperparams": res["hyperparams"]}


# reader_within.py, verbatim except that y and g are arguments instead of module globals
def vec():
    return FeatureUnion([
        ("w", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), strip_accents="unicode", sublinear_tf=True, max_features=40000)),
        ("c", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=60000))])


def oof_reader(texts, y, g):
    texts = np.array(texts, dtype=object); oof = np.zeros(len(texts))
    for a_, b_ in GroupKFold(5).split(texts, y, g):
        best = None
        for C in (0.01, 0.1, 1.0, 10.0):  # inner grouped CV on the training part
            inner = np.zeros(len(a_))
            for c_, d_ in GroupKFold(5).split(a_, y[a_], g[a_]):
                v = vec().fit(texts[a_][c_])
                m = LogisticRegression(C=C, solver="liblinear", max_iter=2000).fit(v.transform(texts[a_][c_]), y[a_][c_])
                inner[d_] = m.predict_proba(v.transform(texts[a_][d_]))[:, 1]
            a = roc_auc_score(y[a_], inner)
            best = (a, C) if best is None or a > best[0] else best
        v = vec().fit(texts[a_])
        m = LogisticRegression(C=best[1], solver="liblinear", max_iter=2000).fit(v.transform(texts[a_]), y[a_])
        oof[b_] = m.predict_proba(v.transform(texts[b_]))[:, 1]
    return oof


recs = r1data.load()
for r in recs:
    r["task_id"] = r["task_uid"]  # as controls.py
A1.answer_free_table(recs)
n_star, _ = A1.choose_n_star(recs)
assert n_star == N
bad = set()
if EXC == "inwindow":
    iw = json.load(open(os.path.join(HERE, "inwindow_r1.json")))
    bad = {(d["task_uid"], d["candidate_idx"]) for d in iw["inwindow"]}
    assert len(bad) == 15
out = {"exclude": EXC, "n_excluded_from_published_cohort": len(bad)}
parse = [r for r in recs if not r["malformed"]]
primary = [r for r in parse if r["free"][N] and key(r) not in bad]
out["cohort"] = {"n_answers": len(primary), "n_questions": len({r["task_uid"] for r in primary}),
                 "n_incorrect": sum(not r["success"] for r in primary)}
print("cohort", out["cohort"], flush=True)

# 1 + 5: sealed primary and reader (analyze_run1.main)
res, sc, y_ho, t_ho = A1.run_arm("primary", primary, A1.feat_at(N), A1.prefix_lp_vec(N))
out["item1_sealed_primary"] = brief(res) | {"verdict": A1.verdict(res)}
vr = A1.visible_reader(primary, N, sc["combined"], y_ho, t_ho)
out["item5_reader_across"] = {"reader": vr["auroc_visible_reader"], "C": vr["C"], "hidden_plus_surface": vr["auroc_hidden_plus_surface"],
                              "diff": vr["paired_diff_hidden_plus_surface_minus_visible"], "ci95": vr["bootstrap"]["ci95"]}
print("item1", json.dumps(out["item1_sealed_primary"], default=float), "\nitem5", json.dumps(out["item5_reader_across"]), flush=True)

# 4: difficulty (controls.py section 2) and held-out within-question
diff = lambda r: torch.tensor([float(r["prompt_tok"]), float(r["proof_depth"])])
S = lambda r: torch.cat([A1.prefix_lp_vec(N)(r), diff(r)])
res4, sc4 = A3.proper_arm("primary_difficulty", primary, [A1.feat_at(N)], S)
out["item4_difficulty"] = {"baseline": res4["auroc_shortcut_only"], "hidden": res4["auroc_hidden_only"],
                           "combined": res4["auroc_hidden_plus_shortcuts"], "increase": res4["incremental_auroc_combined_minus_shortcut"],
                           "ci95": res4["paired_task_clustered_bootstrap_combined_minus_shortcut"]["ci95"],
                           "n_heldout": res4["n_heldout"], "n_heldout_negatives": res4["n_heldout_negatives"], "hyperparams": res4["hyperparams"]}
out["item3b_within_heldout"] = {k: A3.within_task_auroc(sc4[k].numpy(), sc4["y"].numpy(), sc4["tasks"]) for k in ("combined", "hidden", "shortcut")}
print("item4", json.dumps(out["item4_difficulty"], default=float), "\nitem3b", json.dumps(out["item3b_within_heldout"]), flush=True)

# 3a: within-question out of fold, all splits (controls.py / reader_within.py / paired_within.py)
y = np.array([int(r["success"]) for r in primary]); g = np.array([r["task_id"] for r in primary])
H = np.stack([A1.feat_at(N)(r).flatten().float().numpy() for r in primary])
hid = np.zeros(len(primary))
for a_, b_ in GroupKFold(5).split(H, y, g):
    tv = np.zeros(len(primary), bool); tv[a_] = True
    hid[b_] = A3._fit_score([H], np.zeros((len(primary), 0)), y, g, tv, ~tv)[0]
del H
X = [f"Prompt:\n{mcq_prompt(r['task_prompt'], r['options'])}\n\nVisible reasoning prefix:\n{r['prefix_texts'][3]}" for r in primary]
rdr = oof_reader(X, y, g)
by = {}
for i, t in enumerate(g):
    by.setdefault(t, []).append(i)
mixed = sorted(t for t, ix in by.items() if 0 < y[ix].mean() < 1)
pairs = {t: [(p, n) for p in by[t] if y[p] == 1 for n in by[t] if y[n] == 0] for t in mixed}
wq = lambda s, ts: np.mean([1.0 if s[p] > s[n] else 0.5 if s[p] == s[n] else 0.0 for t in ts for p, n in pairs[t]])
rng = np.random.default_rng(20260724)
d = sorted(wq(hid, ts) - wq(rdr, ts) for ts in (rng.choice(mixed, len(mixed)) for _ in range(2000)))
out["item3a_within_oof"] = {"hidden": A3.within_task_auroc(hid, y, list(g)), "reader": A3.within_task_auroc(rdr, y, list(g)),
                            "reader_auroc_across_all": roc_auc_score(y, rdr),
                            "paired_hidden_minus_reader": {"diff": wq(hid, mixed) - wq(rdr, mixed), "ci95": [d[50], d[1949]]},
                            "n_questions": len(mixed), "n_pairs": sum(len(v) for v in pairs.values())}
print("item3a", json.dumps(out["item3a_within_oof"], default=float), flush=True)

# 2: strict-label row (strict_labels_run1_final_rule.py)
for r in recs:
    a = strict_answer(r["generated_text"], r["options"])
    r["malformed"], r["success"] = a is None, a == r["gold_letter"]
strict_cohort = [r for r in recs if not r["malformed"] and r["free"][N]]
bad2 = {key(r) for r in strict_cohort if strict_answer(r["prefix_texts"][3], r["options"]) is not None} if EXC == "inwindow" else set()
p2 = [r for r in strict_cohort if key(r) not in bad2]
res2 = A1.run_arm("primary_strict", p2, A1.feat_at(N), A1.prefix_lp_vec(N))[0]
out["item2_strict_row"] = brief(res2) | {"n_excluded": len(bad2), "excluded_overlap_with_15": len(bad2 & bad),
                                         "cohort": {"n_answers": len(p2), "n_questions": len({r["task_uid"] for r in p2}),
                                                    "n_incorrect": sum(not r["success"] for r in p2)}}
print("item2", json.dumps(out["item2_strict_row"], default=float), flush=True)
json.dump(out, open(os.path.join(HERE, f"c1_{EXC}.json"), "w"), indent=2, default=float)
print("saved", f"c1_{EXC}.json")
