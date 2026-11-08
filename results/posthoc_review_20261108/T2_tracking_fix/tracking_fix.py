"""Post hoc, 2026-11-08 (OPI v4 review item 1.1). Not preregistered.

results/fixed_prefix_rerun_20260927/posthoc_20260927/controls.py, part 3, computes the share of
mixed-verdict sibling pairs in which the verdict probe "tracks each answer's own verdict" as
    mean[(p_i(own_i) > p_j(own_i)) + (p_j(own_j) > p_i(own_j))] / 2
Both terms are NumPy bools, and NumPy bool + bool is a logical OR, so a pair where both
comparisons hold counts 0.5 instead of 1 and the statistic cannot exceed 0.5.

This script reruns part 3 exactly (same data, folds, model, RNG and draw order), asserts that the
original expression reproduces the published controls.json values, then computes the intended
statistic, mean[(int(a) + int(b)) / 2], on the same out-of-fold predictions and the same bootstrap
draws. Run 1 shards are read from the archive drive (read-only)."""
import json, sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True

R1 = Path("/run/media/moloch/ARCH_BACKUP/OURO_COLD_ARCHIVE_20260927/tree/opi/preanswer/fixed_prefix_rerun_20260927")
OUT = Path(__file__).resolve().parent
# The original run had ROOT = ~/ouro_project, so its helpers came from ~/ouro_project/shared; the archive's
# shared/ copy lacks them (checked: none of the imported helper modules exist on the archive paths A1 adds).
P0 = Path("/home/moloch/ouro_project")
sys.path[:0] = [str(R1), str(P0), str(P0 / "shared"), str(P0 / "shared/utilities/tests/manual")]
import analyze_run1 as A1  # noqa: E402  (inserts the archive's shared/ paths itself)
from bg_v2_overnight_horizon_analysis import SEED  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

PUB = json.load(open(R1 / "posthoc_20260927/controls.json"))
N = 48
recs1 = A1.load()
for r in recs1:
    r["task_id"] = r["task_uid"]
A1.answer_free_table(recs1)
fresh = [r for r in recs1 if not r["malformed"]]
P = [r for r in fresh if r["free"][N]]
y = np.array([int(r["success"]) for r in P]); g = np.array([r["task_id"] for r in P])
H = np.stack([A1.feat_at(N)(r).flatten().float().numpy() for r in P])
V = ["True", "False", "Unknown"]
norm = lambda t: t.strip().rstrip(".").capitalize()
own = np.array([V.index(norm(r["options"][r["parsed_answer"]])) for r in P])
gold = np.array([V.index(norm(r["options"][r["gold_letter"]])) for r in P])
rng = np.random.default_rng(SEED)


def oof_verdict(train_mask):  # verbatim from controls.py
    pr = np.zeros((len(P), 3))
    for a_, b_ in GroupKFold(5).split(H, own, g):
        tr = np.intersect1d(a_, np.where(train_mask)[0])
        m = make_pipeline(StandardScaler(), PCA(24, random_state=0), LogisticRegression(C=0.1, max_iter=5000)).fit(H[tr], own[tr])
        p = np.zeros((len(b_), 3)); p[:, m.classes_] = m.predict_proba(H[b_]); pr[b_] = p
    return pr


def boot(stat_fn, tasks):  # verbatim from controls.py (consumes the shared rng)
    u = np.unique(tasks); ix = {t: np.where(tasks == t)[0] for t in u}
    vals = sorted(stat_fn(np.concatenate([ix[t] for t in rng.choice(u, len(u))])) for _ in range(2000))
    return [round(vals[50], 3), round(vals[1949], 3)]


wrong = np.where(y == 0)[0]
pr = oof_verdict(np.ones(len(P), bool))
pw, ow, gw, tw = pr[wrong], own[wrong], gold[wrong], g[wrong]
acc_own = lambda ix: float((pw[ix].argmax(1) == ow[ix]).mean())
acc_gold = lambda ix: float((pw[ix].argmax(1) == gw[ix]).mean())
allix = np.arange(len(wrong))
wrong_res = {"n_wrong": len(wrong), "argmax_is_own_verdict": round(acc_own(allix), 3), "ci_own": boot(acc_own, tw),
             "argmax_is_gold_verdict": round(acc_gold(allix), 3), "ci_gold": boot(acc_gold, tw),
             "mean_p_own": round(float(pw[allix, ow].mean()), 3), "mean_p_gold": round(float(pw[allix, gw].mean()), 3)}

pairs = [(i, j) for t in np.unique(g) for i in np.where(g == t)[0] for j in np.where(g == t)[0] if i < j and own[i] != own[j]]
a_ = lambda i, j: pr[i, own[i]] > pr[j, own[i]]   # answer i's own verdict is likelier under i than under its sibling
b_ = lambda i, j: pr[j, own[j]] > pr[i, own[j]]
tr_orig = lambda ps: np.mean([a_(i, j) + b_(i, j) for i, j in ps]) / 2           # original expression (bool OR)
tr_fixed = lambda ps: float(np.mean([(int(a_(i, j)) + int(b_(i, j))) / 2 for i, j in ps]))
both = lambda ps: float(np.mean([bool(a_(i, j)) and bool(b_(i, j)) for i, j in ps]))
neither = lambda ps: float(np.mean([not a_(i, j) and not b_(i, j) for i, j in ps]))
by = {}
for i, j in pairs:
    by.setdefault(g[i], []).append((i, j))
ts = sorted(by)
vo, vf = [], []
for _ in range(2000):
    s = [p for t in rng.choice(ts, len(ts)) for p in by[t]]
    vo.append(tr_orig(s)); vf.append(tr_fixed(s))
vo, vf = sorted(vo), sorted(vf)
orig = {"n_questions": len(ts), "n_pairs": len(pairs), "share_tracking_own_verdict": round(float(tr_orig(pairs)), 3),
        "ci95": [round(vo[50], 3), round(vo[1949], 3)]}

# Gate: the original expression reproduces the published numbers exactly (at their 3-dp rounding).
assert wrong_res == PUB["verdict_on_wrong_answers_trained_on_all"], (wrong_res, PUB["verdict_on_wrong_answers_trained_on_all"])
assert orig == PUB["verdict_siblings_different_verdicts"], (orig, PUB["verdict_siblings_different_verdicts"])
print("gate passed: published controls.json part 3 reproduced exactly", flush=True)

out = {"post_hoc": True, "date": "2026-11-08", "inputs": str(R1), "gate": "controls.json verdict_on_wrong_answers_trained_on_all and verdict_siblings_different_verdicts reproduced exactly",
       "original_expression": orig,
       "corrected": {"n_questions": len(ts), "n_pairs": len(pairs), "share_tracking_own_verdict": round(tr_fixed(pairs), 4),
                     "ci95": [round(vf[50], 4), round(vf[1949], 4)], "chance": 0.5,
                     "both_comparisons_hold": round(both(pairs), 4), "neither_holds": round(neither(pairs), 4),
                     "first_term_holds": round(float(np.mean([bool(a_(i, j)) for i, j in pairs])), 4),
                     "second_term_holds": round(float(np.mean([bool(b_(i, j)) for i, j in pairs])), 4)},
       "verdict_on_wrong_answers_trained_on_all": wrong_res,
       "bootstrap": "2,000 question-clustered draws, same draws for both expressions, percentile 2.5/97.5 as in controls.py"}
json.dump(out, open(OUT / "tracking_fix_results.json", "w"), indent=2)
print(json.dumps(out["corrected"], indent=1))
