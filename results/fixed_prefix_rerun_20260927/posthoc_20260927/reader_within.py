"""Post-hoc (NOT preregistered): within-question AUROC of the TF-IDF reader of the same 48 tokens, out-of-fold
by question over all splits, next to the hidden-state probe's (controls.json)."""
import json, sys
import numpy as np

ROOT = "/home/moloch/ouro_project"
R1 = ROOT + "/opi/preanswer/fixed_prefix_rerun_20260927"
sys.path[:0] = [R1, ROOT + "/opi/preanswer/gsm8k_fixed_prefix_20260927"]
import analyze_run1 as A1  # noqa: E402
import analyze_run3 as A3  # noqa: E402
from bg_steering_suite_lib import mcq_prompt  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import FeatureUnion  # noqa: E402

recs = A1.load()
A1.answer_free_table(recs)
N = 48
P = [r for r in recs if not r["malformed"] and r["free"][N]]
i = A1.GRID.index(N)
X = [f"Prompt:\n{mcq_prompt(r['task_prompt'], r['options'])}\n\nVisible reasoning prefix:\n{r['prefix_texts'][i]}" for r in P]
Xp = [r["prefix_texts"][i] for r in P]  # the generated tokens only, without the prompt
y = np.array([int(r["success"]) for r in P]); g = np.array([r["task_uid"] for r in P])


def vec():
    return FeatureUnion([
        ("w", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), strip_accents="unicode", sublinear_tf=True, max_features=40000)),
        ("c", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=60000))])


def oof_reader(texts):
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


out = {}
for name, texts in (("reader_prompt_plus_prefix", X), ("reader_prefix_only", Xp)):
    s = oof_reader(texts)
    out[name] = {"auroc_across_all": round(roc_auc_score(y, s), 3), "within_question": A3.within_task_auroc(s, y, list(g))}
    print(name, json.dumps(out[name]), flush=True)
json.dump(out, open(R1 + "/posthoc_20260927/reader_within.json", "w"), indent=2, default=float)
