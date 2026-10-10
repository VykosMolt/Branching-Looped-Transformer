"""reader_within.py oof_reader, verbatim except y and g are arguments (same as in c1_analysis.py)."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion


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
