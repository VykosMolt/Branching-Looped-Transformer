from rows import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
rows = primary
y = np.array([int(r["success"]) for r in rows])
X = [r["prefix_texts"][3] for r in rows]   # generated tokens only: within question the prompt is constant
v = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True, min_df=5)
M = v.fit_transform(X); m = LogisticRegression(C=1.0, solver="liblinear").fit(M, y)
w = m.coef_[0]; names = np.array(v.get_feature_names_out())
o = np.argsort(w)
print("most 'incorrect':", list(names[o[:25]]))
print("most 'correct':", list(names[o[-25:]]))
# simple surface pattern: prefix starts by continuing the option list ("D." on first line) or truncated marker
import re
pat = {"starts_with_invented_option": lambda t: bool(re.match(r"\s*[D-H][\.\)]", t)),
       "truncated_marker_FINAL_ANSW": lambda t: bool(re.search(r"FINAL ANSW(?!ER)", t)),
       "okay_lets_see": lambda t: bool(re.search(r"\b(okay|let me|let's)\b", t, re.I))}
for k, f in pat.items():
    h = np.array([f(t) for t in X]); print(k, "share", round(h.mean(), 3), "acc if present", round(y[h].mean(), 3), "acc if absent", round(y[~h].mean(), 3))
