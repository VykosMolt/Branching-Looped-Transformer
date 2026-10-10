import pickle, numpy as np, sys
sys.path.insert(0, "/home/moloch/ouro_project/opi/preanswer/gsm8k_fixed_prefix_20260927")
meta = pickle.load(open("r4cache/meta.pkl", "rb"))
d = pickle.load(open("repro_primary.pkl", "rb"))
ho = d["ho_idx"]; s = d["sc"]["hidden"]; y = d["sc"]["y"].astype(int); t = d["sc"]["tasks"]
L = np.array([meta[i]["n_gen_tok"] for i in ho]); cap = np.array([meta[i]["hit_max_tokens"] for i in ho])
by = {}
for k, q in enumerate(t): by.setdefault(q, []).append(k)
pairs = [(p, n) for q, ix in by.items() for p in ix if y[p] == 1 for n in ix if y[n] == 0]
print("pairs", len(pairs))
print("within-question AUROC of shorter total generation:", np.mean([1 if L[p] < L[n] else .5 if L[p] == L[n] else 0 for p, n in pairs]))
print("hidden within-question AUROC:", np.mean([1 if s[p] > s[n] else 0 for p, n in pairs]))
a = [(p, n) for p, n in pairs if L[p] < L[n]]; b = [(p, n) for p, n in pairs if L[p] >= L[n]]
for nm, ps in (("correct one shorter", a), ("correct one same/longer", b)):
    print(nm, len(ps), "hidden AUROC", round(np.mean([1 if s[p] > s[n] else 0 for p, n in ps]), 3))
print("pairs where wrong one hit the 448 cap:", sum(cap[n] for p, n in pairs), " correct one hit cap:", sum(cap[p] for p, n in pairs))
# score-length association within question among all held-out answers
from scipy.stats import spearmanr
r = [spearmanr(s[ix], L[ix])[0] for q, ix in by.items() if len(ix) >= 3 and len(set(L[ix])) > 1]
print("mean within-question Spearman(hidden score, total length):", round(np.nanmean(r), 3), "over", len(r), "questions")
