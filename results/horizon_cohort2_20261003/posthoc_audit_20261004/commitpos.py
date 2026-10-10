import pickle, re, numpy as np
meta = pickle.load(open("r4cache/meta.pkl", "rb"))
d = pickle.load(open("repro_primary.pkl", "rb"))
ho = d["ho_idx"]; s = d["sc"]["hidden"]; y = d["sc"]["y"].astype(int); t = d["sc"]["tasks"]
lead = r"(?i:final\s*answer)[\s\*]*[:\-]?[\s\*\(\[<\"']*"
def pos(r):
    tx, opts = r["generated_text"], r["options"]; L = "".join(sorted(opts))
    ms = [m for m in (re.search(lead + rf"([{L}])\b", tx), re.search(rf"\\boxed\{{\s*([{L}])\s*\}}", tx)) if m]
    if ms: return min(m.start() for m in ms)
    w = {v.strip().rstrip(".").lower() for v in opts.values()}
    m = re.search(lead + r"(" + "|".join(map(re.escape, w)) + r")\b", tx, flags=re.I); return m.start()
P = np.array([pos(meta[i]) for i in ho])
by = {}
for k, q in enumerate(t): by.setdefault(q, []).append(k)
pairs = [(p, n) for q, ix in by.items() for p in ix if y[p] == 1 for n in ix if y[n] == 0]
print("within-question AUROC of 'commits earlier' (char offset of label):", np.mean([1 if P[p] < P[n] else .5 if P[p] == P[n] else 0 for p, n in pairs]))
for nm, ps in (("correct commits earlier", [(p, n) for p, n in pairs if P[p] < P[n]]), ("correct commits later", [(p, n) for p, n in pairs if P[p] > P[n]])):
    print(nm, len(ps), "hidden AUROC", round(np.mean([1 if s[p] > s[n] else 0 for p, n in ps]), 3) if ps else None)
from scipy.stats import spearmanr
print("held-out Spearman(hidden score, commit offset) across answers:", round(spearmanr(s, P)[0], 3))
