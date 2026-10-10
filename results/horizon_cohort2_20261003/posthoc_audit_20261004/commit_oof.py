import pickle, re, numpy as np, sys
sys.path.insert(0, ".")
from rows import primary
O = np.load("oof_scores.npy"); names = ("hidden", "combined", "surface", "reader")
lead = r"(?i:final\s*answer)[\s\*]*[:\-]?[\s\*\(\[<\"']*"
def pos(r):
    tx, opts = r["generated_text"], r["options"]; L = "".join(sorted(opts))
    ms = [m for m in (re.search(lead + rf"([{L}])\b", tx), re.search(rf"\\boxed\{{\s*([{L}])\s*\}}", tx)) if m]
    if ms: return min(m.start() for m in ms)
    w = {v.strip().rstrip(".").lower() for v in opts.values()}
    return re.search(lead + r"(" + "|".join(map(re.escape, w)) + r")\b", tx, flags=re.I).start()
y = np.array([int(r["success"]) for r in primary]); g = [r["task_id"] for r in primary]; P = np.array([pos(r) for r in primary])
by = {}
for k, t in enumerate(g): by.setdefault(t, []).append(k)
pairs = [(p, n) for t, ix in by.items() for p in ix if y[p] == 1 for n in ix if y[n] == 0]
f = lambda s, ps: round(np.mean([1 if s[p] > s[n] else .5 if s[p] == s[n] else 0 for p, n in ps]), 3)
print("pairs", len(pairs), "| AUROC of 'commits later':", f(P, pairs))
for nm, ps in (("correct commits later", [(p, n) for p, n in pairs if P[p] > P[n]]), ("correct commits earlier", [(p, n) for p, n in pairs if P[p] < P[n]])):
    print(nm, len(ps), {k: f(O[i], ps) for i, k in enumerate(names)})
