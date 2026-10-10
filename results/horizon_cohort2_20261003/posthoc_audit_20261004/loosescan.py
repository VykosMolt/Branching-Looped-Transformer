import pickle, re, collections, sys
sys.path.insert(0, "/home/moloch/ouro_project/opi/preanswer/horizon_cohort2_20261003")
from answer_detector import is_answer_bearing, task_statement, _LIST, statement_forms
meta = pickle.load(open("r4cache/meta.pkl", "rb"))
LOOSE = {
  "optword_anywhere": re.compile(r"\b(true|false|unknown)\b", re.I),
  "undetermined": re.compile(r"\b(undetermin\w*|indetermin\w*|insufficient|not enough information|can'?t (be )?(tell|say)|no way to|uncertain|not (be )?(determined|known|derivable|provable))\b", re.I),
  "must_be": re.compile(r"\bmust (not )?be\b", re.I),
  "conclusion_word": re.compile(r"\b(therefore|thus|hence|consequently|so the statement|which means)\b", re.I),
}
c = collections.Counter(); hits = {}
prim = [r for r in meta if r["labelled"] and not is_answer_bearing(r["prefix_texts"][3], task_statement(r["task_prompt"]))]
print("primary cohort", len(prim))
for r in prim:
    t = _LIST.sub(" OPTION_LIST ", r["prefix_texts"][3])
    h = [k for k, p in LOOSE.items() if p.search(t)]
    hits[r["i"] if "i" in r else (r["task_uid"], r["candidate_idx"])] = h
    for k in h: c[k] += 1
    if h: c["any"] += 1
print({k: (v, round(v / len(prim), 3)) for k, v in c.items()})
# show a few optword examples
n = 0
for r in prim:
    t = _LIST.sub(" OPTION_LIST ", r["prefix_texts"][3])
    m = LOOSE["optword_anywhere"].search(t)
    if m and n < 12:
        s = max(0, m.start() - 90); print(repr(t[s:m.end() + 40])); n += 1
pickle.dump({(r["task_uid"], r["candidate_idx"]): [k for k, p in LOOSE.items() if p.search(_LIST.sub(" OPTION_LIST ", r["prefix_texts"][3]))] for r in prim}, open("loose_hits.pkl", "wb"))
