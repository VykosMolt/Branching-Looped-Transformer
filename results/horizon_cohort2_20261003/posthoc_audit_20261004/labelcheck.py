import pickle, re, collections
meta = pickle.load(open("r4cache/meta.pkl", "rb"))
lead = r"(?i:final\s*answer)[\s\*]*[:\-]?[\s\*\(\[<\"']*"
c = collections.Counter(); ex = []
for r in meta:
    t, opts = r["generated_text"], r["options"]
    letters = "".join(sorted(opts))
    ml = [m for m in (re.search(lead + rf"([{letters}])\b", t), re.search(rf"\\boxed\{{\s*([{letters}])\s*\}}", t)) if m]
    words = {v.strip().rstrip(".").lower(): k for k, v in opts.items()}
    mw = re.search(lead + r"(" + "|".join(map(re.escape, words)) + r")\b", t, flags=re.I)
    if ml and mw:
        m1 = min(ml, key=lambda m: m.start())
        if mw.start() < m1.start():
            wl = words[mw.group(1).lower()]
            same = wl == m1.group(1).upper()
            c["word_first_then_letter", "same" if same else "DIFFERENT"] += 1
            if not same:
                c["diff_and_free48_unknownyet"] += 1
                ex.append((r["task_uid"], r["candidate_idx"], r["split"], wl, m1.group(1), r["gold_letter"]))
    # position of the label relative to total text
    if r["labelled"]:
        c["labelled"] += 1
print(dict(c))
for e in ex[:15]: print(e)
# how do flips matter: would 'first stated' label change correctness?
flip = sum(1 for e in ex if (e[3] == e[5]) != (e[4] == e[5]))
print("examples where the alternative first-word label would flip correctness:", flip, "of", len(ex))
print("of which held-out:", sum(1 for e in ex if e[2] == "heldout"))
