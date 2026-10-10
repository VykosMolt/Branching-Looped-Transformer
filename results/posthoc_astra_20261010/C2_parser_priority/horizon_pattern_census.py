"""Post hoc (2026-10-10): why executed and chronological Horizon labels never differ. Counts answers in which each
accepted pattern (marker+letter, boxed letter, marker+option word) matches anywhere, and in which a word match
precedes the first letter/boxed match (the only case where the two parsers can differ). Writes horizon_pattern_census.json."""
import json, os, pickle, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import chrono_parsers as C
out = {}
for run in ("run1", "run4"):
    c = dict(n=0, letter=0, boxed=0, word=0, word_and_letter_or_boxed=0, word_before_letter_or_boxed=0)
    for r in pickle.load(open(os.path.join(HERE, "cache", f"meta_{run}.pkl"), "rb")):
        m = [p.search(r["generated_text"]) for p, _ in C._horizon_patterns(r["options"])]
        c["n"] += 1; c["letter"] += bool(m[0]); c["boxed"] += bool(m[1]); c["word"] += bool(m[2])
        lb = [x.start() for x in m[:2] if x]
        if m[2] and lb:
            c["word_and_letter_or_boxed"] += 1; c["word_before_letter_or_boxed"] += m[2].start() < min(lb)
    out[run] = c
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(HERE, "horizon_pattern_census.json"), "w"), indent=1)
