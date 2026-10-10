import pickle, re, collections, sys
sys.path.insert(0, "/home/moloch/ouro_project/opi/preanswer/horizon_cohort2_20261003")
sys.path[:0] = ["/home/moloch/ouro_project", "/home/moloch/ouro_project/shared", "/home/moloch/ouro_project/shared/utilities/tests/manual"]
from answer_detector import is_answer_bearing, task_statement
from gen_run4 import strict_answer
meta = pickle.load(open("r4cache/meta.pkl", "rb"))
GRID = (0, 16, 32, 48, 64, 96, 128)
out = {}
for N in (16, 32, 48, 64, 96, 128):
    k = GRID.index(N)
    prim = [r for r in meta if r["labelled"] and not is_answer_bearing(r["prefix_texts"][k], task_statement(r["task_prompt"]))]
    inw = [r for r in prim if strict_answer(r["prefix_texts"][k], r["options"]) is not None]
    agree = sum(strict_answer(r["prefix_texts"][k], r["options"]) == r["strict_answer"] for r in inw)
    c = collections.Counter((r["split"], r["success"]) for r in inw)
    print(f"N={N}: cohort {len(prim)}, label found inside window {len(inw)} ({len(inw)/len(prim):.3%}); same as label {agree}; by split/success {dict(c)}")
    out[N] = [(r["task_uid"], r["candidate_idx"]) for r in inw]
pickle.dump(out, open("inwindow.pkl", "wb"))
k = 3
for r in [r for r in meta if r["labelled"] and not is_answer_bearing(r["prefix_texts"][k], task_statement(r["task_prompt"])) and strict_answer(r["prefix_texts"][k], r["options"])][:6]:
    print(repr(r["prefix_texts"][k][:260]), "| label", r["strict_answer"], "gold", r["gold_letter"])
