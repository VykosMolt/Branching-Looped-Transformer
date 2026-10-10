"""Post hoc (2026-10-10): Run 3 adjudicability under chronological labels (sealed minimums: >=30 held-out tasks and
>=30 held-out incorrect in the N*=96 primary cohort). Text only, from cache/meta_run3.pkl. Writes run3_adjudicability.json."""
import json, os, pickle, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from chrono_parsers import first_commitment, first_commitment_chrono
import answer_detector_gsm8k as G
recs = pickle.load(open(os.path.join(HERE, "cache", "meta_run3.pkl"), "rb"))
out = {}
for cond, lab, det in (("executed", first_commitment, first_commitment), ("chrono_labels", first_commitment_chrono, first_commitment),
                       ("chrono_all", first_commitment_chrono, first_commitment_chrono)):
    ho = []
    for r in recs:
        own, own_det = lab(r["generated_text"]), det(r["generated_text"])
        if G.value(own) is not None and not G.is_answer_bearing(r["prefix_texts"][5], own_det) and r["split"] == "heldout":
            ho.append((r["task_id"], G.value(own) == G.value(r["gold"])))
    out[cond] = {"n_heldout": len(ho), "n_heldout_tasks": len({t for t, _ in ho}), "n_heldout_incorrect": sum(not s for _, s in ho)}
    out[cond]["adjudicable"] = out[cond]["n_heldout_tasks"] >= 30 and out[cond]["n_heldout_incorrect"] >= 30
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(HERE, "run3_adjudicability.json"), "w"), indent=1)
