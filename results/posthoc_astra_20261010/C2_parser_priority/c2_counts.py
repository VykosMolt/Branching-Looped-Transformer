"""Post hoc (2026-10-10, Astra audit correction 5): executed (format-priority) vs chronological labels on the saved
generations of Run 1 (strict rule), Run 4, Run 3 and Run 6. Text only; shards are read one at a time from the SSD and
reduced to metadata (cache/meta_run*.pkl). Writes c2_counts.json and c2_examples.md.

Cohorts: 'primary' = labelled under the executed rule and answer-free at the run's N* with the run's sealed detector
(Run 1 strict row and Run 4: Horizon detector at N*=48; Run 3 and Run 6: GSM8K detector at N*=96 with the executed own
answer). 'heldout' = split == heldout. For GSM8K the detector uses the candidate's own answer, so we also count how many
answers would change answer-free status at N* if the own answer were the chronological one."""
import collections
import glob
import json
import os
import pickle
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from chrono_parsers import first_commitment, first_commitment_chrono, strict_answer, strict_answer_chrono  # noqa: E402
import answer_detector_gsm8k as G  # noqa: E402

ROOT = "/home/moloch/ouro_project"
SSD = "/run/media/moloch/ARCH_BACKUP/OURO_COLD_ARCHIVE_20260927/tree/opi/preanswer"
sys.path.insert(0, ROOT + "/opi/preanswer/horizon_cohort2_20261003")
from answer_detector import is_answer_bearing as hz_bearing, task_statement  # noqa: E402  (sha = Run 1's detector)

RUNS = {
    "run4": (SSD + "/horizon_cohort2_20261003/run4_shard_*.pt", 48, 3),
    "run3": (SSD + "/gsm8k_fixed_prefix_20260927/run3_shard_*.pt", 96, 5),
    "run6": (SSD + "/gsm8k_train_20261003/run6_shard_*.pt", 96, 5),
}
KEEP = ("task_uid", "task_id", "candidate_idx", "split", "generated_text", "options", "gold_letter", "gold", "prefix_texts",
        "task_prompt", "strict_answer", "labelled", "success", "own_answer", "parseable", "malformed", "parsed_answer")


def meta(run):
    path = os.path.join(HERE, "cache", f"meta_{run}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    if run == "run1":
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "C1_first_set_clean"))
        import r1data
        recs = [{k: r[k] for k in KEEP if k in r} for r in r1data.load()]
    else:
        recs = []
        for f in sorted(glob.glob(RUNS[run][0])):
            d = torch.load(f, map_location="cpu", weights_only=False)
            assert not d.get("smoke") and max(d["equivalence_checks_maxabs"] or [0.0]) <= 1e-3, f
            recs += [{k: r[k] for k in KEEP if k in r} for r in d["records"]]
            del d
            print(run, os.path.basename(f), len(recs), flush=True)
    pickle.dump(recs, open(path, "wb"))
    return recs


def snippet(text, a, b, w=70):
    lo, hi = max(0, min(a, b) - 25), min(len(text), max(a, b) + w)
    s = text[lo:hi].replace("\n", "\\n")
    return ("..." if lo else "") + s + ("..." if hi < len(text) else "")


def first_pos_horizon(text, options, which):
    import chrono_parsers as C
    hits = []
    for prio, (pat, get) in enumerate(C._horizon_patterns(options)):
        m = pat.search(text)
        if m:
            hits.append((m.start(), prio, get(m)))
    hits.sort()
    ex = strict_answer(text, options)
    pos_chrono = hits[0][0]
    pos_exec = min(h[0] for h in hits if h[2] == ex)
    return pos_exec, pos_chrono


def first_pos_gsm(text):
    hits = sorted((m.start(), prio, m.group(1).strip()) for prio, pat in enumerate(G.PATTERNS.values()) for m in [pat.search(text)] if m)
    ex = first_commitment(text)
    return min(h[0] for h in hits if h[2] == ex), hits[0][0]


def analyse(run):
    recs = meta(run)
    gsm = run in ("run3", "run6")
    n_star, gi = (48, 3) if run == "run1" else RUNS[run][1:]
    rows, ex_lines = [], []
    stored_mismatch = 0
    for r in recs:
        if gsm:
            ex, ch = first_commitment(r["generated_text"]), first_commitment_chrono(r["generated_text"])
            stored_mismatch += ex != r["own_answer"]
            lab_ex, lab_ch = G.value(ex) is not None, G.value(ch) is not None
            succ = lambda a: G.value(a) is not None and G.value(a) == G.value(r["gold"])
            free_ex = not G.is_answer_bearing(r["prefix_texts"][gi], ex)
            free_ch = not G.is_answer_bearing(r["prefix_texts"][gi], ch)
            stored_mismatch += (lab_ex != r["parseable"]) + (lab_ex and succ(ex) != r["success"])
        else:
            ex, ch = strict_answer(r["generated_text"], r["options"]), strict_answer_chrono(r["generated_text"], r["options"])
            if run == "run4":
                stored_mismatch += (ex != r["strict_answer"]) + ((ex is not None) != r["labelled"]) + ((ex is not None) and (ex == r["gold_letter"]) != r["success"])
            lab_ex, lab_ch = ex is not None, ch is not None
            succ = lambda a: a == r["gold_letter"]
            free_ex = free_ch = not hz_bearing(r["prefix_texts"][gi], task_statement(r["task_prompt"]))
        rows.append({"lab_ex": lab_ex, "lab_ch": lab_ch, "label_diff": lab_ex and lab_ch and ex != ch,
                     "value_diff": lab_ex and lab_ch and (G.value(ex) != G.value(ch) if gsm else ex != ch),
                     "succ_ex": lab_ex and succ(ex), "succ_ch": lab_ch and succ(ch), "primary": lab_ex and free_ex,
                     "primary_ch": lab_ch and free_ch, "heldout": r["split"] == "heldout"})
        if rows[-1]["label_diff"]:
            pe, pc = first_pos_gsm(r["generated_text"]) if gsm else first_pos_horizon(r["generated_text"], r["options"], None)
            gold = r["gold"] if gsm else r["gold_letter"]
            x = rows[-1]
            ex_lines.append(((x["succ_ex"] == x["succ_ch"], not x["value_diff"], not x["primary"]),f"- [{r.get('task_id', r.get('task_uid'))} #{r['candidate_idx']}, {r['split']}, "
                             f"primary={rows[-1]['primary']}] executed {ex!r} (char {pe}), chronological {ch!r} (char {pc}), gold {gold!r}: "
                             f"`{snippet(r['generated_text'], pc, pe)}`"))

    def count(sel):
        rs = [x for x in rows if sel(x)]
        lab = [x for x in rs if x["lab_ex"]]
        dif = [x for x in lab if x["label_diff"]]
        flips = collections.Counter(f"{x['succ_ex']}->{x['succ_ch']}" for x in dif if x["succ_ex"] != x["succ_ch"])
        return {"n_answers": len(rs), "n_labelled_executed": len(lab), "n_label_string_differs": len(dif),
                "n_label_value_differs": sum(x["value_diff"] for x in lab),
                "n_labelled_status_changes": sum(x["lab_ex"] != x["lab_ch"] for x in rs),
                "n_correctness_flips": sum(flips.values()), "flips": dict(flips)}
    res = {"all": count(lambda x: True), "primary": count(lambda x: x["primary"]), "heldout": count(lambda x: x["heldout"]),
           "primary_heldout": count(lambda x: x["primary"] and x["heldout"]),
           "n_star": n_star, "executed_parser_vs_stored_labels_mismatches": stored_mismatch}
    if gsm:
        res["primary_membership_changes_with_chrono_own_answer"] = sum(x["primary"] != x["primary_ch"] for x in rows)
    ex_lines.sort(key=lambda t: t[0])
    return res, [l for _, l in ex_lines]


if __name__ == "__main__":
    runs = sys.argv[1:] or ["run1", "run4", "run3", "run6"]
    path = os.path.join(HERE, "c2_counts.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    md = {}
    for run in runs:
        res, ex = analyse(run)
        out[run] = res | {"n_examples_total": len(ex), "examples": ex[:5]}
        print(run, json.dumps(res, indent=1), flush=True)
        for l in ex[:5]:
            print("  ", l[:400])
    json.dump(out, open(path, "w"), indent=2)
    with open(os.path.join(HERE, "c2_examples.md"), "w") as f:
        f.write("# Executed vs chronological label: examples (post hoc, 2026-10-10)\n\n")
        for run, res in out.items():
            f.write(f"## {run}\n\n" + ("\n".join(res["examples"]) or "(no differences)") + "\n\n")
