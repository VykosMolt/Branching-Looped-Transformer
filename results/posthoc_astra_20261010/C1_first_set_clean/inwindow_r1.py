"""Post hoc (2026-10-10, Astra audit correction 2). Re-derives the Run 1 in-window detector misses with the
definition of horizon_cohort2_20261003/posthoc_audit_20261004/r1_inwindow.py, scans the same prefixes with the
chronological parser (C2), and checks whether the sealed 75% rule would still pick N* = 48 had they been flagged.
Writes inwindow_r1.json."""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r1data  # noqa: E402

r1data.setup_paths()
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "C2_parser_priority"))
import analyze_run1 as A  # noqa: E402  sealed Run 1 analysis (SSD, read-only)
from answer_detector import is_answer_bearing, task_statement  # noqa: E402
from chrono_parsers import horizon_any_match, strict_answer  # noqa: E402

recs = r1data.load()
A.answer_free_table(recs)
key = lambda r: (r["task_uid"], r["candidate_idx"])

# r1_inwindow.py definition: primary cohort (not malformed, not flagged at N=48 = prefix_texts[3]) with strict_answer(prefix) not None
prim = [r for r in recs if not r["malformed"] and not is_answer_bearing(r["prefix_texts"][3], task_statement(r["task_prompt"]))]
assert [key(r) for r in prim] == [key(r) for r in recs if not r["malformed"] and r["free"][48]]
inw = [r for r in prim if strict_answer(r["prefix_texts"][3], r["options"]) is not None]
chrono = [r for r in prim if horizon_any_match(r["prefix_texts"][3], r["options"])]
out = {
    "n_primary_cohort": len(prim),
    "n_inwindow_r1_inwindow_definition": len(inw),
    "inwindow": [{"task_uid": r["task_uid"], "candidate_idx": r["candidate_idx"], "split": r["split"], "success": bool(r["success"]),
                  "in_window_answer": strict_answer(r["prefix_texts"][3], r["options"]), "published_label": r["parsed_answer"],
                  "gold": r["gold_letter"], "prefix48_tail": r["prefix_texts"][3][-160:]} for r in inw],
    "by_split_success": {f"{k[0]}|{k[1]}": v for k, v in collections.Counter((r["split"], bool(r["success"])) for r in inw).items()},
    "in_window_answer_equals_published_label": sum(strict_answer(r["prefix_texts"][3], r["options"]) == r["parsed_answer"] for r in inw),
    "chronological_scan_n": len(chrono),
    "chronological_scan_extra_beyond_inwindow": sorted(set(map(key, chrono)) - set(map(key, inw))),
    "chronological_scan_missing_vs_inwindow": sorted(set(map(key, inw)) - set(map(key, chrono))),
}

# N* rule: training answer-free share at N=48 with the misses counted as answer-bearing
train = [r for r in recs if r["split"] == "train"]
free48 = [r for r in train if r["free"][48]]
inw_keys = set(map(key, inw))
miss_train = [r for r in free48 if key(r) in inw_keys]
any_train = [r for r in free48 if strict_answer(r["prefix_texts"][3], r["options"]) is not None]  # incl. malformed answers
n_star, frac = A.choose_n_star(recs)
out["n_star_rule"] = {
    "sealed_n_star": n_star, "train_share_48_sealed": frac[48], "n_train": len(train), "n_train_free48": len(free48),
    "a_the_15_flagged": {"n_train_among_15": len(miss_train), "train_share_48": (len(free48) - len(miss_train)) / len(train)},
    "b_all_train_answers_with_inwindow_strict_answer_flagged_incl_malformed": {
        "n": len(any_train), "train_share_48": (len(free48) - len(any_train)) / len(train)},
    "threshold": A.COVERAGE,
}
for k in ("a_the_15_flagged", "b_all_train_answers_with_inwindow_strict_answer_flagged_incl_malformed"):
    out["n_star_rule"][k]["still_chooses_48"] = out["n_star_rule"][k]["train_share_48"] >= A.COVERAGE
json.dump(out, open(os.path.join(HERE, "inwindow_r1.json"), "w"), indent=2)
print(json.dumps({k: v for k, v in out.items() if k != "inwindow"}, indent=1))
for r in out["inwindow"]:
    print(r["split"], r["success"], r["in_window_answer"], r["published_label"], r["gold"], repr(r["prefix48_tail"][-90:]))
