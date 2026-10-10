"""Post-hoc (NOT preregistered), 2026-10-03: Run 1 with a strict first-commitment label rule.
The published labels come from parse_mcq_answer on the whole generation, which returns the LAST match of
'FINAL ANSWER: <letter>', else the last 'answer is <letter>', else the last '<letter>) ' or '<letter>. ' anywhere.
Ouro-RLTT keeps writing after its answer (invented follow-up problems), and Horizon reasoning uses capital-letter
proposition symbols, so those fallbacks can label an answer from text that is not its answer.
Strict rule (fixed before looking at any outcome): the first 'FINAL ANSWER' followed directly by a letter
(optionally after ** ( [ < or quotes) or a boxed letter, whichever comes first; failing that, the first
'FINAL ANSWER' followed by an option word; otherwise unlabelled. Everything else is Run 1's sealed analysis."""
import collections, json, re, sys
sys.path[:0] = ["/home/moloch/ouro_project", "/home/moloch/ouro_project/shared", "/home/moloch/ouro_project/shared/utilities/tests/manual"]
sys.path.insert(0, "/run/media/moloch/ARCH_BACKUP/OURO_COLD_ARCHIVE_20260927/tree/opi/preanswer/fixed_prefix_rerun_20260927")
import analyze_run1 as A
from bg_v2_overnight_horizon_analysis import shortcut_vec

A.HERE = A.Path("/run/media/moloch/ARCH_BACKUP/OURO_COLD_ARCHIVE_20260927/tree/opi/preanswer/fixed_prefix_rerun_20260927")


def strict_answer_v1(text, options):  # first version (case-sensitive marker); superseded by gen_run4.strict_answer
    letters = "".join(sorted(options))
    lead = r"FINAL\s*ANSWER\s*[:\-]?\s*[\*\(\[<\"']*\s*"
    cands = [m for m in (re.search(lead + rf"([{letters}])\b", text), re.search(rf"\\boxed\{{\s*([{letters}])\s*\}}", text)) if m]
    if cands:
        return min(cands, key=lambda m: m.start()).group(1).upper()
    words = {v.strip().rstrip(".").lower(): k for k, v in options.items()}
    m = re.search(lead + r"(" + "|".join(map(re.escape, words)) + r")\b", text, flags=re.I)
    return words[m.group(1).lower()] if m else None


sys.path.insert(0, "/home/moloch/ouro_project/opi/preanswer/horizon_cohort2_20261003")
from gen_run4 import strict_answer  # final rule
recs = A.load()
A.answer_free_table(recs)
chg = collections.Counter()
for r in recs:
    a = strict_answer(r["generated_text"], r["options"])
    old = None if r["malformed"] else bool(r["success"])
    r["malformed_pub"], r["success_pub"] = r["malformed"], r["success"]
    r["malformed"], r["success"] = a is None, a == r["gold_letter"]
    new = None if r["malformed"] else bool(r["success"])
    chg[(old, new)] += 1
print("label changes (published -> strict):", {f"{k[0]}->{k[1]}": v for k, v in sorted(chg.items(), key=str)}, flush=True)
out = {"label_changes": {f"{k[0]}->{k[1]}": v for k, v in chg.items()}}
N = 48
primary = [r for r in recs if not r["malformed"] and r["free"][N]]
res = A.run_arm("primary_strict", primary, A.feat_at(N), A.prefix_lp_vec(N))[0]
out["primary_strict_labels"] = res
parse = [r for r in recs if not r["malformed"]]
out["final_marker_strict_labels"] = A.run_arm("final_marker_strict", parse, lambda r: r["preanswer_features"], shortcut_vec)[0]
fmt = lambda r: (f"surface {r['auroc_shortcut_only']:.3f} hidden {r['auroc_hidden_only']:.3f} both {r['auroc_hidden_plus_shortcuts']:.3f} "
                 f"increase {r['incremental_auroc_combined_minus_shortcut']:+.3f} {[round(x, 3) for x in r['paired_task_clustered_bootstrap_combined_minus_shortcut']['ci95']]} "
                 f"held-out {r['n_heldout']} answers, {r['n_heldout_negatives']} incorrect")
print("primary, strict labels:", fmt(res)); print("final marker, strict labels:", fmt(out["final_marker_strict_labels"]))
json.dump(out, open("/home/moloch/ouro_project/opi_posthoc_20261003/strict_labels_run1_final_rule.json", "w"), indent=2, default=float)
