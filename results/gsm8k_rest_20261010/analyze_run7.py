"""Run 7 analysis (GSM8K, every unused problem), implementing PLAN.md as sealed. Stage 'audit' is label-free
(integrity, answer-free fractions, what Run 6's 75% rule would choose, hand-audit list at N = 96) and stops; stage
'full' opens labels. Runs on the laptop CPU after the three pods' shards are collected (the pooled secondary also
needs Run 6's shards). Labels, detector, S and helpers come from Run 6 / Run 3 (imported unchanged; hashes in SEAL);
the within-question out-of-fold machinery from within_oof.py. --data/--out/--synthetic exist only for dry runs."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
R3, R6 = ROOT / "opi/preanswer/gsm8k_fixed_prefix_20260927", ROOT / "opi/preanswer/gsm8k_train_20261003"
for p in (ROOT, ROOT / "shared", ROOT / "shared/utilities/tests/manual", R3, R6, HERE):
    sys.path.insert(0, str(p))

import analyze_run3 as A3  # noqa: E402
import analyze_run6 as A6  # noqa: E402
import within_oof as W  # noqa: E402
from answer_detector_gsm8k import PATTERNS, is_answer_bearing, value  # noqa: E402
from bg_steering_suite_lib import gsm_prompt  # noqa: E402
from bg_v2_overnight_horizon_generate import split_for  # noqa: E402
from gen_run7 import PREFIX_GRID as GRID, earlier_runs, select_tasks  # noqa: E402
from proto_introspection_controls_analysis import auroc  # noqa: E402

N_CUT, MIN_MIXED = 96, 100
SEALED = ("PLAN.md", "gen_run7.py", "analyze_run7.py", "within_oof.py", "test_run7.py",
          "../gsm8k_train_20261003/gen_run6.py", "../gsm8k_train_20261003/analyze_run6.py",
          "../gsm8k_fixed_prefix_20260927/gen_run3.py", "../gsm8k_fixed_prefix_20260927/analyze_run3.py",
          "../gsm8k_fixed_prefix_20260927/answer_detector_gsm8k.py")
CUT_I = GRID.index(N_CUT)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def first_commitment_chrono(text: str) -> str | None:
    """Sensitivity variant of answer_detector_gsm8k.first_commitment: the same patterns, the earliest match wins."""
    ms = [m for m in (p.search(text) for p in PATTERNS.values()) if m]
    return min(ms, key=lambda m: m.start()).group(1).strip() if ms else None


def commit_pos(text: str) -> int | None:
    """Character offset of the match that first_commitment (the primary label) uses."""
    for p in PATTERNS.values():
        m = p.search(text)
        if m:
            return m.start()
    return None


def load(data: Path, prefix: str = "run7"):
    """Shards with only the N = 96 pooled features kept (the other cuts are dropped to fit in memory)."""
    recs, errors = [], []
    for f in sorted(glob.glob(str(data / f"{prefix}_shard_*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        assert not d["smoke"] and d["prefix_grid"] == list(GRID), f
        assert max(d["equivalence_checks_maxabs"] or [0.0]) <= 1e-3, f
        for r in d["records"]:
            r["h96"] = r.pop("prefix_features")[CUT_I].clone()
        recs.extend(d["records"]); errors.extend(d["errors"])
        del d
    return recs, errors


def integrity(recs, errors, synthetic):
    """Offline: the problem list comes from the local, SHA-256-pinned parquet files (Run 6's audit crashed offline)."""
    expected = {t["task_id"] for t in select_tasks()}
    got = {r["task_id"] for r in recs}
    if not synthetic:
        seal = dict(line.split()[::-1] for line in (HERE / "SEAL.sha256").read_text().splitlines())
        for name in SEALED:
            assert seal[name] == sha(HERE / name), f"{name} changed after sealing"
    assert got <= expected
    assert all(r["split"] == split_for(r["task_id"]) for r in recs)
    assert all(r["decode_roundtrip_ok"] for r in recs)
    return {"n_records": len(recs), "n_tasks": len(got), "tasks_missing": len(expected - got), "errors": errors,
            "tasks_by_dataset_split": dict(Counter(r["dataset_split"] for r in recs if r["candidate_idx"] == 0))}


S = lambda r: torch.cat([A6.lp_at(N_CUT)(r), A6.s_diff(r)])  # Run 6's S at N = 96  # noqa: E731
H = lambda r: r["h96"]  # noqa: E731
PEND = lambda r: r["prompt_end_state"]  # noqa: E731


def text(r):  # Run 6's reader text
    return f"Prompt:\n{gsm_prompt(r['question'])}\n\nVisible reasoning prefix:\n{r['prefix_texts'][CUT_I]}"


def oof(rows, label="success", arms=("hidden", "combined", "surface")):
    y = np.array([int(r[label]) for r in rows]); t = [r["task_id"] for r in rows]
    Hm = np.stack([H(r).flatten().float().numpy() for r in rows])
    Sm = np.stack([S(r).float().numpy() for r in rows])
    sc, hp = W.oof_scores([Hm], Sm, y, t, arms)
    return sc, hp, y, t


def verdict_across(res):  # Run 6's verdict rule, renamed
    if not A3.adjudicable(res):
        return "GSM8K_REST_FIXED_PREFIX_NOT_ADJUDICABLE"
    lo, hi = res["paired_task_clustered_bootstrap_combined_minus_shortcut"]["ci95"]
    return ("GSM8K_REST_FIXED_PREFIX_INCREMENT_POSITIVE" if lo > 0 else
            "GSM8K_REST_FIXED_PREFIX_INCREMENT_NEGATIVE" if hi < 0 else "GSM8K_REST_FIXED_PREFIX_INCREMENT_UNRESOLVED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["audit", "full"], required=True)
    ap.add_argument("--data", type=Path, default=HERE)
    ap.add_argument("--run6-data", type=Path, default=None, help="Run 6 shards, for the pooled secondary")
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    recs, errors = load(a.data)
    out = {"integrity": integrity(recs, errors, a.synthetic), "machine": {"platform": platform.platform(),
           "processor": platform.processor(), "numpy": np.__version__, "torch": torch.__version__}}
    A6.answer_free_table(recs)
    rule_n, frac = A6.choose_n_star(recs)
    out.update({"n_cut": N_CUT, "run6_rule_would_choose": rule_n, "answer_free_fraction_train": frac,
                "parseable_fraction": sum(r["parseable"] for r in recs) / len(recs),
                "hit_max_tokens_fraction": sum(r["hit_max_tokens"] for r in recs) / len(recs)})
    A6.hand_audit(recs, N_CUT, a.out)
    if a.stage == "audit":
        (a.out / "audit_stage.json").write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2)); return

    parse = [r for r in recs if r["parseable"]]
    primary = [r for r in parse if r["free"][N_CUT]]
    # 1. primary: within-question AUROC of the hidden-state probe, out of fold over 5 hash folds, all splits
    sc, hp, y, t = oof(primary)
    wq = {k: W.within_task_auroc(v, y, t) for k, v in sc.items()}
    out["primary"] = {"within_question_oof": wq, "verdict": W.verdict(wq["hidden"], MIN_MIXED), "n_answers": len(primary),
                      "n_incorrect": int((y == 0).sum()), "across_question_oof_auroc": {k: float(auroc(torch.tensor(v), torch.tensor(y).double())) for k, v in sc.items()},
                      "hyperparams_by_fold": hp}
    # 4. TF-IDF reader, out of fold, and the paired hidden-minus-reader difference
    sc["reader"] = W.reader_oof([text(r) for r in primary], y, t)
    out["secondary_reader_oof"] = {"within_question": W.within_task_auroc(sc["reader"], y, t),
                                   "paired_hidden_minus_reader": W.paired_diff(sc["hidden"], sc["reader"], y, t)}
    # 6. commit timing
    out["secondary_commit_timing"] = W.commit_timing(sc, y, t, [commit_pos(r["generated_text"]) for r in primary])
    # 5. + 2. Run 6's across-question estimator (proper_arm, GroupKFold) with the same PCA computed by ARPACK; held-out within-question
    A3._pca = W.pca
    res, sh = A3.proper_arm("across_question", primary, [H], S)
    res["verdict"] = verdict_across(res)
    out["secondary_across_question"] = res
    if sh is not None:
        out["secondary_heldout_within_question"] = {k: W.within_task_auroc(sh[k].numpy(), sh["y"].numpy(), sh["tasks"]) for k in ("combined", "hidden", "shortcut")}
    pe = out["secondary_prompt_end_control"] = A3.proper_arm("prompt_end_control", primary, [H], S, base=[PEND])[0]
    if "paired_task_clustered_bootstrap_combined_minus_shortcut" in pe:
        survives = pe["paired_task_clustered_bootstrap_combined_minus_shortcut"]["ci95"][0] > 0
        out["secondary_headline_rule"] = "READABILITY_BEYOND_DIFFICULTY" if survives and res["verdict"].endswith("POSITIVE") else "MOSTLY_DIFFICULTY"
    # 8. chronological first-commitment labels (sensitivity)
    for r in recs:
        r["own_chrono"] = first_commitment_chrono(r["generated_text"])
        r["parseable_chrono"] = value(r["own_chrono"]) is not None
        r["success_chrono"] = r["parseable_chrono"] and value(r["own_chrono"]) == value(r["gold"])
        r["free_chrono"] = not is_answer_bearing(r["prefix_texts"][CUT_I], r["own_chrono"])
    prim_c = [r for r in recs if r["parseable_chrono"] and r["free_chrono"]]
    ch = {"answers_own_answer_differs": sum(r["own_chrono"] != r["own_answer"] for r in recs),
          "answers_label_differs": sum(r["success_chrono"] != r["success"] for r in recs),
          "cohort_size_primary_vs_chrono": [len(primary), len(prim_c)],
          "cohort_or_label_changed": {(r["task_id"], r["candidate_idx"], r["success"]) for r in primary} !=
                                     {(r["task_id"], r["candidate_idx"], r["success_chrono"]) for r in prim_c}}
    if ch["cohort_or_label_changed"]:
        scc, _, yc, tc = oof(prim_c, "success_chrono", ("hidden",))
        ch["within_question_hidden_oof"] = W.within_task_auroc(scc["hidden"], yc, tc)
    out["secondary_chronological_labels"] = ch
    # 7. pooled Run 6 + Run 7, out of fold with hash folds over both
    if a.run6_data is not None:
        r6, e6 = load(a.run6_data, "run6")
        ids6 = {r["task_id"] for r in r6}
        assert ids6 <= {t_["task_id"] for t_ in earlier_runs()["run6_cohort"]} and not ids6 & {r["task_id"] for r in recs}
        A6.answer_free_table(r6)
        prim6 = [r for r in r6 if r["parseable"] and r["free"][N_CUT]]
        scp, _, yp, tp = oof(primary + prim6, arms=("hidden",))
        out["secondary_pooled_run6_run7"] = {"within_question_hidden_oof": W.within_task_auroc(scp["hidden"], yp, tp),
                                             "run6_records": len(r6), "run6_errors": len(e6), "run6_primary_answers": len(prim6)}
    else:
        out["secondary_pooled_run6_run7"] = {"not_run": "--run6-data not given"}
    out["accuracy_parseable"] = sum(r["success"] for r in parse) / len(parse)
    (a.out / "results.json").write_text(json.dumps(out, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
    write_md(out, a.out)


def write_md(out, d):
    def wq(x):
        return f"{x['estimate']:.3f} [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}] ({x['n_pairs']} pairs, {x['n_mixed_tasks']} problems)" if x.get("n_mixed_tasks") else "n/a"

    def arm(r):
        if "incremental_auroc_combined_minus_shortcut" not in r:
            return f"not run, below the minimums ({r['n_heldout_tasks']} held-out problems, {r['n_heldout_negatives']} incorrect)"
        b = r["paired_task_clustered_bootstrap_combined_minus_shortcut"]["ci95"]
        return (f"surface {r['auroc_shortcut_only']:.3f}, hidden {r['auroc_hidden_only']:.3f}, both {r['auroc_hidden_plus_shortcuts']:.3f}, "
                f"increase {r['incremental_auroc_combined_minus_shortcut']:+.3f} [{b[0]:+.3f}, {b[1]:+.3f}]; held-out {r['n_heldout']} answers, "
                f"{r['n_heldout_tasks']} problems, {r['n_heldout_negatives']} incorrect")
    p, rd, ct, ch = out["primary"], out["secondary_reader_oof"], out["secondary_commit_timing"], out["secondary_chronological_labels"]
    pd = rd["paired_hidden_minus_reader"]
    L = ["# Run 7 results (GSM8K, every unused problem)", "",
         f"N = {out['n_cut']} (fixed; Run 6's 75% rule would choose {out['run6_rule_would_choose']}). Verdict: **{p['verdict']}**", "",
         f"- Primary, within-question AUROC of the hidden-state probe, out of fold (5 hash folds): {wq(p['within_question_oof']['hidden'])}; "
         f"{p['n_answers']} answers, {p['n_incorrect']} incorrect",
         f"- Combined probe (hidden + S): {wq(p['within_question_oof']['combined'])}; baseline S: {wq(p['within_question_oof']['surface'])}",
         f"- TF-IDF reader: {wq(rd['within_question'])}; hidden minus reader {pd['diff']:+.3f} [{pd['ci95'][0]:+.3f}, {pd['ci95'][1]:+.3f}]",
         f"- Commit timing: correct answer commits later in {ct['share_correct_commits_later']:.3f} of {ct['n_pairs']} pairs; "
         f"commit position alone {wq(ct['auroc_commit_position_alone'])}; by group {ct['by_group']}",
         f"- Across-question increment (Run 6's estimator): {arm(out['secondary_across_question'])}; verdict {out['secondary_across_question']['verdict']}",
         f"- End-of-prompt control: {arm(out['secondary_prompt_end_control'])}; headline rule {out.get('secondary_headline_rule', 'n/a')}"]
    if "secondary_heldout_within_question" in out:
        h = out["secondary_heldout_within_question"]
        L.append(f"- Held-out only, within-question: hidden {wq(h['hidden'])}; combined {wq(h['combined'])}; surface {wq(h['shortcut'])}")
    po = out["secondary_pooled_run6_run7"]
    L.append(f"- Pooled Run 6 + Run 7, hidden, out of fold: {wq(po['within_question_hidden_oof']) if 'within_question_hidden_oof' in po else po['not_run']}")
    L.append(f"- Chronological labels: {ch['answers_own_answer_differs']} answers change their first answer, {ch['answers_label_differs']} their label; "
             f"cohort {ch['cohort_size_primary_vs_chrono']}; hidden within-question "
             f"{wq(ch['within_question_hidden_oof']) if 'within_question_hidden_oof' in ch else 'unchanged (same cohort and labels)'}")
    L.append(f"- Accuracy (parseable) {out['accuracy_parseable']:.3f}; parseable {out['parseable_fraction']:.3f}; hit 448 {out['hit_max_tokens_fraction']:.3f}")
    (d / "RESULTS.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
