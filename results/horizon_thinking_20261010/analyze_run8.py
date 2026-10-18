"""Run 8 analysis (Horizon, Ouro-2.6B-Thinking on Run 4's questions), implementing PLAN.md as sealed. A copy of
Run 4's analyze_run4.py with the model and paths changed; Run 4's endpoints are kept as secondaries, and the new
primary (within-question AUROC out of fold over hash folds, within_oof.py from Run 7) and secondaries are added.
Stage 'audit' is label-free (integrity, answer-free fractions, N*, hand-audit list) and stops; stage 'full' opens
labels. The estimators come from Run 3's sealed analysis (analyze_run3.py, imported unchanged; hash in SEAL).
--rltt-data points at Run 4's shards for the comparison with Ouro-RLTT; --data/--out/--synthetic exist for dry runs."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
R3 = ROOT / "opi/preanswer/gsm8k_fixed_prefix_20260927"
R4 = ROOT / "opi/preanswer/horizon_cohort2_20261003"
R7 = ROOT / "opi/preanswer/gsm8k_rest_20261010"
for p in (ROOT, ROOT / "shared", ROOT / "shared/utilities/tests/manual", R3, R4, R7, HERE):
    sys.path.insert(0, str(p))

import analyze_run3 as A3  # noqa: E402
import within_oof as W  # noqa: E402
from answer_detector import is_answer_bearing, task_statement  # noqa: E402
from bg_steering_suite_lib import mcq_prompt  # noqa: E402
from bg_v2_overnight_horizon_generate import split_for  # noqa: E402
from bg_v2_overnight_horizon_analysis import SEED, shortcut_vec, bootstrap_task_clustered  # noqa: E402
from gen_run8 import MODEL_FILES, PREFIX_GRID as GRID, select_tasks  # noqa: E402
from proto_introspection_controls_analysis import auroc  # noqa: E402

COVERAGE, MIN_MIXED, RLTT_N_STAR = 0.75, 60, 48
SEALED = ("PLAN.md", "gen_run8.py", "analyze_run8.py", "modeling_ouro_shim.py", "test_run8.py",
          "../horizon_cohort2_20261003/answer_detector.py", "../gsm8k_rest_20261010/within_oof.py",
          "../gsm8k_fixed_prefix_20260927/analyze_run3.py", "../gsm8k_fixed_prefix_20260927/gen_run3.py",
          "../gsm8k_fixed_prefix_20260927/answer_detector_gsm8k.py")
CLASSES = ("True", "False", "Unknown")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(data: Path, prefix: str = "run8"):
    recs, errors = [], []
    for f in sorted(glob.glob(str(data / f"{prefix}_shard_*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        assert not d["smoke"], f
        assert max(d["equivalence_checks_maxabs"] or [0.0]) <= 1e-3, f
        if prefix == "run8":
            assert d["model_sha256"] == MODEL_FILES, f"{f}: generated with other weights"
        recs.extend(d["records"]); errors.extend(d["errors"])
    for r in recs:
        r["task_id"] = r["task_uid"]
    return recs, errors


def integrity(recs, errors, synthetic):
    expected = {t["task_uid"] for t in select_tasks()}
    got = {r["task_uid"] for r in recs}
    if not synthetic:
        seal = dict(line.split()[::-1] for line in (HERE / "SEAL.sha256").read_text().splitlines())
        for name in SEALED:
            assert seal[name] == sha(HERE / name), f"{name} changed after sealing"
    assert got <= expected
    assert all(r["split"] == split_for(r["task_uid"]) for r in recs)
    assert all(r["decode_roundtrip_ok"] for r in recs)
    return {"n_records": len(recs), "n_tasks": len(got), "tasks_missing": len(expected - got), "errors": errors}


def answer_free_table(recs):
    for r in recs:
        st = task_statement(r["task_prompt"])
        r["free"] = {n: (n == 0 or not is_answer_bearing(r["prefix_texts"][i], st)) for i, n in enumerate(GRID)}


def choose_n_star(recs):
    train = [r for r in recs if r["split"] == "train"]
    frac = {n: sum(r["free"][n] for r in train) / len(train) for n in GRID if n > 0}
    ok = [n for n, f in frac.items() if f >= COVERAGE]
    return (max(ok) if ok else 16), frac


def hand_audit(recs, n_star, out):
    free = [r for r in recs if r["split"] == "train" and r["free"][n_star]]
    free.sort(key=lambda r: hashlib.sha256(f"20261010|{r['task_uid']}|{r['candidate_idx']}".encode()).hexdigest())
    i = GRID.index(n_star)
    lines = [f"# Hand audit: 20 answer-free training prefixes at N* = {n_star} (label-free)", ""]
    for r in free[:20]:
        lines += [f"## {r['task_uid']} #{r['candidate_idx']}", f"Statement: {task_statement(r['task_prompt'])}",
                  "```", r["prefix_texts"][i], "```", ""]
    (out / "hand_audit.md").write_text("\n".join(lines))


def s_diff(r):
    """Difficulty known before generation: prompt tokens, proof depth, variables, facts, rules, answer class."""
    return torch.tensor([float(r["prompt_tok"]), float(r["proof_depth"]), float(r["n_vars"]), float(r["n_facts"]),
                         float(r["n_rules"])] + [float(r["gold_answer"] == c) for c in CLASSES])


def lp_at(n):
    def f(r):
        lp = r["gen_logprobs"][:n] or [0.0]
        return torch.tensor([sum(lp) / len(lp), min(lp)])
    return f


def feat_at(n):
    i = GRID.index(n)
    return lambda r: r["prefix_features"][i]


_LEAD = r"(?i:final\s*answer)[\s\*]*[:\-]?[\s\*\(\[<\"']*"  # as gen_run4.strict_answer


def _label_matches(text, options):
    letters = "".join(sorted(options))
    words = {v.strip().rstrip(".").lower(): k for k, v in options.items()}
    let = [m for m in (re.search(_LEAD + rf"([{letters}])\b", text), re.search(rf"\\boxed\{{\s*([{letters}])\s*\}}", text)) if m]
    w = re.search(_LEAD + r"(" + "|".join(map(re.escape, words)) + r")\b", text, flags=re.I)
    return let, w, words


def chrono_answer(text, options):
    """Sensitivity variant of gen_run4.strict_answer: the same patterns, the earliest match wins (strict_answer gives
    a letter priority over an option word)."""
    let, w, words = _label_matches(text, options)
    cands = [(m.start(), m.group(1).upper()) for m in let] + ([(w.start(), words[w.group(1).lower()])] if w else [])
    return min(cands)[1] if cands else None


def strict_pos(text, options):
    """Character offset of the match that strict_answer (the primary label) uses (the 2026-10-04 audit's pos())."""
    let, w, _ = _label_matches(text, options)
    return min(m.start() for m in let) if let else (w.start() if w else None)


def visible_reader(rows, n, target_scores, y_ho, t_ho, sc):
    """TF-IDF reader of the prompt and the first n tokens, with the same S appended (Run 3's reader, Horizon prompt)."""
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import FeatureUnion
    i = GRID.index(n)
    text = lambda r: f"Prompt:\n{mcq_prompt(r['task_prompt'], r['options'])}\n\nVisible reasoning prefix:\n{r['prefix_texts'][i]}"
    tv = [r for r in rows if r["split"] in ("train", "val")]
    ho = [r for r in rows if r["split"] == "heldout"]
    y_tv = [int(r["success"]) for r in tv]
    S_tv, S_ho = np.stack([sc(r).float().numpy() for r in tv]), np.stack([sc(r).float().numpy() for r in ho])
    uniq = sorted({r["task_id"] for r in tv})
    order = [uniq[j] for j in torch.randperm(len(uniq), generator=torch.Generator().manual_seed(SEED)).tolist()]
    fold = {u: k % 5 for k, u in enumerate(order)}
    folds = [fold[r["task_id"]] for r in tv]

    def add_s(M, S_fit, S_use):
        mu, sd = S_fit.mean(0), S_fit.std(0) + 1e-6
        return hstack([M, csr_matrix((S_use - mu) / sd)]).tocsr()

    def vec():
        return FeatureUnion([
            ("w", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), strip_accents="unicode", sublinear_tf=True, max_features=40000)),
            ("c", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=60000))])
    best = None
    for C in (0.01, 0.1, 1.0, 10.0):
        oof = [0.0] * len(tv)
        for k in range(5):
            tr = [j for j, f in enumerate(folds) if f != k]; va = [j for j, f in enumerate(folds) if f == k]
            v = vec().fit([text(tv[j]) for j in tr])
            m = LogisticRegression(C=C, solver="liblinear", max_iter=2000, random_state=20260729)
            m.fit(add_s(v.transform([text(tv[j]) for j in tr]), S_tv[tr], S_tv[tr]), [y_tv[j] for j in tr])
            for j, pj in zip(va, m.predict_proba(add_s(v.transform([text(tv[j]) for j in va]), S_tv[tr], S_tv[va]))[:, 1]):
                oof[j] = float(pj)
        a = roc_auc_score(y_tv, oof)
        best = (a, C) if best is None or a > best[0] else best
    v = vec().fit([text(r) for r in tv])
    m = LogisticRegression(C=best[1], solver="liblinear", max_iter=2000, random_state=20260729)
    m.fit(add_s(v.transform([text(r) for r in tv]), S_tv, S_tv), y_tv)
    s_v = torch.tensor(m.predict_proba(add_s(v.transform([text(r) for r in ho]), S_tv, S_ho))[:, 1], dtype=torch.double)
    return {"cv_auroc": best[0], "C": best[1], "auroc_visible_reader": auroc(s_v, y_ho),
            "paired_diff_combined_minus_reader": auroc(target_scores, y_ho) - auroc(s_v, y_ho),
            "bootstrap": bootstrap_task_clustered(target_scores, s_v, y_ho, t_ho, rounds=2000)}


def verdict(res):  # Run 4's across-question verdict, renamed; a secondary in this run
    if not A3.adjudicable(res):
        return "HORIZON_THINKING_FIXED_PREFIX_NOT_ADJUDICABLE"
    lo, hi = res["paired_task_clustered_bootstrap_combined_minus_shortcut"]["ci95"]
    return ("HORIZON_THINKING_FIXED_PREFIX_INCREMENT_POSITIVE" if lo > 0 else
            "HORIZON_THINKING_FIXED_PREFIX_INCREMENT_NEGATIVE" if hi < 0 else "HORIZON_THINKING_FIXED_PREFIX_INCREMENT_UNRESOLVED")


def oof_within(rows, n, label="success", arms=("hidden", "combined", "surface")):
    """Every answer scored out of fold (5 hash folds), hidden = PCA(H at n), combined = + S, surface = S."""
    y = np.array([int(r[label]) for r in rows]); t = [r["task_id"] for r in rows]
    Hm = np.stack([feat_at(n)(r).flatten().float().numpy() for r in rows])
    Sm = np.stack([torch.cat([lp_at(n)(r), s_diff(r)]).float().numpy() for r in rows])
    sc, hp = W.oof_scores([Hm], Sm, y, t, arms)
    return sc, hp, y, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["audit", "full"], required=True)
    ap.add_argument("--data", type=Path, default=HERE)
    ap.add_argument("--rltt-data", type=Path, default=None, help="Run 4's shards (Ouro-RLTT), for the comparison")
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    recs, errors = load(a.data)
    out = {"integrity": integrity(recs, errors, a.synthetic)}
    answer_free_table(recs)
    n_star, frac = choose_n_star(recs)
    out.update({"n_star": n_star, "answer_free_fraction_train": frac,
                "labelled_fraction": sum(r["labelled"] for r in recs) / len(recs),
                "hit_max_tokens_fraction": sum(r["hit_max_tokens"] for r in recs) / len(recs)})
    hand_audit(recs, n_star, a.out)
    if a.stage == "audit":
        (a.out / "audit_stage.json").write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2)); return

    lab = [r for r in recs if r["labelled"]]
    primary = [r for r in lab if r["free"][n_star]]
    # primary: within-question AUROC of the hidden-state probe, out of fold over 5 hash folds, all splits
    sc_o, hp_o, y_o, t_o = oof_within(primary, n_star)
    wq_o = {k: W.within_task_auroc(v, y_o, t_o) for k, v in sc_o.items()}
    out["primary"] = {"within_question_oof": wq_o, "verdict": W.verdict(wq_o["hidden"], MIN_MIXED), "n_answers": len(primary),
                      "n_incorrect": int((y_o == 0).sum()), "hyperparams_by_fold": hp_o}
    sc_o["reader"] = W.reader_oof([f"Prompt:\n{mcq_prompt(r['task_prompt'], r['options'])}\n\nVisible reasoning prefix:\n"
                                   f"{r['prefix_texts'][GRID.index(n_star)]}" for r in primary], y_o, t_o)
    out["secondary_reader_oof"] = {"within_question": W.within_task_auroc(sc_o["reader"], y_o, t_o),
                                   "paired_hidden_minus_reader": W.paired_diff(sc_o["hidden"], sc_o["reader"], y_o, t_o)}
    out["secondary_commit_timing"] = W.commit_timing(sc_o, y_o, t_o, [strict_pos(r["generated_text"], r["options"]) for r in primary])
    if n_star != RLTT_N_STAR:
        p48 = [r for r in lab if r["free"][RLTT_N_STAR]]
        s48, _, y48, t48 = oof_within(p48, RLTT_N_STAR, arms=("hidden",))
        out["secondary_within_question_oof_n48"] = W.within_task_auroc(s48["hidden"], y48, t48)
    # chronological first-commitment labels (sensitivity)
    for r in recs:
        r["chrono_answer"] = chrono_answer(r["generated_text"], r["options"])
        r["labelled_chrono"], r["success_chrono"] = r["chrono_answer"] is not None, r["chrono_answer"] == r["gold_letter"]
    prim_c = [r for r in recs if r["labelled_chrono"] and r["free"][n_star]]
    ch = {"answers_label_differs": sum(r["chrono_answer"] != r["strict_answer"] for r in recs),
          "cohort_size_primary_vs_chrono": [len(primary), len(prim_c)],
          "cohort_or_label_changed": {(r["task_id"], r["candidate_idx"], r["success"]) for r in primary} !=
                                     {(r["task_id"], r["candidate_idx"], r["success_chrono"]) for r in prim_c}}
    if ch["cohort_or_label_changed"]:
        scc, _, yc, tc = oof_within(prim_c, n_star, "success_chrono", ("hidden",))
        ch["within_question_hidden_oof"] = W.within_task_auroc(scc["hidden"], yc, tc)
    out["secondary_chronological_labels"] = ch
    # comparison with Ouro-RLTT on the same questions (Run 4's shards), unpaired, both with hash folds
    if a.rltt_data is not None:
        r4, _ = load(a.rltt_data, "run4")
        p4 = {(r["task_uid"], r["candidate_idx"]): r for r in r4}
        same = [r for r in recs if (r["task_uid"], r["candidate_idx"]) in p4]
        answer_free_table(r4)
        cmp_ = {"inputs_identical": {"answers_compared": len(same),
                                     "prompt_ids_equal": sum(r["prompt_ids"] == p4[(r["task_uid"], r["candidate_idx"])]["prompt_ids"] for r in same),
                                     "seeds_equal": sum(r["seed"] == p4[(r["task_uid"], r["candidate_idx"])]["seed"] for r in same)}}
        for name, nt, nr in (("own_n_star", n_star, RLTT_N_STAR), ("both_at_48", RLTT_N_STAR, RLTT_N_STAR)):
            if name == "both_at_48" and n_star == RLTT_N_STAR:
                continue
            st, _, yt, tt = oof_within([r for r in lab if r["free"][nt]], nt, arms=("hidden",))
            sr, _, yr, tr = oof_within([r for r in r4 if r["labelled"] and r["free"][nr]], nr, arms=("hidden",))
            cmp_[name] = {"n_thinking": nt, "n_rltt": nr, "rltt_within_question": W.within_task_auroc(sr["hidden"], yr, tr),
                          "thinking_minus_rltt": W.unpaired_diff({"s": st["hidden"], "y": yt, "tasks": tt}, {"s": sr["hidden"], "y": yr, "tasks": tr})}
        out["secondary_rltt_comparison"] = cmp_
    else:
        out["secondary_rltt_comparison"] = {"not_run": "--rltt-data not given (post-run step on the laptop)"}

    # Run 4's analyses, unchanged, as secondaries
    S = lambda r: torch.cat([lp_at(n_star)(r), s_diff(r)])
    pend = lambda r: r["prompt_end_state"]
    res, sc = A3.proper_arm("primary", primary, [feat_at(n_star)], S)
    res["verdict"] = verdict(res)
    if sc is not None:
        y_ho, t_ho = sc["y"], sc["tasks"]
        perm = torch.randperm(y_ho.shape[0], generator=torch.Generator().manual_seed(SEED))
        res["shuffled_label_auroc_combined"] = auroc(sc["combined"], y_ho[perm])
        out["control_within_question"] = {k: A3.within_task_auroc(sc[k].numpy(), y_ho.numpy(), t_ho) for k in ("combined", "hidden", "shortcut")}
        out["secondary_visible_reader"] = visible_reader(primary, n_star, sc["combined"], y_ho, t_ho, S)
    out["run4_primary_across_question"] = res
    out["control_prompt_end"] = A3.proper_arm("prompt_end_control", primary, [feat_at(n_star)], S, base=[pend])[0]
    pe = out["control_prompt_end"]
    if "paired_task_clustered_bootstrap_combined_minus_shortcut" in pe:
        survives = pe["paired_task_clustered_bootstrap_combined_minus_shortcut"]["ci95"][0] > 0
        out["headline_rule"] = "READABILITY_BEYOND_DIFFICULTY" if survives and res["verdict"].endswith("POSITIVE") else "MOSTLY_DIFFICULTY"
    out["secondary_prompt_end_only"] = A3.proper_arm("prompt_end_only", lab, [pend], s_diff)[0]
    out["secondary_run1_style"] = A3.guarded("run1_style", primary, feat_at(n_star), lp_at(n_star))[0]
    out["secondary_grid"] = {str(n): A3.proper_arm(f"grid_{n}", [r for r in lab if r["free"][n]], [feat_at(n)],
                                                  lambda r, n=n: torch.cat([lp_at(n)(r), s_diff(r)]))[0] for n in GRID[1:]}
    pub = [r for r in recs if not r["malformed_published"]]
    for r in pub:
        r["success_pub_tmp"] = r["success_published"]
    out["secondary_final_line_published_protocol"] = A3.guarded("final_line_published", pub, lambda r: r["preanswer_features"],
                                                                shortcut_vec, "success_pub_tmp")[0]
    out["secondary_final_line_strict_difficulty"] = A3.proper_arm("final_line_strict", lab, [lambda r: r["preanswer_features"]],
                                                                  lambda r: torch.cat([shortcut_vec(r), s_diff(r)]))[0]
    lc = Counter((None if r["malformed_published"] else r["success_published"], r["success"] if r["labelled"] else None) for r in recs)
    out["labels_published_vs_strict"] = {f"published={k[0]}|strict={k[1]}": v for k, v in sorted(lc.items(), key=str)}
    out["accuracy_strict_labelled"] = sum(r["success"] for r in lab) / len(lab)
    (a.out / "results.json").write_text(json.dumps(out, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
    write_md(out, n_star, a.out)


def write_md(out, n_star, d):
    def fmt(r):
        if "incremental_auroc_combined_minus_shortcut" not in r:
            return f"not run, below the minimums ({r['n_heldout_tasks']} held-out tasks, {r['n_heldout_negatives']} incorrect)"
        b = r["paired_task_clustered_bootstrap_combined_minus_shortcut"]["ci95"]
        return (f"surface {r['auroc_shortcut_only']:.3f}, hidden {r['auroc_hidden_only']:.3f}, both {r['auroc_hidden_plus_shortcuts']:.3f}, "
                f"increase {r['incremental_auroc_combined_minus_shortcut']:+.3f} [{b[0]:+.3f}, {b[1]:+.3f}]; held-out {r['n_heldout']} answers, "
                f"{r['n_heldout_tasks']} tasks, {r['n_heldout_negatives']} incorrect")

    def wq(x):
        return f"{x['estimate']:.3f} [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}] ({x['n_pairs']} pairs, {x['n_mixed_tasks']} questions)" if x.get("n_mixed_tasks") else "n/a"

    def df(x):
        return f"{x['diff']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]"
    p, res, pe = out["primary"], out["run4_primary_across_question"], out["control_prompt_end"]
    rd, ct, ch, cm = out["secondary_reader_oof"], out["secondary_commit_timing"], out["secondary_chronological_labels"], out["secondary_rltt_comparison"]
    L = ["# Run 8 results (Horizon, Ouro-2.6B-Thinking, Run 4's questions)", "",
         f"N* = {n_star}. Verdict: **{p['verdict']}**. Run 4's across-question verdict: {res['verdict']}; headline rule: {out.get('headline_rule', 'n/a')}", "",
         f"- Primary, within-question AUROC of the hidden-state probe, out of fold (5 hash folds): {wq(p['within_question_oof']['hidden'])}; "
         f"{p['n_answers']} answers, {p['n_incorrect']} incorrect",
         f"- Combined (hidden + S): {wq(p['within_question_oof']['combined'])}; S only: {wq(p['within_question_oof']['surface'])}",
         f"- TF-IDF reader, out of fold: {wq(rd['within_question'])}; hidden minus reader {df(rd['paired_hidden_minus_reader'])}",
         f"- Commit timing: correct answer commits later in {ct['share_correct_commits_later']:.3f} of {ct['n_pairs']} pairs; "
         f"commit position alone {wq(ct['auroc_commit_position_alone'])}; by group {ct['by_group']}"]
    if "secondary_within_question_oof_n48" in out:
        L.append(f"- Within-question, out of fold, at N = 48: {wq(out['secondary_within_question_oof_n48'])}")
    L.append(f"- Chronological labels: {ch['answers_label_differs']} answers change label; cohort {ch['cohort_size_primary_vs_chrono']}; hidden within-question "
             f"{wq(ch['within_question_hidden_oof']) if 'within_question_hidden_oof' in ch else 'unchanged (same cohort and labels)'}")
    if "not_run" in cm:
        L.append(f"- Comparison with Ouro-RLTT: {cm['not_run']}")
    else:
        L.append(f"- Inputs identical to Run 4: {cm['inputs_identical']}")
        for k in ("own_n_star", "both_at_48"):
            if k in cm:
                L.append(f"- Thinking minus RLTT ({k}: N = {cm[k]['n_thinking']} vs {cm[k]['n_rltt']}): {df(cm[k]['thinking_minus_rltt'])}; "
                         f"RLTT {wq(cm[k]['rltt_within_question'])}")
    L += [f"- Run 4's primary (across questions, S = log-prob + difficulty): {fmt(res)}",
          f"- End-of-prompt state in the baseline: {fmt(pe)}"]
    if "control_within_question" in out:
        cw, vr = out["control_within_question"], out["secondary_visible_reader"]
        L += [f"- Held-out only, within-question: combined {wq(cw['combined'])}; hidden {wq(cw['hidden'])}; surface {wq(cw['shortcut'])}",
              f"- Shuffled labels {res['shuffled_label_auroc_combined']:.3f}",
              f"- Visible reader + S (held-out): AUROC {vr['auroc_visible_reader']:.3f}; combined minus reader {vr['paired_diff_combined_minus_reader']:+.3f} "
              f"[{vr['bootstrap']['ci95'][0]:+.3f}, {vr['bootstrap']['ci95'][1]:+.3f}]"]
    L += [f"- End-of-prompt state alone vs difficulty: {fmt(out['secondary_prompt_end_only'])}",
          f"- Run 1-style (S = log-prob only, arm()): {fmt(out['secondary_run1_style'])}",
          f"- Final-answer line, published protocol: {fmt(out['secondary_final_line_published_protocol'])}",
          f"- Final-answer line, strict labels, difficulty in S: {fmt(out['secondary_final_line_strict_difficulty'])}",
          f"- Labels, published vs strict: {out['labels_published_vs_strict']}",
          f"- Accuracy (strict, labelled) {out['accuracy_strict_labelled']:.3f}; labelled {out['labelled_fraction']:.3f}; hit 448 {out['hit_max_tokens_fraction']:.3f}",
          "", "Grid:"] + [f"- N={n}: {fmt(r)}" for n, r in out["secondary_grid"].items()]
    (d / "RESULTS.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
