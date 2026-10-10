"""Run 6 analysis (GSM8K training split), implementing PLAN.md as sealed. Stage 'audit' is label-free and stops;
stage 'full' opens labels. Estimators from Run 3's sealed analysis (imported unchanged; hash in SEAL)."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
R3 = ROOT / "opi/preanswer/gsm8k_fixed_prefix_20260927"
for p in (ROOT, ROOT / "shared", ROOT / "shared/utilities/tests/manual", R3, HERE):
    sys.path.insert(0, str(p))

import analyze_run3 as A3  # noqa: E402
from answer_detector_gsm8k import is_answer_bearing  # noqa: E402
from bg_steering_suite_lib import gsm_prompt  # noqa: E402
from bg_v2_overnight_horizon_generate import split_for  # noqa: E402
from bg_v2_overnight_horizon_analysis import SEED, bootstrap_task_clustered  # noqa: E402
from gen_run6 import N_TASKS, PREFIX_GRID as GRID, select_tasks  # noqa: E402
from proto_introspection_controls_analysis import auroc  # noqa: E402

COVERAGE = 0.75
SEALED = ("PLAN.md", "gen_run6.py", "analyze_run6.py")
R3_SEALED = ("../gsm8k_fixed_prefix_20260927/analyze_run3.py", "../gsm8k_fixed_prefix_20260927/gen_run3.py",
             "../gsm8k_fixed_prefix_20260927/answer_detector_gsm8k.py")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(data):
    recs, errors = [], []
    for f in sorted(glob.glob(str(data / "run6_shard_*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        assert not d["smoke"], f
        assert max(d["equivalence_checks_maxabs"] or [0.0]) <= 1e-3, f
        recs.extend(d["records"]); errors.extend(d["errors"])
    return recs, errors


def integrity(recs, errors, synthetic):
    expected = {t["task_id"] for t in select_tasks()[:N_TASKS]}
    got = {r["task_id"] for r in recs}
    if not synthetic:
        seal = dict(line.split()[::-1] for line in (HERE / "SEAL.sha256").read_text().splitlines())
        for name in SEALED + R3_SEALED:
            assert seal[name] == sha(HERE / name), f"{name} changed after sealing"
    assert got <= expected
    assert all(r["split"] == split_for(r["task_id"]) for r in recs)
    assert all(r["decode_roundtrip_ok"] for r in recs)
    return {"n_records": len(recs), "n_tasks": len(got), "tasks_missing": len(expected - got), "errors": errors}


def answer_free_table(recs):
    for r in recs:
        r["free"] = {n: (n == 0 or not is_answer_bearing(r["prefix_texts"][i], r["own_answer"])) for i, n in enumerate(GRID)}


def choose_n_star(recs):
    train = [r for r in recs if r["split"] == "train"]
    frac = {n: sum(r["free"][n] for r in train) / len(train) for n in GRID if n > 0}
    ok = [n for n, f in frac.items() if f >= COVERAGE]
    return (max(ok) if ok else 16), frac


def hand_audit(recs, n_star, out):
    free = [r for r in recs if r["split"] == "train" and r["free"][n_star]]
    free.sort(key=lambda r: hashlib.sha256(f"20261003|{r['task_id']}|{r['candidate_idx']}".encode()).hexdigest())
    i = GRID.index(n_star)
    lines = [f"# Hand audit: 20 answer-free training prefixes at N* = {n_star} (label-free)", ""]
    for r in free[:20]:
        lines += [f"## {r['task_id']} #{r['candidate_idx']}", f"Problem: {r['question']}",
                  f"Own answer (first committed): {r['own_answer']}", "```", r["prefix_texts"][i], "```", ""]
    (out / "hand_audit.md").write_text("\n".join(lines))


def s_diff(r):
    return torch.tensor([float(r["prompt_tok"]), float(r["ref_steps"])])


def lp_at(n):
    def f(r):
        lp = r["gen_logprobs"][:n] or [0.0]
        return torch.tensor([sum(lp) / len(lp), min(lp)])
    return f


def feat_at(n):
    i = GRID.index(n)
    return lambda r: r["prefix_features"][i]


def visible_reader(rows, n, target_scores, y_ho, t_ho, sc):
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import FeatureUnion
    i = GRID.index(n)
    text = lambda r: f"Prompt:\n{gsm_prompt(r['question'])}\n\nVisible reasoning prefix:\n{r['prefix_texts'][i]}"
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


def verdict(res):
    if not A3.adjudicable(res):
        return "GSM8K_TRAIN_FIXED_PREFIX_NOT_ADJUDICABLE"
    lo, hi = res["paired_task_clustered_bootstrap_combined_minus_shortcut"]["ci95"]
    return ("GSM8K_TRAIN_FIXED_PREFIX_INCREMENT_POSITIVE" if lo > 0 else
            "GSM8K_TRAIN_FIXED_PREFIX_INCREMENT_NEGATIVE" if hi < 0 else "GSM8K_TRAIN_FIXED_PREFIX_INCREMENT_UNRESOLVED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["audit", "full"], required=True)
    ap.add_argument("--data", type=Path, default=HERE)
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    recs, errors = load(a.data)
    out = {"integrity": integrity(recs, errors, a.synthetic)}
    answer_free_table(recs)
    n_star, frac = choose_n_star(recs)
    out.update({"n_star": n_star, "answer_free_fraction_train": frac,
                "parseable_fraction": sum(r["parseable"] for r in recs) / len(recs),
                "hit_max_tokens_fraction": sum(r["hit_max_tokens"] for r in recs) / len(recs)})
    hand_audit(recs, n_star, a.out)
    if a.stage == "audit":
        (a.out / "audit_stage.json").write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2)); return

    parse = [r for r in recs if r["parseable"]]
    primary = [r for r in parse if r["free"][n_star]]
    gi = GRID.index(n_star)
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
    out["primary"] = res
    out["control_prompt_end"] = pe = A3.proper_arm("prompt_end_control", primary, [feat_at(n_star)], S, base=[pend])[0]
    if "paired_task_clustered_bootstrap_combined_minus_shortcut" in pe:
        survives = pe["paired_task_clustered_bootstrap_combined_minus_shortcut"]["ci95"][0] > 0
        out["headline_rule"] = "READABILITY_BEYOND_DIFFICULTY" if survives and res["verdict"].endswith("POSITIVE") else "MOSTLY_DIFFICULTY"
    out["secondary_prompt_end_only"] = A3.proper_arm("prompt_end_only", parse, [pend], s_diff)[0]
    out["secondary_run1_style"] = A3.guarded("run1_style", primary, feat_at(n_star), lp_at(n_star))[0]
    tasks_free = {}
    for r in recs:
        tasks_free.setdefault(r["task_id"], []).append(r["free"][n_star])
    out["secondary_task_level"] = A3.proper_arm("task_level", [r for r in parse if all(tasks_free[r["task_id"]])], [feat_at(n_star)], S)[0]
    out["secondary_excluding_reference_number"] = A3.proper_arm(
        "excluding_reference_number", [r for r in primary if not is_answer_bearing(r["prefix_texts"][gi], r["gold"])], [feat_at(n_star)], S)[0]
    out["secondary_grid"] = {str(n): A3.proper_arm(f"grid_{n}", [r for r in parse if r["free"][n]], [feat_at(n)],
                                                  lambda r, n=n: torch.cat([lp_at(n)(r), s_diff(r)]))[0] for n in GRID[1:]}
    out["accuracy_parseable"] = sum(r["success"] for r in parse) / len(parse)
    (a.out / "results.json").write_text(json.dumps(out, indent=2, default=float))

    def fmt(r):
        if "incremental_auroc_combined_minus_shortcut" not in r:
            return f"not run, below the minimums ({r['n_heldout_tasks']} held-out tasks, {r['n_heldout_negatives']} incorrect)"
        b = r["paired_task_clustered_bootstrap_combined_minus_shortcut"]["ci95"]
        return (f"surface {r['auroc_shortcut_only']:.3f}, hidden {r['auroc_hidden_only']:.3f}, both {r['auroc_hidden_plus_shortcuts']:.3f}, "
                f"increase {r['incremental_auroc_combined_minus_shortcut']:+.3f} [{b[0]:+.3f}, {b[1]:+.3f}]; held-out {r['n_heldout']} answers, "
                f"{r['n_heldout_tasks']} tasks, {r['n_heldout_negatives']} incorrect")

    def wq(d):
        return f"{d['estimate']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}] ({d['n_pairs']} pairs, {d['n_mixed_tasks']} questions)" if d.get("n_mixed_tasks") else "n/a"
    lines = ["# Run 6 results (GSM8K training split)", "", f"N* = {n_star}. Verdict: **{res['verdict']}**. Headline rule: **{out.get('headline_rule', 'n/a')}**", "",
             f"- Primary (S = log-prob + prompt tokens + reference steps; fixed combined model): {fmt(res)}",
             f"- Control, end-of-prompt state in the baseline: {fmt(pe)}"]
    if "control_within_question" in out:
        cw, vr = out["control_within_question"], out["secondary_visible_reader"]
        lines += [f"- Control, within-question AUROC: combined {wq(cw['combined'])}; hidden {wq(cw['hidden'])}; surface {wq(cw['shortcut'])}",
                  f"- Shuffled labels {res['shuffled_label_auroc_combined']:.3f}",
                  f"- Visible reader + S: AUROC {vr['auroc_visible_reader']:.3f}; combined minus reader {vr['paired_diff_combined_minus_reader']:+.3f} "
                  f"[{vr['bootstrap']['ci95'][0]:+.3f}, {vr['bootstrap']['ci95'][1]:+.3f}]"]
    lines += [f"- End-of-prompt state alone vs difficulty: {fmt(out['secondary_prompt_end_only'])}",
              f"- Run 1-style (S = log-prob only, arm()): {fmt(out['secondary_run1_style'])}",
              f"- Task level: {fmt(out['secondary_task_level'])}",
              f"- Also excluding prefixes with the reference number: {fmt(out['secondary_excluding_reference_number'])}",
              f"- Accuracy (parseable) {out['accuracy_parseable']:.3f}; parseable {out['parseable_fraction']:.3f}; hit 448 {out['hit_max_tokens_fraction']:.3f}",
              "", "Grid:"] + [f"- N={n}: {fmt(r)}" for n, r in out["secondary_grid"].items()]
    (a.out / "RESULTS.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()
