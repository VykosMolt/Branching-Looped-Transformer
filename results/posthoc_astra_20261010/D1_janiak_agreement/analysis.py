"""D1 analysis, implementing PLAN.md (post hoc). Reads cache/ (extract.py), writes results.json, probe_scores_<run>.npz
and SUMMARY_FOR_DENIS.md (tables generated here; the plain reading is included from reading.md if present).
If entropy/entropy_ouro_rltt.npz exists (entropy_pass.py), entropy is added as PLAN.md section 4 says.

usage: python -B analysis.py [run4 run6 run1]
"""
import hashlib
import importlib.util
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

import janiak_vendor as V

ROOT = "/home/moloch/ouro_project"
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
ENTROPY = os.path.join(HERE, "entropy", "entropy_ouro_rltt.npz")
sys.path[:0] = [ROOT, ROOT + "/shared", ROOT + "/shared/utilities/tests/manual", ROOT + "/opi/preanswer/gsm8k_fixed_prefix_20260927"]
import analyze_run3 as A3  # noqa: E402  (within_task_auroc, sealed)

_spec = importlib.util.spec_from_file_location(
    "fastarm", ROOT + "/opi/preanswer/horizon_cohort2_20261003/posthoc_audit_20261004/fastarm.py")
fastarm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fastarm)

N_STAR = {"run4": 48, "run6": 96, "run1": 48}
LABEL = {"run4": "Horizon Logic, second set (Run 4)", "run6": "GSM8K train (Run 6)", "run1": "Horizon Logic, first set (Run 1)"}
QFEAT = {"run4": ["prompt_tok", "proof_depth", "n_vars", "n_facts", "n_rules", "class_true", "class_false", "class_unknown"],
         "run6": ["prompt_tok", "ref_steps"]}
QFEAT["run1"] = QFEAT["run4"]
NB, SEED = 1000, 42
T1_METRICS = ("auroc", "auacc", "aurc", "brier", "log_loss")


def fold_of(qid, salt="D1_janiak"):
    return int(hashlib.sha256(f"{salt}|{qid}".encode()).hexdigest(), 16) % 5


# ---- question-clustered bootstrap (T1 rows are answers; siblings share a question) --------------------------------

class Clusters:
    def __init__(self, groups):
        uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
        order = np.argsort(inv, kind="stable")
        self.members = np.split(order, np.cumsum(np.bincount(inv))[:-1])

    def draws(self, n=NB, seed=SEED):
        rng = np.random.default_rng(seed)
        for _ in range(n):
            yield np.concatenate([self.members[i] for i in rng.integers(0, len(self.members), len(self.members))])


def metric_value(metric, s, y):
    return V.auroc(s, y) if metric == "auroc" else float(V.prompt_metrics(s, y)[metric])


def clustered_delta(left, right, y, groups, metric):
    """Paired left - right, question-clustered; same output and p rule as Denis's paired_bootstrap_delta."""
    point = metric_value(metric, left, y) - metric_value(metric, right, y)
    d = np.array([metric_value(metric, left[i], y[i]) - metric_value(metric, right[i], y[i]) for i in Clusters(groups).draws()])
    d = d[np.isfinite(d)]
    return {"metric": metric, "point_estimate": point, "ci_low": float(np.percentile(d, 2.5)),
            "ci_high": float(np.percentile(d, 97.5)),
            "p_two_sided": float(min(1.0, 2.0 * min(np.mean(d <= 0), np.mean(d >= 0)))), "n_valid": int(len(d))}


def clustered_auroc(s, y, groups):
    point = V.auroc(s, y)
    d = np.array([V.auroc(s[i], y[i]) for i in Clusters(groups).draws()])
    d = d[np.isfinite(d)]
    if not len(d):
        return {"point_estimate": point, "ci_low": None, "ci_high": None, "n_valid": 0}
    return {"point_estimate": point, "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)),
            "n_valid": int(len(d))}


def clustered_stratum(s, y, groups, mask):
    """Denis's stratum_readout with a question-clustered interval."""
    s, y, g = s[mask], y[mask], np.asarray(groups)[mask]
    body = {"n": int(mask.sum()), "n_correct": int((y > 0.5).sum()), "n_wrong": int((y <= 0.5).sum()),
            "n_questions": int(len(set(g.tolist()))), "base_accuracy": float(np.mean(y)) if len(y) else float("nan")}
    if min(body["n_correct"], body["n_wrong"]) < V.MIN_STRATUM_CLASS:
        return {**body, "auroc": None, "reported": False}
    return {**body, "auroc": clustered_auroc(s, y, g), "reported": True}


def crossfit_residuals_multi(target, covariates, folds):
    """Denis's crossfit_residuals with several covariates (out-of-fold least squares)."""
    target, X, folds = np.asarray(target, float), np.asarray(covariates, float), np.asarray(folds)
    res = np.full(len(target), np.nan)
    for f in np.unique(folds):
        te = folds == f
        tr = ~te & np.isfinite(target) & np.isfinite(X).all(1)
        D = np.column_stack([np.ones(int(tr.sum())), X[tr]])
        coef, *_ = np.linalg.lstsq(D, target[tr], rcond=None)
        res[te] = target[te] - np.column_stack([np.ones(int(te.sum())), X[te]]) @ coef
    return res


# ---- data ---------------------------------------------------------------------------------------------------------

def load(run):
    meta = pickle.load(open(os.path.join(CACHE, f"{run}_meta.pkl"), "rb"))
    H = np.load(os.path.join(CACHE, f"{run}_H.npy"), mmap_mode="r")
    free = [i for i, m in enumerate(meta) if m["free"]]
    assert [meta[i]["h_row"] for i in free] == list(range(len(free))) and H.shape[0] == len(free)
    entropy = False
    if os.path.exists(ENTROPY):
        z = np.load(ENTROPY)
        ent = {(str(r), str(q), int(c)): (float(a), float(b))
               for r, q, c, a, b in zip(z["run"], z["qid"], z["cand"], z["ent_full"], z["ent_pre"])}
        keys = [(run, m["qid"], m["cand"]) for m in meta]
        missing = [k for k in keys if k not in ent]
        assert not missing, f"entropy file lacks {len(missing)} {run} answers; refusing to mix"
        for m, k in zip(meta, keys):
            m["ent_full"], m["ent_pre"] = ent[k]
        entropy = True
    # per-question plurality (Denis's rule) and each answer's vote share / agreement with the plurality
    by_q = defaultdict(list)
    for i, m in enumerate(meta):
        by_q[m["qid"]].append(i)
    questions = {}
    for q, ix in by_q.items():
        group = [{"predicted_answer": meta[i]["pred"], "gold_answer": meta[i]["gold"], "logprob_score": meta[i]["lp_full"],
                  "length_score": meta[i]["n_gen_tok"], "entropy_score": meta[i].get("ent_full")} for i in ix]
        agg = V.aggregate_question(group)
        parsed = [meta[i]["pred"] for i in ix if V.is_parseable_answer(meta[i]["pred"])]
        for i in ix:
            p = meta[i]["pred"]
            meta[i]["vote_share"] = parsed.count(p) / len(parsed) if V.is_parseable_answer(p) else np.nan
            meta[i]["agrees"] = float(V.is_parseable_answer(p) and str(p) == agg["winner"])
        questions[q] = {**agg, "ix": ix, "qfeat": meta[ix[0]]["qfeat"], "n": len(ix)}
    return meta, H, free, questions, entropy


def fit_probe(meta, H, free):
    """Section 1: OOF probe score for every answer-free answer, fixed combined estimator on hidden states alone
    (fastarm.Arm = analyze_run3._fit_score with PCA via the Gram matrix), outer folds by question hash, inner folds by
    a second hash over the training questions."""
    y = np.array([meta[i]["label"] if meta[i]["label"] is not None else -1 for i in free])
    lab = y >= 0
    fo = np.array([fold_of(meta[i]["qid"]) for i in free])
    fi = np.array([fold_of(meta[i]["qid"], "D1_janiak_inner") for i in free])
    scores, settings = np.full(len(free), np.nan), []
    S0 = np.zeros((len(free), 0), dtype=np.float32)
    for k in range(5):
        itv, iho = np.where(lab & (fo != k))[0], np.where(fo == k)[0]
        inner = fi[itv]
        folds_tv = [(np.where(inner != j)[0], np.where(inner == j)[0]) for j in range(5)]
        p, h = fastarm.Arm([H], S0, np.where(lab, y, 0), folds_tv, itv, iho).fit_score([0], False)
        scores[iho] = p
        settings.append({"fold": k, "n_train": int(len(itv)), "n_scored": int(len(iho)), **h})
        print(f"  probe fold {k}: train {len(itv)}, scored {len(iho)}, k={h['k']} C={h['C']} inner AUROC {h['cv']:.3f}", flush=True)
    out = np.full(len(meta), np.nan)
    out[free] = scores
    return out, settings


# ---- T1 -----------------------------------------------------------------------------------------------------------

def t1(run, meta, probe, entropy):
    rows = [i for i, m in enumerate(meta) if m["free"] and m["label"] is not None]
    y = np.array([meta[i]["label"] for i in rows], float)
    g = np.array([meta[i]["qid"] for i in rows])
    folds = np.array([fold_of(q) for q in g])
    col = lambda k: np.array([meta[i][k] for i in rows], float)
    qf = np.array([meta[i]["qfeat"] for i in rows], float)
    vote, agrees, pr = col("vote_share"), col("agrees"), probe[rows]
    b0 = [col("n_gen_tok"), col("lp_full"), col("hit_max"), vote, agrees] + ([col("ent_full")] if entropy else [])
    f0 = [col("lp_win_mean"), col("lp_win_min")] + ([col("ent_pre")] if entropy else [])
    X = {"B0_traj": np.column_stack(b0 + [qf]), "B1_traj": np.column_stack(b0 + [qf, pr]),
         "F0_window": np.column_stack(f0 + [qf]), "F1_window": np.column_stack(f0 + [qf, pr])}
    pred = {k: V.crossfit_logistic_predictions(x, y, folds, seed=SEED) for k, x in X.items()}
    pred["probe_alone"] = pr
    by_q = defaultdict(list)
    for i, q in enumerate(g):
        by_q[q].append(i)
    mixed = [q for q, ix in by_q.items() if 0 < y[ix].mean() < 1]
    res = {
        "n_answers": len(rows), "n_questions": len(by_q), "n_incorrect": int((y == 0).sum()),
        "accuracy": float(y.mean()), "n_mixed_questions": len(mixed),
        "n_mixed_pairs": int(sum(y[by_q[q]].sum() * (len(by_q[q]) - y[by_q[q]].sum()) for q in mixed)),
        "n_in_window_commit_missed_by_detector": int(sum(meta[i]["in_window_commit"] for i in rows)),
        "features": {"B0_traj": ["n_gen_tok", "mean_logprob_full", "hit_max_tokens", "vote_share", "agrees_with_plurality"]
                     + (["entropy_full"] if entropy else []) + QFEAT[run],
                     "B1_traj": "B0_traj + probe", "F0_window": [f"mean_logprob_first{N_STAR[run]}", f"min_logprob_first{N_STAR[run]}"]
                     + ([f"entropy_first{N_STAR[run]}"] if entropy else []) + QFEAT[run], "F1_window": "F0_window + probe"},
        "readouts": {k: {"auroc": V.auroc(p, y), **V.prompt_metrics(p, y)} for k, p in pred.items()},
        "paired_deltas": {}, "within_question": {},
    }
    for left, right in (("B1_traj", "B0_traj"), ("F1_window", "F0_window")):
        res["paired_deltas"][f"{left}_minus_{right}"] = {m: clustered_delta(pred[left], pred[right], y, g, m) for m in T1_METRICS}
    for k in ("B0_traj", "B1_traj", "probe_alone", "F0_window", "F1_window"):
        res["within_question"][k] = A3.within_task_auroc(pred[k], y.astype(int), list(g))
    # Orgad-style controls, agreement = the answer's own vote share
    levels = sorted(set(np.round(vote, 6).tolist()))
    resid = crossfit_residuals_multi(pr, np.column_stack([vote, agrees]), folds)
    res["orgad"] = {
        "pearson_probe_vote_share": V._pearson(pr, vote), "spearman_probe_vote_share": V.spearman(pr, vote),
        "marginal_auroc": {"probe": clustered_auroc(pr, y, g), "vote_share": clustered_auroc(vote, y, g),
                           "agrees_with_plurality": clustered_auroc(agrees, y, g)},
        "strata_agrees": {f"agrees={a}": clustered_stratum(pr, y, g, agrees == a) for a in (1.0, 0.0)},
        "strata_vote_share": {f"{lv:.3f}": clustered_stratum(pr, y, g, np.round(vote, 6) == lv) for lv in levels},
        "residual_auroc_probe_given_vote_share_and_agrees": clustered_auroc(resid, y, g),
    }
    return res, pred, rows


# ---- T2 -----------------------------------------------------------------------------------------------------------

def orgad_t2(geometry, agreement, outcomes, folds, cols):
    """controls/orgad_agreement_control.analyze_model on arrays, the probe aggregate in place of rmd_tail_q20."""
    base = [c for c in cols if c != "probe_mean"]
    voteless = [c for c in base if c != "vote_agreement"]
    specs = {"B0": base, "B1": base + ["probe_mean"], "B0_voteless": voteless, "B0_rmd_for_vote": voteless + ["probe_mean"]}
    pred = {k: V.crossfit_logistic_predictions(np.column_stack([cols[c] for c in v]), outcomes, folds, seed=SEED)
            for k, v in specs.items()}
    deltas = {f"{l}_minus_{r}": V.paired_bootstrap_delta(pred[l], pred[r], outcomes, metric="aurc", n_bootstrap=NB, seed=SEED)
              for l, r in (("B1", "B0"), ("B0_rmd_for_vote", "B0"), ("B1", "B0_rmd_for_vote"), ("B0_rmd_for_vote", "B0_voteless"))}
    return {
        "n_prompts": int(len(outcomes)), "n_with_probe": int(np.isfinite(geometry).sum()),
        "base_accuracy": float(np.mean(outcomes)),
        "redundancy": {"pearson": V._pearson(geometry, agreement), "spearman": V.spearman(geometry, agreement)},
        "marginal_auroc": {"probe_mean": V.bootstrap_auroc(geometry, outcomes, n_bootstrap=NB, seed=SEED),
                           "vote_agreement": V.bootstrap_auroc(agreement, outcomes, n_bootstrap=NB, seed=SEED)},
        "strata": {n: V.stratum_readout(geometry, outcomes, m, n_bootstrap=NB, seed=SEED)
                   for n, m in V.agreement_strata(agreement).items()},
        "agreement_levels": V.agreement_level_counts(agreement, outcomes),
        "residual_auroc": {
            "probe_given_agreement": V.bootstrap_auroc(V.crossfit_residuals(geometry, agreement, folds), outcomes, n_bootstrap=NB, seed=SEED),
            "agreement_given_probe": V.bootstrap_auroc(V.crossfit_residuals(agreement, geometry, folds), outcomes, n_bootstrap=NB, seed=SEED)},
        "readouts": {k: V.prompt_metrics(p, outcomes) for k, p in pred.items()},
        "paired_deltas_aurc": deltas,
    }


def t2(run, meta, probe, questions, entropy):
    qids = sorted(questions)
    Q = [questions[q] for q in qids]
    y = np.array([d["outcome"] for d in Q], float)
    folds = np.array([fold_of(q) for q in qids])
    pm = np.array([np.nanmean(probe[d["ix"]]) if np.isfinite(probe[d["ix"]]).any() else np.nan for d in Q])
    cols = {"length": np.array([d["length"] for d in Q]), "logprob": np.array([d["logprob"] for d in Q]),
            "vote_agreement": np.array([d["vote_agreement"] for d in Q])}
    if entropy:
        cols = {"length": cols["length"], "entropy": np.array([d["entropy"] for d in Q]), **{k: cols[k] for k in ("logprob", "vote_agreement")}}
    qf = np.array([d["qfeat"] for d in Q], float)
    base = np.column_stack(list(cols.values()))
    X = {"B0": base, "B0q": np.column_stack([base, qf]), "B1": np.column_stack([base, pm]), "B1q": np.column_stack([base, qf, pm])}
    valid = np.array([d["valid_plurality"] for d in Q])
    out = {"features": {"B0": list(cols), "B0q": list(cols) + QFEAT[run], "B1": list(cols) + ["probe_mean"],
                        "B1q": list(cols) + QFEAT[run] + ["probe_mean"]}, "populations": {}}
    for pop, mask in (("full_population", np.ones(len(Q), bool)), ("valid_plurality", valid)):
        yy, ff = y[mask], folds[mask]
        pred = {k: V.crossfit_logistic_predictions(x[mask], yy, ff, seed=SEED) for k, x in X.items()}
        e = {"n_prompts": int(mask.sum()), "n_incorrect_plurality": int((yy == 0).sum()),
             "n_automatic_failures": int((~valid[mask]).sum()), "n_no_answer_free_answer": int((~np.isfinite(pm[mask])).sum()),
             "base_accuracy": float(yy.mean()),
             "models": {k: {"features": out["features"][k], "metrics": V.prompt_metrics(p, yy), "auroc": V.auroc(p, yy)}
                        for k, p in pred.items()},
             "paired_deltas": {}}
        for left, right in (("B1", "B0"), ("B1q", "B0q")):
            label = f"{left}_minus_{right}"
            for metric in V.METRIC_NAMES:  # Denis's seed rule from run_incremental_analysis
                e["paired_deltas"][f"{label}_{metric}"] = V.paired_bootstrap_delta(
                    pred[left], pred[right], yy, metric=metric, n_bootstrap=NB, seed=SEED + 1000 + len(metric) + len(label))
        e["orgad"] = orgad_t2(pm[mask], cols["vote_agreement"][mask], yy, ff,
                              {**{k: v[mask] for k, v in cols.items()}, "probe_mean": pm[mask]})
        out["populations"][pop] = e
    return out


# ---- report -------------------------------------------------------------------------------------------------------

def f3(x, d=3):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def iv(e, d=3, signed=True):
    if not e or e.get("point_estimate") is None or e.get("ci_low") is None:
        return "n/a"
    s = "+" if signed else ""
    out = f"{e['point_estimate']:{s}.{d}f} [{e['ci_low']:{s}.{d}f}, {e['ci_high']:{s}.{d}f}]"
    return out + (f" p={e['p_two_sided']:.3f}" if "p_two_sided" in e else "")


def wq(d):
    return f"{d['estimate']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]" if d.get("n_mixed_tasks") else "n/a"


def report(results):
    ent = results["entropy_included"]
    tag = "with entropy" if ent else "WITHOUT entropy (B0 minus entropy; entropy_pass.py not yet run)"
    L = ["# D1 for Denis Janiak: does the pre-answer readout add anything once agreement is in the baseline?", "",
         f"Post hoc (PLAN.md, sha256 `{results['plan_sha256'][:16]}...`, written before any result). Ouro-RLTT, 4 sampled answers per question, "
         f"5 question-hash folds, everything out of fold. Baselines are **{tag}**. AUACC higher is better, AURC lower is better; "
         "deltas are left minus right with 95% bootstrap intervals (1,000 draws). The probe reads only hidden states pooled over "
         "the prompt and the first N* generated tokens of an answer that has stated no answer by then; agreement comes from the "
         "4 completed answers.", ""]
    if os.path.exists(os.path.join(HERE, "reading.md")):
        L += ["## Plain reading", "", open(os.path.join(HERE, "reading.md")).read().strip(), ""]
    L += ["## T1: per-trajectory correctness (rows = answer-free answers)", "",
          "B0 = completed length, completed mean log-prob, hit-limit flag, vote share, agrees-with-plurality, question features. "
          "B1 = B0 + probe score. F0 = the paper's fair baseline (window log-prob mean/min + question features, nothing from completed answers). "
          "Intervals are question-clustered.", "",
          "| run | answers (questions, incorrect) | readout | AUROC | AUACC | AURC | Brier |", "|---|---|---|---:|---:|---:|---:|"]
    for run, r in results["runs"].items():
        t = r["T1"]
        for k in ("B0_traj", "B1_traj", "probe_alone", "F0_window", "F1_window"):
            m = t["readouts"][k]
            L.append(f"| {run} N*={N_STAR[run]} | {t['n_answers']} ({t['n_questions']}, {t['n_incorrect']}) | {k} | {f3(m['auroc'])} | "
                     f"{f3(m['auacc'])} | {f3(m['aurc'])} | {f3(m['brier'])} |")
    L += ["", "| run | contrast | ΔAUROC | ΔAUACC | ΔAURC | ΔBrier |", "|---|---|---|---|---|---|"]
    for run, r in results["runs"].items():
        for c, d in r["T1"]["paired_deltas"].items():
            L.append(f"| {run} | {c} | {iv(d['auroc'])} | {iv(d['auacc'], 4)} | {iv(d['aurc'], 4)} | {iv(d['brier'], 4)} |")
    L += ["", "Within-question AUROC (mixed questions only; the vote share differs between a correct and an incorrect sibling, so B0 is strong here by construction):", "",
          "| run | mixed questions / pairs | B0 | B1 | probe alone | F0 | F1 |", "|---|---|---|---|---|---|---|"]
    for run, r in results["runs"].items():
        t, w = r["T1"], r["T1"]["within_question"]
        L.append(f"| {run} | {t['n_mixed_questions']} / {t['n_mixed_pairs']} | {wq(w['B0_traj'])} | {wq(w['B1_traj'])} | "
                 f"{wq(w['probe_alone'])} | {wq(w['F0_window'])} | {wq(w['F1_window'])} |")
    L += ["", "### T1 Orgad-style controls (agreement = the answer's own vote share)", "",
          "| run | Pearson / Spearman probe vs vote share | AUROC probe | AUROC vote share | probe in agrees=1 | probe in agrees=0 | residual of probe on (vote share, agrees) |",
          "|---|---|---|---|---|---|---|"]
    for run, r in results["runs"].items():
        o = r["T1"]["orgad"]
        st = lambda s: (f"{iv(s['auroc'], signed=False)} (n={s['n']}, {s['n_wrong']} wrong)" if s["reported"]
                        else f"not reported (n={s['n']}, {s['n_wrong']} wrong)")
        L.append(f"| {run} | {f3(o['pearson_probe_vote_share'])} / {f3(o['spearman_probe_vote_share'])} | {iv(o['marginal_auroc']['probe'], signed=False)} | "
                 f"{iv(o['marginal_auroc']['vote_share'], signed=False)} | {st(o['strata_agrees']['agrees=1.0'])} | {st(o['strata_agrees']['agrees=0.0'])} | "
                 f"{iv(o['residual_auroc_probe_given_vote_share_and_agrees'], signed=False)} |")
    L += ["", "Probe AUROC inside each vote-share level (fewer than 10 of either class: not reported):", "",
          "| run | vote share | answers | incorrect | accuracy | probe AUROC |", "|---|---:|---:|---:|---:|---|"]
    for run, r in results["runs"].items():
        for lv, s in r["T1"]["orgad"]["strata_vote_share"].items():
            L.append(f"| {run} | {lv} | {s['n']} | {s['n_wrong']} | {f3(s['base_accuracy'])} | {iv(s['auroc'], signed=False) if s['reported'] else 'not reported'} |")
    L += ["", "## T2: plurality-answer correctness (rows = questions), Denis's table", "",
          "B0 = length, logprob, vote_agreement over the 4 completed answers" + (" plus entropy" if ent else " (entropy omitted)") +
          "; B0q = B0 + question features; B1 = B0 + mean probe score over the question's answer-free answers; B1q = B0q + the same.", ""]
    for run, r in results["runs"].items():
        for pop, e in r["T2"]["populations"].items():
            L += [f"### {run} / {pop}", "",
                  f"{e['n_prompts']} questions; {e['n_incorrect_plurality']} incorrect pluralities (base accuracy {f3(e['base_accuracy'])}); "
                  f"automatic failures {e['n_automatic_failures']}; questions with no answer-free answer (probe imputed) {e['n_no_answer_free_answer']}.", "",
                  "| model | AUROC | AUACC (higher) | AURC (lower) | Brier (lower) | log loss (lower) |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
            for k, m in e["models"].items():
                mm = m["metrics"]
                L.append(f"| {k} | {f3(m['auroc'])} | {f3(mm['auacc'])} | {f3(mm['aurc'])} | {f3(mm['brier'])} | {f3(mm['log_loss'])} |")
            L += ["", "| comparison | AUACC | AURC | Brier | log loss |", "| --- | --- | --- | --- | --- |"]
            for c in ("B1_minus_B0", "B1q_minus_B0q"):
                L.append(f"| {c} | " + " | ".join(iv(e["paired_deltas"][f"{c}_{m}"], 4) for m in V.METRIC_NAMES) + " |")
            L.append("")
    L += ["### T2 Orgad controls (probe aggregate in place of rmd_tail_q20)", "",
          "| run | population | n (with probe) | Pearson / Spearman vs vote | AUROC probe | AUROC vote | unanimous stratum | split stratum | probe given vote (residual) | vote given probe (residual) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for run, r in results["runs"].items():
        for pop, e in r["T2"]["populations"].items():
            o = e["orgad"]
            st = lambda s: (f"{iv(s['auroc'], signed=False)} (n={s['n_prompts']}, {s['n_wrong']} wrong)" if s["reported"]
                            else f"not reported (n={s['n_prompts']}, {s['n_wrong']} wrong)")
            L.append(f"| {run} | {pop} | {o['n_prompts']} ({o['n_with_probe']}) | {f3(o['redundancy']['pearson'])} / {f3(o['redundancy']['spearman'])} | "
                     f"{iv(o['marginal_auroc']['probe_mean'], signed=False)} | {iv(o['marginal_auroc']['vote_agreement'], signed=False)} | "
                     f"{st(o['strata']['unanimous'])} | {st(o['strata']['split'])} | {iv(o['residual_auroc']['probe_given_agreement'], signed=False)} | "
                     f"{iv(o['residual_auroc']['agreement_given_probe'], signed=False)} |")
    L += ["", "Substitution both ways (AURC, lower is better; Denis's four contrasts):", "",
          "| run | population | B1 − B0 | probe for vote − B0 | B1 − (probe for vote) | probe added to voteless |", "|---|---|---|---|---|---|"]
    for run, r in results["runs"].items():
        for pop, e in r["T2"]["populations"].items():
            d = e["orgad"]["paired_deltas_aurc"]
            L.append(f"| {run} | {pop} | {iv(d['B1_minus_B0'], 4)} | {iv(d['B0_rmd_for_vote_minus_B0'], 4)} | "
                     f"{iv(d['B1_minus_B0_rmd_for_vote'], 4)} | {iv(d['B0_rmd_for_vote_minus_B0_voteless'], 4)} |")
    L += ["", "## Primary contrasts (PLAN.md reading rule: AURC delta B1 − B0, interval entirely below zero)", "",
          "| contrast | ΔAURC | raw p | Holm p | reading |", "|---|---|---|---|---|"]
    for c in results["primary_contrasts"]:
        L.append(f"| {c['name']} | {iv(c['delta'], 4)} | {c['delta']['p_two_sided']:.3f} | {c['holm_p']:.3f} | {c['reading']} |")
    open(os.path.join(HERE, "SUMMARY_FOR_DENIS.md"), "w").write("\n".join(L) + "\n")


def main(runs):
    plan_sha = hashlib.sha256(open(os.path.join(HERE, "PLAN.md"), "rb").read()).hexdigest()
    assert open(os.path.join(HERE, "PLAN.sha256")).read().split()[0] == plan_sha, "PLAN.md changed after hashing"
    results = {"plan_sha256": plan_sha, "post_hoc": True, "entropy_included": None, "n_bootstrap": NB, "seed": SEED, "runs": {}}
    for run in runs:
        print(f"[{run}] loading", flush=True)
        meta, H, free, questions, entropy = load(run)
        assert results["entropy_included"] in (None, entropy)
        results["entropy_included"] = entropy
        print(f"[{run}] probe on {len(free)} answer-free answers", flush=True)
        probe, settings = fit_probe(meta, H, free)
        del H
        np.savez(os.path.join(HERE, f"probe_scores_{run}.npz"), qid=np.array([m["qid"] for m in meta]),
                 cand=np.array([m["cand"] for m in meta]), probe=probe, label=np.array([np.nan if m["label"] is None else m["label"] for m in meta]),
                 free=np.array([m["free"] for m in meta]), fold=np.array([fold_of(m["qid"]) for m in meta]))
        print(f"[{run}] T1", flush=True)
        r1, _, _ = t1(run, meta, probe, entropy)
        print(f"[{run}] T2", flush=True)
        r2 = t2(run, meta, probe, questions, entropy)
        results["runs"][run] = {"label": LABEL[run], "n_star": N_STAR[run], "n_answers_total": len(meta),
                                "n_questions_total": len(questions), "n_answer_free": len(free),
                                "n_answer_free_unlabelled": int(sum(meta[i]["label"] is None for i in free)),
                                "probe_settings": settings, "T1": r1, "T2": r2}
        print(f"[{run}] T1 B1-B0 AURC {iv(r1['paired_deltas']['B1_traj_minus_B0_traj']['aurc'], 4)}; "
              f"T2 full B1-B0 AURC {iv(r2['populations']['full_population']['paired_deltas']['B1_minus_B0_aurc'], 4)}", flush=True)
    prim = []
    for run in ("run4", "run6"):
        if run in results["runs"]:
            r = results["runs"][run]
            prim.append({"name": f"{run} T1", "delta": r["T1"]["paired_deltas"]["B1_traj_minus_B0_traj"]["aurc"]})
            prim.append({"name": f"{run} T2 full_population", "delta": r["T2"]["populations"]["full_population"]["paired_deltas"]["B1_minus_B0_aurc"]})
    order = sorted(range(len(prim)), key=lambda i: prim[i]["delta"]["p_two_sided"])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(prim) - rank) * prim[i]["delta"]["p_two_sided"]))
        prim[i]["holm_p"] = running
    for c in prim:
        c["reading"] = "adds beyond agreement" if c["delta"]["ci_high"] < 0 else "not shown"
    results["primary_contrasts"] = prim
    json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=float)
    report(results)
    print("wrote results.json and SUMMARY_FOR_DENIS.md", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["run4", "run6", "run1"])
