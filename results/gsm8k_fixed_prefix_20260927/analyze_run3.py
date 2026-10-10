"""Run 3 analysis, implementing PLAN.md as sealed. Reads the run3 shards and writes results.json,
RESULTS.md and hand_audit.md. Stage 'audit' runs everything label-free (integrity checks,
answer-free fractions, N*, the hand-audit list) and stops; stage 'full' then opens labels and runs
the endpoints. --data/--out exist only for the dry run on synthetic shards.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (ROOT, ROOT / "shared", ROOT / "shared/utilities/tests/manual", HERE):
    sys.path.insert(0, str(p))

from answer_detector_gsm8k import is_answer_bearing  # noqa: E402
from bg_steering_suite_lib import gsm_prompt  # noqa: E402
from bg_v2_overnight_horizon_generate import split_for  # noqa: E402
from bg_v2_overnight_horizon_analysis import SEED, bootstrap_task_clustered  # noqa: E402
from bg_v2_malformed_sibling_control import arm  # noqa: E402
from gen_run3 import N_TASKS, OLD_N, PREFIX_GRID as GRID, select_tasks  # noqa: E402
from proto_introspection_controls_analysis import auroc, cv_oof_combined, cv_oof_scores  # noqa: E402

COVERAGE, MIN_TASKS, MIN_NEG = 0.75, 30, 30


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(data: Path) -> tuple[list[dict], list[str]]:
    recs, errors = [], []
    for f in sorted(glob.glob(str(data / "run3_shard_*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        assert not d["smoke"], f
        assert max(d["equivalence_checks_maxabs"] or [0.0]) <= 1e-3, f
        assert max(d["tf_checks_maxabs"] or [0.0]) <= 1e-3, f
        recs.extend(d["records"])
        errors.extend(d["errors"])
    return recs, errors


def integrity(recs: list[dict], errors: list[str], synthetic: bool) -> dict:
    expected = {t["task_id"] for t in select_tasks()[:N_TASKS]}
    got = {r["task_id"] for r in recs}
    if not synthetic:
        seal = dict(line.split()[::-1] for line in (HERE / "SEAL.sha256").read_text().splitlines())
        for name in ("answer_detector_gsm8k.py", "gen_run3.py", "analyze_run3.py", "PLAN.md"):
            assert seal[name] == sha(HERE / name), f"{name} changed after sealing"
    assert got <= expected and all(r["dataset_index"] >= OLD_N for r in recs)
    assert all(r["split"] == split_for(r["task_id"]) for r in recs)
    assert all(r["decode_roundtrip_ok"] for r in recs)
    return {"n_records": len(recs), "n_tasks": len(got), "tasks_missing": len(expected - got), "errors": errors,
            "disjoint_from_published_170": True, "splits_match_split_for": True, "decode_roundtrip_all": True}


def answer_free_table(recs: list[dict]) -> None:
    for r in recs:
        r["free"] = {n: (n == 0 or not is_answer_bearing(r["prefix_texts"][i], r["own_answer"]))
                     for i, n in enumerate(GRID)}


def choose_n_star(recs: list[dict]) -> tuple[int, dict]:
    train = [r for r in recs if r["split"] == "train"]
    frac = {n: sum(r["free"][n] for r in train) / len(train) for n in GRID if n > 0}
    ok = [n for n, f in frac.items() if f >= COVERAGE]
    return (max(ok) if ok else 16), frac


def fractions_by_split(recs: list[dict]) -> dict:
    out = {}
    for split in ("train", "val", "heldout"):
        rs = [r for r in recs if r["split"] == split]
        out[split] = {str(n): round(sum(r["free"][n] for r in rs) / max(len(rs), 1), 4) for n in GRID if n > 0}
    return out


def hand_audit(recs: list[dict], n_star: int, out: Path) -> None:
    """Label-free: the problem, the candidate's own answer (its output, not its correctness), the prefix."""
    free = [r for r in recs if r["split"] == "train" and r["free"][n_star]]
    free.sort(key=lambda r: hashlib.sha256(f"20260927|{r['task_id']}|{r['candidate_idx']}".encode()).hexdigest())
    i = GRID.index(n_star)
    lines = [f"# Hand audit: 20 answer-free training prefixes at N* = {n_star} (label-free)", ""]
    for r in free[:20]:
        lines += [f"## {r['task_id']} #{r['candidate_idx']}", f"Problem: {r['question']}",
                  f"Own answer (first committed): {r['own_answer']}", "```", r["prefix_texts"][i], "```", ""]
    (out / "hand_audit.md").write_text("\n".join(lines))


def run_arm(name: str, rows: list[dict], feat, sc, label: str = "success"):
    Xh = torch.stack([feat(r).flatten().float() for r in rows])
    Xs = torch.stack([sc(r).float() for r in rows])
    y = torch.tensor([1.0 if r[label] else 0.0 for r in rows])
    tasks = [r["task_id"] for r in rows]
    tv = torch.tensor([r["split"] in ("train", "val") for r in rows])
    ho = torch.tensor([r["split"] == "heldout" for r in rows])
    t_tv = [t for t, m in zip(tasks, tv.tolist()) if m]
    t_ho = [t for t, m in zip(tasks, ho.tolist()) if m]
    res = arm(name, Xh[tv], Xs[tv], y[tv], t_tv, Xh[ho], Xs[ho], y[ho], t_ho)
    scores = res.pop("_scores")
    res.update({"n_heldout": int(ho.sum()), "n_heldout_tasks": len(set(t_ho)),
                "n_heldout_negatives": int((y[ho] == 0).sum()), "n_trainval": int(tv.sum())})
    return res, scores, y[ho], t_ho


def counts(rows: list[dict], label: str = "success") -> dict:
    ho = [r for r in rows if r["split"] == "heldout"]
    return {"n_rows": len(rows), "n_heldout_tasks": len({r["task_id"] for r in ho}),
            "n_heldout_negatives": sum(1 for r in ho if not r[label])}


def adjudicable(c: dict) -> bool:
    return c["n_heldout_tasks"] >= MIN_TASKS and c["n_heldout_negatives"] >= MIN_NEG


def guarded(name: str, rows: list[dict], feat, sc, label: str = "success"):
    """run_arm for a secondary analysis, skipped (not crashed) below the adjudicability minimums."""
    c = counts(rows, label)
    if not adjudicable(c):
        return {"skipped": "below the adjudicability minimums", **c}, None
    res, scores, _, _ = run_arm(name, rows, feat, sc, label)
    return res, scores


def prefix_lp_vec(n: int):
    def f(r):
        lp = r["gen_logprobs"][:n] or [0.0]
        return torch.tensor([sum(lp) / len(lp), min(lp)])
    return f


def prefix_ext_vec(n: int):
    """Surface features at N plus the teacher-forced mean and last entropy of the tokens read."""
    def f(r):
        ent = r["tf_entropies"][:n] or [0.0]
        return torch.cat([prefix_lp_vec(n)(r), torch.tensor([sum(ent) / len(ent), ent[-1]])])
    return f


def feat_at(n: int):
    i = GRID.index(n)
    return lambda r: r["prefix_features"][i]


def old_sc_vec(cut_key: str):
    """The published GSM8K shortcut set: question and prompt length, tokens before the cut, and the
    teacher-forced mean log-probability, mean entropy and last entropy of those tokens."""
    def f(r):
        n = r[cut_key]
        lp, ent = r["tf_logprobs"][:n], r["tf_entropies"][:n]
        return torch.tensor([r["question_chars"] / 1000.0, r["prompt_tok"] / 512.0, n / 256.0,
                             sum(lp) / n, sum(ent) / n, ent[-1]])
    return f


def visible_reader(rows: list[dict], n: int, target_scores, y_ho, t_ho, sc=None) -> dict:
    """TF-IDF reader of the prompt and the first n tokens; with `sc`, the same surface features are appended."""
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
    uniq = sorted({r["task_id"] for r in tv})
    g = torch.Generator().manual_seed(SEED)
    order = [uniq[j] for j in torch.randperm(len(uniq), generator=g).tolist()]
    fold = {u: k % 5 for k, u in enumerate(order)}
    folds = [fold[r["task_id"]] for r in tv]

    S_tv = np.stack([sc(r).float().numpy() for r in tv]) if sc else None
    S_ho = np.stack([sc(r).float().numpy() for r in ho]) if sc else None

    def add_s(M, S_fit, S_use):
        if S_fit is None:
            return M
        mu, sd = S_fit.mean(0), S_fit.std(0) + 1e-6
        return hstack([M, csr_matrix((S_use - mu) / sd)]).tocsr()

    def vec():
        return FeatureUnion([
            ("w", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), strip_accents="unicode", sublinear_tf=True, max_features=40000)),
            ("c", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=60000)),
        ])
    best = None
    for C in (0.01, 0.1, 1.0, 10.0):
        oof = [0.0] * len(tv)
        for k in range(5):
            tr = [j for j, f in enumerate(folds) if f != k]
            va = [j for j, f in enumerate(folds) if f == k]
            v = vec().fit([text(tv[j]) for j in tr])
            m = LogisticRegression(C=C, solver="liblinear", max_iter=2000, random_state=20260729)
            Sf = S_tv[tr] if sc else None
            m.fit(add_s(v.transform([text(tv[j]) for j in tr]), Sf, Sf), [y_tv[j] for j in tr])
            p = m.predict_proba(add_s(v.transform([text(tv[j]) for j in va]), Sf, S_tv[va] if sc else None))[:, 1]
            for j, pj in zip(va, p):
                oof[j] = float(pj)
        a = roc_auc_score(y_tv, oof)
        if best is None or a > best[0]:
            best = (a, C)
    v = vec().fit([text(r) for r in tv])
    m = LogisticRegression(C=best[1], solver="liblinear", max_iter=2000, random_state=20260729)
    m.fit(add_s(v.transform([text(r) for r in tv]), S_tv, S_tv), y_tv)
    s_v = torch.tensor(m.predict_proba(add_s(v.transform([text(r) for r in ho]), S_tv, S_ho))[:, 1], dtype=torch.double)
    boot = bootstrap_task_clustered(target_scores, s_v, y_ho, t_ho, rounds=2000)
    return {"cv_auroc": best[0], "C": best[1], "auroc_visible_reader": auroc(s_v, y_ho),
            "auroc_hidden_plus_surface": auroc(target_scores, y_ho),
            "paired_diff_hidden_plus_surface_minus_visible": auroc(target_scores, y_ho) - auroc(s_v, y_ho),
            "bootstrap": boot}


def published_oof(rows: list[dict]) -> dict:
    """The published estimator: out-of-fold predictions over all rows, 5-fold task-grouped CV,
    shortcuts k_pca=6 l2=1, hidden+shortcuts k_pca=24 l2=2; task-clustered bootstrap."""
    Xh = torch.stack([r["old_feats"].flatten().float() for r in rows])
    Xs = torch.stack([old_sc_vec("old_cut")(r).float() for r in rows])
    y = torch.tensor([1 if r["success224"] else 0 for r in rows]).long()
    groups = [r["task_id"] for r in rows]
    s_sc = cv_oof_scores(Xs, y, groups, k_pca=6, l2=1.0)
    s_comb = cv_oof_combined(Xh, Xs, y, groups, k_pca=24, l2=2.0)
    return {"n": len(rows), "n_tasks": len(set(groups)), "n_negatives": int((y == 0).sum()),
            "auroc_shortcut_only": auroc(s_sc, y), "auroc_hidden_plus_shortcuts": auroc(s_comb, y),
            "incremental_auroc_combined_minus_shortcut": auroc(s_comb, y) - auroc(s_sc, y),
            "bootstrap": bootstrap_task_clustered(s_comb, s_sc, y, groups, rounds=2000)}


def diff_of_increments(a: dict, b: dict, y, tasks, rounds: int = 2000) -> dict:
    """Task-clustered bootstrap of [AUROC(comb_a) - AUROC(sc_a)] - [AUROC(comb_b) - AUROC(sc_b)]."""
    def stat(idx):
        yy = y[idx]
        return (auroc(a["combined"][idx], yy) - auroc(a["shortcut"][idx], yy)
                - auroc(b["combined"][idx], yy) + auroc(b["shortcut"][idx], yy))
    by = {}
    for i, t in enumerate(tasks):
        by.setdefault(t, []).append(i)
    uniq = sorted(by)
    rng = torch.Generator().manual_seed(SEED)
    vals = []
    for _ in range(rounds):
        idx = torch.tensor([i for j in torch.randint(0, len(uniq), (len(uniq),), generator=rng).tolist() for i in by[uniq[j]]])
        if y[idx].float().mean() in (0.0, 1.0):
            continue
        s = stat(idx)
        if not math.isnan(s):
            vals.append(s)
    vals.sort()
    n = len(vals)
    return {"estimate": stat(torch.arange(len(tasks))), "ci95": [vals[int(0.025 * n)], vals[int(0.975 * n)]],
            "rounds_used": n}


def _blocks(rows, fns):
    return [np.stack([f(r).flatten().float().numpy() for r in rows]) for f in fns]


def _pca(X, k=24):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    vt = np.linalg.svd((X - mu) / sd, full_matrices=False)[2][:k]
    return lambda Z, kk: ((Z - mu) / sd) @ vt[:kk].T


def _design(tr, te, blocks, S, ks):
    """PCA per hidden block (fit on tr), then S; everything standardized with tr statistics."""
    parts_tr, parts_te = [], []
    for B, k in zip(blocks, ks):
        f = _pca(B[tr])
        parts_tr.append(f(B[tr], k)); parts_te.append(f(B[te], k))
    parts_tr.append(S[tr]); parts_te.append(S[te])
    A, Bm = np.concatenate(parts_tr, 1), np.concatenate(parts_te, 1)
    mu, sd = A.mean(0), A.std(0) + 1e-6
    return (A - mu) / sd, (Bm - mu) / sd


def _fit_score(blocks, S, y, g, tv, ho):
    """Hyperparameters (k per block in {16, 24}, C in {0.01, 0.1, 1}) by 5-fold task-grouped CV on
    train+val; refit on train+val; score held-out once."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    grid = [(ks, C) for ks in itertools.product((16, 24), repeat=len(blocks)) for C in (0.01, 0.1, 1.0)]
    itv = np.where(tv)[0]
    oof = {cfg: np.zeros(len(itv)) for cfg in grid}
    for a_, b_ in GroupKFold(5).split(itv, y[itv], g[itv]):
        tr, te = itv[a_], itv[b_]
        for ks in {cfg[0] for cfg in grid}:
            Xtr, Xte = _design(tr, te, blocks, S, ks)
            for C in (0.01, 0.1, 1.0):
                m = LogisticRegression(C=C, max_iter=5000).fit(Xtr, y[tr])
                oof[(ks, C)][b_] = m.predict_proba(Xte)[:, 1]
    best = max(grid, key=lambda cfg: roc_auc_score(y[itv], oof[cfg]))
    Xtr, Xte = _design(itv, np.where(ho)[0], blocks, S, best[0])
    m = LogisticRegression(C=best[1], max_iter=5000).fit(Xtr, y[itv])
    return m.predict_proba(Xte)[:, 1], {"k": list(best[0]), "C": best[1], "cv_auroc": roc_auc_score(y[itv], oof[best])}


def proper_arm(name, rows, add, sc, base=(), label="success"):
    """Increase from adding hidden blocks `add` to a baseline of blocks `base` plus surface features S,
    with a combined model that can use S (PCA per hidden block, then S). Skipped below the minimums."""
    c = counts(rows, label)
    if not adjudicable(c):
        return {"skipped": "below the adjudicability minimums", **c}, None
    y = np.array([1 if r[label] else 0 for r in rows]); g = np.array([r["task_id"] for r in rows])
    tv = np.array([r["split"] in ("train", "val") for r in rows]); ho = np.array([r["split"] == "heldout" for r in rows])
    S = np.stack([sc(r).float().numpy() for r in rows])
    Bb, Ba = _blocks(rows, base), _blocks(rows, add)
    s_base, h_base = _fit_score(Bb, S, y, g, tv, ho)
    s_full, h_full = _fit_score(Bb + Ba, S, y, g, tv, ho)
    s_hid, h_hid = _fit_score(Ba, np.zeros((len(rows), 0)), y, g, tv, ho)
    to = lambda v: torch.tensor(v, dtype=torch.double)
    yh, th = torch.tensor(y[ho]).double(), list(g[ho])
    res = {**c, "n_heldout": int(ho.sum()), "n_trainval": int(tv.sum()),
           "hyperparams": {"baseline": h_base, "combined": h_full, "hidden": h_hid},
           "auroc_shortcut_only": auroc(to(s_base), yh), "auroc_hidden_only": auroc(to(s_hid), yh),
           "auroc_hidden_plus_shortcuts": auroc(to(s_full), yh),
           "paired_task_clustered_bootstrap_combined_minus_shortcut": bootstrap_task_clustered(to(s_full), to(s_base), yh, th, rounds=2000)}
    res["incremental_auroc_combined_minus_shortcut"] = res["auroc_hidden_plus_shortcuts"] - res["auroc_shortcut_only"]
    print(f"[{name}] baseline={res['auroc_shortcut_only']:.4f} hidden={res['auroc_hidden_only']:.4f} "
          f"combined={res['auroc_hidden_plus_shortcuts']:.4f} increase={res['incremental_auroc_combined_minus_shortcut']:+.4f}", flush=True)
    return res, {"combined": to(s_full), "shortcut": to(s_base), "hidden": to(s_hid), "y": yh, "tasks": th}


def within_task_auroc(s, y, tasks, rounds: int = 2000) -> dict:
    """Share of (correct, incorrect) pairs from the same question ranked correctly (ties 1/2): question
    difficulty is constant within a pair. Task-clustered bootstrap over questions with mixed outcomes."""
    s, y = np.asarray(s, float), np.asarray(y).astype(int)
    by = {}
    for i, t in enumerate(tasks):
        by.setdefault(t, []).append(i)
    mixed = sorted(t for t, ix in by.items() if 0 < y[ix].mean() < 1)
    def stat(ts):
        num = den = 0.0
        for t in ts:
            ix = by[t]
            for p in (i for i in ix if y[i] == 1):
                for n in (i for i in ix if y[i] == 0):
                    num += 1.0 if s[p] > s[n] else 0.5 if s[p] == s[n] else 0.0
                    den += 1
        return num / den if den else float("nan")
    if not mixed:
        return {"n_mixed_tasks": 0}
    rng = np.random.default_rng(SEED)
    vals = sorted(stat(rng.choice(mixed, len(mixed))) for _ in range(rounds))
    return {"estimate": stat(mixed), "ci95": [vals[int(0.025 * rounds)], vals[int(0.975 * rounds)]],
            "n_mixed_tasks": len(mixed), "n_pairs": int(sum(sum(y[by[t]]) * (len(by[t]) - sum(y[by[t]])) for t in mixed))}


def verdict(res: dict) -> str:
    if not adjudicable(res):
        return "GSM8K_FIXED_PREFIX_NOT_ADJUDICABLE"
    lo, hi = res["paired_task_clustered_bootstrap_combined_minus_shortcut"]["ci95"]
    if lo > 0:
        return "GSM8K_FIXED_PREFIX_INCREMENT_POSITIVE"
    if hi < 0:
        return "GSM8K_FIXED_PREFIX_INCREMENT_NEGATIVE"
    return "GSM8K_FIXED_PREFIX_INCREMENT_UNRESOLVED"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["audit", "full"], required=True)
    ap.add_argument("--data", type=Path, default=HERE)
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--synthetic", action="store_true", help="dry run on synthetic shards: skip the seal check")
    a = ap.parse_args()
    recs, errors = load(a.data)
    out = {"integrity": integrity(recs, errors, a.synthetic)}
    answer_free_table(recs)
    n_star, frac_train = choose_n_star(recs)
    out.update({"n_star": n_star, "answer_free_fraction_train": frac_train,
                "answer_free_fraction_by_split": fractions_by_split(recs),
                "parseable_fraction": sum(r["parseable"] for r in recs) / len(recs),
                "hit_max_tokens_fraction": sum(r["hit_max_tokens"] for r in recs) / len(recs)})
    hand_audit(recs, n_star, a.out)
    if a.stage == "audit":
        (a.out / "audit_stage.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
        return

    parse = [r for r in recs if r["parseable"]]
    out["accuracy"] = {"primary_parseable": sum(r["success"] for r in parse) / len(parse),
                       "published_protocol_224": sum(r["success224"] for r in recs) / len(recs)}
    agree = Counter((bool(r["success224"]), bool(r["success"]) if r["parseable"] else None) for r in recs)
    out["published_vs_primary_labels"] = {f"published={k[0]}|primary={k[1]}": v for k, v in sorted(agree.items(), key=str)}
    primary = [r for r in parse if r["free"][n_star]]
    gi = GRID.index(n_star)
    s_diff = lambda r: torch.tensor([float(r["prompt_tok"]), float(r["ref_steps"])])
    s_primary = lambda r: torch.cat([prefix_lp_vec(n_star)(r), s_diff(r)])
    pend = lambda r: r["prompt_end_state"]
    # primary (amendment 1): hidden states at N* beyond token confidence and question difficulty
    res, sc = proper_arm("primary", primary, [feat_at(n_star)], s_primary)
    res["verdict"] = verdict(res)
    if sc is not None:
        y_ho, t_ho = sc["y"], sc["tasks"]
        perm = torch.randperm(y_ho.shape[0], generator=torch.Generator().manual_seed(SEED))
        res["shuffled_label_auroc_combined"] = auroc(sc["combined"], y_ho[perm])
        inc, worst = res["incremental_auroc_combined_minus_shortcut"], 0.0
        for t in sorted(set(t_ho)):
            keep = torch.tensor([x != t for x in t_ho])
            if y_ho[keep].float().mean() in (0.0, 1.0):
                continue
            worst = max(worst, abs(auroc(sc["combined"][keep], y_ho[keep]) - auroc(sc["shortcut"][keep], y_ho[keep]) - inc))
        res["loto_max_abs_influence"] = worst
        out["control_within_question"] = {k: within_task_auroc(sc[k].numpy(), y_ho.numpy(), t_ho)
                                          for k in ("combined", "hidden", "shortcut")}
        out["secondary_visible_reader"] = visible_reader(primary, n_star, sc["combined"], y_ho, t_ho, s_primary)
    out["primary"] = res
    # prespecified control: does the increase survive the model's own difficulty estimate before reasoning?
    out["control_prompt_end"] = proper_arm("prompt_end_control", primary, [feat_at(n_star)], s_primary, base=[pend])[0]
    out["secondary_prompt_end_only"] = proper_arm("prompt_end_only", parse, [pend], s_diff)[0]
    # the sealed Run 1-style analysis, for comparison with Horizon
    out["secondary_run1_style"] = guarded("run1_style", primary, feat_at(n_star), prefix_lp_vec(n_star))[0]

    tasks_free = {}
    for r in recs:
        tasks_free.setdefault(r["task_id"], []).append(r["free"][n_star])
    task_level = [r for r in parse if all(tasks_free[r["task_id"]])]
    out["secondary_task_level"] = proper_arm("task_level", task_level, [feat_at(n_star)], s_primary)[0]
    out["secondary_grid"] = {}
    for n in GRID[1:]:
        rows = [r for r in parse if r["free"][n]]
        out["secondary_grid"][str(n)] = proper_arm(f"grid_{n}", rows, [feat_at(n)],
                                                   lambda r, n=n: torch.cat([prefix_lp_vec(n)(r), s_diff(r)]))[0]
    out["secondary_extended_surface"] = proper_arm("extended_surface", primary, [feat_at(n_star)],
                                                   lambda r: torch.cat([prefix_ext_vec(n_star)(r), s_diff(r)]))[0]
    no_gold = [r for r in primary if not is_answer_bearing(r["prefix_texts"][gi], r["gold"])]
    out["secondary_excluding_reference_number"] = proper_arm(
        "excluding_reference_number", no_gold, [feat_at(n_star)], s_primary)[0]

    # published protocol on fresh problems, and the same cut with the candidate's own answer
    both = [r for r in recs if r["old_cut"] > 0 and r["own_cut"] > 0]
    old, s_old = guarded("published_cut", both, lambda r: r["old_feats"], old_sc_vec("old_cut"), "success224")
    own, s_own = guarded("own_answer_cut", both, lambda r: r["own_feats"], old_sc_vec("own_cut"), "success224")
    out["secondary_published_cut"] = old
    out["secondary_own_answer_cut"] = own
    if s_old is not None and s_own is not None:
        ho = [r for r in both if r["split"] == "heldout"]
        out["secondary_published_minus_own_cut"] = diff_of_increments(
            s_old, s_own, torch.tensor([1.0 if r["success224"] else 0.0 for r in ho]), [r["task_id"] for r in ho])
    # the published cut with difficulty in the baseline and the fixed combined model (amendment 1)
    for key, cut in (("old", "old_cut"), ("own", "own_cut")):
        out[f"secondary_{key}_cut_with_difficulty"] = proper_arm(
            f"{key}_cut_difficulty", both, [lambda r, k=key: r[f"{k}_feats"]],
            lambda r, c=cut: torch.cat([old_sc_vec(c)(r), torch.tensor([float(r["ref_steps"])])]), label="success224")[0]
    out["secondary_published_cut_oof"] = published_oof([r for r in recs if r["old_cut"] > 0])
    reasons = Counter((r["old_reason"], "correct" if r["success224"] else "incorrect") for r in recs)
    out["published_cut_reason_by_label"] = {f"{k[0]}|{k[1]}": v for k, v in sorted(reasons.items())}
    out["rows_dropped_from_cut_comparison"] = sum(1 for r in recs if r["old_cut"] > 0) - len(both)
    pe = out["control_prompt_end"]
    if "paired_task_clustered_bootstrap_combined_minus_shortcut" in pe:
        survives = pe["paired_task_clustered_bootstrap_combined_minus_shortcut"]["ci95"][0] > 0
        out["headline_rule"] = ("READABILITY_BEYOND_DIFFICULTY" if survives and res["verdict"].endswith("POSITIVE")
                                else "MOSTLY_DIFFICULTY")
    (a.out / "results.json").write_text(json.dumps(out, indent=2, default=float))

    def fmt(r):
        if "incremental_auroc_combined_minus_shortcut" not in r:
            return (f"not run, below the minimums ({r['n_heldout_tasks']} held-out tasks, "
                    f"{r['n_heldout_negatives']} incorrect)")
        b = r["paired_task_clustered_bootstrap_combined_minus_shortcut"]
        return (f"surface {r['auroc_shortcut_only']:.3f}, hidden {r['auroc_hidden_only']:.3f}, both "
                f"{r['auroc_hidden_plus_shortcuts']:.3f}, increase {r['incremental_auroc_combined_minus_shortcut']:+.3f} "
                f"[{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}]; held-out {r['n_heldout']} answers, "
                f"{r['n_heldout_tasks']} tasks, {r['n_heldout_negatives']} incorrect")
    def wq(d):
        return (f"{d['estimate']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}] over {d['n_pairs']} pairs in "
                f"{d['n_mixed_tasks']} questions" if d.get("n_mixed_tasks") else "no questions with mixed outcomes")
    lines = ["# Run 3 results (GSM8K)", "", f"N* = {n_star}. Verdict: **{res['verdict']}**. "
             f"Headline rule: **{out.get('headline_rule', 'n/a')}**", "",
             f"- Primary (N*={n_star}; S = log-prob + prompt tokens + reference steps; fixed combined model): {fmt(res)}",
             f"- Control, end-of-prompt state in the baseline: {fmt(out['control_prompt_end'])}"]
    if "control_within_question" in out:
        cw, vr = out["control_within_question"], out["secondary_visible_reader"]
        lines += [f"- Control, within-question AUROC: combined {wq(cw['combined'])}; hidden {wq(cw['hidden'])}; "
                  f"surface {wq(cw['shortcut'])}",
                  f"- Shuffled labels {res['shuffled_label_auroc_combined']:.3f}; max leave-one-task-out influence "
                  f"{res['loto_max_abs_influence']:.3f}",
                  f"- Visible reader + S at N*: AUROC {vr['auroc_visible_reader']:.3f} (C={vr['C']}); combined minus "
                  f"reader {vr['paired_diff_hidden_plus_surface_minus_visible']:+.3f} "
                  f"[{vr['bootstrap']['ci95'][0]:+.3f}, {vr['bootstrap']['ci95'][1]:+.3f}]"]
    po = out["secondary_published_cut_oof"]
    lines += [f"- End-of-prompt state alone vs prompt tokens + reference steps: {fmt(out['secondary_prompt_end_only'])}",
              f"- Run 1-style (S = log-prob only, sealed arm()): {fmt(out['secondary_run1_style'])}",
              f"- Task level at N*: {fmt(out['secondary_task_level'])}",
              f"- Extended surface at N*: {fmt(out['secondary_extended_surface'])}",
              f"- Also excluding prefixes with the reference number: {fmt(out['secondary_excluding_reference_number'])}",
              f"- Published cut (reference number), 224 tokens: {fmt(old)}",
              f"- Same cut with the candidate's own answer: {fmt(own)}",
              f"- Published cut, difficulty in baseline, fixed combined model: {fmt(out['secondary_old_cut_with_difficulty'])}",
              f"- Own-answer cut, difficulty in baseline, fixed combined model: {fmt(out['secondary_own_cut_with_difficulty'])}"]
    if "secondary_published_minus_own_cut" in out:
        d = out["secondary_published_minus_own_cut"]
        lines.append(f"- Published minus own-answer increase: {d['estimate']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]")
    lines += [f"- Published cut, published OOF estimator: shortcuts {po['auroc_shortcut_only']:.3f}, both "
              f"{po['auroc_hidden_plus_shortcuts']:.3f}, increase {po['incremental_auroc_combined_minus_shortcut']:+.3f} "
              f"[{po['bootstrap']['ci95'][0]:+.3f}, {po['bootstrap']['ci95'][1]:+.3f}]; {po['n']} answers",
              f"- Published cut position by label: {out['published_cut_reason_by_label']}",
              f"- Published (224-token) label vs primary label: {out['published_vs_primary_labels']}",
              f"- Accuracy: {out['accuracy']}; parseable {out['parseable_fraction']:.3f}; hit 448 tokens "
              f"{out['hit_max_tokens_fraction']:.3f}", "",
              "Grid (candidate level, S = log-prob + difficulty, fixed combined model):"] + \
        [f"- N={n}: {fmt(r)}" for n, r in out["secondary_grid"].items()]
    (a.out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
