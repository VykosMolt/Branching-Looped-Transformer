"""Run 1 analysis, implementing PLAN.md as sealed. Reads the run1 shards, writes results.json,
RESULTS.md and hand_audit.md into this directory. Stage 'audit' runs everything label-free
(integrity checks, answer-free fractions, N*, the hand-audit list) and stops; stage 'full'
then opens labels and runs the endpoints.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (ROOT, ROOT / "shared", ROOT / "shared/utilities/tests/manual", HERE):
    sys.path.insert(0, str(p))

from answer_detector import is_answer_bearing, task_statement  # noqa: E402
from bg_steering_suite_lib import mcq_prompt  # noqa: E402
from bg_v2_overnight_horizon_generate import load_task_pool, split_for  # noqa: E402
from bg_v2_overnight_horizon_analysis import SEED, shortcut_vec, bootstrap_task_clustered  # noqa: E402
from bg_v2_malformed_sibling_control import arm  # noqa: E402
from proto_introspection_controls_analysis import auroc  # noqa: E402

GRID = (0, 16, 32, 48, 64, 96, 128)
OFFSET, N_TASKS = 2400, 600
COVERAGE, MIN_TASKS, MIN_NEG = 0.75, 30, 30


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> list[dict]:
    recs = []
    for f in sorted(glob.glob(str(HERE / "run1_shard_off*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        assert max(d["equivalence_checks_maxabs"] or [0.0]) <= 1e-3, f
        recs.extend(d["records"])
    return recs


def integrity(recs: list[dict]) -> dict:
    pool = sorted(load_task_pool(), key=lambda d: hashlib.sha256(d["task_uid"].encode()).hexdigest())
    expected = {t["task_uid"] for t in pool[OFFSET: OFFSET + N_TASKS]}
    used_before = {t["task_uid"] for t in pool[:OFFSET]}
    got = {r["task_uid"] for r in recs}
    seal = dict(line.split()[::-1] for line in (HERE / "SEAL.sha256").read_text().splitlines())
    assert seal["answer_detector.py"] == sha(HERE / "answer_detector.py"), "detector changed after sealing"
    assert got <= expected and not (got & used_before)
    assert all(r["split"] == split_for(r["task_uid"]) for r in recs)
    assert all(r["decode_roundtrip_ok"] for r in recs)
    return {"n_records": len(recs), "n_tasks": len(got), "tasks_missing": len(expected - got),
            "slice_disjoint_from_prior_runs": True, "splits_match_split_for": True, "decode_roundtrip_all": True}


def answer_free_table(recs: list[dict]) -> None:
    for r in recs:
        st = task_statement(r["task_prompt"])
        r["free"] = {n: (n == 0 or not is_answer_bearing(r["prefix_texts"][i], st)) for i, n in enumerate(GRID)}


def choose_n_star(recs: list[dict]) -> tuple[int, dict]:
    train = [r for r in recs if r["split"] == "train"]
    frac = {n: sum(r["free"][n] for r in train) / len(train) for n in GRID if n > 0}
    ok = [n for n, f in frac.items() if f >= COVERAGE]
    return (max(ok) if ok else 16), frac


def fractions_by_split(recs: list[dict]) -> dict:
    out = {}
    for split in ("train", "val", "heldout"):
        rs = [r for r in recs if r["split"] == split]
        out[split] = {str(n): round(sum(r["free"][n] for r in rs) / len(rs), 4) for n in GRID if n > 0}
    return out


def hand_audit(recs: list[dict], n_star: int) -> None:
    free = [r for r in recs if r["split"] == "train" and r["free"][n_star]]
    free.sort(key=lambda r: hashlib.sha256(f"20260927|{r['task_uid']}|{r['candidate_idx']}".encode()).hexdigest())
    i = GRID.index(n_star)
    lines = [f"# Hand audit: 20 answer-free training prefixes at N* = {n_star} (label-free)", ""]
    for r in free[:20]:
        lines += [f"## {r['task_uid']} #{r['candidate_idx']}", f"Statement: {task_statement(r['task_prompt'])}",
                  "```", r["prefix_texts"][i], "```", ""]
    (HERE / "hand_audit.md").write_text("\n".join(lines))


def design(rows: list[dict], feat, sc):
    Xh = torch.stack([feat(r).flatten().float() for r in rows])
    Xs = torch.stack([sc(r).float() for r in rows])
    y = torch.tensor([1.0 if r["success"] else 0.0 for r in rows])
    tasks = [r["task_uid"] for r in rows]
    tv = torch.tensor([r["split"] in ("train", "val") for r in rows])
    ho = torch.tensor([r["split"] == "heldout" for r in rows])
    return Xh, Xs, y, tasks, tv, ho


def run_arm(name: str, rows: list[dict], feat, sc):
    Xh, Xs, y, tasks, tv, ho = design(rows, feat, sc)
    t_tv = [t for t, m in zip(tasks, tv.tolist()) if m]
    t_ho = [t for t, m in zip(tasks, ho.tolist()) if m]
    res = arm(name, Xh[tv], Xs[tv], y[tv], t_tv, Xh[ho], Xs[ho], y[ho], t_ho)
    scores = res.pop("_scores")
    res.update({"n_heldout": int(ho.sum()), "n_heldout_tasks": len(set(t_ho)),
                "n_heldout_negatives": int((y[ho] == 0).sum()), "n_trainval": int(tv.sum())})
    return res, scores, y[ho], t_ho


def prefix_lp_vec(n: int):
    def f(r):
        lp = r["gen_logprobs"][:n] or [0.0]
        return torch.tensor([sum(lp) / len(lp), min(lp)])
    return f


def feat_at(n: int):
    i = GRID.index(n)
    return lambda r: r["prefix_features"][i]


def visible_reader(rows: list[dict], n: int, target_scores, y_ho, t_ho) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import FeatureUnion
    i = GRID.index(n)
    text = lambda r: (f"Prompt:\n{mcq_prompt(r['task_prompt'], r['options'])}\n\n"
                      f"Visible reasoning prefix:\n{r['prefix_texts'][i]}")
    tv = [r for r in rows if r["split"] in ("train", "val")]
    ho = [r for r in rows if r["split"] == "heldout"]
    y_tv = [int(r["success"]) for r in tv]
    uniq = sorted({r["task_uid"] for r in tv})
    g = torch.Generator().manual_seed(SEED)
    order = [uniq[j] for j in torch.randperm(len(uniq), generator=g).tolist()]
    fold = {u: k % 5 for k, u in enumerate(order)}
    folds = [fold[r["task_uid"]] for r in tv]

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
            m.fit(v.transform([text(tv[j]) for j in tr]), [y_tv[j] for j in tr])
            p = m.predict_proba(v.transform([text(tv[j]) for j in va]))[:, 1]
            for j, pj in zip(va, p):
                oof[j] = float(pj)
        a = roc_auc_score(y_tv, oof)
        if best is None or a > best[0]:
            best = (a, C)
    v = vec().fit([text(r) for r in tv])
    m = LogisticRegression(C=best[1], solver="liblinear", max_iter=2000, random_state=20260729)
    m.fit(v.transform([text(r) for r in tv]), y_tv)
    s_v = torch.tensor(m.predict_proba(v.transform([text(r) for r in ho]))[:, 1], dtype=torch.double)
    boot = bootstrap_task_clustered(target_scores, s_v, y_ho, t_ho, rounds=2000)
    return {"cv_auroc": best[0], "C": best[1], "auroc_visible_reader": auroc(s_v, y_ho),
            "auroc_hidden_plus_surface": auroc(target_scores, y_ho),
            "paired_diff_hidden_plus_surface_minus_visible": auroc(target_scores, y_ho) - auroc(s_v, y_ho),
            "bootstrap": boot}


def verdict(res: dict) -> str:
    if res["n_heldout_tasks"] < MIN_TASKS or res["n_heldout_negatives"] < MIN_NEG:
        return "FIXED_PREFIX_NOT_ADJUDICABLE"
    lo, hi = res["paired_task_clustered_bootstrap_combined_minus_shortcut"]["ci95"]
    if lo > 0:
        return "FIXED_PREFIX_INCREMENT_POSITIVE"
    if hi < 0:
        return "FIXED_PREFIX_INCREMENT_NEGATIVE"
    return "FIXED_PREFIX_INCREMENT_UNRESOLVED"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["audit", "full"], required=True)
    stage = ap.parse_args().stage
    recs = load()
    out = {"integrity": integrity(recs)}
    answer_free_table(recs)
    n_star, frac_train = choose_n_star(recs)
    out.update({"n_star": n_star, "answer_free_fraction_train": frac_train,
                "answer_free_fraction_by_split": fractions_by_split(recs)})
    hand_audit(recs, n_star)
    if stage == "audit":
        (HERE / "audit_stage.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
        return

    parse = [r for r in recs if not r["malformed"]]
    primary = [r for r in parse if r["free"][n_star]]
    res, sc, y_ho, t_ho = run_arm("primary", primary, feat_at(n_star), prefix_lp_vec(n_star))
    res["verdict"] = verdict(res)
    # controls on the primary arm
    perm = torch.randperm(y_ho.shape[0], generator=torch.Generator().manual_seed(SEED))
    res["shuffled_label_auroc_combined"] = auroc(sc["combined"], y_ho[perm])
    inc, worst = res["incremental_auroc_combined_minus_shortcut"], 0.0
    for t in sorted(set(t_ho)):
        keep = torch.tensor([x != t for x in t_ho])
        if y_ho[keep].float().mean() in (0.0, 1.0):
            continue
        worst = max(worst, abs(auroc(sc["combined"][keep], y_ho[keep]) - auroc(sc["shortcut"][keep], y_ho[keep]) - inc))
    res["loto_max_abs_influence"] = worst
    out["primary"] = res
    out["secondary_visible_reader"] = visible_reader(primary, n_star, sc["combined"], y_ho, t_ho)

    tasks_free = {}
    for r in recs:
        tasks_free.setdefault(r["task_uid"], []).append(r["free"][n_star])
    task_level = [r for r in parse if all(tasks_free[r["task_uid"]])]
    out["secondary_task_level"] = run_arm("task_level", task_level, feat_at(n_star), prefix_lp_vec(n_star))[0]
    out["secondary_grid"] = {}
    for n in GRID[1:]:
        rows = [r for r in parse if r["free"][n]]
        out["secondary_grid"][str(n)] = run_arm(f"grid_{n}", rows, feat_at(n), prefix_lp_vec(n))[0]
    out["secondary_prompt_only"] = run_arm(
        "prompt_only", parse, feat_at(0), lambda r: torch.tensor([float(r["prompt_tok"]), float(r["proof_depth"])]))[0]
    out["secondary_final_marker_published"] = run_arm(
        "final_marker", parse, lambda r: r["preanswer_features"], shortcut_vec)[0]
    (HERE / "results.json").write_text(json.dumps(out, indent=2, default=float))

    def fmt(r):
        b = r["paired_task_clustered_bootstrap_combined_minus_shortcut"]
        return (f"surface {r['auroc_shortcut_only']:.3f}, hidden {r['auroc_hidden_only']:.3f}, both "
                f"{r['auroc_hidden_plus_shortcuts']:.3f}, increase {r['incremental_auroc_combined_minus_shortcut']:+.3f} "
                f"[{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}]; held-out {r['n_heldout']} answers, "
                f"{r['n_heldout_tasks']} tasks, {r['n_heldout_negatives']} incorrect")
    vr = out["secondary_visible_reader"]
    lines = ["# Run 1 results", "", f"N* = {n_star}. Verdict: **{res['verdict']}**", "",
             f"- Primary (candidate level, N*={n_star}): {fmt(res)}",
             f"- Shuffled labels {res['shuffled_label_auroc_combined']:.3f}; max leave-one-task-out influence {worst:.3f}",
             f"- Visible reader at N*: AUROC {vr['auroc_visible_reader']:.3f} (C={vr['C']}); hidden+surface minus reader "
             f"{vr['paired_diff_hidden_plus_surface_minus_visible']:+.3f} "
             f"[{vr['bootstrap']['ci95'][0]:+.3f}, {vr['bootstrap']['ci95'][1]:+.3f}]",
             f"- Task level at N*: {fmt(out['secondary_task_level'])}",
             f"- Prompt only: {fmt(out['secondary_prompt_only'])}",
             f"- Final-marker cut, published protocol: {fmt(out['secondary_final_marker_published'])}", "",
             "Grid (candidate level):"] + [f"- N={n}: {fmt(r)}" for n, r in out["secondary_grid"].items()]
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
