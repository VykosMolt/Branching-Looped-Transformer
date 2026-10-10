"""C5 results writer: probe minus reader, across questions (held-out) and within questions (held-out pairs; all
pairs out of fold). Implements the evaluation in PROTOCOL.md.
  results.py --reader runs/deberta/reader_scores.json [--reader runs/qwen/...]   -> results.json, RESULTS.md
  results.py --validate   feeds published TF-IDF / probe scores through the same code; must reproduce published numbers"""
import argparse
import json
import sys
from pathlib import Path

from common import DATA, HERE, ROUNDS, cohort, lib_paths, load_rows, mixed_pairs, paired_within

sys.dont_write_bytecode = True
lib_paths()
import numpy as np  # noqa: E402
import torch  # noqa: E402
import analyze_run3 as A3  # noqa: E402
from bg_v2_overnight_horizon_analysis import bootstrap_task_clustered  # noqa: E402
from proto_introspection_controls_analysis import auroc  # noqa: E402

# probe score sets compared with the reader (names in data/probe_scores.json); first = primary
ACROSS = {"r1": ("arm_combined", "hidden"), "r1s": ("arm_combined", "hidden"), "r4": ("hidden", "combined")}
WITHIN_HO = "hidden"
WITHIN_OOF = "hidden"


def across(probe, reader, y, g):
    T = lambda a: torch.tensor(a, dtype=torch.double)
    b = bootstrap_task_clustered(T(probe), T(reader), T(y), list(g), rounds=ROUNDS)  # seed 20260724
    a_p, a_r = auroc(T(probe), T(y)), auroc(T(reader), T(y))
    return {"probe": a_p, "reader": a_r, "diff": a_p - a_r, "ci95": b["ci95"], "n_rounds_used": b["n_rounds_used"],
            "n": len(y), "n_questions": len(set(g)), "n_incorrect": int((y == 0).sum())}


def evaluate(rows, probes, reader):
    """reader: {cohort: {"heldout": {key: s}, "oof": {key: s}}}"""
    out = {}
    for c, rd in reader.items():
        rs, y, g, split, fold = cohort(rows, c)
        pos = {r["key"]: i for i, r in enumerate(rs)}
        o = out[c] = {}
        ho = np.array(sorted(pos[k] for k in rd["heldout"]))
        assert all(split[i] == "heldout" for i in ho)
        keys = [rs[i]["key"] for i in ho]
        s_r = np.array([rd["heldout"][k] for k in keys])
        yh, gh = y[ho], g[ho]
        for name in ACROSS[c]:
            s_p = np.array([probes[c]["heldout"][name][k] for k in keys])
            o[f"across_heldout__{name}"] = across(s_p, s_r, yh, gh)
        pairs = mixed_pairs(gh, yh)
        if pairs:
            s_p = np.array([probes[c]["heldout"][WITHIN_HO][k] for k in keys])
            o["within_heldout"] = paired_within(s_p, s_r, pairs)
            o["within_heldout"]["reader_alone"] = A3.within_task_auroc(s_r, yh, list(gh))
        if rd.get("oof"):
            idx = np.array(sorted(pos[k] for k in rd["oof"]))
            keys = [rs[i]["key"] for i in idx]
            s_r = np.array([rd["oof"][k] for k in keys])
            s_p = np.array([probes[c]["oof"][WITHIN_OOF][k] for k in keys])
            yo, go = y[idx], g[idx]
            o["within_oof"] = paired_within(s_p, s_r, mixed_pairs(go, yo))
            o["within_oof"]["reader_alone"] = A3.within_task_auroc(s_r, yo, list(go))
            T = lambda a: torch.tensor(a, dtype=torch.double)
            o["across_oof_all_splits"] = {"probe": auroc(T(s_p), T(yo)), "reader": auroc(T(s_r), T(yo))}
    return out


def fmt(d):
    return f"{d['diff']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"


def report(name, res, sel):
    L = [f"## {name}", ""]
    for c, o in res.items():
        L.append(f"**{c}** (selected {sel.get(c, '-')})")
        for k, d in o.items():
            if k.startswith("across_heldout"):
                L.append(f"- Across questions, held-out ({d['n']} answers, {d['n_questions']} questions, {d['n_incorrect']} incorrect), "
                         f"probe `{k.split('__')[1]}` {d['probe']:.3f} vs reader {d['reader']:.3f}: probe minus reader {fmt(d)}")
        for k, lab in (("within_heldout", "Within questions, held-out pairs"), ("within_oof", "Within questions, all pairs out of fold")):
            if k in o:
                d = o[k]
                L.append(f"- {lab} ({d['n_pairs']} pairs, {d['n_questions']} questions): probe {d['a']:.3f} vs reader {d['b']:.3f}: "
                         f"probe minus reader {fmt(d)}; reader alone {d['reader_alone']['estimate']:.3f} "
                         f"[{d['reader_alone']['ci95'][0]:.3f}, {d['reader_alone']['ci95'][1]:.3f}]")
        L.append("")
    return L


def validate(rows, probes):
    """The same code on published inputs must give the published numbers."""
    fails = []

    def check(label, got, want):
        ok = all(abs(a - b) < 1e-9 for a, b in zip(got, want))
        print(f"[validate] {label}: got {[round(x, 6) for x in got]} published {[round(x, 6) for x in want]} {'OK' if ok else 'FAIL'}")
        fails.extend([] if ok else [label])
    fake = {"r1": {"heldout": probes["r1"]["heldout"]["tfidf"]}, "r4": {"heldout": probes["r4"]["heldout"]["tfidf_plus_S"]}}
    res = evaluate(rows, probes, fake)
    d = res["r1"]["across_heldout__arm_combined"]  # Run 1 RESULTS.md: hidden+surface minus reader
    check("Run 1 probe minus TF-IDF, held-out", [d["diff"], *d["ci95"]], [0.11777172267635483, 0.032054706699433666, 0.20620614910729562])
    d = res["r4"]["across_heldout__combined"]  # Run 4 results.json secondary_visible_reader
    check("Run 4 combined minus TF-IDF+S, held-out", [d["diff"], *d["ci95"]], [0.014115936215023606, -0.023384763741562176, 0.05023694293204273])
    # Run 4 out-of-fold within question (posthoc_audit_20261004/oof_within.json), from its saved OOF scores
    oof = np.load("/home/moloch/ouro_project/opi/preanswer/horizon_cohort2_20261003/posthoc_audit_20261004/oof_scores.npy")
    rs, y, g, _, _ = cohort(rows, "r4")
    assert oof.shape == (4, len(rs))
    keys = [r["key"] for r in rs]
    pr = {c: dict(v) for c, v in probes.items()}
    pr["r4"] = {**probes["r4"], "oof": {"hidden": dict(zip(keys, oof[0]))}}
    d = evaluate(rows, pr, {"r4": {"heldout": fake["r4"]["heldout"], "oof": dict(zip(keys, oof[3]))}})["r4"]["within_oof"]
    check("Run 4 OOF within, hidden minus TF-IDF", [d["diff"], *d["ci95"], d["n_pairs"]], [-0.05063291139240511, -0.11715481171548114, 0.016597510373443924, 237])
    print("validation FAILED:" if fails else "validation passed", fails)
    return not fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", type=Path, action="append", default=[])
    ap.add_argument("--probes", type=Path, default=DATA / "probe_scores.json")
    ap.add_argument("--out", type=Path, default=HERE / "runs")
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    rows, probes = load_rows(), json.load(open(a.probes))
    if a.validate:
        sys.exit(0 if validate(rows, probes) else 1)
    allres, lines = {}, ["# C5 results: fine-tuned LM reader vs hidden-state probe (post hoc)", ""]
    for p in a.reader:
        rd = json.load(open(p))
        res = evaluate(rows, probes, rd["cohorts"])
        sel = {c: v["selected"] for c, v in rd["cohorts"].items()}
        allres[rd["reader"]] = {"model": rd["model"], "revision": rd["revision"], "smoke": rd["smoke"], "selected": sel, "results": res}
        lines += report(f"{rd['reader']} ({rd['model']}{', SMOKE: tiny random model, numbers meaningless' if rd['smoke'] else ''})", res, sel)
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(allres, open(a.out / "results.json", "w"), indent=1, default=float)
    (a.out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
