"""T1 fair-tie top-1 reanalysis (post hoc, 2026-11-08).

Recomputes candidate-ranking top-1 with fair tie handling from the saved prediction
files of commit b8c95e3, after two gates that reproduce the published numbers:

  Gate A  first-max top-1 recomputed from saved scores (original
          group_selection_metrics) == stored top1_oracle, every row of every file.
  Gate B  the original stats code (bg_xloop_early_v1_stats.main for the sealed
          48-layer runs, bg_v3_family_xloop.generic_stats otherwise), run on the
          original rows, reproduces the published bootstrap_stats*.json exactly.

Fair rule: top1_fair = #(tied at max saved score with reward >= max reward) / #(tied at
max), exact equality of the saved (6-decimal) scores. The fair rows are written as
prediction files with top1_oracle := top1_fair and pushed through the same original
stats code (same seed, same draws).

Run: /home/moloch/ouro_project/shared/venv/bin/python -I fair_ties.py
Everything from the commit is extracted to a temporary directory; nothing in the repo
working tree is read or written except this script's own directory.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import json
import math
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path("/home/moloch/ouro_project")
COMMIT = "b8c95e3"
OUT = Path(__file__).resolve().parent

RUNS = {  # run key -> (results dir in commit, published stats file, driver)
    "rltt": ("results/cross_loop_early_layer_taps_20260720", "bootstrap_stats.json", "sealed"),
    "base26": ("results/family_xloop_v3_20260726/base26", "bootstrap_stats.json", "family"),
    "thinking26": ("results/family_xloop_v3_20260726/thinking26", "bootstrap_stats.json", "family"),
    "ouro14b": ("results/family_xloop_v3_20260726/ouro14b", "bootstrap_stats_v3.json", "family"),
    "huginn": ("results/huginn_probe_v3_20260726", "bootstrap_stats_v3.json", "huginn"),
}
CODE = ["probes/bg_xloop_early_v1_common.py", "probes/bg_xloop_early_v1_stats.py",
        "probes/bg_xloop_early_v1_controls.py",
        "tools/paper1_v3/bg_v3_family_xloop.py", "tools/paper1_v3/bg_v3_huginn_probe.py"]
NONLOOPED = "results/paper_verification/nonlooped_architecture_control_20260710_235252/" \
            "corecontent_control_results_task_disjoint.json"
GAP_WINDOW = 0.01   # window for the empirical top-2 gap density near 0
GRID = 1e-6         # saved-score rounding step


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def extract(dst: Path) -> str:
    full = subprocess.run(["git", "-C", str(REPO), "rev-parse", COMMIT], check=True,
                          capture_output=True, text=True).stdout.strip()
    paths = CODE + [r[0] for r in RUNS.values()] + [NONLOOPED]
    tar = subprocess.run(["git", "-C", str(REPO), "archive", full, *paths],
                         check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
        tf.extractall(dst, filter="data")
    return full


# ------------------------------------------------------------------ per-row rules
def first_max_top1(X, scores, rewards) -> float:
    """Original rule, via the original function."""
    return X.group_selection_metrics({"candidates": [{"reward": r} for r in rewards]},
                                     scores)["top1_oracle"]


def fair_top1(scores, rewards) -> float:
    m, best = max(scores), max(rewards)
    tied = [i for i, s in enumerate(scores) if s == m]
    return sum(1 for i in tied if rewards[i] >= best) / len(tied)


def saved_pairwise(scores, rewards) -> float:
    """Pairwise accuracy recomputed on saved scores (original 0.5 tie credit)."""
    c = t = 0
    for a in range(len(scores)):
        for b in range(a + 1, len(scores)):
            if rewards[a] == rewards[b]:
                continue
            t += 1
            hi, lo = (a, b) if rewards[a] > rewards[b] else (b, a)
            c += 1 if scores[hi] > scores[lo] else (0.5 if scores[hi] == scores[lo] else 0)
    return c / t if t else float("nan")


# ------------------------------------------------------------------ module plumbing
def configure(mods, run, src):
    """Restore module defaults, apply the run's original patch, return (plan or None)."""
    X, ST, FX, HG = mods
    importlib.reload(X)
    rdir, _, driver = RUNS[run]
    plan = None
    if driver == "family":
        mp = json.loads((src / rdir / "model_plan.json").read_text())
        plan = {"n_layers": mp["n_layers"], "hidden": mp["hidden"],
                "layers": tuple(mp["layers"]),
                "capture_idx": {int(k): v for k, v in mp["capture_idx"].items()},
                "boundary": mp["boundary"], "total_ut_steps": mp["total_ut_steps"]}
        FX.patch_common(run, plan)
    elif driver == "huginn":
        HG.patch_common()
    return plan


def run_original_stats(mods, run, src, plan, pred_dir: Path, work: Path) -> dict:
    """Run the original stats entry point with RUN_ROOT/PRED_ROOT redirected."""
    X, ST, FX, HG = mods
    rdir, stats_name, driver = RUNS[run]
    work.mkdir(parents=True)
    shutil.copy2(src / rdir / "train_eval_results.json", work / "train_eval_results.json")
    X.RUN_ROOT, X.PRED_ROOT = work, pred_dir
    with contextlib.redirect_stdout(io.StringIO()):
        if driver == "sealed":
            ST.main()
        elif driver == "family":
            FX.stage_stats(run, plan)
        else:
            HG.stage_stats()
    out = json.loads((work / stats_name).read_text())
    series = ST.load_pred_vectors()
    points, samples = ST.bootstrap(series, X.BOOT_DRAWS, X.BOOT_SEED)
    return {"stats": out, "points": points, "samples": samples}


def paired(res, a, b) -> dict:
    d = res["samples"][a] - res["samples"][b]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"point": round(res["points"][a] - res["points"][b], 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}/{k}: missing in {'regenerated' if k in a else 'published'}"
               for k in sorted(set(a) ^ set(b))]
        for k in sorted(set(a) & set(b)):
            out += diff(a[k], b[k], f"{path}/{k}")
        return out
    return [] if a == b else [f"{path}: published={a!r} regenerated={b!r}"]


def gate_b_keys(stats: dict) -> list[str]:
    return [k for k in ("macro", "paired_deltas", "verdict_labels",
                        "best_late_loop_early_layer_cell", "points", "reference_cells",
                        "draws", "seed", "cluster") if k in stats]


# ------------------------------------------------------------------ main
def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="t1_fair_ties_"))
    try:
        return analyse(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def analyse(tmp: Path) -> int:
    src = tmp / "src"
    src.mkdir()
    full_commit = extract(src)
    results = {"commit": full_commit, "date": "2026-11-08", "post_hoc": True,
               "fair_rule": "top1_fair = #tied-at-max-saved-score with reward >= max reward "
                            "/ #tied-at-max; exact equality of 6-decimal saved scores",
               "inputs_sha256": {}, "gate_a": {}, "gate_b": {}, "runs": {},
               "not_covered": {}}
    for p in sorted(src.rglob("*")):  # hash before import, so no .pyc can enter
        if p.is_file() and "predictions" not in p.parts:
            results["inputs_sha256"][str(p.relative_to(src))] = sha(p)

    sys.dont_write_bytecode = True
    sys.path[:0] = [str(src / "probes"), str(src / "tools/paper1_v3")]
    import bg_xloop_early_v1_common as X
    import bg_xloop_early_v1_stats as ST
    import bg_v3_family_xloop as FX
    import bg_v3_huginn_probe as HG
    mods = (X, ST, FX, HG)

    for run, (rdir, stats_name, driver) in RUNS.items():
        plan = configure(mods, run, src)
        pred_src = src / rdir / "predictions"
        # files the original bootstrap reads, in its own order
        files = [f"refit_{X.cell_name(l, u)}.json" for (l, u) in X.REFIT_CELLS] + \
                [f"transfer_{l}_L{u}_to_L{v}.json" for (l, u, v) in X.TRANSFERS]
        present = sorted(p.name for p in pred_src.glob("*.json")
                         if p.name.startswith(("refit_", "transfer_")))
        assert sorted(files) == present, f"{run}: file set mismatch {set(files) ^ set(present)}"
        fair_dir = tmp / run / "fair_predictions"
        fair_dir.mkdir(parents=True)

        # ---------------- Gate A + fair rows + tie inventory
        gate_a, inv = {}, {}
        tie_pairs_by_group = defaultdict(lambda: defaultdict(int))  # gid -> frozenset -> n files
        allzero_by_group = defaultdict(int)
        partial_rows = []
        n_gap_small = n_ge3 = 0
        for fname in files:
            raw_text = (pred_src / fname).read_text()
            rows = json.loads(raw_text)
            mism = 0
            fair_rows = []
            st = {"groups": len(rows), "ties_at_max": 0, "all_equal": 0, "all_equal_nonzero": 0,
                  "partial": 0, "affected": 0, "affected_all_equal": 0, "affected_partial": 0,
                  "sum_top1_orig": 0.0, "sum_top1_fair": 0.0}
            for r in rows:
                s, rw = r["scores"], r["rewards"]
                assert all(isinstance(v, (int, float)) and math.isfinite(v) for v in s), \
                    f"{run}/{fname}/{r['group_uid']}: non-finite score"
                fm = first_max_top1(X, s, rw)
                if fm != r["top1_oracle"]:
                    mism += 1
                tf = fair_top1(s, rw)
                st["sum_top1_orig"] += r["top1_oracle"]
                st["sum_top1_fair"] += tf
                fair_rows.append({**r, "top1_oracle": tf})
                m = max(s)
                tied = [i for i, v in enumerate(s) if v == m]
                # any exactly-equal score pairs anywhere in the group (duplicate evidence)
                for a in range(len(s)):
                    for b in range(a + 1, len(s)):
                        if s[a] == s[b]:
                            tie_pairs_by_group[r["group_uid"]][(a, b)] += 1
                if len(s) >= 3:
                    n_ge3 += 1
                    srt = sorted(s, reverse=True)
                    if 0 < srt[0] - srt[1] <= GAP_WINDOW:
                        n_gap_small += 1
                if len(tied) < 2:
                    continue
                st["ties_at_max"] += 1
                affected = tf != r["top1_oracle"]
                st["affected"] += affected
                if len(tied) == len(s):
                    st["all_equal"] += 1
                    st["affected_all_equal"] += affected
                    if any(v != 0.0 for v in s):
                        st["all_equal_nonzero"] += 1
                    allzero_by_group[r["group_uid"]] += 1
                else:
                    st["partial"] += 1
                    st["affected_partial"] += affected
                    partial_rows.append({
                        "file": fname, "group_uid": r["group_uid"], "domain": r["domain"],
                        "tied_idx": tied, "scores": s, "rewards": rw,
                        "candidate_uids": r["candidate_uids"], "top1_orig": r["top1_oracle"],
                        "top1_fair": tf, "pairwise_stored": r["pairwise_accuracy"],
                        "pairwise_from_saved_scores": saved_pairwise(s, rw)})
            gate_a[fname] = {"rows": len(rows), "mismatches": mism}
            st["mean_top1_orig"] = st.pop("sum_top1_orig") / len(rows)
            st["mean_top1_fair"] = st.pop("sum_top1_fair") / len(rows)
            inv[fname] = st
            (fair_dir / fname).write_text(json.dumps(fair_rows))
        results["gate_a"][run] = gate_a
        bad = {f: g for f, g in gate_a.items() if g["mismatches"]}
        print(f"[gate A] {run}: {len(gate_a)} files, "
              f"{sum(g['rows'] for g in gate_a.values())} rows, mismatching files: {bad or 'none'}")
        if bad:
            write_failure(results, f"Gate A failed for {run}: {bad}")
            return 1
        for pr in partial_rows:
            pr["files_with_same_tied_set_equal"] = tie_pairs_by_group[pr["group_uid"]].get(
                tuple(pr["tied_idx"][:2]), 0) if len(pr["tied_idx"]) == 2 else None
        n_files = len(files)
        all_eq_groups = {g: n for g, n in sorted(allzero_by_group.items())}

        # ---------------- Gate B: original stats on original rows
        orig = run_original_stats(mods, run, src, plan, pred_src, tmp / run / "orig")
        published = json.loads((src / rdir / stats_name).read_text())
        mismatches = []
        for k in gate_b_keys(published):
            mismatches += diff(published[k], orig["stats"].get(k), f"/{k}")
        results["gate_b"][run] = {"stats_file": f"{rdir}/{stats_name}",
                                  "compared_keys": gate_b_keys(published),
                                  "n_paired_deltas": len(published["paired_deltas"]),
                                  "mismatches": mismatches}
        print(f"[gate B] {run}: compared {gate_b_keys(published)}; "
              f"{len(published['paired_deltas'])} paired deltas; mismatches: {len(mismatches)}")
        if mismatches:
            for m in mismatches:
                print("   MISMATCH", m)
            write_failure(results, f"Gate B failed for {run}")
            return 1

        # ---------------- fair rerun (configure again: stats code mutates nothing, but be explicit)
        plan = configure(mods, run, src)
        fair = run_original_stats(mods, run, src, plan, fair_dir, tmp / run / "fair")

        results["runs"][run] = summarise(run, X, orig, fair, inv, partial_rows, all_eq_groups,
                                         n_files, n_ge3, n_gap_small, tie_pairs_by_group)

    results["not_covered"] = not_covered(src)
    (OUT / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    (OUT / "RESULTS.md").write_text(render(results))
    names = [n for n in ("fair_ties.py", "results.json", "RESULTS.md", "README.md")
             if (OUT / n).exists()]
    (OUT / "SHA256SUMS").write_text("".join(f"{sha(OUT / n)}  {n}\n" for n in names))
    print(f"results.json sha256 {sha(OUT / 'results.json')}")
    return 0


def write_failure(results, msg):
    results["FAILED"] = msg
    (OUT / "gate_failure.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    print("FAILED:", msg)


def summarise(run, X, orig, fair, inv, partial_rows, all_eq_groups, n_files, n_ge3,
              n_gap_small, tie_pairs_by_group):
    po, pf = orig["stats"], fair["stats"]
    macro_o = po.get("macro") or {k: {"point": v} for k, v in po["points"].items()}
    macro_f = pf.get("macro") or {k: {"point": v} for k, v in pf["points"].items()}
    series = {}
    for name in orig["points"]:
        series[name] = {
            "orig": {"point": macro_o[name]["point"],
                     "ci95": macro_o[name].get("ci95") or ST_ci(orig["samples"][name]),
                     "point_unrounded": orig["points"][name]},
            "fair": {"point": macro_f[name]["point"],
                     "ci95": macro_f[name].get("ci95") or ST_ci(fair["samples"][name]),
                     "point_unrounded": fair["points"][name]}}
    deltas = {}
    for name, d in po["paired_deltas"].items():
        o, f = d, pf["paired_deltas"][name]
        deltas[name] = {"orig": o, "fair": f,
                        "excl0_orig": excl0(o["ci95"]), "excl0_fair": excl0(f["ci95"]),
                        "published": True}
    # loop-1 -> loop-4 (and Huginn 1 -> 8) rises not in the published file
    early = sorted({l for (l, u, v) in X.TRANSFERS})
    rises = [(l, 4, 1) for l in early] + ([(0, 8, 1)] if run == "huginn" else [])
    for (l, hi, lo) in rises:
        name = f"refit:{X.cell_name(l, hi)} - refit:{X.cell_name(l, lo)}"
        if name not in deltas:
            o = paired(orig, f"refit:{X.cell_name(l, hi)}", f"refit:{X.cell_name(l, lo)}")
            f = paired(fair, f"refit:{X.cell_name(l, hi)}", f"refit:{X.cell_name(l, lo)}")
            deltas[name] = {"orig": o, "fair": f, "excl0_orig": o["excludes_zero"],
                            "excl0_fair": f["excludes_zero"], "published": False}
    out = {"series_macro": series, "paired_deltas": deltas,
           "rise_keys": [f"refit:{X.cell_name(l, hi)} - refit:{X.cell_name(l, lo)}"
                         for (l, hi, lo) in rises],
           "transfer_keys": [f"transfer:{l}:L{u}->L{v} - refit:{X.cell_name(l, v)}"
                             for (l, u, v) in X.TRANSFERS],
           "step1_transfer_keys": [f"transfer:{l}:L{u}->L{v} - refit:{X.cell_name(l, v)}"
                                   for (l, u, v) in X.TRANSFERS if v == u + 1],
           "tie_inventory": inv,
           "partial_ties": partial_rows,
           "all_equal_groups_files": all_eq_groups,
           "n_files": n_files,
           "chance_coincidence_estimate": {
               "groups_with_3plus_candidates_over_files": n_ge3,
               "groups_with_top2_gap_in_(0,%g]" % GAP_WINDOW: n_gap_small,
               "expected_6dp_ties_at_max": n_gap_small / GAP_WINDOW * GRID / 2,
               "method": "density of top-2 gap near 0 (count/window) x grid/2: P(two values "
                         "a gap g apart round to the same 6-dp value) = max(0, 1-g/1e-6)"},
           "exact_equal_pairs_anywhere": {
               g: {f"{a}-{b}": n for (a, b), n in sorted(p.items())}
               for g, p in sorted(tie_pairs_by_group.items())}}
    if "verdict_labels" in po:
        changed = {}
        for k, lab in po["verdict_labels"].items():
            for kk, v in lab.items():
                if pf["verdict_labels"][k].get(kk) != v:
                    changed[f"{k}/{kk}"] = {"orig": v, "fair": pf["verdict_labels"][k].get(kk)}
        out["verdict_label_changes"] = changed
        out["best_late_loop_early_layer_cell"] = {
            "orig": po["best_late_loop_early_layer_cell"],
            "fair": pf["best_late_loop_early_layer_cell"]}
    return out


def ST_ci(v):
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def excl0(ci):
    return bool(ci[0] > 0 or ci[1] < 0)


def not_covered(src: Path) -> dict:
    nl = json.loads((src / NONLOOPED).read_text())
    nl_has_rows = "scores" in json.dumps(nl)
    ctrl = (src / "probes/bg_xloop_early_v1_controls.py").read_text()
    cm = {}
    for run in ("base26", "thinking26"):
        p = src / RUNS[run][0] / "crossmodel_rltt_transfer.json"
        d = json.loads(p.read_text())
        cm[run] = {"cells": sorted(d), "fields": sorted(next(iter(d.values())))}
    return {
        "nonlooped_minicpm_control": {
            "file": NONLOOPED, "has_per_group_scores": nl_has_rows,
            "status": "checked" if nl_has_rows else
            "not checkable: only per-domain/macro top-1 and pairwise aggregates are stored"},
        "crossmodel_rltt_transfer": {
            "files": cm,
            "status": "not checkable: only macro top-1 per cell is stored, no per-group rows"},
        "length_baseline": {
            "controls_py_mentions_length": "length" in ctrl.lower(),
            "status": "not recomputable: no code in the commit computes length_baseline.json "
                      "(bg_xloop_early_v1_controls.py only runs the extraction-path "
                      "controls), and candidate texts/lengths are not in the commit"},
        "reverse_transfer_and_shuffle": "rows not public; not reconstructed (per task scope)",
        "s3b2_files": "s3b2_* prediction files (secondary generated-branch readout) not in scope",
    }


# ------------------------------------------------------------------ markdown
def f4(x):
    return f"{x:.4f}"


def ci_s(c):
    return f"[{c[0]:.4f}, {c[1]:.4f}]"


def render(res) -> str:
    L = ["# T1 fair-tie top-1: original vs fair", "",
         f"Post hoc, 2026-11-08. Commit `{res['commit']}`. Rule: {res['fair_rule']}.",
         "Same original bootstrap code, seed 20260720, 10,000 task-clustered draws, identical "
         "draws for original and fair. All numbers 4 decimals (original rounding).", "",
         "## Gates", "", "| run | Gate A files / rows / mismatching rows | Gate B keys compared / "
         "paired deltas / mismatches |", "|---|---|---|"]
    for run in RUNS:
        ga, gb = res["gate_a"][run], res["gate_b"][run]
        L.append(f"| {run} | {len(ga)} / {sum(g['rows'] for g in ga.values())} / "
                 f"{sum(g['mismatches'] for g in ga.values())} | {', '.join(gb['compared_keys'])}"
                 f" / {gb['n_paired_deltas']} / {len(gb['mismatches'])} |")
    L += ["", "## Summary", "", "| run | fair-orig shift, all series (unrounded) | ties at max per "
          "file (min-max) | CI-excludes-zero changes | verdict-label changes |",
          "|---|---|---|---|---|"]
    for run, R in res["runs"].items():
        sh = [s["fair"]["point_unrounded"] - s["orig"]["point_unrounded"]
              for k, s in R["series_macro"].items() if k != "chance"]
        tm = [st["ties_at_max"] for st in R["tie_inventory"].values()]
        nch = sum(d["excl0_orig"] != d["excl0_fair"] for d in R["paired_deltas"].values())
        lab = len(R["verdict_label_changes"]) if "verdict_label_changes" in R else "n/a"
        L.append(f"| {run} | {min(sh):+.4f} to {max(sh):+.4f} | {min(tm)}-{max(tm)} | {nch} of "
                 f"{len(R['paired_deltas'])} | {lab} |")
    for run, R in res["runs"].items():
        L += ["", f"## {run}", "", "### Macro top-1 per series", "",
              "| series | orig | orig CI | fair | fair CI | fair-orig (unrounded) |",
              "|---|---|---|---|---|---|"]
        for name, s in sorted(R["series_macro"].items(), key=lambda kv: sort_key(kv[0])):
            o, f = s["orig"], s["fair"]
            L.append(f"| {name} | {f4(o['point'])} | {ci_s(o['ci95'])} | {f4(f['point'])} | "
                     f"{ci_s(f['ci95'])} | {f['point_unrounded'] - o['point_unrounded']:+.4f} |")
        L += ["", "### Loop-1 to loop-4 rise (refit)", ""] + delta_table(
            R, R["rise_keys"], mark_unpublished=True)
        L += ["", "### Transfer minus target refit (step-1 transfers marked *)", ""] + \
            delta_table(R, R["transfer_keys"], star=set(R["step1_transfer_keys"]))
        rest = [k for k in R["paired_deltas"] if k not in set(R["rise_keys"]) |
                set(R["transfer_keys"])]
        L += ["", "### All other published paired deltas", ""] + delta_table(R, rest)
        ch = [k for k in R["paired_deltas"] if R["paired_deltas"][k]["excl0_orig"]
              != R["paired_deltas"][k]["excl0_fair"]]
        L += ["", f"CI-excludes-zero conclusions changed: **{len(ch)}**"
              + (f" ({', '.join(ch)})" if ch else "") + "."]
        if "verdict_label_changes" in R:
            b = R["best_late_loop_early_layer_cell"]
            L.append(f"Verdict-label changes: **{len(R['verdict_label_changes'])}**"
                     + (f" {R['verdict_label_changes']}" if R["verdict_label_changes"] else "")
                     + f". Best late-loop early-layer cell: {b['orig']} -> {b['fair']}.")
        L += ["", "### Tie inventory", "",
              "| file | groups | ties at max | all-equal (all 0.0) | partial | changed top-1 "
              "(all-equal / partial) | mean top-1 orig -> fair |", "|---|---|---|---|---|---|---|"]
        for fn, st in R["tie_inventory"].items():
            L.append(f"| {fn} | {st['groups']} | {st['ties_at_max']} | {st['all_equal']} "
                     f"({st['all_equal'] - st['all_equal_nonzero']}) | {st['partial']} | "
                     f"{st['affected_all_equal']} / {st['affected_partial']} | "
                     f"{st['mean_top1_orig']:.4f} -> {st['mean_top1_fair']:.4f} |")
        ae = R["all_equal_groups_files"]
        L += ["", f"All-equal groups: {len(ae)} distinct groups; files in which each is all-equal "
                  f"(of {R['n_files']}): " + (", ".join(f"`{g}` {n}" for g, n in ae.items())
                                              or "none") + "."]
        ce = R["chance_coincidence_estimate"]
        L += [f"Chance estimate: {ce['groups_with_top2_gap_in_(0,0.01]']} of "
              f"{ce['groups_with_3plus_candidates_over_files']} (group x file) rows with >=3 "
              f"candidates have a top-2 gap in (0, 0.01]; expected 6-decimal coincidental ties "
              f"at the max over all files = {ce['expected_6dp_ties_at_max']:.2e}."]
        if R["partial_ties"]:
            L += ["", "Partial ties at the max:", "",
                  "| file | group_uid | tied idx | scores | rewards | candidate_uids | top-1 "
                  "orig/fair | same pair equal in n files | pairwise stored / from saved |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for p in R["partial_ties"]:
                L.append(f"| {p['file']} | `{p['group_uid']}` | {p['tied_idx']} | {p['scores']} "
                         f"| {p['rewards']} | {', '.join(u.split('::')[-1] for u in p['candidate_uids'])}"
                         f" | {p['top1_orig']:.0f}/{p['top1_fair']:.3f} | "
                         f"{p['files_with_same_tied_set_equal']} | "
                         f"{p['pairwise_stored']:.4f} / {p['pairwise_from_saved_scores']:.4f} |")
        else:
            L += ["", "Partial ties at the max: none."]
    L += ["", "## Not covered", ""]
    for k, v in res["not_covered"].items():
        L.append(f"- **{k}**: {v['status'] if isinstance(v, dict) else v}")
    return "\n".join(L) + "\n"


def sort_key(name):
    import re
    nums = [int(x) for x in re.findall(r"\d+", name)]
    return (name.split(":")[0], nums)


def delta_table(R, keys, star=(), mark_unpublished=False):
    L = ["| delta | orig | orig CI | excl 0 | fair | fair CI | excl 0 |",
         "|---|---|---|---|---|---|---|"]
    for k in keys:
        d = R["paired_deltas"][k]
        tag = ("*" if k in star else "") + (" (not in published file)" if
                                            mark_unpublished and not d["published"] else "")
        L.append(f"| {k}{tag} | {d['orig']['point']:+.4f} | {ci_s(d['orig']['ci95'])} | "
                 f"{'yes' if d['excl0_orig'] else 'no'} | {d['fair']['point']:+.4f} | "
                 f"{ci_s(d['fair']['ci95'])} | {'yes' if d['excl0_fair'] else 'no'} |")
    return L


if __name__ == "__main__":
    raise SystemExit(main())
