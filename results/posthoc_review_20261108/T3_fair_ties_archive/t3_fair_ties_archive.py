"""T3 fair-tie top-1 on the archived (non-public) series. Post hoc, 2026-11-08.

Same fair rule as T1 (imported from T1's fair_ties.py):
    top1_fair = #(tied at max saved score with reward >= max reward) / #(tied at max)
Each task first reproduces the published numbers with the ORIGINAL first-max rule (gate); a
failed gate stops that task and records the mismatch.

  1 C3 reverse transfer   saved xloop_taps.pt weights -> backward per-group rows; gate vs saved
                          refit/transfer files, train_eval_results.json, bootstrap_stats.json and
                          C3's reverse_transfer_results.json (every field of all 36 transfers)
  2 shuffle controls      refit with the original train_eval shuffle path (seed 20260720);
                          gate vs train_eval_results.json and the printed lines in train_eval.log
  3 length baseline       candidate_text length from subset_groups.jsonl; gate vs length_baseline.json
  4 cross-checkpoint      original bg_v3_family_xloop.stage_crossmodel (write redirected to memory);
                          gate vs crossmodel_rltt_transfer.json
  5 MiniCPM / Ouro 0.631  original evaluate functions with saved weights and stored features;
                          gate vs the stored unrounded per-domain and macro values

Reads the archive read-only; code comes from git commit b8c95e3 via `git archive` into a
temporary directory. Writes only results.json, RESULTS.md, SHA256SUMS next to this file.

Run (see README.md):
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" \
  PYTHONDONTWRITEBYTECODE=1 /home/moloch/ouro_project/shared/venv/bin/python -I -u t3_fair_ties_archive.py
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.dont_write_bytecode = True
torch.set_num_threads(4)  # C3: 1-4 threads reproduce the saved refits bitwise for code_reasoning cells

REPO = Path("/home/moloch/ouro_project")
COMMIT = "b8c95e3"
OUT = Path(__file__).resolve().parent
T1 = REPO / "opi/posthoc_review_20261108/T1_fair_ties/fair_ties.py"
A1 = Path("/run/media/moloch/ARCH_BACKUP/OURO_COLD_ARCHIVE_20260729")
A2 = Path("/run/media/moloch/ARCH_BACKUP/OURO_COLD_ARCHIVE_20260927/tree")
RLTT = A1 / "historical_experiments/paper_v3_audit_evidence/artifacts/reports/cross_loop_early_layer_taps_20260720"
C3 = A2 / "opi/posthoc_astra_20261010/C3_reverse_transfer"
FAM = A2 / "opi/localization/family_xloop_v3_20260726"
MINI = A1 / ("historical_experiments/paper_v3_audit_evidence/artifacts/reports/paper_verification/"
             "nonlooped_architecture_control_20260710_235252")
OURO_TD = A2 / "opi/verification/paper_verification/powered_clean_probe_and_corecontent_20260711"
V2_DATA = A2 / "shared/data/corecontent_v2"

CODE = ["probes/bg_xloop_early_v1_common.py", "probes/bg_xloop_early_v1_stats.py",
        "probes/bg_xloop_early_v1_train_eval.py", "probes/bg_xloop_early_v1_controls.py",
        "tools/paper1_v3/bg_v3_family_xloop.py"]
# task 5 needs the original tree layout: <proj>/shared/{utilities/tests/manual,tools/...}, <proj>/src/evaluator
PROJ_TREES = [("shared/", ["utilities/tests/manual", "tools/paper_verification"]), ("src/", ["evaluator"])]
PUBLIC = ["results/family_xloop_v3_20260726/base26/crossmodel_rltt_transfer.json",
          "results/family_xloop_v3_20260726/thinking26/crossmodel_rltt_transfer.json",
          "results/paper_verification/nonlooped_architecture_control_20260710_235252/"
          "corecontent_control_results_task_disjoint.json",
          "results/paper_verification/powered_clean_probe_and_corecontent_20260711/"
          "ouro_corecontent_task_disjoint_results.json"]

INPUTS: dict[str, str] = {}  # every input file read -> sha256


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def note(p: Path) -> Path:
    """Record an archive/repo input's sha256 (once) and return the path."""
    k = str(p)
    if k not in INPUTS:
        INPUTS[k] = sha(p)
    return p


def jload(p: Path):
    return json.loads(note(p).read_text())


def canon(x) -> str:
    return json.dumps(json.loads(json.dumps(x)), sort_keys=True)


def diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}/{k}: missing in {'b' if k in a else 'a'}" for k in sorted(set(a) ^ set(b))]
        for k in sorted(set(a) & set(b)):
            out += diff(a[k], b[k], f"{path}/{k}")
        return out
    return [] if a == b else [f"{path}: published={a!r} regenerated={b!r}"]


def load_t1():
    spec = importlib.util.spec_from_file_location("t1_fair_ties", note(T1))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ shared fair helpers
def tie_inventory(rows, fair_fn, score_key="scores") -> dict:
    st = {"groups": len(rows), "ties_at_max": 0, "all_equal": 0, "partial": 0, "changed": 0}
    for r in rows:
        s = r[score_key]
        n = s.count(max(s))
        if n < 2:
            continue
        st["ties_at_max"] += 1
        st["all_equal" if n == len(s) else "partial"] += 1
        st["changed"] += fair_fn(s, r["rewards"]) != r["top1_oracle"]
    return st


def macro(rows, field, domains) -> tuple[float, dict]:
    by = defaultdict(list)
    for r in rows:
        by[r["domain"]].append(r[field])
    dom = {d: float(np.mean(by[d])) for d in domains}
    return float(np.mean([dom[d] for d in domains])), dom


def add_fair(rows, fair_fn):
    return [{**r, "top1_oracle": fair_fn(r["scores"], r["rewards"])} for r in rows]


def series_of(rows, field="top1_oracle"):
    d = defaultdict(lambda: defaultdict(list))
    for r in rows:
        d[r["domain"]][r["task_uid"]].append(float(r[field]))
    return {dom: dict(t) for dom, t in d.items()}


def excl0(ci):
    return bool(ci[0] > 0 or ci[1] < 0)


# ------------------------------------------------------------------ task 1: C3 reverse transfer
def task_c3(X, TE, S, fair_fn, held_by):
    saved = jload(RLTT / "train_eval_results.json")
    bs = jload(RLTT / "bootstrap_stats.json")
    c3 = jload(C3 / "reverse_transfer_results.json")
    note(C3 / "reverse_transfer.py"), note(C3 / "README.md")
    taps = torch.load(note(RLTT / "xloop_taps.pt"), map_location="cpu", weights_only=False)
    layers, loops = (8, 16, 24), X.LOOPS
    cells = [(l, u) for l in layers for u in loops]
    back = [(l, u, v) for l in layers for u in loops for v in loops if v < u]
    W = {X.cell_name(l, u): taps[X.cell_name(l, u)]["weight"] for l, u in cells}
    mm = []

    refit_rows, refit_top = {}, {}
    for l, u in cells:
        cn = X.cell_name(l, u)
        refit_top[cn], _, refit_rows[cn] = TE.eval_cell(held_by, W[cn], l, u)
        if canon(refit_rows[cn]) != canon(jload(RLTT / f"predictions/refit_{cn}.json")):
            mm.append(f"refit_{cn}.json rows differ")
        if round(refit_top[cn], 4) != saved["cells"][cn]["heldout_macro_top1"]:
            mm.append(f"{cn} heldout_macro_top1")
        if TE.weight_hash(W[cn]) != saved["cells"][cn]["weight_sha"]:
            mm.append(f"{cn} weight_sha")

    def transfer(l, u, v):  # identical to C3's transfer()
        src, tgt = X.cell_name(l, u), X.cell_name(l, v)
        w = W[src]
        h = TE.weight_hash(w)
        mac, per_dom, rows = TE.eval_cell(held_by, w, l, v)
        assert TE.weight_hash(w) == h, "frozen head mutated"
        pear, spear, sign, grank = TE.score_agreement(rows, {r["group_uid"]: r for r in refit_rows[tgt]})
        return {"target_macro_top1": round(mac, 4), "target_macro_top1_unrounded": mac,
                "target_by_domain": {d: round(per_dom[d]["top1"], 4) for d in per_dom},
                "target_macro_pairwise": round(X.finite_mean(per_dom[d]["pairwise"] for d in per_dom), 4),
                "delta_vs_source_self": round(mac - refit_top[src], 4),
                "delta_vs_target_refit": round(mac - refit_top[tgt], 4),
                "score_pearson_vs_target_refit": pear, "score_spearman_vs_target_refit": spear,
                "pairwise_sign_agreement": sign, "mean_group_rank_corr": grank, "weight_sha": h}, rows

    tr, rows_of = {}, {}
    agree_keys = ("target_macro_top1", "target_macro_pairwise", "score_pearson_vs_target_refit",
                  "score_spearman_vs_target_refit", "pairwise_sign_agreement", "mean_group_rank_corr",
                  "weight_sha", "target_by_domain")
    for (l, u, v) in X.TRANSFERS:
        k = f"{l}:L{u}->L{v}"
        tr[k], rows_of[k] = transfer(l, u, v)
        if canon(rows_of[k]) != canon(jload(RLTT / f"predictions/transfer_{l}_L{u}_to_L{v}.json")):
            mm.append(f"transfer_{l}_L{u}_to_L{v}.json rows differ")
        mm += [f"{k} {a}" for a in agree_keys if tr[k][a] != saved["transfers"][k][a]]
    for (l, u, v) in back:
        k = f"{l}:L{u}->L{v}"
        tr[k], rows_of[k] = transfer(l, u, v)

    series = S.load_pred_vectors()  # original series from the saved files (verified equal above)
    for (l, u, v) in back:
        series[f"transfer:{l}:L{u}->L{v}"] = series_of(rows_of[f"{l}:L{u}->L{v}"])
    fair_series = {f"refit:{X.cell_name(l, u)}": series_of(add_fair(refit_rows[X.cell_name(l, u)], fair_fn))
                   for l, u in cells}
    for k, rows in rows_of.items():
        fair_series[f"transfer:{k}"] = series_of(add_fair(rows, fair_fn))
    points, samples = S.bootstrap(series, X.BOOT_DRAWS, X.BOOT_SEED)
    fpoints, fsamples = S.bootstrap(fair_series, X.BOOT_DRAWS, X.BOOT_SEED)

    def paired(P, Smp, a, b):
        d = Smp[a] - Smp[b]
        return {"point": round(P[a] - P[b], 4), "ci95": S.ci(d), "p_gt_zero": round(float((d > 0).mean()), 4)}

    for (l, u, v) in X.TRANSFERS:
        k = f"transfer:{l}:L{u}->L{v} - refit:{l}_L{v}"
        got = paired(points, samples, f"transfer:{l}:L{u}->L{v}", f"refit:{l}_L{v}")
        if got != bs["paired_deltas"][k]:
            mm.append(f"bootstrap {k}: {got} vs {bs['paired_deltas'][k]}")
    for (l, u, v) in X.TRANSFERS + back:
        k = f"{l}:L{u}->L{v}"
        tr[k]["direction"] = "forward" if v > u else "backward"
        tr[k]["paired_delta_vs_target_refit"] = paired(points, samples, f"transfer:{k}", f"refit:{l}_L{v}")
        tr[k]["paired_delta_vs_source_self"] = paired(points, samples, f"transfer:{k}", f"refit:{l}_L{u}")
    c3_mm = diff(json.loads(json.dumps(c3["transfers"])), json.loads(json.dumps(tr)), "/transfers")
    mm += c3_mm
    gate = {"check": "saved weights == saved weight_sha; regenerated refit_* (12) and forward transfer_* (18) "
                     "rows == saved files (canonical JSON); forward agreement stats == train_eval_results.json; "
                     "forward bootstrap deltas == bootstrap_stats.json; all 36 transfer entries (incl. backward "
                     "top-1, by-domain, Spearman/Pearson/sign/rank agreement, paired deltas, CIs, p>0) == "
                     "C3 reverse_transfer_results.json",
            "transfers_compared_with_c3": len(c3["transfers"]),
            "mismatches": mm, "pass": not mm}
    print(f"[task1 C3] gate {'PASS' if not mm else 'FAIL'} ({len(mm)} mismatches)", flush=True)
    for m in mm[:20]:
        print("   MISMATCH", m)
    if mm:
        return {"gate": gate}

    out = {}
    for (l, u, v) in back:
        k = f"{l}:L{u}->L{v}"
        a, b = f"transfer:{k}", f"refit:{l}_L{v}"
        o, f = tr[k]["paired_delta_vs_target_refit"], paired(fpoints, fsamples, a, b)
        out[k] = {"orig": {**o, "excl0": excl0(o["ci95"]), "point_unrounded": points[a] - points[b],
                           "transfer_top1": points[a], "target_refit_top1": points[b]},
                  "fair": {**f, "excl0": excl0(f["ci95"]), "point_unrounded": fpoints[a] - fpoints[b],
                           "transfer_top1": fpoints[a], "target_refit_top1": fpoints[b]},
                  "spearman_vs_target_refit": tr[k]["score_spearman_vs_target_refit"],
                  "spearman_recomputed_equals_c3": tr[k]["score_spearman_vs_target_refit"]
                  == c3["transfers"][k]["score_spearman_vs_target_refit"],
                  "ci_conclusion_changed": excl0(o["ci95"]) != excl0(f["ci95"]),
                  "ties_backward_rows": tie_inventory(rows_of[k], fair_fn),
                  "ties_target_refit_rows": tie_inventory(refit_rows[f"{l}_L{v}"], fair_fn)}
    return {"gate": gate, "backward": out}


# ------------------------------------------------------------------ task 2: shuffle controls
def task_shuffle(X, TE, fair_fn, train_by, val_by, held_by):
    saved = jload(RLTT / "train_eval_results.json")["shuffle_controls"]
    log = note(RLTT / "logs/train_eval.log").read_text().splitlines()
    rng = torch.Generator().manual_seed(X.BOOT_SEED)          # train_eval.main, verbatim order
    shuf_train, shuf_val = TE.shuffle_labels(train_by, rng), TE.shuffle_labels(val_by, rng)
    chance_macro = X.finite_mean(X.finite_mean(X.chance_top1(g) for g in held_by[d]) for d in X.CORE_DOMAINS)
    res, mm = {}, []
    for (layer, loop) in X.SHUFFLE_CELLS:
        cn = X.cell_name(layer, loop)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            best, grid = TE.fit_cell_grid(shuf_train, shuf_val, layer, loop, "shuffle ")
        printed = buf.getvalue().rstrip("\n")
        mac, per_dom, rows = TE.eval_cell(held_by, best["weight"], layer, loop)
        held_line = f"  shuffle {cn}: heldout={mac:.4f} (chance {chance_macro:.4f})"
        regen = {"heldout_macro_top1": round(mac, 4),
                 "by_domain": {d: round(per_dom[d]["top1"], 4) for d in per_dom},
                 "chance_macro": round(chance_macro, 4)}
        cm = diff(saved[cn], regen, f"/{cn}")
        if printed not in log:
            cm.append(f"{cn}: selection line {printed!r} not in train_eval.log")
        if held_line not in log:
            cm.append(f"{cn}: heldout line {held_line!r} not in train_eval.log")
        mm += cm
        fair_rows = add_fair(rows, fair_fn)
        fm, fdom = macro(fair_rows, "top1_oracle", X.CORE_DOMAINS)
        res[cn] = {"orig": {"macro": mac, "by_domain": {d: per_dom[d]["top1"] for d in per_dom}},
                   "fair": {"macro": fm, "by_domain": fdom},
                   "selected": {k: best[k] for k in ("tset", "lr", "seed")}, "val_macro": best["val_macro"],
                   "weight_sha": TE.weight_hash(best["weight"]), "ties": tie_inventory(rows, fair_fn),
                   "log_lines_matched": [printed.strip(), held_line.strip()]}
    gate = {"check": "refit via train_eval.shuffle_labels + fit_cell_grid (seed 20260720, 36-point grid, "
                     "val-only selection): heldout_macro_top1, by_domain and chance_macro (4 dp) == "
                     "train_eval_results.json shuffle_controls; printed selection line (best val, tset, lr, "
                     "seed) and heldout line == train_eval.log",
            "note": "shuffle weights were never saved; the gate is at the 4-dp precision of the record",
            "mismatches": mm, "pass": not mm}
    print(f"[task2 shuffle] gate {'PASS' if not mm else 'FAIL'} ({len(mm)} mismatches)", flush=True)
    for m in mm:
        print("   MISMATCH", m)
    return {"gate": gate, "cells": res} if not mm else {"gate": gate}


# ------------------------------------------------------------------ task 3: length baseline
def task_length(X, fair_fn):
    pub = jload(RLTT / "length_baseline.json")
    held = [g for g in (json.loads(l) for l in note(RLTT / "subset_groups.jsonl").read_text().splitlines() if l)
            if g["split"] == "heldout"]
    measures = {"chars": len, "utf8_bytes": lambda t: len(t.encode()), "stripped_chars": lambda t: len(t.strip())}
    res, mm = {"measures": {}}, []
    for mname, f in measures.items():
        R = {}
        for key, sign in (("length", 1), ("neg_length", -1)):
            rows = []
            for g in held:
                s = [float(sign * f(c["candidate_text"])) for c in g["candidates"]]
                rw = [float(c["reward"]) for c in g["candidates"]]
                m = X.group_selection_metrics({"candidates": [{"reward": r} for r in rw]}, s)
                mx = max(s)
                tied = [i for i, v in enumerate(s) if v == mx]
                rows.append({"domain": g["domain"], "scores": s, "rewards": rw, "top1_oracle": m["top1_oracle"],
                             "tied_texts_identical": len(tied) > 1 and len({g["candidates"][i]["candidate_text"]
                                                                             for i in tied}) == 1})
            om, odom = macro(rows, "top1_oracle", X.CORE_DOMAINS)
            fm, fdom = macro(add_fair(rows, fair_fn), "top1_oracle", X.CORE_DOMAINS)
            regen = {f"{key}_macro_top1": round(om, 4), f"{key}_by_domain": {d: round(odom[d], 4) for d in odom}}
            cm = diff({k: pub[k] for k in regen}, regen, f"/{mname}")
            inv = tie_inventory(rows, fair_fn)
            inv["ties_with_identical_text"] = sum(r["tied_texts_identical"] for r in rows)
            inv["ties_by_domain"] = {d: sum(1 for r in rows if r["domain"] == d and r["scores"].count(max(r["scores"])) > 1)
                                     for d in X.CORE_DOMAINS}
            R[key] = {"orig": {"macro": om, "by_domain": odom}, "fair": {"macro": fm, "by_domain": fdom},
                      "gate_mismatches": cm, "ties": inv}
        res["measures"][mname] = R
    # gate: the primary measure (chars) must reproduce; others reported for the measure ambiguity
    mm = res["measures"]["chars"]["length"]["gate_mismatches"] + res["measures"]["chars"]["neg_length"]["gate_mismatches"]
    res["measures_reproducing_exactly"] = [m for m, R in res["measures"].items()
                                           if not R["length"]["gate_mismatches"] and not R["neg_length"]["gate_mismatches"]]
    res["fair_identical_across_reproducing_measures"] = len({canon([res["measures"][m][k]["fair"] for k in ("length", "neg_length")])
                                                             for m in res["measures_reproducing_exactly"]}) == 1
    res["gate"] = {"check": "score = +/- len(candidate_text) (chars), first-max via original group_selection_metrics, "
                            "domain mean over the 600 heldout groups of subset_groups.jsonl, unweighted macro: "
                            "macro and per-domain (4 dp) == length_baseline.json, both signs",
                   "note": "no code for length_baseline.json exists in the commit or the run; the measure was "
                           "identified by reproduction (chars, UTF-8 bytes and stripped chars all reproduce)",
                   "mismatches": mm, "pass": not mm}
    print(f"[task3 length] gate {'PASS' if not mm else 'FAIL'}; reproducing measures {res['measures_reproducing_exactly']}",
          flush=True)
    return res


# ------------------------------------------------------------------ task 4: cross-checkpoint transfer
def task_crossmodel(X, TE, FX, fair_fn, src):
    out = {}
    taps_p = note(RLTT / "xloop_taps.pt")
    for key in ("base26", "thinking26"):
        root = FAM / key
        importlib.reload(X)
        mp = jload(root / "model_plan.json")
        plan = {"n_layers": mp["n_layers"], "hidden": mp["hidden"], "layers": tuple(mp["layers"]),
                "capture_idx": {int(k): v for k, v in mp["capture_idx"].items()},
                "boundary": mp["boundary"], "total_ut_steps": mp["total_ut_steps"]}
        FX.patch_common(key, plan)
        X.RUN_ROOT, X.FEATURE_ROOT, X.PRED_ROOT = root, root / "features", root / "predictions"
        FX.SEALED_ROOT = RLTT
        man = jload(root / "features/feature_manifest.json")
        shard_ok = all(note(root / "features" / s["file"]) and INPUTS[str(root / "features" / s["file"])] == s["sha256"]
                       for s in man["shards"])
        note(root / "train_eval_results.json")
        for cn in taps_names(X):
            note(root / f"predictions/refit_{cn}.json")
        captured, rows_of = {}, {}
        orig_eval, orig_write = TE.eval_cell, X.write_json

        def eval_wrap(groups_by_dom, weight, layer, loop):
            r = orig_eval(groups_by_dom, weight, layer, loop)
            rows_of[X.cell_name(layer, loop)] = r[2]
            return r
        TE.eval_cell = eval_wrap
        X.write_json = lambda path, payload: captured.__setitem__(str(path), json.loads(json.dumps(payload)))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                FX.stage_crossmodel(key, plan)
        finally:
            TE.eval_cell, X.write_json = orig_eval, orig_write
        regen = captured[str(root / "crossmodel_rltt_transfer.json")]
        pub_arch = jload(root / "crossmodel_rltt_transfer.json")
        pub_repo = json.loads((src / f"results/family_xloop_v3_20260726/{key}/crossmodel_rltt_transfer.json").read_text())
        mm = diff(pub_arch, regen) + (["archive and commit copies differ"] if pub_arch != pub_repo else [])
        if not shard_ok:
            mm.append("feature shard sha256 != manifest")
        cells = {}
        if not mm:
            for cn in regen:
                local_rows = json.loads((root / f"predictions/refit_{cn}.json").read_text())
                tap_rows = rows_of[cn]
                o_tap, _ = macro(tap_rows, "top1_oracle", X.CORE_DOMAINS)
                o_loc, _ = macro(local_rows, "top1_oracle", X.CORE_DOMAINS)
                f_tap, _ = macro(add_fair(tap_rows, fair_fn), "top1_oracle", X.CORE_DOMAINS)
                f_loc, _ = macro(add_fair(local_rows, fair_fn), "top1_oracle", X.CORE_DOMAINS)
                assert round(o_tap, 4) == regen[cn]["rltt_tap_macro_top1"]
                assert round(o_loc, 4) == regen[cn]["local_refit_macro_top1"]
                cells[cn] = {"orig": {"tap": o_tap, "local": o_loc, "delta": o_tap - o_loc},
                             "fair": {"tap": f_tap, "local": f_loc, "delta": f_tap - f_loc},
                             "agreement_pearson_spearman_sign_grouprank": regen[cn]["agreement_vs_local_refit"],
                             "ties_tap_rows": tie_inventory(tap_rows, fair_fn),
                             "ties_local_rows": tie_inventory(local_rows, fair_fn)}
        gate = {"check": "original stage_crossmodel (sealed RLTT xloop_taps.pt on this checkpoint's heldout "
                         "features; write_json captured in memory) == crossmodel_rltt_transfer.json (archive copy, "
                         "identical to commit copy), every cell and field; feature shards == manifest sha256",
                "mismatches": mm, "pass": not mm}
        print(f"[task4 crossmodel {key}] gate {'PASS' if not mm else 'FAIL'} ({len(mm)} mismatches)", flush=True)
        for m in mm[:20]:
            print("   MISMATCH", m)
        r = {"gate": gate}
        if cells:
            r["cells"] = cells
            for tag in ("orig", "fair"):
                ds = {cn: c[tag]["delta"] for cn, c in cells.items()}
                worst = max(ds, key=lambda cn: abs(ds[cn]))
                r[f"max_abs_delta_{tag}"] = {"cell": worst, "delta": ds[worst]}
        out[key] = r
    importlib.reload(X)
    return out


def taps_names(X):
    return [X.cell_name(l, u) for (l, u) in X.REFIT_CELLS]


# ------------------------------------------------------------------ task 5: MiniCPM 0.568 and Ouro 0.631
def fair_summary(per_group, domains, mean_fn) -> dict:
    """per_group: list of (domain, scores, rewards, top1_orig). Fair on exact and on 6-dp scores."""
    out = {}
    for tag, rnd in (("exact", None), ("6dp", 6)):
        by = defaultdict(list)
        for d, s, rw, _ in per_group:
            by[d].append(t1_fair(s if rnd is None else [round(x, rnd) for x in s], rw))
        dom = {d: mean_fn(by[d]) for d in domains}
        out[tag] = {"macro": mean_fn([dom[d] for d in domains]), "by_domain": dom}
    by = defaultdict(list)
    for d, s, rw, t in per_group:
        by[d].append(t)
    dom = {d: mean_fn(by[d]) for d in domains}
    out["orig_recomputed"] = {"macro": mean_fn([dom[d] for d in domains]), "by_domain": dom}
    ties = {"groups": len(per_group), "ties_at_max": 0, "all_equal": 0, "partial": 0, "changed": 0,
            "ties_at_max_6dp_only": 0}
    for d, s, rw, t in per_group:
        n = s.count(max(s))
        s6 = [round(x, 6) for x in s]
        ties["ties_at_max_6dp_only"] += n < 2 <= s6.count(max(s6))
        if n < 2:
            continue
        ties["ties_at_max"] += 1
        ties["all_equal" if n == len(s) else "partial"] += 1
        ties["changed"] += t1_fair(s, rw) != t
    out["ties"] = ties
    return out


def task_minicpm(M):
    pub = jload(MINI / "corecontent_control_results_task_disjoint.json")
    pub2 = jload(A2 / "opi/verification/paper_verification/nonlooped_architecture_control_20260710_235252/"
                      "corecontent_control_results_task_disjoint.json")
    man_p = MINI / "corecontent_features/feature_manifest.json"
    man = jload(man_p)
    man2 = note(A2 / "opi/verification/paper_verification/nonlooped_architecture_control_20260710_235252/"
                     "corecontent_features/feature_manifest.json")
    mm = [] if pub == pub2 else ["results json differs between archives"]
    mm += [] if INPUTS[str(man_p)] == INPUTS[str(man2)] else ["feature manifest differs between archives"]
    mm += [f"shard {s['file']} sha" for s in man["shards"]
           if note(MINI / "corecontent_features" / s["file"]) and
           INPUTS[str(MINI / "corecontent_features" / s["file"])] != s["sha256"]]
    pt = torch.load(note(MINI / "minicpm2_sft_corecontent_v2_task_disjoint.pt"), map_location="cpu", weights_only=False)
    channels = pt["channels"]
    groups = M.load_feature_groups(MINI)
    # verbatim split lines from nonlooped_transformer_control.train_corecontent (train-clean)
    stored = defaultdict(set)
    for g in groups: stored[g["task_uid"]].add(g["split"])
    for g in groups:
        u = int(hashlib.sha256(f"20260711:{g['task_uid']}".encode()).hexdigest()[:16], 16) / 2**64
        g["split"] = "train" if u < .8 else ("val" if u < .9 else "heldout")
    assigned = defaultdict(set)
    for g in groups: assigned[g["task_uid"]].add(g["split"])
    from collections import Counter
    audit = {"mode": "deterministic_task_disjoint_sha256", "seed": 20260711, "unique_task_ids": len(assigned),
             "task_ids_crossing_stored_splits": sum(len(v) > 1 for v in stored.values()),
             "task_ids_crossing_effective_splits": sum(len(v) > 1 for v in assigned.values()),
             "record_counts": dict(Counter(g["split"] for g in groups)),
             "unique_task_counts": dict(Counter(next(iter(v)) for v in assigned.values()))}
    mm += diff(pub["split_audit"], audit, "/split_audit")
    mm += [] if pt["split_audit"] == pub["split_audit"] else ["pt split_audit != json"]
    mm += [f"channel {c} weight != best[{c}]" for c, w in channels if not torch.equal(w, pt["best"][c]["weight"])]
    val, held = M.evaluate(groups, channels, "val"), M.evaluate(groups, channels, "heldout")
    mm += diff(pub["blockwise_val"], json.loads(json.dumps(val)), "/blockwise_val")
    mm += diff(pub["blockwise_heldout"], json.loads(json.dumps(held)), "/blockwise_heldout")
    gate = {"check": "stored features (sha256 == manifest) + saved channels (minicpm2_sft_corecontent_v2_task_disjoint"
                     ".pt) + original split lines and evaluate(): split_audit and blockwise_val / blockwise_heldout "
                     "(unrounded floats, per domain and macro) == corecontent_control_results_task_disjoint.json",
            "mismatches": mm, "pass": not mm}
    print(f"[task5 MiniCPM] gate {'PASS' if not mm else 'FAIL'} ({len(mm)} mismatches)", flush=True)
    for m in mm[:20]:
        print("   MISMATCH", m)
    if mm:
        return {"gate": gate}
    rows = []
    for g in groups:
        if g["split"] == "heldout" and g["domain"] in M.CORE_DOMAINS:
            s = M.channel_scores(g, channels)
            rows.append((g["domain"], s, [float(c["reward"]) for c in g["candidates"]], M.group_metrics(g, s)[0]))
    fs = fair_summary(rows, M.CORE_DOMAINS, lambda v: float(np.mean(v)))
    assert fs["orig_recomputed"]["macro"] == held["macro_top1"]
    # Sensitivity: with 2 candidates each channel's z-scores are exactly (-1, +1); when the 24_L4 and 36_L4
    # channels disagree in sign the averaged scores are mathematically (0, 0), a tie that float32 noise breaks.
    # Treat every such group (and any group already tied) as tied at the max.
    by, n_dis = defaultdict(list), 0
    for g in groups:
        if g["split"] != "heldout" or g["domain"] not in M.CORE_DOMAINS:
            continue
        rw = [float(c["reward"]) for c in g["candidates"]]
        s = M.channel_scores(g, channels)
        if len(rw) == 2:
            signs = []
            for cfg, w in channels:  # same per-channel z as channel_scores
                v = torch.tensor([torch.dot(w, c["features"][cfg].float()) for c in g["candidates"]])
                sd = v.std(unbiased=False)
                z = (v - v.mean()) / sd if sd > 1e-8 else torch.zeros_like(v)
                signs.append((float(z[0]) > 0) - (float(z[0]) < 0))  # zeroed channel -> 0
            if signs[0] * signs[1] == -1 and s.count(max(s)) < 2:
                n_dis += 1
                s = [0.0, 0.0]
        by[g["domain"]].append(t1_fair(s, rw))
    dom = {d: float(np.mean(by[d])) for d in M.CORE_DOMAINS}
    sens = {"macro": float(np.mean(list(dom.values()))), "by_domain": dom,
            "extra_groups_treated_as_tied": n_dis,
            "rule": "2-candidate groups whose two channel z-scores disagree in sign scored (0, 0), i.e. tied"}
    return {"gate": gate, "orig": {"macro": held["macro_top1"],
                                   "by_domain": {d: held["domains"][d]["top1"] for d in M.CORE_DOMAINS}},
            "fair": fs["exact"], "fair_6dp_scores": fs["6dp"], "ties": fs["ties"],
            "sensitivity_channel_disagreement_as_tie": sens}


def task_ouro_td(TD, feat, models, cc):
    pub = jload(OURO_TD / "ouro_corecontent_task_disjoint_results.json")
    mm = []
    feat.FEATURE_ROOT = V2_DATA / "features"
    man = jload(V2_DATA / "features/feature_manifest.json")
    mm += [f"shard {s['file']} checksum" for s in man["shards"]  # manifest stores sha256[:16] (feat._checksum)
           if note(V2_DATA / "features" / s["file"]) and INPUTS[str(V2_DATA / "features" / s["file"])][:16] != s["checksum"]]
    pt = torch.load(note(OURO_TD / "ouro_corecontent_task_disjoint_policy.pt"), map_location="cpu", weights_only=False)
    channels = pt["channels"]
    groups = feat.load_feature_groups()
    # verbatim from rerun_ouro_corecontent_task_disjoint.main
    from collections import Counter
    stored = defaultdict(set)
    for g in groups: stored[g["task_id"]].add(g["split"])
    for g in groups: g["split"] = TD.assigned_split(g["task_id"])
    effective = defaultdict(set)
    for g in groups: effective[g["task_id"]].add(g["split"])
    by = defaultdict(list)
    for g in groups: by[(g["domain"], g["split"])].append(g)
    val_by = {d: by[(d, "val")] for d in models.CORE_DOMAINS}
    held_by = {d: by[(d, "heldout")] for d in models.CORE_DOMAINS}
    audit = {"seed": TD.SEED, "rule": "sha256(seed:task_id), 80/10/10 thresholds", "records": len(groups),
             "unique_tasks": len(effective),
             "stored_task_ids_crossing_splits": sum(len(v) > 1 for v in stored.values()),
             "effective_task_ids_crossing_splits": sum(len(v) > 1 for v in effective.values()),
             "record_counts": dict(Counter(g["split"] for g in groups)),
             "task_counts": dict(Counter(next(iter(v)) for v in effective.values()))}
    mm += diff(pub["split_audit"], audit, "/split_audit")
    mm += [f"channel {c} weight != best[{c}]" for (w, a, c) in channels if not torch.equal(w, pt["best_per_config"][c]["weight"])]
    val_dom, held_dom = models.eval_channels_macro(channels, val_by), models.eval_channels_macro(channels, held_by)
    regen = {"validation": {"groups": {d: len(val_by[d]) for d in models.CORE_DOMAINS}, "top1_by_domain": val_dom,
                            "macro_top1": models.macro_core(val_dom)},
             "heldout": {"groups": {d: len(held_by[d]) for d in models.CORE_DOMAINS}, "top1_by_domain": held_dom,
                         "macro_top1": models.macro_core(held_dom)}}
    mm += diff({k: pub[k] for k in regen}, json.loads(json.dumps(regen)))
    gate = {"check": "CoreContent-v2 feature shards (sha256 == manifest) + saved channels (ouro_corecontent_task_"
                     "disjoint_policy.pt) + original split (assigned_split) and eval_channels_macro: split_audit, "
                     "group counts, validation and heldout top1_by_domain and macro (unrounded floats) == "
                     "ouro_corecontent_task_disjoint_results.json",
            "mismatches": mm, "pass": not mm}
    print(f"[task5 Ouro 0.631] gate {'PASS' if not mm else 'FAIL'} ({len(mm)} mismatches)", flush=True)
    for m in mm[:20]:
        print("   MISMATCH", m)
    if mm:
        return {"gate": gate}
    rows = []
    for d in models.CORE_DOMAINS:
        for g in held_by[d]:
            s = cc.policy_candidate_scores(g, channels)
            m = models.metrics(g, s)
            if m:
                rows.append((d, s, [float(c["reward"]) for c in g["candidates"]], m["top1_oracle"]))
    fs = fair_summary(rows, models.CORE_DOMAINS, models.finite_mean)
    assert fs["orig_recomputed"]["macro"] == regen["heldout"]["macro_top1"]
    return {"gate": gate, "orig": {"macro": regen["heldout"]["macro_top1"], "by_domain": held_dom},
            "fair": fs["exact"], "fair_6dp_scores": fs["6dp"], "ties": fs["ties"]}


# ------------------------------------------------------------------ main
def main() -> int:
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="t3_fair_ties_"))
    try:
        return analyse(tmp, t0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def analyse(tmp: Path, t0: float) -> int:
    src = tmp / "src"
    src.mkdir()
    full = subprocess.run(["git", "-C", str(REPO), "rev-parse", COMMIT], check=True,
                          capture_output=True, text=True).stdout.strip()
    tar = subprocess.run(["git", "-C", str(REPO), "archive", full, *CODE, *PUBLIC],
                         check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
        tf.extractall(src, filter="data")
    proj = tmp / "proj"
    for prefix, paths in PROJ_TREES:
        tar = subprocess.run(["git", "-C", str(REPO), "archive", f"--prefix={prefix}", full, *paths],
                             check=True, capture_output=True).stdout
        with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
            tf.extractall(proj, filter="data")
    commit_sha = {p: sha(src / p) for p in CODE + PUBLIC}

    sys.path[:0] = [str(src / "probes"), str(src / "tools/paper1_v3")]
    import bg_xloop_early_v1_common as X
    import bg_xloop_early_v1_train_eval as TE
    import bg_xloop_early_v1_stats as S
    import bg_v3_family_xloop as FX
    T1m = load_t1()
    fair_fn = T1m.fair_top1
    global t1_fair
    t1_fair = fair_fn

    results = {"date": "2026-11-08", "post_hoc": True, "commit": full,
               "fair_rule": "top1_fair = #tied-at-max-score with reward >= max reward / #tied-at-max; exact "
                            "equality of the scores the original rule ranks (6-dp saved scores for the xloop "
                            "rows; integer lengths; see task 5 for unrounded scores)",
               "torch": torch.__version__, "numpy": np.__version__, "python": sys.version.split()[0],
               "torch_threads": torch.get_num_threads(), "tasks": {}}

    # RLTT run: stage-0 style shard check, then groups once for tasks 1-2
    X.RUN_ROOT, X.FEATURE_ROOT, X.PRED_ROOT = RLTT, RLTT / "features", RLTT / "predictions"
    man = jload(RLTT / "features/feature_manifest.json")
    for s in man["shards"]:
        note(RLTT / "features" / s["file"])
    for p in sorted((RLTT / "predictions").glob("*.json")):
        if p.name.startswith(("refit_", "transfer_")):  # all read by S.load_pred_vectors
            note(p)
    shard_mm = [s["file"] for s in man["shards"] if INPUTS[str(RLTT / "features" / s["file"])] != s["sha256"]]
    results["rltt_feature_shards"] = {"n": len(man["shards"]), "sha256_mismatch_vs_manifest": shard_mm}
    if shard_mm:
        print("RLTT shard sha mismatch", shard_mm)
        return finish(results, t0, 1)
    train_by, val_by, held_by = ({d: [] for d in X.CORE_DOMAINS} for _ in range(3))
    for g in X.load_fresh_groups():
        {"train": train_by, "val": val_by, "heldout": held_by}[g["split"]][g["domain"]].append(g)

    results["tasks"]["1_c3_reverse_transfer"] = task_c3(X, TE, S, fair_fn, held_by)
    results["tasks"]["2_shuffle_controls"] = task_shuffle(X, TE, fair_fn, train_by, val_by, held_by)
    del train_by, val_by, held_by
    results["tasks"]["3_length_baseline"] = task_length(X, fair_fn)
    results["tasks"]["4_crossmodel"] = task_crossmodel(X, TE, FX, fair_fn, src)

    # task 5: original modules from the commit, in the original tree layout
    sys.path[:0] = [str(proj / "shared/utilities/tests/manual"), str(proj / "shared/tools/paper_verification")]
    import nonlooped_transformer_control as M
    results["tasks"]["5a_minicpm"] = task_minicpm(M)
    import rerun_ouro_corecontent_task_disjoint as TD
    import bg_corecontent_v2_features as feat
    import bg_corecontent_v2_models as models
    import bg_core_tap_audit_v1_common as cc
    results["tasks"]["5b_ouro_task_disjoint"] = task_ouro_td(TD, feat, models, cc)
    t5 = results["tasks"]
    if "fair" in t5["5a_minicpm"] and "fair" in t5["5b_ouro_task_disjoint"]:
        t5["5_gap_ouro_minus_minicpm"] = {
            k: t5["5b_ouro_task_disjoint"][k]["macro"] - t5["5a_minicpm"][k]["macro"] for k in ("orig", "fair")}

    commit_sha.update({str(Path(m.__file__).relative_to(proj)): sha(Path(m.__file__))
                       for m in list(sys.modules.values())
                       if getattr(m, "__file__", None) and str(m.__file__).startswith(str(proj))})
    results["commit_files_sha256"] = commit_sha
    return finish(results, t0, 0)


def finish(results, t0, code):
    results["inputs_sha256"] = dict(sorted(INPUTS.items()))
    (OUT / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    (OUT / "RESULTS.md").write_text(render(json.loads((OUT / "results.json").read_text())))
    names = [n for n in ("t3_fair_ties_archive.py", "results.json", "RESULTS.md", "README.md") if (OUT / n).exists()]
    (OUT / "SHA256SUMS").write_text("".join(f"{sha(OUT / n)}  {n}\n" for n in names))
    print(f"results.json sha256 {sha(OUT / 'results.json')}  elapsed {time.time() - t0:.0f}s", flush=True)
    return code


# ------------------------------------------------------------------ markdown
def f4(x):
    return f"{x + 0.0:.4f}"  # + 0.0 turns -0.0 into 0.0


def s4(x):
    return f"{x + 0.0:+.4f}"


def ci_s(c):
    return f"[{c[0] + 0.0:.4f}, {c[1] + 0.0:.4f}]"


def gate_line(name, g):
    return f"| {name} | {'PASS' if g['pass'] else 'FAIL'} | {len(g['mismatches'])} | {g['check']} |"


def render(R) -> str:
    T = R["tasks"]
    L = ["# T3 fair-tie top-1 on archived series: original vs fair", "",
         f"Post hoc, 2026-11-08. Commit `{R['commit']}`; archive inputs listed in README.md. Fair rule (T1's "
         f"`fair_top1`): {R['fair_rule']}. Bootstrap where shown: original `bg_xloop_early_v1_stats.bootstrap`, "
         "seed 20260720, 10,000 task-clustered draws, the same draws for original and fair. "
         f"torch {R['torch']}, {R['torch_threads']} threads. All values 4 dp.", "",
         "## Gates (original first-max rule must reproduce the published numbers first)", "",
         "| task | gate | mismatches | exact check |", "|---|---|---|---|"]
    L.append(gate_line("1 C3 reverse transfer", T["1_c3_reverse_transfer"]["gate"]))
    L.append(gate_line("2 shuffled-label controls", T["2_shuffle_controls"]["gate"]))
    L.append(gate_line("3 length baseline", T["3_length_baseline"]["gate"]))
    for m, v in T["4_crossmodel"].items():
        L.append(gate_line(f"4 cross-checkpoint {m}", v["gate"]))
    for k, name in (("5a_minicpm", "5 MiniCPM 0.568"), ("5b_ouro_task_disjoint", "5 Ouro 0.631")):
        if k in T:
            L.append(gate_line(name, T[k]["gate"]))

    c3 = T["1_c3_reverse_transfer"]
    if "backward" in c3:
        L += ["", "## 1. C3 reverse transfer: move minus probe trained at the target loop", "",
              "Saved `xloop_taps.pt` weights; Spearman = transferred vs target-refit scores on the same candidates "
              "(score-based, so the tie rule cannot change it; recomputed values equal C3's).", "",
              "| move | orig delta | orig CI | excl 0 | fair delta | fair CI | excl 0 | CI conclusion changed | "
              "Spearman | ties at max: move / target rows (changed top-1) |", "|---|---|---|---|---|---|---|---|---|---|"]
        order = [(l, u, v) for l in (8, 16, 24) for (u, v) in ((2, 1), (3, 1), (4, 1), (3, 2), (4, 2), (4, 3))]
        for (l, u, v) in order:
            d = c3["backward"][f"{l}:L{u}->L{v}"]
            o, f = d["orig"], d["fair"]
            tb, tt = d["ties_backward_rows"], d["ties_target_refit_rows"]
            L.append(f"| {l}: {u}->{v} | {s4(o['point'])} | {ci_s(o['ci95'])} | {'yes' if o['excl0'] else 'no'} | "
                     f"{s4(f['point'])} | {ci_s(f['ci95'])} | {'yes' if f['excl0'] else 'no'} | "
                     f"{'YES' if d['ci_conclusion_changed'] else 'no'} | {d['spearman_vs_target_refit']:.4f} | "
                     f"{tb['ties_at_max']} ({tb['changed']}) / {tt['ties_at_max']} ({tt['changed']}) |")
        n = sum(d["ci_conclusion_changed"] for d in c3["backward"].values())
        L += ["", f"CI-excludes-zero conclusions changed: **{n} of {len(c3['backward'])}**."]

    sh = T["2_shuffle_controls"]
    if "cells" in sh:
        L += ["", "## 2. Shuffled-label controls (refit; weights were never saved)", "",
              "| cell | orig | fair | fair - orig | selected (val) | ties at max (all-equal / partial; changed) |",
              "|---|---|---|---|---|---|"]
        for cn in ("8_L1", "8_L4", "16_L2", "24_L4"):
            c = sh["cells"][cn]
            t = c["ties"]
            sel = c["selected"]
            L.append(f"| {cn} | {f4(c['orig']['macro'])} | {f4(c['fair']['macro'])} | "
                     f"{s4(c['fair']['macro'] - c['orig']['macro'])} | {sel['tset']}, lr={sel['lr']}, seed={sel['seed']} "
                     f"({f4(c['val_macro'])}) | {t['ties_at_max']} ({t['all_equal']} / {t['partial']}; {t['changed']}) |")

    ln = T["3_length_baseline"]
    if ln["gate"]["pass"]:
        ch = ln["measures"]["chars"]
        L += ["", "## 3. Length baseline (score = candidate_text length in characters)", "",
              f"Measures reproducing the published values exactly: {', '.join(ln['measures_reproducing_exactly'])}; "
              f"fair values identical across them: {ln['fair_identical_across_reproducing_measures']}.", "",
              "| rule | orig macro | fair macro | " + " | ".join(f"{d} orig -> fair" for d in ch["length"]["orig"]["by_domain"])
              + " | ties at max (identical text) | changed top-1 |",
              "|---|---|---|" + "---|" * len(ch["length"]["orig"]["by_domain"]) + "---|---|"]
        for key, name in (("length", "longest first"), ("neg_length", "shortest first")):
            r = ch[key]
            L.append(f"| {name} | {f4(r['orig']['macro'])} | {f4(r['fair']['macro'])} | "
                     + " | ".join(f"{f4(r['orig']['by_domain'][d])} -> {f4(r['fair']['by_domain'][d])}"
                                  for d in r["orig"]["by_domain"])
                     + f" | {r['ties']['ties_at_max']} of 600 ({r['ties']['ties_with_identical_text']}) | "
                       f"{r['ties']['changed']} |")
        L += ["", "Ties at the max by domain (longest / shortest): " + ", ".join(
            f"{d} {ch['length']['ties']['ties_by_domain'][d]} / {ch['neg_length']['ties']['ties_by_domain'][d]}"
            for d in ch["length"]["ties"]["ties_by_domain"]) + "."]

    cm = T["4_crossmodel"]
    if all("cells" in v for v in cm.values()):
        L += ["", "## 4. Cross-checkpoint transfer: RLTT taps minus each checkpoint's own probe", "",
              "| cell | base26 orig | base26 fair | thinking26 orig | thinking26 fair | base26 / thinking26 Pearson, "
              "Spearman (unchanged) | ties at max base26 tap/local, thinking26 tap/local |", "|---|---|---|---|---|---|---|"]
        for cn in cm["base26"]["cells"]:
            b, t = cm["base26"]["cells"][cn], cm["thinking26"]["cells"][cn]
            ab, at = b["agreement_pearson_spearman_sign_grouprank"], t["agreement_pearson_spearman_sign_grouprank"]
            L.append(f"| {cn} | {s4(b['orig']['delta'])} | {s4(b['fair']['delta'])} | {s4(t['orig']['delta'])} | "
                     f"{s4(t['fair']['delta'])} | {ab[0]:.4f}, {ab[1]:.4f} / {at[0]:.4f}, {at[1]:.4f} | "
                     f"{b['ties_tap_rows']['ties_at_max']}/{b['ties_local_rows']['ties_at_max']}, "
                     f"{t['ties_tap_rows']['ties_at_max']}/{t['ties_local_rows']['ties_at_max']} |")
        L.append("")
        for m, v in cm.items():
            o, f = v["max_abs_delta_orig"], v["max_abs_delta_fair"]
            L.append(f"- {m}: largest |delta| orig {s4(o['delta'])} ({o['cell']}), fair {s4(f['delta'])} ({f['cell']}).")

    if "5a_minicpm" in T:
        L += ["", "## 5. MiniCPM non-looped control vs Ouro task-disjoint CoreContent", "",
              "Fair rule on the unrounded scores the original argmax used (exact equality); `6dp` repeats it on "
              "scores rounded to 6 decimals, as T1's saved files were.", "",
              "| model | orig macro | fair macro | fair macro (6dp) | " + " | ".join(
                  f"{d} orig -> fair" for d in ("coding", "reasoning", "math", "logic", "alignment"))
              + " | heldout groups | ties at max (all-equal / partial; changed) |",
              "|---|---|---|---|" + "---|" * 5 + "---|---|"]
        for k, name in (("5a_minicpm", "MiniCPM-2B-sft"), ("5b_ouro_task_disjoint", "Ouro (task-disjoint)")):
            r = T[k]
            if "fair" not in r:
                L.append(f"| {name} | gate failed | | | | | | | | | |")
                continue
            t = r["ties"]
            L.append(f"| {name} | {f4(r['orig']['macro'])} | {f4(r['fair']['macro'])} | "
                     f"{f4(r['fair_6dp_scores']['macro'])} | " + " | ".join(
                         f"{f4(r['orig']['by_domain'][d])} -> {f4(r['fair']['by_domain'][d])}"
                         for d in ("coding", "reasoning", "math", "logic", "alignment"))
                     + f" | {t['groups']} | {t['ties_at_max']} ({t['all_equal']} / {t['partial']}; {t['changed']}) |")
        sv = T["5a_minicpm"].get("sensitivity_channel_disagreement_as_tie")
        if sv:
            L += ["", "MiniCPM sensitivity: in a 2-candidate group each channel's z-scores are exactly (-1, +1), so "
                  "when the 24_L4 and 36_L4 channels disagree the averaged scores are mathematically (0, 0); "
                  "float32 noise (up to ~2e-5) breaks some of these ties. Treating all of them as ties: macro "
                  f"**{f4(sv['macro'])}** ({sv['extra_groups_treated_as_tied']} groups added to the exact ties; "
                  "alignment " + f4(sv["by_domain"]["alignment"]) + "). Ouro averages 3 channels (each +/-1), so a "
                  "sign disagreement alone cannot cancel there; float-broken ties via a zeroed channel were not "
                  "checked for Ouro."]
        if "5_gap_ouro_minus_minicpm" in T:
            g = T["5_gap_ouro_minus_minicpm"]
            L += ["", f"Ouro minus MiniCPM: orig {s4(g['orig'])}, fair (exact) {s4(g['fair'])}"
                  + (f", fair with the MiniCPM sensitivity {s4(T['5b_ouro_task_disjoint']['fair']['macro'] - sv['macro'])}"
                     if sv else "") + "."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
