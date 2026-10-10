"""C3 post-hoc backward-transfer check for the cross-loop ranking probes (OPI v4 audit, correction 1).

POST HOC: designed 2026-10-10, after the original results, in response to an external audit.
Reads the 2026-07-20 run read-only from the external SSD; writes only into this folder.

Stages:
  0   SHA-256 of every feature shard vs both manifests; `sha256sum -c SHA256SUMS` of the run records.
  1a  Refit gate (reported, not blocking): refit the 12 probes (layers 8/16/24 x loops 1-4) with the
      original code/grid/val-only selection/seeds; compare with saved refit_*/transfer_* prediction
      files, saved weight hashes, saved weights (xloop_taps.pt), selection grid, agreement stats.
  1b  Saved-weight gate (blocking): the original selected weights in xloop_taps.pt must match the saved
      weight hashes and reproduce every saved refit_* and forward transfer_* file and agreement stat
      exactly. Any mismatch -> JSON with discrepancies, exit 1, no stage 2.
  2   Backward transfers (u -> v, v < u) with the saved weights, original eval/agreement code, and the
      original paired task-clustered bootstrap (10,000 draws, seed 20260720). Saved forward bootstrap
      deltas are re-derived as a check. Refit-weight backward top-1 kept as a sensitivity.
  3   Full 4x4 top-1 matrix per layer; cosine similarity between probe weight vectors.

Usage (threads: 1-4 reproduce 11/12 refits bitwise, 8 does not):
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" \
      /home/moloch/ouro_project/shared/venv/bin/python -u reverse_transfer.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import torch

CODE = Path("/home/moloch/ouro_project/shared/utilities/tests/manual")
SRC = Path("/run/media/moloch/ARCH_BACKUP/OURO_COLD_ARCHIVE_20260729/historical_experiments/"
           "paper_v3_audit_evidence/artifacts/reports/cross_loop_early_layer_taps_20260720")
OUT = Path(__file__).resolve().parent / "reverse_transfer_results.json"

sys.path.insert(0, str(CODE))
import bg_xloop_early_v1_common as X  # noqa: E402
import bg_xloop_early_v1_train_eval as TE  # noqa: E402
import bg_xloop_early_v1_stats as S  # noqa: E402

# Point the original modules at the archived run (read-only: only loaders are called, never
# ensure_roots / write_json / main).
X.RUN_ROOT, X.FEATURE_ROOT, X.PRED_ROOT = SRC, SRC / "features", SRC / "predictions"

LAYERS = (8, 16, 24)
LOOPS = X.LOOPS
CELLS = [(l, u) for l in LAYERS for u in LOOPS]
BACK = [(l, u, v) for l in LAYERS for u in LOOPS for v in LOOPS if v < u]
AGREE_KEYS = ("target_macro_top1", "target_macro_pairwise", "score_pearson_vs_target_refit",
              "score_spearman_vs_target_refit", "pairwise_sign_agreement", "mean_group_rank_corr",
              "weight_sha", "target_by_domain")
res: dict = {"post_hoc": True, "torch": torch.__version__, "threads": torch.get_num_threads(),
             "source_run": str(SRC)}


def canon(rows) -> str:
    return json.dumps(json.loads(json.dumps(rows)), sort_keys=True)


def saved_rows(name: str) -> list:
    return json.loads((SRC / f"predictions/{name}.json").read_text())


def dump(code: int) -> int:
    OUT.write_text(json.dumps(res, indent=1, default=str))
    print(f"wrote {OUT}", flush=True)
    return code


def main() -> int:
    t0 = time.time()
    # ---------------------------------------------------------------- stage 0
    man_root = json.loads((SRC / "feature_manifest.json").read_text())
    man_feat = json.loads((SRC / "features/feature_manifest.json").read_text())
    exp_root = {s["file"]: s["sha256"] for s in man_root["per_shard"]}
    exp_feat = {s["file"]: s["sha256"] for s in man_feat["shards"]}
    shard_rows = []
    for f in sorted((SRC / "features").glob("xloop_*.pt")):
        h = X.sha256_file(f)
        shard_rows.append({"file": f.name, "sha256": h, "manifest_root": exp_root.get(f.name) == h,
                           "manifest_features": exp_feat.get(f.name) == h})
    sums = subprocess.run(["sha256sum", "-c", "--quiet", "SHA256SUMS"], cwd=SRC,
                          capture_output=True, text=True)
    ok0 = (len(shard_rows) == man_root["n_shards"] == len(exp_feat)
           and all(r["manifest_root"] and r["manifest_features"] for r in shard_rows))
    res["stage0"] = {
        "n_shards": len(shard_rows), "manifest_n_shards": man_root["n_shards"],
        "all_shards_match_both_manifests": ok0, "shards": shard_rows,
        "SHA256SUMS_entries": sum(1 for _ in open(SRC / "SHA256SUMS")),
        "SHA256SUMS_check_rc": sums.returncode, "SHA256SUMS_check_output": sums.stdout + sums.stderr,
    }
    print(f"stage0: shards ok={ok0} ({len(shard_rows)}), SHA256SUMS rc={sums.returncode}", flush=True)
    if not ok0 or sums.returncode != 0:
        return dump(1)

    train_by = {d: [] for d in X.CORE_DOMAINS}
    val_by = {d: [] for d in X.CORE_DOMAINS}
    held_by = {d: [] for d in X.CORE_DOMAINS}
    for g in X.load_fresh_groups():
        {"train": train_by, "val": val_by, "heldout": held_by}[g["split"]][g["domain"]].append(g)
    saved = json.loads((SRC / "train_eval_results.json").read_text())
    taps = torch.load(SRC / "xloop_taps.pt", map_location="cpu", weights_only=False)

    def cell_eval(Wset):
        rows_, top_ = {}, {}
        for layer, loop in CELLS:
            cn = X.cell_name(layer, loop)
            top_[cn], _, rows_[cn] = TE.eval_cell(held_by, Wset[cn], layer, loop)
        return rows_, top_

    def transfer(Wset, rows_ref, top_ref, layer, u, v):
        src, tgt = X.cell_name(layer, u), X.cell_name(layer, v)
        w = Wset[src]
        h = TE.weight_hash(w)
        macro, per_dom, rows = TE.eval_cell(held_by, w, layer, v)
        assert TE.weight_hash(w) == h, "frozen head mutated"
        pear, spear, sign, grank = TE.score_agreement(rows, {r["group_uid"]: r for r in rows_ref[tgt]})
        return {"target_macro_top1": round(macro, 4), "target_macro_top1_unrounded": macro,
                "target_by_domain": {d: round(per_dom[d]["top1"], 4) for d in per_dom},
                "target_macro_pairwise": round(X.finite_mean(per_dom[d]["pairwise"] for d in per_dom), 4),
                "delta_vs_source_self": round(macro - top_ref[src], 4),
                "delta_vs_target_refit": round(macro - top_ref[tgt], 4),
                "score_pearson_vs_target_refit": pear, "score_spearman_vs_target_refit": spear,
                "pairwise_sign_agreement": sign, "mean_group_rank_corr": grank, "weight_sha": h}, rows

    def check_cells_and_forward(Wset, tag):
        mm, detail = [], {}
        rows_, top_ = cell_eval(Wset)
        for layer, loop in CELLS:
            cn = X.cell_name(layer, loop)
            sv = saved_rows(f"refit_{cn}")
            c = {"rows_equal_saved_file": canon(rows_[cn]) == canon(sv),
                 "per_group_top1_equal": [r["top1_oracle"] for r in rows_[cn]] == [r["top1_oracle"] for r in sv],
                 "macro_top1_equal": round(top_[cn], 4) == saved["cells"][cn]["heldout_macro_top1"],
                 "weight_sha_equal": TE.weight_hash(Wset[cn]) == saved["cells"][cn]["weight_sha"],
                 "weight_equal_xloop_taps": torch.equal(Wset[cn], taps[cn]["weight"]),
                 "max_abs_weight_diff_vs_xloop_taps": float((Wset[cn] - taps[cn]["weight"]).abs().max())}
            detail[cn] = c
            mm += [f"{tag} cell {cn}: {k}" for k, ok in c.items() if ok is False]
        fwd = {}
        for (layer, u, v) in X.TRANSFERS:
            key = f"{layer}:L{u}->L{v}"
            t, rows = transfer(Wset, rows_, top_, layer, u, v)
            fwd[key] = (t, rows)
            sv = saved_rows(f"transfer_{layer}_L{u}_to_L{v}")
            c = {"rows_equal_saved_file": canon(rows) == canon(sv),
                 "per_group_top1_equal": [r["top1_oracle"] for r in rows] == [r["top1_oracle"] for r in sv]}
            c.update({k: t[k] == saved["transfers"][key][k] for k in AGREE_KEYS})
            detail[key] = c
            mm += [f"{tag} transfer {key}: {k}" for k, ok in c.items() if ok is False]
        print(f"GATE {tag}: {'EXACT' if not mm else 'NOT EXACT'} ({len(mm)} mismatches)", flush=True)
        for m in mm:
            print("  MISMATCH", m, flush=True)
        return {"exact": not mm, "mismatches": mm, "checks": detail}, rows_, top_, fwd

    # ---------------------------------------------------------------- stage 1a: refit gate
    W_refit, refit_sel = {}, {}
    for layer, loop in CELLS:
        cn = X.cell_name(layer, loop)
        best, grid_rows = TE.fit_cell_grid(train_by, val_by, layer, loop, "refit ")
        TE._DIFF_CACHE.clear()  # memory only; cache is rebuilt identically
        W_refit[cn] = best["weight"]
        refit_sel[cn] = {"selected_equal": {k: best[k] for k in ("tset", "lr", "seed")} == saved["cells"][cn]["selected"],
                         "grid_val_equal": grid_rows == saved["cells"][cn]["grid"]}
    g1a, rows_r, top_r, _ = check_cells_and_forward(W_refit, "refit")
    for cn, c in refit_sel.items():
        g1a["checks"][cn].update(c)
        g1a["mismatches"] += [f"refit cell {cn}: {k}" for k, ok in c.items() if not ok]
    g1a["exact"] = not g1a["mismatches"]
    res["gate_1a_refit"] = g1a

    # ---------------------------------------------------------------- stage 1b: saved-weight gate
    W = {X.cell_name(l, u): taps[X.cell_name(l, u)]["weight"] for l, u in CELLS}
    g1b, refit_rows, refit_top1, fwd = check_cells_and_forward(W, "saved_weights")
    res["gate_1b_saved_weights"] = g1b
    if not g1b["exact"]:
        return dump(1)

    # ---------------------------------------------------------------- stage 2
    transfers = {k: t for k, (t, _) in fwd.items()}
    transfer_rows = {k: r for k, (_, r) in fwd.items()}
    res["refit_weight_sensitivity_backward_top1"] = {}
    for (layer, u, v) in BACK:
        key = f"{layer}:L{u}->L{v}"
        transfers[key], transfer_rows[key] = transfer(W, refit_rows, refit_top1, layer, u, v)
        res["refit_weight_sensitivity_backward_top1"][key] = transfer(W_refit, rows_r, top_r, layer, u, v)[0][
            "target_macro_top1"]
        print(f"  backward {key}: {transfers[key]['target_macro_top1']:.4f}", flush=True)

    # bootstrap: original series from saved files (verified identical above) + backward series.
    series = S.load_pred_vectors()
    for (layer, u, v) in BACK:
        key = f"{layer}:L{u}->L{v}"
        s = {}
        for r in transfer_rows[key]:
            s.setdefault(r["domain"], {}).setdefault(r["task_uid"], []).append(float(r["top1_oracle"]))
        series[f"transfer:{key}"] = s
    points, samples = S.bootstrap(series, X.BOOT_DRAWS, X.BOOT_SEED)

    def paired(a, b):
        d = samples[a] - samples[b]
        return {"point": round(points[a] - points[b], 4), "ci95": S.ci(d),
                "p_gt_zero": round(float((d > 0).mean()), 4)}

    bs = json.loads((SRC / "bootstrap_stats.json").read_text())
    boot_mm = []
    for (layer, u, v) in X.TRANSFERS:
        k = f"transfer:{layer}:L{u}->L{v} - refit:{layer}_L{v}"
        got = paired(f"transfer:{layer}:L{u}->L{v}", f"refit:{layer}_L{v}")
        if got != bs["paired_deltas"][k]:
            boot_mm.append([k, got, bs["paired_deltas"][k]])
    res["bootstrap_reproduces_saved_forward_deltas"] = not boot_mm
    res["bootstrap_mismatches"] = boot_mm
    print(f"bootstrap forward deltas reproduce saved: {not boot_mm}", flush=True)
    if boot_mm:
        return dump(1)

    for (layer, u, v) in X.TRANSFERS + BACK:
        key = f"{layer}:L{u}->L{v}"
        transfers[key]["direction"] = "forward" if v > u else "backward"
        transfers[key]["paired_delta_vs_target_refit"] = paired(f"transfer:{key}", f"refit:{layer}_L{v}")
        transfers[key]["paired_delta_vs_source_self"] = paired(f"transfer:{key}", f"refit:{layer}_L{u}")
    res["transfers"] = transfers

    # ---------------------------------------------------------------- stage 3
    res["matrix_top1"], res["cosine"], res["weight_norm"] = {}, {}, {}
    for layer in LAYERS:
        m = [[refit_top1[X.cell_name(layer, u)] if u == v else
              transfers[f"{layer}:L{u}->L{v}"]["target_macro_top1_unrounded"] for v in LOOPS] for u in LOOPS]
        ws = [W[X.cell_name(layer, u)].flatten().double() for u in LOOPS]
        cos = [[round(float(a @ b / (a.norm() * b.norm())), 4) for b in ws] for a in ws]
        res["matrix_top1"][str(layer)] = {"rows_trained_on_loop": list(LOOPS),
                                          "cols_evaluated_at_loop": list(LOOPS), "values": m}
        res["cosine"][str(layer)] = cos
        res["weight_norm"][str(layer)] = [round(float(w.norm()), 4) for w in ws]
        print(f"layer {layer} top1 matrix (row=train loop, col=eval loop):", flush=True)
        for u, row in zip(LOOPS, m):
            print(f"   L{u}: " + "  ".join(f"{x:.4f}" for x in row), flush=True)
        print(f"layer {layer} cosine:", flush=True)
        for u, row in zip(LOOPS, cos):
            print(f"   L{u}: " + "  ".join(f"{x:+.4f}" for x in row), flush=True)

    print("\ntransfers (layer src->tgt: top1, delta vs target refit [95% CI], delta vs source self [95% CI], "
          "spearman, sign agree)")
    for (layer, u, v) in X.TRANSFERS + BACK:
        t = transfers[f"{layer}:L{u}->L{v}"]
        p, q = t["paired_delta_vs_target_refit"], t["paired_delta_vs_source_self"]
        print(f"  {t['direction'][:4]} {layer}:L{u}->L{v}: {t['target_macro_top1']:.4f}  "
              f"{p['point']:+.4f} {p['ci95']}  {q['point']:+.4f} {q['ci95']}  "
              f"rho={t['score_spearman_vs_target_refit']:.4f}  sign={t['pairwise_sign_agreement']:.4f}", flush=True)
    res["elapsed_seconds"] = round(time.time() - t0, 1)
    return dump(0)


if __name__ == "__main__":
    raise SystemExit(main())
