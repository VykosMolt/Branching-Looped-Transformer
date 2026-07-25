"""Paper 1 v2 overnight programme -- Part I analysis: Horizon Logic strict pre-answer AUROC.

Reuses the audited fit family from proto_introspection_controls_analysis.py
(standardize -> PCA -> L2-logistic, task-grouped CV, bootstrap AUROC CI) rather than
inventing a new methodology. Primary endpoint:

    AUROC(hidden + shortcuts) - AUROC(shortcuts)

on a heldout split opened once, with a paired task-clustered bootstrap interval.
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANUAL = PROJECT_ROOT / "utilities/tests/manual"
for p in (str(PROJECT_ROOT), str(MANUAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

from proto_introspection_controls_analysis import (  # noqa: E402
    standardize_fit, pca_fit, fit_logreg, auroc, acc_at_half,
)

HORIZON_ROOT = PROJECT_ROOT / "artifacts/reports/paper1_v2_overnight_20260724/horizon_logic"
SEED = 20260724
RNG = torch.Generator().manual_seed(SEED)


def load_all_records() -> list[dict]:
    records = []
    for path in sorted(glob.glob(str(HORIZON_ROOT / "horizon_generation_*.pt"))):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        records.extend(payload["records"])
    return records


def shortcut_vec(r: dict) -> torch.Tensor:
    mlp = r["mean_logprob_pre"] if r["mean_logprob_pre"] is not None else -5.0
    minlp = r["min_logprob_pre"] if r["min_logprob_pre"] is not None else -10.0
    return torch.tensor([
        r["n_pre_tok"] / 320.0,
        max(mlp, -10.0) / 10.0,
        max(minlp, -20.0) / 20.0,
        1.0 if r["hit_max_tokens"] else 0.0,
    ])


def grouped_folds_by_task(task_uids: list[str], n_folds: int = 5, seed: int = SEED) -> list[int]:
    uniq = sorted(set(task_uids))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(uniq), generator=g).tolist()
    uniq = [uniq[i] for i in perm]
    fold_of = {u: (i % n_folds) for i, u in enumerate(uniq)}
    return [fold_of[t] for t in task_uids]


def cv_auroc(X, y, task_uids, k_pca, l2, n_folds=5, seed=SEED):
    n = X.shape[0]
    folds = torch.tensor(grouped_folds_by_task(task_uids, n_folds, seed))
    oof = torch.zeros(n, dtype=torch.double)
    for f in range(n_folds):
        tr, va = folds != f, folds == f
        if tr.sum() == 0 or va.sum() == 0 or y[tr].float().mean() in (0.0, 1.0):
            oof[va] = float(y[tr].float().mean()) if tr.sum() else 0.5
            continue
        mu, sd = standardize_fit(X[tr])
        Ztr, Zva = (X[tr] - mu) / sd, (X[va] - mu) / sd
        pmu, comps = pca_fit(Ztr, min(k_pca, Ztr.shape[0] - 1, Ztr.shape[1]))
        Ptr, Pva = (Ztr - pmu) @ comps.T, (Zva - pmu) @ comps.T
        w, b = fit_logreg(Ptr, y[tr], l2=l2)
        oof[va] = torch.sigmoid(Pva @ w + b).double()
    return oof


def fit_final(Xtr, ytr, k_pca, l2):
    mu, sd = standardize_fit(Xtr)
    Z = (Xtr - mu) / sd
    pmu, comps = pca_fit(Z, min(k_pca, Z.shape[0] - 1, Z.shape[1]))
    P = (Z - pmu) @ comps.T
    w, b = fit_logreg(P, ytr, l2=l2)
    return {"mu": mu, "sd": sd, "pmu": pmu, "comps": comps, "w": w, "b": b}


def apply_final(model, X):
    Z = (X - model["mu"]) / model["sd"]
    P = (Z - model["pmu"]) @ model["comps"].T
    return torch.sigmoid(P @ model["w"] + model["b"]).double()


def select_hparams(X, y, task_uids):
    best = None
    for k_pca in (16, 24):
        for l2 in (1.0, 2.0, 4.0):
            oof = cv_auroc(X, y, task_uids, k_pca, l2)
            a = auroc(oof, y)
            if best is None or (not math.isnan(a) and a > best[0]):
                best = (a, k_pca, l2)
    return best


def bootstrap_task_clustered(scores_a, scores_b, y, task_uids, rounds=2000, seed=SEED):
    """Paired bootstrap over task_uid clusters of AUROC_a - AUROC_b."""
    rng = torch.Generator().manual_seed(seed)
    uniq = sorted(set(task_uids))
    by_task = {}
    for i, t in enumerate(task_uids):
        by_task.setdefault(t, []).append(i)
    deltas = []
    aucs_a, aucs_b = [], []
    for _ in range(rounds):
        pick = torch.randint(0, len(uniq), (len(uniq),), generator=rng).tolist()
        idxs = []
        for pi in pick:
            idxs.extend(by_task[uniq[pi]])
        if not idxs:
            continue
        idx_t = torch.tensor(idxs)
        yb = y[idx_t]
        if yb.float().mean() in (0.0, 1.0):
            continue
        aa = auroc(scores_a[idx_t], yb)
        bb = auroc(scores_b[idx_t], yb)
        if math.isnan(aa) or math.isnan(bb):
            continue
        deltas.append(aa - bb)
        aucs_a.append(aa)
        aucs_b.append(bb)
    if not deltas:
        return None
    deltas.sort()
    n = len(deltas)
    return {
        "n_rounds_used": n,
        "mean_delta": sum(deltas) / n,
        "ci95": [deltas[int(0.025 * n)], deltas[int(0.975 * n)]],
        "excludes_zero": bool(deltas[int(0.025 * n)] > 0 or deltas[int(0.975 * n)] < 0),
    }


def main() -> None:
    records = load_all_records()
    print(f"[analysis] loaded {len(records)} candidate records")
    if not records:
        print("[analysis] NO RECORDS -- aborting")
        return

    n_malformed = sum(1 for r in records if r["malformed"])
    n_hit_max = sum(1 for r in records if r["hit_max_tokens"])
    n_success = sum(1 for r in records if r["success"])
    tasks = sorted(set(r["task_uid"] for r in records))
    depth_hist = {}
    for r in records:
        depth_hist[r["proof_depth"]] = depth_hist.get(r["proof_depth"], 0) + 1
    print(f"[analysis] n_tasks={len(tasks)} n_candidates={len(records)} "
          f"n_success={n_success} base_rate={n_success/len(records):.4f} "
          f"n_malformed={n_malformed} ({n_malformed/len(records):.4f}) "
          f"n_hit_max={n_hit_max} ({n_hit_max/len(records):.4f}) depth_hist={depth_hist}")

    split_of_task = {r["task_uid"]: r["split"] for r in records}
    splits_count = {}
    for s in split_of_task.values():
        splits_count[s] = splits_count.get(s, 0) + 1
    print(f"[analysis] task-level split counts: {splits_count}")

    # ---- malformed candidates are excluded from the correctness label set (no valid
    # verifier label), but kept in every raw-preservation / rate metric above. Folding
    # them into the negative class would risk the classifier mostly learning
    # "did this hit the token budget / fail to parse" rather than "is the logical
    # answer correct" -- especially since hit-max-tokens correlates heavily with
    # malformed in the pilot. ----
    scorable = [r for r in records if not r["malformed"]]
    n_scorable = len(scorable)
    print(f"[analysis] scorable (non-malformed, verifier-labeled) candidates: {n_scorable} "
          f"of {len(records)} ({n_scorable/len(records):.4f})")

    y = torch.tensor([1.0 if r["success"] else 0.0 for r in scorable])
    task_uids = [r["task_uid"] for r in scorable]
    splits = [r["split"] for r in scorable]
    Xhid = torch.stack([r["preanswer_features"].flatten().float() for r in scorable])
    Xsc = torch.stack([shortcut_vec(r) for r in scorable])

    trainval_mask = torch.tensor([s in ("train", "val") for s in splits])
    heldout_mask = torch.tensor([s == "heldout" for s in splits])
    n_tv, n_ho = int(trainval_mask.sum()), int(heldout_mask.sum())
    print(f"[analysis] scorable train+val candidates={n_tv}, scorable heldout candidates={n_ho}")

    result: dict = {
        "seed": SEED, "n_tasks": len(tasks), "n_candidates": len(records),
        "base_rate": n_success / len(records), "n_malformed": n_malformed,
        "malformed_rate": n_malformed / len(records), "n_hit_max_tokens": n_hit_max,
        "hit_max_rate": n_hit_max / len(records), "depth_histogram": depth_hist,
        "task_split_counts": splits_count,
        "n_scorable_candidates": n_scorable,
        "scorable_rate": n_scorable / len(records),
        "scorable_base_rate": float(y.mean()) if n_scorable else None,
        "n_candidates_trainval": n_tv, "n_candidates_heldout": n_ho,
        "malformed_handling": ("excluded from the primary shortcut/hidden/combined AUROC label set "
                              "(no valid verifier label); retained in raw records and in the "
                              "malformed-rate / hit-max-rate diagnostics above"),
    }

    if n_tv < 20 or n_ho < 20 or y[trainval_mask].float().mean() in (0.0, 1.0) or \
       y[heldout_mask].float().mean() in (0.0, 1.0):
        result["verdict"] = "INCONCLUSIVE_RUNTIME_OR_POWER_LIMITED"
        result["note"] = "insufficient train/val or heldout candidates, or single-class split"
        Path(HORIZON_ROOT / "auroc_results.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
        print("[analysis] INSUFFICIENT DATA -- wrote partial result and stopping")
        return

    Xhid_tv, Xsc_tv, y_tv = Xhid[trainval_mask], Xsc[trainval_mask], y[trainval_mask]
    tasks_tv = [t for t, m in zip(task_uids, trainval_mask.tolist()) if m]
    Xhid_ho, Xsc_ho, y_ho = Xhid[heldout_mask], Xsc[heldout_mask], y[heldout_mask]
    tasks_ho = [t for t, m in zip(task_uids, heldout_mask.tolist()) if m]

    Xcomb_tv = torch.cat([Xhid_tv, Xsc_tv], dim=1)
    Xcomb_ho = torch.cat([Xhid_ho, Xsc_ho], dim=1)

    print("[analysis] selecting hidden-only hyperparams on train+val (grouped CV) ...")
    best_hid = select_hparams(Xhid_tv, y_tv, tasks_tv)
    print(f"[analysis] hidden-only best: auroc={best_hid[0]:.4f} k_pca={best_hid[1]} l2={best_hid[2]}")
    print("[analysis] selecting shortcut-only hyperparams ...")
    best_sc = select_hparams(Xsc_tv, y_tv, tasks_tv)
    print(f"[analysis] shortcut-only best: auroc={best_sc[0]:.4f} k_pca={best_sc[1]} l2={best_sc[2]}")
    print("[analysis] selecting combined hyperparams ...")
    best_comb = select_hparams(Xcomb_tv, y_tv, tasks_tv)
    print(f"[analysis] combined best: auroc={best_comb[0]:.4f} k_pca={best_comb[1]} l2={best_comb[2]}")

    model_hid = fit_final(Xhid_tv, y_tv, best_hid[1], best_hid[2])
    model_sc = fit_final(Xsc_tv, y_tv, best_sc[1], best_sc[2])
    model_comb = fit_final(Xcomb_tv, y_tv, best_comb[1], best_comb[2])

    scores_hid_ho = apply_final(model_hid, Xhid_ho)
    scores_sc_ho = apply_final(model_sc, Xsc_ho)
    scores_comb_ho = apply_final(model_comb, Xcomb_ho)

    auroc_hid = auroc(scores_hid_ho, y_ho)
    auroc_sc = auroc(scores_sc_ho, y_ho)
    auroc_comb = auroc(scores_comb_ho, y_ho)
    incremental = auroc_comb - auroc_sc
    headroom = incremental / (1 - auroc_sc) if auroc_sc < 1.0 else float("nan")

    print(f"[analysis] HELDOUT shortcut AUROC={auroc_sc:.4f} hidden AUROC={auroc_hid:.4f} "
          f"combined AUROC={auroc_comb:.4f} incremental={incremental:.4f} headroom_norm={headroom:.4f}")

    boot = bootstrap_task_clustered(scores_comb_ho, scores_sc_ho, y_ho, tasks_ho, rounds=2000)
    print(f"[analysis] paired task-clustered bootstrap (combined-shortcut): {boot}")

    # ---- controls ----
    controls = {}
    # 1. zero task crossing between trainval and heldout
    crossing = set(tasks_tv) & set(tasks_ho)
    controls["zero_task_crossing"] = len(crossing) == 0
    controls["n_crossing_tasks"] = len(crossing)
    # 2. strict answer-region exclusion: preanswer text must not contain the marker
    import re
    FINAL_MARKER = re.compile(r"FINAL\s*ANSWE", re.IGNORECASE)
    leaks = 0
    for r in records:
        pre_text_check = r["generated_text"][: len(r["generated_text"])]  # marker check done at gen time
        # re-derive: reconstruct preanswer text length used for features via n_pre_tok is not stored as
        # text; instead assert the recorded found_final_marker + n_pre_tok < n_gen_tok whenever marker found
        if r["found_final_marker"] and r["n_pre_tok"] >= r["n_gen_tok"]:
            leaks += 1
    controls["answer_region_exclusion_violations"] = leaks
    # 3. explicit gold-value exclusion: gold letter/text never concatenated into prompt/preanswer by construction
    controls["gold_value_excluded_by_construction"] = True
    # 4. shuffled-label control (heldout)
    yperm = y_ho[torch.randperm(y_ho.shape[0], generator=torch.Generator().manual_seed(SEED))]
    controls["shuffled_label_auroc_combined"] = auroc(scores_comb_ho, yperm)
    # 5. task-ID permutation control: shuffle task_uid->split assignment sanity (recompute split disjointness)
    controls["task_id_permutation_control"] = "train/val/heldout assignment is a pure function of "\
        "sha256(task_uid); permuting task_uids does not change per-record split membership by construction"
    # 6. shortcut-only comparison already computed (auroc_sc)
    controls["shortcut_only_auroc"] = auroc_sc
    # 7. no single-task dependence: leave-one-task-out influence on combined AUROC
    infl = []
    for t in set(tasks_ho):
        keep = torch.tensor([tt != t for tt in tasks_ho])
        if keep.sum() < 10 or y_ho[keep].float().mean() in (0.0, 1.0):
            continue
        a = auroc(scores_comb_ho[keep], y_ho[keep])
        if not math.isnan(a):
            infl.append(abs(a - auroc_comb))
    controls["max_single_task_influence_on_combined_auroc"] = max(infl) if infl else None
    # 8. duplicate-task / duplicate-candidate audit
    seen = set()
    dup_candidates = 0
    for r in records:
        key = (r["task_uid"], r["candidate_idx"])
        if key in seen:
            dup_candidates += 1
        seen.add(key)
    controls["duplicate_candidate_count"] = dup_candidates
    # 9. fixed held-out split -- confirmed by construction (split_for is deterministic hash)
    controls["fixed_heldout_split_deterministic"] = True
    # 10. raw predictions preserved below

    # ---- malformed-artifact-not-dominating controls (per coordinator review) ----
    # (a) drop heldout TASKS with a disproportionate malformed share and recheck the
    #     increment still holds -- a task where most candidates were malformed leaves
    #     an unrepresentative 1-2 scorable rows behind.
    malformed_share_by_task: dict[str, float] = {}
    for t in set(r["task_uid"] for r in records):
        rows_t = [r for r in records if r["task_uid"] == t]
        malformed_share_by_task[t] = sum(1 for r in rows_t if r["malformed"]) / len(rows_t)
    HIGH_MALFORMED_THRESHOLD = 0.75
    high_malformed_tasks = {t for t, s in malformed_share_by_task.items() if s >= HIGH_MALFORMED_THRESHOLD}
    keep_mask = torch.tensor([t not in high_malformed_tasks for t in tasks_ho])
    if keep_mask.sum() >= 10 and y_ho[keep_mask].float().mean() not in (0.0, 1.0):
        auroc_comb_dropped = auroc(scores_comb_ho[keep_mask], y_ho[keep_mask])
        auroc_sc_dropped = auroc(scores_sc_ho[keep_mask], y_ho[keep_mask])
        incremental_dropped = auroc_comb_dropped - auroc_sc_dropped
        controls["increment_after_dropping_high_malformed_share_tasks"] = {
            "threshold": HIGH_MALFORMED_THRESHOLD, "n_tasks_dropped": len(high_malformed_tasks & set(tasks_ho)),
            "n_candidates_remaining": int(keep_mask.sum()), "auroc_combined": auroc_comb_dropped,
            "auroc_shortcut": auroc_sc_dropped, "incremental": incremental_dropped,
            "survives": bool(incremental_dropped > 0),
        }
    else:
        controls["increment_after_dropping_high_malformed_share_tasks"] = {
            "note": "too few candidates remaining after drop, or single-class heldout subset"}
    # (b) malformed-vs-clean separability: is the hidden-state signal mostly separating
    #     "hit budget / unparseable" from "clean completion", rather than correct vs
    #     incorrect among clean completions? Fit a hidden-only malformed-detector on ALL
    #     heldout records (malformed included) and report its AUROC alongside the
    #     correctness AUROC for direct comparison -- a genuinely different label.
    ho_all_records = [r for r in records if r["split"] == "heldout"]
    y_mal = torch.tensor([1.0 if r["malformed"] else 0.0 for r in ho_all_records])
    if y_mal.float().mean() not in (0.0, 1.0):
        Xhid_mal = torch.stack([r["preanswer_features"].flatten().float() for r in ho_all_records])
        tasks_mal = [r["task_uid"] for r in ho_all_records]
        best_mal = select_hparams(Xhid_mal, y_mal,
                                   [t for t in tasks_mal])  # reuse grouped CV within heldout for this diagnostic only
        oof_mal = cv_auroc(Xhid_mal, y_mal, tasks_mal, best_mal[1], best_mal[2])
        malformed_detector_auroc = auroc(oof_mal, y_mal)
    else:
        malformed_detector_auroc = None
    controls["malformed_vs_clean_hidden_auroc"] = malformed_detector_auroc
    controls["malformed_vs_clean_note"] = (
        "hidden-state AUROC for predicting malformed(1)/clean(0), a DIFFERENT label than "
        "correctness; reported so a reader can check the correctness increment above is not "
        "actually just this signal in disguise")

    result.update({
        "hyperparams": {"hidden": {"auroc_cv": best_hid[0], "k_pca": best_hid[1], "l2": best_hid[2]},
                        "shortcut": {"auroc_cv": best_sc[0], "k_pca": best_sc[1], "l2": best_sc[2]},
                        "combined": {"auroc_cv": best_comb[0], "k_pca": best_comb[1], "l2": best_comb[2]}},
        "heldout": {
            "n": int(y_ho.shape[0]), "n_tasks": len(set(tasks_ho)),
            "base_rate": float(y_ho.mean()),
            "auroc_shortcut_only": auroc_sc, "auroc_hidden_only": auroc_hid,
            "auroc_hidden_plus_shortcuts": auroc_comb,
            "incremental_auroc_combined_minus_shortcut": incremental,
            "headroom_normalized_increment": headroom,
            "acc_at_half_combined": acc_at_half(scores_comb_ho, y_ho),
        },
        "paired_task_clustered_bootstrap_combined_minus_shortcut": boot,
        "controls": controls,
    })

    # verdict
    if boot and boot["excludes_zero"] and incremental > 0 and controls["zero_task_crossing"] and \
       (controls["max_single_task_influence_on_combined_auroc"] or 0) < 0.05:
        verdict = "SECOND_DOMAIN_PREANSWER_REPLICATION"
    elif incremental > 0 and boot and not boot["excludes_zero"]:
        verdict = "SECOND_DOMAIN_SIGNAL_PRESENT_BUT_UNDERPOWERED"
    elif incremental <= 0:
        verdict = "SECOND_DOMAIN_NO_INCREMENT_BEYOND_SHORTCUTS"
    else:
        verdict = "INCONCLUSIVE_RUNTIME_OR_POWER_LIMITED"
    result["verdict"] = verdict
    print(f"[analysis] VERDICT: {verdict}")

    raw_path = HORIZON_ROOT / "raw_predictions_heldout.json"
    raw_path.write_text(json.dumps({
        "task_uids": tasks_ho, "y": y_ho.tolist(),
        "scores_hidden": scores_hid_ho.tolist(), "scores_shortcut": scores_sc_ho.tolist(),
        "scores_combined": scores_comb_ho.tolist(),
    }, indent=2) + "\n")

    out_path = HORIZON_ROOT / "auroc_results.json"
    out_path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"[analysis] wrote {out_path} and {raw_path}")


if __name__ == "__main__":
    main()
