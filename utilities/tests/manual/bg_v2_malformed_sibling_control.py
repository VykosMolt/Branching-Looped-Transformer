"""Malformed-sibling shortcut control for the Horizon Logic pre-answer result.

Question this answers
---------------------
The Horizon Logic held-out increment (combined - shortcut) is computed on the
*scorable* (non-malformed) candidates only. A reviewer can reasonably ask whether
the hidden-state advantage is really "how much of this task went wrong", a
task-level property the four pre-answer shortcuts cannot see. This script hands
that property to the shortcut baseline directly, as a fifth shortcut feature:

    malformed_sibling_count(candidate) = #{siblings of this candidate that were malformed}

For a scorable candidate the count equals its task's malformed total (the
candidate itself is non-malformed by construction), so the feature is constant
within a task and ranges over 0..3 at k=4 candidates per task.

Note this deliberately makes the shortcut baseline STRONGER than a legitimately
pre-answer one: the count is derived from the siblings' completed generations,
which are not available before the candidate's own answer region. That is
conservative for the test -- it can only shrink the increment, never inflate it.

Protocol
--------
Identical to bg_v2_overnight_horizon_analysis.py: the same fit family
(standardize -> PCA -> L2-logistic), the same task-grouped CV, the same
hyperparameter grid, the same fixed heldout split, the same seed, and the same
2000-round paired task-clustered bootstrap. Because the seed and the held-out
task list are unchanged, the bootstrap resamples are identical replicate-for-
replicate across the baseline and augmented arms, so the two intervals are
directly comparable.

Stage A reproduces the published baseline and ABORTS unless it matches
auroc_results.json to 1e-12. Stage B is the augmented refit.

This script is read-only with respect to the original artifact root. It writes
only into its own new root.
"""
from __future__ import annotations

import glob
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
MANUAL = PROJECT_ROOT / "shared/utilities/tests/manual"
for p in (str(PROJECT_ROOT), str(MANUAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

from proto_introspection_controls_analysis import (  # noqa: E402
    standardize_fit, pca_fit, fit_logreg, auroc, acc_at_half,
)
from bg_v2_overnight_horizon_analysis import (  # noqa: E402
    HORIZON_ROOT, SEED, shortcut_vec, select_hparams, fit_final, apply_final,
    bootstrap_task_clustered,
)

OUT_ROOT = PROJECT_ROOT / "opi/preanswer/paper1_v2_malformed_sibling_control_20260725"
TOL = 1e-12


def load_all_records() -> list[dict]:
    """Read-only load from the ORIGINAL artifact root."""
    records = []
    for path in sorted(glob.glob(str(HORIZON_ROOT / "horizon_generation_*.pt"))):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        records.extend(payload["records"])
    return records


def malformed_count_by_task(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        counts.setdefault(r["task_uid"], 0)
        if r["malformed"]:
            counts[r["task_uid"]] += 1
    return counts


def shortcut_vec_aug(r: dict, mal_by_task: dict[str, int]) -> torch.Tensor:
    """Original four shortcuts plus the malformed-sibling count."""
    base = shortcut_vec(r)
    n_mal_sib = mal_by_task[r["task_uid"]] - (1 if r["malformed"] else 0)
    return torch.cat([base, torch.tensor([n_mal_sib / 3.0])])


def arm(name, Xhid_tv, Xsc_tv, y_tv, tasks_tv, Xhid_ho, Xsc_ho, y_ho, tasks_ho):
    """Run one full selection -> fit -> heldout -> bootstrap arm."""
    Xcomb_tv = torch.cat([Xhid_tv, Xsc_tv], dim=1)
    Xcomb_ho = torch.cat([Xhid_ho, Xsc_ho], dim=1)

    best_hid = select_hparams(Xhid_tv, y_tv, tasks_tv)
    best_sc = select_hparams(Xsc_tv, y_tv, tasks_tv)
    best_comb = select_hparams(Xcomb_tv, y_tv, tasks_tv)

    m_hid = fit_final(Xhid_tv, y_tv, best_hid[1], best_hid[2])
    m_sc = fit_final(Xsc_tv, y_tv, best_sc[1], best_sc[2])
    m_comb = fit_final(Xcomb_tv, y_tv, best_comb[1], best_comb[2])

    s_hid = apply_final(m_hid, Xhid_ho)
    s_sc = apply_final(m_sc, Xsc_ho)
    s_comb = apply_final(m_comb, Xcomb_ho)

    a_hid, a_sc, a_comb = auroc(s_hid, y_ho), auroc(s_sc, y_ho), auroc(s_comb, y_ho)
    inc = a_comb - a_sc
    boot = bootstrap_task_clustered(s_comb, s_sc, y_ho, tasks_ho, rounds=2000)

    print(f"[{name}] shortcut={a_sc:.6f} hidden={a_hid:.6f} combined={a_comb:.6f} "
          f"incremental={inc:+.6f}")
    print(f"[{name}] bootstrap mean={boot['mean_delta']:+.6f} "
          f"ci95=[{boot['ci95'][0]:+.6f}, {boot['ci95'][1]:+.6f}] "
          f"excludes_zero={boot['excludes_zero']}")

    return {
        "hyperparams": {
            "hidden": {"auroc_cv": best_hid[0], "k_pca": best_hid[1], "l2": best_hid[2]},
            "shortcut": {"auroc_cv": best_sc[0], "k_pca": best_sc[1], "l2": best_sc[2]},
            "combined": {"auroc_cv": best_comb[0], "k_pca": best_comb[1], "l2": best_comb[2]},
        },
        "shortcut_n_features": int(Xsc_tv.shape[1]),
        "auroc_shortcut_only": a_sc,
        "auroc_hidden_only": a_hid,
        "auroc_hidden_plus_shortcuts": a_comb,
        "incremental_auroc_combined_minus_shortcut": inc,
        "headroom_normalized_increment": inc / (1 - a_sc) if a_sc < 1.0 else float("nan"),
        "acc_at_half_combined": acc_at_half(s_comb, y_ho),
        "paired_task_clustered_bootstrap_combined_minus_shortcut": boot,
        "_scores": {"hidden": s_hid, "shortcut": s_sc, "combined": s_comb},
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    records = load_all_records()
    print(f"[control] loaded {len(records)} candidate records (read-only) from {HORIZON_ROOT}")

    mal_by_task = malformed_count_by_task(records)
    per_task_n = Counter(r["task_uid"] for r in records)
    assert set(per_task_n.values()) == {4}, f"non-uniform k per task: {set(per_task_n.values())}"

    scorable = [r for r in records if not r["malformed"]]
    y = torch.tensor([1.0 if r["success"] else 0.0 for r in scorable])
    task_uids = [r["task_uid"] for r in scorable]
    splits = [r["split"] for r in scorable]
    Xhid = torch.stack([r["preanswer_features"].flatten().float() for r in scorable])
    Xsc = torch.stack([shortcut_vec(r) for r in scorable])
    Xsc_aug = torch.stack([shortcut_vec_aug(r, mal_by_task) for r in scorable])

    # the fifth feature must be constant within a task and in 0..3 for scorables
    sib = (Xsc_aug[:, 4] * 3.0).round().long()
    assert int(sib.min()) >= 0 and int(sib.max()) <= 3, "sibling count out of range"
    by_task_vals: dict[str, set] = {}
    for t, v in zip(task_uids, sib.tolist()):
        by_task_vals.setdefault(t, set()).add(v)
    assert all(len(v) == 1 for v in by_task_vals.values()), "sibling count varies within a task"

    tv_mask = torch.tensor([s in ("train", "val") for s in splits])
    ho_mask = torch.tensor([s == "heldout" for s in splits])
    tasks_tv = [t for t, m in zip(task_uids, tv_mask.tolist()) if m]
    tasks_ho = [t for t, m in zip(task_uids, ho_mask.tolist()) if m]
    y_tv, y_ho = y[tv_mask], y[ho_mask]

    print(f"[control] scorable={len(scorable)} trainval={int(tv_mask.sum())} "
          f"heldout={int(ho_mask.sum())} heldout_tasks={len(set(tasks_ho))}")
    print(f"[control] heldout malformed-sibling-count histogram: "
          f"{dict(sorted(Counter(sib[ho_mask].tolist()).items()))}")

    # ---------------- Stage A: exact reproduction gate ----------------
    print("\n[control] STAGE A -- reproducing published baseline")
    base = arm("baseline", Xhid[tv_mask], Xsc[tv_mask], y_tv, tasks_tv,
               Xhid[ho_mask], Xsc[ho_mask], y_ho, tasks_ho)

    published = json.loads((HORIZON_ROOT / "auroc_results.json").read_text())
    ph = published["heldout"]
    pb = published["paired_task_clustered_bootstrap_combined_minus_shortcut"]
    checks = [
        ("auroc_shortcut_only", base["auroc_shortcut_only"], ph["auroc_shortcut_only"]),
        ("auroc_hidden_only", base["auroc_hidden_only"], ph["auroc_hidden_only"]),
        ("auroc_combined", base["auroc_hidden_plus_shortcuts"], ph["auroc_hidden_plus_shortcuts"]),
        ("incremental", base["incremental_auroc_combined_minus_shortcut"],
         ph["incremental_auroc_combined_minus_shortcut"]),
        ("boot_mean", base["paired_task_clustered_bootstrap_combined_minus_shortcut"]["mean_delta"],
         pb["mean_delta"]),
        ("boot_ci_lo", base["paired_task_clustered_bootstrap_combined_minus_shortcut"]["ci95"][0],
         pb["ci95"][0]),
        ("boot_ci_hi", base["paired_task_clustered_bootstrap_combined_minus_shortcut"]["ci95"][1],
         pb["ci95"][1]),
    ]
    repro = {}
    ok = True
    for label, got, want in checks:
        d = abs(got - want)
        repro[label] = {"reproduced": got, "published": want, "abs_diff": d, "match": d < TOL}
        flag = "OK " if d < TOL else "FAIL"
        print(f"  [{flag}] {label}: {got!r} vs published {want!r} (|d|={d:.3e})")
        ok = ok and d < TOL

    # also check the preserved per-candidate heldout scores
    raw = json.loads((HORIZON_ROOT / "raw_predictions_heldout.json").read_text())
    for key, tensor in (("scores_shortcut", base["_scores"]["shortcut"]),
                        ("scores_hidden", base["_scores"]["hidden"]),
                        ("scores_combined", base["_scores"]["combined"])):
        md = max(abs(a - b) for a, b in zip(raw[key], tensor.tolist()))
        repro[key + "_max_abs_diff"] = md
        flag = "OK " if md < TOL else "FAIL"
        print(f"  [{flag}] {key}: max |diff| = {md:.3e}")
        ok = ok and md < TOL
    assert raw["task_uids"] == tasks_ho, "heldout task order drifted"
    assert raw["y"] == y_ho.tolist(), "heldout labels drifted"

    if not ok:
        (OUT_ROOT / "REPRODUCTION_FAILED.json").write_text(
            json.dumps(repro, indent=2) + "\n")
        print("\n[control] BASELINE REPRODUCTION FAILED -- refusing to report a new number")
        sys.exit(1)
    print("[control] baseline reproduced exactly; proceeding")

    # ---------------- Stage B: augmented shortcut arm ----------------
    print("\n[control] STAGE B -- shortcuts + malformed-sibling count")
    aug = arm("augmented", Xhid[tv_mask], Xsc_aug[tv_mask], y_tv, tasks_tv,
              Xhid[ho_mask], Xsc_aug[ho_mask], y_ho, tasks_ho)

    # ---------------- diagnostics ----------------
    sib_ho = sib[ho_mask].double()
    # univariate: does the count predict success at all, in either direction?
    uni_pos = auroc(sib_ho, y_ho)
    diag = {
        "univariate_auroc_malformed_sibling_count": uni_pos,
        "univariate_auroc_negated": auroc(-sib_ho, y_ho),
        "heldout_success_rate_by_sibling_count": {
            str(c): {
                "n": int((sib_ho == c).sum()),
                "success_rate": float(y_ho[sib_ho == c].mean()) if int((sib_ho == c).sum()) else None,
            } for c in sorted(set(sib_ho.tolist()))
        },
    }
    print(f"[diag] univariate AUROC of sibling count = {uni_pos:.6f} "
          f"(negated {diag['univariate_auroc_negated']:.6f})")
    for c, v in diag["heldout_success_rate_by_sibling_count"].items():
        print(f"[diag]   count={c}: n={v['n']} success_rate={v['success_rate']}")

    # cross-arm: hidden still beats the STRONGER shortcut baseline?
    cross = bootstrap_task_clustered(
        base["_scores"]["combined"], aug["_scores"]["shortcut"], y_ho, tasks_ho, rounds=2000)
    cross_inc = base["auroc_hidden_plus_shortcuts"] - aug["auroc_shortcut_only"]
    print(f"[cross] baseline combined - augmented shortcut = {cross_inc:+.6f} "
          f"ci95=[{cross['ci95'][0]:+.6f}, {cross['ci95'][1]:+.6f}] "
          f"excludes_zero={cross['excludes_zero']}")

    # permutation sanity: shuffle the feature ACROSS TASKS -- if the augmented
    # shortcut gain is real, it should mostly vanish here. Guards against a
    # coding error that merely adds a fifth noise column.
    g = torch.Generator().manual_seed(SEED)
    uniq_tasks = sorted(set(task_uids))
    perm = torch.randperm(len(uniq_tasks), generator=g).tolist()
    permuted_val = {uniq_tasks[i]: (mal_by_task[uniq_tasks[perm[i]]]) for i in range(len(uniq_tasks))}
    Xsc_perm = torch.stack([
        torch.cat([shortcut_vec(r), torch.tensor([min(permuted_val[r["task_uid"]], 3) / 3.0])])
        for r in scorable])
    perm_arm = arm("permuted", Xhid[tv_mask], Xsc_perm[tv_mask], y_tv, tasks_tv,
                   Xhid[ho_mask], Xsc_perm[ho_mask], y_ho, tasks_ho)

    for a in (base, aug, perm_arm):
        a.pop("_scores", None)

    result = {
        "control": "malformed_sibling_count_added_to_shortcut_baseline",
        "question": ("does the Horizon Logic held-out pre-answer increment survive when the "
                     "shortcut baseline is given the per-task malformed-sibling count?"),
        "source_artifact_root": str(HORIZON_ROOT),
        "source_root_written_to": False,
        "seed": SEED,
        "protocol": ("identical fit family, grid, grouped-CV, fixed heldout split, seed and "
                     "2000-round paired task-clustered bootstrap as "
                     "bg_v2_overnight_horizon_analysis.py; bootstrap replicates are identical "
                     "across arms so the intervals are directly comparable"),
        "feature_definition": ("malformed_sibling_count / 3.0, where the count is the number of "
                              "the candidate's 3 siblings that were malformed; constant within a "
                              "task for scorable candidates; range 0..3"),
        "conservativeness_note": ("the feature is derived from siblings' completed generations and "
                                  "is therefore NOT available pre-answer; it can only strengthen "
                                  "the shortcut baseline and shrink the increment"),
        "n_scorable": len(scorable),
        "n_heldout": int(ho_mask.sum()),
        "n_heldout_tasks": len(set(tasks_ho)),
        "heldout_sibling_count_histogram": {str(k): v for k, v in
                                            sorted(Counter(sib[ho_mask].tolist()).items())},
        "stage_a_baseline_reproduction": repro,
        "baseline_arm": base,
        "augmented_arm": aug,
        "permuted_feature_arm": perm_arm,
        "diagnostics": diag,
        "cross_arm_baseline_combined_minus_augmented_shortcut": {
            "incremental": cross_inc, "bootstrap": cross,
        },
    }

    delta_inc = (aug["incremental_auroc_combined_minus_shortcut"]
                 - base["incremental_auroc_combined_minus_shortcut"])
    result["increment_change_vs_baseline"] = delta_inc
    if aug["paired_task_clustered_bootstrap_combined_minus_shortcut"]["excludes_zero"] \
       and aug["incremental_auroc_combined_minus_shortcut"] > 0:
        verdict = "INCREMENT_SURVIVES_MALFORMED_SIBLING_SHORTCUT"
    elif aug["incremental_auroc_combined_minus_shortcut"] > 0:
        verdict = "INCREMENT_POSITIVE_BUT_INTERVAL_NOW_INCLUDES_ZERO"
    else:
        verdict = "INCREMENT_ELIMINATED_BY_MALFORMED_SIBLING_SHORTCUT"
    result["verdict"] = verdict
    print(f"\n[control] increment {base['incremental_auroc_combined_minus_shortcut']:+.6f} "
          f"-> {aug['incremental_auroc_combined_minus_shortcut']:+.6f} "
          f"(change {delta_inc:+.6f})")
    print(f"[control] VERDICT: {verdict}")

    out = OUT_ROOT / "malformed_sibling_control.json"
    out.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"[control] wrote {out}")


if __name__ == "__main__":
    main()
