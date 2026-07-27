"""Frozen conversion #4 -- all-well-formed terminal-selection pool: analysis.

Sealed plan: artifacts/reports/wellformed_terminal_v4_20260727/EXPERIMENT_PLAN.md.
Verbatim v2 terminal-selection protocol (selector family, grouped-CV hyperparams on
train+val only, exact Poisson-binomial null, task-clustered bootstrap) on the
all-well-formed pool, plus the sealed secondary endpoint: paired hidden-vs-shortcut
top-1 difference on the same informative heldout groups.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANUAL = PROJECT_ROOT / "utilities/tests/manual"
for p in (str(PROJECT_ROOT), str(MANUAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

from bg_v2_overnight_horizon_analysis import select_hparams, fit_final, apply_final  # noqa: E402
from proto_introspection_controls_analysis import auroc  # noqa: E402
from bg_v2_overnight_terminal_selection import (  # noqa: E402
    poisson_binomial_sf, terminal_shortcut_vec, build_groups, informative,
)

OUT_ROOT = PROJECT_ROOT / "artifacts/reports/wellformed_terminal_v4_20260727"
SEED = 20260727
MIN_USEFUL = 25
ROUNDS = 2000


def main() -> None:
    pool_files = sorted(OUT_ROOT.glob("wellformed_pool*.pt"))
    records = []
    cohorts = {}
    for pf in pool_files:
        payload = torch.load(pf, map_location="cpu", weights_only=False)
        records.extend(payload["records"])
        cohorts[pf.name] = {"n_records": len(payload["records"]),
                            "task_offset": payload.get("task_offset")}
    print(f"[wf-sel] pooled cohorts: {cohorts}")
    assert all(r["found_final_marker"] and not r["malformed"] for r in records), \
        "sealed invariant: every candidate in this pool is well-formed"
    groups = build_groups(records)
    print(f"[wf-sel] {len(records)} candidates over {len(groups)} tasks")

    heldout = [tu for tu, g in groups.items() if g[0]["split"] == "heldout"]
    informative_heldout = [tu for tu in heldout if informative(groups[tu])]
    result: dict = {
        "seed": SEED, "n_tasks": len(groups), "n_heldout_groups": len(heldout),
        "n_informative_heldout_groups": len(informative_heldout),
        "cohorts": cohorts,
        "sealed_plan": "EXPERIMENT_PLAN.md (incl. SEALED EXTENSION 1)",
    }
    print(f"[wf-sel] informative heldout groups: {len(informative_heldout)}")
    if len(informative_heldout) < MIN_USEFUL:
        result["verdict"] = "WF_INFORMATIVE_YIELD_TOO_LOW"
        (OUT_ROOT / "wellformed_selection_results.json").write_text(
            json.dumps(result, indent=2, default=str) + "\n")
        print(f"[wf-sel] VERDICT: {result['verdict']}")
        return

    trainval = [r for r in records if r["split"] in ("train", "val")]
    y_tv = torch.tensor([1.0 if r["success"] else 0.0 for r in trainval])
    tasks_tv = [r["task_uid"] for r in trainval]
    Xhid_tv = torch.stack([r["full_features"].flatten().float() for r in trainval])
    Xsc_tv = torch.stack([terminal_shortcut_vec(r) for r in trainval])

    best_hid = select_hparams(Xhid_tv, y_tv, tasks_tv)
    model_hid = fit_final(Xhid_tv, y_tv, best_hid[1], best_hid[2])
    print(f"[wf-sel] hidden selector cv_auroc={best_hid[0]:.4f} k={best_hid[1]} l2={best_hid[2]}")
    best_sc = select_hparams(Xsc_tv, y_tv, tasks_tv)
    model_sc = fit_final(Xsc_tv, y_tv, best_sc[1], best_sc[2])
    print(f"[wf-sel] shortcut selector cv_auroc={best_sc[0]:.4f} k={best_sc[1]} l2={best_sc[2]}")

    def group_top1(tu: str, which: str) -> tuple[bool, float]:
        cands = groups[tu]
        if which == "hid":
            X = torch.stack([r["full_features"].flatten().float() for r in cands])
            scores = apply_final(model_hid, X)
        else:
            X = torch.stack([terminal_shortcut_vec(r) for r in cands])
            scores = apply_final(model_sc, X)
        succ = [bool(r["success"]) for r in cands]
        top1 = max(range(len(cands)), key=lambda i: float(scores[i]))
        return succ[top1], sum(succ) / len(cands)

    def evaluate(which: str):
        n = len(informative_heldout)
        wins, p_sum, per_p, per_win = 0, 0.0, [], []
        for tu in informative_heldout:
            w, p = group_top1(tu, which)
            wins += int(w)
            p_sum += p
            per_p.append(p)
            per_win.append(int(w))
        return {"n_groups": n, "observed_successes": wins,
                "observed_top1_rate": wins / n, "matched_random_rate": p_sum / n,
                "paired_diff": (wins - p_sum) / n}, per_p, per_win

    hid_eval, per_p, hid_win = evaluate("hid")
    hid_eval["exact_poisson_binomial_p_value_ge_observed"] = poisson_binomial_sf(
        per_p, hid_eval["observed_successes"])
    sc_eval, _, sc_win = evaluate("sc")
    per_group_detail = [
        {"task_uid": tu, "n_candidates": len(groups[tu]),
         "n_correct": sum(1 for r in groups[tu] if r["success"]),
         "matched_random_p": round(p, 4), "hidden_top1_correct": bool(hw),
         "shortcut_top1_correct": bool(sw)}
        for tu, p, hw, sw in zip(informative_heldout, per_p, hid_win, sc_win)
    ]
    print(f"[wf-sel] hidden: {hid_eval}")
    print(f"[wf-sel] shortcut: {sc_eval}")

    # task-clustered bootstraps: hidden-vs-random and hidden-vs-shortcut (paired)
    rng = torch.Generator().manual_seed(SEED)
    n_g = len(informative_heldout)
    d_rand, d_sc = [], []
    for _ in range(ROUNDS):
        pick = torch.randint(0, n_g, (n_g,), generator=rng).tolist()
        w_h = sum(hid_win[i] for i in pick)
        w_s = sum(sc_win[i] for i in pick)
        p_b = sum(per_p[i] for i in pick)
        d_rand.append((w_h - p_b) / n_g)
        d_sc.append((w_h - w_s) / n_g)

    def ci(xs):
        xs = sorted(xs)
        lo, hi = xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs))]
        return {"mean": sum(xs) / len(xs), "ci95": [lo, hi],
                "excludes_zero": bool(lo > 0 or hi < 0)}

    boot_rand, boot_sc = ci(d_rand), ci(d_sc)
    result.update({
        "hidden_selector": {"cv_auroc": best_hid[0], "k_pca": best_hid[1], "l2": best_hid[2]},
        "shortcut_selector": {"cv_auroc": best_sc[0], "k_pca": best_sc[1], "l2": best_sc[2]},
        "hidden_eval": hid_eval, "shortcut_eval": sc_eval,
        "hidden_vs_random_bootstrap": boot_rand,
        "hidden_vs_shortcut_bootstrap": boot_sc,
        "per_group_detail": per_group_detail,
    })
    print(f"[wf-sel] hidden-vs-random bootstrap: {boot_rand}")
    print(f"[wf-sel] hidden-vs-shortcut bootstrap: {boot_sc}")

    # candidate-level AUROC on all heldout candidates
    ho_records = [r for tu in heldout for r in groups[tu]]
    y_ho = torch.tensor([1.0 if r["success"] else 0.0 for r in ho_records])
    if 0.0 < float(y_ho.mean()) < 1.0:
        Xho = torch.stack([r["full_features"].flatten().float() for r in ho_records])
        result["candidate_level_auroc_all_heldout"] = auroc(apply_final(model_hid, Xho), y_ho)

    # controls: order invariance + duplicates (sealed, same as v2)
    controls = {"duplicate_candidate_count": 0, "candidate_order_invariance_failures": 0}
    seen = set()
    for tu in informative_heldout:
        for r in groups[tu]:
            key = (r["task_uid"], r["candidate_idx"])
            controls["duplicate_candidate_count"] += int(key in seen)
            seen.add(key)
    for tu in informative_heldout[:20]:
        cands = groups[tu]
        X = torch.stack([r["full_features"].flatten().float() for r in cands])
        Xr = torch.stack([r["full_features"].flatten().float() for r in reversed(cands)])
        f_top = int(torch.argmax(apply_final(model_hid, X)))
        r_top = len(cands) - 1 - int(torch.argmax(apply_final(model_hid, Xr)))
        controls["candidate_order_invariance_failures"] += int(f_top != r_top)
    tv_tasks = {r["task_uid"] for r in trainval}
    controls["zero_task_crossing"] = len(tv_tasks & set(informative_heldout)) == 0
    result["controls"] = controls

    exact_p = hid_eval["exact_poisson_binomial_p_value_ge_observed"]
    if (hid_eval["paired_diff"] > 0 and boot_rand["excludes_zero"] and exact_p < 0.05
            and controls["candidate_order_invariance_failures"] == 0
            and controls["duplicate_candidate_count"] == 0):
        verdict = "WF_CONTENT_SELECTION_ESTABLISHED"
    elif hid_eval["paired_diff"] > 0:
        verdict = "WF_CONTENT_SELECTION_ABOVE_RANDOM_UNDERPOWERED"
    else:
        verdict = "WF_CONTENT_SELECTION_NOT_ABOVE_RANDOM"
    if boot_sc["excludes_zero"]:
        flag = "HIDDEN_BEATS_SHORTCUT" if boot_sc["mean"] > 0 else "SHORTCUT_ABOVE_HIDDEN"
    else:
        flag = "HIDDEN_MATCHES_SHORTCUT"
    result["verdict"] = verdict
    result["hidden_vs_shortcut_flag"] = flag
    print(f"[wf-sel] VERDICT: {verdict} · {flag}")
    (OUT_ROOT / "wellformed_selection_results.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n")
    print(f"[wf-sel] wrote {OUT_ROOT / 'wellformed_selection_results.json'}")


if __name__ == "__main__":
    main()
