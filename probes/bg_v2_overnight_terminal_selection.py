"""Paper 1 v2 overnight programme -- Part II: powered terminal-selection evaluation.

Uses the SAME Horizon Logic candidate pool as Part I (shared-pool design), scored on the
completed-candidate ("full") pooled hidden feature rather than the pre-answer cut, since
terminal selection is forced choice among already-generated candidates (no leakage
constraint applies here -- the candidates are complete).

Primary endpoint: forced top-1 success on informative held-out groups vs. the EXACT
matched-random baseline (Poisson-binomial null over per-group p_i = n_correct_i / n_i).
"""
from __future__ import annotations

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

from proto_introspection_controls_analysis import standardize_fit, pca_fit, fit_logreg, auroc  # noqa: E402
from bg_v2_overnight_horizon_analysis import (  # noqa: E402
    grouped_folds_by_task, select_hparams, fit_final, apply_final,
)

TERMINAL_ROOT = PROJECT_ROOT / "artifacts/reports/paper1_v2_overnight_20260724/terminal_selection"
SEED = 20260724


def poisson_binomial_sf(ps: list[float], k: int) -> float:
    """P(X >= k) for X = sum of independent Bernoulli(p_i), exact DP, O(n^2)."""
    dp = [1.0] + [0.0] * len(ps)
    for p in ps:
        new = [0.0] * len(dp)
        for j, dpj in enumerate(dp):
            if dpj == 0.0:
                continue
            new[j] += dpj * (1 - p)
            if j + 1 < len(new):
                new[j + 1] += dpj * p
        dp = new
    return float(sum(dp[k:]))


def terminal_shortcut_vec(r: dict) -> torch.Tensor:
    return torch.tensor([
        r["n_gen_tok"] / 320.0,
        1.0 if r["found_final_marker"] else 0.0,
        1.0 if r["hit_max_tokens"] else 0.0,
        (r["n_pre_tok"] / r["n_gen_tok"]) if r["n_gen_tok"] else 0.0,
    ])


def build_groups(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r["task_uid"], []).append(r)
    return groups


def informative(group: list[dict]) -> bool:
    if len(group) < 2:
        return False
    succ = [bool(r["success"]) for r in group]
    return any(succ) and not all(succ)


def main() -> None:
    pool_path = TERMINAL_ROOT / "terminal_pool.pt"
    payload = torch.load(pool_path, map_location="cpu", weights_only=False)
    records = payload["records"]
    print(f"[terminal] loaded {len(records)} candidate records ({len(payload.get('errors', []))} prep errors)")

    groups = build_groups(records)
    print(f"[terminal] n_source_tasks={len(groups)}")

    by_split_groups: dict[str, list[str]] = {}
    for tu, g in groups.items():
        split = g[0]["split"]
        by_split_groups.setdefault(split, []).append(tu)
    print(f"[terminal] groups by split: { {k: len(v) for k, v in by_split_groups.items()} }")

    heldout_tasks = by_split_groups.get("heldout", [])
    informative_heldout = [tu for tu in heldout_tasks if informative(groups[tu])]
    all_correct_heldout = [tu for tu in heldout_tasks if all(bool(r["success"]) for r in groups[tu]) and len(groups[tu]) >= 2]
    all_wrong_heldout = [tu for tu in heldout_tasks if not any(bool(r["success"]) for r in groups[tu]) and len(groups[tu]) >= 2]

    result: dict = {
        "seed": SEED, "n_source_tasks": len(groups),
        "groups_by_split": {k: len(v) for k, v in by_split_groups.items()},
        "n_informative_heldout_groups": len(informative_heldout),
        "n_all_correct_heldout_groups": len(all_correct_heldout),
        "n_all_wrong_heldout_groups": len(all_wrong_heldout),
    }
    print(f"[terminal] informative heldout groups: {len(informative_heldout)} "
          f"(all-correct={len(all_correct_heldout)}, all-wrong={len(all_wrong_heldout)})")

    MIN_USEFUL = 25
    if len(informative_heldout) < MIN_USEFUL:
        result["verdict"] = "INFORMATIVE_GROUP_YIELD_TOO_LOW"
        result["note"] = (f"only {len(informative_heldout)} informative heldout groups; "
                          f"minimum useful target is {MIN_USEFUL}")
        out_path = TERMINAL_ROOT / "terminal_results.json"
        out_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
        print(f"[terminal] VERDICT: {result['verdict']} -- wrote {out_path}")
        return

    # -------------------------------------------------- train primary selector (train+val only)
    trainval_records = [r for r in records if r["split"] in ("train", "val")]
    y_tv = torch.tensor([1.0 if r["success"] else 0.0 for r in trainval_records])
    tasks_tv = [r["task_uid"] for r in trainval_records]
    Xhid_tv = torch.stack([r["full_features"].flatten().float() for r in trainval_records])
    Xsc_tv = torch.stack([terminal_shortcut_vec(r) for r in trainval_records])

    print("[terminal] selecting primary (hidden) selector hyperparams on train+val ...")
    best_hid = select_hparams(Xhid_tv, y_tv, tasks_tv)
    model_hid = fit_final(Xhid_tv, y_tv, best_hid[1], best_hid[2])
    print(f"[terminal] primary selector: cv_auroc={best_hid[0]:.4f} k_pca={best_hid[1]} l2={best_hid[2]}")

    print("[terminal] selecting shortcut-only selector hyperparams ...")
    best_sc = select_hparams(Xsc_tv, y_tv, tasks_tv)
    model_sc = fit_final(Xsc_tv, y_tv, best_sc[1], best_sc[2])
    print(f"[terminal] shortcut selector: cv_auroc={best_sc[0]:.4f} k_pca={best_sc[1]} l2={best_sc[2]}")

    def score_group(task_uid, model, feat_key, shortcut=False):
        cands = groups[task_uid]
        if shortcut:
            X = torch.stack([terminal_shortcut_vec(r) for r in cands])
        else:
            X = torch.stack([r[feat_key].flatten().float() for r in cands])
        scores = apply_final(model, X)
        return cands, scores

    def evaluate_selector(task_list, model, feat_key="full_features", shortcut=False, seed_tag="primary"):
        successes = 0
        p_sum = 0.0
        n_groups = len(task_list)
        pairwise_win = pairwise_tie = pairwise_tot = 0
        mrr_sum = 0.0
        top2_hit = top2_n = 0
        top3_hit = top3_n = 0
        per_group_p = []
        for tu in task_list:
            cands, scores = score_group(tu, model, feat_key, shortcut)
            succ = [bool(r["success"]) for r in cands]
            n = len(cands)
            p_i = sum(succ) / n
            per_group_p.append(p_i)
            p_sum += p_i
            order = sorted(range(n), key=lambda i: -float(scores[i]))
            top1 = order[0]
            if succ[top1]:
                successes += 1
            # pairwise ranking accuracy among (correct, incorrect) pairs
            for i in range(n):
                for j in range(n):
                    if succ[i] and not succ[j]:
                        pairwise_tot += 1
                        if scores[i] > scores[j]:
                            pairwise_win += 1
                        elif scores[i] == scores[j]:
                            pairwise_tie += 1
            # MRR: rank (1-indexed) of first correct candidate in score order
            for rank, idx in enumerate(order, start=1):
                if succ[idx]:
                    mrr_sum += 1.0 / rank
                    break
            if n >= 2:
                top2_n += 1
                if any(succ[idx] for idx in order[:2]):
                    top2_hit += 1
            if n >= 3:
                top3_n += 1
                if any(succ[idx] for idx in order[:3]):
                    top3_hit += 1
        pairwise_acc = ((pairwise_win + 0.5 * pairwise_tie) / pairwise_tot) if pairwise_tot else None
        return {
            "n_groups": n_groups, "observed_successes": successes,
            "observed_top1_rate": successes / n_groups if n_groups else None,
            "matched_random_expected_successes": p_sum,
            "matched_random_rate": p_sum / n_groups if n_groups else None,
            "paired_diff": (successes - p_sum) / n_groups if n_groups else None,
            "pairwise_ranking_accuracy": pairwise_acc, "pairwise_pairs": pairwise_tot,
            "mrr": mrr_sum / n_groups if n_groups else None,
            "top2_oracle_retention": top2_hit / top2_n if top2_n else None,
            "top3_oracle_retention": top3_hit / top3_n if top3_n else None,
            "per_group_p": per_group_p,
        }, successes, per_group_p

    print("[terminal] evaluating primary selector on informative heldout groups ...")
    primary_eval, obs_succ, per_group_p = evaluate_selector(informative_heldout, model_hid, "full_features")
    exact_p = poisson_binomial_sf(per_group_p, obs_succ)
    primary_eval["exact_poisson_binomial_p_value_ge_observed"] = exact_p

    print("[terminal] evaluating shortcut-only selector (control) ...")
    shortcut_eval, _, _ = evaluate_selector(informative_heldout, model_sc, None, shortcut=True)

    # candidate-level AUROC on ALL heldout candidates (not just informative groups)
    heldout_records = [r for tu in heldout_tasks for r in groups[tu]]
    y_ho_all = torch.tensor([1.0 if r["success"] else 0.0 for r in heldout_records])
    if y_ho_all.float().mean() not in (0.0, 1.0):
        Xhid_ho_all = torch.stack([r["full_features"].flatten().float() for r in heldout_records])
        scores_ho_all = apply_final(model_hid, Xhid_ho_all)
        candidate_level_auroc = auroc(scores_ho_all, y_ho_all)
    else:
        candidate_level_auroc = None

    # -------------------------------------------------- task-clustered bootstrap on paired_diff
    rng = torch.Generator().manual_seed(SEED)
    diffs = []
    for _ in range(2000):
        pick = torch.randint(0, len(informative_heldout), (len(informative_heldout),), generator=rng).tolist()
        picked = [informative_heldout[i] for i in pick]
        succ_b = 0
        p_b = 0.0
        for tu in picked:
            cands, scores = score_group(tu, model_hid, "full_features")
            succ = [bool(r["success"]) for r in cands]
            order = sorted(range(len(cands)), key=lambda i: -float(scores[i]))
            if succ[order[0]]:
                succ_b += 1
            p_b += sum(succ) / len(cands)
        diffs.append((succ_b - p_b) / len(picked))
    diffs.sort()
    n_d = len(diffs)
    boot_ci = {"mean": sum(diffs) / n_d, "ci95": [diffs[int(0.025 * n_d)], diffs[int(0.975 * n_d)]],
              "excludes_zero": bool(diffs[int(0.025 * n_d)] > 0 or diffs[int(0.975 * n_d)] < 0)}
    print(f"[terminal] paired diff bootstrap: {boot_ci}")

    # -------------------------------------------------- controls
    controls = {}
    controls["zero_task_crossing"] = len(set(tasks_tv) & set(informative_heldout)) == 0
    seen_cand = set()
    dup = 0
    for tu in informative_heldout:
        for r in groups[tu]:
            key = (r["task_uid"], r["candidate_idx"])
            if key in seen_cand:
                dup += 1
            seen_cand.add(key)
    controls["duplicate_candidate_count"] = dup
    # candidate-order invariance: reversing candidate order within group should not change argmax
    order_check_fail = 0
    for tu in informative_heldout[: min(20, len(informative_heldout))]:
        cands, scores = score_group(tu, model_hid, "full_features")
        rev_cands = list(reversed(cands))
        rev_scores = apply_final(model_hid, torch.stack([r["full_features"].flatten().float() for r in rev_cands]))
        top1_fwd = cands[int(torch.argmax(scores))]["candidate_idx"]
        top1_rev = rev_cands[int(torch.argmax(rev_scores))]["candidate_idx"]
        if top1_fwd != top1_rev:
            order_check_fail += 1
    controls["candidate_order_invariance_failures"] = order_check_fail
    # shuffled-score control: random scores should have near-zero paired diff
    rng2 = torch.Generator().manual_seed(SEED + 1)
    shuf_succ = 0
    shuf_p = 0.0
    for tu in informative_heldout:
        cands = groups[tu]
        n = len(cands)
        rand_scores = torch.rand(n, generator=rng2)
        succ = [bool(r["success"]) for r in cands]
        top1 = int(torch.argmax(rand_scores))
        if succ[top1]:
            shuf_succ += 1
        shuf_p += sum(succ) / n
    controls["shuffled_score_paired_diff"] = (shuf_succ - shuf_p) / len(informative_heldout)
    controls["no_held_out_label_tuning"] = "hyperparameters selected via grouped CV on train+val only"

    result.update({
        "primary_selector": {"family": "standardize->PCA->L2-logistic", "cv_auroc": best_hid[0],
                             "k_pca": best_hid[1], "l2": best_hid[2], "feature": "full_features (terminal)"},
        "primary_selector_heldout_informative": primary_eval,
        "shortcut_only_selector_heldout_informative": shortcut_eval,
        "candidate_level_auroc_all_heldout": candidate_level_auroc,
        "paired_diff_task_clustered_bootstrap": boot_ci,
        "controls": controls,
    })

    if (boot_ci["excludes_zero"] and primary_eval["paired_diff"] > 0 and exact_p < 0.05 and
            controls["candidate_order_invariance_failures"] == 0 and controls["duplicate_candidate_count"] == 0):
        verdict = "TERMINAL_SELECTION_ESTABLISHED"
    elif primary_eval["paired_diff"] > 0 and not boot_ci["excludes_zero"]:
        verdict = "TERMINAL_SELECTION_ABOVE_RANDOM_BUT_UNDERPOWERED"
    elif primary_eval["paired_diff"] <= 0:
        verdict = "TERMINAL_SELECTION_NOT_MATERIALLY_ABOVE_RANDOM"
    else:
        verdict = "INCONCLUSIVE_RUNTIME_OR_POWER_LIMITED"
    result["verdict"] = verdict
    print(f"[terminal] VERDICT: {verdict}")

    out_path = TERMINAL_ROOT / "terminal_results.json"
    out_path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"[terminal] wrote {out_path}")


if __name__ == "__main__":
    main()
