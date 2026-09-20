"""Controls A/B/C analysis for the proto-introspection package.

Pure-torch tiny probes (standardize -> PCA -> L2 logistic regression) with
grouped/random cross-validation, AUROC + accuracy + bootstrap CIs.

Control A (timing / pre-answer leakage):
  A1 recapture cuts: prompt_only & preanswer (clean) vs gen16/gen32/full (leak gradient)
     -> predict external-verifier task success from hidden states with ZERO answer leak.
  A2 existing trajectory features: at short prefixes, stratify pre-answer vs leaked
     -> show the original "strong" prefix result is leakage-graded.

Control B (shortcut baselines):
  hidden-state probe vs shortcut-only (domain/length/parse/prefix/leak-flag) vs combined.

Control C (task-grouped heldout):
  existing branch-success probe under random vs task-grouped splits (+ pairwise acc).

Correctness labels are EXTERNAL only (gold answer key / verifier). Tap/evaluator
scores are never used as labels.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROBE = PROJECT_ROOT / "opi/taps/probes/bg_trajectory_prediction_2026-05-18"
PROTO = PROJECT_ROOT / "artifacts/reports/proto_introspection"
FINAL_MARKER = re.compile(r"FINAL\s*ANSWE", re.IGNORECASE)
SEED = 20260617
torch.manual_seed(SEED)
RNG = torch.Generator().manual_seed(SEED)


# ----------------------------- probe utilities -----------------------------
def standardize_fit(X):
    mu = X.mean(0)
    sd = X.std(0).clamp(min=1e-6)
    return mu, sd


def pca_fit(Xz, k):
    mu = Xz.mean(0)
    Xc = Xz - mu
    # economy SVD; components are right singular vectors
    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
    k = min(k, Vh.shape[0])
    return mu, Vh[:k]


def fit_logreg(X, y, l2=1.0, iters=120):
    n, d = X.shape
    w = torch.zeros(d, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    yf = y.float()
    opt = torch.optim.LBFGS([w, b], lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = X @ w + b
        loss = F.binary_cross_entropy_with_logits(z, yf) + l2 * (w @ w) / n
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def auroc(scores, labels):
    s = scores.flatten().double()
    y = labels.flatten().long()
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # O(n_pos*n_neg) Mann-Whitney with 0.5 for ties
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)
    wins = (diff > 0).double().sum() + 0.5 * (diff == 0).double().sum()
    return float(wins / (len(pos) * len(neg)))


def acc_at_half(probs, labels):
    pred = (probs >= 0.5).long()
    return float((pred == labels.long()).float().mean())


def bootstrap_auroc_ci(scores, labels, rounds=1000):
    s = scores.flatten().double()
    y = labels.flatten().long()
    n = len(s)
    if n < 8 or y.sum() == 0 or y.sum() == n:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(rounds):
        idx = torch.randint(0, n, (n,), generator=RNG)
        a = auroc(s[idx], y[idx])
        if not math.isnan(a):
            vals.append(a)
    if not vals:
        return (float("nan"), float("nan"))
    vals.sort()
    lo = vals[int(0.025 * len(vals))]
    hi = vals[int(0.975 * len(vals))]
    return (round(lo, 4), round(hi, 4))


def grouped_folds(groups, n_folds=5):
    uniq = sorted(set(groups))
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(uniq), generator=g).tolist()
    uniq = [uniq[i] for i in perm]
    fold_of = {gid: (i % n_folds) for i, gid in enumerate(uniq)}
    return [fold_of[gg] for gg in groups]


def cv_oof_scores(X, y, groups, k_pca=24, l2=1.0, n_folds=5):
    """Return out-of-fold predicted probabilities using standardize->PCA->logreg."""
    n = X.shape[0]
    foldids = grouped_folds(groups, n_folds)
    foldids = torch.tensor(foldids)
    oof = torch.zeros(n, dtype=torch.double)
    for f in range(n_folds):
        tr = foldids != f
        va = foldids == f
        if tr.sum() == 0 or va.sum() == 0:
            continue
        Xtr, ytr = X[tr], y[tr]
        Xva = X[va]
        if ytr.float().mean() in (0.0, 1.0):
            oof[va] = float(ytr.float().mean())
            continue
        mu, sd = standardize_fit(Xtr)
        Ztr = (Xtr - mu) / sd
        Zva = (Xva - mu) / sd
        pmu, comps = pca_fit(Ztr, k_pca)
        Ptr = (Ztr - pmu) @ comps.T
        Pva = (Zva - pmu) @ comps.T
        w, b = fit_logreg(Ptr, ytr, l2=l2)
        oof[va] = torch.sigmoid(Pva @ w + b).double()
    return oof


def cv_oof_combined(Xhid, Xsc, y, groups, k_pca=24, l2=2.0, n_folds=5):
    """Combined probe: PCA(hidden,k) concatenated with passthrough standardized
    shortcut features (so the few shortcut dims are NOT washed out by hidden PCA)."""
    n = Xhid.shape[0]
    foldids = torch.tensor(grouped_folds(groups, n_folds))
    oof = torch.zeros(n, dtype=torch.double)
    for f in range(n_folds):
        tr = foldids != f
        va = foldids == f
        if tr.sum() == 0 or va.sum() == 0 or y[tr].float().mean() in (0.0, 1.0):
            oof[va] = float(y[tr].float().mean()) if tr.sum() else 0.0
            continue
        hmu, hsd = standardize_fit(Xhid[tr])
        Htr = (Xhid[tr] - hmu) / hsd
        Hva = (Xhid[va] - hmu) / hsd
        pmu, comps = pca_fit(Htr, k_pca)
        Ptr = (Htr - pmu) @ comps.T
        Pva = (Hva - pmu) @ comps.T
        smu, ssd = standardize_fit(Xsc[tr])
        Str = (Xsc[tr] - smu) / ssd
        Sva = (Xsc[va] - smu) / ssd
        Ctr = torch.cat([Ptr, Str], dim=1)
        Cva = torch.cat([Pva, Sva], dim=1)
        w, b = fit_logreg(Ctr, y[tr], l2=l2)
        oof[va] = torch.sigmoid(Cva @ w + b).double()
    return oof


def combined_report(Xhid, Xsc, y, groups, k_pca=24, l2=2.0, label=""):
    y = y.long()
    oof = cv_oof_combined(Xhid, Xsc, y, groups, k_pca, l2)
    a = auroc(oof, y)
    return {"label": label, "n": int(Xhid.shape[0]), "auroc": round(a, 4) if not math.isnan(a) else None,
            "auroc_ci95": bootstrap_auroc_ci(oof, y), "acc_at_0.5": round(acc_at_half(oof, y), 4)}


def probe_report(X, y, groups, k_pca=24, l2=1.0, n_folds=5, label=""):
    y = y.long()
    oof = cv_oof_scores(X, y, groups, k_pca, l2, n_folds)
    a = auroc(oof, y)
    ci = bootstrap_auroc_ci(oof, y)
    return {
        "label": label, "n": int(X.shape[0]), "n_pos": int(y.sum()),
        "base_rate": round(float(y.float().mean()), 4),
        "auroc": round(a, 4) if not math.isnan(a) else None,
        "auroc_ci95": ci, "acc_at_0.5": round(acc_at_half(oof, y), 4),
        "n_groups": len(set(groups)), "k_pca": k_pca, "l2": l2,
    }, oof


# ----------------------------- data loading -----------------------------
def load_existing():
    feat = torch.load(PROBE / "prefix_features.pt", map_location="cpu", weights_only=False)["records"]
    cont = json.load(open(PROBE / "continued_prefixes.json"))["continued_prefixes"]
    cmap = {(r["task_id"], r["branch_id"], r["prefix_length"]): r for r in cont}
    rows = []
    for fr in feat:
        key = (fr["task_id"], fr["branch_id"], fr["prefix_length"])
        cr = cmap.get(key)
        if cr is None:
            continue
        ptext = cr.get("prefix_text", "") or ""
        pa = str(cr.get("parsed_answer") or "").strip()
        leak = bool(FINAL_MARKER.search(ptext)) or (pa != "" and pa in ptext)
        rows.append({
            "task_id": fr["task_id"], "domain": fr["domain"], "branch_id": fr["branch_id"],
            "prefix_length": fr["prefix_length"], "features": fr["features"].flatten().float(),
            "is_correct": bool(cr.get("is_correct")), "leak": leak,
            "prefix_tok": int(cr.get("prefix_token_count", 0)),
            "prefix_chars": len(ptext), "parse_failed": bool(cr.get("parse_failed")),
        })
    return rows


def load_recapture():
    cands = sorted(PROTO.glob("preanswer_recapture.pt"))
    if not cands:
        return None
    return torch.load(cands[0], map_location="cpu", weights_only=False)


DOMS = ["reasoning", "science", "gsm8k"]


def dom_onehot(domain):
    return torch.tensor([1.0 if domain == d else 0.0 for d in DOMS])


# ----------------------------- Control A1: recapture cuts -----------------------------
def control_a1(rec):
    out = {"present": rec is not None}
    if rec is None:
        out["note"] = "MISSING recapture artifact; A1 not run"
        return out, {}
    recs = rec["records"]
    out["n_tasks"] = len(recs)
    out["counts_by_domain"] = {d: sum(1 for r in recs if r["domain"] == d) for d in DOMS}
    out["samples_per_task"] = rec.get("samples_per_task")
    # leak audit at cut level
    cutnames = ["prompt_only", "gen16", "gen32", "preanswer", "full"]
    leak_rate = {}
    for c in cutnames:
        present = [r for r in recs if c in r["cut_features"]]
        if present:
            leak_rate[c] = round(sum(1 for r in present if r["cut_leak"].get(c)) / len(present), 4)
    out["cut_leak_rate"] = leak_rate
    # Label per cut: prompt_only is sample-independent -> maj_correct (task solvability);
    # generated cuts (gen16/gen32/preanswer/full) are sample-0's own trajectory -> s0_correct.
    cut_results = {}
    detail = {}
    for c in cutnames:
        present = [r for r in recs if c in r["cut_features"]]
        if len(present) < 20:
            cut_results[c] = {"n": len(present), "note": "underpowered (<20)"}
            continue
        X = torch.stack([r["cut_features"][c].flatten().float() for r in present])
        label_field = "maj_correct" if c == "prompt_only" else "s0_correct"
        y = torch.tensor([1 if r[label_field] else 0 for r in present])
        groups = [r["task_id"] for r in present]  # inherently 1/task
        rep, oof = probe_report(X, y, groups, k_pca=24, l2=2.0, label=f"cut={c} -> {label_field}")
        rep["label_field"] = label_field
        rep["clean_preanswer"] = c in ("prompt_only", "preanswer")
        rep["cut_leak_rate"] = leak_rate.get(c)
        # domain breakdown auroc
        dbrk = {}
        for d in DOMS:
            mask = torch.tensor([1 if r["domain"] == d else 0 for r in present]).bool()
            if mask.sum() >= 12 and y[mask].sum() not in (0, int(mask.sum())):
                dbrk[d] = round(auroc(oof[mask], y[mask]), 4)
        rep["auroc_by_domain"] = dbrk
        cut_results[c] = rep
        detail[c] = oof
    out["cuts"] = cut_results
    return out, detail


# ----------------------------- Control A2: existing leakage stratification -----------------------------
def control_a2(rows):
    out = {}
    for pl in (32, 64, 128, 256):
        sub = [r for r in rows if r["prefix_length"] == pl]
        leaked = [r for r in sub if r["leak"]]
        clean = [r for r in sub if not r["leak"]]
        entry = {"n_total": len(sub),
                 "leak_rate": round(len(leaked) / max(len(sub), 1), 4),
                 "n_clean": len(clean), "n_leaked": len(leaked)}
        for name, grp in (("clean_preanswer", clean), ("leaked", leaked)):
            if len(grp) >= 24 and 0 < sum(r["is_correct"] for r in grp) < len(grp):
                X = torch.stack([r["features"] for r in grp])
                y = torch.tensor([1 if r["is_correct"] else 0 for r in grp])
                groups = [r["task_id"] for r in grp]
                rep, _ = probe_report(X, y, groups, k_pca=24, l2=2.0, label=f"p{pl}:{name}")
                entry[name] = rep
            else:
                entry[name] = {"n": len(grp), "note": "underpowered or single-class"}
        out[f"prefix_{pl}"] = entry
    return out


# ----------------------------- Control B: shortcut baselines -----------------------------
def shortcut_feats_existing(r):
    return torch.cat([
        dom_onehot(r["domain"]),
        torch.tensor([
            r["prefix_length"] / 256.0,
            r["prefix_tok"] / 256.0,
            r["prefix_chars"] / 1000.0,
            1.0 if r["parse_failed"] else 0.0,
            1.0 if r["leak"] else 0.0,
        ]),
    ])


def control_b_existing(rows):
    # predict is_correct at the cleanest short prefix (32) on the PRE-ANSWER subset
    out = {}
    for pl, subset in (("p32_clean", [r for r in rows if r["prefix_length"] == 32 and not r["leak"]]),
                       ("p32_all", [r for r in rows if r["prefix_length"] == 32]),
                       ("all_prefixes", rows)):
        if len(subset) < 24:
            out[pl] = {"n": len(subset), "note": "underpowered"}
            continue
        y = torch.tensor([1 if r["is_correct"] else 0 for r in subset])
        groups = [r["task_id"] for r in subset]
        Xhid = torch.stack([r["features"] for r in subset])
        Xsc = torch.stack([shortcut_feats_existing(r) for r in subset])
        hid, _ = probe_report(Xhid, y, groups, k_pca=24, l2=2.0, label=f"{pl}:hidden")
        sc, _ = probe_report(Xsc, y, groups, k_pca=min(8, Xsc.shape[1]), l2=1.0, label=f"{pl}:shortcut")
        comb = combined_report(Xhid, Xsc, y, groups, k_pca=24, l2=2.0, label=f"{pl}:hidden+shortcut")
        out[pl] = {
            "hidden": hid, "shortcut": sc, "combined": comb,
            "delta_hidden_minus_shortcut_auroc": (
                round(hid["auroc"] - sc["auroc"], 4)
                if hid["auroc"] is not None and sc["auroc"] is not None else None),
            "delta_combined_minus_shortcut_auroc": (
                round(comb["auroc"] - sc["auroc"], 4)
                if comb["auroc"] is not None and sc["auroc"] is not None else None),
        }
    return out


def control_b_recapture(rec):
    if rec is None:
        return {"note": "MISSING recapture; B-prompt_only not run"}
    recs = [r for r in rec["records"] if "prompt_only" in r["cut_features"]]
    if len(recs) < 20:
        return {"n": len(recs), "note": "underpowered"}
    y = torch.tensor([1 if r["maj_correct"] else 0 for r in recs])
    groups = [r["task_id"] for r in recs]
    Xhid = torch.stack([r["cut_features"]["prompt_only"].flatten().float() for r in recs])
    # pre-answer shortcuts: only question/prompt length + domain (no answer info)
    Xsc = torch.stack([torch.cat([dom_onehot(r["domain"]),
                                  torch.tensor([r["question_chars"] / 1000.0,
                                                r["prompt_tok"] / 512.0])]) for r in recs])
    hid, _ = probe_report(Xhid, y, groups, k_pca=24, l2=2.0, label="prompt_only:hidden")
    sc, _ = probe_report(Xsc, y, groups, k_pca=min(5, Xsc.shape[1]), l2=1.0, label="prompt_only:shortcut")
    comb = combined_report(Xhid, Xsc, y, groups, k_pca=24, l2=2.0, label="prompt_only:hidden+shortcut")
    return {"hidden": hid, "shortcut": sc, "combined": comb,
            "delta_hidden_minus_shortcut_auroc": (
                round(hid["auroc"] - sc["auroc"], 4)
                if hid["auroc"] is not None and sc["auroc"] is not None else None),
            "delta_combined_minus_shortcut_auroc": (
                round(comb["auroc"] - sc["auroc"], 4)
                if comb["auroc"] is not None and sc["auroc"] is not None else None)}


# ----------------------------- Control C: grouped vs random -----------------------------
def random_folds(n, n_folds=5):
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(n, generator=g).tolist()
    return [perm.index(i) % n_folds if False else (perm[i] % n_folds) for i in range(n)]


def cv_oof_random(X, y, n_folds=5, k_pca=24, l2=2.0):
    n = X.shape[0]
    g = torch.Generator().manual_seed(SEED)
    foldids = (torch.randperm(n, generator=g) % n_folds)
    oof = torch.zeros(n, dtype=torch.double)
    for f in range(n_folds):
        tr = foldids != f
        va = foldids == f
        if tr.sum() == 0 or va.sum() == 0 or y[tr].float().mean() in (0.0, 1.0):
            oof[va] = float(y[tr].float().mean()) if tr.sum() else 0.0
            continue
        mu, sd = standardize_fit(X[tr])
        Ztr = (X[tr] - mu) / sd
        Zva = (X[va] - mu) / sd
        pmu, comps = pca_fit(Ztr, k_pca)
        w, b = fit_logreg((Ztr - pmu) @ comps.T, y[tr], l2=l2)
        oof[va] = torch.sigmoid(((Zva - pmu) @ comps.T) @ w + b).double()
    return oof


def pairwise_acc(rows_subset, oof, by_prefix=None):
    """within (task, prefix) groups, fraction of (correct,incorrect) pairs ranked right."""
    from collections import defaultdict
    groups = defaultdict(list)
    for i, r in enumerate(rows_subset):
        if by_prefix is not None and r["prefix_length"] != by_prefix:
            continue
        groups[(r["task_id"], r["prefix_length"])].append((float(oof[i]), r["is_correct"]))
    win = tie = tot = 0
    for vals in groups.values():
        cor = [s for s, c in vals if c]
        inc = [s for s, c in vals if not c]
        for sc in cor:
            for si in inc:
                tot += 1
                if sc > si:
                    win += 1
                elif sc == si:
                    tie += 1
    if tot == 0:
        return None, 0
    return round((win + 0.5 * tie) / tot, 4), tot


def control_c(rows):
    out = {}
    # branch-success probe over all prefixes; random vs task-grouped
    y = torch.tensor([1 if r["is_correct"] else 0 for r in rows])
    X = torch.stack([r["features"] for r in rows])
    groups = [r["task_id"] for r in rows]
    oof_grp = cv_oof_scores(X, y, groups, k_pca=24, l2=2.0)
    oof_rnd = cv_oof_random(X, y)
    out["n"] = len(rows)
    out["n_groups"] = len(set(groups))
    out["random_split"] = {"auroc": round(auroc(oof_rnd, y), 4),
                           "auroc_ci95": bootstrap_auroc_ci(oof_rnd, y),
                           "acc": round(acc_at_half(oof_rnd, y), 4)}
    out["task_grouped_split"] = {"auroc": round(auroc(oof_grp, y), 4),
                                 "auroc_ci95": bootstrap_auroc_ci(oof_grp, y),
                                 "acc": round(acc_at_half(oof_grp, y), 4)}
    out["delta_random_minus_grouped_auroc"] = round(
        auroc(oof_rnd, y) - auroc(oof_grp, y), 4)
    # pairwise accuracy under grouped split, by prefix + clean(p32) for the timing tie-in
    pw = {}
    for pl in (32, 64, 128, 256):
        acc, tot = pairwise_acc(rows, oof_grp, by_prefix=pl)
        pw[f"prefix_{pl}"] = {"pairwise_acc_grouped": acc, "n_pairs": tot}
    # clean p32 pairwise (pre-answer subset only)
    clean32_idx = [i for i, r in enumerate(rows) if r["prefix_length"] == 32 and not r["leak"]]
    if len(clean32_idx) >= 24:
        out["pairwise_clean_p32_note"] = ("pairwise on pre-answer p32 is sparse because "
                                          "within-task clean branches vary; see A2 for clean probe")
    out["pairwise_by_prefix_grouped"] = pw
    return out


# ----------------------------- main -----------------------------
def main():
    print("[analysis] loading existing trajectory data ...")
    rows = load_existing()
    print(f"[analysis] existing rows: {len(rows)}")
    rec = load_recapture()
    print(f"[analysis] recapture present: {rec is not None}"
          + (f" ({len(rec['records'])} tasks)" if rec else ""))

    results = {"seed": SEED}
    print("[analysis] Control A1 (recapture cuts) ...")
    a1, _ = control_a1(rec)
    results["control_a1_recapture_cuts"] = a1
    print("[analysis] Control A2 (existing leakage stratification) ...")
    results["control_a2_existing_leakage_strat"] = control_a2(rows)
    print("[analysis] Control B (shortcut baselines) ...")
    results["control_b_existing"] = control_b_existing(rows)
    results["control_b_recapture_prompt_only"] = control_b_recapture(rec)
    print("[analysis] Control C (grouped vs random) ...")
    results["control_c_grouped_heldout"] = control_c(rows)

    outp = PROTO / "controls_analysis_results.json"
    outp.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"[analysis] wrote {outp}")
    # brief console summary
    def g(d, *ks, default=None):
        for k in ks:
            d = d.get(k, {}) if isinstance(d, dict) else {}
        return d if d != {} else default
    print("\n=== SUMMARY ===")
    if rec is not None and "cuts" in a1:
        for c, r in a1["cuts"].items():
            if isinstance(r, dict) and r.get("auroc") is not None:
                print(f"A1 {c:11s} clean={r.get('clean_preanswer')} leak_rate={r.get('cut_leak_rate')} "
                      f"AUROC={r['auroc']} CI{r['auroc_ci95']} n={r['n']} base={r['base_rate']}")
    cc = results["control_c_grouped_heldout"]
    print(f"C random AUROC={cc['random_split']['auroc']} vs grouped AUROC={cc['task_grouped_split']['auroc']} "
          f"(delta {cc['delta_random_minus_grouped_auroc']})")


if __name__ == "__main__":
    main()
