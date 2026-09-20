"""Within-domain strict-preanswer specificity analysis.

For each domain SEPARATELY (no domain mixing), test whether strict-preanswer
Ouro-RLTT hidden states predict per-sample correctness beyond length and
logprob/entropy shortcuts, under task-grouped CV.

Baselines per domain/cut: random, shortcut-only (length metadata), logprob/entropy,
shortcut+logprob, hidden, hidden+all. Reuses probe utilities from the controls
analysis module. External verifier labels only.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "shared/utilities/tests/manual"))
PROTO = PROJECT_ROOT / "artifacts/reports/proto_introspection"

import proto_introspection_controls_analysis as C  # standardize/pca/logreg/auroc/grouped CV/bootstrap


def balanced_acc(probs, labels, thr=0.5):
    pred = (probs >= thr).long()
    y = labels.long()
    out = []
    for c in (0, 1):
        m = y == c
        if m.sum() > 0:
            out.append(float((pred[m] == c).float().mean()))
    return round(sum(out) / len(out), 4) if out else float("nan")


def quantiles(xs):
    if not xs:
        return {}
    t = torch.tensor(sorted(xs)).float()
    q = lambda p: round(float(t[min(len(t) - 1, int(p * len(t)))]), 1)
    return {"mean": round(float(t.mean()), 1), "median": q(0.5), "p10": q(0.1), "p90": q(0.9), "min": round(float(t.min()),1), "max": round(float(t.max()),1)}


def probe(X, y, groups, label, k_pca=24, l2=2.0):
    y = y.long()
    oof = C.cv_oof_scores(X, y, groups, k_pca=k_pca, l2=l2)
    a = C.auroc(oof, y)
    return {"label": label, "n": int(X.shape[0]), "auroc": round(a, 4) if not math.isnan(a) else None,
            "auroc_ci95": C.bootstrap_auroc_ci(oof, y), "acc": round(C.acc_at_half(oof, y), 4),
            "balanced_acc": balanced_acc(oof, y)}, oof


def combined_probe(Xhid, Xsc, y, groups, label, k_pca=24, l2=2.0):
    y = y.long()
    oof = C.cv_oof_combined(Xhid, Xsc, y, groups, k_pca=k_pca, l2=l2)
    a = C.auroc(oof, y)
    return {"label": label, "n": int(Xhid.shape[0]), "auroc": round(a, 4) if not math.isnan(a) else None,
            "auroc_ci95": C.bootstrap_auroc_ci(oof, y), "acc": round(C.acc_at_half(oof, y), 4),
            "balanced_acc": balanced_acc(oof, y)}


def analyze_domain(domain, recs):
    # ---- build per-sample preanswer examples (grouped by task) ----
    sx_hid, sx_len, sx_lp, sy, sgroups, npre = [], [], [], [], [], []
    for r in recs:
        for s in r["samples"]:
            if not s.get("has_preanswer") or "preanswer_feat" not in s:
                continue
            lp = s.get("lp_mean_logprob"); en = s.get("lp_mean_entropy"); le = s.get("lp_last_entropy")
            if lp is None or (isinstance(lp, float) and math.isnan(lp)):
                continue
            sx_hid.append(s["preanswer_feat"].flatten().float())
            sx_len.append(torch.tensor([r["question_chars"] / 1000.0, r["prompt_tok"] / 512.0, s["n_pre_tok"] / 256.0]))
            sx_lp.append(torch.tensor([float(lp), float(en), float(le)]))
            sy.append(1 if s["correct"] else 0)
            sgroups.append(r["task_id"])
            npre.append(s["n_pre_tok"])
    out = {"n_tasks": len(recs), "n_groups": len(set(sgroups)), "n_sample_examples": len(sy),
           "n_pre_tok_dist": quantiles(npre)}
    if len(sy) < 40 or len(set(sy)) < 2:
        out["preanswer"] = {"note": "insufficient examples", "n": len(sy)}
        out["verdict"] = "UNDERPOWERED"
        return out
    y = torch.tensor(sy)
    groups = sgroups
    Xhid = torch.stack(sx_hid); Xlen = torch.stack(sx_len); Xlp = torch.stack(sx_lp)
    Xsc_lp = torch.cat([Xlen, Xlp], dim=1)
    out["base_rate"] = round(float(y.float().mean()), 4)
    out["class_balance"] = {"pos": int(y.sum()), "neg": int((1 - y).sum())}

    sc, _ = probe(Xlen, y, groups, "shortcut_length", k_pca=min(3, Xlen.shape[1]), l2=1.0)
    lp, _ = probe(Xlp, y, groups, "logprob_entropy", k_pca=min(3, Xlp.shape[1]), l2=1.0)
    sclp, _ = probe(Xsc_lp, y, groups, "shortcut+logprob", k_pca=min(6, Xsc_lp.shape[1]), l2=1.0)
    hid, _ = probe(Xhid, y, groups, "hidden", k_pca=24, l2=2.0)
    hid_all = combined_probe(Xhid, Xsc_lp, y, groups, "hidden+shortcut+logprob")

    best_shortcut = max(v["auroc"] for v in (sc, lp, sclp) if v["auroc"] is not None)
    deltas = {
        "hidden_minus_shortcut_len": round(hid["auroc"] - sc["auroc"], 4),
        "hidden_minus_logprob": round(hid["auroc"] - lp["auroc"], 4),
        "hidden_minus_best_shortcut": round(hid["auroc"] - best_shortcut, 4),
        "hidden_all_minus_shortcut_logprob": round(hid_all["auroc"] - sclp["auroc"], 4),
    }
    out["preanswer"] = {"random": {"auroc": 0.5, "acc": out["base_rate"]},
                        "shortcut_length": sc, "logprob_entropy": lp, "shortcut_plus_logprob": sclp,
                        "hidden": hid, "hidden_plus_all": hid_all, "best_shortcut_auroc": round(best_shortcut, 4),
                        "deltas": deltas}

    # ---- prompt_only per-task (sample-independent) -> maj_correct ----
    py, pgroups, px_hid, px_len = [], [], [], []
    for r in recs:
        maj = 1 if (r["n_correct"] / max(r["n_samples"], 1)) >= 0.5 else 0
        px_hid.append(r["prompt_only_feat"].flatten().float())
        px_len.append(torch.tensor([r["question_chars"] / 1000.0, r["prompt_tok"] / 512.0]))
        py.append(maj); pgroups.append(r["task_id"])
    if len(py) >= 40 and len(set(py)) == 2:
        yp = torch.tensor(py)
        ph, _ = probe(torch.stack(px_hid), yp, pgroups, "prompt_only_hidden", k_pca=24, l2=2.0)
        ps, _ = probe(torch.stack(px_len), yp, pgroups, "prompt_only_shortcut", k_pca=2, l2=1.0)
        out["prompt_only"] = {"hidden": ph, "shortcut": ps,
                              "delta_hidden_minus_shortcut": round(ph["auroc"] - ps["auroc"], 4) if ph["auroc"] and ps["auroc"] else None}
    else:
        out["prompt_only"] = {"note": "insufficient/degenerate", "n": len(py)}

    # ---- verdict ----
    powered = len(set(groups)) >= 150
    hid_au = hid["auroc"]; hid_lo = hid["auroc_ci95"][0]
    d_best = deltas["hidden_minus_best_shortcut"]
    above_chance = hid_lo is not None and hid_lo > 0.5
    combined_helps = deltas["hidden_all_minus_shortcut_logprob"] >= 0.01
    if not powered:
        verdict = "UNDERPOWERED"
    elif above_chance and d_best >= 0.05 and combined_helps:
        verdict = "PASS"
    elif above_chance and d_best >= 0.03 and combined_helps:
        verdict = "PASS"  # useful-positive margin
    elif above_chance and d_best <= 0.02 and d_best > -0.03:
        verdict = "PARTIAL"  # above chance but not shortcut-free
    elif not above_chance:
        verdict = "FAIL"  # not above chance within domain
    else:
        verdict = "PARTIAL"
    out["verdict"] = verdict
    out["verdict_inputs"] = {"powered_ge150_groups": powered, "hidden_auroc": hid_au,
                             "hidden_ci_lo": hid_lo, "above_chance": above_chance,
                             "hidden_minus_best_shortcut": d_best, "combined_helps": combined_helps}
    return out


def main():
    cands = sorted(PROTO.glob("within_domain_recapture.pt"))
    if not cands:
        print("MISSING within_domain_recapture.pt"); return
    payload = torch.load(cands[0], map_location="cpu", weights_only=False)
    results = {"seed": payload.get("seed"), "samples_per_task": payload.get("samples_per_task"),
               "meta": payload.get("meta"), "domains": {}}
    for domain, recs in payload["records"].items():
        print(f"[wd-analysis] domain={domain} tasks={len(recs)}")
        results["domains"][domain] = analyze_domain(domain, recs)
    outp = PROTO / "within_domain_specificity_results.json"
    outp.write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(f"[wd-analysis] wrote {outp}")
    print("\n=== SUMMARY ===")
    for d, r in results["domains"].items():
        if "preanswer" in r and "hidden" in r["preanswer"]:
            pa = r["preanswer"]
            print(f"{d}: n_grp={r['n_groups']} n_ex={r['n_sample_examples']} base={r.get('base_rate')} "
                  f"VERDICT={r['verdict']}")
            print(f"   hidden={pa['hidden']['auroc']} CI{pa['hidden']['auroc_ci95']} | "
                  f"shortcut_len={pa['shortcut_length']['auroc']} logprob={pa['logprob_entropy']['auroc']} "
                  f"sc+lp={pa['shortcut_plus_logprob']['auroc']} | hidden+all={pa['hidden_plus_all']['auroc']}")
            print(f"   deltas: {pa['deltas']}")
            print(f"   n_pre_tok: {r['n_pre_tok_dist']}")
        else:
            print(f"{d}: {r.get('verdict')} ({r.get('preanswer', {}).get('note')})")


if __name__ == "__main__":
    main()
