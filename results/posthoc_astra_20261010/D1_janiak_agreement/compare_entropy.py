"""Side-by-side of results_no_entropy.json and results.json (with entropy): primary contrasts, T1 and fair-comparison
deltas, T2 deltas, Orgad controls. Reads only; prints markdown.  usage: python -B compare_entropy.py"""
import json

A = json.load(open("results_no_entropy.json"))
B = json.load(open("results.json"))
assert A["entropy_included"] is False and B["entropy_included"] is True and A["plan_sha256"] == B["plan_sha256"]


def iv(e, d=4):
    if not e or e.get("ci_low") is None:
        return "n/a"
    return f"{e['point_estimate']:+.{d}f} [{e['ci_low']:+.{d}f}, {e['ci_high']:+.{d}f}]"


def au(e):
    return iv(e, 3).replace("+", "") if e and e.get("ci_low") is not None else "not reported"


def row(name, get, fmt=iv):
    return f"| {name} | {fmt(get(A))} | {fmt(get(B))} |"


L = ["| quantity | without entropy | with entropy |", "|---|---|---|"]
for c_a, c_b in zip(A["primary_contrasts"], B["primary_contrasts"]):
    assert c_a["name"] == c_b["name"]
    L.append(f"| PRIMARY {c_a['name']} ΔAURC B1−B0 (Holm p) | {iv(c_a['delta'])} ({c_a['holm_p']:.3f}, {c_a['reading']}) | "
             f"{iv(c_b['delta'])} ({c_b['holm_p']:.3f}, {c_b['reading']}) |")
for run in A["runs"]:
    t = lambda r: r["runs"][run]["T1"]
    L += [row(f"{run} T1 B0 AUROC", lambda r: {"point_estimate": t(r)["readouts"]["B0_traj"]["auroc"], "ci_low": 0, "ci_high": 0},
              lambda e: f"{e['point_estimate']:.3f}"),
          row(f"{run} T1 ΔAUROC B1−B0", lambda r: t(r)["paired_deltas"]["B1_traj_minus_B0_traj"]["auroc"], lambda e: iv(e, 3)),
          row(f"{run} T1 ΔAURC B1−B0", lambda r: t(r)["paired_deltas"]["B1_traj_minus_B0_traj"]["aurc"]),
          row(f"{run} fair F0 AUROC", lambda r: {"point_estimate": t(r)["readouts"]["F0_window"]["auroc"], "ci_low": 0, "ci_high": 0},
              lambda e: f"{e['point_estimate']:.3f}"),
          row(f"{run} fair ΔAUROC F1−F0", lambda r: t(r)["paired_deltas"]["F1_window_minus_F0_window"]["auroc"], lambda e: iv(e, 3)),
          row(f"{run} fair ΔAURC F1−F0", lambda r: t(r)["paired_deltas"]["F1_window_minus_F0_window"]["aurc"]),
          row(f"{run} T1 Orgad probe in agrees=1", lambda r: t(r)["orgad"]["strata_agrees"]["agrees=1.0"]["auroc"], au),
          row(f"{run} T1 Orgad probe residual on (vote share, agrees)", lambda r: t(r)["orgad"]["residual_auroc_probe_given_vote_share_and_agrees"], au)]
    for pop in ("full_population", "valid_plurality"):
        p = lambda r: r["runs"][run]["T2"]["populations"][pop]
        L += [row(f"{run} T2 {pop} ΔAURC B1−B0", lambda r: p(r)["paired_deltas"]["B1_minus_B0_aurc"]),
              row(f"{run} T2 {pop} ΔAURC B1q−B0q", lambda r: p(r)["paired_deltas"]["B1q_minus_B0q_aurc"]),
              row(f"{run} T2 {pop} Orgad unanimous stratum", lambda r: p(r)["orgad"]["strata"]["unanimous"]["auroc"], au),
              row(f"{run} T2 {pop} Orgad probe given vote (residual)", lambda r: p(r)["orgad"]["residual_auroc"]["probe_given_agreement"], au),
              row(f"{run} T2 {pop} probe for vote − B0 (ΔAURC)", lambda r: p(r)["orgad"]["paired_deltas_aurc"]["B0_rmd_for_vote_minus_B0"]),
              row(f"{run} T2 {pop} B1 − (probe for vote) (ΔAURC)", lambda r: p(r)["orgad"]["paired_deltas_aurc"]["B1_minus_B0_rmd_for_vote"])]
print("\n".join(L))
