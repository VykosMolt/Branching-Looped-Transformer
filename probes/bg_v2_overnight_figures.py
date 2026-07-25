"""Paper 1 v2 overnight programme -- the 6 minimal figures (spec cap).

Reads only already-written result JSONs; makes no new computation.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/moloch/ouro_project/artifacts/reports/paper1_v2_overnight_20260724")


def fig1_horizon_auroc_and_increment():
    h = json.loads((ROOT / "horizon_logic/auroc_results.json").read_text())
    ho = h["heldout"]
    boot = h["paired_task_clustered_bootstrap_combined_minus_shortcut"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    ax = axes[0]
    labels = ["Shortcut\nonly", "Hidden\nonly", "Hidden+\nshortcuts"]
    vals = [ho["auroc_shortcut_only"], ho["auroc_hidden_only"], ho["auroc_hidden_plus_shortcuts"]]
    bars = ax.bar(labels, vals, color=["#888", "#4C72B0", "#55A868"])
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Heldout AUROC")
    ax.set_title("Horizon Logic: strict pre-answer AUROC")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

    ax = axes[1]
    mean_d = ho["incremental_auroc_combined_minus_shortcut"]
    lo, hi = boot["ci95"]
    ax.errorbar([0], [mean_d], yerr=[[mean_d - lo], [hi - mean_d]], fmt="o", color="#55A868",
                capsize=6, markersize=8)
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1)
    ax.set_xlim(-1, 1)
    ax.set_xticks([])
    ax.set_ylabel("Incremental AUROC (combined - shortcut)")
    ax.set_title(f"Paired task-clustered 95% CI\n[{lo:.3f}, {hi:.3f}]")
    fig.tight_layout()
    out = ROOT / "horizon_logic/figures/fig1_auroc_and_increment.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def fig3_terminal_observed_vs_random():
    t = json.loads((ROOT / "terminal_selection/terminal_results.json").read_text())
    prim = t["primary_selector_heldout_informative"]
    sc = t["shortcut_only_selector_heldout_informative"]
    n = prim["n_groups"]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    labels = ["Matched\nrandom", "Hidden-state\nselector", "Shortcut-only\nselector"]
    rates = [prim["matched_random_rate"], prim["observed_top1_rate"], sc["observed_top1_rate"]]
    colors = ["#888", "#4C72B0", "#C44E52"]
    bars = ax.bar(labels, rates, color=colors)
    for b, v in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(f"Forced top-1 success rate (n={n} informative groups)")
    ax.set_title("Terminal selection vs. exact matched-random\n(shortcut control shown for honesty, not as the primary claim)")
    fig.tight_layout()
    out = ROOT / "terminal_selection/figures/fig3_observed_vs_matched_random.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def fig4_terminal_oracle_retention():
    t = json.loads((ROOT / "terminal_selection/terminal_results.json").read_text())
    prim = t["primary_selector_heldout_informative"]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    labels = ["Top-1\n(forced)", "Top-2\noracle retention", "Top-3\noracle retention"]
    vals = [prim["observed_top1_rate"], prim["top2_oracle_retention"], prim["top3_oracle_retention"]]
    bars = ax.bar(labels, vals, color="#4C72B0")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title(f"Hidden-state selector: detection vs. selection\nMRR={prim['mrr']:.3f}, pairwise acc={prim['pairwise_ranking_accuracy']:.3f}")
    fig.tight_layout()
    out = ROOT / "terminal_selection/figures/fig4_detection_vs_selection.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def fig5_geometry_alignment_by_locus():
    g = json.loads((ROOT / "subspace_geometry/geometry_results.json").read_text())
    ro = g["readable_outcome_principal_angles"]
    null3 = g["null_random_rankmatched"]["3"]["mean_angle_deg"]
    loci = ["L1_16", "L2_16", "L3_8", "L3_16", "L4_8", "L4_16", "L4_24", "L4_36", "L4_47"]
    vals = [ro[loc]["3"]["mean_angle_deg"] for loc in loci]
    colors = ["#999" if loc in ("L1_16", "L2_16") else ("#C44E52" if loc.startswith("L4_2") or loc in ("L4_24", "L4_36", "L4_47") else "#4C72B0") for loc in loci]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(loci, vals, color=colors)
    ax.axhline(null3, color="gray", linestyle="--", linewidth=1, label=f"random-subspace null ({null3:.1f} deg)")
    ax.set_ylabel("Rank-3 mean principal angle (deg)\nreadable <-> outcome")
    ax.set_title("Readable<->outcome alignment by locus (lower = more aligned)\nblue=early, red=late, gray=loop-transition controls")
    ax.legend(fontsize=8)
    plt.xticks(rotation=30)
    fig.tight_layout()
    out = ROOT / "subspace_geometry/figures/fig5_alignment_by_locus.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def fig6_loop_rotation():
    g = json.loads((ROOT / "subspace_geometry/geometry_results.json").read_text())
    rot = g["loop_to_loop_readable_rotation_L16"]
    transitions = ["L1_16_to_L2_16", "L2_16_to_L3_16", "L3_16_to_L4_16"]
    vals = [rot[t]["1"]["mean_angle_deg"] for t in transitions]
    labels = ["L1->L2", "L2->L3", "L3->L4"]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(labels, vals, marker="o", color="#4C72B0", markersize=8, linewidth=2)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.5, f"{v:.1f} deg", ha="center", fontsize=9)
    ax.set_ylabel("Rank-1 principal angle (deg), physical layer 16")
    ax.set_title("Loop-to-loop readable-subspace rotation\n(largest at L1->L2, consistent with the frozen study's finding)")
    ax.set_ylim(0, max(vals) + 10)
    fig.tight_layout()
    out = ROOT / "subspace_geometry/figures/fig6_loop_rotation.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    for d in ("horizon_logic/figures", "terminal_selection/figures", "subspace_geometry/figures"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    fig1_horizon_auroc_and_increment()
    fig3_terminal_observed_vs_random()
    fig4_terminal_oracle_retention()
    fig5_geometry_alignment_by_locus()
    fig6_loop_rotation()
    print("done")
