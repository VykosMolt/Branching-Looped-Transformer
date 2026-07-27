#!/usr/bin/env python3
"""Paper 1 vector figures.

Every constant below is carried from a preserved result artifact or from the
pre-redesign figure source. This file does not read experimental artifacts at
build time and does not recompute results; it redraws already-verified values
through the shared plotting system. Artifact provenance for the v2 figures:

  fig_crossloop_localization  artifacts/reports/cross_loop_early_layer_taps_20260720/
                              {train_eval_results,bootstrap_stats}.json
  fig_preanswer_auroc         (GSM8K) paper section 5.2 / Appendix I.1;
                              (Horizon) .../paper1_v2_overnight_20260724/
                              horizon_logic/auroc_results.json
  fig_terminal_selection      .../paper1_v2_overnight_20260724/terminal_selection/
                              {terminal_results,group_manifest}.json
  fig_audit_schematic         schematic; the only numbers shown are Appendix B's
                              audited evaluator constants and the closed-form
                              sibling-in-training probability.
  fig_family_replication      (v3) .../family_xloop_v3_20260726/<model>/
                              train_eval_results.json, .../huginn_probe_v3_20260726/
                              {train_eval_results,bootstrap_stats_v3}.json
  fig_preanswer_auroc Horizon (v3) .../horizon_power_v3_20260726/auroc_results_v3.json
                              (pooled + new-only arms)

Colour discipline: identity never rests on colour alone. Every mark carries a
direct value label or an axis-side name, ordered feature sets use one hue
light->dark so they survive grayscale, and the one genuinely categorical
contrast (hidden vs shortcut-only selector) uses the teal/orange pair, which
clears the CVD and normal-vision separation floors. The teal/green pair does
not clear them and is no longer used for two adjacent series.
"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import matplotlib.patheffects as pe
from scipy.stats import beta as beta_dist
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper_style import figure_style as st

st.apply(paper=1)
OUT = Path(__file__).resolve().parent

# One hue, light -> dark, for the ordered feature sets of the pre-answer panels.
SHORTCUT = st.BASELINE      # neutral control
HIDDEN = st.P1              # mid teal
COMBINED = st.P1_DARK       # dark teal, the load-bearing endpoint


def save(fig, name: str) -> None:
    st.save(fig, OUT / name)
    plt.close(fig)
    print("wrote", name)


# --------------------------------------------------------------------------
# Figure 4 - two-domain strict pre-answer evidence
# --------------------------------------------------------------------------

def _preanswer_panel(ax, *, title, note, rows, increment, ci, chance=0.50,
                     xmax=0.90, caveat=None, extra=None):
    """One domain. Rows are ordered shortcut -> hidden -> combined, drawn in a
    single hue light->dark so the ordering survives grayscale; the load-bearing
    quantity is the paired increment bracket, not any absolute score."""
    n = len(rows)
    y_lo = y_hi = None
    for i, (name, val, color, role) in enumerate(rows):
        y = n - 1 - i
        strong = role == "hi"
        ax.plot([chance, val], [y, y], color=color, lw=1.7, alpha=.45, zorder=1,
                solid_capstyle="round")
        ax.scatter(val, y, s=112 if strong else 58, color=color,
                   edgecolor="white", lw=1.1 if strong else .7, zorder=3)
        ax.text(val + 0.008, y, f"{val:.3f}", va="center", ha="left",
                fontsize=8.2 if strong else 7.9,
                fontweight="semibold" if strong else "normal",
                color=st.INK if strong else st.MUTED)
        if role == "lo":
            y_lo = y
        elif role == "hi":
            y_hi = y
    ax.set_yticks(range(n), [r[0] for r in rows][::-1])
    ax.set_ylim(-1.62, n - 0.28)
    ax.set_xlim(chance, xmax)
    ax.set_xticks([0.50, 0.60, 0.70, 0.80])
    ax.set_xlabel("AUROC")
    ax.axvline(chance, color=st.RULE, lw=1.0, zorder=0)
    ax.text(chance + 0.006, n - 0.42, "chance = 0.50", va="center", ha="left",
            fontsize=7.0, color=st.MUTED)
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left", fontsize=10.0, pad=19)
    ax.text(0.0, 1.015, note, transform=ax.transAxes, ha="left", va="bottom",
            color=st.MUTED, fontsize=7.4)

    # Paired-increment bracket. Its length is the increment on the AUROC scale;
    # it is not an error bar around the absolute combined score. The value
    # label is ~0.056 axis units wide at 8pt; leaders start 0.075 past the
    # marker so a clear gap separates them from the printed values.
    xb = xmax - 0.015
    ax.plot([rows[0][1] + 0.075, xb], [y_lo, y_lo], color=st.UNRESOLVED, lw=.8, zorder=2)
    ax.plot([rows[-1][1] + 0.075, xb], [y_hi, y_hi], color=st.UNRESOLVED, lw=.8, zorder=2)
    ax.plot([xb, xb], [y_hi, y_lo], color=st.UNRESOLVED, lw=.8, zorder=2)
    ax.text(xb - 0.012, (y_lo + y_hi) / 2 + 0.24, increment, va="bottom", ha="right",
            color=st.UNRESOLVED, fontsize=8.0, fontweight="semibold")
    ax.text(chance + 0.004, -0.72, ci, va="center", ha="left",
            fontsize=7.2, color=st.UNRESOLVED, fontweight="semibold")
    if extra is not None:
        ax.text(chance + 0.004, -1.06, extra, va="center", ha="left",
                fontsize=7.0, color=st.MUTED)
    if caveat is not None:
        ax.text(chance + 0.004, -1.40, caveat, va="center", ha="left",
                fontsize=7.0, color=st.UNRESOLVED)


def fig_preanswer() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.45, 2.92),
                                   gridspec_kw={"wspace": 0.50})
    _preanswer_panel(
        ax1,
        title="GSM8K",
        note="arithmetic word problems · 170 tasks / 680 candidates",
        rows=[("Shortcut composite", 0.731, SHORTCUT, "lo"),
              ("Hidden alone", 0.745, HIDDEN, False),
              ("Hidden + shortcuts", 0.797, COMBINED, "hi")],
        increment="paired $\\Delta$ +0.066",
        ci="95% CI [+0.021, +0.112] · excludes zero",
        extra="components: length 0.687 · log-probability 0.569",
    )
    _preanswer_panel(
        ax2,
        title="Horizon Logic",
        note="proof-depth-2-4 entailment · 201 heldout tasks / 614 scorable (pooled)",
        rows=[("Shortcut composite", 0.652, SHORTCUT, "lo"),
              ("Hidden alone", 0.763, HIDDEN, False),
              ("Hidden + shortcuts", 0.763, COMBINED, "hi")],
        increment="paired $\\Delta$ +0.111",
        ci="95% CI [+0.056, +0.169] · excludes zero",
        extra="independent replication (510 new tasks): +0.095 [+0.038, +0.157]",
        caveat="robust to the adversarial malformed-sibling baseline (+0.107)",
    )
    fig.suptitle("Strict pre-answer success prediction  ·  the load-bearing "
                 "quantity is the within-domain nested increment",
                 x=.012, ha="left", y=1.125, va="bottom", fontsize=10.4, fontweight="semibold")
    fig.text(.012, 1.062, "Absolute AUROCs differ between domains by construction "
             "and are not comparable to each other; both nested increments exclude zero.",
             ha="left", fontsize=8.0, color=st.MUTED)
    save(fig, "fig_preanswer_auroc.pdf")


# --------------------------------------------------------------------------
# Figure 2 - cross-loop early-layer localization
# --------------------------------------------------------------------------

# Local-refit heldout macro top-1, layers x loops. Chance (tie-aware) = 0.2818.
XL_LAYERS = [8, 16, 24]
XL_REFIT = np.array([
    [0.4100, 0.5683, 0.6333, 0.6217],   # layer 8,  L1..L4
    [0.4400, 0.5833, 0.6167, 0.6283],   # layer 16
    [0.4283, 0.5683, 0.6100, 0.6183],   # layer 24
])
XL_CHANCE = 0.2818
XL_REF = {"L4_36": 0.6467, "L4_47": 0.6200}
# EARLY_READABLE cells: above chance AND within 0.03 of the layer-24 final-loop
# refit. (layer index, loop index).
XL_QUALIFY = {(0, 2), (0, 3), (1, 2), (1, 3)}
# Frozen transplant: paired difference against the target-local refit, by
# source loop, evaluated at L4.
XL_GAP = {
    8:  [(-0.2117, 0.527), (0.0066, 0.939), (0.0066, 0.914)],
    16: [(-0.1316, 0.775), (0.0000, 0.951), (0.0017, 0.985)],
    24: [(-0.1083, 0.771), (-0.0083, 0.903), (0.0184, 0.954)],
}


def fig_crossloop() -> None:
    fig = plt.figure(figsize=(7.45, 3.00))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.28, 1.06, 1.06], wspace=0.55)
    axA, axB, axC = (fig.add_subplot(gs[i]) for i in range(3))

    # --- Panel A: loop x layer readability. One sequential hue; cells are
    # separated by surface gaps, and the four qualifying cells form a single
    # 2x2 block marked by ONE rounded outline instead of per-cell boxes.
    # The displayed shade passes through a mild gamma (frac^1.25): it lightens
    # the low first-pass cells so the L1->L2 step is easier to see, while the
    # parity cluster at the top end moves by under 0.01 - the numeric mapping
    # and ordering are unchanged.
    cmap = mpl_cmap_from(st.SEQ_BLUES)
    lo, hi = XL_CHANCE, 0.66
    for r in range(3):                       # r = 0 is layer 8, drawn on top
        for c in range(4):
            v = XL_REFIT[r, c]
            frac = (v - lo) / (hi - lo)
            disp = frac ** 1.25
            axA.add_patch(Rectangle((c + 0.045, 2 - r + 0.045), 0.91, 0.91,
                                    facecolor=cmap(disp), edgecolor="none",
                                    zorder=2))
            axA.text(c + 0.5, 2 - r + 0.5, f"{v:.3f}", ha="center", va="center",
                     fontsize=8.0, color="white" if disp > 0.55 else st.INK,
                     fontweight="semibold" if (r, c) in XL_QUALIFY else "normal")
    axA.add_patch(FancyBboxPatch((2.075, 1.075), 1.85, 1.85,
                                 boxstyle="round,pad=0.03,rounding_size=0.10",
                                 fill=False, edgecolor=st.UNRESOLVED, lw=1.25,
                                 zorder=4))
    axA.set_xlim(0, 4)
    axA.set_ylim(0, 3.06)
    axA.set_xticks([0.5, 1.5, 2.5, 3.5], [f"L{i}" for i in range(1, 5)])
    axA.set_yticks([2.5, 1.5, 0.5], [f"layer {l}" for l in XL_LAYERS])
    axA.tick_params(length=0)
    axA.grid(visible=False)
    for s in axA.spines.values():
        s.set_visible(False)
    axA.set_title("A     Local refit · macro top-1", loc="left",
                  fontsize=9.4, pad=14)
    axA.text(0.0, 1.015, f"chance {XL_CHANCE:.3f} · outline = parity with 24_L4",
             transform=axA.transAxes, ha="left",
             va="bottom", fontsize=7.1, color=st.MUTED)

    # --- Panel B: frozen transplant as a recovery slope. One thin line per
    # layer from its L1 loss up to the L2/L3 parity cluster; the shape (deep
    # dip, then flat at zero) is the finding.
    xs = np.arange(3)
    # A small per-layer dodge keeps the three markers distinct where the
    # L2/L3 values nearly coincide, without disturbing the shared shape.
    # Trajectory identity comes from compact direct labels at the L1 end -
    # the only end where the three lines are separated enough to label
    # without collision (they converge to within 0.017 at L2/L3).
    LINE = "#B9C3CA"                         # visible in print, still recessive
    for j, layer in enumerate(XL_LAYERS):
        xd = xs + (j - 1) * 0.055
        vals = [g[0] for g in XL_GAP[layer]]
        axB.plot(xd, vals, color=LINE, lw=1.2, zorder=1)
    for j, layer in enumerate(XL_LAYERS):
        xd = xs + (j - 1) * 0.055
        vals = [g[0] for g in XL_GAP[layer]]
        for i, v in enumerate(vals):
            axB.scatter(xd[i], v, s=48, color=st.P2 if i == 0 else st.P1,
                        edgecolor="white", lw=0.9, zorder=3)
    for j, layer in enumerate(XL_LAYERS):
        axB.text(-0.17 + (j - 1) * 0.055, XL_GAP[layer][0][0], f"L{layer}",
                 ha="right", va="center", fontsize=7.0, color=st.MUTED)
    axB.axhline(0, color=st.MUTED, lw=0.8, zorder=2)
    axB.text(1.5, 0.062, "L2/L3 transfer at parity", ha="center", va="bottom",
             fontsize=7.4, color=st.P1_DARK, fontweight="semibold")
    axB.text(1.5, 0.054, "$|\\Delta| \\leq 0.02$", ha="center", va="top",
             fontsize=6.8, color=st.MUTED)
    axB.text(0.46, -0.185, "first-pass taps rotate:\n$-$0.11 to $-$0.21",
             ha="left", va="center", fontsize=7.0, color=st.P2_DARK,
             fontweight="semibold", linespacing=1.3)
    axB.set_xlim(-0.68, 2.35)
    axB.set_ylim(-0.245, 0.096)
    axB.set_xticks(xs, ["L1", "L2", "L3"])
    axB.set_yticks([-0.2, -0.1, 0.0])
    axB.set_xlabel("source loop  (evaluated at L4)", fontsize=8.2)
    axB.set_ylabel("$\\Delta$ vs target-local refit", fontsize=8.4)
    axB.grid(axis="x", visible=False)
    axB.set_title("B     Frozen transplant", loc="left", fontsize=9.4, pad=14)
    axB.text(0.0, 1.015, "one line per layer: 8 / 16 / 24",
             transform=axB.transAxes, ha="left", va="bottom", fontsize=7.1,
             color=st.MUTED)

    # --- Panel C: readability against remaining same-loop depth, with the
    # parity criterion drawn as a band instead of implied by bold labels.
    band_lo, band_hi = 0.618 - 0.03, 0.618 + 0.03
    axC.axhspan(band_lo, band_hi, color=st.P1_PALE, zorder=0)
    axC.axhline(0.618, color=st.P1, lw=0.8, ls=(0, (4, 3)), zorder=1)
    axC.text(55.0, band_lo + 0.0022, "24_L4 $\\pm$ 0.03",
             ha="right", va="bottom", fontsize=6.8, color=st.MUTED)
    OFFSET = {"above": (0.0, 0.0035, "center", "bottom"),
              "above_l": (-3.0, 0.0035, "center", "bottom"),
              "below": (0.0, -0.0035, "center", "top"),
              "left": (-2.6, 0.0, "right", "center"),
              "right": (2.6, 0.0, "left", "center"),
              "right_u": (2.6, 0.0024, "left", "center")}
    pts = [("8_L3", 40, 0.6333, True, "above"), ("8_L4", 40, 0.6217, True, "right"),
           ("16_L3", 32, 0.6167, True, "below"), ("16_L4", 32, 0.6283, True, "above_l"),
           ("24_L4", 24, 0.6183, False, "above"), ("36_L4", 12, 0.6467, False, "above"),
           ("47_L4", 1, 0.6200, False, "right_u")]
    REF = "#6E7880"    # a step darker than BASELINE: legible in print, still recessive
    for name, depth, val, early, place in pts:
        axC.scatter(depth, val, s=64 if early else 42,
                    color=st.P1 if early else REF,
                    edgecolor="white", lw=.9, zorder=3)
        dx, dy, ha, va = OFFSET[place]
        axC.text(depth + dx, val + dy, name, ha=ha, va=va, fontsize=7.0,
                 color=st.INK,
                 fontweight="semibold" if early else "normal")
    axC.set_xlim(-3, 56)
    axC.set_ylim(0.582, 0.663)
    axC.set_yticks([0.59, 0.61, 0.63, 0.65])
    axC.set_xticks([0, 20, 40])
    axC.set_xlabel("layers left in the same loop", fontsize=8.2)
    axC.set_ylabel("macro top-1", fontsize=8.4)
    axC.set_title("C     Depth left after the tap", loc="left", fontsize=9.4,
                  pad=14)
    axC.text(0.0, 1.015, "readable early, with most of the loop still ahead",
             transform=axC.transAxes, ha="left", va="bottom", fontsize=7.1,
             color=st.MUTED)

    # The clause "in the reused physical stack" lives in the manuscript
    # caption (its opening sentence), so the figure title stays short.
    fig.suptitle("Recurrent refinement moves candidate-quality readability earlier",
                 x=.012, ha="left", y=1.155, va="bottom", fontsize=10.4, fontweight="semibold")
    fig.text(.012, 1.085, "Readout only. No intervention was performed at any "
             "locus; these are candidate readout surfaces, not steering sites.",
             ha="left", fontsize=8.0, color=st.MUTED)
    save(fig, "fig_crossloop_localization.pdf")


def mpl_cmap_from(colors):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("kirin_seq", colors)


# --------------------------------------------------------------------------
# Figure 3 (v3) - replication: family loop-slopes, Huginn trend, rotation
# --------------------------------------------------------------------------
# Values carried from artifacts/reports/{family_xloop_v3,huginn_probe_v3,
# cross_loop_early_layer_taps}_*/train_eval_results.json (heldout macro top-1,
# task-clustered protocol identical to Figure 2). Chance 0.2818 throughout.

FAM_SLOPES = {                     # earliest mapped layer, loops L1..L4
    "RLTT 2.6B · layer 8": [0.4100, 0.5683, 0.6333, 0.6217],
    "base 2.6B · layer 8": [0.4150, 0.5533, 0.6083, 0.6300],
    "Thinking 2.6B · layer 8": [0.4083, 0.5667, 0.6233, 0.6133],
    "1.4B · layer 4": [0.4650, 0.5250, 0.6017, 0.6033],
}
HUG_STEPS = [0.3317, 0.3200, 0.3033, 0.3617, 0.3833, 0.3783, 0.4017, 0.4383]
# Frozen transfer, Delta vs target-local refit at matched distance
# (first/second-pass source -> pass 4), earliest mapped layer per model
# (Huginn: recurrence steps at the core locus). (point, ci_lo, ci_hi) from the
# sealed clustered bootstraps of each run.
ROT_L1 = {"RLTT": (-0.2117, -0.2583, -0.1650), "base": (-0.2500, -0.2967, -0.2050),
          "Thinking": (-0.2017, -0.2500, -0.1550), "1.4B": (-0.1783, -0.2250, -0.1317),
          "Huginn": (-0.0183, -0.0683, 0.0300)}
ROT_L2 = {"RLTT": (0.0067, -0.0200, 0.0317), "base": (-0.0150, -0.0417, 0.0117),
          "Thinking": (0.0050, -0.0217, 0.0317), "1.4B": (-0.0350, -0.0700, 0.0000),
          "Huginn": (-0.0133, -0.0633, 0.0367)}


def fig_family_replication() -> None:
    # Two-row layout: A and B share the top row (same y-concept and limits);
    # C spans the full width below, giving the whiskers and row labels room.
    fig = plt.figure(figsize=(7.45, 5.55))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1.0],
                          height_ratios=[1.50, 1.0], wspace=0.34, hspace=0.55)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])
    STYLES = [(st.P1_DARK, "-", "o"), (st.P1, "-", "s"),
              ("#7D93B5", "-", "^"), (st.P2, "-", "D")]

    # --- A: family loop-slopes at the earliest mapped layer. -----------------
    xs = [1, 2, 3, 4]
    for (name, vals), (color, ls, mk) in zip(FAM_SLOPES.items(), STYLES):
        axA.plot(xs, vals, ls, marker=mk, ms=5.2, lw=1.7, color=color,
                 markeredgecolor="white", markeredgewidth=0.8)
    axA.axhline(0.2818, color=st.RULE, lw=1.0)
    axA.text(0.85, 0.2890, "chance", ha="left", va="bottom", fontsize=7.4,
             color=st.MUTED)
    axA.set_xticks(xs, [f"L{i}" for i in xs])
    axA.set_xlim(0.72, 4.28)
    axA.set_ylim(0.25, 0.70)
    axA.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7])
    axA.set_xlabel("loop", fontsize=8.8)
    axA.set_ylabel("heldout macro top-1", fontsize=8.8)
    axA.grid(axis="x", visible=False)
    axA.set_title("A     Ouro family, earliest layer", loc="left",
                  fontsize=9.8, pad=15)
    axA.text(0.0, 1.015, "same subset, splits, taps, and grid as Figure 2",
             transform=axA.transAxes, ha="left", va="bottom", fontsize=7.4,
             color=st.MUTED)
    axA.legend(list(FAM_SLOPES), fontsize=7.0, frameon=False, loc="lower right",
               handlelength=1.5, borderaxespad=0.1, labelspacing=0.32)

    # --- B: Huginn readability across recurrence steps. ----------------------
    steps = list(range(1, 9))
    axB.plot(steps, HUG_STEPS, "-", marker="o", ms=5.2, lw=1.7, color=st.P1,
             markeredgecolor="white", markeredgewidth=0.8)
    axB.axhline(0.2818, color=st.RULE, lw=1.0)
    axB.text(8.3, 0.2890, "chance", ha="right", va="bottom", fontsize=7.4,
             color=st.MUTED)
    axB.set_xticks([1, 2, 4, 6, 8])
    axB.set_xlim(0.5, 8.5)
    axB.set_ylim(0.25, 0.70)
    axB.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7])
    axB.set_xlabel("recurrence step", fontsize=8.8)
    axB.grid(axis="x", visible=False)
    axB.set_title("B     Huginn-0125", loc="left", fontsize=9.8, pad=15)
    axB.text(0.0, 1.015, "out-of-family · core-boundary readout",
             transform=axB.transAxes, ha="left", va="bottom", fontsize=7.4,
             color=st.MUTED)
    axB.text(8.2, 0.545, "step 8 $-$ step 1:\n+0.107 [+0.058, +0.157]",
             ha="right", va="bottom", fontsize=7.6, color=st.P1_DARK,
             fontweight="semibold", linespacing=1.35)

    # --- C: first-pass rotation, full width. Shape = source pass (circle
    # first, square second); colour = architecture (copper Ouro family,
    # indigo Huginn); whiskers = 95% clustered-bootstrap CIs, so unresolved
    # Huginn losses read as unresolved, not as zeros.
    names = list(ROT_L1)
    y = np.arange(len(names))[::-1]
    axC.axvline(0, color=st.INK, lw=1.1, zorder=2)
    axC.text(0.004, len(names) - 0.42, "0 = target-local parity", ha="left",
             va="top", fontsize=7.4, color=st.MUTED)
    for series, dy, marker, size in [(ROT_L1, 0.17, "o", 58), (ROT_L2, -0.17, "s", 40)]:
        for n, yy in zip(names, y):
            pt, lo, hi = series[n]
            color = st.P1 if n == "Huginn" else st.P2
            axC.plot([lo, hi], [yy + dy, yy + dy], color=color, lw=1.3,
                     alpha=0.55, zorder=2, solid_capstyle="butt")
            axC.scatter(pt, yy + dy, s=size, color=color, marker=marker,
                        edgecolor="white", lw=1.0, zorder=3)
    axC.set_yticks(y, names)
    axC.tick_params(axis="y", labelsize=8.8)
    axC.set_xlim(-0.33, 0.115)
    axC.set_xticks([-0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1])
    axC.set_ylim(-0.95, len(names) - 0.20)
    axC.set_xlabel("frozen-transfer $\\Delta$ vs target-local refit", fontsize=8.8)
    axC.grid(axis="y", visible=False)
    axC.set_title("C     First-pass rotation", loc="left", fontsize=9.8, pad=15)
    axC.text(0.0, 1.03, "circles: first-pass source · squares: second-pass "
             "source · evaluated at step/pass 4 · whiskers: 95% task-clustered CIs",
             transform=axC.transAxes, ha="left", va="bottom", fontsize=7.4,
             color=st.MUTED)
    axC.text(-0.322, -0.68, "Ouro rotates after the first pass; no detectable "
             "Huginn analogue (weak step-1 signal)", ha="left", va="center",
             fontsize=7.4, color=st.INK)

    fig.suptitle("Readability rises with recurrent depth across Ouro "
                 "checkpoints and qualitatively in Huginn",
                 x=.012, ha="left", y=1.055, va="bottom", fontsize=11.2,
                 fontweight="semibold")
    fig.text(.012, 1.042, "Readout only. Absolute levels are not comparable "
             "across architectures; the shared trend and differing transfer "
             "geometry are the findings.", ha="left", fontsize=8.0,
             color=st.MUTED)
    save(fig, "fig_family_replication.pdf")


def fig_domain_transfer() -> None:
    panels = [
        ("Coding", [("Code tap", 0.9528, st.P1), ("HH tap", 0.6944, st.BASELINE)]),
        ("Alignment", [("HH tap", 0.6902, st.P1), ("Code tap", 0.5609, st.BASELINE),
                       ("Random-20 HH", 0.6038, st.BASELINE)]),
        # One hue light->dark, ordered by value. The teal/green pair used here
        # before separates by only 0.015 grayscale luminance (P1 vs POSITIVE),
        # i.e. not at all in print; P1_DARK vs P1 gives 0.058.
        ("Reasoning", [("Specialist", 0.7671, st.P1_DARK),
                       ("Balanced generalist", 0.7613, st.P1)]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.45, 1.90))
    for index, (ax, (title, rows)) in enumerate(zip(axes, panels)):
        names = [row[0] for row in rows][::-1]
        values = [row[1] for row in rows][::-1]
        colors = [row[2] for row in rows][::-1]
        y = np.arange(len(rows))
        ax.hlines(y, 0.5, values, color=st.RULE, lw=2.4, zorder=1)
        ax.scatter(values, y, s=52, color=colors, edgecolor="white", lw=0.7, zorder=3)
        for yi, value in zip(y, values):
            if value > 0.87:
                # A label to the right would spill into the next panel's ticks.
                ax.text(value, yi + 0.40, f"{value:.4f}", ha="center", va="center",
                        fontsize=8.1, color=st.INK)
            else:
                ax.text(value + 0.030, yi, f"{value:.4f}", va="center", fontsize=8.1,
                        color=st.INK)
        ax.set_yticks(y, names)
        ax.set_ylim(-0.75, len(rows) - 0.25)
        ax.set_xlim(0.50, 1.00)
        ax.set_xticks([0.5, 0.75, 1.0])
        ax.set_title(f"Evaluated on {title.lower()}", loc="left", fontsize=9.6)
        ax.grid(axis="y", visible=False)
        st.panel_label(ax, chr(ord("A") + index), x=-0.10, y=1.14)
    axes[1].set_xlabel("Held-out macro top-1")
    fig.suptitle("Task-disjoint domain transfer · zero crossing task IDs",
                 x=0.01, ha="left", y=1.22, fontsize=10.5, fontweight="semibold")
    fig.subplots_adjust(wspace=0.52)
    save(fig, "fig_domain_transfer.pdf")


def fig_survival_ablation() -> None:
    labels = ["Stage retention\n(task-disjoint)", "Terminal retention",
              "Terminal retention\nL47 channel ablated"]
    values = [0.9697, 1.0000, 0.0417]
    colors = [st.POSITIVE, st.POSITIVE, st.P2]
    # Three rows previously sat in a 2.65in panel, leaving two empty bands taller
    # than the rows themselves and stranding the ablation note in the middle of
    # them. Tighter height plus padded ylim keeps the rows as the figure's mass.
    fig, ax = plt.subplots(figsize=(7.15, 1.95))
    y = np.arange(3)[::-1]
    ax.hlines(y, 0, values, color=st.RULE, lw=6, zorder=1)
    ax.scatter(values, y, s=74, color=colors, edgecolor="white", lw=0.8, zorder=3)
    for yi, value in zip(y, values):
        ax.text(min(value + 0.025, 1.015), yi, f"{value:.4f}", va="center",
                fontsize=8.5, fontweight="semibold")
    ax.set_yticks(y, labels)
    ax.set_ylim(-0.92, 2.45)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Oracle retention")
    ax.set_title("Branch survival is readable; the layer-47 channel is causally load-bearing",
                 loc="left")
    # Anchored just under the ablated row it describes, not floating mid-panel.
    ax.text(0.055, -0.62, "causal ablation: 1.0000 to 0.0417  —  the layer-47 "
            "channel is load-bearing, not merely correlated", color=st.P2,
            fontsize=7.6, fontweight="semibold", va="center")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_survival_ablation.pdf")


def fig_s3b2() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.35, 1.72),
                                   gridspec_kw={"wspace": 0.50})
    # Detection: two candidate-level readers.
    y1 = np.array([1, 0])
    vals1 = [0.7515, 0.7755]
    names1 = ["L2 logistic", "Hidden ridge"]
    ax1.hlines(y1, 0.5, vals1, color=st.RULE, lw=3.4, zorder=1)
    ax1.scatter(vals1, y1, s=58, color=[st.BASELINE, st.P1], edgecolor="white",
                lw=.7, zorder=3)
    for y, value in zip(y1, vals1):
        ax1.text(value + .017, y, f"{value:.4f}", va="center", fontsize=8.3)
    ax1.set_yticks(y1, names1)
    ax1.set_ylim(-0.65, 1.65)
    ax1.set_xlim(.5, .86)
    ax1.set_xticks([.5, .6, .7, .8])
    ax1.set_xlabel("Candidate AUROC")
    ax1.set_title("Detection · correctness is readable", loc="left", pad=15)
    ax1.grid(axis="y", visible=False)
    st.panel_label(ax1, "A", x=-0.16, y=1.24)

    # Selection: point expectation and the observed underpowered count.
    y2 = np.array([1, 0])
    vals2 = [0.625, 0.3625]
    names2 = ["Observed selector", "Matched-random\nexpectation"]
    ax2.hlines(y2, 0, vals2, color=st.RULE, lw=3.4, zorder=1)
    ax2.scatter([vals2[0]], [y2[0]], s=68, facecolor="white", edgecolor=st.UNRESOLVED,
                lw=1.8, zorder=3)
    ax2.scatter([vals2[1]], [y2[1]], s=52, color=st.BASELINE, edgecolor="white",
                lw=.7, zorder=3)
    for y, value in zip(y2, vals2):
        ax2.text(value + .036, y, f"{value:.4f}".rstrip("0"), va="center", fontsize=8.3)
    ax2.set_yticks(y2, names2)
    ax2.set_ylim(-0.65, 1.65)
    ax2.set_xlim(0, .80)
    ax2.set_xticks([0, .2, .4, .6, .8])
    ax2.set_xlabel("Success on oracle-present groups")
    ax2.set_title("Forced selection · historical slice", loc="left",
                  color=st.UNRESOLVED, pad=15)
    ax2.text(0, 1.045, "N = 8 · exact p = 0.087 · superseded by §6.5",
             transform=ax2.transAxes, ha="left", va="bottom",
             color=st.UNRESOLVED, fontsize=7.9)
    ax2.grid(axis="y", visible=False)
    st.panel_label(ax2, "B", x=-0.16, y=1.24)
    save(fig, "fig_s3b2_detection_selection.pdf")


# --------------------------------------------------------------------------
# Figure 7 - terminal selection: observed, matched-random, and what decides it
# --------------------------------------------------------------------------

def fig_terminal_selection() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.45, 2.62),
                                   gridspec_kw={"wspace": 0.52,
                                                "width_ratios": [1.0, 1.0]})

    # --- Panel A: forced top-1 against the exact matched-random null. ---------
    # The null is a dashed reference line, not a third data row: both selectors
    # are read against it directly, and the margin bracket hangs off the hidden
    # selector because that is the tested comparison.
    null_x = 0.5897
    rows = [
        ("Hidden-state\nselector", 0.8718, "34 / 39", st.P1, True),
        ("Shortcut-only selector\n(no hidden state)", 0.8974, "35 / 39", st.P2, False),
    ]
    for i, (name, val, count, color, strong) in enumerate(rows):
        y = 1 - i
        ax1.plot([0, val], [y, y], color=color, lw=1.7, alpha=.40, zorder=1,
                 solid_capstyle="round")
        ax1.scatter(val, y, s=112 if strong else 66, color=color,
                    edgecolor="white", lw=1.0 if strong else .7, zorder=3)
        ax1.text(val + 0.026, y, f"{val:.3f}  ({count})", va="center", ha="left",
                 fontsize=8.1 if strong else 7.8,
                 fontweight="semibold" if strong else "normal",
                 color=st.INK if strong else st.MUTED)
    ax1.plot([null_x, null_x], [-0.52, 1.58], color=st.BASELINE, lw=1.0,
             ls=(0, (4, 3)), zorder=2)
    ax1.text(null_x, 1.70, "matched-random 0.590  (23.0 / 39)", ha="center",
             va="bottom", fontsize=7.2, color=st.MUTED)
    ax1.set_yticks([1, 0], [r[0] for r in rows])
    ax1.set_ylim(-2.05, 2.30)
    ax1.set_xlim(0, 1.24)
    ax1.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
    ax1.set_xlabel("forced top-1 success on informative groups")
    ax1.grid(axis="y", visible=False)
    ax1.set_title("A     Selection beats matched random", loc="left",
                  fontsize=9.6, pad=18)
    ax1.text(0.0, 1.015, "39 informative heldout groups · exact Poisson-binomial null",
             transform=ax1.transAxes, ha="left", va="bottom", fontsize=7.4,
             color=st.MUTED)
    ax1.text(0.0, -1.05, "hidden $-$ null: +0.282 · CI [+0.167, +0.391] · "
             "exact $p = 2.44\\times10^{-5}$", va="center", ha="left",
             fontsize=7.2, color=st.INK)
    ax1.text(0.0, -1.68, "no hidden-state-specific advantage",
             va="center", ha="left", fontsize=7.2,
             color=st.P2_DARK, fontweight="semibold")

    # --- Panel B: what each informative group actually requires, one cell per
    # group. 26 + 8 + 5 = 39 from the preserved per-candidate manifest. A
    # waffle keeps the unit discrete (these are 39 individual groups, not a
    # continuous share) and the two orange steps are one ordered ramp: how far
    # malformed-avoidance alone carries the group.
    # Local softened copper for the dominant block: the shared P2 rust pulled
    # too much attention at 26 cells; the mapping (warm = form-driven) is
    # unchanged and the shared palette is untouched.
    RUST_SOFT = "#AD6A3E"
    LIGHT_ORANGE = "#DEAC85"
    segs = [
        (26, "malformed-avoidance alone suffices", RUST_SOFT),
        (8, "malformed present, plus a well-formed wrong candidate", LIGHT_ORANGE),
        (5, "pure content-quality choice", st.P1),
    ]
    cells = [color for count, _, color in segs for _ in range(count)]
    NCOL = 13
    for idx, color in enumerate(cells):
        r, c = divmod(idx, NCOL)
        ax2.add_patch(FancyBboxPatch((c + 0.10, 2 - r + 0.10), 0.80, 0.80,
                                     boxstyle="round,pad=0,rounding_size=0.16",
                                     facecolor=color, edgecolor="none", zorder=3))
    ax2.set_xlim(0, NCOL)
    ax2.set_ylim(-2.75, 3.30)
    ax2.set_aspect("auto")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.grid(visible=False)
    for s in ax2.spines.values():
        s.set_visible(False)
    ax2.set_title("B     What decides each group", loc="left", fontsize=9.6,
                  pad=18)
    ax2.text(0.0, 1.015, "one cell per group · no malformed candidate is ever "
             "scored correct", transform=ax2.transAxes, ha="left", va="bottom",
             fontsize=7.4, color=st.MUTED)
    ys = [-0.55, -1.10, -1.65]
    for (count, label, color), y in zip(segs, ys):
        ax2.add_patch(FancyBboxPatch((0.10, y - 0.175), 0.62, 0.38,
                                     boxstyle="round,pad=0,rounding_size=0.10",
                                     facecolor=color, edgecolor="none", zorder=3,
                                     clip_on=False))
        ax2.text(1.00, y, f"{count}", va="center", ha="left", fontsize=7.6,
                 color=st.INK, fontweight="semibold")
        ax2.text(1.90, y, label, va="center", ha="left", fontsize=7.2,
                 color=st.INK)
    ax2.text(0.10, -2.42, "Content-only: 5/5 vs 3.75 expected — suggestive, "
             "underpowered", ha="left", va="center",
             fontsize=7.2, color=st.P1_DARK, fontweight="semibold")

    fig.suptitle("Terminal selection is established on output form; "
                 "this pool leaves content-sensitivity open",
                 x=.012, ha="left", y=1.165, va="bottom", fontsize=10.4, fontweight="semibold")
    fig.text(.012, 1.098, "The endpoint clears its exact null decisively; two thirds "
             "of the informative groups are won by avoiding a non-committing candidate.",
             ha="left", fontsize=8.0, color=st.MUTED)
    fig.text(.012, 1.040, "The content question is resolved by the all-well-formed pool (§8.6).",
             ha="left", fontsize=8.0, color=st.MUTED)
    save(fig, "fig_terminal_selection.pdf")


# --------------------------------------------------------------------------
# Figure 1 - the two evaluation traps, as a schematic
# --------------------------------------------------------------------------

def fig_audit_schematic() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.45, 2.78),
                                   gridspec_kw={"wspace": 0.16})
    for ax in (ax1, ax2):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

    box = {"boxstyle": "round,pad=0.32", "linewidth": 0.8}

    # --- Left: source-item leakage across a row-level split. -----------------
    ax1.text(0, 9.9, "Source-item leakage", ha="left", va="top", fontsize=9.6,
             fontweight="semibold", color=st.INK)
    ax1.text(0, 9.0, "rows constructed from one source item, then split by row",
             ha="left", va="top", fontsize=7.3, color=st.MUTED)
    # The split boundary, drawn in two segments so it never crosses the source
    # box or the leak arrow's arc.
    ax1.text(2.4, 7.80, "TRAIN", ha="center", va="bottom", fontsize=7.2,
             color=st.MUTED, fontweight="semibold")
    ax1.text(7.7, 7.80, "EVALUATION", ha="center", va="bottom", fontsize=7.2,
             color=st.MUTED, fontweight="semibold")
    ax1.plot([5.05, 5.05], [7.58, 8.22], color=st.RULE, lw=1.1, ls=(0, (3, 3)),
             zorder=1)
    ax1.plot([5.05, 5.05], [5.42, 6.45], color=st.RULE, lw=1.1, ls=(0, (3, 3)),
             zorder=1)
    # The shared source item: the unit that should have been split, and is not.
    ax1.text(5.05, 7.05, "source item  $x_i$", ha="center", va="center",
             fontsize=8.2, color=st.P1_DARK, fontweight="semibold",
             bbox={**box, "facecolor": st.P1_PALE, "edgecolor": st.P1})
    # Its two constructed rows, one on each side of the boundary.
    ax1.annotate("", xy=(2.9, 5.62), xytext=(4.30, 6.70),
                 arrowprops={"arrowstyle": "-|>", "color": st.MUTED, "lw": .9})
    ax1.annotate("", xy=(7.2, 5.62), xytext=(5.80, 6.70),
                 arrowprops={"arrowstyle": "-|>", "color": st.MUTED, "lw": .9})
    ax1.text(2.4, 5.25, "row $+\\Delta_i$", ha="center", va="center", fontsize=8.2,
             color=st.INK, bbox={**box, "facecolor": "white", "edgecolor": st.RULE})
    ax1.text(7.7, 5.25, "row $-\\Delta_i$", ha="center", va="center", fontsize=8.2,
             color=st.INK, bbox={**box, "facecolor": "white", "edgecolor": st.RULE})
    # The leak: an exactly antisymmetric scorer carries the memorised score
    # across the boundary with certainty. The arc dips BELOW the row boxes and
    # the formula sits underneath it, in clear space.
    ax1.add_patch(FancyArrowPatch((3.35, 4.65), (6.75, 4.65),
                                  connectionstyle="arc3,rad=0.32",
                                  arrowstyle="-|>", mutation_scale=9,
                                  color=st.P2, lw=1.2, zorder=4))
    ax1.text(5.05, 3.42, "$w^{\\top}(-\\Delta_i) = -\\,w^{\\top}\\Delta_i$",
             ha="center", va="center", fontsize=8.8, color=st.P2_DARK,
             fontweight="semibold")
    ax1.plot([0.30, 9.90], [2.62, 2.62], color=st.RULE, lw=0.8)
    ax1.text(0.30, 2.02, "exact antisymmetry transfers the memorised score "
             "with certainty", ha="left", va="center", fontsize=7.0,
             color=st.P2_DARK)
    ax1.text(0.30, 0.90, "$P$(sibling in training) $= 1-(1-f)^{\\,m-1} = 0.80$"
             "  at  $f = 0.8$, $m = 2$",
             ha="left", va="center", fontsize=7.0, color=st.MUTED)

    # --- Right: a fixed-order score decomposes into signal plus prior. --------
    ax2.text(0, 9.9, "Presentation-order prior", ha="left", va="top",
             fontsize=9.6, fontweight="semibold", color=st.INK)
    ax2.text(0, 9.0, "the split is clean; the label crosses through the protocol",
             ha="left", va="top", fontsize=7.3, color=st.MUTED)
    # A stacked decomposition of one fixed-order score. Widths are the audited
    # magnitudes: the symmetric component averages 1.50x the antisymmetric one.
    # Flat fills separated by a surface gap; identity is carried by position
    # and the ink labels under each segment, not by coloured text.
    x0, y0, h = 0.30, 6.00, 0.95
    w_anti, w_sym, gap = 3.55, 5.33, 0.12
    ax2.annotate("", xy=(x0 + w_anti + gap + w_sym, y0 + h + 0.34),
                 xytext=(x0, y0 + h + 0.34),
                 arrowprops={"arrowstyle": "<->", "color": st.MUTED, "lw": .8})
    ax2.text(x0 + (w_anti + gap + w_sym) / 2, y0 + h + 0.56,
             "one fixed-order accuracy score:  $\\mathrm{sign}(s_{\\mathrm{anti}} "
             "+ s_{\\mathrm{sym}})$", ha="center", va="bottom", fontsize=7.5,
             color=st.INK)
    ax2.add_patch(Rectangle((x0, y0), w_anti, h, facecolor=st.P1,
                            edgecolor="none", zorder=3))
    ax2.add_patch(Rectangle((x0 + w_anti + gap, y0), w_sym, h, facecolor=st.P2,
                            edgecolor="none", zorder=3))
    ax2.text(x0, y0 - 0.30, "$s_{\\mathrm{anti}}$ · relational signal",
             ha="left", va="top", fontsize=7.2, color=st.INK)
    ax2.text(x0 + w_anti + gap + w_sym, y0 - 0.30,
             "$s_{\\mathrm{sym}}$ · order prior  ($1.50\\times$ larger)",
             ha="right", va="top", fontsize=7.2, color=st.INK)
    # What each metric can and cannot see.
    ax2.text(0.30, 4.10, "flip correlation $-0.925$ — insensitive to a constant "
             "offset,", ha="left", va="center", fontsize=7.1, color=st.MUTED)
    ax2.text(0.30, 3.42, "so it reports nothing wrong", ha="left", va="center",
             fontsize=7.1, color=st.MUTED)
    ax2.plot([0.30, 9.90], [2.62, 2.62], color=st.RULE, lw=0.8)
    ax2.text(0.30, 2.02, "fixed-order  0.948", ha="left", va="center",
             fontsize=8.2, color=st.MUTED)
    ax2.text(3.88, 2.02, "$\\rightarrow$", ha="left", va="center", fontsize=8.6,
             color=st.MUTED)
    ax2.text(4.56, 2.02, "strict antisymmetrized  0.639", ha="left", va="center",
             fontsize=8.0, color=st.INK, fontweight="semibold")
    ax2.text(0.30, 0.90, "a 50% training-time swap protocol did not prevent this;\n"
             "selecting the checkpoint on fixed-order accuracy banked it",
             ha="left", va="center", fontsize=7.0, color=st.P2_DARK,
             linespacing=1.30)

    fig.subplots_adjust(top=0.97, bottom=0.02, left=0.03, right=0.99)
    fig.suptitle("Two evaluation traps, each invisible to the other's diagnostic",
                 x=.012, ha="left", y=1.005, va="bottom", fontsize=10.4,
                 fontweight="semibold")
    save(fig, "fig_audit_schematic.pdf")


def fig_splice() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.35, 2.72),
                                   gridspec_kw={"wspace": 0.34})
    loops = np.arange(1, 5)
    values = [13, 38, 63, 88]
    ax1.plot(loops, values, color=st.P1, marker="o")
    ax1.fill_between(loops, values, color=st.P1_PALE, alpha=1)
    for x, value in zip(loops, values):
        ax1.text(x, value + 4, f"{value}%", ha="center", fontsize=8.2)
    ax1.set_xticks(loops, [f"loop {i}" for i in loops])
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Layer passes saved")
    ax1.set_title("Deeper fork boundary", loc="left")
    ax1.grid(axis="x", visible=False)
    st.panel_label(ax1, "A")

    branches = [2, 4, 8]
    savings = [32, 47, 55]
    bars = ax2.bar([str(k) for k in branches], savings, color=st.P1, width=.55)
    st.bar_label(ax2, bars, fmt="{:.0f}%", dy=2)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("Branches K")
    ax2.set_title("More shared branches", loc="left")
    ax2.grid(axis="x", visible=False)
    st.panel_label(ax2, "B")
    fig.suptitle("Bit-exact suffix recomputation · savings versus full perturbed prefills",
                 x=.01, ha="left", y=1.07, fontsize=10.5, fontweight="semibold")
    save(fig, "fig_splice_savings.pdf")


def fig_two_null() -> None:
    # Two honestly-encoded strips share the x-axis. TOP: the random-direction
    # null as its EXACT Beta density (faithful analytic shape). BOTTOM: the
    # outcome-shaped shuffled-label null shown as an interval/quantile SUMMARY,
    # NOT a fitted density: the 10^4 raw draws were not preserved to the
    # plotting code, so only the mean and the observed percentile are drawn.
    # Encoding the two nulls differently also avoids any peak-height comparison.
    k, dimension = 344, 24576
    observed = 0.018296                      # reported 0.0183
    rand_mean = k / dimension                # 0.013997 -> 0.0140
    # Global shuffled-label null (10^4 draws): mean 0.022990 -> 0.0230, with the
    # observed value at its 3.44th percentile. Mean and percentile are taken from
    # the SAME null (no mixing); the domain-stratified variant (mean 0.0227,
    # observed at percentile 0.0) is reported in the prose and Appendix J.
    shuf_mean = 0.0230
    # Two equal-height aligned panels on one shared x-axis, so the two nulls
    # read as comparable visual objects and the disagreement is carried by the
    # geometry, not the annotation. The raw shuffled-label draws were never
    # preserved, so the lower null cannot be drawn as a density; what IS known
    # exactly is that 96.56% of its 10,000 draws lie above the observed value
    # (observed = 3.44th percentile), so that mass is drawn as a block whose
    # left edge is the observed line itself.
    xlo, xhi = .0092, .0252
    x = np.linspace(xlo, xhi, 1600)
    pdf = beta_dist.pdf(x, k / 2, (dimension - k) / 2)
    pdf = pdf / pdf.max()

    fig = plt.figure(figsize=(7.4, 2.95))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.14)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)

    # --- Top panel: random-direction null, exact Beta density. ---------------
    live = pdf > 1e-3
    ax0.fill_between(x[live], 0, pdf[live], color=st.P1_PALE, lw=0, zorder=2)
    ax0.plot(x[live], pdf[live], color=st.P1, lw=1.2, zorder=3)
    ax0.plot([rand_mean, rand_mean], [0, 1.0], color=st.P1_DARK, lw=.9,
             ls=(0, (1, 2)), zorder=3)
    ax0.text(xlo + .0003, 1.24, "Random-direction null · exact Beta density",
             ha="left", va="center", fontsize=7.9, color=st.P1_DARK,
             fontweight="semibold")
    ax0.text(xlo + .0003, 1.04, f"expectation {rand_mean:.4f} (dotted)",
             ha="left", va="center", fontsize=7.0, color=st.P1)
    ax0.plot([observed, observed], [0, 1.02], color=st.INK, lw=1.6, zorder=5,
             solid_capstyle="butt")
    ax0.scatter([observed], [1.02], marker="v", s=34, color=st.INK, zorder=6)
    ax0.text(observed, 1.22, "observed 0.0183", ha="center", va="center",
             fontsize=8.6, fontweight="semibold", color=st.INK)
    ax0.text(observed + .0005, .52, "99.99th pct — above this null",
             ha="left", va="center", fontsize=7.5, fontweight="semibold",
             color=st.P1_DARK)
    ax0.set_ylim(0, 1.42)
    ax0.set_yticks([])
    ax0.spines["left"].set_visible(False)
    ax0.tick_params(axis="x", labelbottom=False, length=0)
    ax0.grid(visible=False)

    # --- Bottom panel: shuffled-label null, drawn as its known mass. ---------
    ax1.axvspan(observed, xhi, ymin=0.05, ymax=0.78, color=st.P2_PALE, zorder=1)
    ax1.plot([shuf_mean, shuf_mean], [0.06, 0.90], color=st.P2_DARK, lw=.9,
             ls=(0, (1, 2)), zorder=3)
    ax1.scatter([shuf_mean], [.48], marker="D", s=44, color=st.P2_DARK,
                edgecolor="white", lw=.6, zorder=5)
    ax1.text(shuf_mean, .97, "mean 0.0230", ha="center", va="center",
             fontsize=7.1, color=st.P2_DARK)
    ax1.text(xlo + .0003, 1.22, "Shuffled-label null · empirical "
             "(10,000 draws)", ha="left", va="center", fontsize=7.9,
             color=st.P2_DARK, fontweight="semibold")
    ax1.text(observed + .0005, .48, "3.44th pct — below this null's mean:\n"
             "96.6% of draws lie right of the line", ha="left", va="center",
             fontsize=7.5, fontweight="semibold", color=st.P2_DARK,
             linespacing=1.3)
    ax1.plot([observed, observed], [0, 1.42], color=st.INK, lw=1.6, zorder=5,
             solid_capstyle="butt")
    ax1.set_ylim(0, 1.42)
    ax1.set_yticks([])
    ax1.spines["left"].set_visible(False)
    ax1.set_xlim(xlo, xhi)
    ax1.xaxis.set_major_formatter(lambda v, _: f"{v:.3f}")
    ax1.set_xticks([.010, .012, .014, .016, .018, .020, .022, .024])
    ax1.set_xlabel("Projection fraction of the outcome direction into the injection span")
    ax1.grid(visible=False)

    fig.suptitle("Two-null audit · the simplest one-dimensional explanation is not "
                 "supported", x=.012, ha="left", y=1.02, va="bottom", fontsize=10.4,
                 fontweight="semibold")
    fig.text(.012, .995, "One observed value, two nulls: above the whole "
             "random-direction density, inside the lower tail of the "
             "shuffled-label mass.", ha="left", va="top", fontsize=8.3,
             color=st.MUTED)
    fig.subplots_adjust(top=0.89, bottom=0.17, left=0.02, right=0.99)
    save(fig, "fig_two_null.pdf")


# --------------------------------------------------------------------------
# Figure 13 - the frozen conversions of §8.6: one positive curve, two nulls
# --------------------------------------------------------------------------
def fig_frozen_conversions() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.45, 5.70),
                             gridspec_kw={"wspace": 0.40, "hspace": 0.58})
    (ax1, ax4), (ax2, ax3) = axes

    # -- A: C1 selective prediction, Horizon pooled 4-shortcut arm ----------
    cov = np.array([.50, .55, .60, .65, .70, .75, .80, .85, .90, .95, 1.0])
    comb = np.array([.9609, .9438, .9348, .9298, .9256, .9109, .9043, .8966,
                     .8897, .8765, .8632])
    sc = np.array([.9251, .9112, .9022, .8897, .8837, .8870, .8819, .8793,
                   .8716, .8748, .8632])
    ax1.fill_between(cov, sc, comb, color=st.P1_PALE, zorder=1)
    ax1.plot(cov, sc, color=st.BASELINE, lw=1.7, zorder=2)
    ax1.plot(cov, comb, color=st.P1_DARK, lw=2.0, zorder=3)
    ax1.text(.503, .9680, "hidden + shortcuts", color=st.P1_DARK, fontsize=8.2,
             fontweight="semibold")
    ax1.text(.567, .8738, "shortcuts only", color=st.MUTED, fontsize=8.2)
    ax1.plot([.70, .70], [sc[4], comb[4]], color=st.P2_DARK, lw=1.1, zorder=4)
    ax1.text(.728, .9395, "+4.2 pts at 70% coverage", color=st.P2_DARK,
             fontsize=7.8, fontweight="semibold", va="center", ha="left")
    ax1.set_xlim(.49, 1.01)
    ax1.set_ylim(.855, .975)
    ax1.set_xticks([.5, .75, 1.0], ["50%", "75%", "100%"])
    ax1.set_xlabel("answer coverage")
    ax1.set_ylabel("selective accuracy")
    ax1.set_title("A · Abstention converts", loc="left", color=st.POSITIVE,
                  pad=14, fontsize=10.6)
    ax1.text(0, 1.035, "ΔAUARC CI > 0 in all four sealed arms",
             transform=ax1.transAxes, fontsize=7.8, color=st.MUTED)

    # -- B: C2 fixed-depth curves; allocation not signal-driven -------------
    d = np.array([1, 2, 3, 4])
    rltt = np.array([.38, .84, .86, .80])
    think = np.array([.18, .44, .64, .70])
    ax2.plot(d, rltt, color=st.P1_DARK, lw=2.0, marker="o", ms=4.6, zorder=3)
    ax2.plot(d, think, color=st.P1, lw=1.7, marker="^", ms=4.6, ls="--", zorder=2)
    ax2.text(2.02, .925, "RLTT", color=st.P1_DARK, fontsize=8.4,
             fontweight="semibold", va="center")
    ax2.text(2.82, .47, "Thinking", color=st.P1, fontsize=8.4, va="center")
    ax2.set_xlim(.8, 4.30)
    ax2.set_ylim(.10, .99)
    ax2.set_xticks(d)
    ax2.set_xlabel("fixed recurrent depth $d$")
    ax2.set_ylabel("held-out success")
    ax2.set_title("C · Allocation not signal-driven", loc="left",
                  color=st.INK, pad=14, fontsize=10.6)
    ax2.text(0, 1.035, "depth pays; tap $\\approx$ gate $\\approx$ random control",
             transform=ax2.transAxes, fontsize=7.8, color=st.MUTED)

    # -- C: C5 binding deltas with 95% CIs -----------------------------------
    labels = ["binding ($+\\alpha d$)", "shuffled control", "frozen base"]
    pts = np.array([0.008, -0.011, 0.044])
    lo = np.array([-0.036, -0.056, -0.006])
    hi = np.array([0.050, 0.036, 0.100])
    y = np.arange(3)[::-1]
    ax3.axvline(0, color=st.INK, lw=1.0, zorder=1)
    ax3.hlines(y, lo, hi, color=st.RULE, lw=2.6, zorder=2)
    ax3.scatter(pts, y, s=46, color=[st.P1_DARK, st.BASELINE, st.MUTED],
                edgecolor="white", lw=.7, zorder=3)
    for yi, p in zip(y, pts):
        ax3.text(p + (.007 if p >= 0 else -.007), yi + .34, f"{p:+.3f}",
                 ha="left" if p >= 0 else "right", fontsize=7.8, color=st.INK)
    ax3.set_yticks(y, labels)
    ax3.set_xlim(-.085, .125)
    ax3.set_xticks([-.05, 0, .05, .10])
    ax3.set_ylim(-.6, 2.75)
    ax3.set_xlabel("Δ success, inject − clean")
    ax3.set_title("D · Binding not detected", loc="left", color=st.INK,
                  pad=14, fontsize=10.6)
    ax3.text(0, 1.035, "every 95% CI crosses zero",
             transform=ax3.transAxes, fontsize=7.8, color=st.MUTED)
    ax3.grid(axis="y", visible=False)

    # -- B (top right): C4 all-well-formed terminal selection ----------------
    yb = np.arange(3)[::-1]
    vals = [0.844, 0.750, 0.648]
    labs = ["hidden-state\nselector  27/32", "surface-only\ncontrol  24/32",
            "matched-random\nexpectation"]
    cols = [st.P1_DARK, st.P2, st.BASELINE]
    ax4.hlines(yb, 0, vals, color=st.RULE, lw=4.4, zorder=1)
    ax4.scatter(vals, yb, s=62, color=cols, edgecolor="white", lw=.8, zorder=3)
    for yi, v in zip(yb, vals):
        ax4.text(v + .028, yi, f"{v:.3f}", va="center", fontsize=8.4,
                 fontweight="semibold", color=st.INK)
    ax4.set_yticks(yb, labs)
    ax4.set_xlim(0, 1.09)
    ax4.set_xticks([0, .25, .5, .75, 1.0])
    ax4.set_ylim(-.75, 2.6)
    ax4.set_xlabel("forced top-1 on 32 informative all-well-formed groups")
    ax4.set_title("B · Content selection converts", loc="left",
                  color=st.POSITIVE, pad=14, fontsize=10.6)
    ax4.text(0, 1.035, "hidden vs random $p=0.0086$; hidden vs surface unresolved",
             transform=ax4.transAxes, fontsize=7.8, color=st.MUTED)
    ax4.grid(axis="y", visible=False)

    fig.suptitle("Four of the five sealed conversions, drawn: the two "
                 "decision-level gains, and two generative non-conversions",
                 x=.012, ha="left", y=1.055, va="bottom", fontsize=11.2,
                 fontweight="semibold")
    fig.text(.012, 1.026, "A: risk–coverage on Horizon (pooled, 4-shortcut arm). "
             "B: the all-well-formed pool (C4). C: the fixed-depth tables behind "
             "the allocation null. D: the sign-conditioned LoRA pilot.",
             ha="left", fontsize=8.6, color=st.MUTED)
    save(fig, "fig_frozen_conversions.pdf")


if __name__ == "__main__":
    fig_audit_schematic()        # Figure 1
    fig_crossloop()              # Figure 2
    fig_family_replication()     # Figure 3 (v3)
    fig_domain_transfer()        # Figure 4
    fig_preanswer()              # Figure 5
    fig_survival_ablation()      # Figure 6
    fig_s3b2()                   # Figure 7
    fig_terminal_selection()     # Figure 8
    fig_splice()                 # Figure 10
    fig_two_null()               # Figure 12
    fig_frozen_conversions()     # Figure 13 (§8.6)
