"""Cross-loop early-layer v1 — figures (vector SVG + PNG preview).

1. local-refit heatmap (loop x layer, macro top-1)
2. frozen-transfer matrices per layer (absolute target macro; retention vs target refit)
3. readability vs remaining same-loop depth
4. transplant gap (target refit - frozen transplant)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bg_xloop_early_v1_common as X

RES = X.read_json(X.RUN_ROOT / "train_eval_results.json")
STATS = X.read_json(X.RUN_ROOT / "bootstrap_stats.json")
CHANCE = RES["chance"]["macro"]


def save(fig, name: str) -> None:
    for ext in ("svg", "png"):
        fig.savefig(X.FIG_ROOT / f"{name}.{ext}", bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
    plt.close(fig)


def cell_macro(cn: str) -> float:
    return RES["cells"][cn]["heldout_macro_top1"]


def heatmap_refit() -> None:
    layers = [8, 16, 24]
    data = np.array([[cell_macro(X.cell_name(l, u)) for u in X.LOOPS] for l in layers])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    im = ax.imshow(data, cmap="viridis", aspect="auto",
                   vmin=min(CHANCE, data.min()), vmax=max(data.max(), cell_macro("36_L4")))
    for i, l in enumerate(layers):
        for j in range(4):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                    color="white" if data[i, j] < data.max() * 0.92 else "black", fontsize=9)
    ax.set_xticks(range(4), [f"L{u}" for u in X.LOOPS])
    ax.set_yticks(range(len(layers)), [f"layer {l}" for l in layers])
    ax.set_title(f"Local-refit macro top-1 (heldout; chance {CHANCE:.3f}; "
                 f"refs L4_36={cell_macro('36_L4'):.3f}, L4_47={cell_macro('47_L4'):.3f})", fontsize=9)
    fig.colorbar(im, ax=ax, label="macro top-1")
    save(fig, "fig1_local_refit_heatmap")


def transfer_matrices() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.5))
    for col, layer in enumerate((8, 16, 24)):
        abs_m = np.full((4, 4), np.nan)
        gap_m = np.full((4, 4), np.nan)
        for u in X.LOOPS:
            abs_m[u - 1, u - 1] = cell_macro(X.cell_name(layer, u))
            gap_m[u - 1, u - 1] = 0.0
        for (l, u, v) in X.TRANSFERS:
            if l != layer:
                continue
            tr = RES["transfers"][f"{layer}:L{u}->L{v}"]
            abs_m[u - 1, v - 1] = tr["target_macro_top1"]
            gap_m[u - 1, v - 1] = -tr["delta_vs_target_refit"]
        for row, (m, title, cmap, kw) in enumerate((
                (abs_m, f"layer {layer}: frozen-transplant macro top-1", "viridis",
                 dict(vmin=CHANCE - 0.05, vmax=np.nanmax(abs_m))),
                (gap_m, f"layer {layer}: refit minus transplant (rotation gap)", "magma", dict()))):
            ax = axes[row, col]
            im = ax.imshow(m, cmap=cmap, aspect="auto", **kw)
            for i in range(4):
                for j in range(4):
                    if np.isfinite(m[i, j]):
                        ax.text(j, i, f"{m[i, j]:.3f}", ha="center", va="center", fontsize=8,
                                color="white" if m[i, j] < np.nanmax(m) * 0.9 else "black")
            ax.set_xticks(range(4), [f"L{v}" for v in X.LOOPS])
            ax.set_yticks(range(4), [f"src L{u}" for u in X.LOOPS])
            ax.set_title(title, fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle("Frozen cross-loop transplant (diagonal = source self / local refit)", fontsize=11)
    fig.tight_layout()
    save(fig, "fig2_transfer_matrices")


def depth_plot() -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    markers = {1: "o", 2: "s", 3: "^", 4: "D"}
    ref_layers = {8: 8, 16: 16, 24: 24, 36: 36, 47: 47}
    for (layer, loop) in X.REFIT_CELLS:
        cn = X.cell_name(layer, loop)
        remaining = 48 - ref_layers[layer]
        ax.scatter(remaining, cell_macro(cn), marker=markers[loop], s=55,
                   color=plt.cm.viridis((loop - 1) / 3), zorder=3)
        ax.annotate(cn, (remaining, cell_macro(cn)), textcoords="offset points",
                    xytext=(5, 4), fontsize=7)
    ax.axhline(CHANCE, color="gray", ls="--", lw=1, label=f"matched chance {CHANCE:.3f}")
    for loop in X.LOOPS:
        ax.scatter([], [], marker=markers[loop], color=plt.cm.viridis((loop - 1) / 3),
                   label=f"loop L{loop}")
    ax.set_xlabel("remaining same-loop physical layers (48 - layer)")
    ax.set_ylabel("local-refit macro top-1 (heldout)")
    ax.set_title("Readability vs remaining downstream depth", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    save(fig, "fig3_readability_vs_depth")


def gap_plot() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    xs, ys, labels = [], [], []
    for i, (layer, u, v) in enumerate(X.TRANSFERS):
        tr = RES["transfers"][f"{layer}:L{u}->L{v}"]
        xs.append(i)
        ys.append(-tr["delta_vs_target_refit"])
        labels.append(f"{layer}:L{u}→L{v}")
    colors = ["#4c72b0" if l.startswith("8") else "#dd8452" if l.startswith("16") else "#55a868"
              for l in labels]
    ax.bar(xs, ys, color=colors)
    ax.set_xticks(xs, labels, rotation=60, fontsize=7, ha="right")
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(0.03, color="gray", ls="--", lw=0.8, label="0.03 stability margin")
    ax.set_ylabel("target-local refit − frozen transplant (macro top-1)")
    ax.set_title("Transplant gap (coordinate rotation) by layer and loop pair", fontsize=10)
    ax.legend(fontsize=8)
    save(fig, "fig4_transplant_gap")


def main() -> int:
    heatmap_refit()
    transfer_matrices()
    depth_plot()
    gap_plot()
    print("figures written to", X.FIG_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
