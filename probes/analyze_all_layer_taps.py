"""Offline analysis for probe_all_layers_centered.py.

Ranks individual insertion taps, checks early-layer agreement with boundary,
and runs a small greedy multi-tap centered-score combination search.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-json", default="artifacts/reports/evaluator/probe_all_layers_centered.json")
    p.add_argument("--output-json", default="artifacts/reports/evaluator/probe_all_layers_analysis.json")
    p.add_argument("--output-md", default="artifacts/reports/evaluator/probe_all_layers_analysis.md")
    p.add_argument("--max-candidates", type=int, default=80)
    p.add_argument("--greedy-steps", type=int, default=8)
    return p.parse_args()


def arr(values: Iterable[float]) -> np.ndarray:
    return np.array(list(values), dtype=np.float64)


def centered_scores(data: Dict, config: str) -> np.ndarray:
    s = arr(data["raw_scores"][config])
    sf = arr(data["raw_scores_flipped"][config])
    return s - sf


def acc_from_centered(centered: np.ndarray) -> float:
    valid = np.isfinite(centered)
    if not valid.any():
        return 0.0
    return float(np.mean(centered[valid] > 0))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if a.size == 0 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def sign_agreement(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if not valid.any():
        return 0.0
    return float(np.mean(np.sign(a[valid]) == np.sign(b[valid])))


def config_layer(meta: Dict) -> int:
    layer = meta.get("layer")
    return -1 if layer is None else int(layer)


def select_candidate_names(data: Dict, max_candidates: int) -> List[str]:
    per_config = data["per_config"]
    meta = data["config_metadata"]
    candidates = []
    for name, row in per_config.items():
        m = meta[name]
        if not str(name).startswith("layer_"):
            continue
        if row.get("n_valid", 0) == 0:
            continue
        # Keep all natural taps and the stronger loop taps; this gives greedy
        # search room without letting 480 near-duplicates dominate.
        if m.get("tap_kind") == "layer_natural" or row["centered_acc"] >= 0.58:
            candidates.append(name)
    candidates.sort(
        key=lambda name: (
            -float(per_config[name]["centered_acc"]),
            -float(per_config[name]["flipped_acc"]),
            config_layer(meta[name]),
        )
    )
    return candidates[:max_candidates]


def greedy_combo(data: Dict, candidates: List[str], steps: int) -> List[Dict]:
    selected: List[str] = []
    selected_sum = None
    history = []
    current_acc = -1.0

    for _ in range(steps):
        best = None
        best_scores = None
        for name in candidates:
            if name in selected:
                continue
            scores = centered_scores(data, name)
            combo = scores if selected_sum is None else selected_sum + scores
            acc = acc_from_centered(combo)
            if best is None or acc > best[1]:
                best = (name, acc)
                best_scores = combo
        if best is None or best_scores is None:
            break
        if best[1] <= current_acc + 1e-12:
            break
        selected.append(best[0])
        selected_sum = best_scores
        current_acc = best[1]
        history.append(
            {
                "step": len(selected),
                "added": best[0],
                "centered_acc": float(best[1]),
                "members": list(selected),
            }
        )
    return history


def top_rows(data: Dict, predicate, key: str, n: int = 20) -> List[Tuple[str, Dict]]:
    rows = [(name, row) for name, row in data["per_config"].items() if predicate(name, row)]
    rows.sort(key=lambda item: (-float(item[1][key]), -float(item[1]["flipped_acc"])))
    return rows[:n]


def write_md(data: Dict, analysis: Dict, path: Path) -> None:
    def table(title: str, rows: List[Tuple[str, Dict]]) -> List[str]:
        lines = [
            f"## {title}",
            "",
            "| config | canonical | centered | flipped | strict | corr | bias/signal |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name, row in rows:
            lines.append(
                f"| {name} | {row['canonical_acc']:.3f} | {row['centered_acc']:.3f} | "
                f"{row['flipped_acc']:.3f} | {row['strict_sign_reversal_rate']:.3f} | "
                f"{row['antisym_corr']:.3f} | {row['bias_to_signal']:.2f} |"
            )
        lines.append("")
        return lines

    lines = ["# All-Layer Tap Analysis", ""]
    lines.extend(table("Top Natural Taps By Centered Accuracy", analysis["top_natural_centered"]))
    lines.extend(table("Top Per-Loop Taps By Centered Accuracy", analysis["top_loop_centered"]))
    lines.extend(table("Top Natural Taps By Canonical Accuracy", analysis["top_natural_canonical"]))

    lines.extend(
        [
            "## Earliest Useful Natural Normed Layers",
            "",
            "| threshold | first_layer | config | centered | canonical | boundary_sign_agree | boundary_corr |",
            "|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["earliest_thresholds"]:
        lines.append(
            f"| {row['threshold']:.3f} | {row['first_layer']} | {row['config']} | "
            f"{row['centered_acc']:.3f} | {row['canonical_acc']:.3f} | "
            f"{row['boundary_sign_agreement']:.3f} | {row['boundary_corr']:.3f} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Greedy Multi-Tap Combination",
            "",
            "| step | added | centered_acc | members |",
            "|---:|---|---:|---|",
        ]
    )
    for row in analysis["greedy_combo"]:
        lines.append(
            f"| {row['step']} | {row['added']} | {row['centered_acc']:.3f} | "
            f"{', '.join(row['members'])} |"
        )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    per_config = data["per_config"]
    meta = data["config_metadata"]

    boundary_centered = centered_scores(data, "boundary_natural")

    natural_normed = [
        name for name, m in meta.items()
        if m.get("tap_kind") == "layer_natural" and bool(m.get("normed"))
    ]
    threshold_rows = []
    for threshold in (0.55, 0.58, 0.60, 0.62, 0.65):
        found = None
        for name in sorted(natural_normed, key=lambda n: int(meta[n]["layer"])):
            row = per_config[name]
            if row["centered_acc"] >= threshold:
                cs = centered_scores(data, name)
                found = {
                    "threshold": threshold,
                    "first_layer": int(meta[name]["layer"]),
                    "config": name,
                    "centered_acc": float(row["centered_acc"]),
                    "canonical_acc": float(row["canonical_acc"]),
                    "boundary_sign_agreement": sign_agreement(cs, boundary_centered),
                    "boundary_corr": pearson(cs, boundary_centered),
                }
                break
        if found is None:
            found = {
                "threshold": threshold,
                "first_layer": None,
                "config": "",
                "centered_acc": 0.0,
                "canonical_acc": 0.0,
                "boundary_sign_agreement": 0.0,
                "boundary_corr": 0.0,
            }
        threshold_rows.append(found)

    candidate_names = select_candidate_names(data, args.max_candidates)
    combo = greedy_combo(data, candidate_names, args.greedy_steps)

    analysis = {
        "input_json": args.input_json,
        "examples_evaluated": data.get("examples_evaluated"),
        "top_natural_centered": top_rows(
            data,
            lambda name, row: name.startswith("layer_") and meta[name].get("tap_kind") == "layer_natural",
            "centered_acc",
        ),
        "top_loop_centered": top_rows(
            data,
            lambda name, row: name.startswith("layer_") and meta[name].get("tap_kind") == "layer_loop",
            "centered_acc",
        ),
        "top_natural_canonical": top_rows(
            data,
            lambda name, row: name.startswith("layer_") and meta[name].get("tap_kind") == "layer_natural",
            "canonical_acc",
        ),
        "earliest_thresholds": threshold_rows,
        "greedy_candidates": candidate_names,
        "greedy_combo": combo,
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    write_md(data, analysis, out_md)

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print("\nTop natural centered:")
    for name, row in analysis["top_natural_centered"][:12]:
        print(f"{name:>30} center={row['centered_acc']:.3f} canon={row['canonical_acc']:.3f}")
    print("\nGreedy combo:")
    for row in combo:
        print(f"step {row['step']}: +{row['added']} center={row['centered_acc']:.3f}")


if __name__ == "__main__":
    main()
