"""Offline bias decomposition for pairwise evaluator probe JSON files.

Consumes probe outputs that contain:
  - raw_scores[config] = score(chosen, rejected)
  - raw_scores_flipped[config] = score(rejected, chosen)

Writes a compact JSON plus a Markdown table with canonical, flipped, strict
sign-reversal, antisymmetry-correlation, centered/debiased, and bias metrics.
No model or GPU is required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


DEFAULT_INPUTS = [
    "artifacts/reports/evaluator/probe_evaluator_v4_redo_masked.json",
    "artifacts/reports/evaluator/probe_evaluator_v5_extended.json",
    "artifacts/reports/evaluator/probe_evaluator_math.json",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="*", default=DEFAULT_INPUTS)
    p.add_argument("--output-json", default="artifacts/reports/evaluator/bias_decomposition_all.json")
    p.add_argument("--output-md", default="artifacts/reports/evaluator/bias_decomposition_all.md")
    return p.parse_args()


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def metric_block(normal: Iterable[float], flipped: Iterable[float]) -> Dict[str, Any]:
    s = np.array(list(normal), dtype=np.float64)
    sf = np.array(list(flipped), dtype=np.float64)
    valid = np.isfinite(s) & np.isfinite(sf)
    s = s[valid]
    sf = sf[valid]
    if s.size == 0:
        return {"n": 0}

    centered = s - sf
    bias = 0.5 * (s + sf)
    antisym = 0.5 * (s - sf)
    strict = np.sign(s) != np.sign(sf)
    canonical_acc = np.mean(s > 0)
    flipped_acc = np.mean(sf < 0)
    centered_acc = np.mean(centered > 0)

    return {
        "n": int(s.size),
        "canonical_acc": float(canonical_acc),
        "flipped_acc": float(flipped_acc),
        "normal_pos_rate": float(np.mean(s > 0)),
        "flipped_pos_rate": float(np.mean(sf > 0)),
        "strict_sign_reversal_rate": float(np.mean(strict)),
        "same_sign_rate": float(1.0 - np.mean(strict)),
        "antisym_corr": pearson(s, -sf),
        "centered_acc": float(centered_acc),
        "centered_mean": float(np.mean(centered)),
        "centered_std": float(np.std(centered)),
        "bias_mean": float(np.mean(bias)),
        "bias_std": float(np.std(bias)),
        "antisym_mean": float(np.mean(antisym)),
        "antisym_std": float(np.std(antisym)),
        "bias_to_signal": float(abs(np.mean(bias)) / (abs(np.mean(antisym)) + 1e-8)),
        "mean_sum": float(np.mean(s + sf)),
        "mean_abs_sum": float(np.mean(np.abs(s + sf))),
        "score_mean": float(np.mean(s)),
        "score_std": float(np.std(s)),
        "flipped_mean": float(np.mean(sf)),
        "flipped_std": float(np.std(sf)),
    }


def load_rows(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("raw_scores", {})
    raw_flip = data.get("raw_scores_flipped", {})
    rows: List[Dict[str, Any]] = []
    for config in sorted(raw):
        if config not in raw_flip:
            continue
        metrics = metric_block(raw[config], raw_flip[config])
        metrics.update(
            {
                "source": str(path),
                "config": config,
                "examples_evaluated": data.get("examples_evaluated"),
                "dataset": data.get("dataset"),
                "split": data.get("split"),
            }
        )
        rows.append(metrics)
    return rows


def fmt_float(value: Any, places: int = 3) -> str:
    if not isinstance(value, (float, int)):
        return ""
    return f"{value:.{places}f}"


def write_markdown(rows: List[Dict[str, Any]], path: Path) -> None:
    columns = [
        "source",
        "config",
        "n",
        "canonical_acc",
        "flipped_acc",
        "centered_acc",
        "strict_sign_reversal_rate",
        "antisym_corr",
        "bias_mean",
        "antisym_mean",
        "bias_to_signal",
        "mean_sum",
    ]
    lines = [
        "# Pairwise Bias Decomposition",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col)
            if col in {"source", "config"}:
                vals.append(str(value).replace("|", "\\|"))
            elif col == "n":
                vals.append(str(value))
            else:
                vals.append(fmt_float(value))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, Any]] = []
    for input_path in args.inputs:
        path = Path(input_path)
        if not path.exists():
            print(f"Skipping missing input: {path}")
            continue
        rows.extend(load_rows(path))

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    write_markdown(rows, out_md)

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(f"Rows: {len(rows)}")

    interesting = [
        "boundary",
        "only_loop_2",
        "loop2_norm_x4",
        "loop2_norm_x8",
        "mean_loops_quadnorm",
        "quadnorm_each_natural_seq",
    ]
    for row in rows:
        if row.get("config") in interesting and "probe_v5_extended" in row.get("source", ""):
            print(
                f"{row['config']:>28} "
                f"canon={row['canonical_acc']:.3f} "
                f"centered={row['centered_acc']:.3f} "
                f"strict={row['strict_sign_reversal_rate']:.3f} "
                f"corr={row['antisym_corr']:+.3f} "
                f"bias/signal={row['bias_to_signal']:.2f}"
            )


if __name__ == "__main__":
    main()
