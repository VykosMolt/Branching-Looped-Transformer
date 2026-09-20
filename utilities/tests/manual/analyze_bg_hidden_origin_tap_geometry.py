"""Compare hidden-origin tap geometry to old frozen taps and empirical diffs."""
from __future__ import annotations

import math
import time
from collections import defaultdict
from statistics import mean
from typing import Any

import torch

from bg_hidden_origin_tap_common import (
    OUT_ROOT,
    PROBE_ROOT,
    cosine,
    direction_from_state_dict,
    ensure_out_root,
    load_json,
    md_table,
    rate,
    rel,
    write_json,
    write_md,
)


DATASET_PT = OUT_ROOT / "hidden_origin_tap_dataset.pt"
HEADS_PT = OUT_ROOT / "hidden_origin_tap_heads.pt"
EVAL_JSON = OUT_ROOT / "heldout_eval.json"
OUT_JSON = OUT_ROOT / "geometry_analysis.json"
OUT_MD = OUT_ROOT / "geometry_analysis.md"


def load_old_head_rows() -> list[dict[str, Any]]:
    rows = []
    for path in (
        PROBE_ROOT / "mixed_domain_tiny_heads_2026-05-17.pt",
        PROBE_ROOT / "bg_head_registry_2026-05-17.pt",
    ):
        if not path.exists():
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        for row in list(payload.get("heads") or []):
            state = row.get("state_dict")
            direction = direction_from_state_dict(state) if isinstance(state, dict) else None
            if isinstance(direction, torch.Tensor):
                rows.append({
                    "source": rel(path),
                    "group": row.get("head_group") or row.get("head_family") or row.get("group"),
                    "config": row.get("config"),
                    "architecture": row.get("architecture"),
                    "direction": direction,
                })
    return rows


def empirical_mean_diffs(dataset: dict[str, Any]) -> dict[str, torch.Tensor]:
    pairs = [pair for pair in list(dataset.get("pairs") or []) if pair.get("split") == "train"]
    out = {}
    by_config: dict[str, list[torch.Tensor]] = defaultdict(list)
    for pair in pairs:
        for config, feats in (pair.get("features") or {}).items():
            by_config[config].append((feats["preferred"] - feats["rejected"]).detach().cpu().to(torch.float32))
    for config, vals in by_config.items():
        if vals:
            out[config] = torch.stack(vals, dim=0).mean(dim=0)
    return out


def head_id(row: dict[str, Any]) -> str:
    m = row.get("metrics", {})
    return f"{row['config']}::{row['architecture']}::seed={m.get('seed')}::lr={m.get('lr')}"


def correlation(xs: list[float], ys: list[float]) -> float:
    vals = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(vals) < 3:
        return float("nan")
    xt = torch.tensor([v[0] for v in vals], dtype=torch.float32)
    yt = torch.tensor([v[1] for v in vals], dtype=torch.float32)
    if float(xt.std(unbiased=False).item()) <= 1e-8 or float(yt.std(unbiased=False).item()) <= 1e-8:
        return float("nan")
    return float(torch.corrcoef(torch.stack([xt, yt]))[0, 1].item())


def main() -> int:
    ensure_out_root()
    started = time.time()
    if not HEADS_PT.exists() or not DATASET_PT.exists():
        payload = {"BG_HIDDEN_ORIGIN_TAP_GEOMETRY_VERDICT": "INCONCLUSIVE", "blocker": "missing heads or dataset"}
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# Hidden-Origin Tap Geometry", "", "BG_HIDDEN_ORIGIN_TAP_GEOMETRY_VERDICT = INCONCLUSIVE"])
        print("BG_HIDDEN_ORIGIN_TAP_GEOMETRY_VERDICT = INCONCLUSIVE", flush=True)
        return 1
    heads_payload = torch.load(HEADS_PT, map_location="cpu", weights_only=False)
    dataset = torch.load(DATASET_PT, map_location="cpu", weights_only=False)
    eval_payload = load_json(EVAL_JSON, {}) or {}
    new_heads = [row for row in list(heads_payload.get("heads") or []) if row.get("flip_diagnostics", {}).get("passes")]
    old_heads = load_old_head_rows()
    mean_diffs = empirical_mean_diffs(dataset)

    cosine_rows = []
    for row in new_heads:
        ndir = row.get("direction")
        if not isinstance(ndir, torch.Tensor):
            ndir = direction_from_state_dict(row.get("state_dict") or {})
        if not isinstance(ndir, torch.Tensor):
            continue
        cfg = row["config"]
        empirical = mean_diffs.get(cfg)
        cosine_rows.append({
            "new_head_id": head_id(row),
            "comparison": "empirical_hidden_origin_mean_diff",
            "old_group": None,
            "old_config": cfg,
            "old_architecture": None,
            "cosine": cosine(ndir, empirical) if isinstance(empirical, torch.Tensor) else float("nan"),
        })
        for old in old_heads:
            if old["direction"].numel() != ndir.numel():
                continue
            label = "old_objective_mixed" if old.get("group") == "MIX_CODE_REASONING" and old.get("config") == "36_L4" else "old_frozen_tap"
            cosine_rows.append({
                "new_head_id": head_id(row),
                "comparison": label,
                "old_group": old.get("group"),
                "old_config": old.get("config"),
                "old_architecture": old.get("architecture"),
                "cosine": cosine(ndir, old["direction"]),
            })

    cluster_rows = []
    grouped = defaultdict(list)
    for row in new_heads:
        direction = row.get("direction")
        if not isinstance(direction, torch.Tensor):
            direction = direction_from_state_dict(row.get("state_dict") or {})
        if isinstance(direction, torch.Tensor):
            grouped[(row["config"], row["architecture"])].append((head_id(row), direction))
    for (config, arch), items in sorted(grouped.items()):
        vals = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                vals.append(abs(cosine(items[i][1], items[j][1])))
        cluster_rows.append({
            "config": config,
            "architecture": arch,
            "heads": len(items),
            "mean_abs_pairwise_cosine": float(mean(vals)) if vals else float("nan"),
            "min_abs_pairwise_cosine": min(vals) if vals else float("nan"),
        })

    heldout_by_head = {
        row.get("head_id"): float(row.get("pairwise_accuracy", float("nan")))
        for row in list(eval_payload.get("new_pairwise_by_head") or [])
    }
    empirical_cos = []
    heldout_scores = []
    for row in cosine_rows:
        if row["comparison"] != "empirical_hidden_origin_mean_diff":
            continue
        empirical_cos.append(abs(float(row["cosine"])))
        heldout_scores.append(heldout_by_head.get(row["new_head_id"], float("nan")))
    geom_perf_corr = correlation(empirical_cos, heldout_scores)

    max_old_align = max((abs(float(row["cosine"])) for row in cosine_rows if row["comparison"].startswith("old") and math.isfinite(float(row["cosine"]))), default=float("nan"))
    stable_scores = [float(row["mean_abs_pairwise_cosine"]) for row in cluster_rows if math.isfinite(float(row["mean_abs_pairwise_cosine"]))]
    stable_mean = float(mean(stable_scores)) if stable_scores else float("nan")
    if math.isfinite(max_old_align) and max_old_align >= 0.50:
        verdict = "ALIGNS_WITH_OLD_TAPS"
    elif math.isfinite(stable_mean) and stable_mean >= 0.50:
        verdict = "NEW_STABLE_GEOMETRY"
    elif new_heads:
        verdict = "UNSTABLE_GEOMETRY"
    else:
        verdict = "INCONCLUSIVE"

    payload = {
        "BG_HIDDEN_ORIGIN_TAP_GEOMETRY_VERDICT": verdict,
        "verdict": verdict,
        "training_verdict": heads_payload.get("verdict"),
        "eval_verdict": eval_payload.get("verdict"),
        "old_head_count": len(old_heads),
        "new_head_count": len(new_heads),
        "cosine_rows": cosine_rows,
        "cluster_rows": cluster_rows,
        "max_abs_old_tap_alignment": max_old_align,
        "mean_seed_config_stability": stable_mean,
        "empirical_geometry_vs_heldout_pairwise_correlation": geom_perf_corr,
        "adapter_proxy_comparisons": {
            "teacher_forced_adapter_proxy": "not_found",
            "sequence_adapter_proxy": "not_found",
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    display_cos = [
        {
            "new_head_id": row["new_head_id"],
            "comparison": row["comparison"],
            "old_group": row["old_group"],
            "old_config": row["old_config"],
            "old_architecture": row["old_architecture"],
            "cosine": rate(row["cosine"]),
        }
        for row in sorted(cosine_rows, key=lambda r: abs(float(r["cosine"])) if math.isfinite(float(r["cosine"])) else -1, reverse=True)[:80]
    ]
    lines = [
        "# Hidden-Origin Tap Geometry",
        "",
        f"BG_HIDDEN_ORIGIN_TAP_GEOMETRY_VERDICT = {verdict}",
        "",
        f"- max_abs_old_tap_alignment: `{rate(max_old_align)}`",
        f"- mean_seed_config_stability: `{rate(stable_mean)}`",
        f"- empirical_geometry_vs_heldout_pairwise_correlation: `{rate(geom_perf_corr)}`",
        "",
        "## Direction Stability",
        "",
    ]
    lines.extend(md_table(
        [
            {
                "config": row["config"],
                "architecture": row["architecture"],
                "heads": row["heads"],
                "mean_abs_cos": rate(row["mean_abs_pairwise_cosine"]),
                "min_abs_cos": rate(row["min_abs_pairwise_cosine"]),
            }
            for row in cluster_rows
        ],
        ["config", "architecture", "heads", "mean_abs_cos", "min_abs_cos"],
    ))
    lines.extend(["", "## Cosines", ""])
    lines.extend(md_table(display_cos, ["new_head_id", "comparison", "old_group", "old_config", "old_architecture", "cosine"]))
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_TAP_GEOMETRY_VERDICT = {verdict}", flush=True)
    print(f"Wrote {rel(OUT_JSON)}", flush=True)
    print(f"Wrote {rel(OUT_MD)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
