"""Compare v2 hidden-origin tap geometry to old, v1, and empirical directions."""
from __future__ import annotations

import math
import time
from collections import defaultdict
from statistics import mean
from typing import Any

import torch

from bg_hidden_origin_diversity_v2_common import (
    DATASET_V2_PT,
    HEADS_V2_PT,
    PROBE_ROOT,
    V1_ROOT,
    V2_ROOT,
    cosine,
    load_json,
    md_table,
    rate,
    rel,
    write_json,
    write_md,
)
from bg_hidden_origin_tap_common import direction_from_state_dict


EVAL_JSON = V2_ROOT / "heldout_eval_v2.json"
OUT_JSON = V2_ROOT / "geometry_analysis_v2.json"
OUT_MD = V2_ROOT / "geometry_analysis_v2.md"


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
                rows.append(
                    {
                        "source": rel(path),
                        "group": row.get("head_group") or row.get("head_family") or row.get("group"),
                        "config": row.get("config"),
                        "architecture": row.get("architecture"),
                        "direction": direction,
                    }
                )
    return rows


def load_v1_head_rows() -> list[dict[str, Any]]:
    path = V1_ROOT / "hidden_origin_tap_heads.pt"
    if not path.exists():
        return []
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return []
    rows = []
    for row in list(payload.get("heads") or []):
        if not row.get("flip_diagnostics", {}).get("passes"):
            continue
        direction = row.get("direction")
        if not isinstance(direction, torch.Tensor):
            direction = direction_from_state_dict(row.get("state_dict") or {})
        if isinstance(direction, torch.Tensor):
            rows.append({**row, "direction": direction, "source": rel(path)})
    return rows


def empirical_mean_diffs(dataset: dict[str, Any], variant: str = "primary_safe_deterministic") -> dict[str, torch.Tensor]:
    pairs_by_variant = dataset.get("pairs_by_variant") or {variant: dataset.get("pairs") or []}
    pairs = [pair for pair in list(pairs_by_variant.get(variant) or []) if pair.get("split") == "train"]
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
    variant = row.get("variant", "primary")
    return f"{variant}::{row['config']}::{row['architecture']}::seed={m.get('seed')}::lr={m.get('lr')}"


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
    started = time.time()
    if not HEADS_V2_PT.exists() or not DATASET_V2_PT.exists():
        payload = {"BG_HIDDEN_ORIGIN_TAP_GEOMETRY_V2_VERDICT": "INCONCLUSIVE", "blocker": "missing heads or dataset"}
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# Hidden-Origin Tap Geometry V2", "", "BG_HIDDEN_ORIGIN_TAP_GEOMETRY_V2_VERDICT = INCONCLUSIVE"])
        print("BG_HIDDEN_ORIGIN_TAP_GEOMETRY_V2_VERDICT = INCONCLUSIVE", flush=True)
        return 1
    heads_payload = torch.load(HEADS_V2_PT, map_location="cpu", weights_only=False)
    dataset = torch.load(DATASET_V2_PT, map_location="cpu", weights_only=False)
    eval_payload = load_json(EVAL_JSON, {}) or {}
    v2_heads = [row for row in list(heads_payload.get("heads") or []) if row.get("flip_diagnostics", {}).get("passes")]
    primary_heads = [row for row in v2_heads if row.get("variant") == "primary_safe_deterministic"]
    old_heads = load_old_head_rows()
    v1_heads = load_v1_head_rows()
    mean_diffs = empirical_mean_diffs(dataset)
    alpha02_diffs = empirical_mean_diffs(dataset, "safe_plus_alpha_0_02_diagnostic")

    cosine_rows = []
    for row in v2_heads:
        ndir = row.get("direction")
        if not isinstance(ndir, torch.Tensor):
            ndir = direction_from_state_dict(row.get("state_dict") or {})
        if not isinstance(ndir, torch.Tensor):
            continue
        cfg = row["config"]
        for label, diffs in (("empirical_hidden_origin_mean_diff", mean_diffs), ("alpha_0_02_diagnostic_mean_diff", alpha02_diffs)):
            empirical = diffs.get(cfg)
            cosine_rows.append(
                {
                    "new_head_id": head_id(row),
                    "comparison": label,
                    "old_group": None,
                    "old_config": cfg,
                    "old_architecture": None,
                    "cosine": cosine(ndir, empirical) if isinstance(empirical, torch.Tensor) else float("nan"),
                }
            )
        for old in old_heads:
            if old["direction"].numel() != ndir.numel():
                continue
            label = "old_objective_mixed" if old.get("group") == "MIX_CODE_REASONING" and old.get("config") == "36_L4" else "old_frozen_tap"
            cosine_rows.append(
                {
                    "new_head_id": head_id(row),
                    "comparison": label,
                    "old_group": old.get("group"),
                    "old_config": old.get("config"),
                    "old_architecture": old.get("architecture"),
                    "cosine": cosine(ndir, old["direction"]),
                }
            )
        for old in v1_heads:
            old_dir = old.get("direction")
            if not isinstance(old_dir, torch.Tensor) or old_dir.numel() != ndir.numel():
                continue
            cosine_rows.append(
                {
                    "new_head_id": head_id(row),
                    "comparison": "v1_hidden_origin_tap",
                    "old_group": old.get("variant", "v1"),
                    "old_config": old.get("config"),
                    "old_architecture": old.get("architecture"),
                    "cosine": cosine(ndir, old_dir),
                }
            )

    cluster_rows = []
    grouped = defaultdict(list)
    for row in primary_heads:
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
        cluster_rows.append(
            {
                "config": config,
                "architecture": arch,
                "heads": len(items),
                "mean_abs_pairwise_cosine": float(mean(vals)) if vals else float("nan"),
                "min_abs_pairwise_cosine": min(vals) if vals else float("nan"),
            }
        )

    heldout_by_head = {row.get("head_id"): float(row.get("pairwise_accuracy", float("nan"))) for row in list(eval_payload.get("new_pairwise_by_head") or [])}
    empirical_cos = []
    heldout_scores = []
    for row in cosine_rows:
        if row["comparison"] != "empirical_hidden_origin_mean_diff":
            continue
        empirical_cos.append(abs(float(row["cosine"])))
        heldout_scores.append(heldout_by_head.get(row["new_head_id"], float("nan")))
    geom_perf_corr = correlation(empirical_cos, heldout_scores)
    max_old_align = max((abs(float(row["cosine"])) for row in cosine_rows if row["comparison"].startswith("old") and math.isfinite(float(row["cosine"]))), default=float("nan"))
    mean_old_align_vals = [abs(float(row["cosine"])) for row in cosine_rows if row["comparison"].startswith("old") and math.isfinite(float(row["cosine"]))]
    mean_old_align = float(mean(mean_old_align_vals)) if mean_old_align_vals else float("nan")
    v1_align_vals = [abs(float(row["cosine"])) for row in cosine_rows if row["comparison"] == "v1_hidden_origin_tap" and math.isfinite(float(row["cosine"]))]
    v1_v2_alignment = max(v1_align_vals) if v1_align_vals else float("nan")
    stable_scores = [float(row["mean_abs_pairwise_cosine"]) for row in cluster_rows if math.isfinite(float(row["mean_abs_pairwise_cosine"]))]
    stable_mean = float(mean(stable_scores)) if stable_scores else float("nan")
    if math.isfinite(max_old_align) and max_old_align >= 0.50:
        verdict = "OLD_GEOMETRY_CONFIRMED"
    elif math.isfinite(stable_mean) and stable_mean >= 0.50:
        verdict = "NEW_STABLE_GEOMETRY"
    elif math.isfinite(v1_v2_alignment) and v1_v2_alignment >= 0.50:
        verdict = "MIXED_GEOMETRY"
    elif primary_heads:
        verdict = "UNSTABLE_GEOMETRY"
    else:
        verdict = "INCONCLUSIVE"

    payload = {
        "BG_HIDDEN_ORIGIN_TAP_GEOMETRY_V2_VERDICT": verdict,
        "verdict": verdict,
        "training_verdict": heads_payload.get("verdict"),
        "eval_verdict": eval_payload.get("verdict"),
        "old_head_count": len(old_heads),
        "v1_head_count": len(v1_heads),
        "v2_head_count": len(v2_heads),
        "cosine_rows": cosine_rows,
        "cluster_rows": cluster_rows,
        "max_abs_old_tap_alignment": max_old_align,
        "mean_old_tap_alignment": mean_old_align,
        "v1_v2_alignment": v1_v2_alignment,
        "mean_seed_config_stability": stable_mean,
        "geometry_vs_heldout_performance_correlation": geom_perf_corr,
        "behaviorally_diverse_only_geometry": "approximated_by_primary_non_tie_pairs",
        "alpha_0_02_geometry_available": bool(alpha02_diffs),
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
        for row in sorted(cosine_rows, key=lambda r: abs(float(r["cosine"])) if math.isfinite(float(r["cosine"])) else -1, reverse=True)[:100]
    ]
    lines = [
        "# Hidden-Origin Tap Geometry V2",
        "",
        f"BG_HIDDEN_ORIGIN_TAP_GEOMETRY_V2_VERDICT = {verdict}",
        "",
        f"- max_abs_old_tap_alignment: `{rate(max_old_align)}`",
        f"- mean_old_tap_alignment: `{rate(mean_old_align)}`",
        f"- v1_v2_alignment: `{rate(v1_v2_alignment)}`",
        f"- mean_seed_config_stability: `{rate(stable_mean)}`",
        f"- geometry_vs_heldout_performance_correlation: `{rate(geom_perf_corr)}`",
        "",
        "## Direction Stability",
        "",
    ]
    lines.extend(
        md_table(
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
        )
    )
    lines.extend(["", "## Cosines", ""])
    lines.extend(md_table(display_cos, ["new_head_id", "comparison", "old_group", "old_config", "old_architecture", "cosine"]))
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_TAP_GEOMETRY_V2_VERDICT = {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

