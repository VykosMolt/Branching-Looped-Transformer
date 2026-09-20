"""Analyze v4 hidden-origin tap geometry and old-context compatibility."""
from __future__ import annotations

import math
import time
from collections import defaultdict
from statistics import mean
from typing import Any

import torch

from bg_hidden_origin_quota_v4_common import (
    DATASET_V4_PT,
    DIRECTION_BANK_V4_PT,
    HEADS_V4_PT,
    PROBE_ROOT,
    SALVAGE_HEADS_PT,
    V1_ROOT,
    V2_ROOT,
    V3_ROOT,
    V4_ROOT,
    compact_head_id,
    cosine,
    direction_from_state_dict,
    ensure_v4_root,
    load_head_rows,
    load_json,
    md_table,
    rate,
    rel,
    write_json,
    write_md,
)


OUT_JSON = V4_ROOT / "geometry_analysis_v4.json"
OUT_MD = V4_ROOT / "geometry_analysis_v4.md"


def head_direction_rows(path: Any, family: str, variant: str | None = None, only_passing: bool = True) -> list[dict[str, Any]]:
    rows = []
    for row in load_head_rows(path, variant=variant, only_passing=only_passing):
        direction = row.get("direction")
        if not isinstance(direction, torch.Tensor):
            direction = direction_from_state_dict(row.get("state_dict") or {})
        if isinstance(direction, torch.Tensor):
            rows.append({**row, "family": family, "source": rel(path), "direction": direction.detach().cpu().to(torch.float32)})
    return rows


def old_head_rows() -> list[dict[str, Any]]:
    rows = []
    for path in (PROBE_ROOT / "mixed_domain_tiny_heads_2026-05-17.pt", PROBE_ROOT / "bg_head_registry_2026-05-17.pt"):
        rows.extend(head_direction_rows(path, "old_frozen_bg_tap", None, only_passing=False))
    return rows


def direction_bank_rows() -> list[dict[str, Any]]:
    if not DIRECTION_BANK_V4_PT.exists():
        return []
    try:
        payload = torch.load(DIRECTION_BANK_V4_PT, map_location="cpu", weights_only=False)
    except Exception:
        return []
    rows = []
    for row in list(payload.get("directions") or []):
        tensor = row.get("tensor")
        if isinstance(tensor, torch.Tensor):
            rows.append({"family": row.get("family"), "source": row.get("source"), "config": row.get("target_config"), "architecture": "direction_bank", "name": row.get("name"), "direction": tensor.detach().cpu().to(torch.float32)})
    return rows


def empirical_mean_diffs(dataset: dict[str, Any], variant: str = "primary_safe_deterministic", split: str = "train") -> dict[str, torch.Tensor]:
    pairs_by_variant = dataset.get("pairs_by_variant") or {variant: dataset.get("pairs") or []}
    pairs = [pair for pair in list(pairs_by_variant.get(variant) or []) if pair.get("split") == split]
    out = {}
    by_config: dict[str, list[torch.Tensor]] = defaultdict(list)
    for pair in pairs:
        for config, feats in (pair.get("features") or {}).items():
            by_config[config].append((feats["preferred"] - feats["rejected"]).detach().cpu().to(torch.float32))
    for config, vals in by_config.items():
        if vals:
            out[config] = torch.stack(vals, dim=0).mean(dim=0)
    return out


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
    ensure_v4_root()
    if not HEADS_V4_PT.exists() or not DATASET_V4_PT.exists():
        payload = {"BG_HIDDEN_ORIGIN_TAP_GEOMETRY_V4_VERDICT": "INCONCLUSIVE", "verdict": "INCONCLUSIVE", "blocker": "missing v4 heads or dataset"}
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# Hidden-Origin Tap Geometry V4", "", "BG_HIDDEN_ORIGIN_TAP_GEOMETRY_V4_VERDICT = INCONCLUSIVE"])
        print("BG_HIDDEN_ORIGIN_TAP_GEOMETRY_V4_VERDICT = INCONCLUSIVE", flush=True)
        return 0
    heads_payload = torch.load(HEADS_V4_PT, map_location="cpu", weights_only=False)
    dataset = torch.load(DATASET_V4_PT, map_location="cpu", weights_only=False)
    eval_payload = load_json(V4_ROOT / "heldout_eval_v4.json", {}) or {}
    old_context = load_json(V4_ROOT / "old_context_replay_v4.json", {}) or {}
    v4_heads = [row for row in list(heads_payload.get("heads") or []) if row.get("flip_diagnostics", {}).get("passes")]
    primary_heads = [row for row in v4_heads if row.get("variant") == "v4_only_primary_safe"]
    comparisons = []
    comparisons.extend(old_head_rows())
    comparisons.extend(head_direction_rows(V1_ROOT / "hidden_origin_tap_heads.pt", "v1_hidden_origin_tap"))
    comparisons.extend(head_direction_rows(V2_ROOT / "hidden_origin_tap_heads_v2.pt", "v2_hidden_origin_tap", "primary_safe_deterministic"))
    comparisons.extend(head_direction_rows(V3_ROOT / "hidden_origin_tap_heads_v3.pt", "v3_hidden_origin_tap", "primary_safe_deterministic"))
    comparisons.extend(head_direction_rows(SALVAGE_HEADS_PT, "salvage_retrained_head"))
    comparisons.extend(direction_bank_rows())
    train_diffs = empirical_mean_diffs(dataset)
    cosine_rows = []
    for row in v4_heads:
        ndir = row.get("direction")
        if not isinstance(ndir, torch.Tensor):
            ndir = direction_from_state_dict(row.get("state_dict") or {})
        if not isinstance(ndir, torch.Tensor):
            continue
        cfg = row["config"]
        empirical = train_diffs.get(cfg)
        cosine_rows.append({"new_head_id": compact_head_id(row), "comparison": "v4_train_empirical_mean_diff", "old_group": None, "old_config": cfg, "old_architecture": None, "cosine": cosine(ndir, empirical) if isinstance(empirical, torch.Tensor) else float("nan")})
        for old in comparisons:
            odir = old.get("direction")
            if not isinstance(odir, torch.Tensor) or int(odir.numel()) != int(ndir.numel()):
                continue
            cosine_rows.append(
                {
                    "new_head_id": compact_head_id(row),
                    "comparison": old.get("family"),
                    "old_group": old.get("head_group") or old.get("group") or old.get("name"),
                    "old_config": old.get("config"),
                    "old_architecture": old.get("architecture"),
                    "cosine": cosine(ndir, odir),
                }
            )
    cluster_rows = []
    grouped: dict[tuple[str, str], list[tuple[str, torch.Tensor]]] = defaultdict(list)
    for row in primary_heads:
        direction = row.get("direction")
        if not isinstance(direction, torch.Tensor):
            direction = direction_from_state_dict(row.get("state_dict") or {})
        if isinstance(direction, torch.Tensor):
            grouped[(row["config"], row["architecture"])].append((compact_head_id(row), direction))
    for (config, arch), items in sorted(grouped.items()):
        vals = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                vals.append(abs(cosine(items[i][1], items[j][1])))
        cluster_rows.append({"config": config, "architecture": arch, "heads": len(items), "mean_abs_pairwise_cosine": float(mean(vals)) if vals else float("nan"), "min_abs_pairwise_cosine": min(vals) if vals else float("nan")})
    heldout_by_head = {row.get("head_id"): float(row.get("pairwise_accuracy", float("nan"))) for row in list(eval_payload.get("pairwise_by_head") or []) if row.get("source") == "v4"}
    empirical_cos = []
    heldout_scores = []
    for row in cosine_rows:
        if row["comparison"] != "v4_train_empirical_mean_diff":
            continue
        empirical_cos.append(abs(float(row["cosine"])))
        heldout_scores.append(heldout_by_head.get(row["new_head_id"], float("nan")))
    geom_perf_corr = correlation(empirical_cos, heldout_scores)
    old_vals = [abs(float(row["cosine"])) for row in cosine_rows if row["comparison"] == "old_frozen_bg_tap" and math.isfinite(float(row["cosine"]))]
    v_prev_vals = [abs(float(row["cosine"])) for row in cosine_rows if row["comparison"] in {"v1_hidden_origin_tap", "v2_hidden_origin_tap", "v3_hidden_origin_tap", "salvage_retrained_head"} and math.isfinite(float(row["cosine"]))]
    stable_vals = [float(row["mean_abs_pairwise_cosine"]) for row in cluster_rows if math.isfinite(float(row["mean_abs_pairwise_cosine"]))]
    max_old = max(old_vals, default=float("nan"))
    max_prev = max(v_prev_vals, default=float("nan"))
    stable_mean = float(mean(stable_vals)) if stable_vals else float("nan")
    old_context_verdict = str(old_context.get("verdict") or "INSUFFICIENT")
    if old_context_verdict in {"MATCHES_OLD_TAPS", "DIVERGES_BUT_USEFUL"}:
        verdict = "OLD_CONTEXT_COMPATIBLE"
    elif math.isfinite(max_old) and max_old >= 0.50:
        verdict = "OLD_GEOMETRY_CONFIRMED"
    elif math.isfinite(stable_mean) and stable_mean >= 0.50:
        verdict = "NEW_STABLE_GEOMETRY"
    elif math.isfinite(max_prev) and max_prev >= 0.50:
        verdict = "MIXED_GEOMETRY"
    elif primary_heads:
        verdict = "HIDDEN_ORIGIN_SPECIALIZED"
    else:
        verdict = "INCONCLUSIVE"
    payload = {
        "BG_HIDDEN_ORIGIN_TAP_GEOMETRY_V4_VERDICT": verdict,
        "verdict": verdict,
        "training_verdict": heads_payload.get("verdict"),
        "eval_verdict": eval_payload.get("verdict"),
        "old_context_replay_verdict": old_context_verdict,
        "v4_head_count": len(v4_heads),
        "cosine_rows": cosine_rows,
        "cluster_rows": cluster_rows,
        "max_abs_old_tap_alignment": max_old,
        "v1_v2_v3_v4_alignment": max_prev,
        "v4_vs_salvage_alignment": max([abs(float(row["cosine"])) for row in cosine_rows if row["comparison"] == "salvage_retrained_head" and math.isfinite(float(row["cosine"]))], default=float("nan")),
        "mean_seed_config_stability": stable_mean,
        "geometry_vs_heldout_performance_correlation": geom_perf_corr,
        "old_context_replay_geometry_compatibility": old_context_verdict,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    rows_md = [
        {
            "new_head_id": row["new_head_id"],
            "comparison": row["comparison"],
            "old_group": row.get("old_group"),
            "old_config": row.get("old_config"),
            "cosine": rate(row["cosine"]),
        }
        for row in sorted(cosine_rows, key=lambda r: abs(float(r["cosine"])) if math.isfinite(float(r["cosine"])) else -1, reverse=True)[:160]
    ]
    lines = [
        "# Hidden-Origin Tap Geometry V4",
        "",
        f"BG_HIDDEN_ORIGIN_TAP_GEOMETRY_V4_VERDICT = {verdict}",
        "",
        f"- max_abs_old_tap_alignment: `{rate(max_old)}`",
        f"- v1_v2_v3_v4_alignment: `{rate(max_prev)}`",
        f"- v4_vs_salvage_alignment: `{rate(payload['v4_vs_salvage_alignment'])}`",
        f"- mean_seed_config_stability: `{rate(stable_mean)}`",
        f"- geometry_vs_heldout_performance_correlation: `{rate(geom_perf_corr)}`",
        f"- old_context_replay_geometry_compatibility: `{old_context_verdict}`",
        "",
        "## Strongest Alignments",
        "",
    ]
    lines.extend(md_table(rows_md, ["new_head_id", "comparison", "old_group", "old_config", "cosine"]))
    lines.extend(["", "## Seed/Config Stability", ""])
    lines.extend(md_table([{**row, "mean_abs_pairwise_cosine": rate(row["mean_abs_pairwise_cosine"]), "min_abs_pairwise_cosine": rate(row["min_abs_pairwise_cosine"])} for row in cluster_rows], ["config", "architecture", "heads", "mean_abs_pairwise_cosine", "min_abs_pairwise_cosine"]))
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_TAP_GEOMETRY_V4_VERDICT = {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
