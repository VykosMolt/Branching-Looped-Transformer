"""Analyze layer/config performance for hidden-origin branch taps."""
from __future__ import annotations

import time
from typing import Any

from bg_hidden_origin_tap_common import CONFIGS, OUT_ROOT, load_json, md_table, rate, rel, write_json, write_md


EVAL_JSON = OUT_ROOT / "heldout_eval.json"
OUT_JSON = OUT_ROOT / "layer_config_analysis.json"
OUT_MD = OUT_ROOT / "layer_config_analysis.md"


def config_family(config: str) -> str:
    if config.startswith("24_"):
        return "early_L24"
    if config.startswith("30_") or config.startswith("36_") or config.startswith("42_"):
        return "mid"
    if config.startswith("47_"):
        return "late_L47"
    if config.startswith("concat"):
        return "concat"
    return "unknown"


def verdict_for(best: dict[str, Any] | None, eval_verdict: str, rows: list[dict[str, Any]]) -> str:
    if eval_verdict == "INSUFFICIENT" or not rows:
        return "INSUFFICIENT"
    if best is None:
        return "NO_CLEAR_LAYER"
    cfg = str(best.get("config"))
    top1 = float(best.get("top1_success", 0.0))
    if top1 <= 0.0:
        return "NO_CLEAR_LAYER"
    fam = config_family(cfg)
    if fam == "early_L24":
        return "EARLY_READOUT_SUFFICIENT"
    if fam == "mid":
        return "MID_LAYER_BEST"
    if fam == "late_L47":
        return "LATE_READOUT_REQUIRED"
    if fam == "concat":
        return "CONCAT_REQUIRED"
    return "NO_CLEAR_LAYER"


def main() -> int:
    started = time.time()
    eval_payload = load_json(EVAL_JSON, {}) or {}
    behavior = eval_payload.get("behaviorally_diverse_metrics", {}) or {}
    all_metrics = eval_payload.get("metrics", {}) or {}
    rows = []
    for source_name, metrics in (("behaviorally_diverse", behavior), ("all", all_metrics)):
        for row in metrics.values():
            policy = str(row.get("policy") or "")
            if not policy.startswith("new_hidden_origin_tap_config_") or not policy.endswith("_pairwise_tournament"):
                continue
            cfg = str(row.get("config"))
            rows.append(
                {
                    "subset": source_name,
                    "config": cfg,
                    "family": config_family(cfg),
                    "architecture": row.get("architecture"),
                    "groups": row.get("groups"),
                    "top1_success": row.get("top1_success"),
                    "reward_mean": row.get("reward_mean"),
                    "pairwise_proxy": None,
                    "selection_regret": row.get("selection_regret"),
                    "top2_oracle_coverage": row.get("top2_oracle_coverage"),
                }
            )
    behavior_rows = [row for row in rows if row["subset"] == "behaviorally_diverse"]
    best = max(behavior_rows, key=lambda row: (float(row.get("top1_success", 0.0)), float(row.get("reward_mean", 0.0)), -float(row.get("selection_regret", 999.0)))) if behavior_rows else None
    verdict = verdict_for(best, str(eval_payload.get("verdict")), behavior_rows)
    recommendation = "insufficient"
    if verdict == "EARLY_READOUT_SUFFICIENT":
        recommendation = "L24"
    elif verdict == "MID_LAYER_BEST":
        recommendation = "L36"
    elif verdict == "LATE_READOUT_REQUIRED":
        recommendation = "L47"
    elif verdict == "CONCAT_REQUIRED":
        recommendation = "concat"

    by_family = {}
    for fam in sorted({row["family"] for row in behavior_rows}):
        vals = [row for row in behavior_rows if row["family"] == fam]
        if vals:
            by_family[fam] = max(vals, key=lambda row: (float(row.get("top1_success", 0.0)), float(row.get("reward_mean", 0.0))))

    payload = {
        "BG_HIDDEN_ORIGIN_LAYER_CONFIG_VERDICT": verdict,
        "verdict": verdict,
        "eval_verdict": eval_payload.get("verdict"),
        "best_behaviorally_diverse_config": best,
        "best_by_family": by_family,
        "first_phase2_scoring_point_recommendation": recommendation,
        "rows": rows,
        "questions": {
            "branch_quality_readable_early_L24": verdict == "EARLY_READOUT_SUFFICIENT",
            "readability_emerges_mid_or_late": verdict in {"MID_LAYER_BEST", "LATE_READOUT_REQUIRED"},
            "L30_L42_useful": any(row["config"] in {"30_L4", "42_L4", "concat_24_30_36", "concat_36_42_47"} for row in behavior_rows[:3]),
            "concat_better_than_single_layer": verdict == "CONCAT_REQUIRED",
            "mean_over_loops_helped": bool(best and str(best.get("config")).endswith("_mean")),
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    display_rows = [
        {
            "subset": row["subset"],
            "config": row["config"],
            "family": row["family"],
            "architecture": row["architecture"],
            "groups": row["groups"],
            "top1": rate(row["top1_success"]),
            "reward": rate(row["reward_mean"]),
            "regret": rate(row["selection_regret"]),
        }
        for row in sorted(rows, key=lambda r: (r["subset"], -float(r.get("top1_success", 0.0)), str(r["config"])))
    ]
    lines = [
        "# Hidden-Origin Tap Layer/Config Analysis",
        "",
        f"BG_HIDDEN_ORIGIN_LAYER_CONFIG_VERDICT = {verdict}",
        "",
        f"- eval_verdict: `{eval_payload.get('verdict')}`",
        f"- best_behaviorally_diverse_config: `{best}`",
        f"- first_phase2_scoring_point_recommendation: `{recommendation}`",
        "",
        "## Config Rows",
        "",
    ]
    lines.extend(md_table(display_rows, ["subset", "config", "family", "architecture", "groups", "top1", "reward", "regret"]))
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_LAYER_CONFIG_VERDICT = {verdict}", flush=True)
    print(f"Wrote {rel(OUT_JSON)}", flush=True)
    print(f"Wrote {rel(OUT_MD)}", flush=True)
    return 0 if verdict != "INSUFFICIENT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
