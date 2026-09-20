"""Replay hidden-origin taps on old cached evaluation contexts.

Diagnostic only: this reads cached candidate feature pools and does not generate
new candidates, run wrapper/local-agent code, or alter production routing.
"""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import torch

from bg_hidden_origin_quota_v4_common import (
    HEADS_V4_PT,
    PROBE_ROOT,
    SALVAGE_HEADS_PT,
    V1_ROOT,
    V2_ROOT,
    V3_ROOT,
    V4_ROOT,
    best_primary_head,
    compact_head_id,
    ensure_v4_root,
    md_table,
    rate,
    rel,
    write_csv,
    write_json,
    write_md,
)
from bg_hidden_origin_tap_common import HEAD_CLASSES, config_dim, ranking_from_matrix, score_matrix
from evaluate_bg_hidden_origin_taps import build_head


OUT_JSON = V4_ROOT / "old_context_replay_v4.json"
OUT_MD = V4_ROOT / "old_context_replay_v4.md"
OUT_CSV = V4_ROOT / "old_context_replay_v4_rows.csv"
HIDDEN_ORIGIN_COMPATIBLE_CONFIGS = {
    "24_L4",
    "24_mean",
    "36_L4",
    "36_mean",
    "47_L4",
    "47_mean",
    "concat_24_36",
    "concat_36_47",
    "concat_24_36_47",
}


def load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def pooled_map(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    out = {}
    for row in payload.get("candidate_features", []) or []:
        pooled = row.get("pooled")
        if isinstance(pooled, torch.Tensor):
            out[str(row["candidate_uid"])] = pooled.detach().cpu().to(torch.float32)
    return out


def config_vector_from_pooled(pooled: torch.Tensor, config: str) -> torch.Tensor | None:
    if tuple(pooled.shape[-3:]) != (3, 4, 2048):
        return None
    layer_idx = {24: 0, 36: 1, 47: 2}
    def l4(layer: int) -> torch.Tensor:
        return pooled[layer_idx[layer], 3, :].to(torch.float32)
    def mean_layer(layer: int) -> torch.Tensor:
        return pooled[layer_idx[layer], :, :].mean(dim=0).to(torch.float32)
    if config == "24_L4":
        return l4(24)
    if config == "24_mean":
        return mean_layer(24)
    if config == "36_L4":
        return l4(36)
    if config == "36_mean":
        return mean_layer(36)
    if config == "47_L4":
        return l4(47)
    if config == "47_mean":
        return mean_layer(47)
    if config == "concat_24_36":
        return torch.cat([l4(24), l4(36)], dim=0)
    if config == "concat_36_47":
        return torch.cat([l4(36), l4(47)], dim=0)
    if config == "concat_24_36_47":
        return torch.cat([l4(24), l4(36), l4(47)], dim=0)
    return None


def records_from_tournaments(source_name: str, data_path: Path, features_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = load_json_file(data_path)
    if not data or not features_path.exists():
        return [], {"source": source_name, "status": "missing", "data": rel(data_path), "features": rel(features_path)}
    payload = torch.load(features_path, map_location="cpu", weights_only=False)
    by_uid = pooled_map(payload)
    records = []
    for idx, row in enumerate(data.get("tournaments", []) or []):
        uids = [str(uid) for uid in row.get("candidate_uids", [])]
        labels_raw = list(row.get("labels", []))
        if not uids or len(uids) != len(labels_raw):
            continue
        missing = [uid for uid in uids if uid not in by_uid]
        if missing:
            continue
        labels = torch.tensor([str(label) == "correct" for label in labels_raw], dtype=torch.bool)
        if labels.numel() < 2 or int(labels.sum().item()) == 0:
            continue
        records.append(
            {
                "context": source_name,
                "record_id": f"{source_name}:{idx}",
                "task_id": row.get("task_id"),
                "domain": "science" if "science" in source_name else "reasoning",
                "source": row.get("source_dataset") or row.get("dataset") or source_name,
                "candidate_uids": uids,
                "labels": labels,
                "pooled": torch.stack([by_uid[uid] for uid in uids], dim=0).to(torch.float32),
            }
        )
    return records, {"source": source_name, "status": "loaded", "records": len(records), "data": rel(data_path), "features": rel(features_path)}


def load_old_context_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources = [
        (
            "reasoning_natural_distractor",
            PROBE_ROOT / "reasoning_natural_distractor_set_2026-05-17.json",
            PROBE_ROOT / "reasoning_natural_distractor_features_2026-05-17.pt",
        ),
        (
            "science_natural_distractor",
            PROBE_ROOT / "science_natural_distractor_set_2026-05-17.json",
            PROBE_ROOT / "science_natural_distractor_features_2026-05-17.pt",
        ),
        (
            "reasoning_trace_generated_answer",
            PROBE_ROOT / "reasoning_option_traces_2026-05-17.json",
            PROBE_ROOT / "reasoning_trace_features_2026-05-17.pt",
        ),
    ]
    records: list[dict[str, Any]] = []
    inventory = []
    for source_name, data_path, features_path in sources:
        rows, info = records_from_tournaments(source_name, data_path, features_path)
        records.extend(rows)
        inventory.append(info)
    return records, inventory


def score_metrics(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    order = sorted(range(int(scores.numel())), key=lambda idx: (-float(scores[idx].item()), idx))
    top1 = 1.0 if bool(labels[order[0]].item()) else 0.0
    top2 = 1.0 if any(bool(labels[idx].item()) for idx in order[: min(2, len(order))]) else 0.0
    correct = total = 0
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            if bool(labels[i].item()) == bool(labels[j].item()):
                continue
            preferred, rejected = (i, j) if bool(labels[i].item()) else (j, i)
            correct += int(float(scores[preferred]) > float(scores[rejected]))
            total += 1
    return {
        "top1_success": top1,
        "top2_oracle_coverage": top2,
        "pairwise_accuracy": correct / max(total, 1) if total else float("nan"),
        "pairwise_count": total,
        "oracle_gap": 1.0 - top1,
        "reward_mean": top1,
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["selector"], row.get("context", "all"), row.get("config", ""))].append(row)
    for (selector, context, config), vals in grouped.items():
        out[f"{selector}::{context}::{config}"] = {
            "selector": selector,
            "context": context,
            "config": config,
            "records": len(vals),
            "top1_success": mean(float(v["top1_success"]) for v in vals),
            "top2_oracle_coverage": mean(float(v["top2_oracle_coverage"]) for v in vals),
            "pairwise_accuracy": mean(float(v["pairwise_accuracy"]) for v in vals if math.isfinite(float(v["pairwise_accuracy"]))) if any(math.isfinite(float(v["pairwise_accuracy"])) for v in vals) else float("nan"),
            "oracle_gap": mean(float(v["oracle_gap"]) for v in vals),
            "reward_mean": mean(float(v["reward_mean"]) for v in vals),
        }
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


def load_hidden_heads() -> dict[str, dict[str, Any] | None]:
    return {
        "v1_hidden_origin_tap": best_primary_head(V1_ROOT / "hidden_origin_tap_heads.pt"),
        "v2_hidden_origin_tap": best_primary_head(V2_ROOT / "hidden_origin_tap_heads_v2.pt", "primary_safe_deterministic"),
        "v3_hidden_origin_tap": best_primary_head(V3_ROOT / "hidden_origin_tap_heads_v3.pt", "primary_safe_deterministic"),
        "salvage_retrained_head": best_primary_head(SALVAGE_HEADS_PT),
        "v4_hidden_origin_tap": best_primary_head(HEADS_V4_PT, "v4_only_primary_safe"),
    }


def main() -> int:
    started = time.time()
    ensure_v4_root()
    records, inventory = load_old_context_records()
    if not records:
        payload = {"BG_HIDDEN_ORIGIN_OLD_CONTEXT_REPLAY_V4_VERDICT": "INSUFFICIENT", "verdict": "INSUFFICIENT", "source_inventory": inventory}
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# Hidden-Origin Old-Context Replay V4", "", "BG_HIDDEN_ORIGIN_OLD_CONTEXT_REPLAY_V4_VERDICT = INSUFFICIENT"])
        print("BG_HIDDEN_ORIGIN_OLD_CONTEXT_REPLAY_V4_VERDICT = INSUFFICIENT", flush=True)
        return 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_rows: list[dict[str, Any]] = []
    compatibility_rows: list[dict[str, Any]] = []
    old_scores_by_record: dict[str, torch.Tensor] = {}
    try:
        from src.evaluator.bg_controller import BGController

        controller = BGController.from_artifacts(device="cpu")
    except Exception:
        controller = None
    for record in records:
        labels = record["labels"]
        random_scores = torch.zeros(int(labels.numel()), dtype=torch.float32)
        random_metric = {
            "top1_success": float(labels.to(torch.float32).mean().item()),
            "top2_oracle_coverage": min(1.0, 2.0 * float(labels.sum().item()) / max(int(labels.numel()), 1)),
            "pairwise_accuracy": 0.5,
            "pairwise_count": int(labels.sum().item()) * int((~labels).sum().item()),
            "oracle_gap": 1.0 - float(labels.to(torch.float32).mean().item()),
            "reward_mean": float(labels.to(torch.float32).mean().item()),
        }
        eval_rows.append({"selector": "random_baseline", "config": "baseline", **{k: v for k, v in record.items() if k not in {"pooled", "labels"}}, **random_metric})
        first_scores = torch.arange(int(labels.numel()), 0, -1, dtype=torch.float32)
        eval_rows.append({"selector": "first_clean_baseline", "config": "baseline", **{k: v for k, v in record.items() if k not in {"pooled", "labels"}}, **score_metrics(first_scores, labels)})
        if controller is not None:
            try:
                details = controller.rank_candidates(list(record["pooled"]), domain_hint=str(record.get("domain") or "reasoning"), mode="conservative", return_details=True)
                old_scores = details["margin_sum"].detach().cpu().to(torch.float32)
                old_scores_by_record[str(record["record_id"])] = old_scores
                eval_rows.append({"selector": "old_frozen_bg_production", "config": "controller", **{k: v for k, v in record.items() if k not in {"pooled", "labels"}}, **score_metrics(old_scores, labels)})
            except Exception as exc:
                compatibility_rows.append({"context": record["context"], "selector": "old_frozen_bg_production", "status": "incompatible", "reason": str(exc)[:240]})
    heads = load_hidden_heads()
    for selector, head_row in heads.items():
        if head_row is None:
            compatibility_rows.append({"selector": selector, "status": "missing_head"})
            continue
        config = str(head_row.get("config") or "")
        if config not in HIDDEN_ORIGIN_COMPATIBLE_CONFIGS:
            compatibility_rows.append({"selector": selector, "config": config, "status": "incompatible_config"})
            continue
        head = build_head(head_row, device)
        used = 0
        old_corr_x: list[float] = []
        old_corr_y: list[float] = []
        for record in records:
            feats = [config_vector_from_pooled(pooled, config) for pooled in record["pooled"]]
            if not feats or not all(isinstance(vec, torch.Tensor) and int(vec.numel()) == config_dim(config) for vec in feats):
                continue
            mat = score_matrix(head, [vec for vec in feats if isinstance(vec, torch.Tensor)], device)
            scores = mat.sum(dim=1).detach().cpu().to(torch.float32)
            metric = score_metrics(scores, record["labels"])
            eval_rows.append({"selector": selector, "config": config, "head_id": compact_head_id(head_row), **{k: v for k, v in record.items() if k not in {"pooled", "labels"}}, **metric})
            if str(record["record_id"]) in old_scores_by_record:
                old = old_scores_by_record[str(record["record_id"])]
                if int(old.numel()) == int(scores.numel()):
                    old_corr_x.extend(float(x) for x in old.tolist())
                    old_corr_y.extend(float(y) for y in scores.tolist())
            used += 1
        head.to("cpu")
        compatibility_rows.append({"selector": selector, "config": config, "status": "evaluated" if used else "incompatible_features", "records": used, "score_correlation_with_old_frozen_bg": correlation(old_corr_x, old_corr_y)})
    aggregate = aggregate_rows(eval_rows)
    old_vals = [row for row in aggregate.values() if row["selector"] == "old_frozen_bg_production"]
    hidden_vals = [row for row in aggregate.values() if row["selector"].endswith("hidden_origin_tap") or row["selector"] == "salvage_retrained_head"]
    compat_count = sum(1 for row in compatibility_rows if row.get("status") == "evaluated")
    if compat_count == 0:
        verdict = "INCOMPATIBLE"
    elif not old_vals:
        verdict = "INSUFFICIENT"
    else:
        old_best = max(old_vals, key=lambda row: float(row.get("pairwise_accuracy", 0.0)))
        hidden_best = max(hidden_vals, key=lambda row: float(row.get("pairwise_accuracy", 0.0))) if hidden_vals else None
        if hidden_best and abs(float(hidden_best["pairwise_accuracy"]) - float(old_best["pairwise_accuracy"])) <= 0.03:
            verdict = "MATCHES_OLD_TAPS"
        elif hidden_best and float(hidden_best["pairwise_accuracy"]) > float(old_best["pairwise_accuracy"]) + 0.03:
            verdict = "DIVERGES_BUT_USEFUL"
        elif hidden_best:
            verdict = "PARTIAL_MATCH"
        else:
            verdict = "FAILS_OLD_CONTEXTS"
    payload = {
        "BG_HIDDEN_ORIGIN_OLD_CONTEXT_REPLAY_V4_VERDICT": verdict,
        "verdict": verdict,
        "diagnostic_only": True,
        "production_routing_changed": False,
        "source_inventory": inventory,
        "record_count": len(records),
        "feature_compatibility": compatibility_rows,
        "metrics": aggregate,
        "rows": eval_rows,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    write_csv(OUT_CSV, [{k: v for k, v in row.items() if k not in {"candidate_uids"}} for row in eval_rows])
    metric_rows = [
        {
            "selector": row["selector"],
            "context": row["context"],
            "config": row["config"],
            "records": row["records"],
            "top1": rate(row["top1_success"]),
            "top2": rate(row["top2_oracle_coverage"]),
            "pairwise": rate(row["pairwise_accuracy"]),
        }
        for row in aggregate.values()
    ]
    metric_rows.sort(key=lambda row: (row["context"], row["selector"], row["config"]))
    lines = [
        "# Hidden-Origin Old-Context Replay V4",
        "",
        f"BG_HIDDEN_ORIGIN_OLD_CONTEXT_REPLAY_V4_VERDICT = {verdict}",
        "",
        "This is diagnostic only and does not alter production routing.",
        "",
        f"- record_count: `{len(records)}`",
        f"- compatible_hidden_selectors: `{compat_count}`",
        "",
        "## Metrics",
        "",
    ]
    lines.extend(md_table(metric_rows[:220], ["selector", "context", "config", "records", "top1", "top2", "pairwise"]))
    lines.extend(["", "## Feature Compatibility", ""])
    lines.extend(md_table(compatibility_rows, ["selector", "config", "status", "records", "score_correlation_with_old_frozen_bg", "reason"]))
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_OLD_CONTEXT_REPLAY_V4_VERDICT = {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
