"""Evaluate old/v1/v2/v3/salvage selectors under leakage-aware modes."""
from __future__ import annotations

import math
import time
from collections import defaultdict

import torch

from bg_hidden_origin_split_salvage_common import (
    EVAL_MODES_JSON,
    SALVAGE_DATASETS_PT,
    SALVAGE_EVAL_JSON,
    SALVAGE_HEADS_PT,
    SALVAGE_ROOT,
    aggregate_selector_rows,
    baseline_selector_rows,
    best_metric,
    compact_head_for_json,
    ensemble_selector_rows,
    ensure_salvage_root,
    groups_for_tasks,
    load_json,
    load_pt,
    md_table,
    primary_rows,
    rate,
    rel,
    tap_selector_rows,
    write_csv,
    write_json,
    write_md,
)
from bg_hidden_origin_diversity_v3_common import HEADS_V3_PT, V1_ROOT, V2_ROOT, load_head_rows, primary_safe_v3_row
from bg_hidden_origin_tap_common import pairwise_accuracy_from_pairs
from evaluate_bg_hidden_origin_taps import build_head


OUT_MD = SALVAGE_ROOT / "salvage_selector_eval.md"
OUT_CSV = SALVAGE_ROOT / "salvage_selector_eval_rows.csv"


def best_existing_head(path, variant: str | None = None):
    rows = load_head_rows(path, variant=variant, only_passing=True)
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("metrics", {}).get("validation_pairwise_accuracy", -1.0)))


def best_salvage_heads_by_mode(heads: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for row in heads:
        if not row.get("flip_diagnostics", {}).get("passes"):
            continue
        key = str(row.get("mode_key"))
        old = out.get(key)
        if old is None or float(row["metrics"].get("validation_pairwise_accuracy", -1.0)) > float(old["metrics"].get("validation_pairwise_accuracy", -1.0)):
            out[key] = row
    return out


def old_pairwise_accuracy(pairs: list[dict[str, object]]) -> float:
    total = 0
    ok = 0
    for pair in pairs:
        pref = float(pair.get("old_frozen_tap_score_preferred", 0.0))
        rej = float(pair.get("old_frozen_tap_score_rejected", 0.0))
        ok += int(pref > rej)
        total += 1
    return ok / max(total, 1) if total else float("nan")


def pairwise_rows_for_mode(mode_key: str, mode_payload: dict[str, object], heads: dict[str, dict[str, object] | None], device: torch.device) -> list[dict[str, object]]:
    pairs = [pair for pair in list(mode_payload.get("pairs") or []) if pair.get("split") == "test"]
    out = [
        {
            "mode_key": mode_key,
            "source": "old_frozen_bg",
            "heldout_pairs": len(pairs),
            "pairwise_accuracy": old_pairwise_accuracy(pairs),
        }
    ]
    for source, head_row in heads.items():
        if not head_row:
            continue
        head = build_head(head_row, device)
        cfg = str(head_row["config"])
        cfg_pairs = [pair for pair in pairs if cfg in pair.get("features", {})]
        out.append(
            {
                "mode_key": mode_key,
                "source": source,
                "head_id": f"{head_row['config']}::{head_row['architecture']}",
                "heldout_pairs": len(cfg_pairs),
                "pairwise_accuracy": pairwise_accuracy_from_pairs(head, cfg_pairs, cfg, device),
                "validation_pairwise_accuracy": head_row.get("metrics", {}).get("validation_pairwise_accuracy"),
            }
        )
        head.to("cpu")
    return out


def selector_verdict(records: list[dict[str, object]], metrics: dict[str, object]) -> tuple[str, str]:
    readiness_records = [r for r in records if (r.get("heldout_support") or {}).get("readiness_support")]
    weak_records = [r for r in records if (r.get("heldout_support") or {}).get("weak_support")]

    def is_top1_selector(row: dict[str, object]) -> bool:
        policy = str(row.get("policy", ""))
        if policy in {"random_top1", "random_top2", "clean_branch_baseline", "simple_top2_branch_id_order"}:
            return False
        if policy.endswith("_top2") or policy == "diagnostic_rank_aggregation_ensemble_top2":
            return False
        return bool(row.get("selector_clean_for_mode"))

    selector_candidates = [
        row for row in metrics.values()
        if row.get("subset") == "behaviorally_diverse" and is_top1_selector(row)
    ]
    if not selector_candidates:
        return "INSUFFICIENT", "insufficient"
    best = max(selector_candidates, key=lambda row: (float(row["task_macro_top1_success"]), float(row["top1_success"]), float(row["reward_mean"])))
    best_selector = selector_name(str(best["policy"]))

    ready_positive = False
    ready_strong = False
    old_best = False
    ensemble_best = False
    checked_readiness = False
    for record in readiness_records:
        mode = str(record["mode_name"])
        if mode not in {"strict_cross_version_clean", "v3_clean", "old_frozen_tap_clean"}:
            continue
        mode_vals = [m for m in metrics.values() if m.get("mode_name") == mode and m.get("subset") == "behaviorally_diverse"]
        rand_vals = [m for m in mode_vals if m.get("policy") == "random_top1"]
        old_vals = [m for m in mode_vals if m.get("policy") == "old_frozen_bg_pairwise_tournament"]
        candidates = [m for m in mode_vals if is_top1_selector(m)]
        if not rand_vals or not candidates:
            continue
        checked_readiness = True
        rand = rand_vals[0]
        old = old_vals[0] if old_vals else None
        best_mode = max(candidates, key=lambda row: (float(row["task_macro_top1_success"]), float(row["top1_success"]), float(row["reward_mean"])))
        macro_lift = float(best_mode["task_macro_top1_success"]) - float(rand["task_macro_top1_success"])
        micro_lift = float(best_mode["top1_success"]) - float(rand["top1_success"])
        if macro_lift <= 0.0:
            continue
        ready_positive = True
        if str(best_mode["policy"]).startswith("old_frozen_bg"):
            old_best = True
            if micro_lift > 0.0 and macro_lift >= 0.05:
                ready_strong = True
        elif old and float(best_mode["task_macro_top1_success"]) > float(old["task_macro_top1_success"]) and micro_lift > 0.0:
            ready_strong = True
        if "ensemble" in str(best_mode["policy"]):
            ensemble_best = True
        best_selector = selector_name(str(best_mode["policy"]))

    if ready_strong and old_best:
        return "OLD_TAPS_BEST", "old_frozen_bg"
    if ready_strong and ensemble_best:
        return "ENSEMBLE_BEST", "ensemble"
    if ready_strong:
        return "SELECTOR_READY", best_selector
    if ready_positive:
        return "WEAK_SELECTOR", best_selector
    if not checked_readiness:
        return ("WEAK_SELECTOR" if weak_records else "STILL_DATA_LIMITED"), best_selector
    return "NO_SELECTOR_SIGNAL", best_selector


def selector_name(policy: str) -> str:
    if policy.startswith("old_frozen_bg"):
        return "old_frozen_bg"
    if policy.startswith("previous_hidden_origin_tap_v1"):
        return "v1_hidden_origin_tap"
    if policy.startswith("previous_hidden_origin_tap_v2"):
        return "v2_hidden_origin_tap"
    if policy.startswith("existing_hidden_origin_tap_v3"):
        return "v3_hidden_origin_tap"
    if policy.startswith("salvage_retrained_head"):
        return "salvage_retrained_head"
    if "ensemble" in policy:
        return "ensemble"
    if policy.startswith("random"):
        return "random"
    return "insufficient"


def main() -> int:
    started = time.time()
    ensure_salvage_root()
    modes_payload = load_json(EVAL_MODES_JSON, {}) or {}
    dataset = load_pt(SALVAGE_DATASETS_PT, {}) or {}
    trained = load_pt(SALVAGE_HEADS_PT, {}) or {"heads": []}
    if not dataset or not modes_payload:
        payload = {"BG_HIDDEN_ORIGIN_SALVAGE_EVAL_VERDICT": "INSUFFICIENT", "verdict": "INSUFFICIENT", "blocker": "missing modes or datasets"}
        write_json(SALVAGE_EVAL_JSON, payload)
        write_md(OUT_MD, ["# Hidden-Origin Salvage Selector Eval", "", "BG_HIDDEN_ORIGIN_SALVAGE_EVAL_VERDICT = INSUFFICIENT"])
        print("BG_HIDDEN_ORIGIN_SALVAGE_EVAL_VERDICT = INSUFFICIENT", flush=True)
        return 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = primary_rows()
    records = list(modes_payload.get("records") or [])
    record_by_key = {f"{record['mode_name']}::{record['fold_id']}": record for record in records}
    modes = dict(dataset.get("modes") or {})

    v1_head = best_existing_head(V1_ROOT / "hidden_origin_tap_heads.pt")
    v2_head = best_existing_head(V2_ROOT / "hidden_origin_tap_heads_v2.pt", "primary_safe_deterministic")
    v3_head = best_existing_head(HEADS_V3_PT, "primary_safe_deterministic")
    salvage_by_mode = best_salvage_heads_by_mode(list(trained.get("heads") or []))

    eval_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []
    for mode_key, mode_payload in sorted(modes.items()):
        record = record_by_key.get(mode_key) or mode_payload.get("record")
        if not record:
            continue
        heldout_tasks = record.get("heldout_task_ids") or []
        all_groups = groups_for_tasks(rows, heldout_tasks, primary_safe_v3_row)
        subsets = {
            "all_groups": all_groups,
            "behaviorally_diverse": [g for g in all_groups if any(True for _ in [g]) and __import__("bg_hidden_origin_diversity_v3_common").group_is_behaviorally_diverse_v2(g)],
            "reward_diverse": [g for g in all_groups if __import__("bg_hidden_origin_diversity_v3_common").group_is_reward_diverse_v2(g)],
        }
        salvage_head = salvage_by_mode.get(mode_key)
        flags = record.get("contamination_flags", {})
        heads_for_pairwise = {
            "v1_hidden_origin_tap": v1_head,
            "v2_hidden_origin_tap": v2_head,
            "v3_hidden_origin_tap": v3_head,
            "salvage_retrained_head": salvage_head,
        }
        pairwise_rows.extend(pairwise_rows_for_mode(mode_key, mode_payload, heads_for_pairwise, device))
        for subset_name, groups in subsets.items():
            if not groups:
                continue
            eval_rows.extend(baseline_selector_rows(groups, record, subset_name))
            eval_rows.extend(tap_selector_rows(groups, record, subset_name, v1_head, device, "previous_hidden_origin_tap_v1", bool(flags.get("v1_clean"))))
            eval_rows.extend(tap_selector_rows(groups, record, subset_name, v2_head, device, "previous_hidden_origin_tap_v2", bool(flags.get("v2_clean"))))
            eval_rows.extend(tap_selector_rows(groups, record, subset_name, v3_head, device, "existing_hidden_origin_tap_v3", bool(flags.get("v3_clean"))))
            eval_rows.extend(tap_selector_rows(groups, record, subset_name, salvage_head, device, "salvage_retrained_head", True))
            eval_rows.extend(
                ensemble_selector_rows(
                    groups,
                    record,
                    subset_name,
                    [
                        ("v1", v1_head, bool(flags.get("v1_clean"))),
                        ("v2", v2_head, bool(flags.get("v2_clean"))),
                        ("v3", v3_head, bool(flags.get("v3_clean"))),
                        ("salvage", salvage_head, True),
                    ],
                    device,
                )
            )

    metrics = aggregate_selector_rows(eval_rows)
    verdict, best_available = selector_verdict(records, metrics)
    payload = {
        "BG_HIDDEN_ORIGIN_SALVAGE_EVAL_VERDICT": verdict,
        "verdict": verdict,
        "HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE": best_available,
        "device": str(device),
        "v1_head": compact_head_for_json(v1_head) if v1_head else None,
        "v2_head": compact_head_for_json(v2_head) if v2_head else None,
        "v3_head": compact_head_for_json(v3_head) if v3_head else None,
        "salvage_heads_by_mode": {key: compact_head_for_json(row) for key, row in salvage_by_mode.items()},
        "metrics": metrics,
        "pairwise_rows": pairwise_rows,
        "rows": eval_rows,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(SALVAGE_EVAL_JSON, payload)
    write_csv(OUT_CSV, eval_rows)

    behavior_rows = [
        row for row in metrics.values()
        if row.get("subset") == "behaviorally_diverse"
        and row.get("mode_name") in {"strict_cross_version_clean", "v3_clean", "old_frozen_tap_clean", "grouped_kfold_v3"}
    ]
    best_rows = sorted(
        behavior_rows,
        key=lambda row: (float(row.get("task_macro_top1_success", float("nan"))), float(row.get("top1_success", float("nan")))),
        reverse=True,
    )[:80]
    lines = [
        "# Hidden-Origin Salvage Selector Evaluation",
        "",
        f"BG_HIDDEN_ORIGIN_SALVAGE_EVAL_VERDICT = {verdict}",
        f"HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE = {best_available}",
        "",
        "Metrics report both group micro-averages and task macro-averages. Selector-readiness decisions use primary-safe deterministic behaviorally diverse groups and contamination-aware mode flags.",
        "",
        "## Best Behaviorally Diverse Metrics",
        "",
    ]
    lines.extend(
        md_table(
            [
                {
                    "mode": row["mode_name"],
                    "fold": row["fold_id"],
                    "policy": row["policy"],
                    "groups": row["groups"],
                    "tasks": row["task_count"],
                    "task_macro_top1": rate(row["task_macro_top1_success"]),
                    "top1": rate(row["top1_success"]),
                    "reward": rate(row["reward_mean"]),
                    "clean": row["selector_clean_for_mode"],
                }
                for row in best_rows
            ],
            ["mode", "fold", "policy", "groups", "tasks", "task_macro_top1", "top1", "reward", "clean"],
        )
    )
    lines.extend(["", "## Pairwise Accuracy Rows", ""])
    lines.extend(
        md_table(
            [
                {
                    "mode": row["mode_key"],
                    "source": row["source"],
                    "pairs": row["heldout_pairs"],
                    "pairwise": rate(row["pairwise_accuracy"]),
                    "val": rate(row.get("validation_pairwise_accuracy")),
                }
                for row in pairwise_rows[:120]
            ],
            ["mode", "source", "pairs", "pairwise", "val"],
        )
    )
    lines.extend(["", f"Wrote `{rel(SALVAGE_EVAL_JSON)}` and `{rel(OUT_CSV)}`."])
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_SALVAGE_EVAL_VERDICT = {verdict}", flush=True)
    print(f"HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE = {best_available}", flush=True)
    print(f"Wrote {rel(SALVAGE_EVAL_JSON)}", flush=True)
    return 0 if verdict != "INSUFFICIENT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
