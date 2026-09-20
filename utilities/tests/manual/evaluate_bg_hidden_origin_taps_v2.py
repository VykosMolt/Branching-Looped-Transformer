"""Heldout evaluation for hidden-origin branch taps v2."""
from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any, Sequence

import torch

from bg_hidden_origin_diversity_v2_common import (
    DATASET_V2_PT,
    HEADS_V2_PT,
    V1_ROOT,
    V2_ROOT,
    diagnostic_alpha_row,
    ensure_v2_root,
    group_is_behaviorally_diverse_v2,
    group_is_reward_diverse_v2,
    group_rows,
    load_all_v2_branch_rows,
    md_table,
    rate,
    rel,
    safe_primary_row,
    stable_v2_row,
    write_csv,
    write_json,
    write_md,
)
from evaluate_bg_hidden_origin_taps import (
    aggregate,
    baseline_policy_rows,
    best_metric,
    breakdown,
    build_head,
    compact_head_id,
    new_tap_policy_rows,
    old_pairwise_accuracy,
)
from bg_hidden_origin_tap_common import pairwise_accuracy_from_pairs


OUT_JSON = V2_ROOT / "heldout_eval_v2.json"
OUT_MD = V2_ROOT / "heldout_eval_v2.md"
OUT_CSV = V2_ROOT / "heldout_eval_v2_rows.csv"


def best_heads_by_config_v2(heads: Sequence[dict[str, Any]], variant: str = "primary_safe_deterministic") -> list[dict[str, Any]]:
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in heads:
        if row.get("variant") != variant:
            continue
        if row.get("flip_diagnostics", {}).get("passes"):
            by_config[row["config"]].append(row)
    out = []
    for items in by_config.values():
        out.append(max(items, key=lambda row: float(row["metrics"].get("validation_pairwise_accuracy", -1.0))))
    return out


def load_v1_heads() -> list[dict[str, Any]]:
    path = V1_ROOT / "hidden_origin_tap_heads.pt"
    if not path.exists():
        return []
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return []
    return [row for row in list(payload.get("heads") or []) if row.get("flip_diagnostics", {}).get("passes")]


def alpha_label(row: dict[str, Any]) -> str:
    return str(row.get("alpha_bucket") or row.get("alpha"))


def aggregate_subsets(eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    behavior_rows = [row for row in eval_rows if bool(row.get("behaviorally_diverse"))]
    reward_rows = [row for row in eval_rows if bool(row.get("reward_diverse"))]
    return {
        "metrics": aggregate(eval_rows),
        "behaviorally_diverse_metrics": aggregate(behavior_rows),
        "reward_diverse_metrics": aggregate(reward_rows),
    }


def main() -> int:
    ensure_v2_root()
    started = time.time()
    if not DATASET_V2_PT.exists() or not HEADS_V2_PT.exists():
        payload = {"BG_HIDDEN_ORIGIN_TAP_EVAL_V2_VERDICT": "INSUFFICIENT", "blocker": "missing dataset v2 or heads v2"}
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# Hidden-Origin Tap Heldout Evaluation V2", "", "BG_HIDDEN_ORIGIN_TAP_EVAL_V2_VERDICT = INSUFFICIENT"])
        print("BG_HIDDEN_ORIGIN_TAP_EVAL_V2_VERDICT = INSUFFICIENT", flush=True)
        return 1
    dataset = torch.load(DATASET_V2_PT, map_location="cpu", weights_only=False)
    trained = torch.load(HEADS_V2_PT, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    primary_pairs = list(dataset.get("pairs") or [])
    test_pairs = [pair for pair in primary_pairs if pair.get("split") == "test"]
    test_tasks = set((dataset.get("tasks_by_split") or {}).get("test", []))
    all_rows = load_all_v2_branch_rows(include_prior=True)
    heldout_rows = [row for row in all_rows if str(row.get("task_id")) in test_tasks and safe_primary_row(row) and stable_v2_row(row)]
    heldout_groups = [vals for vals in group_rows(heldout_rows).values() if len(vals) >= 2]
    diagnostic_rows = [row for row in all_rows if str(row.get("task_id")) in test_tasks and diagnostic_alpha_row(row) and stable_v2_row(row)]
    diagnostic_groups = [vals for vals in group_rows(diagnostic_rows).values() if len(vals) >= 2]

    heads = list(trained.get("heads") or [])
    primary_heads = [row for row in heads if row.get("variant") == "primary_safe_deterministic" and row.get("flip_diagnostics", {}).get("passes")]
    best_head = max(primary_heads, key=lambda row: float(row["metrics"].get("validation_pairwise_accuracy", -1.0))) if primary_heads else None
    eval_head_rows = best_heads_by_config_v2(heads)
    if best_head is not None and all(compact_head_id(row) != compact_head_id(best_head) for row in eval_head_rows):
        eval_head_rows.append(best_head)

    eval_rows = baseline_policy_rows(heldout_groups)
    if best_head is not None:
        eval_rows.extend(new_tap_policy_rows(heldout_groups, best_head, device, "new_hidden_origin_tap_v2"))
    for head_row in eval_head_rows:
        eval_rows.extend(new_tap_policy_rows(heldout_groups, head_row, device, f"new_hidden_origin_tap_v2_config_{head_row['config']}"))

    v1_heads = load_v1_heads()
    if v1_heads:
        best_v1 = max(v1_heads, key=lambda row: float(row.get("metrics", {}).get("validation_pairwise_accuracy", -1.0)))
        eval_rows.extend(new_tap_policy_rows(heldout_groups, best_v1, device, "previous_hidden_origin_tap_v1"))
    else:
        best_v1 = None

    diagnostic_eval_rows = baseline_policy_rows(diagnostic_groups)
    if best_head is not None:
        diagnostic_eval_rows.extend(new_tap_policy_rows(diagnostic_groups, best_head, device, "new_hidden_origin_tap_v2"))

    subsets = aggregate_subsets(eval_rows)
    diagnostic_subsets = aggregate_subsets(diagnostic_eval_rows)
    behavior_metrics = subsets["behaviorally_diverse_metrics"]
    behavior_rows = [row for row in eval_rows if bool(row.get("behaviorally_diverse"))]
    diverse_groups = len({row["branch_group_id"] for row in behavior_rows if row["policy"] == "random_top1"})

    pairwise_by_head = []
    for head_row in eval_head_rows:
        head = build_head(head_row, device)
        cfg_pairs = [pair for pair in test_pairs if head_row["config"] in pair.get("features", {})]
        pairwise_by_head.append(
            {
                "head_id": compact_head_id(head_row),
                "variant": head_row.get("variant"),
                "config": head_row["config"],
                "architecture": head_row["architecture"],
                "heldout_pairs": len(cfg_pairs),
                "pairwise_accuracy": pairwise_accuracy_from_pairs(head, cfg_pairs, head_row["config"], device),
                "validation_pairwise_accuracy": head_row["metrics"].get("validation_pairwise_accuracy"),
            }
        )
        head.to("cpu")

    old_pair_acc = old_pairwise_accuracy(test_pairs)
    best_new_behavior = best_metric(behavior_metrics, "new_hidden_origin_tap_v2_pairwise_tournament")
    random_behavior = best_metric(behavior_metrics, "random_top1")
    old_behavior = best_metric(behavior_metrics, "old_frozen_bg_pairwise_tournament")
    if not heldout_groups or not test_pairs:
        verdict = "INSUFFICIENT"
    elif len(test_tasks) < 4 or len(test_pairs) < 30 or diverse_groups < 10:
        verdict = "DATA_LIMITED"
    elif best_new_behavior and random_behavior and old_behavior:
        new_top1 = float(best_new_behavior["top1_success"])
        random_top1 = float(random_behavior["top1_success"])
        old_top1 = float(old_behavior["top1_success"])
        if new_top1 > random_top1 and new_top1 > old_top1:
            verdict = "SELECTOR_READY"
        elif new_top1 >= random_top1 or new_top1 >= old_top1:
            verdict = "WEAK_SELECTOR"
        elif trained.get("verdict") in {"READY", "WEAK"}:
            verdict = "OVERFIT"
        else:
            verdict = "NO_SELECTOR_SIGNAL"
    else:
        verdict = "NO_SELECTOR_SIGNAL"

    payload = {
        "BG_HIDDEN_ORIGIN_TAP_EVAL_V2_VERDICT": verdict,
        "verdict": verdict,
        "dataset": rel(DATASET_V2_PT),
        "heads": rel(HEADS_V2_PT),
        "training_verdict": trained.get("verdict"),
        "dataset_verdict": dataset.get("verdict"),
        "heldout_task_ids": sorted(test_tasks),
        "heldout_group_count": len(heldout_groups),
        "heldout_pair_count": len(test_pairs),
        "behaviorally_diverse_heldout_groups": diverse_groups,
        "diagnostic_alpha_heldout_group_count": len(diagnostic_groups),
        "best_head": compact_head_id(best_head) if best_head else None,
        "previous_v1_head": compact_head_id(best_v1) if best_v1 else None,
        "old_frozen_pairwise_accuracy": old_pair_acc,
        "new_pairwise_by_head": pairwise_by_head,
        **subsets,
        "diagnostic_alpha_0_02": diagnostic_subsets,
        "per_domain_breakdown": breakdown(eval_rows, "domain"),
        "per_branch_point_breakdown": breakdown(eval_rows, "branch_point"),
        "per_alpha_breakdown": breakdown(eval_rows, "alpha"),
        "per_delta_family_breakdown": breakdown(eval_rows, "delta_family") if eval_rows and "delta_family" in eval_rows[0] else {},
        "per_config_breakdown": breakdown(eval_rows, "config"),
        "rows": eval_rows,
        "diagnostic_rows": diagnostic_eval_rows,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    write_csv(OUT_CSV, eval_rows + diagnostic_eval_rows)

    metric_rows = sorted(
        [
            {
                "subset": "behavior",
                "policy": row["policy"],
                "config": row["config"],
                "architecture": row["architecture"],
                "groups": row["groups"],
                "top1": rate(row["top1_success"]),
                "reward": rate(row["reward_mean"]),
                "regret": rate(row["selection_regret"]),
                "top2_oracle": rate(row["top2_oracle_coverage"]),
            }
            for row in behavior_metrics.values()
        ],
        key=lambda row: (row["policy"], row["config"], row["architecture"]),
    )
    lines = [
        "# Hidden-Origin Tap Heldout Evaluation V2",
        "",
        f"BG_HIDDEN_ORIGIN_TAP_EVAL_V2_VERDICT = {verdict}",
        "",
        f"- heldout_task_ids: `{len(test_tasks)}`",
        f"- heldout_group_count: `{len(heldout_groups)}`",
        f"- heldout_pair_count: `{len(test_pairs)}`",
        f"- behaviorally_diverse_heldout_groups: `{diverse_groups}`",
        f"- best_head: `{payload['best_head']}`",
        f"- previous_v1_head: `{payload['previous_v1_head']}`",
        f"- old_frozen_pairwise_accuracy: `{rate(old_pair_acc)}`",
        "",
        "Behaviorally diverse heldout groups are the load-bearing subset for this verdict.",
        "",
        "## Behaviorally Diverse Metrics",
        "",
    ]
    lines.extend(md_table(metric_rows[:180], ["policy", "config", "architecture", "groups", "top1", "reward", "regret", "top2_oracle"]))
    lines.extend(["", "## Heldout Pairwise By New Head", ""])
    lines.extend(
        md_table(
            [
                {
                    "head_id": row["head_id"],
                    "heldout_pairs": row["heldout_pairs"],
                    "pairwise": rate(row["pairwise_accuracy"]),
                    "val_pairwise": rate(row["validation_pairwise_accuracy"]),
                }
                for row in sorted(pairwise_by_head, key=lambda r: float(r["pairwise_accuracy"]) if math.isfinite(float(r["pairwise_accuracy"])) else -1, reverse=True)[:60]
            ],
            ["head_id", "heldout_pairs", "pairwise", "val_pairwise"],
        )
    )
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_TAP_EVAL_V2_VERDICT = {verdict}", flush=True)
    print(f"Wrote {rel(OUT_JSON)}", flush=True)
    return 0 if verdict not in {"INSUFFICIENT"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

