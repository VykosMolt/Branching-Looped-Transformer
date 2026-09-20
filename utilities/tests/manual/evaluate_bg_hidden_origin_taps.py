"""Heldout evaluation for hidden-origin branch tap selectors."""
from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Sequence

import torch

from bg_hidden_origin_tap_common import (
    HEAD_CLASSES,
    OUT_ROOT,
    config_dim,
    config_vector_from_row,
    ensure_out_root,
    group_is_behaviorally_diverse,
    group_is_reward_diverse,
    group_rows,
    is_safe_alpha,
    load_all_branch_rows,
    md_table,
    pairwise_accuracy_from_pairs,
    ranking_from_matrix,
    rate,
    rel,
    score_matrix,
    stable_row,
    top2_random_reward,
    top2_random_success,
    write_csv,
    write_json,
    write_md,
)


DATASET_PT = OUT_ROOT / "hidden_origin_tap_dataset.pt"
HEADS_PT = OUT_ROOT / "hidden_origin_tap_heads.pt"
OUT_JSON = OUT_ROOT / "heldout_eval.json"
OUT_MD = OUT_ROOT / "heldout_eval.md"
OUT_CSV = OUT_ROOT / "heldout_eval_rows.csv"


def build_head(row: dict[str, Any], device: torch.device) -> torch.nn.Module:
    head = HEAD_CLASSES[row["architecture"]](config_dim(row["config"]))
    head.load_state_dict(row["state_dict"])
    head.to(device=device, dtype=torch.float32)
    head.eval()
    return head


def compact_head_id(row: dict[str, Any]) -> str:
    m = row.get("metrics", {})
    return f"{row['config']}::{row['architecture']}::seed={m.get('seed')}::lr={m.get('lr')}"


def best_heads_by_config(heads: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in heads:
        if row.get("flip_diagnostics", {}).get("passes"):
            by_config[row["config"]].append(row)
    out = []
    for items in by_config.values():
        out.append(max(items, key=lambda row: float(row["metrics"].get("validation_pairwise_accuracy", -1.0))))
    return out


def branch_reward(row: dict[str, Any]) -> float:
    return float(row.get("reward", 0.0))


def selected_metric(group: Sequence[dict[str, Any]], selected_indices: Sequence[int]) -> dict[str, Any]:
    rows = list(group)
    selected = [rows[idx] for idx in selected_indices if 0 <= idx < len(rows)]
    if not selected:
        selected = [rows[0]]
    oracle_reward = max(branch_reward(row) for row in rows)
    oracle_branch_ids = {int(row["branch_id"]) for row in rows if branch_reward(row) == oracle_reward}
    selected_branch_ids = [int(row["branch_id"]) for row in selected]
    selected_reward = max(branch_reward(row) for row in selected)
    selected_success = any(bool(row.get("correct")) for row in selected)
    kept_oracle = any(branch_id in oracle_branch_ids for branch_id in selected_branch_ids)
    return {
        "success": 1.0 if selected_success else 0.0,
        "reward": selected_reward,
        "oracle_coverage": 1.0 if kept_oracle else 0.0,
        "oracle_gap": oracle_reward - selected_reward,
        "selection_regret": oracle_reward - selected_reward,
        "pruned_oracle_branch_rate": 0.0 if kept_oracle else 1.0,
        "kept_good_branch_rate": 1.0 if kept_oracle else 0.0,
        "selected_branch_ids": selected_branch_ids,
        "oracle_branch_ids": sorted(oracle_branch_ids),
        "survivors": len(selected),
    }


def baseline_policy_rows(groups: Sequence[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in groups:
        ordered = sorted(group, key=lambda row: int(row["branch_id"]))
        gid = ordered[0]["branch_group_id"]
        rewards = [branch_reward(row) for row in ordered]
        correct = [1.0 if row.get("correct") else 0.0 for row in ordered]
        oracle_reward = max(rewards)
        oracle_ids = {int(row["branch_id"]) for row in ordered if branch_reward(row) == oracle_reward}
        policies = {
            "clean_branch_baseline": selected_metric(ordered, [0]),
            "old_frozen_bg_highest_tap_score": selected_metric(ordered, [max(range(len(ordered)), key=lambda i: (float(ordered[i].get("tap_margin_sum", 0.0)), -int(ordered[i]["branch_id"])))]),
            "old_frozen_bg_pairwise_tournament": selected_metric(ordered, [max(range(len(ordered)), key=lambda i: (float(ordered[i].get("tap_margin_sum", 0.0)), -int(ordered[i]["branch_id"])))]),
            "simple_top2_branch_id_order": selected_metric(ordered, list(range(min(2, len(ordered))))),
        }
        policies["random_top1"] = {
            "success": float(mean(correct)),
            "reward": float(mean(rewards)),
            "oracle_coverage": len(oracle_ids) / max(len(ordered), 1),
            "oracle_gap": oracle_reward - float(mean(rewards)),
            "selection_regret": oracle_reward - float(mean(rewards)),
            "pruned_oracle_branch_rate": 1.0 - len(oracle_ids) / max(len(ordered), 1),
            "kept_good_branch_rate": len(oracle_ids) / max(len(ordered), 1),
            "selected_branch_ids": [],
            "oracle_branch_ids": sorted(oracle_ids),
            "survivors": 1,
        }
        policies["random_top2"] = {
            "success": top2_random_success(ordered),
            "reward": top2_random_reward(ordered),
            "oracle_coverage": min(1.0, 2.0 * len(oracle_ids) / max(len(ordered), 1)),
            "oracle_gap": oracle_reward - top2_random_reward(ordered),
            "selection_regret": oracle_reward - top2_random_reward(ordered),
            "pruned_oracle_branch_rate": 1.0 - min(1.0, 2.0 * len(oracle_ids) / max(len(ordered), 1)),
            "kept_good_branch_rate": min(1.0, 2.0 * len(oracle_ids) / max(len(ordered), 1)),
            "selected_branch_ids": [],
            "oracle_branch_ids": sorted(oracle_ids),
            "survivors": 2,
        }
        for policy, metric in policies.items():
            rows.append(
                {
                    "branch_group_id": gid,
                    "task_id": ordered[0].get("task_id"),
                    "domain": ordered[0].get("domain"),
                    "branch_point": ordered[0].get("branch_point"),
                    "alpha": ordered[0].get("alpha"),
                    "behaviorally_diverse": group_is_behaviorally_diverse(ordered),
                    "reward_diverse": group_is_reward_diverse(ordered),
                    "policy": policy,
                    "config": "baseline",
                    "architecture": "baseline",
                    **metric,
                }
            )
    return rows


def new_tap_policy_rows(groups: Sequence[list[dict[str, Any]]], head_row: dict[str, Any], device: torch.device, label: str) -> list[dict[str, Any]]:
    head = build_head(head_row, device)
    config = head_row["config"]
    rows: list[dict[str, Any]] = []
    for group in groups:
        ordered = sorted(group, key=lambda row: int(row["branch_id"]))
        vectors = [config_vector_from_row(row, config) for row in ordered]
        if not all(isinstance(v, torch.Tensor) for v in vectors):
            continue
        mat = score_matrix(head, [v for v in vectors if isinstance(v, torch.Tensor)], device)
        ranked = ranking_from_matrix(mat)
        ranking = list(ranked["ranking"])
        if not ranking:
            continue
        top1 = selected_metric(ordered, [ranking[0]])
        top2 = selected_metric(ordered, ranking[:2])
        for policy, metric in (
            (f"{label}_highest_score", top1),
            (f"{label}_pairwise_tournament", top1),
            (f"{label}_top2", top2),
        ):
            rows.append(
                {
                    "branch_group_id": ordered[0]["branch_group_id"],
                    "task_id": ordered[0].get("task_id"),
                    "domain": ordered[0].get("domain"),
                    "branch_point": ordered[0].get("branch_point"),
                    "alpha": ordered[0].get("alpha"),
                    "behaviorally_diverse": group_is_behaviorally_diverse(ordered),
                    "reward_diverse": group_is_reward_diverse(ordered),
                    "policy": policy,
                    "config": config,
                    "architecture": head_row["architecture"],
                    "head_id": compact_head_id(head_row),
                    **metric,
                }
            )
    head.to("cpu")
    return rows


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["policy"], row.get("config", ""), row.get("architecture", ""))].append(row)
    for (policy, config, architecture), vals in sorted(grouped.items()):
        out_key = f"{policy}::{config}::{architecture}"
        out[out_key] = {
            "policy": policy,
            "config": config,
            "architecture": architecture,
            "groups": len(vals),
            "top1_success": float(mean(float(v["success"]) for v in vals)),
            "top2_oracle_coverage": float(mean(float(v["oracle_coverage"]) for v in vals)),
            "reward_mean": float(mean(float(v["reward"]) for v in vals)),
            "oracle_gap": float(mean(float(v["oracle_gap"]) for v in vals)),
            "selection_regret": float(mean(float(v["selection_regret"]) for v in vals)),
            "pruned_oracle_branch_rate": float(mean(float(v["pruned_oracle_branch_rate"]) for v in vals)),
            "kept_good_branch_rate": float(mean(float(v["kept_good_branch_rate"]) for v in vals)),
            "average_survivors": float(mean(float(v["survivors"]) for v in vals)),
        }
    return out


def breakdown(rows: Sequence[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    out = {}
    for value in sorted({str(row.get(field)) for row in rows}):
        subset = [row for row in rows if str(row.get(field)) == value]
        out[value] = aggregate(subset)
    return out


def old_pairwise_accuracy(test_pairs: Sequence[dict[str, Any]]) -> float:
    if not test_pairs:
        return float("nan")
    correct = 0
    total = 0
    for pair in test_pairs:
        pref = float(pair.get("old_frozen_tap_score_preferred", 0.0))
        rej = float(pair.get("old_frozen_tap_score_rejected", 0.0))
        correct += int(pref > rej)
        total += 1
    return correct / max(total, 1)


def best_metric(metrics: dict[str, Any], policy_prefix: str, subset_required_groups: int = 1) -> dict[str, Any] | None:
    candidates = [
        row for row in metrics.values()
        if str(row.get("policy", "")).startswith(policy_prefix) and int(row.get("groups", 0)) >= subset_required_groups
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (float(row["top1_success"]), float(row["reward_mean"]), -float(row["selection_regret"])))


def main() -> int:
    ensure_out_root()
    started = time.time()
    if not DATASET_PT.exists() or not HEADS_PT.exists():
        payload = {"BG_HIDDEN_ORIGIN_TAP_EVAL_VERDICT": "INSUFFICIENT", "blocker": "missing dataset or heads"}
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# Hidden-Origin Tap Heldout Evaluation", "", "BG_HIDDEN_ORIGIN_TAP_EVAL_VERDICT = INSUFFICIENT"])
        print("BG_HIDDEN_ORIGIN_TAP_EVAL_VERDICT = INSUFFICIENT", flush=True)
        return 1
    dataset = torch.load(DATASET_PT, map_location="cpu", weights_only=False)
    trained = torch.load(HEADS_PT, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = list(dataset.get("pairs") or [])
    test_pairs = [pair for pair in pairs if pair.get("split") == "test"]
    test_tasks = set(dataset.get("tasks_by_split", {}).get("test", []))
    all_rows = load_all_branch_rows()
    heldout_rows = [row for row in all_rows if str(row.get("task_id")) in test_tasks and is_safe_alpha(row) and stable_row(row)]
    heldout_groups = [vals for vals in group_rows(heldout_rows).values() if len(vals) >= 2]

    heads = list(trained.get("heads") or [])
    valid_heads = [row for row in heads if row.get("flip_diagnostics", {}).get("passes")]
    best_head = None
    if valid_heads:
        best_head = max(valid_heads, key=lambda row: float(row["metrics"].get("validation_pairwise_accuracy", -1.0)))
    eval_head_rows = best_heads_by_config(heads)
    if best_head is not None and all(compact_head_id(row) != compact_head_id(best_head) for row in eval_head_rows):
        eval_head_rows.append(best_head)

    eval_rows = baseline_policy_rows(heldout_groups)
    if best_head is not None:
        eval_rows.extend(new_tap_policy_rows(heldout_groups, best_head, device, "new_hidden_origin_tap"))
    for head_row in eval_head_rows:
        eval_rows.extend(new_tap_policy_rows(heldout_groups, head_row, device, f"new_hidden_origin_tap_config_{head_row['config']}"))

    all_metrics = aggregate(eval_rows)
    behavior_rows = [row for row in eval_rows if bool(row.get("behaviorally_diverse"))]
    reward_rows = [row for row in eval_rows if bool(row.get("reward_diverse"))]
    behavior_metrics = aggregate(behavior_rows)
    reward_metrics = aggregate(reward_rows)

    pairwise_by_head = []
    for head_row in eval_head_rows:
        head = build_head(head_row, device)
        cfg_pairs = [pair for pair in test_pairs if head_row["config"] in pair.get("features", {})]
        pairwise_by_head.append(
            {
                "head_id": compact_head_id(head_row),
                "config": head_row["config"],
                "architecture": head_row["architecture"],
                "heldout_pairs": len(cfg_pairs),
                "pairwise_accuracy": pairwise_accuracy_from_pairs(head, cfg_pairs, head_row["config"], device),
                "validation_pairwise_accuracy": head_row["metrics"].get("validation_pairwise_accuracy"),
            }
        )
        head.to("cpu")

    old_pair_acc = old_pairwise_accuracy(test_pairs)
    best_new_behavior = best_metric(behavior_metrics, "new_hidden_origin_tap_pairwise_tournament")
    random_behavior = best_metric(behavior_metrics, "random_top1")
    old_behavior = best_metric(behavior_metrics, "old_frozen_bg_pairwise_tournament")
    diverse_groups = len({row["branch_group_id"] for row in behavior_rows if row["policy"] == "random_top1"})
    if not heldout_groups or len(test_pairs) < 10 or len(test_tasks) < 4 or diverse_groups < 2:
        verdict = "INSUFFICIENT"
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
        "BG_HIDDEN_ORIGIN_TAP_EVAL_VERDICT": verdict,
        "verdict": verdict,
        "dataset": rel(DATASET_PT),
        "heads": rel(HEADS_PT),
        "training_verdict": trained.get("verdict"),
        "heldout_task_ids": sorted(test_tasks),
        "heldout_group_count": len(heldout_groups),
        "heldout_pair_count": len(test_pairs),
        "behaviorally_diverse_heldout_groups": diverse_groups,
        "best_head": compact_head_id(best_head) if best_head else None,
        "old_frozen_pairwise_accuracy": old_pair_acc,
        "new_pairwise_by_head": pairwise_by_head,
        "metrics": all_metrics,
        "behaviorally_diverse_metrics": behavior_metrics,
        "reward_diverse_metrics": reward_metrics,
        "per_domain_breakdown": breakdown(eval_rows, "domain"),
        "per_branch_point_breakdown": breakdown(eval_rows, "branch_point"),
        "per_alpha_breakdown": breakdown(eval_rows, "alpha"),
        "per_config_breakdown": breakdown(eval_rows, "config"),
        "rows": eval_rows,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    write_csv(OUT_CSV, eval_rows)

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
        "# Hidden-Origin Tap Heldout Evaluation",
        "",
        f"BG_HIDDEN_ORIGIN_TAP_EVAL_VERDICT = {verdict}",
        "",
        f"- heldout_group_count: `{len(heldout_groups)}`",
        f"- heldout_pair_count: `{len(test_pairs)}`",
        f"- behaviorally_diverse_heldout_groups: `{diverse_groups}`",
        f"- best_head: `{payload['best_head']}`",
        f"- old_frozen_pairwise_accuracy: `{rate(old_pair_acc)}`",
        "",
        "Behaviorally diverse groups are the load-bearing subset for this verdict.",
        "",
        "## Behaviorally Diverse Metrics",
        "",
    ]
    lines.extend(md_table(metric_rows[:120], ["policy", "config", "architecture", "groups", "top1", "reward", "regret", "top2_oracle"]))
    lines.extend(["", "## Heldout Pairwise By New Head", ""])
    lines.extend(md_table(
        [
            {
                "head_id": row["head_id"],
                "heldout_pairs": row["heldout_pairs"],
                "pairwise": rate(row["pairwise_accuracy"]),
                "val_pairwise": rate(row["validation_pairwise_accuracy"]),
            }
            for row in sorted(pairwise_by_head, key=lambda r: float(r["pairwise_accuracy"]) if math.isfinite(float(r["pairwise_accuracy"])) else -1, reverse=True)[:40]
        ],
        ["head_id", "heldout_pairs", "pairwise", "val_pairwise"],
    ))
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_TAP_EVAL_VERDICT = {verdict}", flush=True)
    print(f"Wrote {rel(OUT_JSON)}", flush=True)
    print(f"Wrote {rel(OUT_MD)}", flush=True)
    print(f"Wrote {rel(OUT_CSV)}", flush=True)
    return 0 if verdict != "INSUFFICIENT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
