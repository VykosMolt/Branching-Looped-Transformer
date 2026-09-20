"""Heldout evaluation for hidden-origin tap selectors v3."""
from __future__ import annotations

import math
import time
from collections import defaultdict
from statistics import mean
from typing import Any, Sequence

import torch

from bg_hidden_origin_diversity_v3_common import (
    DATASET_V3_PT,
    HEADS_V3_PT,
    V1_ROOT,
    V2_ROOT,
    V3_ROOT,
    compact_head_id,
    config_vector_from_row,
    diagnostic_alpha_v3_row,
    ensure_v3_root,
    group_is_behaviorally_diverse_v2,
    group_is_reward_diverse_v2,
    group_rows,
    load_all_v3_branch_rows,
    load_head_rows,
    md_table,
    primary_safe_v3_row,
    rate,
    rel,
    sampled_reward,
    stable_v2_row,
    write_csv,
    write_json,
    write_md,
)
from bg_hidden_origin_tap_common import config_dim, pairwise_accuracy_from_pairs, ranking_from_matrix, score_matrix, top2_random_reward, top2_random_success
from evaluate_bg_hidden_origin_taps import aggregate, best_metric, build_head, old_pairwise_accuracy, selected_metric


OUT_JSON = V3_ROOT / "heldout_eval_v3.json"
OUT_MD = V3_ROOT / "heldout_eval_v3.md"
OUT_CSV = V3_ROOT / "heldout_eval_v3_rows.csv"


def best_heads_by_config(heads: Sequence[dict[str, Any]], variant: str = "primary_safe_deterministic") -> list[dict[str, Any]]:
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in heads:
        if variant and row.get("variant") != variant:
            continue
        if row.get("flip_diagnostics", {}).get("passes"):
            by_config[str(row["config"])].append(row)
    return [max(items, key=lambda row: float(row["metrics"].get("validation_pairwise_accuracy", -1.0))) for items in by_config.values()]


def best_primary_head(path: Any, variant: str | None = None) -> dict[str, Any] | None:
    rows = load_head_rows(path, variant=variant, only_passing=True)
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("metrics", {}).get("validation_pairwise_accuracy", -1.0)))


def annotate_metric_row(group: Sequence[dict[str, Any]], row: dict[str, Any], subset: str) -> dict[str, Any]:
    first = group[0]
    out = dict(row)
    out.update(
        {
            "branch_group_id": first.get("branch_group_id"),
            "task_id": first.get("task_id"),
            "domain": first.get("domain"),
            "branch_point": first.get("branch_point"),
            "alpha": first.get("alpha"),
            "alpha_bucket": first.get("alpha_bucket"),
            "delta_family": first.get("primary_delta_family") or first.get("delta_family"),
            "task_screening_class": first.get("task_screening_class"),
            "split_guard_role": first.get("split_guard_role"),
            "behaviorally_diverse": group_is_behaviorally_diverse_v2(group),
            "reward_diverse": group_is_reward_diverse_v2(group),
            "subset": subset,
        }
    )
    return out


def baseline_policy_rows(groups: Sequence[list[dict[str, Any]]], subset: str) -> list[dict[str, Any]]:
    rows = []
    for group in groups:
        ordered = sorted(group, key=lambda row: int(row["branch_id"]))
        rewards = [float(row.get("deterministic_reward", row.get("reward", 0.0))) for row in ordered]
        correct = [1.0 if row.get("deterministic_correct", row.get("correct")) else 0.0 for row in ordered]
        oracle_reward = max(rewards)
        oracle_ids = {int(row["branch_id"]) for row in ordered if float(row.get("deterministic_reward", row.get("reward", 0.0))) == oracle_reward}
        policies = {
            "clean_branch_baseline": selected_metric(ordered, [0]),
            "old_frozen_bg_highest_tap_score": selected_metric(ordered, [max(range(len(ordered)), key=lambda i: (float(ordered[i].get("tap_margin_sum", ordered[i].get("old_frozen_tap_score", 0.0))), -int(ordered[i]["branch_id"])))]),
            "old_frozen_bg_pairwise_tournament": selected_metric(ordered, [max(range(len(ordered)), key=lambda i: (float(ordered[i].get("tap_margin_sum", ordered[i].get("old_frozen_tap_score", 0.0))), -int(ordered[i]["branch_id"])))]),
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
                annotate_metric_row(
                    ordered,
                    {
                        "policy": policy,
                        "config": "baseline",
                        "architecture": "baseline",
                        **metric,
                    },
                    subset,
                )
            )
    return rows


def tap_policy_rows(groups: Sequence[list[dict[str, Any]]], head_row: dict[str, Any] | None, device: torch.device, label: str, subset: str) -> list[dict[str, Any]]:
    if head_row is None:
        return []
    head = build_head(head_row, device)
    config = str(head_row["config"])
    rows = []
    for group in groups:
        ordered = sorted(group, key=lambda row: int(row["branch_id"]))
        vectors = [config_vector_from_row(row, config) for row in ordered]
        if not vectors or not all(isinstance(vec, torch.Tensor) and int(vec.numel()) == config_dim(config) for vec in vectors):
            continue
        mat = score_matrix(head, [vec for vec in vectors if isinstance(vec, torch.Tensor)], device)
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
                annotate_metric_row(
                    ordered,
                    {
                        "policy": policy,
                        "config": config,
                        "architecture": head_row["architecture"],
                        "head_id": compact_head_id(head_row),
                        **metric,
                    },
                    subset,
                )
            )
    head.to("cpu")
    return rows


def eval_subset(groups: list[list[dict[str, Any]]], heads: dict[str, dict[str, Any] | None], device: torch.device, subset: str) -> list[dict[str, Any]]:
    rows = baseline_policy_rows(groups, subset)
    rows.extend(tap_policy_rows(groups, heads.get("v1"), device, "previous_hidden_origin_tap_v1", subset))
    rows.extend(tap_policy_rows(groups, heads.get("v2"), device, "previous_hidden_origin_tap_v2", subset))
    rows.extend(tap_policy_rows(groups, heads.get("v3"), device, "new_hidden_origin_tap_v3", subset))
    return rows


def aggregate_subsets(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for subset in sorted({str(row.get("subset")) for row in rows}):
        vals = [row for row in rows if str(row.get("subset")) == subset]
        out[f"{subset}_metrics"] = aggregate(vals)
        behavior = [row for row in vals if bool(row.get("behaviorally_diverse"))]
        reward = [row for row in vals if bool(row.get("reward_diverse"))]
        out[f"{subset}_behaviorally_diverse_metrics"] = aggregate(behavior)
        out[f"{subset}_reward_diverse_metrics"] = aggregate(reward)
    return out


def breakdown(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    out = {}
    for value in sorted({str(row.get(field)) for row in rows}):
        subset = [row for row in rows if str(row.get(field)) == value]
        out[value] = aggregate(subset)
    return out


def pairwise_rows(heads: list[dict[str, Any]], test_pairs: list[dict[str, Any]], device: torch.device, source: str) -> list[dict[str, Any]]:
    out = []
    for head_row in heads:
        head = build_head(head_row, device)
        cfg_pairs = [pair for pair in test_pairs if head_row["config"] in pair.get("features", {})]
        out.append(
            {
                "source": source,
                "head_id": compact_head_id(head_row),
                "variant": head_row.get("variant"),
                "config": head_row["config"],
                "architecture": head_row["architecture"],
                "heldout_pairs": len(cfg_pairs),
                "pairwise_accuracy": pairwise_accuracy_from_pairs(head, cfg_pairs, head_row["config"], device),
                "validation_pairwise_accuracy": head_row.get("metrics", {}).get("validation_pairwise_accuracy"),
            }
        )
        head.to("cpu")
    return out


def verdict_for(dataset_verdict: str, training_verdict: str, heldout_tasks: int, heldout_pairs: int, diverse_groups: int, behavior_metrics: dict[str, Any]) -> str:
    if heldout_tasks < 6 or heldout_pairs < 80 or diverse_groups < 15:
        return "DATA_LIMITED"
    best_new = best_metric(behavior_metrics, "new_hidden_origin_tap_v3_pairwise_tournament")
    random = best_metric(behavior_metrics, "random_top1")
    old = best_metric(behavior_metrics, "old_frozen_bg_pairwise_tournament")
    if best_new and random and old:
        new_top1 = float(best_new["top1_success"])
        random_top1 = float(random["top1_success"])
        old_top1 = float(old["top1_success"])
        if new_top1 > random_top1 and new_top1 > old_top1:
            return "SELECTOR_READY"
        if new_top1 >= random_top1 or new_top1 >= old_top1:
            return "WEAK_SELECTOR"
        if training_verdict in {"READY", "WEAK"} and dataset_verdict in {"READY", "SMALL_BUT_USABLE"}:
            return "OVERFIT"
    return "NO_SELECTOR_SIGNAL"


def main() -> int:
    ensure_v3_root()
    started = time.time()
    if not DATASET_V3_PT.exists():
        payload = {"BG_HIDDEN_ORIGIN_TAP_EVAL_V3_VERDICT": "INSUFFICIENT", "blocker": "missing dataset v3"}
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# Hidden-Origin Tap Heldout Evaluation V3", "", "BG_HIDDEN_ORIGIN_TAP_EVAL_V3_VERDICT = INSUFFICIENT"])
        print("BG_HIDDEN_ORIGIN_TAP_EVAL_V3_VERDICT = INSUFFICIENT", flush=True)
        return 1
    dataset = torch.load(DATASET_V3_PT, map_location="cpu", weights_only=False)
    trained = torch.load(HEADS_V3_PT, map_location="cpu", weights_only=False) if HEADS_V3_PT.exists() else {"verdict": "INSUFFICIENT", "heads": []}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    primary_pairs = list(dataset.get("pairs") or [])
    test_pairs = [pair for pair in primary_pairs if pair.get("split") == "test"]
    test_tasks = set((dataset.get("tasks_by_split") or {}).get("test", []))
    all_rows = load_all_v3_branch_rows(include_prior=True)
    heldout_primary_rows = [row for row in all_rows if str(row.get("task_id")) in test_tasks and primary_safe_v3_row(row)]
    heldout_groups = [vals for vals in group_rows(heldout_primary_rows).values() if len(vals) >= 2]
    diagnostic_alpha_rows = [row for row in all_rows if str(row.get("task_id")) in test_tasks and diagnostic_alpha_v3_row(row)]
    diagnostic_groups = [vals for vals in group_rows(diagnostic_alpha_rows).values() if len(vals) >= 2]
    sampled_rows = [row for row in all_rows if str(row.get("task_id")) in test_tasks and sampled_reward(row) is not None and stable_v2_row(row)]
    sampled_groups = [vals for vals in group_rows(sampled_rows).values() if len(vals) >= 2]

    v1_head = best_primary_head(V1_ROOT / "hidden_origin_tap_heads.pt")
    v2_head = best_primary_head(V2_ROOT / "hidden_origin_tap_heads_v2.pt", "primary_safe_deterministic")
    v3_heads = [row for row in list(trained.get("heads") or []) if row.get("variant") == "primary_safe_deterministic" and row.get("flip_diagnostics", {}).get("passes")]
    v3_head = max(v3_heads, key=lambda row: float(row["metrics"].get("validation_pairwise_accuracy", -1.0))) if v3_heads else None
    heads = {"v1": v1_head, "v2": v2_head, "v3": v3_head}

    eval_rows = []
    eval_rows.extend(eval_subset(heldout_groups, heads, device, "primary_safe_deterministic"))
    eval_rows.extend(eval_subset([g for g in heldout_groups if group_is_behaviorally_diverse_v2(g)], heads, device, "behaviorally_diverse"))
    eval_rows.extend(eval_subset([g for g in heldout_groups if group_is_reward_diverse_v2(g)], heads, device, "reward_diverse"))
    eval_rows.extend(eval_subset(diagnostic_groups, heads, device, "alpha_0_02_diagnostic"))
    eval_rows.extend(eval_subset(sampled_groups, heads, device, "sampled_expected_diagnostic"))

    subsets = aggregate_subsets(eval_rows)
    behavior_metrics = subsets.get("behaviorally_diverse_metrics", {})
    behavior_rows = [row for row in eval_rows if row.get("subset") == "behaviorally_diverse" and row.get("policy") == "random_top1"]
    diverse_groups = len({row["branch_group_id"] for row in behavior_rows})
    pairwise_by_head = []
    pairwise_by_head.extend(pairwise_rows(best_heads_by_config(load_head_rows(HEADS_V3_PT, variant="primary_safe_deterministic")), test_pairs, device, "v3"))
    if v1_head:
        pairwise_by_head.extend(pairwise_rows([v1_head], test_pairs, device, "v1"))
    if v2_head:
        pairwise_by_head.extend(pairwise_rows([v2_head], test_pairs, device, "v2"))
    old_pair_acc = old_pairwise_accuracy(test_pairs)
    verdict = verdict_for(str(dataset.get("verdict")), str(trained.get("verdict")), len(test_tasks), len(test_pairs), diverse_groups, behavior_metrics)
    payload = {
        "BG_HIDDEN_ORIGIN_TAP_EVAL_V3_VERDICT": verdict,
        "verdict": verdict,
        "dataset": rel(DATASET_V3_PT),
        "heads": rel(HEADS_V3_PT),
        "training_verdict": trained.get("verdict"),
        "dataset_verdict": dataset.get("verdict"),
        "heldout_task_ids": sorted(test_tasks),
        "heldout_group_count": len(heldout_groups),
        "heldout_pair_count": len(test_pairs),
        "behaviorally_diverse_heldout_groups": diverse_groups,
        "diagnostic_alpha_heldout_group_count": len(diagnostic_groups),
        "sampled_expected_heldout_group_count": len(sampled_groups),
        "best_v3_head": compact_head_id(v3_head) if v3_head else None,
        "previous_v1_head": compact_head_id(v1_head) if v1_head else None,
        "previous_v2_head": compact_head_id(v2_head) if v2_head else None,
        "old_frozen_pairwise_accuracy": old_pair_acc,
        "new_pairwise_by_head": pairwise_by_head,
        **subsets,
        "per_domain_breakdown": breakdown(eval_rows, "domain"),
        "per_branch_point_breakdown": breakdown(eval_rows, "branch_point"),
        "per_alpha_breakdown": breakdown(eval_rows, "alpha_bucket"),
        "per_delta_family_breakdown": breakdown(eval_rows, "delta_family"),
        "per_task_class_breakdown": breakdown(eval_rows, "task_screening_class"),
        "rows": eval_rows,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    write_csv(OUT_CSV, eval_rows)
    metric_rows = sorted(
        [
            {
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
        "# Hidden-Origin Tap Heldout Evaluation V3",
        "",
        f"BG_HIDDEN_ORIGIN_TAP_EVAL_V3_VERDICT = {verdict}",
        "",
        f"- heldout_task_ids: `{len(test_tasks)}`",
        f"- heldout_group_count: `{len(heldout_groups)}`",
        f"- heldout_pair_count: `{len(test_pairs)}`",
        f"- behaviorally_diverse_heldout_groups: `{diverse_groups}`",
        f"- best_v3_head: `{payload['best_v3_head']}`",
        f"- previous_v1_head: `{payload['previous_v1_head']}`",
        f"- previous_v2_head: `{payload['previous_v2_head']}`",
        f"- old_frozen_pairwise_accuracy: `{rate(old_pair_acc)}`",
        "",
        "Behaviorally diverse primary-safe heldout groups are the load-bearing subset for this verdict.",
        "",
        "## Behaviorally Diverse Metrics",
        "",
    ]
    lines.extend(md_table(metric_rows[:180], ["policy", "config", "architecture", "groups", "top1", "reward", "regret", "top2_oracle"]))
    lines.extend(["", "## Heldout Pairwise", ""])
    lines.extend(md_table(
        [
            {
                "source": row["source"],
                "head_id": row["head_id"],
                "heldout_pairs": row["heldout_pairs"],
                "pairwise": rate(row["pairwise_accuracy"]),
                "val_pairwise": rate(row["validation_pairwise_accuracy"]),
            }
            for row in sorted(pairwise_by_head, key=lambda r: float(r["pairwise_accuracy"]) if math.isfinite(float(r["pairwise_accuracy"])) else -1, reverse=True)[:80]
        ],
        ["source", "head_id", "heldout_pairs", "pairwise", "val_pairwise"],
    ))
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_TAP_EVAL_V3_VERDICT = {verdict}", flush=True)
    print(f"Wrote {rel(OUT_JSON)}", flush=True)
    return 0 if verdict != "INSUFFICIENT" else 1


if __name__ == "__main__":
    raise SystemExit(main())

