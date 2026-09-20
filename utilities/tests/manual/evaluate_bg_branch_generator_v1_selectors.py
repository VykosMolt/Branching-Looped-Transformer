"""Evaluate selectors on Branch Generator v1 heldout outputs."""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Sequence

import torch

from bg_branch_generator_v1_common import (
    PRIMARY_MINIMUMS_V1,
    SALVAGE_HEADS_PT,
    SELECTOR_DATASET_PT,
    SELECTOR_EVAL_CSV,
    SELECTOR_EVAL_JSON,
    SELECTOR_EVAL_MD,
    SELECTOR_HEADS_PT,
    V1_ROOT,
    V2_ROOT,
    V3_ROOT,
    aggregate_subsets,
    baseline_policy_rows,
    best_metric,
    best_primary_head,
    breakdown,
    compact_head_id,
    ensemble_policy_rows,
    ensure_bgv1_root,
    group_is_behaviorally_diverse_v2,
    group_is_reward_diverse_v2,
    group_rows,
    load_generator_rows,
    md_table,
    pairwise_accuracy_from_pairs,
    primary_safe_generator_row,
    rate,
    sampled_reward,
    score_matrix,
    stable_v2_row,
    tap_policy_rows,
    write_csv,
    write_json,
    write_md,
)
from bg_hidden_origin_tap_common import config_dim
from evaluate_bg_hidden_origin_taps import build_head


def best_heads_by_config(heads: Sequence[dict[str, Any]], variant: str = "generator_v1_only_primary_safe") -> list[dict[str, Any]]:
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in heads:
        if variant and row.get("variant") != variant:
            continue
        if row.get("flip_diagnostics", {}).get("passes"):
            by_config[str(row["config"])].append(row)
    return [max(items, key=lambda row: float(row["metrics"].get("validation_pairwise_accuracy", -1.0))) for items in by_config.values()]


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


def eval_subset(groups: list[list[dict[str, Any]]], heads: dict[str, dict[str, Any] | None], device: torch.device, subset: str) -> list[dict[str, Any]]:
    rows = baseline_policy_rows(groups, subset)
    rows.extend(tap_policy_rows(groups, heads.get("v1"), device, "v1_hidden_origin_tap", subset))
    rows.extend(tap_policy_rows(groups, heads.get("v2"), device, "v2_hidden_origin_tap", subset))
    rows.extend(tap_policy_rows(groups, heads.get("v3"), device, "v3_hidden_origin_tap", subset))
    rows.extend(tap_policy_rows(groups, heads.get("v4"), device, "v4_hidden_origin_tap", subset))
    rows.extend(tap_policy_rows(groups, heads.get("salvage"), device, "salvage_retrained_head", subset))
    rows.extend(tap_policy_rows(groups, heads.get("generator_v1"), device, "generator_v1_selector", subset))
    rows.extend(ensemble_policy_rows(groups, heads, device, subset))
    return rows


def support_ready(heldout_tasks: int, diverse_groups: int, heldout_pairs: int) -> bool:
    return (
        heldout_tasks >= PRIMARY_MINIMUMS_V1["heldout"]["task_ids"]
        and diverse_groups >= PRIMARY_MINIMUMS_V1["heldout"]["behaviorally_diverse_groups"]
        and heldout_pairs >= PRIMARY_MINIMUMS_V1["heldout"]["non_tie_pairs"]
    )


def choose_best_selector(metrics: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    candidates = list(metrics.values())
    if not candidates:
        return "insufficient", None
    best = max(candidates, key=lambda row: (float(row.get("top1_success", 0.0)), float(row.get("top2_oracle_coverage", 0.0)), -float(row.get("selection_regret", 999.0))))
    policy = str(best.get("policy"))
    if policy.startswith("old_frozen_bg"):
        name = "old_frozen_bg"
    elif policy.startswith("generator_v1"):
        name = "generator_v1_selector"
    elif policy.startswith("v4_hidden_origin"):
        name = "v4_hidden_origin_tap"
    elif policy.startswith("v3_hidden_origin"):
        name = "v3_hidden_origin_tap"
    elif policy.startswith("v2_hidden_origin"):
        name = "v2_hidden_origin_tap"
    elif policy.startswith("v1_hidden_origin"):
        name = "v1_hidden_origin_tap"
    elif policy.startswith("salvage"):
        name = "salvage_retrained_head"
    elif policy.startswith("ensemble"):
        name = "ensemble"
    elif policy.startswith("random"):
        name = "random"
    else:
        name = policy
    return name, best


def verdict_for(ready: bool, behavior_metrics: dict[str, Any]) -> tuple[str, str]:
    best_name, best = choose_best_selector(behavior_metrics)
    if not ready:
        return "STILL_DATA_LIMITED", best_name
    random = best_metric(behavior_metrics, "random_top1")
    clean = best_metric(behavior_metrics, "clean_branch_baseline")
    old = best_metric(behavior_metrics, "old_frozen_bg_pairwise_tournament")
    if not best or not random or not clean:
        return "NO_SELECTOR_SIGNAL", best_name
    best_top1 = float(best["top1_success"])
    random_top1 = float(random["top1_success"])
    clean_top1 = float(clean["top1_success"])
    old_top1 = float(old["top1_success"]) if old else float("-inf")
    if best_name == "old_frozen_bg" and best_top1 >= random_top1 and best_top1 >= clean_top1:
        return "OLD_TAPS_BEST", best_name
    if best_name == "ensemble" and best_top1 > random_top1 and best_top1 > clean_top1:
        return "ENSEMBLE_BEST", best_name
    if best_name == "generator_v1_selector" and best_top1 > random_top1 and best_top1 > clean_top1 and best_top1 >= old_top1:
        return "SELECTOR_READY", best_name
    if best_top1 > random_top1 or best_top1 > clean_top1:
        return "WEAK_SELECTOR", best_name
    return "NO_SELECTOR_SIGNAL", best_name


def main() -> int:
    ensure_bgv1_root()
    started = time.time()
    if not SELECTOR_DATASET_PT.exists():
        payload = {"BG_BRANCH_GENERATOR_V1_SELECTOR_EVAL_VERDICT": "INSUFFICIENT", "verdict": "INSUFFICIENT", "blocker": "missing dataset"}
        write_json(SELECTOR_EVAL_JSON, payload)
        write_md(SELECTOR_EVAL_MD, ["# Branch Generator V1 Selector Evaluation", "", "BG_BRANCH_GENERATOR_V1_SELECTOR_EVAL_VERDICT = INSUFFICIENT"])
        print("BG_BRANCH_GENERATOR_V1_SELECTOR_EVAL_VERDICT = INSUFFICIENT", flush=True)
        return 1
    dataset = torch.load(SELECTOR_DATASET_PT, map_location="cpu", weights_only=False)
    trained = torch.load(SELECTOR_HEADS_PT, map_location="cpu", weights_only=False) if SELECTOR_HEADS_PT.exists() else {"verdict": "SKIPPED", "heads": []}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    primary_pairs = list(dataset.get("pairs") or [])
    test_pairs = [pair for pair in primary_pairs if pair.get("split") == "heldout"]
    test_tasks = set((dataset.get("tasks_by_split") or {}).get("heldout", []))
    all_rows = load_generator_rows()
    heldout_primary_rows = [row for row in all_rows if str(row.get("task_id")) in test_tasks and primary_safe_generator_row(row)]
    heldout_groups = [vals for vals in group_rows(heldout_primary_rows).values() if len(vals) >= 2]
    diagnostic_rows = [row for row in all_rows if str(row.get("task_id")) in test_tasks and stable_v2_row(row) and float(row.get("alpha") or 0.0) == 0.02]
    diagnostic_groups = [vals for vals in group_rows(diagnostic_rows).values() if len(vals) >= 2]
    sampled_rows = [row for row in all_rows if str(row.get("task_id")) in test_tasks and sampled_reward(row) is not None and stable_v2_row(row)]
    sampled_groups = [vals for vals in group_rows(sampled_rows).values() if len(vals) >= 2]
    gen_heads = [row for row in list(trained.get("heads") or []) if row.get("variant") == "generator_v1_only_primary_safe" and row.get("flip_diagnostics", {}).get("passes")]
    heads = {
        "v1": best_primary_head(V1_ROOT / "hidden_origin_tap_heads.pt"),
        "v2": best_primary_head(V2_ROOT / "hidden_origin_tap_heads_v2.pt", "primary_safe_deterministic"),
        "v3": best_primary_head(V3_ROOT / "hidden_origin_tap_heads_v3.pt", "primary_safe_deterministic"),
        "v4": best_primary_head(SELECTOR_HEADS_PT.parent.parent / "bg_hidden_origin_quota_v4_2026-05-18" / "hidden_origin_tap_heads_v4.pt", "v4_only_primary_safe"),
        "salvage": best_primary_head(SALVAGE_HEADS_PT),
        "generator_v1": max(gen_heads, key=lambda row: float(row["metrics"].get("validation_pairwise_accuracy", -1.0))) if gen_heads else None,
    }
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
    pairwise_by_head: list[dict[str, Any]] = []
    pairwise_by_head.extend(pairwise_rows(best_heads_by_config(list(trained.get("heads") or []), "generator_v1_only_primary_safe"), test_pairs, device, "generator_v1"))
    for label, path, variant in (
        ("v1", V1_ROOT / "hidden_origin_tap_heads.pt", None),
        ("v2", V2_ROOT / "hidden_origin_tap_heads_v2.pt", "primary_safe_deterministic"),
        ("v3", V3_ROOT / "hidden_origin_tap_heads_v3.pt", "primary_safe_deterministic"),
        ("salvage", SALVAGE_HEADS_PT, None),
    ):
        head = best_primary_head(path, variant)
        if head:
            pairwise_by_head.extend(pairwise_rows([head], test_pairs, device, label))
    old_pair_acc = old_pairwise_accuracy(test_pairs)
    ready = support_ready(len(test_tasks), diverse_groups, len(test_pairs))
    verdict, best_selector = verdict_for(ready, behavior_metrics)
    payload = {
        "BG_BRANCH_GENERATOR_V1_SELECTOR_EVAL_VERDICT": verdict,
        "verdict": verdict,
        "HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE_AFTER_GENERATOR_V1_EVAL": best_selector,
        "training_verdict": trained.get("verdict"),
        "dataset_verdict": dataset.get("verdict"),
        "heldout_task_ids": sorted(test_tasks),
        "heldout_group_count": len(heldout_groups),
        "heldout_pair_count": len(test_pairs),
        "behaviorally_diverse_heldout_groups": diverse_groups,
        "readiness_support_met": ready,
        "best_heads": {key: compact_head_id(head) if head else None for key, head in heads.items()},
        "old_frozen_pairwise_accuracy": old_pair_acc,
        "pairwise_by_head": pairwise_by_head,
        **subsets,
        "per_domain_breakdown": breakdown(eval_rows, "domain"),
        "per_branch_point_breakdown": breakdown(eval_rows, "branch_point"),
        "per_alpha_breakdown": breakdown(eval_rows, "alpha_bucket"),
        "per_delta_family_breakdown": breakdown(eval_rows, "delta_family"),
        "per_generator_method_breakdown": breakdown(eval_rows, "generator_method"),
        "per_task_class_breakdown": breakdown(eval_rows, "task_screening_class"),
        "rows": eval_rows,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(SELECTOR_EVAL_JSON, payload)
    write_csv(SELECTOR_EVAL_CSV, eval_rows)
    metric_rows = sorted(
        [{"policy": row["policy"], "config": row["config"], "architecture": row["architecture"], "groups": row["groups"], "top1": rate(row["top1_success"]), "reward": rate(row["reward_mean"]), "regret": rate(row["selection_regret"]), "top2_oracle": rate(row["top2_oracle_coverage"])} for row in behavior_metrics.values()],
        key=lambda row: (row["policy"], row["config"], row["architecture"]),
    )
    lines = [
        "# Branch Generator V1 Selector Evaluation",
        "",
        f"BG_BRANCH_GENERATOR_V1_SELECTOR_EVAL_VERDICT = {verdict}",
        "",
        f"- heldout_task_ids: `{len(test_tasks)}`",
        f"- heldout_pair_count: `{len(test_pairs)}`",
        f"- behaviorally_diverse_heldout_groups: `{diverse_groups}`",
        f"- readiness_support_met: `{ready}`",
        f"- best_selector: `{best_selector}`",
        f"- old_frozen_pairwise_accuracy: `{rate(old_pair_acc)}`",
        "",
        "Readiness uses only primary-safe deterministic alpha <= 0.01 heldout groups.",
        "",
        "## Behaviorally Diverse Metrics",
        "",
    ]
    lines.extend(md_table(metric_rows[:240], ["policy", "config", "architecture", "groups", "top1", "reward", "regret", "top2_oracle"]))
    lines.extend(["", "## Heldout Pairwise", ""])
    lines.extend(md_table([{**row, "pairwise_accuracy": rate(row["pairwise_accuracy"]), "validation_pairwise_accuracy": rate(row["validation_pairwise_accuracy"])} for row in pairwise_by_head[:160]], ["source", "head_id", "heldout_pairs", "config", "architecture", "pairwise_accuracy", "validation_pairwise_accuracy"]))
    write_md(SELECTOR_EVAL_MD, lines)
    print(f"BG_BRANCH_GENERATOR_V1_SELECTOR_EVAL_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "INSUFFICIENT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
