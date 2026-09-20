"""Build task-disjoint pairwise hidden-origin tap datasets v3."""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from typing import Any, Callable

import torch

from bg_hidden_origin_diversity_v3_common import (
    CONFIGS,
    DATASET_V3_PT,
    V3_ROOT,
    alpha_bucket,
    available_configs_for_rows,
    config_dim,
    config_vector_from_row,
    deterministic_correct,
    deterministic_reward,
    diagnostic_alpha_v3_row,
    ensure_v3_root,
    group_is_behaviorally_diverse_v2,
    group_is_reward_diverse_v2,
    group_rows,
    load_all_v3_branch_rows,
    load_json,
    md_table,
    primary_safe_v3_row,
    rel,
    row_reward,
    sampled_reward,
    split_role_for_task,
    stable_v2_row,
    write_json,
    write_md,
)


OUT_PT = DATASET_V3_PT
OUT_JSON = V3_ROOT / "hidden_origin_tap_dataset_v3.json"
OUT_MD = V3_ROOT / "hidden_origin_tap_dataset_v3.md"


def pair_metadata(
    pref: dict[str, Any],
    rej: dict[str, Any],
    *,
    pair_id: str,
    split: str,
    variant: str,
    label_source: str,
    features: dict[str, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "variant": variant,
        "label_source": label_source,
        "task_id": pref["task_id"],
        "branch_group_id": pref["branch_group_id"],
        "domain": pref.get("domain"),
        "preferred_branch_id": int(pref["branch_id"]),
        "rejected_branch_id": int(rej["branch_id"]),
        "reward_preferred": row_reward(pref, label_source),
        "reward_rejected": row_reward(rej, label_source),
        "reward_gap": row_reward(pref, label_source) - row_reward(rej, label_source),
        "deterministic_reward_preferred": deterministic_reward(pref),
        "deterministic_reward_rejected": deterministic_reward(rej),
        "correctness_preferred": deterministic_correct(pref),
        "correctness_rejected": deterministic_correct(rej),
        "branch_point_preferred": pref.get("branch_point"),
        "branch_point_rejected": rej.get("branch_point"),
        "alpha_preferred": float(pref.get("alpha", 0.0)),
        "alpha_rejected": float(rej.get("alpha", 0.0)),
        "alpha_bucket_preferred": alpha_bucket(pref),
        "alpha_bucket_rejected": alpha_bucket(rej),
        "delta_family_preferred": pref.get("primary_delta_family") or pref.get("delta_family"),
        "delta_family_rejected": rej.get("primary_delta_family") or rej.get("delta_family"),
        "delta_type_preferred": pref.get("delta_type"),
        "delta_type_rejected": rej.get("delta_type"),
        "task_screening_class": pref.get("task_screening_class"),
        "split_guard_role": pref.get("split_guard_role"),
        "old_frozen_tap_score_preferred": float(pref.get("tap_margin_sum", pref.get("old_frozen_tap_score", 0.0))),
        "old_frozen_tap_score_rejected": float(rej.get("tap_margin_sum", rej.get("old_frozen_tap_score", 0.0))),
        "v1_tap_score_preferred": pref.get("v1_tap_score"),
        "v1_tap_score_rejected": rej.get("v1_tap_score"),
        "v2_tap_score_preferred": pref.get("v2_tap_score"),
        "v2_tap_score_rejected": rej.get("v2_tap_score"),
        "available_configs": sorted(features.keys()),
        "features": features,
        "split": split,
    }


def compact_pair(pair: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in pair.items() if k != "features"}
    out["feature_dims"] = {cfg: int(pair["features"][cfg]["preferred"].numel()) for cfg in pair.get("features", {})}
    return out


def split_for_task(task_id: str, split_payload: dict[str, Any], extra_val_ids: set[str] | None = None) -> str:
    tid = str(task_id)
    if tid in {str(x) for x in split_payload.get("v3_heldout_candidate_task_ids", [])}:
        return "test"
    if tid in {str(x) for x in split_payload.get("v3_val_candidate_task_ids", [])} or tid in (extra_val_ids or set()):
        return "val"
    return "train"


def features_for_pair(pref: dict[str, Any], rej: dict[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
    features: dict[str, dict[str, torch.Tensor]] = {}
    for config in CONFIGS:
        left = config_vector_from_row(pref, config)
        right = config_vector_from_row(rej, config)
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            if tuple(left.shape) == (config_dim(config),) and tuple(right.shape) == (config_dim(config),):
                features[config] = {"preferred": left.to(torch.float32), "rejected": right.to(torch.float32)}
    return features


def build_pairs(
    rows: list[dict[str, Any]],
    *,
    variant: str,
    label_source: str,
    row_filter: Callable[[dict[str, Any]], bool],
    split_payload: dict[str, Any],
    extra_val_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    valid_rows = [row for row in rows if row_filter(row) and stable_v2_row(row)]
    groups = {gid: vals for gid, vals in group_rows(valid_rows).items() if len(vals) >= 2}
    pairs = []
    tie_rows = []
    candidate_pairs = 0
    tie_pairs = 0
    omitted_missing_features = 0
    for gid, vals in sorted(groups.items()):
        ordered = sorted(vals, key=lambda row: int(row.get("branch_id", -1)))
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a = ordered[i]
                b = ordered[j]
                if label_source == "sampled_expected" and (sampled_reward(a) is None or sampled_reward(b) is None):
                    continue
                candidate_pairs += 1
                ra = row_reward(a, label_source)
                rb = row_reward(b, label_source)
                if ra == rb:
                    tie_pairs += 1
                    tie_rows.append(
                        {
                            "variant": variant,
                            "task_id": a.get("task_id"),
                            "branch_group_id": gid,
                            "branch_i": int(a.get("branch_id", -1)),
                            "branch_j": int(b.get("branch_id", -1)),
                            "reward": ra,
                        }
                    )
                    continue
                pref, rej = (a, b) if ra > rb else (b, a)
                features = features_for_pair(pref, rej)
                if not any(config in features for config in CONFIGS):
                    omitted_missing_features += 1
                    continue
                split = split_for_task(str(pref["task_id"]), split_payload, extra_val_ids)
                pair_id = f"{variant}::{gid}::pair={int(a['branch_id'])}-{int(b['branch_id'])}"
                pairs.append(pair_metadata(pref, rej, pair_id=pair_id, split=split, variant=variant, label_source=label_source, features=features))
    meta = {
        "valid_branch_count": len(valid_rows),
        "valid_group_count": len(groups),
        "candidate_unordered_pairs": candidate_pairs,
        "tie_pairs_omitted": tie_pairs,
        "tie_rate": tie_pairs / max(candidate_pairs, 1),
        "omitted_missing_features": omitted_missing_features,
        "behaviorally_diverse_groups": len([gid for gid, vals in groups.items() if group_is_behaviorally_diverse_v2(vals, label_source)]),
        "reward_diverse_groups": len([gid for gid, vals in groups.items() if group_is_reward_diverse_v2(vals, label_source)]),
    }
    return pairs, meta, tie_rows


def choose_extra_val_ids(rows: list[dict[str, Any]], split_payload: dict[str, Any]) -> set[str]:
    heldout = {str(x) for x in split_payload.get("v3_heldout_candidate_task_ids", [])}
    val = {str(x) for x in split_payload.get("v3_val_candidate_task_ids", [])}
    val_groups = [vals for vals in group_rows([row for row in rows if str(row.get("task_id")) in val and primary_safe_v3_row(row)]).values() if len(vals) >= 2]
    if val_groups:
        return set()
    candidate_tasks = sorted({str(row.get("task_id")) for row in rows if str(row.get("task_id")) not in heldout and str(row.get("task_id")) not in val and primary_safe_v3_row(row)})
    return set(candidate_tasks[: max(1, min(4, len(candidate_tasks) // 5))])


def split_counts(pairs: list[dict[str, Any]], rows: list[dict[str, Any]], label_source: str = "deterministic") -> dict[str, Any]:
    pairs_by_split = Counter(pair["split"] for pair in pairs)
    tasks_by_split = {name: sorted({str(pair["task_id"]) for pair in pairs if pair["split"] == name}) for name in ("train", "val", "test", "unused")}
    groups_by_split = {name: sorted({str(pair["branch_group_id"]) for pair in pairs if pair["split"] == name}) for name in ("train", "val", "test", "unused")}
    valid_groups = {gid: vals for gid, vals in group_rows(rows).items() if len(vals) >= 2}
    behavior_by_split = {}
    reward_by_split = {}
    for name, task_ids in tasks_by_split.items():
        behavior_by_split[name] = len(
            [
                gid
                for gid, vals in valid_groups.items()
                if str(vals[0].get("task_id")) in task_ids and group_is_behaviorally_diverse_v2(vals, label_source)
            ]
        )
        reward_by_split[name] = len(
            [
                gid
                for gid, vals in valid_groups.items()
                if str(vals[0].get("task_id")) in task_ids and group_is_reward_diverse_v2(vals, label_source)
            ]
        )
    return {
        "pairs_by_split": dict(pairs_by_split),
        "tasks_by_split": tasks_by_split,
        "groups_by_split": {k: len(v) for k, v in groups_by_split.items()},
        "behaviorally_diverse_groups_by_split": behavior_by_split,
        "reward_diverse_groups_by_split": reward_by_split,
    }


def verdict_for(train_pairs: int, val_pairs: int, test_pairs: int, test_tasks: int, test_behavior_groups: int) -> str:
    if train_pairs <= 0 and test_pairs <= 0:
        return "BLOCKED"
    if test_tasks >= 8 and test_behavior_groups >= 20 and test_pairs >= 120 and train_pairs >= 150 and val_pairs >= 20:
        return "READY"
    if test_tasks >= 6 and test_behavior_groups >= 15 and test_pairs >= 80 and train_pairs >= 60 and val_pairs >= 10:
        return "SMALL_BUT_USABLE"
    return "STILL_DATA_LIMITED"


def main() -> int:
    started = time.time()
    ensure_v3_root()
    rows = load_all_v3_branch_rows(include_prior=True)
    split_payload = load_json(V3_ROOT / "split_guard_v3.json", {}) or {}
    if not rows:
        payload = {"BG_HIDDEN_ORIGIN_TAP_DATASET_V3_VERDICT": "BLOCKED", "verdict": "BLOCKED", "blocker": "no branch rows"}
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# Hidden-Origin Tap Dataset V3", "", "BG_HIDDEN_ORIGIN_TAP_DATASET_V3_VERDICT = BLOCKED"])
        print("BG_HIDDEN_ORIGIN_TAP_DATASET_V3_VERDICT = BLOCKED", flush=True)
        return 1
    if split_payload.get("verdict") == "BLOCKED":
        payload = {"BG_HIDDEN_ORIGIN_TAP_DATASET_V3_VERDICT": "BLOCKED", "verdict": "BLOCKED", "blocker": "split guard blocked"}
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# Hidden-Origin Tap Dataset V3", "", "BG_HIDDEN_ORIGIN_TAP_DATASET_V3_VERDICT = BLOCKED"])
        print("BG_HIDDEN_ORIGIN_TAP_DATASET_V3_VERDICT = BLOCKED", flush=True)
        return 1

    extra_val_ids = choose_extra_val_ids(rows, split_payload)
    drivers = load_json(V3_ROOT / "diversity_drivers_v3.json", {}) or {}
    recipe = drivers.get("recommended_branch_generation_recipe") or {}

    def high_yield_filter(row: dict[str, Any]) -> bool:
        if not primary_safe_v3_row(row):
            return False
        if not recipe:
            return False
        factor = str(recipe.get("factor") or "")
        condition = str(recipe.get("condition") or "")
        if factor == "branch_point":
            return str(row.get("branch_point")) == condition or str(row.get("branch_point")).startswith(condition)
        if factor == "delta_family":
            return str(row.get("primary_delta_family") or row.get("delta_family")) == condition
        return str(row.get(factor)) == condition

    variants_spec = {
        "primary_safe_deterministic": {
            "label_source": "deterministic",
            "filter": primary_safe_v3_row,
        },
        "alpha_0_02_diagnostic": {
            "label_source": "deterministic",
            "filter": diagnostic_alpha_v3_row,
        },
        "sampled_expected_diagnostic": {
            "label_source": "sampled_expected",
            "filter": lambda row: (primary_safe_v3_row(row) or diagnostic_alpha_v3_row(row)) and sampled_reward(row) is not None,
        },
        "high_yield_recipe_subset": {
            "label_source": "deterministic",
            "filter": high_yield_filter,
        },
    }
    variant_pairs: dict[str, list[dict[str, Any]]] = {}
    variant_meta: dict[str, dict[str, Any]] = {}
    tie_rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    for variant, spec in variants_spec.items():
        pairs, meta, tie_rows = build_pairs(
            rows,
            variant=variant,
            label_source=spec["label_source"],
            row_filter=spec["filter"],
            split_payload=split_payload,
            extra_val_ids=extra_val_ids,
        )
        variant_pairs[variant] = pairs
        variant_meta[variant] = meta
        tie_rows_by_variant[variant] = tie_rows

    primary_pairs = variant_pairs["primary_safe_deterministic"]
    primary_rows = [row for row in rows if primary_safe_v3_row(row)]
    counts = split_counts(primary_pairs, primary_rows)
    train_pairs = int(counts["pairs_by_split"].get("train", 0))
    val_pairs = int(counts["pairs_by_split"].get("val", 0))
    test_pairs = int(counts["pairs_by_split"].get("test", 0))
    heldout_tasks = len(counts["tasks_by_split"]["test"])
    heldout_behavior_groups = int(counts["behaviorally_diverse_groups_by_split"].get("test", 0))
    verdict = verdict_for(train_pairs, val_pairs, test_pairs, heldout_tasks, heldout_behavior_groups)
    if not primary_pairs:
        verdict = "BLOCKED"

    feature_coverage_pairs = {config: sum(1 for pair in primary_pairs if config in pair.get("features", {})) for config in CONFIGS}
    reward_gaps = [float(pair["reward_gap"]) for pair in primary_pairs]
    payload = {
        "BG_HIDDEN_ORIGIN_TAP_DATASET_V3_VERDICT": verdict,
        "verdict": verdict,
        "row_count": len(rows),
        "primary_valid_branch_count": len(primary_rows),
        "pairs": primary_pairs,
        "pairs_by_variant": variant_pairs,
        "variant_meta": variant_meta,
        "tie_rows_by_variant": tie_rows_by_variant,
        "split_guard_status": split_payload.get("verdict"),
        "split_guard_counts": split_payload.get("counts"),
        "baseline_leakage_warning_task_ids": split_payload.get("v1_v2_baseline_may_have_seen_v3_heldout_task_ids", []),
        "extra_val_task_ids": sorted(extra_val_ids),
        **counts,
        "reward_gap_distribution": {
            "min": min(reward_gaps) if reward_gaps else None,
            "max": max(reward_gaps) if reward_gaps else None,
            "mean": sum(reward_gaps) / max(len(reward_gaps), 1),
        },
        "domain_distribution": dict(Counter(str(pair.get("domain")) for pair in primary_pairs)),
        "branch_point_distribution": dict(Counter(str(pair.get("branch_point_preferred")) for pair in primary_pairs)),
        "alpha_distribution": dict(Counter(str(pair.get("alpha_bucket_preferred")) for pair in primary_pairs)),
        "delta_family_distribution": dict(Counter(str(pair.get("delta_family_preferred")) for pair in primary_pairs)),
        "task_class_distribution": dict(Counter(str(pair.get("task_screening_class")) for pair in primary_pairs)),
        "label_source_distribution": dict(Counter(str(pair.get("label_source")) for pair in primary_pairs)),
        "config_feature_coverage_pairs": feature_coverage_pairs,
        "config_feature_coverage_branches": available_configs_for_rows(primary_rows),
        "no_task_overlap": not (
            set(counts["tasks_by_split"]["train"]) & set(counts["tasks_by_split"]["val"])
            or set(counts["tasks_by_split"]["train"]) & set(counts["tasks_by_split"]["test"])
            or set(counts["tasks_by_split"]["val"]) & set(counts["tasks_by_split"]["test"])
        ),
        "heldout_requirements": {
            "minimum": {"task_ids": 6, "behaviorally_diverse_groups": 15, "non_tie_pairs": 80},
            "preferred": {"task_ids": 8, "behaviorally_diverse_groups": 20, "non_tie_pairs": 120},
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(payload, OUT_PT)
    json_payload = {k: v for k, v in payload.items() if k not in {"pairs", "pairs_by_variant", "tie_rows_by_variant"}}
    json_payload["pairs"] = [compact_pair(pair) for pair in primary_pairs]
    json_payload["pairs_by_variant"] = {variant: [compact_pair(pair) for pair in pairs] for variant, pairs in variant_pairs.items()}
    json_payload["tie_rows_by_variant"] = tie_rows_by_variant
    write_json(OUT_JSON, json_payload)

    split_rows = [
        {
            "split": name,
            "tasks": len(counts["tasks_by_split"][name]),
            "groups": counts["groups_by_split"].get(name, 0),
            "behavior_groups": counts["behaviorally_diverse_groups_by_split"].get(name, 0),
            "reward_groups": counts["reward_diverse_groups_by_split"].get(name, 0),
            "pairs": counts["pairs_by_split"].get(name, 0),
        }
        for name in ("train", "val", "test", "unused")
    ]
    variant_rows = [
        {
            "variant": variant,
            "branches": meta["valid_branch_count"],
            "groups": meta["valid_group_count"],
            "candidate_pairs": meta["candidate_unordered_pairs"],
            "tie_rate": f"{meta['tie_rate']:.3f}",
            "pairs": len(variant_pairs[variant]),
            "behavior_groups": meta["behaviorally_diverse_groups"],
        }
        for variant, meta in variant_meta.items()
    ]
    lines = [
        "# Hidden-Origin Tap Dataset V3",
        "",
        f"BG_HIDDEN_ORIGIN_TAP_DATASET_V3_VERDICT = {verdict}",
        "",
        f"- primary_valid_branch_count: `{len(primary_rows)}`",
        f"- primary_pairs: `{len(primary_pairs)}`",
        f"- deterministic_tie_rate: `{variant_meta['primary_safe_deterministic']['tie_rate']:.3f}`",
        f"- heldout_task_ids: `{heldout_tasks}`",
        f"- heldout_pairs: `{test_pairs}`",
        f"- heldout_behaviorally_diverse_groups: `{heldout_behavior_groups}`",
        f"- no_task_overlap: `{payload['no_task_overlap']}`",
        f"- split_guard_status: `{split_payload.get('verdict')}`",
        f"- baseline_leakage_warning_task_ids: `{payload['baseline_leakage_warning_task_ids']}`",
        "",
        "Primary labels use stable deterministic reward labels, alpha `<= 0.01`, reasoning/science domains, same-group pairs, and non-L47 same-prefix hidden-origin branch groups. Ties are omitted.",
        "",
        "## Splits",
        "",
    ]
    lines.extend(md_table(split_rows, ["split", "tasks", "groups", "behavior_groups", "reward_groups", "pairs"]))
    lines.extend(["", "## Variants", ""])
    lines.extend(md_table(variant_rows, ["variant", "branches", "groups", "candidate_pairs", "tie_rate", "pairs", "behavior_groups"]))
    lines.extend(["", "## Feature Coverage", ""])
    lines.extend(md_table([{"config": cfg, "pairs": count} for cfg, count in feature_coverage_pairs.items()], ["config", "pairs"]))
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_TAP_DATASET_V3_VERDICT = {verdict}", flush=True)
    print(f"Wrote {rel(OUT_PT)}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

