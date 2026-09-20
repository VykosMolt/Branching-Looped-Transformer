"""Build task-disjoint pairwise hidden-origin tap datasets v2."""
from __future__ import annotations

import time
from collections import Counter
from typing import Any, Callable

import torch

from bg_hidden_origin_diversity_v2_common import (
    CONFIGS,
    DATASET_V2_PT,
    SEED,
    V2_ROOT,
    alpha_bucket,
    available_configs_for_rows,
    config_dim,
    config_vector_from_row,
    deterministic_correct,
    deterministic_reward,
    diagnostic_alpha_row,
    ensure_v2_root,
    group_is_behaviorally_diverse_v2,
    group_is_reward_diverse_v2,
    group_rows,
    load_all_v2_branch_rows,
    md_table,
    rel,
    row_reward,
    safe_primary_row,
    sampled_reward,
    split_task_ids_for_pairs,
    stable_v2_row,
    write_json,
    write_md,
)


OUT_PT = DATASET_V2_PT
OUT_JSON = V2_ROOT / "hidden_origin_tap_dataset_v2.json"
OUT_MD = V2_ROOT / "hidden_origin_tap_dataset_v2.md"


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
        "delta_family_preferred": pref.get("delta_family"),
        "delta_family_rejected": rej.get("delta_family"),
        "delta_type_preferred": pref.get("delta_type"),
        "delta_type_rejected": rej.get("delta_type"),
        "task_screening_class": pref.get("task_screening_class"),
        "old_frozen_tap_score_preferred": float(pref.get("tap_margin_sum", 0.0)),
        "old_frozen_tap_score_rejected": float(rej.get("tap_margin_sum", 0.0)),
        "available_configs": sorted(features.keys()),
        "features": features,
        "split": split,
    }


def compact_pair(pair: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in pair.items() if k != "features"}
    out["feature_dims"] = {cfg: int(pair["features"][cfg]["preferred"].numel()) for cfg in pair.get("features", {})}
    return out


def build_pairs(
    rows: list[dict[str, Any]],
    *,
    variant: str,
    label_source: str,
    row_filter: Callable[[dict[str, Any]], bool],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    valid_rows = [row for row in rows if row_filter(row) and stable_v2_row(row)]
    groups = {gid: vals for gid, vals in group_rows(valid_rows).items() if len(vals) >= 2}
    pairs: list[dict[str, Any]] = []
    tie_pairs = 0
    candidate_pairs = 0
    omitted_missing_features = 0
    tie_rows: list[dict[str, Any]] = []
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
                            "label_source": label_source,
                            "task_id": a.get("task_id"),
                            "branch_group_id": gid,
                            "branch_i": int(a.get("branch_id", -1)),
                            "branch_j": int(b.get("branch_id", -1)),
                            "reward": ra,
                        }
                    )
                    continue
                pref, rej = (a, b) if ra > rb else (b, a)
                features: dict[str, dict[str, torch.Tensor]] = {}
                for config in CONFIGS:
                    left = config_vector_from_row(pref, config)
                    right = config_vector_from_row(rej, config)
                    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
                        if tuple(left.shape) == (config_dim(config),) and tuple(right.shape) == (config_dim(config),):
                            features[config] = {"preferred": left.to(torch.float32), "rejected": right.to(torch.float32)}
                if not any(config in features for config in CONFIGS[:9]):
                    omitted_missing_features += 1
                    continue
                pair_id = f"{variant}::{gid}::pair={int(a['branch_id'])}-{int(b['branch_id'])}"
                pairs.append(
                    pair_metadata(
                        pref,
                        rej,
                        pair_id=pair_id,
                        split="unassigned",
                        variant=variant,
                        label_source=label_source,
                        features=features,
                    )
                )
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


def assign_splits(pairs: list[dict[str, Any]], split: dict[str, set[str]]) -> list[dict[str, Any]]:
    out = []
    for pair in pairs:
        task_id = str(pair["task_id"])
        item = dict(pair)
        item["split"] = "unused"
        for name, task_ids in split.items():
            if task_id in task_ids:
                item["split"] = name
                break
        out.append(item)
    return out


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


def main() -> int:
    ensure_v2_root()
    started = time.time()
    rows = load_all_v2_branch_rows(include_prior=True)
    if not rows:
        payload = {"BG_HIDDEN_ORIGIN_TAP_DATASET_V2_VERDICT": "BLOCKED", "verdict": "BLOCKED", "blocker": "no branch rows"}
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# Hidden-Origin Tap Dataset V2", "", "BG_HIDDEN_ORIGIN_TAP_DATASET_V2_VERDICT = BLOCKED"])
        print("BG_HIDDEN_ORIGIN_TAP_DATASET_V2_VERDICT = BLOCKED", flush=True)
        return 1

    variants_spec = {
        "primary_safe_deterministic": {
            "label_source": "deterministic",
            "filter": lambda row: safe_primary_row(row),
        },
        "safe_plus_alpha_0_02_diagnostic": {
            "label_source": "deterministic",
            "filter": lambda row: safe_primary_row(row) or diagnostic_alpha_row(row),
        },
        "sampled_expected_diagnostic": {
            "label_source": "sampled_expected",
            "filter": lambda row: (safe_primary_row(row) or diagnostic_alpha_row(row)) and sampled_reward(row) is not None,
        },
    }
    variant_pairs: dict[str, list[dict[str, Any]]] = {}
    variant_meta: dict[str, dict[str, Any]] = {}
    tie_rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    primary_rows = [row for row in rows if safe_primary_row(row) and stable_v2_row(row)]
    for variant, spec in variants_spec.items():
        pairs, meta, tie_rows = build_pairs(rows, variant=variant, label_source=spec["label_source"], row_filter=spec["filter"])
        variant_pairs[variant] = pairs
        variant_meta[variant] = meta
        tie_rows_by_variant[variant] = tie_rows

    primary_pairs = variant_pairs["primary_safe_deterministic"]
    if not primary_pairs:
        payload = {
            "BG_HIDDEN_ORIGIN_TAP_DATASET_V2_VERDICT": "BLOCKED",
            "verdict": "BLOCKED",
            "blocker": "no non-tie primary safe deterministic pairs",
            "variant_meta": variant_meta,
        }
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# Hidden-Origin Tap Dataset V2", "", "BG_HIDDEN_ORIGIN_TAP_DATASET_V2_VERDICT = BLOCKED"])
        print("BG_HIDDEN_ORIGIN_TAP_DATASET_V2_VERDICT = BLOCKED", flush=True)
        return 1

    split, split_meta = split_task_ids_for_pairs(primary_pairs, seed=SEED + 2)
    for variant, pairs in list(variant_pairs.items()):
        variant_pairs[variant] = assign_splits(pairs, split)
    primary_pairs = variant_pairs["primary_safe_deterministic"]
    counts = split_counts(primary_pairs, primary_rows)
    train_pairs = int(counts["pairs_by_split"].get("train", 0))
    test_pairs = int(counts["pairs_by_split"].get("test", 0))
    heldout_tasks = len(counts["tasks_by_split"]["test"])
    heldout_behavior_groups = int(counts["behaviorally_diverse_groups_by_split"].get("test", 0))
    if train_pairs >= 150 and test_pairs >= 60 and heldout_tasks >= 8 and heldout_behavior_groups >= 20:
        verdict = "READY"
    elif train_pairs >= 60 and test_pairs >= 30 and heldout_tasks >= 4 and heldout_behavior_groups >= 10:
        verdict = "SMALL_BUT_USABLE"
    else:
        verdict = "DATA_LIMITED"

    feature_coverage_pairs = {config: sum(1 for pair in primary_pairs if config in pair.get("features", {})) for config in CONFIGS}
    reward_gaps = [float(pair["reward_gap"]) for pair in primary_pairs]
    payload = {
        "BG_HIDDEN_ORIGIN_TAP_DATASET_V2_VERDICT": verdict,
        "verdict": verdict,
        "row_count": len(rows),
        "primary_valid_branch_count": len(primary_rows),
        "pairs": primary_pairs,
        "pairs_by_variant": variant_pairs,
        "variant_meta": variant_meta,
        "tie_rows_by_variant": tie_rows_by_variant,
        "split_meta": split_meta,
        **counts,
        "reward_gap_distribution": {
            "min": min(reward_gaps) if reward_gaps else None,
            "max": max(reward_gaps) if reward_gaps else None,
            "mean": sum(reward_gaps) / max(len(reward_gaps), 1),
        },
        "domain_distribution": dict(Counter(str(pair.get("domain")) for pair in primary_pairs)),
        "branch_point_distribution": dict(Counter(str(pair.get("branch_point_preferred")) for pair in primary_pairs)),
        "alpha_bucket_distribution": dict(Counter(str(pair.get("alpha_bucket_preferred")) for pair in primary_pairs)),
        "delta_family_distribution": dict(Counter(str(pair.get("delta_family_preferred")) for pair in primary_pairs)),
        "config_feature_coverage_pairs": feature_coverage_pairs,
        "config_feature_coverage_branches": available_configs_for_rows(primary_rows),
        "no_task_overlap": not (
            set(counts["tasks_by_split"]["train"]) & set(counts["tasks_by_split"]["val"])
            or set(counts["tasks_by_split"]["train"]) & set(counts["tasks_by_split"]["test"])
            or set(counts["tasks_by_split"]["val"]) & set(counts["tasks_by_split"]["test"])
        ),
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
        "# Hidden-Origin Tap Dataset V2",
        "",
        f"BG_HIDDEN_ORIGIN_TAP_DATASET_V2_VERDICT = {verdict}",
        "",
        f"- primary_valid_branch_count: `{len(primary_rows)}`",
        f"- primary_pairs: `{len(primary_pairs)}`",
        f"- deterministic_tie_rate: `{variant_meta['primary_safe_deterministic']['tie_rate']:.3f}`",
        f"- heldout_task_ids: `{heldout_tasks}`",
        f"- heldout_pairs: `{test_pairs}`",
        f"- heldout_behaviorally_diverse_groups: `{heldout_behavior_groups}`",
        f"- no_task_overlap: `{payload['no_task_overlap']}`",
        "",
        "Alpha `0.02` rows are isolated in the diagnostic variant and are not mixed into the primary safe deterministic headline.",
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
    print(f"BG_HIDDEN_ORIGIN_TAP_DATASET_V2_VERDICT = {verdict}", flush=True)
    print(f"Wrote {rel(OUT_PT)}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
