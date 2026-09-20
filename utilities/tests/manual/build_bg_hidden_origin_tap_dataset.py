"""Build pairwise hidden-origin branch tap datasets from downstream outcomes."""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from typing import Any

import torch

from bg_hidden_origin_tap_common import (
    CONFIGS,
    OUT_ROOT,
    available_configs_for_rows,
    branch_key,
    config_dim,
    config_vector_from_row,
    ensure_out_root,
    group_is_behaviorally_diverse,
    group_is_reward_diverse,
    group_rows,
    is_safe_alpha,
    load_all_branch_rows,
    md_table,
    rel,
    split_name_for_task,
    split_pairs,
    stable_row,
    write_json,
    write_md,
)


OUT_PT = OUT_ROOT / "hidden_origin_tap_dataset.pt"
OUT_JSON = OUT_ROOT / "hidden_origin_tap_dataset.json"
OUT_MD = OUT_ROOT / "hidden_origin_tap_dataset.md"


def pair_metadata(pref: dict[str, Any], rej: dict[str, Any], pair_id: str, split: str, features: dict[str, dict[str, torch.Tensor]]) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "task_id": pref["task_id"],
        "branch_group_id": pref["branch_group_id"],
        "domain": pref.get("domain"),
        "preferred_branch_id": int(pref["branch_id"]),
        "rejected_branch_id": int(rej["branch_id"]),
        "reward_preferred": float(pref.get("reward", 0.0)),
        "reward_rejected": float(rej.get("reward", 0.0)),
        "reward_gap": float(pref.get("reward", 0.0)) - float(rej.get("reward", 0.0)),
        "correctness_preferred": bool(pref.get("correct")),
        "correctness_rejected": bool(rej.get("correct")),
        "branch_point_preferred": pref.get("branch_point"),
        "branch_point_rejected": rej.get("branch_point"),
        "alpha_preferred": float(pref.get("alpha", 0.0)),
        "alpha_rejected": float(rej.get("alpha", 0.0)),
        "delta_type_preferred": pref.get("delta_type"),
        "delta_type_rejected": rej.get("delta_type"),
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


def build_pairs(valid_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups = group_rows(valid_rows)
    raw_pairs: list[dict[str, Any]] = []
    tie_pairs = 0
    candidate_pairs = 0
    omitted_missing_features = 0
    for gid, vals in sorted(groups.items()):
        vals = sorted(vals, key=lambda row: int(row.get("branch_id", -1)))
        if len(vals) < 2:
            continue
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                candidate_pairs += 1
                a = vals[i]
                b = vals[j]
                ra = float(a.get("reward", 0.0))
                rb = float(b.get("reward", 0.0))
                if ra == rb:
                    tie_pairs += 1
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
                pair_id = f"{gid}::pair={int(a['branch_id'])}-{int(b['branch_id'])}"
                raw_pairs.append(pair_metadata(pref, rej, pair_id, "unassigned", features))
    meta = {
        "candidate_unordered_pairs": candidate_pairs,
        "tie_pairs_omitted": tie_pairs,
        "tie_rate": tie_pairs / max(candidate_pairs, 1),
        "omitted_missing_features": omitted_missing_features,
    }
    return raw_pairs, meta


def main() -> int:
    ensure_out_root()
    started = time.time()
    all_rows = load_all_branch_rows()
    valid_rows = [row for row in all_rows if is_safe_alpha(row) and stable_row(row)]
    raw_pairs, pair_meta = build_pairs(valid_rows)
    if not raw_pairs:
        payload = {
            "BG_HIDDEN_ORIGIN_TAP_DATASET_VERDICT": "BLOCKED",
            "verdict": "BLOCKED",
            "blocker": "no reward-ordered pairs after safe/stable filtering",
            "row_count": len(all_rows),
            "valid_branch_count": len(valid_rows),
            "pair_meta": pair_meta,
        }
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# Hidden-Origin Tap Dataset", "", "BG_HIDDEN_ORIGIN_TAP_DATASET_VERDICT = BLOCKED"])
        print("BG_HIDDEN_ORIGIN_TAP_DATASET_VERDICT = BLOCKED", flush=True)
        return 1

    split, split_meta = split_pairs(raw_pairs, seed=42)
    pairs: list[dict[str, Any]] = []
    for pair in raw_pairs:
        name = split_name_for_task(str(pair["task_id"]), split)
        item = dict(pair)
        item["split"] = name
        pairs.append(item)

    pairs_by_split = Counter(pair["split"] for pair in pairs)
    tasks_by_split = {
        name: sorted({str(pair["task_id"]) for pair in pairs if pair["split"] == name})
        for name in ("train", "val", "test", "unused")
    }
    groups_by_split = {
        name: sorted({str(pair["branch_group_id"]) for pair in pairs if pair["split"] == name})
        for name in ("train", "val", "test", "unused")
    }
    branches_by_split = {}
    for name in ("train", "val", "test", "unused"):
        keys = set()
        for pair in pairs:
            if pair["split"] != name:
                continue
            keys.add((pair["branch_group_id"], pair["preferred_branch_id"]))
            keys.add((pair["branch_group_id"], pair["rejected_branch_id"]))
        branches_by_split[name] = len(keys)

    feature_coverage_pairs = {
        config: sum(1 for pair in pairs if config in pair.get("features", {}))
        for config in CONFIGS
    }
    reward_gaps = [float(pair["reward_gap"]) for pair in pairs]
    valid_groups = group_rows(valid_rows)
    behaviorally_diverse_groups = [gid for gid, vals in valid_groups.items() if len(vals) >= 2 and group_is_behaviorally_diverse(vals)]
    reward_diverse_groups = [gid for gid, vals in valid_groups.items() if len(vals) >= 2 and group_is_reward_diverse(vals)]

    train_pairs = int(pairs_by_split.get("train", 0))
    test_pairs = int(pairs_by_split.get("test", 0))
    if train_pairs >= 100 and test_pairs >= 30 and len(tasks_by_split["train"]) >= 10 and len(tasks_by_split["test"]) >= 4:
        verdict = "READY"
    elif train_pairs >= 30 and test_pairs >= 10:
        verdict = "SMALL_BUT_USABLE"
    elif pairs:
        verdict = "TOO_SMALL"
    else:
        verdict = "BLOCKED"

    payload = {
        "BG_HIDDEN_ORIGIN_TAP_DATASET_VERDICT": verdict,
        "verdict": verdict,
        "row_count": len(all_rows),
        "valid_branch_count": len(valid_rows),
        "valid_branch_groups": len(valid_groups),
        "behaviorally_diverse_group_count": len(behaviorally_diverse_groups),
        "reward_diverse_group_count": len(reward_diverse_groups),
        "pairs": pairs,
        "pair_meta": pair_meta,
        "split_meta": split_meta,
        "tasks_by_split": tasks_by_split,
        "groups_by_split": {k: len(v) for k, v in groups_by_split.items()},
        "branches_by_split": branches_by_split,
        "pairs_by_split": dict(pairs_by_split),
        "reward_gap_distribution": {
            "min": min(reward_gaps) if reward_gaps else None,
            "max": max(reward_gaps) if reward_gaps else None,
            "mean": sum(reward_gaps) / max(len(reward_gaps), 1),
        },
        "domain_distribution": dict(Counter(pair["domain"] for pair in pairs)),
        "config_feature_coverage_pairs": feature_coverage_pairs,
        "config_feature_coverage_branches": available_configs_for_rows(valid_rows),
        "no_task_overlap": not (set(tasks_by_split["train"]) & set(tasks_by_split["val"]) or set(tasks_by_split["train"]) & set(tasks_by_split["test"]) or set(tasks_by_split["val"]) & set(tasks_by_split["test"])),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    torch.save(payload, OUT_PT)

    json_payload = {k: v for k, v in payload.items() if k != "pairs"}
    json_payload["pairs"] = [compact_pair(pair) for pair in pairs]
    write_json(OUT_JSON, json_payload)

    split_rows = [
        {
            "split": name,
            "tasks": len(tasks_by_split[name]),
            "groups": len(groups_by_split[name]),
            "branches": branches_by_split[name],
            "pairs": int(pairs_by_split.get(name, 0)),
        }
        for name in ("train", "val", "test", "unused")
    ]
    lines = [
        "# Hidden-Origin Tap Dataset",
        "",
        f"BG_HIDDEN_ORIGIN_TAP_DATASET_VERDICT = {verdict}",
        "",
        f"- valid_branch_count: `{len(valid_rows)}`",
        f"- valid_branch_groups: `{len(valid_groups)}`",
        f"- behaviorally_diverse_group_count: `{len(behaviorally_diverse_groups)}`",
        f"- reward_diverse_group_count: `{len(reward_diverse_groups)}`",
        f"- candidate_pairs: `{pair_meta['candidate_unordered_pairs']}`",
        f"- tie_pairs_omitted: `{pair_meta['tie_pairs_omitted']}`",
        f"- tie_rate: `{pair_meta['tie_rate']:.3f}`",
        f"- no_task_overlap: `{payload['no_task_overlap']}`",
        "",
        "## Splits",
        "",
    ]
    lines.extend(md_table(split_rows, ["split", "tasks", "groups", "branches", "pairs"]))
    lines.extend(["", "## Feature Coverage", ""])
    lines.extend(md_table(
        [{"config": config, "pairs": feature_coverage_pairs.get(config, 0)} for config in CONFIGS],
        ["config", "pairs"],
    ))
    lines.extend(["", "## Pair Examples", ""])
    lines.extend(md_table(
        [
            {
                "pair_id": pair["pair_id"],
                "split": pair["split"],
                "domain": pair["domain"],
                "reward_gap": pair["reward_gap"],
                "configs": len(pair["features"]),
            }
            for pair in pairs[:30]
        ],
        ["pair_id", "split", "domain", "reward_gap", "configs"],
    ))
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_TAP_DATASET_VERDICT = {verdict}", flush=True)
    print(f"Wrote {rel(OUT_PT)}", flush=True)
    print(f"Wrote {rel(OUT_JSON)}", flush=True)
    print(f"Wrote {rel(OUT_MD)}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
