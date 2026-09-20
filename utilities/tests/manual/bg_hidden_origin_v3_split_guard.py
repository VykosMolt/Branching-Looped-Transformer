"""Define v3 task-level split/leakage guard before empirical directions."""
from __future__ import annotations

import random
import time
from collections import Counter, defaultdict
from typing import Any

from bg_hidden_origin_diversity_v3_common import (
    SEED,
    SPLIT_GUARD_JSON,
    V1_ROOT,
    V2_ROOT,
    V3_ROOT,
    ensure_v3_root,
    load_json,
    md_table,
    rel,
    selected_v3_tasks,
    write_json,
    write_md,
)


OUT_JSON = SPLIT_GUARD_JSON
OUT_MD = V3_ROOT / "split_guard_v3.md"


def prior_split_task_ids() -> dict[str, Any]:
    paths = {
        "v1_dataset": V1_ROOT / "hidden_origin_tap_dataset.json",
        "v2_dataset": V2_ROOT / "hidden_origin_tap_dataset_v2.json",
    }
    out: dict[str, Any] = {"sources": {}, "all_prior_split_task_ids": []}
    all_ids = set()
    for label, path in paths.items():
        payload = load_json(path, {}) or {}
        split = payload.get("tasks_by_split") or {}
        source = {name: sorted({str(x) for x in ids}) for name, ids in split.items()}
        out["sources"][label] = source
        for ids in source.values():
            all_ids.update(ids)
    out["all_prior_split_task_ids"] = sorted(all_ids)
    return out


def stratified_pick(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(str(row.get("domain")), str(row.get("screening_class")))].append(row)
    for vals in buckets.values():
        vals.sort(key=lambda row: (-float(row.get("priority_score", 0.0)), str(row.get("task_id"))))
        rng.shuffle(vals)
        vals.sort(key=lambda row: (-float(row.get("priority_score", 0.0)), str(row.get("task_id"))))
    selected = []
    used = set()
    while len(selected) < count and any(buckets.values()):
        progressed = False
        for key in sorted(buckets, key=lambda k: (-len(buckets[k]), k)):
            vals = buckets[key]
            while vals:
                row = vals.pop(0)
                tid = str(row["task_id"])
                if tid in used:
                    continue
                selected.append(row)
                used.add(tid)
                progressed = True
                break
            if len(selected) >= count:
                break
        if not progressed:
            break
    return selected


def main() -> int:
    started = time.time()
    ensure_v3_root()
    tasks = selected_v3_tasks()
    if not tasks:
        payload = {
            "BG_HIDDEN_ORIGIN_SPLIT_GUARD_V3_VERDICT": "BLOCKED",
            "verdict": "BLOCKED",
            "blocker": "missing selected v3 tasks",
        }
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# Hidden-Origin V3 Split Guard", "", "BG_HIDDEN_ORIGIN_SPLIT_GUARD_V3_VERDICT = BLOCKED"])
        print("BG_HIDDEN_ORIGIN_SPLIT_GUARD_V3_VERDICT = BLOCKED", flush=True)
        return 1

    by_id = {str(task["task_id"]): task for task in tasks}
    candidate_ids = sorted(by_id)
    prior = prior_split_task_ids()
    prior_seen = set(prior["all_prior_split_task_ids"])
    unseen_rows = [by_id[tid] for tid in candidate_ids if tid not in prior_seen]
    seen_rows = [by_id[tid] for tid in candidate_ids if tid in prior_seen]
    n = len(candidate_ids)
    if n < 6:
        heldout_n = max(1, n // 4)
    else:
        heldout_n = min(max(8, round(0.20 * n)), max(1, n - 3))
    val_n = min(max(4, round(0.15 * n)), max(1, n - heldout_n - 1))
    clean_heldout = stratified_pick(unseen_rows, min(heldout_n, len(unseen_rows)), SEED + 301)
    remaining_needed = heldout_n - len(clean_heldout)
    if remaining_needed > 0:
        clean_ids = {str(row["task_id"]) for row in clean_heldout}
        clean_heldout.extend(stratified_pick([row for row in seen_rows if str(row["task_id"]) not in clean_ids], remaining_needed, SEED + 302))
    heldout_ids = {str(row["task_id"]) for row in clean_heldout}
    remaining = [by_id[tid] for tid in candidate_ids if tid not in heldout_ids]
    val_rows = stratified_pick(remaining, min(val_n, max(len(remaining) - 1, 0)), SEED + 303)
    val_ids = {str(row["task_id"]) for row in val_rows}
    train_ids = {tid for tid in candidate_ids if tid not in heldout_ids and tid not in val_ids}
    clean_cross = sorted(heldout_ids - prior_seen)
    baseline_overlap = sorted(heldout_ids & prior_seen)
    no_overlap = not (train_ids & val_ids or train_ids & heldout_ids or val_ids & heldout_ids)
    if len(heldout_ids) < 1 or not no_overlap:
        verdict = "BLOCKED"
    elif len(heldout_ids) < 6 or len(train_ids) < 2:
        verdict = "PARTIAL"
    elif baseline_overlap:
        verdict = "READY_WITH_BASELINE_LEAKAGE_WARNING"
    else:
        verdict = "READY"
    payload = {
        "BG_HIDDEN_ORIGIN_SPLIT_GUARD_V3_VERDICT": verdict,
        "verdict": verdict,
        "candidate_task_ids": candidate_ids,
        "v3_train_candidate_task_ids": sorted(train_ids),
        "v3_val_candidate_task_ids": sorted(val_ids),
        "v3_heldout_candidate_task_ids": sorted(heldout_ids),
        "prior_seen_by_v1_v2_task_ids": sorted(prior_seen & set(candidate_ids)),
        "prior_seen_all_task_ids": sorted(prior_seen),
        "prior_split_sources": prior["sources"],
        "clean_cross_version_heldout_task_ids": clean_cross,
        "v1_v2_baseline_may_have_seen_v3_heldout_task_ids": baseline_overlap,
        "v3_empirical_direction_train_eligible_task_ids": sorted(train_ids | val_ids),
        "v3_tap_training_allowed_task_ids": sorted(train_ids | val_ids),
        "guards": {
            "v3_heldout_excluded_from_v3_empirical_direction_construction": True,
            "v3_heldout_excluded_from_v3_tap_training": True,
            "task_disjoint_v3_train_val_heldout": no_overlap,
            "empirical_hidden_origin_directions_from_heldout_forbidden": True,
        },
        "counts": {
            "candidate_tasks": n,
            "train_candidates": len(train_ids),
            "val_candidates": len(val_ids),
            "heldout_candidates": len(heldout_ids),
            "clean_cross_version_heldout": len(clean_cross),
            "baseline_overlap_heldout": len(baseline_overlap),
        },
        "domain_counts_by_split": {
            "train": dict(Counter(str(by_id[tid].get("domain")) for tid in train_ids)),
            "val": dict(Counter(str(by_id[tid].get("domain")) for tid in val_ids)),
            "heldout": dict(Counter(str(by_id[tid].get("domain")) for tid in heldout_ids)),
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    split_rows = [
        {
            "split": "train_candidate",
            "tasks": len(train_ids),
            "domains": payload["domain_counts_by_split"]["train"],
        },
        {
            "split": "val_candidate",
            "tasks": len(val_ids),
            "domains": payload["domain_counts_by_split"]["val"],
        },
        {
            "split": "heldout_candidate",
            "tasks": len(heldout_ids),
            "domains": payload["domain_counts_by_split"]["heldout"],
        },
    ]
    lines = [
        "# Hidden-Origin V3 Split Guard",
        "",
        f"BG_HIDDEN_ORIGIN_SPLIT_GUARD_V3_VERDICT = {verdict}",
        "",
        f"- candidate_tasks: `{n}`",
        f"- heldout_candidates: `{len(heldout_ids)}`",
        f"- clean_cross_version_heldout_task_ids: `{len(clean_cross)}`",
        f"- v1_v2_baseline_may_have_seen_v3_heldout_task_ids: `{baseline_overlap}`",
        f"- v3_heldout_excluded_from_v3_empirical_direction_construction: `True`",
        f"- v3_heldout_excluded_from_v3_tap_training: `True`",
        "",
        "## Splits",
        "",
    ]
    lines.extend(md_table(split_rows, ["split", "tasks", "domains"]))
    lines.extend(["", "## Heldout IDs", "", *[f"- `{tid}`" for tid in sorted(heldout_ids)[:80]], "", f"JSON: `{rel(OUT_JSON)}`"])
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_SPLIT_GUARD_V3_VERDICT = {verdict}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

