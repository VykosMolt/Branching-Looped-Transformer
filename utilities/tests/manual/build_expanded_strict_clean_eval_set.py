"""Build the expanded strict-clean code eval set and leakage-safe train pool.

This script inventories existing generated code candidates and unit-test
labels only. It does not generate code, capture features, or train heads.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

THIS_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == THIS_DIR:
    sys.path.pop(0)
import torch
sys.path.insert(0, str(THIS_DIR))

try:
    from utilities.tests.manual.code_branch_pilot_lib import REPORT_DIR, repo_path, write_json
except ModuleNotFoundError:
    from code_branch_pilot_lib import REPORT_DIR, repo_path, write_json

from build_code_specific_training_control_splits import (  # noqa: E402
    PRIMARY_TRAIN_INCORRECT_LABELS,
    add_candidate,
    build_pairs,
    candidate_recap_state,
    load_feature_uid_inventory,
    load_json,
    merge_task_meta,
    relevant_feature_paths,
    task_tests,
)


OLD6_TASK_IDS = [
    "mbpp/100",
    "mbpp/129",
    "mbpp/283",
    "mbpp/291",
    "mbpp/391",
    "mbpp/392",
]

NEW10_TASK_IDS = [
    "mbpp/11",
    "mbpp/20",
    "mbpp/434",
    "HumanEval/10",
    "HumanEval/118",
    "HumanEval/123",
    "HumanEval/125",
    "HumanEval/141",
    "HumanEval/148",
    "HumanEval/69",
]

ALL16_TASK_IDS = OLD6_TASK_IDS + NEW10_TASK_IDS

PRIMARY_EVAL_LABELS = {"correct", "near_miss"}
SECONDARY_EVAL_LABELS = {"correct", "near_miss", "wrong_code"}

OUTPUT_JSON = REPORT_DIR / "code_expanded_strict_clean_eval_set_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "code_expanded_strict_clean_eval_set_2026-05-17.md"
PREVIOUS_SPLIT_JSON = REPORT_DIR / "code_specific_training_control_splits_2026-05-17.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    parser.add_argument("--previous-split", default=str(PREVIOUS_SPLIT_JSON))
    return parser.parse_args()


def rel(path: Path | str) -> str:
    return repo_path(Path(path))


def discover_json_artifacts() -> list[Path]:
    found: set[Path] = set()
    patterns = (
        "code_strict_clean_transfer_*2026-05-17*.json",
        "code_strict_clean_screening_*2026-05-17*.json",
        "code_specific_control_and_screening_2026-05-17*.json",
        "code_specific_tiny_head_control_2026-05-17*.json",
        "code_branch_*v2*2026-05-16*.json",
        "code_branch_tournaments_v2_near_miss10_2026-05-17*.json",
        "code_branch_near_miss_balanced_tournaments_2026-05-17*.json",
    )
    for pattern in patterns:
        found.update(REPORT_DIR.glob(pattern))
    return sorted(path for path in found if path.is_file())


def ingest_task_contexts(
    *,
    artifact: Path,
    payload: dict[str, Any],
    task_meta: dict[str, dict[str, Any]],
) -> None:
    for task in payload.get("tasks", []) or []:
        if isinstance(task, dict):
            merge_task_meta(task_meta, task, artifact)
    for task in payload.get("task_rows", []) or []:
        if isinstance(task, dict):
            merge_task_meta(task_meta, task, artifact)
    by_task = payload.get("by_task", {})
    if isinstance(by_task, dict):
        for task_id, task in by_task.items():
            if isinstance(task, dict):
                merge_task_meta(task_meta, {"task_id": task_id, **task}, artifact)


def ingest_artifact(
    *,
    artifact: Path,
    payload: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    by_task: dict[str, set[str]],
    task_meta: dict[str, dict[str, Any]],
) -> int:
    count = 0
    ingest_task_contexts(artifact=artifact, payload=payload, task_meta=task_meta)
    if isinstance(payload.get("candidate_index"), dict):
        for row_any in payload.get("candidate_index", {}).values():
            if not isinstance(row_any, dict):
                continue
            task_id = str(row_any.get("task_id") or "")
            task_context = task_meta.get(task_id, {"task_id": task_id})
            add_candidate(
                candidates=candidates,
                by_task=by_task,
                task_meta=task_meta,
                row=row_any,
                task_context=task_context,
                artifact=artifact,
                candidate_set="candidate_index",
            )
            count += 1
    for row_any in payload.get("candidate_evaluations", []) or []:
        if not isinstance(row_any, dict):
            continue
        task_id = str(row_any.get("task_id") or "")
        task_context = task_meta.get(task_id, {"task_id": task_id})
        add_candidate(
            candidates=candidates,
            by_task=by_task,
            task_meta=task_meta,
            row=row_any,
            task_context=task_context,
            artifact=artifact,
            candidate_set="candidate_evaluations",
        )
        count += 1
    for tournament in payload.get("tournaments", []) or []:
        if not isinstance(tournament, dict):
            continue
        task_context = {
            "task_id": tournament.get("task_id"),
            "source": tournament.get("source", "unknown"),
            "difficulty": tournament.get("difficulty", "unknown"),
            "function_name": tournament.get("function_name", ""),
            "prompt": tournament.get("prompt", ""),
            "signature": tournament.get("signature", ""),
            "tests_visibility": tournament.get("tests_visibility", ""),
            "number_of_tests": tournament.get("number_of_tests"),
        }
        merge_task_meta(task_meta, task_context, artifact)
        keys = (
            "strict_clean_primary_candidates",
            "strict_clean_plus_wrong_code_candidates",
            "strict_candidates",
            "diagnostic_runnable_candidates",
            "diagnostic_mixed_primary_candidates",
            "diagnostic_candidates",
        )
        for key in keys:
            for row_any in tournament.get(key, []) or []:
                if not isinstance(row_any, dict):
                    continue
                add_candidate(
                    candidates=candidates,
                    by_task=by_task,
                    task_meta=task_meta,
                    row=row_any,
                    task_context=task_context,
                    artifact=artifact,
                    candidate_set=key,
                )
                count += 1
    return count


def rows_for_labels(
    candidates: dict[str, dict[str, Any]],
    task_uids: Iterable[str],
    labels: set[str],
) -> list[dict[str, Any]]:
    rows = [candidates[uid] for uid in task_uids if candidates[uid].get("label") in labels]
    return sorted(rows, key=lambda row: (row.get("label") != "correct", row.get("label"), row["candidate_uid"]))


def eval_tournament(
    *,
    task_id: str,
    candidates: dict[str, dict[str, Any]],
    task_uids: Iterable[str],
    labels: set[str],
    task_meta: dict[str, dict[str, Any]],
    group: str,
) -> dict[str, Any] | None:
    rows = rows_for_labels(candidates, task_uids, labels)
    if not any(row.get("label") == "correct" for row in rows):
        return None
    if not any(row.get("label") == "near_miss" for row in rows):
        return None
    meta = task_meta.get(task_id, {"task_id": task_id})
    return {
        "task_id": task_id,
        "group": group,
        "source": meta.get("source", "unknown"),
        "difficulty": meta.get("difficulty", "unknown"),
        "function_name": meta.get("function_name", ""),
        "prompt": meta.get("prompt", ""),
        "tests": task_tests(meta),
        "candidate_uids": [row["candidate_uid"] for row in rows],
        "labels": [row["label"] for row in rows],
        "label_counts": dict(Counter(row["label"] for row in rows)),
    }


def split_eval_rows(rows: list[dict[str, Any]], task_ids: list[str], name: str) -> list[dict[str, Any]]:
    wanted = set(task_ids)
    out = [dict(row) for row in rows if row["task_id"] in wanted]
    order = {task_id: idx for idx, task_id in enumerate(task_ids)}
    out.sort(key=lambda row: order.get(row["task_id"], 9999))
    for idx, row in enumerate(out):
        row["tournament_id"] = idx
        row["eval_set"] = name
    return out


def random_top1_baseline(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return float("nan")
    rates = []
    for row in rows:
        counts = Counter(row["labels"])
        rates.append(float(counts.get("correct", 0)) / max(len(row["labels"]), 1))
    return sum(rates) / len(rates)


def aggregate_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for row in rows:
        counts.update(row["labels"])
    return {
        "n_tasks": len(rows),
        "n_candidates": sum(len(row["candidate_uids"]) for row in rows),
        "n_correct": counts.get("correct", 0),
        "n_near_miss": counts.get("near_miss", 0),
        "n_wrong_code": counts.get("wrong_code", 0),
        "label_counts": dict(counts),
        "random_top1_baseline": random_top1_baseline(rows),
    }


def inspect_previous_leakage(path: Path, eval_task_ids: set[str]) -> tuple[str, dict[str, Any]]:
    if not path.exists():
        return "UNKNOWN", {"reason": "previous split file missing", "path": rel(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return "UNKNOWN", {"reason": f"previous split load failed: {type(exc).__name__}: {exc}", "path": rel(path)}
    train_ids = {
        str(row.get("task_id"))
        for row in payload.get("training_tasks", []) or []
        if row.get("task_id")
    }
    for pair in payload.get("training_pairs_primary", []) or []:
        if pair.get("task_id"):
            train_ids.add(str(pair["task_id"]))
    if not train_ids:
        return "UNKNOWN", {"reason": "previous split had no inspectable training task IDs", "path": rel(path)}
    overlap = sorted(train_ids & eval_task_ids)
    verdict = "LEAKAGE_RISK" if overlap else "CLEAN"
    return verdict, {
        "path": rel(path),
        "previous_training_task_count": len(train_ids),
        "eval_task_overlap": overlap,
        "old6_overlap": sorted(set(OLD6_TASK_IDS) & train_ids),
        "new10_overlap": sorted(set(NEW10_TASK_IDS) & train_ids),
    }


def build_training_pool(
    *,
    candidates: dict[str, dict[str, Any]],
    by_task: dict[str, set[str]],
    task_meta: dict[str, dict[str, Any]],
    excluded_task_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    training_tasks: list[dict[str, Any]] = []
    training_pairs_primary: list[dict[str, Any]] = []
    training_pairs_near_miss_only: list[dict[str, Any]] = []
    for task_id in sorted(by_task):
        if task_id in excluded_task_ids:
            continue
        task_candidates = [
            candidates[uid]
            for uid in by_task[task_id]
            if candidates[uid].get("label") in {"correct", "near_miss", "wrong_code"}
        ]
        if not any(row.get("label") == "correct" for row in task_candidates):
            continue
        if not any(row.get("label") in PRIMARY_TRAIN_INCORRECT_LABELS for row in task_candidates):
            continue
        primary_pairs = build_pairs(candidates, by_task[task_id], PRIMARY_TRAIN_INCORRECT_LABELS)
        near_pairs = build_pairs(candidates, by_task[task_id], {"near_miss"})
        if not primary_pairs:
            continue
        labels = Counter(row["label"] for row in task_candidates)
        meta = task_meta.get(task_id, {"task_id": task_id})
        training_tasks.append({
            "task_id": task_id,
            "source": meta.get("source", "unknown"),
            "difficulty": meta.get("difficulty", "unknown"),
            "function_name": meta.get("function_name", ""),
            "label_counts": dict(labels),
            "candidate_uids": sorted(row["candidate_uid"] for row in task_candidates),
            "primary_pair_count": len(primary_pairs),
            "near_miss_only_pair_count": len(near_pairs),
        })
        training_pairs_primary.extend(primary_pairs)
        training_pairs_near_miss_only.extend(near_pairs)
    return training_tasks, training_pairs_primary, training_pairs_near_miss_only


def write_md(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# Expanded Strict-Clean Code Eval Set",
        "",
        f"EXPANDED_STRICT_CLEAN_SET_VERDICT = {payload['expanded_strict_clean_set_verdict']}",
        f"CODE_SPECIFIC_LEAKAGE_VERDICT = {payload['code_specific_leakage_verdict']}",
        "",
        "## Eval Set",
        "",
        f"- loaded_strict_clean_tasks: `{s['loaded_strict_clean_tasks']}` / `16`",
        f"- loaded_old6: `{s['loaded_old6']}` / `6`",
        f"- loaded_new10: `{s['loaded_new10']}` / `10`",
        f"- all16_primary_candidates: `{s['eval_sets']['ALL16_primary']['n_candidates']}`",
        f"- all16_correct / near_miss: `{s['eval_sets']['ALL16_primary']['n_correct']}` / `{s['eval_sets']['ALL16_primary']['n_near_miss']}`",
        f"- all16_wrong_code_secondary: `{s['eval_sets']['ALL16_plus_wrong_code']['n_wrong_code']}`",
        f"- all16_random_top1_baseline: `{s['eval_sets']['ALL16_primary']['random_top1_baseline']}`",
        "",
        "## Split Baselines",
        "",
    ]
    for name in ("OLD6_primary", "NEW10_primary", "ALL16_primary"):
        row = s["eval_sets"][name]
        lines.append(
            f"- `{name}` tasks=`{row['n_tasks']}` candidates=`{row['n_candidates']}` "
            f"labels=`{row['label_counts']}` random_top1=`{row['random_top1_baseline']}`"
        )
    lines.extend([
        "",
        "## Code-Specific Training Pool",
        "",
        f"- training_tasks_excluding_all16: `{s['training_task_count']}`",
        f"- primary_training_pairs: `{s['primary_training_pair_count']}`",
        f"- near_miss_only_training_pairs: `{s['near_miss_only_training_pair_count']}`",
        f"- required_feature_candidates: `{s['required_feature_candidate_count']}`",
        f"- missing_feature_candidates: `{s['missing_feature_candidate_count']}`",
        "",
        "## Strict-Clean Tasks",
        "",
    ])
    for row in payload["strict_clean_tasks"]:
        lines.append(f"- `{row['task_id']}` group=`{row['group']}` labels=`{row['label_counts']}`")
    if payload.get("missing_or_non_strict_tasks"):
        lines.extend(["", "## Missing Or Non-Strict Tasks", ""])
        for row in payload["missing_or_non_strict_tasks"]:
            lines.append(f"- `{row['task_id']}` group=`{row['group']}` labels=`{row.get('label_counts', {})}`")
    if payload.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {item}" for item in payload["blockers"])
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_json = Path(args.output)
    out_md = Path(args.output_md)
    eval_task_ids = set(ALL16_TASK_IDS)

    candidates: dict[str, dict[str, Any]] = {}
    by_task: dict[str, set[str]] = defaultdict(set)
    task_meta: dict[str, dict[str, Any]] = {}
    inventory_json: list[dict[str, Any]] = []
    blockers: list[str] = []

    for path in discover_json_artifacts():
        try:
            payload = load_json(path)
            before = len(candidates)
            mentions = ingest_artifact(
                artifact=path,
                payload=payload,
                candidates=candidates,
                by_task=by_task,
                task_meta=task_meta,
            )
            inventory_json.append({
                "path": rel(path),
                "loaded": True,
                "raw_candidate_mentions": mentions,
                "new_unique_candidates": len(candidates) - before,
            })
        except Exception as exc:
            inventory_json.append({"path": rel(path), "loaded": False, "error": f"{type(exc).__name__}: {exc}"})

    if not any(row.get("loaded") and row.get("raw_candidate_mentions", 0) for row in inventory_json):
        blockers.append("No loadable code candidate artifacts with unit-test labels were found.")

    primary_all: list[dict[str, Any]] = []
    secondary_all: list[dict[str, Any]] = []
    missing_or_non_strict: list[dict[str, Any]] = []
    for task_id in ALL16_TASK_IDS:
        group = "OLD6" if task_id in OLD6_TASK_IDS else "NEW10"
        task_uids = by_task.get(task_id, set())
        primary = eval_tournament(
            task_id=task_id,
            candidates=candidates,
            task_uids=task_uids,
            labels=PRIMARY_EVAL_LABELS,
            task_meta=task_meta,
            group=group,
        ) if task_uids else None
        secondary = eval_tournament(
            task_id=task_id,
            candidates=candidates,
            task_uids=task_uids,
            labels=SECONDARY_EVAL_LABELS,
            task_meta=task_meta,
            group=group,
        ) if task_uids else None
        if primary:
            primary_all.append(primary)
            if secondary:
                secondary_all.append(secondary)
        else:
            labels = Counter(candidates[uid].get("label") for uid in task_uids)
            missing_or_non_strict.append({"task_id": task_id, "group": group, "label_counts": dict(labels)})

    old6_primary = split_eval_rows(primary_all, OLD6_TASK_IDS, "OLD6_primary")
    new10_primary = split_eval_rows(primary_all, NEW10_TASK_IDS, "NEW10_primary")
    all16_primary = split_eval_rows(primary_all, ALL16_TASK_IDS, "ALL16_primary")
    old6_secondary = split_eval_rows(secondary_all, OLD6_TASK_IDS, "OLD6_plus_wrong_code")
    new10_secondary = split_eval_rows(secondary_all, NEW10_TASK_IDS, "NEW10_plus_wrong_code")
    all16_secondary = split_eval_rows(secondary_all, ALL16_TASK_IDS, "ALL16_plus_wrong_code")
    mbpp_primary = [dict(row, tournament_id=idx, eval_set="MBPP_primary") for idx, row in enumerate(r for r in all16_primary if str(r.get("source")).lower() == "mbpp" or r["task_id"].startswith("mbpp/"))]
    humaneval_primary = [dict(row, tournament_id=idx, eval_set="HumanEval_primary") for idx, row in enumerate(r for r in all16_primary if r["task_id"].startswith("HumanEval/"))]

    training_tasks, training_pairs_primary, training_pairs_near_miss_only = build_training_pool(
        candidates=candidates,
        by_task=by_task,
        task_meta=task_meta,
        excluded_task_ids=eval_task_ids,
    )

    feature_paths = relevant_feature_paths()
    feature_uids, feature_inventory = load_feature_uid_inventory(feature_paths)

    eval_sets = {
        "OLD6_primary": old6_primary,
        "NEW10_primary": new10_primary,
        "ALL16_primary": all16_primary,
        "OLD6_plus_wrong_code": old6_secondary,
        "NEW10_plus_wrong_code": new10_secondary,
        "ALL16_plus_wrong_code": all16_secondary,
        "MBPP_primary": mbpp_primary,
        "HumanEval_primary": humaneval_primary,
    }
    required_uids = sorted({
        uid
        for rows in eval_sets.values()
        for tournament in rows
        for uid in tournament["candidate_uids"]
    } | {
        pair["preferred_uid"] for pair in training_pairs_primary
    } | {
        pair["rejected_uid"] for pair in training_pairs_primary
    })
    missing_feature_rows = []
    for uid in required_uids:
        if uid in feature_uids:
            continue
        row = candidates.get(uid, {"candidate_uid": uid})
        missing_feature_rows.append({
            "candidate_uid": uid,
            "task_id": row.get("task_id", ""),
            "label": row.get("label", ""),
            "source_artifacts": row.get("source_artifacts", []),
            "recapture_state": candidate_recap_state(row),
        })

    loaded_old6 = len(old6_primary)
    loaded_new10 = len(new10_primary)
    loaded_total = len(all16_primary)
    if blockers or loaded_total < 8:
        set_verdict = "BLOCKED"
    elif loaded_total >= 12 and loaded_old6 >= 5 and loaded_new10 >= 7:
        set_verdict = "READY"
    else:
        set_verdict = "PARTIAL"

    leakage_verdict, leakage = inspect_previous_leakage(Path(args.previous_split), eval_task_ids)
    train_label_counts = Counter()
    for row in training_tasks:
        train_label_counts.update(row["label_counts"])
    summary = {
        "EXPANDED_STRICT_CLEAN_SET_VERDICT": set_verdict,
        "CODE_SPECIFIC_LEAKAGE_VERDICT": leakage_verdict,
        "json_artifacts_loaded": sum(1 for row in inventory_json if row.get("loaded")),
        "json_artifacts_failed": sum(1 for row in inventory_json if not row.get("loaded")),
        "total_tasks_with_candidates": len(by_task),
        "total_unique_candidates": len(candidates),
        "loaded_strict_clean_tasks": loaded_total,
        "loaded_old6": loaded_old6,
        "loaded_new10": loaded_new10,
        "eval_sets": {name: aggregate_counts(rows) for name, rows in eval_sets.items()},
        "training_task_count": len(training_tasks),
        "primary_training_pair_count": len(training_pairs_primary),
        "near_miss_only_training_pair_count": len(training_pairs_near_miss_only),
        "training_label_counts": dict(train_label_counts),
        "feature_artifacts_found": len(feature_paths),
        "required_feature_candidate_count": len(required_uids),
        "covered_feature_candidate_count": len(required_uids) - len(missing_feature_rows),
        "missing_feature_candidate_count": len(missing_feature_rows),
        "missing_feature_recapturable_count": sum(1 for row in missing_feature_rows if row["recapture_state"] == "recapturable"),
        "missing_feature_blocked_count": sum(1 for row in missing_feature_rows if row["recapture_state"] != "recapturable"),
    }
    payload = {
        "expanded_strict_clean_set_verdict": set_verdict,
        "code_specific_leakage_verdict": leakage_verdict,
        "old6_task_ids": OLD6_TASK_IDS,
        "new10_task_ids": NEW10_TASK_IDS,
        "all16_task_ids": ALL16_TASK_IDS,
        "summary": summary,
        "json_artifact_inventory": inventory_json,
        "feature_artifact_inventory": feature_inventory,
        "previous_split_leakage": leakage,
        "candidate_index": {uid: candidates[uid] for uid in sorted(candidates)},
        "strict_clean_tasks": all16_primary,
        "missing_or_non_strict_tasks": missing_or_non_strict,
        "eval_sets": eval_sets,
        "training_tasks": training_tasks,
        "training_pairs_primary": training_pairs_primary,
        "training_pairs_near_miss_only": training_pairs_near_miss_only,
        "required_feature_candidate_uids": required_uids,
        "missing_feature_candidates": missing_feature_rows,
        "blockers": blockers,
        "notes": [
            "All 16 expanded strict-clean task IDs are excluded from code-specific training.",
            "Primary eval uses correct and near_miss candidates only.",
            "Secondary diagnostics add wrong_code candidates when present.",
            "Candidate labels come only from unit-test outcomes; tap/evaluator outputs are not labels.",
        ],
        "outputs": {"json": rel(out_json), "md": rel(out_md)},
    }
    write_json(out_json, payload)
    write_md(out_md, payload)
    print(f"EXPANDED_STRICT_CLEAN_SET_VERDICT = {set_verdict}")
    print(f"CODE_SPECIFIC_LEAKAGE_VERDICT = {leakage_verdict}")
    print(f"loaded_old6 = {loaded_old6}")
    print(f"loaded_new10 = {loaded_new10}")
    print(f"all16_primary_candidates = {summary['eval_sets']['ALL16_primary']['n_candidates']}")
    print(f"training_tasks = {len(training_tasks)}")
    print(f"primary_training_pairs = {len(training_pairs_primary)}")
    print(f"missing_feature_candidates = {len(missing_feature_rows)}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    if set_verdict == "BLOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
