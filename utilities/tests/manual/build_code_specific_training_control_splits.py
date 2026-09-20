"""Build task-disjoint code-specific tiny-head train/eval splits.

This script only inventories existing generated code candidates and unit-test
labels. It does not generate candidates, capture features, or train heads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    from utilities.tests.manual.code_branch_pilot_lib import PROJECT_ROOT, REPORT_DIR, repo_path, write_json
except ModuleNotFoundError:
    from code_branch_pilot_lib import PROJECT_ROOT, REPORT_DIR, repo_path, write_json


HELDOUT_STRICT_CLEAN_TASK_IDS = {
    "mbpp/100",
    "mbpp/129",
    "mbpp/283",
    "mbpp/291",
    "mbpp/391",
    "mbpp/392",
}

OUTPUT_JSON = REPORT_DIR / "code_specific_training_control_splits_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "code_specific_training_control_splits_2026-05-17.md"

PRIMARY_TRAIN_INCORRECT_LABELS = {"near_miss", "wrong_code"}
RUNTIME_DIAGNOSTIC_LABELS = {"near_miss", "wrong_code", "runtime_error"}
PRIMARY_EVAL_LABELS = {"correct", "near_miss"}
SECONDARY_EVAL_LABELS = {"correct", "near_miss", "wrong_code"}

PREFERRED_JSON_ARTIFACTS = [
    REPORT_DIR / "code_branch_tournaments_v2_mini_patched_2026-05-16.json",
    REPORT_DIR / "code_branch_tournaments_v2_near_miss10_2026-05-17.json",
    REPORT_DIR / "code_branch_near_miss_balanced_tournaments_2026-05-17.json",
    REPORT_DIR / "code_strict_clean_screening_results_2026-05-17.json",
    REPORT_DIR / "code_strict_clean_transfer_set_2026-05-17.json",
]

TASK_CONTEXT_ARTIFACTS = [
    REPORT_DIR / "code_branch_taskset_v2_mini_patched_2026-05-16.json",
    REPORT_DIR / "code_branch_taskset_v2_near_miss10_2026-05-17.json",
    REPORT_DIR / "code_strict_clean_screening_taskpool_2026-05-17.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha16(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8", errors="replace")).hexdigest()[:16]


def normalized_code(code: str) -> str:
    return "\n".join(line.rstrip() for line in str(code or "").strip().splitlines() if line.strip())


def candidate_code(row: dict[str, Any]) -> str:
    return str(
        row.get("final_code_for_unit_tests")
        or row.get("final_code")
        or row.get("candidate_code")
        or row.get("sanitized_code")
        or ""
    )


def candidate_fingerprint(row: dict[str, Any], code: str | None = None) -> str:
    for key in ("ast_hash", "normalized_code_hash", "raw_code_hash"):
        value = str(row.get(key) or "").strip()
        if value:
            return value[:16]
    code = normalized_code(code if code is not None else candidate_code(row))
    return sha16(code) if code else sha16(json.dumps(row, sort_keys=True, default=str))


def stable_candidate_uid(row: dict[str, Any], task_id: str | None = None, code: str | None = None) -> str:
    explicit = str(row.get("candidate_uid") or "").strip()
    if explicit:
        return explicit
    tid = str(task_id or row.get("task_id") or "unknown_task")
    return f"generated::{tid}::{candidate_fingerprint(row, code)}"


def candidate_aliases(row: dict[str, Any], task_id: str | None = None, code: str | None = None) -> set[str]:
    tid = str(task_id or row.get("task_id") or "unknown_task")
    code_text = candidate_code(row) if code is None else str(code or "")
    aliases = {stable_candidate_uid(row, tid, code_text)}
    explicit = str(row.get("candidate_uid") or "").strip()
    if explicit:
        aliases.add(explicit)
        aliases.add(f"generated::{tid}::{candidate_fingerprint(row, code_text)}")
    if code_text.strip():
        aliases.add(f"generated::{tid}::{sha16(normalized_code(code_text))}")
    return aliases


def rel(path: Path | str) -> str:
    return repo_path(Path(path))


def task_tests(task: dict[str, Any]) -> list[str]:
    tests = list(task.get("tests") or [])
    if not tests:
        tests = list(task.get("public_tests", [])) + list(task.get("hidden_tests", []))
    return [str(test).strip() for test in tests if str(test).strip()]


def merge_task_meta(task_meta: dict[str, dict[str, Any]], task: dict[str, Any], artifact: Path) -> None:
    task_id = str(task.get("task_id") or "")
    if not task_id:
        return
    current = task_meta.setdefault(task_id, {"task_id": task_id, "source_artifacts": []})
    for key in (
        "source",
        "difficulty",
        "function_name",
        "prompt",
        "signature",
        "signature_hint",
        "tests_visibility",
        "timeout_seconds",
    ):
        value = task.get(key)
        if value not in (None, "", []):
            current.setdefault(key, value)
    tests = task_tests(task)
    if tests:
        current.setdefault("tests", tests)
        current.setdefault("number_of_tests", len(tests))
    artifact_rel = rel(artifact)
    if artifact_rel not in current["source_artifacts"]:
        current["source_artifacts"].append(artifact_rel)


def add_candidate(
    *,
    candidates: dict[str, dict[str, Any]],
    by_task: dict[str, set[str]],
    task_meta: dict[str, dict[str, Any]],
    row: dict[str, Any],
    task_context: dict[str, Any],
    artifact: Path,
    candidate_set: str,
) -> None:
    if row.get("duplicate_of"):
        return
    task_id = str(row.get("task_id") or task_context.get("task_id") or "")
    if not task_id:
        return
    label = str(row.get("unit_test_label") or row.get("label") or "").strip()
    if not label:
        return
    code = candidate_code(row)
    uid = stable_candidate_uid(row, task_id, code)
    prompt = str(task_context.get("prompt") or row.get("prompt") or "")
    merge_task_meta(task_meta, {**task_context, "task_id": task_id, "prompt": prompt}, artifact)
    record = candidates.setdefault(uid, {
        "candidate_uid": uid,
        "task_id": task_id,
        "source": row.get("source") or task_context.get("source", "unknown"),
        "difficulty": row.get("difficulty") or task_context.get("difficulty", "unknown"),
        "function_name": row.get("function_name") or task_context.get("function_name", ""),
        "prompt": prompt,
        "final_code": code,
        "label": label,
        "unit_test_label": label,
        "is_correct": label == "correct",
        "is_runnable": bool(row.get("is_runnable")),
        "tests_total": row.get("tests_total"),
        "tests_passed": row.get("tests_passed"),
        "pass_rate": row.get("pass_rate"),
        "mode": row.get("mode", ""),
        "candidate_stage": row.get("candidate_stage", ""),
        "screening_role": row.get("screening_role", ""),
        "route": row.get("route", ""),
        "ast_hash": row.get("ast_hash", ""),
        "normalized_code_hash": row.get("normalized_code_hash", ""),
        "raw_code_hash": row.get("raw_code_hash", ""),
        "aliases": sorted(candidate_aliases(row, task_id, code)),
        "source_artifacts": [],
        "candidate_sets": [],
    })
    if not record.get("prompt") and prompt:
        record["prompt"] = prompt
    if not record.get("final_code") and code:
        record["final_code"] = code
    if label != record.get("label"):
        conflicts = record.setdefault("label_conflicts", [])
        conflicts.append({"artifact": rel(artifact), "label": label})
    artifact_rel = rel(artifact)
    if artifact_rel not in record["source_artifacts"]:
        record["source_artifacts"].append(artifact_rel)
    if candidate_set not in record["candidate_sets"]:
        record["candidate_sets"].append(candidate_set)
    by_task[task_id].add(uid)


def generic_candidate_lists(tournament: dict[str, Any]) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    keys = (
        "strict_candidates",
        "diagnostic_runnable_candidates",
        "diagnostic_mixed_primary_candidates",
        "diagnostic_candidates",
    )
    seen: set[str] = set()
    for key in keys:
        for row in tournament.get(key, []) or []:
            uid = stable_candidate_uid(row, str(row.get("task_id") or tournament.get("task_id") or ""), candidate_code(row))
            if uid in seen:
                continue
            seen.add(uid)
            yield key, [row]


def ingest_tournament_payload(
    *,
    artifact: Path,
    payload: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    by_task: dict[str, set[str]],
    task_meta: dict[str, dict[str, Any]],
) -> int:
    count = 0
    for tournament in payload.get("tournaments", []) or []:
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
        if "strict_clean_primary_candidates" in tournament:
            for key in ("strict_clean_primary_candidates", "strict_clean_plus_wrong_code_candidates"):
                for row in tournament.get(key, []) or []:
                    add_candidate(
                        candidates=candidates,
                        by_task=by_task,
                        task_meta=task_meta,
                        row=row,
                        task_context=task_context,
                        artifact=artifact,
                        candidate_set=key,
                    )
                    count += 1
            continue
        for key, rows in generic_candidate_lists(tournament):
            for row in rows:
                add_candidate(
                    candidates=candidates,
                    by_task=by_task,
                    task_meta=task_meta,
                    row=row,
                    task_context=task_context,
                    artifact=artifact,
                    candidate_set=key,
                )
                count += 1
    return count


def ingest_screening_results(
    *,
    artifact: Path,
    payload: dict[str, Any],
    task_meta: dict[str, dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
    by_task: dict[str, set[str]],
) -> int:
    count = 0
    for row in payload.get("candidate_evaluations", []) or []:
        task_id = str(row.get("task_id") or "")
        task_context = task_meta.get(task_id, {"task_id": task_id})
        add_candidate(
            candidates=candidates,
            by_task=by_task,
            task_meta=task_meta,
            row=row,
            task_context=task_context,
            artifact=artifact,
            candidate_set="screening_candidate_evaluations",
        )
        count += 1
    return count


def discover_json_artifacts() -> list[Path]:
    found = {path for path in PREFERRED_JSON_ARTIFACTS if path.exists()}
    for pattern in (
        "code_branch_pilot_v2_mini_patched_2026-05-16*.json",
        "code_branch_near_miss_enrichment10_2026-05-17*.json",
        "code_branch_near_miss_balancing_2026-05-17*.json",
        "code_strict_clean_screening_*2026-05-17*.json",
        "code_strict_clean_transfer_*2026-05-17*.json",
    ):
        found.update(REPORT_DIR.glob(pattern))
    return sorted(found)


def relevant_feature_paths() -> list[Path]:
    found: set[Path] = set()
    for path in REPORT_DIR.glob("code*.pt"):
        name = path.name
        if "feature" in name or "tap_features" in name:
            found.add(path)
    for path in (
        REPORT_DIR / "code_branch_tap_features_v2_mini_patched_2026-05-16.pt",
        REPORT_DIR / "code_strict_clean_transfer_features_2026-05-17.pt",
    ):
        if path.exists():
            found.add(path)
    return sorted(found)


def feature_text(prompt: str, code: str) -> str:
    return (
        "Problem:\n"
        + str(prompt).strip()
        + "\n\nCandidate solution:\n```python\n"
        + str(code).strip()
        + "\n```"
    )


def feature_uids_from_payload(payload: dict[str, Any]) -> set[str]:
    uids: set[str] = set()
    if "candidate_features" in payload:
        for row in payload.get("candidate_features", []) or []:
            meta = dict(row.get("candidate_metadata") or {})
            uid = str(row.get("candidate_uid") or meta.get("candidate_uid") or "").strip()
            if uid:
                uids.add(uid)
            code = candidate_code(meta)
            task_id = str(meta.get("task_id") or row.get("task_id") or "")
            if task_id:
                uids.update(candidate_aliases(meta, task_id, code))
    for record in payload.get("records", []) or []:
        task_id = str(record.get("task_id") or "")
        codes = list(record.get("candidate_codes", []) or [])
        metadata = list(record.get("candidate_metadata", []) or [])
        for idx, meta in enumerate(metadata):
            code = codes[idx] if idx < len(codes) else candidate_code(meta)
            uids.update(candidate_aliases(dict(meta), task_id, code))
    return uids


def load_feature_uid_inventory(paths: list[Path]) -> tuple[set[str], list[dict[str, Any]]]:
    all_uids: set[str] = set()
    inventory: list[dict[str, Any]] = []
    for path in paths:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            uids = feature_uids_from_payload(payload)
            meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
            inventory.append({
                "path": rel(path),
                "loaded": True,
                "uids": len(uids),
                "meta": {k: v for k, v in dict(meta).items() if k not in {"records", "candidate_features"}},
            })
            all_uids.update(uids)
        except Exception as exc:
            inventory.append({"path": rel(path), "loaded": False, "error": f"{type(exc).__name__}: {exc}"})
    return all_uids, inventory


def build_pairs(candidates: dict[str, dict[str, Any]], task_uids: Iterable[str], labels: set[str]) -> list[dict[str, Any]]:
    rows = [candidates[uid] for uid in task_uids]
    correct = [row for row in rows if row.get("label") == "correct"]
    incorrect = [row for row in rows if row.get("label") in labels]
    pairs: list[dict[str, Any]] = []
    for left in correct:
        for right in incorrect:
            pairs.append({
                "task_id": left["task_id"],
                "preferred_uid": left["candidate_uid"],
                "rejected_uid": right["candidate_uid"],
                "preferred_label": "correct",
                "rejected_label": right["label"],
            })
    return pairs


def tournament_for_task(
    *,
    task_id: str,
    candidates: dict[str, dict[str, Any]],
    task_uids: Iterable[str],
    labels: set[str],
    task_meta: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    rows = [candidates[uid] for uid in task_uids if candidates[uid].get("label") in labels]
    rows = [row for row in rows if row.get("is_runnable", True)]
    if not any(row.get("label") == "correct" for row in rows):
        return None
    if "near_miss" in labels and not any(row.get("label") == "near_miss" for row in rows):
        return None
    meta = task_meta.get(task_id, {"task_id": task_id})
    rows = sorted(rows, key=lambda row: (row["label"] != "correct", row["candidate_uid"]))
    return {
        "task_id": task_id,
        "source": meta.get("source", "unknown"),
        "difficulty": meta.get("difficulty", "unknown"),
        "function_name": meta.get("function_name", ""),
        "prompt": meta.get("prompt", ""),
        "candidate_uids": [row["candidate_uid"] for row in rows],
        "labels": [row["label"] for row in rows],
        "label_counts": dict(Counter(row["label"] for row in rows)),
    }


def candidate_recap_state(row: dict[str, Any]) -> str:
    if str(row.get("prompt") or "").strip() and str(row.get("final_code") or "").strip():
        return "recapturable"
    return "missing_prompt_or_code"


def write_md(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# Code-Specific Training-Control Splits",
        "",
        f"CODE_SPECIFIC_SPLIT_VERDICT = {payload['code_specific_split_verdict']}",
        "",
        "## Inventory",
        "",
        f"- json_artifacts_loaded: `{s['json_artifacts_loaded']}`",
        f"- json_artifacts_failed: `{s['json_artifacts_failed']}`",
        f"- feature_artifacts_found: `{s['feature_artifacts_found']}`",
        f"- total_tasks_with_candidates: `{s['total_tasks_with_candidates']}`",
        f"- total_unique_candidates: `{s['total_unique_candidates']}`",
        "",
        "## Train Split",
        "",
        f"- heldout_strict_clean_task_ids: `{payload['heldout_strict_clean_task_ids']}`",
        f"- training_tasks: `{s['training_task_count']}`",
        f"- primary_training_pairs: `{s['primary_training_pair_count']}`",
        f"- near_miss_only_training_pairs: `{s['near_miss_only_training_pair_count']}`",
        f"- runtime_diagnostic_training_pairs: `{s['runtime_diagnostic_training_pair_count']}`",
        f"- training_label_counts: `{s['training_label_counts']}`",
        "",
        "## Eval Sets",
        "",
        f"- primary_strict_clean_tasks: `{s['primary_strict_clean_task_count']}`",
        f"- primary_strict_clean_candidates: `{s['primary_strict_clean_candidate_count']}`",
        f"- secondary_plus_wrong_code_candidates: `{s['secondary_plus_wrong_code_candidate_count']}`",
        f"- diagnostic_runnable_holdout_tasks: `{s['diagnostic_runnable_holdout_task_count']}`",
        "",
        "## Feature Coverage",
        "",
        f"- required_feature_candidates: `{s['required_feature_candidate_count']}`",
        f"- covered_feature_candidates: `{s['covered_feature_candidate_count']}`",
        f"- missing_feature_candidates: `{s['missing_feature_candidate_count']}`",
        f"- missing_recapturable: `{s['missing_feature_recapturable_count']}`",
        f"- missing_blocked: `{s['missing_feature_blocked_count']}`",
        "",
        "Missing feature candidates are listed in the JSON with task IDs, labels, and recapture state.",
        "",
        "## Training Tasks",
        "",
    ]
    for row in payload["training_tasks"][:80]:
        lines.append(
            f"- `{row['task_id']}` labels=`{row['label_counts']}` pairs=`{row['primary_pair_count']}` "
            f"source=`{row.get('source', 'unknown')}`"
        )
    lines.extend(["", "## Held-Out Primary Strict-Clean Tasks", ""])
    for row in payload["eval_sets"]["primary_strict_clean"]:
        lines.append(f"- `{row['task_id']}` labels=`{row['label_counts']}` candidates=`{len(row['candidate_uids'])}`")
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

    task_meta: dict[str, dict[str, Any]] = {}
    for path in TASK_CONTEXT_ARTIFACTS:
        if not path.exists():
            continue
        try:
            payload = load_json(path)
            for task in payload.get("tasks", []) or []:
                merge_task_meta(task_meta, task, path)
        except Exception:
            continue

    candidates: dict[str, dict[str, Any]] = {}
    by_task: dict[str, set[str]] = defaultdict(set)
    inventory_json: list[dict[str, Any]] = []
    blockers: list[str] = []
    for path in discover_json_artifacts():
        if not path.exists():
            continue
        try:
            payload = load_json(path)
            before = len(candidates)
            count = 0
            if path.name == "code_strict_clean_screening_results_2026-05-17.json":
                count = ingest_screening_results(
                    artifact=path,
                    payload=payload,
                    task_meta=task_meta,
                    candidates=candidates,
                    by_task=by_task,
                )
            elif "tournaments" in payload:
                count = ingest_tournament_payload(
                    artifact=path,
                    payload=payload,
                    candidates=candidates,
                    by_task=by_task,
                    task_meta=task_meta,
                )
            inventory_json.append({
                "path": rel(path),
                "loaded": True,
                "raw_candidate_mentions": count,
                "new_unique_candidates": len(candidates) - before,
                "has_tournaments": "tournaments" in payload,
                "has_candidate_evaluations": "candidate_evaluations" in payload,
            })
        except Exception as exc:
            inventory_json.append({"path": rel(path), "loaded": False, "error": f"{type(exc).__name__}: {exc}"})

    if not any(row.get("loaded") and row.get("raw_candidate_mentions", 0) for row in inventory_json):
        blockers.append("No loadable code candidate/tournament artifacts with unit-test labels were found.")

    feature_paths = relevant_feature_paths()
    feature_uids, feature_inventory = load_feature_uid_inventory(feature_paths)

    training_tasks: list[dict[str, Any]] = []
    training_pairs_primary: list[dict[str, Any]] = []
    training_pairs_near_miss_only: list[dict[str, Any]] = []
    training_pairs_runtime_diagnostic: list[dict[str, Any]] = []
    for task_id in sorted(by_task):
        if task_id in HELDOUT_STRICT_CLEAN_TASK_IDS:
            continue
        task_candidates = [candidates[uid] for uid in by_task[task_id]]
        runnable = [row for row in task_candidates if row.get("label") in {"correct", "near_miss", "wrong_code", "runtime_error"}]
        if not any(row.get("label") == "correct" for row in runnable):
            continue
        if not any(row.get("label") in PRIMARY_TRAIN_INCORRECT_LABELS for row in runnable):
            continue
        primary_pairs = build_pairs(candidates, by_task[task_id], PRIMARY_TRAIN_INCORRECT_LABELS)
        near_pairs = build_pairs(candidates, by_task[task_id], {"near_miss"})
        runtime_pairs = build_pairs(candidates, by_task[task_id], RUNTIME_DIAGNOSTIC_LABELS)
        if not primary_pairs:
            continue
        labels = Counter(row["label"] for row in runnable)
        meta = task_meta.get(task_id, {"task_id": task_id})
        training_tasks.append({
            "task_id": task_id,
            "source": meta.get("source", "unknown"),
            "difficulty": meta.get("difficulty", "unknown"),
            "function_name": meta.get("function_name", ""),
            "label_counts": dict(labels),
            "candidate_uids": sorted(row["candidate_uid"] for row in runnable if row["label"] in {"correct", "near_miss", "wrong_code"}),
            "primary_pair_count": len(primary_pairs),
            "near_miss_only_pair_count": len(near_pairs),
            "runtime_diagnostic_pair_count": len(runtime_pairs),
        })
        training_pairs_primary.extend(primary_pairs)
        training_pairs_near_miss_only.extend(near_pairs)
        training_pairs_runtime_diagnostic.extend(runtime_pairs)

    primary_eval: list[dict[str, Any]] = []
    secondary_eval: list[dict[str, Any]] = []
    for task_id in sorted(HELDOUT_STRICT_CLEAN_TASK_IDS):
        if task_id not in by_task:
            continue
        primary = tournament_for_task(
            task_id=task_id,
            candidates=candidates,
            task_uids=by_task[task_id],
            labels=PRIMARY_EVAL_LABELS,
            task_meta=task_meta,
        )
        secondary = tournament_for_task(
            task_id=task_id,
            candidates=candidates,
            task_uids=by_task[task_id],
            labels=SECONDARY_EVAL_LABELS,
            task_meta=task_meta,
        )
        if primary:
            primary_eval.append(primary)
        if secondary:
            secondary_eval.append(secondary)

    eval_sets = {
        "primary_strict_clean": primary_eval,
        "secondary_plus_wrong_code": secondary_eval,
        "diagnostic_runnable_holdout": [],
    }

    required_uids = sorted({
        pair["preferred_uid"]
        for pair in training_pairs_primary
    } | {
        pair["rejected_uid"]
        for pair in training_pairs_primary
    } | {
        uid
        for rows in eval_sets.values()
        for tournament in rows
        for uid in tournament["candidate_uids"]
    })
    missing = [uid for uid in required_uids if uid not in feature_uids]
    missing_rows = []
    for uid in missing:
        row = candidates.get(uid, {"candidate_uid": uid})
        missing_rows.append({
            "candidate_uid": uid,
            "task_id": row.get("task_id", ""),
            "label": row.get("label", ""),
            "source_artifacts": row.get("source_artifacts", []),
            "recapture_state": candidate_recap_state(row),
        })

    train_label_counts = Counter()
    for row in training_tasks:
        train_label_counts.update(row["label_counts"])
    primary_candidate_count = sum(len(row["candidate_uids"]) for row in primary_eval)
    secondary_candidate_count = sum(len(row["candidate_uids"]) for row in secondary_eval)

    if blockers:
        verdict = "BLOCKED"
    elif len(training_tasks) < 8 or len(training_pairs_primary) < 30:
        verdict = "INSUFFICIENT_TRAIN"
    elif len(primary_eval) < 5:
        verdict = "INSUFFICIENT_EVAL"
    elif missing:
        verdict = "MISSING_FEATURES"
    else:
        verdict = "READY"

    summary = {
        "json_artifacts_loaded": sum(1 for row in inventory_json if row.get("loaded")),
        "json_artifacts_failed": sum(1 for row in inventory_json if not row.get("loaded")),
        "feature_artifacts_found": len(feature_paths),
        "total_tasks_with_candidates": len(by_task),
        "total_unique_candidates": len(candidates),
        "training_task_count": len(training_tasks),
        "primary_training_pair_count": len(training_pairs_primary),
        "near_miss_only_training_pair_count": len(training_pairs_near_miss_only),
        "runtime_diagnostic_training_pair_count": len(training_pairs_runtime_diagnostic),
        "training_label_counts": dict(train_label_counts),
        "primary_strict_clean_task_count": len(primary_eval),
        "primary_strict_clean_candidate_count": primary_candidate_count,
        "secondary_plus_wrong_code_task_count": len(secondary_eval),
        "secondary_plus_wrong_code_candidate_count": secondary_candidate_count,
        "diagnostic_runnable_holdout_task_count": 0,
        "required_feature_candidate_count": len(required_uids),
        "covered_feature_candidate_count": len(required_uids) - len(missing),
        "missing_feature_candidate_count": len(missing),
        "missing_feature_recapturable_count": sum(1 for row in missing_rows if row["recapture_state"] == "recapturable"),
        "missing_feature_blocked_count": sum(1 for row in missing_rows if row["recapture_state"] != "recapturable"),
        "feature_coverage_complete": not missing,
    }
    payload = {
        "code_specific_split_verdict": verdict,
        "heldout_strict_clean_task_ids": sorted(HELDOUT_STRICT_CLEAN_TASK_IDS),
        "summary": summary,
        "json_artifact_inventory": inventory_json,
        "feature_artifact_inventory": feature_inventory,
        "candidate_index": {uid: candidates[uid] for uid in sorted(candidates)},
        "training_tasks": training_tasks,
        "training_pairs_primary": training_pairs_primary,
        "training_pairs_near_miss_only": training_pairs_near_miss_only,
        "training_pairs_runtime_diagnostic": training_pairs_runtime_diagnostic,
        "eval_sets": eval_sets,
        "required_feature_candidate_uids": required_uids,
        "missing_feature_candidates": missing_rows,
        "blockers": blockers,
        "notes": [
            "Training excludes every candidate from the six held-out strict-clean task IDs.",
            "Primary training pairs use correct-over-near_miss and correct-over-wrong_code only.",
            "Runtime-error candidates, if present, are only included in the separate diagnostic variant.",
            "Candidate labels come only from unit-test outcomes; tap/evaluator scores are not labels.",
        ],
        "outputs": {"json": rel(out_json), "md": rel(out_md)},
    }
    write_json(out_json, payload)
    write_md(out_md, payload)
    print(f"CODE_SPECIFIC_SPLIT_VERDICT = {verdict}")
    print(f"training_tasks = {len(training_tasks)}")
    print(f"primary_training_pairs = {len(training_pairs_primary)}")
    print(f"primary_strict_clean_tasks = {len(primary_eval)}")
    print(f"missing_feature_candidates = {len(missing)}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    if verdict == "BLOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
