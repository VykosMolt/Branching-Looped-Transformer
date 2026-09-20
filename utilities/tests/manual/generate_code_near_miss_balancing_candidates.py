"""Generate only missing-side candidates for the near-miss enrichment10 set."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import torch

from code_branch_pilot_lib import REPORT_DIR, configure_local_agent_env, import_agent_modules, repo_path, snippet, write_json
from evaluate_code_branch_candidates_v2 import eval_candidate, label_from_eval, legacy_label_from_eval
from generate_code_branch_candidates_v2 import (
    action_code_candidate,
    code_shape_error,
    direct_code_candidate,
    hard_code_final_candidate,
    hash_fields,
    repair_or_failed_candidate,
)


TASKSET_JSON = REPORT_DIR / "code_branch_taskset_v2_near_miss10_2026-05-17.json"
OLD_CANDIDATES_JSON = REPORT_DIR / "code_branch_candidates_v2_near_miss10_2026-05-17.json"
OLD_TOURNAMENTS_JSON = REPORT_DIR / "code_branch_tournaments_v2_near_miss10_2026-05-17.json"
INSPECTION_JSON = REPORT_DIR / "code_branch_near_miss_balance_inspection_2026-05-17.json"
OUTPUT_JSON = REPORT_DIR / "code_branch_near_miss_balancing_candidates_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "code_branch_near_miss_balancing_candidates_2026-05-17.md"
OUTPUT_LOG = REPORT_DIR / "code_branch_near_miss_balancing_candidates_2026-05-17.log"

CUDA_AVAILABLE_AT_IMPORT = torch.cuda.is_available()


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-candidates-total", type=int, default=30)
    parser.add_argument("--max-new-candidates-per-task", type=int, default=4)
    parser.add_argument("--target-strict-clean", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=str(OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    parser.add_argument("--log-file", default=str(OUTPUT_LOG))
    parser.add_argument("--direct-max-tokens", type=int, default=512)
    parser.add_argument("--direct-short-tokens", type=int, default=192)
    parser.add_argument("--first-action-max-tokens", type=int, default=768)
    parser.add_argument("--repair-max-tokens", type=int, default=512)
    return parser.parse_args()


@contextlib.contextmanager
def tee_log(path: str):
    if not path or os.environ.get("CODE_BRANCH_BALANCE_LOG_TEE_ACTIVE") == "1":
        yield
        return
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    os.environ["CODE_BRANCH_BALANCE_LOG_TEE_ACTIVE"] = "1"
    sys.stdout = Tee(old_stdout, log)  # type: ignore[assignment]
    sys.stderr = Tee(old_stderr, log)  # type: ignore[assignment]
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log.close()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_key(candidate: dict[str, Any]) -> str:
    return str(candidate.get("ast_hash") or candidate.get("normalized_code_hash") or candidate.get("raw_code_hash") or "")


def with_old_metadata(candidate: dict[str, Any]) -> dict[str, Any]:
    out = dict(candidate)
    task_id = str(out.get("task_id", ""))
    idx = out.get("candidate_index", 0)
    key = candidate_key(out)
    out.setdefault("candidate_uid", f"old_enrichment10::{task_id}::{idx}::{key}")
    out["generation_source"] = "old_enrichment10"
    out.setdefault("target_bucket", "")
    out.setdefault("target_subbucket", "")
    return out


def tests_for(task: dict[str, Any]) -> list[str]:
    return list(task.get("tests") or list(task.get("public_tests", [])) + list(task.get("hidden_tests", [])))


def evaluate_generated(candidate: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    result = eval_candidate(
        candidate.get("final_code", ""),
        tests_for(task),
        str(task.get("function_name", "")),
        float(task.get("timeout_seconds", 5.0)),
    )
    label = label_from_eval(result, candidate.get("final_code", ""), str(task.get("function_name", "")))
    return {
        **candidate,
        **result,
        "unit_test_label": label,
        "legacy_unit_test_label": legacy_label_from_eval(result),
        "is_malformed": label == "malformed",
        "is_code_like_wrong": label in {"wrong_code", "runtime_error", "near_miss"},
        "is_correct": label == "correct",
        "is_runnable": bool(result.get("safety_ok") and result.get("syntax_ok") and result.get("import_ok") and result.get("runtime_ok")),
    }


def strict_clean_count(rows: list[dict[str, Any]], task_ids: list[str]) -> int:
    total = 0
    for task_id in task_ids:
        task_rows = [row for row in rows if row.get("task_id") == task_id]
        has_correct = any(row.get("unit_test_label") == "correct" for row in task_rows)
        has_near = any(row.get("unit_test_label") == "near_miss" for row in task_rows)
        runnable = sum(1 for row in task_rows if row.get("is_runnable"))
        if has_correct and has_near and runnable >= 2:
            total += 1
    return total


def generation_plan(
    row: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[int, list[str]]:
    bucket = row["bucket"]
    subbucket = row.get("subbucket", "")
    cap = min(int(args.max_new_candidates_per_task), 4)
    if bucket == "needs_correct_anchor":
        return min(cap, 3), ["repaired_final", "direct_deterministic_final", "direct_sampled_low", "hard_code_final"]
    if bucket == "needs_near_miss" and subbucket == "correct_plus_wrong_code_no_near_miss":
        return cap, ["first_tool_code", "direct_short_budget", "direct_sampled_high", "first_failed_or_first_repair_code"]
    if bucket == "needs_near_miss":
        return cap, ["direct_short_budget", "first_tool_code", "first_failed_or_first_repair_code", "direct_sampled_high"]
    if bucket == "all_wrong_collapse":
        return min(cap, 2), ["repaired_final", "direct_deterministic_final"]
    if bucket == "all_correct_collapse":
        return min(cap, 3), ["direct_short_budget", "first_tool_code", "direct_sampled_high"]
    return min(cap, 2), ["direct_short_budget", "direct_sampled_high"]


def make_mode_fn(
    mode: str,
    model_mgr: Any,
    modules: dict[str, Any],
    project: Any,
    history: list[str],
    classifier: Any,
    self_model: Any,
    args: argparse.Namespace,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    if mode == "repaired_final":
        return lambda task: hard_code_final_candidate(model_mgr, modules, project, history, classifier, self_model, task)
    if mode == "hard_code_final":
        def _hard(task: dict[str, Any]) -> dict[str, Any]:
            cand = hard_code_final_candidate(model_mgr, modules, project, history, classifier, self_model, task)
            cand["mode"] = "hard_code_final"
            cand["candidate_stage"] = "repaired_final"
            return cand
        return _hard
    if mode == "direct_deterministic_final":
        return lambda task: direct_code_candidate(
            model_mgr,
            modules,
            task,
            mode="direct_deterministic_final",
            stage="direct_final",
            temperature=0.05,
            max_tokens=args.direct_max_tokens,
        )
    if mode == "direct_sampled_low":
        return lambda task: direct_code_candidate(
            model_mgr,
            modules,
            task,
            mode="direct_sampled_low",
            stage="direct_final",
            temperature=0.25,
            max_tokens=args.direct_max_tokens,
        )
    if mode == "direct_sampled_high":
        return lambda task: direct_code_candidate(
            model_mgr,
            modules,
            task,
            mode="direct_sampled_high",
            stage="direct_final",
            temperature=1.05,
            max_tokens=args.direct_max_tokens,
        )
    if mode == "direct_short_budget":
        return lambda task: direct_code_candidate(
            model_mgr,
            modules,
            task,
            mode="direct_short_budget",
            stage="direct_short_budget",
            temperature=0.85,
            max_tokens=args.direct_short_tokens,
        )
    if mode == "first_tool_code":
        return lambda task: action_code_candidate(
            model_mgr,
            modules,
            task,
            mode="first_tool_code",
            stage="first_tool_code",
            temperature=0.65,
            max_tokens=args.first_action_max_tokens,
            extra_directive="Prefer a compact first attempt. Do not over-repair edge cases.",
        )
    if mode == "first_failed_or_first_repair_code":
        return lambda task: repair_or_failed_candidate(
            model_mgr,
            modules,
            task,
            first_tokens=args.first_action_max_tokens,
            repair_tokens=args.repair_max_tokens,
        )
    raise KeyError(mode)


def admit_candidate(
    *,
    candidate: dict[str, Any],
    task: dict[str, Any],
    seen: dict[str, str],
    candidate_index: int,
    target_bucket: str,
    target_subbucket: str,
    duplicates: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, Any] | None:
    raw_code = candidate.get("sanitized_code") or candidate.get("raw_tool_input") or candidate.get("raw_model_text") or ""
    final_code = candidate.get("final_code") or ""
    candidate.update(hash_fields(raw_code, final_code))
    candidate.update({
        "task_id": task["task_id"],
        "source": task["source"],
        "difficulty": task.get("difficulty", "unknown"),
        "function_name": task["function_name"],
        "candidate_index": candidate_index,
        "generation_source": "balancing_pass",
        "target_bucket": target_bucket,
        "target_subbucket": target_subbucket,
    })
    key = candidate_key(candidate)
    candidate["candidate_uid"] = f"balancing_pass::{task['task_id']}::{candidate_index}::{key or int(time.time() * 1000)}"
    shape_error = code_shape_error(final_code)
    if shape_error:
        errors.append({
            "task_id": task["task_id"],
            "mode": candidate.get("mode"),
            "candidate_stage": candidate.get("candidate_stage"),
            "target_bucket": target_bucket,
            "error": candidate.get("generation_error") or shape_error,
            "raw_snippet": snippet(candidate.get("raw_model_text", ""), 300),
            "final_code_snippet": snippet(final_code, 300),
        })
        return None
    if key and key in seen:
        dup = dict(candidate)
        dup["duplicate_of"] = seen[key]
        duplicates.append(dup)
        return None
    if key:
        seen[key] = candidate["candidate_uid"]
    candidate["duplicate_of"] = None
    return candidate


def write_md(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# Code Branch Near-Miss Balancing Candidates",
        "",
        f"BALANCING_GENERATION_VERDICT = {payload['balancing_generation_verdict']}",
        "",
        f"- before_strict_clean: `{s['before_strict_clean']}`",
        f"- provisional_after_strict_clean: `{s['provisional_after_strict_clean']}`",
        f"- old_unique_candidates: `{s['old_unique_candidates']}`",
        f"- new_unique_candidates: `{s['new_unique_candidates']}`",
        f"- new_duplicate_candidates: `{s['new_duplicate_candidates']}`",
        f"- generation_errors: `{s['generation_errors']}`",
        f"- new_label_counts: `{s['new_label_counts']}`",
        f"- modes_attempted: `{s['modes_attempted']}`",
        "",
        "## Task Attempts",
        "",
        "| task_id | bucket | modes | added | duplicates | errors | labels | strict_after |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in payload["task_attempts"]:
        lines.append(
            f"| `{row['task_id']}` | `{row['bucket']}` | `{row['modes_attempted']}` | "
            f"{row['new_unique']} | {row['duplicates']} | {row['errors']} | `{row['new_labels']}` | {row['strict_clean_after']} |"
        )
    lines.extend(["", "## New Candidate Preview", ""])
    for cand in payload["new_candidates"][:30]:
        lines.append(
            f"- `{cand['task_id']}` `{cand['mode']}` label=`{cand.get('provisional_unit_test_label', 'NA')}` "
            f"hash=`{cand.get('ast_hash') or cand.get('normalized_code_hash')}` code={snippet(cand.get('final_code', ''), 120)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    with tee_log(args.log_file):
        _main(args)


def _main(args: argparse.Namespace) -> None:
    configure_local_agent_env()
    if args.device == "cuda" and not CUDA_AVAILABLE_AT_IMPORT:
        print("[balance-gen] warning: torch.cuda.is_available() false at import; continuing for backend CUDA check", flush=True)
    taskset = load(TASKSET_JSON)
    old_payload = load(OLD_CANDIDATES_JSON)
    old_tournaments = load(OLD_TOURNAMENTS_JSON)
    inspection = load(INSPECTION_JSON)
    if inspection.get("balance_inspection_verdict") == "BLOCKED":
        raise SystemExit("BALANCE_INSPECTION_VERDICT=BLOCKED")

    tasks = {str(task["task_id"]): task for task in taskset.get("tasks", [])}
    task_order = [str(task["task_id"]) for task in taskset.get("tasks", [])]
    old_candidates = [with_old_metadata(c) for c in old_payload.get("candidates", [])]
    old_eval_rows = [dict(row) for row in old_tournaments.get("candidate_evaluations", [])]
    current_rows = list(old_eval_rows)
    before_strict = strict_clean_count(current_rows, task_order)
    combined_candidates = list(old_candidates)
    new_candidates: list[dict[str, Any]] = []
    new_eval_rows: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    task_attempts: list[dict[str, Any]] = []
    modes_attempted: Counter[str] = Counter()

    seen_by_task: dict[str, dict[str, str]] = {}
    next_index: dict[str, int] = {}
    for task_id in task_order:
        seen_by_task[task_id] = {}
        max_idx = -1
        for cand in old_candidates:
            if cand.get("task_id") != task_id:
                continue
            key = candidate_key(cand)
            if key:
                seen_by_task[task_id][key] = str(cand.get("candidate_uid"))
            try:
                max_idx = max(max_idx, int(cand.get("candidate_index", -1)))
            except Exception:
                pass
        next_index[task_id] = max_idx + 1

    modules = import_agent_modules()
    agent = modules["agent"]
    model_mgr, _, classifier, project, self_model, history = agent.create_agent_runtime()
    try:
        for target in inspection.get("ranked_targets", []):
            if strict_clean_count(current_rows, task_order) >= int(args.target_strict_clean):
                break
            task_id = str(target["task_id"])
            task = tasks.get(task_id)
            if not task or target.get("bucket") == "already_strict_clean":
                continue
            cap, modes = generation_plan(target, args)
            print(f"[balance-gen] task {task_id} bucket={target['bucket']} modes={modes[:cap]}", flush=True)
            before_new = len(new_candidates)
            before_dup = len(duplicates)
            before_err = len(errors)
            before_eval = len(new_eval_rows)
            for mode in modes:
                if len(new_candidates) >= int(args.max_new_candidates_total):
                    break
                if len(new_candidates) - before_new >= cap:
                    break
                modes_attempted[mode] += 1
                try:
                    fn = make_mode_fn(mode, model_mgr, modules, project, history, classifier, self_model, args)
                    cand = fn(task)
                    admitted = admit_candidate(
                        candidate=cand,
                        task=task,
                        seen=seen_by_task[task_id],
                        candidate_index=next_index[task_id],
                        target_bucket=str(target["bucket"]),
                        target_subbucket=str(target.get("subbucket", "")),
                        duplicates=duplicates,
                        errors=errors,
                    )
                except Exception as exc:
                    errors.append({
                        "task_id": task_id,
                        "mode": mode,
                        "target_bucket": target.get("bucket"),
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    print(f"[balance-gen] error {task_id} {mode}: {type(exc).__name__}: {exc}", flush=True)
                    continue
                if admitted is None:
                    continue
                next_index[task_id] += 1
                eval_row = evaluate_generated(admitted, task)
                admitted["provisional_unit_test_label"] = eval_row["unit_test_label"]
                admitted["provisional_tests_passed"] = eval_row.get("tests_passed")
                admitted["provisional_tests_total"] = eval_row.get("tests_total")
                combined_candidates.append(admitted)
                new_candidates.append(admitted)
                new_eval_rows.append(eval_row)
                current_rows.append(eval_row)
                if strict_clean_count(current_rows, task_order) >= int(args.target_strict_clean):
                    break
            task_eval_rows = [row for row in current_rows if row.get("task_id") == task_id]
            task_attempts.append({
                "task_id": task_id,
                "bucket": target.get("bucket"),
                "subbucket": target.get("subbucket"),
                "modes_attempted": modes[:cap],
                "new_unique": len(new_candidates) - before_new,
                "duplicates": len(duplicates) - before_dup,
                "errors": len(errors) - before_err,
                "new_labels": dict(Counter(row["unit_test_label"] for row in new_eval_rows[before_eval:])),
                "strict_clean_after": any(row.get("unit_test_label") == "correct" for row in task_eval_rows)
                and any(row.get("unit_test_label") == "near_miss" for row in task_eval_rows),
            })
            if len(new_candidates) >= int(args.max_new_candidates_total):
                break
    finally:
        if hasattr(model_mgr, "unload"):
            with contextlib.suppress(Exception):
                model_mgr.unload()

    after_strict = strict_clean_count(current_rows, task_order)
    if not new_candidates and len(errors) > len(duplicates):
        verdict = "WRAPPER_BLOCKED"
    elif after_strict >= int(args.target_strict_clean):
        verdict = "EARLY_GREEN"
    elif not new_candidates:
        verdict = "TOO_FEW_NEW_CANDIDATES"
    else:
        verdict = "COMPLETED"
    summary = {
        "before_strict_clean": before_strict,
        "provisional_after_strict_clean": after_strict,
        "old_unique_candidates": len(old_candidates),
        "new_unique_candidates": len(new_candidates),
        "new_duplicate_candidates": len(duplicates),
        "generation_errors": len(errors),
        "total_unique_candidates": len(combined_candidates),
        "new_label_counts": dict(Counter(row["unit_test_label"] for row in new_eval_rows)),
        "modes_attempted": dict(modes_attempted),
        "hard_cap_total_new_candidates": int(args.max_new_candidates_total),
    }
    by_task: dict[str, dict[str, Any]] = {}
    for task_id in task_order:
        by_task[task_id] = {
            "old_unique": sum(1 for c in old_candidates if c.get("task_id") == task_id),
            "new_unique": sum(1 for c in new_candidates if c.get("task_id") == task_id),
            "new_duplicates": sum(1 for c in duplicates if c.get("task_id") == task_id),
            "errors": sum(1 for e in errors if e.get("task_id") == task_id),
        }
    payload = {
        "balancing_generation_verdict": verdict,
        "inputs": {
            "taskset": repo_path(TASKSET_JSON),
            "old_candidates": repo_path(OLD_CANDIDATES_JSON),
            "old_tournaments": repo_path(OLD_TOURNAMENTS_JSON),
            "inspection": repo_path(INSPECTION_JSON),
        },
        "summary": summary,
        "by_task": by_task,
        "candidates": combined_candidates,
        "old_candidates": old_candidates,
        "new_candidates": new_candidates,
        "duplicate_candidates": duplicates,
        "generation_errors": errors,
        "new_candidate_evaluations_for_stopping": new_eval_rows,
        "task_attempts": task_attempts,
        "notes": "Generation used only the local Ouro-RLTT wrapper. Provisional labels were computed from unit tests for early stopping and are recomputed by evaluate_code_near_miss_balancing.py.",
        "outputs": {"json": repo_path(Path(args.output)), "md": repo_path(Path(args.output_md)), "log": repo_path(Path(args.log_file))},
    }
    out_json = Path(args.output)
    out_md = Path(args.output_md)
    write_json(out_json, payload)
    write_md(out_md, payload)
    print(f"BALANCING_GENERATION_VERDICT = {verdict}")
    print(f"provisional_strict_clean = {after_strict}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    if verdict == "WRAPPER_BLOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
