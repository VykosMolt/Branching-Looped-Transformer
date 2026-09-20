"""Bounded expansion screen for additional strict-clean code tasks.

This is intentionally a task-screening pass only. It generates three bounded
candidate attempts per new task, labels only with unit tests, and does not
capture features or evaluate taps on the newly screened tasks.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import torch

from build_code_branch_taskset import LOCAL_TASKS
from build_code_branch_taskset_v2 import DEVIL_TASKS, finalize_task, load_humaneval_tasks, load_mbpp_tasks
from code_branch_pilot_lib import PROJECT_ROOT, REPORT_DIR, configure_local_agent_env, import_agent_modules, repo_path, snippet, write_json
from evaluate_code_branch_candidates_v2 import (
    candidate_code_for_unit_tests,
    eval_candidate,
    label_from_eval,
    legacy_label_from_eval,
)
from generate_code_branch_candidates_v2 import (
    action_code_candidate,
    code_shape_error,
    direct_code_candidate,
    hash_fields,
)


OUTPUT_JSON = REPORT_DIR / "code_strict_clean_screening_expansion_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "code_strict_clean_screening_expansion_2026-05-17.md"
OUTPUT_LOG = REPORT_DIR / "code_strict_clean_screening_expansion_2026-05-17.log"
SUMMARY_JSON = REPORT_DIR / "code_specific_control_and_screening_2026-05-17_summary.json"
SUMMARY_MD = REPORT_DIR / "code_specific_control_and_screening_2026-05-17_summary.md"

SPLITS_JSON = REPORT_DIR / "code_specific_training_control_splits_2026-05-17.json"
FEATURES_PT = REPORT_DIR / "code_specific_training_features_2026-05-17.pt"
TINY_HEAD_JSON = REPORT_DIR / "code_specific_tiny_head_control_2026-05-17.json"

PRIOR_SCREENING_RESULTS = REPORT_DIR / "code_strict_clean_screening_results_2026-05-17.json"
PRIOR_SCREENING_SUMMARY = REPORT_DIR / "code_strict_clean_screening_2026-05-17_summary.json"
PRIOR_V2_MINI = REPORT_DIR / "code_branch_tournaments_v2_mini_patched_2026-05-16.json"
PRIOR_NEARMISS = REPORT_DIR / "code_branch_tournaments_v2_near_miss10_2026-05-17.json"
PRIOR_BALANCED = REPORT_DIR / "code_branch_near_miss_balanced_tournaments_2026-05-17.json"

HELDOUT_STRICT_CLEAN_TASK_IDS = {
    "mbpp/100",
    "mbpp/129",
    "mbpp/283",
    "mbpp/291",
    "mbpp/391",
    "mbpp/392",
}

DOCS_TO_APPEND = [
    PROJECT_ROOT / "docs/evaluator/evaluator_domain_transfer_notes.md",
    PROJECT_ROOT / "docs/evaluator/math_bg_gate_pilot_2026-05-15.md",
    PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-15_v4.md",
    PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-16_v5_clean_gsm8k_code_next.md",
    PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-17_v7_actual_state_and_next.md",
]

CUDA_AVAILABLE_AT_IMPORT = torch.cuda.is_available()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-additional-strict-clean", type=int, default=10)
    parser.add_argument("--target-new-tasks", type=int, default=40)
    parser.add_argument("--max-new-tasks", type=int, default=60)
    parser.add_argument("--output", default=str(OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    parser.add_argument("--log-file", default=str(OUTPUT_LOG))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--source-priority", default="local_dsa,mbpp,humaneval")
    parser.add_argument("--direct-max-tokens", type=int, default=512)
    parser.add_argument("--direct-short-tokens", type=int, default=192)
    parser.add_argument("--first-action-max-tokens", type=int, default=640)
    return parser.parse_args()


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


@contextlib.contextmanager
def tee_log(path: str):
    if not path or os.environ.get("CODE_STRICT_EXPANSION_LOG_TEE_ACTIVE") == "1":
        yield
        return
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    os.environ["CODE_STRICT_EXPANSION_LOG_TEE_ACTIVE"] = "1"
    sys.stdout = Tee(old_stdout, log)  # type: ignore[assignment]
    sys.stderr = Tee(old_stderr, log)  # type: ignore[assignment]
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log.close()


def load_json_if_present(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def tests_for(task: dict[str, Any]) -> list[str]:
    tests = list(task.get("tests") or [])
    if not tests:
        tests = list(task.get("public_tests", [])) + list(task.get("hidden_tests", []))
    return [str(test).strip() for test in tests if str(test).strip()]


def signature_for(task: dict[str, Any]) -> str:
    if task.get("signature_hint"):
        return str(task["signature_hint"])
    starter = str(task.get("starter_code", "")).strip()
    for line in starter.splitlines():
        line = line.strip()
        if line.startswith("def "):
            return line.rstrip(":")
    fn = str(task.get("function_name", "candidate"))
    return f"def {fn}(...)"


def outcome_from_labels(labels: Counter[str]) -> str:
    usable = labels.get("correct", 0) + labels.get("near_miss", 0) + labels.get("wrong_code", 0) + labels.get("runtime_error", 0)
    if labels.get("correct", 0) and labels.get("near_miss", 0):
        return "strict_clean_ready"
    if usable and labels.get("correct", 0) == usable:
        return "all_correct"
    if labels.get("correct", 0):
        return "anchor_only"
    if labels.get("near_miss", 0):
        return "near_miss_only"
    if labels.get("wrong_code", 0) or labels.get("runtime_error", 0):
        return "all_wrong"
    return "malformed_only"


def load_prior_history() -> dict[str, dict[str, Any]]:
    prior: dict[str, dict[str, Any]] = {}
    screening = load_json_if_present(PRIOR_SCREENING_RESULTS)
    for row in screening.get("task_rows", []) or []:
        task_id = str(row.get("task_id") or "")
        if not task_id:
            continue
        prior[task_id] = {
            "prior_outcome": row.get("classification", "unknown"),
            "prior_label_counts": row.get("label_counts", {}),
            "prior_source": repo_path(PRIOR_SCREENING_RESULTS),
        }
    for path in (PRIOR_V2_MINI, PRIOR_NEARMISS, PRIOR_BALANCED):
        payload = load_json_if_present(path)
        for tournament in payload.get("tournaments", []) or []:
            task_id = str(tournament.get("task_id") or "")
            if not task_id or task_id in prior:
                continue
            rows = tournament.get("diagnostic_candidates") or tournament.get("diagnostic_runnable_candidates") or []
            labels = Counter(str(row.get("unit_test_label", "unknown")) for row in rows)
            if labels:
                prior[task_id] = {
                    "prior_outcome": outcome_from_labels(labels),
                    "prior_label_counts": dict(labels),
                    "prior_source": repo_path(path),
                }
    for task_id in HELDOUT_STRICT_CLEAN_TASK_IDS:
        prior.setdefault(task_id, {
            "prior_outcome": "heldout_strict_clean_eval",
            "prior_label_counts": {},
            "prior_source": "heldout_eval_set",
        })
    expansion = load_json_if_present(OUTPUT_JSON)
    for row in expansion.get("task_rows", []) or []:
        task_id = str(row.get("task_id") or "")
        if not task_id:
            continue
        prior[task_id] = {
            "prior_outcome": row.get("classification", "screened_in_expansion"),
            "prior_label_counts": row.get("label_counts", {}),
            "prior_source": repo_path(OUTPUT_JSON),
        }
    return prior


def add_screening_metadata(task: dict[str, Any], prior: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = finalize_task(dict(task))
    tests = tests_for(out)
    task_id = str(out["task_id"])
    prior_row = prior.get(task_id, {})
    out["tests"] = tests
    out["number_of_tests"] = len(tests)
    out["signature"] = signature_for(out)
    out["prior_outcome"] = prior_row.get("prior_outcome", "unknown")
    out["prior_label_counts"] = prior_row.get("prior_label_counts", {})
    out["prior_source"] = prior_row.get("prior_source", "")
    out.setdefault("tests_visibility", "public_and_hidden")
    return out


def task_priority(task: dict[str, Any], source_priority: list[str]) -> tuple[int, int, int, str]:
    source = str(task.get("source", "unknown"))
    difficulty = str(task.get("difficulty", "unknown"))
    n_tests = int(task.get("number_of_tests", 0))
    source_rank = {name: idx for idx, name in enumerate(source_priority)}.get(source, len(source_priority) + 1)
    difficulty_rank = {"hard": 0, "medium": 1, "unknown": 2, "easy": 4, "devil": 5}.get(difficulty, 3)
    granular_rank = 0 if n_tests >= 3 else 1
    return (source_rank, difficulty_rank, granular_rank, str(task.get("task_id", "")))


def granularize_humaneval_task(task: dict[str, Any]) -> dict[str, Any]:
    if task.get("source") != "humaneval":
        return task
    hidden = list(task.get("hidden_tests", []) or [])
    if not hidden:
        return task
    source = str(hidden[0])
    fn = str(task.get("function_name", ""))
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return task
    check_fn = None
    prefix_nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "check":
            check_fn = node
        elif not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "check"
        ):
            prefix_nodes.append(node)
    if check_fn is None:
        return task
    prefix = "\n".join(ast.unparse(node) for node in prefix_nodes)
    tests: list[str] = []
    for stmt in check_fn.body:
        if isinstance(stmt, ast.Assert):
            tests.append((prefix + "\n" if prefix else "") + f"candidate = {fn}\n" + ast.unparse(stmt))
        elif isinstance(stmt, (ast.For, ast.While, ast.If)):
            tests.append((prefix + "\n" if prefix else "") + f"candidate = {fn}\n" + ast.unparse(stmt))
    if len(tests) >= 2:
        out = dict(task)
        out["hidden_tests"] = tests
        out["tests"] = list(out.get("public_tests", [])) + tests
        out["number_of_tests"] = len(out["tests"])
        out["tests_visibility"] = "official_hidden_granularized"
        out["humaneval_granularized"] = True
        return out
    return task


def build_task_pool(max_new_tasks: int, source_priority: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prior = load_prior_history()
    observed = set(prior) | set(HELDOUT_STRICT_CLEAN_TASK_IDS)
    pool: list[dict[str, Any]] = []

    local = [add_screening_metadata({**task, "tests_visibility": "local_public_and_hidden"}, prior) for task in LOCAL_TASKS]
    local.extend(add_screening_metadata(task, prior) for task in DEVIL_TASKS)
    pool.extend(local)
    existing_names = {task["function_name"] for task in pool}

    mbpp, mbpp_error = load_mbpp_tasks(max(300, max_new_tasks * 8), existing_names, set())
    mbpp = [add_screening_metadata(task, prior) for task in mbpp]
    pool.extend(mbpp)

    human, human_error = load_humaneval_tasks(max(40, max_new_tasks))
    human = [granularize_humaneval_task(add_screening_metadata(task, prior)) for task in human]
    pool.extend(human)

    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_functions: set[str] = set()
    for task in sorted(pool, key=lambda task: task_priority(task, source_priority)):
        task_id = str(task.get("task_id", ""))
        fn = str(task.get("function_name", ""))
        if not task_id or task_id in observed or task_id in seen_ids:
            continue
        if fn and fn in seen_functions:
            continue
        if int(task.get("number_of_tests", 0)) < 2:
            continue
        if str(task.get("difficulty")) == "devil":
            continue
        selected.append(task)
        seen_ids.add(task_id)
        if fn:
            seen_functions.add(fn)
        if len(selected) >= max_new_tasks:
            break

    meta = {
        "prior_task_count": len(prior),
        "heldout_excluded": sorted(HELDOUT_STRICT_CLEAN_TASK_IDS),
        "source_mix_available": dict(Counter(task.get("source", "unknown") for task in pool)),
        "source_mix_selected": dict(Counter(task.get("source", "unknown") for task in selected)),
        "difficulty_mix_selected": dict(Counter(task.get("difficulty", "unknown") for task in selected)),
        "source_priority": source_priority,
        "dataset_status": {
            "mbpp": "loaded" if mbpp else f"unavailable: {mbpp_error}",
            "humaneval": "loaded" if human else f"unavailable: {human_error}",
            "local_dsa": len(local),
        },
    }
    return selected, meta


def load_existing_expansion(path: str | Path) -> dict[str, Any]:
    payload = load_json_if_present(Path(path))
    if not payload or payload.get("wrapper_blocked"):
        return {}
    if "candidate_evaluations" not in payload:
        return {}
    return payload


def make_role_fn(role: str, model_mgr: Any, modules: dict[str, Any], args: argparse.Namespace) -> Callable[[dict[str, Any]], dict[str, Any]]:
    if role == "strong_anchor":
        return lambda task: direct_code_candidate(
            model_mgr,
            modules,
            task,
            mode="direct_deterministic_final",
            stage="direct_final",
            temperature=0.05,
            max_tokens=args.direct_max_tokens,
        )
    if role == "near_miss_probe_1":
        return lambda task: action_code_candidate(
            model_mgr,
            modules,
            task,
            mode="first_tool_code",
            stage="first_tool_code",
            temperature=0.7,
            max_tokens=args.first_action_max_tokens,
            extra_directive=(
                "Prefer a compact first attempt. Do not spend tokens on explanations. "
                "Do not over-repair edge cases before returning the Python action."
            ),
        )
    if role == "near_miss_probe_2":
        return lambda task: direct_code_candidate(
            model_mgr,
            modules,
            task,
            mode="direct_sampled_high",
            stage="direct_final",
            temperature=1.05,
            max_tokens=args.direct_max_tokens,
        )
    raise KeyError(role)


def candidate_key(candidate: dict[str, Any]) -> str:
    return str(candidate.get("ast_hash") or candidate.get("normalized_code_hash") or candidate.get("raw_code_hash") or "")


def admit_candidate(
    *,
    candidate: dict[str, Any],
    task: dict[str, Any],
    role: str,
    seen: dict[str, str],
    candidate_index: int,
) -> dict[str, Any]:
    raw_code = candidate.get("sanitized_code") or candidate.get("raw_tool_input") or candidate.get("raw_model_text") or ""
    final_code = candidate.get("final_code") or ""
    candidate.update(hash_fields(raw_code, final_code))
    key = candidate_key(candidate)
    uid_tail = key or str(int(time.time() * 1000))
    uid = f"screening_expansion::{task['task_id']}::{role}::{candidate_index}::{uid_tail}"
    shape_error = code_shape_error(final_code)
    duplicate_of = seen.get(key) if key else None
    if key and not duplicate_of and not shape_error:
        seen[key] = uid
    candidate.update({
        "task_id": task["task_id"],
        "source": task.get("source", "unknown"),
        "difficulty": task.get("difficulty", "unknown"),
        "function_name": task.get("function_name", ""),
        "candidate_uid": uid,
        "candidate_index": candidate_index,
        "screening_role": role,
        "generation_source": "strict_clean_screening_expansion",
        "duplicate_of": duplicate_of,
        "admitted_primary": not bool(duplicate_of),
        "admission_error": shape_error,
    })
    if shape_error and not candidate.get("generation_error"):
        candidate["generation_error"] = shape_error
    return candidate


def evaluate_candidate(candidate: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    code, recovered = candidate_code_for_unit_tests(candidate, task)
    result = eval_candidate(
        code,
        tests_for(task),
        str(task.get("function_name", "")),
        float(task.get("timeout_seconds", 5.0)),
    )
    label = label_from_eval(result, code, str(task.get("function_name", "")))
    return {
        **candidate,
        **result,
        "unit_test_label": label,
        "legacy_unit_test_label": legacy_label_from_eval(result),
        "is_correct": label == "correct",
        "is_runnable": bool(result.get("safety_ok") and result.get("syntax_ok") and result.get("import_ok") and result.get("runtime_ok")),
        "unit_test_code_recovered": recovered,
        "final_code_for_unit_tests": code,
    }


def is_primary(row: dict[str, Any]) -> bool:
    return not bool(row.get("duplicate_of"))


def classify_task(rows: list[dict[str, Any]]) -> str:
    labels = Counter(str(row.get("unit_test_label", "unknown")) for row in rows if is_primary(row))
    usable = labels.get("correct", 0) + labels.get("near_miss", 0) + labels.get("wrong_code", 0) + labels.get("runtime_error", 0)
    if labels.get("correct", 0) and labels.get("near_miss", 0):
        return "strict_clean_ready"
    if usable and labels.get("correct", 0) == usable:
        return "all_correct"
    if labels.get("correct", 0):
        return "anchor_only"
    if labels.get("near_miss", 0):
        return "near_miss_only"
    if labels.get("wrong_code", 0) or labels.get("runtime_error", 0):
        return "all_wrong"
    return "malformed_only"


def strict_clean_ready_count(eval_rows: list[dict[str, Any]]) -> int:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eval_rows:
        if is_primary(row):
            by_task[str(row.get("task_id", ""))].append(row)
    total = 0
    for rows in by_task.values():
        labels = {row.get("unit_test_label") for row in rows}
        total += int("correct" in labels and "near_miss" in labels)
    return total


def role_success(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for role in sorted({str(row.get("screening_role", "unknown")) for row in rows}):
        role_rows = [row for row in rows if row.get("screening_role") == role and is_primary(row)]
        out[role] = {
            "n": len(role_rows),
            "label_counts": dict(Counter(row.get("unit_test_label", "unknown") for row in role_rows)),
            "correct_rate": sum(1 for row in role_rows if row.get("unit_test_label") == "correct") / max(len(role_rows), 1),
            "near_miss_rate": sum(1 for row in role_rows if row.get("unit_test_label") == "near_miss") / max(len(role_rows), 1),
        }
    return out


def verdict_for(strict_count: int) -> str:
    if strict_count >= 10:
        return "GREEN"
    if strict_count >= 5:
        return "YELLOW"
    return "RED"


def recommended_next(tiny_verdict: str, expansion_verdict: str) -> str:
    if tiny_verdict == "GOOD" and expansion_verdict in {"GREEN", "YELLOW"}:
        return "evaluate_code_specific_and_hh_trained_taps_on_expanded_strict_clean_set"
    if tiny_verdict == "GOOD" and expansion_verdict == "RED":
        return "keep_code_specific_result_but_improve_task_sources"
    if tiny_verdict == "WEAK" and expansion_verdict in {"GREEN", "YELLOW"}:
        return "rerun_strict_clean_transfer_with_larger_eval_before_concluding"
    if tiny_verdict == "POOR":
        return "investigate_feature/pooling_or_richer_head_before_more_code_data"
    if tiny_verdict == "NOT_RUN":
        return "fix_artifact_or_feature_blocker"
    return "improve_task_sources_before_concluding"


def screen_tasks(args: argparse.Namespace) -> dict[str, Any]:
    existing = load_existing_expansion(args.output)
    previous_completed = list(dict.fromkeys(str(x) for x in existing.get("completed_task_ids", []) or []))
    remaining_budget = max(0, int(args.max_new_tasks) - len(previous_completed))
    source_priority = [item.strip() for item in str(args.source_priority).split(",") if item.strip()]
    selected_tasks, pool_meta = build_task_pool(remaining_budget, source_priority)
    if args.device == "cuda" and not CUDA_AVAILABLE_AT_IMPORT:
        print("[screen-expansion] warning: torch.cuda.is_available() false at import; continuing for backend CUDA check", flush=True)
    configure_local_agent_env()
    modules = import_agent_modules()
    agent = modules["agent"]
    model_mgr, _, _, _, _, _ = agent.create_agent_runtime()
    roles = ["strong_anchor", "near_miss_probe_1", "near_miss_probe_2"]
    candidates: list[dict[str, Any]] = list(existing.get("candidate_generation_records", []) or [])
    eval_rows: list[dict[str, Any]] = list(existing.get("candidate_evaluations", []) or [])
    generation_errors: list[dict[str, Any]] = list(existing.get("generation_errors", []) or [])
    completed_task_ids: list[str] = previous_completed[:]
    seen_by_task: dict[str, dict[str, str]] = defaultdict(dict)
    stop_reason = ""
    try:
        if strict_clean_ready_count(eval_rows) >= int(args.target_additional_strict_clean):
            stop_reason = "target_additional_strict_clean_already_reached"
        if remaining_budget <= 0 and not stop_reason:
            stop_reason = "max_total_screened_reached"
        for task_index, task in enumerate(selected_tasks[: int(args.max_new_tasks)], start=1):
            if stop_reason:
                break
            task_id = str(task["task_id"])
            print(f"[screen-expansion] task {task_index}/{len(selected_tasks)} {task_id}", flush=True)
            for role in roles:
                current_rows = [row for row in eval_rows if row.get("task_id") == task_id and is_primary(row)]
                if any(row.get("unit_test_label") == "correct" for row in current_rows) and any(
                    row.get("unit_test_label") == "near_miss" for row in current_rows
                ):
                    break
                try:
                    fn = make_role_fn(role, model_mgr, modules, args)
                    raw_candidate = fn(task)
                    candidate = admit_candidate(
                        candidate=raw_candidate,
                        task=task,
                        role=role,
                        seen=seen_by_task[task_id],
                        candidate_index=sum(1 for cand in candidates if cand.get("task_id") == task_id),
                    )
                    eval_row = evaluate_candidate(candidate, task)
                    candidates.append(candidate)
                    eval_rows.append(eval_row)
                except Exception as exc:
                    generation_errors.append({
                        "task_id": task_id,
                        "screening_role": role,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    print(f"[screen-expansion] error {task_id} {role}: {type(exc).__name__}: {exc}", flush=True)
            completed_task_ids.append(task_id)
            strict_ready = strict_clean_ready_count(eval_rows)
            print(f"[screen-expansion] completed={len(completed_task_ids)} strict_clean_ready={strict_ready}", flush=True)
            if strict_ready >= int(args.target_additional_strict_clean):
                stop_reason = "target_additional_strict_clean_reached"
                break
            if len(completed_task_ids) >= 30 and strict_ready == 0:
                stop_reason = "zero_strict_clean_ready_after_30_tasks"
                break
    finally:
        if hasattr(model_mgr, "unload"):
            try:
                model_mgr.unload()
            except Exception:
                pass
    if not stop_reason:
        stop_reason = "completed_task_cap_or_pool"

    task_rows: list[dict[str, Any]] = []
    by_task: dict[str, Any] = {}
    all_tasks = list(existing.get("tasks", []) or []) + selected_tasks
    seen_task_rows: set[str] = set()
    for task in all_tasks:
        task_id = str(task["task_id"])
        if task_id in seen_task_rows:
            continue
        seen_task_rows.add(task_id)
        rows = [row for row in eval_rows if row.get("task_id") == task_id]
        if not rows and task_id not in completed_task_ids:
            continue
        primary = [row for row in rows if is_primary(row)]
        labels = Counter(str(row.get("unit_test_label", "unknown")) for row in primary)
        classification = classify_task(rows)
        row = {
            "task_id": task_id,
            "source": task.get("source", "unknown"),
            "difficulty": task.get("difficulty", "unknown"),
            "function_name": task.get("function_name", ""),
            "prior_outcome": task.get("prior_outcome", "unknown"),
            "number_of_tests": task.get("number_of_tests", len(tests_for(task))),
            "n_candidates": len(rows),
            "n_primary_candidates": len(primary),
            "label_counts": dict(labels),
            "classification": classification,
            "strict_clean_ready": classification == "strict_clean_ready",
        }
        by_task[task_id] = row
        if rows:
            task_rows.append(row)

    strict_rows = [row for row in task_rows if row["classification"] == "strict_clean_ready"]
    classification_counts = Counter(row["classification"] for row in task_rows)
    primary_rows = [row for row in eval_rows if is_primary(row)]
    label_totals = dict(Counter(row.get("unit_test_label", "unknown") for row in primary_rows))
    verdict = verdict_for(len(strict_rows))
    summary = {
        "STRICT_CLEAN_SCREENING_EXPANSION_VERDICT": verdict,
        "target_additional_strict_clean": int(args.target_additional_strict_clean),
        "target_new_tasks": int(args.target_new_tasks),
        "max_new_tasks": int(args.max_new_tasks),
        "resumed_existing_tasks": len(previous_completed),
        "remaining_budget_started": remaining_budget,
        "tasks_selected": len(selected_tasks),
        "tasks_screened": len(task_rows),
        "total_screening_candidates": len(candidates),
        "primary_unique_candidates": len(primary_rows),
        "duplicate_candidates": sum(1 for row in eval_rows if row.get("duplicate_of")),
        "generation_errors": len(generation_errors),
        "strict_clean_ready": len(strict_rows),
        "anchor_only": classification_counts.get("anchor_only", 0),
        "near_miss_only": classification_counts.get("near_miss_only", 0),
        "all_correct": classification_counts.get("all_correct", 0),
        "all_wrong": classification_counts.get("all_wrong", 0),
        "malformed_only": classification_counts.get("malformed_only", 0),
        "label_totals": label_totals,
        "role_success": role_success(eval_rows),
        "per_source_classification": {
            source: dict(Counter(row["classification"] for row in task_rows if row["source"] == source))
            for source in sorted({row["source"] for row in task_rows})
        },
        "per_difficulty_classification": {
            diff: dict(Counter(row["classification"] for row in task_rows if row["difficulty"] == diff))
            for diff in sorted({row["difficulty"] for row in task_rows})
        },
        "strict_clean_ready_task_ids": [row["task_id"] for row in strict_rows],
        "stop_reason": stop_reason,
        "pool_meta": pool_meta,
    }
    payload = {
        "strict_clean_screening_expansion_verdict": verdict,
        "summary": summary,
        "tasks": all_tasks,
        "completed_task_ids": completed_task_ids,
        "candidate_evaluations": eval_rows,
        "candidate_generation_records": candidates,
        "generation_errors": generation_errors,
        "by_task": by_task,
        "task_rows": task_rows,
        "strict_clean_ready_tasks": strict_rows,
        "notes": [
            "Candidate labels come only from unit tests.",
            "Tap/evaluator scores are not used as labels.",
            "No feature capture or transfer evaluation is run on expansion tasks in this prompt.",
        ],
        "outputs": {"json": repo_path(Path(args.output)), "md": repo_path(Path(args.output_md)), "log": repo_path(Path(args.log_file))},
    }
    if not eval_rows and generation_errors:
        payload["strict_clean_screening_expansion_verdict"] = "BLOCKED"
        payload["wrapper_blocked"] = True
        payload["blocker"] = "all_generation_attempts_failed_before unit-test labeling"
        payload["summary"]["STRICT_CLEAN_SCREENING_EXPANSION_VERDICT"] = "BLOCKED"
        payload["summary"]["wrapper_blocked"] = True
    return payload


def write_expansion_md(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# Strict-Clean Code Screening Expansion",
        "",
        f"STRICT_CLEAN_SCREENING_EXPANSION_VERDICT = {payload['strict_clean_screening_expansion_verdict']}",
        "",
        f"- stop_reason: `{s['stop_reason']}`",
        f"- tasks_screened: `{s['tasks_screened']}`",
        f"- strict_clean_ready: `{s['strict_clean_ready']}`",
        f"- anchor_only / near_miss_only / all_correct / all_wrong: `{s['anchor_only']} / {s['near_miss_only']} / {s['all_correct']} / {s['all_wrong']}`",
        f"- label_totals: `{s['label_totals']}`",
        f"- per_source: `{s['per_source_classification']}`",
        f"- per_difficulty: `{s['per_difficulty_classification']}`",
        f"- role_success: `{s['role_success']}`",
        "",
        "## Strict-Clean-Ready Task List",
        "",
    ]
    if s["strict_clean_ready_task_ids"]:
        for row in payload["strict_clean_ready_tasks"]:
            lines.append(f"- `{row['task_id']}` `{row['source']}` `{row['difficulty']}` labels=`{row['label_counts']}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Task Classification", "", "| task_id | source | difficulty | classification | labels |", "| --- | --- | --- | --- | --- |"])
    for row in payload["task_rows"]:
        lines.append(
            f"| `{row['task_id']}` | `{row['source']}` | `{row['difficulty']}` | `{row['classification']}` | `{row['label_counts']}` |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def load_feature_verdict() -> str:
    if not FEATURES_PT.exists():
        return "BLOCKED"
    try:
        payload = torch.load(FEATURES_PT, map_location="cpu", weights_only=False)
        return str(payload.get("meta", {}).get("code_specific_feature_verdict", "BLOCKED"))
    except Exception:
        return "BLOCKED"


def load_result_context(expansion_payload: dict[str, Any]) -> dict[str, Any]:
    splits = load_json_if_present(SPLITS_JSON)
    tiny = load_json_if_present(TINY_HEAD_JSON)
    split_verdict = str(splits.get("code_specific_split_verdict", "BLOCKED"))
    feature_verdict = load_feature_verdict()
    tiny_verdict = str(tiny.get("code_specific_tiny_head_verdict", "NOT_RUN"))
    tiny_summary = tiny.get("summary", {})
    expansion_summary = expansion_payload["summary"]
    expansion_verdict = str(expansion_payload["strict_clean_screening_expansion_verdict"])
    return {
        "splits": splits,
        "tiny": tiny,
        "split_verdict": split_verdict,
        "feature_verdict": feature_verdict,
        "tiny_verdict": tiny_verdict,
        "tiny_summary": tiny_summary,
        "expansion_verdict": expansion_verdict,
        "expansion_summary": expansion_summary,
        "recommended_next": recommended_next(tiny_verdict, expansion_verdict),
    }


def append_docs(context: dict[str, Any]) -> list[str]:
    title = "## Code-specific tiny-head control + strict-clean screening expansion (2026-05-17)"
    ts = context["tiny_summary"]
    es = context["expansion_summary"]
    rerun_note = ""
    text = "\n".join([
        "",
        title,
        "",
        rerun_note,
        f"- CODE_SPECIFIC_SPLIT_VERDICT: `{context['split_verdict']}`",
        f"- CODE_SPECIFIC_FEATURE_VERDICT: `{context['feature_verdict']}`",
        f"- CODE_SPECIFIC_TINY_HEAD_VERDICT: `{context['tiny_verdict']}`",
        f"- STRICT_CLEAN_SCREENING_EXPANSION_VERDICT: `{context['expansion_verdict']}`",
        f"- training task count / pair count: `{ts.get('training_task_count')}` / `{ts.get('primary_training_pair_count')}`",
        f"- held-out strict-clean task count: `{ts.get('heldout_strict_clean_task_count')}`",
        f"- best code-trained AntisymLinear row: `{ts.get('best_code_trained_antisymlinear', 'NA')}`",
        f"- best code-trained NoNorm row: `{ts.get('best_code_trained_nonorm', 'NA')}`",
        "- comparison to HH-trained strict-clean WEAK result: `47_concat_all_loops / AntisymLinearNoNorm`, top1=0.500, pairwise=0.571, cycle=0.000.",
        f"- screening expansion counts: tasks_screened=`{es.get('tasks_screened')}`, strict_clean_ready=`{es.get('strict_clean_ready')}`, label_totals=`{es.get('label_totals')}`",
        f"- new strict_clean_ready task IDs: `{es.get('strict_clean_ready_task_ids', [])}`",
        f"- interpretation: {ts.get('one_sentence_interpretation', 'Code-specific control completed; screening expansion supplies the next eval-pool signal.')}",
        "",
    ])
    appended: list[str] = []
    for path in DOCS_TO_APPEND:
        if not path.exists():
            continue
        current = path.read_text(encoding="utf-8")
        final_text = text
        if title in current:
            final_text = text.replace(
                "\n\n- CODE_SPECIFIC_SPLIT_VERDICT",
                "\n\nNote: this appended block is the valid rerun after the wrapper-blocked shell-redirection attempt.\n\n- CODE_SPECIFIC_SPLIT_VERDICT",
                1,
            )
        path.write_text(current.rstrip() + "\n" + final_text, encoding="utf-8")
        appended.append(repo_path(path))
    return appended


def command_log() -> list[str]:
    return [
        "venv/bin/python -m py_compile utilities/tests/manual/build_code_specific_training_control_splits.py",
        "venv/bin/python -m py_compile utilities/tests/manual/ensure_code_specific_training_features.py",
        "venv/bin/python -m py_compile utilities/tests/manual/train_code_specific_tiny_heads_and_eval.py",
        "venv/bin/python -m py_compile utilities/tests/manual/screen_more_code_strict_clean_tasks.py",
        "venv/bin/python -u utilities/tests/manual/build_code_specific_training_control_splits.py",
        "venv/bin/python -u utilities/tests/manual/ensure_code_specific_training_features.py --splits opi/taps/probes/code_specific_training_control_splits_2026-05-17.json --device cuda",
        "venv/bin/python -u utilities/tests/manual/train_code_specific_tiny_heads_and_eval.py --splits opi/taps/probes/code_specific_training_control_splits_2026-05-17.json --features opi/taps/probes/code_specific_training_features_2026-05-17.pt",
        "venv/bin/python -u utilities/tests/manual/screen_more_code_strict_clean_tasks.py --target-additional-strict-clean 10 --max-new-tasks 60 --device cuda > opi/taps/probes/code_strict_clean_screening_expansion_2026-05-17.log 2>&1  # wrapper-blocked attempt; superseded",
        "venv/bin/python -u utilities/tests/manual/screen_more_code_strict_clean_tasks.py --target-additional-strict-clean 10 --max-new-tasks 60 --device cuda",
        "venv/bin/python -u utilities/tests/manual/screen_more_code_strict_clean_tasks.py --target-additional-strict-clean 10 --max-new-tasks 100 --device cuda",
        "venv/bin/python -u utilities/tests/manual/screen_more_code_strict_clean_tasks.py --target-additional-strict-clean 10 --max-new-tasks 160 --source-priority humaneval,local_dsa,mbpp --device cuda",
    ]


def create_final_summary(context: dict[str, Any], docs_updated: list[str], expansion_payload: dict[str, Any]) -> dict[str, Any]:
    ts = context["tiny_summary"]
    es = context["expansion_summary"]
    blockers: list[str] = []
    if context["tiny_verdict"] == "NOT_RUN":
        blockers.append("Code-specific tiny-head control did not run.")
    summary = {
        "CODE_SPECIFIC_SPLIT_VERDICT": context["split_verdict"],
        "CODE_SPECIFIC_FEATURE_VERDICT": context["feature_verdict"],
        "CODE_SPECIFIC_TINY_HEAD_VERDICT": context["tiny_verdict"],
        "STRICT_CLEAN_SCREENING_EXPANSION_VERDICT": context["expansion_verdict"],
        "RECOMMENDED_NEXT": context["recommended_next"],
        "training_task_count": ts.get("training_task_count"),
        "primary_training_pair_count": ts.get("primary_training_pair_count"),
        "heldout_strict_clean_task_count": ts.get("heldout_strict_clean_task_count"),
        "best_code_trained_antisymlinear": ts.get("best_code_trained_antisymlinear", "NA"),
        "best_code_trained_nonorm": ts.get("best_code_trained_nonorm", "NA"),
        "hh_trained_strict_clean_baseline": {
            "verdict": "WEAK",
            "config": "47_concat_all_loops",
            "architecture": "AntisymLinearNoNorm",
            "top1": 0.500,
            "pairwise": 0.571,
            "cycle": 0.000,
        },
        "screening_expansion_counts": {
            "tasks_screened": es.get("tasks_screened"),
            "strict_clean_ready": es.get("strict_clean_ready"),
            "anchor_only": es.get("anchor_only"),
            "near_miss_only": es.get("near_miss_only"),
            "all_correct": es.get("all_correct"),
            "all_wrong": es.get("all_wrong"),
            "label_totals": es.get("label_totals"),
        },
        "new_strict_clean_ready_task_ids": es.get("strict_clean_ready_task_ids", []),
        "interpretation": ts.get("one_sentence_interpretation", ""),
        "docs_updated": docs_updated,
        "files_modified_or_created": [
            "shared/utilities/tests/manual/build_code_specific_training_control_splits.py",
            "shared/utilities/tests/manual/ensure_code_specific_training_features.py",
            "shared/utilities/tests/manual/train_code_specific_tiny_heads_and_eval.py",
            "shared/utilities/tests/manual/screen_more_code_strict_clean_tasks.py",
            "opi/taps/probes/code_specific_training_control_splits_2026-05-17.json",
            "opi/taps/probes/code_specific_training_control_splits_2026-05-17.md",
            "opi/taps/probes/code_specific_training_features_2026-05-17.pt",
            "opi/taps/probes/code_specific_training_features_2026-05-17.md",
            "opi/taps/probes/code_specific_tiny_head_control_2026-05-17.json",
            "opi/taps/probes/code_specific_tiny_head_control_2026-05-17.md",
            "opi/taps/probes/code_strict_clean_screening_expansion_2026-05-17.json",
            "opi/taps/probes/code_strict_clean_screening_expansion_2026-05-17.md",
            "opi/taps/probes/code_strict_clean_screening_expansion_2026-05-17.log",
            "opi/taps/probes/code_specific_control_and_screening_2026-05-17_summary.json",
            "opi/taps/probes/code_specific_control_and_screening_2026-05-17_summary.md",
            *docs_updated,
        ],
        "commands_run": command_log(),
        "blockers": blockers,
    }
    return summary


def write_final_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Code-Specific Control And Screening Summary",
        "",
        f"CODE_SPECIFIC_SPLIT_VERDICT = {summary['CODE_SPECIFIC_SPLIT_VERDICT']}",
        f"CODE_SPECIFIC_FEATURE_VERDICT = {summary['CODE_SPECIFIC_FEATURE_VERDICT']}",
        f"CODE_SPECIFIC_TINY_HEAD_VERDICT = {summary['CODE_SPECIFIC_TINY_HEAD_VERDICT']}",
        f"STRICT_CLEAN_SCREENING_EXPANSION_VERDICT = {summary['STRICT_CLEAN_SCREENING_EXPANSION_VERDICT']}",
        f"RECOMMENDED_NEXT = {summary['RECOMMENDED_NEXT']}",
        "",
        "## 1. Code-specific training-control split",
        "",
        f"- training tasks / primary pairs: `{summary['training_task_count']}` / `{summary['primary_training_pair_count']}`",
        f"- held-out strict-clean tasks: `{summary['heldout_strict_clean_task_count']}`",
        "",
        "## 2. Feature coverage",
        "",
        f"- feature verdict: `{summary['CODE_SPECIFIC_FEATURE_VERDICT']}`",
        "",
        "## 3. Code-specific tiny-head results",
        "",
        f"- best AntisymLinear: `{summary['best_code_trained_antisymlinear']}`",
        f"- best NoNorm: `{summary['best_code_trained_nonorm']}`",
        "",
        "## 4. Comparison to HH-trained strict-clean transfer",
        "",
        f"- HH-trained strict-clean baseline: `{summary['hh_trained_strict_clean_baseline']}`",
        "",
        "## 5. Screening expansion results",
        "",
        f"- counts: `{summary['screening_expansion_counts']}`",
        f"- new strict_clean_ready task IDs: `{summary['new_strict_clean_ready_task_ids']}`",
        "",
        "## 6. Interpretation",
        "",
        summary.get("interpretation") or "No interpretation available.",
        "",
        "## 7. Docs updated",
        "",
    ]
    lines.extend(f"- `{path}`" for path in summary.get("docs_updated", []))
    if not summary.get("docs_updated"):
        lines.append("- none appended; section already present or doc missing")
    lines.extend(["", "## 8. Files modified / created", ""])
    lines.extend(f"- `{path}`" for path in summary["files_modified_or_created"])
    lines.extend(["", "## 9. Commands run", "", "```bash"])
    lines.extend(summary["commands_run"])
    lines.extend(["```", "", "## 10. Blockers", ""])
    if summary["blockers"]:
        lines.extend(f"- {item}" for item in summary["blockers"])
    else:
        lines.append("None.")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    with tee_log(args.log_file):
        payload = screen_tasks(args)
    out_json = Path(args.output)
    out_md = Path(args.output_md)
    write_json(out_json, payload)
    write_expansion_md(out_md, payload)
    print(f"STRICT_CLEAN_SCREENING_EXPANSION_VERDICT = {payload['strict_clean_screening_expansion_verdict']}", flush=True)
    print(f"strict_clean_ready = {payload['summary']['strict_clean_ready']}", flush=True)
    print(f"Wrote {out_json}", flush=True)
    print(f"Wrote {out_md}", flush=True)
    if payload.get("wrapper_blocked"):
        raise SystemExit("STRICT_CLEAN_SCREENING_EXPANSION_VERDICT=BLOCKED")

    context = load_result_context(payload)
    docs_updated = append_docs(context)
    final_summary = create_final_summary(context, docs_updated, payload)
    write_json(SUMMARY_JSON, final_summary)
    write_final_summary_md(SUMMARY_MD, final_summary)
    print(f"Wrote {SUMMARY_JSON}", flush=True)
    print(f"Wrote {SUMMARY_MD}", flush=True)


if __name__ == "__main__":
    main()
