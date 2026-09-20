"""Write final reports for the patched code-branch v2-mini pilot."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from code_branch_pilot_lib import PROJECT_ROOT, REPORT_DIR, output_path, repo_path, write_json


PATCH_JSON = REPORT_DIR / "code_branch_v2_patch_status_2026-05-16.json"
TASKSET_JSON = REPORT_DIR / "code_branch_taskset_v2_mini_patched_2026-05-16.json"
CANDIDATES_JSON = REPORT_DIR / "code_branch_candidates_v2_mini_patched_2026-05-16.json"
TOURNAMENTS_JSON = REPORT_DIR / "code_branch_tournaments_v2_mini_patched_2026-05-16.json"
FEATURES_PT = REPORT_DIR / "code_branch_tap_features_v2_mini_patched_2026-05-16.pt"
TRANSFER_JSON = REPORT_DIR / "code_branch_transfer_v2_mini_patched_2026-05-16.json"
RISKS_JSON = REPORT_DIR / "code_branch_future_risks_2026-05-16.json"
RISKS_MD = REPORT_DIR / "code_branch_future_risks_2026-05-16.md"
SUMMARY_JSON = REPORT_DIR / "code_branch_pilot_v2_mini_patched_2026-05-16_summary.json"
SUMMARY_MD = REPORT_DIR / "code_branch_pilot_v2_mini_patched_2026-05-16_summary.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-status", default=str(PATCH_JSON))
    parser.add_argument("--taskset", default=str(TASKSET_JSON))
    parser.add_argument("--candidates", default=str(CANDIDATES_JSON))
    parser.add_argument("--tournaments", default=str(TOURNAMENTS_JSON))
    parser.add_argument("--features", default=str(FEATURES_PT))
    parser.add_argument("--transfer", default=str(TRANSFER_JSON))
    parser.add_argument("--summary-output", default=str(SUMMARY_JSON))
    parser.add_argument("--summary-md", default=str(SUMMARY_MD))
    parser.add_argument("--risks-output", default=str(RISKS_JSON))
    parser.add_argument("--risks-md", default=str(RISKS_MD))
    return parser.parse_args()


def load_json_optional(path: str | Path) -> dict[str, Any]:
    p = output_path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def rate(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return "NA"
    if math.isnan(x):
        return "NA"
    return f"{x:.3f}"


def compact_row(row: dict[str, Any] | None) -> dict[str, Any] | str:
    if not row:
        return "NOT_RUN"
    metrics = row.get("metrics", {})
    return {
        "config": row.get("config"),
        "architecture": row.get("architecture"),
        "top1": metrics.get("top1_tournament_acc"),
        "pairwise": metrics.get("pairwise_acc"),
        "cycle": metrics.get("cycle_rate"),
        "margin_mean": metrics.get("margin_mean"),
        "margin_std": metrics.get("margin_std"),
    }


def recommended_next(tournament_verdict: str, transfer_verdict: str) -> str:
    if tournament_verdict in {"CLEAN", "RUNNABLE_DIAGNOSTIC"} and transfer_verdict == "GOOD":
        return "run_independent_10_task_near_miss_enrichment_or_expand_code_to_30_clean_tournaments"
    if tournament_verdict == "BROAD_DIAGNOSTIC" and transfer_verdict == "GOOD":
        return "improve_runnable_near_miss_yield_before_scaling"
    if transfer_verdict == "WEAK":
        return "run_independent_10_task_near_miss_enrichment"
    if transfer_verdict == "POOR":
        return "investigate_code_domain_mismatch_or_train_code_specific_taps"
    if tournament_verdict == "TOO_EASY":
        return "harder_tasks_and_less_repaired_final"
    if tournament_verdict == "TOO_HARD":
        return "more_medium_tasks_and_repaired_final_anchor"
    if tournament_verdict == "TOO_FEW_TOURNAMENTS":
        return "adjust_candidate_modes_or_task_mix"
    return "finish_blocked_stage_before_scaling"


def future_risks() -> list[dict[str, Any]]:
    rows = [
        ("taskset_function_name_parsing_bugs", True, "high", "AST-test extraction that skips outer builtins", "Pin affected tasks and inspect resolved function names", True),
        ("signature_ambiguity", True, "high", "Infer exact arity from official tests and include signature hints", "Add targeted task prompts or exclude ambiguous tasks", True),
        ("wrapper_prose_status_leakage", True, "high", "Reject status/prose in sanitizer and final-code extraction", "Treat as malformed and inspect raw wrapper text", True),
        ("repaired_final_over_success_all_correct_collapse", True, "medium", "Cap repaired_final to one per task and prioritize pre-final stages", "Add harder tasks or reduce final repair", True),
        ("all_nonsense_collapse_on_too_hard_tasks", True, "medium", "Mix medium/local DSA with a small devil quota", "Add easier anchor tasks and inspect safety/runtime labels", True),
        ("duplicate_near_duplicate_candidates", True, "medium", "Deduplicate raw, normalized, and AST hashes", "Increase sampling diversity or change modes", True),
        ("humaneval_coarse_pass_fail_labels", False, "medium", "Keep HumanEval mixed with MBPP/local tests", "Add public unit slices or local hidden tests", False),
        ("public_only_weak_test_suites", True, "medium", "Use official hidden tests where available and local hidden tests", "Add stronger local tests for weak tasks", True),
        ("unsafe_code_false_positives_sys_setrecursionlimit", True, "medium", "Allow only sys.setrecursionlimit", "Whitelist exact safe sys usage after AST review", True),
        ("unsafe_code_false_negatives", False, "high", "AST import/call/attribute denylist", "Expand denylist and sandbox subprocess eval", True),
        ("feature_candidate_id_mismatch", False, "high", "Capture features from tournament candidate metadata", "Validate tournament_id/task_id/candidate_index joins", True),
        ("label_leakage_in_feature_capture", False, "high", "Feature text includes only prompt and candidate code", "Audit captured texts before transfer", True),
        ("random_top1_baseline_mistakes", True, "medium", "Compute baseline from actual primary candidate pools", "Report baselines for every eval set", True),
        ("nonorm_cycle_rate_bugs", False, "medium", "Assert NoNorm cycle rate remains zero", "Stop transfer and inspect antisymmetry math", True),
        ("diagnostic_all_inflation_from_malformed_candidates", True, "medium", "Separate strict/runnable/mixed sets and exclude malformed from primary pools", "Use runnable primary sets before scaling", True),
        ("candidate_self_tests_contaminating_evaluation", False, "medium", "Ignore candidate self-test text; labels come from harness tests", "Strip generated asserts if they interfere", True),
        ("time_cost_runaway", True, "medium", "Hard caps on tasks, candidates, and repaired finals", "Stop on partial checkpoints and report partials", True),
    ]
    return [
        {
            "risk_name": name,
            "observed_in_v2": observed,
            "severity": severity,
            "prevention": prevention,
            "remediation": remediation,
            "current_patched_rerun_addresses_it": addressed,
        }
        for name, observed, severity, prevention, remediation, addressed in rows
    ]


def write_risk_reports(json_path: Path, md_path: Path) -> list[dict[str, Any]]:
    risks = future_risks()
    write_json(json_path, {"risks": risks, "risk_count": len(risks)})
    lines = [
        "# Code Branch Future Risks",
        "",
        f"- risk_count: `{len(risks)}`",
        "",
        "| risk | observed_in_v2 | severity | prevention | remediation | addressed_now |",
        "| --- | ---: | --- | --- | --- | ---: |",
    ]
    for row in risks:
        lines.append(
            f"| `{row['risk_name']}` | {row['observed_in_v2']} | `{row['severity']}` | "
            f"{row['prevention']} | {row['remediation']} | {row['current_patched_rerun_addresses_it']} |"
        )
    lines.append("")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return risks


def append_docs(summary: dict[str, Any], docs: list[Path]) -> list[str]:
    title = "## Patched code branch pilot v2-mini (2026-05-16)"
    best_a = summary.get("best_antisymlinear", "NOT_RUN")
    best_n = summary.get("best_nonorm", "NOT_RUN")
    text = "\n".join([
        "",
        title,
        "",
        f"- CODE_V2_PATCH_STATUS: `{summary['CODE_V2_PATCH_STATUS']}`",
        f"- CODE_V2_MINI_TASKSET_VERDICT: `{summary['CODE_V2_MINI_TASKSET_VERDICT']}`",
        f"- CODE_V2_MINI_GENERATION_VERDICT: `{summary['CODE_V2_MINI_GENERATION_VERDICT']}`",
        f"- CODE_V2_MINI_TOURNAMENT_VERDICT: `{summary['CODE_V2_MINI_TOURNAMENT_VERDICT']}`",
        f"- CODE_V2_MINI_TRANSFER_VERDICT: `{summary['CODE_V2_MINI_TRANSFER_VERDICT']}`",
        f"- tasks: `{summary['tasks']}`",
        f"- unique candidates: `{summary['unique_candidates']}`",
        f"- label counts: `{summary['label_counts']}`",
        f"- strict_clean / diagnostic_runnable / diagnostic_mixed: `{summary['strict_clean_tournaments']} / {summary['diagnostic_runnable_tournaments']} / {summary['diagnostic_mixed_tournaments']}`",
        f"- primary_eval_set: `{summary['primary_eval_set']}`",
        f"- random_top1_baseline: `{summary['random_top1_baseline']}`",
        f"- best AntisymLinear row: `{best_a}`",
        f"- best NoNorm row: `{best_n}`",
        f"- winner top1 / pairwise / cycle: `{summary['winner_top1_pairwise_cycle']}`",
        f"- transfer signal survived patched path: `{summary['transfer_signal_survived']}`",
        f"- wrapper-prose artifacts eliminated: `{summary['wrapper_prose_artifacts_eliminated']}`",
        f"- mbpp/232 fixed: `{summary['mbpp_232_fixed']}`",
        f"- mbpp/306 fixed: `{summary['mbpp_306_fixed']}`",
        f"- future-risk-register path: `{summary['future_risk_register_path']}`",
        f"- full report path: `{summary['full_report_path']}`",
        f"- interpretation: {summary['interpretation']}",
        "",
    ])
    appended = []
    for path in docs:
        if not path.exists():
            continue
        current = path.read_text(encoding="utf-8")
        if title in current:
            continue
        path.write_text(current.rstrip() + "\n" + text, encoding="utf-8")
        appended.append(repo_path(path))
    return appended


def append_clarification_docs(summary: dict[str, Any], docs: list[Path]) -> list[str]:
    title = "## Patched code branch pilot v2-mini clarification (2026-05-16)"
    text = "\n".join([
        "",
        title,
        "",
        f"- wrapper-prose artifacts eliminated from admitted candidates: `{summary['wrapper_prose_artifacts_eliminated']}`",
        f"- raw wrapper/status generations rejected before candidate admission: `{summary['raw_wrapper_status_rejected_count']}`",
        f"- admitted wrapper/status artifact count: `{summary['admitted_wrapper_artifact_count']}`",
        "",
    ])
    appended = []
    for path in docs:
        if not path.exists():
            continue
        current = path.read_text(encoding="utf-8")
        if title in current:
            continue
        path.write_text(current.rstrip() + "\n" + text, encoding="utf-8")
        appended.append(repo_path(path))
    return appended


def build_summary(args: argparse.Namespace, risks: list[dict[str, Any]]) -> dict[str, Any]:
    patch = load_json_optional(args.patch_status)
    taskset = load_json_optional(args.taskset)
    candidates = load_json_optional(args.candidates)
    tournaments = load_json_optional(args.tournaments)
    transfer = load_json_optional(args.transfer)
    feature_path = output_path(args.features)

    t_summary = tournaments.get("summary", {})
    label_counts = {
        "correct": t_summary.get("correct_candidates", 0),
        "near_miss": t_summary.get("near_miss_candidates", 0),
        "wrong_code": t_summary.get("wrong_code_candidates", 0),
        "runtime_error": t_summary.get("runtime_error_candidates", 0),
        "malformed": t_summary.get("malformed_candidates", 0),
        "safety_rejected": t_summary.get("safety_rejected_candidates", 0),
    }
    best_a = compact_row(transfer.get("best_antisymlinear") if transfer else None)
    best_n = compact_row(transfer.get("best_nonorm") if transfer else None)
    best = compact_row(transfer.get("best_hh_trained") if transfer else None)
    if isinstance(best, dict):
        winner_tuple = {
            "top1": best.get("top1"),
            "pairwise": best.get("pairwise"),
            "cycle": best.get("cycle"),
        }
    else:
        winner_tuple = "NOT_RUN"

    generation_errors = candidates.get("generation_errors", []) if candidates else []
    raw_wrapper_status_rejections = [
        row for row in generation_errors
        if "wrapper" in str(row).lower() or "max steps reached" in str(row).lower() or "prose" in str(row).lower()
    ]
    admitted_wrapper_artifacts = [
        row
        for row in tournaments.get("candidate_evaluations", [])
        if "max steps reached" in str(row.get("final_code", "")).lower()
        or "max steps reached" in str(row.get("raw_final_code", "")).lower()
    ]
    tasks = taskset.get("tasks", [])
    mbpp_232 = next((task for task in tasks if task.get("task_id") == "mbpp/232"), {})
    mbpp_306 = next((task for task in tasks if task.get("task_id") == "mbpp/306"), {})
    tournament_verdict = tournaments.get("code_v2_tournament_verdict", "NOT_RUN")
    transfer_verdict = transfer.get("code_v2_mini_transfer_verdict") or transfer.get("code_v2_transfer_verdict") or "NOT_RUN"
    summary = {
        "CODE_V2_PATCH_STATUS": patch.get("code_v2_patch_status", "BLOCKED" if not patch else "PARTIAL"),
        "CODE_V2_MINI_TASKSET_VERDICT": taskset.get("code_v2_taskset_verdict", "NOT_RUN"),
        "CODE_V2_MINI_GENERATION_VERDICT": candidates.get("code_v2_generation_verdict", "NOT_RUN"),
        "CODE_V2_MINI_TOURNAMENT_VERDICT": tournament_verdict,
        "CODE_V2_MINI_TRANSFER_VERDICT": transfer_verdict,
        "RECOMMENDED_NEXT": recommended_next(tournament_verdict, transfer_verdict),
        "tasks": len(tasks),
        "source_mix": taskset.get("source_mix", {}),
        "difficulty_mix": taskset.get("difficulty_mix", {}),
        "unique_candidates": candidates.get("summary", {}).get("unique_candidates", t_summary.get("unique_candidates", 0)),
        "duplicate_rate": candidates.get("summary", {}).get("duplicate_rate", t_summary.get("duplicate_rate", 0.0)),
        "stage_breakdown": candidates.get("summary", {}).get("stage_breakdown", t_summary.get("stage_breakdown", {})),
        "label_counts": label_counts,
        "label_by_stage": t_summary.get("label_by_stage", {}),
        "strict_clean_tournaments": t_summary.get("strict_clean_tournaments", 0),
        "diagnostic_runnable_tournaments": t_summary.get("diagnostic_runnable_tournaments", 0),
        "diagnostic_mixed_tournaments": t_summary.get("diagnostic_mixed_tournaments", 0),
        "primary_eval_set": tournaments.get("primary_eval_set", "none"),
        "random_top1_baseline": t_summary.get("random_top1_baseline", float("nan")),
        "random_top1_baselines": t_summary.get("random_top1_baselines", {}),
        "best_antisymlinear": best_a,
        "best_nonorm": best_n,
        "best_hh_trained": best,
        "winner_top1_pairwise_cycle": winner_tuple,
        "winner_family": best.get("architecture", "NOT_RUN") if isinstance(best, dict) else "NOT_RUN",
        "winner_layer_family": str(best.get("config", "NOT_RUN")).split("_", 1)[0] if isinstance(best, dict) else "NOT_RUN",
        "transfer_signal_survived": transfer_verdict == "GOOD",
        "wrapper_prose_artifacts_eliminated": not admitted_wrapper_artifacts,
        "raw_wrapper_status_rejected_count": len(raw_wrapper_status_rejections),
        "admitted_wrapper_artifact_count": len(admitted_wrapper_artifacts),
        "mbpp_232_fixed": mbpp_232.get("function_name") == "larg_nnum",
        "mbpp_306_fixed": "arg4" in str(mbpp_306.get("signature_hint", "")),
        "mbpp_232_outcome": t_summary.get("mbpp_232_outcome", {}),
        "mbpp_306_outcome": t_summary.get("mbpp_306_outcome", {}),
        "feature_file": repo_path(feature_path) if feature_path.exists() else "NOT_RUN",
        "feature_candidates": transfer.get("feature_summary", {}).get("n_candidates", 0) if transfer else 0,
        "future_risk_register_path": repo_path(output_path(args.risks_md)),
        "full_report_path": repo_path(output_path(args.summary_md)),
        "risk_count": len(risks),
        "expanded_after_initial_mini": taskset.get("target_tasks", 0) > 15,
        "interpretation": interpretation(tournament_verdict, transfer_verdict),
        "comparison_to_pre_fix_v2": {
            "pre_fix_best_nonorm_36_mean_top1": 0.727,
            "pre_fix_best_nonorm_36_mean_pairwise": 0.691,
            "pre_fix_best_antisymlinear_36_L4_top1": 0.682,
            "pre_fix_best_antisymlinear_36_L4_pairwise": 0.709,
            "pre_fix_best_47_concat_all_loops_pairwise": 0.764,
            "current_result_validity": "patched_pipeline_current" if transfer else "not_run",
        },
    }
    docs = [
        PROJECT_ROOT / "docs/evaluator/evaluator_domain_transfer_notes.md",
        PROJECT_ROOT / "docs/evaluator/math_bg_gate_pilot_2026-05-15.md",
        PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-15_v4.md",
        PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-16_v5_clean_gsm8k_code_next.md",
    ]
    summary["docs_updated"] = append_docs(summary, docs)
    summary["docs_updated"].extend(append_clarification_docs(summary, docs))
    summary["files_modified_or_created"] = files_list(args, bool(transfer), feature_path.exists())
    summary["commands_run"] = commands_run(bool(transfer), feature_path.exists())
    summary["blockers"] = blockers(summary)
    return summary


def interpretation(tournament_verdict: str, transfer_verdict: str) -> str:
    if transfer_verdict == "GOOD":
        return "The patched harness produced usable code-branch tournaments and HH-trained linear transfer remained above the random branch-selection baseline."
    if tournament_verdict in {"TOO_EASY", "TOO_HARD", "TOO_FEW_TOURNAMENTS"}:
        return "The patched harness ran, but the mini task/candidate mix did not produce enough usable tournaments for transfer."
    if transfer_verdict == "WEAK":
        return "The patched harness produced usable tournaments, but the HH transfer signal was weaker than the pre-fix v2 diagnostic result."
    if transfer_verdict == "POOR":
        return "The patched harness produced usable tournaments, but HH-trained transfer did not survive this code-domain check."
    return "The patched rerun stopped before a transfer verdict."


def files_list(args: argparse.Namespace, transfer_ran: bool, features_exist: bool) -> list[str]:
    files = [
        "shared/utilities/tests/manual/verify_code_branch_v2_patch_status.py",
        "shared/utilities/tests/manual/build_code_branch_taskset_v2.py",
        "shared/utilities/tests/manual/generate_code_branch_candidates_v2.py",
        "shared/utilities/tests/manual/evaluate_code_branch_candidates_v2.py",
        "shared/utilities/tests/manual/capture_code_branch_tap_features_v2.py",
        "shared/utilities/tests/manual/evaluate_hh_transfer_on_code_branches_v2.py",
        "shared/utilities/tests/manual/finalize_code_branch_v2_mini_patched.py",
        repo_path(output_path(args.patch_status)),
        repo_path(Path(str(output_path(args.patch_status))).with_suffix(".md")),
        repo_path(output_path(args.taskset)),
        repo_path(Path(str(output_path(args.taskset))).with_suffix(".md")),
        repo_path(output_path(args.candidates)),
        repo_path(Path(str(output_path(args.candidates))).with_suffix(".md")),
        repo_path(Path(str(output_path(args.candidates))).with_suffix(".log")),
        repo_path(output_path(args.tournaments)),
        repo_path(Path(str(output_path(args.tournaments))).with_suffix(".md")),
        repo_path(output_path(args.risks_output)),
        repo_path(output_path(args.risks_md)),
        repo_path(output_path(args.summary_output)),
        repo_path(output_path(args.summary_md)),
    ]
    if features_exist:
        files.append(repo_path(output_path(args.features)))
        files.append(repo_path(Path(str(output_path(args.features))).with_suffix(".md")))
    if transfer_ran:
        files.append(repo_path(output_path(args.transfer)))
        files.append(repo_path(Path(str(output_path(args.transfer))).with_suffix(".md")))
    files.extend([
        "docs/evaluator/evaluator_domain_transfer_notes.md",
        "docs/evaluator/math_bg_gate_pilot_2026-05-15.md",
        "docs/evaluator/post_v10_synthesis_2026-05-15_v4.md",
        "docs/evaluator/post_v10_synthesis_2026-05-16_v5_clean_gsm8k_code_next.md",
    ])
    return files


def commands_run(transfer_ran: bool, features_exist: bool) -> list[str]:
    commands = [
        "venv/bin/python -m py_compile utilities/tests/manual/verify_code_branch_v2_patch_status.py utilities/tests/manual/build_code_branch_taskset_v2.py utilities/tests/manual/generate_code_branch_candidates_v2.py utilities/tests/manual/evaluate_code_branch_candidates_v2.py utilities/tests/manual/capture_code_branch_tap_features_v2.py utilities/tests/manual/evaluate_hh_transfer_on_code_branches_v2.py utilities/tests/manual/finalize_code_branch_v2_mini_patched.py",
        "venv/bin/python -u utilities/tests/manual/verify_code_branch_v2_patch_status.py",
        "venv/bin/python -u utilities/tests/manual/build_code_branch_taskset_v2.py --target-tasks 12 --min-tasks 8 --include-devil --include-specific mbpp/232 mbpp/306 mbpp/237 --max-easy-fraction 0.40 --output opi/taps/probes/code_branch_taskset_v2_mini_patched_2026-05-16.json",
        "venv/bin/python -u utilities/tests/manual/generate_code_branch_candidates_v2.py --taskset opi/taps/probes/code_branch_taskset_v2_mini_patched_2026-05-16.json --max-tasks 15 --min-tasks 8 --max-candidates-per-task 6 --hard-cap-total-candidates 90 --target-usable-tournaments 10 --prefer-prefinal --limit-repaired-final-per-task 1 --device cuda --no-resume",
        "venv/bin/python -u utilities/tests/manual/evaluate_code_branch_candidates_v2.py --candidates opi/taps/probes/code_branch_candidates_v2_mini_patched_2026-05-16.json --taskset opi/taps/probes/code_branch_taskset_v2_mini_patched_2026-05-16.json --output opi/taps/probes/code_branch_tournaments_v2_mini_patched_2026-05-16.json",
        "venv/bin/python -u utilities/tests/manual/build_code_branch_taskset_v2.py --target-tasks 30 --min-tasks 20 --include-devil --include-specific mbpp/232 mbpp/306 mbpp/237 --max-easy-fraction 0.40 --output opi/taps/probes/code_branch_taskset_v2_mini_patched_2026-05-16.json",
        "venv/bin/python -u utilities/tests/manual/generate_code_branch_candidates_v2.py --taskset opi/taps/probes/code_branch_taskset_v2_mini_patched_2026-05-16.json --max-tasks 30 --min-tasks 20 --max-candidates-per-task 5 --hard-cap-total-candidates 180 --target-usable-tournaments 20 --prefer-prefinal --limit-repaired-final-per-task 0 --device cuda",
        "venv/bin/python -u utilities/tests/manual/evaluate_code_branch_candidates_v2.py --candidates opi/taps/probes/code_branch_candidates_v2_mini_patched_2026-05-16.json --taskset opi/taps/probes/code_branch_taskset_v2_mini_patched_2026-05-16.json --output opi/taps/probes/code_branch_tournaments_v2_mini_patched_2026-05-16.json --no-resume",
    ]
    if features_exist:
        commands.append("venv/bin/python -u utilities/tests/manual/capture_code_branch_tap_features_v2.py --input opi/taps/probes/code_branch_tournaments_v2_mini_patched_2026-05-16.json --output opi/taps/probes/code_branch_tap_features_v2_mini_patched_2026-05-16.pt --device cuda")
    if transfer_ran:
        commands.append("venv/bin/python -u utilities/tests/manual/evaluate_hh_transfer_on_code_branches_v2.py --features opi/taps/probes/code_branch_tap_features_v2_mini_patched_2026-05-16.pt --hh-capture rpe/evaluator/hh_layer_states_200_rltt.pt --output opi/taps/probes/code_branch_transfer_v2_mini_patched_2026-05-16.json")
    commands.append("venv/bin/python -u utilities/tests/manual/finalize_code_branch_v2_mini_patched.py")
    return commands


def blockers(summary: dict[str, Any]) -> str:
    if summary["CODE_V2_PATCH_STATUS"] == "BLOCKED":
        return "Patch verification blocked the run."
    if summary["CODE_V2_MINI_TASKSET_VERDICT"] == "BLOCKED":
        return "Taskset construction blocked the run."
    if summary["CODE_V2_MINI_GENERATION_VERDICT"] == "WRAPPER_BLOCKED":
        return "Candidate generation was wrapper-blocked."
    if summary["CODE_V2_MINI_TOURNAMENT_VERDICT"] in {"TOO_EASY", "TOO_HARD", "TOO_FEW_TOURNAMENTS"}:
        return "Transfer was intentionally not run because tournament construction did not meet the guarded threshold."
    if summary["CODE_V2_MINI_TRANSFER_VERDICT"] == "NOT_RUN":
        return "Transfer was not run because features or HH capture were unavailable."
    return "None."


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Code Branch Pilot v2 Mini Patched Summary",
        "",
        f"CODE_V2_PATCH_STATUS = {summary['CODE_V2_PATCH_STATUS']}",
        f"CODE_V2_MINI_TASKSET_VERDICT = {summary['CODE_V2_MINI_TASKSET_VERDICT']}",
        f"CODE_V2_MINI_GENERATION_VERDICT = {summary['CODE_V2_MINI_GENERATION_VERDICT']}",
        f"CODE_V2_MINI_TOURNAMENT_VERDICT = {summary['CODE_V2_MINI_TOURNAMENT_VERDICT']}",
        f"CODE_V2_MINI_TRANSFER_VERDICT = {summary['CODE_V2_MINI_TRANSFER_VERDICT']}",
        f"RECOMMENDED_NEXT = {summary['RECOMMENDED_NEXT']}",
        "",
        "## Patch Verification",
        "",
        f"- patch_status: `{summary['CODE_V2_PATCH_STATUS']}`",
        f"- wrapper_prose_artifacts_eliminated: `{summary['wrapper_prose_artifacts_eliminated']}`",
        f"- raw_wrapper_status_rejected_count: `{summary['raw_wrapper_status_rejected_count']}`",
        f"- admitted_wrapper_artifact_count: `{summary['admitted_wrapper_artifact_count']}`",
        f"- mbpp/232 fixed: `{summary['mbpp_232_fixed']}`",
        f"- mbpp/306 fixed: `{summary['mbpp_306_fixed']}`",
        "",
        "## Taskset Summary",
        "",
        f"- tasks: `{summary['tasks']}`",
        f"- expanded_after_initial_mini: `{summary['expanded_after_initial_mini']}`",
        f"- source_mix: `{summary['source_mix']}`",
        f"- difficulty_mix: `{summary['difficulty_mix']}`",
        "",
        "## Candidate Generation Summary",
        "",
        f"- unique_candidates: `{summary['unique_candidates']}`",
        f"- duplicate_rate: `{summary['duplicate_rate']}`",
        f"- stage_breakdown: `{summary['stage_breakdown']}`",
        "",
        "## Tournament Construction Summary",
        "",
        f"- label_counts: `{summary['label_counts']}`",
        f"- strict_clean_tournaments: `{summary['strict_clean_tournaments']}`",
        f"- diagnostic_runnable_tournaments: `{summary['diagnostic_runnable_tournaments']}`",
        f"- diagnostic_mixed_tournaments: `{summary['diagnostic_mixed_tournaments']}`",
        f"- primary_eval_set: `{summary['primary_eval_set']}`",
        f"- random_top1_baseline: `{summary['random_top1_baseline']}`",
        f"- mbpp_232_outcome: `{summary['mbpp_232_outcome']}`",
        f"- mbpp_306_outcome: `{summary['mbpp_306_outcome']}`",
        "",
        "## Transfer Evaluation Summary",
        "",
        f"- best AntisymLinear row: `{summary['best_antisymlinear']}`",
        f"- best NoNorm row: `{summary['best_nonorm']}`",
        f"- winner: `{summary['best_hh_trained']}`",
        f"- winner_family: `{summary['winner_family']}`",
        f"- winner_layer_family: `{summary['winner_layer_family']}`",
        f"- transfer_signal_survived: `{summary['transfer_signal_survived']}`",
        "",
        "## Comparison To Pre-Fix v2",
        "",
        f"- pre_fix: `{summary['comparison_to_pre_fix_v2']}`",
        "",
        "## Future-Risk Register Summary",
        "",
        f"- risk_count: `{summary['risk_count']}`",
        f"- future_risk_register_path: `{summary['future_risk_register_path']}`",
        "",
        "## Markdown Docs Updated",
        "",
    ]
    lines.extend(f"- `{path}`" for path in summary.get("docs_updated", []))
    if not summary.get("docs_updated"):
        lines.append("- `already_present_or_missing`")
    lines.extend(["", "## Files Modified / Created", ""])
    lines.extend(f"- `{path}`" for path in summary["files_modified_or_created"])
    lines.extend(["", "## Commands Run", "", "```bash"])
    lines.extend(summary["commands_run"])
    lines.extend(["```", "", "## Blockers", "", summary["blockers"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    risks = write_risk_reports(output_path(args.risks_output), output_path(args.risks_md))
    summary = build_summary(args, risks)
    write_json(output_path(args.summary_output), summary)
    write_summary_md(output_path(args.summary_md), summary)
    print(f"CODE_V2_MINI_TRANSFER_VERDICT = {summary['CODE_V2_MINI_TRANSFER_VERDICT']}")
    print(f"RECOMMENDED_NEXT = {summary['RECOMMENDED_NEXT']}")
    print(f"Wrote {output_path(args.summary_output)}")
    print(f"Wrote {output_path(args.summary_md)}")


if __name__ == "__main__":
    main()
