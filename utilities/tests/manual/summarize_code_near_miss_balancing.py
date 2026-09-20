"""Write final report and append docs for the near-miss balancing pass."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from code_branch_pilot_lib import PROJECT_ROOT, REPORT_DIR, repo_path, write_json


INSPECTION_JSON = REPORT_DIR / "code_branch_near_miss_balance_inspection_2026-05-17.json"
GEN_JSON = REPORT_DIR / "code_branch_near_miss_balancing_candidates_2026-05-17.json"
TOURN_JSON = REPORT_DIR / "code_branch_near_miss_balanced_tournaments_2026-05-17.json"
FEATURE_JSON = REPORT_DIR / "code_branch_near_miss_balanced_tap_features_2026-05-17.json"
TRANSFER_JSON = REPORT_DIR / "code_branch_near_miss_balanced_transfer_2026-05-17.json"
SUMMARY_JSON = REPORT_DIR / "code_branch_near_miss_balancing_2026-05-17_summary.json"
SUMMARY_MD = REPORT_DIR / "code_branch_near_miss_balancing_2026-05-17_summary.md"


def load_if(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def recommend(tournament_verdict: str, transfer_verdict: str) -> str:
    if tournament_verdict == "GREEN" and transfer_verdict == "GOOD":
        return "near_miss_code_transfer_confirmed_small_n__consider_one_more_independent_10_or_stop_code_for_now"
    if tournament_verdict == "GREEN" and transfer_verdict == "NOT_RUN":
        return "run_balanced_transfer_eval"
    if tournament_verdict == "YELLOW":
        return "accept_diagnostic_runnable_or_run_one_more_targeted_balancing"
    if tournament_verdict == "RED":
        return "task_design_or_test_granularity_is_bottleneck"
    return "inspect_balancing_artifacts"


def row_compact(row: Any) -> Any:
    if not isinstance(row, dict):
        return row or "NA"
    if "metrics" in row:
        m = row["metrics"]
        return {
            "config": row.get("config"),
            "architecture": row.get("architecture"),
            "top1": m.get("top1_tournament_acc"),
            "pairwise": m.get("pairwise_acc"),
            "cycle": m.get("cycle_rate"),
        }
    return row


def append_docs(summary: dict[str, Any]) -> list[str]:
    docs = [
        PROJECT_ROOT / "docs/evaluator/evaluator_domain_transfer_notes.md",
        PROJECT_ROOT / "docs/evaluator/math_bg_gate_pilot_2026-05-15.md",
        PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-15_v4.md",
        PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-16_v5_clean_gsm8k_code_next.md",
    ]
    title = "## Code near-miss balancing pass (2026-05-17)"
    text = "\n".join([
        "",
        title,
        "",
        f"- BALANCE_INSPECTION_VERDICT: `{summary['BALANCE_INSPECTION_VERDICT']}`",
        f"- BALANCING_GENERATION_VERDICT: `{summary['BALANCING_GENERATION_VERDICT']}`",
        f"- BALANCED_TOURNAMENT_VERDICT: `{summary['BALANCED_TOURNAMENT_VERDICT']}`",
        f"- BALANCED_FEATURE_VERDICT: `{summary['BALANCED_FEATURE_VERDICT']}`",
        f"- BALANCED_TRANSFER_VERDICT: `{summary['BALANCED_TRANSFER_VERDICT']}`",
        f"- tasks: `{summary['tasks']}`",
        f"- old strict_clean / new strict_clean: `{summary['old_strict_clean']} / {summary['new_strict_clean']}`",
        f"- old label counts / new label counts: `{summary['old_label_counts']} / {summary['new_label_counts']}`",
        f"- tasks converted to strict_clean: `{summary['tasks_converted_to_strict_clean']}`",
        f"- modes that helped: `{summary['modes_that_helped']}`",
        f"- best AntisymLinear row: `{summary['best_antisymlinear']}`",
        f"- best NoNorm row: `{summary['best_nonorm']}`",
        "- full report path: `opi/taps/probes/code_branch_near_miss_balancing_2026-05-17_summary.md`",
        f"- interpretation: {summary['interpretation']}",
        "",
    ])
    touched = []
    for path in docs:
        if not path.exists():
            continue
        touched.append(repo_path(path))
        current = path.read_text(encoding="utf-8")
        if title in current:
            continue
        path.write_text(current.rstrip() + "\n" + text, encoding="utf-8")
    return touched


def write_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Code Branch Near-Miss Balancing Summary",
        "",
        f"BALANCE_INSPECTION_VERDICT = {summary['BALANCE_INSPECTION_VERDICT']}",
        f"BALANCING_GENERATION_VERDICT = {summary['BALANCING_GENERATION_VERDICT']}",
        f"BALANCED_TOURNAMENT_VERDICT = {summary['BALANCED_TOURNAMENT_VERDICT']}",
        f"BALANCED_FEATURE_VERDICT = {summary['BALANCED_FEATURE_VERDICT']}",
        f"BALANCED_TRANSFER_VERDICT = {summary['BALANCED_TRANSFER_VERDICT']}",
        f"RECOMMENDED_NEXT = {summary['RECOMMENDED_NEXT']}",
        "",
        "## Balance Inspection",
        "",
        f"- existing strict_clean: `{summary['old_strict_clean']}`",
        f"- plausible target tasks: `{summary['plausible_target_tasks']}`",
        f"- bucket counts: `{summary['bucket_counts']}`",
        "",
        "## Targeted Generation Summary",
        "",
        f"- new unique candidates: `{summary['new_unique_candidates']}`",
        f"- new duplicate candidates: `{summary['new_duplicate_candidates']}`",
        f"- generation errors: `{summary['generation_errors']}`",
        f"- new label counts: `{summary['new_generation_label_counts']}`",
        "",
        "## Tournament Before/After",
        "",
        f"- old strict_clean / new strict_clean: `{summary['old_strict_clean']} / {summary['new_strict_clean']}`",
        f"- old label counts: `{summary['old_label_counts']}`",
        f"- new label counts: `{summary['new_label_counts']}`",
        f"- diagnostic_runnable: `{summary['diagnostic_runnable']}`",
        "",
        "## Candidate Modes That Helped",
        "",
        f"- tasks converted: `{summary['tasks_converted_to_strict_clean']}`",
        f"- modes that helped: `{summary['modes_that_helped']}`",
        "",
        "## Feature Handling",
        "",
        f"- feature verdict: `{summary['BALANCED_FEATURE_VERDICT']}`",
        f"- feature path: `{summary['feature_path']}`",
        "",
        "## Transfer Results",
        "",
        f"- transfer verdict: `{summary['BALANCED_TRANSFER_VERDICT']}`",
        f"- random_top1_baseline: `{summary['random_top1_baseline']}`",
        f"- best AntisymLinear: `{summary['best_antisymlinear']}`",
        f"- best NoNorm: `{summary['best_nonorm']}`",
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
        "",
        "## Docs Updated",
        "",
    ]
    lines.extend(f"- `{path}`" for path in summary.get("docs_updated", []))
    lines.extend(["", "## Files Modified / Created", ""])
    lines.extend(f"- `{path}`" for path in summary.get("files_modified_or_created", []))
    lines.extend(["", "## Commands Run", "", "```bash"])
    lines.extend(summary.get("commands_run", []))
    lines.extend(["```", "", "## Blockers", "", summary.get("blockers") or "None.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    inspection = load_if(INSPECTION_JSON)
    gen = load_if(GEN_JSON)
    tourn = load_if(TOURN_JSON)
    feature = load_if(FEATURE_JSON)
    transfer = load_if(TRANSFER_JSON)
    tournament_summary = tourn.get("summary", {})
    before = tournament_summary.get("before", {})
    after = tournament_summary.get("after", {})
    converted = tourn.get("converted_tasks", [])
    modes_that_helped = sorted({mode for row in converted for mode in row.get("helpful_modes", [])})
    transfer_verdict = transfer.get("balanced_transfer_verdict", "NOT_RUN")
    feature_verdict = feature.get("balanced_feature_verdict", "NOT_RUN")
    tournament_verdict = tourn.get("balanced_tournament_verdict", "NOT_RUN")
    commands_run = [
        "venv/bin/python -m py_compile utilities/tests/manual/inspect_code_near_miss_enrichment10_balance.py utilities/tests/manual/generate_code_near_miss_balancing_candidates.py utilities/tests/manual/evaluate_code_near_miss_balancing.py utilities/tests/manual/capture_code_near_miss_balanced_tap_features.py utilities/tests/manual/evaluate_hh_transfer_on_code_near_miss_balanced.py utilities/tests/manual/summarize_code_near_miss_balancing.py",
        "venv/bin/python -u utilities/tests/manual/inspect_code_near_miss_enrichment10_balance.py",
        "venv/bin/python -u utilities/tests/manual/generate_code_near_miss_balancing_candidates.py --max-new-candidates-total 30 --max-new-candidates-per-task 4 --target-strict-clean 5 --device cuda",
        "venv/bin/python -u utilities/tests/manual/evaluate_code_near_miss_balancing.py",
    ]
    if feature:
        commands_run.append("venv/bin/python -u utilities/tests/manual/capture_code_near_miss_balanced_tap_features.py --device cuda")
    if transfer:
        commands_run.append("venv/bin/python -u utilities/tests/manual/evaluate_hh_transfer_on_code_near_miss_balanced.py")
    commands_run.append("venv/bin/python -u utilities/tests/manual/summarize_code_near_miss_balancing.py")

    files_modified_or_created = [
        "shared/utilities/tests/manual/inspect_code_near_miss_enrichment10_balance.py",
        "shared/utilities/tests/manual/generate_code_near_miss_balancing_candidates.py",
        "shared/utilities/tests/manual/evaluate_code_near_miss_balancing.py",
        "shared/utilities/tests/manual/capture_code_near_miss_balanced_tap_features.py",
        "shared/utilities/tests/manual/evaluate_hh_transfer_on_code_near_miss_balanced.py",
        "shared/utilities/tests/manual/summarize_code_near_miss_balancing.py",
        repo_path(INSPECTION_JSON),
        "opi/taps/probes/code_branch_near_miss_balance_inspection_2026-05-17.md",
        repo_path(GEN_JSON),
        "opi/taps/probes/code_branch_near_miss_balancing_candidates_2026-05-17.md",
        "opi/taps/probes/code_branch_near_miss_balancing_candidates_2026-05-17.log",
        repo_path(TOURN_JSON),
        "opi/taps/probes/code_branch_near_miss_balanced_tournaments_2026-05-17.md",
    ]
    if feature:
        files_modified_or_created.extend([
            repo_path(FEATURE_JSON),
            "opi/taps/probes/code_branch_near_miss_balanced_tap_features_2026-05-17.pt",
            "opi/taps/probes/code_branch_near_miss_balanced_tap_features_2026-05-17.md",
        ])
    if transfer:
        files_modified_or_created.extend([
            repo_path(TRANSFER_JSON),
            "opi/taps/probes/code_branch_near_miss_balanced_transfer_2026-05-17.md",
        ])
    files_modified_or_created.extend([
        "opi/taps/probes/code_branch_near_miss_balancing_2026-05-17_summary.json",
        "opi/taps/probes/code_branch_near_miss_balancing_2026-05-17_summary.md",
        "docs/evaluator/evaluator_domain_transfer_notes.md",
        "docs/evaluator/math_bg_gate_pilot_2026-05-15.md",
        "docs/evaluator/post_v10_synthesis_2026-05-15_v4.md",
        "docs/evaluator/post_v10_synthesis_2026-05-16_v5_clean_gsm8k_code_next.md",
    ])

    summary = {
        "BALANCE_INSPECTION_VERDICT": inspection.get("balance_inspection_verdict", "BLOCKED"),
        "BALANCING_GENERATION_VERDICT": gen.get("balancing_generation_verdict", "NOT_RUN"),
        "BALANCED_TOURNAMENT_VERDICT": tournament_verdict,
        "BALANCED_FEATURE_VERDICT": feature_verdict,
        "BALANCED_TRANSFER_VERDICT": transfer_verdict,
        "RECOMMENDED_NEXT": recommend(tournament_verdict, transfer_verdict),
        "tasks": after.get("tasks", inspection.get("summary", {}).get("tasks", 0)),
        "old_strict_clean": before.get("strict_clean", inspection.get("summary", {}).get("existing_strict_clean", 0)),
        "new_strict_clean": after.get("strict_clean", 0),
        "diagnostic_runnable": after.get("diagnostic_runnable", 0),
        "old_label_counts": before.get("label_counts", {}),
        "new_label_counts": after.get("label_counts", {}),
        "plausible_target_tasks": inspection.get("summary", {}).get("plausible_target_tasks", 0),
        "bucket_counts": inspection.get("summary", {}).get("bucket_counts", {}),
        "new_unique_candidates": gen.get("summary", {}).get("new_unique_candidates", 0),
        "new_duplicate_candidates": gen.get("summary", {}).get("new_duplicate_candidates", 0),
        "generation_errors": gen.get("summary", {}).get("generation_errors", 0),
        "new_generation_label_counts": gen.get("summary", {}).get("new_label_counts", {}),
        "tasks_converted_to_strict_clean": [row.get("task_id") for row in converted],
        "modes_that_helped": modes_that_helped,
        "feature_path": feature.get("output", "NOT_RUN"),
        "random_top1_baseline": transfer.get("random_top1_baseline", "NOT_RUN"),
        "best_antisymlinear": transfer.get("best_antisymlinear_compact", row_compact(transfer.get("best_antisymlinear"))),
        "best_nonorm": transfer.get("best_nonorm_compact", row_compact(transfer.get("best_nonorm"))),
        "interpretation": (
            "The targeted pass converted the enrichment set to the requested strict-clean threshold."
            if tournament_verdict == "GREEN"
            else "The targeted pass improved pairing but did not reach the strict-clean threshold."
            if tournament_verdict == "YELLOW"
            else "The bottleneck remains task/test design or missing-side generation reliability."
        ),
        "files_modified_or_created": files_modified_or_created,
        "commands_run": commands_run,
        "blockers": "" if tournament_verdict in {"GREEN", "YELLOW"} else tourn.get("recommended_next_if_stopped", ""),
    }
    summary["docs_updated"] = append_docs(summary)
    write_json(SUMMARY_JSON, summary)
    write_md(SUMMARY_MD, summary)
    print(f"BALANCED_TOURNAMENT_VERDICT = {tournament_verdict}")
    print(f"BALANCED_TRANSFER_VERDICT = {transfer_verdict}")
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
