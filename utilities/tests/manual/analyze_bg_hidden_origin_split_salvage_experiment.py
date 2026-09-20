"""Synthesize hidden-origin split-salvage experiment and update docs."""
from __future__ import annotations

import time
from pathlib import Path

from bg_hidden_origin_split_salvage_common import (
    AUDIT_JSON,
    CV_STABILITY_JSON,
    EVAL_MODES_JSON,
    SALVAGE_DATASETS_PT,
    SALVAGE_EVAL_JSON,
    SALVAGE_HEADS_PT,
    SALVAGE_ROOT,
    V4_QUOTA_JSON,
    append_doc_section,
    ensure_salvage_root,
    load_json,
    load_pt,
    md_table,
    rel,
    write_json,
    write_md,
)
from bg_hidden_origin_tap_common import PROJECT_ROOT


OUT_SUMMARY_MD = SALVAGE_ROOT / "summary.md"
OUT_SUMMARY_JSON = SALVAGE_ROOT / "summary.json"
OUT_ANALYSIS_MD = SALVAGE_ROOT / "analysis.md"
OUT_ANALYSIS_JSON = SALVAGE_ROOT / "analysis.json"
DOC_MD = PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_split_salvage.md"
SECTION_TITLE = "Hidden-origin branch split salvage and selector reevaluation (2026-05-18)"


def phase_status(eval_verdict: str, v4_verdict: str, cv_verdict: str) -> str:
    if eval_verdict in {"SELECTOR_READY", "OLD_TAPS_BEST", "ENSEMBLE_BEST"}:
        return "READY"
    if eval_verdict == "WEAK_SELECTOR" or cv_verdict in {"STABLE_POSITIVE", "WEAK_POSITIVE"}:
        return "WEAK"
    if v4_verdict == "V4_REQUIRED_HELDOUT_BALANCE":
        return "NEEDS_QUOTA_V4"
    if v4_verdict == "V4_REQUIRED_BRANCH_GENERATOR":
        return "NEEDS_BETTER_BRANCH_GENERATOR"
    if eval_verdict == "NO_SELECTOR_SIGNAL":
        return "NOT_READY"
    if eval_verdict == "STILL_DATA_LIMITED":
        return "STILL_DATA_LIMITED"
    return "INSUFFICIENT"


def command_list() -> list[str]:
    scripts = [
        "bg_hidden_origin_split_salvage_audit.py",
        "bg_hidden_origin_define_eval_modes.py",
        "build_bg_hidden_origin_salvage_datasets.py",
        "train_bg_hidden_origin_salvage_heads.py",
        "evaluate_bg_hidden_origin_salvage_selectors.py",
        "analyze_bg_hidden_origin_salvage_cv.py",
        "analyze_bg_hidden_origin_v4_quota_need.py",
        "analyze_bg_hidden_origin_split_salvage_experiment.py",
    ]
    return [f"venv/bin/python -m py_compile utilities/tests/manual/{script}" for script in scripts] + [
        f"venv/bin/python -u utilities/tests/manual/{script}" for script in scripts
    ]


def main() -> int:
    started = time.time()
    ensure_salvage_root()
    audit = load_json(AUDIT_JSON, {}) or {}
    modes = load_json(EVAL_MODES_JSON, {}) or {}
    dataset = load_pt(SALVAGE_DATASETS_PT, {}) or {}
    training = load_pt(SALVAGE_HEADS_PT, {}) or {}
    selector_eval = load_json(SALVAGE_EVAL_JSON, {}) or {}
    cv = load_json(CV_STABILITY_JSON, {}) or {}
    v4 = load_json(V4_QUOTA_JSON, {}) or {}

    audit_verdict = str(audit.get("verdict", "INSUFFICIENT"))
    modes_verdict = str(modes.get("verdict", "INSUFFICIENT"))
    dataset_verdict = str(dataset.get("verdict", "INSUFFICIENT"))
    training_verdict = str(training.get("verdict", "INSUFFICIENT"))
    eval_verdict = str(selector_eval.get("verdict", "INSUFFICIENT"))
    cv_verdict = str(cv.get("verdict", "INSUFFICIENT"))
    v4_verdict = str(v4.get("verdict", "INSUFFICIENT"))
    best_selector = str(selector_eval.get("HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE") or "insufficient")
    phase = phase_status(eval_verdict, v4_verdict, cv_verdict)

    top_lines = {
        "BG_HIDDEN_ORIGIN_SPLIT_AUDIT_VERDICT": audit_verdict,
        "BG_HIDDEN_ORIGIN_EVAL_MODE_VERDICT": modes_verdict,
        "BG_HIDDEN_ORIGIN_SALVAGE_DATASET_VERDICT": dataset_verdict,
        "BG_HIDDEN_ORIGIN_SALVAGE_TRAINING_VERDICT": training_verdict,
        "BG_HIDDEN_ORIGIN_SALVAGE_EVAL_VERDICT": eval_verdict,
        "BG_HIDDEN_ORIGIN_CV_STABILITY_VERDICT": cv_verdict,
        "BG_HIDDEN_ORIGIN_V4_QUOTA_NEED_VERDICT": v4_verdict,
        "HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE": best_selector,
        "PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_SALVAGE": phase,
    }
    strict_support = (audit.get("strict_split_bottleneck") or {}).get("support") or {}
    all_signal = audit.get("all_reward_signal_support") or {}
    combined = (audit.get("version_summary") or {}).get("combined") or {}
    best_policy = cv.get("best_policy_summary")
    payload = {
        **top_lines,
        "strict_support": strict_support,
        "all_signal_support": all_signal,
        "combined_summary": combined,
        "best_cv_policy_summary": best_policy,
        "v4_decision": v4.get("decision"),
        "files_created": [
            rel(path)
            for path in [
                SALVAGE_ROOT / "audit.md",
                AUDIT_JSON,
                SALVAGE_ROOT / "task_distribution.csv",
                SALVAGE_ROOT / "eval_modes.md",
                EVAL_MODES_JSON,
                SALVAGE_DATASETS_PT,
                SALVAGE_ROOT / "salvage_datasets.json",
                SALVAGE_ROOT / "salvage_datasets.md",
                SALVAGE_HEADS_PT,
                SALVAGE_ROOT / "salvage_training_log.json",
                SALVAGE_ROOT / "salvage_training_report.md",
                SALVAGE_ROOT / "salvage_selector_eval.md",
                SALVAGE_EVAL_JSON,
                SALVAGE_ROOT / "salvage_selector_eval_rows.csv",
                SALVAGE_ROOT / "cv_stability.md",
                CV_STABILITY_JSON,
                SALVAGE_ROOT / "v4_quota_need.md",
                V4_QUOTA_JSON,
                OUT_SUMMARY_MD,
                OUT_SUMMARY_JSON,
                OUT_ANALYSIS_MD,
                OUT_ANALYSIS_JSON,
                DOC_MD,
            ]
        ],
        "commands_run": command_list(),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_SUMMARY_JSON, payload)
    write_json(OUT_ANALYSIS_JSON, payload)

    lines = ["# Hidden-Origin Split Salvage Summary", ""]
    lines.extend(f"{key} = {value}" for key, value in top_lines.items())
    lines.extend(
        [
            "",
            "## Motivation",
            "",
            "V3 improved hidden-origin branch diversity, but the strict clean heldout split had too few non-tie reward pairs to support selector readiness. This salvage run reused existing branch rows only.",
            "",
            "## V3 Data Distribution",
            "",
            f"- combined behaviorally diverse groups: `{combined.get('behaviorally_diverse_groups')}`",
            f"- combined non-tie pairs: `{combined.get('non_tie_pairs')}`",
            f"- combined tie rate: `{combined.get('tie_rate')}`",
            f"- all reward-signal support tasks: `{all_signal.get('support_task_count')}`",
            f"- all reward-signal behaviorally diverse groups: `{all_signal.get('behaviorally_diverse_groups')}`",
            f"- all reward-signal non-tie pairs: `{all_signal.get('non_tie_pairs')}`",
            "",
            "## Evaluation Modes",
            "",
            f"- eval mode verdict: `{modes_verdict}`",
            f"- strict clean support tasks: `{strict_support.get('support_task_count')}`",
            f"- strict clean behaviorally diverse groups: `{strict_support.get('behaviorally_diverse_groups')}`",
            f"- strict clean non-tie pairs: `{strict_support.get('non_tie_pairs')}`",
            "",
            "## Split-Specific Datasets",
            "",
            f"- dataset verdict: `{dataset_verdict}`",
            f"- readiness minimums: `{modes.get('readiness_minimums')}`",
            "",
            "## Retraining Under Salvage Splits",
            "",
            f"- training verdict: `{training_verdict}`",
            f"- trained heads: `{training.get('trained_heads')}`",
            f"- trainable folds: `{training.get('trainable_folds')}`",
            "",
            "## Selector Evaluation",
            "",
            f"- selector eval verdict: `{eval_verdict}`",
            f"- best selector available: `{best_selector}`",
            "",
            "## CV Stability",
            "",
            f"- cv stability verdict: `{cv_verdict}`",
            f"- best cv policy summary: `{best_policy}`",
            "",
            "## V4 Quota Decision",
            "",
            f"- v4 quota verdict: `{v4_verdict}`",
            f"- decision: `{v4.get('decision')}`",
            "",
            "## Phase 2 Readiness",
            "",
            f"PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_SALVAGE = `{phase}`",
            "",
            "## Files Created",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in payload["files_created"])
    lines.extend(["", "## Commands Run", ""])
    lines.extend(f"- `{cmd}`" for cmd in payload["commands_run"])
    blockers = []
    if strict_support.get("non_tie_pairs", 0) < 80:
        blockers.append("strict/v3-clean heldout still lacks enough non-tie reward pairs")
    if phase in {"NEEDS_QUOTA_V4", "STILL_DATA_LIMITED"}:
        blockers.append("selector readiness cannot be claimed without heldout-balanced quota data")
    lines.extend(["", "## Blockers", ""])
    lines.extend(f"- {item}" for item in blockers or ["none"])
    write_md(OUT_SUMMARY_MD, lines)
    write_md(OUT_ANALYSIS_MD, lines)

    doc_lines = [
        "# Hidden-Origin Branch Split Salvage",
        "",
        *[f"{key} = {value}" for key, value in top_lines.items()],
        "",
        "## Why Salvage Was Needed",
        "",
        "The v3 strict heldout set had only one task with non-tie reward pairs, despite v3 improving overall diversity. The salvage run tested whether alternate leakage-aware task splits and grouped CV could establish selector readiness without new branch generation.",
        "",
        "## Leakage Handling",
        "",
        "Strict and v3-clean modes preserve the original v3 empirical-direction heldout guard. Grouped CV and leave-one-task-out are task-disjoint for retrained tiny heads but are marked diagnostic when original empirical direction construction may have seen the heldout task.",
        "",
        "## Result",
        "",
        f"The salvage status is `{phase}` with best available selector `{best_selector}` and v4 quota verdict `{v4_verdict}`.",
        "",
        "## Phase 2 Implication",
        "",
        "Do not claim action steering or selector readiness unless the clean heldout support and selector comparisons meet the readiness thresholds. If quota v4 is required, generate by pre-reserved split quotas instead of generate-then-split.",
    ]
    write_md(DOC_MD, doc_lines)

    append_lines = [
        f"## {SECTION_TITLE}",
        "",
        f"- `BG_HIDDEN_ORIGIN_SALVAGE_EVAL_VERDICT = {eval_verdict}`",
        f"- `BG_HIDDEN_ORIGIN_CV_STABILITY_VERDICT = {cv_verdict}`",
        f"- `BG_HIDDEN_ORIGIN_V4_QUOTA_NEED_VERDICT = {v4_verdict}`",
        f"- `HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE = {best_selector}`",
        f"- `PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_SALVAGE = {phase}`",
        "",
        "Split salvage reused existing v3 branch data only. It reports strict/v3-clean heldout support separately from grouped-CV diagnostics and marks baseline contamination where applicable.",
    ]
    append_targets = [
        PROJECT_ROOT / "docs/evaluator/current_state.md",
        PROJECT_ROOT / "docs/evaluator/domain_transfer_ledger.md",
        PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_diversity_v3.md",
        PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_diversity_v2.md",
        PROJECT_ROOT / "docs/evaluator/bg_hidden_origin_taps.md",
        PROJECT_ROOT / "docs/evaluator/bg_hidden_state_branch_generation.md",
        PROJECT_ROOT / "docs/evaluator/bg_steering_consolidation_2026-05-18.md",
        PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-18_v8.1_routing_locked.md",
    ]
    for path in append_targets:
        append_doc_section(path, SECTION_TITLE, append_lines)

    print(f"PHASE2_HIDDEN_BRANCH_EVALUATOR_STATUS_SALVAGE = {phase}", flush=True)
    print(f"HIDDEN_ORIGIN_SELECTOR_BEST_AVAILABLE = {best_selector}", flush=True)
    print(f"Wrote {rel(OUT_SUMMARY_MD)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

