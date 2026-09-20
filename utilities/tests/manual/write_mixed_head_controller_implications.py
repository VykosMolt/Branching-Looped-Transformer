"""Write controller implications and final summary for mixed-domain heads."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from code_branch_pilot_lib import PROJECT_ROOT, REPORT_DIR, output_path, repo_path, write_json  # noqa: E402


SPLITS_JSON = REPORT_DIR / "mixed_tap_domain_splits_2026-05-17.json"
FEATURES_JSON = REPORT_DIR / "mixed_tap_features_2026-05-17.json"
TRAIN_JSON = REPORT_DIR / "mixed_domain_tiny_heads_2026-05-17.json"
EVAL_JSON = REPORT_DIR / "mixed_domain_head_evaluation_2026-05-17.json"
LAYER_JSON = REPORT_DIR / "mixed_head_layer_choice_analysis_2026-05-17.json"
OUTPUT_JSON = REPORT_DIR / "mixed_head_controller_implications_2026-05-17.json"
OUTPUT_MD = REPORT_DIR / "mixed_head_controller_implications_2026-05-17.md"
SUMMARY_JSON = REPORT_DIR / "mixed_domain_heads_audit_2026-05-17_summary.json"
SUMMARY_MD = REPORT_DIR / "mixed_domain_heads_audit_2026-05-17_summary.md"

DOCS_TO_APPEND = [
    PROJECT_ROOT / "docs/evaluator/current_state.md",
    PROJECT_ROOT / "docs/evaluator/domain_transfer_ledger.md",
    PROJECT_ROOT / "docs/evaluator/evaluator_domain_transfer_notes.md",
    PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-17_v7_actual_state_and_next.md",
    PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-16_v5_clean_gsm8k_code_next.md",
    PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-15_v4.md",
    PROJECT_ROOT / "docs/evaluator/history/current_state_snapshot_2026-05-17.md",
    PROJECT_ROOT / "docs/evaluator/history/clean_gsm8k_and_code_next_2026-05-16.md",
    PROJECT_ROOT / "docs/evaluator/history/handoff_after_v4_2026-05-17.md",
    PROJECT_ROOT / "docs/evaluator/history/antisymlinear_pivot_2026-05-15.md",
    PROJECT_ROOT / "docs/evaluator/raw_archive_2026-05-18/evaluator_domain_transfer_notes.md",
    PROJECT_ROOT / "docs/evaluator/raw_archive_2026-05-18/post_v10_synthesis_2026-05-17_v7_actual_state_and_next.md",
    PROJECT_ROOT / "docs/evaluator/raw_archive_2026-05-18/post_v10_synthesis_2026-05-16_v5_clean_gsm8k_code_next.md",
    PROJECT_ROOT / "docs/evaluator/raw_archive_2026-05-18/post_v10_synthesis_2026-05-15_v4.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", default=str(SPLITS_JSON))
    parser.add_argument("--features-report", default=str(FEATURES_JSON))
    parser.add_argument("--training", default=str(TRAIN_JSON))
    parser.add_argument("--evaluation", default=str(EVAL_JSON))
    parser.add_argument("--layer-analysis", default=str(LAYER_JSON))
    parser.add_argument("--output", default=str(OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    parser.add_argument("--summary", default=str(SUMMARY_JSON))
    parser.add_argument("--summary-md", default=str(SUMMARY_MD))
    return parser.parse_args()


def load_json(path: str | Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    p = output_path(path)
    if not p.exists():
        return default or {}
    return json.loads(p.read_text(encoding="utf-8"))


def safe_rate(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return "NA"
    if math.isnan(x):
        return "NA"
    return f"{x:.3f}"


def recommended_next(utility: str) -> str:
    if utility == "OBJECTIVE_MIXED_USEFUL":
        return "add_objective_mixed_head_to_BG_phase1_and_run_controller_policy_simulator"
    if utility == "MIXED_DILUTES_SPECIALISTS":
        return "keep_HH_general_plus_code_specialist_and_do_not_use_mixed_as_default"
    if utility == "MIXED_HH_OBJECTIVE_USEFUL":
        return "consider_HH_objective_mixed_generalist_but_validate_on_larger_HH_and_code"
    if utility == "SPECIALISTS_WIN":
        return "keep_specialist_registry_and_move_to_controller_policy_simulator"
    return "repair_mixed_training_artifacts_before_architecture_decision"


def phase1_head_set(utility: str, strict_status: str) -> str:
    if utility == "OBJECTIVE_MIXED_USEFUL":
        return "HH_general_plus_code_specialist_plus_objective_mixed_head"
    if utility == "MIXED_HH_OBJECTIVE_USEFUL" and strict_status != "CLEAN_LOSS":
        return "HH_objective_mixed_generalist_plus_code_specialist"
    if utility in {"SPECIALISTS_WIN", "MIXED_DILUTES_SPECIALISTS"}:
        return "HH_general_plus_code_specialist_only"
    return "insufficient"


def best_mixed_family(eval_report: dict[str, Any]) -> str:
    regrets = eval_report.get("summary", {}).get("regret_summary", {}).get("per_eval", {})
    counts: dict[str, int] = {}
    pairwise: dict[str, float] = {}
    for item in regrets.values():
        row = item.get("best_mixed")
        if not isinstance(row, dict):
            continue
        fam = str(row.get("head_group", "unknown"))
        counts[fam] = counts.get(fam, 0) + 1
        pairwise[fam] = pairwise.get(fam, 0.0) + float(row.get("pairwise", 0.0))
    if not counts:
        return "NA"
    return max(counts, key=lambda fam: (counts[fam], pairwise[fam] / max(counts[fam], 1)))


def write_implications_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Mixed Head Controller Implications (2026-05-17)",
        "",
        f"Should there be an objective mixed head? {payload['controller_implications']['objective_mixed_head']}",
        f"Should the code specialist remain separate? {payload['controller_implications']['code_specialist_separate']}",
        f"Should HH/general remain separate? {payload['controller_implications']['hh_general_separate']}",
        f"Recommended Phase 1 head set: {payload['controller_implications']['recommended_phase1_head_set']}",
        "",
        "## Routing",
        "- HH/preference/unknown inputs route to the HH/general head.",
        "- Strict-clean code or high local code similarity routes to the code specialist.",
        "- Objective MCQ/science/reasoning/GSM8K routes to the objective mixed head only if the utility verdict supports it; otherwise use the HH/general head with specialist disagreement checks.",
        "- Low margin or head disagreement should defer to rollout/test/controller simulation.",
        "",
        "## Open Items",
        "- generated hard reasoning near-misses",
        "- larger strict-clean code eval",
        "- full HH split if needed",
        "- controller-policy simulator",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    top = payload["top_lines"]
    lines = [
        "# Mixed-Domain Heads Audit Summary (2026-05-17)",
        "",
        f"MIXED_TAP_SPLIT_VERDICT = {top['MIXED_TAP_SPLIT_VERDICT']}",
        f"MIXED_TAP_FEATURE_VERDICT = {top['MIXED_TAP_FEATURE_VERDICT']}",
        f"MIXED_TAP_TRAINING_VERDICT = {top['MIXED_TAP_TRAINING_VERDICT']}",
        f"MIXED_HEAD_UTILITY_VERDICT = {top['MIXED_HEAD_UTILITY_VERDICT']}",
        f"STRICT_CLEAN_CODE_REGRET_STATUS = {top['STRICT_CLEAN_CODE_REGRET_STATUS']}",
        f"SMALL_DOMAIN_OVERFIT = {top['SMALL_DOMAIN_OVERFIT']}",
        f"DOMAIN_OVERFIT_WARNING = {top['DOMAIN_OVERFIT_WARNING']}",
        f"GSM8K_EVAL_STATUS = {top['GSM8K_EVAL_STATUS']}",
        f"RECOMMENDED_PHASE1_HEAD_SET = {top['RECOMMENDED_PHASE1_HEAD_SET']}",
        f"RECOMMENDED_NEXT = {top['RECOMMENDED_NEXT']}",
        "",
        "## Split Construction",
        payload["sections"]["split_construction"],
        "",
        "## Feature Coverage",
        payload["sections"]["feature_coverage"],
        "",
        "## Mixed Head Training",
        payload["sections"]["mixed_head_training"],
        "",
        "## Domain Balancing And Overfit",
        payload["sections"]["domain_balancing"],
        "",
        "## Cross-Domain Evaluation",
        payload["sections"]["cross_domain_evaluation"],
        "",
        "## Regret Tables",
        payload["sections"]["regret_tables"],
        "",
        "## Borderline Regret Interpretation",
        "Borderline regret at current n should be treated as directionally informative, not decisive.",
        "",
        "## NoNorm vs AntisymLinear",
        payload["sections"]["nonorm_vs_antisym"],
        "",
        "## Layer/Config Winners",
        payload["sections"]["layer_config_winners"],
        "",
        "## Controller Implications",
        payload["sections"]["controller_implications"],
        "",
        "## Docs Updated",
    ]
    lines.extend(f"- `{path}`" for path in payload["docs_updated"])
    lines.extend(["", "## Files Modified / Created"])
    lines.extend(f"- `{path}`" for path in payload["files_modified_or_created"])
    lines.extend(["", "## Commands Run"])
    lines.extend(f"- `{cmd}`" for cmd in payload["commands_run"])
    lines.extend(["", "## Blockers"])
    lines.extend(f"- {blocker}" for blocker in payload["blockers"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(section: str) -> list[str]:
    updated = []
    for path in DOCS_TO_APPEND:
        if not path.exists():
            continue
        existing = path.read_text(encoding="utf-8")
        if "## Mixed-domain tiny heads (2026-05-17)" in existing:
            updated.append(repo_path(path))
            continue
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n\n" + section.strip() + "\n")
        updated.append(repo_path(path))
    return updated


def main() -> None:
    args = parse_args()
    splits = load_json(args.splits)
    features = load_json(args.features_report)
    training = load_json(args.training)
    eval_report = load_json(args.evaluation)
    layer = load_json(args.layer_analysis, default={"meta": {"mixed_head_layer_analysis_verdict": "NOT_RUN"}})

    split_verdict = splits.get("verdicts", {}).get("mixed_tap_split_verdict", "BLOCKED")
    feature_verdict = features.get("meta", {}).get("mixed_tap_feature_verdict", "BLOCKED")
    train_verdict = training.get("meta", {}).get("mixed_tap_training_verdict", "BLOCKED")
    utility = eval_report.get("meta", {}).get("mixed_head_utility_verdict", "INSUFFICIENT")
    strict_status = eval_report.get("meta", {}).get("strict_clean_code_regret_status", "NOT_APPLICABLE")
    utility_provisional = bool(eval_report.get("meta", {}).get("mixed_head_utility_provisional", False))
    small_overfit = bool(training.get("meta", {}).get("small_domain_overfit", False))
    domain_warning = bool(training.get("meta", {}).get("domain_overfit_warning", False))
    gsm8k_status = splits.get("gsm8k_eval_status", "NOT_FOUND")
    phase1 = phase1_head_set(utility, strict_status)
    next_step = recommended_next(utility)
    best_family = best_mixed_family(eval_report)

    regret = eval_report.get("summary", {}).get("regret_summary", {})
    per_eval = regret.get("per_eval", {})
    regret_lines = []
    for name, item in per_eval.items():
        regret_lines.append(
            f"{name}: pairwise {safe_rate(item.get('regret_pairwise'))}, top1 {safe_rate(item.get('regret_top1'))}"
        )
    best_eval_lines = []
    for name, row in eval_report.get("summary", {}).get("best_per_eval", {}).items():
        if isinstance(row, dict):
            best_eval_lines.append(f"{name}: {row.get('head_group')} {row.get('config')} {row.get('architecture')} pairwise={safe_rate(row.get('pairwise'))}")

    implication_payload = {
        "meta": {
            "mixed_head_utility_verdict": utility,
            "mixed_head_utility_provisional": utility_provisional,
            "strict_clean_code_regret_status": strict_status,
            "best_mixed_family": best_family,
        },
        "controller_implications": {
            "objective_mixed_head": "yes" if utility == "OBJECTIVE_MIXED_USEFUL" else ("provisional" if utility == "MIXED_HH_OBJECTIVE_USEFUL" else "no"),
            "code_specialist_separate": "yes" if "code_specialist" in phase1 else "provisional",
            "hh_general_separate": "yes" if utility != "MIXED_HH_OBJECTIVE_USEFUL" else "provisional",
            "recommended_phase1_head_set": phase1,
            "best_mixed_family": best_family,
            "recommended_next": next_step,
        },
        "open_items": [
            "generated hard reasoning near-misses",
            "larger strict-clean code eval",
            "full HH split if needed",
            "controller-policy simulator",
        ],
    }
    write_json(output_path(args.output), implication_payload)
    write_implications_md(output_path(args.output_md), implication_payload)

    doc_section = "\n".join(
        [
            "## Mixed-domain tiny heads (2026-05-17)",
            "",
            f"- MIXED_TAP_SPLIT_VERDICT = {split_verdict}",
            f"- MIXED_TAP_FEATURE_VERDICT = {feature_verdict}",
            f"- MIXED_TAP_TRAINING_VERDICT = {train_verdict}",
            f"- MIXED_HEAD_UTILITY_VERDICT = {utility}",
            f"- MIXED_HEAD_UTILITY_PROVISIONAL = {utility_provisional}",
            f"- STRICT_CLEAN_CODE_REGRET_STATUS = {strict_status}",
            f"- SMALL_DOMAIN_OVERFIT = {small_overfit}",
            f"- DOMAIN_OVERFIT_WARNING = {domain_warning}",
            f"- GSM8K_EVAL_STATUS = {gsm8k_status}",
            f"- best mixed family = {best_family}",
            f"- average objective regret pairwise = {safe_rate(regret.get('average_objective_regret_pairwise'))}",
            f"- worst objective regret pairwise = {safe_rate(regret.get('worst_objective_regret_pairwise'))}",
            f"- strict-clean code regret pairwise = {safe_rate(regret.get('strict_clean_code_regret_pairwise'))}",
            f"- reasoning trace regret pairwise = {safe_rate(regret.get('reasoning_trace_regret_pairwise'))}",
            f"- science medicine regret pairwise = {safe_rate(regret.get('science_medicine_regret_pairwise'))}",
            f"- clean GSM8K regret pairwise = {safe_rate(per_eval.get('CLEAN_GSM8K_EXPANDED', {}).get('regret_pairwise'))}",
            f"- HH regret pairwise = {safe_rate(regret.get('hh_regret_pairwise'))}",
            f"- recommended Phase 1 head set = {phase1}",
            f"- full reports: `{repo_path(SUMMARY_MD)}`, `{repo_path(EVAL_JSON)}`, `{repo_path(OUTPUT_MD)}`",
            "- interpretation: mixed heads are controller-routing candidates and should complement, not erase, the established HH/general and code-specialist roles unless regret is cleanly positive outside the current small strict-clean sample.",
        ]
    )
    docs_updated = append_docs(doc_section)

    files = [
        "shared/utilities/tests/manual/build_mixed_tap_domain_splits.py",
        "shared/utilities/tests/manual/ensure_mixed_tap_features.py",
        "shared/utilities/tests/manual/train_mixed_domain_tiny_heads.py",
        "shared/utilities/tests/manual/evaluate_mixed_domain_heads.py",
        "shared/utilities/tests/manual/analyze_mixed_head_layer_choices.py",
        "shared/utilities/tests/manual/write_mixed_head_controller_implications.py",
        repo_path(SPLITS_JSON),
        "opi/taps/probes/mixed_tap_domain_splits_2026-05-17.md",
        repo_path(FEATURES_JSON),
        "opi/taps/probes/mixed_tap_features_2026-05-17.md",
        "opi/taps/probes/mixed_tap_features_2026-05-17.pt",
        repo_path(TRAIN_JSON),
        "opi/taps/probes/mixed_domain_tiny_heads_2026-05-17.md",
        "opi/taps/probes/mixed_domain_tiny_heads_2026-05-17.pt",
        repo_path(EVAL_JSON),
        "opi/taps/probes/mixed_domain_head_evaluation_2026-05-17.md",
        repo_path(LAYER_JSON),
        "opi/taps/probes/mixed_head_layer_choice_analysis_2026-05-17.md",
        repo_path(OUTPUT_JSON),
        repo_path(OUTPUT_MD),
        repo_path(SUMMARY_JSON),
        repo_path(SUMMARY_MD),
    ]
    commands = [
        "venv/bin/python -m py_compile utilities/tests/manual/build_mixed_tap_domain_splits.py",
        "venv/bin/python -m py_compile utilities/tests/manual/ensure_mixed_tap_features.py",
        "venv/bin/python -m py_compile utilities/tests/manual/train_mixed_domain_tiny_heads.py",
        "venv/bin/python -m py_compile utilities/tests/manual/evaluate_mixed_domain_heads.py",
        "venv/bin/python -m py_compile utilities/tests/manual/analyze_mixed_head_layer_choices.py",
        "venv/bin/python -m py_compile utilities/tests/manual/write_mixed_head_controller_implications.py",
        "venv/bin/python -u utilities/tests/manual/build_mixed_tap_domain_splits.py",
        "venv/bin/python -u utilities/tests/manual/ensure_mixed_tap_features.py --splits opi/taps/probes/mixed_tap_domain_splits_2026-05-17.json --device cuda",
        "venv/bin/python -u utilities/tests/manual/train_mixed_domain_tiny_heads.py --splits opi/taps/probes/mixed_tap_domain_splits_2026-05-17.json --features opi/taps/probes/mixed_tap_features_2026-05-17.pt",
        "venv/bin/python -u utilities/tests/manual/evaluate_mixed_domain_heads.py --splits opi/taps/probes/mixed_tap_domain_splits_2026-05-17.json --features opi/taps/probes/mixed_tap_features_2026-05-17.pt --mixed-heads opi/taps/probes/mixed_domain_tiny_heads_2026-05-17.pt --registry opi/taps/probes/bg_head_registry_2026-05-17.pt",
        "venv/bin/python -u utilities/tests/manual/analyze_mixed_head_layer_choices.py",
        "venv/bin/python -u utilities/tests/manual/write_mixed_head_controller_implications.py",
    ]
    summary = {
        "top_lines": {
            "MIXED_TAP_SPLIT_VERDICT": split_verdict,
            "MIXED_TAP_FEATURE_VERDICT": feature_verdict,
            "MIXED_TAP_TRAINING_VERDICT": train_verdict,
            "MIXED_HEAD_UTILITY_VERDICT": utility,
            "STRICT_CLEAN_CODE_REGRET_STATUS": strict_status,
            "SMALL_DOMAIN_OVERFIT": small_overfit,
            "DOMAIN_OVERFIT_WARNING": domain_warning,
            "GSM8K_EVAL_STATUS": gsm8k_status,
            "RECOMMENDED_PHASE1_HEAD_SET": phase1,
            "RECOMMENDED_NEXT": next_step,
        },
        "sections": {
            "split_construction": f"Built splits for domains: {', '.join(sorted(splits.get('domains', {}).keys()))}. Split verdict {split_verdict}; GSM8K status {gsm8k_status}.",
            "feature_coverage": f"Feature verdict {feature_verdict}; coverage report `{repo_path(FEATURES_JSON)}`.",
            "mixed_head_training": f"Training verdict {train_verdict}; trained heads {training.get('meta', {}).get('trained_head_count', 'NA')}.",
            "domain_balancing": f"SMALL_DOMAIN_OVERFIT={small_overfit}; DOMAIN_OVERFIT_WARNING={domain_warning}. Equal-domain sampling diagnostics are in `{repo_path(TRAIN_JSON)}`.",
            "cross_domain_evaluation": "; ".join(best_eval_lines[:16]) if best_eval_lines else "No evaluation rows available.",
            "regret_tables": "; ".join(regret_lines) if regret_lines else "No regret rows available.",
            "nonorm_vs_antisym": f"Layer/config analysis verdict {layer.get('meta', {}).get('mixed_head_layer_analysis_verdict', 'NA')}; architecture winners `{layer.get('winner_counts', {}).get('architecture', {})}`.",
            "layer_config_winners": f"Config winners `{layer.get('winner_counts', {}).get('config', {})}`.",
            "controller_implications": f"Recommended Phase 1 head set: {phase1}; recommended next: {next_step}.",
        },
        "docs_updated": docs_updated,
        "files_modified_or_created": files,
        "commands_run": commands,
        "blockers": eval_report.get("blockers", []) or training.get("blockers", []) or ["none"],
    }
    write_json(output_path(args.summary), summary)
    write_summary_md(output_path(args.summary_md), summary)
    print(f"MIXED_HEAD_UTILITY_VERDICT = {utility}")
    print(f"RECOMMENDED_PHASE1_HEAD_SET = {phase1}")
    print(f"RECOMMENDED_NEXT = {next_step}")
    print(f"wrote {repo_path(output_path(args.summary))}")


if __name__ == "__main__":
    main()
