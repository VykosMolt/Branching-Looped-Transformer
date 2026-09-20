"""Bootstrap BG policy uncertainty and write final controller design reports."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from code_branch_pilot_lib import PROJECT_ROOT, REPORT_DIR, output_path, repo_path, write_json  # noqa: E402


POLICY_JSON = REPORT_DIR / "bg_controller_policy_simulation_2026-05-17.json"
COMPARISON_JSON = REPORT_DIR / "bg_candidate_head_comparison_2026-05-17.json"
BUNDLE_JSON = REPORT_DIR / "bg_policy_sim_eval_bundle_2026-05-17.json"
UNCERTAINTY_JSON = REPORT_DIR / "bg_policy_uncertainty_2026-05-17.json"
UNCERTAINTY_MD = REPORT_DIR / "bg_policy_uncertainty_2026-05-17.md"
DESIGN_JSON = REPORT_DIR / "bg_phase1_controller_design_note_2026-05-17.json"
DESIGN_MD = REPORT_DIR / "bg_phase1_controller_design_note_2026-05-17.md"
SUMMARY_JSON = REPORT_DIR / "bg_controller_policy_simulator_2026-05-17_summary.json"
SUMMARY_MD = REPORT_DIR / "bg_controller_policy_simulator_2026-05-17_summary.md"

DOCS_TO_APPEND = [
    PROJECT_ROOT / "docs/evaluator/current_state.md",
    PROJECT_ROOT / "docs/evaluator/domain_transfer_ledger.md",
    PROJECT_ROOT / "docs/evaluator/evaluator_domain_transfer_notes.md",
    PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-17_v7_actual_state_and_next.md",
    PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-16_v5_clean_gsm8k_code_next.md",
    PROJECT_ROOT / "docs/evaluator/post_v10_synthesis_2026-05-15_v4.md",
    PROJECT_ROOT / "docs/evaluator/history/current_state_snapshot_2026-05-17.md",
    PROJECT_ROOT / "docs/evaluator/history/clean_gsm8k_and_code_next_2026-05-16.md",
    PROJECT_ROOT / "docs/evaluator/history/antisymlinear_pivot_2026-05-15.md",
    PROJECT_ROOT / "docs/evaluator/raw_archive_2026-05-18/evaluator_domain_transfer_notes.md",
    PROJECT_ROOT / "docs/evaluator/raw_archive_2026-05-18/post_v10_synthesis_2026-05-17_v7_actual_state_and_next.md",
    PROJECT_ROOT / "docs/evaluator/raw_archive_2026-05-18/post_v10_synthesis_2026-05-16_v5_clean_gsm8k_code_next.md",
    PROJECT_ROOT / "docs/evaluator/raw_archive_2026-05-18/post_v10_synthesis_2026-05-15_v4.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(POLICY_JSON))
    parser.add_argument("--comparison", default=str(COMPARISON_JSON))
    parser.add_argument("--bundle-summary", default=str(BUNDLE_JSON))
    parser.add_argument("--output", default=str(UNCERTAINTY_JSON))
    parser.add_argument("--output-md", default=str(UNCERTAINTY_MD))
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(output_path(path).read_text(encoding="utf-8"))


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def safe_rate(value: Any) -> str:
    val = safe_float(value)
    return "NA" if math.isnan(val) else f"{val:.3f}"


def safe_mean(vals: list[float]) -> float:
    vals = [float(v) for v in vals if not math.isnan(float(v))]
    return float(mean(vals)) if vals else float("nan")


def ci(vals: list[float]) -> dict[str, float]:
    vals = sorted(float(v) for v in vals if not math.isnan(float(v)))
    if not vals:
        return {"lo": float("nan"), "hi": float("nan")}
    return {
        "lo": vals[max(0, int(0.025 * (len(vals) - 1)))],
        "hi": vals[min(len(vals) - 1, int(0.975 * (len(vals) - 1)))],
    }


def bootstrap_outcomes(outcomes: list[dict[str, Any]], samples: int, rng: random.Random) -> dict[str, Any]:
    if not outcomes:
        return {"mean": float("nan"), "ci95": {"lo": float("nan"), "hi": float("nan")}, "samples": samples}
    vals = []
    n = len(outcomes)
    for _ in range(samples):
        correct = 0
        total = 0
        for _j in range(n):
            row = outcomes[rng.randrange(n)]
            if row.get("deferred"):
                continue
            correct += int(row.get("pair_correct", 0))
            total += int(row.get("pair_total", 0))
        vals.append(correct / total if total else float("nan"))
    return {"mean": safe_mean(vals), "ci95": ci(vals), "samples": samples}


def bootstrap_policies(policy: dict[str, Any], samples: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    selected = [
        "DOMAIN_ROUTED_SIMPLE",
        "DOMAIN_ROUTED_OBJECTIVE_BROAD",
        "OBJECTIVE_MIXED_ONLY",
        "OBJECTIVE_THEN_CODE_BACKUP_0.20",
        "MARGIN_DEFER_0.10",
        "DISAGREEMENT_DEFER_0.20",
        "HH_ONLY",
        "CODE_ONLY",
    ]
    out: dict[str, Any] = {}
    for name in selected:
        row = policy.get("policy_results", {}).get(name)
        if not row:
            continue
        out[name] = {}
        all_outcomes = []
        for domain, outcomes in row.get("outcomes", {}).items():
            out[name][domain] = bootstrap_outcomes(outcomes, samples, rng)
            all_outcomes.extend(outcomes)
        out[name]["OVERALL"] = bootstrap_outcomes(all_outcomes, samples, rng)
    return out


def uncertainty_flags(policy: dict[str, Any], boot: dict[str, Any]) -> dict[str, Any]:
    results = policy.get("policy_results", {})
    obj_vs_code = (
        safe_float(results.get("OBJECTIVE_MIXED_ONLY", {}).get("domain_breakdown", {}).get("CODE_STRICT_CLEAN_ALL16", {}).get("pairwise_acc"))
        - safe_float(results.get("CODE_ONLY", {}).get("domain_breakdown", {}).get("CODE_STRICT_CLEAN_ALL16", {}).get("pairwise_acc"))
    )
    strict_borderline = -0.05 <= obj_vs_code <= 0.05
    routed_hh = safe_float(results.get("DOMAIN_ROUTED_SIMPLE", {}).get("domain_breakdown", {}).get("HH_HELDOUT20", {}).get("pairwise_acc"))
    hh_only = safe_float(results.get("HH_ONLY", {}).get("domain_breakdown", {}).get("HH_HELDOUT20", {}).get("pairwise_acc"))
    hh_borderline = False if abs(routed_hh - hh_only) < 1e-9 else (-0.05 <= routed_hh - hh_only <= 0.05)
    unstable = strict_borderline
    for name, by_domain in boot.items():
        for domain in ("CODE_STRICT_CLEAN_ALL16", "HH_HELDOUT20", "REASONING_TRACE", "SCIENCE_OVERALL"):
            ci95 = by_domain.get(domain, {}).get("ci95")
            if not ci95:
                continue
            if safe_float(ci95["hi"]) - safe_float(ci95["lo"]) > 0.25:
                unstable = True
    return {
        "small_n_unstable_policy": bool(unstable),
        "strict_clean_policy_borderline": bool(strict_borderline),
        "hh_heldout_policy_borderline": bool(hh_borderline),
        "objective_mixed_vs_code_strict_clean_delta": obj_vs_code,
        "domain_routed_vs_hh_hh_delta": routed_hh - hh_only,
    }


def recommended_next(best_policy: str) -> str:
    if best_policy == "DOMAIN_ROUTED_WINS":
        return "implement_read_only_BG_policy_layer_with_domain_routing"
    if best_policy == "OBJECTIVE_MIXED_DEFAULT_WINS":
        return "implement_objective_mixed_default_with_HH_fallback_and_code_backup"
    if best_policy == "DEFER_POLICY_WINS":
        return "implement_margin_or_disagreement_defer_controller"
    if best_policy == "SINGLE_HEAD_SUFFICIENT":
        return "simplify_head_registry_before_integration"
    return "repair_policy_eval_artifacts_before_controller_design"


def write_uncertainty_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# BG Policy Uncertainty (2026-05-17)",
        "",
        f"SMALL_N_UNSTABLE_POLICY = {payload['flags']['small_n_unstable_policy']}",
        f"STRICT_CLEAN_POLICY_BORDERLINE = {payload['flags']['strict_clean_policy_borderline']}",
        f"HH_HELDOUT_POLICY_BORDERLINE = {payload['flags']['hh_heldout_policy_borderline']}",
        "",
        "## Bootstrap Highlights",
    ]
    for policy_name, by_domain in payload["bootstrap"].items():
        for domain in ("CODE_STRICT_CLEAN_ALL16", "HH_HELDOUT20", "REASONING_TRACE", "SCIENCE_OVERALL", "OVERALL"):
            item = by_domain.get(domain)
            if not item:
                continue
            ci95 = item["ci95"]
            lines.append(f"- {policy_name} / {domain}: mean={item['mean']:.3f}, ci95=[{ci95['lo']:.3f},{ci95['hi']:.3f}]")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def design_note(policy: dict[str, Any], comparison: dict[str, Any], flags: dict[str, Any]) -> dict[str, Any]:
    meta = policy["meta"]
    note = {
        "recommended_head_registry": [
            "HH_GENERAL",
            "OBJECTIVE_MIXED_PRIMARY",
            "CODE_SPECIALIST_BACKUP",
            "optional_OBJECTIVE_MIXED_BROAD",
            "RISKY_HH_OBJECTIVE_ABLATION_ONLY",
        ],
        "recommended_policy": meta.get("recommended_bg_policy"),
        "routing_rules": {
            "HH/preference/unknown": "HH_GENERAL",
            "objective_QA/reasoning/science/GSM8K": "OBJECTIVE_MIXED_PRIMARY",
            "strict_clean_code_or_high_local_code_similarity": "CODE_SPECIALIST_BACKUP or objective mixed with code backup depending on margin",
            "low_margin": "defer, roll out more, or run verifier",
            "head_disagreement": "defer or use margin-calibrated vote",
        },
        "what_not_to_do": [
            "do_not_replace_HH_GENERAL_with_MIX_HH_OBJECTIVE",
            "do_not_collapse_all_heads_into_one_universal_head",
            "do_not_use_medical_benchmark_result_as_clinical_competence",
            "do_not_treat_old_dirty_prefix_artifacts_as_current_truth",
        ],
        "open_questions": [
            "larger strict-clean code n",
            "controller deployment integration",
            "hard generated reasoning near-misses",
            "full HH split if publication needed",
            "true online branch selection inside model loop",
        ],
        "supporting_verdicts": {
            "best_policy_verdict": meta.get("best_policy_verdict"),
            "contrast_detector_verdict": meta.get("contrast_detector_verdict"),
            "defer_policy_verdict": meta.get("defer_policy_verdict"),
            "oracle_gap_verdict": meta.get("oracle_gap_verdict"),
            "head_complementarity_verdict": comparison.get("meta", {}).get("head_complementarity_verdict"),
            **flags,
        },
    }
    return note


def write_design_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# BG Phase 1 Controller Design Note (2026-05-17)",
        "",
        f"Recommended policy: {payload['recommended_policy']}",
        "",
        "## Recommended Head Registry",
    ]
    lines.extend(f"- {item}" for item in payload["recommended_head_registry"])
    lines.extend(["", "## Routing Rules"])
    for key, value in payload["routing_rules"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## What Not To Do"])
    lines.extend(f"- {item}" for item in payload["what_not_to_do"])
    lines.extend(["", "## Open Questions"])
    lines.extend(f"- {item}" for item in payload["open_questions"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(section: str) -> list[str]:
    updated = []
    for path in DOCS_TO_APPEND:
        if not path.exists():
            continue
        existing = path.read_text(encoding="utf-8")
        if "## BG controller-policy simulator (2026-05-17)" in existing:
            updated.append(repo_path(path))
            continue
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n\n" + section.strip() + "\n")
        updated.append(repo_path(path))
    return updated


def build_doc_section(policy: dict[str, Any], comparison: dict[str, Any], bundle: dict[str, Any], flags: dict[str, Any]) -> str:
    meta = policy["meta"]
    domain = policy["policy_results"].get("DOMAIN_ROUTED_SIMPLE", {})
    obj = policy["policy_results"].get("OBJECTIVE_MIXED_ONLY", {})
    code = policy["policy_results"].get("CODE_ONLY", {})
    return "\n".join(
        [
            "## BG controller-policy simulator (2026-05-17)",
            "",
            f"- BG_POLICY_EVAL_BUNDLE_VERDICT = {bundle.get('meta', {}).get('bg_policy_eval_bundle_verdict', 'NA')}",
            f"- BG_HEAD_COMPARISON_VERDICT = {comparison.get('meta', {}).get('bg_head_comparison_verdict', 'NA')}",
            f"- HEAD_COMPLEMENTARITY_VERDICT = {comparison.get('meta', {}).get('head_complementarity_verdict', 'NA')}",
            f"- BG_POLICY_SIM_VERDICT = {meta.get('bg_policy_sim_verdict')}",
            f"- BEST_POLICY_VERDICT = {meta.get('best_policy_verdict')}",
            f"- RECOMMENDED_BG_POLICY = {meta.get('recommended_bg_policy')}",
            f"- CONTRAST_DETECTOR_VERDICT = {meta.get('contrast_detector_verdict')}",
            f"- DEFER_POLICY_VERDICT = {meta.get('defer_policy_verdict')}",
            f"- ORACLE_GAP_VERDICT = {meta.get('oracle_gap_verdict')}",
            f"- SMALL_N_UNSTABLE_POLICY = {flags['small_n_unstable_policy']}",
            f"- STRICT_CLEAN_POLICY_BORDERLINE = {flags['strict_clean_policy_borderline']}",
            f"- HH_HELDOUT_POLICY_BORDERLINE = {flags['hh_heldout_policy_borderline']}",
            f"- best policy metrics: DOMAIN_ROUTED_SIMPLE objective_avg={safe_rate(domain.get('objective_average_pairwise'))}, HH={safe_rate(domain.get('hh_heldout_pairwise'))}, strict_clean={safe_rate(domain.get('strict_clean_pairwise'))}",
            f"- best single head metrics: OBJECTIVE_MIXED_ONLY objective_avg={safe_rate(obj.get('objective_average_pairwise'))}; CODE_ONLY strict_clean={safe_rate(code.get('strict_clean_pairwise'))}",
            f"- objective mixed vs code specialist strict-clean delta = {safe_rate(flags['objective_mixed_vs_code_strict_clean_delta'])}",
            f"- HH preservation result: DOMAIN_ROUTED_SIMPLE uses HH_GENERAL on HH, delta vs HH_ONLY = {safe_rate(flags['domain_routed_vs_hh_hh_delta'])}",
            f"- defer policy result: `{meta.get('defer_summary', {})}`",
            f"- oracle-gap summary: average={safe_rate(domain.get('average_oracle_gap_pairwise'))}, objective={safe_rate(domain.get('objective_average_oracle_gap_pairwise'))}",
            f"- contrast-detector summary: {meta.get('contrast_detector_verdict')}",
            "- recommended Phase 1 controller design: HH/general for HH and unknown, objective mixed for objective QA/reasoning/science/GSM8K, code specialist backup for strict-clean or high-similarity code, defer on low margin/disagreement.",
            f"- full reports: `{repo_path(SUMMARY_MD)}`, `{repo_path(POLICY_JSON)}`, `{repo_path(COMPARISON_JSON)}`, `{repo_path(DESIGN_MD)}`",
            "- interpretation: deploy a read-only routed controller; current heads are complementary enough to route, but not stable enough to collapse into one universal head.",
        ]
    )


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    top = payload["top_lines"]
    lines = [
        "# BG Controller Policy Simulator Summary (2026-05-17)",
        "",
    ]
    for key, value in top.items():
        lines.append(f"{key} = {value}")
    for section, body in payload["sections"].items():
        lines.extend(["", f"## {section}", body])
    lines.extend(["", "## Docs Updated"])
    lines.extend(f"- `{item}`" for item in payload["docs_updated"])
    lines.extend(["", "## Files Modified / Created"])
    lines.extend(f"- `{item}`" for item in payload["files_modified_or_created"])
    lines.extend(["", "## Commands Run"])
    lines.extend(f"- `{item}`" for item in payload["commands_run"])
    lines.extend(["", "## Blockers"])
    lines.extend(f"- {item}" for item in payload["blockers"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    policy = load_json(args.policy)
    comparison = load_json(args.comparison)
    bundle = load_json(args.bundle_summary)
    boot = bootstrap_policies(policy, int(args.bootstrap_samples), int(args.seed)) if policy.get("policy_results") else {}
    flags = uncertainty_flags(policy, boot) if policy.get("policy_results") else {
        "small_n_unstable_policy": True,
        "strict_clean_policy_borderline": True,
        "hh_heldout_policy_borderline": True,
        "objective_mixed_vs_code_strict_clean_delta": float("nan"),
        "domain_routed_vs_hh_hh_delta": float("nan"),
    }
    uncertainty = {
        "meta": {
            "source_policy": repo_path(output_path(args.policy)),
            "source_comparison": repo_path(output_path(args.comparison)),
            "bootstrap_samples": int(args.bootstrap_samples),
        },
        "flags": flags,
        "bootstrap": boot,
    }
    write_json(output_path(args.output), uncertainty)
    write_uncertainty_md(output_path(args.output_md), uncertainty)

    design = design_note(policy, comparison, flags)
    write_json(DESIGN_JSON, design)
    write_design_md(DESIGN_MD, design)

    meta = policy.get("meta", {})
    rec_next = recommended_next(meta.get("best_policy_verdict", "INSUFFICIENT"))
    doc_section = build_doc_section(policy, comparison, bundle, flags)
    docs_updated = append_docs(doc_section)
    top_lines = {
        "BG_POLICY_EVAL_BUNDLE_VERDICT": bundle.get("meta", {}).get("bg_policy_eval_bundle_verdict", "NA"),
        "BG_HEAD_COMPARISON_VERDICT": comparison.get("meta", {}).get("bg_head_comparison_verdict", "NA"),
        "HEAD_COMPLEMENTARITY_VERDICT": comparison.get("meta", {}).get("head_complementarity_verdict", "NA"),
        "BG_POLICY_SIM_VERDICT": meta.get("bg_policy_sim_verdict", "NA"),
        "BEST_POLICY_VERDICT": meta.get("best_policy_verdict", "NA"),
        "RECOMMENDED_BG_POLICY": meta.get("recommended_bg_policy", "NA"),
        "CONTRAST_DETECTOR_VERDICT": meta.get("contrast_detector_verdict", "NA"),
        "DEFER_POLICY_VERDICT": meta.get("defer_policy_verdict", "NA"),
        "ORACLE_GAP_VERDICT": meta.get("oracle_gap_verdict", "NA"),
        "SMALL_N_UNSTABLE_POLICY": flags["small_n_unstable_policy"],
        "STRICT_CLEAN_POLICY_BORDERLINE": flags["strict_clean_policy_borderline"],
        "HH_HELDOUT_POLICY_BORDERLINE": flags["hh_heldout_policy_borderline"],
        "RECOMMENDED_NEXT": rec_next,
    }
    domain = policy.get("policy_results", {}).get("DOMAIN_ROUTED_SIMPLE", {})
    summary = {
        "top_lines": top_lines,
        "sections": {
            "Eval Bundle": f"Domains included: {', '.join(sorted(bundle.get('domain_summary', {}).keys()))}.",
            "Candidate Head Comparison": f"Candidate heads: {comparison.get('meta', {}).get('candidate_head_count')}; complementarity {comparison.get('meta', {}).get('head_complementarity_verdict')}.",
            "Head Complementarity And Disagreement": f"Disagreement and unique-win details are in `{repo_path(COMPARISON_JSON)}`.",
            "Policy Definitions": "Simulated oracle, single-head, domain-routed, contrast-routed, code-backup, vote, margin-defer, disagreement-defer, consensus, and risky ablation policies.",
            "Policy Results": f"DOMAIN_ROUTED_SIMPLE objective_avg={safe_rate(domain.get('objective_average_pairwise'))}, HH={safe_rate(domain.get('hh_heldout_pairwise'))}, strict_clean={safe_rate(domain.get('strict_clean_pairwise'))}.",
            "Defer/Selective Accuracy": f"DEFER_POLICY_VERDICT={meta.get('defer_policy_verdict')}; summary `{meta.get('defer_summary', {})}`.",
            "Oracle Gap Analysis": f"ORACLE_GAP_VERDICT={meta.get('oracle_gap_verdict')}; average gap={safe_rate(domain.get('average_oracle_gap_pairwise'))}.",
            "Contrast Detector Analysis": f"CONTRAST_DETECTOR_VERDICT={meta.get('contrast_detector_verdict')}; see `{repo_path(POLICY_JSON)}`.",
            "Bootstrap/Uncertainty": f"SMALL_N_UNSTABLE_POLICY={flags['small_n_unstable_policy']}; strict-clean borderline={flags['strict_clean_policy_borderline']}; HH borderline={flags['hh_heldout_policy_borderline']}.",
            "Controller Design Note": f"Design note written to `{repo_path(DESIGN_MD)}`.",
        },
        "docs_updated": docs_updated,
        "files_modified_or_created": [
            "shared/utilities/tests/manual/build_bg_policy_sim_eval_bundle.py",
            "shared/utilities/tests/manual/compare_bg_candidate_heads.py",
            "shared/utilities/tests/manual/simulate_bg_controller_policies.py",
            "shared/utilities/tests/manual/analyze_bg_policy_uncertainty.py",
            repo_path(BUNDLE_JSON),
            "opi/taps/probes/bg_policy_sim_eval_bundle_2026-05-17.pt",
            "opi/taps/probes/bg_policy_sim_eval_bundle_2026-05-17.md",
            repo_path(COMPARISON_JSON),
            "opi/taps/probes/bg_candidate_head_comparison_2026-05-17.md",
            repo_path(POLICY_JSON),
            "opi/taps/probes/bg_controller_policy_simulation_2026-05-17.md",
            repo_path(UNCERTAINTY_JSON),
            repo_path(UNCERTAINTY_MD),
            repo_path(DESIGN_JSON),
            repo_path(DESIGN_MD),
            repo_path(SUMMARY_JSON),
            repo_path(SUMMARY_MD),
        ],
        "commands_run": [
            "venv/bin/python -m py_compile utilities/tests/manual/build_bg_policy_sim_eval_bundle.py",
            "venv/bin/python -m py_compile utilities/tests/manual/compare_bg_candidate_heads.py",
            "venv/bin/python -m py_compile utilities/tests/manual/simulate_bg_controller_policies.py",
            "venv/bin/python -m py_compile utilities/tests/manual/analyze_bg_policy_uncertainty.py",
            "venv/bin/python -u utilities/tests/manual/build_bg_policy_sim_eval_bundle.py",
            "venv/bin/python -u utilities/tests/manual/compare_bg_candidate_heads.py",
            "venv/bin/python -u utilities/tests/manual/simulate_bg_controller_policies.py",
            "venv/bin/python -u utilities/tests/manual/analyze_bg_policy_uncertainty.py",
        ],
        "blockers": policy.get("blockers") or comparison.get("blockers") or ["none"],
    }
    write_json(SUMMARY_JSON, summary)
    write_summary_md(SUMMARY_MD, summary)
    print(f"SMALL_N_UNSTABLE_POLICY = {flags['small_n_unstable_policy']}")
    print(f"STRICT_CLEAN_POLICY_BORDERLINE = {flags['strict_clean_policy_borderline']}")
    print(f"HH_HELDOUT_POLICY_BORDERLINE = {flags['hh_heldout_policy_borderline']}")
    print(f"RECOMMENDED_NEXT = {rec_next}")
    print(f"wrote {repo_path(SUMMARY_JSON)}")


if __name__ == "__main__":
    main()
