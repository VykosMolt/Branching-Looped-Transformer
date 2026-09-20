"""Aggregate the BG same-prefix hidden-state branch generation suite."""
from __future__ import annotations

import time
from typing import Any

from bg_hidden_branch_suite_common import REPORT_ROOT, ensure_report_root, load_json, md_table, rel, write_json, write_md


SUMMARY_JSON = REPORT_ROOT / "summary.json"
SUMMARY_MD = REPORT_ROOT / "summary.md"
ANALYSIS_JSON = REPORT_ROOT / "analysis.json"
ANALYSIS_MD = REPORT_ROOT / "analysis.md"


def verdict(path: str, key: str, default: str = "MISSING") -> str:
    return str(load_json(REPORT_ROOT / path, {}).get(key, default))


def readiness(v: dict[str, str]) -> str:
    if v["feasibility"] in {"BLOCKED", "MISSING"}:
        return "NEEDS_STATE_HANDLING_WORK"
    if v["generation"] in {"BLOCKED", "MISSING"}:
        return "NEEDS_BRANCH_GENERATOR"
    if v["persistence"] in {"LATENT_BRANCHES_COLLAPSE_IMMEDIATELY", "LATENT_BRANCHES_COLLAPSE_BY_30"}:
        return "NEEDS_BRANCH_GENERATOR"
    if v["outcomes"] == "NO_BEHAVIORAL_DIVERSITY":
        return "BEHAVIORALLY_NEUTRAL_BRANCHES"
    if v["outcomes"] in {"BLOCKED", "MISSING"}:
        return "INSUFFICIENT"
    if v["selection"] == "NO_HIDDEN_BRANCH_SELECTION_SIGNAL":
        return "NEEDS_BETTER_BRANCH_EVALUATOR"
    if v["selection"] in {"FROZEN_TAPS_SELECT_GOOD_HIDDEN_BRANCHES", "WEAK_HIDDEN_BRANCH_SELECTION_SIGNAL"}:
        return "READY_FOR_MINIMAL_TRAINABLE_PROTOTYPE"
    return "INSUFFICIENT"


def main() -> int:
    ensure_report_root()
    started = time.time()
    feasibility = load_json(REPORT_ROOT / "feasibility.json", {})
    persistence = load_json(REPORT_ROOT / "hidden_branch_persistence.json", {})
    outcomes = load_json(REPORT_ROOT / "hidden_branch_outcomes.json", {})
    selection = load_json(REPORT_ROOT / "hidden_origin_branch_selection.json", {})
    gates = load_json(REPORT_ROOT / "l30_l42_gate_assessment.json", {})
    thresholds = load_json(REPORT_ROOT / "hidden_branch_adaptive_threshold_sweep.json", {})
    cached = load_json(REPORT_ROOT / "cached_branch_selection_sanity.json", {})
    task_subset = load_json(REPORT_ROOT / "task_subset.json", {})
    verdicts = {
        "feasibility": feasibility.get("BG_HIDDEN_BRANCH_FEASIBILITY_VERDICT", "MISSING"),
        "live_method": feasibility.get("LIVE_BRANCH_METHOD", "MISSING"),
        "task_subset": task_subset.get("BG_HIDDEN_BRANCH_TASK_SUBSET_VERDICT", "MISSING"),
        "utility": "READY",
        "generation": persistence.get("BG_HIDDEN_BRANCH_GENERATION_VERDICT", "MISSING"),
        "persistence": persistence.get("BG_LATENT_BRANCH_PERSISTENCE_VERDICT", "MISSING"),
        "outcomes": outcomes.get("BG_HIDDEN_BRANCH_OUTCOME_DATASET_VERDICT", "MISSING"),
        "selection": selection.get("BG_HIDDEN_ORIGIN_BRANCH_SELECTION_VERDICT", "MISSING"),
        "gates": gates.get("BG_HIDDEN_BRANCH_L30_L42_GATE_VERDICT", "MISSING"),
        "thresholds": thresholds.get("BG_HIDDEN_BRANCH_ADAPTIVE_THRESHOLD_VERDICT", "MISSING"),
        "cached": cached.get("BG_CACHED_BRANCH_SELECTION_SANITY_VERDICT", "SKIPPED"),
    }
    phase2 = readiness(verdicts)
    top = {
        "BG_HIDDEN_BRANCH_FEASIBILITY_VERDICT": verdicts["feasibility"],
        "LIVE_BRANCH_METHOD": verdicts["live_method"],
        "BG_HIDDEN_BRANCH_TASK_SUBSET_VERDICT": verdicts["task_subset"],
        "BG_HIDDEN_BRANCH_UTILITY_VERDICT": verdicts["utility"],
        "BG_HIDDEN_BRANCH_GENERATION_VERDICT": verdicts["generation"],
        "BG_LATENT_BRANCH_PERSISTENCE_VERDICT": verdicts["persistence"],
        "BG_HIDDEN_BRANCH_OUTCOME_DATASET_VERDICT": verdicts["outcomes"],
        "BG_HIDDEN_ORIGIN_BRANCH_SELECTION_VERDICT": verdicts["selection"],
        "BG_HIDDEN_BRANCH_L30_L42_GATE_VERDICT": verdicts["gates"],
        "BG_HIDDEN_BRANCH_ADAPTIVE_THRESHOLD_VERDICT": verdicts["thresholds"],
        "BG_CACHED_BRANCH_SELECTION_SANITY_VERDICT": verdicts["cached"],
        "PHASE2_HIDDEN_BRANCH_READINESS": phase2,
    }
    high_alpha = persistence.get("high_alpha_diagnostic", {})
    answers = {
        "actual_same_prefix_hidden_branches_generated": verdicts["generation"] in {"HOOK_HIDDEN_ORIGIN_BRANCHES_GENERATED", "TRUE_FORK_BRANCHES_GENERATED"},
        "true_fork_or_hook": verdicts["live_method"],
        "persistence": verdicts["persistence"],
        "behavioral_diversity": outcomes.get("stats", {}),
        "frozen_tap_selection": {
            "verdict": verdicts["selection"],
            "lift": selection.get("tap_selection_vs_random_lift"),
        },
        "l30_l42_gates": verdicts["gates"],
        "adaptive_thresholds": verdicts["thresholds"],
        "minimal_prototype": {
            "READY_FOR_MINIMAL_TRAINABLE_PROTOTYPE": "L24 hook-hidden-origin branch generator, capture at L30/L36/L42/L47, frozen tap comparator at L36, selection-only evaluation before any steering.",
            "NEEDS_BETTER_BRANCH_EVALUATOR": "Build a hidden-origin branch evaluator/calibration dataset from these same-prefix outcomes; keep frozen BG taps as diagnostics, not as the selection policy.",
            "NEEDS_BRANCH_GENERATOR": "Focus on stronger branch generation/diversity before trainable Phase 2; use high-alpha diagnostics only to separate strength from convergence pressure.",
            "NEEDS_STATE_HANDLING_WORK": "Implement branch-aware Ouro forward/cache before claiming true fork/carry.",
            "BEHAVIORALLY_NEUTRAL_BRANCHES": "Improve hidden-branch generation so same-prefix perturbations produce downstream outcome variation.",
        }.get(phase2, "Collect more live hidden-origin branch data before choosing a trainable prototype."),
    }
    payload = {
        **top,
        "answers": answers,
        "high_alpha_diagnostic": high_alpha,
        "artifact_root": rel(REPORT_ROOT),
        "files_created": [
            rel(REPORT_ROOT / name)
            for name in [
                "feasibility.md",
                "task_subset.md",
                "hidden_branch_persistence.md",
                "hidden_branch_outcomes.md",
                "hidden_origin_branch_selection.md",
                "l30_l42_gate_assessment.md",
                "hidden_branch_adaptive_threshold_sweep.md",
                "cached_branch_selection_sanity.md",
                "summary.md",
                "analysis.md",
            ]
        ],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(SUMMARY_JSON, payload)
    write_json(ANALYSIS_JSON, payload)
    rows = [{"verdict": k, "value": v} for k, v in top.items()]
    lines = [
        "# BG Hidden-State Branch Generation Suite",
        "",
        *[f"{k} = {v}" for k, v in top.items()],
        "",
        "## Interpretation",
        "",
        "Existing cached candidate branches are not latent forks. This suite tests same-prefix hidden-origin branches. The current live method is explicitly separated from true fork/carry.",
        "",
        f"- same-prefix hidden branches generated: `{answers['actual_same_prefix_hidden_branches_generated']}`",
        f"- method used: `{answers['true_fork_or_hook']}`",
        f"- high-alpha diagnostic: `{high_alpha.get('interpretation', 'not available')}`",
        f"- minimal prototype recommendation: {answers['minimal_prototype']}",
        "",
        "## Verdicts",
        "",
    ]
    lines.extend(md_table(rows, ["verdict", "value"]))
    lines.extend(
        [
            "",
            "## Blockers / Caveats",
            "",
            "- True autoregressive hidden-state fork/carry remains blocked without branch-aware cache/state handling.",
            "- Safe-alpha persistence and behavioral diversity are alpha-scope-limited; the high-alpha diagnostic is non-headline and only separates generation-strength artifacts from convergence pressure.",
            "- Tap-score spread is not used as the collapse criterion because prior steering work found signed/unsigned readout ambiguity.",
        ]
    )
    write_md(SUMMARY_MD, lines)
    analysis_lines = [
        "# BG Hidden-State Branch Generation Analysis",
        "",
        "## 1. Motivation and corrected distinction",
        "",
        "Cached text/candidate branches are useful offline selection sanity checks, but Phase 2 needs same-prefix hidden-origin branch data.",
        "",
        "## 2. Feasibility and method used",
        "",
        f"`LIVE_BRANCH_METHOD = {verdicts['live_method']}`. True generation-ready fork/carry was not claimed unless the feasibility report says so.",
        "",
        "## 3. Task subset",
        "",
        f"`{verdicts['task_subset']}` with `{task_subset.get('task_count', 0)}` reasoning/science MCQ tasks.",
        "",
        "## 4. Hidden branch generation",
        "",
        f"`{verdicts['generation']}`; branch count `{persistence.get('branch_count', 0)}`.",
        "",
        "## 5. Persistence/convergence",
        "",
        f"`{verdicts['persistence']}`. Geometry, not tap-score spread, drives this verdict.",
        "",
        "## 6. Outcome diversity",
        "",
        f"`{verdicts['outcomes']}` with stats `{outcomes.get('stats', {})}`.",
        "",
        "## 7. Hidden-origin branch selection",
        "",
        f"`{verdicts['selection']}`; lift `{selection.get('tap_selection_vs_random_lift')}`.",
        "",
        "## 8. L30/L42 gate assessment",
        "",
        f"`{verdicts['gates']}`.",
        "",
        "## 9. Adaptive thresholds",
        "",
        f"`{verdicts['thresholds']}`.",
        "",
        "## 10. Cached branch sanity comparison",
        "",
        f"`{verdicts['cached']}`; sanity only, not hidden-origin evidence.",
        "",
        "## 11. Phase 2 readiness",
        "",
        f"`PHASE2_HIDDEN_BRANCH_READINESS = {phase2}`.",
        "",
        "## 12. Recommended minimal prototype",
        "",
        answers["minimal_prototype"],
        "",
        "## 13. Files created",
        "",
        *[f"- `{path}`" for path in payload["files_created"]],
        "",
        "## 14. Commands run",
        "",
        "See terminal history for compile/run commands from this suite.",
        "",
        "## 15. Blockers",
        "",
        "- Branch-aware Ouro forward/cache is still required for true autoregressive fork/carry.",
    ]
    write_md(ANALYSIS_MD, analysis_lines)
    print(f"PHASE2_HIDDEN_BRANCH_READINESS = {phase2}")
    print(f"Wrote {rel(SUMMARY_MD)}")
    print(f"Wrote {rel(ANALYSIS_MD)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
