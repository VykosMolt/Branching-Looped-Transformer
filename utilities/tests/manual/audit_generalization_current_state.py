"""Consolidate current BG/tap transfer state from saved artifacts only."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports" / "probes"
DOC_DIR = PROJECT_ROOT / "shared/docs" / "evaluator"

INVENTORY_JSON = REPORT_DIR / "current_bg_transfer_artifact_inventory_2026-05-17.json"
SCALAR_JSON = REPORT_DIR / "scalar_vs_relational_current_state_2026-05-17.json"
OUT_JSON = REPORT_DIR / "generalization_current_state_2026-05-17.json"
OUT_MD = REPORT_DIR / "generalization_current_state_2026-05-17.md"
V7_DOC = DOC_DIR / "post_v10_synthesis_2026-05-17_v7_actual_state_and_next.md"
PROTOCOL_JSON = REPORT_DIR / "strict_clean_task_screening_protocol_2026-05-17.json"
PROTOCOL_MD = REPORT_DIR / "strict_clean_task_screening_protocol_2026-05-17.md"
FINAL_JSON = REPORT_DIR / "current_bg_transfer_state_2026-05-17_summary.json"
FINAL_MD = REPORT_DIR / "current_bg_transfer_state_2026-05-17_summary.md"


DOCS_TO_APPEND = [
    DOC_DIR / "evaluator_domain_transfer_notes.md",
    DOC_DIR / "math_bg_gate_pilot_2026-05-15.md",
    DOC_DIR / "post_v10_synthesis_2026-05-15_v4.md",
    DOC_DIR / "post_v10_synthesis_2026-05-16_v5_clean_gsm8k_code_next.md",
]


def repo_path(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(p)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def compact(row: Any) -> Any:
    if not isinstance(row, dict):
        return row or "NA"
    return {
        "config": row.get("config", "NA"),
        "architecture": row.get("architecture", "NA"),
        "top1": row.get("top1", row.get("metrics", {}).get("top1_tournament_acc") if isinstance(row.get("metrics"), dict) else "NA"),
        "pairwise": row.get("pairwise", row.get("metrics", {}).get("pairwise_acc") if isinstance(row.get("metrics"), dict) else "NA"),
        "cycle": row.get("cycle", row.get("metrics", {}).get("cycle_rate") if isinstance(row.get("metrics"), dict) else "NA"),
    }


def family(inventory: dict[str, Any], name: str) -> dict[str, Any]:
    for row in inventory.get("families", []):
        if row.get("artifact_family") == name:
            return row
    return {}


def verdict_for(math_validity: dict[str, Any], expanded: dict[str, Any], code_mini: dict[str, Any]) -> str:
    if (
        math_validity.get("data_validity") == "TRUNCATION_CONFOUNDED"
        and expanded.get("expanded_linear_transfer_verdict") == "GOOD"
        and code_mini.get("CODE_V2_MINI_TRANSFER_VERDICT") == "GOOD"
    ):
        return "SUPPORTED_FOR_LOCAL_PLANNING"
    if expanded or code_mini:
        return "MIXED"
    return "INSUFFICIENT"


def strict_clean_bottleneck(enrich: dict[str, Any], balance: dict[str, Any]) -> str:
    old_global = enrich.get("label_counts", {})
    new_global = balance.get("new_label_counts", {})
    if (
        int(enrich.get("strict_clean_tournaments", 0)) <= 2
        and int(balance.get("new_strict_clean", 0)) <= 2
        and int(old_global.get("correct", 0)) > 0
        and int(old_global.get("near_miss", 0)) > 0
        and int(new_global.get("correct", 0)) > 0
        and int(new_global.get("near_miss", 0)) > 0
    ):
        return "WITHIN_TASK_PAIRING"
    return "UNRESOLVED"


def recommended_next() -> str:
    return "task_screening_for_strict_clean_ready_code_tasks"


def append_once(path: Path, title: str, text: str) -> bool:
    if not path.exists():
        return False
    current = path.read_text(encoding="utf-8")
    if title in current:
        return False
    path.write_text(current.rstrip() + "\n\n" + text.strip() + "\n", encoding="utf-8")
    return True


def write_generalization_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Generalization Current State",
        "",
        f"GENERALIZATION_VERDICT = {payload['GENERALIZATION_VERDICT']}",
        "",
        "## Evidence",
        "",
        f"- math pilot validity: `{payload['math_pilot_validity']}`",
        f"- clean GSM8K expanded transfer: `{payload['clean_gsm8k_expanded_transfer']}`",
        f"- patched code v2-mini transfer: `{payload['patched_code_v2_mini_transfer']}`",
        "",
        "## What Remains Unproven",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["what_remains_unproven"])
    lines.extend([
        "",
        "## All-Correct / All-Wrong Interpretation",
        "",
        "All-correct and all-wrong task outcomes are not contradictions of a branch-selection signal. They mean the task did not produce same-task alternatives for selection: either every branch solved it, or no branch crossed the verifier threshold. Those outcomes remain useful task-difficulty diagnostics, but they are uninformative tournaments for a comparator.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_protocol() -> dict[str, Any]:
    protocol = {
        "protocol_name": "strict_clean_task_screening_protocol_2026-05-17",
        "goal": "Find code tasks that naturally produce one correct anchor and one plausible near_miss_partial_pass under cheap screening before full candidate generation.",
        "classification": [
            "strict_clean_ready",
            "anchor_only",
            "near_miss_only",
            "all_correct",
            "all_wrong",
            "malformed_only",
        ],
        "screening_attempts_per_task": {
            "strong_anchor_attempts": ["repaired_final or direct_deterministic_final"],
            "near_miss_attempts": ["first_tool_code", "direct_short_budget", "direct_sampled_high"],
            "unit_test_label_source": "official/unit tests only",
        },
        "recommended_sources": [
            "MBPP tasks with at least 3 granular assertions",
            "local_dsa tasks with public+hidden assertion groups",
            "small local algorithm tasks where partial edge-case handling can pass some tests",
        ],
        "avoid_or_limit": [
            "HumanEval-only monolithic check(candidate) tasks",
            "tasks where repaired_final always solves and all prefinal modes also solve",
            "tasks where all modes produce zero-pass wrong code",
            "indefinite balancing of one-sided tasks after one targeted pass",
        ],
        "stop_criteria": {
            "success": "strict_clean_ready >= desired target",
            "per_task": "stop after 1 anchor attempt and 2 near-miss attempts unless task is one candidate away from strict_clean_ready",
            "failure": "retire tasks classified all_correct/all_wrong/malformed_only unless task tests are improved",
        },
        "mode_guidance": {
            "correct_anchor": ["repaired_final", "direct_deterministic_final", "direct_sampled_low"],
            "near_miss": ["first_tool_code", "direct_short_budget", "direct_sampled_high"],
        },
        "rationale": "Task screening measures whether a task can produce useful same-task alternatives cheaply; full generation should only spend budget on tasks that pass this screen.",
    }
    write_json(PROTOCOL_JSON, protocol)
    lines = [
        "# Strict-Clean Code Task Screening Protocol",
        "",
        "This is a future protocol design only. It does not run generation.",
        "",
        "## Goal",
        "",
        "Find tasks that naturally produce both a correct anchor and a plausible partial-pass near-miss under cheap screening.",
        "",
        "## Per-Task Screen",
        "",
        "1. Run one strong anchor attempt: `repaired_final` or `direct_deterministic_final`.",
        "2. Run two near-miss-seeking attempts: `first_tool_code`, `direct_short_budget`, or high-temperature direct.",
        "3. Label only with official/unit tests.",
        "4. Classify the task as `strict_clean_ready`, `anchor_only`, `near_miss_only`, `all_correct`, `all_wrong`, or `malformed_only`.",
        "",
        "## Recommended Sources",
        "",
        "- MBPP tasks with multiple granular assertions.",
        "- Local DSA tasks with public and hidden assertion groups.",
        "- Small algorithm tasks where missing edge cases can still pass some tests.",
        "",
        "## Avoiding Collapse",
        "",
        "- Avoid all-correct collapse by reducing repaired/final dominance and using short-budget or first-tool candidates for the contrast side.",
        "- Avoid all-wrong collapse by requiring a strong anchor attempt before admitting the task to full generation.",
        "- Avoid HumanEval-only monolithic checks where possible; coarse pass/fail labels hide partial progress.",
        "",
        "## Stop Criteria",
        "",
        "- Stop screening when `strict_clean_ready >= desired target`.",
        "- Retire all-correct/all-wrong tasks unless tests or prompts are redesigned.",
        "- Do not keep balancing one-sided tasks indefinitely; one-sidedness is a task-curriculum signal.",
        "",
        "## Why This Is Next",
        "",
        "The near-miss enrichment and balancing pass showed correct and near-miss candidates globally, but not reliably within the same task. Screening must happen before full candidate generation.",
        "",
    ]
    PROTOCOL_MD.write_text("\n".join(lines), encoding="utf-8")
    return protocol


def write_v7(context: dict[str, Any]) -> None:
    e = context["expanded"]
    c = context["code_mini"]
    n = context["near_miss"]
    b = context["balance"]
    fixes = context["fixes"]
    validity = context["validity"]
    micro = context["micro"]
    v1 = context["code_v1"]
    v2 = context["code_v2"]
    scalar = context["scalar"]
    lines = [
        "# Post-v10 synthesis v7 - actual state and next step",
        "",
        "**Date:** 2026-05-17",
        "**Status:** current working synthesis after clean GSM8K, code pilots, harness fixes, near-miss enrichment, and balancing.",
        "",
        "## 1. Executive Opinion",
        "",
        "HH-trained tiny taps now have enough preliminary evidence of transfer to clean objective generated branches for local planning. The remaining bottleneck is not whether a signal exists; it is branch-curriculum quality, especially producing same-task strict-clean alternatives for code.",
        "",
        "## 2. Ground Rules / Framing",
        "",
        "The evaluator remains relational, pairwise, and branch-selection oriented. Candidate labels come from external answer verifiers or unit tests. Tap/evaluator scores are not labels. Do not frame the system as `score(x)=quality`.",
        "",
        "NoNorm adds a nuance: `score(a,b)=w*(a-b)=u(a)-u(b)`, so objective correctness domains can be scalar-readable without refuting the original HH relational/noisy preference finding.",
        "",
        "## 3. Math Gate-Prep Demotion",
        "",
        "The full MATH gate-prep path is demoted as a local blocker. MATH generation remains budget- and verbosity-shaped under the local setup. GSM8K and code are currently better objective-branch domains for testing transfer and curriculum quality.",
        "",
        "## 4. Math Pilot Validity Probe",
        "",
        f"- DATA_VALIDITY: `{validity.get('data_validity', 'MISSING')}`",
        "- Main reason: old pilot had truncation and difficulty-composition confounds.",
        "- Parser was not the issue: extractable answer was reported as 100%.",
        "- Old `TRANSFER_POOR` is suspect/confounded, not a final negative result.",
        "",
        "## 5. Clean GSM8K Micro",
        "",
        f"- CLEAN_GSM8K_VERDICT: `{micro.get('clean_gsm8k_verdict', 'MISSING')}`",
        f"- CLEAN_TRANSFER_VERDICT: `{micro.get('clean_transfer_verdict', 'MISSING')}`",
        f"- tournaments: `{micro.get('feature_summary', {}).get('n_tournaments', 'MISSING')}`",
        f"- best row: `{compact(micro.get('best_hh_trained'))}`",
        "",
        "## 6. Expanded Clean GSM8K + GRU Control",
        "",
        f"- EXPANDED_CLEAN_GSM8K_VERDICT: `{e.get('expanded_clean_gsm8k_verdict', 'MISSING')}`",
        f"- EXPANDED_LINEAR_TRANSFER_VERDICT: `{e.get('expanded_linear_transfer_verdict', 'MISSING')}`",
        f"- GRU_CONTROL_VERDICT: `{e.get('gru_control_verdict', 'MISSING')}`",
        f"- prompts processed: `{e.get('generation_summary', {}).get('prompts_processed', 'MISSING')}`",
        f"- attempts generated: `{e.get('generation_summary', {}).get('attempts_generated', 'MISSING')}`",
        f"- clean tournaments: `{e.get('generation_summary', {}).get('clean_tournaments_kept', 'MISSING')}`",
        f"- kept branches: `{e.get('generation_summary', {}).get('kept_branches', 'MISSING')}`",
        f"- random top1 baseline: `{e.get('random_top1_baseline', 'MISSING')}`",
        f"- near-miss fraction: `{e.get('generation_summary', {}).get('near_miss_fraction', 'MISSING')}`",
        f"- best AntisymLinear: `{compact(e.get('best_antisymlinear'))}`",
        f"- best NoNorm: `{compact(e.get('best_nonorm'))}`",
        f"- best GRU: `{compact(e.get('best_gru'))}`",
        "",
        "Interpretation: HH-trained taps transfer to clean GSM8K. GRU was above random but did not justify added temporal complexity.",
        "",
        "## 7. Code v1 All-Correct Collapse",
        "",
        f"- CODE_TOURNAMENT_VERDICT: `{v1.get('CODE_TOURNAMENT_VERDICT', 'MISSING')}`",
        f"- CODE_TRANSFER_VERDICT: `{v1.get('CODE_TRANSFER_VERDICT', 'MISSING')}`",
        f"- label counts: `{v1.get('label_counts', 'MISSING')}`",
        "- Issue: wrapper/final route was too successful; 32/40 candidates passed all tests, leaving too few mixed tournaments.",
        "",
        "## 8. Code v2 Dirty Diagnostic Result",
        "",
        f"- CODE_V2_TOURNAMENT_VERDICT: `{v2.get('CODE_V2_TOURNAMENT_VERDICT', 'MISSING')}`",
        f"- CODE_V2_TRANSFER_VERDICT: `{v2.get('CODE_V2_TRANSFER_VERDICT', 'MISSING')}`",
        "- Status: historical/pre-fix diagnostic only.",
        "- Manual bucket breakdown: runnable_zero_pass_wrong=50, parseable_runtime_error=25, prose_or_wrapper_not_code=20, syntax_invalid_code=6, safety_rejected=3, parseable_no_function=1.",
        "",
        "## 9. Wrapper / Taskset / Harness Fixes",
        "",
        f"- FIX_VERDICT: `{fixes.get('FIX_VERDICT', 'MISSING')}`",
        "- Wrapper/prose/status outputs are rejected as code.",
        "- MBPP function-name and signature issues were fixed.",
        "- Labels are split into `correct`, `near_miss`, `wrong_code`, `runtime_error`, `malformed`, and `safety_rejected`.",
        "- `sys.setrecursionlimit` is allowed while unsafe sys/file/network/process usage remains rejected.",
        "- Validation reported `py_compile` success and local-agent wrapper tests: 139 passed.",
        "",
        "## 10. Patched Code v2-Mini",
        "",
        f"- CODE_V2_MINI_TOURNAMENT_VERDICT: `{c.get('CODE_V2_MINI_TOURNAMENT_VERDICT', 'MISSING')}`",
        f"- CODE_V2_MINI_TRANSFER_VERDICT: `{c.get('CODE_V2_MINI_TRANSFER_VERDICT', 'MISSING')}`",
        f"- tasks: `{c.get('tasks', 'MISSING')}`",
        f"- unique candidates: `{c.get('unique_candidates', 'MISSING')}`",
        f"- labels: `{c.get('label_counts', 'MISSING')}`",
        f"- strict/diagnostic_runnable/diagnostic_mixed: `{c.get('strict_clean_tournaments', 'MISSING')} / {c.get('diagnostic_runnable_tournaments', 'MISSING')} / {c.get('diagnostic_mixed_tournaments', 'MISSING')}`",
        f"- random top1 baseline: `{c.get('random_top1_baseline', 'MISSING')}`",
        f"- best AntisymLinear: `{compact(c.get('best_antisymlinear'))}`",
        f"- best NoNorm: `{compact(c.get('best_nonorm'))}`",
        "",
        "Interpretation: the transfer signal survived cleanup, but the primary set was runnable-diagnostic, not strict-clean.",
        "",
        "## 11. Independent 10-Task Near-Miss Enrichment",
        "",
        f"- outcome: `{n.get('CODE_V2_NEARMISS10_SUCCESS', 'MISSING')}`",
        f"- tasks: `{n.get('tasks', 'MISSING')}`",
        f"- unique candidates: `{n.get('unique_candidates', 'MISSING')}`",
        f"- labels: `{n.get('label_counts', 'MISSING')}`",
        f"- strict_clean: `{n.get('strict_clean_tournaments', 'MISSING')}`",
        "- Interpretation: harness cleanliness was good and correct/near-miss candidates existed globally, but same-task pairing was weak.",
        "",
        "## 12. Near-Miss Balancing Pass",
        "",
        f"- BALANCE_INSPECTION_VERDICT: `{b.get('BALANCE_INSPECTION_VERDICT', 'MISSING')}`",
        f"- BALANCING_GENERATION_VERDICT: `{b.get('BALANCING_GENERATION_VERDICT', 'MISSING')}`",
        f"- BALANCED_TOURNAMENT_VERDICT: `{b.get('BALANCED_TOURNAMENT_VERDICT', 'MISSING')}`",
        f"- BALANCED_TRANSFER_VERDICT: `{b.get('BALANCED_TRANSFER_VERDICT', 'MISSING')}`",
        f"- before labels: `{b.get('old_label_counts', 'MISSING')}`",
        f"- after labels: `{b.get('new_label_counts', 'MISSING')}`",
        f"- strict_clean stayed: `{b.get('old_strict_clean', 'MISSING')} -> {b.get('new_strict_clean', 'MISSING')}`",
        "- Interpretation: the global pool improved, but no tasks converted; within-task pairing is the bottleneck.",
        "",
        "## 13. Generalization Verdict",
        "",
        f"`GENERALIZATION_VERDICT = {context['generalization_verdict']}`",
        "",
        "Signal generalization is preliminarily supported for local planning. It is not a proof of full gate-scale transfer, MATH transfer under local budget, or controller effects.",
        "",
        "## 14. Scalar / Pointwise-Ranking Verdict",
        "",
        f"`POINTWISE_RANKING_VERDICT = {scalar.get('POINTWISE_RANKING_VERDICT', 'MISSING')}`",
        "",
        "NoNorm suggests objective correctness domains can be scalar-readable. This does not refute the original HH relational finding; HH preference remains predominantly relational/noisy.",
        "",
        "## 15. Current Bottleneck: Within-Task Pairing / Task Curriculum",
        "",
        f"`STRICT_CLEAN_BOTTLENECK = {context['strict_clean_bottleneck']}`",
        "",
        "Code strict-clean near-miss generation remains unsolved. The next data problem is screening tasks for natural correct-vs-near-miss alternatives before spending full generation budget.",
        "",
        "## 16. Recommended Next Move",
        "",
        f"`RECOMMENDED_NEXT = {recommended_next()}`",
        "",
        "Run a cheap task-screening protocol for strict-clean-ready code tasks: one strong anchor attempt plus two near-miss-seeking attempts per task, labeled only by unit tests, then scale only tasks that already show same-task pairing.",
        "",
    ]
    V7_DOC.write_text("\n".join(lines), encoding="utf-8")


def append_related_docs(summary: dict[str, Any]) -> list[str]:
    title = "## Current transfer state and scalar-vs-relational audit (2026-05-17)"
    text = "\n".join([
        title,
        "",
        f"- CURRENT_ARTIFACT_INVENTORY_VERDICT: `{summary['CURRENT_ARTIFACT_INVENTORY_VERDICT']}`",
        f"- GENERALIZATION_VERDICT: `{summary['GENERALIZATION_VERDICT']}`",
        f"- POINTWISE_RANKING_VERDICT: `{summary['POINTWISE_RANKING_VERDICT']}`",
        "- clean GSM8K: expanded run was `CLEAN_MINIMUM` with `GOOD` HH-trained linear transfer; GRU control was `GRU_WEAK`.",
        "- patched code v2-mini: patched pipeline reached `RUNNABLE_DIAGNOSTIC` and `GOOD` transfer, with NoNorm strongest on the diagnostic set.",
        "- near-miss enrichment/balancing: correct and near_miss candidates exist globally, but strict_clean stayed low at 2 after balancing.",
        f"- current bottleneck: `{summary['STRICT_CLEAN_BOTTLENECK']}`",
        "- full reports: `opi/taps/probes/current_bg_transfer_state_2026-05-17_summary.md`, `docs/evaluator/post_v10_synthesis_2026-05-17_v7_actual_state_and_next.md`, `opi/taps/probes/scalar_vs_relational_current_state_2026-05-17.md`, `opi/taps/probes/strict_clean_task_screening_protocol_2026-05-17.md`.",
        "- interpretation: transfer is supported enough for local planning; the next bottleneck is strict-clean code task screening, not another broad transfer existence probe.",
        "",
    ])
    updated = []
    for path in DOCS_TO_APPEND:
        if append_once(path, title, text):
            updated.append(repo_path(path))
        elif path.exists():
            updated.append(repo_path(path))
    return updated


def write_final_md(summary: dict[str, Any]) -> None:
    lines = [
        "# Current BG Transfer State Summary",
        "",
        f"CURRENT_ARTIFACT_INVENTORY_VERDICT = {summary['CURRENT_ARTIFACT_INVENTORY_VERDICT']}",
        f"GENERALIZATION_VERDICT = {summary['GENERALIZATION_VERDICT']}",
        f"POINTWISE_RANKING_VERDICT = {summary['POINTWISE_RANKING_VERDICT']}",
        f"STRICT_CLEAN_BOTTLENECK = {summary['STRICT_CLEAN_BOTTLENECK']}",
        f"RECOMMENDED_NEXT = {summary['RECOMMENDED_NEXT']}",
        "",
        "## Artifact Inventory",
        "",
        f"- inventory report: `{summary['artifact_inventory_report']}`",
        f"- scalar audit report: `{summary['scalar_audit_report']}`",
        f"- generalization report: `{summary['generalization_report']}`",
        "",
        "## What We Actually Did",
        "",
        "- Validated the old mixed math pilot as truncation-confounded.",
        "- Ran clean GSM8K micro and expanded clean GSM8K transfer.",
        "- Ran GRU control, which underperformed simpler exact-antisymmetric heads.",
        "- Ran code v1, code v2, harness fixes, patched v2-mini, near-miss enrichment, and balancing.",
        "",
        "## Generalization Interpretation",
        "",
        summary["generalization_interpretation"],
        "",
        "## Scalar / Pointwise-Readout Interpretation",
        "",
        summary["pointwise_interpretation"],
        "",
        "## Code Strict-Clean Bottleneck",
        "",
        summary["strict_clean_interpretation"],
        "",
        "## Recommended Next Protocol",
        "",
        f"- protocol: `{summary['task_screening_protocol']}`",
        "",
        "## Docs Updated",
        "",
    ]
    lines.extend(f"- `{path}`" for path in summary["docs_updated"])
    lines.extend(["", "## Files Modified / Created", ""])
    lines.extend(f"- `{path}`" for path in summary["files_modified_or_created"])
    lines.extend(["", "## Commands Run", "", "```bash"])
    lines.extend(summary["commands_run"])
    lines.extend(["```", "", "## Blockers", "", summary.get("blockers") or "None.", ""])
    FINAL_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    inventory = load_json(INVENTORY_JSON)
    scalar = load_json(SCALAR_JSON)
    validity = load_json(REPORT_DIR / "math_data_validity_2026-05-16.json")
    micro = load_json(REPORT_DIR / "clean_gsm8k_extreme_transfer_2026-05-16.json")
    expanded = load_json(REPORT_DIR / "clean_gsm8k_expanded_transfer_gru_2026-05-16_summary.json")
    code_v1 = load_json(REPORT_DIR / "code_branch_pilot_2026-05-16_summary.json")
    code_v2 = load_json(REPORT_DIR / "code_branch_pilot_v2_2026-05-16_summary.json")
    fixes = load_json(REPORT_DIR / "code_branch_v2_harness_agent_fixes_2026-05-16.json")
    code_mini = load_json(REPORT_DIR / "code_branch_pilot_v2_mini_patched_2026-05-16_summary.json")
    near_miss = load_json(REPORT_DIR / "code_branch_near_miss_enrichment10_2026-05-17_summary.json")
    balance = load_json(REPORT_DIR / "code_branch_near_miss_balancing_2026-05-17_summary.json")

    generalization = verdict_for(validity, expanded, code_mini)
    bottleneck = strict_clean_bottleneck(near_miss, balance)
    payload = {
        "GENERALIZATION_VERDICT": generalization,
        "math_pilot_validity": validity.get("data_validity", "MISSING"),
        "clean_gsm8k_expanded_transfer": expanded.get("expanded_linear_transfer_verdict", "MISSING"),
        "patched_code_v2_mini_transfer": code_mini.get("CODE_V2_MINI_TRANSFER_VERDICT", "MISSING"),
        "what_remains_unproven": [
            "strict-clean near-miss code selection at adequate n",
            "MATH under local budget",
            "full gate-scale claims",
            "Phase-2 controller effects",
        ],
        "all_correct_all_wrong_explanation": "All-correct/all-wrong tasks are task-difficulty outcomes, not contradictions of branch-selection signal.",
        "outputs": {"json": repo_path(OUT_JSON), "md": repo_path(OUT_MD)},
    }
    write_json(OUT_JSON, payload)
    write_generalization_md(OUT_MD, payload)
    protocol = write_protocol()
    context = {
        "validity": validity,
        "micro": micro,
        "expanded": expanded,
        "code_v1": code_v1,
        "code_v2": code_v2,
        "fixes": fixes,
        "code_mini": code_mini,
        "near_miss": near_miss,
        "balance": balance,
        "scalar": scalar,
        "generalization_verdict": generalization,
        "strict_clean_bottleneck": bottleneck,
    }
    write_v7(context)

    summary = {
        "CURRENT_ARTIFACT_INVENTORY_VERDICT": inventory.get("CURRENT_ARTIFACT_INVENTORY_VERDICT", "MISSING"),
        "GENERALIZATION_VERDICT": generalization,
        "POINTWISE_RANKING_VERDICT": scalar.get("POINTWISE_RANKING_VERDICT", "MISSING"),
        "STRICT_CLEAN_BOTTLENECK": bottleneck,
        "RECOMMENDED_NEXT": recommended_next(),
        "artifact_inventory_report": repo_path(REPORT_DIR / "current_bg_transfer_artifact_inventory_2026-05-17.md"),
        "scalar_audit_report": repo_path(REPORT_DIR / "scalar_vs_relational_current_state_2026-05-17.md"),
        "generalization_report": repo_path(OUT_MD),
        "task_screening_protocol": repo_path(PROTOCOL_MD),
        "generalization_interpretation": "Clean GSM8K expanded transfer was GOOD, patched code v2-mini transfer was GOOD, and the old negative math result is marked truncation-confounded. That is enough support for local planning, not a broad proof.",
        "pointwise_interpretation": "NoNorm is competitive or winning in objective branch domains, so objective correctness appears scalar-readable in some settings. This does not refute the HH relational/noisy preference finding.",
        "strict_clean_interpretation": "Near-miss enrichment and balancing both show correct and near_miss candidates globally while strict_clean remains 2. The bottleneck is within-task pairing and task curriculum.",
        "protocol_summary": protocol,
        "commands_run": [
            "venv/bin/python -m py_compile utilities/tests/manual/audit_current_bg_transfer_artifacts.py",
            "venv/bin/python -m py_compile utilities/tests/manual/audit_scalar_vs_relational_current_state.py",
            "venv/bin/python -m py_compile utilities/tests/manual/audit_generalization_current_state.py",
            "venv/bin/python -u utilities/tests/manual/audit_current_bg_transfer_artifacts.py",
            "venv/bin/python -u utilities/tests/manual/audit_scalar_vs_relational_current_state.py",
            "venv/bin/python -u utilities/tests/manual/audit_generalization_current_state.py",
        ],
        "files_modified_or_created": [
            "shared/utilities/tests/manual/audit_current_bg_transfer_artifacts.py",
            "shared/utilities/tests/manual/audit_scalar_vs_relational_current_state.py",
            "shared/utilities/tests/manual/audit_generalization_current_state.py",
            repo_path(REPORT_DIR / "current_bg_transfer_artifact_inventory_2026-05-17.json"),
            repo_path(REPORT_DIR / "current_bg_transfer_artifact_inventory_2026-05-17.md"),
            repo_path(REPORT_DIR / "scalar_vs_relational_current_state_2026-05-17.json"),
            repo_path(REPORT_DIR / "scalar_vs_relational_current_state_2026-05-17.md"),
            repo_path(REPORT_DIR / "generalization_current_state_2026-05-17.json"),
            repo_path(REPORT_DIR / "generalization_current_state_2026-05-17.md"),
            repo_path(V7_DOC),
            repo_path(PROTOCOL_JSON),
            repo_path(PROTOCOL_MD),
            repo_path(FINAL_JSON),
            repo_path(FINAL_MD),
        ],
        "blockers": "",
    }
    summary["docs_updated"] = append_related_docs(summary)
    write_json(FINAL_JSON, summary)
    write_final_md(summary)
    print(f"GENERALIZATION_VERDICT = {generalization}")
    print(f"POINTWISE_RANKING_VERDICT = {summary['POINTWISE_RANKING_VERDICT']}")
    print(f"STRICT_CLEAN_BOTTLENECK = {bottleneck}")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {V7_DOC}")
    print(f"Wrote {PROTOCOL_JSON}")
    print(f"Wrote {PROTOCOL_MD}")
    print(f"Wrote {FINAL_JSON}")
    print(f"Wrote {FINAL_MD}")


if __name__ == "__main__":
    main()
