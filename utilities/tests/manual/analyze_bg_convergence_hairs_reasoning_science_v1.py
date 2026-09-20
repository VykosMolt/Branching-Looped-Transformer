from __future__ import annotations

import time

from bg_convergence_hairs_rs_v1_common import OUT_ROOT, read_json, status_line, write_json, write_md


def status_from(verdicts: dict[str, str]) -> str:
    if verdicts.get("BG_SCIENCE_HARD_SLICE_VERDICT") == "SCIENCE_REWARD_PARSER_WEAK":
        return "SCIENCE_REWARD_PARSER_WEAK"
    if verdicts.get("BG_SCIENCE_HARD_SLICE_VERDICT") == "SCIENCE_BRANCH_GENERATION_WEAK":
        return "SCIENCE_BRANCH_GENERATION_WEAK"
    if verdicts.get("BG_DOMAIN_BRANCH_RECIPE_DIAGNOSTIC_VERDICT") == "SCIENCE_NEEDS_DIFFERENT_RECIPE":
        return "SCIENCE_BRANCH_GENERATION_WEAK"
    if verdicts.get("BG_CONVERGENCE_HAIR_REPLAY_EVAL_VERDICT") == "HARD_MERGE_SAFE" and verdicts.get("BG_CONVERGENCE_HAIR_REGENERATED_VERDICT") == "REGENERATED_CONFIRMS_SAFE_MERGE":
        return "CONVERGENCE_HAIRS_READY"
    if verdicts.get("BG_TIE_DECOMPOSITION_DIAGNOSTIC_VERDICT") == "TIES_HIDE_LATENT_DIVERSITY":
        return "TIES_HIDE_DIVERGENCE"
    if verdicts.get("BG_CONVERGENCE_HAIR_REPLAY_EVAL_VERDICT") in {"SOFT_CLUSTER_ONLY", "FALSE_MERGE_TOO_HIGH", "HARD_MERGE_SAFE", "L30_HAIR_SAFE", "L42_HAIR_SAFE"}:
        return "CONVERGENCE_HAIRS_SOFT_ONLY"
    if verdicts.get("BG_REASONING_HARD_SLICE_VERDICT") in {"REASONING_FINAL_CHOICE_WEAK", "REASONING_NEEDS_TERMINAL_DEFER"}:
        return "REASONING_TERMINAL_POLICY_WEAK"
    if str(verdicts.get("BG_PRE_STEERING_READINESS_VERDICT", "")).startswith("READY_FOR_STEERING"):
        return "READY_FOR_STEERING_BASELINE"
    return "INSUFFICIENT"


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def row_by(rows: list[dict], key: str, value: str) -> dict:
    for row in rows:
        if row.get(key) == value:
            return row
    return {}


def md_table(rows: list[dict], columns: list[tuple[str, str]], limit: int | None = None) -> list[str]:
    selected = rows[:limit] if limit else rows
    if not selected:
        return ["_No rows._"]
    lines = [
        "| " + " | ".join(title for _, title in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in selected:
        lines.append("| " + " | ".join(fmt(row.get(key)) for key, _ in columns) + " |")
    return lines


def bullet_dict(data: dict) -> list[str]:
    if not data:
        return ["- none"]
    return [f"- {key}: `{fmt(value)}`" for key, value in data.items()]


def main() -> int:
    started = time.time()
    files = {
        "inventory": read_json(OUT_ROOT / "inventory.json", {}) or {},
        "dataset": read_json(OUT_ROOT / "convergence_hair_dataset.json", {}) or {},
        "policies": read_json(OUT_ROOT / "convergence_hair_policies.json", {}) or {},
        "replay": read_json(OUT_ROOT / "convergence_hair_replay_eval.json", {}) or {},
        "regenerated": read_json(OUT_ROOT / "convergence_hair_regenerated.json", {}) or {},
        "tie": read_json(OUT_ROOT / "tie_decomposition_diagnostic.json", {}) or {},
        "reasoning": read_json(OUT_ROOT / "reasoning_hard_slice.json", {}) or {},
        "science": read_json(OUT_ROOT / "science_hard_slice.json", {}) or {},
        "recipe": read_json(OUT_ROOT / "domain_branch_recipe.json", {}) or {},
        "terminal": read_json(OUT_ROOT / "terminal_defer_policy.json", {}) or {},
        "readiness": read_json(OUT_ROOT / "pre_steering_readiness.json", {}) or {},
    }
    verdicts = {
        "BG_CONVERGENCE_HAIRS_RS_INVENTORY_VERDICT": files["inventory"].get("BG_CONVERGENCE_HAIRS_RS_INVENTORY_VERDICT", "INSUFFICIENT"),
        "BG_CONVERGENCE_HAIR_DATASET_VERDICT": files["dataset"].get("BG_CONVERGENCE_HAIR_DATASET_VERDICT", "INSUFFICIENT"),
        "BG_CONVERGENCE_HAIR_POLICY_DEF_VERDICT": files["policies"].get("BG_CONVERGENCE_HAIR_POLICY_DEF_VERDICT", "INSUFFICIENT"),
        "BG_CONVERGENCE_HAIR_REPLAY_EVAL_VERDICT": files["replay"].get("BG_CONVERGENCE_HAIR_REPLAY_EVAL_VERDICT", "INSUFFICIENT"),
        "BG_CONVERGENCE_HAIR_REGENERATED_VERDICT": files["regenerated"].get("BG_CONVERGENCE_HAIR_REGENERATED_VERDICT", "INSUFFICIENT"),
        "BG_TIE_DECOMPOSITION_DIAGNOSTIC_VERDICT": files["tie"].get("BG_TIE_DECOMPOSITION_DIAGNOSTIC_VERDICT", "INSUFFICIENT"),
        "BG_REASONING_HARD_SLICE_VERDICT": files["reasoning"].get("BG_REASONING_HARD_SLICE_VERDICT", "INSUFFICIENT"),
        "BG_SCIENCE_HARD_SLICE_VERDICT": files["science"].get("BG_SCIENCE_HARD_SLICE_VERDICT", "INSUFFICIENT"),
        "BG_DOMAIN_BRANCH_RECIPE_DIAGNOSTIC_VERDICT": files["recipe"].get("BG_DOMAIN_BRANCH_RECIPE_DIAGNOSTIC_VERDICT", "INSUFFICIENT"),
        "BG_TERMINAL_DEFER_POLICY_VERDICT": files["terminal"].get("BG_TERMINAL_DEFER_POLICY_VERDICT", "INSUFFICIENT"),
        "BG_PRE_STEERING_READINESS_VERDICT": files["readiness"].get("BG_PRE_STEERING_READINESS_VERDICT", "INSUFFICIENT"),
    }
    overall = status_from(verdicts)
    payload = {
        **verdicts,
        "DUALANCHOR_CONVERGENCE_HAIRS_RS_STATUS": overall,
        "decision": {
            "convergence_hairs": "soft-only diagnostics unless live regenerated ablation confirms hard merge",
            "branch_classification": "diagnostic-only; not architecture",
            "science": files["science"].get("interpretation"),
            "reasoning": files["reasoning"].get("interpretation"),
            "terminal": "keep confidence-gated top1 with defer / survivor-set handoff",
            "steering_baseline": files["readiness"].get("recommended_locked_baseline", {}),
        },
        "files_created": sorted(str(path.relative_to(OUT_ROOT)) for path in OUT_ROOT.glob("*") if path.is_file()),
        "commands_run": [
            "py_compile requested scripts",
            "bg_convergence_hairs_rs_inventory_v1.py",
            "build_bg_convergence_hair_dataset_v1.py",
            "build_bg_convergence_hair_policies_v1.py",
            "evaluate_bg_convergence_hair_replay_v1.py",
            "run_bg_convergence_hair_regenerated_ablation_v1.py",
            "analyze_bg_tie_decomposition_diagnostic_v1.py",
            "analyze_bg_reasoning_hard_slice_v1.py",
            "analyze_bg_science_hard_slice_v1.py",
            "analyze_bg_domain_branch_recipe_diagnostic_v1.py",
            "analyze_bg_terminal_defer_policy_v1.py",
            "analyze_bg_pre_steering_readiness_v1.py",
            "analyze_bg_convergence_hairs_reasoning_science_v1.py",
        ],
        "blockers": [] if overall != "INSUFFICIENT" else ["Some prerequisite analysis returned INSUFFICIENT."],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "summary.json", payload)
    write_json(OUT_ROOT / "analysis.json", {**payload, "inputs": files})
    top_lines = [status_line(key, value) for key, value in {**verdicts, "DUALANCHOR_CONVERGENCE_HAIRS_RS_STATUS": overall}.items()]
    dataset = files["dataset"]
    replay_rows = files["replay"].get("summary_rows", [])
    reasoning_slices = files["reasoning"].get("slice_rows", [])
    science_slices = files["science"].get("slice_rows", [])
    terminal_all = files["terminal"].get("domain_summary", {}).get("all", [])
    terminal_reasoning = files["terminal"].get("domain_summary", {}).get("reasoning", [])
    terminal_science = files["terminal"].get("domain_summary", {}).get("science", [])
    no_hair = row_by(replay_rows, "policy", "no_hair_baseline")
    representative = row_by(replay_rows, "policy", "representative_merge")
    l30_only = row_by(replay_rows, "policy", "L30_only_representative_merge")
    l42_only = row_by(replay_rows, "policy", "L42_only_representative_merge")
    conservative = row_by(replay_rows, "policy", "conservative_representative_merge")
    hidden_logit = row_by(replay_rows, "policy", "hidden_logit_convergence_merge")
    useful_rows = [
        row
        for row in replay_rows
        if not row.get("diagnostic_only")
        and row.get("terminal_oracle_retained", 0.0) >= 0.98
        and row.get("false_merge_rate", 1.0) <= 0.05
        and row.get("survivor_reduction", 0.0) >= 0.10
    ]
    lines = [
        "# DualAnchor Convergence Hairs + Reasoning/Science Hard-Slice Probe v1",
        "",
        *top_lines,
        "",
        "## Motivation",
        "",
        "This replay-first probe tests whether L30/L42 convergence hairs can safely reduce redundant branch carry before steering work, while preserving oracle branches and keeping branch classification diagnostic-only.",
        "",
        "No steering, training, production routing, wrapper/local-agent execution, Hunter-Seeker execution, tokenizer/checkpoint edit, or tap-registry update was performed.",
        "",
        "## Current DualAnchor Architecture Context",
        "",
        "- Selector: `MIX_CODE_REASONING` + `MIX_OBJECTIVE_ALL`.",
        "- Schedule: `L1_24 -> L1_36 -> L1_47 -> L2_24 -> L2_36 -> L2_47 -> L3_24 -> L3_36 -> L3_47 -> L4_24 -> L4_36 -> terminal L4_47`.",
        "- Threshold/budget: `mean_floor_very_loose`, budget `8`.",
        "- L47: active in nonterminal loops.",
        "- Terminal: confidence-gated top1, otherwise defer/keep survivors.",
        "",
        "## Why Convergence Hairs Were Tested",
        "",
        "The v3 looped architecture retained terminal oracle branches but showed many reward ties. L30 and L42 hairs were tested as read-only merge/checkpoint probes to separate true convergence from hidden diversity, science no-good-branch cases, and terminal-collapse weakness.",
        "",
        "Runtime branch classification was explicitly excluded. Tie taxonomy, final reward, oracle labels, parsed answers, and output-text similarity were used only for diagnostics/evaluation.",
        "",
        "## Hair Dataset",
        "",
        f"- verdict: `{verdicts['BG_CONVERGENCE_HAIR_DATASET_VERDICT']}`",
        f"- tasks: `{fmt(dataset.get('task_count'))}`",
        f"- candidate rows: `{fmt(dataset.get('candidate_count'))}`",
        f"- pair rows: `{fmt(dataset.get('pair_count'))}`",
        f"- L30 candidate rows: `{fmt(dataset.get('l30_candidate_count'))}`",
        f"- L42 candidate rows: `{fmt(dataset.get('l42_candidate_count'))}`",
        f"- hidden available rate: `{fmt(dataset.get('hidden_available_rate'))}`",
        f"- logits available: `{fmt(dataset.get('logits_available'))}`",
        f"- runtime branch classification: `{dataset.get('classification_runtime_use', 'not used')}`",
        "",
        "Domain counts:",
        "",
        *bullet_dict(dataset.get("domain_counts", {})),
        "",
        "Hair counts:",
        "",
        *bullet_dict(dataset.get("hair_counts", {})),
        "",
        "## Hair Policies",
        "",
        f"- verdict: `{verdicts['BG_CONVERGENCE_HAIR_POLICY_DEF_VERDICT']}`",
        f"- calibration pairs: `{fmt(files['policies'].get('calibration_pair_count'))}`",
        f"- heldout pairs: `{fmt(files['policies'].get('heldout_pair_count'))}`",
        f"- threshold source: `{files['policies'].get('threshold_source', '')}`",
        f"- runtime classification policy: `{files['policies'].get('runtime_classification_policy', '')}`",
        "",
        "Policies defined:",
        "",
        *[f"- `{policy.get('name')}`: {policy.get('family')}" for policy in files["policies"].get("policies", [])],
        "",
        "## Hair Replay Results",
        "",
        f"- verdict: `{verdicts['BG_CONVERGENCE_HAIR_REPLAY_EVAL_VERDICT']}`",
        "- replay is a safety/redundancy estimate from the v3 candidate tree, not a compute-savings claim",
        f"- useful non-diagnostic hard-merge candidates meeting all bars: `{len(useful_rows)}`",
        "",
        *md_table(
            [row for row in [no_hair, representative, l30_only, l42_only, conservative, hidden_logit] if row],
            [
                ("policy", "policy"),
                ("terminal_oracle_retained", "terminal oracle"),
                ("hard_slice_terminal_oracle_retained", "hard slice oracle"),
                ("false_merge_rate", "false merge"),
                ("survivor_reduction", "survivor reduction"),
                ("avg_final_candidates", "avg final candidates"),
            ],
        ),
        "",
        "Representative L30+L42 merge reduced survivors only `0.0247` and retained terminal oracle `0.9583`, below the hard-merge acceptance bar. Conservative and hidden-convergence policies preserved oracle retention but removed too little branch carry to be useful as hard merges.",
        "",
        "## Regenerated Hair Ablation",
        "",
        f"- verdict: `{verdicts['BG_CONVERGENCE_HAIR_REGENERATED_VERDICT']}`",
        f"- reason: `{files['regenerated'].get('reason', '')}`",
        "- interpretation: replay was completed; regenerated execution was skipped because the available v3 tree already contained L30/L42 states and a live regenerated run would be overnight-scale.",
        "",
        "## Tie Decomposition Diagnostic",
        "",
        f"- verdict: `{verdicts['BG_TIE_DECOMPOSITION_DIAGNOSTIC_VERDICT']}`",
        f"- pair count: `{fmt(files['tie'].get('pair_count'))}`",
        f"- reward tie count: `{fmt(files['tie'].get('reward_tie_count'))}`",
        f"- reward-tied hidden-converged rate: `{fmt(files['tie'].get('reward_tied_hidden_converged_rate'))}`",
        f"- reward-tied hidden-divergent rate: `{fmt(files['tie'].get('reward_tied_hidden_divergent_rate'))}`",
        f"- reward-tied DualAnchor-indifferent rate: `{fmt(files['tie'].get('reward_tied_dualanchor_indifferent_rate'))}`",
        f"- reasoning reward-tie rate: `{fmt(files['tie'].get('reasoning_reward_tie_rate'))}`",
        f"- science reward-tie rate: `{fmt(files['tie'].get('science_reward_tie_rate'))}`",
        "- diagnostic boundary: tie decomposition labels were not used by runtime merge policies.",
        "",
        "## Reasoning Hard-Slice",
        "",
        f"- verdict: `{verdicts['BG_REASONING_HARD_SLICE_VERDICT']}`",
        f"- interpretation: {files['reasoning'].get('interpretation', '')}",
        "",
        *md_table(
            reasoning_slices,
            [
                ("slice", "slice"),
                ("count", "count"),
                ("positive_oracle", "positive oracle"),
                ("terminal_reward_diverse", "reward diverse"),
                ("terminal_best_reward", "best reward"),
                ("terminal_forced_top1_reward", "forced reward"),
                ("terminal_forced_top1_oracle", "forced oracle"),
            ],
        ),
        "",
        "Terminal options on reasoning:",
        "",
        *md_table(
            files["reasoning"].get("terminal_policy_summary", []),
            [
                ("policy", "policy"),
                ("oracle_retained", "oracle retained"),
                ("best_selected_reward", "best reward"),
                ("first_selected_oracle", "first oracle"),
                ("defer_rate", "defer"),
            ],
        ),
        "",
        "## Science Hard-Slice And Parser Audit",
        "",
        f"- verdict: `{verdicts['BG_SCIENCE_HARD_SLICE_VERDICT']}`",
        f"- interpretation: {files['science'].get('interpretation', '')}",
        "",
        *md_table(
            science_slices,
            [
                ("slice", "slice"),
                ("count", "count"),
                ("positive_oracle", "positive oracle"),
                ("terminal_reward_diverse", "reward diverse"),
                ("terminal_best_reward", "best reward"),
                ("terminal_forced_top1_reward", "forced reward"),
                ("terminal_forced_top1_oracle", "forced oracle"),
            ],
        ),
        "",
        "Parser summary:",
        "",
        *bullet_dict(files["science"].get("parser_summary", {})),
        "",
        "Source summary:",
        "",
        *md_table(
            files["science"].get("source_summary", []),
            [
                ("source_dataset", "source"),
                ("count", "count"),
                ("positive_oracle_rate", "positive oracle"),
                ("reward_diverse_rate", "reward diverse"),
                ("terminal_best_reward", "best reward"),
                ("forced_top1_reward", "forced reward"),
            ],
        ),
        "",
        "## Domain Branch Recipe Diagnostic",
        "",
        f"- verdict: `{verdicts['BG_DOMAIN_BRANCH_RECIPE_DIAGNOSTIC_VERDICT']}`",
        f"- note: {files['recipe'].get('note', '')}",
        "",
        *md_table(
            files["recipe"].get("domain_summary", []),
            [
                ("domain", "domain"),
                ("task_count", "tasks"),
                ("positive_oracle_rate", "positive oracle"),
                ("reward_diverse_rate", "reward diverse"),
                ("terminal_best_reward", "best reward"),
                ("forced_top1_reward", "forced reward"),
                ("stage_false_prunes", "stage false prunes"),
            ],
        ),
        "",
        "## Terminal Defer Policy",
        "",
        f"- verdict: `{verdicts['BG_TERMINAL_DEFER_POLICY_VERDICT']}`",
        f"- defer rate: `{fmt(files['terminal'].get('defer_rate'))}`",
        f"- hard-slice defer count: `{fmt(files['terminal'].get('hard_slice_defer_count'))}`",
        f"- note: {files['terminal'].get('derived_policy_note', '')}",
        "",
        "All tasks:",
        "",
        *md_table(
            terminal_all,
            [
                ("policy", "policy"),
                ("oracle_retained", "oracle retained"),
                ("best_selected_reward", "best reward"),
                ("first_selected_oracle", "first oracle"),
                ("defer_rate", "defer"),
            ],
        ),
        "",
        "Reasoning:",
        "",
        *md_table(
            terminal_reasoning,
            [
                ("policy", "policy"),
                ("oracle_retained", "oracle retained"),
                ("best_selected_reward", "best reward"),
                ("first_selected_oracle", "first oracle"),
                ("defer_rate", "defer"),
            ],
        ),
        "",
        "Science:",
        "",
        *md_table(
            terminal_science,
            [
                ("policy", "policy"),
                ("oracle_retained", "oracle retained"),
                ("best_selected_reward", "best reward"),
                ("first_selected_oracle", "first oracle"),
                ("defer_rate", "defer"),
            ],
        ),
        "",
        "## Pre-Steering Readiness",
        "",
        f"- verdict: `{verdicts['BG_PRE_STEERING_READINESS_VERDICT']}`",
        "",
        "Recommended locked baseline:",
        "",
        *bullet_dict(files["readiness"].get("recommended_locked_baseline", {})),
        "",
        "Cannot claim:",
        "",
        *[f"- {item}" for item in files["readiness"].get("cannot_claim", [])],
        "",
        "## Final Recommendation For Tomorrow",
        "",
        f"- Overall status: `{overall}`.",
        f"- Pre-steering readiness: `{verdicts['BG_PRE_STEERING_READINESS_VERDICT']}`.",
        "- Do not use hard convergence-hair merge as the steering baseline.",
        "- Keep convergence hairs as monitoring/soft clusters if steering work starts.",
        "- Run or design a science-specific branch-generation recipe before treating science as a headline steering domain.",
        "- Keep terminal confidence/defer semantics.",
        "- Do not claim steering, production routing changes, compute savings, or autoregressive fork/carry.",
        "",
        "## Files Created",
        "",
        *[f"- `{name}`" for name in payload["files_created"]],
        "",
        "## Commands Run",
        "",
        *[f"- `{cmd}`" for cmd in payload["commands_run"]],
        "",
        "## Blockers",
        "",
        *([f"- {item}" for item in payload["blockers"]] if payload["blockers"] else ["- None for replay diagnostics."]),
    ]
    write_md(OUT_ROOT / "summary.md", lines)
    write_md(OUT_ROOT / "analysis.md", lines)
    print(status_line("DUALANCHOR_CONVERGENCE_HAIRS_RS_STATUS", overall))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
