from __future__ import annotations

import time

from bg_convergence_hairs_rs_v1_common import OUT_ROOT, read_json, status_line, write_json, write_md


def main() -> int:
    started = time.time()
    replay = read_json(OUT_ROOT / "convergence_hair_replay_eval.json", {}) or {}
    regenerated = read_json(OUT_ROOT / "convergence_hair_regenerated.json", {}) or {}
    tie = read_json(OUT_ROOT / "tie_decomposition_diagnostic.json", {}) or {}
    reasoning = read_json(OUT_ROOT / "reasoning_hard_slice.json", {}) or {}
    science = read_json(OUT_ROOT / "science_hard_slice.json", {}) or {}
    recipe = read_json(OUT_ROOT / "domain_branch_recipe.json", {}) or {}
    terminal = read_json(OUT_ROOT / "terminal_defer_policy.json", {}) or {}
    replay_verdict = replay.get("BG_CONVERGENCE_HAIR_REPLAY_EVAL_VERDICT", "INSUFFICIENT")
    science_verdict = science.get("BG_SCIENCE_HARD_SLICE_VERDICT", "INSUFFICIENT")
    reasoning_verdict = reasoning.get("BG_REASONING_HARD_SLICE_VERDICT", "INSUFFICIENT")
    terminal_verdict = terminal.get("BG_TERMINAL_DEFER_POLICY_VERDICT", "INSUFFICIENT")
    recipe_verdict = recipe.get("BG_DOMAIN_BRANCH_RECIPE_DIAGNOSTIC_VERDICT", "INSUFFICIENT")
    if science_verdict == "SCIENCE_REWARD_PARSER_WEAK":
        verdict = "NEEDS_SCIENCE_REWARD_PARSER_FIRST"
    elif science_verdict in {"SCIENCE_BRANCH_GENERATION_WEAK", "SCIENCE_TIE_HEAVY_NO_GOOD_BRANCH"} or recipe_verdict == "SCIENCE_NEEDS_DIFFERENT_RECIPE":
        verdict = "NEEDS_SCIENCE_BRANCH_RECIPE_FIRST"
    elif reasoning_verdict in {"REASONING_FINAL_CHOICE_WEAK", "REASONING_NEEDS_TERMINAL_DEFER"} and terminal_verdict not in {"CONFIDENCE_TOP1_READY_WITH_DEFER", "TOP2_HANDOFF_SUFFICIENT", "TOP4_HANDOFF_REQUIRED", "FULL_SURVIVOR_SET_REQUIRED"}:
        verdict = "NEEDS_REASONING_TERMINAL_POLICY_FIRST"
    elif replay_verdict == "HARD_MERGE_SAFE" and regenerated.get("BG_CONVERGENCE_HAIR_REGENERATED_VERDICT") == "REGENERATED_CONFIRMS_SAFE_MERGE":
        verdict = "READY_FOR_STEERING_WITH_HARD_HAIRS"
    elif replay_verdict in {"HARD_MERGE_SAFE", "L30_HAIR_SAFE", "L42_HAIR_SAFE"}:
        verdict = "READY_FOR_STEERING_WITH_SOFT_HAIRS"
    elif replay_verdict in {"SOFT_CLUSTER_ONLY", "FALSE_MERGE_TOO_HIGH", "TIES_HIDE_DIVERGENCE"}:
        verdict = "READY_FOR_STEERING_WITH_SOFT_HAIRS"
    elif terminal_verdict in {"CONFIDENCE_TOP1_READY_WITH_DEFER", "TOP2_HANDOFF_SUFFICIENT", "TOP4_HANDOFF_REQUIRED", "FULL_SURVIVOR_SET_REQUIRED"}:
        verdict = "READY_FOR_STEERING_WITH_NO_HARD_MERGE"
    else:
        verdict = "INSUFFICIENT"
    payload = {
        "BG_PRE_STEERING_READINESS_VERDICT": verdict,
        "input_verdicts": {
            "BG_CONVERGENCE_HAIR_REPLAY_EVAL_VERDICT": replay_verdict,
            "BG_CONVERGENCE_HAIR_REGENERATED_VERDICT": regenerated.get("BG_CONVERGENCE_HAIR_REGENERATED_VERDICT", "INSUFFICIENT"),
            "BG_TIE_DECOMPOSITION_DIAGNOSTIC_VERDICT": tie.get("BG_TIE_DECOMPOSITION_DIAGNOSTIC_VERDICT", "INSUFFICIENT"),
            "BG_REASONING_HARD_SLICE_VERDICT": reasoning_verdict,
            "BG_SCIENCE_HARD_SLICE_VERDICT": science_verdict,
            "BG_DOMAIN_BRANCH_RECIPE_DIAGNOSTIC_VERDICT": recipe_verdict,
            "BG_TERMINAL_DEFER_POLICY_VERDICT": terminal_verdict,
        },
        "recommended_locked_baseline": {
            "schedule": "L1_24 -> L1_36 -> L1_47 -> L2_24 -> L2_36 -> L2_47 -> L3_24 -> L3_36 -> L3_47 -> L4_24 -> L4_36 -> terminal L4_47",
            "selector": ["MIX_CODE_REASONING", "MIX_OBJECTIVE_ALL"],
            "threshold": "mean_floor_very_loose",
            "budget": 8,
            "l47": "active in nonterminal loops",
            "terminal": "confidence-gated top1; otherwise terminal defer / survivor-set handoff",
            "convergence_hairs": "soft-only diagnostics unless a regenerated run later confirms hard merge safety",
            "lineage": "required",
            "steering": "not run in this probe",
        },
        "cannot_claim": [
            "steering was tested",
            "production routing changed",
            "autoregressive branch-specific KV/cache fork-carry",
            "compute savings",
            "unconditional final top1 readiness",
            "branch classification as runtime architecture",
        ],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "pre_steering_readiness.json", payload)
    lines = [
        "# Pre-Steering Readiness v1",
        "",
        status_line("BG_PRE_STEERING_READINESS_VERDICT", verdict),
        "",
        "## Input Verdicts",
        "",
        *[f"- {key} = `{value}`" for key, value in payload["input_verdicts"].items()],
        "",
        "## Recommended Locked Baseline",
        "",
        *[f"- {key}: `{value}`" for key, value in payload["recommended_locked_baseline"].items()],
        "",
        "## Non-Claims",
        "",
        *[f"- {item}" for item in payload["cannot_claim"]],
    ]
    write_md(OUT_ROOT / "pre_steering_readiness.md", lines)
    print(status_line("BG_PRE_STEERING_READINESS_VERDICT", verdict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

