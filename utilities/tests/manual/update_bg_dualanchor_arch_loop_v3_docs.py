from __future__ import annotations

from pathlib import Path

from bg_dualanchor_arch_loop_v3_common import OUT_ROOT, read_json

DOC = Path("docs/evaluator/bg_dualanchor_architecture_looped_stratified_probe_v3.md")
APPEND_DOCS = [
    Path("docs/evaluator/current_state.md"),
    Path("docs/evaluator/domain_transfer_ledger.md"),
    Path("docs/evaluator/bg_selection_only_phase2_prototype_v1.md"),
    Path("docs/evaluator/bg_fixed_composite_branch_survival_policy_v1.md"),
    Path("docs/evaluator/bg_gated_branch_content_selector_v1.md"),
    Path("docs/evaluator/bg_universal_branch_content_taps_v1.md"),
    Path("docs/evaluator/bg_hidden_origin_branch_generator_v1.md"),
    Path("docs/evaluator/bg_hidden_origin_quota_v4.md"),
    Path("docs/evaluator/bg_hidden_origin_taps.md"),
    Path("docs/evaluator/bg_hidden_state_branch_generation.md"),
    Path("docs/evaluator/bg_steering_consolidation_2026-05-18.md"),
    Path("docs/evaluator/post_v10_synthesis_2026-05-18_v8.1_routing_locked.md"),
]


def main() -> int:
    summary = read_json(OUT_ROOT / "summary.json", {}) or {}
    run = read_json(OUT_ROOT / "run_report.json", {}) or {}
    readiness = read_json(OUT_ROOT / "phase2a_readiness.json", {}) or {}
    status = summary.get("DUALANCHOR_ARCHITECTURE_LOOPED_V3_STATUS", "INSUFFICIENT")
    headline = summary.get("run_headline", {})
    section = f"""## DualAnchor architecture-looped stratified probe v3 (2026-05-31)

Status: `{status}`.

This run scaled the DualAnchor architecture-shaped loop without steering. Taps were active at layers 24, 36, and 47 across loops L1-L4, with only terminal `L4_47` eligible for confidence-gated collapse. It uses cumulative hook approximation at decoder-layer surfaces; it does not claim autoregressive branch-specific KV/cache fork/carry or compute savings.

Headline metrics:

- tasks: `{headline.get('tasks')}`
- stage oracle retention: `{headline.get('stage_oracle_retention')}`
- terminal oracle retained: `{headline.get('terminal_oracle_retained')}`
- terminal forced top1 oracle: `{headline.get('terminal_forced_top1_oracle')}`
- terminal reward-diverse rate: `{headline.get('terminal_reward_diverse')}`
- positive-oracle rate: `{headline.get('positive_oracle')}`

Locked-baseline candidate:

- selector: DualAnchor `MIX_CODE_REASONING` + `MIX_OBJECTIVE_ALL`
- schedule: `L1_24 -> L1_36 -> L1_47 -> L2_24 -> L2_36 -> L2_47 -> L3_24 -> L3_36 -> L3_47 -> L4_24 -> L4_36 -> terminal L4_47`
- threshold: `mean_floor_very_loose`
- budget: `8`
- L47: active in nonterminal loops
- terminal: confidence-gated top1; otherwise defer/keep terminal survivors

Readiness verdict: `{readiness.get('BG_DUALANCHOR_PHASE2A_READINESS_V3_VERDICT', 'INSUFFICIENT')}`.
No steering was tested.
"""
    DOC.write_text(
        "# DualAnchor architecture-looped stratified probe v3\n\n"
        "This document records the v3 architecture-shaped DualAnchor branch/prune probe.\n\n"
        + section
        + "\n## Artifact Root\n\n"
        f"`{OUT_ROOT}`\n"
    )
    append_section = "\n\n" + section
    for path in APPEND_DOCS:
        if not path.exists():
            continue
        text = path.read_text()
        title = "## DualAnchor architecture-looped stratified probe v3 (2026-05-31)"
        if title in text:
            continue
        path.write_text(text.rstrip() + append_section + "\n")
    print("BG_DUALANCHOR_ARCH_LOOP_V3_DOCS_UPDATED = true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

