from __future__ import annotations

from pathlib import Path

from bg_dualanchor_arch_loop_v3_common import OUT_ROOT, TRUE_CARRY_ROOT, V1_ROOT, V2_ROOT, PERTURB_LIFT_ROOT, ensure_root, report_lines, write_json, write_md


def main() -> int:
    ensure_root()
    checks = {
        "v1_lineage_report": V1_ROOT / "architecture_looped_lineage_probe.md",
        "v2_stratified_report": V2_ROOT / "architecture_looped_stratified_probe.md",
        "true_carry_report": TRUE_CARRY_ROOT / "true_carry_equivalence.md",
        "perturbation_lift_report": PERTURB_LIFT_ROOT / "perturbation_lift.md",
        "branch_generator_v1": Path("shared/utilities/tests/manual/generate_bg_branch_generator_v1_quota_branches.py"),
        "quota_v4_generator": Path("shared/utilities/tests/manual/generate_bg_hidden_origin_quota_branches_v4.py"),
        "dualanchor_v2_runner": Path("shared/utilities/tests/manual/run_bg_dualanchor_architecture_looped_stratified_probe_v2.py"),
        "guarded_policy": Path("shared/utilities/tests/manual/run_bg_dualanchor_all_loop_guarded_policy_v1.py"),
    }
    present = {name: path.exists() for name, path in checks.items()}
    verdict = "READY" if all(present.values()) else "PARTIAL"
    payload = {
        "BG_DUALANCHOR_ARCH_LOOP_V3_INVENTORY_VERDICT": verdict,
        "checks": {name: {"path": str(path), "present": present[name]} for name, path in checks.items()},
        "dualanchor_taps": ["MIX_CODE_REASONING", "MIX_OBJECTIVE_ALL"],
        "lineage_fields_required": [
            "parent_branch_id",
            "root_branch_id",
            "generation_depth",
            "perturb_count",
            "birth_layer",
            "birth_loop",
            "birth_stage",
            "perturbation_family",
            "perturbation_alpha",
            "perturbation_seed",
            "lineage_path",
        ],
        "known_boundary": "Prompt-only layer carry validated; autoregressive KV/cache fork-carry not claimed.",
    }
    write_json(OUT_ROOT / "inventory.json", payload)
    lines = report_lines(
        "DualAnchor Architecture Loop v3 Inventory",
        "BG_DUALANCHOR_ARCH_LOOP_V3_INVENTORY_VERDICT",
        verdict,
        [
            ("Checks", [f"- {name}: `{present[name]}` ({path})" for name, path in checks.items()]),
            ("Boundary", ["- No steering.", "- No autoregressive true fork/carry claim.", "- No compute-savings claim."]),
        ],
    )
    write_md(OUT_ROOT / "inventory.md", lines)
    print(f"BG_DUALANCHOR_ARCH_LOOP_V3_INVENTORY_VERDICT = {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

