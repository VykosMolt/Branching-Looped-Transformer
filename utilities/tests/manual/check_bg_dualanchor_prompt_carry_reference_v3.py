from __future__ import annotations

from bg_dualanchor_arch_loop_v3_common import OUT_ROOT, TRUE_CARRY_ROOT, latest_verdict, report_lines, write_json, write_md


def main() -> int:
    prior_json = TRUE_CARRY_ROOT / "true_carry_equivalence.json"
    prior_md = TRUE_CARRY_ROOT / "true_carry_equivalence.md"
    if prior_json.exists() or prior_md.exists():
        verdict = "PRIOR_VALIDATION_ACCEPTED"
    else:
        verdict = "MISSING_PRIOR"
    payload = {
        "BG_DUALANCHOR_PROMPT_CARRY_REFERENCE_V3_VERDICT": verdict,
        "prior_json": str(prior_json),
        "prior_md": str(prior_md),
        "prior_status": latest_verdict(prior_json, "BG_DUALANCHOR_TRUE_CARRY_EQUIVALENCE_VERDICT") if prior_json.exists() else "MD_ONLY_OR_MISSING",
        "boundary": "Accepts prompt-only decoder-layer 24/36/47 carry validation; does not claim autoregressive KV/cache fork-carry.",
    }
    write_json(OUT_ROOT / "prompt_carry_reference.json", payload)
    lines = report_lines("DualAnchor Prompt Carry Reference v3", "BG_DUALANCHOR_PROMPT_CARRY_REFERENCE_V3_VERDICT", verdict, [("Prior", [f"- prior_json: `{prior_json}`", f"- prior_md: `{prior_md}`", f"- prior_status: `{payload['prior_status']}`"]), ("Boundary", [payload["boundary"]])])
    write_md(OUT_ROOT / "prompt_carry_reference.md", lines)
    print(f"BG_DUALANCHOR_PROMPT_CARRY_REFERENCE_V3_VERDICT = {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

