from __future__ import annotations

from bg_dualanchor_arch_loop_v3_common import OUT_ROOT, finite_mean, load_stage_rows, report_lines, safe_float, write_csv, write_json, write_md


def main() -> int:
    stages = load_stage_rows()
    trigger_rows = []
    for row in stages:
        low_survivors = safe_float(row.get("survivors_after"), 99) < 4
        false_prune = safe_float(row.get("stage_false_prune"), 0.0) > 0
        if low_survivors or false_prune:
            trigger_rows.append({**row, "would_trigger_escalation": 1.0, "trigger_low_survivors": 1.0 if low_survivors else 0.0, "trigger_false_prune": 1.0 if false_prune else 0.0})
    trigger_rate = len(trigger_rows) / len(stages) if stages else 0.0
    retention = finite_mean(row.get("stage_oracle_retained") for row in stages)
    verdict = "ESCALATION_UNNEEDED" if retention >= 0.98 and trigger_rate <= 0.05 else "DIAGNOSTIC_ONLY"
    payload = {
        "BG_DUALANCHOR_PERTURBATION_ESCALATION_V3_VERDICT": verdict,
        "stage_count": len(stages),
        "trigger_count": len(trigger_rows),
        "trigger_rate": trigger_rate,
        "stage_oracle_retention": retention,
        "note": "No escalated generation was run; this is a trigger audit over v3 primary rows.",
    }
    write_json(OUT_ROOT / "perturbation_escalation.json", payload)
    write_csv(OUT_ROOT / "perturbation_escalation_rows.csv", trigger_rows)
    lines = report_lines("DualAnchor Perturbation Escalation v3", "BG_DUALANCHOR_PERTURBATION_ESCALATION_V3_VERDICT", verdict, [("Trigger Audit", [f"- {k}: `{v}`" for k, v in payload.items() if k != "note"]), ("Note", [payload["note"]])])
    write_md(OUT_ROOT / "perturbation_escalation.md", lines)
    print(f"BG_DUALANCHOR_PERTURBATION_ESCALATION_V3_VERDICT = {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

