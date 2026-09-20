from __future__ import annotations

import time

from bg_convergence_hairs_rs_v1_common import (
    OUT_ROOT,
    POLICIES_JSON,
    define_policies,
    load_dataset,
    md_table,
    status_line,
    write_json,
    write_md,
)


def main() -> int:
    started = time.time()
    dataset = load_dataset()
    pair_rows = dataset.get("pair_rows") or []
    payload = define_policies(pair_rows)
    verdict = payload.get("BG_CONVERGENCE_HAIR_POLICY_DEF_VERDICT", "BLOCKED")
    payload["elapsed_seconds"] = round(time.time() - started, 3)
    write_json(POLICIES_JSON, payload)
    rows = [
        {
            "name": p.get("name"),
            "family": p.get("family"),
            "hard_merge": p.get("hard_merge"),
            "diagnostic_only": p.get("diagnostic_only", False),
            "hidden_cosine_max": p.get("hidden_cosine_max"),
            "hidden_rms_norm_max": p.get("hidden_rms_norm_max"),
            "dualanchor_abs_margin_max": p.get("dualanchor_abs_margin_max"),
            "lineage_condition": p.get("lineage_condition"),
        }
        for p in payload.get("policies", [])
    ]
    lines = [
        "# DualAnchor Convergence Hair Policies v1",
        "",
        status_line("BG_CONVERGENCE_HAIR_POLICY_DEF_VERDICT", str(verdict)),
        "",
        "## Threshold Source",
        "",
        f"- threshold source: `{payload.get('threshold_source')}`",
        f"- calibration pairs: `{payload.get('calibration_pair_count')}`",
        f"- heldout pairs: `{payload.get('heldout_pair_count')}`",
        f"- hidden RMS quantiles: `{payload.get('quantiles')}`",
        "",
        "Thresholds are selected from the calibration/non-heldout subset when available. Heldout and hard slices are evaluation-only.",
        "",
        "## Policies",
        "",
        *md_table(rows, ["name", "family", "hard_merge", "diagnostic_only", "hidden_cosine_max", "hidden_rms_norm_max", "dualanchor_abs_margin_max", "lineage_condition"]),
        "",
        "## Boundary",
        "",
        "- Representative merge keeps one existing branch; hidden averaging/weighted hidden merge are not used.",
        "- Reward-tie and random policies are negative controls and are diagnostic-only.",
        "- Runtime branch classification is forbidden and not used by these policy definitions.",
    ]
    write_md(OUT_ROOT / "convergence_hair_policies.md", lines)
    print(status_line("BG_CONVERGENCE_HAIR_POLICY_DEF_VERDICT", str(verdict)))
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

