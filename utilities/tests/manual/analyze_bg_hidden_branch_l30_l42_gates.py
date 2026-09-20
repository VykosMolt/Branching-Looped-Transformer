"""Assess whether L30/L42 convergence gates are justified for hidden branches."""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from bg_hidden_branch_suite_common import REPORT_ROOT, ensure_report_root, finite, load_json, md_table, rel, write_csv, write_json, write_md


PERSISTENCE_JSON = REPORT_ROOT / "hidden_branch_persistence.json"
OUTCOMES_JSON = REPORT_ROOT / "hidden_branch_outcomes.json"
SELECTION_JSON = REPORT_ROOT / "hidden_origin_branch_selection.json"
OUT_JSON = REPORT_ROOT / "l30_l42_gate_assessment.json"
OUT_MD = REPORT_ROOT / "l30_l42_gate_assessment.md"
OUT_CSV = REPORT_ROOT / "l30_l42_gate_rows.csv"


def gate_stats(groups: list[dict[str, Any]], outcomes_by_group: dict[str, list[dict[str, Any]]], key: str) -> dict[str, Any]:
    rows = []
    for group in groups:
        geom = group.get("geometry", {}).get(key, {})
        retention = finite(geom.get("retention_vs_branch_point"))
        collapsed = retention < 0.10 or finite(geom.get("mean_rms_distance")) < 1e-4
        outs = outcomes_by_group.get(group["branch_group_id"], [])
        oracle = any(o.get("correct") for o in outs)
        mixed_reward = len({float(o.get("reward", 0.0)) for o in outs}) > 1
        rows.append(
            {
                "branch_group_id": group["branch_group_id"],
                "task_id": group["task_id"],
                "domain": group["domain"],
                "alpha": group["alpha"],
                "gate": key,
                "retention": retention,
                "collapsed": collapsed,
                "oracle_available": oracle,
                "outcome_mixed_cluster": mixed_reward and collapsed,
            }
        )
    if not rows:
        return {"gate": key, "rows": 0, "collapse_rate": 0.0, "oracle_retention_if_merge": 0.0, "mixed_cluster_rate": 0.0, "compute_saved_proxy": 0.0}
    collapse_rate = sum(1 for r in rows if r["collapsed"]) / len(rows)
    mixed_rate = sum(1 for r in rows if r["outcome_mixed_cluster"]) / len(rows)
    oracle_groups = [r for r in rows if r["oracle_available"]]
    false_merge = [r for r in oracle_groups if r["outcome_mixed_cluster"]]
    oracle_retention = 1.0 - len(false_merge) / max(len(oracle_groups), 1)
    return {
        "gate": key,
        "rows": len(rows),
        "collapse_rate": collapse_rate,
        "oracle_retention_if_merge": oracle_retention,
        "mixed_cluster_rate": mixed_rate,
        "compute_saved_proxy": collapse_rate * 0.5,
        "rows_detail": rows,
    }


def main() -> int:
    ensure_report_root()
    started = time.time()
    persistence = load_json(PERSISTENCE_JSON, {})
    outcomes = load_json(OUTCOMES_JSON, {})
    selection = load_json(SELECTION_JSON, {})
    groups = [g for g in persistence.get("groups", []) if g.get("safety_envelope")]
    outcomes_by_group = defaultdict(list)
    for row in outcomes.get("rows") or []:
        if row.get("safety_envelope"):
            outcomes_by_group[row["branch_group_id"]].append(row)
    l30 = gate_stats(groups, outcomes_by_group, "L30_L1")
    l42 = gate_stats(groups, outcomes_by_group, "L42_L1")
    rows = list(l30.get("rows_detail", [])) + list(l42.get("rows_detail", []))
    if not groups:
        verdict = "NEEDS_MORE_LIVE_DATA"
    elif persistence.get("BG_HIDDEN_BRANCH_GENERATION_VERDICT") == "BLOCKED":
        verdict = "STATE_HANDLING_BLOCKED"
    elif l30["collapse_rate"] > 0.70 and l42["collapse_rate"] > 0.70:
        verdict = "NEEDS_STRONGER_BRANCH_GENERATOR"
    elif l30["mixed_cluster_rate"] > 0.05 or l42["mixed_cluster_rate"] > 0.05:
        verdict = "OMIT_CONVERGENCE_GATES_FOR_FIRST_PROTOTYPE"
    elif l30["collapse_rate"] > 0.40 and l30["oracle_retention_if_merge"] >= 0.95 and l42["collapse_rate"] > 0.40:
        verdict = "INCLUDE_L30_AND_L42"
    elif l42["collapse_rate"] > 0.40 and l42["oracle_retention_if_merge"] >= 0.95:
        verdict = "INCLUDE_L42_ONLY"
    elif l30["collapse_rate"] > 0.40 and l30["oracle_retention_if_merge"] >= 0.95:
        verdict = "INCLUDE_L30_ONLY"
    else:
        verdict = "OMIT_CONVERGENCE_GATES_FOR_FIRST_PROTOTYPE"
    payload = {
        "BG_HIDDEN_BRANCH_L30_L42_GATE_VERDICT": verdict,
        "verdict": verdict,
        "group_count": len(groups),
        "l30": {k: v for k, v in l30.items() if k != "rows_detail"},
        "l42": {k: v for k, v in l42.items() if k != "rows_detail"},
        "selection_verdict": selection.get("BG_HIDDEN_ORIGIN_BRANCH_SELECTION_VERDICT"),
        "interpretation": "Gates are assessed only from hidden-origin branch geometry/outcomes, not cached token/candidate branch persistence.",
        "rows": rows,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)
    write_csv(OUT_CSV, rows)
    lines = [
        "# BG Hidden Branch L30/L42 Gate Assessment",
        "",
        f"BG_HIDDEN_BRANCH_L30_L42_GATE_VERDICT = {verdict}",
        "",
        f"- group_count: `{len(groups)}`",
        f"- L30 collapse_rate: `{l30['collapse_rate']:.3f}`",
        f"- L42 collapse_rate: `{l42['collapse_rate']:.3f}`",
        f"- L30 mixed_cluster_rate: `{l30['mixed_cluster_rate']:.3f}`",
        f"- L42 mixed_cluster_rate: `{l42['mixed_cluster_rate']:.3f}`",
        "",
        "Representative merge is not recommended unless geometric collapse is strong and outcome clusters are not mixed.",
        "",
        "## Rows",
        "",
    ]
    lines.extend(md_table(rows[:30], ["branch_group_id", "domain", "alpha", "gate", "retention", "collapsed", "oracle_available", "outcome_mixed_cluster"]))
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_BRANCH_L30_L42_GATE_VERDICT = {verdict}")
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")
    return 0 if groups else 1


if __name__ == "__main__":
    raise SystemExit(main())
