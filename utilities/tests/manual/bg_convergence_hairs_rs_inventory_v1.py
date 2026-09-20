from __future__ import annotations

import math
import time
from collections import Counter

from bg_convergence_hairs_rs_v1_common import (
    ANCHOR_A,
    ANCHOR_B,
    OUT_ROOT,
    TRUE_CARRY_ROOT,
    V3_HARD_ROWS,
    V3_PT,
    artifact_status,
    finite_mean,
    hair_vector,
    load_dualanchor_taps,
    load_v3_rows,
    md_table,
    read_csv,
    read_json,
    safe_float,
    status_line,
    write_json,
    write_md,
)


def main() -> int:
    started = time.time()
    payload, rows, stage_rows, task_rows, terminal_rows = load_v3_rows()
    artifacts = artifact_status()
    taps = load_dualanchor_taps()
    tap_layers = sorted(taps)
    dualanchor_ready = any(ANCHOR_A in layer_taps and ANCHOR_B in layer_taps for layer_taps in taps.values())
    sample_rows = rows[: min(len(rows), 128)]
    l30_available = bool(sample_rows) and all(hair_vector(row, 30, 1) is not None for row in sample_rows[:10])
    l42_available = bool(sample_rows) and all(hair_vector(row, 42, 1) is not None for row in sample_rows[:10])
    hair_keys = sorted((rows[0].get("pooled_vectors") or {}).keys()) if rows else []
    hard_rows = read_csv(V3_HARD_ROWS)
    task_domains = Counter(str(row.get("domain")) for row in task_rows)
    hard_counts = {
        "positive_oracle": sum(1 for row in task_rows if safe_float(row.get("positive_oracle"), 0.0) > 0),
        "reward_diverse": sum(1 for row in task_rows if safe_float(row.get("terminal_reward_diverse"), 0.0) > 0),
        "positive_and_reward_diverse": sum(
            1
            for row in task_rows
            if safe_float(row.get("positive_oracle"), 0.0) > 0 and safe_float(row.get("terminal_reward_diverse"), 0.0) > 0
        ),
        "science_positive_oracle": sum(1 for row in task_rows if row.get("domain") == "science" and safe_float(row.get("positive_oracle"), 0.0) > 0),
        "reasoning_reward_diverse": sum(1 for row in task_rows if row.get("domain") == "reasoning" and safe_float(row.get("terminal_reward_diverse"), 0.0) > 0),
    }
    expected_missing = [row for row in artifacts if row["status"] == "MISSING"]
    partial = [row for row in artifacts if row["status"] == "PARTIAL"]
    true_carry = read_json(TRUE_CARRY_ROOT / "summary.json", {}) or {}
    if not V3_PT.exists() or not rows:
        verdict = "BLOCKED"
    elif not l30_available or not l42_available:
        verdict = "MISSING_HAIR_FEATURES"
    elif expected_missing:
        verdict = "PARTIAL"
    elif partial:
        verdict = "PARTIAL"
    elif not dualanchor_ready:
        verdict = "BLOCKED"
    else:
        verdict = "READY"
    out = {
        "BG_CONVERGENCE_HAIRS_RS_INVENTORY_VERDICT": verdict,
        "artifact_rows": artifacts,
        "dualanchor_taps_ready": dualanchor_ready,
        "tap_layers": tap_layers,
        "task_counts_by_domain": dict(task_domains),
        "task_count": len(task_rows),
        "row_count": len(rows),
        "stage_decision_count": len(stage_rows),
        "terminal_policy_rows": len(terminal_rows),
        "hard_slice_rows": hard_rows,
        "hard_slice_counts": hard_counts,
        "lineage_coverage": {
            "rows_with_branch_id": sum(1 for row in rows if row.get("branch_id")),
            "rows_with_parent": sum(1 for row in rows if row.get("parent_branch_id")),
            "rows_with_lineage_path": sum(1 for row in rows if row.get("lineage_path")),
        },
        "feature_availability": {
            "l30_pooled_available": l30_available,
            "l42_pooled_available": l42_available,
            "l30_l42_keys": [key for key in hair_keys if key.startswith("L30") or key.startswith("L42")],
            "logits_available": False,
            "topk_logits_available": False,
        },
        "true_carry_summary": true_carry,
        "missing_fields": {
            "logits_at_hairs": "not present in v3 artifact",
            "autoregressive_kv_fork_carry": "not implemented; out of scope",
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_ROOT / "inventory.json", out)
    lines = [
        "# DualAnchor Convergence Hairs RS Inventory v1",
        "",
        status_line("BG_CONVERGENCE_HAIRS_RS_INVENTORY_VERDICT", verdict),
        "",
        "## Artifact Availability",
        "",
        *md_table(artifacts, ["artifact", "status", "missing_expected"]),
        "",
        "## Feature Availability",
        "",
        f"- L30 pooled hidden states: `{l30_available}`",
        f"- L42 pooled hidden states: `{l42_available}`",
        "- logits/top-k logits at hairs: `False`",
        "- DualAnchor margins use adjacent next-stage tap scores as a read-only proxy; no L30/L42 tap is introduced.",
        "",
        "## Counts",
        "",
        f"- tasks: `{len(task_rows)}`",
        f"- domains: `{dict(task_domains)}`",
        f"- generated/evaluated rows: `{len(rows)}`",
        f"- nonterminal stage decisions: `{len(stage_rows)}`",
        f"- hard-slice counts: `{hard_counts}`",
        "",
        "## Boundary",
        "",
        "- Branch classification is diagnostic-only and is not part of the runtime architecture.",
        "- This inventory does not run steering, wrapper/local-agent code, training, or true autoregressive fork/carry.",
    ]
    write_md(OUT_ROOT / "inventory.md", lines)
    print(status_line("BG_CONVERGENCE_HAIRS_RS_INVENTORY_VERDICT", verdict))
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

