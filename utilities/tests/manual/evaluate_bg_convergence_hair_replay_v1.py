from __future__ import annotations

import time

from bg_convergence_hairs_rs_v1_common import (
    OUT_ROOT,
    POLICIES_JSON,
    REPLAY_JSON,
    REPLAY_ROWS_CSV,
    load_dataset,
    md_table,
    read_json,
    replay_policies,
    safe_float,
    status_line,
    write_csv,
    write_json,
    write_md,
)


def main() -> int:
    started = time.time()
    dataset = load_dataset()
    policies = read_json(POLICIES_JSON, {}) or {}
    if not policies:
        policies = {"policies": []}
    result = replay_policies(dataset, policies)
    result["elapsed_seconds"] = round(time.time() - started, 3)
    rows = result.get("rows") or []
    summary_rows = result.get("summary_rows") or []
    write_json(REPLAY_JSON, result)
    write_csv(REPLAY_ROWS_CSV, rows)
    write_csv(OUT_ROOT / "convergence_hair_replay_summary_rows.csv", summary_rows)
    verdict = result.get("BG_CONVERGENCE_HAIR_REPLAY_EVAL_VERDICT", "INSUFFICIENT")
    useful = [
        row
        for row in summary_rows
        if row.get("policy") not in {"no_hair_baseline", "soft_cluster_diagnostic"}
        and not row.get("diagnostic_only")
        and safe_float(row.get("survivor_reduction"), 0.0) >= 0.10
    ]
    lines = [
        "# DualAnchor Convergence Hair Replay Evaluation v1",
        "",
        status_line("BG_CONVERGENCE_HAIR_REPLAY_EVAL_VERDICT", str(verdict)),
        "",
        "Replay evaluation estimates merge safety and redundancy from the existing v3 candidate tree. It is not a compute-savings claim.",
        "",
        "## Policy Summary",
        "",
        *md_table(summary_rows, ["policy", "task_count", "diagnostic_only", "terminal_oracle_retained", "hard_slice_terminal_oracle_retained", "false_merge_rate", "survivor_reduction", "avg_final_candidates"]),
        "",
        "## Useful Non-Diagnostic Merge Candidates",
        "",
        *md_table(useful, ["policy", "terminal_oracle_retained", "hard_slice_terminal_oracle_retained", "false_merge_rate", "survivor_reduction"]),
        "",
        "## Boundary",
        "",
        "- Counterfactual continuation is represented by the already generated v3 tree and subtree rewards.",
        "- No branch is actually skipped during generation in this replay, so compute savings are not claimed.",
    ]
    write_md(OUT_ROOT / "convergence_hair_replay_eval.md", lines)
    print(status_line("BG_CONVERGENCE_HAIR_REPLAY_EVAL_VERDICT", str(verdict)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

