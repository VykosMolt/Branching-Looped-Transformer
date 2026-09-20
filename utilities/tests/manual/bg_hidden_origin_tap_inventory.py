"""Inventory existing same-prefix hidden-origin branch outcome data."""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any

from bg_hidden_origin_tap_common import (
    CONFIGS,
    OLD_ROOT,
    OUT_ROOT,
    available_configs_for_rows,
    branch_group_subset,
    ensure_out_root,
    group_is_behaviorally_diverse,
    group_is_reward_diverse,
    group_rows,
    is_diagnostic_high_alpha,
    is_safe_alpha,
    load_all_branch_rows,
    load_json,
    md_table,
    rel,
    stable_row,
    write_json,
    write_md,
)


OUT_JSON = OUT_ROOT / "inventory.json"
OUT_MD = OUT_ROOT / "inventory.md"


def _branches_per_group_distribution(groups: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    counts = Counter(len(vals) for vals in groups.values())
    return {str(k): int(v) for k, v in sorted(counts.items())}


def main() -> int:
    ensure_out_root()
    started = time.time()
    rows = load_all_branch_rows()
    groups = group_rows(rows)
    stable_rows = [row for row in rows if stable_row(row)]
    safe_rows = [row for row in rows if is_safe_alpha(row)]
    diagnostic_rows = [row for row in rows if is_diagnostic_high_alpha(row)]
    stable_safe = [row for row in rows if is_safe_alpha(row) and stable_row(row)]
    stable_groups = group_rows(stable_safe)
    behaviorally_diverse = [gid for gid, vals in stable_groups.items() if len(vals) >= 2 and group_is_behaviorally_diverse(vals)]
    reward_diverse = [gid for gid, vals in stable_groups.items() if len(vals) >= 2 and group_is_reward_diverse(vals)]

    persistence_pt = OLD_ROOT / "hidden_branch_persistence.pt"
    outcomes_json = OLD_ROOT / "hidden_branch_outcomes.json"
    selection_json = OLD_ROOT / "hidden_origin_branch_selection.json"
    task_subset_json = OLD_ROOT / "task_subset.json"
    persistence_meta = load_json(OLD_ROOT / "hidden_branch_persistence.json", {}) or {}
    outcomes_meta = load_json(outcomes_json, {}) or {}
    selection_meta = load_json(selection_json, {}) or {}

    feature_coverage = available_configs_for_rows(rows)
    layers_available = sorted(
        {
            key.split("_")[0]
            for row in rows
            for key in (row.get("pooled_vectors") or {}).keys()
            if key.startswith("L")
        }
    )
    if not layers_available and any(row.get("features") is not None for row in rows):
        layers_available = ["L24", "L36", "L47"]

    if not rows:
        verdict = "BLOCKED"
        enough_for = "only data expansion"
    elif len(stable_groups) >= 50 and len(behaviorally_diverse) >= 15:
        verdict = "READY"
        enough_for = "real heldout training"
    elif len(stable_groups) >= 20 and len(behaviorally_diverse) >= 5:
        verdict = "SMALL_BUT_USABLE"
        enough_for = "tiny sanity training; heldout is data-limited unless expanded"
    else:
        verdict = "NEEDS_DATA_EXPANSION"
        enough_for = "only data expansion"

    payload = {
        "BG_HIDDEN_ORIGIN_TAP_INVENTORY_VERDICT": verdict,
        "verdict": verdict,
        "input_root": rel(OLD_ROOT),
        "files": {
            "hidden_branch_persistence_pt": persistence_pt.exists(),
            "hidden_branch_outcomes_json": outcomes_json.exists(),
            "hidden_origin_branch_selection_json": selection_json.exists(),
            "task_subset_json": task_subset_json.exists(),
        },
        "total_rows": len(rows),
        "total_branch_groups": len(groups),
        "safe_rows": len(safe_rows),
        "diagnostic_high_alpha_rows": len(diagnostic_rows),
        "stable_rows": len(stable_rows),
        "stable_safe_rows": len(stable_safe),
        "stable_safe_branch_groups": len(stable_groups),
        "tasks": len({row.get("task_id") for row in rows}),
        "task_ids": sorted({str(row.get("task_id")) for row in rows}),
        "domains": dict(Counter(str(row.get("domain")) for row in rows)),
        "branches_per_group_distribution": _branches_per_group_distribution(groups),
        "behaviorally_diverse_group_count": len(behaviorally_diverse),
        "reward_diverse_group_count": len(reward_diverse),
        "behaviorally_diverse_group_ids": sorted(behaviorally_diverse),
        "reward_diverse_group_ids": sorted(reward_diverse),
        "layers_available": layers_available,
        "config_feature_coverage": feature_coverage,
        "features_present": any(count > 0 for count in feature_coverage.values()),
        "missing_primary_configs": [config for config in CONFIGS[:9] if feature_coverage.get(config, 0) == 0],
        "enough_for": enough_for,
        "prior_outcome_stats": outcomes_meta.get("stats", {}),
        "prior_selection_verdict": selection_meta.get("BG_HIDDEN_ORIGIN_BRANCH_SELECTION_VERDICT"),
        "prior_generation_verdict": persistence_meta.get("BG_HIDDEN_BRANCH_GENERATION_VERDICT"),
        "prior_persistence_verdict": persistence_meta.get("BG_LATENT_BRANCH_PERSISTENCE_VERDICT"),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(OUT_JSON, payload)

    group_rows_for_md = [
        {
            "branch_group_id": gid,
            "task_id": vals[0].get("task_id"),
            "domain": vals[0].get("domain"),
            "alpha": vals[0].get("alpha"),
            "branches": len(vals),
            "stable": sum(1 for row in vals if stable_row(row)),
            "rewards": sorted({row.get("reward") for row in vals}),
            "behaviorally_diverse": group_is_behaviorally_diverse(vals),
        }
        for gid, vals in sorted(groups.items())
    ]
    lines = [
        "# Hidden-Origin Branch Tap Inventory",
        "",
        f"BG_HIDDEN_ORIGIN_TAP_INVENTORY_VERDICT = {verdict}",
        "",
        f"- total_rows: `{len(rows)}`",
        f"- total_branch_groups: `{len(groups)}`",
        f"- stable_safe_branch_groups: `{len(stable_groups)}`",
        f"- behaviorally_diverse_group_count: `{len(behaviorally_diverse)}`",
        f"- reward_diverse_group_count: `{len(reward_diverse)}`",
        f"- tasks: `{payload['tasks']}`",
        f"- domains: `{payload['domains']}`",
        f"- enough_for: `{enough_for}`",
        "",
        "## Feature Coverage",
        "",
    ]
    lines.extend(md_table(
        [{"config": config, "rows": feature_coverage.get(config, 0)} for config in CONFIGS],
        ["config", "rows"],
    ))
    lines.extend(["", "## Branch Groups", ""])
    lines.extend(md_table(
        group_rows_for_md[:80],
        ["branch_group_id", "task_id", "domain", "alpha", "branches", "stable", "rewards", "behaviorally_diverse"],
    ))
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_TAP_INVENTORY_VERDICT = {verdict}", flush=True)
    print(f"Wrote {rel(OUT_JSON)}", flush=True)
    print(f"Wrote {rel(OUT_MD)}", flush=True)
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
