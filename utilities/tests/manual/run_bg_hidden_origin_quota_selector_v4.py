"""Run the v4 hidden-origin selector anyway as a diagnostic.

This script consumes the already-generated v4 branch groups and the heldout
selector-eval rows. It writes selected branches for the best available v4
diagnostic selector without changing production routing or claiming readiness.
"""
from __future__ import annotations

import ast
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from bg_hidden_origin_quota_v4_common import (
    BRANCHES_JSON,
    V4_ROOT,
    md_table,
    rate,
    rel,
    write_csv,
    write_json,
    write_md,
)


EVAL_JSON = V4_ROOT / "heldout_eval_v4.json"
EVAL_ROWS_CSV = V4_ROOT / "heldout_eval_v4_rows.csv"
OUT_JSON = V4_ROOT / "selector_run_anyway_v4.json"
OUT_MD = V4_ROOT / "selector_run_anyway_v4.md"
OUT_CSV = V4_ROOT / "selector_run_anyway_v4.csv"
OUT_TOP2_CSV = V4_ROOT / "selector_run_anyway_v4_top2.csv"

PRIMARY_POLICY = "ensemble_rank_aggregation"
TOP2_POLICY = "ensemble_rank_aggregation_top2"
SUBSET = "primary_safe_deterministic"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_ids(value: Any) -> list[int]:
    if value in (None, "", "[]"):
        return []
    if isinstance(value, list):
        return [int(v) for v in value]
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return []
    if isinstance(parsed, list):
        return [int(v) for v in parsed]
    return []


def branch_reward(row: dict[str, Any]) -> float:
    return float(row.get("deterministic_reward", row.get("reward", 0.0)) or 0.0)


def compact_text(value: Any, limit: int = 600) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def branch_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        out[(str(row.get("branch_group_id")), int(row.get("branch_id", -1)))] = row
    return out


def selected_rows(
    eval_rows: list[dict[str, Any]],
    branches_by_key: dict[tuple[str, int], dict[str, Any]],
    policy: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for eval_row in eval_rows:
        if eval_row.get("subset") != SUBSET or eval_row.get("policy") != policy:
            continue
        group_id = str(eval_row.get("branch_group_id"))
        oracle_ids = set(parse_ids(eval_row.get("oracle_branch_ids")))
        selected_ids = parse_ids(eval_row.get("selected_branch_ids"))
        if not selected_ids:
            continue
        for rank, branch_id in enumerate(selected_ids, start=1):
            branch = branches_by_key.get((group_id, branch_id))
            if not branch:
                continue
            out.append(
                {
                    "policy": policy,
                    "selection_rank": rank,
                    "branch_group_id": group_id,
                    "branch_id": branch_id,
                    "is_oracle_branch": branch_id in oracle_ids,
                    "selected_reward": branch_reward(branch),
                    "oracle_branch_ids": sorted(oracle_ids),
                    "task_id": branch.get("task_id"),
                    "split": branch.get("split"),
                    "domain": branch.get("domain"),
                    "branch_point": branch.get("branch_point"),
                    "alpha": branch.get("alpha"),
                    "alpha_bucket": branch.get("alpha_bucket"),
                    "delta_family": branch.get("primary_delta_family") or branch.get("delta_family"),
                    "recipe_id": branch.get("recipe_id"),
                    "recipe_source": branch.get("recipe_source"),
                    "task_screening_class": branch.get("task_screening_class"),
                    "parse_success": branch.get("parse_success"),
                    "parsed_answer": branch.get("parsed_answer"),
                    "deterministic_correct": branch.get("deterministic_correct"),
                    "deterministic_reward": branch.get("deterministic_reward"),
                    "old_frozen_tap_score": branch.get("old_frozen_tap_score"),
                    "v1_tap_score": branch.get("v1_tap_score"),
                    "v2_tap_score": branch.get("v2_tap_score"),
                    "v3_tap_score": branch.get("v3_tap_score"),
                    "salvage_tap_score": branch.get("salvage_tap_score"),
                    "output_text": compact_text(branch.get("output_text")),
                }
            )
    return out


def summarize(selection_rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = defaultdict(list)
    for row in selection_rows:
        groups[str(row["branch_group_id"])].append(row)
    selected_group_count = len(groups)
    oracle_hit_groups = sum(any(bool(row.get("is_oracle_branch")) for row in rows) for rows in groups.values())
    reward_mean = (
        sum(float(rows[0].get("selected_reward", 0.0)) for rows in groups.values()) / selected_group_count
        if selected_group_count
        else 0.0
    )
    return {
        "selected_group_count": selected_group_count,
        "selected_branch_rows": len(selection_rows),
        "oracle_hit_groups": oracle_hit_groups,
        "oracle_hit_rate": oracle_hit_groups / max(selected_group_count, 1),
        "top_rank_reward_mean": reward_mean,
    }


def main() -> int:
    if not EVAL_JSON.exists() or not EVAL_ROWS_CSV.exists() or not BRANCHES_JSON.exists():
        payload = {
            "BG_HIDDEN_ORIGIN_SELECTOR_RUN_ANYWAY_V4_VERDICT": "INSUFFICIENT",
            "blocker": "missing heldout eval or branch artifacts",
        }
        write_json(OUT_JSON, payload)
        write_md(OUT_MD, ["# Hidden-Origin Selector Run Anyway V4", "", "BG_HIDDEN_ORIGIN_SELECTOR_RUN_ANYWAY_V4_VERDICT = INSUFFICIENT"])
        print("BG_HIDDEN_ORIGIN_SELECTOR_RUN_ANYWAY_V4_VERDICT = INSUFFICIENT", flush=True)
        return 1

    eval_payload = load_json(EVAL_JSON)
    eval_rows = read_csv(EVAL_ROWS_CSV)
    branch_payload = load_json(BRANCHES_JSON)
    branch_rows = list(branch_payload.get("rows") or [])
    branches_by_key = branch_lookup(branch_rows)

    primary_selected = selected_rows(eval_rows, branches_by_key, PRIMARY_POLICY)
    top2_selected = selected_rows(eval_rows, branches_by_key, TOP2_POLICY)
    primary_summary = summarize(primary_selected)
    top2_summary = summarize(top2_selected)

    heldout_support = {
        "heldout_pair_count": eval_payload.get("heldout_pair_count"),
        "behaviorally_diverse_heldout_groups": eval_payload.get("behaviorally_diverse_heldout_groups"),
        "readiness_support_met": bool(eval_payload.get("readiness_support_met")),
        "selector_eval_verdict": eval_payload.get("BG_HIDDEN_ORIGIN_SELECTOR_EVAL_V4_VERDICT"),
    }
    verdict = "DIAGNOSTIC_SELECTOR_RUN"
    if not primary_selected:
        verdict = "NO_COMPATIBLE_SELECTIONS"

    write_csv(OUT_CSV, primary_selected)
    write_csv(OUT_TOP2_CSV, top2_selected)
    payload = {
        "BG_HIDDEN_ORIGIN_SELECTOR_RUN_ANYWAY_V4_VERDICT": verdict,
        "policy": PRIMARY_POLICY,
        "top2_policy": TOP2_POLICY,
        "subset": SUBSET,
        "diagnostic_only": True,
        "production_routing_changed": False,
        "readiness_claim": False,
        "heldout_support": heldout_support,
        "primary_summary": primary_summary,
        "top2_summary": top2_summary,
        "inputs": {
            "heldout_eval": rel(EVAL_JSON),
            "heldout_eval_rows": rel(EVAL_ROWS_CSV),
            "branches": rel(BRANCHES_JSON),
        },
        "outputs": {
            "primary_csv": rel(OUT_CSV),
            "top2_csv": rel(OUT_TOP2_CSV),
            "report": rel(OUT_MD),
        },
    }
    write_json(OUT_JSON, payload)

    metric_rows = [
        {"policy": PRIMARY_POLICY, **primary_summary},
        {"policy": TOP2_POLICY, **top2_summary},
    ]
    lines = [
        "# Hidden-Origin Selector Run Anyway V4",
        "",
        f"BG_HIDDEN_ORIGIN_SELECTOR_RUN_ANYWAY_V4_VERDICT = {verdict}",
        "",
        "This is a diagnostic run over the v4 quota-heldout primary-safe branch groups. It does not certify selector readiness and does not change production routing.",
        "",
        "## Policy",
        "",
        f"- primary policy: `{PRIMARY_POLICY}`",
        f"- top2 companion policy: `{TOP2_POLICY}`",
        f"- subset: `{SUBSET}`",
        "- diagnostic only: `true`",
        "- production routing changed: `false`",
        "- readiness claim: `false`",
        "",
        "## Heldout Support",
        "",
        f"- selector eval verdict: `{heldout_support['selector_eval_verdict']}`",
        f"- readiness support met: `{heldout_support['readiness_support_met']}`",
        f"- behaviorally diverse heldout groups: `{heldout_support['behaviorally_diverse_heldout_groups']}`",
        f"- heldout non-tie pairs: `{heldout_support['heldout_pair_count']}`",
        "",
        "## Selection Summary",
        "",
    ]
    lines.extend(
        md_table(
            [
                {
                    **row,
                    "oracle_hit_rate": rate(row["oracle_hit_rate"]),
                    "top_rank_reward_mean": rate(row["top_rank_reward_mean"]),
                }
                for row in metric_rows
            ],
            ["policy", "selected_group_count", "selected_branch_rows", "oracle_hit_groups", "oracle_hit_rate", "top_rank_reward_mean"],
        )
    )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- selected top1 branches: `{rel(OUT_CSV)}`",
            f"- selected top2 branches: `{rel(OUT_TOP2_CSV)}`",
            f"- JSON summary: `{rel(OUT_JSON)}`",
        ]
    )
    write_md(OUT_MD, lines)
    print(f"BG_HIDDEN_ORIGIN_SELECTOR_RUN_ANYWAY_V4_VERDICT = {verdict}", flush=True)
    print(f"Wrote {OUT_CSV}", flush=True)
    return 0 if primary_selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
