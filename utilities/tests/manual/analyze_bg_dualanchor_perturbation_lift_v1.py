"""Normalize DualAnchor perturbed-winner claims against candidate base rate.

This is a cached analysis over the all-loop DualAnchor branch groups and the
latest guarded policy output. It does not run generation, action steering,
true fork/carry, model training, registry updates, or production routing.
"""
from __future__ import annotations

import ast
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

from bg_hidden_origin_tap_common import PROBE_ROOT, md_table, rel
from bg_merged_tap_v1_common import safe_float
from run_bg_layer_native_two_tap_readiness_v1 import write_csv, write_json, write_md

import run_bg_dualanchor_all_loop_audit_v1 as base


OUT_ROOT = PROBE_ROOT / "bg_dualanchor_perturbation_lift_v1_2026-05-31"
REPORT_JSON = OUT_ROOT / "perturbation_lift.json"
REPORT_MD = OUT_ROOT / "perturbation_lift.md"
ROWS_CSV = OUT_ROOT / "perturbation_lift_rows.csv"
SUMMARY_CSV = OUT_ROOT / "perturbation_lift_summary.csv"

GUARDED_ROOT = PROBE_ROOT / "bg_dualanchor_all_loop_guarded_policy_v1_2026-05-31"
GUARDED_JSON = GUARDED_ROOT / "guarded_policy.json"
GUARDED_ROWS = GUARDED_ROOT / "guarded_group_rows.csv"


def ensure_root() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def finite_mean(vals: Iterable[Any]) -> float:
    xs: list[float] = []
    for value in vals:
        try:
            x = float(value)
        except Exception:
            continue
        if math.isfinite(x):
            xs.append(x)
    return float(mean(xs)) if xs else float("nan")


def parse_indices(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(v) for v in value]
    if value in (None, ""):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except Exception:
        return []
    if isinstance(parsed, list):
        return [int(v) for v in parsed]
    return []


def is_clean(row: dict[str, Any]) -> bool:
    return (
        str(row.get("delta_family")) == "clean"
        or str(row.get("direction_name")) == "clean_zero"
        or str(row.get("branch_id")) == "0"
    )


def perturb_count(row: dict[str, Any]) -> int:
    if is_clean(row):
        return 0
    for key in ("perturb_count", "generation_depth"):
        try:
            value = int(row.get(key))
        except Exception:
            continue
        if value >= 0:
            return value
    return 1


def birth_layer(row: dict[str, Any]) -> str:
    value = row.get("birth_layer") or row.get("branch_point") or ""
    return str(value) if value not in (None, "") else "unknown"


def load_selected_guard_rows() -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(GUARDED_JSON.read_text())
    best = payload.get("best_summary") or {}
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    with GUARDED_ROWS.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("policy") != best.get("policy"):
                continue
            if row.get("threshold_policy") != best.get("threshold_policy"):
                continue
            if row.get("guard_policy") != best.get("guard_policy"):
                continue
            selected[(str(row.get("dataset_name")), str(row.get("group_id")))] = row
    return selected


def reward(row: dict[str, Any]) -> float:
    return base.row_reward(row)


def group_row(dataset_name: str, group: Sequence[dict[str, Any]], guarded: dict[str, Any] | None) -> dict[str, Any]:
    rewards = [reward(row) for row in group]
    oracle = max(rewards)
    oracle_indices = {idx for idx, value in enumerate(rewards) if value == oracle}
    clean_indices = [idx for idx, row in enumerate(group) if is_clean(row)]
    pert_indices = [idx for idx, row in enumerate(group) if not is_clean(row)]
    clean_rewards = [rewards[idx] for idx in clean_indices]
    pert_rewards = [rewards[idx] for idx in pert_indices]
    max_clean = max(clean_rewards) if clean_rewards else float("nan")
    max_pert = max(pert_rewards) if pert_rewards else float("nan")
    mean_clean = finite_mean(clean_rewards)
    mean_pert = finite_mean(pert_rewards)
    random_pert_better = finite_mean(1.0 if r > max_clean else 0.0 for r in pert_rewards)
    random_pert_tied = finite_mean(1.0 if r == max_clean else 0.0 for r in pert_rewards)
    forced_idx = int(float(guarded.get("forced_terminal_index", -1))) if guarded else -1
    pre_indices = parse_indices(guarded.get("pre_terminal_indices")) if guarded else []
    terminal_indices = parse_indices(guarded.get("terminal_indices")) if guarded else []
    forced_valid = 0 <= forced_idx < len(group)
    forced_perturbed = bool(forced_valid and forced_idx in pert_indices)
    terminal_count = len(terminal_indices)
    terminal_pert_fraction = (
        sum(1 for idx in terminal_indices if idx in pert_indices) / terminal_count if terminal_count else float("nan")
    )
    pre_count = len(pre_indices)
    pre_pert_fraction = sum(1 for idx in pre_indices if idx in pert_indices) / pre_count if pre_count else float("nan")
    oracle_clean = bool(oracle_indices & set(clean_indices))
    oracle_pert = bool(oracle_indices & set(pert_indices))
    perturb_counts = Counter(perturb_count(row) for row in group)
    oracle_perturb_counts = Counter(perturb_count(group[idx]) for idx in oracle_indices)
    forced_perturb_count = perturb_count(group[forced_idx]) if forced_valid else None
    forced_birth_layer = birth_layer(group[forced_idx]) if forced_valid and forced_perturbed else ""
    pert_fraction = len(pert_indices) / len(group) if group else float("nan")
    row = {
        "dataset_name": dataset_name,
        "group_id": str(group[0].get("group_id")),
        "domain": str(group[0].get("domain")),
        "group_size": len(group),
        "clean_count": len(clean_indices),
        "perturbed_count": len(pert_indices),
        "perturbed_candidate_fraction": pert_fraction,
        "random_expected_perturbed_winner_rate": pert_fraction,
        "oracle_reward": oracle,
        "max_clean_reward": max_clean,
        "max_perturbed_reward": max_pert,
        "mean_clean_reward": mean_clean,
        "mean_perturbed_reward": mean_pert,
        "max_perturbed_gt_clean": 1.0 if max_pert > max_clean else 0.0,
        "max_perturbed_eq_clean": 1.0 if max_pert == max_clean else 0.0,
        "max_perturbed_lt_clean": 1.0 if max_pert < max_clean else 0.0,
        "random_perturbed_candidate_gt_clean": random_pert_better,
        "random_perturbed_candidate_eq_clean": random_pert_tied,
        "oracle_includes_clean": 1.0 if oracle_clean else 0.0,
        "oracle_includes_perturbed": 1.0 if oracle_pert else 0.0,
        "oracle_only_clean": 1.0 if oracle_clean and not oracle_pert else 0.0,
        "oracle_only_perturbed": 1.0 if oracle_pert and not oracle_clean else 0.0,
        "oracle_clean_perturbed_tie": 1.0 if oracle_clean and oracle_pert else 0.0,
        "forced_terminal_index": forced_idx,
        "forced_terminal_perturbed": 1.0 if forced_perturbed else 0.0,
        "forced_terminal_perturbed_lift_over_count_share": (1.0 if forced_perturbed else 0.0) - pert_fraction,
        "forced_terminal_reward": rewards[forced_idx] if forced_valid else float("nan"),
        "forced_terminal_oracle": 1.0 if forced_idx in oracle_indices else 0.0,
        "forced_terminal_perturb_count": forced_perturb_count,
        "forced_terminal_birth_layer": forced_birth_layer,
        "terminal_confident": safe_float(guarded.get("terminal_confident"), float("nan")) if guarded else float("nan"),
        "terminal_deferred": safe_float(guarded.get("terminal_deferred"), float("nan")) if guarded else float("nan"),
        "terminal_survivor_count": terminal_count,
        "terminal_perturbed_fraction": terminal_pert_fraction,
        "terminal_oracle_retained": safe_float(guarded.get("terminal_oracle_retained"), float("nan")) if guarded else float("nan"),
        "pre_terminal_survivor_count": pre_count,
        "pre_terminal_perturbed_fraction": pre_pert_fraction,
        "pre_terminal_oracle_retained": safe_float(guarded.get("pre_terminal_oracle_retained"), float("nan")) if guarded else float("nan"),
        "candidate_count_by_perturb_count": dict(perturb_counts),
        "oracle_count_by_perturb_count": dict(oracle_perturb_counts),
        "lineage_depth_available": any(row.get("parent_branch_id") not in (None, "") or row.get("generation_depth") not in (None, "") for row in group),
    }
    return row


def summarize(rows: Sequence[dict[str, Any]], label: str) -> dict[str, Any]:
    keys = [
        "group_size",
        "perturbed_candidate_fraction",
        "oracle_includes_perturbed",
        "oracle_only_perturbed",
        "oracle_clean_perturbed_tie",
        "oracle_only_clean",
        "max_perturbed_gt_clean",
        "max_perturbed_eq_clean",
        "max_perturbed_lt_clean",
        "random_perturbed_candidate_gt_clean",
        "random_perturbed_candidate_eq_clean",
        "mean_clean_reward",
        "mean_perturbed_reward",
        "max_clean_reward",
        "max_perturbed_reward",
        "forced_terminal_perturbed",
        "forced_terminal_perturbed_lift_over_count_share",
        "forced_terminal_reward",
        "forced_terminal_oracle",
        "terminal_confident",
        "terminal_deferred",
        "terminal_perturbed_fraction",
        "terminal_oracle_retained",
        "pre_terminal_perturbed_fraction",
        "pre_terminal_oracle_retained",
    ]
    out: dict[str, Any] = {"split": label, "group_count": len(rows)}
    for key in keys:
        out[key] = finite_mean(row.get(key) for row in rows)
    return out


def main() -> int:
    ensure_root()
    guarded_rows = load_selected_guard_rows()
    datasets, coverage = base.load_full_loop_branch_groups()
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for group in dataset["groups"]:
            key = (dataset["dataset_name"], str(group[0].get("group_id")))
            rows.append(group_row(dataset["dataset_name"], group, guarded_rows.get(key)))

    summaries = [summarize(rows, "ALL_FULL_LOOP")]
    for dataset_name in sorted({row["dataset_name"] for row in rows}):
        summaries.append(summarize([row for row in rows if row["dataset_name"] == dataset_name], dataset_name))
    for domain in sorted({row["domain"] for row in rows}):
        summaries.append(summarize([row for row in rows if row["domain"] == domain], f"domain:{domain}"))
    for state, label in ((1.0, "terminal_confident"), (1.0, "terminal_deferred")):
        key = "terminal_confident" if label == "terminal_confident" else "terminal_deferred"
        summaries.append(summarize([row for row in rows if safe_float(row.get(key), 0.0) == state], label))

    perturb_count_rows = []
    for count in sorted({int(k) for row in rows for k in (row.get("candidate_count_by_perturb_count") or {}).keys()}):
        candidate_total = sum((row.get("candidate_count_by_perturb_count") or {}).get(count, 0) for row in rows)
        oracle_total = sum((row.get("oracle_count_by_perturb_count") or {}).get(count, 0) for row in rows)
        forced_total = sum(1 for row in rows if row.get("forced_terminal_perturb_count") == count)
        perturb_count_rows.append(
            {
                "perturb_count": count,
                "candidate_count": candidate_total,
                "candidate_fraction": candidate_total / sum(row["group_size"] for row in rows),
                "oracle_tied_best_count": oracle_total,
                "forced_terminal_top1_count": forced_total,
                "forced_terminal_top1_rate": forced_total / len(rows) if rows else float("nan"),
            }
        )

    birth_rows = []
    perturbed_forced = [row for row in rows if row.get("forced_terminal_perturbed")]
    for layer in sorted({row.get("forced_terminal_birth_layer") for row in perturbed_forced if row.get("forced_terminal_birth_layer")}):
        count = sum(1 for row in perturbed_forced if row.get("forced_terminal_birth_layer") == layer)
        birth_rows.append({"birth_layer": layer, "forced_terminal_perturbed_count": count, "rate_over_all_groups": count / len(rows)})

    verdict = "PERTURBATION_USEFUL_BUT_COUNT_NORMALIZED"
    if summaries[0]["forced_terminal_perturbed_lift_over_count_share"] > 0.0:
        verdict = "PERTURBED_SELECTION_LIFT_POSITIVE"
    payload = {
        "BG_DUALANCHOR_PERTURBATION_LIFT_VERDICT": verdict,
        "status": verdict,
        "coverage": coverage,
        "lineage_note": "Current cached BGV1/v4 rows are one-generation hook perturbations. Parent/child recursive lineage is not available in these artifacts.",
        "lineage_instrumentation_required": [
            "branch_id",
            "parent_branch_id",
            "root_branch_id",
            "generation_depth",
            "perturb_count",
            "birth_layer",
            "birth_loop",
            "birth_stage",
            "perturbation_family",
            "perturbation_alpha",
            "perturbation_rms",
            "perturbation_seed",
            "mutation_index",
            "lineage_path",
        ],
        "summaries": summaries,
        "perturb_count_rows": perturb_count_rows,
        "birth_rows": birth_rows,
    }
    write_json(REPORT_JSON, payload)
    write_csv(ROWS_CSV, rows)
    write_csv(SUMMARY_CSV, summaries + perturb_count_rows + birth_rows)

    overall = summaries[0]
    lines = [
        "# DualAnchor Perturbation Lift v1",
        "",
        f"BG_DUALANCHOR_PERTURBATION_LIFT_VERDICT = {verdict}",
        "",
        "This analysis normalizes perturbed-selection claims against the perturbed candidate base rate.",
        "",
        "## Headline",
        "",
        f"- full-loop branch groups: `{overall['group_count']}`",
        f"- perturbed candidate fraction / random expected perturbed winner rate: `{overall['perturbed_candidate_fraction']}`",
        f"- observed forced terminal top1 perturbed rate: `{overall['forced_terminal_perturbed']}`",
        f"- lift over count-share random: `{overall['forced_terminal_perturbed_lift_over_count_share']}`",
        f"- max perturbed reward > clean reward: `{overall['max_perturbed_gt_clean']}`",
        f"- max perturbed reward = clean reward: `{overall['max_perturbed_eq_clean']}`",
        f"- max perturbed reward < clean reward: `{overall['max_perturbed_lt_clean']}`",
        f"- oracle includes perturbed: `{overall['oracle_includes_perturbed']}`",
        f"- oracle only perturbed, clean not tied: `{overall['oracle_only_perturbed']}`",
        f"- oracle clean + perturbed tied: `{overall['oracle_clean_perturbed_tie']}`",
        f"- random perturbed candidate > clean: `{overall['random_perturbed_candidate_gt_clean']}`",
        f"- random perturbed candidate = clean: `{overall['random_perturbed_candidate_eq_clean']}`",
        "",
        "## Summary Rows",
        "",
    ]
    lines.extend(
        md_table(
            summaries,
            [
                "split",
                "group_count",
                "perturbed_candidate_fraction",
                "forced_terminal_perturbed",
                "forced_terminal_perturbed_lift_over_count_share",
                "max_perturbed_gt_clean",
                "max_perturbed_eq_clean",
                "oracle_only_perturbed",
                "oracle_clean_perturbed_tie",
                "random_perturbed_candidate_gt_clean",
                "mean_clean_reward",
                "mean_perturbed_reward",
                "terminal_oracle_retained",
            ],
        )
    )
    lines.extend(["", "## Perturb Count", ""])
    lines.extend(md_table(perturb_count_rows, ["perturb_count", "candidate_count", "candidate_fraction", "oracle_tied_best_count", "forced_terminal_top1_count", "forced_terminal_top1_rate"]))
    lines.extend(["", "## Forced Terminal Perturbed Birth Layer", ""])
    lines.extend(md_table(birth_rows, ["birth_layer", "forced_terminal_perturbed_count", "rate_over_all_groups"]))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The earlier `~80% perturbed output` statement is not positive lift by count share: perturbed candidates are `7/8 = 87.5%` of each group, while forced terminal top1 is perturbed about `79.9%` of the time.",
            "- Perturbation is still useful: the perturbed pool strictly beats clean in a large minority of groups and ties clean in the rest; clean is never the only oracle in these 139 groups.",
            "- Current artifacts cannot answer descendant lineage questions because parent/child branch IDs are not recorded. They only support clean vs one-generation perturbation and birth-layer analysis.",
            "",
            "## Required Lineage Patch For Future Generation",
            "",
            "Write these fields on every generated branch: `branch_id`, `parent_branch_id`, `root_branch_id`, `generation_depth`, `perturb_count`, `birth_layer`, `birth_loop`, `birth_stage`, `perturbation_family`, `perturbation_alpha`, `perturbation_rms`, `perturbation_seed`, `mutation_index`, `lineage_path`.",
            "",
            "## Files",
            "",
            f"- report json: `{rel(REPORT_JSON)}`",
            f"- rows: `{rel(ROWS_CSV)}`",
            f"- summary csv: `{rel(SUMMARY_CSV)}`",
        ]
    )
    write_md(REPORT_MD, lines)
    print(f"BG_DUALANCHOR_PERTURBATION_LIFT_VERDICT = {verdict}", flush=True)
    print(f"perturbed_candidate_fraction = {overall['perturbed_candidate_fraction']}", flush=True)
    print(f"observed_forced_terminal_perturbed = {overall['forced_terminal_perturbed']}", flush=True)
    print(f"lift_over_count_share = {overall['forced_terminal_perturbed_lift_over_count_share']}", flush=True)
    print(f"max_perturbed_gt_clean = {overall['max_perturbed_gt_clean']}", flush=True)
    print(f"report = {rel(REPORT_MD)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
